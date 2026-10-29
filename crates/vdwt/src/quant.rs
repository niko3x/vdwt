//! Dead-zone uniform scalar quantization (DESIGN.md §5.1).
//!
//! Subband `b` uses the step `Δ_b = Δ(qp) · w_b`, where `Δ(qp) = 2^(qp/8)`. The weights `w_b`
//! travel in the sequence header (Q12). Steps are fixed point (Q12) and never below 1, and
//! reconstruction is integer-only, so every decoder reproduces the encoder's reconstruction
//! exactly. The encoder's own choices are not normative: the rounding offset θ and the weights
//! themselves.

use crate::dwt::Wavelet;

/// `2^(i/8)` in Q16.
const POW2_EIGHTHS_Q16: [u64; 8] = [65536, 71468, 77936, 84990, 92682, 101070, 110218, 120194];
const ONE_Q12: u64 = 1 << 12;
/// Steps are capped at 32768: coefficients never exceed that, so larger steps change nothing,
/// and the cap keeps reconstruction in 32-bit arithmetic.
const MAX_STEP_Q12: u64 = 1 << 27;

/// `Δ_b` in Q12 for a subband with weight `weight_q12` at quality `qp`, for coefficients that
/// carry `frac_bits` fraction bits (normative).
pub fn step_q12(qp: u8, weight_q12: u16, frac_bits: u32) -> u64 {
    let base_q12 = (POW2_EIGHTHS_Q16[usize::from(qp & 7)] << (qp >> 3)) >> 4;
    ((base_q12.saturating_mul(u64::from(weight_q12)) >> 12) << frac_bits)
        .clamp(ONE_Q12, MAX_STEP_Q12)
}

/// Encoder-side quantizer: `q = sign(c) · ⌊|c|/Δ + θ⌋`.
///
/// Δ ≥ 1, so `|c| · 2^16/Δ + θ` stays below 2^32 and all arithmetic is 32-bit and branch-free,
/// which the compiler vectorizes.
#[derive(Clone, Copy, Debug)]
pub struct Quantizer {
    /// `2^16 / Δ`, rounded; at most 2^16.
    inv_q16: u32,
    /// θ in Q16.
    bias_q16: u32,
}

impl Quantizer {
    /// `rounding` is θ: 0.5 rounds to nearest, smaller values widen the dead zone around zero.
    pub fn new(step_q12: u64, rounding: f64) -> Self {
        assert!(step_q12 >= ONE_Q12, "steps are at least 1");
        Self {
            inv_q16: (((1u64 << 28) + step_q12 / 2) / step_q12) as u32,
            bias_q16: (rounding.clamp(0.0, 0.999) * 65536.0) as u32,
        }
    }

    #[inline(always)]
    pub fn quantize(&self, c: i16) -> i32 {
        let m = ((u32::from(c.unsigned_abs()) * self.inv_q16 + self.bias_q16) >> 16)
            .min(i16::MAX as u32) as i32;
        if c < 0 { -m } else { m }
    }

    pub fn quantize_row(&self, src: &[i16], dst: &mut [i32]) {
        for (d, &c) in dst.iter_mut().zip(src) {
            *d = self.quantize(c);
        }
    }

    /// [`Self::quantize_row`], also returning the nonzero count and the summed bit lengths of the
    /// magnitudes: the rate statistics, in the same pass.
    pub fn quantize_row_stats(&self, src: &[i16], dst: &mut [i32]) -> (u32, u64) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::avx2_detected() {
            // SAFETY: AVX2 is present.
            return unsafe { self.quantize_row_stats_avx2(src, dst) };
        }
        self.quantize_row_stats_any(src, dst)
    }

    /// [`Self::quantize_row_stats`] compiled for AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn quantize_row_stats_avx2(&self, src: &[i16], dst: &mut [i32]) -> (u32, u64) {
        use std::arch::x86_64::*;

        let n = src.len().min(dst.len());
        let inv = _mm256_set1_epi32(self.inv_q16 as i32);
        let bias = _mm256_set1_epi32(self.bias_q16 as i32);
        let cap = _mm256_set1_epi32(i16::MAX as i32);
        let zero = _mm256_setzero_si256();
        let mut nonzero = 0u32;
        // The bit lengths accumulate in a vector and are summed once at the end.
        let mut acc = zero;
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: eight coefficients are inside `src`, and eight values inside `dst`.
            unsafe {
                let c = _mm256_cvtepi16_epi32(_mm_loadu_si128(src.as_ptr().add(i).cast()));
                let m = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(_mm256_abs_epi32(c), inv), bias),
                    16,
                );
                let m = _mm256_min_epi32(m, cap);
                let sign = _mm256_srai_epi32(c, 31);
                let q = _mm256_sub_epi32(_mm256_xor_si256(m, sign), sign);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), q);
                // The statistics: the nonzeros, and the bit lengths of the magnitudes.
                let mag = _mm256_abs_epi32(q);
                let is_zero = _mm256_cmpeq_epi32(mag, zero);
                nonzero +=
                    8 - (_mm256_movemask_ps(_mm256_castsi256_ps(is_zero)).count_ones() as u32);
                // A magnitude's bit length is the exponent of its float form, which is exact
                // here (the magnitudes are below 2^15).
                let exponent = _mm256_sub_epi32(
                    _mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(mag)), 23),
                    _mm256_set1_epi32(126),
                );
                acc = _mm256_add_epi32(acc, _mm256_max_epi32(exponent, zero));
            }
            i += 8;
        }
        // The tail and the totals.
        let (tail_nonzero, tail_bits) = self.quantize_row_stats_any(&src[i..n], &mut dst[i..n]);
        let mut bits = tail_bits;
        if i > 0 {
            let lo = _mm256_castsi256_si128(acc);
            let hi = _mm256_extracti128_si256(acc, 1);
            let sum = _mm_add_epi32(lo, hi);
            let sum = _mm_hadd_epi32(sum, sum);
            let sum = _mm_hadd_epi32(sum, sum);
            bits += _mm_cvtsi128_si32(sum) as u64;
        }
        (nonzero + tail_nonzero, bits)
    }

    /// [`Self::quantize_row_stats`] compiled for NEON. The A55's auto-vectorized scalar loop
    /// measured faster (1.63 vs 2.26 ms per 921k coefficients), so dispatch does not use it.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(dead_code)]
    unsafe fn quantize_row_stats_neon(&self, src: &[i16], dst: &mut [i32]) -> (u32, u64) {
        use std::arch::aarch64::*;

        let n = src.len().min(dst.len());
        let mut nonzero = 0u32;
        let mut bits = 0u64;
        let mut i = 0;
        // SAFETY: NEON is present; the loads stay inside the slices.
        unsafe {
            let inv = vdupq_n_s32(self.inv_q16 as i32);
            let bias = vdupq_n_s32(self.bias_q16 as i32);
            let cap = vdupq_n_s32(i16::MAX as i32);
            let zero = vdupq_n_s32(0);
            while i + 8 <= n {
                let c0 = vmovl_s16(vld1_s16(src.as_ptr().add(i)));
                let c1 = vmovl_s16(vld1_s16(src.as_ptr().add(i + 4)));
                for (k, c) in [c0, c1].into_iter().enumerate() {
                    let m = vshrq_n_s32(vaddq_s32(vmulq_s32(vabsq_s32(c), inv), bias), 16);
                    let m = vminq_s32(m, cap);
                    let sign = vshrq_n_s32(c, 31);
                    let q = vsubq_s32(veorq_s32(m, sign), sign);
                    vst1q_s32(dst.as_mut_ptr().add(i + 4 * k).cast(), q);
                    let mag = vabsq_s32(q);
                    nonzero += 4 - vaddvq_u32(vshrq_n_u32(vceqq_s32(mag, zero), 31));
                    // The bit length is the exponent of the float form, exact below 2^15.
                    let f = vcvtq_f32_s32(mag);
                    let e = vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(f), 23));
                    let e = vsubq_s32(e, vdupq_n_s32(126));
                    bits += u64::from(vaddvq_u32(vreinterpretq_u32_s32(vmaxq_s32(e, zero))));
                }
                i += 8;
            }
        }
        let (tail_nonzero, tail_bits) = self.quantize_row_stats_any(&src[i..n], &mut dst[i..n]);
        (nonzero + tail_nonzero, bits + tail_bits)
    }

    #[inline(always)]
    fn quantize_row_stats_any(&self, src: &[i16], dst: &mut [i32]) -> (u32, u64) {
        let (mut nonzero, mut bits) = (0u32, 0u32);
        for (d, &c) in dst.iter_mut().zip(src) {
            let q = self.quantize(c);
            *d = q;
            let m = q.unsigned_abs();
            nonzero += u32::from(m != 0);
            bits += 32 - m.leading_zeros();
        }
        (nonzero, u64::from(bits))
    }
}

/// Normative reconstruction: magnitudes come back as `(|q| + δ) · Δ`, rounded and saturated to
/// the `i16` range.
///
/// Magnitudes are first capped at `m_max`, the smallest that already saturates. With Δ ≤ 2^15
/// everything then fits in 32 bits and stays branch-free; the result equals the exact formula.
#[derive(Clone, Copy, Debug)]
pub struct Dequantizer {
    step_q12: u32,
    /// δ·Δ in Q12, plus ½ for rounding.
    offset_q12: u32,
    m_max: u32,
}

impl Dequantizer {
    /// `recon_offset` is δ in 1/256 units.
    pub fn new(step_q12: u64, recon_offset: u8) -> Self {
        assert!(
            (ONE_Q12..=MAX_STEP_Q12).contains(&step_q12),
            "step out of range"
        );
        let offset_q12 = ((step_q12 * u64::from(recon_offset)) >> 8) + (1 << 11);
        let saturating = (i16::MAX as u64 + 1) << 12;
        let m_max = saturating.div_ceil(step_q12).max(1);
        Self {
            step_q12: step_q12 as u32,
            offset_q12: offset_q12 as u32,
            m_max: m_max as u32,
        }
    }

    #[inline]
    pub fn dequantize(&self, q: i32) -> i16 {
        let m = q.unsigned_abs().min(self.m_max);
        let v = ((m * self.step_q12 + self.offset_q12) >> 12).min(i16::MAX as u32) as i16;
        let v = if m == 0 { 0 } else { v };
        if q < 0 { -v } else { v }
    }

    /// [`Self::dequantize`] of a row.
    pub fn dequantize_row(&self, q: &[i32], out: &mut [i16]) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::avx2_detected() {
            // SAFETY: the CPU has AVX2.
            return unsafe { self.dequantize_row_avx2(q, out) };
        }
        self.dequantize_row_any(q, out);
    }

    /// [`Self::dequantize`] of a row of quantized values, in place.
    pub fn dequantize_in_place(&self, row: &mut [i16]) {
        for o in row.iter_mut() {
            let q = i32::from(*o);
            let m = q.unsigned_abs().min(self.m_max);
            let v = ((m * self.step_q12 + self.offset_q12) >> 12).min(i16::MAX as u32) as i32;
            let v = if m == 0 { 0 } else { v };
            let sign = q >> 31;
            *o = ((v ^ sign) - sign) as i16;
        }
    }

    /// The row loop without branches, so that it vectorizes.
    #[inline(always)]
    fn dequantize_row_any(&self, q: &[i32], out: &mut [i16]) {
        for (o, &q) in out.iter_mut().zip(q) {
            let m = q.unsigned_abs().min(self.m_max);
            let v = ((m * self.step_q12 + self.offset_q12) >> 12).min(i16::MAX as u32) as i32;
            let v = if m == 0 { 0 } else { v };
            let sign = q >> 31;
            *o = ((v ^ sign) - sign) as i16;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dequantize_row_avx2(&self, q: &[i32], out: &mut [i16]) {
        use std::arch::x86_64::*;

        let n = q.len().min(out.len());
        let step = _mm256_set1_epi32(self.step_q12 as i32);
        let offset = _mm256_set1_epi32(self.offset_q12 as i32);
        let m_max = _mm256_set1_epi32(self.m_max as i32);
        let cap = _mm256_set1_epi32(i16::MAX as i32);
        let zero = _mm256_setzero_si256();
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: eight values are inside `q`, and eight outputs inside `out`.
            unsafe {
                let v = _mm256_loadu_si256(q.as_ptr().add(i).cast());
                let m = _mm256_min_epi32(_mm256_abs_epi32(v), m_max);
                let mag =
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(m, step), offset), 12);
                let mag = _mm256_min_epi32(mag, cap);
                // A zero magnitude stays zero, whatever the offset rounds to.
                let mag = _mm256_blendv_epi8(mag, zero, _mm256_cmpeq_epi32(m, zero));
                let sign = _mm256_srai_epi32(v, 31);
                let r = _mm256_sub_epi32(_mm256_xor_si256(mag, sign), sign);
                // Pack the eight values back to 16 bits.
                let lo = _mm256_castsi256_si128(r);
                let hi = _mm256_extracti128_si256(r, 1);
                let packed = _mm_packs_epi32(lo, hi);
                _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), packed);
            }
            i += 8;
        }
        self.dequantize_row_any(&q[i..n], &mut out[i..n]);
    }

    /// [`Self::dequantize_row`] compiled for NEON.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(dead_code)]
    unsafe fn dequantize_row_neon(&self, q: &[i32], out: &mut [i16]) {
        use std::arch::aarch64::*;

        let n = q.len().min(out.len());
        let mut i = 0;
        // SAFETY: NEON is present; the loads stay inside the slices.
        unsafe {
            let step = vdupq_n_s32(self.step_q12 as i32);
            let offset = vdupq_n_s32(self.offset_q12 as i32);
            let m_max = vdupq_n_s32(self.m_max as i32);
            let cap = vdupq_n_s32(i16::MAX as i32);
            let zero = vdupq_n_s32(0);
            while i + 8 <= n {
                let v0 = vld1q_s32(q.as_ptr().add(i).cast());
                let v1 = vld1q_s32(q.as_ptr().add(i + 4).cast());
                let mut packed = [vdup_n_s16(0), vdup_n_s16(0)];
                for (k, v) in [v0, v1].into_iter().enumerate() {
                    let m = vminq_s32(vabsq_s32(v), m_max);
                    let mag = vshrq_n_s32(vaddq_s32(vmulq_s32(m, step), offset), 12);
                    let mag = vminq_s32(mag, cap);
                    // A zero magnitude stays zero, whatever the offset rounds to.
                    let mag = vbslq_s32(vceqq_s32(m, zero), zero, mag);
                    let sign = vshrq_n_s32(v, 31);
                    let r = vsubq_s32(veorq_s32(mag, sign), sign);
                    packed[k] = vqmovn_s32(r);
                }
                vst1q_s16(
                    out.as_mut_ptr().add(i).cast(),
                    vcombine_s16(packed[0], packed[1]),
                );
                i += 8;
            }
        }
        self.dequantize_row_any(&q[i..n], &mut out[i..n]);
    }
}

/// Step weights that equalize every subband's contribution to pixel-domain squared error for
/// `wavelet`: `1/√G_b`, where `G_b` is the energy of the subband's synthesis basis. Q12, in
/// [`Geometry::subbands`](crate::dwt::Geometry::subbands) order. Encoder-side only: the decoder
/// reads weights from the sequence header.
pub fn mse_weights(levels: usize, wavelet: Wavelet) -> Vec<u16> {
    let gain = |level: usize, high: bool| wavelet.synthesis_gain_1d(level, high);
    let weight = |g: f64| (4096.0 / g.sqrt()).round().clamp(1.0, 65535.0) as u16;
    let mut w = vec![weight(gain(levels, false).powi(2))];
    for level in (1..=levels).rev() {
        let (lo, hi) = (gain(level, false), gain(level, true));
        w.extend([weight(hi * lo), weight(lo * hi), weight(hi * hi)]); // HL, LH, HH
    }
    w
}

/// Energy of the 1-D synthesis basis for a low- or high-pass coefficient `level` levels deep,
/// from a linear (unrounded) 5/3 synthesis of an impulse placed far from the borders.
pub(crate) fn synthesis_gain_53(level: usize, high: bool) -> f64 {
    let n = 64;
    let (mut low, mut hi) = (vec![0.0; n], vec![0.0; n]);
    if high {
        hi[n / 2] = 1.0
    } else {
        low[n / 2] = 1.0
    }
    let mut x = synthesize(&low, &hi);
    for _ in 1..level {
        x = synthesize(&x, &vec![0.0; x.len()]);
    }
    x.iter().map(|v| v * v).sum()
}

/// One linear 5/3 synthesis step of equal-length bands.
fn synthesize(low: &[f64], high: &[f64]) -> Vec<f64> {
    let n = low.len();
    let mut x = vec![0.0; 2 * n];
    for j in 0..n {
        x[2 * j] = low[j] - (high[j.saturating_sub(1)] + high[j]) / 4.0;
    }
    for j in 0..n {
        let right = if j + 1 < n { x[2 * j + 2] } else { x[2 * j] };
        x[2 * j + 1] = high[j] + (x[2 * j] + right) / 2.0;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantizer_matches_the_exact_formula() {
        let exact = |q: i32, step: u64, off: u8| -> i16 {
            if q == 0 {
                return 0;
            }
            let m = u128::from(q.unsigned_abs());
            let v = ((m * u128::from(step) + ((u128::from(step) * u128::from(off)) >> 8) + 2048)
                >> 12)
                .min(32767);
            if q < 0 { -(v as i16) } else { v as i16 }
        };
        for step in [4096, 4097, 5000, 77936, 1 << 20, MAX_STEP_Q12] {
            for off in [0, 51, 128, 255] {
                let d = Dequantizer::new(step, off);
                for q in (-70000..=70000)
                    .step_by(7)
                    .chain([i32::MIN, i32::MAX, -1, 0, 1])
                {
                    assert_eq!(
                        d.dequantize(q),
                        exact(q, step, off),
                        "q {q}, step {step}, δ {off}"
                    );
                }
            }
        }
    }

    #[test]
    fn steps_double_every_eight_qp() {
        assert_eq!(step_q12(0, 4096, 0), 4096);
        assert_eq!(step_q12(8, 4096, 0), 2 * 4096);
        assert_eq!(step_q12(8, 4096, 3), 16 * 4096);
        assert_eq!(step_q12(16, 2048, 0), 2 * 4096);
        assert_eq!(step_q12(0, 1, 0), 4096, "never below 1");
        assert_eq!(step_q12(255, 65535, 0), MAX_STEP_Q12);
    }

    #[test]
    fn unit_step_is_lossless() {
        let (q, d) = (Quantizer::new(4096, 0.3), Dequantizer::new(4096, 0));
        for c in i16::MIN + 1..=i16::MAX {
            assert_eq!(i32::from(d.dequantize(q.quantize(c))), i32::from(c));
        }
    }

    #[test]
    fn reconstruction_error_is_bounded() {
        let step = step_q12(37, 4096, 0); // Δ ≈ 24.7
        let delta = step as f64 / 4096.0;
        // Rounding to nearest pairs with δ = 0; a full dead zone (θ = 0) with δ = 1/2 (midpoint).
        for (rounding, offset) in [(0.5, 0), (0.0, 128)] {
            let (q, d) = (
                Quantizer::new(step, rounding),
                Dequantizer::new(step, offset),
            );
            for c in -3000i16..=3000 {
                let err = (f64::from(d.dequantize(q.quantize(c))) - f64::from(c)).abs();
                let bound = if q.quantize(c) == 0 {
                    delta
                } else {
                    delta / 2.0 + 1.0
                };
                assert!(err <= bound, "c={c}, θ={rounding}: error {err}");
            }
        }
    }

    #[test]
    fn mse_weights_match_known_5_3_gains() {
        // Level 1: G_L = 1.5 and G_H = 46/64, so HH1 gets 1/0.71875 and HL1 1/√(1.5·0.71875).
        let w = mse_weights(1, Wavelet::Reversible53);
        assert_eq!(w.len(), 4);
        let expect = |g: f64| (4096.0 / g.sqrt()).round() as u16;
        assert_eq!(w[1], expect(1.5 * 0.71875));
        assert_eq!(w[3], expect(0.71875 * 0.71875));
        assert_eq!(w[0], expect(1.5 * 1.5));
        // Deeper bands carry more energy per coefficient, so they get finer steps.
        let w5 = mse_weights(5, Wavelet::Reversible53);
        assert!(w5.windows(4).step_by(3).all(|g| g[0] <= g[3]), "{w5:?}");
        // The 9/7's weights are in range.
        let w97 = mse_weights(5, Wavelet::Lifted97);
        assert!(w97.iter().all(|&w| (512..65535).contains(&w)), "{w97:?}");
    }
}

#[cfg(test)]
mod bench_tests {
    use super::*;
    use std::time::Instant;

    /// Temporary micro-benchmark: the scalar path against the vector one.
    #[test]
    #[ignore]
    fn bench_quantize() {
        let n = 1280 * 720;
        let src: Vec<i16> = (0..n)
            .map(|i| ((i * 2654435761usize) % 65536) as i16)
            .collect();
        let mut dst = vec![0i32; n];
        let q = Quantizer::new(4096, 0.25);
        let mut best_scalar = f64::MAX;
        let mut best_simd = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..4 {
                let (a, b) = q.quantize_row_stats_any(&src, &mut dst);
                acc += u64::from(a) + b;
            }
            let e = t.elapsed().as_secs_f64() * 1000.0 / 4.0;
            std::hint::black_box(acc);
            best_scalar = best_scalar.min(e);
            let t = Instant::now();
            let mut acc = 0u64;
            for _ in 0..4 {
                let (a, b) = q.quantize_row_stats(&src, &mut dst);
                acc += u64::from(a) + b;
            }
            let e = t.elapsed().as_secs_f64() * 1000.0 / 4.0;
            std::hint::black_box(acc);
            best_simd = best_simd.min(e);
        }
        println!(
            "quantize 921k coeffs: scalar {:.3} ms   vector {:.3} ms   speedup {:.2}x",
            best_scalar,
            best_simd,
            best_scalar / best_simd
        );

        let qv: Vec<i32> = (0..n)
            .map(|i| ((i * 40503usize) % 65536) as i32 - 32768)
            .collect();
        let mut out = vec![0i16; n];
        let d = Dequantizer::new(4096, 64);
        let mut bs = f64::MAX;
        let mut bv = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..4 {
                d.dequantize_row_any(&qv, &mut out);
            }
            bs = bs.min(t.elapsed().as_secs_f64() * 1000.0 / 4.0);
            let t = Instant::now();
            for _ in 0..4 {
                d.dequantize_row(&qv, &mut out);
            }
            bv = bv.min(t.elapsed().as_secs_f64() * 1000.0 / 4.0);
        }
        println!(
            "dequantize 921k coeffs: scalar {:.3} ms   vector {:.3} ms   speedup {:.2}x",
            bs,
            bv,
            bs / bv
        );
    }
}

#[cfg(test)]
mod simd_tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// The vector kernels must give exactly what the scalar path gives, value for value and
    /// statistic for statistic: the stream may not change by a single bit.
    #[test]
    fn the_vector_kernels_match_the_scalar_ones() {
        let mut rng = SplitMix64::new(5);
        for step_q12 in [4096u64, 9000, 32768, 1 << 20, 1 << 27] {
            for rounding in [0.0, 0.25, 0.5, 0.9] {
                let q = Quantizer::new(step_q12, rounding);
                for len in [0usize, 1, 7, 8, 9, 16, 33, 64, 257] {
                    let src: Vec<i16> = (0..len)
                        .map(|_| {
                            let v = rng.below(65536) as i32 - 32768;
                            v as i16
                        })
                        .collect();
                    let mut want = vec![0i32; len];
                    let (wn, wb) = q.quantize_row_stats_any(&src, &mut want);
                    let mut got = vec![0i32; len];
                    let (gn, gb) = q.quantize_row_stats(&src, &mut got);
                    assert_eq!(want, got, "quantize mismatch (step {step_q12})");
                    assert_eq!(
                        (wn, wb),
                        (gn, gb),
                        "quantize stats mismatch (step {step_q12})"
                    );

                    let d = Dequantizer::new(step_q12, 64);
                    let mut want = vec![0i16; len];
                    d.dequantize_row_any(
                        &src.iter().map(|&c| c as i32).collect::<Vec<_>>(),
                        &mut want,
                    );
                    let mut got = vec![0i16; len];
                    d.dequantize_row(&src.iter().map(|&c| c as i32).collect::<Vec<_>>(), &mut got);
                    assert_eq!(want, got, "dequantize mismatch (step {step_q12})");
                }
            }
        }
    }

    /// The extremes: the largest magnitude, the cap, and the zero.
    #[test]
    fn the_vector_kernels_handle_the_extremes() {
        let q = Quantizer::new(4096, 0.25);
        let src = [i16::MIN, -1, 0, 1, i16::MAX, 32767, -32768];
        let mut want = vec![0i32; src.len()];
        let (wn, wb) = q.quantize_row_stats_any(&src, &mut want);
        let mut got = vec![0i32; src.len()];
        let (gn, gb) = q.quantize_row_stats(&src, &mut got);
        assert_eq!(want, got);
        assert_eq!((wn, wb), (gn, gb));

        let d = Dequantizer::new(4096, 64);
        let qv = [i32::MIN, -1, 0, 1, i32::MAX, 32767, -32768];
        let mut want = vec![0i16; qv.len()];
        d.dequantize_row_any(&qv, &mut want);
        let mut got = vec![0i16; qv.len()];
        d.dequantize_row(&qv, &mut got);
        assert_eq!(want, got);
    }
}
