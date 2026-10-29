//! The in-loop Wiener filter (DESIGN.md §5.4): a symmetric 5×5 filter per plane and frame,
//! fitted by the encoder to bring the reconstruction closer to the source, and applied to the
//! reconstruction before it is shown and predicted from.
//!
//! The filter adds to every sample `r` a weighted sum of second differences,
//! `Σ_k c_k·(r[p + d_k] + r[p − d_k] − 2·r[p])` over the twelve offsets `d_k` of one half of the
//! 5×5 square, rounded from 1/64 and clamped to 0..=255. Flat areas pass unchanged. Outside the
//! plane, samples repeat the nearest edge sample.
//!
//! Least squares give the encoder the coefficients that minimize the squared error over
//! sampled rows; it sends them when the error they save is worth their bits. Coefficients are
//! coded as differences from the plane's previous ones (zero after an intra frame).

use crate::Plane;
use crate::pool::Pool;

/// Coefficient pairs.
pub(crate) const TAPS: usize = 12;
/// The offsets `(dy, dx)` of the pairs: one half of the 5×5 square around a sample.
const OFFSETS: [(i32, i32); TAPS] = [
    (0, 1),
    (0, 2),
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
];
/// Coefficients are in `1/2^SHIFT`.
const SHIFT: u32 = 6;
/// Largest coefficient magnitude: products with second differences fit 16 bits.
pub(crate) const MAX_COEF: i32 = 64;
/// Largest sum of coefficient magnitudes: whole sums fit 16 bits.
pub(crate) const MAX_SUM: i32 = 64;
/// Columns the fit handles at a time: its thirteen series of 16-bit values (6.5 KB) stay in L1.
const FIT_SPAN: usize = 256;
/// Rows per thread task.
const BAND: usize = 32;

/// One plane's filter coefficients, in 1/64.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Coefs(pub [i16; TAPS]);

impl Coefs {
    /// `Σ |c|`.
    pub(crate) fn sum(&self) -> i32 {
        self.0.iter().map(|&c| i32::from(c).abs()).sum()
    }
}

/// Filters `src` into `dst` (the same size).
pub(crate) fn apply(pool: &Pool, src: &Plane<u8>, dst: &mut Plane<u8>, coefs: &Coefs) {
    assert_eq!((src.width(), src.height()), (dst.width(), dst.height()));
    let (w, h) = (src.width(), src.height());
    let mut bands = dst.bands_mut(BAND);
    pool.for_each_mut(&mut bands, |_, band| {
        let mut acc = vec![0i32; w];
        // Five rows around `y`, two samples of edge on either side.
        let mut padded: [Vec<u8>; 5] = std::array::from_fn(|_| vec![0; w + 4]);
        let pad = |p: &mut Vec<u8>, y: isize| {
            let row = src.row(y.clamp(0, h as isize - 1) as usize);
            p[..2].fill(row[0]);
            p[2..w + 2].copy_from_slice(row);
            p[w + 2..].fill(row[w - 1]);
        };
        let rows = band.rows();
        for (r, p) in padded.iter_mut().enumerate() {
            pad(p, rows.start as isize + r as isize - 2);
        }
        for y in rows {
            if y > band.rows().start {
                padded.rotate_left(1);
                pad(&mut padded[4], y as isize + 2);
            }
            filter_row(&padded, coefs, &mut acc, band.row_mut(y));
        }
    });
}

/// Coefficients of the custom mid-frequency band-pass `c + a·(blur3(c) − blur5(c))`, in the
/// same symmetric 5×5 pair form as the Wiener filter. `strength_q8` is `a·256`.
pub(crate) fn band_coefs(strength_q8: u16) -> Coefs {
    // Separable binomial kernel [1 4 6 4 1]/16 against [1 2 1]/4.
    const K5: [i32; 5] = [1, 4, 6, 4, 1];
    band_pair_coefs(strength_q8, &K5, 256.0)
}

/// Coefficients of the wider band-pass `c + a·(blur3(c) − blur_w(c))`, where `blur_w` is the
/// five-tap binomial [1 2 3 2 1]/9. It moves the boosted octave down one step, which measures
/// better on the samtime clips than narrowing the high-pass side.
pub(crate) fn band_wide_coefs(strength_q8: u16) -> Coefs {
    const KW: [i32; 5] = [1, 2, 3, 2, 1];
    band_pair_coefs(strength_q8, &KW, 81.0)
}

/// Shared coefficient builder for `c + a·(K3 − wide)`, where `wide` is a normalised separable
/// 5-tap kernel. Off-centre terms are the direct kernel; the centre is implicit in the
/// second-difference form, exactly as for the six-parameter Wiener filter.
fn band_pair_coefs(strength_q8: u16, wide: &[i32; 5], wide_norm: f64) -> Coefs {
    const K3: [i32; 3] = [1, 2, 1];
    let a = f64::from(strength_q8) / 256.0;
    let k3 = |dy: i32, dx: i32| -> f64 {
        if dy.abs() > 1 || dx.abs() > 1 {
            0.0
        } else {
            f64::from(K3[(dy + 1) as usize] * K3[(dx + 1) as usize]) / 16.0
        }
    };
    let kw = |dy: i32, dx: i32| -> f64 {
        if dy.abs() > 2 || dx.abs() > 2 {
            0.0
        } else {
            f64::from(wide[(dy + 2) as usize] * wide[(dx + 2) as usize]) / wide_norm
        }
    };
    let mut coefs = [0i16; TAPS];
    for (k, &(dy, dx)) in OFFSETS.iter().enumerate() {
        coefs[k] = (a * (k3(dy, dx) - kw(dy, dx)) * f64::from(1 << SHIFT))
            .round()
            .clamp(-f64::from(MAX_COEF), f64::from(MAX_COEF)) as i16;
    }
    // Keep the same coefficient bounds as the Wiener filter.
    let mut c = Coefs(coefs);
    while c.sum() > MAX_SUM {
        let k = (0..TAPS).max_by_key(|&k| c.0[k].abs()).expect("taps");
        c.0[k] -= c.0[k].signum();
    }
    c
}

/// One-plane serial application, for the display post-filter (which runs after decode and has
/// no codec pool at hand). The row kernel is still the SIMD one.
pub(crate) fn apply_serial(src: &Plane<u8>, dst: &mut Plane<u8>, coefs: &Coefs) {
    assert_eq!((src.width(), src.height()), (dst.width(), dst.height()));
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return;
    }
    let mut acc = vec![0i32; w];
    let mut padded: [Vec<u8>; 5] = std::array::from_fn(|_| vec![0; w + 4]);
    let pad = |p: &mut Vec<u8>, y: isize| {
        let row = src.row(y.clamp(0, h as isize - 1) as usize);
        p[..2].fill(row[0]);
        p[2..w + 2].copy_from_slice(row);
        p[w + 2..].fill(row[w - 1]);
    };
    for (r, p) in padded.iter_mut().enumerate() {
        pad(p, r as isize - 2);
    }
    for y in 0..h {
        if y > 0 {
            padded.rotate_left(1);
            pad(&mut padded[4], y as isize + 2);
        }
        filter_row(&padded, coefs, &mut acc, dst.row_mut(y));
    }
}

/// Filters one row from its five padded neighbourhood rows.
fn filter_row(rows: &[Vec<u8>; 5], coefs: &Coefs, acc: &mut [i32], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx2_detected() {
        // SAFETY: AVX2 is present.
        return unsafe { filter_row_avx2(rows, coefs, acc, out) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    // SAFETY: NEON is present.
    return unsafe { filter_row_neon(rows, coefs, acc, out) };
    #[allow(unreachable_code)]
    filter_row_any(rows, coefs, acc, out)
}

/// [`filter_row_avx2`] on NEON, 16 samples at a time: `c·(a + b)` by widening adds and
/// multiply-accumulates in 16-bit lanes, wrapping as there.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn filter_row_neon(rows: &[Vec<u8>; 5], coefs: &Coefs, acc: &mut [i32], out: &mut [u8]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let total: i32 = coefs.0.iter().map(|&c| i32::from(c)).sum();
    assert!(rows.iter().all(|r| r.len() >= n + 4));
    let at = |dy: i32, dx: i32| rows[(2 + dy) as usize][(2 + dx) as usize..].as_ptr();
    let pairs = OFFSETS.map(|(dy, dx)| (at(dy, dx), at(-dy, -dx)));
    let centre = at(0, 0);
    let mut x = 0;
    while x + 16 <= n {
        // SAFETY: the padded rows hold `n + 4` samples, so 16 from `x + 4` at most are inside;
        // `out` holds `n`.
        unsafe {
            let ld = |p: *const u8| vld1q_u8(p.add(x));
            let (mut lo, mut hi) = (vdupq_n_s16(0), vdupq_n_s16(0));
            for (&(pa, pb), &c) in pairs.iter().zip(&coefs.0) {
                if c == 0 {
                    continue;
                }
                let (a, b) = (ld(pa), ld(pb));
                let sl = vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(a), vget_low_u8(b)));
                let sh = vreinterpretq_s16_u16(vaddl_high_u8(a, b));
                lo = vmlaq_n_s16(lo, sl, c);
                hi = vmlaq_n_s16(hi, sh, c);
            }
            let r = ld(centre);
            let fix = |sum: int16x8_t, r: int16x8_t| {
                let v = vmlsq_n_s16(sum, r, (2 * total) as i16);
                vaddq_s16(
                    r,
                    vshrq_n_s16::<{ SHIFT as i32 }>(vaddq_s16(v, vdupq_n_s16(1 << (SHIFT - 1)))),
                )
            };
            let rl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r)));
            let rh = vreinterpretq_s16_u16(vmovl_high_u8(r));
            let o = vcombine_u8(vqmovun_s16(fix(lo, rl)), vqmovun_s16(fix(hi, rh)));
            vst1q_u8(out.as_mut_ptr().add(x), o);
        }
        x += 16;
    }
    filter_span(rows, coefs, &mut acc[x..], &mut out[x..], x);
}

/// 32 samples at a time in 16-bit lanes: `c·(a + b)` by one multiply-add of interleaved
/// bytes, summed with wrapping adds. The sum of coefficient magnitudes keeps every whole sum
/// (`|Σ c·(a + b − 2r)| ≤ 64·510`) inside 16 bits, so wrapping partial sums give it exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_row_avx2(rows: &[Vec<u8>; 5], coefs: &Coefs, acc: &mut [i32], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let n = out.len();
    let taps = coefs
        .0
        .map(|c| _mm256_set1_epi16((c as u8 as i16) << 8 | c as u8 as i16));
    let total: i32 = coefs.0.iter().map(|&c| i32::from(c)).sum();
    let twice = _mm256_set1_epi16((2 * total) as i16);
    let round = _mm256_set1_epi16(1 << (SHIFT - 1));
    let zero = _mm256_setzero_si256();
    assert!(rows.iter().all(|r| r.len() >= n + 4));
    // The first sample of every pair's rows.
    let at = |dy: i32, dx: i32| rows[(2 + dy) as usize][(2 + dx) as usize..].as_ptr();
    let pairs = OFFSETS.map(|(dy, dx)| (at(dy, dx), at(-dy, -dx)));
    let centre = at(0, 0);
    let mut x = 0;
    while x + 32 <= n {
        // SAFETY: the padded rows hold `n + 4` samples, so 32 from `x + 4` at most are inside;
        // `out` holds `n`.
        unsafe {
            let ld = |p: *const u8| _mm256_loadu_si256(p.add(x).cast());
            let (mut lo, mut hi) = (zero, zero);
            for k in 0..TAPS {
                let (a, b) = (ld(pairs[k].0), ld(pairs[k].1));
                lo = _mm256_add_epi16(
                    lo,
                    _mm256_maddubs_epi16(_mm256_unpacklo_epi8(a, b), taps[k]),
                );
                hi = _mm256_add_epi16(
                    hi,
                    _mm256_maddubs_epi16(_mm256_unpackhi_epi8(a, b), taps[k]),
                );
            }
            let r = ld(centre);
            let fix = |sum: __m256i, r: __m256i| {
                let v = _mm256_sub_epi16(sum, _mm256_mullo_epi16(r, twice));
                _mm256_add_epi16(
                    r,
                    _mm256_srai_epi16::<{ SHIFT as i32 }>(_mm256_add_epi16(v, round)),
                )
            };
            let lo = fix(lo, _mm256_unpacklo_epi8(r, zero));
            let hi = fix(hi, _mm256_unpackhi_epi8(r, zero));
            // Lanes hold samples 0–7 and 16–23 (lo), 8–15 and 24–31 (hi): packing restores
            // the order.
            _mm256_storeu_si256(out[x..].as_mut_ptr().cast(), _mm256_packus_epi16(lo, hi));
        }
        x += 32;
    }
    filter_span(rows, coefs, &mut acc[x..], &mut out[x..], x);
}

#[inline(always)]
fn filter_row_any(rows: &[Vec<u8>; 5], coefs: &Coefs, acc: &mut [i32], out: &mut [u8]) {
    filter_span(rows, coefs, acc, out, 0)
}

/// Filters `out`, the samples from `x0` on.
#[inline(always)]
fn filter_span(rows: &[Vec<u8>; 5], coefs: &Coefs, acc: &mut [i32], out: &mut [u8], x0: usize) {
    let n = out.len();
    let acc = &mut acc[..n];
    let centre = &rows[2][2 + x0..2 + x0 + n];
    acc.fill(0);
    for (&(dy, dx), &c) in OFFSETS.iter().zip(&coefs.0) {
        if c == 0 {
            continue;
        }
        let c = i32::from(c);
        let a = &rows[(2 + dy) as usize][(2 + dx) as usize + x0..][..n];
        let b = &rows[(2 - dy) as usize][(2 - dx) as usize + x0..][..n];
        for (((acc, &a), &b), &r) in acc.iter_mut().zip(a).zip(b).zip(centre) {
            *acc += c * (i32::from(a) + i32::from(b) - 2 * i32::from(r));
        }
    }
    for ((o, &acc), &r) in out.iter_mut().zip(&*acc).zip(centre) {
        *o = (i32::from(r) + ((acc + (1 << (SHIFT - 1))) >> SHIFT)).clamp(0, 255) as u8;
    }
}

/// The encoder's choice for one plane: the coefficients and the squared error they save over
/// the whole plane (estimated).
pub(crate) struct Fit {
    pub coefs: Coefs,
    pub saved: f64,
}

/// Fits coefficients bringing `rec` closer to `src` on the rows `y` with `⌊y/2⌋` a multiple
/// of `step` (both row parities: the wavelet treats them differently).
pub(crate) fn fit(pool: &Pool, src: &Plane<u8>, rec: &Plane<u8>, step: usize) -> Fit {
    let (w, h) = (src.width(), src.height());
    let none = Fit {
        coefs: Coefs::default(),
        saved: 0.0,
    };
    if w < 5 || h < 5 {
        return none;
    }
    // The normal equations: Σ t·tᵀ and Σ t·e over interior samples, t the second differences
    // and e the error to remove.
    // Pairs of rows, so both row parities of the wavelet count.
    let rows: Vec<usize> = (2..h - 2).filter(|y| y / 2 % step == 0).collect();
    // Each row in spans of FIT_SPAN columns: the thirteen series of a whole HD row would not
    // fit an A55's L1 between being written and read back.
    let parts = pool.map(rows.len().div_ceil(8), |part| {
        let mut normal = Normal::default();
        let span = FIT_SPAN.min(w - 4);
        let mut t = vec![vec![0i16; span]; TAPS];
        let mut e = vec![0i16; span];
        for &y in rows.iter().skip(8 * part).take(8) {
            for x0 in (0..w - 4).step_by(span) {
                let n = span.min(w - 4 - x0);
                normal.add_row(src, rec, y, x0, &mut t, &mut e[..n]);
            }
        }
        normal
    });
    let mut normal = Normal::default();
    for p in &parts {
        normal.merge(p);
    }
    let Some(x) = normal.solve() else {
        return none;
    };
    // Out of range, the whole solution shrinks: any fraction of it still saves error (the
    // error is a convex quadratic, least at the full solution).
    let largest = x.iter().map(|v| v.abs()).fold(0.0, f64::max) * f64::from(1 << SHIFT);
    let total = x.iter().map(|v| v.abs()).sum::<f64>() * f64::from(1 << SHIFT);
    let shrink = (f64::from(MAX_COEF) / largest)
        .min(f64::from(MAX_SUM) / total)
        .min(1.0);
    let mut coefs = Coefs(std::array::from_fn(|k| {
        (x[k] * shrink * f64::from(1 << SHIFT))
            .round()
            .clamp(-f64::from(MAX_COEF), f64::from(MAX_COEF)) as i16
    }));
    // Rounding may still overshoot the sum.
    while coefs.sum() > MAX_SUM {
        let k = (0..TAPS).max_by_key(|&k| coefs.0[k].abs()).expect("taps");
        coefs.0[k] -= coefs.0[k].signum();
    }
    // Squared error saved with the rounded coefficients, scaled from the sampled rows.
    let c: [f64; TAPS] = std::array::from_fn(|k| f64::from(coefs.0[k]) / f64::from(1 << SHIFT));
    let mut change = 0.0;
    for k in 0..TAPS {
        change -= 2.0 * c[k] * normal.te[k] as f64;
        for l in 0..TAPS {
            change += c[k] * c[l] * normal.tt[k.min(l)][k.max(l)] as f64;
        }
    }
    let scale = (h - 4) as f64 / rows.len().max(1) as f64;
    Fit {
        coefs,
        saved: -change * scale,
    }
}

/// Sums for the normal equations (upper triangle of `tt`).
#[derive(Default)]
struct Normal {
    tt: [[i64; TAPS]; TAPS],
    te: [i64; TAPS],
}

impl Normal {
    /// Adds the interior of row `y`, with `t` and `e` as scratch (`TAPS` rows and one, each
    /// `w − 4` long).
    fn add_row(
        &mut self,
        src: &Plane<u8>,
        rec: &Plane<u8>,
        y: usize,
        x0: usize,
        t: &mut [Vec<i16>],
        e: &mut [i16],
    ) {
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::avx2_detected() {
            // SAFETY: AVX2 is present.
            return unsafe { self.add_row_avx2(src, rec, y, x0, t, e) };
        }
        Self::differences(src, rec, y, x0, t, e);
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        return self.add_neon(t, e);
        #[allow(unreachable_code)]
        self.add_any(t, e)
    }

    /// [`Self::add_any`] on NEON, blocked like [`Self::add_row_avx2`]: dot products two series
    /// by four at a time, eight values per step.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn add_neon(&mut self, t: &[Vec<i16>], e: &[i16]) {
        use std::arch::aarch64::*;
        let n = e.len();
        assert!(t.iter().all(|t| t.len() >= n));
        let series: [&[i16]; TAPS + 1] =
            std::array::from_fn(|k| if k < TAPS { &t[k][..n] } else { &e[..n] });
        let block = |rows: [usize; 2], cols: [usize; 4]| -> [[i64; 4]; 2] {
            let mut out = [[0i64; 4]; 2];
            let mut k = 0;
            while k + 8 <= n {
                // Products are below 2^18, two per lane per step: 2048 steps fit 32 bits.
                let end = (k + 8 * 2048).min(n);
                // SAFETY: (all below) NEON is present.
                let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; 2];
                while k + 8 <= end {
                    // SAFETY: every series holds `n` values.
                    unsafe {
                        let ld = |s: usize| vld1q_s16(series[s].as_ptr().add(k));
                        let a = rows.map(ld);
                        let b = cols.map(ld);
                        for (acc, a) in acc.iter_mut().zip(a) {
                            for (acc, b) in acc.iter_mut().zip(b) {
                                *acc = vmlal_s16(*acc, vget_low_s16(a), vget_low_s16(b));
                                *acc = vmlal_high_s16(*acc, a, b);
                            }
                        }
                    }
                    k += 8;
                }
                for (out, acc) in out.iter_mut().zip(&acc) {
                    for (out, &acc) in out.iter_mut().zip(acc) {
                        // SAFETY: NEON is present.
                        *out += unsafe { vaddlvq_s32(acc) };
                    }
                }
            }
            for (out, &r) in out.iter_mut().zip(&rows) {
                for (out, &c) in out.iter_mut().zip(&cols) {
                    *out += (k..n)
                        .map(|i| i64::from(series[r][i]) * i64::from(series[c][i]))
                        .sum::<i64>();
                }
            }
            out
        };
        for k0 in (0..TAPS).step_by(2) {
            for l0 in (k0..=TAPS).step_by(4) {
                let cols = std::array::from_fn(|j| (l0 + j).min(TAPS));
                let sums = block([k0, k0 + 1], cols);
                for (i, sums) in sums.iter().enumerate() {
                    let k = k0 + i;
                    for (j, &v) in sums.iter().enumerate() {
                        let l = l0 + j;
                        if l < k || l > TAPS {
                            continue;
                        }
                        if l == TAPS {
                            self.te[k] += v;
                        } else {
                            self.tt[k][l] += v;
                        }
                    }
                }
            }
        }
    }

    /// The second differences `t` and errors `e` of row `y`'s interior, `e.len()` columns from
    /// interior column `x0`.
    #[inline(always)]
    fn differences(
        src: &Plane<u8>,
        rec: &Plane<u8>,
        y: usize,
        x0: usize,
        t: &mut [Vec<i16>],
        e: &mut [i16],
    ) {
        let (n, c) = (e.len(), 2 + x0);
        let r = &rec.row(y)[c..c + n];
        for (k, &(dy, dx)) in OFFSETS.iter().enumerate() {
            let a = &rec.row((y as i32 + dy) as usize)[(c as i32 + dx) as usize..][..n];
            let b = &rec.row((y as i32 - dy) as usize)[(c as i32 - dx) as usize..][..n];
            for (((t, &a), &b), &r) in t[k][..n].iter_mut().zip(a).zip(b).zip(r) {
                *t = i16::from(a) + i16::from(b) - 2 * i16::from(r);
            }
        }
        for ((e, &s), &r) in e.iter_mut().zip(&src.row(y)[c..c + n]).zip(r) {
            *e = i16::from(s) - i16::from(r);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_row_avx2(
        &mut self,
        src: &Plane<u8>,
        rec: &Plane<u8>,
        y: usize,
        x0: usize,
        t: &mut [Vec<i16>],
        e: &mut [i16],
    ) {
        use std::arch::x86_64::*;
        Self::differences(src, rec, y, x0, t, e);
        let n = e.len();
        assert!(t.iter().all(|t| t.len() >= n));
        // The thirteen series: the second differences, then the errors.
        let series: [&[i16]; TAPS + 1] =
            std::array::from_fn(|k| if k < TAPS { &t[k][..n] } else { &e[..n] });
        // Dot products two series by four at a time, from registers (each loaded value serves
        // four or two products).
        let block = |rows: [usize; 2], cols: [usize; 4]| -> [[i64; 4]; 2] {
            let mut out = [[0i64; 4]; 2];
            let mut k = 0;
            while k + 16 <= n {
                // Pairs of products are below 2^19: 2048 of them per lane fit 32 bits.
                let end = (k + 16 * 2048).min(n);
                let mut acc = [[_mm256_setzero_si256(); 4]; 2];
                while k + 16 <= end {
                    // SAFETY: every series holds `n` values.
                    unsafe {
                        let ld = |s: usize| _mm256_loadu_si256(series[s].as_ptr().add(k).cast());
                        let a = rows.map(ld);
                        let b = cols.map(ld);
                        for (acc, a) in acc.iter_mut().zip(a) {
                            for (acc, b) in acc.iter_mut().zip(b) {
                                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(a, b));
                            }
                        }
                    }
                    k += 16;
                }
                for (out, acc) in out.iter_mut().zip(&acc) {
                    for (out, &acc) in out.iter_mut().zip(acc) {
                        let mut lanes = [0i32; 8];
                        // SAFETY: eight 32-bit lanes.
                        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
                        *out += lanes.iter().map(|&v| i64::from(v)).sum::<i64>();
                    }
                }
            }
            for (out, &r) in out.iter_mut().zip(&rows) {
                for (out, &c) in out.iter_mut().zip(&cols) {
                    *out += (k..n)
                        .map(|i| i64::from(series[r][i]) * i64::from(series[c][i]))
                        .sum::<i64>();
                }
            }
            out
        };
        for k0 in (0..TAPS).step_by(2) {
            for l0 in (k0..=TAPS).step_by(4) {
                let cols = std::array::from_fn(|j| (l0 + j).min(TAPS));
                let sums = block([k0, k0 + 1], cols);
                for (i, sums) in sums.iter().enumerate() {
                    let k = k0 + i;
                    for (j, &v) in sums.iter().enumerate() {
                        let l = l0 + j;
                        if l < k || l > TAPS {
                            continue;
                        }
                        if l == TAPS {
                            self.te[k] += v;
                        } else {
                            self.tt[k][l] += v;
                        }
                    }
                }
            }
        }
    }

    #[cfg_attr(
        all(target_arch = "aarch64", target_feature = "neon"),
        allow(dead_code)
    )]
    #[inline(always)]
    fn add_any(&mut self, t: &[Vec<i16>], e: &[i16]) {
        // Products are below 2^18: 4096 of them fit 32 bits.
        let dot = |a: &[i16], b: &[i16]| -> i64 {
            a.chunks(4096)
                .zip(b.chunks(4096))
                .map(|(a, b)| {
                    i64::from(
                        a.iter()
                            .zip(b)
                            .map(|(&a, &b)| i32::from(a) * i32::from(b))
                            .sum::<i32>(),
                    )
                })
                .sum()
        };
        for k in 0..TAPS {
            for l in k..TAPS {
                self.tt[k][l] += dot(&t[k], &t[l]);
            }
            self.te[k] += dot(&t[k], e);
        }
    }

    fn merge(&mut self, other: &Normal) {
        for k in 0..TAPS {
            for l in k..TAPS {
                self.tt[k][l] += other.tt[k][l];
            }
            self.te[k] += other.te[k];
        }
    }

    /// The least-squares coefficients (in sample units), by Gaussian elimination, or `None`
    /// for a singular system (a flat plane).
    fn solve(&self) -> Option<[f64; TAPS]> {
        let mut a: [[f64; TAPS + 1]; TAPS] = std::array::from_fn(|k| {
            std::array::from_fn(|l| {
                if l < TAPS {
                    self.tt[k.min(l)][k.max(l)] as f64
                } else {
                    self.te[k] as f64
                }
            })
        });
        let trace: f64 = (0..TAPS).map(|k| a[k][k]).sum();
        if trace <= 0.0 {
            return None;
        }
        // A little ridge keeps nearly flat planes stable.
        for (k, row) in a.iter_mut().enumerate() {
            row[k] += 1e-6 * trace;
        }
        for col in 0..TAPS {
            let pivot = (col..TAPS).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
            if a[pivot][col].abs() < 1e-9 * trace {
                return None;
            }
            a.swap(col, pivot);
            let pivot_row = a[col];
            for (r, row) in a.iter_mut().enumerate() {
                if r != col {
                    let f = row[col] / pivot_row[col];
                    for (v, &p) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                        *v -= f * p;
                    }
                }
            }
        }
        Some(std::array::from_fn(|k| a[k][TAPS] / a[k][k]))
    }
}

/// Checks the filter kernel this CPU runs against the portable code on `cases` random rows
/// and coefficient sets (at and within the bounds).
pub fn verify(cases: usize, seed: u64) -> Result<(), String> {
    let mut rng = crate::rng::SplitMix64::new(seed);
    for case in 0..cases {
        let n = 1 + rng.below(100) as usize;
        let rows: [Vec<u8>; 5] = std::array::from_fn(|_| {
            (0..n + 4)
                .map(|_| {
                    if case % 3 == 0 {
                        255 * rng.below(2) as u8
                    } else {
                        rng.below(256) as u8
                    }
                })
                .collect()
        });
        let mut coefs = Coefs::default();
        let mut left = MAX_SUM;
        for c in &mut coefs.0 {
            let m = left.min(MAX_COEF);
            let v = if case % 2 == 0 {
                m
            } else {
                rng.below(m as u64 + 1) as i32
            };
            *c = if rng.below(2) == 0 { v } else { -v } as i16;
            left -= v;
        }
        let mut acc = vec![0; n];
        let (mut want, mut got) = (vec![0; n], vec![0; n]);
        filter_row_any(&rows, &coefs, &mut acc, &mut want);
        filter_row(&rows, &coefs, &mut acc, &mut got);
        if want != got {
            return Err(format!("loop filter case {case}: {coefs:?}"));
        }
    }
    Ok(())
}

/// Bits of `coefs` coded against `previous`.
pub(crate) fn coded_bits(coefs: &Coefs, previous: &Coefs) -> u32 {
    let mut w = BitWriter::default();
    write_coefs(&mut w, coefs, previous);
    w.len
}

/// Writes `coefs` as differences from `previous`.
pub(crate) fn write_coefs(w: &mut BitWriter, coefs: &Coefs, previous: &Coefs) {
    for (&c, &p) in coefs.0.iter().zip(&previous.0) {
        w.signed(i32::from(c) - i32::from(p));
    }
}

/// Reads coefficients written by [`write_coefs`].
pub(crate) fn read_coefs(r: &mut BitReader, previous: &Coefs) -> Option<Coefs> {
    let mut out = Coefs::default();
    for (c, &p) in out.0.iter_mut().zip(&previous.0) {
        let v = i32::from(p) + r.signed()?;
        if v.abs() > MAX_COEF {
            return None;
        }
        *c = v as i16;
    }
    (out.sum() <= MAX_SUM).then_some(out)
}

/// Bits, most significant first, into bytes.
#[derive(Default)]
pub(crate) struct BitWriter {
    pub bytes: Vec<u8>,
    len: u32,
}

impl BitWriter {
    pub fn bit(&mut self, b: bool) {
        if self.len.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if b {
            *self.bytes.last_mut().expect("pushed above") |= 0x80 >> (self.len % 8);
        }
        self.len += 1;
    }

    /// Exp-Golomb code of `v ≥ 0`.
    fn unsigned(&mut self, v: u32) {
        let v = v + 1;
        let n = 32 - v.leading_zeros();
        for _ in 1..n {
            self.bit(false);
        }
        for i in (0..n).rev() {
            self.bit(v >> i & 1 == 1);
        }
    }

    /// Signed Exp-Golomb: 0, 1, −1, 2, −2, …
    fn signed(&mut self, v: i32) {
        self.unsigned(if v > 0 {
            2 * v as u32 - 1
        } else {
            2 * v.unsigned_abs()
        });
    }
}

/// Reads what a [`BitWriter`] wrote.
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub fn bit(&mut self) -> Option<bool> {
        let byte = *self.bytes.get(self.pos / 8)?;
        let b = byte & (0x80 >> (self.pos % 8)) != 0;
        self.pos += 1;
        Some(b)
    }

    fn unsigned(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while !self.bit()? {
            zeros += 1;
            if zeros > 16 {
                return None;
            }
        }
        let mut v = 1u32;
        for _ in 0..zeros {
            v = v << 1 | u32::from(self.bit()?);
        }
        Some(v - 1)
    }

    fn signed(&mut self) -> Option<i32> {
        let u = self.unsigned()?;
        Some(if u % 2 == 1 {
            u.div_ceil(2) as i32
        } else {
            -((u / 2) as i32)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_round_trip() {
        let previous = Coefs([3, -2, 0, 0, 1, 24, -14, 5, 0, 0, -1, 2]);
        let coefs = Coefs([0, 7, -9, 0, 1, 10, -6, 4, -14, 3, 0, 0]);
        let mut w = BitWriter::default();
        write_coefs(&mut w, &coefs, &previous);
        w.bit(true);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(read_coefs(&mut r, &previous), Some(coefs));
        assert_eq!(r.bit(), Some(true));
    }

    #[test]
    fn simd_rows_match_the_portable_code() {
        verify(300, 9).unwrap();
    }

    #[test]
    #[ignore]
    fn speed() {
        let (w, h) = (1280, 720);
        let mut rng = crate::rng::SplitMix64::new(1);
        let mut src = Plane::new(w, h);
        let mut rec = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = (128.0 + 60.0 * ((x as f64 * 0.1).sin() + (y as f64 * 0.07).cos())) as u8;
                src.set(x, y, v);
                rec.set(x, y, v.saturating_add(rng.below(5) as u8));
            }
        }
        let pool = Pool::new(1);
        let t = std::time::Instant::now();
        let f = fit(&pool, &src, &rec, 1);
        eprintln!(
            "fit {:.2} ms {:?}",
            t.elapsed().as_secs_f64() * 1e3,
            f.coefs
        );
        let mut out = Plane::new(w, h);
        let t = std::time::Instant::now();
        for _ in 0..10 {
            apply(
                &pool,
                &rec,
                &mut out,
                &Coefs([3, -2, 1, 0, 4, -5, 2, 1, 0, -1, 2, 1]),
            );
        }
        eprintln!("apply {:.2} ms", t.elapsed().as_secs_f64() * 1e2);
    }

    #[test]
    fn the_fitted_filter_undoes_a_blur() {
        let (w, h) = (96, 64);
        let mut rng = crate::rng::SplitMix64::new(5);
        let mut src = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = 128.0
                    + 60.0 * ((x as f64 * 0.7).sin() * (y as f64 * 0.45).cos())
                    + (rng.below(21) as f64 - 10.0);
                src.set(x, y, v.clamp(0.0, 255.0) as u8);
            }
        }
        // A horizontal [1 6 1]/8 blur.
        let mut rec = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let at = |x: isize| u32::from(src.get(x.clamp(0, w as isize - 1) as usize, y));
                let v = (at(x as isize - 1) + 6 * at(x as isize) + at(x as isize + 1) + 4) / 8;
                rec.set(x, y, v as u8);
            }
        }
        let pool = Pool::new(2);
        let fit = fit(&pool, &src, &rec, 1);
        assert!(fit.saved > 0.0);
        let mut out = Plane::new(w, h);
        apply(&pool, &rec, &mut out, &fit.coefs);
        let sse = |a: &Plane<u8>| -> f64 {
            (2..h - 2)
                .flat_map(|y| (2..w - 2).map(move |x| (x, y)))
                .map(|(x, y)| (f64::from(a.get(x, y)) - f64::from(src.get(x, y))).powi(2))
                .sum()
        };
        let (before, after) = (sse(&rec), sse(&out));
        assert!(after < 0.7 * before, "{before} -> {after}");
        // The estimate is close to what filtering saves.
        let saved = before - after;
        assert!(
            (fit.saved - saved).abs() < 0.1 * saved,
            "{} vs {saved}",
            fit.saved
        );
    }
}
