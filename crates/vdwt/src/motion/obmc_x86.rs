//! x86-64 kernels for overlapped prediction ([`obmc`](super::obmc)), bit-identical to the
//! portable code: 16 pixels per AVX2 register, and 8 per SSE register (like NEON) for blocks
//! 8 pixels wide.

use std::arch::x86_64::*;

use super::obmc::Kernels;

/// The kernels, or `None` when the CPU lacks AVX2 (which implies the SSE4.1 used). The
/// functions below assume it; this is the only way to reach them, and it checks.
pub(super) fn kernels() -> Option<Kernels> {
    std::is_x86_feature_detected!("avx2").then_some(Kernels { translate, blend })
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn translate(
    src: &[u8],
    stride: usize,
    height: usize,
    iy0: isize,
    ix: usize,
    hx: &[i16; 4],
    hy: &[i16; 4],
    taps: usize,
    tmp: &mut [i16],
    out: &mut [u16],
) {
    // `tmp` holds `rows + taps − 1` rows of the block and `out` `rows` rows.
    let len = (tmp.len() - out.len()) / (taps - 1);
    if len.is_multiple_of(16) {
        // SAFETY: as documented for `Kernels::translate`.
        return unsafe {
            if taps == 4 {
                translate16::<4>(src, stride, height, iy0, ix, hx, hy, len, out)
            } else {
                translate16::<2>(src, stride, height, iy0, ix, hx, hy, len, out)
            }
        };
    }
    for (r, t) in tmp.chunks_exact_mut(len).enumerate() {
        let y = (iy0 + r as isize).clamp(0, height as isize - 1) as usize;
        let start = y * stride + ix;
        // SAFETY: as documented for `Kernels::translate`.
        unsafe { horizontal(&src[start..start + len + taps - 1], hx, taps, t) };
    }
    let tmp = &*tmp;
    for (r, o) in out.chunks_exact_mut(len).enumerate() {
        let row = |j: usize| &tmp[(r + j.min(taps - 1)) * len..][..len];
        // SAFETY: every row holds `len` values.
        unsafe { vertical(&[row(0), row(1), row(2), row(3)], hy, taps, o) };
    }
}

/// [`translate`] for a width that is a multiple of 16: per 16 columns, each reference row is
/// filtered horizontally once into a register window of the last `T` rows, and each output row
/// filtered vertically from it.
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn translate16<const T: usize>(
    src: &[u8],
    stride: usize,
    height: usize,
    iy0: isize,
    ix: usize,
    hx: &[i16; 4],
    hy: &[i16; 4],
    len: usize,
    out: &mut [u16],
) {
    let rows = out.len() / len;
    assert!(rows > 0 && height > 0 && ix + len + T - 1 <= stride);
    assert!((height - 1) * stride + ix + len + T - 1 <= src.len());
    // Row `r`'s first sample: always inside `src` by the asserts above.
    let base = |r: usize| {
        let y = (iy0 + r as isize).clamp(0, height as isize - 1) as usize;
        // SAFETY: y < height, so the row's `len + T − 1` samples from `ix` are in `src`.
        unsafe { src.as_ptr().add(y * stride + ix) }
    };
    let (bias, t) = (_mm256_set1_epi16(128), hx.map(|v| _mm256_set1_epi16(v)));
    let pair = |a: i16, b: i16| i32::from(a as u16) | i32::from(b) << 16;
    let (p01, p23) = (
        _mm256_set1_epi32(pair(hy[0], hy[1])),
        _mm256_set1_epi32(pair(hy[2], hy[3])),
    );
    let (round, mid, max) = (
        _mm256_set1_epi32(1 << 11),
        _mm256_set1_epi32(512),
        _mm256_set1_epi16(1023),
    );
    let out = out.as_mut_ptr();
    for k in (0..len).step_by(16) {
        // Horizontal sums of 16 pixels of reference row `r`.
        let horizontal = |r: usize| {
            let p = base(r);
            // SAFETY: 16 samples from `k + i` (i < T) lie within the row's `len + T − 1`.
            unsafe {
                let tap = |i: usize| {
                    let v = _mm_loadu_si128(p.add(k + i).cast());
                    _mm256_mullo_epi16(_mm256_sub_epi16(_mm256_cvtepu8_epi16(v), bias), t[i])
                };
                let acc = _mm256_add_epi16(tap(0), tap(1));
                if T == 4 {
                    _mm256_add_epi16(acc, _mm256_add_epi16(tap(2), tap(3)))
                } else {
                    acc
                }
            }
        };
        let (mut h0, mut h1, mut h2) = (
            horizontal(0),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        if T == 4 {
            h1 = horizontal(1);
            h2 = horizontal(2);
        }
        for r in 0..rows {
            let h3 = horizontal(r + T - 1);
            let (a, b) = if T == 4 { (h0, h1) } else { (h0, h3) };
            let mut lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), p01);
            let mut hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), p01);
            if T == 4 {
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(h2, h3), p23));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(h2, h3), p23));
            }
            let fix =
                |v| _mm256_add_epi32(_mm256_srai_epi32::<12>(_mm256_add_epi32(v, round)), mid);
            let v = _mm256_min_epu16(_mm256_packus_epi32(fix(lo), fix(hi)), max);
            // SAFETY: row r of `out` holds `len` values, and k + 16 ≤ len.
            unsafe { _mm256_storeu_si256(out.add(r * len + k).cast(), v) };
            if T == 4 {
                (h0, h1, h2) = (h1, h2, h3);
            } else {
                h0 = h3;
            }
        }
    }
}

/// `out[k] = Σ_i hx[i]·(row[k + i] − 128)`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal(row: &[u8], hx: &[i16; 4], taps: usize, out: &mut [i16]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && row.len() >= len + taps - 1);
    let mut k = 0;
    let (bias, t) = (_mm256_set1_epi16(128), hx.map(|v| _mm256_set1_epi16(v)));
    while k + 16 <= len {
        // SAFETY: the row holds `len + taps − 1` samples, so 16 bytes from `k + i` (i < taps)
        // are inside it; the output holds `len` values.
        unsafe {
            let tap = |i: usize| {
                let v = _mm_loadu_si128(row.as_ptr().add(k + i).cast());
                _mm256_mullo_epi16(_mm256_sub_epi16(_mm256_cvtepu8_epi16(v), bias), t[i])
            };
            let mut acc = _mm256_add_epi16(tap(0), tap(1));
            if taps == 4 {
                acc = _mm256_add_epi16(acc, _mm256_add_epi16(tap(2), tap(3)));
            }
            _mm256_storeu_si256(out.as_mut_ptr().add(k).cast(), acc);
        }
        k += 16;
    }
    if k < len {
        let (bias, t) = (_mm_set1_epi16(128), hx.map(|v| _mm_set1_epi16(v)));
        // SAFETY: as above, 8 bytes from `k + i`.
        unsafe {
            let tap = |i: usize| {
                let v = _mm_loadl_epi64(row.as_ptr().add(k + i).cast());
                _mm_mullo_epi16(_mm_sub_epi16(_mm_cvtepu8_epi16(v), bias), t[i])
            };
            let mut acc = _mm_add_epi16(tap(0), tap(1));
            if taps == 4 {
                acc = _mm_add_epi16(acc, _mm_add_epi16(tap(2), tap(3)));
            }
            _mm_storeu_si128(out.as_mut_ptr().add(k).cast(), acc);
        }
    }
}

/// `out[k] = clamp(((Σ_j hy[j]·rows[j][k] + 2^11) >> 12) + 512, 0, 1023)`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn vertical(rows: &[&[i16]; 4], hy: &[i16; 4], taps: usize, out: &mut [u16]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && rows.iter().all(|r| r.len() >= len));
    let pair = |a: i16, b: i16| i32::from(a as u16) | i32::from(b) << 16;
    let (p01, p23) = (pair(hy[0], hy[1]), pair(hy[2], hy[3]));
    let mut k = 0;
    {
        let (p01, p23) = (_mm256_set1_epi32(p01), _mm256_set1_epi32(p23));
        let (round, mid, max) = (
            _mm256_set1_epi32(1 << 11),
            _mm256_set1_epi32(512),
            _mm256_set1_epi16(1023),
        );
        while k + 16 <= len {
            // SAFETY: every row and the output hold `len` values.
            unsafe {
                let load = |j: usize| _mm256_loadu_si256(rows[j].as_ptr().add(k).cast());
                let (h0, h1) = (load(0), load(1));
                // Within each 128-bit lane: values 0–3 (8–11) and 4–7 (12–15).
                let mut lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(h0, h1), p01);
                let mut hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(h0, h1), p01);
                if taps == 4 {
                    let (h2, h3) = (load(2), load(3));
                    lo =
                        _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(h2, h3), p23));
                    hi =
                        _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(h2, h3), p23));
                }
                let fix =
                    |v| _mm256_add_epi32(_mm256_srai_epi32::<12>(_mm256_add_epi32(v, round)), mid);
                // packus restores the order within lanes and clamps below at 0.
                let s = _mm256_min_epu16(_mm256_packus_epi32(fix(lo), fix(hi)), max);
                _mm256_storeu_si256(out.as_mut_ptr().add(k).cast(), s);
            }
            k += 16;
        }
    }
    if k < len {
        let (p01, p23) = (_mm_set1_epi32(p01), _mm_set1_epi32(p23));
        let (round, mid, max) = (
            _mm_set1_epi32(1 << 11),
            _mm_set1_epi32(512),
            _mm_set1_epi16(1023),
        );
        // SAFETY: every row and the output hold `len` values.
        unsafe {
            let load = |j: usize| _mm_loadu_si128(rows[j].as_ptr().add(k).cast());
            let (h0, h1) = (load(0), load(1));
            let mut lo = _mm_madd_epi16(_mm_unpacklo_epi16(h0, h1), p01);
            let mut hi = _mm_madd_epi16(_mm_unpackhi_epi16(h0, h1), p01);
            if taps == 4 {
                let (h2, h3) = (load(2), load(3));
                lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(h2, h3), p23));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(h2, h3), p23));
            }
            let fix = |v| _mm_add_epi32(_mm_srai_epi32::<12>(_mm_add_epi32(v, round)), mid);
            let s = _mm_min_epu16(_mm_packus_epi32(fix(lo), fix(hi)), max);
            _mm_storeu_si128(out.as_mut_ptr().add(k).cast(), s);
        }
    }
}

#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn blend(
    s: &[(&[u16], usize); 4],
    len: usize,
    rows: usize,
    v0: u32,
    cs: u32,
    shift: u32,
    dst: &mut [u8],
    stride: usize,
) {
    for r in 0..rows {
        let row = |q: usize| &s[q].0[r * s[q].1..r * s[q].1 + len];
        // SAFETY: as documented for `Kernels::blend`.
        unsafe {
            blend_row(
                &[row(0), row(1), row(2), row(3)],
                cs,
                v0 + r as u32,
                shift,
                &mut dst[r * stride..r * stride + len],
            )
        };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn blend_row(s: &[&[u16]; 4], cs: u32, v: u32, shift: u32, out: &mut [u8]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && cs <= 32 && s.iter().all(|r| r.len() >= len));
    let wv = ((cs - v) | v << 16) as i32;
    let mut k = 0;
    {
        let lanes = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let (cs16, wv) = (_mm256_set1_epi16(cs as i16), _mm256_set1_epi32(wv));
        let round = _mm256_set1_epi32(1 << (shift - 1));
        let count = _mm_cvtsi32_si128(shift as i32);
        while k + 16 <= len {
            // SAFETY: every row holds `len` values and the output `len` bytes.
            unsafe {
                let u = _mm256_add_epi16(lanes, _mm256_set1_epi16(k as i16));
                let wl = _mm256_sub_epi16(cs16, u);
                let load = |j: usize| _mm256_loadu_si256(s[j].as_ptr().add(k).cast());
                // At most 32·1023: these fit signed 16 bits.
                let top = _mm256_add_epi16(
                    _mm256_mullo_epi16(wl, load(0)),
                    _mm256_mullo_epi16(u, load(1)),
                );
                let bottom = _mm256_add_epi16(
                    _mm256_mullo_epi16(wl, load(2)),
                    _mm256_mullo_epi16(u, load(3)),
                );
                let fix =
                    |p| _mm256_sra_epi32(_mm256_add_epi32(_mm256_madd_epi16(p, wv), round), count);
                let lo = fix(_mm256_unpacklo_epi16(top, bottom));
                let hi = fix(_mm256_unpackhi_epi16(top, bottom));
                // Words 0–15 in order; bytes 0–7 of each lane, then the lanes' low halves.
                let words = _mm256_packus_epi32(lo, hi);
                let bytes = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi16(words, words));
                _mm_storeu_si128(
                    out.as_mut_ptr().add(k).cast(),
                    _mm256_castsi256_si128(bytes),
                );
            }
            k += 16;
        }
    }
    if k < len {
        let lanes = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let (cs16, wv) = (_mm_set1_epi16(cs as i16), _mm_set1_epi32(wv));
        let round = _mm_set1_epi32(1 << (shift - 1));
        let count = _mm_cvtsi32_si128(shift as i32);
        // SAFETY: every row holds `len` values and the output `len` bytes.
        unsafe {
            let u = _mm_add_epi16(lanes, _mm_set1_epi16(k as i16));
            let wl = _mm_sub_epi16(cs16, u);
            let load = |j: usize| _mm_loadu_si128(s[j].as_ptr().add(k).cast());
            let top = _mm_add_epi16(_mm_mullo_epi16(wl, load(0)), _mm_mullo_epi16(u, load(1)));
            let bottom = _mm_add_epi16(_mm_mullo_epi16(wl, load(2)), _mm_mullo_epi16(u, load(3)));
            let fix = |p| _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(p, wv), round), count);
            let lo = fix(_mm_unpacklo_epi16(top, bottom));
            let hi = fix(_mm_unpackhi_epi16(top, bottom));
            let words = _mm_packus_epi32(lo, hi);
            _mm_storel_epi64(
                out.as_mut_ptr().add(k).cast(),
                _mm_packus_epi16(words, words),
            );
        }
    }
}
