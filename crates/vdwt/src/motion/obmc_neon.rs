//! NEON kernels for overlapped prediction ([`obmc`](super::obmc)), 8 pixels per register. The
//! same arithmetic as the x86 kernels, which are verified against the portable code.

use std::arch::aarch64::*;

use super::obmc::Kernels;

pub(super) const KERNELS: Kernels = Kernels { translate, blend };

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
    if len.is_multiple_of(8) {
        // SAFETY: as documented for `Kernels::translate`.
        return unsafe {
            if taps == 4 {
                translate8::<4>(src, stride, height, iy0, ix, hx, hy, len, out)
            } else {
                translate8::<2>(src, stride, height, iy0, ix, hx, hy, len, out)
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

/// [`translate`] for a width that is a multiple of 8: per 8 columns, each reference row is
/// filtered horizontally once into a register window of the last `T` rows, and each output row
/// filtered vertically from it (the x86 `translate16`).
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn translate8<const T: usize>(
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
    let out = out.as_mut_ptr();
    for k in (0..len).step_by(8) {
        // Horizontal sums of 8 pixels of reference row `r`, centred on 128 (wrapping like the
        // portable code; the whole sums fit 16 bits).
        let horizontal = |r: usize| {
            let p = base(r);
            // SAFETY: 8 samples from `k + i` (i < T) lie within the row's `len + T − 1`.
            unsafe {
                let c = |i: usize| {
                    vreinterpretq_s16_u16(vsubl_u8(vld1_u8(p.add(k + i)), vdup_n_u8(128)))
                };
                let acc = vmlaq_n_s16(vmulq_n_s16(c(0), hx[0]), c(1), hx[1]);
                if T == 4 {
                    vmlaq_n_s16(vmlaq_n_s16(acc, c(2), hx[2]), c(3), hx[3])
                } else {
                    acc
                }
            }
        };
        let zero = vdupq_n_s16(0);
        let (mut h0, mut h1, mut h2) = (horizontal(0), zero, zero);
        if T == 4 {
            h1 = horizontal(1);
            h2 = horizontal(2);
        }
        for r in 0..rows {
            let h3 = horizontal(r + T - 1);
            let (a, b) = if T == 4 { (h0, h1) } else { (h0, h3) };
            let mut lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(a), hy[0]), vget_low_s16(b), hy[1]);
            let mut hi = vmlal_high_n_s16(vmull_high_n_s16(a, hy[0]), b, hy[1]);
            if T == 4 {
                lo = vmlal_n_s16(
                    vmlal_n_s16(lo, vget_low_s16(h2), hy[2]),
                    vget_low_s16(h3),
                    hy[3],
                );
                hi = vmlal_high_n_s16(vmlal_high_n_s16(hi, h2, hy[2]), h3, hy[3]);
            }
            // ((x + 2^11) >> 12) + 512, saturated to 0..=65535 and clamped at 1023.
            let fix = |v| vaddq_s32(vrshrq_n_s32::<12>(v), vdupq_n_s32(512));
            let v = vcombine_u16(vqmovun_s32(fix(lo)), vqmovun_s32(fix(hi)));
            // SAFETY: row r of `out` holds `len` values, and k + 8 ≤ len.
            unsafe { vst1q_u16(out.add(r * len + k), vminq_u16(v, vdupq_n_u16(1023))) };
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
unsafe fn horizontal(row: &[u8], hx: &[i16; 4], taps: usize, out: &mut [i16]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && row.len() >= len + taps - 1);
    let mut k = 0;
    while k < len {
        // SAFETY: the row holds `len + taps − 1` samples, so 8 bytes from `k + i` (i < taps)
        // are inside it; the output holds `len` values.
        unsafe {
            let tap = |i: usize| {
                let v = vld1_u8(row.as_ptr().add(k + i));
                // Centred on 128: x − 128 as a signed 16-bit value.
                let c = vreinterpretq_s16_u16(vsubl_u8(v, vdup_n_u8(128)));
                vmulq_n_s16(c, hx[i])
            };
            let mut acc = vaddq_s16(tap(0), tap(1));
            if taps == 4 {
                acc = vaddq_s16(acc, vaddq_s16(tap(2), tap(3)));
            }
            vst1q_s16(out.as_mut_ptr().add(k), acc);
        }
        k += 8;
    }
}

/// `out[k] = clamp(((Σ_j hy[j]·rows[j][k] + 2^11) >> 12) + 512, 0, 1023)`.
#[inline]
unsafe fn vertical(rows: &[&[i16]; 4], hy: &[i16; 4], taps: usize, out: &mut [u16]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && rows.iter().all(|r| r.len() >= len));
    let mut k = 0;
    while k < len {
        // SAFETY: every row and the output hold `len` values.
        unsafe {
            let h = |j: usize| vld1q_s16(rows[j].as_ptr().add(k));
            let (h0, h1) = (h(0), h(1));
            let mut lo = vmull_n_s16(vget_low_s16(h0), hy[0]);
            let mut hi = vmull_high_n_s16(h0, hy[0]);
            lo = vmlal_n_s16(lo, vget_low_s16(h1), hy[1]);
            hi = vmlal_high_n_s16(hi, h1, hy[1]);
            if taps == 4 {
                let (h2, h3) = (h(2), h(3));
                lo = vmlal_n_s16(lo, vget_low_s16(h2), hy[2]);
                hi = vmlal_high_n_s16(hi, h2, hy[2]);
                lo = vmlal_n_s16(lo, vget_low_s16(h3), hy[3]);
                hi = vmlal_high_n_s16(hi, h3, hy[3]);
            }
            // ((x + 2^11) >> 12) + 512, then saturated to 0..=65535 and clamped at 1023.
            let fix = |v| vaddq_s32(vrshrq_n_s32::<12>(v), vdupq_n_s32(512));
            let s = vcombine_u16(vqmovun_s32(fix(lo)), vqmovun_s32(fix(hi)));
            vst1q_u16(out.as_mut_ptr().add(k), vminq_u16(s, vdupq_n_u16(1023)));
        }
        k += 8;
    }
}

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
unsafe fn blend_row(s: &[&[u16]; 4], cs: u32, v: u32, shift: u32, out: &mut [u8]) {
    let len = out.len();
    debug_assert!(len.is_multiple_of(8) && cs <= 32 && s.iter().all(|r| r.len() >= len));
    const LANES: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    // SAFETY: NEON is part of the aarch64 baseline.
    let neg_shift = unsafe { vdupq_n_s32(-(shift as i32)) };
    let mut k = 0;
    while k < len {
        // SAFETY: every row holds `len` values and the output `len` bytes.
        unsafe {
            let u = vaddq_u16(vld1q_u16(LANES.as_ptr()), vdupq_n_u16(k as u16));
            let wl = vsubq_u16(vdupq_n_u16(cs as u16), u);
            let load = |j: usize| vld1q_u16(s[j].as_ptr().add(k));
            let top = vmlaq_u16(vmulq_u16(wl, load(0)), u, load(1));
            let bottom = vmlaq_u16(vmulq_u16(wl, load(2)), u, load(3));
            let (a, b) = ((cs - v) as u16, v as u16);
            let lo = vmlal_n_u16(vmull_n_u16(vget_low_u16(top), a), vget_low_u16(bottom), b);
            let hi = vmlal_high_n_u16(vmull_high_n_u16(top, a), bottom, b);
            // Rounding shift right by `shift`, then saturated to 8 bits.
            let fix = |x: uint32x4_t| vqmovn_u32(vrshlq_u32(x, neg_shift));
            let words = vcombine_u16(fix(lo), fix(hi));
            vst1_u8(out.as_mut_ptr().add(k), vqmovn_u16(words));
        }
        k += 8;
    }
}
