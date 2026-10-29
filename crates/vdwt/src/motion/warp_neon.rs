//! AArch64 Advanced SIMD motion-compensation kernel (the production target), bit-identical to
//! the portable span sampler in [`warp`](super::warp). It mirrors the x86 kernel step for step.
//!
//! Groups of 8 pixels, one 16-bit lane each (see "SIMD kernels" in [`warp`](super::warp)). The
//! group's samples come from one 16-byte load per reference row and `tbl`; single-lane `ld4`
//! loads each lane's taps from the table already deinterleaved.

use std::arch::aarch64::*;

use super::warp::{PHASE_BITS, PHASES, Span, SpanKernel, Taps, span_inside};
use crate::Plane;

/// The span sampler for `taps`-tap filters. Advanced SIMD is part of the AArch64 baseline.
pub(super) fn kernel(taps: usize) -> Option<SpanKernel> {
    match taps {
        2 => Some(span::<2>),
        4 => Some(span::<4>),
        _ => None,
    }
}

fn span<const T: usize>(reference: &Plane<u8>, span: &Span, taps: &Taps, dst: &mut [u8]) {
    // SAFETY: Advanced SIMD is always present on AArch64.
    let done = unsafe { groups::<T>(reference, span, taps, dst) };
    span_inside::<T>(reference, span, done..dst.len(), taps, dst);
}

/// Four registers of taps, lane `v` of register `i` holding tap `i` of lane `v`'s phase.
#[inline]
#[target_feature(enable = "neon")]
fn lane_taps(taps: &Taps, positions: &[i32; 8]) -> [int16x8_t; 4] {
    let tap = |v: usize| taps[(positions[v] & (PHASES as i32 - 1)) as usize].as_ptr();
    let z = vdupq_n_s16(0);
    let mut r = int16x8x4_t(z, z, z, z);
    // SAFETY: every table entry holds the 4 values `ld4` reads.
    unsafe {
        r = vld4q_lane_s16::<0>(tap(0), r);
        r = vld4q_lane_s16::<1>(tap(1), r);
        r = vld4q_lane_s16::<2>(tap(2), r);
        r = vld4q_lane_s16::<3>(tap(3), r);
        r = vld4q_lane_s16::<4>(tap(4), r);
        r = vld4q_lane_s16::<5>(tap(5), r);
        r = vld4q_lane_s16::<6>(tap(6), r);
        r = vld4q_lane_s16::<7>(tap(7), r);
    }
    [r.0, r.1, r.2, r.3]
}

/// Integer positions of 8 lanes minus `base`, as 16-bit values.
#[inline]
#[target_feature(enable = "neon")]
fn offsets(p: [int32x4_t; 2], base: i32) -> int16x8_t {
    let b = vdupq_n_s32(base);
    vcombine_s16(
        vmovn_s32(vsubq_s32(vshrq_n_s32::<{ PHASE_BITS as i32 }>(p[0]), b)),
        vmovn_s32(vsubq_s32(vshrq_n_s32::<{ PHASE_BITS as i32 }>(p[1]), b)),
    )
}

/// `(Σ_j hy_j·t_j + 2^13) >> 14` per lane, over rows `t[0..T]`.
#[inline]
#[target_feature(enable = "neon")]
fn vertical<const T: usize>(t: &[int16x8_t], hy: &[int16x8_t; 4]) -> int16x8_t {
    let mut lo = vmull_s16(vget_low_s16(t[0]), vget_low_s16(hy[0]));
    let mut hi = vmull_high_s16(t[0], hy[0]);
    for j in 1..T {
        lo = vmlal_s16(lo, vget_low_s16(t[j]), vget_low_s16(hy[j]));
        hi = vmlal_high_s16(hi, t[j], hy[j]);
    }
    // Rounding shift: (x + 2^13) >> 14. The results are within ±300, so narrowing keeps them.
    vcombine_s16(vrshrn_n_s32::<14>(lo), vrshrn_n_s32::<14>(hi))
}

/// Samples every whole group of 8 it can and returns how many pixels that covers.
#[target_feature(enable = "neon")]
fn groups<const T: usize>(reference: &Plane<u8>, s: &Span, taps: &Taps, dst: &mut [u8]) -> usize {
    let n = s.len.min(dst.len()) / 8 * 8;
    let (data, stride) = (reference.data(), reference.stride());
    let o = (T / 2 - 1) as i32;
    let centre = vdup_n_u8(128);
    let grey = vdupq_n_s16(128);
    let one = vdupq_n_s16(1);
    // Positions as in `Span::px` and `Span::py`, 8 lanes (two registers) at a time.
    let lanes = [0, 4].map(|first: i32| {
        let l = [first, first + 1, first + 2, first + 3];
        // SAFETY: `l` holds 4 values.
        unsafe { vld1q_s32(l.as_ptr()) }
    });
    let shift = vdupq_n_s32(-(s.k as i32));
    let mut mx = lanes.map(|l| vmlaq_n_s32(vdupq_n_s32(s.bx), l, s.dx));
    let mut my = lanes.map(|l| vmlaq_n_s32(vdupq_n_s32(s.by), l, s.dy));
    let mut columns =
        lanes.map(|l| vshlq_n_s32::<{ PHASE_BITS as i32 }>(vaddq_s32(l, vdupq_n_s32(s.x))));
    let row = vdupq_n_s32(s.y << PHASE_BITS);
    let (step_x, step_y) = (vdupq_n_s32(8 * s.dx), vdupq_n_s32(8 * s.dy));
    let step_columns = vdupq_n_s32(8 << PHASE_BITS);
    let (mut gx, mut gy) = ([0i32; 8], [0i32; 8]);
    for g in (0..n).step_by(8) {
        let px: [int32x4_t; 2] =
            std::array::from_fn(|h| vaddq_s32(columns[h], vshlq_s32(mx[h], shift)));
        let py: [int32x4_t; 2] = std::array::from_fn(|h| vaddq_s32(row, vshlq_s32(my[h], shift)));
        for h in 0..2 {
            mx[h] = vaddq_s32(mx[h], step_x);
            my[h] = vaddq_s32(my[h], step_y);
            columns[h] = vaddq_s32(columns[h], step_columns);
        }
        // SAFETY: both arrays hold 8 values.
        unsafe {
            vst1q_s32(gx.as_mut_ptr(), px[0]);
            vst1q_s32(gx.as_mut_ptr().add(4), px[1]);
            vst1q_s32(gy.as_mut_ptr(), py[0]);
            vst1q_s32(gy.as_mut_ptr().add(4), py[1]);
        }
        // Positions are monotonic along a span, so the group's ends bound it.
        let (xa, xb) = ((gx[0] >> PHASE_BITS) - o, (gx[7] >> PHASE_BITS) - o);
        let (ya, yb) = ((gy[0] >> PHASE_BITS) - o, (gy[7] >> PHASE_BITS) - o);
        let (x0, x1, y0, y1) = (xa.min(xb), xa.max(xb), ya.min(yb), ya.max(yb));
        let straddles = y1 > y0;
        // The caller guarantees every tap inside the plane; the 16-byte loads must also end
        // inside the row's allocation.
        if x1 - x0 + T as i32 > 16 || y1 - y0 > 1 || x0 as usize + 16 > stride {
            span_inside::<T>(reference, s, g..g + 8, taps, dst);
            continue;
        }
        let hx = lane_taps(taps, &gx);
        let hy = lane_taps(taps, &gy);
        let idx = vmovn_u16(vreinterpretq_u16_s16(offsets(px, x0 + o)));
        let idx: [uint8x8_t; T] = std::array::from_fn(|i| vadd_u8(idx, vdup_n_u8(i as u8)));
        // Horizontal pass over T rows, or T + 1 for a group straddling two.
        let mut t = [vdupq_n_s16(0); 5];
        for (r, t) in t.iter_mut().enumerate().take(T + usize::from(straddles)) {
            let at = (y0 as usize + r) * stride + x0 as usize;
            // SAFETY: row y0 + r lies inside the plane and x0 + 16 <= stride (checked above).
            let row = unsafe { vld1q_u8(data[at..at + 16].as_ptr()) };
            for (i, idx) in idx.iter().enumerate() {
                // s − 128 as a signed 16-bit value.
                let s = vreinterpretq_s16_u16(vsubl_u8(vqtbl1_u8(row, *idx), centre));
                *t = vmlaq_s16(*t, s, hx[i]);
            }
        }
        let mut v = vertical::<T>(&t, &hy);
        if straddles {
            // Lanes whose rows start one lower take the alignment shifted by a row.
            let lower = vceqq_s16(offsets(py, y0 + o), one);
            v = vbslq_s16(lower, vertical::<T>(&t[1..], &hy), v);
        }
        let v = vqmovun_s16(vaddq_s16(v, grey));
        // SAFETY: `dst` holds pixels g .. g + 8 (g + 8 <= n <= dst.len()).
        unsafe { vst1_u8(dst[g..g + 8].as_mut_ptr(), v) };
    }
    n
}
