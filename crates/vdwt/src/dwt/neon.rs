//! NEON row kernels for AArch64 (the Cortex-A55 target): 8 lanes of `i16`, bit-identical to
//! `scalar`.
//!
//! The 5/3 lifting maps directly onto NEON without widening:
//! - `⌊(a+b)/2⌋` is one `SHADD` (signed halving add, exact for all inputs);
//! - `⌊(a+b+2)/4⌋` is `SRSHR(SHADD(a, b), 1)`;
//! - `LD2`/`ST2` deinterleave and interleave even/odd samples for free.
//!
//! The A55 is in-order, so the vertical kernels process two independent vectors per iteration.

use core::arch::aarch64::*;

use super::kernels::Kernels;
use super::scalar;
use crate::Isa;

/// NEON is part of the AArch64 baseline, so this table is always usable where it compiles. The
/// safe wrappers below rely on that.
pub(super) const KERNELS: Kernels = Kernels {
    isa: Isa::Neon,
    v_predict,
    v_update,
    v_unpredict,
    v_unupdate,
    h_forward,
    h_inverse,
};

/// `⌊(a + b) / 2⌋` per lane.
#[target_feature(enable = "neon")]
#[inline]
fn half(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vhaddq_s16(a, b)
}

/// `⌊(a + b + 2) / 4⌋` per lane.
#[target_feature(enable = "neon")]
#[inline]
fn quarter(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vrshrq_n_s16::<1>(vhaddq_s16(a, b))
}

/// `x[i] = op(x[i], term(a[i], b[i]))`, 16 lanes per iteration in two independent chains, as a
/// safe wrapper plus its NEON body.
macro_rules! vertical_kernel {
    ($name:ident, $body:ident, $op:ident, $term:ident, $tail:path) => {
        fn $name(x: &mut [i16], a: &[i16], b: &[i16]) {
            // SAFETY: NEON is mandatory on AArch64, and this module only builds with it enabled.
            unsafe { $body(x, a, b) }
        }

        #[target_feature(enable = "neon")]
        fn $body(x: &mut [i16], a: &[i16], b: &[i16]) {
            let n = x.len();
            assert!(a.len() == n && b.len() == n);
            let mut i = 0;
            while i + 16 <= n {
                // SAFETY: i + 16 <= n, and all three rows hold n samples.
                unsafe {
                    let (p, pa, pb) = (x.as_mut_ptr().add(i), a.as_ptr().add(i), b.as_ptr().add(i));
                    let t0 = $term(vld1q_s16(pa), vld1q_s16(pb));
                    let t1 = $term(vld1q_s16(pa.add(8)), vld1q_s16(pb.add(8)));
                    let r0 = $op(vld1q_s16(p), t0);
                    let r1 = $op(vld1q_s16(p.add(8)), t1);
                    vst1q_s16(p, r0);
                    vst1q_s16(p.add(8), r1);
                }
                i += 16;
            }
            if i + 8 <= n {
                // SAFETY: i + 8 <= n.
                unsafe {
                    let p = x.as_mut_ptr().add(i);
                    let t = $term(vld1q_s16(a.as_ptr().add(i)), vld1q_s16(b.as_ptr().add(i)));
                    vst1q_s16(p, $op(vld1q_s16(p), t));
                }
                i += 8;
            }
            $tail(&mut x[i..], &a[i..], &b[i..]);
        }
    };
}

vertical_kernel!(
    v_predict,
    v_predict_neon,
    vsubq_s16,
    half,
    scalar::v_predict
);
vertical_kernel!(
    v_update,
    v_update_neon,
    vaddq_s16,
    quarter,
    scalar::v_update
);
vertical_kernel!(
    v_unpredict,
    v_unpredict_neon,
    vaddq_s16,
    half,
    scalar::v_unpredict
);
vertical_kernel!(
    v_unupdate,
    v_unupdate_neon,
    vsubq_s16,
    quarter,
    scalar::v_unupdate
);

fn h_forward(src: &[i16], low: &mut [i16], high: &mut [i16]) {
    // SAFETY: NEON is mandatory on AArch64, and this module only builds with it enabled.
    unsafe { h_forward_neon(src, low, high) }
}

#[target_feature(enable = "neon")]
fn h_forward_neon(src: &[i16], low: &mut [i16], high: &mut [i16]) {
    let n = src.len();
    let (ns, nd) = (n.div_ceil(2), n / 2);
    assert!(n >= 1 && low.len() == ns && high.len() == nd);
    if n == 1 {
        low[0] = src[0];
        return;
    }
    // Predict, 8 outputs per step from src[2j .. 2j+18).
    let mut j = 0;
    while 2 * j + 18 <= n {
        // SAFETY: 2j + 18 <= n bounds both 16-sample loads; it implies j + 8 <= nd for the store.
        unsafe {
            let p = src.as_ptr().add(2 * j);
            let pair = vld2q_s16(p);
            let next = vld2q_s16(p.add(2));
            vst1q_s16(
                high.as_mut_ptr().add(j),
                vsubq_s16(pair.1, half(pair.0, next.0)),
            );
        }
        j += 8;
    }
    for (j, d) in high.iter_mut().enumerate().skip(j) {
        *d = scalar::fwd_predict_at(src, j);
    }
    // Update. s[0] needs the mirrored d[−1], so the vector loop starts at 1.
    low[0] = scalar::fwd_update_at(src, high, 0);
    let mut j = 1;
    while j + 8 <= nd && 2 * j + 16 <= n {
        // SAFETY: 2j + 16 <= n bounds the load; j + 8 <= nd <= ns bounds high[j−1 .. j+8) and
        // the store.
        unsafe {
            let even = vld2q_s16(src.as_ptr().add(2 * j)).0;
            let t = quarter(
                vld1q_s16(high.as_ptr().add(j - 1)),
                vld1q_s16(high.as_ptr().add(j)),
            );
            vst1q_s16(low.as_mut_ptr().add(j), vaddq_s16(even, t));
        }
        j += 8;
    }
    for (j, s) in low.iter_mut().enumerate().skip(j) {
        *s = scalar::fwd_update_at(src, high, j);
    }
}

fn h_inverse(low: &[i16], high: &[i16], dst: &mut [i16], tmp: &mut [i16]) {
    // SAFETY: NEON is mandatory on AArch64, and this module only builds with it enabled.
    unsafe { h_inverse_neon(low, high, dst, tmp) }
}

#[target_feature(enable = "neon")]
fn h_inverse_neon(low: &[i16], high: &[i16], dst: &mut [i16], tmp: &mut [i16]) {
    let n = dst.len();
    let (ns, nd) = (n.div_ceil(2), n / 2);
    assert!(n >= 1 && low.len() == ns && high.len() == nd && tmp.len() >= ns);
    if n == 1 {
        dst[0] = low[0];
        return;
    }
    // Undo the update: restored even samples go to `even`.
    let even = &mut tmp[..ns];
    even[0] = scalar::inv_update_at(low, high, 0);
    let mut j = 1;
    while j + 8 <= nd {
        // SAFETY: j + 8 <= nd <= ns bounds high[j−1 .. j+8), low and even.
        unsafe {
            let t = quarter(
                vld1q_s16(high.as_ptr().add(j - 1)),
                vld1q_s16(high.as_ptr().add(j)),
            );
            vst1q_s16(
                even.as_mut_ptr().add(j),
                vsubq_s16(vld1q_s16(low.as_ptr().add(j)), t),
            );
        }
        j += 8;
    }
    for (j, e) in even.iter_mut().enumerate().skip(j) {
        *e = scalar::inv_update_at(low, high, j);
    }
    // Undo the predict and interleave.
    let mut j = 0;
    while j + 9 <= ns && j + 8 <= nd {
        // SAFETY: even[j .. j+9) and high[j .. j+8) are in bounds, and dst[2j .. 2j+16) lies
        // within 2·nd <= n.
        unsafe {
            let e = vld1q_s16(even.as_ptr().add(j));
            let odd = vaddq_s16(
                vld1q_s16(high.as_ptr().add(j)),
                half(e, vld1q_s16(even.as_ptr().add(j + 1))),
            );
            vst2q_s16(dst.as_mut_ptr().add(2 * j), int16x8x2_t(e, odd));
        }
        j += 8;
    }
    for j in j..nd {
        dst[2 * j] = even[j];
        dst[2 * j + 1] = scalar::inv_predict_at(even, high, j);
    }
    if ns > nd {
        dst[n - 1] = even[ns - 1];
    }
}

/// 9/7 lifting step (`d97`): `x[i] ±= ⌊(c·(a[i] + b[i]) + 2^13) / 2^14⌋`, wrapping to 16 bits,
/// 8 lanes at a time with 32-bit products (`SMULL`/`SMLAL`, then a rounding shift).
pub(super) fn lift97(x: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    // SAFETY: NEON is part of the AArch64 baseline.
    unsafe {
        if sub {
            lift97_neon::<true>(x, a, b, c)
        } else {
            lift97_neon::<false>(x, a, b, c)
        }
    }
}

/// One loop per direction: a runtime choice costs a select in every iteration.
#[target_feature(enable = "neon")]
fn lift97_neon<const SUB: bool>(x: &mut [i16], a: &[i16], b: &[i16], c: i32) {
    let sub = SUB;
    let n = x.len();
    assert!(a.len() >= n && b.len() >= n);
    // The 9/7's coefficients fit 16 bits.
    let c = c as i16;
    // `RSHRN` rounds, shifts and keeps the low 16 bits: the wrap of the scalar code.
    let term = |va: int16x8_t, vb: int16x8_t| {
        let lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(va), c), vget_low_s16(vb), c);
        let hi = vmlal_high_n_s16(vmull_high_n_s16(va, c), vb, c);
        vcombine_s16(vrshrn_n_s32::<14>(lo), vrshrn_n_s32::<14>(hi))
    };
    let apply = |v: int16x8_t, t: int16x8_t| {
        if sub {
            vsubq_s16(v, t)
        } else {
            vaddq_s16(v, t)
        }
    };
    let (xp, ap, bp) = (x.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // Four independent vectors per iteration, every load issued before the first use: the
    // in-order A55 stalls on a load's result otherwise.
    while i + 32 <= n {
        // SAFETY: i + 32 <= n, and all three rows hold at least n samples.
        unsafe {
            let va = [0, 8, 16, 24].map(|k| vld1q_s16(ap.add(i + k)));
            let vb = [0, 8, 16, 24].map(|k| vld1q_s16(bp.add(i + k)));
            let vx = [0, 8, 16, 24].map(|k| vld1q_s16(xp.add(i + k)));
            let t: [int16x8_t; 4] = std::array::from_fn(|k| term(va[k], vb[k]));
            for k in 0..4 {
                vst1q_s16(xp.add(i + 8 * k), apply(vx[k], t[k]));
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: i + 8 <= n, as above.
        unsafe {
            let t = term(vld1q_s16(ap.add(i)), vld1q_s16(bp.add(i)));
            vst1q_s16(xp.add(i), apply(vld1q_s16(xp.add(i)), t));
        }
        i += 8;
    }
    super::d97::lift_tail(&mut x[i..], &a[i..], &b[i..], c.into(), sub);
}

/// The rt codec's fast lifting: `x[i] ±= g(c, a[i], b[i])` (see `d97::FAST_STEPS`), 8 lanes
/// per step with `SQRDMULH`.
pub(super) fn lift97_fast(x: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    // SAFETY: NEON is part of the AArch64 baseline.
    unsafe {
        match (super::d97::fast_parts(c).0, sub) {
            (false, false) => lift97_fast_neon::<false, false>(x, a, b, c),
            (false, true) => lift97_fast_neon::<false, true>(x, a, b, c),
            (true, false) => lift97_fast_neon::<true, false>(x, a, b, c),
            (true, true) => lift97_fast_neon::<true, true>(x, a, b, c),
        }
    }
}

#[target_feature(enable = "neon")]
fn lift97_fast_neon<const WHOLE: bool, const SUB: bool>(
    x: &mut [i16],
    a: &[i16],
    b: &[i16],
    c: i32,
) {
    let n = x.len();
    assert!(a.len() >= n && b.len() >= n);
    let frac = super::d97::fast_parts(c).1 as i16;
    let term = |va: int16x8_t, vb: int16x8_t| {
        let s = vqaddq_s16(va, vb);
        let t = vqrdmulhq_n_s16(s, frac);
        if WHOLE { vsubq_s16(t, s) } else { t }
    };
    let apply = |v: int16x8_t, t: int16x8_t| {
        if SUB {
            vsubq_s16(v, t)
        } else {
            vaddq_s16(v, t)
        }
    };
    let (xp, ap, bp) = (x.as_mut_ptr(), a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // Four independent vectors per iteration, every load issued before the first use: the
    // in-order A55 stalls on a load's result otherwise.
    while i + 32 <= n {
        // SAFETY: i + 32 <= n, and all three rows hold at least n samples.
        unsafe {
            let va = [0, 8, 16, 24].map(|k| vld1q_s16(ap.add(i + k)));
            let vb = [0, 8, 16, 24].map(|k| vld1q_s16(bp.add(i + k)));
            let vx = [0, 8, 16, 24].map(|k| vld1q_s16(xp.add(i + k)));
            let t: [int16x8_t; 4] = std::array::from_fn(|k| term(va[k], vb[k]));
            for k in 0..4 {
                vst1q_s16(xp.add(i + 8 * k), apply(vx[k], t[k]));
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: i + 8 <= n, as above.
        unsafe {
            let t = term(vld1q_s16(ap.add(i)), vld1q_s16(bp.add(i)));
            vst1q_s16(xp.add(i), apply(vld1q_s16(xp.add(i)), t));
        }
        i += 8;
    }
    super::d97::lift_fast_portable(&mut x[i..], &a[i..], &b[i..], c, SUB);
}

/// Splits `src` into its even samples (`even`) and odd ones (`odd`).
pub(super) fn split(src: &[i16], even: &mut [i16], odd: &mut [i16]) {
    let n = src.len();
    assert!(even.len() == n.div_ceil(2) && odd.len() == n / 2);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: 16 samples from `i` are in `src`, 8 from `i / 2` in both halves.
        unsafe {
            let v = vld2q_s16(src.as_ptr().add(i));
            vst1q_s16(even.as_mut_ptr().add(i / 2), v.0);
            vst1q_s16(odd.as_mut_ptr().add(i / 2), v.1);
        }
        i += 16;
    }
    for (o, v) in even[i / 2..].iter_mut().zip(src[i..].iter().step_by(2)) {
        *o = *v;
    }
    for (o, v) in odd[i / 2..]
        .iter_mut()
        .zip(src[i..].iter().skip(1).step_by(2))
    {
        *o = *v;
    }
}

/// The inverse of [`split`].
pub(super) fn merge(even: &[i16], odd: &[i16], dst: &mut [i16]) {
    let n = dst.len();
    assert!(even.len() == n.div_ceil(2) && odd.len() == n / 2);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: as in `split`.
        unsafe {
            let v = int16x8x2_t(
                vld1q_s16(even.as_ptr().add(i / 2)),
                vld1q_s16(odd.as_ptr().add(i / 2)),
            );
            vst2q_s16(dst.as_mut_ptr().add(i), v);
        }
        i += 16;
    }
    for (o, v) in dst[i..].iter_mut().step_by(2).zip(&even[i / 2..]) {
        *o = *v;
    }
    for (o, v) in dst[i..].iter_mut().skip(1).step_by(2).zip(&odd[i / 2..]) {
        *o = *v;
    }
}
