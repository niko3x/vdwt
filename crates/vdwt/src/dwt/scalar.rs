//! Portable row kernels, and the per-sample formulas the SIMD kernels use for their edges and
//! tails.
//!
//! Borders use whole-sample symmetric extension, which for the 5/3 lifting reduces to three
//! mirrors: `x[n] = x[n−2]`, `d[−1] = d[0]` and, for odd `n`, `d[n/2] = d[n/2 − 1]`.

use super::kernels::Kernels;
use crate::Isa;

pub(super) const KERNELS: Kernels = Kernels {
    isa: Isa::Scalar,
    v_predict,
    v_update,
    v_unpredict,
    v_unupdate,
    h_forward,
    h_inverse,
};

/// `⌊(a + b) / 2⌋`, exact.
#[inline(always)]
pub(super) fn half(a: i16, b: i16) -> i16 {
    ((i32::from(a) + i32::from(b)) >> 1) as i16
}

/// `⌊(a + b + 2) / 4⌋`, exact.
#[inline(always)]
pub(super) fn quarter(a: i16, b: i16) -> i16 {
    ((i32::from(a) + i32::from(b) + 2) >> 2) as i16
}

pub(super) fn v_predict(d: &mut [i16], s0: &[i16], s1: &[i16]) {
    debug_assert!(s0.len() == d.len() && s1.len() == d.len());
    for ((d, &a), &b) in d.iter_mut().zip(s0).zip(s1) {
        *d = d.wrapping_sub(half(a, b));
    }
}

pub(super) fn v_update(s: &mut [i16], d0: &[i16], d1: &[i16]) {
    debug_assert!(d0.len() == s.len() && d1.len() == s.len());
    for ((s, &a), &b) in s.iter_mut().zip(d0).zip(d1) {
        *s = s.wrapping_add(quarter(a, b));
    }
}

pub(super) fn v_unpredict(d: &mut [i16], s0: &[i16], s1: &[i16]) {
    debug_assert!(s0.len() == d.len() && s1.len() == d.len());
    for ((d, &a), &b) in d.iter_mut().zip(s0).zip(s1) {
        *d = d.wrapping_add(half(a, b));
    }
}

pub(super) fn v_unupdate(s: &mut [i16], d0: &[i16], d1: &[i16]) {
    debug_assert!(d0.len() == s.len() && d1.len() == s.len());
    for ((s, &a), &b) in s.iter_mut().zip(d0).zip(d1) {
        *s = s.wrapping_sub(quarter(a, b));
    }
}

/// Forward predict for output `j`: `d[j] = x[2j+1] − ⌊(x[2j] + x[2j+2]) / 2⌋`.
#[inline(always)]
pub(super) fn fwd_predict_at(src: &[i16], j: usize) -> i16 {
    let right = if 2 * j + 2 < src.len() {
        src[2 * j + 2]
    } else {
        src[2 * j]
    };
    src[2 * j + 1].wrapping_sub(half(src[2 * j], right))
}

/// Forward update for output `j`: `s[j] = x[2j] + ⌊(d[j−1] + d[j] + 2) / 4⌋`.
#[inline(always)]
pub(super) fn fwd_update_at(src: &[i16], high: &[i16], j: usize) -> i16 {
    let left = high[j.saturating_sub(1)];
    let right = high[j.min(high.len() - 1)];
    src[2 * j].wrapping_add(quarter(left, right))
}

/// Inverse update for even sample `j`: `x[2j] = s[j] − ⌊(d[j−1] + d[j] + 2) / 4⌋`.
#[inline(always)]
pub(super) fn inv_update_at(low: &[i16], high: &[i16], j: usize) -> i16 {
    let left = high[j.saturating_sub(1)];
    let right = high[j.min(high.len() - 1)];
    low[j].wrapping_sub(quarter(left, right))
}

/// Inverse predict for odd sample `j`: `x[2j+1] = d[j] + ⌊(x[2j] + x[2j+2]) / 2⌋`, where `even`
/// holds the already restored even samples.
#[inline(always)]
pub(super) fn inv_predict_at(even: &[i16], high: &[i16], j: usize) -> i16 {
    let right = if j + 1 < even.len() {
        even[j + 1]
    } else {
        even[j]
    };
    high[j].wrapping_add(half(even[j], right))
}

pub(super) fn h_forward(src: &[i16], low: &mut [i16], high: &mut [i16]) {
    let n = src.len();
    debug_assert!(n >= 1 && low.len() == n.div_ceil(2) && high.len() == n / 2);
    if n == 1 {
        low[0] = src[0];
        return;
    }
    for (j, d) in high.iter_mut().enumerate() {
        *d = fwd_predict_at(src, j);
    }
    for (j, s) in low.iter_mut().enumerate() {
        *s = fwd_update_at(src, high, j);
    }
}

pub(super) fn h_inverse(low: &[i16], high: &[i16], dst: &mut [i16], tmp: &mut [i16]) {
    let n = dst.len();
    let (ns, nd) = (low.len(), high.len());
    debug_assert!(n >= 1 && ns == n.div_ceil(2) && nd == n / 2 && tmp.len() >= ns);
    if n == 1 {
        dst[0] = low[0];
        return;
    }
    let even = &mut tmp[..ns];
    for (j, e) in even.iter_mut().enumerate() {
        *e = inv_update_at(low, high, j);
    }
    for j in 0..nd {
        dst[2 * j] = even[j];
        dst[2 * j + 1] = inv_predict_at(even, high, j);
    }
    if ns > nd {
        dst[n - 1] = even[ns - 1];
    }
}
