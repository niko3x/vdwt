//! AVX2 row kernels for x86-64 development machines: 16 lanes of `i16`, bit-identical to
//! `scalar`.
//!
//! x86 has no signed halving add, so `⌊(a+b)/2⌋` is computed overflow-free as
//! `(a & b) + ((a ^ b) >> 1)`.

use std::arch::x86_64::*;

use super::kernels::Kernels;
use super::scalar;
use crate::Isa;

/// The AVX2 kernel table, or `None` when the CPU lacks AVX2.
///
/// Every wrapper below assumes AVX2 is present. This function is the only way to reach them,
/// and it checks.
pub(super) fn kernels() -> Option<Kernels> {
    std::is_x86_feature_detected!("avx2").then_some(Kernels {
        isa: Isa::Avx2,
        v_predict,
        v_update,
        v_unpredict,
        v_unupdate,
        h_forward,
        h_inverse,
    })
}

/// `⌊(a + b) / 2⌋` per lane, without overflow.
#[target_feature(enable = "avx2")]
#[inline]
fn half(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi16(
        _mm256_and_si256(a, b),
        _mm256_srai_epi16::<1>(_mm256_xor_si256(a, b)),
    )
}

/// `⌊(a + b + 2) / 4⌋` per lane: `h = ⌊(a + b) / 2⌋`, then `⌊(h + 1) / 2⌋ = (h >> 1) + (h & 1)`.
#[target_feature(enable = "avx2")]
#[inline]
fn quarter(a: __m256i, b: __m256i) -> __m256i {
    let h = half(a, b);
    _mm256_add_epi16(
        _mm256_srai_epi16::<1>(h),
        _mm256_and_si256(h, _mm256_set1_epi16(1)),
    )
}

/// # Safety
/// 16 samples must be readable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load(p: *const i16) -> __m256i {
    // SAFETY: guaranteed by the caller.
    unsafe { _mm256_loadu_si256(p.cast()) }
}

/// # Safety
/// 16 samples must be writable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store(p: *mut i16, v: __m256i) {
    // SAFETY: guaranteed by the caller.
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

/// Loads 32 samples and splits them into 16 even and 16 odd ones.
///
/// # Safety
/// 32 samples must be readable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_deinterleave(p: *const i16) -> (__m256i, __m256i) {
    // SAFETY: guaranteed by the caller.
    let (a, b) = unsafe { (load(p), load(p.add(16))) };
    // Sign-extend the low (even) and high (odd) half of each 32-bit pair, then narrow back. The
    // in-lane pack leaves 64-bit groups in order 0,2,1,3, so a cross-lane permute fixes them.
    let even_a = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(a));
    let even_b = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(b));
    let odd_a = _mm256_srai_epi32::<16>(a);
    let odd_b = _mm256_srai_epi32::<16>(b);
    (
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(even_a, even_b)),
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(odd_a, odd_b)),
    )
}

/// Stores `even[i], odd[i]` pairs: 32 samples.
///
/// # Safety
/// 32 samples must be writable at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_interleave(p: *mut i16, even: __m256i, odd: __m256i) {
    let lo = _mm256_unpacklo_epi16(even, odd); // pairs 0..4 | 8..12
    let hi = _mm256_unpackhi_epi16(even, odd); // pairs 4..8 | 12..16
    // SAFETY: guaranteed by the caller.
    unsafe {
        store(p, _mm256_permute2x128_si256::<0x20>(lo, hi));
        store(p.add(16), _mm256_permute2x128_si256::<0x31>(lo, hi));
    }
}

/// `x[i] = op(x[i], term(a[i], b[i]))`, as a safe wrapper plus its AVX2 body.
macro_rules! vertical_kernel {
    ($name:ident, $body:ident, $op:ident, $term:ident, $tail:path) => {
        fn $name(x: &mut [i16], a: &[i16], b: &[i16]) {
            // SAFETY: only reachable through `kernels()`, which checked for AVX2.
            unsafe { $body(x, a, b) }
        }

        #[target_feature(enable = "avx2")]
        fn $body(x: &mut [i16], a: &[i16], b: &[i16]) {
            let n = x.len();
            assert!(a.len() == n && b.len() == n);
            let mut i = 0;
            while i + 16 <= n {
                // SAFETY: i + 16 <= n, and all three rows hold n samples.
                unsafe {
                    let p = x.as_mut_ptr().add(i);
                    let t = $term(load(a.as_ptr().add(i)), load(b.as_ptr().add(i)));
                    store(p, $op(load(p), t));
                }
                i += 16;
            }
            $tail(&mut x[i..], &a[i..], &b[i..]);
        }
    };
}

vertical_kernel!(
    v_predict,
    v_predict_avx2,
    _mm256_sub_epi16,
    half,
    scalar::v_predict
);
vertical_kernel!(
    v_update,
    v_update_avx2,
    _mm256_add_epi16,
    quarter,
    scalar::v_update
);
vertical_kernel!(
    v_unpredict,
    v_unpredict_avx2,
    _mm256_add_epi16,
    half,
    scalar::v_unpredict
);
vertical_kernel!(
    v_unupdate,
    v_unupdate_avx2,
    _mm256_sub_epi16,
    quarter,
    scalar::v_unupdate
);

fn h_forward(src: &[i16], low: &mut [i16], high: &mut [i16]) {
    // SAFETY: only reachable through `kernels()`, which checked for AVX2.
    unsafe { h_forward_avx2(src, low, high) }
}

#[target_feature(enable = "avx2")]
fn h_forward_avx2(src: &[i16], low: &mut [i16], high: &mut [i16]) {
    let n = src.len();
    let (ns, nd) = (n.div_ceil(2), n / 2);
    assert!(n >= 1 && low.len() == ns && high.len() == nd);
    if n == 1 {
        low[0] = src[0];
        return;
    }
    // Predict, 16 outputs per step from src[2j .. 2j+34).
    let mut j = 0;
    while 2 * j + 34 <= n {
        // SAFETY: 2j + 34 <= n bounds both loads; it implies j + 16 <= nd for the store.
        unsafe {
            let p = src.as_ptr().add(2 * j);
            let (even, odd) = load_deinterleave(p);
            let (next_even, _) = load_deinterleave(p.add(2));
            store(
                high.as_mut_ptr().add(j),
                _mm256_sub_epi16(odd, half(even, next_even)),
            );
        }
        j += 16;
    }
    for (j, d) in high.iter_mut().enumerate().skip(j) {
        *d = scalar::fwd_predict_at(src, j);
    }
    // Update. s[0] needs the mirrored d[−1], so the vector loop starts at 1.
    low[0] = scalar::fwd_update_at(src, high, 0);
    let mut j = 1;
    while j + 16 <= nd && 2 * j + 32 <= n {
        // SAFETY: 2j + 32 <= n bounds the load; j + 16 <= nd <= ns bounds high[j−1 .. j+16) and
        // the store.
        unsafe {
            let (even, _) = load_deinterleave(src.as_ptr().add(2 * j));
            let t = quarter(load(high.as_ptr().add(j - 1)), load(high.as_ptr().add(j)));
            store(low.as_mut_ptr().add(j), _mm256_add_epi16(even, t));
        }
        j += 16;
    }
    for (j, s) in low.iter_mut().enumerate().skip(j) {
        *s = scalar::fwd_update_at(src, high, j);
    }
}

fn h_inverse(low: &[i16], high: &[i16], dst: &mut [i16], tmp: &mut [i16]) {
    // SAFETY: only reachable through `kernels()`, which checked for AVX2.
    unsafe { h_inverse_avx2(low, high, dst, tmp) }
}

#[target_feature(enable = "avx2")]
fn h_inverse_avx2(low: &[i16], high: &[i16], dst: &mut [i16], tmp: &mut [i16]) {
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
    while j + 16 <= nd {
        // SAFETY: j + 16 <= nd <= ns bounds high[j−1 .. j+16), low and even.
        unsafe {
            let t = quarter(load(high.as_ptr().add(j - 1)), load(high.as_ptr().add(j)));
            store(
                even.as_mut_ptr().add(j),
                _mm256_sub_epi16(load(low.as_ptr().add(j)), t),
            );
        }
        j += 16;
    }
    for (j, e) in even.iter_mut().enumerate().skip(j) {
        *e = scalar::inv_update_at(low, high, j);
    }
    // Undo the predict and interleave.
    let mut j = 0;
    while j + 17 <= ns && j + 16 <= nd {
        // SAFETY: even[j .. j+17) and high[j .. j+16) are in bounds, and dst[2j .. 2j+32) lies
        // within 2·nd <= n.
        unsafe {
            let e = load(even.as_ptr().add(j));
            let odd = _mm256_add_epi16(
                load(high.as_ptr().add(j)),
                half(e, load(even.as_ptr().add(j + 1))),
            );
            store_interleave(dst.as_mut_ptr().add(2 * j), e, odd);
        }
        j += 16;
    }
    for j in j..nd {
        dst[2 * j] = even[j];
        dst[2 * j + 1] = scalar::inv_predict_at(even, high, j);
    }
    if ns > nd {
        dst[n - 1] = even[ns - 1];
    }
}

/// 9/7 lifting: `x[i] ±= f(c, a[i], b[i])` (see `d97`), 16 lanes per step. `madd` forms
/// `c·a + c·b` exactly in 32 bits; the result wraps to 16 bits like the scalar cast.
pub(super) fn lift97(x: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    // SAFETY: only reachable through `d97`'s AVX2 table, which checked for AVX2.
    unsafe { lift97_avx2(x, a, b, c, sub) }
}

#[target_feature(enable = "avx2")]
fn lift97_avx2(x: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    let n = x.len();
    assert!(a.len() >= n && b.len() >= n);
    let pair = _mm256_set1_epi32(((c as u32 & 0xFFFF) | ((c as u32) << 16)) as i32);
    let round = _mm256_set1_epi32(1 << 13);
    let term = |va: __m256i, vb: __m256i| {
        let f = |v: __m256i| {
            let v = _mm256_srai_epi32::<14>(_mm256_add_epi32(_mm256_madd_epi16(v, pair), round));
            _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(v))
        };
        let lo = f(_mm256_unpacklo_epi16(va, vb));
        let hi = f(_mm256_unpackhi_epi16(va, vb));
        _mm256_packs_epi32(lo, hi)
    };
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: i + 16 <= n, and all three rows hold at least n samples.
        unsafe {
            let p = x.as_mut_ptr().add(i);
            let t = term(load(a.as_ptr().add(i)), load(b.as_ptr().add(i)));
            let v = load(p);
            store(
                p,
                if sub {
                    _mm256_sub_epi16(v, t)
                } else {
                    _mm256_add_epi16(v, t)
                },
            );
        }
        i += 16;
    }
    super::d97::lift_tail(&mut x[i..], &a[i..], &b[i..], c, sub);
}

/// The rt codec's fast lifting: `x[i] ±= g(c, a[i], b[i])` (see `d97::FAST_STEPS`), 16 lanes
/// per step with `PMULHRSW`.
pub(super) fn lift97_fast(x: &mut [i16], a: &[i16], b: &[i16], c: i32, sub: bool) {
    // SAFETY: only reachable through `d97`'s AVX2 table, which checked for AVX2.
    unsafe {
        match (super::d97::fast_parts(c).0, sub) {
            (false, false) => lift97_fast_avx2::<false, false>(x, a, b, c),
            (false, true) => lift97_fast_avx2::<false, true>(x, a, b, c),
            (true, false) => lift97_fast_avx2::<true, false>(x, a, b, c),
            (true, true) => lift97_fast_avx2::<true, true>(x, a, b, c),
        }
    }
}

#[target_feature(enable = "avx2")]
fn lift97_fast_avx2<const WHOLE: bool, const SUB: bool>(
    x: &mut [i16],
    a: &[i16],
    b: &[i16],
    c: i32,
) {
    let n = x.len();
    assert!(a.len() >= n && b.len() >= n);
    let frac = _mm256_set1_epi16(super::d97::fast_parts(c).1 as i16);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: i + 16 <= n, and all three rows hold at least n samples.
        unsafe {
            let p = x.as_mut_ptr().add(i);
            let s = _mm256_adds_epi16(load(a.as_ptr().add(i)), load(b.as_ptr().add(i)));
            let mut t = _mm256_mulhrs_epi16(s, frac);
            if WHOLE {
                t = _mm256_sub_epi16(t, s);
            }
            let v = load(p);
            store(
                p,
                if SUB {
                    _mm256_sub_epi16(v, t)
                } else {
                    _mm256_add_epi16(v, t)
                },
            );
        }
        i += 16;
    }
    super::d97::lift_fast_portable(&mut x[i..], &a[i..], &b[i..], c, SUB);
}

/// Splits `src` into its even samples (`even`) and odd ones (`odd`).
pub(super) fn split(src: &[i16], even: &mut [i16], odd: &mut [i16]) {
    // SAFETY: as for `lift97`.
    unsafe { split_avx2(src, even, odd) }
}

#[target_feature(enable = "avx2")]
fn split_avx2(src: &[i16], even: &mut [i16], odd: &mut [i16]) {
    let n = src.len();
    assert!(even.len() == n.div_ceil(2) && odd.len() == n / 2);
    let mut i = 0;
    while 2 * i + 32 <= n {
        // SAFETY: 32 samples from 2i are in `src`; 16 from i fit in both halves.
        unsafe {
            let (e, o) = load_deinterleave(src.as_ptr().add(2 * i));
            store(even.as_mut_ptr().add(i), e);
            store(odd.as_mut_ptr().add(i), o);
        }
        i += 16;
    }
    for (j, &v) in src.iter().enumerate().skip(2 * i) {
        if j % 2 == 0 {
            even[j / 2] = v;
        } else {
            odd[j / 2] = v;
        }
    }
}

/// The inverse of [`split`].
pub(super) fn merge(even: &[i16], odd: &[i16], dst: &mut [i16]) {
    // SAFETY: as for `lift97`.
    unsafe { merge_avx2(even, odd, dst) }
}

#[target_feature(enable = "avx2")]
fn merge_avx2(even: &[i16], odd: &[i16], dst: &mut [i16]) {
    let n = dst.len();
    assert!(even.len() == n.div_ceil(2) && odd.len() == n / 2);
    let mut i = 0;
    while 2 * i + 32 <= n {
        // SAFETY: as in `split_avx2`.
        unsafe {
            store_interleave(
                dst.as_mut_ptr().add(2 * i),
                load(even.as_ptr().add(i)),
                load(odd.as_ptr().add(i)),
            );
        }
        i += 16;
    }
    for (j, v) in dst.iter_mut().enumerate().skip(2 * i) {
        *v = if j % 2 == 0 { even[j / 2] } else { odd[j / 2] };
    }
}
