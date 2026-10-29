//! x86-64 motion-compensation kernel for development machines, bit-identical to the portable
//! span sampler in [`warp`](super::warp) and structured like the NEON one, so that the algorithm
//! is tested on x86 too.
//!
//! Groups of 8 pixels, one 16-bit lane each (see "SIMD kernels" in [`warp`](super::warp)). The
//! group's samples come from one 16-byte load per reference row and `pshufb`, never a gather.

use std::arch::x86_64::*;

use super::warp::{PHASE_BITS, PHASES, Span, SpanKernel, Taps, span_inside};
use crate::Plane;

/// The span sampler for `taps`-tap filters, or `None` when the CPU lacks AVX2 (which implies
/// the SSSE3 and SSE4.1 instructions used).
///
/// The wrappers below assume AVX2 is present. This function is the only way to reach them, and
/// it checks.
pub(super) fn kernel(taps: usize) -> Option<SpanKernel> {
    if !std::is_x86_feature_detected!("avx2") {
        return None;
    }
    match taps {
        2 => Some(span::<2>),
        4 => Some(span::<4>),
        _ => None,
    }
}

fn span<const T: usize>(reference: &Plane<u8>, span: &Span, taps: &Taps, dst: &mut [u8]) {
    // SAFETY: only reachable through `kernel()`, which checked for AVX2.
    let done = unsafe { groups::<T>(reference, span, taps, dst) };
    span_inside::<T>(reference, span, done..dst.len(), taps, dst);
}

/// Four registers of taps, lane `v` of register `i` holding tap `i` of lane `v`'s phase.
#[inline]
#[target_feature(enable = "avx2")]
fn lane_taps(taps: &Taps, positions: &[i32; 8]) -> [__m128i; 4] {
    // Each lane's four taps straight from the table (8 bytes), two lanes per register.
    let tap = |v: usize| {
        let t = &taps[(positions[v] & (PHASES as i32 - 1)) as usize];
        // SAFETY: `t` is 8 bytes.
        unsafe { _mm_loadl_epi64((t as *const [i16; 4]).cast()) }
    };
    let [r0, r1, r2, r3] = std::array::from_fn(|k| _mm_unpacklo_epi64(tap(2 * k), tap(2 * k + 1)));
    // An 8×4 transpose of 16-bit values.
    let (t0, t1) = (_mm_unpacklo_epi16(r0, r1), _mm_unpackhi_epi16(r0, r1));
    let (t2, t3) = (_mm_unpacklo_epi16(r2, r3), _mm_unpackhi_epi16(r2, r3));
    let (u0, u1) = (_mm_unpacklo_epi16(t0, t1), _mm_unpackhi_epi16(t0, t1));
    let (u2, u3) = (_mm_unpacklo_epi16(t2, t3), _mm_unpackhi_epi16(t2, t3));
    [
        _mm_unpacklo_epi64(u0, u2),
        _mm_unpackhi_epi64(u0, u2),
        _mm_unpacklo_epi64(u1, u3),
        _mm_unpackhi_epi64(u1, u3),
    ]
}

/// Integer positions of 8 lanes minus `base`, as 16-bit values.
#[inline]
#[target_feature(enable = "avx2")]
fn offsets(p: __m256i, base: i32) -> __m128i {
    let v = _mm256_sub_epi32(
        _mm256_srai_epi32::<{ PHASE_BITS as i32 }>(p),
        _mm256_set1_epi32(base),
    );
    _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
}

/// `(Σ_j hy_j·t_j + 2^13) >> 14` per lane, over rows `t[0..T]`.
#[inline]
#[target_feature(enable = "avx2")]
fn vertical<const T: usize>(t: &[__m128i], pairs: &[[__m128i; 2]; 2]) -> __m128i {
    let (mut lo, mut hi) = (_mm_setzero_si128(), _mm_setzero_si128());
    for (j, [plo, phi]) in pairs.iter().enumerate().take(T / 2) {
        let (a, b) = (t[2 * j], t[2 * j + 1]);
        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *plo));
        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *phi));
    }
    let round = _mm_set1_epi32(1 << 13);
    _mm_packs_epi32(
        _mm_srai_epi32::<14>(_mm_add_epi32(lo, round)),
        _mm_srai_epi32::<14>(_mm_add_epi32(hi, round)),
    )
}

/// Samples every whole group of 8 it can and returns how many pixels that covers.
#[target_feature(enable = "avx2")]
fn groups<const T: usize>(reference: &Plane<u8>, s: &Span, taps: &Taps, dst: &mut [u8]) -> usize {
    let n = s.len.min(dst.len()) / 8 * 8;
    let (data, stride) = (reference.data(), reference.stride());
    let o = (T / 2 - 1) as i32;
    let centre = _mm_set1_epi8(-128);
    let grey = _mm_set1_epi16(128);
    let one = _mm_set1_epi16(1);
    // Positions as in `Span::px` and `Span::py`, 8 lanes at a time.
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let count = _mm_cvtsi32_si128(s.k as i32);
    let mut mx = _mm256_add_epi32(
        _mm256_set1_epi32(s.bx),
        _mm256_mullo_epi32(lane, _mm256_set1_epi32(s.dx)),
    );
    let mut my = _mm256_add_epi32(
        _mm256_set1_epi32(s.by),
        _mm256_mullo_epi32(lane, _mm256_set1_epi32(s.dy)),
    );
    let mut columns =
        _mm256_slli_epi32::<{ PHASE_BITS as i32 }>(_mm256_add_epi32(_mm256_set1_epi32(s.x), lane));
    let row = _mm256_set1_epi32(s.y << PHASE_BITS);
    let (step_x, step_y) = (_mm256_set1_epi32(8 * s.dx), _mm256_set1_epi32(8 * s.dy));
    let step_columns = _mm256_set1_epi32(8 << PHASE_BITS);
    let (mut gx, mut gy) = ([0i32; 8], [0i32; 8]);
    for g in (0..n).step_by(8) {
        let px = _mm256_add_epi32(columns, _mm256_sra_epi32(mx, count));
        let py = _mm256_add_epi32(row, _mm256_sra_epi32(my, count));
        mx = _mm256_add_epi32(mx, step_x);
        my = _mm256_add_epi32(my, step_y);
        columns = _mm256_add_epi32(columns, step_columns);
        // SAFETY: both arrays hold 8 values.
        unsafe {
            _mm256_storeu_si256(gx.as_mut_ptr().cast(), px);
            _mm256_storeu_si256(gy.as_mut_ptr().cast(), py);
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
        let pairs = [
            [
                _mm_unpacklo_epi16(hy[0], hy[1]),
                _mm_unpackhi_epi16(hy[0], hy[1]),
            ],
            [
                _mm_unpacklo_epi16(hy[2], hy[3]),
                _mm_unpackhi_epi16(hy[2], hy[3]),
            ],
        ];
        let idx = _mm_packus_epi16(offsets(px, x0 + o), _mm_setzero_si128());
        let idx: [__m128i; T] = std::array::from_fn(|i| _mm_add_epi8(idx, _mm_set1_epi8(i as i8)));
        // Horizontal pass over T rows, or T + 1 for a group straddling two.
        let mut t = [_mm_setzero_si128(); 5];
        for (r, t) in t.iter_mut().enumerate().take(T + usize::from(straddles)) {
            let at = (y0 as usize + r) * stride + x0 as usize;
            // SAFETY: row y0 + r lies inside the plane and x0 + 16 <= stride (checked above).
            let row = unsafe { _mm_loadu_si128(data[at..at + 16].as_ptr().cast()) };
            // s − 128 as signed bytes.
            let row = _mm_xor_si128(row, centre);
            for (i, idx) in idx.iter().enumerate() {
                let s = _mm_cvtepi8_epi16(_mm_shuffle_epi8(row, *idx));
                *t = _mm_add_epi16(*t, _mm_mullo_epi16(s, hx[i]));
            }
        }
        let mut v = vertical::<T>(&t, &pairs);
        if straddles {
            // Lanes whose rows start one lower take the alignment shifted by a row.
            let lower = _mm_cmpeq_epi16(offsets(py, y0 + o), one);
            v = _mm_blendv_epi8(v, vertical::<T>(&t[1..], &pairs), lower);
        }
        let v = _mm_add_epi16(v, grey);
        // SAFETY: `dst` holds pixels g .. g + 8 (g + 8 <= n <= dst.len()).
        unsafe { _mm_storel_epi64(dst[g..g + 8].as_mut_ptr().cast(), _mm_packus_epi16(v, v)) };
    }
    n
}
