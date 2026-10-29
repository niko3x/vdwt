//! The rt codec's motion compensation (DESIGN.md §10.3): the main codec's overlapped prediction
//! ([`crate::motion::obmc`]) with quarter-pixel translations made of 8-bit rounding averages.
//! Normative, exact integer arithmetic, for a pixel `(u, v)` of a cell of `C = 2^c` pixels:
//!
//! ```text
//! Q(a, b, p) = a, ⌈(a + ⌈(a + b)/2⌉)/2⌉, ⌈(a + b)/2⌉, ⌈(b + ⌈(a + b)/2⌉)/2⌉   for p = 0, 1, 2, 3
//! s_n = Q(H(iy), H(iy + 1), py)                                      H(y) = Q(R(ix, y), R(ix + 1, y), px)
//! top = (C − u)·s_tl + u·s_tr        bottom = (C − u)·s_bl + u·s_br    (each >> e, rounded)
//! out = ((C − v)·top + v·bottom + 2^(k−1)) >> k                   e = max(0, 2c − 8), k = 2c − e
//! ```
//!
//! Node `n`'s vector (luma vectors are in 1/8 luma pixel, chroma ones 1/16 chroma pixel) rounds
//! half up to quarter pixels and splits into the whole part `(ix, iy)` and the phases `px, py`;
//! coordinates clamp to the plane, or above a virtual boundary (gradual refresh). An intra node
//! contributes mid-grey (128). The blend fits 16 bits: the pre-shift `e` keeps `C·top ≤ 65280`.
//!
//! A translation costs one or two halving adds per 16 pixels and direction (`URHADD`, `PAVGB`)
//! instead of widening multiplies: against bilinear taps of 1/16 pixel this costs about X dB.

use std::ops::Range;

use crate::motion::MotionField;
use crate::plane::RowsMut;
use crate::pool::Pool;
use crate::{Frame, Plane};

use super::clean_bound;

/// Motion-compensates `reference` into `pred`: rows inside this frame's clean zone (`clean[1]`
/// node rows) read the reference only above its own clean zone's boundary (`clean[0]` rows),
/// the rest read all of it. With no clean reference rows the zone is intra and reads nothing.
pub(crate) fn predict(
    pool: &Pool,
    reference: &Frame,
    pred: &mut Frame,
    field: &MotionField,
    clean: [usize; 2],
) {
    let (bands, mut tasks) = Bands::new(reference, pred, field, clean);
    pool.for_each_mut(&mut tasks, |_, (c, band)| bands.run(*c, band));
}

/// [`predict`] as tasks, to run alongside other work: every plane's rows of cells.
pub(crate) struct Bands<'a> {
    refs: [&'a Plane<u8>; 3],
    clean: [usize; 3],
    bounds: [usize; 3],
    field: &'a MotionField,
}

impl<'a> Bands<'a> {
    /// The tasks: (plane, its rows of one row of cells).
    #[allow(clippy::type_complexity)]
    pub(crate) fn new(
        reference: &'a Frame,
        pred: &'a mut Frame,
        field: &'a MotionField,
        [ref_clean, clean]: [usize; 2],
    ) -> (Self, Vec<(usize, RowsMut<'a, u8>)>) {
        let spacing = field.grid.spacing();
        let h = reference.height();
        let (ref_bound, bound) = (
            clean_bound(ref_clean, spacing).min(h),
            clean_bound(clean, spacing).min(h),
        );
        let half = |b: usize| [b, b / 2, b / 2];
        let (clean, bounds) = (half(ref_bound), half(bound));
        let mut tasks = Vec::new();
        for (c, out) in pred.planes_mut().into_iter().enumerate() {
            let cell = spacing >> usize::from(c > 0);
            debug_assert!(bounds[c].is_multiple_of(cell));
            tasks.extend(out.bands_mut(cell).into_iter().map(|band| (c, band)));
        }
        let refs = reference.planes();
        let bands = Self {
            refs: [refs[0], refs[1], refs[2]],
            clean,
            bounds,
            field,
        };
        (bands, tasks)
    }

    /// Plane `c`'s row of cells `band`.
    pub(crate) fn run(&self, c: usize, band: &mut RowsMut<'_, u8>) {
        let r = self.refs[c];
        let height = if band.rows().start < self.bounds[c] && self.clean[c] > 0 {
            self.clean[c]
        } else {
            r.height()
        };
        cell_row(r, height, self.field, c > 0, band);
    }
}

/// Reusable buffers of one cell row.
#[derive(Default)]
struct Scratch {
    /// A reference row with clamped columns, for a translation reaching past the plane's sides.
    edge: Vec<u8>,
    /// Two reference rows filtered horizontally.
    h: [Vec<u8>; 2],
    /// Nodes `k` and `k + 1` of the top and bottom node rows, in slots `k mod 2`: each one's
    /// translation over the cells on either side of it, and the first column that covers.
    nodes: [[(Vec<u8>, usize); 2]; 2],
}

/// One row of cells (`out`'s rows) of a plane, from `reference` read above row `height`.
fn cell_row(
    reference: &Plane<u8>,
    height: usize,
    field: &MotionField,
    chroma: bool,
    out: &mut RowsMut<'_, u8>,
) {
    // This thread's buffers, kept from row to row and frame to frame (allocating them per row
    // made the allocator fault pages in every frame).
    SCRATCH.with(|sc| cell_row_in(reference, height, field, chroma, out, &mut sc.borrow_mut()));
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// [`cell_row`] with `sc` for buffers.
fn cell_row_in(
    reference: &Plane<u8>,
    height: usize,
    field: &MotionField,
    chroma: bool,
    out: &mut RowsMut<'_, u8>,
    sc: &mut Scratch,
) {
    let c = field.grid.log2 - u32::from(chroma);
    let cs = 1usize << c;
    let w = reference.width();
    let nx = field.grid.nx;
    let rows = out.rows();
    let j = rows.start >> c;
    let n_rows = rows.len();
    // Node `i` of node row `j + side`, over the cells left and right of it.
    let node = |sc: &mut Scratch, side: usize, i: usize| {
        let n = (j + side) * nx + i;
        let (xa, xb) = ((i * cs).saturating_sub(cs), ((i + 1) * cs).min(w));
        let Scratch { edge, h, nodes } = sc;
        let (s, first) = &mut nodes[side][i % 2];
        *first = xa;
        s.resize((xb - xa) * n_rows, 0);
        if field.intra[n] {
            s.fill(128);
        } else {
            // Luma vectors are in 1/8 pixel, chroma ones (the same numbers) in 1/16.
            let d = field.mv[n].map(|m| if chroma { m } else { m << 1 });
            translate(reference, height, xa, rows.clone(), d, edge, h, s);
        }
    };
    node(sc, 0, 0);
    node(sc, 1, 0);
    let v0 = (rows.start & (cs - 1)) as u16;
    for i in 0..w.div_ceil(cs) {
        node(sc, 0, i + 1);
        node(sc, 1, i + 1);
        let x0 = i * cs;
        let len = cs.min(w - x0);
        // A node's samples for this cell: from the cell's first column, with its row stride.
        let at = |side: usize, k: usize| {
            let (s, first) = &sc.nodes[side][k % 2];
            (&s[x0 - first..], s.len() / n_rows)
        };
        let s = [at(0, i), at(0, i + 1), at(1, i), at(1, i + 1)];
        #[cfg(target_arch = "aarch64")]
        if len == cs && neon::blends(cs) {
            let stride = out.stride();
            let dst = out.block_mut(rows.start, x0, len, n_rows);
            // SAFETY: every node holds `n_rows` rows of at least `len` samples from the cell's
            // first column; `dst` holds `n_rows` rows `stride` apart.
            unsafe { neon::blend(&s, n_rows, v0, cs, dst, stride) };
            continue;
        }
        for r in 0..n_rows {
            let row = |q: usize| &s[q].0[r * s[q].1..][..len];
            let dst = &mut out.row_mut(rows.start + r)[x0..x0 + len];
            blend_row([row(0), row(1), row(2), row(3)], c, v0 + r as u16, dst);
        }
    }
}

/// Rows `rows` and the `out.len() / rows.len()` pixels from `x0` of `reference` translated by
/// `d` (1/16 pixel, rounded to 1/4), coordinates clamped to the plane and above `height`.
#[allow(clippy::too_many_arguments)]
fn translate(
    reference: &Plane<u8>,
    height: usize,
    x0: usize,
    rows: Range<usize>,
    d: [i32; 2],
    edge: &mut Vec<u8>,
    h: &mut [Vec<u8>; 2],
    out: &mut [u8],
) {
    let (w, hh) = (
        reference.width() as i32,
        height.min(reference.height()) as i32,
    );
    let n = rows.len();
    let len = out.len() / n;
    let span = len + 1;
    // Quarter pixels, rounded half up.
    let d = d.map(|v| (v + 2) >> 2);
    let (px, py) = ((d[0] & 3) as u8, (d[1] & 3) as u8);
    let ix = x0 as i32 + (d[0] >> 2);
    let iy0 = rows.start as i32 + (d[1] >> 2);
    let ref_row = |k: usize| reference.row((iy0 + k as i32).clamp(0, hh - 1) as usize);
    if ix >= 0 && ix + (len as i32) < w {
        let ix = ix as usize;
        #[cfg(target_arch = "aarch64")]
        if neon::fits(len) {
            let base = &reference.data()[ix..];
            // SAFETY: rows clamp to `0..hh` (inside the plane), and each reads `len + 1 ≤ w − ix`
            // samples from `ix`; `out` holds `n` rows of `len`.
            return unsafe {
                neon::translate(
                    base,
                    reference.stride(),
                    iy0 as isize,
                    hh as isize,
                    len,
                    px,
                    py,
                    out,
                )
            };
        }
        let src = |k: usize| &ref_row(k)[ix..ix + span];
        return filter(&src, len, px, py, h, out);
    }
    // Columns past the sides repeat the edge samples: the rows the filter reads, padded.
    edge.resize((n + 1) * span, 0);
    let lo = (-ix).clamp(0, span as i32) as usize;
    let hi = (w - ix).clamp(lo as i32, span as i32) as usize;
    for (k, e) in edge.chunks_exact_mut(span).enumerate() {
        let row = ref_row(k);
        e[..lo].fill(row[0]);
        if hi > lo {
            let x = (ix + lo as i32) as usize;
            e[lo..hi].copy_from_slice(&row[x..x + hi - lo]);
        }
        e[hi..].fill(row[w as usize - 1]);
    }
    let edge = &*edge;
    #[cfg(target_arch = "aarch64")]
    if neon::fits(len) {
        // SAFETY: the `n + 1` padded rows hold `len + 1` samples each.
        return unsafe { neon::translate(edge, span, 0, n as isize + 1, len, px, py, out) };
    }
    let src = |k: usize| &edge[k * span..(k + 1) * span];
    filter(&src, len, px, py, h, out);
}

/// `⌈(a + b) / 2⌉`: one rounding halving add (`URHADD`, `PAVGB`) per 16 lanes.
#[inline(always)]
fn avg(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

/// The sample `p` quarters of the way from `a` to `b`, by halving adds: `a`, `⌈(a + ⌈(a + b)/2⌉)/2⌉`,
/// `⌈(a + b)/2⌉`, `⌈(b + ⌈(a + b)/2⌉)/2⌉`.
#[inline(always)]
fn quarter(a: &[u8], b: &[u8], p: u8, out: &mut [u8]) {
    match p {
        0 => out.copy_from_slice(&a[..out.len()]),
        1 => quarter_p::<1>(a, b, out),
        2 => quarter_p::<2>(a, b, out),
        _ => quarter_p::<3>(a, b, out),
    }
}

/// [`quarter`] for one phase, 16 lanes at a time (explicit: fixed-size loops of it compile to
/// scalar code).
#[inline(always)]
fn quarter_p<const P: u8>(a: &[u8], b: &[u8], out: &mut [u8]) {
    let n = out.len();
    let (a, b) = (&a[..n], &b[..n]);
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the AArch64 baseline; 16 lanes from `i` lie in all three slices.
    unsafe {
        use std::arch::aarch64::*;
        while i + 16 <= n {
            let x = vld1q_u8(a.as_ptr().add(i));
            let y = vld1q_u8(b.as_ptr().add(i));
            let m = vrhaddq_u8(x, y);
            let r = match P {
                1 => vrhaddq_u8(x, m),
                2 => m,
                _ => vrhaddq_u8(y, m),
            };
            vst1q_u8(out.as_mut_ptr().add(i), r);
            i += 16;
        }
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86-64 baseline; 16 lanes from `i` lie in all three slices.
    unsafe {
        use std::arch::x86_64::*;
        while i + 16 <= n {
            let x = _mm_loadu_si128(a.as_ptr().add(i).cast());
            let y = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let m = _mm_avg_epu8(x, y);
            let r = match P {
                1 => _mm_avg_epu8(x, m),
                2 => m,
                _ => _mm_avg_epu8(y, m),
            };
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), r);
            i += 16;
        }
    }
    for k in i..n {
        let m = avg(a[k], b[k]);
        out[k] = match P {
            1 => avg(a[k], m),
            2 => m,
            _ => avg(b[k], m),
        };
    }
}

/// The separable quarter-pixel filter of `out.len() / len` rows `len` wide, row `k` of the source
/// (`len + 1` samples) given by `src(k)`: horizontally into 8-bit rows, then vertically.
fn filter<'a>(
    src: &impl Fn(usize) -> &'a [u8],
    len: usize,
    px: u8,
    py: u8,
    h: &mut [Vec<u8>; 2],
    out: &mut [u8],
) {
    for hr in h.iter_mut() {
        hr.resize(len, 0);
    }
    let hor = |k: usize, dst: &mut [u8]| {
        let row = src(k);
        quarter(&row[..len], &row[1..], px, dst);
    };
    if py == 0 {
        for (k, o) in out.chunks_exact_mut(len).enumerate() {
            hor(k, o);
        }
        return;
    }
    hor(0, &mut h[0]);
    for (k, o) in out.chunks_exact_mut(len).enumerate() {
        let (cur, next) = (k % 2, (k + 1) % 2);
        hor(k + 1, &mut h[next]);
        quarter(&h[cur], &h[next], py, o);
    }
}

/// [`filter`] in NEON for spans of 16, 32 or 64 pixels: per output row, each 16 columns' two
/// reference loads, horizontal averages kept in registers for the next row, and the vertical
/// average stored; specialized by phase pair and width, so nothing branches inside.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    pub(super) fn fits(len: usize) -> bool {
        matches!(len, 16 | 32 | 64)
    }

    /// Whether [`blend`] takes cells of `cs` pixels.
    pub(super) fn blends(cs: usize) -> bool {
        matches!(cs, 16 | 32 | 64)
    }

    /// `blend_row` over a whole cell of `cs` pixels: rows `v0 ..` of `n` rows, node `q`'s samples
    /// `s[q] = (samples, row stride)` from the cell's first column, output rows `stride` apart.
    ///
    /// # Safety
    /// Every node must hold `n` rows of `cs` samples, `dst` `n` rows; `cs` must [`blends`].
    pub(super) unsafe fn blend(
        s: &[(&[u8], usize); 4],
        n: usize,
        v0: u16,
        cs: usize,
        dst: &mut [u8],
        stride: usize,
    ) {
        // SAFETY: as documented; `2c − e = 8` for these cells.
        unsafe {
            match cs {
                16 => blend_kernel::<1, 0>(s, n, v0, dst, stride),
                32 => blend_kernel::<2, 2>(s, n, v0, dst, stride),
                _ => blend_kernel::<4, 4>(s, n, v0, dst, stride),
            }
        }
    }

    /// Cells of `16·R` pixels, pre-shift `E`.
    #[target_feature(enable = "neon")]
    unsafe fn blend_kernel<const R: usize, const E: i32>(
        s: &[(&[u8], usize); 4],
        n: usize,
        v0: u16,
        dst: &mut [u8],
        stride: usize,
    ) {
        let cs = (16 * R) as u16;
        // Per lane: u and C − u.
        let lanes: [u8; 16] = std::array::from_fn(|k| k as u8);
        // SAFETY: 16 lanes from `lanes`.
        let first = unsafe { vld1q_u8(lanes.as_ptr()) };
        let u: [uint8x16_t; R] = std::array::from_fn(|r| vaddq_u8(first, vdupq_n_u8(16 * r as u8)));
        let cu: [uint8x16_t; R] = std::array::from_fn(|r| vsubq_u8(vdupq_n_u8(cs as u8), u[r]));
        let o = dst.as_mut_ptr();
        // `(C − u)·a + u·b` per half, pre-shifted.
        let mix = |a: uint8x16_t, b: uint8x16_t, r: usize| -> [uint16x8_t; 2] {
            let lo = vmlal_u8(
                vmull_u8(vget_low_u8(a), vget_low_u8(cu[r])),
                vget_low_u8(b),
                vget_low_u8(u[r]),
            );
            let hi = vmlal_high_u8(vmull_high_u8(a, cu[r]), b, u[r]);
            // A rounding shift right by E (a shift by −E; none for E = 0).
            let e = vdupq_n_s16(-(E as i16));
            [vrshlq_u16(lo, e), vrshlq_u16(hi, e)]
        };
        for k in 0..n {
            let v = v0 + k as u16;
            let (wt, wb) = (cs - v, v);
            for r in 0..R {
                // SAFETY: row k of every node and of `dst` holds `16·R` samples (the contract).
                unsafe {
                    let ld = |q: usize| vld1q_u8(s[q].0.as_ptr().add(k * s[q].1 + 16 * r));
                    let top = mix(ld(0), ld(1), r);
                    let bottom = mix(ld(2), ld(3), r);
                    let f = |h: usize| vmlaq_n_u16(vmulq_n_u16(top[h], wt), bottom[h], wb);
                    let out = vcombine_u8(vrshrn_n_u16::<8>(f(0)), vrshrn_n_u16::<8>(f(1)));
                    vst1q_u8(o.add(k * stride + 16 * r), out);
                }
            }
        }
    }

    /// `Q(x, y, P)` on 16 lanes.
    #[inline(always)]
    unsafe fn q<const P: u8>(x: uint8x16_t, y: uint8x16_t) -> uint8x16_t {
        // SAFETY: NEON is part of the AArch64 baseline.
        unsafe {
            match P {
                0 => x,
                1 => vrhaddq_u8(x, vrhaddq_u8(x, y)),
                2 => vrhaddq_u8(x, y),
                _ => vrhaddq_u8(y, vrhaddq_u8(x, y)),
            }
        }
    }

    /// Output rows `0..out.len() / len` from source rows `iy0 + k` (clamped to `0..hh`), `stride`
    /// apart in `src`, each read from its start.
    ///
    /// # Safety
    /// Every clamped row must hold `len + 1` samples from its start, `len` must [`fits`].
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn translate(
        src: &[u8],
        stride: usize,
        iy0: isize,
        hh: isize,
        len: usize,
        px: u8,
        py: u8,
        out: &mut [u8],
    ) {
        macro_rules! go {
            ($r:literal) => {
                match (px, py) {
                    (0, 0) => kernel::<$r, 0, 0>,
                    (0, 1) => kernel::<$r, 0, 1>,
                    (0, 2) => kernel::<$r, 0, 2>,
                    (0, _) => kernel::<$r, 0, 3>,
                    (1, 0) => kernel::<$r, 1, 0>,
                    (1, 1) => kernel::<$r, 1, 1>,
                    (1, 2) => kernel::<$r, 1, 2>,
                    (1, _) => kernel::<$r, 1, 3>,
                    (2, 0) => kernel::<$r, 2, 0>,
                    (2, 1) => kernel::<$r, 2, 1>,
                    (2, 2) => kernel::<$r, 2, 2>,
                    (2, _) => kernel::<$r, 2, 3>,
                    (_, 0) => kernel::<$r, 3, 0>,
                    (_, 1) => kernel::<$r, 3, 1>,
                    (_, 2) => kernel::<$r, 3, 2>,
                    (_, _) => kernel::<$r, 3, 3>,
                }
            };
        }
        let f: unsafe fn(&[u8], usize, isize, isize, &mut [u8]) = match len {
            16 => go!(1),
            32 => go!(2),
            _ => go!(4),
        };
        // SAFETY: as documented.
        unsafe { f(src, stride, iy0, hh, out) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn kernel<const R: usize, const P: u8, const Q: u8>(
        src: &[u8],
        stride: usize,
        iy0: isize,
        hh: isize,
        out: &mut [u8],
    ) {
        let n = out.len() / (16 * R);
        let base = src.as_ptr();
        let o = out.as_mut_ptr();
        // SAFETY (all below): rows clamp to `0..hh`, each holding `16·R + 1` samples (the caller's
        // contract); output row `k < n` holds `16·R` bytes.
        let row =
            |k: usize| unsafe { base.add((iy0 + k as isize).clamp(0, hh - 1) as usize * stride) };
        let hor = |k: usize| -> [uint8x16_t; R] {
            let p = row(k);
            let a: [uint8x16_t; R] = std::array::from_fn(|r| unsafe { vld1q_u8(p.add(16 * r)) });
            let b: [uint8x16_t; R] =
                std::array::from_fn(|r| unsafe { vld1q_u8(p.add(16 * r + 1)) });
            std::array::from_fn(|r| unsafe { q::<P>(a[r], b[r]) })
        };
        if Q == 0 {
            for k in 0..n {
                let h = hor(k);
                for (r, v) in h.into_iter().enumerate() {
                    unsafe { vst1q_u8(o.add(k * 16 * R + 16 * r), v) };
                }
            }
            return;
        }
        let mut prev = hor(0);
        for k in 0..n {
            let next = hor(k + 1);
            for r in 0..R {
                unsafe { vst1q_u8(o.add(k * 16 * R + 16 * r), q::<Q>(prev[r], next[r])) };
            }
            prev = next;
        }
    }
}

/// One row `v` of a cell of `2^c` pixels: the four nodes' samples blended bilinearly.
#[inline(always)]
fn blend_row(s: [&[u8]; 4], c: u32, v: u16, out: &mut [u8]) {
    let n = out.len();
    let cs = 1u16 << c;
    let e = (2 * c).saturating_sub(8);
    let k = 2 * c - e;
    let (pre, round) = (if e > 0 { 1u16 << (e - 1) } else { 0 }, 1u16 << (k - 1));
    let (wt, wb) = (cs - v, v);
    let (a, b, cc, d) = (&s[0][..n], &s[1][..n], &s[2][..n], &s[3][..n]);
    for u in 0..n {
        let uu = u as u16;
        let top = (cs - uu) * u16::from(a[u]) + uu * u16::from(b[u]);
        let bottom = (cs - uu) * u16::from(cc[u]) + uu * u16::from(d[u]);
        let (top, bottom) = ((top + pre) >> e, (bottom + pre) >> e);
        out[u] = ((wt * top + wb * bottom + round) >> k) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_never_overflows() {
        for c in 2..=6 {
            let cs = 1u16 << c;
            let full = vec![255u8; cs as usize];
            let mut out = vec![0u8; cs as usize];
            for v in 0..cs {
                blend_row([&full, &full, &full, &full], c, v, &mut out);
                assert!(out.iter().all(|&o| o == 255), "c {c} v {v}: {out:?}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_matches_portable() {
        let mut rng = crate::rng::SplitMix64::new(9);
        let (w, h) = (100, 40);
        let mut r = Plane::new(w, h);
        for y in 0..h {
            for v in r.row_mut(y) {
                *v = rng.below(256) as u8;
            }
        }
        let mut hb: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        for case in 0..400 {
            let len = [16, 32, 64][case % 3];
            let rows = 1 + rng.below(20) as usize;
            let (px, py) = (rng.below(4) as u8, rng.below(4) as u8);
            let ix = rng.below((w - len) as u64) as usize;
            let iy0 = rng.range_i32(-5, h as i32 + 5) as isize;
            let (mut a, mut b) = (vec![0u8; len * rows], vec![0u8; len * rows]);
            let src = |k: usize| {
                let y = (iy0 + k as isize).clamp(0, h as isize - 1) as usize;
                &r.row(y)[ix..ix + len + 1]
            };
            filter(&src, len, px, py, &mut hb, &mut a);
            // SAFETY: rows clamp inside the plane and hold `len + 1` samples from `ix`.
            unsafe {
                neon::translate(
                    &r.data()[ix..],
                    r.stride(),
                    iy0,
                    h as isize,
                    len,
                    px,
                    py,
                    &mut b,
                )
            };
            assert_eq!(a, b, "case {case}: len {len} rows {rows} phases {px} {py}");
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_blend_matches_portable() {
        let mut rng = crate::rng::SplitMix64::new(5);
        for cs in [16usize, 32, 64] {
            let c = cs.trailing_zeros();
            let nodes: Vec<Vec<u8>> = (0..4)
                .map(|_| (0..cs * cs).map(|_| rng.below(256) as u8).collect())
                .collect();
            let s: [(&[u8], usize); 4] = std::array::from_fn(|q| (&nodes[q][..], cs));
            let mut a = vec![0u8; cs * cs];
            for v in 0..cs {
                let row = |q: usize| &nodes[q][v * cs..(v + 1) * cs];
                blend_row(
                    [row(0), row(1), row(2), row(3)],
                    c,
                    v as u16,
                    &mut a[v * cs..(v + 1) * cs],
                );
            }
            let mut b = vec![0u8; cs * cs];
            // SAFETY: every node and the output hold `cs` rows of `cs`.
            unsafe { neon::blend(&s, cs, 0, cs, &mut b, cs) };
            assert_eq!(a, b, "cs {cs}");
        }
    }

    #[test]
    fn whole_vectors_copy() {
        let mut r = Plane::new(40, 20);
        for y in 0..20 {
            for (x, v) in r.row_mut(y).iter_mut().enumerate() {
                *v = (x * 7 + y * 13) as u8;
            }
        }
        let (mut edge, mut h): (Vec<u8>, [Vec<u8>; 2]) = (Vec::new(), [Vec::new(), Vec::new()]);
        let mut out = vec![0u8; 8 * 4];
        translate(&r, 20, 8, 4..8, [32, -16], &mut edge, &mut h, &mut out);
        for k in 0..4 {
            assert_eq!(&out[k * 8..k * 8 + 8], &r.row(3 + k)[10..18]);
        }
        // Half a pixel right: the rounded average of neighbours.
        translate(&r, 20, 8, 4..8, [8, 0], &mut edge, &mut h, &mut out);
        let row = r.row(4);
        for x in 0..8 {
            let want = (u16::from(row[8 + x]) + u16::from(row[9 + x])).div_ceil(2);
            assert_eq!(u16::from(out[x]), want);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::motion::Grid;

    /// `cargo test --release -- --ignored --nocapture rt::predict::bench`: 720p prediction,
    /// one thread and four.
    #[test]
    #[ignore]
    fn speed() {
        let (w, h) = (1280, 720);
        let mut rng = crate::rng::SplitMix64::new(3);
        let mut reference = Frame::new(w, h);
        for p in reference.planes_mut() {
            for y in 0..p.height() {
                for v in p.row_mut(y) {
                    *v = rng.below(256) as u8;
                }
            }
        }
        let mut field = MotionField::new(Grid::new(w, h, 5));
        // A smooth field with fractional vectors, as a pan with some zoom.
        let nx = field.grid.nx;
        for (n, mv) in field.mv.iter_mut().enumerate() {
            let (i, j) = ((n % nx) as i32, (n / nx) as i32);
            *mv = [
                37 + i * 3 + rng.below(3) as i32,
                -21 + j * 2 + rng.below(3) as i32,
            ];
        }
        let mut pred = Frame::new(w, h);
        for threads in [1, 4] {
            let pool = Pool::new(threads);
            for _ in 0..3 {
                let s = std::time::Instant::now();
                for _ in 0..20 {
                    predict(&pool, &reference, &mut pred, &field, [0, 0]);
                }
                let t = s.elapsed().as_secs_f64() / 20.0 * 1e3;
                eprintln!("720p prediction, {threads} threads: {t:.2} ms");
            }
        }
    }
}
