//! Motion compensation: the normative warp of a reference plane by a node motion field.
//!
//! A pixel `(x, y)` lies in the cell of nodes `(i, j) .. (i+1, j+1)`, at offset `(u, v)` inside a
//! cell of `C = 2^c` pixels. Its vector is the bilinear interpolation of the four node vectors
//! `m00, m10, m01, m11` (unit `2^-f` pixel), kept exact:
//!
//! ```text
//! M = (C−u)(C−v)·m00 + u(C−v)·m10 + (C−u)v·m01 + uv·m11          unit 2^-(f+2c) pixel
//! p = 64·x + ((M + 2^(k−1)) >> k),  k = f + 2c − 6                 position in 1/64 pixel
//! ```
//!
//! The sample is a separable `T`-tap filter at phase `p & 63` around `p >> 6`, reading the
//! reference with coordinates clamped to the plane (edge extension). Taps have 7 bits and sum to
//! 128:
//!
//! ```text
//! t_j = Σ_i h[fx][i] · R(ix − o + i, iy − o + j)        o = T/2 − 1
//! out = clamp((Σ_j h[fy][j] · t_j + 2^13) >> 14, 0, 255)
//! ```
//!
//! Luma uses `c` = the grid's log2 spacing and `f = 3`. 4:2:0 chroma uses `c − 1` and `f = 4`:
//! the same vectors read as 1/16 chroma pel.
//!
//! **Intra nodes.** The inter weight `w ∈ [0, 2^2c]` interpolates the node flags (inter = 1,
//! intra = 0) the same way, and where `w < 2^2c` the prediction fades towards mid-grey:
//! `128 + ((w·(out − 128) + 2^(2c−1)) >> 2c)`.
//!
//! Along a row inside a cell `M` is linear in `u`, so positions advance by a constant integer
//! step. A cell row (a *span*) whose samples all stay inside the reference skips the clamping,
//! and goes to the ISA's SIMD kernel. Every kernel is bit-identical to the portable code, which
//! [`verify`] checks.
//!
//! **SIMD kernels.** The field is smooth, so eight neighbouring pixels read from a narrow window:
//! their integer positions span a dozen columns at most, and one or two rows. The kernels take
//! groups of 8 pixels, compute their positions with vector arithmetic, load 16 bytes of each
//! reference row the group needs and hand every lane its samples with a byte shuffle (`pshufb`
//! on x86, `tbl` on NEON) instead of a gather, which NEON lacks and which is slow on many x86
//! cores. Samples are centred on 128 so that every horizontal sum fits 16 bits; the taps sum to
//! 128 in each direction, so the offset comes back as exactly 128 after the final shift. A group
//! that straddles two rows computes both alignments and picks per lane. Wider groups (steep
//! zoom), and the tail of a span, take the portable path.

use super::{MotionField, Prediction};
use crate::plane::RowsMut;
use crate::pool::Pool;
use crate::rng::SplitMix64;
use crate::{Frame, Isa, Plane};

/// Sampling positions are in 1/64 pixel.
pub(super) const PHASE_BITS: u32 = 6;
pub(super) const PHASES: usize = 1 << PHASE_BITS;
/// Taps sum to `1 << TAP_BITS`.
const TAP_BITS: u32 = 7;

/// Interpolation filter for reading the reference between pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    /// 2-tap linear interpolation: the cheapest, and the smoother prediction. Each warp softens
    /// the picture slightly, which compounds over frames of rigid motion but helps where the
    /// motion field cannot follow the true motion exactly.
    Bilinear,
    /// 4-tap Keys cubic with a = −1/2 (Catmull-Rom): keeps detail through repeated warps.
    #[default]
    Cubic,
    /// 4-tap Keys cubic with a = −3/4: sharper, with more overshoot at edges.
    Sharp,
    /// 4-tap cubic B-spline: an approximating filter that blurs even whole-pixel positions
    /// (`[1, 4, 1] / 6`). On real footage that blur cleans coding noise and sensor noise out of
    /// the prediction, like a loop filter would, and wins over bilinear (DESIGN.md §6.3).
    Smooth,
}

pub(super) type Taps = [[i16; 4]; PHASES];

static BILINEAR: Taps = bilinear_taps();
static CUBIC: Taps = keys_taps(-2);
static SHARP: Taps = keys_taps(-3);
static SMOOTH: Taps = mitchell_taps(1, 0, 1);

/// One cell row: output pixels `x .. x + len` of row `y`. Pixel `x + u` samples the reference at
/// `64·(x + u) + ((bx + u·dx) >> k)` horizontally and `64·y + ((by + u·dy) >> k)` vertically, in
/// 1/64 pixel; `bx` and `by` include the rounding term.
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub x: i32,
    pub y: i32,
    pub len: usize,
    pub bx: i32,
    pub dx: i32,
    pub by: i32,
    pub dy: i32,
    pub k: u32,
}

impl Span {
    #[inline]
    pub fn px(&self, u: usize) -> i32 {
        let u = u as i32;
        ((self.x + u) << PHASE_BITS) + ((self.bx + u * self.dx) >> self.k)
    }

    #[inline]
    pub fn py(&self, u: usize) -> i32 {
        (self.y << PHASE_BITS) + ((self.by + u as i32 * self.dy) >> self.k)
    }
}

/// A SIMD span sampler, a drop-in for [`span_inside`]: every tap of the span lies inside the
/// reference.
pub(super) type SpanKernel = fn(&Plane<u8>, &Span, &Taps, &mut [u8]);

/// The SIMD span sampler of `isa` for `taps`-tap filters, if it has one.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(unused_variables)
)]
fn span_kernel(isa: Isa, taps: usize) -> Option<SpanKernel> {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => super::warp_avx2::kernel(taps),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => super::warp_neon::kernel(taps),
        _ => None,
    }
}

impl Filter {
    pub const ALL: [Filter; 4] = [
        Filter::Bilinear,
        Filter::Cubic,
        Filter::Sharp,
        Filter::Smooth,
    ];

    /// Stream code.
    pub fn code(self) -> u8 {
        match self {
            Filter::Bilinear => 0,
            Filter::Cubic => 1,
            Filter::Sharp => 2,
            Filter::Smooth => 3,
        }
    }

    pub fn from_code(code: u8) -> Option<Filter> {
        Self::ALL.into_iter().find(|f| f.code() == code)
    }

    pub fn name(self) -> &'static str {
        match self {
            Filter::Bilinear => "bilinear",
            Filter::Cubic => "cubic",
            Filter::Sharp => "sharp",
            Filter::Smooth => "smooth",
        }
    }

    pub fn from_name(name: &str) -> Option<Filter> {
        Self::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(name))
    }

    /// Filter length.
    pub fn taps(self) -> usize {
        match self {
            Filter::Bilinear => 2,
            Filter::Cubic | Filter::Sharp | Filter::Smooth => 4,
        }
    }

    /// Taps per phase. Two-tap filters use the first two entries.
    pub(super) fn table(self) -> &'static Taps {
        match self {
            Filter::Bilinear => &BILINEAR,
            Filter::Cubic => &CUBIC,
            Filter::Sharp => &SHARP,
            Filter::Smooth => &SMOOTH,
        }
    }
}

const fn bilinear_taps() -> Taps {
    let mut t = [[0i16; 4]; PHASES];
    let mut f = 0;
    while f < PHASES {
        t[f] = [128 - 2 * f as i16, 2 * f as i16, 0, 0];
        f += 1;
    }
    t
}

/// Keys cubic convolution with `a = a4/4`, at phases `t = f/64`, scaled to 128 and rounded, with
/// the rounding error put on the larger centre tap. Integer arithmetic only: every weight is a
/// cubic in `f` with denominator 8192.
const fn keys_taps(a4: i64) -> Taps {
    let mut t = [[0i16; 4]; PHASES];
    let mut f = 0;
    while f < PHASES {
        let x = f as i64;
        let (x2, x3) = (x * x, x * x * x);
        let num = [
            a4 * (x3 - 128 * x2 + 4096 * x),
            (a4 + 8) * x3 - 64 * (a4 + 12) * x2 + 1_048_576,
            -(a4 + 8) * x3 + 64 * (2 * a4 + 12) * x2 - 4096 * a4 * x,
            a4 * (64 * x2 - x3),
        ];
        let mut taps = [0i16; 4];
        let mut sum = 0;
        let mut k = 0;
        while k < 4 {
            // Round half away from zero.
            let r = if num[k] >= 0 {
                (num[k] + 4096) / 8192
            } else {
                -((-num[k] + 4096) / 8192)
            };
            taps[k] = r as i16;
            sum += r;
            k += 1;
        }
        let centre = if f < PHASES / 2 { 1 } else { 2 };
        taps[centre] += (128 - sum) as i16;
        t[f] = taps;
        f += 1;
    }
    t
}

/// Rows per task of the parallel warp: a multiple of every row step used.
const BAND_ROWS: usize = 32;

/// [`warp_frame`] on a thread pool, in bands of rows. The result is the same.
pub fn warp_frame_on(
    pool: &Pool,
    reference: &Frame,
    field: &MotionField,
    filter: Filter,
    isa: Isa,
    out: &mut Frame,
) {
    warp_planes(
        pool,
        &reference.planes(),
        out.planes_mut().into(),
        field,
        Prediction::Warp,
        filter,
        isa,
        1,
    );
}

/// Rows of a sampled prediction: with `row_step` above 1, every `row_step`-th span of
/// [`SAMPLE_ROWS`] rows from row 0. Whole spans keep the overlapped prediction's translations as
/// cheap per row as a full prediction's.
pub(crate) const SAMPLE_ROWS: usize = 16;

/// Whether row `y` is in a sampled prediction (see [`SAMPLE_ROWS`]).
#[inline]
pub(crate) fn sampled(y: usize, row_step: usize) -> bool {
    row_step == 1 || (y / SAMPLE_ROWS).is_multiple_of(row_step)
}

/// Predicts planes (luma, then chroma) on `pool`: every row, or with `row_step` above 1 only the
/// [`sampled`] rows; other rows of `out` are left as they are.
#[allow(clippy::too_many_arguments)]
pub(crate) fn warp_planes(
    pool: &Pool,
    references: &[&Plane<u8>],
    outs: Vec<&mut Plane<u8>>,
    field: &MotionField,
    prediction: Prediction,
    filter: Filter,
    isa: Isa,
    row_step: usize,
) {
    let mut tasks: Vec<(usize, RowsMut<'_, u8>)> = Vec::new();
    for (c, (reference, out)) in references.iter().zip(outs).enumerate() {
        assert_eq!(
            (reference.width(), reference.height()),
            (out.width(), out.height()),
            "plane size mismatch"
        );
        tasks.extend(out.bands_mut(BAND_ROWS).into_iter().map(|band| (c, band)));
    }
    pool.for_each_mut(&mut tasks, |_, (c, band)| match prediction {
        Prediction::Warp => warp_band(references[*c], field, *c > 0, filter, isa, row_step, band),
        Prediction::Overlapped => {
            let r = references[*c];
            super::obmc::obmc_band(r, r.height(), field, *c > 0, filter, isa, row_step, band)
        }
    });
}

/// The Mitchell–Netravali cubic with `B = b/den` and `C = c/den` (the B-spline is `B = 1, C = 0`), at phases `t = f/64`, scaled to
/// 128 and rounded half away from zero, with the rounding error put on the larger centre tap.
/// Integer arithmetic only. Catmull-Rom is `B = 0, C = 1/2` (a test checks it against
/// [`keys_taps`]).
const fn mitchell_taps(b: i64, c: i64, den: i64) -> Taps {
    // k(d) for a tap at distance d (in 1/64 pixel), times 6·den·64³.
    const fn kernel(d: i64, b: i64, c: i64, den: i64) -> i64 {
        if d < 64 {
            (12 * den - 9 * b - 6 * c) * d * d * d
                + (-18 * den + 12 * b + 6 * c) * 64 * d * d
                + (6 * den - 2 * b) * 262_144
        } else {
            (-b - 6 * c) * d * d * d
                + (6 * b + 30 * c) * 64 * d * d
                + (-12 * b - 48 * c) * 4096 * d
                + (8 * b + 24 * c) * 262_144
        }
    }
    let scale = 6 * den * 262_144;
    let mut t = [[0i16; 4]; PHASES];
    let mut f = 0;
    while f < PHASES {
        let x = f as i64;
        let d = [64 + x, x, 64 - x, 128 - x];
        let mut taps = [0i16; 4];
        let mut sum = 0;
        let mut k = 0;
        while k < 4 {
            let num = 128 * kernel(d[k], b, c, den);
            let r = if num >= 0 {
                (num + scale / 2) / scale
            } else {
                -((-num + scale / 2) / scale)
            };
            taps[k] = r as i16;
            sum += r;
            k += 1;
        }
        let centre = if f < PHASES / 2 { 1 } else { 2 };
        taps[centre] += (128 - sum) as i16;
        t[f] = taps;
        f += 1;
    }
    t
}

/// Motion-compensated prediction of a whole 4:2:0 frame, either way (see [`Prediction`]).
/// Every ISA gives the same result.
pub fn predict_frame(
    reference: &Frame,
    field: &MotionField,
    prediction: Prediction,
    filter: Filter,
    isa: Isa,
    out: &mut Frame,
) {
    let pool = Pool::new(1);
    warp_planes(
        &pool,
        &reference.planes(),
        out.planes_mut().into(),
        field,
        prediction,
        filter,
        isa,
        1,
    );
}

/// Motion-compensated prediction of a whole 4:2:0 frame. Every ISA gives the same result.
pub fn warp_frame(
    reference: &Frame,
    field: &MotionField,
    filter: Filter,
    isa: Isa,
    out: &mut Frame,
) {
    for (c, (r, o)) in reference
        .planes()
        .into_iter()
        .zip(out.planes_mut())
        .enumerate()
    {
        warp_plane(r, field, c > 0, filter, isa, o);
    }
}

/// Warps `reference` by `field` into `out` (same size). `chroma` selects the 4:2:0 chroma scale.
pub fn warp_plane(
    reference: &Plane<u8>,
    field: &MotionField,
    chroma: bool,
    filter: Filter,
    isa: Isa,
    out: &mut Plane<u8>,
) {
    assert_eq!(
        (reference.width(), reference.height()),
        (out.width(), out.height()),
        "plane size mismatch"
    );
    warp_rows(reference, field, chroma, filter, isa, 1, out);
}

/// [`warp_plane`] for the [`sampled`] rows only; other rows of `out` are left as they are. A
/// cheap sample of the prediction for encoder decisions.
pub(crate) fn warp_rows(
    reference: &Plane<u8>,
    field: &MotionField,
    chroma: bool,
    filter: Filter,
    isa: Isa,
    row_step: usize,
    out: &mut Plane<u8>,
) {
    assert_eq!(
        (reference.width(), reference.height()),
        (out.width(), out.height()),
        "plane size mismatch"
    );
    let height = out.height();
    for mut band in out.bands_mut(height) {
        warp_band(reference, field, chroma, filter, isa, row_step, &mut band);
    }
}

/// The rows of `band` (a band of a plane the size of `reference`) that are multiples of
/// `row_step`.
fn warp_band(
    reference: &Plane<u8>,
    field: &MotionField,
    chroma: bool,
    filter: Filter,
    isa: Isa,
    row_step: usize,
    band: &mut RowsMut<'_, u8>,
) {
    let simd = span_kernel(isa, filter.taps());
    let taps = filter.table();
    match filter.taps() {
        2 => warp::<2>(reference, field, chroma, taps, simd, row_step, band),
        _ => warp::<4>(reference, field, chroma, taps, simd, row_step, band),
    }
}

/// Checks `isa`'s warp kernels against the portable code on `cases` random planes and fields.
pub fn verify(isa: Isa, cases: usize, seed: u64) -> Result<(), String> {
    if !isa.is_supported() {
        return Err(format!("{isa} is not supported on this CPU"));
    }
    let mut rng = SplitMix64::new(seed);
    for case in 0..cases {
        let (w, h) = (rng.range_usize(1, 300), rng.range_usize(1, 120));
        let grid = super::Grid::new(w, h, 4 + (case % 3) as u32);
        let mut field = MotionField::new(grid);
        let spread = [0, 4, 40, 400, 3000][case % 5];
        let base = [rng.range_i32(-300, 300), rng.range_i32(-300, 300)];
        for n in 0..grid.nodes() {
            field.mv[n] = [
                base[0] + rng.range_i32(-spread, spread),
                base[1] + rng.range_i32(-spread, spread),
            ];
            field.intra[n] = case % 4 == 1 && rng.below(3) == 0;
        }
        let mut reference = Frame::new(w, h);
        for plane in reference.planes_mut() {
            for row in plane.rows_mut() {
                row.fill_with(|| rng.next_u64() as u8);
            }
        }
        for filter in Filter::ALL {
            for prediction in Prediction::ALL {
                let (mut a, mut b) = (Frame::new(w, h), Frame::new(w, h));
                predict_frame(&reference, &field, prediction, filter, Isa::Scalar, &mut a);
                predict_frame(&reference, &field, prediction, filter, isa, &mut b);
                let same = a
                    .planes()
                    .iter()
                    .zip(b.planes())
                    .all(|(p, q)| p.rows().eq(q.rows()));
                if !same {
                    return Err(format!(
                        "{isa} {} prediction differs from scalar: case {case}, {w}x{h}, {}",
                        prediction.name(),
                        filter.name()
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Cell size and vector units of a plane: `(c, k)` as in the module documentation.
fn plane_scale(field: &MotionField, chroma: bool) -> (u32, u32) {
    let c = field.grid.log2 - u32::from(chroma);
    let f = 3 + u32::from(chroma);
    (c, f + 2 * c - PHASE_BITS)
}

#[allow(clippy::too_many_arguments)]
fn warp<const T: usize>(
    reference: &Plane<u8>,
    field: &MotionField,
    chroma: bool,
    taps: &Taps,
    simd: Option<SpanKernel>,
    row_step: usize,
    out: &mut RowsMut<'_, u8>,
) {
    let (c, k) = plane_scale(field, chroma);
    let cs = 1i32 << c;
    let half = 1i32 << (k - 1);
    let (w, h) = (reference.width(), reference.height());
    let o = (T / 2 - 1) as i32;
    let nx = field.grid.nx;
    for y in out.rows().filter(|&y| sampled(y, row_step)) {
        let (j, v) = (y >> c, (y & (cs as usize - 1)) as i32);
        let dst_row = out.row_mut(y);
        for i in 0..w.div_ceil(cs as usize) {
            let x0 = i << c;
            let len = (cs as usize).min(w - x0);
            let n00 = j * nx + i;
            let nodes = [n00, n00 + 1, n00 + nx, n00 + nx + 1];
            // Along the row, M(u) = cs·l + u·(r − l), from the cell's left and right edges.
            let edge = |a: i32, b: i32| (cs - v) * a + v * b;
            let line = |comp: usize| {
                let mv = |n: usize| field.mv[nodes[n]][comp];
                let (l, r) = (edge(mv(0), mv(2)), edge(mv(1), mv(3)));
                (cs * l + half, r - l)
            };
            let ((bx, dx), (by, dy)) = (line(0), line(1));
            let span = Span {
                x: x0 as i32,
                y: y as i32,
                len,
                bx,
                dx,
                by,
                dy,
                k,
            };

            // Positions are monotonic along the span, so its ends bound every tap.
            let (xa, xb) = (span.px(0) >> PHASE_BITS, span.px(len - 1) >> PHASE_BITS);
            let (ya, yb) = (span.py(0) >> PHASE_BITS, span.py(len - 1) >> PHASE_BITS);
            let inside = xa.min(xb) - o >= 0
                && xa.max(xb) - o + T as i32 <= w as i32
                && ya.min(yb) - o >= 0
                && ya.max(yb) - o + T as i32 <= h as i32;
            let dst = &mut dst_row[x0..x0 + len];
            match simd {
                Some(kernel) if inside => kernel(reference, &span, taps, dst),
                _ if inside => span_inside::<T>(reference, &span, 0..len, taps, dst),
                _ => span_clamped::<T>(reference, &span, taps, dst),
            }

            let flags = nodes.map(|n| i32::from(!field.intra[n]));
            if flags != [1; 4] {
                let (l, r) = (edge(flags[0], flags[2]), edge(flags[1], flags[3]));
                let (base, step) = (cs * l, r - l);
                for (u, d) in dst.iter_mut().enumerate() {
                    *d = blend(*d, base + u as i32 * step, 2 * c);
                }
            }
        }
    }
}

#[inline]
fn finish(acc: i32) -> u8 {
    ((acc + (1 << (2 * TAP_BITS - 1))) >> (2 * TAP_BITS)).clamp(0, 255) as u8
}

/// Fades `s` towards 128 by the inter weight `weight / 2^shift`.
#[inline]
fn blend(s: u8, weight: i32, shift: u32) -> u8 {
    (128 + ((weight * (i32::from(s) - 128) + (1 << (shift - 1))) >> shift)) as u8
}

/// Pixels `pixels` of a span whose taps all lie inside the reference, into `dst` (the whole
/// span).
pub(super) fn span_inside<const T: usize>(
    reference: &Plane<u8>,
    span: &Span,
    pixels: std::ops::Range<usize>,
    taps: &Taps,
    dst: &mut [u8],
) {
    let o = T / 2 - 1;
    let (data, stride) = (reference.data(), reference.stride());
    for (d, u) in dst[pixels.clone()].iter_mut().zip(pixels) {
        let (px, py) = (span.px(u), span.py(u));
        let ix = (px >> PHASE_BITS) as usize - o;
        let iy = (py >> PHASE_BITS) as usize - o;
        let hx = &taps[(px & (PHASES as i32 - 1)) as usize];
        let hy = &taps[(py & (PHASES as i32 - 1)) as usize];
        let mut acc = 0i32;
        for (j, &wy) in hy.iter().take(T).enumerate() {
            let at = (iy + j) * stride + ix;
            let t: i32 = data[at..at + T]
                .iter()
                .zip(hx)
                .map(|(&s, &wx)| i32::from(s) * i32::from(wx))
                .sum();
            acc += i32::from(wy) * t;
        }
        *d = finish(acc);
    }
}

/// A span near the reference's edge: every tap's coordinates are clamped.
fn span_clamped<const T: usize>(reference: &Plane<u8>, span: &Span, taps: &Taps, dst: &mut [u8]) {
    let (w, h) = (reference.width() as i32, reference.height() as i32);
    let o = (T / 2 - 1) as i32;
    for (u, d) in dst.iter_mut().enumerate() {
        let (p, q) = (span.px(u), span.py(u));
        let (x, y) = ((p >> PHASE_BITS) - o, (q >> PHASE_BITS) - o);
        let hx = &taps[(p & (PHASES as i32 - 1)) as usize];
        let hy = &taps[(q & (PHASES as i32 - 1)) as usize];
        let mut acc = 0i32;
        for (j, &wy) in hy.iter().take(T).enumerate() {
            let row = reference.row((y + j as i32).clamp(0, h - 1) as usize);
            let t: i32 = (0..T)
                .map(|i| i32::from(row[(x + i as i32).clamp(0, w - 1) as usize]) * i32::from(hx[i]))
                .sum();
            acc += i32::from(wy) * t;
        }
        *d = finish(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::Grid;
    use crate::rng::SplitMix64;

    /// The module-level definition, pixel by pixel, without any incremental shortcuts.
    fn warp_reference(
        reference: &Plane<u8>,
        field: &MotionField,
        chroma: bool,
        filter: Filter,
        out: &mut Plane<u8>,
    ) {
        let (c, k) = plane_scale(field, chroma);
        let cs = 1i64 << c;
        let t = filter.taps() as i64;
        let o = t / 2 - 1;
        let taps = filter.table();
        let (w, h) = (reference.width() as i64, reference.height() as i64);
        for y in 0..out.height() {
            for x in 0..out.width() {
                let (i, j) = (x >> c, y >> c);
                let (u, v) = (x as i64 & (cs - 1), y as i64 & (cs - 1));
                let n = |di: usize, dj: usize| field.node(i + di, j + dj);
                let wts = [(cs - u) * (cs - v), u * (cs - v), (cs - u) * v, u * v];
                let ns = [n(0, 0), n(1, 0), n(0, 1), n(1, 1)];
                let m = |comp: usize| -> i64 {
                    (0..4)
                        .map(|q| wts[q] * i64::from(field.mv[ns[q]][comp]))
                        .sum()
                };
                let p = |at: usize, mm: i64| ((at as i64) << 6) + ((mm + (1 << (k - 1))) >> k);
                let (px, py) = (p(x, m(0)), p(y, m(1)));
                let (fx, fy) = ((px & 63) as usize, (py & 63) as usize);
                let mut acc = 0i64;
                for jj in 0..t {
                    let yy = ((py >> 6) - o + jj).clamp(0, h - 1) as usize;
                    let mut row = 0i64;
                    for ii in 0..t {
                        let xx = ((px >> 6) - o + ii).clamp(0, w - 1) as usize;
                        row += i64::from(reference.get(xx, yy)) * i64::from(taps[fx][ii as usize]);
                    }
                    acc += i64::from(taps[fy][jj as usize]) * row;
                }
                let s = ((acc + 8192) >> 14).clamp(0, 255);
                let weight: i64 = (0..4)
                    .map(|q| wts[q] * i64::from(!field.intra[ns[q]]))
                    .sum();
                let s = if weight < cs * cs {
                    128 + ((weight * (s - 128) + (cs * cs / 2)) >> (2 * c))
                } else {
                    s
                };
                out.set(x, y, s as u8);
            }
        }
    }

    fn random_plane(rng: &mut SplitMix64, w: usize, h: usize) -> Plane<u8> {
        let mut p = Plane::new(w, h);
        for (y, row) in p.rows_mut().enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                // Smooth structure plus noise, so interpolation errors are visible.
                *v = ((x * 7 + y * 3) % 200) as u8 + rng.below(56) as u8;
            }
        }
        p
    }

    #[test]
    fn taps_sum_to_one_and_phase_zero_is_identity() {
        for filter in Filter::ALL {
            let t = filter.table();
            let identity: &[i16] = if filter.taps() == 2 {
                &[128, 0]
            } else {
                &[0, 128, 0, 0]
            };
            if filter != Filter::Smooth {
                assert_eq!(&t[0][..filter.taps()], identity, "{filter:?}");
            }
            for (f, taps) in t.iter().enumerate() {
                assert_eq!(
                    taps.iter().map(|&v| i32::from(v)).sum::<i32>(),
                    128,
                    "{filter:?} phase {f}"
                );
            }
        }
        // Catmull-Rom at the half-pel phase: [-1, 9, 9, -1] / 16.
        assert_eq!(CUBIC[32], [-8, 72, 72, -8]);
        // The Mitchell–Netravali generator agrees with the Keys one on Catmull-Rom.
        assert_eq!(mitchell_taps(0, 1, 2), CUBIC);
        // The B-spline blurs whole-pixel positions: [1, 4, 1] / 6.
        assert_eq!(SMOOTH[0], [21, 86, 21, 0]);
    }

    #[test]
    fn matches_the_definition() {
        let mut rng = SplitMix64::new(3);
        for case in 0..120 {
            let (w, h) = (rng.range_usize(1, 150), rng.range_usize(1, 100));
            let grid = Grid::new(w, h, 4 + (case % 3) as u32);
            let mut field = MotionField::new(grid);
            let spread = [0, 4, 40, 400, 3000][case % 5];
            let base = [rng.range_i32(-200, 200), rng.range_i32(-200, 200)];
            for n in 0..grid.nodes() {
                field.mv[n] = [
                    base[0] + rng.range_i32(-spread, spread),
                    base[1] + rng.range_i32(-spread, spread),
                ];
                field.intra[n] = case % 4 == 1 && rng.below(3) == 0;
            }
            for chroma in [false, true] {
                let (pw, ph) = if chroma {
                    (w.div_ceil(2), h.div_ceil(2))
                } else {
                    (w, h)
                };
                let reference = random_plane(&mut rng, pw, ph);
                for filter in Filter::ALL {
                    let mut b = Plane::new(pw, ph);
                    warp_reference(&reference, &field, chroma, filter, &mut b);
                    for isa in Isa::supported() {
                        let mut a = Plane::new(pw, ph);
                        warp_plane(&reference, &field, chroma, filter, isa, &mut a);
                        assert!(
                            a.rows().eq(b.rows()),
                            "case {case}: {w}x{h}, {filter:?}, chroma {chroma}, {isa}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_isa_matches_scalar() {
        for isa in Isa::supported() {
            verify(isa, 60, 4).unwrap();
        }
    }

    #[test]
    fn integer_motion_is_an_exact_shift() {
        let mut rng = SplitMix64::new(9);
        let reference = random_plane(&mut rng, 96, 64);
        let mut field = MotionField::new(Grid::new(96, 64, 5));
        for mv in &mut field.mv {
            *mv = [3 * 8, -2 * 8]; // (+3, −2) pixels
        }
        for filter in Filter::ALL.into_iter().filter(|&f| f != Filter::Smooth) {
            let mut out = Plane::new(96, 64);
            warp_plane(&reference, &field, false, filter, Isa::best(), &mut out);
            for y in 0..64 {
                for x in 0..96 {
                    let (sx, sy) = ((x + 3).min(95), (y as i32 - 2).max(0) as usize);
                    assert_eq!(
                        out.get(x, y),
                        reference.get(sx, sy),
                        "{filter:?} ({x}, {y})"
                    );
                }
            }
        }
    }
}
