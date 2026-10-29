//! Overlapped motion compensation, the alternative to the warp (DESIGN.md §6.3). Normative, exact
//! integer arithmetic.
//!
//! The warp interpolates the four node vectors of a pixel's cell and samples the reference once.
//! Here every node's vector predicts by plain translation, and the pixel blends the four
//! translated samples with the same bilinear weights:
//!
//! ```text
//! s_n = clamp((Σ_j h[fy_n][j] · Σ_i h[fx_n][i] · R(ix_n − o + i, iy_n − o + j) + 2^11) >> 12, 0, 1023)
//! out = clamp((Σ_n w_n · s_n + 2^(2c+1)) >> (2c + 2), 0, 255)
//! w_n = (C−u)(C−v), u(C−v), (C−u)v, uv                                    (sum C² = 2^2c)
//! ```
//!
//! with node `n`'s position `64·x + (m_n << (6 − f))` (its vector in 1/64 pixel) and coordinates
//! clamped to the plane: `s_n` is a translational prediction with two fractional bits. An intra
//! node contributes mid-grey (`s_n = 512`), which is the warp's fade to grey. Rounding each `s_n`
//! to 10 bits keeps the blend in 16-bit (horizontally) and 32-bit arithmetic; rounding to 8 bits
//! cost 0.04–0.07 dB, mostly in chroma.
//!
//! Neighbouring nodes may move independently: where the motion is discontinuous (an object edge,
//! parallax), the prediction cross-fades between two correct translations instead of stretching
//! the picture between them.
//!
//! **Kernels.** Each node's translation over a cell is an ordinary separable filter with constant
//! taps: a horizontal pass over the cell's rows plus `T − 1`, then a vertical pass. Groups of 8
//! pixels run on SIMD kernels (`obmc_x86`, `obmc_neon`) when the cell is a multiple of 8 wide
//! and its reference columns lie inside the plane; everything else takes the portable code.
//! Samples are centred on 128 so that the horizontal sums fit 16 bits, which changes nothing:
//! the taps sum to 128, so centring subtracts exactly `512 << 12` before the final shift. Nodes
//! of a cell with the same vector (or both intra) share one translation.

use super::MotionField;
use super::warp::{Filter, PHASE_BITS, PHASES, SAMPLE_ROWS, Taps, sampled};
use crate::plane::RowsMut;
use crate::{Isa, Plane};

/// SIMD kernels, working on groups of 8 pixels.
#[derive(Clone, Copy)]
pub(super) struct Kernels {
    /// A node's translation over a block `len` (a multiple of 8) wide: the reference rows the
    /// vertical taps read are `src` rows `iy0 ..` (`out.len() / len + taps − 1` of them),
    /// clamped to `0..height`, `stride` apart, each holding `len + taps − 1` samples from
    /// column `ix`. Horizontal sums of centred samples go to `tmp` (one row of `len` per
    /// reference row), then `out = clamp(((Σ_j hy[j]·tmp[r + j] + 2^11) >> 12) + 512, 0, 1023)`
    /// per output row `r`.
    #[allow(clippy::type_complexity)]
    pub translate: unsafe fn(
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
    ),
    /// The blend of a block `len` (a multiple of 8) wide and `rows` high, rows `v0 ..` of a
    /// cell of `cs ≤ 32` pixels: `out = min(((cs−v)·top + v·bottom + 2^(shift−1)) >> shift, 255)`
    /// with `top = (cs−u)·s[0] + u·s[1]` and `bottom = (cs−u)·s[2] + u·s[3]`. Each `s[q]` is
    /// `(samples, row stride)`; output rows are `stride` apart in `dst`.
    #[allow(clippy::type_complexity)]
    pub blend: unsafe fn(
        s: &[(&[u16], usize); 4],
        len: usize,
        rows: usize,
        v0: u32,
        cs: u32,
        shift: u32,
        dst: &mut [u8],
        stride: usize,
    ),
}

fn kernels(isa: Isa) -> Option<Kernels> {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => super::obmc_x86::kernels(),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => Some(super::obmc_neon::KERNELS),
        _ => None,
    }
}

/// The [`sampled`] rows of `band` (a band of a plane the size of `reference`).
/// `height` is the reference rows reads may reach: the plane's height, or less for a virtual
/// boundary (reads below it repeat its last row, as at the plane's edge).
#[allow(clippy::too_many_arguments)]
pub(super) fn obmc_band(
    reference: &Plane<u8>,
    height: usize,
    field: &MotionField,
    chroma: bool,
    filter: Filter,
    isa: Isa,
    row_step: usize,
    band: &mut RowsMut<'_, u8>,
) {
    let simd = kernels(isa);
    match filter.taps() {
        2 => obmc::<2>(
            reference,
            height,
            field,
            chroma,
            filter.table(),
            simd,
            row_step,
            band,
        ),
        _ => obmc::<4>(
            reference,
            height,
            field,
            chroma,
            filter.table(),
            simd,
            row_step,
            band,
        ),
    }
}

/// Reusable buffers of one band's prediction.
#[derive(Default)]
struct Scratch {
    /// Rows with clamped coordinates, for a translation reaching past the plane's sides.
    edge: Vec<u8>,
    /// Nodes `k` and `k + 1` of the cell row's top and bottom node rows, in slots `k mod 2`:
    /// each one's translation over the cells on either side of it, and the first column that
    /// covers. Computed just ahead of the blend, so they stay in cache.
    nodes: [[(Vec<u16>, usize); 2]; 2],
    /// Horizontal sums.
    tmp: Vec<i16>,
    /// Portable vertical pass accumulators.
    acc: Vec<i32>,
}

#[allow(clippy::too_many_arguments)]
fn obmc<const T: usize>(
    reference: &Plane<u8>,
    height: usize,
    field: &MotionField,
    chroma: bool,
    taps: &Taps,
    simd: Option<Kernels>,
    row_step: usize,
    out: &mut RowsMut<'_, u8>,
) {
    let c = field.grid.log2 - u32::from(chroma);
    let f = 3 + u32::from(chroma);
    let cs = 1usize << c;
    let w = reference.width();
    let nx = field.grid.nx;
    let shift = 2 * c + 2;
    let cells = w.div_ceil(cs);
    let rows = out.rows();
    let wanted = |y: usize| sampled(y, row_step);
    let mut sc = Scratch::default();
    // Cells, by row of cells, intersected with this band.
    let mut y = rows.start;
    while y < rows.end {
        let j = y >> c;
        // A whole cell's rows at once, or, when sampling, at most one span.
        let mut y1 = ((j + 1) << c).min(rows.end);
        if row_step > 1 {
            y1 = y1.min((y / SAMPLE_ROWS + 1) * SAMPLE_ROWS);
        }
        let (ya, yb) = (y, y1);
        y = y1;
        if !(ya..yb).any(wanted) {
            continue;
        }
        let n_rows = yb - ya;
        // Node `i` of node row `j + side`, over the cells left and right of it.
        let node = |sc: &mut Scratch, side: usize, i: usize| {
            let n = (j + side) * nx + i;
            let (xa, xb) = ((i * cs).saturating_sub(cs), ((i + 1) * cs).min(w));
            let (s, first) = &mut sc.nodes[side][i % 2];
            *first = xa;
            // Every value is written below.
            s.resize((xb - xa) * n_rows, 0);
            if field.intra[n] {
                s.fill(512);
            } else {
                let d = field.mv[n].map(|m| m << (PHASE_BITS - f));
                translate::<T>(
                    reference,
                    height,
                    xa,
                    ya..yb,
                    d,
                    taps,
                    simd,
                    &mut sc.edge,
                    &mut sc.tmp,
                    &mut sc.acc,
                    s,
                );
            }
        };
        node(&mut sc, 0, 0);
        node(&mut sc, 1, 0);
        let cs = cs as u32;
        let v0 = (ya & (cs as usize - 1)) as u32;
        for i in 0..cells {
            node(&mut sc, 0, i + 1);
            node(&mut sc, 1, i + 1);
            let x0 = i * cs as usize;
            let len = (cs as usize).min(w - x0);
            // A node's samples for this cell: from the cell's first column, with the node's
            // row stride.
            let at = |side: usize, k: usize| {
                let (s, first) = &sc.nodes[side][k % 2];
                let width = s.len() / n_rows;
                (&s[x0 - first..], width)
            };
            let s = [at(0, i), at(0, i + 1), at(1, i), at(1, i + 1)];
            // 16-bit products need cs ≤ 32.
            let simd_len = simd.filter(|_| cs <= 32).map_or(0, |_| len & !7);
            if let Some(kern) = simd.filter(|_| simd_len > 0) {
                let stride = out.stride();
                let dst = out.block_mut(ya, x0, simd_len, n_rows);
                // SAFETY: the kernels' CPU features were checked when they were selected; every
                // node holds `n_rows` rows of at least `len ≥ simd_len` values from `x0`.
                unsafe { (kern.blend)(&s, simd_len, n_rows, v0, cs, shift, dst, stride) };
            }
            if simd_len == len {
                continue;
            }
            for yy in (ya..yb).filter(|&yy| wanted(yy)) {
                let v = (yy & (cs as usize - 1)) as u32;
                let r = yy - ya;
                let dst = &mut out.row_mut(yy)[x0..x0 + len];
                let row = |q: usize| &s[q].0[r * s[q].1..r * s[q].1 + len];
                let s = [row(0), row(1), row(2), row(3)];
                blend_portable(&s, simd_len, cs, v, shift, dst);
            }
        }
    }
}

/// [`Kernels::blend`] in portable code, from pixel `from` on.
fn blend_portable(s: &[&[u16]; 4], from: usize, cs: u32, v: u32, shift: u32, dst: &mut [u8]) {
    for (u, d) in dst.iter_mut().enumerate().skip(from) {
        let ui = u as u32;
        // Separably: the cell's top and bottom edges, then between them.
        let top = (cs - ui) * u32::from(s[0][u]) + ui * u32::from(s[1][u]);
        let bottom = (cs - ui) * u32::from(s[2][u]) + ui * u32::from(s[3][u]);
        *d = (((cs - v) * top + v * bottom + (1 << (shift - 1))) >> shift).min(255) as u8;
    }
}

/// Rows `rows`, pixels `x0 .. x0 + len` (`out` holds `len` per row), of `reference` translated
/// by `d` (1/64 pixel), coordinates clamped to the plane. Each reference row is filtered
/// horizontally once, into `tmp`.
#[allow(clippy::too_many_arguments)]
fn translate<const T: usize>(
    reference: &Plane<u8>,
    height: usize,
    x0: usize,
    rows: std::ops::Range<usize>,
    d: [i32; 2],
    taps: &Taps,
    simd: Option<Kernels>,
    edge: &mut Vec<u8>,
    tmp: &mut Vec<i16>,
    acc: &mut Vec<i32>,
    out: &mut [u16],
) {
    let (w, h) = (
        reference.width() as i32,
        height.min(reference.height()) as i32,
    );
    let o = (T / 2 - 1) as i32;
    let mask = PHASES as i32 - 1;
    let (hx, hy) = (&taps[(d[0] & mask) as usize], &taps[(d[1] & mask) as usize]);
    let len = out.len() / rows.len();
    let ix = x0 as i32 + (d[0] >> PHASE_BITS) - o;
    let iy0 = rows.start as i32 + (d[1] >> PHASE_BITS) - o;
    let inside = ix >= 0 && ix + len as i32 + T as i32 - 1 <= w;
    let n = rows.len() + T - 1;
    if let Some(k) = simd.filter(|_| len.is_multiple_of(8)) {
        let span = len + T - 1;
        // Every value is written by the kernel.
        tmp.resize(n * len, 0);
        let row = |r: usize| reference.row((iy0 + r as i32).clamp(0, h - 1) as usize);
        let tmp = &mut tmp[..n * len];
        if inside {
            // SAFETY: CPU features checked on selection; `inside` keeps `len + T − 1` samples
            // from `ix` in every row, `len` is a multiple of 8, and `tmp` holds `n · len` values.
            unsafe {
                (k.translate)(
                    reference.data(),
                    reference.stride(),
                    h as usize,
                    iy0 as isize,
                    ix as usize,
                    hx,
                    hy,
                    T,
                    tmp,
                    out,
                )
            };
        } else {
            // Rows with their columns clamped (edge extension), then the same kernel.
            edge.clear();
            for r in 0..n {
                let row = row(r);
                edge.extend((0..span).map(|u| row[(ix + u as i32).clamp(0, w - 1) as usize]));
            }
            // SAFETY: as above, with every padded row holding `len + T − 1` samples.
            unsafe { (k.translate)(edge, span, n, 0, 0, hx, hy, T, tmp, out) };
        }
        return;
    }
    // Horizontal pass over the rows.len() + T − 1 reference rows the vertical taps read, centred
    // on 128.
    tmp.clear();
    tmp.resize(n * len, 0);
    for (r, t) in tmp.chunks_exact_mut(len).enumerate() {
        let row = reference.row((iy0 + r as i32).clamp(0, h - 1) as usize);
        if inside {
            let row = &row[ix as usize..ix as usize + len + T - 1];
            for (u, t) in t.iter_mut().enumerate() {
                *t = (0..T)
                    .map(|k| (i32::from(row[u + k]) - 128) * i32::from(hx[k]))
                    .sum::<i32>() as i16;
            }
        } else {
            for (u, t) in t.iter_mut().enumerate() {
                *t = (0..T)
                    .map(|k| {
                        let x = (ix + (u + k) as i32).clamp(0, w - 1) as usize;
                        (i32::from(row[x]) - 128) * i32::from(hx[k])
                    })
                    .sum::<i32>() as i16;
            }
        }
    }
    // Vertical pass.
    acc.clear();
    acc.resize(len, 0);
    for (r, o) in out.chunks_exact_mut(len).enumerate() {
        acc.fill(0);
        for (jj, &wy) in hy.iter().take(T).enumerate() {
            let t = &tmp[(r + jj) * len..(r + jj + 1) * len];
            for (a, &t) in acc.iter_mut().zip(t) {
                *a += i32::from(wy) * i32::from(t);
            }
        }
        for (o, &a) in o.iter_mut().zip(acc.iter()) {
            *o = (((a + (1 << 11)) >> 12) + 512).clamp(0, 1023) as u16;
        }
    }
}
