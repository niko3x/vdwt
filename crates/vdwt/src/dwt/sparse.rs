//! The inverse transform with zero-region skipping (DESIGN.md §6.5).
//!
//! Integer lifting maps all-zero input to exactly zero output: every predict and update step
//! adds a rounded average of zeros, which is zero. So wherever all the coefficients that reach a
//! region are zero, the region's output is zero and need not be computed. At low bitrates most
//! of a P-frame residual's coefficients are zero, and so are large areas of its output.
//!
//! Each level tracks which 8×8 blocks of its output may be nonzero. An output block depends on
//! the input coefficients of its own 4×4 coefficient block (in the low band and the three detail
//! bands) and, through the filter support of ±1 coefficient, on the neighbouring blocks; so it is
//! marked when any of the 3×3 neighbouring input blocks is. The next level's low-band input is
//! this level's output.
//!
//! The synthesis then runs only over column spans of marked blocks. A horizontal span is
//! synthesized with two extra coefficients on each side, so the lifting near the span's edge
//! sees real neighbours; the margins' (wrong) outputs fall inside the unmarked neighbour blocks,
//! which are then filled with zeros. Vertical lifting is element-wise and needs no margin.
//! The result is bit-identical to [`Dwt53::inverse`](super::Dwt53::inverse).

use super::kernels::Kernels;
use crate::Plane;

/// Which `2^log2 × 2^log2` blocks of a plane may hold nonzero values. Coordinates outside the
/// map read as clear.
#[derive(Clone, Debug, Default)]
pub struct BlockMap {
    w: usize,
    h: usize,
    log2: u32,
    bits: Vec<bool>,
}

impl BlockMap {
    /// Covers a `width × height` plane with clear blocks.
    pub fn reset(&mut self, width: usize, height: usize, log2: u32) {
        self.w = width.div_ceil(1 << log2);
        self.h = height.div_ceil(1 << log2);
        self.log2 = log2;
        self.bits.clear();
        self.bits.resize(self.w * self.h, false);
    }

    /// Size in blocks.
    pub fn dims(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Block size as log2.
    pub fn log2(&self) -> u32 {
        self.log2
    }

    #[inline]
    pub fn set(&mut self, bx: usize, by: usize) {
        self.bits[by * self.w + bx] = true;
    }

    #[inline]
    pub fn get(&self, bx: isize, by: isize) -> bool {
        bx >= 0
            && by >= 0
            && (bx as usize) < self.w
            && (by as usize) < self.h
            && self.bits[by as usize * self.w + bx as usize]
    }

    /// The blocks of row `by`.
    pub fn row(&self, by: usize) -> &[bool] {
        &self.bits[by * self.w..(by + 1) * self.w]
    }

    pub fn fill(&mut self, value: bool) {
        self.bits.fill(value);
    }

    pub fn count(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    /// Marks the `2^log2` blocks of `values` (a `width`-wide rectangle starting at row `y` of
    /// the region) that hold a nonzero value. Used row by row while dequantizing.
    #[inline]
    pub fn mark_row(&mut self, y: usize, values: &[i16]) {
        let by = y >> self.log2;
        for (bx, chunk) in values.chunks(1 << self.log2).enumerate() {
            if chunk.iter().any(|&v| v != 0) {
                self.set(bx, by);
            }
        }
    }
}

/// Maps of every level's output, reused between frames.
#[derive(Debug, Default)]
pub(crate) struct SparseScratch {
    maps: Vec<BlockMap>,
    spans: Vec<(usize, usize)>,
    prev_spans: Vec<(usize, usize)>,
}

impl SparseScratch {
    /// The final (level 1) output map after [`inverse_sparse`].
    pub fn output_map(&self) -> &BlockMap {
        &self.maps[0]
    }
}

/// Output blocks are 8×8 samples; band blocks are 4×4 coefficients.
pub(crate) const OUT_LOG2: u32 = 3;
pub(crate) const BAND_LOG2: u32 = 2;

/// Computes the output map of level `l`'s synthesis (`w × h` output) from its low band (`low`,
/// whose 4×4 block `(bx, by)` is read through `low_get`) and its three detail band maps.
fn level_map(
    out: &mut BlockMap,
    w: usize,
    h: usize,
    low_get: impl Fn(isize, isize) -> bool,
    details: [&BlockMap; 3],
) {
    out.reset(w, h, OUT_LOG2);
    let (bw, bh) = out.dims();
    for by in 0..bh as isize {
        for bx in 0..bw as isize {
            let hit = (-1..=1).any(|dy| {
                (-1..=1).any(|dx| {
                    let (x, y) = (bx + dx, by + dy);
                    low_get(x, y) || details.iter().any(|m| m.get(x, y))
                })
            });
            if hit {
                out.set(bx as usize, by as usize);
            }
        }
    }
}

/// Sparse inverse transform; see the module documentation. `bands` holds the 4×4-block map of
/// every band in [`Geometry::subbands`](super::Geometry::subbands) order. `ll_scratch` and
/// `line` are the engine's buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse_sparse(
    k: &Kernels,
    geo: super::Geometry,
    coeffs: &Plane<i16>,
    out: &mut Plane<i16>,
    ll_scratch: &mut Plane<i16>,
    line: &mut [i16],
    bands: &[BlockMap],
    s: &mut SparseScratch,
) {
    let levels = geo.levels();
    assert_eq!(bands.len(), 1 + 3 * levels, "one map per band");
    if s.maps.len() < levels.max(1) {
        s.maps.resize_with(levels.max(1), BlockMap::default);
    }
    if levels == 0 {
        out.copy_from(coeffs);
        s.maps[0].reset(out.width(), out.height(), OUT_LOG2);
        s.maps[0].fill(true);
        return;
    }
    // Maps from the coarsest level down: maps[l - 1] covers level l's output.
    for l in (1..=levels).rev() {
        let (w, h) = geo.ll_size(l - 1);
        let first = 1 + 3 * (levels - l);
        let details = [&bands[first], &bands[first + 1], &bands[first + 2]];
        let (lower, upper) = s.maps.split_at_mut(l);
        let out_map = &mut lower[l - 1];
        if l == levels {
            let ll = &bands[0];
            level_map(out_map, w, h, |x, y| ll.get(x, y), details);
        } else {
            // The low band is level l+1's output, mapped at 8×8: its 4×4 block (x, y) lies in
            // 8×8 block (x/2, y/2).
            let prev = &upper[0];
            level_map(
                out_map,
                w,
                h,
                |x, y| prev.get(x.div_euclid(2), y.div_euclid(2)),
                details,
            );
        }
    }
    let SparseScratch {
        maps,
        spans,
        prev_spans,
    } = s;
    for l in (1..=levels).rev() {
        let (w, h) = geo.ll_size(l - 1);
        let map = &maps[l - 1];
        if l % 2 == 1 {
            let ll = if l == levels { coeffs } else { &*ll_scratch };
            level(k, ll, coeffs, w, h, out, line, map, spans, prev_spans);
        } else {
            let ll = if l == levels { coeffs } else { &*out };
            level(
                k, ll, coeffs, w, h, ll_scratch, line, map, spans, prev_spans,
            );
        }
    }
}

/// Runs of marked blocks in block row `by`, as sample ranges clipped to `w`.
fn spans_of(map: &BlockMap, by: usize, w: usize, spans: &mut Vec<(usize, usize)>) {
    spans.clear();
    let row = map.row(by);
    let mut bx = 0;
    while bx < row.len() {
        if row[bx] {
            let start = bx;
            while bx < row.len() && row[bx] {
                bx += 1;
            }
            spans.push((start << OUT_LOG2, (bx << OUT_LOG2).min(w)));
        } else {
            bx += 1;
        }
    }
}

/// Horizontal synthesis of one row from `low` and `high` over `spans`; zero elsewhere.
fn synth_row(
    k: &Kernels,
    low: &[i16],
    high: &[i16],
    dst: &mut [i16],
    tmp: &mut [i16],
    spans: &[(usize, usize)],
) {
    let w = dst.len();
    match spans {
        [] => dst.fill(0),
        [(0, end)] if *end == w => (k.h_inverse)(low, high, dst, tmp),
        _ => {
            let (nl, nh) = (low.len(), high.len());
            for &(xa, xb) in spans {
                let n0 = (xa / 2).saturating_sub(2);
                let n1 = (xb.div_ceil(2) + 2).min(nl);
                if n1 == nl {
                    (k.h_inverse)(&low[n0..], &high[n0..nh], &mut dst[2 * n0..], tmp);
                } else {
                    (k.h_inverse)(&low[n0..n1], &high[n0..n1], &mut dst[2 * n0..2 * n1], tmp);
                }
            }
            let mut x = 0;
            for &(xa, xb) in spans {
                dst[x..xa].fill(0);
                x = xb;
            }
            dst[x..].fill(0);
        }
    }
}

/// One synthesis level restricted to the marked blocks of `map` (see `inverse_level` in the
/// parent module for the full version and its row schedule).
#[allow(clippy::too_many_arguments)]
fn level(
    k: &Kernels,
    ll: &Plane<i16>,
    coeffs: &Plane<i16>,
    w: usize,
    h: usize,
    dst: &mut Plane<i16>,
    tmp: &mut [i16],
    map: &BlockMap,
    spans: &mut Vec<(usize, usize)>,
    prev_spans: &mut Vec<(usize, usize)>,
) {
    let wl = w.div_ceil(2);
    let (ns, nd) = (h.div_ceil(2), h / 2);
    let mut block_row = usize::MAX;
    for i in 0..ns {
        // Rows 2i and 2i + 1 share an 8-row block.
        let br = (2 * i) >> OUT_LOG2;
        if br != block_row {
            std::mem::swap(spans, prev_spans);
            spans_of(map, br, w, spans);
            block_row = br;
        }
        synth_row(
            k,
            ll.row_span(i, 0, wl),
            coeffs.row_span(i, wl, w - wl),
            dst.row_span_mut(2 * i, 0, w),
            tmp,
            spans,
        );
        if i < nd {
            synth_row(
                k,
                coeffs.row_span(ns + i, 0, wl),
                coeffs.row_span(ns + i, wl, w - wl),
                dst.row_span_mut(2 * i + 1, 0, w),
                tmp,
                spans,
            );
        }
        if nd > 0 {
            let above = if i == 0 { 1 } else { 2 * i - 1 };
            let below = if i < nd { 2 * i + 1 } else { 2 * i - 1 };
            for &(xa, xb) in spans.iter() {
                let (s, d0, d1) = dst.row_span_mut3(2 * i, above, below, xa, xb - xa);
                (k.v_unupdate)(s, d0, d1);
            }
            if i > 0 {
                // Row 2i − 1 may belong to the previous block row.
                let row_spans = if (2 * i - 1) >> OUT_LOG2 == br {
                    &*spans
                } else {
                    &*prev_spans
                };
                for &(xa, xb) in row_spans {
                    let (d, s0, s1) = dst.row_span_mut3(2 * i - 1, 2 * i - 2, 2 * i, xa, xb - xa);
                    (k.v_unpredict)(d, s0, s1);
                }
            }
        }
    }
    if nd > 0 && nd == ns {
        // Even height: the last odd row's lower neighbour x[h] mirrors to x[h−2].
        let i = nd - 1;
        for &(xa, xb) in spans.iter() {
            let (d, s0, s1) = dst.row_span_mut3(2 * i + 1, 2 * i, 2 * i, xa, xb - xa);
            (k.v_unpredict)(d, s0, s1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::{Dwt53, Geometry};
    use crate::rng::SplitMix64;
    use crate::{Isa, Plane};

    /// Band maps of `coeffs`, exactly (every block with a nonzero value is marked).
    fn band_maps(coeffs: &Plane<i16>, levels: usize) -> Vec<BlockMap> {
        let geo = Geometry::new(coeffs.width(), coeffs.height(), levels);
        geo.subbands()
            .iter()
            .map(|b| {
                let mut m = BlockMap::default();
                m.reset(b.width, b.height, BAND_LOG2);
                for y in 0..b.height {
                    m.mark_row(y, coeffs.row_span(b.y0 + y, b.x0, b.width));
                }
                m
            })
            .collect()
    }

    #[test]
    fn matches_the_full_inverse() {
        let mut rng = SplitMix64::new(17);
        for case in 0..400 {
            let (w, h) = (rng.range_usize(1, 140), rng.range_usize(1, 100));
            let levels = rng.range_usize(0, 5);
            // Sparse clusters of nonzero coefficients, as a low-rate residual has.
            let mut coeffs = Plane::<i16>::new(w, h);
            let clusters = [0, 1, 3, 10, 40][case % 5];
            for _ in 0..clusters {
                let (cx, cy) = (rng.below(w as u64) as usize, rng.below(h as u64) as usize);
                let r = rng.range_usize(0, 6);
                for y in cy.saturating_sub(r)..(cy + r + 1).min(h) {
                    for x in cx.saturating_sub(r)..(cx + r + 1).min(w) {
                        if rng.below(3) == 0 {
                            coeffs.set(x, y, rng.range_i32(-900, 900) as i16);
                        }
                    }
                }
            }
            let mut maps = band_maps(&coeffs, levels);
            // Extra marks (conservative maps) must not change the result.
            if case % 3 == 0 {
                for m in &mut maps {
                    let (bw, bh) = m.dims();
                    if bw * bh > 0 && rng.below(2) == 0 {
                        m.set(rng.below(bw as u64) as usize, rng.below(bh as u64) as usize);
                    }
                }
            }
            for isa in Isa::supported() {
                let mut dwt = Dwt53::with_isa(w, h, isa).unwrap();
                let mut full = Plane::new(w, h);
                dwt.inverse(&coeffs, levels, &mut full);
                // Garbage in the output buffer must not leak into the result.
                let mut sparse = Plane::new(w, h);
                sparse.fill(-77);
                let map = dwt
                    .inverse_sparse(&coeffs, levels, &mut sparse, &maps)
                    .clone();
                assert!(
                    sparse.rows().eq(full.rows()),
                    "case {case}: {w}x{h}, {levels} levels, {isa}"
                );
                for y in 0..h {
                    for x in 0..w {
                        if full.get(x, y) != 0 {
                            assert!(map.get((x >> 3) as isize, (y >> 3) as isize));
                        }
                    }
                }
            }
        }
    }
}
