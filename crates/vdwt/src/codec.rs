//! Frame layout shared by the encoder and the decoder: components, the band coding order,
//! context classes, steps and LL prediction.

use crate::bitstream::SequenceHeader;
use crate::dwt::{BlockMap, Geometry, OUT_LOG2, Orient, Subband};
use crate::entropy::trained::TRAINED;
use crate::entropy::{Contexts, FLAT_GUIDE_CLASS, Guide, Rect, SigPyramid};
use crate::quant::step_q12;
use crate::{Plane, spare};

/// One band, in coding order.
#[derive(Clone, Debug)]
pub(crate) struct BandSlot {
    pub comp: usize,
    pub sb: Subband,
    /// Index into the component's weight list ([`Geometry::subbands`] order).
    pub weight_index: usize,
    /// Intra context class: component group × band type. P-frames add [`FRAME_CLASSES`].
    pub class: usize,
    /// The same-orientation band one level coarser, as a slot index. It is always coded earlier.
    pub parent: Option<usize>,
}

impl BandSlot {
    pub fn rect(&self) -> Rect {
        Rect {
            x0: self.sb.x0,
            y0: self.sb.y0,
            w: self.sb.width,
            h: self.sb.height,
        }
    }

    /// Whether the band codes with transposed sign contexts (an HL band, see `entropy::band`).
    pub fn transposed(&self) -> bool {
        self.sb.orient == Orient::HL
    }

    pub fn is_ll(&self) -> bool {
        self.sb.orient == Orient::LL
    }

    /// The band in its own plane (its quantized values, or a predicted LL band's residuals):
    /// at the origin.
    pub fn local_rect(&self) -> Rect {
        Rect {
            x0: 0,
            y0: 0,
            w: self.sb.width,
            h: self.sb.height,
        }
    }

    /// Context class in an intra frame or a P-frame.
    pub fn coding_class(&self, inter: bool) -> usize {
        self.class + if inter { FRAME_CLASSES } else { 0 }
    }

    /// Quantization step in Q12 at quality `qp`.
    pub fn step_q12(&self, seq: &SequenceHeader, qp: u8) -> u64 {
        let weights = if self.comp == 0 {
            &seq.luma_weights
        } else {
            &seq.chroma_weights
        };
        step_q12(qp, weights[self.weight_index], seq.wavelet.frac_bits())
    }
}

pub(crate) fn component_size(seq: &SequenceHeader, comp: usize) -> (usize, usize) {
    if comp == 0 {
        (seq.width, seq.height)
    } else {
        seq.chroma_size()
    }
}

pub(crate) fn component_levels(seq: &SequenceHeader, comp: usize) -> usize {
    if comp == 0 {
        seq.luma_levels
    } else {
        seq.chroma_levels
    }
}

/// Band types with their own context set: LL, then (HL or LH, HH) at level 1, level 2 and deeper.
const BAND_CLASSES: usize = 7;
/// Luma and chroma have separate classes.
pub(crate) const FRAME_CLASSES: usize = 2 * BAND_CLASSES;
/// Intra classes, then the same set again for P-frame residuals.
pub(crate) const CLASSES: usize = 2 * FRAME_CLASSES;

fn class_of(comp: usize, sb: &Subband) -> usize {
    let band = match sb.orient {
        Orient::LL => 0,
        orient => 1 + 2 * (sb.level.min(3) - 1) + usize::from(orient == Orient::HH),
    };
    usize::from(comp > 0) * BAND_CLASSES + band
}

/// Starting contexts for every class: the trained tables (`entropy/trained.rs`) over
/// [`untrained_contexts`].
pub(crate) fn default_contexts() -> Vec<Contexts> {
    untrained_contexts()
        .into_iter()
        .zip(TRAINED)
        .take(CLASSES)
        .map(|(c, t)| c.with_trained(t))
        .collect()
}

/// The hand-set starting contexts of every context set (band classes, then the motion planes):
/// intra LL bands are dense; residual LL bands and everything else are not.
pub(crate) fn untrained_contexts() -> Vec<Contexts> {
    (0..crate::training::SETS)
        .map(|c| {
            let ctx = Contexts::new(c < FRAME_CLASSES && c % BAND_CLASSES == 0);
            if (FRAME_CLASSES..2 * FRAME_CLASSES).contains(&c) {
                ctx.for_residuals()
            } else {
                ctx
            }
        })
        .collect()
}

/// Every band of a frame in coding order. Resolutions go coarsest first, so a band's parent is
/// always available. Within a resolution the order is Y, U, V, and within a component HL, LH, HH.
/// With 4:2:0 and one fewer chroma level, each resolution pairs luma and chroma bands of the
/// same spatial frequency.
pub(crate) fn band_slots(seq: &SequenceHeader) -> Vec<BandSlot> {
    let subbands: Vec<Vec<Subband>> = (0..3)
        .map(|comp| {
            let (w, h) = component_size(seq, comp);
            Geometry::new(w, h, component_levels(seq, comp)).subbands()
        })
        .collect();
    let mut slots: Vec<BandSlot> = Vec::new();
    for res in 0..=seq.luma_levels.max(seq.chroma_levels) {
        for (comp, bands) in subbands.iter().enumerate() {
            let levels = component_levels(seq, comp);
            if res > levels {
                continue;
            }
            let indices = if res == 0 {
                0..1
            } else {
                1 + 3 * (res - 1)..1 + 3 * res
            };
            for i in indices {
                let sb = bands[i];
                let parent = (sb.orient != Orient::LL && sb.level < levels)
                    .then(|| {
                        slots.iter().position(|s| {
                            s.comp == comp && s.sb.level == sb.level + 1 && s.sb.orient == sb.orient
                        })
                    })
                    .flatten();
                slots.push(BandSlot {
                    comp,
                    sb,
                    weight_index: i,
                    class: class_of(comp, &sb),
                    parent,
                });
            }
        }
    }
    slots
}

/// One plane per band for its quantized values, in coding order.
pub(crate) fn band_planes(slots: &[BandSlot]) -> Vec<Plane<i32>> {
    slots
        .iter()
        .map(|s| Plane::new(s.sb.width.max(1), s.sb.height.max(1)))
        .collect()
}

/// Side of [`zero_isolated`]'s blocks.
const ISOLATED_BLOCK: usize = 4;

/// Encoder maps of a band made while quantizing it, row by row: its nonzero 2×2 blocks (the
/// first level of its [`SigPyramid`]) and its nonzero values per [`zero_isolated`] block.
#[derive(Default)]
pub(crate) struct BandMaps {
    /// 2×2 blocks per row, and 4×4 blocks per row.
    w1: usize,
    w4: usize,
    h1: usize,
    pub sig1: Vec<bool>,
    count: Vec<u8>,
}

impl Drop for BandMaps {
    fn drop(&mut self) {
        spare::give(std::mem::take(&mut self.sig1));
        spare::give(std::mem::take(&mut self.count));
    }
}

impl BandMaps {
    pub fn new(r: Rect) -> Self {
        let (w1, w4) = (r.w.div_ceil(2), r.w.div_ceil(ISOLATED_BLOCK));
        Self {
            w1,
            w4,
            h1: r.h.div_ceil(2),
            sig1: spare::take_zeroed(w1 * r.h.div_ceil(2)),
            count: spare::take_zeroed(w4 * r.h.div_ceil(ISOLATED_BLOCK)),
        }
    }

    /// [`Self::new`] in the buffers this one holds.
    pub fn reset(&mut self, r: Rect) {
        let (w1, w4) = (r.w.div_ceil(2), r.w.div_ceil(ISOLATED_BLOCK));
        (self.w1, self.w4, self.h1) = (w1, w4, r.h.div_ceil(2));
        self.sig1.clear();
        self.sig1.resize(w1 * r.h.div_ceil(2), false);
        self.count.clear();
        self.count.resize(w4 * r.h.div_ceil(ISOLATED_BLOCK), 0);
    }

    /// Adds row `y` of the band's values.
    #[inline]
    pub fn add_row(&mut self, y: usize, row: &[i32]) {
        let k = y / ISOLATED_BLOCK;
        let counts = &mut self.count[k * self.w4..][..self.w4];
        let sig = &mut self.sig1[(y / 2) * self.w1..][..self.w1];
        for (b, chunk) in row.chunks(ISOLATED_BLOCK).enumerate() {
            if chunk.iter().all(|&v| v == 0) {
                continue;
            }
            counts[b] += chunk.iter().filter(|&&v| v != 0).count() as u8;
            for (o, pair) in sig[2 * b..].iter_mut().zip(chunk.chunks(2)) {
                *o |= pair.iter().any(|&v| v != 0);
            }
        }
    }

    /// Zeroes isolated values ([`zero_isolated`]), keeping the 2×2 map current.
    pub fn zero_isolated(
        &mut self,
        values: &mut Plane<i32>,
        coeffs: &Plane<i16>,
        r: Rect,
        step_q12: u64,
        threshold: u64,
        strict: bool,
    ) -> u32 {
        let mut before = spare::take(self.count.len());
        before.copy_from_slice(&self.count);
        let zeroed = zero_isolated(
            values,
            coeffs,
            r,
            step_q12,
            threshold,
            &mut self.count,
            strict,
        );
        if zeroed > 0 {
            // A zeroed block is empty now: so are its 2×2 blocks.
            let b = ISOLATED_BLOCK / 2;
            for (k, _) in before
                .iter()
                .zip(&self.count)
                .enumerate()
                .filter(|(_, (b, a))| b != a)
            {
                let (bx, by) = (k % self.w4, k / self.w4);
                for y in by * b..((by + 1) * b).min(self.h1) {
                    for x in bx * b..((bx + 1) * b).min(self.w1) {
                        self.sig1[y * self.w1 + x] = false;
                    }
                }
            }
        }
        spare::give(before);
        zeroed
    }

    /// The band's significance pyramid.
    pub fn pyramid(&self, max_log2: u32) -> SigPyramid {
        let mut first = spare::take(self.sig1.len());
        first.copy_from_slice(&self.sig1);
        SigPyramid::from_level1(first, self.w1, max_log2)
    }
}

/// Encoder: zeroes every ±1 in band `values` that is alone in its 4×4 block, with the 8
/// surrounding blocks empty or the other three blocks of its 8×8 quadtree node empty, and whose
/// coefficient (at rectangle `r` of `coeffs`) is below `threshold` quantizer steps (both in
/// Q12). Coding such a value costs a dozen bits or more of block and node flags; when it barely
/// cleared the dead zone, the distortion it removes is worth less than that. A zeroed block
/// counts as empty for the blocks after it. `count` holds the nonzero values per block (from
/// [`BandMaps::add_row`]) and is kept current. Returns how many values it zeroed.
fn zero_isolated(
    values: &mut Plane<i32>,
    coeffs: &Plane<i16>,
    r: Rect,
    step_q12: u64,
    threshold: u64,
    count: &mut [u8],
    strict: bool,
) -> u32 {
    const B: usize = ISOLATED_BLOCK;
    let (bw, bh) = (r.w.div_ceil(B), r.h.div_ceil(B));
    let empty = |count: &[u8], bx: isize, by: isize| {
        bx < 0
            || by < 0
            || bx as usize >= bw
            || by as usize >= bh
            || count[by as usize * bw + bx as usize] == 0
    };
    let limit = threshold * step_q12;
    let mut zeroed = 0;
    for by in 0..bh {
        for bx in 0..bw {
            if count[by * bw + bx] != 1 {
                continue;
            }
            if strict {
                let (x, y) = (bx as isize, by as isize);
                let alone = (-1..=1)
                    .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                    .filter(|&d| d != (0, 0))
                    .all(|(dx, dy)| empty(count, x + dx, y + dy));
                // Or alone in its 8×8 quadtree node: then its whole node goes too.
                let (gx, gy) = (x & !1, y & !1);
                let alone_in_node = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .filter(|&&(dx, dy)| (gx + dx, gy + dy) != (x, y))
                    .all(|&(dx, dy)| empty(count, gx + dx, gy + dy));
                if !alone && !alone_in_node {
                    continue;
                }
            }
            let (x0, w) = (bx * B, B.min(r.w - bx * B));
            for yy in by * B..((by + 1) * B).min(r.h) {
                let row = values.row_span_mut(yy, x0, w);
                let src = coeffs.row_span(r.y0 + yy, r.x0 + x0, w);
                for (v, &c) in row.iter_mut().zip(src) {
                    if v.abs() == 1 && u64::from(c.unsigned_abs()) << 24 < limit {
                        *v = 0;
                        zeroed += 1;
                        count[by * bw + bx] = 0;
                    }
                }
            }
        }
    }
    zeroed
}

/// Bands that are some other band's parent.
pub(crate) fn has_child(slots: &[BandSlot]) -> Vec<bool> {
    let mut has = vec![false; slots.len()];
    for p in slots.iter().filter_map(|s| s.parent) {
        has[p] = true;
    }
    has
}

/// Band maps for one component, in [`Geometry::subbands`] order (indexed by
/// [`BandSlot::weight_index`]).
pub(crate) fn band_maps(seq: &SequenceHeader, comp: usize) -> Vec<BlockMap> {
    vec![BlockMap::default(); 1 + 3 * component_levels(seq, comp)]
}

/// P-frame residual: `cur − pred`, in the signed form the transform works on.
pub(crate) fn residual(cur: &Plane<u8>, pred: &Plane<u8>, out: &mut Plane<i16>) {
    for ((d, c), p) in out.rows_mut().zip(cur.rows()).zip(pred.rows()) {
        for ((d, &c), &p) in d.iter_mut().zip(c).zip(p) {
            *d = i16::from(c) - i16::from(p);
        }
    }
}

/// P-frame reconstruction: adds the decoded residual to the prediction in `recon`, clamping to
/// 8 bits, in the 8×8 blocks `map` marks. Elsewhere the residual is zero and the prediction
/// stands.
pub(crate) fn add_residual(res: &Plane<i16>, recon: &mut Plane<u8>, map: &BlockMap) {
    let (w, h) = (recon.width(), recon.height());
    debug_assert_eq!(map.log2(), OUT_LOG2);
    for by in 0..map.dims().1 {
        let blocks = map.row(by);
        let mut bx = 0;
        while bx < blocks.len() {
            if !blocks[bx] {
                bx += 1;
                continue;
            }
            let start = bx;
            while bx < blocks.len() && blocks[bx] {
                bx += 1;
            }
            let (x0, x1) = (start << OUT_LOG2, (bx << OUT_LOG2).min(w));
            for y in by << OUT_LOG2..((by + 1) << OUT_LOG2).min(h) {
                let d = res.row_span(y, x0, x1 - x0);
                for (r, &d) in recon.row_span_mut(y, x0, x1 - x0).iter_mut().zip(d) {
                    // Saturating: a corrupt stream can carry any residual.
                    *r = i16::from(*r).saturating_add(d).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// The LOCO-I median edge predictor.
fn med(a: i32, b: i32, c: i32) -> i32 {
    let (lo, hi) = (a.min(b), a.max(b));
    if c >= hi {
        lo
    } else if c <= lo {
        hi
    } else {
        a + b - c
    }
}

/// Prediction for LL sample `(x, y)` of the band at `r` in `q`, from samples before it in raster
/// order. It always lies between two neighbours, so residuals of `i16`-range values fit 17 bits.
fn ll_prediction(q: &Plane<i32>, r: Rect, x: usize, y: usize) -> i32 {
    let at = |x: usize, y: usize| q.get(r.x0 + x, r.y0 + y);
    match (x, y) {
        (0, 0) => 0,
        (_, 0) => at(x - 1, 0),
        (0, _) => at(0, y - 1),
        _ => med(at(x - 1, y), at(x, y - 1), at(x - 1, y - 1)),
    }
}

/// Nonzero count and summed bit lengths of the prediction residuals of a `w`-wide raster of
/// quantized LL values, for rate estimation.
pub(crate) fn ll_residual_stats(q: &[i32], w: usize) -> (u32, u64) {
    let (mut nonzero, mut bits) = (0u32, 0u64);
    for (i, &v) in q.iter().enumerate() {
        let (x, y) = (i % w, i / w);
        let pred = match (x, y) {
            (0, 0) => 0,
            (_, 0) => q[i - 1],
            (0, _) => q[i - w],
            _ => med(q[i - 1], q[i - w], q[i - w - 1]),
        };
        let m = (v - pred).unsigned_abs();
        nonzero += u32::from(m != 0);
        bits += u64::from(32 - m.leading_zeros());
    }
    (nonzero, bits)
}

/// Encoder: residuals of the LL band at `r` in `q`, written to the origin of `res`.
pub(crate) fn ll_residuals(q: &Plane<i32>, r: Rect, res: &mut Plane<i32>) {
    for y in 0..r.h {
        for x in 0..r.w {
            res.set(x, y, q.get(r.x0 + x, r.y0 + y) - ll_prediction(q, r, x, y));
        }
    }
}

/// Decoder: rebuilds the LL band at `r` in `q` from the residuals at the origin of `res`. The
/// clamp to the `i16` range is a no-op for valid streams and keeps corrupt ones bounded.
pub(crate) fn ll_reconstruct(res: &Plane<i32>, r: Rect, q: &mut Plane<i32>) {
    for y in 0..r.h {
        for x in 0..r.w {
            let v = res.get(x, y) + ll_prediction(q, r, x, y);
            q.set(
                r.x0 + x,
                r.y0 + y,
                v.clamp(-i32::from(i16::MAX), i32::from(i16::MAX)),
            );
        }
    }
}

/// The guide of a P-frame band from the prediction's transform `pred_coeffs` (none for LL bands,
/// whose prediction coefficients say nothing about where residuals are).
pub(crate) fn band_guide(
    slot: &BandSlot,
    seq: &SequenceHeader,
    qp: u8,
    pred_coeffs: &Plane<i16>,
) -> Option<Guide> {
    (!slot.is_ll()).then(|| {
        if seq.guide {
            Guide::build(
                pred_coeffs,
                slot.rect(),
                slot.step_q12(seq, qp),
                seq.block_log2,
            )
        } else {
            Guide::flat(slot.rect(), seq.block_log2, FLAT_GUIDE_CLASS)
        }
    })
}

/// One level's bands of a component, for finding a coefficient row's band in constant time.
#[derive(Clone, Copy)]
pub(crate) struct LevelBands {
    /// The level's low rows (its high bands' first row).
    ns: usize,
    /// Slot indices of the LL (coarsest level only, else none), HL, LH and HH bands, or
    /// `usize::MAX`.
    slots: [usize; 4],
}

impl LevelBands {
    /// Indexed by level (entry 0 unused).
    pub(crate) fn of(slots: &[BandSlot], comp: usize) -> Vec<Self> {
        let levels = slots
            .iter()
            .filter(|s| s.comp == comp)
            .map(|s| s.sb.level)
            .max()
            .unwrap_or(0);
        (0..=levels)
            .map(|l| {
                let mut b = Self {
                    ns: usize::MAX,
                    slots: [usize::MAX; 4],
                };
                for (i, s) in slots.iter().enumerate() {
                    if s.comp != comp || s.sb.level != l {
                        continue;
                    }
                    let (high, right) = (s.sb.y0 > 0, s.sb.x0 > 0);
                    if high {
                        b.ns = s.sb.y0;
                    }
                    b.slots[usize::from(high) * 2 + usize::from(right)] = i;
                }
                b
            })
            .collect()
    }

    /// The band holding row `y` of the level's region in its left or right half, if any (the
    /// left half of a low row is a band only at the coarsest level, the LL).
    pub(crate) fn slot(&self, y: usize, right: bool) -> Option<usize> {
        let i = self.slots[usize::from(y >= self.ns) * 2 + usize::from(right)];
        (i != usize::MAX).then_some(i)
    }
}

/// The inverse 9/7's zero-row test ([`crate::dwt::d97::InverseJob::zero`]) of component
/// `comp`: a band row is zero unless `nonzero[band][row]`.
pub(crate) fn zero_rows<'a>(
    bands: &'a [LevelBands],
    slots: &'a [BandSlot],
    nonzero: &'a [Vec<bool>],
) -> impl Fn(usize, usize, bool) -> bool + Sync + 'a {
    move |l, y, right| {
        bands[l]
            .slot(y, right)
            .is_some_and(|i| !nonzero[i][y - slots[i].sb.y0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq() -> SequenceHeader {
        SequenceHeader {
            width: 1280,
            height: 720,
            fps_num: 60,
            fps_den: 1,
            luma_levels: 5,
            chroma_levels: 4,
            block_log2: 6,
            mesh_log2: 5,
            wavelet: crate::dwt::Wavelet::Lifted97,
            luma_weights: vec![4096; 16],
            chroma_weights: vec![4096; 13],
            guide: true,
            continuous_cdfs: false,
            layout_coding: true,
            band_contexts: false,
            temporal_mv: false,
            temporal_motion: false,
        }
    }

    #[test]
    fn coding_order_puts_parents_first() {
        let slots = band_slots(&seq());
        assert_eq!(slots.len(), 16 + 13 + 13);
        assert!(slots[..3].iter().all(BandSlot::is_ll));
        for (i, s) in slots.iter().enumerate() {
            match s.parent {
                Some(p) => {
                    assert!(p < i);
                    let ps = &slots[p];
                    assert_eq!(
                        (ps.comp, ps.sb.orient, ps.sb.level),
                        (s.comp, s.sb.orient, s.sb.level + 1)
                    );
                }
                None => assert!(s.is_ll() || s.sb.level == component_levels(&seq(), s.comp)),
            }
        }
        // The last band is luma HH at level 1.
        let last = slots.last().unwrap();
        assert_eq!(
            (last.comp, last.sb.level, last.sb.orient),
            (0, 1, Orient::HH)
        );
    }

    #[test]
    fn every_trained_table_fits_its_context() {
        for (set, (contexts, trained)) in untrained_contexts().iter().zip(TRAINED).enumerate() {
            let sizes: Vec<usize> = contexts.cdfs().map(|c| c.symbols()).collect();
            for &(k, weights) in trained {
                let k = usize::from(k);
                assert!(k < sizes.len(), "set {set}: CDF {k}");
                assert_eq!(weights.len(), sizes[k], "set {set}: CDF {k}");
            }
        }
    }

    #[test]
    fn isolated_values_are_zeroed_only_when_alone_and_small() {
        let step = 10 * 4096; // Δ = 10
        let threshold = 3 * 4096 / 2; // 1.5 steps
        let r = Rect {
            x0: 0,
            y0: 0,
            w: 16,
            h: 16,
        };
        let mut coeffs = Plane::<i16>::new(16, 16);
        let mut values = Plane::<i32>::new(16, 16);
        // Alone and small: zeroed.
        coeffs.set(1, 1, 12);
        values.set(1, 1, 1);
        // Alone but large (16 > 15): kept.
        coeffs.set(13, 1, -16);
        values.set(13, 1, -1);
        // Small, but a neighbouring block holds a value: both kept.
        coeffs.set(1, 13, 11);
        values.set(1, 13, 1);
        coeffs.set(5, 13, 30);
        values.set(5, 13, 3);
        let mut maps = BandMaps::new(r);
        for y in 0..r.h {
            maps.add_row(y, values.row_span(y, 0, r.w));
        }
        assert_eq!(
            maps.zero_isolated(&mut values, &coeffs, r, step, threshold, true),
            1
        );
        assert_eq!(values.get(1, 1), 0);
        // The 2×2 map follows: the zeroed value's block is empty, the kept ones' are not.
        assert!(!maps.sig1[0] && maps.sig1[6]);
        let rebuilt = SigPyramid::build(&values, r, 4);
        assert_eq!(maps.pyramid(4), rebuilt);
        assert_eq!(values.get(13, 1), -1);
        assert_eq!((values.get(1, 13), values.get(5, 13)), (1, 3));
    }

    #[test]
    fn ll_prediction_round_trips() {
        let r = Rect {
            x0: 3,
            y0: 2,
            w: 9,
            h: 7,
        };
        let mut q = Plane::new(16, 12);
        for y in 0..r.h {
            for x in 0..r.w {
                q.set(r.x0 + x, r.y0 + y, ((x * 37 + y * 101) % 600) as i32 - 300);
            }
        }
        let mut res = Plane::new(r.w, r.h);
        ll_residuals(&q, r, &mut res);
        let mut back = Plane::new(16, 12);
        ll_reconstruct(&res, r, &mut back);
        for y in 0..r.h {
            assert_eq!(
                back.row_span(r.y0 + y, r.x0, r.w),
                q.row_span(r.y0 + y, r.x0, r.w)
            );
        }
    }
}
