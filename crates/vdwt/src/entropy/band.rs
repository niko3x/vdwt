//! VEC, the band coder (DESIGN.md §5.2).
//!
//! A band is coded as a quadtree. Each `2^B × 2^B` block (64×64 by default) gets a nonzero flag,
//! and nonzero nodes split into four children down to 4×4. A 4×4 node codes its four 2×2 quads:
//!
//! 1. ρ, the quad's significance pattern (one 16-ary symbol);
//! 2. U, the bit length of the quad's largest magnitude (as U − 1);
//! 3. when U > 1, each significant sample's own bit length e (as U − e);
//! 4. the bits below each magnitude's leading 1: the first in a context of the bit length (it is
//!    0 about 70% of the time), the rest as raw bits;
//! 5. each significant sample's sign, in a context of its left and upper neighbours' signs.
//!
//! Contexts come from already coded neighbours (left, above, above-left). Across scales they
//! also come from the parent: the coefficient at half the coordinates, in the same orientation,
//! one level coarser. When a node's earlier children were all zero, its last child must be
//! nonzero, so that flag is inferred, and that quad's ρ uses a table that excludes 0.
//!
//! Signs follow edges: along an edge neighbouring coefficients share a sign, across it they
//! alternate. In an LH band (horizontal edges) the left neighbour is along the edge and the upper
//! one across; HL bands are transposed (their upper neighbour is along), so both orientations,
//! which share contexts, see the same statistics. Raw signs cost about 5% more of an intra frame.
//!
//! Encoder and decoder run the same traversal through [`SymbolCoder`], so they cannot drift
//! apart. Children are visited in Z order, so the left, above and above-left neighbours of every
//! quad are always coded first.

use std::sync::atomic::{AtomicBool, Ordering};

use super::cdf::Cdf;
use super::range::{RangeDecoder, RangeEncoder};
use crate::plane::SharedRows;
use crate::{Plane, spare};

/// One side of the shared traversal: the encoder codes the values it is given, the decoder
/// decodes values and hands them back.
pub(crate) trait SymbolCoder {
    const ENCODER: bool;
    /// Called as a band starts with its contexts (used to record training statistics).
    #[inline]
    fn begin(&mut self, _ctx: &Contexts) {}
    /// Codes `*s` with `cdf`, or decodes into it; then adapts `cdf`.
    fn symbol(&mut self, cdf: &mut Cdf, s: &mut usize);
    /// Codes or decodes `count` (1..=16) raw bits.
    fn bits(&mut self, count: u32, v: &mut u32);
}

impl SymbolCoder for RangeEncoder {
    const ENCODER: bool = true;

    #[inline]
    fn symbol(&mut self, cdf: &mut Cdf, s: &mut usize) {
        self.encode(cdf, *s);
        cdf.update(*s);
    }

    #[inline]
    fn bits(&mut self, count: u32, v: &mut u32) {
        self.encode_bits(count, *v);
    }
}

/// A coder that codes nothing and logs every symbol with the place of its CDF in
/// [`Contexts::cdfs`]: the statistics that default tables are trained on.
#[derive(Default)]
pub(crate) struct Recorder {
    /// One log per band (per [`SymbolCoder::begin`]): `(CDF index, symbol)`.
    pub bands: Vec<Vec<(u8, u8)>>,
    /// Where the current band's CDFs sit, relative to its contexts.
    offsets: Vec<usize>,
    base: usize,
}

impl SymbolCoder for Recorder {
    const ENCODER: bool = true;

    fn begin(&mut self, ctx: &Contexts) {
        self.base = ctx as *const Contexts as usize;
        self.offsets.clear();
        self.offsets
            .extend(ctx.cdfs().map(|c| c as *const Cdf as usize - self.base));
        self.bands.push(Vec::new());
    }

    fn symbol(&mut self, cdf: &mut Cdf, s: &mut usize) {
        // Symbols outside a band's contexts (the global motion model) are not recorded.
        let offset = (cdf as *const Cdf as usize).wrapping_sub(self.base);
        if let (Some(k), Some(log)) = (
            self.offsets.iter().position(|&o| o == offset),
            self.bands.last_mut(),
        ) {
            log.push((k as u8, *s as u8));
        }
        cdf.update(*s);
    }

    fn bits(&mut self, _: u32, _: &mut u32) {}
}

impl SymbolCoder for RangeDecoder<'_> {
    const ENCODER: bool = false;

    #[inline]
    fn symbol(&mut self, cdf: &mut Cdf, s: &mut usize) {
        *s = self.decode(cdf);
        cdf.update(*s);
    }

    #[inline]
    fn bits(&mut self, count: u32, v: &mut u32) {
        *v = self.decode_bits(count);
    }
}

/// A rectangle of a coefficient plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// Which aligned blocks of a band hold a nonzero value, at every quadtree scale.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SigPyramid {
    /// `levels[j - 1]` covers `2^j × 2^j` blocks.
    levels: Vec<SigLevel>,
}

#[derive(Clone, Debug, PartialEq)]
struct SigLevel {
    w: usize,
    h: usize,
    nz: Vec<bool>,
}

impl Drop for SigPyramid {
    fn drop(&mut self) {
        for level in self.levels.drain(..) {
            spare::give(level.nz);
        }
    }
}

impl SigPyramid {
    /// The pyramid of a band known to be all zero, without looking at it.
    pub fn empty() -> Self {
        Self { levels: Vec::new() }
    }

    /// Scales `2^1` up to `2^max_log2` of the band at `band` in `plane`.
    pub fn build(plane: &Plane<i32>, band: Rect, max_log2: u32) -> Self {
        assert!(max_log2 >= 1);
        let (w, h) = (band.w.div_ceil(2), band.h.div_ceil(2));
        // Every flag below is written.
        let mut nz = spare::take(w * h);
        // The OR of the two rows, with a zero column past an odd width.
        let mut pair = spare::take_zeroed::<i32>(2 * w);
        for (y, nz) in nz.chunks_exact_mut(w).enumerate() {
            let top = plane.row_span(band.y0 + 2 * y, band.x0, band.w);
            let head = &mut pair[..band.w];
            if 2 * y + 1 < band.h {
                let bottom = plane.row_span(band.y0 + 2 * y + 1, band.x0, band.w);
                for ((p, &a), &b) in head.iter_mut().zip(top).zip(bottom) {
                    *p = a | b;
                }
            } else {
                head.copy_from_slice(top);
            }
            for (o, c) in nz.iter_mut().zip(pair.chunks_exact(2)) {
                *o = (c[0] | c[1]) != 0;
            }
        }
        spare::give(pair);
        Self::from_first(SigLevel { w, h, nz }, max_log2)
    }

    /// [`Self::build`] from the first level already made: `nz` holds, for every 2×2 block of
    /// the band in raster order (`w` blocks a row), whether it holds a nonzero value.
    pub fn from_level1(nz: Vec<bool>, w: usize, max_log2: u32) -> Self {
        assert!(max_log2 >= 1 && w > 0 && nz.len().is_multiple_of(w));
        let h = nz.len() / w;
        Self::from_first(SigLevel { w, h, nz }, max_log2)
    }

    fn from_first(first: SigLevel, max_log2: u32) -> Self {
        let mut levels = vec![first];
        for _ in 2..=max_log2 {
            let prev = levels.last().expect("level 1 exists");
            let (w, h) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            // Each block's OR of the 2×2 below it, with a zero row and column past odd sides
            // (every flag written).
            let mut nz = spare::take(w * h);
            for (y, out) in nz.chunks_exact_mut(w).enumerate() {
                let top = &prev.nz[2 * y * prev.w..][..prev.w];
                let bottom = if 2 * y + 1 < prev.h {
                    &prev.nz[(2 * y + 1) * prev.w..][..prev.w]
                } else {
                    top
                };
                for ((o, t), b) in out.iter_mut().zip(top.chunks(2)).zip(bottom.chunks(2)) {
                    *o = t.iter().chain(b).any(|&b| b);
                }
            }
            levels.push(SigLevel { w, h, nz });
        }
        Self { levels }
    }

    /// Whether block `(bx, by)` of size `2^j` holds a nonzero value; false outside the band.
    #[inline]
    pub fn get(&self, j: u32, bx: usize, by: usize) -> bool {
        self.levels
            .get(j as usize - 1)
            .is_some_and(|level| bx < level.w && by < level.h && level.nz[by * level.w + bx])
    }

    /// Whether row pair `y1` (rows `2·y1` and `2·y1 + 1`) holds a nonzero value.
    pub fn pair_row_any(&self, y1: usize) -> bool {
        self.levels
            .first()
            .is_some_and(|l| y1 < l.h && l.nz[y1 * l.w..][..l.w].iter().any(|&b| b))
    }

    /// Whether the band holds any nonzero value.
    pub fn any(&self) -> bool {
        self.levels.last().is_some_and(|l| l.nz.iter().any(|&b| b))
    }
}

/// The same-orientation band one level coarser, already coded: its values are at `band` in
/// `values`.
#[derive(Clone, Copy)]
pub(crate) struct Parent<'a> {
    pub values: &'a Plane<i32>,
    pub band: Rect,
    pub sig: &'a SigPyramid,
}

/// Where a band's values come from (encoder: read only, so bands can be coded concurrently) or
/// go (decoder).
pub(crate) trait BandValues {
    fn value(&self, x: usize, y: usize) -> i32;
    fn set_value(&mut self, x: usize, y: usize, v: i32);
}

impl BandValues for &Plane<i32> {
    #[inline]
    fn value(&self, x: usize, y: usize) -> i32 {
        self.get(x, y)
    }

    fn set_value(&mut self, _: usize, _: usize, _: i32) {
        unreachable!("encoding only reads the band");
    }
}

impl BandValues for &mut Plane<i32> {
    #[inline]
    fn value(&self, x: usize, y: usize) -> i32 {
        self.get(x, y)
    }

    #[inline]
    fn set_value(&mut self, x: usize, y: usize, v: i32) {
        self.set(x, y, v);
    }
}

/// Decoder: a band at `(x0, y0)` of a 16-bit coefficient plane, shared by tasks that each decode
/// their own rows (the `rt` codec's tiles). Values are stored quantized (saturated to 16 bits)
/// for dequantization in place; `touched` flags the band rows that received one. The band coder
/// reads and writes only inside the rectangle it is given, so tasks with disjoint rectangles
/// never touch the same samples.
pub(crate) struct CoeffBand<'a, 'b> {
    pub rows: &'a SharedRows<'b, i16>,
    pub x0: usize,
    pub y0: usize,
    pub touched: &'a [AtomicBool],
}

impl BandValues for CoeffBand<'_, '_> {
    #[inline]
    fn value(&self, x: usize, y: usize) -> i32 {
        // SAFETY: only this task touches the rows of its rectangle (see the type).
        i32::from(unsafe { self.rows.row(self.y0 + y, self.x0 + x, 1)[0] })
    }

    #[inline]
    fn set_value(&mut self, x: usize, y: usize, v: i32) {
        // SAFETY: as above.
        unsafe {
            self.rows.row_mut(self.y0 + y, self.x0 + x, 1)[0] =
                v.clamp(-i32::from(i16::MAX), i32::from(i16::MAX)) as i16
        };
        self.touched[y].store(true, Ordering::Relaxed);
    }
}

/// Decoder: a band's plane that also notes which of its rows received a value.
pub(crate) struct TrackedRows<'a> {
    pub plane: &'a mut Plane<i32>,
    /// One flag per row of the band (`band.y0 = 0`).
    pub rows: &'a mut [bool],
}

impl BandValues for TrackedRows<'_> {
    #[inline]
    fn value(&self, x: usize, y: usize) -> i32 {
        self.plane.get(x, y)
    }

    #[inline]
    fn set_value(&mut self, x: usize, y: usize, v: i32) {
        self.plane.set(x, y, v);
        self.rows[y] = true;
    }
}

/// Node flag contexts are split by child size: 4, 8, 16, and 32 or larger.
const NODE_SIZES: usize = 4;
/// Prediction guide classes (see [`Guide`]).
pub(crate) const GUIDES: usize = 4;
/// CDFs per guide class, and the CDFs shared by all.
const GROUP_CDFS: usize = 4 + 4 * NODE_SIZES + 3 * 12;
const SHARED_CDFS: usize = 4;
/// Sign contexts: the sign along the edge × the sign across it, each none, + or −.
const SIGN_CDFS: usize = 9;
/// Initial sign weights (+, −) in ‰, from intra frames of drone footage.
const SIGN_WEIGHTS: [[u64; 2]; SIGN_CDFS] = [
    [500, 500],
    [315, 685],
    [678, 322],
    [737, 263],
    [574, 426],
    [802, 198],
    [252, 748],
    [195, 805],
    [422, 578],
];
/// The same for P-frame residuals, where only the sign along the edge tells anything.
const RESIDUAL_SIGN_WEIGHTS: [[u64; 2]; SIGN_CDFS] = [
    [500, 500],
    [495, 505],
    [486, 514],
    [787, 213],
    [790, 210],
    [738, 262],
    [204, 796],
    [262, 738],
    [229, 771],
];
/// Contexts of the bit below a magnitude's leading 1, by bit length 2, 3, 4, 5 or more.
const MANTISSA_CDFS: usize = 4;
/// Initial weights (0, 1) in ‰, from intra frames of drone footage.
const MANTISSA_WEIGHTS: [[u64; 2]; MANTISSA_CDFS] =
    [[675, 325], [691, 309], [714, 286], [745, 255]];
/// CDFs in one set of [`Contexts`].
pub(crate) const CDFS: usize = GUIDES * GROUP_CDFS + SHARED_CDFS + SIGN_CDFS + MANTISSA_CDFS;

/// The contexts of one guide class.
#[derive(Clone, Debug)]
struct Group {
    /// Block flag: parent region nonzero × (left or above block nonzero).
    block: [Cdf; 4],
    /// Node flag: child size × parent region nonzero × an earlier sibling nonzero.
    node: [Cdf; NODE_SIZES * 4],
    /// Quad pattern ρ: neighbour count bucket (0..=3) × parent bucket (0..=2).
    rho: [Cdf; 12],
    /// ρ for a quad that must be nonzero (the last one of a node whose other quads were zero).
    rho_forced: [Cdf; 12],
    /// Quad bit length U − 1: neighbour bit-length bucket (0..=3) × parent bucket (0..=2).
    exp: [Cdf; 12],
}

impl Group {
    fn cdfs(&self) -> impl Iterator<Item = &Cdf> {
        self.block
            .iter()
            .chain(&self.node)
            .chain(&self.rho)
            .chain(&self.rho_forced)
            .chain(&self.exp)
    }

    fn cdfs_mut(&mut self) -> impl Iterator<Item = &mut Cdf> {
        self.block
            .iter_mut()
            .chain(&mut self.node)
            .chain(&mut self.rho)
            .chain(&mut self.rho_forced)
            .chain(&mut self.exp)
    }
}

/// The adaptive state of one band's coding. Every band starts from a fresh copy of its class's
/// defaults, so bands decode independently once their parents are known.
#[derive(Clone, Debug)]
pub(crate) struct Contexts {
    /// Flag, pattern and exponent contexts per guide class.
    group: [Group; GUIDES],
    /// U − e per sample of a quad with U > 1, by min(U, 5) − 2.
    mag: [Cdf; 4],
    /// Sign: along × across neighbour sign (none, +, −).
    sign: [Cdf; SIGN_CDFS],
    /// The bit below the leading 1, by min(e, 5) − 2.
    mantissa: [Cdf; MANTISSA_CDFS],
}

impl Contexts {
    /// Every CDF, in a fixed order: the order of trained tables. Guide class 0 and the shared
    /// CDFs come first.
    pub(crate) fn cdfs(&self) -> impl Iterator<Item = &Cdf> {
        let [g0, rest @ ..] = &self.group;
        g0.cdfs()
            .chain(&self.mag)
            .chain(rest.iter().flat_map(|g| g.cdfs()))
            .chain(&self.sign)
            .chain(&self.mantissa)
    }

    fn cdfs_mut(&mut self) -> impl Iterator<Item = &mut Cdf> {
        let [g0, rest @ ..] = &mut self.group;
        g0.cdfs_mut()
            .chain(&mut self.mag)
            .chain(rest.iter_mut().flat_map(|g| g.cdfs_mut()))
            .chain(&mut self.sign)
            .chain(&mut self.mantissa)
    }

    /// These contexts with every CDF that `trained` lists, as `(CDF index, probabilities)`,
    /// replaced by a distribution proportional to them. A table that covers only guide class 0
    /// seeds the other classes too.
    pub(crate) fn with_trained(mut self, trained: &[(u8, &[u16])]) -> Self {
        {
            let mut cdfs: Vec<&mut Cdf> = self.cdfs_mut().collect();
            for &(k, weights) in trained {
                let cdf = &mut cdfs[usize::from(k)];
                if weights.len() == cdf.symbols() {
                    let w: Vec<u64> = weights.iter().map(|&w| u64::from(w)).collect();
                    **cdf = Cdf::from_weights(&w);
                }
            }
        }
        if trained
            .iter()
            .all(|&(k, _)| usize::from(k) < GROUP_CDFS + SHARED_CDFS)
        {
            let g0 = self.group[0].clone();
            for g in &mut self.group[1..] {
                *g = g0.clone();
            }
        }
        self
    }

    /// These contexts with the sign weights of P-frame residuals.
    pub(crate) fn for_residuals(mut self) -> Self {
        self.sign = RESIDUAL_SIGN_WEIGHTS.map(|w| Cdf::from_weights(&w));
        self
    }

    /// Untrained starting distributions. `dense` is for low-pass bands, where most coefficients
    /// are significant. These are placeholders until tables are trained on drone footage.
    pub(crate) fn new(dense: bool) -> Self {
        let flag = |p: u64| {
            let p = p.clamp(8, 248);
            Cdf::from_weights(&[256 - p, p])
        };
        let pattern = |p: u64, forced: bool| {
            let p = p.clamp(8, 248);
            let mut w: [u64; 16] = std::array::from_fn(|rho| {
                let k = (rho as u32).count_ones();
                p.pow(k) * (256 - p).pow(4 - k)
            });
            if forced {
                w[0] = 0;
            }
            Cdf::from_weights(&w)
        };
        let geometric = |ratio: u64| {
            let mut w = [0u64; 16];
            let mut v = 1u64 << 40;
            for x in &mut w {
                *x = v.max(1);
                v = v * ratio.min(250) / 256;
            }
            Cdf::from_weights(&w)
        };
        let rho_p = |i: usize| {
            if dense {
                200
            } else {
                30 + 40 * (i / 3) as u64 + 30 * (i % 3) as u64
            }
        };
        let group = || Group {
            block: std::array::from_fn(|i| {
                flag(if dense {
                    230
                } else {
                    30 + 60 * (i / 2) as u64 + 60 * (i % 2) as u64
                })
            }),
            node: std::array::from_fn(|i| {
                let (size, parent, sibling) = (i / 4, (i / 2) % 2, i % 2);
                flag(if dense {
                    230
                } else {
                    70 + 15 * size as u64 + 50 * parent as u64 + 20 * sibling as u64
                })
            }),
            rho: std::array::from_fn(|i| pattern(rho_p(i), false)),
            rho_forced: std::array::from_fn(|i| pattern(rho_p(i), true)),
            exp: std::array::from_fn(|i| {
                geometric(if dense {
                    230
                } else {
                    100 + 30 * (i / 3) as u64 + 20 * (i % 3) as u64
                })
            }),
        };
        Self {
            group: std::array::from_fn(|_| group()),
            mag: std::array::from_fn(|_| geometric(110)),
            sign: SIGN_WEIGHTS.map(|w| Cdf::from_weights(&w)),
            mantissa: MANTISSA_WEIGHTS.map(|w| Cdf::from_weights(&w)),
        }
    }
}

/// Where a P-frame band's prediction has texture: the guide class of every coefficient, from the
/// same band of the prediction's own transform, and the largest class of every block at each
/// quadtree scale. The decoder has the prediction before it decodes the residual, and residual
/// values gather where the prediction has texture (motion errors show at edges), so the classes
/// select contexts. A coefficient's class compares the prediction's coefficient `|p|` with the
/// band's step `Δ`: 0 below `Δ/4`, 1 below `Δ`, 2 below `4Δ`, 3 above.
#[derive(Clone, Debug, Default)]
pub(crate) struct Guide {
    /// `levels[j - 1]` covers `2^j × 2^j` blocks (the coder never asks for single
    /// coefficients).
    levels: Vec<GuideLevel>,
}

#[derive(Clone, Debug, Default)]
struct GuideLevel {
    w: usize,
    h: usize,
    class: Vec<u8>,
}

/// The class a flat (realtime) guide gives every block: mid-texture. The prediction DWT is not
/// computed at all, so the encoder and decoder use this constant for every band block.
pub(crate) const FLAT_GUIDE_CLASS: u8 = 3;

/// The classes of a [`Guide`]'s 2×2 blocks: how many of the thresholds `t` the block's largest
/// magnitude reaches.
#[inline(always)]
#[cfg_attr(
    all(target_arch = "aarch64", target_feature = "neon"),
    allow(dead_code)
)]
fn guide_classes(coeffs: &Plane<i16>, band: Rect, t: [u16; 3], out: &mut [u8]) {
    let w = band.w.div_ceil(2);
    // The larger magnitude of the two rows, with a zero column past an odd width.
    let mut pair = spare::take_zeroed::<u16>(2 * w);
    for (y, out) in out.chunks_exact_mut(w).enumerate() {
        let top = coeffs.row_span(band.y0 + 2 * y, band.x0, band.w);
        let head = &mut pair[..band.w];
        if 2 * y + 1 < band.h {
            let bottom = coeffs.row_span(band.y0 + 2 * y + 1, band.x0, band.w);
            for ((p, &a), &b) in head.iter_mut().zip(top).zip(bottom) {
                *p = a.unsigned_abs().max(b.unsigned_abs());
            }
        } else {
            for (p, &a) in head.iter_mut().zip(top) {
                *p = a.unsigned_abs();
            }
        }
        // Each pair of columns as one word, whose halves are the two columns.
        let words = pair.as_ptr().cast::<u32>();
        for (x, o) in out.iter_mut().enumerate() {
            // SAFETY: `pair` holds 2w u16s, so w unaligned u32 reads are in bounds.
            let v = unsafe { words.add(x).read_unaligned() };
            let m = ((v & 0xFFFF) as u16).max((v >> 16) as u16);
            *o = u8::from(m >= t[0]) + u8::from(m >= t[1]) + u8::from(m >= t[2]);
        }
    }
    spare::give(pair);
}

/// The next coarser guide level of a `pw × ph` one: the largest class of each 2×2 block.
fn coarser_classes(prev: &[u8], pw: usize, ph: usize, out: &mut [u8]) {
    let w = pw.div_ceil(2);
    // The larger class of the two rows, with a zero column past an odd width.
    let mut pair = spare::take_zeroed::<u8>(2 * w);
    for (y, out) in out.chunks_exact_mut(w).enumerate() {
        let top = &prev[2 * y * pw..][..pw];
        let bottom = if 2 * y + 1 < ph {
            &prev[(2 * y + 1) * pw..][..pw]
        } else {
            top
        };
        for ((p, &a), &b) in pair[..pw].iter_mut().zip(top).zip(bottom) {
            *p = a.max(b);
        }
        for (o, c) in out.iter_mut().zip(pair.chunks_exact(2)) {
            *o = c[0].max(c[1]);
        }
    }
    spare::give(pair);
}

/// [`guide_classes`] on NEON: 8 blocks at a time, `LD2` splitting even and odd columns; partial
/// blocks (an odd last row or column) take the portable rule.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
fn guide_classes_neon(coeffs: &Plane<i16>, band: Rect, t: [u16; 3], out: &mut [u8]) {
    use std::arch::aarch64::*;
    let w = band.w.div_ceil(2);
    let full = band.w / 2;
    let class = |m: u16| u8::from(m >= t[0]) + u8::from(m >= t[1]) + u8::from(m >= t[2]);
    let tv = t.map(|v| vdupq_n_u16(v));
    for (y, out) in out.chunks_exact_mut(w).enumerate() {
        let top = coeffs.row_span(band.y0 + 2 * y, band.x0, band.w);
        let bottom =
            (2 * y + 1 < band.h).then(|| coeffs.row_span(band.y0 + 2 * y + 1, band.x0, band.w));
        let mut x = 0;
        if let Some(bottom) = bottom {
            while x + 8 <= full {
                // SAFETY: blocks x..x + 8 are whole, so 16 samples from 2x are in both rows;
                // `out` holds w ≥ full classes.
                unsafe {
                    let a = vld2q_s16(top.as_ptr().add(2 * x));
                    let b = vld2q_s16(bottom.as_ptr().add(2 * x));
                    // |i16::MIN| stays 0x8000: read as unsigned it is 32768, as `unsigned_abs`.
                    let abs = |v: int16x8_t| vreinterpretq_u16_s16(vabsq_s16(v));
                    let m = vmaxq_u16(vmaxq_u16(abs(a.0), abs(a.1)), vmaxq_u16(abs(b.0), abs(b.1)));
                    // Each reached threshold is a mask of all ones (−1).
                    let n = vaddq_u16(
                        vaddq_u16(vcgeq_u16(m, tv[0]), vcgeq_u16(m, tv[1])),
                        vcgeq_u16(m, tv[2]),
                    );
                    let c = vmovn_u16(vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(n))));
                    vst1_u8(out.as_mut_ptr().add(x), c);
                }
                x += 8;
            }
        }
        for (x, o) in out.iter_mut().enumerate().skip(x) {
            let cols = 2 * x..(2 * x + 2).min(band.w);
            let mut m = top[cols.clone()]
                .iter()
                .map(|v| v.unsigned_abs())
                .max()
                .unwrap_or(0);
            if let Some(bottom) = bottom {
                m = m.max(
                    bottom[cols]
                        .iter()
                        .map(|v| v.unsigned_abs())
                        .max()
                        .unwrap_or(0),
                );
            }
            *o = class(m);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn guide_classes_avx2(coeffs: &Plane<i16>, band: Rect, t: [u16; 3], out: &mut [u8]) {
    guide_classes(coeffs, band, t, out);
}

impl Drop for Guide {
    fn drop(&mut self) {
        for level in self.levels.drain(..) {
            spare::give(level.class);
        }
    }
}

impl Guide {
    /// The guide of the band at `band` in `coeffs` (the prediction's transform), with step
    /// `step_q12`, for scales `2^1 ..= 2^max_log2`.
    pub fn build(coeffs: &Plane<i16>, band: Rect, step_q12: u64, max_log2: u32) -> Self {
        // |p|·4096 ≥ t  ⟺  |p| ≥ ⌈t / 4096⌉, for Δ/4, Δ and 4Δ in Q12. No |p| reaches
        // u16::MAX, so clamping there changes nothing.
        let at_least = |t: u64| t.div_ceil(4096).min(u64::from(u16::MAX)) as u16;
        let t = [
            at_least(step_q12.div_ceil(4)),
            at_least(step_q12),
            at_least(4 * step_q12),
        ];
        let (w, h) = (band.w.div_ceil(2), band.h.div_ceil(2));
        // Every class below is written.
        let mut first = spare::take(w * h);
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::avx2_detected() {
            // SAFETY: the CPU has AVX2.
            unsafe { guide_classes_avx2(coeffs, band, t, &mut first) };
        } else {
            guide_classes(coeffs, band, t, &mut first);
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        // SAFETY: NEON is part of the AArch64 baseline.
        unsafe {
            guide_classes_neon(coeffs, band, t, &mut first)
        };
        #[cfg(not(any(
            target_arch = "x86_64",
            all(target_arch = "aarch64", target_feature = "neon")
        )))]
        guide_classes(coeffs, band, t, &mut first);
        let mut levels = vec![GuideLevel { w, h, class: first }];
        for _ in 2..=max_log2 {
            let prev = levels.last().expect("level 1 exists");
            let (w, h) = (prev.w.div_ceil(2), prev.h.div_ceil(2));
            let mut class = spare::take(w * h);
            coarser_classes(&prev.class, prev.w, prev.h, &mut class);
            levels.push(GuideLevel { w, h, class });
        }
        Self { levels }
    }

    /// A guide whose every block has the same class, for the realtime mode that skips the
    /// prediction DWT. The level dimensions match [`Guide::build`].
    pub fn flat(band: Rect, max_log2: u32, class: u8) -> Self {
        let (mut w, mut h) = (band.w.div_ceil(2), band.h.div_ceil(2));
        let mut levels = Vec::new();
        for _ in 1..=max_log2 {
            let mut buf = spare::take(w * h);
            buf.fill(class);
            levels.push(GuideLevel { w, h, class: buf });
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        Self { levels }
    }

    /// The largest class in block `(bx, by)` of size `2^j` (`j ≥ 1`); 0 outside the band.
    #[inline]
    fn get(&self, j: u32, bx: usize, by: usize) -> usize {
        self.levels.get(j as usize - 1).map_or(0, |l| {
            if bx < l.w && by < l.h {
                usize::from(l.class[by * l.w + bx])
            } else {
                0
            }
        })
    }
}

/// Codes (encoder) or decodes (decoder) the band at `band` in `plane`.
///
/// The encoder passes the band's own [`SigPyramid`] as `own`. The decoder passes `None`, and the
/// band area must be zero beforehand: it only writes the nonzero samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn code_band<C: SymbolCoder, V: BandValues>(
    coder: &mut C,
    ctx: &mut Contexts,
    plane: V,
    band: Rect,
    parent: Option<Parent<'_>>,
    own: Option<&SigPyramid>,
    guide: Option<&Guide>,
    block_log2: u32,
    transpose: bool,
) {
    debug_assert_eq!(C::ENCODER, own.is_some());
    debug_assert!((2..=7).contains(&block_log2));
    coder.begin(ctx);
    if band.is_empty() {
        return;
    }
    let parent = parent.filter(|p| !p.band.is_empty());
    Walker {
        coder,
        ctx,
        plane,
        band,
        parent,
        own,
        guide,
        block_log2,
        transpose,
    }
    .blocks();
}

struct Walker<'a, C, V> {
    coder: &'a mut C,
    ctx: &'a mut Contexts,
    plane: V,
    band: Rect,
    parent: Option<Parent<'a>>,
    own: Option<&'a SigPyramid>,
    guide: Option<&'a Guide>,
    block_log2: u32,
    /// An HL band: the upper neighbour is along the edge (sign contexts).
    transpose: bool,
}

/// Bit length of a magnitude: 0 for 0, else ⌊log2 m⌋ + 1.
#[inline]
fn bit_length(m: u32) -> u32 {
    32 - m.leading_zeros()
}

impl<C: SymbolCoder, V: BandValues> Walker<'_, C, V> {
    fn blocks(&mut self) {
        let b = self.block_log2;
        let size = 1usize << b;
        let (nbx, nby) = (self.band.w.div_ceil(size), self.band.h.div_ceil(size));
        let mut above = vec![false; nbx];
        for by in 0..nby {
            let mut left = false;
            for (bx, above) in above.iter_mut().enumerate() {
                let parent_nz = self.parent.is_some_and(|p| p.sig.get(b - 1, bx, by));
                let ci = usize::from(parent_nz) * 2 + usize::from(left || *above);
                let g = self.guide(b, bx, by);
                let mut nz = usize::from(self.own_nz(b, bx, by));
                self.coder.symbol(&mut self.ctx.group[g].block[ci], &mut nz);
                let nz = nz != 0;
                if nz {
                    self.node(bx << b, by << b, b);
                }
                left = nz;
                *above = nz;
            }
        }
    }

    /// The guide class of block `(bx, by)` of size `2^j` (0 without a guide).
    #[inline]
    fn guide(&self, j: u32, bx: usize, by: usize) -> usize {
        self.guide.map_or(0, |g| g.get(j, bx, by))
    }

    /// Whether the encoder's band has a nonzero value in block `(bx, by)` of size `2^j`.
    #[inline]
    fn own_nz(&self, j: u32, bx: usize, by: usize) -> bool {
        C::ENCODER && self.own.is_some_and(|o| o.get(j, bx, by))
    }

    #[inline]
    fn inside(&self, x: usize, y: usize) -> bool {
        x < self.band.w && y < self.band.h
    }

    /// A nonzero node of size `2^j` (`j ≥ 2`) at band coordinates `(x0, y0)`.
    fn node(&mut self, x0: usize, y0: usize, j: u32) {
        if j == 2 {
            return self.quads(x0, y0);
        }
        let half = 1usize << (j - 1);
        let children = [
            (x0, y0),
            (x0 + half, y0),
            (x0, y0 + half),
            (x0 + half, y0 + half),
        ];
        let inside = children.iter().filter(|&&(x, y)| self.inside(x, y)).count();
        let size = (j as usize - 3).min(NODE_SIZES - 1);
        let (mut seen, mut any) = (0, false);
        for (cx, cy) in children {
            if !self.inside(cx, cy) {
                continue;
            }
            seen += 1;
            let (bx, by) = (cx >> (j - 1), cy >> (j - 1));
            let nz = if seen == inside && !any {
                true
            } else {
                let parent_nz = self.parent.is_some_and(|p| p.sig.get(j - 2, bx, by));
                let ci = size * 4 + usize::from(parent_nz) * 2 + usize::from(any);
                let g = self.guide(j - 1, bx, by);
                let mut v = usize::from(self.own_nz(j - 1, bx, by));
                self.coder.symbol(&mut self.ctx.group[g].node[ci], &mut v);
                v != 0
            };
            if nz {
                any = true;
                self.node(cx, cy, j - 1);
            }
        }
    }

    /// The four quads of a nonzero 4×4 node.
    fn quads(&mut self, x0: usize, y0: usize) {
        let quads = [(x0, y0), (x0 + 2, y0), (x0, y0 + 2), (x0 + 2, y0 + 2)];
        let inside = quads.iter().filter(|&&(x, y)| self.inside(x, y)).count();
        let (mut seen, mut any) = (0, false);
        for (qx, qy) in quads {
            if !self.inside(qx, qy) {
                continue;
            }
            seen += 1;
            any |= self.quad(qx, qy, seen == inside && !any);
        }
    }

    /// One 2×2 quad. Returns whether it holds a nonzero value.
    fn quad(&mut self, qx: usize, qy: usize, forced: bool) -> bool {
        let mask = (0..4).fold(0usize, |m, b| {
            m | (usize::from(self.inside(qx + (b & 1), qy + (b >> 1))) << b)
        });
        let (count, max_bits) = self.neighbours(qx, qy);
        let pb = self.parent_bucket(qx, qy);

        let mut rho = 0;
        if C::ENCODER {
            for b in 0..4 {
                if mask & (1 << b) != 0 && self.get(qx + (b & 1), qy + (b >> 1)) != 0 {
                    rho |= 1 << b;
                }
            }
        }
        let ci = count.min(3) * 3 + pb;
        let g = self.guide(1, qx / 2, qy / 2);
        let group = &mut self.ctx.group[g];
        let cdf = if forced {
            &mut group.rho_forced[ci]
        } else {
            &mut group.rho[ci]
        };
        self.coder.symbol(cdf, &mut rho);
        let rho = rho & mask;
        if rho == 0 {
            return false;
        }

        let mut u1 = 0;
        if C::ENCODER {
            u1 = (0..4)
                .filter(|b| rho & (1 << b) != 0)
                .map(|b| bit_length(self.get(qx + (b & 1), qy + (b >> 1)).unsigned_abs()))
                .max()
                .unwrap_or(1) as usize
                - 1;
        }
        self.coder.symbol(
            &mut self.ctx.group[g].exp[(max_bits.min(3) as usize) * 3 + pb],
            &mut u1,
        );
        let u = u1 as u32 + 1;

        let significant = rho.count_ones();
        let (mut k, mut reached) = (0, false);
        for b in 0..4 {
            if rho & (1 << b) == 0 {
                continue;
            }
            k += 1;
            let (x, y) = (qx + (b & 1), qy + (b >> 1));
            let v = if C::ENCODER { self.get(x, y) } else { 0 };
            let m = v.unsigned_abs();
            // Bit length e: free when U = 1, inferred as U for the last sample if none reached U.
            let e = if u == 1 {
                1
            } else if k == significant && !reached {
                u
            } else {
                let mut t = if C::ENCODER {
                    (u - bit_length(m)) as usize
                } else {
                    0
                };
                self.coder
                    .symbol(&mut self.ctx.mag[(u.min(5) - 2) as usize], &mut t);
                u - (t as u32).min(u - 1)
            };
            reached |= e == u;
            // Below the leading 1: the first bit in context, the rest raw.
            let mut raw = if C::ENCODER { m - (1 << (e - 1)) } else { 0 };
            if e > 1 {
                let mut first = (raw >> (e - 2)) as usize & 1;
                self.coder
                    .symbol(&mut self.ctx.mantissa[(e.min(5) - 2) as usize], &mut first);
                let mut rest = raw & ((1 << (e - 2)) - 1);
                if e > 2 {
                    self.coder.bits(e - 2, &mut rest);
                }
                raw = ((first as u32) << (e - 2)) | rest;
            }
            let mut negative = usize::from(v < 0);
            let c = self.sign_context(x, y);
            self.coder.symbol(&mut self.ctx.sign[c], &mut negative);
            if !C::ENCODER {
                let m = ((1u32 << (e - 1)) | raw) as i32;
                self.set(x, y, if negative == 1 { -m } else { m });
            }
        }
        true
    }

    /// Sign context of sample `(x, y)`: its neighbour along the edge (left, or above in an HL
    /// band) and across it, each none, + or −. Both are coded before it.
    fn sign_context(&self, x: usize, y: usize) -> usize {
        let class = |v: i32| match v.signum() {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        let left = if x > 0 {
            class(self.get_or_zero(x - 1, y))
        } else {
            0
        };
        let up = if y > 0 {
            class(self.get_or_zero(x, y - 1))
        } else {
            0
        };
        let (along, across) = if self.transpose {
            (up, left)
        } else {
            (left, up)
        };
        along * 3 + across
    }

    /// Nonzero count and largest bit length among the coded neighbours of quad `(qx, qy)`: the
    /// two samples to its left, the two above it, and the one above-left.
    fn neighbours(&self, qx: usize, qy: usize) -> (usize, u32) {
        let mut count = 0;
        let mut max_bits = 0;
        let mut visit = |v: i32| {
            count += usize::from(v != 0);
            max_bits = max_bits.max(bit_length(v.unsigned_abs()));
        };
        if qx > 0 {
            visit(self.get_or_zero(qx - 1, qy));
            visit(self.get_or_zero(qx - 1, qy + 1));
        }
        if qy > 0 {
            visit(self.get_or_zero(qx, qy - 1));
            visit(self.get_or_zero(qx + 1, qy - 1));
            if qx > 0 {
                visit(self.get_or_zero(qx - 1, qy - 1));
            }
        }
        (count, max_bits)
    }

    /// Parent magnitude bucket for quad `(qx, qy)`: 0 for zero (or no parent), 1 for ±1, 2 above.
    fn parent_bucket(&self, qx: usize, qy: usize) -> usize {
        self.parent.map_or(0, |p| {
            let (px, py) = ((qx / 2).min(p.band.w - 1), (qy / 2).min(p.band.h - 1));
            bit_length(p.values.get(p.band.x0 + px, p.band.y0 + py).unsigned_abs()).min(2) as usize
        })
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> i32 {
        self.plane.value(self.band.x0 + x, self.band.y0 + y)
    }

    #[inline]
    fn get_or_zero(&self, x: usize, y: usize) -> i32 {
        if self.inside(x, y) { self.get(x, y) } else { 0 }
    }

    #[inline]
    fn set(&mut self, x: usize, y: usize, v: i32) {
        self.plane.set_value(self.band.x0 + x, self.band.y0 + y, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// Codes `band` (with an optional parent band in the same plane) and decodes it into a fresh
    /// plane; the decoded band must match.
    fn round_trip(plane: &Plane<i32>, band: Rect, parent: Option<Rect>, block_log2: u32) -> usize {
        let pyr = |p: &Plane<i32>, r: Rect| SigPyramid::build(p, r, block_log2);
        let own = pyr(plane, band);
        let parent_sig = parent.map(|r| pyr(plane, r));
        let mut enc = RangeEncoder::new();
        let mut ctx = Contexts::new(false);
        let p = parent.zip(parent_sig.as_ref()).map(|(band, sig)| Parent {
            values: plane,
            band,
            sig,
        });
        code_band(
            &mut enc,
            &mut ctx,
            plane,
            band,
            p,
            Some(&own),
            None,
            block_log2,
            false,
        );
        let bytes = enc.finish();

        // The decoder starts with the parent band already present and the band itself zeroed.
        let mut dst = Plane::new(plane.width(), plane.height());
        if let Some(r) = parent {
            for y in 0..r.h {
                dst.row_span_mut(r.y0 + y, r.x0, r.w)
                    .copy_from_slice(plane.row_span(r.y0 + y, r.x0, r.w));
            }
        }
        let parent_values = dst.clone();
        let mut dec = RangeDecoder::new(&bytes);
        let mut ctx = Contexts::new(false);
        let p = parent.zip(parent_sig.as_ref()).map(|(band, sig)| Parent {
            values: &parent_values,
            band,
            sig,
        });
        code_band(
            &mut dec, &mut ctx, &mut dst, band, p, None, None, block_log2, false,
        );
        for y in 0..band.h {
            assert_eq!(
                dst.row_span(band.y0 + y, band.x0, band.w),
                plane.row_span(band.y0 + y, band.x0, band.w),
                "row {y} of {band:?}"
            );
        }
        bytes.len()
    }

    /// Random values: mostly zero, sometimes clustered, with magnitudes up to `max`.
    fn fill(rng: &mut SplitMix64, plane: &mut Plane<i32>, r: Rect, density: u64, max: i32) {
        for y in 0..r.h {
            for x in 0..r.w {
                let v = if rng.below(100) < density {
                    rng.range_i32(-max, max)
                } else {
                    0
                };
                plane.set(r.x0 + x, r.y0 + y, v);
            }
        }
    }

    #[test]
    fn round_trips_random_bands() {
        let mut rng = SplitMix64::new(21);
        for case in 0..300 {
            let (w, h) = (rng.range_usize(1, 150), rng.range_usize(1, 150));
            let block_log2 = rng.range_usize(2, 7) as u32;
            let mut plane = Plane::new(2 * w + 3, 2 * h + 3);
            // Child band at the bottom-right, parent band (half size, rounded up) at the origin.
            let band = Rect {
                x0: w + 3,
                y0: h + 3,
                w,
                h,
            };
            let parent = Rect {
                x0: 0,
                y0: 0,
                w: w.div_ceil(2),
                h: h.div_ceil(2),
            };
            let density = [0, 1, 5, 30, 90, 100][case % 6];
            let max = [1, 3, 100, 65535][case % 4];
            fill(&mut rng, &mut plane, band, density, max);
            fill(&mut rng, &mut plane, parent, density, max);
            round_trip(&plane, band, None, block_log2);
            round_trip(&plane, band, Some(parent), block_log2);
        }
    }

    #[test]
    fn sparse_bands_are_cheap() {
        let mut rng = SplitMix64::new(4);
        let mut plane = Plane::new(640, 360);
        let band = Rect {
            x0: 0,
            y0: 0,
            w: 640,
            h: 360,
        };
        assert!(round_trip(&plane, band, None, 6) <= 2, "an all-zero band");
        for _ in 0..20 {
            let (x, y) = (rng.below(640) as usize, rng.below(360) as usize);
            plane.set(x, y, 1);
        }
        // 20 isolated ones among 230k coefficients carry about 40 bytes of information.
        let bytes = round_trip(&plane, band, None, 6);
        assert!(bytes < 80, "{bytes} bytes");
    }

    /// The SIMD guide classes match the portable rule on odd sizes and extreme values.
    #[test]
    fn guide_classes_match_portable() {
        let mut rng = SplitMix64::new(5);
        for _ in 0..200 {
            let (w, h) = (rng.range_usize(1, 70), rng.range_usize(1, 40));
            let mut plane = Plane::new(w + 3, h + 2);
            for v in plane.rows_mut().flatten() {
                *v = match rng.below(8) {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => rng.below(400) as i16 - 200,
                };
            }
            let band = Rect { x0: 1, y0: 2, w, h };
            let step = 4096 * (1 + rng.below(80));
            let guide = Guide::build(&plane, band, step, 1);
            let at_least = |t: u64| t.div_ceil(4096).min(u64::from(u16::MAX)) as u16;
            let t = [
                at_least(step.div_ceil(4)),
                at_least(step),
                at_least(4 * step),
            ];
            let mut want = vec![0u8; w.div_ceil(2) * h.div_ceil(2)];
            guide_classes(&plane, band, t, &mut want);
            assert_eq!(guide.levels[0].class, want, "{w}x{h}");
        }
    }

    #[test]
    fn garbage_decodes_without_panicking() {
        let mut rng = SplitMix64::new(8);
        for _ in 0..200 {
            let data: Vec<u8> = (0..rng.below(64)).map(|_| rng.next_u64() as u8).collect();
            let (w, h) = (rng.range_usize(1, 90), rng.range_usize(1, 90));
            let mut plane = Plane::new(w, h);
            let mut dec = RangeDecoder::new(&data);
            let mut ctx = Contexts::new(rng.below(2) == 0);
            code_band(
                &mut dec,
                &mut ctx,
                &mut plane,
                Rect { x0: 0, y0: 0, w, h },
                None,
                None,
                None,
                6,
                false,
            );
            assert!(plane.rows().flatten().all(|v| v.unsigned_abs() < 1 << 16));
        }
    }
}

#[cfg(test)]
mod guide_bench {
    use super::*;

    #[test]
    #[ignore]
    fn guide_speed() {
        let (w, h) = (1920, 1080);
        let mut p = Plane::<i16>::new(w, h);
        let mut rng = crate::rng::SplitMix64::new(3);
        for y in 0..h {
            for v in p.row_span_mut(y, 0, w) {
                *v = rng.range_i32(-300, 300) as i16;
            }
        }
        let band = Rect {
            x0: 960,
            y0: 540,
            w: 960,
            h: 540,
        };
        for _ in 0..5 {
            let s = std::time::Instant::now();
            for _ in 0..50 {
                std::hint::black_box(Guide::build(&p, band, 40000, 6));
            }
            eprintln!(
                "guide 960x540: {:.3} ms",
                s.elapsed().as_secs_f64() / 50.0 * 1e3
            );
        }
    }
}
