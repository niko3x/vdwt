//! Motion: the node grid, global motion, the motion section of P-frames, and motion compensation
//! (DESIGN.md §6).
//!
//! Motion is a smooth field. Nodes sit every `2^log2` luma pixels (32 by default) and carry a
//! vector in 1/8 luma pel. Between nodes the field is the bilinear interpolation of the four
//! surrounding vectors, so it is continuous everywhere and the prediction has no block edges.
//!
//! A [`GlobalMotion`] model, the vectors at the grid's four corners, predicts every node's vector
//! by bilinear interpolation; nodes code only their difference from it. That represents any
//! affine motion exactly (pan, yaw, zoom), plus a bilinear twist that approximates mild
//! perspective. Each node also carries an intra flag: the prediction's weight is zero there and
//! ramps back up linearly towards the neighbouring nodes, so occlusions and new content get a soft
//! transition instead of a block edge.
//!
//! The encoder side lives in [`pyramid`], [`gme`] (global motion estimation) and [`mesh`] (node
//! refinement); [`warp`] is the normative motion compensation.

pub(crate) mod gme;
pub(crate) mod mesh;
mod obmc;
#[cfg(target_arch = "aarch64")]
mod obmc_neon;
#[cfg(target_arch = "x86_64")]
mod obmc_x86;
pub(crate) mod partition;
pub(crate) mod pyramid;
pub(crate) mod refine;
pub(crate) mod tree;
pub mod warp;
#[cfg(target_arch = "x86_64")]
mod warp_avx2;
#[cfg(target_arch = "aarch64")]
mod warp_neon;

use crate::entropy::{Cdf, RangeDecoder, RangeEncoder, SymbolCoder};
use tree::TreeContexts;

pub use warp::Filter;

/// How node vectors become a prediction (DESIGN.md §6.3), chosen per P-frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Prediction {
    /// Interpolate the vectors bilinearly and sample the reference once per pixel: exact for
    /// smooth camera motion.
    #[default]
    Warp,
    /// Translate by each of the cell's four node vectors and blend the four samples with the
    /// same weights: follows motion discontinuities (parallax, object edges).
    Overlapped,
}

impl Prediction {
    pub const ALL: [Prediction; 2] = [Prediction::Warp, Prediction::Overlapped];

    pub fn name(self) -> &'static str {
        match self {
            Prediction::Warp => "warp",
            Prediction::Overlapped => "overlapped",
        }
    }

    pub fn from_name(name: &str) -> Option<Prediction> {
        Self::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(name))
    }
}

/// Largest node vector component, in 1/8 luma pel (4096 pixels).
pub const MAX_MV: i32 = (1 << 15) - 1;
/// Largest global corner vector component, in 1/8 luma pel. Smaller than [`MAX_MV`] so that a
/// frame-to-frame change always fits the 16-symbol magnitude classes.
pub const MAX_GLOBAL_MV: i32 = (1 << 14) - 1;
/// Grid spacings: 16, 32 or 64 luma pixels.
pub const MESH_LOG2: std::ops::RangeInclusive<u32> = 3..=6;

/// Node layout: nodes at every multiple of the spacing, enough to cover the frame. The last node
/// row and column may lie beyond the frame's edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Luma spacing as log2.
    pub log2: u32,
    /// Nodes per row.
    pub nx: usize,
    /// Node rows.
    pub ny: usize,
}

impl Grid {
    pub fn new(width: usize, height: usize, log2: u32) -> Self {
        assert!(MESH_LOG2.contains(&log2), "mesh spacing 2^{log2}");
        Self {
            log2,
            nx: width.div_ceil(1 << log2) + 1,
            ny: height.div_ceil(1 << log2) + 1,
        }
    }

    /// Luma pixels between nodes.
    pub fn spacing(&self) -> usize {
        1 << self.log2
    }

    pub fn nodes(&self) -> usize {
        self.nx * self.ny
    }

    /// The four corners of the grid in luma pixels, in [`GlobalMotion::corners`] order.
    pub fn corners(&self) -> [(i64, i64); 4] {
        let (x, y) = (
            ((self.nx - 1) << self.log2) as i64,
            ((self.ny - 1) << self.log2) as i64,
        );
        [(0, 0), (x, 0), (0, y), (x, y)]
    }
}

/// Global motion: the vectors at the four corners of the node grid (top-left, top-right,
/// bottom-left, bottom-right) in 1/8 luma pel. A vector points from a pixel of the current frame
/// to where it is in the reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlobalMotion {
    pub corners: [[i32; 2]; 4],
}

impl GlobalMotion {
    /// The predicted vector of node `(i, j)`: the corners interpolated bilinearly and rounded to
    /// nearest (normative).
    pub fn predict(&self, grid: &Grid, i: usize, j: usize) -> [i32; 2] {
        let (nx, ny) = ((grid.nx - 1) as i64, (grid.ny - 1) as i64);
        let (i, j) = (i as i64, j as i64);
        let w = [(nx - i) * (ny - j), i * (ny - j), (nx - i) * j, i * j];
        let den = nx * ny;
        std::array::from_fn(|c| {
            let num: i64 = (0..4).map(|k| w[k] * i64::from(self.corners[k][c])).sum();
            (2 * num + den).div_euclid(2 * den) as i32
        })
    }

    /// The model's displacement at a luma position, in pixels (for analysis and estimation).
    pub fn displacement(&self, grid: &Grid, x: f64, y: f64) -> [f64; 2] {
        let [_, (gx, _), (_, gy), _] = grid.corners();
        let (u, v) = (x / gx as f64, y / gy as f64);
        let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), (1.0 - u) * v, u * v];
        std::array::from_fn(|c| {
            (0..4)
                .map(|k| w[k] * f64::from(self.corners[k][c]))
                .sum::<f64>()
                / 8.0
        })
    }

    pub fn is_zero(&self) -> bool {
        self.corners == [[0; 2]; 4]
    }
}

/// A P-frame's motion: one vector (1/8 luma pel) and one intra flag per node, row-major, and the
/// quadtree partition they were coded with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionField {
    pub grid: Grid,
    pub mv: Vec<[i32; 2]>,
    pub intra: Vec<bool>,
    /// Size of each node's leaf, as log2 of its side in nodes.
    pub depth: Vec<u8>,
}

impl MotionField {
    /// No motion, every node inter.
    pub fn new(grid: Grid) -> Self {
        Self {
            grid,
            mv: vec![[0; 2]; grid.nodes()],
            intra: vec![false; grid.nodes()],
            depth: vec![tree::root_log2(grid.log2) as u8; grid.nodes()],
        }
    }

    /// Every node at its global prediction, inter.
    pub fn set_global(&mut self, global: &GlobalMotion) {
        let grid = self.grid;
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                self.mv[j * grid.nx + i] = global.predict(&grid, i, j);
            }
        }
        self.intra.fill(false);
        self.depth.fill(tree::root_log2(grid.log2) as u8);
    }

    #[inline]
    pub fn node(&self, i: usize, j: usize) -> usize {
        j * self.grid.nx + i
    }
}

/// Coding state of the motion section.
#[derive(Clone, Debug, Default)]
pub(crate) struct MotionCoder {
    /// Global prediction per node, per frame.
    pred: Vec<[i32; 2]>,
    /// Whether the motion-tree CDFs carry across frames (fixed-qp streams).
    continuous: bool,
    /// Whether P-frames predict node offsets from the previous coded frame.
    temporal: bool,
    /// The previous coded frame's offsets from its global prediction, per node.
    prev_delta: Vec<[i32; 2]>,
    /// The carried state; reset at the start of the stream only when `continuous`.
    tree: TreeContexts,
}

/// Bit length: 0 for 0, else ⌊log2 m⌋ + 1.
fn bit_length(m: u32) -> u32 {
    32 - m.leading_zeros()
}

/// Starting distribution of global-vector change classes: small changes are the norm.
fn global_cdf() -> Cdf {
    let mut w = [0u64; 16];
    let mut v = 1u64 << 40;
    for x in &mut w {
        *x = v.max(1);
        v = v * 150 / 256;
    }
    Cdf::from_weights(&w)
}

/// Code the global model (eight corner changes) with a fresh CDF.
fn code_global<C: SymbolCoder>(coder: &mut C, prev: &GlobalMotion, global: &mut GlobalMotion) {
    let mut cdf = global_cdf();
    for (corner, prev) in global.corners.iter_mut().zip(&prev.corners) {
        for (v, &p) in corner.iter_mut().zip(prev) {
            let delta = if C::ENCODER { *v - p } else { 0 };
            let m = delta.unsigned_abs();
            let mut class = bit_length(m) as usize;
            debug_assert!(class < 16, "global vector change {delta}");
            coder.symbol(&mut cdf, &mut class);
            let mut delta = 0;
            if class > 0 {
                let e = class as u32;
                let mut raw = ((m.wrapping_sub(1 << (e - 1))) << 1) | u32::from(*v < p);
                coder.bits(e, &mut raw);
                let mag = ((1u32 << (e - 1)) | (raw >> 1)) as i32;
                delta = if raw & 1 == 1 { -mag } else { mag };
            }
            if !C::ENCODER {
                *v = (p + delta).clamp(-MAX_GLOBAL_MV, MAX_GLOBAL_MV);
            }
        }
    }
}

/// The global prediction per node.
fn prediction(global: &GlobalMotion, grid: &Grid) -> Vec<[i32; 2]> {
    let mut pred = Vec::with_capacity(grid.nodes());
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            pred.push(global.predict(grid, i, j));
        }
    }
    pred
}

impl MotionCoder {
    pub fn new(continuous: bool) -> Self {
        Self {
            continuous,
            ..Self::default()
        }
    }

    /// Enable or disable the temporal node predictor.
    pub fn set_temporal(&mut self, temporal: bool) {
        self.temporal = temporal;
    }

    /// Forget the previous frame's offsets (an intra frame has no motion field).
    pub fn reset(&mut self) {
        self.prev_delta.clear();
    }

    /// Codes or decodes a P-frame's motion: the global model as a change from `prev`, then the
    /// node quadtree ([`tree`]). On decoding, `global` and `field` receive the result; every
    /// value is clamped to its legal range, so corrupt input cannot overflow the warp. On
    /// encoding, `field` is left as the decoder will see it.
    /// Codes or decodes a P-frame's motion: the global model as a change from `prev`, then the
    /// node quadtree ([`tree`]). The tests use this spatial-only path.
    pub fn code<C: SymbolCoder>(
        &mut self,
        coder: &mut C,
        prev: &GlobalMotion,
        global: &mut GlobalMotion,
        field: &mut MotionField,
    ) {
        code_global(coder, prev, global);
        self.pred = prediction(global, &field.grid);
        if self.continuous {
            tree::code_with(coder, &mut self.tree, field, &self.pred, None);
        } else {
            let mut ctx = tree::TreeContexts::default();
            tree::code_with(coder, &mut ctx, field, &self.pred, None);
        }
        self.update_prev(field);
    }

    /// Encodes a motion section. With the temporal predictor enabled, codes it twice — once
    /// from the spatial median and once from the previous frame's offsets — and keeps the
    /// shorter. The section's first bit says which predictor was used.
    pub fn encode(
        &mut self,
        prev: &GlobalMotion,
        global: &GlobalMotion,
        field: &mut MotionField,
    ) -> Vec<u8> {
        let grid = field.grid;
        self.pred = prediction(global, &grid);
        let base = if self.continuous {
            self.tree.clone()
        } else {
            tree::TreeContexts::default()
        };
        let available = self.temporal && self.prev_delta.len() == grid.nodes();
        let modes = if self.temporal { 2 } else { 1 };
        let mut best: Option<(Vec<u8>, MotionField, tree::TreeContexts)> = None;
        for mode in 0..modes {
            let use_temporal = mode == 1;
            if use_temporal && !available {
                continue;
            }
            let mut trial = field.clone();
            let mut ctx = base.clone();
            let mut enc = RangeEncoder::new();
            let mut g = *global;
            code_global(&mut enc, prev, &mut g);
            if self.temporal {
                let mut selector = u32::from(use_temporal);
                enc.bits(1, &mut selector);
            }
            let temporal = use_temporal.then_some(self.prev_delta.as_slice());
            tree::code_with(&mut enc, &mut ctx, &mut trial, &self.pred, temporal);
            let bytes = enc.finish();
            if best.as_ref().is_none_or(|b| bytes.len() < b.0.len()) {
                best = Some((bytes, trial, ctx));
            }
        }
        let (bytes, trial, ctx) = best.expect("at least the spatial pass");
        *field = trial;
        if self.continuous {
            self.tree = ctx;
        }
        self.update_prev(field);
        bytes
    }

    /// Decodes a motion section written by [`Self::encode`].
    pub fn decode(
        &mut self,
        bytes: &[u8],
        prev: &GlobalMotion,
        global: &mut GlobalMotion,
        field: &mut MotionField,
    ) {
        let mut dec = RangeDecoder::new(bytes);
        code_global(&mut dec, prev, global);
        self.pred = prediction(global, &field.grid);
        let mut selector = 0u32;
        if self.temporal {
            dec.bits(1, &mut selector);
        }
        let available = self.temporal && self.prev_delta.len() == field.grid.nodes();
        let temporal = (selector == 1 && available).then_some(self.prev_delta.as_slice());
        if self.continuous {
            tree::code_with(&mut dec, &mut self.tree, field, &self.pred, temporal);
        } else {
            let mut ctx = tree::TreeContexts::default();
            tree::code_with(&mut dec, &mut ctx, field, &self.pred, temporal);
        }
        self.update_prev(field);
    }

    /// Store the decoded offsets for the next frame's temporal predictor.
    fn update_prev(&mut self, field: &MotionField) {
        if !self.temporal {
            return;
        }
        self.prev_delta.resize(field.grid.nodes(), [0; 2]);
        for n in 0..field.grid.nodes() {
            self.prev_delta[n] = [
                field.mv[n][0] - self.pred[n][0],
                field.mv[n][1] - self.pred[n][1],
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{RangeDecoder, RangeEncoder};
    use crate::rng::SplitMix64;

    #[test]
    fn grid_covers_the_frame() {
        let g = Grid::new(1280, 720, 5);
        assert_eq!((g.nx, g.ny), (41, 24));
        assert_eq!(g.corners()[3], (1280, 736));
        let g = Grid::new(1, 1, 6);
        assert_eq!((g.nx, g.ny), (2, 2));
    }

    #[test]
    fn global_prediction_reproduces_affine_motion() {
        let grid = Grid::new(640, 360, 5);
        // (12 + x/40 - y/100, -5 + y/50) pixels, in 1/8 pel.
        let exact = |x: f64, y: f64| [96.0 + x / 5.0 - 2.0 * y / 25.0, -40.0 + 4.0 * y / 25.0];
        let global = GlobalMotion {
            corners: grid
                .corners()
                .map(|(x, y)| exact(x as f64, y as f64).map(|v| v.round() as i32)),
        };
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let p = global.predict(&grid, i, j);
                let e = exact((i * 32) as f64, (j * 32) as f64);
                for c in 0..2 {
                    assert!(
                        (f64::from(p[c]) - e[c]).abs() <= 1.0,
                        "node ({i}, {j}): {p:?} vs {e:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn motion_round_trips() {
        let mut rng = SplitMix64::new(11);
        for case in 0..40 {
            let (w, h) = (rng.range_usize(1, 700), rng.range_usize(1, 400));
            let grid = Grid::new(w, h, 4 + (case % 3) as u32);
            let mut prev = GlobalMotion::default();
            let mut global = GlobalMotion::default();
            for c in &mut prev.corners {
                *c = [rng.range_i32(-3000, 3000), rng.range_i32(-3000, 3000)];
            }
            for c in &mut global.corners {
                *c = [rng.range_i32(-3000, 3000), rng.range_i32(-3000, 3000)];
            }
            let mut field = MotionField::new(grid);
            field.set_global(&global);
            let density = [0, 3, 30, 100][case % 4];
            for n in 0..grid.nodes() {
                if rng.below(100) < density {
                    // Offsets from the global prediction are whole quarter pels.
                    field.mv[n][0] += 2 * rng.range_i32(-20, 20);
                    field.mv[n][1] += 2 * rng.range_i32(-20, 20);
                    field.intra[n] = rng.below(4) == 0;
                }
            }
            for d in &mut field.depth {
                *d = rng.below(4) as u8;
            }
            let mut coder = MotionCoder::new(false);
            let mut enc = RangeEncoder::new();
            let mut g = global;
            // Encoding leaves the field as decoded: each leaf takes its top-left node's values.
            coder.code(&mut enc, &prev, &mut g, &mut field);
            let bytes = enc.finish();

            let mut dec_global = GlobalMotion::default();
            let mut dec_field = MotionField::new(grid);
            let mut coder = MotionCoder::new(false);
            coder.code(
                &mut RangeDecoder::new(&bytes),
                &prev,
                &mut dec_global,
                &mut dec_field,
            );
            assert_eq!(dec_global, global, "case {case}");
            assert_eq!(dec_field, field, "case {case}");
        }
    }
}
