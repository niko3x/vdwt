//! The motion section's node vectors: a quadtree of leaves over the node grid (DESIGN.md §6.2).
//!
//! The grid is cut into roots of 64×64 luma pixels (`2^R × 2^R` nodes), in raster order. Each
//! root splits recursively into four quadrants, in Z order, down to single nodes. Every leaf
//! gives all its nodes the same offset `δ` from the global prediction (or makes them intra), in
//! one of five modes:
//!
//! - **Global:** `δ = 0`;
//! - **Left** / **Up:** the `δ` of the node left of (above) the leaf's top-left node;
//! - **New:** a predicted `δ` plus a coded difference in quarter pels, per component a magnitude
//!   class (16-ary symbol) with raw mantissa and sign bits. The prediction is the component-wise median of the
//!   left, upper and upper-right neighbours' `δ` when all three are coded, else the first coded
//!   of left and upper, else zero;
//! - **Intra:** every node of the leaf is an intra node (`δ = 0` for its neighbours).
//!
//! Background that follows the global model costs a fraction of a bit per root; regions moving
//! together (an object, a layer of parallax) cost one vector and then a mode symbol per leaf;
//! fine leaves go only where motion changes.

use super::{MAX_MV, MotionField};
use crate::entropy::{Cdf, SymbolCoder};

/// Leaf modes, as coded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Global = 0,
    Left = 1,
    Up = 2,
    New = 3,
    Intra = 4,
}

/// Split contexts: (level − 1) capped at 2, by how many of the left and upper neighbours sit in
/// finer leaves (0..=2).
const SPLIT_CTX: usize = 9;
/// Mode contexts: leaf size group (one node, up to 4×4, larger) by how many of the left and
/// upper neighbours moved or are intra (0..=2).
const MODE_CTX: usize = 9;
/// Difference-class contexts: component by leaf size group (one node, larger).
const CLASS_CTX: usize = 4;
/// Largest coded difference component: its class must stay below 16.
pub(crate) const MAX_DIFF: i32 = (1 << 15) - 1;

/// The adaptive state of one motion section.
#[derive(Clone, Debug)]
pub(crate) struct TreeContexts {
    split: [Cdf; SPLIT_CTX],
    mode: [Cdf; MODE_CTX],
    class: [Cdf; CLASS_CTX],
}

/// Initial split weights (‰) of the first two levels' contexts, from the training clips: a
/// region splits more often where its neighbours did.
const SPLIT_WEIGHTS: [[u64; 2]; 6] = [
    [700, 300],
    [475, 525],
    [266, 734],
    [835, 165],
    [404, 596],
    [127, 873],
];
/// Initial mode weights (‰) of the two smaller leaf groups' contexts, from the training clips.
/// With no moved neighbour, Left and Up cannot occur (they would be Global).
const MODE_WEIGHTS: [[u64; 5]; 6] = [
    [577, 1, 1, 422, 3],
    [340, 211, 159, 290, 3],
    [132, 317, 176, 374, 3],
    [752, 1, 1, 248, 3],
    [340, 207, 140, 314, 3],
    [110, 299, 175, 416, 3],
];
/// Initial difference-class weights (‰), from the training clips.
const CLASS_WEIGHTS: [[u64; 16]; CLASS_CTX] = [
    [230, 225, 153, 143, 93, 70, 53, 26, 7, 1, 1, 1, 1, 1, 1, 1],
    [255, 197, 141, 149, 95, 68, 54, 30, 10, 1, 1, 1, 1, 1, 1, 1],
    [222, 263, 162, 141, 92, 66, 39, 13, 2, 1, 1, 1, 1, 1, 1, 1],
    [244, 215, 156, 159, 105, 68, 37, 14, 3, 1, 1, 1, 1, 1, 1, 1],
];

impl Default for TreeContexts {
    fn default() -> Self {
        Self {
            split: std::array::from_fn(|c| match SPLIT_WEIGHTS.get(c) {
                Some(w) => Cdf::from_weights(w),
                // Coarser levels (finer grids): finer neighbours make a split likelier.
                None => {
                    let p = 10 + 20 * (c % 3) as u64;
                    Cdf::from_weights(&[100 - p, p])
                }
            }),
            mode: std::array::from_fn(|c| match MODE_WEIGHTS.get(c) {
                Some(w) => Cdf::from_weights(w),
                None => {
                    let near = (c % 3) as u64;
                    Cdf::from_weights(&[60 - 15 * near, 10 + 10 * near, 10 + 10 * near, 15, 5])
                }
            }),
            class: std::array::from_fn(|c| Cdf::from_weights(&CLASS_WEIGHTS[c])),
        }
    }
}

/// Root size in nodes, as log2, for a grid of `2^grid_log2` pixels: roots are 64 pixels.
pub(crate) fn root_log2(grid_log2: u32) -> u32 {
    6u32.saturating_sub(grid_log2)
}

/// Codes or decodes a field's node offsets from the global prediction `pred` (one vector per
/// node). The encoder reads `field.mv`, `field.intra` and the partition in `field.depth`, taking
/// each leaf's offset and intra flag from its top-left node; the decoder writes them, with every
/// vector clamped to its legal range. Both leave the field as decoded.
pub(crate) fn code<C: SymbolCoder>(
    coder: &mut C,
    ctx: &mut TreeContexts,
    field: &mut MotionField,
    pred: &[[i32; 2]],
) {
    code_with(coder, ctx, field, pred, None);
}

/// [`code`] with a temporal predictor: `temporal[n]` is the previous coded frame's offset at
/// node `n`, used as one of the three values the New mode's median predicts from.
pub(crate) fn code_with<C: SymbolCoder>(
    coder: &mut C,
    ctx: &mut TreeContexts,
    field: &mut MotionField,
    pred: &[[i32; 2]],
    temporal: Option<&[[i32; 2]]>,
) {
    let roots = field.grid.ny.div_ceil(1 << root_log2(field.grid.log2));
    code_rows_with(coder, ctx, field, pred, 0..roots, temporal);
}

/// [`code`] for the roots in root rows `rows` alone, as if no node outside them were coded: no
/// neighbour above the first row predicts or conditions anything. Only those rows' nodes are
/// read (encoder) or written (decoder).
pub(crate) fn code_rows<C: SymbolCoder>(
    coder: &mut C,
    ctx: &mut TreeContexts,
    field: &mut MotionField,
    pred: &[[i32; 2]],
    rows: std::ops::Range<usize>,
) {
    code_rows_with(coder, ctx, field, pred, rows, None);
}

fn code_rows_with<C: SymbolCoder>(
    coder: &mut C,
    ctx: &mut TreeContexts,
    field: &mut MotionField,
    pred: &[[i32; 2]],
    rows: std::ops::Range<usize>,
    temporal: Option<&[[i32; 2]]>,
) {
    let grid = field.grid;
    let r = root_log2(grid.log2);
    let size = 1usize << r;
    let (j_lo, j_hi) = (
        (rows.start * size).min(grid.ny),
        (rows.end * size).min(grid.ny),
    );
    let nodes = j_lo * grid.nx..j_hi * grid.nx;
    let mut state = State {
        nx: grid.nx,
        ny: grid.ny,
        delta: vec![[0; 2]; grid.nodes()],
        coded: vec![false; grid.nodes()],
        moved: vec![false; grid.nodes()],
    };
    if C::ENCODER {
        for n in nodes.clone() {
            if !field.intra[n] {
                state.delta[n] = [field.mv[n][0] - pred[n][0], field.mv[n][1] - pred[n][1]];
            }
        }
    }
    for j0 in (j_lo..j_hi).step_by(size) {
        for i0 in (0..grid.nx).step_by(size) {
            quad(coder, ctx, field, &mut state, i0, j0, r, temporal);
        }
    }
    for n in nodes {
        let d = state.delta[n];
        field.mv[n] = std::array::from_fn(|c| (pred[n][c] + d[c]).clamp(-MAX_MV, MAX_MV));
    }
}

struct State {
    nx: usize,
    ny: usize,
    /// Every node's offset (zero for intra nodes); on decoding, as far as decoded.
    delta: Vec<[i32; 2]>,
    coded: Vec<bool>,
    /// Nodes in leaves that are not Global.
    moved: Vec<bool>,
}

impl State {
    fn at(&self, i: usize, j: usize) -> Option<usize> {
        (i < self.nx && j < self.ny).then(|| j * self.nx + i)
    }

    /// The coded node at `(i, j)`, if any.
    fn coded_at(&self, i: Option<usize>, j: Option<usize>) -> Option<usize> {
        let n = self.at(i?, j?)?;
        self.coded[n].then_some(n)
    }
}

#[allow(clippy::too_many_arguments)]
fn quad<C: SymbolCoder>(
    coder: &mut C,
    ctx: &mut TreeContexts,
    field: &mut MotionField,
    st: &mut State,
    i0: usize,
    j0: usize,
    level: u32,
    temporal: Option<&[[i32; 2]]>,
) {
    if i0 >= st.nx || j0 >= st.ny {
        return;
    }
    let size = 1usize << level;
    let left = st.coded_at(i0.checked_sub(1), Some(j0));
    let up = st.coded_at(Some(i0), j0.checked_sub(1));
    let n0 = j0 * st.nx + i0;
    if level > 0 {
        let finer = |n: Option<usize>| usize::from(n.is_some_and(|n| field.depth[n] < level as u8));
        let c = (level as usize - 1).min(2) * 3 + finer(left) + finer(up);
        let mut split = usize::from(C::ENCODER && field.depth[n0] < level as u8);
        coder.symbol(&mut ctx.split[c], &mut split);
        if split == 1 {
            let h = size / 2;
            for (di, dj) in [(0, 0), (h, 0), (0, h), (h, h)] {
                quad(coder, ctx, field, st, i0 + di, j0 + dj, level - 1, temporal);
            }
            return;
        }
    }
    let delta_of = |n: Option<usize>| n.map_or([0; 2], |n| st.delta[n]);
    let (dl, du) = (delta_of(left), delta_of(up));
    let group = match level {
        0 => 0,
        1 | 2 => 1,
        _ => 2,
    };
    let moved = |n: Option<usize>| usize::from(n.is_some_and(|n| st.moved[n]));
    let mut mode = Mode::Global as usize;
    let mut delta = [0; 2];
    if C::ENCODER {
        delta = st.delta[n0];
        mode = if field.intra[n0] {
            Mode::Intra
        } else if delta == [0; 2] {
            Mode::Global
        } else if left.is_some() && delta == dl {
            Mode::Left
        } else if up.is_some() && delta == du {
            Mode::Up
        } else {
            Mode::New
        } as usize;
    }
    coder.symbol(
        &mut ctx.mode[group * 3 + moved(left) + moved(up)],
        &mut mode,
    );
    let intra = mode == Mode::Intra as usize;
    match mode {
        1 => delta = dl,
        2 => delta = du,
        3 => {
            let upright = st.coded_at(Some(i0 + size), j0.checked_sub(1));
            let p = match temporal {
                Some(t) => t[j0 * st.nx + i0],
                None => match (left, up, upright) {
                    (Some(_), Some(_), Some(ur)) => {
                        let dr = st.delta[ur];
                        std::array::from_fn(|c| median(dl[c], du[c], dr[c]))
                    }
                    (Some(_), _, _) => dl,
                    (None, Some(_), _) => du,
                    _ => [0; 2],
                },
            };
            for c in 0..2 {
                let k = c * 2 + usize::from(level > 0);
                // Offsets are whole quarter pels (even in 1/8 pel).
                debug_assert!(
                    !C::ENCODER || (delta[c] - p[c]) % 2 == 0,
                    "offset {delta:?}"
                );
                let mut d = if C::ENCODER { (delta[c] - p[c]) / 2 } else { 0 };
                debug_assert!(d.abs() <= MAX_DIFF, "motion difference {d}");
                let m = d.unsigned_abs();
                let mut class = (32 - m.leading_zeros()) as usize;
                coder.symbol(&mut ctx.class[k], &mut class);
                let negative = d < 0;
                d = 0;
                if class > 0 {
                    let e = class as u32;
                    let mut raw = ((m.wrapping_sub(1 << (e - 1))) << 1) | u32::from(negative);
                    coder.bits(e, &mut raw);
                    let mag = ((1u32 << (e - 1)) | (raw >> 1)) as i32;
                    d = if raw & 1 == 1 { -mag } else { mag };
                }
                // The encoder too: an odd offset becomes the one the decoder sees.
                delta[c] = (p[c] + 2 * d).clamp(-2 * MAX_MV, 2 * MAX_MV) & !1;
            }
        }
        _ => {}
    }
    if intra || mode == Mode::Global as usize {
        delta = [0; 2];
    }
    for j in j0..(j0 + size).min(st.ny) {
        for i in i0..(i0 + size).min(st.nx) {
            let n = j * st.nx + i;
            st.delta[n] = delta;
            st.coded[n] = true;
            st.moved[n] = mode != Mode::Global as usize;
            field.intra[n] = intra;
            field.depth[n] = level as u8;
        }
    }
}

fn median(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

/// Approximate bits of a leaf with offset `delta` given its neighbours' offsets, for the
/// encoder's decisions: the mode symbol and, for a new vector, its difference from `pred`.
pub(crate) fn leaf_bits(
    delta: [i32; 2],
    left: Option<[i32; 2]>,
    up: Option<[i32; 2]>,
    pred: [i32; 2],
) -> f64 {
    if delta == [0; 2] {
        1.0
    } else if left == Some(delta) || up == Some(delta) {
        2.5
    } else {
        2.5 + (0..2)
            .map(|c| {
                let d = (delta[c] - pred[c]).unsigned_abs() / 2;
                if d == 0 {
                    1.0
                } else {
                    2.0 + 2.0 * f64::from(32 - d.leading_zeros()) - 1.0
                }
            })
            .sum::<f64>()
    }
}

/// The prediction a New leaf at `(i0, j0)` of `size` nodes would use, from offsets known so far
/// (`known[n]`): the same rule as [`code`].
pub(crate) fn predict_new(
    delta: &[[i32; 2]],
    known: &[bool],
    nx: usize,
    ny: usize,
    i0: usize,
    j0: usize,
    size: usize,
) -> [i32; 2] {
    let at = |i: Option<usize>, j: Option<usize>| -> Option<usize> {
        let (i, j) = (i?, j?);
        (i < nx && j < ny && known[j * nx + i]).then_some(j * nx + i)
    };
    let left = at(i0.checked_sub(1), Some(j0));
    let up = at(Some(i0), j0.checked_sub(1));
    let upright = at(Some(i0 + size), j0.checked_sub(1));
    match (left, up, upright) {
        (Some(l), Some(u), Some(r)) => {
            std::array::from_fn(|c| median(delta[l][c], delta[u][c], delta[r][c]))
        }
        (Some(l), _, _) => delta[l],
        (None, Some(u), _) => delta[u],
        _ => [0; 2],
    }
}
