//! B-frames (DESIGN.md §6.9): the parts the encoder and decoder share.
//!
//! A group of `n` frames (2 or 4) is coded anchor first: its last frame as a P-frame whose
//! header keeps the decoder's picture (the past anchor, position 0), then its B-frames at
//! positions `k` between the anchors (the anchor is position `n`). A B-frame predicts from two
//! references at positions `past < k < next`: either anchor, or the group's middle B-frame,
//! which a group of 4 keeps as a reference (its header's keep flag). B-frames change no other
//! decoder state.
//!
//! Each direction's vectors are predicted from the anchor's own field (intra nodes taking its
//! global motion), which spans the group, scaled to the distance: `(k − position)/n`, rounded
//! half up in integers. A searched direction codes its field's offsets from that prediction with
//! the motion quadtree ([`tree::code`]); a direct one uses the prediction as it is, for no bits.
//! After an intra anchor the prediction is zero.
//!
//! The B extension follows the frame header: the mode, `k`, `n`, and the past and next
//! reference positions (a byte each), then a motion section (length-prefixed) per searched
//! direction, past first. The blend mode codes both fields, then (length-prefixed) every root's
//! choice of past, next or both, and blends the two predictions with weights interpolated
//! between nodes ([`blend`]).

use crate::Frame;
use crate::entropy::{Cdf, SymbolCoder};
use crate::motion::tree::root_log2;
use crate::motion::{GlobalMotion, Grid, MotionField};

/// A root's blend choice: from the past anchor, the next, or both averaged.
pub const ROOT_PAST: u8 = 0;
pub const ROOT_NEXT: u8 = 1;
pub const ROOT_BOTH: u8 = 2;

/// Roots across and down a grid.
pub(crate) fn root_count(grid: &Grid) -> (usize, usize) {
    let size = 1usize << root_log2(grid.log2);
    (grid.nx.div_ceil(size), grid.ny.div_ceil(size))
}

/// Codes or decodes every root's blend choice in raster order, each in the context of the one to
/// its left (adaptive). Decoded values are always valid.
pub(crate) fn code_root_modes<C: SymbolCoder>(coder: &mut C, modes: &mut [u8], across: usize) {
    let mut cdfs: [Cdf; 3] = std::array::from_fn(|_| Cdf::from_weights(&[1, 1, 1]));
    for k in 0..modes.len() {
        let left = if k % across == 0 {
            ROOT_BOTH
        } else {
            modes[k - 1]
        };
        let mut s = usize::from(modes[k]);
        coder.symbol(&mut cdfs[usize::from(left)], &mut s);
        modes[k] = s.min(2) as u8;
    }
}

/// `past` replaced by itself and `next` weighted per node by its root's choice (64 past, 0
/// next, 32 both), the weights interpolated bilinearly across each cell (normative, integers
/// only).
pub(crate) fn blend(past: &mut Frame, next: &Frame, grid: &Grid, modes: &[u8]) {
    let r = root_log2(grid.log2);
    let (across, _) = root_count(grid);
    let weight = |i: usize, j: usize| -> u32 {
        match modes[(j >> r) * across + (i >> r)] {
            ROOT_PAST => 64,
            ROOT_NEXT => 0,
            _ => 32,
        }
    };
    for (k, (o, n)) in past.planes_mut().into_iter().zip(next.planes()).enumerate() {
        let c = grid.log2 - u32::from(k > 0);
        let cs = 1u32 << c;
        let shift = 6 + 2 * c;
        let w = o.width();
        for y in 0..o.height() {
            let (j, v) = (y >> c, (y as u32) & (cs - 1));
            let nrow = n.row_span(y, 0, w);
            let orow = o.row_mut(y);
            for x in 0..w {
                let (i, u) = (x >> c, (x as u32) & (cs - 1));
                let wt = (cs - u) * (cs - v) * weight(i, j)
                    + u * (cs - v) * weight(i + 1, j)
                    + (cs - u) * v * weight(i, j + 1)
                    + u * v * weight(i + 1, j + 1);
                let full = 64u32 << (2 * c);
                let s = wt * u32::from(orow[x]) + (full - wt) * u32::from(nrow[x]);
                orow[x] = ((s + (full >> 1)) >> shift) as u8;
            }
        }
    }
}

/// How a B-frame predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiMode {
    /// From the past anchor, by a coded field.
    Past,
    /// From the next anchor, by a coded field.
    Next,
    /// The average of both, by two coded fields.
    Both,
    /// The past anchor, by the scaled prediction alone.
    DirectPast,
    DirectNext,
    DirectBoth,
    /// Both coded fields, and per root past, next or both ([`blend`]).
    Blend,
}

impl BiMode {
    pub const ALL: [BiMode; 7] = [
        BiMode::Past,
        BiMode::Next,
        BiMode::Both,
        BiMode::DirectPast,
        BiMode::DirectNext,
        BiMode::DirectBoth,
        BiMode::Blend,
    ];

    pub fn code(self) -> u8 {
        Self::ALL.iter().position(|&m| m == self).expect("listed") as u8
    }

    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(usize::from(code)).copied()
    }

    /// Whether it predicts from the past anchor, and from the next one.
    pub fn directions(self) -> [bool; 2] {
        match self {
            BiMode::Past | BiMode::DirectPast => [true, false],
            BiMode::Next | BiMode::DirectNext => [false, true],
            BiMode::Both | BiMode::DirectBoth | BiMode::Blend => [true, true],
        }
    }

    /// Whether its fields are coded (else direct).
    pub fn searched(self) -> bool {
        matches!(
            self,
            BiMode::Past | BiMode::Next | BiMode::Both | BiMode::Blend
        )
    }
}

/// `a·num/den` rounded half up, in integers (normative).
fn scale(a: i32, num: i32, den: i32) -> i32 {
    let (a, num, den) = (i64::from(a), i64::from(num), i64::from(den));
    (2 * a * num + den).div_euclid(2 * den) as i32
}

/// Every node's predicted vector towards an anchor `num/den` of the anchor field's span away
/// (`k/n` to the past anchor, `−(n−k)/n` to the next): the anchor's vectors (its global motion's
/// at intra nodes) scaled; zero when the anchor was intra.
pub(crate) fn scaled_vectors(
    anchor: &MotionField,
    global: &GlobalMotion,
    anchor_inter: bool,
    num: i32,
    den: i32,
) -> Vec<[i32; 2]> {
    let grid = anchor.grid;
    (0..grid.nodes())
        .map(|n| {
            if !anchor_inter {
                return [0; 2];
            }
            let v = if anchor.intra[n] {
                global.predict(&grid, n % grid.nx, n / grid.nx)
            } else {
                anchor.mv[n]
            };
            v.map(|c| scale(c, num, den))
        })
        .collect()
}

/// The field of a direct direction: the prediction as it is, one leaf per root, no intra nodes.
pub(crate) fn direct_field(like: &MotionField, pred: &[[i32; 2]]) -> MotionField {
    let mut field = MotionField::new(like.grid);
    field.mv.copy_from_slice(pred);
    field.depth.fill(root_log2(like.grid.log2) as u8);
    field
}

/// `dst = (dst + other + 1) / 2`, sample by sample (normative).
pub(crate) fn average_into(dst: &mut Frame, other: &Frame) {
    for (d, o) in dst.planes_mut().into_iter().zip(other.planes()) {
        let w = d.width();
        for y in 0..d.height() {
            for (a, &b) in d.row_mut(y).iter_mut().zip(o.row_span(y, 0, w)) {
                *a = ((u16::from(*a) + u16::from(b) + 1) >> 1) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_rounds_half_up() {
        assert_eq!(scale(10, 1, 2), 5);
        assert_eq!(scale(5, 1, 2), 3);
        assert_eq!(scale(-5, 1, 2), -2);
        assert_eq!(scale(-6, -1, 2), 3);
        assert_eq!(scale(7, -1, 2), -3);
    }

    #[test]
    fn modes_round_trip() {
        for m in BiMode::ALL {
            assert_eq!(BiMode::from_code(m.code()), Some(m));
        }
        assert_eq!(BiMode::from_code(7), None);
    }
}
