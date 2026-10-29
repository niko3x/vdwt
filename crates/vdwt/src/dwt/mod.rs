//! The discrete wavelet transforms: the JPEG 2000 / HTJ2K reversible 5/3, for lossless coding,
//! and a lifted 9/7 with fraction bits ([`d97`]), for everything else. [`Transform`] runs
//! whichever a stream uses.
//!
//! The rest of this page describes the 5/3, the transform of ISO/IEC 15444-1 Annex F, which
//! HTJ2K (Part 15) keeps unchanged:
//!
//! This is the transform of ISO/IEC 15444-1 Annex F, which HTJ2K (Part 15) keeps unchanged:
//!
//! ```text
//! predict: d[n] = x[2n+1] − ⌊(x[2n] + x[2n+2]) / 2⌋
//! update : s[n] = x[2n]   + ⌊(d[n−1] + d[n] + 2) / 4⌋
//! ```
//!
//! It uses whole-sample symmetric extension at the borders. Each level transforms vertically,
//! then horizontally (the Annex F order), and coefficients are stored in the Mallat layout
//! ([`Geometry`]).
//!
//! Samples are `i16` with wrapping arithmetic. Lifting is exactly invertible even when it wraps,
//! and centred 8-bit input or 8-bit residuals cannot wrap at 5 levels or fewer (DESIGN.md §4.2).
//!
//! [`Dwt53`] streams each level two rows at a time. It runs the vertical lifting in place, and
//! each finished row goes straight through the horizontal transform into its subbands, so a
//! level touches each sample about once while it is still in cache. The per-ISA row kernels must
//! match the scalar ones, and [`oracle`] (a literal transcription of the standard) checks the
//! whole transform. [`verify`] runs both checks, in `cargo test` and via `vdwt selftest` on the
//! target CPU.

#[cfg(target_arch = "x86_64")]
mod avx2;
pub mod d97;
mod geometry;
mod kernels;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
pub mod oracle;
mod scalar;
mod sparse;
mod verify;

use std::fmt;

pub use d97::Dwt97;
pub use geometry::{Geometry, MAX_LEVELS, Orient, Subband};
pub use sparse::BlockMap;
pub(crate) use sparse::{BAND_LOG2, OUT_LOG2};
pub use verify::{verify, verify_size};

use crate::pool::Pool;
use crate::{Isa, Plane};
use kernels::Kernels;

/// The wavelet of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wavelet {
    /// The reversible 5/3: exact integer reconstruction, for lossless coding.
    Reversible53,
    /// The 9/7 in integer lifting with fraction bits ([`d97`]): better compaction and much less
    /// rounding noise at every rate.
    Lifted97,
}

impl Wavelet {
    /// Fraction bits of its coefficients: steps scale by `2^frac_bits`.
    pub fn frac_bits(self) -> u32 {
        match self {
            Wavelet::Reversible53 => 0,
            Wavelet::Lifted97 => d97::FRAC_BITS,
        }
    }

    /// Energy of the 1-D synthesis basis of a low- or high-pass coefficient `level` levels deep,
    /// per unit of input sample.
    pub fn synthesis_gain_1d(self, level: usize, high: bool) -> f64 {
        match self {
            Wavelet::Reversible53 => crate::quant::synthesis_gain_53(level, high),
            Wavelet::Lifted97 => d97::synthesis_gain_1d(level, high),
        }
    }
}

/// A transform engine for either wavelet.
#[derive(Debug)]
pub enum Transform {
    W53(Dwt53),
    W97(Dwt97),
}

impl Transform {
    /// An engine for `wavelet` with `isa`'s kernels, or `None` if this CPU cannot run them.
    pub fn with_isa(wavelet: Wavelet, width: usize, height: usize, isa: Isa) -> Option<Self> {
        Some(match wavelet {
            Wavelet::Reversible53 => Transform::W53(Dwt53::with_isa(width, height, isa)?),
            Wavelet::Lifted97 => Transform::W97(Dwt97::with_isa(width, height, isa)?),
        })
    }

    /// Whether this engine spreads planes over the pool (the 9/7: callers run every component
    /// together with [`Dwt97::forward_many`] and [`Dwt97::inverse_many`]); otherwise callers run
    /// components in parallel, without it.
    pub fn uses_pool(&self) -> bool {
        matches!(self, Transform::W97(_))
    }

    /// Forward transform of `input` into `coeffs`; `input` may be destroyed.
    pub fn forward(
        &mut self,
        pool: Option<&Pool>,
        input: &mut Plane<i16>,
        levels: usize,
        coeffs: &mut Plane<i16>,
    ) {
        match self {
            Transform::W53(t) => t.forward(input, levels, coeffs),
            Transform::W97(t) => t.forward(pool, input, levels, coeffs),
        }
    }

    /// Inverse transform. `bands` marks the possibly nonzero 4×4 blocks of every band (see
    /// [`Dwt53::inverse_sparse`]); the result marks the possibly nonzero 8×8 output blocks.
    pub fn inverse_sparse(
        &mut self,
        pool: Option<&Pool>,
        coeffs: &Plane<i16>,
        levels: usize,
        out: &mut Plane<i16>,
        bands: &[BlockMap],
    ) -> &BlockMap {
        match self {
            Transform::W53(t) => t.inverse_sparse(coeffs, levels, out, bands),
            Transform::W97(t) => t.inverse_sparse(pool, coeffs, levels, out),
        }
    }

    /// The output map of the last [`Self::inverse_sparse`].
    pub fn output_map(&self) -> &BlockMap {
        match self {
            Transform::W53(t) => t.sparse.output_map(),
            Transform::W97(t) => t.output_map(),
        }
    }
}

/// Reusable 5/3 transform engine with scratch space for planes up to a maximum size.
pub struct Dwt53 {
    kernels: Kernels,
    /// Holds the level-1 low band: `⌈W/2⌉ × ⌈H/2⌉`.
    scratch: Plane<i16>,
    /// Line buffer for the horizontal inverse.
    line: Vec<i16>,
    /// Block maps for the sparse inverse.
    sparse: sparse::SparseScratch,
    max_width: usize,
    max_height: usize,
}

impl Dwt53 {
    /// An engine using the fastest kernels this CPU supports.
    pub fn new(max_width: usize, max_height: usize) -> Self {
        Self::with_isa(max_width, max_height, Isa::best())
            .expect("the best ISA is always supported")
    }

    /// An engine using a specific kernel family, or `None` if this CPU cannot run it.
    pub fn with_isa(max_width: usize, max_height: usize, isa: Isa) -> Option<Self> {
        let kernels = Kernels::for_isa(isa)?;
        Some(Self {
            kernels,
            scratch: Plane::new(max_width.div_ceil(2), max_height.div_ceil(2)),
            line: vec![0; max_width.div_ceil(2)],
            sparse: sparse::SparseScratch::default(),
            max_width,
            max_height,
        })
    }

    pub fn isa(&self) -> Isa {
        self.kernels.isa
    }

    /// Forward transform of `input` into `coeffs` (same size). `input` is used as scratch, so its
    /// contents are destroyed.
    pub fn forward(&mut self, input: &mut Plane<i16>, levels: usize, coeffs: &mut Plane<i16>) {
        let geo = self.check_size(input, coeffs, levels);
        if levels == 0 {
            coeffs.copy_from(input);
            return;
        }
        // Low bands ping-pong between `input` and `scratch`. The last level writes into `coeffs`.
        for l in 1..=levels {
            let (w, h) = geo.ll_size(l - 1);
            let (src, ll) = if l % 2 == 1 {
                (&mut *input, &mut self.scratch)
            } else {
                (&mut self.scratch, &mut *input)
            };
            let ll = if l == levels { None } else { Some(ll) };
            forward_level(&self.kernels, src, w, h, ll, coeffs);
        }
    }

    /// Inverse transform of `coeffs` into `out` (same size).
    pub fn inverse(&mut self, coeffs: &Plane<i16>, levels: usize, out: &mut Plane<i16>) {
        let geo = self.check_size(coeffs, out, levels);
        if levels == 0 {
            out.copy_from(coeffs);
            return;
        }
        // Level l writes to `out` when l is odd and to `scratch` when it is even, so level 1 ends
        // in `out`. Each level reads its low band from the previous level's output, or from
        // `coeffs` at the coarsest level.
        for l in (1..=levels).rev() {
            let (w, h) = geo.ll_size(l - 1);
            if l % 2 == 1 {
                let ll = if l == levels { coeffs } else { &self.scratch };
                inverse_level(&self.kernels, ll, coeffs, w, h, out, &mut self.line);
            } else {
                let ll = if l == levels { coeffs } else { &*out };
                inverse_level(
                    &self.kernels,
                    ll,
                    coeffs,
                    w,
                    h,
                    &mut self.scratch,
                    &mut self.line,
                );
            }
        }
    }

    /// The inverse transform, skipping regions whose coefficients are all zero (bit-identical
    /// to [`Self::inverse`]; see [`sparse`]). `bands` marks the 4×4-coefficient blocks of every
    /// band that may hold nonzero values, in [`Geometry::subbands`] order. Returns which 8×8
    /// blocks of `out` may be nonzero; all others are zero.
    pub fn inverse_sparse(
        &mut self,
        coeffs: &Plane<i16>,
        levels: usize,
        out: &mut Plane<i16>,
        bands: &[BlockMap],
    ) -> &BlockMap {
        let geo = self.check_size(coeffs, out, levels);
        sparse::inverse_sparse(
            &self.kernels,
            geo,
            coeffs,
            out,
            &mut self.scratch,
            &mut self.line,
            bands,
            &mut self.sparse,
        );
        self.sparse.output_map()
    }

    fn check_size(&self, a: &Plane<i16>, b: &Plane<i16>, levels: usize) -> Geometry {
        let (w, h) = (a.width(), a.height());
        assert_eq!(
            (b.width(), b.height()),
            (w, h),
            "input and output planes differ in size"
        );
        assert!(
            w <= self.max_width && h <= self.max_height,
            "{w}x{h} plane exceeds the engine's {}x{}",
            self.max_width,
            self.max_height
        );
        Geometry::new(w, h, levels)
    }
}

impl fmt::Debug for Dwt53 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dwt53")
            .field("isa", &self.kernels.isa)
            .field("max_width", &self.max_width)
            .field("max_height", &self.max_height)
            .finish()
    }
}

/// One analysis level of the `w × h` region at the origin of `src`, which it destroys.
///
/// It works through row pairs top to bottom. Each pair gets its vertical predict and update in
/// place. Then the finished low row (`2i`) and high row (`2i + 1`) go through the horizontal
/// transform into their subbands. The low band goes to `ll`, or into `coeffs` when `ll` is `None`
/// (the last level).
fn forward_level(
    k: &Kernels,
    src: &mut Plane<i16>,
    w: usize,
    h: usize,
    mut ll: Option<&mut Plane<i16>>,
    coeffs: &mut Plane<i16>,
) {
    let wl = w.div_ceil(2);
    let (ns, nd) = (h.div_ceil(2), h / 2);
    for i in 0..ns {
        if nd > 0 {
            if i < nd {
                // d[i]: the row below mirrors x[h] → x[h−2] at the bottom edge.
                let below = if 2 * i + 2 < h { 2 * i + 2 } else { 2 * i };
                let (d, s0, s1) = src.row_span_mut3(2 * i + 1, 2 * i, below, 0, w);
                (k.v_predict)(d, s0, s1);
            }
            // s[i]: d[−1] mirrors to d[0], and for odd h, d[nd] mirrors to d[nd−1].
            let above = if i == 0 { 1 } else { 2 * i - 1 };
            let below = if i < nd { 2 * i + 1 } else { 2 * i - 1 };
            let (s, d0, d1) = src.row_span_mut3(2 * i, above, below, 0, w);
            (k.v_update)(s, d0, d1);
        }
        // Vertical low row i → LL row i and HL row i.
        let row = src.row_span(2 * i, 0, w);
        match ll.as_deref_mut() {
            Some(ll) => (k.h_forward)(
                row,
                ll.row_span_mut(i, 0, wl),
                coeffs.row_span_mut(i, wl, w - wl),
            ),
            None => {
                let (lo, hi) = coeffs.row_span_mut(i, 0, w).split_at_mut(wl);
                (k.h_forward)(row, lo, hi);
            }
        }
        // Vertical high row i → LH row i and HH row i.
        if i < nd {
            let row = src.row_span(2 * i + 1, 0, w);
            let (lo, hi) = coeffs.row_span_mut(ns + i, 0, w).split_at_mut(wl);
            (k.h_forward)(row, lo, hi);
        }
    }
}

/// One synthesis level: rebuilds the `w × h` region at the origin of `dst` from the low band in
/// `ll` and the HL, LH and HH bands in `coeffs`.
///
/// Horizontal synthesis puts low row `i` at `dst` row `2i` and high row `i` at row `2i + 1`.
/// The vertical lifting then undoes row `2i`'s update and row `2i − 1`'s predict, one step
/// behind the rows it needs.
fn inverse_level(
    k: &Kernels,
    ll: &Plane<i16>,
    coeffs: &Plane<i16>,
    w: usize,
    h: usize,
    dst: &mut Plane<i16>,
    tmp: &mut [i16],
) {
    let wl = w.div_ceil(2);
    let (ns, nd) = (h.div_ceil(2), h / 2);
    for i in 0..ns {
        (k.h_inverse)(
            ll.row_span(i, 0, wl),
            coeffs.row_span(i, wl, w - wl),
            dst.row_span_mut(2 * i, 0, w),
            tmp,
        );
        if i < nd {
            let (lo, hi) = (
                coeffs.row_span(ns + i, 0, wl),
                coeffs.row_span(ns + i, wl, w - wl),
            );
            (k.h_inverse)(lo, hi, dst.row_span_mut(2 * i + 1, 0, w), tmp);
        }
        if nd > 0 {
            let above = if i == 0 { 1 } else { 2 * i - 1 };
            let below = if i < nd { 2 * i + 1 } else { 2 * i - 1 };
            let (s, d0, d1) = dst.row_span_mut3(2 * i, above, below, 0, w);
            (k.v_unupdate)(s, d0, d1);
            if i > 0 {
                let (d, s0, s1) = dst.row_span_mut3(2 * i - 1, 2 * i - 2, 2 * i, 0, w);
                (k.v_unpredict)(d, s0, s1);
            }
        }
    }
    if nd > 0 && nd == ns {
        // Even height: the last odd row's lower neighbour x[h] mirrors to x[h−2].
        let i = nd - 1;
        let (d, s0, s1) = dst.row_span_mut3(2 * i + 1, 2 * i, 2 * i, 0, w);
        (k.v_unpredict)(d, s0, s1);
    }
}
