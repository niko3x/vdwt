//! Entropy coding: the VEC band coder over an adaptive multi-symbol range coder (DESIGN.md §5.2).

mod band;
pub(crate) use band::FLAT_GUIDE_CLASS;
mod cdf;
mod layout;
mod range;
pub(crate) mod trained;

pub(crate) use band::{
    CDFS, CoeffBand, Contexts, Guide, Parent, Recorder, Rect, SigPyramid, SymbolCoder, TracedSymbol, Tracer, TrackedRows,
    code_band,
};
pub use cdf::{Cdf, MAX_SYMBOLS, MIN_PROB, PROB_BITS, PROB_ONE};
pub(crate) use layout::{
    decode as decode_layout, decode_from as decode_layout_from, decode_presence,
    encode as encode_layout, encode_into as encode_layout_into, encode_presence,
};
pub use range::{RangeDecoder, RangeEncoder};
