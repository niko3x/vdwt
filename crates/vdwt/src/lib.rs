//! VDWT: a low-latency wavelet video codec for drone links.
//!
//! The design lives in `DESIGN.md` at the repository root. Implemented so far:
//!
//! - [`Encoder`] and [`Decoder`]: intra frames, P-frames and Repeat frames
//! - [`bitstream`]: stream syntax
//! - [`motion`]: the motion grid, global motion and the normative mesh warp (P-frames)
//! - [`dwt`]: the JPEG 2000 / HTJ2K reversible 5/3 wavelet (scalar, AVX2 and NEON kernels)
//! - [`quant`]: dead-zone scalar quantization
//! - [`ratectl`]: bit estimation and link-following CBR
//! - [`entropy`]: the VEC band coder on an adaptive range coder
//! - [`loopfilter`]: the in-loop Wiener filter
//! - [`plane`], [`frame`]: aligned sample planes and 8-bit 4:2:0 frames
//! - [`metrics`]: quality metrics (PSNR)
//! - [`profile`]: per-stage frame times
//! - [`pool`]: the worker threads of the parallel stages
//! - [`cpu`]: kernel-family (ISA) selection

pub mod aligned;
pub mod bframe;
pub mod bitstream;
mod codec;
pub mod cpu;
pub mod decoder;
pub mod dwt;
pub mod encoder;
pub mod entropy;
pub mod error;
pub mod frame;
pub mod loopfilter;
pub mod metrics;
pub mod motion;
pub mod plane;
pub mod pool;
pub mod postfilter;
pub mod profile;
pub mod quant;
pub mod ratectl;
pub mod rng;
pub mod rt;
mod spare;
pub mod training;

pub use cpu::Isa;
pub use decoder::Decoder;
pub use encoder::{BandStats, Encoder, EncoderConfig, FrameStats, MotionStats, RateStats};
pub use error::{Error, Result};
pub use frame::Frame;
pub use plane::Plane;
pub use ratectl::RateControl;
