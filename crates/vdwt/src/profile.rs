//! Per-stage wall-clock times of a frame, for profiling on the target board without a profiler.

use std::ops::AddAssign;
use std::time::{Duration, Instant};

/// A coding stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Encoder: global motion estimation (with the scene-cut check).
    GlobalMotion,
    /// Encoder: node search and intra-node decisions.
    Mesh,
    /// Motion compensation, and in the encoder the filter choice and the residual.
    Prediction,
    /// Encoder: forward wavelet transform.
    Transform,
    /// Encoder: rate-control planning, quantization and significance pyramids.
    Quantize,
    /// Range coding of the motion section and the bands (decoding in the decoder).
    Entropy,
    /// Dequantization, inverse transform and adding the residual to the prediction.
    Reconstruct,
}

impl Stage {
    pub const ALL: [Stage; 7] = [
        Stage::GlobalMotion,
        Stage::Mesh,
        Stage::Prediction,
        Stage::Transform,
        Stage::Quantize,
        Stage::Entropy,
        Stage::Reconstruct,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::GlobalMotion => "global motion",
            Stage::Mesh => "mesh search",
            Stage::Prediction => "prediction",
            Stage::Transform => "transform",
            Stage::Quantize => "quantize",
            Stage::Entropy => "entropy",
            Stage::Reconstruct => "reconstruct",
        }
    }
}

/// Time spent per [`Stage`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageTimes([Duration; Stage::ALL.len()]);

impl StageTimes {
    pub fn get(&self, stage: Stage) -> Duration {
        self.0[stage as usize]
    }

    pub fn total(&self) -> Duration {
        self.0.iter().sum()
    }
}

impl AddAssign for StageTimes {
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a += b;
        }
    }
}

/// Charges the time since the last lap to a stage.
pub(crate) struct Stopwatch(Instant);

impl Stopwatch {
    pub fn start() -> Self {
        Self(Instant::now())
    }

    pub fn lap(&mut self, times: &mut StageTimes, stage: Stage) {
        let now = Instant::now();
        times.0[stage as usize] += now - self.0;
        self.0 = now;
    }
}
