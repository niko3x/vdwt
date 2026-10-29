//! Rate control (DESIGN.md §5.3): a per-band bit estimator and a link-following CBR controller.
//!
//! **Estimator.** After the transform, each band gets a histogram of coefficient magnitudes in
//! 1/8-octave bins, the qp resolution. From it, the nonzero count `n1` and the raw magnitude
//! bits `Σe` (each nonzero value's bit length) follow for any step without quantizing.
//! A per-band model turns them into coded bits; it is fitted offline and corrected online
//! (see `BitsModel`). Big bands are subsampled 2×2. The small LL bands are quantized and predicted
//! exactly instead, since a histogram cannot see what prediction does to them.
//!
//! **Controller.** It models the radio link as a transmit queue drained at the current link
//! rate. Each frame gets a target that drains the queue, and a hard cap that keeps queueing delay
//! within `max_delay_frames`. qp is the smallest one whose estimate meets the target. The encoder
//! enforces the cap exactly after coding by dropping bands (see `Encoder`).

use crate::Plane;
use crate::entropy::Rect;

/// Histogram bins per octave of magnitude: matches qp, where +8 doubles the step.
const BINS_PER_OCTAVE: u32 = 8;
/// Magnitudes up to 2^16 (LL prediction residuals reach 65534).
pub(crate) const HISTOGRAM_BINS: usize = 16 * BINS_PER_OCTAVE as usize;

/// Bin of a magnitude `v ≥ 1`: the octave, plus the three bits below the leading one.
#[inline]
fn bin_of(v: u32) -> usize {
    let octave = 31 - v.leading_zeros();
    let frac = if octave >= 3 {
        (v >> (octave - 3)) & 7
    } else {
        (v << (3 - octave)) & 7
    };
    (BINS_PER_OCTAVE * octave + frac) as usize
}

const BIN_TABLE_SIZE: usize = 4096;

/// [`bin_of`] for every magnitude below [`BIN_TABLE_SIZE`] (entry 0 unused).
fn bin_table() -> &'static [u8; BIN_TABLE_SIZE] {
    static TABLE: std::sync::OnceLock<[u8; BIN_TABLE_SIZE]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|m| if m == 0 { 0 } else { bin_of(m as u32) as u8 }))
}

/// Value range `[lo, hi)` covered by bin `k`. In the lowest four octaves each bin holds at most one
/// integer, `lo` itself, so those bins are exact points.
fn bin_bounds(k: usize) -> (f64, f64, bool) {
    let octave = (k / 8) as i32;
    let frac = (k % 8) as f64;
    let base = 2f64.powi(octave);
    (
        base * (1.0 + frac / 8.0),
        base * (1.0 + (frac + 1.0) / 8.0),
        octave <= 3,
    )
}

/// Magnitude histogram of one band's coefficients.
#[derive(Clone, Debug)]
pub(crate) struct MagnitudeHistogram {
    counts: [u32; HISTOGRAM_BINS],
    /// Each sample stands for this many coefficients.
    scale: u32,
}

impl Default for MagnitudeHistogram {
    fn default() -> Self {
        Self {
            counts: [0; HISTOGRAM_BINS],
            scale: 1,
        }
    }
}

impl MagnitudeHistogram {
    /// Histogram of the band at `r` in `coeffs`, sampled: every 8th row and column of the
    /// largest bands, fewer skipped in smaller ones (the rate model's correction absorbs the
    /// difference). Reading the coefficients back is latency-bound on small in-order cores: at
    /// 720p on the A55 this sampling takes a third of the time every other row and column took.
    pub fn sparse(coeffs: &Plane<i16>, r: Rect) -> Self {
        let step = match r.w * r.h {
            a if a >= 256 * 256 => 8,
            a if a >= 64 * 64 => 4,
            a if a >= 32 * 32 => 2,
            _ => 1,
        };
        Self::sampled(coeffs, r, step)
    }

    /// Histogram of every `step`th sample of every `step`th row of the band at `r`.
    ///
    /// Magnitudes below 4096 (nearly all of them) take their bin from a table. Four interleaved
    /// sub-histograms keep runs of equal bins from stalling on their own increments.
    pub fn sampled(coeffs: &Plane<i16>, r: Rect, step: usize) -> Self {
        let table = bin_table();
        let mut parts = [[0u32; HISTOGRAM_BINS]; 4];
        for y in (0..r.h).step_by(step) {
            let row = coeffs.row_span(r.y0 + y, r.x0, r.w);
            for (i, &c) in row.iter().step_by(step).enumerate() {
                let m = usize::from(c.unsigned_abs());
                if m != 0 {
                    let bin = if m < BIN_TABLE_SIZE {
                        usize::from(table[m])
                    } else {
                        bin_of(m as u32)
                    };
                    parts[i & 3][bin] += 1;
                }
            }
        }
        let mut h = Self {
            scale: (step * step) as u32,
            ..Self::default()
        };
        for (k, count) in h.counts.iter_mut().enumerate() {
            *count = parts.iter().map(|p| p[k]).sum();
        }
        h
    }

    /// Estimated squared error `Σ (c − r)²` of the band after quantizing with step `step`,
    /// rounding θ and reconstruction offset δ (`r = (q + δ)·step`, 0 for `q = 0`), in coefficient
    /// units. Values within a bin are taken at the bin's midpoint: exact in the low octaves, and
    /// close enough at the finer steps where a quality ceiling matters.
    pub fn distortion(&self, step: f64, rounding: f64, delta: f64) -> f64 {
        let mut total = 0.0;
        for (k, &count) in self.counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let (lo, hi, point) = bin_bounds(k);
            let m = if point { lo } else { (lo + hi) * 0.5 };
            let q = (m / step + rounding).floor().max(0.0);
            let r = if q <= 0.0 { 0.0 } else { (q + delta) * step };
            total += f64::from(count) * (m - r) * (m - r);
        }
        total * f64::from(self.scale)
    }

    /// Estimated nonzero count and raw magnitude bits `Σe` after quantizing with step `step` and
    /// rounding θ, i.e. `q = ⌊|c|/Δ + θ⌋`. Values within a bin are taken as uniform.
    pub fn estimate(&self, step: f64, rounding: f64) -> (f64, f64) {
        let threshold = (1.0 - rounding) * step;
        let (mut n1, mut bits) = (0.0, 0.0);
        for (k, &count) in self.counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let (lo, hi, point) = bin_bounds(k);
            let (share, centre) = if point {
                if lo < threshold {
                    continue;
                }
                (1.0, lo)
            } else {
                let share = ((hi - threshold) / (hi - lo)).clamp(0.0, 1.0);
                if share == 0.0 {
                    continue;
                }
                (share, (lo.max(threshold) + hi) / 2.0)
            };
            let q = (centre / step + rounding).floor().max(1.0) as u32;
            let e = 32 - q.leading_zeros();
            let n = f64::from(count) * share;
            n1 += n;
            bits += n * f64::from(e);
        }
        (n1 * f64::from(self.scale), bits * f64::from(self.scale))
    }
}

/// The quality ceiling the command-line encoder uses by default. The library's
/// [`RateControl::new`] leaves the ceiling off, so embedding a rate controller changes nothing
/// unless the caller asks for one.
pub const DEFAULT_QUALITY_CAP: f64 = 48.0;

/// Link-following constant-bitrate control settings.
#[derive(Clone, Debug, PartialEq)]
pub struct RateControl {
    /// Link rate in bits per second. Change it at any frame with
    /// [`Encoder::set_bitrate`](crate::Encoder::set_bitrate).
    pub bitrate: u64,
    /// Transmit-queue limit, in frame durations. While the link delivers `bitrate`, every frame
    /// finishes sending within this delay of being queued.
    pub max_delay_frames: f64,
    pub min_qp: u8,
    pub max_qp: u8,
    /// Quality ceiling in dB (estimated PSNR of the luma domain): a frame is not coded finer than
    /// the qp where its estimated distortion meets this, and the link capacity left over is banked
    /// for later, harder frames. 0 disables it (the library default); the command-line encoder
    /// sets [`DEFAULT_QUALITY_CAP`]. Static scenes are where it pays: without it, a
    /// constant-bitrate encoder spends the whole link refining an unchanging picture, and
    /// quality pumps from frame to frame.
    pub quality_cap: f64,
}

impl RateControl {
    pub fn new(bitrate: u64) -> Self {
        Self {
            bitrate,
            max_delay_frames: 1.5,
            min_qp: 0,
            max_qp: 127,
            quality_cap: 0.0,
        }
    }
}

/// Bits a band costs: its raw magnitude bits `Σe`, plus `β` bits per nonzero value for the
/// significance tree, patterns and exponents, plus a fixed per-segment overhead.
///
/// `β` grows as nonzeros get sparser, because locating each one costs more:
/// `β = a + b·log2(1/p)`, where `p` is the band's nonzero density. `(a, b)` per context class
/// were least-squares fitted: for intra frames on about 5,900 coded bands from three clips
/// (natural photos, terrain textures, synthetic graphics), for P-frames on about 12,000 residual
/// bands from synthetic drone clips. Given exact statistics they predict frame sizes within
/// about 8% (intra) and 6% (P) rms. What remains is content-dependent bias of around ±10%, stable across qp, which one
/// learned factor removes (`correction`).
#[derive(Clone, Debug)]
pub(crate) struct BitsModel {
    /// Learned ratio of actual to modelled frame size, for intra frames and for P-frames.
    correction: [f64; 2],
}

impl BitsModel {
    /// `(a, b)` per class, in [`codec`](crate::codec) class order: luma LL, L1 HL/LH, L1 HH,
    /// L2 HL/LH, L2 HH, L3+ HL/LH, L3+ HH, then the same seven for chroma; then all fourteen
    /// again for P-frame residuals.
    const BETA: [(f64, f64); 28] = [
        (2.87, 0.35),
        (2.61, 0.48),
        (2.62, 0.54),
        (2.64, 0.51),
        (2.62, 0.55),
        (2.97, 0.51),
        (2.81, 0.56),
        (2.75, 0.31),
        (2.30, 0.63),
        (1.98, 0.79),
        (2.66, 0.58),
        (2.54, 0.63),
        (3.06, 0.57),
        (2.82, 0.60),
        // P-frame residuals.
        (1.55, 1.29),
        (2.69, 0.58),
        (2.75, 0.60),
        (2.34, 0.75),
        (2.40, 0.74),
        (2.61, 0.77),
        (2.61, 0.76),
        (1.80, 1.11),
        (2.03, 0.80),
        (1.79, 0.85),
        (2.36, 0.82),
        (2.41, 0.80),
        (2.85, 0.69),
        (2.44, 0.78),
    ];
    /// Segment length prefix and range-coder flush, on average.
    const SEGMENT_BITS: f64 = 20.0;
    /// Weight of each new frame's observation of the correction.
    const LEARNING_RATE: f64 = 0.3;

    pub fn new() -> Self {
        Self {
            correction: [1.0; 2],
        }
    }

    /// Modelled bits of a band of `coefficients` values with the given statistics (uncorrected).
    pub fn band_bits(
        &self,
        class: usize,
        coefficients: usize,
        nonzero: f64,
        magnitude_bits: f64,
    ) -> f64 {
        if nonzero < 0.5 {
            return 0.0;
        }
        let (a, b) = Self::BETA[class];
        let density = (nonzero / coefficients.max(1) as f64).clamp(1e-6, 1.0);
        magnitude_bits + (a - b * density.log2()) * nonzero + Self::SEGMENT_BITS
    }

    pub fn correction(&self, inter: bool) -> f64 {
        self.correction[usize::from(inter)]
    }

    /// Learns from a coded frame: its actual size against the uncorrected model at the same qp.
    pub fn learn(&mut self, inter: bool, actual_bits: f64, modelled_bits: f64) {
        if modelled_bits > 0.0 && actual_bits > 0.0 {
            let observed = (actual_bits / modelled_bits).clamp(0.5, 2.0);
            let c = &mut self.correction[usize::from(inter)];
            *c += Self::LEARNING_RATE * (observed - *c);
        }
    }
}

/// One frame's rate decision.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FramePlan {
    pub qp: u8,
    /// Bits the frame should use: what drains the queue, kept below the cap with headroom.
    pub target: f64,
    /// Bits the frame must not exceed, so that queueing delay stays within the limit.
    pub cap: f64,
    /// Estimated bits at `qp`.
    pub estimate: f64,
}

/// The link-following controller: a transmit queue drained at the link rate.
#[derive(Clone, Debug)]
pub(crate) struct RateController {
    cfg: RateControl,
    frame_rate: f64,
    /// Bits still queued after the previous frame interval.
    queue: f64,
    pub model: BitsModel,
}

impl RateController {
    /// The target stays this far below the cap, to absorb estimation error.
    const HEADROOM: f64 = 0.15;
    /// An intra frame that P-frames will predict from aims at this many frame budgets (less the
    /// queue), still below the cap. Its quality carries into every P-frame after it. 1.3 is
    /// where the cap binds at the default 1.5-frame delay limit; with a looser limit, 2 starved
    /// the P-frames after it on real footage (§10.1).
    const KEYFRAME_SHARE: f64 = 1.3;
    /// Keyframes may reach this many dB above the quality ceiling before it applies.
    const KEYFRAME_BONUS: f64 = 2.0;

    pub fn new(cfg: RateControl, frame_rate: f64) -> Self {
        Self {
            cfg,
            frame_rate,
            queue: 0.0,
            model: BitsModel::new(),
        }
    }

    /// The quality ceiling in force (see [`RateControl::quality_cap`]).
    pub fn quality_cap(&self) -> f64 {
        self.cfg.quality_cap
    }

    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.cfg.bitrate = bitrate;
    }

    pub fn queue_bits(&self) -> f64 {
        self.queue
    }

    /// Bits the link drains per frame interval.
    pub fn budget(&self) -> f64 {
        self.cfg.bitrate as f64 / self.frame_rate
    }

    /// Picks the smallest qp whose estimated size meets the target. `frame_bits(qp)` is the
    /// uncorrected model; it must fall as qp rises. `keyframe` marks an intra frame that P-frames
    /// will predict from.
    pub fn plan(&self, inter: bool, keyframe: bool, frame_bits: impl Fn(u8) -> f64) -> FramePlan {
        self.plan_quality(inter, keyframe, frame_bits, |_| f64::INFINITY, 1.0)
    }

    /// [`Self::plan`] with the quality ceiling: `frame_sse(qp)` is the frame's estimated squared
    /// error in the pixel domain, over `pixels` luma samples. When the rate target would code the
    /// frame finer than `quality_cap` dB, the qp is raised until it does not. The bits left over
    /// stay in the queue (it may go negative) for later frames.
    pub fn plan_quality(
        &self,
        inter: bool,
        keyframe: bool,
        frame_bits: impl Fn(u8) -> f64,
        frame_sse: impl Fn(u8) -> f64,
        pixels: f64,
    ) -> FramePlan {
        self.plan_quality_capped(
            inter,
            keyframe,
            frame_bits,
            frame_sse,
            pixels,
            self.cfg.quality_cap,
        )
    }

    /// [`Self::plan_quality`] with the ceiling given by the caller (0: none). The encoder uses
    /// this to hold the ceiling back while the picture is still recovering from a coarse
    /// keyframe: refining the reference is worth more than the ceiling saves.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_quality_capped(
        &self,
        inter: bool,
        keyframe: bool,
        frame_bits: impl Fn(u8) -> f64,
        frame_sse: impl Fn(u8) -> f64,
        pixels: f64,
        quality_cap: f64,
    ) -> FramePlan {
        let correction = self.model.correction(inter);
        let bits = |qp| correction * frame_bits(qp);
        let plan = self.plan_rate(keyframe, f64::NAN, &bits);
        self.apply_quality_cap(plan, keyframe, &bits, frame_sse, pixels, quality_cap)
    }

    /// The rate-only half of [`Self::plan_quality`], for a frame aiming at `target` bits
    /// (`f64::NAN`: the queue-following target).
    fn plan_rate(
        &self,
        keyframe: bool,
        target_bits: f64,
        frame_bits: &impl Fn(u8) -> f64,
    ) -> FramePlan {
        let budget = self.budget();
        let cap = (self.cfg.max_delay_frames * budget - self.queue).max(0.0);
        let target = if target_bits.is_nan() {
            let share = if keyframe { Self::KEYFRAME_SHARE } else { 1.0 };
            (share * budget - self.queue)
                .min(cap / (1.0 + Self::HEADROOM))
                .max(0.0)
        } else {
            target_bits.min(cap / (1.0 + Self::HEADROOM)).max(0.0)
        };
        let (mut lo, mut hi) = (self.cfg.min_qp, self.cfg.max_qp.max(self.cfg.min_qp));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if frame_bits(mid) <= target {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        // qp steps are about 9% in rate. Taking the one just under the target would leave about
        // half a step unused on average, so take the nearer neighbour when it stays safely below
        // the cap; the queue absorbs the difference and the next target corrects it.
        let mut plan = FramePlan {
            qp: lo,
            target,
            cap,
            estimate: frame_bits(lo),
        };
        if lo > self.cfg.min_qp {
            let above = frame_bits(lo - 1);
            if above - target < target - plan.estimate && above <= cap / (1.0 + Self::HEADROOM) {
                plan.qp = lo - 1;
                plan.estimate = above;
            }
        }
        plan
    }

    /// Raises `plan.qp` until the estimated PSNR no longer exceeds the quality ceiling (keyframes
    /// get a small bonus, since their quality carries into every frame that predicts from them).
    fn apply_quality_cap(
        &self,
        mut plan: FramePlan,
        keyframe: bool,
        frame_bits: &impl Fn(u8) -> f64,
        frame_sse: impl Fn(u8) -> f64,
        pixels: f64,
        cap_psnr: f64,
    ) -> FramePlan {
        if cap_psnr > 0.0 && pixels > 0.0 {
            let cap = cap_psnr + if keyframe { Self::KEYFRAME_BONUS } else { 0.0 };
            let psnr = |qp: u8| {
                let sse = frame_sse(qp);
                if sse <= 0.0 {
                    f64::INFINITY
                } else {
                    10.0 * (255.0 * 255.0 / (sse / pixels)).log10()
                }
            };
            if psnr(plan.qp) > cap {
                let (mut lo, mut hi) = (plan.qp, self.cfg.max_qp.max(self.cfg.min_qp));
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if psnr(mid) <= cap {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                if lo > plan.qp {
                    plan.qp = lo;
                    plan.estimate = frame_bits(lo);
                }
            }
        }
        plan
    }

    /// The qp for a frame aiming at `target` bits instead of the queue-following target (a
    /// B-frame pair's anchor), still under the cap, with the ceiling given by the caller (0:
    /// none).
    #[allow(clippy::too_many_arguments)]
    pub fn plan_target_quality_capped(
        &self,
        inter: bool,
        target: f64,
        frame_bits: impl Fn(u8) -> f64,
        frame_sse: impl Fn(u8) -> f64,
        pixels: f64,
        keyframe: bool,
        quality_cap: f64,
    ) -> FramePlan {
        let correction = self.model.correction(inter);
        let bits = |qp| correction * frame_bits(qp);
        let plan = self.plan_rate(keyframe, target, &bits);
        self.apply_quality_cap(plan, keyframe, &bits, frame_sse, pixels, quality_cap)
    }

    /// Queues a sent frame of `bits` and drains one frame interval. Unused capacity becomes a
    /// credit, bounded by one delay limit's worth of bits, so an easy stretch can pay for the
    /// hard frame that follows it without ever breaking the delay bound (the cap in
    /// [`Self::plan_rate`] accounts for a negative queue).
    pub fn finish_frame(&mut self, bits: f64) {
        let credit = self.cfg.max_delay_frames * self.budget();
        self.queue = (self.queue + bits - self.budget()).max(-credit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Quantizer;
    use crate::rng::SplitMix64;

    #[test]
    fn bins_are_monotonic_and_cover_every_magnitude() {
        let mut last = 0;
        for v in 1..=65535u32 {
            let k = bin_of(v);
            assert!(k >= last && k < HISTOGRAM_BINS);
            let (lo, hi, _) = bin_bounds(k);
            assert!(
                lo <= f64::from(v) && f64::from(v) < hi,
                "v {v} not in bin {k} [{lo}, {hi})"
            );
            last = k;
        }
    }

    #[test]
    fn estimates_track_real_quantization() {
        // Laplacian-like coefficients, as wavelet detail bands have.
        let mut rng = SplitMix64::new(3);
        let (w, h) = (256, 256);
        let mut plane = Plane::<i16>::new(w, h);
        for row in plane.rows_mut() {
            for v in row {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let mag = (-30.0 * (1.0 - u).ln()).min(32767.0);
                *v = if rng.below(2) == 0 {
                    mag as i16
                } else {
                    -(mag as i16)
                };
            }
        }
        let r = Rect { x0: 0, y0: 0, w, h };
        let hist = MagnitudeHistogram::sampled(&plane, r, 1);
        for step in [1.0, 3.0, 10.0, 40.0, 150.0] {
            let quant = Quantizer::new((step * 4096.0) as u64, 0.25);
            let (mut n1, mut bits) = (0.0, 0.0);
            for row in plane.rows() {
                for &c in row {
                    let q = quant.quantize(c).unsigned_abs();
                    if q != 0 {
                        n1 += 1.0;
                        bits += f64::from(32 - q.leading_zeros());
                    }
                }
            }
            let (en1, ebits) = hist.estimate(step, 0.25);
            assert!(
                (en1 / n1 - 1.0).abs() < 0.05,
                "step {step}: nonzero {en1:.0} vs {n1}"
            );
            assert!(
                (ebits / bits - 1.0).abs() < 0.05,
                "step {step}: bits {ebits:.0} vs {bits}"
            );
        }
    }
}
