//! The encoder: intra frames and P-frames.
//!
//! A frame goes through these steps:
//! 1. P-frames only: motion estimation against the previous reconstruction ([`crate::motion`]),
//!    motion compensation, and the residual. A frame that motion compensation predicts badly
//!    (a scene cut) is coded intra instead.
//! 2. the wavelet transform of the picture (intra) or of the residual (P);
//! 3. with rate control, a qp choice from per-band magnitude histograms ([`crate::ratectl`]);
//! 4. quantization;
//! 5. coding of every band into its own segment;
//! 6. with rate control, dropping the finest bands until the frame fits the delay cap. A P-frame
//!    can shed every band and still move the picture with its motion alone; when not even that
//!    fits, a Repeat frame goes out, or nothing;
//! 7. reconstruction, exactly as the decoder will see it.

use crate::bframe::{self, BiMode};
use crate::bitstream::{
    FrameHeader, FrameHeaderState, FrameType, MAX_DIMENSION, SequenceHeader, put_varint,
};
use crate::codec::{self, BandSlot, LevelBands};
use crate::dwt::d97::{ForwardJob, InverseJob};
use crate::dwt::{BAND_LOG2, BlockMap, Dwt97, MAX_LEVELS, Orient, Transform, Wavelet};
use crate::entropy::{
    Contexts, Guide, Parent, RangeEncoder, Recorder, Rect, SigPyramid, code_band, encode_layout,
};
use crate::error::{Error, Result};
use crate::loopfilter::{self, Coefs};
use crate::motion::gme::{Affine, GlobalEstimator};
use crate::motion::mesh::MeshParams;
use crate::motion::partition;
use crate::motion::tree::{self, TreeContexts};
use crate::motion::warp::{sampled, warp_planes};
use crate::motion::{Filter, GlobalMotion, Grid, MESH_LOG2, MotionCoder, MotionField, Prediction};
use crate::plane::{
    add_residual_row, center_row, center_u8, residual_row, uncenter_row, uncenter_to_u8,
};
use crate::pool::Pool;
use crate::postfilter::{self, PostFilterMode};
use crate::profile::{Stage, StageTimes, Stopwatch};
use crate::quant::{Dequantizer, Quantizer, mse_weights};
use crate::ratectl::{MagnitudeHistogram, RateControl, RateController};
use crate::training::ContextStats;
use crate::{Frame, Isa, Plane};

/// Default θ: quantizer rounding offset (see [`EncoderConfig::rounding`]).
pub const DEFAULT_ROUNDING: f64 = 0.25;
/// Default δ: reconstruction offset (see [`EncoderConfig::recon_offset`]).
pub const DEFAULT_RECON_OFFSET: f64 = 0.2;
/// Default θ for P-frames (see [`EncoderConfig::inter_rounding`]).
pub const DEFAULT_INTER_ROUNDING: f64 = 0.12;
/// Default δ for P-frames (see [`EncoderConfig::inter_recon_offset`]).
pub const DEFAULT_INTER_RECON_OFFSET: f64 = 0.1;
/// Default chroma step multiplier (see [`EncoderConfig::chroma_step_scale`]): 0.4 is the
/// optimum on YUV-PSNR 6:1:1, 10–11% fewer bits on every tier than 0.95 (2026-10-28; 0.95 and
/// 0.7 were tuned on the VMAF hybrid and on PSNR-Y, which do not see what chroma costs).
pub const DEFAULT_CHROMA_STEP_SCALE: f64 = 0.4;
/// Above this share of badly predicted blocks the global model's scene-cut verdict needs no
/// confirmation by the prediction.
const CERTAIN_CUT: f64 = 0.75;
/// Default scene-cut threshold (see [`EncoderConfig::scene_cut`]).
pub const DEFAULT_SCENE_CUT: f64 = 0.5;
/// Default qp offset of intra frames (see [`EncoderConfig::intra_qp_offset`]).
pub const DEFAULT_INTRA_QP_OFFSET: i8 = -8;
/// Default isolated-value threshold (see [`EncoderConfig::isolated_threshold`]).
pub const DEFAULT_ISOLATED_THRESHOLD: f64 = 1.5;
/// Root-row stripes of the motion search: each starts its wavefront at once instead of waiting
/// for the row above, which fills a four-core pool. The first row of each stripe loses the
/// upper-neighbour prediction; on 720p that costs under 0.02 dB and 0.3% of the bits, and buys
/// 7–17% of the encode time.
const SEARCH_STRIPES: usize = 4;

/// Default encoder effort, the medium tier (see [`EncoderConfig::effort`]).
pub const DEFAULT_EFFORT: u8 = 1;
/// Most threads an encoder or decoder runs on.
pub const MAX_THREADS: usize = 64;

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
    pub luma_levels: usize,
    pub chroma_levels: usize,
    /// Quadtree root of the band coder as log2: 2..=7 (4×4 to 128×128).
    pub block_log2: u32,
    /// Quality index; the base step is `2^(qp/8)`. Lower is better.
    pub qp: u8,
    /// Added to `qp` for intra frames when P-frames follow, without rate control. Negative
    /// values make the frames everything else predicts from sharper: −4 (a step ratio of √2, like x264's default
    /// I/P ratio) gains 0.1 dB where the scene persists and costs nothing where it does not.
    pub intra_qp_offset: i8,
    /// Offset added to every P-frame's qp (and therefore its B-frames' anchor qp).
    pub inter_qp_offset: i8,
    /// Offset for an intra frame at the end of the stream, where nothing predicts from it.
    pub final_intra_qp_offset: i8,
    /// Total frames when known; enables [`Self::final_intra_qp_offset`].
    pub total_frames: Option<u16>,
    /// Exact reconstruction. Overrides `qp`, the step weights, `recon_offset` and `wavelet`
    /// (lossless coding uses the 5/3).
    pub lossless: bool,
    /// The wavelet of lossy coding.
    pub wavelet: Wavelet,
    /// θ in `q = ⌊|c|/Δ + θ⌋`, in `0..1`. Smaller values widen the dead zone around zero.
    pub rounding: f64,
    /// δ: magnitudes reconstruct to `(|q| + δ)·Δ`, in `0..1`.
    pub recon_offset: f64,
    /// θ for P-frames. Residuals are mostly small, noise-like values, so a wider dead zone pays
    /// off: about +0.5 dB BD-PSNR together with `inter_recon_offset` on the synthetic clips. On
    /// real footage 0.12 beats 0.04 by 1.0% of the bits at 720p and 1.7% at 1080p (§10.1).
    pub inter_rounding: f64,
    /// δ for P-frames.
    pub inter_recon_offset: f64,
    /// Multiplier on every chroma step. Above 1 moves bits from chroma to luma.
    pub chroma_step_scale: f64,
    /// Condition contexts on the parent band (one level coarser).
    pub parent_contexts: bool,
    /// Code intra LL bands as residuals of a median predictor.
    pub ll_prediction: bool,
    /// Transform kernels. Every ISA gives the same output.
    pub isa: Isa,
    /// Link-following constant bitrate. `None` codes every frame at `qp`.
    pub rate_control: Option<RateControl>,
    /// Code P-frames. Without them every frame is intra.
    pub p_frames: bool,
    /// Frames from one intra frame to the next. 0 sends intra frames only at the start and at
    /// scene cuts.
    pub intra_period: u32,
    /// Motion grid spacing in luma pixels, as log2: 3..=6 (8 to 64).
    pub mesh_log2: u32,
    /// Interpolation filter for motion compensation. `None` picks one per P-frame: the one that
    /// predicts the frame better. Rigid motion favours the sharper cubic filter; parallax and
    /// other motion the field cannot follow exactly favour bilinear's smoother prediction.
    pub filter: Option<Filter>,
    /// Warp or overlapped prediction. `None` picks per P-frame, with the filter.
    pub prediction: Option<Prediction>,
    /// Refine node vectors on top of the global model. Without it P-frames use global motion
    /// alone.
    pub mesh_refinement: bool,
    /// Multiplier on the rate-distortion trade-off of motion decisions. Higher values send fewer
    /// node vectors.
    pub motion_lambda: f64,
    /// Causal frame-level activity adaptation: qp moves by `-frame_aq·log2(activity/average)`,
    /// clamped to ±`frame_aq_clamp`. 0 disables it (fixed qp). Encoder-only: the qp is already
    /// in every frame header.
    pub frame_aq: f64,
    /// Largest qp change the activity adaptation may make.
    pub frame_aq_clamp: u8,
    /// Fit the in-loop Wiener filter to a sharpened source instead of the source, to trade
    /// squared error for detail. 0 keeps the MSE fit. Encoder-side only: the decoder still just
    /// applies the coded coefficients. It costs PSNR (gen_b up to 13%), so it is off by default
    /// and part of the CLI's `--tune perceptual`.
    pub loop_sharpen: f64,
    /// Multiplier on the finest luma level's step weights. 1.0 leaves the MSE weights unchanged;
    /// larger values coarsen level 1 (fewer high-frequency bits). [`Self::finest_tilt_auto`]
    /// picks it from the base qp (the CLI's `--tune perceptual`).
    pub finest_tilt: f64,
    /// Pick [`Self::finest_tilt`] from the base qp at fixed qp: high rates can afford a coarser
    /// finest level, low rates need it for the last detail. Encoder-only; the weights are in
    /// the sequence header, so the decoder is unchanged. Off under rate control and lossless.
    pub finest_tilt_auto: bool,
    /// Extra multiplier on the finest level's HH (diagonal) step. Diagonal detail is the least
    /// visible orientation; the weights travel in the sequence header, so this is free-form.
    pub hh_tilt: f64,
    /// Multiplier on the second-finest level's three step weights (level 2).
    pub mid_tilt: f64,
    /// qp offset of a non-reference B-frame from its anchor (default 9).
    pub b_qp_offset: u8,
    /// qp offset of a B-frame on a static camera (default 6).
    pub b_static_qp_offset: u8,
    /// Zero a ±1 detail value that is alone in its 4×4 block even when neighbouring blocks
    /// are nonzero (a less conservative `isolated_threshold`). Encoder-only.
    pub sparse_zero: bool,
    /// Try both the spatial and the previous-frame motion predictor for each P-frame and keep
    /// the shorter motion section (fixed-qp feature; VERSION 11).
    pub temporal_motion: bool,
    /// Scene-cut threshold: when the estimated global motion predicts more than this fraction of
    /// the frame from the previous source frame worse than intra coding would, the frame is
    /// coded intra. 0 disables the check.
    pub scene_cut: f64,
    /// Zero a ±1 detail value when it is alone in its 4×4 block with the 8 blocks around it
    /// empty and its coefficient is below this many quantizer steps (0 disables). Such a value
    /// costs more bits than the distortion it removes is worth.
    pub isolated_threshold: f64,
    /// Threads the encoder runs on, the caller's included. The stream does not depend on it.
    pub threads: usize,
    /// Motion search effort, 0..=3 (the CLI's tiers light, medium, heavy and ultraheavy): passes
    /// of overlapped-prediction-aware refinement of the motion offsets (each costs about as much
    /// as the rest of the search and saves bits on real footage, §10.1), four at 3; from 2 on
    /// every quadrant also searches as widely as its region.
    pub effort: u8,
    /// Realtime mode, for small CPUs (§10.2): effort 0 with a cheaper search (an unrefined
    /// global model from fewer RANSAC rounds, exact node errors on every other row, fewer
    /// searches and Lucas–Kanade steps) and overlapped prediction unless one is forced. About
    /// 18% faster than effort 0 for under 1% more bits.
    pub realtime: bool,
    /// Fit and send the in-loop Wiener filter where it pays (§5.4).
    pub loop_filter: bool,
    /// Derive inter bands' texture guide from the prediction's transform. Off (the realtime mode)
    /// uses a flat guide and skips the prediction DWT on both sides; a small bitrate cost.
    pub guide: bool,
    /// Display post-filter strength to store in frame headers: off, fixed, or one per scene.
    /// Display-only; it never enters prediction or references.
    pub postfilter: PostFilterMode,
    /// Write a short header when qp/recon/post-filter repeat (fixed-qp streams). The decoder
    /// always understands both forms; off by default for tests and tools.
    pub short_headers: bool,
}

impl EncoderConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            fps_num: 60,
            fps_den: 1,
            luma_levels: 5,
            chroma_levels: 4,
            block_log2: 6,
            qp: 32,
            intra_qp_offset: DEFAULT_INTRA_QP_OFFSET,
            inter_qp_offset: 0,
            final_intra_qp_offset: 0,
            total_frames: None,
            lossless: false,
            wavelet: Wavelet::Lifted97,
            rounding: DEFAULT_ROUNDING,
            recon_offset: DEFAULT_RECON_OFFSET,
            inter_rounding: DEFAULT_INTER_ROUNDING,
            inter_recon_offset: DEFAULT_INTER_RECON_OFFSET,
            chroma_step_scale: DEFAULT_CHROMA_STEP_SCALE,
            parent_contexts: true,
            ll_prediction: true,
            isa: Isa::best(),
            rate_control: None,
            p_frames: true,
            intra_period: 0,
            mesh_log2: default_mesh_log2(width, height),
            loop_filter: true,
            filter: None,
            prediction: None,
            mesh_refinement: true,
            motion_lambda: 1.0,
            frame_aq: 2.0,
            frame_aq_clamp: 6,
            loop_sharpen: 0.0,
            finest_tilt: 1.0,
            finest_tilt_auto: false,
            hh_tilt: 1.0,
            mid_tilt: 1.0,
            b_qp_offset: B_QP_OFFSET,
            b_static_qp_offset: B_STATIC_QP_OFFSET,
            sparse_zero: false,
            temporal_motion: false,
            scene_cut: DEFAULT_SCENE_CUT,
            isolated_threshold: DEFAULT_ISOLATED_THRESHOLD,
            threads: 1,
            effort: DEFAULT_EFFORT,
            realtime: false,
            guide: true,
            postfilter: PostFilterMode::Off,
            short_headers: false,
        }
    }

    fn validate(&self) -> Result<()> {
        let fail = |what: String| Err(Error::InvalidConfig(what));
        if !(1..=MAX_DIMENSION).contains(&self.width) || !(1..=MAX_DIMENSION).contains(&self.height)
        {
            return fail(format!(
                "frame size {}x{} (1..={MAX_DIMENSION})",
                self.width, self.height
            ));
        }
        if self.luma_levels > MAX_LEVELS || self.chroma_levels > MAX_LEVELS {
            return fail(format!("at most {MAX_LEVELS} decomposition levels"));
        }
        if !(2..=7).contains(&self.block_log2) {
            return fail(format!("block_log2 {} (2..=7)", self.block_log2));
        }
        if !MESH_LOG2.contains(&self.mesh_log2) {
            return fail(format!("mesh_log2 {} (3..=6)", self.mesh_log2));
        }
        if let PostFilterMode::Fixed(strength) = self.postfilter
            && strength > 40
        {
            return fail("post-filter strength must be in 0..=40".into());
        }
        if ![
            self.rounding,
            self.recon_offset,
            self.inter_rounding,
            self.inter_recon_offset,
        ]
        .iter()
        .all(|v| (0.0..1.0).contains(v))
        {
            return fail("rounding and reconstruction offsets must be in 0..1".into());
        }
        if !(0.25..=8.0).contains(&self.chroma_step_scale) {
            return fail(format!(
                "chroma_step_scale {} (0.25..=8)",
                self.chroma_step_scale
            ));
        }
        if !(0.0..=4.0).contains(&self.isolated_threshold) {
            return fail(format!(
                "isolated_threshold {} (0..=4)",
                self.isolated_threshold
            ));
        }
        if !(0.0..=100.0).contains(&self.motion_lambda) || !(0.0..=1.0).contains(&self.scene_cut) {
            return fail("motion_lambda must be in 0..=100 and scene_cut in 0..=1".into());
        }
        if !(0.0..=4.0).contains(&self.frame_aq) || self.frame_aq_clamp > 16 {
            return fail("frame_aq must be in 0..=4 and frame_aq_clamp in 0..=16".into());
        }
        if !(0.0..=2.0).contains(&self.loop_sharpen) {
            return fail("loop_sharpen must be in 0..=2".into());
        }
        if !(0.5..=2.0).contains(&self.finest_tilt) {
            return fail("finest_tilt must be in 0.5..=2".into());
        }
        if !(0.5..=2.0).contains(&self.hh_tilt) || !(0.5..=2.0).contains(&self.mid_tilt) {
            return fail("hh_tilt and mid_tilt must be in 0.5..=2".into());
        }
        if self.fps_num == 0 || self.fps_den == 0 {
            return fail("frame rate".into());
        }
        if self.effort > 3 {
            return fail(format!("effort {} (0..=3)", self.effort));
        }
        if !(1..=MAX_THREADS).contains(&self.threads) {
            return fail(format!("threads {} (1..={MAX_THREADS})", self.threads));
        }
        if let Some(rc) = &self.rate_control {
            if self.lossless {
                return fail("lossless coding has no rate control".into());
            }
            if rc.bitrate == 0
                || !(0.5..=60.0).contains(&rc.max_delay_frames)
                || rc.min_qp > rc.max_qp
            {
                return fail(
                    "rate control: bitrate > 0, max delay 0.5..=60 frames, min_qp <= max_qp".into(),
                );
            }
        }
        if !self.isa.is_supported() {
            return fail(format!(
                "{} kernels are not available on this CPU",
                self.isa
            ));
        }
        Ok(())
    }
}

/// What one frame cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    /// Bytes written, including the frame's length prefix.
    pub bytes: usize,
    /// Display post-filter strength written in the header, in 1/8 units (0 = off).
    pub postfilter_strength: u8,
    /// Band segment bytes per component.
    pub component_bytes: [usize; 3],
    /// Frame header bytes (including the payload length prefix and the type/state bytes).
    pub header_bytes: usize,
    /// The layout section: its varint length prefix plus the range-coded lengths.
    pub layout_bytes: usize,
    /// The in-loop filter section: its varint length prefix plus the coefficients.
    pub loop_bytes: usize,
    pub qp: u8,
    /// Coded as a P-frame or a B-frame. False for intra frames, and for frames repeated or
    /// skipped.
    pub inter: bool,
    /// Coded as a B-frame (§6.9).
    pub bi: bool,
    /// P-frames: the motion section; B-frames: their extension and motion sections (the
    /// global-motion and node figures are the P-frames' alone).
    pub motion: Option<MotionStats>,
    /// Bands dropped to stay within the rate-control cap.
    pub dropped_bands: usize,
    /// Nothing fitted the cap, so a Repeat frame was sent instead.
    pub repeated: bool,
    /// Not even a Repeat frame fitted the cap, so nothing was written: the receiver keeps showing
    /// its last picture.
    pub skipped: bool,
    /// Rate control only: the frame's target, cap and estimate in bits, and the transmit queue
    /// after this frame's interval.
    pub rate: Option<RateStats>,
    /// Encoding time per stage.
    pub times: StageTimes,
}

/// A P-frame's motion.
#[derive(Clone, Copy, Debug, Default)]
pub struct MotionStats {
    /// Motion section bytes, including its length prefix.
    pub bytes: usize,
    pub global: GlobalMotion,
    pub nodes: usize,
    /// The interpolation filter the frame uses.
    pub filter: Filter,
    /// The frame's prediction mode.
    pub prediction: Prediction,
    /// Nodes whose vector differs from the global prediction.
    pub refined_nodes: usize,
    pub intra_nodes: usize,
    /// Patches block-matched for global motion estimation, and how many fitted the model.
    pub patches: usize,
    pub inlier_patches: usize,
    /// PSNR of the luma prediction against the source, before the residual.
    pub prediction_psnr: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RateStats {
    pub target_bits: f64,
    pub cap_bits: f64,
    pub estimated_bits: f64,
    pub queue_bits: f64,
}

/// Per-band diagnostics of the most recent frame.
#[derive(Clone, Copy, Debug)]
pub struct BandStats {
    pub comp: usize,
    pub level: usize,
    pub orient: Orient,
    pub coefficients: usize,
    /// Nonzero coded values (for predicted LL bands, nonzero residuals).
    pub nonzero: u32,
    /// Sum of the nonzero values' bit lengths: their raw magnitude-and-sign bits.
    pub magnitude_bits: u64,
    /// Segment bytes; 0 when the band is empty or dropped.
    pub bytes: usize,
    pub dropped: bool,
    /// Rate control only: histogram estimates of `nonzero` and `magnitude_bits` at this qp.
    pub estimate: Option<(f64, f64)>,
}

/// Per-component buffers.
struct Component {
    levels: usize,
    /// Each component has its own transform engine, for its own thread.
    dwt: Transform,
    /// Centred input or residual; the forward transform uses it as scratch.
    work: Plane<i16>,
    /// Transform coefficients, then dequantized coefficients for the reconstruction.
    coeffs: Plane<i16>,
    /// LL residuals when LL prediction is on.
    ll_res: Plane<i32>,
    /// Nonzero 4×4 blocks of every band, for the sparse inverse transform.
    maps: Vec<BlockMap>,
    /// Reconstructed samples or residual, centred.
    out: Plane<i16>,
    /// P-frames: the prediction's transform, for the band coder's guide.
    pred_coeffs: Plane<i16>,
    /// The 9/7 engine for the prediction's transform, which runs beside the residual's.
    pred_dwt: Option<Dwt97>,
}

pub struct Encoder {
    cfg: EncoderConfig,
    seq: SequenceHeader,
    slots: Vec<BandSlot>,
    contexts: Vec<Contexts>,
    /// Per band and coding class, entropy contexts carried across fixed-qp frames.
    band_contexts: Vec<std::sync::Mutex<[Contexts; 2]>>,
    comps: [Component; 3],
    /// Every band's quantized values, in coding order.
    values: Vec<Plane<i32>>,
    pyramids: Vec<SigPyramid>,
    /// Per band, its rows holding a nonzero value after the last reconstruction (by row pairs
    /// for detail bands), and per component and level its bands: the inverse 9/7 skips the rest.
    nonzero_rows: Vec<Vec<bool>>,
    level_bands: [Vec<LevelBands>; 3],
    /// P-frames: every band's prediction guide.
    guides: Vec<Option<Guide>>,
    /// One coded segment per band (empty when the band is all zero or dropped).
    segments: Vec<Vec<u8>>,
    histograms: Vec<MagnitudeHistogram>,
    band_stats: Vec<BandStats>,
    rate: Option<RateController>,
    frame_num: u16,
    /// Fields a short header can inherit from the previous coded frame.
    last_header: Option<FrameHeaderState>,
    /// The decoder's picture: the output of the last sent frame and the next frame's reference.
    recon: Frame,
    /// Reused luma/chroma sources sharpened for the perceptual loop-filter fit.
    sharp_src: [Plane<u8>; 3],
    /// Whether `recon` holds a picture to repeat or predict from.
    has_frame: bool,
    /// The current frame's motion-compensated prediction.
    pred: Frame,
    field: MotionField,
    /// The previous frame's field: its vectors seed the node search.
    previous_field: MotionField,
    /// The global motion of the last sent P-frame (zero after an intra frame). P-frames code
    /// theirs as a change from it.
    global: GlobalMotion,
    motion_coder: MotionCoder,
    /// B-frame motion contexts, per direction, carried on fixed-qp streams.
    b_tree: [TreeContexts; 2],
    /// The coded motion section of the current P-frame.
    motion_segment: Vec<u8>,
    /// Every plane's last loop-filter coefficients since the last intra frame.
    loop_coefs: [Coefs; 3],
    /// The planes the loop filter chose in the last P-frame that fitted it.
    loop_chosen: [Option<Coefs>; 3],
    /// Frames since the last intra frame.
    since_intra: u32,
    /// The scene's display post-filter strength in 1/8 units; chosen at intra frames under
    /// [`PostFilterMode::Auto`] and held through the scene.
    scene_postfilter: u8,
    /// Whether this frame's LL bands are coded with prediction.
    ll_prediction: bool,
    /// This frame's θ.
    rounding: f64,
    /// This P-frame's interpolation filter.
    filter: Filter,
    /// This P-frame's prediction mode.
    prediction: Prediction,
    estimator: GlobalEstimator,
    /// B-frames (§6.9): the picture kept for the next B-frame, the last B-frame's
    /// reconstruction, whether the frame being coded is a pair's anchor, whether the last anchor
    /// was a P-frame, the last anchor's bytes and a B-frame's bytes relative to its anchor's.
    past: Option<Frame>,
    /// The last P-frame's motion section over its bytes (0 after an intra frame): what decides
    /// the ultraheavy tier's next pair of frames (§10.4.1).
    motion_share: f64,
    /// The link bits a B-frame group's frames may use together, fixed when its anchor is planned:
    /// the B-frames then take their learned share of it.
    group_budget: f64,
    /// Mean global displacement over the frame's corners, in pixels per frame, smoothed: the
    /// camera speed. Static content (a talking head, a still) wants different motion, B-frame
    /// and in-loop filter decisions than a fast flight.
    camera_speed: f64,
    /// Reusable B-frame prediction buffers, so the per-B search does not allocate four 1.4 MB
    /// pictures every time.
    b_spare: Vec<Frame>,
    /// Scratch for a B-frame's reconstruction; the kept middle B-frame of a group and its
    /// position; every frame's reconstruction of the last group (display order); its length.
    b_recon: Option<Frame>,
    bref: Option<(Frame, usize)>,
    group_recons: Vec<Frame>,
    group_len: usize,
    pair_anchor: bool,
    anchor_inter: bool,
    anchor_bytes: usize,
    b_share: f64,
    /// A coarse luma signature of the last source frame, and its mean absolute change to this
    /// one: a cheap static/held-shot detector (realtime only).
    source_sample: Vec<u8>,
    source_change: f64,
    /// The last sent frame's luma PSNR, for the rate controller's quality ceiling: the ceiling
    /// waits until the picture has recovered from a coarse keyframe.
    last_psnr: f64,
    /// The last estimated global motion, where the next search starts.
    motion_hint: Affine,
    /// Whether the global model failed the scene-cut check this frame.
    cut_suspect: bool,
    /// qp chosen for the last frame (sent or not), for motion decisions made before this
    /// frame's qp is known.
    last_qp: u8,
    /// Running activity for the causal frame-qp adaptation, separately for intra and inter
    /// frames, and how many frames have fed it.
    aq_intra: f64,
    aq_inter: f64,
    aq_intra_frames: u32,
    aq_inter_frames: u32,
    /// One-shot intra qp offset, used by the CLI for an unreferenced opening frame.
    next_intra_offset: Option<i8>,
    /// The current frame's stage times.
    times: StageTimes,
    pool: Pool,
    /// Band indices, largest first, and each band's place in that order.
    by_size: Vec<usize>,
    /// Symbol statistics, when recording them (see [`Self::record_contexts`]).
    training: Option<ContextStats>,
    rank: Vec<usize>,
}

/// Bytes a varint takes.
fn varint_len(v: usize) -> usize {
    (usize::BITS - v.leading_zeros()).div_ceil(7).max(1) as usize
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Result<Self> {
        cfg.validate()?;
        let wavelet = if cfg.lossless {
            Wavelet::Reversible53
        } else {
            cfg.wavelet
        };
        let finest_tilt = if cfg.finest_tilt_auto && cfg.rate_control.is_none() && !cfg.lossless {
            auto_finest_tilt(cfg.qp)
        } else {
            cfg.finest_tilt
        };
        let weights = |levels: usize, scale: f64, finest_tilt: f64, hh_tilt: f64, mid_tilt: f64| {
            if cfg.lossless {
                vec![4096; 1 + 3 * levels]
            } else {
                let mut w = mse_weights(levels, wavelet);
                if levels > 0 && finest_tilt != 1.0 {
                    let start = 1 + 3 * (levels - 1);
                    for weight in &mut w[start..1 + 3 * levels] {
                        *weight = (f64::from(*weight) * finest_tilt)
                            .round()
                            .clamp(1.0, 65535.0) as u16;
                    }
                }
                if levels > 0 && hh_tilt != 1.0 {
                    let hh = 3 * levels;
                    w[hh] = (f64::from(w[hh]) * hh_tilt).round().clamp(1.0, 65535.0) as u16;
                }
                if levels >= 2 && mid_tilt != 1.0 {
                    let start = 1 + 3 * (levels - 2);
                    for weight in &mut w[start..start + 3] {
                        *weight =
                            (f64::from(*weight) * mid_tilt).round().clamp(1.0, 65535.0) as u16;
                    }
                }
                w.into_iter()
                    .map(|w| (f64::from(w) * scale).round().clamp(1.0, 65535.0) as u16)
                    .collect()
            }
        };
        let seq = SequenceHeader {
            width: cfg.width,
            height: cfg.height,
            fps_num: cfg.fps_num,
            fps_den: cfg.fps_den,
            luma_levels: cfg.luma_levels,
            chroma_levels: cfg.chroma_levels,
            block_log2: cfg.block_log2,
            mesh_log2: cfg.mesh_log2,
            wavelet,
            luma_weights: weights(cfg.luma_levels, 1.0, finest_tilt, cfg.hh_tilt, cfg.mid_tilt),
            chroma_weights: weights(cfg.chroma_levels, cfg.chroma_step_scale, 1.0, 1.0, 1.0),
            guide: cfg.guide,
            continuous_cdfs: cfg.rate_control.is_none(),
            layout_coding: true,
            band_contexts: cfg.rate_control.is_none(),
            temporal_mv: false,
            temporal_motion: cfg.temporal_motion && cfg.rate_control.is_none(),
        };
        let slots = codec::band_slots(&seq);
        let comps = std::array::from_fn(|comp| {
            let (w, h) = codec::component_size(&seq, comp);
            let ll = &slots
                .iter()
                .find(|s| s.comp == comp && s.is_ll())
                .expect("every component has an LL band")
                .sb;
            Component {
                levels: codec::component_levels(&seq, comp),
                dwt: Transform::with_isa(wavelet, w, h, cfg.isa).expect("validated ISA"),
                work: Plane::new(w, h),
                coeffs: Plane::new(w, h),
                ll_res: Plane::new(ll.width, ll.height),
                maps: codec::band_maps(&seq, comp),
                out: Plane::new(w, h),
                pred_coeffs: Plane::new(w, h),
                pred_dwt: match wavelet {
                    Wavelet::Lifted97 => {
                        Some(Dwt97::with_isa(w, h, cfg.isa).expect("validated ISA"))
                    }
                    Wavelet::Reversible53 => None,
                },
            }
        });
        let frame_rate = f64::from(cfg.fps_num) / f64::from(cfg.fps_den);
        let grid = Grid::new(cfg.width, cfg.height, cfg.mesh_log2);
        let contexts = codec::default_contexts();
        let band_contexts = if seq.band_contexts {
            slots
                .iter()
                .map(|slot| {
                    std::sync::Mutex::new([
                        contexts[slot.class].clone(),
                        contexts[slot.class + codec::FRAME_CLASSES].clone(),
                    ])
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            contexts,
            band_contexts,
            recon: Frame::new(cfg.width, cfg.height),
            pred: Frame::new(cfg.width, cfg.height),
            sharp_src: std::array::from_fn(|comp| {
                let (w, h) = codec::component_size(&seq, comp);
                Plane::new(w, h)
            }),
            field: MotionField::new(grid),
            previous_field: MotionField::new(grid),
            global: GlobalMotion::default(),
            motion_coder: {
                let mut m = MotionCoder::new(cfg.rate_control.is_none());
                m.set_temporal(cfg.temporal_motion && cfg.rate_control.is_none());
                m
            },
            b_tree: std::array::from_fn(|_| TreeContexts::default()),
            motion_segment: Vec::new(),
            loop_coefs: Default::default(),
            loop_chosen: [None; 3],
            values: codec::band_planes(&slots),
            pyramids: Vec::with_capacity(slots.len()),
            nonzero_rows: slots.iter().map(|s| vec![false; s.sb.height]).collect(),
            level_bands: std::array::from_fn(|comp| LevelBands::of(&slots, comp)),
            guides: Vec::with_capacity(slots.len()),
            segments: vec![Vec::new(); slots.len()],
            histograms: Vec::with_capacity(slots.len()),
            band_stats: Vec::with_capacity(slots.len()),
            rate: cfg
                .rate_control
                .clone()
                .map(|rc| RateController::new(rc, frame_rate)),
            frame_num: 0,
            last_header: None,
            has_frame: false,
            since_intra: 0,
            scene_postfilter: 0,
            ll_prediction: cfg.ll_prediction,
            rounding: cfg.rounding,
            filter: cfg.filter.unwrap_or_default(),
            prediction: Prediction::Warp,
            estimator: {
                let mut e = GlobalEstimator::new(cfg.width, cfg.height);
                // Light and medium keep the full node search, so the cheaper realtime fit of the
                // global model costs under 1% of the bits and saves 1.4–1.6 ms a frame.
                e.realtime = cfg.realtime || cfg.effort <= 1;
                e
            },
            past: None,
            motion_share: 0.0,
            group_budget: 0.0,
            camera_speed: f64::INFINITY,
            source_sample: Vec::new(),
            source_change: f64::INFINITY,
            b_spare: Vec::new(),
            b_recon: None,
            bref: None,
            group_recons: Vec::new(),
            group_len: 2,
            pair_anchor: false,
            anchor_inter: false,
            anchor_bytes: 0,
            b_share: 0.4,
            last_psnr: 0.0,
            motion_hint: Affine::default(),
            cut_suspect: false,
            last_qp: cfg.qp,
            aq_intra: 0.0,
            aq_inter: 0.0,
            aq_intra_frames: 0,
            aq_inter_frames: 0,
            next_intra_offset: None,
            times: StageTimes::default(),
            pool: Pool::new(cfg.threads),
            by_size: by_size(&slots),
            training: None,
            rank: {
                let mut rank = vec![0; slots.len()];
                for (r, i) in by_size(&slots).into_iter().enumerate() {
                    rank[i] = r;
                }
                rank
            },
            cfg,
            seq,
            slots,
            comps,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    pub fn sequence_header(&self) -> &SequenceHeader {
        &self.seq
    }

    /// The decoder's output for the most recently encoded frame.
    pub fn recon(&self) -> &Frame {
        &self.recon
    }

    /// The motion-compensated prediction of the most recent coded frame. For an intra frame this
    /// is stale; diagnostics should use a flat frame there.
    pub fn prediction(&self) -> &Frame {
        &self.pred
    }

    /// The motion field of the most recent P-frame.
    pub fn motion_field(&self) -> &MotionField {
        &self.field
    }

    /// Per-band diagnostics of the most recently encoded frame, in coding order.
    pub fn band_stats(&self) -> &[BandStats] {
        &self.band_stats
    }

    /// Changes the link rate from the next frame on. Needs rate control.
    pub fn set_bitrate(&mut self, bitrate: u64) -> Result<()> {
        match &mut self.rate {
            Some(rc) if bitrate > 0 => {
                rc.set_bitrate(bitrate);
                Ok(())
            }
            Some(_) => Err(Error::InvalidConfig("bitrate must be positive".into())),
            None => Err(Error::InvalidConfig("rate control is not enabled".into())),
        }
    }

    /// Overrides the intra qp offset for the next intra frame only.
    pub fn set_next_intra_offset(&mut self, offset: Option<i8>) {
        self.next_intra_offset = offset;
    }

    /// The strength written into this frame's header (1/8 units).
    fn frame_postfilter(&mut self, frame: &Frame, inter: bool, qp: u8) -> u8 {
        match self.cfg.postfilter {
            PostFilterMode::Off => 0,
            PostFilterMode::Fixed(strength) => strength.min(40),
            PostFilterMode::Auto => {
                if !inter {
                    // Intra frames are coded finer than the inter frames around them; choose
                    // for the scene's nominal quality, not the boosted keyframe.
                    self.scene_postfilter = postfilter::auto_strength(frame, qp.max(self.cfg.qp));
                }
                self.scene_postfilter
            }
        }
    }

    /// Encodes one frame and appends it, length-prefixed, to `out`.
    /// Mean squared prediction error of the luma plane (every other row and column), the
    /// inter-frame activity for [`Self::adapt_frame_qp`].
    fn inter_activity(&self, frame: &Frame) -> f64 {
        let (w, h) = (frame.width(), frame.height());
        let mut sum = 0u64;
        let mut n = 0u64;
        for y in (0..h).step_by(2) {
            let a = frame.y.row(y);
            let b = self.pred.y.row(y);
            for x in (0..w).step_by(2) {
                let d = i32::from(a[x]) - i32::from(b[x]);
                sum += (d * d) as u64;
                n += 1;
            }
        }
        (sum as f64 / n.max(1) as f64).max(1.0)
    }

    /// Mean absolute luma Laplacian (every fourth row and column) for intra frames.
    fn intra_activity(&self, frame: &Frame) -> f64 {
        let (w, h) = (frame.width(), frame.height());
        if w < 3 || h < 3 {
            return 1.0;
        }
        let mut sum = 0u64;
        let mut n = 0u64;
        for y in (1..h - 1).step_by(4) {
            let p = frame.y.row(y);
            let up = frame.y.row(y - 1);
            let down = frame.y.row(y + 1);
            for x in (1..w - 1).step_by(4) {
                let c = 4 * i32::from(p[x])
                    - i32::from(p[x - 1])
                    - i32::from(p[x + 1])
                    - i32::from(up[x])
                    - i32::from(down[x]);
                sum += c.unsigned_abs() as u64;
                n += 1;
            }
        }
        (sum as f64 / n.max(1) as f64).max(1.0)
    }

    /// Moves `qp` against a causal running activity: hard frames get a finer qp, easy ones a
    /// coarser one. The mean move is near zero, so the rate-quality curve keeps its operating
    /// point; only the allocation across frames changes. Encoder-only: qp is in the header.
    fn adapt_frame_qp(&mut self, qp: u8, activity: Option<f64>, inter: bool) -> u8 {
        let Some(activity) = activity else {
            return qp;
        };
        let (avg, frames) = if inter {
            (&mut self.aq_inter, &mut self.aq_inter_frames)
        } else {
            (&mut self.aq_intra, &mut self.aq_intra_frames)
        };
        let delta = if *frames >= 2 && self.cfg.frame_aq > 0.0 {
            let ratio = (*avg / activity).log2();
            (self.cfg.frame_aq * ratio).round().clamp(
                -f64::from(self.cfg.frame_aq_clamp),
                f64::from(self.cfg.frame_aq_clamp),
            ) as i32
        } else {
            0
        };
        *avg = if *frames == 0 {
            activity
        } else {
            0.8 * *avg + 0.2 * activity
        };
        *frames += 1;
        qp.saturating_add_signed(delta as i8)
    }

    /// Start every band's carried contexts from the trained tables (an intra frame must not
    /// inherit the statistics of a corrupt or dropped frame).
    fn reset_band_contexts(&mut self) {
        if self.band_contexts.is_empty() {
            return;
        }
        let defaults = codec::default_contexts();
        for (slot, ctx) in self.slots.iter().zip(&self.band_contexts) {
            let mut g = ctx.lock().expect("band contexts");
            g[0] = defaults[slot.class].clone();
            g[1] = defaults[slot.class + codec::FRAME_CLASSES].clone();
        }
    }

    pub fn encode_frame(&mut self, frame: &Frame, out: &mut Vec<u8>) -> Result<FrameStats> {
        if (frame.width(), frame.height()) != (self.seq.width, self.seq.height) {
            return Err(Error::InvalidConfig(format!(
                "frame is {}x{}, stream is {}x{}",
                frame.width(),
                frame.height(),
                self.seq.width,
                self.seq.height
            )));
        }
        let frame_num = self.frame_num;
        self.frame_num = self.frame_num.wrapping_add(1);
        self.times = StageTimes::default();
        let mut watch = Stopwatch::start();

        let mut inter = self.cfg.p_frames
            && self.has_frame
            && (self.cfg.intra_period == 0 || self.since_intra + 1 < self.cfg.intra_period);
        let mut global = GlobalMotion::default();
        let mut patches = (0, 0);
        let hint = self.motion_hint;
        if self.camera_speed.is_finite() && self.camera_speed < 2.0 {
            self.source_change = self.sample_source(frame);
        }
        if inter {
            match self.estimate_motion(frame, &mut watch) {
                Some((g, matched, inliers)) => (global, patches) = (g, (matched, inliers)),
                None => inter = false,
            }
        }
        if inter {
            (self.prediction, self.filter) = self.choose_filter(frame);
            self.predict();
            if self.cut_suspect && self.prediction_fails(frame) {
                // A scene cut after all: undo the motion state, as if never estimated.
                inter = false;
                self.motion_hint = hint;
                std::mem::swap(&mut self.previous_field, &mut self.field);
            }
        }
        watch.lap(&mut self.times, Stage::Prediction);
        let activity =
            (self.cfg.frame_aq > 0.0 && self.rate.is_none() && !self.cfg.lossless).then(|| {
                if inter {
                    self.inter_activity(frame)
                } else {
                    self.intra_activity(frame)
                }
            });
        self.transform(frame, inter);
        watch.lap(&mut self.times, Stage::Transform);
        self.motion_segment.clear();
        if !inter {
            self.motion_coder.reset();
            self.reset_band_contexts();
        }
        if inter {
            let g = global;
            // The field as decoded (offsets normalized to quarter pels).
            self.motion_segment = self.motion_coder.encode(&self.global, &g, &mut self.field);
        }
        watch.lap(&mut self.times, Stage::Entropy);
        self.ll_prediction = self.cfg.ll_prediction && !inter;
        self.rounding = if inter {
            self.cfg.inter_rounding
        } else {
            self.cfg.rounding
        };

        if self.pair_anchor {
            if let Some(rc) = &self.rate {
                self.group_budget = self.group_len as f64 * rc.budget() - rc.queue_bits();
            }
        }
        let plan = if self.rate.is_some() {
            self.build_histograms();
            self.rate.as_ref().map(|rc| {
                let bits = |qp| self.estimate_bits(rc, inter, qp);
                let sse = |qp| self.estimate_luma_sse(inter, qp);
                let pixels = (self.seq.width * self.seq.height) as f64;
                let cap = self.quality_cap(rc);
                if self.pair_anchor {
                    // The pair's two frame budgets, less the B-frame's expected share.
                    let target = (self.group_len as f64 * rc.budget() - rc.queue_bits())
                        / (1.0 + self.b_share);
                    rc.plan_target_quality_capped(inter, target, bits, sse, pixels, false, cap)
                } else {
                    rc.plan_quality_capped(
                        inter,
                        !inter && self.cfg.p_frames,
                        bits,
                        sse,
                        pixels,
                        cap,
                    )
                }
            })
        } else {
            None
        };
        if !inter && plan.is_some() {
            self.next_intra_offset = None;
        }
        let nominal_qp = match plan {
            Some(p) => p.qp,
            None if self.cfg.lossless => 0,
            // The offset sharpens intra frames relative to the P-frames that predict from
            // them; with no P-frames there is nothing to be relative to.
            None if !inter && self.cfg.total_frames.is_some_and(|t| frame_num + 2 >= t) => self
                .cfg
                .qp
                .saturating_add_signed(self.cfg.final_intra_qp_offset),
            None if inter => self.cfg.qp.saturating_add_signed(self.cfg.inter_qp_offset),
            None if !self.cfg.p_frames => {
                self.next_intra_offset = None;
                self.cfg.qp
            }
            None => self.cfg.qp.saturating_add_signed(
                self.next_intra_offset
                    .take()
                    .unwrap_or(self.cfg.intra_qp_offset),
            ),
        };
        let qp = if plan.is_none() {
            self.adapt_frame_qp(nominal_qp, activity, inter)
        } else {
            nominal_qp
        };
        let postfilter_strength = self.frame_postfilter(frame, inter, qp);
        let mut header = FrameHeader {
            frame_type: if inter {
                FrameType::Inter
            } else {
                FrameType::Intra
            },
            frame_num,
            qp,
            postfilter_strength,
            recon_offset: if self.cfg.lossless {
                0
            } else {
                let offset = if inter {
                    self.cfg.inter_recon_offset + fine_ramp(plan.map_or(self.cfg.qp, |p| p.qp))
                } else {
                    self.cfg.recon_offset
                };
                (offset * 256.0).round().min(255.0) as u8
            },
            parent_contexts: self.cfg.parent_contexts,
            ll_prediction: self.ll_prediction,
            filter: if inter { self.filter } else { Filter::Bilinear },
            prediction: if inter {
                self.prediction
            } else {
                Prediction::Warp
            },
            loop_filter: false,
            keep_reference: self.pair_anchor,
        };
        if inter {
            self.rounding = self.cfg.inter_rounding + fine_ramp(header.qp);
        }
        self.quantize(&header);
        watch.lap(&mut self.times, Stage::Quantize);
        self.code_bands(&header);
        watch.lap(&mut self.times, Stage::Entropy);

        let coded_bits = 8.0
            * (2 + self.fixed_bytes(inter)
                + self.layout_cost()
                + self.segments.iter().map(|s| s.len()).sum::<usize>()) as f64;
        let (dropped_bands, repeated) = match plan {
            Some(p) => {
                let (dropped, mut overflow) = self.fit_cap(p.cap, inter);
                if overflow && inter {
                    overflow = !self.fall_back_to_global(&global, p.cap);
                }
                (dropped, overflow || (!inter && self.starved()))
            }
            None => (0, false),
        };
        self.last_qp = header.qp;
        let start = out.len();
        let mut stats = FrameStats {
            qp: header.qp,
            postfilter_strength: header.postfilter_strength,
            dropped_bands,
            repeated,
            ..FrameStats::default()
        };
        if repeated {
            let mut payload = Vec::new();
            FrameHeader {
                frame_type: FrameType::Repeat,
                qp: 0,
                recon_offset: 0,
                ..header
            }
            .write(&mut payload);
            if 8.0 * (payload.len() + 1) as f64 > plan.map_or(f64::INFINITY, |p| p.cap) {
                stats.skipped = true;
            } else {
                put_varint(out, payload.len() as u64);
                out.extend_from_slice(&payload);
            }
        } else {
            watch.lap(&mut self.times, Stage::Entropy);
            // Before the reconstruction takes over the prediction's buffer.
            let prediction_psnr = inter.then(|| crate::metrics::psnr_u8(&frame.y, &self.pred.y));
            self.reconstruct(&header);
            // What the cap leaves for the loop filter, the frame's own length prefix included.
            let budget = plan.map(|p| {
                let payload = self.fixed_bytes(inter)
                    + self.layout_cost()
                    + self.segments.iter().map(|s| s.len()).sum::<usize>();
                ((p.cap / 8.0).floor() as usize).saturating_sub(payload + varint_len(payload) + 1)
            });
            let loop_segment = self.loop_filter(frame, inter, header.qp, budget);
            header.loop_filter = loop_segment.is_some();
            if self.rate.is_some() {
                self.last_psnr = crate::metrics::psnr_u8(&frame.y, &self.recon.y);
            }
            watch.lap(&mut self.times, Stage::Reconstruct);
            let mut payload = Vec::new();
            if self.cfg.short_headers {
                header.write_with(&mut payload, self.last_header.as_ref());
            } else {
                header.write(&mut payload);
            }
            self.last_header = Some(header.state());
            let header_len = payload.len();
            stats.header_bytes = varint_len(payload.len()) + header_len;
            if inter {
                put_varint(&mut payload, self.motion_segment.len() as u64);
                payload.extend_from_slice(&self.motion_segment);
                stats.inter = true;
                stats.motion = Some(MotionStats {
                    bytes: payload.len() - header_len,
                    global,
                    nodes: self.field.grid.nodes(),
                    filter: self.filter,
                    prediction: self.prediction,
                    refined_nodes: self.refined_nodes(&global),
                    intra_nodes: self.field.intra.iter().filter(|&&i| i).count(),
                    patches: patches.0,
                    inlier_patches: patches.1,
                    prediction_psnr: prediction_psnr.unwrap_or_default(),
                });
            }
            self.write_bands(&mut payload, &mut stats);
            if let Some(segment) = loop_segment {
                put_varint(&mut payload, segment.len() as u64);
                payload.extend_from_slice(&segment);
                stats.loop_bytes = varint_len(segment.len()) + segment.len();
            }
            put_varint(out, payload.len() as u64);
            out.extend_from_slice(&payload);
            watch.lap(&mut self.times, Stage::Entropy);
            self.estimator.set_reference(&self.pool, &frame.y);
            watch.lap(&mut self.times, Stage::GlobalMotion);
            self.has_frame = true;
            self.anchor_inter = inter;
            if inter {
                self.global = global;
                self.since_intra += 1;
            } else {
                self.global = GlobalMotion::default();
                self.since_intra = 0;
                // A cut tells nothing about the new scene's motion.
                self.camera_speed = f64::INFINITY;
            }
        }
        stats.bytes = out.len() - start;
        self.anchor_bytes = stats.bytes;
        self.motion_share = stats
            .motion
            .map_or(0.0, |m| m.bytes as f64 / stats.bytes.max(1) as f64);

        if let Some(p) = plan {
            let modelled = self
                .rate
                .as_ref()
                .map_or(0.0, |rc| self.estimate_bits(rc, inter, p.qp));
            let rc = self.rate.as_mut().expect("a plan implies rate control");
            rc.model.learn(inter, coded_bits, modelled);
            rc.finish_frame(8.0 * stats.bytes as f64);
            stats.rate = Some(RateStats {
                target_bits: p.target,
                cap_bits: p.cap,
                estimated_bits: p.estimate,
                queue_bits: rc.queue_bits(),
            });
        }
        watch.lap(&mut self.times, Stage::Quantize);
        stats.times = self.times;
        Ok(stats)
    }

    /// The forward transforms of the frame (intra) or of its residual from `self.pred` and of
    /// the prediction itself (inter, for the guide).
    fn transform(&mut self, frame: &Frame, inter: bool) {
        let (sources, preds) = (frame.planes(), self.pred.planes());
        let guide = self.seq.guide;
        self.pool.for_each_mut(&mut self.comps, |comp, c| {
            if c.dwt.uses_pool() {
                return;
            }
            if inter {
                codec::residual(sources[comp], preds[comp], &mut c.work);
            } else {
                center_u8(sources[comp], &mut c.work);
            }
            c.dwt.forward(None, &mut c.work, c.levels, &mut c.coeffs);
            if inter && guide {
                // `out` is free until the reconstruction.
                center_u8(preds[comp], &mut c.out);
                c.dwt
                    .forward(None, &mut c.out, c.levels, &mut c.pred_coeffs);
            }
        });
        // The 9/7 reads the residual or the centred samples straight from the planes. A frame's
        // transforms (a P-frame's residual and prediction of every component) run level by level
        // together, so the small ones fill in around luma's.
        let input = |comp: usize, residual: bool| {
            let (src, pred) = (sources[comp], preds[comp]);
            move |y: usize, row: &mut [i16]| {
                let w = src.width();
                match (inter, residual) {
                    (true, true) => {
                        residual_row(src.row_span(y, 0, w), pred.row_span(y, 0, w), row)
                    }
                    (true, false) => center_row(pred.row_span(y, 0, w), row),
                    (false, _) => center_row(src.row_span(y, 0, w), row),
                }
            }
        };
        let inputs: Vec<_> = (0..3)
            .flat_map(|comp| [input(comp, true), input(comp, false)])
            .collect();
        let mut jobs = Vec::with_capacity(6);
        for (comp, c) in self.comps.iter_mut().enumerate() {
            let Component {
                dwt: Transform::W97(dwt),
                pred_dwt,
                coeffs,
                pred_coeffs,
                levels,
                ..
            } = c
            else {
                continue;
            };
            jobs.push(ForwardJob {
                dwt,
                input: &inputs[2 * comp],
                levels: *levels,
                coeffs,
            });
            if inter && self.seq.guide {
                jobs.push(ForwardJob {
                    dwt: pred_dwt.as_mut().expect("the 9/7 has a prediction engine"),
                    input: &inputs[2 * comp + 1],
                    levels: *levels,
                    coeffs: pred_coeffs,
                });
            }
        }
        Dwt97::forward_many(Some(&self.pool), &mut jobs);
    }

    /// Encodes a group of frames (display order): the last as an anchor P-frame whose header
    /// keeps its reference, then the others as B-frames between it and the picture before the
    /// group, hierarchically (§6.9): of 2, the first; of 4, the middle one (kept as a reference
    /// for the others), then the first and the third. Before the first picture, or without
    /// P-frames, the frames are coded as ordinary frames in display order. Appends the frames
    /// to `out` in coding order and returns their stats in display order; their
    /// reconstructions are [`Self::group_recon`].
    pub fn encode_group(
        &mut self,
        frames: &[&Frame],
        out: &mut Vec<u8>,
    ) -> Result<Vec<FrameStats>> {
        let n = frames.len();
        if !matches!(n, 2 | 4) {
            return Err(Error::InvalidConfig(
                "B-frame groups have 2 or 4 frames".into(),
            ));
        }
        self.group_recons.resize_with(n, || self.recon.clone());
        if !self.has_frame || !self.cfg.p_frames {
            let mut stats = Vec::with_capacity(n);
            for (i, f) in frames.iter().enumerate() {
                stats.push(self.encode_frame(f, out)?);
                self.group_recons[i].clone_from(&self.recon);
            }
            return Ok(stats);
        }
        let first_num = self.frame_num;
        self.frame_num = first_num.wrapping_add(n as u16 - 1);
        match &mut self.past {
            Some(past) => past.clone_from(&self.recon),
            None => self.past = Some(self.recon.clone()),
        }
        self.group_len = n;
        self.pair_anchor = true;
        let anchor_stats = self.encode_frame(frames[n - 1], out);
        self.pair_anchor = false;
        let anchor_stats = anchor_stats?;
        self.group_recons[n - 1].clone_from(&self.recon);
        let mut stats = vec![FrameStats::default(); n];
        // (position, past reference, next reference, kept as a reference)
        let order: &[(usize, usize, usize, bool)] = if n == 2 {
            &[(1, 0, 2, false)]
        } else {
            &[(2, 0, 4, true), (1, 0, 2, false), (3, 2, 4, false)]
        };
        let mut b_bytes = 0;
        for &(k, past_pos, next_pos, keep) in order {
            let num = first_num.wrapping_add(k as u16 - 1);
            stats[k - 1] = if anchor_stats.repeated || anchor_stats.skipped {
                // No new anchor: the B-frames show the last picture again.
                let start = out.len();
                let mut payload = Vec::new();
                FrameHeader {
                    frame_type: FrameType::Repeat,
                    frame_num: num,
                    qp: 0,
                    recon_offset: 0,
                    parent_contexts: false,
                    ll_prediction: false,
                    filter: Filter::Bilinear,
                    prediction: Prediction::Warp,
                    loop_filter: false,
                    keep_reference: false,
                    postfilter_strength: self.scene_postfilter,
                }
                .write(&mut payload);
                put_varint(out, payload.len() as u64);
                out.extend_from_slice(&payload);
                if let Some(rc) = &mut self.rate {
                    rc.finish_frame(8.0 * (out.len() - start) as f64);
                }
                self.group_recons[k - 1].clone_from(&self.recon);
                FrameStats {
                    repeated: true,
                    bytes: out.len() - start,
                    postfilter_strength: self.scene_postfilter,
                    ..FrameStats::default()
                }
            } else {
                // A static scene's B-frame predicts almost perfectly: it can afford the
                // anchor's quality (a fine offset) and saves its P-frame sibling's residual.
                let offset = if self.static_camera() {
                    // At the very low rates the samtime chart samples, a coarser B leaves the
                    // anchor more of the group budget; above qp 50 it measures 1–2% better.
                    if anchor_stats.qp >= 50 {
                        self.cfg.b_static_qp_offset + 2
                    } else {
                        self.cfg.b_static_qp_offset
                    }
                } else if keep {
                    B_REF_QP_OFFSET
                } else {
                    self.cfg.b_qp_offset
                };
                let s = self.encode_b(
                    frames[k - 1],
                    num,
                    anchor_stats.qp,
                    [k, n, past_pos, next_pos],
                    keep,
                    offset,
                    out,
                );
                b_bytes += s.bytes;
                s
            };
        }
        // The B-frames' share of a group, for the next anchor's target.
        let share = b_bytes as f64 / self.anchor_bytes.max(1) as f64;
        self.b_share = 0.8 * self.b_share + 0.2 * share;
        stats[n - 1] = anchor_stats;
        self.frame_num = first_num.wrapping_add(n as u16);
        Ok(stats)
    }

    /// Encodes one, two or four frames (display order) as the ultraheavy tier does (§10.4.1):
    /// two as a B-frame group when the last P-frame's motion section was over
    /// [`PAIR_MOTION_SHARE`] of its bytes (motion is dear, so a B-frame that predicts from both
    /// sides for a few bytes of motion pays), else as P-frames, as one frame is; four as one
    /// group of 4 on a still camera ([`STILL_GROUP_SPEED`]), else as two such pairs. Returns
    /// the stats in display order; the reconstructions are [`Self::group_recon`].
    pub fn encode_pair(&mut self, frames: &[&Frame], out: &mut Vec<u8>) -> Result<Vec<FrameStats>> {
        match frames {
            // A still camera: one hierarchical group of 4 (the middle frame a reference for the
            // outer ones). Static talking heads gain 8% over pairs; moving cameras never do
            // (§10.4.1), so they code two pairs.
            [a, b, c, d] => {
                if self.has_frame && self.cfg.p_frames && self.camera_speed < STILL_GROUP_SPEED {
                    self.encode_group(&[a, b, c, d], out)
                } else {
                    let mut stats = self.encode_pair(&[a, b], out)?;
                    let first = self.group_recons.clone();
                    stats.extend(self.encode_pair(&[c, d], out)?);
                    let second = std::mem::replace(&mut self.group_recons, first);
                    self.group_recons.extend(second);
                    Ok(stats)
                }
            }
            [a, b]
                if self.has_frame
                    && self.cfg.p_frames
                    && (self.motion_share > PAIR_MOTION_SHARE || self.static_camera()) =>
            {
                self.encode_group(&[a, b], out)
            }
            [_] | [_, _] => {
                let mut stats = Vec::with_capacity(frames.len());
                self.group_recons
                    .resize_with(frames.len(), || self.recon.clone());
                for (i, f) in frames.iter().enumerate() {
                    stats.push(self.encode_frame(f, out)?);
                    self.group_recons[i].clone_from(&self.recon);
                }
                Ok(stats)
            }
            _ => Err(Error::InvalidConfig("pairs have 1, 2 or 4 frames".into())),
        }
    }

    /// Sends a Repeat frame: the decoder shows its last picture again. The caller must know the
    /// source is identical to the previous frame; a still video then costs 2 bytes a frame and
    /// keeps the picture the decoder already has. The reconstruction, motion state and reference
    /// are unchanged, so the next P-frame is unaffected.
    pub fn encode_repeat(&mut self, out: &mut Vec<u8>) -> Result<FrameStats> {
        if !self.has_frame {
            return Err(Error::InvalidConfig(
                "a Repeat frame needs a previous picture".into(),
            ));
        }
        let frame_num = self.frame_num;
        self.frame_num = self.frame_num.wrapping_add(1);
        let start = out.len();
        let mut payload = Vec::new();
        FrameHeader {
            frame_type: FrameType::Repeat,
            frame_num,
            qp: 0,
            recon_offset: 0,
            parent_contexts: false,
            ll_prediction: false,
            filter: Filter::Bilinear,
            prediction: Prediction::Warp,
            loop_filter: false,
            keep_reference: false,
            postfilter_strength: self.scene_postfilter,
        }
        .write(&mut payload);
        put_varint(out, payload.len() as u64);
        out.extend_from_slice(&payload);
        let bytes = out.len() - start;
        let mut stats = FrameStats {
            repeated: true,
            bytes,
            postfilter_strength: self.scene_postfilter,
            ..FrameStats::default()
        };
        if let Some(rc) = &mut self.rate {
            rc.finish_frame(8.0 * bytes as f64);
            stats.rate = Some(RateStats {
                target_bits: 0.0,
                cap_bits: 0.0,
                estimated_bits: 0.0,
                queue_bits: rc.queue_bits(),
            });
        }
        self.anchor_bytes = bytes;
        Ok(stats)
    }

    /// The last sent frame's luma PSNR when rate control is on (0 before the first frame).
    pub fn last_psnr(&self) -> f64 {
        self.last_psnr
    }

    /// The reconstruction of frame `i` (display order) of the last [`Self::encode_group`].
    pub fn group_recon(&self, i: usize) -> &Frame {
        &self.group_recons[i]
    }

    /// Warps `reference` into `out` with this B-frame's filter.
    fn warp_b(&self, reference: &Frame, field: &MotionField, out: &mut Frame) {
        warp_planes(
            &self.pool,
            &reference.planes(),
            out.planes_mut().into(),
            field,
            Prediction::Overlapped,
            self.b_filter(),
            self.cfg.isa,
            1,
        );
    }

    fn b_filter(&self) -> Filter {
        self.cfg.filter.unwrap_or(Filter::Cubic)
    }

    /// The blend mode's prediction for a B-frame, with its two field sections, its root-mode
    /// section and their bytes in all. `dirs` holds per direction its section, prediction,
    /// direct prediction, field (as decoded) and scaled prediction.
    #[allow(clippy::type_complexity)]
    fn blend_candidate(
        &self,
        refs: &[Frame; 2],
        dirs: &[(Vec<u8>, Frame, Frame, MotionField, Vec<[i32; 2]>)],
        err: &[[f64; 3]],
        lambda: f64,
    ) -> (Frame, [Vec<u8>; 2], Vec<u8>, usize) {
        let grid = dirs[0].3.grid;
        let r = tree::root_log2(grid.log2);
        let size = 1usize << r;
        let (across, down) = bframe::root_count(&grid);
        // `err` holds per root the luma squared error of the past, next and averaged searched
        // predictions, from the mode decision's pass.
        // Approximate bits of a root's offsets in a field.
        let root_nodes = |root: usize| {
            let (i0, j0) = ((root % across) * size, (root / across) * size);
            (j0..(j0 + size).min(grid.ny))
                .flat_map(move |j| (i0..(i0 + size).min(grid.nx)).map(move |i| j * grid.nx + i))
        };
        let bits = |dir: usize, root: usize| -> f64 {
            let (field, pred) = (&dirs[dir].3, &dirs[dir].4);
            root_nodes(root)
                .map(|n| {
                    if field.intra[n] {
                        return 3.0;
                    }
                    (0..2)
                        .map(|c| {
                            let d = (field.mv[n][c] - pred[n][c]) / 2;
                            if d == 0 {
                                0.5
                            } else {
                                3.0 + 2.0 * f64::from(d.unsigned_abs()).log2()
                            }
                        })
                        .sum::<f64>()
                })
                .sum()
        };
        let mut modes: Vec<u8> = (0..across * down)
            .map(|root| {
                let e = err[root];
                let costs = [
                    e[0] + lambda * bits(0, root),
                    e[1] + lambda * bits(1, root),
                    e[2] + lambda * (bits(0, root) + bits(1, root)),
                ];
                (0..3)
                    .min_by(|&a, &b| costs[a].total_cmp(&costs[b]))
                    .expect("three") as u8
            })
            .collect();
        let mut fields = [dirs[0].3.clone(), dirs[1].3.clone()];
        for (root, &m) in modes.iter().enumerate() {
            let drop = match m {
                bframe::ROOT_PAST => Some(1),
                bframe::ROOT_NEXT => Some(0),
                _ => None,
            };
            if let Some(d) = drop {
                for n in root_nodes(root) {
                    fields[d].mv[n] = dirs[d].4[n];
                    fields[d].intra[n] = false;
                    fields[d].depth[n] = r as u8;
                }
            }
        }
        let sections: [Vec<u8>; 2] = std::array::from_fn(|d| {
            let mut coder = RangeEncoder::new();
            tree::code(
                &mut coder,
                &mut TreeContexts::default(),
                &mut fields[d],
                &dirs[d].4,
            );
            coder.finish()
        });
        let mut coder = RangeEncoder::new();
        bframe::code_root_modes(&mut coder, &mut modes, across);
        let mode_section = coder.finish();
        let mut pred = Frame::new(self.seq.width, self.seq.height);
        self.warp_b(&refs[0], &fields[0], &mut pred);
        let mut next = Frame::new(self.seq.width, self.seq.height);
        self.warp_b(&refs[1], &fields[1], &mut next);
        bframe::blend(&mut pred, &next, &grid, &modes);
        let bytes = sections
            .iter()
            .chain([&mode_section])
            .map(|s| varint_len(s.len()) + s.len())
            .sum();
        (pred, sections, mode_section, bytes)
    }

    fn take_b_frame(&mut self) -> Frame {
        self.b_spare
            .pop()
            .unwrap_or_else(|| Frame::new(self.seq.width, self.seq.height))
    }

    /// Codes `frame` as the B-frame between the kept picture (`self.past`) and the anchor just
    /// coded (`self.recon`), at `anchor_qp` plus [`B_QP_OFFSET`]. Leaves every state the next
    /// frames read as it was: the anchor's reconstruction, field and global motion.
    #[allow(clippy::too_many_arguments)]
    fn encode_b(
        &mut self,
        frame: &Frame,
        frame_num: u16,
        anchor_qp: u8,
        [k, n, past_pos, next_pos]: [usize; 4],
        keep: bool,
        offset: u8,
        out: &mut Vec<u8>,
    ) -> FrameStats {
        let mut watch = Stopwatch::start();
        self.times = StageTimes::default();
        // The qp the motion decisions assume; with rate control the frame's own follows below.
        let base = if self.rate.is_some() {
            anchor_qp
        } else {
            self.cfg.qp
        };
        let mut qp = base.saturating_add(offset);
        let anchor_field = self.field.clone();
        // The references by position: the picture before the group, the anchor, or the kept
        // middle B-frame.
        let pick = |pos: usize| -> Frame {
            if pos == 0 {
                self.past.clone().expect("encode_group kept it")
            } else if pos == n {
                self.recon.clone()
            } else {
                self.bref.as_ref().expect("coded before").0.clone()
            }
        };
        let refs = [pick(past_pos), pick(next_pos)];
        let (k, n) = (k as i32, n as i32);
        let step = 2f64.powf(f64::from(qp) / 8.0);
        let lambda = 0.12 * self.cfg.motion_lambda * B_MOTION_LAMBDA * step * step;
        let params = MeshParams {
            lambda,
            refine: self.cfg.mesh_refinement,
            wide: self.effort() >= 2,
            realtime: self.cfg.realtime,
            stripe_rows: self.search_stripes(),
        };
        let passes = self.refine_passes();
        // Per direction: the scaled prediction, the searched field (as decoded) with its coded
        // section, and both predictions.
        let mut dirs = Vec::with_capacity(2);
        for (dir, reference) in refs.iter().enumerate() {
            let num = k - if dir == 0 { past_pos } else { next_pos } as i32;
            let pred =
                bframe::scaled_vectors(&anchor_field, &self.global, self.anchor_inter, num, n);
            let direct = bframe::direct_field(&anchor_field, &pred);
            let mut field = direct.clone();
            partition::search(
                &self.pool,
                &frame.y,
                &reference.y,
                &mut field,
                Some(&direct),
                &params,
            );
            if params.refine && passes > 0 {
                let skip_half = (self.cfg.effort == 2 && self.static_camera())
                    || (self.cfg.effort >= 3 && self.camera_speed < 0.2);
                let half_pel = params.wide && !skip_half;
                crate::motion::refine::refine(
                    &self.pool,
                    &frame.y,
                    &reference.y,
                    &mut field,
                    &pred,
                    lambda,
                    passes,
                    half_pel,
                    true,
                );
            }
            let mut coder = RangeEncoder::new();
            tree::code(&mut coder, &mut TreeContexts::default(), &mut field, &pred);
            let section = coder.finish();
            let mut searched = self.take_b_frame();
            self.warp_b(reference, &field, &mut searched);
            let mut direct_pred = self.take_b_frame();
            self.warp_b(reference, &direct, &mut direct_pred);
            dirs.push((section, searched, direct_pred, field, pred));
        }
        watch.lap(&mut self.times, Stage::Mesh);
        // The mode: luma squared error plus λ times the motion's bits. One pass per pair of
        // predictions gives the past/next/both errors per root (the blend needs them too), so
        // six full-frame passes and six candidate copies become two passes and one copy.
        let grid = dirs[0].3.grid;
        let (err, searched_sse) =
            root_errors(&self.pool, &frame.y, &dirs[0].1.y, &dirs[1].1.y, &grid);
        let (_, direct_sse) = root_errors(&self.pool, &frame.y, &dirs[0].2.y, &dirs[1].2.y, &grid);
        let sse = |m: BiMode| -> f64 {
            let e = match m {
                BiMode::Past | BiMode::DirectPast => 0,
                BiMode::Next | BiMode::DirectNext => 1,
                _ => 2,
            };
            if m.searched() {
                searched_sse[e]
            } else {
                direct_sse[e]
            }
        };
        let motion_bytes = |m: BiMode| -> usize {
            if !m.searched() {
                return 0;
            }
            m.directions()
                .iter()
                .zip(&dirs)
                .filter(|(used, _)| **used)
                .map(|(_, d)| varint_len(d.0.len()) + d.0.len())
                .sum()
        };
        let (best_cost, mut mode) = [
            BiMode::Past,
            BiMode::Next,
            BiMode::Both,
            BiMode::DirectPast,
            BiMode::DirectNext,
            BiMode::DirectBoth,
        ]
        .into_iter()
        .map(|m| (sse(m) + lambda * 8.0 * motion_bytes(m) as f64, m))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("six candidates");
        let both = |a: &Frame, b: &Frame| {
            let mut avg = a.clone();
            bframe::average_into(&mut avg, b);
            avg
        };
        let mut pred = match mode {
            BiMode::Past => dirs[0].1.clone(),
            BiMode::Next => dirs[1].1.clone(),
            BiMode::Both => both(&dirs[0].1, &dirs[1].1),
            BiMode::DirectPast => dirs[0].2.clone(),
            BiMode::DirectNext => dirs[1].2.clone(),
            BiMode::DirectBoth => both(&dirs[0].2, &dirs[1].2),
            BiMode::Blend => unreachable!("the blend is considered below"),
        };
        // Blend: every root its own choice, with the motion of the directions it drops zeroed.
        let blended = self.blend_candidate(&refs, &dirs, &err, lambda);
        let blend_cost =
            frame_sse(&self.pool, &frame.y, &blended.0.y) + lambda * 8.0 * blended.3 as f64;
        let mut blend_sections = None;
        if blend_cost < best_cost {
            (mode, pred) = (BiMode::Blend, blended.0);
            blend_sections = Some((blended.1, blended.2));
        }
        let prediction_psnr = crate::metrics::psnr_u8(&frame.y, &pred.y);
        self.pred = pred;
        watch.lap(&mut self.times, Stage::Prediction);
        self.transform(frame, true);
        watch.lap(&mut self.times, Stage::Transform);
        if self.rate.is_some() {
            // What the anchor left of the link, within a few qp steps of the fixed offset.
            self.ll_prediction = false;
            self.rounding = self.cfg.inter_rounding;
            self.build_histograms();
            let rc = self.rate.as_ref().expect("checked");
            // The group's plan set the anchor's target with the B-frames' share in it: a
            // B-frame keeps its offset, coarser only if the delay cap needs it.
            let pixels = (self.seq.width * self.seq.height) as f64;
            let estimate = |q: u8| self.estimate_bits(rc, true, q);
            let sse = |q: u8| self.estimate_luma_sse(true, q);
            let cap = self.quality_cap(rc);
            if self.static_camera() {
                // A static scene's B-frame is cheap and its budget share is meaningful: let the
                // share decide, between a little below the anchor's quality (a B-frame is not a
                // reference) and the offset above it.
                let b_frames = (self.group_len as f64 - 1.0).max(1.0);
                let target = self.group_budget * self.b_share / (1.0 + self.b_share) / b_frames;
                let planned =
                    rc.plan_target_quality_capped(true, target, estimate, sse, pixels, false, cap);
                qp = planned.qp.clamp(base.saturating_add(B_MIN_QP_OFFSET), qp);
            } else {
                // Moving content: keep the measured offset and only coarsen if the cap needs it.
                let capped = rc.plan_target_quality_capped(
                    true,
                    f64::INFINITY,
                    estimate,
                    sse,
                    pixels,
                    false,
                    cap,
                );
                qp = qp.max(capped.qp);
            }
        }
        self.ll_prediction = false;
        self.rounding = self.cfg.inter_rounding + fine_ramp(qp);
        let header = FrameHeader {
            frame_type: FrameType::Bi,
            frame_num,
            qp,
            recon_offset: ((self.cfg.inter_recon_offset + fine_ramp(qp)) * 256.0)
                .round()
                .min(255.0) as u8,
            parent_contexts: self.cfg.parent_contexts,
            ll_prediction: false,
            filter: self.b_filter(),
            prediction: Prediction::Overlapped,
            loop_filter: false,
            keep_reference: keep,
            postfilter_strength: self.scene_postfilter,
        };
        self.quantize(&header);
        watch.lap(&mut self.times, Stage::Quantize);
        self.code_bands(&header);
        watch.lap(&mut self.times, Stage::Entropy);
        // Reconstructed in place of the anchor's picture, which stays.
        let mut b_recon = self.b_recon.take().unwrap_or_else(|| refs[0].clone());
        std::mem::swap(&mut self.recon, &mut b_recon);
        self.reconstruct(&header);
        std::mem::swap(&mut self.recon, &mut b_recon);
        self.group_recons[k as usize - 1].clone_from(&b_recon);
        if keep {
            match &mut self.bref {
                Some((f, p)) => {
                    f.clone_from(&b_recon);
                    *p = k as usize;
                }
                None => self.bref = Some((b_recon.clone(), k as usize)),
            }
        }
        self.b_recon = Some(b_recon);
        watch.lap(&mut self.times, Stage::Reconstruct);
        let start = out.len();
        let mut payload = Vec::new();
        if self.cfg.short_headers {
            header.write_with(&mut payload, self.last_header.as_ref());
        } else {
            header.write(&mut payload);
        }
        self.last_header = Some(header.state());
        let header_len = payload.len();
        debug_assert!(k < 8 && n < 8 && past_pos < 8 && next_pos < 8);
        let ext = u16::from(mode.code())
            | ((k as u16) << 3)
            | ((n as u16) << 6)
            | ((past_pos as u16) << 9)
            | ((next_pos as u16) << 12);
        payload.extend_from_slice(&ext.to_le_bytes());
        if let Some((sections, modes)) = &blend_sections {
            for section in sections.iter().chain([modes]) {
                put_varint(&mut payload, section.len() as u64);
                payload.extend_from_slice(section);
            }
        } else if mode.searched() {
            for (dir, (used, d)) in mode.directions().iter().zip(&mut dirs).enumerate() {
                if *used {
                    let mut coder = RangeEncoder::new();
                    let mut fresh = TreeContexts::default();
                    let ctx = if self.seq.continuous_cdfs {
                        &mut self.b_tree[dir]
                    } else {
                        &mut fresh
                    };
                    tree::code(&mut coder, ctx, &mut d.3, &d.4);
                    let section = coder.finish();
                    put_varint(&mut payload, section.len() as u64);
                    payload.extend_from_slice(&section);
                }
            }
        }
        if self.seq.continuous_cdfs && mode == BiMode::Blend {
            self.b_tree = std::array::from_fn(|_| TreeContexts::default());
        }
        let mut stats = FrameStats {
            qp,
            postfilter_strength: header.postfilter_strength,
            inter: true,
            bi: true,
            motion: Some(MotionStats {
                // The B extension and the motion sections.
                bytes: payload.len() - header_len,
                nodes: self.field.grid.nodes(),
                filter: self.b_filter(),
                prediction: Prediction::Overlapped,
                prediction_psnr,
                ..MotionStats::default()
            }),
            ..FrameStats::default()
        };
        self.write_bands(&mut payload, &mut stats);
        put_varint(out, payload.len() as u64);
        out.extend_from_slice(&payload);
        stats.bytes = out.len() - start;
        if let Some(rc) = &mut self.rate {
            rc.finish_frame(8.0 * stats.bytes as f64);
        }
        watch.lap(&mut self.times, Stage::Entropy);
        stats.times = self.times;
        for (_, searched, direct, _, _) in dirs {
            self.b_spare.push(searched);
            self.b_spare.push(direct);
        }
        stats
    }

    /// Refinement passes of the motion search: the effort, four at effort 3.
    fn refine_passes(&self) -> usize {
        match self.effort() {
            // A static camera below 50 fps converges in two passes: the third and fourth change
            // offsets without a BD gain on samtime, and cost a fifth of the encode time. At
            // 60 fps the field is still moving through the fourth pass.
            3 if self.static_camera() && self.cfg.fps_num < 50 * self.cfg.fps_den.max(1) => 2,
            3 => 4,
            // Light: one pass over the first candidates only, no descent (§10.1): 4.9% fewer
            // bits (5–9% on talking heads) for 29% more encode time. Realtime mode skips it.
            0 if !self.cfg.realtime => 1,
            e => usize::from(e),
        }
    }

    /// The motion search effort: 0 in realtime mode.
    fn effort(&self) -> u8 {
        if self.cfg.realtime {
            0
        } else {
            self.cfg.effort
        }
    }

    /// Mean absolute luma change between this frame and the last one, on a 32×18 sample. It
    /// costs a few hundred loads and keeps a held shot from paying for a global-motion fit.
    fn sample_source(&mut self, frame: &Frame) -> f64 {
        const COLS: usize = 32;
        const ROWS: usize = 18;
        let (w, h) = (frame.y.width(), frame.y.height());
        let mut now = Vec::with_capacity(COLS * ROWS);
        for r in 0..ROWS {
            let y = r * h / ROWS;
            for c in 0..COLS {
                now.push(frame.y.get(c * w / COLS, y));
            }
        }
        let change = if self.source_sample.len() == now.len() {
            self.source_sample
                .iter()
                .zip(&now)
                .map(|(a, b)| f64::from(a.abs_diff(*b)))
                .sum::<f64>()
                / now.len() as f64
        } else {
            f64::INFINITY
        };
        self.source_sample = now;
        change
    }

    /// Motion estimation: fills `self.field` and returns the global model with the number of
    /// patches matched and fitting it, or `None` at a scene cut.
    fn estimate_motion(
        &mut self,
        frame: &Frame,
        watch: &mut Stopwatch,
    ) -> Option<(GlobalMotion, usize, usize)> {
        let held =
            self.camera_speed.is_finite() && self.camera_speed < 0.2 && self.source_change < 2.0;
        let (model, matches, inliers) = if held {
            // A held shot: the previous global model is the identity, and a fresh affine fit
            // cannot move the background. The mesh still follows the foreground.
            self.cut_suspect = false;
            watch.lap(&mut self.times, Stage::GlobalMotion);
            let model = if self.camera_speed < 0.2 {
                Affine::default()
            } else {
                self.motion_hint
            };
            (model, 0, 0)
        } else {
            let estimate = self
                .estimator
                .estimate(&self.pool, &frame.y, &self.motion_hint);
            // Far above the threshold the global model's verdict stands. Just above it the frame
            // is only a suspect: fast flight with rotation or parallax fails the global model
            // too (shares of 0.50–0.51 on real FPV footage, against 0.87–0.97 at real cuts),
            // though the node vectors follow it. Then the prediction decides (see
            // `prediction_fails`).
            let share = self
                .estimator
                .unpredictable_share(&self.pool, &estimate.model);
            let cut = self.cfg.scene_cut > 0.0 && share > self.cfg.scene_cut;
            watch.lap(&mut self.times, Stage::GlobalMotion);
            if cut && share > CERTAIN_CUT.max(self.cfg.scene_cut) {
                return None;
            }
            self.cut_suspect = cut;
            (estimate.model, estimate.matches, estimate.inliers)
        };
        self.motion_hint = model;
        let mut global = model.to_global(&self.field.grid);
        // A static camera with a moving foreground often leaves a fraction of a pixel in the
        // affine fit; zeroing it lets background leaves reach exact [0, 0] and measured better
        // on both samtime clips (bits and PSNR). Fast camera motion keeps the model.
        let raw_speed = global
            .corners
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])) / 8.0)
            .sum::<f64>()
            / 4.0;
        let zero_below = if self.cfg.effort >= 3 { 0.70 } else { 1.25 };
        if raw_speed < zero_below {
            global = GlobalMotion::default();
        }
        // Mean corner displacement in pixels per frame, smoothed: the camera's speed. Static
        // content gets different B-frame, motion and in-loop filter decisions (see
        // `static_camera`).
        let speed = global
            .corners
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])) / 8.0)
            .sum::<f64>()
            / 4.0;
        self.camera_speed = if self.camera_speed.is_finite() {
            0.8 * self.camera_speed + 0.2 * speed
        } else {
            speed
        };
        std::mem::swap(&mut self.previous_field, &mut self.field);
        self.field.set_global(&global);
        // λ for squared error per bit: about 0.12·Δ² for a step Δ in pixel terms.
        let step = 2f64.powf(f64::from(self.motion_qp()) / 8.0);
        let params = MeshParams {
            lambda: 0.12 * self.cfg.motion_lambda * step * step,
            refine: self.cfg.mesh_refinement,
            wide: self.effort() >= 2,
            realtime: self.cfg.realtime,
            stripe_rows: self.search_stripes(),
        };
        partition::search(
            &self.pool,
            &frame.y,
            &self.recon.y,
            &mut self.field,
            Some(&self.previous_field),
            &params,
        );
        let passes = self.refine_passes();
        if self.cfg.mesh_refinement && passes > 0 {
            let grid = self.field.grid;
            let pred: Vec<[i32; 2]> = (0..grid.ny)
                .flat_map(|j| (0..grid.nx).map(move |i| (i, j)))
                .map(|(i, j)| global.predict(&grid, i, j))
                .collect();
            let skip_half = (self.cfg.effort == 2 && self.static_camera())
                || (self.cfg.effort >= 3 && self.camera_speed < 0.2);
            let half_pel = params.wide && !skip_half;
            let descent = self.effort() > 0;
            crate::motion::refine::refine(
                &self.pool,
                &frame.y,
                &self.recon.y,
                &mut self.field,
                &pred,
                params.lambda,
                passes,
                half_pel,
                descent,
            );
        }
        watch.lap(&mut self.times, Stage::Mesh);
        Some((global, matches, inliers))
    }

    /// The rate controller's quality ceiling for this frame, or 0 while the picture is still
    /// recovering: capping a P-frame that is repairing a coarse reference would trade away the
    /// reference's quality, not the surplus bits.
    fn quality_cap(&self, rc: &RateController) -> f64 {
        let cap = rc.quality_cap();
        if cap > 0.0 && self.last_psnr >= cap - QUALITY_CAP_GATE {
            cap
        } else {
            0.0
        }
    }

    /// Whether the camera is (nearly) static: a talking head, a still, a screen. Such content
    /// wants B-frame pairs, which a slow zoom does not.
    fn static_camera(&self) -> bool {
        self.camera_speed.is_finite() && self.camera_speed < STATIC_CAMERA_SPEED
    }

    /// Stripes suit a static camera, where the row above adds little, and the middle tiers;
    /// on a moving camera they cost up to 0.4% of the bits, and ultraheavy's finer search
    /// gains nothing from them.
    fn search_stripes(&self) -> usize {
        match self.effort() {
            0 if !self.static_camera() => 0,
            3 => 0,
            _ => SEARCH_STRIPES,
        }
    }

    /// qp the motion decisions of a P-frame assume, before its own qp is chosen. Without rate
    /// control that is the P-frame qp itself; the last frame's qp would carry
    /// `intra_qp_offset` right after an intra frame, halving λ. With rate control it is the qp
    /// chosen for the previous frame (§5.3).
    fn motion_qp(&self) -> u8 {
        match self.rate {
            Some(_) => self.last_qp,
            None => self.cfg.qp,
        }
    }

    fn refined_nodes(&self, global: &GlobalMotion) -> usize {
        let grid = self.field.grid;
        (0..grid.ny)
            .flat_map(|j| (0..grid.nx).map(move |i| (i, j)))
            .filter(|&(i, j)| self.field.mv[self.field.node(i, j)] != global.predict(&grid, i, j))
            .count()
    }

    /// Payload bytes besides the layout and band segments: header and the motion section of a
    /// P-frame (without the frame's length prefix).
    fn fixed_bytes(&self, inter: bool) -> usize {
        let motion = if inter {
            varint_len(self.motion_segment.len()) + self.motion_segment.len()
        } else {
            0
        };
        // type, varint frame number, qp, recon offset, flags, and the optional post-filter byte.
        1 + varint_len(self.frame_num as usize) + 4 + motion
    }

    /// The frame's layout section, length prefix included.
    fn layout_cost(&self) -> usize {
        let layout = self.encode_layout();
        varint_len(layout.len()) + layout.len()
    }

    fn encode_layout(&self) -> Vec<u8> {
        let lengths: Vec<usize> = self.segments.iter().map(Vec::len).collect();
        let classes: Vec<usize> = self.slots.iter().map(|s| s.class).collect();
        encode_layout(&lengths, &classes)
    }

    /// Appends the layout section and the band segments. The layout's lengths delimit them.
    fn write_bands(&self, payload: &mut Vec<u8>, stats: &mut FrameStats) {
        let layout = self.encode_layout();
        put_varint(payload, layout.len() as u64);
        payload.extend_from_slice(&layout);
        stats.layout_bytes = varint_len(layout.len()) + layout.len();
        for (i, segment) in self.segments.iter().enumerate() {
            if !segment.is_empty() {
                payload.extend_from_slice(segment);
                stats.component_bytes[self.slots[i].comp] += segment.len();
            }
        }
    }

    /// Magnitude histograms of every band, for the rate estimate. Predicted LL bands are
    /// estimated exactly instead (see [`Self::band_estimate`]).
    fn build_histograms(&mut self) {
        // One band per task.
        self.histograms = self.pool.map(self.slots.len(), |i| {
            let slot = &self.slots[i];
            if self.ll_predicted(slot) {
                MagnitudeHistogram::default()
            } else {
                MagnitudeHistogram::sparse(&self.comps[slot.comp].coeffs, slot.rect())
            }
        });
    }

    fn ll_predicted(&self, slot: &BandSlot) -> bool {
        slot.is_ll() && self.ll_prediction
    }

    /// Estimated nonzero count and raw magnitude bits of band `i` at `qp`. Predicted LL bands
    /// are tiny (about 1,400 values at 720p in all), so they are quantized and predicted for real.
    fn band_estimate(&self, i: usize, qp: u8) -> (f64, f64) {
        let slot = &self.slots[i];
        let step = slot.step_q12(&self.seq, qp);
        if !self.ll_predicted(slot) {
            return self.histograms[i].estimate(step as f64 / 4096.0, self.rounding);
        }
        let quant = Quantizer::new(step, self.rounding);
        let r = slot.rect();
        let c = &self.comps[slot.comp];
        let q: Vec<i32> = (0..r.h)
            .flat_map(|y| {
                c.coeffs
                    .row_span(r.y0 + y, r.x0, r.w)
                    .iter()
                    .map(|&v| quant.quantize(v))
            })
            .collect();
        let (nonzero, bits) = codec::ll_residual_stats(&q, r.w);
        (f64::from(nonzero), bits as f64)
    }

    /// Estimated luma squared error of the frame at `qp`, in the pixel domain, from the band
    /// histograms and the MSE step weights. Predicted LL bands are quantized exactly. Used only
    /// for the rate controller's quality ceiling.
    fn estimate_luma_sse(&self, inter: bool, qp: u8) -> f64 {
        let delta = self.recon_delta(inter, qp);
        let frac_scale = 1.0 / f64::from(1u32 << (2 * self.seq.wavelet.frac_bits()));
        let bands: f64 = self
            .pool
            .map(self.slots.len(), |i| {
                let slot = &self.slots[i];
                if slot.comp != 0 {
                    return 0.0;
                }
                let d = self.band_distortion(i, qp, delta);
                let w = f64::from(self.seq.luma_weights[slot.weight_index]);
                let gain = (4096.0 / w) * (4096.0 / w);
                d * gain * frac_scale
            })
            .into_iter()
            .sum();
        bands
    }

    /// Quantization error of one band at `qp`, in coefficient units: from its histogram, or
    /// exactly for a predicted LL band.
    fn band_distortion(&self, i: usize, qp: u8, delta: f64) -> f64 {
        let slot = &self.slots[i];
        let step = slot.step_q12(&self.seq, qp);
        if !self.ll_predicted(slot) {
            return self.histograms[i].distortion(step as f64 / 4096.0, self.rounding, delta);
        }
        let quant = Quantizer::new(step, self.rounding);
        let dequant = Dequantizer::new(step, (delta * 256.0).round().clamp(0.0, 255.0) as u8);
        let r = slot.rect();
        let c = &self.comps[slot.comp];
        let mut total = 0.0;
        for y in 0..r.h {
            for &v in c.coeffs.row_span(r.y0 + y, r.x0, r.w) {
                let q = quant.quantize(v);
                let e = f64::from(v) - f64::from(dequant.dequantize(q));
                total += e * e;
            }
        }
        total
    }

    /// The reconstruction offset δ this frame uses at `qp` (as the header will round it).
    fn recon_delta(&self, inter: bool, qp: u8) -> f64 {
        let offset = if inter {
            self.cfg.inter_recon_offset + fine_ramp(qp)
        } else {
            self.cfg.recon_offset
        };
        (offset * 256.0).round().clamp(0.0, 255.0) / 256.0
    }

    /// Modelled frame size in bits at `qp`, before the controller's learned correction.
    fn estimate_bits(&self, rc: &RateController, inter: bool, qp: u8) -> f64 {
        // Bands in parallel (the rate search tries about ten qps in turn), summed in order.
        let band_bits: Vec<f64> = self
            .pool
            .map(self.slots.len(), |i| {
                let (nonzero, bits) = self.band_estimate(i, qp);
                let slot = &self.slots[i];
                rc.model.band_bits(
                    slot.coding_class(inter),
                    slot.sb.width * slot.sb.height,
                    nonzero,
                    bits,
                )
            })
            .into_iter()
            .collect();
        // The layout section's cost from the same per-band estimates. `band_bits` includes the
        // range coder's per-segment overhead, which is not part of the layout length.
        let lengths: Vec<usize> = band_bits
            .iter()
            .map(|&bits| {
                if bits > 20.0 {
                    ((bits - 20.0) / 8.0).ceil() as usize
                } else {
                    0
                }
            })
            .collect();
        let classes: Vec<usize> = self.slots.iter().map(|s| s.class).collect();
        let layout = encode_layout(&lengths, &classes);
        let layout_bits = 8.0 * (varint_len(layout.len()) + layout.len()) as f64;
        8.0 * (2 + self.fixed_bytes(inter)) as f64 + band_bits.iter().sum::<f64>() + layout_bits
    }

    /// Quantizes every band, prepares the values each band codes (residuals for predicted LL
    /// bands) with their significance pyramids, and records per-band statistics. One band per
    /// task, the largest first.
    fn quantize(&mut self, header: &FrameHeader) {
        let (seq, slots, rounding) = (&self.seq, &self.slots, self.rounding);
        let isolated = if self.cfg.lossless {
            0
        } else {
            (self.cfg.isolated_threshold * 4096.0).round() as u64
        };
        let inter = header.frame_type.predicted();
        let mut coeffs = Vec::new();
        let mut pred_coeffs = Vec::new();
        let mut ll_res = Vec::new();
        for c in &mut self.comps {
            coeffs.push(&c.coeffs);
            pred_coeffs.push(&c.pred_coeffs);
            ll_res.push(Some(&mut c.ll_res));
        }
        let mut tasks: Vec<BandTask<'_>> = self
            .values
            .iter_mut()
            .zip(slots)
            .enumerate()
            .map(|(i, (values, slot))| BandTask {
                i,
                values,
                ll_res: if slot.is_ll() && header.ll_prediction {
                    ll_res[slot.comp].take()
                } else {
                    None
                },
                out: None,
                guide: None,
            })
            .collect();
        tasks.sort_by_key(|t| self.rank[t.i]);
        self.pool.for_each_mut(&mut tasks, |_, t| {
            let slot = &slots[t.i];
            let quant = Quantizer::new(slot.step_q12(seq, header.qp), rounding);
            let r = slot.rect();
            let (mut nonzero, mut bits) = (0, 0);
            // The band's nonzero 2×2 blocks (its pyramid's first level) and nonzero values per
            // 4×4 block (for `zero_isolated`), made while each row is fresh: no second read.
            let mut maps = codec::BandMaps::new(r);
            for y in 0..r.h {
                let row = t.values.row_span_mut(y, 0, r.w);
                let (n, b) =
                    quant.quantize_row_stats(coeffs[slot.comp].row_span(r.y0 + y, r.x0, r.w), row);
                nonzero += n;
                bits += b;
                if n > 0 {
                    maps.add_row(y, row);
                }
            }
            if !slot.is_ll() && isolated > 0 && nonzero > 0 {
                let step = slot.step_q12(seq, header.qp);
                let zeroed = maps.zero_isolated(
                    t.values,
                    coeffs[slot.comp],
                    r,
                    step,
                    isolated,
                    !self.cfg.sparse_zero,
                );
                // Each was a ±1: one nonzero, one magnitude bit.
                nonzero -= zeroed;
                bits -= u64::from(zeroed);
            }
            let rect = slot.local_rect();
            let residuals = t.ll_res.is_some();
            let plane: &Plane<i32> = match t.ll_res.as_deref_mut() {
                Some(res) => {
                    codec::ll_residuals(t.values, rect, res);
                    (nonzero, bits) = value_stats(res, rect);
                    res
                }
                None => t.values,
            };
            // An all-zero band is not coded, and as a parent it has no significant blocks.
            let (pyramid, guide) = if nonzero == 0 {
                (SigPyramid::empty(), None)
            } else {
                (
                    if residuals {
                        SigPyramid::build(plane, rect, seq.block_log2)
                    } else {
                        let p = maps.pyramid(seq.block_log2);
                        debug_assert_eq!(p, SigPyramid::build(plane, rect, seq.block_log2));
                        p
                    },
                    inter
                        .then(|| codec::band_guide(slot, seq, header.qp, pred_coeffs[slot.comp]))
                        .flatten(),
                )
            };
            t.out = Some((pyramid, nonzero, bits));
            t.guide = guide;
        });
        tasks.sort_by_key(|t| t.i);
        self.guides.clear();
        let outs: Vec<_> = tasks
            .into_iter()
            .map(|t| {
                self.guides.push(t.guide);
                t.out.expect("every band ran")
            })
            .collect();
        self.pyramids.clear();
        self.band_stats.clear();
        for (i, (pyramid, nonzero, magnitude_bits)) in outs.into_iter().enumerate() {
            let slot = &self.slots[i];
            let estimate = self
                .rate
                .is_some()
                .then(|| self.band_estimate(i, header.qp));
            self.band_stats.push(BandStats {
                comp: slot.comp,
                level: slot.sb.level,
                orient: slot.sb.orient,
                coefficients: slot.sb.width * slot.sb.height,
                nonzero,
                magnitude_bits,
                bytes: 0,
                dropped: false,
                estimate,
            });
            self.pyramids.push(pyramid);
        }
    }

    /// Codes every nonzero band into its segment, one band per task, the largest first.
    fn code_bands(&mut self, header: &FrameHeader) {
        let inter = header.frame_type.predicted();
        let (slots, values, pyramids) = (&self.slots, &self.values, &self.pyramids);
        let guides = &self.guides;
        let (comps, contexts, band_contexts, band_flag, block_log2) = (
            &self.comps,
            &self.contexts,
            &self.band_contexts,
            self.seq.band_contexts,
            self.seq.block_log2,
        );
        let by_size = &self.by_size;
        let segments = self.pool.map(slots.len(), |t| {
            let i = by_size[t];
            let slot = &slots[i];
            if !pyramids[i].any() {
                return Vec::new();
            }
            let plane = if slot.is_ll() && header.ll_prediction {
                &comps[slot.comp].ll_res
            } else {
                &values[i]
            };
            let parent = slot
                .parent
                .filter(|_| header.parent_contexts)
                .map(|p| Parent {
                    values: &values[p],
                    band: slots[p].local_rect(),
                    sig: &pyramids[p],
                });
            let class = usize::from(inter);
            let mut ctx = if band_flag {
                band_contexts[i].lock().expect("band contexts")[class].clone()
            } else {
                contexts[slot.coding_class(inter)].clone()
            };
            let mut enc = RangeEncoder::new();
            code_band(
                &mut enc,
                &mut ctx,
                plane,
                slot.local_rect(),
                parent,
                Some(&pyramids[i]),
                guides[i].as_ref(),
                block_log2,
                slot.transposed(),
            );
            if band_flag {
                band_contexts[i].lock().expect("band contexts")[class] = ctx;
            }
            enc.finish()
        });
        for (&i, segment) in self.by_size.iter().zip(segments) {
            self.band_stats[i].bytes = segment.len();
            self.segments[i] = segment;
        }
        if let Some(stats) = &mut self.training {
            // The same bands again, recording their symbols.
            for (i, slot) in slots.iter().enumerate() {
                if self.segments[i].is_empty() {
                    continue;
                }
                let plane = if slot.is_ll() && header.ll_prediction {
                    &comps[slot.comp].ll_res
                } else {
                    &values[i]
                };
                let parent = slot
                    .parent
                    .filter(|_| header.parent_contexts)
                    .map(|p| Parent {
                        values: &values[p],
                        band: slots[p].local_rect(),
                        sig: &pyramids[p],
                    });
                let class = slot.coding_class(inter);
                let mut ctx = contexts[class].clone();
                let mut rec = Recorder::default();
                code_band(
                    &mut rec,
                    &mut ctx,
                    plane,
                    slot.local_rect(),
                    parent,
                    Some(&pyramids[i]),
                    guides[i].as_ref(),
                    block_log2,
                    slot.transposed(),
                );
                stats.add(class, &rec.bands[0]);
            }
        }
    }

    /// Developer: from now on, also count the symbols every band and motion section codes, per
    /// context set, for training the default tables (see [`crate::training`]). Costs a second
    /// coding pass. Bands that rate control drops later in a frame are counted too.
    pub fn record_contexts(&mut self) {
        self.training.get_or_insert_with(ContextStats::default);
    }

    /// The statistics counted since [`Self::record_contexts`], which stops counting.
    pub fn take_context_stats(&mut self) -> Option<ContextStats> {
        self.training.take()
    }

    /// Drops bands from the end of the coding order (finest first) until the frame fits
    /// `cap_bits`. Bands only ever depend on coarser, earlier ones, so what remains decodes
    /// exactly. Intra frames keep their LL bands; P-frames may drop everything, since the motion
    /// alone still moves the picture. Returns the number dropped, and whether the frame still
    /// does not fit, in which case a Repeat frame goes out instead (once a picture exists to
    /// repeat).
    fn fit_cap(&mut self, cap_bits: f64, inter: bool) -> (usize, bool) {
        // Length prefix (generous), header, motion and layout.
        let fixed = 3 + self.fixed_bytes(inter);
        let mut bytes =
            fixed + self.layout_cost() + self.segments.iter().map(|s| s.len()).sum::<usize>();
        let mut dropped = 0;
        for i in (0..self.slots.len()).rev() {
            if 8.0 * bytes as f64 <= cap_bits || (self.slots[i].is_ll() && !inter) {
                break;
            }
            if !self.segments[i].is_empty() {
                self.segments[i].clear();
                bytes = fixed
                    + self.layout_cost()
                    + self.segments.iter().map(|s| s.len()).sum::<usize>();
                self.band_stats[i].bytes = 0;
                self.band_stats[i].dropped = true;
                dropped += 1;
                // The decoder sees zeros: so must the reconstruction.
                self.values[i].fill(0);
            }
        }
        (dropped, 8.0 * bytes as f64 > cap_bits && self.has_frame)
    }

    /// The emergency P-frame: when a P-frame does not fit `cap_bits` even with every band
    /// dropped, its node vectors and intra nodes go too, leaving the global motion alone (a few
    /// bytes). The picture keeps moving where a Repeat frame would freeze it. Returns whether
    /// that fits; if so the prediction is redone for the simpler field.
    fn fall_back_to_global(&mut self, global: &GlobalMotion, cap_bits: f64) -> bool {
        let mut field = MotionField::new(self.field.grid);
        field.set_global(global);
        if field == self.field {
            return false;
        }
        let mut enc = RangeEncoder::new();
        let mut g = *global;
        self.motion_coder
            .code(&mut enc, &self.global, &mut g, &mut field);
        let segment = enc.finish();
        let bytes =
            3 + 6 + varint_len(segment.len()) + segment.len() + self.slots.len().div_ceil(8);
        if 8.0 * bytes as f64 > cap_bits {
            return false;
        }
        self.field = field;
        self.motion_segment = segment;
        self.predict();
        true
    }

    /// Motion-compensates the reference into `self.pred`.
    /// Whether the P-frame prediction is worse than intra coding (the block's own activity plus
    /// a noise margin) on more than the scene-cut share of 16×16 luma blocks.
    fn prediction_fails(&self, frame: &Frame) -> bool {
        const B: usize = 16;
        const MARGIN: u32 = 3;
        let (cur, pred) = (&frame.y, &self.pred.y);
        let (bw, bh) = (cur.width() / B, cur.height() / B);
        let n = (B * B) as u32;
        let worse: usize = self
            .pool
            .map(bh, |by| {
                (0..bw)
                    .filter(|&bx| {
                        let rows = || {
                            (by * B..(by + 1) * B)
                                .map(|y| (cur.row_span(y, bx * B, B), pred.row_span(y, bx * B, B)))
                        };
                        let sum: u32 = rows().flat_map(|(c, _)| c).map(|&v| u32::from(v)).sum();
                        // Both at n times the sample scale.
                        let (mut inter, mut intra) = (0u32, 0u32);
                        for (c, p) in rows() {
                            for (&c, &p) in c.iter().zip(p) {
                                inter += u32::from(c.abs_diff(p)) * n;
                                intra += (u32::from(c) * n).abs_diff(sum);
                            }
                        }
                        inter > intra + MARGIN * n * n
                    })
                    .count()
            })
            .iter()
            .sum();
        bw * bh > 0 && worse as f64 > self.cfg.scene_cut * (bw * bh) as f64
    }

    fn predict(&mut self) {
        warp_planes(
            &self.pool,
            &self.recon.planes(),
            self.pred.planes_mut().into(),
            &self.field,
            self.prediction,
            self.filter,
            self.cfg.isa,
            1,
        );
    }

    /// The P-frame's interpolation filter: the configured one, or else whichever of cubic,
    /// bilinear and smooth predicts every 8th luma row with the smallest squared error.
    fn choose_filter(&mut self, frame: &Frame) -> (Prediction, Filter) {
        const ROW_STEP: usize = 8;
        let filters = match self.cfg.filter {
            Some(filter) => vec![filter],
            // In realtime mode cubic without the trial: bilinear saved a millisecond on the A55
            // for 0.01–0.04 dB of PSNR-Y on drone footage, but costs 3.7% of YUV-PSNR 6:1:1 BD
            // on nine clips (talking heads 4–12%) for 2–5% of x86 encode time.
            None if self.cfg.realtime => vec![Filter::Cubic],
            // With the 9/7, cubic wins over the per-frame choice: the smoother filters lower
            // the prediction error by blurring away detail the residual then has to code.
            None => vec![Filter::Cubic],
        };
        let predictions = match self.cfg.prediction {
            Some(p) => vec![p],
            // Overlapped prediction always, at every effort: it saves trying the warp and measured
            // 0.01–0.06 dB *better* here (0.3–0.6% fewer bits at effort 2), so the choice was a
            // pure cost. The warp stays available when forced.
            None => vec![Prediction::Overlapped],
        };
        if filters.len() == 1 && predictions.len() == 1 {
            return (predictions[0], filters[0]);
        }
        // The warp wins only on rigid motion, where cubic wins: its other filters are not tried
        // unless forced.
        let forced = self.cfg.filter.is_some();
        let mut best = ((predictions[0], filters[0]), u64::MAX);
        for (prediction, filter) in predictions
            .iter()
            .flat_map(|&p| filters.iter().map(move |&f| (p, f)))
            .filter(|&(p, f)| forced || p != Prediction::Warp || f == Filter::Cubic)
        {
            warp_planes(
                &self.pool,
                &[&self.recon.y],
                vec![&mut self.pred.y],
                &self.field,
                prediction,
                filter,
                self.cfg.isa,
                ROW_STEP,
            );
            let pred = &self.pred.y;
            let ssd: u64 = (0..pred.height())
                .filter(|&y| sampled(y, ROW_STEP))
                .map(|y| {
                    frame
                        .y
                        .row(y)
                        .iter()
                        .zip(pred.row(y))
                        .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                        .sum::<u64>()
                })
                .sum();
            if ssd < best.1 {
                best = ((prediction, filter), ssd);
            }
        }
        best.0
    }

    /// Whether rate control squeezed an intra frame so far that even its LL bands quantized to
    /// zero, though the picture is not flat. Decoding that would replace the last picture with a
    /// featureless grey one, so a Repeat frame is better.
    fn starved(&self) -> bool {
        const FLAT: u16 = 16;
        let ll = || self.slots.iter().enumerate().filter(|(_, s)| s.is_ll());
        let has_content = |slot: &BandSlot| {
            let r = slot.rect();
            let coeffs = &self.comps[slot.comp].coeffs;
            (0..r.h).any(|y| {
                coeffs
                    .row_span(r.y0 + y, r.x0, r.w)
                    .iter()
                    .any(|c| c.unsigned_abs() >= FLAT)
            })
        };
        self.has_frame
            && ll().all(|(i, _)| self.segments[i].is_empty())
            && ll().any(|(_, slot)| has_content(slot))
    }

    /// Rebuilds the frame exactly as the decoder will, one component per task.
    /// Fits the loop filter of every plane to `frame`, applies it to the reconstruction where
    /// it saves more than its bits cost, and returns the segment to send, if any (§5.4). The
    /// segment, with its length prefix, stays within `budget` bytes.
    fn loop_filter(
        &mut self,
        frame: &Frame,
        inter: bool,
        qp: u8,
        budget: Option<usize>,
    ) -> Option<Vec<u8>> {
        if !inter {
            self.loop_coefs = Default::default();
        }
        if !self.cfg.loop_filter || self.cfg.lossless {
            return None;
        }
        // On a static camera the filter's coefficients and segment cost more than they return
        // at realtime quality; measured on both samtime clips at qp 36–58.
        if self.cfg.realtime && self.static_camera() {
            return None;
        }
        // Under rate control the filter's bits compete with the bands, and at 300 kbit/s to
        // 3 Mbit/s it measured 0.08 dB better to 0.14 dB worse for 5–16% of the frame time:
        // in the fast tier, spend them on the bands. At fixed qp it is 0.5 dB on detailed
        // motion, so it stays.
        if self.cfg.realtime && self.rate.is_some() {
            return None;
        }
        // Squared error per bit, as for motion decisions. A luma fit on sampled rows overstates
        // what it saves on the whole plane, so light, medium and heavy (every eighth or fourth
        // pair) demand LOOP_SAMPLED_LAMBDA times more for luma. Chroma keeps the plain λ: with
        // the 6× there too, YUV-PSNR 6:1:1 lost 0.3–0.4% (forest 1.5%). Ultraheavy fits on every
        // second pair and realtime reuses fits; neither gained.
        let step = 2f64.powf(f64::from(qp) / 8.0);
        let sampled = !self.cfg.realtime && self.effort() <= 2;
        let chroma_lambda = 0.12 * step * step;
        let lambda = chroma_lambda * if sampled { LOOP_SAMPLED_LAMBDA } else { 1.0 };
        // Fitting on every fourth pair of rows costs nothing measurable at 720p and 1080p
        // (§10.1), and it is four times cheaper than the old every-row fit at effort 2–3;
        // effort 0 uses every eighth pair (0.3% more bits at most).
        let fit_step = match self.effort() {
            0 => 8,
            // The ultraheavy tier keeps a finer fit (every second pair): the every-fourth fit
            // cost up to 2% of the bits at the low-rate end on the slow zoom.
            3 => 2,
            _ => 4,
        };
        // In realtime mode every other P-frame reuses the last fit (0.01 dB, a millisecond on
        // the A55). Filtering luma only would save more but cost chroma 0.4 dB.
        let refit = !self.cfg.realtime || self.since_intra.is_multiple_of(REALTIME_FIT_PERIOD);
        let mut chosen: [Option<Coefs>; 3] = if refit { [None; 3] } else { self.loop_chosen };
        let sources = frame.planes();
        let sharpen = self.cfg.loop_sharpen;
        if sharpen > 0.0 {
            sharpen_source(sources[0], sharpen, &mut self.sharp_src[0]);
        }
        for (k, (_, rec)) in sources
            .iter()
            .zip(self.recon.planes())
            .enumerate()
            .filter(|_| refit)
        {
            let src = if k == 0 && sharpen > 0.0 {
                &self.sharp_src[0]
            } else {
                sources[k]
            };
            let fit = loopfilter::fit(&self.pool, src, rec, fit_step);
            let bits = loopfilter::coded_bits(&fit.coefs, &self.loop_coefs[k]);

            let lambda = if k == 0 { lambda } else { chroma_lambda };
            if fit.coefs != Coefs::default() && fit.saved > lambda * f64::from(bits + 1) {
                chosen[k] = Some(fit.coefs);
            }
        }
        let segment = |chosen: &[Option<Coefs>; 3]| {
            let mut w = loopfilter::BitWriter::default();
            for (coefs, previous) in chosen.iter().zip(&self.loop_coefs) {
                w.bit(coefs.is_some());
                if let Some(coefs) = coefs {
                    loopfilter::write_coefs(&mut w, coefs, previous);
                }
            }
            w.bytes
        };
        self.loop_chosen = chosen;
        // Over the budget, chroma goes first.
        let mut bytes = segment(&chosen);
        for k in (0..3).rev() {
            if budget.is_none_or(|b| varint_len(bytes.len()) + bytes.len() <= b) {
                break;
            }
            chosen[k] = None;
            bytes = segment(&chosen);
        }
        if chosen.iter().all(Option::is_none) {
            return None;
        }
        // The prediction is free once the reconstruction is done.
        let planes = self
            .recon
            .planes_mut()
            .into_iter()
            .zip(self.pred.planes_mut());
        for (k, (rec, tmp)) in planes.enumerate() {
            if let Some(coefs) = chosen[k] {
                self.loop_coefs[k] = coefs;
                loopfilter::apply(&self.pool, rec, tmp, &coefs);
                std::mem::swap(rec, tmp);
            }
        }
        Some(bytes)
    }

    fn reconstruct(&mut self, header: &FrameHeader) {
        let inter = header.frame_type.predicted();
        if inter {
            // The prediction becomes the picture; the old reference is no longer needed.
            std::mem::swap(&mut self.recon, &mut self.pred);
        }
        let (seq, slots, segments, pool) = (&self.seq, &self.slots, &self.segments, &self.pool);
        let pyramids = &self.pyramids;
        let sparse = self.comps.iter().all(|c| !c.dwt.uses_pool());
        // Every band into its component's coefficients, one band per task.
        {
            let mut coeffs = Vec::with_capacity(3);
            let mut maps: Vec<Vec<Option<&mut BlockMap>>> = Vec::with_capacity(3);
            for c in self.comps.iter_mut() {
                coeffs.push(c.coeffs.shared());
                maps.push(c.maps.iter_mut().map(Some).collect());
            }
            let mut items: Vec<_> = slots
                .iter()
                .zip(&mut self.nonzero_rows)
                .map(|(slot, rows)| {
                    let map = maps[slot.comp][slot.weight_index]
                        .take()
                        .expect("one band per map");
                    (map, rows)
                })
                .collect();
            pool.for_each_mut(&mut items, |i, (map, rows)| {
                let (slot, r, values) = (&slots[i], slots[i].rect(), &self.values[i]);
                // Not sent: all zero, or dropped (and zeroed) by `fit_cap`.
                let deq = (!segments[i].is_empty())
                    .then(|| Dequantizer::new(slot.step_q12(seq, header.qp), header.recon_offset));
                map.reset(r.w, r.h, BAND_LOG2);
                for y in 0..r.h {
                    // SAFETY: bands do not overlap, and each is one task.
                    let dst = unsafe { coeffs[slot.comp].row_mut(r.y0 + y, r.x0, r.w) };
                    let src = values.row_span(y, 0, r.w);
                    // Detail bands' pyramids tell zero rows without reading them (an LL band's
                    // may describe its residuals).
                    let nonzero = || {
                        if slot.is_ll() {
                            src.iter().any(|&v| v != 0)
                        } else {
                            pyramids[i].pair_row_any(y / 2)
                        }
                    };
                    rows[y] = match deq {
                        Some(deq) if nonzero() => {
                            deq.dequantize_row(src, dst);
                            if sparse {
                                map.mark_row(y, dst);
                            }
                            true
                        }
                        _ => {
                            // The 9/7 does not read zero rows; the sparse 5/3 reads its blocks.
                            if sparse {
                                dst.fill(0);
                            }
                            false
                        }
                    };
                }
            });
        }
        if sparse {
            let mut tasks: Vec<_> = self.comps.iter_mut().zip(self.recon.planes_mut()).collect();
            pool.for_each_mut(&mut tasks, |_, (c, dst)| {
                let map = c
                    .dwt
                    .inverse_sparse(None, &c.coeffs, c.levels, &mut c.out, &c.maps);
                if inter {
                    codec::add_residual(&c.out, dst, map);
                } else {
                    uncenter_to_u8(&c.out, dst);
                }
            });
        } else {
            // The residual goes straight from the last level onto the prediction; the three
            // components run level by level together.
            let outputs: Vec<_> = self
                .recon
                .planes_mut()
                .into_iter()
                .map(|dst| {
                    let (w, rows) = (dst.width(), dst.shared());
                    move |y: usize, res: &[i16]| {
                        // SAFETY: each row is handed over once.
                        let rec = unsafe { rows.row_mut(y, 0, w) };
                        if inter {
                            add_residual_row(res, rec);
                        } else {
                            uncenter_row(res, rec);
                        }
                    }
                })
                .collect();
            let zeros: Vec<_> = self
                .level_bands
                .iter()
                .map(|bands| codec::zero_rows(bands, &self.slots, &self.nonzero_rows))
                .collect();
            let mut jobs: Vec<InverseJob> = self
                .comps
                .iter_mut()
                .zip(&outputs)
                .zip(&zeros)
                .map(|((c, output), zero)| {
                    let Transform::W97(dwt) = &mut c.dwt else {
                        unreachable!("every component uses the stream's wavelet")
                    };
                    InverseJob {
                        dwt,
                        coeffs: &c.coeffs,
                        levels: c.levels,
                        output,
                        zero: Some(zero),
                    }
                })
                .collect();
            Dwt97::inverse_many(Some(pool), &mut jobs);
        }
    }
}

/// What P-frames add to θ and δ at fine steps: 0.01 per qp below 36. The wide dead zone that
/// pays at low rates throws away too much of the residual at high ones (0.8–1% fewer bits over
/// the full range, about 2% at qp 16–32).
/// The motion node spacing for a frame size: 32 pixels from 480 lines up, 16 below. Coarser
/// nodes cost fewer motion bits; at HD sizes a 32-pixel mesh gives 0.15–0.35 dB more than a
/// 16-pixel one at every rate measured (540p to 1080p, 0.02–0.08 bpp), while at 360p 16 pixels
/// is 4–12% smaller (DESIGN.md §10.1).
/// A source sharpened for the loop-filter fit: `c + a·(c − blur3(c))`, clamped. Encoder-only
/// (the fit target is not normative); the decoder receives ordinary filter coefficients.
fn sharpen_source(src: &Plane<u8>, strength: f64, out: &mut Plane<u8>) {
    let (w, h) = (src.width(), src.height());
    let a = (strength * 256.0).round() as i32;
    for y in 0..h {
        let row = src.row(y);
        let row_up = src.row(y.saturating_sub(1));
        let row_dn = src.row((y + 1).min(h - 1));
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let blur = (i32::from(row_up[xl])
                + 2 * i32::from(row_up[x])
                + i32::from(row_up[xr])
                + 2 * i32::from(row[xl])
                + 4 * i32::from(row[x])
                + 2 * i32::from(row[xr])
                + i32::from(row_dn[xl])
                + 2 * i32::from(row_dn[x])
                + i32::from(row_dn[xr])
                + 8)
                / 16;
            let v = i32::from(row[x]) + ((a * (i32::from(row[x]) - blur) + 128) >> 8);
            out.set(x, y, v.clamp(0, 255) as u8);
        }
    }
}

/// The finest-level step multiplier the auto policy picks for a base qp.
///
/// At qp 42 the finest luma band is dense enough that coarsening it saves several percent of
/// the bits for a fraction of a VMAF point. Above qp 46 the same tilt starts to remove detail
/// that low-rate content needs: a 60 fps aerial clip held out from tuning lost 2–3% with the
/// tilt at qp 50, while qp 42 stayed positive. Measured over five tuning clips plus four
/// held-out clips, medium and ultraheavy (2026-10-26): +0.71% hybrid per-tier against 1.0; the
/// worst cell was −0.23% on tuning and −0.13% on the held-out clips (noise).
pub fn auto_finest_tilt(qp: u8) -> f64 {
    if qp <= 46 { 1.6 } else { 1.0 }
}

pub fn default_mesh_log2(width: usize, height: usize) -> u32 {
    if width.min(height) >= 480 { 5 } else { 4 }
}

pub(crate) fn fine_ramp(qp: u8) -> f64 {
    (36.0 - f64::from(qp)).max(0.0) * 0.01
}

/// A non-reference B-frame's qp over its anchor's: its quality only shows, while the anchors
/// carry every later frame (best measured with adaptive groups: 9, with 6 for the middle frame,
/// §10.4).
pub const B_QP_OFFSET: u8 = 11;
/// The qp offset of a group's middle B-frame, a reference for the others.
pub const B_REF_QP_OFFSET: u8 = 6;

/// The ultraheavy tier codes a pair of frames as a B-frame group when the last P-frame's motion
/// section was over this share of its bytes (§10.4.1).
pub const PAIR_MOTION_SHARE: f64 = 0.3;
/// B-frames over a static camera's anchor: the prediction is nearly exact, so they carry the
/// anchor's quality instead of the usual coarse offset.
pub const B_STATIC_QP_OFFSET: u8 = 6;
/// Never code a B-frame this much finer than its anchor, whatever budget is left: its quality
/// only shows, while the anchors carry every later frame.
pub const B_MIN_QP_OFFSET: u8 = 2;
/// Below this camera speed (pixels per frame) content counts as static (see
/// [`Encoder::static_camera`]).
pub const STATIC_CAMERA_SPEED: f64 = 1.5;
/// Camera speed (pixels per frame) below which [`Encoder::encode_pair`] codes a step of 4 frames
/// as one B-frame group: a camera that does not move at all (static talking heads measure 0;
/// FPV footage passes 1 while the smoothed speed settles, where groups of 4 lose 3%).
pub const STILL_GROUP_SPEED: f64 = 0.25;
/// Fastest camera speed (pixels per frame) over the probed frame pairs below which a fixed-qp
/// 60 fps sequence gets the finer static grid ([`probe_static_mesh`]); talking heads probe at
/// 0.2–0.5.
const STILL_60_SPEED: f64 = 1.0;
/// The quality ceiling waits until the last frame was this many dB below it, so a P-frame that
/// is still repairing a coarse keyframe is not capped.
pub const QUALITY_CAP_GATE: f64 = 2.0;
/// B-frames' motion λ over the P-frames': their motion costs more than it saves sooner.
const B_MOTION_LAMBDA: f64 = 1.5;

/// Whether the next frames show a (nearly) static camera, and if so the motion grid: half the
/// spacing (32 px becomes 16 at 720p) below 50 fps. Above 50 fps the grid is kept: motion is
/// already 37% of a 600 kbit/s 60 fps stream, the opening frames can be still while the rest moves,
/// and the finer grid measured 0.2 dB worse there. Three frame pairs are probed and the slowest
/// wins, so a cut on the first pair (the samtime B clip cuts at frame 1) does not hide a still
/// camera.
/// Per root, the luma squared error of the past, next and averaged predictions, and the three
/// frame totals. `pool` splits the root rows; sums are reduced in row order, so the result does
/// not depend on the thread count.
fn root_errors(
    pool: &Pool,
    src: &Plane<u8>,
    past: &Plane<u8>,
    next: &Plane<u8>,
    grid: &Grid,
) -> (Vec<[f64; 3]>, [f64; 3]) {
    let r = tree::root_log2(grid.log2);
    let (across, down) = bframe::root_count(grid);
    let shift = grid.log2 + r;
    let cell = 1usize << shift;
    let (w, h) = (src.width(), src.height());
    let parts = pool.map(down, |row| {
        let y0 = row * cell;
        let y1 = ((row + 1) * cell).min(h);
        let mut err = vec![[0u64; 3]; across];
        for y in y0..y1 {
            let (s, p, n) = (
                src.row_span(y, 0, w),
                past.row_span(y, 0, w),
                next.row_span(y, 0, w),
            );
            for x in 0..w {
                let (s, p, n) = (i32::from(s[x]), i32::from(p[x]), i32::from(n[x]));
                let b = (p + n + 1) >> 1;
                let e = &mut err[x >> shift];
                let (d0, d1, d2) = (s - p, s - n, s - b);
                e[0] += (d0 * d0) as u64;
                e[1] += (d1 * d1) as u64;
                e[2] += (d2 * d2) as u64;
            }
        }
        err
    });
    let mut all = Vec::with_capacity(across * down);
    let mut total = [0f64; 3];
    for err in parts {
        for e in &err {
            for k in 0..3 {
                total[k] += e[k] as f64;
            }
            all.push([e[0] as f64, e[1] as f64, e[2] as f64]);
        }
    }
    (all, total)
}

/// Luma squared error of two planes, row by row on the pool; integer sums are exact.
fn frame_sse(pool: &Pool, a: &Plane<u8>, b: &Plane<u8>) -> f64 {
    let w = a.width();
    let rows = pool.map(a.height(), |y| {
        let mut sum = 0u64;
        for (&a, &b) in a.row_span(y, 0, w).iter().zip(b.row_span(y, 0, w)) {
            let d = i32::from(a) - i32::from(b);
            sum += (d * d) as u64;
        }
        sum
    });
    rows.into_iter().map(|s| s as f64).sum()
}

pub fn probe_static_mesh(cfg: &EncoderConfig, frames: &[&Frame]) -> Result<Option<u32>> {
    if frames.len() < 2 || cfg.mesh_log2 <= 3 {
        return Ok(None);
    }
    let mut estimator = GlobalEstimator::new(cfg.width, cfg.height);
    estimator.realtime = cfg.realtime || cfg.effort <= 1;
    let pool = Pool::new(1);
    let grid = Grid::new(cfg.width, cfg.height, cfg.mesh_log2);
    let (mut slowest, mut fastest) = (f64::INFINITY, 0f64);
    for pair in frames.windows(2).take(3) {
        estimator.set_reference(&pool, &pair[0].y);
        let estimate = estimator.estimate(&pool, &pair[1].y, &Affine::default());
        let global = estimate.model.to_global(&grid);
        let speed = global
            .corners
            .iter()
            .map(|c| f64::from(c[0]).hypot(f64::from(c[1])) / 8.0)
            .sum::<f64>()
            / 4.0;
        slowest = slowest.min(speed);
        fastest = fastest.max(speed);
    }
    let fps = f64::from(cfg.fps_num) / f64::from(cfg.fps_den);
    Ok((slowest < STATIC_CAMERA_SPEED).then(|| {
        // At 50 fps and above a rate-controlled stream keeps the grid: motion is a large share
        // of a low-rate 60 fps link and 16 measured 0.2 dB worse at 600 kbit/s. At a fixed qp a
        // 60 fps camera that stays still in every probed pair gains too (talking head: 3–12%
        // fewer bits on every tier); an FPV opening (forest: 0.0 then 1.4 px/frame) does not
        // qualify, so the fast tiers keep their speed there.
        if fps >= 50.0 && (cfg.rate_control.is_some() || fastest >= STILL_60_SPEED) {
            cfg.mesh_log2
        } else {
            cfg.mesh_log2 - 1
        }
    }))
}

/// The motion grid for a sequence, from a probe of its first frames (§10.4.1): half the
/// configured spacing when the probe's last P-frame field has many single-node leaves (the
/// field wants resolution) yet changes smoothly from node to node (not the chaos of foliage
/// close to the camera, where finer nodes only cost bits): `fine − 0.02·gradient > 0.11`.
/// The probe codes up to three frames, the first intra.
pub fn probe_mesh(cfg: &EncoderConfig, frames: &[&Frame]) -> Result<u32> {
    let finer = cfg.mesh_log2.wrapping_sub(1);
    if frames.len() < 2 || !MESH_LOG2.contains(&finer) {
        return Ok(cfg.mesh_log2);
    }
    let mut enc = Encoder::new(cfg.clone())?;
    let mut out = Vec::new();
    for f in frames.iter().take(3) {
        enc.encode_frame(f, &mut out)?;
    }
    let (field, grid) = (&enc.field, enc.field.grid);
    let fine = field.depth.iter().filter(|&&d| d == 0).count() as f64;
    // Mean change between horizontal neighbours, in 1/8 pel, each capped at 8 pixels.
    let mut gradient = 0.0;
    for row in field.mv.chunks_exact(grid.nx) {
        for pair in row.windows(2) {
            let d = (pair[1][0] - pair[0][0]).abs() + (pair[1][1] - pair[0][1]).abs();
            gradient += f64::from(d.min(64));
        }
    }
    let score = (fine - PROBE_GRADIENT * gradient) / grid.nodes() as f64;
    Ok(if score > PROBE_THRESHOLD {
        finer
    } else {
        cfg.mesh_log2
    })
}

/// [`probe_mesh`]'s weight of the field's gradient and its threshold (fitted to trial encodes
/// of both grids, §10.4.1).
const PROBE_GRADIENT: f64 = 0.02;
const PROBE_THRESHOLD: f64 = 0.11;

/// The loop filter's λ multiplier where its fit samples every fourth or eighth pair of rows.
const LOOP_SAMPLED_LAMBDA: f64 = 6.0;
/// P-frames between loop-filter fits in realtime mode.
const REALTIME_FIT_PERIOD: u32 = 2;

/// One band's quantization task.
struct BandTask<'a> {
    i: usize,
    values: &'a mut Plane<i32>,
    /// The component's LL residual plane, for a predicted LL band.
    ll_res: Option<&'a mut Plane<i32>>,
    /// Significance pyramid, nonzero count and magnitude bits.
    out: Option<(SigPyramid, u32, u64)>,
    guide: Option<Guide>,
}

/// Band indices, largest band first: the order parallel band tasks start in, so that no thread
/// is left with a big band at the end.
fn by_size(slots: &[BandSlot]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..slots.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(slots[i].sb.width * slots[i].sb.height));
    order
}

/// Nonzero count and summed bit lengths of the values in `r`.
fn value_stats(plane: &Plane<i32>, r: Rect) -> (u32, u64) {
    let (mut nonzero, mut bits) = (0u32, 0u64);
    for y in 0..r.h {
        for &v in plane.row_span(r.y0 + y, r.x0, r.w) {
            let m = v.unsigned_abs();
            nonzero += u32::from(m != 0);
            bits += u64::from(32 - m.leading_zeros());
        }
    }
    (nonzero, bits)
}
