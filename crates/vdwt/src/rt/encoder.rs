//! The rt encoder (see the [module](super) docs).

use crate::bitstream::{FrameHeader, FrameType, MAX_DIMENSION, SequenceHeader, put_varint};
use crate::codec::{self, BandSlot, LevelBands};
use crate::dwt::d97::{ForwardJob, InverseJob};
use crate::dwt::{Dwt97, Wavelet};
use crate::encoder::{MAX_THREADS, default_mesh_log2, fine_ramp};
use crate::entropy::{Contexts, RangeEncoder, SigPyramid, code_band};
use crate::error::{Error, Result};
use crate::loopfilter::{self, BitWriter, Coefs};
use crate::motion::gme::{Affine, GlobalEstimator};
use crate::motion::mesh::MeshParams;
use crate::motion::tree::{self, TreeContexts};
use crate::motion::{Filter, GlobalMotion, Grid, MotionField, Prediction, partition};
use crate::plane::{add_residual_row, center_row, residual_row, uncenter_row};
use crate::pool::Pool;
use crate::profile::{Stage, StageTimes, Stopwatch};
use crate::quant::{Dequantizer, Quantizer, mse_weights};
use crate::ratectl::{MagnitudeHistogram, RateControl, RateController};
use crate::{Frame, Isa, Plane};

use super::{PacketHead, RtHeader, Tiles};

/// θ and δ of intra frames and of P-frames (as the main encoder's defaults).
const ROUNDING: f64 = 0.25;
const RECON_OFFSET: f64 = 0.2;
const INTER_ROUNDING: f64 = 0.12;
const INTER_RECON_OFFSET: f64 = 0.1;
const CHROMA_STEP_SCALE: f64 = 0.7;
/// Isolated ±1 values below this many steps are zeroed (as the main encoder).
const ISOLATED: f64 = 1.5;
/// Loop-filter fit on every eighth pair of rows (as effort 0).
const FIT_STEP: usize = 8;
/// P-frames between loop-filter fits; the frames between reuse the last coefficients (sent in
/// full every frame all the same). Every other frame costs 0.02 dB, every fourth 0.2.
const FIT_PERIOD: usize = 2;
/// qp steps a frame over its cap is re-coded with, and how often.
const RETRY_QP: u8 = 3;
const RETRIES: usize = 4;
/// Tiles a frame is cut into at least, by default: the decoder's entropy stage runs one task per
/// tile, so fewer leave cores of a four-core receiver idle.
const MIN_TILES: usize = 4;
/// The motion search's λ: squared error per bit, times the squared quantizer step.
const MOTION_LAMBDA: f64 = 0.12;

/// rt encoder settings.
#[derive(Clone, Debug)]
pub struct RtConfig {
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
    /// Link rate in bits per second.
    pub bitrate: u64,
    /// Transmit-queue limit in frame durations (see [`RateControl`]).
    pub max_delay_frames: f64,
    pub threads: usize,
    pub isa: Isa,
    /// Motion root rows (64 luma pixels) per tile; 0 picks about one tile per packet at the
    /// link rate (a tile finer than a packet buys no resilience, and each costs bits), and at
    /// least four (one per decoding core).
    pub tile_roots: usize,
    /// Frames over which the whole picture is refreshed once, a root row at a time (0: never).
    pub refresh: usize,
    /// Largest packet in bytes, unless one tile alone is larger.
    pub mtu: usize,
    /// The in-loop Wiener filter (luma): +0.01 dB for 4 ms a frame on the A55, so off by default.
    pub loop_filter: bool,
    /// No local motion search: every node follows the global motion.
    pub global_only: bool,
    /// The main encoder's global estimation and mesh search (wavefront, exact prediction error)
    /// instead of the rt search: about 0.6 dB better, at twice the motion time.
    pub mesh_search: bool,
}

impl RtConfig {
    pub fn new(width: usize, height: usize, bitrate: u64) -> Self {
        Self {
            width,
            height,
            fps_num: 60,
            fps_den: 1,
            bitrate,
            max_delay_frames: 1.5,
            threads: 4,
            isa: Isa::best(),
            tile_roots: 0,
            refresh: 30,
            mtu: 1200,
            loop_filter: false,
            global_only: false,
            mesh_search: false,
        }
    }
}

/// One encoded frame: its packets, in order.
#[derive(Clone, Debug)]
pub struct RtFrame {
    pub packets: Vec<Vec<u8>>,
    pub stats: RtFrameStats,
}

#[derive(Clone, Debug, Default)]
pub struct RtFrameStats {
    pub inter: bool,
    pub qp: u8,
    /// All packets' bytes.
    pub bytes: usize,
    pub motion_bytes: usize,
    pub tiles: usize,
    /// The motion root row coded without prediction this frame, if any.
    pub refreshed: Option<usize>,
    /// Re-codings at a coarser qp to fit the cap.
    pub retries: usize,
    pub queue_bits: f64,
    pub times: StageTimes,
}

struct Component {
    levels: usize,
    dwt: Dwt97,
    /// Transform coefficients, then the dequantized ones for the reconstruction.
    coeffs: Plane<i16>,
}

pub struct RtEncoder {
    cfg: RtConfig,
    header: RtHeader,
    slots: Vec<BandSlot>,
    tiles: Tiles,
    contexts: Vec<Contexts>,
    comps: [Component; 3],
    values: Vec<Plane<i32>>,
    recon: Frame,
    pred: Frame,
    field: MotionField,
    previous_field: MotionField,
    /// The reconstruction's luma at half resolution, for the next frame's search.
    ref_half: Plane<u8>,
    /// This frame's luma at half resolution.
    cur_half: Plane<u8>,
    estimator: GlobalEstimator,
    motion_hint: Affine,
    rate: RateController,
    pool: Pool,
    frame_num: u16,
    has_frame: bool,
    p_frames: usize,
    last_qp: u8,
    /// Every band's magnitude histogram of this frame, for the rate model.
    hists: Vec<MagnitudeHistogram>,
    /// Every band's nonzero 2×2 blocks (see [`Self::quantize`]).
    sig1: Vec<Vec<bool>>,
    /// Every band's maps, kept for their buffers.
    maps: Vec<codec::BandMaps>,
    /// Every band's rows that hold a nonzero value (or may: by row pairs).
    nonzero_rows: Vec<Vec<bool>>,
    /// Per component and level, its bands (see [`LevelBands`]).
    level_bands: [Vec<LevelBands>; 3],
    /// Motion root rows in a frame.
    roots: usize,
    /// The loop filter's last fitted coefficients per plane.
    loop_coefs: Option<[Option<Coefs>; 3]>,
    /// Node rows from the top that gradual refresh has cleaned in the last frame.
    clean_rows: usize,
}

impl RtEncoder {
    pub fn new(cfg: RtConfig) -> Result<Self> {
        let fail = |what: &str| Err(Error::InvalidConfig(what.to_string()));
        if !(64..=MAX_DIMENSION).contains(&cfg.width) || !(64..=MAX_DIMENSION).contains(&cfg.height)
        {
            return fail("frame size (64 to 16384)");
        }
        if !cfg.width.is_multiple_of(2) || !cfg.height.is_multiple_of(2) {
            return fail("odd frame size");
        }
        if !(1..=MAX_THREADS).contains(&cfg.threads) {
            return fail("thread count");
        }
        if cfg.tile_roots > 255 || cfg.mtu < 64 || cfg.bitrate == 0 {
            return fail("tile height, packet size or bitrate");
        }
        if !cfg.isa.is_supported() {
            return Err(Error::Unsupported("ISA not available"));
        }
        let (luma_levels, chroma_levels) = (5, 4);
        let weights = |levels: usize, scale: f64| -> Vec<u16> {
            mse_weights(levels, Wavelet::Lifted97)
                .into_iter()
                .map(|w| (f64::from(w) * scale).round().clamp(1.0, 65535.0) as u16)
                .collect()
        };
        let seq = SequenceHeader {
            width: cfg.width,
            height: cfg.height,
            fps_num: cfg.fps_num,
            fps_den: cfg.fps_den,
            luma_levels,
            chroma_levels,
            block_log2: 6,
            mesh_log2: default_mesh_log2(cfg.width, cfg.height),
            wavelet: Wavelet::Lifted97,
            luma_weights: weights(luma_levels, 1.0),
            chroma_weights: weights(chroma_levels, CHROMA_STEP_SCALE),
            guide: true,
            continuous_cdfs: false,
            layout_coding: false,
            band_contexts: false,
            temporal_mv: false,
            temporal_motion: false,
        };
        let slots = codec::band_slots(&seq);
        let grid = Grid::new(cfg.width, cfg.height, seq.mesh_log2);
        let roots = grid.ny.div_ceil(1 << tree::root_log2(grid.log2));
        let frame_rate = f64::from(cfg.fps_num) / f64::from(cfg.fps_den.max(1));
        let mut cfg = cfg;
        if cfg.tile_roots == 0 {
            let frame_bytes = cfg.bitrate as f64 / 8.0 / frame_rate;
            // At least MIN_TILES: tiles decode in parallel, one per core.
            let tiles = ((frame_bytes / (0.8 * cfg.mtu as f64)).ceil() as usize)
                .max(MIN_TILES)
                .clamp(1, roots);
            cfg.tile_roots = roots.div_ceil(tiles);
        }
        let tiles = Tiles::new(&seq, &slots, cfg.tile_roots);
        let comps = std::array::from_fn(|comp| {
            let (w, h) = codec::component_size(&seq, comp);
            Component {
                levels: codec::component_levels(&seq, comp),
                dwt: Dwt97::fast(w, h, cfg.isa).expect("checked ISA"),
                coeffs: Plane::new(w, h),
            }
        });
        let mut estimator = GlobalEstimator::new(cfg.width, cfg.height);
        estimator.realtime = true;
        let mut rc = RateControl::new(cfg.bitrate);
        rc.max_delay_frames = cfg.max_delay_frames;
        Ok(Self {
            header: RtHeader {
                seq: seq.clone(),
                tile_roots: cfg.tile_roots,
            },
            contexts: codec::default_contexts(),
            values: codec::band_planes(&slots),
            recon: Frame::new(cfg.width, cfg.height),
            pred: Frame::new(cfg.width, cfg.height),
            field: MotionField::new(grid),
            previous_field: MotionField::new(grid),
            ref_half: Plane::new(cfg.width.div_ceil(2), cfg.height.div_ceil(2)),
            cur_half: Plane::new(cfg.width.div_ceil(2), cfg.height.div_ceil(2)),
            estimator,
            motion_hint: Affine::default(),
            rate: RateController::new(rc, frame_rate),
            pool: Pool::new(cfg.threads),
            frame_num: 0,
            has_frame: false,
            p_frames: 0,
            last_qp: 40,
            hists: Vec::new(),
            sig1: vec![Vec::new(); slots.len()],
            nonzero_rows: vec![Vec::new(); slots.len()],
            maps: (0..slots.len())
                .map(|_| codec::BandMaps::default())
                .collect(),
            level_bands: std::array::from_fn(|comp| LevelBands::of(&slots, comp)),
            roots,
            loop_coefs: None,
            clean_rows: 0,
            slots,
            tiles,
            comps,
            cfg,
        })
    }

    pub fn header(&self) -> &RtHeader {
        &self.header
    }

    /// The reconstruction of the last frame, as a decoder that received every packet has it.
    pub fn reconstruction(&self) -> &Frame {
        &self.recon
    }

    /// Changes the link rate from the next frame on.
    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.rate.set_bitrate(bitrate.max(1));
    }

    /// The root row to refresh this P-frame, if any: top to bottom, spread over `refresh`
    /// frames.
    fn refresh_row(&self) -> Option<usize> {
        if self.cfg.refresh == 0 {
            return None;
        }
        let step = (self.cfg.refresh / self.roots).max(1);
        self.p_frames
            .is_multiple_of(step)
            .then(|| (self.p_frames / step) % self.roots)
    }

    pub fn encode_frame(&mut self, frame: &Frame) -> Result<RtFrame> {
        if (frame.width(), frame.height()) != (self.cfg.width, self.cfg.height) {
            return Err(Error::InvalidConfig(
                "frame size differs from the stream's".into(),
            ));
        }
        let mut times = StageTimes::default();
        let mut watch = Stopwatch::start();
        let inter = self.has_frame;
        let frame_num = self.frame_num;
        self.frame_num = self.frame_num.wrapping_add(1);
        let mut global = GlobalMotion::default();
        let mut refreshed = None;
        let mut ref_clean = 0;
        if inter {
            let grid = self.field.grid;
            std::mem::swap(&mut self.previous_field, &mut self.field);
            let step = 2f64.powf(f64::from(self.last_qp) / 8.0);
            if self.cfg.mesh_search || self.cfg.global_only {
                let estimate = self
                    .estimator
                    .estimate(&self.pool, &frame.y, &self.motion_hint);
                self.motion_hint = estimate.model;
                global = estimate.model.to_global(&grid);
                watch.lap(&mut times, Stage::GlobalMotion);
                self.field.set_global(&global);
                if self.cfg.mesh_search {
                    let params = MeshParams {
                        lambda: 0.12 * step * step,
                        refine: true,
                        wide: false,
                        realtime: true,
                        stripe_rows: self.cfg.tile_roots,
                    };
                    partition::search(
                        &self.pool,
                        &frame.y,
                        &self.recon.y,
                        &mut self.field,
                        Some(&self.previous_field),
                        &params,
                    );
                }
            } else {
                // Global motion from the roots' vectors, searched from the last frame's.
                crate::motion::pyramid::downsample(&self.pool, &frame.y, &mut self.cur_half);
                let inputs = super::motion::Inputs {
                    pool: &self.pool,
                    cur: &frame.y,
                    cur_half: &self.cur_half,
                    reference: &self.recon.y,
                    ref_half: &self.ref_half,
                    previous: Some(&self.previous_field),
                    lambda: MOTION_LAMBDA * step * step,
                };
                self.field.set_global(&self.motion_hint.to_global(&grid));
                let roots = super::motion::search_roots(&inputs, &self.field);
                if let Some(model) = super::motion::fit_global(&roots) {
                    self.motion_hint = model;
                }
                global = self.motion_hint.to_global(&grid);
                watch.lap(&mut times, Stage::GlobalMotion);
                self.field.set_global(&global);
                super::motion::search_nodes(&inputs, &mut self.field, &roots);
            }
            ref_clean = self.clean_rows;
            if let Some(r) = self.refresh_row() {
                self.refresh(r, &global);
                refreshed = Some(r);
                let root = 1usize << tree::root_log2(self.field.grid.log2);
                // A new sweep starts clean from the top; the zone grows by the refreshed row.
                self.clean_rows = ((r + 1) * root).min(self.field.grid.ny);
            }
            // The field as the decoder will see it (offsets normalized per leaf). The rt search
            // and the refresh leave it so already.
            let pred = global_prediction(&self.field.grid, &global);
            for t in (0..self.tiles.count).filter(|_| self.cfg.mesh_search) {
                let mut sink = RangeEncoder::new();
                tree::code_rows(
                    &mut sink,
                    &mut TreeContexts::default(),
                    &mut self.field,
                    &pred,
                    self.tiles.roots(t),
                );
            }
            watch.lap(&mut times, Stage::Mesh);
            super::predict::predict(
                &self.pool,
                &self.recon,
                &mut self.pred,
                &self.field,
                [ref_clean, self.clean_rows],
            );
            watch.lap(&mut times, Stage::Prediction);
        }
        self.transform(frame, inter);
        watch.lap(&mut times, Stage::Transform);

        let plan = self.plan(inter);
        let mut qp = plan.qp;
        let mut retries = 0;
        let (mut tiles, mut motion_bytes);
        loop {
            self.quantize(inter, qp);
            watch.lap(&mut times, Stage::Quantize);
            (tiles, motion_bytes) = self.code_tiles(inter, &global);
            let bytes = tiles.iter().map(Vec::len).sum::<usize>() + self.overhead(&tiles, inter);
            if 8.0 * bytes as f64 <= plan.cap || retries == RETRIES || qp >= 255 - RETRY_QP {
                break;
            }
            qp += RETRY_QP;
            retries += 1;
        }
        let over = |tiles: &[Vec<u8>], s: &Self| {
            8.0 * (tiles.iter().map(Vec::len).sum::<usize>() + s.overhead(tiles, inter)) as f64
                > plan.cap
        };
        if over(&tiles, self) {
            // Still too large: the detail bands go, leaving motion and the LL bands.
            for ((v, sig1), slot) in self.values.iter_mut().zip(&mut self.sig1).zip(&self.slots) {
                if !slot.is_ll() {
                    v.fill(0);
                    sig1.fill(false);
                }
            }
            (tiles, motion_bytes) = self.code_tiles(inter, &global);
        }
        watch.lap(&mut times, Stage::Entropy);

        let recon_offset = if inter {
            INTER_RECON_OFFSET + fine_ramp(qp)
        } else {
            RECON_OFFSET
        };
        let recon_offset = (recon_offset * 256.0).round().min(255.0) as u8;
        self.reconstruct(inter, qp, recon_offset);
        let tile_bytes = tiles.iter().map(Vec::len).sum::<usize>() + self.overhead(&tiles, inter);
        let budget = ((plan.cap / 8.0) as usize).saturating_sub(tile_bytes);
        let loop_segment = self.loop_filter(frame, qp, budget);
        watch.lap(&mut times, Stage::Reconstruct);

        let header = FrameHeader {
            frame_type: if inter {
                FrameType::Inter
            } else {
                FrameType::Intra
            },
            frame_num,
            qp,
            recon_offset,
            parent_contexts: false,
            ll_prediction: false,
            filter: Filter::Bilinear,
            prediction: Prediction::Overlapped,
            loop_filter: loop_segment.is_some(),
            keep_reference: false,
            postfilter_strength: 0,
        };
        let clean = [self.clean_rows, ref_clean].map(|r| if inter { r } else { 0 });
        let packets = self.packetize(header, inter.then_some(global), clean, &tiles, loop_segment);
        let bytes = packets.iter().map(Vec::len).sum::<usize>();
        let modelled = self.plan_bits(inter, qp);
        self.rate.model.learn(inter, 8.0 * bytes as f64, modelled);
        self.rate.finish_frame(8.0 * bytes as f64);
        if self.cfg.mesh_search || self.cfg.global_only {
            self.estimator.set_reference(&self.pool, &frame.y);
        } else {
            crate::motion::pyramid::downsample(&self.pool, &self.recon.y, &mut self.ref_half);
        }
        self.has_frame = true;
        self.last_qp = qp;
        if inter {
            self.p_frames += 1;
        }
        watch.lap(&mut times, Stage::Entropy);
        Ok(RtFrame {
            stats: RtFrameStats {
                inter,
                qp,
                bytes,
                motion_bytes,
                tiles: self.tiles.count,
                refreshed,
                retries,
                queue_bits: self.rate.queue_bits(),
                times,
            },
            packets,
        })
    }

    /// Codes root row `r`'s nodes without prediction: every root one intra leaf.
    fn refresh(&mut self, r: usize, global: &GlobalMotion) {
        let grid = self.field.grid;
        let root = tree::root_log2(grid.log2) as u8;
        let size = 1usize << root;
        for j in (r * size).min(grid.ny)..((r + 1) * size).min(grid.ny) {
            for i in 0..grid.nx {
                let n = j * grid.nx + i;
                self.field.intra[n] = true;
                self.field.mv[n] = global.predict(&grid, i, j);
                self.field.depth[n] = root;
            }
        }
    }

    /// The residual's (or an intra frame's centred samples') transform, every plane together.
    fn transform(&mut self, frame: &Frame, inter: bool) {
        let (sources, preds) = (frame.planes(), self.pred.planes());
        let input = |comp: usize| {
            let (src, pred) = (sources[comp], preds[comp]);
            move |y: usize, row: &mut [i16]| {
                let w = src.width();
                if inter {
                    residual_row(src.row_span(y, 0, w), pred.row_span(y, 0, w), row)
                } else {
                    center_row(src.row_span(y, 0, w), row)
                }
            }
        };
        let inputs: Vec<_> = (0..3).map(input).collect();
        let mut jobs: Vec<ForwardJob> = self
            .comps
            .iter_mut()
            .zip(&inputs)
            .map(|(c, input)| ForwardJob {
                dwt: &mut c.dwt,
                input,
                levels: c.levels,
                coeffs: &mut c.coeffs,
            })
            .collect();
        Dwt97::forward_many(Some(&self.pool), &mut jobs);
    }

    fn rounding(inter: bool, qp: u8) -> f64 {
        if inter {
            INTER_ROUNDING + fine_ramp(qp)
        } else {
            ROUNDING
        }
    }

    /// Modelled frame size at `qp` from this frame's histograms, before the learned correction.
    fn plan_bits(&self, inter: bool, qp: u8) -> f64 {
        let rounding = Self::rounding(inter, qp);
        // Bands in parallel: the rate search tries about ten qps in turn.
        let bands: f64 = self
            .pool
            .map(self.slots.len(), |i| {
                let (slot, hist) = (&self.slots[i], &self.hists[i]);
                let step = slot.step_q12(&self.header.seq, qp) as f64 / 4096.0;
                let (nonzero, bits) = hist.estimate(step, rounding);
                self.rate.model.band_bits(
                    slot.coding_class(inter),
                    slot.sb.width * slot.sb.height,
                    nonzero,
                    bits,
                )
            })
            .into_iter()
            .sum();
        8.0 * (32 + 2 * self.tiles.count) as f64 + bands
    }

    /// The link-following qp for this frame.
    fn plan(&mut self, inter: bool) -> crate::ratectl::FramePlan {
        let (slots, comps) = (&self.slots, &self.comps);
        self.hists = self.pool.map(slots.len(), |i| {
            MagnitudeHistogram::sparse(&comps[slots[i].comp].coeffs, slots[i].rect())
        });
        self.rate
            .plan(inter, !inter, |qp| self.plan_bits(inter, qp))
    }

    /// Every band's values at `qp`, one band per task, with its map of nonzero 2×2 blocks for
    /// the tiles' significance pyramids, made while each row is fresh.
    fn quantize(&mut self, inter: bool, qp: u8) {
        let (seq, slots, comps) = (&self.header.seq, &self.slots, &self.comps);
        let rounding = Self::rounding(inter, qp);
        let isolated = (ISOLATED * 4096.0).round() as u64;
        let mut work: Vec<_> = self
            .values
            .iter_mut()
            .zip(&mut self.sig1)
            .zip(&mut self.maps)
            .collect();
        self.pool
            .for_each_mut(&mut work, |i, ((values, sig1), maps)| {
                let slot = &slots[i];
                let step = slot.step_q12(seq, qp);
                let quant = Quantizer::new(step, rounding);
                let (r, coeffs) = (slot.rect(), &comps[slot.comp].coeffs);
                // Last frame's buffers (allocating them anew faulted pages in every frame).
                maps.reset(r);
                let mut nonzero = 0;
                for y in 0..r.h {
                    let row = values.row_span_mut(y, 0, r.w);
                    let (n, _) =
                        quant.quantize_row_stats(coeffs.row_span(r.y0 + y, r.x0, r.w), row);
                    if n > 0 {
                        nonzero += n;
                        maps.add_row(y, row);
                    }
                }
                if !slot.is_ll() && nonzero > 0 {
                    maps.zero_isolated(values, coeffs, r, step, isolated, true);
                }
                std::mem::swap(*sig1, &mut maps.sig1);
            });
    }

    /// Every tile's segment, and the motion's share of the bytes (estimated from the coder's
    /// position): first every tile's band pyramids, then the tiles, each its motion and bands,
    /// all in parallel. The field must be as the decoder will see it (see
    /// [`Self::encode_frame`]).
    fn code_tiles(&mut self, inter: bool, global: &GlobalMotion) -> (Vec<Vec<u8>>, usize) {
        let n = self.tiles.count;
        let (slots, values, tiles, sig1) = (&self.slots, &self.values, &self.tiles, &self.sig1);
        let (contexts, block_log2) = (&self.contexts, self.header.seq.block_log2);
        let nb = slots.len();
        let pyramids = self.pool.map(n * nb, |k| {
            let (t, i) = (k / nb, k % nb);
            let rect = tiles.band_rect(slots, i, t);
            let w1 = rect.w.div_ceil(2);
            let rows = rect.y0 / 2..(rect.y0 + rect.h).div_ceil(2);
            let nz = &sig1[i][rows.start * w1..rows.end * w1];
            if rect.h == 0 || !nz.iter().any(|&b| b) {
                SigPyramid::empty()
            } else {
                debug_assert!(rect.y0.is_multiple_of(2) && rect.x0 == 0);
                let p = SigPyramid::from_level1(nz.to_vec(), w1, block_log2);
                debug_assert_eq!(p, SigPyramid::build(&values[i], rect, block_log2));
                p
            }
        });
        let pred = inter.then(|| global_prediction(&self.field.grid, global));
        let field = &self.field;
        let coded = self.pool.map(n, |t| {
            let mut enc = RangeEncoder::new();
            let mut motion = 0;
            if let Some(pred) = &pred {
                // Coding leaves a normalized field as it was: a copy to satisfy the coder.
                let mut field = field.clone();
                tree::code_rows(
                    &mut enc,
                    &mut TreeContexts::default(),
                    &mut field,
                    pred,
                    tiles.roots(t),
                );
                motion = enc.bytes_so_far();
            }
            for (i, slot) in slots.iter().enumerate() {
                let rect = tiles.band_rect(slots, i, t);
                if rect.h == 0 {
                    continue;
                }
                let own = &pyramids[t * nb + i];
                let mut ctx = contexts[slot.coding_class(inter)].clone();
                code_band(
                    &mut enc,
                    &mut ctx,
                    &values[i],
                    rect,
                    None,
                    Some(own),
                    None,
                    block_log2,
                    slot.transposed(),
                );
            }
            (enc.finish(), motion)
        });
        let motion_bytes = coded.iter().map(|c| c.1).sum();
        (coded.into_iter().map(|c| c.0).collect(), motion_bytes)
    }

    /// Bytes besides the tile segments: packet heads and length prefixes (estimated).
    fn overhead(&self, tiles: &[Vec<u8>], inter: bool) -> usize {
        let total: usize = tiles.iter().map(Vec::len).sum();
        let packets = total.div_ceil(self.cfg.mtu).max(1);
        packets * (if inter { 24 } else { 12 }) + 2 * tiles.len() + 2
    }

    /// Dequantizes every band into the coefficients and inverts the transform onto the
    /// prediction (or mid-grey).
    fn reconstruct(&mut self, inter: bool, qp: u8, recon_offset: u8) {
        if inter {
            std::mem::swap(&mut self.recon, &mut self.pred);
        }
        let (seq, slots, values, sig1) = (&self.header.seq, &self.slots, &self.values, &self.sig1);
        // Rows whose 2×2 blocks are all zero are left alone: the transform reads them as zero.
        self.pool.for_each_mut(&mut self.nonzero_rows, |i, rows| {
            let w1 = slots[i].sb.width.div_ceil(2);
            rows.clear();
            rows.extend(
                (0..slots[i].sb.height).map(|y| sig1[i][(y / 2) * w1..][..w1].contains(&true)),
            );
        });
        let nonzero_rows = &self.nonzero_rows;
        {
            let coeffs: Vec<_> = self.comps.iter_mut().map(|c| c.coeffs.shared()).collect();
            self.pool.run(slots.len(), &|i| {
                let (slot, r) = (&slots[i], slots[i].rect());
                let deq = Dequantizer::new(slot.step_q12(seq, qp), recon_offset);
                for y in (0..r.h).filter(|&y| nonzero_rows[i][y]) {
                    // SAFETY: bands do not overlap, and each is one task.
                    let dst = unsafe { coeffs[slot.comp].row_mut(r.y0 + y, r.x0, r.w) };
                    deq.dequantize_row(values[i].row_span(y, 0, r.w), dst);
                }
            });
        }
        let bands = &self.level_bands;
        let zeros: Vec<_> = (0..3)
            .map(|comp| {
                move |l: usize, y: usize, right: bool| {
                    bands[comp][l]
                        .slot(y, right)
                        .is_some_and(|i| !nonzero_rows[i][y - slots[i].sb.y0])
                }
            })
            .collect();
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
        let mut jobs: Vec<InverseJob> = self
            .comps
            .iter_mut()
            .zip(&outputs)
            .zip(&zeros)
            .map(|((c, output), zero)| InverseJob {
                dwt: &mut c.dwt,
                coeffs: &c.coeffs,
                levels: c.levels,
                output,
                zero: Some(zero),
            })
            .collect();
        Dwt97::inverse_many(Some(&self.pool), &mut jobs);
    }

    /// Fits and applies the loop filter where it pays; returns its segment (absolute
    /// coefficients), if any, within `budget` bytes.
    fn loop_filter(&mut self, frame: &Frame, qp: u8, budget: usize) -> Option<Vec<u8>> {
        if !self.cfg.loop_filter {
            return None;
        }
        let step = 2f64.powf(f64::from(qp) / 8.0);
        let lambda = 0.12 * step * step;
        let none = Coefs::default();
        let chosen = match self.loop_coefs {
            Some(c) if self.has_frame && !self.p_frames.is_multiple_of(FIT_PERIOD) => c,
            _ => {
                // Luma only: filtering chroma gained nothing measurable (YUV PSNR ±0.01 dB).
                let mut chosen: [Option<Coefs>; 3] = [None; 3];
                let fit = loopfilter::fit(&self.pool, &frame.y, &self.recon.y, FIT_STEP);
                let bits = loopfilter::coded_bits(&fit.coefs, &none);
                if fit.coefs != none && fit.saved > lambda * f64::from(bits + 1) {
                    chosen[0] = Some(fit.coefs);
                }
                self.loop_coefs = Some(chosen);
                chosen
            }
        };
        let mut w = BitWriter::default();
        for coefs in &chosen {
            w.bit(coefs.is_some());
            if let Some(coefs) = coefs {
                loopfilter::write_coefs(&mut w, coefs, &none);
            }
        }
        if chosen.iter().all(Option::is_none) || w.bytes.len() + 2 > budget {
            return None;
        }
        let planes = self
            .recon
            .planes_mut()
            .into_iter()
            .zip(self.pred.planes_mut());
        for ((rec, tmp), coefs) in planes.zip(&chosen) {
            if let Some(coefs) = coefs {
                loopfilter::apply(&self.pool, rec, tmp, coefs);
                std::mem::swap(rec, tmp);
            }
        }
        Some(w.bytes)
    }

    /// Groups the tiles into packets of at most `mtu` bytes, each with the frame's head.
    fn packetize(
        &self,
        header: FrameHeader,
        global: Option<GlobalMotion>,
        [clean, ref_clean]: [usize; 2],
        tiles: &[Vec<u8>],
        loop_segment: Option<Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        let head = |index, first_tile, count| PacketHead {
            header,
            global,
            index,
            count: 0,
            first_tile,
            tiles: count,
            clean,
            ref_clean,
        };
        let head_len = {
            let mut b = Vec::new();
            head(0, 0, 0).write(&mut b);
            b.len() + 4
        };
        // Tile ranges first, then the packets (the count goes into every head).
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut size = head_len + loop_segment.as_ref().map_or(0, |s| s.len() + 2);
        let mut first = 0;
        for (t, tile) in tiles.iter().enumerate() {
            let cost = tile.len() + 3;
            if t > first && size + cost > self.cfg.mtu {
                groups.push((first, t - first));
                first = t;
                size = head_len;
            }
            size += cost;
        }
        groups.push((first, tiles.len() - first));
        let count = groups.len();
        groups
            .iter()
            .enumerate()
            .map(|(index, &(first, n))| {
                let mut p = Vec::with_capacity(self.cfg.mtu);
                PacketHead {
                    count,
                    ..head(index, first, n)
                }
                .write(&mut p);
                if index == 0
                    && let Some(seg) = &loop_segment
                {
                    put_varint(&mut p, seg.len() as u64);
                    p.extend_from_slice(seg);
                }
                for tile in &tiles[first..first + n] {
                    put_varint(&mut p, tile.len() as u64);
                    p.extend_from_slice(tile);
                }
                p
            })
            .collect()
    }
}

/// The global motion's vector at every node.
pub(crate) fn global_prediction(grid: &Grid, global: &GlobalMotion) -> Vec<[i32; 2]> {
    (0..grid.ny)
        .flat_map(|j| (0..grid.nx).map(move |i| (i, j)))
        .map(|(i, j)| global.predict(grid, i, j))
        .collect()
}
