//! The decoder: intra frames, P-frames and Repeat frames.
//!
//! Malformed input yields an [`Error`] or a garbage picture, never a panic.

use crate::bframe::{self, BiMode};
use crate::bitstream::{ByteReader, FrameHeader, FrameHeaderState, FrameType, SequenceHeader};
use crate::codec::{self, BandSlot, LevelBands};
use crate::dwt::d97::{ForwardJob, InverseJob};
use crate::dwt::{BAND_LOG2, BlockMap, Dwt97, Transform, Wavelet};
use crate::entropy::{
    Contexts, Parent, RangeDecoder, SigPyramid, TrackedRows, code_band, decode_layout,
};
use crate::error::{Error, Result};
use crate::loopfilter::{self, BitReader, Coefs};
use crate::motion::tree::{self, TreeContexts};
use crate::motion::warp::warp_planes;
use crate::motion::{GlobalMotion, Grid, MotionCoder, MotionField};
use crate::plane::{add_residual_row, center_row, center_u8, uncenter_row, uncenter_to_u8};
use crate::pool::Pool;
use crate::profile::{Stage, StageTimes, Stopwatch};
use crate::quant::Dequantizer;
use crate::{Frame, Isa, Plane};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{OnceLock, RwLock};

/// Per-component buffers.
struct Component {
    levels: usize,
    /// Each component has its own transform engine, for its own thread.
    dwt: Transform,
    /// LL residuals when LL prediction is on.
    ll_res: Plane<i32>,
    /// Nonzero 4×4 blocks of every band, for the sparse inverse transform.
    maps: Vec<BlockMap>,
    coeffs: Plane<i16>,
    out: Plane<i16>,
    /// P-frames: the prediction's transform, for the band coder's guide.
    pred_coeffs: Plane<i16>,
}

/// A band's values, with its component's LL residual plane for a predicted LL band, and its
/// row flags.
type BandBuffers<'a> = (
    &'a mut Plane<i32>,
    Option<&'a mut Plane<i32>>,
    &'a mut [bool],
);

pub struct Decoder {
    seq: SequenceHeader,
    isa: Isa,
    slots: Vec<BandSlot>,
    contexts: Vec<Contexts>,
    /// Per band and coding class, entropy contexts carried across fixed-qp frames.
    band_contexts: Vec<std::sync::Mutex<[Contexts; 2]>>,
    comps: [Component; 3],
    /// Every band's quantized values, in coding order.
    values: Vec<Plane<i32>>,
    pyramids: Vec<SigPyramid>,
    /// The last decoded picture: the output, and the next P-frame's reference.
    frame: Frame,
    /// Whether `frame` holds a picture to repeat or predict from.
    has_frame: bool,
    /// Motion-compensation scratch.
    pred: Frame,
    field: MotionField,
    /// Global motion of the last P-frame (zero after an intra frame).
    global: GlobalMotion,
    /// Every plane's last loop-filter coefficients since the last intra frame.
    loop_coefs: [Coefs; 3],
    motion_coder: MotionCoder,
    /// B-frame motion contexts, per direction, carried on fixed-qp streams.
    b_tree: [TreeContexts; 2],
    /// Stage times of the last frame.
    times: StageTimes,
    pool: Pool,
    /// Bands that are some band's parent: only their significance pyramids are ever read.
    has_child: Vec<bool>,
    /// B-frames: the picture kept when an anchor replaced it (see `FrameHeader::keep_reference`),
    /// the B-frame's output, and the second direction's prediction. Made on first use.
    past: Option<Frame>,
    b_out: Option<Frame>,
    pred2: Option<Frame>,
    /// The B-frame kept as a reference inside a group, and its position.
    bref: Option<(Frame, i32)>,
    /// Whether the last anchor (non-B frame) was a P-frame.
    anchor_inter: bool,
    /// The last decoded frame's header.
    last_header: Option<FrameHeader>,
    /// Bands whose value plane may hold nonzero values from an earlier frame.
    dirty: Vec<bool>,
    /// Per band, the rows that received a value this frame: the only ones to dequantize.
    rows: Vec<Vec<bool>>,
    /// Per band, the rows of the component's coefficients left nonzero by the last frame.
    coeff_rows: Vec<Vec<bool>>,
    /// Per component and level, its bands (see [`LevelBands`]).
    level_bands: [Vec<LevelBands>; 3],
}

impl Decoder {
    pub fn new(seq: SequenceHeader) -> Result<Self> {
        Self::with_isa(seq, Isa::best())
    }

    /// A decoder using a specific transform kernel family; every ISA gives the same output.
    pub fn with_isa(seq: SequenceHeader, isa: Isa) -> Result<Self> {
        Self::with_threads(seq, isa, 1)
    }

    /// A decoder running on `threads` threads (the caller's included); the output does not
    /// depend on it.
    pub fn with_threads(seq: SequenceHeader, isa: Isa, threads: usize) -> Result<Self> {
        if !(1..=crate::encoder::MAX_THREADS).contains(&threads) {
            return Err(Error::Unsupported("thread count"));
        }
        if seq.luma_weights.len() != 1 + 3 * seq.luma_levels
            || seq.chroma_weights.len() != 1 + 3 * seq.chroma_levels
        {
            return Err(Error::InvalidData("step weight count"));
        }
        if !isa.is_supported() {
            return Err(Error::Unsupported("ISA not available"));
        }
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
                dwt: Transform::with_isa(seq.wavelet, w, h, isa).expect("supported ISA"),
                ll_res: Plane::new(ll.width, ll.height),
                maps: codec::band_maps(&seq, comp),
                coeffs: Plane::new(w, h),
                out: Plane::new(w, h),
                pred_coeffs: Plane::new(w, h),
            }
        });
        let grid = Grid::new(seq.width, seq.height, seq.mesh_log2);
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
            isa,
            contexts,
            band_contexts,
            values: codec::band_planes(&slots),
            pyramids: Vec::with_capacity(slots.len()),
            frame: Frame::new(seq.width, seq.height),
            has_frame: false,
            pred: Frame::new(seq.width, seq.height),
            field: MotionField::new(grid),
            global: GlobalMotion::default(),
            loop_coefs: Default::default(),
            motion_coder: {
                let mut m = MotionCoder::new(seq.continuous_cdfs);
                m.set_temporal(seq.temporal_motion);
                m
            },
            b_tree: std::array::from_fn(|_| TreeContexts::default()),
            times: StageTimes::default(),
            pool: Pool::new(threads),
            has_child: codec::has_child(&slots),
            past: None,
            b_out: None,
            pred2: None,
            bref: None,
            anchor_inter: false,
            last_header: None,
            dirty: vec![false; slots.len()],
            rows: slots.iter().map(|s| vec![false; s.sb.height]).collect(),
            coeff_rows: slots.iter().map(|s| vec![true; s.sb.height]).collect(),
            level_bands: std::array::from_fn(|comp| LevelBands::of(&slots, comp)),
            seq,
            slots,
            comps,
        })
    }

    /// Decodes every band into `self.values` and builds `self.pyramids`, one band per task. A band
    /// waits only for its parent (always earlier in coding order, so claimed earlier).
    fn decode_bands(&mut self, header: &FrameHeader, segments: &[Option<&[u8]>]) {
        let inter = header.frame_type.predicted();
        let (slots, contexts, band_contexts, band_flag, block_log2) = (
            &self.slots,
            &self.contexts,
            &self.band_contexts,
            self.seq.band_contexts,
            self.seq.block_log2,
        );
        let has_child = &self.has_child;
        // Planes must start at zero; most are already.
        for (v, dirty) in self.values.iter_mut().zip(&mut self.dirty) {
            if *dirty {
                v.fill(0);
            }
            *dirty = false;
        }
        for ((dirty, slot), segment) in self.dirty.iter_mut().zip(slots).zip(segments) {
            *dirty = segment.is_some() || (slot.is_ll() && header.ll_prediction);
        }
        let seq = &self.seq;
        let mut pred_coeffs = Vec::new();
        let mut ll_res: Vec<Option<&mut Plane<i32>>> = self
            .comps
            .iter_mut()
            .map(|c| {
                c.ll_res.fill(0);
                pred_coeffs.push(&c.pred_coeffs);
                Some(&mut c.ll_res)
            })
            .collect();
        let cells: Vec<RwLock<BandBuffers<'_>>> = self
            .values
            .iter_mut()
            .zip(&mut self.rows)
            .zip(slots)
            .map(|((values, rows), slot)| {
                let ll = if slot.is_ll() && header.ll_prediction {
                    ll_res[slot.comp].take()
                } else {
                    None
                };
                RwLock::new((values, ll, rows.as_mut_slice()))
            })
            .collect();
        let pyramids: Vec<OnceLock<SigPyramid>> = slots.iter().map(|_| OnceLock::new()).collect();
        let failed = AtomicBool::new(false);
        self.pool.run(slots.len(), &|i| {
            // A task that unwinds tells the tasks waiting for it.
            struct Failed<'a>(&'a AtomicBool);
            impl Drop for Failed<'_> {
                fn drop(&mut self) {
                    if std::thread::panicking() {
                        self.0.store(true, Ordering::Release);
                    }
                }
            }
            let _failed = Failed(&failed);
            let slot = &slots[i];
            let rect = slot.local_rect();
            let parent = slot.parent.filter(|_| header.parent_contexts).map(|p| {
                let mut spins = 0u32;
                let sig = loop {
                    if let Some(sig) = pyramids[p].get() {
                        break sig;
                    }
                    assert!(!failed.load(Ordering::Acquire), "a band task failed");
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                };
                (p, sig, cells[p].read().unwrap_or_else(|e| e.into_inner()))
            });
            let mut cell = cells[i].write().unwrap_or_else(|e| e.into_inner());
            let (values, ll, rows) = &mut *cell;
            rows.fill(ll.is_some());
            if let Some(segment) = segments[i] {
                let parent = parent.as_ref().map(|(p, sig, values)| Parent {
                    values: values.0,
                    band: slots[*p].local_rect(),
                    sig,
                });
                let class = usize::from(inter);
                let mut ctx = if band_flag {
                    band_contexts[i].lock().expect("band contexts")[class].clone()
                } else {
                    contexts[slot.coding_class(inter)].clone()
                };
                let guide = inter
                    .then(|| codec::band_guide(slot, seq, header.qp, pred_coeffs[slot.comp]))
                    .flatten();
                let mut decoder = RangeDecoder::new(segment);
                match ll.as_deref_mut() {
                    // A predicted LL band: every row is rebuilt from the residuals below.
                    Some(res) => code_band(
                        &mut decoder,
                        &mut ctx,
                        res,
                        rect,
                        parent,
                        None,
                        guide.as_ref(),
                        block_log2,
                        slot.transposed(),
                    ),
                    None => code_band(
                        &mut decoder,
                        &mut ctx,
                        TrackedRows {
                            plane: values,
                            rows,
                        },
                        rect,
                        parent,
                        None,
                        guide.as_ref(),
                        block_log2,
                        slot.transposed(),
                    ),
                }
                if band_flag {
                    band_contexts[i].lock().expect("band contexts")[class] = ctx;
                }
            }
            let plane: &Plane<i32> = match ll.as_deref() {
                Some(res) => res,
                None => values,
            };
            // Only a child band reads a pyramid, and an absent band's is empty.
            let pyramid = if has_child[i] && header.parent_contexts && segments[i].is_some() {
                SigPyramid::build(plane, rect, block_log2)
            } else {
                SigPyramid::empty()
            };
            if let Some(res) = ll {
                codec::ll_reconstruct(res, rect, values);
            }
            let _ = pyramids[i].set(pyramid);
        });
        self.pyramids.clear();
        self.pyramids.extend(
            pyramids
                .into_iter()
                .map(|p| p.into_inner().expect("every band ran")),
        );
    }

    pub fn sequence_header(&self) -> &SequenceHeader {
        &self.seq
    }

    /// The motion field of the most recent P-frame.
    pub fn motion_field(&self) -> &MotionField {
        &self.field
    }

    /// The most recent frame's header. After one with `keep_reference` a B-frame follows that
    /// shows before it.
    pub fn last_header(&self) -> Option<&FrameHeader> {
        self.last_header.as_ref()
    }

    /// Decoding time per stage of the most recent frame.
    pub fn times(&self) -> &StageTimes {
        &self.times
    }

    /// A B-frame's prediction into `pred`, from its extension and motion sections.
    /// Returns the B-frame's position in its group.
    fn predict_b(&mut self, header: &FrameHeader, r: &mut ByteReader<'_>) -> Result<u8> {
        let bad = || Error::InvalidData("B-frame");
        let ext = r.bytes(2)?;
        let packed = u16::from_le_bytes([ext[0], ext[1]]);
        let mode = BiMode::from_code((packed & 7) as u8).ok_or_else(bad)?;
        let [k, n, past_pos, next_pos] =
            [3, 6, 9, 12].map(|shift| i32::from((packed >> shift) & 7));
        if !(past_pos < k && k < next_pos && next_pos <= n) {
            return Err(bad());
        }
        // A reference position: the kept picture, the current one, or the group's kept B-frame.
        let reference = |pos: i32| -> Option<&Frame> {
            if pos == 0 {
                self.past.as_ref()
            } else if pos == n {
                Some(&self.frame)
            } else {
                self.bref
                    .as_ref()
                    .filter(|(_, p)| *p == pos)
                    .map(|(f, _)| f)
            }
        };
        let refs = [
            reference(past_pos).ok_or_else(bad)?,
            reference(next_pos).ok_or_else(bad)?,
        ];
        let mut pred2 = self.pred2.take().unwrap_or_else(|| self.pred.clone());
        let mut first = true;
        let mut b_tree = if self.seq.continuous_cdfs {
            self.b_tree.clone()
        } else {
            std::array::from_fn(|_| TreeContexts::default())
        };
        let mut blend_tree: [TreeContexts; 2] = std::array::from_fn(|_| TreeContexts::default());
        let mut sections = [None, None];
        if mode.searched() {
            for (dir, used) in mode.directions().into_iter().enumerate() {
                if used {
                    sections[dir] = Some(r.chunk()?);
                }
            }
        }
        let blend_modes = if mode == BiMode::Blend {
            let (across, down) = bframe::root_count(&self.field.grid);
            let mut modes = vec![0u8; across * down];
            bframe::code_root_modes(&mut RangeDecoder::new(r.chunk()?), &mut modes, across);
            Some(modes)
        } else {
            None
        };
        for (dir, used) in mode.directions().into_iter().enumerate() {
            if !used {
                continue;
            }
            let num = k - if dir == 0 { past_pos } else { next_pos };
            let pred = bframe::scaled_vectors(&self.field, &self.global, self.anchor_inter, num, n);
            let field = if mode.searched() {
                let mut field = MotionField::new(self.field.grid);
                let ctx = if mode == BiMode::Blend {
                    &mut blend_tree[dir]
                } else {
                    &mut b_tree[dir]
                };
                tree::code(
                    &mut RangeDecoder::new(sections[dir].expect("read above")),
                    ctx,
                    &mut field,
                    &pred,
                );
                field
            } else {
                bframe::direct_field(&self.field, &pred)
            };
            let out = if first { &mut self.pred } else { &mut pred2 };
            warp_planes(
                &self.pool,
                &refs[dir].planes(),
                out.planes_mut().into(),
                &field,
                header.prediction,
                header.filter,
                self.isa,
                1,
            );
            if !first {
                match &blend_modes {
                    Some(modes) => bframe::blend(&mut self.pred, &pred2, &self.field.grid, modes),
                    None => bframe::average_into(&mut self.pred, &pred2),
                }
            }
            first = false;
        }
        if self.seq.continuous_cdfs {
            self.b_tree = if mode == BiMode::Blend {
                std::array::from_fn(|_| TreeContexts::default())
            } else {
                b_tree
            };
        }
        self.pred2 = Some(pred2);
        Ok(k as u8)
    }

    /// Decodes one frame payload (without its length prefix; see
    /// [`split_stream`](crate::bitstream::split_stream)).
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

    pub fn decode_frame(&mut self, payload: &[u8]) -> Result<&Frame> {
        self.times = StageTimes::default();
        let mut watch = Stopwatch::start();
        let mut r = ByteReader::new(payload);
        let prev = self.last_header.as_ref().map(|h| FrameHeaderState {
            qp: h.qp,
            recon_offset: h.recon_offset,
            postfilter_strength: h.postfilter_strength,
        });
        let header = FrameHeader::parse_with(&mut r, prev.as_ref())?;
        if header.frame_type != FrameType::Repeat {
            self.last_header = Some(header);
        }
        let bi = header.frame_type == FrameType::Bi;
        let inter = match header.frame_type {
            FrameType::Repeat | FrameType::Inter | FrameType::Bi if !self.has_frame => {
                return Err(Error::InvalidData("no previous picture"));
            }
            FrameType::Bi if self.past.is_none() || header.loop_filter => {
                return Err(Error::InvalidData("B-frame"));
            }
            FrameType::Repeat => return Ok(&self.frame),
            FrameType::Inter | FrameType::Bi => true,
            FrameType::Intra => false,
        };
        if !inter && header.frame_type == FrameType::Intra {
            self.motion_coder.reset();
            self.reset_band_contexts();
        }
        if header.keep_reference && self.has_frame && !bi {
            // The picture this frame replaces stays the next B-frame's past anchor.
            match &mut self.past {
                Some(past) => past.clone_from(&self.frame),
                None => self.past = Some(self.frame.clone()),
            }
        }
        // Committed only once the whole frame has decoded.
        let mut global = GlobalMotion::default();
        let mut r_keep_pos = 0;
        if bi {
            r_keep_pos = self.predict_b(&header, &mut r)?;
            watch.lap(&mut self.times, Stage::Entropy);
        } else if inter {
            let motion = r.chunk()?;
            self.motion_coder
                .decode(motion, &self.global, &mut global, &mut self.field);
            watch.lap(&mut self.times, Stage::Entropy);
            warp_planes(
                &self.pool,
                &self.frame.planes(),
                self.pred.planes_mut().into(),
                &self.field,
                header.prediction,
                header.filter,
                self.isa,
                1,
            );
            watch.lap(&mut self.times, Stage::Prediction);
        }
        if inter && self.seq.guide {
            {
                let preds = self.pred.planes();
                self.pool.for_each_mut(&mut self.comps, |comp, c| {
                    if !c.dwt.uses_pool() {
                        // `out` is free until the reconstruction.
                        center_u8(preds[comp], &mut c.out);
                        c.dwt
                            .forward(None, &mut c.out, c.levels, &mut c.pred_coeffs);
                    }
                });
                // The 9/7: the three components level by level together.
                let inputs: Vec<_> = preds
                    .iter()
                    .map(|&pred| {
                        move |y: usize, row: &mut [i16]| {
                            center_row(pred.row_span(y, 0, pred.width()), row)
                        }
                    })
                    .collect();
                let mut jobs: Vec<ForwardJob> = self
                    .comps
                    .iter_mut()
                    .zip(&inputs)
                    .filter_map(|(c, input)| match &mut c.dwt {
                        Transform::W97(dwt) => Some(ForwardJob {
                            dwt,
                            input,
                            levels: c.levels,
                            coeffs: &mut c.pred_coeffs,
                        }),
                        Transform::W53(_) => None,
                    })
                    .collect();
                Dwt97::forward_many(Some(&self.pool), &mut jobs);
            }
            watch.lap(&mut self.times, Stage::Prediction);
        }
        // Every band's segment first: a corrupt length fails before anything decodes.
        let mut segments = Vec::with_capacity(self.slots.len());
        if self.seq.layout_coding {
            let classes: Vec<usize> = self.slots.iter().map(|s| s.class).collect();
            let lengths = decode_layout(r.chunk()?, &classes)?;
            for length in lengths {
                segments.push(if length == 0 {
                    None
                } else {
                    Some(r.bytes(length)?)
                });
            }
        } else {
            let presence = r.bytes(self.slots.len().div_ceil(8))?;
            for i in 0..self.slots.len() {
                segments.push(if presence[i / 8] & (1 << (i % 8)) != 0 {
                    Some(r.chunk()?)
                } else {
                    None
                });
            }
        }
        // Loop-filter coefficients per plane, from the plane's previous ones (none after an
        // intra frame).
        let previous = if inter {
            self.loop_coefs
        } else {
            Default::default()
        };
        let mut loop_coefs = [None; 3];
        if header.loop_filter {
            let mut bits = BitReader::new(r.chunk()?);
            for (coefs, previous) in loop_coefs.iter_mut().zip(&previous) {
                let bad = || Error::InvalidData("loop filter");
                if bits.bit().ok_or_else(bad)? {
                    *coefs = Some(loopfilter::read_coefs(&mut bits, previous).ok_or_else(bad)?);
                }
            }
        }
        if self.seq.layout_coding && r.remaining() != 0 {
            return Err(Error::InvalidData("trailing frame data"));
        }
        self.decode_bands(&header, &segments);
        watch.lap(&mut self.times, Stage::Entropy);

        if bi {
            // The B-frame reconstructs in place of the picture, which it leaves untouched.
            std::mem::swap(
                &mut self.frame,
                self.b_out.get_or_insert_with(|| self.pred.clone()),
            );
        }
        if inter {
            std::mem::swap(&mut self.frame, &mut self.pred);
        }
        let (seq, slots, pool) = (&self.seq, &self.slots, &self.pool);
        // Every band into its component's coefficients, one band per task. Its values are
        // cleared for the next frame on the way.
        {
            let sparse = self.comps.iter().all(|c| !c.dwt.uses_pool());
            let mut coeffs = Vec::with_capacity(3);
            let mut maps: Vec<Vec<Option<&mut BlockMap>>> = Vec::with_capacity(3);
            for c in self.comps.iter_mut() {
                coeffs.push(c.coeffs.shared());
                maps.push(c.maps.iter_mut().map(Some).collect());
            }
            let dirty = &self.dirty;
            let mut items: Vec<_> = self
                .values
                .iter_mut()
                .zip(&self.rows)
                .zip(&mut self.coeff_rows)
                .zip(slots)
                .map(|(((v, rows), coeff_rows), slot)| {
                    let map = maps[slot.comp][slot.weight_index].take();
                    (v, rows, coeff_rows, map.expect("one band per map"))
                })
                .collect();
            pool.for_each_mut(&mut items, |i, (values, rows, coeff_rows, map)| {
                let (slot, r) = (&slots[i], slots[i].rect());
                let deq = segments[i]
                    .map(|_| Dequantizer::new(slot.step_q12(seq, header.qp), header.recon_offset));
                map.reset(r.w, r.h, BAND_LOG2);
                for y in 0..r.h {
                    // Only rows that received values can be nonzero, and only rows that were
                    // nonzero need clearing.
                    let (now, before) = (rows[y], coeff_rows[y]);
                    coeff_rows[y] = now;
                    if !now && !before {
                        continue;
                    }
                    // SAFETY: bands do not overlap, and each is one task.
                    let dst = unsafe { coeffs[slot.comp].row_mut(r.y0 + y, r.x0, r.w) };
                    let src = values.row_span_mut(y, 0, r.w);
                    match deq {
                        Some(deq) if now => {
                            deq.dequantize_row(src, dst);
                            if sparse {
                                map.mark_row(y, dst);
                            }
                            src.fill(0);
                        }
                        _ => {
                            // Absent bands are all zero.
                            dst.fill(0);
                            if dirty[i] {
                                src.fill(0);
                            }
                        }
                    }
                }
            });
        }
        self.dirty.fill(false);
        if self.seq.wavelet == Wavelet::Lifted97 {
            // The residual goes straight from the last level onto the prediction; the three
            // components run level by level together.
            let outputs: Vec<_> = self
                .frame
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
            // Band rows that received no value are zero: the transform does not read them.
            let zeros: Vec<_> = self
                .level_bands
                .iter()
                .map(|bands| codec::zero_rows(bands, &self.slots, &self.coeff_rows))
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
        } else {
            let mut tasks: Vec<_> = self.comps.iter_mut().zip(self.frame.planes_mut()).collect();
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
        }
        // The prediction is free once the reconstruction is done.
        let planes = self
            .frame
            .planes_mut()
            .into_iter()
            .zip(self.pred.planes_mut());
        for ((rec, tmp), coefs) in planes.zip(&loop_coefs) {
            if let Some(coefs) = coefs {
                loopfilter::apply(&self.pool, rec, tmp, coefs);
                std::mem::swap(rec, tmp);
            }
        }
        watch.lap(&mut self.times, Stage::Reconstruct);
        if bi {
            let out = self.b_out.as_mut().expect("made above");
            std::mem::swap(&mut self.frame, out);
            if header.keep_reference {
                // The group's middle frame: a reference for the B-frames either side of it.
                let k = i32::from(r_keep_pos);
                match &mut self.bref {
                    Some((f, p)) => {
                        f.clone_from(out);
                        *p = k;
                    }
                    None => self.bref = Some((out.clone(), k)),
                }
            }
            return Ok(self.b_out.as_ref().expect("made above"));
        }
        self.anchor_inter = inter;
        self.global = global;
        self.loop_coefs = std::array::from_fn(|k| loop_coefs[k].unwrap_or(previous[k]));
        self.has_frame = true;
        Ok(&self.frame)
    }
}
