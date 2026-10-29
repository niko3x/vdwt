//! The rt decoder (see the [module](super) docs). It decodes whatever packets of a frame
//! arrived, in any order; malformed or missing ones only cost their tiles.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::bitstream::{ByteReader, FrameType};
use crate::codec::{self, BandSlot, LevelBands};
use crate::dwt::Dwt97;
use crate::dwt::d97::InverseJob;
use crate::encoder::MAX_THREADS;
use crate::entropy::{CoeffBand, Contexts, RangeDecoder, code_band};
use crate::error::{Error, Result};
use crate::loopfilter::{self, BitReader, Coefs};
use crate::motion::tree::{self, TreeContexts};
use crate::motion::{Grid, MotionField};
use crate::plane::{RowsMut, add_residual_row, uncenter_row};
use crate::pool::Pool;
use crate::profile::{Stage, StageTimes, Stopwatch};
use crate::quant::Dequantizer;
use crate::{Frame, Isa, Plane};

use super::encoder::global_prediction;
use super::predict::Bands;
use super::{PacketHead, RtHeader, Tiles};

/// What one frame's decoding had to work with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtDecoded {
    /// Packets that arrived and parsed.
    pub packets: usize,
    pub tiles: usize,
    pub tiles_received: usize,
    /// No usable packet arrived (or a Repeat frame): the last picture stands.
    pub repeated: bool,
    pub inter: bool,
}

struct Component {
    levels: usize,
    dwt: Dwt97,
    coeffs: Plane<i16>,
}

pub struct RtDecoder {
    header: RtHeader,
    slots: Vec<BandSlot>,
    tiles: Tiles,
    contexts: Vec<Contexts>,
    comps: [Component; 3],
    /// Per component and level, its bands (see [`LevelBands`]).
    level_bands: [Vec<LevelBands>; 3],
    /// Per band, the rows that received a value this frame.
    touched: Vec<Vec<AtomicBool>>,
    /// The last decoded picture: the output and the next reference.
    frame: Frame,
    has_frame: bool,
    pred: Frame,
    field: MotionField,
    /// Every node's offset from the global motion in the last P-frame, for concealment.
    offsets: Vec<[i32; 2]>,
    pool: Pool,
    times: StageTimes,
}

impl RtDecoder {
    pub fn new(header: RtHeader, isa: Isa, threads: usize) -> Result<Self> {
        if !(1..=MAX_THREADS).contains(&threads) {
            return Err(Error::Unsupported("thread count"));
        }
        if !isa.is_supported() {
            return Err(Error::Unsupported("ISA not available"));
        }
        let seq = &header.seq;
        if seq.wavelet != crate::dwt::Wavelet::Lifted97
            || seq.luma_weights.len() != 1 + 3 * seq.luma_levels
            || seq.chroma_weights.len() != 1 + 3 * seq.chroma_levels
        {
            return Err(Error::InvalidData("rt sequence header"));
        }
        let slots = codec::band_slots(seq);
        let tiles = Tiles::new(seq, &slots, header.tile_roots);
        let comps = std::array::from_fn(|comp| {
            let (w, h) = codec::component_size(seq, comp);
            Component {
                levels: codec::component_levels(seq, comp),
                dwt: Dwt97::fast(w, h, isa).expect("checked ISA"),
                coeffs: Plane::new(w, h),
            }
        });
        let grid = Grid::new(seq.width, seq.height, seq.mesh_log2);
        let level_bands = std::array::from_fn(|comp| LevelBands::of(&slots, comp));
        Ok(Self {
            level_bands,
            contexts: codec::default_contexts(),
            touched: slots
                .iter()
                .map(|s| (0..s.sb.height).map(|_| AtomicBool::new(false)).collect())
                .collect(),
            frame: Frame::new(seq.width, seq.height),
            has_frame: false,
            pred: Frame::new(seq.width, seq.height),
            field: MotionField::new(grid),
            offsets: vec![[0; 2]; grid.nodes()],
            pool: Pool::new(threads),
            times: StageTimes::default(),
            slots,
            tiles,
            comps,
            header,
        })
    }

    /// The last decoded (or repeated) picture.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn times(&self) -> &StageTimes {
        &self.times
    }

    /// Decodes one frame from the packets that arrived (`None` for a lost one).
    pub fn decode_frame(&mut self, packets: &[Option<&[u8]>]) -> Result<RtDecoded> {
        self.times = StageTimes::default();
        let mut watch = Stopwatch::start();
        let count = self.tiles.count;
        let mut head: Option<PacketHead> = None;
        let mut segments: Vec<Option<&[u8]>> = vec![None; count];
        let mut loop_bits: Option<&[u8]> = None;
        let mut received = 0;
        for packet in packets.iter().flatten() {
            let mut r = ByteReader::new(packet);
            let Ok(h) = PacketHead::parse(&mut r) else {
                continue;
            };
            if head.is_some_and(|first| first.header != h.header || first.global != h.global) {
                continue;
            }
            // A packet cut short still yields the tiles before the cut.
            if h.index == 0 && h.header.loop_filter {
                loop_bits = r.chunk().ok();
            }
            for slot in segments.iter_mut().skip(h.first_tile).take(h.tiles) {
                match r.chunk() {
                    Ok(seg) => *slot = Some(seg),
                    Err(_) => break,
                }
            }
            head.get_or_insert(h);
            received += 1;
        }
        let mut info = RtDecoded {
            packets: received,
            tiles: count,
            tiles_received: segments.iter().filter(|s| s.is_some()).count(),
            ..RtDecoded::default()
        };
        let head = match head {
            Some(h) if h.header.frame_type != FrameType::Repeat => h,
            _ if self.has_frame => {
                info.repeated = true;
                return Ok(info);
            }
            _ => return Err(Error::InvalidData("no picture yet")),
        };
        let inter = head.header.frame_type == FrameType::Inter;
        if inter && !self.has_frame {
            return Err(Error::InvalidData("no previous picture"));
        }
        if !inter {
            self.offsets.fill([0; 2]);
        }
        info.inter = inter;
        let (qp, recon_offset) = (head.header.qp, head.header.recon_offset);
        let mut decoders: Vec<(usize, Option<RangeDecoder>)> = segments
            .iter()
            .enumerate()
            .map(|(t, s)| (t, s.map(RangeDecoder::new)))
            .collect();
        if let Some(global) = head.global.filter(|_| inter) {
            // Lost tiles keep each node's last offset from the global motion: motion changes
            // little from frame to frame, local motion (parallax, the foreground) included.
            self.field.set_global(&global);
            let pred = global_prediction(&self.field.grid, &global);
            for ((mv, p), o) in self.field.mv.iter_mut().zip(&pred).zip(&self.offsets) {
                *mv = std::array::from_fn(|c| p[c] + o[c]);
            }
            for (t, dec) in decoders
                .iter_mut()
                .filter_map(|(t, d)| Some((t, d.as_mut()?)))
            {
                tree::code_rows(
                    dec,
                    &mut TreeContexts::default(),
                    &mut self.field,
                    &pred,
                    self.tiles.roots(*t),
                );
            }
            for (((o, mv), p), &intra) in self
                .offsets
                .iter_mut()
                .zip(&self.field.mv)
                .zip(&pred)
                .zip(&self.field.intra)
            {
                *o = if intra {
                    [0; 2]
                } else {
                    [mv[0] - p[0], mv[1] - p[1]]
                };
            }
            watch.lap(&mut self.times, Stage::Entropy);
        }
        // One batch: the tiles' bands (first, the longer tasks) and the prediction's rows of
        // cells, which need nothing from each other. Each tile zeroes its rows of every band that
        // the last frame left a value in (the rest are zero already; a lost tile's stay zero)
        // and decodes its quantized values into the coefficients; the rows that received any are
        // then dequantized in place.
        {
            enum Job<'a, 'b> {
                Tile(usize, Option<RangeDecoder<'b>>),
                Cells(usize, RowsMut<'a, u8>),
            }
            let (slots, tiles, contexts) = (&self.slots, &self.tiles, &self.contexts);
            let (block_log2, touched) = (self.header.seq.block_log2, &self.touched);
            let coeffs: Vec<_> = self.comps.iter_mut().map(|c| c.coeffs.shared()).collect();
            let mut jobs: Vec<Job> = decoders
                .into_iter()
                .map(|(t, dec)| Job::Tile(t, dec))
                .collect();
            let bands = inter.then(|| {
                let (bands, cells) = Bands::new(
                    &self.frame,
                    &mut self.pred,
                    &self.field,
                    [head.ref_clean, head.clean],
                );
                jobs.extend(cells.into_iter().map(|(c, band)| Job::Cells(c, band)));
                bands
            });
            let decode_tile = |t: usize, dec: &mut Option<RangeDecoder>| {
                for (i, slot) in slots.iter().enumerate() {
                    let rect = tiles.band_rect(slots, i, t);
                    if rect.h == 0 {
                        continue;
                    }
                    let (r, rows) = (slot.rect(), &coeffs[slot.comp]);
                    let flags = &touched[i][rect.y0..rect.y0 + rect.h];
                    for (y, flag) in (rect.y0..).zip(flags) {
                        if flag.swap(false, Ordering::Relaxed) {
                            // SAFETY: tiles hold disjoint rows of every band.
                            unsafe { rows.row_mut(r.y0 + y, r.x0, r.w) }.fill(0);
                        }
                    }
                    let Some(dec) = dec else { continue };
                    let mut ctx = contexts[slot.coding_class(inter)].clone();
                    code_band(
                        dec,
                        &mut ctx,
                        CoeffBand {
                            rows,
                            x0: r.x0,
                            y0: r.y0,
                            touched: &touched[i],
                        },
                        rect,
                        None,
                        None,
                        None,
                        block_log2,
                        slot.transposed(),
                    );
                }
            };
            self.pool.for_each_mut(&mut jobs, |_, job| match job {
                Job::Tile(t, dec) => decode_tile(*t, dec),
                Job::Cells(c, band) => bands.as_ref().expect("P-frame").run(*c, band),
            });
        }
        watch.lap(&mut self.times, Stage::Prediction);

        if inter {
            std::mem::swap(&mut self.frame, &mut self.pred);
        }
        let (seq, slots, touched) = (&self.header.seq, &self.slots, &self.touched);
        {
            let coeffs: Vec<_> = self.comps.iter_mut().map(|c| c.coeffs.shared()).collect();
            self.pool.run(slots.len(), &|i| {
                let (slot, r) = (&slots[i], slots[i].rect());
                let deq = Dequantizer::new(slot.step_q12(seq, qp), recon_offset);
                for (y, _) in touched[i]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.load(Ordering::Relaxed))
                {
                    // SAFETY: bands do not overlap, and each is one task.
                    deq.dequantize_in_place(unsafe {
                        coeffs[slot.comp].row_mut(r.y0 + y, r.x0, r.w)
                    });
                }
            });
        }
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
        let bands = &self.level_bands;
        let zeros: Vec<_> = (0..3)
            .map(|comp| {
                move |l: usize, y: usize, right: bool| {
                    bands[comp][l]
                        .slot(y, right)
                        .is_some_and(|i| !touched[i][y - slots[i].sb.y0].load(Ordering::Relaxed))
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
        if let Some(coefs) = loop_bits.and_then(read_loop_filter) {
            let planes = self
                .frame
                .planes_mut()
                .into_iter()
                .zip(self.pred.planes_mut());
            for ((rec, tmp), coefs) in planes.zip(&coefs) {
                if let Some(coefs) = coefs {
                    loopfilter::apply(&self.pool, rec, tmp, coefs);
                    std::mem::swap(rec, tmp);
                }
            }
        }
        watch.lap(&mut self.times, Stage::Reconstruct);
        self.has_frame = true;
        Ok(info)
    }
}

/// The loop filter's absolute coefficients per plane, or `None` if malformed.
fn read_loop_filter(bytes: &[u8]) -> Option<[Option<Coefs>; 3]> {
    let mut r = BitReader::new(bytes);
    let none = Coefs::default();
    let mut out = [None; 3];
    for coefs in &mut out {
        if r.bit()? {
            *coefs = Some(loopfilter::read_coefs(&mut r, &none)?);
        }
    }
    Some(out)
}
