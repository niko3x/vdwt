//! Developer: the offline context simulator (decoder roadmap, Phase 0).
//!
//! An [`Encoder`](crate::Encoder) asked to [`trace_bands`](crate::Encoder::trace_bands) keeps
//! every written band's symbols (CDF, value, position) and quantized values. [`Simulator`]
//! replays them through exact copies of the coder's adaptive CDFs, carried per band and frame
//! class and reset on intra frames as the encoder's are, so its baseline is the band section's
//! real cost to within the range coder's flush bytes. A [`Candidate`] refines some CDFs by an
//! extra key computed from already-coded data (neighbours, the co-located luma band, the
//! previous frame); each refined CDF starts from the same default table and adapts on its own,
//! so the dilution of adaptation that sank earlier context ideas is priced in. The difference
//! is what the context change would save, without touching the bitstream.
//!
//! The motion section's node quadtree is traced and priced the same way ([`MotionCandidate`]),
//! including its raw bits (vector mantissas and signs), which a candidate may put in context.
//!
//! `vdwt ctx-sim` runs it on clips.

use std::collections::HashMap;

use crate::codec::{self, FRAME_CLASSES};
use crate::dwt::Orient;
use crate::entropy::{Cdf, TracedSymbol};
use crate::motion::tree::{self, MotionSymbol};

/// One written band of a traced frame.
#[derive(Clone, Debug)]
pub struct BandTrace {
    pub(crate) slot: usize,
    pub(crate) comp: usize,
    pub(crate) level: usize,
    pub(crate) orient: Orient,
    /// Intra context class (the band's `BandSlot::class`).
    pub(crate) class: usize,
    pub(crate) inter: bool,
    pub(crate) w: usize,
    pub(crate) h: usize,
    /// The coded values (quantized coefficients, or a predicted LL band's residuals).
    pub(crate) values: Vec<i32>,
    pub(crate) symbols: Vec<TracedSymbol>,
    pub(crate) raw_bits: u64,
    /// The band's separate segment size.
    pub(crate) bytes: usize,
}

/// The written bands of one frame, whether band contexts were reset before it, and a P-frame's
/// motion section.
#[derive(Clone, Debug)]
pub struct FrameTrace {
    pub(crate) reset: bool,
    pub(crate) bands: Vec<BandTrace>,
    pub(crate) motion: Option<MotionTrace>,
}

/// A P-frame's node quadtree: its symbols and every node's offset from the global
/// prediction, intra flag and leaf depth, as coded.
#[derive(Clone, Debug, Default)]
pub struct MotionTrace {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) symbols: Vec<MotionSymbol>,
    pub(crate) nodes: Vec<([i32; 2], bool, u8)>,
    /// The whole motion section's size.
    pub(crate) bytes: usize,
}

impl MotionTrace {
    fn node(&self, i: isize, j: isize) -> Option<&([i32; 2], bool, u8)> {
        (i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny)
            .then(|| &self.nodes[j as usize * self.nx + i as usize])
    }

    /// 0 global, 1 moved, 2 intra (absent: 0).
    fn state(&self, i: isize, j: isize) -> u32 {
        self.node(i, j).map_or(0, |&(d, intra, _)| {
            if intra {
                2
            } else {
                u32::from(d != [0, 0])
            }
        })
    }
}

/// A motion context refinement. `symbols[k]` is the symbol priced; the key refines its CDF.
/// With `raw_bit` set, the candidate instead codes one raw bit after class symbols (the sign,
/// or the mantissa's top bit) with an adaptive binary CDF per key, against 1 bit raw.
pub struct MotionCandidate {
    pub name: &'static str,
    pub about: &'static str,
    /// Kinds keyed (0 split, 1 mode, 2 class); ignored with `raw_bit`.
    pub kinds: &'static [u8],
    pub raw_bit: Option<RawBit>,
    pub(crate) key: fn(Option<&MotionTrace>, &MotionTrace, &[MotionSymbol], usize) -> u32,
}

/// Which raw bit a [`MotionCandidate`] puts in context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawBit {
    Sign,
    TopMantissa,
}

/// What a CDF codes, from its place in `Contexts::cdfs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Block,
    Node,
    Rho,
    RhoForced,
    Exp,
    Mag,
    Sign,
    Mantissa,
}

const GROUP: usize = 56;
const KINDS: [Kind; 8] = [
    Kind::Block,
    Kind::Node,
    Kind::Rho,
    Kind::RhoForced,
    Kind::Exp,
    Kind::Mag,
    Kind::Sign,
    Kind::Mantissa,
];

/// The kind of CDF `k` (layout of `Contexts::cdfs`: guide class 0's group, the magnitude CDFs,
/// the other three groups, signs, mantissas).
pub fn kind(k: usize) -> Kind {
    let in_group = |i: usize| match i {
        0..4 => Kind::Block,
        4..20 => Kind::Node,
        20..32 => Kind::Rho,
        32..44 => Kind::RhoForced,
        _ => Kind::Exp,
    };
    match k {
        0..GROUP => in_group(k),
        56..60 => Kind::Mag,
        60..228 => in_group((k - 60) % GROUP),
        228..237 => Kind::Sign,
        _ => Kind::Mantissa,
    }
}

fn bit_length(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

impl BandTrace {
    /// Value at `(x, y)`, 0 outside the band.
    pub fn value(&self, x: isize, y: isize) -> i32 {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            0
        } else {
            self.values[y as usize * self.w + x as usize]
        }
    }

    /// Largest bit length in the `size × size` region at `(x, y)` (clipped to the band).
    pub fn region_bits(&self, x: isize, y: isize, size: usize) -> u32 {
        let (x0, y0) = (x.max(0) as usize, y.max(0) as usize);
        let (x1, y1) = (
            ((x + size as isize).max(0) as usize).min(self.w),
            ((y + size as isize).max(0) as usize).min(self.h),
        );
        let mut m = 0;
        for yy in y0..y1 {
            for &v in &self.values[yy * self.w + x0..yy * self.w + x1.max(x0)] {
                m = m.max(v.unsigned_abs());
            }
        }
        32 - m.leading_zeros()
    }
}

/// What a candidate sees besides the band itself: the frame's other written bands and the
/// previous traced frame's (absent bands were not written, so are zero).
pub struct Neighbourhood<'a> {
    frame: &'a FrameTrace,
    previous: Option<&'a FrameTrace>,
}

impl Neighbourhood<'_> {
    /// The written band of component `comp` at `level` and `orient` in this frame.
    pub fn band(&self, comp: usize, level: usize, orient: Orient) -> Option<&BandTrace> {
        self.frame
            .bands
            .iter()
            .find(|b| b.comp == comp && b.level == level && b.orient == orient)
    }

    /// The same band (slot) in the previous traced frame.
    pub fn previous(&self, band: &BandTrace) -> Option<&BandTrace> {
        self.previous?.bands.iter().find(|b| b.slot == band.slot)
    }

    /// The luma band that covers the same picture area and frequencies as chroma band `band`
    /// (one level finer in luma, same size for 4:2:0).
    pub fn luma_of(&self, band: &BandTrace) -> Option<&BandTrace> {
        if band.comp == 0 || band.orient == Orient::LL {
            return None;
        }
        self.band(0, band.level + 1, band.orient)
            .filter(|l| l.w == band.w && l.h == band.h)
    }
}

/// A context refinement: CDFs of the kinds it lists get an extra key.
pub struct Candidate {
    pub name: &'static str,
    pub about: &'static str,
    pub kinds: &'static [Kind],
    pub(crate) key: fn(&Neighbourhood, &BandTrace, &TracedSymbol) -> u32,
}

/// The built-in candidates (decoder roadmap Phase 2 and a few more).
pub fn candidates() -> Vec<Candidate> {
    fn left_above_nz(b: &BandTrace, s: &TracedSymbol) -> u32 {
        let size = 1usize << s.j;
        let (x, y) = (s.x as isize, s.y as isize);
        let left = b.region_bits(x - size as isize, y, size) > 0;
        let above = b.region_bits(x, y - size as isize, size) > 0;
        u32::from(left) + 2 * u32::from(above)
    }
    vec![
        Candidate {
            name: "node-left-above",
            about: "block and node flags: the left and above node of the same size (2.1)",
            kinds: &[Kind::Block, Kind::Node],
            key: |_, b, s| left_above_nz(b, s),
        },
        Candidate {
            name: "rho-left-above",
            about: "quad pattern: nonzero count of the left pair × the above pair (2.2)",
            kinds: &[Kind::Rho, Kind::RhoForced],
            key: |_, b, s| {
                let (x, y) = (s.x as isize, s.y as isize);
                let nz = |dx, dy| u32::from(b.value(x + dx, y + dy) != 0);
                let left = nz(-1, 0) + nz(-1, 1);
                let above = nz(0, -1) + nz(1, -1);
                left * 3 + above
            },
        },
        Candidate {
            name: "rho-far",
            about: "quad pattern: any nonzero one quad further left or above",
            kinds: &[Kind::Rho, Kind::RhoForced],
            key: |_, b, s| {
                let (x, y) = (s.x as isize, s.y as isize);
                u32::from(b.region_bits(x - 4, y - 4, 4) > 0 || b.region_bits(x, y - 4, 2) > 0)
                    + 2 * u32::from(b.region_bits(x - 4, y, 2) > 0)
            },
        },
        Candidate {
            name: "exp-left-above",
            about: "quad bit length: left pair's and above pair's bit length separately",
            kinds: &[Kind::Exp],
            key: |_, b, s| {
                let (x, y) = (s.x as isize, s.y as isize);
                let l = bit_length(b.value(x - 1, y)).max(bit_length(b.value(x - 1, y + 1)));
                let a = bit_length(b.value(x, y - 1)).max(bit_length(b.value(x + 1, y - 1)));
                l.min(3) * 4 + a.min(3)
            },
        },
        Candidate {
            name: "chroma-from-luma",
            about: "chroma flags, patterns and bit lengths: co-located luma band's bit length (2.4)",
            kinds: &[Kind::Block, Kind::Node, Kind::Rho, Kind::RhoForced, Kind::Exp],
            key: |n, b, s| match n.luma_of(b) {
                None => 0,
                Some(l) => {
                    let size = 1usize << s.j.max(1);
                    1 + l.region_bits(s.x as isize, s.y as isize, size).min(3)
                }
            },
        },
        Candidate {
            name: "mag-neighbours",
            about: "per-sample bit length (U − e): left and above samples' bit lengths",
            kinds: &[Kind::Mag],
            key: |_, b, s| {
                let (x, y) = (s.x as isize, s.y as isize);
                bit_length(b.value(x - 1, y)).min(3) * 4 + bit_length(b.value(x, y - 1)).min(3)
            },
        },
        Candidate {
            name: "mantissa-neighbours",
            about: "bit below the leading 1: larger neighbour's bit length",
            kinds: &[Kind::Mantissa],
            key: |_, b, s| {
                let (x, y) = (s.x as isize, s.y as isize);
                bit_length(b.value(x - 1, y))
                    .max(bit_length(b.value(x, y - 1)))
                    .min(4)
            },
        },
        Candidate {
            name: "temporal-quad",
            about: "control: quad pattern by the previous frame's same quad (rejected before)",
            kinds: &[Kind::Rho, Kind::RhoForced],
            key: |n, b, s| match n.previous(b) {
                None => 0,
                Some(p) => 1 + p.region_bits(s.x as isize, s.y as isize, 2).min(2),
            },
        },
    ]
}

fn class_of(syms: &[MotionSymbol], k: usize) -> Option<u32> {
    syms.get(k).filter(|s| s.kind == 2).map(|s| u32::from(s.s))
}

/// The built-in motion candidates (decoder roadmap 2.3, 2.5 and the raw bits).
pub fn motion_candidates() -> Vec<MotionCandidate> {
    vec![
        MotionCandidate {
            name: "mv-y-from-x",
            about: "vertical difference class by the horizontal one's (2.3)",
            kinds: &[2],
            raw_bit: None,
            key: |_, _, syms, k| {
                if syms[k].ctx / 2 == 1 {
                    1 + class_of(syms, k - 1).unwrap_or(0).min(4)
                } else {
                    0
                }
            },
        },
        MotionCandidate {
            name: "mv-class-spread",
            about: "difference class by the spread of the left and upper offsets (2.3)",
            kinds: &[2],
            raw_bit: None,
            key: |_, m, syms, k| {
                let s = &syms[k];
                let (i, j) = (s.i as isize, s.j as isize);
                let d = |n: Option<&([i32; 2], bool, u8)>| n.map_or([0; 2], |n| n.0);
                let (l, u) = (d(m.node(i - 1, j)), d(m.node(i, j - 1)));
                let c = usize::from(s.ctx / 2);
                let spread = (l[c] - u[c]).unsigned_abs() / 2;
                (32 - spread.leading_zeros()).min(3)
            },
        },
        MotionCandidate {
            name: "mode-temporal",
            about: "leaf mode by the previous frame's node there: global, moved, intra (2.5)",
            kinds: &[1],
            raw_bit: None,
            key: |p, m, syms, k| match p.filter(|p| p.nx == m.nx && p.ny == m.ny) {
                None => 0,
                Some(p) => 1 + p.state(syms[k].i as isize, syms[k].j as isize),
            },
        },
        MotionCandidate {
            name: "split-temporal",
            about: "split flag by whether the previous frame split this region",
            kinds: &[0],
            raw_bit: None,
            key: |p, m, syms, k| match p.filter(|p| p.nx == m.nx && p.ny == m.ny) {
                None => 0,
                Some(p) => {
                    let s = &syms[k];
                    let d = p.node(s.i as isize, s.j as isize).map_or(0, |n| n.2);
                    1 + u32::from(d < s.level)
                }
            },
        },
        MotionCandidate {
            name: "mv-sign",
            about: "difference sign in context: component × the horizontal sign (for vertical)",
            kinds: &[],
            raw_bit: Some(RawBit::Sign),
            key: |_, _, syms, k| {
                let c = u32::from(syms[k].ctx / 2);
                let xs = if c == 1 && k > 0 && syms[k - 1].kind == 2 && syms[k - 1].s > 0 {
                    1 + (syms[k - 1].raw & 1)
                } else {
                    0
                };
                c * 3 + xs
            },
        },
        MotionCandidate {
            name: "mv-top-mantissa",
            about: "difference mantissa's top bit in context of component and class",
            kinds: &[],
            raw_bit: Some(RawBit::TopMantissa),
            key: |_, _, syms, k| u32::from(syms[k].ctx / 2) * 16 + u32::from(syms[k].s),
        },
    ]
}

/// Adaptive state: per band slot and frame class, the CDFs (base) or keyed CDFs (candidate).
#[derive(Default)]
struct State {
    cdfs: HashMap<(usize, bool, usize, u32), Cdf>,
}

impl State {
    fn cost_and_update(&mut self, key: (usize, bool, usize, u32), init: &Cdf, s: usize) -> f64 {
        let cdf = self.cdfs.entry(key).or_insert(*init);
        let bits = cdf.cost(s);
        cdf.update(s);
        bits
    }
}

/// Totals per kind of symbol, in bits.
#[derive(Clone, Default)]
struct Tally {
    bits: HashMap<Kind, f64>,
    contexts: usize,
}

/// Replays traced frames for the baseline and every candidate.
pub struct Simulator {
    candidates: Vec<Candidate>,
    defaults: Vec<Vec<Cdf>>,
    base: State,
    states: Vec<State>,
    base_tally: Tally,
    tallies: Vec<Tally>,
    raw_bits: u64,
    segment_bytes: u64,
    symbols: u64,
    frames: usize,
    previous: Option<FrameTrace>,
    motion: Vec<MotionCandidate>,
    motion_defaults: [Vec<Cdf>; 3],
    motion_base: State,
    motion_states: Vec<State>,
    /// Bits: base symbols per kind, base raw bits; per candidate total.
    motion_base_bits: [f64; 3],
    motion_raw_bits: f64,
    motion_bits: Vec<f64>,
    motion_bytes: u64,
    previous_motion: Option<MotionTrace>,
    /// Leaves per mode (global, left, up, new, intra), the nodes they cover, and the
    /// difference classes of new vectors (both components).
    modes: [u64; 5],
    mode_nodes: [u64; 5],
    classes: [u64; 16],
    /// Roots coded, roots whose every node repeats the previous frame's offset and intra flag
    /// (and partition), and the modelled bits those repeating roots cost.
    roots: u64,
    same_roots: u64,
    same_root_bits: f64,
}

impl Simulator {
    pub fn new(candidates: Vec<Candidate>, motion: Vec<MotionCandidate>) -> Self {
        let m = motion.len();
        let defaults = codec::default_contexts()
            .iter()
            .map(|c| c.cdfs().copied().collect())
            .collect();
        let n = candidates.len();
        Self {
            candidates,
            defaults,
            base: State::default(),
            states: (0..n).map(|_| State::default()).collect(),
            base_tally: Tally::default(),
            tallies: vec![Tally::default(); n],
            raw_bits: 0,
            segment_bytes: 0,
            symbols: 0,
            frames: 0,
            previous: None,
            motion,
            motion_defaults: tree::default_cdfs(),
            motion_base: State::default(),
            motion_states: (0..m).map(|_| State::default()).collect(),
            motion_base_bits: [0.0; 3],
            motion_raw_bits: 0.0,
            motion_bits: vec![0.0; m],
            motion_bytes: 0,
            previous_motion: None,
            modes: [0; 5],
            mode_nodes: [0; 5],
            classes: [0; 16],
            roots: 0,
            same_roots: 0,
            same_root_bits: 0.0,
        }
    }

    fn add_motion(&mut self, m: &MotionTrace) {
        self.motion_bytes += m.bytes as u64;
        let prev = self.previous_motion.as_ref();
        let syms = &m.symbols;
        // Root size in nodes: the largest level any symbol sits at.
        let root = 1usize << syms.iter().map(|s| s.level).max().unwrap_or(0);
        let same: Vec<bool> = match prev.filter(|p| p.nx == m.nx && p.ny == m.ny) {
            None => Vec::new(),
            Some(p) => {
                let (rx, ry) = (m.nx.div_ceil(root), m.ny.div_ceil(root));
                (0..rx * ry)
                    .map(|r| {
                        let (i0, j0) = ((r % rx) * root, (r / rx) * root);
                        (j0..(j0 + root).min(m.ny)).all(|j| {
                            (i0..(i0 + root).min(m.nx)).all(|i| {
                                let n = j * m.nx + i;
                                m.nodes[n] == p.nodes[n]
                            })
                        })
                    })
                    .collect()
            }
        };
        if !same.is_empty() {
            self.roots += same.len() as u64;
            self.same_roots += same.iter().filter(|&&s| s).count() as u64;
        }
        let rx = m.nx.div_ceil(root);
        for (k, sym) in syms.iter().enumerate() {
            let kind = usize::from(sym.kind);
            let ctx = usize::from(sym.ctx);
            let init = &self.motion_defaults[kind][ctx];
            let s = usize::from(sym.s);
            let base = self.motion_base.cost_and_update((kind, false, ctx, 0), init, s);
            self.motion_base_bits[kind] += base;
            let r = (usize::from(sym.j) / root) * rx + usize::from(sym.i) / root;
            if same.get(r).copied().unwrap_or(false) {
                self.same_root_bits += base + f64::from(sym.raw_n);
            }
            let raw = f64::from(sym.raw_n);
            self.motion_raw_bits += raw;
            match sym.kind {
                1 => {
                    self.modes[s.min(4)] += 1;
                    self.mode_nodes[s.min(4)] += 1 << (2 * u64::from(sym.level));
                }
                2 => self.classes[s.min(15)] += 1,
                _ => {}
            }
            for (c, (state, bits)) in self
                .motion
                .iter()
                .zip(self.motion_states.iter_mut().zip(&mut self.motion_bits))
            {
                match c.raw_bit {
                    None => {
                        let key = if c.kinds.contains(&sym.kind) {
                            (c.key)(prev, m, syms, k) + 1
                        } else {
                            0
                        };
                        *bits += state.cost_and_update((kind, false, ctx, key), init, s) + raw;
                    }
                    Some(which) => {
                        *bits += base;
                        // Class e > 0: e raw bits, the sign last; the mantissa has e − 1 bits.
                        let bit = match which {
                            RawBit::Sign if sym.kind == 2 && sym.raw_n > 0 => Some(sym.raw & 1),
                            RawBit::TopMantissa if sym.kind == 2 && sym.raw_n > 1 => {
                                Some((sym.raw >> (sym.raw_n - 1)) & 1)
                            }
                            _ => None,
                        };
                        match bit {
                            None => *bits += raw,
                            Some(b) => {
                                let key = (c.key)(prev, m, syms, k);
                                let flat = Cdf::uniform(2);
                                *bits += raw - 1.0
                                    + state.cost_and_update((3, false, 0, key), &flat, b as usize);
                            }
                        }
                    }
                }
            }
        }
        self.previous_motion = Some(m.clone());
    }

    pub fn add(&mut self, frame: FrameTrace) {
        if frame.reset {
            self.base.cdfs.clear();
            for s in &mut self.states {
                s.cdfs.clear();
            }
        }
        let n = Neighbourhood {
            frame: &frame,
            previous: self.previous.as_ref(),
        };
        for band in &frame.bands {
            let set = band.class + if band.inter { FRAME_CLASSES } else { 0 };
            let defaults = &self.defaults[set];
            self.raw_bits += band.raw_bits;
            self.segment_bytes += band.bytes as u64;
            self.symbols += band.symbols.len() as u64;
            for sym in &band.symbols {
                let k = usize::from(sym.k);
                let s = usize::from(sym.s);
                let kd = kind(k);
                let bits = self
                    .base
                    .cost_and_update((band.slot, band.inter, k, 0), &defaults[k], s);
                *self.base_tally.bits.entry(kd).or_default() += bits;
                for (c, (state, tally)) in self
                    .candidates
                    .iter()
                    .zip(self.states.iter_mut().zip(&mut self.tallies))
                {
                    let key = if c.kinds.contains(&kd) {
                        (c.key)(&n, band, sym) + 1
                    } else {
                        0
                    };
                    let bits =
                        state.cost_and_update((band.slot, band.inter, k, key), &defaults[k], s);
                    *tally.bits.entry(kd).or_default() += bits;
                }
            }
        }
        for (state, tally) in self.states.iter().zip(&mut self.tallies) {
            tally.contexts = tally.contexts.max(state.cdfs.len());
        }
        self.base_tally.contexts = self.base_tally.contexts.max(self.base.cdfs.len());
        if let Some(m) = &frame.motion {
            self.add_motion(m);
        }
        self.frames += 1;
        self.previous = Some(frame);
    }

    /// Modelled band bytes (symbols and raw bits) and the band segments' real size; modelled
    /// motion-tree bytes and the motion sections' real size.
    pub fn baseline(&self) -> (f64, u64, f64, u64) {
        let bands = self.base_tally.bits.values().sum::<f64>() + self.raw_bits as f64;
        let motion = self.motion_base_bits.iter().sum::<f64>() + self.motion_raw_bits;
        (bands / 8.0, self.segment_bytes, motion / 8.0, self.motion_bytes)
    }

    /// The report: baseline per kind against the segments' real size, then every candidate's
    /// saving as a share of the band bits and of `stream_bytes` (the whole stream).
    pub fn report(&self, stream_bytes: u64) -> String {
        use std::fmt::Write;
        let base_total: f64 = self.base_tally.bits.values().sum();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} frames, {} symbols; modelled {:.0} B + {} B raw bits = {:.0} B against {} B of \
             band segments ({:+.2}%); stream {} B",
            self.frames,
            self.symbols,
            base_total / 8.0,
            self.raw_bits / 8,
            (base_total + self.raw_bits as f64) / 8.0,
            self.segment_bytes,
            100.0 * ((base_total + self.raw_bits as f64) / 8.0 / self.segment_bytes as f64 - 1.0),
            stream_bytes,
        );
        let _ = write!(out, "baseline by kind:");
        for k in KINDS {
            let b = self.base_tally.bits.get(&k).copied().unwrap_or(0.0);
            let _ = write!(out, " {k:?} {:.1}%", 100.0 * b / base_total.max(1.0));
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{:22}{:>10}{:>10}{:>10}{:>9}  on its kinds",
            "candidate", "Δ bytes", "% bands", "% stream", "ctxs"
        );
        for (c, t) in self.candidates.iter().zip(&self.tallies) {
            let total: f64 = t.bits.values().sum();
            let d = (total - base_total) / 8.0;
            let own: f64 = c
                .kinds
                .iter()
                .map(|k| self.base_tally.bits.get(k).copied().unwrap_or(0.0))
                .sum();
            let _ = writeln!(
                out,
                "{:22}{:>10.0}{:>9.2}%{:>9.2}%{:>9}  {:+.2}%",
                c.name,
                d,
                100.0 * d * 8.0 / (base_total + self.raw_bits as f64),
                100.0 * d / stream_bytes.max(1) as f64,
                t.contexts,
                100.0 * (total - base_total) / own.max(1.0),
            );
        }
        if !self.motion.is_empty() && self.motion_bytes > 0 {
            let base: f64 = self.motion_base_bits.iter().sum::<f64>() + self.motion_raw_bits;
            let _ = writeln!(
                out,
                "motion: {} B in sections ({:.1}% of the stream); tree modelled {:.0} B: split                  {:.1}%, mode {:.1}%, class {:.1}%, raw {:.1}%",
                self.motion_bytes,
                100.0 * self.motion_bytes as f64 / stream_bytes.max(1) as f64,
                base / 8.0,
                100.0 * self.motion_base_bits[0] / base,
                100.0 * self.motion_base_bits[1] / base,
                100.0 * self.motion_base_bits[2] / base,
                100.0 * self.motion_raw_bits / base,
            );
            let _ = writeln!(
                out,
                "leaves global/left/up/new/intra {:?}, nodes {:?}; new-vector difference classes \
                 {:?}",
                self.modes, self.mode_nodes, self.classes
            );
            let _ = writeln!(
                out,
                "roots repeating the previous frame's motion exactly: {} of {} ({:.1}%), costing \
                 {:.0} B ({:.1}% of motion)",
                self.same_roots,
                self.roots,
                100.0 * self.same_roots as f64 / self.roots.max(1) as f64,
                self.same_root_bits / 8.0,
                100.0 * self.same_root_bits / 8.0 / self.motion_bytes.max(1) as f64,
            );
            for (c, &bits) in self.motion.iter().zip(&self.motion_bits) {
                let d = (bits - base) / 8.0;
                let _ = writeln!(
                    out,
                    "{:22}{:>10.0}{:>9.2}%{:>9.2}%  of motion / of stream",
                    c.name,
                    d,
                    100.0 * d / self.motion_bytes as f64,
                    100.0 * d / stream_bytes.max(1) as f64,
                );
            }
        }
        out
    }
}
