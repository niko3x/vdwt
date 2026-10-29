//! The realtime codec for lossy drone links (DESIGN.md §10.3): its own stream format and
//! pipeline, built for packet loss and for parallelism, on the main codec's kernels (the
//! quantizer, the band coder, the loop filter) and faster ones of its own (§10.3.2): the 9/7 in
//! Q15 lifting steps, motion search ([`motion`]), and 8-bit overlapped prediction ([`predict`]).
//!
//! **Tiles.** A frame is cut into horizontal tiles of whole motion-quadtree root rows (64 luma
//! pixels each). A tile's segment holds its rows of the motion field and its rows of every band,
//! range-coded with fresh contexts. Nothing in it depends on another tile or on the
//! reconstructed picture: no guide (the prediction's transform), no parent-band or LL-prediction
//! contexts, the global motion and the loop filter sent as absolute values. Any subset of a
//! frame's tiles therefore decodes, and tiles code and decode in parallel.
//!
//! **Packets.** Tiles are grouped into packets of at most `mtu` bytes (a larger tile goes alone).
//! Every packet repeats the frame header and the global motion, so any one that arrives can be
//! placed and decoded; the first also carries the loop filter.
//!
//! **Loss.** A lost tile's nodes follow the global motion and its residual is zero: that stripe
//! shows the motion-compensated previous picture, slightly stale rather than broken. A frame
//! with no packet repeats the last picture. The decoder's reference then differs from the
//! encoder's; gradual refresh bounds how long. Over every `refresh` frames the picture is swept
//! top to bottom, one motion root row at a time coded without prediction (its nodes intra), so
//! no frame is an intra spike. The rows swept so far form a clean zone that predicts only from
//! the reference's clean zone: its reads stop at a virtual boundary, the reference's last clean
//! row repeated below it, as at the picture's edge. After one sweep the whole picture is clean,
//! whatever was lost before it.
//!
//! ```text
//! stream  := "VDRT" version tile_roots sequence_header frame*
//! frame   := varint(n) (varint(len) packet){n}
//! packet  := frame_header [global] index count first_tile tiles
//!            [varint(len) loop_filter]                       first packet, if the header says so
//!            (varint(len) tile){tiles}
//! global  := 8 zigzag varints: the four corner vectors      P-frames
//! tile    := range-coded: motion rows (P-frames), then the tile's rows of every band
//! ```

pub mod decoder;
pub mod encoder;
mod motion;
mod predict;

pub use decoder::{RtDecoded, RtDecoder};
pub use encoder::{RtConfig, RtEncoder, RtFrame, RtFrameStats};

use std::ops::Range;

use crate::bitstream::{ByteReader, FrameHeader, FrameType, SequenceHeader, put_varint};
use crate::codec::BandSlot;
use crate::entropy::Rect;
use crate::error::{Error, Result};
use crate::motion::tree::root_log2;
use crate::motion::{GlobalMotion, Grid, MAX_GLOBAL_MV};

pub const RT_MAGIC: [u8; 4] = *b"VDRT";
pub const RT_VERSION: u8 = 3;
/// Luma rows of one motion root: tiles are whole roots tall.
const ROOT_PX: usize = 64;

/// Stream-wide parameters: the main codec's sequence header and the tile height.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RtHeader {
    pub seq: SequenceHeader,
    /// Motion root rows (64 luma pixels each) per tile.
    pub tile_roots: usize,
}

impl RtHeader {
    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&RT_MAGIC);
        out.push(RT_VERSION);
        out.push(self.tile_roots as u8);
        self.seq.write(out);
    }

    /// Parses a header. Returns it and its length in bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut r = ByteReader::new(data);
        if r.bytes(4)? != RT_MAGIC {
            return Err(Error::InvalidData("not a VDRT stream"));
        }
        if r.u8()? != RT_VERSION {
            return Err(Error::Unsupported("rt stream version"));
        }
        let tile_roots = usize::from(r.u8()?);
        if tile_roots == 0 {
            return Err(Error::InvalidData("tile height"));
        }
        let (seq, len) = SequenceHeader::parse(&data[r.position()..])?;
        Ok((Self { seq, tile_roots }, r.position() + len))
    }
}

/// Splits a whole rt stream into its header and every frame's packets.
pub fn split_stream(data: &[u8]) -> Result<(RtHeader, Vec<Vec<&[u8]>>)> {
    let (header, len) = RtHeader::parse(data)?;
    let mut r = ByteReader::new(&data[len..]);
    let mut frames = Vec::new();
    while r.remaining() > 0 {
        let n = r.varint()?;
        let packets = (0..n).map(|_| r.chunk()).collect::<Result<Vec<_>>>()?;
        frames.push(packets);
    }
    Ok((header, frames))
}

/// Appends one frame's packets to a stream.
pub fn write_frame(out: &mut Vec<u8>, packets: &[Vec<u8>]) {
    put_varint(out, packets.len() as u64);
    for p in packets {
        put_varint(out, p.len() as u64);
        out.extend_from_slice(p);
    }
}

/// Which rows of every band and which motion root rows each tile holds.
#[derive(Clone, Debug)]
pub(crate) struct Tiles {
    pub count: usize,
    /// Per band slot, the first row of every tile and then the band's height.
    band_rows: Vec<Vec<usize>>,
    root_rows: Vec<Range<usize>>,
}

impl Tiles {
    pub fn new(seq: &SequenceHeader, slots: &[BandSlot], tile_roots: usize) -> Self {
        let grid = Grid::new(seq.width, seq.height, seq.mesh_log2);
        let roots = grid.ny.div_ceil(1 << root_log2(grid.log2));
        let count = roots.div_ceil(tile_roots);
        let tile_px = tile_roots * ROOT_PX;
        let band_rows = slots
            .iter()
            .map(|slot| {
                // Band row y stands for luma rows near y·2^shift (chroma is half height).
                let shift = slot.sb.level + usize::from(slot.comp > 0);
                let h = slot.sb.height;
                (0..=count)
                    .map(|t| {
                        if t == count {
                            h
                        } else {
                            ((t * tile_px) >> shift).min(h)
                        }
                    })
                    .collect()
            })
            .collect();
        let root_rows = (0..count)
            .map(|t| t * tile_roots..((t + 1) * tile_roots).min(roots))
            .collect();
        Self {
            count,
            band_rows,
            root_rows,
        }
    }

    /// Tile `t`'s rows of band `i`, in the band's own plane.
    pub fn band_rect(&self, slots: &[BandSlot], i: usize, t: usize) -> Rect {
        let rows = &self.band_rows[i];
        Rect {
            x0: 0,
            y0: rows[t],
            w: slots[i].sb.width,
            h: rows[t + 1] - rows[t],
        }
    }

    /// Tile `t`'s motion root rows.
    pub fn roots(&self, t: usize) -> Range<usize> {
        self.root_rows[t].clone()
    }
}

/// What every packet of a frame starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PacketHead {
    pub header: FrameHeader,
    /// P-frames: the global motion.
    pub global: Option<GlobalMotion>,
    pub index: usize,
    pub count: usize,
    pub first_tile: usize,
    pub tiles: usize,
    /// Gradual refresh: node rows from the top that are clean in this frame and in its
    /// reference (see [`clean_bound`]).
    pub clean: usize,
    pub ref_clean: usize,
}

/// The virtual boundary of a clean zone of `rows` node rows `spacing` pixels apart: luma rows
/// above it depend only on the zone's nodes (0 for no zone).
pub(crate) fn clean_bound(rows: usize, spacing: usize) -> usize {
    rows.saturating_sub(1) * spacing
}

fn put_zigzag(out: &mut Vec<u8>, v: i32) {
    put_varint(out, ((v << 1) ^ (v >> 31)) as u32 as u64);
}

fn zigzag(r: &mut ByteReader<'_>) -> Result<i32> {
    let v = u32::try_from(r.varint()?).map_err(|_| Error::InvalidData("global motion"))?;
    Ok(((v >> 1) as i32) ^ -((v & 1) as i32))
}

impl PacketHead {
    pub fn write(&self, out: &mut Vec<u8>) {
        self.header.write(out);
        if let Some(g) = &self.global {
            for v in g.corners.iter().flatten() {
                put_zigzag(out, *v);
            }
        }
        for v in [
            self.index,
            self.count,
            self.first_tile,
            self.tiles,
            self.clean,
            self.ref_clean,
        ] {
            put_varint(out, v as u64);
        }
    }

    pub fn parse(r: &mut ByteReader<'_>) -> Result<Self> {
        let header = FrameHeader::parse(r)?;
        let global = match header.frame_type {
            FrameType::Inter => {
                let mut g = GlobalMotion::default();
                for v in g.corners.iter_mut().flatten() {
                    *v = zigzag(r)?.clamp(-MAX_GLOBAL_MV, MAX_GLOBAL_MV);
                }
                Some(g)
            }
            _ => None,
        };
        let mut next = || -> Result<usize> {
            usize::try_from(r.varint()?).map_err(|_| Error::InvalidData("packet"))
        };
        Ok(Self {
            header,
            global,
            index: next()?,
            count: next()?,
            first_tile: next()?,
            tiles: next()?,
            clean: next()?,
            ref_clean: next()?,
        })
    }
}

/// Drops packets as a lossy link would: each independently with probability `rate`, or in
/// bursts when `burst` > 1 (a Gilbert channel whose bad state lasts `burst` packets on average
/// and loses everything, with the same long-run loss rate). Deterministic for a `seed`.
pub struct LossChannel {
    rate: f64,
    burst: f64,
    bad: bool,
    rng: crate::rng::SplitMix64,
}

impl LossChannel {
    pub fn new(rate: f64, burst: f64, seed: u64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            burst: burst.max(1.0),
            bad: false,
            rng: crate::rng::SplitMix64::new(seed),
        }
    }

    /// Whether the next packet arrives.
    pub fn arrives(&mut self) -> bool {
        let u = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        if self.burst <= 1.0 {
            return u >= self.rate;
        }
        // Bad → good with 1/burst; good → bad so that the bad share is `rate`.
        let leave = 1.0 / self.burst;
        let enter = (self.rate * leave / (1.0 - self.rate).max(1e-9)).min(1.0);
        self.bad = if self.bad { u >= leave } else { u < enter };
        !self.bad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frame;
    use crate::rng::SplitMix64;

    /// A textured scene panning a few pixels a frame.
    fn clip(w: usize, h: usize, frames: usize) -> Vec<Frame> {
        let mut rng = SplitMix64::new(3);
        let tex: Vec<u8> = (0..(w + 64) * (h + 64))
            .map(|_| rng.below(256) as u8)
            .collect();
        let smooth = |x: usize, y: usize| {
            let at = |dx: usize, dy: usize| u32::from(tex[(y + dy) * (w + 64) + x + dx]);
            ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4) as u8
        };
        (0..frames)
            .map(|k| {
                let mut f = Frame::new(w, h);
                for (c, p) in f.planes_mut().into_iter().enumerate() {
                    let s = if c == 0 { 1 } else { 2 };
                    for y in 0..p.height() {
                        for x in 0..p.width() {
                            p.set(
                                x,
                                y,
                                smooth((x * s + 3 * k) % (w + 32), (y * s + k) % (h + 32)),
                            );
                        }
                    }
                }
                f
            })
            .collect()
    }

    fn encode(frames: &[Frame], mtu: usize) -> (RtHeader, Vec<Vec<Vec<u8>>>, Vec<Frame>) {
        let (w, h) = (frames[0].width(), frames[0].height());
        let mut cfg = RtConfig::new(w, h, 600_000);
        cfg.threads = 3;
        cfg.mtu = mtu;
        cfg.refresh = 8;
        let mut enc = RtEncoder::new(cfg).unwrap();
        let mut packets = Vec::new();
        let mut recons = Vec::new();
        for f in frames {
            packets.push(enc.encode_frame(f).unwrap().packets);
            recons.push(enc.reconstruction().clone());
        }
        (enc.header().clone(), packets, recons)
    }

    #[test]
    fn decodes_exactly_what_the_encoder_reconstructs() {
        let frames = clip(320, 192, 12);
        let (header, packets, recons) = encode(&frames, 300);
        let mut stream = Vec::new();
        header.write(&mut stream);
        for p in &packets {
            write_frame(&mut stream, p);
        }
        let (parsed, frames_in) = split_stream(&stream).unwrap();
        assert_eq!(parsed, header);
        let mut dec = RtDecoder::new(parsed, crate::Isa::best(), 2).unwrap();
        for (k, (p, recon)) in frames_in.iter().zip(&recons).enumerate() {
            let got: Vec<Option<&[u8]>> = p.iter().map(|&p| Some(p)).collect();
            let info = dec.decode_frame(&got).unwrap();
            assert_eq!(info.tiles_received, info.tiles, "frame {k}");
            for (got, want) in dec.frame().planes().iter().zip(recon.planes()) {
                assert!(
                    got.rows().eq(want.rows()),
                    "frame {k} differs from the reconstruction"
                );
            }
        }
        assert!(
            packets.iter().any(|p| p.len() > 1),
            "small packets split frames"
        );
    }

    #[test]
    fn survives_lost_reordered_and_corrupt_packets() {
        let frames = clip(320, 192, 16);
        let (header, packets, _) = encode(&frames, 200);
        let mut rng = SplitMix64::new(9);
        for trial in 0..6 {
            let mut dec = RtDecoder::new(header.clone(), crate::Isa::best(), 2).unwrap();
            for p in &packets {
                let mut owned: Vec<Vec<u8>> = p.clone();
                owned.reverse();
                for bytes in &mut owned {
                    if trial % 2 == 1 && rng.below(4) == 0 && !bytes.is_empty() {
                        let i = rng.below(bytes.len() as u64) as usize;
                        bytes[i] ^= 1 << rng.below(8);
                    }
                    if trial >= 4 && rng.below(3) == 0 {
                        bytes.truncate(rng.below(bytes.len() as u64 + 1) as usize);
                    }
                }
                let got: Vec<Option<&[u8]>> = owned
                    .iter()
                    .map(|b| (rng.below(5) != 0).then_some(b.as_slice()))
                    .collect();
                // Loss may leave nothing to show at the start; it must never panic.
                let _ = dec.decode_frame(&got);
            }
        }
    }

    #[test]
    fn loss_channel_matches_its_rate() {
        for burst in [1.0, 5.0] {
            let mut ch = LossChannel::new(0.1, burst, 4);
            let lost = (0..100_000).filter(|_| !ch.arrives()).count();
            assert!(
                (8_500..11_500).contains(&lost),
                "burst {burst}: {lost} lost"
            );
        }
    }
}
