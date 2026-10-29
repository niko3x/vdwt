//! Stream syntax (DESIGN.md §8). All multi-byte integers are little-endian.
//!
//! ```text
//! stream    := sequence_header frame*
//! frame     := varint(len) payload                   len = payload bytes
//! payload   := frame_header [motion] layout segment*
//!                                                    (a Repeat frame: frame_header only)
//! motion    := varint(len) range-coded motion        P-frames only (see crate::motion)
//! layout    := varint(len) range-coded length per band, in coding order; 0 = absent
//! segment   := range-coded band data, concatenated in coding order; layout gives each length
//! ```
//!
//! Bands are coded coarse to fine: the LL bands of Y, U, V, then each finer resolution's HL,
//! LH, HH bands per component (see [`crate::codec`]). In a P-frame the bands hold the residual
//! after motion compensation.

use crate::dwt::{MAX_LEVELS, Wavelet};
use crate::error::{Error, Result};
use crate::motion::{Filter, MESH_LOG2, Prediction};

pub const MAGIC: [u8; 4] = *b"VDWT";
/// Incremented on every incompatible change until the format is frozen.
pub const VERSION: u8 = 10;
/// Streams that use the opt-in temporal motion predictor write this version.
pub const VERSION_TEMPORAL: u8 = 11;
/// Largest accepted frame width or height.
pub const MAX_DIMENSION: usize = 16384;

const CHROMA_420: u8 = 1;
const BIT_DEPTH_8: u8 = 8;
const KERNEL_5_3: u8 = 0;
const KERNEL_9_7: u8 = 1;

/// Appends `v` as an unsigned LEB128 varint.
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Reads from a byte slice; every read fails cleanly on truncation.
#[derive(Debug)]
pub struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(Error::InvalidData("truncated"));
        }
        let b = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(b)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.bytes(2)?.try_into().expect("2 bytes"),
        ))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("4 bytes"),
        ))
    }

    pub fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for i in 0..10 {
            let b = self.u8()?;
            let bits = u64::from(b & 0x7F);
            if i == 9 && bits > 1 {
                return Err(Error::InvalidData("varint overflows 64 bits"));
            }
            v |= bits << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(Error::InvalidData("varint too long"))
    }

    /// A varint-prefixed byte string.
    pub fn chunk(&mut self) -> Result<&'a [u8]> {
        let len =
            usize::try_from(self.varint()?).map_err(|_| Error::InvalidData("chunk too long"))?;
        self.bytes(len)
    }
}

/// Stream-wide parameters, sent once at the start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceHeader {
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
    pub luma_levels: usize,
    pub chroma_levels: usize,
    /// Quadtree root size of the band coder, as log2 (6 = 64×64).
    pub block_log2: u32,
    /// Motion grid spacing in luma pixels, as log2 (5 = 32).
    pub mesh_log2: u32,
    /// The wavelet of every frame.
    pub wavelet: Wavelet,
    /// Step weight per luma subband (Q12), in [`Geometry::subbands`](crate::dwt::Geometry::subbands)
    /// order.
    pub luma_weights: Vec<u16>,
    /// The same for both chroma components.
    pub chroma_weights: Vec<u16>,
    /// Whether inter frames carry the texture guide from the prediction's transform. Off is the
    /// realtime mode: no prediction DWT on either side, at a small bitrate cost.
    pub guide: bool,
    /// Whether the motion-tree CDFs carry from one coded frame to the next. Set only when the
    /// encoder has no rate control, so a frame cannot be skipped after its CDFs advanced.
    pub continuous_cdfs: bool,
    /// Whether band lengths travel in one range-coded layout section (VERSION 9). Without it,
    /// the old presence mask and per-band length varints are used.
    pub layout_coding: bool,
    /// Whether every band keeps its own entropy contexts from frame to frame (fixed-qp
    /// streams). Each band adapts to its own coefficient statistics; no cross-band merge and
    /// no ordering dependence.
    pub band_contexts: bool,
    /// Whether P-frames use the previous coded frame's node offsets as a temporal predictor
    /// in the motion quadtree.
    pub temporal_mv: bool,
    /// Whether P-frames trial the previous frame's offsets as an alternative motion predictor
    /// (VERSION 11, fixed-qp).
    pub temporal_motion: bool,
}

impl SequenceHeader {
    pub fn chroma_size(&self) -> (usize, usize) {
        crate::frame::chroma_size(self.width, self.height)
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&MAGIC);
        out.push(if self.temporal_motion {
            VERSION_TEMPORAL
        } else {
            VERSION
        });
        out.extend_from_slice(&(self.width as u16).to_le_bytes());
        out.extend_from_slice(&(self.height as u16).to_le_bytes());
        let kernel = match self.wavelet {
            Wavelet::Reversible53 => KERNEL_5_3,
            Wavelet::Lifted97 => KERNEL_9_7,
        };
        out.extend_from_slice(&[CHROMA_420, BIT_DEPTH_8, kernel]);
        out.extend_from_slice(&[
            self.luma_levels as u8,
            self.chroma_levels as u8,
            self.block_log2 as u8,
            self.mesh_log2 as u8,
            u8::from(self.guide)
                | (u8::from(self.continuous_cdfs) << 1)
                | (u8::from(self.layout_coding) << 2)
                | (u8::from(self.band_contexts) << 3)
                | (u8::from(self.temporal_mv) << 4)
                | (u8::from(self.temporal_motion) << 5),
        ]);
        out.extend_from_slice(&self.fps_num.to_le_bytes());
        out.extend_from_slice(&self.fps_den.to_le_bytes());
        for w in self.luma_weights.iter().chain(&self.chroma_weights) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    /// Parses and validates a header. Returns it and its length in bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut r = ByteReader::new(data);
        if r.bytes(4)? != MAGIC {
            return Err(Error::InvalidData("not a VDWT stream"));
        }
        let version = r.u8()?;
        if !matches!(version, 9 | 10 | VERSION_TEMPORAL) {
            return Err(Error::Unsupported("stream version"));
        }
        let (width, height) = (usize::from(r.u16()?), usize::from(r.u16()?));
        if !(1..=MAX_DIMENSION).contains(&width) || !(1..=MAX_DIMENSION).contains(&height) {
            return Err(Error::InvalidData("frame size"));
        }
        if r.u8()? != CHROMA_420 || r.u8()? != BIT_DEPTH_8 {
            return Err(Error::Unsupported("only 8-bit 4:2:0 is implemented"));
        }
        let wavelet = match r.u8()? {
            KERNEL_5_3 => Wavelet::Reversible53,
            KERNEL_9_7 => Wavelet::Lifted97,
            _ => return Err(Error::Unsupported("wavelet kernel")),
        };
        let (luma_levels, chroma_levels) = (usize::from(r.u8()?), usize::from(r.u8()?));
        if luma_levels > MAX_LEVELS || chroma_levels > MAX_LEVELS {
            return Err(Error::InvalidData("decomposition depth"));
        }
        let block_log2 = u32::from(r.u8()?);
        if !(2..=7).contains(&block_log2) {
            return Err(Error::InvalidData("block size"));
        }
        let mesh_log2 = u32::from(r.u8()?);
        if !MESH_LOG2.contains(&mesh_log2) {
            return Err(Error::InvalidData("motion grid spacing"));
        }
        let flags = r.u8()?;
        let flag_mask = if version >= 11 {
            0b111111
        } else if version == 10 {
            0b11111
        } else {
            0b111
        };
        if flags & !flag_mask != 0 {
            return Err(Error::InvalidData("sequence flags"));
        }
        let guide = flags & 1 != 0;
        let continuous_cdfs = flags & 0b10 != 0;
        let layout_coding = flags & 0b100 != 0;
        let band_contexts = version >= 10 && flags & 0b1000 != 0;
        let temporal_mv = version == 10 && flags & 0b10000 != 0;
        let temporal_motion = version >= 11 && flags & 0b100000 != 0;
        let (fps_num, fps_den) = (r.u32()?, r.u32()?);
        let mut weights = |levels: usize| {
            (0..1 + 3 * levels)
                .map(|_| r.u16())
                .collect::<Result<Vec<_>>>()
        };
        let luma_weights = weights(luma_levels)?;
        let chroma_weights = weights(chroma_levels)?;
        let header = Self {
            width,
            height,
            fps_num,
            fps_den,
            luma_levels,
            chroma_levels,
            block_log2,
            mesh_log2,
            wavelet,
            luma_weights,
            chroma_weights,
            guide,
            continuous_cdfs,
            layout_coding,
            band_contexts,
            temporal_mv,
            temporal_motion,
        };
        Ok((header, r.position()))
    }
}

/// Frame coding type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    Intra,
    /// Show the previous frame again. The payload is the header alone. Rate control sends it
    /// when even a minimal frame would overflow the link's delay budget.
    Repeat,
    /// A P-frame: motion compensation from the previous picture, plus a coded residual.
    Inter,
    /// A B-frame: motion compensation from two references either side of it in display order
    /// (the picture before its group's anchor, kept for it, the anchor, or the group's kept
    /// middle B-frame), plus a coded residual (§6.9).
    Bi,
}

impl FrameType {
    /// Whether the frame codes a residual from a motion-compensated prediction.
    pub fn predicted(self) -> bool {
        matches!(self, FrameType::Inter | FrameType::Bi)
    }

    fn code(self) -> u8 {
        match self {
            FrameType::Intra => 0,
            FrameType::Repeat => 1,
            FrameType::Inter => 2,
            FrameType::Bi => 3,
        }
    }
}

const FLAG_PARENT_CONTEXTS: u8 = 1;
const FLAG_LL_PREDICTION: u8 = 2;
/// Bits 2–3: the P-frame's interpolation filter.
const FILTER_SHIFT: u32 = 2;
const FILTER_MASK: u8 = 3 << FILTER_SHIFT;
/// Bit 4: overlapped prediction instead of the warp.
const FLAG_OVERLAPPED: u8 = 16;
/// Bit 5: a loop-filter segment follows the bands.
const FLAG_LOOP_FILTER: u8 = 32;
/// Bit 6: keep a reference for B-frames (see [`FrameHeader::keep_reference`]).
const FLAG_KEEP_REFERENCE: u8 = 64;
/// A post-filter strength byte follows the flags (0 is omitted).
const FLAG_POSTFILTER: u8 = 128;

/// Per-frame parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub frame_num: u16,
    /// Quality index: base step `2^(qp/8)`.
    pub qp: u8,
    /// Reconstruction offset δ in 1/256 units.
    pub recon_offset: u8,
    /// Contexts use the parent band (one level coarser).
    pub parent_contexts: bool,
    /// LL bands are coded as residuals of a median (MED) predictor.
    pub ll_prediction: bool,
    /// P-frames: the motion-compensation filter. The encoder may pick it per frame.
    pub filter: Filter,
    /// P-frames: how node vectors become the prediction.
    pub prediction: Prediction,
    /// A loop-filter segment follows the bands (§5.4).
    pub loop_filter: bool,
    /// On an anchor: the decoder keeps the picture this frame replaces, for the B-frames that
    /// follow. On a B-frame: the decoder keeps this frame as its group's middle reference.
    pub keep_reference: bool,
    /// Display post-filter strength in 1/8 units, 0..=40 (`band2` only). The encoder picks one
    /// per scene and holds it until the next intra frame; 0 disables the filter.
    pub postfilter_strength: u8,
}

/// The fields a short frame header inherits from the previous coded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeaderState {
    pub qp: u8,
    pub recon_offset: u8,
    pub postfilter_strength: u8,
}

impl FrameHeader {
    pub(crate) fn state(&self) -> FrameHeaderState {
        FrameHeaderState {
            qp: self.qp,
            recon_offset: self.recon_offset,
            postfilter_strength: self.postfilter_strength,
        }
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        self.write_with(out, None);
    }

    /// Writes a header. When `prev` carries the same qp, recon offset and post-filter strength,
    /// and the frame number fits in five bits, those three bytes are omitted (bit 2 set).
    pub fn write_with(&self, out: &mut Vec<u8>, prev: Option<&FrameHeaderState>) {
        // The type byte's upper five bits hold frame numbers 0..30; 31 escapes to a varint.
        const ESCAPE: u8 = 31;
        let escaped = self.frame_num >= u16::from(ESCAPE);
        let omit = !escaped
            && self.frame_type != FrameType::Repeat
            && prev.is_some_and(|p| {
                p.qp == self.qp
                    && p.recon_offset == self.recon_offset
                    && p.postfilter_strength == self.postfilter_strength
            });
        let packed = if escaped {
            ESCAPE << 3
        } else {
            ((self.frame_num as u8) << 3) | if omit { 4 } else { 0 }
        };
        out.push(self.frame_type.code() | packed);
        if escaped {
            put_varint(out, u64::from(self.frame_num));
        }
        if self.frame_type == FrameType::Repeat {
            // A Repeat shows the last picture: qp, reconstruction and flags are irrelevant,
            // and the display post-filter is the scene's, inherited with the rest of the state.
            return;
        }
        if !omit {
            out.push(self.qp);
            out.push(self.recon_offset);
        }
        let flags = if self.parent_contexts {
            FLAG_PARENT_CONTEXTS
        } else {
            0
        } | if self.ll_prediction {
            FLAG_LL_PREDICTION
        } else {
            0
        } | self.filter.code() << FILTER_SHIFT
            | if self.prediction == Prediction::Overlapped {
                FLAG_OVERLAPPED
            } else {
                0
            }
            | if self.loop_filter {
                FLAG_LOOP_FILTER
            } else {
                0
            }
            | if self.keep_reference {
                FLAG_KEEP_REFERENCE
            } else {
                0
            }
            | if !omit && self.postfilter_strength != 0 {
                FLAG_POSTFILTER
            } else {
                0
            };
        out.push(flags);
        if !omit && self.postfilter_strength != 0 {
            out.push(self.postfilter_strength);
        }
    }

    pub fn parse(r: &mut ByteReader<'_>) -> Result<Self> {
        Self::parse_with(r, None)
    }

    /// Parses a header. `prev` supplies qp, recon offset and post-filter strength for a short
    /// header; a stream that uses one cannot be parsed with `prev = None`.
    pub fn parse_with(r: &mut ByteReader<'_>, prev: Option<&FrameHeaderState>) -> Result<Self> {
        let packed = r.u8()?;
        let omit = packed & 4 != 0;
        let frame_type = match packed & 3 {
            0 => FrameType::Intra,
            1 => FrameType::Repeat,
            2 => FrameType::Inter,
            3 => FrameType::Bi,
            _ => unreachable!(),
        };
        let frame_num = match packed >> 3 {
            31 => u16::try_from(r.varint()?).map_err(|_| Error::InvalidData("frame number"))?,
            n => u16::from(n),
        };
        if frame_type == FrameType::Repeat {
            let state = prev.copied().unwrap_or(FrameHeaderState {
                qp: 0,
                recon_offset: 0,
                postfilter_strength: 0,
            });
            return Ok(Self {
                frame_type,
                frame_num,
                qp: state.qp,
                recon_offset: state.recon_offset,
                parent_contexts: false,
                ll_prediction: false,
                filter: Filter::Bilinear,
                prediction: Prediction::Warp,
                loop_filter: false,
                keep_reference: false,
                postfilter_strength: state.postfilter_strength,
            });
        }
        let (qp, recon_offset) = if omit {
            let p = prev.ok_or(Error::Unsupported("short frame header"))?;
            (p.qp, p.recon_offset)
        } else {
            (r.u8()?, r.u8()?)
        };
        let flags = r.u8()?;
        if omit && flags & FLAG_POSTFILTER != 0 {
            return Err(Error::Unsupported("short frame header"));
        }
        let postfilter_strength = if omit {
            prev.expect("checked").postfilter_strength
        } else if flags & FLAG_POSTFILTER != 0 {
            let strength = r.u8()?;
            if strength > 40 {
                return Err(Error::Unsupported("post-filter strength"));
            }
            strength
        } else {
            0
        };
        let filter = Filter::from_code((flags & FILTER_MASK) >> FILTER_SHIFT)
            .ok_or(Error::Unsupported("motion filter"))?;
        Ok(Self {
            frame_type,
            frame_num,
            qp,
            recon_offset,
            parent_contexts: flags & FLAG_PARENT_CONTEXTS != 0,
            ll_prediction: flags & FLAG_LL_PREDICTION != 0,
            filter,
            prediction: if flags & FLAG_OVERLAPPED != 0 {
                Prediction::Overlapped
            } else {
                Prediction::Warp
            },
            loop_filter: flags & FLAG_LOOP_FILTER != 0,
            keep_reference: flags & FLAG_KEEP_REFERENCE != 0,
            postfilter_strength,
        })
    }
}

/// Splits a whole stream into its sequence header and frame payloads.
pub fn split_stream(data: &[u8]) -> Result<(SequenceHeader, Vec<&[u8]>)> {
    let (header, len) = SequenceHeader::parse(data)?;
    let mut r = ByteReader::new(&data[len..]);
    let mut frames = Vec::new();
    while r.remaining() > 0 {
        frames.push(r.chunk()?);
    }
    Ok((header, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> SequenceHeader {
        SequenceHeader {
            width: 1280,
            height: 720,
            fps_num: 60,
            fps_den: 1,
            luma_levels: 5,
            chroma_levels: 4,
            block_log2: 6,
            mesh_log2: 5,
            wavelet: Wavelet::Lifted97,
            luma_weights: (0..16).map(|i| 100 + i).collect(),
            chroma_weights: (0..13).map(|i| 200 + i).collect(),
            guide: true,
            continuous_cdfs: true,
            layout_coding: true,
            band_contexts: false,
            temporal_mv: false,
            temporal_motion: false,
        }
    }

    #[test]
    fn varints_round_trip() {
        let values = [0, 1, 127, 128, 300, 16383, 16384, u32::MAX as u64, u64::MAX];
        let mut buf = Vec::new();
        for &v in &values {
            put_varint(&mut buf, v);
        }
        let mut r = ByteReader::new(&buf);
        for &v in &values {
            assert_eq!(r.varint().unwrap(), v);
        }
        assert_eq!(r.remaining(), 0);
        assert!(ByteReader::new(&[0xFF; 11]).varint().is_err());
        assert!(ByteReader::new(&[0x80]).varint().is_err());
    }

    #[test]
    fn sequence_header_round_trips_and_rejects_garbage() {
        let mut buf = Vec::new();
        header().write(&mut buf);
        let (parsed, len) = SequenceHeader::parse(&buf).unwrap();
        assert_eq!((parsed, len), (header(), buf.len()));
        for cut in 0..buf.len() {
            assert!(
                SequenceHeader::parse(&buf[..cut]).is_err(),
                "accepted a {cut}-byte prefix"
            );
        }
        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(SequenceHeader::parse(&bad).is_err());
    }

    #[test]
    fn frame_header_round_trips() {
        for (frame_type, filter, prediction) in [
            (FrameType::Intra, Filter::Bilinear, Prediction::Warp),
            (FrameType::Inter, Filter::Cubic, Prediction::Warp),
            (FrameType::Inter, Filter::Sharp, Prediction::Overlapped),
            (FrameType::Inter, Filter::Smooth, Prediction::Overlapped),
            (FrameType::Bi, Filter::Cubic, Prediction::Overlapped),
        ] {
            let h = FrameHeader {
                frame_type,
                frame_num: 513,
                qp: 40,
                recon_offset: 96,
                parent_contexts: true,
                ll_prediction: false,
                filter,
                prediction,
                loop_filter: frame_type == FrameType::Inter,
                keep_reference: frame_type == FrameType::Inter,
                postfilter_strength: 20,
            };
            let mut buf = Vec::new();
            h.write(&mut buf);
            assert_eq!(FrameHeader::parse(&mut ByteReader::new(&buf)).unwrap(), h);
        }
    }

    #[test]
    fn repeat_header_is_one_byte_and_inherits_display_state() {
        let state = FrameHeaderState {
            qp: 42,
            recon_offset: 96,
            postfilter_strength: 20,
        };
        let h = FrameHeader {
            frame_type: FrameType::Repeat,
            frame_num: 7,
            qp: 0,
            recon_offset: 0,
            parent_contexts: true,
            ll_prediction: true,
            filter: Filter::Cubic,
            prediction: Prediction::Overlapped,
            loop_filter: true,
            keep_reference: true,
            postfilter_strength: 0,
        };
        let mut buf = Vec::new();
        h.write_with(&mut buf, Some(&state));
        assert_eq!(buf.len(), 1);
        let parsed = FrameHeader::parse_with(&mut ByteReader::new(&buf), Some(&state)).unwrap();
        assert_eq!(parsed.frame_type, FrameType::Repeat);
        assert_eq!(parsed.frame_num, 7);
        assert_eq!(parsed.qp, 42);
        assert_eq!(parsed.recon_offset, 96);
        assert_eq!(parsed.postfilter_strength, 20);
        assert!(!parsed.loop_filter && !parsed.keep_reference);
    }

    #[test]
    fn frame_header_omits_zero_postfilter() {
        let h = FrameHeader {
            frame_type: FrameType::Inter,
            frame_num: 7,
            qp: 42,
            recon_offset: 96,
            parent_contexts: true,
            ll_prediction: false,
            filter: Filter::Cubic,
            prediction: Prediction::Overlapped,
            loop_filter: false,
            keep_reference: false,
            postfilter_strength: 0,
        };
        let mut buf = Vec::new();
        h.write(&mut buf);
        // packed type/frame number + qp + recon + flags; no post-filter byte.
        assert_eq!(buf.len(), 4);
        assert_eq!(FrameHeader::parse(&mut ByteReader::new(&buf)).unwrap(), h);
    }
}
