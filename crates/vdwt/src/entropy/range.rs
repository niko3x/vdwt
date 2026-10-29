//! A multi-symbol range coder: 32-bit range, byte-wise output with carry propagation,
//! [`PROB_BITS`]-bit distributions and batches of raw bits.
//!
//! The decoder reads zeros past the end of its input. That lets termination write at most one
//! byte beyond the pending carry bytes and drop trailing zeros, so a coded segment is no longer
//! than it must be. At low bitrates a frame has dozens of segments, so this matters.

use super::cdf::{Cdf, PROB_BITS};

/// The range is renormalized to stay at or above this.
const TOP: u32 = 1 << 24;

#[derive(Debug)]
pub struct RangeEncoder {
    /// Low end of the interval: 32 bits plus a carry in bit 32.
    low: u64,
    range: u32,
    /// The newest byte not yet written, because a carry could still change it.
    cache: Option<u8>,
    /// 0xFF bytes after `cache` that a carry would also ripple through.
    ff_run: usize,
    out: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: None,
            ff_run: 0,
            out: Vec::new(),
        }
    }

    /// Codes symbol `s` with probabilities from `cdf`. The last symbol absorbs the range's rounding
    /// remainder.
    #[inline]
    pub fn encode(&mut self, cdf: &Cdf, s: usize) {
        let r = self.range >> PROB_BITS;
        let lo = r * cdf.cum(s);
        self.low += u64::from(lo);
        self.range = if s + 1 == cdf.symbols() {
            self.range - lo
        } else {
            r * cdf.prob(s)
        };
        self.normalize();
    }

    /// Codes the low `count` bits of `v` (1 ≤ `count` ≤ 16) as equiprobable bits.
    #[inline]
    pub fn encode_bits(&mut self, count: u32, v: u32) {
        debug_assert!((1..=16).contains(&count) && v >> count == 0);
        self.range >>= count;
        self.low += u64::from(self.range * v);
        self.normalize();
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    /// Moves the top byte of `low` out, resolving any carry into the bytes held back so far.
    fn shift_low(&mut self) {
        let carry = (self.low >> 32) as u8;
        let byte = (self.low >> 24) as u8;
        if byte == 0xFF && carry == 0 {
            self.ff_run += 1;
        } else {
            // With no byte held yet there is nothing a carry could reach: the coded value stays
            // inside the initial interval [0, 2^32), so the first byte never overflows.
            debug_assert!(self.cache.is_some() || carry == 0);
            if let Some(c) = self.cache {
                self.out.push(c.wrapping_add(carry));
            }
            for _ in 0..self.ff_run {
                self.out.push(0xFFu8.wrapping_add(carry));
            }
            self.ff_run = 0;
            self.cache = Some(byte);
        }
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Bytes produced so far, the pending ones included: the segment's size up to here,
    /// within a few bytes.
    pub fn bytes_so_far(&self) -> usize {
        self.out.len() + usize::from(self.cache.is_some()) + self.ff_run
    }

    /// Finishes the segment and returns its bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Settle on the value in [low, low + range) with the most trailing zero bytes. Zeros past
        // the end come for free, because the decoder supplies them.
        let end = self.low + u64::from(self.range);
        for bits in [32u32, 24, 16, 8, 0] {
            let mask = (1u64 << bits) - 1;
            let v = (self.low + mask) & !mask;
            if v < end {
                self.low = v;
                break;
            }
        }
        for _ in 0..5 {
            self.shift_low();
        }
        while self.out.last() == Some(&0) {
            self.out.pop();
        }
        self.out
    }
}

/// Decodes a segment. Never panics on malformed input; it just decodes garbage.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// The coded value minus the interval's low end.
    code: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self {
            data,
            pos: 0,
            code: 0,
            range: u32::MAX,
        };
        for _ in 0..4 {
            d.code = (d.code << 8) | u32::from(d.next_byte());
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos = self.pos.saturating_add(1);
        b
    }

    #[inline]
    pub fn decode(&mut self, cdf: &Cdf) -> usize {
        let r = self.range >> PROB_BITS;
        let n = cdf.symbols();
        let mut s = 0;
        while s + 1 < n && r * cdf.cum(s + 1) <= self.code {
            s += 1;
        }
        let lo = r * cdf.cum(s);
        self.code -= lo;
        self.range = if s + 1 == n {
            self.range - lo
        } else {
            r * cdf.prob(s)
        };
        self.normalize();
        s
    }

    /// Inverse of [`RangeEncoder::encode_bits`].
    #[inline]
    pub fn decode_bits(&mut self, count: u32) -> u32 {
        debug_assert!((1..=16).contains(&count));
        self.range >>= count;
        let v = (self.code / self.range).min((1 << count) - 1);
        self.code -= v * self.range;
        self.normalize();
        v
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | u32::from(self.next_byte());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    enum Op {
        Symbol { ctx: usize, s: usize },
        Bits { count: u32, v: u32 },
    }

    /// A random program of symbols (over adaptive contexts of every alphabet size) and raw bits.
    fn program(rng: &mut SplitMix64, len: usize, contexts: &[Cdf], skew: u64) -> Vec<Op> {
        (0..len)
            .map(|_| {
                if rng.below(4) == 0 {
                    let count = rng.range_usize(1, 16) as u32;
                    Op::Bits {
                        count,
                        v: (rng.next_u64() as u32) & ((1 << count) - 1),
                    }
                } else {
                    let ctx = rng.below(contexts.len() as u64) as usize;
                    let n = contexts[ctx].symbols() as u64;
                    // Mostly the first symbol, so that probabilities skew and carries happen.
                    let s = if rng.below(skew) != 0 {
                        0
                    } else {
                        rng.below(n) as usize
                    };
                    Op::Symbol { ctx, s }
                }
            })
            .collect()
    }

    fn round_trip(ops: &[Op], contexts: &[Cdf]) -> usize {
        let mut enc_ctx = contexts.to_vec();
        let mut enc = RangeEncoder::new();
        for op in ops {
            match *op {
                Op::Symbol { ctx, s } => {
                    enc.encode(&enc_ctx[ctx], s);
                    enc_ctx[ctx].update(s);
                }
                Op::Bits { count, v } => enc.encode_bits(count, v),
            }
        }
        let bytes = enc.finish();
        let mut dec_ctx = contexts.to_vec();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, op) in ops.iter().enumerate() {
            match *op {
                Op::Symbol { ctx, s } => {
                    let got = dec.decode(&dec_ctx[ctx]);
                    assert_eq!(got, s, "symbol {i}");
                    dec_ctx[ctx].update(got);
                }
                Op::Bits { count, v } => assert_eq!(dec.decode_bits(count), v, "bits {i}"),
            }
        }
        bytes.len()
    }

    #[test]
    fn round_trips_random_programs() {
        let mut rng = SplitMix64::new(11);
        let contexts: Vec<Cdf> = (2..=16).map(Cdf::uniform).collect();
        for case in 0..400 {
            let len = rng.range_usize(0, if case % 10 == 0 { 5000 } else { 60 });
            let skew = [2, 8, 64, 1000][case % 4];
            round_trip(&program(&mut rng, len, &contexts, skew), &contexts);
        }
    }

    #[test]
    fn empty_and_certain_segments_cost_nothing() {
        assert!(RangeEncoder::new().finish().is_empty());
        // A highly predictable symbol run fits in a byte or two.
        let mut cdf = Cdf::uniform(2);
        for _ in 0..2000 {
            cdf.update(0);
        }
        let ops: Vec<Op> = (0..50).map(|_| Op::Symbol { ctx: 0, s: 0 }).collect();
        assert!(round_trip(&ops, &[cdf]) <= 1);
    }

    #[test]
    fn size_tracks_information_content() {
        // 4000 uniform 16-ary symbols carry 16000 bits: the segment should be within a few bytes.
        let mut rng = SplitMix64::new(5);
        let cdf = Cdf::uniform(16);
        let mut enc = RangeEncoder::new();
        for _ in 0..4000 {
            enc.encode(&cdf, rng.below(16) as usize);
        }
        let len = enc.finish().len();
        assert!((2000..=2008).contains(&len), "{len} bytes");
    }

    #[test]
    fn garbage_never_panics() {
        let mut rng = SplitMix64::new(13);
        let mut cdfs: Vec<Cdf> = (2..=16).map(Cdf::uniform).collect();
        for _ in 0..200 {
            let data: Vec<u8> = (0..rng.below(40)).map(|_| rng.next_u64() as u8).collect();
            let mut dec = RangeDecoder::new(&data);
            for _ in 0..500 {
                if rng.below(3) == 0 {
                    dec.decode_bits(rng.range_usize(1, 16) as u32);
                } else {
                    let i = rng.below(cdfs.len() as u64) as usize;
                    let s = dec.decode(&cdfs[i]);
                    assert!(s < cdfs[i].symbols());
                    cdfs[i].update(s);
                }
            }
        }
    }
}
