//! The frame layout section: one range-coded length per band, zero for an absent band.
//!
//! The band segments follow with no per-band prefixes; these lengths delimit them. A length is
//! coded as its bit length (0 means absent) plus the raw mantissa. The bit-length models are
//! seeded from the disjoint training corpus. The section is stateless: a corrupt frame can never
//! desynchronise later frames.

use super::{Cdf, RangeDecoder, RangeEncoder};
use crate::error::{Error, Result};

/// One context per [`BandSlot::class`](crate::codec::BandSlot); the table is `crate::codec`'s
/// 14 band classes (luma/chroma by LL, detail level and HH).
const CLASSES: usize = 14;

/// Initial bit-length weights (per mille) per class, from 60-frame encodes of the disjoint
/// `bench/contexts` corpus at qp 32/42/50. Entry 0 is an absent band; entry `b` means the length
/// has `b` bits, and 15 is the escape for 15 or more.
#[rustfmt::skip]
const PRIORS: [[u64; 16]; CLASSES] = [
    [  0,   0,   0,   4,    5,   39,  212,  308,  256,  158,   19,    0,    0,    0,   0,   0],
    [ 21,   0,  20,  32,   67,  132,  219,  201,  199,  104,    7,    0,    0,    0,   0,   0],
    [ 71,   0,  23,  40,   78,  191,  169,  206,  131,   89,    3,    0,    0,    0,   0,   0],
    [ 44,   0,  16,  39,   52,  101,  136,  153,  144,  153,   81,   81,    1,    0,   0,   0],
    [ 86,   0,  24,  63,   85,  141,   89,  114,  185,   55,   83,   74,    1,    0,   0,   0],
    [159,   0,  17,  59,   74,   83,   95,  105,   74,  111,  101,   47,   57,   17,   1,   0],
    [335,   0,  30,  63,   55,   82,   89,   47,   99,  109,   30,    9,   50,    1,   0,   0],
    [108,   0,  31,  88,  147,  174,  168,  147,  103,   33,    2,    0,    0,    0,   0,   0],
    [421,   1,  79,  80,  101,  131,   78,   63,   44,    3,    0,    0,    0,    0,   0,   0],
    [565,   1,  46,  78,  135,   55,   53,   54,   12,    1,    0,    0,    0,    0,   0,   0],
    [567,   0,  43,  75,   92,   89,   40,   36,   37,   20,    1,    0,    0,    0,   0,   0],
    [707,   0,  46,  75,   55,   35,   34,   32,   15,    1,    1,    0,    0,    0,   0,   0],
    [831,   0,  24,  42,   39,   19,   19,   14,   10,    2,    0,    0,    0,    0,   0,   0],
    [952,   0,   6,   8,   12,   15,    5,    1,    0,    0,    0,    0,    0,    0,   0,   0],
];

fn cdfs() -> [Cdf; CLASSES] {
    std::array::from_fn(|c| Cdf::from_weights(&PRIORS[c]))
}

/// Encodes one length per class. Zero marks an absent band.
pub(crate) fn encode(lengths: &[usize], classes: &[usize]) -> Vec<u8> {
    debug_assert_eq!(lengths.len(), classes.len());
    let mut cdfs = cdfs();
    let mut enc = RangeEncoder::new();
    for (&len, &class) in lengths.iter().zip(classes) {
        debug_assert!(class < CLASSES);
        let bits = usize::BITS - len.leading_zeros();
        let symbol = bits.min(15) as usize;
        enc.encode(&cdfs[class], symbol);
        cdfs[class].update(symbol);
        if bits >= 15 {
            enc.encode_bits(6, bits - 15);
        }
        if bits > 0 {
            let mantissa = len - (1usize << (bits - 1));
            encode_raw(&mut enc, mantissa as u64, bits - 1);
        }
    }
    enc.finish()
}

/// Decodes one length per class. The caller validates them against the payload.
pub(crate) fn decode(data: &[u8], classes: &[usize]) -> Result<Vec<usize>> {
    let mut cdfs = cdfs();
    let mut dec = RangeDecoder::new(data);
    let mut lengths = Vec::with_capacity(classes.len());
    for &class in classes {
        if class >= CLASSES {
            return Err(Error::InvalidData("band class"));
        }
        let mut bits = dec.decode(&cdfs[class]) as u32;
        cdfs[class].update(bits as usize);
        if bits == 15 {
            bits += dec.decode_bits(6);
        }
        if bits == 0 {
            lengths.push(0);
            continue;
        }
        if bits > usize::BITS {
            return Err(Error::InvalidData("band length"));
        }
        let mantissa = decode_raw(&mut dec, bits - 1);
        let value = (1u64 << (bits - 1))
            .checked_add(mantissa)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or(Error::InvalidData("band length"))?;
        lengths.push(value);
    }
    Ok(lengths)
}

/// Codes the low `bits` bits of `v`, least significant chunk first. `bits` may exceed 16.
fn encode_raw(enc: &mut RangeEncoder, mut v: u64, mut bits: u32) {
    while bits > 0 {
        let n = bits.min(16);
        enc.encode_bits(n, (v & ((1u64 << n) - 1)) as u32);
        v >>= n;
        bits -= n;
    }
}

fn decode_raw(dec: &mut RangeDecoder<'_>, mut bits: u32) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    while bits > 0 {
        let n = bits.min(16);
        v |= u64::from(dec.decode_bits(n)) << shift;
        shift += n;
        bits -= n;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_round_trip() {
        let classes: Vec<usize> = (0..42).map(|i| i % CLASSES).collect();
        let values = [
            0,
            1,
            2,
            3,
            127,
            128,
            255,
            256,
            1024,
            32767,
            32768,
            100_000,
            1 << 20,
        ];
        for &start in &[0usize, 1, 5] {
            let lengths: Vec<usize> = classes
                .iter()
                .enumerate()
                .map(|(i, _)| values[(i + start) % values.len()])
                .collect();
            let data = encode(&lengths, &classes);
            assert_eq!(decode(&data, &classes).unwrap(), lengths);
        }
    }

    #[test]
    fn a_run_of_zeros_is_compact() {
        let classes: Vec<usize> = (0..42).map(|i| i % CLASSES).collect();
        let data = encode(&[0; 42], &classes);
        assert!(data.len() <= 8, "{} bytes", data.len());
        assert_eq!(decode(&data, &classes).unwrap(), [0; 42]);
    }

    #[test]
    fn corrupt_class_is_rejected() {
        assert!(decode(&[], &[CLASSES]).is_err());
    }
}
