//! A small deterministic PRNG for tests, self-tests and synthetic content. Not cryptographic.

/// SplitMix64 (Steele, Lea and Flood, 2014).
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`). The modulo bias is negligible for test data.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// Uniform in `lo..=hi`.
    pub fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo) as u64 + 1) as i32
    }

    /// Uniform in `lo..=hi`.
    pub fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below((hi - lo) as u64 + 1) as usize
    }
}
