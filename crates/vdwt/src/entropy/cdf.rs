//! Adaptive symbol distributions for the range coder.

/// Probabilities are fractions of [`PROB_ONE`].
pub const PROB_BITS: u32 = 15;
pub const PROB_ONE: u32 = 1 << PROB_BITS;
/// The smallest probability any symbol can have, so adaptation never makes a symbol uncodable.
pub const MIN_PROB: u32 = 4;
/// The largest alphabet.
pub const MAX_SYMBOLS: usize = 16;

/// An adaptive distribution over 2 to 16 symbols. Symbol `s` owns `cum[s] .. cum[s + 1]` of
/// [`PROB_ONE`].
///
/// Adaptation follows AV1. Each coded symbol pulls the distribution toward itself by
/// `2^-rate`, and `rate` grows over the first 32 symbols, so a fresh context learns quickly and
/// then settles. The pull targets a distribution that still leaves [`MIN_PROB`] to every other
/// symbol. That keeps every probability at or above `MIN_PROB` exactly, despite rounding: a
/// floor-rounded step toward a target never undershoots the smaller of the old and target
/// probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cdf {
    cum: [u16; MAX_SYMBOLS + 1],
    n: u8,
    count: u8,
}

impl Cdf {
    /// A distribution proportional to `weights` (each below 2^48), with every symbol given at
    /// least [`MIN_PROB`]. Integer-only, so encoder and decoder build identical tables everywhere.
    pub fn from_weights(weights: &[u64]) -> Cdf {
        let n = weights.len();
        assert!((2..=MAX_SYMBOLS).contains(&n), "alphabet of {n} symbols");
        debug_assert!(weights.iter().all(|&w| w < 1 << 48));
        let spare = u64::from(PROB_ONE - MIN_PROB * n as u32);
        let total = weights.iter().sum::<u64>().max(1);
        let mut freq = [0u64; MAX_SYMBOLS];
        for (f, &w) in freq.iter_mut().zip(weights) {
            *f = w * spare / total;
        }
        // Rounding leftovers go to the most likely symbol (the first, on ties).
        let top = (0..n)
            .max_by_key(|&i| (weights[i], std::cmp::Reverse(i)))
            .unwrap_or(0);
        freq[top] += spare - freq[..n].iter().sum::<u64>();
        let mut cum = [0u16; MAX_SYMBOLS + 1];
        let mut acc = 0u64;
        for s in 0..n {
            acc += u64::from(MIN_PROB) + freq[s];
            cum[s + 1] = acc as u16;
        }
        debug_assert_eq!(acc, u64::from(PROB_ONE));
        Cdf {
            cum,
            n: n as u8,
            count: 0,
        }
    }

    pub fn uniform(n: usize) -> Cdf {
        Self::from_weights(&[1; MAX_SYMBOLS][..n])
    }

    /// Alphabet size.
    #[inline]
    pub fn symbols(&self) -> usize {
        usize::from(self.n)
    }

    /// Cumulative probability of symbols `0 .. s`.
    #[inline]
    pub fn cum(&self, s: usize) -> u32 {
        u32::from(self.cum[s])
    }

    /// Probability of `s`, out of [`PROB_ONE`].
    #[inline]
    pub fn prob(&self, s: usize) -> u32 {
        self.cum(s + 1) - self.cum(s)
    }

    /// The cost of coding `s` now, in bits.
    pub fn cost(&self, s: usize) -> f64 {
        f64::from(PROB_BITS) - f64::from(self.prob(s)).log2()
    }

    /// Adapts toward `s` after it was coded.
    #[inline]
    pub fn update(&mut self, s: usize) {
        #[cfg(target_arch = "x86_64")]
        if self.n == 16 && crate::cpu::avx2_detected() {
            // SAFETY: AVX2 is present; `cum` holds 17 u16s and the kernel reads/writes the
            // first 16.
            unsafe { self.update16_avx2(s) };
            return;
        }
        self.update_scalar(s);
    }

    #[inline]
    fn update_scalar(&mut self, s: usize) {
        let n = self.symbols();
        let rate =
            3 + u32::from(self.count > 15) + u32::from(self.count > 31) + if n > 3 { 2 } else { 1 };
        for i in 1..n {
            let c = i32::from(self.cum[i]);
            let target = if i <= s {
                MIN_PROB * i as u32
            } else {
                PROB_ONE - MIN_PROB * (n - i) as u32
            };
            self.cum[i] = (c + ((target as i32 - c) >> rate)) as u16;
        }
        self.count = (self.count + 1).min(32);
    }

    /// The alphabet-16 update in one AVX2 vector. `cum[0]` stays 0; `cum[16]` is not read or
    /// written. The two target lines are `4·i` below the symbol and `32704 + 4·i` above it;
    /// both and the shifted step fit i16 for every lane.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn update16_avx2(&mut self, s: usize) {
        use std::arch::x86_64::*;
        let rate = 3 + u32::from(self.count > 15) + u32::from(self.count > 31) + 2;
        let c = unsafe { _mm256_loadu_si256(self.cum.as_ptr().cast()) };
        let idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let below = _mm256_cmpgt_epi16(_mm256_set1_epi16(s as i16 + 1), idx);
        let t1 = _mm256_slli_epi16(idx, 2);
        let t2 = _mm256_add_epi16(_mm256_set1_epi16(32704), t1);
        let target = _mm256_blendv_epi8(t2, t1, below);
        let diff = _mm256_sub_epi16(target, c);
        let step = match rate {
            5 => _mm256_srai_epi16::<5>(diff),
            6 => _mm256_srai_epi16::<6>(diff),
            _ => _mm256_srai_epi16::<7>(diff),
        };
        let next = _mm256_add_epi16(c, step);
        unsafe { _mm256_storeu_si256(self.cum.as_mut_ptr().cast(), next) };
        self.count = (self.count + 1).min(32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn check_invariants(cdf: &Cdf) {
        assert_eq!(cdf.cum(0), 0);
        assert_eq!(cdf.cum(cdf.symbols()), PROB_ONE);
        for s in 0..cdf.symbols() {
            assert!(
                cdf.prob(s) >= MIN_PROB,
                "symbol {s} fell to {}",
                cdf.prob(s)
            );
        }
    }

    #[test]
    fn from_weights_is_proportional_and_complete() {
        let cdf = Cdf::from_weights(&[1, 3]);
        check_invariants(&cdf);
        assert!((f64::from(cdf.prob(1)) / f64::from(cdf.prob(0)) - 3.0).abs() < 0.01);
        check_invariants(&Cdf::from_weights(&[0; 16]));
        check_invariants(&Cdf::from_weights(&[1 << 40, 0, 0, 1]));
    }

    #[test]
    fn adaptation_keeps_every_symbol_codable() {
        let mut rng = SplitMix64::new(9);
        for n in 2..=MAX_SYMBOLS {
            let mut cdf = Cdf::uniform(n);
            // Long runs of one symbol drive the others to the floor, then a switch drives them back.
            for round in 0..40 {
                let favourite = rng.below(n as u64) as usize;
                for _ in 0..(200 + round * 50) {
                    let s = if rng.below(50) == 0 {
                        rng.below(n as u64) as usize
                    } else {
                        favourite
                    };
                    cdf.update(s);
                    check_invariants(&cdf);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn cdf_speed() {
        let mut a = Cdf::from_weights(&[1; 16]);
        let mut b = a;
        let n = 20_000_000u32;
        let t = std::time::Instant::now();
        let mut sink = 0u64;
        for i in 0..n {
            a.update_scalar((i * 7 % 16) as usize);
            sink += u64::from(a.cum[15]);
        }
        let scalar = t.elapsed().as_secs_f64() / f64::from(n);
        let t = std::time::Instant::now();
        #[cfg(target_arch = "x86_64")]
        if crate::cpu::avx2_detected() {
            for i in 0..n {
                unsafe { b.update16_avx2((i * 7 % 16) as usize) };
                sink += u64::from(b.cum[15]);
            }
        }
        std::hint::black_box(sink);
        let vector = t.elapsed().as_secs_f64() / f64::from(n);
        eprintln!(
            "cdf update: scalar {:.3} ns, vector {:.3} ns, {:.2}x",
            scalar * 1e9,
            vector * 1e9,
            scalar / vector
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_update_matches_scalar() {
        if !crate::cpu::avx2_detected() {
            return;
        }
        let mut rng = SplitMix64::new(17);
        for symbols in [2usize, 3, 4, 8, 15] {
            let mut a = Cdf::from_weights(&[symbols as u64; 16]);
            let mut b = a;
            for i in 0..2000 {
                let s = rng.below(16) as usize;
                a.update(s);
                unsafe { b.update16_avx2(s) };
                assert_eq!(a, b, "symbol {s} at step {i}");
            }
        }
    }

    #[test]
    fn adaptation_learns_a_skewed_source() {
        let mut cdf = Cdf::uniform(2);
        for _ in 0..500 {
            cdf.update(0);
        }
        // The shift-based step stalls within 2^rate of the target, so the rare symbol settles
        // near 64/32768 (about 9 bits) rather than at the floor.
        assert!(
            cdf.cost(0) < 0.01 && cdf.cost(1) > 8.5,
            "{} / {}",
            cdf.cost(0),
            cdf.cost(1)
        );
    }
}
