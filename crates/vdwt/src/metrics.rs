//! Objective quality metrics.
//!
//! PSNR is a sanity check, not the decider for perceptual choices (DESIGN.md §9.3).

use crate::{Frame, Plane};

/// Sum of squared differences between two planes of the same size.
pub fn sse_u8(a: &Plane<u8>, b: &Plane<u8>) -> u64 {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "plane size mismatch"
    );
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::avx2_detected() {
        #[target_feature(enable = "avx2")]
        fn sse_avx2(a: &Plane<u8>, b: &Plane<u8>) -> u64 {
            sse_any(a, b)
        }
        // SAFETY: the CPU has AVX2.
        return unsafe { sse_avx2(a, b) };
    }
    sse_any(a, b)
}

/// [`sse_u8`]'s loop: rows summed in 32 bits (at most 16384 samples of at most 255² cannot overflow
/// them), which vectorizes.
#[inline(always)]
fn sse_any(a: &Plane<u8>, b: &Plane<u8>) -> u64 {
    a.rows()
        .zip(b.rows())
        .map(|(ra, rb)| {
            u64::from(
                ra.iter()
                    .zip(rb)
                    .map(|(&x, &y)| (i32::from(x) - i32::from(y)).pow(2) as u32)
                    .sum::<u32>(),
            )
        })
        .sum()
}

/// PSNR in dB for a mean squared error; infinite for identical signals.
pub fn psnr(mse: f64, peak: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    }
}

pub fn psnr_u8(a: &Plane<u8>, b: &Plane<u8>) -> f64 {
    psnr(sse_u8(a, b) as f64 / (a.width() * a.height()) as f64, 255.0)
}

/// Per-plane PSNR of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuvPsnr {
    pub y: f64,
    pub u: f64,
    pub v: f64,
}

impl YuvPsnr {
    pub fn between(a: &Frame, b: &Frame) -> Self {
        Self {
            y: psnr_u8(&a.y, &b.y),
            u: psnr_u8(&a.u, &b.u),
            v: psnr_u8(&a.v, &b.v),
        }
    }

    /// The common 6:1:1 luma-weighted combination.
    pub fn weighted(&self) -> f64 {
        (6.0 * self.y + self.u + self.v) / 8.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_of_known_error() {
        let a = Plane::<u8>::new(4, 4);
        let mut b = Plane::<u8>::new(4, 4);
        assert_eq!(psnr_u8(&a, &b), f64::INFINITY);
        b.fill(1);
        assert!((psnr_u8(&a, &b) - 48.1308).abs() < 1e-3);
    }
}
