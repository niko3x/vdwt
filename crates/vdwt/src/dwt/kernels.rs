//! The row-kernel table the streaming transform runs on, selected per ISA.

use super::scalar;
use crate::Isa;

/// `x[i] ±= f(a[i], b[i])` over equal-length rows.
pub(crate) type VerticalFn = fn(&mut [i16], &[i16], &[i16]);
/// `src` → (`low`, `high`).
pub(crate) type ForwardFn = fn(&[i16], &mut [i16], &mut [i16]);
/// (`low`, `high`) → `dst`, with scratch `tmp`.
pub(crate) type InverseFn = fn(&[i16], &[i16], &mut [i16], &mut [i16]);

/// Row kernels for one ISA. Every table computes bit-identical results.
///
/// Vertical kernels work element-wise on equal-length rows, so they vectorize without shuffles;
/// the horizontal kernels run a complete 1-D transform over one row.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub isa: Isa,
    /// `d[i] -= ⌊(s0[i] + s1[i]) / 2⌋` (predict)
    pub v_predict: VerticalFn,
    /// `s[i] += ⌊(d0[i] + d1[i] + 2) / 4⌋` (update)
    pub v_update: VerticalFn,
    /// Inverse of `v_predict`.
    pub v_unpredict: VerticalFn,
    /// Inverse of `v_update`.
    pub v_unupdate: VerticalFn,
    /// Forward 1-D transform of `src` (length `n ≥ 1`) into `low` (`⌈n/2⌉`) and `high` (`⌊n/2⌋`).
    pub h_forward: ForwardFn,
    /// Inverse 1-D transform of `low` and `high` into `dst`; `tmp` holds at least `⌈n/2⌉` samples.
    pub h_inverse: InverseFn,
}

impl Kernels {
    pub fn for_isa(isa: Isa) -> Option<Kernels> {
        match isa {
            Isa::Scalar => Some(scalar::KERNELS),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => super::avx2::kernels(),
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            Isa::Neon => Some(super::neon::KERNELS),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}
