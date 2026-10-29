//! Kernel-family (instruction set) selection.

use std::fmt;

/// A family of SIMD kernels. Every family produces bit-identical results; they differ only in
/// speed, so encoder and decoder may pick different ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable Rust. Always available, and the reference the others are tested against.
    Scalar,
    /// x86-64 AVX2, for development machines.
    Avx2,
    /// AArch64 Advanced SIMD, the production target (Cortex-A55).
    Neon,
}

impl Isa {
    pub const ALL: [Isa; 3] = [Isa::Scalar, Isa::Avx2, Isa::Neon];

    /// Whether this build, on this CPU, can run the family.
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => avx2_detected(),
            Isa::Neon => cfg!(all(target_arch = "aarch64", target_feature = "neon")),
        }
    }

    /// The fastest supported family.
    pub fn best() -> Isa {
        [Isa::Neon, Isa::Avx2]
            .into_iter()
            .find(|isa| isa.is_supported())
            .unwrap_or(Isa::Scalar)
    }

    pub fn supported() -> impl Iterator<Item = Isa> {
        Self::ALL.into_iter().filter(|isa| isa.is_supported())
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Neon => "neon",
        }
    }

    pub fn from_name(name: &str) -> Option<Isa> {
        Self::ALL
            .into_iter()
            .find(|isa| isa.name().eq_ignore_ascii_case(name))
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.name())
    }
}

pub(crate) fn avx2_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
