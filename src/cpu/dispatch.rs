//! Process-wide runtime dispatch for the host kernels: one place picks the implementation.

use std::sync::OnceLock;

use super::{Q8Activations, QuantFormat, scalar};

/// One quantized matmul: packed weights, activations, rows, token-major output. Shapes are
/// validated by the caller; a kernel stays memory-safe (it panics) if they are not.
pub(super) type QuantKernel = fn(&[u8], &Q8Activations, usize, &mut [f32]);
pub(super) type Bf16Dot = fn(&[u16], &[u16]) -> f32;

/// Instruction-set tiers, lowest first. Each tier includes the ones below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Isa {
    /// Portable Rust; the reference.
    Scalar,
    /// AArch64 Advanced SIMD (NEON), part of the architecture baseline.
    Neon,
    /// [`Isa::Neon`] plus the ARMv8.2-A dot-product extension (`sdot`).
    NeonDotprod,
    /// AVX2 + FMA + F16C (`vpmaddubsw` integer dots).
    Avx2,
    /// [`Isa::Avx2`] plus AVX-VNNI (VEX `vpdpbusd`): Intel cores without AVX-512 (Alder Lake
    /// and later client parts).
    AvxVnni,
    /// AVX-512 F/BW/VL + VNNI (`vpdpbusd` on 256-bit vectors).
    Avx512Vnni,
    /// [`Isa::Avx512Vnni`] plus AVX-512 BF16 (`vdpbf16ps`) for the BF16 dot.
    Avx512Bf16,
    /// [`Isa::Avx512Vnni`] on 512-bit vectors, two groups per `vpdpbusd` (`vdpbf16ps` for the
    /// BF16 dot where present). Opt-in: never chosen automatically (Zen 4 splits 512-bit ops
    /// into two 256-bit halves); `legion cpu-bench` times it.
    Avx512Zmm,
    /// POWER AltiVec/VSX (`vmsummbm` integer dots; every ppc64le CPU has it).
    Vsx,
}

impl Isa {
    pub const ALL: [Isa; 9] = [
        Isa::Scalar,
        Isa::Neon,
        Isa::NeonDotprod,
        Isa::Avx2,
        Isa::AvxVnni,
        Isa::Avx512Vnni,
        Isa::Avx512Bf16,
        Isa::Avx512Zmm,
        Isa::Vsx,
    ];

    /// The tier's short name, as reports print it.
    pub const fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Neon => "neon",
            Isa::NeonDotprod => "dotprod",
            Isa::Avx2 => "avx2",
            Isa::AvxVnni => "avxvnni",
            Isa::Avx512Vnni => "avx512vnni",
            Isa::Avx512Bf16 => "avx512bf16",
            Isa::Avx512Zmm => "avx512zmm",
            Isa::Vsx => "vsx",
        }
    }

    /// Whether the tier runs only when forced ([`super::matmul_q8_with_isa`]), never as the
    /// automatic choice.
    pub const fn opt_in(self) -> bool {
        matches!(self, Isa::Avx512Zmm)
    }

    pub fn from_name(name: &str) -> Option<Isa> {
        Isa::ALL
            .into_iter()
            .find(|i| i.name().eq_ignore_ascii_case(name.trim()))
    }
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub isa: Isa,
    pub q8_0: QuantKernel,
    pub q5_1: QuantKernel,
    pub q4_k: QuantKernel,
    pub q5_k: QuantKernel,
    pub q6_k: QuantKernel,
    pub iq4_xs: QuantKernel,
    pub bf16_dot: Bf16Dot,
}

impl Kernels {
    pub fn quant(&self, format: QuantFormat) -> QuantKernel {
        match format {
            QuantFormat::Q8_0 => self.q8_0,
            QuantFormat::Q5_1 => self.q5_1,
            QuantFormat::Q4K => self.q4_k,
            QuantFormat::Q5K => self.q5_k,
            QuantFormat::Q6K => self.q6_k,
            QuantFormat::Iq4Xs => self.iq4_xs,
        }
    }
}

pub(super) const SCALAR: Kernels = Kernels {
    isa: Isa::Scalar,
    q8_0: scalar::q8_0,
    q5_1: scalar::q5_1,
    q4_k: scalar::q4_k,
    q5_k: scalar::q5_k,
    q6_k: scalar::q6_k,
    iq4_xs: scalar::iq4_xs,
    bf16_dot: scalar::bf16_dot,
};

static KERNELS: OnceLock<Kernels> = OnceLock::new();

/// The tier [`super::matmul_q8`] and [`super::matmul_bf16`] use in this process: the highest
/// supported one that is not [`Isa::opt_in`] (`legion cpu-bench` times every tier).
pub fn detected_isa() -> Isa {
    kernels().isa
}

/// Every tier this CPU can run, lowest first (always starts with [`Isa::Scalar`]).
pub fn supported_isas() -> Vec<Isa> {
    Isa::ALL
        .into_iter()
        .filter(|&i| kernels_for(i).is_some())
        .collect()
}

pub(super) fn kernels() -> Kernels {
    *KERNELS.get_or_init(|| {
        let best = supported_isas()
            .into_iter()
            .filter(|i| !i.opt_in())
            .max()
            .unwrap_or(Isa::Scalar);
        kernels_for(best).unwrap_or(SCALAR)
    })
}

/// The kernels of `isa`, if this CPU supports it.
pub(super) fn kernels_for(isa: Isa) -> Option<Kernels> {
    if isa == Isa::Scalar {
        return Some(SCALAR);
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::x86::kernels_for(isa)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::aarch64::kernels_for(isa)
    }
    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    {
        super::power::kernels_for(isa)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "powerpc64", target_endian = "little")
    )))]
    {
        None
    }
}
