//! Model implementations: each turns loaded weights into ggml graphs. [`core`] holds what every
//! architecture shares; each architecture module plugs into it through [`core::Arch`].

pub mod bf16_experts;
pub mod cluster_select;
pub mod core;
pub mod deepseek41;
pub mod dense_mv;
pub mod fp8;
pub mod fp8_act;
pub mod fp8_block;
pub mod gemma4;
pub mod glm5next;
pub mod moe_ffn;
pub mod muse;
pub mod nemotron_h;
pub mod nvfp4_a4;
pub mod nvfp4_mm;
pub mod q8_mmq;
pub mod qwen35;
pub mod qwen4;
pub mod wgmma_gemm;

use self::core::{Hparams, ModelError};
use crate::gguf::GgufFile;

/// The GGUF architectures Legion runs, for messages.
pub const ARCHITECTURES: &str = "qwen35, qwen35moe, qwen4exp, nemotron_h, nemotron_h_moe, gemma4, \
     glm5-next, deepseek41 or muse-glimmer";

/// The hyperparameters of a GGUF of architecture `arch` (`general.architecture`), read by the
/// architecture's module.
pub(crate) fn parse_hparams(g: &GgufFile, arch: &str) -> Result<Hparams, ModelError> {
    match arch {
        "qwen35" | "qwen35moe" => qwen35::parse(g, arch),
        "qwen4exp" => qwen4::parse(g, arch),
        "nemotron_h" | "nemotron_h_moe" => nemotron_h::parse(g, arch),
        "gemma4" => gemma4::parse(g),
        "glm5-next" => glm5next::parse(g),
        deepseek41::ARCH => deepseek41::parse(g),
        muse::ARCH => muse::parse(g),
        _ => Err(ModelError::Architecture(arch.into())),
    }
}

#[cfg(test)]
mod tests;
