//! Qwen3.5 / 3.6 / 3.8-27B (`qwen35` dense and `qwen35moe` in GGUF): the Qwen3-Next family.
//! Every `full_attention_interval`th layer is gated full attention with multi-section RoPE; the
//! others are gated DeltaNet (linear attention with a recurrent state). In `qwen35moe` every
//! layer ends in a 256-expert MoE plus a sigmoid-gated shared expert; in `qwen35` in a plain
//! SwiGLU FFN. The blocks are the core's ([`crate::model::core`]); this module stacks them,
//! following llama.cpp's `src/models/qwen35moe.cpp` and `qwen35.cpp` (MIT) op for op.

mod mtp;
pub mod tiny;

use std::ops::Range;

use crate::ggml::{Context, Graph, Tensor};
use crate::gguf::GgufFile;
use crate::model::core::arch::same_as;
use crate::model::core::mtp::MtpGraph;
use crate::model::core::state::KvPool;
use crate::model::core::{
    Arch, ArchRef, FfnFusion, GroupShape, Hparams, Model, ModelError, Pass, StatePool,
};

/// The Qwen3.5 architecture: its hyperparameters are all in [`Hparams`].
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35;

/// The hyperparameters of a Qwen3-Next style GGUF (`qwen35`, `qwen35moe`, and the stack
/// `qwen4exp` builds on) under the metadata prefix `arch`; `extra` gives the architecture,
/// called after the shared keys are read and before the shapes.
pub(crate) fn parse_next(
    g: &GgufFile,
    arch: &str,
    moe: bool,
    extra: impl FnOnce() -> Result<ArchRef, ModelError>,
) -> Result<Hparams, ModelError> {
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| -> Result<u64, ModelError> {
        g.get_u64(&key(k))
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let f = |k: &str| -> Result<f32, ModelError> {
        g.get(&key(k))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let sections = g
        .get(&key("rope.dimension_sections"))
        .and_then(|v| v.as_array().map(|a| a.to_vec()))
        .ok_or_else(|| ModelError::Metadata(key("rope.dimension_sections")))?;
    let mut rope_sections = [0i32; 4];
    for (i, s) in sections.iter().take(4).enumerate() {
        rope_sections[i] = s.as_u64().unwrap_or(0) as i32;
    }
    let nextn = g.get_u64(&key("nextn_predict_layers")).unwrap_or(0);
    let n_vocab = g
        .tensor("token_embd.weight")
        .map(|t| t.shape[1] as i64)
        .ok_or_else(|| ModelError::Tensor("token_embd.weight".into()))?;
    let arch_ref = extra()?;
    Ok(Hparams {
        n_layer: (u("block_count")? - nextn) as usize,
        n_embd: u("embedding_length")? as i64,
        n_head: u("attention.head_count")? as i64,
        n_head_kv: u("attention.head_count_kv")? as i64,
        head_dim: u("attention.key_length")? as i64,
        n_rot: u("rope.dimension_count")? as i32,
        rope_sections,
        rope_freq_base: f("rope.freq_base")?,
        n_ctx_train: u("context_length")? as i32,
        rms_eps: f("attention.layer_norm_rms_epsilon")?,
        moe,
        n_expert: if moe { u("expert_count")? as i64 } else { 0 },
        n_expert_used: if moe {
            u("expert_used_count")? as i64
        } else {
            0
        },
        full_attention_interval: u("full_attention_interval").unwrap_or(4) as usize,
        ssm_d_conv: u("ssm.conv_kernel")? as i64,
        ssm_d_state: u("ssm.state_size")? as i64,
        ssm_n_group: u("ssm.group_count")? as i64,
        ssm_dt_rank: u("ssm.time_step_rank")? as i64,
        ssm_d_inner: u("ssm.inner_size")? as i64,
        n_vocab,
        arch: arch_ref,
    })
}

/// The hyperparameters of a `qwen35` (dense) or `qwen35moe` GGUF.
pub(crate) fn parse(g: &GgufFile, arch: &str) -> Result<Hparams, ModelError> {
    parse_next(g, arch, arch == "qwen35moe", || Ok(ArchRef::new(Qwen35)))
}

/// Whether layer `il` of a Qwen3-Next stack is gated DeltaNet (every layer but each
/// `full_attention_interval`th).
pub(crate) fn is_delta_net(hp: &Hparams, il: usize) -> bool {
    !(il + 1).is_multiple_of(hp.full_attention_interval)
}

impl Arch for Qwen35 {
    fn name(&self) -> &'static str {
        "qwen35"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, hp: &Hparams, il: usize) -> bool {
        // Block `n_layer` is the MTP head: full attention.
        il < hp.n_layer && is_delta_net(hp, il)
    }

    fn state_captures(&self) -> bool {
        true
    }

    /// The MTP head reads the target's last layer output after the final norm (see `mtp.rs`).
    fn mtp_capture_normed(&self) -> bool {
        true
    }

    fn mtp(&self) -> bool {
        true
    }

    fn mtp_head(&self, hp: &Hparams, gguf: &GgufFile) -> Result<ArchRef, ModelError> {
        mtp::check_head(hp, gguf)?;
        Ok(mtp::head_arch())
    }

    fn build_mtp<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        m.build_mtp_qwen35(ctx, kv, rec, shape)
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        mut x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let (ctx, n) = (pass.ctx, pass.n);
        let hp = &m.hp;
        let mut hidden = Vec::new();
        // Dense prompt steps on the opt-in prefill kernels (`--fp8-act a8-dynamic`, `--nvfp4-a4`):
        // the residual adds in the output projections' epilogues and the post-attention norm in
        // the W4A4 SwiGLU's quantization, each with ggml's bits (#749).
        let fuse = !pass.tap.on && !hp.moe && m.linear.residual_fusion(n);
        for il in layers {
            // A gated DeltaNet layer's input feeds BF16 (beta / alpha) and FP8 matmuls.
            let both = fuse && hp.is_recurrent(il);
            let h = m.norm_mul_with(pass, x, m.lw(il, "attn_norm.weight")?, both);
            if fuse {
                let x_attn = if hp.is_recurrent(il) {
                    m.delta_net_res(pass, il, h, Some(x))?
                } else {
                    m.attention_res(pass, il, h, Some(x))?
                };
                let norm_w = m.lw(il, "post_attention_norm.weight")?;
                let h = m.norm_mul(pass, x_attn, norm_w);
                let fusion = FfnFusion {
                    res: Some(x_attn),
                    norm: Some((x_attn, norm_w, hp.rms_eps)),
                };
                x = m.ffn_fused(ctx, il, h, fusion)?;
                if capture.contains(&il) {
                    hidden.push((il, x));
                }
                continue;
            }
            let attn = if hp.is_recurrent(il) {
                m.delta_net(pass, il, h)?
            } else {
                m.attention(pass, il, h)?
            };
            let x_attn = ctx.add(attn, x);
            pass.tap.tap("attn_residual", il, x_attn);
            let h = m.norm_mul(pass, x_attn, m.lw(il, "post_attention_norm.weight")?);
            pass.tap.tap("attn_post_norm", il, h);
            let ffn = if hp.moe {
                m.moe(ctx, &pass.tap, &pass.graph, il, h, n, pass.prompt_only())?
            } else {
                m.ffn(ctx, il, h)?
            };
            pass.tap.tap("ffn_out", il, ffn);
            x = ctx.add(ffn, x_attn);
            pass.tap.tap("l_out", il, x);
            if capture.contains(&il) {
                hidden.push((il, x));
            }
        }
        Ok((x, hidden))
    }
}
