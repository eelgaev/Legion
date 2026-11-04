//! A tiny `nemotron_h_moe` (Nemotron-H MoE layout) with random weights: Mamba-2, MoE and
//! attention blocks in Nemotron 3.5's order (`M E M * E M`), width 64, routed experts, shared
//! expert and LM head in NVFP4 with their per-tensor `.scale` (as llama.cpp's converter writes
//! an NVFP4 checkpoint), the Mamba-2 and attention projections BF16 (as the converter writes
//! the checkpoint's FP8 and BF16 ones), the rest F32, and the byte-level tokenizer under the
//! `pixtral` pre-tokenizer. Like the released GGUF it carries the MTP head as block `n_layer`
//! (`nextn.*`, attention and MoE sub-layers, 16-bit experts without scales).

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::{self, ffi, types};
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::*;

const ARCH: &str = "nemotron_h_moe";
/// Block kinds per layer: Mamba-2, MoE, Mamba-2, attention, MoE, Mamba-2.
pub const PATTERN: &str = "MEM*EM";
const E: i64 = 64;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HEAD_DIM: i64 = 16;
/// Mamba-2: heads × head size = d_inner.
const SSM_HEADS: i64 = 8;
/// d_inner + 2 × groups × state is a multiple of 128 (ggml's CUDA conv kernel needs it).
const SSM_HEAD_DIM: i64 = 16;
/// 128: one of the state sizes ggml's CUDA scan kernel has (96, 128, 256).
const SSM_STATE: i64 = 128;
const SSM_GROUPS: i64 = 2;
const D_CONV: i64 = 4;
const N_EXPERT: i64 = 8;
const N_EXPERT_USED: i64 = 2;
/// Expert and shared-expert sizes: multiples of NVFP4's 64-element block.
const N_FF: i64 = 64;
const N_FF_SHEXP: i64 = 128;

struct Builder {
    g: GgufWriter,
    rng: Rng,
}

impl Builder {
    fn key(&mut self, k: &str, v: Value) -> Result<(), GgufError> {
        self.g.set(&format!("{ARCH}.{k}"), &v)?;
        Ok(())
    }

    /// `bias + scale * U(-1, 1)`, stored as `ty`.
    fn tensor(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: &[i64],
        scale: f32,
        bias: f32,
    ) -> Result<(), GgufError> {
        let n: i64 = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| bias + scale * (self.rng.next_f64() as f32 * 2.0 - 1.0))
            .collect();
        let bytes = if ty == types::F32 {
            v.iter().flat_map(|x| x.to_ne_bytes()).collect()
        } else {
            ggml::quantize(ty, &v, shape[0] as usize)
        };
        self.g.add_tensor(name, ty, shape, &bytes)?;
        Ok(())
    }

    /// A weight `[n_in, n_out, ..]` scaled for unit-variance outputs.
    fn weight(&mut self, name: &str, shape: &[i64]) -> Result<(), GgufError> {
        self.typed_weight(name, types::F32, shape)
    }

    fn typed_weight(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: &[i64],
    ) -> Result<(), GgufError> {
        let s = (3.0 / shape[0] as f32).sqrt();
        self.tensor(name, ty, shape, s, 0.0)
    }

    /// An NVFP4 weight `{base}.weight` plus its second-level scale `{base}.scale` (`[n]`, one
    /// per expert, or `[1]`). The stored values are 4x too large; the scale brings them back.
    fn nvfp4(&mut self, base: &str, shape: &[i64], n_scales: i64) -> Result<(), GgufError> {
        let s = 4.0 * (3.0 / shape[0] as f32).sqrt();
        self.tensor(&format!("{base}.weight"), types::NVFP4, shape, s, 0.0)?;
        self.tensor(
            &format!("{base}.scale"),
            types::F32,
            &[n_scales],
            0.05,
            0.25,
        )
    }

    fn norm(&mut self, name: &str, shape: &[i64]) -> Result<(), GgufError> {
        self.tensor(name, types::F32, shape, 0.1, 1.0)
    }

    fn mamba(&mut self, il: usize) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        let d_inner = SSM_HEADS * SSM_HEAD_DIM;
        let conv_dim = d_inner + 2 * SSM_GROUPS * SSM_STATE;
        let d_in_proj = 2 * d_inner + 2 * SSM_GROUPS * SSM_STATE + SSM_HEADS;
        self.typed_weight(&t("ssm_in.weight"), types::BF16, &[E, d_in_proj])?;
        self.tensor(
            &t("ssm_conv1d.weight"),
            types::F32,
            &[D_CONV, conv_dim],
            0.5,
            0.0,
        )?;
        self.tensor(&t("ssm_conv1d.bias"), types::F32, &[conv_dim], 0.1, 0.0)?;
        self.tensor(&t("ssm_dt.bias"), types::F32, &[SSM_HEADS], 0.5, -1.0)?;
        // A = -exp(A_log) < 0: the state decays.
        self.tensor(&t("ssm_a"), types::F32, &[1, SSM_HEADS], 0.5, -1.0)?;
        self.tensor(&t("ssm_d"), types::F32, &[1, SSM_HEADS], 0.5, 1.0)?;
        self.norm(&t("ssm_norm.weight"), &[d_inner / SSM_GROUPS, SSM_GROUPS])?;
        self.typed_weight(&t("ssm_out.weight"), types::BF16, &[d_inner, E])
    }

    fn attention(&mut self, il: usize) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        let bf16 = types::BF16;
        self.typed_weight(&t("attn_q.weight"), bf16, &[E, N_HEAD * HEAD_DIM])?;
        self.typed_weight(&t("attn_k.weight"), bf16, &[E, N_HEAD_KV * HEAD_DIM])?;
        self.typed_weight(&t("attn_v.weight"), bf16, &[E, N_HEAD_KV * HEAD_DIM])?;
        self.typed_weight(&t("attn_output.weight"), bf16, &[N_HEAD * HEAD_DIM, E])
    }

    /// The MTP head, block `il`.
    fn mtp(&mut self, il: usize) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        for n in [
            "nextn.enorm.weight",
            "nextn.hnorm.weight",
            "nextn.shared_head_norm.weight",
            "attn_norm.weight",
            "post_attention_norm.weight",
        ] {
            self.norm(&t(n), &[E])?;
        }
        self.typed_weight(&t("nextn.eh_proj.weight"), types::BF16, &[2 * E, E])?;
        self.attention(il)?;
        self.weight(&t("ffn_gate_inp.weight"), &[E, N_EXPERT])?;
        self.tensor(&t("exp_probs_b.bias"), types::F32, &[N_EXPERT], 0.1, 0.0)?;
        let bf16 = types::BF16;
        self.typed_weight(&t("ffn_up_exps.weight"), bf16, &[E, N_FF, N_EXPERT])?;
        self.typed_weight(&t("ffn_down_exps.weight"), bf16, &[N_FF, E, N_EXPERT])?;
        self.typed_weight(&t("ffn_up_shexp.weight"), bf16, &[E, N_FF_SHEXP])?;
        self.typed_weight(&t("ffn_down_shexp.weight"), bf16, &[N_FF_SHEXP, E])
    }

    fn moe(&mut self, il: usize) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        self.weight(&t("ffn_gate_inp.weight"), &[E, N_EXPERT])?;
        self.tensor(&t("exp_probs_b.bias"), types::F32, &[N_EXPERT], 0.1, 0.0)?;
        self.nvfp4(&t("ffn_up_exps"), &[E, N_FF, N_EXPERT], N_EXPERT)?;
        self.nvfp4(&t("ffn_down_exps"), &[N_FF, E, N_EXPERT], N_EXPERT)?;
        self.nvfp4(&t("ffn_up_shexp"), &[E, N_FF_SHEXP], 1)?;
        self.nvfp4(&t("ffn_down_shexp"), &[N_FF_SHEXP, E], 1)
    }
}

/// Writes the tiny model to `path`.
pub fn write(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng::new(seed),
    };
    let n_layer = PATTERN.len();
    // The MTP block has attention and an FFN.
    let all = format!("{PATTERN}+");
    b.g.set("general.architecture", &Value::String(ARCH.into()))?;
    b.g.set("general.name", &Value::String("tiny-nemotron-h".into()))?;
    byte_tokenizer(&mut b.g)?;
    b.g.set("tokenizer.ggml.pre", &Value::String("pixtral".into()))?;
    let u = Value::U32;
    let per_layer =
        |f: &dyn Fn(char) -> u32| Value::Array(all.chars().map(|c| Value::U32(f(c))).collect());
    b.key("block_count", u(n_layer as u32 + 1))?;
    b.key("nextn_predict_layers", u(1))?;
    b.key("context_length", u(4096))?;
    b.key("embedding_length", u(E as u32))?;
    b.key(
        "feed_forward_length",
        per_layer(&|c| if c == 'E' || c == '+' { N_FF as u32 } else { 0 }),
    )?;
    b.key("attention.head_count", u(N_HEAD as u32))?;
    b.key(
        "attention.head_count_kv",
        per_layer(&|c| {
            if c == '*' || c == '+' {
                N_HEAD_KV as u32
            } else {
                0
            }
        }),
    )?;
    b.key("attention.key_length", u(HEAD_DIM as u32))?;
    b.key("attention.value_length", u(HEAD_DIM as u32))?;
    b.key("attention.layer_norm_rms_epsilon", Value::F32(1e-5))?;
    b.key("attention.layer_norm_epsilon", Value::F32(1e-5))?;
    b.key("rope.freq_base", Value::F32(10_000.0))?;
    b.key("expert_count", u(N_EXPERT as u32))?;
    b.key("expert_used_count", u(N_EXPERT_USED as u32))?;
    b.key("expert_feed_forward_length", u(N_FF as u32))?;
    b.key("expert_shared_feed_forward_length", u(N_FF_SHEXP as u32))?;
    b.key("expert_shared_count", u(1))?;
    b.key("expert_weights_norm", Value::Bool(true))?;
    b.key("expert_weights_scale", Value::F32(2.5))?;
    b.key("ssm.conv_kernel", u(D_CONV as u32))?;
    b.key("ssm.state_size", u(SSM_STATE as u32))?;
    b.key("ssm.group_count", u(SSM_GROUPS as u32))?;
    b.key("ssm.time_step_rank", u(SSM_HEADS as u32))?;
    b.key("ssm.inner_size", u((SSM_HEADS * SSM_HEAD_DIM) as u32))?;

    let n_vocab = (256 + SPECIAL_TOKENS.len()) as i64;
    b.tensor("token_embd.weight", types::F32, &[E, n_vocab], 1.0, 0.0)?;
    b.norm("output_norm.weight", &[E])?;
    b.nvfp4("output", &[E, n_vocab], 1)?;
    for (il, kind) in PATTERN.chars().enumerate() {
        b.norm(&format!("blk.{il}.attn_norm.weight"), &[E])?;
        match kind {
            'M' => b.mamba(il)?,
            '*' => b.attention(il)?,
            _ => b.moe(il)?,
        }
    }
    b.mtp(n_layer)?;
    // A W4A16 checkpoint: every NVFP4 weight keeps 8-bit activations (`prec_a4 = false`), as
    // the safetensors path tags them.
    let mut nvfp4 = vec!["output.weight".to_string()];
    for (il, kind) in PATTERN.chars().enumerate() {
        if kind == 'E' {
            for t in [
                "ffn_up_exps",
                "ffn_down_exps",
                "ffn_up_shexp",
                "ffn_down_shexp",
            ] {
                nvfp4.push(format!("blk.{il}.{t}.weight"));
            }
        }
    }
    b.g.set(
        "general.tensor_extra.prec_a4",
        &Value::Array(nvfp4.iter().map(|_| Value::Bool(false)).collect()),
    )?;
    b.g.set(
        "general.tensor_extra.name",
        &Value::Array(nvfp4.into_iter().map(Value::String).collect()),
    )?;
    b.g.write(path)
}
