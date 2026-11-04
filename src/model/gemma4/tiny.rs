//! A tiny `gemma4` with random weights and every part of the architecture: sliding-window and
//! global layers with their own head sizes and kv-head counts (window [`WINDOW`], so short
//! prompts already leave it), a global layer without `attn_v` (keys as values), shared-KV layers
//! at the end, MoE layers (fused `ffn_gate_up_exps`, per-expert down scales, the router's input
//! scale) beside the dense FFN, per-layer token embeddings, per-layer output scales and the
//! final logit soft-cap. Weights F32 (the per-layer table F16), byte-level tokenizer.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::{self, ffi, types};
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::*;

const ARCH: &str = "gemma4";
/// Per layer: sliding-window (`s`) or global (`g`). The last [`SHARED`] layers read the caches
/// of layers 2 (sliding) and 3 (global).
pub const PATTERN: &str = "sgsgssg";
pub const SHARED: usize = 3;
/// Layers with routed experts.
pub const MOE_LAYERS: [usize; 2] = [1, 4];
/// The global layer without a value projection.
pub const K_EQ_V_LAYER: usize = 3;
pub const WINDOW: u32 = 8;
const E: i64 = 64;
const N_HEAD: i64 = 4;
/// Head sizes the paged CUDA kernels have, so a GPU run takes them.
const HEAD_SWA: i64 = 128;
const HEAD_FULL: i64 = 256;
const KV_SWA: i64 = 2;
const KV_FULL: i64 = 1;
const N_FF: i64 = 96;
const N_EXPERT: i64 = 8;
const N_EXPERT_USED: i64 = 2;
const N_FF_EXP: i64 = 32;
const PER_LAYER: i64 = 8;

struct Builder {
    g: GgufWriter,
    rng: Rng,
    /// The global layers' kv heads.
    kv_full: i64,
    /// [`MOE_LAYERS`] have routed experts (else every layer is dense).
    moe: bool,
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
        let s = (3.0 / shape[0] as f32).sqrt();
        self.tensor(name, types::F32, shape, s, 0.0)
    }

    fn norm(&mut self, name: &str, shape: &[i64]) -> Result<(), GgufError> {
        self.tensor(name, types::F32, shape, 0.1, 1.0)
    }

    fn layer(&mut self, il: usize, swa: bool, ple: bool) -> Result<(), GgufError> {
        let t = |s: &str| format!("blk.{il}.{s}");
        let (d, kv) = if swa {
            (HEAD_SWA, KV_SWA)
        } else {
            (HEAD_FULL, self.kv_full)
        };
        for n in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
            "post_norm",
        ] {
            self.norm(&t(&format!("{n}.weight")), &[E])?;
        }
        self.weight(&t("attn_q.weight"), &[E, N_HEAD * d])?;
        // Small Q/K norm weights keep the unscaled attention logits moderate (unit ones over
        // 128-256-wide heads make the softmax pick one key, and every rounding a different one).
        self.tensor(&t("attn_q_norm.weight"), types::F32, &[d], 0.05, 0.3)?;
        self.weight(&t("attn_k.weight"), &[E, kv * d])?;
        if il != K_EQ_V_LAYER {
            self.weight(&t("attn_v.weight"), &[E, kv * d])?;
        }
        self.tensor(&t("attn_k_norm.weight"), types::F32, &[d], 0.05, 0.3)?;
        self.weight(&t("attn_output.weight"), &[N_HEAD * d, E])?;
        self.weight(&t("ffn_gate.weight"), &[E, N_FF])?;
        self.weight(&t("ffn_up.weight"), &[E, N_FF])?;
        self.weight(&t("ffn_down.weight"), &[N_FF, E])?;
        if ple {
            self.weight(&t("inp_gate.weight"), &[E, PER_LAYER])?;
            self.weight(&t("proj.weight"), &[PER_LAYER, E])?;
        }
        self.tensor(&t("layer_output_scale.weight"), types::F32, &[1], 0.2, 0.8)?;
        if self.moe && MOE_LAYERS.contains(&il) {
            for n in ["pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"] {
                self.norm(&t(&format!("{n}.weight")), &[E])?;
            }
            self.weight(&t("ffn_gate_inp.weight"), &[E, N_EXPERT])?;
            self.norm(&t("ffn_gate_inp.scale"), &[E])?;
            self.weight(&t("ffn_gate_up_exps.weight"), &[E, 2 * N_FF_EXP, N_EXPERT])?;
            self.weight(&t("ffn_down_exps.weight"), &[N_FF_EXP, E, N_EXPERT])?;
            self.norm(&t("ffn_down_exps.scale"), &[N_EXPERT])?;
        }
        Ok(())
    }
}

/// Writes the tiny model to `path`.
pub fn write(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, seed, true, false, KV_FULL)
}

/// The tiny model with the 31B's partial RoPE on the global layers: the first
/// [`ROTATED_PAIRS`] dimension pairs rotate, the others' frequency factors are 1e30 (they stay
/// put), so [`K_EQ_V_LAYER`] can keep a compact cache (`--gemma4-compact-global-kv`).
pub fn write_partial_rope(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, seed, false, true, KV_FULL)
}

/// [`write_partial_rope`] with `kv_full` kv heads on the global layers (a cache that tensor
/// parallelism splits by heads).
pub fn write_partial_rope_heads(
    path: impl AsRef<Path>,
    seed: u64,
    kv_full: i64,
) -> Result<(), GgufError> {
    write_with(path, seed, false, true, kv_full)
}

/// Rotated dimension pairs of the global heads in [`write_partial_rope`]'s model.
pub const ROTATED_PAIRS: usize = 32;

/// The tiny model without per-layer embeddings (as Gemma 4 26B and 31B), which can run as
/// several pipeline stages.
pub fn write_without_ple(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, seed, false, false, KV_FULL)
}

/// The tiny model as Gemma 4 31B: no per-layer embeddings, no experts, and two kv heads on
/// every layer, so tensor parallelism over two machines ([`crate::tp_net`]) can split it.
pub fn write_dense(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    write_model(path, seed, false, false, 2, false)
}

fn write_with(
    path: impl AsRef<Path>,
    seed: u64,
    ple: bool,
    partial_rope: bool,
    kv_full: i64,
) -> Result<(), GgufError> {
    write_model(path, seed, ple, partial_rope, kv_full, true)
}

fn write_model(
    path: impl AsRef<Path>,
    seed: u64,
    ple: bool,
    partial_rope: bool,
    kv_full: i64,
    moe: bool,
) -> Result<(), GgufError> {
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng::new(seed),
        kv_full,
        moe,
    };
    let n_layer = PATTERN.len();
    let swa: Vec<bool> = PATTERN.chars().map(|c| c == 's').collect();
    b.g.set("general.architecture", &Value::String(ARCH.into()))?;
    b.g.set("general.name", &Value::String("tiny-gemma4".into()))?;
    byte_tokenizer(&mut b.g)?;
    let u = Value::U32;
    b.key("block_count", u(n_layer as u32))?;
    b.key("context_length", u(4096))?;
    b.key("embedding_length", u(E as u32))?;
    b.key(
        "feed_forward_length",
        Value::Array(swa.iter().map(|_| Value::I32(N_FF as i32)).collect()),
    )?;
    b.key("attention.head_count", u(N_HEAD as u32))?;
    b.key(
        "attention.head_count_kv",
        Value::Array(
            swa.iter()
                .map(|&s| Value::I32(if s { KV_SWA } else { kv_full } as i32))
                .collect(),
        ),
    )?;
    b.key("attention.key_length", u(HEAD_FULL as u32))?;
    b.key("attention.value_length", u(HEAD_FULL as u32))?;
    b.key("attention.key_length_swa", u(HEAD_SWA as u32))?;
    b.key("attention.value_length_swa", u(HEAD_SWA as u32))?;
    b.key("rope.dimension_count", u(HEAD_FULL as u32))?;
    b.key("rope.dimension_count_swa", u(HEAD_SWA as u32))?;
    b.key("rope.freq_base", Value::F32(1_000_000.0))?;
    b.key("rope.freq_base_swa", Value::F32(10_000.0))?;
    b.key("attention.layer_norm_rms_epsilon", Value::F32(1e-6))?;
    b.key("attention.sliding_window", u(WINDOW))?;
    b.key(
        "attention.sliding_window_pattern",
        Value::Array(swa.iter().map(|&s| Value::Bool(s)).collect()),
    )?;
    b.key("attention.shared_kv_layers", u(SHARED as u32))?;
    if ple {
        b.key("embedding_length_per_layer_input", u(PER_LAYER as u32))?;
    }
    b.key("final_logit_softcapping", Value::F32(30.0))?;
    if moe {
        b.key("expert_count", u(N_EXPERT as u32))?;
        b.key("expert_used_count", u(N_EXPERT_USED as u32))?;
        b.key("expert_feed_forward_length", u(N_FF_EXP as u32))?;
    }

    let n_vocab = (256 + SPECIAL_TOKENS.len()) as i64;
    b.tensor("token_embd.weight", types::F32, &[E, n_vocab], 0.2, 0.0)?;
    b.norm("output_norm.weight", &[E])?;
    // Proportional RoPE: some dimension pairs barely rotate.
    if partial_rope {
        // Drawn all the same (unused), so the other weights stay those of the plain model.
        for _ in 0..HEAD_FULL / 2 {
            b.rng.next_f64();
        }
        let f: Vec<u8> = (0..HEAD_FULL as usize / 2)
            .map(|i| if i < ROTATED_PAIRS { 1.0f32 } else { 1e30 })
            .flat_map(|x| x.to_ne_bytes())
            .collect();
        b.g.add_tensor("rope_freqs.weight", types::F32, &[HEAD_FULL / 2], &f)?;
    } else {
        b.tensor("rope_freqs.weight", types::F32, &[HEAD_FULL / 2], 0.5, 1.0)?;
    }
    if ple {
        let per_layer = PER_LAYER * n_layer as i64;
        b.tensor(
            "per_layer_token_embd.weight",
            types::F16,
            &[per_layer, n_vocab],
            1.0,
            0.0,
        )?;
        b.weight("per_layer_model_proj.weight", &[E, per_layer])?;
        b.norm("per_layer_proj_norm.weight", &[PER_LAYER])?;
    }
    for (il, &s) in swa.iter().enumerate() {
        b.layer(il, s, ple)?;
    }
    b.g.write(path)
}

/// Writes a tiny `gemma4-assistant` for the tiny model to `path`: two layers (sliding-window,
/// global) of width 32 with four query heads over the target's caches, a 64-wide FFN, its own
/// tied output head, and the projections from and back to the target's width.
pub fn write_assistant(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    const ARCH_A: &str = "gemma4-assistant";
    const EA: i64 = 32;
    const HEADS: i64 = 4;
    const FF: i64 = 64;
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng::new(seed),
        kv_full: KV_FULL,
        moe: true,
    };
    let swa = [true, false];
    let u = Value::U32;
    b.g.set("general.architecture", &Value::String(ARCH_A.into()))?;
    b.g.set(
        "general.name",
        &Value::String("tiny-gemma4-assistant".into()),
    )?;
    byte_tokenizer(&mut b.g)?;
    let mut key = |k: &str, v: Value| b.g.set(&format!("{ARCH_A}.{k}"), &v).map(|_| ());
    key("block_count", u(swa.len() as u32))?;
    key("nextn_predict_layers", u(swa.len() as u32))?;
    key("context_length", u(4096))?;
    key("embedding_length", u(EA as u32))?;
    key("embedding_length_out", u(E as u32))?;
    key("feed_forward_length", u(FF as u32))?;
    key("attention.head_count", u(HEADS as u32))?;
    key("attention.head_count_kv", u(1))?;
    key("attention.key_length", u(HEAD_FULL as u32))?;
    key("attention.value_length", u(HEAD_FULL as u32))?;
    key("attention.key_length_swa", u(HEAD_SWA as u32))?;
    key("attention.value_length_swa", u(HEAD_SWA as u32))?;
    key("rope.dimension_count", u(HEAD_FULL as u32))?;
    key("rope.dimension_count_swa", u(HEAD_SWA as u32))?;
    key("rope.freq_base", Value::F32(1_000_000.0))?;
    key("rope.freq_base_swa", Value::F32(10_000.0))?;
    key("attention.layer_norm_rms_epsilon", Value::F32(1e-6))?;
    key("attention.sliding_window", u(WINDOW))?;
    key(
        "attention.sliding_window_pattern",
        Value::Array(swa.iter().map(|&s| Value::Bool(s)).collect()),
    )?;
    key("attention.shared_kv_layers", u(swa.len() as u32))?;
    let n_vocab = (256 + SPECIAL_TOKENS.len()) as i64;
    b.tensor("token_embd.weight", types::F32, &[EA, n_vocab], 0.5, 0.0)?;
    b.norm("output_norm.weight", &[EA])?;
    b.tensor("rope_freqs.weight", types::F32, &[HEAD_FULL / 2], 0.5, 1.0)?;
    b.weight("nextn.pre_projection.weight", &[2 * E, EA])?;
    b.weight("nextn.post_projection.weight", &[EA, E])?;
    for (il, &s) in swa.iter().enumerate() {
        let t = |n: &str| format!("blk.{il}.{n}");
        let d = if s { HEAD_SWA } else { HEAD_FULL };
        for n in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            b.norm(&t(&format!("{n}.weight")), &[EA])?;
        }
        b.weight(&t("attn_q.weight"), &[EA, HEADS * d])?;
        b.tensor(&t("attn_q_norm.weight"), types::F32, &[d], 0.05, 0.3)?;
        b.weight(&t("attn_output.weight"), &[HEADS * d, EA])?;
        b.weight(&t("ffn_gate.weight"), &[EA, FF])?;
        b.weight(&t("ffn_up.weight"), &[EA, FF])?;
        b.weight(&t("ffn_down.weight"), &[FF, EA])?;
        b.tensor(&t("layer_output_scale.weight"), types::F32, &[1], 0.2, 0.8)?;
    }
    b.g.write(path)
}
