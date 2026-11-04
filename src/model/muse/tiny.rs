//! A tiny `muse-glimmer` with random F32 weights: [`N_LAYER`] layers with the GGUF's numeric
//! sliding-window pattern ([`PATTERN`]: layer 3 global and NoPE, the others sliding with RoPE,
//! window [`WINDOW`] so short prompts already leave it), the attention output gate, QK norms,
//! pre and post norms, the logit scale and the soft-cap; byte-level tokenizer. [`tensors`]
//! gives the weights as written, for a reference forward pass in plain Rust.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::*;

use super::ARCH;

pub const N_LAYER: usize = 6;
pub const PATTERN: u32 = 4;
pub const WINDOW: u32 = 8;
pub const E: i64 = 64;
pub const N_HEAD: i64 = 4;
pub const N_HEAD_KV: i64 = 1;
/// A head size the paged CUDA kernels have, so a GPU run takes them.
pub const HEAD: i64 = 128;
pub const N_FF: i64 = 96;
pub const RMS_EPS: f32 = 1e-5;
pub const ROPE_BASE: f32 = 10_000.0;
pub const LOGIT_SCALE: f32 = 0.5;
pub const SOFTCAP: f32 = 20.0;

/// The vocabulary size of the byte tokenizer.
pub fn n_vocab() -> i64 {
    (256 + SPECIAL_TOKENS.len()) as i64
}

/// A tensor of the tiny model: name, shape (ggml order) and F32 values.
pub type TinyTensor = (String, Vec<i64>, Vec<f32>);

/// The tiny model's tensors for `seed`, in the order they are written.
pub fn tensors(seed: u64) -> Vec<TinyTensor> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    let mut add = |name: String, shape: &[i64], scale: f32, bias: f32| {
        let n: i64 = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| bias + scale * (rng.next_f64() as f32 * 2.0 - 1.0))
            .collect();
        out.push((name, shape.to_vec(), v));
    };
    // A weight `[n_in, n_out]` scaled for unit-variance outputs.
    let w = |shape: &[i64]| (3.0 / shape[0] as f32).sqrt();
    let n_vocab = n_vocab();
    add("token_embd.weight".into(), &[E, n_vocab], 0.5, 0.0);
    add("output_norm.weight".into(), &[E], 0.1, 1.0);
    add("output.weight".into(), &[E, n_vocab], w(&[E]), 0.0);
    for il in 0..N_LAYER {
        let t = |s: &str| format!("blk.{il}.{s}");
        for n in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            add(t(&format!("{n}.weight")), &[E], 0.1, 1.0);
        }
        add(t("attn_q.weight"), &[E, N_HEAD * HEAD], w(&[E]), 0.0);
        add(t("attn_k.weight"), &[E, N_HEAD_KV * HEAD], w(&[E]), 0.0);
        add(t("attn_v.weight"), &[E, N_HEAD_KV * HEAD], w(&[E]), 0.0);
        add(t("attn_gate.weight"), &[E, N_HEAD * HEAD], w(&[E]), 0.0);
        // The query norm carries the checkpoint's qk scale; the key norm is ones.
        add(t("attn_q_norm.weight"), &[HEAD], 0.0, 0.7);
        add(t("attn_k_norm.weight"), &[HEAD], 0.0, 1.0);
        add(
            t("attn_output.weight"),
            &[N_HEAD * HEAD, E],
            w(&[N_HEAD * HEAD]),
            0.0,
        );
        add(t("ffn_gate.weight"), &[E, N_FF], w(&[E]), 0.0);
        add(t("ffn_up.weight"), &[E, N_FF], w(&[E]), 0.0);
        add(t("ffn_down.weight"), &[N_FF, E], w(&[N_FF]), 0.0);
    }
    out
}

/// Writes the tiny model to `path`.
pub fn write(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String(ARCH.into()))?;
    g.set("general.name", &Value::String("tiny-muse-glimmer".into()))?;
    byte_tokenizer(&mut g)?;
    let u = Value::U32;
    let mut key = |k: &str, v: Value| g.set(&format!("{ARCH}.{k}"), &v).map(|_| ());
    key("block_count", u(N_LAYER as u32))?;
    key("context_length", u(4096))?;
    key("embedding_length", u(E as u32))?;
    key("feed_forward_length", u(N_FF as u32))?;
    key("attention.head_count", u(N_HEAD as u32))?;
    key("attention.head_count_kv", u(N_HEAD_KV as u32))?;
    key("attention.key_length", u(HEAD as u32))?;
    key("attention.value_length", u(HEAD as u32))?;
    key("rope.freq_base", Value::F32(ROPE_BASE))?;
    key("attention.layer_norm_rms_epsilon", Value::F32(RMS_EPS))?;
    key("final_logit_softcapping", Value::F32(SOFTCAP))?;
    key("logit_scale", Value::F32(LOGIT_SCALE))?;
    key("attention.sliding_window", u(WINDOW))?;
    key("attention.sliding_window_pattern", u(PATTERN))?;
    for (name, shape, v) in tensors(seed) {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
        g.add_tensor(&name, types::F32, &shape, &bytes)?;
    }
    g.write(path)
}
