//! A tiny Inkling-2 with random F32 weights, in either GGUF layout ([`Layout`]): [`N_PHYS`]
//! blocks with the core [`CORE`] looped [`PASSES`] times (16 slots), depth blocks of
//! [`BLOCK`] layers, global layers 3 and 7 among sliding ones (window [`WINDOW`], so short
//! prompts already leave it), relative-bias tables shorter than the prompts, a log-scale floor
//! of [`LOG_FLOOR`] positions, 8 experts (2 a token) and a padded vocabulary (byte tokenizer).
//! [`tensors`] gives the weights under the `inkling2_loop` names, for a reference forward pass
//! in plain Rust.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::{SPECIAL_TOKENS, byte_tokenizer};

use super::{ARCH, ARCH_OLD};

pub const N_PHYS: usize = 8;
pub const CORE: std::ops::Range<usize> = 2..6;
pub const PASSES: usize = 3;
pub const BLOCK: usize = 2;
pub const E: i64 = 64;
/// A head size the attention kernels have.
pub const HEAD: i64 = 32;
pub const N_HEAD: i64 = 4;
pub const N_HEAD_KV: i64 = 2;
pub const N_REL: i64 = 4;
/// The global layers' relative-bias extent (the sliding layers' is the window).
pub const EXTENT: i64 = 16;
pub const WINDOW: i64 = 8;
pub const CONV: i64 = 4;
/// The loop controllers' rank: the gain and forget products' K, which the CUDA linears take in
/// multiples of 32 (the model's is 64).
pub const RANK: i64 = 32;
pub const N_EXPERT: i64 = 8;
pub const N_USED: i64 = 2;
pub const N_FF: i64 = 32;
/// Embedding rows past the vocabulary.
pub const PAD: i64 = 5;
pub const RMS_EPS: f32 = 1e-6;
pub const QK_CLIP: f32 = 3.0;
pub const REL_CLIP: f32 = 1.5;
pub const LOG_FLOOR: u32 = 4;
pub const LOG_ALPHA: f32 = 0.3;
pub const WEIGHTS_SCALE: f32 = 2.0;
pub const SITU_BETA: f32 = 2.0;
pub const SITU_LB: f32 = 3.0;
pub const SOFTCAP: f32 = 10.0;

/// The GGUF layout to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// llama.cpp's `inkling2_loop`.
    Mainline,
    /// The earlier `inkling2`.
    Old,
}

/// The vocabulary size of the byte tokenizer (unpadded).
pub fn n_vocab() -> i64 {
    (256 + SPECIAL_TOKENS.len()) as i64
}

/// Layer slots.
pub fn n_slots() -> usize {
    N_PHYS + (PASSES - 1) * CORE.len()
}

/// Whether physical block `ip` is a sliding-window layer.
pub fn swa(ip: usize) -> bool {
    ip % 4 != 3
}

/// A tensor of the tiny model: name, shape (ggml order) and F32 values.
pub type TinyTensor = (String, Vec<i64>, Vec<f32>);

/// The tiny model's tensors for `seed` under the `inkling2_loop` names, in the order they are
/// written.
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
    // A weight `[n_in, ...]` scaled for unit-variance outputs.
    let w = |n_in: i64| (3.0 / n_in as f32).sqrt();
    let nv = n_vocab() + PAD;
    let (q, kv) = (N_HEAD * HEAD, N_HEAD_KV * HEAD);
    add("token_embd.weight".into(), &[E, nv], 0.5, 0.0);
    add("token_embd_norm.weight".into(), &[E], 0.1, 1.0);
    add("output.weight".into(), &[E, nv], w(E), 0.0);
    add("output_norm.weight".into(), &[E], 0.1, 1.0);
    add("output_res_query.weight".into(), &[E], 0.5, 0.0);
    for ip in 0..N_PHYS {
        let t = |s: &str| format!("blk.{ip}.{s}");
        let swa = swa(ip);
        add(t("attn_q.weight"), &[E, q], w(E), 0.0);
        add(t("attn_k.weight"), &[E, kv], w(E), 0.0);
        add(t("attn_v.weight"), &[E, kv], w(E), 0.0);
        add(t("attn_output.weight"), &[q, E], w(q), 0.0);
        add(t("attn_gate.weight"), &[E, q], w(E), 0.0);
        add(t("attn_q_clip.weight"), &[HEAD, N_HEAD], 0.3, 1.0);
        add(t("attn_sink_proj.weight"), &[E, N_HEAD], w(E), 0.0);
        add(t("attn_rel.weight"), &[E, N_REL * N_HEAD], w(E), 0.0);
        let extent = if swa { WINDOW } else { EXTENT };
        add(t("attn_rel_bias.weight"), &[N_REL, extent], 0.5, 0.0);
        add(t("attn_rel_clip.weight"), &[N_REL, N_HEAD], 0.3, 1.0);
        add(t("attn_k_conv.weight"), &[CONV, kv], 0.4, 0.0);
        add(t("attn_v_conv.weight"), &[CONV, kv], 0.4, 0.0);
        add(t("attn_post_conv.weight"), &[CONV, E], 0.4, 0.0);
        add(t("ffn_post_conv.weight"), &[CONV, E], 0.4, 0.0);
        if swa {
            add(t("attn_norm.weight"), &[E], 0.1, 1.0);
        } else {
            for n in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
                add(t(&format!("{n}.weight")), &[E], 0.1, 1.0);
            }
            add(
                t("attn_k_conv_proj.weight"),
                &[E, CONV * N_HEAD_KV],
                0.3 * w(E),
                0.0,
            );
            add(
                t("attn_v_conv_proj.weight"),
                &[E, CONV * N_HEAD_KV],
                0.3 * w(E),
                0.0,
            );
        }
        if ip > 0 {
            let shape: &[i64] = if swa { &[E] } else { &[E, 3] };
            add(t("attn_res_query.weight"), shape, 0.5, 0.0);
        }
        add(t("ffn_res_query.weight"), &[E], 0.5, 0.0);
        add(t("ffn_norm.weight"), &[E], 0.1, 1.0);
        add(t("ffn_gate_inp.weight"), &[E, N_EXPERT], w(E), 0.0);
        add(t("exp_probs_b.bias"), &[N_EXPERT], 0.1, 0.0);
        add(
            t("ffn_gate_up_exps.weight"),
            &[E, 2 * N_FF, N_EXPERT],
            w(E),
            0.0,
        );
        add(
            t("ffn_down_exps.weight"),
            &[N_FF, E, N_EXPERT],
            w(N_FF),
            0.0,
        );
        if CORE.contains(&ip) {
            add(t("loop_msg_query.weight"), &[E, 4], 0.5, 0.0);
            for c in super::CTRL_NAMES {
                add(
                    t(&format!("loop_{c}_ctrl_down.weight")),
                    &[2 * E, RANK],
                    w(2 * E),
                    0.0,
                );
                add(
                    t(&format!("loop_{c}_ctrl_gain.weight")),
                    &[RANK, E],
                    0.5 * w(RANK),
                    0.0,
                );
                add(
                    t(&format!("loop_{c}_ctrl_forget.weight")),
                    &[RANK, E],
                    0.5 * w(RANK),
                    0.0,
                );
            }
        }
    }
    out
}

/// `tensors` under the earlier layout's names: the controllers renamed, the message queries
/// in the two global tables (core block `i`'s `[q, k, v, ffn]` at q column `2i`, kv columns
/// `2i` and `2i + 1`, q column `2i + 1`).
fn old_tensors(seed: u64) -> Vec<TinyTensor> {
    let n_core = CORE.len();
    let mut qt = vec![0f32; E as usize * 2 * n_core];
    let mut kvt = qt.clone();
    let mut out = Vec::new();
    for (name, shape, v) in tensors(seed) {
        if let Some(ip) = name
            .strip_suffix(".loop_msg_query.weight")
            .and_then(|b| b.strip_prefix("blk."))
        {
            let i = ip.parse::<usize>().unwrap() - CORE.start;
            let e = E as usize;
            let col = |j: usize| &v[j * e..(j + 1) * e];
            qt[2 * i * e..][..e].copy_from_slice(col(0));
            kvt[2 * i * e..][..e].copy_from_slice(col(1));
            kvt[(2 * i + 1) * e..][..e].copy_from_slice(col(2));
            qt[(2 * i + 1) * e..][..e].copy_from_slice(col(3));
            continue;
        }
        let mut name = name;
        for (new, old) in super::CTRL_NAMES.iter().zip(super::CTRL_NAMES_OLD) {
            for part in ["down", "gain", "forget"] {
                name = name.replace(
                    &format!(".loop_{new}_ctrl_{part}."),
                    &format!(".loop_{old}_{part}."),
                );
            }
        }
        out.push((name, shape, v));
    }
    let shape = vec![E, 2 * n_core as i64];
    out.push(("loop_res_query.weight".into(), shape.clone(), qt));
    out.push(("loop_kv_res_query.weight".into(), shape, kvt));
    out
}

/// Writes the tiny model to `path` in `layout`.
pub fn write(path: impl AsRef<Path>, seed: u64, layout: Layout) -> Result<(), GgufError> {
    write_edited(path, seed, layout, &|_| None)
}

/// [`write`] with the architecture keys `edit` gives a value for (the key without the
/// architecture prefix) set to that value.
pub fn write_edited(
    path: impl AsRef<Path>,
    seed: u64,
    layout: Layout,
    edit: &dyn Fn(&str) -> Option<Value>,
) -> Result<(), GgufError> {
    let arch = match layout {
        Layout::Mainline => ARCH,
        Layout::Old => ARCH_OLD,
    };
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String(arch.into()))?;
    g.set("general.name", &Value::String("tiny-inkling2".into()))?;
    byte_tokenizer(&mut g)?;
    let u = Value::U32;
    // The per-layer arrays: per block (mainline) or per slot (the earlier layout).
    let entries: Vec<usize> = match layout {
        Layout::Mainline => (0..N_PHYS).collect(),
        Layout::Old => (0..n_slots())
            .map(|il| {
                if il < N_PHYS {
                    il
                } else {
                    CORE.start + (il - N_PHYS) % CORE.len()
                }
            })
            .collect(),
    };
    let arr = |v: u32| Value::Array(entries.iter().map(|_| Value::U32(v)).collect());
    let mut key = |k: &str, v: Value| {
        let v = edit(k).unwrap_or(v);
        g.set(&format!("{arch}.{k}"), &v).map(|_| ())
    };
    key("block_count", u(entries.len() as u32))?;
    match layout {
        Layout::Mainline => {
            key("loop.count", u(PASSES as u32))?;
            key("loop.start", u(CORE.start as u32))?;
            key("loop.end", u(CORE.end as u32))?;
            key("loop.varying", Value::Bool(false))?;
        }
        Layout::Old => {
            key("num_loops", u(PASSES as u32))?;
            key("loop.core_start", u(CORE.start as u32))?;
            key("loop.core_end", u(CORE.end as u32))?;
        }
    }
    key("loop.gate_rank", u(RANK as u32))?;
    key("context_length", u(4096))?;
    key("embedding_length", u(E as u32))?;
    key("feed_forward_length", u(N_FF as u32))?;
    key("attention.head_count", arr(N_HEAD as u32))?;
    key("attention.head_count_kv", arr(N_HEAD_KV as u32))?;
    key("attention.layer_norm_rms_epsilon", Value::F32(RMS_EPS))?;
    key("expert_count", u(N_EXPERT as u32))?;
    key("expert_used_count", u(N_USED as u32))?;
    key("attention.key_length", u(HEAD as u32))?;
    key("attention.value_length", u(HEAD as u32))?;
    key("attention.key_length_swa", u(HEAD as u32))?;
    key("attention.value_length_swa", u(HEAD as u32))?;
    key("vocab_size", u((n_vocab() + PAD) as u32))?;
    key("vocab_size_unpadded", u(n_vocab() as u32))?;
    key("expert_feed_forward_length", u(N_FF as u32))?;
    key("rope.dimension_count", u(0))?;
    key("attention.sliding_window", u(WINDOW as u32))?;
    key(
        "attention.sliding_window_pattern",
        Value::Array(entries.iter().map(|&ip| Value::Bool(swa(ip))).collect()),
    )?;
    key("shortconv.l_cache", u(CONV as u32))?;
    key("expert_weights_scale", Value::F32(WEIGHTS_SCALE))?;
    key("activation.situ_beta", Value::F32(SITU_BETA))?;
    key("activation.situ_linear_beta", Value::F32(SITU_LB))?;
    key("attn_res.block_size", u(BLOCK as u32))?;
    key("final_logit_softcapping", Value::F32(SOFTCAP))?;
    key("attention.relative_length", u(N_REL as u32))?;
    key("attention.relative_extent", u(EXTENT as u32))?;
    key("attention.qk_clip", Value::F32(QK_CLIP))?;
    key("attention.relative_clip", Value::F32(REL_CLIP))?;
    key("attention.log_scale_floor", u(LOG_FLOOR))?;
    key("attention.log_scale_alpha", Value::F32(LOG_ALPHA))?;
    let all = match layout {
        Layout::Mainline => tensors(seed),
        Layout::Old => old_tensors(seed),
    };
    for (name, shape, v) in all {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
        g.add_tensor(&name, types::F32, &shape, &bytes)?;
    }
    g.write(path)
}
