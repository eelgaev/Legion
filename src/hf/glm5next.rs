//! GLM-5.3-Flash checkpoints (`glm5_next`, `Glm5NextForConditionalGeneration`; the text model
//! only) as the `glm5-next` GGUF llama.cpp's converter writes.
//!
//! What the converter does beyond renaming, and how it is reproduced here:
//! - norms, biases, the hyper-connection `scale` / `base` vectors and the indexer's pool
//!   position embedding (`index_kpool_compress_ape`) are F32 (BF16 widened exactly);
//! - the hyper-connection mixes and the KDA, MLA and indexer projections are used as shipped
//!   (BF16); the router is F32;
//! - `A_log` becomes `ssm_a = -exp(A_log)`;
//! - `kv_b_proj` (`[n_head × (k + v), kv_lora]`) is split per head into `attn_k_b`
//!   (`[k, kv_lora, n_head]`, each head's key rows transposed) and `attn_v_b` (`[kv_lora, v,
//!   n_head]`, each head's value rows as they are).
//!
//! ModelOpt NVFP4 weights (the routed experts, and the dense MLPs of the leading layers) are
//! repacked as for Nemotron-H: the experts stacked with their per-expert scales
//! ([`super::nemotron_h::stack_nvfp4`]), the dense MLPs as single NVFP4 matrices with their
//! scale ([`Builder::quantized`]).
//!
//! The MTP block (`num_nextn_predict_layers`, the checkpoint's layer `num_hidden_layers`) is
//! block `n_layer` of the GGUF, as the converter writes GLM's NextN layer ([`mtp_block`]): the
//! loader leaves it out unless `--spec mtp` drafts with it. The vision tower is not loaded.

use serde_json::Value as Json;

use super::qwen35::text_config;
use super::{Builder, num_f, num_u};
use crate::gguf::Value;
use crate::safetensors::Dtype;

const ARCH: &str = "glm5-next";

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_embd = num_u(tc, "hidden_size")?;
    let n_head = num_u(tc, "num_attention_heads")?;
    let n_expert = num_u(tc, "n_routed_experts")?;
    let ff_exp = num_u(tc, "moe_intermediate_size")?;
    let kv_lora = num_u(tc, "kv_lora_rank")?;
    let k_nope = num_u(tc, "qk_nope_head_dim")?;
    let k_rope = tc
        .get("qk_rope_head_dim")
        .and_then(Json::as_u64)
        .unwrap_or(0);
    let v_dim = num_u(tc, "v_head_dim")?;
    let lin = tc
        .get("linear_attn_config")
        .ok_or("config.json: missing `linear_attn_config`")?;
    let kda_head = num_u(lin, "head_dim")?;
    let strs = |key: &str| -> Result<Vec<String>, String> {
        let a = tc
            .get(key)
            .and_then(Json::as_array)
            .ok_or_else(|| format!("config.json: missing `{key}`"))?;
        if a.len() as u64 != n_layer {
            return Err(format!(
                "`{key}` has {} entries, expected {n_layer}",
                a.len()
            ));
        }
        Ok(a.iter()
            .map(|v| v.as_str().unwrap_or_default().to_string())
            .collect())
    };
    let mla: Vec<bool> = strs("layer_types")?
        .iter()
        .map(|t| t != "linear_attention")
        .collect();
    let first_dense = num_u(tc, "first_k_dense_replace")?;
    let dense: Vec<bool> = match tc.get("mlp_layer_types") {
        Some(_) => strs("mlp_layer_types")?
            .iter()
            .map(|t| t == "dense")
            .collect(),
        None => (0..n_layer).map(|il| il < first_dense).collect(),
    };
    let prefix = if b.has("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };
    let nextn = nextn_layers(
        tc,
        b.has(&format!("{prefix}layers.{n_layer}.eh_proj.weight")),
    );
    let n_all = n_layer + nextn;

    // Metadata: what llama.cpp's converter writes (the MTP block's layer in the per-layer
    // arrays: a DSA layer).
    let u = |v: u64| Value::U32(v as u32);
    let f = |v: f64| Value::F32(v as f32);
    let k = |s: &str| format!("{ARCH}.{s}");
    let limit = num_f(tc, "swiglu_limit")?;
    let per_layer = |v: f64| Value::Array((0..n_all).map(|_| f(v)).collect());
    b.set("general.architecture", Value::String(ARCH.into()))?;
    b.set(&k("block_count"), u(n_all))?;
    b.set(
        &k("context_length"),
        u(num_u(tc, "max_position_embeddings")?),
    )?;
    b.set(&k("embedding_length"), u(n_embd))?;
    b.set(
        &k("feed_forward_length"),
        u(num_u(tc, "intermediate_size")?),
    )?;
    b.set(&k("attention.head_count"), u(n_head))?;
    b.set(
        &k("attention.layer_norm_rms_epsilon"),
        f(num_f(tc, "rms_norm_eps")?),
    )?;
    // The indexer's key LayerNorm: torch's default.
    b.set(&k("attention.layer_norm_epsilon"), f(1e-6))?;
    b.set(&k("expert_count"), u(n_expert))?;
    b.set(
        &k("expert_used_count"),
        u(num_u(tc, "num_experts_per_tok")?),
    )?;
    b.set(&k("expert_group_count"), u(num_u(tc, "n_group")?))?;
    b.set(&k("expert_group_used_count"), u(num_u(tc, "topk_group")?))?;
    match tc.get("scoring_func").and_then(Json::as_str) {
        Some("sigmoid") => b.set(&k("expert_gating_func"), u(2))?,
        other => return Err(format!("scoring_func {other:?}: only sigmoid is supported")),
    }
    let vocab_size = num_u(tc, "vocab_size")?;
    b.set(&k("vocab_size"), u(vocab_size))?;
    b.set(
        &k("attention.head_count_kv"),
        Value::Array(
            mla.iter()
                .copied()
                .chain((0..nextn).map(|_| true))
                .map(|m| Value::I32(m as i32))
                .collect(),
        ),
    )?;
    b.set(&k("attention.q_lora_rank"), u(num_u(tc, "q_lora_rank")?))?;
    b.set(&k("attention.kv_lora_rank"), u(kv_lora))?;
    b.set(&k("rope.dimension_count"), u(k_rope))?;
    b.set(&k("attention.key_length"), u(kv_lora + k_rope))?;
    b.set(&k("attention.value_length"), u(kv_lora))?;
    b.set(&k("attention.key_length_mla"), u(k_nope + k_rope))?;
    b.set(&k("attention.value_length_mla"), u(v_dim))?;
    b.set(
        &k("ssm.conv_kernel"),
        u(num_u(lin, "short_conv_kernel_size")?),
    )?;
    b.set(&k("kda.head_dim"), u(kda_head))?;
    b.set(
        &k("kda.gate_lower_bound"),
        f(num_f(lin, "gate_lower_bound")?),
    )?;
    b.set(
        &k("attention.indexer.head_count"),
        u(num_u(tc, "index_n_heads")?),
    )?;
    b.set(
        &k("attention.indexer.key_length"),
        u(num_u(tc, "index_head_dim")?),
    )?;
    b.set(&k("attention.indexer.top_k"), u(num_u(tc, "index_topk")?))?;
    b.set(&k("attention.indexer.kpool"), u(num_u(tc, "index_kpool")?))?;
    // What llama.cpp implements (and so the GGUF cannot say otherwise): pooled keys, the
    // newest pool always selected, every indexer full.
    let flag = |key: &str| tc.get(key).and_then(Json::as_bool).unwrap_or(true);
    if !flag("index_kpool_compress") || !flag("index_kpool_always_select_tail") {
        return Err(
            "only indexers with compressed key pools that keep the tail are supported".into(),
        );
    }
    if let Some(types) = tc.get("indexer_types").and_then(Json::as_array)
        && types.iter().any(|t| t.as_str() != Some("full"))
    {
        return Err("only `full` indexer_types are supported".into());
    }
    if let Some(share) = tc
        .get("index_share_for_mtp_iteration")
        .and_then(Json::as_bool)
    {
        b.set(&k("attention.indexer.index_share_mtp"), Value::Bool(share))?;
    }
    b.set(&k("hyper_connection.count"), u(num_u(tc, "hc_mult")?))?;
    b.set(
        &k("hyper_connection.sinkhorn_iterations"),
        u(num_u(tc, "hc_sinkhorn_iters")?),
    )?;
    b.set(&k("hyper_connection.epsilon"), f(num_f(tc, "hc_eps")?))?;
    b.set(&k("expert_feed_forward_length"), u(ff_exp))?;
    let n_shared = num_u(tc, "n_shared_experts")?;
    b.set(
        &k("expert_shared_feed_forward_length"),
        u(ff_exp * n_shared),
    )?;
    b.set(&k("expert_shared_count"), u(n_shared))?;
    b.set(&k("leading_dense_block_count"), u(first_dense))?;
    b.set(
        &k("expert_weights_scale"),
        f(num_f(tc, "routed_scaling_factor")?),
    )?;
    b.set(
        &k("expert_weights_norm"),
        Value::Bool(
            tc.get("norm_topk_prob")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        ),
    )?;
    b.set(&k("swiglu_clamp_exp"), per_layer(limit))?;
    b.set(&k("swiglu_clamp_shexp"), per_layer(limit))?;
    b.set(&k("nextn_predict_layers"), u(nextn))?;

    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, config, vocab_size, "glm4")?;
    // A turn ends where the next user turn would start, as llama.cpp's GLM converter sets it
    // (`Glm4MoeModel.set_vocab`); `<|endoftext|>` alone does not end a chat reply.
    if let Some(id) = super::vocab::added_token_id(&dir, "<|user|>")? {
        b.set("tokenizer.ggml.eot_token_id", u(id))?;
    }

    // Tensors.
    b.direct(
        "token_embd.weight",
        &format!("{prefix}embed_tokens.weight"),
        None,
    )?;
    b.direct("output.weight", "lm_head.weight", None)?;
    widen(b, "output_norm.weight", &format!("{prefix}norm.weight"))?;
    for il in 0..n_layer {
        let src = format!("{prefix}layers.{il}.");
        let blk = format!("blk.{il}.");
        let s = |t: &str| format!("{src}{t}");
        let d = |t: &str| format!("{blk}{t}");
        widen(b, &d("attn_norm.weight"), &s("input_layernorm.weight"))?;
        widen(
            b,
            &d("ffn_norm.weight"),
            &s("post_attention_layernorm.weight"),
        )?;
        for hc in ["hc_attn", "hc_ffn"] {
            b.direct(
                &d(&format!("{hc}_fn.weight")),
                &s(&format!("{hc}_fn")),
                None,
            )?;
            widen(
                b,
                &d(&format!("{hc}_scale.weight")),
                &s(&format!("{hc}_scale")),
            )?;
            widen(
                b,
                &d(&format!("{hc}_base.weight")),
                &s(&format!("{hc}_base")),
            )?;
        }
        let a = |t: &str| s(&format!("self_attn.{t}"));
        if mla[il as usize] {
            mla_attention(b, &a, &d, n_head, kv_lora, k_nope + k_rope, v_dim)?;
        } else {
            kda(b, &a, &d)?;
        }
        b.direct(&d("attn_output.weight"), &a("o_proj.weight"), None)?;
        if dense[il as usize] {
            for (g, h) in [
                ("ffn_gate", "gate_proj"),
                ("ffn_up", "up_proj"),
                ("ffn_down", "down_proj"),
            ] {
                b.direct(
                    &d(&format!("{g}.weight")),
                    &s(&format!("mlp.{h}.weight")),
                    None,
                )?;
            }
        } else {
            moe(b, &s, &d, n_embd, n_expert, ff_exp)?;
        }
    }
    if nextn > 0 {
        let dims = MtpDims {
            n_embd,
            n_head,
            n_expert,
            ff: ff_exp,
            kv_lora,
            k_dim: k_nope + k_rope,
            v_dim,
        };
        mtp_block(b, prefix, n_layer, &dims)?;
    }
    Ok(())
}

/// The MTP blocks the GGUF holds: the checkpoint's `num_nextn_predict_layers` when its first
/// MTP layer is there (`present`), at most one (the head drafts with block `n_layer` only).
pub(super) fn nextn_layers(tc: &Json, present: bool) -> u64 {
    let n = tc
        .get("num_nextn_predict_layers")
        .and_then(Json::as_u64)
        .unwrap_or(0);
    if present { n.min(1) } else { 0 }
}

/// `src` (a norm or bias) widened to F32 under `name`.
fn widen(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    b.computed_f32(name, src, super::ggml_shape(&t.shape), |x| x)
}

/// The dimensions of the MTP block's attention and MoE.
pub(super) struct MtpDims {
    pub n_embd: u64,
    pub n_head: u64,
    pub n_expert: u64,
    pub ff: u64,
    pub kv_lora: u64,
    pub k_dim: u64,
    pub v_dim: u64,
}

/// The MTP block, the checkpoint's layer `il` (right after the decoder's), as block `il` of the
/// GGUF the way llama.cpp's converter writes GLM's NextN layer: a DSA layer and a routed-expert
/// MoE with its shared expert like the decoder's, without hyper-connections (no `hc_*`), plus
/// `nextn.eh_proj` (BF16) and the `nextn.enorm` / `nextn.hnorm` / `nextn.shared_head_norm` norms
/// (F32). Its embedding and output head are the model's: the checkpoint ships none of its own.
/// As shipped: the block is BF16 in the NVFP4 release, its 288 routed experts too.
pub(super) fn mtp_block(b: &mut Builder, prefix: &str, il: u64, m: &MtpDims) -> Result<(), String> {
    let src = format!("{prefix}layers.{il}.");
    let blk = format!("blk.{il}.");
    let s = |t: &str| format!("{src}{t}");
    let d = |t: &str| format!("{blk}{t}");
    widen(b, &d("attn_norm.weight"), &s("input_layernorm.weight"))?;
    widen(
        b,
        &d("ffn_norm.weight"),
        &s("post_attention_layernorm.weight"),
    )?;
    let a = |t: &str| s(&format!("self_attn.{t}"));
    mla_attention(b, &a, &d, m.n_head, m.kv_lora, m.k_dim, m.v_dim)?;
    b.direct(&d("attn_output.weight"), &a("o_proj.weight"), None)?;
    moe(b, &s, &d, m.n_embd, m.n_expert, m.ff)?;
    b.direct(&d("nextn.eh_proj.weight"), &s("eh_proj.weight"), None)?;
    for (g, h) in [
        ("nextn.enorm.weight", "enorm.weight"),
        ("nextn.hnorm.weight", "hnorm.weight"),
        ("nextn.shared_head_norm.weight", "shared_head.norm.weight"),
    ] {
        widen(b, &d(g), &s(h))?;
    }
    Ok(())
}

/// A KDA (Kimi delta attention) layer's projections, convolutions, gates and output norm.
fn kda(
    b: &mut Builder,
    a: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
) -> Result<(), String> {
    for (g, h) in [
        ("attn_q", "q_proj"),
        ("attn_k", "k_proj"),
        ("attn_v", "v_proj"),
        ("ssm_beta", "b_proj"),
        ("ssm_f_a", "f_a_proj"),
        ("ssm_f_b", "f_b_proj"),
        ("ssm_g_a", "g_a_proj"),
        ("ssm_g_b", "g_b_proj"),
    ] {
        b.direct(&d(&format!("{g}.weight")), &a(&format!("{h}.weight")), None)?;
    }
    // `[channels, 1, kernel]` F32: ggml's `[kernel, 1, channels]` as stored.
    for x in ["q", "k", "v"] {
        b.direct(
            &d(&format!("ssm_conv1d_{x}.weight")),
            &a(&format!("{x}_conv1d.weight")),
            None,
        )?;
    }
    let (_, t) = b.source(&a("A_log"))?;
    b.computed_f32(&d("ssm_a"), &a("A_log"), vec![t.numel() as u64], |x| {
        x.into_iter().map(|v| -v.exp()).collect()
    })?;
    let (_, t) = b.source(&a("dt_bias"))?;
    b.computed_f32(
        &d("ssm_dt.bias"),
        &a("dt_bias"),
        vec![t.numel() as u64],
        |x| x,
    )?;
    let (_, t) = b.source(&a("o_norm.weight"))?;
    b.computed_f32(
        &d("ssm_norm.weight"),
        &a("o_norm.weight"),
        vec![t.numel() as u64],
        |x| x,
    )
}

/// An MLA (DeepSeek sparse attention) layer: the low-rank q and kv projections, the absorbed
/// per-head key and value matrices, and the lightning indexer.
fn mla_attention(
    b: &mut Builder,
    a: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    n_head: u64,
    kv_lora: u64,
    k_dim: u64,
    v_dim: u64,
) -> Result<(), String> {
    for (g, h) in [
        ("attn_q_a", "q_a_proj"),
        ("attn_q_b", "q_b_proj"),
        ("attn_kv_a_mqa", "kv_a_proj_with_mqa"),
        ("indexer.attn_k", "indexer.wk"),
        ("indexer.attn_q_b", "indexer.wq_b"),
    ] {
        b.direct(&d(&format!("{g}.weight")), &a(&format!("{h}.weight")), None)?;
    }
    b.direct(
        &d("indexer_compressor_gate.weight"),
        &a("indexer.index_kpool_compress_gate"),
        None,
    )?;
    for (g, h) in [
        ("attn_q_a_norm.weight", "q_a_layernorm.weight"),
        ("attn_kv_a_norm.weight", "kv_a_layernorm.weight"),
        ("indexer.k_norm.weight", "indexer.k_norm.weight"),
        ("indexer.k_norm.bias", "indexer.k_norm.bias"),
        // `[kpool, d]`: ggml's `[d, kpool]`.
        (
            "indexer_compressor_ape.weight",
            "indexer.index_kpool_compress_ape",
        ),
        // The head weights: F32 as the converter keeps them.
        ("indexer.proj.weight", "indexer.weights_proj.weight"),
    ] {
        let src = a(h);
        let (_, t) = b.source(&src)?;
        b.computed_f32(&d(g), &src, super::ggml_shape(&t.shape), |x| x)?;
    }

    // kv_b: per head, `k_dim` key rows then `v_dim` value rows of `kv_lora` values.
    let src = a("kv_b_proj.weight");
    let (_, t) = b.source(&src)?;
    let per_head = k_dim + v_dim;
    if t.shape != [(n_head * per_head) as usize, kv_lora as usize] {
        return Err(format!(
            "`{src}`: shape {:?}, expected [{}, {kv_lora}]",
            t.shape,
            n_head * per_head
        ));
    }
    let elem = t.nbytes() / t.numel();
    // Values: each head's value rows as they are.
    let rows: Vec<(u64, u64)> = (0..n_head).map(|h| (h * per_head + k_dim, v_dim)).collect();
    b.gather_rows(
        &d("attn_v_b.weight"),
        &src,
        &rows,
        vec![kv_lora, v_dim, n_head],
    )?;
    // Keys: each head's key rows transposed, `[k_dim, kv_lora]` per head (ggml order).
    let (n_head, kv_lora, k_dim, per_head) = (
        n_head as usize,
        kv_lora as usize,
        k_dim as usize,
        per_head as usize,
    );
    b.computed_raw(
        &d("attn_k_b.weight"),
        &src,
        vec![k_dim as u64, kv_lora as u64, n_head as u64],
        move |x: &[u8]| {
            let mut out = vec![0u8; n_head * kv_lora * k_dim * elem];
            for h in 0..n_head {
                for c in 0..kv_lora {
                    for r in 0..k_dim {
                        let from = ((h * per_head + r) * kv_lora + c) * elem;
                        let to = ((h * kv_lora + c) * k_dim + r) * elem;
                        out[to..to + elem].copy_from_slice(&x[from..from + elem]);
                    }
                }
            }
            out
        },
    )
}

/// A routed-expert layer: router, bias, the stacked experts and the shared expert.
fn moe(
    b: &mut Builder,
    s: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    n_embd: u64,
    n_expert: u64,
    ff: u64,
) -> Result<(), String> {
    b.computed_f32(
        &d("ffn_gate_inp.weight"),
        &s("mlp.gate.weight"),
        vec![n_embd, n_expert],
        |x| x,
    )?;
    b.computed_f32(
        &d("exp_probs_b.bias"),
        &s("mlp.gate.e_score_correction_bias"),
        vec![n_expert],
        |x| x,
    )?;
    for (g, h, shape) in [
        ("ffn_gate_exps", "gate_proj", vec![n_embd, ff, n_expert]),
        ("ffn_up_exps", "up_proj", vec![n_embd, ff, n_expert]),
        ("ffn_down_exps", "down_proj", vec![ff, n_embd, n_expert]),
    ] {
        let srcs: Vec<String> = (0..n_expert)
            .map(|e| s(&format!("mlp.experts.{e}.{h}.weight")))
            .collect();
        let name = d(&format!("{g}.weight"));
        let (_, t) = b.source(&srcs[0])?;
        let scale = srcs[0].replace(".weight", ".weight_scale");
        if t.dtype == Dtype::U8 && b.has(&scale) {
            super::nemotron_h::stack_nvfp4(b, &name, &srcs, shape)?;
        } else {
            b.stack(&name, &srcs, shape)?;
        }
    }
    for (g, h) in [
        ("ffn_gate_shexp", "gate_proj"),
        ("ffn_up_shexp", "up_proj"),
        ("ffn_down_shexp", "down_proj"),
    ] {
        b.direct(
            &d(&format!("{g}.weight")),
            &s(&format!("mlp.shared_experts.{h}.weight")),
            None,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
