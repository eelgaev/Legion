//! Qwen3.8-Flash-Next checkpoints (`qwen4_exp`, `Qwen4ExpForConditionalGeneration`; the text
//! model only) as the `qwen4exp` GGUF llama.cpp's converter writes: `Qwen4ExpTextModel` in
//! `conversion/qwen4exp.py` on top of the Qwen3.5 classes in `conversion/qwen.py` (llama.cpp,
//! MIT), whose DeltaNet, attention and MoE transforms are [`super::qwen35`]'s.
//!
//! What the converter does beyond renaming, and how it is reproduced here:
//! - every `*norm.weight` but the DeltaNet output norm gets `+ 1` (F32), which here covers the
//!   hyper-connection mixers' `hc_norm`, the attention `q/k_norm`, the indexer's
//!   `q/k_layernorm` and the PLE `norm_key/query/conv`;
//! - the indexer's one `index_qk_proj` is split into `indexer.q_proj` (the first
//!   `indexer_n_heads × indexer_head_dim` rows) and `indexer.k_proj` (the rest);
//! - `ple.conv1d` drops its size-1 dim; `ple_layer_ids` is 1-based;
//! - the hyper-connection inject weights are F32 here (BF16 widened exactly), as in the released
//!   GGUFs, so that prompt steps take the fused mixer norm;
//! - the hash constants (`layer_multipliers`, `ngram_heads_offsets/vocab_sizes`, I64 tensors)
//!   become metadata, read exactly;
//! - the n-gram table's shards (`ngram_embedding.shard_<i>`) are one `per_layer_token_embd`
//!   in index order. Here the table is not copied: the shards lie back to back in one file,
//!   so the tensor is that file range as shipped (FP8 E4M3 bytes as `I8`, or BF16), read row
//!   by row from the file by the n-gram reader like a GGUF's table (never loaded, never in
//!   VRAM). Rows of shards stored out of index order are found through
//!   [`crate::model::qwen4::KEY_PLE_BLOCK_ORDER`]; an FP8 table's scale is
//!   [`crate::model::qwen4::KEY_PLE_FP8_SCALE`].
//!
//! ModelOpt NVFP4 routed experts are repacked and stacked with their per-expert scales as for
//! Nemotron-H ([`super::nemotron_h::stack_nvfp4`]); every other weight is used as shipped
//! (BF16). The vision tower is not loaded. The MTP head (`mtp.*`) is a model of its own,
//! opened by [`build_mtp`] (`--spec mtp` without `--spec-draft`), with its block-scaled FP8
//! experts kept as shipped.

use serde_json::Value as Json;

use super::qwen35::{linear_attention, text_config, v_order};
use super::{Builder, num_f, num_u};
use crate::ggml::types;
use crate::gguf::Value;
use crate::model::qwen4::{KEY_PLE_BLOCK_ORDER, KEY_PLE_BLOCK_ROWS, KEY_PLE_FP8_SCALE};
use crate::safetensors::Dtype;

const ARCH: &str = "qwen4exp";

/// The text config's `key` as u64, else `default`.
fn opt_u(tc: &Json, key: &str, default: u64) -> u64 {
    tc.get(key).and_then(Json::as_u64).unwrap_or(default)
}

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_embd = num_u(tc, "hidden_size")?;
    let n_head = num_u(tc, "num_attention_heads")?;
    let head_dim = opt_u(tc, "head_dim", n_embd / n_head);
    let nk = num_u(tc, "linear_num_key_heads")?;
    let nv = num_u(tc, "linear_num_value_heads")?;
    let dk = num_u(tc, "linear_key_head_dim")?;
    let dv = num_u(tc, "linear_value_head_dim")?;
    if nv % nk != 0 {
        return Err(format!(
            "linear_num_value_heads {nv} is not a multiple of linear_num_key_heads {nk}"
        ));
    }
    let n_expert = num_u(tc, "num_experts")?;
    let ff = num_u(tc, "moe_intermediate_size")?;
    let rope = tc
        .get("rope_parameters")
        .or_else(|| tc.get("rope_scaling"))
        .cloned()
        .unwrap_or(Json::Null);
    let rope_get = |k: &str| rope.get(k).or_else(|| tc.get(k));
    let theta = rope_get("rope_theta")
        .and_then(Json::as_f64)
        .ok_or("config.json: missing `rope_theta`")?;
    let partial = rope_get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(0.25);
    let mut sections: Vec<i32> = rope_get("mrope_section")
        .and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Json::as_i64)
                .map(|v| v as i32)
                .collect()
        })
        .unwrap_or_else(|| vec![11, 11, 10]);
    sections.resize(4, 0);
    let interval = opt_u(tc, "full_attention_interval", 4);
    let recurrent: Vec<bool> = match tc.get("layer_types").and_then(Json::as_array) {
        Some(t) if t.len() as u64 == n_layer => t
            .iter()
            .map(|v| v.as_str() == Some("linear_attention"))
            .collect(),
        Some(t) => {
            return Err(format!(
                "layer_types has {} entries, expected {n_layer}",
                t.len()
            ));
        }
        None => (0..n_layer).map(|il| (il + 1) % interval != 0).collect(),
    };
    let prefix = if b.has("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };

    // Metadata: the Qwen3.5 MoE keys, then qwen4exp's.
    let u = |v: u64| Value::U32(v as u32);
    let k = |s: &str| format!("{ARCH}.{s}");
    b.set("general.architecture", Value::String(ARCH.into()))?;
    b.set(&k("block_count"), u(n_layer))?;
    b.set(
        &k("context_length"),
        u(num_u(tc, "max_position_embeddings")?),
    )?;
    b.set(&k("embedding_length"), u(n_embd))?;
    b.set(&k("attention.head_count"), u(n_head))?;
    b.set(
        &k("attention.head_count_kv"),
        u(num_u(tc, "num_key_value_heads")?),
    )?;
    b.set(
        &k("rope.dimension_sections"),
        Value::Array(sections.into_iter().map(Value::I32).collect()),
    )?;
    b.set(&k("rope.freq_base"), Value::F32(theta as f32))?;
    b.set(
        &k("attention.layer_norm_rms_epsilon"),
        Value::F32(num_f(tc, "rms_norm_eps")? as f32),
    )?;
    b.set(&k("expert_count"), u(n_expert))?;
    b.set(
        &k("expert_used_count"),
        u(num_u(tc, "num_experts_per_tok")?),
    )?;
    b.set(&k("attention.key_length"), u(head_dim))?;
    b.set(&k("attention.value_length"), u(head_dim))?;
    b.set(
        &k("ssm.conv_kernel"),
        u(num_u(tc, "linear_conv_kernel_dim")?),
    )?;
    b.set(&k("ssm.state_size"), u(dk))?;
    b.set(&k("ssm.group_count"), u(nk))?;
    b.set(&k("ssm.time_step_rank"), u(nv))?;
    b.set(&k("ssm.inner_size"), u(dv * nv))?;
    b.set(&k("full_attention_interval"), u(interval))?;
    b.set(
        &k("rope.dimension_count"),
        u((head_dim as f64 * partial) as u64),
    )?;
    b.set(&k("expert_feed_forward_length"), u(ff))?;
    b.set(
        &k("expert_shared_feed_forward_length"),
        u(num_u(tc, "shared_expert_intermediate_size")?),
    )?;
    b.set(&k("hyper_connection.count"), u(num_u(tc, "hc_count")?))?;
    b.set(&k("hyper_connection.low_rank"), u(num_u(tc, "hc_lowrank")?))?;
    let idx_heads = num_u(tc, "indexer_n_heads")?;
    let idx_dim = num_u(tc, "indexer_head_dim")?;
    b.set(&k("attention.indexer.head_count"), u(idx_heads))?;
    b.set(&k("attention.indexer.key_length"), u(idx_dim))?;
    b.set(
        &k("attention.indexer.top_k"),
        u(num_u(tc, "indexer_budget")?),
    )?;
    let ratio = num_u(tc, "indexer_compress_ratio")?;
    b.set(
        &k("attention.compress_ratios"),
        Value::Array(
            recurrent
                .iter()
                .map(|&r| Value::I32(if r { 0 } else { ratio as i32 }))
                .collect(),
        ),
    )?;

    let vocab_size = num_u(tc, "vocab_size")?;
    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, config, vocab_size, "qwen35")?;

    // Tensors.
    b.direct(
        "token_embd.weight",
        &format!("{prefix}embed_tokens.weight"),
        None,
    )?;
    if b.has("lm_head.weight") && !config_bool(config, tc, "tie_word_embeddings") {
        b.direct("output.weight", "lm_head.weight", None)?;
    }
    mixer(
        b,
        "output_hc",
        &format!("{prefix}hyper_connection_mixer"),
        false,
    )?;

    let order = v_order(nk, nv / nk);
    let ple_layers: Vec<u64> = tc
        .get("ple_layer_ids")
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_u64).collect())
        .unwrap_or_default();
    for il in 0..n_layer {
        let src = format!("{prefix}layers.{il}.");
        let blk = format!("blk.{il}.");
        let s = |t: &str| format!("{src}{t}");
        let d = |t: &str| format!("{blk}{t}");
        mixer(b, &d("hc_attn"), &s("attn_hyper_connection"), true)?;
        mixer(b, &d("hc_ffn"), &s("mlp_hyper_connection"), true)?;
        if recurrent[il as usize] {
            linear_attention(b, &s, &d, n_embd, nk, nv, dk, dv, &order, nk != nv)?;
        } else {
            full_attention(b, &s, &d, n_embd, idx_heads, idx_dim)?;
        }
        if ple_layers.contains(&(il + 1)) {
            for (g, h) in [("ple_key", "key_proj"), ("ple_value", "value_proj")] {
                b.direct(
                    &d(&format!("{g}.weight")),
                    &s(&format!("ple.{h}.weight")),
                    None,
                )?;
            }
            for (g, h) in [
                ("ple_norm_key", "norm_key"),
                ("ple_norm_query", "norm_query"),
                ("ple_norm_conv", "norm_conv"),
            ] {
                norm(
                    b,
                    &d(&format!("{g}.weight")),
                    &s(&format!("ple.{h}.weight")),
                )?;
            }
            // [channels, 1, kernel] -> ggml [kernel, channels], the size-1 dim dropped.
            let conv = s("ple.conv1d.weight");
            let (_, t) = b.source(&conv)?;
            let shape = vec![*t.shape.last().unwrap() as u64, t.shape[0] as u64];
            b.direct(&d("ple_conv1d.weight"), &conv, Some(shape))?;
        }
        moe(b, &s, &d, n_embd, n_expert, ff)?;
    }
    ple(b, tc, config, prefix, &ple_layers)
}

/// A boolean from the text config, else the top-level one.
fn config_bool(config: &Json, tc: &Json, key: &str) -> bool {
    tc.get(key)
        .or_else(|| config.get(key))
        .and_then(Json::as_bool)
        .unwrap_or(false)
}

/// A zero-centred norm weight (`src`) as the converter writes it: `+ 1`, F32.
fn norm(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    b.computed_f32(name, src, vec![t.numel() as u64], |x| {
        x.into_iter().map(|v| v + 1.0).collect()
    })
}

/// A hyper-connection mixer (`inject`: with the block's per-stream output weights).
fn mixer(b: &mut Builder, name: &str, src: &str, inject: bool) -> Result<(), String> {
    norm(
        b,
        &format!("{name}_norm.weight"),
        &format!("{src}.hc_norm.weight"),
    )?;
    b.direct(
        &format!("{name}_down.weight"),
        &format!("{src}.input_mix_weight_down.weight"),
        None,
    )?;
    b.direct(
        &format!("{name}_up.weight"),
        &format!("{src}.input_mix_weight_up.weight"),
        None,
    )?;
    if inject {
        // F32, exactly (the BF16 values widened), as the released GGUFs store them: prompt
        // steps then take the fused mixer norm instead of ggml's chain, whose BF16 matmul
        // rounds the activations to BF16 (what `--hc-inject-f32` does for GGUFs that keep
        // them BF16; +7% prefill on the GB10).
        let src = format!("{src}.block_inject_weight.weight");
        let (_, t) = b.source(&src)?;
        let shape = super::ggml_shape(&t.shape);
        b.computed_f32(&format!("{name}_inject.weight"), &src, shape, |x| x)?;
    }
    Ok(())
}

/// A full-attention block's projections and norms, the indexer's included.
fn full_attention(
    b: &mut Builder,
    s: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    n_embd: u64,
    idx_heads: u64,
    idx_dim: u64,
) -> Result<(), String> {
    for (g, h) in [
        ("attn_q", "q_proj"),
        ("attn_k", "k_proj"),
        ("attn_v", "v_proj"),
        ("attn_output", "o_proj"),
    ] {
        b.direct(
            &d(&format!("{g}.weight")),
            &s(&format!("self_attn.{h}.weight")),
            None,
        )?;
    }
    norm(b, &d("attn_q_norm.weight"), &s("self_attn.q_norm.weight"))?;
    norm(b, &d("attn_k_norm.weight"), &s("self_attn.k_norm.weight"))?;
    // One projection feeds the indexer's query heads and its key.
    let qk = s("self_attn.indexer.index_qk_proj.weight");
    let (_, t) = b.source(&qk)?;
    let n_q = idx_heads * idx_dim;
    if t.shape != [(n_q + idx_dim) as usize, n_embd as usize] {
        return Err(format!(
            "`{qk}`: shape {:?}, expected [{}, {n_embd}]",
            t.shape,
            n_q + idx_dim
        ));
    }
    b.gather_rows(
        &d("indexer.q_proj.weight"),
        &qk,
        &[(0, n_q)],
        vec![n_embd, n_q],
    )?;
    b.gather_rows(
        &d("indexer.k_proj.weight"),
        &qk,
        &[(n_q, idx_dim)],
        vec![n_embd, idx_dim],
    )?;
    norm(
        b,
        &d("indexer.q_norm.weight"),
        &s("self_attn.indexer.q_layernorm.weight"),
    )?;
    norm(
        b,
        &d("indexer.k_norm.weight"),
        &s("self_attn.indexer.k_layernorm.weight"),
    )
}

/// The checkpoint's MTP head (`mtp.*`) as the head GGUF llama.cpp's converter writes
/// (`mtp-*.gguf`, `nextn_shared_target_tensors`: the target's embeddings and output head serve
/// it): block `num_hidden_layers` with the trunk's full-attention layer names, plus
/// - `nextn.eh_proj`: `fc_embedding` and `fc_hidden` side by side along the input
///   (`eh_proj([e ; h]) = fc_embedding(e) + fc_hidden(h)`), read row by row as shipped;
/// - `nextn.enorm` / `nextn.hnorm`: `pre_fc_norm_embedding` / `pre_fc_norm_hidden`, `+ 1`;
/// - `nextn.hc_head_*`: the head's output mixer (`mtp.hyper_connection_mixer`).
///
/// Its routed experts stay as shipped too: block-scaled FP8 ones (`FP8_BLOCK_SCALES`, the
/// NVFP4 checkpoint's) as I8 E4M3 bytes plus an F32 scale grid ([`crate::model::fp8_block`]).
/// The head layer's attention is dense (`compress_ratios` 0), as in the released head GGUF.
pub(super) fn build_mtp(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_mtp = opt_u(tc, "mtp_num_hidden_layers", 0);
    if n_mtp != 1 {
        return Err(format!(
            "mtp_num_hidden_layers is {n_mtp}: one MTP layer is supported"
        ));
    }
    if !b.has("mtp.fc_embedding.weight") {
        return Err("the checkpoint has no MTP head (`mtp.*` tensors)".into());
    }
    let n_embd = num_u(tc, "hidden_size")?;
    let n_expert = num_u(tc, "num_experts")?;
    let ff = num_u(tc, "moe_intermediate_size")?;
    let idx_heads = num_u(tc, "indexer_n_heads")?;
    let idx_dim = num_u(tc, "indexer_head_dim")?;
    let u = |v: u64| Value::U32(v as u32);
    let k = |s: &str| format!("{ARCH}.{s}");
    b.set("general.architecture", Value::String(ARCH.into()))?;
    b.set(&k("block_count"), u(n_layer + n_mtp))?;
    b.set(&k("nextn_predict_layers"), u(n_mtp))?;
    b.set(&k("nextn_shared_target_tensors"), Value::Bool(true))?;
    b.set(&k("embedding_length"), u(n_embd))?;
    b.set(
        &k("attention.compress_ratios"),
        Value::Array(vec![Value::I32(0); (n_layer + n_mtp) as usize]),
    )?;
    // The head and expert shapes, as the released head GGUF carries them: tensor parallelism
    // over machines cuts the head by them (#1150). The model reads its own from the target.
    let n_head = num_u(tc, "num_attention_heads")?;
    let (nk, nv) = (
        num_u(tc, "linear_num_key_heads")?,
        num_u(tc, "linear_num_value_heads")?,
    );
    b.set(&k("attention.head_count"), u(n_head))?;
    b.set(
        &k("attention.head_count_kv"),
        u(num_u(tc, "num_key_value_heads")?),
    )?;
    let head_dim = opt_u(tc, "head_dim", n_embd / n_head);
    b.set(&k("attention.key_length"), u(head_dim))?;
    b.set(&k("attention.value_length"), u(head_dim))?;
    b.set(&k("ssm.state_size"), u(num_u(tc, "linear_key_head_dim")?))?;
    b.set(&k("ssm.group_count"), u(nk))?;
    b.set(&k("ssm.time_step_rank"), u(nv))?;
    b.set(
        &k("ssm.inner_size"),
        u(num_u(tc, "linear_value_head_dim")? * nv),
    )?;
    b.set(&k("expert_count"), u(n_expert))?;
    b.set(
        &k("expert_used_count"),
        u(num_u(tc, "num_experts_per_tok")?),
    )?;

    let blk = format!("blk.{n_layer}.");
    let s = |t: &str| format!("mtp.layers.0.{t}");
    let d = |t: &str| format!("{blk}{t}");
    mixer(b, &d("hc_attn"), &s("attn_hyper_connection"), true)?;
    mixer(b, &d("hc_ffn"), &s("mlp_hyper_connection"), true)?;
    full_attention(b, &s, &d, n_embd, idx_heads, idx_dim)?;
    moe(b, &s, &d, n_embd, n_expert, ff)?;
    norm(
        b,
        &d("nextn.enorm.weight"),
        "mtp.pre_fc_norm_embedding.weight",
    )?;
    norm(b, &d("nextn.hnorm.weight"), "mtp.pre_fc_norm_hidden.weight")?;
    mixer(b, &d("nextn.hc_head"), "mtp.hyper_connection_mixer", false)?;
    eh_proj(b, &d("nextn.eh_proj.weight"), n_embd)
}

/// `nextn.eh_proj` (ggml `[2 n_embd, n_embd]`): output row `r` is `fc_embedding`'s row `r`
/// and then `fc_hidden`'s, both `[n_embd, n_embd]` in one dtype.
fn eh_proj(b: &mut Builder, name: &str, n_embd: u64) -> Result<(), String> {
    let (fe, te) = b.source("mtp.fc_embedding.weight")?;
    let (fh, th) = b.source("mtp.fc_hidden.weight")?;
    let square = [n_embd as usize, n_embd as usize];
    if te.shape != square || th.shape != square || te.dtype != th.dtype {
        return Err(format!(
            "mtp.fc_embedding {} {:?} and fc_hidden {} {:?}: expected two {square:?} of one dtype",
            te.dtype.name(),
            te.shape,
            th.dtype.name(),
            th.shape
        ));
    }
    let ty = super::ggml_type(&te.dtype).ok_or_else(|| {
        format!(
            "mtp.fc_embedding: dtype {} is not loadable as is",
            te.dtype.name()
        )
    })?;
    let row = (te.nbytes() / n_embd as usize) as u64;
    let at = |file, start: usize, r: u64| super::FileRange {
        file,
        offset: start as u64 + r * row,
        len: row,
    };
    let parts = (0..n_embd).flat_map(|r| {
        [
            at(fe, te.data_range.start, r),
            at(fh, th.data_range.start, r),
        ]
    });
    b.gather(name, ty, vec![2 * n_embd, n_embd], super::merge(parts))
}

/// Block-scaled FP8 experts (`weight_scale_inv`, `FP8_BLOCK_SCALES`) stacked as shipped:
/// `name` is the E4M3 bytes as I8 (ggml `shape`, `[k, m, n_expert]`), and its `.scale` the
/// experts' scale grids widened to F32 (`[k / block_k, m / block_m, n_expert]`).
fn stack_fp8_blocks(
    b: &mut Builder,
    name: &str,
    srcs: &[String],
    shape: Vec<u64>,
) -> Result<(), String> {
    let scales: Vec<String> = srcs.iter().map(|s| format!("{s}_scale_inv")).collect();
    let (_, s0) = b.source(&scales[0])?;
    let [mb, kb] = s0.shape[..] else {
        return Err(format!(
            "`{}`: scale grid {:?}, expected 2-D",
            scales[0], s0.shape
        ));
    };
    // The blocks the matmul takes: whole blocks, 16-byte loads within one.
    let (k, m) = (shape[0] as usize, shape[1] as usize);
    if kb == 0 || mb == 0 || k % kb != 0 || m % mb != 0 || (k / kb) % 16 != 0 {
        return Err(format!(
            "`{}`: scale grid {:?} for a [{m}, {k}] weight: blocks must tile it, 16-wide or more",
            scales[0], s0.shape
        ));
    }
    let mut parts = Vec::with_capacity(srcs.len());
    for (src, sc) in srcs.iter().zip(&scales) {
        let (file, t) = b.source(src)?;
        let (_, ts) = b.source(sc)?;
        if t.dtype != Dtype::F8E4M3 || t.shape != [shape[1] as usize, shape[0] as usize] {
            return Err(format!(
                "`{src}`: {} {:?}, expected F8_E4M3 [{}, {}]",
                t.dtype.name(),
                t.shape,
                shape[1],
                shape[0]
            ));
        }
        if ts.shape != s0.shape {
            return Err(format!(
                "`{sc}`: scale grid {:?}, the first expert's is {:?}",
                ts.shape, s0.shape
            ));
        }
        parts.push(super::FileRange {
            file,
            offset: t.data_range.start as u64,
            len: t.nbytes() as u64,
        });
    }
    b.gather(
        name,
        types::I8,
        shape.clone(),
        super::merge(parts.into_iter()),
    )?;
    let out = name
        .strip_suffix(".weight")
        .map(|n| format!("{n}.scale"))
        .ok_or_else(|| format!("`{name}`: no .weight"))?;
    let ckpt = b.ckpt.clone();
    let compute: super::Compute = std::sync::Arc::new(move || {
        let mut out = Vec::new();
        for sc in &scales {
            let (_, t) = ckpt.locate(sc).ok_or("tensor disappeared")?;
            let bytes = ckpt.data(sc).map_err(|e| e.to_string())?;
            let x = super::to_f32(&t.dtype, bytes)
                .ok_or_else(|| format!("`{sc}`: not a float tensor"))?;
            out.extend(x.iter().flat_map(|v| v.to_le_bytes()));
        }
        Ok(out)
    });
    let grid = vec![kb as u64, mb as u64, shape[2]];
    b.add_computed(&out, types::F32, grid, compute)
}

/// One MoE block: router (F32), routed experts (ModelOpt NVFP4 repacked and stacked, or
/// stacked as shipped), the gated shared expert.
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
        } else if t.dtype == Dtype::F8E4M3 && b.has(&format!("{}_scale_inv", srcs[0])) {
            stack_fp8_blocks(b, &name, &srcs, shape)?;
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
            &s(&format!("mlp.shared_expert.{h}.weight")),
            None,
        )?;
    }
    b.computed_f32(
        &d("ffn_gate_inp_shexp.weight"),
        &s("mlp.shared_expert_gate.weight"),
        vec![n_embd],
        |x| x,
    )
}

/// The per-layer n-gram embedding's metadata and table (see the module docs).
fn ple(
    b: &mut Builder,
    tc: &Json,
    config: &Json,
    prefix: &str,
    ple_layers: &[u64],
) -> Result<(), String> {
    let [layer] = ple_layers else {
        if ple_layers.is_empty() {
            return Ok(());
        }
        return Err(format!(
            "ple_layer_ids {ple_layers:?}: one n-gram layer is supported"
        ));
    };
    let il = layer
        .checked_sub(1)
        .ok_or("ple_layer_ids are 1-based, got 0")?;
    let base = format!("{prefix}layers.{il}.ple.ple_embedding.");
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: u64| Value::U32(v as u32);
    b.set(&k("ple.layers"), Value::Array(vec![Value::I32(il as i32)]))?;
    b.set(&k("ple.ngram_size"), u(num_u(tc, "ngram_size")?))?;
    b.set(&k("ple.heads_per_ngram"), u(num_u(tc, "heads_per_ngram")?))?;
    b.set(&k("ple.conv_kernel"), u(num_u(tc, "ple_conv_kernel_size")?))?;
    // The primary EOS resets the n-grams (the last of a list).
    let eos = tc
        .get("eos_token_id")
        .or_else(|| config.get("eos_token_id"))
        .and_then(|v| match v {
            Json::Array(a) => a.last().and_then(Json::as_u64),
            v => v.as_u64(),
        })
        .ok_or("config.json: `eos_token_id` is required (the n-gram hash resets on it)")?;
    b.set(&k("ple.eos_token_id"), u(eos))?;
    if let Some(img) = config.get("image_token_id").and_then(Json::as_u64) {
        b.set(&k("ple.image_token_id"), u(img))?;
    }
    for (key, src) in [
        ("ple.layer_multipliers", "layer_multipliers"),
        ("ple.head_offsets", "ngram_heads_offsets"),
        ("ple.head_vocab_sizes", "ngram_heads_vocab_sizes"),
    ] {
        let v = int_constants(b, &format!("{base}{src}"))?;
        b.set(
            &k(key),
            Value::Array(v.into_iter().map(Value::U64).collect()),
        )?;
    }

    // The shards, in index order.
    let n_parts = num_u(tc, "split_ngram_parts")?;
    let mut shards = Vec::with_capacity(n_parts as usize);
    for i in 0..n_parts {
        let name = format!("{base}ngram_embedding.shard_{i}.weight");
        let (file, t) = b.source(&name)?;
        shards.push((name, file, t));
    }
    let (_, file, first) = &shards[0];
    let (file, dtype, shape) = (*file, first.dtype.clone(), first.shape.clone());
    let [rows, dim] = shape[..] else {
        return Err(format!("n-gram shard 0: shape {shape:?}, expected 2-D"));
    };
    let bytes = first.nbytes() as u64;
    for (name, f, t) in &shards {
        if *f != file || t.dtype != dtype || t.shape != shape {
            return Err(format!(
                "`{name}`: the n-gram shards must share one file, dtype and shape"
            ));
        }
    }
    // Back to back in the file, in some order: block `i`'s position in it.
    let mut by_offset: Vec<(u64, usize)> = shards
        .iter()
        .enumerate()
        .map(|(i, (_, _, t))| (t.data_range.start as u64, i))
        .collect();
    by_offset.sort_unstable();
    let start = by_offset[0].0;
    let mut position = vec![0u64; shards.len()];
    for (p, &(off, i)) in by_offset.iter().enumerate() {
        if off != start + p as u64 * bytes {
            return Err(format!(
                "`{}`: the n-gram shards are not back to back in their file",
                shards[i].0
            ));
        }
        position[i] = p as u64;
    }
    let ty = match dtype {
        Dtype::F8E4M3 => {
            let scale =
                super::nvfp4::scalar(&b.ckpt, &format!("{base}ngram_embedding.weight_scale"))?;
            b.set(KEY_PLE_FP8_SCALE, Value::F32(scale))?;
            types::I8
        }
        Dtype::BF16 => types::BF16,
        Dtype::F16 => types::F16,
        Dtype::F32 => types::F32,
        other => {
            return Err(format!(
                "n-gram table: dtype {} is not supported",
                other.name()
            ));
        }
    };
    if position.iter().enumerate().any(|(i, &p)| i as u64 != p) {
        b.set(KEY_PLE_BLOCK_ROWS, Value::U64(rows as u64))?;
        b.set(
            KEY_PLE_BLOCK_ORDER,
            Value::Array(position.into_iter().map(Value::U64).collect()),
        )?;
    }
    b.set(&k("embedding_length_per_layer_input"), u(dim as u64))?;
    b.push(
        "per_layer_token_embd.weight",
        ty,
        vec![dim as u64, rows as u64 * n_parts],
        file,
        start,
    );
    Ok(())
}

/// An integer tensor's values, exactly (the hash constants are 45-bit).
fn int_constants(b: &Builder, src: &str) -> Result<Vec<u64>, String> {
    let (_, t) = b.source(src)?;
    let bytes = b.ckpt.data(src).map_err(|e| e.to_string())?;
    let v: Vec<i64> = match t.dtype {
        Dtype::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect(),
        Dtype::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as i64)
            .collect(),
        _ => {
            return Err(format!(
                "`{src}`: expected an integer tensor, got {}",
                t.dtype.name()
            ));
        }
    };
    v.into_iter()
        .map(|x| u64::try_from(x).map_err(|_| format!("`{src}`: negative value {x}")))
        .collect()
}
