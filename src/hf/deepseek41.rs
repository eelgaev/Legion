//! DeepSeek-V4.1-Flash checkpoints (`deepseek_v41`, `DeepseekV41ForCausalLM`; the text model
//! only) as the `deepseek41` GGUF that [`crate::model::deepseek41`] reads: the metadata and
//! tensor names of the released GGUFs, the tensors in the formats the checkpoint ships, never
//! converted ahead of time. The names follow DeepSeek's reference `inference/convert.py` and
//! `model.py` (MIT).
//!
//! - Dense weights shipped as FP8 E4M3 with an E8M0 scale per 32 × 32 block
//!   (`<w>.weight` / `<w>.scale`) are the E4M3 bytes in an `I8` tensor (Legion's FP8 carrier,
//!   [`crate::model::fp8`]) as stored, plus `<w>.scale`: the block scales as F32 (exactly
//!   `2^(e - 127)`), ggml shape `[k / 32, m / 32]`. BF16 / F32 weights are used as shipped.
//! - Routed experts shipped as FP4 E2M1 (`I8`, two codes a byte along the row, the even
//!   element in the low nibble) with an E8M0 scale per 32 elements are OCP MXFP4: repacked
//!   losslessly into ggml's `MXFP4` blocks (the scale byte, then element `j` and `j + 16` in a
//!   byte's low and high nibble) and stacked per layer, at load ([`repack_mxfp4`]).
//! - Norms, sinks, hyper-connection mixers and the router bias are F32 (BF16 widened
//!   exactly); `wo_a` is `attn_output_a` `[n_head · head / groups, o_lora, groups]`.
//! - The Engram tables (`layers.{l}.engram.embed`, FP8 rows of `engram_head_dim` with an E8M0
//!   scale per 32) are not copied: `blk.{l}.engram_embd.weight` is the E4M3 rows as `I8`
//!   where the shard has them, `blk.{l}.engram_embd.scale` their scale bytes (`I8`, `[dim / 32,
//!   rows]`), both file ranges read row by row ([`crate::model::deepseek41::engram`]).
//! - The Engram hash parameters, which the checkpoint does not ship, are made from its config
//!   and tokenizer as `engram.py` makes them ([`hash`]): `engram_token_map`, `engram_primes`,
//!   `engram_offsets` and `engram_multipliers`, as the released GGUFs hold them.
//!
//! Not loaded: the MTP / DSpark layers (`mtp.*`), the vision tower and its aligner
//! (`vision.*`, `aligner.*`, `image_*`) and the router's vision bias (`ffn.gate.bias_vl`).

pub mod hash;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use serde_json::Value as Json;

use super::qwen35::text_config;
use super::{Builder, ggml_shape, num_f, num_u};
use crate::ggml::{ffi, types};
use crate::gguf::{Compute, Recipe, Value};
use crate::model::fp8::e8m0_to_f32;
use crate::safetensors::Dtype;

const ARCH: &str = "deepseek41";

/// Elements per E8M0 scale along a row (and rows per dense block).
const BLOCK: u64 = 32;

/// The config's `key` as a list of integers.
fn ints(tc: &Json, key: &str) -> Result<Vec<u64>, String> {
    tc.get(key)
        .and_then(Json::as_array)
        .ok_or_else(|| format!("config.json: missing `{key}`"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| format!("config.json: `{key}` holds {v}"))
        })
        .collect()
}

fn i32s(v: &[u64]) -> Value {
    Value::Array(v.iter().map(|&x| Value::I32(x as i32)).collect())
}

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_embd = num_u(tc, "hidden_size")?;
    let n_head = num_u(tc, "num_attention_heads")?;
    let head = num_u(tc, "head_dim")?;
    let n_expert = num_u(tc, "n_routed_experts")?;
    let ff = num_u(tc, "moe_intermediate_size")?;
    let o_groups = num_u(tc, "o_groups")?;
    let o_lora = num_u(tc, "o_lora_rank")?;
    let scoring = tc
        .get("scoring_func")
        .and_then(Json::as_str)
        .unwrap_or("sqrtsoftplus");
    if scoring != "sqrtsoftplus" {
        return Err(format!(
            "scoring_func `{scoring}`: only DeepSeek-V4.1's sqrtsoftplus router is supported"
        ));
    }
    let prefix = if b.has("model.embed.weight") {
        "model."
    } else {
        ""
    };

    // Metadata, as the released GGUFs have it.
    let u = |v: u64| Value::U32(v as u32);
    let f = |v: f64| Value::F32(v as f32);
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
    if let Some(rs) = tc.get("rope_scaling").filter(|r| !r.is_null()) {
        let kind = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(Json::as_str)
            .unwrap_or_default();
        if kind != "yarn" {
            return Err(format!(
                "rope_scaling of type `{kind}`: only yarn is supported"
            ));
        }
        b.set(&k("rope.scaling.type"), Value::String(kind.into()))?;
        b.set(&k("rope.scaling.factor"), f(num_f(rs, "factor")?))?;
        b.set(
            &k("rope.scaling.original_context_length"),
            u(num_u(rs, "original_max_position_embeddings")?),
        )?;
        b.set(
            &k("rope.scaling.yarn_beta_fast"),
            f(num_f(rs, "beta_fast")?),
        )?;
        b.set(
            &k("rope.scaling.yarn_beta_slow"),
            f(num_f(rs, "beta_slow")?),
        )?;
    }
    b.set(&k("rope.freq_base"), f(num_f(tc, "rope_theta")?))?;
    b.set(
        &k("attention.layer_norm_rms_epsilon"),
        f(num_f(tc, "rms_norm_eps")?),
    )?;
    b.set(&k("expert_count"), u(n_expert))?;
    b.set(
        &k("expert_used_count"),
        u(num_u(tc, "num_experts_per_tok")?),
    )?;
    b.set(&k("expert_gating_func"), u(4))?;
    b.set(&k("attention.key_length"), u(head))?;
    b.set(&k("attention.value_length"), u(head))?;
    b.set(
        &k("rope.dimension_count"),
        u(num_u(tc, "qk_rope_head_dim")?),
    )?;
    b.set(&k("attention.q_lora_rank"), u(num_u(tc, "q_lora_rank")?))?;
    b.set(
        &k("attention.sliding_window"),
        u(num_u(tc, "sliding_window")?),
    )?;
    b.set(&k("expert_feed_forward_length"), u(ff))?;
    b.set(&k("expert_shared_count"), u(num_u(tc, "n_shared_experts")?))?;
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
    let clamp = Value::Array(vec![f(num_f(tc, "swiglu_limit")?); n_layer as usize]);
    b.set(&k("swiglu_clamp_exp"), clamp.clone())?;
    b.set(&k("swiglu_clamp_shexp"), clamp)?;
    let idx_heads = num_u(tc, "index_n_heads")?;
    let idx_dim = num_u(tc, "index_head_dim")?;
    b.set(&k("attention.indexer.head_count"), u(idx_heads))?;
    b.set(&k("attention.indexer.key_length"), u(idx_dim))?;
    b.set(&k("attention.indexer.top_k"), u(num_u(tc, "index_topk")?))?;
    b.set(&k("attention.output_group_count"), u(o_groups))?;
    b.set(&k("attention.output_lora_rank"), u(o_lora))?;
    b.set(
        &k("attention.compress_ratios"),
        i32s(&ints(tc, "compress_ratios")?),
    )?;
    b.set(
        &k("attention.compress_rope_freq_base"),
        f(num_f(tc, "compress_rope_theta")?),
    )?;
    b.set(&k("hyper_connection.count"), u(num_u(tc, "hc_mult")?))?;
    b.set(
        &k("hyper_connection.sinkhorn_iterations"),
        u(num_u(tc, "hc_sinkhorn_iters")?),
    )?;
    b.set(&k("hyper_connection.epsilon"), f(num_f(tc, "hc_eps")?))?;
    b.set(&k("hash_layer_count"), u(0))?;
    if tc.get("candidate_source_layer_id").is_some() {
        b.set(
            &k("attention.candidate_block_size"),
            u(num_u(tc, "candidate_block_size")?),
        )?;
        b.set(
            &k("attention.candidate_topk_blocks"),
            u(num_u(tc, "candidate_topk_blocks")?),
        )?;
        b.set(
            &k("attention.candidate_source_layer_id"),
            u(num_u(tc, "candidate_source_layer_id")?),
        )?;
    }
    for key in [
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "dspark_target_layer_ids",
    ] {
        if tc.get(key).is_some() {
            b.set(&k(key), i32s(&ints(tc, key)?))?;
        }
    }
    b.set(&k("expert_scoring_func"), Value::String(scoring.into()))?;

    let vocab_size = num_u(tc, "vocab_size")?;
    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, config, vocab_size, "joyai-llm")?;

    // Tensors.
    let s = |il: u64, t: &str| format!("{prefix}layers.{il}.{t}");
    let d = |il: u64, t: &str| format!("blk.{il}.{t}");
    linear(b, "token_embd.weight", &format!("{prefix}embed"), None)?;
    linear(b, "output.weight", &format!("{prefix}head"), None)?;
    f32_tensor(b, "output_norm.weight", &format!("{prefix}norm.weight"))?;
    let engram_layers = match tc.get("engram_layer_ids") {
        Some(_) => ints(tc, "engram_layer_ids")?,
        None => Vec::new(),
    };
    for il in 0..n_layer {
        let s = |t: &str| s(il, t);
        let d = |t: &str| d(il, t);
        f32_tensor(b, &d("attn_norm.weight"), &s("attn_norm.weight"))?;
        f32_tensor(b, &d("ffn_norm.weight"), &s("ffn_norm.weight"))?;
        linear(b, &d("attn_q_a.weight"), &s("attn.wq_a"), None)?;
        f32_tensor(b, &d("attn_q_a_norm.weight"), &s("attn.q_norm.weight"))?;
        linear(b, &d("attn_q_b.weight"), &s("attn.wq_b"), None)?;
        linear(b, &d("attn_kv.weight"), &s("attn.wkv"), None)?;
        f32_tensor(b, &d("attn_kv_a_norm.weight"), &s("attn.kv_norm.weight"))?;
        f32_tensor(b, &d("attn_sinks.weight"), &s("attn.attn_sink"))?;
        // `[groups · o_lora, n_head · head / groups]`: per group its `o_lora` rows.
        let o_a = vec![n_head * head / o_groups, o_lora, o_groups];
        linear(b, &d("attn_output_a.weight"), &s("attn.wo_a"), Some(o_a))?;
        linear(b, &d("attn_output_b.weight"), &s("attn.wo_b"), None)?;
        for which in ["attn", "ffn"] {
            for part in ["fn", "base", "scale"] {
                f32_tensor(
                    b,
                    &d(&format!("hc_{which}_{part}.weight")),
                    &s(&format!("hc_{which}_{part}")),
                )?;
            }
        }
        // Compression and the indexer, on the layers that have them.
        for (g, h) in [
            ("attn_compressor_kv", "attn.compressor.wkv"),
            ("attn_compressor_gate", "attn.compressor.wgate"),
            ("indexer.attn_k", "attn.indexer.wk"),
            ("indexer.attn_q_b", "attn.indexer.wq_b"),
            ("indexer.proj", "attn.indexer.weights_proj"),
        ] {
            if b.has(&s(&format!("{h}.weight"))) {
                linear(b, &d(&format!("{g}.weight")), &s(h), None)?;
            }
        }
        for (g, h) in [
            ("attn_compressor_norm", "attn.compressor.norm"),
            ("indexer.k_norm", "attn.indexer.k_norm"),
        ] {
            let src = s(&format!("{h}.weight"));
            if b.has(&src) {
                f32_tensor(b, &d(&format!("{g}.weight")), &src)?;
            }
        }
        // MoE: the router (as shipped), its selection bias, the routed and the shared experts.
        linear(b, &d("ffn_gate_inp.weight"), &s("ffn.gate"), None)?;
        f32_tensor(b, &d("exp_probs_b.bias"), &s("ffn.gate.bias"))?;
        for (g, h, shape) in [
            ("ffn_gate_exps", "w1", vec![n_embd, ff, n_expert]),
            ("ffn_up_exps", "w3", vec![n_embd, ff, n_expert]),
            ("ffn_down_exps", "w2", vec![ff, n_embd, n_expert]),
        ] {
            let srcs: Vec<String> = (0..n_expert)
                .map(|e| s(&format!("ffn.experts.{e}.{h}")))
                .collect();
            experts(b, &d(&format!("{g}.weight")), &srcs, shape)?;
        }
        for (g, h) in [
            ("ffn_gate_shexp", "w1"),
            ("ffn_up_shexp", "w3"),
            ("ffn_down_shexp", "w2"),
        ] {
            linear(
                b,
                &d(&format!("{g}.weight")),
                &s(&format!("ffn.shared_experts.{h}")),
                None,
            )?;
        }
        if engram_layers.contains(&il) {
            for (g, h) in [("engram_q", "q_weight"), ("engram_k", "k_weight")] {
                b.direct(&d(&format!("{g}.weight")), &s(&format!("engram.{h}")), None)?;
            }
            linear(b, &d("engram_wkv.weight"), &s("engram.wkv"), None)?;
            engram_table(b, il, &s("engram.embed"))?;
        }
    }
    if !engram_layers.is_empty() {
        engram(b, tc, &engram_layers)?;
    }
    Ok(())
}

/// `name` is the source tensor `src` (F32, or BF16 / F16 widened exactly), its shape kept.
fn f32_tensor(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let shape = ggml_shape(&t.shape);
    if t.dtype == Dtype::F32 {
        return b.direct(name, src, Some(shape));
    }
    b.computed_f32(name, src, shape, |x| x)
}

/// The dense weight `<base>.weight` as `name` (ggml `shape`, default its own): BF16 / F16 /
/// F32 as shipped, or FP8 E4M3 as `I8` with its 32 × 32 E8M0 block scales as F32
/// `<name base>.scale` (ggml `[k / 32, m / 32]`).
fn linear(b: &mut Builder, name: &str, base: &str, shape: Option<Vec<u64>>) -> Result<(), String> {
    let src = format!("{base}.weight");
    let (file, t) = b.source(&src)?;
    if t.dtype != Dtype::F8E4M3 {
        return b.direct(name, &src, shape);
    }
    let [m, k] = t.shape[..] else {
        return Err(format!("`{src}`: FP8 weight of shape {:?}", t.shape));
    };
    let (m, k) = (m as u64, k as u64);
    let scale = format!("{base}.scale");
    let (_, st) = b.source(&scale)?;
    let blocks = [m.div_ceil(BLOCK) as usize, k.div_ceil(BLOCK) as usize];
    if st.dtype != Dtype::F8E8M0 || st.shape != blocks {
        return Err(format!(
            "`{scale}`: {} {:?}, expected F8_E8M0 {blocks:?} (32 x 32 blocks)",
            st.dtype.name(),
            st.shape
        ));
    }
    let shape = shape.unwrap_or(vec![k, m]);
    if shape.iter().product::<u64>() != m * k {
        return Err(format!(
            "`{src}`: shape {shape:?} does not match {:?}",
            t.shape
        ));
    }
    b.push(name, types::I8, shape, file, t.data_range.start as u64);
    let scale_name = name
        .strip_suffix(".weight")
        .map(|n| format!("{n}.scale"))
        .ok_or_else(|| format!("`{name}`: no .weight"))?;
    let ckpt = b.ckpt.clone();
    let compute: Compute = Arc::new(move || {
        let bytes = ckpt.data(&scale).map_err(|e| e.to_string())?;
        Ok(bytes
            .iter()
            .flat_map(|&e| e8m0_to_f32(e).to_le_bytes())
            .collect())
    });
    b.push(
        &scale_name,
        types::F32,
        vec![blocks[1] as u64, blocks[0] as u64],
        0,
        0,
    );
    b.recipes.insert(scale_name, Recipe::Computed(compute));
    Ok(())
}

/// One FP4 expert matrix (`rows` × `k`: `w` the E2M1 codes, two a byte along the row, the even
/// element in the low nibble; `s` one E8M0 scale per 32 elements) as ggml `MXFP4` rows. The
/// same values: ggml's E2M1 table is doubled and its scale halved.
pub fn repack_mxfp4(w: &[u8], s: &[u8], rows: usize, k: usize) -> Vec<u8> {
    let nb = k / 32;
    let mut out = Vec::with_capacity(rows * nb * 17);
    for r in 0..rows {
        for i in 0..nb {
            let src = &w[r * k / 2 + i * 16..][..16];
            let code = |e: usize| (src[e / 2] >> (4 * (e % 2))) & 0x0f;
            out.push(s[r * nb + i]);
            out.extend((0..16).map(|j| code(j) | code(j + 16) << 4));
        }
    }
    out
}

/// The routed experts `srcs` (`<src>.weight`, in expert order) stacked as `name` (ggml
/// `shape`): FP4 with E8M0 scales repacked into `MXFP4`, BF16 / F16 / F32 as shipped.
fn experts(b: &mut Builder, name: &str, srcs: &[String], shape: Vec<u64>) -> Result<(), String> {
    let weights: Vec<String> = srcs.iter().map(|s| format!("{s}.weight")).collect();
    let (_, t) = b.source(&weights[0])?;
    match t.dtype {
        Dtype::I8 => {}
        Dtype::F8E4M3 => {
            return Err(format!(
                "`{}`: FP8 routed experts (a re-converted checkpoint) are not supported; the \
                 released FP4 ones are",
                weights[0]
            ));
        }
        _ => return b.stack(name, &weights, shape),
    }
    let (k, rows) = (shape[0] as usize, shape[1] as usize);
    if !k.is_multiple_of(32) {
        return Err(format!("`{name}`: rows of {k}, not a multiple of 32"));
    }
    for src in srcs {
        let (_, w) = b.source(&format!("{src}.weight"))?;
        let (_, sc) = b.source(&format!("{src}.scale"))?;
        if w.dtype != Dtype::I8
            || w.shape != [rows, k / 2]
            || sc.dtype != Dtype::F8E8M0
            || sc.shape != [rows, k / 32]
        {
            return Err(format!(
                "`{src}`: FP4 weight {} {:?} / scale {} {:?} for {rows} x {k}",
                w.dtype.name(),
                w.shape,
                sc.dtype.name(),
                sc.shape
            ));
        }
    }
    let ckpt = b.ckpt.clone();
    let srcs = srcs.to_vec();
    let compute: Compute = Arc::new(move || {
        let mut out = Vec::with_capacity(srcs.len() * rows * k / 32 * 17);
        for src in &srcs {
            let w = ckpt
                .data(&format!("{src}.weight"))
                .map_err(|e| e.to_string())?;
            let s = ckpt
                .data(&format!("{src}.scale"))
                .map_err(|e| e.to_string())?;
            out.extend(repack_mxfp4(w, s, rows, k));
        }
        Ok(out)
    });
    b.push(name, types::MXFP4, shape, 0, 0);
    b.recipes
        .insert(name.to_string(), Recipe::Computed(compute));
    Ok(())
}

/// Layer `il`'s Engram table `<base>.weight` (+ `<base>.scale` for FP8) as file ranges.
fn engram_table(b: &mut Builder, il: u64, base: &str) -> Result<(), String> {
    let name = format!("blk.{il}.engram_embd.weight");
    let src = format!("{base}.weight");
    let (file, t) = b.source(&src)?;
    if t.dtype != Dtype::F8E4M3 {
        return b.direct(&name, &src, None);
    }
    let [rows, dim] = t.shape[..] else {
        return Err(format!("`{src}`: shape {:?}", t.shape));
    };
    let scale = format!("{base}.scale");
    let (sfile, st) = b.source(&scale)?;
    if st.dtype != Dtype::F8E8M0 || st.shape != [rows, dim.div_ceil(BLOCK as usize)] {
        return Err(format!(
            "`{scale}`: {} {:?}, expected F8_E8M0 [{rows}, {}]",
            st.dtype.name(),
            st.shape,
            dim.div_ceil(BLOCK as usize)
        ));
    }
    let (rows, dim) = (rows as u64, dim as u64);
    b.push(
        &name,
        types::I8,
        vec![dim, rows],
        file,
        t.data_range.start as u64,
    );
    b.push(
        &format!("blk.{il}.engram_embd.scale"),
        types::I8,
        vec![dim.div_ceil(BLOCK), rows],
        sfile,
        st.data_range.start as u64,
    );
    Ok(())
}

/// `name` (`ty`, ggml `shape`) is `bytes`.
fn constant(b: &mut Builder, name: &str, ty: ffi::ggml_type, shape: Vec<u64>, bytes: Vec<u8>) {
    b.push(name, ty, shape, 0, 0);
    b.recipes.insert(
        name.to_string(),
        Recipe::Computed(Arc::new(move || Ok(bytes.clone()))),
    );
}

/// The Engram metadata and hash parameters (see the module docs).
fn engram(b: &mut Builder, tc: &Json, layers: &[u64]) -> Result<(), String> {
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: u64| Value::U32(v as u32);
    let layout = hash::Layout {
        layers: layers.to_vec(),
        max_ngram: num_u(tc, "engram_max_ngram_size")? as usize,
        heads: num_u(tc, "engram_n_heads")? as usize,
        vocab: num_u(tc, "engram_vocab_size")?,
    };
    if layout.max_ngram < 2 || layout.heads == 0 {
        return Err(format!("Engram of {layout:?}"));
    }
    let primes = hash::primes(&layout);
    let rows: Vec<u64> = primes.iter().map(|p| p.iter().sum()).collect();
    if let Some(want) = tc.get("engram_num_embeddings") {
        let want = ints(tc, "engram_num_embeddings").map_err(|e| format!("{e} ({want})"))?;
        if want != rows {
            return Err(format!(
                "engram_num_embeddings {want:?}, but the hash buckets add up to {rows:?}"
            ));
        }
    }
    for (&l, &n) in layers.iter().zip(&rows) {
        let name = format!("blk.{l}.engram_embd.weight");
        let got = b
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.shape[1]);
        if got != Some(n) {
            return Err(format!("`{name}`: {got:?} rows, the hash buckets {n}"));
        }
    }
    let dir = b.ckpt.dir().to_path_buf();
    let text = std::fs::read_to_string(dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer.json (the Engram token map): {e}"))?;
    let tok: Json = serde_json::from_str(&text).map_err(|e| format!("tokenizer.json: {e}"))?;
    let (map, compressed) = hash::token_map(&tok)?;
    let want = num_u(tc, "engram_compressed_vocab_size")?;
    if compressed as u64 != want {
        return Err(format!(
            "the Engram token map has {compressed} ids, engram_compressed_vocab_size is {want}: \
             the hash multipliers would differ"
        ));
    }
    let mult = hash::multipliers(&layout, compressed as u64)?;
    let offsets = hash::offsets(&primes);

    b.set(&k("engram.present"), Value::Bool(true))?;
    b.set(&k("engram.layer_ids"), i32s(layers))?;
    b.set(&k("engram.num_embeddings"), i32s(&rows))?;
    b.set(&k("engram_head_dim"), u(num_u(tc, "engram_head_dim")?))?;
    b.set(&k("engram_n_heads"), u(layout.heads as u64))?;
    b.set(&k("engram_max_ngram_size"), u(layout.max_ngram as u64))?;
    b.set(&k("engram_vocab_size"), u(layout.vocab))?;
    b.set(&k("engram_compressed_vocab_size"), u(want))?;
    b.set(
        &k("engram_pad_token_id"),
        u(num_u(tc, "engram_pad_token_id")?),
    )?;

    let nl = layers.len() as u64;
    let cols = ((layout.max_ngram - 1) * layout.heads) as u64;
    let i64_bytes = |v: Vec<Vec<u64>>| -> Vec<u8> {
        v.concat()
            .iter()
            .flat_map(|&x| (x as i64).to_le_bytes())
            .collect()
    };
    constant(
        b,
        "engram_token_map.weight",
        types::I32,
        vec![map.len() as u64],
        map.iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    constant(
        b,
        "engram_primes.weight",
        types::I64,
        vec![cols, nl],
        i64_bytes(primes),
    );
    constant(
        b,
        "engram_offsets.weight",
        types::I64,
        vec![cols, nl],
        i64_bytes(offsets),
    );
    let mult: Vec<u8> = mult.concat().iter().flat_map(|v| v.to_le_bytes()).collect();
    constant(
        b,
        "engram_multipliers.weight",
        types::I64,
        vec![layout.max_ngram as u64, nl],
        mult,
    );
    Ok(())
}
