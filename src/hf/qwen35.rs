//! Qwen3.5 / 3.6 checkpoints (`qwen3_5` dense, `qwen3_5_moe`) as the `qwen35` / `qwen35moe` GGUF
//! llama.cpp's converter writes: `Qwen2MoeModel`, `Qwen3NextModel`, `_LinearAttentionVReorderBase`,
//! `Qwen3_5TextModel`, `Qwen3_5MoeTextModel` and `_QwenMtpMixin` in `conversion/qwen.py`
//! (llama.cpp, MIT).
//!
//! What the converter does beyond renaming, and how it is reproduced here:
//! - `A_log` becomes `-exp(A_log)` and every `*norm.weight` except the DeltaNet output norm
//!   gets `+ 1` (computed in f32, as the converter widens BF16 before transforming);
//! - when there are fewer key heads than value heads, the value heads are moved from grouped
//!   (`[k0v0, k0v1, k1v0, k1v1]`) to tiled order (`[k0v0, k1v0, k0v1, k1v1]`) in every DeltaNet
//!   tensor that has them: rows of `in_proj_qkv` (V part), `in_proj_z`, `in_proj_a/b`, `A_log`,
//!   `dt_bias`, `conv1d` (V part), and the columns of `out_proj`;
//! - `conv1d` drops its middle size-1 dim and is F32;
//! - the MTP layers (`mtp.*`) are appended as blocks `n_layer..`, with `nextn.*` names;
//! - MoE experts become 3-D `ffn_{gate,up,down}_exps`: a fused `gate_up_proj` `[E, 2*ff, embd]`
//!   is split per expert into its gate and up rows (gathered), per-expert tensors are stacked;
//!   the router and shared-expert gate are F32.

use serde_json::Value as Json;

use super::{Builder, num_f, num_u};
use crate::gguf::Value;

/// The text part of `config.json` (multimodal checkpoints nest it under `text_config`).
pub(super) fn text_config(config: &Json) -> &Json {
    config.get("text_config").unwrap_or(config)
}

/// The value-head reorder as a permutation of `nk * nv_per_k` blocks: entry `j` is the
/// source block of destination block `j` (destination `(v, k)` <- source `(k, v)`).
pub(super) fn v_order(nk: u64, nv_per_k: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity((nk * nv_per_k) as usize);
    for v in 0..nv_per_k {
        for k in 0..nk {
            out.push(k * nv_per_k + v);
        }
    }
    out
}

/// Row ranges `(first, count)` of the reordered blocks of `block` rows each, after `skip` rows
/// that stay in place.
fn block_rows(skip: u64, order: &[u64], block: u64) -> Vec<(u64, u64)> {
    let mut rows = Vec::with_capacity(order.len() + 1);
    if skip > 0 {
        rows.push((0, skip));
    }
    rows.extend(order.iter().map(|&b| (skip + b * block, block)));
    rows
}

/// Reorders rows of `width` elements (`skip` leading rows unchanged) by `order` over blocks of
/// `block` rows.
fn permute_rows<T: Copy>(
    x: &[T],
    width: usize,
    skip: usize,
    order: &[u64],
    block: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(x.len());
    out.extend_from_slice(&x[..skip * width]);
    for &b in order {
        let start = (skip + b as usize * block) * width;
        out.extend_from_slice(&x[start..start + block * width]);
    }
    out
}

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_expert = tc.get("num_experts").and_then(Json::as_u64).unwrap_or(0);
    let moe = n_expert > 0;
    let arch = if moe { "qwen35moe" } else { "qwen35" };
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_embd = num_u(tc, "hidden_size")?;
    let n_head = num_u(tc, "num_attention_heads")?;
    let head_dim = tc
        .get("head_dim")
        .and_then(Json::as_u64)
        .unwrap_or(n_embd / n_head);
    let nk = num_u(tc, "linear_num_key_heads")?;
    let nv = num_u(tc, "linear_num_value_heads")?;
    let dk = num_u(tc, "linear_key_head_dim")?;
    let dv = num_u(tc, "linear_value_head_dim")?;
    if nv % nk != 0 {
        return Err(format!(
            "linear_num_value_heads {nv} is not a multiple of linear_num_key_heads {nk}"
        ));
    }
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

    let prefix = if b.has("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };
    let n_mtp = if b.has("mtp.fc.weight") {
        tc.get("mtp_num_hidden_layers")
            .and_then(Json::as_u64)
            .unwrap_or(1)
    } else {
        0
    };
    let interval = tc
        .get("full_attention_interval")
        .and_then(Json::as_u64)
        .unwrap_or(4);
    let recurrent: Vec<bool> = match tc.get("layer_types").and_then(Json::as_array) {
        Some(t) => {
            if t.len() as u64 != n_layer {
                return Err(format!(
                    "layer_types has {} entries, expected {n_layer}",
                    t.len()
                ));
            }
            t.iter()
                .map(|v| v.as_str() == Some("linear_attention"))
                .collect()
        }
        None => (0..n_layer).map(|il| (il + 1) % interval != 0).collect(),
    };

    // Metadata, in the converter's order (the order does not matter to readers).
    let u = |v: u64| Value::U32(v as u32);
    let k = |s: &str| format!("{arch}.{s}");
    b.set("general.architecture", Value::String(arch.into()))?;
    b.set(&k("block_count"), u(n_layer + n_mtp))?;
    b.set(
        &k("context_length"),
        u(num_u(tc, "max_position_embeddings")?),
    )?;
    b.set(&k("embedding_length"), u(n_embd))?;
    if let Some(ff) = tc.get("intermediate_size").and_then(Json::as_u64) {
        b.set(&k("feed_forward_length"), u(ff))?;
    }
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
    if moe {
        b.set(&k("expert_count"), u(n_expert))?;
        b.set(
            &k("expert_used_count"),
            u(num_u(tc, "num_experts_per_tok")?),
        )?;
    }
    b.set(&k("attention.key_length"), u(head_dim))?;
    b.set(&k("attention.value_length"), u(head_dim))?;
    if n_mtp > 0 {
        b.set(&k("nextn_predict_layers"), u(n_mtp))?;
    }
    b.set(
        &k("ssm.conv_kernel"),
        u(num_u(tc, "linear_conv_kernel_dim")?),
    )?;
    b.set(&k("ssm.state_size"), u(dk))?;
    b.set(&k("ssm.group_count"), u(nk))?;
    b.set(&k("ssm.time_step_rank"), u(nv))?;
    b.set(&k("ssm.inner_size"), u(dv * nv))?;
    if tc.get("layer_types").is_some() {
        let mut flags: Vec<Value> = recurrent.iter().map(|&r| Value::Bool(r)).collect();
        flags.extend((0..n_mtp).map(|_| Value::Bool(false)));
        b.set(&k("attention.recurrent_layers"), Value::Array(flags))?;
    }
    b.set(&k("full_attention_interval"), u(interval))?;
    b.set(
        &k("rope.dimension_count"),
        u((head_dim as f64 * partial) as u64),
    )?;
    if moe {
        b.set(
            &k("expert_feed_forward_length"),
            u(num_u(tc, "moe_intermediate_size")?),
        )?;
        if let Some(sh) = tc
            .get("shared_expert_intermediate_size")
            .and_then(Json::as_u64)
        {
            b.set(&k("expert_shared_feed_forward_length"), u(sh))?;
        }
    }

    let vocab_size = num_u(tc, "vocab_size")?;
    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, config, vocab_size, "qwen35")?;

    // Tensors.
    let plus_one = |x: Vec<f32>| x.into_iter().map(|v| v + 1.0).collect::<Vec<f32>>();
    let norm = |b: &mut Builder, name: &str, src: &str| -> Result<(), String> {
        let (_, t) = b.source(src)?;
        b.computed_f32(name, src, vec![t.numel() as u64], plus_one)
    };
    b.direct(
        "token_embd.weight",
        &format!("{prefix}embed_tokens.weight"),
        None,
    )?;
    norm(b, "output_norm.weight", &format!("{prefix}norm.weight"))?;
    if b.has("lm_head.weight")
        && !tc
            .get("tie_word_embeddings")
            .and_then(Json::as_bool)
            .unwrap_or(false)
    {
        b.direct("output.weight", "lm_head.weight", None)?;
    }

    let order = v_order(nk, nv / nk);
    let reorder = nk != nv;
    for il in 0..n_layer + n_mtp {
        let src = if il < n_layer {
            format!("{prefix}layers.{il}.")
        } else {
            format!("mtp.layers.{}.", il - n_layer)
        };
        let blk = format!("blk.{il}.");
        let s = |t: &str| format!("{src}{t}");
        let d = |t: &str| format!("{blk}{t}");
        norm(b, &d("attn_norm.weight"), &s("input_layernorm.weight"))?;
        norm(
            b,
            &d("post_attention_norm.weight"),
            &s("post_attention_layernorm.weight"),
        )?;
        if moe {
            moe_ffn(
                b,
                &s,
                &d,
                n_embd,
                n_expert,
                num_u(tc, "moe_intermediate_size")?,
            )?;
        } else {
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
        }
        if il < n_layer && recurrent[il as usize] {
            linear_attention(b, &s, &d, n_embd, nk, nv, dk, dv, &order, reorder)?;
        } else {
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
        }
        if il >= n_layer {
            b.direct(&d("nextn.eh_proj.weight"), "mtp.fc.weight", None)?;
            norm(
                b,
                &d("nextn.enorm.weight"),
                "mtp.pre_fc_norm_embedding.weight",
            )?;
            norm(b, &d("nextn.hnorm.weight"), "mtp.pre_fc_norm_hidden.weight")?;
            norm(b, &d("nextn.shared_head_norm.weight"), "mtp.norm.weight")?;
        }
    }
    Ok(())
}

/// One MoE block: router, routed experts (fused or per expert) and the gated shared expert.
fn moe_ffn(
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
    if b.has(&s("mlp.experts.gate_up_proj")) {
        let src = s("mlp.experts.gate_up_proj");
        let (_, t) = b.source(&src)?;
        if t.shape != [n_expert as usize, 2 * ff as usize, n_embd as usize] {
            return Err(format!(
                "`{src}`: shape {:?}, expected [{n_expert}, {}, {n_embd}]",
                t.shape,
                2 * ff
            ));
        }
        let half = |first: u64| {
            (0..n_expert)
                .map(|e| (e * 2 * ff + first, ff))
                .collect::<Vec<_>>()
        };
        b.gather_rows(
            &d("ffn_gate_exps.weight"),
            &src,
            &half(0),
            vec![n_embd, ff, n_expert],
        )?;
        b.gather_rows(
            &d("ffn_up_exps.weight"),
            &src,
            &half(ff),
            vec![n_embd, ff, n_expert],
        )?;
        b.direct(
            &d("ffn_down_exps.weight"),
            &s("mlp.experts.down_proj"),
            None,
        )?;
    } else {
        let each = |w: &str| -> Vec<String> {
            (0..n_expert)
                .map(|e| s(&format!("mlp.experts.{e}.{w}.weight")))
                .collect()
        };
        b.stack(
            &d("ffn_gate_exps.weight"),
            &each("gate_proj"),
            vec![n_embd, ff, n_expert],
        )?;
        b.stack(
            &d("ffn_up_exps.weight"),
            &each("up_proj"),
            vec![n_embd, ff, n_expert],
        )?;
        b.stack(
            &d("ffn_down_exps.weight"),
            &each("down_proj"),
            vec![ff, n_embd, n_expert],
        )?;
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
    )?;
    Ok(())
}

/// One gated-DeltaNet layer's tensors.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_attention(
    b: &mut Builder,
    s: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    n_embd: u64,
    nk: u64,
    nv: u64,
    dk: u64,
    dv: u64,
    order: &[u64],
    reorder: bool,
) -> Result<(), String> {
    let qk_rows = 2 * nk * dk;
    let v_rows = nv * dv;
    let rows_of = |skip: u64, block: u64, total: u64| -> Vec<(u64, u64)> {
        if reorder {
            block_rows(skip, order, block)
        } else {
            vec![(0, total)]
        }
    };
    // Row-reordered BF16 matrices: QKV (V part only), the Z gate, alpha and beta.
    b.gather_rows(
        &d("attn_qkv.weight"),
        &s("linear_attn.in_proj_qkv.weight"),
        &rows_of(qk_rows, dv, qk_rows + v_rows),
        vec![n_embd, qk_rows + v_rows],
    )?;
    b.gather_rows(
        &d("attn_gate.weight"),
        &s("linear_attn.in_proj_z.weight"),
        &rows_of(0, dv, v_rows),
        vec![n_embd, v_rows],
    )?;
    b.gather_rows(
        &d("ssm_alpha.weight"),
        &s("linear_attn.in_proj_a.weight"),
        &rows_of(0, 1, nv),
        vec![n_embd, nv],
    )?;
    b.gather_rows(
        &d("ssm_beta.weight"),
        &s("linear_attn.in_proj_b.weight"),
        &rows_of(0, 1, nv),
        vec![n_embd, nv],
    )?;

    // Per-value-head vectors, F32.
    let ord = order.to_vec();
    let heads = move |x: Vec<f32>| {
        if reorder {
            permute_rows(&x, 1, 0, &ord, 1)
        } else {
            x
        }
    };
    let heads2 = heads.clone();
    b.computed_f32(&d("ssm_a"), &s("linear_attn.A_log"), vec![nv], move |x| {
        // Correctly rounded: exp in f64, one rounding. torch (which the converter uses) agrees
        // on all but ~1 in 1000 values, where its vectorized exp is 1 ulp off; f32::exp (libm)
        // misses more often.
        heads(x)
            .into_iter()
            .map(|v| -((v as f64).exp() as f32))
            .collect()
    })?;
    b.computed_f32(
        &d("ssm_dt.bias"),
        &s("linear_attn.dt_bias"),
        vec![nv],
        heads2,
    )?;

    // conv1d: [channels, 1, kernel] -> ggml [kernel, channels], F32, V channels reordered.
    let (_, conv) = b.source(&s("linear_attn.conv1d.weight"))?;
    let kernel = *conv.shape.last().unwrap() as u64;
    let channels = conv.shape[0] as u64;
    if channels != qk_rows + v_rows {
        return Err(format!(
            "conv1d has {channels} channels, expected {}",
            qk_rows + v_rows
        ));
    }
    let ord = order.to_vec();
    b.computed_f32(
        &d("ssm_conv1d.weight"),
        &s("linear_attn.conv1d.weight"),
        vec![kernel, channels],
        move |x| {
            if reorder {
                permute_rows(&x, kernel as usize, qk_rows as usize, &ord, dv as usize)
            } else {
                x
            }
        },
    )?;

    // The DeltaNet output norm is used as is (no +1).
    let (_, norm) = b.source(&s("linear_attn.norm.weight"))?;
    if norm.dtype == crate::safetensors::Dtype::F32 {
        b.direct(&d("ssm_norm.weight"), &s("linear_attn.norm.weight"), None)?;
    } else {
        b.computed_f32(
            &d("ssm_norm.weight"),
            &s("linear_attn.norm.weight"),
            vec![dv],
            |x| x,
        )?;
    }

    // out_proj: columns are value channels; reorder them per row.
    let ord = order.to_vec();
    let (_, out) = b.source(&s("linear_attn.out_proj.weight"))?;
    let elem = out.nbytes() / out.numel();
    let cols = v_rows as usize;
    b.computed_raw(
        &d("ssm_out.weight"),
        &s("linear_attn.out_proj.weight"),
        vec![v_rows, n_embd],
        move |bytes| {
            if !reorder {
                return bytes.to_vec();
            }
            let row = cols * elem;
            let mut out = Vec::with_capacity(bytes.len());
            for r in bytes.chunks_exact(row) {
                for &blk in &ord {
                    let start = blk as usize * dv as usize * elem;
                    out.extend_from_slice(&r[start..start + dv as usize * elem]);
                }
            }
            out
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_heads_go_from_grouped_to_tiled() {
        // 2 key heads, 3 value heads each: grouped k0v0 k0v1 k0v2 k1v0 k1v1 k1v2.
        assert_eq!(v_order(2, 3), vec![0, 3, 1, 4, 2, 5]);
        let x: Vec<u32> = (0..6).collect();
        assert_eq!(
            permute_rows(&x, 1, 0, &v_order(2, 3), 1),
            vec![0, 3, 1, 4, 2, 5]
        );
        // Two leading rows kept, blocks of two rows of width 2.
        let x: Vec<u32> = (0..2 * (2 + 4 * 2)).collect();
        let got = permute_rows(&x, 2, 2, &v_order(2, 2), 2);
        let row = |r: u32| [2 * r, 2 * r + 1];
        let want: Vec<u32> = [0, 1, 2, 3, 6, 7, 4, 5, 8, 9]
            .into_iter()
            .flat_map(row)
            .collect();
        assert_eq!(got, want);
        assert_eq!(
            block_rows(2, &v_order(2, 2), 2),
            vec![(0, 2), (2, 2), (6, 2), (4, 2), (8, 2)]
        );
    }
}
