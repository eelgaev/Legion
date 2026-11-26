//! Inkling-2 checkpoints (`inkling2_loop`, `Inkling2LoopForConditionalGeneration`; the text
//! model only) as the `inkling2_loop` GGUF that [`crate::model::inkling2`] reads: llama.cpp's
//! names and layouts (`convert_hf_to_gguf.py`'s `Inkling2Model` / `Inkling2LoopModel`, MIT,
//! ggml-org/llama.cpp PR #32694), the weights in the formats the checkpoint ships.
//!
//! - Dense projections, the embeddings and the head: BF16 as shipped.
//! - Norms, clip scales, conv taps, the relative-bias tables, the depth-mix queries and the
//!   router: F32 (BF16 widened exactly), as the converter forces them. The global layers'
//!   static conv taps (`static_w`, newest tap first) are flipped to the convs' order (oldest
//!   first); the relative-bias projection is transposed.
//! - The cross-pass message queries (`middle_cycle_cross_pass_{,kv_}queries`, two columns per
//!   core block) become each core block's `loop_msg_query` `[n_embd, 4]`: `[q 2i, kv 2i,
//!   kv 2i + 1, q 2i + 1]`.
//! - Routed experts shipped as NVFP4 (E2M1 codes two a byte, an E4M3 scale per 16, and an
//!   `amax / denominator` scale per expert): repacked losslessly into ggml `block_nvfp4` rows
//!   ([`super::nvfp4::repack_nvfp4`]) plus `<name>.scale` (F32 `[n_expert]`). Shipped as MXFP8
//!   (E4M3 bytes with an E8M0 scale per 32): the bytes as `I8` plus `<name>.scale`, the scales
//!   as F32 `[k / 32, m, n_expert]` (exactly `2^(e - 127)`). The checkpoint interleaves the
//!   gate and up rows of `w13`; they are taken gate rows first, then up rows.
//!
//! Not loaded: the vision tower (`model.visual.*`) and the MTP drafter (`mtp.safetensors`).

use std::sync::Arc;

use serde_json::Value as Json;

use super::qwen35::text_config;
use super::{Builder, ggml_shape, num_f, num_u};
use crate::ggml::types;
use crate::gguf::{Compute, Recipe, Value};
use crate::model::fp8::e8m0_to_f32;
use crate::safetensors::Dtype;

const ARCH: &str = "inkling2_loop";
const PREFIX: &str = "model.llm.";

/// llama.cpp's controller names, in the message queries' order, and the checkpoint's.
const CTRLS: [(&str, &str); 4] = [("q", "attn"), ("k", "k"), ("v", "v"), ("ffn", "mlp")];

/// `src` as F32 (BF16 / F16 widened exactly), ggml `shape` (default its own).
fn f32_tensor(
    b: &mut Builder,
    name: &str,
    src: &str,
    shape: Option<Vec<u64>>,
) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let shape = shape.unwrap_or_else(|| ggml_shape(&t.shape));
    if t.dtype == Dtype::F32 {
        return b.direct(name, src, Some(shape));
    }
    b.computed_f32(name, src, shape, |x| x)
}

/// A conv's taps `[c, 1, K]` (oldest first) as F32 ggml `[K, c]`.
fn conv_taps(b: &mut Builder, name: &str, src: &str, k: u64) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let c = match t.shape[..] {
        [c, 1, kk] if kk as u64 == k => c as u64,
        _ => return Err(format!("`{src}`: conv taps of shape {:?}", t.shape)),
    };
    f32_tensor(b, name, src, Some(vec![k, c]))
}

/// The global layers' static taps `static_w` `[K, c]` (newest first) as F32 ggml `[K, c]`
/// oldest first (the converter's `flip(0).T`).
fn static_taps(b: &mut Builder, name: &str, src: &str, k: u64) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    if t.shape.len() != 2 || t.shape[0] as u64 != k {
        return Err(format!("`{src}`: static taps of shape {:?}", t.shape));
    }
    let (k, c) = (k as usize, t.shape[1]);
    b.computed_f32(name, src, vec![k as u64, c as u64], move |x| {
        let mut out = vec![0f32; k * c];
        for ch in 0..c {
            for j in 0..k {
                out[ch * k + j] = x[(k - 1 - j) * c + ch];
            }
        }
        out
    })
}

/// The relative-bias projection `[n_rel, extent]` transposed: F32 ggml `[n_rel, extent]`.
fn rel_table(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let [nr, extent] = t.shape[..] else {
        return Err(format!("`{src}`: relative bias of shape {:?}", t.shape));
    };
    b.computed_f32(name, src, vec![nr as u64, extent as u64], move |x| {
        let mut out = vec![0f32; nr * extent];
        for r in 0..nr {
            for d in 0..extent {
                out[d * nr + r] = x[r * extent + d];
            }
        }
        out
    })
}

/// Core block `ip`'s message queries (`i = ip - core_start`): F32 `[n_embd, 4]` from the
/// checkpoint's two tables `[2 · core_len, n_embd]`.
fn msg_query(b: &mut Builder, name: &str, i: usize, n_embd: usize) -> Result<(), String> {
    let q = format!("{PREFIX}middle_cycle_cross_pass_queries");
    let kv = format!("{PREFIX}middle_cycle_cross_pass_kv_queries");
    for src in [&q, &kv] {
        let (_, t) = b.source(src)?;
        if t.dtype != Dtype::F32
            || t.shape.len() != 2
            || t.shape[1] != n_embd
            || t.shape[0] < 2 * i + 2
        {
            return Err(format!("`{src}`: {} {:?}", t.dtype.name(), t.shape));
        }
    }
    let ckpt = b.ckpt.clone();
    let compute: Compute = Arc::new(move || {
        let row = |src: &str, r: usize| -> Result<Vec<u8>, String> {
            let bytes = ckpt.data(src).map_err(|e| e.to_string())?;
            Ok(bytes[r * n_embd * 4..][..n_embd * 4].to_vec())
        };
        let mut out = row(&q, 2 * i)?;
        out.extend(row(&kv, 2 * i)?);
        out.extend(row(&kv, 2 * i + 1)?);
        out.extend(row(&q, 2 * i + 1)?);
        Ok(out)
    });
    b.push(name, types::F32, vec![n_embd as u64, 4], 0, 0);
    b.recipes
        .insert(name.to_string(), Recipe::Computed(compute));
    Ok(())
}

/// The source row of output row `j` of an expert matrix of `rows` rows: `w13` interleaves gate
/// and up rows (`2i` gate `i`, `2i + 1` up `i`); the output takes the gate rows, then the up rows.
fn src_row(j: usize, rows: usize, gate_up: bool) -> usize {
    if !gate_up {
        j
    } else if j < rows / 2 {
        2 * j
    } else {
        2 * (j - rows / 2) + 1
    }
}

/// Layer `il`'s routed expert stack `src` (`w13_weight` or `w2_weight`, `[n_expert, rows, ..]`)
/// as `name` (ggml `[k, rows, n_expert]`) and its `<name base>.scale`.
fn experts(
    b: &mut Builder,
    name: &str,
    src: &str,
    n_expert: usize,
    rows: usize,
    k: usize,
) -> Result<(), String> {
    let gate_up = src.ends_with(".w13_weight");
    let scale_name = format!("{}.scale", name.strip_suffix(".weight").unwrap_or(name));
    let scale_src = format!("{src}_scale");
    let (_, t) = b.source(src)?;
    let (_, st) = b.source(&scale_src)?;
    let shape = vec![k as u64, rows as u64, n_expert as u64];
    let ckpt = b.ckpt.clone();
    let (src, scale_src) = (src.to_string(), scale_src.clone());
    match (&t.dtype, &st.dtype) {
        (Dtype::U8, Dtype::F8E4M3) => {
            // NVFP4: block_nvfp4 rows, and the per-expert `amax / denominator`.
            if t.shape != [n_expert, rows, k / 2]
                || st.shape != [n_expert, rows, k / 16]
                || k % 64 != 0
            {
                return Err(format!(
                    "`{src}`: NVFP4 {:?} / {:?} for [{n_expert}, {rows}, {k}]",
                    t.shape, st.shape
                ));
            }
            let amax = format!("{src}_amax");
            let denom = format!("{src}_outer_scale_denominator");
            for n in [&amax, &denom] {
                let (_, ot) = b.source(n)?;
                if ot.dtype != Dtype::F32 {
                    return Err(format!("`{n}`: {:?}, want F32", ot.dtype));
                }
            }
            let compute: Compute = Arc::new(move || {
                let w = ckpt.data(&src).map_err(|e| e.to_string())?;
                let s = ckpt.data(&scale_src).map_err(|e| e.to_string())?;
                let (wr, sr) = (k / 2, k / 16);
                let row_out = k / 64 * 36;
                let mut out = vec![0u8; n_expert * rows * row_out];
                for (e, oe) in out.chunks_mut(rows * row_out).enumerate() {
                    for (j, o) in oe.chunks_mut(row_out).enumerate() {
                        let r = e * rows + src_row(j, rows, gate_up);
                        super::nvfp4::repack_nvfp4_into(
                            &w[r * wr..][..wr],
                            &s[r * sr..][..sr],
                            1,
                            k,
                            o,
                        );
                    }
                }
                Ok(out)
            });
            b.push(name, types::NVFP4, shape, 0, 0);
            b.recipes
                .insert(name.to_string(), Recipe::Computed(compute));
            let ckpt = b.ckpt.clone();
            let src = name.to_string();
            let compute: Compute = Arc::new(move || {
                let floats = |n: &str| -> Result<Vec<f32>, String> {
                    let bytes = ckpt.data(n).map_err(|e| e.to_string())?;
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect())
                };
                let a = floats(&amax)?;
                let ds = floats(&denom)?;
                let d = *ds
                    .first()
                    .ok_or_else(|| format!("`{src}`: no denominator"))?;
                // One value (a broadcast copy is fine); never drop a differing one.
                if ds.iter().any(|&x| x.to_bits() != d.to_bits()) {
                    return Err(format!("`{src}`: {} differing denominators", ds.len()));
                }
                if a.len() != n_expert || !(d.is_finite() && d > 0.0) {
                    return Err(format!("`{src}`: {} amax values, denominator {d}", a.len()));
                }
                Ok(a.iter().flat_map(|v| (v / d).to_le_bytes()).collect())
            });
            b.push(&scale_name, types::F32, vec![n_expert as u64], 0, 0);
            b.recipes.insert(scale_name, Recipe::Computed(compute));
        }
        (Dtype::F8E4M3, Dtype::U8 | Dtype::F8E8M0) => {
            // MXFP8: the E4M3 bytes as I8, the E8M0 scales as F32.
            if t.shape != [n_expert, rows, k] || st.shape != [n_expert, rows, k / 32] || k % 32 != 0
            {
                return Err(format!(
                    "`{src}`: MXFP8 {:?} / {:?} for [{n_expert}, {rows}, {k}]",
                    t.shape, st.shape
                ));
            }
            let permute = |row: usize| {
                move |bytes: &[u8]| -> Vec<u8> {
                    let mut out = Vec::with_capacity(bytes.len());
                    for e in 0..n_expert {
                        for j in 0..rows {
                            let r = e * rows + src_row(j, rows, gate_up);
                            out.extend_from_slice(&bytes[r * row..][..row]);
                        }
                    }
                    out
                }
            };
            let wp = permute(k);
            let compute: Compute =
                Arc::new(move || Ok(wp(ckpt.data(&src).map_err(|e| e.to_string())?)));
            b.push(name, types::I8, shape, 0, 0);
            b.recipes
                .insert(name.to_string(), Recipe::Computed(compute));
            let ckpt = b.ckpt.clone();
            let sp = permute(k / 32);
            let compute: Compute = Arc::new(move || {
                let s = sp(ckpt.data(&scale_src).map_err(|e| e.to_string())?);
                Ok(s.iter()
                    .flat_map(|&e| e8m0_to_f32(e).to_le_bytes())
                    .collect())
            });
            b.push(
                &scale_name,
                types::F32,
                vec![k as u64 / 32, rows as u64, n_expert as u64],
                0,
                0,
            );
            b.recipes.insert(scale_name, Recipe::Computed(compute));
        }
        (Dtype::BF16 | Dtype::F16 | Dtype::F32, _) => {
            return Err(format!(
                "`{src}`: unquantized experts are not supported yet"
            ));
        }
        (w, s) => {
            return Err(format!(
                "`{src}`: experts of {} with {} scales",
                w.name(),
                s.name()
            ));
        }
    }
    Ok(())
}

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: u64| Value::U32(v as u32);
    let f = |v: f64| Value::F32(v as f32);
    let lc = tc
        .get("loop_config")
        .ok_or("config.json: no `loop_config`")?;
    if lc.get("is_varloop").and_then(Json::as_bool) == Some(true) {
        return Err("Inkling-2 with a variable number of loop passes is not supported".into());
    }
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let (start, end) = (num_u(lc, "core_start")?, num_u(lc, "core_end")?);
    let n_embd = num_u(tc, "hidden_size")?;
    let n_head = num_u(tc, "num_attention_heads")?;
    let n_head_kv = num_u(tc, "num_key_value_heads")?;
    let head = num_u(tc, "head_dim")?;
    let n_expert = num_u(tc, "n_routed_experts")?;
    let n_ff = num_u(tc, "intermediate_size")?;
    let conv_k = num_u(tc, "sconv_kernel_size")?;
    if num_u(tc, "n_shared_experts").unwrap_or(0) != 0 {
        return Err("Inkling-2 with shared experts is not supported".into());
    }
    for (key, want) in [
        ("swa_head_dim", head),
        ("swa_num_attention_heads", n_head),
        ("swa_num_key_value_heads", n_head_kv),
    ] {
        if tc.get(key).is_some() && num_u(tc, key)? != want {
            return Err(format!(
                "Inkling-2 with `{key}` other than the global layers'"
            ));
        }
    }
    let local: Vec<u64> = tc
        .get("local_layer_ids")
        .and_then(Json::as_array)
        .ok_or("config.json: no `local_layer_ids`")?
        .iter()
        .map(|v| v.as_u64().ok_or("config.json: `local_layer_ids`"))
        .collect::<Result<_, _>>()?;
    let swa: Vec<bool> = (0..n_layer).map(|il| local.contains(&il)).collect();

    b.set("general.architecture", Value::String(ARCH.into()))?;
    b.set(&k("block_count"), u(n_layer))?;
    b.set(&k("context_length"), u(num_u(tc, "model_max_length")?))?;
    b.set(&k("embedding_length"), u(n_embd))?;
    b.set(&k("feed_forward_length"), u(n_ff))?;
    b.set(&k("expert_feed_forward_length"), u(n_ff))?;
    b.set(&k("attention.head_count"), u(n_head))?;
    b.set(&k("attention.head_count_kv"), u(n_head_kv))?;
    b.set(
        &k("attention.layer_norm_rms_epsilon"),
        f(num_f(tc, "rms_norm_eps")?),
    )?;
    b.set(&k("expert_count"), u(n_expert))?;
    b.set(
        &k("expert_used_count"),
        u(num_u(tc, "num_experts_per_tok")?),
    )?;
    for key in [
        "key_length",
        "value_length",
        "key_length_swa",
        "value_length_swa",
    ] {
        b.set(&k(&format!("attention.{key}")), u(head))?;
    }
    b.set(&k("loop.count"), u(num_u(lc, "num_passes")?))?;
    b.set(&k("loop.start"), u(start))?;
    b.set(&k("loop.end"), u(end))?;
    b.set(&k("loop.gate_rank"), u(num_u(lc, "gate_rank")?))?;
    b.set(&k("loop.varying"), Value::Bool(false))?;
    let vocab_size = num_u(tc, "unpadded_vocab_size")?;
    b.set(&k("vocab_size"), u(num_u(tc, "vocab_size")?))?;
    b.set(&k("vocab_size_unpadded"), u(vocab_size))?;
    b.set(&k("rope.dimension_count"), u(0))?;
    b.set(
        &k("attention.sliding_window"),
        u(num_u(tc, "sliding_window_size")?),
    )?;
    b.set(
        &k("attention.sliding_window_pattern"),
        Value::Array(swa.iter().map(|&s| Value::Bool(s)).collect()),
    )?;
    b.set(&k("shortconv.l_cache"), u(conv_k))?;
    b.set(&k("expert_weights_scale"), f(num_f(tc, "route_scale")?))?;
    b.set(
        &k("activation.situ_beta"),
        f(num_f(tc, "mlp_gate_softcap")?),
    )?;
    b.set(
        &k("activation.situ_linear_beta"),
        f(num_f(tc, "mlp_up_softcap")?),
    )?;
    b.set(
        &k("attn_res.block_size"),
        u(num_u(tc, "n_layers_per_attn_res_block")?),
    )?;
    b.set(
        &k("final_logit_softcapping"),
        f(num_f(tc, "final_logit_softcapping")?),
    )?;
    b.set(&k("attention.relative_length"), u(num_u(tc, "d_rel")?))?;
    b.set(&k("attention.relative_extent"), u(num_u(tc, "rel_extent")?))?;
    b.set(
        &k("attention.qk_clip"),
        f(num_f(tc, "qk_l2_clip_threshold")?),
    )?;
    b.set(
        &k("attention.relative_clip"),
        f(num_f(tc, "rel_l2_clip_threshold")?),
    )?;
    b.set(
        &k("attention.log_scale_floor"),
        u(num_u(tc, "log_scaling_n_floor")?),
    )?;
    b.set(
        &k("attention.log_scale_alpha"),
        f(num_f(tc, "log_scaling_alpha")?),
    )?;

    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, config, vocab_size, "inkling2")?;
    // The end of a model turn (`<|content_model_end_sampling|>`) is also its end of turn.
    if let Some(eos) = config.get("eos_token_id").and_then(Json::as_u64) {
        b.set("tokenizer.ggml.eot_token_id", u(eos))?;
    }

    // Tensors.
    let g = |t: &str| format!("{PREFIX}{t}");
    b.direct("token_embd.weight", &g("embed.weight"), None)?;
    f32_tensor(b, "token_embd_norm.weight", &g("embed_norm.weight"), None)?;
    b.direct("output.weight", &g("unembed.weight"), None)?;
    f32_tensor(b, "output_norm.weight", &g("norm.weight"), None)?;
    f32_tensor(
        b,
        "output_res_query.weight",
        &g("attn_res.output_query"),
        None,
    )?;
    for il in 0..n_layer {
        let s = |t: &str| format!("{PREFIX}layers.{il}.{t}");
        let d = |t: &str| format!("blk.{il}.{t}");
        for (to, from) in [
            ("attn_q", "attn.wq_du"),
            ("attn_k", "attn.wk_dv"),
            ("attn_v", "attn.wv_dv"),
            ("attn_output", "attn.wo_ud"),
            ("attn_gate", "attn.gate_attn"),
            ("attn_rel", "attn.wr_du"),
            ("attn_sink_proj", "attn.attn_sink_proj"),
        ] {
            b.direct(
                &d(&format!("{to}.weight")),
                &s(&format!("{from}.weight")),
                None,
            )?;
        }
        f32_tensor(
            b,
            &d("attn_q_clip.weight"),
            &s("attn.q_l2_clip.scale"),
            None,
        )?;
        f32_tensor(
            b,
            &d("attn_rel_clip.weight"),
            &s("attn.rel_l2_clip.scale"),
            None,
        )?;
        rel_table(
            b,
            &d("attn_rel_bias.weight"),
            &s("attn.rel_logits_proj.proj"),
        )?;
        conv_taps(
            b,
            &d("attn_post_conv.weight"),
            &s("attn_sconv.weight"),
            conv_k,
        )?;
        conv_taps(
            b,
            &d("ffn_post_conv.weight"),
            &s("mlp_sconv.weight"),
            conv_k,
        )?;
        if swa[il as usize] {
            f32_tensor(b, &d("attn_norm.weight"), &s("attn_norm.weight"), None)?;
            conv_taps(
                b,
                &d("attn_k_conv.weight"),
                &s("attn.k_sconv.weight"),
                conv_k,
            )?;
            conv_taps(
                b,
                &d("attn_v_conv.weight"),
                &s("attn.v_sconv.weight"),
                conv_k,
            )?;
        } else {
            for c in ["q", "k", "v"] {
                f32_tensor(
                    b,
                    &d(&format!("attn_{c}_inp_norm.weight")),
                    &s(&format!("{c}_input_norm.weight")),
                    None,
                )?;
            }
            for c in ["k", "v"] {
                static_taps(
                    b,
                    &d(&format!("attn_{c}_conv.weight")),
                    &s(&format!("attn.{c}_sconv.static_w")),
                    conv_k,
                )?;
                b.direct(
                    &d(&format!("attn_{c}_conv_proj.weight")),
                    &s(&format!("attn.{c}_sconv.weight_proj")),
                    None,
                )?;
            }
        }
        // `attn_queries.{i}` feeds block `i + 1`.
        if il > 0 {
            f32_tensor(
                b,
                &d("attn_res_query.weight"),
                &g(&format!("attn_res.attn_queries.{}", il - 1)),
                None,
            )?;
        }
        f32_tensor(
            b,
            &d("ffn_res_query.weight"),
            &g(&format!("attn_res.mlp_queries.{il}")),
            None,
        )?;
        f32_tensor(b, &d("ffn_norm.weight"), &s("mlp_norm.weight"), None)?;
        f32_tensor(b, &d("ffn_gate_inp.weight"), &s("mlp.gate.weight"), None)?;
        f32_tensor(b, &d("exp_probs_b.bias"), &s("mlp.gate.bias"), None)?;
        experts(
            b,
            &d("ffn_gate_up_exps.weight"),
            &s("mlp.experts.w13_weight"),
            n_expert as usize,
            2 * n_ff as usize,
            n_embd as usize,
        )?;
        experts(
            b,
            &d("ffn_down_exps.weight"),
            &s("mlp.experts.w2_weight"),
            n_expert as usize,
            n_embd as usize,
            n_ff as usize,
        )?;
        if (start..end).contains(&il) {
            msg_query(
                b,
                &d("loop_msg_query.weight"),
                (il - start) as usize,
                n_embd as usize,
            )?;
            for (to, from) in CTRLS {
                for part in ["down", "gain", "forget"] {
                    f32_tensor(
                        b,
                        &d(&format!("loop_{to}_ctrl_{part}.weight")),
                        &s(&format!("controllers.{from}.{part}_weight")),
                        None,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
