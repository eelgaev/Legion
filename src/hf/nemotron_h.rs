//! Nemotron-H checkpoints (`model_type: nemotron_h`; dense, or MoE with `num_experts_per_tok`) as
//! the `nemotron_h` / `nemotron_h_moe` GGUF llama.cpp's converter writes: `NemotronHModel` in
//! `conversion/nemotron.py` on top of `GraniteHybridModel` / `Mamba2Model` (llama.cpp, MIT).
//!
//! What the converter does beyond renaming, and how it is reproduced here:
//! - the block kinds (`layers_block_type` or `hybrid_override_pattern`) become the per-layer
//!   arrays `attention.head_count_kv` (kv heads for attention layers) and `feed_forward_length`
//!   (the FFN / expert size for FFN layers), 0 elsewhere;
//! - Mamba-2: `A_log` becomes `-exp(A_log)` `[1, n_head]`, `D` `[1, n_head]`, `conv1d.weight`
//!   drops its middle size-1 dim, `norm.weight` is grouped `[d_inner / n_group, n_group]`, all F32
//!   with the 1-D tensors and norms;
//! - MoE: the router is F32, `gate.e_score_correction_bias` is `exp_probs_b.bias`, the per-expert
//!   `up_proj` / `down_proj` are stacked into `ffn_{up,down}_exps` (for NVFP4 experts: each
//!   repacked, with the per-expert `weight_scale_2` as a `[n_expert]` `.scale`);
//! - q and k rows are permuted per head as for a RoPE model (`LlamaModel.permute`), though
//!   Nemotron-H has none;
//! - the MTP head (`mtp.layers.0`: input fusion and attention, `mtp.layers.1`: MoE and the final
//!   norm) is appended as block `n_layer`;
//! - quantized weights (ModelOpt NVFP4 / FP8) go through [`super::nvfp4`] like any other model's.

use std::sync::Arc;

use serde_json::Value as Json;

use super::{Builder, num_u};
use crate::ggml::types;
use crate::gguf::{Compute, Recipe, Value};
use crate::safetensors::Dtype;

/// What one layer of the checkpoint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Mamba,
    Attention,
    Ffn,
}

fn kinds(cfg: &Json, moe: bool) -> Result<Vec<Kind>, String> {
    if let Some(p) = cfg.get("hybrid_override_pattern").and_then(Json::as_str) {
        return p
            .chars()
            .map(|c| match c {
                'M' => Ok(Kind::Mamba),
                '*' => Ok(Kind::Attention),
                'E' if moe => Ok(Kind::Ffn),
                '-' if !moe => Ok(Kind::Ffn),
                other => Err(format!("hybrid_override_pattern: unknown block `{other}`")),
            })
            .collect();
    }
    cfg.get("layers_block_type")
        .and_then(Json::as_array)
        .ok_or("config.json: no layers_block_type or hybrid_override_pattern")?
        .iter()
        .map(|v| match v.as_str() {
            Some("mamba" | "linear_attention") => Ok(Kind::Mamba),
            Some("attention" | "full_attention") => Ok(Kind::Attention),
            Some("moe" | "mlp") => Ok(Kind::Ffn),
            other => Err(format!("layers_block_type: unknown block {other:?}")),
        })
        .collect()
}

pub(super) fn build(b: &mut Builder, cfg: &Json) -> Result<(), String> {
    let moe = cfg.get("num_experts_per_tok").is_some();
    let arch = if moe { "nemotron_h_moe" } else { "nemotron_h" };
    let blocks = kinds(cfg, moe)?;
    let n_layer = blocks.len() as u64;
    let n_mtp = if moe && b.has("mtp.layers.0.eh_proj.weight") {
        cfg.get("num_nextn_predict_layers")
            .and_then(Json::as_u64)
            .unwrap_or(0)
    } else {
        0
    };
    if n_mtp > 1 {
        return Err(format!("{n_mtp} MTP layers (only 1 is supported)"));
    }
    let n_embd = num_u(cfg, "hidden_size")?;
    let n_head = num_u(cfg, "num_attention_heads")?;
    let head_dim = cfg
        .get("head_dim")
        .and_then(Json::as_u64)
        .unwrap_or(n_embd / n_head);
    let n_kv = num_u(cfg, "num_key_value_heads")?;
    let ssm_heads = num_u(cfg, "mamba_num_heads")?;
    let ssm_head_dim = num_u(cfg, "mamba_head_dim")?;
    let d_inner = ssm_heads * ssm_head_dim;
    let n_group = num_u(cfg, "n_groups")?;
    let d_state = num_u(cfg, "ssm_state_size")?;
    let d_conv = num_u(cfg, "conv_kernel")?;
    let ff = if moe {
        num_u(cfg, "moe_intermediate_size")?
    } else {
        num_u(cfg, "intermediate_size")?
    };
    let eps = cfg
        .get("layer_norm_epsilon")
        .or_else(|| cfg.get("norm_eps"))
        .and_then(Json::as_f64)
        .ok_or("config.json: missing `layer_norm_epsilon`")? as f32;

    // Metadata, as the converter writes it (the order does not matter to readers).
    let u = |v: u64| Value::U32(v as u32);
    let k = |s: &str| format!("{arch}.{s}");
    let all = |f: &dyn Fn(Option<Kind>) -> u64| {
        Value::Array(
            (0..n_layer + n_mtp)
                .map(|il| Value::I32(f(blocks.get(il as usize).copied()) as i32))
                .collect(),
        )
    };
    b.set("general.architecture", Value::String(arch.into()))?;
    b.set(&k("block_count"), u(n_layer + n_mtp))?;
    b.set(
        &k("context_length"),
        u(num_u(cfg, "max_position_embeddings")?),
    )?;
    b.set(&k("embedding_length"), u(n_embd))?;
    // The MTP block has both an attention and a MoE sub-layer (`None` here).
    b.set(
        &k("feed_forward_length"),
        all(&|kind| match kind {
            Some(Kind::Ffn) | None => ff,
            _ => 0,
        }),
    )?;
    b.set(&k("attention.head_count"), u(n_head))?;
    b.set(
        &k("attention.head_count_kv"),
        all(&|kind| match kind {
            Some(Kind::Attention) | None => n_kv,
            _ => 0,
        }),
    )?;
    let theta = cfg
        .get("rope_theta")
        .and_then(Json::as_f64)
        .unwrap_or(10_000.0);
    b.set(&k("rope.freq_base"), Value::F32(theta as f32))?;
    b.set(&k("attention.layer_norm_rms_epsilon"), Value::F32(eps))?;
    b.set(&k("attention.layer_norm_epsilon"), Value::F32(eps))?;
    if moe {
        b.set(&k("expert_count"), u(num_u(cfg, "n_routed_experts")?))?;
        b.set(
            &k("expert_used_count"),
            u(num_u(cfg, "num_experts_per_tok")?),
        )?;
        b.set(&k("expert_group_count"), u(num_u(cfg, "n_group")?))?;
        b.set(&k("expert_group_used_count"), u(num_u(cfg, "topk_group")?))?;
    }
    let vocab_size = num_u(cfg, "vocab_size")?.div_ceil(8) * 8;
    b.set(&k("vocab_size"), u(vocab_size))?;
    b.set(&k("rope.dimension_count"), u(n_embd / n_head))?;
    b.set(&k("ssm.conv_kernel"), u(d_conv))?;
    b.set(&k("ssm.state_size"), u(d_state))?;
    b.set(&k("ssm.group_count"), u(n_group))?;
    b.set(&k("ssm.inner_size"), u(d_inner))?;
    b.set(&k("ssm.time_step_rank"), u(ssm_heads))?;
    b.set(&k("rope.scaling.finetuned"), Value::Bool(false))?;
    b.set(&k("attention.key_length"), u(head_dim))?;
    b.set(&k("attention.value_length"), u(head_dim))?;
    if moe {
        b.set(&k("expert_feed_forward_length"), u(ff))?;
        b.set(
            &k("expert_shared_feed_forward_length"),
            u(num_u(cfg, "moe_shared_expert_intermediate_size")?),
        )?;
        b.set(
            &k("expert_shared_count"),
            u(num_u(cfg, "n_shared_experts")?),
        )?;
        b.set(
            &k("expert_weights_norm"),
            Value::Bool(
                cfg.get("norm_topk_prob")
                    .and_then(Json::as_bool)
                    .unwrap_or(false),
            ),
        )?;
        b.set(
            &k("expert_weights_scale"),
            Value::F32(
                cfg.get("routed_scaling_factor")
                    .and_then(Json::as_f64)
                    .unwrap_or(1.0) as f32,
            ),
        )?;
    }
    if n_mtp > 0 {
        b.set(&k("nextn_predict_layers"), u(n_mtp))?;
    }
    let dir = b.ckpt.dir().to_path_buf();
    super::vocab::gpt2(b, &dir, cfg, vocab_size, "pixtral")?;

    // Tensors.
    let attn = [n_embd, n_head, n_kv, head_dim];
    let f32 = |b: &mut Builder, name: &str, src: &str, shape: Vec<u64>| -> Result<(), String> {
        b.computed_f32(name, src, shape, |x| x)
    };
    let vec = |b: &mut Builder, name: &str, src: &str| -> Result<(), String> {
        let (_, t) = b.source(src)?;
        f32(b, name, src, vec![t.numel() as u64])
    };
    b.direct("token_embd.weight", "backbone.embeddings.weight", None)?;
    vec(b, "output_norm.weight", "backbone.norm_f.weight")?;
    if b.has("lm_head.weight") {
        b.direct("output.weight", "lm_head.weight", None)?;
    }
    for (il, &kind) in blocks.iter().enumerate() {
        let s = |t: &str| format!("backbone.layers.{il}.{t}");
        let d = |t: &str| format!("blk.{il}.{t}");
        vec(b, &d("attn_norm.weight"), &s("norm.weight"))?;
        match kind {
            Kind::Mamba => {
                b.direct(&d("ssm_in.weight"), &s("mixer.in_proj.weight"), None)?;
                let conv_dim = d_inner + 2 * n_group * d_state;
                f32(
                    b,
                    &d("ssm_conv1d.weight"),
                    &s("mixer.conv1d.weight"),
                    vec![d_conv, conv_dim],
                )?;
                if b.has(&s("mixer.conv1d.bias")) {
                    vec(b, &d("ssm_conv1d.bias"), &s("mixer.conv1d.bias"))?;
                }
                vec(b, &d("ssm_dt.bias"), &s("mixer.dt_bias"))?;
                // Correctly rounded, as in qwen35.rs (the converter's torch exp may be 1 ulp off).
                b.computed_f32(&d("ssm_a"), &s("mixer.A_log"), vec![1, ssm_heads], |x| {
                    x.into_iter().map(|v| -((v as f64).exp() as f32)).collect()
                })?;
                f32(b, &d("ssm_d"), &s("mixer.D"), vec![1, ssm_heads])?;
                f32(
                    b,
                    &d("ssm_norm.weight"),
                    &s("mixer.norm.weight"),
                    vec![d_inner / n_group, n_group],
                )?;
                b.direct(&d("ssm_out.weight"), &s("mixer.out_proj.weight"), None)?;
            }
            Kind::Attention => attention(b, &s, &d, attn)?,
            Kind::Ffn if moe => moe_ffn(b, &s, &d, n_embd, ff, cfg)?,
            Kind::Ffn => {
                b.direct(&d("ffn_up.weight"), &s("mixer.up_proj.weight"), None)?;
                b.direct(&d("ffn_down.weight"), &s("mixer.down_proj.weight"), None)?;
            }
        }
    }
    if n_mtp > 0 {
        let il = n_layer;
        let d = |t: &str| format!("blk.{il}.{t}");
        let m0 = |t: &str| format!("mtp.layers.0.{t}");
        let m1 = |t: &str| format!("mtp.layers.1.{t}");
        vec(b, &d("nextn.enorm.weight"), &m0("enorm.weight"))?;
        vec(b, &d("nextn.hnorm.weight"), &m0("hnorm.weight"))?;
        b.direct(&d("nextn.eh_proj.weight"), &m0("eh_proj.weight"), None)?;
        vec(b, &d("attn_norm.weight"), &m0("norm.weight"))?;
        attention(b, &m0, &d, attn)?;
        vec(b, &d("post_attention_norm.weight"), &m1("norm.weight"))?;
        moe_ffn(b, &m1, &d, n_embd, ff, cfg)?;
        vec(
            b,
            &d("nextn.shared_head_norm.weight"),
            &m1("final_layernorm.weight"),
        )?;
    }
    Ok(())
}

/// The rows of a q/k projection of `heads` heads of `dim` rows each as llama.cpp's converter
/// permutes them (`LlamaModel.permute`, which `GraniteHybridModel` applies even though
/// Nemotron-H has no RoPE): within each head, the two halves are interleaved, so destination row
/// `2 i + j` is source row `j · dim / 2 + i`. q and k get the same order, so q·k is unchanged
/// up to summation order; matching it keeps the GGUF bytes, and llama.cpp's results, identical.
fn rope_rows(heads: u64, dim: u64) -> Vec<(u64, u64)> {
    let half = dim / 2;
    (0..heads)
        .flat_map(|h| (0..half).flat_map(move |i| (0..2).map(move |j| (h * dim + j * half + i, 1))))
        .collect()
}

fn attention(
    b: &mut Builder,
    s: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    [n_embd, n_head, n_kv, head_dim]: [u64; 4],
) -> Result<(), String> {
    for (g, h, heads) in [("attn_q", "q_proj", n_head), ("attn_k", "k_proj", n_kv)] {
        b.gather_rows(
            &d(&format!("{g}.weight")),
            &s(&format!("mixer.{h}.weight")),
            &rope_rows(heads, head_dim),
            vec![n_embd, heads * head_dim],
        )?;
    }
    for (g, h) in [("attn_v", "v_proj"), ("attn_output", "o_proj")] {
        b.direct(
            &d(&format!("{g}.weight")),
            &s(&format!("mixer.{h}.weight")),
            None,
        )?;
    }
    Ok(())
}

/// Router, selection bias, stacked routed experts and the shared expert of one MoE layer.
fn moe_ffn(
    b: &mut Builder,
    s: &dyn Fn(&str) -> String,
    d: &dyn Fn(&str) -> String,
    n_embd: u64,
    ff: u64,
    cfg: &Json,
) -> Result<(), String> {
    let n_expert = num_u(cfg, "n_routed_experts")?;
    b.computed_f32(
        &d("ffn_gate_inp.weight"),
        &s("mixer.gate.weight"),
        vec![n_embd, n_expert],
        |x| x,
    )?;
    let bias = s("mixer.gate.e_score_correction_bias");
    b.computed_f32(&d("exp_probs_b.bias"), &bias, vec![n_expert], |x| x)?;
    for (g, h, shape) in [
        ("ffn_up_exps", "up_proj", vec![n_embd, ff, n_expert]),
        ("ffn_down_exps", "down_proj", vec![ff, n_embd, n_expert]),
    ] {
        let srcs: Vec<String> = (0..n_expert)
            .map(|e| s(&format!("mixer.experts.{e}.{h}.weight")))
            .collect();
        let name = d(&format!("{g}.weight"));
        if b.source(&srcs[0])?.1.dtype == Dtype::U8 {
            stack_nvfp4(b, &name, &srcs, shape)?;
        } else {
            b.stack(&name, &srcs, shape)?;
        }
    }
    b.direct(
        &d("ffn_up_shexp.weight"),
        &s("mixer.shared_experts.up_proj.weight"),
        None,
    )?;
    b.direct(
        &d("ffn_down_shexp.weight"),
        &s("mixer.shared_experts.down_proj.weight"),
        None,
    )?;
    Ok(())
}

/// `name` (NVFP4, ggml `shape` `[k, rows, n_expert]`) is the ModelOpt NVFP4 experts `srcs`, each
/// repacked as [`super::nvfp4::repack_nvfp4`] does it, back to back; their `weight_scale_2`
/// become `<name>.scale` `[n_expert]` (left out when all are 1, as the converter does). Like the
/// converter, this does not list the stacked experts in `prec_a4`, so on Blackwell their
/// prefill matmuls may use 4-bit activations, as llama.cpp's do.
pub(super) fn stack_nvfp4(
    b: &mut Builder,
    name: &str,
    srcs: &[String],
    shape: Vec<u64>,
) -> Result<(), String> {
    let k = shape[0] as usize;
    let mut scales = Vec::with_capacity(srcs.len());
    let mut sources = Vec::with_capacity(srcs.len());
    let range = |file: usize, t: &crate::safetensors::TensorInfo| crate::gguf::FileRange {
        file,
        offset: t.data_range.start as u64,
        len: t.data_range.len() as u64,
    };
    for src in srcs {
        let base = src.strip_suffix(".weight").unwrap_or(src);
        let (wf, w) = b.source(src)?;
        let (sf, sc) = b.source(&format!("{base}.weight_scale"))?;
        if w.shape != [shape[1] as usize, k / 2] || sc.shape != [shape[1] as usize, k / 16] {
            return Err(format!(
                "`{src}`: NVFP4 weight {:?} / scales {:?} for {shape:?}",
                w.shape, sc.shape
            ));
        }
        sources.push((range(wf, &w), range(sf, &sc)));
        let s2 = b
            .ckpt
            .data(&format!("{base}.weight_scale_2"))
            .map_err(|e| e.to_string())?;
        if s2.len() != 4 {
            return Err(format!("`{base}.weight_scale_2` is not one F32"));
        }
        scales.push(f32::from_le_bytes(s2.try_into().unwrap()));
    }
    let rows = shape[1] as usize;
    let stack = crate::gguf::Nvfp4Stack {
        rows,
        k,
        experts: sources
            .into_iter()
            .zip(&scales)
            .map(|((weight, scale), &scale2)| crate::gguf::Nvfp4Source {
                weight,
                scale,
                scale2,
            })
            .collect(),
    };
    b.stacks.insert(name.to_string(), Arc::new(stack));
    let ckpt = b.ckpt.clone();
    let srcs = srcs.to_vec();
    let compute: Compute = Arc::new(move || {
        let mut out = Vec::new();
        for src in &srcs {
            let base = src.strip_suffix(".weight").unwrap_or(src);
            let w = ckpt.data(src).map_err(|e| e.to_string())?;
            let s = ckpt
                .data(&format!("{base}.weight_scale"))
                .map_err(|e| e.to_string())?;
            out.extend(super::nvfp4::repack_nvfp4(w, s, rows, k));
        }
        Ok(out)
    });
    b.push(name, types::NVFP4, shape, 0, 0);
    b.recipes
        .insert(name.to_string(), Recipe::Computed(compute));
    if scales.iter().any(|&v| (v - 1.0).abs() >= 1e-6) {
        let scale = name
            .strip_suffix(".weight")
            .map(|b| format!("{b}.scale"))
            .ok_or_else(|| format!("`{name}`: no .weight"))?;
        let bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let n = scales.len() as u64;
        b.push(&scale, types::F32, vec![n], 0, 0);
        b.recipes
            .insert(scale, Recipe::Computed(Arc::new(move || Ok(bytes.clone()))));
    }
    Ok(())
}
