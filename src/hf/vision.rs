//! The vision tower of a Qwen3.5 / Qwen3.8-Flash-Next checkpoint (`model.visual.*`, a Qwen3-VL
//! ViT) as the `clip` / `qwen3vl_merger` mmproj llama.cpp's converter writes for it
//! (`Qwen3VLVisionModel` in `conversion/qwen3vl.py` on `MmprojModel` in `conversion/base.py`,
//! llama.cpp, MIT), so that [`crate::vision::Encoder`] runs it as it runs an mmproj GGUF.
//!
//! The converter's renames: `blocks.{i}.norm1/2` -> `v.blk.{i}.ln1/2`, `attn.qkv` / `attn.proj`
//! -> `attn_qkv` / `attn_out`, `mlp.linear_fc1/2` -> `ffn_up/down`, `pos_embed` ->
//! `v.position_embd`, `merger.norm` -> `v.post_ln`, `merger.linear_fc1/2` -> `mm.0` / `mm.2`;
//! the Conv3D patch kernel `[n, 3, 2, p, p]` is split along time into `v.patch_embd.weight`
//! and `v.patch_embd.weight.1`. The layer-norm epsilon is the text model's `rms_norm_eps`, the
//! mean and std are the preprocessor's.
//!
//! BF16 matrices become F16, as in the converter's `--outtype f16` mmproj (unsloth's
//! `mmproj-F16.gguf`). Every 1-D tensor, the patch kernels and the position table are F32 (the
//! shipped values widened exactly): ggml's CPU adds and multiplies by them, and its `im2col`
//! takes no BF16 kernel. Deepstack layers are refused, as the encoder refuses them.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value as Json;

use super::{Builder, num_f, num_u, to_f32};
use crate::ggml::types;
use crate::gguf::{Compute, GgufError, GgufFile, Recipe, Value};
use crate::safetensors::{Checkpoint, Dtype};

/// The checkpoints whose `model.visual.*` is a Qwen3-VL tower without deepstack layers.
const MODEL_TYPES: [&str; 3] = ["qwen3_5", "qwen3_5_moe", "qwen4_exp"];

const PREFIX: &str = "model.visual.";

/// Gemma 4 checkpoints (#1217): `model.vision_tower.*` and `model.embed_vision.*`.
const GEMMA_MODEL_TYPE: &str = "gemma4";
const GEMMA_PREFIX: &str = "model.vision_tower.";

/// The vision tower of the checkpoint in `dir` as a synthetic mmproj, or `None` when it has
/// none (a text-only checkpoint, or a model type without a Qwen3-VL tower).
pub fn open_vision(dir: &Path) -> Result<Option<GgufFile>, GgufError> {
    let err = |reason: String| GgufError::Checkpoint {
        path: dir.to_path_buf(),
        reason,
    };
    let ckpt = Checkpoint::open(dir).map_err(|e| err(e.to_string()))?;
    let Some(config) = ckpt.config().cloned() else {
        return Ok(None);
    };
    let model_type = config.get("model_type").and_then(Json::as_str);
    let Some(vision) = config.get("vision_config").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if model_type == Some(GEMMA_MODEL_TYPE) {
        if ckpt
            .locate(&format!("{GEMMA_PREFIX}patch_embedder.input_proj.weight"))
            .is_none()
        {
            return Ok(None);
        }
        let mut b = Builder::new(Arc::new(ckpt));
        build_gemma(&mut b, vision, dir).map_err(err)?;
        return b.finish(dir.to_path_buf()).map(Some);
    }
    if !model_type.is_some_and(|t| MODEL_TYPES.contains(&t))
        || ckpt
            .locate(&format!("{PREFIX}patch_embed.proj.weight"))
            .is_none()
    {
        return Ok(None);
    }
    let mut b = Builder::new(Arc::new(ckpt));
    build(&mut b, &config, vision, dir).map_err(err)?;
    b.finish(dir.to_path_buf()).map(Some)
}

fn build(b: &mut Builder, config: &Json, v: &Json, dir: &Path) -> Result<(), String> {
    let text = super::qwen35::text_config(config);
    let vu = |k: &str| num_u(v, k).map_err(|e| format!("vision_config: {e}"));
    let depth = vu("depth")?;
    let width = vu("hidden_size")?;
    let patch = vu("patch_size")?;
    let merge = v
        .get("spatial_merge_size")
        .and_then(Json::as_u64)
        .unwrap_or(2);
    let temporal = v
        .get("temporal_patch_size")
        .and_then(Json::as_u64)
        .unwrap_or(2);
    if temporal != 2 {
        return Err(format!(
            "vision_config: temporal_patch_size {temporal} (only 2 is supported)"
        ));
    }
    if v.get("deepstack_visual_indexes")
        .and_then(Json::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return Err("vision_config: deepstack layers are not supported".into());
    }
    let n_pos = v
        .get("num_position_embeddings")
        .and_then(Json::as_u64)
        .unwrap_or(2304);
    let image_size = v
        .get("image_size")
        .and_then(Json::as_u64)
        .unwrap_or((n_pos as f64).sqrt() as u64 * patch);
    let (mean, std) = image_mean_std(dir)?;

    let u = Value::U32;
    b.set("general.architecture", Value::String("clip".into()))?;
    b.set("general.type", Value::String("mmproj".into()))?;
    b.set(
        "clip.projector_type",
        Value::String("qwen3vl_merger".into()),
    )?;
    b.set("clip.has_vision_encoder", Value::Bool(true))?;
    b.set("clip.use_gelu", Value::Bool(true))?;
    let n_out = v
        .get("out_hidden_size")
        .and_then(Json::as_u64)
        .map_or_else(|| num_u(text, "hidden_size"), Ok)?;
    for (k, val) in [
        ("projection_dim", u(n_out as u32)),
        ("image_size", u(image_size as u32)),
        ("patch_size", u(patch as u32)),
        ("embedding_length", u(width as u32)),
        ("feed_forward_length", u(vu("intermediate_size")? as u32)),
        ("block_count", u(depth as u32)),
        ("attention.head_count", u(vu("num_heads")? as u32)),
        (
            "attention.layer_norm_epsilon",
            Value::F32(num_f(text, "rms_norm_eps").unwrap_or(1e-6) as f32),
        ),
        ("spatial_merge_size", u(merge as u32)),
        ("image_mean", Value::Array(mean)),
        ("image_std", Value::Array(std)),
        (
            "is_deepstack_layers",
            Value::Array(vec![Value::Bool(false); depth as usize]),
        ),
    ] {
        b.set(&format!("clip.vision.{k}"), val)?;
    }

    let src = |s: &str| format!("{PREFIX}{s}");
    patch_kernels(b, &src("patch_embed.proj.weight"))?;
    widened(b, "v.patch_embd.bias", &src("patch_embed.proj.bias"))?;
    widened(b, "v.position_embd.weight", &src("pos_embed.weight"))?;
    for il in 0..depth {
        let blk = |s: &str| src(&format!("blocks.{il}.{s}"));
        let to = |s: &str| format!("v.blk.{il}.{s}");
        for (from, name) in [
            ("norm1", "ln1"),
            ("norm2", "ln2"),
            ("attn.qkv", "attn_qkv"),
            ("attn.proj", "attn_out"),
            ("mlp.linear_fc1", "ffn_up"),
            ("mlp.linear_fc2", "ffn_down"),
        ] {
            linear(b, &to(name), &blk(from))?;
        }
    }
    for (from, name) in [
        ("merger.norm", "v.post_ln"),
        ("merger.linear_fc1", "mm.0"),
        ("merger.linear_fc2", "mm.2"),
    ] {
        linear(b, name, &src(from))?;
    }
    Ok(())
}

/// A Gemma 4 checkpoint's tower (`vision_config` `v`) as the `gemma4v` mmproj llama.cpp's
/// converter writes for it (`Gemma4VisionAudioModel` in `conversion/gemma.py`, and its
/// `tensor_mapping.py` names, MIT): `encoder.layers.{i}.input_layernorm` /
/// `pre_feedforward_layernorm` -> `v.blk.{i}.ln1` / `ln2`, `post_attention_layernorm` /
/// `post_feedforward_layernorm` -> `attn_post_norm` / `ffn_post_norm`, `self_attn.{q,k,v,o}_proj`
/// -> `attn_q/k/v/out`, `self_attn.{q,k}_norm` -> `attn_{q,k}_norm`, `mlp.{gate,up,down}_proj`
/// -> `ffn_gate/up/down`; `patch_embedder.input_proj` (a linear over a patch's `p × p × 3`
/// values) -> the conv kernel `v.patch_embd.weight` (`[n, 3, p, p]`), its position table ->
/// `v.position_embd.weight` (F32, as the converter forces), `std_bias` / `std_scale` ->
/// `v.std_bias` / `v.std_scale`, and `model.embed_vision.embedding_projection` ->
/// `mm.input_projection.weight`. Clipped linears (`use_clipped_linears`) are refused, as the
/// encoder refuses them.
fn build_gemma(b: &mut Builder, v: &Json, dir: &Path) -> Result<(), String> {
    let vu = |k: &str| num_u(v, k).map_err(|e| format!("vision_config: {e}"));
    if v.get("use_clipped_linears").and_then(Json::as_bool) == Some(true) {
        return Err("vision_config: clipped linears are not supported".into());
    }
    let depth = vu("num_hidden_layers")?;
    let width = vu("hidden_size")?;
    let patch = vu("patch_size")?;
    let pool = v
        .get("pooling_kernel_size")
        .and_then(Json::as_u64)
        .unwrap_or(3);
    let (mean, std) = image_mean_std(dir)?;
    let u = Value::U32;
    b.set("general.architecture", Value::String("clip".into()))?;
    b.set("general.type", Value::String("mmproj".into()))?;
    b.set("clip.has_vision_encoder", Value::Bool(true))?;
    b.set("clip.use_gelu", Value::Bool(true))?;
    b.set(
        "clip.vision.projector_type",
        Value::String("gemma4v".into()),
    )?;
    for (k, val) in [
        ("image_size", u(224)),
        ("patch_size", u(patch as u32)),
        ("embedding_length", u(width as u32)),
        ("feed_forward_length", u(vu("intermediate_size")? as u32)),
        ("block_count", u(depth as u32)),
        ("attention.head_count", u(vu("num_attention_heads")? as u32)),
        (
            "attention.layer_norm_epsilon",
            Value::F32(num_f(v, "rms_norm_eps").unwrap_or(1e-6) as f32),
        ),
        ("projector.scale_factor", u(pool as u32)),
        ("image_mean", Value::Array(mean)),
        ("image_std", Value::Array(std)),
    ] {
        b.set(&format!("clip.vision.{k}"), val)?;
    }
    let src = |s: &str| format!("{GEMMA_PREFIX}{s}");
    gemma_patch_kernel(b, &src("patch_embedder.input_proj.weight"))?;
    widened(
        b,
        "v.position_embd.weight",
        &src("patch_embedder.position_embedding_table"),
    )?;
    widened(b, "v.std_bias", &src("std_bias"))?;
    widened(b, "v.std_scale", &src("std_scale"))?;
    weight(
        b,
        "mm.input_projection.weight",
        "model.embed_vision.embedding_projection.weight",
    )?;
    for il in 0..depth {
        let blk = |s: &str| src(&format!("encoder.layers.{il}.{s}.weight"));
        let to = |s: &str| format!("v.blk.{il}.{s}.weight");
        for (from, name) in [
            ("input_layernorm", "ln1"),
            ("pre_feedforward_layernorm", "ln2"),
            ("post_attention_layernorm", "attn_post_norm"),
            ("post_feedforward_layernorm", "ffn_post_norm"),
            ("self_attn.q_proj.linear", "attn_q"),
            ("self_attn.k_proj.linear", "attn_k"),
            ("self_attn.v_proj.linear", "attn_v"),
            ("self_attn.o_proj.linear", "attn_out"),
            ("self_attn.q_norm", "attn_q_norm"),
            ("self_attn.k_norm", "attn_k_norm"),
            ("mlp.gate_proj.linear", "ffn_gate"),
            ("mlp.up_proj.linear", "ffn_up"),
            ("mlp.down_proj.linear", "ffn_down"),
        ] {
            weight(b, &to(name), &blk(from))?;
        }
    }
    Ok(())
}

/// `name` from `src` without a bias: a 1-D weight widened to F32, a BF16 matrix F16, any
/// other as shipped ([`linear`]).
fn weight(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    if t.shape.len() == 1 {
        widened(b, name, src)
    } else if t.dtype == Dtype::BF16 {
        f16_matrix(b, name, src)
    } else {
        b.direct(name, src, None)
    }
}

/// Gemma 4's patch linear `src` (`[n, p × p × 3]`, a patch's values row by row, channels
/// last) as the conv kernel `v.patch_embd.weight` (`[n, 3, p, p]`, ggml `[p, p, 3, n]`), F32:
/// the converter's `reshape(n, p, p, 3).permute(0, 3, 1, 2)`.
fn gemma_patch_kernel(b: &mut Builder, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let [n, k] = t.shape[..] else {
        return Err(format!(
            "`{src}`: shape {:?}, expected [n, p * p * 3]",
            t.shape
        ));
    };
    let p = (1..=k).find(|p| p * p * 3 >= k).unwrap_or(0);
    if p * p * 3 != k {
        return Err(format!("`{src}`: {k} values a patch, not p * p * 3"));
    }
    let (n, p) = (n as usize, p as usize);
    let ckpt = b.ckpt.clone();
    let from = src.to_string();
    let compute: Compute = Arc::new(move || {
        let (_, t) = ckpt.locate(&from).ok_or("tensor disappeared")?;
        let bytes = ckpt.data(&from).map_err(|e| e.to_string())?;
        let x = to_f32(&t.dtype, bytes).ok_or_else(|| format!("`{from}`: not a float tensor"))?;
        let mut out = Vec::with_capacity(x.len() * 4);
        for row in x.chunks_exact(p * p * 3).take(n) {
            for c in 0..3 {
                for y in 0..p {
                    for xx in 0..p {
                        out.extend_from_slice(&row[(y * p + xx) * 3 + c].to_le_bytes());
                    }
                }
            }
        }
        Ok(out)
    });
    b.push(
        "v.patch_embd.weight",
        types::F32,
        vec![p as u64, p as u64, 3, n as u64],
        0,
        0,
    );
    b.recipes
        .insert("v.patch_embd.weight".to_string(), Recipe::Computed(compute));
    Ok(())
}

/// The image processor's mean and std (`preprocessor_config.json`, else
/// `processor_config.json`'s `image_processor`), as the converter reads them.
fn image_mean_std(dir: &Path) -> Result<(Vec<Value>, Vec<Value>), String> {
    let mut mean_std = None;
    for (file, key) in [
        ("preprocessor_config.json", None),
        ("processor_config.json", Some("image_processor")),
    ] {
        let Ok(s) = std::fs::read_to_string(dir.join(file)) else {
            continue;
        };
        let j: Json = serde_json::from_str(&s).map_err(|e| format!("{file}: {e}"))?;
        let j = key.and_then(|k| j.get(k)).unwrap_or(&j);
        let three = |k: &str| -> Option<Vec<Value>> {
            let a = j.get(k)?.as_array()?;
            (a.len() == 3)
                .then(|| {
                    a.iter()
                        .map(|x| Some(Value::F32(x.as_f64()? as f32)))
                        .collect()
                })
                .flatten()
        };
        if let (Some(m), Some(s)) = (three("image_mean"), three("image_std")) {
            mean_std = Some((m, s));
            break;
        }
    }
    mean_std.ok_or_else(|| {
        "no image_mean / image_std in preprocessor_config.json or processor_config.json".into()
    })
}

/// `name.weight` / `name.bias` from `src.weight` / `src.bias`: a 1-D weight (a norm's) widened
/// to F32 like the bias, a matrix F16 (or as shipped when it already is F16 or F32).
fn linear(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (w, wsrc) = (format!("{name}.weight"), format!("{src}.weight"));
    let (_, t) = b.source(&wsrc)?;
    if t.shape.len() == 1 {
        widened(b, &w, &wsrc)?;
    } else if t.dtype == Dtype::BF16 {
        f16_matrix(b, &w, &wsrc)?;
    } else {
        b.direct(&w, &wsrc, None)?;
    }
    widened(b, &format!("{name}.bias"), &format!("{src}.bias"))
}

/// `name` is the BF16 matrix `src` rounded to F16 (nearest even, as the converter's
/// `--outtype f16` rounds), its shape kept. Every BF16 value of a magnitude F16 holds as a
/// normal number (6.1e-5 to 65504) converts exactly. ggml's CPU multiplies F16 weights on its
/// fast paths; BF16 ones it widens element by element on hosts built without BF16 dot products
/// (a 1024 x 768 image took 89 s with BF16 weights on a GB10, 11.5 s with F16 ones).
fn f16_matrix(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let shape = super::ggml_shape(&t.shape);
    let ckpt = b.ckpt.clone();
    let from = src.to_string();
    let compute: Compute = Arc::new(move || {
        let (_, t) = ckpt.locate(&from).ok_or("tensor disappeared")?;
        let bytes = ckpt.data(&from).map_err(|e| e.to_string())?;
        let x = to_f32(&t.dtype, bytes).ok_or_else(|| format!("`{from}`: not a float tensor"))?;
        Ok(x.iter()
            .flat_map(|&v| crate::cpu::gpu_exact::f32_to_f16(v).to_le_bytes())
            .collect())
    });
    b.push(name, types::F16, shape, 0, 0);
    b.recipes
        .insert(name.to_string(), Recipe::Computed(compute));
    Ok(())
}

/// `name` is `src` widened to F32 (exactly), its shape kept.
fn widened(b: &mut Builder, name: &str, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let shape = super::ggml_shape(&t.shape);
    b.computed_f32(name, src, shape, |x| x)
}

/// The Conv3D patch kernel `src` (`[n, c, 2, p, p]`) as two 2-D kernels, F32: frame 0 is
/// `v.patch_embd.weight`, frame 1 `v.patch_embd.weight.1`.
fn patch_kernels(b: &mut Builder, src: &str) -> Result<(), String> {
    let (_, t) = b.source(src)?;
    let [n, c, 2, ph, pw] = t.shape[..] else {
        return Err(format!(
            "`{src}`: shape {:?}, expected [n, c, 2, p, p]",
            t.shape
        ));
    };
    let plane = ph * pw;
    for (frame, name) in ["v.patch_embd.weight", "v.patch_embd.weight.1"]
        .into_iter()
        .enumerate()
    {
        let ckpt = b.ckpt.clone();
        let src = src.to_string();
        let compute: Compute = Arc::new(move || {
            let (_, t) = ckpt.locate(&src).ok_or("tensor disappeared")?;
            let bytes = ckpt.data(&src).map_err(|e| e.to_string())?;
            let x =
                to_f32(&t.dtype, bytes).ok_or_else(|| format!("`{src}`: not a float tensor"))?;
            Ok(x.chunks_exact(2 * plane)
                .flat_map(|both| &both[frame * plane..(frame + 1) * plane])
                .flat_map(|v| v.to_le_bytes())
                .collect())
        });
        b.push(
            name,
            types::F32,
            vec![pw as u64, ph as u64, c as u64, n as u64],
            0,
            0,
        );
        b.recipes
            .insert(name.to_string(), Recipe::Computed(compute));
    }
    Ok(())
}
