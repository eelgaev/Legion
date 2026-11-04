//! Gemma 4 checkpoints (`gemma4` / `gemma4_text`: E2B / E4B, 26B-A4B, 31B) as the `gemma4` GGUF
//! llama.cpp's converter writes: `Gemma3Model` and `Gemma4Model` in `conversion/gemma.py`
//! (llama.cpp, MIT), the text model only (the vision and audio towers go to an mmproj GGUF).
//!
//! What the converter does beyond renaming, and how it is reproduced here:
//! - no `norm + 1` (Gemma 4's RMS norms use the weight as is, unlike Gemma 3's);
//! - 1-D tensors (norms, `layer_output_scale`, the router's input scale, the per-expert scales)
//!   and the router are F32 (widened from BF16 exactly);
//! - `rope_freqs.weight` is made up: the global layers' "proportional" RoPE rotates only the
//!   first `partial_rotary_factor` of each head's dimension pairs, which NEOX RoPE gets as
//!   frequency factors of 1 for those and 1e30 (no rotation) for the rest;
//! - the router becomes `ffn_gate_inp` (`.scale`: its input scale), its per-expert scale
//!   `ffn_down_exps.scale`, the experts' fused `gate_up_proj` `ffn_gate_up_exps` as is;
//! - the vocabulary is `gguf.LlamaHfVocab`'s: every token of `tokenizer.json` in id order, score
//!   -1000, `<0x..>` byte tokens BYTE, the tokenizer's special tokens CONTROL, the rest NORMAL,
//!   then the seven tokens the chat format's parser reads forced to USER_DEFINED (so they are
//!   rendered); its merges and special ids as `gguf.SpecialVocab` writes them.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::Value as Json;

use super::{Builder, num_u};
use crate::gguf::Value;

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

/// Tokens the converter always renders, so the chat parser can read them.
const VISIBLE: [&str; 7] = [
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
];

fn text_config(config: &Json) -> &Json {
    config.get("text_config").unwrap_or(config)
}

/// A key of `tc` that may be missing or null.
fn opt_u(tc: &Json, key: &str) -> Option<u64> {
    tc.get(key).and_then(Json::as_u64)
}

pub(super) fn build(b: &mut Builder, config: &Json) -> Result<(), String> {
    let tc = text_config(config);
    let n_layer = num_u(tc, "num_hidden_layers")?;
    let n_embd = num_u(tc, "hidden_size")?;
    let head_dim_swa = num_u(tc, "head_dim")?;
    let layer_types: Vec<bool> = tc
        .get("layer_types")
        .and_then(Json::as_array)
        .ok_or("config.json: missing `layer_types`")?
        .iter()
        .map(|t| t.as_str() == Some("sliding_attention"))
        .collect();
    if layer_types.len() as u64 != n_layer {
        return Err(format!(
            "layer_types has {} entries, expected {n_layer}",
            layer_types.len()
        ));
    }
    let head_dim_full = opt_u(tc, "global_head_dim").unwrap_or(head_dim_swa);
    let n_shared = opt_u(tc, "num_kv_shared_layers").unwrap_or(0);
    let rope = tc
        .get("rope_parameters")
        .ok_or("config.json: missing `rope_parameters`")?;
    let full = rope.get("full_attention").unwrap_or(rope);
    let swa_rope = rope.get("sliding_attention").unwrap_or(rope);
    let theta = |r: &Json| r.get("rope_theta").and_then(Json::as_f64);
    let n_expert = opt_u(tc, "num_experts").unwrap_or(0);

    let u = |v: u64| Value::U32(v as u32);
    let k = |s: &str| format!("gemma4.{s}");
    b.set("general.architecture", Value::String("gemma4".into()))?;
    b.set(&k("block_count"), u(n_layer))?;
    b.set(
        &k("context_length"),
        u(opt_u(tc, "max_position_embeddings").unwrap_or(131072)),
    )?;
    b.set(&k("embedding_length"), u(n_embd))?;
    let n_ff = num_u(tc, "intermediate_size")?;
    let double_wide = tc
        .get("use_double_wide_mlp")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    if double_wide {
        let first_shared = n_layer - n_shared;
        let arr = (0..n_layer)
            .map(|il| Value::I32((if il < first_shared { n_ff } else { 2 * n_ff }) as i32))
            .collect();
        b.set(&k("feed_forward_length"), Value::Array(arr))?;
    } else {
        b.set(&k("feed_forward_length"), u(n_ff))?;
    }
    b.set(
        &k("attention.head_count"),
        u(opt_u(tc, "num_attention_heads").unwrap_or(8)),
    )?;
    let kv_swa = opt_u(tc, "num_key_value_heads").unwrap_or(4);
    match opt_u(tc, "num_global_key_value_heads") {
        Some(kv_full) => {
            let arr = layer_types
                .iter()
                .map(|&s| Value::I32((if s { kv_swa } else { kv_full }) as i32))
                .collect();
            b.set(&k("attention.head_count_kv"), Value::Array(arr))?;
        }
        None => b.set(&k("attention.head_count_kv"), u(kv_swa))?,
    }
    b.set(
        &k("rope.freq_base"),
        Value::F32(theta(full).unwrap_or(1_000_000.0) as f32),
    )?;
    if let Some(t) = theta(swa_rope) {
        b.set(&k("rope.freq_base_swa"), Value::F32(t as f32))?;
    }
    b.set(
        &k("attention.layer_norm_rms_epsilon"),
        Value::F32(
            tc.get("rms_norm_eps")
                .and_then(Json::as_f64)
                .unwrap_or(1e-6) as f32,
        ),
    )?;
    if n_expert > 0 {
        b.set(&k("expert_count"), u(n_expert))?;
        b.set(&k("expert_used_count"), u(num_u(tc, "top_k_experts")?))?;
    }
    b.set(&k("attention.key_length"), u(head_dim_full))?;
    b.set(&k("attention.value_length"), u(head_dim_full))?;
    if let Some(c) = tc.get("final_logit_softcapping").and_then(Json::as_f64) {
        b.set(&k("final_logit_softcapping"), Value::F32(c as f32))?;
    }
    if tc.get("sliding_window_pattern").and_then(Json::as_u64) != Some(1) {
        b.set(
            &k("attention.sliding_window"),
            u(num_u(tc, "sliding_window")?),
        )?;
    }
    b.set(&k("attention.shared_kv_layers"), u(n_shared))?;
    b.set(
        &k("embedding_length_per_layer_input"),
        u(opt_u(tc, "hidden_size_per_layer_input").unwrap_or(0)),
    )?;
    b.set(
        &k("attention.sliding_window_pattern"),
        Value::Array(layer_types.iter().map(|&s| Value::Bool(s)).collect()),
    )?;
    b.set(&k("attention.key_length_swa"), u(head_dim_swa))?;
    b.set(&k("attention.value_length_swa"), u(head_dim_swa))?;
    if let Some(ff) = opt_u(tc, "expert_intermediate_size").or(opt_u(tc, "moe_intermediate_size")) {
        b.set(&k("expert_feed_forward_length"), u(ff))?;
    }
    let partial_swa = swa_rope
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(1.0);
    b.set(&k("rope.dimension_count"), u(head_dim_full))?;
    b.set(
        &k("rope.dimension_count_swa"),
        u((head_dim_swa as f64 * partial_swa) as u64),
    )?;

    let dir = b.ckpt.dir().to_path_buf();
    vocab(b, &dir, num_u(tc, "vocab_size")?)?;

    // Tensors.
    let prefix = if b.has("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };
    let p = |t: &str| format!("{prefix}{t}");
    // 1-D tensors and the router: F32.
    let f32_of = |b: &mut Builder, name: &str, src: &str| -> Result<(), String> {
        let (_, t) = b.source(src)?;
        b.computed_f32(name, src, super::ggml_shape(&t.shape), |x| x)
    };
    b.direct("token_embd.weight", &p("embed_tokens.weight"), None)?;
    if b.has(&p("embed_tokens_per_layer.weight")) {
        b.direct(
            "per_layer_token_embd.weight",
            &p("embed_tokens_per_layer.weight"),
            None,
        )?;
        b.direct(
            "per_layer_model_proj.weight",
            &p("per_layer_model_projection.weight"),
            None,
        )?;
        f32_of(
            b,
            "per_layer_proj_norm.weight",
            &p("per_layer_projection_norm.weight"),
        )?;
    }
    f32_of(b, "output_norm.weight", &p("norm.weight"))?;
    // Proportional RoPE of the global layers (see the module docs).
    let partial_full = full
        .get("partial_rotary_factor")
        .and_then(Json::as_f64)
        .unwrap_or(1.0);
    let n_rot = (head_dim_full as f64 * partial_full / 2.0) as usize;
    let half = (head_dim_full / 2) as usize;
    let factors: Vec<f32> = (0..half)
        .map(|i| if i < n_rot { 1.0 } else { 1e30 })
        .collect();
    b.constant_f32("rope_freqs.weight", factors)?;

    for il in 0..n_layer {
        let s = |t: &str| format!("{prefix}layers.{il}.{t}");
        let d = |t: &str| format!("blk.{il}.{t}");
        for (g, h) in [
            ("attn_q", "self_attn.q_proj"),
            ("attn_k", "self_attn.k_proj"),
            ("attn_v", "self_attn.v_proj"),
            ("attn_output", "self_attn.o_proj"),
            ("ffn_gate", "mlp.gate_proj"),
            ("ffn_up", "mlp.up_proj"),
            ("ffn_down", "mlp.down_proj"),
            ("inp_gate", "per_layer_input_gate"),
            ("proj", "per_layer_projection"),
        ] {
            let src = s(&format!("{h}.weight"));
            // Global layers with keys as values have no v_proj; shared-KV layers may lack k/v.
            // A compressed-tensors NVFP4 weight is only its `weight_packed` (RedHatAI's).
            if b.has(&src) || b.has(&s(&format!("{h}.weight_packed"))) {
                b.direct(&d(&format!("{g}.weight")), &src, None)?;
            }
        }
        for (g, h) in [
            ("attn_norm", "input_layernorm.weight"),
            ("attn_q_norm", "self_attn.q_norm.weight"),
            ("attn_k_norm", "self_attn.k_norm.weight"),
            ("post_attention_norm", "post_attention_layernorm.weight"),
            ("ffn_norm", "pre_feedforward_layernorm.weight"),
            ("post_ffw_norm", "post_feedforward_layernorm.weight"),
            ("post_norm", "post_per_layer_input_norm.weight"),
            ("layer_output_scale", "layer_scalar"),
            ("pre_ffw_norm_2", "pre_feedforward_layernorm_2.weight"),
            ("post_ffw_norm_1", "post_feedforward_layernorm_1.weight"),
            ("post_ffw_norm_2", "post_feedforward_layernorm_2.weight"),
        ] {
            if b.has(&s(h)) {
                f32_of(b, &d(&format!("{g}.weight")), &s(h))?;
            }
        }
        if b.has(&s("router.proj.weight")) {
            f32_of(b, &d("ffn_gate_inp.weight"), &s("router.proj.weight"))?;
            f32_of(b, &d("ffn_gate_inp.scale"), &s("router.scale"))?;
            f32_of(b, &d("ffn_down_exps.scale"), &s("router.per_expert_scale"))?;
            b.direct(
                &d("ffn_gate_up_exps.weight"),
                &s("experts.gate_up_proj"),
                None,
            )?;
            b.direct(&d("ffn_down_exps.weight"), &s("experts.down_proj"), None)?;
        }
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Option<Json>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(_) => Ok(None),
    }
}

/// The vocabulary (see the module docs).
fn vocab(b: &mut Builder, dir: &Path, vocab_size: u64) -> Result<(), String> {
    let tok = read_json(&dir.join("tokenizer.json"))?.ok_or("no tokenizer.json")?;
    let model = tok.get("model").ok_or("tokenizer.json: no `model`")?;
    let mut by_id: HashMap<u64, String> = HashMap::new();
    for (t, id) in model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or("tokenizer.json: no `model.vocab`")?
    {
        by_id.insert(id.as_u64().ok_or("tokenizer.json: bad id")?, t.clone());
    }
    let added: Vec<Json> = tok
        .get("added_tokens")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let mut id_of: HashMap<String, u64> = HashMap::new();
    for a in &added {
        if let (Some(id), Some(c)) = (
            a.get("id").and_then(Json::as_u64),
            a.get("content").and_then(Json::as_str),
        ) {
            by_id.insert(id, c.to_string());
        }
    }
    for (id, t) in &by_id {
        id_of.insert(t.clone(), *id);
    }
    let n = by_id.len() as u64;
    if n != vocab_size {
        return Err(format!(
            "tokenizer.json has {n} tokens, the model's vocab_size is {vocab_size}"
        ));
    }

    // The tokenizer's special tokens: every `*_token` of tokenizer_config.json, and its extra
    // and additional special tokens (transformers' `all_special_tokens`).
    let tc = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Json::Null);
    let content = |v: &Json| -> Option<String> {
        match v {
            Json::String(s) => Some(s.clone()),
            Json::Object(o) => o.get("content").and_then(Json::as_str).map(str::to_string),
            _ => None,
        }
    };
    let mut special: HashSet<u64> = HashSet::new();
    if let Some(o) = tc.as_object() {
        for (key, v) in o {
            if key.ends_with("_token")
                && let Some(id) = content(v).and_then(|c| id_of.get(&c).copied())
            {
                special.insert(id);
            }
            if key == "extra_special_tokens" || key == "additional_special_tokens" {
                for v in v.as_array().into_iter().flatten() {
                    if let Some(id) = content(v).and_then(|c| id_of.get(&c).copied()) {
                        special.insert(id);
                    }
                }
            }
        }
    }

    let mut tokens = Vec::with_capacity(n as usize);
    let mut types = Vec::with_capacity(n as usize);
    for id in 0..n {
        let t = by_id
            .get(&id)
            .ok_or_else(|| format!("tokenizer.json: no token {id}"))?;
        let is_byte = t.len() == 6
            && t.starts_with("<0x")
            && t.ends_with('>')
            && t[3..5].chars().all(|c| c.is_ascii_hexdigit());
        let ty = if VISIBLE.contains(&t.as_str()) {
            USER_DEFINED
        } else if is_byte {
            BYTE
        } else if special.contains(&id) {
            CONTROL
        } else {
            NORMAL
        };
        tokens.push(Value::String(t.clone()));
        types.push(Value::I32(ty));
    }
    b.set("tokenizer.ggml.model", Value::String("gemma4".into()))?;
    b.set("tokenizer.ggml.tokens", Value::Array(tokens))?;
    b.set(
        "tokenizer.ggml.scores",
        Value::Array(vec![Value::F32(-1000.0); n as usize]),
    )?;
    b.set("tokenizer.ggml.token_type", Value::Array(types))?;
    let merges: Vec<Value> = match model.get("merges").and_then(Json::as_array) {
        Some(m) => m
            .iter()
            .map(|e| match e {
                Json::String(s) => Ok(Value::String(s.clone())),
                Json::Array(pair) if pair.len() == 2 => Ok(Value::String(format!(
                    "{} {}",
                    pair[0].as_str().unwrap_or_default(),
                    pair[1].as_str().unwrap_or_default()
                ))),
                _ => Err("tokenizer.json: unknown merges format".to_string()),
            })
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    if !merges.is_empty() {
        b.set("tokenizer.ggml.merges", Value::Array(merges))?;
    }
    for (typ, stem) in [
        ("bos", "bos_token_id"),
        ("eos", "eos_token_id"),
        ("unk", "unknown_token_id"),
        ("pad", "padding_token_id"),
        ("mask", "mask_token_id"),
    ] {
        if let Some(id) = tc
            .get(format!("{typ}_token"))
            .and_then(content)
            .and_then(|c| id_of.get(&c).copied())
        {
            b.set(&format!("tokenizer.ggml.{stem}"), Value::U32(id as u32))?;
        }
    }
    b.set("tokenizer.ggml.add_space_prefix", Value::Bool(false))?;
    b.set("tokenizer.ggml.add_bos_token", Value::Bool(true))?;
    let template = match tc.get("chat_template") {
        Some(Json::String(s)) => Some(s.clone()),
        _ => std::fs::read_to_string(dir.join("chat_template.jinja")).ok(),
    };
    if let Some(t) = template {
        b.set("tokenizer.chat_template", Value::String(t))?;
    }
    Ok(())
}
