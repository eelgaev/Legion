//! A Gemma 4 checkpoint opened as llama.cpp's converter would write it (`src/hf/gemma4.rs`): a
//! tiny synthetic MoE checkpoint with a sliding-window layer and a global layer whose keys are its
//! values (no `v_proj`), checked against what `conversion/gemma.py` writes. The real-model check
//! (E2B against its converted GGUF, byte for byte) is `tests/hf_checkpoint.rs`.

use std::path::Path;

use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, Value};
use legion::model::core::Model;
use legion::weights::Weights;
use serde_json::json;

const EMBD: usize = 64;
const HEADS: usize = 2;
const D_SWA: usize = 32;
const D_FULL: usize = 64;
const FF: usize = 32;
const EXPERTS: usize = 4;
const FF_EXP: usize = 64;

/// The vocabulary: the 256 byte tokens, the specials, two plain tokens.
fn vocab() -> Vec<&'static str> {
    let mut v = vec![
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "<mask>",
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|\"|>",
        "\u{2581}hello",
        "world",
    ];
    let bytes: Vec<&'static str> = (0..=255u8)
        .map(|b| &*Box::leak(format!("<0x{b:02X}>").into_boxed_str()))
        .collect();
    v.extend(bytes);
    v
}

fn rand_bf16(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6364136223846793005) | 1;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            f32::from_bits(v.to_bits() & 0xffff_0000)
        })
        .collect()
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        header.insert(
            name.clone(),
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend(bytes);
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

fn checkpoint(dir: &Path) {
    let v = vocab();
    let n_vocab = v.len();
    let config = json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 2,
            "hidden_size": EMBD,
            "intermediate_size": FF,
            "num_attention_heads": HEADS,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": D_SWA,
            "global_head_dim": D_FULL,
            "attention_k_eq_v": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "vocab_size": n_vocab,
            "num_experts": EXPERTS,
            "top_k_experts": 2,
            "expert_intermediate_size": FF_EXP,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let vocab_map: serde_json::Map<String, serde_json::Value> = v
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), json!(i)))
        .collect();
    let added: Vec<serde_json::Value> = v
        .iter()
        .enumerate()
        .take(15)
        .map(|(i, t)| json!({"id": i, "content": t, "special": true}))
        .collect();
    let tokenizer = json!({
        "model": {"type": "BPE", "vocab": vocab_map, "merges": [["\u{2581}hel", "lo"]], "byte_fallback": true},
        "added_tokens": added,
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let tc = json!({
        "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>",
        "mask_token": "<mask>", "sot_token": "<|turn>", "eot_token": "<turn|>",
        "think_token": "<|think|>", "soc_token": "<|channel>", "extra_special_tokens": [],
    });
    std::fs::write(dir.join("tokenizer_config.json"), tc.to_string()).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ bos_token }}").unwrap();

    let mut ts = Vec::new();
    let mut seed = 0;
    let mut t = |name: String, shape: &[usize]| {
        seed += 1;
        ts.push((
            name,
            shape.to_vec(),
            rand_bf16(shape.iter().product(), seed),
        ));
    };
    let p = "model.language_model.";
    t(format!("{p}embed_tokens.weight"), &[n_vocab, EMBD]);
    t(format!("{p}norm.weight"), &[EMBD]);
    for (il, d, kv) in [(0, D_SWA, 2), (1, D_FULL, 1)] {
        let l = format!("{p}layers.{il}.");
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            t(format!("{l}{n}.weight"), &[EMBD]);
        }
        t(format!("{l}self_attn.q_proj.weight"), &[HEADS * d, EMBD]);
        t(format!("{l}self_attn.k_proj.weight"), &[kv * d, EMBD]);
        if il == 0 {
            t(format!("{l}self_attn.v_proj.weight"), &[kv * d, EMBD]);
        }
        t(format!("{l}self_attn.o_proj.weight"), &[EMBD, HEADS * d]);
        t(format!("{l}self_attn.q_norm.weight"), &[d]);
        t(format!("{l}self_attn.k_norm.weight"), &[d]);
        t(format!("{l}mlp.gate_proj.weight"), &[FF, EMBD]);
        t(format!("{l}mlp.up_proj.weight"), &[FF, EMBD]);
        t(format!("{l}mlp.down_proj.weight"), &[EMBD, FF]);
        t(format!("{l}layer_scalar"), &[1]);
        t(format!("{l}router.proj.weight"), &[EXPERTS, EMBD]);
        t(format!("{l}router.scale"), &[EMBD]);
        t(format!("{l}router.per_expert_scale"), &[EXPERTS]);
        t(
            format!("{l}experts.gate_up_proj"),
            &[EXPERTS, 2 * FF_EXP, EMBD],
        );
        t(format!("{l}experts.down_proj"), &[EXPERTS, EMBD, FF_EXP]);
    }
    write_safetensors(&dir.join("model.safetensors"), &ts);
}

fn u32s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

#[test]
fn moe_checkpoint_opens_as_the_converter_writes_it() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    assert!(g.is_synthetic());
    let get = |k: &str| g.get(k).unwrap_or_else(|| panic!("missing {k}"));
    assert_eq!(g.architecture().as_deref(), Some("gemma4"));
    assert_eq!(get("gemma4.block_count").as_u64(), Some(2));
    assert_eq!(u32s(&get("gemma4.attention.head_count_kv")), [2, 1]);
    assert_eq!(
        get("gemma4.attention.key_length").as_u64(),
        Some(D_FULL as u64)
    );
    assert_eq!(
        get("gemma4.attention.key_length_swa").as_u64(),
        Some(D_SWA as u64)
    );
    assert_eq!(
        get("gemma4.rope.dimension_count").as_u64(),
        Some(D_FULL as u64)
    );
    assert_eq!(
        get("gemma4.rope.dimension_count_swa").as_u64(),
        Some(D_SWA as u64)
    );
    assert_eq!(get("gemma4.expert_count").as_u64(), Some(EXPERTS as u64));
    assert_eq!(get("gemma4.expert_used_count").as_u64(), Some(2));
    assert_eq!(
        get("gemma4.expert_feed_forward_length").as_u64(),
        Some(FF_EXP as u64)
    );
    assert_eq!(get("gemma4.attention.sliding_window").as_u64(), Some(8));
    assert_eq!(get("tokenizer.ggml.model").as_str(), Some("gemma4"));

    // Token types: specials CONTROL (3), the chat parser's tokens USER_DEFINED (4) even when
    // special, bytes BYTE (6), the rest NORMAL (1); scores all -1000.
    let types = get("tokenizer.ggml.token_type");
    let types: Vec<i64> = types
        .as_array()
        .unwrap()
        .iter()
        .map(|x| match x {
            Value::I32(t) => *t as i64,
            other => panic!("token type {other:?}"),
        })
        .collect();
    let v = vocab();
    for (i, t) in v.iter().enumerate() {
        let want = match *t {
            "<pad>" | "<eos>" | "<bos>" | "<unk>" | "<mask>" | "<|turn>" | "<turn|>"
            | "<|think|>" => 3,
            "<|channel>" | "<channel|>" | "<|tool_call>" | "<tool_call|>" | "<|tool_response>"
            | "<tool_response|>" | "<|\"|>" => 4,
            t if t.starts_with("<0x") => 6,
            _ => 1,
        };
        assert_eq!(types[i], want, "token {i} `{t}`");
    }
    let scores = get("tokenizer.ggml.scores");
    assert!(
        scores
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s.as_f64() == Some(-1000.0))
    );
    assert_eq!(get("tokenizer.ggml.bos_token_id").as_u64(), Some(2));
    assert_eq!(get("tokenizer.ggml.eos_token_id").as_u64(), Some(1));

    // Tensors: names, types, shapes.
    let info = |n: &str| g.tensor(n).unwrap_or_else(|| panic!("missing tensor {n}"));
    let check = |n: &str, ty, shape: &[u64]| {
        let t = info(n);
        assert_eq!((t.ggml_type, t.shape.as_slice()), (ty, shape), "{n}");
    };
    let (e, d, f) = (EMBD as u64, D_FULL as u64, FF_EXP as u64);
    check("token_embd.weight", types::BF16, &[e, v.len() as u64]);
    check("rope_freqs.weight", types::F32, &[d / 2]);
    check("blk.0.attn_v.weight", types::BF16, &[e, 2 * D_SWA as u64]);
    assert!(g.tensor("blk.1.attn_v.weight").is_none(), "keys as values");
    check("blk.1.attn_k.weight", types::BF16, &[e, d]);
    check(
        "blk.1.ffn_gate_inp.weight",
        types::F32,
        &[e, EXPERTS as u64],
    );
    check("blk.1.ffn_gate_inp.scale", types::F32, &[e]);
    check("blk.1.ffn_down_exps.scale", types::F32, &[EXPERTS as u64]);
    check(
        "blk.1.ffn_gate_up_exps.weight",
        types::BF16,
        &[e, 2 * f, EXPERTS as u64],
    );
    check(
        "blk.1.ffn_down_exps.weight",
        types::BF16,
        &[f, e, EXPERTS as u64],
    );
    check("blk.1.layer_output_scale.weight", types::F32, &[1]);
    check("blk.1.post_ffw_norm_2.weight", types::F32, &[e]);
    assert!(g.tensor("output.weight").is_none(), "the head is tied");

    // The proportional RoPE factors: a quarter of the dimensions rotate (pairs 0..8 of 32).
    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let rf = w.get("rope_freqs.weight").unwrap().to_f32();
    assert!(rf[..8].iter().all(|&x| x == 1.0));
    assert!(rf[8..].iter().all(|&x| x == 1e30));
    drop(w);

    // It runs.
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let model = Model::load(&g, |_| backend.buffer_type()).unwrap();
    let logits = model.prompt_logits(&mut backend, &[2, 15, 16, 15]).unwrap();
    assert_eq!(logits.len(), v.len());
    assert!(logits.iter().all(|x| x.is_finite() && x.abs() <= 30.0));
}
