//! A tiny ModelOpt-quantized Qwen3.5 checkpoint (one full-attention layer: FP8 attention, NVFP4
//! W4A16 MLP and LM head) opened as the GGUF llama.cpp's converter would write: NVFP4 weights
//! repacked into `GGML_TYPE_NVFP4`, per-tensor `.scale` / `.input_scale` tensors unless they are
//! 1, and the `prec_a4` list; FP8 weights kept as shipped (`I8` bytes, `.scale`, listed in
//! `legion.tensor_extra.fp8_e4m3`) or, with `Fp8Mode::Bf16`, dequantized to BF16. The byte-level transforms are
//! checked against the converter itself in `src/hf/nvfp4.rs`; this test checks the wiring, and
//! that the model runs.

use std::path::Path;

use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, Value};
use legion::hf::nvfp4::{dequant_fp8_bf16, repack_nvfp4};
use legion::hf::{Fp8Mode, OpenOptions};
use legion::model::core::{Fp8Act, Model};
use legion::weights::Weights;
use serde_json::json;

const EMBD: usize = 64;
const FF: usize = 128;
const HEAD: usize = 16;
const VOCAB: usize = 64;

/// Deterministic pseudo-random bytes.
fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

/// Small E4M3 scales (exponent 1..=4, either sign: the converter clears the sign bit).
fn e4m3_scales(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (b & 0x80) | (1 + (b >> 3) % 4) << 3 | (b & 7))
        .collect()
}

/// Small finite E4M3 weights.
fn e4m3_weights(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (b & 0x80) | (2 + (b >> 3) % 6) << 3 | (b & 7))
        .collect()
}

fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 0x3f80u16.to_le_bytes()).collect()
}

struct T {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn write_safetensors(path: &Path, tensors: &[T]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for t in tensors {
        header.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape,
                   "data_offsets": [data.len(), data.len() + t.bytes.len()]}),
        );
        data.extend(&t.bytes);
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

/// Writes the checkpoint; returns its tensors by name.
fn checkpoint(dir: &Path) -> std::collections::HashMap<String, T> {
    let mut ts = Vec::new();
    let mut seed = 0;
    let mut add = |name: String, dtype: &'static str, shape: &[usize], bytes: Vec<u8>| {
        ts.push(T {
            name,
            dtype,
            shape: shape.to_vec(),
            bytes,
        })
    };
    let mut next = || {
        seed += 1;
        seed
    };
    let p = "model.language_model.";
    let l = format!("{p}layers.0.");
    let embd: Vec<u8> = rand_bytes(2 * VOCAB * EMBD, next())
        .chunks(2)
        .flat_map(|c| [c[0], 0x3c | (c[1] & 0x80)]) // small BF16 values
        .collect();
    add(
        format!("{p}embed_tokens.weight"),
        "BF16",
        &[VOCAB, EMBD],
        embd,
    );
    for n in ["norm.weight", "layers.0.input_layernorm.weight"] {
        add(format!("{p}{n}"), "BF16", &[EMBD], bf16_ones(EMBD));
    }
    add(
        format!("{l}post_attention_layernorm.weight"),
        "BF16",
        &[EMBD],
        bf16_ones(EMBD),
    );
    add(
        format!("{l}self_attn.q_norm.weight"),
        "BF16",
        &[HEAD],
        bf16_ones(HEAD),
    );
    add(
        format!("{l}self_attn.k_norm.weight"),
        "BF16",
        &[HEAD],
        bf16_ones(HEAD),
    );
    // NVFP4 (W4A16): the MLP and the LM head. `up_proj` has a global scale of 1 (written as no
    // `.scale` tensor), the others do not.
    for (module, rows, k, scale2) in [
        (format!("{l}mlp.gate_proj"), FF, EMBD, 0.5f32),
        (format!("{l}mlp.up_proj"), FF, EMBD, 1.0),
        (format!("{l}mlp.down_proj"), EMBD, FF, 0.25),
        ("lm_head".to_string(), VOCAB, EMBD, 0.75),
    ] {
        add(
            format!("{module}.weight"),
            "U8",
            &[rows, k / 2],
            rand_bytes(rows * k / 2, next()),
        );
        add(
            format!("{module}.weight_scale"),
            "F8_E4M3",
            &[rows, k / 16],
            e4m3_scales(rows * k / 16, next()),
        );
        add(
            format!("{module}.weight_scale_2"),
            "F32",
            &[],
            scale2.to_le_bytes().to_vec(),
        );
        add(
            format!("{module}.input_scale"),
            "F32",
            &[],
            2f32.to_le_bytes().to_vec(),
        );
    }
    // FP8: attention.
    for (proj, rows, k) in [
        ("q_proj", 2 * 2 * HEAD, EMBD),
        ("k_proj", HEAD, EMBD),
        ("v_proj", HEAD, EMBD),
        ("o_proj", EMBD, 2 * HEAD),
    ] {
        let m = format!("{l}self_attn.{proj}");
        add(
            format!("{m}.weight"),
            "F8_E4M3",
            &[rows, k],
            e4m3_weights(rows * k, next()),
        );
        add(
            format!("{m}.weight_scale"),
            "F32",
            &[],
            0.0123f32.to_le_bytes().to_vec(),
        );
        add(
            format!("{m}.input_scale"),
            "F32",
            &[],
            0.25f32.to_le_bytes().to_vec(),
        );
    }
    write_safetensors(&dir.join("model.safetensors"), &ts);
    let layers: serde_json::Map<String, serde_json::Value> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|m| (format!("{l}mlp.{m}"), json!({"quant_algo": "W4A16_NVFP4"})))
        .chain(
            ["q_proj", "k_proj", "v_proj", "o_proj"]
                .iter()
                .map(|m| (format!("{l}self_attn.{m}"), json!({"quant_algo": "FP8"}))),
        )
        .chain([("lm_head".to_string(), json!({"quant_algo": "W4A16_NVFP4"}))])
        .collect();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen3_5",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": layers},
            "text_config": {
                "num_hidden_layers": 1, "hidden_size": EMBD, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": HEAD, "intermediate_size": FF,
                "linear_num_key_heads": 1, "linear_num_value_heads": 1,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "layer_types": ["full_attention"],
                "full_attention_interval": 1, "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6, "vocab_size": VOCAB, "tie_word_embeddings": false,
                "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                    "mrope_section": [2, 2, 0]}
            }
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
            "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}],
            "pre_tokenizer": null
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"eos_token": "<|im_end|>"})).unwrap(),
    )
    .unwrap();
    ts.into_iter().map(|t| (t.name.clone(), t)).collect()
}

#[test]
fn modelopt_checkpoint_opens_as_the_converter_would_write_it() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let bf16 = OpenOptions { fp8: Fp8Mode::Bf16 };
    let g = GgufFile::open_with(dir.path(), &bf16).unwrap();

    let ty = |n: &str| g.tensor(n).map(|t| (t.ggml_type, t.shape));
    let fp4 = [
        ("blk.0.ffn_gate.weight", "mlp.gate_proj", EMBD, FF),
        ("blk.0.ffn_up.weight", "mlp.up_proj", EMBD, FF),
        ("blk.0.ffn_down.weight", "mlp.down_proj", FF, EMBD),
        ("output.weight", "lm_head", EMBD, VOCAB),
    ];
    for (name, _, k, rows) in fp4 {
        assert_eq!(
            ty(name),
            Some((types::NVFP4, vec![k as u64, rows as u64])),
            "{name}"
        );
    }
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        assert_eq!(ty(&format!("blk.0.{name}.weight")).unwrap().0, types::BF16);
    }
    // Scales of 1 are left out; `input_scale` (2) is kept although nothing applies it.
    assert!(g.tensor("blk.0.ffn_up.scale").is_none());
    for n in [
        "blk.0.ffn_gate.scale",
        "blk.0.ffn_down.scale",
        "output.scale",
        "blk.0.ffn_up.input_scale",
    ] {
        assert_eq!(ty(n), Some((types::F32, vec![1])), "{n}");
    }
    assert!(g.tensor("blk.0.attn_q.scale").is_none());
    assert!(g.tensor("blk.0.attn_q.input_scale").is_none());
    let names: Vec<Value> = [
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "output.weight",
    ]
    .iter()
    .map(|s| Value::String(s.to_string()))
    .collect();
    assert_eq!(
        g.get("general.tensor_extra.name"),
        Some(Value::Array(names))
    );
    assert_eq!(
        g.get("general.tensor_extra.prec_a4"),
        Some(Value::Array(vec![Value::Bool(false); 4]))
    );

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let get = |n: &str| w.get(n).unwrap_or_else(|| panic!("no {n}")).bytes();
    let s = |m: &str| {
        let full = if m == "lm_head" {
            m.to_string()
        } else {
            format!("model.language_model.layers.0.{m}")
        };
        (
            &src[&format!("{full}.weight")].bytes,
            &src[&format!("{full}.weight_scale")].bytes,
        )
    };
    for (name, module, k, rows) in fp4 {
        let (wq, sc) = s(module);
        assert_eq!(get(name), repack_nvfp4(wq, sc, rows, k), "{name}");
    }
    assert_eq!(get("blk.0.ffn_gate.scale"), 0.5f32.to_le_bytes());
    assert_eq!(get("output.scale"), 0.75f32.to_le_bytes());
    let (q, _) = s("self_attn.q_proj");
    assert_eq!(get("blk.0.attn_q.weight"), dequant_fp8_bf16(q, 0.0123));
    drop(w);

    // The model runs on it.
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    let model = Model::load(&g, |_| backend.buffer_type()).unwrap();
    let logits = model.prompt_logits_all(&mut backend, &[0, 1, 2]).unwrap();
    assert_eq!(logits.len(), 3 * VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn fp8_weights_stay_raw_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    let names = ["attn_k", "attn_output", "attn_q", "attn_v"];
    for name in names {
        let t = g.tensor(&format!("blk.0.{name}.weight")).unwrap();
        assert_eq!(t.ggml_type, types::I8, "{name}");
        for scale in ["scale", "input_scale"] {
            let s = g.tensor(&format!("blk.0.{name}.{scale}")).unwrap();
            assert_eq!(
                (s.ggml_type, s.shape),
                (types::F32, vec![1]),
                "{name}.{scale}"
            );
        }
    }
    let Some(Value::Array(listed)) = g.get("legion.tensor_extra.fp8_e4m3") else {
        panic!("no FP8 list")
    };
    let mut listed: Vec<&str> = listed.iter().map(|v| v.as_str().unwrap()).collect();
    listed.sort();
    let want: Vec<String> = names.iter().map(|n| format!("blk.0.{n}.weight")).collect();
    assert_eq!(listed, want);
    // NVFP4 tensors are unaffected.
    assert_eq!(g.tensor("output.weight").unwrap().ggml_type, types::NVFP4);

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let q = &src["model.language_model.layers.0.self_attn.q_proj.weight"].bytes;
    assert_eq!(&w.get("blk.0.attn_q.weight").unwrap().bytes(), q);
    assert_eq!(
        w.get("blk.0.attn_q.scale").unwrap().bytes(),
        0.0123f32.to_le_bytes()
    );
    assert_eq!(
        w.get("blk.0.attn_q.input_scale").unwrap().bytes(),
        0.25f32.to_le_bytes()
    );
    drop(w);

    // The model runs the raw FP8 weights through the FP8 matmul: W8A8 with the static input
    // scale where `k` allows it (q/k/v; `attn_output` has k = 32 and stays W8A16), or W8A16.
    let mut backend = backend;
    let mut model = Model::load(&g, |_| backend.buffer_type()).unwrap();
    let mut runs = Vec::new();
    for act in [Fp8Act::A8, Fp8Act::A16] {
        model.set_fp8_act(act);
        let logits = model.prompt_logits(&mut backend, &[0, 1, 2]).unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        runs.push(logits);
    }
    assert_ne!(runs[0], runs[1], "a8 did not quantize the activations");
    let scale = runs[1].iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = runs[0]
        .iter()
        .zip(&runs[1])
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff < 0.1 * scale,
        "a8 vs a16: {diff} (logits up to {scale})"
    );
}
