//! A tiny compressed-tensors Qwen3.5 checkpoint, laid out as llm-compressor writes a
//! "mixed-precision" one (the layout of unsloth/Qwen3.8-27B-NVFP4):
//! - layer 0: gated DeltaNet with channel-wise FP8 projections and an NVFP4 MLP;
//! - layer 1: full attention and an MLP, both channel-wise FP8;
//! - an FP8 LM head.
//!
//! NVFP4 weights are `weight_packed` with reciprocal `*_global_scale`s. FP8 weights have one BF16
//! `weight_scale` per output row and no `input_scale` (dynamic per-token activations).
//!
//! The test opens the checkpoint as shipped: NVFP4 repacked, and FP8 bytes raw with an F32
//! `.scale` per row that follows the rows the mapping reorders. With `Fp8Mode::Bf16` it is
//! dequantized row by row instead, and the model gives nearly the same logits on both.

use std::path::Path;

use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, Value};
use legion::hf::nvfp4::{dequant_fp8_bf16, repack_nvfp4};
use legion::hf::{Fp8Mode, OpenOptions};
use legion::model::core::Model;
use legion::weights::Weights;
use serde_json::json;

const EMBD: usize = 64;
const FF: usize = 128;
const HEAD: usize = 16;
const VOCAB: usize = 64;
const NK: usize = 2;
const NV: usize = 4;
const DK: usize = 16;
const DV: usize = 16;
const QKV: usize = 2 * NK * DK + NV * DV;

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

/// Small E4M3 scales (exponent 1..=4).
fn e4m3_scales(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (1 + (b >> 3) % 4) << 3 | (b & 7))
        .collect()
}

/// Small finite E4M3 weights.
fn e4m3_weights(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (b & 0x80) | (2 + (b >> 3) % 6) << 3 | (b & 7))
        .collect()
}

/// Small BF16 values of either sign.
fn bf16_small(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(2 * n, seed)
        .chunks(2)
        .flat_map(|c| [c[0], 0x3c | (c[1] & 0x80)])
        .collect()
}

fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 0x3f80u16.to_le_bytes()).collect()
}

/// `rows` distinct BF16 scales (2^-7 · (1 + r/128)), one per row.
fn row_scales(rows: usize) -> Vec<u8> {
    assert!(rows <= 128);
    (0..rows)
        .flat_map(|r| (0x3c00u16 + r as u16).to_le_bytes())
        .collect()
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    let (c, _) = b.as_chunks::<2>();
    c.iter()
        .map(|&c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16))
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    let (c, _) = b.as_chunks::<4>();
    c.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

/// Adds a tensor (name, dtype, shape, bytes) to the checkpoint being written.
type Add<'a> = dyn FnMut(String, &'static str, &[usize], Vec<u8>) + 'a;

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

/// The `quantization_config`; `a4`: the NVFP4 group declares 4-bit activations.
fn quantization_config(a4: bool) -> serde_json::Value {
    let fp4_act = if a4 {
        json!({"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"})
    } else {
        serde_json::Value::Null
    };
    json!({
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
                "input_activations": {"num_bits": 8, "type": "float", "dynamic": true,
                                      "strategy": "token"},
                "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$",
                            "re:.*linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)$",
                            "re:.*lm_head"]
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "weights": {"num_bits": 4, "type": "float", "group_size": 16,
                            "strategy": "tensor_group"},
                "input_activations": fp4_act,
                "targets": ["re:.*mlp\\.(gate|up|down)_proj$"]
            }
        }
    })
}

fn write_config(dir: &Path, a4: bool) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen3_5",
            "quantization_config": quantization_config(a4),
            "text_config": {
                "num_hidden_layers": 2, "hidden_size": EMBD, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": HEAD, "intermediate_size": FF,
                "linear_num_key_heads": NK, "linear_num_value_heads": NV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV,
                "linear_conv_kernel_dim": 4,
                "layer_types": ["linear_attention", "full_attention"],
                "full_attention_interval": 2, "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6, "vocab_size": VOCAB, "tie_word_embeddings": false,
                "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                    "mrope_section": [2, 2, 0]}
            }
        }))
        .unwrap(),
    )
    .unwrap();
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
    add(
        format!("{p}embed_tokens.weight"),
        "BF16",
        &[VOCAB, EMBD],
        bf16_small(VOCAB * EMBD, next()),
    );
    add(format!("{p}norm.weight"), "BF16", &[EMBD], bf16_ones(EMBD));
    let fp8 = |add: &mut Add, module: String, rows: usize, k: usize, seed: u64| {
        add(
            format!("{module}.weight"),
            "F8_E4M3",
            &[rows, k],
            e4m3_weights(rows * k, seed),
        );
        add(
            format!("{module}.weight_scale"),
            "BF16",
            &[rows, 1],
            row_scales(rows),
        );
    };
    for il in 0..2 {
        let l = format!("{p}layers.{il}.");
        for n in ["input_layernorm", "post_attention_layernorm"] {
            add(format!("{l}{n}.weight"), "BF16", &[EMBD], bf16_ones(EMBD));
        }
        let mlp = [
            ("gate_proj", FF, EMBD),
            ("up_proj", FF, EMBD),
            ("down_proj", EMBD, FF),
        ];
        if il == 0 {
            let a = format!("{l}linear_attn.");
            fp8(&mut add, format!("{a}in_proj_qkv"), QKV, EMBD, next());
            fp8(&mut add, format!("{a}in_proj_z"), NV * DV, EMBD, next());
            fp8(&mut add, format!("{a}out_proj"), EMBD, NV * DV, next());
            for n in ["in_proj_a", "in_proj_b"] {
                add(
                    format!("{a}{n}.weight"),
                    "BF16",
                    &[NV, EMBD],
                    bf16_small(NV * EMBD, next()),
                );
            }
            add(format!("{a}A_log"), "BF16", &[NV], bf16_small(NV, next()));
            add(format!("{a}dt_bias"), "BF16", &[NV], bf16_small(NV, next()));
            add(
                format!("{a}conv1d.weight"),
                "BF16",
                &[QKV, 1, 4],
                bf16_small(QKV * 4, next()),
            );
            add(format!("{a}norm.weight"), "BF16", &[DV], bf16_ones(DV));
            // NVFP4: global scales 2 (gate) and 1 (up, down) divide the values.
            for (proj, rows, k) in mlp {
                let m = format!("{l}mlp.{proj}");
                add(
                    format!("{m}.weight_packed"),
                    "U8",
                    &[rows, k / 2],
                    rand_bytes(rows * k / 2, next()),
                );
                add(
                    format!("{m}.weight_scale"),
                    "F8_E4M3",
                    &[rows, k / 16],
                    e4m3_scales(rows * k / 16, next()),
                );
                let g = if proj == "gate_proj" { 2f32 } else { 1.0 };
                add(
                    format!("{m}.weight_global_scale"),
                    "F32",
                    &[1],
                    g.to_le_bytes().to_vec(),
                );
                add(
                    format!("{m}.input_global_scale"),
                    "F32",
                    &[1],
                    4f32.to_le_bytes().to_vec(),
                );
            }
        } else {
            let a = format!("{l}self_attn.");
            for (proj, rows, k) in [
                ("q_proj", 2 * 2 * HEAD, EMBD),
                ("k_proj", HEAD, EMBD),
                ("v_proj", HEAD, EMBD),
                ("o_proj", EMBD, 2 * HEAD),
            ] {
                fp8(&mut add, format!("{a}{proj}"), rows, k, next());
            }
            for n in ["q_norm", "k_norm"] {
                add(format!("{a}{n}.weight"), "BF16", &[HEAD], bf16_ones(HEAD));
            }
            // The FP8 KV-cache scales the checkpoint ships; not used.
            for n in ["k_scale", "v_scale"] {
                add(format!("{a}{n}"), "BF16", &[1], bf16_ones(1));
            }
            for (proj, rows, k) in mlp {
                fp8(&mut add, format!("{l}mlp.{proj}"), rows, k, next());
            }
        }
    }
    fp8(&mut add, "lm_head".to_string(), VOCAB, EMBD, next());
    write_safetensors(&dir.join("model.safetensors"), &ts);
    write_config(dir, true);
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

/// GGUF FP8 weights, their source modules and shapes `(rows, k)`.
fn fp8_weights() -> Vec<(&'static str, &'static str, usize, usize)> {
    vec![
        (
            "blk.0.attn_qkv.weight",
            "0.linear_attn.in_proj_qkv",
            QKV,
            EMBD,
        ),
        (
            "blk.0.attn_gate.weight",
            "0.linear_attn.in_proj_z",
            NV * DV,
            EMBD,
        ),
        (
            "blk.0.ssm_out.weight",
            "0.linear_attn.out_proj",
            EMBD,
            NV * DV,
        ),
        ("blk.1.attn_q.weight", "1.self_attn.q_proj", 4 * HEAD, EMBD),
        ("blk.1.attn_k.weight", "1.self_attn.k_proj", HEAD, EMBD),
        ("blk.1.attn_v.weight", "1.self_attn.v_proj", HEAD, EMBD),
        (
            "blk.1.attn_output.weight",
            "1.self_attn.o_proj",
            EMBD,
            2 * HEAD,
        ),
        ("blk.1.ffn_gate.weight", "1.mlp.gate_proj", FF, EMBD),
        ("blk.1.ffn_up.weight", "1.mlp.up_proj", FF, EMBD),
        ("blk.1.ffn_down.weight", "1.mlp.down_proj", EMBD, FF),
        ("output.weight", "lm_head", VOCAB, EMBD),
    ]
}

fn source_name(module: &str) -> String {
    if module == "lm_head" {
        module.to_string()
    } else {
        format!("model.language_model.layers.{module}")
    }
}

#[test]
fn compressed_tensors_checkpoint_loads_as_shipped() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    let ty = |n: &str| g.tensor(n).map(|t| (t.ggml_type, t.shape));

    // FP8: raw bytes and an F32 scale per row, listed for the FP8 matmul.
    for (name, _, rows, k) in fp8_weights() {
        assert_eq!(
            ty(name),
            Some((types::I8, vec![k as u64, rows as u64])),
            "{name}"
        );
        let scale = name.replace(".weight", ".scale");
        assert_eq!(ty(&scale), Some((types::F32, vec![rows as u64])), "{scale}");
        assert!(g.tensor(&name.replace(".weight", ".input_scale")).is_none());
    }
    let Some(Value::Array(listed)) = g.get("legion.tensor_extra.fp8_e4m3") else {
        panic!("no FP8 list")
    };
    assert_eq!(listed.len(), fp8_weights().len());
    // NVFP4: repacked, `.scale` / `.input_scale` the reciprocals of the global scales (1 left
    // out); 4-bit activations declared, so no `prec_a4` list.
    for n in ["ffn_gate", "ffn_up", "ffn_down"] {
        assert_eq!(ty(&format!("blk.0.{n}.weight")).unwrap().0, types::NVFP4);
    }
    assert!(g.tensor("blk.0.ffn_up.scale").is_none());
    assert!(g.get("general.tensor_extra.prec_a4").is_none());

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let get = |n: &str| w.get(n).unwrap_or_else(|| panic!("no {n}")).bytes();
    let l0 = "model.language_model.layers.0.mlp.";
    for (name, proj, rows, k) in [
        ("blk.0.ffn_gate.weight", "gate_proj", FF, EMBD),
        ("blk.0.ffn_down.weight", "down_proj", EMBD, FF),
    ] {
        let packed = &src[&format!("{l0}{proj}.weight_packed")].bytes;
        let scales = &src[&format!("{l0}{proj}.weight_scale")].bytes;
        assert_eq!(get(name), repack_nvfp4(packed, scales, rows, k), "{name}");
    }
    assert_eq!(get("blk.0.ffn_gate.scale"), 0.5f32.to_le_bytes());
    assert_eq!(get("blk.0.ffn_gate.input_scale"), 0.25f32.to_le_bytes());

    // Every output row keeps its own source row's bytes and scale, through the DeltaNet value
    // head reorder of `in_proj_qkv` / `in_proj_z` (scales are distinct per source row).
    let mut moved = 0;
    for (name, module, rows, k) in fp8_weights() {
        let s = source_name(module);
        let (src_w, src_s) = (
            &src[&format!("{s}.weight")].bytes,
            bf16_to_f32(&src[&format!("{s}.weight_scale")].bytes),
        );
        let (out_w, out_s) = (get(name), f32s(&get(&name.replace(".weight", ".scale"))));
        assert_eq!(out_s.len(), rows);
        for (j, sj) in out_s.iter().enumerate() {
            let i = src_s
                .iter()
                .position(|v| v == sj)
                .expect("a scale of no row");
            moved += (i != j) as usize;
            let want = &src_w[i * k..(i + 1) * k];
            if name == "blk.0.ssm_out.weight" {
                // Columns reordered per value head, rows kept.
                assert_eq!(i, j, "{name}");
                let mut a = out_w[j * k..(j + 1) * k].to_vec();
                let mut b = want.to_vec();
                a.sort();
                b.sort();
                assert_eq!(a, b, "{name} row {j}");
            } else {
                assert_eq!(&out_w[j * k..(j + 1) * k], want, "{name} row {j}");
            }
        }
    }
    assert!(
        moved > 0,
        "no row was reordered: the test misses the gather"
    );
}

#[test]
fn compressed_tensors_w4a16_marks_nvfp4_weights() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    write_config(dir.path(), false);
    let g = GgufFile::open(dir.path()).unwrap();
    let Some(Value::Array(names)) = g.get("general.tensor_extra.name") else {
        panic!("no prec_a4 names")
    };
    let names: Vec<&str> = names.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight"
        ]
    );
    assert_eq!(
        g.get("general.tensor_extra.prec_a4"),
        Some(Value::Array(vec![Value::Bool(false); 3]))
    );
}

/// Raw FP8 with row scales in the matmul's epilogue against the weights dequantized row by row
/// to BF16 at load: the same model up to BF16 rounding of the weights.
#[test]
fn row_scaled_fp8_runs_like_its_bf16_dequant() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let raw = GgufFile::open(dir.path()).unwrap();
    let bf16 = GgufFile::open_with(dir.path(), &OpenOptions { fp8: Fp8Mode::Bf16 }).unwrap();
    assert!(bf16.get("legion.tensor_extra.fp8_e4m3").is_none());

    let backend = Backend::cpu().unwrap();
    {
        let w = Weights::load(&bf16, |_| backend.buffer_type()).unwrap();
        let s = source_name("1.self_attn.q_proj");
        let (q, qs) = (
            &src[&format!("{s}.weight")].bytes,
            bf16_to_f32(&src[&format!("{s}.weight_scale")].bytes),
        );
        let (rows, _) = q.as_chunks::<EMBD>();
        let want: Vec<u8> = rows
            .iter()
            .zip(&qs)
            .flat_map(|(r, &s)| dequant_fp8_bf16(r, s))
            .collect();
        assert_eq!(w.get("blk.1.attn_q.weight").unwrap().bytes(), want);
    }

    let mut backend = backend;
    backend.set_threads(2);
    let mut runs = Vec::new();
    for g in [&raw, &bf16] {
        let model = Model::load(g, |_| backend.buffer_type()).unwrap();
        let logits = model
            .prompt_logits_all(&mut backend, &[0, 1, 2, 1])
            .unwrap();
        assert_eq!(logits.len(), 4 * VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()));
        runs.push(logits);
    }
    let scale = runs[1].iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = runs[0]
        .iter()
        .zip(&runs[1])
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(scale > 0.0);
    assert!(
        diff < 2e-2 * scale,
        "raw vs BF16 dequant: {diff} (logits up to {scale})"
    );
}
