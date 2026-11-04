//! A tiny ModelOpt-quantized Nemotron-H MoE checkpoint (Nemotron 3.5's layout: FP8 Mamba-2
//! in/out projections, NVFP4 W4A16 routed and shared experts and LM head, BF16 attention)
//! opened as a synthetic GGUF (`src/hf/nemotron_h.rs`): the block arrays, the Mamba-2 tensor
//! transforms, the NVFP4 experts stacked with their per-expert scales, the FP8 weights kept as
//! shipped; and the model runs with them, close to the same checkpoint with FP8 dequantized to
//! BF16 (`--hf-fp8 bf16`, what llama.cpp's converter writes).
//!
//! The real checkpoint against the converter's GGUF, byte for byte, is `tests/hf_checkpoint.rs`'s
//! ignored test (`LEGION_TEST_HF_DIR` / `LEGION_TEST_HF_REF`).

use std::path::Path;

use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, Value};
use legion::hf::nvfp4::repack_nvfp4;
use legion::hf::{Fp8Mode, OpenOptions};
use legion::model::core::{Fp8Act, Model};
use legion::model::nemotron_h::Block;
use legion::weights::Weights;
use serde_json::json;

const EMBD: usize = 64;
const HEADS: usize = 4;
const HEAD_DIM: usize = 16;
const KV: usize = 2;
const SSM_HEADS: usize = 4;
const SSM_HEAD_DIM: usize = 16;
const GROUPS: usize = 2;
const STATE: usize = 16;
const EXPERTS: usize = 4;
const FF: usize = 64;
const SHEXP: usize = 128;
const VOCAB: usize = 64;
const BLOCKS: [&str; 4] = ["mamba", "moe", "attention", "mamba"];

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

/// Small BF16 values (|x| < 2), either sign.
fn bf16s(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(2 * n, seed)
        .chunks(2)
        .flat_map(|c| [c[0], 0x3c | (c[1] & 0x80) | (c[1] & 0x03)])
        .collect()
}

fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 0x3f80u16.to_le_bytes()).collect()
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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

/// Adds one tensor to the checkpoint being written.
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

/// Writes the checkpoint; returns its tensors by name.
fn checkpoint(dir: &Path) -> std::collections::HashMap<String, T> {
    let mut ts: Vec<T> = Vec::new();
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
    let mut quantized = serde_json::Map::new();
    let mut nvfp4 = |add: &mut Add<'_>,
                     next: &mut dyn FnMut() -> u64,
                     module: String,
                     rows: usize,
                     k: usize,
                     scale2: f32| {
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
        quantized.insert(
            module,
            json!({"quant_algo": "W4A16_NVFP4", "group_size": 16}),
        );
    };
    let d_inner = SSM_HEADS * SSM_HEAD_DIM;
    let conv_dim = d_inner + 2 * GROUPS * STATE;
    add(
        "backbone.embeddings.weight".into(),
        "BF16",
        &[VOCAB, EMBD],
        bf16s(VOCAB * EMBD, next()),
    );
    add(
        "backbone.norm_f.weight".into(),
        "BF16",
        &[EMBD],
        bf16_ones(EMBD),
    );
    nvfp4(&mut add, &mut next, "lm_head".into(), VOCAB, EMBD, 0.75);
    let mut fp8 = Vec::new();
    for (il, kind) in BLOCKS.iter().enumerate() {
        let l = format!("backbone.layers.{il}.");
        add(format!("{l}norm.weight"), "BF16", &[EMBD], bf16_ones(EMBD));
        match *kind {
            "mamba" => {
                for (proj, rows, k) in [
                    (
                        "in_proj",
                        2 * d_inner + 2 * GROUPS * STATE + SSM_HEADS,
                        EMBD,
                    ),
                    ("out_proj", EMBD, d_inner),
                ] {
                    let m = format!("{l}mixer.{proj}");
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
                        &[1],
                        1f32.to_le_bytes().to_vec(),
                    );
                    fp8.push(m);
                }
                add(
                    format!("{l}mixer.conv1d.weight"),
                    "BF16",
                    &[conv_dim, 1, 4],
                    bf16s(conv_dim * 4, next()),
                );
                add(
                    format!("{l}mixer.conv1d.bias"),
                    "BF16",
                    &[conv_dim],
                    bf16s(conv_dim, next()),
                );
                add(
                    format!("{l}mixer.A_log"),
                    "BF16",
                    &[SSM_HEADS],
                    bf16s(SSM_HEADS, next()),
                );
                add(
                    format!("{l}mixer.D"),
                    "BF16",
                    &[SSM_HEADS],
                    bf16s(SSM_HEADS, next()),
                );
                add(
                    format!("{l}mixer.dt_bias"),
                    "BF16",
                    &[SSM_HEADS],
                    bf16s(SSM_HEADS, next()),
                );
                add(
                    format!("{l}mixer.norm.weight"),
                    "BF16",
                    &[d_inner],
                    bf16_ones(d_inner),
                );
            }
            "attention" => {
                for (proj, rows, k) in [
                    ("q_proj", HEADS * HEAD_DIM, EMBD),
                    ("k_proj", KV * HEAD_DIM, EMBD),
                    ("v_proj", KV * HEAD_DIM, EMBD),
                    ("o_proj", EMBD, HEADS * HEAD_DIM),
                ] {
                    add(
                        format!("{l}mixer.{proj}.weight"),
                        "BF16",
                        &[rows, k],
                        bf16s(rows * k, next()),
                    );
                }
            }
            _ => {
                let gate: Vec<f32> = rand_bytes(EXPERTS * EMBD, next())
                    .into_iter()
                    .map(|b| (b as f32 - 128.0) / 512.0)
                    .collect();
                add(
                    format!("{l}mixer.gate.weight"),
                    "F32",
                    &[EXPERTS, EMBD],
                    f32s(&gate),
                );
                add(
                    format!("{l}mixer.gate.e_score_correction_bias"),
                    "F32",
                    &[EXPERTS],
                    f32s(&[0.01, -0.02, 0.0, 0.03]),
                );
                for e in 0..EXPERTS {
                    let x = format!("{l}mixer.experts.{e}.");
                    let s2 = 0.5 + 0.125 * e as f32;
                    nvfp4(&mut add, &mut next, format!("{x}up_proj"), FF, EMBD, s2);
                    nvfp4(&mut add, &mut next, format!("{x}down_proj"), EMBD, FF, s2);
                }
                let x = format!("{l}mixer.shared_experts.");
                nvfp4(
                    &mut add,
                    &mut next,
                    format!("{x}up_proj"),
                    SHEXP,
                    EMBD,
                    0.25,
                );
                nvfp4(
                    &mut add,
                    &mut next,
                    format!("{x}down_proj"),
                    EMBD,
                    SHEXP,
                    0.25,
                );
            }
        }
    }
    for m in fp8 {
        quantized.insert(m, json!({"quant_algo": "FP8"}));
    }
    write_safetensors(&dir.join("model.safetensors"), &ts);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "nemotron_h",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": quantized},
            "layers_block_type": BLOCKS,
            "hidden_size": EMBD, "num_attention_heads": HEADS, "num_key_value_heads": KV,
            "head_dim": HEAD_DIM, "mamba_num_heads": SSM_HEADS, "mamba_head_dim": SSM_HEAD_DIM,
            "n_groups": GROUPS, "ssm_state_size": STATE, "conv_kernel": 4,
            "num_experts_per_tok": 2, "n_routed_experts": EXPERTS, "moe_intermediate_size": FF,
            "moe_shared_expert_intermediate_size": SHEXP, "n_shared_experts": 1,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": true, "routed_scaling_factor": 2.5,
            "intermediate_size": FF, "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 4096, "rope_theta": 10000, "vocab_size": VOCAB,
            "num_nextn_predict_layers": 0
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
fn modelopt_nemotron_opens_as_the_converter_would_write_it() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    assert_eq!(g.architecture().as_deref(), Some("nemotron_h_moe"));
    let arr = |k: &str| -> Vec<u64> {
        g.get(&format!("nemotron_h_moe.{k}"))
            .and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            })
            .unwrap()
    };
    assert_eq!(arr("attention.head_count_kv"), vec![0, 0, KV as u64, 0]);
    assert_eq!(arr("feed_forward_length"), vec![0, FF as u64, 0, 0]);
    assert_eq!(
        g.get("tokenizer.ggml.pre").unwrap().as_str(),
        Some("pixtral")
    );

    // FP8 kept raw, NVFP4 repacked, the rest as the converter writes it.
    let ty = |n: &str| g.tensor(n).unwrap_or_else(|| panic!("no {n}")).ggml_type;
    for n in ["blk.0.ssm_in.weight", "blk.3.ssm_out.weight"] {
        assert_eq!(ty(n), types::I8, "{n}");
    }
    let Some(Value::Array(listed)) = g.get("legion.tensor_extra.fp8_e4m3") else {
        panic!("no FP8 list")
    };
    assert_eq!(listed.len(), 4);
    for n in [
        "output.weight",
        "blk.1.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight",
        "blk.1.ffn_up_shexp.weight",
    ] {
        assert_eq!(ty(n), types::NVFP4, "{n}");
    }
    assert_eq!(ty("blk.2.attn_q.weight"), types::BF16);
    let conv = g.tensor("blk.0.ssm_conv1d.weight").unwrap();
    assert_eq!(
        (conv.ggml_type, conv.shape.clone()),
        (types::F32, vec![4, 128])
    );
    let scale = g.tensor("blk.1.ffn_up_exps.scale").unwrap();
    assert_eq!(scale.shape, vec![EXPERTS as u64]);

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    // Stacked experts: each one repacked, back to back; their global scales as `.scale`.
    let mut want = Vec::new();
    for e in 0..EXPERTS {
        let b = format!("backbone.layers.1.mixer.experts.{e}.up_proj");
        want.extend(repack_nvfp4(
            &src[&format!("{b}.weight")].bytes,
            &src[&format!("{b}.weight_scale")].bytes,
            FF,
            EMBD,
        ));
    }
    assert_eq!(w.get("blk.1.ffn_up_exps.weight").unwrap().bytes(), want);
    let s: Vec<f32> = w
        .get("blk.1.ffn_up_exps.scale")
        .unwrap()
        .bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(s, vec![0.5, 0.625, 0.75, 0.875]);
    // FP8 bytes as shipped, Mamba-2 transforms.
    assert_eq!(
        w.get("blk.0.ssm_in.weight").unwrap().bytes(),
        src["backbone.layers.0.mixer.in_proj.weight"].bytes
    );
    let a_log = &src["backbone.layers.0.mixer.A_log"].bytes;
    let a: Vec<f32> = w
        .get("blk.0.ssm_a")
        .unwrap()
        .bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, v) in a.iter().enumerate() {
        let x = f32::from_bits((u16::from_le_bytes([a_log[2 * i], a_log[2 * i + 1]]) as u32) << 16);
        assert_eq!(*v, -((x as f64).exp() as f32));
    }
}

/// The checkpoint runs with its FP8 weights as shipped (the FP8 matmul) and agrees with the
/// converter's BF16 dequantization up to that rounding.
#[test]
fn raw_fp8_runs_close_to_bf16_dequantized() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let ids: Vec<u32> = vec![0, 1, 2, 0, 0, 1, 2, 2, 1, 0];
    let mut runs = Vec::new();
    // W8A16 against the dequantized weights (same activations); W8A8 (the checkpoint's static
    // input scales, the default) only has to run.
    for (fp8, act) in [
        (Fp8Mode::Raw, Fp8Act::A16),
        (Fp8Mode::Bf16, Fp8Act::A16),
        (Fp8Mode::Raw, Fp8Act::A8),
    ] {
        let g = GgufFile::open_with(dir.path(), &OpenOptions { fp8 }).unwrap();
        let mut model = Model::load(&g, |_| backend.buffer_type()).unwrap();
        model.set_fp8_act(act);
        let nm = legion::model::nemotron_h::params(&model.hp).unwrap();
        assert_eq!(
            nm.blocks,
            vec![Block::Mamba, Block::Ffn, Block::Attention, Block::Mamba]
        );
        runs.push(model.prompt_logits_all(&mut backend, &ids).unwrap());
    }
    let (raw, bf16, a8) = (&runs[0], &runs[1], &runs[2]);
    assert!(raw.iter().all(|x| x.is_finite()));
    assert!(a8.iter().all(|x| x.is_finite()));
    assert_ne!(a8, raw, "a8 did not quantize the activations");
    let scale = bf16.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = raw
        .iter()
        .zip(bf16)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff > 0.0,
        "the two paths cannot be identical (BF16 rounding)"
    );
    assert!(diff < 0.05 * scale, "max |diff| {diff} of {scale}");
}
