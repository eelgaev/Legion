//! A tiny ModelOpt NVFP4 Qwen3.8-Flash-Next checkpoint (`qwen4_exp`: NVFP4 routed experts,
//! everything else BF16, the n-gram table as FP8 shards in a file of its own, stored out of
//! index order) opened as a synthetic `qwen4exp` GGUF (`src/hf/qwen4.rs`): the metadata, the
//! converter's tensor transforms, the experts stacked with their scales, the n-gram table left
//! in its file and read through the n-gram store; and the model runs the same with the shards
//! in any file order.
//!
//! The real checkpoint against a GGUF of the same model (`real_checkpoint_matches_the_gguf`,
//! ignored; run it on a machine that has both, CPU only):
//!
//! ```text
//! LEGION_TEST_QWEN4_HF=<snapshot dir> LEGION_TEST_QWEN4_GGUF=<first .gguf shard> \
//!   cargo test --release --test hf_qwen4 real_checkpoint_matches -- --ignored --nocapture
//! ```
//!
//! `real_checkpoint_logits_agree_with_the_gguf` (ignored, same variables, a `cuda` build) runs
//! both models on the GPU, one after the other, and compares their prompt logits.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use legion::ggml::{self, Backend, types};
use legion::gguf::{GgufFile, Recipe, TensorInfo, Value};
use legion::hf::nvfp4::{e4m3_to_f32, f32_to_bf16, repack_nvfp4};
use legion::model::core::{Hparams, Model};
use legion::model::qwen4::{KEY_PLE_BLOCK_ORDER, KEY_PLE_BLOCK_ROWS, KEY_PLE_FP8_SCALE};
use legion::residency::lookup_table::{self, LookupTable, TableIo, TableOptions};
use legion::weights::Weights;
use serde_json::json;

const EMBD: usize = 64;
const HC: usize = 4;
const HC_RANK: usize = 32;
const HEADS: usize = 4;
const KV: usize = 2;
const HEAD_DIM: usize = 32;
const NK: usize = 2;
const NV: usize = 4;
const DK: usize = 16;
const EXPERTS: usize = 8;
const FF: usize = 64;
const SHEXP: usize = 64;
const IDX_HEADS: usize = 2;
const IDX_DIM: usize = 16;
const VOCAB: usize = 64;
const EOS: u64 = 3;
const LAYERS: [&str; 4] = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
];
const PLE_DIM: usize = 32;
const PLE_SIZES: [u64; 4] = [1009, 1011, 1013, 1015];
const MULTIPLIERS: [i64; 3] = [23703573157769, 20109073645365, 8052911324071];
/// Shards of the n-gram table and their rows (4 x 1012 = the heads' 4048 rows).
const PARTS: usize = 4;
const PART_ROWS: usize = 1012;
const PLE_SCALE: f32 = 0.0625;
/// The shards' order in their file (index order is `0..PARTS`).
const SHUFFLED: [usize; PARTS] = [2, 0, 3, 1];
const PROMPT: &[u32] = &[5, 9, 17, 5, 9, 33, 3, 5, 9, 17, 40, 41, 5, 9, 17, 22, 8];

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

/// `n` BF16 values `bias + scale * U(-1, 1)`.
fn bf16s(n: usize, seed: u64, scale: f32, bias: f32) -> Vec<u8> {
    rand_bytes(2 * n, seed)
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0 - 1.0)
        .flat_map(|u| f32_to_bf16(bias + scale * u).to_le_bytes())
        .collect()
}

fn bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Small E4M3 scales (exponent 1..=4, either sign: the converter clears the sign bit).
fn e4m3_scales(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (b & 0x80) | (1 + (b >> 3) % 4) << 3 | (b & 7))
        .collect()
}

/// Finite E4M3 values around 1.
fn e4m3_values(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| (b & 0x80) | (4 + (b >> 3) % 6) << 3 | (b & 7))
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

/// The tensors of the checkpoint (the n-gram shards in index order).
struct Ckpt {
    main: Vec<T>,
    ple: Vec<T>,
    /// The MTP head's block-scaled FP8 experts, in the n-gram table's file as shipped.
    mtp_fp8: Vec<T>,
    quantized: serde_json::Map<String, serde_json::Value>,
}

/// The MTP experts' scale blocks (128 x 128 in the real checkpoint).
const MTP_BLOCK: usize = 32;

/// The checkpoint's tensors; the MTP head's routed experts block-scaled FP8 (`fp8_mtp`), else
/// the same values dequantized, F32.
fn tensors(fp8_mtp: bool) -> Ckpt {
    let mut main: Vec<T> = Vec::new();
    let mut seed = 0;
    let mut next = || {
        seed += 1;
        seed
    };
    let mut add = |name: String, dtype: &'static str, shape: &[usize], bytes: Vec<u8>| {
        main.push(T {
            name,
            dtype,
            shape: shape.to_vec(),
            bytes,
        })
    };
    // A BF16 weight [rows, k] scaled for unit-variance outputs; a norm near 0 (the converter
    // adds 1).
    let weight =
        |rows: usize, k: usize, seed: u64| bf16s(rows * k, seed, (3.0 / k as f32).sqrt(), 0.0);
    let norm = |n: usize, seed: u64| bf16s(n, seed, 0.1, 0.0);
    let p = "model.language_model.";
    let hc_dim = HC * EMBD;
    let mixer = |add: &mut Add<'_>, next: &mut dyn FnMut() -> u64, m: String, inject: bool| {
        add(
            format!("{m}.hc_norm.weight"),
            "BF16",
            &[hc_dim],
            norm(hc_dim, next()),
        );
        add(
            format!("{m}.input_mix_weight_down.weight"),
            "BF16",
            &[HC_RANK, hc_dim],
            weight(HC_RANK, hc_dim, next()),
        );
        add(
            format!("{m}.input_mix_weight_up.weight"),
            "BF16",
            &[hc_dim, HC_RANK],
            weight(hc_dim, HC_RANK, next()),
        );
        if inject {
            add(
                format!("{m}.block_inject_weight.weight"),
                "BF16",
                &[HC, hc_dim],
                bf16s(HC * hc_dim, next(), 0.05, 0.0),
            );
        }
    };
    // A full-attention block's projections and norms (`a`: `...self_attn.`).
    let attention = |add: &mut Add<'_>, next: &mut dyn FnMut() -> u64, a: String| {
        for (n, rows, k) in [
            ("q_proj", 2 * HEADS * HEAD_DIM, EMBD),
            ("k_proj", KV * HEAD_DIM, EMBD),
            ("v_proj", KV * HEAD_DIM, EMBD),
            ("o_proj", EMBD, HEADS * HEAD_DIM),
            ("indexer.index_qk_proj", (IDX_HEADS + 1) * IDX_DIM, EMBD),
        ] {
            add(
                format!("{a}{n}.weight"),
                "BF16",
                &[rows, k],
                weight(rows, k, next()),
            );
        }
        for (n, d) in [
            ("q_norm", HEAD_DIM),
            ("k_norm", HEAD_DIM),
            ("indexer.q_layernorm", IDX_DIM),
            ("indexer.k_layernorm", IDX_DIM),
        ] {
            add(format!("{a}{n}.weight"), "BF16", &[d], norm(d, next()));
        }
    };
    add(
        format!("{p}embed_tokens.weight"),
        "BF16",
        &[VOCAB, EMBD],
        bf16s(VOCAB * EMBD, next(), 1.0, 0.0),
    );
    add(
        "lm_head.weight".into(),
        "BF16",
        &[VOCAB, EMBD],
        weight(VOCAB, EMBD, next()),
    );
    mixer(
        &mut add,
        &mut next,
        format!("{p}hyper_connection_mixer"),
        false,
    );
    let mut quantized = serde_json::Map::new();
    let (qk, v) = (2 * NK * DK, NV * DK);
    for (il, kind) in LAYERS.iter().enumerate() {
        let l = format!("{p}layers.{il}.");
        mixer(
            &mut add,
            &mut next,
            format!("{l}attn_hyper_connection"),
            true,
        );
        mixer(
            &mut add,
            &mut next,
            format!("{l}mlp_hyper_connection"),
            true,
        );
        if *kind == "linear_attention" {
            let a = format!("{l}linear_attn.");
            for (n, rows, k) in [
                ("in_proj_qkv", qk + v, EMBD),
                ("in_proj_z", v, EMBD),
                ("in_proj_a", NV, EMBD),
                ("in_proj_b", NV, EMBD),
                ("out_proj", EMBD, v),
            ] {
                add(
                    format!("{a}{n}.weight"),
                    "BF16",
                    &[rows, k],
                    weight(rows, k, next()),
                );
            }
            add(
                format!("{a}conv1d.weight"),
                "BF16",
                &[qk + v, 1, 4],
                bf16s((qk + v) * 4, next(), 0.5, 0.0),
            );
            add(
                format!("{a}A_log"),
                "BF16",
                &[NV],
                bf16s(NV, next(), 0.5, 0.0),
            );
            add(
                format!("{a}dt_bias"),
                "BF16",
                &[NV],
                bf16s(NV, next(), 0.5, 0.0),
            );
            add(
                format!("{a}norm.weight"),
                "BF16",
                &[DK],
                bf16s(DK, next(), 0.1, 1.0),
            );
        } else {
            attention(&mut add, &mut next, format!("{l}self_attn."));
        }
        if il == 1 {
            let pe = PLE_DIM * PLE_SIZES.len();
            let a = format!("{l}ple.");
            add(
                format!("{a}key_proj.weight"),
                "BF16",
                &[hc_dim, pe],
                weight(hc_dim, pe, next()),
            );
            add(
                format!("{a}value_proj.weight"),
                "BF16",
                &[EMBD, pe],
                weight(EMBD, pe, next()),
            );
            for n in ["norm_key", "norm_query", "norm_conv"] {
                add(
                    format!("{a}{n}.weight"),
                    "BF16",
                    &[hc_dim],
                    norm(hc_dim, next()),
                );
            }
            add(
                format!("{a}conv1d.weight"),
                "BF16",
                &[hc_dim, 1, 4],
                bf16s(hc_dim * 4, next(), 0.5, 0.0),
            );
            let ints = |v: &[i64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let offsets: Vec<i64> = PLE_SIZES
                .iter()
                .scan(0i64, |acc, &s| {
                    let o = *acc;
                    *acc += s as i64;
                    Some(o)
                })
                .collect();
            let sizes: Vec<i64> = PLE_SIZES.iter().map(|&s| s as i64).collect();
            let e = format!("{a}ple_embedding.");
            add(
                format!("{e}layer_multipliers"),
                "I64",
                &[3],
                ints(&MULTIPLIERS),
            );
            add(
                format!("{e}ngram_heads_offsets"),
                "I64",
                &[PLE_SIZES.len()],
                ints(&offsets),
            );
            add(
                format!("{e}ngram_heads_vocab_sizes"),
                "I64",
                &[PLE_SIZES.len()],
                ints(&sizes),
            );
        }
        let m = format!("{l}mlp.");
        add(
            format!("{m}gate.weight"),
            "BF16",
            &[EXPERTS, EMBD],
            bf16s(EXPERTS * EMBD, next(), 0.25, 0.0),
        );
        for e in 0..EXPERTS {
            for (n, rows, k) in [
                ("gate_proj", FF, EMBD),
                ("up_proj", FF, EMBD),
                ("down_proj", EMBD, FF),
            ] {
                let x = format!("{m}experts.{e}.{n}");
                add(
                    format!("{x}.weight"),
                    "U8",
                    &[rows, k / 2],
                    rand_bytes(rows * k / 2, next()),
                );
                add(
                    format!("{x}.weight_scale"),
                    "F8_E4M3",
                    &[rows, k / 16],
                    e4m3_scales(rows * k / 16, next()),
                );
                let s2 = 0.5 + 0.0625 * e as f32;
                add(
                    format!("{x}.weight_scale_2"),
                    "F32",
                    &[],
                    s2.to_le_bytes().to_vec(),
                );
                add(
                    format!("{x}.input_scale"),
                    "F32",
                    &[],
                    0.25f32.to_le_bytes().to_vec(),
                );
            }
        }
        quantized.insert(
            format!("{m}experts"),
            json!({"quant_algo": "NVFP4", "group_size": 16}),
        );
        for (n, rows, k) in [
            ("gate_proj", SHEXP, EMBD),
            ("up_proj", SHEXP, EMBD),
            ("down_proj", EMBD, SHEXP),
        ] {
            add(
                format!("{m}shared_expert.{n}.weight"),
                "BF16",
                &[rows, k],
                weight(rows, k, next()),
            );
        }
        add(
            format!("{m}shared_expert_gate.weight"),
            "BF16",
            &[1, EMBD],
            weight(1, EMBD, next()),
        );
    }

    // The MTP head: BF16 in the main file, its routed experts E4M3 with one power-of-two
    // `weight_scale_inv` per 32 x 32 block in the FP8 file (or F32 values, dequantized).
    let mut mtp_fp8 = Vec::new();
    let l = "mtp.layers.0.";
    mixer(
        &mut add,
        &mut next,
        format!("{l}attn_hyper_connection"),
        true,
    );
    mixer(
        &mut add,
        &mut next,
        format!("{l}mlp_hyper_connection"),
        true,
    );
    attention(&mut add, &mut next, format!("{l}self_attn."));
    let m = format!("{l}mlp.");
    add(
        format!("{m}gate.weight"),
        "BF16",
        &[EXPERTS, EMBD],
        bf16s(EXPERTS * EMBD, next(), 0.25, 0.0),
    );
    for e in 0..EXPERTS {
        for (n, rows, k) in [
            ("gate_proj", FF, EMBD),
            ("up_proj", FF, EMBD),
            ("down_proj", EMBD, FF),
        ] {
            let x = format!("{m}experts.{e}.{n}");
            let q = e4m3_values(rows * k, next());
            let (mb, kb) = (rows / MTP_BLOCK, k / MTP_BLOCK);
            let s: Vec<f32> = rand_bytes(mb * kb, next())
                .iter()
                .map(|b| 2f32.powi(-3 - (b % 3) as i32))
                .collect();
            if fp8_mtp {
                mtp_fp8.push(T {
                    name: format!("{x}.weight"),
                    dtype: "F8_E4M3",
                    shape: vec![rows, k],
                    bytes: q,
                });
                mtp_fp8.push(T {
                    name: format!("{x}.weight_scale_inv"),
                    dtype: "BF16",
                    shape: vec![mb, kb],
                    bytes: s
                        .iter()
                        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
                        .collect(),
                });
            } else {
                let w: Vec<u8> = q
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &b)| {
                        let (r, c) = (i / k, i % k);
                        let v = e4m3_to_f32(b) * s[(r / MTP_BLOCK) * kb + c / MTP_BLOCK];
                        v.to_le_bytes()
                    })
                    .collect();
                add(format!("{x}.weight"), "F32", &[rows, k], w);
            }
        }
    }
    if fp8_mtp {
        quantized.insert(
            format!("{m}experts"),
            json!({"quant_algo": "FP8_PB_WO", "group_size": 128}),
        );
    }
    for (n, rows, k) in [
        ("gate_proj", SHEXP, EMBD),
        ("up_proj", SHEXP, EMBD),
        ("down_proj", EMBD, SHEXP),
    ] {
        add(
            format!("{m}shared_expert.{n}.weight"),
            "BF16",
            &[rows, k],
            weight(rows, k, next()),
        );
    }
    add(
        format!("{m}shared_expert_gate.weight"),
        "BF16",
        &[1, EMBD],
        weight(1, EMBD, next()),
    );
    for n in ["fc_embedding", "fc_hidden"] {
        add(
            format!("mtp.{n}.weight"),
            "BF16",
            &[EMBD, EMBD],
            weight(EMBD, EMBD, next()),
        );
    }
    add(
        "mtp.pre_fc_norm_embedding.weight".into(),
        "BF16",
        &[EMBD],
        norm(EMBD, next()),
    );
    add(
        "mtp.pre_fc_norm_hidden.weight".into(),
        "BF16",
        &[hc_dim],
        norm(hc_dim, next()),
    );
    mixer(
        &mut add,
        &mut next,
        "mtp.hyper_connection_mixer".into(),
        false,
    );

    let e = format!("{p}layers.1.ple.ple_embedding.ngram_embedding.");
    let mut ple = vec![T {
        name: format!("{e}weight_scale"),
        dtype: "BF16",
        shape: vec![1],
        bytes: f32_to_bf16(PLE_SCALE).to_le_bytes().to_vec(),
    }];
    for i in 0..PARTS {
        ple.push(T {
            name: format!("{e}shard_{i}.weight"),
            dtype: "F8_E4M3",
            shape: vec![PART_ROWS, PLE_DIM],
            bytes: e4m3_values(PART_ROWS * PLE_DIM, 1000 + i as u64),
        });
    }
    quantized.insert(
        format!("{p}layers.1.ple.ple_embedding.ngram_embedding"),
        json!({"quant_algo": "FP8"}),
    );
    Ckpt {
        main,
        ple,
        mtp_fp8,
        quantized,
    }
}

/// Writes the checkpoint with the n-gram shards in file order `order`; returns its tensors.
fn checkpoint(dir: &Path, order: [usize; PARTS]) -> Ckpt {
    checkpoint_with(dir, order, true)
}

/// [`checkpoint`] with the MTP head's routed experts FP8 (`fp8_mtp`) or F32.
fn checkpoint_with(dir: &Path, order: [usize; PARTS], fp8_mtp: bool) -> Ckpt {
    let mut c = tensors(fp8_mtp);
    let mut ple: Vec<T> = Vec::new();
    let mut shards: Vec<Option<T>> = c.ple.drain(1..).map(Some).collect();
    ple.push(c.ple.pop().unwrap());
    for i in order {
        ple.push(shards[i].take().unwrap());
    }
    // The FP8 file holds the n-gram table and the MTP experts, as shipped.
    let n_ple = ple.len();
    ple.append(&mut c.mtp_fp8);
    write_safetensors(&dir.join("model-00001-of-00002.safetensors"), &c.main);
    write_safetensors(&dir.join("model-fp8-mtp-ple.safetensors"), &ple);
    let mut map = serde_json::Map::new();
    for t in &c.main {
        map.insert(t.name.clone(), json!("model-00001-of-00002.safetensors"));
    }
    for t in &ple {
        map.insert(t.name.clone(), json!("model-fp8-mtp-ple.safetensors"));
    }
    c.mtp_fp8 = ple.split_off(n_ple);
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": map})).unwrap(),
    )
    .unwrap();
    let text = json!({
        "model_type": "qwen4_exp_text",
        "hidden_size": EMBD, "num_hidden_layers": LAYERS.len(), "layer_types": LAYERS,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HEAD_DIM,
        "linear_num_key_heads": NK, "linear_num_value_heads": NV,
        "linear_key_head_dim": DK, "linear_value_head_dim": DK, "linear_conv_kernel_dim": 4,
        "num_experts": EXPERTS, "num_experts_per_tok": 2, "moe_intermediate_size": FF,
        "shared_expert_intermediate_size": SHEXP, "full_attention_interval": 4,
        "hc_count": HC, "hc_lowrank": HC_RANK,
        "indexer_n_heads": IDX_HEADS, "indexer_head_dim": IDX_DIM, "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "ngram_size": 3, "heads_per_ngram": 2, "ple_conv_kernel_size": 4, "ple_layer_ids": [2],
        "split_ngram_parts": PARTS,
        "rope_parameters": {"mrope_section": [3, 3, 2], "partial_rotary_factor": 0.5,
                            "rope_theta": 10000000.0, "rope_type": "default"},
        "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "vocab_size": VOCAB,
        "eos_token_id": EOS, "tie_word_embeddings": false, "mtp_num_hidden_layers": 1
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "image_token_id": 60,
            "text_config": text,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "quantized_layers": c.quantized.clone()}
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
    c.ple = ple;
    c
}

fn by_name(ts: &[T]) -> HashMap<&str, &T> {
    ts.iter().map(|t| (t.name.as_str(), t)).collect()
}

fn ints(g: &GgufFile, key: &str) -> Vec<u64> {
    g.get(key)
        .unwrap_or_else(|| panic!("no {key}"))
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}

#[test]
fn nvfp4_qwen4_opens_as_the_converter_would_write_it() {
    let dir = tempfile::tempdir().unwrap();
    let c = checkpoint(dir.path(), SHUFFLED);
    let src = by_name(&c.main);
    let g = GgufFile::open(dir.path()).unwrap();
    assert_eq!(g.architecture().as_deref(), Some("qwen4exp"));
    let k = |s: &str| format!("qwen4exp.{s}");
    assert_eq!(ints(&g, &k("attention.compress_ratios")), vec![0, 0, 0, 4]);
    assert_eq!(ints(&g, &k("rope.dimension_sections")), vec![3, 3, 2, 0]);
    assert_eq!(g.get_u64(&k("rope.dimension_count")), Some(16));
    assert_eq!(ints(&g, &k("ple.layers")), vec![1]);
    assert_eq!(g.get_u64(&k("ple.eos_token_id")), Some(EOS));
    assert_eq!(g.get_u64(&k("ple.image_token_id")), Some(60));
    assert_eq!(
        g.get_u64(&k("embedding_length_per_layer_input")),
        Some(PLE_DIM as u64)
    );
    assert_eq!(
        ints(&g, &k("ple.layer_multipliers")),
        MULTIPLIERS.map(|m| m as u64)
    );
    assert_eq!(ints(&g, &k("ple.head_vocab_sizes")), PLE_SIZES);
    assert_eq!(ints(&g, &k("ple.head_offsets")), vec![0, 1009, 2020, 3033]);
    assert_eq!(g.get_u64(KEY_PLE_BLOCK_ROWS), Some(PART_ROWS as u64));
    // Block i (index order) sits at the position of i in SHUFFLED.
    assert_eq!(ints(&g, KEY_PLE_BLOCK_ORDER), vec![1, 3, 0, 2]);
    assert_eq!(
        g.get(KEY_PLE_FP8_SCALE).and_then(|v| v.as_f64()),
        Some(PLE_SCALE as f64)
    );

    let info = |n: &str| g.tensor(n).unwrap_or_else(|| panic!("no {n}"));
    for n in [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.3.ffn_down_exps.weight",
    ] {
        assert_eq!(info(n).ggml_type, types::NVFP4, "{n}");
    }
    assert_eq!(
        info("blk.0.ffn_gate_exps.weight").shape,
        vec![EMBD as u64, FF as u64, EXPERTS as u64]
    );
    assert_eq!(info("blk.0.ffn_up_exps.scale").shape, vec![EXPERTS as u64]);
    for (n, ty) in [
        ("blk.0.hc_attn_norm.weight", types::F32),
        ("blk.0.hc_attn_inject.weight", types::F32),
        ("blk.0.hc_ffn_down.weight", types::BF16),
        ("output_hc_norm.weight", types::F32),
        ("blk.0.ssm_conv1d.weight", types::F32),
        ("blk.0.ssm_norm.weight", types::F32),
        ("blk.0.ffn_gate_inp.weight", types::F32),
        ("blk.0.ffn_gate_inp_shexp.weight", types::F32),
        ("blk.3.attn_q.weight", types::BF16),
        ("blk.3.indexer.q_proj.weight", types::BF16),
        ("blk.1.ple_conv1d.weight", types::BF16),
        ("blk.1.ple_norm_key.weight", types::F32),
    ] {
        assert_eq!(info(n).ggml_type, ty, "{n}");
    }
    assert_eq!(
        info("blk.1.ple_conv1d.weight").shape,
        vec![4, (HC * EMBD) as u64]
    );
    assert_eq!(
        info("blk.3.indexer.q_proj.weight").shape,
        vec![EMBD as u64, (IDX_HEADS * IDX_DIM) as u64]
    );
    assert!(g.tensor("output_norm.weight").is_none());
    assert!(g.tensor("blk.4.nextn.eh_proj.weight").is_none());

    // The n-gram table is the shards' file range as shipped.
    let t = info("per_layer_token_embd.weight");
    assert_eq!(t.ggml_type, types::I8);
    assert_eq!(t.shape, vec![PLE_DIM as u64, (PARTS * PART_ROWS) as u64]);
    assert!(g.recipe("per_layer_token_embd.weight").is_none());

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let bytes = |n: &str| w.get(n).unwrap_or_else(|| panic!("no {n}")).bytes();
    // `c.ple`: the scale, then the shards in file order.
    let table: Vec<u8> = c.ple[1..].iter().flat_map(|t| t.bytes.clone()).collect();
    assert_eq!(bytes("per_layer_token_embd.weight"), table);
    // Norms + 1 (F32), the DeltaNet output norm as is.
    let p = "model.language_model.";
    let src_f32 = |n: &str| bf16_f32(&src[n].bytes);
    let plus1 = |n: &str| src_f32(n).iter().map(|v| v + 1.0).collect::<Vec<f32>>();
    assert_eq!(
        f32s(&bytes("blk.0.hc_attn_norm.weight")),
        plus1(&format!("{p}layers.0.attn_hyper_connection.hc_norm.weight"))
    );
    // Inject weights widened to F32, exactly.
    assert_eq!(
        f32s(&bytes("blk.2.hc_ffn_inject.weight")),
        src_f32(&format!(
            "{p}layers.2.mlp_hyper_connection.block_inject_weight.weight"
        ))
    );
    assert_eq!(
        f32s(&bytes("output_hc_norm.weight")),
        plus1(&format!("{p}hyper_connection_mixer.hc_norm.weight"))
    );
    assert_eq!(
        f32s(&bytes("blk.3.indexer.k_norm.weight")),
        plus1(&format!("{p}layers.3.self_attn.indexer.k_layernorm.weight"))
    );
    assert_eq!(
        f32s(&bytes("blk.1.ple_norm_query.weight")),
        plus1(&format!("{p}layers.1.ple.norm_query.weight"))
    );
    assert_eq!(
        f32s(&bytes("blk.0.ssm_norm.weight")),
        src_f32(&format!("{p}layers.0.linear_attn.norm.weight"))
    );
    // The indexer's projection split into its query heads and its key.
    let qk = &src[format!("{p}layers.3.self_attn.indexer.index_qk_proj.weight").as_str()].bytes;
    let split = IDX_HEADS * IDX_DIM * EMBD * 2;
    assert_eq!(bytes("blk.3.indexer.q_proj.weight"), qk[..split]);
    assert_eq!(bytes("blk.3.indexer.k_proj.weight"), qk[split..]);
    assert_eq!(
        bytes("blk.1.ple_conv1d.weight"),
        src[format!("{p}layers.1.ple.conv1d.weight").as_str()].bytes
    );
    // Experts: each one repacked, back to back; their global scales as `.scale`.
    let mut want = Vec::new();
    for e in 0..EXPERTS {
        let b = format!("{p}layers.2.mlp.experts.{e}.down_proj");
        want.extend(repack_nvfp4(
            &src[format!("{b}.weight").as_str()].bytes,
            &src[format!("{b}.weight_scale").as_str()].bytes,
            EMBD,
            FF,
        ));
    }
    assert_eq!(bytes("blk.2.ffn_down_exps.weight"), want);
    let want: Vec<f32> = (0..EXPERTS).map(|e| 0.5 + 0.0625 * e as f32).collect();
    assert_eq!(f32s(&bytes("blk.2.ffn_down_exps.scale")), want);

    // The model reads the table through the block order: logical row r of head h is the E4M3
    // codes of shard r / PART_ROWS times the scale.
    let hp = Hparams::from_gguf(&g).unwrap();
    let ple = legion::model::qwen4::params(&hp)
        .unwrap()
        .ple
        .clone()
        .unwrap();
    assert_eq!(ple.fp8_scale, Some(PLE_SCALE));
    let mut rows = Vec::new();
    ple.rows(17, &[5, 9], &mut rows);
    let plain = legion::model::qwen4::PleParams {
        blocks: None,
        ..ple.clone()
    };
    let mut logical = Vec::new();
    plain.rows(17, &[5, 9], &mut logical);
    let stored = bytes("per_layer_token_embd.weight");
    for (&r, &l) in rows.iter().zip(&logical) {
        let (shard, i) = (l as usize / PART_ROWS, l as usize % PART_ROWS);
        let want = &c.ple[1 + SHUFFLED.iter().position(|&j| j == shard).unwrap()].bytes
            [i * PLE_DIM..][..PLE_DIM];
        assert_eq!(&stored[r as usize * PLE_DIM..][..PLE_DIM], want);
        let vals = legion::model::qwen4::ple_row_f32(types::I8, ple.fp8_scale, want, PLE_DIM);
        assert_eq!(vals[0], e4m3_to_f32(want[0]) * PLE_SCALE);
    }
}

/// The model runs, the same whatever the shards' file order, and the same with the table read
/// through the n-gram store (from an offset that is not page- or block-aligned).
#[test]
fn nvfp4_qwen4_runs_the_same_with_any_shard_order() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let mut runs = Vec::new();
    let mut dirs = Vec::new();
    for order in [SHUFFLED, [0, 1, 2, 3]] {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(dir.path(), order);
        let g = GgufFile::open(dir.path()).unwrap();
        assert_eq!(
            g.get(KEY_PLE_BLOCK_ORDER).is_some(),
            order != [0, 1, 2, 3],
            "{order:?}"
        );
        let model = Model::load(&g, |_| backend.buffer_type()).unwrap();
        runs.push(model.prompt_logits_all(&mut backend, PROMPT).unwrap());
        dirs.push((dir, g));
    }
    assert!(runs[0].iter().all(|x| x.is_finite()));
    assert_eq!(runs[0], runs[1]);
    let (_, g) = &dirs[0];
    let file = Model::ngram_table_info(g).unwrap();
    assert_eq!(file.row_bytes, PLE_DIM);
    assert_ne!(file.offset % 512, 0);
    for io in [TableIo::Mmap, TableIo::Direct] {
        let mut model = Model::load(g, |_| backend.buffer_type()).unwrap();
        let opts = TableOptions {
            io,
            ram: Default::default(),
            ram_cache_bytes: Some(0),
            inflight: 4,
            aio_max: lookup_table::DEFAULT_AIO_MAX,
            keep_awake_ms: 0,
            node: None,
            unified: false,
        };
        model.set_ngram_table(LookupTable::open(file.clone(), &opts, None).unwrap());
        let got = model.prompt_logits_all(&mut backend, PROMPT).unwrap();
        assert_eq!(got, runs[0], "{io:?}");
    }
}

/// The checkpoint's `mtp.*` head opens as the head GGUF llama.cpp's converter writes for
/// `--spec-draft`: block `n_layer` plus `nextn.*`, its routed experts the shipped E4M3 bytes
/// with their scale grids widened to F32.
#[test]
fn mtp_head_opens_as_shipped() {
    let dir = tempfile::tempdir().unwrap();
    let c = checkpoint(dir.path(), SHUFFLED);
    let opts = legion::hf::OpenOptions {
        fp8: Default::default(),
    };
    let g = legion::hf::open_mtp(dir.path(), &opts).unwrap();
    let n = LAYERS.len();
    let k = |s: &str| format!("qwen4exp.{s}");
    assert_eq!(g.architecture().as_deref(), Some("qwen4exp"));
    assert_eq!(g.get_u64(&k("block_count")), Some(n as u64 + 1));
    assert_eq!(g.get_u64(&k("nextn_predict_layers")), Some(1));
    assert_eq!(ints(&g, &k("attention.compress_ratios")), vec![0; n + 1]);
    let info = |t: &str| g.tensor(t).unwrap_or_else(|| panic!("no {t}"));
    let blk = |t: &str| format!("blk.{n}.{t}");
    let gate = info(&blk("ffn_gate_exps.weight"));
    assert_eq!(gate.ggml_type, types::I8);
    assert_eq!(gate.shape, vec![EMBD as u64, FF as u64, EXPERTS as u64]);
    let grid = |rows: usize, k: usize| {
        vec![
            (k / MTP_BLOCK) as u64,
            (rows / MTP_BLOCK) as u64,
            EXPERTS as u64,
        ]
    };
    assert_eq!(info(&blk("ffn_gate_exps.scale")).shape, grid(FF, EMBD));
    assert_eq!(info(&blk("ffn_down_exps.scale")).shape, grid(EMBD, FF));
    assert_eq!(info(&blk("ffn_down_exps.scale")).ggml_type, types::F32);
    assert_eq!(
        info(&blk("nextn.eh_proj.weight")).shape,
        vec![2 * EMBD as u64, EMBD as u64]
    );
    for t in [
        "attn_q.weight",
        "indexer.q_proj.weight",
        "ffn_gate_inp.weight",
        "hc_attn_inject.weight",
        "nextn.enorm.weight",
        "nextn.hnorm.weight",
        "nextn.hc_head_down.weight",
    ] {
        assert!(g.tensor(&blk(t)).is_some(), "{t}");
    }
    // Only the head: no trunk layer, no embedding.
    assert!(g.tensor("blk.0.hc_attn_norm.weight").is_none());
    assert!(g.tensor("token_embd.weight").is_none());

    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let bytes = |t: &str| w.get(t).unwrap_or_else(|| panic!("no {t}")).bytes();
    let fp8 = by_name(&c.mtp_fp8);
    let main = by_name(&c.main);
    let x = |e: usize, p: &str| format!("mtp.layers.0.mlp.experts.{e}.{p}");
    let want: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| fp8[x(e, "up_proj.weight").as_str()].bytes.clone())
        .collect();
    assert_eq!(bytes(&blk("ffn_up_exps.weight")), want);
    let want: Vec<f32> = (0..EXPERTS)
        .flat_map(|e| bf16_f32(&fp8[x(e, "down_proj.weight_scale_inv").as_str()].bytes))
        .collect();
    assert_eq!(f32s(&bytes(&blk("ffn_down_exps.scale"))), want);
    // eh_proj row r: fc_embedding's row r, then fc_hidden's.
    let (fe, fh) = (
        &main["mtp.fc_embedding.weight"].bytes,
        &main["mtp.fc_hidden.weight"].bytes,
    );
    let row = EMBD * 2;
    let want: Vec<u8> = (0..EMBD)
        .flat_map(|r| [&fe[r * row..][..row], &fh[r * row..][..row]].concat())
        .collect();
    assert_eq!(bytes(&blk("nextn.eh_proj.weight")), want);
    let plus1 = |t: &str| {
        bf16_f32(&main[t].bytes)
            .iter()
            .map(|v| v + 1.0)
            .collect::<Vec<f32>>()
    };
    assert_eq!(
        f32s(&bytes(&blk("nextn.hnorm.weight"))),
        plus1("mtp.pre_fc_norm_hidden.weight")
    );
    assert_eq!(
        f32s(&bytes(&blk("nextn.hc_head_norm.weight"))),
        plus1("mtp.hyper_connection_mixer.hc_norm.weight")
    );
}

/// The checkpoint's head drafts through its block-scaled FP8 experts what the same head with
/// the experts dequantized to F32 drafts: same draft tokens, residuals equal to rounding.
#[test]
fn mtp_head_fp8_experts_match_their_dequantized_twin() {
    use legion::model::core::{Executor, Group, Logits, SeqInput};
    let opts = legion::hf::OpenOptions {
        fp8: Default::default(),
    };
    fn group<'a>(tokens: &'a [u32], logits: Logits) -> Group<'a> {
        Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past: 0,
                blocks: &[0, 1, 2, 3],
                ring: 0,
                logits,
                media: None,
            }],
            verify: false,
        }
    }
    let mut outs = Vec::new();
    for fp8 in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        checkpoint_with(dir.path(), SHUFFLED, fp8);
        let head = legion::hf::open_mtp(dir.path(), &opts).unwrap();
        let ty = head
            .tensor(&format!("blk.{}.ffn_gate_exps.weight", LAYERS.len()))
            .unwrap()
            .ggml_type;
        assert_eq!(ty, if fp8 { types::I8 } else { types::F32 });
        let mut b = Backend::cpu().unwrap();
        b.set_threads(4);
        let g = GgufFile::open(dir.path()).unwrap();
        let mut m = Model::load(&g, |_| b.buffer_type()).unwrap();
        m.load_mtp(&head, |_| b.buffer_type()).unwrap();
        let n_layer = m.hp.n_layer;
        let mut e = Executor::new(m, b, 4, 1, 0).unwrap();
        e.enable_mtp().unwrap();
        e.capture_hidden(vec![n_layer - 1]);
        let out = e.step(&[group(PROMPT, Logits::Last)]).unwrap();
        let w = e.mtp_width();
        let mut fed = PROMPT.to_vec();
        fed.push(out.argmax[0]);
        let mut rows = vec![0f32; w];
        rows.extend_from_slice(&out.hidden[0]);
        let mtp = e.mtp_step(&[group(&fed, Logits::All)], &rows).unwrap();
        assert!(mtp.hidden.iter().all(|v| v.is_finite()));
        outs.push(mtp);
    }
    assert_eq!(outs[0].argmax, outs[1].argmax);
    let err = rel_rmse(&outs[0].hidden, &outs[1].hidden);
    assert!(err < 1e-4, "residual rel rmse {err}");
}

// ---- The real checkpoint against a GGUF of the same model ------------------------------------

/// `t`'s bytes as `g` makes them (its recipe, else its file range), the first `limit` of them
/// when given.
fn read_tensor(g: &GgufFile, t: &TensorInfo, limit: Option<u64>) -> Vec<u8> {
    let len = limit.map_or(t.size, |l| l.min(t.size));
    let read = |file: usize, offset: u64, len: u64| {
        let path = g.shard_paths().nth(file).unwrap().to_path_buf();
        let mut f = std::fs::File::open(path).unwrap();
        f.seek(SeekFrom::Start(offset)).unwrap();
        let mut v = vec![0u8; len as usize];
        f.read_exact(&mut v).unwrap();
        v
    };
    let mut v = match g.recipe(&t.name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        Some(Recipe::Gather(parts)) => parts
            .iter()
            .flat_map(|p| read(p.file, p.offset, p.len))
            .collect(),
        None => read(t.file_index, t.offset, len),
    };
    v.truncate(len as usize);
    v
}

/// `t` as f32 (its first `rows` rows when given), times its per-tensor or per-expert `.scale`.
fn values(g: &GgufFile, t: &TensorInfo, rows: Option<u64>) -> Vec<f32> {
    let row = ggml::row_size(t.ggml_type, t.shape[0] as usize) as u64;
    let bytes = read_tensor(g, t, rows.map(|r| r * row));
    let n = bytes.len() / row as usize * t.shape[0] as usize;
    let mut v = if t.ggml_type == types::F32 {
        f32s(&bytes)
    } else {
        ggml::dequantize(t.ggml_type, &bytes, n)
    };
    let scale = t.name.strip_suffix(".weight").map(|b| format!("{b}.scale"));
    if let Some(s) = scale.and_then(|s| g.tensor(&s)) {
        let s = f32s(&read_tensor(g, &s, None));
        // One scale, or one per expert (the outermost dim).
        let per = if s.len() == 1 {
            v.len()
        } else {
            (t.shape[0] * t.shape[1]) as usize
        };
        for (i, x) in v.iter_mut().enumerate() {
            *x *= s[i / per];
        }
    }
    v
}

/// `|a - b| / |b|` over all values.
fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
#[ignore = "needs the NVFP4 checkpoint (LEGION_TEST_QWEN4_HF) and a GGUF (LEGION_TEST_QWEN4_GGUF)"]
fn real_checkpoint_matches_the_gguf() {
    let (Ok(hf), Ok(gguf)) = (
        std::env::var("LEGION_TEST_QWEN4_HF"),
        std::env::var("LEGION_TEST_QWEN4_GGUF"),
    ) else {
        eprintln!("skipping: LEGION_TEST_QWEN4_HF / LEGION_TEST_QWEN4_GGUF not set");
        return;
    };
    let t0 = std::time::Instant::now();
    let ours = GgufFile::open(&hf).unwrap();
    let want = GgufFile::open(&gguf).unwrap();
    eprintln!("opened in {:.1} s", t0.elapsed().as_secs_f64());
    let mut problems = Vec::new();

    // Metadata: every architecture key of the GGUF, with the same value (numbers compared as
    // numbers: the GGUF's integer types may differ).
    let num = |v: &Value| -> Vec<f64> {
        match v {
            Value::Array(a) => a.iter().filter_map(Value::as_f64).collect(),
            v => v.as_f64().into_iter().collect(),
        }
    };
    for (key, value) in want.metadata() {
        if !key.starts_with("qwen4exp.") && key != "tokenizer.ggml.pre" {
            continue;
        }
        match ours.get(&key) {
            None => problems.push(format!("missing key {key}")),
            Some(v) if num(&v) != num(&value) || v.as_str() != value.as_str() => {
                problems.push(format!("key {key}: {v} vs GGUF {value}"))
            }
            Some(_) => {}
        }
    }
    for key in ours.keys() {
        if key.starts_with("qwen4exp.") && want.get(&key).is_none() {
            problems.push(format!("extra key {key}"));
        }
    }
    let n_tokens = |g: &GgufFile| {
        g.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array().map(<[Value]>::len))
    };
    if n_tokens(&ours) != n_tokens(&want) {
        problems.push(format!(
            "tokens: {:?} vs GGUF {:?}",
            n_tokens(&ours),
            n_tokens(&want)
        ));
    }

    // Tensor names and shapes (types differ: the GGUF is quantized its own way).
    let mut theirs: std::collections::BTreeMap<String, TensorInfo> = want
        .tensors()
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    let mut mine = std::collections::BTreeMap::new();
    for t in ours.tensors() {
        match theirs.remove(&t.name) {
            None if t.name.ends_with(".scale") => {}
            None => problems.push(format!("extra tensor {}", t.name)),
            Some(r) => {
                if r.shape != t.shape {
                    problems.push(format!(
                        "tensor {}: {:?} vs GGUF {:?}",
                        t.name, t.shape, r.shape
                    ));
                }
                mine.insert(t.name.clone(), (t, r));
            }
        }
    }
    problems.extend(theirs.keys().map(|n| format!("missing tensor {n}")));

    // Values of every tensor of a recurrent layer, the PLE layer, an attention layer, the last
    // one, and the globals (the big ones by their first rows).
    let layers = ["blk.0.", "blk.1.", "blk.3.", "blk.47."];
    for (name, (a, b)) in &mine {
        let global = !name.starts_with("blk.");
        if name == "per_layer_token_embd.weight"
            || !(global || layers.iter().any(|l| name.starts_with(l)))
        {
            continue;
        }
        // Big tensors by their first rows: 8192 of a matrix, 32 experts of a stack.
        let big = a.shape.iter().product::<u64>() > (64 << 20);
        let rows = big.then(|| {
            if a.shape.len() > 2 {
                32 * a.shape[1]
            } else {
                8192
            }
        });
        let t = std::time::Instant::now();
        let (x, y) = (values(&ours, a, rows), values(&want, b, rows));
        let e = rel_rmse(&x, &y);
        // Tensors the GGUF keeps in F32 (norms, routers, ...) come from the same BF16 values.
        let limit = if b.ggml_type == types::F32 { 1e-6 } else { 0.2 };
        eprintln!(
            "{name:40} {:>6} vs {:>6}  rel rmse {e:.5}{}  ({:.1} s)",
            a.type_name(),
            b.type_name(),
            if e > limit { "  <-- over" } else { "" },
            t.elapsed().as_secs_f64()
        );
        if e.is_nan() || e > limit {
            problems.push(format!("tensor {name}: rel rmse {e:.5} > {limit}"));
        }
    }

    // The n-gram table: the rows of a few tokens' n-grams, through each model's hash and the
    // n-gram store (direct reads, as on a GPU run), FP8 times the scale against the GGUF's.
    let params = |g: &GgufFile| {
        let hp = Hparams::from_gguf(g).unwrap();
        legion::model::qwen4::params(&hp)
            .unwrap()
            .ple
            .clone()
            .unwrap()
    };
    let (pa, pb) = (params(&ours), params(&want));
    let opts = TableOptions {
        io: TableIo::Direct,
        ram: Default::default(),
        ram_cache_bytes: Some(0),
        inflight: 8,
        aio_max: lookup_table::DEFAULT_AIO_MAX,
        keep_awake_ms: 0,
        node: None,
        unified: false,
    };
    let open =
        |g: &GgufFile| LookupTable::open(Model::ngram_table_info(g).unwrap(), &opts, None).unwrap();
    let (ta, tb) = (open(&ours), open(&want));
    let ty = |g: &GgufFile| g.tensor("per_layer_token_embd.weight").unwrap().ggml_type;
    let (ya, yb) = (ty(&ours), ty(&want));
    let dim = pa.head_dim as usize;
    let none = u32::MAX;
    let mut worst = 0f64;
    let mut remapped = 0;
    // A few hand-picked n-grams (sequence start, EOS cuts), then 32 more.
    let mut grams = vec![
        (9707u32, [none, none]),
        (11, [9707, 1879]),
        (248044, [151, 9707]),
        (1879, [248044, 11]),
        (100, [200, 300]),
        (247000, [5, 6]),
    ];
    grams.extend((0..32u32).map(|i| (1000 + 7919 * i, [3 + 104_729 * i % 248_000, 17 + i])));
    for (tok, prev) in grams {
        let (mut ra, mut rb) = (Vec::new(), Vec::new());
        pa.rows(tok, &prev, &mut ra);
        pb.rows(tok, &prev, &mut rb);
        let ids = |r: &[i32]| r.iter().map(|&x| x as u32).collect::<Vec<u32>>();
        let (ba, bb) = (ta.fetch(&ids(&ra)).unwrap(), tb.fetch(&ids(&rb)).unwrap());
        let (sa, sb) = (ta.row_bytes(), tb.row_bytes());
        for h in 0..ra.len() {
            let x = legion::model::qwen4::ple_row_f32(ya, pa.fp8_scale, &ba[h * sa..][..sa], dim);
            let y = legion::model::qwen4::ple_row_f32(yb, pb.fp8_scale, &bb[h * sb..][..sb], dim);
            let e = rel_rmse(&x, &y);
            worst = worst.max(e);
            // Rows of shards stored out of index order are read from elsewhere in the file.
            let moved = ra[h] != rb[h];
            remapped += moved as usize;
            if h == 0 || moved {
                eprintln!(
                    "token {tok} after {prev:?} head {h}: row {} (stored at {}) rel rmse {e:.4}; x[..4] {:?} y[..4] {:?}",
                    rb[h],
                    ra[h],
                    &x[..4],
                    &y[..4]
                );
            }
        }
    }
    eprintln!("n-gram rows: {remapped} read through the block order");
    if pa.blocks.is_some() && remapped == 0 {
        problems.push("no n-gram row came from a shard stored out of order".into());
    }
    eprintln!("n-gram rows: worst rel rmse {worst:.4}");
    if worst.is_nan() || worst >= 0.2 {
        problems.push(format!("n-gram rows: rel rmse {worst:.4}"));
    }
    assert!(
        problems.is_empty(),
        "{} differences:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

// ---- The real checkpoint's logits against the GGUF's, on the GPU -----------------------------

const QUALITY_PROMPTS: &[(&str, &str)] = &[
    (
        "apollo",
        "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the lunar module \
         Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled the goal \
         President Kennedy had set eight years earlier, and the next mission",
    ),
    (
        "code",
        "fn binary_search(xs: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = \
         (0, xs.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match \
         xs[mid].cmp(&target) {\n            std::cmp::Ordering::Less => lo = mid + 1,\n            \
         std::cmp::Ordering::Greater => hi = mid,\n            std::cmp::Ordering::Equal => return \
         Some(mid),\n        }\n    }\n    None\n}\n\nfn main() {\n    let xs = [1, 3, 5, 7, 9, 11];\n",
    ),
    (
        "chat",
        "<|im_start|>user\nExplain in three sentences why the sky is blue.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\nThe sky looks blue because",
    ),
    (
        "chinese",
        "长江是中国最长的河流，全长约六千三百公里，发源于青藏高原，流经十一个省级行政区，最后在上海注入东海。\
         长江流域",
    ),
];

/// `prompts`' all-position logits from the model in `g`, on the GPU with the weights placed as
/// the engine places them (the n-gram table read through the n-gram store, direct reads).
fn gpu_logits(g: &GgufFile, prompts: &[Vec<u32>]) -> Vec<Vec<f32>> {
    use legion::ggml::{DeviceKind, device_by_kind};
    use legion::model::core::Placement;
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let t = std::time::Instant::now();
    let mut model = Model::load(g, |info| match Model::placement(&info.name, false) {
        Placement::Host => host,
        Placement::Device => backend.buffer_type(),
    })
    .unwrap();
    let opts = TableOptions {
        io: TableIo::Direct,
        ram: Default::default(),
        ram_cache_bytes: Some(1 << 30),
        inflight: 32,
        aio_max: lookup_table::DEFAULT_AIO_MAX,
        keep_awake_ms: 0,
        node: None,
        unified: true,
    };
    model.set_ngram_table(
        LookupTable::open(Model::ngram_table_info(g).unwrap(), &opts, None).unwrap(),
    );
    eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    prompts
        .iter()
        .map(|ids| {
            let t = std::time::Instant::now();
            let l = model.prompt_logits_all(&mut backend, ids).unwrap();
            eprintln!(
                "  {} tokens in {:.2} s",
                ids.len(),
                t.elapsed().as_secs_f64()
            );
            l
        })
        .collect()
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

/// The NVFP4 checkpoint's prompt logits against the GGUF's (a different quantization of the same
/// model), both on the GPU: the same tokens, and per position the same top-1 token and a small
/// KL(GGUF || ours). Needs a GPU and both models (loaded one after the other).
#[test]
#[ignore = "needs a GPU, the NVFP4 checkpoint (LEGION_TEST_QWEN4_HF) and a GGUF (LEGION_TEST_QWEN4_GGUF)"]
fn real_checkpoint_logits_agree_with_the_gguf() {
    let (Ok(hf), Ok(gguf)) = (
        std::env::var("LEGION_TEST_QWEN4_HF"),
        std::env::var("LEGION_TEST_QWEN4_GGUF"),
    ) else {
        eprintln!("skipping: LEGION_TEST_QWEN4_HF / LEGION_TEST_QWEN4_GGUF not set");
        return;
    };
    let long: String = (1..=60)
        .map(|i| {
            format!(
                "Fact {i}: {i} times {} is {}, and {i} squared is {}. ",
                i % 7 + 3,
                i * (i % 7 + 3),
                i * i
            )
        })
        .collect();
    let mut texts: Vec<(&str, String)> = QUALITY_PROMPTS
        .iter()
        .map(|(n, t)| (*n, t.to_string()))
        .collect();
    texts.push(("facts", long + "Fact 61:"));
    let encode = |g: &GgufFile| {
        let tok = legion::tokenizer::Tokenizer::from_gguf(g).unwrap();
        texts
            .iter()
            .map(|(_, t)| tok.encode(t, true))
            .collect::<Vec<_>>()
    };
    let want = GgufFile::open(&gguf).unwrap();
    let ids = encode(&want);
    let theirs = gpu_logits(&want, &ids);
    drop(want);
    let ours_file = GgufFile::open(&hf).unwrap();
    assert_eq!(encode(&ours_file), ids, "the tokenizers differ");
    let n_vocab = Hparams::from_gguf(&ours_file).unwrap().n_vocab as usize;
    let ours = gpu_logits(&ours_file, &ids);
    let (mut all_top1, mut all_n) = (0, 0);
    for (((name, _), a), b) in texts.iter().zip(&ours).zip(&theirs) {
        assert_eq!(a.len(), b.len(), "{name}: logit count");
        let (mut top1, mut kl_mean, mut kl_max) = (0, 0f64, 0f64);
        let n = a.len() / n_vocab;
        for (o, t) in a.chunks_exact(n_vocab).zip(b.chunks_exact(n_vocab)) {
            assert!(o.iter().all(|x| x.is_finite()), "{name}: non-finite logits");
            top1 += (argmax(o) == argmax(t)) as usize;
            let (lo, lt) = (log_softmax(o), log_softmax(t));
            let kl: f64 = lt.iter().zip(&lo).map(|(t, o)| t.exp() * (t - o)).sum();
            kl_mean += kl / n as f64;
            kl_max = kl_max.max(kl);
        }
        eprintln!(
            "{name:8}: top-1 {top1}/{n}, KL(GGUF || NVFP4) mean {kl_mean:.4} max {kl_max:.4}"
        );
        all_top1 += top1;
        all_n += n;
        assert!(kl_mean < 0.1, "{name}: mean KL {kl_mean}");
    }
    eprintln!(
        "all: top-1 {all_top1}/{all_n} ({:.1}%)",
        100.0 * all_top1 as f64 / all_n as f64
    );
    assert!(all_top1 * 100 >= all_n * 85, "top-1 {all_top1}/{all_n}");
}
