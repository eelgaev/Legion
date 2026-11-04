//! A tiny DeepSeek-V4.1-Flash checkpoint (`deepseek_v41`) with the released one's names and
//! formats (dense FP8 E4M3 with 32 x 32 E8M0 block scales, FP4 E2M1 routed experts with an
//! E8M0 scale per 32, BF16 / F32 elsewhere, FP8 Engram tables, MTP / vision tensors that are not
//! loaded) opened as a synthetic `deepseek41` GGUF (`src/hf/deepseek41.rs`): the metadata, the
//! tensors as shipped (FP8 bytes, exact F32 block scales, experts repacked into MXFP4), the
//! Engram tables left in their file with their scales, the hash parameters made from the config
//! and tokenizer, and `legion gguf check`'s report on it.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use legion::ggml::{self, types};
use legion::gguf::{GgufFile, Recipe, TensorInfo};
use legion::model::core::Hparams;
use legion::model::deepseek41::check;
use legion::model::fp8::e8m0_to_f32;
use serde_json::json;

const LAYERS: usize = 4;
const EMBD: usize = 64;
const HEADS: usize = 4;
const HEAD: usize = 64;
const ROT: usize = 16;
const Q_LORA: usize = 32;
const O_GROUPS: usize = 2;
const O_LORA: usize = 32;
const IDX_HEADS: usize = 2;
const IDX_DIM: usize = 32;
const HC: usize = 2;
const EXPERTS: usize = 4;
const FF: usize = 64;
const VOCAB: usize = 66;
/// Engram on layers 1 and 2: bigrams and trigrams, 2 heads, 32-wide rows, buckets above 49.
const ENGRAM_LAYERS: [usize; 2] = [1, 2];
const ENGRAM_DIM: usize = 32;
const ENGRAM_COLS: usize = 4;
const PRIMES: [u64; 8] = [53, 59, 61, 67, 71, 73, 79, 83];
const ENGRAM_ROWS: [usize; 2] = [240, 306];
/// Compressed ids: the two specials, then the 26 letters (upper case and a leading space fold).
const COMPRESSED: usize = 28;

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

fn bf16s(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(2 * n, seed)
        .chunks(2)
        // Exponents near 1: finite, normal.
        .flat_map(|c| [c[0], (c[1] & 0x80) | 0x3c | (c[1] & 0x03)])
        .collect()
}

/// Finite E4M3 values.
fn e4m3s(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b })
        .collect()
}

/// E8M0 scales around 1.
fn e8m0s(n: usize, seed: u64) -> Vec<u8> {
    rand_bytes(n, seed)
        .into_iter()
        .map(|b| 120 + b % 12)
        .collect()
}

/// Adds one tensor (random bytes of its dtype) to a shard's list.
type Add<'a> = dyn FnMut(&mut Vec<T>, String, &'static str, &[usize]) + 'a;

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

/// Writes the checkpoint into `dir`: the model in one shard, the Engram tables in another (as
/// the release has them in shards of their own).
fn checkpoint(dir: &Path) -> HashMap<String, T> {
    let mut main: Vec<T> = Vec::new();
    let mut tables: Vec<T> = Vec::new();
    let mut seed = 0;
    let mut add = |to: &mut Vec<T>, name: String, dtype: &'static str, shape: &[usize]| {
        seed += 1;
        let n: usize = shape.iter().product();
        let bytes = match dtype {
            "BF16" => bf16s(n, seed),
            "F32" => rand_bytes(n, seed)
                .iter()
                .flat_map(|&b| (b as f32 / 64.0 - 2.0).to_le_bytes())
                .collect(),
            "F8_E4M3" => e4m3s(n, seed),
            "F8_E8M0" => e8m0s(n, seed),
            "I8" => rand_bytes(n, seed),
            _ => unreachable!(),
        };
        to.push(T {
            name,
            dtype,
            shape: shape.to_vec(),
            bytes,
        });
    };
    // A dense FP8 weight [rows, k] and its block scales.
    let fp8 = |add: &mut Add<'_>, to: &mut Vec<T>, base: String, rows: usize, k: usize| {
        add(to, format!("{base}.weight"), "F8_E4M3", &[rows, k]);
        add(
            to,
            format!("{base}.scale"),
            "F8_E8M0",
            &[rows.div_ceil(32), k.div_ceil(32)],
        );
    };
    let m = &mut main;
    add(m, "embed.weight".into(), "BF16", &[VOCAB, EMBD]);
    add(m, "head.weight".into(), "BF16", &[VOCAB, EMBD]);
    add(m, "norm.weight".into(), "BF16", &[EMBD]);
    let n_mix = (2 + HC) * HC;
    for il in 0..LAYERS {
        let l = |s: &str| format!("layers.{il}.{s}");
        add(m, l("attn_norm.weight"), "BF16", &[EMBD]);
        add(m, l("ffn_norm.weight"), "BF16", &[EMBD]);
        fp8(&mut add, m, l("attn.wq_a"), Q_LORA, EMBD);
        add(m, l("attn.q_norm.weight"), "BF16", &[Q_LORA]);
        fp8(&mut add, m, l("attn.wq_b"), HEADS * HEAD, Q_LORA);
        fp8(&mut add, m, l("attn.wkv"), HEAD, EMBD);
        add(m, l("attn.kv_norm.weight"), "BF16", &[HEAD]);
        add(m, l("attn.attn_sink"), "F32", &[HEADS]);
        let wo_a = (O_GROUPS * O_LORA, HEADS * HEAD / O_GROUPS);
        fp8(&mut add, m, l("attn.wo_a"), wo_a.0, wo_a.1);
        fp8(&mut add, m, l("attn.wo_b"), EMBD, O_GROUPS * O_LORA);
        for which in ["attn", "ffn"] {
            add(m, l(&format!("hc_{which}_fn")), "F32", &[n_mix, HC * EMBD]);
            add(m, l(&format!("hc_{which}_base")), "F32", &[n_mix]);
            add(m, l(&format!("hc_{which}_scale")), "F32", &[3]);
        }
        if il == 1 {
            // The kv and index source (compression ratio 2).
            add(m, l("attn.compressor.wkv.weight"), "BF16", &[HEAD, EMBD]);
            add(m, l("attn.compressor.wgate.weight"), "BF16", &[HEAD, EMBD]);
            add(m, l("attn.compressor.norm.weight"), "BF16", &[HEAD]);
            add(m, l("attn.indexer.wk.weight"), "BF16", &[IDX_DIM, HEAD]);
            add(m, l("attn.indexer.k_norm.weight"), "BF16", &[IDX_DIM]);
            fp8(
                &mut add,
                m,
                l("attn.indexer.wq_b"),
                IDX_HEADS * IDX_DIM,
                Q_LORA,
            );
            add(
                m,
                l("attn.indexer.weights_proj.weight"),
                "BF16",
                &[IDX_HEADS, EMBD],
            );
        }
        add(m, l("ffn.gate.weight"), "BF16", &[EXPERTS, EMBD]);
        add(m, l("ffn.gate.bias"), "F32", &[EXPERTS]);
        add(m, l("ffn.gate.bias_vl"), "F32", &[EXPERTS]);
        for e in 0..EXPERTS {
            for (w, rows, k) in [("w1", FF, EMBD), ("w3", FF, EMBD), ("w2", EMBD, FF)] {
                let b = l(&format!("ffn.experts.{e}.{w}"));
                add(m, format!("{b}.weight"), "I8", &[rows, k / 2]);
                add(m, format!("{b}.scale"), "F8_E8M0", &[rows, k / 32]);
            }
        }
        for (w, rows, k) in [("w1", FF, EMBD), ("w3", FF, EMBD), ("w2", EMBD, FF)] {
            fp8(&mut add, m, l(&format!("ffn.shared_experts.{w}")), rows, k);
        }
        if let Some(li) = ENGRAM_LAYERS.iter().position(|&x| x == il) {
            add(m, l("engram.q_weight"), "BF16", &[HC, EMBD]);
            add(m, l("engram.k_weight"), "BF16", &[HC, EMBD]);
            let width = ENGRAM_COLS * ENGRAM_DIM;
            fp8(&mut add, m, l("engram.wkv"), (HC + 1) * EMBD, width);
            let rows = ENGRAM_ROWS[li];
            add(
                &mut tables,
                l("engram.embed.weight"),
                "F8_E4M3",
                &[rows, ENGRAM_DIM],
            );
            add(
                &mut tables,
                l("engram.embed.scale"),
                "F8_E8M0",
                &[rows, ENGRAM_DIM / 32],
            );
        }
    }
    // Not loaded: the MTP / DSpark layer, the vision tower and its aligner.
    add(m, "mtp.0.attn_norm.weight".into(), "BF16", &[EMBD]);
    add(m, "vision.blocks.0.norm.weight".into(), "BF16", &[EMBD]);
    add(m, "aligner.w1.weight".into(), "BF16", &[EMBD, EMBD]);
    add(m, "image_newline".into(), "BF16", &[EMBD]);

    write_safetensors(&dir.join("model-00001-of-00002.safetensors"), &main);
    write_safetensors(&dir.join("model-00002-of-00002.safetensors"), &tables);
    let mut map = serde_json::Map::new();
    for t in &main {
        map.insert(t.name.clone(), json!("model-00001-of-00002.safetensors"));
    }
    for t in &tables {
        map.insert(t.name.clone(), json!("model-00002-of-00002.safetensors"));
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"metadata": {}, "weight_map": map})).unwrap(),
    )
    .unwrap();
    let text = json!({
        "model_type": "deepseek_v41_text",
        "vocab_size": VOCAB, "hidden_size": EMBD, "moe_intermediate_size": FF,
        "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": 1,
        "head_dim": HEAD, "qk_rope_head_dim": ROT, "q_lora_rank": Q_LORA,
        "o_lora_rank": O_LORA, "o_groups": O_GROUPS, "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-20, "max_position_embeddings": 4096, "rope_theta": 10000,
        "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 1024},
        "n_routed_experts": EXPERTS, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus", "norm_topk_prob": true, "routed_scaling_factor": 1.5,
        "sliding_window": 8, "compress_ratios": [0, 2, 2, 0, 0],
        "compress_rope_theta": 160000, "kv_source_layer_ids": [1],
        "index_source_layer_ids": [1], "index_n_heads": IDX_HEADS, "index_head_dim": IDX_DIM,
        "index_topk": 16, "candidate_source_layer_id": 1, "candidate_topk_blocks": 4,
        "candidate_block_size": 8, "hc_mult": HC, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6
    });
    let engram = json!({
        "engram_layer_ids": ENGRAM_LAYERS, "engram_num_embeddings": ENGRAM_ROWS,
        "engram_max_ngram_size": 3, "engram_vocab_size": 50, "engram_n_heads": 2,
        "engram_head_dim": ENGRAM_DIM, "engram_pad_token_id": 65,
        "engram_compressed_vocab_size": COMPRESSED,
        "num_nextn_predict_layers": 1, "dspark_target_layer_ids": [3]
    });
    let mut text = text;
    text.as_object_mut()
        .unwrap()
        .extend(engram.as_object().unwrap().clone());
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
            "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                    "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                    "expert_dtype": "fp4"},
            "text_config": text,
            "vision_config": {"model_type": "deepseek_v41_vision"}
        }))
        .unwrap(),
    )
    .unwrap();
    // Byte-level BPE: the two specials, `a`..`z`, `A`..`Z`, `Ġa`..`Ġl`.
    let mut vocab = serde_json::Map::new();
    let letters = |c: u8| (0..26).map(move |i| ((c + i) as char).to_string());
    for (i, t) in letters(b'a')
        .chain(letters(b'A'))
        .chain(letters(b'a').take(12).map(|s| format!("Ġ{s}")))
        .enumerate()
    {
        vocab.insert(t, json!(i + 2));
    }
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 0, "content": "<｜begin▁of▁sentence｜>", "special": true},
                {"id": 1, "content": "<｜end▁of▁sentence｜>", "special": true}
            ],
            "pre_tokenizer": null,
            "decoder": {"type": "ByteLevel"}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({
            "add_bos_token": false,
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜end▁of▁sentence｜>"
        }))
        .unwrap(),
    )
    .unwrap();
    main.into_iter()
        .chain(tables)
        .map(|t| (t.name.clone(), t))
        .collect()
}

fn read_tensor(g: &GgufFile, t: &TensorInfo) -> Vec<u8> {
    let read = |file: usize, offset: u64, len: u64| {
        let path = g.shard_paths().nth(file).unwrap().to_path_buf();
        let mut f = std::fs::File::open(path).unwrap();
        f.seek(SeekFrom::Start(offset)).unwrap();
        let mut v = vec![0u8; len as usize];
        f.read_exact(&mut v).unwrap();
        v
    };
    match g.recipe(&t.name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        Some(Recipe::Gather(parts)) => parts
            .iter()
            .flat_map(|p| read(p.file, p.offset, p.len))
            .collect(),
        None => read(t.file_index, t.offset, t.size),
    }
}

fn bytes(g: &GgufFile, name: &str) -> Vec<u8> {
    read_tensor(g, &g.tensor(name).unwrap_or_else(|| panic!("no {name}")))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|&c| i64::from_le_bytes(c))
        .collect()
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

/// E2M1 code -> value.
fn e2m1(c: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

#[test]
fn deepseek41_opens_as_shipped() {
    let dir = tempfile::tempdir().unwrap();
    let src = checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    assert_eq!(g.architecture().as_deref(), Some("deepseek41"));
    let k = |s: &str| format!("deepseek41.{s}");
    assert_eq!(g.get_u64(&k("block_count")), Some(LAYERS as u64));
    assert_eq!(g.get_u64(&k("expert_gating_func")), Some(4));
    assert_eq!(g.get_u64(&k("rope.dimension_count")), Some(ROT as u64));
    assert_eq!(
        g.get_u64(&k("attention.output_lora_rank")),
        Some(O_LORA as u64)
    );
    assert_eq!(
        ints(&g, &k("attention.compress_ratios")),
        vec![0, 2, 2, 0, 0]
    );
    assert_eq!(ints(&g, &k("engram.num_embeddings")), vec![240, 306]);
    assert_eq!(
        g.get_str("tokenizer.ggml.pre").as_deref(),
        Some("joyai-llm")
    );
    assert_eq!(g.get_u64("tokenizer.ggml.bos_token_id"), Some(0));
    assert_eq!(g.get_u64("tokenizer.ggml.eos_token_id"), Some(1));
    // Dense FP8 with block scales: refused before any weight loads (no matmul takes them yet).
    let err = Hparams::from_gguf(&g).unwrap_err().to_string();
    assert!(
        err.contains("blk.0.attn_q_a.weight") && err.contains("block scales"),
        "{err}"
    );

    // Dense FP8: the E4M3 bytes as stored, the block scales as F32 [k / 32, m / 32].
    let q_b = g.tensor("blk.0.attn_q_b.weight").unwrap();
    assert_eq!(
        (q_b.ggml_type, q_b.shape.clone()),
        (types::I8, vec![32, 256])
    );
    assert_eq!(
        bytes(&g, "blk.0.attn_q_b.weight"),
        src["layers.0.attn.wq_b.weight"].bytes
    );
    let s = g.tensor("blk.0.attn_q_b.scale").unwrap();
    assert_eq!((s.ggml_type, s.shape.clone()), (types::F32, vec![1, 8]));
    let want: Vec<f32> = src["layers.0.attn.wq_b.scale"]
        .bytes
        .iter()
        .map(|&b| e8m0_to_f32(b))
        .collect();
    assert_eq!(f32s(&bytes(&g, "blk.0.attn_q_b.scale")), want);
    let o_a = g.tensor("blk.3.attn_output_a.weight").unwrap();
    assert_eq!(o_a.shape, vec![128, 32, 2]);
    assert_eq!(
        g.tensor("blk.3.attn_output_a.scale").unwrap().shape,
        vec![4, 2]
    );
    // BF16 as shipped; norms widened to F32 exactly; F32 mixers and the router bias as is.
    assert_eq!(
        g.tensor("token_embd.weight").unwrap().ggml_type,
        types::BF16
    );
    let widen = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect()
    };
    assert_eq!(
        f32s(&bytes(&g, "blk.2.attn_q_a_norm.weight")),
        widen(&src["layers.2.attn.q_norm.weight"].bytes)
    );
    assert_eq!(
        bytes(&g, "blk.2.hc_ffn_fn.weight"),
        src["layers.2.hc_ffn_fn"].bytes
    );
    assert_eq!(
        bytes(&g, "blk.2.exp_probs_b.bias"),
        src["layers.2.ffn.gate.bias"].bytes
    );
    assert_eq!(
        bytes(&g, "blk.0.attn_sinks.weight"),
        src["layers.0.attn.attn_sink"].bytes
    );

    // FP4 experts: stacked MXFP4 with the same values.
    let down = g.tensor("blk.1.ffn_down_exps.weight").unwrap();
    assert_eq!(
        (down.ggml_type, down.shape.clone()),
        (types::MXFP4, vec![FF as u64, EMBD as u64, EXPERTS as u64])
    );
    let got = ggml::dequantize(types::MXFP4, &bytes(&g, &down.name), FF * EMBD * EXPERTS);
    for e in 0..EXPERTS {
        let w = &src[&format!("layers.1.ffn.experts.{e}.w2.weight")].bytes;
        let s = &src[&format!("layers.1.ffn.experts.{e}.w2.scale")].bytes;
        for i in 0..EMBD * FF {
            let (r, c) = (i / FF, i % FF);
            let code = (w[r * FF / 2 + c / 2] >> (4 * (c % 2))) & 15;
            let want = e2m1(code) * e8m0_to_f32(s[r * FF / 32 + c / 32]);
            assert_eq!(got[e * EMBD * FF + i], want, "expert {e} element {i}");
        }
    }

    // The Engram tables stay in their shard, with their scale bytes.
    for (li, &il) in ENGRAM_LAYERS.iter().enumerate() {
        let name = format!("blk.{il}.engram_embd.weight");
        let t = g.tensor(&name).unwrap();
        assert!(g.recipe(&name).is_none());
        assert_eq!(t.file_index, 1);
        assert_eq!(
            (t.ggml_type, t.shape.clone()),
            (types::I8, vec![ENGRAM_DIM as u64, ENGRAM_ROWS[li] as u64])
        );
        assert_eq!(
            bytes(&g, &name),
            src[&format!("layers.{il}.engram.embed.weight")].bytes
        );
        let s = format!("blk.{il}.engram_embd.scale");
        assert_eq!(g.tensor(&s).unwrap().shape, vec![1, ENGRAM_ROWS[li] as u64]);
        assert_eq!(
            bytes(&g, &s),
            src[&format!("layers.{il}.engram.embed.scale")].bytes
        );
    }
    // The hash parameters, made from the config and the tokenizer.
    assert_eq!(
        i64s(&bytes(&g, "engram_primes.weight")),
        PRIMES.map(|p| p as i64)
    );
    assert_eq!(
        i64s(&bytes(&g, "engram_offsets.weight")),
        [0, 53, 112, 173, 0, 71, 144, 223]
    );
    assert_eq!(g.tensor("engram_multipliers.weight").unwrap().shape, [3, 2]);
    let map: Vec<i32> = bytes(&g, "engram_token_map.weight")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect();
    let want: Vec<i32> = (0..VOCAB as i32)
        .map(|i| match i {
            0..28 => i,
            28..54 => i - 26,
            _ => i - 52,
        })
        .collect();
    assert_eq!(map, want);
    // Not loaded.
    for t in g.tensors() {
        assert!(
            !t.name.contains("mtp") && !t.name.contains("vision") && !t.name.contains("aligner"),
            "{}",
            t.name
        );
    }
}

#[test]
fn gguf_check_reports_what_does_not_run_yet() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    let r = check::check(&g, 8).unwrap();
    assert!(r.missing.is_empty(), "{:?}", r.missing);
    assert!(r.shapes.is_empty(), "{:?}", r.shapes);
    assert!(r.unread.is_empty(), "{:?}", r.unread);
    assert!(r.engram.is_empty(), "{:?}", r.engram);
    assert!(
        r.engram_notes.iter().any(|n| n.contains("all finite")),
        "{:?}",
        r.engram_notes
    );
    // Every layer's MXFP4 experts and every dense FP8 weight: no kernel takes them yet.
    assert_eq!(r.experts.len(), LAYERS);
    let dense_fp8 = LAYERS * 8 + 1 + ENGRAM_LAYERS.len();
    assert_eq!(r.block_fp8.len(), dense_fp8, "{:?}", r.block_fp8);
    assert!(r.cuda_types.is_empty(), "{:?}", r.cuda_types);
    assert!(!r.ok());
}
