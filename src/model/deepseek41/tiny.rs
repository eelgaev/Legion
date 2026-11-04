//! A tiny `deepseek41` with random weights and every DeepSeek-V4.1-Flash feature on a small
//! scale: [`N_LAYER`] layers with the compression ratios [`RATIOS`] (window-only, ratio 2 and
//! ratio 1), shared compression (kv sources [`KV_SOURCES`], index sources [`INDEX_SOURCES`]),
//! candidate blocks from layer [`CAND_LAYER`], a [`WINDOW`]-position window and [`TOP_K`]
//! selected positions (so a few dozen tokens exercise them all), hyper-connections, the grouped
//! low-rank output, YaRN, `sqrt(softplus)` routing over Q8_0 experts and two Engram layers.
//! Byte-level tokenizer.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufError, GgufWriter, Value};
use crate::model::qwen35::tiny::{SPECIAL_TOKENS, byte_tokenizer};

use super::ARCH;

pub const N_LAYER: usize = 6;
pub const RATIOS: [u32; N_LAYER] = [0, 2, 2, 1, 1, 1];
pub const KV_SOURCES: [u32; 2] = [1, 3];
pub const INDEX_SOURCES: [u32; 3] = [1, 3, 5];
pub const CAND_LAYER: u32 = 3;
pub const CAND_BLOCK: u32 = 2;
pub const CAND_BLOCKS: u32 = 3;
pub const E: i64 = 128;
pub const HC: i64 = 4;
pub const N_HEAD: i64 = 4;
pub const HEAD: i64 = 32;
pub const N_ROT: i64 = 16;
pub const Q_LORA: i64 = 32;
pub const O_GROUPS: i64 = 2;
pub const O_LORA: i64 = 16;
pub const IDX_HEADS: i64 = 2;
pub const IDX_DIM: i64 = 16;
pub const TOP_K: u32 = 4;
pub const WINDOW: u32 = 8;
pub const N_EXPERT: i64 = 8;
pub const N_USED: i64 = 2;
pub const N_FF: i64 = 64;
pub const RMS_EPS: f32 = 1e-6;
pub const ENGRAM_LAYERS: [u32; 2] = [1, 3];
pub const NGRAM: i64 = 3;
pub const ENGRAM_HEADS: i64 = 2;
pub const ENGRAM_DIM: i64 = 8;
pub const ENGRAM_PAD: u32 = 2;
/// The Engram hash primes per layer and column (n-gram size, head).
pub const PRIMES: [[i64; 4]; 2] = [[13, 17, 19, 23], [29, 31, 37, 41]];

/// Table rows per n-gram column: column `c` takes rows `offset(c) .. offset(c) + prime`.
fn offsets(li: usize) -> Vec<i64> {
    PRIMES[li]
        .iter()
        .scan(0, |o, p| {
            let at = *o;
            *o += p;
            Some(at)
        })
        .collect()
}

/// The vocabulary size of the byte tokenizer.
pub fn n_vocab() -> i64 {
    (256 + SPECIAL_TOKENS.len()) as i64
}

/// The Engram tables file the main GGUF names (`deepseek41.engram.file`).
pub const ENGRAM_FILE: &str = "tiny-engram.gguf";

/// Where [`write_with`] puts the Engram tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tables {
    /// In the model's GGUF.
    Inline,
    /// In [`ENGRAM_FILE`] next to it, as the real model ships them.
    Separate,
    /// Nowhere: the main GGUF alone.
    Missing,
}

/// Writes the tiny model to `path` with its Engram tables where `tables` says. The weights are
/// the same wherever the tables go.
pub fn write_with(path: impl AsRef<Path>, seed: u64, tables: Tables) -> Result<(), GgufError> {
    let path = path.as_ref();
    let source = Value::String(format!("tiny-deepseek41-seed-{seed}"));
    let mut tables_file = GgufWriter::new();
    tables_file.set(
        "general.architecture",
        &Value::String(format!("{ARCH}-engram")),
    )?;
    tables_file.set(&format!("{ARCH}.engram.source"), &source)?;
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String(ARCH.into()))?;
    g.set("general.name", &Value::String("tiny-deepseek41".into()))?;
    byte_tokenizer(&mut g)?;
    let u = Value::U32;
    let f = Value::F32;
    let arr = |v: &[u32]| Value::Array(v.iter().map(|&x| Value::U32(x)).collect());
    let layers = N_LAYER as u32;
    let mut key = |k: &str, v: Value| g.set(&format!("{ARCH}.{k}"), &v).map(|_| ());
    key("block_count", u(layers))?;
    key("context_length", u(4096))?;
    key("embedding_length", u(E as u32))?;
    key("attention.head_count", u(N_HEAD as u32))?;
    key("attention.head_count_kv", u(1))?;
    key("attention.key_length", u(HEAD as u32))?;
    key("attention.value_length", u(HEAD as u32))?;
    key("attention.layer_norm_rms_epsilon", f(RMS_EPS))?;
    key("attention.q_lora_rank", u(Q_LORA as u32))?;
    key("attention.sliding_window", u(WINDOW))?;
    key("attention.indexer.head_count", u(IDX_HEADS as u32))?;
    key("attention.indexer.key_length", u(IDX_DIM as u32))?;
    key("attention.indexer.top_k", u(TOP_K))?;
    key("attention.output_group_count", u(O_GROUPS as u32))?;
    key("attention.output_lora_rank", u(O_LORA as u32))?;
    // As the real GGUF: three trailing (MTP / DSpark) entries past the layers.
    let mut ratios = RATIOS.to_vec();
    ratios.extend([0, 0, 0]);
    key("attention.compress_ratios", arr(&ratios))?;
    key("attention.compress_rope_freq_base", f(160_000.0))?;
    key("attention.candidate_block_size", u(CAND_BLOCK))?;
    key("attention.candidate_topk_blocks", u(CAND_BLOCKS))?;
    key(
        "attention.candidate_source_layer_id",
        Value::I32(CAND_LAYER as i32),
    )?;
    key("kv_source_layer_ids", arr(&KV_SOURCES))?;
    key("index_source_layer_ids", arr(&INDEX_SOURCES))?;
    key("rope.dimension_count", u(N_ROT as u32))?;
    key("rope.freq_base", f(10_000.0))?;
    key("rope.scaling.type", Value::String("yarn".into()))?;
    key("rope.scaling.factor", f(4.0))?;
    key("rope.scaling.original_context_length", u(32))?;
    key("rope.scaling.yarn_beta_fast", f(32.0))?;
    key("rope.scaling.yarn_beta_slow", f(1.0))?;
    key("expert_count", u(N_EXPERT as u32))?;
    key("expert_used_count", u(N_USED as u32))?;
    key("expert_gating_func", u(4))?;
    key("expert_scoring_func", Value::String("sqrtsoftplus".into()))?;
    key("expert_feed_forward_length", u(N_FF as u32))?;
    key("expert_shared_count", u(1))?;
    key("expert_weights_scale", f(1.5))?;
    key("expert_weights_norm", Value::Bool(true))?;
    let clamp = Value::Array(vec![Value::F32(1.5); N_LAYER]);
    key("swiglu_clamp_exp", clamp.clone())?;
    key("swiglu_clamp_shexp", clamp)?;
    key("hyper_connection.count", u(HC as u32))?;
    key("hyper_connection.sinkhorn_iterations", u(20))?;
    key("hyper_connection.epsilon", f(1e-6))?;
    key("hash_layer_count", u(0))?;
    key("engram.present", Value::Bool(true))?;
    key("engram.layer_ids", arr(&ENGRAM_LAYERS))?;
    key("engram_head_dim", u(ENGRAM_DIM as u32))?;
    key("engram_n_heads", u(ENGRAM_HEADS as u32))?;
    key("engram_max_ngram_size", u(NGRAM as u32))?;
    key("engram_pad_token_id", u(ENGRAM_PAD))?;
    key("engram.file", Value::String(ENGRAM_FILE.into()))?;
    key("engram.source", source)?;
    let rows: Vec<u32> = PRIMES
        .iter()
        .map(|p| p.iter().sum::<i64>() as u32)
        .collect();
    key("engram.num_embeddings", arr(&rows))?;

    // Shared by the F32 weights and the quantized experts, drawn in file order.
    let rng = std::cell::RefCell::new(Rng::new(seed));
    let rand = |n: i64, scale: f32, bias: f32| -> Vec<f32> {
        let mut rng = rng.borrow_mut();
        (0..n)
            .map(|_| bias + scale * (rng.next_f64() as f32 * 2.0 - 1.0))
            .collect()
    };
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    let add = |g: &mut GgufWriter, name: &str, shape: &[i64], scale: f32, bias: f32| {
        let v = rand(shape.iter().product(), scale, bias);
        g.add_tensor(name, types::F32, shape, &bytes(&v))
            .map(|_| ())
    };
    // A weight `[n_in, n_out]` scaled for unit-variance outputs.
    let w = |n_in: i64| (3.0 / n_in as f32).sqrt();
    let nv = n_vocab();
    add(&mut g, "token_embd.weight", &[E, nv], 1.0, 0.0)?;
    add(&mut g, "output_norm.weight", &[E], 0.1, 1.0)?;
    add(&mut g, "output.weight", &[E, nv], w(E), 0.0)?;
    let is = |list: &[u32], il: usize| list.contains(&(il as u32));
    for (il, &ratio) in RATIOS.iter().enumerate() {
        let t = |s: &str| format!("blk.{il}.{s}");
        add(&mut g, &t("attn_norm.weight"), &[E], 0.1, 1.0)?;
        add(&mut g, &t("ffn_norm.weight"), &[E], 0.1, 1.0)?;
        add(&mut g, &t("attn_q_a.weight"), &[E, Q_LORA], w(E), 0.0)?;
        add(&mut g, &t("attn_q_a_norm.weight"), &[Q_LORA], 0.1, 1.0)?;
        add(
            &mut g,
            &t("attn_q_b.weight"),
            &[Q_LORA, N_HEAD * HEAD],
            w(Q_LORA),
            0.0,
        )?;
        add(&mut g, &t("attn_kv.weight"), &[E, HEAD], w(E), 0.0)?;
        add(&mut g, &t("attn_kv_a_norm.weight"), &[HEAD], 0.1, 1.0)?;
        add(&mut g, &t("attn_sinks.weight"), &[N_HEAD], 1.0, 0.0)?;
        let per_group = N_HEAD * HEAD / O_GROUPS;
        add(
            &mut g,
            &t("attn_output_a.weight"),
            &[per_group, O_LORA, O_GROUPS],
            w(per_group),
            0.0,
        )?;
        add(
            &mut g,
            &t("attn_output_b.weight"),
            &[O_LORA * O_GROUPS, E],
            w(O_LORA * O_GROUPS),
            0.0,
        )?;
        for which in ["attn", "ffn"] {
            let n_mix = (2 + HC) * HC;
            add(
                &mut g,
                &t(&format!("hc_{which}_fn.weight")),
                &[E * HC, n_mix],
                0.05,
                0.0,
            )?;
            add(
                &mut g,
                &t(&format!("hc_{which}_base.weight")),
                &[n_mix],
                0.5,
                0.0,
            )?;
            add(
                &mut g,
                &t(&format!("hc_{which}_scale.weight")),
                &[3],
                0.2,
                1.0,
            )?;
        }
        add(&mut g, &t("ffn_gate_inp.weight"), &[E, N_EXPERT], w(E), 0.0)?;
        add(&mut g, &t("exp_probs_b.bias"), &[N_EXPERT], 0.1, 0.0)?;
        add(&mut g, &t("ffn_gate_shexp.weight"), &[E, N_FF], w(E), 0.0)?;
        add(&mut g, &t("ffn_up_shexp.weight"), &[E, N_FF], w(E), 0.0)?;
        add(
            &mut g,
            &t("ffn_down_shexp.weight"),
            &[N_FF, E],
            w(N_FF),
            0.0,
        )?;
        for (name, k, m) in [("gate", E, N_FF), ("up", E, N_FF), ("down", N_FF, E)] {
            let v = rand(k * m * N_EXPERT, w(k), 0.0);
            let q = crate::ggml::quantize(types::Q8_0, &v, k as usize);
            g.add_tensor(
                &t(&format!("ffn_{name}_exps.weight")),
                types::Q8_0,
                &[k, m, N_EXPERT],
                &q,
            )?;
        }
        if is(&KV_SOURCES, il) {
            add(
                &mut g,
                &t("attn_compressor_kv.weight"),
                &[E, HEAD],
                w(E),
                0.0,
            )?;
            if ratio > 1 {
                add(
                    &mut g,
                    &t("attn_compressor_gate.weight"),
                    &[E, HEAD],
                    w(E),
                    0.0,
                )?;
            }
            add(&mut g, &t("attn_compressor_norm.weight"), &[HEAD], 0.1, 1.0)?;
            add(
                &mut g,
                &t("indexer.attn_k.weight"),
                &[HEAD, IDX_DIM],
                w(HEAD),
                0.0,
            )?;
            add(&mut g, &t("indexer.k_norm.weight"), &[IDX_DIM], 0.1, 1.0)?;
        }
        if is(&INDEX_SOURCES, il) {
            add(
                &mut g,
                &t("indexer.proj.weight"),
                &[E, IDX_HEADS],
                w(E),
                0.0,
            )?;
            add(
                &mut g,
                &t("indexer.attn_q_b.weight"),
                &[Q_LORA, IDX_DIM * IDX_HEADS],
                w(Q_LORA),
                0.0,
            )?;
        }
        if is(&ENGRAM_LAYERS, il) {
            let li = ENGRAM_LAYERS
                .iter()
                .position(|&l| l as usize == il)
                .unwrap();
            add(&mut g, &t("engram_q.weight"), &[E, HC], 0.5, 1.0)?;
            add(&mut g, &t("engram_k.weight"), &[E, HC], 0.5, 1.0)?;
            let width = (NGRAM - 1) * ENGRAM_HEADS * ENGRAM_DIM;
            add(
                &mut g,
                &t("engram_wkv.weight"),
                &[width, (HC + 1) * E],
                w(width),
                0.0,
            )?;
            let rows: i64 = PRIMES[li].iter().sum();
            let shape = [ENGRAM_DIM, rows];
            let v = rand(ENGRAM_DIM * rows, 1.0, 0.0);
            let into = match tables {
                Tables::Inline => &mut g,
                Tables::Separate => &mut tables_file,
                Tables::Missing => continue,
            };
            into.add_tensor(&t("engram_embd.weight"), types::F32, &shape, &bytes(&v))?;
        }
    }
    // The hash tables: the token map folds the vocabulary onto 97 ids.
    let map: Vec<u8> = (0..nv as i32)
        .flat_map(|t| ((t * 31 + 7) % 97).to_ne_bytes())
        .collect();
    g.add_tensor("engram_token_map.weight", types::I32, &[nv], &map)?;
    let i64s = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    let n_l = ENGRAM_LAYERS.len() as i64;
    let primes: Vec<i64> = PRIMES.iter().flatten().copied().collect();
    let offs: Vec<i64> = (0..PRIMES.len()).flat_map(offsets).collect();
    let mult: Vec<i64> = (0..n_l * NGRAM)
        .map(|i| 2 * i + 1 + 1_000_003 * (i % 3))
        .collect();
    let cols = (NGRAM - 1) * ENGRAM_HEADS;
    g.add_tensor(
        "engram_primes.weight",
        types::I64,
        &[cols, n_l],
        &i64s(&primes),
    )?;
    g.add_tensor(
        "engram_offsets.weight",
        types::I64,
        &[cols, n_l],
        &i64s(&offs),
    )?;
    g.add_tensor(
        "engram_multipliers.weight",
        types::I64,
        &[NGRAM, n_l],
        &i64s(&mult),
    )?;
    if tables == Tables::Separate {
        let dir = path.parent().unwrap_or_else(|| Path::new(""));
        tables_file.write(dir.join(ENGRAM_FILE))?;
    }
    g.write(path)
}

/// Writes the tiny model, Engram tables included, to `path`.
pub fn write(path: impl AsRef<Path>, seed: u64) -> Result<(), GgufError> {
    write_with(path, seed, Tables::Inline)
}
