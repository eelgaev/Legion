//! qwen4exp (Qwen3.8-Flash-Next): hyper-connections, the per-layer n-gram embedding and the
//! sigmoid DeltaNet gate, on a tiny random model with the real model's layout (tensor names,
//! types, layer pattern, PLE in layer 1, 4 streams) built here, so the graph is tested without
//! the 117 GB model.
//!
//! The non-ignored tests check the decode state: prefill in chunks, token-by-token decode and a
//! batched step must reproduce one whole-prompt pass bit for bit (the PLE conv history and the
//! n-gram token history carry across steps like the DeltaNet state does).
//!
//! The ignored tests compare against llama.cpp (`tools/llama-ref/llama_ref.cpp`, exact mode,
//! CPU), on the tiny model or the real one:
//!
//! ```sh
//! # tiny model: write it, run llama.cpp on it, compare
//! LEGION_TEST_QWEN4EXP_TINY=/tmp/tiny.gguf cargo test --test qwen4exp write_tiny_model -- --ignored
//! llama_ref /tmp/tiny.gguf <ids> /tmp/tiny.exact.f32        # ids: see TINY_PROMPT below
//! LEGION_TEST_QWEN4EXP_TINY=/tmp/tiny.gguf LEGION_TEST_QWEN4EXP_TINY_REF=/tmp/tiny.exact.f32 \
//!   cargo test --release --test qwen4exp tiny_matches_llama_cpp -- --ignored --nocapture
//! # real model (any shard of the split GGUF), references as in tests/qwen35.rs
//! LEGION_TEST_QWEN4EXP=$M LEGION_TEST_QWEN4EXP_REF_DIR=$D \
//!   cargo test --release --test qwen4exp real_ -- --ignored --nocapture --test-threads 1
//! ```

use legion::ggml::{self, Backend, types};
use legion::gguf::{GgufFile, GgufWriter, Value};
use legion::model::core::{
    Attention, Executor, Group, KvType, KvTypes, Logits, Model, MoeKernel, Placement, Pool,
    SeqInput,
};
use legion::model::qwen4::{PleBlocks, PleParams, ple_row_f32};
use legion::tokenizer::Tokenizer;

const ARCH: &str = "qwen4exp";
const N_EMBD: i64 = 128;
const HC: i64 = 4;
const HC_RANK: i64 = 32;
const N_LAYER: usize = 8;
const N_HEAD: i64 = 4;
const N_HEAD_KV: i64 = 2;
const HEAD_DIM: i64 = 64;
const SSM_STATE: i64 = 32;
const SSM_GROUPS: i64 = 2;
const SSM_HEADS: i64 = 6;
const N_EXPERT: i64 = 16;
const N_EXPERT_USED: i64 = 4;
const N_FF: i64 = 64;
const PLE_LAYER: usize = 1;
const PLE_DIM: i64 = 32;
const PLE_HEADS: usize = 4;
/// Token ids of the tiny model's test prompt (any ids of the vocabulary will do). It repeats
/// itself and contains the EOS token, so the n-gram window resets once.
const TINY_PROMPT: &[u32] = &[
    785, 6722, 315, 9625, 374, 12095, 13, 151643, 785, 6722, 315, 9625, 374, 4419, 11, 323, 279,
    6722, 315, 9856, 374, 19846, 13, 785, 6722,
];

/// Deterministic pseudo-random numbers in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..n).map(|_| bias + scale * self.next()).collect()
    }
}

struct Builder {
    g: GgufWriter,
    rng: Rng,
}

impl Builder {
    fn key(&mut self, k: &str, v: Value) {
        self.g.set(&format!("{ARCH}.{k}"), &v).unwrap();
    }

    /// A tensor of shape `shape` (ggml order) with values `bias + scale * U(-1, 1)`.
    fn tensor(
        &mut self,
        name: &str,
        ty: ggml::ffi::ggml_type,
        shape: &[i64],
        scale: f32,
        bias: f32,
    ) {
        let n: i64 = shape.iter().product();
        let v = self.rng.vec(n as usize, scale, bias);
        let bytes = if ty == types::F32 {
            v.iter().flat_map(|x| x.to_ne_bytes()).collect()
        } else {
            ggml::quantize(ty, &v, shape[0] as usize)
        };
        self.g.add_tensor(name, ty, shape, &bytes).unwrap();
    }

    /// A weight `[n_in, n_out]` scaled for unit-variance outputs.
    fn weight(&mut self, name: &str, ty: ggml::ffi::ggml_type, shape: &[i64]) {
        let s = (3.0 / shape[0] as f32).sqrt();
        self.tensor(name, ty, shape, s, 0.0);
    }

    fn norm(&mut self, name: &str, n: i64) {
        self.tensor(name, types::F32, &[n], 0.1, 1.0);
    }
}

/// Writes the tiny qwen4exp model (tokenizer of tests/data/ggml-vocab-qwen35.gguf).
fn write_tiny_model_to(path: &std::path::Path, indexer_top_k: u32) {
    let vocab = GgufFile::open("tests/data/ggml-vocab-qwen35.gguf").unwrap();
    let mut b = Builder {
        g: GgufWriter::new(),
        rng: Rng(0x9e37_79b9_7f4a_7c15),
    };
    b.g.set("general.architecture", &Value::String(ARCH.into()))
        .unwrap();
    b.g.set("general.name", &Value::String("tiny qwen4exp".into()))
        .unwrap();
    for (k, v) in vocab.metadata() {
        if k.starts_with("tokenizer.") {
            b.g.set(&k, &v).unwrap();
        }
    }
    let n_vocab = vocab
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array().map(|a| a.len()))
        .unwrap() as i64;
    let eos = vocab.get_u64("tokenizer.ggml.eos_token_id").unwrap() as u32;
    let u = Value::U32;
    b.key("block_count", u(N_LAYER as u32));
    b.key("context_length", u(4096));
    b.key("embedding_length", u(N_EMBD as u32));
    b.key("attention.head_count", u(N_HEAD as u32));
    b.key("attention.head_count_kv", u(N_HEAD_KV as u32));
    b.key("attention.key_length", u(HEAD_DIM as u32));
    b.key("attention.value_length", u(HEAD_DIM as u32));
    b.key("rope.dimension_count", u(32));
    b.key(
        "rope.dimension_sections",
        Value::Array([6, 5, 5, 0].into_iter().map(Value::I32).collect()),
    );
    b.key("rope.freq_base", Value::F32(10_000_000.0));
    b.key("attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    b.key("expert_count", u(N_EXPERT as u32));
    b.key("expert_used_count", u(N_EXPERT_USED as u32));
    b.key("expert_feed_forward_length", u(N_FF as u32));
    b.key("expert_shared_feed_forward_length", u(N_FF as u32));
    b.key("ssm.conv_kernel", u(4));
    b.key("ssm.state_size", u(SSM_STATE as u32));
    b.key("ssm.group_count", u(SSM_GROUPS as u32));
    b.key("ssm.time_step_rank", u(SSM_HEADS as u32));
    b.key("ssm.inner_size", u((SSM_STATE * SSM_HEADS) as u32));
    b.key("full_attention_interval", u(4));
    b.key("hyper_connection.count", u(HC as u32));
    b.key("hyper_connection.low_rank", u(HC_RANK as u32));
    b.key("attention.indexer.head_count", u(2));
    b.key("attention.indexer.key_length", u(32));
    b.key("attention.indexer.top_k", u(indexer_top_k));
    b.key(
        "attention.compress_ratios",
        Value::Array(
            (0..N_LAYER)
                .map(|il| Value::I32(if (il + 1) % 4 == 0 { 4 } else { 0 }))
                .collect(),
        ),
    );
    // PLE: the real model's hash constants, small head tables.
    let sizes: Vec<u64> = (0..PLE_HEADS as u64).map(|h| 1009 + 2 * h).collect();
    let offsets: Vec<u64> = sizes
        .iter()
        .scan(0, |acc, &s| {
            let o = *acc;
            *acc += s;
            Some(o)
        })
        .collect();
    let ple_rows = (offsets[PLE_HEADS - 1] + sizes[PLE_HEADS - 1]) as i64;
    b.key(
        "ple.layers",
        Value::Array(vec![Value::I32(PLE_LAYER as i32)]),
    );
    b.key("ple.ngram_size", u(3));
    b.key("ple.heads_per_ngram", u(2));
    b.key("ple.conv_kernel", u(4));
    b.key("ple.eos_token_id", u(eos));
    b.key("embedding_length_per_layer_input", u(PLE_DIM as u32));
    b.key(
        "ple.layer_multipliers",
        Value::Array(
            [23703573157769u64, 20109073645365, 8052911324071]
                .into_iter()
                .map(Value::U64)
                .collect(),
        ),
    );
    b.key(
        "ple.head_offsets",
        Value::Array(offsets.into_iter().map(Value::U64).collect()),
    );
    b.key(
        "ple.head_vocab_sizes",
        Value::Array(sizes.into_iter().map(Value::U64).collect()),
    );

    let e = N_EMBD;
    let hc_dim = HC * e;
    b.tensor("token_embd.weight", types::Q8_0, &[e, n_vocab], 1.0, 0.0);
    b.weight("output.weight", types::Q8_0, &[e, n_vocab]);
    let hc_block = |b: &mut Builder, prefix: &str, inject: bool| {
        b.norm(&format!("{prefix}_norm.weight"), hc_dim);
        b.weight(
            &format!("{prefix}_down.weight"),
            types::Q8_0,
            &[hc_dim, HC_RANK],
        );
        b.weight(
            &format!("{prefix}_up.weight"),
            types::Q8_0,
            &[HC_RANK, hc_dim],
        );
        if inject {
            b.tensor(
                &format!("{prefix}_inject.weight"),
                types::F32,
                &[hc_dim, HC],
                0.05,
                0.0,
            );
        }
    };
    hc_block(&mut b, "output_hc", false);
    b.tensor(
        "per_layer_token_embd.weight",
        types::IQ4_NL,
        &[PLE_DIM, ple_rows],
        1.0,
        0.0,
    );
    let (ks, vs) = (SSM_STATE * SSM_GROUPS, SSM_STATE * SSM_HEADS);
    for il in 0..N_LAYER {
        let t = |s: &str| format!("blk.{il}.{s}");
        hc_block(&mut b, &t("hc_attn"), true);
        hc_block(&mut b, &t("hc_ffn"), true);
        if (il + 1) % 4 != 0 {
            b.weight(&t("attn_qkv.weight"), types::Q8_0, &[e, 2 * ks + vs]);
            b.weight(&t("attn_gate.weight"), types::Q8_0, &[e, vs]);
            b.tensor(
                &t("ssm_conv1d.weight"),
                types::F32,
                &[4, 2 * ks + vs],
                0.5,
                0.0,
            );
            b.tensor(&t("ssm_dt.bias"), types::F32, &[SSM_HEADS], 0.5, 0.0);
            b.tensor(&t("ssm_a"), types::F32, &[SSM_HEADS], 0.5, -1.0);
            b.weight(&t("ssm_alpha.weight"), types::F32, &[e, SSM_HEADS]);
            b.weight(&t("ssm_beta.weight"), types::F32, &[e, SSM_HEADS]);
            b.norm(&t("ssm_norm.weight"), SSM_STATE);
            b.weight(&t("ssm_out.weight"), types::Q8_0, &[vs, e]);
        } else {
            b.weight(
                &t("attn_q.weight"),
                types::Q8_0,
                &[e, 2 * N_HEAD * HEAD_DIM],
            );
            b.weight(&t("attn_k.weight"), types::Q8_0, &[e, N_HEAD_KV * HEAD_DIM]);
            b.weight(&t("attn_v.weight"), types::Q8_0, &[e, N_HEAD_KV * HEAD_DIM]);
            b.weight(
                &t("attn_output.weight"),
                types::Q8_0,
                &[N_HEAD * HEAD_DIM, e],
            );
            b.norm(&t("attn_q_norm.weight"), HEAD_DIM);
            b.norm(&t("attn_k_norm.weight"), HEAD_DIM);
            b.norm(&t("indexer.k_norm.weight"), 32);
            b.norm(&t("indexer.q_norm.weight"), 32);
            b.weight(&t("indexer.k_proj.weight"), types::BF16, &[e, 32]);
            b.weight(&t("indexer.q_proj.weight"), types::BF16, &[e, 2 * 32]);
        }
        if il == PLE_LAYER {
            let pe = PLE_DIM * PLE_HEADS as i64;
            b.tensor(&t("ple_conv1d.weight"), types::F32, &[4, hc_dim], 0.5, 0.0);
            b.weight(&t("ple_key.weight"), types::Q8_0, &[pe, hc_dim]);
            b.weight(&t("ple_value.weight"), types::Q8_0, &[pe, e]);
            b.norm(&t("ple_norm_conv.weight"), hc_dim);
            b.norm(&t("ple_norm_key.weight"), hc_dim);
            b.norm(&t("ple_norm_query.weight"), hc_dim);
        }
        b.weight(&t("ffn_gate_inp.weight"), types::F32, &[e, N_EXPERT]);
        b.weight(&t("ffn_gate_inp_shexp.weight"), types::F32, &[e]);
        b.weight(&t("ffn_up_exps.weight"), types::Q8_0, &[e, N_FF, N_EXPERT]);
        b.weight(
            &t("ffn_gate_exps.weight"),
            types::Q8_0,
            &[e, N_FF, N_EXPERT],
        );
        b.weight(
            &t("ffn_down_exps.weight"),
            types::Q5_1,
            &[N_FF, e, N_EXPERT],
        );
        b.weight(&t("ffn_up_shexp.weight"), types::Q8_0, &[e, N_FF]);
        b.weight(&t("ffn_gate_shexp.weight"), types::Q8_0, &[e, N_FF]);
        b.weight(&t("ffn_down_shexp.weight"), types::Q8_0, &[N_FF, e]);
    }
    b.g.write(path).unwrap();
}

/// Sparse-attention budget of the tiny model: 2 pools of 4 (plus the token's own pool), so
/// past position 11 attention drops keys.
const TINY_TOP_K: u32 = 8;

fn tiny_with(top_k: u32) -> (tempfile::TempDir, GgufFile) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-qwen4exp.gguf");
    write_tiny_model_to(&path, top_k);
    let gguf = GgufFile::open(&path).unwrap();
    (dir, gguf)
}

fn tiny() -> (tempfile::TempDir, GgufFile) {
    tiny_with(TINY_TOP_K)
}

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn reads_the_hyper_connection_and_ple_metadata() {
    let (_dir, gguf) = tiny();
    let backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let hp = &model.hp;
    assert!(hp.moe);
    assert_eq!(hp.n_layer, N_LAYER);
    let q4 = legion::model::qwen4::params(hp).unwrap();
    assert_eq!(q4.hc.n, HC);
    assert_eq!(q4.hc.low_rank, HC_RANK);
    let ple = q4.ple.as_ref().unwrap();
    assert_eq!(ple.layer, PLE_LAYER);
    assert_eq!(ple.n_heads(), PLE_HEADS);
    assert_eq!(ple.history(), 9);
    assert_eq!(ple.head_dim, PLE_DIM);
    assert_eq!(ple.eos, 151643);
    assert!(hp.is_recurrent(0) && hp.is_recurrent(1) && !hp.is_recurrent(3));
}

/// The n-gram hash, against values worked out by hand from llama.cpp's formula.
#[test]
fn ngram_rows_follow_llama_cpp_hash() {
    let p = PleParams {
        layer: 1,
        ngram: 3,
        heads_per_ngram: 2,
        conv_kernel: 4,
        eos: 7,
        head_dim: 32,
        multipliers: vec![3, 5, 11],
        offsets: vec![0, 100, 200, 300],
        vocab_sizes: vec![97, 89, 83, 79],
        blocks: None,
        fp8_scale: None,
    };
    const NO: u32 = u32::MAX;
    let rows = |tok, prev: [u32; 2]| {
        let mut v = Vec::new();
        p.rows(tok, &prev, &mut v);
        v
    };
    // Token 10 after 20 after 30: bigram 10*3 ^ 20*5, trigram that ^ 30*11.
    let bi = (10u64 * 3) ^ (20 * 5);
    let tri = bi ^ (30 * 11);
    let want = vec![
        (bi % 97) as i32,
        (bi % 89 + 100) as i32,
        (tri % 83 + 200) as i32,
        (tri % 79 + 300) as i32,
    ];
    assert_eq!(rows(10, [30, 20]), want);
    // A missing or EOS predecessor cuts the window: everything at or before it reads as EOS.
    let cut = |t1: u64, t2: u64| {
        let bi = (10u64 * 3) ^ (t1 * 5);
        let tri = bi ^ (t2 * 11);
        vec![
            (bi % 97) as i32,
            (bi % 89 + 100) as i32,
            (tri % 83 + 200) as i32,
            (tri % 79 + 300) as i32,
        ]
    };
    assert_eq!(rows(10, [NO, NO]), cut(7, 7));
    assert_eq!(rows(10, [NO, 20]), cut(20, 7));
    assert_eq!(rows(10, [30, 7]), cut(7, 7));
    assert_eq!(rows(10, [7, 20]), cut(20, 7));
    // The token itself being EOS does not cut its own context.
    let bi = (7u64 * 3) ^ (20 * 5);
    assert_eq!(rows(7, [30, 20])[0], (bi % 97) as i32);
    // Wrapping 64-bit products, as in C.
    let big = PleParams {
        multipliers: vec![u64::MAX / 3, u64::MAX / 5, u64::MAX / 7],
        ..p.clone()
    };
    let mut v = Vec::new();
    big.rows(248_000, &[248_001, 248_002], &mut v);
    let bi = 248_000u64.wrapping_mul(u64::MAX / 3) ^ 248_002u64.wrapping_mul(u64::MAX / 5);
    assert_eq!(v[0], (bi % 97) as i32);
}

/// A table stored as blocks out of row order (a safetensors checkpoint's shards in file order)
/// maps every hashed row to where it is stored; without the order the rows are unchanged.
#[test]
fn ngram_rows_follow_the_block_order() {
    let p = PleParams {
        layer: 1,
        ngram: 3,
        heads_per_ngram: 2,
        conv_kernel: 4,
        eos: 7,
        head_dim: 32,
        multipliers: vec![3, 5, 11],
        offsets: vec![0, 100, 200, 300],
        vocab_sizes: vec![97, 89, 83, 79],
        blocks: None,
        fp8_scale: None,
    };
    let blocks = PleBlocks {
        rows: 50,
        position: vec![7, 6, 5, 4, 3, 2, 1, 0],
    };
    assert_eq!(blocks.physical(0), 350);
    assert_eq!(blocks.physical(49), 399);
    assert_eq!(blocks.physical(50), 300);
    assert_eq!(blocks.physical(399), 49);
    let moved = PleParams {
        blocks: Some(blocks.clone()),
        ..p.clone()
    };
    for (tok, prev) in [(10u32, [30u32, 20]), (5, [u32::MAX, 9]), (248_000, [1, 2])] {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        p.rows(tok, &prev, &mut a);
        moved.rows(tok, &prev, &mut b);
        let want: Vec<i32> = a
            .iter()
            .map(|&r| blocks.physical(r as u64) as i32)
            .collect();
        assert_eq!(b, want);
    }
}

/// FP8 table rows are their E4M3 codes times the table's scale; other types go through ggml.
#[test]
fn ngram_fp8_rows_are_scaled_e4m3() {
    let codes = [0x38u8, 0xb8, 0x00, 0x7e, 0x01, 0x40];
    let got = ple_row_f32(types::I8, Some(0.5), &codes, codes.len());
    let want: Vec<f32> = codes
        .iter()
        .map(|&b| legion::hf::nvfp4::e4m3_to_f32(b) * 0.5)
        .collect();
    assert_eq!(got, want);
    assert_eq!(got[..2], [0.5, -0.5]);
    // F16 1.5 and -2.0.
    let f: Vec<u8> = [0x3e00u16, 0xc000]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(ple_row_f32(types::F16, None, &f, 2), vec![1.5, -2.0]);
}

/// Sparse attention is the same as dense while a token sees at most `top_k` keys (here up to
/// position 11: two complete pools of 4 and its own), and drops keys after that.
#[test]
fn sparse_attention_equals_dense_until_the_budget() {
    let mut backend = cpu();
    backend.set_use_ref(true);
    let mut logits = Vec::new();
    for top_k in [TINY_TOP_K, 4096] {
        let (_dir, gguf) = tiny_with(top_k);
        let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        logits.push(model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap());
    }
    let n_vocab = logits[0].len() / TINY_PROMPT.len();
    for (p, (a, b)) in logits[0]
        .chunks(n_vocab)
        .zip(logits[1].chunks(n_vocab))
        .enumerate()
    {
        let d = max_diff(a, b);
        if p < 11 {
            assert_eq!(d, 0.0, "position {p}");
        } else {
            assert!(d > 1e-3, "position {p}: sparse attention changed nothing");
        }
    }
}

/// The paged sparse attention op (`Attention::Paged`: each token reads only its selected keys)
/// against the gather path (dense attention over the span plus the selection as a mask) on the
/// CPU, with keys dropped (budget of 2 pools) and without; then chunked prefill and decode under
/// the paged op against its own whole-prompt pass, exactly.
#[test]
#[cfg_attr(
    target_arch = "aarch64",
    ignore = "ggml's CPU F16 dot products accumulate in F16 on aarch64, outside this tolerance, see #906"
)]
fn paged_sparse_attention_matches_gather() {
    let mut backend = cpu();
    backend.set_use_ref(true);
    for top_k in [TINY_TOP_K, 4096] {
        let (_dir, gguf) = tiny_with(top_k);
        let mut model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        let n_vocab = model.hp.n_vocab as usize;
        let gather = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
        model.set_attention(Attention::Paged);
        let paged = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
        let scale = gather.iter().fold(0f32, |m, x| m.max(x.abs()));
        // The op sums in the gather path's order, so most positions come out bit-identical;
        // its exp is Rust's, not ggml's, and a 1-ulp difference that flips one softmax
        // weight's F16 rounding moves a later position's logits by ~5e-3 here (Q8_0
        // activations amplify it). A wrong selection moves them by far more.
        let d = max_diff(&gather, &paged);
        assert!(
            d < 1e-2 * scale.max(1.0),
            "top_k {top_k}: max |paged - gather| {d} (logits up to {scale})"
        );

        let mut state = model.new_state(backend.buffer_type(), 64).unwrap();
        let mut alloc = None;
        let mut got = Vec::new();
        let mut at = 0;
        for len in [7, 5].into_iter().chain(std::iter::repeat(1)) {
            if at == TINY_PROMPT.len() {
                break;
            }
            let chunk = &TINY_PROMPT[at..(at + len).min(TINY_PROMPT.len())];
            got.extend(
                model
                    .forward(&mut backend, &mut alloc, &mut state, chunk, true)
                    .unwrap(),
            );
            at += chunk.len();
        }
        for (i, (a, b)) in got.chunks(n_vocab).zip(paged.chunks(n_vocab)).enumerate() {
            assert_eq!(max_diff(a, b), 0.0, "top_k {top_k}: position {i}");
        }
    }
}

/// Quantized K/V caches (#263): the paged sparse attention op (on the CPU, its reference)
/// reads q8_0 / q4_0 as the gather path does, over a prompt in two chunks and then decode
/// steps, in blocks out of order; and the cache is quantized (the logits move from F16's).
#[test]
#[cfg_attr(
    target_arch = "aarch64",
    ignore = "ggml's CPU F16 dot products accumulate in F16 on aarch64, outside this tolerance, see #906"
)]
fn quantized_cache_paged_sparse_attention_matches_gather() {
    let (_dir, gguf) = tiny();
    let run = |types: KvTypes, attention: Attention| -> Vec<f32> {
        let mut backend = cpu();
        backend.set_use_ref(true);
        let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        let n = model.hp.n_layer;
        let mut exec =
            Executor::new_staged_ring(model, vec![(backend, 0..n)], 4, 1, 0, None, types).unwrap();
        exec.set_attention(attention);
        let blocks = [2u32, 0, 3, 1];
        let (mut rows, mut at) = (Vec::new(), 0);
        for len in [7, 5].into_iter().chain(std::iter::repeat(1)) {
            if at == TINY_PROMPT.len() {
                break;
            }
            let chunk = &TINY_PROMPT[at..(at + len).min(TINY_PROMPT.len())];
            let out = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: chunk,
                        n_past: at,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::All,
                        media: None,
                    }],
                }])
                .unwrap();
            rows.extend(out.rows.into_iter().flatten());
            at += chunk.len();
        }
        rows
    };
    let f16 = run(KvTypes::default(), Attention::Gather);
    for (k, v) in [(KvType::Q8_0, KvType::Q8_0), (KvType::Q4_0, KvType::Q4_0)] {
        let types = KvTypes { k, v };
        let gather = run(types, Attention::Gather);
        let paged = run(types, Attention::Paged);
        let scale = gather.iter().fold(0f32, |m, x| m.max(x.abs()));
        // As in `paged_sparse_attention_matches_gather`: the same sums up to rounding.
        let d = max_diff(&gather, &paged);
        assert!(
            d < 1e-2 * scale.max(1.0),
            "{k:?} / {v:?}: max |paged - gather| {d} (logits up to {scale})"
        );
        assert!(
            max_diff(&gather, &f16) > 0.0,
            "{k:?} / {v:?}: a quantized cache"
        );
    }
}

/// The fused expert op (forced: on the CPU its F32 reference) against ggml's `mul_mat_id`
/// path, which rounds the activations to Q8 for the quantized experts: layer 0's MoE output
/// (the same input on both paths) agrees to that rounding. Then chunked prefill with the op
/// against its own whole-prompt pass, exactly.
#[test]
fn fused_moe_matches_graph_path() {
    let (_dir, gguf) = tiny();
    let mut backend = cpu();
    backend.set_use_ref(true);
    let mut model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let moe_out = |model: &Model, backend: &mut Backend| {
        let (_, taps) = model.prompt_trace(backend, TINY_PROMPT).unwrap();
        taps.into_iter()
            .find(|t| t.name == "ffn_moe_out-0")
            .expect("ffn_moe_out-0")
            .data
    };
    let graph = moe_out(&model, &mut backend);
    model.set_moe_kernel(MoeKernel::Fused);
    let fused_out = moe_out(&model, &mut backend);
    let scale = graph.iter().fold(0f32, |m, x| m.max(x.abs()));
    let d = max_diff(&graph, &fused_out);
    assert!(d > 0.0, "the fused op did not run");
    assert!(d < 2e-2 * scale, "max |fused - graph| {d} of {scale}");
    let fused = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();

    let mut state = model.new_state(backend.buffer_type(), 64).unwrap();
    let mut alloc = None;
    let mut got = Vec::new();
    for chunk in TINY_PROMPT.chunks(9) {
        got.extend(
            model
                .forward(&mut backend, &mut alloc, &mut state, chunk, true)
                .unwrap(),
        );
    }
    for (i, (a, b)) in got.chunks(n_vocab).zip(fused.chunks(n_vocab)).enumerate() {
        assert_eq!(max_diff(a, b), 0.0, "position {i}");
    }
}

/// Prefill in chunks, then decode token by token, against one whole-prompt pass: every
/// position's logits must agree exactly (CPU reference kernels: the matmul kernels then do not
/// depend on the batch size).
#[test]
fn chunked_prefill_and_decode_reproduce_the_full_pass() {
    let (_dir, gguf) = tiny();
    let mut backend = cpu();
    backend.set_use_ref(true);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let full = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
    assert!(full.iter().all(|x| x.is_finite()));

    let mut state = model.new_state(backend.buffer_type(), 64).unwrap();
    let mut alloc = None;
    let mut got = Vec::new();
    // Chunks of 7, 5, then single tokens: the conv histories (9 PLE positions, 3 DeltaNet
    // ones) and the n-gram window all straddle chunk boundaries.
    let mut at = 0;
    for len in [7, 5].into_iter().chain(std::iter::repeat(1)) {
        if at == TINY_PROMPT.len() {
            break;
        }
        let chunk = &TINY_PROMPT[at..(at + len).min(TINY_PROMPT.len())];
        got.extend(
            model
                .forward(&mut backend, &mut alloc, &mut state, chunk, true)
                .unwrap(),
        );
        at += chunk.len();
    }
    assert_eq!(got.len(), full.len());
    for (i, (a, b)) in got.chunks(n_vocab).zip(full.chunks(n_vocab)).enumerate() {
        assert_eq!(max_diff(a, b), 0.0, "position {i}");
    }
}

/// The n-gram table read through the lookup-table store in every tier (locked in RAM on a
/// NUMA node, mapped, direct reads with and without a row cache): logits bit-identical to
/// reading the weights' own memory.
#[test]
fn ngram_table_tiers_change_nothing() {
    use legion::residency::lookup_table::{DEFAULT_AIO_MAX, LookupTable, TableIo, TableOptions};
    let (_dir, gguf) = tiny();
    let mut backend = cpu();
    backend.set_use_ref(true);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let want = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
    let topo = legion::hw::numa::Topology::detect().ok();
    let file = Model::ngram_table_info(&gguf).expect("the tiny model has an n-gram table");
    // Direct reads through the I/O threads only, and small batches as AIO.
    for (io, cache, aio_max) in [
        (TableIo::Ram, None, 0),
        (TableIo::Mmap, None, 0),
        (TableIo::Direct, Some(0), 0),
        (TableIo::Direct, Some(1 << 20), 0),
        (TableIo::Direct, Some(0), DEFAULT_AIO_MAX),
        (TableIo::Direct, Some(1 << 20), DEFAULT_AIO_MAX),
    ] {
        let mut model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        let opts = TableOptions {
            io,
            ram_cache_bytes: cache,
            inflight: 4,
            aio_max,
            keep_awake_ms: 0,
            node: topo.as_ref().map(|t| t.nodes[0].id),
            unified: false,
        };
        model.set_ngram_table(LookupTable::open(file.clone(), &opts, topo.as_ref()).unwrap());
        for round in 0..2 {
            let got = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
            assert_eq!(
                max_diff(&got, &want),
                0.0,
                "{io:?} cache {cache:?} aio {aio_max} round {round}"
            );
        }
        assert!(model.ngram_table().unwrap().stats().rows > 0 || io != TableIo::Direct);
    }
}

/// Two sequences in one batch step (slots 0 and 1), against each run alone; then the state of
/// one copied to another slot continues identically (the token history moves with it).
#[test]
fn batched_steps_and_state_copies_match_single_runs() {
    let (_dir, gguf) = tiny();
    let mut backend = cpu();
    backend.set_use_ref(true);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    // Long enough for sparse attention to drop keys (past position 11).
    let a = &TINY_PROMPT[..14];
    let b = &TINY_PROMPT[11..24];
    let alone_a = model.prompt_logits(&mut backend, a).unwrap();
    let alone_b = model.prompt_logits(&mut backend, b).unwrap();
    let whole = model
        .prompt_logits(&mut backend, &TINY_PROMPT[..15])
        .unwrap();

    let mut exec = Executor::new(model, backend, 8, 3, 0).unwrap();
    // Cells are not positions: the sequences sit in blocks out of order.
    let blocks_a: Vec<u32> = vec![6, 1, 2, 3];
    let blocks_b: Vec<u32> = vec![0, 4, 5, 7];
    let seq = |tokens, blocks| SeqInput {
        tokens,
        n_past: 0,
        blocks,
        ring: 0,
        logits: Logits::Last,
        media: None,
    };
    let out = exec
        .step(&[
            Group {
                slot: 0,
                verify: false,
                seqs: vec![seq(a, &blocks_a)],
            },
            Group {
                slot: 1,
                verify: false,
                seqs: vec![seq(b, &blocks_b)],
            },
        ])
        .unwrap();
    assert_eq!(out.rows[0].len(), n_vocab);
    assert_eq!(max_diff(&out.rows[0], &alone_a), 0.0, "sequence a");
    assert_eq!(max_diff(&out.rows[1], &alone_b), 0.0, "sequence b");

    // Continue a from a copy of its state in slot 2 (its KV blocks stay the same).
    exec.copy_states(&[(Pool::Seq, 0, Pool::Seq, 2)]).unwrap();
    exec.clear_slot(0);
    let next = [TINY_PROMPT[14]];
    let step = exec
        .step(&[Group {
            slot: 2,
            verify: false,
            seqs: vec![SeqInput {
                tokens: &next,
                n_past: a.len(),
                blocks: &blocks_a,
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
        }])
        .unwrap();
    assert_eq!(
        max_diff(&step.rows[0], &whole),
        0.0,
        "continued from a copied state"
    );
}

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

/// Writes the tiny model to `LEGION_TEST_QWEN4EXP_TINY` (for llama_ref), with
/// `LEGION_TEST_QWEN4EXP_TINY_TOP_K` as the sparse-attention budget (default `TINY_TOP_K`;
/// 4096 selects every key).
#[test]
#[ignore = "writes a file for llama.cpp"]
fn write_tiny_model() {
    let top_k = std::env::var("LEGION_TEST_QWEN4EXP_TINY_TOP_K")
        .map(|v| v.parse().unwrap())
        .unwrap_or(TINY_TOP_K);
    write_tiny_model_to(env("LEGION_TEST_QWEN4EXP_TINY").as_ref(), top_k);
    let ids: Vec<String> = TINY_PROMPT.iter().map(u32::to_string).collect();
    println!("ids: {}", ids.join(","));
}

/// All-position logits of the tiny model against llama.cpp's (exact mode).
#[test]
#[ignore = "needs llama.cpp reference logits"]
fn tiny_matches_llama_cpp() {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN4EXP_TINY")).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let ours = model.prompt_logits_all(&mut backend, TINY_PROMPT).unwrap();
    let theirs = read_f32(&env("LEGION_TEST_QWEN4EXP_TINY_REF"));
    assert!(ours.iter().chain(&theirs).all(|x| x.is_finite()));
    let d = max_diff(&ours, &theirs);
    println!("tiny qwen4exp vs llama.cpp: max |diff| {d:.3e}");
    assert!(d <= 1e-4, "logits differ by {d}");
    // llama_ref's REF_GEN tokens, if it wrote them.
    let gen_path = format!("{}.gen", env("LEGION_TEST_QWEN4EXP_TINY_REF"));
    if let Some(want) = read_ids(&gen_path) {
        let got = greedy(&model, &mut backend, TINY_PROMPT, want.len());
        assert_eq!(got, want, "greedy tokens");
        println!("greedy: {} tokens identical", got.len());
    }
}

fn read_ids(path: &str) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(text.trim().split(',').map(|t| t.parse().unwrap()).collect())
}

/// `n` greedy tokens after `prompt`, decoded one at a time.
fn greedy(model: &Model, backend: &mut Backend, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut state = model
        .new_state(backend.buffer_type(), prompt.len() + n)
        .unwrap();
    let mut alloc = None;
    let mut logits = model
        .forward(backend, &mut alloc, &mut state, prompt, false)
        .unwrap();
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        let next = argmax(&logits) as u32;
        ids.push(next);
        if ids.len() < n {
            logits = model
                .forward(backend, &mut alloc, &mut state, &[next], false)
                .unwrap();
        }
    }
    ids
}

/// Writes `<ref dir>/<name>.ids` for the real model's prompts (see tests/qwen35.rs).
#[test]
#[ignore = "needs the real model"]
fn real_write_prompt_ids() {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN4EXP")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let dir = std::path::PathBuf::from(env("LEGION_TEST_QWEN4EXP_REF_DIR"));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in prompts() {
        let ids = tok.encode(&text, true);
        let list: Vec<String> = ids.iter().map(u32::to_string).collect();
        std::fs::write(dir.join(format!("{name}.ids")), list.join(",")).unwrap();
        println!("{name}: {} tokens", ids.len());
    }
}

/// The prompts, plus `long`: ~3000 tokens of numbered facts, past the 2048-key budget of
/// the sparse attention, so the indexer picks keys.
fn prompts() -> Vec<(&'static str, String)> {
    let mut v: Vec<(&str, String)> = PROMPTS.iter().map(|(n, t)| (*n, t.to_string())).collect();
    let long: String = (1..=100)
        .map(|i| {
            format!(
                "Fact {i}: {i} times {} is {}, and {i} squared is {}. ",
                i % 7 + 3,
                i * (i % 7 + 3),
                i * i
            )
        })
        .collect();
    v.push(("long", long + "Fact 101:"));
    v
}

const PROMPTS: &[(&str, &str)] = &[
    ("france", "The capital of France is"),
    (
        "apollo",
        "In July 1969, Neil Armstrong and Buzz Aldrin landed on the Moon in the lunar module \
         Eagle while Michael Collins stayed in orbit aboard Columbia. Apollo 11 fulfilled the goal \
         President Kennedy had set eight years earlier, and the next mission",
    ),
];

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
    x.iter().map(|&v| v as f64 - lse).collect()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
}

/// Agreement of all-position logits: max |diff|, top-1 matches, mean and max KL(theirs || ours).
fn agreement(ours: &[f32], theirs: &[f32], n_vocab: usize) -> (f32, usize, usize, f64, f64) {
    assert_eq!(ours.len(), theirs.len(), "logit count");
    let n = ours.len() / n_vocab;
    let (mut top1, mut kl_mean, mut kl_max) = (0, 0.0, 0f64);
    for (o, t) in ours.chunks_exact(n_vocab).zip(theirs.chunks_exact(n_vocab)) {
        top1 += (argmax(o) == argmax(t)) as usize;
        let (lo, lt) = (log_softmax(o), log_softmax(t));
        let kl: f64 = lt.iter().zip(&lo).map(|(t, o)| t.exp() * (t - o)).sum();
        kl_mean += kl / n as f64;
        kl_max = kl_max.max(kl);
    }
    (max_diff(ours, theirs), top1, n, kl_mean, kl_max)
}

/// The real model's all-position prompt logits against llama.cpp's: `<name>.exact.f32` must
/// agree to rounding noise (same CPU kernels), `<name>.default.f32` (llama.cpp's defaults, or
/// its CUDA build) statistically. With `LEGION_TEST_QWEN4EXP_CUDA=1` the model runs on the GPU
/// (weights placed as the engine places them) and every reference is checked statistically.
#[test]
#[ignore = "needs the real model and llama.cpp reference logits"]
fn real_prompt_logits_match_llama_cpp() {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN4EXP")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let cuda = std::env::var("LEGION_TEST_QWEN4EXP_CUDA").is_ok_and(|v| v == "1");
    let mut backend = if cuda {
        use legion::ggml::{DeviceKind, device_by_kind};
        device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU")
            .init()
            .unwrap()
    } else {
        Backend::cpu().unwrap()
    };
    backend.set_threads(
        std::env::var("LEGION_TEST_THREADS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(8),
    );
    let t = std::time::Instant::now();
    let host = Backend::cpu().unwrap().buffer_type();
    let model = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
        Placement::Host => host,
        Placement::Device => backend.buffer_type(),
    })
    .unwrap();
    println!("loaded in {:.1} s", t.elapsed().as_secs_f64());
    let n_vocab = model.hp.n_vocab as usize;
    let dir = std::path::PathBuf::from(env("LEGION_TEST_QWEN4EXP_REF_DIR"));
    let mut checked = 0;
    for (name, text) in prompts() {
        let ids = tok.encode(&text, true);
        let t = std::time::Instant::now();
        let logits = model.prompt_logits_all(&mut backend, &ids).unwrap();
        println!(
            "{name}: {} tokens in {:.2} s",
            ids.len(),
            t.elapsed().as_secs_f64()
        );
        for mode in ["exact", "default"] {
            let path = dir.join(format!("{name}.{mode}.f32"));
            let Ok(theirs) = std::fs::read(&path) else {
                continue;
            };
            let theirs: Vec<f32> = theirs
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes(*c))
                .collect();
            // A long prompt's reference may hold only the last positions (REF_LAST).
            let ours = &logits[logits.len() - theirs.len()..];
            let (d, top1, n, kl_mean, kl_max) = agreement(ours, &theirs, n_vocab);
            println!(
                "  {mode:7}: max |diff| {d:.2e}, top-1 {top1}/{n}, KL mean {kl_mean:.2e} max {kl_max:.2e}"
            );
            if mode == "exact" && !cuda {
                assert!(d <= 1e-4, "{name}: logits differ by {d}");
            } else {
                assert!(kl_mean < 0.015, "{name}: mean KL {kl_mean}");
                assert!(kl_max < 0.1, "{name}: max KL {kl_max}");
                assert!(top1 * 10 >= n * 9, "{name}: top-1 {top1}/{n}");
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "no reference logits in {}", dir.display());
}

/// Greedy decoding of the real model against llama.cpp's (`llama_ref` with `REF_GEN=256`,
/// exact mode, written next to the logits as `<name>.exact.f32.gen`), on the CPU.
#[test]
#[ignore = "needs the real model and llama.cpp reference tokens"]
fn real_greedy_matches_llama_cpp() {
    let gguf = GgufFile::open(env("LEGION_TEST_QWEN4EXP")).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(
        std::env::var("LEGION_TEST_THREADS")
            .map(|v| v.parse().unwrap())
            .unwrap_or(8),
    );
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let dir = env("LEGION_TEST_QWEN4EXP_REF_DIR");
    let mut checked = 0;
    for (name, text) in prompts() {
        let Some(want) = read_ids(&format!("{dir}/{name}.exact.f32.gen")) else {
            continue;
        };
        let ids = tok.encode(&text, true);
        let t = std::time::Instant::now();
        let got = greedy(&model, &mut backend, &ids, want.len());
        let same = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        println!(
            "{name}: {same}/{} greedy tokens identical ({:.1} s): {:?}",
            want.len(),
            t.elapsed().as_secs_f64(),
            tok.decode(&got)
        );
        assert_eq!(got, want, "{name}: greedy tokens differ from token {same}");
        checked += 1;
    }
    assert!(checked > 0, "no reference tokens in {dir}");
}

/// A greedy raw-text request's result: the prompt length, its output tokens with their top
/// log probabilities, and how it finished.
struct Run {
    prompt_tokens: u32,
    tokens: Vec<legion::engine::TokenLogprob>,
    finished: String,
}

async fn engine_run(engine: &legion::engine::lm::LmEngine, prompt: &str, max_tokens: u32) -> Run {
    use futures::StreamExt;
    use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
    let req = GenerateRequest {
        model: engine.models()[0].id.clone(),
        prompt: Prompt::Text(prompt.into()),
        sampling: SamplingParams {
            temperature: 0.0,
            logprobs: Some(3),
            ..Default::default()
        },
        max_tokens,
        stop: vec![],
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        ignore_eos: false,
        min_tokens: 0,
        cancel: tokio_util::sync::CancellationToken::new(),
        scheduling: Default::default(),
    };
    let mut stream = engine.generate(req).unwrap();
    let mut run = Run {
        prompt_tokens: 0,
        tokens: Vec::new(),
        finished: String::new(),
    };
    while let Some(ev) = stream.next().await {
        match ev {
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            TokenEvent::Started { prompt_tokens } => run.prompt_tokens = prompt_tokens,
            TokenEvent::Logprobs(t) => run.tokens.extend(t),
            TokenEvent::Finished { reason, usage, .. } => {
                run.finished = format!("{reason:?} {}", usage.completion_tokens);
                break;
            }
            _ => {}
        }
    }
    run
}

/// How far a token's log probability may move with the batch shape. The CPU matmul kernels
/// (llamafile sgemm for several columns, ggml's dot products for one) sum in different orders
/// on aarch64, and the Q8_0 quantization of every activation turns a last-bit difference into
/// a 1/127 step of one element; the tiny model's logits move by up to ~7e-3 that way (x86:
/// bit-identical). A wrong slot, state or position moves them by orders of magnitude more.
const LOGPROB_TOLERANCE: f32 = 3e-2;

/// Checks that two runs of one request are the same up to batch-shape noise: the same tokens
/// with log probabilities within [`LOGPROB_TOLERANCE`]. The tiny model's logits are nearly
/// flat (its top tokens are ~1e-3 apart), so noise can legitimately flip an argmax; the runs
/// may then differ from that token on, provided it was a near-tie in both runs.
fn assert_same_run(i: usize, alone: &Run, together: &Run) {
    assert_eq!(alone.prompt_tokens, together.prompt_tokens, "request {i}");
    assert_eq!(alone.finished, together.finished, "request {i}");
    assert!(
        alone.tokens.len() > 5,
        "request {i}: {} tokens",
        alone.tokens.len()
    );
    // (A token whose text is an incomplete character gets no entry, so the counts may differ.)
    // The log probability `run` gave `token` at this position, if it is among its top tokens.
    let lp_of = |run: &legion::engine::TokenLogprob, token: &str| {
        run.top.iter().find(|(t, _)| t == token).map(|(_, lp)| *lp)
    };
    for (n, (a, b)) in alone.tokens.iter().zip(&together.tokens).enumerate() {
        if a.token == b.token {
            let d = (a.logprob - b.logprob).abs();
            assert!(
                d < LOGPROB_TOLERANCE,
                "request {i}, token {n} {:?}: logprob {} vs {}",
                a.token,
                a.logprob,
                b.logprob
            );
            continue;
        }
        let tie = |x: &legion::engine::TokenLogprob, other: &legion::engine::TokenLogprob| {
            lp_of(x, &other.token).is_some_and(|lp| (x.logprob - lp).abs() < LOGPROB_TOLERANCE)
        };
        assert!(
            tie(a, b) && tie(b, a),
            "request {i}, token {n}: {:?} vs {:?} is not a near-tie ({:?} / {:?})",
            a.token,
            b.token,
            a.top,
            b.top
        );
        return;
    }
}

/// The engine on the tiny model: requests run one at a time and the same requests run
/// together (continuous batching, prefill chunks of 8 tokens next to decode steps, slot moves,
/// prefix-cache restores) produce the same greedy output, up to [`assert_same_run`]'s noise.
#[tokio::test(flavor = "multi_thread")]
async fn engine_batching_matches_single_requests() {
    use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
    // Dense: which pools sparse attention keeps among equal scores (the indexer's ReLU makes
    // exact zeros common in this tiny model) depends on where they sit in the step's pool list,
    // in llama.cpp too (its top_k is a partial sort); the model-level tests cover sparse steps.
    let (_dir, gguf) = tiny_with(4096);
    let path = gguf.path().to_path_buf();
    let load = || {
        LmEngine::load(LmConfig {
            ctx_size: 128,
            max_seqs: 3,
            max_batch_tokens: 8,
            threads: 4,
            ..LmConfig::new(&path, DeviceChoice::Cpu)
        })
        .unwrap()
    };
    let prompts = [
        "The capital of France is Paris, and the capital of Germany is",
        "One two three four five six seven eight nine ten eleven",
        "The capital of France is Paris, and the capital of Italy is",
    ];
    let alone = {
        let engine = load();
        let mut v = Vec::new();
        for p in prompts {
            v.push(engine_run(&engine, p, 24).await);
        }
        v
    };
    let engine = std::sync::Arc::new(load());
    let together = futures::future::join_all(prompts.iter().map(|p| {
        let engine = engine.clone();
        async move { engine_run(&engine, p, 24).await }
    }))
    .await;
    for (i, (a, b)) in alone.iter().zip(&together).enumerate() {
        assert_same_run(i, a, b);
    }
}

/// Greedy rows of each prompt run alone: the prompt's last row, then one per decoded token.
fn alone_rows(model: &Model, backend: &mut Backend, prompt: &[u32], steps: usize) -> Vec<Vec<f32>> {
    let mut state = model.new_state(backend.buffer_type(), 64).unwrap();
    let mut alloc = None;
    let mut rows = vec![
        model
            .forward(backend, &mut alloc, &mut state, prompt, false)
            .unwrap(),
    ];
    for _ in 0..steps {
        let next = argmax(rows.last().unwrap()) as u32;
        rows.push(
            model
                .forward(backend, &mut alloc, &mut state, &[next], false)
                .unwrap(),
        );
    }
    rows
}

const MIXED_PROMPTS: [std::ops::Range<usize>; 3] = [0..13, 3..9, 5..22];
const MIXED_STEPS: usize = 16;

/// Steps as the engine runs them, through `step` (one batch step, returns the logits rows):
/// prompts 0 and 1 prefill (a group each), then decode as one group while prompt 2 prefills
/// in two chunks next to them (no logits for the first, so the last layer runs on the output
/// rows only) in slot 4; `join` then moves its state to decode slot 2 and the three decode
/// together. Every row must equal the sequence's own run, `alone`.
fn check_mixed_steps(
    alone: &[Vec<Vec<f32>>],
    step: &mut dyn FnMut(&[Group<'_>]) -> Vec<Vec<f32>>,
    join: &mut dyn FnMut(),
) {
    let prompts: Vec<&[u32]> = MIXED_PROMPTS
        .iter()
        .map(|r| &TINY_PROMPT[r.clone()])
        .collect();
    // Out of order, as the engine's block tables are.
    let blocks: Vec<Vec<u32>> = vec![vec![9, 2, 5, 0], vec![1, 3, 4, 6], vec![10, 7, 8, 11]];
    fn seq<'b>(
        blocks: &'b [u32],
        tokens: &'b [u32],
        n_past: usize,
        logits: Logits,
    ) -> SeqInput<'b> {
        SeqInput {
            tokens,
            n_past,
            blocks,
            ring: 0,
            logits,
            media: None,
        }
    }
    let mut last = step(&[
        Group {
            slot: 0,
            verify: false,
            seqs: vec![seq(&blocks[0], prompts[0], 0, Logits::Last)],
        },
        Group {
            slot: 1,
            verify: false,
            seqs: vec![seq(&blocks[1], prompts[1], 0, Logits::Last)],
        },
    ]);
    let mut n_past = [prompts[0].len(), prompts[1].len(), 0];
    let mut done = [0usize; 3];
    let chunks = [&prompts[2][..9], &prompts[2][9..]];
    for s in 0..MIXED_STEPS {
        for (i, row) in last.iter().enumerate() {
            let want = &alone[i][done[i]];
            assert_eq!(max_diff(row, want), 0.0, "sequence {i}, row {}", done[i]);
        }
        let next: Vec<u32> = last.iter().map(|r| argmax(r) as u32).collect();
        let mut groups = vec![Group {
            slot: 0,
            verify: false,
            seqs: (0..last.len())
                .map(|i| seq(&blocks[i], &next[i..=i], n_past[i], Logits::Last))
                .collect(),
        }];
        if let Some(chunk) = chunks.get(s) {
            let logits = if s + 1 == chunks.len() {
                Logits::Last
            } else {
                Logits::None
            };
            groups.push(Group {
                slot: 4,
                verify: false,
                seqs: vec![seq(&blocks[2], chunk, n_past[2], logits)],
            });
        }
        let rows = step(&groups);
        for i in 0..last.len() {
            n_past[i] += 1;
            done[i] += 1;
        }
        if let Some(chunk) = chunks.get(s) {
            n_past[2] += chunk.len();
        }
        last = rows;
        if s + 1 == chunks.len() {
            join();
        }
    }
}

/// Mixed steps through the executor (whose decode-only and prefill steps replay kept
/// graphs) against each sequence alone.
#[test]
fn executor_steps_match_single_sequences() {
    let (_dir, gguf) = tiny();
    let mut backend = cpu();
    backend.set_use_ref(true);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let alone: Vec<_> = MIXED_PROMPTS
        .iter()
        .map(|r| alone_rows(&model, &mut backend, &TINY_PROMPT[r.clone()], MIXED_STEPS))
        .collect();
    let exec = std::cell::RefCell::new(Executor::new(model, backend, 12, 5, 0).unwrap());
    check_mixed_steps(
        &alone,
        &mut |groups| exec.borrow_mut().step(groups).unwrap().rows,
        &mut || {
            let mut e = exec.borrow_mut();
            e.copy_states(&[(Pool::Seq, 4, Pool::Seq, 2)]).unwrap();
            e.clear_slot(4);
        },
    );
}

/// A model split over two machines (#562): rank 0 runs the first layers and sends the
/// residual streams over TCP, rank 1 (a thread here, on loopback) runs the rest. Prompt
/// chunks without logits do not wait for rank 1; the logits of every step, and of decode
/// groups in flight over the two ranks, are those of one executor bit for bit.
#[test]
fn split_over_two_nodes_matches_one() {
    use legion::model::core::{Remote, from_wire};
    use legion::node::{Link, Reply, Request, serve_one};
    const SPLIT: usize = 3;
    let (dir, gguf) = tiny();
    let path = dir.path().join("tiny-qwen4exp.gguf");
    let seq = |tokens, n_past, blocks, logits| SeqInput {
        tokens,
        n_past,
        blocks,
        ring: 0,
        logits,
        media: None,
    };
    let group = |slot, s| Group {
        slot,
        seqs: vec![s],
        verify: false,
    };
    let blocks: [&[u32]; 2] = [&[0, 1], &[2, 3]];
    let prompts: [&[u32]; 2] = [TINY_PROMPT, &TINY_PROMPT[3..17]];
    // Each sequence's prompt in chunks (the first without logits), then two decode steps per
    // sequence through `pipe` (groups in flight) or whole steps.
    let run = |exec: &mut Executor, pipe: bool| -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let (a, b) = p.split_at(7);
            let none = exec
                .step(&[group(i, seq(a, 0, blocks[i], Logits::None))])
                .unwrap();
            assert!(none.rows.iter().all(|r| r.is_empty()));
            let o = exec
                .step(&[group(i, seq(b, 7, blocks[i], Logits::All))])
                .unwrap();
            out.push(o.rows.concat());
        }
        static NEXT: [[u32; 1]; 4] = [[11], [19], [23], [29]];
        let next = &NEXT;
        if pipe {
            exec.pipe_begin();
            let mut done = Vec::new();
            for (k, t) in next.iter().enumerate() {
                let i = k % 2;
                let n_past = prompts[i].len() + k / 2;
                let g = [group(i, seq(t, n_past, blocks[i], Logits::Last))];
                if let Some((tag, o)) = exec.pipe_round(Some((k, &g))).unwrap() {
                    done.push((tag, o.rows.concat()));
                }
            }
            while exec.pipe_in_flight() > 0 {
                if let Some((tag, o)) = exec.pipe_round(None).unwrap() {
                    done.push((tag, o.rows.concat()));
                }
            }
            exec.pipe_end();
            done.sort_by_key(|d| d.0);
            out.extend(done.into_iter().map(|d| d.1));
        } else {
            for (k, t) in next.iter().enumerate() {
                let i = k % 2;
                let n_past = prompts[i].len() + k / 2;
                let o = exec
                    .step(&[group(i, seq(t, n_past, blocks[i], Logits::Last))])
                    .unwrap();
                out.push(o.rows.concat());
            }
        }
        out
    };
    let backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n = model.hp.n_layer;
    let mut one = Executor::new_staged_ring(
        model,
        vec![(backend, 0..n)],
        4,
        2,
        0,
        None,
        KvTypes::default(),
    )
    .unwrap();
    let want = run(&mut one, false);
    drop(one);
    for pipe in [false, true] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let p = path.clone();
        let rank1 = std::thread::spawn(move || {
            let gguf = GgufFile::open(&p).unwrap();
            let backend = cpu();
            let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
            let n = model.hp.n_layer;
            let mut exec = Executor::new_staged_ring(
                model,
                vec![(backend, SPLIT..n)],
                4,
                2,
                0,
                None,
                KvTypes::default(),
            )
            .unwrap();
            let (stream, _) = listener.accept().unwrap();
            serve_one(stream, &mut |req| match req {
                Request::Step { groups, rows } => {
                    let g = from_wire(&groups).unwrap();
                    match exec.step_from(&g, rows) {
                        Ok(o) => Reply::Step {
                            argmax: o.argmax,
                            rows: o.rows,
                            ms: 0.0,
                        },
                        Err(e) => Reply::Error(e.to_string()),
                    }
                }
                _ => Reply::Done,
            })
            .unwrap();
        });
        let backend = cpu();
        let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
        let mut rank0 = Executor::new_staged_ring(
            model,
            vec![(backend, 0..SPLIT)],
            4,
            2,
            0,
            None,
            KvTypes::default(),
        )
        .unwrap();
        let link = Link::connect(&addr, std::time::Duration::from_secs(5)).unwrap();
        rank0.set_remote(Remote::new(link));
        assert!(rank0.can_pipe());
        let got = run(&mut rank0, pipe);
        drop(rank0);
        rank1.join().unwrap();
        assert_eq!(got.len(), want.len());
        for (k, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(!a.is_empty(), "pipe {pipe}: step {k} without logits");
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert!(
                bits(a) == bits(b),
                "pipe {pipe}: step {k}: max |diff| {}",
                max_diff(a, b)
            );
        }
    }
}
