//! A tiny DFlash drafter with random weights, written as a GGUF or a Hugging Face directory, for
//! tests that need the real graph and no real model. It fits the tiny `qwen35` target of
//! [`crate::model::qwen35::tiny`] (hidden size 64, four layers, byte-level vocabulary): two
//! layers of width 64 that read the target's layers 1 and 3 and draft blocks of up to 8 tokens
//! (7 drafts). Its drafts are meaningless; what it computes is the same as for a real drafter.
//! With `markov_rank` it is a DSpark drafter (Markov head, optionally a confidence head,
//! attention sinks, and quantized matrices in the GGUF).

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::{ffi, types};
use crate::gguf::{GgufError, GgufWriter, Value};

use super::Dflash2;

/// The tiny target's vocabulary.
pub const TARGET_VOCAB: usize = 259;

/// How the attention sinks are named in a GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sinks {
    None,
    /// `blk.N.attn_sinks.weight`.
    Weight,
    /// `blk.N.attn_sinks`.
    Bare,
}

/// Shapes of the tiny drafter.
#[derive(Debug, Clone, PartialEq)]
pub struct TinyDraft {
    pub n_layer: usize,
    /// The target's hidden size too.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub block_size: usize,
    /// 0-based target layers whose outputs are fused.
    pub target_layers: Vec<usize>,
    pub mask_token: u32,
    pub rope_base: f32,
    /// Sliding-window attention over the context.
    pub sliding_window: Option<usize>,
    pub embedding_scale: f32,
    /// Leave out the V projections (V is the normed K projection).
    pub shared_kv: bool,
    /// A reduced draft vocabulary: its own output head, and the target token id of each row.
    pub draft_vocab: Option<Vec<u32>>,
    /// DSpark: the rank of the Markov head (makes it a DSpark drafter).
    pub markov_rank: Option<usize>,
    /// DSpark: with a confidence head.
    pub confidence: bool,
    pub sinks: Sinks,
    /// DSpark: whether block position 0 gives a draft.
    pub sample_from_anchor: bool,
    /// GGUF only: store the matrices in this ggml type where a row fits its blocks.
    pub quant: Option<ffi::ggml_type>,
    /// DFlash 2: convolutions and a candidate selector.
    pub v2: Option<Dflash2>,
    pub seed: u64,
}

impl Default for TinyDraft {
    fn default() -> Self {
        Self {
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 16,
            n_ff: 96,
            block_size: 8,
            target_layers: vec![1, 3],
            mask_token: 256,
            rope_base: 10_000.0,
            sliding_window: None,
            embedding_scale: 1.0,
            shared_kv: false,
            draft_vocab: None,
            markov_rank: None,
            confidence: false,
            sinks: Sinks::None,
            sample_from_anchor: true,
            quant: None,
            v2: None,
            seed: 7,
        }
    }
}

/// DFlash 2 settings that fit the tiny drafter: four groups of 16 channels, two taps, a
/// rank-8 selector over the top 4 tokens.
pub const TINY_V2: Dflash2 = Dflash2 {
    conv_kernel: 2,
    conv_group: 16,
    selector_rank: 8,
    selector_top_k: 4,
};

/// `x` rounded the way [`write_hf`] stores it as BF16 (truncated).
pub fn bf16_round(x: f32) -> f32 {
    f32::from_bits((x.to_bits() >> 16) << 16)
}

/// A tensor: GGUF name, shape in ggml order, values.
pub type Tensor = (String, Vec<i64>, Vec<f32>);

/// The drafter's weights, under the GGUF names.
pub fn tensors(c: &TinyDraft) -> Vec<Tensor> {
    let rng = std::cell::RefCell::new(Rng::new(c.seed));
    let uniform = |n: usize, a: f32| -> Vec<f32> {
        let mut rng = rng.borrow_mut();
        (0..n)
            .map(|_| (rng.next_f64() as f32 * 2.0 - 1.0) * a)
            .collect()
    };
    let mut out: Vec<Tensor> = Vec::new();
    let e = c.n_embd as i64;
    let (q, kv) = (
        (c.head_dim * c.n_head) as i64,
        (c.head_dim * c.n_head_kv) as i64,
    );
    let ff = c.n_ff as i64;
    let hd = c.head_dim as i64;
    let mat = |out: &mut Vec<Tensor>, name: String, k: i64, n: i64| {
        let v = uniform((k * n) as usize, 1.5 / (k as f32).sqrt());
        out.push((name, vec![k, n], v));
    };
    let norm = |out: &mut Vec<Tensor>, name: String, n: i64| {
        let v: Vec<f32> = uniform(n as usize, 0.2)
            .into_iter()
            .map(|x| 1.0 + x)
            .collect();
        out.push((name, vec![n], v));
    };
    mat(
        &mut out,
        "fc.weight".into(),
        c.target_layers.len() as i64 * e,
        e,
    );
    norm(&mut out, "enc.output_norm.weight".into(), e);
    norm(&mut out, "output_norm.weight".into(), e);
    for il in 0..c.n_layer {
        let t = |n: &str| format!("blk.{il}.{n}");
        norm(&mut out, t("attn_norm.weight"), e);
        mat(&mut out, t("attn_q.weight"), e, q);
        mat(&mut out, t("attn_k.weight"), e, kv);
        if !c.shared_kv {
            mat(&mut out, t("attn_v.weight"), e, kv);
        }
        mat(&mut out, t("attn_output.weight"), q, e);
        norm(&mut out, t("attn_q_norm.weight"), hd);
        norm(&mut out, t("attn_k_norm.weight"), hd);
        norm(&mut out, t("ffn_norm.weight"), e);
        mat(&mut out, t("ffn_gate.weight"), e, ff);
        mat(&mut out, t("ffn_up.weight"), e, ff);
        mat(&mut out, t("ffn_down.weight"), ff, e);
        if let Some(v2) = &c.v2 {
            for part in ["attn", "ffn"] {
                // Taps near the identity (tap 0 about 1, tap 1 about 0.3), as trained ones are.
                let k = v2.conv_kernel;
                let base: Vec<f32> = uniform(c.n_embd * k * 2, 0.3)
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| {
                        if (i / c.n_embd).is_multiple_of(k) {
                            1.0 + x
                        } else {
                            x
                        }
                    })
                    .collect();
                out.push((t(&format!("{part}_conv_base")), vec![e, k as i64, 2], base));
                mat(
                    &mut out,
                    t(&format!("{part}_conv_proj.weight")),
                    e,
                    v2.projected(c.n_embd) as i64,
                );
            }
        }
    }
    if let Some(v2) = &c.v2 {
        let r = v2.selector_rank as i64;
        mat(&mut out, "selector_hidden.weight".into(), e, r);
        for book in ["selector_predecessor.weight", "selector_successor.weight"] {
            let v = uniform(v2.selector_rank * TARGET_VOCAB, 1.0);
            out.push((book.into(), vec![r, TARGET_VOCAB as i64], v));
        }
    }
    if let Some(ids) = &c.draft_vocab {
        mat(&mut out, "output.weight".into(), e, ids.len() as i64);
    }
    if c.sinks != Sinks::None {
        for il in 0..c.n_layer {
            let v = uniform(c.n_head, 1.5);
            out.push((
                format!("blk.{il}.attn_sinks.weight"),
                vec![c.n_head as i64],
                v,
            ));
        }
    }
    if let Some(rank) = c.markov_rank {
        let r = rank as i64;
        let n_out = c.draft_vocab.as_ref().map_or(TARGET_VOCAB, Vec::len) as i64;
        let w1 = uniform(rank * TARGET_VOCAB, 1.0);
        out.push(("markov_w1.weight".into(), vec![r, TARGET_VOCAB as i64], w1));
        let w2 = uniform(rank * n_out as usize, 2.0 / (rank as f32).sqrt());
        out.push(("markov_w2.weight".into(), vec![r, n_out], w2));
        if c.confidence {
            let w = uniform(c.n_embd + rank, 1.0 / (c.n_embd as f32).sqrt());
            out.push(("conf_proj.weight".into(), vec![e + r, 1], w));
            out.push(("conf_proj.bias".into(), vec![1], uniform(1, 0.5)));
        }
    }
    out
}

/// The weights a GGUF of `c` holds, after the quantization round trip (what the network
/// computes with).
pub fn effective_tensors(c: &TinyDraft) -> Vec<Tensor> {
    tensors(c)
        .into_iter()
        .map(|(n, shape, v)| match quant_of(c, &shape) {
            Some(ty) => {
                let q = crate::ggml::quantize(ty, &v, shape[0] as usize);
                let v = crate::ggml::dequantize(ty, &q, v.len());
                (n, shape, v)
            }
            None => (n, shape, v),
        })
        .collect()
}

/// The type a tensor of `shape` is stored in, when it is quantized.
fn quant_of(c: &TinyDraft, shape: &[i64]) -> Option<ffi::ggml_type> {
    let ty = c.quant?;
    // SAFETY: a static table lookup.
    let block = unsafe { ffi::ggml_blck_size(ty) };
    (shape.len() == 2 && shape[1] > 1 && shape[0] % block == 0).then_some(ty)
}

/// `d2t` as a GGUF stores it: absolute target ids.
fn d2t_bytes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|&i| (i as i64).to_ne_bytes()).collect()
}

/// Writes the drafter as a GGUF of llama.cpp's `dflash` architecture.
pub fn write_gguf(path: impl AsRef<Path>, c: &TinyDraft) -> Result<(), GgufError> {
    write_gguf_with(path, c, &[])
}

/// Like [`write_gguf`], plus `extra` tensors (F32).
pub fn write_gguf_with(
    path: impl AsRef<Path>,
    c: &TinyDraft,
    extra: &[Tensor],
) -> Result<(), GgufError> {
    let mut w = GgufWriter::new();
    let u32v = |v: usize| Value::U32(v as u32);
    w.set("general.architecture", &Value::String("dflash".into()))?;
    w.set("general.name", &Value::String("tiny-dflash".into()))?;
    for (k, v) in [
        ("block_count", u32v(c.n_layer)),
        ("context_length", u32v(4096)),
        ("embedding_length", u32v(c.n_embd)),
        ("feed_forward_length", u32v(c.n_ff)),
        ("attention.head_count", u32v(c.n_head)),
        ("attention.head_count_kv", u32v(c.n_head_kv)),
        ("attention.key_length", u32v(c.head_dim)),
        ("attention.value_length", u32v(c.head_dim)),
        ("rope.dimension_count", u32v(c.head_dim)),
        ("rope.freq_base", Value::F32(c.rope_base)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("block_size", u32v(c.block_size)),
        ("embedding_scale", Value::F32(c.embedding_scale)),
    ] {
        w.set(&format!("dflash.{k}"), &v)?;
    }
    // The file stores the layers whose inputs are read: output of layer i = input of i + 1.
    w.set(
        "dflash.target_layers",
        &Value::Array(c.target_layers.iter().map(|&l| u32v(l + 1)).collect()),
    )?;
    if let Some(win) = c.sliding_window {
        w.set("dflash.attention.sliding_window", &u32v(win))?;
        w.set(
            "dflash.attention.sliding_window_pattern",
            &Value::Array(vec![Value::Bool(true); c.n_layer]),
        )?;
    }
    w.set("tokenizer.ggml.mask_token_id", &Value::U32(c.mask_token))?;
    if c.markov_rank.is_some() {
        w.set(
            "dflash.sample_from_anchor",
            &Value::Bool(c.sample_from_anchor),
        )?;
        w.set("dflash.has_confidence_head", &Value::Bool(c.confidence))?;
    }
    if let Some(v2) = &c.v2 {
        w.set("dflash.conv_kernel_size", &u32v(v2.conv_kernel))?;
        w.set("dflash.conv_group_size", &u32v(v2.conv_group))?;
        w.set("dflash.selector_rank", &u32v(v2.selector_rank))?;
        w.set("dflash.selector_top_k", &u32v(v2.selector_top_k))?;
    }
    for (name, shape, v) in tensors(c).iter().chain(extra) {
        let name = match (c.sinks, name.strip_suffix(".weight")) {
            (Sinks::Bare, Some(bare)) if bare.ends_with("attn_sinks") => bare.to_string(),
            _ => name.clone(),
        };
        match quant_of(c, shape) {
            Some(ty) => {
                let q = crate::ggml::quantize(ty, v, shape[0] as usize);
                w.add_tensor(&name, ty, shape, &q)?;
            }
            None => {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
                w.add_tensor(&name, types::F32, shape, &bytes)?;
            }
        }
    }
    if let Some(ids) = &c.draft_vocab {
        w.add_tensor(
            "d2t",
            crate::ggml::ffi::ggml_type_GGML_TYPE_I64,
            &[ids.len() as i64],
            &d2t_bytes(ids),
        )?;
    }
    w.write(path)
}

/// Element type of a Hugging Face checkpoint written by [`write_hf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfDtype {
    F32,
    Bf16,
}

/// A GGUF tensor name in the Hugging Face naming (the inverse of `weights::hf_name`).
fn hf_of(name: &str) -> String {
    match name {
        "output.weight" => return "lm_head.weight".into(),
        "markov_w1.weight" => return "markov_head.markov_w1.weight".into(),
        "markov_w2.weight" => return "markov_head.markov_w2.weight".into(),
        "conf_proj.weight" => return "confidence_head.proj.weight".into(),
        "conf_proj.bias" => return "confidence_head.proj.bias".into(),
        "fc.weight" => return "fc.weight".into(),
        "enc.output_norm.weight" => return "hidden_norm.weight".into(),
        "output_norm.weight" => return "norm.weight".into(),
        "selector_hidden.weight" => return "candidate_selector.hidden_projection.weight".into(),
        // As z-lab's checkpoints store them (no `.weight`).
        "selector_predecessor.weight" => return "candidate_selector.predecessor_codebook".into(),
        "selector_successor.weight" => return "candidate_selector.successor_codebook".into(),
        _ => {}
    }
    let rest = name.strip_prefix("blk.").expect("a layer tensor");
    let (n, leaf) = rest.split_once('.').unwrap();
    let leaf = match leaf {
        "attn_sinks.weight" => "self_attn.sinks",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_conv_base" => "attention_conv.base_kernel",
        "attn_conv_proj.weight" => "attention_conv.kernel_projection.weight",
        "ffn_conv_base" => "mlp_conv.base_kernel",
        "ffn_conv_proj.weight" => "mlp_conv.kernel_projection.weight",
        other => panic!("no Hugging Face name for {other}"),
    };
    format!("layers.{n}.{leaf}")
}

/// Whether the FP8 checkpoints (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`, [`write_hf_fp8`]) store
/// the tensor `name` (GGUF naming) as FP8: the layers' seven projections and `fc`.
pub fn fp8_stored(name: &str) -> bool {
    const LEAVES: [&str; 7] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ];
    name == "fc.weight" || (name.starts_with("blk.") && LEAVES.iter().any(|l| name.ends_with(l)))
}

/// `v` (`rows` x `cols`, row major) as FP8 with block scales: per `block[0]` x `block[1]` block
/// the scale is its absolute maximum over 448 rounded to BF16 (1 for an all-zero block), and
/// each element the E4M3 value nearest to it over the scale. Returns the bytes and the scales
/// (row major, `rows.div_ceil(block[0])` x `cols.div_ceil(block[1])`).
pub fn fp8_blocks(v: &[f32], rows: usize, cols: usize, block: [usize; 2]) -> (Vec<u8>, Vec<f32>) {
    let (sr, sc) = (rows.div_ceil(block[0]), cols.div_ceil(block[1]));
    let mut scales = vec![0f32; sr * sc];
    for (i, s) in scales.iter_mut().enumerate() {
        let (br, bc) = (i / sc * block[0], i % sc * block[1]);
        let mut amax = 0f32;
        for r in br..(br + block[0]).min(rows) {
            for c in bc..(bc + block[1]).min(cols) {
                amax = amax.max(v[r * cols + c].abs());
            }
        }
        *s = if amax == 0.0 {
            1.0
        } else {
            bf16_round(amax / 448.0)
        };
    }
    let bytes = (0..rows * cols)
        .map(|i| {
            let s = scales[i / cols / block[0] * sc + i % cols / block[1]];
            crate::model::fp8::f32_to_e4m3(v[i] / s)
        })
        .collect();
    (bytes, scales)
}

/// Writes the drafter as a Hugging Face directory: `config.json` and `model.safetensors`.
pub fn write_hf(dir: impl AsRef<Path>, c: &TinyDraft, dtype: HfDtype) -> std::io::Result<()> {
    write_hf_with(dir.as_ref(), c, dtype, None)
}

/// [`write_hf`] in BF16 with the [`fp8_stored`] matrices as FP8 ([`fp8_blocks`]) and their
/// `weight_scale_inv` (BF16), as `quant_method: fp8` checkpoints with `weight_block_size` `block`.
pub fn write_hf_fp8(
    dir: impl AsRef<Path>,
    c: &TinyDraft,
    block: [usize; 2],
) -> std::io::Result<()> {
    write_hf_with(dir.as_ref(), c, HfDtype::Bf16, Some(block))
}

fn write_hf_with(
    dir: &Path,
    c: &TinyDraft,
    dtype: HfDtype,
    fp8: Option<[usize; 2]>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let arch = if c.markov_rank.is_some() {
        "DSparkDraftModel"
    } else if c.v2.is_some() {
        "DFlash2DraftModel"
    } else {
        "DFlashDraftModel"
    };
    let mut config = serde_json::json!({
        "architectures": [arch],
        "hidden_size": c.n_embd,
        "num_hidden_layers": c.n_layer,
        "num_attention_heads": c.n_head,
        "num_key_value_heads": c.n_head_kv,
        "head_dim": c.head_dim,
        "intermediate_size": c.n_ff,
        "rms_norm_eps": 1e-6,
        "rope_theta": c.rope_base,
        "dflash_config": {
            "block_size": c.block_size,
            "mask_token_id": c.mask_token,
            "target_layer_ids": c.target_layers,
        },
    });
    if let Some(w) = c.sliding_window {
        config["dflash_config"]["use_swa"] = true.into();
        config["dflash_config"]["swa_window_size"] = w.into();
    }
    if let Some(rank) = c.markov_rank {
        config["markov_rank"] = rank.into();
        config["markov_head_type"] = "vanilla".into();
        config["sample_from_anchor"] = c.sample_from_anchor.into();
    }
    if c.embedding_scale != 1.0 {
        config["dflash_config"]["input_embedding_scale"] = c.embedding_scale.into();
    }
    if let Some(v2) = &c.v2 {
        let d = &mut config["dflash_config"];
        d["conv_kernel_size"] = v2.conv_kernel.into();
        d["conv_group_size"] = v2.conv_group.into();
        d["selector_rank"] = v2.selector_rank.into();
        d["selector_top_k"] = v2.selector_top_k.into();
    }
    if let Some(block) = fp8 {
        config["quantization_config"] = serde_json::json!({
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": block,
        });
    }
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config)?)?;

    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, v) in tensors(c) {
        if let Some(block) = fp8.filter(|_| fp8_stored(&name)) {
            // F8_E4M3 bytes are 1 each and BF16 scales 2: offsets stay even for what follows.
            let (rows, cols) = (shape[1] as usize, shape[0] as usize);
            let (bytes, scales) = fp8_blocks(&v, rows, cols, block);
            let start = data.len();
            data.extend_from_slice(&bytes);
            header.insert(
                hf_of(&name),
                serde_json::json!({
                    "dtype": "F8_E4M3",
                    "shape": [rows, cols],
                    "data_offsets": [start, data.len()],
                }),
            );
            let start = data.len();
            for s in &scales {
                data.extend_from_slice(&((s.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                format!("{}_scale_inv", hf_of(&name)),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": [rows.div_ceil(block[0]), cols.div_ceil(block[1])],
                    "data_offsets": [start, data.len()],
                }),
            );
            continue;
        }
        let start = data.len();
        for &x in &v {
            match dtype {
                HfDtype::F32 => data.extend_from_slice(&x.to_le_bytes()),
                HfDtype::Bf16 => {
                    data.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes())
                }
            }
        }
        let dims: Vec<i64> = shape.iter().rev().copied().collect();
        header.insert(
            hf_of(&name),
            serde_json::json!({
                "dtype": match dtype { HfDtype::F32 => "F32", HfDtype::Bf16 => "BF16" },
                "shape": dims,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    if let Some(ids) = &c.draft_vocab {
        // Hugging Face stores offsets from the row's own index.
        let start = data.len();
        for (i, &t) in ids.iter().enumerate() {
            data.extend_from_slice(&(t as i64 - i as i64).to_le_bytes());
        }
        header.insert(
            "d2t".into(),
            serde_json::json!({
                "dtype": "I64",
                "shape": [ids.len()],
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut json = serde_json::to_vec(&header)?;
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&json);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file)
}
