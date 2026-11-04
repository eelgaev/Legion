//! A tiny `qwen35` model with random weights, written as a GGUF, for tests that need the real
//! graph (gated DeltaNet + gated attention, paged KV, recurrent states) but no real model:
//! three DeltaNet layers and one attention layer of width 64, F32 weights, and a byte-level
//! tokenizer (256 byte tokens plus `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`).
//! Its outputs are meaningless; what it computes is the same as for a real model.

use std::path::Path;

use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufError, GgufWriter, Value};

/// Shapes of the tiny model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyConfig {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub seed: u64,
    /// Tokens added to the vocabulary after the specials (a different vocabulary of the same
    /// architecture, for tests of vocabulary checks).
    pub extra_tokens: u32,
    /// NVFP4 FFN and output weights, laid out as llama.cpp's converter writes them.
    pub nvfp4: Option<TinyNvfp4>,
    /// An MTP head (block `n_layer`, `nextn.*`) in the model's file, as the HF converter writes
    /// Qwen3.5's (see [`write_mtp_head`] for one in a file of its own).
    pub mtp: bool,
    /// User-defined tokens (type 4, as converters write chat markup such as `<tool_call>`)
    /// added after `extra_tokens`.
    pub user_tokens: &'static [&'static str],
    /// `tokenizer.chat_template` (`None`: no template).
    pub chat_template: Option<&'static str>,
}

/// NVFP4 weights for the tiny model: `ffn_gate`, `ffn_up`, `ffn_down` and `output` get random
/// E2M1 codes and UE4M3 block scales (their own random stream, so these settings change values,
/// never which random numbers are drawn), plus the converter's per-tensor tensors and
/// `general.tensor_extra.prec_a4 = false` (a W4A16 checkpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyNvfp4 {
    /// Every `<name>.scale` is 2^this; `None` leaves the scales out, as the converter does for a
    /// scale of 1.
    pub scale_exp: Option<i8>,
    /// Added to every block scale's exponent, which multiplies the weights by 2^this exactly.
    pub block_exp_shift: u8,
}

impl Default for TinyConfig {
    fn default() -> Self {
        Self {
            n_layer: 4,
            n_embd: 64,
            n_ff: 128,
            seed: 1,
            extra_tokens: 0,
            nvfp4: None,
            mtp: false,
            user_tokens: &[],
            chat_template: None,
        }
    }
}

/// Byte tokens come first (id = byte), then these.
pub const SPECIAL_TOKENS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

/// The end-of-generation token id (`<|im_end|>`).
pub const EOS: u32 = 258;

const N_HEAD: u32 = 4;
const N_HEAD_KV: u32 = 2;
const HEAD_DIM: u32 = 16;
const D_CONV: u32 = 4;
/// DeltaNet key heads, value heads, head size.
const N_K: u32 = 2;
const N_V: u32 = 4;
const D_STATE: u32 = 16;

/// GPT-2's `bytes_to_unicode` (the byte-level BPE alphabet).
fn byte_char(b: u8) -> char {
    let printable = |c: u32| matches!(c, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable(b as u32) {
        return char::from_u32(b as u32).unwrap();
    }
    let rank = (0..b as u32).filter(|&c| !printable(c)).count() as u32;
    char::from_u32(256 + rank).unwrap()
}

/// Writes the tiny model to `path`.
pub fn write(path: impl AsRef<Path>, cfg: &TinyConfig) -> Result<(), GgufError> {
    let mut rng = Rng::new(cfg.seed);
    let mut uniform = move |n: usize, a: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|_| ((rng.next_f64() as f32 * 2.0 - 1.0) * a).to_ne_bytes())
            .collect()
    };
    let n_vocab =
        256 + SPECIAL_TOKENS.len() as u32 + cfg.extra_tokens + cfg.user_tokens.len() as u32;
    let e = cfg.n_embd as i64;
    let d_inner = (D_STATE * N_V) as i64;
    let conv_dim = d_inner + 2 * (N_K * D_STATE) as i64;

    let mut w = GgufWriter::new();
    let u32v = |v: u32| Value::U32(v);
    let arch = "qwen35";
    w.set("general.architecture", &Value::String(arch.into()))?;
    w.set("general.name", &Value::String("tiny-qwen35".into()))?;
    if cfg.mtp {
        w.set(&format!("{arch}.nextn_predict_layers"), &u32v(1))?;
    }
    for (k, v) in [
        ("block_count", u32v(cfg.n_layer + cfg.mtp as u32)),
        ("context_length", u32v(4096)),
        ("embedding_length", u32v(cfg.n_embd)),
        ("feed_forward_length", u32v(cfg.n_ff)),
        ("attention.head_count", u32v(N_HEAD)),
        ("attention.head_count_kv", u32v(N_HEAD_KV)),
        ("attention.key_length", u32v(HEAD_DIM)),
        ("attention.value_length", u32v(HEAD_DIM)),
        ("rope.dimension_count", u32v(HEAD_DIM)),
        ("rope.freq_base", Value::F32(10_000.0)),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("full_attention_interval", u32v(4)),
        ("ssm.conv_kernel", u32v(D_CONV)),
        ("ssm.state_size", u32v(D_STATE)),
        ("ssm.group_count", u32v(N_K)),
        ("ssm.time_step_rank", u32v(N_V)),
        ("ssm.inner_size", u32v(d_inner as u32)),
    ] {
        w.set(&format!("{arch}.{k}"), &v)?;
    }
    w.set(
        &format!("{arch}.rope.dimension_sections"),
        &Value::Array(vec![
            Value::I32(3),
            Value::I32(3),
            Value::I32(2),
            Value::I32(0),
        ]),
    )?;
    let mut tokens: Vec<Value> = (0..=255u8)
        .map(|b| Value::String(byte_char(b).to_string()))
        .collect();
    tokens.extend(SPECIAL_TOKENS.iter().map(|t| Value::String(t.to_string())));
    tokens.extend((0..cfg.extra_tokens).map(|i| Value::String(format!("<|extra_{i}|>"))));
    tokens.extend(cfg.user_tokens.iter().map(|t| Value::String(t.to_string())));
    let mut kinds = vec![Value::I32(1); 256];
    kinds.extend(SPECIAL_TOKENS.iter().map(|_| Value::I32(3)));
    kinds.extend((0..cfg.extra_tokens).map(|_| Value::I32(3)));
    kinds.extend(cfg.user_tokens.iter().map(|_| Value::I32(4)));
    if let Some(template) = cfg.chat_template {
        w.set("tokenizer.chat_template", &Value::String(template.into()))?;
    }
    w.set("tokenizer.ggml.model", &Value::String("gpt2".into()))?;
    w.set("tokenizer.ggml.pre", &Value::String("qwen2".into()))?;
    w.set("tokenizer.ggml.tokens", &Value::Array(tokens))?;
    w.set("tokenizer.ggml.token_type", &Value::Array(kinds))?;
    // One merge whose result is not a token: never applied, keeps every text byte-level.
    w.set(
        "tokenizer.ggml.merges",
        &Value::Array(vec![Value::String("\u{120} \u{120}".into())]),
    )?;
    w.set("tokenizer.ggml.eos_token_id", &u32v(EOS))?;
    w.set("tokenizer.ggml.add_bos_token", &Value::Bool(false))?;

    let nvfp4_names: Vec<String> = match cfg.nvfp4 {
        None => Vec::new(),
        Some(_) => std::iter::once("output.weight".to_string())
            .chain((0..cfg.n_layer).flat_map(|il| {
                ["ffn_gate", "ffn_up", "ffn_down"].map(|n| format!("blk.{il}.{n}.weight"))
            }))
            .collect(),
    };
    if !nvfp4_names.is_empty() {
        w.set(
            "general.tensor_extra.name",
            &Value::Array(nvfp4_names.iter().cloned().map(Value::String).collect()),
        )?;
        w.set(
            "general.tensor_extra.prec_a4",
            &Value::Array(vec![Value::Bool(false); nvfp4_names.len()]),
        )?;
    }
    let mut fp4_rng = Rng::new(cfg.seed ^ 0x4e56_4650_3400);
    let mut add = |name: &str, shape: &[i64], data: Vec<u8>| -> Result<(), GgufError> {
        match cfg.nvfp4 {
            Some(q) if nvfp4_names.iter().any(|n| n == name) => {
                let k = shape[0];
                assert_eq!(k % 64, 0, "{name}: NVFP4 rows hold whole 64-element blocks");
                let data = nvfp4_random(&mut fp4_rng, (k / 64 * shape[1]) as usize, q);
                w.add_tensor(name, types::NVFP4, shape, &data)?;
                if let Some(e) = q.scale_exp {
                    let base = name.strip_suffix(".weight").unwrap();
                    let s = 2f32.powi(e as i32).to_ne_bytes();
                    w.add_tensor(&format!("{base}.scale"), types::F32, &[1], &s)?;
                    let one = 1f32.to_ne_bytes();
                    w.add_tensor(&format!("{base}.input_scale"), types::F32, &[1], &one)?;
                }
            }
            _ => {
                w.add_tensor(name, types::F32, shape, &data)?;
            }
        }
        Ok(())
    };
    // Norm weights near 1.
    let ones = |n: i64, r: &mut dyn FnMut(usize, f32) -> Vec<u8>| -> Vec<u8> {
        r(n as usize, 0.1)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| (1.0 + f32::from_ne_bytes(*c)).to_ne_bytes())
            .collect()
    };
    let mat = |r: &mut dyn FnMut(usize, f32) -> Vec<u8>, k: i64, n: i64| {
        r((k * n) as usize, 1.0 / (k as f32).sqrt())
    };

    add(
        "token_embd.weight",
        &[e, n_vocab as i64],
        uniform((e * n_vocab as i64) as usize, 1.0),
    )?;
    add("output_norm.weight", &[e], ones(e, &mut uniform))?;
    add(
        "output.weight",
        &[e, n_vocab as i64],
        mat(&mut uniform, e, n_vocab as i64),
    )?;
    let ff = cfg.n_ff as i64;
    let (hd, nh, nkv) = (HEAD_DIM as i64, N_HEAD as i64, N_HEAD_KV as i64);
    for il in 0..cfg.n_layer {
        let t = |n: &str| format!("blk.{il}.{n}");
        add(&t("attn_norm.weight"), &[e], ones(e, &mut uniform))?;
        add(
            &t("post_attention_norm.weight"),
            &[e],
            ones(e, &mut uniform),
        )?;
        add(&t("ffn_gate.weight"), &[e, ff], mat(&mut uniform, e, ff))?;
        add(&t("ffn_up.weight"), &[e, ff], mat(&mut uniform, e, ff))?;
        add(&t("ffn_down.weight"), &[ff, e], mat(&mut uniform, ff, e))?;
        if (il + 1) % 4 != 0 {
            let nv = N_V as i64;
            add(
                &t("attn_qkv.weight"),
                &[e, conv_dim],
                mat(&mut uniform, e, conv_dim),
            )?;
            add(
                &t("attn_gate.weight"),
                &[e, d_inner],
                mat(&mut uniform, e, d_inner),
            )?;
            add(&t("ssm_beta.weight"), &[e, nv], mat(&mut uniform, e, nv))?;
            add(&t("ssm_alpha.weight"), &[e, nv], mat(&mut uniform, e, nv))?;
            add(&t("ssm_dt.bias"), &[nv], uniform(nv as usize, 0.5))?;
            // A = -exp(A_log) < 0: the state decays.
            let a: Vec<u8> = uniform(nv as usize, 0.5)
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|c| (-1.0 - f32::from_ne_bytes(*c)).to_ne_bytes())
                .collect();
            add(&t("ssm_a"), &[nv], a)?;
            add(
                &t("ssm_conv1d.weight"),
                &[D_CONV as i64, conv_dim],
                uniform((D_CONV as i64 * conv_dim) as usize, 0.5),
            )?;
            add(
                &t("ssm_norm.weight"),
                &[D_STATE as i64],
                ones(D_STATE as i64, &mut uniform),
            )?;
            add(
                &t("ssm_out.weight"),
                &[d_inner, e],
                mat(&mut uniform, d_inner, e),
            )?;
        } else {
            add(
                &t("attn_q.weight"),
                &[e, 2 * hd * nh],
                mat(&mut uniform, e, 2 * hd * nh),
            )?;
            add(
                &t("attn_k.weight"),
                &[e, hd * nkv],
                mat(&mut uniform, e, hd * nkv),
            )?;
            add(
                &t("attn_v.weight"),
                &[e, hd * nkv],
                mat(&mut uniform, e, hd * nkv),
            )?;
            add(&t("attn_q_norm.weight"), &[hd], ones(hd, &mut uniform))?;
            add(&t("attn_k_norm.weight"), &[hd], ones(hd, &mut uniform))?;
            add(
                &t("attn_output.weight"),
                &[hd * nh, e],
                mat(&mut uniform, hd * nh, e),
            )?;
        }
    }
    if cfg.mtp {
        mtp_head_tensors(cfg, &mut |name, shape, data| {
            w.add_tensor(name, types::F32, shape, &data).map(|_| ())
        })?;
    }
    w.write(path)
}

/// Writes the MTP head of the tiny model of `cfg` as a GGUF of its own (block `n_layer` only,
/// `nextn_predict_layers` 1): the same head [`write`] puts in the model's file with `mtp`.
pub fn write_mtp_head(path: impl AsRef<Path>, cfg: &TinyConfig) -> Result<(), GgufError> {
    let mut w = GgufWriter::new();
    let arch = "qwen35";
    w.set("general.architecture", &Value::String(arch.into()))?;
    w.set("general.name", &Value::String("tiny-qwen35-mtp".into()))?;
    for (k, v) in [
        ("block_count", cfg.n_layer + 1),
        ("nextn_predict_layers", 1),
        ("embedding_length", cfg.n_embd),
    ] {
        w.set(&format!("{arch}.{k}"), &Value::U32(v))?;
    }
    mtp_head_tensors(cfg, &mut |name, shape, data| {
        w.add_tensor(name, types::F32, shape, &data).map(|_| ())
    })?;
    w.write(path)
}

/// Adds a tensor (name, shape, F32 bytes) to a file being written.
type AddTensor<'a> = dyn FnMut(&str, &[i64], Vec<u8>) -> Result<(), GgufError> + 'a;

/// The tiny MTP head's tensors (block `n_layer`: a gated attention block with its FFN, and the
/// `nextn` projection and norms), from a random stream of their own.
fn mtp_head_tensors(cfg: &TinyConfig, add: &mut AddTensor<'_>) -> Result<(), GgufError> {
    let mut rng = Rng::new(cfg.seed ^ 0x006d_7470_6865_6164);
    let mut uniform = move |n: usize, a: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|_| ((rng.next_f64() as f32 * 2.0 - 1.0) * a).to_ne_bytes())
            .collect()
    };
    let ones = |n: i64, r: &mut dyn FnMut(usize, f32) -> Vec<u8>| -> Vec<u8> {
        r(n as usize, 0.1)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| (1.0 + f32::from_ne_bytes(*c)).to_ne_bytes())
            .collect()
    };
    let mat = |r: &mut dyn FnMut(usize, f32) -> Vec<u8>, k: i64, n: i64| {
        r((k * n) as usize, 1.0 / (k as f32).sqrt())
    };
    let (e, ff) = (cfg.n_embd as i64, cfg.n_ff as i64);
    let (hd, nh, nkv) = (HEAD_DIM as i64, N_HEAD as i64, N_HEAD_KV as i64);
    let t = |n: &str| format!("blk.{}.{n}", cfg.n_layer);
    add(
        &t("nextn.eh_proj.weight"),
        &[2 * e, e],
        mat(&mut uniform, 2 * e, e),
    )?;
    add(&t("nextn.enorm.weight"), &[e], ones(e, &mut uniform))?;
    add(&t("nextn.hnorm.weight"), &[e], ones(e, &mut uniform))?;
    add(
        &t("nextn.shared_head_norm.weight"),
        &[e],
        ones(e, &mut uniform),
    )?;
    add(&t("attn_norm.weight"), &[e], ones(e, &mut uniform))?;
    add(
        &t("post_attention_norm.weight"),
        &[e],
        ones(e, &mut uniform),
    )?;
    add(&t("ffn_gate.weight"), &[e, ff], mat(&mut uniform, e, ff))?;
    add(&t("ffn_up.weight"), &[e, ff], mat(&mut uniform, e, ff))?;
    add(&t("ffn_down.weight"), &[ff, e], mat(&mut uniform, ff, e))?;
    add(
        &t("attn_q.weight"),
        &[e, 2 * hd * nh],
        mat(&mut uniform, e, 2 * hd * nh),
    )?;
    add(
        &t("attn_k.weight"),
        &[e, hd * nkv],
        mat(&mut uniform, e, hd * nkv),
    )?;
    add(
        &t("attn_v.weight"),
        &[e, hd * nkv],
        mat(&mut uniform, e, hd * nkv),
    )?;
    add(&t("attn_q_norm.weight"), &[hd], ones(hd, &mut uniform))?;
    add(&t("attn_k_norm.weight"), &[hd], ones(hd, &mut uniform))?;
    add(
        &t("attn_output.weight"),
        &[hd * nh, e],
        mat(&mut uniform, hd * nh, e),
    )?;
    Ok(())
}

/// `blocks` random NVFP4 blocks (`block_nvfp4`: four UE4M3 sub-block scales, then 32 bytes of
/// E2M1 codes). Scales stay within 2^-7..2^-5 before the shift, so values are about as large as
/// the F32 weights' and a shift of a few never overflows E4M3.
fn nvfp4_random(rng: &mut Rng, blocks: usize, q: TinyNvfp4) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks * 36);
    for _ in 0..blocks {
        for _ in 0..4 {
            let r = rng.next_u64();
            let exp = 1 + (r % 3) as u8 + q.block_exp_shift;
            let man = ((r >> 8) % 8) as u8;
            assert!(exp < 15, "block scale exponent overflows");
            out.push(exp << 3 | man);
        }
        let r = rng.next_u64().to_ne_bytes();
        let r2 = rng.next_u64().to_ne_bytes();
        let r3 = rng.next_u64().to_ne_bytes();
        let r4 = rng.next_u64().to_ne_bytes();
        out.extend(r.iter().chain(&r2).chain(&r3).chain(&r4));
    }
    out
}

/// Byte-level tokenizer metadata (see the module docs).
pub(crate) fn byte_tokenizer(w: &mut GgufWriter) -> Result<(), GgufError> {
    let mut tokens: Vec<Value> = (0..=255u8)
        .map(|b| Value::String(byte_char(b).to_string()))
        .collect();
    tokens.extend(SPECIAL_TOKENS.iter().map(|t| Value::String(t.to_string())));
    let mut kinds = vec![Value::I32(1); 256];
    kinds.extend(SPECIAL_TOKENS.iter().map(|_| Value::I32(3)));
    w.set("tokenizer.ggml.model", &Value::String("gpt2".into()))?;
    w.set("tokenizer.ggml.pre", &Value::String("qwen2".into()))?;
    w.set("tokenizer.ggml.tokens", &Value::Array(tokens))?;
    w.set("tokenizer.ggml.token_type", &Value::Array(kinds))?;
    w.set(
        "tokenizer.ggml.merges",
        &Value::Array(vec![Value::String("\u{120} \u{120}".into())]),
    )?;
    w.set("tokenizer.ggml.eos_token_id", &Value::U32(EOS))?;
    w.set("tokenizer.ggml.add_bos_token", &Value::Bool(false))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_matches_gpt2() {
        assert_eq!(byte_char(b'a'), 'a');
        assert_eq!(byte_char(b' '), '\u{120}');
        assert_eq!(byte_char(0), '\u{100}');
        assert_eq!(byte_char(0xAD), '\u{143}');
    }
}
