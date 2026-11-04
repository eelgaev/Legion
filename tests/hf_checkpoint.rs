//! Hugging Face checkpoints opened as synthetic GGUF models (`src/hf/`).
//!
//! The real-model check needs a checkpoint and the GGUF llama.cpp's converter writes from it:
//!
//! ```sh
//! python convert_hf_to_gguf.py <dir> --outtype bf16 --outfile ref.gguf
//! LEGION_TEST_HF_DIR=<dir> LEGION_TEST_HF_REF=ref.gguf \
//!     cargo test --release --test hf_checkpoint -- --ignored
//! ```

use std::path::Path;

use legion::ggml::{Backend, types};
use legion::gguf::{GgufFile, Value};
use legion::model::core::{Hparams, Model};
use legion::tokenizer::Tokenizer;
use legion::weights::Weights;
use serde_json::json;

/// Metadata keys whose values are allowed to differ: provenance only.
fn provenance(key: &str) -> bool {
    key.starts_with("general.")
}

/// Compares two values; float arrays and scalars must match bit for bit.
fn same(a: &Value, b: &Value) -> bool {
    a == b
}

#[test]
#[ignore = "needs a HF checkpoint (LEGION_TEST_HF_DIR) and its converted GGUF (LEGION_TEST_HF_REF)"]
fn checkpoint_matches_the_converted_gguf() {
    let (Ok(dir), Ok(reference)) = (
        std::env::var("LEGION_TEST_HF_DIR"),
        std::env::var("LEGION_TEST_HF_REF"),
    ) else {
        eprintln!("skipping: LEGION_TEST_HF_DIR / LEGION_TEST_HF_REF not set");
        return;
    };
    // The converter dequantizes FP8 weights to BF16; compare like with like.
    let bf16 = legion::hf::OpenOptions {
        fp8: legion::hf::Fp8Mode::Bf16,
    };
    let ours = GgufFile::open_with(&dir, &bf16).unwrap();
    let want = GgufFile::open(&reference).unwrap();
    assert!(ours.is_synthetic());

    // Metadata: every non-provenance key of the reference, with the same value and type.
    let mut problems = Vec::new();
    for (key, value) in want.metadata() {
        if provenance(&key) {
            continue;
        }
        match ours.get(&key) {
            None => problems.push(format!("missing key {key}")),
            Some(v) if !same(&v, &value) => problems.push(format!(
                "key {key}: {} vs reference {}",
                short(&v),
                short(&value)
            )),
            Some(_) => {}
        }
    }
    for key in ours.keys() {
        if !provenance(&key) && want.get(&key).is_none() {
            problems.push(format!("extra key {key}"));
        }
    }

    // Tensor infos.
    let mut theirs: std::collections::BTreeMap<String, _> = want
        .tensors()
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    for t in ours.tensors() {
        match theirs.remove(&t.name) {
            None => problems.push(format!("extra tensor {}", t.name)),
            Some(r) => {
                if r.ggml_type != t.ggml_type || r.shape != t.shape {
                    problems.push(format!(
                        "tensor {}: {} {:?} vs reference {} {:?}",
                        t.name,
                        t.type_name(),
                        t.shape,
                        r.type_name(),
                        r.shape
                    ));
                }
            }
        }
    }
    problems.extend(theirs.keys().map(|n| format!("missing tensor {n}")));
    assert!(
        problems.is_empty(),
        "{} differences:\n{}",
        problems.len(),
        problems.join("\n")
    );

    // Tensor bytes, loaded through each model's own path.
    let mut ssm_a_off = false;
    let backend = Backend::cpu().unwrap();
    let a = Weights::load(&ours, |_| backend.buffer_type()).unwrap();
    let b = Weights::load(&want, |_| backend.buffer_type()).unwrap();
    for t in want.tensors() {
        let (x, y) = (
            a.get(&t.name).unwrap().bytes(),
            b.get(&t.name).unwrap().bytes(),
        );
        // `-exp(A_log)` is computed correctly rounded here; the converter's torch exp is 1 ulp
        // off on rare inputs (and differs between CPU builds), so `ssm_a` may differ by 1 ulp.
        if t.name.ends_with(".ssm_a") && x.len() == y.len() {
            let ulps = x
                .as_chunks::<4>()
                .0
                .iter()
                .zip(y.as_chunks::<4>().0.iter())
                .map(|(p, q)| {
                    let (p, q) = (i32::from_le_bytes(*p), i32::from_le_bytes(*q));
                    (p - q).unsigned_abs()
                })
                .collect::<Vec<_>>();
            let off = ulps.iter().filter(|&&u| u > 0).count();
            if off > 0 {
                eprintln!("{}: {off} value(s) 1 ulp from the converter's", t.name);
                ssm_a_off = true;
            }
            if ulps.iter().any(|&u| u > 1) {
                problems.push(format!("tensor {} differs by more than 1 ulp", t.name));
            }
            continue;
        }
        if x != y {
            let first = x.iter().zip(&y).position(|(p, q)| p != q).unwrap_or(0);
            problems.push(format!(
                "tensor {} bytes differ (first at byte {first})",
                t.name
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} differences:\n{}",
        problems.len(),
        problems.join("\n")
    );
    drop((a, b));

    // Prompt logits on the CPU, every position: bit-identical (measured on Qwen3.5-4B, even with
    // the 1-ulp `ssm_a` value).
    let text = "<|im_start|>user\nWhat is the capital of France, and why?<|im_end|>\n<|im_start|>assistant\n";
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    let mut runs = Vec::new();
    for g in [&ours, &want] {
        let ids = Tokenizer::from_gguf(g)
            .unwrap()
            .encode_with(text, false, true);
        let model = Model::load(g, |_| backend.buffer_type()).unwrap();
        runs.push((
            ids.clone(),
            model.prompt_logits_all(&mut backend, &ids).unwrap(),
        ));
    }
    assert_eq!(runs[0].0, runs[1].0, "tokenization differs");
    if runs[0].1 == runs[1].1 {
        return;
    }
    // A 1-ulp `ssm_a` value moves the logits by rounding, which the Q8_0 activations of the
    // quantized matmuls amplify layer by layer (measured on Nemotron 3.5 30B-A3B: max |diff|
    // 0.73, same argmax at every position); anything else must be bit-identical.
    assert!(ssm_a_off, "prompt logits differ");
    let n_vocab = runs[0].1.len() / runs[0].0.len();
    let argmax = |r: &[f32]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
    let mut max_diff = 0f32;
    for (a, b) in runs[0].1.chunks(n_vocab).zip(runs[1].1.chunks(n_vocab)) {
        assert_eq!(argmax(a), argmax(b), "top-1 differs");
        for (x, y) in a.iter().zip(b) {
            max_diff = max_diff.max((x - y).abs());
        }
    }
    eprintln!("prompt logits: max |diff| {max_diff} (from the 1-ulp ssm_a values)");
    assert!(max_diff < 2.5, "prompt logits differ by {max_diff}");
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 120 {
        format!("{}...", &s[..120])
    } else {
        s
    }
}

// ---------------------------------------------------------------------------------------------
// A tiny synthetic Qwen3.5 checkpoint: 2 layers (DeltaNet, then full attention), 2 key heads and
// 4 value heads (so the value heads are reordered), a BPE tokenizer with added and padded tokens.

const EMBD: usize = 8;
const NK: usize = 2;
const NV: usize = 4;
const DK: usize = 4;
const DV: usize = 4;
const KERNEL: usize = 4;
const FF: usize = 16;
const HEAD: usize = 4;
const VOCAB: usize = 300;

/// Deterministic pseudo-random floats in [-1, 1).
fn rand_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bf16(v: &[f32]) -> Vec<u8> {
    // Values are generated as exact BF16 (low half zero), so truncation is exact.
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn as_bf16(v: Vec<f32>) -> Vec<f32> {
    v.into_iter()
        .map(|x| f32::from_bits(x.to_bits() & 0xffff_0000))
        .collect()
}

fn f32le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct T {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn write_safetensors(path: &Path, tensors: &[T]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for t in tensors {
        let bytes = if t.dtype == "BF16" {
            bf16(&t.values)
        } else {
            f32le(&t.values)
        };
        header.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
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

/// Writes the checkpoint; returns its tensors by name.
fn tiny_checkpoint(dir: &Path) -> std::collections::HashMap<String, T> {
    let mut ts = Vec::new();
    let mut seed = 1;
    let mut t = |name: &str, dtype: &'static str, shape: &[usize]| {
        seed += 1;
        let n = shape.iter().product();
        let mut v = as_bf16(rand_f32(n, seed));
        if name.ends_with("A_log") {
            v = v.into_iter().map(|x| x * 0.5).collect();
        }
        ts.push(T {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            values: v,
        });
    };
    let p = "model.language_model.";
    t(&format!("{p}embed_tokens.weight"), "BF16", &[VOCAB, EMBD]);
    t(&format!("{p}norm.weight"), "BF16", &[EMBD]);
    for il in 0..2 {
        let l = format!("{p}layers.{il}.");
        t(&format!("{l}input_layernorm.weight"), "BF16", &[EMBD]);
        t(
            &format!("{l}post_attention_layernorm.weight"),
            "BF16",
            &[EMBD],
        );
        t(&format!("{l}mlp.gate_proj.weight"), "BF16", &[FF, EMBD]);
        t(&format!("{l}mlp.up_proj.weight"), "BF16", &[FF, EMBD]);
        t(&format!("{l}mlp.down_proj.weight"), "BF16", &[EMBD, FF]);
        if il == 0 {
            let a = format!("{l}linear_attn.");
            t(
                &format!("{a}in_proj_qkv.weight"),
                "BF16",
                &[2 * NK * DK + NV * DV, EMBD],
            );
            t(&format!("{a}in_proj_z.weight"), "BF16", &[NV * DV, EMBD]);
            t(&format!("{a}in_proj_a.weight"), "BF16", &[NV, EMBD]);
            t(&format!("{a}in_proj_b.weight"), "BF16", &[NV, EMBD]);
            t(&format!("{a}A_log"), "F32", &[NV]);
            t(&format!("{a}dt_bias"), "BF16", &[NV]);
            t(
                &format!("{a}conv1d.weight"),
                "BF16",
                &[2 * NK * DK + NV * DV, 1, KERNEL],
            );
            t(&format!("{a}norm.weight"), "F32", &[DV]);
            t(&format!("{a}out_proj.weight"), "BF16", &[EMBD, NV * DV]);
        } else {
            let a = format!("{l}self_attn.");
            t(&format!("{a}q_proj.weight"), "BF16", &[2 * 2 * HEAD, EMBD]);
            t(&format!("{a}k_proj.weight"), "BF16", &[HEAD, EMBD]);
            t(&format!("{a}v_proj.weight"), "BF16", &[HEAD, EMBD]);
            t(&format!("{a}o_proj.weight"), "BF16", &[EMBD, 2 * HEAD]);
            t(&format!("{a}q_norm.weight"), "BF16", &[HEAD]);
            t(&format!("{a}k_norm.weight"), "BF16", &[HEAD]);
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &ts);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen3_5",
            "text_config": {
                "num_hidden_layers": 2, "hidden_size": EMBD, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": HEAD, "intermediate_size": FF,
                "linear_num_key_heads": NK, "linear_num_value_heads": NV,
                "linear_key_head_dim": DK, "linear_value_head_dim": DV,
                "linear_conv_kernel_dim": KERNEL,
                "layer_types": ["linear_attention", "full_attention"],
                "full_attention_interval": 2, "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6, "vocab_size": VOCAB, "tie_word_embeddings": true,
                "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                    "mrope_section": [1, 1, 0]}
            }
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]]},
            "added_tokens": [
                {"id": 3, "content": "<|im_end|>", "special": true},
                {"id": 4, "content": "hello", "special": false}
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": null,
            "post_processor": {"type": "ByteLevel"}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({
            "eos_token": "<|im_end|>",
            "added_tokens_decoder": {"5": {"content": "<|x|>", "special": true}},
            "chat_template": "{{ messages[0]['content'] }}"
        }))
        .unwrap(),
    )
    .unwrap();
    ts.into_iter().map(|t| (t.name.clone(), t)).collect()
}

/// Destination value head `j` (tiled order) comes from this source head (grouped order).
fn src_head(j: usize) -> usize {
    let (v, k) = (j / NK, j % NK);
    k * (NV / NK) + v
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn tiny_checkpoint_opens_as_the_converter_would_write_it() {
    let dir = tempfile::tempdir().unwrap();
    let src = tiny_checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    assert!(g.is_synthetic());

    // Metadata.
    let u = |k: &str| g.get_u64(&format!("qwen35.{k}")).unwrap();
    assert_eq!(g.architecture().as_deref(), Some("qwen35"));
    assert_eq!(
        (
            u("block_count"),
            u("embedding_length"),
            u("feed_forward_length")
        ),
        (2, 8, 16)
    );
    assert_eq!(
        (
            u("ssm.group_count"),
            u("ssm.time_step_rank"),
            u("ssm.inner_size")
        ),
        (2, 4, 16)
    );
    assert_eq!(
        (u("rope.dimension_count"), u("attention.key_length")),
        (2, 4)
    );
    assert_eq!(
        g.get("qwen35.rope.dimension_sections"),
        Some(Value::Array(vec![
            Value::I32(1),
            Value::I32(1),
            Value::I32(0),
            Value::I32(0)
        ]))
    );
    assert_eq!(
        g.get("qwen35.attention.recurrent_layers"),
        Some(Value::Array(vec![Value::Bool(true), Value::Bool(false)]))
    );
    let Some(Value::Array(tokens)) = g.get("tokenizer.ggml.tokens") else {
        panic!()
    };
    let Some(Value::Array(kinds)) = g.get("tokenizer.ggml.token_type") else {
        panic!()
    };
    let toks: Vec<&str> = tokens.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(
        &toks[..7],
        ["a", "b", "ab", "<|im_end|>", "hello", "<|x|>", "[PAD6]"]
    );
    assert_eq!(toks.len(), VOCAB);
    let kinds: Vec<u64> = kinds.iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(&kinds[..7], [1, 1, 1, 3, 4, 3, 5]);
    assert_eq!(g.get_u64("tokenizer.ggml.eos_token_id"), Some(3));
    assert_eq!(
        g.get("tokenizer.ggml.add_bos_token"),
        Some(Value::Bool(false))
    );
    assert_eq!(
        g.get("tokenizer.ggml.normalizer.nfc"),
        Some(Value::Bool(true))
    );
    assert_eq!(
        g.get_str("tokenizer.chat_template").as_deref(),
        Some("{{ messages[0]['content'] }}")
    );
    assert!(Tokenizer::from_gguf(&g).is_ok());
    let hp = Hparams::from_gguf(&g).unwrap();
    assert_eq!((hp.n_layer, hp.ssm_n_group, hp.ssm_dt_rank), (2, 2, 4));

    // Tensor bytes against naive versions of the converter's transforms.
    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let get = |n: &str| w.get(n).unwrap_or_else(|| panic!("no {n}"));
    let s = |n: &str| &src[&format!("model.language_model.{n}")];
    assert_eq!(
        get("token_embd.weight").bytes(),
        bf16(&s("embed_tokens.weight").values)
    );
    let plus1: Vec<f32> = s("norm.weight").values.iter().map(|v| v + 1.0).collect();
    assert_eq!(f32s(&get("output_norm.weight").bytes()), plus1);
    assert_eq!(
        get("blk.0.ffn_down.weight").bytes(),
        bf16(&s("layers.0.mlp.down_proj.weight").values)
    );
    assert_eq!(
        get("blk.1.attn_q.weight").bytes(),
        bf16(&s("layers.1.self_attn.q_proj.weight").values)
    );
    let qn: Vec<f32> = s("layers.1.self_attn.q_norm.weight")
        .values
        .iter()
        .map(|v| v + 1.0)
        .collect();
    assert_eq!(f32s(&get("blk.1.attn_q_norm.weight").bytes()), qn);

    let a = |n: &str| &s(&format!("layers.0.linear_attn.{n}")).values;
    // QKV: q and k rows unchanged, value-head row blocks reordered.
    let qkv = a("in_proj_qkv.weight");
    let qk = 2 * NK * DK;
    let mut want = qkv[..qk * EMBD].to_vec();
    for j in 0..NV {
        let h = src_head(j);
        want.extend_from_slice(&qkv[(qk + h * DV) * EMBD..(qk + (h + 1) * DV) * EMBD]);
    }
    assert_eq!(get("blk.0.attn_qkv.weight").bytes(), bf16(&want));
    let z = a("in_proj_z.weight");
    let want: Vec<f32> = (0..NV)
        .flat_map(|j| z[src_head(j) * DV * EMBD..(src_head(j) + 1) * DV * EMBD].to_vec())
        .collect();
    assert_eq!(get("blk.0.attn_gate.weight").bytes(), bf16(&want));
    let al = a("in_proj_a.weight");
    let want: Vec<f32> = (0..NV)
        .flat_map(|j| al[src_head(j) * EMBD..(src_head(j) + 1) * EMBD].to_vec())
        .collect();
    assert_eq!(get("blk.0.ssm_alpha.weight").bytes(), bf16(&want));
    let want: Vec<f32> = (0..NV)
        .map(|j| -((a("A_log")[src_head(j)] as f64).exp() as f32))
        .collect();
    assert_eq!(f32s(&get("blk.0.ssm_a").bytes()), want);
    let want: Vec<f32> = (0..NV).map(|j| a("dt_bias")[src_head(j)]).collect();
    assert_eq!(f32s(&get("blk.0.ssm_dt.bias").bytes()), want);
    // conv1d: [channels, 1, kernel] -> [kernel, channels] F32, value channels reordered.
    let conv = a("conv1d.weight");
    let mut want = conv[..qk * KERNEL].to_vec();
    for j in 0..NV {
        let h = src_head(j);
        want.extend_from_slice(&conv[(qk + h * DV) * KERNEL..(qk + (h + 1) * DV) * KERNEL]);
    }
    let c = get("blk.0.ssm_conv1d.weight");
    assert_eq!(
        (c.ty(), c.shape()),
        (types::F32, vec![KERNEL as i64, (qk + NV * DV) as i64])
    );
    assert_eq!(f32s(&c.bytes()), want);
    assert_eq!(
        f32s(&get("blk.0.ssm_norm.weight").bytes()),
        *a("norm.weight")
    );
    // out_proj: value-channel columns reordered within each row.
    let o = a("out_proj.weight");
    let want: Vec<f32> = (0..EMBD)
        .flat_map(|r| {
            (0..NV).flat_map(move |j| {
                let h = src_head(j);
                o[r * NV * DV + h * DV..r * NV * DV + (h + 1) * DV].to_vec()
            })
        })
        .collect();
    assert_eq!(get("blk.0.ssm_out.weight").bytes(), bf16(&want));

    // And the model runs.
    let mut backend = Backend::cpu().unwrap();
    let model = Model::load(&g, |_| backend.buffer_type()).unwrap();
    let logits = model.prompt_logits(&mut backend, &[0, 2, 4]).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// MoE: a fused `gate_up_proj` is split per expert, per-expert tensors are stacked, the router
/// and the shared-expert gate are widened to F32.
#[test]
fn moe_experts_are_split_and_stacked() {
    let dir = tempfile::tempdir().unwrap();
    let mut src = tiny_checkpoint(dir.path());
    const E: usize = 3;
    const EFF: usize = 6;
    let p = "model.language_model.";
    // Replace the dense FFNs with MoE blocks: layer 0 fused, layer 1 per expert.
    let mut ts: Vec<T> = src
        .drain()
        .map(|(_, t)| t)
        .filter(|t| !t.name.contains(".mlp."))
        .collect();
    let mut seed = 100;
    let mut add = |ts: &mut Vec<T>, name: String, dtype: &'static str, shape: &[usize]| {
        seed += 1;
        let n = shape.iter().product();
        ts.push(T {
            name,
            dtype,
            shape: shape.to_vec(),
            values: as_bf16(rand_f32(n, seed)),
        });
    };
    for il in 0..2 {
        let l = format!("{p}layers.{il}.mlp.");
        add(&mut ts, format!("{l}gate.weight"), "BF16", &[E, EMBD]);
        if il == 0 {
            add(
                &mut ts,
                format!("{l}experts.gate_up_proj"),
                "BF16",
                &[E, 2 * EFF, EMBD],
            );
            add(
                &mut ts,
                format!("{l}experts.down_proj"),
                "BF16",
                &[E, EMBD, EFF],
            );
        } else {
            for e in 0..E {
                add(
                    &mut ts,
                    format!("{l}experts.{e}.gate_proj.weight"),
                    "BF16",
                    &[EFF, EMBD],
                );
                add(
                    &mut ts,
                    format!("{l}experts.{e}.up_proj.weight"),
                    "BF16",
                    &[EFF, EMBD],
                );
                add(
                    &mut ts,
                    format!("{l}experts.{e}.down_proj.weight"),
                    "BF16",
                    &[EMBD, EFF],
                );
            }
        }
        add(
            &mut ts,
            format!("{l}shared_expert.gate_proj.weight"),
            "BF16",
            &[EFF, EMBD],
        );
        add(
            &mut ts,
            format!("{l}shared_expert.up_proj.weight"),
            "BF16",
            &[EFF, EMBD],
        );
        add(
            &mut ts,
            format!("{l}shared_expert.down_proj.weight"),
            "BF16",
            &[EMBD, EFF],
        );
        add(
            &mut ts,
            format!("{l}shared_expert_gate.weight"),
            "BF16",
            &[1, EMBD],
        );
    }
    write_safetensors(&dir.path().join("model.safetensors"), &ts);
    let mut cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("config.json")).unwrap()).unwrap();
    let tc = cfg["text_config"].as_object_mut().unwrap();
    tc.remove("intermediate_size");
    tc.insert("num_experts".into(), json!(E));
    tc.insert("num_experts_per_tok".into(), json!(2));
    tc.insert("moe_intermediate_size".into(), json!(EFF));
    tc.insert("shared_expert_intermediate_size".into(), json!(EFF));
    cfg["model_type"] = json!("qwen3_5_moe");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&cfg).unwrap(),
    )
    .unwrap();
    let src: std::collections::HashMap<String, T> =
        ts.into_iter().map(|t| (t.name.clone(), t)).collect();

    let g = GgufFile::open(dir.path()).unwrap();
    assert_eq!(g.architecture().as_deref(), Some("qwen35moe"));
    assert_eq!(g.get_u64("qwen35moe.expert_count"), Some(E as u64));
    assert_eq!(
        g.get_u64("qwen35moe.expert_feed_forward_length"),
        Some(EFF as u64)
    );
    assert_eq!(g.get("qwen35moe.feed_forward_length"), None);
    let backend = Backend::cpu().unwrap();
    let w = Weights::load(&g, |_| backend.buffer_type()).unwrap();
    let s = |n: &str| &src[&format!("{p}{n}")].values;

    // Layer 0, fused: gate is the first EFF rows of each expert, up the second.
    let gu = s("layers.0.mlp.experts.gate_up_proj");
    let half = |first: usize| -> Vec<f32> {
        (0..E)
            .flat_map(|e| {
                gu[(e * 2 * EFF + first) * EMBD..(e * 2 * EFF + first + EFF) * EMBD].to_vec()
            })
            .collect()
    };
    let gate = w.get("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(gate.shape(), vec![EMBD as i64, EFF as i64, E as i64]);
    assert_eq!(gate.bytes(), bf16(&half(0)));
    assert_eq!(
        w.get("blk.0.ffn_up_exps.weight").unwrap().bytes(),
        bf16(&half(EFF))
    );
    assert_eq!(
        w.get("blk.0.ffn_down_exps.weight").unwrap().bytes(),
        bf16(s("layers.0.mlp.experts.down_proj"))
    );
    // Layer 1, per expert: stacked in expert order.
    let stacked = |wn: &str| -> Vec<f32> {
        (0..E)
            .flat_map(|e| s(&format!("layers.1.mlp.experts.{e}.{wn}.weight")).clone())
            .collect()
    };
    assert_eq!(
        w.get("blk.1.ffn_gate_exps.weight").unwrap().bytes(),
        bf16(&stacked("gate_proj"))
    );
    assert_eq!(
        w.get("blk.1.ffn_down_exps.weight").unwrap().bytes(),
        bf16(&stacked("down_proj"))
    );
    // Router and shared-expert gate: F32, the latter trimmed to [EMBD].
    let r = w.get("blk.0.ffn_gate_inp.weight").unwrap();
    assert_eq!(
        (r.ty(), r.shape()),
        (types::F32, vec![EMBD as i64, E as i64])
    );
    assert_eq!(f32s(&r.bytes()), *s("layers.0.mlp.gate.weight"));
    let sg = w.get("blk.1.ffn_gate_inp_shexp.weight").unwrap();
    assert_eq!((sg.ty(), sg.shape()), (types::F32, vec![EMBD as i64]));
    assert_eq!(
        w.get("blk.1.ffn_up_shexp.weight").unwrap().bytes(),
        bf16(s("layers.1.mlp.shared_expert.up_proj.weight"))
    );
}

/// A checkpoint's `generation_config.json` becomes `general.sampling.*` keys, which the server
/// reads as the model's sampling defaults; without the file there are none.
#[test]
fn generation_config_gives_the_sampling_keys() {
    use legion::engine::sampling_defaults::{ModelSampling, SamplingLayer};
    let dir = tempfile::tempdir().unwrap();
    tiny_checkpoint(dir.path());
    let g = GgufFile::open(dir.path()).unwrap();
    assert!(ModelSampling::from_gguf(&g).file.is_empty());
    std::fs::write(
        dir.path().join("generation_config.json"),
        json!({"do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0})
            .to_string(),
    )
    .unwrap();
    let g = GgufFile::open(dir.path()).unwrap();
    assert_eq!(g.get("general.sampling.top_k"), Some(Value::I32(20)));
    assert_eq!(
        ModelSampling::from_gguf(&g).file,
        SamplingLayer {
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            ..Default::default()
        }
    );
}
