//! The MTP block of a GLM-5.3-Flash checkpoint: names, shapes and types as llama.cpp's
//! converter writes GLM's NextN layer, on a fake checkpoint holding only that layer (tiny dims,
//! the NVFP4 release's dtypes: BF16 everywhere but the router bias).

use std::path::Path;
use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::ggml::types;
use crate::gguf::Recipe;
use crate::safetensors::Checkpoint;

const E: usize = 8;
const H: usize = 2;
const Q: usize = 6;
const KV: usize = 4;
const K: usize = 4;
const V: usize = 4;
const NE: usize = 3;
const FF: usize = 4;
const IH: usize = 2;
const ID: usize = 4;
const KP: usize = 4;
/// The MTP layer's number: right after the decoder's layers.
const IL: u64 = 2;

/// A one-file safetensors checkpoint of `(name, dtype, shape)` zeros.
fn write_checkpoint(dir: &Path, tensors: &[(String, &str, Vec<usize>)]) {
    let mut header = serde_json::Map::new();
    let mut len = 0usize;
    for (name, dtype, shape) in tensors {
        let bytes = shape.iter().product::<usize>() * if *dtype == "F32" { 4 } else { 2 };
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [len, len + bytes]}),
        );
        len += bytes;
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.resize(out.len() + len, 0);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

/// The checkpoint's MTP layer `IL` as the NVFP4 release ships it (row-major shapes).
fn mtp_layer() -> Vec<(String, &'static str, Vec<usize>)> {
    let l = format!("model.language_model.layers.{IL}.");
    let mut t: Vec<(String, &str, Vec<usize>)> = Vec::new();
    let mut add = |name: &str, dtype: &'static str, shape: &[usize]| {
        t.push((format!("{l}{name}"), dtype, shape.to_vec()));
    };
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "enorm.weight",
        "hnorm.weight",
        "shared_head.norm.weight",
    ] {
        add(n, "BF16", &[E]);
    }
    add("eh_proj.weight", "BF16", &[E, 2 * E]);
    let a = "self_attn.";
    add(&format!("{a}q_a_proj.weight"), "BF16", &[Q, E]);
    add(&format!("{a}q_a_layernorm.weight"), "BF16", &[Q]);
    add(&format!("{a}q_b_proj.weight"), "BF16", &[H * K, Q]);
    add(&format!("{a}kv_a_proj_with_mqa.weight"), "BF16", &[KV, E]);
    add(&format!("{a}kv_a_layernorm.weight"), "BF16", &[KV]);
    add(&format!("{a}kv_b_proj.weight"), "BF16", &[H * (K + V), KV]);
    add(&format!("{a}o_proj.weight"), "BF16", &[E, H * V]);
    add(&format!("{a}indexer.wk.weight"), "BF16", &[ID, E]);
    add(&format!("{a}indexer.wq_b.weight"), "BF16", &[IH * ID, Q]);
    add(&format!("{a}indexer.k_norm.weight"), "BF16", &[ID]);
    add(&format!("{a}indexer.k_norm.bias"), "BF16", &[ID]);
    add(&format!("{a}indexer.weights_proj.weight"), "BF16", &[IH, E]);
    add(
        &format!("{a}indexer.index_kpool_compress_ape"),
        "BF16",
        &[KP, ID],
    );
    add(
        &format!("{a}indexer.index_kpool_compress_gate"),
        "BF16",
        &[ID, E],
    );
    add("mlp.gate.weight", "BF16", &[NE, E]);
    add("mlp.gate.e_score_correction_bias", "F32", &[NE]);
    for e in 0..NE {
        add(
            &format!("mlp.experts.{e}.gate_proj.weight"),
            "BF16",
            &[FF, E],
        );
        add(&format!("mlp.experts.{e}.up_proj.weight"), "BF16", &[FF, E]);
        add(
            &format!("mlp.experts.{e}.down_proj.weight"),
            "BF16",
            &[E, FF],
        );
    }
    add("mlp.shared_experts.gate_proj.weight", "BF16", &[FF, E]);
    add("mlp.shared_experts.up_proj.weight", "BF16", &[FF, E]);
    add("mlp.shared_experts.down_proj.weight", "BF16", &[E, FF]);
    t
}

#[test]
fn mtp_block_is_the_converters_nextn_layer() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path(), &mtp_layer());
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
    let dims = MtpDims {
        n_embd: E as u64,
        n_head: H as u64,
        n_expert: NE as u64,
        ff: FF as u64,
        kv_lora: KV as u64,
        k_dim: K as u64,
        v_dim: V as u64,
    };
    mtp_block(&mut b, "model.language_model.", IL, &dims).unwrap();
    let (bf, f) = (types::BF16, types::F32);
    let n = |x: usize| x as u64;
    // GGUF names, ggml shapes and types: the UD-Q4_K_XL GGUF's block 45 (its 16-bit tensors as
    // the converter writes them, before quantization).
    let mut want: Vec<(&str, _, Vec<u64>)> = vec![
        ("attn_k_b.weight", bf, vec![n(K), n(KV), n(H)]),
        ("attn_kv_a_mqa.weight", bf, vec![n(E), n(KV)]),
        ("attn_kv_a_norm.weight", f, vec![n(KV)]),
        ("attn_norm.weight", f, vec![n(E)]),
        ("attn_output.weight", bf, vec![n(H * V), n(E)]),
        ("attn_q_a.weight", bf, vec![n(E), n(Q)]),
        ("attn_q_a_norm.weight", f, vec![n(Q)]),
        ("attn_q_b.weight", bf, vec![n(Q), n(H * K)]),
        ("attn_v_b.weight", bf, vec![n(KV), n(V), n(H)]),
        ("exp_probs_b.bias", f, vec![n(NE)]),
        ("ffn_down_exps.weight", bf, vec![n(FF), n(E), n(NE)]),
        ("ffn_down_shexp.weight", bf, vec![n(FF), n(E)]),
        ("ffn_gate_exps.weight", bf, vec![n(E), n(FF), n(NE)]),
        ("ffn_gate_inp.weight", f, vec![n(E), n(NE)]),
        ("ffn_gate_shexp.weight", bf, vec![n(E), n(FF)]),
        ("ffn_norm.weight", f, vec![n(E)]),
        ("ffn_up_exps.weight", bf, vec![n(E), n(FF), n(NE)]),
        ("ffn_up_shexp.weight", bf, vec![n(E), n(FF)]),
        ("indexer.attn_k.weight", bf, vec![n(E), n(ID)]),
        ("indexer.attn_q_b.weight", bf, vec![n(Q), n(IH * ID)]),
        ("indexer.k_norm.bias", f, vec![n(ID)]),
        ("indexer.k_norm.weight", f, vec![n(ID)]),
        ("indexer.proj.weight", f, vec![n(E), n(IH)]),
        ("indexer_compressor_ape.weight", f, vec![n(ID), n(KP)]),
        ("indexer_compressor_gate.weight", bf, vec![n(E), n(ID)]),
        ("nextn.eh_proj.weight", bf, vec![n(2 * E), n(E)]),
        ("nextn.enorm.weight", f, vec![n(E)]),
        ("nextn.hnorm.weight", f, vec![n(E)]),
        ("nextn.shared_head_norm.weight", f, vec![n(E)]),
    ];
    want.sort_by_key(|w| w.0);
    let mut got: Vec<_> = b
        .tensors
        .iter()
        .map(|t| {
            let name = t.name.strip_prefix(&format!("blk.{IL}.")).unwrap();
            (name, t.ggml_type, t.shape.clone())
        })
        .collect();
    got.sort_by_key(|g| g.0);
    assert_eq!(got.len(), want.len(), "{got:?}");
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.0, w.0);
        assert_eq!(
            (crate::ggml::type_name(g.1), &g.2),
            (crate::ggml::type_name(w.1), &w.2),
            "{}",
            w.0
        );
    }
    // No hyper-connections: the block is a plain residual.
    assert!(b.tensors.iter().all(|t| !t.name.contains(".hc_")));
    // The routed experts stay BF16, stacked from the shards as shipped (no NVFP4 sources).
    let name = format!("blk.{IL}.ffn_gate_exps.weight");
    let Some(Recipe::Gather(parts)) = b.recipes.get(&name) else {
        panic!("{name}: not a gather");
    };
    assert_eq!(
        parts.iter().map(|p| p.len).sum::<u64>(),
        (NE * FF * E * 2) as u64
    );
    assert!(b.stacks.is_empty());
}

/// A one-file safetensors checkpoint of `(name, dtype, shape, bytes)`.
fn write_checkpoint_data(dir: &Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [data.len(), data.len() + bytes.len()]}),
        );
        data.extend_from_slice(bytes);
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

/// `n` BF16 values in [-0.05, 0.05) from a fixed stream.
fn bf16_values(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 0.05;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// Over two nodes the MTP layer's BF16 routed experts load as NVFP4 (#560 round 4): the
/// decoder's layout (ggml blocks plus a per-expert `.scale`), each value within one unit of its
/// 16-element block's scale and the largest exact; the shared expert stays BF16.
#[test]
fn bf16_experts_load_as_nvfp4_when_asked() {
    const E: usize = 128;
    const FF: usize = 64;
    const NE: usize = 2;
    let l = "model.language_model.layers.3.";
    let s = |t: &str| format!("{l}{t}");
    let d = |t: &str| format!("blk.3.{t}");
    let mut t: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut seed = 1;
    let mut add = |name: String, shape: Vec<usize>| {
        seed += 1;
        let n = shape.iter().product();
        t.push((name, "BF16", shape, bf16_values(n, seed)));
    };
    add(s("mlp.gate.weight"), vec![NE, E]);
    for e in 0..NE {
        add(s(&format!("mlp.experts.{e}.gate_proj.weight")), vec![FF, E]);
        add(s(&format!("mlp.experts.{e}.up_proj.weight")), vec![FF, E]);
        add(s(&format!("mlp.experts.{e}.down_proj.weight")), vec![E, FF]);
    }
    for (h, shape) in [
        ("gate_proj", vec![FF, E]),
        ("up_proj", vec![FF, E]),
        ("down_proj", vec![E, FF]),
    ] {
        add(s(&format!("mlp.shared_experts.{h}.weight")), shape);
    }
    t.push((
        s("mlp.gate.e_score_correction_bias"),
        "F32",
        vec![NE],
        vec![0; 4 * NE],
    ));
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint_data(dir.path(), &t);
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
    moe(&mut b, &s, &d, E as u64, NE as u64, FF as u64, true).unwrap();
    let to_f32 = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    for (g, h, k, rows) in [
        ("gate", "gate_proj", E, FF),
        ("up", "up_proj", E, FF),
        ("down", "down_proj", FF, E),
    ] {
        let name = d(&format!("ffn_{g}_exps.weight"));
        let info = b.tensors.iter().find(|t| t.name == name).unwrap();
        assert_eq!(info.ggml_type, types::NVFP4, "{name}");
        assert_eq!(info.shape, vec![k as u64, rows as u64, NE as u64]);
        let Some(Recipe::Computed(w)) = b.recipes.get(&name) else {
            panic!("{name}: not computed");
        };
        let Some(Recipe::Computed(sc)) = b.recipes.get(&d(&format!("ffn_{g}_exps.scale"))) else {
            panic!("{name}: no computed scale");
        };
        let w = w().unwrap();
        assert_eq!(w.len() as u64, info.size);
        let sc: Vec<f32> = sc()
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(sc.len(), NE);
        let per = w.len() / NE;
        for e in 0..NE {
            let x = to_f32(
                b.ckpt
                    .data(&s(&format!("mlp.experts.{e}.{h}.weight")))
                    .unwrap(),
            );
            let q = crate::ggml::dequantize(types::NVFP4, &w[e * per..(e + 1) * per], rows * k);
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert_eq!(sc[e], amax / 2688.0);
            for (i, (&a, &v)) in x.iter().zip(&q).enumerate() {
                let block = &x[i / 16 * 16..i / 16 * 16 + 16];
                let bmax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let y = v * sc[e];
                assert!(
                    (a - y).abs() <= bmax / 6.0 * 1.07,
                    "{name} expert {e} [{i}]: {a} -> {y}"
                );
                if a.abs() == amax {
                    assert!(
                        (a - y).abs() <= amax * 1e-6,
                        "{name} expert {e}: max {a} -> {y}"
                    );
                }
            }
        }
    }
    let shexp = b
        .tensors
        .iter()
        .find(|t| t.name == d("ffn_gate_shexp.weight"))
        .unwrap();
    assert_eq!(shexp.ggml_type, types::BF16);
    // No SSD-tier sources: the NVFP4 bytes are made at load.
    assert!(b.stacks.is_empty());
}

/// Experts not of whole 64-element blocks stay BF16 even when asked.
#[test]
fn odd_sized_experts_stay_bf16() {
    let dir = tempfile::tempdir().unwrap();
    write_checkpoint(dir.path(), &mtp_layer());
    let mut b = Builder::new(Arc::new(Checkpoint::open(dir.path()).unwrap()));
    b.mtp_nvfp4 = true;
    let dims = MtpDims {
        n_embd: E as u64,
        n_head: H as u64,
        n_expert: NE as u64,
        ff: FF as u64,
        kv_lora: KV as u64,
        k_dim: K as u64,
        v_dim: V as u64,
    };
    mtp_block(&mut b, "model.language_model.", IL, &dims).unwrap();
    let t = b
        .tensors
        .iter()
        .find(|t| t.name == format!("blk.{IL}.ffn_up_exps.weight"))
        .unwrap();
    assert_eq!(t.ggml_type, types::BF16);
}

#[test]
fn nextn_layers_follow_the_checkpoint() {
    let tc = json!({"num_nextn_predict_layers": 1});
    assert_eq!(nextn_layers(&tc, true), 1);
    // A config naming an MTP layer the shards do not have (a truncated copy): none.
    assert_eq!(nextn_layers(&tc, false), 0);
    assert_eq!(nextn_layers(&json!({}), true), 0);
    // The head drafts with one block.
    assert_eq!(
        nextn_layers(&json!({"num_nextn_predict_layers": 3}), true),
        1
    );
}
