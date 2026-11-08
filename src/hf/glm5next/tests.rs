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
