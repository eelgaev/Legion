//! glm5-next hyperparameters from GGUF metadata (the layer stack is checked against llama.cpp
//! by `tests/glm5next.rs`).

use super::*;
use crate::gguf::{GgufWriter, Value};

#[test]
fn per_layer_keys_take_arrays_or_one_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("keys.gguf");
    let mut w = GgufWriter::new();
    w.set("a", &Value::F32(10.0)).unwrap();
    w.set(
        "b",
        &Value::Array(vec![Value::F32(1.0), Value::F32(2.0), Value::F32(3.0)]),
    )
    .unwrap();
    w.write(&path).unwrap();
    let g = GgufFile::open(&path).unwrap();
    assert_eq!(per_layer_f(&g, "a", 3).unwrap(), vec![10.0; 3]);
    assert_eq!(per_layer_f(&g, "b", 3).unwrap(), vec![1.0, 2.0, 3.0]);
    // Too short for the layers, or missing: a metadata error.
    assert!(per_layer_f(&g, "b", 4).is_err());
    assert!(per_layer_f(&g, "c", 1).is_err());
}

#[test]
fn not_a_glm5_model_has_no_params() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arch.gguf");
    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("glm5-next".into()))
        .unwrap();
    w.write(&path).unwrap();
    let g = GgufFile::open(&path).unwrap();
    // Required keys missing: the parser names one instead of guessing.
    assert!(matches!(parse(&g), Err(ModelError::Metadata(_))));
}

/// The arch hooks the MTP head runs by: the NextN layer is a DSA (indexed) layer, the head
/// reads one stream, pages sparsely, and the recurrent layers commit through KDA replays.
#[test]
fn mtp_hooks_take_the_nextn_layer_as_a_dsa_layer() {
    let p = Glm5Params {
        // Three decoder layers (KDA, KDA, DSA), then the NextN layer.
        mla: vec![false, false, true, true],
        n_dense: 1,
        n_ff: 8,
        q_lora: 4,
        kv_lora: 4,
        head_mla: 4,
        kda_head: 4,
        kda_lower: -5.0,
        idx_heads: 2,
        idx_dim: 4,
        top_k: 4,
        kpool: 2,
        hc: 4,
        hc_iters: 3,
        hc_eps: 1e-6,
        weights_norm: true,
        weights_scale: 1.0,
        clamp_exp: vec![0.0; 4],
        clamp_shexp: vec![0.0; 4],
        norm_eps: 1e-6,
        nextn: 1,
    };
    assert!(p.mtp() && p.mtp_sparse());
    assert_eq!(p.mtp_streams(), 1);
    assert_eq!(p.residual_streams(), 4);
    assert_eq!(p.recurrent(), Recurrent::Kda);
    assert!(p.indexed(3) && p.indexed(2) && !p.indexed(0));
}

#[test]
fn prune_lists_read_layer_expert_pairs() {
    let text = "# REAP, least salient\n3 188\n3 97\n\n44 5 # last layer\n";
    assert_eq!(
        parse_prune_list(text).unwrap(),
        vec![(3, 188), (3, 97), (44, 5)]
    );
    assert!(parse_prune_list("3\n").unwrap_err().contains("line 1"));
    assert!(parse_prune_list("3 4 5\n").is_err());
    assert!(parse_prune_list("3 x\n").is_err());
}

#[test]
fn route_bias_is_off_by_default() {
    assert_eq!(route_bias(), 0.0);
}

/// #569: a verify step past `--expert-mv-max-tokens` (4 sequences at a fixed MTP k = 2: 12 rows)
/// keeps the tiered expert matvec, up to its exact range, instead of the fused op reading the
/// uncached experts in place; prompt steps and plain decode steps keep the matvec's own limit.
#[test]
fn verify_steps_raise_the_expert_matvec_limit() {
    use crate::model::core::expert_mv::{EXACT_MAX_TOKENS, MAX_TOKENS};
    let c4_k2 = [(true, 3); 4];
    assert!(mv_pass(c4_k2, false));
    assert_eq!(mv_tokens(true, MAX_TOKENS, 0), EXACT_MAX_TOKENS);
    assert!(mv_tokens(mv_pass(c4_k2, false), MAX_TOKENS, 0) >= 12);
    // 16 sequences at k = 3: the most a step takes by default
    assert!(mv_tokens(mv_pass([(true, 4); 16], false), MAX_TOKENS, 0) >= 64);
    // Verify groups beside one-token decode groups (sequences without drafts): a verify step.
    assert!(mv_pass([(true, 3), (false, 1), (true, 2)], false));
    // A prompt chunk, alone or beside verify groups, plain decoding, nothing: not one.
    assert!(!mv_pass([(false, 12)], false));
    assert!(!mv_pass([(true, 3), (false, 12)], false));
    assert!(!mv_pass([(false, 1); 16], false));
    assert!(!mv_pass([], false));
    // Other steps keep --expert-mv-max-tokens.
    assert_eq!(mv_tokens(false, MAX_TOKENS, 0), MAX_TOKENS);
    assert_eq!(mv_tokens(false, 16, 64), 16);
    // --glm-verify-mv-max-tokens 8: the limit before #569; never below --expert-mv-max-tokens.
    assert_eq!(mv_tokens(true, MAX_TOKENS, 8), 8);
    assert_eq!(mv_tokens(true, 16, 8), 16);
    assert_eq!(mv_tokens(true, 8, 32), 32);
}

/// #659: with --glm-decode-mv, plain decode steps of one-token groups take the verify limit too,
/// so 9-64 decoding sequences keep the tiered expert matvec; prompt chunks never do, and without
/// the switch nothing changes.
#[test]
fn decode_mv_raises_the_limit_for_plain_decode_only() {
    use crate::model::core::expert_mv::{EXACT_MAX_TOKENS, MAX_TOKENS};
    let c16 = [(false, 1); 16];
    assert!(!mv_pass(c16, false));
    assert_eq!(mv_tokens(mv_pass(c16, false), MAX_TOKENS, 0), MAX_TOKENS);
    assert!(mv_pass(c16, true));
    assert_eq!(
        mv_tokens(mv_pass(c16, true), MAX_TOKENS, 0),
        EXACT_MAX_TOKENS
    );
    // --glm-verify-mv-max-tokens bounds plain decode as it bounds verify steps.
    assert_eq!(mv_tokens(mv_pass(c16, true), MAX_TOKENS, 32), 32);
    // A prompt chunk, alone or beside decode groups, nothing: not one.
    assert!(!mv_pass([(false, 12)], true));
    assert!(!mv_pass([(false, 1), (false, 12)], true));
    assert!(!mv_pass([], true));
    // Verify steps stay verify steps either way.
    assert!(mv_pass([(true, 3), (false, 1)], true));
}
