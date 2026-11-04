//! The DSpark additions (Markov head, confidence head, attention sinks, anchor handling,
//! NVFP4 matrices) against the plain-Rust reference network.

use std::path::Path;

use super::tests::*;
use super::tiny::{self, HfDtype, Sinks, TinyDraft};
use super::*;
use crate::ggml::ffi;
use crate::spec::dspark::{DsparkDrafter, DsparkOptions};

const NVFP4: ffi::ggml_type = ffi::ggml_type_GGML_TYPE_NVFP4;

fn dspark(fx: &Fixture, path: &Path, opts: DsparkOptions) -> DsparkDrafter {
    DsparkDrafter::load(&fx.info(path), opts).unwrap()
}

fn spark() -> TinyDraft {
    TinyDraft {
        markov_rank: Some(16),
        ..TinyDraft::default()
    }
}

/// `markov_w2 · markov_w1[prev]` from the checkpoint's own tensors.
struct MarkovRef {
    rank: usize,
    w1: Vec<f32>,
    w2: Vec<f32>,
    conf: Option<(Vec<f32>, f32)>,
}

impl MarkovRef {
    fn new(c: &TinyDraft) -> Self {
        let t = tiny::effective_tensors(c);
        let get = |n: &str| t.iter().find(|t| t.0 == n).map(|t| t.2.clone());
        Self {
            rank: c.markov_rank.unwrap(),
            w1: get("markov_w1.weight").unwrap(),
            w2: get("markov_w2.weight").unwrap(),
            conf: get("conf_proj.weight").map(|w| (w, get("conf_proj.bias").unwrap()[0])),
        }
    }

    fn w1_row(&self, prev: u32) -> &[f32] {
        &self.w1[prev as usize * self.rank..(prev as usize + 1) * self.rank]
    }

    fn bias(&self, prev: u32) -> Vec<f32> {
        matvec(&self.w2, self.rank, self.w1_row(prev))
    }

    fn confidence(&self, hidden: &[f32], prev: u32) -> f32 {
        let (w, b) = self.conf.as_ref().unwrap();
        let x: Vec<f32> = hidden.iter().chain(self.w1_row(prev)).copied().collect();
        let z: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() + b;
        1.0 / (1.0 + (-z).exp())
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

/// Feeds a 16-row context and drafts; returns the draft and the reference network's block
/// (`last` token, then masks) and anchor.
fn run(
    fx: &Fixture,
    c: &TinyDraft,
    path: &Path,
    opts: DsparkOptions,
) -> (Draft, Reference, Vec<u32>, usize, Vec<u32>) {
    let mut d = dspark(fx, path, opts);
    let tokens = tokens_to(17);
    let max = d.max_drafts();
    let out = step(&mut d, 1, &tokens, 0, 16, max, 0);
    let first = usize::from(!c.sample_from_anchor);
    let mut block = vec![c.mask_token; max + first];
    block[0] = tokens[16];
    let r = Reference::new(fx, c, |x| x);
    (out, r, block, 16, tokens)
}

fn check_chain(c: &TinyDraft, path_name: &str, tol: f32) {
    let fx = Fixture::new();
    let path = fx.gguf(path_name, c);
    let (out, r, block, anchor, tokens) = run(&fx, c, &path, sampling(64));
    let first = usize::from(!c.sample_from_anchor);
    let most = if c.sample_from_anchor { 8 } else { 7 };
    assert_eq!(out.tokens.len(), most);
    assert_eq!(out.probs.len(), most);
    let base = r.logits(&ctx_rows(0..16, 0), &block, anchor, first);
    let m = MarkovRef::new(c);
    let mut prev = tokens[16];
    for (i, row) in base.iter().enumerate() {
        let l: Vec<f32> = row.iter().zip(m.bias(prev)).map(|(a, b)| a + b).collect();
        let diff = max_diff(&r.probs(&l), &out.probs[i]);
        assert!(diff < tol, "position {i}: probabilities differ by {diff}");
        // The next position's bias follows the token drawn here.
        prev = out.tokens[i];
    }
}

#[test]
fn the_markov_chain_follows_the_drafted_tokens() {
    for sample_from_anchor in [true, false] {
        check_chain(
            &TinyDraft {
                sample_from_anchor,
                ..spark()
            },
            "d.gguf",
            2e-3,
        );
    }
}

#[test]
fn sinks_join_the_softmax_under_both_names() {
    let fx = Fixture::new();
    let weight = TinyDraft {
        sinks: Sinks::Weight,
        ..spark()
    };
    let bare = TinyDraft {
        sinks: Sinks::Bare,
        ..spark()
    };
    check_chain(&weight, "w.gguf", 2e-3);
    check_chain(&bare, "b.gguf", 2e-3);
    let (a, ..) = run(&fx, &weight, &fx.gguf("a.gguf", &weight), sampling(64));
    let (b, ..) = run(&fx, &bare, &fx.gguf("b.gguf", &bare), sampling(64));
    assert_eq!(a, b);
    // And they change the result: the same weights without the sink logit.
    let none = TinyDraft {
        sinks: Sinks::None,
        ..spark()
    };
    let plain = TinyDraft { ..none };
    let (p, ..) = run(&fx, &plain, &fx.gguf("p.gguf", &plain), sampling(64));
    assert!(max_diff(&a.probs[0], &p.probs[0]) > 1e-3);
}

#[test]
fn greedy_drafts_walk_the_chain_with_their_own_tokens() {
    let fx = Fixture::new();
    let c = spark();
    let path = fx.gguf("d.gguf", &c);
    let (out, r, block, anchor, tokens) = run(&fx, &c, &path, DsparkOptions::default());
    assert_eq!(out.tokens.len(), 8);
    assert!(out.probs.is_empty());
    let base = r.logits(&ctx_rows(0..16, 0), &block, anchor, 0);
    let m = MarkovRef::new(&c);
    let mut prev = tokens[16];
    for (i, row) in base.iter().enumerate() {
        let l: Vec<f32> = row.iter().zip(m.bias(prev)).map(|(a, b)| a + b).collect();
        prev = argmax(&l);
        assert_eq!(out.tokens[i], prev, "position {i}");
    }
}

#[test]
fn the_confidence_head_cuts_the_block() {
    let fx = Fixture::new();
    let c = TinyDraft {
        confidence: true,
        ..spark()
    };
    let path = fx.gguf("d.gguf", &c);
    let (full, r, block, anchor, tokens) = run(&fx, &c, &path, DsparkOptions::default());
    assert_eq!(full.tokens.len(), 8);
    let hidden = r.hidden_rows(&ctx_rows(0..16, 0), &block, anchor, 0);
    let m = MarkovRef::new(&c);
    let confs: Vec<f32> = (0..8)
        .map(|i| {
            let prev = if i == 0 {
                tokens[16]
            } else {
                full.tokens[i - 1]
            };
            m.confidence(&hidden[i], prev)
        })
        .collect();
    let (lo, hi) = confs
        .iter()
        .fold((1f32, 0f32), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        hi - lo > 0.01,
        "the test needs confidences that differ: {confs:?}"
    );
    let mut d = dspark(&fx, &path, DsparkOptions::default());
    for p in [0.0, lo - 0.01, (lo + hi) / 2.0, confs[3] + 1e-3, hi + 0.01] {
        let opts = DsparkOptions {
            min_confidence: p,
            ..DsparkOptions::default()
        };
        let mut d2 = dspark(&fx, &path, opts);
        let out = step(&mut d2, 1, &tokens, 0, 16, 8, 0);
        let want = confs.iter().take_while(|&&x| x >= p).count();
        assert_eq!(out.tokens.len(), want, "threshold {p}: {confs:?}");
        assert_eq!(out.tokens, full.tokens[..want], "threshold {p}");
    }
    let _ = &mut d;
}

#[test]
fn nvfp4_matrices_are_read_natively() {
    let c = TinyDraft {
        n_ff: 128,
        markov_rank: Some(64),
        confidence: true,
        sinks: Sinks::Weight,
        quant: Some(NVFP4),
        ..TinyDraft::default()
    };
    // Matrices whose rows are a multiple of 64 wide (NVFP4's block) are 4-bit in the file,
    // the rest and the norms stay F32. The reference computes with the dequantized weights;
    // the CPU also quantizes activations for the 4-bit dot product, which the bound allows.
    let fx = Fixture::new();
    let path = fx.gguf("q.gguf", &c);
    let g = crate::gguf::GgufFile::open(&path).unwrap();
    for n in [
        "fc.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "markov_w1.weight",
    ] {
        assert_eq!(g.tensor(n).unwrap().ggml_type, NVFP4, "{n}");
    }
    assert_eq!(
        g.tensor("blk.0.attn_norm.weight").unwrap().ggml_type,
        tiny_f32()
    );
    let (out, r, block, anchor, tokens) = run(&fx, &c, &path, sampling(64));
    let base = r.logits(&ctx_rows(0..16, 0), &block, anchor, 0);
    let m = MarkovRef::new(&c);
    let mut prev = tokens[16];
    for (i, row) in base.iter().enumerate() {
        let l: Vec<f32> = row.iter().zip(m.bias(prev)).map(|(a, b)| a + b).collect();
        let diff = max_diff(&r.probs(&l), &out.probs[i]);
        assert!(diff < 2e-3, "position {i}: probabilities differ by {diff}");
        prev = out.tokens[i];
    }
}

fn tiny_f32() -> ffi::ggml_type {
    crate::ggml::types::F32
}

#[test]
fn hugging_face_dspark_checkpoints_load_like_gguf() {
    let fx = Fixture::new();
    let c = TinyDraft {
        confidence: true,
        sinks: Sinks::Weight,
        ..spark()
    };
    let gguf = fx.gguf("d.gguf", &c);
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::F32).unwrap();
    let (a, ..) = run(&fx, &c, &gguf, sampling(64));
    let (b, ..) = run(&fx, &c, &hf, sampling(64));
    assert_eq!(a, b);
}

#[test]
fn rollback_leaves_no_trace_in_a_dspark_drafter() {
    let fx = Fixture::new();
    let c = TinyDraft {
        confidence: true,
        sinks: Sinks::Weight,
        ..spark()
    };
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(15);
    let mut a = dspark(&fx, &path, sampling(64));
    step(&mut a, 1, &tokens[..13], 0, 12, 0, 0);
    let first = step(&mut a, 1, &tokens[..15], 12, 3, 4, 0);
    assert_eq!(first.tokens.len(), 4);
    let after = step(&mut a, 1, &tokens[..14], 12, 2, 5, 1);
    // The Markov chain follows sampled tokens, so the other drafter makes the same number of
    // draws first (on a sequence of its own) to be in the same random state.
    let mut b = dspark(&fx, &path, sampling(64));
    step(&mut b, 9, &tokens[..13], 0, 12, 0, 0);
    step(&mut b, 9, &tokens[..15], 12, 3, 4, 0);
    step(&mut b, 1, &tokens[..13], 0, 12, 0, 0);
    let fresh = step(&mut b, 1, &tokens[..14], 12, 2, 5, 1);
    assert_eq!(after.probs, fresh.probs);
}

#[test]
fn a_block_has_the_trained_size_and_a_bonus_anchor_one_less() {
    let fx = Fixture::new();
    for (anchor, most) in [(true, 8), (false, 7)] {
        let c = TinyDraft {
            sample_from_anchor: anchor,
            ..spark()
        };
        let path = fx.gguf("d.gguf", &c);
        let mut d = dspark(&fx, &path, DsparkOptions::default());
        assert_eq!(d.max_drafts(), most);
        assert!(!d.has_confidence());
        let tokens = tokens_to(9);
        step(&mut d, 1, &tokens, 0, 8, 0, 0);
        for (max, want) in [(1, 1), (4, 4), (most, most), (20, most)] {
            let out = step(&mut d, 1, &tokens, 8, 0, max, 0);
            assert_eq!(out.tokens.len(), want, "anchor {anchor}, max {max}");
        }
    }
}

#[test]
fn dspark_and_dflash_checkpoints_do_not_mix() {
    let fx = Fixture::new();
    let flash = fx.gguf("flash.gguf", &TinyDraft::default());
    let e = DsparkDrafter::load(&fx.info(&flash), DsparkOptions::default()).err();
    assert!(e.unwrap().contains("DFlash"));
    let sp = fx.gguf("spark.gguf", &spark());
    let e = DflashDrafter::load(&fx.info(&sp), DflashOptions::default()).err();
    assert!(e.unwrap().contains("--spec dspark"));
    // A confidence threshold needs the head.
    let opts = DsparkOptions {
        min_confidence: 0.5,
        ..DsparkOptions::default()
    };
    let e = DsparkDrafter::load(&fx.info(&sp), opts).err();
    assert!(e.unwrap().contains("confidence head"));
    // Without --spec-draft.
    let mut info = fx.info(&sp);
    info.draft = None;
    let e = DsparkDrafter::load(&info, DsparkOptions::default()).err();
    assert!(e.unwrap().contains("--spec dspark"));
}

#[test]
fn dspark_hugging_face_configs() {
    let cfg = |v: serde_json::Value| Config::from_hf(&v, Flavor::Dspark);
    // DFlash lineage: everything at the top level or in dflash_config, drafts from the anchor.
    let base = serde_json::json!({
        "architectures": ["Qwen3DSparkModel"],
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 96,
        "target_layer_ids": [1, 3], "mask_token_id": 5, "block_size": 7,
        "markov_rank": 256, "markov_head_type": "vanilla",
    });
    let c = cfg(base.clone()).unwrap();
    assert!(c.sample_from_anchor);
    assert_eq!(
        (c.target_layers.clone(), c.mask_token, c.block_size),
        (vec![1, 3], 5, 7)
    );
    // EAGLE-3 lineage: nested layer config, layers counted from 1, a bonus anchor.
    let eagle = serde_json::json!({
        "architectures": ["DSparkSpeculator"],
        "transformer_layer_config": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "intermediate_size": 96,
        },
        "aux_hidden_state_layer_ids": [2, 4], "mask_token_id": 5, "block_size": 7,
    });
    let c = cfg(eagle.clone()).unwrap();
    assert!(!c.sample_from_anchor);
    assert_eq!(c.target_layers, vec![1, 3]);
    let mut v = eagle;
    v["sample_from_anchor"] = true.into();
    assert!(cfg(v).unwrap().sample_from_anchor);
    // Other Markov heads, and DFlash's class, are refused.
    let mut v = base.clone();
    v["markov_head_type"] = "gated".into();
    assert!(matches!(cfg(v), Err(ConfigError::Unsupported(_))));
    let mut v = base;
    v["architectures"] = serde_json::json!(["DFlashDraftModel"]);
    assert!(matches!(cfg(v), Err(ConfigError::Unsupported(_))));
}
