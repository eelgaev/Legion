//! DFlash 2 against a plain-Rust implementation (convolutions in [`super::tests::Reference`],
//! the selector here), over the tiny checkpoint with [`tiny::TINY_V2`]; loaders and refusals.

use super::tests::{Fixture, Reference, ctx_rows, matvec, max_diff, sampling, step, tokens_to};
use super::tiny::{self, HfDtype, TINY_V2, TinyDraft};
use super::v2::Selector;
use super::*;
use crate::ggml::types;
use crate::spec::DraftWeights;

pub(super) fn v2_draft() -> TinyDraft {
    TinyDraft {
        v2: Some(TINY_V2),
        sliding_window: Some(12),
        ..TinyDraft::default()
    }
}

/// The selector's inputs of every draft position from the reference network: candidates
/// (best first), their logits, the hidden projection.
pub(super) fn selector_inputs(
    r: &Reference,
    hidden: &[Vec<f32>],
) -> Vec<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    let e = hidden[0].len();
    hidden
        .iter()
        .map(|h| {
            let logits = matvec(&r.head, e, h);
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            order.truncate(TINY_V2.selector_top_k);
            let proj = matvec(&r.w["selector_hidden.weight"], e, h);
            (
                order.iter().map(|&i| i as u32).collect(),
                order.iter().map(|&i| logits[i]).collect(),
                proj,
            )
        })
        .collect()
}

/// The selector's scores of `candidates` after `prev`.
pub(super) fn scores(
    r: &Reference,
    prev: u32,
    cand: &[u32],
    unary: &[f32],
    proj: &[f32],
) -> Vec<f32> {
    let rank = TINY_V2.selector_rank;
    let row = |book: &str, t: u32| r.w[book][t as usize * rank..(t as usize + 1) * rank].to_vec();
    let pred = row("selector_predecessor.weight", prev);
    cand.iter()
        .zip(unary)
        .map(|(&c, &u)| {
            let succ = row("selector_successor.weight", c);
            u + (0..rank).map(|i| pred[i] * proj[i] * succ[i]).sum::<f32>()
        })
        .collect()
}

/// The block's hidden rows from the reference, for a context `0..n_ctx` and the anchor
/// `tokens[n_ctx]`.
pub(super) fn reference_hidden(
    fx: &Fixture,
    c: &TinyDraft,
    tokens: &[u32],
    n_ctx: usize,
) -> (Reference, Vec<Vec<f32>>) {
    let r = Reference::new(fx, c, |x| x);
    let mut block = vec![c.mask_token; c.block_size];
    block[0] = tokens[n_ctx];
    // The reference applies the sliding window itself.
    let hidden = r.hidden_rows(&ctx_rows(0..n_ctx, 0), &block, n_ctx, 1);
    (r, hidden)
}

#[test]
fn greedy_drafts_are_the_reference_selector_walk() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(21);
    let mut d = fx.load(&path, DflashOptions::default());
    assert_eq!(d.max_drafts(), 7);
    step(&mut d, 1, &tokens[..8], 0, 7, 0, 0);
    let out = step(&mut d, 1, &tokens, 7, 13, 7, 0);
    assert_eq!(out.tokens.len(), 7);
    assert!(out.probs.is_empty());
    let (r, hidden) = reference_hidden(&fx, &c, &tokens, 20);
    let inputs = selector_inputs(&r, &hidden);
    // Every pick is the best candidate after the previous pick (to rounding).
    let mut prev = tokens[20];
    for (i, ((cand, unary, proj), &got)) in inputs.iter().zip(&out.tokens).enumerate() {
        let s = scores(&r, prev, cand, unary, proj);
        let best = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let at = cand.iter().position(|&t| t == got);
        let at = at.unwrap_or_else(|| panic!("position {i}: {got} is not a candidate {cand:?}"));
        assert!(
            best - s[at] < 1e-3,
            "position {i}: {} vs best {best}",
            s[at]
        );
        prev = got;
    }
    // The selector matters: the walk differs from the head's argmax somewhere.
    let argmax: Vec<u32> = inputs.iter().map(|(c, _, _)| c[0]).collect();
    assert_ne!(argmax, out.tokens);
}

#[test]
fn sampled_drafts_carry_the_candidate_distribution() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(21);
    let mut d = fx.load(&path, sampling(64));
    let out = step(&mut d, 1, &tokens, 0, 20, 7, 0);
    assert_eq!(out.tokens.len(), 7);
    assert_eq!(out.probs.len(), 7);
    let (r, hidden) = reference_hidden(&fx, &c, &tokens, 20);
    let inputs = selector_inputs(&r, &hidden);
    let mut prev = tokens[20];
    for (i, (cand, unary, proj)) in inputs.iter().enumerate() {
        let s = scores(&r, prev, cand, unary, proj);
        let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = s.iter().map(|x| (x - m).exp()).sum();
        let mut want = vec![0f32; tests::N_VOCAB];
        for (&t, x) in cand.iter().zip(&s) {
            want[t as usize] = (x - m).exp() / z;
        }
        let q = &out.probs[i];
        assert!(max_diff(&want, q) < 2e-3, "position {i}");
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(q[out.tokens[i] as usize] > 0.0);
        assert_eq!(
            q.iter().filter(|&&p| p > 0.0).count(),
            TINY_V2.selector_top_k
        );
        prev = out.tokens[i];
    }
}

#[test]
fn the_whole_block_runs_and_max_keeps_a_prefix() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(17);
    let mut d = fx.load(&path, DflashOptions::default());
    step(&mut d, 1, &tokens, 0, 16, 0, 0);
    let full = step(&mut d, 1, &tokens, 16, 0, 7, 0);
    for max in [1, 3, 6, 20] {
        let out = step(&mut d, 1, &tokens, 16, 0, max, 0);
        assert_eq!(out.tokens[..], full.tokens[..max.min(7)], "max {max}");
    }
    // So the engine prices its depth as one pass; a DFlash 1 drafter runs what it is asked for.
    assert_eq!(d.block_drafts(), Some(7));
    let v1 = fx.load(
        &fx.gguf("v1.gguf", &TinyDraft::default()),
        DflashOptions::default(),
    );
    assert_eq!(v1.block_drafts(), None);
}

/// Ranks above 0 of tensor parallelism over machines size their verify rows by the config
/// alone: the same block as the loaded drafter's, from a GGUF and a Hugging Face directory.
#[test]
fn block_drafts_from_the_config_alone() {
    let fx = Fixture::new();
    let c = v2_draft();
    let gguf = fx.gguf("d.gguf", &c);
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::F32).unwrap();
    for p in [&gguf, &hf] {
        let loaded = fx.load(p, DflashOptions::default()).block_drafts();
        assert_eq!(loaded, Some(7));
        assert_eq!(block_drafts_of(p).unwrap(), loaded, "{}", p.display());
    }
    let v1 = fx.gguf("v1.gguf", &TinyDraft::default());
    assert_eq!(block_drafts_of(&v1).unwrap(), None);
    assert!(block_drafts_of(&fx.root.join("missing.gguf")).is_err());
}

#[test]
fn hugging_face_dflash2_loads_like_gguf() {
    let fx = Fixture::new();
    let c = v2_draft();
    let gguf = fx.gguf("d.gguf", &c);
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::F32).unwrap();
    let tokens = tokens_to(13);
    let mut a = fx.load(&gguf, sampling(64));
    let mut b = fx.load(&hf, sampling(64));
    assert_eq!(a.config().v2, Some(TINY_V2));
    assert_eq!(b.config().v2, Some(TINY_V2));
    assert_eq!(
        step(&mut a, 1, &tokens, 0, 12, 7, 0),
        step(&mut b, 1, &tokens, 0, 12, 7, 0)
    );
    // BF16, as z-lab ships it: the base kernels are computed in F32, the rest runs.
    let bf = fx.root.join("bf16");
    tiny::write_hf(&bf, &c, HfDtype::Bf16).unwrap();
    let mut d = fx.load(&bf, DflashOptions::default());
    assert_eq!(step(&mut d, 1, &tokens, 0, 12, 7, 0).tokens.len(), 7);
}

#[test]
fn rollback_equals_never_having_seen_the_rejected_rows() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(15);
    // Greedy: the walk is a function of the block pass alone. (Sampled walks also depend on
    // the draws before, so only the first position, after the anchor, compares.)
    for opts in [DflashOptions::default(), sampling(64)] {
        let mut a = fx.load(&path, opts.clone());
        step(&mut a, 1, &tokens[..13], 0, 12, 0, 0);
        step(&mut a, 1, &tokens[..15], 12, 3, 4, 0);
        let after = step(&mut a, 1, &tokens[..14], 12, 2, 5, 1);
        let mut b = fx.load(&path, opts);
        step(&mut b, 1, &tokens[..13], 0, 12, 0, 0);
        let fresh = step(&mut b, 1, &tokens[..14], 12, 2, 5, 1);
        match after.probs.first() {
            None => assert_eq!(after.tokens, fresh.tokens),
            Some(q) => assert_eq!(q, &fresh.probs[0]),
        }
    }
}

#[test]
fn the_cache_of_an_all_sliding_drafter_is_the_window_plus_blocks() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let d = fx.load(&path, DflashOptions::default());
    assert_eq!(d.model.window, 12 + 2 * 8);
    // Not sliding: the configured window.
    let open = TinyDraft {
        sliding_window: None,
        ..c
    };
    let d = fx.load(&fx.gguf("o.gguf", &open), DflashOptions::default());
    assert_eq!(d.model.window, 4096);
}

/// A request of [`step_many`]: `(seq, tokens, start, ran, max, salt)`, as [`step`] takes them.
type Req<'a> = (u64, &'a [u32], usize, usize, usize, u64);

/// One `step` over several sequences at once.
fn step_many(d: &mut DflashDrafter, reqs: &[Req<'_>]) -> Vec<Draft> {
    let layers = d.hidden_layers().len();
    let rows: Vec<Vec<Vec<f32>>> = reqs
        .iter()
        .map(|&(_, _, start, ran, _, salt)| {
            (0..layers)
                .map(|li| {
                    (start..start + ran)
                        .flat_map(|p| tests::hidden(p, li, salt))
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut outs = vec![Draft::default(); reqs.len()];
    let mut batch: Vec<DraftRequest<'_>> = reqs
        .iter()
        .zip(&rows)
        .zip(outs.iter_mut())
        .map(
            |((&(seq, tokens, start, ran, max, _), rows), out)| DraftRequest {
                seq,
                tokens,
                start,
                ran,
                hidden: rows.iter().map(|v| v.as_slice()).collect(),
                max,
                out,
            },
        )
        .collect();
    d.step(&mut batch).unwrap();
    drop(batch);
    outs
}

#[test]
fn sequences_drafted_together_draft_what_they_draft_alone() {
    // Different context lengths (so different masks and cache reads), a sliding window, the
    // take-in batched too (prompts, then a decode step's few rows), and one sequence that only
    // takes in; greedy and sampled (the first position's distribution compares).
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let (a, b, e) = (tokens_to(16), tokens_to(12), tokens_to(9));
    for opts in [DflashOptions::default(), sampling(64)] {
        let mut together = fx.load(&path, opts.clone());
        let first = step_many(
            &mut together,
            &[
                (1, &a[..13], 0, 13, 7, 0),
                (2, &b[..10], 0, 10, 7, 1),
                (3, &e, 0, 9, 0, 2),
            ],
        );
        let second = step_many(&mut together, &[(1, &a, 13, 3, 7, 0), (2, &b, 10, 2, 5, 1)]);
        let mut alone = fx.load(&path, opts);
        let want = [
            step(&mut alone, 1, &a[..13], 0, 13, 7, 0),
            step(&mut alone, 2, &b[..10], 0, 10, 7, 1),
            step(&mut alone, 3, &e, 0, 9, 0, 2),
            step(&mut alone, 1, &a, 13, 3, 7, 0),
            step(&mut alone, 2, &b, 10, 2, 5, 1),
        ];
        let got = first.iter().chain(&second);
        for (i, (got, want)) in got.zip(&want).enumerate() {
            assert_eq!(got.tokens.len(), want.tokens.len(), "request {i}");
            match want.probs.first() {
                None => assert_eq!(got.tokens, want.tokens, "request {i}"),
                Some(q) => assert!(max_diff(&got.probs[0], q) < 1e-5, "request {i}"),
            }
        }
        assert_eq!(second[1].tokens.len(), 5);
    }
}

#[test]
fn ties_resolve_by_logit_then_token() {
    // Rank 1, codebooks all ones, a zero hidden projection: the scores are the logits.
    let ones: Vec<u8> = (0..4).flat_map(|_| 1f32.to_ne_bytes()).collect();
    let s = Selector::new(1, 3, types::F32, ones.clone(), ones);
    let mut rng = crate::engine::sampling::Rng::new(1);
    // Device order is arbitrary: 3 and 1 tie on top.
    let w = s.walk(0, &[2, 3, 1], &[0.5, 2.0, 2.0], &[0.0], 0.0, &mut rng);
    assert_eq!(w.tokens, vec![1]);
    assert!(w.probs.is_none());
    // The pairwise term decides: predecessor and successor rows times the projection.
    let pred: Vec<u8> = [1f32, 1.0, 1.0, 1.0]
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    let succ: Vec<u8> = [0f32, 0.0, 0.0, 5.0]
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    let s = Selector::new(1, 2, types::F32, pred, succ);
    let w = s.walk(
        0,
        &[1, 3, 3, 1],
        &[1.0, 0.0, 0.0, 1.0],
        &[1.0, 1.0],
        0.0,
        &mut rng,
    );
    assert_eq!(w.tokens, vec![3, 3]);
    let w = s.walk(0, &[1, 3], &[1.0, 0.0], &[1.0], 2.0, &mut rng);
    let p = &w.probs.unwrap()[0];
    let total: f32 = p.iter().map(|x| x.1).sum();
    assert!((total - 1.0).abs() < 1e-6);
    // softmax([1, 5] / 2) on (1, 3).
    let want = 1.0 / (1.0 + 2f32.exp());
    assert!((p.iter().find(|x| x.0 == 1).unwrap().1 - want).abs() < 1e-6);
}

#[test]
fn incomplete_or_misplaced_dflash2_settings_are_refused() {
    let fx = Fixture::new();
    let c = v2_draft();
    let err = |p: &std::path::Path, flavor| {
        DflashDrafter::load_as(&fx.info(p), DflashOptions::default(), flavor)
            .err()
            .unwrap()
    };
    let edit = |name: &str, f: &dyn Fn(&mut serde_json::Value)| {
        let dir = fx.root.join(name);
        tiny::write_hf(&dir, &c, HfDtype::F32).unwrap();
        let p = dir.join("config.json");
        let mut j: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        f(&mut j);
        std::fs::write(&p, serde_json::to_vec(&j).unwrap()).unwrap();
        dir
    };
    // Only some of the settings.
    let dir = edit("partial", &|j| {
        j["dflash_config"]
            .as_object_mut()
            .unwrap()
            .remove("selector_top_k");
    });
    let e = err(&dir, Flavor::Dflash);
    assert!(e.contains("selector_top_k"), "{e}");
    // Convolution tensors without the settings: not a network this computes.
    let dir = edit("plain", &|j| {
        let d = j["dflash_config"].as_object_mut().unwrap();
        for k in [
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ] {
            d.remove(k);
        }
        j["architectures"] = serde_json::json!(["DFlashDraftModel"]);
    });
    let e = err(&dir, Flavor::Dflash);
    assert!(e.contains("_conv_") || e.contains("selector_"), "{e}");
    assert!(e.contains("not implemented"), "{e}");
    // The settings on a DSpark.
    let dspark = TinyDraft {
        markov_rank: Some(4),
        ..c.clone()
    };
    let path = fx.gguf("s.gguf", &dspark);
    let e = err(&path, Flavor::Dspark);
    assert!(e.contains("DSpark"), "{e}");
    // A selector wider than the vocabulary.
    let wide = TinyDraft {
        v2: Some(Dflash2 {
            selector_top_k: 300,
            ..TINY_V2
        }),
        ..c.clone()
    };
    let e = err(&fx.gguf("w.gguf", &wide), Flavor::Dflash);
    assert!(e.contains("top-k"), "{e}");
}

#[test]
fn q8_0_drafter_weights_are_q8_copies_of_the_matrices() {
    let fx = Fixture::new();
    let c = v2_draft();
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::Bf16).unwrap();
    let source = weights::Source::open(&hf).unwrap();
    let cpu = Backend::cpu().unwrap();
    let plain = source.load(cpu.buffer_type(), DraftWeights::Model).unwrap();
    let q8 = source.load(cpu.buffer_type(), DraftWeights::Q8_0).unwrap();
    let mut copied = 0;
    for name in source.tensor_names() {
        let (Some(a), Some(b)) = (plain.get(&name), q8.get(&name)) else {
            continue;
        };
        if weights::q8_matrix(&name, &a.shape()) {
            copied += 1;
            assert_eq!(a.ty(), types::BF16, "{name}");
            assert_eq!(b.ty(), types::Q8_0, "{name}");
            let n = a.elements();
            let want = crate::ggml::dequantize(types::BF16, &a.bytes(), n);
            let got = crate::ggml::dequantize(types::Q8_0, &b.bytes(), n);
            let amax = want.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(max_diff(&want, &got) <= amax / 127.0, "{name}");
        } else {
            assert_eq!(b.ty(), a.ty(), "{name}");
            assert_eq!(b.bytes(), a.bytes(), "{name}");
        }
    }
    // Every layer's seven projections and two kernel projections, `fc`, the selector's.
    assert_eq!(copied, c.n_layer * 9 + 2);
    for (name, shape, q) in [
        ("blk.0.ffn_down.weight", &[96, 64][..], true),
        ("blk.1.attn_conv_proj.weight", &[64, 16], true),
        ("fc.weight", &[128, 64], true),
        ("selector_hidden.weight", &[64, 8], true),
        ("blk.0.ffn_down.weight", &[48, 64], false),
        ("blk.0.attn_norm.weight", &[64], false),
        ("blk.0.attn_conv_base", &[64, 2, 2], false),
        ("token_embd.weight", &[64, 259], false),
        ("output.weight", &[64, 259], false),
        ("markov_w1.weight", &[64, 8], false),
    ] {
        assert_eq!(weights::q8_matrix(name, shape), q, "{name} {shape:?}");
    }
    // The drafter drafts with them.
    let opts = DflashOptions {
        weights: crate::spec::DraftWeights::Q8_0,
        ..DflashOptions::default()
    };
    let mut d = fx.load(&hf, opts);
    assert_eq!(step(&mut d, 1, &tokens_to(13), 0, 12, 7, 0).tokens.len(), 7);
}

#[test]
fn q4_drafter_weights_are_4_bit_copies_of_the_q8_0_matrices() {
    let fx = Fixture::new();
    let c = v2_draft();
    let hf = fx.root.join("hf");
    tiny::write_hf(&hf, &c, HfDtype::Bf16).unwrap();
    let source = weights::Source::open(&hf).unwrap();
    let cpu = Backend::cpu().unwrap();
    let plain = source.load(cpu.buffer_type(), DraftWeights::Model).unwrap();
    let q4 = source.load(cpu.buffer_type(), DraftWeights::Q4).unwrap();
    let (mut k_quants, mut copied) = (0, 0);
    for name in source.tensor_names() {
        let (Some(a), Some(b)) = (plain.get(&name), q4.get(&name)) else {
            continue;
        };
        if weights::q8_matrix(&name, &a.shape()) {
            copied += 1;
            let k = a.shape()[0] as usize;
            let want_ty = if k.is_multiple_of(256) {
                types::Q4_K
            } else {
                types::Q4_0
            };
            k_quants += usize::from(want_ty == types::Q4_K);
            assert_eq!(b.ty(), want_ty, "{name}");
            let n = a.elements();
            let want = crate::ggml::dequantize(types::BF16, &a.bytes(), n);
            assert_eq!(
                b.bytes(),
                crate::ggml::quantize(want_ty, &want, k),
                "{name}"
            );
        } else {
            assert_eq!(b.ty(), a.ty(), "{name}");
            assert_eq!(b.bytes(), a.bytes(), "{name}");
        }
    }
    // The same matrices as `q8_0`; the tiny drafter's are all narrower than 256: Q4_0.
    assert_eq!(copied, c.n_layer * 9 + 2);
    assert_eq!(k_quants, 0);
    // The drafter drafts with them.
    let opts = DflashOptions {
        weights: DraftWeights::Q4,
        ..DflashOptions::default()
    };
    let mut d = fx.load(&hf, opts);
    assert_eq!(step(&mut d, 1, &tokens_to(13), 0, 12, 7, 0).tokens.len(), 7);
}

#[test]
fn fp8_block_scaled_matrices_are_read_as_q8_0() {
    let fx = Fixture::new();
    let c = v2_draft();
    // Blocks of unequal sides, so that a transposed scale index reads the wrong factors.
    let block = [16, 32];
    let (bf, fp8) = (fx.root.join("bf16"), fx.root.join("fp8"));
    tiny::write_hf(&bf, &c, HfDtype::Bf16).unwrap();
    tiny::write_hf_fp8(&fp8, &c, block).unwrap();
    let cpu = Backend::cpu().unwrap();
    let plain = weights::Source::open(&bf)
        .unwrap()
        .load(cpu.buffer_type(), DraftWeights::Model)
        .unwrap();
    let source = weights::Source::open(&fp8).unwrap();
    let got = source.load(cpu.buffer_type(), DraftWeights::Model).unwrap();
    let mut read = 0;
    for (name, shape, v) in tiny::tensors(&c) {
        let Some(b) = got.get(&name) else {
            assert!(plain.get(&name).is_none(), "{name}");
            continue;
        };
        let a = plain.get(&name).unwrap();
        if !tiny::fp8_stored(&name) {
            assert_eq!(b.ty(), a.ty(), "{name}");
            assert_eq!(b.bytes(), a.bytes(), "{name}");
            continue;
        }
        read += 1;
        assert_eq!(b.ty(), types::Q8_0, "{name}");
        // The checkpoint's values, element by element, then Q8_0.
        let (rows, cols) = (shape[1] as usize, shape[0] as usize);
        let (bytes, scales) = tiny::fp8_blocks(&v, rows, cols, block);
        let sc = cols.div_ceil(block[1]);
        let want: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                crate::model::fp8::e4m3_to_f32(bytes[i]) * scales[r / block[0] * sc + c / block[1]]
            })
            .collect();
        assert_eq!(
            b.bytes(),
            crate::ggml::quantize(types::Q8_0, &want, cols),
            "{name}"
        );
    }
    assert_eq!(read, c.n_layer * 7 + 1);
    // `--spec-draft-weights q8_0` and `q4` change nothing for them.
    for w in [DraftWeights::Q8_0, DraftWeights::Q4] {
        let other = source.load(cpu.buffer_type(), w).unwrap();
        assert_eq!(
            other.get("blk.0.ffn_down.weight").unwrap().bytes(),
            got.get("blk.0.ffn_down.weight").unwrap().bytes(),
            "{w:?}"
        );
    }
    // The drafter drafts with them.
    let mut d = fx.load(&fp8, DflashOptions::default());
    assert_eq!(step(&mut d, 1, &tokens_to(13), 0, 12, 7, 0).tokens.len(), 7);

    // Scales that do not fit the declared blocks, and blocks that are not two sizes.
    let config = fp8.join("config.json");
    let text = std::fs::read_to_string(&config).unwrap();
    for (bad, why) in [
        ("[32, 32]", "expected a float"),
        ("[0, 32]", "two positive integers"),
    ] {
        let mut j: serde_json::Value = serde_json::from_str(&text).unwrap();
        j["quantization_config"]["weight_block_size"] = serde_json::from_str(bad).unwrap();
        std::fs::write(&config, serde_json::to_vec(&j).unwrap()).unwrap();
        let e = weights::Source::open(&fp8)
            .unwrap()
            .load(cpu.buffer_type(), DraftWeights::Model)
            .err()
            .unwrap();
        assert!(e.contains(why), "{bad}: {e}");
    }
}
