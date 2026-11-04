//! The target's head cut down for drafting ([`head`]: `--spec-draft-vocab`,
//! `--spec-draft-lm-head q4`) and an FP8 target head, against the reference network with the
//! head the drafter should score with.

use std::path::{Path, PathBuf};

use super::tests::{
    Fixture, N_VOCAB, Reference, ctx_rows, max_diff, read_f32, sampling, step, tokens_to,
};
use super::tests_v2::{reference_hidden, scores, selector_inputs, v2_draft};
use super::tiny::TinyDraft;
use super::*;
use crate::engine::sampling::Rng;
use crate::ggml::types;
use crate::gguf::{GgufWriter, Value};
use crate::spec::DraftLmHead;
use crate::spec::q4_head::to_q4;

/// The shortlist the tests draft from: fewer rows than the tiny target's 259.
const ROWS: usize = 100;
const N_EMBD: usize = 64;

fn opts(base: DflashOptions, draft_vocab: usize, lm_head: DraftLmHead) -> DflashOptions {
    DflashOptions {
        draft_vocab,
        lm_head,
        ..base
    }
}

/// The drafter's probabilities over a 13-token context against the reference's with `head`.
fn check_probs(fx: &Fixture, info: TargetInfo, c: &TinyDraft, o: DflashOptions, head: Vec<f32>) {
    let rows = head.len() / N_EMBD;
    let mut r = Reference::new(fx, c, |x| x);
    r.head = head;
    let tokens = tokens_to(14);
    let mut d = DflashDrafter::load(&info, o).unwrap();
    let out = step(&mut d, 1, &tokens, 0, 13, 7, 0);
    assert_eq!(out.probs.len(), 7);
    let mut block = vec![c.mask_token; c.block_size];
    block[0] = tokens[13];
    let want = r.logits(&ctx_rows(0..13, 0), &block, 13, 1);
    for (i, row) in want.iter().enumerate() {
        let diff = max_diff(&r.probs(row), &out.probs[i]);
        assert!(diff < 2e-3, "position {i}: probabilities differ by {diff}");
        assert!((out.tokens[i] as usize) < rows, "position {i}");
        // No mass past the head's rows.
        assert_eq!(out.probs[i][rows..].iter().sum::<f32>(), 0.0);
    }
}

#[test]
fn a_shortlist_drafts_from_the_first_rows_of_the_head() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let head = read_f32(&fx.target, "output.weight");
    let o = opts(sampling(64), ROWS, DraftLmHead::Model);
    check_probs(&fx, fx.info(&path), &c, o, head[..ROWS * N_EMBD].to_vec());
    // Greedy: the most likely of those rows.
    let tokens = tokens_to(14);
    let mut g = fx.load(
        &path,
        opts(DflashOptions::default(), ROWS, DraftLmHead::Model),
    );
    let mut s = fx.load(&path, opts(sampling(64), ROWS, DraftLmHead::Model));
    let greedy = step(&mut g, 1, &tokens, 0, 13, 7, 0);
    let dist = step(&mut s, 1, &tokens, 0, 13, 7, 0);
    for (i, q) in dist.probs.iter().enumerate() {
        let best = q.iter().copied().fold(0.0, f32::max);
        assert!(best - q[greedy.tokens[i] as usize] < 1e-5, "position {i}");
    }
}

#[test]
fn a_q4_head_drafts_with_the_dequantized_rows() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let full = read_f32(&fx.target, "output.weight");
    for rows in [ROWS, 0] {
        let n = if rows == 0 { N_VOCAB } else { rows };
        let bytes: Vec<u8> = full[..n * N_EMBD]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        // 64-wide rows are no multiple of 256: Q4_0.
        let (ty, q) = to_q4(types::F32, &bytes, None, N_EMBD).unwrap();
        assert_eq!(ty, types::Q4_0);
        let want = crate::ggml::dequantize(ty, &q, n * N_EMBD);
        let o = opts(sampling(64), rows, DraftLmHead::Q4);
        check_probs(&fx, fx.info(&path), &c, o, want);
    }
}

/// `n` E4M3 bytes of moderate magnitude (1/8 to 2, both signs; never NaN).
fn e4m3_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    (0..n)
        .map(|_| {
            let v = (r.next_f64() * 256.0) as u8;
            (v & 0x87) | ((4 + (v >> 3) % 4) << 3)
        })
        .collect()
}

#[test]
fn fp8_rows_are_quantized_with_their_scales_folded_in() {
    for (k, ty) in [(256, types::Q4_K), (64, types::Q4_0)] {
        let rows = 3;
        let w = e4m3_bytes(rows * k, 3);
        for scale in [vec![0.5, 2.0, 0.01], vec![0.25]] {
            let (got, q) = to_q4(types::I8, &w, Some(&scale), k).unwrap();
            assert_eq!(got, ty);
            let back = crate::ggml::dequantize(ty, &q, rows * k);
            for r in 0..rows {
                let s = scale[if scale.len() == 1 { 0 } else { r }];
                let want: Vec<f32> = w[r * k..(r + 1) * k]
                    .iter()
                    .map(|&b| crate::model::fp8::e4m3_to_f32(b) * s)
                    .collect();
                let top = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                let diff = max_diff(&want, &back[r * k..(r + 1) * k]);
                assert!(diff <= 0.15 * top, "k {k} row {r}: {diff} of {top}");
            }
        }
    }
    let w = e4m3_bytes(48, 1);
    assert!(to_q4(types::I8, &w, None, 48).is_err());
    assert!(to_q4(types::I8, &e4m3_bytes(128, 1), Some(&[1.0, 2.0, 3.0]), 64).is_err());
}

/// A target GGUF with the tiny target's embeddings and an FP8 head kept raw (`I8` E4M3 bytes,
/// one F32 scale per row), as a synthesized FP8 checkpoint has it; and the head's values.
fn fp8_target(fx: &Fixture) -> (PathBuf, Vec<f32>) {
    let embd = read_f32(&fx.target, "token_embd.weight");
    let w = e4m3_bytes(N_VOCAB * N_EMBD, 9);
    let scale: Vec<f32> = (0..N_VOCAB).map(|r| 0.02 + 0.001 * r as f32).collect();
    let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
    let shape = [N_EMBD as i64, N_VOCAB as i64];
    let mut g = GgufWriter::new();
    g.set("general.architecture", &Value::String("qwen35".into()))
        .unwrap();
    g.add_tensor("token_embd.weight", types::F32, &shape, &f32_bytes(&embd))
        .unwrap();
    g.add_tensor("output.weight", types::I8, &shape, &w)
        .unwrap();
    g.add_tensor(
        "output.scale",
        types::F32,
        &[N_VOCAB as i64],
        &f32_bytes(&scale),
    )
    .unwrap();
    let path = fx.root.join("fp8-target.gguf");
    g.write(&path).unwrap();
    let head = w
        .chunks(N_EMBD)
        .zip(&scale)
        .flat_map(|(row, s)| {
            row.iter()
                .map(move |&b| crate::model::fp8::e4m3_to_f32(b) * s)
        })
        .collect();
    (path, head)
}

fn info_with_target(fx: &Fixture, draft: &Path, target: &Path) -> TargetInfo {
    TargetInfo {
        model: target.to_path_buf(),
        ..fx.info(draft)
    }
}

#[test]
fn an_fp8_target_head_drafts_through_the_fp8_matmul() {
    let fx = Fixture::new();
    let c = TinyDraft::default();
    let path = fx.gguf("d.gguf", &c);
    let (target, head) = fp8_target(&fx);
    let info = || info_with_target(&fx, &path, &target);
    // All of it, then its first rows (the per-row scale sliced to match).
    let o = opts(sampling(64), 0, DraftLmHead::Model);
    check_probs(&fx, info(), &c, o, head.clone());
    let o = opts(sampling(64), ROWS, DraftLmHead::Model);
    check_probs(&fx, info(), &c, o, head[..ROWS * N_EMBD].to_vec());
    // 4 bits, the scales folded in.
    let rows: Vec<u8> = {
        let g = crate::gguf::GgufFile::open(&target).unwrap();
        let t = g.tensor("output.weight").unwrap();
        let all = std::fs::read(&target).unwrap();
        all[t.offset as usize..][..ROWS * N_EMBD].to_vec()
    };
    let scale: Vec<f32> = (0..ROWS).map(|r| 0.02 + 0.001 * r as f32).collect();
    let (ty, q) = to_q4(types::I8, &rows, Some(&scale), N_EMBD).unwrap();
    let want = crate::ggml::dequantize(ty, &q, ROWS * N_EMBD);
    let o = opts(sampling(64), ROWS, DraftLmHead::Q4);
    check_probs(&fx, info(), &c, o, want);
}

#[test]
fn a_dflash2_shortlist_walks_candidates_from_the_first_rows() {
    let fx = Fixture::new();
    let c = v2_draft();
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(21);
    // The anchor is outside the shortlist: its codebook row is still there.
    assert!(tokens[20] as usize >= ROWS);
    let mut d = fx.load(
        &path,
        opts(DflashOptions::default(), ROWS, DraftLmHead::Model),
    );
    step(&mut d, 1, &tokens[..8], 0, 7, 0, 0);
    let out = step(&mut d, 1, &tokens, 7, 13, 7, 0);
    assert_eq!(out.tokens.len(), 7);
    let (mut r, hidden) = reference_hidden(&fx, &c, &tokens, 20);
    r.head.truncate(ROWS * N_EMBD);
    let inputs = selector_inputs(&r, &hidden);
    let mut prev = tokens[20];
    for (i, ((cand, unary, proj), &got)) in inputs.iter().zip(&out.tokens).enumerate() {
        assert!((got as usize) < ROWS, "position {i}: {got}");
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
}

#[test]
fn a_drafter_with_its_own_head_ignores_the_cut() {
    let fx = Fixture::new();
    // A reduced vocabulary of its own (`d2t`): the options do not apply.
    let ids: Vec<u32> = (0..40).map(|i| 3 + 6 * i).collect();
    let c = TinyDraft {
        draft_vocab: Some(ids.clone()),
        ..TinyDraft::default()
    };
    let path = fx.gguf("d.gguf", &c);
    let tokens = tokens_to(14);
    let mut plain = fx.load(&path, sampling(64));
    let mut cut = fx.load(&path, opts(sampling(64), 10, DraftLmHead::Q4));
    let a = step(&mut plain, 1, &tokens, 0, 13, 7, 0);
    let b = step(&mut cut, 1, &tokens, 0, 13, 7, 0);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.probs, b.probs);
}
