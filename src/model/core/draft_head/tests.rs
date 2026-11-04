//! The MTP head drafting with a 4-bit copy of the output head (`--spec-draft-lm-head q4`)
//! against the head's hidden output times the copy's dequantized rows, on the tiny qwen35 on
//! the CPU.

use crate::ggml::{Backend, types};
use crate::gguf::GgufFile;
use crate::model::core::{Executor, Group, Logits, Model, SeqInput};
use crate::model::qwen35::tiny::{self, TinyConfig, TinyNvfp4};
use crate::spec::DraftLmHead;

/// The tiny model with an MTP head, its head drafting with the 4-bit copy of the first `vocab`
/// rows (0: all) when `q4`.
fn executor(path: &std::path::Path, vocab: usize, q4: bool) -> Executor {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    let mut m = Model::load(&GgufFile::open(path).unwrap(), |_| b.buffer_type()).unwrap();
    m.set_draft_vocab(vocab);
    let n_layer = m.hp.n_layer;
    let mut e = Executor::new(m, b, 4, 1, 0).unwrap();
    if q4 {
        e.set_mtp_lm_head(DraftLmHead::Q4);
    }
    e.enable_mtp().unwrap();
    e.capture_hidden(vec![n_layer - 1]);
    e
}

/// One pass at position 0: the head's (argmax, pmax, output hidden).
fn pass(e: &mut Executor) -> (u32, f32, Vec<f32>) {
    let w = e.mtp_width();
    let h: Vec<f32> = (0..w)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
        .collect();
    let (tok, blocks) = ([42u32], [0u32, 1, 2, 3]);
    let g = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &tok,
            n_past: 0,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let out = e.mtp_step(&[g], &h).unwrap();
    (out.argmax[0], out.pmax[0], out.hidden)
}

/// The output head's values (scale folded in) as F32, row-major.
fn head_values(m: &Model) -> Vec<f32> {
    let w = m.tensor("output.weight").unwrap();
    let mut v = match w.ty() {
        types::F32 => w.to_f32(),
        ty => crate::ggml::dequantize(ty, &w.bytes(), w.elements()),
    };
    if let Some(s) = m.tensor("output.scale") {
        let s = s.to_f32();
        let k = w.ne()[0] as usize;
        for (r, row) in v.chunks_mut(k).enumerate() {
            let s = s[if s.len() == 1 { 0 } else { r }];
            row.iter_mut().for_each(|x| *x *= s);
        }
    }
    v
}

fn check(path: &std::path::Path, vocab: usize) {
    let mut full = executor(path, vocab, false);
    let mut e = executor(path, vocab, true);
    let m = e.model();
    let n_vocab = m.hp.n_vocab as usize;
    let rows = if vocab == 0 { n_vocab } else { vocab };
    let head = m.draft_head().expect("a 4-bit copy");
    assert_eq!(head.ty(), types::Q4_0, "64-wide rows: Q4_0");
    assert_eq!(head.ne()[1] as usize, rows);
    // What the copy should hold: the head's rows quantized the same way.
    let k = head.ne()[0] as usize;
    let values = head_values(m);
    let bytes: Vec<u8> = values[..rows * k]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let (_, want) = crate::spec::q4_head::to_q4(types::F32, &bytes, None, k).unwrap();
    assert_eq!(head.bytes(), want, "the copy's bytes");
    let rows_q4 = crate::ggml::dequantize(types::Q4_0, &want, rows * k);

    let (a_full, _, h_full) = pass(&mut full);
    let (a, p, h) = pass(&mut e);
    // The head's layer is untouched: only its output matrix changed.
    assert_eq!(h, h_full);
    assert!((a_full as usize) < rows);
    // Greedy pick and probability: the copy's rows times the head's output.
    let logits: Vec<f32> = rows_q4
        .chunks(k)
        .map(|r| r.iter().zip(&h).map(|(x, y)| x * y).sum())
        .collect();
    let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(best - logits[a as usize] < 1e-4, "argmax {a}");
    let z: f32 = logits.iter().map(|l| (l - best).exp()).sum();
    assert!((p - 1.0 / z).abs() < 1e-4, "pmax {p} vs {}", 1.0 / z);
}

#[test]
fn the_mtp_head_drafts_with_the_4_bit_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-mtp.gguf");
    let cfg = TinyConfig {
        seed: 7,
        mtp: true,
        ..Default::default()
    };
    tiny::write(&path, &cfg).unwrap();
    for vocab in [0, 40] {
        check(&path, vocab);
    }
}

/// An NVFP4 output head with a per-tensor scale: the copy has the scale folded in.
#[test]
fn an_nvfp4_head_is_copied_with_its_scale() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-mtp-nvfp4.gguf");
    let cfg = TinyConfig {
        seed: 7,
        mtp: true,
        nvfp4: Some(TinyNvfp4 {
            scale_exp: Some(-1),
            block_exp_shift: 0,
        }),
        ..Default::default()
    };
    tiny::write(&path, &cfg).unwrap();
    check(&path, 40);
}

#[test]
fn without_q4_there_is_no_copy() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-mtp.gguf");
    tiny::write(
        &path,
        &TinyConfig {
            mtp: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(executor(&path, 40, false).model().draft_head().is_none());
}
