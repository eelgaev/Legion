use crate::ggml::Backend;
use crate::gguf::GgufFile;
use crate::model::core::{Executor, Group, Logits, Model, Pool, SeqInput};

use super::StateCapture;

const PROMPT: usize = 320;
const AT: usize = 128;

fn prompt() -> Vec<u32> {
    (0..PROMPT as u32).map(|i| 32 + (i * 13 + 7) % 90).collect()
}

/// The tiny qwen4exp (DeltaNet layers, the per-layer embedding's conv and n-gram history) on
/// the CPU: two sequence slots, two saved ones.
fn exec(gguf: &GgufFile) -> Executor {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    b.set_use_ref(true);
    let host = b.buffer_type();
    let model = Model::load(gguf, |_| host).unwrap();
    Executor::new(model, b, 8, 2, 2).unwrap()
}

/// One prompt step of `prompt[from..to]` in `slot`: the last token's logits.
fn step(e: &mut Executor, slot: usize, prompt: &[u32], from: usize, to: usize) -> Vec<f32> {
    let blocks: Vec<u32> = (0..8).collect();
    let out = e
        .step(&[Group {
            slot,
            verify: false,
            seqs: vec![SeqInput {
                tokens: &prompt[from..to],
                n_past: from,
                blocks: &blocks,
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
        }])
        .unwrap();
    out.rows.concat()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A state saved inside a prompt step: the step's logits are those of the same step without
/// it, and a prompt resumed from it gives the logits of a run whose step ended there (the
/// state the old save points copied) bit for bit.
#[test]
fn a_state_saved_inside_a_step_resumes_as_one_saved_at_a_step_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::qwen4::tiny::write(&path, None, 5).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let prompt = prompt();

    let mut plain = exec(&gguf);
    let whole = step(&mut plain, 0, &prompt, 0, PROMPT);

    let mut cap = exec(&gguf);
    assert!(cap.can_capture_states());
    let asked = StateCapture {
        slot: 0,
        at: AT,
        saved: 1,
    };
    cap.set_state_captures(vec![asked]);
    let captured = step(&mut cap, 0, &prompt, 0, PROMPT);
    assert_eq!(cap.take_applied_captures(), [asked]);
    assert_eq!(bits(&captured), bits(&whole), "a capture changed the step");

    // The old way: a step ending at the point, then the rest.
    let mut cut = exec(&gguf);
    step(&mut cut, 0, &prompt, 0, AT);
    let reference = step(&mut cut, 0, &prompt, AT, PROMPT);

    // Resumed in the other slot from the saved state (the K/V of the first tokens is the
    // whole step's, in the same blocks).
    cap.copy_states(&[(Pool::Saved, 1, Pool::Seq, 1)]).unwrap();
    let resumed = step(&mut cap, 1, &prompt, AT, PROMPT);
    assert_eq!(bits(&resumed), bits(&reference), "resumed from the capture");
}

/// A point the step cannot split at (not a multiple of the alignment past the step's start) is
/// not taken, and the step runs as without it.
#[test]
fn an_unaligned_point_is_not_taken() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::qwen4::tiny::write(&path, None, 5).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let prompt = prompt();
    let mut e = exec(&gguf);
    e.set_state_captures(vec![StateCapture {
        slot: 0,
        at: 100,
        saved: 0,
    }]);
    step(&mut e, 0, &prompt, 0, PROMPT);
    assert!(e.take_applied_captures().is_empty());
    // Requests last one step only.
    e.set_state_captures(vec![StateCapture {
        slot: 1,
        at: AT,
        saved: 0,
    }]);
    step(&mut e, 0, &prompt, 0, 64);
    assert!(e.take_applied_captures().is_empty());
}
