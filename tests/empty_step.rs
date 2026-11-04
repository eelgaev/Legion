//! A step with nothing left to run is a no-op: at shutdown with requests in flight, every
//! prompt of a step can be cancelled before it runs, and the step's graph builder used to
//! panic on the empty batch ("empty batch", core.rs).

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
use legion::model::gemma4::tiny;

#[test]
fn a_step_without_tokens_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(2);
    cpu.set_use_ref(true);
    let model = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let mut exec = Executor::new(model, cpu, 4, 2, 0).unwrap();
    // No group at all, and a group whose only sequence has no token.
    let out = exec.step(&[]).unwrap();
    assert!(out.rows.is_empty() && out.argmax.is_empty());
    let blocks = [0u32];
    let empty = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &[],
            n_past: 0,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let out = exec.step(std::slice::from_ref(&empty)).unwrap();
    assert!(out.rows.is_empty());
    // A real step still runs after them.
    let tokens = [1u32, 2, 3];
    let step = Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens: &tokens,
            n_past: 0,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    let out = exec.step(std::slice::from_ref(&step)).unwrap();
    assert_eq!(out.rows.len(), 1);
}
