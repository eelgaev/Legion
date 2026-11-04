//! The decode and verify step buffers the expert cache leaves room for (#371): with paged
//! sparse attention a batch's buffers grow with the context its sequences read together, so a
//! full batch sharing the KV pool needs at least what any batch within the pool needs, however
//! the context is split among its sequences.
//!
//! `cargo test --release --features cuda --test step_buffers_cuda -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use legion::ggml::{DeviceKind, device_by_kind};
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Model, StepPart};

#[test]
#[ignore = "needs a CUDA GPU"]
fn a_batch_sharing_the_pool_bounds_every_split() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-qwen4exp.gguf");
    legion::model::qwen4::tiny::write_cuda_indexer(&path, None, 3).unwrap();
    let dev = device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let model = Model::load(&GgufFile::open(&path).unwrap(), |_| dev.buffer_type()).unwrap();
    let (seqs, pool_tokens) = (8usize, 8192usize);
    let mut e = Executor::new(model, dev.init().unwrap(), pool_tokens / 64, seqs, 0).unwrap();
    e.set_max_seq_tokens(pool_tokens);
    e.enable_spec(3).unwrap();
    assert!(
        e.steps_paged(),
        "the tiny qwen4exp should page its sparse attention on CUDA"
    );
    for (n_tok, verify) in [(1, false), (3, true)] {
        let part = |n_seqs, n_past| StepPart {
            n_seqs,
            n_tok,
            n_past,
            verify,
        };
        let shared = e
            .measure_groups(&[part(seqs, pool_tokens / seqs - n_tok)])
            .unwrap();
        for split in [
            vec![part(1, pool_tokens - n_tok)],
            vec![
                part(1, pool_tokens / 2),
                part(seqs - 1, pool_tokens / (2 * seqs)),
            ],
            vec![part(2, pool_tokens / 4), part(seqs - 2, 64)],
        ] {
            let got = e.measure_groups(&split).unwrap();
            for (s, (g, b)) in got.iter().zip(&shared).enumerate() {
                assert!(
                    g <= b,
                    "{n_tok} tokens, stage {s}: a split needs {g} bytes, the shared batch {b}"
                );
            }
        }
    }
}
