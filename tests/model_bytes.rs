//! Model bytes (#391) on a tiny model: a step's counted bytes are the bytes of the weights its
//! graph reads (every tensor once, the embedding table by the rows looked up) and the KV its
//! attention reads (every position so far, the step's own included), by kind of step.

use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
use legion::model::qwen35::tiny::{self, TinyConfig};

fn group<'a>(tokens: &'a [u32], n_past: usize, blocks: &'a [u32]) -> Group<'a> {
    Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens,
            n_past,
            blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    }
}

#[test]
fn counted_bytes_are_the_weights_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut b = Backend::cpu().unwrap();
    b.set_threads(2);
    let model = Model::load(&gguf, |_| b.buffer_type()).unwrap();
    // Every weight once, but of the embedding table only the rows looked up.
    let mut all = 0u64;
    let mut emb_row = 0u64;
    for t in gguf.tensors() {
        let bytes = model.tensor(&t.name).unwrap().nbytes() as u64;
        if t.name == "token_embd.weight" {
            emb_row = bytes / t.shape[1];
        } else {
            all += bytes;
        }
    }
    let mut e = Executor::new(model, b, 4, 1, 0).unwrap();
    let blocks = [0u32, 1, 2, 3];
    let prompt: Vec<u32> = "counting the bytes a step reads"
        .bytes()
        .map(u32::from)
        .collect();
    let n = prompt.len() as u64;
    e.step(&[group(&prompt, 0, &blocks)]).unwrap();
    let t = e.model_bytes();
    assert_eq!(t.prefill.steps, 1);
    assert_eq!(t.prefill.dense, all + n * emb_row, "prompt step");
    assert_eq!(t.prefill.experts_upper, 0);
    assert!(t.prefill.seconds > 0.0);
    // The attention layers' K and V of every position: the same bytes per position.
    assert!(t.prefill.kv > 0);
    assert_eq!(t.prefill.kv % n, 0);
    assert_eq!(t.prefill.kv_host, 0);
    assert_eq!(t.prefill.total(), t.prefill.dense + t.prefill.kv);
    assert_eq!(t.stages[0].prefill.kv, t.prefill.kv);
    assert_eq!(t.stages[0].prefill.dense, t.prefill.dense);
    assert_eq!(t.decode.steps, 0);
    e.step(&[group(&[b'!' as u32], prompt.len(), &blocks)])
        .unwrap();
    let t2 = e.model_bytes();
    assert_eq!(t2.prefill, t.prefill);
    assert_eq!(t2.decode.steps, 1);
    assert_eq!(t2.decode.dense, all + emb_row, "decode step");
    assert_eq!(
        t2.decode.kv,
        t.prefill.kv / n * (n + 1),
        "decode reads n + 1 positions"
    );
    assert!(t2.decode.gb_s() > 0.0);
}
