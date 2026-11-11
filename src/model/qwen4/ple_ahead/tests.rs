use super::*;

#[test]
fn held_values_go_to_the_same_rows_only() {
    let mut a = PleAhead::default();
    assert_eq!(a.take(&[1, 2]), None);
    a.put(vec![1, 2], vec![0.5, 1.5]);
    assert_eq!(a.take(&[1, 2]), Some(vec![0.5, 1.5]));
    assert_eq!(a.hits(), 1);
    // Taken once.
    assert_eq!(a.take(&[1, 2]), None);
    // Other rows: nothing, and the held values are gone.
    a.put(vec![1, 2], vec![0.5, 1.5]);
    assert_eq!(a.take(&[1, 3]), None);
    assert_eq!(a.take(&[1, 2]), None);
    assert_eq!(a.hits(), 1);
    // The newest put wins.
    a.put(vec![4], vec![4.0]);
    a.put(vec![5], vec![5.0]);
    assert_eq!(a.take(&[5]), Some(vec![5.0]));
    assert_eq!(a.hits(), 2);
}

#[test]
fn recycled_values_are_the_next_buffer() {
    let mut a = PleAhead::default();
    assert_eq!(a.buffer(3), vec![0.0; 3]);
    let v = vec![1.0f32; 1000];
    let at = v.as_ptr();
    a.recycle(v);
    // The same allocation, at the asked length.
    let b = a.buffer(500);
    assert_eq!((b.len(), b.as_ptr()), (500, at));
    // A smaller one does not replace a bigger spare.
    a.recycle(b);
    a.recycle(vec![2.0; 10]);
    assert_eq!(a.buffer(1000).as_ptr(), at);
    // Taken: the next one is new.
    assert_eq!(a.buffer(2), vec![0.0; 2]);
}

/// The tiny qwen4exp over two CPU stages, the per-layer n-gram embedding in the first, a
/// prompt in micro-chunks: rounds of both stages gather the next micro-chunk's rows ahead,
/// and the logits are bit-identical to the same steps without it.
#[test]
fn rows_gathered_ahead_change_no_bit() {
    use crate::ggml::Backend;
    use crate::model::core::{Executor, Group, Logits, Model, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::qwen4::tiny::write(&path, None, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let prompt: Vec<u32> = (0..70u32).map(|i| 32 + (i * 13 + 7) % 90).collect();
    let run = |ahead: bool| -> (Vec<f32>, u64) {
        let cpu = || {
            let mut b = Backend::cpu().unwrap();
            b.set_threads(2);
            b.set_use_ref(true);
            b
        };
        let (b0, b1) = (cpu(), cpu());
        // Both stages read host memory.
        let host = b0.buffer_type();
        let split = 2;
        assert!(super::super::tiny::N_LAYER > split);
        let model = Model::load(&gguf, |_| host).unwrap();
        let n = model.hp.n_layer;
        let mut exec =
            Executor::new_staged(model, vec![(b0, 0..split), (b1, split..n)], 8, 1, 0).unwrap();
        exec.set_pipeline_micro(16);
        exec.set_ple_ahead(ahead);
        let blocks: Vec<u32> = (0..8).collect();
        let mut rows = Vec::new();
        let mut n_past = 0;
        // Two prompt steps (the second continues the slot's n-gram history), then a decode one.
        for len in [50, 19, 1] {
            let out = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: &prompt[n_past..n_past + len],
                        n_past,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::Last,
                        media: None,
                    }],
                }])
                .unwrap();
            rows.extend(out.rows.into_iter().flatten());
            n_past += len;
        }
        (rows, exec.ple_ahead_hits())
    };
    let (plain, no_hits) = run(false);
    let (ahead, hits) = run(true);
    assert_eq!(no_hits, 0);
    // 50 tokens in 4 micro-chunks and 19 in 2 over two stages: every micro-chunk after the
    // first of a step is filled after a round of both stages.
    assert_eq!(hits, 3 + 1);
    assert!(plain.iter().all(|x| x.is_finite()));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&plain), bits(&ahead));
}
