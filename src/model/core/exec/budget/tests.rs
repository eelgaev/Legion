use super::*;

const MIB: usize = 1 << 20;

/// #430 as measured on an AC922 (Qwen3.8-Flash-Next over two V100s, MTP k=2, 16 sequences),
/// stage 0: a 512-token prompt step 182 MiB, the verify batch 62, a decode batch 27, so the
/// expert cache left 2 x 182 + 62 + 27 + 256 = 709 MiB. Batched prompts made 1024-token steps
/// of ~370 MiB, and all three kept mixed buffers grew to that: 1.1 GiB with the two decode
/// graphs, out of the reserve beside the cache, until a step's buffer did not fit.
#[test]
fn drops_the_oldest_mixed_graphs_past_the_budget() {
    let budget = (2 * 182 + 62 + 27 + 256) * MIB;
    let decode = 2 * 27 * MIB;
    let big = 370 * MIB;
    assert_eq!(evictions(&[big, big, big], decode, true, budget), 2);
    // A big step, the verify batch, another big step: the oldest big one goes.
    assert_eq!(evictions(&[big, 62 * MIB, big], decode, true, budget), 1);
}

/// The steps the budget was measured for keep all their graphs.
#[test]
fn measured_steps_keep_every_graph() {
    let budget = (2 * 182 + 62 + 27 + 256) * MIB;
    let decode = 2 * 27 * MIB;
    assert_eq!(
        evictions(&[182 * MIB, 182 * MIB, 62 * MIB], decode, true, budget),
        0
    );
    assert_eq!(evictions(&[], decode, true, budget), 0);
    assert_eq!(evictions(&[], 0, false, 0), 0);
}

/// The graph just built stays even alone over the budget; after a decode graph was built every
/// mixed one may go.
#[test]
fn the_newest_mixed_graph_stays() {
    assert_eq!(evictions(&[100, 900], 0, true, 500), 1);
    assert_eq!(evictions(&[900], 0, true, 500), 0);
    assert_eq!(evictions(&[300, 300], 600, false, 500), 2);
    assert_eq!(evictions(&[300, 300], 100, false, 500), 1);
}

/// On an executor (the tiny Gemma 4 on the CPU): prompt steps of four shapes, each a new
/// mixed graph. Under a budget of one byte a stage keeps only the newest, without one the
/// last three; the tokens are the same.
#[test]
fn a_budget_bounds_the_kept_graphs_not_the_tokens() {
    use crate::ggml::Backend;
    use crate::model::core::{Attention, Executor, Group, Logits, Model, Placement, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    crate::model::gemma4::tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let run = |budget: usize| -> (Vec<u32>, Vec<usize>) {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let host = Backend::cpu().unwrap().buffer_type();
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            Placement::Host => host,
            Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let mut exec = Executor::new(m, b, 8, 2, 0).unwrap();
        exec.set_attention(Attention::Gather);
        if budget > 0 {
            exec.set_compute_budget(vec![budget]);
        }
        let blocks: Vec<u32> = (0..8).collect();
        let prompt: Vec<u32> = (0..100).map(|i| (i * 7 + 3) % 50).collect();
        let (mut out, mut kept) = (Vec::new(), Vec::new());
        let mut n_past = 0;
        for len in [40, 25, 20, 15] {
            let toks = &prompt[n_past..n_past + len];
            let r = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::Greedy,
                        media: None,
                    }],
                }])
                .unwrap();
            out.push(r.argmax[0]);
            kept.push(exec.stages[0].mixed.len());
            n_past += len;
        }
        (out, kept)
    };
    let (free, kept_free) = run(0);
    let (bounded, kept_bounded) = run(1);
    assert_eq!(free, bounded);
    assert_eq!(kept_free, [1, 2, 3, 3]);
    assert_eq!(kept_bounded, [1, 1, 1, 1]);
}
