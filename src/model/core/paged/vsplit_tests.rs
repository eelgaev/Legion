//! Small multi-token groups split over the key blocks (#697, `--paged-verify-split`): the CUDA
//! op given [`split_scratch_shape`]'s scratch against the f64 oracle, at 1 to 16 tokens per
//! sequence, contexts that end before, on and after a block boundary and long ones, GQA 6 and
//! 8, with and without a sliding window. On sm_70 / sm_75 the op ignores the split scratch and
//! runs its own kernels, so the cases hold there too.

use super::*;

/// The split scratch: as many splits as a decode step of as many sequences, a row per token;
/// none when a decode step has one split. With the switch off (the default) multi-token groups
/// keep the placeholder.
#[test]
fn split_scratch_shapes() {
    // Qwen3.8-27B: 4 kv heads, 32 splits for up to four sequences.
    assert_eq!(split_scratch_shape(256, 24, 4, 3, 1), [258, 32, 24, 3]);
    assert_eq!(split_scratch_shape(256, 24, 4, 16, 4), [258, 32, 24, 64]);
    assert_eq!(decode_splits(4, 4), 32);
    assert_eq!(split_scratch_shape(128, 16, 2, 5, 16), [130, 16, 16, 80]);
    assert_eq!(split_scratch_shape(128, 64, 8, 4, 64), [1, 1, 1, 1]);
    assert_eq!(scratch_shape(256, 24, 4, 3, 1), [1, 1, 1, 1]);
    // The switch is off unless `legion serve --paged-verify-split` sets it.
    assert!(!verify_split(3));
}

/// Shuffled block tables (block i of sequence s at pool block (i × n + s) × 7 mod total) for
/// sequences of `len` positions each.
#[cfg(feature = "cuda")]
fn tables(len: &[usize]) -> Vec<Vec<u32>> {
    let width = len.iter().max().unwrap().div_ceil(KV_BLOCK);
    let total = width * len.len();
    assert_ne!(total % 7, 0);
    (0..len.len())
        .map(|s| {
            (0..len[s].div_ceil(KV_BLOCK))
                .map(|i| ((i * len.len() + s) * 7 % total) as u32)
                .collect()
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_verify_split_matches_oracle() {
    // Qwen3.5 / 3.6 / 3.8-27B's heads (GQA 6), GQA 8 at 256 and 128 wide.
    for (d, n_head, n_head_kv) in [(256, 24, 4), (256, 32, 4), (128, 16, 2)] {
        for n_tok in [1, 2, 3, 5, 8, 10, 11, 13, 16] {
            // Contexts (keys of the last token) of 1 (n_tok 1 only), 63, 64, 65 and 5000: one
            // sequence (the most splits), then several at once (fewer splits, mixed lengths).
            let ctx: Vec<usize> = [1, 63, 64, 65, 5000]
                .into_iter()
                .filter(|&c| c >= n_tok)
                .collect();
            let mut runs: Vec<Vec<usize>> = ctx.iter().map(|&c| vec![c]).collect();
            runs.push(ctx.clone());
            for len in &runs {
                let n_past: Vec<usize> = len.iter().map(|&c| c - n_tok).collect();
                let tb = tables(len);
                for (window, scale) in [(0, 1.0 / (d as f32).sqrt()), (100, 0.08)] {
                    let ne = split_scratch_shape(
                        d as i64,
                        n_head as i64,
                        n_head_kv as i64,
                        n_tok,
                        len.len(),
                    );
                    assert!(ne[1] > 1, "{ne:?}");
                    super::tests::cuda_check_with(
                        d, n_head, n_head_kv, n_tok, &n_past, &tb, scale, window, ne,
                    );
                }
            }
        }
    }
}

/// The op's time per pass at Qwen3.8-27B's attention shape (24 heads of 256 over 4 kv heads, one
/// sequence), unsplit and split, for verify-sized groups at short and long contexts.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_verify_split_timing() {
    use super::tests::hparams;
    use crate::ggml::{ComputeAllocator, DeviceKind, device_by_kind};
    use crate::model::core::KvPool;
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let (d, n_head, n_head_kv) = (256usize, 24usize, 4usize);
    let max_ctx = 64 * 1024 + 64;
    let n_blocks = max_ctx / KV_BLOCK;
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let (kp, vp) = kv.pools(0);
    kp.set_bytes(&super::tests::f16_bytes(&super::tests::noise(
        kp.elements(),
        61,
    )));
    vp.set_bytes(&super::tests::f16_bytes(&super::tests::noise(
        vp.elements(),
        62,
    )));
    let table: Vec<i32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as i32).collect();
    let mut time = |n_tok: usize, ctx_len: usize, split: bool| {
        let ctx = Context::new(64, 64);
        let q = ctx
            .new_tensor(types::F32, &[d as i64, n_head as i64, n_tok as i64, 1])
            .as_input();
        let tab = ctx.new_tensor(types::I32, &[n_blocks as i64, 1]).as_input();
        let past = ctx.new_tensor(types::I32, &[1]).as_input();
        let ne = if split {
            split_scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, 1)
        } else {
            [1, 1, 1, 1]
        };
        let scratch = ctx.new_tensor(types::F32, &ne).as_input();
        let out = attention_with(&ctx, q, kp, vp, tab, past, scratch, 1.0 / 16.0, 0, 0).as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        q.set_f32(&super::tests::noise(q.elements(), 63));
        tab.set_i32(&table);
        past.set_i32(&[(ctx_len - n_tok) as i32]);
        backend.compute(&graph).unwrap();
        let reps = 50;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(&graph).unwrap();
        }
        let _ = out.to_f32();
        start.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    for ctx_len in [128, 512, 2048, 16384, 65536] {
        for n_tok in [4, 8, 9, 16] {
            let (full, split) = (time(n_tok, ctx_len, false), time(n_tok, ctx_len, true));
            println!("{ctx_len} keys, {n_tok} tokens: {full:.3} ms -> {split:.3} ms");
        }
    }
}
