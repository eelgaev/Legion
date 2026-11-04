//! [`attention_last`] (#436): the last token's rows of each sequence, bit for bit what
//! [`attention_tiered`] gives them, in the same graph over the same cache. On sm_70 / sm_75 the
//! CUDA op computes them split over the key blocks (sm70/paged_split.cuh) instead of in the
//! WMMA prefill kernel; elsewhere (the CPU, sm_80+) it runs the same kernel.

use super::tests::{f16_bytes, hparams, noise};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvPool;

/// One graph with both ops over `tables` (per sequence, padded to the widest) and `n_past`,
/// the last token's rows of each sequence compared bit for bit; `split_seqs`: sequences the
/// split scratch holds.
#[allow(clippy::too_many_arguments)]
fn case(
    backend: &mut Backend,
    kv: &KvPool,
    (d, n_head, n_head_kv): (usize, usize, usize),
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
    scale: f32,
    window: usize,
    split_seqs: usize,
) {
    let (kp, vp) = kv.pools(0);
    let ns = n_past.len();
    let width = tables.iter().map(Vec::len).max().unwrap();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, ns);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let split_len = split_floats(split_seqs, n_head_kv, width);
    let split = ctx.new_tensor(types::F32, &[split_len as i64]).as_input();
    let full = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0).as_output();
    let last = attention_last(
        &ctx, q, kp, vp, tab, past, scratch, scale, window, None, split,
    )
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(full);
    graph.expand(last);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let qmul = 2.0 * scale * (d as f32).sqrt().min(16.0);
    let qv: Vec<f32> = noise(q.elements(), 31).iter().map(|x| qmul * x).collect();
    q.set_f32(&qv);
    let mut t = Vec::new();
    for (s, tb) in tables.iter().enumerate() {
        assert!(tb.len() * KV_BLOCK >= n_past[s] + n_tok);
        t.extend(tb.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
    }
    tab.set_i32(&t);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    // Garbage in the split scratch: every value it holds is written before it is read.
    split.set_f32(&vec![f32::NAN; split_len]);
    backend.compute(&graph).unwrap();

    let (a, b) = (full.to_f32(), last.to_f32());
    let row = d * n_head;
    for (s, &np) in n_past.iter().enumerate() {
        let at = (s * n_tok + n_tok - 1) * row;
        let (x, y) = (&a[at..at + row], &b[at..at + row]);
        let differ = x
            .iter()
            .zip(y)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differ, 0,
            "d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, n_past {}, window {window}, \
             {split_seqs} scratch sequences: {differ} of {row} values differ (e.g. {:?} vs {:?})",
            np, x[0], y[0]
        );
        assert!(x.iter().all(|v| v.is_finite()), "non-finite output");
    }
}

/// A pool of `n_blocks` blocks of noise (V in [-1, 1)).
fn pool(backend: &Backend, d: usize, n_head: usize, n_head_kv: usize, n_blocks: usize) -> KvPool {
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let (kp, vp) = kv.pools(0);
    kp.set_bytes(&f16_bytes(&noise(kp.elements(), 41)));
    vp.set_bytes(&f16_bytes(&noise(vp.elements(), 42)));
    kv
}

/// Shuffled tables (one padded past its blocks), causal rows that start mid-block, windows that
/// start inside blocks, every GQA ratio the kernels take, heads of 128, 256 and 512, and a
/// scratch for fewer sequences than the group has (run in turns); `shapes` (head size, heads,
/// kv heads) and `n_toks` of them.
fn edges(backend: &mut Backend, shapes: &[(usize, usize, usize)], n_toks: &[usize]) {
    let tables = [vec![5, 0, 9, 3, 7, 11], vec![2, 10], vec![8, 1, 6, 4]];
    let n_past = [190, 3, 120];
    for &(d, n_head, n_head_kv) in shapes {
        let kv = pool(backend, d, n_head, n_head_kv, 12);
        for &n_tok in n_toks {
            for (window, scale) in [
                (0, 1.0 / (d as f32).sqrt()),
                (100, 1.0 / (d as f32).sqrt()),
                (64, 0.08),
            ] {
                for split_seqs in [3, 1] {
                    case(
                        backend,
                        &kv,
                        (d, n_head, n_head_kv),
                        n_tok,
                        &n_past,
                        &tables,
                        scale,
                        window,
                        split_seqs,
                    );
                }
            }
        }
    }
}

/// The plumbing on the CPU (the split scratch among the op's sources, its parameters): the
/// same result as without it.
#[test]
fn cpu_last_rows_match() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    edges(&mut backend, &[(128, 8, 2)], &[2, 7]);
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// The prefill kernel's edges, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_last_rows_bitwise() {
    println!("split path on this device: {}", split_ok());
    let shapes = [
        (128, 32, 2),
        (128, 8, 8),
        (256, 16, 8),
        (256, 16, 4),
        (256, 24, 4),
        (256, 24, 2),
        (256, 16, 2),
        (512, 16, 2),
        (512, 32, 4),
    ];
    edges(&mut gpu(), &shapes, &[2, 3, 7, 37, 70]);
}

/// Long contexts (qwen4exp's MTP head shape, 24 heads of 256 over 2 kv heads), bit for bit,
/// with the time of a take-in pass of 1 to 3 tokens through either op.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_long_context_bitwise_and_timing() {
    let mut backend = gpu();
    println!("split path on this device: {}", split_ok());
    let (d, n_head, n_head_kv) = (256, 24, 2);
    let max_ctx = 96 * 1024;
    let n_blocks = max_ctx / KV_BLOCK;
    let kv = pool(&backend, d, n_head, n_head_kv, n_blocks);
    let scale = 1.0 / (d as f32).sqrt();
    // Sequence block i in pool block i * 37 mod n_blocks (a 2,048-block table, as at 128K).
    let width = 2048;
    let table: Vec<u32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as u32).collect();
    for ctx_len in [1024usize, 8192, 32768, 95 * 1024] {
        for n_tok in [2, 3] {
            let n_past = ctx_len - n_tok;
            let tables = [table[..ctx_len.div_ceil(KV_BLOCK)].to_vec()];
            case(
                &mut backend,
                &kv,
                (d, n_head, n_head_kv),
                n_tok,
                &[n_past],
                &tables,
                scale,
                0,
                1,
            );
            let full = time(&mut backend, &kv, n_tok, n_past, &table, width, false);
            let split = time(&mut backend, &kv, n_tok, n_past, &table, width, true);
            let decode = time(&mut backend, &kv, 1, n_past, &table, width, false);
            println!(
                "{ctx_len} keys, {n_tok} tokens: {full:.3} ms -> {split:.3} ms \
                 (a 1-token decode pass: {decode:.3} ms)"
            );
        }
    }
}

/// Milliseconds per run of one op (`last`: [`attention_last`]) over a `width`-entry table.
#[cfg(feature = "cuda")]
fn time(
    backend: &mut Backend,
    kv: &KvPool,
    n_tok: usize,
    n_past: usize,
    table: &[u32],
    width: usize,
    last: bool,
) -> f64 {
    let (d, n_head, n_head_kv) = (256i64, 24i64, 2usize);
    let (kp, vp) = kv.pools(0);
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(types::F32, &[d, n_head, n_tok as i64, 1])
        .as_input();
    let tab = ctx.new_tensor(types::I32, &[width as i64, 1]).as_input();
    let past = ctx.new_tensor(types::I32, &[1]).as_input();
    let scratch_ne = scratch_shape(d, n_head, n_head_kv as i64, n_tok, 1);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let split_len = split_floats(1, n_head_kv, width);
    let split = ctx.new_tensor(types::F32, &[split_len as i64]).as_input();
    let scale = 1.0 / 16.0;
    let out = if last {
        attention_last(&ctx, q, kp, vp, tab, past, scratch, scale, 0, None, split)
    } else {
        attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, 0, 0)
    }
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    graph.expand(split);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    q.set_f32(&noise(q.elements(), 51));
    let mut t: Vec<i32> = table.iter().map(|&b| b as i32).collect();
    t.resize(width, 0);
    tab.set_i32(&t);
    past.set_i32(&[n_past as i32]);
    backend.compute(&graph).unwrap();
    let reps = 10;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        backend.compute(&graph).unwrap();
    }
    let _ = out.to_f32();
    start.elapsed().as_secs_f64() * 1e3 / reps as f64
}
