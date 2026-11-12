//! The queries-as-M prefill kernel (`src/cuda/sm80/paged_attn_qm.cuh`, #1133) against
//! `paged_prefill_mma`, the kernel it stands in for on compute capability 12.x: the op over the
//! same inputs with the kernel off and on for the calling thread ([`set_qm_prefill_here`]),
//! every output value compared bit for bit, and the time of either. Used by `legion
//! paged-qm-bench`, which runs them from a release build on the GPU it is given. Elsewhere (other
//! GPUs, the CPU) both runs take the same kernel and the launch count says so.

use super::wmma_bench::{Group, fill};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::KvPool;

/// One run of the op for `g` over `kv` with the kernel in mode `mode` for this thread (0 off, 1
/// the default, 2 on for every group it covers, [`set_qm_mode_here`]): its output,
/// its milliseconds (best of `rounds` rounds of `reps` runs after the first; 0 without), and
/// whether the queries-as-M kernel ran.
pub fn run(
    backend: &mut Backend,
    kv: &KvPool,
    g: &Group,
    mode: i32,
    reps: usize,
    rounds: usize,
) -> (Vec<f32>, f64, bool) {
    set_qm_mode_here(mode);
    let (kp, vp) = kv.pools(0);
    let (d, n_head, _) = g.heads;
    let ns = g.n_past.len();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, g.n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[g.width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch = ctx.new_tensor(types::F32, &[1, 1, 1, 1]).as_input();
    let out = attention_tiered(
        &ctx, q, kp, vp, tab, past, scratch, g.scale, g.window, g.ring, None,
    )
    .as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .expect("graph allocation");
    fill(q, tab, past, g);
    // Garbage in the output: every value the kernels write they compute.
    out.set_f32(&vec![f32::NAN; out.elements()]);
    let before = qm_launches();
    backend.compute(&graph).expect("compute");
    let ran = qm_launches() > before;
    let first = out.to_f32();
    let mut best = 0.0f64;
    if reps > 0 {
        best = f64::INFINITY;
        for _ in 0..rounds.max(1) {
            let start = std::time::Instant::now();
            for _ in 0..reps {
                backend.compute(&graph).expect("compute");
            }
            best = best.min(start.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
    }
    set_qm_prefill_here(None);
    (first, best, ran)
}

/// Values of the op's output (rows the kernels write) whose bits differ between the kernel off
/// and in mode `mode`, of how many, and whether the queries-as-M kernel ran. Panics on a
/// non-finite output.
pub fn compare(backend: &mut Backend, kv: &KvPool, g: &Group, mode: i32) -> (usize, usize, bool) {
    let (a, _, _) = run(backend, kv, g, 0, 0, 0);
    let (b, _, ran) = run(backend, kv, g, mode, 0, 0);
    assert!(
        a.iter().chain(&b).all(|v| v.is_finite()),
        "non-finite output"
    );
    let differ = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    (differ, a.len(), ran)
}

/// The edge cases for heads of 256 with the kernel in mode `mode`: every GQA ratio, groups from
/// 2 tokens (verify steps) to more than one 128-row tile per token run, several sequences over
/// shuffled tables, contexts that end before, on and after block boundaries, windows that start
/// inside blocks (Gemma 4's sliding window of 1,024 among them), a ring. Returns (cases, cases
/// the kernel ran, values compared, values that differ), printing each failing case.
pub fn edges(backend: &mut Backend, mode: i32) -> (usize, usize, usize, usize) {
    let (mut cases, mut ran_cases, mut total, mut differ) = (0, 0, 0, 0);
    let n_blocks = 160;
    for (n_head, n_head_kv) in [
        (16, 16),
        (32, 16),
        (16, 4),
        (24, 4),
        (16, 2),
        (24, 2),
        (32, 2),
    ] {
        let heads = (256, n_head, n_head_kv);
        let kv = super::wmma_bench::pool(backend, heads, n_blocks);
        let g_ratio = n_head / n_head_kv;
        let rsd = 1.0 / 16.0;
        // Verify-sized groups (2 to 64 tokens), then prompt chunks around the old threshold of
        // 256 rows and past one 128-row tile per token run.
        let sizes = [2, 3, 8, 11, 16, 21, 33, 64, 256 / g_ratio + 5, 300];
        for n_tok in sizes {
            for (n_past, window, scale) in [
                (vec![0], 0, rsd),
                (vec![447, 3, 190], 0, rsd),
                (vec![512 - (n_tok % 64), 64], 0, rsd),
                (vec![500, 129, 63], 100, rsd),
                (vec![1800, 2], 1024, rsd),
                (vec![449, 60, 250], 64, 0.08),
            ] {
                let tables: Vec<Vec<u32>> = n_past
                    .iter()
                    .enumerate()
                    .map(|(s, &p)| {
                        let len = (p + n_tok).div_ceil(KV_BLOCK);
                        (0..len)
                            .map(|i| ((i * 37 + s * 11 + 5) % n_blocks) as u32)
                            .collect()
                    })
                    .collect();
                let width = tables.iter().map(Vec::len).max().unwrap_or(1);
                let g = Group {
                    heads,
                    n_tok,
                    n_past: &n_past,
                    tables: &tables,
                    width,
                    window,
                    ring: 0,
                    scale,
                };
                let (x, n, ran) = compare(backend, &kv, &g, mode);
                cases += 1;
                ran_cases += usize::from(ran);
                total += n;
                differ += x;
                if x > 0 {
                    println!(
                        "DIFFER {x} of {n}: heads {heads:?}, {n_tok} tokens, n_past {n_past:?}, \
                         window {window}"
                    );
                }
            }
            // A ring under a window of 100 positions, past its first lap.
            let ring = (100 + n_tok).div_ceil(KV_BLOCK) + 2;
            let ring_tables: Vec<Vec<u32>> = (0..2)
                .map(|s| {
                    (0..ring)
                        .map(|i| ((i * 7 + s * 50 + 3) % n_blocks) as u32)
                        .collect()
                })
                .collect();
            let n_past = [3000, 77];
            let g = Group {
                heads,
                n_tok,
                n_past: &n_past,
                tables: &ring_tables,
                width: ring,
                window: 100,
                ring,
                scale: rsd,
            };
            let (x, n, ran) = compare(backend, &kv, &g, mode);
            cases += 1;
            ran_cases += usize::from(ran);
            total += n;
            differ += x;
            if x > 0 {
                println!("DIFFER {x} of {n}: heads {heads:?}, {n_tok} tokens, ring {ring}");
            }
        }
    }
    (cases, ran_cases, total, differ)
}
