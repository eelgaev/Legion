//! `gdn_scan` against ggml's `gated_delta_net`: on the CPU (its reference implementation, to
//! rounding), and on CUDA (ignored tests) bit for bit on compute capability 7.x.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

/// A case: value heads `h`, q/k heads `n_qk`, tokens `nt`, sequences `ns`.
type Case = (i64, i64, i64, i64);

/// `gdn_scan` and ggml's `gated_delta_net` over the same inputs on `backend`, with q, k and v
/// views of one conv-like row per token as a DeltaNet layer has them: (op, ggml), the tokens'
/// outputs then the final states, and the milliseconds of `reps` more runs of each (0: none).
fn case(
    backend: &mut Backend,
    (h, n_qk, nt, ns): Case,
    reps: usize,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let s = HEAD;
    let row = 2 * s * n_qk + s * h;
    let ctx = Context::new(32, 32);
    let conv = ctx.new_tensor(types::F32, &[row, nt, ns]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, h, nt, ns]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, h, nt, ns]).as_input();
    let state = ctx.new_tensor(types::F32, &[s, s, h, ns]).as_input();
    let es = 4usize;
    let nb = [s as usize * es, row as usize * es, (row * nt) as usize * es];
    let q = ctx.view_4d(conv, [s, n_qk, nt, ns], nb, 0);
    let k = ctx.view_4d(conv, [s, n_qk, nt, ns], nb, (s * n_qk) as usize * es);
    let v = ctx.view_4d(conv, [s, h, nt, ns], nb, (2 * s * n_qk) as usize * es);
    let op = gdn_scan(&ctx, q, k, v, g, beta, state).as_output();
    let reference = ctx.gated_delta_net(q, k, v, g, beta, state, 1).as_output();
    assert_eq!(op.ne(), reference.ne(), "the op keeps ggml's shape");
    let mut op_graph = ctx.new_graph(16);
    op_graph.expand(op);
    let mut ref_graph = ctx.new_graph(16);
    ref_graph.expand(reference);
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(reference);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // q and k unit rows (as the l2 norms leave them), g < 0 (a decay), beta in (0, 1).
    let mut x = noise((row * nt * ns) as usize, 1);
    for tok in x.chunks_mut(row as usize) {
        for head in tok[..(2 * s * n_qk) as usize].chunks_mut(s as usize) {
            let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            head.iter_mut().for_each(|v| *v /= n);
        }
    }
    conv.set_f32(&x);
    let gb = (h * nt * ns) as usize;
    g.set_f32(
        &noise(gb, 2)
            .iter()
            .map(|v| 0.5 * (v - 1.0))
            .collect::<Vec<_>>(),
    );
    beta.set_f32(
        &noise(gb, 3)
            .iter()
            .map(|v| 0.5 * (v + 1.0))
            .collect::<Vec<_>>(),
    );
    state.set_f32(
        &noise((s * s * h * ns) as usize, 4)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>(),
    );
    backend.compute(&graph).unwrap();
    let mut time = |graph: &crate::ggml::Graph<'_>| {
        if reps == 0 {
            return 0.0;
        }
        backend.compute(graph).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(graph).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let (op_ms, ref_ms) = (time(&op_graph), time(&ref_graph));
    (op.to_f32(), reference.to_f32(), op_ms, ref_ms)
}

/// Prompt-sized groups: chunk tails (37 tokens), several sequences, ggml's q/k head broadcast.
const CASES: [Case; 4] = [(4, 2, 37, 1), (2, 2, 16, 3), (6, 3, 70, 2), (32, 16, 20, 1)];

#[test]
fn gdn_scan_matches_ggml() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for c in CASES {
        let (op, reference, _, _) = case(&mut backend, c, 0);
        let r = rel(&op, &reference);
        assert!(r < 1e-5, "{c:?}: {r:e}");
    }
}

#[test]
fn fits_only_a_7x_gpu_prompt_group() {
    let ctx = Context::new(8, 8);
    let s = HEAD;
    let q = ctx.new_tensor(types::F32, &[s, 2, 64, 1]);
    let v = ctx.new_tensor(types::F32, &[s, 4, 64, 1]);
    let g = ctx.new_tensor(types::F32, &[1, 4, 64, 1]);
    let state = ctx.new_tensor(types::F32, &[s, s, 4, 1]);
    let backend = Backend::cpu().unwrap();
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    // A state in host memory (the CPU graph) keeps ggml's op whatever the GPU.
    assert!(!fits(q, v, g, state, 64, true));
    assert!(!fits(q, v, g, state, 1, true));
    assert!(!fits(q, v, g, state, 64, false));
}

/// On a 7.x GPU the op is ggml-cuda's kernel bit for bit; elsewhere (ggml may run its chunked
/// kernel there) it agrees to rounding. With the time of each at qwen4exp's shape (48 value
/// heads, 16 q/k heads) and its prompt steps' 2058 and 4096 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gdn_scan_is_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let sm7x = crate::residency::cuda::caps(0).is_some_and(|c| c.cc_major == 7);
    for c in CASES
        .into_iter()
        .chain([(48, 16, 4096, 1), (48, 16, 2058, 1), (32, 16, 1000, 2)])
    {
        let reps = if c.0 == 48 { 10 } else { 0 };
        let (op, reference, op_ms, ref_ms) = case(&mut gpu, c, reps);
        if reps > 0 {
            println!("gdn_scan {c:?}: {op_ms:.3} ms, ggml gated_delta_net {ref_ms:.3} ms");
        }
        if sm7x {
            let differ = op
                .iter()
                .zip(&reference)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                differ,
                0,
                "{c:?}: {differ} values differ (rel {:e})",
                rel(&op, &reference)
            );
        } else {
            let r = rel(&op, &reference);
            assert!(r < 1e-4, "{c:?}: {r:e}");
        }
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    let picks: Vec<bool> = (0..6)
        .map(|d| gpu_for_by(Some(d), false, test_cc))
        .collect();
    // Only the V100 (device 2) takes the op.
    assert_eq!(picks, [false, false, true, false, false, false]);
    let picks: Vec<bool> = (0..6).map(|d| gpu_for_by(Some(d), true, test_cc)).collect();
    assert_eq!(picks, [false, false, true, false, false, false]);
    // A state that names no device: the engine's answer.
    assert!(gpu_for_by(None, true, test_cc));
    assert!(!gpu_for_by(None, false, test_cc));
    assert!(gpu_takes(Some(7)) && !gpu_takes(Some(8)) && !gpu_takes(None));
}
