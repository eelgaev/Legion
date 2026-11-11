//! `gdn_mma` against ggml's `gated_delta_net` over q, k and v views of one conv-like row per
//! token, as a DeltaNet layer has them: the op's CPU side (the per-token reference) on the CPU,
//! the chunked kernel (ggml's, ported to src/cuda/gdn_mma.cu) on a GPU (ignored test).

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

/// Largest difference relative to the largest magnitude.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let top = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / top
}

/// A case: key heads `n_k`, value heads `n_v`, tokens `nt`, sequences `ns`.
type Case = (i64, i64, i64, i64);

/// (op, ggml) on `backend`: the tokens' outputs then the final states, and the milliseconds of
/// `reps` more runs of each (0: none).
fn case(
    backend: &mut Backend,
    (n_k, n_v, nt, ns): Case,
    reps: usize,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let s = HEAD;
    let row = 2 * s * n_k + s * n_v;
    let ctx = Context::new(32, 32);
    let conv = ctx.new_tensor(types::F32, &[row, nt, ns]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, n_v, nt, ns]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, n_v, nt, ns]).as_input();
    let state = ctx.new_tensor(types::F32, &[s, s, n_v, ns]).as_input();
    let es = 4usize;
    let nb = [s as usize * es, row as usize * es, (row * nt) as usize * es];
    let q = ctx.view_4d(conv, [s, n_k, nt, ns], nb, 0);
    let k = ctx.view_4d(conv, [s, n_k, nt, ns], nb, (s * n_k) as usize * es);
    let v = ctx.view_4d(conv, [s, n_v, nt, ns], nb, (2 * s * n_k) as usize * es);
    assert!(shapes_fit(q, k, v, g, beta, state), "{:?}", (n_k, n_v, nt));
    let op = gdn_mma(&ctx, q, k, v, g, beta, state).as_output();
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
        for head in tok[..(2 * s * n_k) as usize].chunks_mut(s as usize) {
            let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            head.iter_mut().for_each(|v| *v /= n);
        }
    }
    conv.set_f32(&x);
    let gb = (n_v * nt * ns) as usize;
    let map = |seed, f: fn(f32) -> f32| noise(gb, seed).into_iter().map(f).collect::<Vec<_>>();
    g.set_f32(&map(2, |v| 0.5 * (v - 1.0)));
    beta.set_f32(&map(3, |v| 0.5 * (v + 1.0)));
    state.set_f32(
        &noise((s * s * n_v * ns) as usize, 4)
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

#[test]
fn op_matches_ggml_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    // A rank's ratio of value to key heads (3), a chunk tail, two sequences.
    for c in [(2, 6, 64, 1), (1, 3, 70, 2)] {
        let (op, reference, _, _) = case(&mut backend, c, 0);
        let r = rel(&op, &reference);
        assert!(r < 1e-5, "{c:?}: {r:e}");
    }
}

#[test]
fn shapes_fit_prompt_groups_of_any_head_counts() {
    let ctx = Context::new(32, 32);
    let s = HEAD;
    let fit = |n_k: i64, n_v: i64, nt: i64| {
        let q = ctx.new_tensor(types::F32, &[s, n_k, nt, 1]);
        let v = ctx.new_tensor(types::F32, &[s, n_v, nt, 1]);
        let g = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]);
        let state = ctx.new_tensor(types::F32, &[s, s, n_v, 1]);
        shapes_fit(q, q, v, g, g, state)
    };
    // A tensor parallel rank of Qwen3.8-Flash-Next (8 of 16 key, 24 of 48 value heads).
    assert!(fit(8, 24, 64));
    assert!(fit(8, 24, 4096));
    // One GPU's layer (ggml-cuda's own chunked kernel's head counts; `takes` decides).
    assert!(fit(16, 48, 4096));
    assert!(fit(16, 32, 4096));
    // Too few tokens (decode, verify), value heads not a multiple of the key heads.
    assert!(!fit(8, 24, 63));
    assert!(!fit(16, 48, 63));
    assert!(!fit(16, 24, 4096));
}

/// The head counts the op takes: a rank's share with `--gdn-mma`, ggml-cuda's own groups with
/// the staged kernel on compute capability 12.0 and 12.1 only. Restores the switches.
#[test]
fn takes_ggmls_groups_staged_on_consumer_blackwell_only() {
    assert!(staged_gpu(Some((12, 1))));
    assert!(staged_gpu(Some((12, 0))));
    for cc in [
        None,
        Some((8, 6)),
        Some((8, 9)),
        Some((9, 0)),
        Some((10, 0)),
        Some((12, 2)),
    ] {
        assert!(!staged_gpu(cc), "{cc:?}");
    }
    let (on, staged, gpu) = (
        ON.load(Ordering::Relaxed),
        STAGED.load(Ordering::Relaxed),
        STAGED_GPU.load(Ordering::Relaxed),
    );
    assert!(staged, "on by default");
    for (o, st, g) in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        ON.store(o, Ordering::Relaxed);
        STAGED.store(st, Ordering::Relaxed);
        STAGED_GPU.store(g, Ordering::Relaxed);
        assert_eq!(takes(16, 48, None), st && g, "{:?}", (o, st, g));
        assert_eq!(takes(16, 64, None), st && g);
        assert_eq!(takes(8, 24, None), o);
    }
    ON.store(on, Ordering::Relaxed);
    STAGED.store(staged, Ordering::Relaxed);
    STAGED_GPU.store(gpu, Ordering::Relaxed);
}

/// `gdn_staged` copies 16-byte pieces: aligned pointers and strides of whole 4-float groups.
#[test]
fn staged_needs_aligned_rows() {
    let base = std::ptr::without_provenance::<f32>(0x1000);
    let args = |off: usize, sq1: i64| Args {
        q: base,
        k: base.wrapping_add(off),
        v: base.wrapping_add(2 * off),
        g: base,
        beta: base,
        state: base,
        dst: std::ptr::null_mut(),
        state_out: std::ptr::null_mut(),
        h: 48,
        h_k: 16,
        n_tokens: 4096,
        n_seqs: 1,
        rq3: 1,
        sq: [sq1, 10240, 10240 * 4096],
        sv: [128, 10240, 10240 * 4096],
        sb: [1, 48, 48 * 4096],
        scale: 1.0,
    };
    assert!(args(2048, 128).staged_aligned());
    assert!(!args(2049, 128).staged_aligned());
    assert!(!args(2048, 130).staged_aligned());
}

#[test]
fn shapes_fit_needs_value_heads_and_sequences() {
    let ctx = Context::new(32, 32);
    let s = HEAD;
    let fit = |n_k: i64, n_v: i64, ns: i64| {
        let q = ctx.new_tensor(types::F32, &[s, n_k, 64, ns.max(1)]);
        let v = ctx.new_tensor(types::F32, &[s, n_v, 64, ns]);
        let g = ctx.new_tensor(types::F32, &[1, n_v, 64, ns]);
        let state = ctx.new_tensor(types::F32, &[s, s, n_v, ns]);
        shapes_fit(q, q, v, g, g, state)
    };
    assert!(fit(8, 24, 2));
    assert!(!fit(8, 0, 1));
    assert!(!fit(8, 24, 0));
}

/// ggml's `select_gdn_mma_path` on NVIDIA GPUs: value slices when whole heads leave half the SMs
/// idle, one CTA an SM when the slices fit one wave.
#[test]
fn schedule_is_ggmls_rule() {
    // A two-Spark rank's 24 value heads on a GB10 (48 SMs): slices at two CTAs an SM.
    assert_eq!(Schedule::pick(24, 1, 48), Schedule::SlicesTwo);
    assert_eq!(Schedule::SlicesTwo.grid(24, 1), 96);
    // One Spark's 48 heads, two sequences of a rank: whole heads.
    assert_eq!(Schedule::pick(48, 1, 48), Schedule::Heads);
    assert_eq!(Schedule::pick(24, 2, 48), Schedule::Heads);
    assert_eq!(Schedule::Heads.grid(24, 2), 48);
    // 25 heads leave fewer than half of 48 SMs idle.
    assert_eq!(Schedule::pick(25, 1, 48), Schedule::Heads);
    // 12 heads: the slices (48 CTAs) fit one wave.
    assert_eq!(Schedule::pick(12, 1, 48), Schedule::Slices);
    assert_eq!(Schedule::Slices.grid(12, 1), 48);
    assert_eq!(Schedule::pick(13, 1, 48), Schedule::SlicesTwo);
}

/// The shared memory sizes `gdn_mma.cu` static_asserts for `shared<16, 32>` and
/// `shared<16, 128>`, and the argument layout it static_asserts.
#[test]
fn launch_sizes_match_the_kernels() {
    assert_eq!(Schedule::Slices.smem(), 38016);
    assert_eq!(Schedule::SlicesTwo.smem(), 38016);
    assert_eq!(Schedule::Heads.smem(), 93312);
    assert_eq!(Schedule::Staged.smem(), 85376);
    assert_eq!(Schedule::Staged.grid(48, 2), 96);
    assert_eq!(std::mem::size_of::<Args>(), 184);
    assert_eq!(std::mem::offset_of!(Args, h), 64);
    assert_eq!(std::mem::offset_of!(Args, sq), 104);
    assert_eq!(std::mem::offset_of!(Args, scale), 176);
}

#[test]
fn fits_no_host_group() {
    let ctx = Context::new(8, 8);
    let s = HEAD;
    let q = ctx.new_tensor(types::F32, &[s, 8, 64, 1]);
    let v = ctx.new_tensor(types::F32, &[s, 24, 64, 1]);
    let g = ctx.new_tensor(types::F32, &[1, 24, 64, 1]);
    let state = ctx.new_tensor(types::F32, &[s, s, 24, 1]);
    let backend = Backend::cpu().unwrap();
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    assert!(shapes_fit(q, q, v, g, g, state));
    // A state in host memory (the CPU graph) keeps ggml's op whatever the switches.
    assert!(!fits(q, q, v, g, g, state));
}

/// The ported chunked kernel at a rank's shapes agrees with ggml's per-token kernel to rounding,
/// with the time of each at a 2-Spark rank's shape and prompt steps of 2304 and 4608 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+"]
fn cuda_gdn_mma_matches_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for c in [
        (2, 6, 64, 1),
        (1, 3, 70, 2),
        (8, 24, 4608, 1),
        (8, 24, 2304, 1),
    ] {
        let reps = if c.0 == 8 { 10 } else { 0 };
        let (op, reference, op_ms, ref_ms) = case(&mut gpu, c, reps);
        if reps > 0 {
            println!("gdn_mma {c:?}: {op_ms:.3} ms, ggml gated_delta_net {ref_ms:.3} ms");
        }
        let r = rel(&op, &reference);
        assert!(r < 1e-3, "{c:?}: {r:e}");
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    // The GB10, the 4090, the A100 and the 5090 have the tensor cores; the V100 and an unknown
    // device do not, whatever the engine's GPU is.
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, true, false, true, true, false],
            "engine {engine}"
        );
        assert_eq!(gpu_for_by(None, engine, test_cc), engine);
    }
    assert!(gpu_takes(Some(8)) && gpu_takes(Some(12)) && !gpu_takes(Some(7)) && !gpu_takes(None));
}

/// The staged kernel takes ggml-cuda's groups on the GB10 and the 5090 of a mixed box only,
/// whatever the engine's GPU is; the engine's answer where the state names no device.
#[test]
fn staged_answers_per_device_of_a_mixed_box() {
    use crate::hw::per_device::test_cc;
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| staged_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, false, false, false, true, false],
            "engine {engine}"
        );
        assert_eq!(staged_for_by(None, engine, test_cc), engine);
    }
}
