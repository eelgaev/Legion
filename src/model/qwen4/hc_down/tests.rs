//! The rebuilt down matmul (#1183) against the BF16 `xn` the norm writes: with one-hot down
//! weights every output is one value of the GEMM's B operand, which must be `xnb`'s value; the
//! norm without `xnb` writes the same streams, scales and inject weights. On the CPU (the
//! reference) and on CUDA (ignored test, the kernel at qwen4exp's shape).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::nvfp4_mm::bf16_bits;
use crate::model::qwen4::hc_norm;

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

/// `x` rounded to BF16 (ties to even), as bytes.
fn bf16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| bf16_bits(v).to_ne_bytes()).collect()
}

/// One-hot down weights `j` of `k` columns: row `r` is 1 at column `(j · M + r) % k`, so the
/// `ceil(k / M)` of them read every column.
fn one_hot(k: i64, j: i64) -> Vec<u8> {
    let mut w = vec![0f32; (k * M) as usize];
    for r in 0..M {
        w[(r * k + (j * M + r) % k) as usize] = 1.0;
    }
    bf16_bytes(&w)
}

/// What [`case`] returns.
struct Case {
    /// The norm writing `xnb` and the one without: streams (widened), scales and inject weights
    /// as bytes.
    with: [Vec<u8>; 3],
    without: [Vec<u8>; 3],
    /// `xnb` widened to F32 (`[k, n]`).
    xnb: Vec<f32>,
    /// [`mul_mat`] of each one-hot weight (`[M, n]` each).
    hot: Vec<Vec<f32>>,
    /// [`mul_mat`] of the random weight and ggml's `mul_mat` of it and `xnb`.
    lo: Vec<f32>,
    lo_ref: Vec<f32>,
}

/// The norm with BF16 streams on `n` tokens of 4 streams of `e` values, writing `xnb` and not,
/// [`mul_mat`] of the second's streams with one-hot and random down weights, ggml's `mul_mat` of
/// the random weight and `xnb`.
fn case(backend: &mut Backend, e: i64, n: i64) -> Case {
    let (hc, n_inj) = (4i64, 4i64);
    let k = e * hc;
    let hot_n = (k + M - 1) / M;
    let ctx = Context::new(64 + 2 * hot_n as usize, 64 + 2 * hot_n as usize);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let w = ctx.new_tensor(types::F32, &[k, n_inj]).as_input();
    let down = ctx.new_tensor(types::BF16, &[k, M]).as_input();
    let hot: Vec<Tensor<'_>> = (0..hot_n)
        .map(|_| ctx.new_tensor(types::BF16, &[k, M]).as_input())
        .collect();
    let a = hc_norm::hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, g, w, 1e-6, down, true);
    let b = hc_norm::hc_post_norm_b16_streams(&ctx, res, out, inj, g, w, 1e-6, down);
    assert_eq!((b.res.ty(), b.res.ne()), (types::BF16, [e, hc, n, 1]));
    let f32_of = |t| ctx.cont(ctx.cast(t, types::F32)).as_output();
    let with = [a.res, a.scales, a.inject].map(f32_of);
    let without = [b.res, b.scales, b.inject].map(f32_of);
    let xnb = ctx.cont(ctx.cast(a.xnb, types::F32)).as_output();
    let hot_lo: Vec<Tensor<'_>> = hot
        .iter()
        .map(|&h| mul_mat(&ctx, h, b.res, b.scales, g).as_output())
        .collect();
    let lo = mul_mat(&ctx, down, b.res, b.scales, g).as_output();
    let lo_ref = ctx.mul_mat(down, a.xnb).as_output();
    let mut graph = ctx.new_graph(64 + 2 * hot_n as usize);
    for t in with
        .iter()
        .chain(&without)
        .chain(&hot_lo)
        .chain([&xnb, &lo, &lo_ref])
    {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(
        &noise((k * n) as usize, 71)
            .iter()
            .map(|v| 3.0 * v)
            .collect::<Vec<_>>(),
    );
    out.set_f32(&noise((e * n) as usize, 72));
    inj.set_f32(&noise((hc * n) as usize, 73));
    g.set_f32(
        &noise(k as usize, 74)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect::<Vec<_>>(),
    );
    w.set_f32(&noise((k * n_inj) as usize, 75));
    down.set_bytes(&bf16_bytes(
        &noise((k * M) as usize, 76)
            .iter()
            .map(|v| 0.02 * v)
            .collect::<Vec<_>>(),
    ));
    for (j, h) in hot.iter().enumerate() {
        h.set_bytes(&one_hot(k, j as i64));
    }
    backend.compute(&graph).unwrap();
    Case {
        with: with.map(|t| t.bytes()),
        without: without.map(|t| t.bytes()),
        xnb: xnb.to_f32(),
        hot: hot_lo.iter().map(|t| t.to_f32()).collect(),
        lo: lo.to_f32(),
        lo_ref: lo_ref.to_f32(),
    }
}

/// The checks of [`case`]: the norm's other outputs do not move without `xnb`; every value the
/// rebuilt GEMM multiplies is `xnb`'s (equal values; a one-hot sum turns -0 into +0); the GEMM
/// is ggml's `mul_mat` of `xnb` within F32 rounding of the sum order.
fn check(backend: &mut Backend, e: i64, ns: &[i64]) {
    for &n in ns {
        let c = case(backend, e, n);
        for (i, name) in ["streams", "scales", "inject"].iter().enumerate() {
            assert!(c.with[i] == c.without[i], "{n} tokens: {name} moved");
        }
        let k = e * 4;
        let mut seen = vec![false; k as usize];
        for (j, lo) in c.hot.iter().enumerate() {
            for t in 0..n {
                for r in 0..M {
                    let col = (j as i64 * M + r) % k;
                    let got = lo[(t * M + r) as usize];
                    let want = c.xnb[(t * k + col) as usize];
                    assert!(
                        got == want,
                        "{n} tokens: token {t} k {col}: {got} vs xnb {want}"
                    );
                    seen[col as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "{n} tokens: a k value not read");
        let (num, den) =
            c.lo.iter()
                .zip(&c.lo_ref)
                .fold((0f64, 0f64), |(a, b), (&x, &y)| {
                    let d = f64::from(x) - f64::from(y);
                    (a + d * d, b + f64::from(y) * f64::from(y))
                });
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        assert!(rel < 1e-5, "{n} tokens: off ggml's mul_mat by {rel}");
        assert!(den > 0.0, "{n} tokens: zero output");
    }
}

/// [`check`] on the CPU (the reference).
#[test]
fn rebuilt_down_reads_xnb_on_the_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check(&mut cpu, 256, &[1, 9]);
}

/// [`rebuild`] is the norm's arithmetic: two rounded F32 multiplies, then BF16 nearest even.
#[test]
fn rebuild_rounds_as_the_norm() {
    for (i, v) in noise(4096, 81).iter().enumerate() {
        let b = bf16_bits(4.0 * v);
        let s = 0.5 + v.abs();
        let g = 1.0 - 0.25 * v;
        let x = f32::from_bits(u32::from(b) << 16);
        assert_eq!(rebuild(b, s, g), bf16_bits((x * s) * g), "value {i}");
    }
    // A tie: 1 + 2^-8 sits halfway between two BF16 values; nearest even rounds it down.
    assert_eq!(rebuild(0x3f80, 1.0 + 1.0 / 256.0, 1.0), 0x3f80);
}

/// The token tile: fewest waves times a CTA's cost, the smaller tile on ties.
#[test]
fn token_tile_by_step_size() {
    // 4,096 tokens on 48 SMs: 64, 43 or 32 CTAs: one wave of 96-token tiles.
    assert_eq!(pick_nt(4096, 48), 3);
    assert_eq!(pick_nt(256, 48), 2);
    assert_eq!(pick_nt(6144, 48), 4);
    assert_eq!(pick_nt(8192, 48), 3);
    assert_eq!(pick_nt(1, 0), 2);
}

/// The shapes the kernel takes: qwen4exp's BF16 10240 x 320 weight and 4 BF16 streams.
#[test]
fn shapes_the_kernel_takes() {
    let nb = |k: usize| [2, 2 * k, 2 * k * 320, 2 * k * 320];
    assert!(shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(shape_fits(
        types::BF16,
        [1024, 320, 1, 1],
        nb(1024),
        [256, 4, 1, 1]
    ));
    assert!(!shape_fits(
        types::F16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 256, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 0, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [8000, 320, 1, 1],
        nb(8000),
        [2000, 4, 9, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [5120, 2, 9, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        [2, 20488, 0, 0],
        [2560, 4, 9, 1]
    ));
}

/// [`check`] on CUDA at qwen4exp's shape (sm_80+): the kernel's B operand is `xnb` bit for bit
/// (whole tiles, a partial last tile, every token tile).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rebuilt_down_reads_xnb() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() || !supported_on(0) {
        println!("no sm_80+ kernel here: skipped");
        return;
    }
    for nt in [0, 2, 3, 4] {
        force_nt(nt);
        check(&mut gpu, 2560, &[1, 33, 300, 4096]);
    }
    force_nt(0);
}

/// The mixer's tail of a `--hc-bf16` prompt step as qwen4exp builds it: [`mul_mat`] of the
/// BF16 streams, the gate's activation in place on it (`hc_lo_act`), the BF16 up-mix reading it
/// as BF16 (`hc_up_lob`, then the mix): its graph on `n` tokens of 4 streams of `e` values,
/// rewritten by the node fusion when `fuse` (`cuda`: as on a CUDA backend). `keep_lo`: the F32
/// `lo` is an output too, so another reader of it.
struct Tail<'a> {
    graph: crate::ggml::Graph<'a>,
    /// The down weight, the up weight, the streams, their norm scales and gamma.
    inputs: [Tensor<'a>; 5],
    mixed: Tensor<'a>,
    /// The `hc_up_lob` node, whose `lo` part the mix reads.
    lob: *mut ffi::ggml_tensor,
    counts: crate::model::core::node_fuse::Counts,
}

/// The activation's and the mix's scale (`1 / hc`, as qwen4exp's mixers).
const TAIL_SCALE: f32 = 0.25;

fn tail<'a>(
    ctx: &'a Context,
    backend: &Backend,
    e: i64,
    n: i64,
    fuse: bool,
    cuda: bool,
    keep_lo: bool,
) -> Tail<'a> {
    let hc = 4i64;
    let k = e * hc;
    // Inputs kept for the whole graph (outputs too), as weights and the norm's node are while
    // the mix runs.
    let kept = |ty, ne: &[i64]| ctx.new_tensor(ty, ne).as_input().as_output();
    let down = kept(types::BF16, &[k, M]);
    let up = kept(types::BF16, &[M, k]);
    let res = kept(types::BF16, &[e, hc, n]);
    let sc = kept(types::F32, &[hc, n]);
    let g = kept(types::F32, &[k]);
    let lo = mul_mat(ctx, down, res, sc, g);
    let lo = if keep_lo { lo.as_output() } else { lo };
    let act = super::super::hc_chain::lo_act_op(ctx, lo, TAIL_SCALE);
    let mixed = super::super::hc_up_mix::up_mix_rebuilt(ctx, up, act, res, sc, g, hc, TAIL_SCALE)
        .as_output();
    // SAFETY: a live mix node of the BF16 form: src[1] is its `hc_up_lob` node.
    let lob = unsafe { (*mixed.raw()).src[1] };
    let mut graph = ctx.new_graph(64);
    graph.expand(mixed);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let counts = if fuse {
        // SAFETY: an allocated graph that has not run.
        unsafe { crate::model::core::node_fuse::rewrite_graph(graph.raw(), cuda) }
    } else {
        Default::default()
    };
    Tail {
        graph,
        inputs: [down, up, res, sc, g],
        mixed,
        lob,
        counts,
    }
}

/// Fills [`tail`]'s inputs with the same values whatever the graph.
fn fill_tail(t: &Tail<'_>, e: i64, n: i64) {
    let k = e * 4;
    let [down, up, res, sc, g] = t.inputs;
    let scaled = |len: i64, seed: u64, s: f32| -> Vec<f32> {
        noise(len as usize, seed).iter().map(|v| s * v).collect()
    };
    down.set_bytes(&bf16_bytes(&scaled(k * M, 91, 0.02)));
    up.set_bytes(&bf16_bytes(&scaled(M * k, 92, 0.05)));
    res.set_bytes(&bf16_bytes(&scaled(k * n, 93, 3.0)));
    sc.set_f32(
        &noise((4 * n) as usize, 94)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<_>>(),
    );
    g.set_f32(
        &noise(k as usize, 95)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect::<Vec<_>>(),
    );
}

/// The `lo` bytes the mix of `t` read (its `hc_up_lob` node's BF16 part).
fn tail_lob(t: &Tail<'_>, n: i64) -> Vec<u8> {
    let len = (M * n * 2) as usize;
    let mut out = vec![0u8; len];
    // SAFETY: an allocated node; the part lies within it (`lob_offset`).
    unsafe {
        let at =
            super::super::hc_up_mix::lob_offset(ffi::ggml_nbytes(t.lob), M as usize, n as usize);
        ffi::ggml_backend_tensor_get(t.lob, out.as_mut_ptr().cast(), at, len);
    }
    out
}

/// [`tail`] on `backend` as built and fused: the fusion takes the chain, and what the mix reads
/// (`lo` as BF16) and writes are the same bytes.
fn check_tail(backend: &mut Backend, e: i64, ns: &[i64], cuda: bool) {
    for &n in ns {
        let ctx_a = Context::new(64, 64);
        let ctx_b = Context::new(64, 64);
        let built = tail(&ctx_a, backend, e, n, false, cuda, false);
        let fused = tail(&ctx_b, backend, e, n, true, cuda, false);
        assert_eq!(fused.counts.down_lob, 1, "{n} tokens: the chain fused");
        // SAFETY: live nodes of the fused graph.
        unsafe {
            assert_eq!(
                (*fused.lob).op,
                ffi::ggml_op_GGML_OP_NONE,
                "{n} tokens: lob off"
            );
            assert_eq!(
                (*(*fused.lob).src[0]).op,
                ffi::ggml_op_GGML_OP_NONE,
                "{n} tokens: activation off"
            );
        }
        let mut out = Vec::new();
        for t in [&built, &fused] {
            fill_tail(t, e, n);
            backend.compute(&t.graph).unwrap();
            out.push((tail_lob(t, n), t.mixed.bytes()));
        }
        assert!(out[0].0.iter().any(|&b| b != 0), "{n} tokens: no lo");
        assert!(
            out[0].1.iter().any(|&b| b != 0),
            "{n} tokens: nothing mixed"
        );
        let lo_diff = out[0]
            .0
            .chunks(2)
            .zip(out[1].0.chunks(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(lo_diff, 0, "{n} tokens: {lo_diff} BF16 lo values moved");
        assert!(out[0].1 == out[1].1, "{n} tokens: the mix moved");
    }
}

/// [`check_tail`] on the CPU: the down op's fused epilogue is the activation's and the BF16
/// rounding's CPU code.
#[test]
fn down_lob_is_the_chain_on_the_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_tail(&mut cpu, 256, &[1, 9], false);
}

/// Another reader of the F32 `lo` (here: an output) keeps the chain as built.
#[test]
fn down_lob_needs_lo_read_once() {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(64, 64);
    let t = tail(&ctx, &cpu, 256, 9, true, false, true);
    assert_eq!(t.counts.down_lob, 0);
    // SAFETY: a live node of the graph.
    assert_ne!(unsafe { (*t.lob).op }, ffi::ggml_op_GGML_OP_NONE);
}

/// The GPU, sm_80+ with the kernels.
#[cfg(feature = "cuda")]
fn tail_gpu() -> Option<Backend> {
    use crate::ggml::{DeviceKind, device_by_kind};
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() || !supported_on(0) {
        println!("no sm_80+ kernel here: skipped");
        return None;
    }
    Some(gpu)
}

/// [`check_tail`] on CUDA at qwen4exp's shape (#1183): the down kernel's epilogue writes the
/// BF16 `lo` the `hc_lo_act` and `hc_up_lob` kernels write, byte for byte, and the mix reading
/// it is unmoved, at a 12,288-token step (`--max-batch-tokens` of the docs command), a prompt's
/// 8,067-token tail and 129 tokens (a partial last tile), every token tile.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cuda_down_lob_is_the_chain() {
    let Some(mut gpu) = tail_gpu() else {
        return;
    };
    for nt in [0, 2, 3, 4] {
        force_nt(nt);
        check_tail(&mut gpu, 2560, &[12288, 8067, 129], true);
    }
    force_nt(0);
}

/// The fused and the built tail timed in alternating order (graph-level: the down matmul, the
/// activation, the BF16 rounding and the up-mix) at 12,288 and 8,067 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU; a timing run"]
fn cuda_down_lob_timing() {
    let Some(mut gpu) = tail_gpu() else {
        return;
    };
    const ROUNDS: usize = 20;
    for n in [12288i64, 8067] {
        let ctx_a = Context::new(64, 64);
        let ctx_b = Context::new(64, 64);
        let tails = [
            tail(&ctx_a, &gpu, 2560, n, false, true, false),
            tail(&ctx_b, &gpu, 2560, n, true, true, false),
        ];
        for t in &tails {
            fill_tail(t, 2560, n);
            // Warm-up.
            gpu.compute(&t.graph).unwrap();
        }
        let mut secs = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for i in order {
                let t0 = std::time::Instant::now();
                gpu.compute(&tails[i].graph).unwrap();
                secs[i].push(t0.elapsed().as_secs_f64());
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2] * 1e3
        };
        let (a, b) = (median(&mut secs[0]), median(&mut secs[1]));
        println!(
            "{n} tokens: built {a:.3} ms, fused {b:.3} ms ({:.3}x), medians of {ROUNDS}, alternating",
            b / a
        );
    }
}

/// The mix overlap's split (#1183): the down matmul's full waves where a partial one follows.
#[test]
fn mix_overlap_splits_at_the_last_full_wave() {
    // 12,288 tokens on 48 SMs: 96 CTAs of 128 tokens, two full waves: no split.
    assert_eq!(split_at(12288, 48), 0);
    // 8,067: 85 CTAs of 96 tokens, one full wave of 48 (4,608 tokens), 37 after.
    assert_eq!(split_at(8067, 48), 4608);
    assert_eq!(split_at(8192, 48), 4608);
    // 14,000: 110 CTAs of 128 tokens, two full waves, 14 after.
    assert_eq!(split_at(14000, 48), 12288);
    // One wave or less: nothing to overlap.
    assert_eq!(split_at(2049, 48), 0);
    assert_eq!(split_at(4608, 48), 0);
    assert_eq!(split_at(1, 0), 0);
}

/// The CPU graph keeps the chain whole (the overlap is a CUDA launch order).
#[test]
fn mix_overlap_is_cuda_only() {
    let cpu = Backend::cpu().unwrap();
    let ctx = Context::new(64, 64);
    let t = tail(&ctx, &cpu, 256, 4700, true, false, false);
    assert_eq!(t.counts.down_lob, 1);
    assert_eq!(t.counts.mix_overlap, 0);
    // SAFETY: a live mix node of the graph: src[1] its `lob`, whose source the activation, whose
    // source the down node.
    let down = unsafe { &*(*(*t.lob).src[0]).src[0] };
    assert_eq!(split_of(down), 0);
}

/// Bytes of the `lo` part of [`tail`]'s `lob` and its mixed values, both poisoned (all ones, NaN
/// in BF16 and F32) before the run: a token either part misses shows.
#[cfg(feature = "cuda")]
fn overlap_run(gpu: &mut Backend, t: &Tail<'_>, e: i64, n: i64) -> (Vec<u8>, Vec<u8>) {
    fill_tail(t, e, n);
    t.mixed.set_bytes(&vec![0xff; t.mixed.bytes().len()]);
    let len = (M * n * 2) as usize;
    // SAFETY: an allocated node; the part lies within it (`lob_offset`).
    unsafe {
        let at =
            super::super::hc_up_mix::lob_offset(ffi::ggml_nbytes(t.lob), M as usize, n as usize);
        ffi::ggml_backend_tensor_set(t.lob, vec![0xffu8; len].as_ptr().cast(), at, len);
    }
    gpu.compute(&t.graph).unwrap();
    (tail_lob(t, n), t.mixed.bytes())
}

/// The mix overlap on CUDA at qwen4exp's shape (#1183): the split chain (by [`split_at`], and
/// half the tokens) writes the unsplit chain's `lo` and mixed values byte for byte, outputs
/// poisoned first, at a 12,288-token step, a prompt's 8,067-token tail and 2,049 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU"]
fn cuda_mix_overlap_is_the_same_bytes() {
    let Some(mut gpu) = tail_gpu() else {
        return;
    };
    let sms = i64::from(crate::hw::per_device::sm_count(0));
    const E: i64 = 2560;
    for n in [12288i64, 8067, 2049] {
        force_split(-1);
        let ctx = Context::new(64, 64);
        let whole = tail(&ctx, &gpu, E, n, true, true, false);
        assert_eq!(whole.counts.mix_overlap, 0, "{n} tokens: split, forced off");
        let want = overlap_run(&mut gpu, &whole, E, n);
        assert!(
            want.0.chunks(2).all(|v| v != [0xff, 0xff]),
            "{n} tokens: lo not all written"
        );
        assert!(
            want.1.chunks(4).all(|v| v != [0xff; 4]),
            "{n} tokens: mix not all written"
        );
        for (mode, split) in [(0i64, split_at(n, sms)), (-2, n / 2)] {
            force_split(mode);
            let ctx = Context::new(64, 64);
            let t = tail(&ctx, &gpu, E, n, true, true, false);
            assert_eq!(
                t.counts.mix_overlap,
                usize::from(split > 0),
                "{n} tokens, mode {mode}: split"
            );
            let got = overlap_run(&mut gpu, &t, E, n);
            let lo_diff = (want.0.chunks(2).zip(got.0.chunks(2))).filter(|(a, b)| a != b);
            let mix_diff = (want.1.chunks(4).zip(got.1.chunks(4))).filter(|(a, b)| a != b);
            let (lo_diff, mix_diff) = (lo_diff.count(), mix_diff.count());
            println!("{n} tokens, split at {split}: {lo_diff} lo, {mix_diff} mixed values moved");
            assert_eq!((lo_diff, mix_diff), (0, 0), "{n} tokens, split at {split}");
        }
    }
    force_split(0);
}

/// The unsplit and split chains timed in alternating order (graph-level: the down matmul and
/// the up-mix, the activation and rounding in the matmul's epilogue) at 12,288 and 8,067
/// tokens: never split, split by [`split_at`] and split in halves.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_80+ GPU; a timing run"]
fn cuda_mix_overlap_timing() {
    let Some(mut gpu) = tail_gpu() else {
        return;
    };
    const ROUNDS: usize = 20;
    const E: i64 = 2560;
    let sms = i64::from(crate::hw::per_device::sm_count(0));
    for n in [12288i64, 8067] {
        let modes = [(-1i64, "unsplit"), (0, "split_at"), (-2, "halves")];
        let ctxs = [0; 3].map(|_| Context::new(64, 64));
        let tails: Vec<Tail<'_>> = modes
            .iter()
            .zip(&ctxs)
            .map(|(&(mode, _), ctx)| {
                force_split(mode);
                tail(ctx, &gpu, E, n, true, true, false)
            })
            .collect();
        force_split(0);
        for t in &tails {
            fill_tail(t, E, n);
            // Warm-up.
            gpu.compute(&t.graph).unwrap();
        }
        let mut secs = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            for j in 0..3 {
                let i = (round + j) % 3;
                let t0 = std::time::Instant::now();
                gpu.compute(&tails[i].graph).unwrap();
                secs[i].push(t0.elapsed().as_secs_f64());
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2] * 1e3
        };
        let ms: Vec<f64> = secs.iter_mut().map(median).collect();
        println!(
            "{n} tokens ({} SMs, split_at {}): {} {:.3} ms, {} {:.3} ms ({:.3}x), {} {:.3} ms \
             ({:.3}x), medians of {ROUNDS}, rotating",
            sms,
            split_at(n, sms),
            modes[0].1,
            ms[0],
            modes[1].1,
            ms[1],
            ms[1] / ms[0],
            modes[2].1,
            ms[2],
            ms[2] / ms[0]
        );
    }
}
