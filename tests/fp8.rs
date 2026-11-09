//! The FP8 W8A16 matmul op (`legion::model::fp8`): the CPU reference against an f64 product of
//! the exact E4M3 weights, and (`--features cuda`, ignored: needs a GPU) the CUDA GEMV and GEMM
//! against the CPU, plus a bandwidth/throughput table:
//! `cargo test --release --features cuda --test fp8 -- --ignored --nocapture`.

use legion::ggml::{Backend, ComputeAllocator, Context, types};
use legion::model::fp8::{self, SmallBatch, e4m3_to_f32, f32_to_e4m3};

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// E4M3 weight bytes `[m][k]` spread over the whole range (no NaNs).
fn weights(k: usize, m: usize, seed: u32) -> Vec<u8> {
    values(k * m, seed)
        .into_iter()
        .map(|v| f32_to_e4m3(v * v * v * 300.0))
        .collect()
}

/// The weight's per-tensor scale in `run` and `run_a8` (folded into the ops' epilogues).
const WS: f32 = 0.375;

/// out[t][j] = WS · sum_k w[j][k] x[t][k], in f64.
fn reference(w: &[u8], x: &[f32], k: usize, m: usize, n: usize) -> Vec<f64> {
    scaled_reference(w, x, k, m, n, &[WS])
}

/// out[t][j] = ws[j] (or ws[0] for all rows) · sum_k w[j][k] x[t][k], in f64.
fn scaled_reference(w: &[u8], x: &[f32], k: usize, m: usize, n: usize, ws: &[f32]) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for t in 0..n {
        for j in 0..m {
            let s = if ws.len() == 1 { ws[0] } else { ws[j] };
            out[t * m + j] = s as f64
                * (0..k)
                    .map(|p| e4m3_to_f32(w[j * k + p]) as f64 * x[t * k + p] as f64)
                    .sum::<f64>();
        }
    }
    out
}

/// One scale per output row (compressed-tensors' channel-wise FP8): powers of two times small
/// odd factors, far apart, so a row that takes another row's scale shows.
fn row_scales(m: usize) -> Vec<f32> {
    (0..m)
        .map(|j| 2f32.powi(j as i32 % 7 - 3) * (1.0 + (j % 5) as f32 / 4.0))
        .collect()
}

/// Runs the op on `backend`; `x` is `[n][k]`, given as `[k, n/2, 2]` when `n` is even (a
/// batched view of tokens).
fn run(backend: &mut Backend, w: &[u8], x: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    run_scaled(backend, w, x, k, m, n, &[WS])
}

/// [`run`] with the weight scale `scales` (one value, or one per row).
fn run_scaled(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    k: usize,
    m: usize,
    n: usize,
    scales: &[f32],
) -> Vec<f32> {
    run_with(backend, w, x, k, m, n, scales, false)
}

/// [`run_scaled`] through [`fp8::mul_mat_with`] (`small_batch_gemv`: 5-9 tokens on the GEMV).
#[allow(clippy::too_many_arguments)]
fn run_with(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    k: usize,
    m: usize,
    n: usize,
    scales: &[f32],
    small_batch_gemv: bool,
) -> Vec<f32> {
    let small = SmallBatch {
        gemv: small_batch_gemv,
        ..SmallBatch::default()
    };
    run_small(backend, w, x, k, m, n, scales, small)
}

/// [`run_scaled`] through [`fp8::mul_mat_small`] with the small-batch kernels `small`.
#[allow(clippy::too_many_arguments)]
fn run_small(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    k: usize,
    m: usize,
    n: usize,
    scales: &[f32],
    small: SmallBatch,
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let wt = weights.new_tensor(types::I8, &[k as i64, m as i64]);
    let ws = weights.new_tensor(types::F32, &[scales.len() as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    wt.set_bytes(w);
    ws.set_f32(scales);
    let ctx = Context::new(8, 16);
    let shape: Vec<i64> = if n.is_multiple_of(2) {
        vec![k as i64, n as i64 / 2, 2]
    } else {
        vec![k as i64, n as i64]
    };
    let xt = ctx.new_tensor(types::F32, &shape).as_input();
    let out = fp8::mul_mat_small(&ctx, wt, xt, Some(ws), small).as_output();
    assert_eq!(out.ne()[0], m as i64);
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(x);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

/// Max |got - want| relative to the largest |want|.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - w).abs())
        .fold(0f64, f64::max)
        / scale
}

#[test]
fn cpu_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (k, m, n) in [(64, 48, 1), (256, 33, 4), (96, 7, 9)] {
        let w = weights(k, m, 1);
        let x = values(k * n, 2);
        let got = run(&mut cpu, &w, &x, k, m, n);
        let e = rel_err(&got, &reference(&w, &x, k, m, n));
        assert!(e < 1e-6, "k {k}, m {m}, n {n}: relative error {e}");
    }
}

/// A scale per output row: each row times its own.
#[test]
fn cpu_row_scales_match_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (k, m, n) in [(64, 48, 1), (256, 33, 4), (96, 7, 9)] {
        let (w, x, ws) = (weights(k, m, 1), values(k * n, 2), row_scales(m));
        let got = run_scaled(&mut cpu, &w, &x, k, m, n, &ws);
        let e = rel_err(&got, &scaled_reference(&w, &x, k, m, n, &ws));
        assert!(e < 1e-6, "k {k}, m {m}, n {n}: relative error {e}");
    }
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use legion::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap()
}

/// GEMV (n <= 4: F32 math, so only summation order differs) and GEMM (n > 4: BF16 activations,
/// relative error up to ~2^-9 per product) against the exact product, odd sizes included.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_exact_product() {
    let mut gpu = gpu();
    for (k, m, n, tol) in [
        (2688, 300, 1, 1e-5),
        (4096, 129, 3, 1e-5),
        (2688, 64, 8, 4e-3),
        (4096, 2688, 16, 4e-3),
        (2688, 100, 32, 4e-3),
        (2688, 200, 9, 4e-3),
        (4096, 257, 100, 4e-3),
        (512, 64, 512, 4e-3),
    ] {
        let w = weights(k, m, 3);
        let x = values(k * n, 4);
        let got = run(&mut gpu, &w, &x, k, m, n);
        let e = rel_err(&got, &reference(&w, &x, k, m, n));
        println!("k {k} m {m} n {n}: relative error {e:.2e}");
        assert!(e < tol, "k {k}, m {m}, n {n}: relative error {e}");
        // A scale per row: every kernel's epilogue applies row j's (n > 32 with k % 64 == 0
        // takes the BF16-activation GEMM).
        let ws = row_scales(m);
        let got = run_scaled(&mut gpu, &w, &x, k, m, n, &ws);
        let e = rel_err(&got, &scaled_reference(&w, &x, k, m, n, &ws));
        println!("k {k} m {m} n {n}, row scales: relative error {e:.2e}");
        assert!(
            e < tol,
            "k {k}, m {m}, n {n}, row scales: relative error {e}"
        );
        // 5-9 tokens on the GEMV (`small_batch_gemv`): F32 math as for one token.
        if (5..=8).contains(&n) {
            let got = run_with(&mut gpu, &w, &x, k, m, n, &ws, true);
            let e = rel_err(&got, &scaled_reference(&w, &x, k, m, n, &ws));
            println!("k {k} m {m} n {n}, small-batch GEMV: relative error {e:.2e}");
            assert!(
                e < 1e-5,
                "k {k}, m {m}, n {n}, small-batch GEMV: relative error {e}"
            );
        }
    }
}

/// The large-batch GEMM (BF16 activations, n > 32) gives every token the bits the small GEMM
/// (n <= 32, F32 activations rounded to BF16 in the kernel) gives it: both sum the same BF16
/// MMAs over k in the same order, so a change to either one's pipelining or weight widening shows
/// here. Partial row and token tiles, a scale per row, and (the last case) weights cycling
/// through every E4M3 code but NaN, subnormals and both zeros included.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_large_gemm_matches_small_gemm_bits() {
    let mut gpu = gpu();
    let every_code = |k: usize, m: usize| -> Vec<u8> {
        (0..k * m)
            .map(|i| (i * 37 % 256) as u8)
            .map(|b| if b & 0x7f == 0x7f { 0x80 } else { b })
            .collect()
    };
    for (k, m, n, all) in [
        (2688, 300, 224, false),
        (5120, 136, 160, false),
        (1024, 200, 96, true),
    ] {
        let w = if all {
            every_code(k, m)
        } else {
            weights(k, m, 7)
        };
        let (x, ws) = (values(k * n, 8), row_scales(m));
        let big = run_scaled(&mut gpu, &w, &x, k, m, n, &ws);
        for t0 in (0..n).step_by(32) {
            let small = run_scaled(&mut gpu, &w, &x[t0 * k..(t0 + 32) * k], k, m, 32, &ws);
            let same = big[t0 * m..(t0 + 32) * m]
                .iter()
                .zip(&small)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "k {k}, m {m}: tokens {t0}..{} differ", t0 + 32);
        }
    }
}

/// Times the op at Nemotron 3.5's Mamba-2 projection shapes, streaming 16 different weights per
/// graph (440 MB, past every cache, as a decode step streams its layers): GB/s of weights
/// (decode) and TFLOPS (prefill).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_timings() {
    const L: usize = 16;
    let mut gpu = gpu();
    for (k, m) in [(2688, 10304), (4096, 2688), (5120, 10240)] {
        let w = weights(k, m, 5);
        let weights = Context::new(L, 0);
        let ws: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for wt in &ws {
            wt.set_bytes(&w);
        }
        for n in [1usize, 2, 4, 8, 16, 32, 128, 512, 1024, 4096] {
            let ctx = Context::new(4 * L, 4 * L);
            let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let mut graph = ctx.new_graph(4 * L);
            for wt in &ws {
                graph.expand(fp8::mul_mat(&ctx, *wt, xt, None).as_output());
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            xt.set_f32(&values(k * n, 6));
            gpu.compute(&graph).unwrap();
            let reps = 10;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                gpu.compute(&graph).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / (reps * L) as f64;
            println!(
                "k {k} m {m} n {n:>3}: {:8.1} us  {:6.1} GB/s weights  {:6.2} TFLOPS",
                s * 1e6,
                (k * m) as f64 / s / 1e9,
                2.0 * (k * m * n) as f64 / s / 1e12
            );
        }
    }
}

/// The GEMV gives every token of a small batch the bits that token gets alone (one token: a
/// plain decode step's GEMV), with a scale per row: 2-4 tokens by default, 2-9 with
/// `small_batch_gemv` (a verify step's batch). Row counts that take 2 and (up to 48 SMs) 4 rows
/// per warp with a partial last warp, and a k that is not a multiple of 512 (lanes past the end
/// of the row).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_small_batches_match_single_token_bits() {
    let mut gpu = gpu();
    for (k, m) in [(5120, 301), (1056, 2047), (2688, 6151)] {
        let (w, ws) = (weights(k, m, 9), row_scales(m));
        let x = values(k * 9, 10);
        let alone: Vec<Vec<f32>> = (0..9)
            .map(|t| run_scaled(&mut gpu, &w, &x[t * k..(t + 1) * k], k, m, 1, &ws))
            .collect();
        for (small_batch_gemv, max_n) in [(false, 4), (true, 9)] {
            for n in 2..=max_n {
                let got = run_with(&mut gpu, &w, &x[..n * k], k, m, n, &ws, small_batch_gemv);
                for (t, one) in alone.iter().enumerate().take(n) {
                    let same = got[t * m..(t + 1) * m]
                        .iter()
                        .zip(one)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(
                        same,
                        "k {k}, m {m}, n {n}, small_batch_gemv {small_batch_gemv}: token {t} \
                         differs from its decode bits"
                    );
                }
            }
        }
    }
}

/// Times the op at Qwen3.8-27B's channel-wise FP8 shapes (compressed-tensors: attention, linear
/// attention, the last MLPs and the 248K-row lm_head) for decode and verify batches, with a
/// scale per row and enough weight copies per graph to stream past the L2.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_verify_timings() {
    let mut gpu = gpu();
    for (k, m) in [
        (5120, 1024),
        (5120, 6144),
        (5120, 10240),
        (5120, 12288),
        (5120, 17408),
        (17408, 5120),
        (6144, 5120),
        (5120, 65536),
        (5120, 248320),
    ] {
        let copies = (256usize << 20).div_ceil(k * m).clamp(1, 32);
        // One pseudo-random block repeated: the big heads would take long to generate.
        let block = weights(k, m.min(1024), 5);
        let w: Vec<u8> = block.iter().copied().cycle().take(k * m).collect();
        let weights = Context::new(copies + 1, 0);
        let wts: Vec<_> = (0..copies)
            .map(|_| weights.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        let st = weights.new_tensor(types::F32, &[m as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for wt in &wts {
            wt.set_bytes(&w);
        }
        st.set_f32(&row_scales(m));
        let runs = [1usize, 2, 3, 4, 5, 6, 7, 8, 9]
            .into_iter()
            .flat_map(|n| [(n, false), (n, true)])
            .filter(|&(n, small)| !small || n > 4);
        for (n, small_batch_gemv) in runs {
            let ctx = Context::new(4 * copies, 4 * copies);
            let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let mut graph = ctx.new_graph(4 * copies);
            for wt in &wts {
                let y = fp8::mul_mat_with(&ctx, *wt, xt, Some(st), small_batch_gemv);
                graph.expand(y.as_output());
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            xt.set_f32(&values(k * n, 6));
            gpu.compute(&graph).unwrap();
            let reps = 20;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                gpu.compute(&graph).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / (reps * copies) as f64;
            println!(
                "verify k {k:>5} m {m:>6} n {n:>2}{}: {:8.1} us  {:6.1} GB/s weights",
                if small_batch_gemv { " gemv" } else { "     " },
                s * 1e6,
                (k * m) as f64 / s / 1e9
            );
        }
    }
}

/// The W8A8 matmul's relative error to the CPU's F32 sums at `n` tokens: summation order (1e-5),
/// except for the E4M3 tensor-core GEMMs (more than 8 tokens; up to 8 the F32 GEMV) on Ada
/// (compute capability 8.9). Its E4M3 MMAs add C and their 32 products in a window
/// of ~14 bits under the largest exponent among them and truncate the rest (an RTX 4090: 1 +
/// 2^-14 comes back as 1), so chained over k the sums drift from F32 ones as k grows: 2.2e-3
/// at k 2,688, ~4e-3 at 5,120, an order of magnitude under what E4M3 activations themselves
/// cost (W8A8 against W8A16 on the CPU: up to 0.1).
#[cfg(feature = "cuda")]
fn a8_tolerance(n: usize) -> f64 {
    if n > 8 && legion::hw::per_device::compute_capability(0) == Some(89) {
        1e-2
    } else {
        1e-5
    }
}

/// W8A8 on `backend`: quantize `x` with `scale`, then the E4M3 x E4M3 matmul.
fn run_a8(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    scale: f32,
    k: usize,
    m: usize,
    n: usize,
) -> (Vec<u8>, Vec<f32>) {
    let weights = Context::new(3, 0);
    let wt = weights.new_tensor(types::I8, &[k as i64, m as i64]);
    let st = weights.new_tensor(types::F32, &[1]);
    let ws = weights.new_tensor(types::F32, &[1]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    wt.set_bytes(w);
    st.set_f32(&[scale]);
    ws.set_f32(&[WS]);
    let ctx = Context::new(8, 16);
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let q = fp8::quantize(&ctx, xt, st).as_output();
    let out = fp8::mul_mat_a8(&ctx, wt, q, st, Some(ws)).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(x);
    backend.compute(&graph).unwrap();
    (q.bytes(), out.to_f32())
}

#[test]
fn cpu_a8_matches_exact_product_of_quantized_activations() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n, scale) = (128, 24, 5, 0.01f32);
    let w = weights(k, m, 7);
    let x = values(k * n, 8);
    let (q, got) = run_a8(&mut cpu, &w, &x, scale, k, m, n);
    let want_q: Vec<u8> = x.iter().map(|v| f32_to_e4m3(v / scale)).collect();
    assert_eq!(q, want_q);
    let xq: Vec<f32> = q.iter().map(|&b| e4m3_to_f32(b)).collect();
    let want: Vec<f64> = reference(&w, &xq, k, m, n)
        .into_iter()
        .map(|v| v * scale as f64)
        .collect();
    let e = rel_err(&got, &want);
    assert!(e < 1e-6, "relative error {e}");
}

/// The CUDA quantization is bit-identical to the CPU's; the matmul agrees to summation order.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_a8_matches_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let mut gpu = gpu();
    for (k, m, n) in [
        (2688, 300, 1),
        (4096, 129, 5),
        (2688, 200, 9),
        (4096, 1000, 30),
        (2688, 517, 64),
        (576, 130, 20),
        // Tall enough for the 8-warp skinny tiles at 64 tokens.
        (512, 12800, 40),
        (4096, 257, 100),
        (512, 64, 512),
    ] {
        let w = weights(k, m, 9);
        let x: Vec<f32> = values(k * n, 10).into_iter().map(|v| v * 3.0).collect();
        let (qc, yc) = run_a8(&mut cpu, &w, &x, 0.02, k, m, n);
        let (qg, yg) = run_a8(&mut gpu, &w, &x, 0.02, k, m, n);
        assert_eq!(qc, qg, "k {k} n {n}: quantized activations differ");
        let want: Vec<f64> = yc.iter().map(|&v| v as f64).collect();
        let e = rel_err(&yg, &want);
        println!("a8 k {k} m {m} n {n}: relative error {e:.2e}");
        let tol = a8_tolerance(n);
        assert!(e < tol, "k {k}, m {m}, n {n}: relative error {e}");
    }
}

/// Past 16 MiB of activations the big GEMMs walk their tiles in groups of 8 row tiles (11 row
/// tiles here: a full group and a partial one) instead of token-fastest; every tile is computed
/// the same way, so each token's outputs are bitwise those of a call on a slice of the tokens
/// below the threshold (slices start on a 128-token tile; the last tile is partial).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_grouped_tile_order_keeps_bits() {
    let mut gpu = gpu();
    let (k, m) = (4096, 1320);
    let w = weights(k, m, 11);
    // The first 1024 tokens and the tail from the last tile start at least 1000 tokens back.
    let slices = |n: usize| {
        let t0 = (n - 1000) / 128 * 128;
        [(0, 1024), (t0, n - t0)]
    };
    // W8A16 (BF16 activations): 2150 tokens are 17.6 MB.
    let n = 2150;
    assert!(n * k * 2 > 16 << 20);
    let x = values(k * n, 12);
    let all = run(&mut gpu, &w, &x, k, m, n);
    for (t0, len) in slices(n) {
        assert!(len * k * 2 <= 16 << 20);
        let part = run(&mut gpu, &w, &x[t0 * k..(t0 + len) * k], k, m, len);
        assert!(
            all[t0 * m..(t0 + len) * m]
                .iter()
                .zip(&part)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "W8A16 tokens {t0}..{}: grouped order changed bits",
            t0 + len
        );
    }
    // W8A8 (E4M3 activations): 4200 tokens are 17.2 MB.
    let n = 4200;
    assert!(n * k > 16 << 20);
    let x: Vec<f32> = values(k * n, 13).into_iter().map(|v| v * 3.0).collect();
    let (_, all) = run_a8(&mut gpu, &w, &x, 0.02, k, m, n);
    for (t0, len) in slices(n) {
        assert!(len * k <= 16 << 20);
        let (_, part) = run_a8(&mut gpu, &w, &x[t0 * k..(t0 + len) * k], 0.02, k, m, len);
        assert!(
            all[t0 * m..(t0 + len) * m]
                .iter()
                .zip(&part)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "W8A8 tokens {t0}..{}: grouped order changed bits",
            t0 + len
        );
    }
}

/// W8A8 timings at the same shapes as `cuda_timings` (quantization included).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_a8_timings() {
    const L: usize = 16;
    let mut gpu = gpu();
    for (k, m) in [(2688, 10304), (4096, 2688), (5120, 10240)] {
        let w = weights(k, m, 5);
        let weights = Context::new(L + 1, 0);
        let ws: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        let st = weights.new_tensor(types::F32, &[1]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for wt in &ws {
            wt.set_bytes(&w);
        }
        st.set_f32(&[0.02]);
        for n in [1usize, 8, 16, 32, 64, 128, 512, 1024] {
            let ctx = Context::new(4 * L + 4, 4 * L + 4);
            let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let q = fp8::quantize(&ctx, xt, st);
            let mut graph = ctx.new_graph(4 * L + 4);
            for wt in &ws {
                graph.expand(fp8::mul_mat_a8(&ctx, *wt, q, st, None).as_output());
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            xt.set_f32(&values(k * n, 6));
            gpu.compute(&graph).unwrap();
            let reps = 10;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                gpu.compute(&graph).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / (reps * L) as f64;
            println!(
                "a8 k {k} m {m} n {n:>4}: {:8.1} us  {:6.1} GB/s weights  {:6.2} TFLOPS",
                s * 1e6,
                (k * m) as f64 / s / 1e9,
                2.0 * (k * m * n) as f64 / s / 1e12
            );
        }
    }
}

/// W8A8 with per-token scales (`--fp8-act a8-dynamic`) on `backend`: the packed codes and token
/// scales of `fp8::quantize_dynamic`, then the E4M3 x E4M3 matmul with the weight scale
/// `scales` (one value, or one per row). `x` is `[n][k]`, given as `[k, n/2, 2]` when `n` is
/// even.
fn run_a8_dynamic(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    k: usize,
    m: usize,
    n: usize,
    scales: &[f32],
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let weights = Context::new(2, 0);
    let wt = weights.new_tensor(types::I8, &[k as i64, m as i64]);
    let ws = weights.new_tensor(types::F32, &[scales.len() as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    wt.set_bytes(w);
    ws.set_f32(scales);
    let ctx = Context::new(8, 16);
    let shape: Vec<i64> = if n.is_multiple_of(2) {
        vec![k as i64, n as i64 / 2, 2]
    } else {
        vec![k as i64, n as i64]
    };
    let xt = ctx.new_tensor(types::F32, &shape).as_input();
    let q = fp8::quantize_dynamic(&ctx, xt).as_output();
    let out = fp8::mul_mat_a8_dynamic(&ctx, wt, q, xt.ne(), Some(ws)).as_output();
    assert_eq!(out.ne()[1..], xt.ne()[1..]);
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(x);
    backend.compute(&graph).unwrap();
    let bytes = q.bytes();
    let (codes, total) = fp8::dynamic_layout(k as i64, n as i64);
    assert_eq!(bytes.len(), total);
    let ts = bytes[codes..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_ne_bytes(b))
        .collect();
    (bytes[..codes].to_vec(), ts, out.to_f32())
}

/// `fp8::mul_mat_dynamic(w, x) + res` on `backend` (`x` `[k, n]`, `res` `[m, n]`): the add in
/// the GEMM's epilogue (`fused`, `fp8::mul_mat_dynamic_add`), or ggml's `add` after it.
#[allow(clippy::too_many_arguments)]
fn run_dynamic_add(
    backend: &mut Backend,
    w: &[u8],
    x: &[f32],
    res: &[f32],
    (k, m, n): (usize, usize, usize),
    scales: Option<&[f32]>,
    fused: bool,
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let wt = weights.new_tensor(types::I8, &[k as i64, m as i64]);
    let ws = weights.new_tensor(types::F32, &[scales.map_or(1, |s| s.len()) as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    wt.set_bytes(w);
    ws.set_f32(scales.unwrap_or(&[1.0]));
    let ctx = Context::new(8, 16);
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let rt = ctx.new_tensor(types::F32, &[m as i64, n as i64]).as_input();
    let wscale = scales.map(|_| ws);
    let out = if fused {
        fp8::mul_mat_dynamic_add(&ctx, wt, xt, wscale, rt)
    } else {
        ctx.add(fp8::mul_mat_dynamic(&ctx, wt, xt, wscale), rt)
    }
    .as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(x);
    rt.set_f32(res);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

/// The residual added in the dynamic W8A8 matmul gives the bits of the matmul followed by
/// ggml's `add`, with no, a per-tensor and a per-row weight scale (#749).
#[test]
fn cpu_a8_dynamic_add_matches_matmul_then_add() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (k, m, n) in [(128, 24, 5), (64, 33, 70)] {
        let w = weights(k, m, 7);
        let x = dynamic_values(k, n, 8);
        let res = values(m * n, 9);
        let rows = row_scales(m);
        for ws in [None, Some(&[WS][..]), Some(&rows[..])] {
            let plain = run_dynamic_add(&mut cpu, &w, &x, &res, (k, m, n), ws, false);
            let fused = run_dynamic_add(&mut cpu, &w, &x, &res, (k, m, n), ws, true);
            assert!(same_bits(&plain, &fused), "k {k} m {m} n {n} {ws:?}");
        }
    }
}

/// The large-batch GEMM's fused residual add: ggml-cuda's matmul-then-add bits at every tile
/// shape it takes (more than 64 tokens), with and without weight scales (#749).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU with E4M3 tensor cores"]
fn cuda_a8_dynamic_add_matches_matmul_then_add() {
    let mut gpu = gpu();
    for (k, m, n) in [
        (2688, 300, 65),
        (4096, 257, 100),
        (5120, 300, 257),
        (512, 64, 512),
    ] {
        let w = weights(k, m, 9);
        let x = dynamic_values(k, n, 10);
        let res = values(m * n, 11);
        let rows = row_scales(m);
        for ws in [None, Some(&[WS][..]), Some(&rows[..])] {
            let plain = run_dynamic_add(&mut gpu, &w, &x, &res, (k, m, n), ws, false);
            let fused = run_dynamic_add(&mut gpu, &w, &x, &res, (k, m, n), ws, true);
            assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: other bits");
        }
        println!("a8-dynamic add k {k} m {m} n {n}: bit-exact");
    }
}

/// The large-batch W8A8 GEMM's 256-row tiles (m a multiple of 256 that gives at least 8 waves
/// of them, #749) give the bits of its 128-row tiles: the same rows of a weight 128 rows
/// shorter (not a multiple of 256) take the 128-row tiles. Plain and with the fused residual,
/// in both of the wide tile's stage layouts.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU with E4M3 tensor cores"]
fn cuda_a8_wide_tiles_keep_bits() {
    let mut gpu = gpu();
    // 24 x 32 = 768 tiles of 256 rows: the wide tiles on up to 96 SMs. k 2048 takes their
    // 128-byte K steps, k 2112 (not a multiple of 128) their 64-byte ones.
    for k in [2048, 2112] {
        let (m, n) = (6144, 4096);
        let short = m - 128;
        let w = weights(k, m, 21);
        let x = dynamic_values(k, n, 22);
        let res = values(m * n, 23);
        let rows = row_scales(m);
        // Each token's first `short` outputs of a [width, n] result.
        let first_rows = |out: &[f32], width: usize| -> Vec<f32> {
            out.chunks(width)
                .flat_map(|tok| tok[..short].iter().copied())
                .collect()
        };
        let (_, _, wide) = run_a8_dynamic(&mut gpu, &w, &x, k, m, n, &rows);
        let (_, _, narrow) =
            run_a8_dynamic(&mut gpu, &w[..short * k], &x, k, short, n, &rows[..short]);
        assert!(
            same_bits(&first_rows(&wide, m), &narrow),
            "256-row tiles changed bits"
        );
        let wide = run_dynamic_add(&mut gpu, &w, &x, &res, (k, m, n), Some(&rows), true);
        let narrow = run_dynamic_add(
            &mut gpu,
            &w[..short * k],
            &x,
            &first_rows(&res, m),
            (k, short, n),
            Some(&rows[..short]),
            true,
        );
        assert!(
            same_bits(&first_rows(&wide, m), &narrow),
            "256-row tiles with the residual changed bits"
        );
        println!("a8-dynamic k {k} m {m} n {n}: 256-row tiles bit-exact");
    }
}

/// Activations with per-token magnitudes far apart (one all-zero token), so a token that takes
/// another's scale shows.
fn dynamic_values(k: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut x = values(k * n, seed);
    for (t, tok) in x.chunks_mut(k).enumerate() {
        let mag = if t == 1 {
            0.0
        } else {
            2f32.powi(t as i32 % 9 - 4) * 7.0
        };
        tok.iter_mut().for_each(|v| *v *= mag);
    }
    x
}

/// Each token's codes and scale are `quantize_token`'s; the matmul is the exact product of the
/// quantized activations times each token's scale and each row's weight scale.
#[test]
fn cpu_a8_dynamic_matches_exact_product_of_quantized_activations() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (k, m, n) in [(128, 24, 5), (64, 33, 8)] {
        let w = weights(k, m, 7);
        let x = dynamic_values(k, n, 8);
        for ws in [vec![WS], row_scales(m)] {
            let (q, ts, got) = run_a8_dynamic(&mut cpu, &w, &x, k, m, n, &ws);
            let mut xq = Vec::new();
            for t in 0..n {
                let (want_q, want_s) = fp8::quantize_token(&x[t * k..(t + 1) * k]);
                assert_eq!(q[t * k..(t + 1) * k], want_q[..], "token {t}: codes");
                assert_eq!(ts[t].to_bits(), want_s.to_bits(), "token {t}: scale");
                xq.extend(want_q.iter().map(|&b| e4m3_to_f32(b) * want_s));
            }
            assert_eq!(ts[1], 0.0, "the all-zero token");
            let e = rel_err(&got, &scaled_reference(&w, &xq, k, m, n, &ws));
            assert!(e < 1e-6, "k {k}, m {m}, n {n}: relative error {e}");
            // Against the unquantized product: E4M3's 3 mantissa bits per activation.
            let e = rel_err(&got, &scaled_reference(&w, &x, k, m, n, &ws));
            assert!(e < 0.1, "k {k}, m {m}, n {n}: relative error {e} to W8A16");
        }
    }
}

/// The CUDA quantization (codes and token scales) is bit-identical to the CPU's on every path of
/// the matmul (GEMV, skinny and big GEMM tiles, partial tiles); the matmul agrees to summation
/// order, with a per-tensor and a per-row weight scale.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_a8_dynamic_matches_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let mut gpu = gpu();
    for (k, m, n) in [
        (2688, 300, 1),
        (4096, 129, 5),
        (2688, 200, 9),
        (4096, 1000, 30),
        (2688, 517, 64),
        (576, 130, 20),
        (512, 12800, 40),
        (4096, 257, 100),
        (5120, 300, 257),
        (512, 64, 512),
    ] {
        let w = weights(k, m, 9);
        let x = dynamic_values(k, n, 10);
        for ws in [vec![WS], row_scales(m)] {
            let (qc, tc, yc) = run_a8_dynamic(&mut cpu, &w, &x, k, m, n, &ws);
            let (qg, tg, yg) = run_a8_dynamic(&mut gpu, &w, &x, k, m, n, &ws);
            assert_eq!(qc, qg, "k {k} n {n}: quantized activations differ");
            let same = tc.iter().zip(&tg).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "k {k} n {n}: token scales differ");
            if n > 1 {
                let zero = &yg[m..2 * m];
                assert!(
                    zero.iter().all(|&v| v == 0.0),
                    "k {k} n {n}: the all-zero token"
                );
            }
            // Each token against its own scale: magnitudes span 256x, so a batch-wide scale
            // would hide a broken small token.
            let tol = a8_tolerance(n);
            for (t, (g, c)) in yg.chunks(m).zip(yc.chunks(m)).enumerate() {
                if t == 1 {
                    continue;
                }
                let want: Vec<f64> = c.iter().map(|&v| v as f64).collect();
                let e = rel_err(g, &want);
                assert!(
                    e < tol,
                    "k {k}, m {m}, n {n}, token {t}: relative error {e}"
                );
            }
            let want: Vec<f64> = yc.iter().map(|&v| v as f64).collect();
            let e = rel_err(&yg, &want);
            println!(
                "a8-dynamic k {k} m {m} n {n} ({} scales): relative error {e:.2e}",
                ws.len()
            );
        }
    }
}

/// Prefill GEMMs at Qwen3.8-27B's channel-wise FP8 shapes: W8A16 (`fp8::mul_mat`, BF16 tensor
/// cores) against W8A8 with per-token scales (`fp8::mul_mat_dynamic`, E4M3 tensor cores, one
/// quantization shared by all the weights of the graph as in a layer), TFLOPS.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_a8_dynamic_timings() {
    const L: usize = 8;
    let mut gpu = gpu();
    for (k, m) in [
        (5120, 6144),
        (5120, 10240),
        (5120, 17408),
        (17408, 5120),
        (6144, 5120),
    ] {
        let w = weights(k, m.min(1024), 5);
        let w: Vec<u8> = w.iter().copied().cycle().take(k * m).collect();
        let weights = Context::new(L + 1, 0);
        let wts: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        let st = weights.new_tensor(types::F32, &[m as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for wt in &wts {
            wt.set_bytes(&w);
        }
        st.set_f32(&row_scales(m));
        for n in [128usize, 512, 2048, 4096] {
            let mut line = format!("k {k:>5} m {m:>5} n {n:>4}:");
            for dynamic in [false, true] {
                let ctx = Context::new(4 * L + 4, 4 * L + 4);
                let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                let mut graph = ctx.new_graph(4 * L + 4);
                for wt in &wts {
                    let y = if dynamic {
                        fp8::mul_mat_dynamic(&ctx, *wt, xt, Some(st))
                    } else {
                        fp8::mul_mat(&ctx, *wt, xt, Some(st))
                    };
                    graph.expand(y.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                xt.set_f32(&values(k * n, 6));
                gpu.compute(&graph).unwrap();
                let reps = 5;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    gpu.compute(&graph).unwrap();
                }
                let s = t.elapsed().as_secs_f64() / (reps * L) as f64;
                line += &format!(
                    "  {} {:8.1} us {:6.1} TFLOPS",
                    if dynamic { "a8-dyn" } else { "a16" },
                    s * 1e6,
                    2.0 * (k * m * n) as f64 / s / 1e12
                );
            }
            println!("{line}");
        }
    }
}

/// The sliced-K GEMM (`--fp8-sliced-k`, 5-64 tokens) against the exact product, with a scale
/// per row: every tile it has (forced, where k allows it) and the one it picks by n, partial row
/// and token tiles, k that only the 4-warp tiles divide (768), and the BF16 activations' error
/// bound of the other GEMMs. With the GEMV on too, 5-9 tokens keep the GEMV's one-token bits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_sliced_k_matches_exact_product() {
    let mut gpu = gpu();
    let sliced = SmallBatch {
        sliced_k: true,
        ..SmallBatch::default()
    };
    for (k, m) in [(5120, 301), (768, 2050), (6144, 64), (17408, 40)] {
        let (w, ws) = (weights(k, m, 11), row_scales(m));
        let x = values(k * 64, 12);
        for n in [5usize, 8, 9, 13, 16, 17, 24, 25, 32, 33, 40, 57, 64] {
            let want = scaled_reference(&w, &x[..n * k], k, m, n, &ws);
            let tiles = (0..12u8).map(Some).chain([None]);
            for tile in tiles {
                let small = SmallBatch {
                    sliced_k_tile: tile,
                    ..sliced
                };
                let got = run_small(&mut gpu, &w, &x[..n * k], k, m, n, &ws, small);
                let e = rel_err(&got, &want);
                assert!(
                    e < 4e-3,
                    "k {k}, m {m}, n {n}, tile {tile:?}: relative error {e}"
                );
            }
        }
        let both = SmallBatch {
            gemv: true,
            ..sliced
        };
        for n in 5..=9 {
            let got = run_small(&mut gpu, &w, &x[..n * k], k, m, n, &ws, both);
            let gemv = run_with(&mut gpu, &w, &x[..n * k], k, m, n, &ws, true);
            let same = got
                .iter()
                .zip(&gemv)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "k {k}, m {m}, n {n}: not the GEMV's bits");
        }
    }
}

/// Times the default GEMMs against the sliced-K GEMM (`--fp8-sliced-k`) at Qwen3.8-27B's FP8
/// shapes for verify-sized batches, with a scale per row and enough weight copies per graph to
/// stream past the L2. `LEGION_FP8_TIMING_N` (comma list) picks the token counts,
/// `LEGION_FP8_SLICED_TILES` (comma list of tile numbers) adds forced tiles to compare.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_sliced_k_timings() {
    let list = |var: &str, default: &str| -> Vec<usize> {
        std::env::var(var)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().parse().unwrap())
            .collect()
    };
    let ns = list("LEGION_FP8_TIMING_N", "5,8,9,12,16,24,32,36,40,48,64");
    let tiles = list("LEGION_FP8_SLICED_TILES", "");
    let mut gpu = gpu();
    for (k, m) in [
        (5120, 1024),
        (5120, 6144),
        (5120, 10240),
        (5120, 12288),
        (5120, 17408),
        (17408, 5120),
        (6144, 5120),
        (5120, 248320),
    ] {
        let copies = (256usize << 20).div_ceil(k * m).clamp(1, 32);
        let block = weights(k, m.min(1024), 5);
        let w: Vec<u8> = block.iter().copied().cycle().take(k * m).collect();
        let weights = Context::new(copies + 1, 0);
        let wts: Vec<_> = (0..copies)
            .map(|_| weights.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        let st = weights.new_tensor(types::F32, &[m as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for wt in &wts {
            wt.set_bytes(&w);
        }
        st.set_f32(&row_scales(m));
        for &n in &ns {
            let mut line = format!("k {k:>5} m {m:>6} n {n:>2}:");
            let variants = [("gemm", None), ("gemv", None), ("sliced", None)]
                .into_iter()
                .chain(tiles.iter().map(|&t| ("tile", Some(t as u8))));
            for (name, tile) in variants {
                if name == "gemv" && !(5..=9).contains(&n) {
                    continue;
                }
                let small = SmallBatch {
                    gemv: name == "gemv",
                    sliced_k: name == "sliced" || name == "tile",
                    sliced_k_tile: tile,
                };
                let ctx = Context::new(4 * copies + 4, 4 * copies + 4);
                let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                let mut graph = ctx.new_graph(4 * copies + 4);
                for wt in &wts {
                    let y = fp8::mul_mat_small(&ctx, *wt, xt, Some(st), small);
                    graph.expand(y.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                xt.set_f32(&values(k * n, 6));
                gpu.compute(&graph).unwrap();
                let reps = 20;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    gpu.compute(&graph).unwrap();
                }
                let s = t.elapsed().as_secs_f64() / (reps * copies) as f64;
                let label = match tile {
                    Some(t) => format!("t{t}"),
                    None => name.to_string(),
                };
                line += &format!(
                    "  {label} {:7.1} us {:4.0} GB/s",
                    s * 1e6,
                    (k * m) as f64 / s / 1e9
                );
            }
            println!("{line}");
        }
    }
}
