//! `legion_router` against ggml's matmul: on the CPU with values whose sums are exact in any
//! order; on CUDA (ignored tests) the same bits for a token in steps of every size, close to an
//! f64 oracle, and its speed next to ggml-cuda's cuBLAS GEMM.

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

/// The BF16 bytes of `v` (each value exact in BF16).
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| {
            let b = f.to_bits();
            assert_eq!(b & 0xffff, 0, "not exact in BF16");
            ((b >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// `v` truncated to BF16.
#[cfg(feature = "cuda")]
fn to_bf16(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|f| f32::from_bits(f.to_bits() & 0xffff_0000))
        .collect()
}

/// The op and ggml's F32 matmul of the same values on `backend`: `(op, ggml)` logits.
fn pair(
    backend: &mut Backend,
    wv: &[f32],
    k_embd: i64,
    n_expert: i64,
    xv: &[f32],
) -> [Vec<f32>; 2] {
    let n = xv.len() as i64 / k_embd;
    let ctx = Context::new(16, 16);
    let wb = ctx.new_tensor(types::BF16, &[k_embd, n_expert]).as_input();
    let wf = ctx.new_tensor(types::F32, &[k_embd, n_expert]).as_input();
    let x = ctx.new_tensor(types::F32, &[k_embd, n]).as_input();
    let outs = [mul_mat(&ctx, wb, x), ctx.mul_mat(wf, x)].map(|t| t.as_output());
    let mut graph = ctx.new_graph(16);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    wb.set_bytes(&bf16_bytes(wv));
    wf.set_f32(wv);
    x.set_f32(xv);
    backend.compute(&graph).unwrap();
    outs.map(|t| t.to_f32())
}

#[test]
fn legion_router_is_the_matmul() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (k_embd, n_expert, n) in [(256, 512, 1), (64, 33, 20), (128, 7, 3)] {
        // Small integers in the weight and halves in x: every order sums exactly.
        let wv: Vec<f32> = noise((k_embd * n_expert) as usize, 1)
            .iter()
            .map(|v| (v * 8.0).round())
            .collect();
        let xv: Vec<f32> = noise((k_embd * n) as usize, 2)
            .iter()
            .map(|v| (v * 4.0).round() / 2.0)
            .collect();
        let [op, mm] = pair(&mut backend, &wv, k_embd, n_expert, &xv);
        assert_eq!(op, mm, "{k_embd} x {n_expert}, {n} tokens");
    }
}

#[test]
fn fits_takes_bf16_routers_only() {
    let ctx = Context::new(16, 0);
    let x = ctx.new_tensor(types::F32, &[2560, 4096]);
    assert!(fits(ctx.new_tensor(types::BF16, &[2560, 512]), x));
    assert!(!fits(ctx.new_tensor(types::F32, &[2560, 512]), x));
    // K a multiple of 64
    let x96 = ctx.new_tensor(types::F32, &[96, 4]);
    assert!(!fits(ctx.new_tensor(types::BF16, &[96, 8]), x96));
    let xh = ctx.new_tensor(types::F16, &[2560, 4]);
    assert!(!fits(ctx.new_tensor(types::BF16, &[2560, 512]), xh));
    // the K of both
    let x2 = ctx.new_tensor(types::F32, &[2048, 4]);
    assert!(!fits(ctx.new_tensor(types::BF16, &[2560, 512]), x2));
}

#[test]
fn off_by_default() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::BF16, &[2560, 512]);
    let x = ctx.new_tensor(types::F32, &[2560, 4096]);
    assert!(!on());
    assert!(!takes(w, x));
    set_legion_router(true);
    // Steps of up to 16 tokens keep their own kernels, and without CUDA nothing takes the op.
    assert!(!takes(w, ctx.new_tensor(types::F32, &[2560, 16])));
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
    }
    set_legion_router(false);
}

/// A router-like case: BF16 weights of ~0.05, activations of ~1 with a few large channels.
#[cfg(feature = "cuda")]
fn router_case(k_embd: usize, n_expert: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let wv = to_bf16(
        &noise(k_embd * n_expert, 31)
            .iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>(),
    );
    let mut xv = noise(k_embd * n, 32);
    for (i, v) in xv.iter_mut().enumerate() {
        if i % k_embd % 97 == 5 {
            *v *= 40.0;
        }
    }
    (wv, xv)
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

/// The top `k` ids of each token's logits (descending, ties to the lower id).
#[cfg(feature = "cuda")]
fn topk(l: &[f32], n_expert: usize, k: usize) -> Vec<Vec<usize>> {
    l.chunks(n_expert)
        .map(|r| {
            let mut o: Vec<usize> = (0..n_expert).collect();
            o.sort_by(|&a, &b| r[b].total_cmp(&r[a]).then(a.cmp(&b)));
            o.truncate(k);
            o.sort_unstable();
            o
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rows_do_not_depend_on_the_step() {
    let mut gpu = gpu();
    if !device_supported() {
        return;
    }
    let (k_embd, n_expert, n_all) = (2560usize, 512usize, 6182usize);
    let (wv, xv) = router_case(k_embd, n_expert, n_all);
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let [all, cublas] = pair(&mut gpu, &wv, k_embd as i64, n_expert as i64, &xv);
    // (first token, tokens): steps of 1, 7, 17, 64, 65, 1024, 1025, 4096 tokens at the start and
    // the 2,086-token tail alone.
    for (t0, n) in [
        (0, 1),
        (0, 7),
        (0, 17),
        (3, 64),
        (0, 65),
        (0, 1024),
        (9, 1025),
        (0, 4096),
        (4096, 2086),
    ] {
        let [op, _] = pair(
            &mut gpu,
            &wv,
            k_embd as i64,
            n_expert as i64,
            &xv[t0 * k_embd..(t0 + n) * k_embd],
        );
        assert_eq!(
            bits(&op),
            bits(&all[t0 * n_expert..(t0 + n) * n_expert]),
            "tokens {t0}..{} alone vs in a step of {n_all}",
            t0 + n
        );
    }
    // Close to the f64 sums of the same values; cuBLAS (TF32) for comparison.
    let (mut err, mut err_cublas, mut scale) = (0f64, 0f64, 0f64);
    for t in 0..n_all {
        let x = &xv[t * k_embd..(t + 1) * k_embd];
        for e in 0..n_expert {
            let w = &wv[e * k_embd..(e + 1) * k_embd];
            let (s, a) = w.iter().zip(x).fold((0f64, 0f64), |(s, a), (&w, &x)| {
                (
                    s + f64::from(w) * f64::from(x),
                    a + (f64::from(w) * f64::from(x)).abs(),
                )
            });
            let i = t * n_expert + e;
            err = err.max((f64::from(all[i]) - s).abs());
            err_cublas = err_cublas.max((f64::from(cublas[i]) - s).abs());
            scale = scale.max(a);
        }
    }
    let k = 10;
    let agree = topk(&all, n_expert, k)
        .iter()
        .zip(topk(&cublas, n_expert, k))
        .filter(|(a, b)| **a == *b)
        .count();
    eprintln!(
        "legion_router: max |op - f64| {err:.3e}, max |cuBLAS - f64| {err_cublas:.3e} (max sum |w x| {scale:.1}); \
         top-{k} sets equal to cuBLAS's for {agree} / {n_all} tokens"
    );
    // x is carried to ~2^-17 of itself and every product is exact: the bound of the split's
    // error alone, well under TF32's 2^-11.
    assert!(
        err <= 4e-6 * scale.max(1.0),
        "op vs f64: {err:e} (scale {scale})"
    );
}

/// Milliseconds per call of the op and of ggml-cuda's matmul (cuBLAS from 17 tokens) at the
/// router's shape, over a graph of 48 calls (one per MoE layer).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; prints timings"]
fn cuda_speed_next_to_cublas() {
    let mut gpu = gpu();
    if !device_supported() {
        return;
    }
    const CALLS: usize = 48;
    let (k_embd, n_expert) = (2560usize, 512usize);
    for n in [17usize, 32, 64, 128, 256, 1024, 2086, 4096, 12288] {
        let (wv, xv) = router_case(k_embd, n_expert, n);
        let mut ms = [0f64; 2];
        for (which, slot) in ms.iter_mut().enumerate() {
            let ctx = Context::new(4 + CALLS, CALLS + 8);
            let wb = ctx
                .new_tensor(types::BF16, &[k_embd as i64, n_expert as i64])
                .as_input();
            let wf = ctx
                .new_tensor(types::F32, &[k_embd as i64, n_expert as i64])
                .as_input();
            let x = ctx
                .new_tensor(types::F32, &[k_embd as i64, n as i64])
                .as_input();
            let mut graph = ctx.new_graph(CALLS + 8);
            for _ in 0..CALLS {
                let l = if which == 0 {
                    mul_mat(&ctx, wb, x)
                } else {
                    ctx.mul_mat(wf, x)
                };
                graph.expand(l.as_output());
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            // Only the graph's own weight has memory.
            if which == 0 {
                wb.set_bytes(&bf16_bytes(&wv));
            } else {
                wf.set_f32(&wv);
            }
            x.set_f32(&xv);
            gpu.compute(&graph).unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                gpu.compute(&graph).unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
            }
            *slot = best;
        }
        eprintln!(
            "legion_router {n_expert} x {k_embd}, {n:>5} tokens: op {:.4} ms, ggml-cuda {:.4} ms",
            ms[0], ms[1]
        );
    }
}
