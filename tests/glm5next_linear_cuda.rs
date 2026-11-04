//! GLM-5.3-Flash's dense linear (`src/cuda/glm5next.cu`) on Project Maya's Q6_K and Q4_K weights
//! (#559, `src/cuda/glm5next_kquant.cuh`) against the op's CPU reference (weights dequantized by
//! ggml, F32 sums), next to Q8_0: the GEMV (up to 8 tokens, 16-byte aligned activations), the
//! generic GEMV (unaligned activations), the tensor-core GEMM (more tokens), per-head matrices
//! (MLA's `attn_k_b` / `attn_v_b`) and the embedding rows. `timings` prints each type's GEMV rate.
//!
//! `cargo test --release --features cuda --test glm5next_linear_cuda -- --ignored --nocapture --test-threads 1`

#![cfg(feature = "cuda")]

use legion::ggml::{
    Backend, ComputeAllocator, Context, Tensor, backend_by_name, quantize, row_size, types,
};
use legion::model::glm5next::ops;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32() * scale).collect()
    }
}

const TYPES: [(u32, &str); 3] = [
    (types::Q8_0, "Q8_0"),
    (types::Q6_K, "Q6_K"),
    (types::Q4_K, "Q4_K"),
];

/// `linear(w, x)` on `backend`: `w` `[k, m, h]` of `ty` (bytes `wq`), `x` F32 `[k, h or n, n or 1]`
/// read through a view `pad` floats into rows of `k + pad` (pad 1: activations off 16 bytes).
#[allow(clippy::too_many_arguments)]
fn linear(
    backend: &mut Backend,
    ty: u32,
    wq: &[u8],
    [k, m, h]: [usize; 3],
    x: &[f32],
    n: usize,
    pad: usize,
) -> Vec<f32> {
    let weights = Context::new(1, 0);
    let w = weights.new_tensor(ty, &[k as i64, m as i64, h as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    let ctx = Context::new(8, 16);
    let rows = if h == 1 { n } else { h * n };
    let xt = ctx.new_tensor(types::F32, &[(k + pad) as i64, rows as i64]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let mut xp = vec![0f32; (k + pad) * rows];
    for r in 0..rows {
        xp[r * (k + pad) + pad..(r + 1) * (k + pad)].copy_from_slice(&x[r * k..(r + 1) * k]);
    }
    xt.set_f32(&xp);
    let nb1 = (k + pad) * 4;
    let xv: Tensor<'_> = if h == 1 {
        ctx.view_2d(xt, [k as i64, n as i64], nb1, pad * 4)
    } else {
        ctx.view_3d(xt, [k as i64, h as i64, n as i64], [nb1, nb1 * h], pad * 4)
    };
    let out = ops::linear(&ctx, w, xv, 0.75, None).as_output();
    let mut g = ctx.new_graph(16);
    g.expand(out);
    ctx.alloc_graph(&g, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    backend.compute(&g).unwrap();
    out.to_f32()
}

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn linear_matches_cpu() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    // A dense projection, and MLA's per-head attn_k_b ([256, 512, heads]).
    for shape in [[1024, 136, 1], [256, 64, 4]] {
        let [k, m, h] = shape;
        let mut rng = Rng(7 + k as u64);
        let wf = rng.vec(k * m * h, 0.05);
        for (ty, name) in TYPES {
            let wq = quantize(ty, &wf, k);
            for n in [1, 2, 3, 4, 5, 8, 9, 40] {
                let x = rng.vec(k * h.max(1) * n, 1.0);
                for pad in [0, 1] {
                    let want = linear(&mut cpu, ty, &wq, shape, &x, n, pad);
                    let got = linear(&mut gpu, ty, &wq, shape, &x, n, pad);
                    let d = rel_diff(&got, &want);
                    // GEMV: F32 sums; GEMM: x rounded to F16 (as for Q8_0).
                    let tol = if n <= 8 { 1e-5 } else { 3e-3 };
                    eprintln!("{name} {shape:?} n {n} pad {pad}: {d:e}");
                    assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite");
                    assert!(d < tol, "{name} {shape:?} n {n} pad {pad}: {d:e}");
                }
            }
        }
    }
}

/// Embedding rows of each type (`token_embd`): the dequantized values exactly.
#[test]
#[ignore = "needs a CUDA GPU"]
fn embed_matches_cpu() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    let mut cpu = Backend::cpu().unwrap();
    let (k, vocab, hc) = (512usize, 40usize, 4i64);
    let mut rng = Rng(3);
    let wf = rng.vec(k * vocab, 0.1);
    let tokens = [0i32, 39, 7, 7, 21];
    for (ty, name) in TYPES {
        let wq = quantize(ty, &wf, k);
        let run = |b: &mut Backend| {
            let weights = Context::new(1, 0);
            let w = weights.new_tensor(ty, &[k as i64, vocab as i64]);
            weights.alloc_tensors(b.buffer_type()).unwrap();
            w.set_bytes(&wq);
            let ctx = Context::new(4, 8);
            let t = ctx.new_tensor(types::I32, &[tokens.len() as i64]);
            ctx.alloc_tensors(b.buffer_type()).unwrap();
            t.set_i32(&tokens);
            let out = ops::embed(&ctx, w, t, hc).as_output();
            let mut g = ctx.new_graph(8);
            g.expand(out);
            ctx.alloc_graph(&g, ComputeAllocator::new(b.buffer_type()))
                .unwrap();
            b.compute(&g).unwrap();
            out.to_f32()
        };
        let (got, want) = (run(&mut gpu), run(&mut cpu));
        assert_eq!(got, want, "{name}");
    }
}

/// Copies of the matrix a timed graph holds, so each product reads other memory than the last.
const LAYERS: usize = 8;

/// GEMV time per product and weight bytes per second for GLM-5.3-Flash's dense shapes (attention
/// and shared-expert projections 4096 <-> 2048, the dense FFN's 4096 -> 10240) and 1 / 4 / 8
/// tokens, each type next to Q8_0.
#[test]
#[ignore = "needs a CUDA GPU"]
fn timings() {
    let mut gpu = backend_by_name("CUDA0").unwrap();
    println!(
        "{:<6} {:>6} {:>6} {:>4} {:>10} {:>8}",
        "type", "k", "m", "n", "µs", "GB/s"
    );
    for [k, m] in [[4096usize, 2048usize], [2048, 4096], [4096, 10240]] {
        let mut rng = Rng(5);
        let wf = rng.vec(k * m, 0.05);
        for (ty, name) in TYPES {
            let wq = quantize(ty, &wf, k);
            let bytes = row_size(ty, k) * m;
            for n in [1, 4, 8] {
                let weights = Context::new(LAYERS, 0);
                let ws: Vec<_> = (0..LAYERS)
                    .map(|_| weights.new_tensor(ty, &[k as i64, m as i64]))
                    .collect();
                weights.alloc_tensors(gpu.buffer_type()).unwrap();
                for w in &ws {
                    w.set_bytes(&wq);
                }
                let ctx = Context::new(4 * LAYERS + 4, 4 * LAYERS + 8);
                let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]);
                ctx.alloc_tensors(gpu.buffer_type()).unwrap();
                xt.set_f32(&rng.vec(k * n, 1.0));
                let mut g = ctx.new_graph(4 * LAYERS + 8);
                for &w in &ws {
                    g.expand(ops::linear(&ctx, w, xt, 1.0, None));
                }
                ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                gpu.compute(&g).unwrap();
                let reps = 20;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    gpu.compute(&g).unwrap();
                }
                let s = t.elapsed().as_secs_f64() / (reps * LAYERS) as f64;
                println!(
                    "{name:<6} {k:>6} {m:>6} {n:>4} {:>10.1} {:>8.1}",
                    s * 1e6,
                    bytes as f64 / s / 1e9
                );
            }
        }
    }
}
