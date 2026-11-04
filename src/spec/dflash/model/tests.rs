//! The drafter's projections: which take the dense matmul op, and that without it (no GPU) they
//! are ggml's matmuls, bit for bit.

use super::*;
use crate::model::dense_mv;

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

/// BF16 bytes of `n` noise values.
fn bf16_noise(n: usize, seed: u64) -> Vec<u8> {
    noise(n, seed)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_ne_bytes())
        .collect()
}

/// Gemma 4 31B's DFlash 2 drafter (hidden 5,376, 32 query and 16 KV heads of 256, FFN 21,504,
/// the tied 262,144-row head): a draft block of one or two sequences (8 or 16 rows) and a short
/// ingest fit the dense op, four sequences' blocks (32 rows) its wide chain.
#[test]
fn gemma_dflash2_projections_fit_the_dense_op() {
    let ctx = Context::new(64, 0);
    let shapes = [
        (5376, 8192),
        (5376, 4096),
        (8192, 5376),
        (5376, 21504),
        (21504, 5376),
        (5376, 262_144),
    ];
    for (k, m) in shapes {
        let w = ctx.new_tensor(types::BF16, &[k, m]);
        let x = |n: i64| ctx.new_tensor(types::F32, &[k, n]);
        for n in [2, 7, 8, 14, 16] {
            assert!(dense_mv::fits(w, x(n)), "[{k}, {m}] at {n} rows");
        }
        for n in [17, 28, 32, 48] {
            assert!(!dense_mv::fits(w, x(n)), "[{k}, {m}] at {n} rows");
            assert!(dense_mv::fits_wide_chain(w, x(n)), "[{k}, {m}] at {n} rows");
        }
    }
}

/// On the CPU (the op takes GPU weights only) [`mul_mats`] and [`mul_mat`] are ggml's matmul.
#[test]
fn without_a_gpu_the_projections_are_ggml_matmuls() {
    let mut backend = Backend::cpu().unwrap();
    let (k, n) = (128i64, 8i64);
    let ms = [64i64, 32];
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = ms
        .iter()
        .map(|&m| ctx.new_tensor(types::BF16, &[k, m]).as_input())
        .collect();
    let joint: Vec<_> = mul_mats(&ctx, &ws, x)
        .into_iter()
        .map(|t| ctx.cont(t).as_output())
        .collect();
    let one = mul_mat(&ctx, ws[0], x).as_output();
    let ggml: Vec<_> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
    let mut graph = ctx.new_graph(32);
    for &t in joint.iter().chain(&ggml).chain([&one]) {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((k * n) as usize, 11));
    for (i, (w, &m)) in ws.iter().zip(&ms).enumerate() {
        w.set_bytes(&bf16_noise((k * m) as usize, 20 + i as u64));
    }
    backend.compute(&graph).unwrap();
    let bits = |t: Tensor<'_>| -> Vec<u32> { t.to_f32().iter().map(|v| v.to_bits()).collect() };
    for (i, (&a, &b)) in joint.iter().zip(&ggml).enumerate() {
        assert_eq!(a.ne(), [ms[i], n, 1, 1], "weight {i} shape");
        assert_eq!(bits(a), bits(b), "weight {i}");
    }
    assert_eq!(bits(one), bits(ggml[0]));
}
