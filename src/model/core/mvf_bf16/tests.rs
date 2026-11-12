//! `mvf_bf16` against ggml's matmuls: on the CPU with values whose sums are exact in any order;
//! on CUDA (an ignored test) bit for bit.

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

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The op's results and ggml's `mul_mat` of each weight (`rows`) over `n` tokens of `k`, as bits.
/// `exact`: small integers in the weights and halves in `x`, so every order sums exactly.
fn case(backend: &mut Backend, k: i64, rows: &[i64], n: i64, exact: bool) -> (Vec<u32>, Vec<u32>) {
    let ctx = Context::new(64, 64);
    let ws: Vec<Tensor<'_>> = rows
        .iter()
        .map(|&m| ctx.new_tensor(types::BF16, &[k, m]).as_input())
        .collect();
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let got: Vec<Tensor<'_>> = mul_mat(&ctx, &ws, x)
        .into_iter()
        .map(|t| ctx.cont(t).as_output())
        .collect();
    let want: Vec<Tensor<'_>> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
    let mut graph = ctx.new_graph(64);
    for &t in got.iter().chain(&want) {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    for (i, (w, &m)) in ws.iter().zip(rows).enumerate() {
        let v: Vec<f32> = noise((k * m) as usize, 3 + i as u64)
            .iter()
            .map(|v| if exact { (v * 8.0).round() } else { v * 0.04 })
            .collect();
        w.set_bytes(&bf16_bytes(&v));
    }
    let xv: Vec<f32> = noise((k * n) as usize, 17)
        .iter()
        .map(|v| if exact { (v * 4.0).round() / 2.0 } else { *v })
        .collect();
    x.set_f32(&xv);
    backend.compute(&graph).unwrap();
    let bits = |ts: &[Tensor<'_>]| -> Vec<u32> {
        ts.iter()
            .flat_map(|t| t.to_f32())
            .map(f32::to_bits)
            .collect()
    };
    (bits(&got), bits(&want))
}

#[test]
fn mvf_bf16_is_ggmls_matmuls() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (k, rows, n) in [
        (256, &[48, 48][..], 8),
        (256, &[48, 48][..], 1),
        (64, &[16, 48, 3][..], 5),
        (2, &[1][..], 2),
    ] {
        let (got, want) = case(&mut backend, k, rows, n, true);
        assert_eq!(got, want, "k {k}, rows {rows:?}, {n} tokens");
    }
}

#[test]
fn fits_takes_mul_mat_vec_f_shapes_only() {
    let ctx = Context::new(64, 16);
    let w = || ctx.new_tensor(types::BF16, &[5120, 48]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[5120, n]);
    for n in 1..=MAX_TOKENS {
        assert!(fits(&[w(), w()], x(n)), "{n} tokens");
    }
    // ggml runs 9 tokens on cuBLAS.
    assert!(!fits(&[w(), w()], x(MAX_TOKENS + 1)));
    // 32 or 64 rows: ggml's mul_mat_f from 2 tokens.
    assert!(!fits(
        &[ctx.new_tensor(types::BF16, &[5120, 64]), w()],
        x(4)
    ));
    // an F16 or F32 weight, a mismatched input, an odd row, too many weights
    assert!(!fits(&[ctx.new_tensor(types::F16, &[5120, 48])], x(1)));
    assert!(!fits(&[ctx.new_tensor(types::F32, &[5120, 48])], x(1)));
    assert!(!fits(&[w()], ctx.new_tensor(types::F32, &[2048, 1])));
    assert!(!fits(
        &[ctx.new_tensor(types::BF16, &[63, 48])],
        ctx.new_tensor(types::F32, &[63, 1])
    ));
    assert!(!fits(&[w(), w(), w(), w(), w()], x(1)));
    assert!(!fits(&[], x(1)));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mvf_bf16_has_ggmls_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        return;
    }
    for (k, rows) in [(5120, &[48, 48][..]), (2048, &[16, 48][..])] {
        for n in 1..=MAX_TOKENS {
            let (got, want) = case(&mut gpu, k, rows, n, false);
            let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "k {k}, rows {rows:?}, {n} tokens");
        }
    }
}
