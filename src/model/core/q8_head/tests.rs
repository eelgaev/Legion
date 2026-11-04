//! The one-column Q8_0 matvec op's shape rules, its quantization and its CPU implementation. The
//! GPU bits against ggml's `mul_mat`: `legion q8-head-bench --check` on the GPU box
//! ([`super::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn fits_takes_one_column_q8_0_heads() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 248320]);
    let col = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    assert!(fits(w, col(1)));
    // ggml's 2-8 column kernels already take two rows per CTA.
    assert!(!fits(w, col(2)));
    assert!(!fits(w, col(4)));
    // Row lengths the kernel is not built for, another weight type, an F16 input.
    let x = col(1);
    for k in [2304, 512] {
        let w = ctx.new_tensor(types::Q8_0, &[k, 4096]);
        assert!(!fits(w, ctx.new_tensor(types::F32, &[k, 1])), "k {k}");
    }
    assert!(!fits(ctx.new_tensor(types::Q4_0, &[2560, 4096]), x));
    assert!(!fits(w, ctx.new_tensor(types::F16, &[2560, 1])));
    for k in K {
        let w = ctx.new_tensor(types::Q8_0, &[k, 64]);
        assert!(fits(w, ctx.new_tensor(types::F32, &[k, 1])), "k {k}");
    }
    // The first rows of the head (a view, `--spec-draft-vocab`) fit.
    assert!(fits(ctx.view_2d(w, [2560, 65536], w.nb()[1], 0), x));
}

#[test]
fn takes_needs_a_cuda_build() {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 248320]);
    let x = ctx.new_tensor(types::F32, &[2560, 1]);
    assert!(fits(w, x));
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
        assert!(!device_supported());
    }
}

#[test]
fn quantize_block_is_ggml_quantize_q8_1() {
    // All zero: scale 0, values 0.
    let (d, q) = quantize_block(&[0.0; GROUP]);
    assert_eq!(d, 0.0);
    assert_eq!(q, [0i8; GROUP]);
    // The largest magnitude maps to +-127, the rest to the nearest step.
    let mut x = [0f32; GROUP];
    x[3] = -2.54;
    x[4] = 0.061;
    x[5] = 1.3;
    let (d, q) = quantize_block(&x);
    assert!((d - 0.02).abs() < 1e-4, "{d}");
    assert_eq!(q[3], -127);
    assert_eq!(q[4], 3);
    assert_eq!(q[5], 65);
    // The scale is rounded to F16.
    // SAFETY: plain conversions.
    let h = unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(2.54f32 / 127.0)) };
    assert_eq!(d, h);
}

/// `mul_mat` on the CPU against the product of the dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    cpu_product(2560, false);
}

/// The same for the 6,144-value projections' node ([`mul_mat_out`]).
#[test]
fn cpu_mul_mat_out_is_the_dequantized_product() {
    cpu_product(K_OUT as usize, true);
}

#[test]
fn fits_out_takes_one_column_projections_of_6144() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::Q8_0, &[K_OUT, 2560]);
    let col = |k: i64, n: i64| ctx.new_tensor(types::F32, &[k, n]);
    assert!(fits_out(w, col(K_OUT, 1)));
    // One column only; the head's row lengths stay the head's.
    assert!(!fits_out(w, col(K_OUT, 3)));
    assert!(!fits(w, col(K_OUT, 1)));
    let w2560 = ctx.new_tensor(types::Q8_0, &[2560, 2560]);
    assert!(!fits_out(w2560, col(2560, 1)));
    assert!(!fits_out(
        ctx.new_tensor(types::Q4_K, &[K_OUT, 2560]),
        col(K_OUT, 1)
    ));
    if !cfg!(feature = "cuda") {
        assert!(!takes_out(w, col(K_OUT, 1)));
    }
}

/// `rows` rows of `k` values on the CPU (`out`: the [`mul_mat_out`] node, else [`mul_mat`])
/// against the product of the dequantized operands in f64.
fn cpu_product(k: usize, out: bool) {
    let rows = 96usize;
    let wb = q8_0_matrix(k, rows, 11);
    let x = noise(k, 12);
    let xq: Vec<_> = x.chunks(GROUP).map(quantize_block).collect();

    let ctx = Context::new(16, 16);
    let w = ctx
        .new_tensor(types::Q8_0, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let y = if out {
        mul_mat_out(&ctx, w, xt)
    } else {
        mul_mat(&ctx, w, xt)
    }
    .as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&wb);
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();
    let got = y.to_f32();
    assert_eq!(got.len(), rows);

    let mut worst = 0f64;
    for (r, g) in got.iter().enumerate() {
        let (mut want, mut mag) = (0f64, 0f64);
        for v in 0..k {
            let blk = &wb[(r * k + v) / GROUP * Q8_0_BYTES..];
            // SAFETY: a plain conversion.
            let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
            let qw = blk[2 + v % GROUP] as i8;
            let (dx, qx) = &xq[v / GROUP];
            let t = dw as f64 * qw as f64 * *dx as f64 * qx[v % GROUP] as f64;
            want += t;
            mag += t.abs();
        }
        // F32 sums of k terms: off by a few ulps of the terms' magnitude.
        worst = worst.max((*g as f64 - want).abs() / mag);
    }
    assert!(worst < 1e-5, "rel {worst:e}");
}

/// The order of ggml's sums matters to the bits: a row whose blocks only feed one virtual thread
/// gives that thread's sum alone.
#[test]
fn row_dot_follows_ggml_threads() {
    let k = 1024usize;
    let nb = k / GROUP;
    let mut row = vec![0u8; nb * Q8_0_BYTES];
    // Block 5, values 8..16 (ints 2 and 3: virtual thread 4 * 5 + 1 = 21) only.
    // SAFETY: a plain conversion.
    let one = unsafe { ffi::ggml_fp32_to_fp16(1.0) };
    row[5 * Q8_0_BYTES..][..2].copy_from_slice(&one.to_le_bytes());
    for i in 8..16 {
        row[5 * Q8_0_BYTES + 2 + i] = 2;
    }
    let mut xq = vec![(0.5f32, [0i8; GROUP]); nb];
    xq[5].1 = [3; GROUP];
    assert_eq!(row_dot(&row, &xq), 0.5 * 8.0 * 6.0);
}

#[test]
fn force_variant_needs_cuda() {
    assert_eq!(force_variant(None), cfg!(feature = "cuda"));
    assert_eq!(force_variant(Some((8, 0, 2))), cfg!(feature = "cuda"));
    assert!(!force_variant(Some((6, 0, 1))));
    assert!(!force_variant(Some((4, 0, 3))));
    force_variant(None);
}

#[test]
fn row_mv_fits_takes_aligned_q8_0_weights() {
    let row = |k: i64| k as usize / GROUP * Q8_0_BYTES;
    assert!(row_mv_fits(2560, &[(0, row(2560))]));
    assert!(row_mv_fits(1024, &[(4096, row(1024)); 4]));
    // Row lengths the kernel is not built for, no weight or more than it walks.
    assert!(!row_mv_fits(6144, &[(0, row(6144))]));
    assert!(!row_mv_fits(2560, &[]));
    assert!(!row_mv_fits(2560, &[(0, row(2560)); XQ_WEIGHTS + 1]));
    // 16-byte loads: an odd base or row stride, and a row shorter than k values.
    assert!(!row_mv_fits(2560, &[(0, row(2560)), (8, row(2560))]));
    assert!(!row_mv_fits(2560, &[(0, row(2560) + 8)]));
    assert!(!row_mv_fits(2560, &[(0, row(2560) - 16)]));
}
