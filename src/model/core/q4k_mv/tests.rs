//! The Q4_K matvec op's shape rules, its kernel table and its CPU implementation. The GPU bits
//! against ggml's `mul_mat`: `legion q4k-mv-bench --check` on a GB10 ([`super::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;
use crate::model::core::q4k_head::quantize_block;

#[test]
fn fits_takes_the_drafter_shapes_of_one_to_eight_columns() {
    let ctx = Context::new(64, 0);
    for &(k, rows) in &bench::DRAFTER_SHAPES {
        let w = ctx.new_tensor(types::Q4_K, &[k as i64, rows as i64]);
        for n in 1..=MAX_COLS {
            assert!(
                fits(w, ctx.new_tensor(types::F32, &[k as i64, n])),
                "k {k} rows {rows} cols {n}"
            );
        }
        // ggml runs MMQ from 9 columns.
        assert!(!fits(w, ctx.new_tensor(types::F32, &[k as i64, 9])));
    }
    // Row lengths the kernels are not built for (the drafter's down projection, the MTP head's),
    // another weight type, an F16 input.
    for k in [2560, 17408] {
        let w = ctx.new_tensor(types::Q4_K, &[k, 4096]);
        assert!(!fits(w, ctx.new_tensor(types::F32, &[k, 1])), "k {k}");
    }
    let col = ctx.new_tensor(types::F32, &[5120, 1]);
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[5120, 4096]), col));
    assert!(!fits(ctx.new_tensor(types::Q4_0, &[5120, 4096]), col));
    let w = ctx.new_tensor(types::Q4_K, &[5120, 4096]);
    assert!(!fits(w, ctx.new_tensor(types::F16, &[5120, 1])));
    // The first rows of a larger weight (a view) fit.
    assert!(fits(ctx.view_2d(w, [5120, 1024], w.nb()[1], 0), col));
}

#[test]
fn takes_needs_a_cuda_build() {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q4_K, &[5120, 6144]);
    let x = ctx.new_tensor(types::F32, &[5120, 8]);
    assert!(fits(w, x));
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
    }
}

#[test]
fn kernel_index_covers_the_table_once() {
    let mut seen = [false; 64];
    for k in [4096, 5120] {
        for n in 1..=8 {
            for warps in [8, 16] {
                for rows in [1, 2] {
                    let i = kernel_index(k, n, Shape { warps, rows }).unwrap();
                    assert!(!seen[i], "k {k} n {n} {warps}x{rows}");
                    seen[i] = true;
                }
            }
        }
    }
    assert!(seen.iter().all(|&s| s));
    assert_eq!(kernel_index(2560, 1, default_shape(1)), None);
    assert_eq!(kernel_index(5120, 9, default_shape(8)), None);
    assert_eq!(kernel_index(5120, 1, Shape { warps: 4, rows: 1 }), None);
    assert_eq!(kernel_index(5120, 1, Shape { warps: 8, rows: 4 }), None);
}

#[test]
fn default_shapes_fit_the_gb10_shared_memory() {
    // A GB10 SM's opt-in shared memory a CTA: 99 KiB.
    let limit = 99 * 1024;
    for k in [4096, 5120] {
        for n in 1..=8 {
            assert!(smem_bytes(k, n, default_shape(n)) <= limit, "k {k} n {n}");
        }
    }
    // 8 columns of 5,120 values: 46,080 bytes of input, 2,880 of weights a warp row.
    assert_eq!(smem_bytes(5120, 8, Shape { warps: 16, rows: 1 }), 92160);
    assert!(smem_bytes(5120, 8, Shape { warps: 16, rows: 2 }) > limit);
}

#[test]
fn force_shape_takes_the_kernels_shapes_only() {
    assert!(force_shape(Some(Shape { warps: 16, rows: 2 })));
    assert_eq!(shape(3), Shape { warps: 16, rows: 2 });
    assert!(!force_shape(Some(Shape { warps: 4, rows: 1 })));
    assert!(force_shape(None));
    assert_eq!(shape(3), default_shape(3));
}

#[test]
fn draft_mode_picks_the_matmuls() {
    assert!(DraftMode::All.on(true) && DraftMode::All.on(false));
    assert!(DraftMode::Head.on(true) && !DraftMode::Head.on(false));
    assert!(!DraftMode::Layers.on(true) && DraftMode::Layers.on(false));
    assert!(!DraftMode::Off.on(true) && !DraftMode::Off.on(false));
}

/// `mul_mat` on the CPU, 1 and 5 columns of both row lengths, against the product of the
/// dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let rows = 24usize;
    for (k, n) in [(4096usize, 1usize), (5120, 5)] {
        let f: Vec<f32> = noise(k * rows, 11).iter().map(|v| 0.05 * v).collect();
        let wb = crate::ggml::quantize(types::Q4_K, &f, k);
        let x = noise(k * n, 12);

        let ctx = Context::new(16, 16);
        let w = ctx
            .new_tensor(types::Q4_K, &[k as i64, rows as i64])
            .as_input();
        let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
        let y = mul_mat(&ctx, w, xt).as_output();
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
        assert_eq!(got.len(), rows * n);

        let row_bytes = k / QK_K * Q4_K_BYTES;
        let mut worst = 0f64;
        for j in 0..n {
            let xq: Vec<_> = x[j * k..(j + 1) * k]
                .chunks(32)
                .map(quantize_block)
                .collect();
            for r in 0..rows {
                let wd = crate::ggml::dequantize(
                    types::Q4_K,
                    &wb[r * row_bytes..(r + 1) * row_bytes],
                    k,
                );
                let (mut want, mut mag) = (0f64, 0f64);
                for v in 0..k {
                    let (dx, qx) = &xq[v / 32];
                    let t = f64::from(wd[v]) * f64::from(*dx) * f64::from(qx[v % 32]);
                    want += t;
                    mag += t.abs();
                }
                worst = worst.max((f64::from(got[j * rows + r]) - want).abs() / mag);
            }
        }
        assert!(worst < 1e-5, "k {k} {n} columns: rel {worst:e}");
    }
}
