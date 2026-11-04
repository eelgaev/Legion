//! The Q8_0 matvec op's shape rules, ggml's stream-k rule, and its CPU implementation. The GPU
//! bits against ggml's `mul_mat`: `legion q8-mv-bench --check` on the GPU box ([`super::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn ggml_whole_rows_follows_mmq_stream_k() {
    // 48 SMs (GB10). The draft heads of 65,536 rows (512 tiles, 11 waves at 97%) and the whole
    // 248,320-row head (1,940 tiles, 41 waves at 98.6%) run one CTA per tile.
    assert!(ggml_whole_rows(65536, 48));
    assert!(ggml_whole_rows(248320, 48));
    // 48 tiles: one full wave.
    assert!(ggml_whole_rows(6144, 48));
    // 32 tiles fill 67% of a wave: 48 CTAs share them, partial sums added.
    assert!(!ggml_whole_rows(4096, 48));
    // 96 tiles over 48 CTAs: two whole tiles each, no fix-up.
    assert!(ggml_whole_rows(96 * 128, 48));
    // 60 tiles: 2 waves at 62%, 48 CTAs, 60 % 48 != 0.
    assert!(!ggml_whole_rows(60 * 128, 48));
    // A partial last tile counts as a tile.
    assert!(ggml_whole_rows(65536 - 32, 48));
    assert!(!ggml_whole_rows(0, 48));
    assert!(!ggml_whole_rows(65536, 0));
}

#[test]
fn fits_takes_q8_0_heads_of_9_to_16_columns() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 65536]);
    let col = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in MIN_COLS..=MAX_COLS {
        assert!(fits(w, col(n)), "{n} columns");
    }
    assert!(!fits(w, col(8)));
    assert!(!fits(w, col(17)));
    // k not a multiple of 512, rows not of 32, another weight type, a strided input.
    let x = col(16);
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[2304, 65536]), col(16)));
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[2560, 65520]), x));
    assert!(!fits(ctx.new_tensor(types::Q4_0, &[2560, 65536]), x));
    let wide = ctx.new_tensor(types::F32, &[5120, 16]);
    assert!(!fits(w, ctx.view_2d(wide, [2560, 16], 5120 * 4, 0)));
    // The first rows of a larger head (a view) fit.
    let head = ctx.new_tensor(types::Q8_0, &[2560, 248320]);
    assert!(fits(ctx.view_2d(head, [2560, 65536], head.nb()[1], 0), x));
}

#[test]
fn off_without_a_gb10() {
    // The engine sets the GPU on load; the tests never do (and run without CUDA here).
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 65536]);
    let x = ctx.new_tensor(types::F32, &[2560, 16]);
    assert!(fits(w, x));
    assert!(!takes(w, x));
}

/// `mul_mat` on the CPU (the activations quantized by `shared_q8`'s CPU path) against the
/// product of the dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, rows, n) = (1024usize, 64usize, 13usize);
    let wb = q8_0_matrix(k, rows, 7);
    let x = noise(k * n, 8);
    let mut xq = vec![0u8; mmq_preq::q8_1_bytes(k as i64, n as i64)];
    mmq_preq::quantize_ref(&x, k, n, &mut xq);

    let ctx = Context::new(16, 16);
    let w = ctx
        .new_tensor(types::Q8_0, &[k as i64, rows as i64])
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

    let mut worst = 0f64;
    for r in 0..rows {
        for j in 0..n {
            let (mut want, mut mag) = (0f64, 0f64);
            for v in 0..k {
                let blk = &wb[(r * k + v) / GROUP * Q8_0_BYTES..];
                // SAFETY: a plain conversion.
                let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                let qw = blk[2 + v % GROUP] as i8;
                let xb = &xq[((v / 128) * n + j) * mmq_preq::BLOCK_BYTES..];
                let g = (v % 128) / GROUP;
                let dx = f32::from_ne_bytes(xb[g * 4..g * 4 + 4].try_into().unwrap());
                let qx = xb[16 + v % 128] as i8;
                let t = dw as f64 * qw as f64 * dx as f64 * qx as f64;
                want += t;
                mag += t.abs();
            }
            // F32 sums of k terms: off by a few ulps of the terms' magnitude.
            let g = got[j * rows + r] as f64;
            worst = worst.max((g - want).abs() / mag);
        }
    }
    assert!(worst < 1e-5, "rel {worst:e}");
}

#[test]
fn force_variant_needs_cuda() {
    // Without CUDA no variant exists; the default is always accepted where CUDA is.
    assert_eq!(force_variant(None), cfg!(feature = "cuda"));
    assert_eq!(force_variant(Some((8, 4, 2))), cfg!(feature = "cuda"));
    assert!(!force_variant(Some((2, 2, 2))));
    force_variant(None);
}
