//! The small-M op's shape rules, ggml's MMVQ bounds, the Q8_1 reference quantization and the
//! CPU implementation. The GPU bits against ggml's `mul_mat`: `legion q-small-m-bench --check`
//! on a V100 ([`super::bench`]).

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

#[test]
fn mmvq_bounds_follow_ggml_on_volta() {
    assert_eq!(ggml_mmvq_max(types::Q2_K), 4);
    assert_eq!(ggml_mmvq_max(types::Q4_K), 5);
    assert_eq!(ggml_mmvq_max(types::Q3_K), 6);
    assert_eq!(ggml_mmvq_max(types::Q5_K), 6);
    assert_eq!(ggml_mmvq_max(types::Q6_K), 7);
    for ty in [types::IQ4_XS, types::IQ4_NL, types::Q8_0] {
        assert_eq!(ggml_mmvq_max(ty), MAX_COLS);
    }
    // The default keeps ggml's bound; wide takes up to 8 columns.
    assert_eq!(max_cols(types::Q4_K), 5);
    set_wide(true);
    assert_eq!(max_cols(types::Q4_K), MAX_COLS);
    set_wide(false);
}

#[test]
fn fits_takes_the_types_at_2_to_8_columns() {
    let ctx = Context::new(64, 0);
    let col = |k: i64, n: i64| ctx.new_tensor(types::F32, &[k, n]);
    for ty in TYPES {
        let w = ctx.new_tensor(ty, &[5120, 17408]);
        for n in MIN_COLS..=MAX_COLS {
            assert!(fits(w, col(5120, n)), "{} {n}", crate::ggml::type_name(ty));
        }
        assert!(!fits(w, col(5120, 1)));
        assert!(!fits(w, col(5120, 9)));
        assert!(!fits(w, col(6144, 4)));
    }
    // Another weight type, a 3D weight, a strided input, a non-F32 input.
    assert!(!fits(
        ctx.new_tensor(types::Q4_0, &[5120, 64]),
        col(5120, 4)
    ));
    assert!(!fits(ctx.new_tensor(types::F32, &[5120, 64]), col(5120, 4)));
    assert!(!fits(
        ctx.new_tensor(types::Q4_K, &[5120, 64, 2]),
        col(5120, 4)
    ));
    let w = ctx.new_tensor(types::Q4_K, &[5120, 64]);
    let wide = col(10240, 4);
    assert!(!fits(w, ctx.view_2d(wide, [5120, 4], 10240 * 4, 0)));
    assert!(!fits(w, ctx.new_tensor(types::F16, &[5120, 4])));
}

#[test]
fn off_without_a_v100() {
    // The engine sets the GPU on load; the tests never do (and run without CUDA here).
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q4_K, &[5120, 256]);
    let x = ctx.new_tensor(types::F32, &[5120, 4]);
    assert!(fits(w, x));
    assert!(!takes(w, x));
}

#[test]
fn variant_choice() {
    assert!(force_variant(None));
    assert_eq!(variant(types::IQ4_XS, 2), 2);
    assert_eq!(variant(types::IQ4_XS, 3), 2);
    assert_eq!(variant(types::IQ4_XS, 5), -4);
    assert_eq!(variant(types::IQ4_XS, 8), 4);
    assert_eq!(variant(types::Q4_K, 2), -4);
    assert_eq!(variant(types::Q6_K, 7), -4);
    assert_eq!(variant(types::Q6_K, 8), 2);
    for r in VARIANTS {
        assert!(force_variant(Some(r)));
        assert_eq!(variant(types::Q4_K, 5), r);
    }
    assert!(!force_variant(Some(3)));
    assert!(!force_variant(Some(8)));
    assert!(force_variant(None));
}

/// ggml's quantize_q8_1: d = amax / 127 and the sum as F16, q = round(x / d), an all-zero block
/// all zeros.
#[test]
fn quantize_ref_is_ggml_q8_1() {
    let (k, n) = (64usize, 2usize);
    let mut xs = noise(k * n, 3);
    xs[k..k + 32].fill(0.0);
    xs[5] = -2.5;
    let mut out = vec![0u8; q8_1_bytes(k as i64, n as i64)];
    quantize_ref(&xs, k, n, &mut out);
    // SAFETY: a plain conversion.
    let half = |b: &[u8]| unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([b[0], b[1]])) };
    for (b, blk) in out.chunks(Q8_1_BYTES).enumerate() {
        let vals = &xs[b * QK8_1..(b + 1) * QK8_1];
        let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        // SAFETY: plain conversions.
        let (dh, sh) = unsafe {
            (
                ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(d)),
                ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(vals.iter().sum())),
            )
        };
        assert_eq!(half(&blk[0..2]).to_bits(), dh.to_bits(), "block {b}");
        assert_eq!(half(&blk[2..4]).to_bits(), sh.to_bits(), "block {b}");
        for (i, v) in vals.iter().enumerate() {
            let q = blk[4 + i] as i8;
            if amax == 0.0 {
                assert_eq!(q, 0);
            } else {
                assert_eq!(q, (v / d).round() as i8, "block {b} value {i}");
            }
        }
    }
    // The block with -2.5 holds -127 there.
    assert_eq!(out[4 + 5] as i8, -127);
}

/// `mul_mat` on the CPU against the product of the dequantized weight and the dequantized Q8_1
/// activations in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, rows, n) = (512usize, 24usize, 3usize);
    for ty in [types::Q4_K, types::Q6_K, types::IQ4_XS, types::Q8_0] {
        let wb = crate::ggml::quantize(ty, &noise(k * rows, 7), k);
        let x = noise(k * n, 8);
        let mut xq = vec![0u8; q8_1_bytes(k as i64, n as i64)];
        quantize_ref(&x, k, n, &mut xq);

        let ctx = Context::new(16, 16);
        let w = ctx.new_tensor(ty, &[k as i64, rows as i64]).as_input();
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

        let row_bytes = crate::ggml::row_size(ty, k);
        let mut worst = 0f64;
        for r in 0..rows {
            let wr = crate::ggml::dequantize(ty, &wb[r * row_bytes..(r + 1) * row_bytes], k);
            for j in 0..n {
                let (mut want, mut mag) = (0f64, 0f64);
                for v in 0..k {
                    let blk = &xq[(j * (k / QK8_1) + v / QK8_1) * Q8_1_BYTES..];
                    // SAFETY: a plain conversion.
                    let d = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                    let t = f64::from(wr[v]) * f64::from(blk[4 + v % QK8_1] as i8) * f64::from(d);
                    want += t;
                    mag += t.abs();
                }
                let g = f64::from(got[j * rows + r]);
                worst = worst.max((g - want).abs() / mag.max(1e-30));
            }
        }
        assert!(worst < 1e-5, "{} rel {worst:e}", crate::ggml::type_name(ty));
    }
}
