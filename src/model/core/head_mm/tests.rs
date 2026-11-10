//! The head op's shape rules, its table and its CPU implementation. The GPU bits against
//! ggml's `mul_mat`, soft-cap and argmax: `legion head-mm-bench --check` on a GB10
//! ([`super::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;

#[test]
fn fits_takes_bf16_heads_of_49_to_128_columns() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::BF16, &[5376, 262144]);
    let col = |n: i64| ctx.new_tensor(types::F32, &[5376, n]);
    for n in MIN_COLS..=MAX_COLS {
        assert!(fits(w, col(n)), "{n} columns");
    }
    // The dense ops take up to 48; more than 16 sequences of 8 are not verify steps here.
    assert!(!fits(w, col(48)));
    assert!(!fits(w, col(129)));
    // Other weight types, an F16 input, row lengths and counts the tiles do not divide.
    assert!(!fits(ctx.new_tensor(types::F16, &[5376, 262144]), col(64)));
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[5376, 262144]), col(64)));
    assert!(!fits(w, ctx.new_tensor(types::F16, &[5376, 64])));
    let w2 = ctx.new_tensor(types::BF16, &[5390, 4096]);
    assert!(!fits(w2, ctx.new_tensor(types::F32, &[5390, 64])));
    let w3 = ctx.new_tensor(types::BF16, &[5376, 4096 + 64]);
    assert!(!fits(w3, col(64)));
    // Too few rows for argmax_f32's 1,024 threads, too many for the partials.
    let w4 = ctx.new_tensor(types::BF16, &[5376, 768]);
    assert!(!fits(w4, col(64)));
    let w5 = ctx.new_tensor(types::BF16, &[256, ROWS * (MAX_PARTS + 1)]);
    assert!(!fits(w5, ctx.new_tensor(types::F32, &[256, 64])));
    // A non-contiguous input.
    let x = col(128);
    assert!(!fits(w, ctx.view_2d(x, [5376, 64], 2 * x.nb()[1], 0)));
}

#[test]
fn bitwise_lists_gemma_4_31b_at_every_verify_size() {
    for n in MIN_COLS..=MAX_COLS {
        assert!(bitwise(5376, 262144, n), "{n} columns");
    }
    assert!(!bitwise(5376, 262144, 48));
    assert!(!bitwise(5376, 262144, 129));
    assert!(!bitwise(2560, 248320, 64));
}

#[test]
fn takes_needs_a_cuda_build() {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::BF16, &[5376, 262144]);
    let x = ctx.new_tensor(types::F32, &[5376, 128]);
    assert!(fits(w, x));
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
        assert!(!device_supported());
    }
}

#[test]
fn bf16_round_is_round_to_nearest_even() {
    assert_eq!(bf16_round(1.0), 1.0);
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: to even, 1.
    assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0);
    // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even, 1 + 2^-6.
    assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0);
    // Above halfway rounds up.
    assert_eq!(
        bf16_round(1.0 + 1.0 / 256.0 + 1.0 / 4096.0),
        1.0 + 1.0 / 128.0
    );
    assert_eq!(bf16_round(-2.5), -2.5);
    assert!(bf16_round(f32::NAN).is_nan());
    assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
}

/// BF16 bits of `v` rounded to nearest.
fn bf16_bits(v: f32) -> u16 {
    (bf16_round(v).to_bits() >> 16) as u16
}

#[test]
fn cpu_head_is_the_reference_and_its_argmax() {
    let (k, m, n) = (64usize, 1024usize, MIN_COLS as usize);
    let ctx = Context::new(16, 16);
    let w = ctx
        .new_tensor(types::BF16, &[k as i64, m as i64])
        .as_input();
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let (logits, argmax) = head(&ctx, w, x, 30.0);
    let (logits, argmax) = (logits.as_output(), argmax.as_output());
    let mut graph = ctx.new_graph(16);
    graph.expand(logits);
    graph.expand(argmax);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let mut wv: Vec<u16> = noise(k * m, 7)
        .iter()
        .map(|&v| bf16_bits(0.3 * v))
        .collect();
    // Row 700 repeats row 3: a tie, which argmax_f32 breaks by its threads' order.
    let row3 = wv[3 * k..4 * k].to_vec();
    wv[700 * k..701 * k].copy_from_slice(&row3);
    let xv: Vec<f32> = noise(k * n, 9).iter().map(|v| 4.0 * v).collect();
    w.set_bytes(&wv.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());
    x.set_f32(&xv);
    backend.compute(&graph).unwrap();
    let l = logits.to_f32();
    let a = argmax.to_i32();
    assert_eq!(l.len(), m * n);
    for c in 0..n {
        let col = &l[c * m..(c + 1) * m];
        for r in [0, 3, 511, 700, m - 1] {
            let want = cpu_logit(
                &wv[r * k..(r + 1) * k],
                &xv[c * k..(c + 1) * k],
                1.0 / 30.0,
                30.0,
            );
            assert_eq!(col[r].to_bits(), want.to_bits(), "column {c} row {r}");
            assert!(col[r].abs() <= 30.0);
        }
        assert_eq!(
            a[c],
            crate::model::cluster_select::argmax_row(col),
            "column {c}"
        );
    }
}

#[test]
fn no_cap_leaves_the_sums() {
    let w: Vec<u16> = [1.0f32, -2.0, 0.5, 4.0]
        .iter()
        .map(|&v| bf16_bits(v))
        .collect();
    let x = [1.0f32, 1.0, 2.0, 0.25];
    assert_eq!(cpu_logit(&w, &x, 1.0, 0.0), 1.0);
    let capped = cpu_logit(&w, &x, 0.5, 2.0);
    assert_eq!(capped, (0.5f32).tanh() * 2.0);
}
