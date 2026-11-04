//! The draft pick's graph and CPU references. The GPU bits against the split argmax, ggml's
//! soft_max and get_rows: `legion q8-head-bench --pick --check` on the GPU box
//! ([`crate::model::core::q8_head::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::cluster_select::{argmax_rank, argmax_row};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

/// `head` on the CPU: the head's logits, their argmax_f32 pick and the pick's probability as
/// ggml's soft_max + get_rows give it (within F32 rounding of the sum).
#[test]
fn cpu_head_is_mul_mat_argmax_and_softmax() {
    // More rows than one argmax_f32 thread's stride, so ties across its threads can occur.
    let (k, rows) = (1024usize, 3000usize);
    let wb = q8_0_matrix(k, rows, 21);
    let x = noise(k, 22);

    let ctx = Context::new(32, 32);
    let w = ctx
        .new_tensor(types::Q8_0, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let (logits, argmax, pmax) = head(&ctx, w, xt);
    let (logits, argmax, pmax) = (logits.as_output(), argmax.as_output(), pmax.as_output());
    let plain = crate::model::core::q8_head::mul_mat(&ctx, w, xt).as_output();
    let probs = ctx.soft_max(plain).as_output();
    let mut graph = ctx.new_graph(32);
    for t in [logits, argmax, pmax, plain, probs] {
        graph.expand(t);
    }
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&wb);
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();

    let l = logits.to_f32();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&l), bits(&plain.to_f32()));
    let a = argmax.to_i32();
    assert_eq!(a, vec![argmax_row(&l)]);
    let p = pmax.to_f32()[0];
    let want = probs.to_f32()[a[0] as usize];
    // ggml's CPU soft_max sums in double: off by F32 rounding of 3,000 terms at most.
    assert!((p - want).abs() <= 1e-4 * want, "{p} vs {want}");
}

/// The pick follows argmax_f32's tie order, not the lowest column.
#[test]
fn ties_take_argmax_f32_order() {
    let mut x = vec![0.0f32; 4096];
    // Columns 3 and 2048 + 3 hold the maximum: thread 3 scans both, the first one wins.
    x[3] = 1.0;
    x[2048 + 3] = 1.0;
    assert_eq!(argmax_row(&x), 3);
    // Column 1 (thread 1, bitrev5 16) loses to column 2 (thread 2, bitrev5 8).
    let mut x = vec![0.0f32; 4096];
    x[1] = 1.0;
    x[2] = 1.0;
    assert!(argmax_rank(2) < argmax_rank(1));
    assert_eq!(argmax_row(&x), 2);
}

#[test]
fn probability_reference() {
    let x = [1.0f32, 2.0, 3.0, 3.0];
    let e: Vec<f32> = x.iter().map(|v| (v - 3.0f32).exp()).collect();
    let want = 1.0 / e.iter().sum::<f32>();
    assert_eq!(pick_probability(&x, 2), want);
    // No pick (every value NaN or below -f32::MAX): NaN, which no confidence floor passes.
    assert!(pick_probability(&x, -1).is_nan());
    // A NaN logit makes ggml's sum NaN, and so the probability.
    assert!(pick_probability(&[1.0, f32::NAN, 0.5], 0).is_nan());
    assert_eq!(
        argmax_row(&[f32::NAN; 2048]),
        -1,
        "argmax_f32 never takes NaN"
    );
}

#[test]
fn buffer_holds_the_sums_and_every_cta() {
    // Header (16 bytes) and ggml's 256-thread block's worth of CTA sums before the partials.
    const { assert!(16 + 4 * 256 <= PARTS_OFFSET) };
    assert_eq!(buffer_bytes(80), 2048 + 80 * 32 * 16);
    const { assert!(MAX_SMS < 256) };
    // A slice per column the pick takes; slices stay 16-byte aligned.
    assert!(SLICES as i64 >= MAX_COLS);
    assert_eq!(buffer_bytes(48) % 16, 0);
}

#[test]
fn takes_needs_cuda() {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 248320]);
    let x = ctx.new_tensor(types::F32, &[2560, 1]);
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
        assert!(!device_supported());
    }
    set_pick(false);
    assert!(!takes(w, x));
    set_pick(true);
}
