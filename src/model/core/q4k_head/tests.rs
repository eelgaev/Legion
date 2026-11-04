//! The Q4_K draft head op's shape rules, ggml's warp count it follows and its CPU implementation.
//! The GPU bits against ggml's `mul_mat`: `legion q4k-head-bench --check` on a GB10 (the pick's
//! against the split argmax, soft_max and get_rows: `--pick --check`) ([`super::bench`]).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;

#[test]
fn fits_takes_q4_k_heads_of_one_to_eight_columns() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::Q4_K, &[2560, 65536]);
    let col = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in 1..=MAX_COLS {
        assert!(fits(w, col(n)), "{n} columns");
    }
    // ggml runs MMQ from 9 columns.
    assert!(!fits(w, col(9)));
    assert!(!fits(w, col(16)));
    // Row lengths the kernel is not built for, another weight type, an F16 input.
    for k in [2048, 4096] {
        let w = ctx.new_tensor(types::Q4_K, &[k, 4096]);
        assert!(!fits(w, ctx.new_tensor(types::F32, &[k, 1])), "k {k}");
    }
    assert!(!fits(ctx.new_tensor(types::Q8_0, &[2560, 4096]), col(1)));
    assert!(!fits(ctx.new_tensor(types::Q4_0, &[2560, 4096]), col(1)));
    assert!(!fits(w, ctx.new_tensor(types::F16, &[2560, 1])));
    // The first rows of a whole Q4_K head (a view) fit.
    let whole = ctx.new_tensor(types::Q4_K, &[2560, 248320]);
    assert!(fits(
        ctx.view_2d(whole, [2560, 65536], whole.nb()[1], 0),
        col(1)
    ));
}

#[test]
fn takes_needs_a_cuda_build() {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q4_K, &[2560, 65536]);
    let x = ctx.new_tensor(types::F32, &[2560, 1]);
    assert!(fits(w, x));
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x));
        assert!(!device_supported());
    }
}

#[test]
fn ggml_warps_follow_the_gb10_table() {
    // 2,560 values (10 blocks), one column: the K loop of 4 warps takes 2 trips, of 8 one with
    // nothing idle, so ggml widens the CTA.
    assert_eq!(ggml_warps(1, 10), 8);
    // 4,096 (16 blocks): 2 trips -> 1, none idle.
    assert_eq!(ggml_warps(1, 16), 8);
    // 1,024 (4 blocks): the small-K launch keeps 4.
    assert_eq!(ggml_warps(1, 4), 4);
    // 3,072 (12 blocks): 2 trips -> 1, none idle in ggml's count (trips, not blocks).
    assert_eq!(ggml_warps(1, 12), 8);
    // 5,120 (20 blocks): 3 trips -> 2, one of 4 half-trips idle (more than an eighth): kept.
    assert_eq!(ggml_warps(1, 20), 4);
    // 18,432 (72 blocks): 9 trips -> 5, one of 10 idle: widened.
    assert_eq!(ggml_warps(1, 72), 8);
    for n in 2..=4 {
        assert_eq!(ggml_warps(n, 10), 4, "{n} columns");
    }
    for n in 5..=8 {
        assert_eq!(ggml_warps(n, 10), 2, "{n} columns");
    }
}

#[test]
fn quantize_block_is_ggml_quantize_q8_1() {
    let (d, q) = quantize_block(&[0.0; GROUP]);
    assert_eq!(d, 0.0);
    assert_eq!(q, [0i8; GROUP]);
    let mut x = [0f32; GROUP];
    x[3] = -2.54;
    x[4] = 0.061;
    x[5] = 1.3;
    let (d, q) = quantize_block(&x);
    assert!((d - 0.02).abs() < 1e-4, "{d}");
    assert_eq!(q[3], -127);
    assert_eq!(q[4], 3);
    assert_eq!(q[5], 65);
}

/// The dot products of every `iqs` of a block add up to the block's dequantized product.
#[test]
fn vec_dot_covers_the_block() {
    let k = QK_K;
    let f: Vec<f32> = noise(k, 21).iter().map(|v| 0.05 * v).collect();
    let blk = crate::ggml::quantize(types::Q4_K, &f, k);
    let wd = crate::ggml::dequantize(types::Q4_K, &blk, k);
    let x = noise(k, 22);
    let xq: Vec<_> = x.chunks(GROUP).map(quantize_block).collect();
    let got: f64 = (0..16).map(|t| f64::from(vec_dot(&blk, &xq, 2 * t))).sum();
    let (mut want, mut mag) = (0f64, 0f64);
    for v in 0..k {
        let (dx, qx) = &xq[v / GROUP];
        let t = f64::from(wd[v]) * f64::from(*dx) * f64::from(qx[v % GROUP]);
        want += t;
        mag += t.abs();
    }
    assert!((got - want).abs() / mag < 1e-5, "{got} vs {want}");
}

/// `mul_mat` on the CPU, 1 and 3 columns, against the product of the dequantized operands in
/// f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, rows) = (2560usize, 48usize);
    for n in [1usize, 3] {
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
                .chunks(GROUP)
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
                    let (dx, qx) = &xq[v / GROUP];
                    let t = f64::from(wd[v]) * f64::from(*dx) * f64::from(qx[v % GROUP]);
                    want += t;
                    mag += t.abs();
                }
                worst = worst.max((f64::from(got[j * rows + r]) - want).abs() / mag);
            }
        }
        assert!(worst < 1e-5, "{n} columns: rel {worst:e}");
    }
}

/// Every block of a row reaches the sum at any warp count (8 warps a CTA, one column: lane l of
/// warp w takes block 2 w + l / 16 alone, and warps 5-7 none; 4 warps: blocks 8 and 9 go to
/// warp 0's second trip).
#[test]
fn row_dot_follows_ggml_threads() {
    let nb = 10;
    let mut row = vec![0u8; nb * Q4_K_BYTES];
    // Blocks 3 (virtual threads 16-31 of warp 1) and 9 only: d 1, dmin 0, scale 1 for every
    // group, nibbles 2.
    // SAFETY: plain conversions.
    let one = unsafe { ffi::ggml_fp32_to_fp16(1.0) };
    for blk in [3, 9] {
        let b = &mut row[blk * Q4_K_BYTES..][..Q4_K_BYTES];
        b[0..2].copy_from_slice(&one.to_le_bytes());
        b[4..8].fill(1);
        b[12..16].fill(1);
        b[16..].fill(0x22);
    }
    let xq = vec![(0.5f32, [3i8; GROUP]); nb * QK_K / GROUP];
    // 2 x 256 values of 2 * 3 * 0.5.
    for g in [2, 4, 8] {
        assert_eq!(row_dot(&row, &xq, g), 512.0 * 3.0, "{g} warps");
    }
}

#[test]
fn force_variant_needs_cuda() {
    assert_eq!(force_variant(None), cfg!(feature = "cuda"));
    assert_eq!(force_variant(Some((8, 0, 2))), cfg!(feature = "cuda"));
    assert!(!force_variant(Some((6, 0, 1))));
    assert!(!force_variant(Some((4, 0, 3))));
    force_variant(None);
}

/// `head` on the CPU: the op's logits, their argmax_f32 pick and the pick's probability as
/// ggml's soft_max + get_rows give it (within F32 rounding of the sum).
#[test]
fn cpu_head_is_mul_mat_argmax_and_softmax() {
    // More rows than one argmax_f32 thread's stride, so ties across its threads can occur.
    let (k, rows) = (2560usize, 3000usize);
    let f: Vec<f32> = noise(k * rows, 31).iter().map(|v| 0.05 * v).collect();
    let wb = crate::ggml::quantize(types::Q4_K, &f, k);
    let x = noise(k, 32);

    let ctx = Context::new(32, 32);
    let w = ctx
        .new_tensor(types::Q4_K, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, 1]).as_input();
    let (logits, argmax, pmax) = head(&ctx, w, xt);
    let (logits, argmax, pmax) = (logits.as_output(), argmax.as_output(), pmax.as_output());
    let plain = mul_mat(&ctx, w, xt).as_output();
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
    assert_eq!(a, vec![crate::model::cluster_select::argmax_row(&l)]);
    let p = pmax.to_f32()[0];
    let want = probs.to_f32()[a[0] as usize];
    // ggml's CPU soft_max sums in double: off by F32 rounding of 3,000 terms at most.
    assert!((p - want).abs() <= 1e-4 * want, "{p} vs {want}");
}

#[test]
fn pick_takes_up_to_seven_columns_on_a_cuda_build() {
    let ctx = Context::new(8, 0);
    let w = ctx.new_tensor(types::Q4_K, &[2560, 65536]);
    let w32k = ctx.new_tensor(types::Q4_K, &[2560, 32768]);
    let x1 = ctx.new_tensor(types::F32, &[2560, 1]);
    let x4 = ctx.new_tensor(types::F32, &[2560, 4]);
    let x8 = ctx.new_tensor(types::F32, &[2560, 8]);
    // Eight columns of 65,536 logits (or four of 32,768): ggml's soft_max runs a CTA a row, so
    // the split argmax, soft_max and get_rows stay after the op.
    assert!(!pick_takes(w, x8));
    assert!(!pick_takes(w32k, x4));
    if !cfg!(feature = "cuda") {
        assert!(!pick_takes(w, x1));
        assert!(!pick_takes(w, x4));
    }
    let one = pick_takes(w, x1);
    set_pick_cols(false);
    assert!(!pick_takes(w, x4));
    assert_eq!(pick_takes(w, x1), one, "one column keeps the pick");
    set_pick_cols(true);
    set_pick(false);
    assert!(!pick_takes(w, x1));
    assert!(!pick_takes(w, x4));
    set_pick(true);
}

/// ggml-cuda's `soft_max_f32_cuda` choice for rows too long for shared memory: the cooperative
/// kernel while `ncols / nrows > 8192` (integer division).
#[test]
fn coop_follows_ggml_soft_max() {
    use crate::model::core::q8_head::pick::coop;
    assert!(coop(65536, 1));
    assert!(coop(65536, 7));
    assert!(!coop(65536, 8));
    assert!(coop(32768, 3));
    assert!(!coop(32768, 4));
    assert!(coop(248320, 7));
    assert!(!coop(248320, 8), "more columns than the pick takes");
    assert!(!coop(65536, 0));
}

/// `head` at several columns on the CPU: every column's logits, argmax_f32 pick and probability.
#[test]
fn cpu_head_of_columns_is_mul_mat_argmax_and_softmax() {
    let (k, rows, n) = (2560usize, 3000usize, 3usize);
    let f: Vec<f32> = noise(k * rows, 41).iter().map(|v| 0.05 * v).collect();
    let wb = crate::ggml::quantize(types::Q4_K, &f, k);
    let x = noise(k * n, 42);

    let ctx = Context::new(32, 32);
    let w = ctx
        .new_tensor(types::Q4_K, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let (logits, argmax, pmax) = head(&ctx, w, xt);
    let (logits, argmax, pmax) = (logits.as_output(), argmax.as_output(), pmax.as_output());
    assert_eq!(argmax.ne(), [n as i64, 1, 1, 1]);
    assert_eq!(pmax.ne(), [n as i64, 1, 1, 1]);
    let plain = mul_mat(&ctx, w, xt).as_output();
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

    let (l, a, p, pr) = (
        logits.to_f32(),
        argmax.to_i32(),
        pmax.to_f32(),
        probs.to_f32(),
    );
    assert_eq!(
        l.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        plain
            .to_f32()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    );
    for j in 0..n {
        let col = &l[j * rows..][..rows];
        assert_eq!(
            a[j],
            crate::model::cluster_select::argmax_row(col),
            "col {j}"
        );
        let want = pr[j * rows + a[j] as usize];
        assert!(
            (p[j] - want).abs() <= 1e-4 * want,
            "col {j}: {} vs {want}",
            p[j]
        );
    }
    // The columns differ, so a pick read from the wrong column would show.
    assert!(a.iter().any(|&v| v != a[0]) || p.iter().any(|&v| v != p[0]));
}
