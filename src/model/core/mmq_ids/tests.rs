//! The op's shape checks, scratch layout and CPU reference (the GPU bits against ggml's MMQ:
//! `legion mmq-ids-bench` on the GPU box).

use super::*;
use crate::ggml::{Backend, ComputeAllocator, quantize};

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
fn fits_shape_takes_the_expert_grid_steps() {
    let ctx = Context::new(64, 0);
    let gate = ctx.new_tensor(types::Q4_K, &[2560, 640, 512]);
    let down = ctx.new_tensor(types::Q5_1, &[640, 2560, 512]);
    let q8 = ctx.new_tensor(types::Q8_0, &[2560, 640, 512]);
    let x = |k, n_x, n| ctx.new_tensor(types::F32, &[k, n_x, n]);
    let ids = |n| ctx.new_tensor(types::I32, &[10, n]);
    for n in [9, 16, 48, 63] {
        assert!(fits_shape(gate, x(2560, 1, n), ids(n)), "gate at {n}");
        assert!(fits_shape(down, x(640, 10, n), ids(n)), "down at {n}");
    }
    // MMVQ's steps, the fused op's, another type, a column count that is neither 1 nor n_used.
    assert!(!fits_shape(gate, x(2560, 1, 8), ids(8)));
    assert!(!fits_shape(gate, x(2560, 1, 64), ids(64)));
    assert!(!fits_shape(q8, x(2560, 1, 16), ids(16)));
    assert!(!fits_shape(down, x(640, 3, 16), ids(16)));
    // On no GPU (the default until the engine sets one) nothing is used.
    assert!(!usable(gate, x(2560, 1, 16), ids(16)));
}

/// The node has `mul_mat_id`'s shape and sources and nothing else (a scratch tensor in the graph
/// moved the graph's layout, and with it ggml-cuda's fusions, #814).
#[test]
fn node_has_mul_mat_ids_sources_only() {
    let ctx = Context::new(64, 0);
    let down = ctx.new_tensor(types::Q5_1, &[640, 2560, 512]);
    let x = ctx.new_tensor(types::F32, &[640, 10, 16]);
    let ids = ctx.new_tensor(types::I32, &[10, 16]);
    let want = ctx.mul_mat_id(down, x, ids);
    let out = mul_mat_id(&ctx, down, x, ids);
    assert_eq!(out.ne(), want.ne());
    assert_eq!(out.ty(), want.ty());
    // SAFETY: live nodes of `ctx`.
    let (o, m) = unsafe { (&*out.raw(), &*want.raw()) };
    assert_eq!(o.src[..3], m.src[..3]);
    assert!(o.src[3..].iter().all(|s| s.is_null()), "no other source");
}

/// A step whose activations and plan would not fit the per-stream scratch stays on ggml.
#[test]
fn fits_shape_refuses_a_scratch_past_the_buffer() {
    let ctx = Context::new(16, 0);
    let wide = ctx.new_tensor(types::Q5_1, &[40960, 128, 64]);
    let x = |n| ctx.new_tensor(types::F32, &[40960, 10, n]);
    let ids = |n| ctx.new_tensor(types::I32, &[10, n]);
    assert!(scratch_layout(40960, 10, 10, 9).2 <= SCRATCH_BYTES);
    assert!(fits_shape(wide, x(9), ids(9)));
    assert!(scratch_layout(40960, 10, 10, 16).2 > SCRATCH_BYTES);
    assert!(!fits_shape(wide, x(16), ids(16)));
}

#[test]
fn scratch_layout_is_aligned_and_holds_the_plan() {
    let (ds, plan, total) = scratch_layout(640, 10, 10, 63);
    assert_eq!(ds, 640 * 630);
    assert_eq!(ds % 16, 0);
    assert_eq!(plan % 16, 0);
    assert!(plan >= ds + 630 * 20 * 4);
    assert_eq!(total, plan + (1 + 630 + 631 + 630) * 4);
    assert!(total <= SCRATCH_BYTES);
    let (ds, plan, _) = scratch_layout(2560, 1, 10, 9);
    assert_eq!(ds, 2560 * 9);
    assert_eq!(plan, (ds + 9 * 80 * 4).next_multiple_of(16));
}

/// The GB10 takes MMQ's tensor-core chain, a V100 its dp4a chain (#814), other GPUs neither
/// (a Turing or Ampere GPU's MMQ chain is not the one the kernel computes).
#[test]
fn chain_follows_the_compute_capability() {
    assert_eq!(chain_for(Some((12, 1))), CHAIN_MMA);
    assert_eq!(chain_for(Some((7, 0))), CHAIN_DP4A);
    for cc in [None, Some((7, 5)), Some((8, 0)), Some((9, 0)), Some((6, 1))] {
        assert_eq!(chain_for(cc), CHAIN_NONE, "{cc:?}");
    }
}

/// The op on the CPU is the dequantized product, row by row, for both activation layouts.
#[test]
fn cpu_op_is_the_dequantized_product() {
    let (k, rows, n_expert, n_used, n) = (256usize, 8usize, 6usize, 3usize, 9usize);
    for (ty, n_x) in [(types::Q4_K, 1usize), (types::Q5_1, n_used)] {
        let wf = noise(k * rows * n_expert, 3);
        let wq = quantize(ty, &wf, k);
        let xs = noise(k * n_x * n, 5);
        let ids: Vec<i32> = (0..n * n_used)
            .map(|i| ((i * 7 + i / n_used) % n_expert) as i32)
            .collect();
        let mut be = Backend::cpu().unwrap();
        let wctx = Context::new(1, 0);
        let w = wctx.new_tensor(ty, &[k as i64, rows as i64, n_expert as i64]);
        wctx.alloc_tensors(be.buffer_type()).unwrap();
        w.set_bytes(&wq);
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n_x as i64, n as i64]);
        let it = ctx.new_tensor(types::I32, &[n_used as i64, n as i64]);
        ctx.alloc_tensors(be.buffer_type()).unwrap();
        x.set_f32(&xs);
        it.set_i32(&ids);
        let out = mul_mat_id(&ctx, w, x, it);
        let mut g = ctx.new_graph(16);
        g.expand(out);
        ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
            .unwrap();
        be.compute(&g).unwrap();
        let got = out.to_f32();
        let row_bytes = crate::ggml::row_size(ty, k);
        for t in 0..n {
            for j in 0..n_used {
                let e = ids[t * n_used + j] as usize;
                let col = &xs[(t * n_x + if n_x == 1 { 0 } else { j }) * k..][..k];
                for r in 0..rows {
                    let at = (e * rows + r) * row_bytes;
                    let wr = crate::ggml::dequantize(ty, &wq[at..at + row_bytes], k);
                    let want: f32 = wr.iter().zip(col).map(|(a, b)| a * b).sum();
                    let v = got[(t * n_used + j) * rows + r];
                    assert_eq!(v.to_bits(), want.to_bits(), "{ty} t {t} j {j} r {r}");
                }
            }
        }
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    // The GB10 and the 5090 MMQ's tensor-core chain, the V100 its dp4a chain, the 4090, the
    // A100 and an unknown device neither, whatever the engine's GPU is.
    let want = [
        CHAIN_MMA, CHAIN_NONE, CHAIN_DP4A, CHAIN_NONE, CHAIN_MMA, CHAIN_NONE,
    ];
    for engine in [CHAIN_NONE, CHAIN_MMA, CHAIN_DP4A] {
        let picks: Vec<u8> = (0..6)
            .map(|d| chain_on_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(picks, want, "engine {engine}");
        assert_eq!(chain_on_by(None, engine, test_cc), engine);
    }
}
