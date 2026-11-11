//! The op's shape checks, kernel table and CPU reference (the GPU bits against ggml's MMVQ:
//! `tests/nvfp4_mvq.rs` and `legion nvfp4-mvq-bench` on the GPU box).

use super::bench::{activations, nvfp4_bytes};
use super::*;
use crate::ggml::{Backend, ComputeAllocator};

#[test]
fn fits_takes_the_dense_mmvq_steps() {
    let ctx = Context::new(64, 0);
    let gate = ctx.new_tensor(types::NVFP4, &[5120, 17408]);
    let down = ctx.new_tensor(types::NVFP4, &[17408, 5120]);
    let stack = ctx.new_tensor(types::NVFP4, &[5120, 1024, 4]);
    let odd = ctx.new_tensor(types::NVFP4, &[5120, 1002]);
    let q8 = ctx.new_tensor(types::Q8_0, &[5120, 17408]);
    let x = |k, n| ctx.new_tensor(types::F32, &[k, n]);
    let s = ctx.new_tensor(types::F32, &[1]);
    let rows = ctx.new_tensor(types::F32, &[17408]);
    for n in 1..=MAX_N {
        assert!(fits(gate, x(5120, n), None), "gate at {n}");
        assert!(fits(down, x(17408, n), Some(s)), "down at {n}");
    }
    // ggml's MMQ steps, a 3D activation, an expert stack, rows not in whole CTAs, rows not of
    // whole 256 values, another type, a per-row scale.
    assert!(!fits(gate, x(5120, 9), None));
    assert!(!fits(gate, ctx.new_tensor(types::F32, &[5120, 1, 4]), None));
    assert!(!fits(stack, x(5120, 4), None));
    assert!(!fits(odd, x(5120, 4), None));
    let short = ctx.new_tensor(types::NVFP4, &[5120 + 64, 1024]);
    assert!(!fits(short, x(5120 + 64, 4), None));
    assert!(!fits(q8, x(5120, 4), None));
    assert!(!fits(gate, x(5120, 4), Some(rows)));
    // On no GPU (the default until the engine sets one) nothing is taken.
    assert!(!takes(gate, x(5120, 4), None));
}

#[test]
fn compute_capability_12_takes_the_op() {
    assert!(gpu_takes(Some((12, 1))));
    assert!(gpu_takes(Some((12, 0))));
    for cc in [(7, 0), (8, 0), (8, 9), (9, 0), (10, 0), (11, 0)] {
        assert!(!gpu_takes(Some(cc)), "{cc:?}");
    }
    assert!(!gpu_takes(None));
}

#[test]
fn kernels_follow_ggmls_warps() {
    // The table: the quantization, then ggml's 4 warps (1-4 columns), then its 2 (5-8).
    assert_eq!(
        (1..=8).map(kernel_index).collect::<Vec<_>>(),
        [1, 1, 1, 1, 2, 2, 2, 2]
    );
    const { assert!(REGION as i64 % ROW_MULTIPLE == 0) };
    const { assert!(MAX_N == crate::model::core::linear::NVFP4_GGML_MAX_N) };
}

#[test]
fn plans_fit_the_shared_memory() {
    // Qwen3.8-27B's MLP on a 12.x GPU: whole rows at k 5120, quarter rows at 17408.
    assert_eq!(
        plan(5120, 17408, SMEM_12X),
        Some(Plan {
            chunks: 1,
            ns: 28,
            smem: 89504
        })
    );
    assert_eq!(
        plan(17408, 5120, SMEM_12X),
        Some(Plan {
            chunks: 4,
            ns: 28,
            smem: 77408
        })
    );
    for (k, rows) in [(256, 4), (5120, 17408), (17408, 5120), (32768, 1024)] {
        let p = plan(k, rows, SMEM_12X).unwrap();
        assert!(p.smem <= SMEM_12X && p.ns > REGION, "{k} {rows}");
    }
    // Rows not in whole groups; pieces not of whole 16-byte units; pieces over two 64-block
    // steps.
    assert_eq!(plan(5120, 1030, SMEM_12X), None);
    assert_eq!(Plan::with(17408, 8, 20, SMEM_12X), None);
    assert_eq!(Plan::with(17408, 2, 18, SMEM_12X), None);
    assert_eq!(
        plan(MAX_K as usize, 1024, SMEM_12X).map(|p| p.chunks),
        Some(16)
    );
}

/// The op on the CPU is the dequantized product, column by column, times the scale.
#[test]
fn cpu_op_is_the_dequantized_product() {
    let (k, rows, n) = (256usize, 16usize, 3usize);
    let row_bytes = crate::ggml::row_size(types::NVFP4, k);
    let wq = nvfp4_bytes(row_bytes * rows, 3, false);
    let xs = activations(k, n, 5);
    for scaled in [false, true] {
        let mut be = Backend::cpu().unwrap();
        let wctx = Context::new(2, 0);
        let w = wctx.new_tensor(types::NVFP4, &[k as i64, rows as i64]);
        let s = wctx.new_tensor(types::F32, &[1]);
        wctx.alloc_tensors(be.buffer_type()).unwrap();
        w.set_bytes(&wq);
        s.set_f32(&[0.25]);
        let ctx = Context::new(8, 16);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]);
        ctx.alloc_tensors(be.buffer_type()).unwrap();
        x.set_f32(&xs);
        let out = mul_mat(&ctx, w, x, scaled.then_some(s));
        assert_eq!(out.ne(), [rows as i64, n as i64, 1, 1]);
        let mut g = ctx.new_graph(16);
        g.expand(out);
        ctx.alloc_graph(&g, ComputeAllocator::new(be.buffer_type()))
            .unwrap();
        be.compute(&g).unwrap();
        let got = out.to_f32();
        for j in 0..n {
            let col = &xs[j * k..][..k];
            for r in 0..rows {
                let wr =
                    crate::ggml::dequantize(types::NVFP4, &wq[r * row_bytes..][..row_bytes], k);
                let v: f32 = wr.iter().zip(col).map(|(a, b)| a * b).sum();
                let want = if scaled { v * 0.25 } else { v };
                assert_eq!(
                    got[j * rows + r].to_bits(),
                    want.to_bits(),
                    "scaled {scaled} j {j} r {r}"
                );
            }
        }
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    // The GB10 and the 5090 (12.x) take the op; the others keep ggml's MMVQ, whatever the
    // engine's GPU is.
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, false, false, false, true, false],
            "engine {engine}"
        );
        assert_eq!(gpu_for_by(None, engine, test_cc), engine);
    }
}
