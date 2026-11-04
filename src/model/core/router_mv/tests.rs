//! `router_mv` against ggml's matmul and the fused router ops it replaces: on the CPU with
//! values whose sums are exact in any order; on CUDA (ignored tests) bit for bit.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::fused;

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

fn i32s(t: Tensor<'_>) -> Vec<i32> {
    t.bytes()
        .chunks(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

/// Outputs of the op and of ggml's matmul plus `fused::router` for `n` tokens.
struct Case {
    logits: Vec<f32>,
    ids: Vec<i32>,
    w: Vec<f32>,
    ref_logits: Vec<f32>,
    ref_ids: Vec<i32>,
    ref_w: Vec<f32>,
}

/// The op and the chain on `backend` over `k_embd`-wide rows of `n_expert` experts, top `k`.
/// `exact`: small integers in the weight and halves in `x`, so every order sums exactly.
fn case(backend: &mut Backend, k_embd: i64, n_expert: i64, k: i64, n: i64, exact: bool) -> Case {
    let ctx = Context::new(32, 64);
    let w = ctx.new_tensor(types::F32, &[k_embd, n_expert]).as_input();
    let x = ctx.new_tensor(types::F32, &[k_embd, n]).as_input();
    let (logits, ids, wts) = router_mv(&ctx, w, x, k, 5);
    let ref_logits = ctx.mul_mat(w, x);
    let (ref_ids, ref_w) = fused::router(&ctx, ref_logits, k);
    // The op's outputs are contiguous views of its node.
    let outs = [
        logits.as_output(),
        ids.as_output(),
        wts.as_output(),
        ref_logits.as_output(),
        ref_ids.as_output(),
        ref_w.as_output(),
    ];
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let (wv, xv) = (
        noise((k_embd * n_expert) as usize, 11),
        noise((k_embd * n) as usize, 12),
    );
    if exact {
        w.set_f32(&wv.iter().map(|v| (v * 8.0).round()).collect::<Vec<_>>());
        x.set_f32(
            &xv.iter()
                .map(|v| (v * 4.0).round() / 2.0)
                .collect::<Vec<_>>(),
        );
    } else {
        w.set_f32(&wv.iter().map(|v| v * 0.05).collect::<Vec<_>>());
        x.set_f32(&xv);
    }
    backend.compute(&graph).unwrap();
    Case {
        logits: outs[0].to_f32(),
        ids: i32s(outs[1]),
        w: outs[2].to_f32(),
        ref_logits: outs[3].to_f32(),
        ref_ids: i32s(outs[4]),
        ref_w: outs[5].to_f32(),
    }
}

#[test]
fn router_mv_is_the_matmul_and_the_router() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    // Integer logits tie often: the order of ties is the router's (the lower id first).
    for (k_embd, n_expert, k, n) in [(256, 512, 10, 1), (64, 64, 8, 3), (2, 33, 1, 2)] {
        let c = case(&mut backend, k_embd, n_expert, k, n, true);
        let at = format!("{k_embd} x {n_expert}, top {k}, {n} tokens");
        assert_eq!(c.logits, c.ref_logits, "{at}: logits");
        assert_eq!(c.ids, c.ref_ids, "{at}: ids");
        assert_eq!(c.w, c.ref_w, "{at}: weights");
    }
}

#[test]
fn fits_takes_mul_mat_vec_f_shapes_only() {
    // Each check makes its tensors: ~30 with the BF16 cases.
    let ctx = Context::new(64, 16);
    let w = ctx.new_tensor(types::F32, &[2560, 512]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in 1..=MAX_TOKENS {
        assert!(fits(w, x(n), 10), "{n} tokens");
    }
    // ggml runs 4 tokens on mul_mat_f<float>, other bits.
    assert!(!fits(w, x(MAX_TOKENS + 1), 10));
    // top k of the experts there are
    assert!(!fits(w, x(1), 0));
    assert!(!fits(w, x(1), 513));
    // more experts than one warp's lanes hold
    assert!(!fits(ctx.new_tensor(types::F32, &[2560, 1024]), x(1), 10));
    // an odd row (float2 loads), an F16 weight, a mismatched input
    assert!(!fits(
        ctx.new_tensor(types::F32, &[63, 64]),
        ctx.new_tensor(types::F32, &[63, 1]),
        4
    ));
    assert!(!fits(ctx.new_tensor(types::F16, &[2560, 512]), x(1), 10));
    assert!(!fits(w, ctx.new_tensor(types::F32, &[2048, 1]), 10));
    // A router's BF16 copy (#836) of the same shapes.
    let wb = ctx.new_tensor(types::BF16, &[2560, 512]);
    for n in 1..=MAX_TOKENS {
        assert!(fits(wb, x(n), 10), "BF16, {n} tokens");
    }
    assert!(!fits(wb, x(MAX_TOKENS + 1), 10));
    assert!(!fits(
        ctx.new_tensor(types::BF16, &[63, 64]),
        ctx.new_tensor(types::F32, &[63, 1]),
        4
    ));
}

/// The op's outputs (logits, ids, weights) over an F32 router whose values are all exact in
/// BF16 and over its BF16 copy ([`super::super::router_bf16`]), `n` tokens.
fn wide_case(backend: &mut Backend, k_embd: i64, n_expert: i64, k: i64, n: i64) -> [Vec<u32>; 6] {
    let ctx = Context::new(32, 64);
    let wf = ctx.new_tensor(types::F32, &[k_embd, n_expert]).as_input();
    let wb = ctx.new_tensor(types::BF16, &[k_embd, n_expert]).as_input();
    let x = ctx.new_tensor(types::F32, &[k_embd, n]).as_input();
    let (l, i, w) = router_mv(&ctx, wf, x, k, 5);
    let (lb, ib, wbt) = router_mv(&ctx, wb, x, k, 6);
    let outs = [l, i, w, lb, ib, wbt].map(|t| t.as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // F32 values truncated to BF16 (scaled like a router's).
    let vals: Vec<f32> = noise((k_embd * n_expert) as usize, 21)
        .iter()
        .map(|v| f32::from_bits((v * 0.05).to_bits() & 0xffff_0000))
        .collect();
    let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    wf.set_bytes(&f32_bytes);
    wb.set_bytes(&super::super::router_bf16::bf16_copy(&f32_bytes).expect("exact"));
    x.set_f32(&noise((k_embd * n) as usize, 22));
    backend.compute(&graph).unwrap();
    outs.map(|t| {
        t.bytes()
            .chunks(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    })
}

#[test]
fn bf16_copy_gives_the_f32_routers_bits() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (k_embd, n_expert, k, n) in [(256, 512, 10, 1), (64, 64, 8, 3), (2, 33, 1, 2)] {
        let [l, i, w, lb, ib, wb] = wide_case(&mut backend, k_embd, n_expert, k, n);
        let at = format!("{k_embd} x {n_expert}, top {k}, {n} tokens");
        assert_eq!(l, lb, "{at}: logits");
        assert_eq!(i, ib, "{at}: ids");
        assert_eq!(w, wb, "{at}: weights");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_bf16_copy_gives_the_f32_routers_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        return;
    }
    for (k_embd, n_expert, k, n) in [
        (2560, 512, 10, 1),
        (2560, 512, 10, 2),
        (2560, 512, 10, 3),
        (2048, 256, 8, 3),
        (96, 33, 1, 2),
    ] {
        let [l, i, w, lb, ib, wb] = wide_case(&mut gpu, k_embd, n_expert, k, n);
        let at = format!("{k_embd} x {n_expert}, top {k}, {n} tokens");
        assert_eq!(l, lb, "{at}: logits");
        assert_eq!(i, ib, "{at}: ids");
        assert_eq!(w, wb, "{at}: weights");
    }
}

#[test]
fn off_by_switch() {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::F32, &[2560, 512]);
    let x = ctx.new_tensor(types::F32, &[2560, 1]);
    set_router_mv(false);
    assert!(!takes(w, x, 10, 0));
    set_router_mv(true);
    // Without CUDA the op is never taken.
    if !cfg!(feature = "cuda") {
        assert!(!takes(w, x, 10, 0));
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_router_mv_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        return;
    }
    for (k_embd, n_expert, k, n) in [
        (2560, 512, 10, 1),
        (2560, 512, 10, 2),
        (2560, 512, 10, 3),
        (2048, 256, 8, 3),
        (4096, 128, 8, 1),
        (96, 33, 1, 2),
        // the weights from the picked logits a warp holds (k <= 32), and the reads above that
        (2560, 512, 32, 3),
        (512, 64, 33, 2),
    ] {
        let c = case(&mut gpu, k_embd, n_expert, k, n, false);
        let at = format!("{k_embd} x {n_expert}, top {k}, {n} tokens");
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&c.logits), bits(&c.ref_logits), "{at}: logits");
        assert_eq!(c.ids, c.ref_ids, "{at}: ids");
        assert_eq!(bits(&c.w), bits(&c.ref_w), "{at}: weights");
    }
}
