//! `moe_tail` against the ops it replaces: `fused::router`'s weights, `moe_combine` and
//! `shexp_gate`, one after the other.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::fused;

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// `moe_tail` and the three ops on `backend` for `n` tokens of `e` columns, `used` of
/// `n_expert` experts, the router logits and the gate logit one joined matmul's rows
/// (`[n_expert + 1, n]`, the gate last): (tail, chain).
fn tail_case(
    backend: &mut Backend,
    e: i64,
    used: i64,
    n_expert: i64,
    n: i64,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(64, 64);
    let joined = ctx.new_tensor(types::F32, &[n_expert + 1, n]).as_input();
    let nb1 = joined.nb()[1];
    let logits = ctx.view_2d(joined, [n_expert, n], nb1, 0);
    let g = ctx.view_2d(joined, [1, n], nb1, n_expert as usize * 4);
    let experts = ctx.new_tensor(types::F32, &[e, used, n]).as_input();
    let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let (ids, w) = fused::router(&ctx, logits, used);
    let tail = moe_tail(&ctx, experts, logits, ids, sh, g).as_output();
    let moe = fused::moe_combine(&ctx, experts, ctx.reshape_3d(w, 1, used, n));
    let chain = fused::shexp_gate(&ctx, moe, sh, g).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(tail);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Logits over a wide range, so the sigmoid saturates and the weights spread.
    let l: Vec<f32> = noise(((n_expert + 1) * n) as usize, 71)
        .iter()
        .map(|v| v * 8.0)
        .collect();
    joined.set_f32(&l);
    experts.set_f32(&noise((e * used * n) as usize, 72));
    sh.set_f32(&noise((e * n) as usize, 73));
    backend.compute(&graph).unwrap();
    (tail.to_f32(), chain.to_f32())
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn moe_tail_is_the_three_ops() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, used, n_expert, n) in [(2560, 10, 512, 1), (2560, 10, 512, 3), (96, 4, 16, 7)] {
        let (tail, chain) = tail_case(&mut backend, e, used, n_expert, n);
        assert!(same_bits(&tail, &chain), "{e}x{used}/{n_expert}x{n}");
    }
}

/// On a GPU the kernel gives `router_weights`, `moe_combine` and `shexp_gate`'s bits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_moe_tail_is_the_three_ops() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, used, n_expert, n) in [
        (2560, 10, 512, 1),
        (2560, 10, 512, 3),
        (2560, 10, 512, 64),
        (96, 4, 16, 7),
    ] {
        let (tail, chain) = tail_case(&mut gpu, e, used, n_expert, n);
        assert!(same_bits(&tail, &chain), "{e}x{used}/{n_expert}x{n}");
    }
}

#[test]
fn off_by_default() {
    assert!(!ON.load(Ordering::Relaxed));
}

#[test]
fn moe_tail_fits_checks_the_shapes() {
    let ctx = Context::new(16, 16);
    let (e, used, n_expert, n) = (64, 4, 16, 3);
    let joined = ctx.new_tensor(types::F32, &[n_expert + 1, n]);
    let nb1 = joined.nb()[1];
    let logits = ctx.view_2d(joined, [n_expert, n], nb1, 0);
    let g = ctx.view_2d(joined, [1, n], nb1, n_expert as usize * 4);
    let experts = ctx.new_tensor(types::F32, &[e, used, n]);
    let ids = ctx.new_tensor(types::I32, &[used, n]);
    let sh = ctx.new_tensor(types::F32, &[e, n]);
    let wrong_sh = ctx.new_tensor(types::F32, &[e, n + 1]);
    let wrong_ids = ctx.new_tensor(types::I32, &[used + 1, n]);
    let too_many = ctx.new_tensor(types::F32, &[e, MAX_USED + 1, n]);
    // Unallocated tensors are not in host memory: the shapes decide.
    assert!(moe_tail_fits(experts, logits, ids, sh, g));
    assert!(!moe_tail_fits(experts, logits, ids, wrong_sh, g));
    assert!(!moe_tail_fits(experts, logits, wrong_ids, sh, g));
    assert!(!moe_tail_fits(too_many, logits, ids, sh, g));
    assert!(!moe_tail_fits(experts, logits, ids, sh, sh));
}
