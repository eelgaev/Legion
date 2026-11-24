//! The MTP input op against ggml's chain as [`Model::build_mtp_qwen4`](super::super::Model)
//! builds it: on the CPU within rounding (the CPU reference), on CUDA (an ignored test) bit for
//! bit.

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

/// Max |a - b| over max |b|, of finite values only (`f32::max` would skip a NaN).
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "a value is not finite"
    );
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const EPS: f32 = 1e-6;
const HC: i64 = 4;

/// (op's output, chain's output) for `n` tokens of width `e`.
fn run(backend: &mut Backend, e: i64, n: i64, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(64, 64);
    let input = |shape: &[i64]| ctx.new_tensor(types::F32, shape).as_input();
    let tok = input(&[e, n]);
    let h = input(&[e * HC, n]);
    let ge = input(&[e]);
    let gh = input(&[e * HC]);
    // The chain as the head's graph builds it.
    let hn = ctx.mul(
        ctx.rms_norm(ctx.reshape_3d(h, e, HC, n), EPS),
        ctx.reshape_2d(gh, e, HC),
    );
    let en = ctx.mul(ctx.rms_norm(tok, EPS), ge);
    let en = ctx.repeat_4d(ctx.reshape_3d(en, e, 1, n), [e, HC, n, 1]);
    let ob = ctx.cont(ctx.concat(en, hn, 0)).as_output();
    let oa = super::input(&ctx, tok, h, ge, gh, HC, EPS).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(ob);
    graph.expand(oa);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // The embedding and the streams at different scales, the gammas around 1 as trained norms.
    let gamma = |len: i64, s| -> Vec<f32> {
        noise(len as usize, s)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect()
    };
    tok.set_f32(
        &noise((e * n) as usize, seed)
            .iter()
            .map(|v| 0.05 * v)
            .collect::<Vec<_>>(),
    );
    h.set_f32(
        &noise((e * HC * n) as usize, seed + 1)
            .iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>(),
    );
    ge.set_f32(&gamma(e, seed + 2));
    gh.set_f32(&gamma(e * HC, seed + 3));
    backend.compute(&graph).unwrap();
    (oa.to_f32(), ob.to_f32())
}

#[test]
fn mtp_in_matches_the_chain_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, n) in [(64, 1), (64, 7), (1024, 3), (1500, 5)] {
        let (a, b) = run(&mut backend, e, n, 11 + n as u64);
        let r = rel(&a, &b);
        assert!(r < 1e-5, "e {e}, n {n}: rel {r:e}");
    }
}

#[test]
fn why_not_wants_the_kernels_shapes() {
    let ctx = Context::new(64, 0);
    let t = |shape: &[i64]| ctx.new_tensor(types::F32, shape);
    let fits = |a, b, c, d, hc| why_not(a, b, c, d, hc).is_none();
    let (e, n) = (2560, 8);
    let ok = fits(t(&[e, n]), t(&[e * HC, n]), t(&[e]), t(&[e * HC]), HC);
    assert_eq!(ok, cfg!(feature = "cuda"));
    // Other stream counts, narrow rows, mismatched streams and F16 gammas stay on the chain.
    assert!(!fits(t(&[e, n]), t(&[e * 2, n]), t(&[e]), t(&[e * 2]), 2));
    assert!(!fits(
        t(&[512, n]),
        t(&[512 * HC, n]),
        t(&[512]),
        t(&[512 * HC]),
        HC
    ));
    assert!(!fits(
        t(&[e, n]),
        t(&[e * HC, n + 1]),
        t(&[e]),
        t(&[e * HC]),
        HC
    ));
    let half = ctx.new_tensor(types::F16, &[e]);
    assert!(!fits(t(&[e, n]), t(&[e * HC, n]), half, t(&[e * HC]), HC));
}

/// On CUDA the op computes what ggml-cuda's kernels for the chain compute, bit for bit
/// (Qwen3.8-Flash-Next's width and a take-in's 1,024 rows among others).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mtp_in_is_bit_identical_to_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, n) in [
        (2560, 1024),
        (2560, 1016),
        (2560, 65),
        (2560, 1),
        (1024, 9),
        (1500, 33),
    ] {
        let (a, b) = run(&mut gpu, e, n, 5 + n as u64);
        assert!(
            same_bits(&a, &b),
            "e {e}, n {n}: differs (rel {:e})",
            rel(&a, &b)
        );
    }
}
