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

/// The quantized op ([`input_q8`]) and `eh_proj`'s MMQ on its blocks
/// ([`mmq_preq::mul_mat_cols`]) for `n` tokens of width `e` and a Q8_0 weight of `rows` rows,
/// next to `chain`'s values (ggml's chain as the head builds it, or with `!chain` the F32 op)
/// quantized by ggml ([`ggml_quantize`]) and ggml's `mul_mat` of them: [our blocks, ggml's
/// blocks] (without MMQ's padding) and [our product, ggml's product].
fn run_q8(backend: &mut Backend, chain: bool, e: i64, n: i64, rows: i64) -> [Vec<u8>; 4] {
    use crate::model::core::mmq_preq::tests::{ggml_quantize, q8_0_matrix};
    let ctx = Context::new(64, 64);
    let input = |shape: &[i64]| ctx.new_tensor(types::F32, shape).as_input();
    let tok = input(&[e, n]);
    let h = input(&[e * HC, n]);
    let ge = input(&[e]);
    let gh = input(&[e * HC]);
    let w = ctx.new_tensor(types::Q8_0, &[2 * e, rows]).as_input();
    let x = if chain {
        let hn = ctx.mul(
            ctx.rms_norm(ctx.reshape_3d(h, e, HC, n), EPS),
            ctx.reshape_2d(gh, e, HC),
        );
        let en = ctx.mul(ctx.rms_norm(tok, EPS), ge);
        let en = ctx.repeat_4d(ctx.reshape_3d(en, e, 1, n), [e, HC, n, 1]);
        ctx.cont(ctx.concat(en, hn, 0))
    } else {
        super::input(&ctx, tok, h, ge, gh, HC, EPS)
    };
    let x = ctx.reshape_2d(x, 2 * e, HC * n);
    let xq = input_q8(&ctx, tok, h, ge, gh, HC, EPS);
    let outs = [
        xq,
        ggml_quantize(&ctx, x),
        mmq_preq::mul_mat_cols(&ctx, w, xq, HC * n),
        ctx.mul_mat(w, x),
    ]
    .map(|t| t.as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let scaled = |len: i64, seed, s: f32| -> Vec<f32> {
        noise(len as usize, seed).iter().map(|v| s * v).collect()
    };
    let gamma = |len: i64, s| -> Vec<f32> {
        noise(len as usize, s)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect()
    };
    // The embedding with an all-zero group in token 0 (a NaN scale's group, written as 0).
    let mut tv = scaled(e * n, 7, 0.05);
    tv[32..64].fill(0.0);
    tok.set_f32(&tv);
    h.set_f32(&scaled(e * HC * n, 8, 4.0));
    ge.set_f32(&gamma(e, 9));
    gh.set_f32(&gamma(e * HC, 10));
    w.set_bytes(&q8_0_matrix((2 * e) as usize, rows as usize, 11));
    backend.compute(&graph).unwrap();
    let data = (2 * e / mmq_preq::BLOCK * HC * n) as usize * mmq_preq::BLOCK_BYTES;
    let [a, b, c, d] = outs.map(|t| t.bytes());
    [a[..data].to_vec(), b[..data].to_vec(), c, d]
}

/// On the CPU the quantized op's blocks are the reference quantization of the F32 op's values,
/// column `t · hc + c` for token `t`, stream `c`.
#[test]
fn cpu_mtp_in_q8_is_the_reference_quantization() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(3);
    for (e, n) in [(256, 1), (256, 5), (512, 3)] {
        let [ours, reference, ..] = run_q8(&mut backend, false, e, n, 32);
        assert!(ours.iter().any(|&b| b != 0), "e {e} n {n}: none");
        let diff = ours.iter().zip(&reference).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "e {e}, n {n}: {diff} bytes moved");
    }
}

#[test]
fn why_not_q8_wants_whole_blocks_and_a_q8_0_weight() {
    let ctx = Context::new(64, 0);
    let w = |ty, k| ctx.new_tensor(ty, &[k, 2560]);
    // Rows of 2e only, e a multiple of 256.
    assert!(why_not_q8(w(types::Q8_0, 4096), 2560, HC, 1024).is_some());
    assert!(why_not_q8(w(types::Q8_0, 2 * 1088), 1088, HC, 1024).is_some());
    // An F16 weight stays on ggml's quantize, and so does one not in a 12.x GPU's memory
    // (these have no memory at all).
    assert!(why_not_q8(w(types::Q8_0, 5120), 2560, HC, 1024).is_some());
    assert!(why_not_q8(w(types::F16, 5120), 2560, HC, 1024).is_some());
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

/// On CUDA the quantized op writes ggml-cuda's blocks for ggml's chain's values (its
/// `quantize_mmq_q8_1`), and MMQ on them gives ggml's `mul_mat` of the chain, bit for bit:
/// Qwen3.8-Flash-Next's `eh_proj` (2,560 rows of 5,120) at a take-in's 1,024 tokens and fewer.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_mtp_in_q8_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for n in [1024, 1016, 896, 65, 64] {
        let out = run_q8(&mut gpu, true, 2560, n, 2560);
        for (i, what) in ["quantized", "product"].iter().enumerate() {
            let (ours, ggml) = (&out[2 * i], &out[2 * i + 1]);
            let diff = ours.iter().zip(ggml).filter(|(a, b)| a != b).count();
            assert!(ours.iter().any(|&b| b != 0), "{n} tokens: no {what} bytes");
            assert_eq!(diff, 0, "{n} tokens: {diff} {what} bytes moved");
        }
    }
}
