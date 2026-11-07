//! The ops against `fp8::quantize_dynamic` of the ggml chains they replace: on the CPU to the
//! reference's rounding (it sums in another order), on CUDA (ignored tests) bit for bit.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::fp8::{e4m3_to_f32, quantize_dynamic};

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

/// Packed activations of `k` values per token: the dequantized values `[n][k]` and the scales.
fn unpack(bytes: &[u8], k: usize) -> (Vec<f32>, Vec<f32>) {
    let n = bytes.len() / (k + 4);
    let (codes, total) = fp8::dynamic_layout(k as i64, n as i64);
    assert_eq!(total, bytes.len());
    let scales: Vec<f32> = bytes[codes..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_ne_bytes(b))
        .collect();
    let values = (0..k * n)
        .map(|i| e4m3_to_f32(bytes[i]) * scales[i / k])
        .collect();
    (values, scales)
}

/// The op's and the chain's packed activations (op, chain) on `backend`: the CPU's within the
/// reference's rounding, a GPU's the same bytes.
fn compare(op: &[u8], chain: &[u8], k: usize, exact: bool, what: &str) {
    assert_eq!(op.len(), chain.len(), "{what}: sizes");
    if exact {
        let first = op.iter().zip(chain).position(|(a, b)| a != b);
        assert!(first.is_none(), "{what}: byte {first:?} differs");
        return;
    }
    let ((ov, os), (cv, cs)) = (unpack(op, k), unpack(chain, k));
    for (t, (a, b)) in os.iter().zip(&cs).enumerate() {
        assert!(
            (a - b).abs() <= b.abs() * 1e-5,
            "{what}: token {t} scale {a} vs {b}"
        );
    }
    // Codes may round the other way where the sums differ in the last bit: one E4M3 step
    // (1/8 of the value, or the scale's 1/512 near zero) at most.
    for (i, (a, b)) in ov.iter().zip(&cv).enumerate() {
        let s = cs[i / k];
        assert!(
            (a - b).abs() <= b.abs() / 8.0 + s / 256.0,
            "{what}: value {i}: {a} vs {b}"
        );
    }
}

/// Runs `op` and `quantize_dynamic(chain)` in one graph on `backend` after `fill` set the
/// inputs: (op bytes, chain bytes).
fn run<'a>(
    backend: &mut Backend,
    ctx: &'a Context,
    op: Tensor<'a>,
    chain: Tensor<'a>,
    fill: impl FnOnce(),
) -> (Vec<u8>, Vec<u8>) {
    let op = op.as_output();
    let q = quantize_dynamic(ctx, ctx.cont(chain)).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(op);
    graph.expand(q);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    fill();
    backend.compute(&graph).unwrap();
    (op.bytes(), q.bytes())
}

/// Rows whose magnitudes spread over six orders, and one all-zero token.
fn spread(k: usize, n: usize, seed: u64) -> Vec<f32> {
    noise(k * n, seed)
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let t = i / k;
            if t == 1 {
                0.0
            } else {
                v * 10f32.powi((t % 7) as i32 - 3)
            }
        })
        .collect()
}

fn rms_case(backend: &mut Backend, k: i64, n: i64) -> (Vec<u8>, Vec<u8>) {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[k]).as_input();
    let op = rms_norm(&ctx, x, w, eps);
    let chain = ctx.mul(ctx.rms_norm(x, eps), w);
    run(backend, &ctx, op, chain, || {
        x.set_f32(&spread(k as usize, n as usize, 61));
        w.set_f32(&noise(k as usize, 62));
    })
}

/// [`rms_norm_bf16`] against the two ops it replaces, in one graph on `backend`: (its packed
/// activations, [`rms_norm`]'s, its BF16 rows, [`fused::rms_norm_bf16`]'s).
///
/// [`fused::rms_norm_bf16`]: crate::model::core::fused::rms_norm_bf16
fn rms_bf16_case(backend: &mut Backend, k: i64, n: i64) -> [Vec<u8>; 4] {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[k]).as_input();
    let (q, b) = rms_norm_bf16(&ctx, x, w, eps);
    let outs = [
        q,
        rms_norm(&ctx, x, w, eps),
        b,
        crate::model::core::fused::rms_norm_bf16(&ctx, x, w, eps),
    ]
    .map(|t| ctx.cont(t).as_output());
    let mut graph = ctx.new_graph(64);
    for t in outs {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&spread(k as usize, n as usize, 81));
    w.set_f32(&noise(k as usize, 82));
    backend.compute(&graph).unwrap();
    outs.map(|t| t.bytes())
}

fn check_rms_bf16(backend: &mut Backend, k: i64, n: i64) {
    let [q, q1, b, b1] = rms_bf16_case(backend, k, n);
    compare(
        &q,
        &q1,
        k as usize,
        true,
        &format!("rms_bf16 codes {k}x{n}"),
    );
    assert_eq!(b.len(), 2 * (k * n) as usize, "rms_bf16 {k}x{n}: BF16 size");
    assert!(b == b1, "rms_bf16 {k}x{n}: BF16 rows differ");
}

/// `silu`: SwiGLU of a plain `[k, n]` gate; else `o * sigmoid(gate)` with the gate read from
/// interleaved `[q | gate]` heads of `d`.
fn gate_case(backend: &mut Backend, d: i64, n_head: i64, n: i64, silu: bool) -> (Vec<u8>, Vec<u8>) {
    let ctx = Context::new(32, 32);
    let k = d * n_head;
    let (src, x) = if silu {
        let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
        (x, x)
    } else {
        let qg = ctx.new_tensor(types::F32, &[2 * k, n]).as_input();
        let nb = [(2 * d) as usize * 4, (2 * k) as usize * 4];
        (qg, ctx.view_3d(qg, [d, n_head, n], nb, d as usize * 4))
    };
    let g = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let op = gate(&ctx, x, g, silu);
    let chain = if silu {
        ctx.swiglu_split(x, g)
    } else {
        ctx.mul(g, ctx.sigmoid(ctx.cont_2d(x, k, n)))
    };
    run(backend, &ctx, op, chain, || {
        let wide: Vec<f32> = noise(src.elements(), 71).iter().map(|v| v * 8.0).collect();
        src.set_f32(&wide);
        g.set_f32(&spread(k as usize, n as usize, 72));
    })
}

fn gated_case(
    backend: &mut Backend,
    head_v: i64,
    n_v: i64,
    n: i64,
    silu: bool,
) -> (Vec<u8>, Vec<u8>) {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let k = head_v * n_v;
    let o = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[head_v]).as_input();
    let z = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let op = gated_norm(&ctx, o, w, z, head_v, eps, silu);
    let z4 = ctx.reshape_4d(z, [head_v, n_v, n, 1]);
    let g = if silu { ctx.silu(z4) } else { ctx.sigmoid(z4) };
    let o4 = ctx.reshape_4d(o, [head_v, n_v, n, 1]);
    let chain = ctx.mul(ctx.mul(ctx.rms_norm(o4, eps), w), g);
    let chain = ctx.reshape_2d(chain, k, n);
    run(backend, &ctx, op, chain, || {
        o.set_f32(&spread(k as usize, n as usize, 81));
        w.set_f32(&noise(head_v as usize, 82));
        let wide: Vec<f32> = noise((k * n) as usize, 83)
            .iter()
            .map(|v| v * 8.0)
            .collect();
        z.set_f32(&wide);
    })
}

/// Row widths: the 1024-thread order (5120, 2048, 1024) and the 256-thread one (1000, 128).
const RMS_CASES: [(i64, i64); 5] = [(5120, 40), (2048, 33), (1024, 35), (1000, 34), (128, 36)];
/// (d, n_head, n, silu): the attention output gate of interleaved heads, SwiGLU rows.
const GATE_CASES: [(i64, i64, i64, bool); 4] = [
    (256, 24, 37, false),
    (64, 3, 40, false),
    (17408, 1, 35, true),
    (640, 1, 33, true),
];
/// (head_v, n_v, n): every head size the kernel takes.
const GATED_CASES: [(i64, i64, i64); 4] = [(128, 48, 40), (128, 3, 33), (64, 20, 34), (256, 5, 35)];

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b
}

#[test]
fn rms_norm_matches_the_chain_quantized() {
    let mut b = cpu();
    for (k, n) in RMS_CASES {
        let (op, chain) = rms_case(&mut b, k, n);
        compare(&op, &chain, k as usize, false, &format!("rms {k}x{n}"));
    }
}

/// The fused rows are the two ops' bytes (the CPU references share their arithmetic).
#[test]
fn rms_norm_bf16_is_both_ops() {
    let mut b = cpu();
    for (k, n) in [(256, 3), (1024, 5), (2056, 9)] {
        check_rms_bf16(&mut b, k, n);
    }
}

/// The fused node: BF16 rows `[k, n]`, then the packed activations, 16-byte aligned.
#[test]
fn rms_norm_bf16_layout() {
    let ctx = Context::new(32, 0);
    let x = ctx.new_tensor(types::F32, &[256, 71]);
    let w = ctx.new_tensor(types::F32, &[256]);
    let (q, b) = rms_norm_bf16(&ctx, x, w, 1e-6);
    assert_eq!(q.elements(), fp8::dynamic_layout(256, 71).1);
    assert_eq!((b.ty(), b.ne()), (types::BF16, [256, 71, 1, 1]));
    // SAFETY: live nodes of `ctx`.
    unsafe {
        let node = &*(*q.raw()).view_src;
        assert_eq!((*b.raw()).view_src, node as *const _ as *mut _);
        assert_eq!(tokens(node), 71);
        assert_eq!((*q.raw()).view_offs, 2 * 256 * 71);
    }
}

#[test]
fn gate_matches_the_chain_quantized() {
    let mut b = cpu();
    for (d, h, n, silu) in GATE_CASES {
        let (op, chain) = gate_case(&mut b, d, h, n, silu);
        compare(
            &op,
            &chain,
            (d * h) as usize,
            false,
            &format!("gate {d}x{h}x{n} {silu}"),
        );
    }
}

#[test]
fn gated_norm_matches_the_chain_quantized() {
    let mut b = cpu();
    for (head_v, n_v, n) in GATED_CASES {
        for silu in [false, true] {
            let (op, chain) = gated_case(&mut b, head_v, n_v, n, silu);
            let what = format!("gated {head_v}x{n_v}x{n} {silu}");
            compare(&op, &chain, (head_v * n_v) as usize, false, &what);
        }
    }
}

/// `share` makes every dynamic FP8 matmul of the activation read the op instead of a
/// quantization of its own.
#[test]
fn shared_op_feeds_the_dynamic_matmuls() {
    let ctx = Context::new(32, 0);
    let x = ctx.new_tensor(types::F32, &[128, 80]);
    let w = ctx.new_tensor(types::F32, &[128]);
    let h = ctx.mul(ctx.rms_norm(x, 1e-6), w);
    let q = rms_norm(&ctx, x, w, 1e-6);
    share(&ctx, h, q);
    for m in [64, 192] {
        let wt = ctx.new_tensor(types::I8, &[128, m]);
        let y = fp8::mul_mat_dynamic(&ctx, wt, h, None);
        // SAFETY: a live node of `ctx`.
        assert_eq!(unsafe { (*y.raw()).src[1] }, q.raw(), "matmul {m}");
    }
    // Another activation still gets its own quantization.
    let wt = ctx.new_tensor(types::I8, &[128, 64]);
    let y = fp8::mul_mat_dynamic(&ctx, wt, x, None);
    // SAFETY: as above.
    assert_ne!(unsafe { (*y.raw()).src[1] }, q.raw());
}

/// The ops' nodes hold the packed layout of their activations, `n` tokens from any shape.
#[test]
fn ops_are_packed_activations() {
    let ctx = Context::new(32, 0);
    let x = ctx.new_tensor(types::F32, &[256, 70]);
    let w = ctx.new_tensor(types::F32, &[256]);
    let total = fp8::dynamic_layout(256, 70).1;
    assert_eq!(rms_norm(&ctx, x, w, 1e-6).elements(), total);
    let qg = ctx.new_tensor(types::F32, &[512, 70]);
    let g = ctx.view_3d(qg, [64, 4, 70], [512, 2048], 256);
    assert_eq!(gate(&ctx, g, x, false).elements(), total);
    let wv = ctx.new_tensor(types::F32, &[64]);
    assert_eq!(gated_norm(&ctx, x, wv, x, 64, 1e-6, true).elements(), total);
    let node = gate(&ctx, g, x, false);
    // SAFETY: a live node of `ctx`.
    assert_eq!(tokens(unsafe { &*node.raw() }), 70);
}

/// Without CUDA (or on the host) nothing fits: the CPU graphs keep the chains.
#[cfg(not(feature = "cuda"))]
#[test]
fn nothing_fits_without_cuda() {
    let ctx = Context::new(8, 0);
    let x = ctx.new_tensor(types::F32, &[256, 70]);
    let w = ctx.new_tensor(types::F32, &[256]);
    assert!(!rms_norm_fits(x, w));
    assert!(!gate_fits(x, x));
    assert!(!gated_norm_fits(
        x,
        ctx.new_tensor(types::F32, &[128]),
        x,
        128
    ));
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// On a GPU each op writes the bytes of ggml-cuda's chain followed by `quantize_dynamic`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ops_are_the_chain_quantized() {
    let mut g = gpu();
    for (k, n) in RMS_CASES.into_iter().chain([(5120, 700)]) {
        let (op, chain) = rms_case(&mut g, k, n);
        compare(&op, &chain, k as usize, true, &format!("rms {k}x{n}"));
    }
    for (k, n) in [(256, 40), (1024, 33), (5120, 700)] {
        check_rms_bf16(&mut g, k, n);
    }
    for (d, h, n, silu) in GATE_CASES.into_iter().chain([(256, 24, 500, false)]) {
        let (op, chain) = gate_case(&mut g, d, h, n, silu);
        compare(
            &op,
            &chain,
            (d * h) as usize,
            true,
            &format!("gate {d}x{h}x{n} {silu}"),
        );
    }
    for (head_v, n_v, n) in GATED_CASES.into_iter().chain([(128, 48, 600)]) {
        for silu in [false, true] {
            let (op, chain) = gated_case(&mut g, head_v, n_v, n, silu);
            let what = format!("gated {head_v}x{n_v}x{n} {silu}");
            compare(&op, &chain, (head_v * n_v) as usize, true, &what);
        }
    }
}

/// `quantize_dynamic` on a GPU (one pass where the row fits in registers, two passes beyond)
/// gives `quantize_token`'s bytes at every row width.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_quantize_dynamic_is_quantize_token() {
    let mut g = gpu();
    for (k, n) in [
        (4usize, 9usize),
        (1000, 33),
        (6144, 70),
        (6148, 65),
        (17408, 66),
        (24576, 3),
        (30000, 5),
    ] {
        let ctx = Context::new(8, 8);
        let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
        let q = quantize_dynamic(&ctx, x).as_output();
        let mut graph = ctx.new_graph(8);
        graph.expand(q);
        ctx.alloc_graph(&graph, ComputeAllocator::new(g.buffer_type()))
            .unwrap();
        let xs = spread(k, n, 91);
        x.set_f32(&xs);
        g.compute(&graph).unwrap();
        let got = q.bytes();
        let (codes, _) = fp8::dynamic_layout(k as i64, n as i64);
        for t in 0..n {
            let (want, s) = fp8::quantize_token(&xs[t * k..(t + 1) * k]);
            assert_eq!(got[t * k..(t + 1) * k], want[..], "k {k} token {t}: codes");
            let at = codes + 4 * t;
            let gs = f32::from_ne_bytes(got[at..at + 4].try_into().unwrap());
            assert_eq!(gs.to_bits(), s.to_bits(), "k {k} token {t}: scale");
        }
    }
}
