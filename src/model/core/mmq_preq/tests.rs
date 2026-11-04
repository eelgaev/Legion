//! The layout and the CPU reference of the pre-quantized MMQ matmul, and (for the GPU tests of
//! the kernels that write the blocks, `src/model/qwen4/hc_norm/tests.rs`) ggml's own
//! quantization as an op.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
pub(crate) fn noise(n: usize, seed: u64) -> Vec<f32> {
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

/// A Q8_0 matrix of `rows` rows of `k` values: random int8 in -127..=127, scales in
/// [0.001, 0.011).
pub(crate) fn q8_0_matrix(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let qs = noise(k * rows, seed);
    let ds = noise(k * rows / GROUP, seed + 1);
    let mut out = Vec::with_capacity(k * rows / GROUP * 34);
    for (g, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        out.extend_from_slice(&h.to_le_bytes());
        for q in &qs[g * GROUP..(g + 1) * GROUP] {
            out.push(((q * 127.0).round() as i8) as u8);
        }
    }
    out
}

/// The F32 value of a block's scale `g` and quant `i` of column `j`'s block `b`.
fn at(xq: &[u8], n: usize, j: usize, b: usize, i: usize) -> (f32, i8) {
    let blk = &xq[(b * n + j) * BLOCK_BYTES..][..BLOCK_BYTES];
    let g = i / GROUP;
    let d = f32::from_ne_bytes(blk[g * 4..g * 4 + 4].try_into().unwrap());
    (d, blk[16 + i] as i8)
}

#[test]
fn quantize_ref_writes_mmq_blocks() {
    let (k, n) = (512, 3);
    let mut x = noise(k * n, 1);
    // Column 1, block 2, group 1: all zero; column 2, block 0, group 3: one large value.
    x[k + 2 * 128 + 32..k + 2 * 128 + 64].fill(0.0);
    x[2 * k + 3 * 32 + 5] = 1e6;
    let mut xq = vec![0u8; q8_1_bytes(k as i64, n as i64)];
    assert_eq!(xq.len(), (k / 128 * n) * 144 + PAD_BYTES);
    quantize_ref(&x, k, n, &mut xq);
    for j in 0..n {
        for v in 0..k {
            let (b, i) = (v / 128, v % 128);
            let (d, q) = at(&xq, n, j, b, i);
            let group = &x[j * k + v / GROUP * GROUP..][..GROUP];
            let amax = group.iter().fold(0f32, |m, x| m.max(x.abs()));
            let want = x[j * k + v];
            if amax == 0.0 {
                assert_eq!((d, q), (0.0, 0), "column {j} value {v}: an all-zero group");
            } else {
                assert!(
                    (d - amax / 127.0).abs() <= amax * 1e-6,
                    "column {j} value {v}: scale"
                );
                assert!(
                    (q as f32 * d - want).abs() <= d * 0.5 + amax * 1e-6,
                    "column {j} value {v}: {q} x {d} for {want}"
                );
            }
        }
    }
}

/// `mul_mat` on the CPU against the product of the dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, rows, n) = (1024usize, 5usize, 11usize);
    let wb = q8_0_matrix(k, rows, 3);
    let x = noise(k * n, 4);
    let mut xq = vec![0u8; q8_1_bytes(k as i64, n as i64)];
    quantize_ref(&x, k, n, &mut xq);

    let ctx = Context::new(16, 16);
    let w = ctx
        .new_tensor(types::Q8_0, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let qt = ctx
        .new_tensor(types::F32, &[(xq.len() / 4) as i64])
        .as_input();
    let y = mul_mat(&ctx, w, qt, xt).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&wb);
    xt.set_f32(&x);
    qt.set_bytes(&xq);
    backend.compute(&graph).unwrap();
    let got = y.to_f32();

    let mut worst = 0f64;
    for r in 0..rows {
        for j in 0..n {
            let mut want = 0f64;
            for v in 0..k {
                let blk = &wb[(r * k + v) / GROUP * 34..];
                // SAFETY: a plain conversion.
                let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                let qw = blk[2 + v % GROUP] as i8;
                let (dx, qx) = at(&xq, n, j, v / 128, v % 128);
                want += dw as f64 * qw as f64 * dx as f64 * qx as f64;
            }
            let g = got[j * rows + r] as f64;
            worst = worst.max((g - want).abs() / want.abs().max(1e-3));
        }
    }
    assert!(worst < 1e-5, "rel {worst:e}");
}

static GGML_Q: ExternalOp = ExternalOp {
    name: "ggml_quantize_mmq_q8_1",
    cpu: cpu_ggml_q,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ggml_q),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// ggml's quantization of `x` (F32 `[k, n]`, contiguous) for a Q8_0 weight's MMQ, as an op
/// ([`q8_1_bytes`] bytes, F32 elements): ggml-cuda's `quantize_mmq_q8_1` on CUDA,
/// [`quantize_ref`] on the CPU.
pub(crate) fn ggml_quantize<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let [k, n, _, _] = x.ne();
    let floats = q8_1_bytes(k, n).div_ceil(4) as i64;
    ctx.external(&GGML_Q, types::F32, [floats, 1, 1, 1], &[x])
}

unsafe fn cpu_ggml_q(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `ggml_quantize`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let xs = std::slice::from_raw_parts(x.data as *const f32, k * n);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut u8, ffi::ggml_nbytes(dst));
        quantize_ref(xs, k, n, out);
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ggml_q(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `ggml_quantize`, device tensors.
    unsafe {
        let x = &*dst.src[0];
        match legion_ggml_quantize_mmq_q8_0(
            x.data as *const f32,
            dst.data,
            x.ne[0],
            x.ne[1],
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("error {e}")),
        }
    }
}

#[test]
fn ggml_quantize_on_the_cpu_is_quantize_ref() {
    let (k, n) = (512i64, 9i64);
    let x = noise((k * n) as usize, 5);
    let ctx = Context::new(8, 8);
    let xt = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let q = ggml_quantize(&ctx, xt).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(q);
    let mut backend = Backend::cpu().unwrap();
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();
    let mut want = vec![0u8; q8_1_bytes(k, n)];
    quantize_ref(&x, k as usize, n as usize, &mut want);
    let data = (k / BLOCK * n) as usize * BLOCK_BYTES;
    assert_eq!(q.bytes()[..data], want[..data]);
}

#[test]
fn fits_needs_the_gpu_and_the_switch() {
    // On no GPU (the default until the engine sets one) nothing fits.
    let ctx = Context::new(4, 4);
    let w = ctx.new_tensor(types::Q8_0, &[10240, 320]);
    assert!(!fits(w, 4096));
}
