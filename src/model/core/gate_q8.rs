//! The sigmoid output gate in front of a Q8_0 output projection, prompt steps on a GB10:
//! `mul_mat(w, x · sigmoid(g))` with the gated values quantized for ggml-cuda's MMQ in the pass
//! that gates them (an external op, [`crate::ggml::ExternalOp`], the kernel in
//! `src/cuda/gate_q8.cu`), then MMQ on those blocks ([`mmq_preq::mul_mat_cols`]). ggml instead
//! writes the gated values as F32 (its fused sigmoid + mul) and the matmul quantizes them in a
//! pass of its own that reads them back (#526). A gated attention layer's `attn_output` and a
//! gated DeltaNet layer's `ssm_out` (qwen4exp's sigmoid gate) take it.
//!
//! The same bytes as ggml's: the kernel computes `sigmoid(g) · x` as ggml's `op_sigmoid` and
//! `unary_gated_op_kernel` do and quantizes as `quantize_mmq_q8_1` does (both with ggml-cuda's
//! flags), and the matmul is ggml's MMQ with ggml's arguments. The gate may be a view whose
//! values come in runs (the attention's per-head gate inside its query projection), so ggml's
//! copy of it into a contiguous tensor goes too.
//!
//! The DeltaNet layer's output RMS norm (ggml's `rms_norm` + `mul` by the norm weight, on rows of
//! 128 values) can go into the same pass ([`Norm`]): the kernel sums each row's squares in
//! ggml-cuda's order, so ggml's normed values, and the normed F32 tensor is never written.
//!
//! Chosen where [`mmq_preq::fits`] (compute capability 12.x, 256+ tokens); `legion serve
//! --gate-quant-ggml` keeps ggml's gate and quantization, `--gate-norm-ggml` only ggml's norm.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::mmq_preq;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static GATE_Q8: ExternalOp = ExternalOp {
    name: "gate_q8",
    cpu: cpu_gate_q8,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gate_q8),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Values per scale.
const GROUP: i64 = 32;

/// Whether graphs built from now on may use [`mul_mat`] (`legion serve --gate-quant-ggml` turns
/// it off for comparisons).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on gate and quantize a sigmoid-gated output projection's
/// input in one op where it fits (on by default); `false` keeps ggml's ops.
pub fn set_gate_q8(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on may fold an output RMS norm into the op ([`Norm`];
/// `legion serve --gate-norm-ggml` turns it off for comparisons).
static NORM_ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on norm a gated DeltaNet layer's output inside the
/// gate-and-quantize op where it fits (on by default); `false` keeps ggml's `rms_norm` + `mul`.
pub fn set_gate_norm(on: bool) {
    NORM_ON.store(on, Ordering::Relaxed);
}

/// Values per normed row: one block.
const NORM_ROW: i64 = mmq_preq::BLOCK;

/// `x` RMS-normed per row of [`NORM_ROW`] values and times `w` (F32 `[128]`) before the gate,
/// as ggml's `mul(rms_norm(x, eps), w)`.
#[derive(Clone, Copy)]
pub(crate) struct Norm<'a> {
    pub w: Tensor<'a>,
    pub eps: f32,
}

/// Whether the op takes [`Norm`] `w` on `x`: switched on, `x` `[128, heads, n]` (each block of
/// 128 values one row of the norm), `w` a contiguous F32 `[128]`.
pub(crate) fn norm_fits(x: Tensor<'_>, w: Tensor<'_>) -> bool {
    NORM_ON.load(Ordering::Relaxed)
        && x.ne()[0] == NORM_ROW
        && w.ty() == types::F32
        && w.ne() == [NORM_ROW, 1, 1, 1]
        && w.is_contiguous()
}

/// Where an operand's values sit, in floats: value `i` (of `k`) of token `t` at `t · ld + (i /
/// run) · stride + i % run`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Rows {
    pub ld: i64,
    pub run: i64,
    pub stride: i64,
}

/// The [`Rows`] of an F32 operand of shape `ne` and strides `nb` holding `k` values for each of
/// `n` tokens: `[k, n]` with contiguous rows, or `[run, k / run, n]` with contiguous runs (a
/// view of heads); `None` for anything else or runs that split a group of 32.
pub(super) fn rows_of(
    ty: ffi::ggml_type,
    ne: [i64; 4],
    nb: [usize; 4],
    k: i64,
    n: i64,
) -> Option<Rows> {
    if ty != types::F32 || nb[0] != 4 || nb.iter().any(|b| !b.is_multiple_of(4)) || ne[3] != 1 {
        return None;
    }
    let f = |b: usize| b as i64 / 4;
    let r = if ne[..3] == [k, n, 1] {
        Rows {
            ld: f(nb[1]),
            run: k,
            stride: k,
        }
    } else if ne[0] * ne[1] == k && ne[2] == n {
        // One run when the heads follow each other.
        let run = if ne[1] == 1 || f(nb[1]) == ne[0] {
            k
        } else {
            ne[0]
        };
        Rows {
            ld: f(nb[2]),
            run,
            stride: if run == k { k } else { f(nb[1]) },
        }
    } else {
        return None;
    };
    (r.run > 0 && r.run % GROUP == 0).then_some(r)
}

fn rows(t: Tensor<'_>, k: i64, n: i64) -> Option<Rows> {
    rows_of(t.ty(), t.ne(), t.nb(), k, n)
}

/// Whether [`mul_mat`] takes `w` times `x · sigmoid(g)` for `n` tokens: switched on, a weight
/// and step [`mmq_preq::fits`] takes, and F32 operands laid out as [`rows_of`] reads them.
pub(crate) fn fits(w: Tensor<'_>, x: Tensor<'_>, g: Tensor<'_>, n: i64) -> bool {
    let k = w.ne()[0];
    ON.load(Ordering::Relaxed)
        && mmq_preq::fits(w, n)
        && small(k, n)
        && rows(x, k, n).is_some()
        && rows(g, k, n).is_some()
}

/// Whether the kernel's 32-bit indices hold `n` tokens of `k` values (a warp per 128 values).
fn small(k: i64, n: i64) -> bool {
    k <= i32::MAX as i64 && n <= i32::MAX as i64 && n * (k / mmq_preq::BLOCK) <= i32::MAX as i64
}

/// `x · sigmoid(g)` for `n` tokens of `k` values, `x` first normed by `norm` if given (see
/// [`norm_fits`]), quantized in MMQ's layout ([`mmq_preq::q8_1_bytes`]): an F32 node of those
/// bytes. `x` and `g` as [`rows_of`] takes.
pub(crate) fn gate_quantized<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    g: Tensor<'a>,
    norm: Option<Norm<'a>>,
    k: i64,
    n: i64,
) -> Tensor<'a> {
    assert!(k % 512 == 0, "gate_q8: {k} values a token");
    assert!(small(k, n), "gate_q8: {n} tokens of {k} values");
    assert!(rows(x, k, n).is_some(), "gate_q8: x {:?}", x.ne());
    assert!(rows(g, k, n).is_some(), "gate_q8: g {:?}", g.ne());
    let floats = mmq_preq::q8_1_bytes(k, n).div_ceil(4) as i64;
    let (k, n) = (k as i32, n as i32);
    match norm {
        None => ctx.external_with(&GATE_Q8, types::F32, [floats, 1, 1, 1], &[x, g], &[k, n]),
        Some(m) => {
            assert!(
                x.ne()[0] == NORM_ROW && m.w.ne() == [NORM_ROW, 1, 1, 1],
                "gate_q8: norm on {:?}",
                x.ne()
            );
            let eps = m.eps.to_bits() as i32;
            let srcs = [x, g, m.w];
            ctx.external_with(&GATE_Q8, types::F32, [floats, 1, 1, 1], &srcs, &[k, n, eps])
        }
    }
}

/// `mul_mat(w, x · sigmoid(g))` (F32 `[rows, n]`) for `w` Q8_0 `[k, rows]` and `x`, `g` F32
/// holding `k` values for each of `n` tokens (see [`fits`]), `x` normed first by `norm` if given
/// ([`norm_fits`]): ggml's bytes.
pub(crate) fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    g: Tensor<'a>,
    norm: Option<Norm<'a>>,
    n: i64,
) -> Tensor<'a> {
    let xq = gate_quantized(ctx, x, g, norm, w.ne()[0], n);
    mmq_preq::mul_mat_cols(ctx, w, xq, n)
}

/// The sum of `v` as ggml-cuda's `warp_reduce_sum` leaves it (a butterfly over the 32 lanes).
pub(super) fn warp_sum(mut v: [f32; 32]) -> f32 {
    let mut m = 16;
    while m > 0 {
        let w = v;
        for (l, x) in v.iter_mut().enumerate() {
            *x = w[l] + w[l ^ m];
        }
        m /= 2;
    }
    v[0]
}

/// The CPU side of [`Norm`] on one row of 128 values, in the GPU's order (see `norm128` in
/// `src/cuda/gate_q8.cu`); the CPU's square root.
pub(super) fn norm_row(x: &mut [f32], w: &[f32], eps: f32) {
    let s: [f32; 4] =
        std::array::from_fn(|j| warp_sum(std::array::from_fn(|l| x[32 * j + l].powi(2))));
    let mean = ((s[0] + s[2]) + (s[1] + s[3])) / NORM_ROW as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v = scale * *v * w;
    }
}

unsafe fn cpu_gate_q8(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `gate_quantized`, host tensors laid out as `rows_of` read them.
    unsafe {
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let params = crate::ggml::external_params(dst);
        let (k, n) = (params[0] as i64, params[1] as i64);
        let xr = rows_of(x.type_, x.ne, x.nb, k, n).expect("gate_q8: x");
        let gr = rows_of(g.type_, g.ne, g.nb, k, n).expect("gate_q8: g");
        let at = |p: *const f32, r: Rows, t: i64, i: i64| {
            *p.add((t * r.ld + i / r.run * r.stride + i % r.run) as usize)
        };
        let out = dst.data as *mut u8;
        let per = mmq_preq::BLOCK;
        let norm = (!dst.src[2].is_null()).then(|| {
            let w = std::slice::from_raw_parts((*dst.src[2]).data as *const f32, NORM_ROW as usize);
            (w, f32::from_bits(params[2] as u32))
        });
        let mut y = vec![0f32; k as usize];
        for t in (ith as i64..n).step_by(nth) {
            for (i, v) in y.iter_mut().enumerate() {
                *v = at(x.data as *const f32, xr, t, i as i64);
            }
            if let Some((w, eps)) = norm {
                for row in y.chunks_mut(NORM_ROW as usize) {
                    norm_row(row, w, eps);
                }
            }
            for (i, v) in y.iter_mut().enumerate() {
                let z = at(g.data as *const f32, gr, t, i as i64);
                *v *= 1.0 / (1.0 + (-z).exp());
            }
            for b in 0..k / per {
                let blk = out.add(((b * n + t) as usize) * mmq_preq::BLOCK_BYTES);
                for (j, v) in y[(b * per) as usize..][..per as usize]
                    .chunks(GROUP as usize)
                    .enumerate()
                {
                    let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
                    let d_inv = 127.0 / amax;
                    for (i, x) in v.iter().enumerate() {
                        // NaN (an all-zero group) converts to 0, as on the GPU.
                        *blk.add(16 + j * GROUP as usize + i) = ((x * d_inv).round() as i8) as u8;
                    }
                    let d = (1.0 / d_inv).to_ne_bytes();
                    std::ptr::copy_nonoverlapping(d.as_ptr(), blk.add(j * 4), 4);
                }
            }
        }
    }
}

/// `legion_gate_q8_rows` (`src/cuda/gate_q8.cu`).
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuRows {
    p: *const f32,
    ld: i64,
    run: i64,
    stride: i64,
}

/// `legion_gate_q8` (`src/cuda/gate_q8.cu`).
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct GateQ8 {
    x: GpuRows,
    g: GpuRows,
    xq: *mut i8,
    k: i32,
    n: i32,
    nw: *const f32,
    eps: f32,
    ncols: i32,
}

/// `dim3`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// The kernel as the CUDA runtime knows it (its host stub's address), for
    /// `cudaLaunchKernel`.
    static legion_gate_q8_kernel: *const c_void;
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// Warps per CTA (a warp per block of 128 values).
#[cfg(feature = "cuda")]
const WARPS: i64 = 8;

#[cfg(feature = "cuda")]
unsafe fn cuda_gate_q8(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gate_quantized`, device tensors of the stream's device; the
    // kernel reads `p` from the launch's copy of its argument.
    unsafe {
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let params = crate::ggml::external_params(dst);
        let (k, n) = (params[0] as i64, params[1] as i64);
        let gpu = |t: &ffi::ggml_tensor| {
            rows_of(t.type_, t.ne, t.nb, k, n).map(|r| GpuRows {
                p: t.data as *const f32,
                ld: r.ld,
                run: r.run,
                stride: r.stride,
            })
        };
        let (Some(xr), Some(gr)) = (gpu(x), gpu(g)) else {
            return Err("gate_q8: operands".into());
        };
        // The norm's weight and eps (`gate_quantized`), if it has one.
        let (nw, eps) = if dst.src[2].is_null() {
            (std::ptr::null(), 0.0)
        } else {
            let eps = f32::from_bits(params[2] as u32);
            ((*dst.src[2]).data as *const f32, eps)
        };
        let mut p = GateQ8 {
            x: xr,
            g: gr,
            xq: dst.data as *mut i8,
            k: k as i32,
            n: n as i32,
            nw,
            eps,
            ncols: NORM_ROW as i32,
        };
        let warps = n * (k / mmq_preq::BLOCK);
        let grid = Dim3 {
            x: ((warps + WARPS - 1) / WARPS) as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: (WARPS * 32) as u32,
            y: 1,
            z: 1,
        };
        let mut args = [(&raw mut p).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_gate_q8_kernel,
            grid,
            block,
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("gate_q8: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
