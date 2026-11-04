//! qwen4exp's hyper-connection mixer chain on a GPU without ggml-cuda's elementwise ops
//! (`src/cuda/hc_chain.cu`, #836: all compute in Legion kernels):
//!
//! - [`lo_act`]: the low-rank gate's `silu(scale(lo, 1/hc))` in every mixer's tail, one kernel in
//!   place of ggml's two (both in place on `lo` there too).
//! - [`combine`]: `hc_post(out, res, 2 · sigmoid(inject / hc))`, where the chain combines the
//!   block output into the streams (decode and verify steps of BF16 mixers without
//!   `--hc-mix-bf16`, the MTP head): one kernel in place of ggml's scale, sigmoid, scale and
//!   hc_post.
//! - [`rms_mul`]: the mixer's grouped norm there, `rms_norm(x) · gamma`, one kernel in place of
//!   ggml's rms_norm and mul.
//!
//! Each kernel is ggml-cuda's expressions in ggml's order with the same fast-math flags, so the
//! same bits; and each node takes the memory ggml's allocator gives the ops it replaces (the
//! in-place ones in place, the others one tensor of the result's shape), so the graph's
//! allocation stays ggml's. The CPU graph keeps ggml's ops (the reference); the CPU bodies here
//! are references for the tests. On by default; `legion serve --hc-chain-ggml` keeps ggml's ops.
//! The matmuls (down, up, inject) stay ggml's.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static HC_LO_ACT: ExternalOp = ExternalOp {
    name: "hc_lo_act",
    cpu: cpu_lo_act,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_lo_act),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static HC_COMBINE: ExternalOp = ExternalOp {
    name: "hc_combine",
    cpu: cpu_combine,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_combine),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static HC_RMS_MUL: ExternalOp = ExternalOp {
    name: "hc_rms_mul",
    cpu: cpu_rms_mul,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_rms_mul),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs built from now on run the chain's ops as this module's (on by default).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on run the mixer chain's scale+SiLU, combine and grouped
/// norm as Legion's kernels on a GPU (`legion serve --hc-chain-ggml` turns it off; the same
/// bits either way).
pub fn set_hc_chain(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_hc_chain`].
pub(super) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Whether the mixer whose weight `w` is runs on one GPU: `w` in device memory, not host
/// memory and not split over devices (tensor parallelism), as the other mixer ops check.
pub(super) fn on_gpu(w: Tensor<'_>) -> bool {
    !w.is_host() && !w.is_meta()
}

/// Whether this module's ops may take a step: on, built with CUDA, `on_gpu` (from
/// [`on_gpu`]).
fn gpu(on_gpu: bool) -> bool {
    on() && cfg!(feature = "cuda") && on_gpu
}

/// Whether `t` is a plain node of the graph (not a view, not a leaf), which ggml's allocator
/// lets its only reader take over in place.
fn plain_node(t: Tensor<'_>) -> bool {
    // SAFETY: a live tensor; the fields are read only.
    unsafe {
        let r = t.raw();
        (*r).view_src.is_null() && (*r).op != ffi::ggml_op_GGML_OP_NONE
    }
}

/// `silu(scale(lo, scale))` as ggml computes it (`lo` F32, contiguous); on a GPU one op in
/// place on `lo`, as ggml's two run there. `lo` must have no other reader.
pub(super) fn lo_act<'a>(ctx: &'a Context, lo: Tensor<'a>, scale: f32, on_gpu: bool) -> Tensor<'a> {
    if gpu(on_gpu) && lo.ty() == types::F32 && lo.is_contiguous() && plain_node(lo) {
        return ctx.external_inplace(&HC_LO_ACT, lo, &[], &[scale.to_bits() as i32]);
    }
    ctx.silu(ctx.scale(lo, scale))
}

/// Whether [`combine`] takes these operands: `out` F32 `[e, n]`, `res` F32 `[e, hc, n]`,
/// `inject` F32 `[hc, n]` a plain node (ggml's scales and sigmoid run in place on it), rows
/// contiguous.
pub(super) fn combine_fits(
    out: Tensor<'_>,
    res: Tensor<'_>,
    inject: Tensor<'_>,
    on_gpu: bool,
) -> bool {
    let [e, hc, n, one] = res.ne();
    let f32_rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
    gpu(on_gpu)
        && one == 1
        && out.ne() == [e, n, 1, 1]
        && inject.ne() == [hc, n, 1, 1]
        && f32_rows(out)
        && f32_rows(res)
        && f32_rows(inject)
        && plain_node(inject)
}

/// `hc_post(out, res, scale(sigmoid(scale(inject, 1/hc)), 2))`, ggml's combine of the block
/// output into the streams: F32 `[e, hc, n]`, contiguous. Needs [`combine_fits`].
pub(super) fn combine<'a>(
    ctx: &'a Context,
    out: Tensor<'a>,
    res: Tensor<'a>,
    inject: Tensor<'a>,
) -> Tensor<'a> {
    let [e, hc, n, _] = res.ne();
    let inv = 1.0 / hc as f32;
    ctx.external_with(
        &HC_COMBINE,
        types::F32,
        [e, hc, n, 1],
        &[out, res, inject],
        &[inv.to_bits() as i32, 2f32.to_bits() as i32],
    )
}

/// Whether [`rms_mul`] takes `x` (F32 `[e, hc, n]`, rows contiguous) and `gamma` (F32, `e · hc`
/// values, contiguous).
pub(super) fn rms_mul_fits(x: Tensor<'_>, gamma: Tensor<'_>, on_gpu: bool) -> bool {
    let [e, hc, n, one] = x.ne();
    gpu(on_gpu)
        && one == 1
        && x.ty() == types::F32
        && x.nb()[0] == 4
        && gamma.ty() == types::F32
        && gamma.is_contiguous()
        && gamma.elements() == (e * hc) as usize
        && e <= i32::MAX as i64
        && hc <= 65535
        && n <= 65535
}

/// `mul(rms_norm(x, eps), gamma)` with `gamma` as `[e, hc]`, as ggml computes it: F32
/// `[e, hc, n]`, contiguous. Needs [`rms_mul_fits`].
pub(super) fn rms_mul<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    gamma: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    ctx.external_with(
        &HC_RMS_MUL,
        types::F32,
        x.ne(),
        &[x, gamma],
        &[eps.to_bits() as i32],
    )
}

fn param(dst: &ffi::ggml_tensor, i: usize) -> f32 {
    f32::from_bits(crate::ggml::external_params(dst)[i] as u32)
}

/// A float at `t`'s byte offset `off`.
///
/// # Safety
/// `off` is within `t`'s host data.
unsafe fn at(t: &ffi::ggml_tensor, off: usize) -> f32 {
    unsafe { *((t.data as *const u8).add(off) as *const f32) }
}

unsafe fn cpu_lo_act(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `lo_act`: dst is a view of src[0], F32 contiguous, host memory.
    unsafe {
        let scale = param(dst, 0);
        let n = ffi::ggml_nelements(dst) as usize;
        let x = (*dst.src[0]).data as *const f32;
        let y = dst.data as *mut f32;
        for i in (ith..n).step_by(nth) {
            let s = scale * *x.add(i);
            *y.add(i) = s / (1.0 + (-s).exp());
        }
    }
}

unsafe fn cpu_combine(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `combine`, host tensors of the checked shapes.
    unsafe {
        let (out, res, inj) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (s1, s2) = (param(dst, 0), param(dst, 1));
        let (e, hc, n) = (res.ne[0] as usize, res.ne[1] as usize, res.ne[2] as usize);
        let y = dst.data as *mut f32;
        for t in (ith..n).step_by(nth) {
            for h in 0..hc {
                let a = s1 * at(inj, h * 4 + t * inj.nb[1]);
                let post = s2 * (1.0 / (1.0 + (-a).exp()));
                for i in 0..e {
                    let x = at(out, i * 4 + t * out.nb[1]);
                    let r = at(res, i * 4 + h * res.nb[1] + t * res.nb[2]);
                    *y.add(i + h * e + t * e * hc) = x.mul_add(post, r);
                }
            }
        }
    }
}

unsafe fn cpu_rms_mul(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `rms_mul`, host tensors of the checked shapes.
    unsafe {
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let eps = param(dst, 0);
        let (e, hc, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let gp = g.data as *const f32;
        let y = dst.data as *mut f32;
        for r in (ith..hc * n).step_by(nth) {
            let (h, t) = (r % hc, r / hc);
            let row = |i: usize| at(x, i * 4 + h * x.nb[1] + t * x.nb[2]);
            let sum: f64 = (0..e).map(|i| f64::from(row(i)) * f64::from(row(i))).sum();
            let scale = 1.0 / ((sum / e as f64) as f32 + eps).sqrt();
            for i in 0..e {
                *y.add(i + h * e + t * e * hc) = (scale * row(i)) * *gp.add(i + h * e);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct HcChain {
    x: *const f32,
    res: *const f32,
    inj: *const f32,
    g: *const f32,
    dst: *mut f32,
    e: i64,
    hc: i64,
    n: i64,
    x1: i64,
    x2: i64,
    r1: i64,
    r2: i64,
    i1: i64,
    scale: f32,
    bias: f32,
    scale2: f32,
}

#[cfg(feature = "cuda")]
impl HcChain {
    fn empty(dst: *mut f32) -> Self {
        HcChain {
            x: std::ptr::null(),
            res: std::ptr::null(),
            inj: std::ptr::null(),
            g: std::ptr::null(),
            dst,
            e: 0,
            hc: 0,
            n: 0,
            x1: 0,
            x2: 0,
            r1: 0,
            r2: 0,
            i1: 0,
            scale: 0.0,
            // ggml_scale's bias (a kernel argument there too).
            bias: 0.0,
            scale2: 0.0,
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_lo_act_launch(p: *const HcChain, stream: *mut c_void) -> i32;
    fn legion_hc_combine_launch(p: *const HcChain, stream: *mut c_void) -> i32;
    fn legion_hc_rms_mul_launch(p: *const HcChain, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
fn launched(name: &str, e: i32) -> Result<(), String> {
    match e {
        0 => Ok(()),
        e => Err(format!("{name}: error {e}")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_lo_act(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `lo_act`, device tensors of the stream's device.
    unsafe {
        let mut p = HcChain::empty(dst.data as *mut f32);
        p.x = (*dst.src[0]).data as *const f32;
        p.n = ffi::ggml_nelements(dst);
        p.scale = param(dst, 0);
        launched("hc_lo_act", legion_hc_lo_act_launch(&p, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_combine(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `combine`, device tensors of the stream's device.
    unsafe {
        let (out, res, inj) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let mut p = HcChain::empty(dst.data as *mut f32);
        p.x = out.data as *const f32;
        p.res = res.data as *const f32;
        p.inj = inj.data as *const f32;
        (p.e, p.hc, p.n) = (res.ne[0], res.ne[1], res.ne[2]);
        p.x1 = (out.nb[1] / 4) as i64;
        (p.r1, p.r2) = ((res.nb[1] / 4) as i64, (res.nb[2] / 4) as i64);
        p.i1 = (inj.nb[1] / 4) as i64;
        (p.scale, p.scale2) = (param(dst, 0), param(dst, 1));
        launched("hc_combine", legion_hc_combine_launch(&p, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_rms_mul(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `rms_mul`, device tensors of the stream's device.
    unsafe {
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let mut p = HcChain::empty(dst.data as *mut f32);
        p.x = x.data as *const f32;
        p.g = g.data as *const f32;
        (p.e, p.hc, p.n) = (x.ne[0], x.ne[1], x.ne[2]);
        (p.x1, p.x2) = ((x.nb[1] / 4) as i64, (x.nb[2] / 4) as i64);
        p.scale = param(dst, 0);
        launched("hc_rms_mul", legion_hc_rms_mul_launch(&p, stream))
    }
}

#[cfg(test)]
mod tests;
