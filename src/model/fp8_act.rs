//! Per-token FP8 activations (`--fp8-act a8-dynamic`) written by the op that computes them, so
//! the matmul input is never stored in F32 and read again by [`fp8::quantize_dynamic`].
//!
//! Each op here returns what `quantize_dynamic` of a ggml chain returns, in its packed layout
//! ([`fp8::dynamic_layout`]: the E4M3 codes `[k, n]`, then the tokens' F32 scales), with the same
//! bits on a GPU: the CUDA kernels (`src/cuda/fp8_act.cu`) compute the chain's F32 values in
//! ggml-cuda's order, as [`fused`]'s BF16 ops do, then [`fp8::quantize_token`]'s arithmetic.
//! The model hands one to [`Context::set_shared_node`] for the chain's output under
//! [`fp8::DYNAMIC_KEY`], so every [`fp8::mul_mat_dynamic`] of that activation reads it.
//!
//! - [`rms_norm`]: `mul(rms_norm(x, eps), w)` (the dense models' attention and FFN norms).
//! - [`gate`]: `mul(g, sigmoid(x))` (the attention output gate) or `swiglu_split(x, g)`.
//! - [`gated_norm`]: a gated DeltaNet layer's output norm, `rms_norm(o) * w * gate(z)` per head.
//! - [`quantize_plain`]: `quantize_dynamic` itself in one pass (the row held in registers), for
//!   activations no op above makes.
//!
//! The CPU implementations are references for tests (sums in F32 order, not ggml-cuda's): the
//! CPU graphs keep the chain and `quantize_dynamic`.
//!
//! [`fused`]: crate::model::core::fused

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::fp8;

static FP8_ACT: ExternalOp = ExternalOp {
    name: "fp8_act",
    cpu: cpu_fp8_act,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_fp8_act),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The row an op computes (`kind` in `src/cuda/fp8_act.cu`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum Kind {
    RmsNorm = 1,
    Gate = 2,
    GatedNorm = 3,
}

/// Most values per row the CUDA kernels hold (`MAX_NV` x 1024 threads).
const MAX_K: i64 = 24 * 1024;

/// Whether `x` is F32 with contiguous rows and at most `i32::MAX` rows (2-D) of at most
/// [`MAX_K`] values.
fn rows_fit(x: Tensor<'_>) -> bool {
    let [k, n, x2, x3] = x.ne();
    x.ty() == types::F32
        && x.nb()[0] == 4
        && x2 == 1
        && x3 == 1
        && k <= MAX_K
        && n <= i32::MAX as i64
}

/// A weight vector of `k` F32 values the kernels read in place.
fn vector_fits(w: Tensor<'_>, k: i64) -> bool {
    w.ty() == types::F32
        && w.is_contiguous()
        && w.elements() == k as usize
        && !w.is_host()
        && !w.is_meta()
}

/// Whether this build runs the ops on the current GPU: E4M3 tensor cores (sm_89+), where
/// `a8-dynamic`'s W8A8 GEMMs run.
fn gpu_fits() -> bool {
    cfg!(feature = "cuda") && fp8::fp8_mma()
}

fn node<'a>(
    ctx: &'a Context,
    kind: Kind,
    k: i64,
    n: i64,
    args: &[Tensor<'a>],
    params: [i32; 3],
) -> Tensor<'a> {
    let (_, total) = fp8::dynamic_layout(k, n);
    ctx.external_with(
        &FP8_ACT,
        types::I8,
        [total as i64, 1, 1, 1],
        args,
        &[kind as i32, k as i32, params[0], params[1], params[2]],
    )
}

/// `quantize_dynamic(mul(rms_norm(x, eps), w))` for `x` F32 `[k, n]` (rows contiguous) and `w`
/// F32 `[k]`.
pub(crate) fn rms_norm<'a>(ctx: &'a Context, x: Tensor<'a>, w: Tensor<'a>, eps: f32) -> Tensor<'a> {
    let [k, n, _, _] = x.ne();
    assert_eq!(w.elements(), k as usize, "fp8_act::rms_norm: weight size");
    node(
        ctx,
        Kind::RmsNorm,
        k,
        n,
        &[x, w],
        [eps.to_bits() as i32, 0, 0],
    )
}

/// Whether [`rms_norm`] takes `x` and `w` on this GPU.
pub(crate) fn rms_norm_fits(x: Tensor<'_>, w: Tensor<'_>) -> bool {
    gpu_fits() && rows_fit(x) && vector_fits(w, x.ne()[0])
}

/// `quantize_dynamic` of `mul(g, sigmoid(x))` or (`silu`) `swiglu_split(x, g)`, as
/// [`crate::model::core::fused::gate_bf16`] reads them: `x` F32 `[d, n_head, n]` (columns
/// contiguous, any head and token strides) or `[d * n_head, n]`, `g` F32 `[d * n_head, n]`,
/// rows contiguous.
pub(crate) fn gate<'a>(ctx: &'a Context, x: Tensor<'a>, g: Tensor<'a>, silu: bool) -> Tensor<'a> {
    let [k, n, _, _] = g.ne();
    assert_eq!(x.elements(), (k * n) as usize, "fp8_act::gate: shapes");
    node(ctx, Kind::Gate, k, n, &[x, g], [0, i32::from(silu), 0])
}

/// Whether [`gate`] takes `x` and `g` on this GPU.
pub(crate) fn gate_fits(x: Tensor<'_>, g: Tensor<'_>) -> bool {
    let [k, n, _, _] = g.ne();
    let [d, xh, xn, x3] = x.ne();
    let x_ok = if xn == 1 && x3 == 1 {
        d == k && xh == n
    } else {
        x3 == 1 && xn == n && d * xh == k
    };
    gpu_fits() && rows_fit(g) && x_ok && x.ty() == types::F32 && x.nb()[0] == 4
}

/// `quantize_dynamic` of a gated DeltaNet layer's output norm, `rms_norm(o) * w * gate(z)` per
/// `head_v`-wide head (`gate` sigmoid or SiLU), as
/// [`crate::model::core::fused::gated_norm`] computes it: `o` and `z` F32 `[head_v * n_v,
/// n]` (rows contiguous), `w` F32 `[head_v]`.
pub(crate) fn gated_norm<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    w: Tensor<'a>,
    z: Tensor<'a>,
    head_v: i64,
    eps: f32,
    silu: bool,
) -> Tensor<'a> {
    let [k, n, _, _] = o.ne();
    assert_eq!(z.ne()[..2], [k, n], "fp8_act::gated_norm: z shape");
    assert_eq!(k % head_v, 0, "fp8_act::gated_norm: heads");
    node(
        ctx,
        Kind::GatedNorm,
        k,
        n,
        &[o, w, z],
        [eps.to_bits() as i32, i32::from(silu), head_v as i32],
    )
}

/// Whether [`gated_norm`] takes these tensors on this GPU: heads of 32, 64, 128 or 256 values.
pub(crate) fn gated_norm_fits(o: Tensor<'_>, w: Tensor<'_>, z: Tensor<'_>, head_v: i64) -> bool {
    gpu_fits()
        && matches!(head_v, 32 | 64 | 128 | 256)
        && rows_fit(o)
        && rows_fit(z)
        && z.ne()[..2] == o.ne()[..2]
        && o.ne()[0] % head_v == 0
        && vector_fits(w, head_v)
}

/// Gives every [`fp8::mul_mat_dynamic`] of `a` the packed activations `q` (one of the ops above,
/// computed from `a`'s inputs) instead of quantizing `a`.
pub(crate) fn share<'a>(ctx: &'a Context, a: Tensor<'a>, q: Tensor<'a>) {
    let [k, n1, n2, n3] = a.ne();
    assert_eq!(
        q.elements(),
        fp8::dynamic_layout(k, n1 * n2 * n3).1,
        "fp8_act::share: not the packed activations of {:?}",
        a.ne()
    );
    ctx.set_shared_node(a, fp8::DYNAMIC_KEY, q);
}

/// The tokens of a node made by this module: its `k * n` codes and `n` scales hold `n (k + 4)`
/// bytes.
fn tokens(dst: &ffi::ggml_tensor) -> i64 {
    let k = crate::ggml::external_params(dst)[1] as i64;
    dst.ne[0] / (k + 4)
}

/// One token's activation row (`k` values) as the chain computes it on the CPU.
unsafe fn cpu_row(dst: &ffi::ggml_tensor, t: usize, row: &mut [f32]) {
    let p = crate::ggml::external_params(dst);
    let kind = p[0];
    let k = row.len();
    // SAFETY: a node made by this module (`kind`, its sources as made there), host tensors.
    unsafe {
        let x = &*dst.src[0];
        let xrow = |t: usize| (x.data as *const u8).add(t * x.nb[1]) as *const f32;
        if kind == Kind::RmsNorm as i32 {
            let eps = f32::from_bits(p[2] as u32);
            let w = std::slice::from_raw_parts((*dst.src[1]).data as *const f32, k);
            let xs = std::slice::from_raw_parts(xrow(t), k);
            let mean = xs.iter().map(|v| v * v).sum::<f32>() / k as f32;
            let scale = 1.0 / (mean + eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = scale * xs[i] * w[i];
            }
        } else if kind == Kind::Gate as i32 {
            let silu = p[3] != 0;
            let g = &*dst.src[1];
            let d = x.ne[0] as usize;
            let (x_hd, x_ld) = if x.ne[2] == 1 && x.ne[3] == 1 {
                (0, x.nb[1] / 4)
            } else {
                (x.nb[1] / 4, x.nb[2] / 4)
            };
            let xp = x.data as *const f32;
            let gp = (g.data as *const u8).add(t * g.nb[1]) as *const f32;
            for (i, v) in row.iter_mut().enumerate() {
                let xv = *xp.add(t * x_ld + (i / d) * x_hd + i % d);
                let e = 1.0 + (-xv).exp();
                let a = if silu { xv / e } else { 1.0 / e };
                *v = a * *gp.add(i);
            }
        } else {
            assert_eq!(kind, Kind::GatedNorm as i32, "fp8_act: kind {kind}");
            let (eps, silu, head_v) = (f32::from_bits(p[2] as u32), p[3] != 0, p[4] as usize);
            let w = std::slice::from_raw_parts((*dst.src[1]).data as *const f32, head_v);
            let z = &*dst.src[2];
            let xs = std::slice::from_raw_parts(xrow(t), k);
            let zs =
                std::slice::from_raw_parts((z.data as *const u8).add(t * z.nb[1]) as *const f32, k);
            for h in 0..k / head_v {
                let cols = h * head_v..(h + 1) * head_v;
                let mean = xs[cols.clone()].iter().map(|v| v * v).sum::<f32>() / head_v as f32;
                let scale = 1.0 / (mean + eps).sqrt();
                for i in cols {
                    let zv = zs[i];
                    let e = 1.0 + (-zv).exp();
                    let g = if silu { zv / e } else { 1.0 / e };
                    row[i] = scale * xs[i] * w[i % head_v] * g;
                }
            }
        }
    }
}

unsafe fn cpu_fp8_act(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by this module, host tensors.
    unsafe {
        let k = crate::ggml::external_params(dst)[1] as usize;
        let n = tokens(dst) as usize;
        let (codes, _) = fp8::dynamic_layout(k as i64, n as i64);
        let out = dst.data as *mut u8;
        let scales = out.add(codes) as *mut f32;
        let mut row = vec![0f32; k];
        for t in (ith..n).step_by(nth) {
            cpu_row(dst, t, &mut row);
            let (q, s) = fp8::quantize_token(&row);
            std::ptr::copy_nonoverlapping(q.as_ptr(), out.add(t * k), k);
            scales.add(t).write_unaligned(s);
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct Fp8Act {
    kind: i32,
    x: *const f32,
    x_ld: i64,
    x_hd: i64,
    w: *const f32,
    g: *const f32,
    g_ld: i64,
    out: *mut u8,
    scales: *mut f32,
    k: i32,
    n: i32,
    d: i32,
    silu: i32,
    eps: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_fp8_act_launch(p: *const Fp8Act, stream: *mut c_void) -> i32;
}

/// [`fp8::quantize_dynamic`]'s CUDA kernel in one pass: `Some` of the launch's result, `None`
/// where the row does not fit in registers (the caller runs the two-pass kernel).
///
/// # Safety
/// Device pointers on the stream's device: `x` `[k, n]` F32 (rows `x_ld` floats apart), `out`
/// `k * n` codes, `scales` `n` F32.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn quantize_plain(
    x: *const f32,
    x_ld: i64,
    out: *mut u8,
    scales: *mut f32,
    k: i64,
    n: i64,
    stream: *mut c_void,
) -> Option<i32> {
    if k > MAX_K || n > i32::MAX as i64 {
        return None;
    }
    let p = Fp8Act {
        kind: 0,
        x,
        x_ld,
        x_hd: 0,
        w: std::ptr::null(),
        g: std::ptr::null(),
        g_ld: 0,
        out,
        scales,
        k: k as i32,
        n: n as i32,
        d: 0,
        silu: 0,
        eps: 0.0,
    };
    // SAFETY: the caller's pointers, as documented.
    match unsafe { legion_fp8_act_launch(&p, stream) } {
        1 => None,
        r => Some(r),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_fp8_act(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by this module, device tensors of the stream's device.
    unsafe {
        let p = crate::ggml::external_params(dst);
        let x = &*dst.src[0];
        let (k, n) = (p[1], tokens(dst));
        let (codes, _) = fp8::dynamic_layout(k as i64, n);
        let out = dst.data as *mut u8;
        let mut a = Fp8Act {
            kind: p[0],
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            x_hd: 0,
            w: std::ptr::null(),
            g: std::ptr::null(),
            g_ld: 0,
            out,
            scales: out.add(codes) as *mut f32,
            k,
            n: n as i32,
            d: 0,
            silu: 0,
            eps: 0.0,
        };
        if p[0] == Kind::RmsNorm as i32 {
            a.w = (*dst.src[1]).data as *const f32;
            a.eps = f32::from_bits(p[2] as u32);
        } else if p[0] == Kind::Gate as i32 {
            let g = &*dst.src[1];
            (a.x_hd, a.x_ld) = if x.ne[2] == 1 && x.ne[3] == 1 {
                (0, (x.nb[1] / 4) as i64)
            } else {
                ((x.nb[1] / 4) as i64, (x.nb[2] / 4) as i64)
            };
            a.d = x.ne[0] as i32;
            a.g = g.data as *const f32;
            a.g_ld = (g.nb[1] / 4) as i64;
            a.silu = p[3];
        } else {
            let z = &*dst.src[2];
            a.w = (*dst.src[1]).data as *const f32;
            a.g = z.data as *const f32;
            a.g_ld = (z.nb[1] / 4) as i64;
            a.eps = f32::from_bits(p[2] as u32);
            a.silu = p[3];
            a.d = p[4];
        }
        match legion_fp8_act_launch(&a, stream) {
            0 => Ok(()),
            1 => Err(format!(
                "fp8_act: kind {} shape {:?} unsupported",
                p[0], x.ne
            )),
            e => Err(format!("fp8_act: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
