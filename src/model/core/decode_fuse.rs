//! `legion serve --decode-fuse` (#773): Qwen3.8-Flash-Next (`qwen4exp`) decode-sized steps (up to
//! [`MAX_TOKENS`] tokens) with fewer, larger kernels in place of ggml-cuda's small ops:
//!
//! - MoE layers: the router's and the shared expert's gate logits from one matmul (the joined
//!   pair of [`super::joined`], up to its 3 tokens), and the layer's tail as one op,
//!   [`moe_tail`]: the router weights, the weighted sum of the routed experts' outputs and the
//!   shared expert gated by its sigmoid, added. That replaces `router_weights`, ggml's `mul` and
//!   adds of the weighted sum (one fused kernel or nine adds, by memory layout), and ggml's
//!   `sigmoid`, `mul` and `add` (and the `cont` of a joined gate logit).
//! - Gated DeltaNet layers: the output norm and its sigmoid gate as one op
//!   ([`super::fused::gated_norm`], `--gated-norm`'s op) instead of ggml's `rms_norm`, `mul`,
//!   `sigmoid` and `mul`.
//!
//! The ops round as `--moe-combine`, `--shexp-gate` and `--gated-norm` do: the weighted sum
//! always as ggml-cuda's fused reduction (the default's graph rounds some layers twice per expert
//! when its tensors overlap in memory, #410), so greedy text can differ from the default's. Off
//! by default. Prompt steps and other models are unchanged.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::Hparams;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static MOE_TAIL: ExternalOp = ExternalOp {
    name: "moe_tail",
    cpu: cpu_moe_tail,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_moe_tail),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs built from now on fuse decode steps' small ops ([`set_decode_fuse`]).
static ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on run Qwen3.8-Flash-Next's decode-sized steps on the
/// fused ops (off by default; `legion serve --decode-fuse` turns it on).
pub fn set_decode_fuse(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Most tokens a fused step takes: decode steps of up to 16 sequences with up to 3 MTP drafts
/// each. Prompt chunks keep their ops.
pub(crate) const MAX_TOKENS: i64 = 64;

/// Most routed experts per token [`moe_tail`] takes (the kernel's shared weights).
pub(crate) const MAX_USED: i64 = 32;

/// Whether a step of `n` tokens of the model `hp` takes the fused ops: switched on, a CUDA
/// build, a `qwen4exp` model, at most [`MAX_TOKENS`] tokens.
pub(crate) fn takes(hp: &Hparams, n: i64) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && crate::model::qwen4::params(hp).is_some()
        && (1..=MAX_TOKENS).contains(&n)
}

/// Whether [`moe_tail`] takes these: F32 experts' outputs `[e, used, n]` (contiguous), router
/// logits `[n_expert, n]` with contiguous rows, I32 ids `[used, n]` (contiguous), a contiguous
/// shared expert's output `[e, n]` and its gate logits `[1, n]` (any row stride), all in GPU
/// memory.
pub(crate) fn moe_tail_fits(
    experts: Tensor<'_>,
    logits: Tensor<'_>,
    ids: Tensor<'_>,
    sh: Tensor<'_>,
    g: Tensor<'_>,
) -> bool {
    let [e, used, n, s3] = experts.ne();
    let f32_rows =
        |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4 && t.nb()[1].is_multiple_of(4);
    s3 == 1
        && (1..=MAX_USED).contains(&used)
        && n <= i32::MAX as i64
        && experts.ty() == types::F32
        && experts.is_contiguous()
        && f32_rows(logits)
        && logits.ne()[1] == n
        && logits.ne()[0] >= used
        && logits.ne()[2] * logits.ne()[3] == 1
        && ids.ty() == types::I32
        && ids.is_contiguous()
        && ids.ne() == [used, n, 1, 1]
        && sh.ty() == types::F32
        && sh.is_contiguous()
        && sh.ne() == [e, n, 1, 1]
        && f32_rows(g)
        && g.ne() == [1, n, 1, 1]
        && [experts, logits, ids, sh, g]
            .iter()
            .all(|t| !t.is_host() && !t.is_meta())
}

/// A MoE layer's output from its routed experts' outputs `experts` (F32 `[e, used, n]`), the
/// router `logits` and the chosen `ids` (as [`super::fused::router`] picks them), the shared
/// expert's output `sh` and its gate logits `g` (see [`moe_tail_fits`]):
/// `Σ_j experts[:, j, t] · w_j + sh · sigmoid(g)`, `[e, n]`, with `w` the router's normalized
/// weights. Rounded as `router_weights`, `moe_combine` and `shexp_gate` round, one after the
/// other.
pub(crate) fn moe_tail<'a>(
    ctx: &'a Context,
    experts: Tensor<'a>,
    logits: Tensor<'a>,
    ids: Tensor<'a>,
    sh: Tensor<'a>,
    g: Tensor<'a>,
) -> Tensor<'a> {
    let [e, _, n, _] = experts.ne();
    ctx.external(
        &MOE_TAIL,
        types::F32,
        [e, n, 1, 1],
        &[experts, logits, ids, sh, g],
    )
}

unsafe fn cpu_moe_tail(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `moe_tail`, host tensors laid out as `moe_tail_fits` checks.
    unsafe {
        let (x, l, ids, sh, g) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let (e, used, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let (xp, sp, out) = (
            x.data as *const f32,
            sh.data as *const f32,
            dst.data as *mut f32,
        );
        let mut w = vec![0f32; used];
        for t in (ith..n).step_by(nth) {
            // The router weights, as `router_weights` computes them.
            let row = (l.data as *const u8).add(t * l.nb[1]) as *const f32;
            let id = (ids.data as *const i32).add(t * used);
            let m = *row.add(*id as usize);
            let mut sum = 0f32;
            for j in 0..used {
                sum += (*row.add(*id.add(j) as usize) - m).exp();
            }
            for (j, wj) in w.iter_mut().enumerate() {
                *wj = (*row.add(*id.add(j) as usize) - m).exp() / sum;
            }
            // The gate, as `shexp_gate` computes it.
            let gv = *((g.data as *const u8).add(t * g.nb[1]) as *const f32);
            let s = 1.0 / (1.0 + (-gv).exp());
            for c in 0..e {
                // The weighted sum, as `moe_combine` rounds it.
                let mut moe = *xp.add(t * used * e + c) * w[0];
                for (k, wk) in w.iter().enumerate().skip(1) {
                    moe = (*xp.add((t * used + k) * e + c)).mul_add(*wk, moe);
                }
                let i = t * e + c;
                *out.add(i) = moe + *sp.add(i) * s;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct MoeTail {
    experts: *const f32,
    logits: *const f32,
    l_ld: i64,
    ids: *const i32,
    sh: *const f32,
    g: *const f32,
    g_ld: i64,
    out: *mut f32,
    n_embd: i64,
    n_used: i32,
    n_tok: i32,
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
    static legion_moe_tail_kernel: *const c_void;
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// Threads per CTA: one column each.
#[cfg(feature = "cuda")]
const THREADS: i64 = 256;

#[cfg(feature = "cuda")]
unsafe fn cuda_moe_tail(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `moe_tail`, device tensors of the stream's device; the kernel reads
    // `p` from the launch's copy of its argument.
    unsafe {
        let (x, l, ids, sh, g) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let mut p = MoeTail {
            experts: x.data as *const f32,
            logits: l.data as *const f32,
            l_ld: (l.nb[1] / 4) as i64,
            ids: ids.data as *const i32,
            sh: sh.data as *const f32,
            g: g.data as *const f32,
            g_ld: (g.nb[1] / 4) as i64,
            out: dst.data as *mut f32,
            n_embd: x.ne[0],
            n_used: x.ne[1] as i32,
            n_tok: x.ne[2] as i32,
        };
        let grid = Dim3 {
            x: x.ne[2] as u32,
            y: (x.ne[0] as u64).div_ceil(THREADS as u64) as u32,
            z: 1,
        };
        let block = Dim3 {
            x: THREADS as u32,
            y: 1,
            z: 1,
        };
        let mut args = [(&raw mut p).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_moe_tail_kernel,
            grid,
            block,
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("moe_tail: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
