//! A MoE layer's router for steps of 1-3 tokens as one external op (#836): the logits
//! `w @ x` of the F32 router weight, the `k` best experts per token and their weights, which
//! ggml's `mul_mat` and [`super::fused::router`]'s two ops (`router_topk`, `router_weights`)
//! compute otherwise: one launch per MoE layer instead of three.
//!
//! ggml-cuda runs F32 matmuls of up to 3 tokens on `mul_mat_vec_f` (compute capability 8.0+).
//! The op's kernel (`src/cuda/router_mv.cu`) does its arithmetic in its order (one CTA per row,
//! ggml's block size, an FMA per value, warp then block sums) and the selection with the
//! fused router's own device code (`src/cuda/router_select.cuh`), so every output has the bits
//! of the three launches it replaces. The last CTA of the grid to finish selects. On a GB10 the
//! kernel is a programmatic dependent launch: its weight rows start loading while the kernel
//! ahead still runs.
//!
//! On by default where it applies (CUDA, compute capability 9.0+); `legion serve
//! --no-router-mv` keeps the three launches. The CPU implementation is the reference (F32 sums
//! in row order).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;

// Three nodes where ggml's matmul and the two router ops were, of their types, shapes and
// sources (plus the matmul's for the last), so the graph allocator lays out the step's memory
// as it does on main and ggml-cuda fuses its own nodes as there (fusions depend on node order
// and memory overlap). On CUDA the last one's kernel writes all three; the first two launch
// nothing.
static ROUTER_MV_LOGITS: ExternalOp = ExternalOp {
    name: "router_mv_logits",
    cpu: cpu_logits,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_written_by_router_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ROUTER_MV_IDS: ExternalOp = ExternalOp {
    name: "router_mv_ids",
    cpu: cpu_ids,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_written_by_router_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ROUTER_MV: ExternalOp = ExternalOp {
    name: "router_mv",
    cpu: cpu_weights,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_router_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most tokens the op takes: ggml-cuda's `mul_mat_vec_f` range for F32 weights on tensor-core
/// GPUs (from 4 on it runs `mul_mat_f<float>`).
pub const MAX_TOKENS: i64 = 3;

/// Completion counters of the CUDA kernel (`SLOTS` in `router_mv.cu`): one per layer.
pub const SLOTS: usize = 1024;

/// Whether graphs built from now on may take the op ([`set_router_mv`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether graphs built from now on run MoE routers of 1-3 tokens as one op ([`router_mv`]):
/// on by default; `legion serve --no-router-mv` keeps ggml's matmul and the two selection ops
/// (for comparisons).
pub fn set_router_mv(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether the router weight `w` (F32 `[k_embd, n_expert]`) of layer `il` over `x` (F32
/// `[k_embd, n]`) takes the op for the top `k`: switched on, a CUDA build on a GPU that runs it
/// ([`device_supported`]), the weight in that GPU's memory, and [`fits`].
pub fn takes(w: Tensor<'_>, x: Tensor<'_>, k: i64, il: usize) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && il < SLOTS
        && !w.is_host()
        && !w.is_meta()
        && fits(w, x, k)
        && device_supported()
}

/// Whether the op computes these as ggml-cuda's `mul_mat_vec_f` and the fused router would:
/// an F32 matrix `w` (`[k_embd, n_expert]`, `k_embd` even, rows contiguous and 8-byte aligned,
/// at most [`super::fused::ROUTER_MAX_EXPERTS`] of them) and an F32 `x` (`[k_embd, n]`, values
/// contiguous, columns 8-byte aligned) of 1 to [`MAX_TOKENS`] tokens, `1 <= k <= n_expert`.
/// Or a BF16 `w` (rows 4-byte aligned): the BF16 copy of an F32 router whose values are all
/// exact in BF16 ([`super::router_bf16`]), which the op widens back to those F32 values.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>, k: i64) -> bool {
    let [k_embd, n_expert, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let (wnb, xnb) = (w.nb(), x.nb());
    let elem = match w.ty() {
        types::F32 => 4,
        types::BF16 => 2,
        _ => return false,
    };
    x.ty() == types::F32
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k_embd
        && k_embd % 2 == 0
        && n_expert <= super::fused::ROUTER_MAX_EXPERTS
        && (1..=n_expert).contains(&k)
        && wnb[0] == elem
        && wnb[1..].iter().all(|b| b % (2 * elem) == 0)
        && xnb[0] == 4
        && xnb[1] % 8 == 0
        && (1..=MAX_TOKENS).contains(&n)
}

/// A value of row `e` of the router weight `w` (F32, or the BF16 copy [`fits`] takes) at `c`.
///
/// # Safety
/// `w` is a host tensor laid out as [`fits`] checks; `e` and `c` are in range.
unsafe fn weight_at(w: &ffi::ggml_tensor, e: usize, c: usize) -> f32 {
    // SAFETY: the caller's.
    unsafe {
        let row = (w.data as *const u8).add(e * w.nb[1]);
        if w.type_ == types::BF16 {
            f32::from_bits(u32::from(*(row as *const u16).add(c)) << 16)
        } else {
            *(row as *const f32).add(c)
        }
    }
}

/// `--hopper-pdl` ([`set_hopper_pdl`]).
static HOPPER_PDL: AtomicBool = AtomicBool::new(false);

/// Whether Legion's programmatic dependent launches (each kernel set up while the one ahead of
/// it still runs, then waiting for it in `griddepcontrol.wait`; this op, the GLM and Qwen4 decode
/// kernels of `src/cuda/launch_pdl.cuh`, and the [`super::q8_group`] chain, which 9.0 does not
/// run today: [`super::q8_group::cc_supported`]) take compute capability
/// 9.0 (H100, H200, GH200) too, as they take 12.x (`legion serve --hopper-pdl`, #1215). Off by
/// default: untested on hardware. The CUDA side launches early only kernels whose binary code is
/// sm_90+ (the wait is compiled for sm_90+ only); other GPUs never change.
pub fn set_hopper_pdl(on: bool) {
    HOPPER_PDL.store(on, Ordering::Relaxed);
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_set_hopper_pdl(on: i32);
        }
        // SAFETY: stores a flag.
        unsafe { legion_set_hopper_pdl(i32::from(on)) }
    }
}

/// [`set_hopper_pdl`]'s state.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn hopper_pdl() -> bool {
    HOPPER_PDL.load(Ordering::Relaxed)
}

/// Whether a programmatic dependent launch takes a device of compute capability `cc` (`10 *
/// major + minor`): 12.x, and 9.0 under [`set_hopper_pdl`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn pdl_cc(cc: i32) -> bool {
    pdl_cc_with(cc, hopper_pdl())
}

/// [`pdl_cc`] with `--hopper-pdl` given as `hopper`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pdl_cc_with(cc: i32, hopper: bool) -> bool {
    cc / 10 == 12 || (cc == 90 && hopper)
}

/// Whether the current GPU runs the op: compute capability 9.0 or newer (false without CUDA).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_router_mv_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_router_mv_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Layer `il`'s routing of `x` (F32 `[k_embd, n]`) by the router weight `w` (F32 `[k_embd,
/// n_expert]`, see [`fits`]): `(logits, ids, weights)`, the logits `w @ x` (F32 `[n_expert,
/// n]`), the `k` best experts per token (I32 `[k, n]`, descending logit, ties to the lower id)
/// and their softmax probabilities normalized over the `k` (F32 `[k, n]`), as
/// `fused::router(mul_mat(w, x), k)` gives them. `il` (below [`SLOTS`]) picks the kernel's
/// completion counter: layers that may run at the same time on one GPU need different ones.
pub fn router_mv<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    k: i64,
    il: usize,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    assert!(fits(w, x, k), "router_mv: shapes");
    assert!(il < SLOTS, "router_mv: layer {il}");
    let n_expert = w.ne()[1];
    let n = x.ne()[1];
    // As `ctx.mul_mat(w, x)` and `fused::router` make them (see `ROUTER_MV_LOGITS`).
    let logits = ctx.external(&ROUTER_MV_LOGITS, types::F32, [n_expert, n, 1, 1], &[w, x]);
    let ids = ctx.external(&ROUTER_MV_IDS, types::I32, [k, n, 1, 1], &[logits]);
    let wts = ctx.external_with(
        &ROUTER_MV,
        types::F32,
        [k, n, 1, 1],
        &[logits, ids, w, x],
        &[il as i32],
    );
    (logits, ids, wts)
}

/// The logits node's matmul on the CPU: F32 sums in row order.
unsafe fn cpu_logits(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: the logits node of `router_mv`, host tensors laid out as `fits` checks.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k_embd, n_expert, n) = (w.ne[0] as usize, w.ne[1] as usize, x.ne[1] as usize);
        for t in (ith..n).step_by(nth) {
            let xt = (x.data as *const u8).add(t * x.nb[1]) as *const f32;
            let l =
                std::slice::from_raw_parts_mut((dst.data as *mut f32).add(t * n_expert), n_expert);
            for (e, v) in l.iter_mut().enumerate() {
                *v = (0..k_embd).fold(0f32, |s, c| s + weight_at(w, e, c) * *xt.add(c));
            }
        }
    }
}

/// The ids node on the CPU, as `router_topk` (fused.rs): descending logit, ties to the lower id.
unsafe fn cpu_ids(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: the ids node of `router_mv`, host tensors.
    unsafe {
        let l = &*dst.src[0];
        let (n_expert, n, k) = (l.ne[0] as usize, l.ne[1] as usize, dst.ne[0] as usize);
        for t in (ith..n).step_by(nth) {
            let row =
                std::slice::from_raw_parts((l.data as *const f32).add(t * n_expert), n_expert);
            let mut order: Vec<usize> = (0..n_expert).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let out = (dst.data as *mut i32).add(t * k);
            for (j, &e) in order[..k].iter().enumerate() {
                *out.add(j) = e as i32;
            }
        }
    }
}

/// The weights node on the CPU, as `router_weights` (fused.rs).
unsafe fn cpu_weights(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: the weights node of `router_mv`, host tensors.
    unsafe {
        let (l, ids) = (&*dst.src[0], &*dst.src[1]);
        let (n_expert, n, k) = (l.ne[0] as usize, l.ne[1] as usize, dst.ne[0] as usize);
        for t in (ith..n).step_by(nth) {
            let row = (l.data as *const f32).add(t * n_expert);
            let ids = (ids.data as *const i32).add(t * k);
            let m = *row.add(*ids as usize);
            let mut sum = 0f32;
            for j in 0..k {
                sum += (*row.add(*ids.add(j) as usize) - m).exp();
            }
            let out = (dst.data as *mut f32).add(t * k);
            for j in 0..k {
                *out.add(j) = (*row.add(*ids.add(j) as usize) - m).exp() / sum;
            }
        }
    }
}

/// The logits and ids nodes on CUDA: the weights node's kernel writes them.
#[cfg(feature = "cuda")]
unsafe fn cuda_written_by_router_mv(
    _dst: &ffi::ggml_tensor,
    _stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct RouterMv {
    w: *const f32,
    w_ld: i64,
    x: *const f32,
    x_ld: i64,
    logits: *mut f32,
    ids: *mut i32,
    wts: *mut f32,
    k_embd: i32,
    n_expert: i32,
    k: i32,
    n_tok: i32,
    slot: i32,
    wb: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_router_mv_launch(p: *const RouterMv, stream: *mut std::os::raw::c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_router_mv(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: the weights node of `router_mv` (sources: logits, ids, w, x), device tensors of
    // the stream's device.
    unsafe {
        let (l, ids) = (&*dst.src[0], &*dst.src[1]);
        let (w, x) = (&*dst.src[2], &*dst.src[3]);
        let wb = w.type_ == types::BF16;
        let p = RouterMv {
            w: w.data as *const f32,
            // In values (BF16 ones for the copy).
            w_ld: (w.nb[1] / if wb { 2 } else { 4 }) as i64,
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            logits: l.data as *mut f32,
            ids: ids.data as *mut i32,
            wts: dst.data as *mut f32,
            k_embd: w.ne[0] as i32,
            n_expert: w.ne[1] as i32,
            k: dst.ne[0] as i32,
            n_tok: x.ne[1] as i32,
            slot: crate::ggml::external_params(dst)[0],
            wb: i32::from(wb),
        };
        match legion_router_mv_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("router_mv: error {e}")),
        }
    }
}
