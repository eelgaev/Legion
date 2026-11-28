//! `--legion-router` (#1183): a MoE layer's router logits `w @ x` in Legion's own kernel
//! (`src/cuda/legion_router.cu`) for the steps ggml-cuda runs on cuBLAS: F32 router weights of
//! more than 16 tokens (prompt steps, verify steps of 5+ sequences). Steps of up to 16 tokens
//! keep what they run on main ([`super::router_mv`], `dense_mv::mul_mat_wide`, ggml's
//! `mul_mat_vec_f` / `mul_mat_f`).
//!
//! cuBLAS picks its Sgemm algorithm from the token count, so a token's logits depend on the size
//! of the step it is in: the tail of a long prompt alone gets other router logits than the same
//! tokens inside a bigger step (the first such op of a Qwen3.8-Flash-Next prompt step). The
//! kernel computes every output as one fixed chain over K (16-deep tensor-core slices in K
//! order, summed in F32), whatever the step's token count, so a token's logits have the same bits
//! in every step. It reads the router's BF16 copy ([`super::router_bf16`]: every value exact),
//! and splits each activation into two BF16 values (`hi + lo`, ~16 bits of the F32 value): each
//! product is exact, closer to the F32 sums than cuBLAS's TF32 (10 bits). Not cuBLAS's bits, so
//! opt-in (off by default); routers without a BF16 copy keep cuBLAS.
//!
//! The CPU implementation is the reference: F32 sums in K order.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;

static LEGION_ROUTER: ExternalOp = ExternalOp {
    name: "legion_router",
    cpu: cpu_router,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_router),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest tokens the model routes to the op: ggml-cuda runs F32 matmuls of up to 16 tokens on
/// its own kernels (`mul_mat_vec_f`, `mul_mat_f`) and bigger ones on cuBLAS.
pub const MIN_TOKENS: i64 = 17;

/// Whether graphs built from now on take the op ([`set_legion_router`]).
static ON: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on compute the MoE routers' logits of steps over 16 tokens in
/// Legion's kernel instead of ggml-cuda's cuBLAS GEMM (`legion serve --legion-router`, off by
/// default: other bits than cuBLAS's).
pub fn set_legion_router(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_legion_router`].
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Whether the router's BF16 copy `w` over `x` takes the op: switched on, a CUDA build on a GPU
/// that runs it ([`device_supported`]), the weight in that GPU's memory, [`fits`], and at least
/// [`MIN_TOKENS`] tokens.
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    takes_weight(w) && x.ne()[1] >= MIN_TOKENS && fits(w, x)
}

/// Whether the router's BF16 copy `w` takes the op in the steps of [`MIN_TOKENS`] or more
/// tokens ([`takes`] without the step's activations): switched on, a CUDA build on a GPU that
/// runs it, the weight in that GPU's memory and [`fits_weight`].
pub fn takes_weight(w: Tensor<'_>) -> bool {
    weight_miss(w).is_none()
}

/// Why [`takes_weight`] refuses the router's BF16 copy `w`; `None` where it takes it.
pub fn weight_miss(w: Tensor<'_>) -> Option<&'static str> {
    if !on() {
        Some("--legion-router is off")
    } else if !cfg!(feature = "cuda") {
        Some("not a CUDA build")
    } else if w.is_host() {
        Some("the copy is in host memory")
    } else if w.is_meta() {
        Some("the copy is split over GPUs")
    } else if !fits_weight(w) {
        Some("the copy's shape or layout")
    } else if !device_supported() {
        Some("the GPU does not run the kernel")
    } else {
        None
    }
}

/// Whether the op computes `w @ x` for these: a BF16 matrix `w` ([`fits_weight`]) and an F32
/// `x` (`[k_embd, n]`, values contiguous, columns 16-byte aligned), 2-D.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [xk, n, x2, x3] = x.ne();
    let xnb = x.nb();
    fits_weight(w)
        && x.ty() == types::F32
        && x2 == 1
        && x3 == 1
        && xk == w.ne()[0]
        && n >= 1
        && xnb[0] == 4
        && xnb[1] % 16 == 0
}

/// Whether the op takes `w` as its weight: a 2-D BF16 matrix (`[k_embd, n_expert]`, rows
/// contiguous and 16-byte aligned, `k_embd` a multiple of 64).
pub fn fits_weight(w: Tensor<'_>) -> bool {
    let [k_embd, _, w2, w3] = w.ne();
    let wnb = w.nb();
    w.ty() == types::BF16
        && w2 == 1
        && w3 == 1
        && k_embd >= 64
        && k_embd % 64 == 0
        && wnb[0] == 2
        && wnb[1] % 16 == 0
}

/// Whether the current GPU runs the op: compute capability 8.0 or newer (false without CUDA).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_router_mm_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_router_mm_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// The router logits `w @ x` (F32 `[n_expert, n]`) of the BF16 router copy `w` over `x` (see
/// [`fits`]), as `ctx.mul_mat(w, x)` shapes them.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "legion_router: shapes");
    let (n_expert, n) = (w.ne()[1], x.ne()[1]);
    ctx.external(&LEGION_ROUTER, types::F32, [n_expert, n, 1, 1], &[w, x])
}

/// The matmul on the CPU: F32 sums in K order of the widened BF16 weights.
unsafe fn cpu_router(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: the node of `mul_mat`, host tensors laid out as `fits` checks.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k_embd, n_expert, n) = (w.ne[0] as usize, w.ne[1] as usize, x.ne[1] as usize);
        for t in (ith..n).step_by(nth) {
            let xt = std::slice::from_raw_parts(
                (x.data as *const u8).add(t * x.nb[1]) as *const f32,
                k_embd,
            );
            let l =
                std::slice::from_raw_parts_mut((dst.data as *mut f32).add(t * n_expert), n_expert);
            for (e, v) in l.iter_mut().enumerate() {
                let row = std::slice::from_raw_parts(
                    (w.data as *const u8).add(e * w.nb[1]) as *const u16,
                    k_embd,
                );
                *v = row.iter().zip(xt).fold(0f32, |s, (&wv, &xv)| {
                    s + f32::from_bits(u32::from(wv) << 16) * xv
                });
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct RouterMm {
    w: *const u16,
    w_ld: i64,
    x: *const f32,
    x_ld: i64,
    out: *mut f32,
    k_embd: i32,
    n_expert: i32,
    n_tok: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_router_mm_launch(p: *const RouterMm, stream: *mut std::os::raw::c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_router(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: the node of `mul_mat` (sources: w, x), device tensors of the stream's device.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let p = RouterMm {
            w: w.data as *const u16,
            w_ld: (w.nb[1] / 2) as i64,
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            out: dst.data as *mut f32,
            k_embd: w.ne[0] as i32,
            n_expert: w.ne[1] as i32,
            n_tok: x.ne[1] as i32,
        };
        match legion_router_mm_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("legion_router: error {e}")),
        }
    }
}
