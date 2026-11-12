//! Small BF16 matmuls of 1-8 tokens that share one input as one external op (#689; the engine
//! gives it 8, [`TAKES_MIN_TOKENS`]): a gated
//! DeltaNet layer's alpha and beta gate projections (48 rows each in Qwen3.8-27B and
//! Qwen3.8-Flash-Next), which ggml-cuda runs as two `mul_mat_vec_f` launches.
//!
//! ggml-cuda runs BF16 matmuls of up to 8 tokens whose row count is not a multiple of 32 on
//! `mul_mat_vec_f`: one CTA per row, its block size from the row length, an FMA per value, then
//! warp and block sums. A joined `[k, 96]` weight would go to `mul_mat_f` (other bits), so the
//! op's kernel (`src/cuda/mvf_bf16.cu`) does `mul_mat_vec_f`'s arithmetic in its order for every
//! row of the weights it is given, in one launch: every output has the bits of the launch it
//! replaces. Each thread loads all its weight words before its first FMA (ggml's loop waits for
//! each load in turn: on a GB10 a 48-row launch of 8 tokens takes ~12 us for ~0.5 MB), and on
//! compute capability 12.x the kernel is a programmatic dependent launch: the weights (fixed)
//! load while the kernel ahead, which writes the input, still runs.
//!
//! On by default where it applies (CUDA, compute capability 9.0+, stage graphs on, so the op
//! costs no split of ggml's CUDA graph); `legion serve --no-mvf-ab` keeps ggml's two matmuls.
//! The CPU implementation is the reference (F32 sums in row order).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[doc(hidden)]
pub mod bench;
#[cfg(test)]
mod tests;

static MVF_BF16: ExternalOp = ExternalOp {
    name: "mvf_bf16",
    cpu: cpu_mvf_bf16,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mvf_bf16),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most tokens the op takes: ggml-cuda's `mul_mat_vec_f` range for BF16 weights.
pub const MAX_TOKENS: i64 = 8;

/// Most weights one op reads (`LEGION_MVF_MAX_W` in the kernel).
pub const MAX_WEIGHTS: usize = 4;

/// Fewest tokens [`takes`] gives the op: on a GB10 (`legion gdn-bench --ab`, Qwen3.8-27B's
/// shapes) it beat ggml's two launches at 8 tokens (a one-user verify step: 24.6 against
/// 26.5-27.4 us a layer) but not at 1-7, so those keep ggml's matmuls.
pub const TAKES_MIN_TOKENS: i64 = 8;

/// Whether graphs built from now on may take the op ([`set_mvf_ab`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether graphs built from now on run a DeltaNet layer's BF16 alpha and beta matmuls of 1-8
/// tokens as one op ([`mul_mat`]): on by default; `legion serve --no-mvf-ab` keeps ggml's two
/// matmuls (for comparisons).
pub fn set_mvf_ab(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether `w @ x` for every weight of `ws` takes one op: switched on, a CUDA build with stage
/// graphs on, a GPU that runs it ([`device_supported`]), every weight in that GPU's memory, at
/// least [`TAKES_MIN_TOKENS`] tokens, and [`fits`].
pub fn takes(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && crate::ggml::stage_graph::enabled()
        && x.ne()[1] >= TAKES_MIN_TOKENS
        && ws.iter().all(|w| !w.is_host() && !w.is_meta())
        && fits(ws, x)
        && device_supported()
}

/// Whether the op computes `w @ x` for every weight `w` of `ws` (1 to [`MAX_WEIGHTS`]) as
/// ggml-cuda's `mul_mat_vec_f` does: BF16 matrices (`[k, m]`, rows contiguous and 4-byte
/// aligned, `k` even) whose `m` is not a multiple of 32 (ggml's `mul_mat_f` takes those), and
/// an F32 `x` (`[k, n]`, values contiguous, columns 8-byte aligned) of 1 to [`MAX_TOKENS`]
/// tokens.
pub fn fits(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    let [xk, n, x2, x3] = x.ne();
    let xnb = x.nb();
    (1..=MAX_WEIGHTS).contains(&ws.len())
        && x.ty() == types::F32
        && x2 == 1
        && x3 == 1
        && xk % 2 == 0
        && xk > 0
        && xnb[0] == 4
        && xnb[1] % 8 == 0
        && (1..=MAX_TOKENS).contains(&n)
        && ws.iter().all(|&w| {
            let [k, m, w2, w3] = w.ne();
            let wnb = w.nb();
            w.ty() == types::BF16
                && k == xk
                && w2 == 1
                && w3 == 1
                && m % 32 != 0
                && wnb[0] == 2
                && wnb[1..].iter().all(|b| b % 4 == 0)
        })
}

/// Whether the current GPU runs the op: compute capability 9.0 or newer (false without CUDA).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_mvf_bf16_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_mvf_bf16_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// `w @ x` for each weight of `ws` ([`fits`]) from one op: F32 `[m_i, n]` views of its
/// `[sum m_i, n]` result (each weight's rows in turn, so a view's columns are the sum of the
/// rows apart).
pub fn mul_mat<'a>(ctx: &'a Context, ws: &[Tensor<'a>], x: Tensor<'a>) -> Vec<Tensor<'a>> {
    assert!(fits(ws, x), "mvf_bf16: shapes");
    let n = x.ne()[1];
    let total: i64 = ws.iter().map(|w| w.ne()[1]).sum();
    let mut args = vec![x];
    args.extend_from_slice(ws);
    let node = ctx.external(&MVF_BF16, types::F32, [total, n, 1, 1], &args);
    if let [_] = ws {
        return vec![node];
    }
    let mut off = 0;
    ws.iter()
        .map(|w| {
            let m = w.ne()[1];
            let v = ctx.view_2d(node, [m, n], total as usize * 4, off);
            off += m as usize * 4;
            v
        })
        .collect()
}

/// The weights of a node made by [`mul_mat`].
fn weights(dst: &ffi::ggml_tensor) -> usize {
    dst.src[1..]
        .iter()
        .take(MAX_WEIGHTS)
        .take_while(|s| !s.is_null())
        .count()
}

unsafe fn cpu_mvf_bf16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors laid out as `fits` checks.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let total = dst.ne[0] as usize;
        let mut base = 0;
        for i in 0..weights(dst) {
            let w = &*dst.src[1 + i];
            let m = w.ne[1] as usize;
            for r in (ith..m).step_by(nth) {
                let row = (w.data as *const u8).add(r * w.nb[1]) as *const u16;
                for t in 0..n {
                    let col = (x.data as *const u8).add(t * x.nb[1]) as *const f32;
                    let mut s = 0f32;
                    for c in 0..k {
                        s += f32::from_bits(u32::from(*row.add(c)) << 16) * *col.add(c);
                    }
                    *(dst.data as *mut f32).add(t * total + base + r) = s;
                }
            }
            base += m;
        }
    }
}

/// `legion_mvf_bf16` of `src/cuda/mvf_bf16.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Params {
    w: [*const u16; MAX_WEIGHTS],
    w_ld: [i64; MAX_WEIGHTS],
    rows: [i32; MAX_WEIGHTS],
    n_w: i32,
    x: *const f32,
    x_ld: i64,
    dst: *mut f32,
    dst_ld: i64,
    k: i32,
    n_tok: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_mvf_bf16_launch(p: *const Params, stream: *mut std::os::raw::c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_mvf_bf16(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat` (sources: x, then the weights), device tensors of the
    // stream's device.
    unsafe {
        let x = &*dst.src[0];
        let nw = weights(dst);
        let mut p = Params {
            w: [std::ptr::null(); MAX_WEIGHTS],
            w_ld: [0; MAX_WEIGHTS],
            rows: [0; MAX_WEIGHTS],
            n_w: nw as i32,
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            dst: dst.data as *mut f32,
            dst_ld: (dst.nb[1] / 4) as i64,
            k: x.ne[0] as i32,
            n_tok: x.ne[1] as i32,
        };
        for i in 0..nw {
            let w = &*dst.src[1 + i];
            p.w[i] = w.data as *const u16;
            p.w_ld[i] = (w.nb[1] / 2) as i64;
            p.rows[i] = w.ne[1] as i32;
        }
        match legion_mvf_bf16_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("mvf_bf16: error {e}")),
        }
    }
}
