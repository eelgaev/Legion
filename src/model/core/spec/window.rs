//! A replay commit's window copy for every slot at once, as an external op ([`ExternalOp`],
//! CUDA kernel `spec_window_kernel` in `src/cuda/gdn.cu`): each slot's new conv history (or
//! per-layer-embedding history) is the stored window from its own kept count on. One launch per
//! layer replaces one copy per run of slots keeping the same count (#836); the same bytes land.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static WINDOW: ExternalOp = ExternalOp {
    name: "spec_window",
    cpu: cpu_window,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_window),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// For every slot `s` with `keep[s] > 0` and every row `r`: `dst[.., r, s]` becomes
/// `src[keep[s] .. keep[s] + w, r, s]`, `w = dst.ne()[0]`. `src` `[>= w + max keep, rows, n]`
/// and `dst` `[w, rows, n]` are F32 with contiguous rows (any row and slot strides); `keep` is
/// I32 `[n]` (a graph input). Slots keeping 0 are left alone. The returned node (one float,
/// never written) is what the graph expands.
pub(super) fn window<'a>(
    ctx: &'a Context,
    src: Tensor<'a>,
    dst: Tensor<'a>,
    keep: Tensor<'a>,
) -> Tensor<'a> {
    assert!(fits(src, dst, keep), "spec_window: shapes");
    ctx.external(&WINDOW, types::F32, [1, 1, 1, 1], &[src, dst, keep])
}

/// Whether [`window`] takes these tensors.
pub(super) fn fits(src: Tensor<'_>, dst: Tensor<'_>, keep: Tensor<'_>) -> bool {
    let (s, d) = (src.ne(), dst.ne());
    src.ty() == types::F32
        && dst.ty() == types::F32
        && keep.ty() == types::I32
        && src.nb()[0] == 4
        && dst.nb()[0] == 4
        && s[0] >= d[0]
        && s[1] == d[1]
        && s[2] == d[2]
        && s[3] == 1
        && d[3] == 1
        && keep.is_contiguous()
        && keep.elements() as i64 == d[2]
}

/// The op's sizes and float strides.
struct Args {
    w: usize,
    rows: usize,
    n: usize,
    src_len: usize,
    s: [usize; 2],
    d: [usize; 2],
}

/// SAFETY: `dst` is a node made by [`window`].
unsafe fn args(node: &ffi::ggml_tensor) -> Args {
    unsafe {
        let (src, dst) = (&*node.src[0], &*node.src[1]);
        Args {
            w: dst.ne[0] as usize,
            rows: dst.ne[1] as usize,
            n: dst.ne[2] as usize,
            src_len: src.ne[0] as usize,
            s: [src.nb[1] / 4, src.nb[2] / 4],
            d: [dst.nb[1] / 4, dst.nb[2] / 4],
        }
    }
}

unsafe fn cpu_window(node: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `window`, host tensors.
    unsafe {
        let a = args(node);
        let src = (*node.src[0]).data as *const f32;
        let dst = (*node.src[1]).data as *mut f32;
        let keep = (*node.src[2]).data as *const i32;
        for job in (ith..a.rows * a.n).step_by(nth) {
            let (r, s) = (job % a.rows, job / a.rows);
            let k = *keep.add(s);
            if k <= 0 {
                continue;
            }
            let k = k as usize;
            assert!(
                k + a.w <= a.src_len,
                "spec_window: keep {k} past the window"
            );
            std::ptr::copy_nonoverlapping(
                src.add(s * a.s[1] + r * a.s[0] + k),
                dst.add(s * a.d[1] + r * a.d[0]),
                a.w,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LegionSpecWindow {
    src: *const f32,
    dst: *mut f32,
    keep: *const i32,
    w: i64,
    rows: i64,
    n: i64,
    s1: i64,
    s2: i64,
    d1: i64,
    d2: i64,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_spec_window_launch(p: *const LegionSpecWindow, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_window(node: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `window`, device tensors of the stream's device.
    unsafe {
        let a = args(node);
        let p = LegionSpecWindow {
            src: (*node.src[0]).data as *const f32,
            dst: (*node.src[1]).data as *mut f32,
            keep: (*node.src[2]).data as *const i32,
            w: a.w as i64,
            rows: a.rows as i64,
            n: a.n as i64,
            s1: a.s[0] as i64,
            s2: a.s[1] as i64,
            d1: a.d[0] as i64,
            d2: a.d[1] as i64,
        };
        match legion_spec_window_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err("spec_window: arguments".to_string()),
            e => Err(format!("spec_window: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
