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

static WINDOWS: ExternalOp = ExternalOp {
    name: "spec_windows",
    cpu: cpu_windows,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_windows),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// [`window`] for every `(src, dst)` of `pairs` in one launch (#689): all of one shape and
/// strides ([`fits_all`]), so one grid covers them, a layer a grid row. The pairs' data
/// pointers go in an I64 table (`[2, pairs]`, src then dst; the second tensor returned) that the
/// caller fills with [`fill_table`] once the graph is allocated. The pairs' memory must stay
/// where it is while the graph lives (pool tensors). The first tensor returned is the node the
/// graph expands.
pub(super) fn windows<'a>(
    ctx: &'a Context,
    pairs: &[(Tensor<'a>, Tensor<'a>)],
    keep: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>) {
    assert!(fits_all(pairs, keep), "spec_windows: shapes");
    // An output too, so the allocator never places anything over it.
    let table = ctx
        .new_tensor(types::I64, &[2, pairs.len() as i64])
        .as_input()
        .as_output();
    let (src, dst) = pairs[0];
    let node = ctx.external(&WINDOWS, types::F32, [1, 1, 1, 1], &[src, dst, keep, table]);
    (node, table)
}

/// Whether [`windows`] takes these pairs: at least one, each as [`window`] takes it, all of the
/// first one's sizes and strides, all with memory.
pub(super) fn fits_all(pairs: &[(Tensor<'_>, Tensor<'_>)], keep: Tensor<'_>) -> bool {
    let Some(&(s0, d0)) = pairs.first() else {
        return false;
    };
    pairs.iter().all(|&(s, d)| {
        fits(s, d, keep)
            && s.ne() == s0.ne()
            && s.nb() == s0.nb()
            && d.ne() == d0.ne()
            && d.nb() == d0.nb()
            // SAFETY: live tensors.
            && unsafe { !(*s.raw()).data.is_null() && !(*d.raw()).data.is_null() }
    })
}

/// Writes `pairs`' data pointers into `table` ([`windows`]); after the graph's allocation.
pub(super) fn fill_table(table: Tensor<'_>, pairs: &[(Tensor<'_>, Tensor<'_>)]) {
    // SAFETY: live tensors.
    let ptr = |t: Tensor<'_>| unsafe { (*t.raw()).data as u64 };
    let bytes: Vec<u8> = pairs
        .iter()
        .flat_map(|&(s, d)| [ptr(s), ptr(d)])
        .flat_map(u64::to_ne_bytes)
        .collect();
    table.set_bytes(&bytes);
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

/// The table of a [`windows`] node: `(src, dst)` pointers.
///
/// SAFETY: `node` is made by [`windows`], its table filled ([`fill_table`]) and on the host.
unsafe fn table(node: &ffi::ggml_tensor) -> Vec<(*const f32, *mut f32)> {
    unsafe {
        let t = &*node.src[3];
        let n = t.ne[1] as usize;
        let p = std::slice::from_raw_parts(t.data as *const u64, 2 * n);
        p.chunks(2)
            .map(|c| (c[0] as *const f32, c[1] as *mut f32))
            .collect()
    }
}

unsafe fn cpu_windows(node: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `windows`, host tensors, the table filled.
    unsafe {
        let a = args(node);
        let pairs = table(node);
        let keep = (*node.src[2]).data as *const i32;
        let per = a.rows * a.n;
        for job in (ith..pairs.len() * per).step_by(nth) {
            let (l, r, s) = (job / per, job % a.rows, (job % per) / a.rows);
            let k = *keep.add(s);
            if k <= 0 {
                continue;
            }
            let k = k as usize;
            assert!(
                k + a.w <= a.src_len,
                "spec_windows: keep {k} past the window"
            );
            let (src, dst) = pairs[l];
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
    fn legion_spec_windows_launch(
        p: *const LegionSpecWindow,
        table: *const u64,
        layers: i64,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_window(node: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `window`, device tensors of the stream's device.
    unsafe {
        let p = launch_args(node);
        match legion_spec_window_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err("spec_window: arguments".to_string()),
            e => Err(format!("spec_window: CUDA error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_windows(node: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `windows`, device tensors of the stream's device; the table (device
    // memory, filled) holds the pairs' pointers.
    unsafe {
        let p = launch_args(node);
        let t = &*node.src[3];
        match legion_spec_windows_launch(&p, t.data as *const u64, t.ne[1], stream) {
            0 => Ok(()),
            -1 => Err("spec_windows: arguments".to_string()),
            e => Err(format!("spec_windows: CUDA error {e}")),
        }
    }
}

/// The kernel's arguments of a [`window`] or [`windows`] node (the first pair's pointers).
///
/// SAFETY: as [`args`].
#[cfg(feature = "cuda")]
unsafe fn launch_args(node: &ffi::ggml_tensor) -> LegionSpecWindow {
    unsafe {
        let a = args(node);
        LegionSpecWindow {
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
        }
    }
}

#[cfg(test)]
mod tests;
