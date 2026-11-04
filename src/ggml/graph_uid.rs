//! Graph ids (#678): a kept graph's `ggml_cgraph::uid` on a meta backend.
//!
//! ggml marks a graph that is the same as the one computed before with a non-zero `uid`
//! (ggml's scheduler sets one per planned graph). The meta backend (tensor parallelism)
//! rebuilds its per-device subgraphs unless the graph's uid is the one it computed last, and
//! ggml-cuda compares every node of a subgraph with its captured CUDA graph unless the
//! subgraph's uid is the one captured. Legion runs its own kept graphs, which ggml leaves at
//! uid 0, so a decode step of Gemma 31B at `--tp 2` spent ~1.5 ms of host time with the GPUs
//! idle rebuilding 121 subgraphs and comparing their nodes, every token.
//!
//! [`crate::ggml::Backend::ready_kept`] gives a kept graph a new id whenever it initializes the
//! graph's per-device tensors (again): then the meta rebuilds once, and the steps after it
//! that run the same graph again, with nothing else run in between, reuse the subgraphs.
//!
//! ggml keeps `uid` in `struct ggml_cgraph`, which its public header leaves opaque (ggml-impl.h,
//! MIT); [`CGraph`] mirrors it, and the tests check the mirror against a graph ggml made.

use std::sync::atomic::{AtomicU64, Ordering};

use super::ffi;

#[cfg(test)]
mod tests;

/// `struct ggml_hash_set` (ggml-impl.h).
#[repr(C)]
struct HashSet {
    size: usize,
    used: *mut u32,
    keys: *mut *mut ffi::ggml_tensor,
}

/// `struct ggml_cgraph` (ggml-impl.h).
#[repr(C)]
struct CGraph {
    size: std::os::raw::c_int,
    n_nodes: std::os::raw::c_int,
    n_leafs: std::os::raw::c_int,
    nodes: *mut *mut ffi::ggml_tensor,
    grads: *mut *mut ffi::ggml_tensor,
    grad_accs: *mut *mut ffi::ggml_tensor,
    leafs: *mut *mut ffi::ggml_tensor,
    use_counts: *mut i32,
    visited_hash_set: HashSet,
    order: std::os::raw::c_int,
    uid: u64,
}

/// A graph id no other graph has had. Counts from 2^63, away from ggml's own counter
/// (`ggml_graph_next_uid`, from 1), so an id of ours never equals one ggml gave out.
pub(crate) fn next_uid() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1 << 63);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The graph's `uid` (0: none).
///
/// # Safety
/// `g` is a live graph made by ggml.
#[cfg(test)]
pub(crate) unsafe fn uid(g: *const ffi::ggml_cgraph) -> u64 {
    // SAFETY: the caller's; `CGraph` is the struct's layout (checked by the tests).
    unsafe { (*(g as *const CGraph)).uid }
}

/// The graph's leafs (tensors it reads that no node of it makes).
///
/// # Safety
/// `g` is a live graph made by ggml; the slice lives no longer than it, unchanged.
pub(crate) unsafe fn leafs<'a>(g: *const ffi::ggml_cgraph) -> &'a [*mut ffi::ggml_tensor] {
    // SAFETY: as in `uid`; `leafs` holds `n_leafs` tensors.
    unsafe {
        let g = &*(g as *const CGraph);
        if g.n_leafs <= 0 {
            return &[];
        }
        std::slice::from_raw_parts(g.leafs, g.n_leafs as usize)
    }
}

/// Sets the graph's `uid`.
///
/// # Safety
/// `g` is a live graph made by ggml that nothing else uses meanwhile.
pub(crate) unsafe fn set_uid(g: *mut ffi::ggml_cgraph, uid: u64) {
    // SAFETY: as in `uid`.
    unsafe { (*(g as *mut CGraph)).uid = uid }
}
