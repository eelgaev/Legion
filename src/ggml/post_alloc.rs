//! A pass the model layer may run over every CUDA graph right after its allocation
//! ([`set_rewrite`]), to replace chains of small ggml nodes with external ops of its own
//! (src/model/core/node_fuse.rs).
//!
//! It runs after the allocator has planned the graph as built, so every tensor keeps the
//! memory it gets without the pass: ggml-cuda's layout-dependent fusions (the MoE weighted sum,
//! #410) see the same layout and make the same choices, and the pass itself can read them off
//! the tensors' addresses. A rewrite may change a node into an external op (new sources and
//! parameters, the same tensor and memory) and turn nodes it takes over into `GGML_OP_NONE`
//! (ggml-cuda and the CPU backend skip those); it never adds or removes nodes.
//!
//! A rewrite that changes which tensors live how long (a node reading a chain's inputs in place
//! of its output keeps them alive until it runs) needs the graph planned again after it
//! ([`set_relayout`], #836). It still reads the layout of the graph as built (planned first, as
//! without it), the graph is then planned and allocated again as rewritten, and when what the
//! rewrite relied on no longer holds in the new layout, it is undone and the graph planned a
//! third time as built.

use std::sync::OnceLock;

use super::{BufferType, ffi};

/// Rewrites an allocated graph's nodes in place (see the module doc).
///
/// # Safety
/// `graph` is a live, allocated graph none of whose nodes has run yet.
pub type Rewrite = unsafe fn(graph: *mut ffi::ggml_cgraph);

static REWRITE: OnceLock<Rewrite> = OnceLock::new();

/// Sets the pass run on the CUDA graphs allocated from now on (the first call wins). Until it
/// is called, none.
pub fn set_rewrite(f: Rewrite) {
    let _ = REWRITE.set(f);
}

/// Whether graphs allocated in `buft` are computed by ggml's CUDA backend: device memory of a
/// CUDA device (not pinned host memory, not a split or meta buffer).
fn is_cuda(buft: BufferType) -> bool {
    cfg!(feature = "cuda") && !buft.is_host() && {
        let name = buft.name();
        name.strip_prefix("CUDA")
            .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
    }
}

/// Runs the pass on `graph`, just allocated in `buft`, when it is a CUDA graph.
///
/// # Safety
/// As [`Rewrite`].
pub(crate) unsafe fn run(graph: *mut ffi::ggml_cgraph, buft: BufferType) {
    if let Some(f) = REWRITE.get()
        && is_cuda(buft)
    {
        // SAFETY: the caller's.
        unsafe { f(graph) }
    }
}

/// A rewrite planned again after it (see the module doc). `cuda`: ggml-cuda computes the graph.
#[derive(Clone, Copy)]
pub struct Relayout {
    /// Whether the graph, not allocated yet, may take the rewrite (cheap: every graph is asked).
    ///
    /// # Safety
    /// A live graph.
    pub wants: unsafe fn(graph: *mut ffi::ggml_cgraph, cuda: bool) -> bool,
    /// Rewrites the allocated graph (none of whose nodes has run) from its layout, returning
    /// how to check and undo it, or `None` when it rewrote nothing.
    ///
    /// # Safety
    /// As [`Rewrite`].
    pub rewrite: unsafe fn(graph: *mut ffi::ggml_cgraph, cuda: bool) -> Option<Box<dyn Relaid>>,
}

/// What a [`Relayout`] rewrite did.
pub trait Relaid {
    /// Whether, in the layout of the graph allocated again as rewritten, what the rewrite read
    /// off the first layout still holds.
    ///
    /// # Safety
    /// The rewritten graph, allocated, none of whose nodes has run.
    unsafe fn holds(&self, graph: *mut ffi::ggml_cgraph) -> bool;
    /// Puts the graph's nodes back as they were built.
    ///
    /// # Safety
    /// The rewritten graph, none of whose nodes has run.
    unsafe fn undo(&self, graph: *mut ffi::ggml_cgraph);
}

static RELAYOUT: OnceLock<Relayout> = OnceLock::new();

/// Sets the rewrite planned again after it, for the CUDA graphs allocated from now on (the
/// first call wins). Until it is called, none.
pub fn set_relayout(r: Relayout) {
    let _ = RELAYOUT.set(r);
}

/// The [`Relayout`] `graph`, about to be allocated in `buft`, may take: when it is a CUDA graph
/// that [`Relayout::wants`] it.
///
/// # Safety
/// A live graph.
pub(crate) unsafe fn relayout(graph: *mut ffi::ggml_cgraph, buft: BufferType) -> Option<Relayout> {
    let r = *RELAYOUT.get()?;
    // SAFETY: the caller's.
    (is_cuda(buft) && unsafe { (r.wants)(graph, true) }).then_some(r)
}
