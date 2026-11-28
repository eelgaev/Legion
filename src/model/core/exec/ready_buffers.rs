//! Prompt graphs' compute buffers allocated at load (#836).
//!
//! An integrated GPU's first long prompt after a start ran its first chunks at up to twice
//! the time of later ones: each new chunk shape takes over the oldest kept prompt graph's
//! compute allocator, and while the kept graphs were still the warm-up's small ones every
//! takeover grew the buffer. Each growth frees the buffer (`cudaFree`, which waits for the
//! device: ~1 s with a chunk in flight) and allocates a bigger one (~70-85 ms, more under
//! memory pressure). Later prompts found the buffers grown and never paid it.
//!
//! At load, each such stage measures its prompt step at the end of the context (the largest
//! one) and allocates that many bytes now for each graph it keeps, written once so their
//! memory is in place. A graph that would have started a new allocator takes one of these
//! instead ([`Executor::build_kept`](super::Executor)): the same allocator, planning its
//! first graph afresh as a new one does, in a bigger buffer. The plans (so the bits) are the
//! same; only the buffers' sizes differ, as with the headroom ([`super::headroom`]). The
//! memory is what a long prompt would have grown the kept graphs to anyway, held from load.
//! `legion serve --no-prompt-buffer-reserve` turns it off.
//!
//! The kept prompt graphs share these buffers (#836): a one-stage executor reserves
//! [`PROMPT_BUFFERS`] (two) for its three kept prompt graphs, and every prompt graph is
//! allocated from one of them ([`crate::ggml::Context::alloc_graph_shared`]), not from a buffer
//! of its own. Two graphs in one buffer overlap there, which is safe as a kept graph's inputs
//! are filled before every run anyway (ggml's planner already lays a graph's later nodes over
//! its own inputs, see [`crate::ggml::input_audit`]). The one rule is that nothing plans or
//! grows a buffer the running graph is in: a new graph goes into the other one than the most
//! recent prompt graph's, and the graph built while a chunk computes ([`super::prompt_ahead`])
//! into the other one than the running graph's, so a buffer that must grow for it frees memory
//! no running graph reads. The buffers plan every graph afresh (its layout is the graph's own,
//! as in a new buffer, not the one the buffer held before), so the kernels and sums are the
//! ones a graph in a buffer of its own gets. One buffer fewer: at 4,096-token steps at the end
//! of a 262,144-token context, ~2.2 GiB more memory free on a GB10.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use crate::ggml::DeviceKind;

#[cfg(test)]
mod tests;

static ON: AtomicBool = AtomicBool::new(true);

/// Whether the load allocates the kept prompt graphs' compute buffers (on by default;
/// `legion serve --no-prompt-buffer-reserve` turns it off).
pub fn set_enabled(on: bool) {
    ON.store(on, Relaxed);
}

/// See [`set_enabled`].
pub fn enabled() -> bool {
    ON.load(Relaxed)
}

/// Memory a reserve leaves free on the device, at least: a buffer that would leave less is
/// not allocated (graphs then grow their buffers as before).
pub(super) const KEEP_FREE: usize = 2 << 30;

/// The compute buffers a one-stage executor's kept prompt graphs share at most (see the
/// module doc): one for the running graph and one for the graph built beside it.
pub const PROMPT_BUFFERS: usize = 2;

/// How many buffers a stage reserves: one per kept mixed graph (`kept`), at most
/// [`PROMPT_BUFFERS`], on an integrated GPU (where new prompt graphs get headroom,
/// [`super::headroom::roomy`]), none on a meta backend (its kept graphs share one buffer),
/// while the expert cache is `lent`, or when off.
pub(super) fn wanted(on: bool, kind: DeviceKind, meta: bool, lent: bool, kept: usize) -> usize {
    if on && kind == DeviceKind::IntegratedGpu && !meta && !lent {
        kept.min(PROMPT_BUFFERS)
    } else {
        0
    }
}

/// Whether a stage's reserved buffers (`reserved` of them, of a one-stage executor: `one`)
/// are shared by its kept prompt graphs: two or more, else each graph keeps its own as before
/// (one buffer leaves no other for the graph built beside the running one).
pub(super) fn shared(one: bool, reserved: usize) -> bool {
    one && reserved >= PROMPT_BUFFERS
}

/// The compute buffers `kept` prompt graphs take on a GPU sharing the host's memory: one each,
/// or with `sharing` (the reserve on, one stage on an integrated GPU, no byte budget) at most
/// [`PROMPT_BUFFERS`]. What the KV fit charges for them.
pub fn prompt_buffers(kept: usize, sharing: bool) -> usize {
    if sharing && kept >= PROMPT_BUFFERS {
        PROMPT_BUFFERS
    } else {
        kept
    }
}

/// Whether one more buffer of `bytes` fits in `free` device memory, leaving [`KEEP_FREE`].
pub(super) fn fits(free: usize, bytes: usize) -> bool {
    bytes > 0 && free.saturating_sub(KEEP_FREE) >= bytes
}
