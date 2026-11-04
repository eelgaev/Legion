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

/// How many buffers a stage reserves: one per kept mixed graph (`kept`) on an integrated GPU
/// (where new prompt graphs get headroom, [`super::headroom::roomy`]), none on a meta backend
/// (its kept graphs share one buffer), while the expert cache is `lent`, or when off.
pub(super) fn wanted(on: bool, kind: DeviceKind, meta: bool, lent: bool, kept: usize) -> usize {
    if on && kind == DeviceKind::IntegratedGpu && !meta && !lent {
        kept
    } else {
        0
    }
}

/// Whether one more buffer of `bytes` fits in `free` device memory, leaving [`KEEP_FREE`].
pub(super) fn fits(free: usize, bytes: usize) -> bool {
    bytes > 0 && free.saturating_sub(KEEP_FREE) >= bytes
}
