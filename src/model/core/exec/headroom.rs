//! Compute buffers of new step graphs, with headroom where the graphs grow (#363).
//!
//! A long prompt's chunks each need a few more bytes than the one before: the step graph is
//! rebuilt for every new KV length, and its compute allocator (the oldest kept graph's) frees
//! and allocates its buffer again when the new graph needs more. On the GB10 that was 1,575 MiB
//! growing by ~0.04 MiB, every second 4,096-token chunk of Qwen3.8-Flash-Next: a `cudaFree` and
//! a `cudaMalloc` of ~70 ms with the GPU idle, ~2.6% of a long prompt's prefill. An integrated
//! GPU's prompt graphs get a little headroom ([`ComputeAllocator::headroom`]); the memory is
//! the host's and plentiful. Discrete GPUs keep their buffers exact (VRAM is the expert
//! cache's), except while the cache is lent ([`ComputeAllocator::padded`]). The headroom only
//! changes the buffer's size: the allocator plans each graph the same way.

use crate::ggml::{BufferType, ComputeAllocator, DeviceKind};

/// Whether a new step graph's compute buffer gets headroom ([`ComputeAllocator::headroom`]):
/// a graph of prompt chunks or verify batches (not `decode`) on an integrated GPU.
pub(super) fn roomy(kind: DeviceKind, decode: bool) -> bool {
    !decode && kind == DeviceKind::IntegratedGpu
}

/// A new compute allocator for a step graph in `buft`: padded while the expert cache is `lent`,
/// with headroom when `roomy`, else exact.
pub(super) fn step_allocator(buft: BufferType, lent: bool, roomy: bool) -> ComputeAllocator {
    if lent {
        ComputeAllocator::padded(buft)
    } else if roomy {
        ComputeAllocator::headroom(buft)
    } else {
        ComputeAllocator::new(buft)
    }
}

/// What the free memory a failed step buffer allocation reckons with may miss (each buffer's
/// allocation granule): within it, [`hopeless`] still lets the retries run.
const HOPELESS_SLACK: usize = 64 << 20;

/// Whether a step graph's compute buffer whose allocation failed asking `bytes` cannot fit
/// however many kept graphs are dropped (#37): `free` device bytes, `held` by the failed
/// attempt's allocator, `kept` by the stage's kept graphs, and `scratch` more the buffer must
/// leave free (`LENT_SCRATCH` while the cache is lent, when `bytes` does not count it yet).
/// Then each retry only allocates and frees again: a lent 32K-token prompt's 12.3 GiB buffer
/// that left less than the scratch took 0.23 s a try.
pub(super) fn hopeless(
    free: usize,
    held: usize,
    kept: usize,
    bytes: usize,
    scratch: usize,
) -> bool {
    free + held + kept + HOPELESS_SLACK < bytes + scratch
}

#[cfg(test)]
mod tests;
