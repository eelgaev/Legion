//! Less host work per MTP draft window (#362), without changing what it computes.
//!
//! - A pass's inputs and outputs go through pinned memory, as copies queued on the backend's
//!   stream ([`crate::ggml::pinned_io`]). Before, every input and every output was a
//!   synchronous copy of its own.
//! - The target step's deferred expert-cache work ([`Executor::set_after_overlap`]) is split
//!   by stage ([`tail_split`]). The first pass of a window runs the work for the stages before
//!   the last, while its graph computes. The work for the last stage's layers runs while the
//!   second pass computes, or, in a window of one pass, while the next target step's first
//!   stage computes. On Qwen3.8-Flash-Next over two V100s, that work (about 130 queued expert
//!   copies plus the table writes, about 1.3 ms of host time) outlasted the first pass's
//!   ~1.8 ms of GPU work by about 0.25 ms.
//!
//! [`Executor::set_after_overlap`]: super::Executor::set_after_overlap

use std::ops::Range;

use crate::ggml::pinned_io::PinnedIo;
use crate::ggml::{Backend, Tensor};
use crate::model::core::InputSink;

#[cfg(test)]
mod tests;

/// Pinned memory a pass's queued transfers may take at most. A decode pass's inputs are a few
/// KB; a prompt chunk's take-in (rows and masks of thousands of tokens) goes the synchronous
/// way, through the stage's bounce buffer as before.
pub(super) const PINNED_MAX: usize = 4 << 20;

/// Writes a pass's inputs as copies queued through `io`.
pub(super) struct Queued<'p> {
    pub io: &'p mut PinnedIo,
    pub backend: &'p mut Backend,
}

impl InputSink for Queued<'_> {
    fn i32(&mut self, t: Tensor<'_>, values: &[i32]) {
        self.io.write_i32(self.backend, t, values);
    }

    fn f32(&mut self, t: Tensor<'_>, values: &[f32]) {
        self.io.write_f32(self.backend, t, values);
    }

    fn bytes(&mut self, t: Tensor<'_>, bytes: &[u8]) {
        self.io.write_bytes(self.backend, t, bytes);
    }
}

/// The layers whose share of a target step's deferred expert-cache work waits for a later
/// MTP pass (or the next step's first stage): the last stage's target layers, out of
/// `stages`' layer ranges, the MTP layers from `n_layer` on. `None` (no split) with one
/// stage, when the stages before the last or the last have no tiered layer (`device_of`:
/// a layer's cache device, `None` for untiered layers), or when a cache device has layers on
/// both sides. The cache plans swaps per device, so a split that keeps every device whole
/// plans the same swaps as one call over all the layers.
pub(super) fn tail_split(
    stages: &[Range<usize>],
    n_layer: usize,
    device_of: impl Fn(usize) -> Option<usize>,
) -> Option<Range<usize>> {
    let [.., before, last] = stages else {
        return None;
    };
    let tail = last.start..last.end.min(n_layer);
    if tail.is_empty() || before.end != tail.start {
        return None;
    }
    let head: Vec<usize> = (0..tail.start).filter_map(&device_of).collect();
    let rest: Vec<usize> = tail.clone().filter_map(&device_of).collect();
    if head.is_empty() || rest.is_empty() || head.iter().any(|d| rest.contains(d)) {
        return None;
    }
    Some(tail)
}
