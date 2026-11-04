//! A decode or verify step's inputs as copies queued on each stage's stream (#820).
//!
//! On Qwen3.8-Flash-Next over four V100s with MTP drafting (C=1, `--hc-fuse`, MTP k=2), the
//! GPUs were all idle for about 0.31 ms between the end of the second draft pass (last GPU) and
//! the first kernel of the next verify step (first GPU). The expert cache's deferred work was
//! not in it: it already runs beside the draft passes and beside the first stage
//! (`mtp_window`, `after_split`). Two thirds of the gap was the first stage's input fill:
//! about 14 inputs (tokens, positions, slot and block tables, sparse-attention tables, n-gram
//! rows), each a synchronous copy (`cudaMemcpyAsync` from pageable memory plus a
//! `cudaStreamSynchronize`, ~14 µs each, ~200 µs in all) before the first launch.
//!
//! Through the stage's pinned buffer ([`crate::ggml::pinned_io::PinnedIo`], as GLM's parallel
//! decode steps fill theirs, #676) the copies are queued on the stage's stream ahead of its
//! graph and the host goes on to the launches; the stage's synchronize after its graph waits
//! for them too ([`crate::ggml::pinned_io::PinnedIo::finish`]). The later stages' inputs,
//! filled while the stage before computes, go the same way. Every input gets the same bytes,
//! and on its stage's stream every copy still lands before the graph that reads it, so the
//! outputs are byte-identical.
//!
//! Prompt steps keep the synchronous fill (their inputs grow to megabytes, and a long-dataflow
//! step queues its own through `flow_io`), and so do steps whose inputs another GPU may read
//! before the stage's stream reaches the copies: a pair-parallel step's helper (`pair_step`)
//! and a split model's (`remote`).

use super::{Executor, Group, GroupShape};

#[cfg(test)]
mod tests;

/// Whether a step's stages take their inputs as queued copies (see the module docs): a step of
/// one micro-batch (`n_micro`) that is not a prompt step, not pair-parallel (`paired`) and not
/// split over machines (`remote`).
pub(super) fn queued(n_micro: usize, prompt: bool, paired: bool, remote: bool) -> bool {
    n_micro == 1 && !prompt && !paired && !remote
}

impl Executor {
    /// Fills stage `s`'s own inputs for `groups`: as copies queued on its stream
    /// ([`Self::fill_inputs_queued`]) when `queued`, else each copy waited for.
    pub(super) fn fill_step_inputs(
        &mut self,
        s: usize,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        queued: bool,
    ) {
        if queued {
            self.fill_inputs_queued(s, groups, shape);
        } else {
            self.fill_inputs(s, groups, shape);
        }
    }

    /// Waits for stage `s`'s graph and for any input copies queued ahead of it.
    pub(super) fn sync_stage(&mut self, s: usize) {
        let st = &mut self.stages[s];
        st.mtp_io.finish(&mut st.backend);
    }
}
