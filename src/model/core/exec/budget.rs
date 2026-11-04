//! The kept step graphs' compute buffers within what the load left them (#430). With the
//! expert cache, the cache takes the VRAM the measured step buffers leave (two prompt steps
//! of a chunk, a full verify batch and a full decode batch, see `LmEngine::load`), and only a
//! fixed reserve stays free for ggml's pools, cuBLAS and CUDA graphs. Steps the scheduler
//! batches are bigger than one chunk, though: several sequences' prompts and a verify batch
//! fill up to `--max-batch-tokens`, and every kept mixed graph's buffer grows to the biggest
//! step it ran (an evicted graph's allocator is reused, buffer and all). Under many users the
//! kept buffers outgrew the budget, ate the reserve and the GPU ran out of memory. A stage
//! now drops its oldest kept mixed graphs once their buffers pass the budget.

use super::{Executor, KeptGraph, LENT_MIXED_KEPT};
use crate::model::core::GroupShape;

#[cfg(test)]
mod tests;

/// How many of the oldest `mixed` buffers (oldest first) to drop so that they and `decode`
/// bytes fit `budget`. The newest mixed graph stays when `keep_newest` (it was just built for
/// the step about to run), even alone over the budget.
pub(super) fn evictions(mixed: &[usize], decode: usize, keep_newest: bool, budget: usize) -> usize {
    let droppable = mixed.len() - usize::from(keep_newest && !mixed.is_empty());
    let mut total: usize = decode + mixed.iter().sum::<usize>();
    let mut n = 0;
    while n < droppable && total > budget {
        total -= mixed[n];
        n += 1;
    }
    n
}

/// Bytes of the kept graphs' own compute buffers.
fn buffers(list: &[(Vec<GroupShape>, KeptGraph)]) -> impl Iterator<Item = usize> + '_ {
    list.iter().map(|(_, k)| k.allocator_size())
}

impl Executor {
    /// Per stage: the most its kept decode and mixed graphs' compute buffers take before the
    /// oldest mixed ones go (0: no limit, the default). Not while a long prompt borrows the
    /// expert cache (its VRAM is then the prompt's), nor on meta backends (one shared buffer
    /// per kind, sized by tensor parallelism's own fit).
    pub fn set_compute_budget(&mut self, per_stage: Vec<usize>) {
        self.compute_budget = per_stage;
    }

    /// The bytes stage `s`'s kept decode and mixed graphs hold in their compute buffers.
    pub fn kept_buffer_bytes(&self, s: usize) -> usize {
        let st = &self.stages[s];
        buffers(&st.decode).chain(buffers(&st.mixed)).sum()
    }

    /// After stage `s` built a graph (a mixed one when `mixed`): drops its oldest kept mixed
    /// graphs while the kept buffers pass the stage's budget ([`Self::set_compute_budget`]).
    pub(super) fn fit_compute_budget(&mut self, s: usize, mixed: bool) {
        let budget = self.compute_budget.get(s).copied().unwrap_or(0);
        if budget == 0 || self.mixed_kept == LENT_MIXED_KEPT {
            return;
        }
        let st = &mut self.stages[s];
        if st.backend.is_meta() {
            return;
        }
        let sizes: Vec<usize> = buffers(&st.mixed).collect();
        let decode = buffers(&st.decode).sum();
        let n = evictions(&sizes, decode, mixed, budget);
        if n > 0 {
            st.mixed.drain(..n);
            tracing::debug!(
                stage = s,
                dropped = n,
                kept_mib = (decode + sizes[n..].iter().sum::<usize>()) >> 20,
                budget_mib = budget >> 20,
                "kept step graphs over the compute budget: oldest dropped"
            );
        }
    }
}
