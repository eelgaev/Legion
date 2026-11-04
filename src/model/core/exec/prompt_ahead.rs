//! A long prompt's next chunk graph built while the current chunk computes (#836).
//!
//! Every chunk of a long prompt is a new graph: its shape lists the sparse-attention key pools
//! the chunk reads (qwen4exp), and that count grows with every chunk. So before each chunk's
//! first kernel the host builds the graph, plans it with ggml's allocator and allocates its
//! compute buffer, with the GPU idle: on a DGX Spark (GB10, Qwen3.8-Flash-Next NVFP4) ~25-30 ms
//! for each 4,096-token chunk, more when the buffer grows.
//!
//! The engine knows the next chunk (it reads its n-gram rows ahead already) and says so before
//! the step ([`Executor::set_prompt_next`]): its token count and its logits. A one-stage step
//! of that prompt chunk alone builds the next chunk's graph between its launch and its wait,
//! as [`super::verify_ahead`] does for verify shapes. It is the same graph the next step would
//! build (the same shape, the same builder) in the same compute buffer: the running graph is
//! set aside while it is built, and the cap on the kept prompt graphs is one lower meanwhile, so
//! the graph evicted is the one the next step would have evicted and its buffer is reused as it
//! would have been. Only when the graph is built changes, not the kernels or the bits.
//!
//! Not on meta backends (one shared buffer per kind), not while the expert cache is lent (one
//! kept mixed graph), not under a compute budget (`set_compute_budget`), not for steps with state captures, images or other groups (the next
//! step's shape is not known from this one), and not when the next chunk is longer than this
//! one. On by default; `legion serve --no-prompt-build-ahead` turns it off.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use super::{
    Executor, Group, GroupShape, KV_BLOCK, LENT_MIXED_KEPT, Logits, SeqInput, batch_shape, with_kv,
};

#[cfg(test)]
mod tests;

static ON: AtomicBool = AtomicBool::new(true);

/// Whether one-stage prompt steps build the next chunk's graph while they compute (on by
/// default; `legion serve --no-prompt-build-ahead` turns it off).
pub fn set_enabled(on: bool) {
    ON.store(on, Relaxed);
}

/// See [`set_enabled`].
pub fn enabled() -> bool {
    ON.load(Relaxed)
}

/// The prompt chunk the engine runs after this step's ([`Executor::set_prompt_next`]): its
/// token count and the logits it returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptNext {
    pub tokens: usize,
    pub logits: Logits,
}

/// The step after `groups` as [`PromptNext`] says: `groups` must be one prompt chunk of one
/// sequence (no verify, no images) at least as long as the next one, which must still fit a
/// `table`-block table (paged attention; `None`: gather attention, no limit). Its tokens are the
/// first `next.tokens` of this chunk's: a shape depends on their count, not their values.
pub(super) fn next_groups<'b>(
    groups: &[Group<'b>],
    next: PromptNext,
    table: Option<usize>,
) -> Option<Vec<Group<'b>>> {
    let [g] = groups else { return None };
    let [s] = &g.seqs[..] else { return None };
    if g.verify || s.media.is_some() || next.tokens == 0 || next.tokens > s.tokens.len() {
        return None;
    }
    let n_past = s.n_past + s.tokens.len();
    if table.is_some_and(|w| (n_past + next.tokens).div_ceil(KV_BLOCK) > w) {
        return None;
    }
    Some(vec![Group {
        slot: g.slot,
        seqs: vec![SeqInput {
            tokens: &s.tokens[..next.tokens],
            n_past,
            logits: next.logits,
            ..s.clone()
        }],
        verify: false,
    }])
}

impl Executor {
    /// The prompt chunk the next step runs, if the engine knows it: the step about to run
    /// builds its graph while it computes (`prompt_ahead`). Only the next
    /// [`Self::step`] reads it.
    pub fn set_prompt_next(&mut self, next: Option<PromptNext>) {
        self.prompt_next = next;
    }

    /// The prompt shape to build ahead while stage 0 (the only one) runs `shape` for `groups`
    /// with `next` coming after it ([`next_groups`]), if it is not kept yet.
    pub(super) fn prompt_ahead(
        &self,
        groups: &[Group<'_>],
        table: Option<usize>,
        shape: &[GroupShape],
        next: Option<PromptNext>,
    ) -> Option<Vec<GroupShape>> {
        let next = next?;
        // Kept prompt graphs with the running one set aside: one fewer (`build_prompt_ahead`).
        // Not under a compute budget either: it would count the kept buffers without the
        // running graph's, and might drop other graphs than the next step would.
        if !enabled()
            || self.stages.len() != 1
            || self.mixed_kept <= LENT_MIXED_KEPT + 1
            || self.compute_budget.first().is_some_and(|&b| b > 0)
            || shape.iter().any(|g| !g.captures.is_empty())
        {
            return None;
        }
        let st = &self.stages[0];
        if st.backend.is_meta() {
            return None;
        }
        let ahead = next_groups(groups, next, table)?;
        let arch = &*self.model.hp.arch;
        let next = with_kv(batch_shape(&ahead, table, Some(arch)), &ahead, &st.kv);
        (next.as_slice() != shape && st.kept(&next).is_none()).then_some(next)
    }

    /// Builds stage `s`'s graph for the prompt shape `next` while its graph for `running`
    /// computes: `running` set aside meanwhile (no eviction or out-of-memory retry may take
    /// it) with the cap on kept mixed graphs one lower, so the same kept graph is evicted as
    /// when the next step builds it; then put back as the most recent. A failed build only
    /// logs: the next step builds it itself.
    pub(super) fn build_prompt_ahead(
        &mut self,
        s: usize,
        running: &[GroupShape],
        next: &[GroupShape],
    ) {
        let Some(i) = self.stages[s].mixed.iter().position(|(x, _)| x == running) else {
            return;
        };
        let kept = self.stages[s].mixed.remove(i);
        self.mixed_aside = 1;
        super::verify_ahead::set_building_ahead(true);
        let r = self.build_kept(s, next);
        super::verify_ahead::set_building_ahead(false);
        self.mixed_aside = 0;
        self.stages[s].mixed.push(kept);
        if let Err(e) = r {
            tracing::debug!(stage = s, error = %e, "prompt graph built ahead: failed");
        }
    }
}
