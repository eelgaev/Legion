//! Verify steps' own share of a stage's kept mixed graphs (#858, `legion serve
//! --verify-kept-graphs`).
//!
//! A stage keeps its last [`super::MIXED_KEPT`] mixed graphs (prompt chunks and verify
//! batches together). With MTP at several sequences the verify batch's shape follows the
//! sequences' padded pool counts, which grow with the context: a round of requests walks
//! through a handful of verify shapes, and the next round (or a new request's prompt chunks in
//! between) walks the same ones again. Three kept graphs held only the last few, so every
//! shape came back as a new graph: built, planned, allocated in an evicted graph's buffer (new
//! addresses, so the stage-graph capture store does not match either) and captured again.
//!
//! With a verify budget, verify-only shapes (no prompt chunk) keep up to that many graphs of
//! their own, evicting only each other, and prompt shapes keep their [`super::MIXED_KEPT`]
//! apart. A graph kept is the same graph run again: the same kernels, the same bits.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use crate::model::core::GroupShape;

#[cfg(test)]
mod tests;

/// Verify shapes each stage keeps apart from prompt shapes (0: none, verify and prompt
/// shapes share [`super::MIXED_KEPT`], as before #858).
static KEPT: AtomicUsize = AtomicUsize::new(0);

/// Sets [`KEPT`] for the steps from now on (`legion serve --verify-kept-graphs`).
pub fn set_kept(n: usize) {
    KEPT.store(n, Relaxed);
}

/// See [`set_kept`].
pub fn kept() -> usize {
    KEPT.load(Relaxed)
}

/// Whether `shape` is a verify step's: several tokens for some sequence, but no prompt chunk
/// (every group verifies or decodes one token).
pub(super) fn is_verify(shape: &[GroupShape]) -> bool {
    shape.iter().all(|g| g.verify || g.n_tok == 1) && shape.iter().any(|g| g.verify)
}

/// Before a new mixed graph is kept (a verify step's when `verify`): the index of the kept
/// one to evict (`kept` says, oldest first, which are verify graphs), if any. With
/// `verify_kept` 0 the oldest goes once `mixed_kept` are kept; otherwise the oldest of the new
/// one's kind, once its kind keeps its limit (`verify_kept` verify graphs, `mixed_kept`
/// others).
pub(super) fn evict(
    kept: &[bool],
    verify: bool,
    mixed_kept: usize,
    verify_kept: usize,
) -> Option<usize> {
    if verify_kept == 0 {
        return (kept.len() >= mixed_kept).then_some(0);
    }
    let cap = if verify { verify_kept } else { mixed_kept };
    let mut same = kept.iter().enumerate().filter(|&(_, &v)| v == verify);
    let oldest = same.next()?.0;
    (1 + same.count() >= cap).then_some(oldest)
}

/// The most a stage's kept verify graphs hold in their own compute buffers by default
/// ([`default_budget`]): at det C=4 on the GB10 (Qwen3.8-Flash-Next NVFP4 + MTP) its ~30 verify
/// shapes took ~1.4 GiB, which this keeps whole.
pub const BUDGET_MAX: usize = 2 << 30;

/// A stage's verify budget where `--verify-kept-mib` is not given: an eighth of the memory its
/// device had free after the load, at most [`BUDGET_MAX`]. At 16 sequences each verify graph is
/// several times a C=4 one, so the count cap alone (32 by default on integrated GPUs) let the
/// kept set grow by gigabytes; past the budget the least recently run verify graphs go, and the
/// rest stay kept.
pub fn default_budget(free: usize) -> usize {
    (free / 8).clamp(1, BUDGET_MAX)
}

/// Before a new verify graph is built under a byte budget: whether the kept verify graphs'
/// `bytes` and `next` (what the newest one holds, about what the new one will) pass `budget`
/// (0: no budget), so the oldest one goes first and its buffer is reused for the new graph
/// rather than a new one allocated and an old one freed after.
pub(super) fn at_budget(verify: bool, bytes: usize, next: usize, budget: usize) -> bool {
    verify && budget > 0 && bytes + next > budget
}

/// After a graph was kept: which kept verify graphs to drop (indices into `kept`, oldest first;
/// `kept` lists `(is verify, buffer bytes)` oldest first) so that the verify graphs' buffers fit
/// `budget` (0: no budget). The newest verify graph stays even alone over the budget (it was
/// just built for the step about to run); prompt graphs are not counted.
pub(super) fn over_budget(kept: &[(bool, usize)], budget: usize) -> Vec<usize> {
    if budget == 0 {
        return Vec::new();
    }
    let verify: Vec<usize> = (0..kept.len()).filter(|&i| kept[i].0).collect();
    let mut total: usize = verify.iter().map(|&i| kept[i].1).sum();
    let mut drop = Vec::new();
    for &i in verify.iter().take(verify.len().saturating_sub(1)) {
        if total <= budget {
            break;
        }
        total -= kept[i].1;
        drop.push(i);
    }
    drop
}

impl super::Executor {
    /// Per stage: the most its kept verify graphs (`--verify-kept-graphs`) hold in their own
    /// compute buffers before the least recently run go (0: no limit). Only with a verify
    /// share of their own, not on meta backends, nor while a long prompt borrows the expert
    /// cache (verify graphs then share the stage's one kept mixed graph).
    pub fn set_verify_budget(&mut self, per_stage: Vec<usize>) {
        self.verify_budget = per_stage;
    }

    /// The bytes stage `s`'s kept verify graphs hold in their own compute buffers.
    pub fn verify_buffer_bytes(&self, s: usize) -> usize {
        self.stages[s]
            .mixed
            .iter()
            .filter(|(x, _)| is_verify(x))
            .map(|(_, k)| k.allocator_size())
            .sum()
    }

    /// After stage `s` kept a mixed graph: drops its least recently run verify graphs while
    /// the verify graphs' buffers pass `budget` ([`over_budget`]).
    pub(super) fn fit_verify_budget(&mut self, s: usize, budget: usize) {
        let st = &mut self.stages[s];
        let kept: Vec<(bool, usize)> = st
            .mixed
            .iter()
            .map(|(x, k)| (is_verify(x), k.allocator_size()))
            .collect();
        let drop = over_budget(&kept, budget);
        if drop.is_empty() {
            return;
        }
        for &i in drop.iter().rev() {
            st.mixed.remove(i);
        }
        let verify_mib = self.verify_buffer_bytes(s) >> 20;
        if crate::ggml::stage_graph::stats::logging() {
            tracing::info!(
                stage = s,
                dropped = drop.len(),
                verify_mib,
                budget_mib = budget >> 20,
                "kept verify graphs over the verify budget: least recently run dropped \
                 (--stage-graph-stats)"
            );
        }
    }
}
