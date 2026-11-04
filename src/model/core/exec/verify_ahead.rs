//! The next verify shape's graph built while the current verify step computes (#836).
//!
//! With MTP at several sequences, a verify step's shape lists the sequences' sparse-attention
//! key pools padded to 64 (qwen4exp); as the contexts grow that count moves up one padding step
//! every few rounds (~4.5 at 16 sequences on a GB10), and each new count is a shape no kept
//! graph has: the step first builds, plans and allocates its graph (~5-6 ms of host time before
//! its first kernel, the GPU idle), then runs it. Kept verify graphs (`--verify-kept-graphs`)
//! do not help, since the count only grows.
//!
//! So a one-stage verify step whose groups, each sequence `n_tok` positions further on (the
//! most a verify step advances it), would list other pools builds that shape's graph between
//! its launch and its wait, while the GPU computes (~250 ms a step there). When the count
//! moves, the graph is kept already. It is the same graph the step would build (the same
//! shape, the same builder), in a compute buffer of its own: only when it is built changes, not
//! the kernels or the bits.
//!
//! Only where verify shapes keep graphs of their own (`--verify-kept-graphs` above 0, the
//! default for qwen4exp's MTP head on an integrated GPU), not on meta backends (one shared
//! buffer per kind) and not while the expert cache is lent. The running step's graph is set
//! aside while the next one is built, so no eviction can drop it.
//!
//! The graph built ahead is captured too, right after it is built (its stage-graph pieces
//! recorded, nothing run: `Backend::precapture_kept`). Without that, the step that reaches the
//! new count ran it directly the first time, its ~3,000 kernels launched one by one with its
//! capture behind them: ~5 ms more GPU time than a replayed step (launch gaps, and the GPU
//! waiting on the host held back by the launch queue). With it, that step finds the captures
//! by the graph's words and launches them as any later step does: the same pieces of the same
//! graph recorded on the same stream, so the same bits.

use super::{Executor, Group, GroupShape, KV_BLOCK, LENT_MIXED_KEPT, ModelError, SeqInput};
use super::{batch_shape, verify_kept, with_kv};

#[cfg(test)]
mod tests;

thread_local! {
    /// A graph is being built ahead on this thread ([`Executor::build_ahead`]), for
    /// `--stage-graph-stats`' build times.
    static AHEAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the graph built now is built ahead ([`Executor::build_ahead`]).
pub(super) fn building_ahead() -> bool {
    AHEAD.with(|a| a.get())
}

/// Marks the graphs built from now on as built ahead, or not (also for
/// [`super::prompt_ahead`]).
pub(super) fn set_building_ahead(on: bool) {
    AHEAD.with(|a| a.set(on));
}

/// `groups` with every sequence `n_tok` positions further on (its tokens kept), the most a
/// verify step moves it, if the step verifies (no prompt chunk) and every sequence still fits
/// a `table`-block table (paged attention; `None`: gather attention, no limit).
pub(super) fn advanced<'b>(groups: &[Group<'b>], table: Option<usize>) -> Option<Vec<Group<'b>>> {
    let verify = groups.iter().any(|g| g.verify)
        && groups
            .iter()
            .all(|g| g.verify || g.seqs.iter().all(|s| s.tokens.len() == 1));
    if !verify {
        return None;
    }
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let mut seqs: Vec<SeqInput<'b>> = Vec::with_capacity(g.seqs.len());
        for s in &g.seqs {
            let n_past = s.n_past + s.tokens.len();
            if table.is_some_and(|w| (n_past + s.tokens.len()).div_ceil(KV_BLOCK) > w) {
                return None;
            }
            seqs.push(SeqInput {
                n_past,
                ..s.clone()
            });
        }
        out.push(Group {
            slot: g.slot,
            seqs,
            verify: g.verify,
        });
    }
    Some(out)
}

/// Whether `next` is `cur` with other sparse-attention pools only (listed, new, all picked)
/// and not `cur` itself: the shape a verify step of the same sequences takes a few steps on.
pub(super) fn pools_moved(cur: &[GroupShape], next: &[GroupShape]) -> bool {
    if cur.len() != next.len() || cur == next {
        return false;
    }
    cur.iter().zip(next).all(|(c, n)| {
        c.n_pool > 0
            && GroupShape {
                n_pool: c.n_pool,
                n_new: c.n_new,
                pick_all: c.pick_all,
                ..n.clone()
            } == *c
    })
}

impl Executor {
    /// The verify shape to build ahead while stage `s` (the only one) runs `shape` for
    /// `groups` ([`advanced`], [`pools_moved`]), if it is not kept yet.
    pub(super) fn verify_ahead(
        &self,
        groups: &[Group<'_>],
        table: Option<usize>,
        shape: &[GroupShape],
    ) -> Option<Vec<GroupShape>> {
        if self.stages.len() != 1
            || self.mixed_kept == LENT_MIXED_KEPT
            || verify_kept::kept() < 2
            || !verify_kept::is_verify(shape)
        {
            return None;
        }
        let st = &self.stages[0];
        if st.backend.is_meta() {
            return None;
        }
        let ahead = advanced(groups, table)?;
        let arch = &*self.model.hp.arch;
        let next = with_kv(batch_shape(&ahead, table, Some(arch)), &ahead, &st.kv);
        (pools_moved(shape, &next) && st.kept(&next).is_none()).then_some(next)
    }

    /// Builds stage `s`'s graph for `next` while its graph for `running` computes: `running`
    /// set aside meanwhile (no eviction, out-of-memory retry or budget drop may take it), then
    /// put back as the most recent, and then captured for its first run (`late`: that run
    /// writes its n-gram rows late, as this one does). A failed build only logs: the step
    /// builds it itself later.
    pub(super) fn build_ahead(
        &mut self,
        s: usize,
        running: &[GroupShape],
        next: &[GroupShape],
        late: bool,
    ) {
        let Some(i) = self.stages[s].mixed.iter().position(|(x, _)| x == running) else {
            return;
        };
        let kept = self.stages[s].mixed.remove(i);
        AHEAD.with(|a| a.set(true));
        let r: Result<(), ModelError> = self.build_kept(s, next);
        AHEAD.with(|a| a.set(false));
        let st = &mut self.stages[s];
        st.mixed.push(kept);
        // Over the verify graphs' own limit by the one set aside: the oldest other one goes.
        let limit = verify_kept::kept();
        let verify: Vec<usize> = (0..st.mixed.len())
            .filter(|&j| verify_kept::is_verify(&st.mixed[j].0))
            .collect();
        if verify.len() > limit {
            st.mixed.remove(verify[0]);
        }
        // The budgets with it counted again; the running graph, newest, stays.
        let verify_budget = self.verify_budget.get(s).copied().unwrap_or(0);
        if verify_budget > 0 {
            self.fit_verify_budget(s, verify_budget);
        }
        self.fit_compute_budget(s, true);
        if let Err(e) = r {
            tracing::debug!(stage = s, error = %e, "verify graph built ahead: failed");
            return;
        }
        self.capture_ahead(s, next, late);
    }

    /// Captures stage `s`'s kept graph for `next` before its first run
    /// ([`crate::ggml::Backend::precapture_kept`]), its n-gram rows input filled late when
    /// `late` (as [`Self::launch_stage`] passes it).
    fn capture_ahead(&mut self, s: usize, next: &[GroupShape], late: bool) {
        let parts = self.parts(s);
        let super::Stage {
            decode,
            mixed,
            backend,
            ..
        } = &mut self.stages[s];
        // Dropped again by the budgets: nothing to capture.
        let Some(kept) = super::kept_in(decode, mixed, next) else {
            return;
        };
        let ple = late
            .then(|| super::Inputs::from_parts(next, parts, |i| kept.tensor(i)).ple)
            .flatten();
        if late && ple.is_none() {
            return;
        }
        let done = backend.precapture_kept(kept, ple);
        tracing::debug!(stage = s, done, "verify graph built ahead: captured");
    }
}
