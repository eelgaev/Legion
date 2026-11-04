//! A layer split's verify commit queued beside the first MTP draft pass instead of before it
//! (#814).
//!
//! After a verify step [`Executor::commit`] queued every stage's commit graph (the gated
//! DeltaNet layers' conv windows and delta rule over the kept tokens) one stage after another
//! from the host, and only then did the MTP drafter queue its first pass on the last GPU. On
//! Qwen3.8-Flash-Next over four V100s (C=1, MTP k=2) each stage's commit took the host about
//! 180 µs to queue, so the first draft pass started ~0.9 ms after the verify step's read-back,
//! with the last GPU idle until then: its own commit started only after the other three had
//! been queued.
//!
//! The draft passes read no recurrent state (the head is an attention layer with a KV cache of
//! its own), so here the commit only records the kept counts; the first draft pass is queued
//! on the last GPU at once, and while it computes the host queues the other stages' commits
//! (their GPUs are idle) and then the last stage's, behind the pass's reads on its stream. The
//! host waits for those reads on an event, not for the stream, so the last GPU's commit runs
//! while the host takes the picks and fills the next pass. Every commit runs the same graph on
//! the same states and kept counts, on its stage's stream before that stage's next step, so
//! the states and the outputs are byte-identical. Anything else that may touch the states
//! (the next step, [`Executor::wait_commit`], another commit) queues the held commits first.
//!
//! With `--commit-last-late` (off by default) the last stage's commit waits one pass more, for
//! the second draft pass's reads ([`last_now`]; by default it queues behind the first's): behind the first pass it ran in front of the second one on the last GPU, while
//! after the second the last GPU idles until the next step's last stage. The draft passes read
//! no recurrent state, so that changes no bits either.
//!
//! Only for a layer split on one machine with MTP drafting (no tensor or GLM parallel steps,
//! no pipelined rounds, no deferred commits, which run on one stage only). On by default on
//! compute capability 7.x GPUs; `legion serve --no-commit-beside-draft` keeps the commit
//! before the first pass.

use super::{Executor, ModelError, Stage};
use crate::ggml::{Backend, GgmlError};
use crate::model::core::spec::CommitGraphs;
use crate::model::core::state::StatePool;
use crate::model::core::{Model, SpecPool};

#[cfg(test)]
mod tests;

/// Whether a verify step's commit waits for the first MTP pass (see the module docs): the
/// setting (`on`), MTP drafting (`mtp`), several stages, and none of the paths whose commits
/// run elsewhere (`other`: tensor or GLM parallel steps, a remote half, pipelined rounds, or
/// commits deferred into the next verify step).
pub(super) fn wanted(on: bool, mtp: bool, n_stages: usize, other: bool) -> bool {
    on && mtp && n_stages > 1 && !other
}

/// Whether a pass queues the last stage's held commit behind its reads: always, unless `late`
/// (the setting) and this pass just queued the other stages' commits (`front`, the round's
/// first pass). Then it waits for the next pass's reads, or for whatever queues held commits
/// first (the next step, [`Executor::wait_commit`], ...): queued behind the first pass, the
/// commit sat in front of the second pass on the last GPU, while behind the second one the
/// last GPU has nothing else to run until the next step reaches it (#814).
pub(super) fn last_now(late: bool, front: bool) -> bool {
    !(late && front)
}

/// A verify step's commit not queued yet: its kept counts and the stages still to queue it.
#[derive(Debug, Default)]
pub(super) struct HeldCommit {
    commits: Vec<(usize, usize)>,
    /// Stage indices, in the order they are queued.
    stages: Vec<usize>,
    /// The error of a commit queued where it could not be returned
    /// ([`Executor::wait_commit`]), for the next call that can.
    error: Option<GgmlError>,
}

impl HeldCommit {
    /// Whether any stage still waits for its commit.
    pub(super) fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Keeps `e`, a held commit's error that could not be returned, for the next call that can.
    pub(super) fn fail(&mut self, e: GgmlError) {
        self.error.get_or_insert(e);
    }

    /// Takes the stages `which` picks out of the held set, in order, with the kept counts.
    pub(super) fn take(
        &mut self,
        which: impl Fn(usize) -> bool,
    ) -> (Vec<usize>, Vec<(usize, usize)>) {
        let (taken, kept) = std::mem::take(&mut self.stages)
            .into_iter()
            .partition(|&s| which(s));
        self.stages = kept;
        let commits = if self.stages.is_empty() {
            std::mem::take(&mut self.commits)
        } else {
            self.commits.clone()
        };
        (taken, commits)
    }
}

/// Queues a stage's commit of `commits` (as [`Executor::commit`] does), from its parts.
pub(super) fn queue(
    spec: Option<&SpecPool>,
    backend: &mut Backend,
    kept: &mut CommitGraphs,
    rec: &StatePool,
    commits: &[(usize, usize)],
    model: &Model,
) -> Result<(), GgmlError> {
    let spec = spec.expect("commit without Executor::enable_spec");
    // SAFETY: as in `Executor::commit`: the stage's kept commit graphs are dropped before its
    // pools and the model's weights.
    unsafe { spec.commit(backend, kept, rec, commits, model) }
}

/// [`queue`] on stage `st`.
pub(super) fn queue_stage(
    st: &mut Stage,
    commits: &[(usize, usize)],
    model: &Model,
) -> Result<(), GgmlError> {
    queue(
        st.spec.as_ref(),
        &mut st.backend,
        &mut st.commits,
        &st.rec,
        commits,
        model,
    )
}

impl Executor {
    /// Whether a verify step's commit is held for the first MTP pass now ([`wanted`]).
    pub(super) fn holds_commit(&self) -> bool {
        wanted(
            self.commit_beside_draft,
            self.mtp_enabled,
            self.stages.len(),
            self.tp.is_some()
                || self.remote.is_some()
                || self.par.pool.is_some()
                || self.par.every_step
                || self.pipe_run.is_some()
                || !self.pipe.is_empty()
                || self.defers_commits(),
        )
    }

    /// Holds a verify step's commit of `commits` for the next MTP pass ([`Self::holds_commit`]).
    pub(super) fn hold_commit(&mut self, commits: &[(usize, usize)]) {
        self.held_commit.commits = commits.to_vec();
        self.held_commit.stages = (0..self.stages.len()).collect();
    }

    /// Queues the held commits of the stages `which` picks; on an error, the caller waits for
    /// the stages queued so far (`commit_pending` stays set).
    pub(super) fn queue_held(&mut self, which: impl Fn(usize) -> bool) -> Result<(), ModelError> {
        if let Some(e) = self.held_commit.error.take() {
            return Err(e.into());
        }
        if self.held_commit.is_empty() {
            return Ok(());
        }
        let (stages, commits) = self.held_commit.take(which);
        for s in stages {
            queue_stage(&mut self.stages[s], &commits, &self.model)?;
        }
        Ok(())
    }

    /// [`Self::queue_all_held`] where no error can be returned: the error is kept for the next
    /// call that can (the next step or commit).
    pub(super) fn queue_all_held_later(&mut self) {
        if self.held_commit.is_empty() {
            return;
        }
        let (stages, commits) = self.held_commit.take(|_| true);
        for s in stages {
            if let Err(e) = queue_stage(&mut self.stages[s], &commits, &self.model) {
                self.held_commit.fail(e);
                return;
            }
        }
    }

    /// Queues every held commit (before anything else touches the states or the streams).
    pub(super) fn queue_all_held(&mut self) -> Result<(), ModelError> {
        self.queue_held(|_| true)
    }

    /// Whether a verify step's commit is queued beside the first MTP draft pass on a layer
    /// split (see [`commit_hold`](self); byte-identical).
    pub fn set_commit_beside_draft(&mut self, on: bool) {
        self.commit_beside_draft = on;
    }

    /// Whether the last stage's held commit waits for the second draft pass's reads instead of
    /// the first's ([`last_now`]; byte-identical).
    pub fn set_commit_last_late(&mut self, on: bool) {
        self.commit_last_late = on;
    }
}
