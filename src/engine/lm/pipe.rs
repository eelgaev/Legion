//! Decoding groups in flight over the pipeline stages (#415, [`LmConfig::pipeline_groups`]).
//!
//! A whole decode or verify step crosses the stages one after another, so each GPU of a layer
//! split waits for the others most of the time. Here the decoding sequences are split into
//! groups of consecutive state slots; every [`Executor::pipe_round`] moves each group's step in
//! flight on to the next stage and starts the next waiting group's step on stage 0, all stages
//! computing at once. The group whose step leaves the last stage is sampled, committed and
//! drafted for ([`Worker::finish_step`], as a whole step's sequences are) and waits for its
//! next turn; that work runs while the round's other stages still compute when the executor
//! hands the step out early (#762). Each group runs exactly the steps it would run as a batch
//! of its own.
//!
//! Groups form only while nothing else runs: no prompt being read, the expert cache not lent,
//! at least `--pipeline-min-group` sequences per group ([`DEFAULT_MIN_GROUP`] unless given). A
//! prompt to read, or fewer or unequal groups as sequences finish, drain them: their steps in
//! flight finish, their sequences go back to the whole batch, in slots `0..` again, and whole
//! steps run until the groups form again.
//!
//! [`LmConfig::pipeline_groups`]: super::LmConfig::pipeline_groups
//! [`Executor::pipe_round`]: crate::model::core::Executor::pipe_round

use std::collections::VecDeque;
use std::time::Instant;

use super::{Seq, StepPlan, Worker, decode_group, fed_tokens};
use crate::model::core::Pool;

/// The decoding sequences while their groups are formed (see the module docs).
#[derive(Default)]
pub(super) struct Pipe<'t> {
    /// In state-slot order: group `i` holds slots `groups[i].base..` (gaps between groups
    /// where sequences finished).
    groups: Vec<PipeGroup<'t>>,
    /// The groups not in flight, in the order their steps start.
    ready: VecDeque<usize>,
}

struct PipeGroup<'t> {
    seqs: Vec<Seq<'t>>,
    /// Its first state slot.
    base: usize,
    /// Its step in flight's plan.
    plan: Option<StepPlan>,
}

impl<'t> Pipe<'t> {
    /// Whether groups are formed.
    pub(super) fn is_active(&self) -> bool {
        !self.groups.is_empty()
    }

    /// Sequences in the groups.
    pub(super) fn len(&self) -> usize {
        self.groups.iter().map(|g| g.seqs.len()).sum()
    }

    /// The sequences in the groups.
    pub(super) fn seqs(&self) -> impl Iterator<Item = &Seq<'t>> {
        self.groups.iter().flat_map(|g| g.seqs.iter())
    }

    /// Every sequence of the groups, which are gone (after a failure).
    pub(super) fn take_all(&mut self) -> Vec<Seq<'t>> {
        self.ready.clear();
        self.groups.drain(..).flat_map(|g| g.seqs).collect()
    }

    /// Whether the groups are still as [`Self::form`] would make them of their sequences: as
    /// many, at most one sequence apart.
    fn balanced(&self, n_groups: usize, min_group: usize) -> bool {
        let n = self.len();
        let sizes = self.groups.iter().map(|g| g.seqs.len());
        let (lo, hi) = sizes.fold((usize::MAX, 0), |(lo, hi), x| (lo.min(x), hi.max(x)));
        self.groups.len() == groups_of(n, n_groups, min_group) && hi - lo <= 1
    }

    /// Splits `decoding` (slots `0..` in order) into [`groups_of`] groups of consecutive
    /// sequences, at most one sequence apart, the larger first; their slots stay.
    fn form(&mut self, decoding: &mut Vec<Seq<'t>>, n_groups: usize, min_group: usize) {
        debug_assert!(self.groups.is_empty());
        let n = decoding.len();
        let k = groups_of(n, n_groups, min_group);
        let mut rest = std::mem::take(decoding).into_iter();
        let mut base = 0;
        for (i, size) in group_sizes(n, k).into_iter().enumerate() {
            let seqs: Vec<_> = rest.by_ref().take(size).collect();
            debug_assert!(seqs.iter().enumerate().all(|(j, s)| s.slot == base + j));
            self.groups.push(PipeGroup {
                seqs,
                base,
                plan: None,
            });
            self.ready.push_back(i);
            base += size;
        }
    }
}

/// Fewest sequences per group by default (`--pipeline-min-group`): a group of one costs a
/// round's fixed time (inputs, launches, sampling, the expert cache's work per stage) for a
/// single token, which made two sequences as two groups slower than one whole step on two
/// V100s (70 vs 75 tok/s). Over four V100s (#752) four groups of one fill every stage at
/// C=4 but were no faster: with MTP 110 tok/s as two groups of two, without it 94 against
/// 101 for whole steps (a round took ~9.5 ms against ~5 ms of a stage's share of a
/// one-sequence step). Three sequences with MTP gained (95 vs 79: three groups instead of
/// whole steps).
pub const DEFAULT_MIN_GROUP: usize = 2;

/// Groups `n` decoding sequences run as with `--pipeline-groups n_groups` and
/// `--pipeline-min-group min_group`: as many as asked, each at least `min_group` sequences
/// (at least one); below 2, whole steps.
pub(super) fn groups_of(n: usize, n_groups: usize, min_group: usize) -> usize {
    let k = n_groups.min(n / min_group.max(1));
    if k >= 2 { k } else { 0 }
}

/// Sizes of `k` groups of `n` sequences, at most one apart, the larger first.
pub(super) fn group_sizes(n: usize, k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    (0..k).map(|i| n / k + usize::from(i < n % k)).collect()
}

/// The state-slot moves that bring sequences at slots `old` (ascending) to slots `0..`, in an
/// order where no move overwrites a slot still to be read.
pub(super) fn compact_moves(old: &[usize]) -> Vec<(usize, usize)> {
    debug_assert!(old.windows(2).all(|w| w[0] < w[1]));
    // Each target is at most its source and below every later source: ascending is safe.
    (old.iter().enumerate())
        .filter(|&(new, &from)| from != new)
        .map(|(new, &from)| (from, new))
        .collect()
}

impl Worker {
    /// [`Worker::step`] with decoding groups in flight (see the module docs): a round of the
    /// groups' steps when they can run, otherwise (after draining any) a whole step.
    pub(super) fn step_piped<'t>(
        &mut self,
        pipe: &mut Pipe<'t>,
        decoding: &mut Vec<Seq<'t>>,
        prefilling: &mut Vec<Seq<'t>>,
    ) -> Result<(), String> {
        let can = prefilling.is_empty() && !self.lent && self.exec.can_pipe();
        if pipe.is_active() {
            if can
                && decoding.is_empty()
                && pipe.balanced(self.pipeline_groups, self.pipeline_min_group)
            {
                return self.pipe_round(pipe, true);
            }
            self.pipe_drain(pipe, decoding)?;
        }
        if !can
            || groups_of(
                decoding.len(),
                self.pipeline_groups,
                self.pipeline_min_group,
            ) == 0
        {
            return self.step(decoding, prefilling);
        }
        pipe.form(decoding, self.pipeline_groups, self.pipeline_min_group);
        tracing::debug!(
            groups = ?pipe.groups.iter().map(|g| g.seqs.len()).collect::<Vec<_>>(),
            "decoding groups formed"
        );
        self.exec.pipe_begin();
        self.pipe_round(pipe, true)
    }

    /// One round ([`crate::model::core::Executor::pipe_round`]): the next ready group's step
    /// enters (with `enter`), and the group whose step left the last stage is finished.
    fn pipe_round<'t>(&mut self, pipe: &mut Pipe<'t>, enter: bool) -> Result<(), String> {
        let batch = pipe.len();
        let next = if enter { pipe.ready.pop_front() } else { None };
        let mut empty = Vec::new();
        let plan = match next {
            Some(g) => {
                let seqs = &mut pipe.groups[g].seqs;
                let plan = self.plan_step(seqs, &mut empty, batch)?;
                if plan.is_none() {
                    // An empty group (never formed so: the round only moves the others).
                    pipe.ready.push_back(g);
                }
                plan.map(|p| (g, p))
            }
            None => None,
        };
        let t = Instant::now();
        let out = match &plan {
            Some((g, p)) => {
                let group = &pipe.groups[*g];
                let fed = fed_tokens(&group.seqs, p);
                // The verify rows that are padding (`pad_rows`), while the round fills inputs.
                let _pad = crate::model::core::pad_rows::register(
                    &fed,
                    group.seqs.iter().map(|s| 1 + s.draft.tokens.len()),
                );
                let step = [decode_group(&group.seqs, &fed, p, group.base)];
                self.exec.pipe_round_begin(Some((*g, &step)))
            }
            None => self.exec.pipe_round_begin(None),
        }
        .map_err(|e| e.to_string())?;
        // In steady state a round finishes one group's step: the depth policy's cost of it.
        let round_ms = t.elapsed().as_secs_f64() * 1e3;
        if let Some((g, p)) = plan {
            pipe.groups[g].plan = Some(p);
        }
        let mut exit = None;
        if let Some((g, out)) = out {
            // Sampled, committed and drafted for while the round's other stages may still
            // compute (#762): the exiting group's next step is ready when they are done.
            let t_exit = self.split_probe.is_some().then(Instant::now);
            let group = &mut pipe.groups[g];
            let plan = group
                .plan
                .take()
                .expect("a group's step in flight has its plan");
            let base = group.base;
            self.finish_step(
                plan,
                out,
                round_ms,
                &mut group.seqs,
                &mut empty,
                base,
                batch,
            )?;
            debug_assert!(empty.is_empty());
            pipe.ready.push_back(g);
            exit = t_exit.map(|t| t.elapsed());
        }
        self.exec.pipe_round_end().map_err(|e| e.to_string())?;
        // `--pipeline-mtp-split` (#765): the stages' piece times and the exiting step's work.
        if let (Some(probe), Some(exit)) = (&self.split_probe, exit)
            && let Ok(mut p) = probe.lock()
        {
            let (pieces, early) = self.exec.pipe_last_round();
            p.record(pieces, early, exit);
        }
        Ok(())
    }

    /// Lets every group's step in flight finish, then gives their sequences back to
    /// `decoding` in slots `0..`.
    fn pipe_drain<'t>(
        &mut self,
        pipe: &mut Pipe<'t>,
        decoding: &mut Vec<Seq<'t>>,
    ) -> Result<(), String> {
        while self.exec.pipe_in_flight() > 0 {
            self.pipe_round(pipe, false)?;
        }
        self.exec.pipe_end();
        debug_assert!(decoding.is_empty());
        let seqs = pipe.take_all();
        let old: Vec<usize> = seqs.iter().map(|s| s.slot).collect();
        let copies: Vec<_> = compact_moves(&old)
            .into_iter()
            .map(|(from, to)| (Pool::Seq, from, Pool::Seq, to))
            .collect();
        decoding.extend(seqs);
        for (i, s) in decoding.iter_mut().enumerate() {
            s.slot = i;
        }
        tracing::debug!(
            seqs = decoding.len(),
            moves = copies.len(),
            "decoding groups drained"
        );
        self.exec.copy_states(&copies).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests;
