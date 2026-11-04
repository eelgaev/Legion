//! A pipelined step's next round, started while its current round still computes (#251).
//!
//! In a round of several stages ([`Executor::step`]) the stages rarely finish together: on a
//! 123K prompt one of them (mostly the first) ends 25-40 ms after the others. The host used to
//! wait for the slowest stage before building, filling and
//! handing over any stage's next micro-chunk, then did all of it in turn (~25 ms on 2 GPUs,
//! ~35 ms on 4) with every GPU idle. Now it waits for each stage on its own: as a stage's graph
//! finishes, its rows are read out, then every stage whose next micro-chunk no longer waits on
//! a running stage gets its graph built, its inputs filled and its residual handed over. At the
//! end of the round only what waited on the slowest stage is left.
//!
//! What stage `s`'s next micro-chunk waits on, as in the round's own input loop (later stages
//! first): its graph and inputs need its own graph done (they may take its memory) and the
//! stage after it already handed the residual over (the residual may share memory with this
//! stage's inputs, and a rebuilt graph may drop it); its hand-off needs the stage before it
//! done (the residual it copies) and its own inputs built.
//!
//! The expert cache's work after the round's passes still runs once every stage is done, and
//! the next round's graphs start then, as before: only host work moves, the order of every
//! device's work is unchanged, so outputs are byte-identical.

use super::{Executor, Group, GroupShape, ModelError};

#[cfg(test)]
mod tests;

/// How far a stage's next micro-chunk got while the round before computed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Early {
    /// Nothing done yet.
    #[default]
    No,
    /// Graph built and inputs filled; the residual not handed over yet.
    Filled,
    /// Ready to run: inputs filled and the residual handed over.
    Ready,
}

/// The next piece of early work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Work {
    /// Build stage `s`'s graph and fill its inputs.
    Fill(usize),
    /// Hand stage `s - 1`'s residual over to stage `s` (nothing for the first stage).
    HandOff(usize),
}

/// The state [`next_work`] decides on.
#[derive(Clone, Copy)]
pub(super) struct EarlyProgress<'a> {
    /// The next round's stages and micro-chunks ([`round_active`]).
    pub next: &'a [(usize, usize)],
    /// Per stage: whether it runs in the current round.
    pub ran: &'a [bool],
    /// Per stage: whether its graph of the current round is done and its rows read out.
    pub finished: &'a [bool],
}

/// The stages of round `round` of a step of `n_micro` micro-chunks over `n_stages` stages,
/// with their micro-chunks: stage `s` runs micro-chunk `round - s`.
pub(super) fn round_active(round: usize, n_stages: usize, n_micro: usize) -> Vec<(usize, usize)> {
    (0..n_stages)
        .filter_map(|s| {
            round
                .checked_sub(s)
                .filter(|k| *k < n_micro)
                .map(|k| (s, k))
        })
        .collect()
}

/// What can be done next for the next round, `None` when the rest waits on a running stage.
/// Later stages first, each one ready before the stage before it starts.
pub(super) fn next_work(p: EarlyProgress<'_>, early: &[Early]) -> Option<Work> {
    for &(s, _) in p.next.iter().rev() {
        match early[s] {
            Early::Ready => continue,
            Early::Filled => {
                // The residual of the stage before (which ran this round's micro-chunk `k`).
                let source_done = s == 0 || p.finished[s - 1];
                return source_done.then_some(Work::HandOff(s));
            }
            Early::No => {
                // Its own graph of this round first (the stage after it is ready: it was
                // visited first).
                let own_done = !p.ran[s] || p.finished[s];
                return own_done.then_some(Work::Fill(s));
            }
        }
    }
    None
}

impl Executor {
    /// Does every piece of the next round's input work ([`next_work`]) that no longer waits
    /// on a running stage, for the next round's micro-chunks `micros` / `shapes`.
    pub(super) fn early_inputs(
        &mut self,
        p: EarlyProgress<'_>,
        early: &mut [Early],
        micros: &[Vec<Group<'_>>],
        shapes: &[Vec<GroupShape>],
    ) -> Result<(), ModelError> {
        while let Some(work) = next_work(p, early) {
            let (s, k) = match work {
                Work::Fill(s) | Work::HandOff(s) => *p
                    .next
                    .iter()
                    .find(|a| a.0 == s)
                    .expect("a next-round stage"),
            };
            let t = std::time::Instant::now();
            match work {
                Work::Fill(_) => {
                    self.prepare(s, &shapes[k])?;
                    let prepared = t.elapsed();
                    self.fill_inputs(s, &micros[k], &shapes[k]);
                    early[s] = Early::Filled;
                    tracing::trace!(
                        stage = s,
                        prepare_ms = prepared.as_secs_f64() * 1e3,
                        ms = t.elapsed().as_secs_f64() * 1e3,
                        "early fill"
                    );
                }
                Work::HandOff(_) => {
                    self.hand_off(s, &shapes[k]);
                    early[s] = Early::Ready;
                    tracing::trace!(
                        stage = s,
                        ms = t.elapsed().as_secs_f64() * 1e3,
                        "early hand-off"
                    );
                }
            }
        }
        Ok(())
    }
}
