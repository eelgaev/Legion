//! Recurrent states saved inside a prompt step for prefix caching (#46,
//! [`crate::model::core::capture`]): the save points the caller asks of the next step, put on
//! the groups that can take them, and the ones the step took.

use std::collections::HashMap;

use super::Executor;
use crate::model::core::capture::{self, Capture};
use crate::model::core::{Group, GroupShape};

/// The state of the sequence in `slot` after its first `at` tokens, into saved slot `saved`
/// ([`super::Pool::Saved`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateCapture {
    pub slot: usize,
    pub at: usize,
    pub saved: usize,
}

impl Executor {
    /// Whether steps can save recurrent states inside a prompt step
    /// ([`Self::set_state_captures`]): an architecture whose graph does it, on this machine
    /// alone, with each stage's saved states where its graphs run (not moved to pinned host
    /// memory beside a GPU, [`Self::saved_states_to_host`]).
    pub fn can_capture_states(&self) -> bool {
        self.model.hp.arch.state_captures()
            && self.tp.is_none()
            && self.remote.is_none()
            && self
                .stages
                .iter()
                .all(|s| s.saved.is_host() == s.rec.is_host() && s.saved.n_slots() > 0)
    }

    /// Asks the next [`Self::step`] to save these states as it runs. A step takes the ones
    /// inside (or at the end of) a prompt group of one text sequence, aligned as
    /// [`capture::offset`] needs; [`Self::take_applied_captures`] lists them after it.
    pub fn set_state_captures(&mut self, captures: Vec<StateCapture>) {
        self.captures = captures;
        self.applied.clear();
    }

    /// The captures the last step took ([`Self::set_state_captures`]).
    pub fn take_applied_captures(&mut self) -> Vec<StateCapture> {
        std::mem::take(&mut self.applied)
    }

    /// Debug (`--prefix-state-digests`): the digests of slot `slot` of `pool` on every stage of
    /// this machine ([`crate::model::core::StatePool::slot_digests`]), once the work queued
    /// has run.
    pub fn state_digests(&mut self, pool: super::Pool, slot: usize) -> Vec<(String, u64)> {
        self.wait_commit();
        for s in &mut self.stages {
            s.backend.synchronize();
        }
        self.stages
            .iter()
            .flat_map(|s| match pool {
                super::Pool::Seq => s.rec.slot_digests(slot),
                super::Pool::Saved => s.saved.slot_digests(slot),
            })
            .collect()
    }

    /// Puts the pending captures on the step's micro-chunks' group shapes, and sets each
    /// captured saved slot's token history (qwen4exp's n-gram hash) as the state's.
    pub(super) fn annotate_captures(
        &mut self,
        micros: &[Vec<Group<'_>>],
        shapes: &mut [Vec<GroupShape>],
    ) {
        let pending = std::mem::take(&mut self.captures);
        if pending.is_empty() || !self.can_capture_states() {
            return;
        }
        // Each slot's tokens in this step, from its first position in it.
        let mut steps: HashMap<usize, (usize, Vec<u32>)> = HashMap::new();
        for g in micros.iter().flatten() {
            if let [seq] = &g.seqs[..] {
                let (_, toks) = steps.entry(g.slot).or_insert((seq.n_past, Vec::new()));
                toks.extend_from_slice(seq.tokens);
            }
        }
        for (groups, shapes) in micros.iter().zip(shapes.iter_mut()) {
            for (g, sh) in groups.iter().zip(shapes.iter_mut()) {
                let [seq] = &g.seqs[..] else { continue };
                if g.verify || seq.media.is_some() {
                    continue;
                }
                for c in pending.iter().filter(|c| c.slot == g.slot) {
                    let Some(o) = capture::offset(seq.n_past, seq.tokens.len(), c.at) else {
                        continue;
                    };
                    sh.captures.retain(|x| x.saved != c.saved);
                    sh.captures.push(Capture {
                        at: o as i64,
                        saved: c.saved,
                    });
                    let (start, toks) = &steps[&g.slot];
                    for s in &self.stages {
                        let before = s.rec.peek_tokens(g.slot);
                        s.saved.set_tokens(c.saved, &before, &toks[..c.at - start]);
                    }
                    self.applied.retain(|x| x.saved != c.saved);
                    self.applied.push(*c);
                }
                sh.captures.sort_by_key(|c| c.at);
            }
        }
    }
}

#[cfg(test)]
mod tests;
