//! `--prefix-cache-interval` save points inside a prompt step (#46).
//!
//! A spaced save point ([`crate::engine::saved_states::save_points`]) used to end its prompt
//! chunk, so the state there could be copied after the step: each point cost a step, ~8-10%
//! of prefill at a 2048-token interval. Where the executor can save recurrent states inside a
//! step ([`Executor::can_capture_states`]), a spaced point no longer cuts the chunk; the step
//! saves the state there itself ([`crate::model::core::capture`]), and the chunks are the ones
//! a run without the interval takes. Anchors (a branch point, a prompt's last full block) still
//! end their chunk, as before.
//!
//! Exactness: the steps are the ones a run without the interval takes, so every state and
//! output is bitwise the same as with no interval. A captured state is not bitwise the state a
//! cut at the same point gives: a matmul's sums depend on the step's token count (stream-k
//! splits), so the cut's shorter step already moves the layer-0 conv history. The cut was never
//! the exact reference; `--prefix-capture-cut` brings it back for such checks, and
//! `--prefix-state-digests` logs saved and resumed states' digests to compare runs.

use super::{Seq, Worker};
use crate::engine::saved_states::Kind as SaveKind;
use crate::model::core::capture;
use crate::model::core::{Pool, StateCapture, fnv1a};

impl Worker {
    /// Whether `seq`'s spaced save points are saved inside its prompt steps.
    pub(super) fn captures_saves(&self, seq: &Seq<'_>) -> bool {
        self.prefix_interval > 0
            && !self.capture_cut
            && seq.media().is_none()
            && self.exec.can_capture_states()
    }

    /// Where `seq`'s prompt chunk, planned to end at `end`, ends: at its first save point
    /// before `end` that the step cannot save inside it ([`capture::offset`]); `end` if none.
    pub(super) fn save_cut(&self, seq: &Seq<'_>, end: usize) -> usize {
        chunk_end(&seq.saves, seq.n_past, end, self.captures_saves(seq))
    }

    /// Records the save points inside the step's prompt chunks `chunks` (`(prefilling index,
    /// tokens)`, as [`Self::save_cut`] left them) as saved and asks the executor to save them;
    /// returns what it asked for [`Self::settle_captures`].
    pub(super) fn ask_captures(
        &mut self,
        prefilling: &mut [Seq<'_>],
        chunks: &[(usize, usize)],
    ) -> Vec<StateCapture> {
        let mut asked: Vec<StateCapture> = Vec::new();
        for &(i, len) in chunks {
            let seq = &mut prefilling[i];
            let end = seq.n_past + len;
            let n = seq.saves.iter().take_while(|&&(at, _)| at < end).count();
            if n == 0 {
                continue;
            }
            let points: Vec<_> = seq.saves.drain(..n).collect();
            for (at, kind) in points {
                debug_assert_eq!(kind, SaveKind::Spaced, "only spaced points inside a chunk");
                if let Some(s) = self.saved.store(&seq.cache_key()[..at], kind, None) {
                    if let Some(b) = s.dropped {
                        self.pool.decref(b);
                    }
                    // A slot taken again in this step holds the later point.
                    asked.retain(|c| c.saved != s.slot);
                    asked.push(StateCapture {
                        slot: seq.slot,
                        at,
                        saved: s.slot,
                    });
                }
            }
        }
        if !asked.is_empty() {
            self.exec.set_state_captures(asked.clone());
        }
        asked
    }

    /// After the step: forgets the asked saves the step did not take (or all of them when it
    /// failed), whose slots hold no such state.
    pub(super) fn settle_captures(&mut self, asked: Vec<StateCapture>, ran: bool) {
        if asked.is_empty() {
            return;
        }
        let applied = self.exec.take_applied_captures();
        for c in asked {
            if !ran || !applied.contains(&c) {
                tracing::debug!(at = c.at, "spaced save not taken by the step");
                if let Some(b) = self.saved.take(c.saved) {
                    self.pool.decref(b);
                }
            } else if self.state_digests {
                self.log_state_digest(Pool::Saved, c.saved, c.at, "captured in the step");
            }
        }
    }

    /// `--prefix-state-digests`: logs the digests of slot `slot` of `pool`, the state after
    /// `at` tokens ([`crate::model::core::StatePool::slot_digests`]): their combined digest,
    /// and each tensor's, so two runs show the first layer whose state differs.
    pub(super) fn log_state_digest(&mut self, pool: Pool, slot: usize, at: usize, what: &str) {
        let digests = self.exec.state_digests(pool, slot);
        let all: Vec<u8> = digests.iter().flat_map(|(_, d)| d.to_le_bytes()).collect();
        let parts: Vec<String> = digests
            .iter()
            .map(|(name, d)| format!("{name}={:08x}", d >> 32))
            .collect();
        tracing::info!(
            at,
            what,
            digest = %format!("{:016x}", fnv1a(&all)),
            parts = %parts.join(" "),
            "prefix state digest"
        );
    }
}

/// [`Worker::save_cut`] of a prompt chunk from `n_past` planned to end at `end`, whose save
/// points ahead are `saves` (ascending): the first point before `end` where the chunk must end,
/// an anchor or (`inside` false: the step saves no state inside itself) any point, or a spaced
/// one the step cannot save inside the chunk ([`capture::offset`]); `end` if none.
pub(super) fn chunk_end(
    saves: &[(usize, SaveKind)],
    n_past: usize,
    mut end: usize,
    inside: bool,
) -> usize {
    let cut = |end: usize| {
        saves
            .iter()
            .take_while(|&&(at, _)| at < end)
            .find(|&&(at, kind)| {
                let len = end - n_past;
                !(inside
                    && kind == SaveKind::Spaced
                    && capture::offset(n_past, len, at).is_some_and(|o| o < len))
            })
            .map(|&(at, _)| at)
    };
    // A cut can make an earlier point too close to the chunk's new end: cut again.
    while let Some(at) = cut(end) {
        end = at;
    }
    end
}

#[cfg(test)]
mod tests;
