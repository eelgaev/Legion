//! `--spec-mtp-ngram` in the worker (#387, [`crate::spec::mtp_ngram`]): before the MTP head
//! drafts, a sequence whose recent tokens occurred earlier may take that occurrence's
//! continuation as its drafts instead; its head request then only takes in the verified
//! positions.

use super::{Seq, draft_cap};
use crate::model::core::Executor;
use crate::spec::depth::DepthModel;
use crate::spec::mtp_ngram::{Choice, choose};

/// The step-wide part of the choice.
pub(super) struct NgramStep<'a> {
    /// The depth policy (`None` with `--spec-fixed`: a match is always drafted).
    pub depth: Option<&'a DepthModel>,
    /// Sequences drafting for the next step, and whether drafting is off for that many.
    pub batch: usize,
    pub gated: bool,
    /// Whether the head takes in the verified positions every step.
    pub take_in: bool,
    /// `--spec-k`: the deepest n-gram draft.
    pub ceiling: usize,
    /// The depth the policy chose for the batch, which the step verifies at least.
    pub shape: usize,
    pub max_ctx: usize,
}

impl Seq<'_> {
    /// Drafts the n-gram continuation into `self.draft` where it beats the head's drafts at
    /// depth `mtp_max` (this sequence's cap for the head), and returns the head request's
    /// cap: `mtp_max`, or 0 when the n-gram drafts.
    pub(super) fn ngram_drafts(
        &mut self,
        exec: &Executor,
        mtp_max: usize,
        step: &NgramStep<'_>,
    ) -> usize {
        self.draft_ngram = None;
        let Some(ng) = &mut self.ngram else {
            return mtp_max;
        };
        if step.gated || self.active.constrained() {
            return mtp_max;
        }
        let cap = draft_cap(
            step.ceiling,
            self.active.max_tokens,
            self.active.generated,
            step.max_ctx,
            self.n_past,
        );
        let Some(c) = ng.lookup(&self.history, cap) else {
            return mtp_max;
        };
        let depth = match step.depth {
            None => Some(c.tokens.len()),
            Some(d) => {
                let choice = Choice {
                    batch: step.batch,
                    take_in: step.take_in,
                    mtp_k: mtp_max,
                    mtp_gain: self.adaptive.gain(mtp_max),
                    shape: step.shape,
                    avail: c.tokens.len(),
                };
                choose(d, choice, |k| ng.gain(c.class, k))
            }
        };
        let Some(k) = depth else {
            ng.follow(c);
            return mtp_max;
        };
        self.draft.tokens.extend_from_slice(&c.tokens[..k]);
        self.draft_cap = k;
        self.draft_ngram = Some(c.class);
        // The per-layer n-gram rows (qwen4exp's table on the SSD) of the bonus token (the last
        // of `history`) and the drafts, read before the step needs them, as the head's are.
        let (bonus, before) = self.history.split_last().expect("the bonus token at least");
        let fed: Vec<u32> = std::iter::once(*bonus)
            .chain(self.draft.tokens.iter().copied())
            .collect();
        exec.model()
            .prefetch_ngram(&before[before.len().saturating_sub(8)..], &fed);
        0
    }

    /// After a verify step of `drafted` drafts, `accepted` accepted: whether they were the
    /// n-gram's (then they are counted there, not in the head's estimates).
    pub(super) fn ngram_verified(&mut self, drafted: usize, accepted: usize) -> bool {
        match (self.draft_ngram.take(), &mut self.ngram) {
            (Some(class), Some(ng)) => {
                ng.record(class, drafted, accepted);
                true
            }
            _ => false,
        }
    }
}
