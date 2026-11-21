//! A DFlash prompt chunk's take-in left for a later draft round (#1133), as the MTP head's is
//! (#477, [`super::DeferredTakeIn`]).
//!
//! The drafter takes in a prompt's last rows only, its ring's window (2,064 rows for a DFlash
//! 2 drafter with a 2,048-token window). When that window spans a chunk boundary (an 8K prompt
//! over 4,096-token steps: 4,096 + 4,096 + 9 tokens; a 32K one over 16,384-token steps), the
//! earlier chunk's take-in (its rows' features, the fused projection, the window's K and V) ran
//! between that chunk's step and the last one, before the prompt's first token, and on every
//! rank of `--tp-draft-split`, whose other ranks ran it before the last step too. A round of
//! prompt chunks' take-ins only (no sequence decoding, none drafting) now waits for the next
//! round that does more, which runs it first: after the first token is out. At most
//! [`MAX_ROUNDS`] wait (each holds its step's captured rows). A take-in-only round draws nothing
//! from the others' sums, so each rank decides by itself, and every rank still runs the same
//! take-ins of a sequence in the same order before its first drafts: the same drafts and the
//! same text. `LEGION_DEBUG_DFLASH_DEFER_TAKE_IN=0` (on every rank) runs them at once again.
//!
//! A step of one prompt's chunk alone now leaves its captured rows on the device for a drafter
//! that reads them there ([`PromptRows`]), as a decode step does (#1181): no readback of the
//! window's rows after the step and no host gather and upload in the take-in, which then runs
//! at once (rows left on the device are gone by the next step) and only its GPU part. Such a
//! round waits only where the step read its rows back after all.

use std::ffi::OsStr;
use std::sync::Arc;

use crate::spec::{Draft, DraftRequest, ModelDrafter, SeqId};

/// The most rounds left waiting: a round past them runs them and itself at once. A prompt's
/// window spans its last two chunks, three over short ones; each round holds its step's rows.
pub(super) const MAX_ROUNDS: usize = 4;

/// Whether take-in-only rounds wait (the default); `LEGION_DEBUG_DFLASH_DEFER_TAKE_IN=0` runs
/// them at once, as before (for A/B runs only).
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("LEGION_DEBUG_DFLASH_DEFER_TAKE_IN").is_none_or(|v| v != "0")
    })
}

/// Which steps of one prompt's chunk alone leave their captured rows on the device for a
/// drafter that takes them in there ([`ModelDrafter::reads_device_rows`], #1133).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PromptRows {
    /// None: every prompt step reads them back (the way before).
    Host,
    /// The prompt's last chunk only; an earlier chunk's take-in still waits for the next round.
    Last,
    /// Every chunk (the default): an earlier chunk's take-in runs at once, on the device.
    All,
}

impl PromptRows {
    /// `LEGION_DEBUG_DFLASH_PROMPT_DEVICE_ROWS`: `0` [`Self::Host`], `last` [`Self::Last`],
    /// anything else or unset [`Self::All`].
    pub(super) fn parse(v: Option<&OsStr>) -> Self {
        match v.and_then(OsStr::to_str) {
            Some("0") => PromptRows::Host,
            Some("last") => PromptRows::Last,
            _ => PromptRows::All,
        }
    }

    /// Whether a lone chunk's step (`last`: the prompt's last one) leaves its rows on the device.
    pub(super) fn stays(self, last: bool) -> bool {
        match self {
            PromptRows::Host => false,
            PromptRows::Last => last,
            PromptRows::All => true,
        }
    }
}

/// [`PromptRows`] as `LEGION_DEBUG_DFLASH_PROMPT_DEVICE_ROWS` sets it (for A/B runs only).
pub(super) fn prompt_rows() -> PromptRows {
    static ROWS: std::sync::OnceLock<PromptRows> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        PromptRows::parse(std::env::var_os("LEGION_DEBUG_DFLASH_PROMPT_DEVICE_ROWS").as_deref())
    })
}

/// Whether a draft round waits for the next one: with `on`, rows on the host (a step's rows
/// left on the device are gone by the next step), a round of take-ins only (`maxes`, each
/// request's drafts asked for, all 0), and some request taking in rows (`reads`, each request's
/// first row the drafter reads and its rows: [`ModelDrafter::rows_read_from`] and `ran`). A
/// round that takes in nothing costs nothing to run now.
pub(super) fn defers(
    on: bool,
    device_rows: bool,
    maxes: impl IntoIterator<Item = usize>,
    reads: impl IntoIterator<Item = (usize, usize)>,
) -> bool {
    on && !device_rows
        && maxes.into_iter().all(|m| m == 0)
        && reads.into_iter().any(|(from, ran)| from < ran)
}

/// One request of a [`Round`]: what its [`DraftRequest`] named.
pub(super) struct Req {
    pub seq: SeqId,
    /// The sequence's tokens as the round saw them (the prompt; the first sampled token may
    /// have joined since).
    pub tokens: Vec<u32>,
    pub start: usize,
    pub ran: usize,
    /// Its first row among the step's.
    pub off: usize,
}

/// A take-in-only round left for later: the step's captured rows and the round's requests.
pub(super) struct Round {
    hidden: Arc<Vec<Vec<f32>>>,
    /// Positions the step ran (the rows of each captured layer).
    n_step: usize,
    reqs: Vec<Req>,
}

impl Round {
    pub(super) fn new(hidden: Arc<Vec<Vec<f32>>>, n_step: usize, reqs: Vec<Req>) -> Self {
        Round {
            hidden,
            n_step,
            reqs,
        }
    }

    /// `seq` is gone: its requests go (its drafter state went). Whether any are left.
    pub(super) fn release(&mut self, seq: SeqId) -> bool {
        self.reqs.retain(|r| r.seq != seq);
        !self.reqs.is_empty()
    }

    /// Runs the round on `d`: the same requests, take-ins only. The step's rows, unless
    /// another round still holds them.
    pub(super) fn run(
        self,
        d: &mut dyn ModelDrafter,
    ) -> (Result<(), String>, Option<Vec<Vec<f32>>>) {
        let mut outs: Vec<Draft> = self.reqs.iter().map(|_| Draft::default()).collect();
        let mut batch: Vec<DraftRequest<'_>> = (self.reqs.iter().zip(&mut outs))
            .map(|(r, out)| DraftRequest {
                seq: r.seq,
                tokens: &r.tokens,
                start: r.start,
                ran: r.ran,
                hidden: rows_of(&self.hidden, self.n_step, r.off, r.ran),
                max: 0,
                out,
            })
            .collect();
        let result = d.step(&mut batch);
        drop(batch);
        (result, Arc::try_unwrap(self.hidden).ok())
    }
}

/// Rows `off..off + n` of each captured layer of a step of `n_step` positions.
pub(super) fn rows_of(hidden: &[Vec<f32>], n_step: usize, off: usize, n: usize) -> Vec<&[f32]> {
    hidden
        .iter()
        .map(|h| {
            let w = h.len() / n_step.max(1);
            &h[off * w..(off + n) * w]
        })
        .collect()
}

#[cfg(test)]
mod tests;
