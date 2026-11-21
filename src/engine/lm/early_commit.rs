//! A verify step's commit sent to the other ranks of tensor parallelism over machines as soon
//! as its accept counts are known (#1176, [`crate::model::core::Executor::mirror_commit`]):
//! before, it went out with rank 0's own commit, after the step's text (the detokenizer, the
//! stream sends, the stop strings), so the others started their commit that much later.
//!
//! Why the early mirror is the same commit. The engine verifies every sequence's drafts first
//! (`verify_drafts`: it reads only that sequence's rows and sampler, so the order of the
//! sequences does not matter), and sends the pairs only for a step in which no sequence can
//! end ([`plain`] and [`may_end`] over what `Active::advance` checks). A sequence that does
//! not end commits exactly its appended tokens, which the pairs hold ([`pairs`]). One that ends
//! anyway (a cancel between the check and its advance) was committed on the other ranks:
//! this rank then commits the same pairs, so every rank keeps the same states, and only its
//! end save (whose state would hold the tokens after its end) is skipped ([`end_state`]).
//! Steps where a sequence may end send the commit as before.

use super::{Active, Verified};

/// Whether `a`'s text cannot end it before its tokens do: no grammar (which can end it on any
/// token) and no stop strings (which only the detokenized text shows).
pub(super) fn plain(a: &Active<'_>) -> bool {
    a.constraint.is_none() && !a.text.has_stops()
}

/// Whether `a` (a [`plain`] sequence of `start` tokens) may end on one of the tokens a verify
/// step appends to it, `appended`, in `Active::advance`: cancelled, a stop token, or its
/// length or context cap.
pub(super) fn may_end(a: &Active<'_>, start: usize, appended: &[u32], stop_tokens: &[u32]) -> bool {
    a.cancelled()
        || ends_on(
            appended,
            |t| !a.ignore_eos && stop_tokens.contains(&t),
            a.generated,
            a.max_tokens,
            start,
            a.max_ctx,
        )
}

/// Whether advancing a sequence that generated `generated` tokens and holds `start` by
/// `appended` reaches an end: a token for which `stop` holds, or [`at_limit`] after one of
/// them (the last one reaches the most).
pub(super) fn ends_on(
    appended: &[u32],
    stop: impl Fn(u32) -> bool,
    generated: u32,
    max_tokens: u32,
    start: usize,
    max_ctx: usize,
) -> bool {
    let n = appended.len();
    appended.iter().any(|&t| stop(t))
        || (n > 0
            && at_limit(
                generated.saturating_add(n as u32),
                max_tokens,
                start + n,
                max_ctx,
            ))
}

/// `Active::advance`'s length end: `generated` tokens with the next fed back at `n_past`.
pub(super) fn at_limit(generated: u32, max_tokens: u32, n_past: usize, max_ctx: usize) -> bool {
    generated >= max_tokens || n_past >= max_ctx
}

/// The commit pairs of a verify group from slot `base` whose sequences all keep their
/// appended tokens.
pub(super) fn pairs(base: usize, verified: &[Verified]) -> Vec<(usize, usize)> {
    (verified.iter().enumerate())
        .map(|(i, v)| (base + i, v.o.appended.len()))
        .collect()
}

/// Whether the pairs `sent` hold every pair of `commits`, the step's commits as the engine
/// found them (the sequences that did not end).
pub(super) fn covers(sent: &[(usize, usize)], commits: &[(usize, usize)]) -> bool {
    commits.iter().all(|c| sent.contains(c))
}

/// The tokens a finished sequence's recurrent state holds after its step, for its end save:
/// after a verify step `start` (finished sequences are not committed), or none to save when
/// its commit went out early (`committed`: its state holds tokens after its end); after a
/// decode step `n_past`.
pub(super) fn end_state(
    verify: bool,
    committed: bool,
    start: usize,
    n_past: usize,
) -> Option<usize> {
    match (verify, committed) {
        (false, _) => Some(n_past),
        (true, false) => Some(start),
        (true, true) => None,
    }
}

#[cfg(test)]
mod tests;
