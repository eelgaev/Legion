//! The output head of a greedy decode or verify step split over the two ranks of tensor
//! parallelism (`legion serve --tp-head-split`, #1176).
//!
//! The LM head reads its whole weight once per step whatever the batch (on a 248K vocabulary
//! of 2,560 values a row, 1.27 GB of BF16), and under tensor parallelism only rank 0 ran it
//! while the other rank waited for the next step. Split, each rank multiplies its own share of
//! the vocabulary's rows ([`range`]) and finds each output row's best candidate among them
//! ([`crate::model::cluster_select::argmax_best`]); rank 1 answers the step with its
//! candidates and rank 0 keeps the better of the two per row ([`pick`]).
//!
//! The pick is the whole head's: every logit is the same dot product as before (a row of the
//! weight with the same input, rows do not mix), and the candidates compare as the device
//! argmax orders them, by value and then by `argmax_f32`'s tie rank
//! ([`crate::model::cluster_select::argmax_rank`]). That rank depends on a column modulo 1,024
//! and on its multiple, so rank 1's share starts at a multiple of 1,024 ([`ALIGN`]) and its
//! columns keep their order.
//!
//! Only steps whose rows all come back as device argmaxes split ([`fits`]): a sampled row or a
//! pick below `min_tokens` needs logits of the whole vocabulary on rank 0.

use super::{Group, Logits};
use crate::model::cluster_select::argmax_rank;

#[cfg(test)]
mod tests;

/// Rank 1's share starts at a multiple of this many vocabulary rows (see the module doc).
pub(crate) const ALIGN: usize = 1024;

/// Vocabulary rows a share is a multiple of, so that each share's matmul takes the kernels
/// the whole head takes.
const ROWS: usize = 128;

/// Whether a head of `n_vocab` rows splits: both shares whole tiles of [`ROWS`] rows and at
/// least [`ALIGN`] rows each.
pub(crate) fn vocab_fits(n_vocab: usize) -> bool {
    n_vocab.is_multiple_of(ROWS) && n_vocab >= 4 * ALIGN
}

/// The vocabulary rows of share `half` (1: rank 0's, 2: rank 1's) of a head of `n_vocab` rows
/// ([`vocab_fits`]): the first row and how many.
pub(crate) fn range(n_vocab: usize, half: u8) -> (usize, usize) {
    let first = (n_vocab / 2).next_multiple_of(ALIGN);
    match half {
        1 => (0, first),
        _ => (first, n_vocab - first),
    }
}

/// Whether a step of `groups` may split its head: a decode or verify step (at most `max_tok`
/// tokens a sequence) with rows to return, all of them device argmaxes.
pub(crate) fn fits(groups: &[Group<'_>], max_tok: usize) -> bool {
    let mut rows = 0;
    for s in groups.iter().flat_map(|g| &g.seqs) {
        match s.logits {
            Logits::None => {}
            Logits::Greedy => rows += 1,
            Logits::FirstGreedy(n) => rows += n.min(s.tokens.len()),
            _ => return false,
        }
        if s.tokens.len() > max_tok {
            return false;
        }
    }
    rows > 0
}

/// A row's argmax from the two shares' candidates, each a column of the whole vocabulary
/// (`u32::MAX`: the share has none) and its value: the larger value, on equal values the
/// earlier column in the device argmax's tie order; `u32::MAX` when neither share has one.
pub(crate) fn pick(a: (u32, f32), b: (u32, f32)) -> u32 {
    match (a.0, b.0) {
        (_, u32::MAX) => a.0,
        (u32::MAX, _) => b.0,
        _ => {
            let b_wins =
                b.1 > a.1 || (b.1 == a.1 && argmax_rank(b.0 as usize) < argmax_rank(a.0 as usize));
            if b_wins { b.0 } else { a.0 }
        }
    }
}

/// [`pick`] for every row of a step: `mine` this rank's columns and values, `theirs` the
/// other rank's. `None` when the other rank's answer does not match the step's rows.
pub(crate) fn join(mine: (&[u32], &[f32]), theirs: (&[u32], &[f32])) -> Option<Vec<u32>> {
    let n = mine.0.len();
    if mine.1.len() != n || theirs.0.len() != n || theirs.1.len() != n {
        return None;
    }
    Some(
        (0..n)
            .map(|r| pick((mine.0[r], mine.1[r]), (theirs.0[r], theirs.1[r])))
            .collect(),
    )
}
