//! The DFlash 2 drafter's output head split over the two ranks of `--tp-draft-split` (#666).
//!
//! Under `--tp-draft-split` every rank runs each draft pass up to its last layer (the FFN sums
//! keep every rank's residual stream the same), and rank 0 alone ran the head while the other
//! rank waited: on Gemma 4 31B a 4-bit head of 262,144 rows of 5,376 values, about 0.8 GB read
//! a pass. Now each rank multiplies its share of the vocabulary's rows (the shares of the
//! target's `--tp-head-split`, [`crate::model::core::head_split::range`]) and takes its share's
//! top-k candidates and their logits; one exchange gives rank 0 both lists, and it keeps the
//! `k` best of the `2k` ([`merge`]).
//!
//! The candidates are the whole head's, in the same order and with the same logits: every
//! logit is the same dot product as before (a row of the weight with the same input; each
//! share is whole tiles of 128 rows, so its matmul takes the kernels the whole head takes), and
//! the top-k op ranks a candidate by its value and then by the lower column
//! ([`super::fused::topk_key`]). A share keeps the whole head's column order, so its top-k
//! holds every one of the whole head's top-k that falls in it; the merge ranks both lists by
//! the same key with the whole head's columns.
//!
//! The exchange is one all-reduce, the op of the pass's FFN sums (so a failed round's zero
//! sums cover it, [`super::round_sums`]): rank r's `[2k, rows]` block (its candidates' columns
//! as floats, exact below 2^24, then their logits) goes in the r-th half of a `[2k, 2 rows]`
//! tensor and `-0.0` in the other half, and `x + -0.0` is `x` for every float, so the sum is
//! both blocks bit for bit. It stays below the size `--tp-wire bf16` rounds
//! ([`crate::tp_net::BF16_MIN_VALUES`]).
//!
//! Only DFlash 2 (its selector reads the head's top-k and nothing else of the head) with
//! greedy drafts (`--spec-draft-temperature` 0), two ranks, and a head both shares of which the
//! top-k op takes ([`Split::takes`]); everything else keeps the whole head on rank 0.
//! `LEGION_DEBUG_TP_DRAFT_HEAD=0` keeps it too (A/B runs; every rank must set the same), and
//! `LEGION_DEBUG_TP_DRAFT_HEAD_CHECK=1` makes rank 0 also run the whole head and count the rows
//! whose candidates or logits differ from the merge's.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::ggml::types;
use crate::model::core::head_split::{range, vocab_fits};

use super::fused::{topk_fits, topk_key};

#[cfg(test)]
mod tests;

/// A head split over two ranks: the selector's top-k and the head's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Split {
    /// Candidates a row (the selector's top-k).
    pub k: usize,
    /// Rows of the head (the draft vocabulary).
    pub n_vocab: usize,
}

impl Split {
    /// The split of a head of `n_vocab` rows with `k` candidates a row, if its shares are
    /// whole tiles of rows ([`vocab_fits`]).
    pub fn new(k: usize, n_vocab: usize) -> Option<Self> {
        (k > 0 && vocab_fits(n_vocab)).then_some(Self { k, n_vocab })
    }

    /// Rank `rank`'s (0 or 1) share of the head's rows: the first and how many.
    pub fn share(&self, rank: usize) -> (usize, usize) {
        range(self.n_vocab, if rank == 0 { 1 } else { 2 })
    }

    /// Values of a pass's exchange for `rows` output rows: each rank's candidates and logits.
    pub fn values(&self, rows: usize) -> usize {
        2 * 2 * self.k * rows
    }

    /// Whether a pass of `rows` output rows splits its head: the whole head and both shares
    /// take the top-k op (so their candidates rank alike), and the exchange travels in F32.
    /// The same on every rank (a function of the pass's shape).
    pub fn takes(&self, rows: usize) -> bool {
        let fits = |n: usize| {
            let row = 4 * n;
            topk_fits(
                types::F32,
                [n as i64, rows as i64, 1, 1],
                [4, row, row * rows, row * rows],
                self.k as i64,
            )
        };
        rows > 0
            && self.values(rows) < crate::tp_net::BF16_MIN_VALUES
            && fits(self.n_vocab)
            && fits(self.share(0).1)
            && fits(self.share(1).1)
    }
}

/// Whether the split runs where it can: on unless `LEGION_DEBUG_TP_DRAFT_HEAD=0` (A/B runs;
/// every rank must set the same, or the ranks' passes exchange differently). Read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_TP_DRAFT_HEAD").is_none_or(|v| v != "0"))
}

/// Whether rank 0 also runs the whole head and compares (`LEGION_DEBUG_TP_DRAFT_HEAD_CHECK=1`).
pub(crate) fn checking() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("LEGION_DEBUG_TP_DRAFT_HEAD_CHECK").is_some_and(|v| v != "0")
    })
}

/// The `k` best of one row's candidates from the two shares, each `(column of the whole
/// head, logit)`: ranked as the top-k op ranks them ([`topk_key`]), best first.
pub(crate) fn merge(a: &[(u32, f32)], b: &[(u32, f32)], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = a.iter().chain(b).copied().collect();
    all.sort_unstable_by_key(|&(c, v)| std::cmp::Reverse(topk_key(v, c as usize)));
    all.truncate(k);
    all
}

/// One rank's `[2k, rows]` block of a pass's exchange: each row's `k` candidates' columns of
/// its share (as floats), then their logits.
#[cfg(test)]
pub(crate) fn pack(cols: &[u32], logits: &[f32], k: usize) -> Vec<f32> {
    cols.chunks_exact(k)
        .zip(logits.chunks_exact(k))
        .flat_map(|(c, l)| c.iter().map(|&c| c as f32).chain(l.iter().copied()))
        .collect()
}

/// Rank 0's candidates and their logits for every row (`[row][k]`, best first) from a pass's
/// exchange `all` (`[2k, 2 rows]`: rank 0's block, then rank 1's). Each rank's columns are of
/// its share (`shares`: first column and length of each); a column that is not one of its
/// share's (a rank whose round failed sent zeros) is left out, and a row short of `k` is filled
/// with column 0 at minus infinity, which the selector never prefers.
pub(crate) fn unpack(
    all: &[f32],
    k: usize,
    rows: usize,
    shares: [(usize, usize); 2],
) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(all.len(), 2 * 2 * k * rows, "the exchange's values");
    let list = |rank: usize, row: usize| -> Vec<(u32, f32)> {
        let at = (rank * rows + row) * 2 * k;
        let (first, len) = shares[rank];
        (0..k)
            .filter_map(|i| {
                let c = all[at + i];
                let ok = c.is_finite() && c >= 0.0 && c.fract() == 0.0 && (c as usize) < len;
                ok.then(|| ((first + c as usize) as u32, all[at + k + i]))
            })
            .collect()
    };
    let mut cols = Vec::with_capacity(rows * k);
    let mut logits = Vec::with_capacity(rows * k);
    for row in 0..rows {
        let mut best = merge(&list(0, row), &list(1, row), k);
        best.resize(k, (0, f32::NEG_INFINITY));
        for (c, v) in best {
            cols.push(c);
            logits.push(v);
        }
    }
    (cols, logits)
}

/// `LEGION_DEBUG_TP_DRAFT_HEAD_CHECK`: rows whose merged candidates and logits differ from the
/// whole head's, of `rows` this pass; logged every 256 passes and at each pass with a
/// difference.
pub(crate) fn note_check(rows: usize, differ: usize) {
    static PASSES: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);
    static DIFFER: AtomicU64 = AtomicU64::new(0);
    let passes = PASSES.fetch_add(1, Ordering::Relaxed) + 1;
    let total = ROWS.fetch_add(rows as u64, Ordering::Relaxed) + rows as u64;
    let bad = DIFFER.fetch_add(differ as u64, Ordering::Relaxed) + differ as u64;
    if differ > 0 || passes.is_multiple_of(256) {
        tracing::info!(
            passes,
            rows = total,
            rows_differ = bad,
            this_pass = differ,
            "--tp-draft-split head check: merged top-k vs the whole head's"
        );
    }
}

/// Rows of `[row][k]` candidates and logits that differ between `a` and `b`.
pub(crate) fn rows_differ(a: (&[u32], &[f32]), b: (&[u32], &[f32]), k: usize) -> usize {
    a.0.chunks(k)
        .zip(a.1.chunks(k))
        .zip(b.0.chunks(k).zip(b.1.chunks(k)))
        .filter(|((ac, av), (bc, bv))| {
            ac != bc
                || av
                    .iter()
                    .zip(bv.iter())
                    .any(|(x, y)| x.to_bits() != y.to_bits())
        })
        .count()
}
