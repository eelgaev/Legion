//! Greedy picks below `min_tokens` without reading whole logits rows back (#1176).
//!
//! A greedy sequence below `min_tokens` must not pick a stop token, so its pick is the argmax of
//! its row with the stop tokens masked, first of ties ([`crate::engine::sampling::argmax`]).
//! The step's device argmax already is that pick whenever it is not a masked stop token and
//! nothing ties with it before it: it is the row's largest value, and the host takes the first
//! column holding that value. So such a row ([`Logits::Pick`], [`Logits::FirstPick`]) comes back
//!
//! - not at all when the model's device argmax is the first of ties anyway
//!   ([`crate::model::core::Arch::argmax_first_of_ties`]),
//! - up to its device argmax otherwise (the columns a tie could hide in; common tokens have low
//!   ids, so usually a few KB instead of the ~1 MB row of a 248K vocabulary),
//! - whole only when the device argmax is a masked stop token.
//!
//! Reading every row in full (a 1 MB copy and a host scan per row) is what llama-benchy's
//! `--exact-tg` runs paid on every step: about 1 MB per user per step that greedy decoding never
//! needs.

use std::ops::Range;

use super::Logits;

/// What [`Need`] and [`pick`] need besides a row: the stop tokens a masked row may not pick and
/// whether the device argmax is the first of ties.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Picks<'s> {
    pub stop: &'s [u32],
    pub exact: bool,
}

/// How much of a logits row a step reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Need {
    /// Nothing: an argmax-only row, or a pick the device argmax already is.
    Nothing,
    /// The whole row, for the engine to sample from.
    Row,
    /// The first `len` values, for [`pick`] (`masked`: the stop tokens may not be picked).
    Pick { len: usize, masked: bool },
}

impl Need {
    /// What row `i` of a sequence's rows of kind `logits` needs, with device argmax `t`.
    pub(crate) fn of(logits: Logits, i: usize, t: u32, n_vocab: usize, picks: Picks<'_>) -> Self {
        let masked = match logits {
            Logits::Pick => true,
            Logits::FirstPick(_, m) => i < m,
            Logits::Greedy | Logits::FirstGreedy(_) => return Need::Nothing,
            _ => return Need::Row,
        };
        let stopped = masked && picks.stop.contains(&t);
        if stopped || t as usize >= n_vocab {
            Need::Pick {
                len: n_vocab,
                masked,
            }
        } else if picks.exact {
            Need::Nothing
        } else {
            Need::Pick {
                len: t as usize + 1,
                masked,
            }
        }
    }

    /// The values read back for it.
    pub(crate) fn len(self, n_vocab: usize) -> usize {
        match self {
            Need::Nothing => 0,
            Need::Row => n_vocab,
            Need::Pick { len, .. } => len,
        }
    }
}

/// The host's greedy pick ([`crate::engine::sampling::argmax`] of the row with the stop tokens
/// set to -inf when `masked`) from `row`, the row's first values ([`Need::Pick`]), for device
/// argmax `t`. `None` when `row` is too short to tell: then the whole row decides.
///
/// A whole row is masked and scanned as the engine would. A shorter one ends at `t` (not a
/// masked stop token): the pick is the first column up to `t` holding `row[t]`, the row's
/// largest value. The few cases where that may not hold (`row[t]` NaN or -inf, a NaN first value,
/// a larger value before `t`) need the whole row.
pub(crate) fn pick(row: &[f32], t: u32, masked: bool, n_vocab: usize, stop: &[u32]) -> Option<u32> {
    let blocked = |c: usize| masked && stop.contains(&(c as u32));
    let t = t as usize;
    if row.len() < n_vocab {
        let v = *row.get(t)?;
        if row.len() != t + 1
            || blocked(t)
            || v.is_nan()
            || v == f32::NEG_INFINITY
            || row[0].is_nan()
        {
            return None;
        }
        if row.iter().any(|&x| x > v) {
            return None;
        }
        return (0..=t)
            .find(|&c| row[c] == v && !blocked(c))
            .map(|c| c as u32);
    }
    let mut row = row.to_vec();
    if masked {
        for &s in stop {
            if let Some(l) = row.get_mut(s as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    Some(crate::engine::sampling::argmax(&row) as u32)
}

/// Puts the part of a row read back for `need` (`values`, [`Need::len`] long) where a step's
/// output wants it: the row itself into `row`, or the row's pick into `t` (its device argmax
/// until then). `false` when the pick needs the whole row ([`pick`]).
pub(crate) fn took(
    need: Need,
    values: &[f32],
    t: &mut u32,
    row: &mut Vec<f32>,
    n_vocab: usize,
    picks: Picks<'_>,
) -> bool {
    match need {
        Need::Nothing => true,
        Need::Row => {
            *row = values.to_vec();
            true
        }
        Need::Pick { masked, .. } => match pick(values, *t, masked, n_vocab, picks.stop) {
            Some(p) => {
                *t = p;
                true
            }
            None => false,
        },
    }
}

/// Most F32s [`spans`] reads past between two rows' parts to bring them back in one DMA (a read's
/// fixed cost on a DGX Spark is about what copying 256 KB costs).
pub(crate) const SPAN_GAP: usize = 64 << 10;

/// The reads that bring back `parts` (each a start and a length in F32s, in order, not
/// overlapping): runs of consecutive parts whose gaps are at most `gap` F32s and whose span is
/// at most `cap` F32s, as index ranges into `parts`. A part longer than `cap` is a read alone.
/// Whole rows next to each other come back in one read; a pick's few KB at the start of its row
/// do not drag in the ~1 MB rest of the row before the next one.
pub(crate) fn spans(parts: &[(usize, usize)], gap: usize, cap: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        let (start, len) = parts[i];
        let mut end = start + len;
        let mut j = i + 1;
        while let Some(&(s, l)) = parts.get(j) {
            if s.saturating_sub(end) > gap || s + l - start > cap {
                break;
            }
            end = s + l;
            j += 1;
        }
        out.push(i..j);
        i = j;
    }
    out
}

#[cfg(test)]
mod tests;
