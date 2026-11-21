//! Deferred DeltaNet commits in the wait of a MoE sum over machines (#1176).
//!
//! With `--tp-commit-in-verify-min-seqs`, a wide verify step's DeltaNet layers first run the
//! commit of the last step's accepted tokens (`gdn::rows_replay`), in the delta rule's own
//! kernel. Under tensor parallelism over two DGX Sparks (GB10) with `--tp-moe-ep`, rank 0
//! finishes its experts first and waits ~0.18 ms (median) at every layer's MoE sum for rank 1's
//! partial, while rank 1 waits for rank 0 at the attention sums. So on rank 0 (by default) each
//! DeltaNet layer's commits run in the wait of the MoE sum before it instead: that sum's node
//! launches them ([`crate::tp_net::all_reduce_beside`]) once its partial is on its way and
//! before it waits for the peer's, and the layer's delta rule then replays nothing
//! (`SpecPool::zero_counts`). The commits are the same kernel on the same values as the flush
//! of a deferred commit, whose states are the fused replay's bit for bit, so every output keeps
//! its bits.
//!
//! `LEGION_DEBUG_TP_COMMIT_WAIT=<mask>` picks the ranks that do it: bit `r` for rank `r`
//! (`1`, rank 0, by default; `0` none, the fused replay on every rank as before; `3` both).

use std::sync::OnceLock;

#[cfg(test)]
mod tests;

/// The ranks that run the commits in the wait by default: rank 0.
pub const DEFAULT_RANKS: u32 = 1;

/// The ranks' mask: `LEGION_DEBUG_TP_COMMIT_WAIT`, else [`DEFAULT_RANKS`]; read once.
fn ranks() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| ranks_from(std::env::var("LEGION_DEBUG_TP_COMMIT_WAIT").ok().as_deref()))
}

/// [`ranks`] for the switch's value `v` (a mask, decimal or `0x` hex; anything else the
/// default).
pub fn ranks_from(v: Option<&str>) -> u32 {
    let Some(s) = v.map(str::trim) else {
        return DEFAULT_RANKS;
    };
    let parsed = match s.strip_prefix("0x") {
        Some(h) => u32::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    };
    parsed.unwrap_or(DEFAULT_RANKS)
}

/// Whether rank `rank` runs its commits in the MoE sums' waits ([`ranks`]).
pub fn takes(rank: usize) -> bool {
    takes_in(ranks(), rank)
}

/// [`takes`] for the mask `mask`.
pub fn takes_in(mask: u32, rank: usize) -> bool {
    rank < 32 && mask & (1 << rank) != 0
}

/// The (first slot, sequences) of a step's verify groups, from its groups' (verify, first
/// slot, sequences), if at most `max` (a sum runs that many commits); empty otherwise.
pub fn verify_groups(
    groups: impl IntoIterator<Item = (bool, usize, usize)>,
    max: usize,
) -> Vec<(usize, usize)> {
    let v: Vec<_> = groups
        .into_iter()
        .filter(|&(verify, _, n)| verify && n > 0)
        .map(|(_, slot, n)| (slot, n))
        .collect();
    if v.len() > max { Vec::new() } else { v }
}
