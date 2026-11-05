//! Bookkeeping for saved recurrent states, the half of prefix caching that hybrid models need
//! besides the KV blocks of [`crate::kv::radix`].
//!
//! A DeltaNet layer's state after `n` tokens summarizes all of them, and it cannot be rebuilt
//! from cached K/V: a prompt can only resume after a cached prefix if the state at exactly that
//! prefix was saved. The engine saves it where the next prompts are likely to resume:
//! - where a new prompt branches off a cached one (so the next prompt sharing that prefix
//!   resumes there), and at the end of each prompt's last full block;
//! - every `--prefix-cache-interval` tokens of a prompt (opt-in, [`save_points`]): the second
//!   prompt on a new shared prefix (a system prompt) then resumes near where it branches, not
//!   at 0. These are *spaced* saves: at most half the slots hold spaced saves no prompt has
//!   resumed from yet, so a long prompt's points do not push out the other chats' states;
//! - when a sequence ends, after its reply ([`Saved::tail`]): the next turn of a chat resends
//!   the reply and resumes past it instead of computing it again. That state sits wherever the
//!   reply stopped, inside a KV block; the saved entry keeps that partly filled block (the
//!   radix tree holds full blocks only), and the prompt that resumes from it takes the block
//!   over, so such an entry serves one prompt.
//!
//! Slots are few (one state is tens of MB), so they are recycled least recently used first.

use crate::kv::BlockId;

/// Which saved-state slot holds the state after which token prefix.
#[derive(Debug)]
pub struct SavedStates {
    slots: Vec<Option<Saved>>,
    tick: u64,
}

#[derive(Debug)]
struct Saved {
    prefix: Vec<u32>,
    last_used: u64,
    /// A spaced save no prompt has resumed from yet.
    spaced: bool,
    /// With a prefix that ends inside a KV block: that block, holding one pool reference
    /// (its full blocks are in the radix tree).
    tail: Option<BlockId>,
}

/// A saved state [`SavedStates::find`] found.
#[derive(Debug, PartialEq, Eq)]
pub struct Found {
    pub slot: usize,
    /// Tokens it resumes (its prefix's length).
    pub len: usize,
    /// The partly filled KV block after its full blocks ([`Saved::tail`]): the prompt that
    /// resumes takes it over with [`SavedStates::take`].
    pub tail: Option<BlockId>,
}

/// What [`SavedStates::store`] saves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A branch point, a prompt's end or a sequence's end.
    Anchor,
    /// A `--prefix-cache-interval` point.
    Spaced,
}

/// Where [`SavedStates::store`] put a state.
#[derive(Debug, PartialEq, Eq)]
pub struct Stored {
    pub slot: usize,
    /// The tail block of the entry it replaced, whose pool reference the caller gives back.
    pub dropped: Option<BlockId>,
}

impl SavedStates {
    pub fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots).map(|_| None).collect(),
            tick: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The slot holding the longest saved prefix of `tokens` that leaves at least one of them
    /// to compute, whose full `block`-token blocks lie within the `matched` tokens the KV
    /// cache holds. Marks it used (a spaced save becomes an anchor).
    pub fn find(&mut self, tokens: &[u32], matched: usize, block: usize) -> Option<Found> {
        self.tick += 1;
        let (slot, len) = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let s = s.as_ref()?;
                let n = s.prefix.len();
                (n < tokens.len() && n / block * block <= matched && tokens.starts_with(&s.prefix))
                    .then_some((i, n))
            })
            .max_by_key(|&(_, len)| len)?;
        let s = self.slots[slot].as_mut().unwrap();
        s.last_used = self.tick;
        s.spaced = false;
        Some(Found {
            slot,
            len,
            tail: s.tail,
        })
    }

    /// Removes `slot`'s entry after a prompt resumed from it with its tail block: the prompt
    /// writes on into that block. Returns the block, whose entry's reference the caller gives
    /// back.
    pub fn take(&mut self, slot: usize) -> Option<BlockId> {
        self.slots[slot].take().and_then(|s| s.tail)
    }

    /// Drops every entry holding a tail block, returning the blocks (the caller gives their
    /// references back): the KV cache needs them.
    pub fn drop_tails(&mut self) -> Vec<BlockId> {
        let mut out = Vec::new();
        for s in &mut self.slots {
            if let Some(t) = s.as_ref().and_then(|s| s.tail) {
                out.push(t);
                *s = None;
            }
        }
        out
    }

    /// Drops every entry whose tail block is `first` or later, returning the blocks (the
    /// caller gives their references back): a KV pool's blocks past `first` that no longer
    /// fit in host memory (#1196).
    pub fn drop_tails_from(&mut self, first: BlockId) -> Vec<BlockId> {
        let mut out = Vec::new();
        for s in &mut self.slots {
            if let Some(t) = s.as_ref().and_then(|s| s.tail).filter(|t| t.0 >= first.0) {
                out.push(t);
                *s = None;
            }
        }
        out
    }

    /// The tokens `slot`'s entry holds the state after, if it holds one.
    pub fn prefix_len(&self, slot: usize) -> Option<usize> {
        self.slots.get(slot)?.as_ref().map(|s| s.prefix.len())
    }

    pub fn contains(&self, prefix: &[u32]) -> bool {
        self.slots
            .iter()
            .flatten()
            .any(|s| s.prefix.as_slice() == prefix)
    }

    /// A slot to save the state after `prefix` in, now recorded as holding it (with `tail`,
    /// [`Saved::tail`]): a free one, else the least recently used. A spaced save replaces the
    /// least recently used spaced save once those fill half the slots. `None` if that prefix
    /// is already saved or there are no slots (the caller keeps `tail`).
    pub fn store(&mut self, prefix: &[u32], kind: Kind, tail: Option<BlockId>) -> Option<Stored> {
        if self.slots.is_empty() || self.contains(prefix) {
            return None;
        }
        self.tick += 1;
        let lru = |only_spaced: bool| {
            (0..self.slots.len())
                .filter(|&i| !only_spaced || self.slots[i].as_ref().is_some_and(|s| s.spaced))
                .min_by_key(|&i| self.slots[i].as_ref().map_or(0, |s| s.last_used))
        };
        let spaced = self.slots.iter().flatten().filter(|s| s.spaced).count();
        let slot = match self.slots.iter().position(Option::is_none) {
            Some(i) => i,
            None if kind == Kind::Spaced && spaced >= (self.slots.len() / 2).max(1) => {
                lru(true).unwrap()
            }
            None => lru(false).unwrap(),
        };
        let dropped = self.slots[slot].take().and_then(|s| s.tail);
        self.slots[slot] = Some(Saved {
            prefix: prefix.to_vec(),
            last_used: self.tick,
            spaced: kind == Kind::Spaced,
            tail,
        });
        Some(Stored { slot, dropped })
    }
}

/// Positions at which to save the state of a `prompt_len`-token prompt that resumes after
/// `resumed` tokens, given that the KV cache matched its first `matched` tokens: the branch
/// point `matched` (when the state there was not saved) and the end of the last full block,
/// both anchors, and with `interval` > 0 (rounded up to whole blocks) every multiple of it,
/// spaced. Ascending, each a positive multiple of `block` past `resumed`.
pub fn save_points(
    prompt_len: usize,
    matched: usize,
    resumed: usize,
    block: usize,
    interval: usize,
) -> Vec<(usize, Kind)> {
    let mut points: Vec<(usize, Kind)> = Vec::new();
    let end = prompt_len / block * block;
    for p in [matched / block * block, end] {
        if p > resumed && p <= prompt_len && !points.iter().any(|&(q, _)| q == p) {
            points.push((p, Kind::Anchor));
        }
    }
    if interval > 0 {
        let step = interval.div_ceil(block) * block;
        let mut p = (resumed / step + 1) * step;
        while p < end {
            if !points.iter().any(|&(q, _)| q == p) {
                points.push((p, Kind::Spaced));
            }
            p += step;
        }
    }
    points.sort_unstable_by_key(|&(p, _)| p);
    points
}

#[cfg(test)]
mod tests;
