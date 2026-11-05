//! Saved copies of sliding-window rings (context checkpoints), the half of prefix caching that
//! models with rings need besides a warm sequence's KV blocks.
//!
//! A ring ([`crate::model::core::KvRing`]) keeps only the last positions of its sequence, so a
//! warm sequence ([`super::lm`]'s `Warm`) can resume a new prompt only near where its ring was
//! last written, or where a copy of the ring was saved. One copy at the end of the last prompt
//! missed every turn whose prompt left the cached one more than a few hundred tokens before
//! that end (a chat client that rewrites its last turns). Like llama.cpp's context
//! checkpoints, a sequence now keeps several: at the end of each prompt and every
//! `--ring-checkpoint-every` tokens over the last part of a long one, kept across the turns
//! of a chat, in a fixed number of pinned host slots shared by all rings.
//!
//! A checkpoint belongs to a ring: the ring of the sequence that saved it, then of the warm
//! sequence it became, then of the next prompt that resumed from it. A resume at `m` drops the
//! ring's checkpoints past `m` (their tokens may differ); a ring given back drops them all.
//!
//! When every slot is taken, a new checkpoint takes the least recently used one that is not
//! its ring's latest (each warm sequence keeps the copy at its last prompt's end, as before).
//!
//! `--ring-checkpoints 0` turns them off ([`RingCheckpoints::single`]): one copy per ring, in
//! its own slot, at the end of its last prompt only, as before checkpoints.

/// Which checkpoint slot holds which ring's copy.
#[derive(Debug)]
pub struct RingCheckpoints {
    slots: Vec<Option<Checkpoint>>,
    tick: u64,
    /// [`Self::single`]: ring `r`'s copy always in slot `r`.
    single: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Checkpoint {
    ring: usize,
    /// The ring as it was after this many tokens.
    at: usize,
    last_used: u64,
}

impl RingCheckpoints {
    pub fn new(slots: usize) -> Self {
        Self {
            slots: vec![None; slots],
            tick: 0,
            single: false,
        }
    }

    /// Checkpoints off (`--ring-checkpoints 0`): one slot per ring (`rings` of them), ring
    /// `r`'s in slot `r`, each new copy replacing the last: the single save at a prompt's end
    /// that came before checkpoints.
    pub fn single(rings: usize) -> Self {
        Self {
            single: true,
            ..Self::new(rings)
        }
    }

    /// Whether checkpoints are off ([`Self::single`]).
    pub fn is_single(&self) -> bool {
        self.single
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Ring `ring`'s checkpoints: `(at, slot)`, ascending by `at`.
    pub fn of(&self, ring: usize) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = (self.slots.iter().enumerate())
            .filter_map(|(i, c)| c.filter(|c| c.ring == ring).map(|c| (c.at, i)))
            .collect();
        v.sort_unstable();
        v
    }

    /// Ring `ring`'s latest checkpoint's `at`.
    pub fn latest(&self, ring: usize) -> Option<usize> {
        self.of(ring).last().map(|&(at, _)| at)
    }

    /// A slot for ring `ring`'s copy after `at` tokens, which the caller then saves: the one
    /// already holding that point, a free one, or one taken from another checkpoint ([`Self::single`]:
    /// the ring's own). None without slots.
    pub fn store(&mut self, ring: usize, at: usize) -> Option<usize> {
        if self.slots.is_empty() {
            return None;
        }
        self.tick += 1;
        let tick = self.tick;
        let slot = if self.single {
            (ring < self.slots.len()).then_some(ring)?
        } else {
            (self.slots.iter())
                .position(|c| c.is_some_and(|c| c.ring == ring && c.at == at))
                .or_else(|| self.slots.iter().position(Option::is_none))
                .or_else(|| self.victim(ring))?
        };
        self.slots[slot] = Some(Checkpoint {
            ring,
            at,
            last_used: tick,
        });
        Some(slot)
    }

    /// The slot a new checkpoint of `ring` takes when all are in use: the least recently used
    /// one that is not its ring's latest; else `ring`'s own oldest (its latest moves on); else
    /// the least recently used.
    fn victim(&self, ring: usize) -> Option<usize> {
        let latest = |c: &Checkpoint| self.latest(c.ring) == Some(c.at);
        let lru = |keep_latest: bool| {
            (self.slots.iter().enumerate())
                .filter_map(|(i, c)| c.map(|c| (i, c)))
                .filter(|(_, c)| !(keep_latest && latest(c)))
                .min_by_key(|(_, c)| c.last_used)
                .map(|(i, _)| i)
        };
        lru(true)
            .or_else(|| self.of(ring).first().map(|&(_, slot)| slot))
            .or_else(|| lru(false))
    }

    /// Marks slot `slot` used now (a prompt resumed from it).
    pub fn touch(&mut self, slot: usize) {
        self.tick += 1;
        if let Some(c) = &mut self.slots[slot] {
            c.last_used = self.tick;
        }
    }

    /// Drops ring `ring`'s checkpoints past `n` tokens.
    pub fn truncate(&mut self, ring: usize, n: usize) {
        for c in &mut self.slots {
            if c.is_some_and(|c| c.ring == ring && c.at > n) {
                *c = None;
            }
        }
    }

    /// Drops ring `ring`'s checkpoints (the ring is free again).
    pub fn clear(&mut self, ring: usize) {
        for c in &mut self.slots {
            if c.is_some_and(|c| c.ring == ring) {
                *c = None;
            }
        }
    }

    /// Of ring `ring`'s checkpoints, the one to resume a prompt from that shares `common`
    /// tokens with the ring's sequence: the most tokens resumed, `m = min(common, at)`, among
    /// those where `holds(m, at)` (the copy still holds the window before `m`). `(m, at, slot)`.
    pub fn best(
        &self,
        ring: usize,
        common: usize,
        holds: impl Fn(usize, usize) -> bool,
    ) -> Option<(usize, usize, usize)> {
        (self.of(ring).into_iter())
            .map(|(at, slot)| (common.min(at), at, slot))
            .filter(|&(m, at, _)| m > 0 && holds(m, at))
            .max_by_key(|&(m, at, _)| (m, std::cmp::Reverse(at)))
    }
}

/// Whether a prompt of `prompt_len` tokens, whose step just ended after `n_past` of them (not
/// its end, which always saves), saves a checkpoint there: over its last `span` tokens, at
/// least `every` tokens past its ring's latest checkpoint (`latest`). The steps end where they
/// would without checkpoints (a step's sums depend on its length), so the points fall at step
/// ends, `every` apart or a little more.
pub fn due(
    n_past: usize,
    prompt_len: usize,
    latest: Option<usize>,
    every: usize,
    span: usize,
) -> bool {
    every > 0
        && n_past < prompt_len
        && n_past + span >= prompt_len
        && latest.is_none_or(|at| n_past >= at + every)
}

#[cfg(test)]
mod tests;
