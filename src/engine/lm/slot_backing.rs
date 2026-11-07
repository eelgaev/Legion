//! Lazily backed sequence slots (#1131): when the host cannot hold every slot's host tier
//! (`--max-seqs` sequences of the whole `--ctx-size` each), each slot's host blocks are backed
//! as its sequence grows ([`crate::model::core::Executor::back_kv_slot`]), in whole chunks, up
//! to what the host holds; idle slots give theirs back when another slot needs the room. The
//! pool's cap ([`crate::kv::BlockPool::set_slotted_cap`], [`super::lazy_slot_cap`]) keeps the
//! running and warm sequences' blocks within it, so the idle slots always cover a shortfall.
//!
//! Where the slots do not all fit in VRAM, they share it (#1159): `lend` VRAM blocks, in whole
//! chunks, go to the slots' first blocks as they grow; a slot short of them takes the warm
//! slots' (the least recently finished first), whose blocks move to the host tier. Only where
//! the blocks live changes.

use super::LAZY_CHUNK_BLOCKS;

/// The blocks backed per slot, against the host's room and the shared VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotBacking {
    /// Host blocks the slots may hold at once.
    budget: usize,
    /// Each slot's first blocks, in VRAM: always backed, never counted.
    hbm: usize,
    /// Blocks per slot.
    slot_blocks: usize,
    /// Per slot: its first blocks backed (at least `hbm`).
    backed: Vec<usize>,
    /// Host blocks backed over all slots.
    used: usize,
    /// VRAM blocks the slots share past their own `hbm`.
    lend: usize,
    /// Per slot: its shared VRAM blocks (its first backed blocks past `hbm`).
    lent: Vec<usize>,
}

/// What a slot's growth takes: idle slots to give back first, warm slots' shared VRAM moved to
/// the host (slot, shared VRAM blocks it keeps), then the slot backed to `to` blocks, `lent` of
/// them shared VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Backing {
    pub trim: Vec<u32>,
    pub demote: Vec<(u32, usize)>,
    pub to: usize,
    pub lent: usize,
}

impl SlotBacking {
    pub fn new(slots: usize, slot_blocks: usize, hbm: usize, budget: usize, lend: usize) -> Self {
        let hbm = hbm.min(slot_blocks);
        Self {
            budget,
            hbm,
            slot_blocks,
            backed: vec![hbm; slots],
            used: 0,
            lend: lend / LAZY_CHUNK_BLOCKS * LAZY_CHUNK_BLOCKS,
            lent: vec![0; slots],
        }
    }

    /// What slot `slot` holding `blocks` blocks needs backed, the idle slots (`idle`, most
    /// recently freed last) given back first where the room is short, the warm slots (`warm`,
    /// least recently finished first) giving their shared VRAM where it is short; `None` when
    /// it is backed and has all the shared VRAM it can.
    pub fn plan(
        &self,
        slot: u32,
        blocks: usize,
        idle: &[u32],
        warm: &[u32],
    ) -> Result<Option<Backing>, String> {
        let s = slot as usize;
        let have = self.backed[s];
        let to = blocks
            .next_multiple_of(LAZY_CHUNK_BLOCKS)
            .min(self.slot_blocks)
            .max(have);
        let mine = self.lent[s];
        let want = (to - self.hbm).min(self.lend).max(mine);
        let mut free = self.lend - self.lent.iter().sum::<usize>();
        let mut demote = Vec::new();
        for &w in warm {
            if mine + free >= want {
                break;
            }
            let held = self.lent[w as usize];
            if w != slot && held > 0 {
                let take = held.min(want - mine - free);
                demote.push((w, held - take));
                free += take;
            }
        }
        let lent = want.min(mine + free);
        if to == have && lent == mine {
            return Ok(None);
        }
        // A warm slot's blocks moved out take as much host as the slot's moved in give back.
        let moved: usize = demote
            .iter()
            .map(|&(w, keep)| self.lent[w as usize] - keep)
            .sum();
        let mut used = (self.used + (to - have) + moved).saturating_sub(lent - mine);
        let mut trim = Vec::new();
        // The least recently freed first: the others may soon be taken again.
        for &i in idle {
            if used <= self.budget {
                break;
            }
            let held = self.backed[i as usize] - self.hbm - self.lent[i as usize];
            if i != slot && held > 0 {
                used -= held;
                trim.push(i);
            }
        }
        if used > self.budget {
            return Err(format!(
                "the KV host tier backs {} blocks at most, short of slot {slot}'s {to}",
                self.budget
            ));
        }
        Ok(Some(Backing {
            trim,
            demote,
            to,
            lent,
        }))
    }

    /// Slot `slot` was given back: only its VRAM blocks stay backed.
    pub fn trimmed(&mut self, slot: u32) {
        let s = slot as usize;
        self.used -= self.backed[s] - self.hbm - self.lent[s];
        self.backed[s] = self.hbm;
        self.lent[s] = 0;
    }

    /// Slot `slot` is backed to `to` blocks, `lent` of them shared VRAM.
    pub fn backed(&mut self, slot: u32, to: usize, lent: usize) {
        let s = slot as usize;
        let to = to.max(self.backed[s]);
        let lent = lent.min(to - self.hbm);
        let host = self.backed[s] - self.hbm - self.lent[s];
        self.used = self.used - host + (to - self.hbm - lent);
        self.backed[s] = to;
        self.lent[s] = lent;
    }

    /// Slot `slot`'s blocks: backed, and shared VRAM among them.
    pub fn held(&self, slot: u32) -> (usize, usize) {
        (self.backed[slot as usize], self.lent[slot as usize])
    }

    /// Whether the slots share VRAM.
    pub fn lends(&self) -> bool {
        self.lend > 0
    }

    /// Host blocks backed over all slots.
    #[cfg(test)]
    pub fn used(&self) -> usize {
        self.used
    }
}

#[cfg(test)]
mod tests;
