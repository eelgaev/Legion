//! Lazily backed sequence slots (#1131): when the host cannot hold every slot's host tier
//! (`--max-seqs` sequences of the whole `--ctx-size` each), each slot's host blocks are backed
//! as its sequence grows ([`crate::model::core::Executor::back_kv_slot`]), in whole chunks, up
//! to what the host holds; idle slots give theirs back when another slot needs the room. The
//! pool's cap ([`crate::kv::BlockPool::set_slotted_cap`], [`super::lazy_slot_cap`]) keeps the
//! running and warm sequences' blocks within it, so the idle slots always cover a shortfall.

use super::LAZY_CHUNK_BLOCKS;

/// The blocks backed per slot, against the host's room.
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
}

/// What a slot's growth takes: idle slots to give back first, then the slot backed to `to`
/// blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Backing {
    pub trim: Vec<u32>,
    pub to: usize,
}

impl SlotBacking {
    pub fn new(slots: usize, slot_blocks: usize, hbm: usize, budget: usize) -> Self {
        let hbm = hbm.min(slot_blocks);
        Self {
            budget,
            hbm,
            slot_blocks,
            backed: vec![hbm; slots],
            used: 0,
        }
    }

    /// What slot `slot` holding `blocks` blocks needs backed, the idle slots (`idle`, most
    /// recently freed last) given back first where the room is short; `None` when it is backed.
    pub fn plan(&self, slot: u32, blocks: usize, idle: &[u32]) -> Result<Option<Backing>, String> {
        let have = self.backed[slot as usize];
        let to = blocks
            .next_multiple_of(LAZY_CHUNK_BLOCKS)
            .min(self.slot_blocks)
            .max(have);
        if to == have {
            return Ok(None);
        }
        let mut used = self.used + (to - have);
        let mut trim = Vec::new();
        // The least recently freed first: the others may soon be taken again.
        for &s in idle {
            if used <= self.budget {
                break;
            }
            let held = self.backed[s as usize] - self.hbm;
            if s != slot && held > 0 {
                used -= held;
                trim.push(s);
            }
        }
        if used > self.budget {
            return Err(format!(
                "the KV host tier backs {} blocks at most, short of slot {slot}'s {to}",
                self.budget
            ));
        }
        Ok(Some(Backing { trim, to }))
    }

    /// Slot `slot` was given back: only its VRAM blocks stay backed.
    pub fn trimmed(&mut self, slot: u32) {
        let b = &mut self.backed[slot as usize];
        self.used -= *b - self.hbm;
        *b = self.hbm;
    }

    /// Slot `slot` is backed to `to` blocks.
    pub fn backed(&mut self, slot: u32, to: usize) {
        let b = &mut self.backed[slot as usize];
        if to > *b {
            self.used += to - *b;
            *b = to;
        }
    }

    /// Host blocks backed over all slots.
    #[cfg(test)]
    pub fn used(&self) -> usize {
        self.used
    }
}

#[cfg(test)]
mod tests;
