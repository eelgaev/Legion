//! Pure bookkeeping for paged KV storage; no token data is stored or copied here.

/// An index into the pool that allocated it. IDs must only be used with that pool.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(pub u32);

/// A requested reservation cannot fit in the pool or the table's token count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("not enough KV block capacity")]
pub struct OutOfBlocks;

/// Copy `src` into `dst` before writing new tokens into `dst`.
/// The caller must complete the copy before releasing or reusing either block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOp {
    pub src: BlockId,
    pub dst: BlockId,
}

/// Reference-counted blocks with LIFO free lists. Blocks below `fast` (all, by default) are
/// handed out before the others: a KV pool's blocks in device memory before those of its host
/// tier.
#[derive(Debug)]
pub struct BlockPool {
    block_size: u32,
    ref_counts: Vec<u32>,
    free: Vec<BlockId>,
    /// Free blocks from `fast` on, below `slow_limit` (the ones that can be handed out).
    free_slow: Vec<BlockId>,
    fast: u32,
    /// Blocks from here on are not handed out ([`Self::set_slow_limit`]): the pool's size by
    /// default. The free ones among them still count in [`Self::free_blocks`].
    slow_limit: u32,
    /// Blocks per slot of a slotted pool ([`Self::slotted`]); 0 otherwise.
    slot_blocks: u32,
    /// A slotted pool's free blocks (it keeps no free lists).
    slotted_free: u32,
    /// The most blocks a slotted pool hands out at once over all its slots
    /// ([`Self::set_slotted_cap`]); its size by default.
    slotted_cap: u32,
}

impl BlockPool {
    /// Create a pool. `block_size` must be nonzero; an empty pool is allowed.
    pub fn new(num_blocks: u32, block_size: u32) -> Self {
        Self::with_fast(num_blocks, block_size, num_blocks)
    }

    /// A pool whose blocks `0..fast` are allocated before the rest.
    pub fn with_fast(num_blocks: u32, block_size: u32, fast: u32) -> Self {
        assert!(block_size > 0, "block size must be nonzero");
        let fast = fast.min(num_blocks);
        Self {
            block_size,
            ref_counts: vec![0; num_blocks as usize],
            free: (0..fast).rev().map(BlockId).collect(),
            free_slow: (fast..num_blocks).rev().map(BlockId).collect(),
            fast,
            slow_limit: num_blocks,
            slot_blocks: 0,
            slotted_free: 0,
            slotted_cap: 0,
        }
    }

    /// A pool of `slots` slots of `slot_blocks` blocks each, where the `i`-th block of a table
    /// in slot `s` is always block `s × slot_blocks + i` ([`BlockTable::append_in`]): each
    /// sequence's blocks are one contiguous range, which attention can read in place.
    /// [`Self::allocate`] hands out nothing from it.
    pub fn slotted(slots: u32, slot_blocks: u32, block_size: u32) -> Self {
        assert!(slot_blocks > 0, "a slot needs blocks");
        let num_blocks = slots * slot_blocks;
        let mut pool = Self::with_fast(0, block_size, 0);
        pool.ref_counts = vec![0; num_blocks as usize];
        pool.slow_limit = num_blocks;
        pool.slot_blocks = slot_blocks;
        pool.slotted_free = num_blocks;
        pool.slotted_cap = num_blocks;
        pool
    }

    /// Caps a slotted pool's blocks in use over all its slots at `cap` (its size at most): slots
    /// whose memory is backed as they fill, by memory that holds fewer blocks than all of them
    /// (#1131). [`Self::free_blocks`] counts what the cap leaves.
    pub fn set_slotted_cap(&mut self, cap: u32) {
        assert!(self.slot_blocks > 0, "a cap on a slotted pool");
        self.slotted_cap = cap.min(self.ref_counts.len() as u32);
    }

    /// Blocks per slot of a slotted pool, 0 for an unslotted one.
    pub fn slot_blocks(&self) -> u32 {
        self.slot_blocks
    }

    /// Free blocks among the fast ones.
    pub fn free_fast_blocks(&self) -> u32 {
        self.free.len() as u32
    }

    /// Blocks from `fast` on that are allocated (0 for a slotted pool).
    pub fn slow_in_use(&self) -> u32 {
        if self.slot_blocks > 0 {
            return 0;
        }
        let slow = (self.ref_counts.len() as u32).saturating_sub(self.fast);
        // The free ones past the limit are not in `free_slow` (#37).
        slow - self.free_slow.len() as u32 - self.free_past_limit()
    }

    /// The allocated blocks from `fast` on, ascending (none for a slotted pool).
    pub fn slow_used(&self) -> Vec<u32> {
        if self.slot_blocks > 0 {
            return Vec::new();
        }
        (self.fast..self.ref_counts.len() as u32)
            .filter(|&b| self.ref_counts[b as usize] > 0)
            .collect()
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn free_blocks(&self) -> u32 {
        if self.slot_blocks > 0 {
            let in_use = self.ref_counts.len() as u32 - self.slotted_free;
            return self
                .slotted_free
                .min(self.slotted_cap.saturating_sub(in_use));
        }
        (self.free.len() + self.free_slow.len()) as u32 + self.free_past_limit()
    }

    /// Free blocks that can be handed out now: [`Self::free_blocks`] but those past the slow
    /// limit ([`Self::set_slow_limit`]).
    pub fn allocatable(&self) -> u32 {
        if self.slot_blocks > 0 {
            return self.free_blocks();
        }
        (self.free.len() + self.free_slow.len()) as u32
    }

    /// The free blocks from the slow limit on.
    fn free_past_limit(&self) -> u32 {
        let from = self.slow_limit.max(self.fast) as usize;
        (self.ref_counts.get(from..).unwrap_or_default().iter())
            .filter(|&&c| c == 0)
            .count() as u32
    }

    /// The slow limit: blocks from `limit` on (at least `fast`, at most the pool's size) are
    /// not handed out, though the free ones still count as free (#37: a KV pool's tail mapped
    /// only as far as its sequences need, `limit` the first block without memory). Raised, the
    /// free blocks it adds are handed out after every free block below the old limit, lowest
    /// first, as a pool that never had a limit would; lowered, the free ones past it are taken
    /// off the free list (blocks in use stay in use).
    pub fn set_slow_limit(&mut self, limit: u32) {
        if self.slot_blocks > 0 {
            return;
        }
        let n = self.ref_counts.len() as u32;
        let limit = limit.clamp(self.fast, n);
        let old = self.slow_limit.clamp(self.fast, n);
        if limit > old {
            let added: Vec<BlockId> = (old..limit)
                .rev()
                .filter(|&b| self.ref_counts[b as usize] == 0)
                .map(BlockId)
                .collect();
            self.free_slow.splice(0..0, added);
        } else if limit < old {
            self.free_slow.retain(|b| b.0 < limit);
        }
        self.slow_limit = limit;
    }

    /// The slow limit ([`Self::set_slow_limit`]).
    pub fn slow_limit(&self) -> u32 {
        self.slow_limit
    }

    /// The lowest slow limit at which [`Self::allocatable`] is at least `need` (the free
    /// blocks past the current limit counted lowest first); the current limit when it already
    /// is, the pool's size when even all of them are too few (or the pool is slotted).
    pub fn limit_for(&self, need: u32) -> u32 {
        let n = self.ref_counts.len() as u32;
        if self.slot_blocks > 0 {
            return n;
        }
        let have = self.allocatable();
        if have >= need {
            return self.slow_limit;
        }
        let mut more = need - have;
        for b in self.slow_limit.max(self.fast)..n {
            if self.ref_counts[b as usize] == 0 {
                more -= 1;
                if more == 0 {
                    return b + 1;
                }
            }
        }
        n
    }

    /// Allocate one block with a reference count of one.
    pub fn allocate(&mut self) -> Option<BlockId> {
        let block = self.free.pop().or_else(|| self.free_slow.pop())?;
        debug_assert_eq!(self.ref_count(block), 0);
        self.ref_counts[block.0 as usize] = 1;
        Some(block)
    }

    /// Allocate block `b` of a slotted pool with a reference count of one, if it is free.
    fn allocate_at(&mut self, b: BlockId) -> Option<BlockId> {
        let count = self.ref_counts.get_mut(b.0 as usize)?;
        if *count > 0 {
            return None;
        }
        *count = 1;
        self.slotted_free -= 1;
        Some(b)
    }

    /// Add an owner to an allocated block. Free blocks cannot be resurrected.
    pub fn incref(&mut self, b: BlockId) {
        let count = &mut self.ref_counts[b.0 as usize];
        assert!(*count > 0, "cannot incref a free block");
        *count = count
            .checked_add(1)
            .expect("block reference count overflow");
    }

    /// Remove an owner, returning true if the block became free.
    pub fn decref(&mut self, b: BlockId) -> bool {
        let count = &mut self.ref_counts[b.0 as usize];
        debug_assert!(*count > 0, "cannot decref a free block");
        *count = count.checked_sub(1).expect("cannot decref a free block");
        if *count == 0 {
            if self.slot_blocks > 0 {
                self.slotted_free += 1;
            } else if b.0 < self.fast {
                self.free.push(b);
            } else if b.0 < self.slow_limit {
                self.free_slow.push(b);
            }
            true
        } else {
            false
        }
    }

    pub fn ref_count(&self, b: BlockId) -> u32 {
        self.ref_counts[b.0 as usize]
    }
}

/// A sequence's ordered blocks, all belonging to the same pool.
///
/// Use `fork` to share ownership and `release` to return it to the pool. Dropping
/// a table alone cannot release its blocks because it does not borrow the pool.
#[derive(Debug, Default)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
    num_tokens: u32,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// A table starting with `blocks` (e.g. a prefix-cache match), all full, taking one
    /// reference to each.
    pub fn from_shared(pool: &mut BlockPool, blocks: &[BlockId]) -> Self {
        for &block in blocks {
            pool.incref(block);
        }
        Self {
            blocks: blocks.to_vec(),
            num_tokens: blocks.len() as u32 * pool.block_size(),
        }
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }

    /// Reserve room for `n` more tokens. On error neither table nor pool changes.
    /// This only reserves slots: call `make_last_writable` before appending when
    /// new tokens will write into an existing shared, partially filled block.
    pub fn append(&mut self, pool: &mut BlockPool, n: u32) -> Result<(), OutOfBlocks> {
        let num_tokens = self.num_tokens.checked_add(n).ok_or(OutOfBlocks)?;
        let needed = num_tokens.div_ceil(pool.block_size()) as usize - self.blocks.len();
        if needed > pool.allocatable() as usize {
            return Err(OutOfBlocks);
        }
        for _ in 0..needed {
            self.blocks
                .push(pool.allocate().expect("capacity checked before allocation"));
        }
        self.num_tokens = num_tokens;
        Ok(())
    }

    /// [`Self::append`] in slot `slot` of a slotted pool ([`BlockPool::slotted`]): the table's
    /// `i`-th block is the slot's `i`-th. On error neither table nor pool changes.
    pub fn append_in(
        &mut self,
        pool: &mut BlockPool,
        n: u32,
        slot: u32,
    ) -> Result<(), OutOfBlocks> {
        let num_tokens = self.num_tokens.checked_add(n).ok_or(OutOfBlocks)?;
        let (have, total) = (
            self.blocks.len() as u32,
            num_tokens.div_ceil(pool.block_size()),
        );
        let base = slot * pool.slot_blocks();
        if total > pool.slot_blocks()
            || total.saturating_sub(have) > pool.free_blocks()
            || (have..total).any(|i| pool.ref_count(BlockId(base + i)) != 0)
        {
            return Err(OutOfBlocks);
        }
        debug_assert!(
            self.blocks
                .iter()
                .enumerate()
                .all(|(i, b)| b.0 == base + i as u32)
        );
        for i in have..total {
            let b = pool.allocate_at(BlockId(base + i));
            self.blocks.push(b.expect("checked free before allocation"));
        }
        self.num_tokens = num_tokens;
        Ok(())
    }

    /// Share all blocks with a new table, adding one reference to each.
    pub fn fork(&self, pool: &mut BlockPool) -> BlockTable {
        let blocks = self.blocks.clone();
        for &block in &blocks {
            pool.incref(block);
        }
        Self {
            blocks,
            num_tokens: self.num_tokens,
        }
    }

    /// Privatize a shared, partially filled last block, returning the copy to run.
    /// Empty tables, full tails and private tails need no copy. On error neither
    /// table nor pool changes. This does not execute the returned copy itself.
    pub fn make_last_writable(
        &mut self,
        pool: &mut BlockPool,
    ) -> Result<Option<CopyOp>, OutOfBlocks> {
        if self.num_tokens.is_multiple_of(pool.block_size()) {
            return Ok(None);
        }
        let last = self.blocks.last_mut().expect("partial tail has a block");
        let src = *last;
        if pool.ref_count(src) == 1 {
            return Ok(None);
        }
        let dst = pool.allocate().ok_or(OutOfBlocks)?;
        pool.decref(src);
        *last = dst;
        Ok(Some(CopyOp { src, dst }))
    }

    /// Shorten the sequence, releasing whole blocks past the new end.
    /// A length greater than the current length leaves the table unchanged.
    pub fn truncate(&mut self, pool: &mut BlockPool, num_tokens: u32) {
        if num_tokens >= self.num_tokens {
            return;
        }
        let keep = num_tokens.div_ceil(pool.block_size()) as usize;
        for block in self.blocks.drain(keep..) {
            pool.decref(block);
        }
        self.num_tokens = num_tokens;
    }

    /// Consume this table and return all of its references to the pool.
    pub fn release(self, pool: &mut BlockPool) {
        for block in self.blocks {
            pool.decref(block);
        }
    }
}

#[cfg(test)]
mod tests;
