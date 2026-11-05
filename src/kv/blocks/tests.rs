use super::*;
use proptest::prelude::*;
use std::collections::HashSet;

#[test]
fn pool_allocates_until_exhausted_and_reuses_freed_blocks_lifo() {
    let mut pool = BlockPool::new(3, 16);
    assert_eq!(pool.block_size(), 16);
    assert_eq!(pool.free_blocks(), 3);
    for id in 0..3 {
        assert_eq!(pool.ref_count(BlockId(id)), 0);
    }
    let a = pool.allocate().unwrap();
    let b = pool.allocate().unwrap();
    let c = pool.allocate().unwrap();
    assert_eq!(HashSet::from([a, b, c]).len(), 3);
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(pool.allocate(), None);
    for block in [a, b, c] {
        assert_eq!(pool.ref_count(block), 1);
    }
    pool.incref(a);
    assert_eq!(pool.ref_count(a), 2);
    assert!(!pool.decref(a));
    assert_eq!(pool.free_blocks(), 0);
    assert!(pool.decref(a));
    assert!(pool.decref(b));
    assert_eq!(pool.ref_count(a), 0);
    assert_eq!(pool.free_blocks(), 2);
    assert_eq!(pool.allocate(), Some(b));
    assert_eq!(pool.allocate(), Some(a));
    assert_eq!(pool.ref_count(a), 1);
    assert_eq!(pool.ref_count(b), 1);
}

#[test]
#[should_panic(expected = "block size must be nonzero")]
fn zero_block_size_is_rejected() {
    BlockPool::new(1, 0);
}

#[test]
#[should_panic(expected = "cannot decref a free block")]
fn decref_free_block_panics() {
    let mut pool = BlockPool::new(1, 4);
    let block = pool.allocate().unwrap();
    pool.decref(block);
    pool.decref(block);
}

#[test]
#[should_panic(expected = "cannot incref a free block")]
fn incref_free_block_panics() {
    BlockPool::new(1, 4).incref(BlockId(0));
}

#[test]
fn empty_pool_and_empty_tables() {
    let mut pool = BlockPool::new(0, 4);
    let mut table = BlockTable::new();
    assert_eq!(pool.allocate(), None);
    assert!(table.blocks().is_empty());
    assert_eq!(table.num_tokens(), 0);
    assert_eq!(table.append(&mut pool, 0), Ok(()));
    assert_eq!(table.append(&mut pool, 1), Err(OutOfBlocks));
    assert_eq!(table.make_last_writable(&mut pool), Ok(None));
    let fork = table.fork(&mut pool);
    assert!(fork.blocks().is_empty());
    assert_eq!(fork.num_tokens(), 0);
    table.truncate(&mut pool, 0);
    table.truncate(&mut pool, 10);
    table.release(&mut pool);
    fork.release(&mut pool);
    assert_eq!(pool.free_blocks(), 0);
}

#[test]
fn append_uses_partial_tail_and_allocates_only_needed_blocks() {
    let mut pool = BlockPool::new(3, 4);
    let mut table = BlockTable::new();
    table.append(&mut pool, 1).unwrap();
    let first = table.blocks()[0];
    table.append(&mut pool, 3).unwrap();
    assert_eq!(table.blocks(), &[first]);
    assert_eq!(table.num_tokens(), 4);
    assert_eq!(pool.free_blocks(), 2);
    table.append(&mut pool, 5).unwrap();
    assert_eq!(table.blocks().len(), 3);
    assert_eq!(table.blocks()[0], first);
    assert_eq!(table.num_tokens(), 9);
    assert_eq!(pool.free_blocks(), 0);
    table.append(&mut pool, 3).unwrap();
    table.append(&mut pool, 0).unwrap();
    assert_eq!(table.num_tokens(), 12);
    table.release(&mut pool);
    assert_eq!(pool.free_blocks(), 3);
}

#[test]
fn failed_append_preserves_table_counts_and_free_list_order() {
    let mut pool = BlockPool::new(3, 4);
    let mut table = BlockTable::new();
    table.append(&mut pool, 3).unwrap();
    let blocks = table.blocks().to_vec();
    let free = pool.free.clone();
    let counts = pool.ref_counts.clone();
    assert_eq!(table.append(&mut pool, 10), Err(OutOfBlocks));
    assert_eq!(table.num_tokens(), 3);
    assert_eq!(table.blocks(), blocks);
    assert_eq!(pool.free, free);
    assert_eq!(pool.ref_counts, counts);
}

#[test]
fn token_count_overflow_is_atomic_and_large_block_sizes_work() {
    let mut pool = BlockPool::new(2, u32::MAX);
    let mut table = BlockTable::new();
    table.append(&mut pool, u32::MAX).unwrap();
    assert_eq!(table.blocks().len(), 1);
    let blocks = table.blocks().to_vec();
    let free = pool.free.clone();
    let counts = pool.ref_counts.clone();
    assert_eq!(table.append(&mut pool, 1), Err(OutOfBlocks));
    assert_eq!(table.num_tokens(), u32::MAX);
    assert_eq!(table.blocks(), blocks);
    assert_eq!(pool.free, free);
    assert_eq!(pool.ref_counts, counts);
    assert_eq!(table.make_last_writable(&mut pool), Ok(None));
    table.truncate(&mut pool, 1);
    assert_eq!(table.blocks().len(), 1);
    table.release(&mut pool);
    assert_eq!(pool.free_blocks(), 2);
}

#[test]
fn fork_shares_all_blocks_and_copy_on_write_replaces_only_partial_tail() {
    let mut pool = BlockPool::new(3, 4);
    let mut original = BlockTable::new();
    original.append(&mut pool, 5).unwrap();
    let blocks = original.blocks().to_vec();
    let mut fork = original.fork(&mut pool);
    assert_eq!(fork.blocks(), blocks);
    assert_eq!(fork.num_tokens(), 5);
    assert_eq!(pool.free_blocks(), 1);
    for &block in &blocks {
        assert_eq!(pool.ref_count(block), 2);
    }
    let copy = fork.make_last_writable(&mut pool).unwrap().unwrap();
    assert_eq!(copy.src, blocks[1]);
    assert_ne!(copy.src, copy.dst);
    assert_eq!(fork.blocks(), &[blocks[0], copy.dst]);
    assert_eq!(original.blocks(), blocks);
    assert_eq!(fork.num_tokens(), 5);
    assert_eq!(pool.ref_count(blocks[0]), 2);
    assert_eq!(pool.ref_count(copy.src), 1);
    assert_eq!(pool.ref_count(copy.dst), 1);
    assert_eq!(pool.free_blocks(), 0);
    assert_eq!(fork.make_last_writable(&mut pool), Ok(None));
    assert_eq!(original.make_last_writable(&mut pool), Ok(None));
    fork.append(&mut pool, 3).unwrap();
    fork.release(&mut pool);
    assert_eq!(pool.free_blocks(), 1);
    original.release(&mut pool);
    assert_eq!(pool.free_blocks(), 3);
}

#[test]
fn full_shared_tail_needs_no_copy_but_truncated_shared_tail_does() {
    let mut pool = BlockPool::new(1, 4);
    let mut table = BlockTable::new();
    table.append(&mut pool, 4).unwrap();
    let fork = table.fork(&mut pool);
    assert_eq!(table.make_last_writable(&mut pool), Ok(None));
    table.truncate(&mut pool, 3);
    let blocks = table.blocks().to_vec();
    let free = pool.free.clone();
    let counts = pool.ref_counts.clone();
    assert_eq!(table.make_last_writable(&mut pool), Err(OutOfBlocks));
    assert_eq!(table.blocks(), blocks);
    assert_eq!(table.num_tokens(), 3);
    assert_eq!(fork.num_tokens(), 4);
    assert_eq!(pool.free, free);
    assert_eq!(pool.ref_counts, counts);
    fork.release(&mut pool);
    assert_eq!(table.make_last_writable(&mut pool), Ok(None));
    table.release(&mut pool);
    assert_eq!(pool.free_blocks(), 1);
}

#[test]
fn truncate_releases_only_whole_blocks_and_keeps_shared_owners_alive() {
    let mut pool = BlockPool::new(3, 4);
    let mut table = BlockTable::new();
    table.append(&mut pool, 10).unwrap();
    let blocks = table.blocks().to_vec();
    let fork = table.fork(&mut pool);
    table.truncate(&mut pool, 9);
    assert_eq!(table.blocks(), blocks);
    table.truncate(&mut pool, 4);
    assert_eq!(table.blocks(), &blocks[..1]);
    assert_eq!(table.num_tokens(), 4);
    assert_eq!(pool.ref_count(blocks[0]), 2);
    assert_eq!(pool.ref_count(blocks[1]), 1);
    assert_eq!(pool.ref_count(blocks[2]), 1);
    assert_eq!(pool.free_blocks(), 0);
    fork.release(&mut pool);
    assert_eq!(pool.free_blocks(), 2);
    table.truncate(&mut pool, 5);
    table.truncate(&mut pool, 4);
    assert_eq!(table.num_tokens(), 4);
    table.truncate(&mut pool, 0);
    assert!(table.blocks().is_empty());
    assert_eq!(table.num_tokens(), 0);
    assert_eq!(pool.free_blocks(), 3);
    table.append(&mut pool, 1).unwrap();
    assert_eq!(table.blocks(), &blocks[..1]);
    table.release(&mut pool);
}

fn assert_invariants(pool: &BlockPool, tables: &[BlockTable], tokens: &[u32], total: u32) {
    let mut owners = vec![0; total as usize];
    for (table, &expected_tokens) in tables.iter().zip(tokens) {
        assert_eq!(table.num_tokens(), expected_tokens);
        let expected_blocks = u64::from(expected_tokens).div_ceil(u64::from(pool.block_size()));
        assert_eq!(table.blocks().len() as u64, expected_blocks);
        let mut seen = HashSet::new();
        for &block in table.blocks() {
            assert!(block.0 < total);
            assert!(seen.insert(block));
            assert!(pool.ref_count(block) > 0);
            owners[block.0 as usize] += 1;
        }
    }
    let distinct_live = owners.iter().filter(|&&count| count > 0).count();
    assert_eq!(pool.free_blocks() as usize + distinct_live, total as usize);
    for (id, &count) in owners.iter().enumerate() {
        assert_eq!(pool.ref_count(BlockId(id as u32)), count);
    }
    let free: HashSet<_> = pool.free.iter().copied().collect();
    assert_eq!(free.len(), pool.free_blocks() as usize);
    for block in free {
        assert_eq!(owners[block.0 as usize], 0);
    }
}

proptest! {
    #[test]
    fn random_table_operations_preserve_ownership(
        total in 0u32..33,
        block_size in 1u32..17,
        operations in prop::collection::vec((0u8..5, 0usize..8, 0u32..65), 1..400),
    ) {
        let mut pool = BlockPool::new(total, block_size);
        let mut tables: Vec<_> = (0..8).map(|_| BlockTable::new()).collect();
        let mut tokens = vec![0; 8];
        assert_invariants(&pool, &tables, &tokens, total);
        for (op, index, amount) in operations {
            let old_blocks = tables[index].blocks().to_vec();
            let old_free = pool.free.clone();
            let old_counts = pool.ref_counts.clone();
            match op {
                0 => {
                    let old_capacity = old_blocks.len() as u64 * u64::from(block_size);
                    let new_tokens = tokens[index] + amount;
                    let capacity = old_capacity + u64::from(pool.free_blocks()) * u64::from(block_size);
                    let result = tables[index].append(&mut pool, amount);
                    if u64::from(new_tokens) <= capacity {
                        prop_assert_eq!(result, Ok(()));
                        tokens[index] = new_tokens;
                        prop_assert!(tables[index].blocks().starts_with(&old_blocks));
                    } else {
                        prop_assert_eq!(result, Err(OutOfBlocks));
                        prop_assert_eq!(tables[index].blocks(), old_blocks);
                        prop_assert_eq!(&pool.free, &old_free);
                        prop_assert_eq!(&pool.ref_counts, &old_counts);
                    }
                }
                1 => {
                    let target = (index + 1 + amount as usize % 7) % tables.len();
                    std::mem::take(&mut tables[target]).release(&mut pool);
                    tables[target] = tables[index].fork(&mut pool);
                    tokens[target] = tokens[index];
                    prop_assert_eq!(tables[target].blocks(), &old_blocks);
                }
                2 => {
                    let needs_copy = !tokens[index].is_multiple_of(block_size)
                        && old_counts[old_blocks.last().unwrap().0 as usize] > 1;
                    let result = tables[index].make_last_writable(&mut pool);
                    if !needs_copy {
                        prop_assert_eq!(result, Ok(None));
                        prop_assert_eq!(tables[index].blocks(), old_blocks);
                        prop_assert_eq!(&pool.free, &old_free);
                        prop_assert_eq!(&pool.ref_counts, &old_counts);
                    } else if old_free.is_empty() {
                        prop_assert_eq!(result, Err(OutOfBlocks));
                        prop_assert_eq!(tables[index].blocks(), old_blocks);
                        prop_assert_eq!(&pool.free, &old_free);
                        prop_assert_eq!(&pool.ref_counts, &old_counts);
                    } else {
                        let copy = result.unwrap().unwrap();
                        prop_assert_eq!(Some(&copy.src), old_blocks.last());
                        prop_assert_eq!(Some(&copy.dst), old_free.last());
                        prop_assert_eq!(tables[index].blocks().last(), Some(&copy.dst));
                        prop_assert_eq!(&tables[index].blocks()[..old_blocks.len() - 1],
                            &old_blocks[..old_blocks.len() - 1]);
                    }
                }
                3 => {
                    tables[index].truncate(&mut pool, amount);
                    tokens[index] = tokens[index].min(amount);
                    prop_assert!(old_blocks.starts_with(tables[index].blocks()));
                }
                4 => {
                    std::mem::take(&mut tables[index]).release(&mut pool);
                    tokens[index] = 0;
                }
                _ => unreachable!(),
            }
            assert_invariants(&pool, &tables, &tokens, total);
        }
        for table in tables {
            table.release(&mut pool);
        }
        prop_assert_eq!(pool.free_blocks(), total);
        prop_assert!(pool.ref_counts.iter().all(|&count| count == 0));
    }
}

#[test]
fn from_shared_takes_a_reference_per_block() {
    let mut pool = BlockPool::new(4, 16);
    let mut owner = BlockTable::new();
    owner.append(&mut pool, 40).unwrap();
    let shared = BlockTable::from_shared(&mut pool, &owner.blocks()[..2]);
    assert_eq!(shared.num_tokens(), 32);
    assert_eq!(shared.blocks(), &owner.blocks()[..2]);
    assert_eq!(pool.ref_count(owner.blocks()[0]), 2);
    assert_eq!(pool.ref_count(owner.blocks()[2]), 1);
    owner.release(&mut pool);
    assert_eq!(pool.free_blocks(), 2);
    shared.release(&mut pool);
    assert_eq!(pool.free_blocks(), 4);
}

#[test]
fn fast_blocks_go_first_and_come_back_to_their_list() {
    // Blocks 0..3 fast (device memory), 3..6 slow (a host tier).
    let mut pool = BlockPool::with_fast(6, 16, 3);
    let got: Vec<u32> = (0..6).map(|_| pool.allocate().unwrap().0).collect();
    assert_eq!(got, [0, 1, 2, 3, 4, 5]);
    assert!(pool.allocate().is_none());
    // A slow block freed first, then a fast one: the fast one is handed out first.
    assert!(pool.decref(BlockId(4)));
    assert!(pool.decref(BlockId(1)));
    assert_eq!((pool.free_blocks(), pool.free_fast_blocks()), (2, 1));
    assert_eq!(pool.allocate(), Some(BlockId(1)));
    assert_eq!(pool.allocate(), Some(BlockId(4)));
    // Without a slow part the pool is as before.
    let mut plain = BlockPool::new(2, 16);
    assert_eq!(plain.free_fast_blocks(), 2);
    assert_eq!(plain.allocate(), Some(BlockId(0)));
}

#[test]
fn slow_blocks_in_use_are_counted() {
    let mut pool = BlockPool::with_fast(6, 16, 3);
    assert_eq!(pool.slow_in_use(), 0);
    let got: Vec<BlockId> = (0..5).map(|_| pool.allocate().unwrap()).collect();
    assert_eq!(pool.slow_in_use(), 2);
    pool.incref(got[4]);
    assert!(!pool.decref(got[4]));
    assert_eq!(pool.slow_in_use(), 2);
    assert!(pool.decref(got[3]));
    assert!(pool.decref(got[4]));
    assert_eq!(pool.slow_in_use(), 0);
    // Fast blocks never count, nor does a pool without slow blocks or a slotted one.
    assert!(pool.decref(got[0]));
    assert_eq!(pool.slow_in_use(), 0);
    let mut plain = BlockPool::new(2, 16);
    plain.allocate().unwrap();
    assert_eq!(plain.slow_in_use(), 0);
    assert_eq!(BlockPool::slotted(2, 2, 16).slow_in_use(), 0);
}

#[test]
fn slotted_tables_take_their_slots_blocks_in_order_through_trims() {
    let mut pool = BlockPool::slotted(3, 4, 16);
    assert_eq!((pool.free_blocks(), pool.slot_blocks()), (12, 4));
    assert_eq!(pool.allocate(), None);
    let mut a = BlockTable::new();
    let mut b = BlockTable::new();
    a.append_in(&mut pool, 40, 2).unwrap();
    b.append_in(&mut pool, 1, 0).unwrap();
    assert_eq!(a.blocks(), &[BlockId(8), BlockId(9), BlockId(10)]);
    assert_eq!(b.blocks(), &[BlockId(0)]);
    assert_eq!(pool.free_blocks(), 8);
    // Rolled back and grown again: the same blocks, in order.
    a.truncate(&mut pool, 17);
    assert_eq!(pool.free_blocks(), 9);
    a.append_in(&mut pool, 30, 2).unwrap();
    assert_eq!(a.blocks(), &[BlockId(8), BlockId(9), BlockId(10)]);
    // Past the slot: nothing changes.
    assert_eq!(a.append_in(&mut pool, 64, 2), Err(OutOfBlocks));
    assert_eq!((a.num_tokens(), pool.free_blocks()), (47, 8));
    a.release(&mut pool);
    b.release(&mut pool);
    assert_eq!(pool.free_blocks(), 12);
}

#[test]
fn a_capped_slotted_pool_hands_out_no_more_than_its_cap() {
    let mut pool = BlockPool::slotted(3, 4, 16);
    pool.set_slotted_cap(5);
    assert_eq!(pool.free_blocks(), 5);
    let (mut a, mut b) = (BlockTable::new(), BlockTable::new());
    a.append_in(&mut pool, 64, 0).unwrap();
    assert_eq!(pool.free_blocks(), 1);
    // Room in slot 1, not under the cap: nothing changes.
    assert_eq!(b.append_in(&mut pool, 32, 1), Err(OutOfBlocks));
    assert_eq!((b.num_tokens(), pool.free_blocks()), (0, 1));
    b.append_in(&mut pool, 16, 1).unwrap();
    assert_eq!(pool.free_blocks(), 0);
    a.truncate(&mut pool, 16);
    assert_eq!(pool.free_blocks(), 3);
    b.append_in(&mut pool, 32, 1).unwrap();
    assert_eq!(b.blocks(), &[BlockId(4), BlockId(5), BlockId(6)]);
    a.release(&mut pool);
    b.release(&mut pool);
    assert_eq!(pool.free_blocks(), 5);
    // Never past the pool's size.
    pool.set_slotted_cap(100);
    assert_eq!(pool.free_blocks(), 12);
}

#[test]
fn slow_blocks_in_use_are_listed_ascending() {
    let mut pool = BlockPool::with_fast(8, 16, 3);
    assert!(pool.slow_used().is_empty());
    let got: Vec<BlockId> = (0..7).map(|_| pool.allocate().unwrap()).collect();
    // The fast ones first, then the slow ones lowest first (7 is still free).
    assert_eq!(pool.slow_used(), [3, 4, 5, 6]);
    assert!(pool.decref(got[4]));
    pool.incref(got[5]);
    assert!(!pool.decref(got[5]));
    let mut slow: Vec<u32> = got[3..].iter().map(|b| b.0).collect();
    slow.retain(|&b| b != got[4].0);
    slow.sort_unstable();
    assert_eq!(pool.slow_used(), slow);
    // Fast blocks are never listed; nor is anything of a slotted pool.
    assert!(pool.slow_used().iter().all(|&b| b >= 3));
    assert!(BlockPool::slotted(2, 2, 16).slow_used().is_empty());
}
