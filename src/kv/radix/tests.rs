use super::*;
use crate::kv::BlockTable;
use proptest::prelude::*;

fn insert(cache: &mut RadixCache, pool: &mut BlockPool, tokens: &[u32]) -> Vec<BlockId> {
    let mut table = BlockTable::new();
    table.append(pool, tokens.len() as u32).unwrap();
    let blocks = table.blocks().to_vec();
    cache.insert(pool, tokens, &blocks);
    table.release(pool);
    blocks
}

fn check_match(cache: &mut RadixCache, tokens: &[u32], expected: &[BlockId]) {
    let matched = cache.match_prefix(tokens);
    assert_eq!(matched.blocks, expected);
    assert_eq!(matched.num_tokens, expected.len() as u32 * cache.block_size);
    cache.unlock(&matched);
}

#[test]
fn insert_match_and_duplicate_preserve_original_ownership() {
    let mut pool = BlockPool::new(12, 2);
    let mut cache = RadixCache::new(2);
    check_match(&mut cache, &[1, 2], &[]);
    let blocks = insert(&mut cache, &mut pool, &[1, 2, 3, 4, 5, 6]);
    assert_eq!(cache.nodes.len(), 2, "a new path is one compressed edge");
    assert_eq!(cache.cached_blocks(), 3);
    let duplicates = insert(&mut cache, &mut pool, &[1, 2, 3, 4, 5, 6]);
    for block in duplicates {
        assert_eq!(pool.ref_count(block), 0);
    }
    for &block in &blocks {
        assert_eq!(pool.ref_count(block), 1);
    }
    check_match(&mut cache, &[1, 2, 3, 4, 5, 6, 7, 8], &blocks);
    check_match(&mut cache, &[], &[]);
    check_match(&mut cache, &[1, 9], &[]);
    assert_eq!(cache.evict(&mut pool, 1), 3, "whole edges are evicted");
    assert_eq!(cache.cached_blocks(), 0);
    assert_eq!(pool.free_blocks(), 12);
}

#[test]
fn divergence_splits_only_at_full_block_boundaries() {
    let mut pool = BlockPool::new(12, 2);
    let mut cache = RadixCache::new(2);
    let a = insert(&mut cache, &mut pool, &[1, 2, 3, 4, 5, 6]);
    let b = insert(&mut cache, &mut pool, &[1, 2, 3, 9, 7, 8]);
    assert_eq!(cache.nodes.len(), 4);
    assert_eq!(cache.cached_blocks(), 5);
    assert_eq!(pool.ref_count(b[0]), 0);
    check_match(&mut cache, &[1, 2, 3, 4, 5, 6], &a);
    check_match(&mut cache, &[1, 2, 3, 9, 7, 8], &[a[0], b[1], b[2]]);
    check_match(&mut cache, &[1, 2, 3, 5], &a[..1]);
    assert_eq!(cache.evict(&mut pool, u32::MAX), 5);
    assert_eq!(pool.free_blocks(), 12);
}

#[test]
fn partial_last_blocks_are_neither_stored_nor_matched() {
    let mut pool = BlockPool::new(8, 2);
    let mut cache = RadixCache::new(2);
    insert(&mut cache, &mut pool, &[1]);
    assert_eq!(cache.cached_blocks(), 0);
    let blocks = insert(&mut cache, &mut pool, &[1, 2, 3, 4, 5]);
    assert_eq!(cache.cached_blocks(), 2);
    assert_eq!(pool.ref_count(blocks[2]), 0);
    check_match(&mut cache, &[1, 2, 3, 4, 5, 6], &blocks[..2]);
    check_match(&mut cache, &[1, 2, 3], &blocks[..1]);
    check_match(&mut cache, &[1], &[]);
}

#[test]
fn locks_survive_insert_and_match_splits_and_unlock_independently() {
    let mut pool = BlockPool::new(16, 1);
    let mut cache = RadixCache::new(1);
    let blocks = insert(&mut cache, &mut pool, &[1, 2, 3, 4]);
    let full = cache.match_prefix(&[1, 2, 3, 4]);
    let prefix = cache.match_prefix(&[1, 2]);
    let another = cache.match_prefix(&[1, 2]);
    insert(&mut cache, &mut pool, &[1, 8, 9]);
    assert_eq!(cache.evict(&mut pool, u32::MAX), 2);
    assert_eq!(cache.cached_blocks(), 4);
    cache.unlock(&full);
    assert_eq!(cache.evict(&mut pool, u32::MAX), 2);
    assert_eq!(cache.cached_blocks(), 2);
    cache.unlock(&prefix);
    assert_eq!(cache.evict(&mut pool, u32::MAX), 0);
    check_match(&mut cache, &[1, 2, 3, 4], &blocks[..2]);
    cache.unlock(&another);
    assert_eq!(cache.evict(&mut pool, u32::MAX), 2);
    assert_eq!(pool.free_blocks(), 16);
}

#[test]
fn matching_a_short_prefix_leaves_the_suffix_evictable() {
    let mut pool = BlockPool::new(8, 1);
    let mut cache = RadixCache::new(1);
    let blocks = insert(&mut cache, &mut pool, &[1, 2, 3, 4]);
    let prefix = cache.match_prefix(&[1, 2]);
    assert_eq!(cache.evict(&mut pool, 1), 2);
    assert_eq!(prefix.blocks, blocks[..2]);
    assert_eq!(cache.cached_blocks(), 2);
    cache.unlock(&prefix);
    assert_eq!(cache.evict(&mut pool, 1), 2);
}

#[test]
fn lru_refreshes_matches_and_inserts_and_reconsiders_parent_leaves() {
    let mut pool = BlockPool::new(16, 1);
    let mut cache = RadixCache::new(1);
    let a = insert(&mut cache, &mut pool, &[1, 2]);
    let b = insert(&mut cache, &mut pool, &[1, 3]);
    let c = insert(&mut cache, &mut pool, &[4]);
    check_match(&mut cache, &[1, 2], &a);
    assert_eq!(cache.evict(&mut pool, 1), 1);
    assert_eq!(pool.ref_count(b[1]), 0);
    insert(&mut cache, &mut pool, &[4]);
    assert_eq!(cache.evict(&mut pool, 2), 2);
    for block in a {
        assert_eq!(pool.ref_count(block), 0);
    }
    check_match(&mut cache, &[4], &c);
    assert_eq!(cache.evict(&mut pool, 0), 0);
    assert_eq!(cache.cached_blocks(), 1);
}

#[test]
fn eviction_counts_only_blocks_returned_to_the_pool() {
    let mut pool = BlockPool::new(8, 1);
    let mut cache = RadixCache::new(1);
    let mut table = BlockTable::new();
    table.append(&mut pool, 2).unwrap();
    cache.insert(&mut pool, &[1, 2], table.blocks());
    let freeable = insert(&mut cache, &mut pool, &[3]);
    assert_eq!(cache.evict(&mut pool, 1), 1);
    assert_eq!(cache.cached_blocks(), 0);
    assert_eq!(pool.ref_count(freeable[0]), 0);
    for &block in table.blocks() {
        assert_eq!(pool.ref_count(block), 1);
    }
    cache.insert(&mut pool, &[1, 2], table.blocks());
    assert_eq!(cache.evict(&mut pool, 1), 0);
    assert_eq!(cache.cached_blocks(), 0);
    table.release(&mut pool);
    assert_eq!(pool.free_blocks(), 8);
}

#[test]
#[should_panic(expected = "block size must be nonzero")]
fn zero_block_size_is_rejected() {
    RadixCache::new(0);
}

// Independent flat-prefix oracle. Every complete prefix maps to its final block.
// Endpoints record historical split boundaries only, so eviction can remove the
// same whole edges without using the implementation's tree or traversal.
#[derive(Default)]
struct Reference {
    blocks: HashMap<Vec<u32>, BlockId>,
    access: HashMap<Vec<u32>, u64>,
    locks: HashMap<Vec<u32>, u32>,
    endpoints: HashMap<Vec<u32>, usize>,
    next_id: usize,
    tick: u64,
}

impl Reference {
    fn endpoint(&mut self, tokens: &[u32]) {
        if !tokens.is_empty() && !self.endpoints.contains_key(tokens) {
            self.next_id += 1;
            self.endpoints.insert(tokens.to_vec(), self.next_id);
        }
    }

    fn walk(&mut self, tokens: &[u32], size: usize) -> usize {
        self.tick += 1;
        let mut matched = 0;
        for end in (size..=tokens.len()).step_by(size) {
            if !self.blocks.contains_key(&tokens[..end]) {
                break;
            }
            self.access.insert(tokens[..end].to_vec(), self.tick);
            matched = end;
        }
        self.endpoint(&tokens[..matched]);
        matched
    }

    fn insert(&mut self, tokens: &[u32], blocks: &[BlockId], size: usize) {
        self.walk(tokens, size);
        let full = tokens.len() / size * size;
        for end in (size..=full).step_by(size) {
            self.blocks
                .entry(tokens[..end].to_vec())
                .or_insert(blocks[end / size - 1]);
            self.access.insert(tokens[..end].to_vec(), self.tick);
            self.locks.entry(tokens[..end].to_vec()).or_default();
        }
        self.endpoint(&tokens[..full]);
    }

    fn match_prefix(&mut self, tokens: &[u32], size: usize) -> (Vec<BlockId>, Vec<u32>) {
        let matched = self.walk(tokens, size);
        let mut blocks = Vec::new();
        for end in (size..=matched).step_by(size) {
            blocks.push(self.blocks[&tokens[..end]]);
            *self.locks.get_mut(&tokens[..end]).unwrap() += 1;
        }
        (blocks, tokens[..matched].to_vec())
    }

    fn unlock(&mut self, tokens: &[u32], size: usize) {
        for end in (size..=tokens.len()).step_by(size) {
            *self.locks.get_mut(&tokens[..end]).unwrap() -= 1;
        }
    }

    fn evict(&mut self, n: u32, tables: &[BlockTable], size: usize) -> u32 {
        let mut freed = 0;
        while freed < n {
            let leaf = self
                .endpoints
                .iter()
                .filter(|(key, _)| {
                    self.locks[*key] == 0
                        && !self
                            .endpoints
                            .keys()
                            .any(|other| other.len() > key.len() && other.starts_with(key))
                })
                .min_by_key(|(key, id)| (self.access[*key], **id))
                .map(|(key, _)| key.clone());
            let Some(leaf) = leaf else { break };
            let parent_len = self
                .endpoints
                .keys()
                .filter(|key| key.len() < leaf.len() && leaf.starts_with(key))
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            self.endpoints.remove(&leaf);
            for end in (parent_len + size..=leaf.len()).step_by(size) {
                let key = &leaf[..end];
                let block = self.blocks.remove(key).unwrap();
                self.access.remove(key);
                self.locks.remove(key);
                if !self.blocks.values().any(|&b| b == block)
                    && !tables.iter().any(|table| table.blocks().contains(&block))
                {
                    freed += 1;
                }
            }
        }
        freed
    }
}

fn assert_invariants(
    cache: &RadixCache,
    reference: &Reference,
    pool: &BlockPool,
    tables: &[BlockTable],
    total: u32,
) {
    let mut actual = HashMap::new();
    let mut stack = vec![(0, Vec::new())];
    let mut visited = 0;
    while let Some((id, prefix)) = stack.pop() {
        visited += 1;
        let node = &cache.nodes[&id];
        assert_eq!(
            node.tokens.len(),
            node.blocks.len() * cache.block_size as usize
        );
        if id != 0 {
            assert!(!node.blocks.is_empty());
        }
        let mut key = prefix;
        for (chunk, &block) in node
            .tokens
            .chunks_exact(cache.block_size as usize)
            .zip(&node.blocks)
        {
            key.extend_from_slice(chunk);
            assert!(actual.insert(key.clone(), block).is_none());
            assert_eq!(node.locks, reference.locks[&key]);
            assert_eq!(node.last_access, reference.access[&key]);
        }
        for (first_block, &child) in &node.children {
            assert_eq!(cache.nodes[&child].parent, id);
            assert_eq!(
                first_block,
                &cache.nodes[&child].tokens[..cache.block_size as usize]
            );
            stack.push((child, key.clone()));
        }
    }
    assert_eq!(visited, cache.nodes.len());
    assert_eq!(actual, reference.blocks);
    assert_eq!(cache.cached_blocks() as usize, reference.blocks.len());
    let mut owners = vec![0; total as usize];
    for &block in reference
        .blocks
        .values()
        .chain(tables.iter().flat_map(|t| t.blocks()))
    {
        owners[block.0 as usize] += 1;
    }
    assert_eq!(
        pool.free_blocks() as usize + owners.iter().filter(|&&n| n > 0).count(),
        total as usize
    );
    for (id, count) in owners.into_iter().enumerate() {
        assert_eq!(pool.ref_count(BlockId(id as u32)), count);
    }
}

proptest! {
    #[test]
    fn random_operations_match_flat_prefix_reference(
        size in 1u32..5,
        operations in prop::collection::vec(
            (0u8..6, 0usize..4, prop::collection::vec(0u32..3, 0..17), 0u32..8), 1..200),
    ) {
        const TOTAL: u32 = 48;
        let mut pool = BlockPool::new(TOTAL, size);
        let mut cache = RadixCache::new(size);
        let mut reference = Reference::default();
        let mut tables: Vec<_> = (0..4).map(|_| BlockTable::new()).collect();
        let mut table_tokens = vec![Vec::new(); 4];
        let mut matches: Vec<Option<(PrefixMatch, Vec<u32>)>> = (0..4).map(|_| None).collect();
        for (op, index, tokens, amount) in operations {
            match op {
                0 => {
                    std::mem::take(&mut tables[index]).release(&mut pool);
                    table_tokens[index].clear();
                    if tables[index].append(&mut pool, tokens.len() as u32).is_ok() {
                        cache.insert(&mut pool, &tokens, tables[index].blocks());
                        reference.insert(&tokens, tables[index].blocks(), size as usize);
                        table_tokens[index] = tokens;
                    }
                }
                1 => {
                    if let Some((m, key)) = matches[index].take() {
                        cache.unlock(&m);
                        reference.unlock(&key, size as usize);
                    }
                    let (expected, key) = reference.match_prefix(&tokens, size as usize);
                    let m = cache.match_prefix(&tokens);
                    prop_assert_eq!(&m.blocks, &expected);
                    prop_assert_eq!(m.num_tokens as usize, key.len());
                    matches[index] = Some((m, key));
                }
                2 => {
                    if let Some((m, key)) = matches[index].take() {
                        cache.unlock(&m);
                        reference.unlock(&key, size as usize);
                    }
                }
                3 => {
                    let expected = reference.evict(amount, &tables, size as usize);
                    prop_assert_eq!(cache.evict(&mut pool, amount), expected);
                }
                4 => {
                    std::mem::take(&mut tables[index]).release(&mut pool);
                    table_tokens[index].clear();
                }
                5 => {
                    cache.insert(&mut pool, &table_tokens[index], tables[index].blocks());
                    reference.insert(&table_tokens[index], tables[index].blocks(), size as usize);
                }
                _ => unreachable!(),
            }
            assert_invariants(&cache, &reference, &pool, &tables, TOTAL);
        }
        for (m, key) in matches.into_iter().flatten() {
            cache.unlock(&m);
            reference.unlock(&key, size as usize);
        }
        for table in &mut tables {
            std::mem::take(table).release(&mut pool);
        }
        let expected = reference.evict(u32::MAX, &tables, size as usize);
        prop_assert_eq!(cache.evict(&mut pool, u32::MAX), expected);
        assert_invariants(&cache, &reference, &pool, &tables, TOTAL);
        prop_assert_eq!(pool.free_blocks(), TOTAL);
    }
}

#[test]
fn evict_from_drops_entries_past_a_block_and_their_longer_prefixes() {
    // Blocks 0..4 fast, 4.. past the boundary.
    let mut pool = BlockPool::with_fast(12, 2, 4);
    let mut cache = RadixCache::new(2);
    // [1, 2 | 3, 4] in blocks 0, 1; held by the cache only.
    let a = insert(&mut cache, &mut pool, &[1, 2, 3, 4]);
    assert_eq!(a, [BlockId(0), BlockId(1)]);
    // Keep blocks 2, 3 busy so the next prefix lands past the boundary.
    let mut busy = BlockTable::new();
    busy.append(&mut pool, 4).unwrap();
    // [1, 2, 3, 4 | 5, 6] extends `a` in a block past the boundary; so is [7, 8].
    let b = insert(&mut cache, &mut pool, &[1, 2, 3, 4, 5, 6]);
    assert!(b[2].0 >= 4);
    let c = insert(&mut cache, &mut pool, &[7, 8]);
    assert!(c[0].0 >= 4);
    assert_eq!(cache.cached_blocks(), 4);
    assert_eq!(pool.slow_in_use(), 2);
    assert_eq!(cache.evict_from(&mut pool, 4), 2);
    assert_eq!(pool.slow_in_use(), 0);
    // The kept prefix stays.
    check_match(&mut cache, &[1, 2, 3, 4, 5, 6], &a);
    check_match(&mut cache, &[7, 8], &[]);
    assert_eq!(cache.evict_from(&mut pool, 4), 0);
    busy.release(&mut pool);
}

#[test]
fn evict_from_drops_the_longer_prefixes_under_a_past_entry() {
    let mut pool = BlockPool::with_fast(12, 2, 1);
    let mut cache = RadixCache::new(2);
    let mut busy = BlockTable::new();
    busy.append(&mut pool, 2).unwrap();
    // A root entry past the boundary (block 1) with a longer prefix under it.
    let a = insert(&mut cache, &mut pool, &[1, 2]);
    assert_eq!(a, [BlockId(1)]);
    insert(&mut cache, &mut pool, &[1, 2, 3, 4]);
    assert_eq!(cache.cached_blocks(), 2);
    assert_eq!(cache.evict_from(&mut pool, 1), 2);
    assert_eq!(cache.cached_blocks(), 0);
    // A locked entry stays.
    insert(&mut cache, &mut pool, &[5, 6]);
    let m = cache.match_prefix(&[5, 6]);
    assert_eq!(cache.evict_from(&mut pool, 1), 0);
    cache.unlock(&m);
    assert_eq!(cache.evict_from(&mut pool, 1), 1);
    busy.release(&mut pool);
}
