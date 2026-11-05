//! Block-aligned prefix sharing over a compressed radix tree.
//!
//! The cache owns one pool reference per stored block. Matches borrow that
//! ownership through explicit locks; call `unlock` once for each match. Use the
//! same pool throughout, and evict all entries before dropping the cache to
//! release its references. Cached KV contents must remain immutable.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::blocks::{BlockId, BlockPool};

#[derive(Debug, Default)]
struct Node {
    parent: usize,
    tokens: Vec<u32>,
    blocks: Vec<BlockId>,
    children: HashMap<Vec<u32>, usize>,
    locks: u32,
    last_access: u64,
}

#[derive(Debug)]
struct NodeHandle(usize);

/// A cached prefix, protected from eviction until passed to `unlock`.
/// Dropping this value does not unlock it or change pool reference counts.
#[derive(Debug)]
pub struct PrefixMatch {
    pub blocks: Vec<BlockId>,
    pub num_tokens: u32,
    handle: NodeHandle,
}

/// A radix tree whose edges contain one or more complete KV blocks.
#[derive(Debug)]
pub struct RadixCache {
    block_size: u32,
    nodes: HashMap<usize, Node>,
    next_node: usize,
    tick: u64,
    cached_blocks: u32,
}

impl RadixCache {
    /// Create an empty cache. `block_size` must be nonzero.
    pub fn new(block_size: u32) -> Self {
        assert!(block_size > 0, "block size must be nonzero");
        Self {
            block_size,
            nodes: HashMap::from([(0, Node::default())]),
            next_node: 1,
            tick: 0,
            cached_blocks: 0,
        }
    }

    /// Find and lock the longest cached prefix, refreshing its LRU ticks.
    /// A partial final block is ignored. Unlock the result on this cache exactly
    /// once, even if the match is empty. Locks do not add pool references.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        let tokens = self.full_tokens(tokens);
        let (last, num_tokens) = self.walk(tokens);
        let mut path = Vec::new();
        let mut current = last;
        while current != 0 {
            let node = self.nodes.get_mut(&current).unwrap();
            node.locks = node
                .locks
                .checked_add(1)
                .expect("cache lock count overflow");
            path.push(current);
            current = node.parent;
        }
        let mut blocks = Vec::with_capacity(num_tokens / self.block_size as usize);
        for id in path.into_iter().rev() {
            blocks.extend_from_slice(&self.nodes[&id].blocks);
        }
        PrefixMatch {
            blocks,
            num_tokens: num_tokens as u32,
            handle: NodeHandle(last),
        }
    }

    /// Release a match's path lock. `m` must be a still-locked match from this cache.
    pub fn unlock(&mut self, m: &PrefixMatch) {
        let mut current = m.handle.0;
        while current != 0 {
            let node = self
                .nodes
                .get_mut(&current)
                .expect("match belongs to this cache");
            node.locks = node.locks.checked_sub(1).expect("match already unlocked");
            current = node.parent;
        }
    }

    /// Store complete blocks, taking one reference only for newly cached entries.
    /// Existing prefixes retain their original block IDs. `blocks` must cover all
    /// complete token blocks and belong to `pool`, whose block size must match.
    pub fn insert(&mut self, pool: &mut BlockPool, tokens: &[u32], blocks: &[BlockId]) {
        assert_eq!(pool.block_size(), self.block_size, "block sizes must match");
        let tokens = self.full_tokens(tokens);
        let size = self.block_size as usize;
        assert!(blocks.len() >= tokens.len() / size, "missing full blocks");
        let (parent, matched) = self.walk(tokens);
        if matched == tokens.len() {
            return;
        }
        let blocks = &blocks[matched / size..tokens.len() / size];
        for &block in blocks {
            pool.incref(block);
        }
        let node = Node {
            parent,
            tokens: tokens[matched..].to_vec(),
            blocks: blocks.to_vec(),
            last_access: self.tick,
            ..Node::default()
        };
        let id = self.add_node(node);
        self.nodes
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(tokens[matched..matched + size].to_vec(), id);
        self.cached_blocks += blocks.len() as u32;
    }

    /// Evict unlocked leaves in LRU order until at least `n` blocks become free,
    /// or no evictable leaves remain. Whole edges are removed, so this may free
    /// more than `n`. Blocks still owned by sequences do not count as freed.
    pub fn evict(&mut self, pool: &mut BlockPool, n: u32) -> u32 {
        assert_eq!(pool.block_size(), self.block_size, "block sizes must match");
        if n == 0 {
            return 0;
        }
        let mut leaves: BinaryHeap<_> = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| {
                (id != 0 && node.locks == 0 && node.children.is_empty())
                    .then_some(Reverse((node.last_access, id)))
            })
            .collect();
        let mut freed = 0;
        while freed < n {
            let Some(Reverse((_, id))) = leaves.pop() else {
                break;
            };
            let node = self.nodes.remove(&id).unwrap();
            self.cached_blocks -= node.blocks.len() as u32;
            for block in node.blocks {
                freed += u32::from(pool.decref(block));
            }
            let parent = self.nodes.get_mut(&node.parent).unwrap();
            parent
                .children
                .remove(&node.tokens[..self.block_size as usize]);
            if node.parent != 0 && parent.locks == 0 && parent.children.is_empty() {
                leaves.push(Reverse((parent.last_access, node.parent)));
            }
        }
        freed
    }

    /// Evicts every unlocked entry that holds a block from `first` on, and the longer cached
    /// prefixes under it (they cannot stay without it): a KV pool giving its blocks past
    /// `first` back (`--kv-vmm-keep`). Returns the blocks freed.
    pub fn evict_from(&mut self, pool: &mut BlockPool, first: u32) -> u32 {
        let past = |node: &Node| node.blocks.iter().any(|b| b.0 >= first);
        let mut freed = 0;
        loop {
            // Unlocked leaves at or under an entry with such a block.
            let doomed: Vec<usize> = self
                .nodes
                .iter()
                .filter(|(id, node)| **id != 0 && node.locks == 0 && node.children.is_empty())
                .filter(|(id, _)| {
                    let mut at = **id;
                    while at != 0 {
                        let node = &self.nodes[&at];
                        if past(node) {
                            return true;
                        }
                        at = node.parent;
                    }
                    false
                })
                .map(|(&id, _)| id)
                .collect();
            if doomed.is_empty() {
                return freed;
            }
            for id in doomed {
                let node = self.nodes.remove(&id).unwrap();
                self.cached_blocks -= node.blocks.len() as u32;
                for block in node.blocks {
                    freed += u32::from(pool.decref(block));
                }
                let parent = self.nodes.get_mut(&node.parent).unwrap();
                parent
                    .children
                    .remove(&node.tokens[..self.block_size as usize]);
            }
        }
    }

    /// Evicts the least recently used unlocked leaf that holds a block from `first` on or lies
    /// under an entry that does: one step of [`Self::evict_from`] in LRU order, for a KV
    /// pool that keeps what fits of its blocks past `first` in host memory (#1196). False
    /// when there is no such leaf.
    pub fn evict_lru_from(&mut self, pool: &mut BlockPool, first: u32) -> bool {
        let past = |node: &Node| node.blocks.iter().any(|b| b.0 >= first);
        let victim = self
            .nodes
            .iter()
            .filter(|(id, node)| **id != 0 && node.locks == 0 && node.children.is_empty())
            .filter(|(id, _)| {
                let mut at = **id;
                while at != 0 {
                    let node = &self.nodes[&at];
                    if past(node) {
                        return true;
                    }
                    at = node.parent;
                }
                false
            })
            .min_by_key(|(id, node)| (node.last_access, **id))
            .map(|(&id, _)| id);
        let Some(id) = victim else {
            return false;
        };
        let node = self.nodes.remove(&id).unwrap();
        self.cached_blocks -= node.blocks.len() as u32;
        for block in node.blocks {
            pool.decref(block);
        }
        let parent = self.nodes.get_mut(&node.parent).unwrap();
        parent
            .children
            .remove(&node.tokens[..self.block_size as usize]);
        true
    }

    pub fn cached_blocks(&self) -> u32 {
        self.cached_blocks
    }

    fn full_tokens<'a>(&self, tokens: &'a [u32]) -> &'a [u32] {
        assert!(u32::try_from(tokens.len()).is_ok(), "token count overflow");
        let size = self.block_size as usize;
        &tokens[..tokens.len() / size * size]
    }

    fn add_node(&mut self, node: Node) -> usize {
        let id = self.next_node;
        self.next_node = id.checked_add(1).expect("cache node ID overflow");
        self.nodes.insert(id, node);
        id
    }

    // Return the last matched node, exposing an edge boundary when necessary.
    fn walk(&mut self, tokens: &[u32]) -> (usize, usize) {
        self.tick = self.tick.checked_add(1).expect("cache LRU tick overflow");
        let size = self.block_size as usize;
        let mut current = 0;
        let mut matched = 0;
        while matched < tokens.len() {
            let Some(&child) = self.nodes[&current]
                .children
                .get(&tokens[matched..matched + size])
            else {
                break;
            };
            let node = &self.nodes[&child];
            let common = node
                .tokens
                .chunks_exact(size)
                .zip(tokens[matched..].chunks_exact(size))
                .take_while(|(a, b)| a == b)
                .count()
                * size;
            current = if common < node.tokens.len() {
                self.split(child, common)
            } else {
                child
            };
            self.nodes.get_mut(&current).unwrap().last_access = self.tick;
            matched += common;
        }
        (current, matched)
    }

    // Keep the suffix's ID stable so outstanding handles still reach the root.
    // The new parent inherits its locks without taking additional pool refs.
    fn split(&mut self, child: usize, num_tokens: usize) -> usize {
        let size = self.block_size as usize;
        let node = self.nodes.get_mut(&child).unwrap();
        let suffix_tokens = node.tokens.split_off(num_tokens);
        let prefix_tokens = std::mem::replace(&mut node.tokens, suffix_tokens);
        let suffix_blocks = node.blocks.split_off(num_tokens / size);
        let prefix_blocks = std::mem::replace(&mut node.blocks, suffix_blocks);
        let parent = node.parent;
        let key = prefix_tokens[..size].to_vec();
        let prefix = Node {
            parent,
            tokens: prefix_tokens,
            blocks: prefix_blocks,
            children: HashMap::from([(node.tokens[..size].to_vec(), child)]),
            locks: node.locks,
            last_access: node.last_access,
        };
        let id = self.add_node(prefix);
        self.nodes.get_mut(&child).unwrap().parent = id;
        self.nodes
            .get_mut(&parent)
            .unwrap()
            .children
            .insert(key, id);
        id
    }
}

#[cfg(test)]
mod tests;
