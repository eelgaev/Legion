//! Suffix-decoding drafter: drafts from a suffix trie built over earlier requests (a shared
//! [`SuffixCache`]) and over the sequence being generated.
//!
//! Written from the description in the SuffixDecoding paper (Oliaro et al., 2024); the data
//! structure is a suffix trie truncated at `max_depth`. Arctic Inference (Apache-2.0) implements
//! the same idea; no code was copied from it.
//!
//! Data structure: every node is a distinct substring of length <= `max_depth + 1` (a context of up
//! to `max_depth` tokens plus the token after it) and stores how many
//! times it occurred. The children of a node therefore are the observed next tokens after that
//! substring, with their frequencies. Indexing a sequence of `n` tokens costs `O(n)` hash
//! operations (each token extends at most `max_depth` active nodes, so `O(n * max_depth)`), memory
//! is `O(n * max_depth)` nodes in the worst case, a lookup of a context of length `L` is `O(L)`,
//! and evicting a sequence costs the same as inserting it.

use super::TokenDrafter;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

pub const DEFAULT_MAX_DEPTH: usize = 64;
pub const DEFAULT_MAX_TOKENS: usize = 4_000_000;
/// Drafting stops when the best next token's share of all continuations is below this.
const MIN_SHARE: f64 = 0.1;

const ROOT: u32 = 0;

#[derive(Default)]
struct Node {
    /// Occurrences of the substring this node spells.
    count: u32,
    children: HashMap<u32, u32>,
}

/// Suffix trie truncated at `max_depth`, with per-node occurrence counts.
struct Trie {
    nodes: Vec<Node>,
    free: Vec<u32>,
    max_depth: usize,
}

impl Trie {
    fn new(max_depth: usize) -> Self {
        Trie {
            nodes: vec![Node::default()],
            free: Vec::new(),
            max_depth: max_depth.max(1),
        }
    }

    fn alloc(&mut self) -> u32 {
        if let Some(id) = self.free.pop() {
            id
        } else {
            self.nodes.push(Node::default());
            (self.nodes.len() - 1) as u32
        }
    }

    /// Adds token `t` after every active context (nodes of the suffixes of the sequence so far,
    /// of length < max_depth), then replaces `active` by the contexts including `t`.
    fn append(&mut self, active: &mut Vec<u32>, t: u32) {
        if active.is_empty() {
            active.push(ROOT);
        }
        let mut next = Vec::with_capacity(active.len() + 1);
        // The root (empty context) is always active.
        next.push(ROOT);
        for &node in active.iter() {
            let child = match self.nodes[node as usize].children.get(&t) {
                Some(&c) => c,
                None => {
                    let c = self.alloc();
                    self.nodes[node as usize].children.insert(t, c);
                    c
                }
            };
            self.nodes[child as usize].count += 1;
            next.push(child);
        }
        // Keep contexts shorter than max_depth (the longest is last).
        next.truncate(self.max_depth);
        *active = next;
    }

    fn insert(&mut self, seq: &[u32]) {
        let mut active = vec![ROOT];
        for &t in seq {
            self.append(&mut active, t);
        }
    }

    /// Undoes `insert(seq)`: decrements every counted substring and frees nodes that reach zero.
    fn remove(&mut self, seq: &[u32]) {
        for start in 0..seq.len() {
            let end = (start + self.max_depth).min(seq.len());
            let mut parent = ROOT;
            for &t in &seq[start..end] {
                let Some(&child) = self.nodes[parent as usize].children.get(&t) else {
                    break;
                };
                let n = &mut self.nodes[child as usize];
                n.count -= 1;
                if n.count == 0 {
                    // Every pass through this node has been undone, so its subtree is dead too.
                    self.nodes[parent as usize].children.remove(&t);
                    self.release(child);
                    break;
                }
                parent = child;
            }
        }
    }

    fn release(&mut self, root: u32) {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            let node = std::mem::take(&mut self.nodes[id as usize]);
            stack.extend(node.children.into_values());
            self.free.push(id);
        }
    }

    fn find(&self, ctx: &[u32]) -> Option<u32> {
        let mut node = ROOT;
        for t in ctx {
            node = *self.nodes[node as usize].children.get(t)?;
        }
        Some(node)
    }

    /// Adds the continuation counts of `ctx` into `acc`.
    fn continuations(&self, ctx: &[u32], acc: &mut HashMap<u32, u32>) {
        if let Some(node) = self.find(ctx) {
            for (&t, &c) in &self.nodes[node as usize].children {
                *acc.entry(t).or_insert(0) += self.nodes[c as usize].count;
            }
        }
    }

    #[cfg(test)]
    fn node_count(&self) -> usize {
        self.nodes.len() - self.free.len()
    }
}

struct CacheInner {
    trie: Trie,
    seqs: VecDeque<Vec<u32>>,
    total_tokens: usize,
    max_tokens: usize,
}

/// Global, thread-safe store of finished sequences, shared by all drafters.
#[derive(Clone)]
pub struct SuffixCache {
    inner: Arc<Mutex<CacheInner>>,
}

impl Default for SuffixCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_DEPTH, DEFAULT_MAX_TOKENS)
    }
}

impl SuffixCache {
    pub fn new(max_depth: usize, max_tokens: usize) -> Self {
        SuffixCache {
            inner: Arc::new(Mutex::new(CacheInner {
                // Nodes spell context + next token, so they are one deeper than the context.
                trie: Trie::new(max_depth.max(1) + 1),
                seqs: VecDeque::new(),
                total_tokens: 0,
                max_tokens,
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, CacheInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn max_depth(&self) -> usize {
        self.lock().trie.max_depth - 1
    }

    /// Indexes a finished request (prompt + output). Evicts the oldest whole sequences (FIFO)
    /// while the total indexed tokens exceed `max_tokens`.
    pub fn insert(&self, seq: &[u32]) {
        if seq.is_empty() {
            return;
        }
        let mut g = self.lock();
        g.trie.insert(seq);
        g.seqs.push_back(seq.to_vec());
        g.total_tokens += seq.len();
        while g.total_tokens > g.max_tokens {
            let Some(old) = g.seqs.pop_front() else { break };
            g.trie.remove(&old);
            g.total_tokens -= old.len();
        }
    }

    /// Total tokens currently indexed.
    pub fn total_tokens(&self) -> usize {
        self.lock().total_tokens
    }

    /// Number of sequences currently indexed.
    pub fn len(&self) -> usize {
        self.lock().seqs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Per-request drafter over the shared cache plus the request's own tokens.
pub struct SuffixDrafter {
    cache: SuffixCache,
    /// Tokens already indexed in `local_trie`.
    local: Vec<u32>,
    local_trie: Trie,
    local_active: Vec<u32>,
}

impl SuffixDrafter {
    pub fn new(cache: SuffixCache) -> Self {
        let depth = cache.max_depth() + 1;
        SuffixDrafter {
            cache,
            local: Vec::new(),
            local_trie: Trie::new(depth),
            local_active: Vec::new(),
        }
    }

    /// Inserts the finished sequence (prompt + output) into the shared cache.
    pub fn finish(self, all_tokens: &[u32]) {
        self.cache.insert(all_tokens);
    }

    fn sync_local(&mut self, tokens: &[u32]) {
        if !tokens.starts_with(&self.local) {
            // The sequence was rewritten (not just extended): start over.
            self.local.clear();
            self.local_trie = Trie::new(self.local_trie.max_depth);
            self.local_active.clear();
        }
        for &t in &tokens[self.local.len()..] {
            self.local_trie.append(&mut self.local_active, t);
        }
        self.local.clear();
        self.local.extend_from_slice(tokens);
    }
}

impl TokenDrafter for SuffixDrafter {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        if max == 0 || tokens.is_empty() {
            return;
        }
        self.sync_local(tokens);
        let cache = self.cache.lock();
        let depth = cache.trie.max_depth - 1;
        let mut seq = tokens.to_vec();
        let mut acc: HashMap<u32, u32> = HashMap::new();
        for _ in 0..max {
            // Longest suffix of `seq` with at least one observed continuation.
            let mut best: Option<(u32, u32)> = None; // (count, token) with ties to the smaller id
            let mut total = 0u64;
            for l in (1..=depth.min(seq.len())).rev() {
                let ctx = &seq[seq.len() - l..];
                acc.clear();
                cache.trie.continuations(ctx, &mut acc);
                self.local_trie.continuations(ctx, &mut acc);
                if acc.is_empty() {
                    continue;
                }
                total = acc.values().map(|&c| c as u64).sum();
                best = acc.iter().map(|(&t, &c)| (c, t)).reduce(|a, b| {
                    if (b.0, std::cmp::Reverse(b.1)) > (a.0, std::cmp::Reverse(a.1)) {
                        b
                    } else {
                        a
                    }
                });
                break;
            }
            let Some((count, tok)) = best else { break };
            if (count as f64) / (total as f64) < MIN_SHARE {
                break;
            }
            out.push(tok);
            seq.push(tok);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(d: &mut SuffixDrafter, tokens: &[u32], max: usize) -> Vec<u32> {
        let mut out = Vec::new();
        d.propose(tokens, max, &mut out);
        out
    }

    #[test]
    fn drafts_inserted_sequence_from_prefix() {
        let cache = SuffixCache::new(8, 1000);
        cache.insert(&[10, 11, 12, 13, 14, 15]);
        let mut d = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d, &[99, 10, 11], 4), vec![12, 13, 14, 15]);
        assert_eq!(draft(&mut d, &[10, 11], 2), vec![12, 13]);
        assert!(draft(&mut d, &[77, 78], 4).is_empty());
    }

    #[test]
    fn drafts_from_own_earlier_tokens() {
        let mut d = SuffixDrafter::new(SuffixCache::new(8, 1000));
        let toks = [1, 2, 3, 4, 5, 1, 2];
        assert_eq!(draft(&mut d, &toks, 3), vec![3, 4, 5]);
    }

    #[test]
    fn most_frequent_continuation_wins() {
        let cache = SuffixCache::new(8, 1000);
        cache.insert(&[1, 2, 7]);
        cache.insert(&[1, 2, 9]);
        cache.insert(&[1, 2, 9]);
        let mut d = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d, &[1, 2], 1), vec![9]);
    }

    #[test]
    fn ties_go_to_smallest_token_id() {
        let cache = SuffixCache::new(8, 1000);
        cache.insert(&[1, 2, 9]);
        cache.insert(&[1, 2, 7]);
        cache.insert(&[1, 2, 8]);
        let mut d = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d, &[1, 2], 1), vec![7]);
    }

    #[test]
    fn share_cutoff_stops_drafting() {
        let cache = SuffixCache::new(8, 1000);
        // 11 equally likely continuations: best share 1/11 < 0.1.
        for t in 100..111 {
            cache.insert(&[1, t]);
        }
        let mut d = SuffixDrafter::new(cache.clone());
        assert!(draft(&mut d, &[1], 4).is_empty());
        // 10 continuations: share is exactly 0.1, which is not below the cutoff.
        let cache = SuffixCache::new(8, 1000);
        for t in 100..110 {
            cache.insert(&[1, t]);
        }
        let mut d = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d, &[1], 4), vec![100]);
    }

    #[test]
    fn fifo_eviction_past_max_tokens() {
        let cache = SuffixCache::new(8, 10);
        cache.insert(&[1, 2, 3, 4, 5, 6]);
        cache.insert(&[11, 12, 13, 14]);
        assert_eq!(cache.total_tokens(), 10);
        cache.insert(&[21, 22, 23]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_tokens(), 7);
        let mut d = SuffixDrafter::new(cache.clone());
        assert!(draft(&mut d, &[1, 2], 3).is_empty());
        assert_eq!(draft(&mut d, &[11, 12], 2), vec![13, 14]);
        assert_eq!(draft(&mut d, &[21, 22], 1), vec![23]);
        // Evicting everything frees every node but the root.
        cache.insert(&[5; 11]);
        assert!(cache.is_empty());
        assert_eq!(cache.lock().trie.node_count(), 1);
    }

    #[test]
    fn eviction_keeps_counts_of_shared_substrings() {
        let cache = SuffixCache::new(8, 7);
        cache.insert(&[1, 2, 3]);
        cache.insert(&[1, 2, 4, 4]);
        cache.insert(&[8]);
        // [1,2,3] evicted; [1,2,4,4] remains.
        let mut d = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d, &[1, 2], 2), vec![4, 4]);
    }

    #[test]
    fn max_zero_gives_empty_out() {
        let cache = SuffixCache::new(8, 1000);
        cache.insert(&[1, 2, 3]);
        let mut d = SuffixDrafter::new(cache);
        let mut out = Vec::new();
        d.propose(&[1, 2], 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn max_depth_truncates_context() {
        let cache = SuffixCache::new(2, 1000);
        cache.insert(&[1, 2, 3, 4, 5]);
        let mut d = SuffixDrafter::new(cache);
        // Context of the last 2 tokens is enough to continue deeper than max_depth.
        assert_eq!(draft(&mut d, &[0, 1, 2], 3), vec![3, 4, 5]);
    }

    #[test]
    fn finish_inserts_into_cache() {
        let cache = SuffixCache::new(8, 1000);
        let d = SuffixDrafter::new(cache.clone());
        d.finish(&[4, 5, 6, 7]);
        let mut d2 = SuffixDrafter::new(cache);
        assert_eq!(draft(&mut d2, &[4, 5], 2), vec![6, 7]);
    }

    #[test]
    fn rewritten_sequence_resets_local_index() {
        let mut d = SuffixDrafter::new(SuffixCache::new(8, 1000));
        assert_eq!(draft(&mut d, &[1, 2, 3, 1, 2], 1), vec![3]);
        assert!(draft(&mut d, &[5, 6, 1, 2], 1).is_empty());
    }

    #[test]
    fn concurrent_inserts_are_all_findable() {
        let cache = SuffixCache::new(8, 1_000_000);
        let handles: Vec<_> = (0..4u32)
            .map(|k| {
                let c = cache.clone();
                std::thread::spawn(move || {
                    for i in 0..50u32 {
                        let base = 1000 * (k + 1) + 10 * i;
                        c.insert(&[base, base + 1, base + 2, base + 3]);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cache.len(), 200);
        let mut d = SuffixDrafter::new(cache);
        for k in 0..4u32 {
            for i in 0..50u32 {
                let base = 1000 * (k + 1) + 10 * i;
                assert_eq!(
                    draft(&mut d, &[base], 3),
                    vec![base + 1, base + 2, base + 3]
                );
            }
        }
    }
}
