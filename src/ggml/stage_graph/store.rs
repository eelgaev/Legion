//! Stage-graph captures kept by their graph's words (#858), so a graph built again (a kept step
//! graph dropped and rebuilt for a shape seen before, in the same compute buffer) launches the
//! captures of the graph it repeats instead of capturing again.
//!
//! A capture launches the kernels of a graph with the words ([`super::props`], plus the late
//! input's place) it was captured with: what each node computes, from which memory into which.
//! Two graphs with the same words launch the same kernels with the same arguments, which is
//! what a graph's own captures already rely on from run to run. So the captures are kept under
//! the words, per device (`dev`, the backend), not per graph object: a set found for a graph's
//! new words is the same bits as the captures that graph would make.
//!
//! Bounded: sets unused for a while are dropped ([`Store::sweep`]), and a device keeps at most
//! `cap` of them, the least recently used going first ([`Store::insert`]).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// A hash of `words` (a graph's words in their full form): the store's key, the words compared
/// whole on a match. Four lanes, so the multiplies overlap (~40K words a Flash-Next stage).
pub(crate) fn hash(words: &[u64]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = [K, K ^ 1, K ^ 2, K ^ 3];
    let (chunks, rest) = words.as_chunks::<4>();
    for c in chunks {
        for (h, &w) in h.iter_mut().zip(c) {
            *h = (h.rotate_left(23) ^ w).wrapping_mul(K);
        }
    }
    for (h, &w) in h.iter_mut().zip(rest) {
        *h = (h.rotate_left(23) ^ w).wrapping_mul(K);
    }
    let mut out = words.len() as u64;
    for x in h {
        out = (out.rotate_left(29) ^ x).wrapping_mul(K);
    }
    out ^ out >> 32
}

/// One set of captures and the words it was made for.
pub(crate) struct Kept<T> {
    /// The graph's words, full form.
    pub words: Vec<u64>,
    /// Its id, unique: a graph that made or found the set names it by its key and id.
    pub id: u64,
    pub used: Instant,
    pub val: T,
}

impl<T> Kept<T> {
    /// A new set for `words`, with a new id.
    pub(crate) fn new(words: Vec<u64>, val: T, now: Instant) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            words,
            id: NEXT.fetch_add(1, Relaxed),
            used: now,
            val,
        }
    }
}

/// The captures, per device and words.
pub(crate) struct Store<T> {
    sets: HashMap<(usize, u64), Kept<T>>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self {
            sets: HashMap::new(),
        }
    }
}

impl<T> Store<T> {
    /// Takes the set `id` kept under `(dev, hash)`, if it still is there.
    pub(crate) fn take_id(&mut self, dev: usize, hash: u64, id: u64) -> Option<Kept<T>> {
        let key = (dev, hash);
        if self.sets.get(&key)?.id != id {
            return None;
        }
        self.sets.remove(&key)
    }

    /// Takes the set kept on `dev` for exactly `words` (hash `hash`), if any.
    pub(crate) fn take_words(&mut self, dev: usize, hash: u64, words: &[u64]) -> Option<Kept<T>> {
        let key = (dev, hash);
        if self.sets.get(&key)?.words != words {
            return None;
        }
        self.sets.remove(&key)
    }

    /// Whether `dev` keeps a set for exactly `words` (hash `hash`).
    pub(crate) fn has(&self, dev: usize, hash: u64, words: &[u64]) -> bool {
        self.sets
            .get(&(dev, hash))
            .is_some_and(|k| k.words == words)
    }

    /// Puts back a set taken from the store, as the most recently used (it replaces another
    /// set under the same key, made since by a graph of the same words).
    pub(crate) fn put(&mut self, dev: usize, hash: u64, mut k: Kept<T>, now: Instant) {
        k.used = now;
        self.sets.insert((dev, hash), k);
    }

    /// Keeps new set `k` on `dev` (hash `hash` of its words), as the most recently used. Then,
    /// while `dev` keeps more than `cap` sets (0: no limit), drops its least recently used.
    pub(crate) fn insert(&mut self, dev: usize, hash: u64, k: Kept<T>, now: Instant, cap: usize) {
        self.put(dev, hash, k, now);
        if cap == 0 {
            return;
        }
        let mut mine: Vec<(Instant, u64)> = self
            .sets
            .iter()
            .filter(|((d, h), _)| *d == dev && *h != hash)
            .map(|((_, h), k)| (k.used, *h))
            .collect();
        if mine.len() + 1 > cap {
            mine.sort_unstable();
            for (_, h) in &mine[..mine.len() + 1 - cap] {
                self.sets.remove(&(dev, *h));
            }
        }
    }

    /// Drops the sets unused for `unused`.
    pub(crate) fn sweep(&mut self, now: Instant, unused: Duration) {
        self.sets
            .retain(|_, k| now.saturating_duration_since(k.used) < unused);
    }

    /// Sets kept on `dev` (taken ones not counted).
    #[cfg(test)]
    pub(crate) fn len(&self, dev: usize) -> usize {
        self.sets.keys().filter(|(d, _)| *d == dev).count()
    }
}

#[cfg(test)]
mod tests;
