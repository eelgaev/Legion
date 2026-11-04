//! How well a GPU's uncached experts of the next layer could be foreseen (#745,
//! `--glm-prefetch-stats`).
//!
//! Under expert parallelism (GLM's `--glm-parallel-decode`) each GPU reads the chosen experts
//! its cache does not hold in place from host memory, about 115 µs per Maya-S expert on the
//! AC922 against about 9 µs from VRAM, and those reads bunch up inside each layer while the
//! link idles the rest of the step. A prefetch would copy layer `L + 1`'s likely misses into
//! VRAM while layer `L` computes. [`PrefetchStats`] counts, per GPU, how many of a logged
//! pass's misses two predictors would have named, and how many uncached experts each would
//! have copied for it:
//! - **next router**: layer `L + 1`'s router (its weights and bias) applied to layer `L`'s FFN
//!   input, computed in the step's graph and logged beside the real choices;
//! - **previous step**: the experts the same entry chose in its previous logged pass.

#[cfg(test)]
mod tests;

/// One GPU's counts, summed over the logged passes' entries that had a prediction.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GpuCounts {
    /// Distinct chosen experts it owned, and those of them read from host memory.
    pub chosen: u64,
    pub misses: u64,
    /// Misses the next layer's router named / the previous pass chose / either did.
    pub next_hits: u64,
    pub prev_hits: u64,
    pub either_hits: u64,
    /// Chosen experts (cached or not) the next layer's router named.
    pub next_chosen: u64,
    /// Uncached experts each predictor names: what it would copy.
    pub next_copies: u64,
    pub prev_copies: u64,
}

/// The predictors' counts per GPU since the last report ([`Self::take`]).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Logged passes with at least one predicted entry.
    pub steps: u64,
    pub gpus: Vec<GpuCounts>,
}

/// One entry's (one GPU's share of one layer's) logged pass, for [`PrefetchStats::add`].
pub struct EntryPredicted<'a> {
    pub gpu: usize,
    /// Its distinct chosen experts it owns, ascending, each with whether VRAM served it.
    pub chosen: &'a [(u32, bool)],
    /// The experts the next-layer router named that it owns, ascending, distinct.
    pub next: &'a [u32],
    /// The experts it chose in its previous logged pass, ascending, distinct.
    pub prev: &'a [u32],
}

impl PrefetchStats {
    /// Counts one entry; `uncached(e)`: whether its cache does not hold expert `e` now.
    pub fn add(&mut self, p: &EntryPredicted<'_>, uncached: impl Fn(u32) -> bool) {
        if self.gpus.len() <= p.gpu {
            self.gpus.resize(p.gpu + 1, GpuCounts::default());
        }
        let g = &mut self.gpus[p.gpu];
        let has = |v: &[u32], e: u32| v.binary_search(&e).is_ok();
        for &(e, vram) in p.chosen {
            g.chosen += 1;
            let (n, q) = (has(p.next, e), has(p.prev, e));
            g.next_chosen += u64::from(n);
            if !vram {
                g.misses += 1;
                g.next_hits += u64::from(n);
                g.prev_hits += u64::from(q);
                g.either_hits += u64::from(n || q);
            }
        }
        g.next_copies += p.next.iter().filter(|&&e| uncached(e)).count() as u64;
        g.prev_copies += p.prev.iter().filter(|&&e| uncached(e)).count() as u64;
    }

    /// Counts a logged pass that had predicted entries.
    pub fn end_pass(&mut self) {
        self.steps += 1;
    }

    /// The counts so far, reset.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

/// The distinct experts of `ids` (a logged pass's predicted ids, `-1` unwritten) that
/// `owns`, ascending.
pub fn owned_distinct(ids: &[i32], owns: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut v: Vec<u32> = ids
        .iter()
        .filter(|&&id| id >= 0)
        .map(|&id| id as u32)
        .filter(|&e| owns(e))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Most tokens of a step `--glm-prefetch` predicts for (#747): longer steps (prompts) read
/// their experts as before.
pub const MAX_TOKENS: usize = 16;

/// Most (token, expert) pairs of a predicted step: `stage_copy`'s and the slot tables' limit.
pub const MAX_PAIRS: usize = 512;

/// Bytes of a table of one i32 per pair, rounded to 256 (`expert_mv::overlap::stage_header`).
const TABLE: usize = (MAX_PAIRS * 4).next_multiple_of(256);

/// Where things are in one half of a GPU's `--glm-prefetch` staging ring (#747), in bytes from
/// the half's start; the ring holds two halves, layer `L`'s guess in half `L % 2`, so that
/// one layer's copies fill a half while the layer before reads the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingLayout {
    /// `stage_copy`'s slot per guess pair (its header: the slots are at [`Self::slots`]).
    pub copy_table: usize,
    /// The guess: its tiered ids of the GPU's own experts (-1 for the others'), rank-major
    /// (every token's first choice, then their second ones, ...).
    pub guess: usize,
    /// The guess as raw expert ids, laid out as [`Self::guess`].
    pub raw: usize,
    /// The step's real pairs' staging slot (-1: not staged), the matvec's slot table.
    pub real: usize,
    /// A copy of the layer's FFN input, `[MAX_TOKENS][k]` F32.
    pub x: usize,
    /// The router's logits, `[MAX_TOKENS][n_expert]` F32.
    pub logits: usize,
    /// The route kernel's CTA counter (zero between launches).
    pub done: usize,
    /// The staged experts: `cap` slots of [`Self::slot`] bytes, each an expert's gate, up and
    /// down rows one after the other.
    pub slots: usize,
    pub slot: usize,
    pub cap: usize,
    /// One half's bytes.
    pub half: usize,
}

impl RingLayout {
    /// The layout for `cap` experts per layer of at most `slot_bytes` (gate, up and down) each,
    /// FFN inputs of `k` floats and `n_expert` experts.
    pub fn new(cap: usize, slot_bytes: usize, k: usize, n_expert: usize) -> Self {
        let r = |b: usize| b.next_multiple_of(256);
        let copy_table = 0;
        let guess = copy_table + TABLE;
        let raw = guess + TABLE;
        let real = raw + TABLE;
        let x = real + TABLE;
        let logits = x + r(MAX_TOKENS * k * 4);
        let done = logits + r(MAX_TOKENS * n_expert * 4);
        let slots = done + 256;
        let slot = r(slot_bytes);
        Self {
            copy_table,
            guess,
            raw,
            real,
            x,
            logits,
            done,
            slots,
            slot,
            cap,
            half: slots + cap * slot,
        }
    }

    /// The whole ring's bytes: two halves.
    pub fn bytes(&self) -> usize {
        2 * self.half
    }

    /// Where the half for layer `il`'s guess starts.
    pub fn half_of(&self, il: usize) -> usize {
        (il % 2) * self.half
    }
}

/// Whether a step of `n` tokens choosing `n_used` experts each is predicted.
pub fn predicts(n: usize, n_used: usize) -> bool {
    n > 0 && n <= MAX_TOKENS && n * n_used <= MAX_PAIRS
}

/// The experts `stage_copy` stages for each GPU of a pair, as the pair-balance's cached-with-
/// staged kernel counts them: in the order of their first guess pairs (`raw`, rank-major), a
/// GPU's own experts its owner does not cache take its `cap` slots while they last. `lo..mid`
/// are the first GPU's experts, `mid..hi` the second's. The host model of
/// `cached_kernel` in src/cuda/glm_prefetch.cu.
pub fn staged_in_pair(
    raw: &[i32],
    cached: &[i32],
    (lo, mid, hi): (i32, i32, i32),
    cap: usize,
) -> Vec<i32> {
    let mut out = cached.to_vec();
    let mut used = [0usize; 2];
    for &e in raw {
        if e < lo || e >= hi || out[e as usize] != 0 {
            continue;
        }
        let side = usize::from(e >= mid);
        if used[side] < cap {
            used[side] += 1;
            out[e as usize] = 1;
        }
    }
    out
}
