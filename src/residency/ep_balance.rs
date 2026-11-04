//! Expert parallelism's routed work per step, and its balance within GPU pairs (#729).
//!
//! Under expert parallelism (GLM's `--glm-parallel-decode`) GPU `d` computes the chosen experts
//! it owns: those its cache holds read from VRAM, the others read in place from host memory,
//! about 12 times slower per expert on the AC922 (V100 HBM2 against NVLink 2.0). The step's
//! exchanges wait for the GPU with the most work at every MoE layer, and with a few tokens a
//! step that GPU changes from layer to layer: the number of a GPU's experts the router picks,
//! and how many of those miss its cache, vary.
//!
//! **Pair balance** ([`assign_pair`]): the two GPUs of a pair (consecutive CUDA indices, an
//! NVLink pair on one socket of the AC922, [`pair`]) read the same node's host memory equally
//! fast, so either can compute a chosen expert that misses its owner's cache. Each cached
//! expert stays with its owner; the uncached ones, in ascending id order, go to the GPU of the
//! pair with less work so far ([`HIT_COST`], [`MISS_COST`]), its owner on a tie. Both GPUs of
//! a pair run this on the same inputs (the router's choices, the same on every GPU, and which
//! experts each owner caches), so they agree on who computes what without talking.
//!
//! **Cross balance** ([`assign_cross`], #731): the same over every GPU. An uncached expert may
//! also go to a GPU of the other pair, which reads it across the POWER9 X-bus from the owner's
//! node at about two thirds of the speed ([`CROSS_MISS_COST`]), when it still finishes there
//! first. The pair balance leaves about two thirds of the miss imbalance, pair against pair.
//!
//! [`BalanceStats`] counts every logged step's work per GPU and layer, as owned and as the
//! pair and cross balances would give it, for the expert cache's periodic report.

#[cfg(test)]
mod tests;

use std::ops::Range;

/// The work of an expert read from VRAM, in [`MISS_COST`]'s units.
pub const HIT_COST: u32 = 1;
/// The work of an expert read in place from host memory: about 12 VRAM reads on the AC922
/// (a 6.8 MiB Maya-S expert at ~800 GB/s from HBM2, ~62 GB/s over NVLink 2.0).
pub const MISS_COST: u32 = 12;
/// The work of an expert read from the other socket's host memory, across the X-bus: about
/// 2/3 of a local read's speed on the AC922 (62 against 93 GiB/s), so 1.5 [`MISS_COST`].
pub const CROSS_MISS_COST: u32 = 18;

/// The GPUs of `d`'s pair among `n_dev`: `2k..2k + 2` (the last GPU of an odd count alone).
pub fn pair(d: usize, n_dev: usize) -> Range<usize> {
    let p0 = d & !1;
    p0..(p0 + 2).min(n_dev)
}

/// Which GPU computes each of `chosen` (a step's distinct chosen experts of one layer, in
/// ascending id order, each with its owner and whether the owner caches it) on `n_dev` GPUs
/// under the pair balance: cached experts their owner, each uncached one the GPU of its
/// owner's [`pair`] with the least work so far, the owner on a tie (then the lower index).
/// The CUDA kernel of `glm_ep_assign` (`src/cuda/glm_exchange.cu`) is this, bit for bit.
pub fn assign_pair(chosen: &[(u32, usize, bool)], n_dev: usize) -> Vec<usize> {
    let mut cost = vec![0u64; n_dev];
    let mut out = vec![usize::MAX; chosen.len()];
    for (i, &(_, owner, cached)) in chosen.iter().enumerate() {
        if cached {
            out[i] = owner;
            cost[owner] += u64::from(HIT_COST);
        }
    }
    for (i, &(_, owner, cached)) in chosen.iter().enumerate() {
        if cached {
            continue;
        }
        let mut best = owner;
        for d in pair(owner, n_dev) {
            if cost[d] < cost[best] {
                best = d;
            }
        }
        out[i] = best;
        cost[best] += u64::from(MISS_COST);
    }
    out
}

/// The work of GPU `d` computing an uncached expert `owner` owns: [`MISS_COST`] in the
/// owner's [`pair`] (the same node's host memory), [`CROSS_MISS_COST`] across the sockets.
pub fn miss_cost(d: usize, owner: usize, n_dev: usize) -> u32 {
    if pair(owner, n_dev).contains(&d) {
        MISS_COST
    } else {
        CROSS_MISS_COST
    }
}

/// [`assign_pair`] over every GPU (`--glm-cross-balance`, #731): cached experts their owner,
/// each uncached one, in ascending id order, the GPU where it finishes first (its work so far
/// plus [`miss_cost`]); on a tie the owner, then its pair's other GPU, then the lowest index.
/// The CUDA kernel of `glm_ep_assign_cross` (`src/cuda/glm_exchange.cu`) is this, bit for bit.
pub fn assign_cross(chosen: &[(u32, usize, bool)], n_dev: usize) -> Vec<usize> {
    let mut cost = vec![0u64; n_dev];
    let mut out = vec![usize::MAX; chosen.len()];
    for (i, &(_, owner, cached)) in chosen.iter().enumerate() {
        if cached {
            out[i] = owner;
            cost[owner] += u64::from(HIT_COST);
        }
    }
    for (i, &(_, owner, cached)) in chosen.iter().enumerate() {
        if cached {
            continue;
        }
        let at = |d: usize| cost[d] + u64::from(miss_cost(d, owner, n_dev));
        let mut best = owner;
        let candidates =
            pair(owner, n_dev).chain((0..n_dev).filter(|d| !pair(owner, n_dev).contains(d)));
        for d in candidates {
            if at(d) < at(best) {
                best = d;
            }
        }
        out[i] = best;
        cost[best] += u64::from(miss_cost(best, owner, n_dev));
    }
    out
}

/// One entry's (one GPU's share of one layer's) experts in a logged pass: its distinct chosen
/// experts it owns, each with whether its cache held it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPass {
    /// The model layer the entry serves.
    pub real: usize,
    /// Its GPU (CUDA index).
    pub gpu: usize,
    /// (expert, read from VRAM).
    pub chosen: Vec<(u32, bool)>,
}

/// The routed work of the logged steps since the last report ([`Self::take`]), per GPU and
/// summed over the layers the GPUs share: what each GPU owned and how far the slowest GPU
/// was above the mean at each layer, as owned and with the pair balance.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct BalanceStats {
    /// Steps counted (logged passes over two layers or more).
    pub steps: u64,
    /// Layers counted, over the steps.
    pub layers: u64,
    /// Per GPU: chosen experts it owned, read from VRAM / from host memory.
    pub hits: Vec<u64>,
    pub misses: Vec<u64>,
    /// Summed over the layers: the most misses on one GPU less the mean over the GPUs.
    pub miss_excess: f64,
    /// Summed over the layers: the most work ([`HIT_COST`], [`MISS_COST`]) on one GPU less
    /// the mean, as owned and with the pair balance ([`assign_pair`]).
    pub cost_excess: f64,
    pub pair_excess: f64,
    /// The same with the cross balance ([`assign_cross`]): its busiest GPU less the owned work's
    /// mean, since the cross-socket reads add work ([`CROSS_MISS_COST`]).
    pub cross_excess: f64,
}

impl BalanceStats {
    /// Counts one logged pass: `entries`, every entry that ran. Layers served by one GPU alone
    /// and passes over one layer (an MTP head's) are not counted.
    pub fn add_pass(&mut self, entries: &[EntryPass]) {
        let mut reals: Vec<usize> = entries.iter().map(|e| e.real).collect();
        reals.sort_unstable();
        reals.dedup();
        if reals.len() < 2 {
            return;
        }
        let n_dev = entries.iter().map(|e| e.gpu + 1).max().unwrap_or(0);
        if self.hits.len() < n_dev {
            self.hits.resize(n_dev, 0);
            self.misses.resize(n_dev, 0);
        }
        let mut counted = false;
        for real in reals {
            let group: Vec<&EntryPass> = entries.iter().filter(|e| e.real == real).collect();
            if group.len() < 2 {
                continue;
            }
            counted = true;
            self.layers += 1;
            let mut hits = vec![0u64; n_dev];
            let mut misses = vec![0u64; n_dev];
            let mut chosen: Vec<(u32, usize, bool)> = Vec::new();
            for e in &group {
                for &(x, vram) in &e.chosen {
                    if vram {
                        hits[e.gpu] += 1;
                    } else {
                        misses[e.gpu] += 1;
                    }
                    chosen.push((x, e.gpu, vram));
                }
            }
            chosen.sort_unstable_by_key(|c| c.0);
            chosen.dedup_by_key(|c| c.0);
            let gpus: Vec<usize> = {
                let mut g: Vec<usize> = group.iter().map(|e| e.gpu).collect();
                g.sort_unstable();
                g.dedup();
                g
            };
            let max = |v: &[u64]| gpus.iter().map(|&d| v[d] as f64).fold(0.0, f64::max);
            let mean =
                |v: &[u64]| gpus.iter().map(|&d| v[d] as f64).sum::<f64>() / gpus.len() as f64;
            let excess = |v: &[u64]| -> f64 { max(v) - mean(v) };
            let cost: Vec<u64> = (0..n_dev)
                .map(|d| hits[d] * u64::from(HIT_COST) + misses[d] * u64::from(MISS_COST))
                .collect();
            let mut paired = vec![0u64; n_dev];
            for (c, d) in chosen.iter().zip(assign_pair(&chosen, n_dev)) {
                paired[d] += u64::from(if c.2 { HIT_COST } else { MISS_COST });
            }
            let mut crossed = vec![0u64; n_dev];
            for (c, d) in chosen.iter().zip(assign_cross(&chosen, n_dev)) {
                crossed[d] += u64::from(if c.2 {
                    HIT_COST
                } else {
                    miss_cost(d, c.1, n_dev)
                });
            }
            self.miss_excess += excess(&misses);
            self.cost_excess += excess(&cost);
            self.pair_excess += excess(&paired);
            self.cross_excess += max(&crossed) - mean(&cost);
            for d in 0..n_dev {
                self.hits[d] += hits[d];
                self.misses[d] += misses[d];
            }
        }
        if counted {
            self.steps += 1;
        }
    }

    /// The counts so far, reset.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}
