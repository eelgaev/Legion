//! Per-layer expert-cache slot counts from the misses a previous run measured
//! (`--expert-split-misses`).
//!
//! Without a profile, each device's slots go to its layers round robin, so every layer gets
//! about the same count. Layers differ in how much their misses fall per extra slot: on a
//! 4-GPU layer split one layer missed 6x as often as another with the same slots. The cache
//! counts each layer's hits and misses; on exit they are saved to a small JSON file, and the
//! next start sizes each device's layers from it.
//!
//! The model: a layer that had `s` slots and missed `m` times would miss about `m * s / n`
//! times with `n` slots (misses inversely proportional to the slots, which is close to what
//! the 2- and 4-GPU runs show between them). Minimising the sum over a device's layers under
//! its byte budget gives each slot to the layer whose next slot saves the most misses per
//! byte. At its fixed point every layer misses about the same per slot. Placement only: which
//! experts are cached never changes the outputs.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One layer's measured cache use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerMisses {
    pub layer: usize,
    pub slots: usize,
    pub hits: u64,
    pub misses: u64,
}

/// The file: every tiered layer's slots, hits and misses of one run.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MissStats {
    pub layers: Vec<LayerMisses>,
}

impl MissStats {
    /// From [`super::experts::ExpertResidency::layer_stats`]: (layer, slots, hits, misses).
    pub fn from_layer_stats(per: &[(usize, usize, u64, u64)]) -> Self {
        let layers = per
            .iter()
            .map(|&(layer, slots, hits, misses)| LayerMisses {
                layer,
                slots,
                hits,
                misses,
            })
            .collect();
        Self { layers }
    }

    pub fn read(path: &Path) -> io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    /// Writes atomically (a `.tmp` file renamed over `path`).
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        let text = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)
    }

    /// Layer `il`'s measurement, if it was tiered and had slots.
    pub fn get(&self, il: usize) -> Option<LayerMisses> {
        self.layers
            .iter()
            .find(|l| l.layer == il && l.slots > 0)
            .copied()
    }
}

/// New slot counts for one device's layers. `now[i]`: the count without the measurement;
/// `bytes[i]`: one slot's bytes (0: not on this device); `seen[i]`: the measurement;
/// `cap[i]`: the most slots the layer can use. Layers without a measurement keep `now[i]`;
/// the measured ones share the bytes their `now` counts take, each next slot going to the
/// layer whose modelled misses fall the most per byte (ties: the lower layer).
pub fn split_by_misses(
    now: &[usize],
    bytes: &[u64],
    seen: &[Option<LayerMisses>],
    cap: &[usize],
) -> Vec<usize> {
    let mut out = now.to_vec();
    let mine: Vec<usize> = (0..now.len())
        .filter(|&i| bytes[i] > 0 && seen.get(i).copied().flatten().is_some())
        .collect();
    if mine.is_empty() {
        return out;
    }
    let mut left: u64 = mine.iter().map(|&i| now[i] as u64 * bytes[i]).sum();
    // Modelled misses with n slots: a / n, a = (misses + 1) * slots (the +1 keeps a layer
    // that never missed from dropping to nothing).
    let a: Vec<f64> = (0..now.len())
        .map(|i| match seen.get(i).copied().flatten() {
            Some(s) => (s.misses as f64 + 1.0) * s.slots as f64,
            None => 0.0,
        })
        .collect();
    for &i in &mine {
        out[i] = 0;
    }
    // Every measured layer first gets one slot (a / 0 is unbounded), in layer order.
    for &i in &mine {
        if cap[i] > 0 && bytes[i] <= left {
            out[i] = 1;
            left -= bytes[i];
        }
    }
    // Then the greedy: the saving of slot n+1 is a / n - a / (n + 1) = a / (n (n + 1)).
    let gain = |i: usize, n: usize| a[i] / (n as f64 * (n as f64 + 1.0)) / bytes[i] as f64;
    let mut heap: std::collections::BinaryHeap<Pick> = mine
        .iter()
        .filter(|&&i| out[i] > 0 && out[i] < cap[i])
        .map(|&i| Pick(gain(i, out[i]), i))
        .collect();
    while let Some(Pick(_, i)) = heap.pop() {
        if bytes[i] > left {
            // Another layer's smaller slot may still fit.
            continue;
        }
        left -= bytes[i];
        out[i] += 1;
        if out[i] < cap[i] {
            heap.push(Pick(gain(i, out[i]), i));
        }
    }
    out
}

/// One device's first fill (`alloc[layer]`: its experts; `slot_bytes[layer]`: 0 for layers on
/// other devices) resized by `measured`: the counts from [`split_by_misses`], and a layer
/// whose count changed starts with its lowest (owned) experts, as without a profile.
pub fn split_measured(
    alloc: Vec<Vec<u32>>,
    slot_bytes: &[u64],
    measured: &MissStats,
    owned: &[Option<std::ops::Range<u32>>],
    n_expert: usize,
) -> Vec<Vec<u32>> {
    let owned_of = |il: usize| owned.get(il).cloned().flatten();
    let now: Vec<usize> = alloc.iter().map(Vec::len).collect();
    let seen: Vec<Option<LayerMisses>> = (0..alloc.len()).map(|il| measured.get(il)).collect();
    let cap: Vec<usize> = (0..alloc.len())
        .map(|il| owned_of(il).map_or(n_expert, |r| r.len()))
        .collect();
    let counts = split_by_misses(&now, slot_bytes, &seen, &cap);
    alloc
        .into_iter()
        .enumerate()
        .map(|(il, experts)| {
            if counts[il] == experts.len() {
                return experts;
            }
            let r = owned_of(il);
            (0..n_expert as u32)
                .filter(|e| r.as_ref().is_none_or(|r| r.contains(e)))
                .take(counts[il])
                .collect()
        })
        .collect()
}

/// A heap entry: the next slot's saving per byte, then the lower layer first.
struct Pick(f64, usize);

impl PartialEq for Pick {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Pick {}

impl PartialOrd for Pick {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pick {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(other.1.cmp(&self.1))
    }
}

#[cfg(test)]
mod tests;
