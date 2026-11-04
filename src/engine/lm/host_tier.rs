//! The automatic KV host tier (#352): when a context does not fit in a discrete GPU's memory,
//! the KV blocks past the device's go to pinned memory on the GPU's NUMA node instead of the
//! load failing. Sized to what the context (or the automatic pool) needs, capped by what the
//! nodes have free with a margin. `--kv-host-tokens` sets the tier by hand (0: none).

use crate::model::core::KV_BLOCK;

/// Host memory a node keeps free besides the tier: at least this much ...
const MARGIN_MIN: u64 = 16 << 30;
/// ... or this share of the node's memory, if more.
const MARGIN_SHARE: f64 = 0.10;

/// Measured read rate of pinned host memory over the link to the CPU on an AC922 (NVLink 2.0,
/// `hostbw`, see `docs/bandwidth.md`), GB/s.
pub(crate) const NVLINK_GBS: f64 = 68.5;
/// A PCIe 4.0 x16 link's practical rate, GB/s (no PCIe machine measured).
pub(crate) const PCIE_GBS: f64 = 25.0;

/// One NUMA node the tier would use: the bytes each host block takes there (summed over the
/// stages whose GPU is on it) and the node's memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Node {
    pub node: u32,
    pub block_bytes: u64,
    pub free: u64,
    /// The part of `free` that is reclaimable page cache.
    pub cache: u64,
    pub total: u64,
}

/// A NUMA node's memory as the tier sees it, from its meminfo (bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeMemory {
    /// `MemFree` plus the reclaimable page cache, at most `total`.
    pub free: u64,
    /// The part of `free` that is reclaimable page cache.
    pub cache: u64,
    pub total: u64,
}

/// The host blocks for an automatic tier: none when a context fits (`fit` device blocks against
/// `ctx_blocks`), else the rest of `want` (the automatic pool) past the `device` blocks, capped
/// by the nodes' free memory less the margin. An error when not even a context fits that way.
pub(crate) fn auto_host_blocks(
    fit: usize,
    ctx_blocks: usize,
    want: usize,
    device: usize,
    nodes: &[Node],
) -> Result<usize, String> {
    if fit >= ctx_blocks {
        return Ok(0);
    }
    let need = ctx_blocks.saturating_sub(device);
    let wanted = want.max(ctx_blocks).saturating_sub(device);
    let cap = nodes
        .iter()
        .map(|n| {
            let margin = MARGIN_MIN.max((n.total as f64 * MARGIN_SHARE) as u64);
            (n.free.saturating_sub(margin) / n.block_bytes.max(1)) as usize
        })
        .min()
        .unwrap_or(0);
    if cap < need {
        return Err(format!(
            "the context of {} tokens does not fit in GPU memory (room for {}), and the host \
             memory of NUMA node(s) {} ({}) has room for {} more tokens of KV cache, not the {} \
             it needs (keeping {} GiB or {:.0}% of each node free). Lower --ctx-size, quantize \
             the cache, or free host memory",
            ctx_blocks * KV_BLOCK,
            device * KV_BLOCK,
            node_list(nodes),
            room_note(nodes),
            cap * KV_BLOCK,
            need * KV_BLOCK,
            MARGIN_MIN >> 30,
            MARGIN_SHARE * 100.0
        ));
    }
    let blocks = wanted.min(cap);
    if blocks > 0 {
        tracing::info!(
            host_tokens = blocks * KV_BLOCK,
            "KV host tier sized on NUMA node(s) {}: {}",
            node_list(nodes),
            room_note(nodes)
        );
    }
    Ok(blocks)
}

/// Each node's memory the tier counted as free, and how much of it is page cache.
pub(crate) fn room_note(nodes: &[Node]) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    nodes
        .iter()
        .map(|n| {
            format!(
                "node {}: {:.1} GiB free of {:.1} GiB, {:.1} GiB of it page cache",
                n.node,
                gib(n.free),
                gib(n.total),
                gib(n.cache)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn node_list(nodes: &[Node]) -> String {
    nodes
        .iter()
        .map(|n| n.node.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `Node 3 MemTotal: 123 kB` lines of a node's meminfo. Free for the tier is `MemFree` plus the
/// reclaimable page cache, which the kernel gives back for the tier's pinned memory on that node
/// (the per-node meminfo has no `MemAvailable`): the file pages (`Active(file)` +
/// `Inactive(file)`) less the mapped ones (`Mapped`) and the ones not yet written back (`Dirty`,
/// `Writeback`), floored at 0, the sum never more than `MemTotal`.
pub(crate) fn parse_meminfo(text: &str) -> Option<NodeMemory> {
    let field = |name: &str| {
        text.lines().find_map(|l| {
            // "Node", the node, the field, the value, "kB"
            let w: Vec<&str> = l.split_whitespace().collect();
            match w.as_slice() {
                [_, _, key, value, ..] if *key == name => {
                    value.parse::<u64>().ok().map(|kb| kb * 1024)
                }
                _ => None,
            }
        })
    };
    let opt = |name: &str| field(name).unwrap_or(0);
    let mem_free = field("MemFree:")?;
    let total = field("MemTotal:")?;
    let cache = (opt("Active(file):") + opt("Inactive(file):"))
        .saturating_sub(opt("Mapped:") + opt("Dirty:") + opt("Writeback:"));
    let free = mem_free.saturating_add(cache).min(total);
    Some(NodeMemory {
        free,
        cache: free.saturating_sub(mem_free),
        total,
    })
}

/// A NUMA node's memory for the tier (Linux), if known.
pub(crate) fn node_memory(node: u32) -> Option<NodeMemory> {
    let text =
        std::fs::read_to_string(format!("/sys/devices/system/node/node{node}/meminfo")).ok()?;
    parse_meminfo(&text)
}

/// The startup line's speed note: what the link reads host blocks at, and what that costs.
pub(crate) fn speed_note(nvlink: bool) -> String {
    if nvlink {
        format!(
            "host blocks are read over the GPU's NVLink to the CPU (~{NVLINK_GBS} GB/s measured \
             on an AC922): decode reads every host block once per token, so a long context's \
             decode slows by about (host GB per GPU) / {NVLINK_GBS} s per token, and prompts \
             reaching the tier slow down too"
        )
    } else {
        format!(
            "host blocks are read over PCIe (~{PCIE_GBS} GB/s, about 3x slower than an AC922's \
             NVLink; untested on a PCIe machine): decode reads every host block once per token, \
             so a long context's decode slows by about (host GB per GPU) / {PCIE_GBS} s per \
             token"
        )
    }
}

#[cfg(test)]
mod tests;
