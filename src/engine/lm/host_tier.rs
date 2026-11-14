//! The automatic KV host tier (#352): when a context does not fit in a discrete GPU's memory,
//! the KV blocks past the device's go to pinned memory on the GPU's NUMA node instead of the
//! load failing. Sized to what the context (or the automatic pool) needs, capped by a budget
//! per node ([`node_budget`]): what the node has free with half its page cache, less a
//! margin, and never more than half the node for all of this process's tiers together
//! ([`Ledger`]). `--kv-host-tokens` sets the tier by hand (0: none).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::model::core::KV_BLOCK;

/// Host memory a node keeps free besides the tier: at least this much ...
const MARGIN_MIN: u64 = 16 << 30;
/// ... or this share of the node's memory, if more.
const MARGIN_SHARE: f64 = 0.10;
/// The share of a node's reclaimable page cache the tier counts as free (#1163). The kernel
/// gives page cache back as the tier pins, but all at once on a node the process is bound to:
/// two AC922 starts that counted 145 and 182 GiB of page cache as free froze the machine
/// while pinning 223 and 228 GiB of a 255.5 GiB node.
const CACHE_SHARE: f64 = 0.5;
/// The most of a node's memory the KV host tiers of one process (every stage, rank and
/// replica) pin together (#1163): pinned memory is never reclaimed or moved, so the other half
/// stays for the weights' page cache, ring checkpoints, the driver and the system.
const NODE_SHARE: f64 = 0.5;

/// Measured read rate of pinned host memory over the link to the CPU on an AC922 (NVLink 2.0,
/// `hostbw`, see `docs/bandwidth.md`), GB/s.
pub(crate) const NVLINK_GBS: f64 = 68.5;
/// A PCIe 4.0 x16 link's practical rate, GB/s (no PCIe machine measured).
pub(crate) const PCIE_GBS: f64 = 25.0;
/// What a GH200's GPU reads Grace's memory at over NVLink-C2C, GB/s: the nominal figure of
/// `crate::residency::Platform::NvlinkC2c` (vendor figures, unmeasured).
pub(crate) const C2C_GBS: f64 = 375.0;

/// The link a KV host tier's blocks are read over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) enum TierLink {
    /// PCIe (a discrete card).
    Pcie,
    /// NVLink 2.0 to a POWER9 CPU with ATS (an AC922).
    Nvlink,
    /// NVLink-C2C to a Grace CPU with ATS (a GH200).
    C2c,
}

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
    /// The pinned bytes the process's other loads planned on the node ([`Ledger::others`]).
    pub planned: u64,
}

impl Node {
    /// A node of `block_bytes` per host block, as `mem` reads, `planned` taken by other loads.
    pub fn new(node: u32, block_bytes: u64, mem: NodeMemory, planned: u64) -> Self {
        Self {
            node,
            block_bytes,
            free: mem.free,
            cache: mem.cache,
            total: mem.total,
            planned,
        }
    }
}

/// The most the KV host tiers of one process pin on a node of `total` bytes ([`NODE_SHARE`]).
pub(crate) fn node_cap(total: u64) -> u64 {
    (total as f64 * NODE_SHARE) as u64
}

/// The bytes a tier may pin on node `n`: its free memory with [`CACHE_SHARE`] of the page
/// cache, less the margin, and at most [`node_cap`] less what other loads planned there.
pub(crate) fn node_budget(n: &Node) -> u64 {
    let margin = MARGIN_MIN.max((n.total as f64 * MARGIN_SHARE) as u64);
    let cache = n.cache.min(n.free);
    let counted = n.free - cache + (cache as f64 * CACHE_SHARE) as u64;
    let room = counted.saturating_sub(margin);
    room.min(node_cap(n.total).saturating_sub(n.planned))
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

/// The host blocks the nodes' budgets ([`node_budget`]) hold, the tightest node's.
pub(crate) fn budget_blocks(nodes: &[Node]) -> usize {
    nodes
        .iter()
        .map(|n| (node_budget(n) / n.block_bytes.max(1)) as usize)
        .min()
        .unwrap_or(0)
}

/// The host blocks for an automatic tier: none when a context fits (`fit` device blocks against
/// `ctx_blocks`), else the rest of `want` (the automatic pool) past the `device` blocks, capped
/// by the nodes' budgets ([`node_budget`]). An error when not even a context fits that way;
/// a pool cut short says by how much (admission then waits for room, every context fits).
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
    let cap = budget_blocks(nodes);
    if cap < need {
        return Err(format!(
            "the context of {} tokens does not fit in GPU memory (room for {}), and the host \
             memory of NUMA node(s) {} ({}) has room for {} more tokens of KV cache, not the {} \
             it needs (keeping {} GiB or {:.0}% of each node free, counting {:.0}% of its page \
             cache, and pinning at most {:.0}% of a node). Lower --ctx-size, quantize the \
             cache, or free host memory",
            ctx_blocks * KV_BLOCK,
            device * KV_BLOCK,
            node_list(nodes),
            room_note(nodes),
            cap * KV_BLOCK,
            need * KV_BLOCK,
            MARGIN_MIN >> 30,
            MARGIN_SHARE * 100.0,
            CACHE_SHARE * 100.0,
            NODE_SHARE * 100.0
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
    if blocks < wanted {
        let short: u64 =
            nodes.iter().map(|n| n.block_bytes).sum::<u64>() * (wanted - blocks) as u64;
        tracing::warn!(
            host_tokens = blocks * KV_BLOCK,
            wanted_tokens = wanted * KV_BLOCK,
            "the KV host tier holds {} of the {} tokens every sequence at full context would \
             take past the GPUs ({} tokens, {:.1} GiB short; {}): every context still fits, \
             sequences wait for room past that",
            blocks * KV_BLOCK,
            wanted * KV_BLOCK,
            (wanted - blocks) * KV_BLOCK,
            short as f64 / (1u64 << 30) as f64,
            room_note(nodes)
        );
    }
    Ok(blocks)
}

/// Each node's memory the tier counted as free, how much of it is page cache, and the tier's
/// budget there ([`node_budget`]).
pub(crate) fn room_note(nodes: &[Node]) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    nodes
        .iter()
        .map(|n| {
            let others = if n.planned > 0 {
                format!(", {:.1} GiB planned by other loads", gib(n.planned))
            } else {
                String::new()
            };
            format!(
                "node {}: {:.1} GiB free of {:.1} GiB, {:.1} GiB of it page cache{others}, \
                 budget {:.1} GiB",
                crate::residency::fallback::node_label(n.node),
                gib(n.free),
                gib(n.total),
                gib(n.cache),
                gib(node_budget(n))
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn node_list(nodes: &[Node]) -> String {
    nodes
        .iter()
        .map(|n| crate::residency::fallback::node_label(n.node))
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
            // "Node", the node, the field, the value, "kB" (a node's file), or the field, the
            // value, "kB" (`/proc/meminfo`)
            let w: Vec<&str> = l.split_whitespace().collect();
            match w.as_slice() {
                [_, _, key, value, ..] | [key, value, ..] if *key == name => {
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

/// A NUMA node's memory for the tier (Linux), if known; the whole host's for
/// [`crate::residency::fallback::NO_NODE`] (host memory of no node, a GPU with no NUMA node).
pub(crate) fn node_memory(node: u32) -> Option<NodeMemory> {
    let path = if node == crate::residency::fallback::NO_NODE {
        "/proc/meminfo".to_string()
    } else {
        format!("/sys/devices/system/node/node{node}/meminfo")
    };
    parse_meminfo(&std::fs::read_to_string(path).ok()?)
}

/// Whose tier a plan is: a load (`group`, fresh per [`LmEngine::load`](super::LmEngine::load))
/// and its rank in it. `--tp-per-gpu`'s two ranks are one group: each sizes its share of the
/// group's budget (the other rank's blocks counted in [`Node::block_bytes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Owner {
    pub group: u64,
    pub rank: usize,
}

/// A group no load has used.
pub(crate) fn new_group() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// One node of the process's plan: the pinned bytes every load planned there and its cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodePlan {
    pub node: u32,
    pub planned: u64,
    pub cap: u64,
}

/// The automatic KV host tiers this process planned, per owner and node (#1163): every stage,
/// rank and replica of one serve sizes from one budget per node. Also each group's first look
/// at a node's memory, so a pair's two ranks size from the same figures (not the second from
/// what is left once the first pinned).
#[derive(Debug, Default)]
pub(crate) struct Ledger {
    /// Owner, node, bytes, the node's total.
    planned: Vec<(Owner, u32, u64, u64)>,
    seen: Vec<(u64, u32, NodeMemory)>,
}

impl Ledger {
    const fn new() -> Self {
        Self {
            planned: Vec::new(),
            seen: Vec::new(),
        }
    }

    /// The bytes groups other than `group` planned on `node`.
    pub fn others(&self, group: u64, node: u32) -> u64 {
        (self.planned.iter())
            .filter(|p| p.0.group != group && p.1 == node)
            .map(|p| p.2)
            .sum()
    }

    /// `node`'s memory as `group` first saw it: `read` on its first look.
    pub fn memory(
        &mut self,
        group: u64,
        node: u32,
        read: impl FnOnce() -> Option<NodeMemory>,
    ) -> Option<NodeMemory> {
        if let Some(s) = self.seen.iter().find(|s| s.0 == group && s.1 == node) {
            return Some(s.2);
        }
        let m = read()?;
        self.seen.push((group, node, m));
        Some(m)
    }

    /// Records `owner`'s plan of `per_node` (node, pinned bytes), an earlier plan of the
    /// owner's replaced; each node's total as its group saw it (0 if it never looked).
    pub fn plan(&mut self, owner: Owner, per_node: &[(u32, u64)]) {
        self.planned.retain(|p| p.0 != owner);
        for &(node, bytes) in per_node {
            let total = (self.seen.iter())
                .find(|s| s.0 == owner.group && s.1 == node)
                .map_or(0, |s| s.2.total);
            self.planned.push((owner, node, bytes, total));
        }
    }

    /// Drops `owner`'s plan, and its group's looks once no rank of it plans anything.
    pub fn release(&mut self, owner: Owner) {
        self.planned.retain(|p| p.0 != owner);
        if !self.planned.iter().any(|p| p.0.group == owner.group) {
            self.seen.retain(|s| s.0 != owner.group);
        }
    }

    /// The whole process's plan per node, by node.
    pub fn per_node(&self) -> Vec<NodePlan> {
        let mut out: Vec<NodePlan> = Vec::new();
        for &(_, node, bytes, total) in &self.planned {
            match out.iter_mut().find(|p| p.node == node) {
                Some(p) => {
                    p.planned += bytes;
                    p.cap = p.cap.max(node_cap(total));
                }
                None => out.push(NodePlan {
                    node,
                    planned: bytes,
                    cap: node_cap(total),
                }),
            }
        }
        out.sort_by_key(|p| p.node);
        out
    }
}

static LEDGER: Mutex<Ledger> = Mutex::new(Ledger::new());

fn ledger() -> std::sync::MutexGuard<'static, Ledger> {
    LEDGER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Node `node` for `owner`'s tier of `block_bytes` per host block: its memory as the group
/// first read it, less what other loads of this process planned there. `None` when its memory
/// cannot be read.
pub(crate) fn serve_node(owner: Owner, node: u32, block_bytes: u64) -> Option<Node> {
    let mut l = ledger();
    let mem = l.memory(owner.group, node, || node_memory(node))?;
    Some(Node::new(
        node,
        block_bytes,
        mem,
        l.others(owner.group, node),
    ))
}

/// `owner`'s planned tier, in the process's plan until dropped (with the tier).
#[derive(Debug)]
pub(crate) struct Reservation(Owner);

impl Drop for Reservation {
    fn drop(&mut self) {
        ledger().release(self.0);
    }
}

/// Puts `owner`'s tier of `per_node` (node, pinned bytes) in the process's plan and logs the
/// plan of every node.
pub(crate) fn reserve(owner: Owner, per_node: &[(u32, u64)]) -> Reservation {
    let plan = {
        let mut l = ledger();
        l.plan(owner, per_node);
        l.per_node()
    };
    tracing::info!(
        rank = owner.rank,
        "KV host tier: pinned memory planned for the whole process per NUMA node: {}",
        plan_note(&plan)
    );
    Reservation(owner)
}

/// The process's planned KV host tiers per node ([`Ledger::per_node`]).
pub(crate) fn serve_plan() -> Vec<NodePlan> {
    ledger().per_node()
}

/// `node 0: 61.7 GiB of a 127.8 GiB cap; ...`.
pub(crate) fn plan_note(plan: &[NodePlan]) -> String {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    plan.iter()
        .map(|p| {
            format!(
                "node {}: {:.1} GiB of a {:.1} GiB cap",
                crate::residency::fallback::node_label(p.node),
                gib(p.planned),
                gib(p.cap)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The startup line's speed note: what the link reads host blocks at, and what that costs.
pub(crate) fn speed_note(link: TierLink) -> String {
    match link {
        TierLink::Nvlink => format!(
            "host blocks are read over the GPU's NVLink to the CPU (~{NVLINK_GBS} GB/s measured \
             on an AC922): decode reads every host block once per token, so a long context's \
             decode slows by about (host GB per GPU) / {NVLINK_GBS} s per token, and prompts \
             reaching the tier slow down too"
        ),
        TierLink::C2c => format!(
            "host blocks are read over the GPU's NVLink-C2C to the Grace CPU (~{C2C_GBS} GB/s \
             nominal on a GH200, unmeasured): decode reads every host block once per token, so \
             a long context's decode slows by about (host GB per GPU) / {C2C_GBS} s per token"
        ),
        TierLink::Pcie => format!(
            "host blocks are read over PCIe (~{PCIE_GBS} GB/s, about 3x slower than an AC922's \
             NVLink; untested on a PCIe machine): decode reads every host block once per token, \
             so a long context's decode slows by about (host GB per GPU) / {PCIE_GBS} s per \
             token"
        ),
    }
}

#[cfg(test)]
mod tests;
