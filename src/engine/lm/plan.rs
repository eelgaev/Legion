//! The load's memory plan (#1196): one block of "memory plan" log lines, printed once a model
//! has loaded, that says where the memory went: one line for the context, one per GPU (stage
//! order) and one for host memory, plain MiB. It reads what the load decided and what the
//! driver reports; nothing here sizes anything.

use super::report::{HostUse, LoadReport, StageMemory};

#[cfg(test)]
mod tests;

/// A GPU as the driver reports it once loaded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Gpu {
    /// The ggml device name (`CUDA0`).
    pub device: String,
    /// The GPU's own name, if the driver gives one.
    pub name: String,
    /// Its ggml CUDA index, if it is a CUDA device.
    pub cuda: Option<usize>,
    pub total: u64,
    pub free: u64,
}

/// What the load knows of a stage beyond its [`StageMemory`] report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct StageFacts {
    /// The KV pools' tail lent to the expert cache at load (0 without a lend).
    pub kv_lent: u64,
    /// Tokens of every sequence's KV that are always in VRAM (the kept blocks with a lend).
    pub kv_vram_tokens: usize,
    /// Step compute buffers allocated now.
    pub compute_allocated: u64,
    /// The VRAM the load kept free of the expert cache for step compute buffers (measured at
    /// load: the kept prompt, decode and verify graphs, the MTP head's passes and the LM
    /// head's pool copy; 0 where the load measured nothing).
    pub compute_planned: u64,
}

/// One GPU's line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct GpuPlan {
    pub device: String,
    pub name: String,
    pub total: u64,
    pub weights: u64,
    pub kv_vram_tokens: usize,
    /// The KV pool's blocks always in VRAM, its host tier's staging slots and the decode
    /// states (recurrent, speculative and saved prefix-cache states kept in VRAM, and the
    /// sliding-window rings), as the KV fit counts them.
    pub kv_vram: u64,
    /// The pool's tail lent to the expert cache: VRAM the cache uses until a sequence needs
    /// those blocks.
    pub kv_lent: u64,
    /// The expert cache's slots less the lent tail they run on.
    pub expert_cache: u64,
    /// The larger of the compute buffers allocated now and the room kept for them.
    pub compute: u64,
    pub compute_allocated: u64,
    pub free: u64,
}

impl GpuPlan {
    /// What the driver reports in use.
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.free)
    }

    /// What is in use but none of the named parts: the CUDA context, ggml's pools, staging
    /// buffers and the like. Counts only the compute buffers allocated now (room kept for the
    /// rest is free memory). Never negative.
    pub fn other(&self) -> u64 {
        self.used().saturating_sub(
            self.weights + self.kv_vram + self.kv_lent + self.expert_cache + self.compute_allocated,
        )
    }
}

/// The host line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct HostPlan {
    /// The most host memory a lent KV tail's cached chats may take (0 without a lend).
    pub kv_spill_max: u64,
    /// A KV host tier's blocks, past the ones in VRAM.
    pub kv_host_tier: u64,
    /// Saved prefix-cache states in pinned host memory.
    pub saved_states: u64,
    /// Routed experts in host memory (pinned arenas, or mapped from the model file).
    pub expert_host: u64,
    /// Other weights in host memory.
    pub host_weights: u64,
    /// N-gram (and Engram) tables in RAM: the whole table, or a direct-read tier's row cache
    /// (a mapped table lives in the page cache: 0).
    pub ngram_table: u64,
    /// The prompt steps' compute room the expert cache took, which those buffers may spill
    /// into host memory: the most, measured at load.
    pub compute_spill_max: u64,
    /// MemAvailable once loaded.
    pub available: u64,
}

/// The context line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ContextPlan {
    pub ctx_size: usize,
    pub max_seqs: usize,
    pub kv_pool_tokens: usize,
    pub kv_host_tier_tokens: usize,
    /// Tokens of a lent tail whose KV host memory can keep ([`spill_tokens`]).
    pub kv_spill_tokens: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct MemoryPlan {
    pub context: ContextPlan,
    pub gpus: Vec<GpuPlan>,
    pub host: HostPlan,
}

/// Share `i` of `bytes` split evenly over `n` (the first ones take the remainder): a
/// tensor-parallel stage's figures per GPU.
pub(super) fn share(bytes: u64, n: usize, i: usize) -> u64 {
    let n = n.max(1) as u64;
    bytes / n + u64::from((i as u64) < bytes % n)
}

/// Tokens of a lent tail of `tail_tokens` (taking `lent` bytes over every GPU) that a host
/// `budget` keeps.
pub(super) fn spill_tokens(tail_tokens: usize, lent: u64, budget: u64) -> usize {
    if lent == 0 {
        return 0;
    }
    (tail_tokens as u128 * u128::from(budget.min(lent)) / u128::from(lent)) as usize
}

/// A stage's GPU lines: one per GPU under it (several with tensor parallelism, whose stage
/// figures split evenly; expert slots by their CUDA device where `experts` lists them).
pub(super) fn stage_gpus(
    s: &StageMemory,
    f: &StageFacts,
    gpus: &[Gpu],
    experts: &[(usize, u64)],
) -> Vec<GpuPlan> {
    let n = gpus.len();
    let kv_vram = (s.kv_bytes + s.kv_staging_bytes + s.states_bytes).saturating_sub(f.kv_lent);
    let on = |g: &Gpu| -> u64 {
        experts
            .iter()
            .filter(|e| Some(e.0) == g.cuda)
            .map(|e| e.1)
            .sum()
    };
    let by_cuda = n > 1 && gpus.iter().map(on).sum::<u64>() > 0;
    gpus.iter()
        .enumerate()
        .map(|(i, g)| {
            let slots = if by_cuda {
                on(g)
            } else {
                share(s.expert_slot_bytes, n, i)
            };
            let kv_lent = share(f.kv_lent, n, i);
            let compute_allocated = share(f.compute_allocated, n, i);
            GpuPlan {
                device: g.device.clone(),
                name: g.name.clone(),
                total: g.total,
                weights: share(s.weights, n, i),
                kv_vram_tokens: f.kv_vram_tokens,
                kv_vram: share(kv_vram, n, i),
                kv_lent,
                expert_cache: slots.saturating_sub(kv_lent),
                compute: compute_allocated.max(share(f.compute_planned, n, i)),
                compute_allocated,
                free: g.free,
            }
        })
        .collect()
}

/// The host line's parts the load report holds; the caller adds the rest.
pub(super) fn host_from_report(r: &LoadReport) -> HostPlan {
    let sum = |kinds: &[HostUse]| -> u64 {
        r.host
            .iter()
            .filter(|h| kinds.contains(&h.kind))
            .map(|h| h.bytes)
            .sum()
    };
    let ngram = |p: &Option<super::report::NgramPlacement>| -> u64 {
        p.as_ref().map_or(0, |p| match p.io.as_str() {
            "ram" => p.bytes,
            "direct" => p.ram_cache_bytes,
            _ => 0,
        })
    };
    HostPlan {
        kv_host_tier: sum(&[HostUse::KvHostTier]),
        saved_states: sum(&[HostUse::SavedStates]),
        expert_host: sum(&[HostUse::ExpertArena, HostUse::MappedExperts]),
        host_weights: sum(&[HostUse::SharedWeights, HostUse::HostWeights]),
        ngram_table: ngram(&r.ngram) + ngram(&r.engram),
        ..Default::default()
    }
}

fn mib(b: u64) -> u64 {
    b >> 20
}

/// Logs the plan: the context line, one line per GPU, the host line, all "memory plan".
pub(super) fn log(p: &MemoryPlan) {
    let c = &p.context;
    tracing::info!(
        part = "context",
        ctx_size = c.ctx_size,
        max_seqs = c.max_seqs,
        kv_pool_tokens = c.kv_pool_tokens,
        kv_host_tier_tokens = c.kv_host_tier_tokens,
        kv_spill_tokens = c.kv_spill_tokens,
        "memory plan"
    );
    for g in &p.gpus {
        tracing::info!(
            part = %g.device,
            name = ?g.name,
            total_mib = mib(g.total),
            weights_mib = mib(g.weights),
            kv_vram_tokens = g.kv_vram_tokens,
            kv_vram_mib = mib(g.kv_vram),
            kv_lent_mib = mib(g.kv_lent),
            expert_cache_mib = mib(g.expert_cache),
            compute_mib = mib(g.compute),
            other_mib = mib(g.other()),
            free_mib = mib(g.free),
            "memory plan"
        );
    }
    let h = &p.host;
    tracing::info!(
        part = "host",
        kv_spill_max_mib = mib(h.kv_spill_max),
        kv_host_tier_mib = mib(h.kv_host_tier),
        saved_states_mib = mib(h.saved_states),
        expert_host_mib = mib(h.expert_host),
        host_weights_mib = mib(h.host_weights),
        ngram_table_mib = mib(h.ngram_table),
        compute_spill_max_mib = mib(h.compute_spill_max),
        host_available_mib = mib(h.available),
        "memory plan"
    );
}
