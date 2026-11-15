//! `--kv-vmm` (#814): the stages' KV pools made again in a reserved virtual range of their GPU
//! (`crate::residency::kv_vmm`), each cache tensor backed by VRAM allocations of its own.
//! The same blocks, the same reads and writes: a layout for a later lend of a pool's unused
//! tail to the expert cache.

use super::{LoadError, cuda_index, layout};
use crate::engine::saved_states::SavedStates;
use crate::ggml::BufferType;
use crate::kv::radix::RadixCache;
use crate::kv::{BlockId, BlockPool};
use crate::model::core::{Executor, KV_BLOCK};
use crate::residency::kv_vmm::{held_pieces, lend_range};

#[cfg(test)]
mod tests;

/// Remakes `exec`'s KV pools (empty, before anything is cached) with each CUDA stage's in a
/// [`kv_vmm_buft`] buffer. Tensor parallelism (`tp`) keeps its own pools, as do stages off a
/// CUDA GPU. Whether every stage's pool is in one now (what a lend needs).
pub(super) fn apply(
    exec: &mut Executor,
    layout: &layout::Layout,
    tp: bool,
) -> Result<bool, LoadError> {
    if tp {
        tracing::warn!("--kv-vmm: tensor parallelism keeps its own KV pools; ignored");
        return Ok(false);
    }
    let bufts = layout
        .stages
        .iter()
        .map(|(d, _)| d.and_then(cuda_index).map(kv_vmm_buft).transpose())
        .collect::<Result<Vec<_>, String>>()
        .map_err(|e| LoadError::Device(format!("--kv-vmm: {e}")))?;
    if bufts.iter().all(Option::is_none) {
        tracing::warn!("--kv-vmm: no stage on a CUDA GPU; ignored");
        return Ok(false);
    }
    let all = bufts.iter().all(Option::is_some);
    let names: Vec<String> = bufts.iter().flatten().map(|b| b.name()).collect();
    exec.set_kv_bufts(bufts);
    let n_blocks = exec.kv_blocks();
    exec.resize_kv(n_blocks)
        .map_err(|e| LoadError::Device(format!("--kv-vmm: cannot remake the KV pools: {e}")))?;
    tracing::info!(
        buffer_types = ?names,
        chunk_mib = crate::residency::kv_vmm::KV_VMM_CHUNK >> 20,
        "KV pools in reserved virtual ranges (--kv-vmm)"
    );
    Ok(all)
}

/// The lend of the pools' tails to the expert cache (`--kv-vmm-keep`, #814): blocks
/// `keep..n_blocks` of every stage's pool have no memory behind them (the cache was sized
/// with it) until a sequence needs one of them. Then [`Self::grow`] maps them all (the caller
/// lends the expert cache first, which frees that VRAM), and once none of them is in use
/// again [`Self::shrink`] gives their memory back (the caller then restores the cache).
/// The block allocator hands out blocks below `keep` first
/// ([`crate::kv::BlockPool::with_fast`]), so the pool grows only when those run out.
///
/// #1196 (automatic): blocks past `keep` that cached prefixes and finished sequences
/// still hold when the tail goes back need not be evicted: [`Self::shrink`] with a
/// [`HostPlan`] keeps the pieces holding them in host memory, mapped where they were (the
/// blocks keep their addresses and contents), and the next [`Self::grow`] copies them back
/// into VRAM. The expert cache gets the same VRAM back either way.
pub(super) struct KvLend {
    /// Blocks always mapped.
    pub(super) keep: usize,
    n_blocks: usize,
    /// Whether the blocks past `keep` are mapped now.
    pub(super) grown: bool,
    /// Per stage's GPU, `(CUDA index, bytes)`: what the blocks past `keep` take when mapped;
    /// a restore of the expert cache leaves that much of its lent slots without VRAM while
    /// they are ([`Executor::restore_experts_leaving`]).
    pub(super) leave: Vec<(usize, u64)>,
    /// Per cache tensor (stage by stage, as [`Executor::kv_block_tensors`] lists them): its
    /// lent pieces.
    tails: Vec<Tail>,
    /// The most host bytes the pieces holding blocks in use may take while the tail is lent
    /// ([`default_host_budget`]; 0 where the GPUs cannot map host memory: they are evicted).
    pub(super) host_budget: usize,
    /// Host bytes the pieces of the last shrink kept (0 while grown).
    pub(super) host_held: usize,
    /// Whether the GPUs can map host memory into the pools: then a grow with no VRAM left
    /// for a piece backs it with host memory instead of failing (#1216).
    host_ok: bool,
}

/// A cache tensor's lent pieces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Tail {
    /// Its row's bytes.
    pub(super) row: usize,
    /// `(offset from its start, bytes)`, ascending: the pieces a lend gives back.
    pub(super) pieces: Vec<(usize, usize)>,
}

/// What a shrink keeps in host memory (#1196): per cache tensor, the offsets of its pieces
/// that hold a block in use, and their bytes over all tensors.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct HostPlan {
    pub(super) offsets: Vec<Vec<usize>>,
    pub(super) bytes: usize,
}

/// The host budget (#1196), chosen at load with no flag: an eighth of the host memory
/// `available` at load, at most the `lent` bytes (the pools' whole tails; all of them on an
/// AC922).
pub(super) fn default_host_budget(lent: usize, available: u64) -> usize {
    lent.min(usize::try_from(available / 8).unwrap_or(usize::MAX))
}

/// Tokens of the KV pool that keep VRAM under the bounded host expert tier (#37): the rest
/// is lent to the VRAM expert cache until the sequences hold more than this together.
pub(super) const KEEP_TIER: usize = 8192;

/// How long the bounded host tier's mapped tail waits, after the last decoding sequence on
/// it, while a cached chat still holds a block of it, before it is lent again: the next turn
/// of a long chat maps nothing (mapping it again lends and restores the whole expert cache,
/// most of a second before the first token).
pub(super) const TIER_HOLD: std::time::Duration = std::time::Duration::from_secs(300);

/// Whether the KV pool's tail is lent to the expert cache with no flag for the bounded host
/// tier (`tier`: this load has one): `--kv-vmm` / `--no-kv-vmm` not given (`asked`), the VRAM
/// expert cache (`cache`) on one stage, no tensor parallelism or second machine (`shared`),
/// no KV host tier (`host_tokens`).
pub(super) fn tier_default(
    tier: bool,
    asked: Option<bool>,
    cache: bool,
    stages: usize,
    shared: bool,
    host_tokens: usize,
) -> bool {
    tier && asked.is_none() && cache && stages == 1 && !shared && host_tokens == 0
}

/// Whether the bounded host tier's mapped tail stays mapped though no decoding sequence
/// holds a block of it: a cached chat does (`in_use`) and one decoded on it less than
/// [`TIER_HOLD`] ago (`idle`). Nothing cached on it: it is lent again at once.
pub(super) fn tier_holds(tier: bool, in_use: bool, idle: std::time::Duration) -> bool {
    tier && in_use && idle < TIER_HOLD
}

/// The blocks the running sequences will still take for the bounded host tier's lend: per
/// sequence `(need, held)`, the blocks its whole length needs (prompt and `max_tokens`) and
/// those it holds. More than the kept blocks free: the KV pool's tail is mapped when the
/// sequence is admitted, not in the middle of its reply (the expert cache is lent and
/// restored around the mapping, most of a second).
pub(super) fn tier_blocks_ahead(seqs: impl Iterator<Item = (usize, usize)>) -> usize {
    seqs.map(|(need, held)| need.saturating_sub(held)).sum()
}

/// Decode steps a run must count before it decides the next load's lend ([`tier_chats_long`]).
pub(super) const TIER_USE_STEPS: u64 = 512;

/// Whether a run's chats were long for the bounded host tier's lend: of its `steps` decode
/// steps, more than half ran with more KV blocks in use (cached chats included) than the lend
/// keeps (`long`). `None` under [`TIER_USE_STEPS`] steps: too few to tell.
pub(super) fn tier_chats_long(long: u64, steps: u64) -> Option<bool> {
    (steps >= TIER_USE_STEPS).then_some(long * 2 > steps)
}

/// The file beside the routing heat file `heat` whose presence says the last run's chats were
/// long ([`tier_chats_long`]): the next load does not lend the tail.
pub(super) fn tier_long_file(heat: &std::path::Path) -> std::path::PathBuf {
    heat.with_extension("long-chats")
}

/// Counts, for the bounded host tier, a run's decode steps and those with more KV blocks in
/// use than the lend keeps; when dropped (the engine stops), writes or removes
/// [`tier_long_file`] for the next load. With the tail mapped, whole layers of the expert
/// cache are without part of their slots (decode 8% slower than with no lend, measured), so a
/// machine whose chats are long loads without the lend, and with it again once they are not.
#[derive(Debug, Default)]
pub(super) struct TierUse {
    /// `None`: nothing counted or written (no bounded host tier, a flag decided).
    pub(super) file: Option<std::path::PathBuf>,
    /// The pool's blocks, and how many the lend keeps.
    pub(super) total: u32,
    pub(super) keep: u32,
    pub(super) steps: u64,
    pub(super) long: u64,
}

impl TierUse {
    /// A loop turn with `decoding` sequences decoding and `free` of the pool's blocks free.
    pub(super) fn note(&mut self, decoding: usize, free: u32) {
        if self.file.is_none() || decoding == 0 {
            return;
        }
        self.steps += 1;
        self.long += u64::from(self.total.saturating_sub(free) > self.keep);
    }
}

impl Drop for TierUse {
    fn drop(&mut self) {
        let Some(file) = &self.file else { return };
        let Some(long) = tier_chats_long(self.long, self.steps) else {
            return;
        };
        let done = if long {
            file.parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(file, b""))
        } else {
            match std::fs::remove_file(file) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            }
        };
        match done {
            Ok(()) => tracing::info!(
                steps = self.steps,
                long_steps = self.long,
                lend_next = !long,
                "host expert tier: whether the next load lends the KV pool's tail to the \
                 expert cache follows this run's chats (long: more KV tokens in use than \
                 the lend keeps)"
            ),
            Err(e) => tracing::warn!(error = %e, "could not note this run's chat lengths"),
        }
    }
}

/// The pieces of `tails` that hold a block of `held` (ascending ids): what a shrink keeps.
pub(super) fn host_plan(tails: &[Tail], held: &[u32]) -> HostPlan {
    let mut bytes = 0;
    let offsets = tails
        .iter()
        .map(|t| {
            held_pieces(&t.pieces, t.row, KV_BLOCK, held)
                .into_iter()
                .map(|i| {
                    bytes += t.pieces[i].1;
                    t.pieces[i].0
                })
                .collect()
        })
        .collect();
    HostPlan { offsets, bytes }
}

/// The host copy for a shrink within `budget` bytes (#1196): plans it (`plan`) and, while it
/// takes more, evicts a holder of a block past the kept ones (`evict`, least recently used
/// first) and plans again. `None` when nothing is left to evict and it still takes more (the
/// shrink then evicts every holder, as without host memory).
pub(super) fn fit_host<S>(
    budget: usize,
    s: &mut S,
    mut plan: impl FnMut(&mut S) -> HostPlan,
    mut evict: impl FnMut(&mut S) -> bool,
) -> Option<HostPlan> {
    loop {
        let p = plan(s);
        if p.bytes <= budget {
            return Some(p);
        }
        if !evict(s) {
            return None;
        }
    }
}

/// One eviction for [`fit_host`]: drops a holder of a block from `first` on. The least
/// recently used radix entry holding one (or under one); else every saved state whose tail
/// block is one (a chat resumes from it only with its full blocks, which go first); else a
/// warm sequence (`warm`: the least recently finished one holding such a block; false when
/// none does). False when nothing held such a block.
pub(super) fn evict_tail_holder(
    pool: &mut BlockPool,
    radix: Option<&mut RadixCache>,
    saved: &mut SavedStates,
    first: u32,
    warm: impl FnOnce(&mut BlockPool) -> bool,
) -> bool {
    if radix.is_some_and(|r| r.evict_lru_from(pool, first)) {
        return true;
    }
    let tails = saved.drop_tails_from(BlockId(first));
    for &b in &tails {
        pool.decref(b);
    }
    !tails.is_empty() || warm(pool)
}

impl KvLend {
    /// Gives back the memory of every stage's blocks past `keep` (of `n_blocks`) at load,
    /// before the expert cache is sized. Per stage: the bytes given back. `None` when nothing
    /// would be (`keep` covers the pool, or the pools are not `--kv-vmm` ones).
    /// The host budget is [`default_host_budget`] (0 where the GPUs cannot map host memory
    /// into the pools).
    pub(super) fn at_load(
        exec: &Executor,
        layout: &layout::Layout,
        keep: usize,
        n_blocks: usize,
    ) -> Result<Option<(Self, Vec<usize>)>, LoadError> {
        if keep >= n_blocks {
            return Ok(None);
        }
        let mut lend = Self {
            keep,
            n_blocks,
            grown: true,
            leave: Vec::new(),
            tails: Vec::new(),
            host_budget: 0,
            host_held: 0,
            host_ok: false,
        };
        let (bytes, tails) = lend
            .unmap(exec, None)
            .map_err(|e| LoadError::Device(format!("--kv-vmm-keep: {e}")))?;
        if bytes.iter().sum::<usize>() == 0 {
            return Ok(None);
        }
        lend.tails = tails;
        let host_ok = exec.kv_block_tensors().into_iter().flatten().all(host_ok);
        let available = crate::residency::lookup_table::MemInfo::read().available;
        let host_budget = default_host_budget(bytes.iter().sum(), available);
        lend.host_budget = if host_ok { host_budget } else { 0 };
        lend.host_ok = host_ok;
        if host_budget > 0 && !host_ok {
            tracing::warn!(
                "the GPUs cannot map host memory into the KV pools; cached chats past the kept \
                 tokens are evicted when the tail goes back to the expert cache"
            );
        }
        lend.leave = layout
            .stages
            .iter()
            .zip(&bytes)
            .filter_map(|((d, _), &b)| Some((d.and_then(cuda_index)?, b as u64)))
            .filter(|&(_, b)| b > 0)
            .collect();
        tracing::info!(
            pool_tokens = n_blocks * KV_BLOCK,
            vram_tokens = keep * KV_BLOCK,
            lent_mib = ?bytes.iter().map(|b| b >> 20).collect::<Vec<_>>(),
            host_mib = lend.host_budget >> 20,
            "KV cache: each GPU's pool keeps its first vram_tokens in VRAM; the rest (lent_mib \
             a GPU) serves the expert cache until a sequence needs it; when it goes back, the \
             chats cached on it move to host memory (up to host_mib in all) and come back to \
             VRAM when resumed"
        );
        Ok(Some((lend, bytes)))
    }

    /// The host copy a shrink would keep for blocks `held` (ascending, all past `keep`).
    pub(super) fn host_plan(&self, held: &[u32]) -> HostPlan {
        host_plan(&self.tails, held)
    }

    /// Unmaps every stage's blocks past `keep`, keeping `host`'s pieces in host memory. Per
    /// stage: the VRAM bytes given back; per tensor: its lent pieces.
    fn unmap(
        &mut self,
        exec: &Executor,
        host: Option<&HostPlan>,
    ) -> Result<(Vec<usize>, Vec<Tail>), String> {
        // Even if a piece fails: the next grow maps whatever is unmapped.
        self.grown = false;
        let mut tails = Vec::new();
        let mut k = 0;
        let mut bytes = Vec::new();
        for tensors in exec.kv_block_tensors() {
            let mut sum = 0;
            for t in tensors {
                let row = t.nb()[1];
                let (from, to) = lend_range(row, KV_BLOCK, self.keep, self.n_blocks);
                let keep = host
                    .and_then(|h| h.offsets.get(k))
                    .map_or(&[][..], |o| &o[..]);
                let pieces = unmap(t, from, to, keep, self.host_budget)?;
                sum += pieces.iter().map(|p| p.1).sum::<usize>();
                tails.push(Tail { row, pieces });
                k += 1;
            }
            bytes.push(sum);
        }
        self.grown = false;
        Ok((bytes, tails))
    }

    /// Maps the blocks past `keep` again: zeroed, or with the bytes a shrink kept in host
    /// memory. Returns the bytes mapped into VRAM, of them those copied back from host
    /// memory, and the bytes left in (or mapped from) host memory because the GPU had no
    /// VRAM left for them (#1216: where the GPUs can map host memory, a piece goes there
    /// instead of failing the request; the blocks keep their address either way). No step
    /// may have been built over the blocks without memory yet. `reserve`: VRAM a piece must
    /// leave free to be mapped there rather than in host memory (a lent cache's long prompt
    /// step buffers); 0 maps as before wherever VRAM is not out.
    pub(super) fn grow(
        &mut self,
        exec: &Executor,
        reserve: usize,
    ) -> Result<(usize, usize, usize), String> {
        let (mut bytes, mut restored, mut host) = (0, 0, 0);
        let fallback = self.host_ok.then_some(reserve);
        for t in exec.kv_block_tensors().into_iter().flatten() {
            let (b, r, h) = remap(t, fallback)?;
            bytes += b;
            restored += r;
            host += h;
        }
        self.grown = true;
        self.host_held = 0;
        Ok((bytes, restored, host))
    }

    /// Unmaps the blocks past `keep` again once no step runs: none of them may be in use but
    /// those in `host`'s pieces, which stay in host memory (#1196). Returns the VRAM bytes
    /// given back.
    pub(super) fn shrink(
        &mut self,
        exec: &mut Executor,
        host: Option<&HostPlan>,
    ) -> Result<usize, String> {
        exec.quiesce().map_err(|e| e.to_string())?;
        let (bytes, _) = self.unmap(exec, host)?;
        self.host_held = host.map_or(0, |h| h.bytes);
        Ok(bytes.iter().sum())
    }
}

#[cfg(feature = "cuda")]
fn unmap(
    t: crate::ggml::Tensor<'_>,
    from: usize,
    to: usize,
    host: &[usize],
    budget: usize,
) -> Result<Vec<(usize, usize)>, String> {
    crate::residency::cuda::kv_vmm_unmap(t, from, to, host, budget)
}

#[cfg(not(feature = "cuda"))]
fn unmap(
    _: crate::ggml::Tensor<'_>,
    _: usize,
    _: usize,
    _: &[usize],
    _: usize,
) -> Result<Vec<(usize, usize)>, String> {
    Ok(Vec::new())
}

#[cfg(feature = "cuda")]
fn remap(
    t: crate::ggml::Tensor<'_>,
    host_below: Option<usize>,
) -> Result<(usize, usize, usize), String> {
    let r = crate::residency::cuda::kv_vmm_remap(t, host_below)?;
    Ok((r.vram, r.restored, r.host))
}

#[cfg(not(feature = "cuda"))]
fn remap(_: crate::ggml::Tensor<'_>, _: Option<usize>) -> Result<(usize, usize, usize), String> {
    Ok((0, 0, 0))
}

#[cfg(feature = "cuda")]
fn host_ok(t: crate::ggml::Tensor<'_>) -> bool {
    crate::residency::cuda::kv_vmm_host_ok(t)
}

#[cfg(not(feature = "cuda"))]
fn host_ok(_: crate::ggml::Tensor<'_>) -> bool {
    false
}

#[cfg(feature = "cuda")]
fn kv_vmm_buft(device: usize) -> Result<BufferType, String> {
    crate::residency::cuda::kv_vmm_type(device)
}

#[cfg(not(feature = "cuda"))]
fn kv_vmm_buft(_: usize) -> Result<BufferType, String> {
    Err("needs CUDA".into())
}
