//! `--kv-vmm` (#814): the stages' KV pools made again in a reserved virtual range of their GPU
//! (`crate::residency::kv_vmm`), each cache tensor backed by VRAM allocations of its own.
//! The same blocks, the same reads and writes: a layout for a later lend of a pool's unused
//! tail to the expert cache.

use super::{LoadError, cuda_index, layout};
use crate::ggml::BufferType;
use crate::model::core::{Executor, KV_BLOCK};
use crate::residency::kv_vmm::lend_range;

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
}

impl KvLend {
    /// Gives back the memory of every stage's blocks past `keep` (of `n_blocks`) at load,
    /// before the expert cache is sized. Per stage: the bytes given back. `None` when nothing
    /// would be (`keep` covers the pool, or the pools are not `--kv-vmm` ones).
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
        };
        let bytes = lend
            .unmap(exec)
            .map_err(|e| LoadError::Device(format!("--kv-vmm-keep: {e}")))?;
        if bytes.iter().sum::<usize>() == 0 {
            return Ok(None);
        }
        lend.leave = layout
            .stages
            .iter()
            .zip(&bytes)
            .filter_map(|((d, _), &b)| Some((d.and_then(cuda_index)?, b as u64)))
            .filter(|&(_, b)| b > 0)
            .collect();
        tracing::info!(
            keep_tokens = keep * KV_BLOCK,
            pool_tokens = n_blocks * KV_BLOCK,
            lent_mib = ?bytes.iter().map(|b| b >> 20).collect::<Vec<_>>(),
            "KV pools' tails lent to the expert cache until a sequence needs them (--kv-vmm-keep)"
        );
        Ok(Some((lend, bytes)))
    }

    /// Unmaps every stage's blocks past `keep`. Per stage: the bytes given back.
    fn unmap(&mut self, exec: &Executor) -> Result<Vec<usize>, String> {
        // Even if a piece fails: the next grow maps whatever is unmapped.
        self.grown = false;
        let bytes = exec
            .kv_block_tensors()
            .into_iter()
            .map(|tensors| {
                tensors.into_iter().try_fold(0usize, |sum, t| {
                    let row = t.nb()[1];
                    let (from, to) = lend_range(row, KV_BLOCK, self.keep, self.n_blocks);
                    Ok::<_, String>(sum + unmap(t, from, to)?)
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.grown = false;
        Ok(bytes)
    }

    /// Maps the blocks past `keep` again, zeroed. Returns the bytes mapped. No step may have
    /// been built over those blocks yet.
    pub(super) fn grow(&mut self, exec: &Executor) -> Result<usize, String> {
        let mut bytes = 0;
        for t in exec.kv_block_tensors().into_iter().flatten() {
            bytes += remap(t)?;
        }
        self.grown = true;
        Ok(bytes)
    }

    /// Unmaps the blocks past `keep` again once no step runs: none of them may be in use.
    /// Returns the bytes given back.
    pub(super) fn shrink(&mut self, exec: &mut Executor) -> Result<usize, String> {
        exec.quiesce().map_err(|e| e.to_string())?;
        Ok(self.unmap(exec)?.iter().sum())
    }
}

#[cfg(feature = "cuda")]
fn unmap(t: crate::ggml::Tensor<'_>, from: usize, to: usize) -> Result<usize, String> {
    crate::residency::cuda::kv_vmm_unmap(t, from, to)
}

#[cfg(not(feature = "cuda"))]
fn unmap(_: crate::ggml::Tensor<'_>, _: usize, _: usize) -> Result<usize, String> {
    Ok(0)
}

#[cfg(feature = "cuda")]
fn remap(t: crate::ggml::Tensor<'_>) -> Result<usize, String> {
    crate::residency::cuda::kv_vmm_remap(t)
}

#[cfg(not(feature = "cuda"))]
fn remap(_: crate::ggml::Tensor<'_>) -> Result<usize, String> {
    Ok(0)
}

#[cfg(feature = "cuda")]
fn kv_vmm_buft(device: usize) -> Result<BufferType, String> {
    crate::residency::cuda::kv_vmm_type(device)
}

#[cfg(not(feature = "cuda"))]
fn kv_vmm_buft(_: usize) -> Result<BufferType, String> {
    Err("needs CUDA".into())
}
