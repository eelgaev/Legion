//! KV pools in a reserved virtual range of their GPU (`--kv-vmm`, #814): each cache tensor
//! starts on a mapping granule and is backed by physical allocations of its own
//! (`cuda::kv_vmm_type`), at most [`KV_VMM_CHUNK`] bytes each, so the memory behind a tensor's
//! last blocks can later be unmapped and handed to other users (extra expert-cache slots)
//! while the tensor, its address and every graph that reads it stay as they are.
//!
//! This is the layout only: every piece is mapped when the pool is made, so a pool reads and
//! writes exactly like one in ggml's own CUDA buffer.

#[cfg(test)]
mod tests;

/// The most bytes one physical allocation of a KV tensor covers: a lend gives memory back in
/// pieces of this size (4 MiB is ~1,600-3,300 tokens of one Qwen3.8-Flash-Next cache tensor).
pub const KV_VMM_CHUNK: usize = 4 << 20;

/// The bytes `from..to` of a cache tensor whose rows are `row` bytes each, block by block
/// ([`crate::model::core::KV_BLOCK`] rows a block), that a lend past block `keep` of a pool
/// of `n_blocks` blocks may give back: from block `keep` to the end of the last block (pad or
/// scratch rows past the blocks stay). Empty when nothing is past `keep`.
pub fn lend_range(row: usize, block: usize, keep: usize, n_blocks: usize) -> (usize, usize) {
    let from = keep.min(n_blocks) * block * row;
    (from, n_blocks * block * row)
}

/// #1196: of a cache tensor's lent pieces `(offset, bytes)` (from its start, ascending), the
/// indices of those holding a byte of a block in `held` (ascending block ids), the tensor's
/// rows `row` bytes each, `block` rows a block: what a lend keeps in host memory so the blocks
/// still in use keep their contents.
pub fn held_pieces(
    pieces: &[(usize, usize)],
    row: usize,
    block: usize,
    held: &[u32],
) -> Vec<usize> {
    let bytes = row * block;
    if bytes == 0 || held.is_empty() {
        return Vec::new();
    }
    pieces
        .iter()
        .enumerate()
        .filter(|&(_, &(off, n))| {
            if n == 0 {
                return false;
            }
            let (first, last) = (off / bytes, (off + n - 1) / bytes);
            let i = held.partition_point(|&b| (b as usize) < first);
            held.get(i).is_some_and(|&b| b as usize <= last)
        })
        .map(|(i, _)| i)
        .collect()
}

/// The blocks to keep mapped for `tokens` tokens: whole blocks of `block` tokens, at least one,
/// at most `n_blocks`.
pub fn keep_blocks(tokens: usize, block: usize, n_blocks: usize) -> usize {
    tokens.div_ceil(block).clamp(1, n_blocks.max(1))
}

/// The pieces `(offset, bytes)` that back a tensor of `bytes` bytes at `off` (both whole
/// granules of `gran`): runs of `chunk` bytes (cut down to whole granules, at least one) from
/// the tensor's start, the last one shorter. Empty for an empty tensor.
pub fn pieces(off: usize, bytes: usize, gran: usize, chunk: usize) -> Vec<(usize, usize)> {
    assert!(gran > 0 && off.is_multiple_of(gran) && bytes.is_multiple_of(gran));
    let chunk = (chunk / gran).max(1) * gran;
    (0..bytes)
        .step_by(chunk)
        .map(|at| (off + at, chunk.min(bytes - at)))
        .collect()
}
