//! A small compute buffer kept across an expert-cache lend (#491). The restore after a long
//! prompt drops the prompt's graphs, and the first verify step after it built its graphs into
//! new compute buffers. Their `cudaMalloc` queued in the driver behind the restore thread's
//! `cudaFree` of the prompt's 1-2 GiB buffers (#466/#489), which now and then takes 80-120 ms:
//! the first step after the restore waited for it. Each stage that verifies, and the MTP head,
//! now keeps a buffer through the lend that the first graph built after the restore allocates
//! from (a stage's first new verify graph; the MTP head's take-in graphs, which share theirs),
//! so that step allocates nothing.
//!
//! The spare is a kept verify graph's buffer when one is big enough (no allocation at all; after
//! the first lend it is usually the last spare, which the graphs pass on as they evict each
//! other), else a new one of [`SPARE_BYTES`] allocated as the lend starts, while the prompt is
//! about to take seconds anyway.

#[cfg(test)]
mod tests;

/// The spare's size when one is allocated. A verify step of Qwen3.8-Flash-Next (two V100s) took
/// 6-12 MiB per stage for one sequence, 31-48 MiB for a full batch of 16; a graph that needs
/// more grows the buffer as before (a new allocation, as without the spare).
pub(super) const SPARE_BYTES: usize = 64 << 20;

/// Kept buffers above this are not kept as the spare: such a buffer is a short prompt's, and the
/// long prompt needs the memory more.
pub(super) const SPARE_MAX_BYTES: usize = 4 * SPARE_BYTES;

/// Which of the candidate buffers (their sizes) to keep as the spare: the biggest within
/// [`SPARE_BYTES`]..=[`SPARE_MAX_BYTES`]; `None`: none fits, allocate one.
pub(super) fn pick(sizes: &[usize]) -> Option<usize> {
    sizes
        .iter()
        .enumerate()
        .filter(|&(_, &b)| (SPARE_BYTES..=SPARE_MAX_BYTES).contains(&b))
        .max_by_key(|&(i, &b)| (b, std::cmp::Reverse(i)))
        .map(|(i, _)| i)
}
