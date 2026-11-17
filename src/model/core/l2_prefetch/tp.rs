//! The next hyper-connection mixer's matrices read into L2 while a rank waits for the other
//! ranks' partial sums (#1176 round 43): tensor parallelism over machines (`--tp-rank`), CUDA,
//! compute capability 12.x.
//!
//! In a 16-user verify step of Qwen3.8-Flash-Next NVFP4 over two DGX Sparks (GB10), each rank
//! waits at every exchange for the slower rank's partial: rank 0 ~8-9 ms a step at the MoE
//! sums (rank 1 holds more routed expert bytes), rank 1 ~3 ms at the attention sums. The GPU
//! sits idle meanwhile, and the first thing after each sum is a mixer whose BF16 down and up
//! matrices (6.5 MB each, read by cuBLAS GEMMs at ~80-115 GB/s) come from DRAM. The exchange
//! node of [`crate::tp_net::op::all_reduce_ahead`] carries those matrices as sources; its
//! launch calls [`launch_beside`] once this rank's partial is queued to go and before the
//! wait: it forks the prefetch side stream ([`super::side`]) off the compute stream and reads
//! the matrices into L2 there, all at once, so the mixer after the sum finds them in L2.
//! Nothing is written, so every output keeps its bits.
//!
//! The exchange is never captured (it waits on the peers' flags), so this runs in the cut
//! between stage graphs that is there anyway, and needs no join (the side stream's work reads
//! weights only; the next fork queues behind it). Forked before the exchange instead, its
//! launches delayed every send, and the peers waited that much longer (no gain).
//! `LEGION_DEBUG_TP_MIX_PF=0` turns it off, `=<n>` takes steps of `n`-[`MAX_TOKENS`] tokens
//! (default [`MIN_TOKENS`]: the verify steps over 8 tokens, whose mixers are those GEMMs);
//! `LEGION_DEBUG_TP_MIX_PF_PACE=<ns>` paces it as [`super::pace_ns`] (default all at once).

use std::sync::OnceLock;

use crate::ggml::Tensor;
#[cfg(feature = "cuda")]
use crate::ggml::ffi;

#[cfg(test)]
mod tests;

/// The fewest tokens a step takes the prefetch with by default: the steps whose mixers run as
/// GEMMs (1-8 tokens take the fused mixer kernel, not measured with it).
pub const MIN_TOKENS: i64 = 9;

/// The most tokens a step takes it with: a 16-user verify step with 3 drafts a user is 64.
/// Prompt steps (more tokens, the sums sequence-parallel or the mixers compute bound) do not.
pub const MAX_TOKENS: i64 = 64;

/// The fewest tokens of a step with the prefetch: `LEGION_DEBUG_TP_MIX_PF` (0: off), else
/// [`MIN_TOKENS`]; read once.
pub fn min_tokens() -> i64 {
    static V: OnceLock<i64> = OnceLock::new();
    *V.get_or_init(|| min_tokens_from(std::env::var("LEGION_DEBUG_TP_MIX_PF").ok().as_deref()))
}

/// [`min_tokens`] for the switch's value `v`: 0 (off) to [`MAX_TOKENS`], else the default.
pub fn min_tokens_from(v: Option<&str>) -> i64 {
    v.and_then(|s| s.trim().parse::<i64>().ok())
        .filter(|n| (0..=MAX_TOKENS).contains(n))
        .unwrap_or(MIN_TOKENS)
}

/// The prefetch's pace: `LEGION_DEBUG_TP_MIX_PF_PACE=<ns>` (up to 1 ms) as
/// [`super::pace_ns`], else 0 (all at once: the GPU waits idle); read once.
pub fn pace_ns() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| pace_from(std::env::var("LEGION_DEBUG_TP_MIX_PF_PACE").ok().as_deref()))
}

/// [`pace_ns`] for the switch's value `v`.
pub fn pace_from(v: Option<&str>) -> u32 {
    v.and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&ns| ns <= 1_000_000)
        .unwrap_or(0)
}

/// Whether a step of `n` tokens prefetches `w` at an exchange over machines: a CUDA build, a
/// GPU of compute capability 12.x, `n` from [`min_tokens`] to [`MAX_TOKENS`], `w` in that GPU's
/// memory and at least one [`super::CHUNK`].
pub fn takes(w: Tensor<'_>, n: i64) -> bool {
    takes_tokens(n, min_tokens())
        && cfg!(feature = "cuda")
        && !w.is_host()
        && !w.is_meta()
        && w.nbytes() >= super::CHUNK
        && super::super::q8_group::sm12x()
}

/// The token-count half of [`takes`], for the switch's value `min` (0: off).
pub fn takes_tokens(n: i64, min: i64) -> bool {
    min > 0 && (min..=MAX_TOKENS).contains(&n)
}

/// The (device address, bytes) spans [`launch_beside`] reads: each weight's whole
/// [`super::CHUNK`]s, those 16-byte aligned and not empty.
///
/// # Safety
/// `dst`'s sources after the first are null or weights in device memory.
#[cfg(feature = "cuda")]
unsafe fn spans(dst: &ffi::ggml_tensor) -> Vec<(usize, usize)> {
    dst.src[1..]
        .iter()
        .take_while(|w| !w.is_null())
        .filter_map(|&w| {
            // SAFETY: the caller's: a valid weight tensor.
            let (base, bytes) = unsafe { ((*w).data as usize, ffi::ggml_nbytes(w)) };
            let bytes = bytes / super::CHUNK * super::CHUNK;
            (base % 16 == 0 && bytes > 0).then_some((base, bytes))
        })
        .collect()
}

/// Reads the weights among `dst`'s sources after the first into L2 on the side stream, forked
/// from `stream` here (an exchange of [`crate::tp_net::op::all_reduce_ahead`], between its
/// send and its wait).
///
/// # Safety
/// As [`spans`]; `stream` is the stream of those weights' device.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn launch_beside(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: the caller's.
    let spans = unsafe { spans(dst) };
    if spans.is_empty() {
        return Ok(());
    }
    let s = super::side::fork(stream)?;
    for (base, bytes) in spans {
        super::launch::launch(base, bytes, s, false, pace_ns())?;
    }
    super::side::done(s)
}
