//! The first bytes of the weights a rank reads next, read into L2 while it waits for the other
//! ranks' partial sums (tensor parallelism over machines, `--tp-rank`; CUDA, compute
//! capability 12.x).
//!
//! In Gemma 4 31B's verify steps over two DGX Sparks (GB10) every layer sums twice over the
//! network: the attention output and the FFN output. At 16 users (128 rows a step) each sum
//! keeps the GPU idle ~0.14 ms waiting for the wire (16.5 ms of a 95 ms step, 120 sums) and the
//! matmul after it then streams its weights from DRAM (the step's matmuls are bound by the
//! DRAM: 8.7 GB of NVFP4 per rank). The exchange node of [`crate::tp_net::all_reduce_ahead`]
//! carries the next matmul's weights (after the attention sum the FFN's gate and up, after the
//! FFN sum the next layer's q, k and v) and the bytes to read of each; its launch calls
//! [`launch_beside`] once this rank's partial is on its way and before the wait: that forks
//! the prefetch side stream ([`super::side`]) off the compute stream and reads those bytes
//! into L2 there, so the wait does the matmul's first DRAM reads. Nothing is written, so every
//! output keeps its bits.
//!
//! The exchange is never captured (it waits on the peers' flags): this runs in the cut between
//! stage graphs that is there anyway and needs no join (the side stream reads weights only;
//! the next fork queues behind it). Forked before this rank's send, its launches would delay
//! the send (#1176 round 43, where it first ran: Qwen3.8-Flash-Next's mixers, a tie there:
//! that model's critical rank barely waits).
//!
//! `LEGION_DEBUG_TP_AR_PF=<MiB>` sets the bytes read ahead at each sum (0: off; default
//! [`DEFAULT_MIB`]), `LEGION_DEBUG_TP_AR_PF_PACE=<ns>` paces it as [`super::pace_ns`] (default
//! all at once: the compute stream only waits meanwhile).

use std::sync::OnceLock;

use crate::ggml::Tensor;
#[cfg(feature = "cuda")]
use crate::ggml::{external_params, ffi};

#[cfg(test)]
mod tests;

/// MiB read ahead at each sum by default, a little over half the GB10's 24 MiB L2: the rest
/// holds the sum's and the norms' tensors (~11 MB at 128 rows) until the matmul reads its rows.
pub const DEFAULT_MIB: usize = 12;

/// The fewest tokens a step takes it with: more than the 8 of one user's verify step, whose
/// matmuls are the one-pass mv kernels (the sums' waits are short there, ~0.04 ms, and read
/// ahead the step got 1-2 ms slower on two Sparks).
pub const MIN_TOKENS: i64 = 9;

/// The most tokens a step takes it with: verify steps (16 users with 7 drafts each: 128).
/// Prompt steps are compute bound and Gemma's long ones run in halves.
pub const MAX_TOKENS: i64 = 256;

/// The most weights one exchange reads ahead.
pub const MAX_WEIGHTS: usize = 4;

/// Bytes read ahead at each sum: `LEGION_DEBUG_TP_AR_PF` in MiB (0: off, at most 20), else
/// [`DEFAULT_MIB`]; read once.
pub fn budget() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| budget_from(std::env::var("LEGION_DEBUG_TP_AR_PF").ok().as_deref()))
}

/// [`budget`] for the switch's value `v`.
pub fn budget_from(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&m| m <= 20)
        .unwrap_or(DEFAULT_MIB)
        << 20
}

/// The prefetch's pace: `LEGION_DEBUG_TP_AR_PF_PACE=<ns>` (up to 1 ms) as [`super::pace_ns`],
/// else 0 (all at once); read once.
pub fn pace_ns() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| pace_from(std::env::var("LEGION_DEBUG_TP_AR_PF_PACE").ok().as_deref()))
}

/// [`pace_ns`] for the switch's value `v`.
pub fn pace_from(v: Option<&str>) -> u32 {
    v.and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&ns| ns <= 1_000_000)
        .unwrap_or(0)
}

/// Whether a step of `n` tokens reads ahead at its sums: a CUDA build, a GPU of compute
/// capability 12.x, `n` from [`MIN_TOKENS`] to [`MAX_TOKENS`] and a [`budget`].
pub fn takes(n: i64) -> bool {
    takes_tokens(n, budget()) && cfg!(feature = "cuda") && super::super::q8_group::sm12x()
}

/// The token-count half of [`takes`], for a budget of `bytes` (0: off).
pub fn takes_tokens(n: i64, bytes: usize) -> bool {
    bytes >= super::CHUNK && (MIN_TOKENS..=MAX_TOKENS).contains(&n)
}

/// How many bytes of each weight of `sizes` (bytes, in the order the GPU reads them) a budget
/// of `bytes` reads ahead: whole [`super::CHUNK`]s from each weight's start. `together`: the
/// next kernel reads all of them at once (a GLU's gate and up, one tile of each per CTA), so
/// the budget is shared evenly (what a smaller one leaves goes to the others); else in order,
/// each weight whole before the next.
pub fn split(sizes: &[usize], bytes: usize, together: bool) -> Vec<usize> {
    let whole = |b: usize| b / super::CHUNK * super::CHUNK;
    let mut out = vec![0; sizes.len()];
    let mut left = whole(bytes);
    if together {
        // Every round gives each open weight an even share (at least a chunk) or what is left
        // of it, so each round fills a weight or spends a chunk.
        let mut open: Vec<usize> = (0..sizes.len()).filter(|&i| whole(sizes[i]) > 0).collect();
        while left >= super::CHUNK && !open.is_empty() {
            let each = whole(left / open.len()).max(super::CHUNK);
            for &i in &open {
                let take = each.min(whole(sizes[i]) - out[i]).min(left);
                out[i] += take;
                left -= take;
            }
            open.retain(|&i| out[i] < whole(sizes[i]));
        }
    } else {
        for (o, &s) in out.iter_mut().zip(sizes) {
            *o = whole(s).min(left);
            left -= *o;
        }
    }
    out
}

/// The (weight, bytes) pairs an exchange of a step of `n` tokens reads ahead ([`split`] of
/// [`budget`] over `ws`), or none where [`takes`] says no or no weight qualifies (in the GPU's
/// memory, 16-byte aligned, at least a chunk).
pub fn plan<'a>(ws: &[Tensor<'a>], n: i64, together: bool) -> Vec<(Tensor<'a>, usize)> {
    if !takes(n) {
        return Vec::new();
    }
    let ws: Vec<Tensor<'a>> = ws
        .iter()
        .copied()
        .filter(|w| !w.is_host() && !w.is_meta() && w.nbytes() >= super::CHUNK)
        .take(MAX_WEIGHTS)
        .collect();
    let sizes: Vec<usize> = ws.iter().map(|w| w.nbytes()).collect();
    ws.into_iter()
        .zip(split(&sizes, budget(), together))
        .filter(|&(_, b)| b > 0)
        .collect()
}

/// The (device address, bytes) spans [`launch_beside`] reads: the exchange node's weights
/// (its sources from `first`, as many as its params say) and their bytes (the params after).
///
/// # Safety
/// `dst` is a node of [`crate::tp_net::all_reduce_ahead`].
#[cfg(feature = "cuda")]
unsafe fn spans(dst: &ffi::ggml_tensor, first: usize) -> Vec<(usize, usize)> {
    let p = external_params(dst);
    let count = p[2].max(0) as usize;
    (0..count.min(MAX_WEIGHTS))
        .filter_map(|i| {
            let w = dst.src[first + i];
            if w.is_null() {
                return None;
            }
            // SAFETY: the caller's: a valid weight tensor.
            let base = unsafe { (*w).data as usize };
            let bytes = p[3 + i].max(0) as usize * super::CHUNK;
            (base % 16 == 0 && bytes > 0).then_some((base, bytes))
        })
        .collect()
}

/// Reads the bytes [`spans`] names into L2 on the side stream, forked from `stream` here (an
/// exchange of [`crate::tp_net::all_reduce_ahead`], between its send and its wait).
///
/// # Safety
/// As [`spans`]; `stream` is the stream of those weights' device.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn launch_beside(
    dst: &ffi::ggml_tensor,
    first: usize,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: the caller's.
    let spans = unsafe { spans(dst, first) };
    if spans.is_empty() {
        return Ok(());
    }
    let s = super::side::fork(stream)?;
    for (base, bytes) in spans {
        super::launch::launch(base, bytes, s, false, pace_ns())?;
    }
    super::side::done(s)
}
