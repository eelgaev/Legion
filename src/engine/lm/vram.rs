//! What the load leaves the expert cache and the kept step graphs on each GPU (#433).
//!
//! The KV pool is sized first, next to a fixed headroom for the compute buffers, and two
//! things that came after it were not counted. The MTP head's own KV pool on the last stage:
//! with `--experts host` (Qwen3.8-Flash-Next, 2 V100s, 628,928 tokens) it took 1.2 GiB of the
//! 2 GiB headroom, and a 10K-token prompt ran out of memory on that GPU. And a cache of
//! `--expert-cache-gib`: the KV pool had taken its VRAM, so even 6.8 GiB per GPU (less than the
//! default size) failed in `cuMemCreate`.

/// What an expert cache takes of a stage's room for the KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheShare {
    /// No cache, or one sized after the KV pool to what it leaves.
    None,
    /// The automatic cache: the KV pool takes at most half.
    Half,
    /// `--expert-cache-gib`: these bytes, per GPU.
    Bytes(u64),
}

/// A stage's KV blocks of `per` bytes (with the MTP head's own pool's on the last stage) in
/// `room` bytes: (next to the cache's share, all of the room). The second tells whether a
/// context fits at all (the automatic host tier).
pub(super) fn kv_fit(room: usize, per: usize, share: CacheShare) -> (usize, usize) {
    let per = per.max(1);
    let full = room / per;
    let room = match share {
        CacheShare::None => room,
        CacheShare::Half => room / 2,
        CacheShare::Bytes(b) => room.saturating_sub(usize::try_from(b).unwrap_or(usize::MAX)),
    };
    (room / per, full)
}

/// The expert cache's bytes on `device`, `free` bytes free once the weights, the KV pool and
/// the states are in, `keep` of them for the step compute buffers and the reserve beside them:
/// `asked` (`--expert-cache-gib`, per GPU) or else all the rest. An asked size past the rest
/// does not fit, and the load fails naming the GPU and the MiB needed and free.
pub(super) fn cache_budget(
    asked: Option<u64>,
    free: usize,
    keep: usize,
    device: &str,
) -> Result<u64, String> {
    let rest = free.saturating_sub(keep) as u64;
    match asked {
        None => Ok(rest),
        Some(b) if b <= rest => Ok(b),
        Some(b) => {
            let mib = |x: u64| x.div_ceil(1 << 20);
            Err(format!(
                "--expert-cache-gib {:.2} does not fit on {device}: it needs {} MiB for the \
                 cache and {} MiB for the step compute buffers, and {} MiB are free next to the \
                 weights, the KV cache and the states. The size is per GPU: lower it (at most \
                 {:.2} here), or leave it out for the most that fits",
                b as f64 / (1u64 << 30) as f64,
                mib(b),
                mib(keep as u64),
                free >> 20,
                rest as f64 / (1u64 << 30) as f64,
            ))
        }
    }
}

/// The kept step graphs' budget on a discrete GPU without a cache sized next to measured
/// buffers (`--experts host`): what the load left free less `reserve` for ggml's pools, cuBLAS
/// and CUDA graphs, as the measured budget leaves it beside the cache (#430). At least a byte:
/// 0 means no budget.
pub(super) fn free_budget(free: usize, reserve: usize) -> usize {
    free.saturating_sub(reserve).max(1)
}

/// The sequences of a full batch the MTP head's passes must fit beside the expert cache, if
/// any: `max_seqs` when `--spec-gate-cost-model off` lets the head draft at every batch size
/// (`gate_cost_model` false), so its draft and take-in passes run at a full batch. With the
/// gate on (the default) it drafts nothing from `--spec-max-batch` sequences on, and the cache
/// sizing stays as it was. On 4 V100s (Qwen3.8-Flash-Next, 16 sequences, `--spec-k 3`, #827)
/// the head's two shared buffers grew to ~36 and ~53 MiB on the last GPU, which the cache had
/// taken: `cudaMalloc` failed every step, and at `--spec-k 3` ggml's pool could not grow and
/// the server aborted.
pub(super) fn mtp_full_batch(mtp: bool, gate_cost_model: bool, max_seqs: usize) -> Option<usize> {
    (mtp && !gate_cost_model).then_some(max_seqs.max(1))
}

/// Most rows of a step whose quantized matmuls run as matrix-vector products (ggml-cuda's mmvq
/// and Legion's expert matvec); steps of more rows take the tiled kernels.
pub(super) const MV_MAX_ROWS: usize = 8;

/// The `--spec-max-batch` gate's last batch size by default (`SpecConfig::gate_until`, #814),
/// when `--spec-gate-cost-model` is not given: [`MV_MAX_ROWS`] for `--spec mtp` without
/// `--spec-fixed` (whose depth ignores the gate) on a V100 (`volta`) layer split of 2 GPUs
/// (`stages`, no tensor parallelism) with the expert cache (`cache`), if `max_batch` is at most
/// that. Up to 8 sequences a step without drafts stays on the matvec path and drafts would
/// leave it, so the gate pays there: Qwen3.8-Flash-Next with MTP k=2 decoded 8 users at 98
/// tok/s gated and 93 ungated. Past 8 every step is tiled anyway, and the depth policy's
/// drafts pay: 12 users 71 -> 83 tok/s, 16 users 80 -> 89. Elsewhere `None`: the gate covers every batch size
/// (4 V100s: drafting ungated gained nothing, #827).
pub(super) fn mtp_gate_until_default(
    mtp: bool,
    fixed: bool,
    volta: bool,
    cache: bool,
    stages: usize,
    tp: bool,
    max_batch: usize,
) -> Option<usize> {
    (mtp && !fixed && volta && cache && stages == 2 && !tp && max_batch <= MV_MAX_ROWS)
        .then_some(MV_MAX_ROWS)
}

/// Tokens from which ggml-cuda runs a quantized `mul_mat` on a V100 as a cuBLAS F16 GEMM
/// instead of MMQ (`MMQ_DP4A_MAX_BATCH_SIZE`, `ggml_cuda_should_use_mmq`).
pub(super) const CUBLAS_MIN_ROWS: usize = 64;

/// The most rows one step asks the LM head for: every sequence's last token, and `spec_k`
/// drafts more for each sequence that drafts. Drafting sequences are all of them when nothing
/// gates the drafter (`ungated`: an MTP head or block drafter with `--spec-gate-cost-model
/// off`), otherwise fewer than `--spec-max-batch` (`max_batch`): at that many it drafts nothing.
/// Under `--spec-max-rows` (`max_rows`, 0: none; the depth policy's,
/// [`crate::spec::adaptive::rows_depth_cap`]) a step with drafts keeps within that many rows,
/// and one without them takes a row per sequence.
pub(super) fn max_head_rows(
    max_seqs: usize,
    spec_k: usize,
    ungated: bool,
    max_batch: usize,
    max_rows: usize,
) -> usize {
    let drafting = if ungated {
        max_seqs
    } else {
        max_seqs.min(max_batch.saturating_sub(1))
    };
    let rows = max_seqs + drafting * spec_k;
    if max_rows == 0 {
        rows
    } else {
        rows.min(max_seqs.max(max_rows))
    }
}

/// What ggml-cuda's pool keeps for the LM head of a step of `rows` rows, which no measured
/// compute buffer counts: on a V100 (`volta`: compute capability 7.0 / 7.2, F16 tensor cores
/// without Turing's MMA) a quantized head of `head_elems` values (`k` per row) at
/// [`CUBLAS_MIN_ROWS`] rows or more runs as a cuBLAS F16 GEMM
/// (`ggml_cuda_mul_mat_cublas_impl`): the whole head converted to F16 and the `rows x k`
/// input rounded to F16, both from the pool, whose memory stays allocated once it has grown.
/// Qwen3.8-Flash-Next's head (248,320 x 2,560 Q8_0) takes 1,212 MiB there: with
/// `--spec-gate-cost-model off --spec-k 3` a 16-sequence verify step asks for 64 rows, the
/// pool took that from the room long prompts' lends free, and a 32K prompt's step buffer
/// spilled 284 MiB into host memory on the last GPU (prefill -42%, #827). Below 64 rows ggml
/// runs MMQ, which takes only the quantized input (within the reserve): 0.
pub(super) fn head_pool_bytes(
    volta: bool,
    quantized: bool,
    head_elems: usize,
    k: usize,
    rows: usize,
) -> usize {
    if !volta || !quantized || rows < CUBLAS_MIN_ROWS {
        return 0;
    }
    2 * (head_elems + rows * k)
}

#[cfg(test)]
mod tests;
