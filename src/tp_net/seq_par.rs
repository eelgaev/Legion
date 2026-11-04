//! Sequence parallelism for prompt steps (`--tp-seq-par`, #751). Under `--tp-rank` every rank
//! runs the work between the sums on every token: the attention and FFN have their heads and
//! rows split, but what is per token (Qwen3.8-Flash-Next's hyper-connection mixers and
//! combines) is mirrored. In a big step a rank can do that per-token work on its share of the
//! tokens instead (Megatron's sequence parallelism): each sum of partial outputs becomes a
//! [`reduce_scatter`] (every rank gets the sum of its own share of the columns only) and the
//! block input an [`all_gather`] of every rank's share before the split block. The bytes
//! sent are the same as the all-reduce's; the per-token work is divided by the ranks.
//!
//! Rank `r` of `n` owns the columns (tokens) [`share`]`(cols, r, n)`. Both ops add or copy in
//! rank order, as the all-reduce does, so every rank gets the same bits; with `--tp-wire bf16`
//! the gathered columns are rounded to BF16, this rank's own included.

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

#[cfg(test)]
mod tests;

/// Rank `rank`'s share of `cols` columns among `n` ranks: `(first, count)`, contiguous, in
/// rank order, the counts within one of each other.
pub fn share(cols: usize, rank: usize, n: usize) -> (usize, usize) {
    let a = cols * rank / n;
    (a, cols * (rank + 1) / n - a)
}

/// The communicator `comm`'s rank and rank count when a step of `n` tokens runs its per-token
/// work on a share of the tokens: `--tp-seq-par` is on and `n` reaches it (and every rank gets
/// at least one token).
pub fn active(comm: i32, n: i64) -> Option<(usize, usize)> {
    let c = super::comm(comm)?;
    let min = c.seq_par();
    let n = usize::try_from(n).ok()?;
    (min > 0 && n >= min && n >= c.n_ranks()).then(|| (c.rank(), c.n_ranks()))
}

static REDUCE_SCATTER: ExternalOp = ExternalOp {
    name: "tp_reduce_scatter",
    cpu: cpu_reduce_scatter,
    #[cfg(feature = "cuda")]
    cuda: Some(super::op::cuda::launch_reduce_scatter),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ALL_GATHER: ExternalOp = ExternalOp {
    name: "tp_all_gather",
    cpu: cpu_all_gather,
    #[cfg(feature = "cuda")]
    cuda: Some(super::op::cuda::launch_all_gather),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The sum over the ranks of `x` (F32 `[row, cols]`, this rank's partial of every column),
/// for this rank's [`share`] of the columns only: `[row, share]`.
pub fn reduce_scatter<'a>(ctx: &'a Context, comm: i32, x: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "tp reduce-scatter: F32 only");
    let c = super::comm(comm).expect("tp reduce-scatter: the communicator");
    let [row, cols, a, b] = x.ne();
    assert_eq!((a, b), (1, 1), "tp reduce-scatter: a 2-D tensor");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let (_, mine) = share(cols as usize, c.rank(), c.n_ranks());
    ctx.external_with(
        &REDUCE_SCATTER,
        types::F32,
        [row, mine as i64, 1, 1],
        &[x],
        &[comm],
    )
}

/// Every rank's share of the columns of a `[row, cols]` tensor, from this rank's `x` (F32
/// `[row, share]`): `[row, cols]`.
pub fn all_gather<'a>(ctx: &'a Context, comm: i32, x: Tensor<'a>, cols: i64) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "tp all-gather: F32 only");
    let row = x.ne()[0];
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.external_with(&ALL_GATHER, types::F32, [row, cols, 1, 1], &[x], &[comm])
}

/// The node's F32 source and output as host slices, and its row length.
///
/// # Safety
/// A node made by [`reduce_scatter`] or [`all_gather`] on the CPU.
// The output is ggml's buffer that `dst` points to, not the node itself.
#[allow(clippy::mut_from_ref)]
unsafe fn host_io(dst: &ffi::ggml_tensor) -> (&[f32], &mut [f32], usize) {
    // SAFETY: the caller's: contiguous F32 tensors with host data.
    unsafe {
        let src = &*dst.src[0];
        (
            std::slice::from_raw_parts(src.data as *const f32, ffi::ggml_nelements(src) as usize),
            std::slice::from_raw_parts_mut(dst.data as *mut f32, ffi::ggml_nelements(dst) as usize),
            dst.ne[0] as usize,
        )
    }
}

unsafe fn cpu_reduce_scatter(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // One thread runs it; ggml's barrier after the node holds the others.
    if ith != 0 {
        return;
    }
    // SAFETY: a node of `reduce_scatter`.
    let (src, out, row) = unsafe { host_io(dst) };
    match super::comm(external_params(dst)[0]) {
        // A failure stays in the communicator, for the step to report.
        Some(c) => {
            let _ = c.reduce_scatter_host(src, out, row);
        }
        None => {
            tracing::error!("tp reduce-scatter: the communicator is gone");
            out.fill(0.0);
        }
    }
}

unsafe fn cpu_all_gather(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node of `all_gather`.
    let (src, out, row) = unsafe { host_io(dst) };
    match super::comm(external_params(dst)[0]) {
        Some(c) => {
            let _ = c.all_gather_host(src, out, row);
        }
        None => {
            tracing::error!("tp all-gather: the communicator is gone");
            out.fill(0.0);
        }
    }
}
