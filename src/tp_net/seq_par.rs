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

/// The parameter of a [`reduce_scatter`] node that [`defer_sum`] sets.
const P_DEFERRED: usize = 1;

/// Whether `out` is a [`reduce_scatter`] node whose sum a reader may take over ([`defer_sum`]):
/// two ranks, its partials travel in BF16 (`--tp-wire bf16` at its size), it is not a graph
/// output and no reader took it over yet.
pub fn deferrable(out: Tensor<'_>) -> bool {
    // SAFETY: `out` is a valid tensor of its context; a node of ours keeps its params.
    unsafe {
        let t = &*out.raw();
        if !crate::ggml::op_of(t).is_some_and(|op| std::ptr::eq(op, &REDUCE_SCATTER)) {
            return false;
        }
        let p = external_params(t);
        let Some(c) = super::comm(p[0]) else {
            return false;
        };
        // A graph output (a traced intermediate) is read as written.
        let output = t.flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0;
        let full = ffi::ggml_nelements(t.src[0]) as usize;
        !output && p[P_DEFERRED] == 0 && c.n_ranks() == 2 && c.wire(full) == super::Wire::Bf16
    }
}

/// Hands the sum of the [`deferrable`] reduce-scatter `out` to its only reader (#1176: the
/// next hyper-connection norm, `hc_post_norm` in its rebuilt BF16-xn form), which adds the two
/// ranks' BF16 partials itself as the sum kernel does (rank order, one F32 rounding): on CUDA
/// the node then waits for the peer's partial but does not sum, leaves `out` unwritten, puts
/// the partials' addresses in the communicator's slot for the reader ([`sum_slot`]) and keeps
/// the buffers held until the next exchange. Where the graph's order runs another exchange
/// before the reader, [`check_order`] undoes this before the graph's first run, and the node
/// sums into `out` as before. The CPU op writes `out` either way.
pub fn defer_sum(out: Tensor<'_>) {
    assert!(deferrable(out), "seq_par::defer_sum: not a deferrable sum");
    // SAFETY: a node of `reduce_scatter` (checked above).
    unsafe { crate::ggml::set_external_param(out.raw(), P_DEFERRED, 1) };
}

/// Whether the [`reduce_scatter`] node `dst` hands its sum to its reader ([`defer_sum`]).
pub(crate) fn deferred(dst: &ffi::ggml_tensor) -> bool {
    external_params(dst)[P_DEFERRED] != 0
}

/// The parameter of a [`reduce_scatter`] node that [`take_tail`] sets.
const P_TAIL: usize = 2;

/// A post-allocation rewrite on CUDA (#1176): the [`reduce_scatter`] node `rs` whose source
/// is `tail` (a MoE tail made one op in its fused-sum form,
/// [`crate::model::core::node_fuse::MOE_SUM_GATE`], which only `rs` reads, directly or
/// through a view of all of it) runs that tail itself: with `--tp-wire bf16` at its size the
/// tail's values go rounded to BF16 straight into the send buffer (the values the exchange's
/// rounding of `tail` would put there, without `tail` written and read back), else into
/// `tail` first as before. `inputs` (the tail's sources) become `rs`'s sources too, so the
/// graph, planned again after the rewrite, keeps them until `rs`; `tail` launches nothing.
/// Whether it did (not for a micro-batch exchange or one already rewritten).
///
/// # Safety
/// Valid nodes of a graph not yet run; the caller plans the graph again and saves `rs` and
/// `tail` first if it may undo this.
pub(crate) unsafe fn take_tail(
    rs: *mut ffi::ggml_tensor,
    tail: *mut ffi::ggml_tensor,
    inputs: &[*mut ffi::ggml_tensor],
) -> bool {
    // SAFETY: the caller's.
    unsafe {
        let t = &mut *rs;
        if !crate::ggml::op_of(t).is_some_and(|op| std::ptr::eq(op, &REDUCE_SCATTER))
            || external_params(t)[P_TAIL] != 0
            || inputs.len() + 1 > ffi::GGML_MAX_SRC as usize
            || t.src[1..].iter().any(|s| !s.is_null())
        {
            return false;
        }
        let src = t.src[0];
        let whole = src == tail
            || (!(*src).view_src.is_null()
                && (*src).view_src == tail
                && (*src).view_offs == 0
                && ffi::ggml_is_contiguous(src)
                && ffi::ggml_nelements(src) == ffi::ggml_nelements(tail));
        if !whole || !ffi::ggml_is_contiguous(tail) {
            return false;
        }
        t.src[1..1 + inputs.len()].copy_from_slice(inputs);
        crate::ggml::set_external_param(rs, P_TAIL, 1);
        crate::ggml::set_external_param(tail, crate::model::core::node_fuse::P_BY_EXCHANGE, 1);
        true
    }
}

/// The MoE tail node the [`reduce_scatter`] node `dst` runs itself ([`take_tail`]), if any.
#[cfg(feature = "cuda")]
pub(crate) fn tail_of(dst: &ffi::ggml_tensor) -> Option<&ffi::ggml_tensor> {
    if external_params(dst)[P_TAIL] == 0 {
        return None;
    }
    // SAFETY: a node of `reduce_scatter`: its source is the tail or a view of all of it.
    unsafe {
        let src = &*dst.src[0];
        Some(if src.view_src.is_null() {
            src
        } else {
            &*src.view_src
        })
    }
}

/// For the reader of the deferred reduce-scatter node `rs` ([`defer_sum`]), on CUDA: the device
/// address of the communicator's sum slot, two `u64`s that the exchange sets on the stream to
/// the device addresses of rank 0's and rank 1's BF16 partials of this rank's columns
/// (`[row, share]` each). The address is fixed for the communicator's life, so a stage graph
/// may capture the reader's launch: a replay reads the addresses the latest exchange wrote.
/// The exchange's buffers are freed by the next exchange ([`check_order`] makes sure the
/// reader runs before it).
#[cfg(feature = "cuda")]
pub(crate) fn sum_slot(rs: &ffi::ggml_tensor) -> Result<u64, String> {
    let c = super::comm(external_params(rs)[0]).ok_or("tp reduce-scatter: the communicator")?;
    super::op::cuda::sum_slot(&c).ok_or_else(|| "tp reduce-scatter: no sum slot yet".into())
}

/// Whether `t` is a node of one of the exchanges: an all-reduce, a [`reduce_scatter`], an
/// [`all_gather`] or a micro-batch exchange's half ([`super::micro`]).
fn is_exchange(t: &ffi::ggml_tensor) -> bool {
    // SAFETY: a valid tensor.
    unsafe { crate::ggml::op_of(t) }.is_some_and(|op| op.name.starts_with("tp_"))
}

/// Whether `t` is a [`reduce_scatter`] node whose sum its reader makes ([`defer_sum`]).
fn is_deferred_rs(t: &ffi::ggml_tensor) -> bool {
    // SAFETY: a valid tensor.
    unsafe { crate::ggml::op_of(t) }.is_some_and(|op| std::ptr::eq(op, &REDUCE_SCATTER))
        && deferred(t)
}

/// Whether `t` reads its sources' memory when it runs: not a view or a node that runs nothing.
fn reads(t: &ffi::ggml_tensor) -> bool {
    !matches!(
        t.op,
        ffi::ggml_op_GGML_OP_NONE
            | ffi::ggml_op_GGML_OP_VIEW
            | ffi::ggml_op_GGML_OP_RESHAPE
            | ffi::ggml_op_GGML_OP_PERMUTE
            | ffi::ggml_op_GGML_OP_TRANSPOSE
    )
}

/// [`check_order`] as the check run on every allocated graph
/// ([`crate::ggml::post_alloc::set_order`]), whatever memory it got.
///
/// # Safety
/// As [`check_order`].
pub(crate) unsafe fn order_pass(graph: *mut ffi::ggml_cgraph) {
    // SAFETY: the caller's.
    unsafe { check_order(graph) };
}

/// A deferred sum made by its node after all ([`check_order`]), for the log.
static WRITTEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Before `graph` first runs (#1176): every [`defer_sum`] reduce-scatter whose reader does not
/// run before the graph's next exchange, or that no node reads, writes its sum after all, as
/// without [`defer_sum`]. The exchange after a deferred one frees its buffers and its reader
/// reads the addresses the deferred one left in the communicator's slot ([`sum_slot`]), so that
/// reader must come first, in every run of the graph alike (a stage graph's replay too). The
/// readers read the sum from the output then, the same bits. Returns how many it changed.
///
/// # Safety
/// A live graph none of whose nodes has run yet.
pub(crate) unsafe fn check_order(graph: *mut ffi::ggml_cgraph) -> usize {
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n);
        let mut pending: Option<*mut ffi::ggml_tensor> = None;
        let mut late = Vec::new();
        for &t in nodes {
            let node = &*t;
            if let Some(rs) = pending {
                if is_exchange(node) {
                    late.push(rs);
                    pending = None;
                } else if reads(node)
                    && node
                        .src
                        .iter()
                        .any(|&s| !s.is_null() && (s == rs || (*s).view_src == rs))
                {
                    // Its reader (directly or through a view).
                    pending = None;
                }
            }
            if is_deferred_rs(node) {
                pending = Some(t);
            }
        }
        late.extend(pending);
        for &rs in &late {
            crate::ggml::set_external_param(rs, P_DEFERRED, 0);
        }
        let changed = late.len();
        if changed > 0 && WRITTEN.fetch_add(changed, std::sync::atomic::Ordering::Relaxed) < 3 {
            tracing::warn!(
                sums = changed,
                "tp reduce-scatter: a deferred sum's reader does not run before the next \
                 exchange; the sum is written to its output instead (the first graphs)"
            );
        }
        changed
    }
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
