//! Two micro-batches per sequence-parallel step (`--tp-overlap`, #758). In a `--tp-seq-par`
//! step every sum is a reduce-scatter and every block input an all-gather ([`super::seq_par`]),
//! and the GPU waits for each: on two DGX Sparks about 190 exchanges of about 1 ms of wire
//! time in a 4096-token step. Here the per-token work between two blocks runs in two halves of
//! every rank's share of the tokens (micro-batch 0: the first half of each rank's share, 1: the
//! second), and each exchange is split in two graph nodes: a start (this rank's values into
//! the send buffer, the transfer queued) and a finish (the wait for the peers' values and the
//! sum). The model's graph puts one half's compute between the other half's start and finish,
//! so the transfer runs while the GPU computes ([`crate::model::qwen4`]'s `hc_layers`).
//!
//! The columns each rank sends and gets are [`piece`]s: in the natural layout the columns of
//! the step's `[row, n]` tensor (an attention's partial output, its input), in the compact
//! layout a micro-batch's own `[row, n_j]` tensor, every rank's piece after the other in rank
//! order (the MoE's input and output with `--tp-overlap ffn`). A sum adds every rank's value in
//! rank order and with `--tp-wire bf16` rounds every one of them, so each value gets the bits
//! the plain exchange gives it. The model's per-token work on half the columns does not, on a
//! GPU: ggml's MMQ splits K over its CTAs by the number of output tiles (stream-k), so the
//! mixers' and experts' matmuls round differently and long prompts' greedy output differs
//! from `--tp-overlap off` (on the CPU, the reference, it is the same; #758 measured it).
//!
//! A model whose steps are not sequence-parallel splits an all-reduce the same way
//! ([`all_reduce_start`]): a start and a finish per column range of its own (Gemma 4's prompt
//! steps over two halves of their tokens, #619).
//!
//! On a GPU the finishes run in the order of the starts and at most two exchanges are in
//! flight; on the CPU (the reference) the start does nothing and the finish the whole exchange.

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

use super::Overlap;
use super::seq_par::share;

#[cfg(test)]
mod tests;

/// Micro-batches per step.
pub const MICRO: usize = 2;

/// Rank `r`'s piece of micro-batch `j` in a step of `n` columns among `ranks`: `(first column,
/// columns)` in the natural layout (of the step's `n` columns) or the compact one (of the
/// micro-batch's [`span_cols`]).
pub fn piece(n: usize, r: usize, ranks: usize, j: usize, compact: bool) -> (usize, usize) {
    let half = |q: usize| {
        let (a, c) = share(n, q, ranks);
        let (a2, c2) = share(c, j, MICRO);
        (a + a2, c2)
    };
    if compact {
        ((0..r).map(|q| half(q).1).sum(), half(r).1)
    } else {
        half(r)
    }
}

/// The columns of the tensor a micro-batch exchange reads (reduce-scatter) or writes
/// (all-gather): the step's `n` (natural layout) or micro-batch `j`'s (compact).
pub fn span_cols(n: usize, ranks: usize, j: usize, compact: bool) -> usize {
    if compact {
        (0..ranks).map(|r| piece(n, r, ranks, j, true).1).sum()
    } else {
        n
    }
}

/// The communicator's `--tp-overlap` (off without it).
pub fn overlap(comm: i32) -> Overlap {
    super::comm(comm).map_or(Overlap::Off, |c| c.overlap())
}

/// Micro-batches of fewer columns than this leave a step whole ([`worth`]). On a GPU a
/// micro-batch's matmuls may split K differently from the whole share's, and below about this
/// size they did (#1176: a 1,536-token prompt's greedy text moved, a 4,096-token one's did
/// not); such steps also have little wire time to hide.
pub const MIN_COLS: usize = 1024;

/// [`MIN_COLS`], or what [`set_min_cols`] set.
static MIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(MIN_COLS);

/// Sets the smallest micro-batch [`worth`] takes (tests: a tiny model's steps are short).
#[doc(hidden)]
pub fn set_min_cols(cols: usize) {
    MIN.store(cols, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a step of `n` columns over `ranks` ranks runs in micro-batches under
/// `--tp-overlap`: every rank's every micro-batch has at least [`MIN_COLS`] columns. The
/// same answer on every rank.
pub fn worth(n: usize, ranks: usize) -> bool {
    worth_at(n, ranks, usize::MAX)
}

/// [`worth`] with a floor of the caller's, `min` columns a micro-batch (or [`set_min_cols`]'s
/// if lower): for a model whose split work gives every token the same bits at any column
/// count (#560 r12: GLM-5.3-Flash's mixers).
pub fn worth_at(n: usize, ranks: usize, min: usize) -> bool {
    let min = min.min(MIN.load(std::sync::atomic::Ordering::Relaxed));
    (0..ranks)
        .flat_map(|r| (0..MICRO).map(move |j| piece(n, r, ranks, j, false).1))
        .all(|m| m >= min)
}

/// A micro-batch exchange node's parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub comm: i32,
    /// All-gather (else reduce-scatter).
    pub gather: bool,
    /// The step's columns.
    pub n: usize,
    /// The micro-batch.
    pub j: usize,
    /// The compact layout (else the natural one).
    pub compact: bool,
    /// An all-reduce of the micro-batch's own tensor ([`all_reduce_start`]): every rank sends
    /// all of it and gets the sum of all of it; `gather`, `j` and `compact` unused.
    pub reduce: bool,
}

impl Params {
    fn list(&self) -> [i32; 6] {
        [
            self.comm,
            self.gather as i32,
            self.n as i32,
            self.j as i32,
            self.compact as i32,
            self.reduce as i32,
        ]
    }

    /// The parameters of a node made here.
    pub(crate) fn of(dst: &ffi::ggml_tensor) -> Self {
        let p = external_params(dst);
        Self {
            comm: p[0],
            gather: p[1] != 0,
            n: p[2] as usize,
            j: p[3] as usize,
            compact: p[4] != 0,
            reduce: p[5] != 0,
        }
    }
}

static START: ExternalOp = ExternalOp {
    name: "tp_micro_start",
    cpu: cpu_start,
    #[cfg(feature = "cuda")]
    cuda: Some(super::op::cuda::launch_micro_start),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static FINISH: ExternalOp = ExternalOp {
    name: "tp_micro_finish",
    cpu: cpu_finish,
    #[cfg(feature = "cuda")]
    cuda: Some(super::op::cuda::launch_micro_finish),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// A started exchange: its source, its start node (the caller puts it into the graph where
/// the transfer should begin) and its parameters.
#[derive(Clone, Copy)]
pub struct Started<'a> {
    x: Tensor<'a>,
    pub node: Tensor<'a>,
    p: Params,
}

fn start<'a>(ctx: &'a Context, x: Tensor<'a>, p: Params) -> Started<'a> {
    assert_eq!(x.ty(), types::F32, "tp micro exchange: F32 only");
    let [row, cols, a, b] = x.ne();
    assert_eq!((a, b), (1, 1), "tp micro exchange: a 2-D tensor");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let x = ctx.reshape_2d(x, row, cols);
    let node = ctx.external_with(&START, types::F32, [1, 1, 1, 1], &[x], &p.list());
    Started { x, node, p }
}

/// Starts the reduce-scatter of micro-batch `j` of `x` (F32 `[row, cols]`, this rank's partial:
/// the step's `n` columns, or with `compact` the micro-batch's own).
pub fn reduce_scatter_start<'a>(
    ctx: &'a Context,
    comm: i32,
    x: Tensor<'a>,
    n: usize,
    j: usize,
    compact: bool,
) -> Started<'a> {
    let p = Params {
        comm,
        gather: false,
        n,
        j,
        compact,
        reduce: false,
    };
    start(ctx, x, p)
}

/// Starts the all-reduce of `x` (F32 `[row, cols]`, this rank's partial of some of a step's
/// `n` columns, e.g. half of them): its finish is the sum over the ranks of all of `x`, every
/// value the bits the plain [`super::all_reduce`] of the step's whole `[row, n]` tensor gives
/// it (the same wire type, chosen by the whole tensor's values, and the same sum in rank
/// order). For models that split a step's per-token work into column ranges themselves
/// ([`crate::model::gemma4`]'s prompt steps, #619).
pub fn all_reduce_start<'a>(ctx: &'a Context, comm: i32, x: Tensor<'a>, n: usize) -> Started<'a> {
    let p = Params {
        comm,
        gather: false,
        n,
        j: 0,
        compact: false,
        reduce: true,
    };
    start(ctx, x, p)
}

/// Starts the all-gather of micro-batch `j` from `x` (F32 `[row, piece]`, this rank's piece).
pub fn all_gather_start<'a>(
    ctx: &'a Context,
    comm: i32,
    x: Tensor<'a>,
    n: usize,
    j: usize,
    compact: bool,
) -> Started<'a> {
    let p = Params {
        comm,
        gather: true,
        n,
        j,
        compact,
        reduce: false,
    };
    start(ctx, x, p)
}

/// The finish of `s`: a reduce-scatter's sum of this rank's piece (`[row, piece]`), an
/// all-gather's `[row, span]` ([`span_cols`]) with every rank's piece, an all-reduce's sum of
/// its whole source. `into`: an all-gather in the natural layout writes into that tensor (the
/// other micro-batch's finished all-gather, which holds its own pieces), and the result is a
/// view of it holding both.
pub fn finish<'a>(ctx: &'a Context, s: Started<'a>, into: Option<Tensor<'a>>) -> Tensor<'a> {
    let c = super::comm(s.p.comm).expect("tp micro exchange: the communicator");
    let (rank, ranks) = (c.rank(), c.n_ranks());
    let row = s.x.ne()[0];
    let cols = if s.p.reduce {
        assert!(
            into.is_none(),
            "tp micro exchange: an all-reduce has its own output"
        );
        s.x.ne()[1] as usize
    } else if s.p.gather {
        span_cols(s.p.n, ranks, s.p.j, s.p.compact)
    } else {
        piece(s.p.n, rank, ranks, s.p.j, s.p.compact).1
    } as i64;
    let mut args = vec![s.x, s.node];
    if let Some(t) = into {
        assert!(
            s.p.gather && !s.p.compact,
            "tp micro exchange: only a natural all-gather writes into another"
        );
        assert_eq!(
            t.ne(),
            [row, cols, 1, 1],
            "tp micro exchange: the shared output"
        );
        assert!(
            t.is_contiguous(),
            "tp micro exchange: a contiguous shared output"
        );
        args.push(t);
    }
    let out = ctx.external_with(&FINISH, types::F32, [row, cols, 1, 1], &args, &s.p.list());
    if let Some(t) = into {
        // The node writes its pieces into `t`'s memory: ggml's allocator gives a view no memory
        // of its own and keeps `t` alive while the view is used.
        // SAFETY: a fresh node, not allocated yet; `t` is a node of the same graph, ahead of
        // it (one of its sources) and of its shape.
        unsafe {
            let raw = out.raw();
            (*raw).view_src = t.raw();
            (*raw).view_offs = 0;
        }
    }
    out
}

/// The params after [`Params::list`]'s that [`finish_add`] sets: whether the node adds a
/// residual (its source 2) and the column of its output it writes from.
const ADD_PARAMS: usize = 6;

/// The finish of the all-reduce `s` ([`all_reduce_start`]) with `res` (F32, `s`'s source's
/// shape `[row, cols]`) added to every sum, after the partials, in the sum's kernel (#1133 r2):
/// `add(finish(s), res)`'s bits, without the add's pass over the sum. The output has `width`
/// columns (at least `at + cols`) and the sums go to columns `at..at + cols`; with `into` (a
/// [`finish_add`] of the same `width` ahead of this one, holding the other columns) the output
/// is a view of `into`'s memory, so that two halves of a step's tokens end in one tensor
/// without a concatenation. Columns no finish writes are undefined.
pub fn finish_add<'a>(
    ctx: &'a Context,
    s: Started<'a>,
    res: Tensor<'a>,
    width: i64,
    at: i64,
    into: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        s.p.reduce,
        "tp micro exchange: only an all-reduce adds a residual"
    );
    let [row, cols] = [s.x.ne()[0], s.x.ne()[1]];
    assert_eq!(res.ty(), types::F32, "tp micro exchange: an F32 residual");
    assert_eq!(
        res.ne(),
        [row, cols, 1, 1],
        "tp micro exchange: the residual's shape"
    );
    assert!(
        at >= 0 && at + cols <= width,
        "tp micro exchange: columns {at}..{} of {width}",
        at + cols
    );
    let res = if res.is_contiguous() {
        res
    } else {
        ctx.cont(res)
    };
    let mut args = vec![s.x, s.node, res];
    if let Some(t) = into {
        assert_eq!(
            t.ne(),
            [row, width, 1, 1],
            "tp micro exchange: the shared output"
        );
        assert!(
            t.is_contiguous(),
            "tp micro exchange: a contiguous shared output"
        );
        args.push(t);
    }
    let mut p = s.p.list().to_vec();
    p.extend([1, at as i32]);
    let out = ctx.external_with(&FINISH, types::F32, [row, width, 1, 1], &args, &p);
    if let Some(t) = into {
        // As in `finish`: the node writes its columns into `t`'s memory.
        // SAFETY: a fresh node, not allocated yet; `t` is a node of the same graph, ahead of
        // it (one of its sources) and of its shape.
        unsafe {
            let raw = out.raw();
            (*raw).view_src = t.raw();
            (*raw).view_offs = 0;
        }
    }
    out
}

/// A finish node's residual (the device address of its source 2, [`finish_add`]) and the
/// column its sums go to (0 without a residual).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn add_of(dst: &ffi::ggml_tensor) -> (Option<usize>, usize) {
    let p = external_params(dst);
    if p[ADD_PARAMS] == 0 {
        return (None, 0);
    }
    // SAFETY: a node of `finish_add`, whose source 2 is its residual.
    let res = unsafe { (*dst.src[2]).data as usize };
    (Some(res), p[ADD_PARAMS + 1] as usize)
}

unsafe fn cpu_start(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {
    // The CPU's finish runs the whole exchange.
}

unsafe fn cpu_finish(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // One thread runs it; ggml's barrier after the node holds the others.
    if ith != 0 {
        return;
    }
    let p = Params::of(dst);
    let (res, at) = add_of(dst);
    // SAFETY: a node of `finish` or `finish_add` on the CPU: contiguous F32 source, output
    // (and residual, of the source's size) with host data.
    let (src, out, row, res) = unsafe {
        let src = &*dst.src[0];
        let len = ffi::ggml_nelements(src) as usize;
        (
            std::slice::from_raw_parts(src.data as *const f32, len),
            std::slice::from_raw_parts_mut(dst.data as *mut f32, ffi::ggml_nelements(dst) as usize),
            dst.ne[0] as usize,
            res.map(|r| std::slice::from_raw_parts(r as *const f32, len)),
        )
    };
    match super::comm(p.comm) {
        // A failure stays in the communicator, for the step to report.
        Some(c) if p.reduce => {
            let out = &mut out[at * row..at * row + src.len()];
            let _ = c.all_reduce_host_as(src, out, row * p.n);
            if let Some(res) = res {
                super::op::add_host(out, res);
            }
        }
        Some(c) => {
            let ranks = c.n_ranks();
            let pieces = |r: usize| piece(p.n, r, ranks, p.j, p.compact);
            // A failure stays in the communicator, for the step to report.
            let _ = c.micro_host(p.gather, src, out, row, &pieces, row * p.n);
        }
        None => {
            tracing::error!("tp micro exchange: the communicator is gone");
            out.fill(0.0);
        }
    }
}
