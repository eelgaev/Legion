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
}

impl Params {
    fn list(&self) -> [i32; 5] {
        [
            self.comm,
            self.gather as i32,
            self.n as i32,
            self.j as i32,
            self.compact as i32,
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
    };
    start(ctx, x, p)
}

/// The finish of `s`: a reduce-scatter's sum of this rank's piece (`[row, piece]`), an
/// all-gather's `[row, span]` ([`span_cols`]) with every rank's piece. `into`: an all-gather
/// in the natural layout writes into that tensor (the other micro-batch's finished
/// all-gather, which holds its own pieces), and the result is a view of it holding both.
pub fn finish<'a>(ctx: &'a Context, s: Started<'a>, into: Option<Tensor<'a>>) -> Tensor<'a> {
    let c = super::comm(s.p.comm).expect("tp micro exchange: the communicator");
    let (rank, ranks) = (c.rank(), c.n_ranks());
    let row = s.x.ne()[0];
    let cols = if s.p.gather {
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

unsafe fn cpu_start(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {
    // The CPU's finish runs the whole exchange.
}

unsafe fn cpu_finish(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // One thread runs it; ggml's barrier after the node holds the others.
    if ith != 0 {
        return;
    }
    let p = Params::of(dst);
    // SAFETY: a node of `finish` on the CPU: contiguous F32 source and output with host data.
    let (src, out, row) = unsafe {
        let src = &*dst.src[0];
        (
            std::slice::from_raw_parts(src.data as *const f32, ffi::ggml_nelements(src) as usize),
            std::slice::from_raw_parts_mut(dst.data as *mut f32, ffi::ggml_nelements(dst) as usize),
            dst.ne[0] as usize,
        )
    };
    match super::comm(p.comm) {
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
