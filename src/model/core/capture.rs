//! Recurrent states saved inside a prompt step (`--prefix-cache-interval`, #46).
//!
//! Prefix caching on a hybrid model resumes a prompt only where the recurrent state after that
//! many tokens was saved. Ending a prompt step at every spaced save point cost ~8-10% prefill:
//! each point cut a step (one more pass over the weights). Here the step stays whole; inside
//! it each DeltaNet layer runs its gated delta rule in pieces split at the save points, the
//! state between two pieces is copied into the saved slot, and the conv histories (DeltaNet's
//! and qwen4exp's per-layer embedding's) are copied from the step's own conv inputs.
//!
//! The split leaves the step's bits as they were: the delta rule's state stays F32 between
//! pieces as between the tokens of one piece; ggml-cuda's chunked kernel (and Legion's
//! `gdn_mma` port) advance 16 tokens at a time from the group's start and take groups of 64+
//! tokens, so pieces start at multiples of [`ALIGN`] past the group's start and hold at least
//! [`ALIGN`] tokens each ([`offset`]); the per-token kernels take any split. Every other op of
//! the step sees the same tensors.

use std::cell::RefCell;

use crate::ggml::{Context, Graph, Tensor};

/// A capture's offset in its group, and the size of every piece of the split delta rule, are
/// multiples of this (and at least this).
pub const ALIGN: usize = 64;

/// One recurrent-state capture in a prompt group ([`super::GroupShape`]): the state after the
/// group's first `at` tokens goes to saved slot `saved` ([`super::Pool::Saved`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Capture {
    pub(crate) at: i64,
    pub(crate) saved: usize,
}

/// A save point `at` (tokens of the sequence) in a prompt group of `n_tok` tokens after
/// `n_past`: its offset in the group, if a capture there leaves the step's bits as they are
/// (see the module docs). The group's end (a step's micro-chunk boundary) needs no split.
pub fn offset(n_past: usize, n_tok: usize, at: usize) -> Option<usize> {
    let o = at.checked_sub(n_past)?;
    let inside = o < n_tok && o.is_multiple_of(ALIGN) && n_tok - o >= ALIGN;
    (o >= ALIGN && (o == n_tok || inside)).then_some(o)
}

/// `t` (tokens in dim 2, one sequence) from token `start`, `len` tokens: the same strides,
/// the (single) sequence's as if it held `len` tokens.
pub(super) fn tokens<'a>(ctx: &'a Context, t: Tensor<'a>, start: i64, len: i64) -> Tensor<'a> {
    let [a, b, _, d] = t.ne();
    assert_eq!(d, 1, "captures in a group of one sequence");
    let nb = t.nb();
    let nb3 = nb[2] * len as usize;
    ctx.view_4d(
        t,
        [a, b, len, 1],
        [nb[1], nb[2], nb3],
        start as usize * nb[2],
    )
}

/// Copies the `keep` conv inputs before token `at` of one sequence's rows `x` (`[width, n_tok,
/// ..]`, a column per token) into `history` (`[keep, width, ..]`, time fastest), as the conv's
/// own history update after `at` tokens would.
pub(crate) fn history<'a>(
    ctx: &'a Context,
    graph: &RefCell<Graph<'a>>,
    x: Tensor<'a>,
    at: i64,
    keep: i64,
    history: Tensor<'a>,
) {
    assert!(at >= keep, "a capture after {at} tokens keeps {keep}");
    let width = x.ne()[0];
    let rows = ctx.view_2d(
        x,
        [width, keep],
        x.nb()[1],
        (at - keep) as usize * x.nb()[1],
    );
    graph
        .borrow_mut()
        .expand(ctx.cpy(ctx.transpose(rows), history));
}

/// The bounds of the pieces the captures `caps` (ascending, each inside or at the end of the
/// group's `n_tok` tokens) split the group into.
pub(super) fn pieces(caps: &[Capture], n_tok: i64) -> Vec<(i64, i64)> {
    let mut bounds: Vec<i64> = caps.iter().map(|c| c.at).filter(|&a| a < n_tok).collect();
    bounds.dedup();
    let mut out = Vec::with_capacity(bounds.len() + 1);
    let mut start = 0;
    for b in bounds.into_iter().chain([n_tok]) {
        out.push((start, b - start));
        start = b;
    }
    out
}

#[cfg(test)]
mod tests;
