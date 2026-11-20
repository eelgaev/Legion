//! Matmuls that write straight into the send buffer (#1133). With F32 on the wire, an
//! all-reduce start ([`super::all_reduce_start`]) copies this rank's partial into the pinned
//! send buffer, and its finish reads the partial from where the matmul left it. On a Qwen3.8-27B
//! prompt step over two DGX Sparks those partials are the output projections' and the FFN down
//! matmuls' (per half of the tokens, `crate::model::qwen35`'s halves), and the copies took ~220
//! ms of an 8K-token step on each rank.
//!
//! [`take`] (a post-allocation rewrite on CUDA graphs) finds the starts whose source is a
//! column range of an FP8 matmul's output ([`crate::model::fp8::sendable`]) or the whole output
//! of a W4A4 one ([`crate::model::nvfp4_a4::sendable`]), where nothing else reads that output
//! but the starts and their finishes. The matmul node then launches nothing; each start runs its
//! own columns of it with the output in its send buffer, and its finish reads this rank's
//! partial from there. Every value is the same (a column's sums do not depend on the other
//! columns; the W4A4 matmul runs whole, as before), so the sums and the text keep their bits.
//! The F32 copy pass goes, and a start can send its half as soon as its own columns are done.
//!
//! `LEGION_DEBUG_TP_SEND_GEMM=0` turns the rewrite off (for comparisons).

// Only a CUDA build rewrites its graphs.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::ggml::{external_params, ffi};

use super::{FINISH, Params, START};

type Node = *mut ffi::ggml_tensor;

/// The parameter of a start node that [`take`] sets (after [`Params::list`]'s): it runs its
/// source's columns of the matmul ([`taken`]).
const P_TAKE: usize = 6;

/// Whether graphs allocated from now on may take the rewrite: on, unless
/// `LEGION_DEBUG_TP_SEND_GEMM=0`. Read once.
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_of(std::env::var("LEGION_DEBUG_TP_SEND_GEMM").ok().as_deref()))
}

/// [`on`] for the variable's value `v`.
fn on_of(v: Option<&str>) -> bool {
    v.is_none_or(|s| s.trim() != "0")
}

/// Whether the matmuls write the send buffer (mapped host memory) by whole runs of a token's
/// rows (the GEMMs' staged epilogues): on, unless `LEGION_DEBUG_TP_SEND_STAGED=0` (their
/// usual stores, about twice as slow into that memory on the GB10). The same bits either way.
/// Read once.
#[cfg(feature = "cuda")]
fn staged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| on_of(std::env::var("LEGION_DEBUG_TP_SEND_STAGED").ok().as_deref()))
}

/// The matmuls a start may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// [`crate::model::fp8`]'s W8A8 GEMM: any column range.
    Fp8,
    /// [`crate::model::nvfp4_a4`]'s W4A4 GEMM: whole only.
    A4,
}

/// The matmul kind of `t` and its output's `(m, n)` (F32 `[m, n]` from the start of its
/// memory), if a start may run it.
///
/// # Safety
/// `t` is a valid tensor.
unsafe fn kind_of(t: Node) -> Option<(Kind, usize, usize)> {
    // SAFETY: the caller's.
    unsafe {
        if crate::model::fp8::sendable(t) {
            let ne = (*t).ne;
            Some((Kind::Fp8, ne[0] as usize, (ne[1] * ne[2] * ne[3]) as usize))
        } else if crate::model::nvfp4_a4::sendable(t) {
            let (m, n) = crate::model::nvfp4_a4::sendable_shape(t);
            Some((Kind::A4, m, n))
        } else {
            None
        }
    }
}

/// The tensor whose memory `t` is (its view source's, all the way down).
///
/// # Safety
/// `t` is a valid tensor.
unsafe fn root(mut t: Node) -> Node {
    // SAFETY: the caller's; view sources are valid tensors.
    unsafe {
        while !(*t).view_src.is_null() {
            t = (*t).view_src;
        }
        t
    }
}

fn is_view_op(op: ffi::ggml_op) -> bool {
    matches!(
        op,
        ffi::ggml_op_GGML_OP_VIEW
            | ffi::ggml_op_GGML_OP_RESHAPE
            | ffi::ggml_op_GGML_OP_PERMUTE
            | ffi::ggml_op_GGML_OP_TRANSPOSE
            | ffi::ggml_op_GGML_OP_NONE
    )
}

/// Whether `t` is an all-reduce start ([`super::all_reduce_start`]).
///
/// # Safety
/// `t` is a valid tensor.
unsafe fn is_reduce_start(t: Node) -> bool {
    // SAFETY: the caller's.
    unsafe {
        crate::ggml::op_of(t).is_some_and(|op| std::ptr::eq(op, &START)) && Params::of(&*t).reduce
    }
}

/// Whether the column ranges `(first, count)` cover `0..n` once each, each of more than
/// `min` columns (a W4A4 matmul, `whole`: one range, all of them).
fn covers(ranges: &mut [(usize, usize)], n: usize, whole: bool, min: usize) -> bool {
    ranges.sort_unstable();
    if whole && ranges.len() != 1 {
        return false;
    }
    let mut at = 0;
    for &(a, c) in ranges.iter() {
        if a != at || c <= min {
            return false;
        }
        at += c;
    }
    at == n
}

/// Columns a range of an FP8 matmul must exceed: the large-batch GEMM, whose tiles all give
/// the same bits ([`crate::model::fp8::launch_cols`]); fewer go to other kernels.
const FP8_MIN_COLS: usize = 64;

/// A start's matmul columns: `(start index, first column, columns)`.
type Part = (usize, usize, usize);

/// The rewrite (see the module doc) on an allocated CUDA graph none of whose nodes has run:
/// returns how many starts run their matmul's columns now. Where the order of nodes or the
/// readers do not allow it, the graph stays as it was.
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
pub(crate) unsafe fn take(graph: *mut ffi::ggml_cgraph) -> usize {
    if !on() {
        return 0;
    }
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n);
        take_nodes(nodes)
    }
}

/// [`take`] on `nodes` (a graph's, in order).
///
/// # Safety
/// As [`take`].
pub(crate) unsafe fn take_nodes(nodes: &[Node]) -> usize {
    // SAFETY: the caller's; every node and its sources are valid tensors.
    unsafe {
        // The starts of an F32 all-reduce reading a column range of a matmul's output.
        let mut parts: HashMap<usize, Vec<Part>> = HashMap::new();
        for (i, &t) in nodes.iter().enumerate() {
            if !is_reduce_start(t) || external_params(&*t)[P_TAKE] != 0 {
                continue;
            }
            let p = Params::of(&*t);
            let Some(comm) = super::super::comm(p.comm) else {
                continue;
            };
            let src = (*t).src[0];
            let row = (*src).ne[0] as usize;
            if comm.wire(row * p.n) != super::super::Wire::F32 || !ffi::ggml_is_contiguous(src) {
                continue;
            }
            let base = root(src);
            let Some((_, m, cols)) = kind_of(base) else {
                continue;
            };
            let (at, len) = ((*src).data as usize, ffi::ggml_nbytes(src));
            let from = (*base).data as usize;
            let col = m * 4;
            if row != m || at < from || (at - from) % col != 0 || len % col != 0 {
                continue;
            }
            let (c0, cn) = ((at - from) / col, len / col);
            if c0 + cn <= cols {
                parts.entry(base as usize).or_default().push((i, c0, cn));
            }
        }
        if parts.is_empty() {
            return 0;
        }
        // Every other node reading a matmul's output (directly or through views) must be one
        // of its starts' finishes; nothing may be a graph output.
        let index: HashMap<usize, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &t)| (t as usize, i))
            .collect();
        let starts_of = |b: usize, s: Node| {
            parts
                .get(&b)
                .is_some_and(|ps| ps.iter().any(|&(i, _, _)| nodes[i] == s))
        };
        let mut bad: Vec<usize> = Vec::new();
        for &t in nodes {
            let b = root(t) as usize;
            if parts.contains_key(&b)
                && (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0
            {
                bad.push(b);
            }
            if is_view_op((*t).op) {
                continue;
            }
            for (j, &s) in (*t).src.iter().enumerate() {
                if s.is_null() {
                    continue;
                }
                let b = root(s) as usize;
                if !parts.contains_key(&b) || b == t as usize {
                    continue;
                }
                let finish = j == 0
                    && crate::ggml::op_of(t).is_some_and(|op| std::ptr::eq(op, &FINISH))
                    && starts_of(b, (*t).src[1]);
                if !(starts_of(b, t) && j == 0) && !finish {
                    bad.push(b);
                }
            }
        }
        let mut taken = 0;
        for (b, ps) in &parts {
            let base = *b as Node;
            let Some((kind, _, cols)) = kind_of(base) else {
                continue;
            };
            let mut ranges: Vec<(usize, usize)> = ps.iter().map(|&(_, a, c)| (a, c)).collect();
            let (whole, min) = match kind {
                Kind::Fp8 => (false, FP8_MIN_COLS),
                Kind::A4 => (true, 0),
            };
            if bad.contains(b) || !covers(&mut ranges, cols, whole, min) {
                continue;
            }
            // The matmul's inputs stay as they are until its last start runs it: nothing but
            // views and its starts lies between them.
            let Some(&g) = index.get(b) else {
                continue;
            };
            let last = ps.iter().map(|&(i, _, _)| i).max().unwrap_or(0);
            let quiet = g < last
                && (g + 1..last)
                    .all(|i| is_view_op((*nodes[i]).op) || ps.iter().any(|&(s, _, _)| s == i));
            if !quiet {
                continue;
            }
            let p = match kind {
                Kind::Fp8 => crate::model::fp8::P_BY_EXCHANGE,
                Kind::A4 => crate::model::nvfp4_a4::P_BY_EXCHANGE,
            };
            crate::ggml::set_external_param(base, p, 1);
            for &(i, _, _) in ps {
                crate::ggml::set_external_param(nodes[i], P_TAKE, 1);
            }
            taken += ps.len();
        }
        taken
    }
}

/// A start's columns of the matmul it runs ([`taken`]).
#[cfg(feature = "cuda")]
pub(crate) struct Taken<'a> {
    gemm: &'a ffi::ggml_tensor,
    kind: Kind,
    /// `(first column, columns)` of the matmul's output.
    cols: (usize, usize),
    /// Values of the columns.
    pub values: usize,
}

/// The matmul columns the start node `start` runs itself ([`take`]), if any.
///
/// # Safety
/// `start` is an all-reduce start of an allocated graph.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn taken(start: &ffi::ggml_tensor) -> Option<Taken<'_>> {
    if external_params(start)[P_TAKE] == 0 {
        return None;
    }
    // SAFETY: the caller's; `take` checked the source and its matmul.
    unsafe {
        let src = start.src[0];
        let base = root(src);
        let (kind, m, _) = kind_of(base)?;
        let off = (*src).data as usize - (*base).data as usize;
        let cn = ffi::ggml_nbytes(src) / (m * 4);
        Some(Taken {
            gemm: &*base,
            kind,
            cols: (off / (m * 4), cn),
            values: m * cn,
        })
    }
}

#[cfg(feature = "cuda")]
impl Taken<'_> {
    /// Runs the columns on `stream`: into the send buffer at the device address `out`
    /// (`values` F32s, mapped host memory: staged, [`staged`]), else into the matmul's own
    /// output.
    ///
    /// # Safety
    /// The graph's memory holds the matmul's inputs (nothing between it and its last start
    /// ran, [`take`]); `out` device-writable for `values` F32s.
    pub(crate) unsafe fn launch(
        &self,
        out: Option<u64>,
        stream: *mut std::os::raw::c_void,
    ) -> Result<(), String> {
        let staged = out.is_some() && staged();
        // SAFETY: the caller's.
        unsafe {
            match (self.kind, out) {
                (Kind::Fp8, out) => {
                    crate::model::fp8::launch_cols(self.gemm, self.cols, out, staged, stream)
                }
                (Kind::A4, Some(o)) => {
                    crate::model::nvfp4_a4::mul_mat_to(self.gemm, o, staged, stream)
                }
                (Kind::A4, None) => crate::model::nvfp4_a4::mul_mat_to(
                    self.gemm,
                    self.gemm.data as u64,
                    false,
                    stream,
                ),
            }
        }
    }
}
