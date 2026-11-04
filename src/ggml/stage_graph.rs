//! Stage graphs (#817; `legion serve`'s default since #849, `--no-stage-graphs` for off): a
//! graph's whole CUDA compute, its ggml segments and its external ops' kernels together,
//! captured as one CUDA graph and replayed.
//!
//! Without them, a graph with external ops runs as today's segments (see `super::external`):
//! each ggml segment is a CUDA graph of ggml's own and each external op's kernels are plain
//! launches between them. On a Qwen3.8-Flash-Next decode step over 4 V100s that was 1,121
//! eager kernels and 387 graph launches, with a 2-5 µs gap at every boundary (#814: ~3 ms of a
//! 34 ms step). A stage graph makes the same launches inside one stream capture instead, so the
//! GPU runs them back to back: the same kernels with the same arguments in the same order on
//! the same stream, so the same bits.
//!
//! ggml's CUDA graphs are off then (`GGML_CUDA_DISABLE_GRAPHS`, set by `legion serve` before
//! CUDA starts): its segments run their kernels directly, which a capture records. Stage graphs
//! follow ggml's rules for its own graphs: a graph runs directly the first time and whenever
//! its nodes changed since the last run (every node's op, shape, strides, parameters and data
//! pointers, and its sources' tensors and data pointers, as ggml-cuda compares them); the second
//! run in a row of an unchanged graph captures it, and the runs after that launch the capture.
//! A small step's graph is captured right behind its first run instead (`capture_early`, #827),
//! while the GPU runs it, so the second run launches the capture without the GPU waiting.
//!
//! What a capture may hold: a ggml segment whose nodes ggml-cuda would also capture (not a
//! `mul_mat_id` that may synchronize the stream, not a one-row `mean`, whose kernel choice
//! depends on whether ggml's graphs are on), and an external op whose launcher is a pure
//! function of its node (`set_safe_ops`: the model layer lists them). Anything else, and a late
//! input's fill (a host write), runs as before between the captured pieces, so a graph with
//! one such op becomes two captures around it rather than none.
// The planning runs on CUDA only; the CPU build keeps it for the tests.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use super::external::{ExternalOp, op_of};
use super::ffi;

#[cfg(test)]
mod tests;

pub mod cond;
pub(crate) mod copy_kernel;
pub mod kept_words;
pub(crate) mod spare;
pub mod stats;
pub(crate) mod store;
pub(crate) mod topo;

/// Whether CUDA graphs run as stage graphs (`legion serve` turns them on, #849).
static ON: AtomicBool = AtomicBool::new(false);

/// Sets of captures each device keeps by their graph's words ([`store`], #858; `legion serve
/// --stage-graph-cache`): a graph built again with the words of one seen before launches that
/// one's captures. 0: none, a graph's captures go when it changes or is unused (before #858).
pub const DEFAULT_CACHE_SETS: usize = 64;
static CACHE_SETS: AtomicUsize = AtomicUsize::new(DEFAULT_CACHE_SETS);

/// Sets how many sets of captures each device keeps by their words (see [`DEFAULT_CACHE_SETS`]).
pub fn set_cache_sets(n: usize) {
    CACHE_SETS.store(n, Ordering::Relaxed);
}

/// See [`set_cache_sets`].
pub(crate) fn cache_sets() -> usize {
    CACHE_SETS.load(Ordering::Relaxed)
}

/// Spare executable graphs each device keeps from dropped captures ([`spare`], #836; `legion
/// serve --stage-graph-spare-execs`): a new capture of the same topology updates one instead
/// of instantiating its graph. 0: none, every capture is instantiated.
pub const DEFAULT_SPARE_EXECS: usize = 8;
static SPARE_EXECS: AtomicUsize = AtomicUsize::new(DEFAULT_SPARE_EXECS);

/// Sets how many spare executable graphs each device keeps (see [`DEFAULT_SPARE_EXECS`]).
pub fn set_spare_execs(n: usize) {
    SPARE_EXECS.store(n, Ordering::Relaxed);
}

/// See [`set_spare_execs`].
pub(crate) fn spare_execs() -> usize {
    SPARE_EXECS.load(Ordering::Relaxed)
}

/// A new big capture with no spare of its topology at hand takes over the executable graph of
/// a kept capture of the same topology that was not launched for this long ([`spare`], #836;
/// `legion serve --stage-graph-exec-steal-ms`), in milliseconds. 0: never.
pub const DEFAULT_EXEC_STEAL_MS: u64 = 1000;
static EXEC_STEAL_MS: AtomicU64 = AtomicU64::new(DEFAULT_EXEC_STEAL_MS);

/// Sets [`DEFAULT_EXEC_STEAL_MS`]'s setting for the captures from now on.
pub fn set_exec_steal_ms(ms: u64) {
    EXEC_STEAL_MS.store(ms, Ordering::Relaxed);
}

/// See [`set_exec_steal_ms`].
pub(crate) fn exec_steal_ms() -> u64 {
    EXEC_STEAL_MS.load(Ordering::Relaxed)
}

/// Which external ops a capture may hold (see [`set_safe_ops`]).
static SAFE: OnceLock<SafeOp> = OnceLock::new();

/// Whether a capture may hold `node`'s launch of external op `op`: true only when the launch
/// enqueues kernels (or copies) on the compute stream alone, and what it enqueues depends on
/// nothing but the node (its sources, shape and parameters) and settings fixed at startup.
pub type SafeOp = fn(op: &ExternalOp, node: &ffi::ggml_tensor) -> bool;

/// Turns stage graphs on or off for the graphs computed from now on. On only takes effect
/// with ggml's own CUDA graphs off (`GGML_CUDA_DISABLE_GRAPHS` set before CUDA starts): a
/// capture of ggml's graph launches would nest captures.
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether stage graphs are on ([`set_enabled`]).
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Sets which external ops a capture may hold (the first call wins). Until it is called, none.
pub fn set_safe_ops(f: SafeOp) {
    let _ = SAFE.set(f);
}

fn safe_op(op: &ExternalOp, node: &ffi::ggml_tensor) -> bool {
    SAFE.get().is_some_and(|f| f(op, node))
}

/// One step of a graph's compute on CUDA, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unit {
    /// ggml computes nodes `.0..1` (a view of the graph).
    Segment(i32, i32),
    /// The external op at node `.0` launches its kernels.
    External(i32),
    /// The late input is written (by the host).
    Fill,
}

/// The units of a graph of `n` nodes with external ops at `ext` (ascending) and a late input
/// first read at node `late` (if any): exactly the segments the plain path computes
/// (`compute_with_external` in backend.rs), so ggml sees the same views and fuses the same
/// nodes. With no external op the late input is written before the whole graph, as there.
pub(crate) fn units(n: i32, ext: &[i32], late: Option<i32>) -> Vec<Unit> {
    let mut out = Vec::with_capacity(2 * ext.len() + 3);
    if ext.is_empty() {
        if late.is_some() {
            out.push(Unit::Fill);
        }
        if n > 0 {
            out.push(Unit::Segment(0, n));
        }
        return out;
    }
    let mut late = late;
    let mut start = 0;
    let segment = |out: &mut Vec<Unit>, i0: i32, i1: i32| {
        if i0 < i1 {
            out.push(Unit::Segment(i0, i1));
        }
    };
    let mut fill_before = |out: &mut Vec<Unit>, i: i32, start: &mut i32| {
        if let Some(at) = late.take_if(|at| *at <= i) {
            segment(out, *start, at);
            out.push(Unit::Fill);
            *start = at;
        }
    };
    for &i in ext {
        fill_before(&mut out, i, &mut start);
        segment(&mut out, start, i);
        out.push(Unit::External(i));
        start = i + 1;
    }
    fill_before(&mut out, n, &mut start);
    segment(&mut out, start, n);
    out
}

/// How a run of a graph's units goes: captured pieces and units run as before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Units `.0` (a range of indices) captured as one CUDA graph.
    Captured(std::ops::Range<usize>),
    /// Unit `.0` run directly.
    Direct(usize),
}

/// Groups `units` into pieces: each run of consecutive units that `capturable` allows becomes
/// one capture, unless it is a single external op (a capture of one kernel saves nothing).
pub(crate) fn pieces(units: &[Unit], capturable: impl Fn(&Unit) -> bool) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < units.len() {
        if !capturable(&units[i]) {
            out.push(Piece::Direct(i));
            i += 1;
            continue;
        }
        let start = i;
        while i < units.len() && capturable(&units[i]) {
            i += 1;
        }
        if i - start == 1 && matches!(units[start], Unit::External(_)) {
            out.push(Piece::Direct(start));
        } else {
            out.push(Piece::Captured(start..i));
        }
    }
    out
}

/// Fewer tokens than this take ggml-cuda's MMQ in a `mul_mat_id` of an [`mmq_type`] on every
/// NVIDIA GPU of compute capability 6.0+ with 48 KiB of shared memory a block
/// (`MMQ_DP4A_MAX_BATCH_SIZE`, mmq.cuh: `ggml_cuda_should_use_mmq` is true below it whatever
/// architectures ggml was compiled for).
pub(crate) const MMQ_IDS_MAX_TOKENS: i64 = 64;

/// Whether ggml-cuda's MMQ takes weights of type `t` (`ggml_cuda_should_use_mmq`, mmq.cu; the
/// types of its list that Legion loads).
pub(crate) fn mmq_type(t: ffi::ggml_type) -> bool {
    matches!(
        t,
        ffi::ggml_type_GGML_TYPE_Q4_0
            | ffi::ggml_type_GGML_TYPE_Q4_1
            | ffi::ggml_type_GGML_TYPE_Q5_0
            | ffi::ggml_type_GGML_TYPE_Q5_1
            | ffi::ggml_type_GGML_TYPE_Q8_0
            | ffi::ggml_type_GGML_TYPE_Q2_K
            | ffi::ggml_type_GGML_TYPE_Q3_K
            | ffi::ggml_type_GGML_TYPE_Q4_K
            | ffi::ggml_type_GGML_TYPE_Q5_K
            | ffi::ggml_type_GGML_TYPE_Q6_K
            | ffi::ggml_type_GGML_TYPE_IQ1_S
            | ffi::ggml_type_GGML_TYPE_IQ2_XXS
            | ffi::ggml_type_GGML_TYPE_IQ2_XS
            | ffi::ggml_type_GGML_TYPE_IQ2_S
            | ffi::ggml_type_GGML_TYPE_IQ3_XXS
            | ffi::ggml_type_GGML_TYPE_IQ3_S
            | ffi::ggml_type_GGML_TYPE_IQ4_XS
            | ffi::ggml_type_GGML_TYPE_IQ4_NL
            | ffi::ggml_type_GGML_TYPE_MXFP4
    )
}

/// Whether ggml-cuda's kernels for `node` may be captured with its graphs off: what its own
/// graphs capture (`ggml_cuda_graph_check_compability`, ggml-cuda.cu: not a `mul_mat_id` that
/// may synchronize the stream), and not a one-row `mean` (mean.cu picks its kernel by whether
/// ggml's graphs are on).
///
/// A `mul_mat_id` of quantized weights times F32 rows into F32 is captured when it has up to 4
/// tokens (mmvq on every architecture), or, with `mmq_ids` (the device takes MMQ for any
/// expert batch under [`MMQ_IDS_MAX_TOKENS`]: see `run.rs`), weights of an [`mmq_type`] and
/// fewer than [`MMQ_IDS_MAX_TOKENS`] tokens: mmvq or MMQ then, neither of which synchronizes
/// (`ggml_cuda_mul_mat_id_needs_sync`). Anything else may take the path that reads the expert
/// ids on the host, so it stays out (#827: on a GB10 every 9-token MTP verify step's MoE
/// segments ran directly, ~1,000 kernel gaps of 2 µs a step).
///
/// # Safety
/// `node` is a live tensor whose sources are live tensors.
pub(crate) unsafe fn ggml_capturable(node: *const ffi::ggml_tensor, mmq_ids: bool) -> bool {
    // SAFETY: the caller's.
    unsafe {
        let n = &*node;
        match n.op {
            ffi::ggml_op_GGML_OP_MEAN => false,
            ffi::ggml_op_GGML_OP_MUL_MAT_ID => {
                let (w, x) = (n.src[0], n.src[1]);
                if w.is_null()
                    || x.is_null()
                    || !ffi::ggml_is_quantized((*w).type_)
                    || (*x).type_ != ffi::ggml_type_GGML_TYPE_F32
                    || n.type_ != ffi::ggml_type_GGML_TYPE_F32
                {
                    return false;
                }
                n.ne[2] <= 4
                    || (mmq_ids
                        && mmq_type((*w).type_)
                        && (*x).ne[2] < MMQ_IDS_MAX_TOKENS
                        && n.ne[2] < MMQ_IDS_MAX_TOKENS)
            }
            ffi::ggml_op_GGML_OP_CUSTOM => false,
            _ => true,
        }
    }
}

/// Appends what a run of `g` depends on to `out`: per node its op, type, flags, shape,
/// strides, parameters and data, and each source's data, type, shape and strides. Two runs of
/// a graph with the same words launch the same kernels with the same arguments (for the
/// external ops [`SafeOp`] allows): what ggml-cuda compares for its own graphs
/// (`ggml_cuda_graph_update_required`), plus the sources' layouts.
///
/// # Safety
/// `g` is a live graph whose nodes and sources are live tensors.
pub(crate) unsafe fn props(g: *mut ffi::ggml_cgraph, out: &mut Vec<u64>) {
    // SAFETY: the caller's.
    unsafe { each_node(g, false, |w| out.extend_from_slice(w)) }
}

/// Whether a run of `g` has the same words as `last`, the last run's, which it leaves holding
/// this run's: [`props`] (or with `nodes`, its shorter form, below), then `tail`. The same
/// answer and words as building this run's words and comparing them, without writing them
/// while they agree: an unchanged graph (every decode step's) costs a read of its nodes and of
/// `last`, no copy (#849).
///
/// With `nodes` (only for a graph [`srcs_are_nodes`] holds for): each node's own address
/// leads its words, and a source that is a node of `g` is its address alone, since that
/// node's own words (taken in the same walk) already hold its data, type, shape and strides.
/// The other sources (the graph's inputs and weights) keep their words. A node list with the
/// same addresses and words has the same sources then, so it is the same check with about
/// half the words. Each form starts with its own tag, so a change of form is a change.
///
/// # Safety
/// As for [`props`].
pub(crate) unsafe fn props_same(
    g: *mut ffi::ggml_cgraph,
    last: &mut Vec<u64>,
    tail: u64,
    nodes: bool,
) -> bool {
    // SAFETY: the caller's.
    unsafe { props_same_diff(g, last, tail, nodes).0 }
}

/// Where a run's words first differ from the last run's ([`props_same_diff`], for
/// `--stage-graph-stats`): the node (0: the node count word, then node `i` is `i + 1`; past
/// the nodes: the late input's place), the word within that node's words, and the old and new
/// word (old 0 past the last run's words).
pub(crate) type Diff = (usize, usize, u64, u64);

/// [`props_same`], and where the words first differed (if they did).
///
/// # Safety
/// As for [`props`].
pub(crate) unsafe fn props_same_diff(
    g: *mut ffi::ggml_cgraph,
    last: &mut Vec<u64>,
    tail: u64,
    nodes: bool,
) -> (bool, Option<Diff>) {
    let mut w = Words {
        last,
        at: 0,
        same: true,
        calls: 0,
        diff: None,
    };
    // SAFETY: the caller's.
    unsafe { each_node(g, nodes, |words| w.put(words)) };
    w.put(&[tail]);
    let diff = w.diff;
    (w.finish(), diff)
}

/// Whether every source of `g`'s nodes that is computed (not a leaf: an op other than none)
/// is itself one of `g`'s nodes, as in any graph ggml builds (`ggml_build_forward_expand`
/// adds every computed ancestor), so [`props_same`] may take the shorter form.
///
/// # Safety
/// As for [`props`].
pub(crate) unsafe fn srcs_are_nodes(g: *mut ffi::ggml_cgraph) -> bool {
    // SAFETY: the caller's; node indices in range.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(g);
        let set: std::collections::HashSet<usize> = (0..n)
            .map(|i| ffi::ggml_graph_node(g, i) as usize)
            .collect();
        (0..n).all(|i| {
            let t = &*ffi::ggml_graph_node(g, i);
            t.src.iter().all(|&s| {
                s.is_null() || (*s).op == ffi::ggml_op_GGML_OP_NONE || set.contains(&(s as usize))
            })
        })
    }
}

/// This run's words written over the last run's ([`props_same`]).
struct Words<'a> {
    last: &'a mut Vec<u64>,
    /// How many of this run's words were put so far.
    at: usize,
    /// They all equal the last run's so far (else `last` holds this run's up to `at`).
    same: bool,
    /// Calls of [`Self::put`] so far (one per node after the count).
    calls: usize,
    /// The first difference ([`Diff`]).
    diff: Option<Diff>,
}

impl Words<'_> {
    fn put(&mut self, words: &[u64]) {
        if self.same {
            let end = self.at + words.len();
            if self.last.get(self.at..end) == Some(words) {
                self.at = end;
                self.calls += 1;
                return;
            }
            let old = self.last.get(self.at..).unwrap_or(&[]);
            let k = (0..words.len())
                .find(|&k| old.get(k) != Some(&words[k]))
                .unwrap_or(words.len());
            self.diff = Some((
                self.calls,
                k,
                old.get(k).copied().unwrap_or(0),
                words.get(k).copied().unwrap_or(0),
            ));
            self.same = false;
            self.last.truncate(self.at);
        }
        self.last.extend_from_slice(words);
        self.at += words.len();
        self.calls += 1;
    }

    fn finish(self) -> bool {
        // A run with fewer words is a change too.
        let same = self.same && self.at == self.last.len();
        self.last.truncate(self.at);
        same
    }
}

/// Calls `f` with the words of `g` ([`props`]): its node count, then each node's. With
/// `nodes`, their shorter form ([`props_same`]), tagged by a word no full form has second (a
/// node's first word holds its op and type, far below it).
///
/// # Safety
/// As for [`props`].
unsafe fn each_node(g: *mut ffi::ggml_cgraph, nodes: bool, mut f: impl FnMut(&[u64])) {
    let mut out = NodeWords {
        w: [0; NODE_WORDS],
        n: 0,
    };
    // SAFETY: the caller's; node indices in range.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(g);
        if nodes {
            f(&[n as u64, u64::MAX]);
        } else {
            f(&[n as u64]);
        }
        for i in 0..n {
            out.n = 0;
            let t = ffi::ggml_graph_node(g, i);
            if nodes {
                out.push(t as u64);
            }
            let t = &*t;
            out.push(t.op as u64 | (t.type_ as u64) << 32);
            out.push(t.flags as u32 as u64);
            out.extend(t.ne.iter().map(|&v| v as u64));
            out.extend(t.nb.iter().map(|&v| v as u64));
            out.extend(
                t.op_params
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[a, b]| a as u32 as u64 | (b as u32 as u64) << 32),
            );
            out.push(t.data as u64);
            let used = t
                .src
                .iter()
                .rposition(|s| !s.is_null())
                .map_or(0, |k| k + 1);
            out.push(used as u64);
            for &s in &t.src[..used] {
                if s.is_null() {
                    out.push(0);
                    continue;
                }
                // Where the source is and how it is laid out: a node's own words cover a node,
                // these the graph's inputs and weights too (which the external ops read). In
                // the shorter form a source that is a node is that node.
                if nodes {
                    // Its address, the top bit set for a computed one (no user-space address
                    // has it), so the words say which sources have more words after them.
                    let computed = (*s).op != ffi::ggml_op_GGML_OP_NONE;
                    out.push(s as u64 | (computed as u64) << 63);
                    if computed {
                        continue;
                    }
                }
                let s = &*s;
                out.push(s.data as u64);
                out.push(s.type_ as u64);
                out.extend(s.ne.iter().map(|&v| v as u64));
                out.extend(s.nb.iter().map(|&v| v as u64));
            }
            f(&out.w[..out.n]);
        }
    }
}

/// The most words one node has ([`each_node`]): its address, op and type, flags, shape,
/// strides, parameters, data and source count, then per source its address, data, type,
/// shape and strides.
const NODE_WORDS: usize = 1
    + 2
    + 4
    + 4
    + ffi::GGML_MAX_OP_PARAMS as usize / 8
    + 2
    + ffi::GGML_MAX_SRC as usize * (1 + 2 + 4 + 4);

/// One node's words, built in place (no allocation, no growth check).
struct NodeWords {
    w: [u64; NODE_WORDS],
    n: usize,
}

impl NodeWords {
    #[inline(always)]
    fn push(&mut self, x: u64) {
        self.w[self.n] = x;
        self.n += 1;
    }

    #[inline(always)]
    fn extend(&mut self, xs: impl Iterator<Item = u64>) {
        xs.for_each(|x| self.push(x));
    }
}

/// Whether unit `u` of graph `g` may be captured (`mmq_ids` as for [`ggml_capturable`]).
///
/// # Safety
/// As for [`props`].
unsafe fn unit_capturable(g: *mut ffi::ggml_cgraph, u: &Unit, mmq_ids: bool) -> bool {
    // SAFETY: the caller's; node indices in range.
    unsafe {
        match *u {
            Unit::Segment(i0, i1) => {
                (i0..i1).all(|i| ggml_capturable(ffi::ggml_graph_node(g, i), mmq_ids))
            }
            Unit::External(i) => {
                let node = ffi::ggml_graph_node(g, i);
                op_of(node).is_some_and(|op| safe_op(op, &*node))
            }
            Unit::Fill => false,
        }
    }
}

/// The pieces of a run of `g` made of `units` (`mmq_ids` as for [`ggml_capturable`]), big
/// captures split into a head and the rest ([`split_heads`]) unless that is off.
///
/// # Safety
/// As for [`props`].
pub(crate) unsafe fn plan(g: *mut ffi::ggml_cgraph, units: &[Unit], mmq_ids: bool) -> Vec<Piece> {
    // SAFETY: the caller's.
    let p = pieces(units, |u| unsafe { unit_capturable(g, u, mmq_ids) });
    if split_enabled() {
        split_heads(p, units)
    } else {
        p
    }
}

/// Whether big captures are launched as a short head and the rest ([`split_heads`]; on by
/// default, `legion serve --no-stage-graph-split` turns it off).
static SPLIT: AtomicBool = AtomicBool::new(true);

/// Turns the head split of big captures ([`split_heads`]) on or off for graphs planned from now
/// on.
pub fn set_split(on: bool) {
    SPLIT.store(on, Ordering::Relaxed);
}

fn split_enabled() -> bool {
    SPLIT.load(Ordering::Relaxed)
}

/// Whether a small step's new graph is captured right behind its first, direct run
/// ([`capture_early`]; on by default, `legion serve --no-stage-graph-early-capture` turns it
/// off).
static EARLY: AtomicBool = AtomicBool::new(true);

/// Turns early captures ([`capture_early`]) on or off for graphs run from now on.
pub fn set_early(on: bool) {
    EARLY.store(on, Ordering::Relaxed);
}

/// Whether the graphs computed now belong to a small step ([`EarlyScope`]).
static SMALL_STEP: AtomicBool = AtomicBool::new(false);

/// Marks the graphs computed while it lives as a small step's (a decode, verify or MTP draft
/// pass: a few tokens a sequence) or not; the mark before it comes back when it is dropped.
/// Only a small step's graphs are captured early ([`capture_early`]): a prompt's graphs run
/// once each, so capturing one would cost its host time for nothing.
pub struct EarlyScope(bool);

impl EarlyScope {
    pub fn enter(small: bool) -> Self {
        Self(SMALL_STEP.swap(small, Ordering::Relaxed))
    }
}

impl Drop for EarlyScope {
    fn drop(&mut self) {
        SMALL_STEP.store(self.0, Ordering::Relaxed);
    }
}

/// Whether a graph's first run after a change (run directly) also captures it at once, so that
/// its second run launches the capture instead of capturing first (#827). The capture only
/// records (the kernels the direct run queued keep the GPU busy meanwhile); captured on the
/// second run, the GPU waited for it: 10-16 ms for each new verify-step shape of
/// Qwen3.8-Flash-Next with MTP on a GB10, at a request's start and whenever the context grows
/// past a cache padding step. The same units recorded on the same stream as on the second
/// run, so the same kernels with the same arguments: the same bits.
///
/// Early only for a small step's graph (`small`, [`EarlyScope`]) when the setting is on
/// (`on`), on a device whose small steps capture whole (`mmq_ids`, as for
/// [`ggml_capturable`]: older GPUs keep their captures as they were), and when the graph is
/// new to its cache entry (`fresh`) or its last words ran at least twice in a row (`last_same`
/// of 1 or more, [`compute`]'s count): a graph whose words change every run is never captured,
/// early or not.
pub(crate) fn capture_early(
    on: bool,
    small: bool,
    mmq_ids: bool,
    fresh: bool,
    last_same: u32,
) -> bool {
    on && small && mmq_ids && (fresh || last_same >= 1)
}

/// [`capture_early`] with the setting and the step mark as they are now; never for a graph
/// marked to wait for its reuse ([`ReuseScope`]), which [`second_now`] captures instead.
fn early_now(mmq_ids: bool, fresh: bool, last_same: u32) -> bool {
    !reuse_marked()
        && capture_early(
            EARLY.load(Ordering::Relaxed),
            SMALL_STEP.load(Ordering::Relaxed),
            mmq_ids,
            fresh,
            last_same,
        )
}

/// Whether graphs marked by a [`ReuseScope`] wait for their second run to be captured
/// ([`capture_on_reuse`]; on by default, `legion serve --no-stage-graph-capture-on-reuse`
/// turns it off).
static REUSE: AtomicBool = AtomicBool::new(true);

/// Turns the capture on reuse ([`capture_on_reuse`]) on or off for graphs run from now on.
pub fn set_reuse_capture(on: bool) {
    REUSE.store(on, Ordering::Relaxed);
}

/// Whether the graphs computed now are marked to wait for their reuse ([`ReuseScope`]).
static REUSE_STEP: AtomicBool = AtomicBool::new(false);

/// Marks the graphs computed while it lives as ones whose shape seldom comes back (an MTP
/// take-in pass of several sequences: its rows follow each one's accepted count), or not; the
/// mark before it comes back when it is dropped. A marked graph is captured behind its second
/// run rather than its first ([`capture_on_reuse`]).
pub struct ReuseScope(bool);

impl ReuseScope {
    pub fn enter(marked: bool) -> Self {
        Self(REUSE_STEP.swap(marked, Ordering::Relaxed))
    }
}

impl Drop for ReuseScope {
    fn drop(&mut self) {
        REUSE_STEP.store(self.0, Ordering::Relaxed);
    }
}

/// The setting on and the graphs computed now marked ([`ReuseScope`]).
fn reuse_marked() -> bool {
    REUSE.load(Ordering::Relaxed) && REUSE_STEP.load(Ordering::Relaxed)
}

/// Whether a graph that runs again unchanged without captures is run directly and captured
/// right behind that run (#836), rather than captured first with the GPU waiting. Only for a
/// graph marked to wait for its reuse (`marked`, [`ReuseScope`]), where an early capture would
/// apply ([`capture_early`]'s `on`, `small` and `mmq_ids`), on a run that has the same words as
/// the one before (`same` of 1 or more, [`compute`]'s count), no set of captures (`has_set`),
/// no refused capture (`refused`) and no device condition to run behind (`cond`, which only a
/// capture can hold).
///
/// The MTP take-in pass of several sequences takes in rows by each one's accepted count, so
/// its shapes are the combinations of those counts: at 4 parallel requests drafting 8 tokens on
/// a GB10 nearly every round built a new take-in graph and captured it behind its first run,
/// 2.6 CUDA graph instantiations (1.55 ms of host time) per round for graphs that mostly never
/// ran again. Marked, a new one runs directly once (the kernels it launches are the capture's),
/// and only a shape that comes back is captured. The same units on the same stream either
/// way: the same kernels with the same arguments, so the same bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_on_reuse(
    on: bool,
    small: bool,
    mmq_ids: bool,
    marked: bool,
    same: u32,
    has_set: bool,
    refused: bool,
    cond: bool,
) -> bool {
    on && small && mmq_ids && marked && same >= 1 && !has_set && !refused && !cond
}

/// [`capture_on_reuse`] with the settings and the step marks as they are now.
fn second_now(mmq_ids: bool, same: u32, has_set: bool, refused: bool, cond: bool) -> bool {
    capture_on_reuse(
        EARLY.load(Ordering::Relaxed),
        SMALL_STEP.load(Ordering::Relaxed),
        mmq_ids,
        reuse_marked(),
        same,
        has_set,
        refused,
        cond,
    )
}

/// Captures of fewer nodes than this stay whole ([`split_heads`]).
pub(crate) const SPLIT_MIN_NODES: i32 = 512;

/// A split capture's head holds at least 1 / `HEAD_DIV` of its nodes ([`split_heads`]).
pub(crate) const HEAD_DIV: i32 = 32;

/// The graph nodes unit `u` computes.
fn unit_nodes(u: &Unit) -> i32 {
    match *u {
        Unit::Segment(i0, i1) => i1 - i0,
        Unit::External(_) => 1,
        Unit::Fill => 0,
    }
}

/// Splits each capture of [`SPLIT_MIN_NODES`] nodes or more into two captures: a head of the
/// first units holding at least 1 / [`HEAD_DIV`] of its nodes, then the rest. The GPU starts
/// a CUDA graph only some way into its launch, and that delay grows with the graph: on a GB10,
/// a Qwen3.8-Flash-Next decode step's capture after the late n-gram fill (~3,400 kernels) began
/// 1.1-1.9 ms into its `cudaGraphLaunch` (#836), with the GPU idle. A short head starts at
/// once, and the rest is launched while it runs. The same units in the same order on the same
/// stream, so the same kernels with the same arguments: the same bits.
pub(crate) fn split_heads(pieces: Vec<Piece>, units: &[Unit]) -> Vec<Piece> {
    let mut out = Vec::with_capacity(pieces.len() + 2);
    for p in pieces {
        let Piece::Captured(r) = &p else {
            out.push(p);
            continue;
        };
        let total: i32 = units[r.clone()].iter().map(unit_nodes).sum();
        if total < SPLIT_MIN_NODES {
            out.push(p);
            continue;
        }
        let want = (total + HEAD_DIV - 1) / HEAD_DIV;
        let mut acc = 0;
        let mut end = r.start;
        while end < r.end && acc < want {
            acc += unit_nodes(&units[end]);
            end += 1;
        }
        if end < r.end {
            out.push(Piece::Captured(r.start..end));
            out.push(Piece::Captured(end..r.end));
        } else {
            out.push(p);
        }
    }
    out
}

#[cfg(feature = "cuda")]
pub(super) use run::{compute, precapture};

#[cfg(feature = "cuda")]
mod run;
