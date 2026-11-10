//! Post-allocation node fusion (#814; on by default, `legion serve --node-fuse-ggml` turns it
//! off): after a CUDA graph is allocated ([`crate::ggml::post_alloc`]), chains of ggml's small
//! nodes become one external op each, with ggml's arithmetic, so a step launches fewer kernels
//! and the text stays the default's bit for bit.
//!
//! The graph is allocated as built, so ggml-cuda's layout-dependent MoE weighted-sum fusion
//! (#410: one FMA per expert when the sum's output does not overlap its inputs in memory, the
//! `mul` and an add per expert otherwise) makes the same choice per layer as without the pass,
//! and the pass reads that choice off the tensors' addresses ([`ggml_fuses_sum`]):
//!
//! - A MoE layer's weighted expert sum that ggml-cuda runs unfused (the `mul` by the router
//!   weights and an add per expert: ten kernels for ten experts) runs as one op that rounds the
//!   same way ([`MOE_SUM_GATE`]), with the shared expert's gate (`sigmoid`, `mul`, `add`) added
//!   in the same kernel where the inputs' memory allows. The sums ggml-cuda fuses stay its own.
//! - The shared expert's gate behind a sum ggml-cuda fuses: one op instead of three kernels.
//! - Where qwen4exp's next hyper-connection norm (the rebuilt BF16-xn form of prompt steps on an
//!   sm_80+ GPU) is the only reader of a MoE tail's output (the weighted sum and the shared
//!   expert's gate), the norm makes those values itself from the tail's inputs, rounded as the
//!   tail's nodes round them in the graph as built, and the tail's nodes go ([`fold`], #836).
//!   That rewrite keeps the tail's inputs alive until the norm, so the graph is planned again
//!   after it (`crate::ggml::post_alloc::set_relayout`), before this pass runs.
//! - The routed expert ids through the layer's expert table (the expert cache's remap: a `cont`
//!   and a `get_rows` of I32 ids): one op ([`IDS_REMAP`]).
//!
//! An op runs at the position of the chain's last node, into that node's memory; the chain's
//! other nodes become `GGML_OP_NONE`. An op reading a tensor later than the chain did is only
//! made where no node in between writes or reads that tensor's memory (the allocator may give
//! a tensor's memory to another once its last reader ran), and its output shares no memory with
//! its inputs but element for element. Anything that does not fit stays as it was.

use std::collections::HashMap;
use std::os::raw::c_int;
#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::ggml::{ExternalOp, external_params, ffi, make_external};

pub(crate) mod fold;
#[cfg(test)]
mod tests;

type Node = *mut ffi::ggml_tensor;

/// A MoE layer's weighted expert sum with ggml's unfused rounding, and its shared expert gated
/// by the sigmoid added: `src[0]` the experts' outputs `[e, k, n]` (contiguous), then the
/// router weights `[1, k, n]` when `params[0]` (else `src[0]` is weighted already), then the
/// shared expert's output `[e, n]` (contiguous) and the gate logits `[1, n]` (any row stride)
/// when `params[1]`; `params[2]` is `k`. With `k` 1 and no weights, the gate alone.
/// `params[3]` set (weights and gate both there; made by [`fold`], #1176): the sum rounded as
/// ggml-cuda's fused kernel rounds it (the first product, then one FMA per expert). With
/// `params[3]` [`FMA_SKIP`] (only under `--tp-moe-ep`) a row whose router weight is zero is not
/// read, its term made from a zero: the same value where that row holds +0.0, as the rows of
/// the experts another rank holds there; otherwise every row is read, as ggml-cuda reads them.
/// `params[3]` of a [`MOE_SUM_GATE`] in its fused-sum form that skips the rows of zero router
/// weights (`1` reads them all).
pub(crate) const FMA_SKIP: i32 = 2;

/// The parameter of a [`MOE_SUM_GATE`] in its fused-sum form that the reduce-scatter reading
/// it sets ([`crate::tp_net::seq_par::take_tail`], #1176): the node launches nothing, and the
/// exchange runs the tail itself ([`launch_tail`]), its values rounded to BF16 straight into
/// the send buffer.
pub(crate) const P_BY_EXCHANGE: usize = 4;

pub(crate) static MOE_SUM_GATE: ExternalOp = ExternalOp {
    name: "moe_sum_gate",
    cpu: cpu_moe_sum_gate,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_moe_sum_gate),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// I32 ids (`src[1]`, `[a, b]`, any strides) through a table (`src[0]`, one I32 per row, any
/// row stride): `out[i] = table[ids[i]]`, `[1, a * b]`, as ggml's `get_rows(table, cont(ids))`.
pub(crate) static IDS_REMAP: ExternalOp = ExternalOp {
    name: "ids_remap",
    cpu: cpu_ids_remap,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ids_remap),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs allocated from now on are rewritten (on by default).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether the CUDA graphs allocated from now on have their small-node chains fused (on
/// by default; `legion serve --node-fuse-ggml` turns it off, for comparisons).
pub fn set_node_fuse(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// What one rewrite did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
    /// Unfused weighted sums with their gate as one op (the `mul` too).
    pub sum_gate: usize,
    /// Unfused weighted sums of an already weighted tensor, with their gate.
    pub weighted_gate: usize,
    /// Unfused weighted sums alone.
    pub sum: usize,
    /// Gates alone (behind a sum ggml-cuda fuses, or one done alone).
    pub gate: usize,
    /// Expert id remaps.
    pub remap: usize,
    /// Nodes turned into `GGML_OP_NONE`.
    pub none: usize,
}

/// Graphs whose rewrite was logged (the first few only).
static LOGGED: AtomicUsize = AtomicUsize::new(0);

/// The pass on CUDA graphs ([`crate::ggml::post_alloc::set_rewrite`]).
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
pub unsafe fn rewrite(graph: *mut ffi::ggml_cgraph) {
    // With ggml-cuda's fusions off, every sum runs unfused; left alone rather than modelled.
    if !ON.load(Ordering::Relaxed) || std::env::var_os("GGML_CUDA_DISABLE_FUSION").is_some() {
        return;
    }
    // SAFETY: the caller's.
    let c = unsafe { rewrite_graph(graph, true) };
    if c != Counts::default() && LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
        tracing::info!(
            sum_gate = c.sum_gate,
            weighted_gate = c.weighted_gate,
            sum = c.sum,
            gate = c.gate,
            remap = c.remap,
            none = c.none,
            "node fuse: small ggml chains as Legion ops (the first graphs; --node-fuse-ggml off)"
        );
    }
}

/// Rewrites `graph` (see the module doc). `cuda`: ggml-cuda computes it, so the weighted sums
/// it fuses are left to it; on the CPU none is fused.
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
pub(crate) unsafe fn rewrite_graph(graph: *mut ffi::ggml_cgraph, cuda: bool) -> Counts {
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n).to_vec();
        let mut g = Graph::new(graph, nodes);
        let mut c = Counts::default();
        for i in 0..n {
            if g.none[i] {
                continue;
            }
            match (*g.nodes[i]).op {
                ffi::ggml_op_GGML_OP_MUL => g.moe(i, cuda, &mut c),
                ffi::ggml_op_GGML_OP_GET_ROWS => g.remap(i, &mut c),
                _ => {}
            }
        }
        c.none = g.none.iter().filter(|&&x| x).count();
        c
    }
}

/// A hasher for tensor addresses: one multiply, the well-mixed high bits rotated down
/// (rustc-hash's FxHasher, MIT/Apache-2.0: its multiply-and-rotate finish). The default
/// SipHash cost most of the pass's ~0.7 ms on a step graph of ~4,700 nodes.
#[derive(Default)]
struct AddrHasher(u64);

impl std::hash::Hasher for AddrHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(ADDR_K);
        }
    }

    fn write_usize(&mut self, x: usize) {
        self.0 = (self.0 ^ x as u64).wrapping_mul(ADDR_K);
    }

    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

/// rustc-hash's multiplier.
const ADDR_K: u64 = 0xf135_7aea_2e62_a9c5;

/// Node indices by tensor address.
type AddrMap = HashMap<usize, usize, std::hash::BuildHasherDefault<AddrHasher>>;

/// A graph's nodes with who reads what.
struct Graph {
    raw: *mut ffi::ggml_cgraph,
    nodes: Vec<Node>,
    /// Node index by tensor.
    index: AddrMap,
    /// The nodes reading node `i` (by index, ascending; once per source slot) are
    /// `readers[starts[i]..starts[i + 1]]`. Only nodes' readers are asked for.
    starts: Vec<usize>,
    readers: Vec<usize>,
    /// Nodes this rewrite turned into `GGML_OP_NONE`.
    none: Vec<bool>,
}

/// Whether `op` only makes a view (ggml-cuda launches nothing for it).
fn is_view_op(op: ffi::ggml_op) -> bool {
    matches!(
        op,
        ffi::ggml_op_GGML_OP_VIEW
            | ffi::ggml_op_GGML_OP_RESHAPE
            | ffi::ggml_op_GGML_OP_PERMUTE
            | ffi::ggml_op_GGML_OP_TRANSPOSE
    )
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

/// The bytes `t` spans (0 without memory).
///
/// # Safety
/// `t` is a valid tensor.
unsafe fn span(t: Node) -> (usize, usize) {
    // SAFETY: the caller's.
    unsafe {
        let p = (*t).data as usize;
        if p == 0 {
            return (0, 0);
        }
        (p, p + ffi::ggml_nbytes(t))
    }
}

/// Whether `a` and `b` share memory.
///
/// # Safety
/// Valid tensors.
unsafe fn overlap(a: Node, b: Node) -> bool {
    // SAFETY: the caller's.
    let ((a0, a1), (b0, b1)) = unsafe { (span(a), span(b)) };
    a0 < a1 && b0 < b1 && a0 < b1 && b0 < a1
}

/// F32 and contiguous.
///
/// # Safety
/// A valid tensor.
unsafe fn f32_contiguous(t: Node) -> bool {
    // SAFETY: the caller's.
    unsafe {
        !t.is_null() && (*t).type_ == ffi::ggml_type_GGML_TYPE_F32 && ffi::ggml_is_contiguous(t)
    }
}

unsafe extern "C" {
    /// ggml-impl.h: whether nodes `node_idxs` of `cgraph` (of ops `ops`) may run as one fused
    /// kernel writing `outputs` (their other nodes read nowhere else, flagged as computed, not
    /// graph outputs).
    fn ggml_can_fuse_subgraph_ext(
        cgraph: *const ffi::ggml_cgraph,
        node_idxs: *const c_int,
        count: c_int,
        ops: *const ffi::ggml_op,
        outputs: *const c_int,
        num_outputs: c_int,
    ) -> bool;
}

/// A matched weighted sum: `mul` at `at`, its `k` views and adds after it, the last at
/// `at + 2k - 1`.
struct Sum {
    at: usize,
    k: usize,
    experts: Node,
    weights: Node,
    last: usize,
}

/// A matched gate: `add(moe, mul(sh, sigmoid(g)))`, the add at `add` reading `moe`.
struct Gate {
    add: usize,
    sh: Node,
    g: Node,
    /// The `mul`, the `sigmoid` and the `cont` of `g` (if any), by index.
    chain: Vec<usize>,
}

impl Graph {
    /// # Safety
    /// `nodes` are `raw`'s, valid tensors.
    unsafe fn new(raw: *mut ffi::ggml_cgraph, nodes: Vec<Node>) -> Self {
        let n = nodes.len();
        let mut index = AddrMap::with_capacity_and_hasher(n, Default::default());
        for (i, &t) in nodes.iter().enumerate() {
            index.insert(t as usize, i);
        }
        // Each (read node, reader) pair, readers ascending, then grouped by the read node.
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(3 * n);
        let mut count = vec![0usize; n + 1];
        for (i, &t) in nodes.iter().enumerate() {
            // SAFETY: the caller's.
            for &s in unsafe { (*t).src.iter() } {
                if s.is_null() {
                    continue;
                }
                if let Some(&j) = index.get(&(s as usize)) {
                    pairs.push((j, i));
                    count[j + 1] += 1;
                }
            }
        }
        let mut sum = 0;
        for c in &mut count {
            sum += *c;
            *c = sum;
        }
        let starts = count.clone();
        let mut readers = vec![0usize; pairs.len()];
        for (j, i) in pairs {
            readers[count[j]] = i;
            count[j] += 1;
        }
        Self {
            raw,
            nodes,
            index,
            starts,
            readers,
            none: vec![false; n],
        }
    }

    /// The nodes reading `t`, a node of the graph (none for any other tensor).
    fn users(&self, t: Node) -> &[usize] {
        self.index
            .get(&(t as usize))
            .map_or(&[], |&i| &self.readers[self.starts[i]..self.starts[i + 1]])
    }

    /// Whether every read of `t` (directly or through views) is by a node in `allowed`, and `t`
    /// is no graph output.
    ///
    /// # Safety
    /// `t` is a valid tensor.
    unsafe fn only_read_by(&self, t: Node, allowed: &[usize]) -> bool {
        // SAFETY: the caller's; users are valid nodes.
        unsafe {
            if (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0 {
                return false;
            }
            self.users(t).iter().all(|&u| {
                allowed.contains(&u)
                    || (is_view_op((*self.nodes[u]).op)
                        && self.only_read_by(self.nodes[u], allowed))
            })
        }
    }

    /// Whether an op at node `at` writing `out` may read `inputs` there: no node after an
    /// input is made and before `at` (but those in `skip`, about to read nothing) touches its
    /// memory, and `out` shares none with it, but for an input flagged element for element
    /// (read at the index it writes, in the same thread) that is exactly `out`'s memory.
    ///
    /// # Safety
    /// Valid tensors of the graph.
    unsafe fn may_read_at(
        &self,
        at: usize,
        out: Node,
        inputs: &[(Node, bool)],
        skip: &[usize],
    ) -> bool {
        // SAFETY: the caller's.
        unsafe {
            for &(x, same) in inputs {
                if overlap(out, x) && !(same && span(out) == span(x)) {
                    return false;
                }
                let r = root(x);
                let from = self.index.get(&(r as usize)).map_or(0, |&i| i + 1);
                for j in from..at {
                    if skip.contains(&j) || self.none[j] {
                        continue;
                    }
                    let n = self.nodes[j];
                    if is_view_op((*n).op) || (*n).op == ffi::ggml_op_GGML_OP_NONE {
                        continue;
                    }
                    if overlap(n, x) {
                        return false;
                    }
                    for &s in (*n).src.iter() {
                        if !s.is_null() && root(s) != r && overlap(s, x) {
                            return false;
                        }
                    }
                }
            }
            true
        }
    }

    /// The weighted sum starting at the `mul` at `at`, matched as ggml-cuda's
    /// `ggml_cuda_match_moe_weighted_reduction` matches its short form (`experts * weights`,
    /// then per expert a view, and from the second on an add).
    ///
    /// # Safety
    /// `at` is a node index.
    unsafe fn match_sum(&self, at: usize) -> Option<Sum> {
        // SAFETY: the caller's; sources are valid tensors.
        unsafe {
            let m = self.nodes[at];
            if !f32_contiguous(m) {
                return None;
            }
            let is_weights = |t: Node| {
                f32_contiguous(t)
                    && (*t).ne[0] == 1
                    && (*t).ne[1] == (*m).ne[1]
                    && (*t).ne[2] == (*m).ne[2]
                    && (*t).ne[3] == (*m).ne[3]
            };
            let is_experts = |t: Node| f32_contiguous(t) && ffi::ggml_are_same_shape(t, m);
            let (a, b) = ((*m).src[0], (*m).src[1]);
            let (experts, weights) = if is_experts(a) && is_weights(b) {
                (a, b)
            } else if is_experts(b) && is_weights(a) {
                (b, a)
            } else {
                return None;
            };
            // The scaled form (`mul(mul(experts, scale), weights)`) is left alone.
            if let Some(&next) = self.nodes.get(at + 1)
                && (*next).op == ffi::ggml_op_GGML_OP_MUL
                && ((*next).src[0] == m || (*next).src[1] == m)
            {
                return None;
            }
            let k = (*m).ne[1] as usize;
            let n_tok = (*m).ne[2] * (*m).ne[3];
            if !(2..=64).contains(&k) || at + 2 * k > self.nodes.len() {
                return None;
            }
            let mut views: Vec<Node> = Vec::with_capacity(k);
            let mut previous: Node = std::ptr::null_mut();
            let mut adds = 0;
            for off in 1..2 * k {
                let c = self.nodes[at + off];
                if self.none[at + off] {
                    return None;
                }
                if (*c).op == ffi::ggml_op_GGML_OP_VIEW {
                    let e = views.len();
                    if e >= k
                        || (*c).src[0] != m
                        || (*c).view_src != m
                        || (*c).type_ != ffi::ggml_type_GGML_TYPE_F32
                        || (*c).ne[0] != (*m).ne[0]
                        || (*c).ne[1] != n_tok
                        || (*c).ne[2] != 1
                        || (*c).ne[3] != 1
                        || (*c).nb[0] != (*m).nb[0]
                        || (*c).nb[1] != (*m).nb[2]
                        || (*c).view_offs != e * (*m).nb[1]
                    {
                        return None;
                    }
                    views.push(c);
                    continue;
                }
                if (*c).op != ffi::ggml_op_GGML_OP_ADD || views.len() < 2 || adds + 1 >= views.len()
                {
                    return None;
                }
                let lhs = if adds == 0 { views[0] } else { previous };
                if (*c).src[0] != lhs
                    || (*c).src[1] != views[adds + 1]
                    || (*c).type_ != ffi::ggml_type_GGML_TYPE_F32
                {
                    return None;
                }
                previous = c;
                adds += 1;
            }
            if views.len() != k || adds != k - 1 || previous.is_null() {
                return None;
            }
            if !ffi::ggml_is_contiguous(previous)
                || (*previous).ne[0] != (*m).ne[0]
                || (*previous).ne[1] != n_tok
                || (*previous).ne[2] != 1
                || (*previous).ne[3] != 1
            {
                return None;
            }
            Some(Sum {
                at,
                k,
                experts,
                weights,
                last: at + 2 * k - 1,
            })
        }
    }

    /// Whether ggml-cuda fuses `s` into its one kernel: its fusion check
    /// (`ggml_can_fuse_subgraph`), at most 15 experts, and its memory check
    /// (`ggml_cuda_check_fusion_memory_ranges`: the output shares no memory, by the buffer
    /// type's allocation size, with a source of the chain that is not one of the chain's nodes
    /// before it).
    ///
    /// # Safety
    /// `s` is a match of this graph.
    unsafe fn ggml_fuses_sum(&self, s: &Sum) -> bool {
        // SAFETY: the caller's; nodes and their sources are valid tensors.
        unsafe {
            let count = 2 * s.k;
            if s.k > 15 {
                return false;
            }
            let idxs: Vec<c_int> = (s.at..s.at + count).map(|i| i as c_int).collect();
            let ops: Vec<ffi::ggml_op> =
                (s.at..s.at + count).map(|i| (*self.nodes[i]).op).collect();
            let out = s.last as c_int;
            if !ggml_can_fuse_subgraph_ext(
                self.raw,
                idxs.as_ptr(),
                count as c_int,
                ops.as_ptr(),
                &out,
                1,
            ) {
                return false;
            }
            let alloc = |t: Node| -> Option<(i64, i64)> {
                if (*t).buffer.is_null() {
                    return None;
                }
                let buft = ffi::ggml_backend_buffer_get_type((*t).buffer);
                let start = (*t).data as i64;
                Some((
                    start,
                    start + ffi::ggml_backend_buft_get_alloc_size(buft, t) as i64,
                ))
            };
            let dst = self.nodes[s.last];
            let Some((a0, a1)) = alloc(dst) else {
                return false;
            };
            for j in s.at..s.at + count {
                for &src in (*self.nodes[j]).src.iter() {
                    if src.is_null() || (*src).op == ffi::ggml_op_GGML_OP_NONE {
                        continue;
                    }
                    let Some((b0, b1)) = alloc(src) else {
                        return false;
                    };
                    if ((b0 <= a0 && a0 < b1) || (a0 <= b0 && b0 < a1))
                        && !self.nodes[s.at..j].contains(&src)
                    {
                        return false;
                    }
                }
            }
            true
        }
    }

    /// The gate reading the sum's output `moe` (node `last`): its only reader an add of a `mul`
    /// of a contiguous shared expert's output `[e, n]` and a `sigmoid` of the gate logits
    /// `[1, n]` (or of their `cont`), each read only there.
    ///
    /// # Safety
    /// `last` is a node index.
    unsafe fn match_gate(&self, last: usize) -> Option<Gate> {
        // SAFETY: the caller's; nodes and sources are valid tensors.
        unsafe {
            let moe = self.nodes[last];
            let [e, n, _, _] = (*moe).ne;
            let &[add] = self.users(moe) else {
                return None;
            };
            let a = self.nodes[add];
            if self.none[add] || (*a).op != ffi::ggml_op_GGML_OP_ADD || !f32_contiguous(a) {
                return None;
            }
            let mul = if (*a).src[0] == moe {
                (*a).src[1]
            } else if (*a).src[1] == moe {
                (*a).src[0]
            } else {
                return None;
            };
            let &mi = self.index.get(&(mul as usize))?;
            if (*mul).op != ffi::ggml_op_GGML_OP_MUL || self.none[mi] {
                return None;
            }
            let is_sig = |t: Node| {
                (*t).op == ffi::ggml_op_GGML_OP_UNARY
                    && ffi::ggml_get_unary_op(t) == ffi::ggml_unary_op_GGML_UNARY_OP_SIGMOID
                    && (*t).type_ == ffi::ggml_type_GGML_TYPE_F32
                    && (*t).ne == [1, n, 1, 1]
            };
            let (sh, sig) = if is_sig((*mul).src[1]) {
                ((*mul).src[0], (*mul).src[1])
            } else if is_sig((*mul).src[0]) {
                ((*mul).src[1], (*mul).src[0])
            } else {
                return None;
            };
            let &si = self.index.get(&(sig as usize))?;
            if self.none[si] || !f32_contiguous(sh) || (*sh).ne != [e, n, 1, 1] {
                return None;
            }
            let mut chain = vec![mi, si];
            let mut g = (*sig).src[0];
            if (*g).op == ffi::ggml_op_GGML_OP_CONT
                && let Some(&ci) = self.index.get(&(g as usize))
                && !self.none[ci]
                && self.only_read_by(g, &[si])
                && !(*g).src[0].is_null()
                && (*(*g).src[0]).ne == [1, n, 1, 1]
            {
                chain.push(ci);
                g = (*g).src[0];
            }
            if (*g).type_ != ffi::ggml_type_GGML_TYPE_F32
                || (*g).ne != [1, n, 1, 1]
                || (*g).data.is_null()
                || n > i32::MAX as i64
            {
                return None;
            }
            if !self.only_read_by(moe, &[add])
                || !self.only_read_by(mul, &[add])
                || !self.only_read_by(sig, &[mi])
            {
                return None;
            }
            Some(Gate { add, sh, g, chain })
        }
    }

    /// Makes node `at` a [`MOE_SUM_GATE`] reading `x` (with `w`, the router weights, if any)
    /// summing `k` rows per token, plus `gate` (if any); `none` become `GGML_OP_NONE`.
    ///
    /// # Safety
    /// Valid tensors of the graph, checked to fit.
    unsafe fn make_sum_gate(
        &mut self,
        at: usize,
        x: Node,
        w: Option<Node>,
        k: usize,
        gate: Option<(Node, Node)>,
        none: &[usize],
    ) {
        let mut srcs = vec![x];
        srcs.extend(w);
        if let Some((sh, g)) = gate {
            srcs.extend([sh, g]);
        }
        let params = [i32::from(w.is_some()), i32::from(gate.is_some()), k as i32];
        // SAFETY: the caller's.
        unsafe { make_external(self.nodes[at], &MOE_SUM_GATE, &srcs, &params) };
        for &i in none {
            // SAFETY: as above.
            unsafe { (*self.nodes[i]).op = ffi::ggml_op_GGML_OP_NONE };
            self.none[i] = true;
        }
    }

    /// A MoE layer's tail from the `mul` at `at` (see the module doc).
    ///
    /// # Safety
    /// `at` is a node index.
    unsafe fn moe(&mut self, at: usize, cuda: bool, c: &mut Counts) {
        // SAFETY: the caller's; matched nodes are valid tensors of the graph.
        unsafe {
            let Some(s) = self.match_sum(at) else {
                return;
            };
            let gate = self.match_gate(s.last);
            let m = self.nodes[s.at];
            let last = self.nodes[s.last];
            let adds: Vec<usize> = (s.at + 1..=s.last)
                .filter(|&i| (*self.nodes[i]).op == ffi::ggml_op_GGML_OP_ADD)
                .collect();
            let fused = cuda && self.ggml_fuses_sum(&s);
            // The sum's adds other than the last only read in the chain.
            let sum_internal = adds[..adds.len() - 1]
                .iter()
                .all(|&i| self.only_read_by(self.nodes[i], &adds));
            if !fused && sum_internal && self.only_read_by(m, &adds) {
                if let Some(gt) = &gate {
                    let out = self.nodes[gt.add];
                    let mut skip = adds.clone();
                    skip.push(s.at);
                    skip.extend(&gt.chain);
                    let gate_in = [(gt.sh, true), (gt.g, false)];
                    // The whole tail at the gate's add: the mul too, reading the experts' outputs.
                    let all: Vec<(Node, bool)> = [(s.experts, false), (s.weights, false)]
                        .into_iter()
                        .chain(gate_in)
                        .collect();
                    if self.only_read_by(last, &[gt.add])
                        && self.may_read_at(gt.add, out, &all, &skip)
                    {
                        let mut none = skip.clone();
                        none.sort_unstable();
                        let gate = Some((gt.sh, gt.g));
                        self.make_sum_gate(gt.add, s.experts, Some(s.weights), s.k, gate, &none);
                        c.sum_gate += 1;
                        return;
                    }
                    // ggml's mul kept, the adds and the gate reading its output.
                    skip.retain(|&i| i != s.at);
                    let weighted: Vec<(Node, bool)> =
                        [(m, false)].into_iter().chain(gate_in).collect();
                    if self.only_read_by(last, &[gt.add])
                        && self.may_read_at(gt.add, out, &weighted, &skip)
                    {
                        self.make_sum_gate(gt.add, m, None, s.k, Some((gt.sh, gt.g)), &skip);
                        c.weighted_gate += 1;
                        return;
                    }
                }
                // The adds alone, at the last (the mul's output is still allocated there).
                let none: Vec<usize> = adds[..adds.len() - 1].to_vec();
                if self.may_read_at(s.last, last, &[(m, false)], &none) {
                    self.make_sum_gate(s.last, m, None, s.k, None, &none);
                    c.sum += 1;
                }
            }
            // The gate alone, reading the sum (ggml's, or the op's just made).
            if let Some(gt) = gate
                && !self.none[gt.add]
            {
                let out = self.nodes[gt.add];
                let inputs = [(last, true), (gt.sh, true), (gt.g, false)];
                if self.may_read_at(gt.add, out, &inputs, &gt.chain) {
                    self.make_sum_gate(gt.add, last, None, 1, Some((gt.sh, gt.g)), &gt.chain);
                    c.gate += 1;
                }
            }
        }
    }

    /// The expert cache's id remap at the `get_rows` at `at`: `get_rows(table, ids')` with
    /// `ids'` a `cont` of I32 ids (or its reshape) read only there, and a table of one I32 per
    /// row.
    ///
    /// # Safety
    /// `at` is a node index.
    unsafe fn remap(&mut self, at: usize, c: &mut Counts) {
        // SAFETY: the caller's; nodes and sources are valid tensors.
        unsafe {
            let gr = self.nodes[at];
            let (table, idx) = ((*gr).src[0], (*gr).src[1]);
            if table.is_null() || idx.is_null() {
                return;
            }
            let i32_ty = ffi::ggml_type_GGML_TYPE_I32;
            let cont = if (*idx).op == ffi::ggml_op_GGML_OP_CONT {
                idx
            } else if (*idx).op == ffi::ggml_op_GGML_OP_RESHAPE
                && !(*idx).src[0].is_null()
                && (*(*idx).src[0]).op == ffi::ggml_op_GGML_OP_CONT
            {
                (*idx).src[0]
            } else {
                return;
            };
            let Some(&ci) = self.index.get(&(cont as usize)) else {
                return;
            };
            let ids = (*cont).src[0];
            if self.none[ci] || ids.is_null() || (*ids).data.is_null() {
                return;
            }
            let count = (*cont).ne.iter().product::<i64>();
            if (*table).type_ != i32_ty
                || (*table).ne[0] != 1
                || (*table).ne[2] != 1
                || (*table).ne[3] != 1
                || (*table).nb[0] != 4
                || (*gr).type_ != i32_ty
                || !ffi::ggml_is_contiguous(gr)
                || (*gr).ne != [1, count, 1, 1]
                || (*idx).ne != [count, 1, 1, 1]
                || (*cont).type_ != i32_ty
                || (*ids).type_ != i32_ty
                || (*ids).ne != (*cont).ne
                || (*ids).ne[2] != 1
                || (*ids).ne[3] != 1
                || (*ids).nb[0] % 4 != 0
                || (*ids).nb[1] % 4 != 0
                || !self.only_read_by(cont, &[at])
            {
                return;
            }
            // In place over the ids only when they are laid out as the output.
            let same = ffi::ggml_is_contiguous(ids);
            if !self.may_read_at(at, gr, &[(ids, same), (table, false)], &[ci]) {
                return;
            }
            make_external(gr, &IDS_REMAP, &[table, ids], &[]);
            (*cont).op = ffi::ggml_op_GGML_OP_NONE;
            self.none[ci] = true;
            c.remap += 1;
        }
    }
}

unsafe fn cpu_moe_sum_gate(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by the rewrite (see MOE_SUM_GATE), host tensors.
    unsafe {
        let [weights, gated, k, fma, ..] = external_params(dst);
        let x = &*dst.src[0];
        let w = (weights != 0).then(|| &*dst.src[1]);
        let at = 1 + usize::from(weights != 0);
        let gate = (gated != 0).then(|| (&*dst.src[at], &*dst.src[at + 1]));
        let (e, n, k) = (dst.ne[0] as usize, dst.ne[1] as usize, k as usize);
        let (xp, out) = (x.data as *const f32, dst.data as *mut f32);
        for t in (ith..n).step_by(nth) {
            let s = gate.map(|(_, g)| {
                let gv = *((g.data as *const u8).add(t * g.nb[1]) as *const f32);
                1.0 / (1.0 + (-gv).exp())
            });
            for col in 0..e {
                let weight = |j: usize| w.map(|w| *(w.data as *const f32).add(t * k + j));
                let term = |j: usize| {
                    let v = *xp.add((t * k + j) * e + col);
                    match weight(j) {
                        Some(wj) => v * wj,
                        None => v,
                    }
                };
                let mut sum = if fma != 0 {
                    // ggml-cuda's fused sum; with FMA_SKIP a zero weight's row not read (see
                    // MOE_SUM_GATE).
                    let wj = |j: usize| weight(j).unwrap_or(1.0);
                    let row = |j: usize| {
                        if fma == FMA_SKIP && wj(j) == 0.0 {
                            0.0
                        } else {
                            *xp.add((t * k + j) * e + col)
                        }
                    };
                    (1..k).fold(row(0) * wj(0), |acc, j| row(j).mul_add(wj(j), acc))
                } else {
                    (1..k).fold(term(0), |acc, j| acc + term(j))
                };
                if let (Some((sh, _)), Some(s)) = (gate, s) {
                    sum += *(sh.data as *const f32).add(t * e + col) * s;
                }
                *out.add(t * e + col) = sum;
            }
        }
    }
}

unsafe fn cpu_ids_remap(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by the rewrite (see IDS_REMAP), host tensors.
    unsafe {
        let (table, ids) = (&*dst.src[0], &*dst.src[1]);
        let ne0 = ids.ne[0] as usize;
        let n = dst.ne[1] as usize;
        let out = dst.data as *mut i32;
        for i in (ith..n).step_by(nth) {
            let (i0, i1) = (i % ne0, i / ne0);
            let r = *((ids.data as *const u8).add(i0 * ids.nb[0] + i1 * ids.nb[1]) as *const i32);
            *out.add(i) = *((table.data as *const u8).add(r as usize * table.nb[1]) as *const i32);
        }
    }
}

/// `legion_moe_sum_gate` (src/cuda/node_fuse.cu).
#[cfg(feature = "cuda")]
#[repr(C)]
struct SumGate {
    x: *const f32,
    w: *const f32,
    sh: *const f32,
    g: *const f32,
    g_ld: i64,
    out: *mut f32,
    n_embd: i64,
    k: i32,
    n_tok: i32,
    skip_zero: i32,
    out16: *mut u16,
}

/// `legion_ids_remap` (src/cuda/node_fuse.cu).
#[cfg(feature = "cuda")]
#[repr(C)]
struct Remap {
    ids: *const i32,
    s0: i64,
    s1: i64,
    ne0: i64,
    n: i64,
    table: *const u8,
    table_nb1: i64,
    out: *mut i32,
}

/// `dim3`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs' addresses), for
    /// `cudaLaunchKernel`.
    static legion_moe_sum_gate_kernel: *const c_void;
    static legion_moe_tail_fma_kernel: *const c_void;
    static legion_moe_tail_fma4_kernel: *const c_void;
    static legion_ids_remap_kernel: *const c_void;
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// Threads per CTA: one column (or id) each.
#[cfg(feature = "cuda")]
const THREADS: u32 = 256;

#[cfg(feature = "cuda")]
unsafe fn cuda_moe_sum_gate(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // The reduce-scatter reading it runs it (#1176).
    if external_params(dst)[P_BY_EXCHANGE] != 0 {
        return Ok(());
    }
    // SAFETY: the caller's.
    unsafe { launch_sum_gate(dst, None, stream) }
}

/// For the reduce-scatter that took over the [`MOE_SUM_GATE`] node `dst` (#1176, see
/// [`P_BY_EXCHANGE`]): runs it on `stream`, into its own output (`out16` `None`, an F32
/// exchange) or with its values rounded to BF16 at the device address `out16` instead,
/// `[n_embd, n_tok]` contiguous (the send buffer).
///
/// # Safety
/// `dst` is such a node of an allocated graph, its sources' memory still holding their values
/// (the exchange reads them too); `out16` device-writable for the node's values.
pub(crate) unsafe fn launch_tail(
    dst: &ffi::ggml_tensor,
    out16: Option<u64>,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: the caller's.
    unsafe { launch_sum_gate(dst, out16, stream) }
}

/// # Safety
/// `dst` is a [`MOE_SUM_GATE`] node of an allocated graph whose sources hold their values;
/// `out16` (fused-sum form only) as [`launch_tail`]'s.
unsafe fn launch_sum_gate(
    dst: &ffi::ggml_tensor,
    out16: Option<u64>,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: a node made by the rewrite, device tensors of the stream's device; the kernel
    // reads `p` from the launch's copy of its argument.
    unsafe {
        let [weights, gated, k, fma, ..] = external_params(dst);
        if out16.is_some() && fma == 0 {
            return Err("moe_sum_gate: BF16 values from the fused sum only".into());
        }
        let x = &*dst.src[0];
        let w = (weights != 0).then(|| &*dst.src[1]);
        let at = 1 + usize::from(weights != 0);
        let gate = (gated != 0).then(|| (&*dst.src[at], &*dst.src[at + 1]));
        let (e, n) = (dst.ne[0], dst.ne[1]);
        if n < 1 || e < 1 {
            return Ok(());
        }
        if fma != 0 && (w.is_none() || gate.is_none()) {
            return Err("moe_sum_gate: the fused sum needs the weights and the gate".into());
        }
        let mut p = SumGate {
            x: x.data as *const f32,
            w: w.map_or(std::ptr::null(), |w| w.data as *const f32),
            sh: gate.map_or(std::ptr::null(), |(sh, _)| sh.data as *const f32),
            g: gate.map_or(std::ptr::null(), |(_, g)| g.data as *const f32),
            g_ld: gate.map_or(0, |(_, g)| (g.nb[1] / 4) as i64),
            out: dst.data as *mut f32,
            n_embd: e,
            k,
            n_tok: n as i32,
            skip_zero: i32::from(fma == FMA_SKIP),
            out16: out16.map_or(std::ptr::null_mut(), |a| a as *mut u16),
        };
        // The fused sum's kernel: 4 columns a thread where every row is 16-byte aligned (the
        // BF16 values' rows 8-byte aligned).
        let aligned = |t: *const f32| (t as usize).is_multiple_of(16);
        let out_aligned = match out16 {
            Some(a) => a.is_multiple_of(8),
            None => aligned(p.out.cast_const()),
        };
        let v4 = fma != 0 && e % 4 == 0 && aligned(p.x) && aligned(p.sh) && out_aligned;
        let (kernel, cols) = match (fma != 0, v4) {
            (false, _) => (legion_moe_sum_gate_kernel, 1),
            (true, false) => (legion_moe_tail_fma_kernel, 1),
            (true, true) => (legion_moe_tail_fma4_kernel, 4),
        };
        let grid = Dim3 {
            x: n as u32,
            y: (e as u64).div_ceil(u64::from(THREADS) * cols) as u32,
            z: 1,
        };
        let block = Dim3 {
            x: THREADS,
            y: 1,
            z: 1,
        };
        let mut args = [(&raw mut p).cast::<c_void>()];
        match cudaLaunchKernel(kernel, grid, block, args.as_mut_ptr(), 0, stream) {
            0 => Ok(()),
            e => Err(format!("moe_sum_gate: CUDA error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ids_remap(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by the rewrite, device tensors of the stream's device.
    unsafe {
        let (table, ids) = (&*dst.src[0], &*dst.src[1]);
        let n = dst.ne[1];
        if n < 1 {
            return Ok(());
        }
        let mut p = Remap {
            ids: ids.data as *const i32,
            s0: (ids.nb[0] / 4) as i64,
            s1: (ids.nb[1] / 4) as i64,
            ne0: ids.ne[0],
            n,
            table: table.data as *const u8,
            table_nb1: table.nb[1] as i64,
            out: dst.data as *mut i32,
        };
        let grid = Dim3 {
            x: (n as u64).div_ceil(u64::from(THREADS)) as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: THREADS,
            y: 1,
            z: 1,
        };
        let mut args = [(&raw mut p).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_ids_remap_kernel,
            grid,
            block,
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("ids_remap: CUDA error {e}")),
        }
    }
}
