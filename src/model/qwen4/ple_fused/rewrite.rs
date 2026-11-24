//! The PLE ops in place of ggml's chain in an allocated graph (#1183).
//!
//! Built as two ops, the graph is about thirty nodes smaller, so ggml's allocator places the
//! tensors around it elsewhere: the key matmul runs before the layer before the PLE and the
//! value matmul after it, where the chain runs both (and its repeated value and normed key, two
//! `[e · hc, n]` tensors) before it. ggml-cuda fuses an MoE layer's weighted expert sum (FMA
//! adds) only where its output does not overlap its inputs, so a layer whose sum flips between
//! the two rounds differently: on a GB10 the one before the PLE did, and greedy text moved.
//!
//! So [`Model::ple`](super::super::Model) builds the chain, its nodes marked by name
//! ([`KEY`] ... [`OUT`]), and the graph is allocated as built: every sum ggml-cuda fuses is the
//! one it fuses without the ops, and the MoE tail folds read their rounding off that layout
//! (the fold's relayout, `node_fuse/fold.rs`). Then, as part of that relayout, the chain's
//! normed conv input (`xn`, [`XN`]) becomes [`PLE_GATE`] (reading the key, the streams and the
//! value; the gates go after `xn` in its memory, as `gate_norm` lays them out), the new streams
//! ([`OUT`]) become [`PLE_CONV`], one node between them a view of the gates, and the chain's
//! other nodes `GGML_OP_NONE`; the graph is planned again as rewritten. The nodes reading the
//! conv input from outside the chain (the captures, [`X`]) read the op's `xn`.
//!
//! Only a chain of one prompt group is marked. A marked chain that does not match keeps ggml's
//! nodes (logged once, with why).

use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{PLE_CONV, PLE_GATE, PLE_HC, PLE_MIN_E};
use crate::ggml::{ffi, make_external};

type Node = *mut ffi::ggml_tensor;

/// The key matmul's output (`[e · hc, n]`).
pub(crate) const KEY: &str = "ple_ops.key";
/// The value matmul's output (`[e, n]`).
pub(crate) const VALUE: &str = "ple_ops.value";
/// The key's, the query's and the conv input's gammas, as the chain's norms read them.
pub(crate) const GAMMAS: [&str; 3] = ["ple_ops.gk", "ple_ops.gq", "ple_ops.gc"];
/// The conv's first tap of its weight (a view of the whole weight).
pub(crate) const TAP: &str = "ple_ops.tap";
/// The normed conv input (the gated value's rms_norm times its gamma, `[e, hc, n]`).
pub(crate) const XN: &str = "ple_ops.xn";
/// The conv input as the group reads it (`[e · hc, nt, ns]`, a view of [`XN`]).
pub(crate) const X: &str = "ple_ops.x";
/// The copy of the last inputs back to the conv history.
pub(crate) const KEEP: &str = "ple_ops.keep";
/// The new streams (`res + (gated + silu(conv))`).
pub(crate) const OUT: &str = "ple_ops.out";

/// The most tokens, and sequences, in a group the conv kernel takes (its grid's y and z).
pub(crate) const PLE_CONV_MAX_GRID: i64 = 65535;

/// Whether the conv kernel takes a group of `nt` tokens by `ns` sequences.
pub(crate) fn conv_grid_fits(nt: i64, ns: i64) -> bool {
    (1..=PLE_CONV_MAX_GRID).contains(&nt) && (1..=PLE_CONV_MAX_GRID).contains(&ns)
}

/// Whether the reason a marked chain kept ggml's nodes was logged.
static KEPT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Whether `t` is named `name`.
///
/// # Safety
/// `t` is a valid tensor.
unsafe fn named(t: Node, name: &str) -> bool {
    // SAFETY: the caller's; ggml keeps names NUL-terminated.
    unsafe { CStr::from_ptr((*t).name.as_ptr()).to_bytes() == name.as_bytes() }
}

/// Whether `graph` holds a marked chain ([`OUT`]).
///
/// # Safety
/// A live graph.
pub(crate) unsafe fn marked(graph: *mut ffi::ggml_cgraph) -> bool {
    // SAFETY: the caller's: the node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n)
            .iter()
            .any(|&t| named(t, OUT))
    }
}

/// Rewrites every marked chain of `graph` (allocated as built, none of its nodes run) into the
/// two ops, saving each node it changes to `saved` first. How many it rewrote.
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`]; the graph is planned again after it.
pub(crate) unsafe fn rewrite(
    graph: *mut ffi::ggml_cgraph,
    saved: &mut Vec<(Node, ffi::ggml_tensor)>,
) -> usize {
    // SAFETY: the caller's: the node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n).to_vec();
        // The key and the value are the matmuls' outputs in a model's graph; a graph of the
        // chain alone (the tests) has them as leafs.
        let named_in: Vec<Node> = nodes
            .iter()
            .chain(crate::ggml::graph_leafs(graph))
            .copied()
            .collect();
        let mut done = 0;
        for &out in &nodes {
            if !named(out, OUT) {
                continue;
            }
            match Chain::find(&nodes, &named_in, out) {
                Ok(c) => {
                    c.apply(saved);
                    done += 1;
                }
                Err(why) => {
                    if !KEPT_LOGGED.swap(true, Ordering::Relaxed) {
                        tracing::info!(
                            "PLE ops: ggml's chain kept in a prompt graph (logged once): {why}"
                        );
                    }
                }
            }
        }
        done
    }
}

/// A marked chain that fits the ops.
struct Chain {
    key: Node,
    res: Node,
    value: Node,
    gammas: [Node; 3],
    w: Node,
    state: Node,
    xn: Node,
    out: Node,
    /// The node that becomes the view of the gates (one of the chain's, after `xn`).
    gate: Node,
    /// Views of `xn` read from outside the chain ([`X`] among them): they stay, reading `xn`.
    kept: Vec<Node>,
    /// The chain's other nodes.
    retired: Vec<Node>,
    e: i64,
    hc: i64,
    n: i64,
    nt: i64,
    dil: i64,
    eps: f32,
}

/// The one node of `nodes` named `name`.
///
/// # Safety
/// Valid tensors.
unsafe fn one(nodes: &[Node], name: &str) -> Result<Node, String> {
    // SAFETY: the caller's.
    let found: Vec<Node> = nodes
        .iter()
        .copied()
        .filter(|&t| unsafe { named(t, name) })
        .collect();
    match found[..] {
        [t] => Ok(t),
        _ => Err(format!("{} nodes named {name}", found.len())),
    }
}

/// Whether `t` is F32 `ne` with contiguous elements (`nb[0]` 4).
///
/// # Safety
/// A valid tensor.
unsafe fn f32_rows(t: Node, ne: [i64; 4]) -> bool {
    // SAFETY: the caller's.
    unsafe { (*t).type_ == ffi::ggml_type_GGML_TYPE_F32 && (*t).ne == ne && (*t).nb[0] == 4 }
}

/// Whether `op` only makes a view.
fn is_view_op(op: ffi::ggml_op) -> bool {
    matches!(
        op,
        ffi::ggml_op_GGML_OP_VIEW
            | ffi::ggml_op_GGML_OP_RESHAPE
            | ffi::ggml_op_GGML_OP_PERMUTE
            | ffi::ggml_op_GGML_OP_TRANSPOSE
    )
}

/// The most nodes a chain of one group has (about forty with its views; a bound against a mark
/// that leaks the walk into the rest of the graph).
const MAX_CHAIN: usize = 96;

impl Chain {
    /// The chain ending in `out` (named [`OUT`]) of `nodes` (the graph's, in order), or why it
    /// does not fit; the key and the value are found in `named_in` (the nodes and the leafs).
    ///
    /// # Safety
    /// Valid tensors.
    unsafe fn find(nodes: &[Node], named_in: &[Node], out: Node) -> Result<Self, String> {
        // SAFETY: the caller's; sources and view sources are valid tensors.
        unsafe {
            let key = one(named_in, KEY)?;
            let value = one(named_in, VALUE)?;
            let xn = one(nodes, XN)?;
            let x = one(nodes, X)?;
            let keep = one(nodes, KEEP)?;
            let tap = one(nodes, TAP)?;
            let gammas = [
                one(nodes, GAMMAS[0])?,
                one(nodes, GAMMAS[1])?,
                one(nodes, GAMMAS[2])?,
            ];
            let o = &*out;
            if o.op != ffi::ggml_op_GGML_OP_ADD || !ffi::ggml_is_contiguous(out) {
                return Err("the new streams are not a contiguous add".into());
            }
            let [e, hc, n, _] = o.ne;
            let width = e * hc;
            if hc != PLE_HC || e < PLE_MIN_E {
                return Err(format!("{hc} streams of {e}"));
            }
            let res = o.src[0];
            let w = (*tap).src[0];
            let state = (*keep).src[1];
            if !f32_rows(out, [e, hc, n, 1])
                || res.is_null()
                || !f32_rows(res, [e, hc, n, 1])
                || (*res).nb[1] != e as usize * 4
            {
                return Err("the streams' shapes".into());
            }
            if !f32_rows(key, [width, n, 1, 1]) || !f32_rows(value, [e, n, 1, 1]) {
                return Err("the key's or the value's shape".into());
            }
            for g in gammas {
                if (*g).type_ != ffi::ggml_type_GGML_TYPE_F32
                    || !ffi::ggml_is_contiguous(g)
                    || ffi::ggml_nelements(g) != width
                    || (*g).data.is_null()
                {
                    return Err("a gamma is not contiguous F32 in memory".into());
                }
            }
            if w.is_null()
                || !((*w).type_ == ffi::ggml_type_GGML_TYPE_F32
                    || (*w).type_ == ffi::ggml_type_GGML_TYPE_BF16)
                || !ffi::ggml_is_contiguous(w)
                || (*w).ne[1] != width
                || (*w).ne[0] < 2
                || (*w).data.is_null()
            {
                return Err("the conv weight's type or shape".into());
            }
            let kern = (*w).ne[0];
            if (*keep).op != ffi::ggml_op_GGML_OP_CPY || state.is_null() {
                return Err("the history copy".into());
            }
            let [hist, sw, ns, _] = (*state).ne;
            if !f32_rows(state, [hist, width, ns, 1]) || sw != width || hist % (kern - 1) != 0 {
                return Err("the history's shape".into());
            }
            let dil = hist / (kern - 1);
            let [xw, nt, xs, _] = (*x).ne;
            if !f32_rows(x, [width, nt, ns, 1])
                || xw != width
                || xs != ns
                || nt * ns != n
                || (*x).view_src != xn
                || (*x).view_offs != 0
                || !ffi::ggml_is_contiguous(x)
                || dil < 1
            {
                return Err("the conv input's view".into());
            }
            if !conv_grid_fits(nt, ns) {
                return Err(format!(
                    "a group of {nt} tokens by {ns} sequences (the conv kernel takes \
                     {PLE_CONV_MAX_GRID} of each)"
                ));
            }
            let norm = (*xn).src[0];
            if (*xn).op != ffi::ggml_op_GGML_OP_MUL
                || !f32_rows(xn, [e, hc, n, 1])
                || !ffi::ggml_is_contiguous(xn)
                || norm.is_null()
                || (*norm).op != ffi::ggml_op_GGML_OP_RMS_NORM
            {
                return Err("the conv input's norm".into());
            }
            let eps = f32::from_bits((*norm).op_params[0] as u32);

            let index: HashMap<Node, usize> =
                nodes.iter().enumerate().map(|(i, &t)| (t, i)).collect();
            let (ix, io) = (index[&xn], index[&out]);
            if ix >= io {
                return Err("the conv input after the new streams".into());
            }
            for t in [key, value, res].into_iter().chain(gammas) {
                if index.get(&t).is_some_and(|&i| i >= ix) {
                    return Err("an input after the conv input".into());
                }
            }

            // The chain: what the new streams and the history copy read, back to the inputs.
            let stop: HashSet<Node> = [key, value, res, state, w]
                .into_iter()
                .chain(gammas)
                .collect();
            let mut chain: HashSet<Node> = HashSet::new();
            let mut todo = vec![out, keep];
            while let Some(t) = todo.pop() {
                if stop.contains(&t) || !index.contains_key(&t) || !chain.insert(t) {
                    continue;
                }
                if chain.len() > MAX_CHAIN {
                    return Err(format!("more than {MAX_CHAIN} nodes"));
                }
                todo.extend((*t).src.iter().copied().filter(|s| !s.is_null()));
            }
            if !chain.contains(&xn) || !chain.contains(&x) {
                return Err("the conv input is not the chain's".into());
            }
            // Read from outside the chain: the new streams, `xn` and views within `xn` only.
            let xn_bytes = (e * hc * n) as usize * 4;
            let mut kept: HashSet<Node> = HashSet::from([x]);
            for &u in nodes {
                if chain.contains(&u) {
                    continue;
                }
                for &s in (*u).src.iter().filter(|s| !s.is_null()) {
                    if !chain.contains(&s) || s == out {
                        continue;
                    }
                    if is_view_op((*s).op)
                        && (*s).view_src == xn
                        && (*s).view_offs + ffi::ggml_nbytes(s) <= xn_bytes
                    {
                        kept.insert(s);
                    } else {
                        return Err("a node outside the chain reads it".into());
                    }
                }
            }
            for &t in &chain {
                if t != out
                    && (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0
                {
                    return Err("one of its nodes is an output".into());
                }
            }
            // The gates' view: the chain's first node after `xn` that nothing else needs and the
            // graph's plan gave memory (not a view of a weight or of the histories).
            let free = |t: &Node| *t != out && *t != xn && !kept.contains(t);
            let Some(gate) = nodes[ix + 1..io].iter().copied().find(|t| {
                chain.contains(t) && free(t) && ((**t).view_src.is_null() || (**t).view_src == xn)
            }) else {
                return Err("no node between the two ops".into());
            };
            let retired = nodes
                .iter()
                .copied()
                .filter(|t| chain.contains(t) && free(t) && *t != gate)
                .collect();
            Ok(Self {
                key,
                res,
                value,
                gammas,
                w,
                state,
                xn,
                out,
                gate,
                kept: kept.into_iter().collect(),
                retired,
                e,
                hc,
                n,
                nt,
                dil,
                eps,
            })
        }
    }

    /// Rewrites the chain's nodes (see the module doc), each saved to `saved` first.
    ///
    /// # Safety
    /// The chain's nodes are valid tensors of a graph not run yet.
    unsafe fn apply(&self, saved: &mut Vec<(Node, ffi::ggml_tensor)>) {
        // SAFETY: the caller's.
        unsafe {
            let all = [self.xn, self.out, self.gate]
                .into_iter()
                .chain(self.kept.iter().copied())
                .chain(self.retired.iter().copied());
            for t in all {
                saved.push((t, std::ptr::read(t)));
            }
            let (e, hc, n) = (self.e, self.hc, self.n);
            let inv_sqrt_e = 1.0 / (e as f32).sqrt();
            let [gk, gq, gc] = self.gammas;
            make_external(
                self.xn,
                &PLE_GATE,
                &[self.key, self.res, self.value, gk, gq, gc],
                &[
                    self.eps.to_bits() as i32,
                    inv_sqrt_e.to_bits() as i32,
                    1e-6f32.to_bits() as i32,
                ],
            );
            // `xn` (`[e · hc, n]`), then the gates (`[hc, n]`), as `gate_norm` lays them out.
            let total = e * hc * n + hc * n;
            let xn = &mut *self.xn;
            xn.ne = [total, 1, 1, 1];
            let bytes = total as usize * 4;
            xn.nb = [4, bytes, bytes, bytes];

            let g = &mut *self.gate;
            let offs = (e * hc * n) as usize * 4;
            g.op = ffi::ggml_op_GGML_OP_VIEW;
            g.type_ = ffi::ggml_type_GGML_TYPE_F32;
            g.src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
            g.src[0] = self.xn;
            g.view_src = self.xn;
            g.view_offs = offs;
            g.ne = [hc, n, 1, 1];
            let row = hc as usize * 4;
            g.nb = [4, row, row * n as usize, row * n as usize];
            g.op_params = [0; ffi::GGML_MAX_OP_PARAMS as usize / 4];
            // ggml_view's parameter: the offset.
            std::ptr::write_unaligned(g.op_params.as_mut_ptr() as *mut usize, offs);

            for &v in &self.kept {
                (*v).src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
                (*v).src[0] = self.xn;
            }
            let x = *self
                .kept
                .iter()
                .find(|&&v| named(v, X))
                .expect("the conv input is kept");
            make_external(
                self.out,
                &PLE_CONV,
                &[x, self.state, self.w, self.value, self.gate, self.res],
                &[self.nt as i32, self.dil as i32],
            );
            // Views of the weight that read nothing: no memory, nothing kept alive, nothing run.
            for &t in &self.retired {
                let t = &mut *t;
                t.op = ffi::ggml_op_GGML_OP_NONE;
                t.src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
                t.view_src = self.w;
                t.view_offs = 0;
            }
        }
    }
}
