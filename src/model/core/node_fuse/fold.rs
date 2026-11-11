//! MoE tails folded into the next hyper-connection norm (#836; [`RELAYOUT`]).
//!
//! On qwen4exp's prompt steps on an sm_80+ GPU with BF16 mixers (the NVFP4 checkpoints), a MoE
//! layer ends in a tail of small nodes: the routed experts' weighted sum (ggml's `mul` by the
//! router weights and an add per expert, which ggml-cuda runs as one FMA kernel where the sum's
//! output shares no memory with its inputs, #410) and the shared expert's gate (`sigmoid`,
//! `mul`, `add`). Its output is read once, by the next rebuilt BF16-xn `hc_post_norm`. Folded,
//! that norm makes the tail's values itself from the tail's inputs (the experts' rows, the
//! router weights, the shared expert's output and its gate logit), each step rounded as the
//! tail's nodes round it ([`crate::model::qwen4::fold_moe`]), and the tail's nodes go: the
//! tail's output is no longer written and read back, nor the weighted sum's.
//!
//! The norm then reads the tail's inputs, so the allocator must keep them alive until it runs,
//! which means planning the graph as rewritten ([`crate::ggml::post_alloc::set_relayout`]).
//! Which sums ggml-cuda fuses depends on the layout, and the fold must round as the graph as
//! built does: the graph is planned as built first, each tail's rounding read off that layout,
//! then rewritten and planned again. The tails it leaves (the last layer's, any whose output is
//! read elsewhere) must be fused or not as before in the new layout ([`Undo::holds`]); where one
//! is not, the rewrite is undone. The tail's nodes become `GGML_OP_NONE` views of the experts'
//! rows without sources: the allocator gives them no memory and they keep nothing alive.
//!
//! A tail no such norm reads (under tensor parallelism the rank exchange reads it, #1176) whose
//! sum ggml-cuda fuses becomes one [`MOE_SUM_GATE`] at the gate's add instead, in the fused
//! sum's rounding; under `--tp-moe-ep` it does not read the rows of zero router weights (the
//! rows of the experts the other ranks hold, which the grouped kernels stored as zeros). Where
//! that op alone reads the experts' rows, nothing reads those rows at all, so the grouped NVFP4
//! down matmul (and the SwiGLU on its plan that only it reads) stores no zeros for them
//! ([`nvfp4_mm::unread_nodes`]); every value read is the same.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{FMA_SKIP, Gate, Graph, MOE_SUM_GATE, Node, ON, Sum, f32_contiguous, is_view_op, root};
use crate::ggml::post_alloc::{Relaid, Relayout};
use crate::ggml::{ffi, make_external};
use crate::model::nvfp4_mm;
use crate::model::qwen4::{FoldSum, fold_moe, fold_target, moe_fold};
use crate::tp_net::seq_par;

/// The fold as a relayout (`crate::ggml::post_alloc::set_relayout`): only graphs holding a norm
/// a tail may fold into ([`fold_target`]) take it, with [`crate::model::qwen4::set_moe_fold`]
/// and the node fusion (`legion serve --node-fuse-ggml` turns both off) on.
pub(crate) const RELAYOUT: Relayout = Relayout { wants, rewrite };

/// Relayouts whose counts were logged (the first few only).
static LOGGED: AtomicUsize = AtomicUsize::new(0);

/// Relayouts undone (logged the first time).
static UNDONE: AtomicUsize = AtomicUsize::new(0);

/// Grouped NVFP4 nodes left without their skipped pairs' zeros so far (#1176; logged with the
/// first relayouts).
static UNREAD: AtomicUsize = AtomicUsize::new(0);

/// Tails run by the reduce-scatter reading them so far (#1176; logged with the first
/// relayouts).
static TAKEN: AtomicUsize = AtomicUsize::new(0);

/// [`Relayout::wants`].
///
/// # Safety
/// A live graph.
unsafe fn wants(graph: *mut ffi::ggml_cgraph, cuda: bool) -> bool {
    if !moe_fold() || !ON.load(Ordering::Relaxed) {
        return false;
    }
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    let any = unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n)
            .iter()
            .any(|&t| fold_target(t, cuda))
    };
    // With ggml-cuda's fusions off every sum runs unfused; left alone, as the node fusion does.
    any && !(cuda && std::env::var_os("GGML_CUDA_DISABLE_FUSION").is_some())
}

/// [`Relayout::rewrite`]: folds every tail that fits, rounded as the graph as allocated now
/// rounds it.
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
unsafe fn rewrite(graph: *mut ffi::ggml_cgraph, cuda: bool) -> Option<Box<dyn Relaid>> {
    // SAFETY: the caller's.
    let (undo, folded, fma) = unsafe { fold_graph(graph, cuda) };
    if folded == 0 {
        return None;
    }
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
        tracing::info!(
            folded,
            fma,
            left = undo.left.len(),
            unread = UNREAD.load(Ordering::Relaxed),
            by_exchange = TAKEN.load(Ordering::Relaxed),
            "hc moe fold: MoE tails made by the next hyper-connection norm, or as one op at \
             the gate's add where no such norm reads them (the first graphs; --hc-moe-fold)"
        );
    }
    Some(Box::new(undo))
}

/// Folds `graph`'s tails (see the module doc): how to check and undo it, the tails folded,
/// and of those, the sums ggml-cuda fuses.
///
/// # Safety
/// As [`crate::ggml::post_alloc::Rewrite`].
pub(crate) unsafe fn fold_graph(graph: *mut ffi::ggml_cgraph, cuda: bool) -> (Undo, usize, usize) {
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n).to_vec();
        let mut g = Graph::new(graph, nodes);
        let mut undo = Undo {
            left: Vec::new(),
            saved: Vec::new(),
            cuda,
        };
        let (mut folded, mut fma) = (0, 0);
        for i in 0..n {
            if g.none[i] || (*g.nodes[i]).op != ffi::ggml_op_GGML_OP_MUL {
                continue;
            }
            let Some(s) = g.match_sum(i) else {
                continue;
            };
            let fused = cuda && g.ggml_fuses_sum(&s);
            if g.fold_tail(&s, fused, cuda, &mut undo.saved) {
                folded += 1;
                fma += usize::from(fused);
            } else {
                undo.left.push((i, fused));
            }
        }
        (undo, folded, fma)
    }
}

/// What [`fold_graph`] did.
pub(crate) struct Undo {
    /// The tails left as they were: their `mul`'s index, and whether ggml-cuda fused the sum
    /// in the layout of the graph as built.
    left: Vec<(usize, bool)>,
    /// The nodes changed, as they were.
    saved: Vec<(Node, ffi::ggml_tensor)>,
    cuda: bool,
}

impl Relaid for Undo {
    unsafe fn holds(&self, graph: *mut ffi::ggml_cgraph) -> bool {
        if !self.cuda || self.left.is_empty() {
            return true;
        }
        // SAFETY: the caller's: a live, allocated graph of valid tensors.
        unsafe {
            let n = ffi::ggml_graph_n_nodes(graph) as usize;
            let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n).to_vec();
            let g = Graph::new(graph, nodes);
            let held = self.left.iter().all(|&(at, fused)| {
                g.match_sum(at)
                    .is_some_and(|s| g.ggml_fuses_sum(&s) == fused)
            });
            if !held && UNDONE.fetch_add(1, Ordering::Relaxed) == 0 {
                tracing::info!(
                    "hc moe fold: a tail left unfolded would round differently in the folded \
                     graph's layout; that graph is left as built (logged once)"
                );
            }
            held
        }
    }

    unsafe fn undo(&self, _graph: *mut ffi::ggml_cgraph) {
        for (t, was) in &self.saved {
            // SAFETY: the caller's: the graph's nodes, saved whole before the rewrite.
            unsafe { std::ptr::copy_nonoverlapping(was, *t, 1) };
        }
    }
}

impl Graph {
    /// Folds the tail of the sum `s` (ggml-cuda fusing it in the graph as built: `fused`) into
    /// the norm reading it, where its gate's add is read only by that norm (directly or through
    /// a view of all of it) and only the tail reads its other nodes. Saves the nodes it changes
    /// first. Whether it did.
    ///
    /// # Safety
    /// `s` is a match of this graph, whose nodes are valid tensors.
    unsafe fn fold_tail(
        &mut self,
        s: &Sum,
        fused: bool,
        cuda: bool,
        saved: &mut Vec<(Node, ffi::ggml_tensor)>,
    ) -> bool {
        // SAFETY: the caller's; nodes and their sources are valid tensors.
        unsafe {
            let Some(gt) = self.match_gate(s.last) else {
                return false;
            };
            let m = self.nodes[s.at];
            let adds: Vec<usize> = (s.at + 1..=s.last)
                .filter(|&i| (*self.nodes[i]).op == ffi::ggml_op_GGML_OP_ADD)
                .collect();
            if !adds[..adds.len() - 1]
                .iter()
                .all(|&i| self.only_read_by(self.nodes[i], &adds))
                || !self.only_read_by(m, &adds)
            {
                return false;
            }
            let a = self.nodes[gt.add];
            let reader = match *self.users(a) {
                [u] if !is_view_op((*self.nodes[u]).op) => Some((u, None)),
                [u] => {
                    let v = self.nodes[u];
                    match *self.users(v) {
                        [ni] if (*v).view_src == a
                            && (*v).view_offs == 0
                            && (*v).ne == (*a).ne
                            && (*v).nb == (*a).nb =>
                        {
                            Some((ni, Some(u)))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            let [e, n, _, _] = (*a).ne;
            let x = s.experts;
            if !f32_contiguous(x) || (*x).ne != [e, s.k as i64, n, 1] || s.k > i32::MAX as usize {
                return false;
            }
            let into_norm = reader.filter(|&(ni, view)| {
                let norm = self.nodes[ni];
                !self.none[ni]
                    && fold_target(norm, cuda)
                    && (*norm).src[1] == view.map_or(a, |u| self.nodes[u])
                    && self.only_read_by(a, &[ni])
            });
            let Some((ni, view)) = into_norm else {
                // No norm to fold into (under tensor parallelism the rank exchange reads the
                // tail, #1176): a sum ggml-cuda fuses and its gate as one op at the gate's add.
                return fused && self.fuse_tail(s, &gt, saved);
            };
            let norm = self.nodes[ni];
            let mut dead: Vec<usize> = (s.at..=s.last).collect();
            dead.extend(&gt.chain);
            dead.push(gt.add);
            dead.extend(view);
            // BF16 rows read back as F32 (`--nvfp4-expert-rows-bf16`, #1183): the norm reads
            // the BF16 rows (the cast is exact) and the cast goes too.
            let (x, cast) = match self.bf16_rows(x, s.at) {
                Some((rows, i)) => (rows, Some(i)),
                None => (x, None),
            };
            dead.extend(cast);
            saved.push((norm, std::ptr::read(norm)));
            saved.extend(
                dead.iter()
                    .map(|&i| (self.nodes[i], std::ptr::read(self.nodes[i]))),
            );
            let sum = if fused { FoldSum::Fma } else { FoldSum::Plain };
            fold_moe(norm, x, s.weights, s.k, sum, gt.sh, gt.g);
            self.retire(&dead, root(x));
            true
        }
    }

    /// Where `x` is a cast to F32 of contiguous BF16 rows of its shape (a node), read only by the
    /// node `at`: those rows and the cast's index.
    ///
    /// # Safety
    /// `x` is a valid tensor of the graph.
    unsafe fn bf16_rows(&self, x: Node, at: usize) -> Option<(Node, usize)> {
        // SAFETY: the caller's; sources are valid tensors.
        unsafe {
            let src = (*x).src[0];
            let i = *self.index.get(&(x as usize))?;
            ((*x).op == ffi::ggml_op_GGML_OP_CPY
                && !self.none[i]
                && !src.is_null()
                && (*src).type_ == ffi::ggml_type_GGML_TYPE_BF16
                && ffi::ggml_is_contiguous(src)
                && (*src).ne == (*x).ne
                // ggml's cast lists itself as its second source.
                && self.only_read_by(x, &[at, i]))
            .then_some((src, i))
        }
    }

    /// The tail of the sum `s`, which ggml-cuda fuses, with its gate `gt` as one
    /// [`MOE_SUM_GATE`] in its FMA form at the gate's add (#1176): the add's values, rounded as
    /// the tail's nodes round them, without the weighted sum written and read back, and under
    /// `--tp-moe-ep` with the rows of zero router weights not read (the other ranks' experts:
    /// half the rows on two ranks). Saves the nodes it changes first. Whether it did.
    ///
    /// # Safety
    /// As [`Self::fold_tail`]; `s`'s experts are `[e, k, n]` contiguous F32 and `gt` its gate.
    unsafe fn fuse_tail(
        &mut self,
        s: &Sum,
        gt: &Gate,
        saved: &mut Vec<(Node, ffi::ggml_tensor)>,
    ) -> bool {
        // SAFETY: the caller's; nodes and their sources are valid tensors.
        unsafe {
            let a = self.nodes[gt.add];
            let mut dead: Vec<usize> = (s.at..=s.last).collect();
            dead.extend(&gt.chain);
            saved.push((a, std::ptr::read(a)));
            saved.extend(
                dead.iter()
                    .map(|&i| (self.nodes[i], std::ptr::read(self.nodes[i]))),
            );
            let srcs = [s.experts, s.weights, gt.sh, gt.g];
            // Rows of zero weights skipped only under --tp-moe-ep, where they hold +0.0.
            let ep = crate::model::core::tp_ep::active() && crate::model::core::tp_ep::skip();
            let fma = if ep { FMA_SKIP } else { 1 };
            if ep {
                // Then nothing reads the other ranks' rows: the grouped kernels need not store
                // their zeros (the experts' rows read by the tail's `mul` alone, so by the new
                // op alone).
                let down = root(s.experts);
                if self.only_read_by(down, &[s.at]) {
                    let at = |t: Node| self.index.get(&(t as usize)).copied();
                    let di = at(down);
                    let marks = nvfp4_mm::unread_nodes(down, |g| {
                        di.is_some_and(|di| at(g).is_some() && self.only_read_by(g, &[di]))
                    });
                    UNREAD.fetch_add(marks.len(), Ordering::Relaxed);
                    for t in marks {
                        saved.push((t, std::ptr::read(t)));
                        nvfp4_mm::leave_unread(t);
                    }
                }
            }
            make_external(a, &MOE_SUM_GATE, &srcs, &[1, 1, s.k as i32, fma]);
            self.retire(&dead, root(s.experts));
            // The reduce-scatter reading the tail runs it, its values rounded to BF16 straight
            // into the send buffer (#1176; the graph is planned again after this rewrite).
            let rs = match *self.users(a) {
                [u] if is_view_op((*self.nodes[u]).op) => match *self.users(self.nodes[u]) {
                    [r] => Some(r),
                    _ => None,
                },
                [u] => Some(u),
                _ => None,
            };
            if let Some(r) = rs.filter(|&r| !self.none[r] && self.only_read_by(a, &[r])) {
                // (`a` is saved as built above, which an undo puts back whole.)
                let rs = self.nodes[r];
                let was = std::ptr::read(rs);
                if seq_par::take_tail(rs, a, &srcs) {
                    saved.push((rs, was));
                    TAKEN.fetch_add(1, Ordering::Relaxed);
                }
            }
            true
        }
    }

    /// Nodes `dead` of a folded tail become views of `keep` (the experts' rows, alive until the
    /// op that reads them now) that read nothing: no memory of their own, nothing kept alive,
    /// nothing run.
    ///
    /// # Safety
    /// Node indices of this graph; `keep` a valid tensor.
    unsafe fn retire(&mut self, dead: &[usize], keep: Node) {
        for &i in dead {
            // SAFETY: the caller's.
            let t = unsafe { &mut *self.nodes[i] };
            t.op = ffi::ggml_op_GGML_OP_NONE;
            t.src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
            t.view_src = keep;
            t.view_offs = 0;
            self.none[i] = true;
        }
    }
}
