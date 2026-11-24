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
//! ([`nvfp4_mm::unread_nodes`]); every value read is the same. With BF16 expert rows read back
//! as F32 (`--nvfp4-expert-rows-bf16`, #1183) the op reads the BF16 rows and the exact cast
//! goes, as the folded norms read them (not under `--tp-moe-ep`).
//!
//! Lazy streams (#1183, after the folds; `legion serve --no-hc-lazy-streams` turns them off):
//! a layer's FFN mixer combine-and-norm makes the new residual streams `R_f = O_a · w + R_a`
//! (the attention block's output `O_a`, the weights `w` from the attention mixer's inject
//! weights, the streams `R_a` before it) and writes them, 4 x n_embd F32 values a token, for
//! its up-mix and the next layer's folded norm to read back. Where those two are its streams'
//! only readers, the norm does not write them ([`crate::model::qwen4::set_no_res`]) and both
//! readers make them per value from `R_a`, `O_a` and the weights, with the norm's expressions
//! and its one FMA, so the same bits: the streams' write goes, and `O_a` (a quarter of their
//! bytes) is read twice more. `R_a` and `O_a` then stay alive through the MoE layer, which is
//! why it is part of this relayout; the norm's views of its streams, read by nothing now, are
//! retired so they do not keep its memory alive to the end of the step. Both norms' nodes are
//! then laid out without the bytes they do not write ([`crate::model::qwen4::compact_plan`]):
//! the FFN mixer's holds only its BF16 `xn`, scales and inject weights (20 KB a token, not
//! 100), the attention mixer's (the one holding `R_a`) loses the unused rest of `xn`'s place
//! (60 KB, not 100), so a prompt step's compute buffer is no larger than with the streams
//! written, though `R_a` (with that node's other parts) and `O_a` live longer.
//!
//! One BF16 cast (#1183, after the folds; `--dense-fp8-prompt`): a MoE layer's input is cast to
//! BF16 twice, once for the NVFP4 experts and once for the shared expert's FP8 codes
//! ([`crate::model::fp8_act::plain_bf16`]). The later cast becomes a view of the earlier one:
//! the same bytes (one `__float2bfloat16` a value, in order), written once, and the earlier
//! cast stays alive until the codes are made. It is done here, not where the graph is built,
//! because the graph as built must stay as it is: its layout decides which tails round as
//! ggml-cuda's fused sum (a graph built with one cast fewer moved two of them).

use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    AddrMap, FMA_SKIP, Gate, Graph, MOE_SUM_GATE, Node, ON, Sum, f32_contiguous, is_view_op, root,
    span,
};
use crate::ggml::post_alloc::{Relaid, Relayout};
use crate::ggml::{ffi, make_external};
use crate::model::nvfp4_mm;
use crate::model::qwen4::{
    FoldSum, compact_plan, fold_moe, fold_target, lazy_fold_target, lazy_mix_target, lazy_parts,
    lazy_streams, moe_fold, ple_rewrite, set_compact, set_lazy_fold, set_lazy_streams, set_no_res,
};
use crate::tp_net::seq_par;

/// The fold as a relayout (`crate::ggml::post_alloc::set_relayout`): only graphs holding a norm
/// a tail may fold into ([`fold_target`]) take it, with [`crate::model::qwen4::set_moe_fold`]
/// and the node fusion (`legion serve --node-fuse-ggml` turns both off) on. Graphs holding a
/// PLE chain marked for the PLE ops take it too: the ops take the chain's nodes first
/// ([`crate::model::qwen4::ple_rewrite`]), and the sums ggml-cuda fuses, folded or not, are
/// the ones of the graph as built.
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
    // SAFETY: the caller's.
    unsafe { ple_rewrite::marked(graph) || fold_wants(graph, cuda) }
}

/// Whether the fold takes `graph` (see [`RELAYOUT`]).
///
/// # Safety
/// A live graph.
unsafe fn fold_wants(graph: *mut ffi::ggml_cgraph, cuda: bool) -> bool {
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
    // The PLE ops first, the fold then sees the graph's readers as they run; neither moves a
    // node's memory before the graph is planned again.
    let mut ple_saved = Vec::new();
    // SAFETY: the caller's.
    let ple = unsafe { ple_rewrite::rewrite(graph, &mut ple_saved) };
    // SAFETY: the caller's.
    let (mut undo, folded, fma) = unsafe {
        if fold_wants(graph, cuda) {
            fold_graph(graph, cuda)
        } else {
            (unfolded(graph, cuda), 0, 0)
        }
    };
    undo.saved.splice(0..0, ple_saved);
    if folded == 0 && ple == 0 {
        // SAFETY: the caller's; nothing was rewritten unless the fold's streams were.
        unsafe { undo.undo(graph) };
        return None;
    }
    if folded > 0 && LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
        tracing::info!(
            folded,
            fma,
            lazy_streams = undo.lazy,
            compact = undo.compact,
            one_cast = undo.one_cast,
            left = undo.left.len(),
            unread = UNREAD.load(Ordering::Relaxed),
            by_exchange = TAKEN.load(Ordering::Relaxed),
            "hc moe fold: MoE tails made by the next hyper-connection norm, or as one op at \
             the gate's add where no such norm reads them, and the FFN mixers' streams made by \
             their readers (the first graphs; --hc-moe-fold, --no-hc-lazy-streams)"
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
            lazy: 0,
            compact: 0,
            one_cast: 0,
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
        if folded > 0 && lazy_streams() {
            (undo.lazy, undo.compact) = g.lazy_streams(cuda, &mut undo.saved);
        }
        // After every tail's rounding is read off the graph as built (see the module doc), and
        // only where the graph is planned again (some tail folded).
        if folded > 0 {
            undo.one_cast = g.one_cast(&mut undo.saved);
        }
        (undo, folded, fma)
    }
}

/// How to check and undo a graph whose sums are all left as built (only its PLE chains
/// rewritten): every sum, and whether ggml-cuda fuses it in the graph as built.
///
/// # Safety
/// As [`fold_graph`].
unsafe fn unfolded(graph: *mut ffi::ggml_cgraph, cuda: bool) -> Undo {
    // SAFETY: the caller's: a live graph whose node array holds `n` valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph), n).to_vec();
        let g = Graph::new(graph, nodes);
        let left = (0..n)
            .filter(|&i| !g.none[i] && (*g.nodes[i]).op == ffi::ggml_op_GGML_OP_MUL)
            .filter_map(|i| g.match_sum(i).map(|s| (i, cuda && g.ggml_fuses_sum(&s))))
            .collect();
        Undo {
            left,
            saved: Vec::new(),
            cuda,
            lazy: 0,
            compact: 0,
            one_cast: 0,
        }
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
    /// FFN mixers whose streams are made lazily (see the module doc).
    pub(crate) lazy: usize,
    /// Norm nodes of those laid out without their unwritten bytes (see the module doc).
    pub(crate) compact: usize,
    /// BF16 casts that became views of an earlier cast of the same values (see the module doc).
    pub(crate) one_cast: usize,
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
        // Last saved first: a node saved twice ends as it was first.
        for (t, was) in self.saved.iter().rev() {
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
            // Rows of zero weights skipped only under --tp-moe-ep, where they hold +0.0.
            let ep = crate::model::core::tp_ep::active() && crate::model::core::tp_ep::skip();
            let fma = if ep { FMA_SKIP } else { 1 };
            // BF16 rows read back as F32 (`--nvfp4-expert-rows-bf16`, #1183): the op reads the
            // BF16 rows (the cast is exact) and the cast goes too, as in `fold_tail` (not under
            // --tp-moe-ep, whose unread rows are marked from the F32 rows' node below).
            let (x, cast) = match (!ep).then(|| self.bf16_rows(s.experts, s.at)).flatten() {
                Some((rows, i)) => (rows, Some(i)),
                None => (s.experts, None),
            };
            dead.extend(cast);
            saved.push((a, std::ptr::read(a)));
            saved.extend(
                dead.iter()
                    .map(|&i| (self.nodes[i], std::ptr::read(self.nodes[i]))),
            );
            let srcs = [x, s.weights, gt.sh, gt.g];
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
            self.retire(&dead, root(x));
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

    /// Lazy streams (see the module doc): for every folded norm reading an FFN mixer norm's new
    /// streams, where only that mixer's up-mix (after the norm) and the folded norm read them,
    /// the streams before that norm come from another norm's node that writes them, the
    /// attention mixer's, and no node in between writes into that node's memory or the
    /// attention block output's: the FFN norm stops writing its streams and its two readers
    /// make them. Saves the nodes it changes first (each once: an undo puts back the first
    /// copy). Both norms' nodes are then laid out without their unwritten bytes
    /// ([`Self::compact`]). How many FFN mixers it made lazy, and how many nodes it laid out
    /// compactly.
    ///
    /// # Safety
    /// As [`fold_graph`], after its folds.
    unsafe fn lazy_streams(
        &mut self,
        cuda: bool,
        saved: &mut Vec<(Node, ffi::ggml_tensor)>,
    ) -> (usize, usize) {
        // SAFETY: the caller's; nodes and their sources are valid tensors.
        unsafe {
            let (mut made, mut compact) = (0, 0);
            for ni in 0..self.nodes.len() {
                let nn = self.nodes[ni];
                if self.none[ni] || !lazy_fold_target(nn) {
                    continue;
                }
                let rf = (*nn).src[0];
                let nf = (*rf).view_src;
                let Some(&fi) = self.index.get(&(nf as usize)) else {
                    continue;
                };
                if self.none[fi] || fi >= ni {
                    continue;
                }
                let Some([ra, oa, ia]) = lazy_parts(nf, cuda) else {
                    continue;
                };
                let like_ra = |t: Node| {
                    (*t).view_src == nf
                        && (*t).view_offs == 0
                        && (*t).ne == (*ra).ne
                        && (*t).nb == (*ra).nb
                };
                if !like_ra(rf) {
                    continue;
                }
                let Some(ui) = self.streams_reader(nf, ffi::ggml_nbytes(ra), ni) else {
                    continue;
                };
                let uf = self.nodes[ui];
                if !(fi < ui && ui < ni) || !lazy_mix_target(uf) || !like_ra((*uf).src[2]) {
                    continue;
                }
                let (na, ro) = (root(ra), root(oa));
                if (fi + 1..ni).any(|j| self.writes_into(j, na) || self.writes_into(j, ro)) {
                    continue;
                }
                let s0 = (*nf).data as usize;
                let mut views = Vec::new();
                self.views_over(nf, s0, s0 + ffi::ggml_nbytes(ra), &mut views);
                for t in [nf, uf, nn]
                    .into_iter()
                    .chain(views.iter().map(|&v| self.nodes[v]))
                {
                    if !saved.iter().any(|&(s, _)| s == t) {
                        saved.push((t, std::ptr::read(t)));
                    }
                }
                set_no_res(nf);
                set_lazy_streams(uf, ra, oa, ia);
                set_lazy_fold(nn, ra, oa);
                self.retire_unread(&mut views, nf);
                // Both norms' nodes without their unwritten bytes: the FFN mixer's (no streams
                // now) and the attention mixer's, alive until the next layer's norm now.
                compact += usize::from(self.compact(nf, saved));
                compact += usize::from(self.compact(root(ra), saved));
                made += 1;
            }
            (made, compact)
        }
    }

    /// For the norm node `nf` whose first `bytes` are its new streams: where every node reading
    /// it is a view and those views' bytes in the streams are read (through views) only by the
    /// node `ni` and one other node, that node (`None` without one).
    ///
    /// # Safety
    /// `nf` is a node of this graph.
    unsafe fn streams_reader(&self, nf: Node, bytes: usize, ni: usize) -> Option<usize> {
        // SAFETY: the caller's; users are valid nodes.
        unsafe {
            if (*nf).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0 {
                return None;
            }
            let s0 = (*nf).data as usize;
            let mut reads = Vec::new();
            for &u in self.users(nf) {
                let v = self.nodes[u];
                if !is_view_op((*v).op) {
                    return None;
                }
                let (a, b) = span(v);
                if a < b && a < s0 + bytes && s0 < b && !self.reads_through_views(v, &mut reads) {
                    return None;
                }
            }
            let mut other = reads.into_iter().filter(|&r| r != ni);
            let first = other.next()?;
            other.all(|r| r == first).then_some(first)
        }
    }

    /// Pushes onto `reads` every node reading `t` other than views, through views; false when
    /// `t` or a view of it is a graph output.
    ///
    /// # Safety
    /// `t` is a valid tensor.
    unsafe fn reads_through_views(&self, t: Node, reads: &mut Vec<usize>) -> bool {
        // SAFETY: the caller's; users are valid nodes.
        unsafe {
            if (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 != 0 {
                return false;
            }
            for &u in self.users(t) {
                let v = self.nodes[u];
                if !is_view_op((*v).op) {
                    reads.push(u);
                } else if !self.reads_through_views(v, reads) {
                    return false;
                }
            }
            true
        }
    }

    /// Whether node `j` runs and writes into the memory of the node `r` (in place: a view of
    /// it that is not only a view).
    ///
    /// # Safety
    /// `j` is a node index; `r` a valid tensor.
    unsafe fn writes_into(&self, j: usize, r: Node) -> bool {
        // SAFETY: the caller's.
        unsafe {
            let n = self.nodes[j];
            !self.none[j]
                && !is_view_op((*n).op)
                && (*n).op != ffi::ggml_op_GGML_OP_NONE
                && n != r
                && root(n) == r
        }
    }

    /// Pushes onto `out` every view node of `t` (through views) spanning bytes in `[a, b)`.
    ///
    /// # Safety
    /// `t` is a valid tensor.
    unsafe fn views_over(&self, t: Node, a: usize, b: usize, out: &mut Vec<usize>) {
        // SAFETY: the caller's; users are valid nodes.
        unsafe {
            for &u in self.users(t) {
                let v = self.nodes[u];
                if self.none[u] || !is_view_op((*v).op) || (*v).src[0] != t || out.contains(&u) {
                    continue;
                }
                let (x, y) = span(v);
                if x < y && x < b && a < y {
                    out.push(u);
                    self.views_over(v, a, b, out);
                }
            }
        }
    }

    /// Retires ([`Self::retire`], as views of `keep`) the view nodes `views` that no node reads
    /// any more, last first (a view read only by retired views is unread too). A view node in
    /// the graph that nothing reads holds its source's memory to the end of the graph: ggml's
    /// allocator frees a node's memory when the last view of it is freed, and frees a view when
    /// its last reader has run. Unread `R_f` views left by the lazy rewrite kept every FFN
    /// norm's node alive to the end of a prompt step (#1183: 2 GB of compute buffer became
    /// 19.7 GB at 4096 tokens).
    ///
    /// # Safety
    /// Node indices of this graph; `keep` a valid tensor.
    unsafe fn retire_unread(&mut self, views: &mut [usize], keep: Node) {
        views.sort_unstable();
        for k in (0..views.len()).rev() {
            let v = views[k];
            let t = self.nodes[v];
            // SAFETY: the caller's; readers are valid nodes.
            let read = self
                .users(t)
                .iter()
                .any(|&u| !self.none[u] && unsafe { (*self.nodes[u]).src.contains(&t) });
            // SAFETY: the caller's.
            if !read
                && unsafe { (*t).flags } & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 == 0
            {
                // SAFETY: the caller's.
                unsafe { self.retire(&[v], keep) };
            }
        }
    }

    /// Lays the norm node `node` out compactly where [`compact_plan`] takes it (#1183): its
    /// unwritten bytes go, and every view of it that runs moves to its part's new place. Only
    /// where each such view lies within one written part and every node reading `node` itself
    /// is a view. Saves the nodes it changes first (each once). Whether it did.
    ///
    /// # Safety
    /// As [`Self::lazy_streams`]; `node` a valid tensor.
    unsafe fn compact(&mut self, node: Node, saved: &mut Vec<(Node, ffi::ggml_tensor)>) -> bool {
        // SAFETY: the caller's; nodes and their sources are valid tensors.
        unsafe {
            let Some(c) = compact_plan(node) else {
                return false;
            };
            if self
                .users(node)
                .iter()
                .any(|&u| !self.none[u] && !is_view_op((*self.nodes[u]).op))
            {
                return false;
            }
            let mut moves = Vec::new();
            for (j, &v) in self.nodes.iter().enumerate() {
                if self.none[j] || (*v).view_src != node {
                    continue;
                }
                let (a, b) = ((*v).view_offs, (*v).view_offs + ffi::ggml_nbytes(v));
                let Some(&(from, to, _)) = c
                    .parts
                    .iter()
                    .find(|&&(from, _, len)| from <= a && b <= from + len)
                else {
                    return false;
                };
                moves.push((v, to + (a - from)));
            }
            for t in std::iter::once(node).chain(moves.iter().map(|&(v, _)| v)) {
                if !saved.iter().any(|&(s, _)| s == t) {
                    saved.push((t, std::ptr::read(t)));
                }
            }
            for (v, offs) in moves {
                (*v).view_offs = offs;
            }
            set_compact(node, c.floats);
            true
        }
    }

    /// One BF16 cast (see the module doc): where two nodes cast all of one F32 node's values to
    /// contiguous BF16 ([`Self::bf16_cast_source`]), no node between them writes into that node,
    /// and one of them is read by FP8 codes made from BF16
    /// ([`crate::model::fp8_act::plain_bf16`]), the later cast becomes a reshape of the earlier
    /// one (a view node reading it, which runs nothing), and the views of it views of the
    /// earlier one: its readers read the same bytes, and the earlier cast stays alive until
    /// they run. Saves the nodes it changes first (each once). How many casts went.
    ///
    /// # Safety
    /// This graph's nodes are valid tensors.
    pub(crate) unsafe fn one_cast(&mut self, saved: &mut Vec<(Node, ffi::ggml_tensor)>) -> usize {
        // SAFETY: the caller's; nodes, their sources and their readers are valid tensors.
        unsafe {
            let codes = |g: &Self, i: usize| {
                g.users(g.nodes[i])
                    .iter()
                    .any(|&u| !g.none[u] && crate::model::fp8_act::is_plain_bf16(g.nodes[u]))
            };
            let output = ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32;
            let mut first = AddrMap::default();
            let mut gone = Vec::new();
            for j in 0..self.nodes.len() {
                let Some(src) = self.bf16_cast_source(j) else {
                    continue;
                };
                match first.entry(src as usize) {
                    Entry::Vacant(v) => {
                        v.insert(j);
                    }
                    Entry::Occupied(o) => {
                        let i = *o.get();
                        // The values both casts read: no node between them writes into the
                        // source (an op on a view of it, as ggml's in-place ops and copies
                        // into a view make).
                        let kept = (i + 1..j).all(|k| {
                            let n = self.nodes[k];
                            self.none[k]
                                || is_view_op((*n).op)
                                || (*n).op == ffi::ggml_op_GGML_OP_NONE
                                || root(n) != src
                        });
                        if kept
                            && (*self.nodes[j]).flags & output == 0
                            && (codes(self, i) || codes(self, j))
                        {
                            gone.push((j, i));
                        }
                    }
                }
            }
            for &(j, i) in &gone {
                let (t, keep) = (self.nodes[j], self.nodes[i]);
                let views: Vec<Node> = self
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&v| (*v).view_src == t)
                    .collect();
                for v in std::iter::once(t).chain(views.iter().copied()) {
                    if !saved.iter().any(|&(s, _)| s == v) {
                        saved.push((v, std::ptr::read(v)));
                    }
                }
                // (`t` is contiguous and as long as `keep`, so a view of it at offset 0 is one of
                // `keep` at the same offset.)
                for v in views {
                    (*v).view_src = keep;
                }
                // A reshape of `keep` reading it, as ggml makes one: ggml's allocator keeps a
                // view's source alive until the view's readers ran, but not for a
                // `GGML_OP_NONE` view (ggml-alloc.c counts those as no view), so a source-less
                // `NONE` view as in [`Self::retire`] would let `keep`'s memory go to later
                // nodes while `t`'s readers still read it.
                let n = &mut *t;
                n.op = ffi::ggml_op_GGML_OP_RESHAPE;
                n.op_params = [0; ffi::GGML_MAX_OP_PARAMS as usize / 4];
                n.src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
                n.src[0] = keep;
                n.view_src = keep;
                n.view_offs = 0;
                self.none[j] = true;
            }
            gone.len()
        }
    }

    /// Where node `j` casts all of an F32 node's values to BF16 (ggml's `cast`: a `cpy` into a
    /// new contiguous tensor, of the node or of a contiguous view of all of it at offset 0):
    /// that node.
    ///
    /// # Safety
    /// `j` is a node index.
    unsafe fn bf16_cast_source(&self, j: usize) -> Option<Node> {
        // SAFETY: the caller's; sources are valid tensors.
        unsafe {
            let t = self.nodes[j];
            if self.none[j]
                || (*t).op != ffi::ggml_op_GGML_OP_CPY
                || (*t).type_ != ffi::ggml_type_GGML_TYPE_BF16
                || (*t).src[1] != t
                || !(*t).view_src.is_null()
                || !ffi::ggml_is_contiguous(t)
            {
                return None;
            }
            let x = (*t).src[0];
            if x.is_null() {
                return None;
            }
            let src = if (*x).view_src.is_null() {
                x
            } else if is_view_op((*x).op) && (*x).view_offs == 0 && ffi::ggml_is_contiguous(x) {
                (*x).view_src
            } else {
                return None;
            };
            let n = ffi::ggml_nelements(t);
            ((*src).view_src.is_null()
                && f32_contiguous(src)
                && ffi::ggml_nelements(x) == n
                && ffi::ggml_nelements(src) == n)
                .then_some(src)
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
