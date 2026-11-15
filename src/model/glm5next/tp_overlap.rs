//! `--tp-overlap` (#758) in GLM-5.3-Flash's layers (#560 r11): the hyper-connection mixers of
//! a big `--tp-seq-par` step in two micro-batches ([`crate::tp_net::micro`]), so the sums'
//! wire time runs under the mixers instead of the GPU waiting for it. Each rank's share of the
//! streams is kept in two halves; the attention, the indexer, the router and the experts run on
//! all the step's tokens as before (the MoE once, so the expert weights are read once). Per
//! sublayer boundary the graph runs, in this order:
//!
//! - the reduce-scatters of both halves of the sublayer's partial output start;
//! - half 0's finishes, its `hc_post` and the next sublayer's `hc_pre` run, its all-gather
//!   starts;
//! - half 1's finishes (its transfer ran under half 0's mixer), its mixer runs (under half 0's
//!   all-gather), its all-gather starts;
//! - both all-gathers finish into the next sublayer's `[n_embd, n]` input.
//!
//! The values are those of the plain exchanges: every column's partials travel in the type the
//! step's whole tensor would (`--tp-wire`) and add in rank order, the gathered input is the
//! plain all-gather's tensor, and the mixers are per token (their kernels sum each token's
//! values in the same order whatever the token count), so the output keeps its bits.
//! `--tp-overlap ffn` runs as `mixers` here: per-half experts would read the expert weights
//! twice a step, which costs more than the wire time they would hide.
//!
//! Round 13 (share level 5, the default): micro-batch 1's all-gather and micro-batch 0's
//! reduce-scatter were still waited for, with nothing left to run under them. Now a sublayer's
//! first and last products run by micro-batch too, where all their weights are BF16 (GLM's
//! attention and shared experts in the NVFP4 checkpoint), the edges:
//!
//! - the attention's products of its input (KDA's q / k / v / f_a / beta / g_a, MLA's q_a /
//!   kv_a and the indexer's k / gate) on micro-batch 0's columns once its all-gather is in,
//!   while micro-batch 1's still travels, then on micro-batch 1's;
//! - attn_output on micro-batch 0's columns, its reduce-scatter started, then on micro-batch
//!   1's (micro-batch 0's sums travel meanwhile);
//! - a MoE layer's shared expert on micro-batch 0's columns before micro-batch 1's all-gather
//!   finishes, and on micro-batch 1's between the two reduce-scatters' starts.
//!
//! Each product runs on a rank's piece of a micro-batch at a time (256+ columns) and writes
//! its outputs into the step's `[rows, n]` tensor at those columns ([`write_into`]), so the
//! attention reads the same tensor as before. Past the GEMV every linear kernel sums each output
//! from its own row and column in one order whatever the column count (`LINEAR_SHARE_MIN`), so
//! every value keeps its bits; `LEGION_DEBUG_GLM_SHARE_LEVEL=4` gives round 12's graphs.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::{Model, ModelError, Pass, cols};
use crate::tp_net::Overlap;
use crate::tp_net::micro::{self, Started};
use crate::tp_net::seq_par::share;

use super::ops::{self, HcPre};
use super::{LINEAR_SHARE_MIN, Part};

/// The edge products of a KDA layer's input, by [`edge_groups`] bit: q / k / v (1), the gates'
/// small ones (2).
const KDA_IN: [(&str, u8); 6] = [
    ("attn_q.weight", 1),
    ("attn_k.weight", 1),
    ("attn_v.weight", 1),
    ("ssm_f_a.weight", 2),
    ("ssm_beta.weight", 2),
    ("ssm_g_a.weight", 2),
];
/// ... and of an MLA layer's: q_a (4), kv_a and the indexer's (8).
const MLA_IN: [(&str, u8); 4] = [
    ("indexer.attn_k.weight", 8),
    ("indexer_compressor_gate.weight", 8),
    ("attn_q_a.weight", 4),
    ("attn_kv_a_mqa.weight", 8),
];
/// [`edge_groups`]' bits of attn_output (both kinds of layer) and of the shared expert.
const OUT_BIT: u8 = 16;
const SHEXP_BIT: u8 = 32;

/// Which edges run (share level 5): bits of the groups above, `LEGION_DEBUG_GLM_EDGES=<mask>`
/// for A/B runs (on every rank alike); [`EDGES_DEFAULT`] without it.
fn edge_groups() -> u8 {
    static GROUPS: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *GROUPS.get_or_init(|| edge_groups_of(std::env::var("LEGION_DEBUG_GLM_EDGES").ok().as_deref()))
}

/// [`edge_groups`] given the environment's value.
pub(super) fn edge_groups_of(env: Option<&str>) -> u8 {
    env.and_then(|v| v.trim().parse::<u8>().ok())
        .map_or(EDGES_DEFAULT, |v| v & 63)
}

/// [`edge_groups`]' default.
pub(super) const EDGES_DEFAULT: u8 = 63;
/// The attention's output projection.
const ATTN_OUT: &str = "attn_output.weight";
/// The shared expert's weights.
const SHEXP: [&str; 3] = [
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
];

/// What a sublayer takes from its edges ([`Model::glm_kda_edges`], [`Model::glm_dsa_edges`]):
/// products of its input made ahead, by weight name (layer-relative), and with `tail` the
/// output projection's input returned instead of its output.
#[derive(Default)]
pub(super) struct Edges<'a> {
    pre: Vec<(&'static str, Tensor<'a>)>,
    pub(super) tail: bool,
}

impl<'a> Edges<'a> {
    /// The product with weight `name` made ahead.
    pub(super) fn get(&self, name: &str) -> Option<Tensor<'a>> {
        self.pre.iter().find(|(n, _)| *n == name).map(|&(_, t)| t)
    }
}

/// A node that computes nothing: a tensor's memory for [`write_into`]'s writers, or their
/// join ([`joined`]).
static NOTHING: ExternalOp = ExternalOp {
    name: "glm_edge_nothing",
    cpu: cpu_nothing,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_nothing),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

unsafe fn cpu_nothing(_: &ffi::ggml_tensor, _: usize, _: usize) {}

#[cfg(feature = "cuda")]
unsafe fn cuda_nothing(_: &ffi::ggml_tensor, _: *mut std::os::raw::c_void) -> Result<(), String> {
    Ok(())
}

/// A fresh F32 `[m, n]` tensor for [`write_into`]'s writers, its node after `after`'s.
pub(super) fn buffer<'a>(ctx: &'a Context, m: i64, n: i64, after: Tensor<'a>) -> Tensor<'a> {
    ctx.external(&NOTHING, types::F32, [m, n, 1, 1], &[after])
}

/// Makes the fresh node `t` (F32 `[m, c]`) write its values into columns `col..col + c` of
/// `buf` (F32 `[m, n]`, contiguous) instead of memory of its own, as `micro::finish` writes
/// into its shared output: `t` becomes a view of `buf` and reads it (its last source, which
/// the ops here do not look at), so ggml's allocator keeps `buf` alive while `t` is used.
pub(super) fn write_into<'a>(t: Tensor<'a>, buf: Tensor<'a>, col: i64) {
    let [m, c, a, b] = t.ne();
    assert!(
        (a, b) == (1, 1)
            && t.ty() == types::F32
            && buf.ty() == types::F32
            && m == buf.ne()[0]
            && col + c <= buf.ne()[1]
            && t.is_contiguous()
            && buf.is_contiguous(),
        "glm edge: [m, c] into [m, n]"
    );
    // SAFETY: a fresh node, not allocated yet; `buf` is a node of the same graph, ahead of it,
    // and `t`'s bytes fit in it at that offset.
    unsafe {
        let raw = &mut *t.raw();
        assert!(raw.view_src.is_null(), "glm edge: a fresh node");
        let slot = raw
            .src
            .iter()
            .position(|s| s.is_null())
            .expect("glm edge: a free source");
        raw.src[slot] = buf.raw();
        raw.view_src = buf.raw();
        raw.view_offs = col as usize * buf.nb()[1];
    }
}

/// `buf` once every one of `parts` (its [`write_into`] writers) ran: a node that computes
/// nothing, a view of `buf` reading all of them.
pub(super) fn joined<'a>(ctx: &'a Context, buf: Tensor<'a>, parts: &[Tensor<'a>]) -> Tensor<'a> {
    let mut args = vec![buf];
    args.extend_from_slice(parts);
    let t = ctx.external(&NOTHING, types::F32, buf.ne(), &args);
    // SAFETY: a fresh node, not allocated yet; `buf` is one of its sources, of its shape.
    unsafe {
        let raw = t.raw();
        (*raw).view_src = buf.raw();
        (*raw).view_offs = 0;
    }
    t
}

/// The fewest columns a micro-batch takes here (#560 r12), below `micro`'s [`micro::MIN_COLS`]
/// so the 1,650- and 2,470-token tail steps of long prompts split too. Only the mixers run per
/// micro-batch, and theirs are the whole share's kernels at any size from 64 tokens (`hc_pre`'s
/// tiled mix, the norm, `hc_post`), each token's values summed in one order, so a micro-batch
/// gives its tokens the whole share's bits; at 256 the halves of even a 1,024-token step stay
/// clear of that.
const MICRO_MIN: usize = 256;

/// The split layers' state.
pub(super) struct Micro<'a> {
    comm: i32,
    ranks: usize,
    /// The step's tokens.
    n: usize,
    /// Each micro-batch's streams, `[n_embd, hc, tokens]`.
    res: [Tensor<'a>; 2],
    /// The last sublayer's partial output, both reduce-scatters started, and each
    /// micro-batch's mixer weights for its `hc_post`.
    pending: Option<([Started<'a>; 2], [HcPre<'a>; 2])>,
}

/// Adds `t` to the graph now (ggml runs a graph's nodes in the order they were added).
fn add<'a>(pass: &Pass<'a>, t: Tensor<'a>) {
    pass.graph.borrow_mut().expand(t);
}

/// A sublayer input's all-gathers, micro-batch 0's finished (`first`, `[n_embd, n]` holding
/// its columns) and micro-batch 1's started.
#[derive(Clone, Copy)]
struct Gather<'a> {
    first: Tensor<'a>,
    rest: Started<'a>,
}

/// `g`'s input whole: micro-batch 1's all-gather finished into `first`.
fn gathered<'a>(pass: &Pass<'a>, g: Gather<'a>) -> Tensor<'a> {
    let both = micro::finish(pass.ctx, g.rest, Some(g.first));
    add(pass, both);
    both
}

impl Model {
    /// With `--tp-overlap` (else `None`): the split layers' state from this rank's share
    /// `res` (`[n_embd, hc, share]`, contiguous) of the streams of a `--tp-seq-par` step `sp`
    /// of `n` tokens.
    pub(super) fn glm_micro_split<'a>(
        &'a self,
        ctx: &'a Context,
        sp: Option<(i32, usize, usize)>,
        n: i64,
        res: Tensor<'a>,
    ) -> Option<Micro<'a>> {
        let (comm, rank, ranks) = sp?;
        if micro::overlap(comm) == Overlap::Off {
            return None;
        }
        let n = n as usize;
        // Short steps stay whole (the same choice on every rank).
        let floor = if super::share_level() >= 4 {
            MICRO_MIN
        } else {
            micro::MIN_COLS
        };
        if !micro::worth_at(n, ranks, floor) {
            return None;
        }
        let (a, _) = share(n, rank, ranks);
        let [e, hc, m, _] = res.ne();
        let flat = ctx.reshape_2d(res, e * hc, m);
        let res = [0, 1].map(|j| {
            let (b, c) = micro::piece(n, rank, ranks, j, false);
            let t = ctx.cont(cols(ctx, flat, (b - a) as i64, c as i64));
            ctx.reshape_3d(t, e, hc, c as i64)
        });
        Some(Micro {
            comm,
            ranks,
            n,
            res,
            pending: None,
        })
    }

    /// Layer `il`'s sublayer `which` ("attn" or "ffn") input, `[n_embd, n]`: the pending
    /// partial output finished and combined into each micro-batch's streams, the sublayer's
    /// mixer and norm run, both halves gathered from every rank (see the module docs).
    /// Returns it and each micro-batch's mixer weights.
    fn glm_micro_input<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &mut Micro<'a>,
        il: usize,
        which: &str,
    ) -> Result<(Tensor<'a>, [HcPre<'a>; 2]), ModelError> {
        let (g, pres) = self.glm_micro_gather(pass, st, il, which)?;
        Ok((gathered(pass, g), pres))
    }

    /// [`Self::glm_micro_input`] with micro-batch 1's all-gather left to finish
    /// ([`gathered`]).
    fn glm_micro_gather<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &mut Micro<'a>,
        il: usize,
        which: &str,
    ) -> Result<(Gather<'a>, [HcPre<'a>; 2]), ModelError> {
        let ctx = pass.ctx;
        let pending = st.pending.take();
        let mut gathers = Vec::with_capacity(micro::MICRO);
        let mut pres = Vec::with_capacity(micro::MICRO);
        for j in 0..micro::MICRO {
            if let Some((s, pre)) = &pending {
                let out = micro::finish(ctx, s[j], None);
                add(pass, out);
                st.res[j] = ops::hc_post(ctx, out, None, st.res[j], &pre[j]);
            }
            let (pre, h) = self.glm_hc_pre(ctx, il, which, st.res[j])?;
            let g = micro::all_gather_start(ctx, st.comm, h, st.n, j, false);
            add(pass, g.node);
            gathers.push(g);
            pres.push(pre);
        }
        let first = micro::finish(ctx, gathers[0], None);
        add(pass, first);
        let Ok(pres) = pres.try_into() else {
            unreachable!("two micro-batches")
        };
        Ok((
            Gather {
                first,
                rest: gathers[1],
            },
            pres,
        ))
    }

    /// Each rank's piece of micro-batch `j`: `(first column, columns)` of the step's.
    fn glm_micro_pieces(st: &Micro<'_>, j: usize) -> Vec<(i64, i64)> {
        (0..st.ranks)
            .map(|r| {
                let (a, c) = micro::piece(st.n, r, st.ranks, j, false);
                (a as i64, c as i64)
            })
            .collect()
    }

    /// Whether layer `il`'s products with the weights `names` run by micro-batch (the edges,
    /// share level 5): BF16 matrices without a scale, every rank's every piece of
    /// `LINEAR_SHARE_MIN`+ columns (the same kernel as the whole step's).
    fn glm_edges_take(&self, il: usize, st: &Micro<'_>, names: &[&str]) -> bool {
        super::share_level() >= 5
            && (0..micro::MICRO)
                .flat_map(|j| Self::glm_micro_pieces(st, j))
                .all(|(_, c)| c >= LINEAR_SHARE_MIN)
            && names.iter().all(|n| {
                self.lw(il, n)
                    .is_ok_and(|w| w.ty() == types::BF16 && w.ne()[2] == 1 && w.ne()[3] == 1)
                    && self.lw(il, &n.replace(".weight", ".scale")).is_err()
            })
    }

    /// The products of layer `il`'s weights `names` with its sublayer input, by micro-batch:
    /// micro-batch 0's pieces once its all-gather is in, micro-batch 1's all-gather finished,
    /// then its pieces, each written into its product's `[rows, n]` tensor. Returns the
    /// gathered input and the products.
    fn glm_edges_in<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &Micro<'a>,
        il: usize,
        names: &[&'static str],
        g: Gather<'a>,
    ) -> Result<(Tensor<'a>, Edges<'a>), ModelError> {
        let ctx = pass.ctx;
        let n = st.n as i64;
        let mut bufs = Vec::with_capacity(names.len());
        for name in names {
            let buf = buffer(ctx, self.lw(il, name)?.ne()[1], n, g.first);
            add(pass, buf);
            bufs.push((buf, Vec::new()));
        }
        let mut x = g.first;
        for j in 0..micro::MICRO {
            if j == 1 {
                x = gathered(pass, g);
            }
            for (name, (buf, parts)) in names.iter().zip(bufs.iter_mut()) {
                for (a, c) in Self::glm_micro_pieces(st, j) {
                    let y = self.glm_lmm(ctx, il, name, cols(ctx, x, a, c))?;
                    write_into(y, *buf, a);
                    add(pass, y);
                    parts.push(y);
                }
            }
        }
        let pre = names
            .iter()
            .zip(bufs)
            .map(|(&name, (buf, parts))| {
                let t = joined(ctx, buf, &parts);
                add(pass, t);
                (name, t)
            })
            .collect();
        Ok((x, Edges { pre, tail: false }))
    }

    /// The sublayer's partial output by micro-batch: for each, `piece(a, c)` of every rank's
    /// piece (F32 `[n_embd, c]`) written into the step's `[n_embd, n]` tensor, then its
    /// reduce-scatter started, pending with the mixer weights `pres`.
    fn glm_edges_out<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &mut Micro<'a>,
        after: Tensor<'a>,
        pres: [HcPre<'a>; 2],
        mut piece: impl FnMut(usize, i64, i64) -> Result<Tensor<'a>, ModelError>,
    ) -> Result<(), ModelError> {
        let ctx = pass.ctx;
        let buf = buffer(ctx, self.hp.n_embd, st.n as i64, after);
        add(pass, buf);
        let mut starts = Vec::with_capacity(micro::MICRO);
        for j in 0..micro::MICRO {
            let mut parts = Vec::new();
            for (a, c) in Self::glm_micro_pieces(st, j) {
                let y = piece(j, a, c)?;
                write_into(y, buf, a);
                add(pass, y);
                parts.push(y);
            }
            let x = joined(ctx, buf, &parts);
            add(pass, x);
            let s = micro::reduce_scatter_start(ctx, st.comm, x, st.n, j, false);
            add(pass, s.node);
            starts.push(s);
        }
        let Ok(starts) = starts.try_into() else {
            unreachable!("two micro-batches")
        };
        st.pending = Some((starts, pres));
        Ok(())
    }

    /// Both reduce-scatters of the sublayer's partial output `x` (`[n_embd, n]` or
    /// `[n_embd, n_tok, seqs]`), started, pending with the mixer weights `pres`.
    fn glm_micro_output<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &mut Micro<'a>,
        x: Tensor<'a>,
        pres: [HcPre<'a>; 2],
    ) {
        let ctx = pass.ctx;
        let x = if x.is_contiguous() { x } else { ctx.cont(x) };
        let x = ctx.reshape_2d(x, self.hp.n_embd, st.n as i64);
        let s0 = micro::reduce_scatter_start(ctx, st.comm, x, st.n, 0, false);
        add(pass, s0.node);
        let s1 = micro::reduce_scatter_start(ctx, st.comm, x, st.n, 1, false);
        add(pass, s1.node);
        st.pending = Some(([s0, s1], pres));
    }

    /// Layer `il` (of a whole-model graph) in two micro-batches.
    pub(super) fn glm_micro_layer<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        st: &mut Micro<'a>,
        sp: Option<(i32, usize, usize)>,
    ) -> Result<(), ModelError> {
        let ctx = pass.ctx;
        let p = self.glm();
        let groups = edge_groups();
        let all: &[(&'static str, u8)] = if p.mla[il] { &MLA_IN } else { &KDA_IN };
        let ins: Vec<&'static str> = all
            .iter()
            .filter(|&&(_, g)| groups & g != 0)
            .map(|&(n, _)| n)
            .collect();
        let tail = groups & OUT_BIT != 0;
        let mut names = ins.clone();
        if tail {
            names.push(ATTN_OUT);
        }
        let edges = !names.is_empty() && self.glm_edges_take(il, st, &names);
        let shexp_edges =
            groups & SHEXP_BIT != 0 && il >= p.n_dense && self.glm_edges_take(il, st, &SHEXP);
        // Once a process per kind of layer (KDA / MLA, dense / MoE): which edges run.
        static SEEN: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let bit = 1u8 << (p.mla[il] as u8 * 2 + (il >= p.n_dense) as u8);
        if SEEN.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
            tracing::info!(
                layer = il,
                mla = p.mla[il],
                groups,
                attention = edges,
                shared_expert = shexp_edges,
                tokens = st.n,
                "glm --tp-overlap edges"
            );
        }
        if edges {
            // The edges (round 13): the input's products and attn_output by micro-batch.
            let (g, pres) = self.glm_micro_gather(pass, st, il, "attn")?;
            let (h, mut edges) = self.glm_edges_in(pass, st, il, &ins, g)?;
            edges.tail = tail;
            let o = if p.mla[il] {
                self.glm_dsa_edges(pass, il, h, Part::WHOLE, sp, &edges)?
            } else {
                self.glm_kda_edges(pass, il, h, Part::WHOLE, &edges)?
            };
            if tail {
                let o = if o.is_contiguous() { o } else { ctx.cont(o) };
                self.glm_edges_out(pass, st, o, pres, |_, a, c| {
                    self.glm_lmm(ctx, il, ATTN_OUT, cols(ctx, o, a, c))
                })?;
            } else {
                self.glm_micro_output(pass, st, o, pres);
            }
        } else {
            let (h, pres) = self.glm_micro_input(pass, st, il, "attn")?;
            let out = if p.mla[il] {
                self.glm_dsa_on(pass, il, h, Part::WHOLE, sp)?
            } else {
                self.glm_kda(pass, il, h)?
            };
            self.glm_micro_output(pass, st, out, pres);
        }
        if shexp_edges {
            // The shared expert by micro-batch: micro-batch 0's before micro-batch 1's
            // all-gather finishes, micro-batch 1's after micro-batch 0's sums are on the way.
            let shexp = |x: Tensor<'a>, a: i64, c: i64| {
                self.glm_ffn(ctx, il, "_shexp", p.clamp_shexp[il], cols(ctx, x, a, c))
            };
            let (g, pres) = self.glm_micro_gather(pass, st, il, "ffn")?;
            let mut first = Vec::new();
            for (a, c) in Self::glm_micro_pieces(st, 0) {
                let sh = shexp(g.first, a, c)?;
                add(pass, sh);
                first.push((a, sh));
            }
            let h = gathered(pass, g);
            // (No SSD tier under --tp-seq-par: the room is not used.)
            let moe = self.glm_moe(pass, il, h, usize::MAX, sp)?;
            let moe = if moe.is_contiguous() {
                moe
            } else {
                ctx.cont(moe)
            };
            let moe = ctx.reshape_2d(moe, self.hp.n_embd, st.n as i64);
            // This rank's rows of the routed and the shared experts: one sum of both.
            self.glm_edges_out(pass, st, moe, pres, |j, a, c| {
                let sh = match first.iter().find(|&&(b, _)| j == 0 && b == a) {
                    Some(&(_, sh)) => sh,
                    None => shexp(h, a, c)?,
                };
                Ok(ops::add(ctx, cols(ctx, moe, a, c), sh))
            })?;
            return Ok(());
        }
        let (h, pres) = self.glm_micro_input(pass, st, il, "ffn")?;
        let out = if il < p.n_dense {
            self.glm_ffn(ctx, il, "", p.clamp_shexp[il], h)?
        } else {
            // (No SSD tier under --tp-seq-par: the room is not used.)
            let moe = self.glm_moe(pass, il, h, usize::MAX, sp)?;
            let sh = self.glm_ffn(ctx, il, "_shexp", p.clamp_shexp[il], h)?;
            // This rank's rows of the routed and the shared experts: one sum of both.
            ops::add(ctx, moe, sh)
        };
        self.glm_micro_output(pass, st, out, pres);
        Ok(())
    }

    /// Back to one share per rank: the pending output finished and combined into each
    /// micro-batch's streams, the streams joined (`[n_embd, hc, share]`).
    pub(super) fn glm_micro_join<'a>(&'a self, pass: &Pass<'a>, st: Micro<'a>) -> Tensor<'a> {
        let ctx = pass.ctx;
        let mut res = st.res;
        if let Some((s, pre)) = st.pending {
            for (j, r) in res.iter_mut().enumerate() {
                let out = micro::finish(ctx, s[j], None);
                add(pass, out);
                *r = ops::hc_post(ctx, out, None, *r, &pre[j]);
            }
        }
        ctx.concat(res[0], res[1], 2)
    }
}
