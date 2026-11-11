//! `--tp-overlap` (#758) in Qwen3.8-Flash-Next's layers: the sequence-parallel layers of a big
//! `--tp-seq-par` step with every rank's share of the tokens in two micro-batches
//! ([`crate::tp_net::micro`]). The streams and the hyper-connection mixers stay per
//! micro-batch; the attention (and without `ffn` the MoE) runs on all the tokens as before.
//! Per block the graph runs, in this order:
//!
//! - the reduce-scatters of both micro-batches' shares of the block's partial output start;
//! - micro-batch 0's finishes, its mixer runs, its all-gather starts;
//! - micro-batch 1's finishes (its transfer overlapped micro-batch 0's mixer), its mixer
//!   runs (overlapping micro-batch 0's all-gather), its all-gather starts;
//! - the all-gathers finish into the next block's input, or with `ffn` before the MoE:
//!   micro-batch 0's finishes, its experts run and its reduce-scatter starts, then micro-batch
//!   1's (its all-gather overlapped micro-batch 0's experts, micro-batch 0's reduce-scatter
//!   overlaps its own).
//!
//! Every node goes into the graph at once, in that order (ggml runs a graph's nodes in the
//! order they were added), so the GPU stream sees each exchange's start, the other
//! micro-batch's work, then its finish.

use crate::ggml::{Context, Tensor};
use crate::model::core::{Model, ModelError, Pass, cols};
use crate::tp_net::Overlap;
use crate::tp_net::micro::{self, Started};
use crate::tp_net::seq_par::share;

/// One value per micro-batch.
fn two<T>(v: Vec<T>) -> [T; 2] {
    let Ok(v) = v.try_into() else {
        unreachable!("two micro-batches")
    };
    v
}

/// What the next mixer combines into a micro-batch's streams.
pub(super) enum Pending<'a> {
    /// The block outputs, summed already (the layer before the split), and the injects.
    Summed([Option<(Tensor<'a>, Tensor<'a>)>; 2]),
    /// The block's partial output with both reduce-scatters started, and the injects.
    Started([Started<'a>; 2], [Tensor<'a>; 2]),
}

/// The split layers' state.
pub(super) struct Micro<'a> {
    pub(super) comm: i32,
    /// The step's tokens.
    pub(super) n: usize,
    /// The MoE per micro-batch too (`--tp-overlap ffn`).
    pub(super) ffn: bool,
    /// Each micro-batch's streams, `[n_embd, hc, tokens]`.
    pub(super) res: [Tensor<'a>; 2],
    pub(super) pending: Pending<'a>,
}

impl Model {
    /// With `--tp-overlap` (else `None`): the split layers' state from this rank's share of
    /// the streams `res` (`[n_embd, hc, share]`, contiguous) of a step of `n` tokens and the
    /// pending output on that share (taken), each in its micro-batches' columns.
    pub(super) fn micro_split<'a>(
        &'a self,
        sp: Option<(i32, usize, usize)>,
        n: i64,
        res: Tensor<'a>,
        pending: &mut Option<(Tensor<'a>, Tensor<'a>)>,
        ctx: &'a Context,
    ) -> Option<Micro<'a>> {
        let (comm, rank, ranks) = sp?;
        let ffn = match micro::overlap(comm) {
            Overlap::Off => return None,
            Overlap::Mixers => false,
            Overlap::Ffn => true,
        };
        let n = n as usize;
        // Short steps stay whole (the same choice on every rank).
        if !micro::worth(n, ranks) {
            return None;
        }
        let (a, _) = share(n, rank, ranks);
        let [e, hc, ..] = res.ne();
        let flat = ctx.reshape_2d(res, e * hc, res.ne()[2]);
        let part = |t: Tensor<'a>, j: usize| {
            let (b, m) = micro::piece(n, rank, ranks, j, false);
            ctx.cont(cols(ctx, t, (b - a) as i64, m as i64))
        };
        let res = [0, 1].map(|j| {
            let t = part(flat, j);
            ctx.reshape_3d(t, e, hc, t.ne()[1])
        });
        let p = pending.take();
        let pending = Pending::Summed([0, 1].map(|j| p.map(|(o, i)| (part(o, j), part(i, j)))));
        Some(Micro {
            comm,
            n,
            ffn,
            res,
            pending,
        })
    }

    /// Adds `t` to the graph now (see the module docs).
    fn add<'a>(pass: &Pass<'a>, t: Tensor<'a>) {
        pass.graph.borrow_mut().expand(t);
    }

    /// Both micro-batches' mixers `prefix`: the pending outputs combined into the streams,
    /// the block inputs' all-gathers started (`compact`: each to a tensor of its own).
    /// Returns them and the mixers' injects.
    fn micro_mix<'a>(
        &'a self,
        pass: &Pass<'a>,
        st: &mut Micro<'a>,
        prefix: &str,
        compact: bool,
    ) -> Result<([Started<'a>; 2], [Tensor<'a>; 2]), ModelError> {
        let ctx = pass.ctx;
        let mut gathers = Vec::with_capacity(micro::MICRO);
        let mut injects = Vec::with_capacity(micro::MICRO);
        for j in 0..micro::MICRO {
            let pending = match &st.pending {
                Pending::Summed(p) => p[j],
                Pending::Started(s, inject) => {
                    let out = micro::finish(ctx, s[j], None);
                    Self::add(pass, out);
                    Some((out, inject[j]))
                }
            };
            let (res, h, inject) = self.hc_mix_after(ctx, st.res[j], pending, prefix)?;
            st.res[j] = res;
            let g = micro::all_gather_start(ctx, st.comm, h, st.n, j, compact);
            Self::add(pass, g.node);
            gathers.push(g);
            injects.push(inject);
        }
        Ok((two(gathers), two(injects)))
    }

    /// The step's `[n_embd, n]` block input from both micro-batches' all-gathers.
    fn micro_input<'a>(pass: &Pass<'a>, g: [Started<'a>; 2]) -> Tensor<'a> {
        let first = micro::finish(pass.ctx, g[0], None);
        Self::add(pass, first);
        let both = micro::finish(pass.ctx, g[1], Some(first));
        Self::add(pass, both);
        both
    }

    /// Both reduce-scatters of the step's partial output `x` (`[n_embd, n]`), started.
    fn micro_scatter<'a>(pass: &Pass<'a>, st: &Micro<'a>, x: Tensor<'a>) -> [Started<'a>; 2] {
        let s0 = micro::reduce_scatter_start(pass.ctx, st.comm, x, st.n, 0, false);
        Self::add(pass, s0.node);
        let s1 = micro::reduce_scatter_start(pass.ctx, st.comm, x, st.n, 1, false);
        Self::add(pass, s1.node);
        [s0, s1]
    }

    /// Layer `il` of the split layers in two micro-batches.
    pub(super) fn micro_layer<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        st: &mut Micro<'a>,
    ) -> Result<(), ModelError> {
        let ctx = pass.ctx;
        let (g, inject) = self.micro_mix(pass, st, &format!("blk.{il}.hc_attn"), false)?;
        let h = Self::micro_input(pass, g);
        let attn = if self.hp.is_recurrent(il) {
            self.delta_net(pass, il, h)?
        } else {
            self.attention(pass, il, h)?
        };
        st.pending = Pending::Started(Self::micro_scatter(pass, st, attn), inject);
        let (g, inject) = self.micro_mix(pass, st, &format!("blk.{il}.hc_ffn"), st.ffn)?;
        if !st.ffn {
            let h = Self::micro_input(pass, g);
            let n = h.ne()[1];
            let ffn = self.moe(ctx, &pass.tap, &pass.graph, il, h, n, pass.verify())?;
            st.pending = Pending::Started(Self::micro_scatter(pass, st, ffn), inject);
            return Ok(());
        }
        let mut scatters = Vec::with_capacity(micro::MICRO);
        for (j, g) in g.into_iter().enumerate() {
            let h = micro::finish(ctx, g, None);
            Self::add(pass, h);
            let n = h.ne()[1];
            let ffn = self.moe(ctx, &pass.tap, &pass.graph, il, h, n, pass.verify())?;
            let s = micro::reduce_scatter_start(ctx, st.comm, ffn, st.n, j, true);
            Self::add(pass, s.node);
            scatters.push(s);
        }
        st.pending = Pending::Started(two(scatters), inject);
        Ok(())
    }

    /// Back to one share per rank: the pending outputs finished and combined into each
    /// micro-batch's streams, the streams joined (`[n_embd, hc, share]`).
    pub(super) fn micro_join<'a>(&'a self, pass: &Pass<'a>, st: Micro<'a>) -> Tensor<'a> {
        let ctx = pass.ctx;
        let mut res = st.res;
        for (j, r) in res.iter_mut().enumerate() {
            let pending = match &st.pending {
                Pending::Summed(p) => p[j],
                Pending::Started(s, inject) => {
                    let out = micro::finish(ctx, s[j], None);
                    Self::add(pass, out);
                    Some((out, inject[j]))
                }
            };
            if let Some((out, inject)) = pending {
                // `--tp-overlap` keeps ggml's combine (#836 ported the other paths).
                *r = self.hc_combine(ctx, *r, out, inject, false);
            }
        }
        ctx.concat(res[0], res[1], 2)
    }
}
