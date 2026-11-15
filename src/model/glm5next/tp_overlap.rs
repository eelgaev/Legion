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

use crate::ggml::{Context, Tensor};
use crate::model::core::{Model, ModelError, Pass, cols};
use crate::tp_net::Overlap;
use crate::tp_net::micro::{self, Started};
use crate::tp_net::seq_par::share;

use super::ops::{self, HcPre};

/// The split layers' state.
pub(super) struct Micro<'a> {
    comm: i32,
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
        if !micro::worth(n, ranks) {
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
        let both = micro::finish(ctx, gathers[1], Some(first));
        add(pass, both);
        let Ok(pres) = pres.try_into() else {
            unreachable!("two micro-batches")
        };
        Ok((both, pres))
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
        let (h, pres) = self.glm_micro_input(pass, st, il, "attn")?;
        let out = if p.mla[il] {
            self.glm_dsa_on(pass, il, h, super::Part::WHOLE, sp)?
        } else {
            self.glm_kda(pass, il, h)?
        };
        self.glm_micro_output(pass, st, out, pres);
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
