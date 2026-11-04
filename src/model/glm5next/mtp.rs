//! GLM-5.3-Flash's multi-token-prediction (MTP) head, following llama.cpp's `graph_mtp` in
//! `src/models/glm5-next.cpp` (MIT) and SGLang's `glm5_next_nextn.py` (Apache-2.0).
//!
//! The head is the GGUF's NextN layer, block `n_layer`: no hyper-connections (a plain
//! residual), a DSA layer with an indexer and a latent cache of its own, the MoE with its shared
//! expert. At position `p` it reads the embedding of the token at `p` and `h`, the target's
//! output head input at `p - 1` (the streams' normed mean) or its own output when drafting on:
//! `cur = eh_proj [rms(e) · enorm ; rms(h) · hnorm]`, then `cur += dsa(rms(cur) · attn_norm)`,
//! `cur += moe(rms(cur) · ffn_norm) + shexp(..)`, and `rms(cur) · shared_head_norm` is both the
//! next draft's `h` and, through the target's `output.weight`, the draft's logits.
//!
//! Every op is a Legion GLM op ([`ops`]); the routed experts go through the same expert tier
//! as the target's (the head's layer is the last stage's).

use crate::ggml::{Context, Graph, types};
use crate::model::core::mtp::MtpGraph;
use crate::model::core::state::KvPool;
use crate::model::core::{GRAPH_SIZE, GroupShape, Model, ModelError, Pass, StatePool, Tap};
use crate::residency::ssd_experts::trace::Predictor;

use super::ops;

impl Model {
    /// One pass of the MTP head over `shape` (see the module docs): per token its embedding
    /// and its row of `h` (`[n_embd, n]`); the draft's argmax, its probability and the next
    /// `h` of the output rows. A pass without output rows (a prompt's take-in before its last
    /// piece) only writes the head's caches.
    pub(super) fn build_mtp_glm<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let hp = &self.hp;
        let p = self.glm();
        let (e, il) = (hp.n_embd, hp.n_layer);
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        // `--glm-parallel-mtp`: the last GPU's share of a pass on every GPU (its FFN's shards
        // and the head's rows are all it has; the executor runs the other GPUs' beside it).
        if let Some(par) = self.glm_par.as_ref().filter(|p| p.plan.mtp && n_out > 0) {
            return self.glm_par_mtp(ctx, kv, rec, shape, par.plan.last());
        }
        // The sparse attention's per-group inputs too (`Arch::mtp_sparse`).
        let mut inputs = self.new_inputs(ctx, shape, true);
        let h = ctx
            .new_tensor(types::F32, &[e, n])
            .named("mtp_h")
            .as_input();
        let pass = Pass {
            ctx,
            graph: std::cell::RefCell::new(ctx.new_graph(GRAPH_SIZE)),
            tap: Tap {
                ctx,
                on: false,
                taps: Default::default(),
            },
            kv,
            rec,
            saved: None,
            spec: None,
            shape,
            pos: inputs.pos,
            cells: inputs.cells,
            ring_cells: inputs.ring_cells,
            attn: inputs.attn.clone(),
            out_rows: inputs.out_rows,
            ple: None,
            group: inputs.group.clone(),
            seq_ids: None,
            n,
            exclusive: false,
            split: None,
        };
        let w = |name: &str| self.lw(il, name);
        let tok = ctx.reshape_2d(
            ops::embed(ctx, self.w("token_embd.weight")?, inputs.tokens, 1),
            e,
            n,
        );
        let cat = ops::nextn_in(
            ctx,
            tok,
            h,
            w("nextn.enorm.weight")?,
            w("nextn.hnorm.weight")?,
            hp.rms_eps,
        );
        let cur = self.glm_lmm(ctx, il, "nextn.eh_proj.weight", cat)?;
        let x = ops::rms_norm(ctx, cur, Some(w("attn_norm.weight")?), hp.rms_eps);
        if n_out == 0 {
            // Nothing read back: the head's caches are all the pass leaves behind.
            self.glm_dsa_write(&pass, il, x)?;
            let mut graph = pass.graph.into_inner();
            // The inputs the pass does not read, allocated all the same: the step fills them.
            inputs.expand_unread_into(&mut graph);
            return Ok((
                graph,
                MtpGraph {
                    inputs,
                    h,
                    pmax: None,
                    h_out: None,
                },
            ));
        }
        let attn = self.glm_dsa(&pass, il, x)?;
        // The rest is per token: the output rows alone.
        let (attn, cur) = match inputs.out_rows {
            Some(rows) => (
                ops::get_cols(ctx, attn, rows),
                ops::get_cols(ctx, cur, rows),
            ),
            None => (attn, cur),
        };
        let ffn_inp = ops::add(ctx, attn, cur);
        let x = ops::rms_norm(ctx, ffn_inp, Some(w("ffn_norm.weight")?), hp.rms_eps);
        // The SSD expert tier's trace (#611): every tier layer's router on the draft's state.
        for l in 0..hp.n_layer {
            self.glm_ssd_predict(&pass, l, x, Predictor::Mtp, None)?;
        }
        let moe = self.glm_moe(&pass, il, x)?;
        let sh = self.glm_ffn(ctx, il, "_shexp", p.clamp_shexp[il], x)?;
        let out = ops::add(ctx, ops::add(ctx, moe, sh), ffn_inp);
        let hn = ops::rms_norm(
            ctx,
            out,
            Some(w("nextn.shared_head_norm.weight")?),
            hp.rms_eps,
        );
        let h_out = hn.as_output();
        let logits = self.glm_mm(ctx, "output.weight", hn)?.as_output();
        let argmax = ops::argmax_wide(ctx, logits).as_output();
        let pmax = ops::pmax(ctx, logits).as_output();
        let mut graph = pass.graph.into_inner();
        inputs.expand_into(&mut graph);
        for t in [h_out, logits, argmax, pmax] {
            graph.expand(t);
        }
        inputs.logits = Some(logits);
        inputs.argmax = Some(argmax);
        Ok((
            graph,
            MtpGraph {
                inputs,
                h,
                pmax: Some(pmax),
                h_out: Some(h_out),
            },
        ))
    }
}
