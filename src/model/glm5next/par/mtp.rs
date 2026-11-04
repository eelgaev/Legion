//! GLM-5.3-Flash's MTP head on every GPU (`--glm-parallel-mtp`, #717). Without it the head's
//! passes run on the last GPU alone ([`crate::model::glm5next::mtp`]): its attention, its MoE
//! over all the routed experts (one expert-cache entry serving them all) and its draft logits
//! over the whole vocabulary (a full copy of the LM head beside the last GPU's row shard), while
//! the other GPUs idle.
//!
//! Here a pass with output rows runs on every GPU, as a decoder layer's FFN does in a parallel
//! step ([`super`]):
//!
//! - **The last GPU** runs the head's front as before: the embedding, `eh_proj` and the DSA
//!   layer over its own latent cache (the head's KV pool stays on the last GPU), then hands the
//!   FFN's input (`attn + cur`, the output rows only) to every GPU (a broadcast).
//! - **Every GPU** norms it (`ffn_norm`, mirrored), routes it (the router mirrored), computes
//!   its own experts of the chosen ones (its own expert-cache entry of the head's layer,
//!   [`super::ep_layers`]) and its rows of the shared expert; one all-reduce adds the partial
//!   sums in GPU order. Then the residual, `shared_head_norm` (mirrored) and its rows of the
//!   draft logits ([`super::HEAD`] by rows), gathered on the last GPU, which takes the argmax
//!   and its probability. These logits are bitwise the unsplit head's over the same input.
//!
//! The routed sum is added in another order than one GPU's: the drafts may differ from the
//! last-GPU head's at near-ties (the verify keeps the target's output exact either way). A pass
//! without output rows (a prompt's take-in before its last piece) writes the head's caches only
//! and keeps running on the last GPU alone.

use super::{Cut, FFN_UNIT, exchange, mirror_name, shard_name};
use crate::ggml::{Context, Graph, Tensor, types};
use crate::model::core::mtp::MtpGraph;
use crate::model::core::state::KvPool;
use crate::model::core::{GRAPH_SIZE, GroupShape, Model, ModelError, Pass, StatePool, Tap};
use crate::model::glm5next::ops;

/// The weights of the MTP head's layer `il` (block `n_layer`) split for its FFN on every GPU:
/// the norms and the router mirrored, the shared expert by rows. Its front (the embedding
/// norms, `eh_proj`, the attention) stays on the last GPU.
pub fn mtp_cuts(il: usize) -> Vec<(String, Cut)> {
    let mut out: Vec<(String, Cut)> = [
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "nextn.shared_head_norm.weight",
    ]
    .iter()
    .map(|w| (format!("blk.{il}.{w}"), Cut::Mirror))
    .collect();
    for (w, c) in [
        ("gate", Cut::Outer { unit: FFN_UNIT }),
        ("up", Cut::Outer { unit: FFN_UNIT }),
        ("down", Cut::Cols { unit: FFN_UNIT }),
    ] {
        out.push((format!("blk.{il}.ffn_{w}_shexp.weight"), c));
    }
    out
}

impl Model {
    /// GPU `me`'s graph of an MTP pass with output rows over `shape` on every GPU (see the
    /// module docs): on the last GPU the head's inputs (as [`Model::build_mtp`]'s), its caches
    /// (`kv`, the head's own pool) and outputs; on the others no inputs or outputs, `kv` and
    /// `rec` unread.
    pub(crate) fn glm_par_mtp<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
        me: usize,
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let par = self.glm_par.as_ref().ok_or_else(|| {
            ModelError::Unsupported("a parallel MTP pass without its plan".into())
        })?;
        let (plan, ex) = (&par.plan, &*par.ex);
        let hp = &self.hp;
        let (e, il) = (hp.n_embd, hp.n_layer);
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        if n_out == 0 {
            return Err(ModelError::Unsupported(
                "a parallel MTP pass without output rows".into(),
            ));
        }
        let last = plan.last();
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
        let contig = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
        // The FFN's input from the last GPU: the head's front, as `build_mtp_glm`'s.
        let front = if me == last {
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
            let attn = self.glm_dsa(&pass, il, x)?;
            let (attn, cur) = match inputs.out_rows {
                Some(rows) => (
                    ops::get_cols(ctx, attn, rows),
                    ops::get_cols(ctx, cur, rows),
                ),
                None => (attn, cur),
            };
            Some(contig(ops::add(ctx, attn, cur)))
        } else {
            None
        };
        let ffn_inp = exchange::bcast(ctx, ex, me, last, front, e, n_out, None);
        // The last GPU's originals, the others' mirrors and shards.
        let mirror = if me == last {
            String::new()
        } else {
            mirror_name("", me)
        };
        let split = shard_name("", me);
        let lw = |name: &str| self.lw(il, &format!("{name}{mirror}"));
        let x = ops::rms_norm(ctx, ffn_inp, Some(lw("ffn_norm.weight")?), hp.rms_eps);
        let part = self.glm_par_moe(&pass, il, me, x, (&mirror, &split), None)?;
        let sum = exchange::reduce(ctx, ex, me, contig(part), Some(ffn_inp));
        let out = ops::add(ctx, sum, ffn_inp);
        let hn = ops::rms_norm(
            ctx,
            out,
            Some(lw("nextn.shared_head_norm.weight")?),
            hp.rms_eps,
        );
        // The draft logits: every GPU its rows of the vocabulary, gathered on the last.
        let rows = self.glm_mm(ctx, &shard_name(super::HEAD, me), hn)?;
        let logits = exchange::gather(
            ctx,
            ex,
            me,
            &[contig(rows)],
            &[hp.n_vocab],
            Some(last),
            Some(sum),
        );
        let mut graph = pass.graph.into_inner();
        if me != last {
            graph.expand(logits);
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
        let h_out = hn.as_output();
        let logits = logits.as_output();
        let argmax = ops::argmax_wide(ctx, logits).as_output();
        let pmax = ops::pmax(ctx, logits).as_output();
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
