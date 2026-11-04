//! One GPU's graph of a parallel step ([`super`]): every layer, the GPU's share of the work,
//! one all-reduce per sublayer.

use super::{balance, exchange, mirror_name, prefetch, shard_name, split};
use crate::ggml::{Context, Graph, Tensor};
use crate::model::core::state::KvPool;
use crate::model::core::{
    GRAPH_SIZE, GroupShape, Inputs, Model, ModelError, Pass, SpecPool, StatePool, Tap,
};
use crate::model::glm5next::{Part, ops};
use crate::model::moe_ffn;

impl Model {
    /// GPU `me`'s graph of a parallel step over its pools `kv` (every MLA layer's latents and
    /// indexer rows), `rec` (its heads' KDA states, [`super::shard_hparams`]) and `spec` (its
    /// heads' replay inputs, for verify groups): built in place of a stage's own graph for a
    /// [`GroupShape::par`] step. The last GPU's graph ends in the logits and their argmax, and
    /// with `capture` the last layer (the MTP head drafting) in every row's head input too.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(crate) fn glm_par_stage<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        spec: Option<&'a SpecPool>,
        capture: &[usize],
        shape: &'a [GroupShape],
        me: usize,
    ) -> Result<(Graph<'a>, Inputs<'a>, Vec<(String, Tensor<'a>)>), ModelError> {
        let par = self
            .glm_par
            .as_ref()
            .ok_or_else(|| ModelError::Unsupported("a parallel step without its plan".into()))?;
        let (plan, ex) = (&par.plan, &*par.ex);
        let hp = &self.hp;
        let p = self.glm();
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        assert!(
            n > 0 && n as usize <= par.max_tokens,
            "a parallel step of {n} tokens"
        );
        if spec.is_none() && shape.iter().any(|g| g.verify) {
            return Err(ModelError::Unsupported(
                "a verify step without replay buffers".into(),
            ));
        }
        let last = me == plan.last();
        // The MTP head reads the last layer's head input of every row; no other capture.
        let capture = match capture {
            [] => false,
            [il] if *il + 1 == hp.n_layer => last,
            _ => {
                return Err(ModelError::Unsupported(format!(
                    "hidden-state captures {capture:?} of GLM-5.3-Flash (its last layer's only)"
                )));
            }
        };

        let mut inputs = self.stage_inputs(ctx, shape, true, 0..hp.n_layer);
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
            spec,
            shape,
            pos: inputs.pos,
            cells: inputs.cells,
            ring_cells: inputs.ring_cells,
            attn: inputs.attn.clone(),
            out_rows: inputs.out_rows,
            ple: inputs.ple,
            group: inputs.group.clone(),
            seq_ids: inputs.seq_ids,
            n,
            exclusive: false,
            split: None,
        };
        let e = hp.n_embd;
        // The exchanges take contiguous tensors: a copy only where one is not.
        let contig = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
        let split = shard_name("", me);

        // The streams from the first layer's owner, which embeds the tokens.
        let first = plan.owner[0];
        let embedded = if me == first {
            let x = self
                .hp
                .arch
                .embed_tokens(self, ctx, inputs.tokens)?
                .ok_or_else(|| ModelError::Unsupported("GLM embeds its tokens".into()))?;
            Some(ctx.reshape_2d(contig(x), e * p.hc, n))
        } else {
            None
        };
        let x = exchange::bcast(ctx, ex, me, first, embedded, e * p.hc, n, None);
        let mut res = ctx.reshape_3d(x, e, p.hc, n);

        // `--glm-prefetch`: what the layer before staged of this layer's experts.
        let mut staged = None;
        for il in 0..hp.n_layer {
            let mirror = plan.mirror_sfx(il, me);
            let part = Part {
                split: &split,
                mirror: &mirror,
                n_dev: plan.n_dev as i64,
            };
            // Attention: this GPU's heads, their output projection's partial sum.
            let (pre, h) = self.glm_hc_pre_on(ctx, il, "attn", res, &mirror)?;
            let a = if p.mla[il] {
                self.glm_dsa_on(&pass, il, h, part)?
            } else {
                self.glm_kda_on(&pass, il, h, part)?
            };
            let out = exchange::reduce(ctx, ex, me, contig(a), Some(res));
            res = ops::hc_post(ctx, out, None, res, &pre);

            let (pre, h) = self.glm_hc_pre_on(ctx, il, "ffn", res, &mirror)?;
            // The next layer's guessed experts copied on a side stream beside this layer.
            let ahead = std::mem::replace(&mut staged, self.glm_prefetch(&pass, il, me, h)?);
            let f = if il < p.n_dense {
                // The dense FFN: this GPU's hidden rows.
                self.glm_ffn_on(ctx, il, "", p.clamp_shexp[il], h, &split)?
            } else {
                self.glm_par_moe(&pass, il, me, h, (&mirror, &split), ahead)?
            };
            let out = exchange::reduce(ctx, ex, me, contig(f), Some(res));
            res = ops::hc_post(ctx, out, None, res, &pre);
        }

        let mut graph = pass.graph.into_inner();
        inputs.expand_into(&mut graph);
        let norm = if last {
            "output_norm.weight".to_string()
        } else {
            mirror_name("output_norm.weight", me)
        };
        // The MTP head's input, every row's head input (as `Arch::layers` captures it); the
        // logits' rows are taken out of it, each row as the cropped head gives it.
        let h_all = if capture {
            let h = ops::hc_head(ctx, res, None, self.w(&norm)?, hp.rms_eps).as_output();
            graph.expand(h);
            inputs.hidden.push(h);
            Some(h)
        } else {
            None
        };
        if n_out == 0 {
            graph.expand(res);
            return Ok((graph, inputs, Vec::new()));
        }
        // The LM head: every GPU its rows of the vocabulary, gathered on the last.
        let x = match (h_all, pass.out_rows) {
            (Some(h), Some(rows)) => ops::get_cols(ctx, h, rows),
            (Some(h), None) => h,
            (None, rows) => ops::hc_head(ctx, res, rows, self.w(&norm)?, hp.rms_eps),
        };
        let rows = super::rows(hp.n_vocab, plan.n_dev, me);
        // The last GPU reads its rows of its full copy, or (`--glm-parallel-mtp`, no full
        // copy) its shard as the others do: the same rows, the same bits.
        let part = if last && !plan.mtp {
            let w = self.w(super::HEAD)?;
            let nb1 = w.nb()[1];
            let w = ctx.view_2d(
                w,
                [w.ne()[0], rows.end - rows.start],
                nb1,
                rows.start as usize * nb1,
            );
            ops::linear(ctx, w, x, 1.0, None)
        } else {
            self.glm_mm(ctx, &shard_name(super::HEAD, me), x)?
        };
        let logits = exchange::gather(
            ctx,
            ex,
            me,
            &[contig(part)],
            &[hp.n_vocab],
            Some(plan.last()),
            Some(res),
        );
        if !last {
            graph.expand(logits);
            tracing::debug!(
                gpu = me,
                tokens = n,
                nodes = graph.nodes(),
                "parallel step graph"
            );
            return Ok((graph, inputs, Vec::new()));
        }
        let logits = logits.as_output();
        let argmax = ops::argmax(ctx, logits).as_output();
        graph.expand(logits);
        graph.expand(argmax);
        tracing::debug!(
            gpu = me,
            tokens = n,
            nodes = graph.nodes(),
            "parallel step graph"
        );
        inputs.logits = Some(logits);
        inputs.argmax = Some(argmax);
        Ok((graph, inputs, Vec::new()))
    }

    /// GPU `me`'s partial sum of MoE layer `il` over the normed input `x`: the router on every
    /// GPU (weights named with `mirror`), this GPU's experts of the chosen ones, its rows of
    /// the shared expert (named with `split`). `ahead`: what the layer before staged of this
    /// layer's experts (`--glm-prefetch`, [`super::prefetch`]).
    pub(super) fn glm_par_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        me: usize,
        x: Tensor<'a>,
        (mirror, split): (&str, &str),
        ahead: Option<prefetch::Staged<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let p = self.glm();
        let hp = &self.hp;
        let plan = &self.glm_par.as_ref().expect("a parallel step").plan;
        let owned = plan.owned(me);
        let logits = self.glm_lmm(ctx, il, &format!("ffn_gate_inp.weight{mirror}"), x)?;
        let ids = ops::router_ids(
            ctx,
            logits,
            self.lw(il, &format!("exp_probs_b.bias{mirror}"))?,
            hp.n_expert_used,
        );
        let ids = if ids.is_contiguous() {
            ids
        } else {
            ctx.cont(ids)
        };
        let w = ops::router_w(ctx, logits, ids, p.weights_norm, p.weights_scale);
        let shared = self.glm_ffn_on(ctx, il, "_shexp", p.clamp_shexp[il], x, split)?;
        let limit = p.clamp_exp[il];
        let refs = |[up, gate, down]: [Tensor<'a>; 3]| moe_ffn::ExpertRefs {
            up,
            gate,
            down,
            table: None,
        };
        let unsupported = |w: [Tensor<'a>; 3]| {
            Err(ModelError::Unsupported(format!(
                "layer {il}: experts of types {:?}",
                w.map(|t| t.ty())
            )))
        };
        let pair = self.moe_pair();
        let glm = |out| {
            let out = moe_ffn::glm_kernels(out, ops::gb10_linear());
            moe_ffn::glm_q8(out, ops::gb10_linear() && ops::expert_q8())
        };
        let Some(t) = self.expert_tier_on(il, me) else {
            // No cache entry yet (the load measures the step graphs before it builds the
            // cache): the whole experts in host memory, this GPU's choices only.
            let lw = |s: &str| self.lw(il, &format!("ffn_{s}_exps.weight"));
            let whole = [lw("up")?, lw("gate")?, lw("down")?];
            if !moe_ffn::supported_glm(&refs(whole)) {
                return unsupported(whole);
            }
            let mine = exchange::ep_mask(ctx, ids, ids, owned);
            let routed = glm(moe_ffn::moe_ffn_clamped(
                ctx,
                x,
                mine,
                w,
                refs(whole),
                pair,
                limit,
            ));
            return Ok(ops::add(ctx, routed, shared));
        };
        let tiered = [t.weight(0), t.weight(1), t.weight(2)];
        let n = x.ne()[1];
        if plan.predict_next {
            self.glm_log_next_guess(pass, il, me, x)?;
        }
        // The tiered ids of this GPU's choices, -1 for the others' (which every expert op
        // leaves out).
        let mapped = t.remap(ctx, &pass.graph, ids, n);
        let mine = exchange::ep_mask(ctx, mapped, ids, owned.clone());
        // `--glm-pair-split` (#735): the pair's uncached experts in halves on both its GPUs,
        // the sum over the matvec's ids (its halves and whole experts, [`split`]).
        if plan.pair_split
            && !t.gathers(n)
            && let Some(side) = split::side(me, plan.n_dev)
            && let Some(cached) = t.cached_view(ctx)
            && let Some(mv) = split::ep_split(ctx, ids, cached, mapped, &plan.bounds, me)
            && let Some(y) = split::expert_mv(
                ctx,
                (
                    t.n_slots(),
                    crate::model::glm5next::pass_mv_tokens(pass.shape),
                ),
                tiered,
                x,
                mv,
                limit,
                side,
            )
        {
            return Ok(exchange::ep_sum(
                ctx,
                y,
                w,
                mv,
                0..i32::MAX as u32,
                Some(shared),
            ));
        }
        let ahead = ahead.filter(|s| s.takes(ids));
        // `--glm-prefetch`: the experts the layer before staged count as cached.
        let cached = t.cached_view(ctx).map(|c| match ahead {
            Some(s) => {
                let pr = crate::residency::ep_balance::pair(me, plan.n_dev);
                if pr.len() == 2 {
                    let b = &plan.bounds;
                    let pr = (b[pr.start], b[pr.start + 1], b[pr.end]);
                    prefetch::cached_with_staged(ctx, &s, c, pr)
                } else {
                    c
                }
            }
            None => c,
        });
        // `--glm-pair-balance` (#729): the matvec's choices balanced within the GPU pair, the
        // sum over the same choices ([`balance`]); `--glm-cross-balance` (#731) over every GPU.
        let assign = |cached| {
            plan.cross_balance
                .then(|| balance::ep_assign_cross(ctx, ids, cached, &plan.bounds, me))
                .flatten()
                .or_else(|| balance::ep_assign(ctx, ids, cached, &plan.bounds, me))
        };
        let (sel, mine_mv) = match cached.filter(|_| plan.pair_balance) {
            Some(cached) => match assign(cached) {
                Some(sel) => (sel, exchange::ep_mask(ctx, mapped, sel, owned.clone())),
                None => (ids, mine),
            },
            None => (ids, mine),
        };
        // `--glm-prefetch`: where the staged ones of this GPU's choices are.
        let staged =
            ahead.map(|s| prefetch::match_staged(ctx, &s, mine_mv, t.n_slots(), hp.n_expert));
        if !t.gathers(n)
            && let Some(y) = Self::glm_expert_mv(
                ctx,
                // Verify steps past the matvec's own limit too, as the default path's (#569).
                (
                    Some(t.n_slots()),
                    crate::model::glm5next::pass_mv_tokens(pass.shape),
                ),
                tiered,
                x,
                mine_mv,
                limit,
                staged.as_ref(),
            )
        {
            return Ok(exchange::ep_sum(ctx, y, w, sel, owned, Some(shared)));
        }
        // A prompt's step: the fused op over this GPU's choices (staged as the layer stages).
        if !moe_ffn::supported_glm(&refs(tiered)) {
            return unsupported(tiered);
        }
        let routed = t
            .staged_split(ctx, &pass.graph, x, n, |x, ws, stage| {
                let out = moe_ffn::moe_ffn_staged(ctx, x, mine, w, refs(ws), pair, Some(stage));
                glm(moe_ffn::clamp(out, limit))
            })
            .or_else(|| {
                t.staged(ctx, &pass.graph, x, n, |x, ws| {
                    glm(moe_ffn::moe_ffn_clamped(
                        ctx,
                        x,
                        mine,
                        w,
                        refs(ws),
                        pair,
                        limit,
                    ))
                })
            })
            .unwrap_or_else(|| {
                glm(moe_ffn::moe_ffn_clamped(
                    ctx,
                    x,
                    mine,
                    w,
                    refs(tiered),
                    pair,
                    limit,
                ))
            });
        Ok(ops::add(ctx, routed, shared))
    }

    /// `--glm-prefetch-stats` (#745): the next layer's router (its weights and bias on this
    /// GPU) applied to MoE layer `il`'s normed input `x`, logged as that layer's guess for the
    /// prefetch predictors' report ([`crate::residency::ep_prefetch`]). Nothing past the last
    /// decoder layer. The step's own results do not read it.
    fn glm_log_next_guess<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        me: usize,
        x: Tensor<'a>,
    ) -> Result<(), ModelError> {
        let next = il + 1;
        if next >= self.hp.n_layer {
            return Ok(());
        }
        let plan = &self.glm_par.as_ref().expect("a parallel step").plan;
        let Some(t) = self.expert_tier_on(next, me) else {
            return Ok(());
        };
        let ctx = pass.ctx;
        let mirror = plan.mirror_sfx(next, me);
        let logits = self.glm_lmm(ctx, next, &format!("ffn_gate_inp.weight{mirror}"), x)?;
        let guess = ops::router_ids(
            ctx,
            logits,
            self.lw(next, &format!("exp_probs_b.bias{mirror}"))?,
            self.hp.n_expert_used,
        );
        t.log_predicted(ctx, &pass.graph, guess, x.ne()[1]);
        Ok(())
    }
}
