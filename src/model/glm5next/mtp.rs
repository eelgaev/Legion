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

use crate::ggml::{Context, Graph, Tensor, types};
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
        // Under tensor parallelism over machines (#560): this rank's heads' partial output,
        // summed over the ranks.
        let attn = self.tp_sum(ctx, self.glm_dsa(&pass, il, x)?);
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
        // A chunked SSD step's room (#1205): what the context holds past the head's rest.
        let room = ctx.tensors_left().saturating_sub(super::SSD_TAIL_TENSORS);
        let moe = self.glm_moe(&pass, il, x, room)?;
        let sh = self.glm_ffn(ctx, il, "_shexp", p.clamp_shexp[il], x)?;
        let out = ops::add(ctx, self.tp_sum(ctx, ops::add(ctx, moe, sh)), ffn_inp);
        let hn = ops::rms_norm(
            ctx,
            out,
            Some(w("nextn.shared_head_norm.weight")?),
            hp.rms_eps,
        );
        let h_out = hn.as_output();
        let logits = self.glm_draft_logits(ctx, hn)?.as_output();
        // Over a copy's picked rows: their token ids.
        let argmax = ops::argmax_wide_ids(ctx, logits, self.draft_head_ids()).as_output();
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

    /// The draft's logits of `hn` (`[n_embd, n_out]`): over the first `--spec-draft-vocab`
    /// tokens only where it is set (a view of the head's first rows, so a pass reads that many
    /// rows of the 154,880 instead of all of them), else over the whole vocabulary. The rows
    /// are the target's, so a kept token's logit has the same bits; every draft is verified by
    /// the target with its whole head, so only acceptance can change. A checkpoint's NVFP4 or
    /// FP8 head drafts over the whole vocabulary.
    ///
    /// With a copy of the head ([`Model::draft_head`]: the rows of the first tokens and of the
    /// number tokens, [`draft_ids`], as stored; or a 4-bit copy of them, `--spec-draft-lm-head
    /// q4`), the logits are the copy's rows, whose token ids the argmax maps them to
    /// ([`Model::draft_head_ids`]). A Q4_K copy on a GB10 takes the Q4_K head op.
    fn glm_draft_logits<'a>(
        &'a self,
        ctx: &'a Context,
        hn: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        if let Some(head) = self.draft_head() {
            if crate::model::core::q4k_head::takes(head, hn) {
                return Ok(crate::model::core::q4k_head::mul_mat(ctx, head, hn));
            }
            return self.glm_mm_w(ctx, "output.weight", head, None, hn, None);
        }
        let w = self.w("output.weight")?;
        let vocab = draft_rows(self.draft_vocab(), w.ne()[1]);
        if vocab == w.ne()[1]
            || w.ty() == types::NVFP4
            || w.ty() == types::I8
            || self.w("output.scale").is_ok()
        {
            return self.glm_mm(ctx, "output.weight", hn);
        }
        let head = ctx.view_2d(w, [w.ne()[0], vocab], w.nb()[1], 0);
        self.glm_mm_w(ctx, "output.weight", head, None, hn, None)
    }
}

impl Model {
    /// The token ids `--spec-draft-vocab` drafts over on GLM-5.3-Flash, when they are not the
    /// first [`Self::draft_vocab`] ([`draft_ids`] of the tokens' bytes `piece`), for
    /// [`Self::set_draft_ids`]: none for a head the copy does not take (NVFP4, FP8, a scale of
    /// its own), or with `LEGION_DEBUG_GLM_DRAFT_PREFIX=1` (the first rows only, for A/B).
    pub(crate) fn glm_draft_ids<'p>(&self, piece: impl Fn(u32) -> &'p [u8]) -> Option<Vec<u32>> {
        let w = self.w("output.weight").ok()?;
        if w.ty() == types::NVFP4
            || w.ty() == types::I8
            || self.w("output.scale").is_ok()
            || std::env::var_os("LEGION_DEBUG_GLM_DRAFT_PREFIX").is_some_and(|v| v == "1")
        {
            return None;
        }
        draft_ids(self.draft_vocab(), w.ne()[1] as usize, piece)
    }
}

/// The rows a GLM-5.3-Flash draft pass scores with `--spec-draft-vocab n` (#1172): the first
/// `n` tokens and every number token past them (a piece of ASCII digits only, with or without a
/// leading space), ascending; `None` where that is just the first `n` (or `n` is 0 or covers
/// the `rows`).
///
/// GLM's vocabulary is two BPE vocabularies back to back, each in its merge order: ids up to
/// ~98,300 are mostly English and code, and the ids after them start with Chinese, the number
/// pieces and more (541 number tokens; all but 10 single digits sit between ids 98,360 and
/// 127,422). Numbers are written with those 2-5 digit pieces, so the first 65,536 rows alone
/// propose no number but single digits: lists and dates lose acceptance. The number tokens are
/// a fixed, small set (531 rows past 65,536, 0.8% more), not a frequency count of some corpus.
pub(crate) fn draft_ids<'p>(
    n: usize,
    rows: usize,
    piece: impl Fn(u32) -> &'p [u8],
) -> Option<Vec<u32>> {
    if n == 0 || n >= rows {
        return None;
    }
    let number = |b: &[u8]| {
        let d = b.strip_prefix(b" ").unwrap_or(b);
        !d.is_empty() && d.iter().all(u8::is_ascii_digit)
    };
    let extra: Vec<u32> = (n as u32..rows as u32)
        .filter(|&i| number(piece(i)))
        .collect();
    if extra.is_empty() {
        return None;
    }
    let mut ids: Vec<u32> = (0..n as u32).collect();
    ids.extend(extra);
    Some(ids)
}

/// The head rows a draft pass scores: the first `draft_vocab` of the head's `rows` (0 or at
/// least `rows`: all of them).
pub(super) fn draft_rows(draft_vocab: usize, rows: i64) -> i64 {
    match draft_vocab as i64 {
        v if v > 0 && v < rows => v,
        _ => rows,
    }
}
