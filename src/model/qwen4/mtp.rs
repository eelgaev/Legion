//! The multi-token-prediction (MTP) head of qwen4exp (Qwen3.8-Flash-Next), following
//! llama.cpp's `graph_mtp` in `src/models/qwen4exp.cpp` (MIT).
//!
//! The head is one extra decoder block (index `n_layer`: hyper-connections, dense attention
//! with a KV cache of its own, MoE) fed, at position `p`, the target's hyper-connection
//! residual after its last layer at position `p - 1` (`hnorm`) and the embedding of the token
//! at `p` (`enorm`), joined by `eh_proj`. Its output residual both predicts the token at
//! `p + 1` (through its own output mixer `nextn.hc_head_*` and the target's output head) and is
//! the residual fed with that token for the next draft. The released model ships the head as a
//! GGUF of its own that shares the target's embeddings and output head.

use crate::ggml::{Context, Graph, Tensor, types};
use crate::gguf::GgufFile;
use crate::model::core::mtp::MtpGraph;
use crate::model::core::paged;
use crate::model::core::state::KvPool;
use crate::model::core::{
    ArchRef, GroupShape, Hparams, Model, ModelError, Pass, StatePool, Tap, graph_size,
};

use super::Qwen4Params;

impl Qwen4Params {
    /// Checks that `gguf` is an MTP head for a model of `hp` and returns the parameters as
    /// they run with it: the head layer's attention type comes from the head's file.
    pub(super) fn check_mtp_head(
        &self,
        hp: &Hparams,
        gguf: &GgufFile,
    ) -> Result<ArchRef, ModelError> {
        let arch = gguf.architecture().unwrap_or_default();
        let u = |k: &str| gguf.get_u64(&format!("{arch}.{k}")).unwrap_or(0);
        let (blocks, nextn, n_embd) = (
            u("block_count"),
            u("nextn_predict_layers"),
            u("embedding_length"),
        );
        let il = hp.n_layer;
        if arch != "qwen4exp"
            || n_embd != hp.n_embd as u64
            || nextn == 0
            || blocks - nextn != il as u64
        {
            return Err(ModelError::Metadata(format!(
                "{arch}: not an MTP head for this model ({} + {nextn} blocks of {n_embd}, want {il} of {})",
                blocks.saturating_sub(nextn),
                hp.n_embd
            )));
        }
        // The head layer's attention type comes from the head's file.
        let ratio = gguf
            .get(&format!("{arch}.attention.compress_ratios"))
            .and_then(|v| {
                v.as_array()
                    .and_then(|a| a.get(il).and_then(|x| x.as_u64()))
            })
            .unwrap_or(0) as u32;
        let mut p = self.clone();
        if let Some(q) = p.qsa.as_mut() {
            q.ratios.resize(q.ratios.len().max(il + 1), 0);
            q.ratios[il] = ratio;
        }
        Ok(ArchRef::new(p))
    }
}

impl Model {
    /// Builds one pass of qwen4exp's MTP head over `shape` in `ctx`: per token, its embedding
    /// and the residual in `h`; logits (argmax and its probability) and the output residual
    /// for the output rows.
    pub(super) fn build_mtp_qwen4<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let hp = &self.hp;
        let hc = self.qwen4().hc.n;
        let (e, il) = (hp.n_embd, hp.n_layer);
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        let mut inputs = self.new_inputs(ctx, shape, false);
        let h = ctx
            .new_tensor(types::F32, &[e * hc, n])
            .named("mtp_h")
            .as_input();
        // Take-in groups read their last token's attention alone (#436): computed split over
        // the key blocks where the device's prefill kernel would walk them in two CTAs.
        let sparse = self.qwen4().qsa.as_ref().is_some_and(|q| q.layer(il));
        let split = shape
            .iter()
            .filter(|g| g.table > 0 && g.n_tok > 1 && g.last_only)
            .map(|g| {
                paged::split_floats(
                    g.n_seqs.min(paged::SPLIT_SEQS),
                    hp.kv_dims(il).1 as usize,
                    g.table,
                )
            })
            .max()
            .filter(|_| !sparse && n_out > 0 && paged::split_ok())
            .map(|len| {
                ctx.new_tensor(types::F32, &[len as i64])
                    .named("mtp_attn_split_scratch")
                    .as_input()
            });
        let pass = Pass {
            ctx,
            graph: std::cell::RefCell::new(ctx.new_graph(graph_size(shape.len()))),
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
            group: Vec::new(),
            seq_ids: None,
            n,
            exclusive: false,
            split,
        };
        let w = |name: &str| self.lw(il, name);

        // [enorm(embedding) ; hnorm(h)] per stream, projected back to the residual width.
        let tok = ctx.get_rows(self.w("token_embd.weight")?, inputs.tokens);
        let hn = ctx.mul(
            ctx.rms_norm(ctx.reshape_3d(h, e, hc, n), hp.rms_eps),
            ctx.reshape_2d(w("nextn.hnorm.weight")?, e, hc),
        );
        let en = ctx.mul(ctx.rms_norm(tok, hp.rms_eps), w("nextn.enorm.weight")?);
        let en = ctx.repeat_4d(ctx.reshape_3d(en, e, 1, n), [e, hc, n, 1]);
        // One matrix of `hc * n` columns: with the streams as a batch dimension (`ne[1] = hc`,
        // `ne[2] = n`) ggml ran a quantized mat-vec per token, which read the whole weight again
        // for each and took 92% of a prompt's take-in (17 ms for 1,024 tokens).
        let cat = ctx.concat(en, hn, 0);
        let res = ctx.reshape_3d(
            ctx.mul_mat(
                w("nextn.eh_proj.weight")?,
                ctx.reshape_2d(cat, 2 * e, hc * n),
            ),
            e,
            hc,
            n,
        );

        let (x, inject) = self.hc_mix(ctx, res, &format!("blk.{il}.hc_attn"), true, true)?;
        if n_out == 0 {
            // No row gives an output (a prompt's take-in before its last piece): the head is
            // one layer, so its K/V rows are all the pass leaves behind.
            self.write_kv(&pass, il, x)?;
            let mut graph = pass.graph.into_inner();
            // The attention inputs the pass does not read, allocated all the same: the step
            // fills them (as a dependency node's sources: the mask grows with the context).
            let attn: Vec<_> = inputs.attn.iter().flat_map(|a| a.list()).collect();
            graph.expand_unread(&attn);
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
        let attn = self.attention(&pass, il, x)?;
        // The take-in rows only had to write their K/V (the attention above did): the rest of
        // the block, the output mixer and the head run on the output rows alone. Every op from
        // here is per token, so those rows come out as the whole batch would give them.
        let mut inject = inject.expect("inject");
        let (mut res, mut attn) = (res, attn);
        if let Some(rows) = inputs.out_rows {
            res = ctx.reshape_3d(
                ctx.get_rows(ctx.reshape_2d(res, e * hc, n), rows),
                e,
                hc,
                n_out,
            );
            attn = ctx.get_rows(attn, rows);
            inject = ctx.get_rows(inject, rows);
        }
        let n = n_out;
        // Split like the target over `--tp-rank` (#733): partial outputs, summed.
        let attn = self.tp_sum(ctx, attn);
        let gpu = super::hc_chain::on_gpu(w("hc_ffn_up.weight")?);
        let res = self.hc_combine(ctx, res, attn, inject, gpu);
        let (x, inject) = self.hc_mix(ctx, res, &format!("blk.{il}.hc_ffn"), true, true)?;
        let ffn = self.tp_sum(ctx, self.moe(ctx, &pass.tap, &pass.graph, il, x, n)?);
        let res = self.hc_combine(ctx, res, ffn, inject.expect("inject"), gpu);

        let mut graph = pass.graph.into_inner();
        let flat = ctx.reshape_2d(res, e * hc, n);
        let h_out = flat.as_output();
        let (x, _) = self.hc_mix(
            ctx,
            ctx.reshape_3d(flat, e, hc, n_out),
            &format!("blk.{il}.nextn.hc_head"),
            false,
            false,
        )?;
        // Drafts over the first `draft_vocab` tokens only (a view of the head's first rows):
        // the head reads 644 MB per pass over the whole vocabulary on Qwen3.8-Flash-Next.
        // Verification stays exact, so only acceptance can change.
        let out_w = self.w("output.weight")?;
        let vocab = match self.draft_vocab() {
            v if v > 0 && (v as i64) < hp.n_vocab => v as i64,
            _ => hp.n_vocab,
        };
        let out_w = if vocab < hp.n_vocab {
            ctx.view_2d(out_w, [out_w.ne()[0], vocab], out_w.nb()[1], 0)
        } else {
            out_w
        };
        // `--spec-draft-lm-head q4`: a 4-bit copy of those rows.
        let out_w = self.draft_head().unwrap_or(out_w);
        let vocab = out_w.ne()[1];
        // A Q8_0 head at 9-16 drafting sequences on a GB10: ggml's MMQ bits, read faster.
        // One drafting sequence on a V100: ggml's mmvq bits, read faster (#814).
        let (logits, argmax, pmax) = if crate::model::core::q8_mv::takes(out_w, x) {
            let logits = crate::model::core::q8_mv::mul_mat(ctx, out_w, x);
            Self::draft_pick(ctx, logits, vocab, n_out)
        } else if crate::model::core::q8_head::pick::takes(out_w, x) {
            // The pick from the head's partials, with its bits (#814).
            let (logits, argmax, pmax) = crate::model::core::q8_head::pick::head(ctx, out_w, x);
            (logits, argmax.as_output(), pmax.as_output())
        } else if crate::model::core::q8_head::takes(out_w, x) {
            let logits = crate::model::core::q8_head::mul_mat(ctx, out_w, x);
            Self::draft_pick(ctx, logits, vocab, n_out)
        } else if crate::model::core::q4k_head::pick_takes(out_w, x) {
            // 1-7 drafting sequences on that op: the picks from its partials, their bits (#836).
            let (logits, argmax, pmax) = crate::model::core::q4k_head::head(ctx, out_w, x);
            (logits, argmax.as_output(), pmax.as_output())
        } else if crate::model::core::q4k_head::takes(out_w, x) {
            // The Q4_K copy (`--spec-draft-lm-head q4`) of 1-8 sequences on a GB10 (#836).
            let logits = crate::model::core::q4k_head::mul_mat(ctx, out_w, x);
            Self::draft_pick(ctx, logits, vocab, n_out)
        } else {
            Self::draft_pick(ctx, ctx.mul_mat(out_w, x), vocab, n_out)
        };
        graph.expand(h_out);
        graph.expand(argmax);
        graph.expand(pmax);
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

    /// The draft pick of `logits` (`[vocab, n_out]`): (logits, the argmax, its probability).
    fn draft_pick<'a>(
        ctx: &'a Context,
        logits: Tensor<'a>,
        vocab: i64,
        n_out: i64,
    ) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
        let argmax = crate::model::cluster_select::argmax(ctx, logits).as_output();
        let probs = ctx.soft_max(logits);
        let pmax = ctx.get_rows(
            ctx.reshape_3d(probs, 1, vocab, n_out),
            ctx.reshape_2d(argmax, 1, n_out),
        );
        let pmax = ctx.reshape_1d(pmax, n_out).as_output();
        (logits, argmax, pmax)
    }
}
