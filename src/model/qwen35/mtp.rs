//! The multi-token-prediction (MTP) head of Qwen3.5 / 3.8 (`qwen35`, `qwen35moe`), as the
//! Qwen3-Next checkpoints ship it (`mtp.*`, converted to block `n_layer` with `nextn.*` names by
//! `crate::hf::qwen35`) and as vLLM's `qwen3_next_mtp.py` (Apache-2.0) runs it:
//!
//! - at position `p` the head takes the embedding of the token at `p` and the target's last
//!   layer output after its final norm at `p - 1` (or, drafting on, its own output), each
//!   RMS-normed (`enorm`, `hnorm`), joined (`[embedding ; hidden]`) and projected by `eh_proj`
//!   (the checkpoint's `fc`);
//! - one gated full-attention decoder block (its own KV cache) and the block's FFN (dense, or
//!   the MoE of `qwen35moe`), each with its residual;
//! - a final RMS norm (`shared_head_norm`, the checkpoint's `mtp.norm`): the draft's logits
//!   through the target's output head, and the hidden fed with the draft for the next one.

use crate::ggml::{Context, Graph, types};
use crate::gguf::GgufFile;
use crate::model::core::mtp::MtpGraph;
use crate::model::core::state::KvPool;
use crate::model::core::{
    ArchRef, GroupShape, Hparams, Model, ModelError, Pass, StatePool, Tap, graph_size,
};

/// Checks that `gguf` is an MTP head of its own for a `qwen35` / `qwen35moe` model of `hp`:
/// the same architecture and width, and its blocks past the model's are the head.
pub(super) fn check_head(hp: &Hparams, gguf: &GgufFile) -> Result<(), ModelError> {
    let arch = gguf.architecture().unwrap_or_default();
    let u = |k: &str| gguf.get_u64(&format!("{arch}.{k}")).unwrap_or(0);
    let (blocks, nextn, n_embd) = (
        u("block_count"),
        u("nextn_predict_layers"),
        u("embedding_length"),
    );
    let want = if hp.moe { "qwen35moe" } else { "qwen35" };
    if arch != want
        || n_embd != hp.n_embd as u64
        || nextn == 0
        || blocks.saturating_sub(nextn) != hp.n_layer as u64
    {
        return Err(ModelError::Metadata(format!(
            "{arch}: not an MTP head for this model ({} + {nextn} blocks of {n_embd}, want {} \
             of {} for {want})",
            blocks.saturating_sub(nextn),
            hp.n_layer,
            hp.n_embd
        )));
    }
    Ok(())
}

impl Model {
    /// Builds one pass of the qwen35 MTP head over `shape` in `ctx` (see the module): per token,
    /// its embedding and the hidden row in `h`; logits (argmax and its probability) and the
    /// output hidden for the output rows.
    pub(super) fn build_mtp_qwen35<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let hp = &self.hp;
        let (e, il) = (hp.n_embd, hp.n_layer);
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        let mut inputs = self.new_inputs(ctx, shape, false);
        let h = ctx
            .new_tensor(types::F32, &[e, n])
            .named("mtp_h")
            .as_input();
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
            split: None,
        };
        let w = |name: &str| self.lw(il, name);
        let tok = ctx.get_rows(self.w("token_embd.weight")?, inputs.tokens);
        let en = ctx.mul(ctx.rms_norm(tok, hp.rms_eps), w("nextn.enorm.weight")?);
        let hn = ctx.mul(ctx.rms_norm(h, hp.rms_eps), w("nextn.hnorm.weight")?);
        let cur = self.lmm(ctx, il, "nextn.eh_proj.weight", ctx.concat(en, hn, 0))?;

        let x = ctx.mul(ctx.rms_norm(cur, hp.rms_eps), w("attn_norm.weight")?);
        if n_out == 0 {
            // No row gives an output (a prompt's take-in before its last piece): the head is
            // one layer, so its K/V rows are all the pass leaves behind. The same rows as the
            // attention below writes; the queries, the attention and the FFN of every prompt
            // position were ~5% of a 10K-token prompt's prefill on the GB10.
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
        let cur = ctx.add(self.attention(&pass, il, x)?, cur);
        // The take-in rows only had to write their K/V (the attention above did): the FFN and
        // the head run on the output rows alone. Every op from here is per token.
        let (cur, n) = match inputs.out_rows {
            Some(rows) => (ctx.get_rows(cur, rows), n_out),
            None => (cur, n),
        };
        let x = ctx.mul(
            ctx.rms_norm(cur, hp.rms_eps),
            w("post_attention_norm.weight")?,
        );
        let ffn = if hp.moe {
            self.moe(ctx, &pass.tap, &pass.graph, il, x, n)?
        } else {
            self.ffn(ctx, il, x)?
        };
        let cur = ctx.add(ffn, cur);
        let cur = ctx.mul(
            ctx.rms_norm(cur, hp.rms_eps),
            w("nextn.shared_head_norm.weight")?,
        );

        let mut graph = pass.graph.into_inner();
        let h_out = cur.as_output();
        // `--spec-draft-vocab`: drafts over the head's first rows only (on Qwen3.8-27B the NVFP4
        // output head is ~0.7 GB a pass). Verification stays exact; only acceptance can change.
        let part = match self.draft_vocab() {
            _ if self.draft_head().is_some() => None,
            v if v > 0 && (v as i64) < hp.n_vocab => {
                self.mm_rows(ctx, "output.weight", v as i64, cur)?
            }
            _ => None,
        };
        // `--spec-draft-lm-head q4`: a 4-bit copy of those rows, the scale folded in.
        let logits = match (self.draft_head(), part) {
            (Some(h), _) => ctx.mul_mat(h, cur),
            (None, Some(l)) => l,
            (None, None) => self.mm(ctx, "output.weight", cur)?,
        };
        let vocab = logits.ne()[0];
        let argmax = ctx.argmax(logits).as_output();
        let probs = ctx.soft_max(logits);
        let pmax = ctx.get_rows(
            ctx.reshape_3d(probs, 1, vocab, n_out),
            ctx.reshape_2d(argmax, 1, n_out),
        );
        let pmax = ctx.reshape_1d(pmax, n_out).as_output();
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
}

/// The architecture as it runs with a head of its own (unchanged: the head adds no parameter).
pub(super) fn head_arch() -> ArchRef {
    ArchRef::new(super::Qwen35)
}
