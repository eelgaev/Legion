//! Gemma 4's MTP assistant (`gemma4-assistant` GGUFs, one per target model): a small stack that
//! drafts from the target's own KV cache, as llama.cpp's `src/models/gemma4-assistant.cpp` and
//! `common_speculative_impl_draft_mtp` (MIT) run it.
//!
//! A pass takes a token and a residual of the target's width: the token's embedding from the
//! *target's* table (scaled by `sqrt(n_embd)`) next to the residual, projected down
//! (`nextn.pre_projection`) to the assistant's width. Each of its layers has queries only: its
//! sliding-window layers attend over the target's last sliding-window layer's cache, its
//! global layer over the target's last global layer's, with the target's head sizes and the
//! assistant's own query heads (GQA over the target's kv heads); then the usual Gemma 4 post
//! norms, GELU FFN and output scale. The final norm feeds the assistant's own (tied) output
//! head for the draft and `nextn.post_projection` for the residual of the next draft, back at
//! the target's width.
//!
//! Every draft runs at the position of the token after the target's last (the target has not
//! written it, so queries see the positions before it), and the assistant writes no cache:
//! a rejected draft needs no rollback. Its tensor names (`blk.0.attn_q.weight`,
//! `token_embd.weight`, `rope_freqs.weight`, ...) are the target's too, so the graph reads them
//! from the assistant's weights by name, never through the target's lookup.

use crate::ggml::{Context, Graph, ROPE_NEOX, Tensor, types};
use crate::gguf::GgufFile;
use crate::model::core::{GRAPH_SIZE, Tap};
use crate::model::core::{
    GroupShape, Hparams, KvPool, Model, ModelError, MtpGraph, Pass, StatePool,
};

use super::Gemma4Params;

/// The assistant's shapes.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantParams {
    pub n_layer: usize,
    /// Its own width (the target's is `Hparams::n_embd`).
    pub n_embd: i64,
    /// Query heads.
    pub n_head: i64,
    /// Per layer: sliding-window (reads the target's last sliding-window layer) or global.
    pub swa: Vec<bool>,
    pub n_ff: Vec<i64>,
    pub n_rot_full: i32,
    pub n_rot_swa: i32,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub rms_eps: f32,
}

/// The assistant of `gguf`, checked against the target `hp` / `target`.
pub(super) fn parse(
    gguf: &GgufFile,
    hp: &Hparams,
    target: &Gemma4Params,
) -> Result<AssistantParams, ModelError> {
    let arch = gguf.architecture().unwrap_or_default();
    let key = |k: &str| format!("{arch}.{k}");
    let meta = |k: &str| ModelError::Metadata(key(k));
    if arch != "gemma4-assistant" {
        return Err(ModelError::Metadata(format!(
            "{arch}: not a Gemma 4 assistant (gemma4-assistant)"
        )));
    }
    let u = |k: &str| gguf.get_u64(&key(k)).ok_or_else(|| meta(k));
    let f = |k: &str| {
        gguf.get(&key(k))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| meta(k))
    };
    let n_layer = u("block_count")? as usize;
    let swa: Vec<bool> = gguf
        .get(&key("attention.sliding_window_pattern"))
        .and_then(|v| {
            v.as_array()
                .map(|a| a.iter().map(|x| x.as_bool()).collect())
        })
        .and_then(|v: Option<Vec<bool>>| v)
        .filter(|v| v.len() >= n_layer)
        .ok_or_else(|| meta("attention.sliding_window_pattern"))?;
    let n_ff = match gguf.get(&key("feed_forward_length")) {
        Some(v) => match v.as_array() {
            Some(a) => a
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as i64)
                .collect(),
            None => vec![v.as_u64().ok_or_else(|| meta("feed_forward_length"))? as i64; n_layer],
        },
        None => return Err(meta("feed_forward_length")),
    };
    if n_ff.len() < n_layer {
        return Err(meta("feed_forward_length"));
    }
    if u("embedding_length_out")? as i64 != hp.n_embd {
        return Err(ModelError::Metadata(format!(
            "{}: an assistant for a {}-wide model, this one is {} wide",
            key("embedding_length_out"),
            u("embedding_length_out")?,
            hp.n_embd
        )));
    }
    // It reads the target's last two layers: one sliding-window, one global.
    let n = hp.n_layer;
    if n < 2 || !target.swa[n - 2] || target.swa[n - 1] {
        return Err(ModelError::Unsupported(
            "an assistant for a model whose last two layers are not sliding-window, global".into(),
        ));
    }
    if u("attention.key_length")? as i64 != target.head_dim_full
        || u("attention.key_length_swa")? as i64 != target.head_dim_swa
    {
        return Err(meta("attention.key_length"));
    }
    let n_head = u("attention.head_count")? as i64;
    for kv_layer in [n - 2, n - 1] {
        let (_, n_kv) = target.kv_dims(kv_layer);
        if n_head % n_kv != 0 {
            return Err(ModelError::Unsupported(format!(
                "{n_head} assistant heads over the {n_kv} kv heads of target layer {kv_layer}"
            )));
        }
    }
    Ok(AssistantParams {
        n_layer,
        n_embd: u("embedding_length")? as i64,
        n_head,
        swa: swa[..n_layer].to_vec(),
        n_ff: n_ff[..n_layer].to_vec(),
        n_rot_full: u("rope.dimension_count")? as i32,
        n_rot_swa: u("rope.dimension_count_swa")? as i32,
        rope_base: f("rope.freq_base")?,
        rope_base_swa: f("rope.freq_base_swa")?,
        rms_eps: f("attention.layer_norm_rms_epsilon")?,
    })
}

impl Model {
    /// A tensor of the assistant's GGUF.
    fn aw(&self, name: &str) -> Result<Tensor<'_>, ModelError> {
        self.mtp_weights()
            .and_then(|w| w.get(name))
            .ok_or_else(|| ModelError::Tensor(name.into()))
    }

    /// One assistant pass over `shape` (see the module docs): per token, the target's
    /// embedding of it and its residual row in `h`; the greedy draft, its probability and the
    /// next residual for the output rows.
    pub(super) fn build_assistant<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        let hp = &self.hp;
        let g = super::params(hp).expect("a Gemma 4 model");
        let a = g.assistant.as_ref().expect("the assistant's parameters");
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        let mut inputs = self.new_inputs(ctx, shape, false);
        let h = ctx
            .new_tensor(types::F32, &[hp.n_embd, n])
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
            group: Vec::new(),
            seq_ids: None,
            n,
            exclusive: !hp.arch.mtp_sees_own_position(),
            split: None,
        };
        // It writes no cache: keep the cell inputs in the graph so they get memory.
        pass.graph.borrow_mut().expand(inputs.cells);
        if let Some(t) = inputs.ring_cells {
            pass.graph.borrow_mut().expand(t);
        }
        let eps = a.rms_eps;
        let norm = |t: Tensor<'a>, name: &str| -> Result<Tensor<'a>, ModelError> {
            Ok(ctx.mul(ctx.rms_norm(t, eps), self.aw(name)?))
        };
        // The target's embedding of the token, next to the residual.
        let x = self.gemma4_embed(
            ctx,
            ctx.get_rows(self.w("token_embd.weight")?, inputs.tokens),
        );
        let xh = ctx.concat(x, h, 0);
        let mut cur = ctx.mul_mat(self.aw("nextn.pre_projection.weight")?, xh);
        let pos = ctx.view_1d(pass.pos, n, 0);
        for il in 0..a.n_layer {
            let blk = |s: &str| format!("blk.{il}.{s}");
            let swa = a.swa[il];
            let kv_layer = if swa { hp.n_layer - 2 } else { hp.n_layer - 1 };
            let (d, _) = hp.kv_dims(kv_layer);
            let x = norm(cur, &blk("attn_norm.weight"))?;
            let q = ctx.mul_mat(self.aw(&blk("attn_q.weight"))?, x);
            let q = ctx.reshape_3d(q, d, a.n_head, n);
            let q = norm(q, &blk("attn_q_norm.weight"))?;
            let q = if swa {
                ctx.rope_ext(
                    q,
                    pos,
                    None,
                    a.n_rot_swa,
                    ROPE_NEOX,
                    hp.n_ctx_train,
                    a.rope_base_swa,
                    1.0,
                )
            } else {
                ctx.rope_ext(
                    q,
                    pos,
                    self.aw("rope_freqs.weight").ok(),
                    a.n_rot_full,
                    ROPE_NEOX,
                    hp.n_ctx_train,
                    a.rope_base,
                    1.0,
                )
            };
            let o = self.gemma4_attend(&pass, kv_layer, swa, a.n_head, q)?;
            // Over machines (#619) each rank has its share of the heads and FFN rows: the
            // projections back to the residual are partial sums (no-ops otherwise).
            let o = self.tp_sum(ctx, ctx.mul_mat(self.aw(&blk("attn_output.weight"))?, o));
            let o = norm(o, &blk("post_attention_norm.weight"))?;
            let attn_out = ctx.add(o, cur);
            let f = norm(attn_out, &blk("ffn_norm.weight"))?;
            let up = ctx.mul_mat(self.aw(&blk("ffn_up.weight"))?, f);
            let gate = ctx.mul_mat(self.aw(&blk("ffn_gate.weight"))?, f);
            let f = self.tp_sum(
                ctx,
                ctx.mul_mat(self.aw(&blk("ffn_down.weight"))?, ctx.geglu_split(gate, up)),
            );
            let f = norm(f, &blk("post_ffw_norm.weight"))?;
            cur = ctx.mul(
                ctx.add(f, attn_out),
                self.aw(&blk("layer_output_scale.weight"))?,
            );
        }
        let cur = norm(cur, "output_norm.weight")?;

        let mut graph = pass.graph.into_inner();
        if n_out == 0 {
            graph.expand(cur);
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
        let cur = match inputs.out_rows {
            Some(rows) => ctx.get_rows(cur, rows),
            None => cur,
        };
        let embd = self.aw("token_embd.weight")?;
        let logits = ctx.mul_mat(embd, cur);
        let argmax = crate::model::cluster_select::argmax_of(ctx, logits, embd).as_output();
        let probs = ctx.soft_max(logits);
        let pmax = ctx.get_rows(
            ctx.reshape_3d(probs, 1, hp.n_vocab, n_out),
            ctx.reshape_2d(argmax, 1, n_out),
        );
        let pmax = ctx.reshape_1d(pmax, n_out).as_output();
        let h_out = ctx
            .mul_mat(self.aw("nextn.post_projection.weight")?, cur)
            .as_output();
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
