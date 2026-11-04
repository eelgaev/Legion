//! Muse Glimmer (`muse-glimmer` in GGUF; Muse-Glimmer-30B, a dense 28B): a dense decoder whose
//! layers alternate sliding-window and global attention, every sublayer sandwiched between RMS
//! norms (pre and post). What it has beyond a plain stack:
//!
//! - The token embeddings are RMS-normed (without a weight) before the first layer.
//! - Sliding-window layers (window `attention.sliding_window`, RoPE base `rope.freq_base_swa`,
//!   default `rope.freq_base`) rotate queries and keys (llama.cpp's normal, consecutive-pair
//!   RoPE); the global layers are NoPE. `attention.sliding_window_pattern` is one number `n`
//!   (layer `il` slides when `il % n < n - 1`, so every `n`-th layer is global) or a flag per
//!   layer.
//! - Queries and keys are RMS-normed per head (with weights; the query weight carries the
//!   checkpoint's `qk_scale_factor`), and attention is scaled by `1 / sqrt(head_dim)`.
//! - An output gate: the attention output times `sigmoid(attn_gate @ x)` (`x` the layer's
//!   normed input) before the output projection.
//! - The post-attention and post-FFN norms use eps `1e-8`; the FFN is SwiGLU.
//! - The logits are scaled by `logit_scale`, then soft-capped (`c · tanh(x / c)`).
//!
//! It runs on the core executor as Gemma 4 does: no recurrent state, the paged KV pool with
//! rings for the sliding-window layers ([`Model::window_attend`]), batching and prefix caching
//! unchanged. Its DFlash drafter reads the outputs of the layers its GGUF names. The graph
//! follows llama.cpp's `src/models/muse-glimmer.cpp` (MIT), op for op.

pub mod tiny;

use std::ops::Range;

use crate::ggml::{Context, Tensor};
use crate::gguf::GgufFile;
use crate::model::core::arch::same_as;
use crate::model::core::{Arch, ArchRef, Hparams, Model, ModelError, Pass};

/// The GGUF architecture id.
pub const ARCH: &str = "muse-glimmer";

/// `GGML_ROPE_TYPE_NORM`: consecutive pairs of a head rotate together (llama.cpp's
/// `llama_model_rope_type` for this architecture).
const ROPE_NORM: i32 = 0;

/// The eps of the post-attention and post-FFN norms (llama.cpp's `post_norm_eps`).
const POST_NORM_EPS: f32 = 1e-8;

/// llama.cpp's `load_swa_pattern(ml, 4)` default when the GGUF has no pattern.
const DEFAULT_SWA_PATTERN: usize = 4;

/// The Muse Glimmer specifics of [`Hparams`] (its [`Hparams::arch`]).
#[derive(Debug, Clone, PartialEq)]
pub struct MuseParams {
    /// Per layer: sliding-window attention with RoPE (else global, NoPE).
    pub swa: Vec<bool>,
    /// The sliding window in positions: a query at `p` sees keys `p - window + 1 ..= p`.
    pub window: usize,
    pub rope_base_swa: f32,
    /// Multiplier of the logits before the soft-cap.
    pub logit_scale: f32,
    /// Final logit soft-cap, 0 for none.
    pub softcap: f32,
}

/// Per layer, whether it slides, for a pattern of `n`: llama.cpp's `set_swa_pattern(n)`
/// (not dense-first): the last layer of every `n` is global; `n = 0` makes every layer slide.
pub fn swa_layers(n_layer: usize, n: usize) -> Vec<bool> {
    (0..n_layer).map(|il| n == 0 || il % n < n - 1).collect()
}

/// [`Hparams`] of a `muse-glimmer` GGUF.
pub(super) fn parse(g: &GgufFile) -> Result<Hparams, ModelError> {
    let key = |k: &str| format!("{ARCH}.{k}");
    let u = |k: &str| -> Result<u64, ModelError> {
        g.get_u64(&key(k))
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let f = |k: &str| -> Option<f32> { g.get(&key(k)).and_then(|v| v.as_f64()).map(|v| v as f32) };
    let n_layer = u("block_count")? as usize;
    let n_embd = u("embedding_length")? as i64;
    let n_head = u("attention.head_count")? as i64;
    let head_dim = g
        .get_u64(&key("attention.key_length"))
        .map_or(n_embd / n_head.max(1), |v| v as i64);
    if g.get_u64(&key("attention.value_length"))
        .is_some_and(|v| v as i64 != head_dim)
    {
        return Err(ModelError::Metadata(key("attention.value_length")));
    }
    let pattern_key = key("attention.sliding_window_pattern");
    let swa: Vec<bool> = match g.get(&pattern_key) {
        None => swa_layers(n_layer, DEFAULT_SWA_PATTERN),
        Some(v) => match v.as_array() {
            Some(a) => a
                .iter()
                .map(|x| x.as_bool())
                .collect::<Option<Vec<bool>>>()
                .filter(|p| p.len() >= n_layer)
                .map(|p| p[..n_layer].to_vec()),
            None => v.as_u64().map(|n| swa_layers(n_layer, n as usize)),
        }
        .ok_or(ModelError::Metadata(pattern_key))?,
    };
    let n_vocab = g
        .tensor("token_embd.weight")
        .map(|t| t.shape[1] as i64)
        .ok_or_else(|| ModelError::Tensor("token_embd.weight".into()))?;
    let rope_base = f("rope.freq_base").unwrap_or(10_000.0);
    Ok(Hparams {
        n_layer,
        n_embd,
        n_head,
        n_head_kv: u("attention.head_count_kv")? as i64,
        head_dim,
        n_rot: g
            .get_u64(&key("rope.dimension_count"))
            .unwrap_or(head_dim as u64) as i32,
        rope_sections: [0; 4],
        rope_freq_base: rope_base,
        n_ctx_train: u("context_length")? as i32,
        rms_eps: f("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| ModelError::Metadata(key("attention.layer_norm_rms_epsilon")))?,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        // Unused: every layer is attention.
        full_attention_interval: 1,
        // No recurrent layers; the state pools stay empty (the divisor stays non-zero).
        ssm_d_conv: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 1,
        ssm_d_inner: 0,
        n_vocab,
        arch: ArchRef::new(MuseParams {
            swa,
            window: u("attention.sliding_window")? as usize,
            rope_base_swa: f("rope.freq_base_swa").unwrap_or(rope_base),
            logit_scale: f("logit_scale")
                .ok_or_else(|| ModelError::Metadata(key("logit_scale")))?,
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
        }),
    })
}

/// The Muse Glimmer hyperparameters of `hp`, if it is a Muse Glimmer model.
pub fn params(hp: &Hparams) -> Option<&MuseParams> {
    hp.arch.as_any().downcast_ref::<MuseParams>()
}

impl Arch for MuseParams {
    fn name(&self) -> &'static str {
        ARCH
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    /// No recurrent layers: a sequence is its KV cache.
    fn is_recurrent(&self, _hp: &Hparams, _il: usize) -> bool {
        false
    }

    fn is_attention(&self, _hp: &Hparams, _il: usize) -> bool {
        true
    }

    fn sliding_layer(&self, hp: &Hparams, il: usize) -> bool {
        il < hp.n_layer && self.swa[il]
    }

    fn swa_window(&self) -> usize {
        self.window
    }

    /// `rms_norm(embeddings)`, without a weight.
    fn embed<'a>(&self, m: &'a Model, ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
        ctx.rms_norm(x, m.hp.rms_eps)
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        m.muse_layers(pass, x, capture, layers)
    }

    /// `logit_scale · logits`, then the soft-cap `c · tanh(x / c)` if the model has one.
    fn logits<'a>(&self, ctx: &'a Context, logits: Tensor<'a>) -> Tensor<'a> {
        let logits = ctx.scale(logits, self.logit_scale);
        match self.softcap {
            c if c > 0.0 => ctx.scale(ctx.tanh(ctx.scale(logits, 1.0 / c)), c),
            _ => logits,
        }
    }
}

impl Model {
    fn muse(&self) -> &MuseParams {
        params(&self.hp).expect("a Muse Glimmer model")
    }

    /// The layers `layers` of a Muse Glimmer stack over the residual `x`; returns the last
    /// one's output (before the final norm) and the outputs of the layers in `capture`.
    #[allow(clippy::type_complexity)]
    fn muse_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        mut x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let post_norm = |t: Tensor<'a>, il: usize, name: &str| -> Result<Tensor<'a>, ModelError> {
            Ok(ctx.mul(ctx.rms_norm(t, POST_NORM_EPS), self.lw(il, name)?))
        };
        let mut hidden = Vec::new();
        for il in layers {
            let h = self.norm_mul(pass, x, self.lw(il, "attn_norm.weight")?);
            tap.tap("attn_norm", il, h);
            let o = self.muse_attention(pass, il, h)?;
            let o = post_norm(o, il, "post_attention_norm.weight")?;
            tap.tap("attn_post_norm", il, o);
            let ffn_inp = ctx.add(o, x);
            tap.tap("ffn_inp", il, ffn_inp);

            let h = self.norm_mul(pass, ffn_inp, self.lw(il, "ffn_norm.weight")?);
            tap.tap("ffn_norm", il, h);
            let ffn = self.ffn(ctx, il, h)?;
            tap.tap("ffn_out", il, ffn);
            let ffn = post_norm(ffn, il, "post_ffw_norm.weight")?;
            tap.tap("ffn_post_norm", il, ffn);
            x = ctx.add(ffn, ffn_inp);
            tap.tap("l_out", il, x);
            if capture.contains(&il) {
                hidden.push((il, x));
            }
        }
        Ok((x, hidden))
    }

    /// RoPE of `t` (`[head_dim, heads, n]`) for a sliding-window layer.
    fn muse_rope<'a>(&'a self, pass: &Pass<'a>, t: Tensor<'a>) -> Tensor<'a> {
        let (ctx, hp) = (pass.ctx, &self.hp);
        // The positions are the first section of the (multi-section) position input.
        let pos = ctx.view_1d(pass.pos, pass.n, 0);
        ctx.rope_ext(
            t,
            pos,
            None,
            hp.n_rot,
            ROPE_NORM,
            hp.n_ctx_train,
            self.muse().rope_base_swa,
            1.0,
        )
    }

    /// Attention of layer `il` over its normed input `h`: QK-normed queries and keys (rotated
    /// on sliding-window layers), the cache write, the output gate and the output projection.
    fn muse_attention<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let (d, n_kv) = hp.kv_dims(il);
        let eps = hp.rms_eps;
        let swa = self.muse().swa[il];
        let q = self.lmm(ctx, il, "attn_q.weight", h)?;
        let k = self.lmm(ctx, il, "attn_k.weight", h)?;
        let v = self.lmm(ctx, il, "attn_v.weight", h)?;
        let gate = self.lmm(ctx, il, "attn_gate.weight", h)?;
        let q = ctx.reshape_3d(q, d, hp.n_head, n);
        let k = ctx.reshape_3d(k, d, n_kv, n);
        let v = ctx.reshape_3d(v, d, n_kv, n);
        let q = ctx.mul(ctx.rms_norm(q, eps), self.lw(il, "attn_q_norm.weight")?);
        let k = ctx.mul(ctx.rms_norm(k, eps), self.lw(il, "attn_k_norm.weight")?);
        tap.tap("Qcur_normed", il, q);
        tap.tap("Kcur_normed", il, k);
        let (q, k) = if swa {
            (self.muse_rope(pass, q), self.muse_rope(pass, k))
        } else {
            (q, k)
        };
        pass.kv.write(
            ctx,
            &mut pass.graph.borrow_mut(),
            il,
            k,
            v,
            pass.write_cells(il),
        );
        let scale = 1.0 / (d as f32).sqrt();
        let o = self.window_attend(pass, il, swa, hp.n_head, q, scale)?;
        tap.tap("attn_out", il, o);
        let o = ctx.mul(o, ctx.sigmoid(gate));
        tap.tap("attn_gated", il, o);
        self.lmm(ctx, il, "attn_output.weight", o)
    }
}

#[cfg(test)]
mod tests;
