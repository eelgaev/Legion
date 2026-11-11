//! Gemma 4 (`gemma4` in GGUF): the dense 31B, the 26B-A4B MoE and the small E2B / E4B. Every
//! layer is attention plus a GELU-gated FFN, each sandwiched between RMS norms (pre and post),
//! with a per-layer output scale. What varies per layer:
//!
//! - Sliding-window layers (window `attention.sliding_window`, RoPE base
//!   `rope.freq_base_swa`, head size `key_length_swa`) and global layers (head size
//!   `key_length`, RoPE base `rope.freq_base` with per-dimension frequency factors,
//!   `rope_freqs.weight`), each with its own kv-head count. Queries and keys are RMS-normed
//!   (with weights), values too (without), and attention is unscaled (scale 1).
//! - A global layer without `attn_v` (31B) uses its key projection as the values.
//! - Shared KV (E2B / E4B): the last `attention.shared_kv_layers` layers compute no keys or
//!   values; they read the KV cache of the last sliding-window or global layer before them
//!   ([`Gemma4Params::kv_source`]). See [`crate::model::core::KvPool`] for how those reads are bounded.
//! - MoE layers (26B-A4B): the dense FFN becomes a shared expert beside 128 routed GELU experts
//!   with a softmax router on the normed, scaled residual.
//! - Per-layer token embeddings (E2B / E4B): a second embedding table, projected and gated
//!   into every layer's output.
//!
//! The final logits are soft-capped (`30 · tanh(x / 30)`).
//!
//! It runs on the core executor: no recurrent state, the paged KV pool with one row width per
//! layer, batching and prefix caching unchanged. The graph follows llama.cpp's
//! `src/models/gemma4.cpp` and `llm_graph_context`'s `build_ffn` / `build_moe_ffn` /
//! `build_attn_mha` (MIT), op for op, so the CPU logits match llama.cpp's bit for bit with
//! flash attention off.

mod assistant;
pub mod tiny;

pub use assistant::AssistantParams;

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ROPE_NEOX, Tensor};
use crate::gguf::GgufFile;

use crate::ggml::Graph;
use crate::model::core::arch::same_as;
use crate::model::core::{
    Arch, ArchRef, GroupShape, Hparams, Inputs, KvPool, Model, ModelError, MtpGraph, Pass,
    StatePool,
};
use crate::model::core::{Group, stable_fusions};

/// Whether graphs built from now on share one positions view across the layers' RoPEs.
static HEAD_ROPE_FUSED: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on make the positions view once, ahead of the layers,
/// for every RoPE; `legion serve --head-rope-fused` turns it on. A view made per RoPE lands in
/// the graph between the head norm's `mul` and the `rope`, so ggml-cuda runs the q and k heads'
/// norm and weight as one kernel and the RoPE as another; with the shared view it runs its one
/// fused norm + weight + RoPE kernel (8K prompts ~4% faster on a DGX Spark (GB10)). Off by
/// default: the fused kernel's output is not the two kernels' bit for bit, so greedy output
/// moves.
pub fn set_head_rope_fused(on: bool) {
    HEAD_ROPE_FUSED.store(on, Ordering::Relaxed);
}

/// The Gemma 4 specifics of [`Hparams`] (its [`Hparams::arch`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Params {
    /// Per layer: sliding-window attention (else global).
    pub swa: Vec<bool>,
    /// Per layer: kv heads.
    pub n_head_kv: Vec<i64>,
    /// Head size of the global layers (also `Hparams::head_dim`).
    pub head_dim_full: i64,
    /// Head size of the sliding-window layers.
    pub head_dim_swa: i64,
    /// Rotated dimensions of the sliding-window layers (the global ones': `Hparams::n_rot`).
    pub n_rot_swa: i32,
    pub rope_base_swa: f32,
    /// The sliding window in positions: a query at `p` sees keys `p - window + 1 ..= p`.
    pub window: usize,
    /// Layers `n_layer_kv..` read another layer's KV cache (`n_layer` without shared KV).
    pub n_layer_kv: usize,
    /// Per layer: the dense FFN's (or shared expert's) size.
    pub n_ff: Vec<i64>,
    /// Final logit soft-cap, 0 for none.
    pub softcap: f32,
    /// Per-layer token embedding size, 0 for none (31B, 26B-A4B).
    pub n_embd_per_layer: i64,
    /// The MTP assistant, once its GGUF is loaded ([`Model::load_mtp`]).
    pub assistant: Option<AssistantParams>,
    /// Per layer with a compact cache (`--gemma4-compact-global-kv`, #551): its rotated key
    /// dimensions per head ([`Arch::compact_kv`]); empty or 0 for none.
    pub compact: Vec<i64>,
    /// Per dimension of a compact layer's heads: whether RoPE rotates it.
    pub compact_rot: Vec<bool>,
}

impl Gemma4Params {
    /// `(head size, kv heads)` of layer `il`.
    pub fn kv_dims(&self, il: usize) -> (i64, i64) {
        (self.head_dim(il), self.n_head_kv[il])
    }

    fn head_dim(&self, il: usize) -> i64 {
        if self.swa[il] {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    /// The layer whose KV cache layer `il` reads: itself below `n_layer_kv`, else the last
    /// layer of its kind (sliding-window or global) with a cache. llama.cpp's rule
    /// (`n_layer_kv_from_start - (is_swa ? 2 : 1)`): the last two cached layers are one of each.
    pub fn kv_source(&self, il: usize) -> usize {
        if il < self.n_layer_kv {
            il
        } else if self.swa[il] {
            self.n_layer_kv - 2
        } else {
            self.n_layer_kv - 1
        }
    }
}

/// [`Hparams`] of a `gemma4` GGUF.
pub(super) fn parse(g: &GgufFile) -> Result<Hparams, ModelError> {
    let arch = "gemma4";
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| -> Result<u64, ModelError> {
        g.get_u64(&key(k))
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let f = |k: &str| -> Option<f32> { g.get(&key(k)).and_then(|v| v.as_f64()).map(|v| v as f32) };
    let n_layer = u("block_count")? as usize;
    // Per-layer keys are arrays, or one number for every layer.
    let per_layer = |k: &str| -> Result<Vec<i64>, ModelError> {
        let v = g.get(&key(k)).ok_or_else(|| ModelError::Metadata(key(k)))?;
        let out: Option<Vec<i64>> = match v.as_array() {
            Some(a) => a.iter().map(|x| x.as_u64().map(|x| x as i64)).collect(),
            None => v.as_u64().map(|x| vec![x as i64; n_layer]),
        };
        out.filter(|o| o.len() >= n_layer)
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let swa: Vec<bool> = g
        .get(&key("attention.sliding_window_pattern"))
        .and_then(|v| {
            v.as_array()
                .map(|a| a.iter().map(|x| x.as_bool()).collect())
        })
        .and_then(|v: Option<Vec<bool>>| v)
        .filter(|v| v.len() >= n_layer)
        .ok_or_else(|| ModelError::Metadata(key("attention.sliding_window_pattern")))?;
    let n_head_kv = per_layer("attention.head_count_kv")?;
    let n_ff = per_layer("feed_forward_length")?;
    let shared = g.get_u64(&key("attention.shared_kv_layers")).unwrap_or(0) as usize;
    let n_layer_kv = n_layer - shared;
    if shared > 0 {
        // The last two cached layers must be one sliding-window and one global layer.
        let ok = n_layer_kv >= 2 && swa[n_layer_kv - 2] && !swa[n_layer_kv - 1];
        if !ok {
            return Err(ModelError::Metadata(key("attention.shared_kv_layers")));
        }
    }
    let head_dim = u("attention.key_length")? as i64;
    let head_dim_swa = u("attention.key_length_swa")? as i64;
    if u("attention.value_length")? as i64 != head_dim
        || u("attention.value_length_swa")? as i64 != head_dim_swa
    {
        return Err(ModelError::Metadata(key("attention.value_length")));
    }
    let n_vocab = g
        .tensor("token_embd.weight")
        .map(|t| t.shape[1] as i64)
        .ok_or_else(|| ModelError::Tensor("token_embd.weight".into()))?;
    let moe = g.get_u64(&key("expert_count")).unwrap_or(0) > 0;
    let rope_base = f("rope.freq_base").unwrap_or(10_000.0);
    Ok(Hparams {
        n_layer,
        n_embd: u("embedding_length")? as i64,
        n_head: u("attention.head_count")? as i64,
        n_head_kv: n_head_kv[..n_layer].iter().copied().max().unwrap_or(1),
        head_dim,
        n_rot: g
            .get_u64(&key("rope.dimension_count"))
            .unwrap_or(head_dim as u64) as i32,
        rope_sections: [0; 4],
        rope_freq_base: rope_base,
        n_ctx_train: u("context_length")? as i32,
        rms_eps: f("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| ModelError::Metadata(key("attention.layer_norm_rms_epsilon")))?,
        moe,
        n_expert: if moe { u("expert_count")? as i64 } else { 0 },
        n_expert_used: if moe {
            u("expert_used_count")? as i64
        } else {
            0
        },
        // Unused: every layer is attention.
        full_attention_interval: 1,
        // No recurrent layers; the state pools stay empty (the divisor stays non-zero).
        ssm_d_conv: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 1,
        ssm_d_inner: 0,
        n_vocab,
        arch: ArchRef::new(Gemma4Params {
            swa: swa[..n_layer].to_vec(),
            n_head_kv: n_head_kv[..n_layer].to_vec(),
            head_dim_full: head_dim,
            head_dim_swa,
            n_rot_swa: g
                .get_u64(&key("rope.dimension_count_swa"))
                .unwrap_or(head_dim_swa as u64) as i32,
            rope_base_swa: f("rope.freq_base_swa").unwrap_or(rope_base),
            window: u("attention.sliding_window")? as usize,
            n_layer_kv,
            n_ff: n_ff[..n_layer].to_vec(),
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            n_embd_per_layer: g
                .get_u64(&key("embedding_length_per_layer_input"))
                .unwrap_or(0) as i64,
            assistant: None,
            compact: Vec::new(),
            compact_rot: Vec::new(),
        }),
    })
}

/// The Gemma 4 hyperparameters of `hp`, if it is a Gemma 4 model.
pub fn params(hp: &Hparams) -> Option<&Gemma4Params> {
    hp.arch.as_any().downcast_ref::<Gemma4Params>()
}

/// Per dimension of a `d`-wide head: whether NEOX RoPE over the first `n_rot` dims with
/// frequency factors `factors` (one per pair, dims `i` and `i + n_rot / 2`; none: all 1)
/// rotates it. A factor of 1e20 or more turns a pair by under 1e-14 radians at a million
/// positions: it stays put, as dims past `n_rot` do.
pub fn rotated_dims(d: i64, n_rot: i64, factors: Option<&[f32]>) -> Vec<bool> {
    let half = (n_rot.min(d) / 2) as usize;
    let mut out = vec![false; d as usize];
    for i in 0..half {
        if factors.and_then(|f| f.get(i)).is_none_or(|&f| f < 1e20) {
            out[i] = true;
            out[i + half] = true;
        }
    }
    out
}

/// The runs of `rot` ([`rotated_dims`]): `(start, len, rotated)`, in order.
fn dim_runs(rot: &[bool]) -> Vec<(i64, i64, bool)> {
    let mut out: Vec<(i64, i64, bool)> = Vec::new();
    for (i, &r) in rot.iter().enumerate() {
        match out.last_mut() {
            Some(last) if last.2 == r => last.1 += 1,
            _ => out.push((i as i64, 1, r)),
        }
    }
    out
}

/// The dims `[start, start + len)` of each `(x, start, len)` (`x` `[d, a, b]`, F32), side by
/// side in that order: `[Σ len, a, b]`, contiguous.
fn pick<'a>(
    ctx: &'a Context,
    parts: impl IntoIterator<Item = (Tensor<'a>, i64, i64)>,
) -> Tensor<'a> {
    let views = parts.into_iter().map(|(x, start, len)| {
        let (ne, nb) = (x.ne(), x.nb());
        ctx.view_3d(
            x,
            [len, ne[1], ne[2]],
            [nb[1], nb[2]],
            start as usize * nb[0],
        )
    });
    let mut n = 0;
    let out = views
        .inspect(|_| n += 1)
        .reduce(|a, b| ctx.concat(a, b, 0))
        .expect("no dims to pick");
    if n == 1 { ctx.cont(out) } else { out }
}

impl Arch for Gemma4Params {
    fn name(&self) -> &'static str {
        "gemma4"
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

    fn kv_dims(&self, hp: &Hparams, il: usize) -> (i64, i64) {
        if il < hp.n_layer {
            Gemma4Params::kv_dims(self, il)
        } else {
            (hp.head_dim, hp.n_head_kv)
        }
    }

    fn kv_source(&self, hp: &Hparams, il: usize) -> usize {
        if il < hp.n_layer {
            Gemma4Params::kv_source(self, il)
        } else {
            il
        }
    }

    fn compact_kv(&self, _hp: &Hparams, il: usize) -> i64 {
        self.compact.get(il).copied().unwrap_or(0)
    }

    fn sliding_layer(&self, hp: &Hparams, il: usize) -> bool {
        il < hp.n_layer && self.swa[il]
    }

    /// The per-layer token embeddings, read by the first stage.
    fn token_rows(&self, hp: &Hparams, layers: &Range<usize>) -> i64 {
        if layers.start == 0 {
            rows_width(self, hp)
        } else {
            0
        }
    }

    fn swa_window(&self) -> usize {
        self.window
    }

    fn fill_inputs(
        &self,
        m: &Model,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        _shape: &[crate::model::core::GroupShape],
        _rec: &crate::model::core::StatePool,
        _kv: &KvPool,
        _spec: Option<&crate::model::core::SpecPool>,
        _sink: &mut dyn crate::model::core::InputSink,
    ) {
        m.gemma4_fill_rows(inputs, groups);
    }

    fn embed<'a>(&self, m: &'a Model, ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
        m.gemma4_embed(ctx, x)
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let per_layer = match pass.ple {
            Some(rows) => {
                assert!(
                    layers == (0..m.hp.n_layer),
                    "Gemma 4 with per-layer embeddings runs as one pipeline stage"
                );
                Some(m.gemma4_per_layer_inputs(pass.ctx, x, rows)?)
            }
            None => None,
        };
        m.gemma4_layers(pass, x, per_layer, capture, layers)
    }

    fn capture_normed(&self) -> bool {
        true
    }

    // The MTP assistant (`assistant.rs`).

    fn mtp(&self) -> bool {
        true
    }

    fn default_spec_k(&self) -> Option<usize> {
        // The most drafts; the depth policy (`crate::spec::depth`) picks fewer where verify
        // rows cost (GB10, measured: 2-3 on the 26B-A4B MoE, 6-7 on the dense 31B, whose
        // verify step costs the same up to 8 rows).
        Some(8)
    }

    fn mtp_head(&self, hp: &Hparams, gguf: &GgufFile) -> Result<ArchRef, ModelError> {
        let a = assistant::parse(gguf, hp, self)?;
        Ok(ArchRef::new(Gemma4Params {
            assistant: Some(a),
            ..self.clone()
        }))
    }

    fn has_mtp(&self, m: &Model) -> bool {
        self.assistant.is_some()
            && m.mtp_weights()
                .is_some_and(|w| w.get("nextn.pre_projection.weight").is_some())
    }

    fn reads_kv_slots(&self) -> bool {
        true
    }

    fn mtp_shares_kv(&self) -> bool {
        true
    }

    /// The assistant's sliding-window layers read the target's last sliding-window cache, its
    /// global one the last global cache (`gemma4/assistant.rs`).
    fn mtp_kv_layers(&self, hp: &Hparams) -> Vec<usize> {
        match hp.n_layer {
            n if n >= 2 => vec![hp.kv_source(n - 2), hp.kv_source(n - 1)],
            _ => Vec::new(),
        }
    }

    fn mtp_sees_own_position(&self) -> bool {
        false
    }

    fn build_mtp<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        m.build_assistant(ctx, kv, rec, shape)
    }

    /// The soft-cap of [`Self::logits`] (0: none).
    fn logit_softcap(&self) -> Option<f32> {
        Some(self.softcap.max(0.0))
    }

    /// The final logit soft-cap, `c · tanh(logits / c)`, if the model has one.
    fn logits<'a>(&self, ctx: &'a Context, logits: Tensor<'a>) -> Tensor<'a> {
        match self.softcap {
            c if c > 0.0 => ctx.scale(ctx.tanh(ctx.scale(logits, 1.0 / c)), c),
            _ => logits,
        }
    }
}

/// The width of a Gemma 4 model's per-layer embedding rows, 0 without.
fn rows_width(g: &Gemma4Params, hp: &Hparams) -> i64 {
    if g.n_embd_per_layer > 0 {
        g.n_embd_per_layer * hp.n_layer as i64
    } else {
        0
    }
}

impl Model {
    fn gemma(&self) -> &Gemma4Params {
        params(&self.hp).expect("a Gemma 4 model")
    }

    /// `--gemma4-compact-global-kv` (#551): each global layer that owns its cache and has no
    /// value projection keeps a compact one ([`Arch::compact_kv`]). Its keys are RoPE over its
    /// normed key projection times `attn_k_norm.weight`, its values that projection normed
    /// alone: where RoPE leaves a dimension ([`rotated_dims`] by `rope_freqs.weight`), the
    /// key is the value times the weight, so the cache stores the rotated key dimensions only
    /// and attention puts the weight on the queries. Returns those layers (none for other
    /// models). Call before the KV pool is made.
    pub fn set_gemma4_compact_global_kv(&mut self) -> Vec<usize> {
        let Some(g) = params(&self.hp).cloned() else {
            return Vec::new();
        };
        let hp = &self.hp;
        let factors = self
            .w("rope_freqs.weight")
            .ok()
            .filter(|t| t.ty() == crate::ggml::types::F32)
            .map(|t| t.to_f32());
        let mut compact = vec![0; hp.n_layer];
        let mut rot: Vec<bool> = Vec::new();
        for (il, c) in compact.iter_mut().enumerate() {
            if g.swa[il]
                || !hp.owns_kv(il)
                || self.lw(il, "attn_v.weight").is_ok()
                || self.lw(il, "attn_k_norm.weight").is_err()
            {
                continue;
            }
            let (d, _) = hp.kv_dims(il);
            let dims = rotated_dims(d, hp.n_rot as i64, factors.as_deref());
            let r = dims.iter().filter(|&&x| x).count() as i64;
            // Whole quantization blocks for the granules' parts; one rotation for all.
            if r == 0 || r >= d || r % 32 != 0 || (!rot.is_empty() && rot != dims) {
                continue;
            }
            rot = dims;
            *c = r;
        }
        let layers: Vec<usize> = (0..hp.n_layer).filter(|&il| compact[il] > 0).collect();
        if !layers.is_empty() {
            self.hp.arch = ArchRef::new(Gemma4Params {
                compact,
                compact_rot: rot,
                ..g
            });
        }
        layers
    }

    /// The token embeddings scaled by `sqrt(n_embd)` (llama.cpp's `build_inp_embd` with
    /// Gemma's token scale).
    fn gemma4_embed<'a>(&self, ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
        ctx.scale(x, (self.hp.n_embd as f32).sqrt())
    }

    /// The per-layer inputs of every token (`project_per_layer_inputs`): `rows` are the
    /// tokens' rows of `per_layer_token_embd` (F32 `[n_embd_per_layer × n_layer, n]`), `x` the
    /// scaled embeddings. Returns `[n_embd_per_layer, n, n_layer]`.
    fn gemma4_per_layer_inputs<'a>(
        &'a self,
        ctx: &'a Context,
        x: Tensor<'a>,
        rows: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let hp = &self.hp;
        let npl = self.gemma().n_embd_per_layer;
        let n_layer = hp.n_layer as i64;
        let n = x.ne()[1];
        let rows = ctx.scale(rows, (npl as f32).sqrt());
        let rows = ctx.reshape_3d(rows, npl, n_layer, n);
        let proj = ctx.mul_mat(self.w("per_layer_model_proj.weight")?, x);
        let proj = ctx.scale(proj, 1.0 / (hp.n_embd as f32).sqrt());
        let proj = ctx.reshape_3d(proj, npl, n_layer, n);
        let proj = ctx.mul(
            ctx.rms_norm(proj, hp.rms_eps),
            self.w("per_layer_proj_norm.weight")?,
        );
        let inp = ctx.scale(ctx.add(proj, rows), 1.0 / 2f32.sqrt());
        Ok(ctx.cont(ctx.permute(inp, [0, 2, 1, 3])))
    }

    /// The layers `layers` of a Gemma 4 stack over the residual `x`; returns the last one's
    /// output (before the final norm) and the outputs of the layers in `capture`. `per_layer`:
    /// the per-layer inputs ([`Self::gemma4_per_layer_inputs`]), for models that have them.
    #[allow(clippy::type_complexity)]
    fn gemma4_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        mut x: Tensor<'a>,
        per_layer: Option<Tensor<'a>>,
        capture: &[usize],
        layers: std::ops::Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let eps = hp.rms_eps;
        let norm = |t: Tensor<'a>, il: usize, name: &str| -> Result<Tensor<'a>, ModelError> {
            Ok(ctx.mul(ctx.rms_norm(t, eps), self.lw(il, name)?))
        };
        // With `--head-rope-fused`: the positions as one view for every layer's RoPE, in the
        // graph ahead of the layers (else each RoPE makes its own, [`Model::gemma4_rope`]).
        let pos = HEAD_ROPE_FUSED.load(Ordering::Relaxed).then(|| {
            let pos = ctx.view_1d(pass.pos, n, 0);
            pass.graph.borrow_mut().expand(pos);
            pos
        });
        let mut hidden = Vec::new();
        for il in layers {
            let h = norm(x, il, "attn_norm.weight")?;
            tap.tap("attn_norm", il, h);
            let pre = (x, self.lw(il, "attn_norm.weight")?, eps);
            let o = self.gemma4_attention(pass, il, h, pre, pos)?;
            let o = norm(o, il, "post_attention_norm.weight")?;
            tap.tap("attn_post_norm", il, o);
            let attn_out = ctx.add(o, x);
            tap.tap("attn_out", il, attn_out);

            let ffn = if self.w(&format!("blk.{il}.ffn_gate_inp.weight")).is_ok() {
                let mlp = self.gemma4_ffn(ctx, il, attn_out, eps)?;
                let mlp = norm(mlp, il, "post_ffw_norm_1.weight")?;
                tap.tap("ffn_mlp", il, mlp);
                let moe_in = norm(attn_out, il, "pre_ffw_norm_2.weight")?;
                let moe = self.gemma4_moe(pass, il, attn_out, moe_in, n)?;
                let moe = norm(moe, il, "post_ffw_norm_2.weight")?;
                tap.tap("ffn_moe", il, moe);
                ctx.add(mlp, moe)
            } else {
                self.gemma4_ffn(ctx, il, attn_out, eps)?
            };
            let ffn = norm(ffn, il, "post_ffw_norm.weight")?;
            tap.tap("ffn_post_norm", il, ffn);
            let mut cur = ctx.add(ffn, attn_out);

            if let Some(pl) = per_layer {
                // Per-layer embedding: gate the residual, scale by this layer's input, project
                // back and add.
                let pe_in = cur;
                let t = ctx.gelu(self.lmm(ctx, il, "inp_gate.weight", cur)?);
                let nb = pl.nb();
                let this = ctx.view_2d(pl, [pl.ne()[0], pl.ne()[1]], nb[1], il * nb[2]);
                let t = ctx.mul(t, this);
                let t = self.lmm(ctx, il, "proj.weight", t)?;
                let t = norm(t, il, "post_norm.weight")?;
                tap.tap("per_layer_embd_out", il, t);
                cur = ctx.add(pe_in, t);
            }
            if let Ok(s) = self.lw(il, "layer_output_scale.weight") {
                cur = ctx.mul(cur, s);
            }
            tap.tap("l_out", il, cur);
            x = cur;
            if capture.contains(&il) {
                hidden.push((il, x));
            }
        }
        Ok((x, hidden))
    }

    /// RoPE of `t` (`[head_dim, heads, n]`) for layer `il`: the sliding-window layers' base,
    /// or the global layers' base and frequency factors. At positions `pos` (the layers' shared
    /// view, [`set_head_rope_fused`]) or, without them, a view of its own.
    fn gemma4_rope<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        t: Tensor<'a>,
        pos: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, hp, g) = (pass.ctx, &self.hp, self.gemma());
        // The positions are the first section of the (multi-section) position input.
        let pos = pos.unwrap_or_else(|| ctx.view_1d(pass.pos, pass.n, 0));
        Ok(if g.swa[il] {
            ctx.rope_ext(
                t,
                pos,
                None,
                g.n_rot_swa,
                ROPE_NEOX,
                hp.n_ctx_train,
                g.rope_base_swa,
                1.0,
            )
        } else {
            let factors = self.w("rope_freqs.weight").ok();
            ctx.rope_ext(
                t,
                pos,
                factors,
                hp.n_rot,
                ROPE_NEOX,
                hp.n_ctx_train,
                hp.rope_freq_base,
                1.0,
            )
        })
    }

    /// Attention of layer `il` over the normed residual `h`: queries always, keys and values
    /// into the cache unless the layer reads another layer's; then the output projection.
    /// `pos`: the layers' shared positions view, if any ([`Model::gemma4_rope`]).
    fn gemma4_attention<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
        pre: (Tensor<'a>, Tensor<'a>, f32),
        pos: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let (d, n_kv) = hp.kv_dims(il);
        let eps = hp.rms_eps;
        // `--nvfp4-a4` prompt steps: q, k and v from one quantization of `h` (`pre`: the
        // residual and attn_norm, `h` being their norm), normed in it.
        let mut names = vec!["attn_q.weight"];
        if hp.owns_kv(il) {
            names.push("attn_k.weight");
            if self.w(&format!("blk.{il}.attn_v.weight")).is_ok() {
                names.push("attn_v.weight");
            }
        }
        let multi: Vec<Option<Tensor<'a>>> = match self.lmm_multi(ctx, il, &names, h, Some(pre))? {
            Some(outs) => outs.into_iter().map(Some).collect(),
            None => vec![None; names.len()],
        };
        let mut multi = multi.into_iter();
        let mut proj = |name: &str| -> Result<Tensor<'a>, ModelError> {
            match multi.next().flatten() {
                Some(t) => Ok(t),
                None => self.lmm(ctx, il, name, h),
            }
        };
        let q = proj("attn_q.weight")?;
        let q = ctx.reshape_3d(q, d, hp.n_head, n);
        let q = ctx.mul(ctx.rms_norm(q, eps), self.lw(il, "attn_q_norm.weight")?);
        let q = self.gemma4_rope(pass, il, q, pos)?;
        tap.tap("Qcur_pos", il, q);
        if hp.owns_kv(il) {
            let k = proj("attn_k.weight")?;
            // Global layers without a value projection use the keys (before their norm).
            let v = match self.w(&format!("blk.{il}.attn_v.weight")) {
                Ok(_) => proj("attn_v.weight")?,
                Err(_) => k,
            };
            let k = ctx.reshape_3d(k, d, n_kv, n);
            let v = ctx.reshape_3d(v, d, n_kv, n);
            let k = ctx.mul(ctx.rms_norm(k, eps), self.lw(il, "attn_k_norm.weight")?);
            let v = ctx.rms_norm(v, eps);
            let k = self.gemma4_rope(pass, il, k, pos)?;
            tap.tap("Kcur_pos", il, k);
            tap.tap("Vcur_normed", il, v);
            let mut graph = pass.graph.borrow_mut();
            if pass.kv.compact(il) > 0 {
                // [K rotated | V not rotated | V rotated] per head (#551).
                let runs = dim_runs(&self.gemma().compact_rot);
                let rot = |x: Tensor<'a>| runs.iter().filter(|r| r.2).map(move |r| (x, r.0, r.1));
                let other = runs.iter().filter(|r| !r.2).map(|r| (v, r.0, r.1));
                let granules = pick(ctx, rot(k).chain(other).chain(rot(v)));
                pass.kv
                    .write_granules(ctx, &mut graph, il, granules, pass.write_cells(il));
            } else {
                pass.kv
                    .write(ctx, &mut graph, il, k, v, pass.write_cells(il));
            }
        }
        let o = self.gemma4_attend(pass, il, self.gemma().swa[il], hp.n_head, q)?;
        tap.tap("kqv_out", il, o);
        let o = self.lmm(ctx, il, "attn_output.weight", o)?;
        Ok(self.tp_sum(ctx, o))
    }

    /// Unscaled causal attention of every group's queries `q` (`[d, n_head, n]`) over the
    /// cache layer `il` reads, through the sliding-window mask when `swa`; `[d × n_head, n]`
    /// ([`Model::window_attend`] with scale 1).
    fn gemma4_attend<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        swa: bool,
        n_head: i64,
        q: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let compact = pass.kv.compact(il);
        if compact == 0 {
            return self.window_attend(pass, il, swa, n_head, q, 1.0);
        }
        let ctx = pass.ctx;
        let (d, _) = self.hp.kv_dims(il);
        let runs = dim_runs(&self.gemma().compact_rot);
        // A compact cache's keys (#551): the rotated dims, then the values where RoPE
        // leaves them, whose key weights go on the queries instead.
        let w = self.lw(pass.kv.source(il), "attn_k_norm.weight")?;
        let qw = ctx.mul(q, w);
        let rot = runs.iter().filter(|r| r.2).map(|r| (q, r.0, r.1));
        let other = runs.iter().filter(|r| !r.2).map(|r| (qw, r.0, r.1));
        let q = pick(ctx, rot.chain(other));
        let o = self.window_attend(pass, il, swa, n_head, q, 1.0)?;
        // A compact cache's values come not-rotated dims first: back in head order.
        let o = ctx.reshape_3d(o, d, n_head, pass.n);
        // Several outputs (groups, slots) joined, by heads only in a meta buffer's view of the
        // join: it takes the views below past a head's first dims for offsets across heads, and
        // scales them per device (ggml-backend-meta.cpp). Their own tensor, split by heads.
        let o = ctx.cont(o);
        let (mut n_rot, mut n_other) = (0, 0);
        let mut parts = Vec::new();
        for &(_, len, rot) in &runs {
            let at = if rot {
                n_rot += len;
                d - compact + n_rot - len
            } else {
                n_other += len;
                n_other - len
            };
            parts.push((o, at, len));
        }
        Ok(ctx.reshape_2d(pick(ctx, parts), d * n_head, pass.n))
    }

    /// The dense FFN (or the MoE layers' shared expert) over the residual `x`:
    /// `down(gelu(gate(h)) · up(h))` of `h = mul(rms_norm(x, eps), ffn_norm)`.
    fn gemma4_ffn<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        x: Tensor<'a>,
        eps: f32,
    ) -> Result<Tensor<'a>, ModelError> {
        let nw = self.lw(il, "ffn_norm.weight")?;
        let h = ctx.mul(ctx.rms_norm(x, eps), nw);
        // NVFP4 prompt steps: gate, up and the GELU as one op (the same bits; under
        // `--nvfp4-a4` the W4A4 one, with the norm in its activations' quantization).
        if let Some(down) = self.geglu_ffn_fused(ctx, il, h, Some((x, nw, eps)))? {
            return Ok(self.tp_sum(ctx, down));
        }
        let up = self.lmm(ctx, il, "ffn_up.weight", h)?;
        let gate = self.lmm(ctx, il, "ffn_gate.weight", h)?;
        let down = self.lmm(ctx, il, "ffn_down.weight", ctx.geglu_split(gate, up))?;
        Ok(self.tp_sum(ctx, down))
    }

    /// The routed experts of a MoE layer over `h` (the residual after `pre_ffw_norm_2`). The
    /// router reads the residual `attn_out` itself: RMS-normed, scaled by `1 / sqrt(n_embd)`
    /// and by `ffn_gate_inp.scale`. Softmax over the experts, top-k, weights renormalized.
    fn gemma4_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        attn_out: Tensor<'a>,
        h: Tensor<'a>,
        n: i64,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let hp = &self.hp;
        let used = hp.n_expert_used;
        let r = ctx.scale(
            ctx.rms_norm(attn_out, hp.rms_eps),
            1.0 / (hp.n_embd as f32).sqrt(),
        );
        let r = ctx.mul(r, self.lw(il, "ffn_gate_inp.scale")?);
        let gate_inp = format!("blk.{il}.ffn_gate_inp.weight");
        let logits = self
            .linear
            .mul_mat(ctx, &gate_inp, self.w(&gate_inp)?, None, None, r)?;
        tap.tap("ffn_moe_logits", il, logits);
        let probs = ctx.soft_max(logits);
        let selected = ctx.argsort_top_k(probs, used as i32);
        tap.tap("ffn_moe_topk", il, selected);
        let w = ctx.get_rows(ctx.reshape_3d(probs, 1, hp.n_expert, n), selected);
        let w = ctx.reshape_2d(w, used, n);
        let w = ctx.div(w, ctx.clamp(ctx.sum_rows(w), 6.103_515_6e-5, f32::INFINITY));
        let w = ctx.reshape_3d(w, 1, used, n);
        tap.tap("ffn_moe_weights_norm", il, w);
        // Into the graph now, in llama.cpp's order (as `Model::moe` does): ggml-cuda then runs
        // the routing as one fused top-k kernel, and fuses the down projection's per-expert
        // scale, these weights and the sum over the experts into one kernel. Left to the
        // experts' expansion, the weights' nodes land between that scale and the weighting,
        // and the scale costs a pass over [n_embd, used, n] per layer: 0.27 s of a 13K-token
        // prompt on the 26B-A4B (GB10).
        // With `--stable-fusions` the top-k kernel whatever the layout (#472).
        stable_fusions::pin_router(&pass.graph, logits, w);

        let x = ctx.reshape_3d(h, hp.n_embd, 1, n);
        let (gate, up) = if self.w(&format!("blk.{il}.ffn_gate_up_exps.weight")).is_ok() {
            // One matmul for both, gate rows first.
            let gu = self.lmm_id(ctx, il, "ffn_gate_up_exps.weight", x, selected)?;
            let n_ff = gu.ne()[0] / 2;
            let nb = gu.nb();
            let gate = ctx.view_3d(gu, [n_ff, gu.ne()[1], gu.ne()[2]], [nb[1], nb[2]], 0);
            let up = ctx.view_3d(
                gu,
                [n_ff, gu.ne()[1], gu.ne()[2]],
                [nb[1], nb[2]],
                n_ff as usize * nb[0],
            );
            (gate, up)
        } else {
            let up = self.lmm_id(ctx, il, "ffn_up_exps.weight", x, selected)?;
            let gate = self.lmm_id(ctx, il, "ffn_gate_exps.weight", x, selected)?;
            (gate, up)
        };
        let act = ctx.geglu_split(gate, up);
        let experts = self.lmm_id(ctx, il, "ffn_down_exps.weight", act, selected)?;
        let weighted = ctx.mul(experts, w); // [n_embd, used, n]
        let nb = weighted.nb();
        let mut out = ctx.view_2d(weighted, [hp.n_embd, n], nb[2], 0);
        for e in 1..used as usize {
            out = ctx.add(out, ctx.view_2d(weighted, [hp.n_embd, n], nb[2], e * nb[1]));
        }
        // ggml-cuda's one-rounding fusion in every layer (`--stable-fusions`, #472).
        stable_fusions::pin_weighted_sum(&pass.graph, weighted, out);
        if used == 1 {
            out = ctx.cont(out);
        }
        tap.tap("ffn_moe_out", il, out);
        Ok(out)
    }

    /// Fills the per-layer token embedding rows of a step's tokens (models that have them):
    /// dequantized straight from the table in host memory, exactly as `get_rows` would.
    fn gemma4_fill_rows(&self, inputs: &Inputs<'_>, groups: &[Group<'_>]) {
        let Some(t) = inputs.ple else { return };
        let table = self
            .weights
            .get("per_layer_token_embd.weight")
            .expect("per_layer_token_embd.weight");
        let bytes = table
            .host_bytes()
            .expect("per_layer_token_embd.weight must be in host memory");
        let ty = table.ty();
        let dim = table.ne()[0] as usize;
        let row_bytes = crate::ggml::row_size(ty, dim);
        let n_rows = table.ne()[1] as usize;
        let mut out = Vec::new();
        for s in groups.iter().flat_map(|g| &g.seqs) {
            for &tok in s.tokens {
                let r = tok as usize;
                assert!(r < n_rows, "token {r} outside the {n_rows}-row table");
                let src = &bytes[r * row_bytes..][..row_bytes];
                out.extend(crate::ggml::dequantize(ty, src, dim));
            }
        }
        t.set_f32(&out);
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod compact_tests;
