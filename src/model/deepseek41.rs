//! `deepseek41` (DeepSeek-V4.1-Flash), following DeepSeek's reference `inference/model.py` and
//! `engram.py` (MIT) op for op:
//!
//! - **Hyper-connections**: the residual is `hc` parallel streams `[n_embd, hc, n]`. Each
//!   sublayer's mixer ([`glm_ops::hc_pre`], GLM-5.3-Flash's: the same Sinkhorn split) gives the
//!   weights of its output in each new stream and the stream-to-stream mix ([`glm_ops::hc_post`]);
//!   its collapse weights are the *next* sublayer's input mix ([`ops::collapse`]): attention
//!   reads the previous layer's FFN mix (stream 0 before the first layer), the FFN the
//!   attention's. The head reads the last FFN's collapse, normed.
//! - **Attention**: one 512-wide latent per position (keys and values both), low-rank queries
//!   (`attn_q_a` → norm → `attn_q_b`), RoPE on the last `n_rot` dimensions (YaRN with
//!   `compress_rope_freq_base` on the compressing layers, plain `rope.freq_base` on the others),
//!   a per-head sink logit, the output rotated back, then a grouped low-rank output
//!   (`attn_output_a` per group, `attn_output_b`). Each query sees the `sliding_window` latest
//!   positions, and on a layer with `compress_ratio` r > 0 also up to `top_k` compressed
//!   latents (one per group of r positions) that the lightning indexer picks.
//! - **Shared compression**: only `kv_source_layer_ids` compress (softmax-gated pooling for
//!   r > 1) and only they derive index keys; only `index_source_layer_ids` run an indexer. The
//!   other compressing layers read the latest source's cache and selection. The candidate
//!   source layer's scores pick blocks of compressed positions that the later indexers stay in
//!   (`select_candidate_blocks`).
//! - **MoE**: `sqrt(softplus)` router with a selection bias, the chosen weights normalized and
//!   scaled, SwiGLU clamped in the experts and in the one shared expert.
//! - **Engram**: before its layer, an n-gram hash lookup per token (the hashing and the table
//!   rows' dequantization on the host, [`Arch::fill_inputs`]) projected by `engram_wkv` into a
//!   key per stream and one value, added to each stream through a gate ([`ops::engram`]). The
//!   tables (`blk.{l}.engram_embd.weight`) are in the model's GGUF or in the one
//!   `engram.file` names next to it, which [`GgufFile::open`] attaches as one more file. The
//!   engine reads them through a host lookup table ([`engram`]), not as weights.
//!
//! The KV cache: per layer one F16 row per position (see [`ops`]): K holds the window latent and,
//! on a kv source, the compressed latent and index key of the group ending there; V a ratio-2
//! source's raw rows and gates. Every step is paged (read in place through the block tables).
//!
//! Every computation is a Legion op: [`ops`] and GLM-5.3-Flash's [`glm_ops`] (CPU references,
//! CUDA kernels) and [`moe_ffn`] for the routed experts. Not yet: the DSpark / MTP head,
//! pipeline stages, the vision tower, and running the safetensors checkpoint as shipped
//! ([`crate::hf`] loads it, `legion gguf check` checks it): its dense FP8 weights' 32 × 32 block
//! scales have no matmul yet ([`block_fp8`], refused by the parse) and its MXFP4 experts no
//! expert op.

pub mod check;
pub mod engram;
pub mod ops;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod tiny;

use std::ops::Range;

use crate::ggml::{Context, Tensor, types};
use crate::gguf::{GgufFile, TensorInfo};
use crate::model::core::arch::same_as;
use crate::model::core::state::{KvPool, NO_TOKEN};
use crate::model::core::{
    Arch, ArchRef, AttnInputs, Group, GroupShape, Hparams, InputSink, Inputs, Model, ModelError,
    Pass, SpecPool, StatePool, TokenConv, cols,
};
use crate::model::glm5next::ops as glm_ops;
use crate::model::{fp8, moe_ffn};

use ops::{Compressed, Rope};

/// The GGUF architecture name.
pub const ARCH: &str = "deepseek41";

/// Compressed positions a step's scores cover are padded to a multiple of this many positions
/// (the score tensors' width; graphs are reused within it).
const SCORE_PAD: usize = 256;

/// Engram: the hashed n-gram lookups of a few layers.
#[derive(Debug, Clone, PartialEq)]
pub struct EngramParams {
    /// The layers with an Engram, in the order of the hash tables' rows.
    pub layers: Vec<usize>,
    /// The longest n-gram: each token is hashed with its `ngram - 1` predecessors.
    pub ngram: usize,
    /// Hash heads per n-gram size.
    pub heads: usize,
    /// Width of a table row.
    pub dim: i64,
    /// The token that stands for positions before the sequence start.
    pub pad: u32,
    /// Per layer its table's type and rows (`blk.{l}.engram_embd.weight`).
    pub tables: Vec<(crate::ggml::ffi::ggml_type, u64)>,
}

impl EngramParams {
    /// Table rows per token and layer: one per (n-gram size ≥ 2, head).
    pub fn cols(&self) -> usize {
        (self.ngram - 1) * self.heads
    }

    /// Width of one layer's rows per token.
    pub fn width(&self) -> i64 {
        self.cols() as i64 * self.dim
    }
}

/// The DeepSeek-V4.1-Flash hyperparameters beyond [`Hparams`].
#[derive(Debug, Clone, PartialEq)]
pub struct Ds41Params {
    /// Per layer: the compression ratio, 0 for a window-only layer.
    pub ratios: Vec<usize>,
    /// Per layer: whether it compresses (and derives index keys).
    pub kv_source: Vec<bool>,
    /// Per layer: whether it runs an indexer.
    pub index_source: Vec<bool>,
    /// The layer whose scores pick the candidate blocks, the block size (compressed
    /// positions) and the blocks kept.
    pub candidates: Option<(usize, usize, usize)>,
    /// Latent width (keys, values and queries per head).
    pub head: i64,
    pub n_rot: i64,
    pub q_lora: i64,
    pub o_groups: i64,
    pub o_lora: i64,
    pub idx_heads: i64,
    pub idx_dim: i64,
    pub top_k: usize,
    pub window: usize,
    /// RoPE of the window-only layers and of the compressing ones.
    pub rope: Rope,
    pub rope_c: Rope,
    pub hc: i64,
    pub hc_iters: u32,
    pub hc_eps: f32,
    pub weights_norm: bool,
    pub weights_scale: f32,
    /// Per layer: the SwiGLU limit of the routed experts and of the shared expert.
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    pub engram: Option<EngramParams>,
}

/// Whether the dense weight `w` is FP8 (`I8`) with a grid of block scales `scale`
/// (`<w>.scale`, F32), not one scale or one per row.
pub(crate) fn block_scaled(w: &TensorInfo, scale: Option<&TensorInfo>) -> bool {
    w.ggml_type == types::I8
        && scale.is_some_and(|s| {
            s.ggml_type == types::F32
                && s.elements() != 1
                && w.shape.get(1).is_none_or(|&m| s.shape != [m])
        })
}

/// The DeepSeek-V4.1-Flash hyperparameters of `hp`, if it is one.
pub fn params(hp: &Hparams) -> Option<&Ds41Params> {
    hp.arch.as_any().downcast_ref::<Ds41Params>()
}

/// The numbers of the array `key` (or its one value), as many as the array has.
fn numbers(g: &GgufFile, key: &str) -> Option<Vec<f64>> {
    let v = g.get(key)?;
    match v.as_array() {
        Some(a) => a.iter().map(|x| x.as_f64()).collect(),
        None => v.as_f64().map(|x| vec![x]),
    }
}

/// A per-layer key: an array of at least `n` numbers, or one for every layer.
fn per_layer(g: &GgufFile, key: &str, n: usize) -> Result<Vec<f64>, ModelError> {
    let v = numbers(g, key).ok_or_else(|| ModelError::Metadata(key.into()))?;
    match v.len() {
        1 => Ok(vec![v[0]; n]),
        l if l >= n => Ok(v[..n].to_vec()),
        _ => Err(ModelError::Metadata(key.into())),
    }
}

pub(crate) fn parse(g: &GgufFile) -> Result<Hparams, ModelError> {
    let hp = hparams(g)?;
    if let Some(w) = block_fp8(g) {
        return Err(ModelError::Unsupported(format!(
            "`{w}`: FP8 with 32 x 32 block scales (the safetensors checkpoint as shipped) has no \
             matmul in Legion yet; run a GGUF of the model (`legion gguf check` lists what else \
             does not run)"
        )));
    }
    Ok(hp)
}

/// The first dense FP8 weight of `g` with 32 × 32 block scales ([`block_scaled`]): the
/// safetensors checkpoint as shipped, which the graph has no matmul for yet.
pub(crate) fn block_fp8(g: &GgufFile) -> Option<String> {
    g.tensors()
        .into_iter()
        .filter(|t| t.ggml_type == types::I8)
        .find(|t| {
            let scale = t
                .name
                .strip_suffix(".weight")
                .and_then(|b| g.tensor(&format!("{b}.scale")));
            block_scaled(t, scale.as_ref())
        })
        .map(|t| t.name)
}

/// [`parse`] without refusing the weights no kernel takes yet ([`block_fp8`]): for
/// [`check`], which lists them.
pub(crate) fn hparams(g: &GgufFile) -> Result<Hparams, ModelError> {
    let key = |k: &str| format!("{ARCH}.{k}");
    // The first of `names` the GGUF has (converters spell some keys differently).
    let find = |names: &[&str]| names.iter().map(|k| key(k)).find(|k| g.get(k).is_some());
    let u = |k: &str| -> Result<u64, ModelError> {
        g.get_u64(&key(k))
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let f = |k: &str| -> Result<f32, ModelError> {
        g.get(&key(k))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .ok_or_else(|| ModelError::Metadata(key(k)))
    };
    let flag = |k: &str, default: bool| g.get(&key(k)).and_then(|v| v.as_bool()).unwrap_or(default);
    let layer_list = |names: &[&str]| -> Result<Vec<usize>, ModelError> {
        let k = find(names).ok_or_else(|| ModelError::Metadata(key(names[0])))?;
        let v = numbers(g, &k).ok_or_else(|| ModelError::Metadata(k.clone()))?;
        Ok(v.iter().map(|&x| x as usize).collect())
    };
    let unsupported = |what: String| Err(ModelError::Unsupported(what));
    let tensor = |name: &str| {
        g.tensor(name)
            .ok_or_else(|| ModelError::Tensor(name.into()))
    };

    let n_layer = u("block_count")? as usize;
    if g.tensors()
        .iter()
        .any(|t| t.name.starts_with("v.") || t.name.starts_with("mm."))
    {
        return unsupported("DeepSeek-V4.1-Flash's vision tower".into());
    }
    if u("expert_gating_func").ok() != Some(4)
        && g.get_str(&key("expert_scoring_func")).as_deref() != Some("sqrtsoftplus")
    {
        return unsupported("a DeepSeek-V4.1 router other than sqrt(softplus)".into());
    }
    if g.get_u64(&key("hash_layer_count")).unwrap_or(0) != 0 {
        return unsupported("DeepSeek-V4.1's hash-routed MoE layers".into());
    }
    if g.get_u64(&key("leading_dense_block_count")).unwrap_or(0) != 0 {
        return unsupported("DeepSeek-V4.1 dense FFN layers".into());
    }
    if u("attention.head_count_kv")? != 1 {
        return unsupported("DeepSeek-V4.1 attention with more than one latent".into());
    }
    let head = u("attention.key_length")? as i64;
    if g.get_u64(&key("attention.value_length"))
        .unwrap_or(head as u64)
        != head as u64
    {
        return unsupported("DeepSeek-V4.1 values of another width than the keys".into());
    }

    let ratios: Vec<usize> = per_layer(g, &key("attention.compress_ratios"), n_layer)?
        .iter()
        .map(|&r| r as usize)
        .collect();
    let mark = |list: &[usize]| -> Result<Vec<bool>, ModelError> {
        let mut v = vec![false; n_layer];
        for &il in list {
            if il >= n_layer {
                return Err(ModelError::Metadata(format!("source layer {il}")));
            }
            v[il] = true;
        }
        Ok(v)
    };
    let kv_source = mark(&layer_list(&[
        "kv_source_layer_ids",
        "attention.kv_source_layers",
    ])?)?;
    let index_source = mark(&layer_list(&[
        "index_source_layer_ids",
        "attention.indexer.source_layers",
    ])?)?;
    let cand_layer = find(&[
        "attention.candidate_source_layer_id",
        "attention.indexer.candidate_source_layer",
    ])
    .and_then(|k| g.get(&k))
    .and_then(|v| v.as_f64())
    .map(|v| v as usize);
    let candidates = match cand_layer {
        Some(il) => {
            let num = |names: &[&str]| -> Result<usize, ModelError> {
                let k = find(names).ok_or_else(|| ModelError::Metadata(key(names[0])))?;
                g.get_u64(&k)
                    .map(|v| v as usize)
                    .ok_or(ModelError::Metadata(k))
            };
            let bs = num(&[
                "attention.candidate_block_size",
                "attention.indexer.candidate_block_size",
            ])?;
            let tb = num(&[
                "attention.candidate_topk_blocks",
                "attention.indexer.candidate_top_k_blocks",
            ])?;
            Some((il, bs, tb))
        }
        None => None,
    };

    let n_rot = u("rope.dimension_count")? as i64;
    let base = f("rope.freq_base")?;
    let cbase = f("attention.compress_rope_freq_base").unwrap_or(base);
    let rope_c = if g.get_str(&key("rope.scaling.type")).as_deref() == Some("yarn") {
        Rope::yarn(
            n_rot,
            cbase,
            f("rope.scaling.factor")?,
            u("rope.scaling.original_context_length")? as f64,
            f("rope.scaling.yarn_beta_fast").unwrap_or(32.0) as f64,
            f("rope.scaling.yarn_beta_slow").unwrap_or(1.0) as f64,
        )
    } else {
        Rope::plain(n_rot, cbase)
    };

    let engram = if flag("engram.present", false) {
        let layers = layer_list(&["engram.layer_ids"])?;
        let mut e = EngramParams {
            layers: layers.clone(),
            ngram: u("engram_max_ngram_size")? as usize,
            heads: u("engram_n_heads")? as usize,
            dim: u("engram_head_dim")? as i64,
            pad: u("engram_pad_token_id")? as u32,
            tables: Vec::new(),
        };
        if e.ngram < 2 || e.heads == 0 || layers.is_empty() || layers.iter().any(|&l| l >= n_layer)
        {
            return unsupported(format!("Engram of {e:?}"));
        }
        let nl = layers.len() as u64;
        let (cols, ngram) = (e.cols() as u64, e.ngram as u64);
        for (name, ty, shape) in [
            ("engram_token_map.weight", types::I32, None),
            ("engram_primes.weight", types::I64, Some(vec![cols, nl])),
            ("engram_offsets.weight", types::I64, Some(vec![cols, nl])),
            (
                "engram_multipliers.weight",
                types::I64,
                Some(vec![ngram, nl]),
            ),
        ] {
            let t = tensor(name)?;
            if t.ggml_type != ty || shape.is_some_and(|s| t.shape != s) {
                return unsupported(format!("`{name}` of {:?} {:?}", t.ggml_type, t.shape));
            }
        }
        // The tables' rows per layer: the sum of its primes, which converters also write out.
        let rows = numbers(g, &key("engram.num_embeddings"));
        if rows.as_ref().is_some_and(|r| r.len() != layers.len()) {
            return unsupported(format!(
                "`{}` of {} entries for {} Engram layers",
                key("engram.num_embeddings"),
                rows.as_ref().map_or(0, Vec::len),
                layers.len()
            ));
        }
        for (li, &l) in layers.iter().enumerate() {
            let name = format!("blk.{l}.engram_embd.weight");
            // In this file or in the companion `engram.file` names ([`crate::gguf`] attaches it).
            let Some(t) = g.tensor(&name) else {
                let path = crate::gguf::companion_path(g, "engram.file")
                    .map_err(|e| ModelError::Unsupported(e.to_string()))?;
                return Err(match path {
                    Some(p) => ModelError::Unsupported(format!(
                        "`{name}` is not in the model, and its Engram tables file `{}` was not \
                         found next to it",
                        p.display()
                    )),
                    None => ModelError::Tensor(name),
                });
            };
            let want = rows.as_ref().map(|r| r[li] as u64);
            if t.shape.len() != 2
                || t.shape[0] != e.dim as u64
                || want.is_some_and(|n| t.shape[1] != n)
            {
                return unsupported(format!(
                    "`{name}` of shape {:?} (expected [{}, {}])",
                    t.shape,
                    e.dim,
                    want.map_or("rows".into(), |n| n.to_string())
                ));
            }
            // FP8 rows (the safetensors checkpoint's) have an E8M0 scale byte per 32.
            if t.ggml_type == types::I8 {
                let sname = format!("blk.{l}.engram_embd.scale");
                let want = [e.dim as u64 / 32, t.shape[1]];
                match g.tensor(&sname) {
                    Some(s) if e.dim % 32 == 0 && s.ggml_type == types::I8 && s.shape == want => {}
                    s => {
                        return unsupported(format!(
                            "`{name}`: FP8 rows need `{sname}` (I8 {want:?}), not {:?}",
                            s.map(|s| (s.type_name(), s.shape))
                        ));
                    }
                }
            }
            e.tables.push((t.ggml_type, t.shape[1]));
        }
        Some(e)
    } else {
        None
    };

    let n_embd = u("embedding_length")? as i64;
    let rms_eps = f("attention.layer_norm_rms_epsilon")?;
    let params = Ds41Params {
        ratios,
        kv_source,
        index_source,
        candidates,
        head,
        n_rot,
        q_lora: u("attention.q_lora_rank")? as i64,
        o_groups: u("attention.output_group_count")? as i64,
        o_lora: u("attention.output_lora_rank")? as i64,
        idx_heads: u("attention.indexer.head_count")? as i64,
        idx_dim: u("attention.indexer.key_length")? as i64,
        top_k: u("attention.indexer.top_k")? as usize,
        window: u("attention.sliding_window")? as usize,
        rope: Rope::plain(n_rot, base),
        rope_c,
        hc: u("hyper_connection.count")? as i64,
        hc_iters: u("hyper_connection.sinkhorn_iterations")? as u32,
        hc_eps: f("hyper_connection.epsilon")?,
        weights_norm: flag("expert_weights_norm", false),
        weights_scale: f("expert_weights_scale").unwrap_or(1.0),
        clamp_exp: per_layer(g, &key("swiglu_clamp_exp"), n_layer)?
            .iter()
            .map(|&x| x as f32)
            .collect(),
        clamp_shexp: per_layer(g, &key("swiglu_clamp_shexp"), n_layer)?
            .iter()
            .map(|&x| x as f32)
            .collect(),
        engram,
    };
    params.check(n_layer)?;
    let n_head = u("attention.head_count")? as i64;
    Ok(Hparams {
        n_layer,
        n_embd,
        n_head,
        // One latent row per position, keys and values both.
        n_head_kv: 1,
        head_dim: head,
        n_rot: n_rot as i32,
        rope_sections: [0; 4],
        rope_freq_base: base,
        n_ctx_train: u("context_length")? as i32,
        rms_eps,
        moe: true,
        n_expert: u("expert_count")? as i64,
        n_expert_used: u("expert_used_count")? as i64,
        // No recurrent layers.
        full_attention_interval: 1,
        ssm_d_conv: 0,
        ssm_d_state: 0,
        ssm_n_group: 0,
        ssm_dt_rank: 1,
        ssm_d_inner: 0,
        n_vocab: tensor("token_embd.weight")?.shape[1] as i64,
        arch: ArchRef::new(params),
    })
}

impl Ds41Params {
    /// The latest kv source at or before layer `il`.
    pub fn kv_of(&self, il: usize) -> Option<usize> {
        (0..=il).rev().find(|&l| self.kv_source[l])
    }

    /// The latest index source at or before layer `il`.
    pub fn index_of(&self, il: usize) -> Option<usize> {
        (0..=il).rev().find(|&l| self.index_source[l])
    }

    /// Whether layer `il`'s indexer adds the candidate blocks' mask.
    fn masked(&self, il: usize) -> bool {
        self.candidates.is_some_and(|(c, _, _)| il > c)
    }

    /// Layer `il`'s cache row width: the window latent, and on a kv source the compressed
    /// latent and the index key.
    fn row(&self, il: usize) -> i64 {
        if self.kv_source[il] {
            2 * self.head + self.idx_dim
        } else {
            self.head
        }
    }

    /// The sharing structure the graph relies on.
    fn check(&self, n_layer: usize) -> Result<(), ModelError> {
        let bad = |what: String| Err(ModelError::Unsupported(what));
        if self.head % 2 != 0 || self.n_rot % 2 != 0 || self.n_rot > self.head.min(self.idx_dim) {
            return bad(format!(
                "RoPE over {} of {} dimensions",
                self.n_rot, self.head
            ));
        }
        if self.window == 0 || self.top_k == 0 || self.hc < 1 {
            return bad("an empty window, selection or residual".into());
        }
        if self.o_groups < 1 {
            return bad(format!("{} output groups", self.o_groups));
        }
        for il in 0..n_layer {
            let r = self.ratios[il];
            if r == 0 {
                if self.kv_source[il] || self.index_source[il] {
                    return bad(format!("layer {il}: a source without compression"));
                }
                continue;
            }
            if !SCORE_PAD.is_multiple_of(r) {
                return bad(format!("layer {il}: compression ratio {r}"));
            }
            let (Some(kv), Some(ix)) = (self.kv_of(il), self.index_of(il)) else {
                return bad(format!("layer {il}: compressed without an earlier source"));
            };
            if self.ratios[kv] != r || self.ratios[ix] != r || self.kv_of(ix) != Some(kv) {
                return bad(format!(
                    "layer {il}: sources {kv} / {ix} of another ratio or cache"
                ));
            }
            if self.masked(il) && self.index_source[il] {
                let (c, _, _) = self.candidates.unwrap();
                if self.ratios[c] != r {
                    return bad(format!("layer {il}: candidates of another ratio"));
                }
            }
        }
        if let Some((c, bs, tb)) = self.candidates
            && (c >= n_layer || !self.index_source[c] || bs == 0 || tb == 0)
        {
            return bad(format!("candidate source layer {c} ({bs} x {tb})"));
        }
        Ok(())
    }
}

impl Arch for Ds41Params {
    fn name(&self) -> &'static str {
        ARCH
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, _hp: &Hparams, _il: usize) -> bool {
        false
    }

    fn kv_dims(&self, _hp: &Hparams, il: usize) -> (i64, i64) {
        (self.row(il), 1)
    }

    /// The latents are keys and values at once (K): only a kv source pooling groups of more
    /// than one position keeps V, the raw rows and gates its groups pool.
    fn kv_has_v(&self, _hp: &Hparams, il: usize) -> bool {
        self.kv_source[il] && self.ratios[il] > 1
    }

    /// A layer that is not a kv source reads its own cache's window only (its compressed keys
    /// are its source's): its cache can be a ring.
    fn sliding_layer(&self, _hp: &Hparams, il: usize) -> bool {
        !self.kv_source[il]
    }

    fn swa_window(&self) -> usize {
        self.window
    }

    fn residual_streams(&self) -> i64 {
        self.hc
    }

    fn token_rows(&self, _hp: &Hparams, layers: &Range<usize>) -> i64 {
        match &self.engram {
            Some(e) if e.layers.iter().any(|l| layers.contains(l)) => {
                e.layers.len() as i64 * e.width()
            }
            _ => 0,
        }
    }

    /// The token history of the Engram hashes (no conv: a one-value state).
    fn token_conv(&self, _hp: &Hparams) -> Option<TokenConv> {
        self.engram.as_ref().map(|e| TokenConv {
            layer: e.layers[0],
            history: 1,
            width: 1,
            n_hist: e.ngram - 1,
        })
    }

    /// The score tensors' width: every sequence's positions, padded.
    fn group_pools(&self, g: &Group<'_>) -> (usize, usize) {
        let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
        let end = g.seqs.iter().map(|s| s.n_past + n_tok).max().unwrap_or(0);
        (end.div_ceil(SCORE_PAD).max(1) * SCORE_PAD, 0)
    }

    fn fill_inputs(
        &self,
        m: &Model,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        _shape: &[GroupShape],
        rec: &StatePool,
        _kv: &KvPool,
        spec: Option<&SpecPool>,
        sink: &mut dyn InputSink,
    ) {
        let (Some(e), Some(t)) = (&self.engram, inputs.ple) else {
            return;
        };
        let history = |g: &Group<'_>, slot: usize, tokens: &[u32]| {
            if g.verify {
                let spec = spec.expect("a verify group needs the replay pool");
                spec.record_fed(slot, tokens);
                rec.peek_tokens(slot)
            } else {
                rec.push_tokens(slot, tokens)
            }
        };
        // Through the Engram lookup table when the engine opened one ([`engram`]), else from
        // the weights' host memory.
        let rows = match &m.ngram {
            Some(table) => engram::rows_from_table(m, e, table, groups, history),
            None => engram_rows(m, e, groups, history),
        };
        sink.f32(t, &rows);
    }

    /// The ops read the cache in place through the block tables.
    fn paged(&self, _m: &Model, _kv: &KvPool) -> bool {
        true
    }

    fn always_paged(&self) -> bool {
        true
    }

    fn paged_for(&self, _m: &Model, _kv: &KvPool, _groups: &[Group<'_>]) -> bool {
        true
    }

    fn embed_tokens<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        tokens: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        Ok(Some(glm_ops::embed(
            ctx,
            m.w("token_embd.weight")?,
            tokens,
            self.hc,
        )))
    }

    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        if !capture.is_empty() {
            return Err(ModelError::Unsupported(format!(
                "hidden-state captures {capture:?} of DeepSeek-V4.1-Flash"
            )));
        }
        Ok((m.ds41_layers(pass, x, layers)?, Vec::new()))
    }

    /// The last layer ends in the collapsed streams, normed, at the output rows.
    fn output_ready(&self) -> bool {
        true
    }

    fn head<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        x: Tensor<'a>,
    ) -> Result<Option<(Tensor<'a>, Tensor<'a>)>, ModelError> {
        let logits = m.glm_mm(ctx, "output.weight", x)?;
        Ok(Some((logits, glm_ops::argmax(ctx, logits))))
    }

    fn external_drafters(&self) -> bool {
        false
    }

    /// The head's [`glm_ops::argmax`] takes the first of the largest (#695).
    fn argmax_first_of_ties(&self) -> bool {
        true
    }
}

/// The Engram rows of `groups`' tokens (F32 `[layers × cols × dim]` per token, in order): each
/// sequence's tokens hashed with the predecessors `history(group, slot, tokens)` returns for its
/// slot (oldest first, [`NO_TOKEN`] before the start), the rows dequantized from the tables'
/// host memory.
fn engram_rows(
    m: &Model,
    e: &EngramParams,
    groups: &[Group<'_>],
    mut history: impl FnMut(&Group<'_>, usize, &[u32]) -> Vec<u32>,
) -> Vec<f32> {
    let host = |name: &str| host_bytes(m, name);
    let inputs = HashInputs::read(m);
    let hash = inputs.hash(e);
    // Each table, its bytes and (FP8 rows) its scale bytes.
    let tables: Vec<(Tensor<'_>, &[u8], &[u8])> = e
        .layers
        .iter()
        .map(|l| {
            let name = format!("blk.{l}.engram_embd.weight");
            let t = m.w(&name).expect("an Engram table");
            let scale = match t.ty() {
                types::I8 => host(&format!("blk.{l}.engram_embd.scale")),
                _ => &[],
            };
            (t, host(&name), scale)
        })
        .collect();
    let (dim, cols) = (e.dim as usize, e.cols());
    let mut out = Vec::new();
    let mut ids = Vec::new();
    for g in groups {
        for (i, s) in g.seqs.iter().enumerate() {
            let mut prev = history(g, g.slot + i, s.tokens);
            for &tok in s.tokens {
                for (li, (t, bytes, scale)) in tables.iter().enumerate() {
                    hash.ids(li, tok, &prev, &mut ids);
                    let row_bytes = crate::ggml::row_size(t.ty(), dim);
                    let n_rows = t.ne()[1];
                    for &id in &ids {
                        assert!(
                            (0..n_rows).contains(&id),
                            "Engram row {id} outside the {n_rows}-row table"
                        );
                        let src = &bytes[id as usize * row_bytes..][..row_bytes];
                        if t.ty() == types::I8 {
                            out.extend(fp8_row_f32(src, &scale[id as usize * dim / 32..]));
                        } else {
                            out.extend(row_f32(t.ty(), src, dim));
                        }
                    }
                }
                prev.remove(0);
                prev.push(tok);
            }
        }
    }
    debug_assert_eq!(out.len() % (tables.len() * cols * dim).max(1), 0);
    out
}

/// The bytes of the host weight `name`.
fn host_bytes<'m>(m: &'m Model, name: &str) -> &'m [u8] {
    m.w(name)
        .ok()
        .and_then(|t| t.host_bytes())
        .unwrap_or_else(|| panic!("`{name}` must be in host memory"))
}

/// The Engram hash parameters as the host reads them: the compressed token map, per layer the
/// columns' primes and offsets and the n-gram multipliers.
pub(crate) struct HashInputs {
    map: Vec<i64>,
    primes: Vec<i64>,
    offsets: Vec<i64>,
    mult: Vec<i64>,
}

impl HashInputs {
    fn read(m: &Model) -> Self {
        let i64s = |name: &str| -> Vec<i64> {
            host_bytes(m, name)
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_ne_bytes(*b))
                .collect()
        };
        Self {
            map: host_bytes(m, "engram_token_map.weight")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_ne_bytes(*b) as i64)
                .collect(),
            primes: i64s("engram_primes.weight"),
            offsets: i64s("engram_offsets.weight"),
            mult: i64s("engram_multipliers.weight"),
        }
    }

    fn hash<'a>(&'a self, e: &'a EngramParams) -> Hash<'a> {
        Hash {
            map: &self.map,
            primes: self.primes.clone(),
            offsets: self.offsets.clone(),
            mult: self.mult.clone(),
            e,
        }
    }
}

/// One table row as F32.
fn row_f32(ty: crate::ggml::ffi::ggml_type, src: &[u8], n: usize) -> Vec<f32> {
    if ty == types::F32 {
        return src
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
    }
    crate::ggml::dequantize(ty, src, n)
}

/// One FP8 table row (E4M3 `src`) as F32, times its E8M0 scale bytes (one per 32, from the
/// start of `scale`), as `engram.py` dequantizes it.
fn fp8_row_f32(src: &[u8], scale: &[u8]) -> Vec<f32> {
    src.iter()
        .enumerate()
        .map(|(i, &b)| fp8::e4m3_to_f32(b) * fp8::e8m0_to_f32(scale[i / 32]))
        .collect()
}

/// `NgramHashState`: a token's hash ids.
pub(crate) struct Hash<'a> {
    /// The compressed token map.
    pub map: &'a [i64],
    /// Per Engram layer `cols` primes and offsets, `ngram` multipliers.
    pub primes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub mult: Vec<i64>,
    pub e: &'a EngramParams,
}

impl Hash<'_> {
    /// The table rows of Engram layer `li` for `token` after `prev` (its `ngram - 1`
    /// predecessors, oldest first), into `out` (cleared): per n-gram size `2..=ngram` and head,
    /// `(t_0·m_0 ^ .. ^ t_{n-1}·m_{n-1}) mod prime + offset`, the compressed ids `t_i` of the
    /// token and its predecessors, the pad token's for those before the start.
    pub fn ids(&self, li: usize, token: u32, prev: &[u32], out: &mut Vec<i64>) {
        let e = self.e;
        let (n, cols) = (e.ngram, e.cols());
        let map = |t: u32| self.map.get(t as usize).copied().unwrap_or(0);
        let pad = map(e.pad);
        out.clear();
        // int64 products as numpy's: wrapping.
        let mut rolling = map(token).wrapping_mul(self.mult[li * n]);
        let mut blocked = false;
        for i in 1..n {
            let t = prev[n - 1 - i];
            blocked = blocked || t == NO_TOKEN;
            let tok = if blocked { pad } else { map(t) };
            rolling ^= tok.wrapping_mul(self.mult[li * n + i]);
            for h in 0..e.heads {
                let c = li * cols + (i - 1) * e.heads + h;
                out.push(rolling.rem_euclid(self.primes[c]) + self.offsets[c]);
            }
        }
    }
}

/// What the layers of one pass hand down: the cache writes of each kv source (the readers'
/// order) and, per group, the latest index source's selection and the candidate mask.
struct Shared<'a> {
    kv_written: Vec<Option<Tensor<'a>>>,
    sel: Vec<Option<Tensor<'a>>>,
    cand: Vec<Option<Tensor<'a>>>,
}

impl Model {
    /// The DeepSeek-V4.1-Flash hyperparameters of this model.
    fn ds41(&self) -> &Ds41Params {
        params(&self.hp).expect("a deepseek41 model")
    }

    /// [`Self::glm_mm`] for the weight `name` of layer `il`.
    fn ds41_mm<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm(ctx, &format!("blk.{il}.{name}"), x)
    }

    /// `x` RMS-normed times layer `il`'s weight `name`.
    fn ds41_rms<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        Ok(glm_ops::rms_norm(
            ctx,
            x,
            Some(self.lw(il, name)?),
            self.hp.rms_eps,
        ))
    }

    /// The hyper-connection mixer `which` (`attn` or `ffn`) of layer `il` over the streams
    /// `res` (its own input mix, `y`, is unused: the sublayer reads the previous one's).
    fn ds41_hc<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        which: &str,
        res: Tensor<'a>,
    ) -> Result<glm_ops::HcPre<'a>, ModelError> {
        let p = self.ds41();
        let w = |s: &str| self.lw(il, &format!("hc_{which}_{s}.weight"));
        Ok(glm_ops::hc_pre(
            ctx,
            res,
            w("fn")?,
            w("scale")?,
            w("base")?,
            self.hp.rms_eps,
            p.hc_eps,
            p.hc_iters,
        ))
    }

    /// The layers over the streams `x` (`[n_embd · hc, n]`); ends in the head's input.
    fn ds41_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        layers: Range<usize>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ds41();
        let hp = &self.hp;
        if layers != (0..hp.n_layer) {
            return Err(ModelError::Unsupported(format!(
                "DeepSeek-V4.1-Flash in pipeline stages (layers {layers:?})"
            )));
        }
        let types = pass.kv.types();
        if types.k.ggml() != types::F16 || types.v.ggml() != types::F16 {
            return Err(ModelError::Unsupported(
                "DeepSeek-V4.1's latents in a cache type other than F16".into(),
            ));
        }
        let n_groups = pass.shape.len();
        let mut st = Shared {
            kv_written: vec![None; hp.n_layer],
            sel: vec![None; n_groups],
            cand: vec![None; n_groups],
        };
        let mut res = ctx.reshape_3d(x, hp.n_embd, p.hc, pass.n);
        // The first attention reads stream 0 alone.
        let mut mix: Option<Tensor<'a>> = None;
        for il in layers {
            if let Some(li) = p
                .engram
                .as_ref()
                .and_then(|e| e.layers.iter().position(|&l| l == il))
            {
                res = self.ds41_engram(pass, il, li, res)?;
                tap.tap("engram", il, res);
            }
            let am = self.ds41_hc(ctx, il, "attn", res)?;
            let h = ops::collapse(ctx, res, mix, None);
            let h = self.ds41_rms(ctx, il, "attn_norm.weight", h)?;
            tap.tap("attn_norm", il, h);
            let out = self.ds41_attn(pass, il, h, &mut st)?;
            tap.tap("attn_out", il, out);
            res = glm_ops::hc_post(ctx, out, None, res, &am);

            let fm = self.ds41_hc(ctx, il, "ffn", res)?;
            let h = ops::collapse(ctx, res, Some(am.pre), None);
            let h = self.ds41_rms(ctx, il, "ffn_norm.weight", h)?;
            tap.tap("ffn_norm", il, h);
            let moe = self.ds41_moe(pass, il, h)?;
            tap.tap("ffn_moe_out", il, moe);
            let sh = self.ds41_shexp(ctx, il, h)?;
            tap.tap("ffn_shexp", il, sh);
            res = glm_ops::hc_post(ctx, moe, Some(sh), res, &fm);
            tap.tap("l_out", il, res);
            mix = Some(fm.pre);
        }
        let x = ops::collapse(ctx, res, mix, pass.out_rows);
        let x = glm_ops::rms_norm(ctx, x, Some(self.w("output_norm.weight")?), hp.rms_eps);
        tap.tap("result_norm", 0, x);
        Ok(x)
    }

    /// Engram `li` (layer `il`) over the streams `res`.
    fn ds41_engram<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        li: usize,
        res: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let e = self.ds41().engram.as_ref().expect("Engram");
        let ple = pass
            .ple
            .ok_or_else(|| ModelError::Unsupported("Engram without its token rows".into()))?;
        let w = e.width();
        let rows = ctx.view_2d(ple, [w, pass.n], ple.nb()[1], li * w as usize * 4);
        let kv = self.ds41_mm(ctx, il, "engram_wkv.weight", rows)?;
        Ok(ops::engram(
            ctx,
            res,
            kv,
            self.lw(il, "engram_q.weight")?,
            self.lw(il, "engram_k.weight")?,
            self.hp.rms_eps,
        ))
    }

    /// The attention of layer `il` over the normed input `x` (`[n_embd, n]`).
    fn ds41_attn<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        st: &mut Shared<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ds41();
        let n = pass.n;
        let (d, nh) = (p.head, self.hp.n_head);
        let r = p.ratios[il];
        let rope = if r == 0 { p.rope } else { p.rope_c };
        let es = pass.pos.element_size();
        let pos_all = ctx.view_2d(pass.pos, [n, 1], n as usize * es, 0);
        let pos_of =
            |off: i64, ng: i64| ctx.view_2d(pass.pos, [ng, 1], ng as usize * es, off as usize * es);

        let qr = self.ds41_mm(ctx, il, "attn_q_a.weight", x)?;
        let qr = self.ds41_rms(ctx, il, "attn_q_a_norm.weight", qr)?;
        let q = self.ds41_mm(ctx, il, "attn_q_b.weight", qr)?;
        let q = ops::rope(ctx, ctx.reshape_3d(q, d, nh, n), pos_all, &rope, 0, false);
        let kv = self.ds41_mm(ctx, il, "attn_kv.weight", x)?;
        let kv = self.ds41_rms(ctx, il, "attn_kv_a_norm.weight", kv)?;
        let kv = ops::rope(ctx, ctx.reshape_3d(kv, d, 1, n), pos_all, &rope, 0, false);
        let kv = ctx.reshape_2d(kv, d, n);
        tap.tap("kv", il, kv);

        let kcache = pass.kv.pools(il).0;
        let cells = pass.write_cells(il);
        // The compressed keys the attention reads: an earlier source's (written before this
        // layer's own write, which then follows them) or this layer's (written after it, from
        // `ds41_compress`). Either way the attention follows one node, the last write.
        let src = p.kv_of(il).filter(|_| r > 0);
        let earlier: Vec<Tensor<'a>> = (src.filter(|&s| s != il))
            .map(|s| st.kv_written[s].expect("the kv source ran first"))
            .into_iter()
            .collect();
        let own = ops::cache_write(ctx, kcache, kv, cells, pos_all, 0, 1, &earlier);
        pass.graph.borrow_mut().expand(own);
        let tables: Vec<Tensor<'a>> = pass
            .attn
            .iter()
            .map(|a| match a {
                AttnInputs::Paged { table, .. } => Ok(*table),
                _ => Err(ModelError::Unsupported(
                    "DeepSeek-V4.1's attention on the gather path".into(),
                )),
            })
            .collect::<Result<_, _>>()?;

        let last = if p.kv_source[il] {
            self.ds41_compress(pass, il, x, &tables, own, st)?
        } else {
            own
        };
        if p.index_source[il] {
            self.ds41_index(pass, il, x, qr, &tables, st)?;
        }

        let sinks = self.lw(il, "attn_sinks.weight")?;
        let mut outs = Vec::new();
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let ng = sh.tokens() as i64;
            let qnb = q.nb();
            let qg = ctx.view_3d(q, [d, nh, ng], [qnb[1], qnb[2]], off as usize * qnb[2]);
            // A ring layer's window through the sequences' rings.
            let ring = match pass.attn[gi] {
                AttnInputs::Paged { ring: Some(r), .. } if pass.kv.is_ring(il) => {
                    Some((r, sh.ring))
                }
                _ => None,
            };
            let comp = src.map(|s| Compressed {
                kcache: pass.kv.pools(s).0,
                sel: st.sel[gi].expect("the index source ran first"),
                ratio: r,
                off: d,
            });
            outs.push(ops::attn(
                ctx,
                qg,
                kcache,
                tables[gi],
                ring,
                pos_of(off, ng),
                sinks,
                comp,
                sh.n_tok as i64,
                1.0 / (d as f32).sqrt(),
                p.window,
                &[last],
            ));
        }
        let o = ctx.reshape_3d(glm_ops::join_cols(ctx, &outs), d, nh, n);
        let o = ops::rope(ctx, o, pos_all, &rope, 0, true);
        // The grouped low-rank output: each group of heads through its own matrix.
        let g = p.o_groups;
        let o = ctx.reshape_3d(o, d * nh / g, g, n);
        let o = self.ds41_mm(ctx, il, "attn_output_a.weight", o)?;
        let o = ctx.reshape_2d(o, p.o_lora * g, n);
        self.ds41_mm(ctx, il, "attn_output_b.weight", o)
    }

    /// A kv source's compression: the compressed latents (pooled over each completed group of
    /// `ratio` positions for ratio > 1) and their index keys, cached at the group's last cell,
    /// rotated to the group's first position.
    fn ds41_compress<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        tables: &[Tensor<'a>],
        own: Tensor<'a>,
        st: &mut Shared<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let p = self.ds41();
        let (n, d, di) = (pass.n, p.head, p.idx_dim);
        let r = p.ratios[il];
        let es = pass.pos.element_size();
        let pos_all = ctx.view_2d(pass.pos, [n, 1], n as usize * es, 0);
        let (kcache, vcache) = pass.kv.pools(il);
        let cells = pass.write_cells(il);
        let raw = self.ds41_mm(ctx, il, "attn_compressor_kv.weight", x)?;
        let c = if r == 1 {
            raw
        } else {
            // The raw rows and gates of every position, cached for the groups they complete
            // later; then each completed group pooled.
            let gate = self.ds41_mm(ctx, il, "attn_compressor_gate.weight", x)?;
            let w_raw = ops::cache_write(ctx, vcache, raw, cells, pos_all, 0, 1, &[]);
            let w_gate = ops::cache_write(ctx, vcache, gate, cells, pos_all, d, 1, &[w_raw]);
            let parts: Vec<Tensor<'a>> = pass
                .groups()
                .enumerate()
                .map(|(gi, (off, sh))| {
                    let ng = sh.tokens() as i64;
                    let pos = ctx.view_2d(pass.pos, [ng, 1], ng as usize * es, off as usize * es);
                    ops::pool(ctx, vcache, tables[gi], pos, sh.n_tok as i64, r, d, w_gate)
                })
                .collect();
            glm_ops::join_cols(ctx, &parts)
        };
        let c = self.ds41_rms(ctx, il, "attn_compressor_norm.weight", c)?;
        pass.tap.tap("compressed", il, c);
        let ik = self.ds41_mm(ctx, il, "indexer.attn_k.weight", c)?;
        let ik = self.ds41_rms(ctx, il, "indexer.k_norm.weight", ik)?;
        // Group j stands at position j·r: the cell of its last position, r - 1 back.
        let shift = 1 - r as i32;
        let rot = |t: Tensor<'a>, w: i64| {
            let t = ops::rope(
                ctx,
                ctx.reshape_3d(t, w, 1, n),
                pos_all,
                &p.rope_c,
                shift,
                false,
            );
            ctx.reshape_2d(t, w, n)
        };
        let w_c = ops::cache_write(ctx, kcache, rot(c, d), cells, pos_all, d, r, &[own]);
        let w_k = ops::cache_write(ctx, kcache, rot(ik, di), cells, pos_all, 2 * d, r, &[w_c]);
        pass.graph.borrow_mut().expand(w_k);
        st.kv_written[il] = Some(w_k);
        Ok(w_k)
    }

    /// An index source's selection per group: the lightning indexer's scores over the latest
    /// kv source's index keys (inside the candidate blocks after the candidate source), the
    /// best `top_k` of them.
    fn ds41_index<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        qr: Tensor<'a>,
        tables: &[Tensor<'a>],
        st: &mut Shared<'a>,
    ) -> Result<(), ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ds41();
        let (n, d, di, ih) = (pass.n, p.head, p.idx_dim, p.idx_heads);
        let r = p.ratios[il];
        let src = p.kv_of(il).expect("checked at load");
        let kcache = pass.kv.pools(src).0;
        let written = st.kv_written[src].expect("the kv source ran first");
        let es = pass.pos.element_size();
        let pos_all = ctx.view_2d(pass.pos, [n, 1], n as usize * es, 0);
        let iq = self.ds41_mm(ctx, il, "indexer.attn_q_b.weight", qr)?;
        let iq = ops::rope(
            ctx,
            ctx.reshape_3d(iq, di, ih, n),
            pos_all,
            &p.rope_c,
            0,
            false,
        );
        let w = glm_ops::linear(
            ctx,
            self.lw(il, "indexer.proj.weight")?,
            x,
            1.0 / ((di * ih) as f32).sqrt(),
            None,
        );
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let ng = sh.tokens() as i64;
            let nc = (sh.n_pool / r) as i64;
            let pos = ctx.view_2d(pass.pos, [ng, 1], ng as usize * es, off as usize * es);
            let qnb = iq.nb();
            let qg = ctx.view_3d(iq, [di, ih, ng], [qnb[1], qnb[2]], off as usize * qnb[2]);
            let mask = if p.masked(il) {
                Some(st.cand[gi].expect("the candidate source ran first"))
            } else {
                None
            };
            let score = ops::index_scores(
                ctx,
                qg,
                cols(ctx, w, off, ng),
                kcache,
                tables[gi],
                pos,
                mask,
                sh.n_tok as i64,
                r,
                2 * d,
                nc,
                written,
            );
            tap.tap("index_score", il, score);
            if let Some((c, bs, tb)) = p.candidates
                && c == il
            {
                let cand = ops::cand_mask(ctx, score, pos, r, bs, tb);
                tap.tap("cand", il, cand);
                st.cand[gi] = Some(cand);
            }
            let sel = ops::topk(ctx, score, p.top_k.min(nc as usize));
            tap.tap("sel", il, sel);
            st.sel[gi] = Some(sel);
        }
        Ok(())
    }

    /// The shared expert of layer `il`.
    fn ds41_shexp<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let g = self.ds41_mm(ctx, il, "ffn_gate_shexp.weight", x)?;
        let u = self.ds41_mm(ctx, il, "ffn_up_shexp.weight", x)?;
        let h = glm_ops::glu(ctx, g, u, self.ds41().clamp_shexp[il]);
        self.ds41_mm(ctx, il, "ffn_down_shexp.weight", h)
    }

    /// The routed experts of layer `il`: the `sqrt(softplus)` router, GGUF experts through
    /// [`moe_ffn`] (host-resident ones through the expert tier).
    fn ds41_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.ds41();
        let logits = self.ds41_mm(ctx, il, "ffn_gate_inp.weight", x)?;
        tap.tap("ffn_moe_logits", il, logits);
        let ids = ops::router_ids(
            ctx,
            logits,
            self.lw(il, "exp_probs_b.bias")?,
            self.hp.n_expert_used,
        );
        let w = ops::router_w(ctx, logits, ids, p.weights_norm, p.weights_scale);
        tap.tap("ffn_moe_weights_scaled", il, w);
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
        if let Some(t) = self.expert_tier(il) {
            // Tiered as GLM-5.3-Flash's: cached experts from VRAM slots, the others staged
            // from host memory (or read in place).
            let tiered = [t.weight(0), t.weight(1), t.weight(2)];
            if !moe_ffn::supported_glm(&refs(tiered)) {
                return unsupported(tiered);
            }
            let n = x.ne()[1];
            let ids = t.remap(ctx, &pass.graph, ids, n);
            let staged = t
                .staged_split(ctx, &pass.graph, x, n, |x, ws, stage| {
                    let out = moe_ffn::moe_ffn_staged(ctx, x, ids, w, refs(ws), pair, Some(stage));
                    moe_ffn::clamp(out, limit)
                })
                .or_else(|| {
                    t.staged(ctx, &pass.graph, x, n, |x, ws| {
                        moe_ffn::moe_ffn_clamped(ctx, x, ids, w, refs(ws), pair, limit)
                    })
                });
            if let Some(out) = staged {
                return Ok(out);
            }
            return Ok(moe_ffn::moe_ffn_clamped(
                ctx,
                x,
                ids,
                w,
                refs(tiered),
                pair,
                limit,
            ));
        }
        let lw = |s: &str| self.lw(il, &format!("ffn_{s}_exps.weight"));
        let experts = [lw("up")?, lw("gate")?, lw("down")?];
        if !moe_ffn::supported_glm(&refs(experts)) {
            return unsupported(experts);
        }
        Ok(moe_ffn::moe_ffn_clamped(
            ctx,
            x,
            ids,
            w,
            refs(experts),
            pair,
            limit,
        ))
    }
}
