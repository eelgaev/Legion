//! `glm5-next` (GLM-5.3-Flash): a stack of Kimi delta attention (KDA) layers with a sparse
//! multi-head latent attention (DSA) layer every few, hyper-connections around every sublayer
//! and a 288-expert MoE after the first dense layers, following llama.cpp's
//! `src/models/glm5-next.cpp` (MIT) op for op:
//!
//! - **Hyper-connections**: the residual is `hc` parallel streams `[n_embd, hc, n]`. Before a
//!   sublayer a mixer ([`ops::hc_pre`]) weighs the RMS-normed streams into its input and
//!   computes, per token, the weights of the sublayer output in each new stream and a
//!   Sinkhorn-normalized stream-to-stream mix ([`ops::hc_post`]). The output head reads the
//!   streams' mean ([`ops::hc_head`]).
//! - **KDA** (the recurrent layers): q, k and v each through a causal short conv and SiLU, a
//!   per-channel decay gate `lower · sigmoid(-(f_b f_a x + dt_bias) · A)`, the gated delta rule
//!   over a `[d, d]` state per head, an RMS norm gated by `sigmoid(g_b g_a x)`.
//! - **DSA** (the attention layers): MLA with the queries absorbed into the 512-wide latent
//!   space (`attn_k_b`), no RoPE, over the cached latents (keys and values both) that the
//!   indexer selects: complete pools of `kpool` positions scored by a lightning indexer over
//!   one gated-softmax key per pool, the best `top_k / kpool` of them plus the token's own
//!   incomplete pool. The selection tables are qwen4exp's ([`QsaInputs`]).
//! - **MoE**: sigmoid router with a selection bias, the chosen weights normalized and scaled,
//!   SwiGLU with GLM's limit in the experts, one shared expert.
//!
//! Every computation is a Legion op: [`ops`] (CPU references and CUDA kernels), [`moe_ffn`]
//! for GGUF experts (host-resident ones through the expert tier), [`nvfp4_mm`] / [`fp8`] for the matmuls of NVFP4 and FP8
//! checkpoints (opened from safetensors by `crate::hf`) and the sparse selection steps of
//! [`qsa_attn`]. ggml holds the tensors and views only.

mod mtp;
pub mod ops;
pub mod par;
#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use crate::ggml::{Context, Tensor, types};
use crate::gguf::GgufFile;
use crate::model::core::Replay;
use crate::model::core::arch::same_as;
use crate::model::core::expert_mv::ahead as expert_mv_ahead;
use crate::model::core::state::KvPool;
use crate::model::core::{
    Arch, ArchRef, AttnInputs, Group, GroupShape, Hparams, InputSink, Inputs, Model, ModelError,
    Pass, Recurrent, SpecPool, StatePool, cols, concat_tree,
};
use crate::model::qwen4::{QsaInputs, pools_listed, pools_new, qsa_attn};
use crate::model::{fp8, moe_ffn, nvfp4_mm};
use crate::residency::ssd_experts::trace::Predictor;

/// [`Model::glm_xh`]: the CUDA Q6_K prompt GEMM's tokens and weight rows at least (one tile
/// of tokens, 8 of rows).
const XH_MIN_TOKENS: i64 = 256;
const XH_MIN_ROWS: i64 = 1024;

/// A chunked SSD step's pieces (#1205) only go into a graph with room for them and for what
/// is left to build: tensors a piece is taken to add until one has been measured (about
/// three times what one builds) ...
const SSD_PIECE_TENSORS: usize = 48;
/// ... the most a piece has added (0: none built yet) ...
static SSD_PIECE_MEASURED: AtomicUsize = AtomicUsize::new(0);
/// ... and the tensors kept for what comes after the layers (the head, the MTP layer's rest).
const SSD_TAIL_TENSORS: usize = 512;
/// A step was too big for its graph to chunk (logged once).
static SSD_CHUNK_NO_ROOM: AtomicBool = AtomicBool::new(false);
/// Bit `k`: a step went through the slots in `k` pieces (logged once per count).
static SSD_CHUNK_SEEN: AtomicU32 = AtomicU32::new(0);

/// The GLM-5.3-Flash hyperparameters beyond [`Hparams`].
#[derive(Debug, Clone, PartialEq)]
pub struct Glm5Params {
    /// Per layer: an MLA (DSA) layer, else KDA; the MTP layers after the decoder's included.
    pub mla: Vec<bool>,
    /// Layers before the first MoE layer (dense FFN).
    pub n_dense: usize,
    /// Dense FFN width.
    pub n_ff: i64,
    pub q_lora: i64,
    pub kv_lora: i64,
    /// MLA head size (queries before absorption, values after).
    pub head_mla: i64,
    pub kda_head: i64,
    pub kda_lower: f32,
    pub idx_heads: i64,
    pub idx_dim: i64,
    pub top_k: usize,
    pub kpool: usize,
    /// Hyper-connection streams, Sinkhorn iterations and epsilon.
    pub hc: i64,
    pub hc_iters: u32,
    pub hc_eps: f32,
    pub weights_norm: bool,
    pub weights_scale: f32,
    /// Per layer: the SwiGLU limit of the routed experts, and of the shared expert and the
    /// dense FFN.
    pub clamp_exp: Vec<f32>,
    pub clamp_shexp: Vec<f32>,
    /// The layer norm epsilon (the indexer's key norm).
    pub norm_eps: f32,
    /// MTP layers after the `n_layer` decoder layers.
    pub nextn: usize,
}

static EXPERT_MV_DEVICE: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on run the routed experts of steps of up to 8 tokens through
/// Legion's expert matvec ([`crate::model::core::expert_mv`], as tiered experts do) when every
/// expert is in GPU memory, instead of the fused op's GEMM tiles (#567,
/// `legion serve --glm-expert-mv`). Each expert read once in a streaming matvec: GB10 decode
/// with Maya's IQ2 experts ~94 -> ~210 GB/s. The matvec sums Q8_1-quantized activations as
/// ggml's mmvq does, the fused op F16 ones in tiles, so greedy output differs from the default.
pub fn set_expert_mv_device(on: bool) {
    EXPERT_MV_DEVICE.store(on, Ordering::Relaxed);
}

/// See [`set_expert_mv_device`].
pub fn expert_mv_device() -> bool {
    EXPERT_MV_DEVICE.load(Ordering::Relaxed)
}

/// Most tokens of a speculative verify step whose routed experts, tiered over an expert cache
/// (`--experts cache`), take Legion's expert matvec by default: all it computes exactly (#569).
/// Past `--expert-mv-max-tokens` (8) such a step used to run the fused MoE op, whose GEMM tiles
/// read the uncached experts in place over the host link in short strided pieces, a K step at a
/// time (it is built for experts staged into VRAM first, which steps under the staging threshold
/// are not): ~2 s per 12-16 row step on an AC922 against ~90 ms through the matvec. Per (token,
/// expert) pair the matvec computes the same bits at any step size, so a verify step's rows
/// match those of plain decode steps.
pub const VERIFY_MV_TOKENS: usize = crate::model::core::expert_mv::EXACT_MAX_TOKENS as usize;

/// [`set_verify_mv_tokens`]'s value (0: [`VERIFY_MV_TOKENS`]).
static VERIFY_MV_SET: AtomicUsize = AtomicUsize::new(0);

/// The most tokens of a verify step over tiered experts the expert matvec takes in graphs built
/// from now on (0: [`VERIFY_MV_TOKENS`]; `legion serve --glm-verify-mv-max-tokens`, for A/B
/// runs). Never below `--expert-mv-max-tokens`.
pub fn set_verify_mv_tokens(n: usize) {
    VERIFY_MV_SET.store(n, Ordering::Relaxed);
}

/// The most tokens the tiered expert matvec takes in a step: in a pass whose groups all verify
/// or decode one token, with at least one verifying (`verify`), the verify limit `verify_max`
/// (0: [`VERIFY_MV_TOKENS`]) or `max` if that is more; otherwise `max` (the matvec's own limit,
/// [`expert_mv::max_tokens`](crate::model::core::expert_mv::max_tokens)).
pub(crate) fn mv_tokens(verify: bool, max: i64, verify_max: usize) -> i64 {
    if !verify {
        return max;
    }
    let v = match verify_max {
        0 => VERIFY_MV_TOKENS,
        n => n,
    };
    max.max(v as i64)
}

/// [`set_decode_mv`]'s value.
static DECODE_MV: AtomicBool = AtomicBool::new(false);

/// Whether plain decode steps over tiered experts take the verify limit too in graphs built from
/// now on (`legion serve --glm-decode-mv`, off by default; #659): a step of 9-64 sequences then
/// runs the expert matvec instead of the fused MoE op, which reads the uncached experts in place
/// and fell to 13.1 tok/s at C=16 on an AC922 (36.5 through the matvec). The matvec gives each
/// (token, expert) pair the same bits at any step size, so each sequence decodes as it does
/// alone; past 8 sequences that changes today's greedy output.
pub fn set_decode_mv(on: bool) {
    DECODE_MV.store(on, Ordering::Relaxed);
}

/// Whether a pass of groups `(verify, tokens)` (its [`GroupShape`]s') takes the verify limit of
/// [`mv_tokens`]: a verify step (every group verifies or decodes one token, and one verifies at
/// least), or with `decode` ([`set_decode_mv`]) a plain decode step (groups of one token, at
/// least one).
pub(crate) fn mv_pass(groups: impl IntoIterator<Item = (bool, usize)>, decode: bool) -> bool {
    let (mut verify, mut any) = (false, false);
    for (v, n_tok) in groups {
        if !v && n_tok != 1 {
            return false;
        }
        verify |= v;
        any = true;
    }
    verify || (decode && any)
}

/// [`mv_tokens`] for a pass of groups `shape`, with the matvec's own limit and the verify limit
/// set now: what a pass's tiered MoE layers give the expert matvec.
pub(crate) fn pass_mv_tokens(shape: &[GroupShape]) -> i64 {
    mv_tokens(
        mv_pass(
            shape.iter().map(|g| (g.verify, g.n_tok)),
            DECODE_MV.load(Ordering::Relaxed),
        ),
        crate::model::core::expert_mv::max_tokens(),
        VERIFY_MV_SET.load(Ordering::Relaxed),
    )
}

/// [`set_route_bias`]'s bonus, as `f32` bits (0: off).
static ROUTE_BIAS: AtomicU32 = AtomicU32::new(0);

/// The selection bonus graphs built from now on give the routed experts the expert cache holds
/// in VRAM, in passes whose every group decodes or verifies (`legion serve --glm-route-bias`;
/// 0, the default, is off). Prompt tokens route exactly. The router's weights stay unbiased.
/// After Xolotl's `--route-bias` (Indras-Mirror/Xolotl, MIT): more of each step's routings
/// hit the cache and fewer read host memory, at the cost of outputs that differ from the
/// default. Only tiered layers (`--experts cache`) take it.
pub fn set_route_bias(bonus: f32) {
    ROUTE_BIAS.store(bonus.to_bits(), Ordering::Relaxed);
}

/// See [`set_route_bias`].
pub fn route_bias() -> f32 {
    f32::from_bits(ROUTE_BIAS.load(Ordering::Relaxed))
}

/// What [`soft_prune`] penalizes: `(layer, expert)` pairs and the selection penalty.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftPrune {
    pub pairs: Vec<(usize, usize)>,
    pub penalty: f32,
}

/// Soft pruning (`legion serve --glm-prune`), after Xolotl's `--prune` with `--prune-penalty`
/// (Indras-Mirror/Xolotl, MIT): lowers the routing selection bias (`exp_probs_b`) of the
/// listed `(layer, expert)` pairs by `penalty` in the loaded weights, so they win only where
/// the router wants them by more than that. Prompt and decode alike; the weights the router
/// gives the chosen experts do not change. Pairs outside the model's MoE layers are skipped.
/// Returns the pairs penalized.
pub fn soft_prune(m: &Model, prune: &SoftPrune) -> Result<usize, ModelError> {
    let (pairs, penalty) = (&prune.pairs, prune.penalty);
    let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for &(il, e) in pairs {
        by_layer.entry(il).or_default().push(e);
    }
    let mut done = 0;
    for (il, experts) in by_layer {
        let Ok(b) = m.lw(il, "exp_probs_b.bias") else {
            continue;
        };
        let mut v = b.to_f32();
        let mut seen = vec![false; v.len()];
        for e in experts {
            if e < v.len() && !seen[e] {
                seen[e] = true;
                v[e] -= penalty;
                done += 1;
            }
        }
        b.set_f32(&v);
    }
    Ok(done)
}

/// Reads a soft-pruning list: one `layer expert` pair per line (Xolotl's `--prune` files),
/// blank lines and `#` comments skipped.
pub fn parse_prune_list(text: &str) -> Result<Vec<(usize, usize)>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace().map(str::parse::<usize>);
        match (it.next(), it.next(), it.next()) {
            (Some(Ok(l)), Some(Ok(e)), None) => out.push((l, e)),
            _ => return Err(format!("line {}: expected `layer expert`", n + 1)),
        }
    }
    Ok(out)
}

/// The GLM-5.3-Flash hyperparameters of `hp`, if it is one.
pub fn params(hp: &Hparams) -> Option<&Glm5Params> {
    hp.arch.as_any().downcast_ref::<Glm5Params>()
}

/// `x` in F16 for its one product with the Q6_K weights `w`, where [`ops::xh_one_takes`] on
/// the engine's GPU (#862; [`Model::glm_xh`] for an `x` of several products).
fn glm_xh_one<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Option<Tensor<'a>> {
    let n: i64 = x.ne()[1..].iter().product();
    let takes = w.ty() == types::Q6_K
        && w.ne()[2] == 1
        && x.ty() == types::F32
        && x.is_contiguous()
        && ops::xh_one_takes(
            crate::model::core::mmq_preq::gb10(),
            w.ne()[0],
            w.ne()[1],
            n,
        );
    takes.then(|| ops::f16(ctx, x))
}

/// Whether the product with the Q6_K weights `a` hands its outputs to the one with `b` as F16
/// rows at a prompt step of `n` tokens, where [`ops::f16_chain_takes`] on the engine's GPU (#862).
fn glm_f16_chain(a: Tensor<'_>, b: Tensor<'_>, n: i64) -> bool {
    a.ty() == types::Q6_K
        && b.ty() == types::Q6_K
        && ops::f16_chain_takes(crate::model::core::mmq_preq::gb10(), n)
}

/// A per-layer key: an array of at least `n` numbers, or one for every layer.
fn per_layer_f(g: &GgufFile, key: &str, n: usize) -> Result<Vec<f64>, ModelError> {
    let v = g.get(key).ok_or_else(|| ModelError::Metadata(key.into()))?;
    let out: Option<Vec<f64>> = match v.as_array() {
        Some(a) => a.iter().map(|x| x.as_f64()).collect(),
        None => v.as_f64().map(|x| vec![x; n]),
    };
    out.filter(|o| o.len() >= n)
        .ok_or_else(|| ModelError::Metadata(key.into()))
}

pub(crate) fn parse(g: &GgufFile) -> Result<Hparams, ModelError> {
    let arch = "glm5-next";
    let key = |k: &str| format!("{arch}.{k}");
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
    let unsupported = |what: String| Err(ModelError::Unsupported(what));

    let n_all = u("block_count")? as usize;
    let nextn = g.get_u64(&key("nextn_predict_layers")).unwrap_or(0) as usize;
    let n_layer = n_all - nextn;
    let kv_heads = per_layer_f(g, &key("attention.head_count_kv"), n_all)?;
    // The MTP (NextN) layers too: theirs is a DSA layer as well.
    let mla: Vec<bool> = kv_heads.iter().map(|&h| h > 0.0).collect();
    if let Some(types) = g.get(&key("attention.indexer.types")) {
        let full = types
            .as_array()
            .is_some_and(|a| a.iter().all(|v| v.as_bool() != Some(false)));
        if !full {
            return unsupported("GLM shared indexer layers (attention.indexer.types)".into());
        }
    }
    if !flag("attention.indexer.kpool_select_tail", true) {
        return unsupported("GLM indexer without the incomplete pool".into());
    }
    if g.get_u64(&key("rope.dimension_count")).unwrap_or(0) != 0 {
        return unsupported("GLM MLA with RoPE".into());
    }
    if u("expert_gating_func")? != 2 {
        return unsupported("a GLM router other than sigmoid".into());
    }
    if u("ssm.conv_kernel")? != 4 {
        return unsupported("a KDA conv kernel other than 4".into());
    }
    let tensor = |name: &str| {
        g.tensor(name)
            .ok_or_else(|| ModelError::Tensor(name.into()))
    };
    let n_vocab = tensor("token_embd.weight")?.shape[1] as i64;
    let kda_head = u("kda.head_dim")? as i64;
    // The KDA width, from the first KDA layer's q projection.
    let d_inner = match mla.iter().position(|&m| !m) {
        Some(il) => tensor(&format!("blk.{il}.attn_q.weight"))?.shape[1] as i64,
        None => kda_head,
    };
    let n_kda_head = d_inner / kda_head;
    let n_embd = u("embedding_length")? as i64;
    let kv_lora = u("attention.kv_lora_rank")? as i64;
    let rms_eps = f("attention.layer_norm_rms_epsilon")?;
    let params = Glm5Params {
        mla: mla.clone(),
        n_dense: u("leading_dense_block_count")? as usize,
        n_ff: u("feed_forward_length")? as i64,
        q_lora: u("attention.q_lora_rank")? as i64,
        kv_lora,
        head_mla: u("attention.key_length_mla")? as i64,
        kda_head,
        kda_lower: f("kda.gate_lower_bound")?,
        idx_heads: u("attention.indexer.head_count")? as i64,
        idx_dim: u("attention.indexer.key_length")? as i64,
        top_k: u("attention.indexer.top_k")? as usize,
        kpool: u("attention.indexer.kpool")? as usize,
        hc: u("hyper_connection.count")? as i64,
        hc_iters: u("hyper_connection.sinkhorn_iterations")? as u32,
        hc_eps: f("hyper_connection.epsilon")?,
        weights_norm: flag("expert_weights_norm", false),
        weights_scale: f("expert_weights_scale").unwrap_or(1.0),
        clamp_exp: per_layer_f(g, &key("swiglu_clamp_exp"), n_all)?
            .iter()
            .map(|&x| x as f32)
            .collect(),
        clamp_shexp: per_layer_f(g, &key("swiglu_clamp_shexp"), n_all)?
            .iter()
            .map(|&x| x as f32)
            .collect(),
        norm_eps: f("attention.layer_norm_epsilon").unwrap_or(rms_eps),
        nextn,
    };
    if params.kpool < 2 || !params.top_k.is_multiple_of(params.kpool) {
        return unsupported(format!(
            "indexer top_k {} with pools of {}",
            params.top_k, params.kpool
        ));
    }
    Ok(Hparams {
        n_layer,
        n_embd,
        n_head: u("attention.head_count")? as i64,
        // MLA: one latent row per position, keys and values both.
        n_head_kv: 1,
        head_dim: kv_lora,
        n_rot: 0,
        rope_sections: [0; 4],
        rope_freq_base: 10_000.0,
        n_ctx_train: u("context_length")? as i32,
        rms_eps,
        moe: true,
        n_expert: u("expert_count")? as i64,
        n_expert_used: u("expert_used_count")? as i64,
        // Unused: `mla` says which layers are which.
        full_attention_interval: 1,
        ssm_d_conv: 4,
        // The delta-rule state is [kda_head, kda_head] per head; the conv state holds q, k
        // and v (3 × d_inner channels: d_inner + 2 · n_group · d_state).
        ssm_d_state: kda_head,
        ssm_n_group: n_kda_head,
        ssm_dt_rank: n_kda_head,
        ssm_d_inner: d_inner,
        n_vocab,
        arch: ArchRef::new(params),
    })
}

impl Glm5Params {
    /// The indexer cache row: raw key | gate | pooled key, as `2 × index_dim` F16 values.
    fn index_row(&self) -> i64 {
        3 * self.idx_dim
    }
}

impl Arch for Glm5Params {
    fn name(&self) -> &'static str {
        "glm5-next"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, _hp: &Hparams, il: usize) -> bool {
        !self.mla.get(il).copied().unwrap_or(true)
    }

    fn kv_dims(&self, hp: &Hparams, _il: usize) -> (i64, i64) {
        (hp.head_dim, 1)
    }

    fn residual_streams(&self) -> i64 {
        self.hc
    }

    fn group_inputs_len(&self) -> usize {
        8
    }

    fn group_inputs<'a>(&self, ctx: &'a Context, g: &GroupShape, i: usize) -> Vec<Tensor<'a>> {
        QsaInputs::new(ctx, self.kpool, g, i).list().to_vec()
    }

    fn group_pools(&self, g: &Group<'_>) -> (usize, usize) {
        (pools_listed(g, self.kpool), pools_new(g, self.kpool))
    }

    /// As qwen4exp's: when no sequence has more complete pools than the indexer picks, every
    /// visible pool is selected.
    fn group_picks_all(&self, g: &Group<'_>) -> bool {
        let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
        qsa_attn::pick_all()
            && g.seqs
                .iter()
                .all(|s| (s.n_past + n_tok) / self.kpool <= self.top_k / self.kpool)
    }

    fn fill_inputs(
        &self,
        _m: &Model,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        _rec: &StatePool,
        kv: &KvPool,
        _spec: Option<&SpecPool>,
        sink: &mut dyn InputSink,
    ) {
        for ((g, sh), t) in groups.iter().zip(shape).zip(&inputs.group) {
            QsaInputs::from_slice(t).fill(self.kpool, g, sh, kv, sink);
        }
    }

    fn index_dim(&self) -> i64 {
        self.index_row() / 2
    }

    fn indexed(&self, il: usize) -> bool {
        self.mla.get(il).copied().unwrap_or(false)
    }

    fn sparse_keys(&self) -> Option<(usize, usize)> {
        Some((self.top_k, self.kpool))
    }

    /// The MLA op reads the latents in place through the block tables.
    fn paged(&self, _m: &Model, _kv: &KvPool) -> bool {
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
        Ok(Some(ops::embed(
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
        // The MTP head reads the last layer's output head input (the streams' normed mean) of
        // every row; no other capture.
        let last = m.hp.n_layer - 1;
        let mtp = match capture {
            [] => false,
            [il] if *il == last && layers.end == m.hp.n_layer => true,
            _ => {
                return Err(ModelError::Unsupported(format!(
                    "hidden-state captures {capture:?} of GLM-5.3-Flash (its last layer's only)"
                )));
            }
        };
        if !mtp {
            return Ok((m.glm_layers(pass, x, layers, false)?, Vec::new()));
        }
        // Every row's head input; the logits' rows are taken out of it (each row as the
        // cropped head would give it).
        let h = m.glm_layers(pass, x, layers, true)?;
        let x = match pass.out_rows {
            Some(rows) => ops::get_cols(pass.ctx, h, rows),
            None => h,
        };
        Ok((x, vec![(last, h)]))
    }

    /// The last layer ends in the streams' mean, normed, at the output rows.
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
        Ok(Some((logits, ops::argmax_wide(ctx, logits))))
    }

    /// The NextN layer in the GGUF (`blk.{n_layer}.nextn.*`), see `mtp.rs`.
    fn mtp(&self) -> bool {
        true
    }

    /// The head reads the output head's input: one stream.
    fn mtp_streams(&self) -> i64 {
        1
    }

    /// The head's layer is a DSA layer with an indexer of its own.
    fn mtp_sparse(&self) -> bool {
        true
    }

    fn recurrent(&self) -> Recurrent {
        Recurrent::Kda
    }

    /// The kept tokens through the same convs and scan as a decode step's (each runs its
    /// tokens one after another), from the verify step's stored inputs.
    fn spec_replay<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        il: usize,
        r: Replay<'a>,
    ) -> Result<Vec<Tensor<'a>>, ModelError> {
        // The channels the states hold: all, or a parallel step's GPU's heads' (`par`).
        let (d, c) = (self.kda_head, r.x[0].ne()[0]);
        let h = c / d;
        let n = r.x[0].ne()[2];
        let split = r.shard.map_or_else(String::new, |g| par::shard_name("", g));
        let conv_w = |s: &str| m.lw(il, &format!("ssm_conv1d_{s}.weight{split}"));
        // q, k and v in one launch (#676): a commit is ~4 launches a KDA layer, launched
        // while the host does the step's cache work, and each took the driver from it.
        let y = ops::kda_conv3(
            ctx,
            r.x,
            [conv_w("q")?, conv_w("k")?, conv_w("v")?],
            r.conv,
            r.keep,
        );
        let part = |j: usize| {
            let nb1 = d as usize * 4;
            let nb2 = c as usize * 4;
            ctx.view_3d(y, [d, h, r.keep * n], [nb1, nb2], j * y.nb()[2])
        };
        let (q, k, v) = (part(0), part(1), part(2));
        let gnb = r.g.nb();
        let g = ctx.view_4d(r.g, [d, h, r.keep, n], [d as usize * 4, gnb[1], gnb[2]], 0);
        Ok(vec![ops::kda_scan(ctx, q, k, v, g, r.beta, r.ssm, r.keep)])
    }

    fn build_mtp<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<
        (
            crate::ggml::Graph<'a>,
            crate::model::core::mtp::MtpGraph<'a>,
        ),
        ModelError,
    > {
        m.build_mtp_glm(ctx, kv, rec, shape)
    }

    fn external_drafters(&self) -> bool {
        false
    }

    /// [`ops::argmax`] and [`ops::argmax_wide`] take the first of the largest (#695).
    fn argmax_first_of_ties(&self) -> bool {
        true
    }

    /// On all four of an AC922's GPUs the peer fetch with the side-stream miss copy took decode
    /// 16.4 -> 18.5 tok/s and C=4 20.0 -> 26.9 tok/s, text identical (2-row GLU tiles).
    fn peer_decode_pairs(&self) -> bool {
        true
    }
}

/// How one GPU of GLM's parallel steps reads a layer ([`par`]): the suffix of its shards of
/// the split weights (`@s<d>`) and of its copies of the mirrored ones (`@m<d>`, none on the
/// layer's owner), and the GPUs the heads split over.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Part<'s> {
    pub split: &'s str,
    pub mirror: &'s str,
    pub n_dev: i64,
}

impl Part<'static> {
    /// The whole layer (one GPU, the weights by their own names).
    pub const WHOLE: Self = Self {
        split: "",
        mirror: "",
        n_dev: 1,
    };
}

impl Model {
    /// The GLM-5.3-Flash hyperparameters of this model.
    fn glm(&self) -> &Glm5Params {
        params(&self.hp).expect("a glm5-next model")
    }

    /// `name @ x` by the weight's format: NVFP4 (BF16 activations) and FP8 E4M3 checkpoints
    /// take their matmuls, everything else (GGUF blocks, BF16, F32) [`ops::linear`]. A
    /// checkpoint's `<base>.scale` multiplies the result.
    pub(crate) fn glm_mm<'a>(
        &'a self,
        ctx: &'a Context,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm_h(ctx, name, x, None)
    }

    /// [`Self::glm_mm`] with `xh` from [`Self::glm_xh`] for the GGUF K-quant path.
    fn glm_mm_h<'a>(
        &'a self,
        ctx: &'a Context,
        name: &str,
        x: Tensor<'a>,
        xh: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let w = self.w(name)?;
        let ws = name
            .strip_suffix(".weight")
            .and_then(|b| self.w(&format!("{b}.scale")).ok());
        if w.ty() == types::NVFP4 && nvfp4_mm::supports(w) {
            if ws.is_some_and(|s| s.elements() != 1) {
                return Err(ModelError::Unsupported(format!(
                    "`{name}`: an NVFP4 scale per row"
                )));
            }
            let x = if x.is_contiguous() {
                x
            } else {
                return Err(ModelError::Layout(format!("`{name}`: strided NVFP4 input")));
            };
            let n: i64 = x.ne()[1..].iter().product();
            if self.linear.nvfp4_a4_takes(w, n, false) {
                // --nvfp4-a4 prefill / all: 4-bit activations (the dense FFN's, #575)
                return Ok(crate::model::nvfp4_a4::mul_mat(ctx, w, x, ws));
            }
            return Ok(nvfp4_mm::mul_mat(ctx, w, ops::bf16(ctx, x), ws));
        }
        if w.ty() == types::I8 {
            return Ok(fp8::mul_mat(ctx, w, x, ws));
        }
        let xh = xh.filter(|_| w.ty() == types::Q6_K && w.ne()[2] == 1);
        let xh = xh.or_else(|| glm_xh_one(ctx, w, x));
        Ok(ops::linear_h(ctx, w, x, xh, 1.0, ws))
    }

    /// [`Self::glm_mm`] of the Q6_K weights `name` with F16 outputs ([`ops::linear_io`]) for a
    /// next product that reads them as its F16 rows, where [`glm_f16_chain`] (#862).
    fn glm_mm_f16<'a>(
        &'a self,
        ctx: &'a Context,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let w = self.w(name)?;
        let ws = name
            .strip_suffix(".weight")
            .and_then(|b| self.w(&format!("{b}.scale")).ok());
        let xh = glm_xh_one(ctx, w, x);
        Ok(ops::linear_io(
            ctx,
            w,
            x,
            xh,
            1.0,
            ws,
            ops::LinTc::Auto,
            true,
        ))
    }

    /// `x` in F16 for the products of `x` with the weights `names` of layer `il`, when 2+ of
    /// them take the GB10's Q6_K prompt GEMM, which then reads it in place of rounding `x`
    /// itself (the same bits, see [`ops::linear_h`]). Only where that GEMM's `mma.sync` runs
    /// (sm_80+): a V100 with the GB10 kernels on (#574) would round `x` for nothing.
    fn glm_xh<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        names: &[&str],
        x: Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        if !ops::gb10_linear()
            || !crate::model::moe_ffn::mma16()
            || x.ty() != types::F32
            || !x.is_contiguous()
            || x.ne()[1] < XH_MIN_TOKENS
        {
            return None;
        }
        let gemms = names
            .iter()
            .filter_map(|n| self.w(&format!("blk.{il}.{n}")).ok())
            .filter(|w| {
                w.ty() == types::Q6_K
                    && w.ne()[2] == 1
                    && w.ne()[1] >= XH_MIN_ROWS
                    && w.ne()[0] % 512 == 0
            })
            .count();
        (gemms >= 2).then(|| ops::f16(ctx, x))
    }

    /// [`Self::glm_mm`] for the weight `name` of layer `il`.
    fn glm_lmm<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm(ctx, &format!("blk.{il}.{name}"), x)
    }

    /// [`Self::glm_lmm`] with `xh` from [`Self::glm_xh`].
    fn glm_lmm_h<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
        xh: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_mm_h(ctx, &format!("blk.{il}.{name}"), x, xh)
    }

    /// The hyper-connection mixer `which` (`attn` or `ffn`) of layer `il` over the streams
    /// `res`, and its output normed by `{which}_norm` (the sublayer's input), in one node.
    fn glm_hc_pre<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        which: &str,
        res: Tensor<'a>,
    ) -> Result<(ops::HcPre<'a>, Tensor<'a>), ModelError> {
        self.glm_hc_pre_on(ctx, il, which, res, "")
    }

    /// [`Self::glm_hc_pre`] with weights named `<name><sfx>` (a GPU's mirrors, [`par`]).
    pub(crate) fn glm_hc_pre_on<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        which: &str,
        res: Tensor<'a>,
        sfx: &str,
    ) -> Result<(ops::HcPre<'a>, Tensor<'a>), ModelError> {
        let p = self.glm();
        let w = |s: &str| self.lw(il, &format!("hc_{which}_{s}.weight{sfx}"));
        Ok(ops::hc_pre_norm(
            ctx,
            res,
            w("fn")?,
            w("scale")?,
            w("base")?,
            self.hp.rms_eps,
            p.hc_eps,
            p.hc_iters,
            self.lw(il, &format!("{which}_norm.weight{sfx}"))?,
            self.hp.rms_eps,
        ))
    }

    /// Layers `layers` over the streams `x` (`[n_embd, hc, n]`); the last layer ends in the
    /// output head's input, of every row with `all_rows` (else the output rows').
    fn glm_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        layers: Range<usize>,
        all_rows: bool,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.glm();
        let hp = &self.hp;
        let e = hp.n_embd;
        let mut res = ctx.reshape_3d(x, e, p.hc, pass.n);
        // The most tensors a layer has taken, for the room a chunked SSD step may use (#1205).
        let mut most = 0;
        for il in layers.clone() {
            let start = ctx.tensors_left();
            let (pre, h) = self.glm_hc_pre(ctx, il, "attn", res)?;
            tap.tap("hc_attn_pre", il, pre.y);
            tap.tap("attn_norm", il, h);
            let out = if p.mla[il] {
                self.glm_dsa(pass, il, h)?
            } else {
                self.glm_kda(pass, il, h)?
            };
            res = ops::hc_post(ctx, out, None, res, &pre);
            tap.tap("hc_attn_post", il, res);

            let (pre, h) = self.glm_hc_pre(ctx, il, "ffn", res)?;
            tap.tap("hc_ffn_pre", il, pre.y);
            tap.tap("ffn_norm", il, h);
            res = if il < p.n_dense {
                // The first MoE layer's router lookahead (#611) from the last dense layer.
                self.glm_ssd_predict(pass, il + 1, h, Predictor::PostAttn, None)?;
                let out = self.glm_ffn(ctx, il, "", p.clamp_shexp[il], h)?;
                tap.tap("ffn_out", il, out);
                ops::hc_post(ctx, out, None, res, &pre)
            } else {
                // Room for this layer's pieces: what is left once the rest of this layer and
                // each later one (as many as the most a layer took, or twice this one's so
                // far) and the tail have theirs.
                let left = ctx.tensors_left();
                let part = start - left;
                let rest = most.max(2 * part) * (layers.end - il) - part + SSD_TAIL_TENSORS;
                let moe = self.glm_moe(pass, il, h, left.saturating_sub(rest))?;
                tap.tap("ffn_moe_out", il, moe);
                let sh = self.glm_ffn(ctx, il, "_shexp", p.clamp_shexp[il], h)?;
                tap.tap("ffn_shexp", il, sh);
                ops::hc_post(ctx, moe, Some(sh), res, &pre)
            };
            tap.tap("l_out", il, res);
            // The router lookahead from this layer's output (#611 trace, --ssd-prefetch
            // router-post / router-both, #615): the next layer's FFN mixer and norm over it,
            // then that layer's router.
            if il + 1 < layers.end
                && let Some(ssd) = self.ssd_experts()
                && ssd
                    .predict_width(il + 1, Predictor::PostMoe, pass.n as usize)
                    .is_some()
            {
                let (_, h) = self.glm_hc_pre(ctx, il + 1, "ffn", res)?;
                self.glm_ssd_predict(pass, il + 1, h, Predictor::PostMoe, None)?;
            }
            most = most.max(start - ctx.tensors_left());
        }
        if layers.end != hp.n_layer {
            // The next stage's input is [n_embd · hc, n].
            return Ok(ctx.reshape_2d(res, e * p.hc, pass.n));
        }
        let x = ops::hc_head(
            ctx,
            res,
            pass.out_rows.filter(|_| !all_rows),
            self.w("output_norm.weight")?,
            hp.rms_eps,
        );
        tap.tap("result_norm", 0, x);
        Ok(x)
    }

    /// A dense SwiGLU FFN (`suffix` "") or the shared expert (`"_shexp"`) of layer `il`, with
    /// GLM's limit `limit`.
    fn glm_ffn<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        suffix: &str,
        limit: f32,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_ffn_on(ctx, il, suffix, limit, x, "")
    }

    /// [`Self::glm_ffn`] with weights named `<name><sfx>` (a GPU's mirrors or shards, [`par`]).
    pub(crate) fn glm_ffn_on<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        suffix: &str,
        limit: f32,
        x: Tensor<'a>,
        sfx: &str,
    ) -> Result<Tensor<'a>, ModelError> {
        let (gn, un) = (
            format!("ffn_gate{suffix}.weight{sfx}"),
            format!("ffn_up{suffix}.weight{sfx}"),
        );
        let dn = format!("ffn_down{suffix}.weight{sfx}");
        // Decode steps: gate, up and the GLU in one node (#606), the same bits.
        let n: i64 = x.ne()[1..].iter().product();
        let (gw, uw) = (self.lw(il, &gn)?, self.lw(il, &un)?);
        let scaled = [&gn, &un]
            .iter()
            .any(|s| self.lw(il, &s.replace(".weight", ".scale")).is_ok());
        if !scaled && x.ty() == types::F32 && x.is_contiguous() && ops::glu_mv_takes(gw, uw, n) {
            let h = ops::glu_mv(ctx, gw, uw, x, limit);
            return self.glm_lmm(ctx, il, &dn, h);
        }
        let xh = self.glm_xh(ctx, il, &[&gn, &un], x);
        let g = self.glm_lmm_h(ctx, il, &gn, x, xh)?;
        let u = self.glm_lmm_h(ctx, il, &un, x, xh)?;
        let h = ops::glu(ctx, g, u, limit);
        self.glm_lmm(ctx, il, &dn, h)
    }

    /// The routed experts of layer `il`: sigmoid router with the selection bias, the chosen
    /// weights normalized and scaled; GGUF experts through [`moe_ffn`], NVFP4 experts through
    /// [`nvfp4_mm::mul_mat_id`]. `room`: the tensors a step chunked through the SSD tier's
    /// slots may add to the graph (#1205; a step that needs more stages its experts).
    fn glm_moe<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        room: usize,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.glm();
        let hp = &self.hp;
        let logits = self.glm_lmm(ctx, il, "ffn_gate_inp.weight", x)?;
        tap.tap("ffn_moe_logits", il, logits);
        let bias = self.lw(il, "exp_probs_b.bias")?;
        let bonus = route_bias();
        let decoding = pass.shape.iter().all(|g| g.verify || g.n_tok == 1);
        let ids = match self.expert_tier(il) {
            // --glm-route-bias: decode and verify steps favour the cached experts.
            Some(t) if bonus != 0.0 && decoding => ops::router_ids_resident(
                ctx,
                logits,
                bias,
                t.table_view(ctx),
                t.n_slots(),
                bonus,
                hp.n_expert_used,
            ),
            _ => ops::router_ids(ctx, logits, bias, hp.n_expert_used),
        };
        let w = ops::router_w(ctx, logits, ids, p.weights_norm, p.weights_scale);
        tap.tap("ffn_moe_weights_scaled", il, w);
        let limit = p.clamp_exp[il];
        // Routed SwiGLU over NVFP4 experts `[up, gate, down]` with per-expert scales, for the
        // tokens of `x` (all of the step's, or a piece's: `w` their router weights).
        let n_step = x.ne()[1];
        let nvfp4 = |[up, gate, down]: [Tensor<'a>; 3],
                     [su, sg, sd]: [Option<Tensor<'a>>; 3],
                     ids: Tensor<'a>,
                     x: Tensor<'a>,
                     w: Tensor<'a>| {
            let n = x.ne()[1];
            // A piece of a chunked step takes 4-bit activations as its whole step would.
            if self.linear.nvfp4_a4_takes(gate, n_step, true) {
                // --nvfp4-a4: 4-bit activations, the gate, up and down matmuls quantizing F32
                // inputs, GLM's clamped GLU between (#575)
                use crate::model::nvfp4_a4;
                let xf = ctx.reshape_3d(
                    if x.is_contiguous() { x } else { ctx.cont(x) },
                    hp.n_embd,
                    1,
                    n,
                );
                let g = nvfp4_a4::mul_mat_id(ctx, gate, xf, ids, sg);
                let u = nvfp4_a4::mul_mat_id(ctx, up, xf, ids, su);
                let h = ops::glu(ctx, g, u, limit);
                let y = nvfp4_a4::mul_mat_id(ctx, down, h, ids, sd);
                return ops::weighted_sum(ctx, y, w);
            }
            let xb = ctx.reshape_3d(ops::bf16(ctx, x), hp.n_embd, 1, n);
            // Prefill batches (the grouped GEMM's): gate, up and the clamped GLU as one grouped
            // GEMM, the same bits (#575).
            let [k, m, n_expert, _] = gate.ne();
            let gemm = matches!(
                nvfp4_mm::pick_grouped(k, m, ids.ne()[0] * ids.ne()[1], n_expert),
                nvfp4_mm::Kernel::Gemm { .. }
            );
            if gemm && gate.ne() == up.ne() && gate.nb() == up.nb() {
                let h = nvfp4_mm::swiglu_id_clamped(ctx, gate, up, xb, ids, sg, su, limit);
                let y = nvfp4_mm::mul_mat_id(ctx, down, h, ids, sd);
                return ops::weighted_sum(ctx, y, w);
            }
            let g = nvfp4_mm::mul_mat_id(ctx, gate, xb, ids, sg);
            let u = nvfp4_mm::mul_mat_id(ctx, up, xb, ids, su);
            let h = ops::bf16(ctx, ops::glu(ctx, g, u, limit));
            let y = nvfp4_mm::mul_mat_id(ctx, down, h, ids, sd);
            ops::weighted_sum(ctx, y, w)
        };
        if let Some(ssd) = self.ssd_experts().filter(|s| s.has_layer(il)) {
            let piece = SSD_PIECE_MEASURED.load(Ordering::Relaxed);
            let piece = if piece == 0 { SSD_PIECE_TENSORS } else { piece };
            // A piece (at most a pair per slot) always takes the grouped mv kernels; chunk only
            // the steps whose staged matmuls take them too (a 4-bit step of more tokens than
            // the mv kernel takes, or one past --nvfp4-expert-mv-max pairs per expert, is on
            // the GEMM, other sums), so chunking keeps every step's bits.
            let staged_mv = if self.linear.nvfp4_a4_batch(n_step, true) {
                n_step <= crate::model::nvfp4_a4::MV_MAX_TOKENS
            } else {
                let pairs = hp.n_expert_used * n_step;
                matches!(
                    nvfp4_mm::pick_grouped(hp.n_embd, 0, pairs, hp.n_expert),
                    nvfp4_mm::Kernel::Mv { .. }
                )
            };
            let chunks = ssd.pieces(n_step as usize, decoding && staged_mv);
            let chunks = chunks.filter(|&k| {
                let fits = k * piece <= room;
                if !fits && !SSD_CHUNK_NO_ROOM.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "SSD expert tier: a step of {n_step} tokens has no room in its graph for \
                         {k} pieces at layer {il}; its experts go through a staging buffer"
                    );
                }
                fits
            });
            if let Some(k) = chunks {
                let bit = 1u32 << k.min(31);
                if SSD_CHUNK_SEEN.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
                    tracing::info!(
                        "SSD expert tier: a step of {n_step} tokens goes through the slots in {k} \
                         pieces (layer {il})"
                    );
                }
                let before = ctx.tensors_left();
                // A decode or verify step of more pairs than slots (#1205): pieces of rows, each
                // its experts in slots and its matmuls, one after the other (each fetch after the
                // previous piece's result), instead of a staging buffer. Each row's sums are the
                // ones the whole step gives it (the same kernels, one token at a time).
                let ids = if ids.is_contiguous() {
                    ids
                } else {
                    ctx.cont(ids)
                };
                let mut outs = Vec::with_capacity(k);
                let mut after = None;
                for r in crate::residency::ssd_experts::piece_rows(n_step as usize, k) {
                    let (r0, nr) = (r.start as i64, r.len() as i64);
                    let l = ssd
                        .fetch_rows(ctx, il, ids, r, after)
                        .map_err(|e| ModelError::Unsupported(format!("layer {il}: {e}")))?;
                    let scales = match l.scales {
                        Some([u, g, d]) => [Some(u), Some(g), Some(d)],
                        None => [None; 3],
                    };
                    let y = nvfp4(
                        l.weights,
                        scales,
                        l.ids,
                        cols(ctx, x, r0, nr),
                        cols(ctx, w, r0, nr),
                    );
                    after = Some(y);
                    outs.push(y);
                }
                let y = concat_tree(ctx, outs);
                let each = (before - ctx.tensors_left()).div_ceil(k);
                SSD_PIECE_MEASURED.fetch_max(each, Ordering::Relaxed);
                return Ok(y);
            }
            // The router lookahead (#611): the next layer's router on this layer's input, after
            // this layer's fetch (trace, --ssd-prefetch router), or before it with
            // --ssd-prefetch-early (#615: its reads queue behind the fetch's own).
            let early = ssd.prefetch_early();
            if early {
                self.glm_ssd_predict(pass, il + 1, x, Predictor::PostAttn, None)?;
            }
            // Experts read from the checkpoint as routed (#560): slots or a staging buffer.
            let l = ssd
                .fetch(ctx, il, ids)
                .map_err(|e| ModelError::Unsupported(format!("layer {il}: {e}")))?;
            if !early {
                self.glm_ssd_predict(pass, il + 1, x, Predictor::PostAttn, Some(l.ids))?;
            }
            let scales = match l.scales {
                Some([u, g, d]) => [Some(u), Some(g), Some(d)],
                None => [None; 3],
            };
            return Ok(nvfp4(l.weights, scales, l.ids, x, w));
        }
        let lw = |s: &str| self.lw(il, &format!("ffn_{s}_exps.weight"));
        let (up, gate, down) = (lw("up")?, lw("gate")?, lw("down")?);
        if [up, gate, down].iter().all(|w| nvfp4_mm::supports(*w)) {
            let scale = |s: &str| self.lw(il, &format!("ffn_{s}_exps.scale")).ok();
            return Ok(nvfp4(
                [up, gate, down],
                [scale("up"), scale("gate"), scale("down")],
                ids,
                x,
                w,
            ));
        }
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
        // Maya's IQ2 experts on the GLM-only GEMMs where the GB10 kernels are on (same bits)
        // (and, opt-in, their int8 prompt GEMMs: output-moving, #603)
        let glm = |out| {
            let out = moe_ffn::glm_kernels(out, ops::gb10_linear());
            moe_ffn::glm_q8(out, ops::gb10_linear() && ops::expert_q8())
        };
        if let Some(t) = self.expert_tier(il) {
            // Tiered, as the shared MoE layer streams Qwen3.8-Flash-Next's experts (Legion ops
            // only): cached experts from VRAM slots, the others from host memory. A big step
            // reads them from staged VRAM copies (each piece of the copy as it lands, else the
            // whole layer's); a small one through Legion's expert matvec, the missed experts
            // staged on a side stream beside the cached ones (#253) or read over the link; what
            // the matvec does not take (types, switches) in place in the fused op.
            let tiered = [t.weight(0), t.weight(1), t.weight(2)];
            if !moe_ffn::supported_glm(&refs(tiered)) {
                return unsupported(tiered);
            }
            let n = x.ne()[1];
            let ids = t.remap(ctx, &pass.graph, ids, n);
            let staged = t
                .staged_split(ctx, &pass.graph, x, n, |x, ws, stage| {
                    let out = moe_ffn::moe_ffn_staged(ctx, x, ids, w, refs(ws), pair, Some(stage));
                    glm(moe_ffn::clamp(out, limit))
                })
                .or_else(|| {
                    t.staged(ctx, &pass.graph, x, n, |x, ws| {
                        glm(moe_ffn::moe_ffn_clamped(
                            ctx,
                            x,
                            ids,
                            w,
                            refs(ws),
                            pair,
                            limit,
                        ))
                    })
                });
            if let Some(out) = staged {
                return Ok(out);
            }
            // Verify steps past the matvec's limit take it too, up to its exact range (#569), and
            // with --glm-decode-mv plain decode steps (#659): the fused op below reads uncached
            // experts in place, ~20x slower in such small steps.
            let max = mv_tokens(
                mv_pass(
                    pass.shape.iter().map(|g| (g.verify, g.n_tok)),
                    DECODE_MV.load(Ordering::Relaxed),
                ),
                crate::model::core::expert_mv::max_tokens(),
                VERIFY_MV_SET.load(Ordering::Relaxed),
            );
            if !t.gathers(x.ne()[1])
                && let Some(y) =
                    Self::glm_expert_mv(ctx, (Some(t.n_slots()), max), tiered, x, ids, limit, None)
            {
                return Ok(ops::weighted_sum(ctx, y, w));
            }
            return Ok(glm(moe_ffn::moe_ffn_clamped(
                ctx,
                x,
                ids,
                w,
                refs(tiered),
                pair,
                limit,
            )));
        }
        if !moe_ffn::supported_glm(&refs([up, gate, down])) {
            return unsupported([up, gate, down]);
        }
        if expert_mv_device()
            && let Some(y) = Self::glm_expert_mv(
                ctx,
                (None, crate::model::core::expert_mv::max_tokens()),
                [up, gate, down],
                x,
                ids,
                limit,
                None,
            )
        {
            return Ok(ops::weighted_sum(ctx, y, w));
        }
        Ok(glm(moe_ffn::moe_ffn_clamped(
            ctx,
            x,
            ids,
            w,
            refs([up, gate, down]),
            pair,
            limit,
        )))
    }

    /// The SSD expert tier's router lookahead (#611): layer `next`'s router on `x` (an earlier
    /// normed hidden state, `[n_embd, n]`) ranks the experts its step routes to, best first, and
    /// the tier takes the ranking after `after` (the trace records it, `--ssd-prefetch router`
    /// reads ahead by it). Nothing when the tier wants no prediction by `p` for this step.
    pub(crate) fn glm_ssd_predict<'a>(
        &'a self,
        pass: &Pass<'a>,
        next: usize,
        x: Tensor<'a>,
        p: Predictor,
        after: Option<Tensor<'a>>,
    ) -> Result<(), ModelError> {
        let Some(ssd) = self.ssd_experts() else {
            return Ok(());
        };
        let Some(width) = ssd.predict_width(next, p, x.ne()[1] as usize) else {
            return Ok(());
        };
        let ctx = pass.ctx;
        let logits = self.glm_lmm(ctx, next, "ffn_gate_inp.weight", x)?;
        let bias = self.lw(next, "exp_probs_b.bias")?;
        let ids = ops::router_ids(ctx, logits, bias, width as i64);
        let node = ssd
            .predict(ctx, next, ids, p, after)
            .map_err(|e| ModelError::Unsupported(format!("layer {next}: {e}")))?;
        pass.graph.borrow_mut().expand(node);
        Ok(())
    }

    /// A small step's routed experts through Legion's expert matvec (`[up, gate, down]`): the
    /// per-expert outputs (F32 `[n_embd, n_used, n]`), SwiGLU clamped at `limit`. `slots`: the
    /// tiered weights' VRAM slots (`ids` remapped), or `None` for experts all in GPU memory
    /// ([`set_expert_mv_device`]); `max`: the most tokens of a step it takes ([`mv_tokens`]).
    /// `None` when the matvec does not take the weights or the step. `ahead`: where the
    /// previous layer staged some of the step's experts (`--glm-prefetch`, [`par::prefetch`]).
    fn glm_expert_mv<'a>(
        ctx: &'a Context,
        (slots, max): (Option<usize>, i64),
        [up, gate, down]: [Tensor<'a>; 3],
        x: Tensor<'a>,
        ids: Tensor<'a>,
        limit: f32,
        ahead: Option<&expert_mv_ahead::Ahead<'a>>,
    ) -> Option<Tensor<'a>> {
        use crate::model::core::expert_mv;
        let n = x.ne()[1];
        let x = ctx.reshape_3d(x, x.ne()[0], 1, n);
        let w = [gate, up, down];
        // Every expert id is below the stack's size: all read as VRAM experts.
        let tiered = slots.is_some();
        let slots = slots.unwrap_or(gate.ne()[2] as usize);
        if tiered
            && let Some(y) =
                expert_mv::overlap::glu_down_tiered_clamped(ctx, w, x, ids, (slots, limit), max)
        {
            return Some(y);
        }
        let fits = |w, x| expert_mv::fits_glm_up_to(w, x, ids, max);
        if let Some(a) = ahead.filter(|_| tiered)
            && expert_mv::glu_fits_narrow_up_to(gate, up, x, ids, max)
        {
            let act = expert_mv_ahead::glu_clamped(ctx, [gate, up], x, ids, (slots, limit), a);
            return fits(down, act).then(|| expert_mv_ahead::down(ctx, down, act, ids, slots, a));
        }
        let act = if expert_mv::glu_fits_narrow_up_to(gate, up, x, ids, max) {
            expert_mv::glu_tiered_clamped(ctx, gate, up, x, ids, slots, limit)
        } else if fits(gate, x) && fits(up, x) {
            // Rows too wide for gate and up in one tile (Q4_K at 4096: 2 x 4 x 2304 bytes past
            // 16 KiB): a matvec each, then the clamped SwiGLU.
            let g = expert_mv::mul_mat_id_tiered(ctx, gate, x, ids, slots);
            let u = expert_mv::mul_mat_id_tiered(ctx, up, x, ids, slots);
            ops::glu(ctx, g, u, limit)
        } else {
            return None;
        };
        fits(down, act).then(|| expert_mv::mul_mat_id_tiered(ctx, down, act, ids, slots))
    }

    /// A KDA layer over the normed input `x` (`[n_embd, n]`).
    fn glm_kda<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_kda_on(pass, il, x, Part::WHOLE)
    }

    /// [`Self::glm_kda`] as `part` reads it: a GPU's share of the heads (its states' and
    /// shards'), its output projection's partial sum ([`par`]).
    pub(crate) fn glm_kda_on<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        part: Part<'_>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.glm();
        let hp = &self.hp;
        let (d, c) = (p.kda_head, hp.ssm_d_inner / part.n_dev);
        let h = c / d;
        let (s, m) = (part.split, part.mirror);
        let qkv = [
            format!("attn_q.weight{s}"),
            format!("attn_k.weight{s}"),
            format!("attn_v.weight{s}"),
        ];
        let qkv = [qkv[0].as_str(), qkv[1].as_str(), qkv[2].as_str()];
        let xh = self.glm_xh(ctx, il, &qkv, x);
        let q = self.glm_lmm_h(ctx, il, qkv[0], x, xh)?;
        let k = self.glm_lmm_h(ctx, il, qkv[1], x, xh)?;
        let v = self.glm_lmm_h(ctx, il, qkv[2], x, xh)?;
        let (dt, ssm_a) = (
            self.lw(il, &format!("ssm_dt.bias{s}"))?,
            self.lw(il, &format!("ssm_a{s}"))?,
        );
        // The gates' small products: f_a, g_a and beta of x, then f_b and g_b of those, each
        // set in one launch with the sigmoid and kda_gate after it (#606, the same bits), for
        // up to 8 tokens of Q8_0 weights; else a launch each.
        let small = ["ssm_f_a", "ssm_g_a", "ssm_beta", "ssm_f_b", "ssm_g_b"];
        let sfx = [m, m, s, s, s];
        let sw: Vec<Tensor<'a>> = small
            .iter()
            .zip(sfx)
            .map(|(n, x)| self.lw(il, &format!("{n}.weight{x}")))
            .collect::<Result<_, _>>()?;
        let set = ops::linear_set_takes(&sw, x.ne()[1..].iter().product())
            && small
                .iter()
                .all(|s| self.w(&format!("blk.{il}.{s}.scale")).is_err());
        let (g, beta, gate) = if set {
            let a = ops::linear_set(
                ctx,
                &[
                    (sw[0], x, ops::SetEpi::Plain),
                    (sw[1], x, ops::SetEpi::Plain),
                    (sw[2], x, ops::SetEpi::Sigmoid),
                ],
            );
            let b = ops::linear_set(
                ctx,
                &[
                    (sw[3], a[0], ops::SetEpi::KdaGate(dt, ssm_a, p.kda_lower)),
                    (sw[4], a[1], ops::SetEpi::Plain),
                ],
            );
            (b[0], a[2], Some(b[1]))
        } else {
            let z = self.glm_lmm(ctx, il, &format!("ssm_f_a.weight{m}"), x)?;
            let z = self.glm_lmm(ctx, il, &format!("ssm_f_b.weight{s}"), z)?;
            let g = ops::kda_gate(ctx, z, dt, ssm_a, p.kda_lower);
            let beta = ops::sigmoid(
                ctx,
                self.glm_lmm(ctx, il, &format!("ssm_beta.weight{s}"), x)?,
            );
            (g, beta, None)
        };
        tap.tap("kda_g1", il, g);
        tap.tap("kda_beta", il, beta);
        let conv_w = |w: &str| self.lw(il, &format!("ssm_conv1d_{w}.weight{s}"));
        let (wq, wk, wv) = (conv_w("q")?, conv_w("k")?, conv_w("v")?);

        let mut outs = Vec::new();
        for (off, sh) in pass.groups() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let ng = nt * ns;
            // A verify group stores its inputs for the commit and leaves the states.
            let store = if sh.verify {
                let spec = pass.spec.ok_or_else(|| {
                    ModelError::Unsupported("a verify step without replay buffers".into())
                })?;
                Some(spec.kda_views(ctx, il, sh.slot, nt, ns))
            } else {
                None
            };
            let state = pass.rec.conv_slots(ctx, il, sh.slot, sh.n_seqs);
            let conv = |t: Tensor<'a>, w: Tensor<'a>, ch: i64, i: usize| {
                let x = cols(ctx, t, off, ng);
                let y = match store {
                    Some(st) => ops::kda_conv_verify(ctx, x, w, state, ch, nt, st[i]),
                    None => ops::kda_conv(ctx, x, w, state, ch, nt),
                };
                ctx.reshape_3d(y, d, h, ng)
            };
            let (qc, kc, vc) = (conv(q, wq, 0, 0), conv(k, wk, c, 1), conv(v, wv, 2 * c, 2));
            let gg = ctx.reshape_3d(cols(ctx, g, off, ng), d, h, ng);
            let bg = cols(ctx, beta, off, ng);
            let ssm = pass.rec.ssm_slots(ctx, il, sh.slot, sh.n_seqs);
            outs.push(match store {
                Some(st) => ops::kda_scan_verify(ctx, qc, kc, vc, gg, bg, ssm, nt, [st[3], st[4]]),
                None => ops::kda_scan(ctx, qc, kc, vc, gg, bg, ssm, nt),
            });
        }
        let o = ops::join_cols(ctx, &outs);
        let o = ctx.reshape_2d(o, d, h * pass.n);
        tap.tap("kda_scan_out", il, o);
        let norm = self.lw(il, &format!("ssm_norm.weight{m}"))?;
        if tap.on {
            tap.tap(
                "kda_normed",
                il,
                ops::gated_rms(ctx, o, norm, None, hp.rms_eps),
            );
        }
        let gate = match gate {
            Some(gate) => gate,
            None => {
                let gate = self.glm_lmm(ctx, il, &format!("ssm_g_a.weight{m}"), x)?;
                self.glm_lmm(ctx, il, &format!("ssm_g_b.weight{s}"), gate)?
            }
        };
        let gate = ctx.reshape_2d(gate, d, h * pass.n);
        let o = ops::gated_rms(ctx, o, norm, Some(gate), hp.rms_eps);
        let out = self.glm_lmm(
            ctx,
            il,
            &format!("attn_output.weight{s}"),
            ctx.reshape_2d(o, c, pass.n),
        )?;
        tap.tap("kda_out", il, out);
        Ok(out)
    }

    /// [`Self::glm_dsa`]'s cache work alone (an MTP take-in that reads no output): the latents
    /// and indexer rows of this step's tokens cached, the pool keys of the pools they complete.
    fn glm_dsa_write<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
    ) -> Result<(), ModelError> {
        let ctx = pass.ctx;
        let p = self.glm();
        let hp = &self.hp;
        let di = p.idx_dim;
        let rms = |t, name: &str| -> Result<Tensor<'a>, ModelError> {
            Ok(ops::rms_norm(ctx, t, Some(self.lw(il, name)?), hp.rms_eps))
        };
        let ik = self.glm_lmm(ctx, il, "indexer.attn_k.weight", x)?;
        let ik = ops::layer_norm(
            ctx,
            ik,
            self.lw(il, "indexer.k_norm.weight")?,
            self.lw(il, "indexer.k_norm.bias")?,
            p.norm_eps,
        );
        let ig = self.glm_lmm(ctx, il, "indexer_compressor_gate.weight", x)?;
        let kv = rms(
            self.glm_lmm(ctx, il, "attn_kv_a_mqa.weight", x)?,
            "attn_kv_a_norm.weight",
        )?;
        let cache = pass.kv.pools(il).0;
        let idx = pass.kv.idx_cache(il);
        let cells = pass.write_cells(il);
        let latent = ops::cache_rows(ctx, cache, kv, cells, 0);
        let wk = ops::cache_rows(ctx, idx, ik, cells, 0);
        let wg = ops::cache_rows(ctx, idx, ig, cells, di);
        let ape = self.lw(il, "indexer_compressor_ape.weight")?;
        let mut graph = pass.graph.borrow_mut();
        for t in [latent, wk, wg] {
            graph.expand(t);
        }
        for g in &pass.group {
            let inp = QsaInputs::from_slice(g);
            graph.expand(ops::pool_keys(
                ctx,
                idx,
                inp.new_idxs,
                inp.new_rep,
                ape,
                &[wk, wg],
            ));
        }
        Ok(())
    }

    /// A DSA layer over the normed input `x` (`[n_embd, n]`): the latents and indexer rows
    /// of this step's tokens cached, then per group the indexer's selection and the sparse
    /// MLA attention over it.
    fn glm_dsa<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.glm_dsa_on(pass, il, x, Part::WHOLE)
    }

    /// [`Self::glm_dsa`] as `part` reads it: every GPU caches every latent and indexer row and
    /// selects the same positions; a GPU's share of the heads attends, its output projection
    /// giving a partial sum ([`par`]).
    pub(crate) fn glm_dsa_on<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        x: Tensor<'a>,
        part: Part<'_>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let p = self.glm();
        let hp = &self.hp;
        let n = pass.n;
        if pass.kv.types().k.ggml() != types::F16 {
            return Err(ModelError::Unsupported(
                "GLM's MLA latents in a cache type other than F16".into(),
            ));
        }
        let (dl, dh, nh) = (p.kv_lora, p.head_mla, hp.n_head / part.n_dev);
        let (di, ih) = (p.idx_dim, p.idx_heads);
        let (s, m) = (part.split, part.mirror);
        let rms = |t, name: &str| -> Result<Tensor<'a>, ModelError> {
            Ok(ops::rms_norm(
                ctx,
                t,
                Some(self.lw(il, &format!("{name}{m}"))?),
                hp.rms_eps,
            ))
        };
        // A mirrored weight's product / a split one's.
        let mm = |name: &str, x| self.glm_lmm(ctx, il, &format!("{name}{m}"), x);
        let sm = |name: &str, x| self.glm_lmm(ctx, il, &format!("{name}{s}"), x);

        // The indexer's key and gate rows of this step's tokens.
        let ik = mm("indexer.attn_k.weight", x)?;
        let ik = ops::layer_norm(
            ctx,
            ik,
            self.lw(il, &format!("indexer.k_norm.weight{m}"))?,
            self.lw(il, &format!("indexer.k_norm.bias{m}"))?,
            p.norm_eps,
        );
        tap.tap("indexer_k", il, ik);
        let ig = mm("indexer_compressor_gate.weight", x)?;
        tap.tap("indexer_gate", il, ig);

        // Queries, absorbed into the latent space: [kv_lora, n_head, n].
        let qr = rms(mm("attn_q_a.weight", x)?, "attn_q_a_norm.weight")?;
        tap.tap("q_resid", il, qr);
        // At GB10 prompt steps attn_q_b gives attn_k_b its rows in F16 (#862).
        let lw = |name: &str| self.lw(il, &format!("{name}{s}"));
        let q = if glm_f16_chain(lw("attn_q_b.weight")?, lw("attn_k_b.weight")?, n) {
            self.glm_mm_f16(ctx, &format!("blk.{il}.attn_q_b.weight{s}"), qr)?
        } else {
            sm("attn_q_b.weight", qr)?
        };
        let q = ctx.reshape_3d(q, dh, nh, n);
        let q = sm("attn_k_b.weight", q)?;
        // The latents of this step's tokens.
        let kv = rms(mm("attn_kv_a_mqa.weight", x)?, "attn_kv_a_norm.weight")?;

        // The indexer's queries and head weights.
        let iq = mm("indexer.attn_q_b.weight", qr)?;
        let iq = ctx.reshape_3d(iq, di, ih, n);
        let iw = ops::linear(
            ctx,
            self.lw(il, &format!("indexer.proj.weight{m}"))?,
            x,
            1.0 / ((di * ih) as f32).sqrt(),
            None,
        );
        tap.tap("indexer_weights", il, iw);

        let cache = pass.kv.pools(il).0;
        let idx = pass.kv.idx_cache(il);
        let cells = pass.write_cells(il);
        let latent = ops::cache_rows(ctx, cache, kv, cells, 0);
        let wk = ops::cache_rows(ctx, idx, ik, cells, 0);
        let wg = ops::cache_rows(ctx, idx, ig, cells, di);
        {
            let mut graph = pass.graph.borrow_mut();
            for t in [latent, wk, wg] {
                graph.expand(t);
            }
        }
        let ape = self.lw(il, &format!("indexer_compressor_ape.weight{m}"))?;
        let pooled = ctx.view_2d(
            idx,
            [di, idx.ne()[1]],
            idx.nb()[1],
            2 * di as usize * idx.element_size(),
        );
        let es = pass.pos.element_size();
        let mut outs = Vec::new();
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let inp = QsaInputs::from_slice(&pass.group[gi]);
            let ng = sh.tokens() as i64;
            let AttnInputs::Paged { table, .. } = pass.attn[gi] else {
                return Err(ModelError::Unsupported(
                    "GLM's sparse MLA on the gather path".into(),
                ));
            };
            // The pool keys of the pools this step completes, cached before anything scores.
            let pk = ops::pool_keys(ctx, idx, inp.new_idxs, inp.new_rep, ape, &[wk, wg]);
            pass.graph.borrow_mut().expand(pk);
            let pos = ctx.view_2d(pass.pos, [ng, 1], ng as usize * es, off as usize * es);
            let n_top = (sh.n_pool as i64).min((p.top_k / p.kpool) as i64);
            let sel = if sh.pick_all && !tap.on {
                qsa_attn::select_all(ctx, inp.pool_vis, n_top, inp.pool_idxs, inp.tail_idxs, pos)
            } else {
                let qnb = iq.nb();
                let qg = ctx.view_3d(iq, [di, ih, ng], [qnb[1], qnb[2]], off as usize * qnb[2]);
                let score = qsa_attn::scores(
                    ctx,
                    qg,
                    pooled,
                    cols(ctx, iw, off, ng),
                    inp.pool_vis,
                    Some(inp.pool_bias),
                    Some(inp.pool_cells),
                );
                tap.tap("indexer_score", il, score);
                let top = qsa_attn::top_k(ctx, score, n_top);
                qsa_attn::select(ctx, top, score, inp.pool_idxs, inp.tail_idxs, pos)
            };
            let qnb = q.nb();
            let qg = ctx.view_3d(q, [dl, nh, ng], [qnb[1], qnb[2]], off as usize * qnb[2]);
            let scale = 1.0 / (dh as f32).sqrt();
            outs.push(ops::mla_attn(
                ctx,
                qg,
                cache,
                table,
                sel,
                sh.n_tok as i64,
                scale,
                latent,
            ));
        }
        let o = ctx.reshape_3d(ops::join_cols(ctx, &outs), dl, nh, n);
        // Back out of the latent space per head: [head_mla, n_head, n]; at GB10 prompt steps in
        // F16, the rows attn_output would round o to (#862).
        let o = if glm_f16_chain(lw("attn_v_b.weight")?, lw("attn_output.weight")?, n) {
            self.glm_mm_f16(ctx, &format!("blk.{il}.attn_v_b.weight{s}"), o)?
        } else {
            sm("attn_v_b.weight", o)?
        };
        let o = ctx.reshape_2d(o, dh * nh, n);
        let out = sm("attn_output.weight", o)?;
        tap.tap("attn_out", il, out);
        Ok(out)
    }
}
