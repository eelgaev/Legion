//! `qwen4exp` (Qwen3.8-Flash-Next, "a preview of the Qwen4 architecture"): the qwen35moe layer
//! stack (gated DeltaNet every layer but each `full_attention_interval`th, gated attention with
//! multi-section RoPE, 512-expert MoE plus a gated shared expert) with three changes, following
//! llama.cpp's `src/models/qwen4exp.cpp` (MIT) op for op:
//!
//! - **Hyper-connections** replace the residual stream and the pre-norms: the residual is `hc`
//!   parallel copies `[n_embd, hc, n]`. Before each block a low-rank gate mixes the RMS-normed
//!   streams into the block input ([`Model::hc_mix`]); the block output is added back to every
//!   stream with a learned per-stream weight ([`Model::hc_combine`]). The last mixer is the
//!   output norm.
//! - **Per-layer n-gram embedding (PLE)** in one layer: each token's bigram and trigram (with its
//!   predecessors, reset at EOS) are hashed into rows of a huge shared table (320M rows × 160 in
//!   the released model, 29 GB at IQ4_NL). The rows are gathered on the host straight from the
//!   mapped file ([`Model::fill_ple`]); a gate against the residual and a causal conv dilated
//!   by the n-gram size (state: the last `(kernel - 1) × n-gram` inputs) add them to the streams.
//! - The DeltaNet output gate is `sigmoid(z)`, not `silu(z)`.
//!
//! - **Sparse attention (QSA)** in the full-attention layers: a token attends to the keys of
//!   the `indexer.top_k / compress_ratio` best-scoring complete pools of `compress_ratio`
//!   consecutive positions, plus its own incomplete pool. A pool is scored by one key, the
//!   normed and rotated mean of its members' raw indexer keys, against the token's indexer query
//!   heads ([`Model::qsa_mask`]). Raw and pool keys are cached per cell next to K/V (see
//!   `KvPool`); a pool's key is computed once, in the step that completes it. Up to `top_k`
//!   tokens every visible key is selected, so the result equals dense attention.

pub mod hc;
mod hc_chain;
pub mod hc_down;
pub use hc_chain::set_hc_chain;
mod hc_mix16;
pub use hc_mix16::{set_hc_pre, set_mix16, set_rebuild_bf16 as set_mix_rebuild_bf16};
mod hc_norm;
pub use hc_norm::lazy_bench as hc_lazy_bench;
pub(crate) use hc_norm::{
    FoldSum, fold_moe, fold_target, lazy_fold_target, lazy_parts, lazy_streams, moe_fold,
    set_lazy_fold, set_no_res,
};
pub use hc_norm::{
    hc_fuse, set_fused_post_norm, set_generic_post_norm, set_hc_bf16, set_hc_fuse,
    set_hc_lazy_streams, set_moe_fold, set_norm_fma, set_streams_bf16, set_xn16, set_xn16_gpu,
};
mod hc_up_mix;
pub use hc_up_mix::bench as hc_up_mix_bench;
#[cfg(feature = "cuda")]
pub(crate) use hc_up_mix::launch_mix;
pub(crate) use hc_up_mix::{
    lazy_mix_target, mix_cast_reads, set_by_exchange as set_mix_by_exchange, set_lazy_streams,
    set_mix_cast,
};
pub use hc_up_mix::{
    set_gpu as set_up_mix_gpu, set_rebuild as set_up_mix_rebuild, set_up_mix, set_up_mix_bf16,
};
mod mtp;
mod ple_ahead;
pub(crate) use ple_ahead::PleAhead;
mod ple_fused;
pub use ple_fused::set_fused_ple;
mod ple_late;
pub(crate) use ple_late::LateRows;
pub use ple_late::set_ple_early_reads;
pub(crate) mod qsa_attn;
pub use qsa_attn::PromptKernel;
#[cfg(feature = "cuda")]
pub use qsa_attn::bench as qsa_bench;
#[cfg(test)]
mod tests;
pub mod tiny;
mod tp_overlap;

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ggml::{Context, Graph, Tensor, types};
use crate::gguf::GgufFile;
use crate::model::core::arch::same_as;
use crate::model::core::mtp::MtpGraph;
use crate::model::core::pad_rows;
use crate::model::core::state::{KvPool, QSA_SCRATCH};
use crate::model::core::{
    Arch, ArchRef, Attention, Group, GroupShape, Hparams, InputSink, Inputs, Model, ModelError,
    Pass, SpecPool, StatePool, TokenConv, cols, concat_cols, mmq_preq, norm_rope,
};
use crate::model::dense_mv;
use crate::model::q8_mmq;

/// Threads gathering a step's per-layer embedding rows.
const PLE_GATHER_THREADS: usize = 8;

/// Rows per gather thread when they are read from the weights' own memory: page faults, which
/// several threads wait on at once.
const PLE_ROWS_PER_THREAD_MAPPED: usize = 64;

/// Rows per gather thread when the n-gram store already read them into RAM: dequantizing only
/// (~0.1 us a row). A decode step's rows (16 a token) stay on the engine thread: on the GB10 the
/// four threads of a 16-sequence step took 2-3.5 ms to start, map their malloc arenas and join,
/// with the GPU waiting for the rows (#823).
const PLE_ROWS_PER_THREAD_FETCHED: usize = 4096;

/// Threads [`Model::gather_ple`] dequantizes `rows` table rows on: one for a few, at most
/// [`PLE_GATHER_THREADS`]; `fetched` when the n-gram store read the rows (into RAM).
fn ple_gather_threads(rows: usize, fetched: bool) -> usize {
    let per = if fetched {
        PLE_ROWS_PER_THREAD_FETCHED
    } else {
        PLE_ROWS_PER_THREAD_MAPPED
    };
    (rows / per).clamp(1, PLE_GATHER_THREADS)
}

/// Hyper-connection shape.
#[derive(Debug, Clone, PartialEq)]
pub struct HcParams {
    /// Residual streams.
    pub n: i64,
    /// Rank of the gate projection.
    pub low_rank: i64,
}

/// The per-layer n-gram embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct PleParams {
    /// The layer it is added in (a DeltaNet layer).
    pub layer: usize,
    /// Longest n-gram hashed (bigrams up to this).
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: i64,
    /// Resets the n-gram window (and stands in for missing predecessors).
    pub eos: u32,
    /// Width of one table row.
    pub head_dim: i64,
    /// Hash multiplier per n-gram position (`ngram` of them).
    pub multipliers: Vec<u64>,
    /// Per head: first table row and number of rows (`n_heads()` of each).
    pub offsets: Vec<u32>,
    pub vocab_sizes: Vec<u32>,
    /// Where the table's row blocks are in the file, when not in row order (see
    /// [`PleBlocks`]); `None` for GGUF tables.
    pub blocks: Option<PleBlocks>,
    /// The table holds FP8 E4M3 bytes (an `I8` tensor) with this per-tensor scale, as a
    /// ModelOpt checkpoint ships it; `None` for ggml types.
    pub fp8_scale: Option<f32>,
}

/// Metadata keys of a table read from a safetensors checkpoint ([`crate::hf`]), which stores
/// it as shards of `block_rows` rows each, back to back but not necessarily in index order:
/// `block_order[i]` is the position in the file of the rows `i * block_rows ..`.
pub const KEY_PLE_BLOCK_ROWS: &str = "legion.ple.block_rows";
pub const KEY_PLE_BLOCK_ORDER: &str = "legion.ple.block_order";
/// The per-tensor scale of an FP8 E4M3 table (F32).
pub const KEY_PLE_FP8_SCALE: &str = "legion.ple.fp8_scale";

/// The table's rows in blocks stored out of order ([`KEY_PLE_BLOCK_ORDER`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PleBlocks {
    pub rows: u64,
    /// Per block in row order, its position in the file.
    pub position: Vec<u32>,
}

impl PleBlocks {
    /// The stored row of logical row `r`.
    pub fn physical(&self, r: u64) -> u64 {
        let (b, i) = (r / self.rows, r % self.rows);
        self.position[b as usize] as u64 * self.rows + i
    }
}

/// One table row of `ty` (`dim` elements, `src` its bytes) as f32: E4M3 codes times
/// `fp8_scale` when the table is FP8, else ggml's dequantization.
pub fn ple_row_f32(
    ty: crate::ggml::ffi::ggml_type,
    fp8_scale: Option<f32>,
    src: &[u8],
    dim: usize,
) -> Vec<f32> {
    match fp8_scale {
        Some(s) if ty == types::I8 => src.iter().map(|&b| E4M3[b as usize] * s).collect(),
        _ => crate::ggml::dequantize(ty, src, dim),
    }
}

/// Every E4M3 code's value.
static E4M3: std::sync::LazyLock<[f32; 256]> =
    std::sync::LazyLock::new(|| std::array::from_fn(|b| crate::hf::nvfp4::e4m3_to_f32(b as u8)));

impl PleParams {
    /// Table rows gathered per token.
    pub fn n_heads(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }

    /// Conv history in tokens: `(kernel - 1)` taps dilated by the n-gram size.
    pub fn history(&self) -> i64 {
        (self.conv_kernel - 1) * self.ngram as i64
    }

    /// Appends the (stored, see [`PleBlocks`]) table rows of `token` given its predecessors `prev` (oldest first,
    /// `ngram - 1` of them, [`crate::model::core::state::NO_TOKEN`] before the sequence start), as llama.cpp's
    /// `llm_graph_input_qwen4exp_ple::set_input` hashes them: an EOS (or a missing token) cuts
    /// the window, and every position at or before the cut reads as EOS.
    pub fn rows(&self, token: u32, prev: &[u32], out: &mut Vec<i32>) {
        let n_prev = self.ngram - 1;
        debug_assert_eq!(prev.len(), n_prev);
        let mut ctx = vec![token as u64; self.ngram];
        let mut cut = false;
        for s in 1..self.ngram {
            let t = prev[n_prev - s];
            cut = cut || t == crate::model::core::state::NO_TOKEN || t == self.eos;
            ctx[s] = if cut { self.eos } else { t } as u64;
        }
        for n in 2..=self.ngram {
            // t[0]·m[0] ^ t[1]·m[1] ^ ... over the n-gram, products wrapping as in C.
            let mixed = ctx[..n]
                .iter()
                .zip(&self.multipliers)
                .map(|(&t, &m)| t.wrapping_mul(m))
                .fold(0, |acc, x| acc ^ x);
            let base = (n - 2) * self.heads_per_ngram;
            for h in base..base + self.heads_per_ngram {
                let r = mixed % self.vocab_sizes[h] as u64 + self.offsets[h] as u64;
                let r = match &self.blocks {
                    Some(b) => b.physical(r),
                    None => r,
                };
                out.push(r as i32);
            }
        }
    }
}

/// The sparse-attention indexer.
#[derive(Debug, Clone, PartialEq)]
pub struct QsaParams {
    pub n_head: i64,
    /// Indexer key (and query head) size.
    pub head_dim: i64,
    /// Keys a token may attend to (pools × pool size).
    pub top_k: usize,
    /// Pool size (the layers' compress ratio).
    pub kpool: usize,
    /// Per layer: its compress ratio, 0 for dense attention.
    pub ratios: Vec<u32>,
}

impl QsaParams {
    /// Whether layer `il` attends sparsely.
    pub fn layer(&self, il: usize) -> bool {
        self.ratios.get(il).is_some_and(|&r| r > 0)
    }
}

/// qwen4exp's own hyperparameters, on top of the Qwen3-Next stack's in [`Hparams`].
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4Params {
    /// Hyper-connection residual streams (and the sigmoid DeltaNet output gate with them).
    pub hc: HcParams,
    /// The per-layer n-gram embedding.
    pub ple: Option<PleParams>,
    /// Sparse attention (QSA) in the full-attention layers.
    pub qsa: Option<QsaParams>,
}

/// The qwen4exp hyperparameters of `hp`, if it is a qwen4exp model.
pub fn params(hp: &Hparams) -> Option<&Qwen4Params> {
    hp.arch.as_any().downcast_ref::<Qwen4Params>()
}

/// The hyperparameters of a `qwen4exp` GGUF: the Qwen3-Next MoE stack's plus
/// [`Qwen4Params`].
pub(crate) fn parse(g: &GgufFile, arch: &str) -> Result<Hparams, ModelError> {
    crate::model::qwen35::parse_next(g, arch, true, || Ok(ArchRef::new(parse_params(g, arch)?)))
}

/// Reads qwen4exp's hyper-connection, PLE and sparse-attention metadata.
fn parse_params(g: &GgufFile, arch: &str) -> Result<Qwen4Params, ModelError> {
    let key = |k: &str| format!("{arch}.{k}");
    let missing = |k: &str| ModelError::Metadata(key(k));
    let u = |k: &str| g.get_u64(&key(k)).ok_or_else(|| missing(k));
    let arr = |k: &str| -> Result<Vec<u64>, ModelError> {
        g.get(&key(k))
            .and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().map(|x| x.as_u64()).collect::<Option<Vec<_>>>())
            })
            .flatten()
            .ok_or_else(|| missing(k))
    };
    let hc = HcParams {
        n: u("hyper_connection.count")? as i64,
        low_rank: u("hyper_connection.low_rank")? as i64,
    };
    if hc.n <= 1 || hc.low_rank == 0 {
        return Err(missing("hyper_connection.count"));
    }
    let ple = match g.get(&key("ple.layers")) {
        None => None,
        Some(_) => {
            let layers = arr("ple.layers")?;
            // llama.cpp supports exactly one PLE layer too (one set of hash constants).
            let [layer] = layers[..] else {
                return Err(missing("ple.layers"));
            };
            let ngram = u("ple.ngram_size")? as usize;
            let heads_per_ngram = u("ple.heads_per_ngram")? as usize;
            let n_heads = (ngram.max(1) - 1) * heads_per_ngram;
            let multipliers = arr("ple.layer_multipliers")?;
            let offsets = arr("ple.head_offsets")?;
            let sizes = arr("ple.head_vocab_sizes")?;
            let fits = |v: u64| u32::try_from(v).ok().filter(|&v| v <= i32::MAX as u32);
            let offsets: Option<Vec<u32>> =
                offsets.iter().take(n_heads).map(|&v| fits(v)).collect();
            let sizes: Option<Vec<u32>> = sizes.iter().take(n_heads).map(|&v| fits(v)).collect();
            let p = PleParams {
                layer: layer as usize,
                ngram,
                heads_per_ngram,
                conv_kernel: u("ple.conv_kernel")? as i64,
                eos: u("ple.eos_token_id")? as u32,
                head_dim: u("embedding_length_per_layer_input")? as i64,
                multipliers,
                offsets: offsets.ok_or_else(|| missing("ple.head_offsets"))?,
                vocab_sizes: sizes.ok_or_else(|| missing("ple.head_vocab_sizes"))?,
                blocks: ple_blocks(g)?,
                fp8_scale: match g.get(KEY_PLE_FP8_SCALE) {
                    None => None,
                    Some(v) => Some(
                        v.as_f64()
                            .ok_or_else(|| ModelError::Metadata(KEY_PLE_FP8_SCALE.into()))?
                            as f32,
                    ),
                },
            };
            if ngram < 2
                || n_heads == 0
                || p.conv_kernel < 1
                || p.multipliers.len() < ngram
                || p.offsets.len() < n_heads
                || p.vocab_sizes.len() < n_heads
                || p.vocab_sizes.contains(&0)
            {
                return Err(missing("ple.*"));
            }
            // Every row a head hashes to is in some block.
            let end = (0..n_heads)
                .map(|h| p.offsets[h] as u64 + p.vocab_sizes[h] as u64)
                .max()
                .unwrap_or(0);
            if p.blocks
                .as_ref()
                .is_some_and(|b| (b.position.len() as u64) * b.rows < end)
            {
                return Err(ModelError::Metadata(KEY_PLE_BLOCK_ORDER.into()));
            }
            Some(p)
        }
    };
    let qsa = match g.get(&key("attention.indexer.top_k")) {
        None => None,
        Some(_) => {
            let ratios: Vec<u32> = match g.get(&key("attention.compress_ratios")) {
                Some(_) => arr("attention.compress_ratios")?
                    .into_iter()
                    .map(|r| r as u32)
                    .collect(),
                None => Vec::new(),
            };
            // One pool size for the whole model, as llama.cpp requires.
            let mut kpool = 0;
            for &r in ratios.iter().filter(|&&r| r > 0) {
                if kpool != 0 && r != kpool {
                    return Err(missing("attention.compress_ratios"));
                }
                kpool = r;
            }
            let q = QsaParams {
                n_head: u("attention.indexer.head_count")? as i64,
                head_dim: u("attention.indexer.key_length")? as i64,
                top_k: u("attention.indexer.top_k")? as usize,
                kpool: kpool as usize,
                ratios,
            };
            if q.n_head == 0 || q.head_dim == 0 || q.top_k == 0 {
                return Err(missing("attention.indexer.*"));
            }
            if q.kpool == 1 || (q.kpool > 0 && !q.top_k.is_multiple_of(q.kpool)) {
                return Err(missing("attention.compress_ratios"));
            }
            (q.kpool > 0).then_some(q)
        }
    };
    Ok(Qwen4Params { hc, ple, qsa })
}

/// The table's block order ([`KEY_PLE_BLOCK_ORDER`]), `None` when absent or the identity.
fn ple_blocks(g: &GgufFile) -> Result<Option<PleBlocks>, ModelError> {
    let (Some(rows), Some(order)) = (g.get(KEY_PLE_BLOCK_ROWS), g.get(KEY_PLE_BLOCK_ORDER)) else {
        return Ok(None);
    };
    let bad = || ModelError::Metadata(KEY_PLE_BLOCK_ORDER.into());
    let rows = rows.as_u64().filter(|&r| r > 0).ok_or_else(bad)?;
    let position: Vec<u32> = order
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|v| v.as_u64().and_then(|p| u32::try_from(p).ok()))
        .collect::<Option<_>>()
        .ok_or_else(bad)?;
    // A permutation of the blocks.
    let mut seen = vec![false; position.len()];
    for &p in &position {
        let s = seen.get_mut(p as usize).ok_or_else(bad)?;
        if std::mem::replace(s, true) {
            return Err(bad());
        }
    }
    let identity = position.iter().enumerate().all(|(i, &p)| i == p as usize);
    Ok((!identity).then_some(PleBlocks { rows, position }))
}

/// Tokens per sequence from which a group counts as a prompt chunk for [`pools_listed`].
const PROMPT_POOL_TOKENS: usize = 512;
/// Prompt chunks pad their listed pools to a multiple of a quarter of them (a power of two, at
/// most this): consecutive chunks of a long prompt then share one graph shape, so its graph is
/// reused rather than rebuilt (and its compute buffer reallocated a few KB bigger) every chunk,
/// at the price of scoring up to that many masked pools per query.
const PROMPT_POOL_PAD: usize = 4096;

/// Pools a group lists: every complete pool of its sequences, padded (as llama.cpp pads them,
/// to at least one spare entry and a multiple of 64; prompt chunks coarser, see
/// [`PROMPT_POOL_PAD`]).
pub(crate) fn pools_listed(g: &Group<'_>, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    let real: usize = g
        .seqs
        .iter()
        .map(|s| (s.n_past + s.tokens.len()) / kpool)
        .sum();
    let prompt = g.seqs.iter().any(|s| s.tokens.len() >= PROMPT_POOL_TOKENS);
    let pad = if prompt {
        (real / 4).next_power_of_two().clamp(64, PROMPT_POOL_PAD)
    } else {
        64
    };
    (real + 1).div_ceil(pad).max(1) * pad
}

/// `legion serve --qsa-pool-pad-steps` (#858; 0, the default: off): decode and verify steps of
/// several sequences pad their listed pools to cover at least this many steps' growth. A step
/// of `t` tokens completes up to `t / kpool` pools, so at 64 pools a 16-sequence verify step
/// (48 tokens) changes shape every ~5 steps, and every change rebuilds the stages' graphs and
/// captures and instantiates their CUDA graphs again.
static STEP_POOL_STEPS: AtomicUsize = AtomicUsize::new(0);

/// Sets [`STEP_POOL_STEPS`] for graphs built from now on (0: off).
pub fn set_step_pool_steps(steps: usize) {
    STEP_POOL_STEPS.store(steps, Ordering::Relaxed);
}

/// qwen4exp's pools a group lists: [`pools_listed`], except that with `steps` > 0 a decode or
/// verify step (a verify group, or at most `kpool` tokens a sequence) pads to a power of two of at least `steps`
/// steps' new pools (at most [`PROMPT_POOL_PAD`]). One sequence keeps 64 at up to 64 steps (a
/// verify step of up to 4 tokens: 64 × 4 / 4 pools), so single-user decode lists what it did.
/// The padding is masked, so only the shape moves, not the bits: each pool's score depends on
/// its own key alone, the extra pools score -inf at the end of the listing, the selection keeps
/// the finite-scored picks only (sorted by position, ties at `top_k`'s threshold going to the
/// lowest index), and the attention splits a token's selection by its count, not by the row's
/// capacity. The price is scoring the extra pools.
pub(crate) fn step_pools_listed(g: &Group<'_>, kpool: usize, steps: usize) -> usize {
    let n = pools_listed(g, kpool);
    let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
    if kpool == 0 || steps == 0 || (n_tok > kpool && !g.verify) {
        return n;
    }
    let pad = (steps * g.seqs.len() * n_tok / kpool)
        .next_power_of_two()
        .clamp(64, PROMPT_POOL_PAD);
    if pad <= 64 {
        return n;
    }
    let real: usize = g
        .seqs
        .iter()
        .map(|s| (s.n_past + s.tokens.len()) / kpool)
        .sum();
    (real + 1).div_ceil(pad).max(1) * pad
}

/// Pool keys a group computes: the pools its tokens complete, padded to a bound that only
/// depends on the group's shape for decode and verify steps (`ceil(n_tok / kpool)` per
/// sequence), so a kept graph fits the next steps too.
pub(crate) fn pools_new(g: &Group<'_>, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
    let real: usize = g
        .seqs
        .iter()
        .map(|s| (s.n_past + n_tok) / kpool - s.n_past / kpool)
        .sum();
    let bound = g.seqs.len() * n_tok.div_ceil(kpool);
    if n_tok <= kpool && bound < QSA_SCRATCH {
        bound
    } else {
        real.max(1)
    }
}

/// qwen4exp's pool keys a group computes: [`pools_new`], except that a verify group pads to the
/// same shape-only bound (`ceil(n_tok / kpool)` per sequence) when it has more than `kpool`
/// tokens a sequence too (#836). A 16-sequence MTP verify step of 9 rows completes 32-41 pools,
/// a different count almost every step, and each count was a new kept graph: built again
/// (~5 ms of host time before its first kernel on the GB10) and run without its CUDA graphs.
/// The padding goes to scratch rows of its own and each pool key is computed from its own
/// members alone (one block per key in `pool_key`, row-wise ops in the chain), so only the
/// shape moves, not the bits.
pub(crate) fn step_pools_new(g: &Group<'_>, kpool: usize) -> usize {
    let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
    if kpool == 0 || !g.verify || n_tok <= kpool {
        return pools_new(g, kpool);
    }
    let bound = g.seqs.len() * n_tok.div_ceil(kpool);
    if bound < QSA_SCRATCH {
        bound
    } else {
        pools_new(g, kpool)
    }
}

/// The per-layer embedding's conv history at a prompt group's captures
/// ([`crate::model::core::capture`]): the `hist` inputs of `x` (`[width, n_tok, ..]`) before
/// each, into its saved slot.
fn ple_captures<'a>(
    pass: &Pass<'a>,
    il: usize,
    sh: &crate::model::core::GroupShape,
    x: Tensor<'a>,
    hist: i64,
) {
    for c in &sh.captures {
        let saved = pass
            .saved
            .expect("a step with captures has the saved states");
        let to = saved.ple_slots(pass.ctx, il, c.saved, 1);
        crate::model::core::capture::history(pass.ctx, &pass.graph, x, c.at, hist, to);
    }
}

/// The history the next step starts from, out of the token-major window `padded` of a
/// one-token step: its last `hist` positions, back to `state`.
fn ple_keep_history<'a>(
    ctx: &'a Context,
    graph: &std::cell::RefCell<Graph<'a>>,
    padded: Tensor<'a>,
    state: Tensor<'a>,
    nt: i64,
    ple: &PleParams,
) {
    let [_, width, ns, _] = padded.ne();
    let pnb = padded.nb();
    let kept = ctx.view_3d(
        padded,
        [ple.history(), width, ns],
        [pnb[1], pnb[2]],
        nt as usize * padded.element_size(),
    );
    graph.borrow_mut().expand(ctx.cpy(kept, state));
}

/// The per-layer embedding's causal conv over a token-major window `padded` (`[hist + n_tok,
/// width, n_seqs]`, the history then the step's tokens), as verify groups and one-token steps
/// take it: `[width, n_tok * n_seqs]`.
fn ple_conv_time_major<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    padded: Tensor<'a>,
    nt: i64,
    ple: &PleParams,
) -> Tensor<'a> {
    let (kern, dil, hist) = (ple.conv_kernel, ple.ngram as i64, ple.history());
    let [_, width, ns, _] = padded.ne();
    let pnb = padded.nb();
    let es = padded.element_size();
    let window = |start: i64, len: i64| {
        ctx.view_3d(
            padded,
            [len, width, ns],
            [pnb[1], pnb[2]],
            start as usize * es,
        )
    };
    let mut conv: Option<Tensor<'a>> = None;
    for k in 0..kern {
        let start = hist - (kern - 1 - k) * dil;
        let shifted = ctx.cont(ctx.transpose(window(start, nt)));
        let term = ctx.mul(shifted, ple_conv_tap(ctx, w, k, width));
        conv = Some(match conv {
            Some(c) => ctx.add(c, term),
            None => term,
        });
    }
    ctx.reshape_2d(conv.expect("kernel size >= 1"), width, nt * ns)
}

/// Tap `k` of the per-layer embedding's depthwise conv weight `w` as an F32 row of `width`.
fn ple_conv_tap<'a>(ctx: &'a Context, w: Tensor<'a>, k: i64, width: i64) -> Tensor<'a> {
    let wk = ctx.view_2d(w, [1, width], w.nb()[1], k as usize * w.nb()[0]);
    let wk = ctx.cont_2d(wk, width, 1);
    if wk.ty() != types::F32 {
        ctx.cast(wk, types::F32)
    } else {
        wk
    }
}

/// The per-layer embedding's causal conv for a prompt chunk (a group of more than one token
/// that is not a verify): the history and `x` (`[width, n_tok, n_seqs]`) side by side in token
/// order, each tap a strided view of them at its shift; the history the next step starts from
/// goes back to `state` (`[hist, width, n_seqs]`). The products and their sums are
/// [`ple_conv_time_major`]'s, in the same order: only the copies differ. The token-major
/// window copies the chunk transposed into it and back once per tap, for a 4,096-token chunk
/// five copies of 160 MiB on qwen4exp's first pipeline stage, the slowest stage of a long
/// prompt (#462).
fn ple_conv_rows<'a>(
    ctx: &'a Context,
    graph: &std::cell::RefCell<Graph<'a>>,
    w: Tensor<'a>,
    state: Tensor<'a>,
    x: Tensor<'a>,
    ple: &PleParams,
) -> Tensor<'a> {
    let (kern, dil, hist) = (ple.conv_kernel, ple.ngram as i64, ple.history());
    let [width, nt, ns, _] = x.ne();
    let past = ctx.cont(ctx.transpose(state));
    let padded = ctx.concat(past, x, 1);
    let nb = padded.nb();
    let window = |start: i64, len: i64| {
        ctx.view_3d(
            padded,
            [width, len, ns],
            [nb[1], nb[2]],
            start as usize * nb[1],
        )
    };
    graph
        .borrow_mut()
        .expand(ctx.cpy(ctx.transpose(window(nt, hist)), state));
    let mut conv: Option<Tensor<'a>> = None;
    for k in 0..kern {
        let start = hist - (kern - 1 - k) * dil;
        let term = ctx.mul(window(start, nt), ple_conv_tap(ctx, w, k, width));
        conv = Some(match conv {
            Some(c) => ctx.add(c, term),
            None => term,
        });
    }
    ctx.reshape_2d(conv.expect("kernel size >= 1"), width, nt * ns)
}

/// A group's sparse-attention tables, filled on the host each step. Positions in `pool_idxs`
/// and `tail_idxs` are in the sequence's attention span (its position), cells elsewhere are
/// rows of the indexer cache.
#[derive(Clone, Copy)]
pub(crate) struct QsaInputs<'a> {
    /// I32 `[n_pool]`: the cell holding each listed pool's key (its last member's).
    pub(crate) pool_cells: Tensor<'a>,
    /// I32 `[kpool, n_pool]`: each pool's member positions (`n_kv` for padding).
    pub(crate) pool_idxs: Tensor<'a>,
    /// I32 `[2, tokens]`: the listed pools the token sees, `first..end` (the indexer's mask is
    /// built from it on the device, see [`qsa_attn::pool_mask`]).
    pub(crate) pool_vis: Tensor<'a>,
    /// I32 `[kpool - 1, tokens]`: the token's own incomplete pool, itself first (`n_kv` past
    /// its start).
    pub(crate) tail_idxs: Tensor<'a>,
    /// I32 `[kpool × n_new]`: member cells of the pools to compute.
    pub(crate) new_idxs: Tensor<'a>,
    /// I64 `[n_new]`: the cell each computed pool key goes to.
    pub(crate) new_rep: Tensor<'a>,
    /// I32 `[4 × n_new]`: rotary position of each computed pool (its first member's), per
    /// section.
    pub(crate) new_pos: Tensor<'a>,
    /// F32 `[n_pool]`: a tie-break added to every listed pool's score, `index in its sequence
    /// × 1e-30` (0 for padding). The scores are sums of rectified head scores, so a pool
    /// scores exactly 0 whenever all its heads do, and `top_k` (a partial sort) breaks ties by
    /// where the pools sit in the listing: which of several zero-score pools a token attends
    /// to would depend on what else is in the batch. The bias makes the most recent of tied
    /// pools win, the same in every batch; it is far below any nonzero score's rounding step,
    /// so untied rankings stay as llama.cpp's.
    pub(crate) pool_bias: Tensor<'a>,
}

impl<'a> QsaInputs<'a> {
    pub(crate) fn new(ctx: &'a Context, kpool: usize, g: &GroupShape, i: usize) -> Self {
        let kpool = kpool as i64;
        let ng = g.tokens() as i64;
        let (n_pool, n_new) = (g.n_pool as i64, g.n_new as i64);
        let input = |ty, ne: &[i64], name: &str| {
            ctx.new_tensor(ty, ne)
                .named(&format!("{name}-{i}"))
                .as_input()
        };
        Self {
            pool_cells: input(types::I32, &[n_pool], "qsa_pool_cells"),
            pool_idxs: input(types::I32, &[kpool, n_pool], "qsa_pool_idxs"),
            pool_vis: input(types::I32, &[2, ng], "qsa_pool_vis"),
            tail_idxs: input(types::I32, &[kpool - 1, ng], "qsa_tail_idxs"),
            new_idxs: input(types::I32, &[kpool * n_new], "qsa_new_idxs"),
            new_rep: input(types::I64, &[n_new], "qsa_new_rep"),
            new_pos: input(types::I32, &[4 * n_new], "qsa_new_pos"),
            pool_bias: input(types::F32, &[n_pool], "qsa_pool_bias"),
        }
    }

    pub(crate) fn list(&self) -> [Tensor<'a>; 8] {
        [
            self.pool_cells,
            self.pool_idxs,
            self.pool_vis,
            self.tail_idxs,
            self.new_idxs,
            self.new_rep,
            self.new_pos,
            self.pool_bias,
        ]
    }

    /// A group's tables from its inputs in [`Self::list`]'s order.
    pub(crate) fn from_slice(t: &[Tensor<'a>]) -> Self {
        let mut it = t.iter().copied();
        let mut next = || it.next().expect("eight sparse-attention tables");
        Self {
            pool_cells: next(),
            pool_idxs: next(),
            pool_vis: next(),
            tail_idxs: next(),
            new_idxs: next(),
            new_rep: next(),
            new_pos: next(),
            pool_bias: next(),
        }
    }

    /// Fills the tables of group `g` (shape `sh`) of a step.
    pub(crate) fn fill(
        &self,
        kpool: usize,
        g: &Group<'_>,
        sh: &GroupShape,
        kv: &KvPool,
        sink: &mut dyn InputSink,
    ) {
        let block = crate::model::core::KV_BLOCK;
        // Past every key: the span's end for the gather path, which drops it through the
        // mask's dump rows; anything past the token for the paged op.
        let sentinel = if sh.n_kv > 0 {
            sh.n_kv as i32
        } else {
            i32::MAX
        };
        let cell =
            |blocks: &[u32], p: usize| (blocks[p / block] as usize * block + p % block) as i32;
        let mut pool_cells = Vec::with_capacity(sh.n_pool);
        let mut pool_bias: Vec<f32> = Vec::with_capacity(sh.n_pool);
        let mut pool_idxs = Vec::with_capacity(sh.n_pool * kpool);
        let mut new_idxs = Vec::with_capacity(sh.n_new * kpool);
        let mut new_rep: Vec<i64> = Vec::with_capacity(sh.n_new);
        let mut new_first = Vec::with_capacity(sh.n_new);
        // Per sequence: its first listed pool.
        let mut first_pool = Vec::with_capacity(g.seqs.len());
        for s in &g.seqs {
            first_pool.push(pool_cells.len());
            let end = s.n_past + sh.n_tok;
            for b in 0..end / kpool {
                let last = b * kpool + kpool - 1;
                pool_cells.push(cell(s.blocks, last));
                pool_bias.push(b as f32 * 1e-30);
                pool_idxs.extend((0..kpool).map(|k| (b * kpool + k) as i32));
                if last >= s.n_past {
                    new_idxs.extend((0..kpool).map(|k| cell(s.blocks, b * kpool + k)));
                    new_rep.push(cell(s.blocks, last) as i64);
                    new_first.push((b * kpool) as i32);
                }
            }
        }
        assert!(pool_cells.len() < sh.n_pool && new_rep.len() <= sh.n_new);
        // Padding: listed pools read a scratch row (finite, never scored: masked); pool keys
        // to compute go to scratch rows of their own (so no two entries write one row).
        let pad = kv.scratch_cell(0) as i32;
        pool_idxs.resize(sh.n_pool * kpool, sentinel);
        let n_pad = sh.n_new - new_rep.len();
        assert!(n_pad < QSA_SCRATCH, "too many padded pool updates");
        for j in 0..n_pad {
            let c = kv.scratch_cell(j + 1);
            new_idxs.extend(std::iter::repeat_n(c as i32, kpool));
            new_rep.push(c as i64);
            new_first.push(0);
        }
        pool_cells.resize(sh.n_pool, pad);
        pool_bias.resize(sh.n_pool, 0.0);
        sink.f32(self.pool_bias, &pool_bias);
        let mut new_pos = Vec::with_capacity(4 * sh.n_new);
        for _ in 0..4 {
            new_pos.extend(&new_first);
        }

        let mut vis = Vec::with_capacity(2 * sh.tokens());
        let mut tail = Vec::with_capacity((kpool - 1) * sh.tokens());
        for (si, s) in g.seqs.iter().enumerate() {
            for t in 0..sh.n_tok {
                let p = s.n_past + t;
                // Pools ending at or before the token.
                let visible = (p + 1) / kpool;
                let p0 = first_pool[si];
                vis.extend([p0 as i32, (p0 + visible) as i32]);
                let n_tail = (p + 1) % kpool;
                tail.extend((0..kpool - 1).map(
                    |k| {
                        if k < n_tail { (p - k) as i32 } else { sentinel }
                    },
                ));
            }
        }
        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        sink.bytes(self.pool_cells, &bytes(&pool_cells));
        sink.bytes(self.pool_idxs, &bytes(&pool_idxs));
        sink.bytes(self.pool_vis, &bytes(&vis));
        sink.bytes(self.tail_idxs, &bytes(&tail));
        sink.bytes(self.new_idxs, &bytes(&new_idxs));
        sink.bytes(
            self.new_rep,
            &new_rep
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<u8>>(),
        );
        sink.bytes(self.new_pos, &bytes(&new_pos));
    }
}

impl Model {
    /// Mixes the `hc` streams of `x` (`[n_embd, hc, n]`) into one block input `[n_embd, n]`
    /// with the weights `{prefix}_{norm,down,up}` and, if `{prefix}_inject` is given, also
    /// returns the per-stream weights `[hc, n]` for [`Self::hc_combine`]. `fused` uses ggml's
    /// gated hc_pre op, as llama.cpp does for every layer's mixer but not the final one.
    fn hc_mix<'a>(
        &'a self,
        ctx: &'a Context,
        x: Tensor<'a>,
        prefix: &str,
        inject: bool,
        fused: bool,
    ) -> Result<(Tensor<'a>, Option<Tensor<'a>>), ModelError> {
        let hp = &self.hp;
        let hc = &self.qwen4().hc;
        let (e, n) = (hp.n_embd, x.ne()[2]);
        let w = |s: &str| self.w(&format!("{prefix}_{s}.weight"));
        // Small steps on a GPU: the whole mixer as one op (see [`hc`]).
        if fused {
            let (norm, down, up) = (w("norm")?, w("down")?, w("up")?);
            let inj = if inject { Some(w("inject")?) } else { None };
            if hc::fits(x, norm, down, up, inj) {
                return Ok(hc::mix(ctx, x, norm, down, up, inj, hp.rms_eps));
            }
        }
        // Grouped RMS norm: each stream normed on its own, one gamma per stream; with the
        // inject weights (the per-stream weights of hc_combine) in the same pass.
        // (The MTP head may store its weights quantized: ggml's chain takes them.)
        // On a GPU, not for steps of the small-step op's size: it takes them when it fits the
        // weights, and where it does not (BF16 mixer weights without --hc-mix-bf16, with
        // --hc-inject-f32), the
        // fused norm's one block per token is slower than ggml's chain.
        let small_gpu_step = !w("norm")?.is_host() && n <= hc::MAX_TOKENS;
        let fused_norm = inject
            && !small_gpu_step
            && hc.n <= hc_norm::HC_NORM_MAX
            && w("norm")?.ty() == types::F32
            && w("inject")?.ty() == types::F32;
        let (xn, inject) = if fused_norm {
            let x = if x.nb()[1] == e as usize * 4 {
                x
            } else {
                ctx.cont(x)
            };
            let (xn, inj) = hc_norm::hc_norm(ctx, x, w("norm")?, w("inject")?, hp.rms_eps);
            (xn, Some(inj))
        } else {
            // On a GPU one Legion op, ggml's bits (#836).
            let xn = if hc_chain::rms_mul_fits(x, w("norm")?, hc_chain::on_gpu(w("norm")?)) {
                hc_chain::rms_mul(ctx, x, w("norm")?, hp.rms_eps)
            } else {
                ctx.mul(
                    ctx.rms_norm(x, hp.rms_eps),
                    ctx.reshape_2d(w("norm")?, e, hc.n),
                )
            };
            let xn = ctx.reshape_2d(xn, e * hc.n, n);
            let inject = if inject {
                Some(ctx.mul_mat(w("inject")?, xn))
            } else {
                None
            };
            (xn, inject)
        };
        Ok((self.hc_mix_tail(ctx, xn, None, prefix, fused, n)?, inject))
    }

    /// The mixer after its norm: from the normed streams `xn` (`[n_embd · hc, n]`), the
    /// low-rank gate and the gated mix of the streams into `[n_embd, n]`. `lo`: the down
    /// projection `mul_mat(down, xn)` if the caller has it from [`hc_norm::hc_post_norm`]'s
    /// extra outputs, the same bits (on a V100 the F16 GEMM from its F16 `xn` and weights,
    /// [`hc_norm::xn16_fits`]; on a GB10 MMQ from its quantized `xn`, [`mmq_preq::fits`]).
    fn hc_mix_tail<'a>(
        &'a self,
        ctx: &'a Context,
        xn: Tensor<'a>,
        lo: Option<Tensor<'a>>,
        prefix: &str,
        fused: bool,
        n: i64,
    ) -> Result<Tensor<'a>, ModelError> {
        let hp = &self.hp;
        let hc = &self.qwen4().hc;
        let e = hp.n_embd;
        let w = |s: &str| self.w(&format!("{prefix}_{s}.weight"));
        let lo = match lo {
            Some(lo) => lo,
            // Steps of 2-16 tokens on a GB10: Legion's dense matvec, ggml's bits (#860).
            None if dense_mv::takes_small(&[w("down")?], xn) => {
                dense_mv::mul_mat(ctx, &[w("down")?], xn)[0]
            }
            // Steps of 9-255 tokens of Q8_0 on a GB10: ggml's MMQ bits, no fixup launch (#836).
            None if q8_mmq::takes(w("down")?, xn) => q8_mmq::mul_mat(ctx, w("down")?, xn),
            None => ctx.mul_mat(w("down")?, xn),
        };
        let lo = hc_chain::lo_act(ctx, lo, 1.0 / hc.n as f32, hc_chain::on_gpu(w("up")?));
        // On a GB10, the up matmul and the gated mix as one op: ggml's bits (#526).
        if fused && hc_up_mix::fits(w("up")?, lo, xn, hc.n) {
            let mixed = hc_up_mix::up_mix(ctx, w("up")?, lo, xn, hc.n, 1.0 / hc.n as f32);
            return Ok(mixed);
        }
        let gate = if q8_mmq::takes(w("up")?, lo) {
            q8_mmq::mul_mat(ctx, w("up")?, lo)
        } else {
            ctx.mul_mat(w("up")?, lo)
        };
        let mixed = if fused {
            let (x3, g3) = (
                ctx.reshape_3d(xn, e, hc.n, n),
                ctx.reshape_3d(gate, e, hc.n, n),
            );
            // On a GPU, ggml's gated hc_pre as Legion's op: the same bits (#836).
            if hc_mix16::pre_fits(x3, g3, !w("up")?.is_host()) {
                hc_mix16::hc_pre(ctx, x3, g3, 1.0 / hc.n as f32)
            } else {
                ctx.hc_pre_gated(x3, g3, 1.0 / hc.n as f32)
            }
        } else {
            let gated = ctx.reshape_3d(ctx.mul(xn, ctx.sigmoid(gate)), e, hc.n, n);
            let row = gated.nb()[1];
            let stream = |c: usize| ctx.view_2d(gated, [e, n], row * hc.n as usize, c * row);
            let mut m = ctx.cont(stream(0));
            for c in 1..hc.n as usize {
                m = ctx.add(m, stream(c));
            }
            ctx.scale(m, 1.0 / hc.n as f32)
        };
        Ok(mixed)
    }

    /// [`Self::hc_combine`] of `pending` (the previous block's output and inject, if any) into
    /// the streams `res`, then the `prefix` mixer with its inject: `(streams, mixed, inject)`.
    /// On a GPU, in a small step, one op does both (see [`hc`]). `bf16`: the step may keep the
    /// streams in BF16 ([`set_hc_bf16`], #1183; a prompt-only pass).
    fn hc_mix_after<'a>(
        &'a self,
        ctx: &'a Context,
        res: Tensor<'a>,
        pending: Option<(Tensor<'a>, Tensor<'a>)>,
        prefix: &str,
        bf16: bool,
    ) -> Result<(Tensor<'a>, Tensor<'a>, Tensor<'a>), ModelError> {
        let w = |s: &str| self.w(&format!("{prefix}_{s}.weight"));
        let (norm, down, up, inj) = (w("norm")?, w("down")?, w("up")?, w("inject")?);
        let rows_ok = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
        // BF16 streams (`--hc-streams-bf16`, #826) only go into the rebuilt combine-and-norm
        // below; every other path gets them as F32.
        let f32_streams = |r: Tensor<'a>| {
            if r.ty() == types::BF16 {
                ctx.cast(r, types::F32)
            } else {
                r
            }
        };
        // (hc::fits takes F32 streams only; BF16 ones, from prompt steps, go on below.)
        if hc::fits(res, norm, down, up, Some(inj))
            && pending.is_none_or(|(o, i)| rows_ok(o) && rows_ok(i))
        {
            let (mixed, inject, res) = hc::combine_mix(
                ctx,
                res,
                pending,
                norm,
                down,
                up,
                Some(inj),
                self.hp.rms_eps,
            );
            return Ok((res, mixed, inject.expect("inject")));
        }
        // A prompt step: the combine and the mixer's norm in one pass over the streams.
        if let Some((out, inject)) = pending
            && self.post_norm_fits(res, out, inject, norm, inj)
        {
            let n = res.ne()[2];
            // V100, Q8_0 mixers: xn and the weight as F16; sm_80+, BF16 mixers (the NVFP4
            // checkpoints): xn as BF16 (#469). For ggml-cuda's own cuBLAS GEMM either way.
            let f16 = (hc_norm::xn16_fits(down, n) || hc_norm::xnb_fits(down, n)).then_some(down);
            // On a GB10, the down matmul's input quantized for ggml's MMQ by the same kernel
            // (#363).
            let quant =
                f16.is_none() && mmq_preq::fits(down, n) && (res.ne()[0] * res.ne()[1]) % 512 == 0;
            // And the up matmul and mix as one op that rebuilds xn from the streams and their
            // norm scales, so the norm does not write xn (#526).
            let hc_n = self.qwen4().hc.n;
            if quant
                && hc_up_mix::rebuild()
                && hc_up_mix::fits_up(up, n, hc_n)
                && hc_norm::rebuild_fits(res.ne()[0] * res.ne()[1], hc_n, n)
            {
                let bf16 = hc_norm::streams_bf16_fits(res, inj);
                let res = if bf16 { res } else { f32_streams(res) };
                let p = hc_norm::hc_post_norm_rebuilt(
                    ctx,
                    res,
                    out,
                    inject,
                    norm,
                    inj,
                    self.hp.rms_eps,
                    bf16,
                );
                let lo = mmq_preq::mul_mat_cols(ctx, down, p.xq, n);
                let lo = hc_chain::lo_act(ctx, lo, 1.0 / hc_n as f32, hc_chain::on_gpu(up));
                let scale = 1.0 / hc_n as f32;
                let mixed =
                    hc_up_mix::up_mix_rebuilt(ctx, up, lo, p.res, p.scales, norm, hc_n, scale);
                return Ok((p.res, mixed, p.inject));
            }
            // The BF16 mixers' rebuilt norm below takes this step (sm_80+, the NVFP4
            // checkpoints): with `bf16` (`--hc-bf16`, #1183) it reads and writes BF16 streams.
            let xnb_rebuilt = f16.is_some_and(|d| d.ty() == types::BF16)
                && !hc_mix16::on()
                && hc_up_mix::rebuild()
                && hc_up_mix::fits_up_bf16(up, n, hc_n)
                && hc_norm::rebuild_xnb_fits(res.ne()[0], hc_n, inj.ne()[1], n);
            let streams16 = bf16 && xnb_rebuilt;
            let res = if streams16 { res } else { f32_streams(res) };
            // On a V100, the gated mix as one op that also writes the block input as F16, the
            // operand ggml-cuda's cuBLAS GEMMs convert it to in each of its Q8_0 projections, so
            // they take it instead (#774); and, where the norm can write the streams' scales
            // instead of xn, the op rebuilds xn itself (as #526 on a GB10).
            if let Some(down) = f16
                && hc_mix16::on()
            {
                let scale = 1.0 / hc_n as f32;
                let gate = |down16: Tensor<'a>, xn16: Tensor<'a>| {
                    let lo = hc_chain::lo_act(
                        ctx,
                        ctx.mul_mat(down16, xn16),
                        scale,
                        hc_chain::on_gpu(up),
                    );
                    ctx.mul_mat(up, lo)
                };
                let [e, _, _, _] = res.ne();
                let (res, (mixed, mixed16), inject) = if hc_up_mix::rebuild()
                    && hc_norm::rebuild16_fits(e, hc_n, inj.ne()[1], n)
                {
                    let p = hc_norm::hc_post_norm_rebuilt16(
                        ctx,
                        res,
                        out,
                        inject,
                        norm,
                        inj,
                        self.hp.rms_eps,
                        down,
                    );
                    let gate = gate(p.down16, p.xn16);
                    let mix = hc_mix16::mix_rebuilt(ctx, gate, p.res, p.scales, norm, hc_n, scale);
                    (p.res, mix, p.inject)
                } else {
                    let p = hc_norm::hc_post_norm(
                        ctx,
                        res,
                        out,
                        inject,
                        norm,
                        inj,
                        self.hp.rms_eps,
                        f16,
                        false,
                    );
                    let (down16, xn16) = p.down16.zip(p.xn16).expect("the F16 copies");
                    let mix = hc_mix16::mix(ctx, gate(down16, xn16), p.xn, hc_n, scale);
                    (p.res, mix, p.inject)
                };
                ctx.set_shared_cast(mixed, types::F16, mixed16);
                return Ok((res, mixed, inject));
            }
            // sm_80+, BF16 mixers (the NVFP4 checkpoints), where the BF16 up-mix takes the step:
            // the norm writes the streams' scales and the BF16 xn, not the F32 xn, and the up-mix
            // rebuilds xn from the streams with the norm's own two rounded multiplies (#836, as
            // #526 does for Q8_0 mixers). The norm's node keeps hc_post_norm's size and layout
            // and the up-mix's nodes keep theirs, and the down matmul and the activation are the
            // ones `hc_mix_tail` makes: the same bits, 40 KB a token less written by the norm
            // and none more read by the up-mix (the streams in place of xn). `legion serve
            // --hc-xn-written` keeps xn written and read.
            if let Some(down) = f16
                && xnb_rebuilt
            {
                let scale = 1.0 / hc_n as f32;
                // With BF16 streams (`--hc-bf16`) on a GB10, the norm does not write the BF16
                // xn and the down matmul rebuilds it from the streams (#1183): xn's bits, the
                // GEMM's sums in another order than cuBLAS's.
                let [e, _, _, _] = res.ne();
                if streams16 && hc_down::takes(down, norm, [e, hc_n, n, 1]) {
                    let p = hc_norm::hc_post_norm_b16_streams(
                        ctx,
                        res,
                        out,
                        inject,
                        norm,
                        inj,
                        self.hp.rms_eps,
                        down,
                    );
                    let lo = hc_down::mul_mat(ctx, down, p.res, p.scales, norm);
                    let lo = hc_chain::lo_act(ctx, lo, scale, hc_chain::on_gpu(up));
                    let mixed =
                        hc_up_mix::up_mix_rebuilt(ctx, up, lo, p.res, p.scales, norm, hc_n, scale);
                    return Ok((p.res, mixed, p.inject));
                }
                let p = hc_norm::hc_post_norm_rebuilt_xnb(
                    ctx,
                    res,
                    out,
                    inject,
                    norm,
                    inj,
                    self.hp.rms_eps,
                    down,
                    streams16,
                );
                let lo =
                    hc_chain::lo_act(ctx, ctx.mul_mat(down, p.xnb), scale, hc_chain::on_gpu(up));
                let mixed =
                    hc_up_mix::up_mix_rebuilt(ctx, up, lo, p.res, p.scales, norm, hc_n, scale);
                return Ok((p.res, mixed, p.inject));
            }
            // sm_80+, BF16 mixers (the NVFP4 checkpoints): the norm writes the BF16 xn and the
            // streams' scales, not the F32 xn, and the gated mix rebuilds xn (#836). The down and
            // up matmuls as `hc_mix_tail` makes them, and the norm's node in hc_post_norm's
            // layout (the BF16 xn where cuBLAS reads it without the switch), so the same bits.
            // Opt-in (`--hc-mix-rebuild-bf16`): greedy output still moved in serving tests.
            if let Some(down) = f16
                && hc_mix16::rebuild_bf16()
                && down.ty() == types::BF16
                && !hc_up_mix::fits_up(up, n, hc_n)
                // The BF16 up-mix (default on) takes these steps in `hc_mix_tail`.
                && !hc_up_mix::fits_up_bf16(up, n, hc_n)
                && hc_up_mix::rebuild()
                && hc_norm::rebuild16_fits(res.ne()[0], hc_n, inj.ne()[1], n)
            {
                let scale = 1.0 / hc_n as f32;
                let p = hc_norm::hc_post_norm_rebuilt_xnb(
                    ctx,
                    res,
                    out,
                    inject,
                    norm,
                    inj,
                    self.hp.rms_eps,
                    down,
                    false,
                );
                let lo =
                    hc_chain::lo_act(ctx, ctx.mul_mat(down, p.xnb), scale, hc_chain::on_gpu(up));
                let gate = if q8_mmq::takes(up, lo) {
                    q8_mmq::mul_mat(ctx, up, lo)
                } else {
                    ctx.mul_mat(up, lo)
                };
                let mixed =
                    hc_mix16::mix_rebuilt_f32(ctx, gate, p.res, p.scales, norm, hc_n, scale);
                return Ok((p.res, mixed, p.inject));
            }
            let p = hc_norm::hc_post_norm(
                ctx,
                res,
                out,
                inject,
                norm,
                inj,
                self.hp.rms_eps,
                f16,
                quant,
            );
            let lo = match (p.down16.zip(p.xn16), p.xq, p.xnb) {
                (Some((down16, xn16)), ..) => Some(ctx.mul_mat(down16, xn16)),
                (None, Some(xq), _) => Some(mmq_preq::mul_mat(ctx, down, xq, p.xn)),
                (None, None, Some(xnb)) => Some(ctx.mul_mat(down, xnb)),
                (None, None, None) => None,
            };
            let mixed = self.hc_mix_tail(ctx, p.xn, lo, prefix, true, n)?;
            return Ok((p.res, mixed, p.inject));
        }
        let res = f32_streams(res);
        let res = match pending {
            Some((out, inject)) => self.hc_combine(ctx, res, out, inject, hc_chain::on_gpu(up)),
            None => res,
        };
        let (mixed, inject) = self.hc_mix(ctx, res, prefix, true, true)?;
        Ok((res, mixed, inject.expect("inject")))
    }

    /// Whether [`hc_norm::hc_post_norm`] takes this combine and mixer: F32 everything, at most
    /// [`hc_norm::HC_NORM_MAX`] streams and inject outputs, the streams and rows contiguous, the
    /// weights in GPU memory.
    /// [`set_fused_post_norm`]`(false)` keeps ggml's combine and the separate norm (for A/B runs).
    fn post_norm_fits(
        &self,
        res: Tensor<'_>,
        out: Tensor<'_>,
        inject: Tensor<'_>,
        norm: Tensor<'_>,
        inj_w: Tensor<'_>,
    ) -> bool {
        let off = !hc_norm::fused_post_norm();
        let e = self.hp.n_embd;
        let hc = self.qwen4().hc.n;
        let f32_rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
        // On a GPU only: the CPU graph stays ggml's chain (the reference path). Not for steps
        // the small-step op would take (one block per token: slower than the chain there).
        !off && cfg!(feature = "cuda")
            && res.ne()[2] > hc::MAX_TOKENS
            && !norm.is_host()
            && hc <= hc_norm::HC_NORM_MAX
            && inj_w.ne()[1] <= hc_norm::HC_NORM_MAX
            && norm.ty() == types::F32
            && inj_w.ty() == types::F32
            && norm.is_contiguous()
            && inj_w.is_contiguous()
            && (f32_rows(res) && res.nb()[1] == e as usize * 4
                // BF16 streams (#826): converted to F32 unless the rebuilt op takes them
                || res.ty() == types::BF16 && res.nb()[0] == 2 && res.nb()[1] == e as usize * 2)
            && f32_rows(out)
            && f32_rows(inject)
            && out.ne()[0] == e
            && inject.ne()[0] == hc
    }

    /// Adds the block output `out` (`[n_embd, n]`) to every stream of `res`, stream `h`
    /// scaled by `2 sigmoid(inject[h] / hc)`. `gpu`: the combine runs on one GPU
    /// ([`hc_chain::on_gpu`] of a mixer weight of its stage), where it is one Legion op, ggml's
    /// bits (#836).
    fn hc_combine<'a>(
        &'a self,
        ctx: &'a Context,
        res: Tensor<'a>,
        out: Tensor<'a>,
        inject: Tensor<'a>,
        gpu: bool,
    ) -> Tensor<'a> {
        let hc = self.qwen4().hc.n;
        if hc_chain::combine_fits(out, res, inject, gpu) {
            return hc_chain::combine(ctx, out, res, inject);
        }
        let w = ctx.scale(ctx.sigmoid(ctx.scale(inject, 1.0 / hc as f32)), 2.0);
        ctx.hc_post(out, res, w)
    }

    /// The decoder stack of a hyper-connection model, from the token embeddings `x`
    /// (`[n_embd, n]`). With logits to compute, the last layer runs its FFN on the output rows
    /// only and the result is the final mixer's output for those rows (`[n_embd, n_out]`);
    /// without, the residual streams. With `capture`, the last layer runs on every row and the
    /// second result is the residual streams of all of them, `[n_embd × hc, n]` (what an MTP
    /// head reads), before the output rows are cropped.
    fn hc_layers<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: bool,
        layers: std::ops::Range<usize>,
    ) -> Result<(Tensor<'a>, Option<Tensor<'a>>), ModelError> {
        let (ctx, tap) = (pass.ctx, &pass.tap);
        let hp = &self.hp;
        let q4 = self.qwen4();
        let hc = &q4.hc;
        let (e, mut n) = (hp.n_embd, pass.n);
        let n_out: i64 = pass.shape.iter().map(|g| g.out_rows() as i64).sum();
        // A later pipeline stage starts from the streams the previous one handed over.
        let mut res = if layers.start == 0 {
            ctx.repeat_4d(ctx.reshape_3d(x, e, 1, n), [e, hc.n, n, 1])
        } else {
            ctx.reshape_3d(x, e, hc.n, n)
        };
        let last = layers.end == hp.n_layer;
        // A block's output waits to be combined into the streams until the next mixer, which
        // does both in one op on a GPU (see `hc_mix_after`).
        let mut pending: Option<(Tensor<'a>, Tensor<'a>)> = None;
        // `--hc-streams-bf16` (#826): the mixers may hand BF16 streams on; every other reader
        // takes F32 ones.
        let f32_streams = |r: Tensor<'a>| {
            if r.ty() == types::BF16 {
                ctx.cast(r, types::F32)
            } else {
                r
            }
        };
        // Whether this stage's mixers run on one GPU (its first one's weights tell).
        let first = layers.start.min(hp.n_layer.saturating_sub(1));
        let gpu = hc_chain::on_gpu(self.w(&format!("blk.{first}.hc_attn_up.weight"))?);
        let flush = |res: Tensor<'a>, p: &mut Option<(Tensor<'a>, Tensor<'a>)>| {
            let res = f32_streams(res);
            match p.take() {
                Some((out, inject)) => self.hc_combine(ctx, res, out, inject, gpu),
                None => res,
            }
        };
        // `--tp-seq-par` (#751): in a big step, from the first layer after the PLE to the one
        // before the model's last, each rank keeps its share of the tokens' streams and runs
        // the mixers and combines on those only; each block's input is gathered from every
        // rank's share and its partial output reduce-scattered to them.
        let sp = if tap.on { None } else { self.tp_seq_par(n) };
        // `--hc-bf16` (#1183): BF16 streams between the mixers, in prompt-only passes without
        // the sequence-parallel split.
        let bf16 = hc_norm::hc_bf16() && pass.prompt_only() && sp.is_none() && !tap.on;
        let ple_layer = q4.ple.as_ref().map(|p| p.layer);
        let (mut split, n_all) = (false, n);
        // `--tp-overlap` (#758): the split layers in two micro-batches.
        let mut micro: Option<tp_overlap::Micro<'a>> = None;
        // Every rank's share of the streams, and the pending output combined into them.
        let gather = |res: Tensor<'a>, p: &mut Option<(Tensor<'a>, Tensor<'a>)>| {
            let (c, ..) = sp.expect("sequence parallel");
            let res = ctx.reshape_2d(flush(res, p), e * hc.n, res.ne()[2]);
            let res = crate::tp_net::seq_par::all_gather(ctx, c, res, n_all);
            ctx.reshape_3d(res, e, hc.n, n_all)
        };
        // A block input `h` gathered from every rank's share for layer `il`'s projection
        // `name` (its first; the others are of the same kind). Where its columns travel in BF16
        // and that projection takes a BF16 cast of its input (the NVFP4 checkpoints' BF16
        // projections, #469), the gather stays BF16 (#1176): the BF16 projections and the
        // experts read it as that cast, the values they would get from the F32 gather, and only
        // the F32 readers (the router) widen it, exactly.
        let gather_in = |h: Tensor<'a>, il: usize, name: &str| {
            let (c, ..) = sp.expect("sequence parallel");
            // (This rank's share of `h` has the gathered input's type and rows.)
            let takes = self
                .lw(il, name)
                .is_ok_and(|w| crate::model::core::bf16_join_fits(w, h));
            let bf16 = takes
                .then(|| crate::tp_net::seq_par::all_gather_bf16(ctx, c, h, n_all))
                .flatten();
            match bf16 {
                Some(g) => {
                    let h = ctx.cast(g, types::F32);
                    ctx.set_shared_cast(h, types::BF16, g);
                    h
                }
                None => crate::tp_net::seq_par::all_gather(ctx, c, h, n_all),
            }
        };
        for il in layers {
            let want = sp.is_some() && il + 1 < hp.n_layer && ple_layer.is_none_or(|p| p < il);
            if split && !want {
                if let Some(st) = micro.take() {
                    res = self.micro_join(pass, st);
                }
                res = gather(res, &mut pending);
                split = false;
            }
            if q4.ple.as_ref().is_some_and(|p| p.layer == il) {
                res = flush(res, &mut pending);
                res = self.ple(pass, il, res)?;
            }
            if want && !split {
                let (_, rank, ranks) = sp.expect("sequence parallel");
                let (a, m) = crate::tp_net::seq_par::share(n_all as usize, rank, ranks);
                let (a, m) = (a as i64, m as i64);
                // Columns `a..a + m` of a `[row, n]` tensor, contiguous.
                let cols = |t: Tensor<'a>| {
                    let t = if t.is_contiguous() { t } else { ctx.cont(t) };
                    let row = t.ne()[0];
                    let t = ctx.reshape_2d(t, row, n_all);
                    ctx.cont(ctx.view_2d(t, [row, m], t.nb()[1], a as usize * t.nb()[1]))
                };
                let flat = cols(ctx.reshape_2d(f32_streams(res), e * hc.n, n));
                res = ctx.reshape_3d(flat, e, hc.n, m);
                pending = pending.map(|(out, inject)| (cols(out), cols(inject)));
                split = true;
                micro = self.micro_split(sp, n_all, res, &mut pending, ctx);
            }
            if let Some(st) = micro.as_mut() {
                self.micro_layer(pass, il, st)?;
                continue;
            }
            let pre = format!("blk.{il}.hc_attn");
            let (r, h, mut inject) = self.hc_mix_after(ctx, res, pending.take(), &pre, bf16)?;
            res = r;
            let h = if !split {
                h
            } else if hp.is_recurrent(il) {
                gather_in(h, il, "attn_qkv.weight")
            } else {
                gather_in(h, il, "attn_q.weight")
            };
            let mut attn = if hp.is_recurrent(il) {
                self.delta_net(pass, il, h)?
            } else {
                self.attention(pass, il, h)?
            };
            if il + 1 == hp.n_layer
                && !capture
                && let Some(rows) = pass.out_rows
            {
                // Everything below is per token: keep only the rows that produce logits.
                attn = ctx.get_rows(attn, rows);
                inject = ctx.get_rows(inject, rows);
                // (BF16 streams, #826: get_rows gives their rows as F32.)
                let flat = ctx.get_rows(ctx.reshape_2d(res, e * hc.n, n), rows);
                n = n_out;
                res = ctx.reshape_3d(flat, e, hc.n, n);
            }
            // A rank's share of the heads gives a partial output (`--tp-rank`, #733).
            let attn = self.tp_sum_split(ctx, attn, sp.filter(|_| split));
            pending = Some((attn, inject));
            let pre = format!("blk.{il}.hc_ffn");
            let (r, h, inject) = self.hc_mix_after(ctx, res, pending.take(), &pre, bf16)?;
            res = r;
            let h = if split {
                gather_in(h, il, "ffn_up_shexp.weight")
            } else {
                h
            };
            // A paired step's layer shares its experts with the stage GPU's pair (`--pair-tp`).
            let ffn = match self.pair_moe(pass, il, h, n)? {
                Some(ffn) => ffn,
                None => {
                    let rows = self.qwen4().moe_rows(pass);
                    self.moe_rows(ctx, tap, &pass.graph, il, h, n, rows, pass.prompt_only())?
                }
            };
            let ffn = self.tp_sum_split(ctx, ffn, sp.filter(|_| split));
            tap.tap("ffn_out", il, ffn);
            pending = Some((ffn, inject));
            if tap.on {
                res = flush(res, &mut pending);
            }
            tap.tap("l_last", il, res);
        }
        if split {
            if let Some(st) = micro.take() {
                res = self.micro_join(pass, st);
            }
            res = gather(res, &mut pending);
        }
        res = flush(res, &mut pending);
        if !last {
            return Ok((ctx.reshape_2d(res, e * hc.n, n), None));
        }
        let all = capture.then(|| ctx.reshape_2d(res, e * hc.n, n));
        if n_out == 0 {
            return Ok((res, all));
        }
        if capture && let Some(rows) = pass.out_rows {
            let flat = ctx.get_rows(ctx.reshape_2d(res, e * hc.n, n), rows);
            res = ctx.reshape_3d(flat, e, hc.n, n_out);
        }
        Ok((self.hc_mix(ctx, res, "output_hc", false, false)?.0, all))
    }

    /// Adds the per-layer n-gram embedding to the residual streams `res` in layer `il`.
    fn ple<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        res: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        let hp = &self.hp;
        let hc = self.qwen4().hc.n;
        let ple = self.qwen4().ple.as_ref().expect("per-layer embedding");
        let e = hp.n_embd;
        let hc_dim = hc * e;
        let emb = pass.ple.expect("per-layer embedding input");

        let key = ctx.mul_mat(self.lw(il, "ple_key.weight")?, emb);
        let value = ctx.mul_mat(self.lw(il, "ple_value.weight")?, emb);
        if let Some(out) = self.ple_ops(pass, il, res, key, value)? {
            return Ok(out);
        }
        // RMS norm per stream, one gamma per stream.
        let grouped_norm = |x: Tensor<'a>, w: &str| -> Result<Tensor<'a>, ModelError> {
            let x = ctx.reshape_3d(x, e, hc, n);
            Ok(ctx.mul(
                ctx.rms_norm(x, hp.rms_eps),
                ctx.reshape_2d(self.lw(il, w)?, e, hc),
            ))
        };
        let key = grouped_norm(key, "ple_norm_key.weight")?;
        let query = grouped_norm(res, "ple_norm_query.weight")?;
        // Per-stream dot product, signed square root, sigmoid.
        let s = ctx.sum_rows(ctx.mul(key, query));
        let s = ctx.scale(s, 1.0 / (e as f32).sqrt());
        let mag = ctx.sqrt(ctx.clamp(ctx.abs(s), 1e-6, f32::INFINITY));
        let gate = ctx.sigmoid(ctx.mul(ctx.sgn(s), mag));
        tap.tap("ple_gate", il, gate);
        let v3 = ctx.repeat_4d(ctx.reshape_3d(value, e, 1, n), [e, hc, n, 1]);
        let gated = ctx.mul(v3, gate);
        let normalized = grouped_norm(ctx.reshape_2d(gated, hc_dim, n), "ple_norm_conv.weight")?;
        let normalized = ctx.reshape_2d(normalized, hc_dim, n);

        // Depthwise causal conv dilated by the n-gram size, as a sum of shifted copies over
        // [history | this chunk]; tap k reads (kernel - 1 - k) × n-gram positions back.
        let w = self.lw(il, "ple_conv1d.weight")?;
        let mut outs = Vec::new();
        for (off, sh) in pass.groups() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let x = ctx.reshape_3d(cols(ctx, normalized, off, nt * ns), hc_dim, nt, ns);
            let state = pass.rec.ple_slots(ctx, il, sh.slot, sh.n_seqs);
            ple_captures(pass, il, sh, x, ple.history());
            if !sh.verify && nt > 1 {
                outs.push(ple_conv_rows(ctx, &pass.graph, w, state, x, ple));
                continue;
            }
            let padded = ctx.concat(state, ctx.transpose(x), 0);
            if sh.verify {
                // A verify group keeps its history; the commit takes the kept window.
                let spec = pass.spec.expect("checked in build");
                spec.store_ple(ctx, &mut pass.graph.borrow_mut(), sh.slot, padded);
            } else {
                ple_keep_history(ctx, &pass.graph, padded, state, nt, ple);
            }
            outs.push(ple_conv_time_major(ctx, w, padded, nt, ple));
        }
        let conv = ctx.silu(concat_cols(ctx, outs));
        let conv = ctx.reshape_3d(ctx.cont(conv), e, hc, n);
        tap.tap("ple_conv_out", il, conv);
        Ok(ctx.add(res, ctx.add(gated, conv)))
    }

    /// [`Self::ple`] after its two matmuls as Legion's two ops ([`ple_fused`]: ggml-cuda's bits
    /// in two launches per group instead of about thirty nodes), for prompt steps on a GPU:
    /// every group a prompt chunk (more than one token, not a verify), the weights and the
    /// histories in GPU memory, no tensors tapped. `None` where they do not fit or are off
    /// (`--ple-ops`; ggml's chain).
    fn ple_ops<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        res: Tensor<'a>,
        key: Tensor<'a>,
        value: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        if !ple_fused::fused_ple() {
            return Ok(None);
        }
        let ctx = pass.ctx;
        let hp = &self.hp;
        let (e, n) = (hp.n_embd, pass.n);
        let hc = self.qwen4().hc.n;
        let ple = self.qwen4().ple.as_ref().expect("per-layer embedding");
        let gammas = [
            self.lw(il, "ple_norm_key.weight")?,
            self.lw(il, "ple_norm_query.weight")?,
            self.lw(il, "ple_norm_conv.weight")?,
        ];
        let w = self.lw(il, "ple_conv1d.weight")?;
        let on_gpu = |t: Tensor<'_>| !t.is_host() && !t.is_meta();
        let f32_rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
        let groups_fit = pass.groups().all(|(_, sh)| !sh.verify && sh.n_tok > 1);
        let states: Vec<_> = if groups_fit {
            pass.groups()
                .map(|(_, sh)| pass.rec.ple_slots(ctx, il, sh.slot, sh.n_seqs))
                .collect()
        } else {
            Vec::new()
        };
        let fits = cfg!(feature = "cuda")
            && !pass.tap.on
            && hc == ple_fused::PLE_HC
            && e >= ple_fused::PLE_MIN_E
            && gammas
                .iter()
                .chain([&w])
                .all(|&g| g.ty() == types::F32 && g.is_contiguous() && on_gpu(g))
            && w.ne()[0] == ple.conv_kernel
            && f32_rows(res)
            && res.nb()[1] == e as usize * 4
            && f32_rows(key)
            && f32_rows(value)
            && groups_fit
            && states.iter().all(|&s| on_gpu(s));
        if !fits {
            return Ok(None);
        }
        let (xn, gate) = ple_fused::gate_norm(
            ctx,
            key,
            res,
            value,
            gammas,
            hp.rms_eps,
            1.0 / (e as f32).sqrt(),
            1e-6,
        );
        let streams = ctx.view_2d(res, [e * hc, n], res.nb()[2], 0);
        let mut outs = Vec::new();
        for ((off, sh), state) in pass.groups().zip(states) {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let m = nt * ns;
            ple_captures(pass, il, sh, cols(ctx, xn, off, m), ple.history());
            outs.push(ple_fused::conv_out(
                ctx,
                cols(ctx, xn, off, m),
                state,
                w,
                cols(ctx, value, off, m),
                cols(ctx, gate, off, m),
                cols(ctx, streams, off, m),
                nt,
                ple.ngram as i64,
            ));
        }
        Ok(Some(ctx.reshape_3d(concat_cols(ctx, outs), e, hc, n)))
    }

    /// Where the per-layer n-gram table is in the model file (`(tensor, type, row elements)`),
    /// for [`Self::set_ngram_table`]; `None` for models without one.
    pub fn ngram_table_info(
        gguf: &crate::gguf::GgufFile,
    ) -> Option<crate::residency::lookup_table::TableFile> {
        let t = gguf.tensor("per_layer_token_embd.weight")?;
        let path = gguf.shard_paths().nth(t.file_index)?.to_path_buf();
        let (dim, n_rows) = (*t.shape.first()?, *t.shape.get(1)?);
        Some(crate::residency::lookup_table::TableFile {
            path,
            offset: t.offset,
            n_rows,
            row_bytes: crate::ggml::row_size(t.ggml_type, dim as usize),
        })
    }

    /// Reads the per-layer n-gram table's rows through `table` (opened on
    /// [`Self::ngram_table_info`]) instead of the weights' host memory.
    pub fn set_ngram_table(&mut self, table: crate::residency::lookup_table::LookupTable) {
        self.ngram = Some(table);
    }

    /// Starts reading the n-gram rows of `tokens`, which follow `before` in a sequence (its
    /// prompt so far; empty at its start), so that their step finds them read: the next step's
    /// tokens. A hint (see [`crate::residency::lookup_table::LookupTable::prefetch`]).
    pub fn prefetch_ngram(&self, before: &[u32], tokens: &[u32]) {
        self.prefetch_ngram_for(before, tokens, false);
    }

    /// [`Self::prefetch_ngram`] of a prompt's next chunk, before the step of its current one:
    /// that step's fill does not wait for these reads, they run while it computes (see
    /// [`crate::residency::lookup_table::LookupTable::prefetch_next`]).
    pub fn prefetch_ngram_next(&self, before: &[u32], tokens: &[u32]) {
        self.prefetch_ngram_for(before, tokens, true);
    }

    fn prefetch_ngram_for(&self, before: &[u32], tokens: &[u32], next: bool) {
        let ple = params(&self.hp).and_then(|p| p.ple.as_ref());
        let (Some(ple), Some(store)) = (ple, &self.ngram) else {
            return;
        };
        let n_prev = ple.ngram - 1;
        let mut prev: Vec<u32> = std::iter::repeat_n(crate::model::core::state::NO_TOKEN, n_prev)
            .chain(
                before[before.len().saturating_sub(n_prev)..]
                    .iter()
                    .copied(),
            )
            .collect();
        prev.drain(..prev.len() - n_prev);
        let mut rows = Vec::new();
        for &tok in tokens {
            ple.rows(tok, &prev, &mut rows);
            prev.remove(0);
            prev.push(tok);
        }
        let rows: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
        if next {
            store.prefetch_next(&rows);
        } else {
            store.prefetch(&rows);
        }
    }

    /// The store the n-gram rows come from, if one is set.
    pub fn ngram_table(&self) -> Option<&crate::residency::lookup_table::LookupTable> {
        self.ngram.as_ref()
    }

    /// Fills the per-layer embedding input of a step: hashes every token's n-grams (with the
    /// predecessors kept per slot in `rec`, which this advances; a verify group leaves them and
    /// records its tokens in `spec` for the commit) and dequantizes the table rows, fetched
    /// through the n-gram store when one is set, else read from the weights' host memory.
    fn fill_ple(
        &self,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        rec: &crate::model::core::StatePool,
        spec: Option<&crate::model::core::SpecPool>,
        sink: &mut dyn InputSink,
    ) {
        let ple = params(&self.hp).and_then(|p| p.ple.as_ref());
        let (Some(ple), Some(t)) = (ple, inputs.ple) else {
            return;
        };
        let rows = ngram_rows(ple, groups, |g, slot, tokens| {
            if g.verify {
                let spec = spec.expect("a verify group needs the replay pool");
                spec.record_fed(slot, tokens);
                rec.peek_tokens(slot)
            } else {
                rec.push_tokens(slot, tokens)
            }
        });
        // Rows read through a store (the SSD, on a GPU sharing the host's memory) wait for the
        // step's compute when it asked for them late (`Self::set_ple_late`); their reads are
        // issued now (`ple_late`).
        let mut late = self.ple_late.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(store) = &self.ngram
            && late.0
        {
            late.1 = Some(LateRows::new(rows, store));
            return;
        }
        drop(late);
        // Gathered while the round before computed (`Self::gather_ple_ahead`).
        let ahead = self
            .ple_ahead
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(&rows);
        match ahead {
            Some(values) => sink.f32(t, &values),
            None => sink.f32(t, &self.gather_ple(&rows)),
        }
    }

    /// Gathers the per-layer embedding rows of the next fill of `groups` (prompt groups of the
    /// slots whose token history `rec` holds) now, for that fill to write
    /// ([`ple_ahead`]): a pipelined prompt's next micro-chunk, while the stages compute.
    /// Nothing for a model without the embedding or for verify groups.
    pub(crate) fn gather_ple_ahead(&self, groups: &[Group<'_>], rec: &StatePool) {
        let Some(ple) = params(&self.hp).and_then(|p| p.ple.as_ref()) else {
            return;
        };
        if groups.iter().any(|g| g.verify) {
            return;
        }
        // The history the fill's `push_tokens` will return: the fill before it already ran.
        let rows = ngram_rows(ple, groups, |_, slot, _| rec.peek_tokens(slot));
        let values = self.gather_ple(&rows);
        self.ple_ahead
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(rows, values);
    }

    /// Fills so far that wrote rows gathered ahead ([`Self::gather_ple_ahead`]).
    #[cfg(test)]
    pub(crate) fn ple_ahead_hits(&self) -> u64 {
        self.ple_ahead
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .hits()
    }

    /// While `on`, a step's fill leaves the per-layer n-gram rows it reads through the n-gram
    /// store for [`Self::take_ple_late`], whose caller writes them with [`Self::write_ple`]
    /// while the step's first layer computes: the newest draft's rows come from the SSD
    /// (~1 ms on the GB10), which the verify step waited for before it started. Turning it on
    /// drops rows a failed step left.
    pub(crate) fn set_ple_late(&self, on: bool) {
        let mut late = self.ple_late.lock().unwrap_or_else(|e| e.into_inner());
        late.0 = on;
        if on {
            late.1 = None;
        }
    }

    /// Whether the last fill left rows for its step's compute ([`Self::take_ple_late`]).
    pub(crate) fn ple_late_pending(&self) -> bool {
        self.ple_late
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .is_some()
    }

    /// The rows the last fill left for its step's compute ([`Self::set_ple_late`]).
    pub(crate) fn take_ple_late(&self) -> Option<LateRows> {
        self.ple_late
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .take()
    }

    /// Writes the per-layer embedding input `t` from the n-gram table's rows the fill left
    /// (see [`Self::fill_ple`]).
    pub(crate) fn write_ple(&self, t: Tensor<'_>, late: LateRows) {
        t.set_f32(&self.gather_ple_from(late));
    }

    /// The per-layer embedding input's values for the n-gram table's `rows` (see
    /// [`Self::fill_ple`]).
    fn gather_ple(&self, rows: &[i32]) -> Vec<f32> {
        self.gather_ple_rows(rows, None)
    }

    /// [`Self::gather_ple`] of the rows a fill left, their bytes from the reads it issued.
    fn gather_ple_from(&self, late: LateRows) -> Vec<f32> {
        let rows = late.rows.clone();
        self.gather_ple_rows(&rows, Some(late))
    }

    /// [`Self::gather_ple`], the store's bytes from `late`'s reads where given (`late` holds
    /// `rows`).
    fn gather_ple_rows(&self, rows: &[i32], late: Option<LateRows>) -> Vec<f32> {
        let ple = params(&self.hp)
            .and_then(|p| p.ple.as_ref())
            .expect("per-layer embedding");
        let table = self
            .weights
            .get("per_layer_token_embd.weight")
            .expect("per_layer_token_embd.weight");
        let ty = table.ty();
        let dim = ple.head_dim as usize;
        let row_bytes = crate::ggml::row_size(ty, dim);
        let n_rows = table.ne()[1] as usize;
        assert!(
            rows.iter().all(|&r| (r as usize) < n_rows),
            "n-gram row outside the {n_rows}-row table"
        );
        // The rows' bytes: from the store (RAM, page cache or the SSD), or straight from the
        // weights' host memory.
        let fetched;
        let (bytes, at): (&[u8], Vec<usize>) = match &self.ngram {
            Some(store) => {
                fetched = match late {
                    Some(late) => late.bytes(store),
                    None => store.fetch(&ple_late::ids(rows)),
                }
                .unwrap_or_else(|e| panic!("reading the n-gram table: {e}"));
                (&fetched, (0..rows.len()).collect())
            }
            None => (
                table
                    .host_bytes()
                    .expect("per_layer_token_embd.weight must be in host memory"),
                rows.iter().map(|&r| r as usize).collect(),
            ),
        };
        // Big steps dequantize on several threads; with the weights' own memory they also wait
        // on page faults at once.
        let mut out = vec![0f32; rows.len() * dim];
        let threads = ple_gather_threads(rows.len(), self.ngram.is_some());
        let per = rows.len().div_ceil(threads).max(1);
        let fp8 = ple.fp8_scale;
        let gather = |at: &[usize], out: &mut [f32]| {
            for (&r, dst) in at.iter().zip(out.chunks_mut(dim)) {
                let src = &bytes[r * row_bytes..][..row_bytes];
                dst.copy_from_slice(&ple_row_f32(ty, fp8, src, dim));
            }
        };
        if threads == 1 {
            // A decode step's few rows: starting a thread costs more than the work.
            gather(&at, &mut out);
        } else {
            // Off the engine thread's core (#250).
            let pin = &crate::hw::host_core::off_host_core(&[]);
            std::thread::scope(|scope| {
                for (at, out) in at.chunks(per).zip(out.chunks_mut(per * dim)) {
                    scope.spawn(move || {
                        crate::hw::host_core::pin_helper(pin);
                        gather(at, out)
                    });
                }
            });
        }
        out
    }
}

/// The n-gram table rows of `groups`' tokens, in order: each sequence's tokens hashed with the
/// predecessors `history(group, slot, tokens)` returns for its slot (oldest first).
fn ngram_rows(
    ple: &PleParams,
    groups: &[Group<'_>],
    mut history: impl FnMut(&Group<'_>, usize, &[u32]) -> Vec<u32>,
) -> Vec<i32> {
    let mut rows = Vec::new();
    for g in groups {
        for (i, s) in g.seqs.iter().enumerate() {
            let mut prev = history(g, g.slot + i, s.tokens);
            for &tok in s.tokens {
                ple.rows(tok, &prev, &mut rows);
                prev.remove(0);
                prev.push(tok);
            }
        }
    }
    rows
}

impl Model {
    /// For a sparse-attention layer: caches this step's raw indexer keys (with a zero pool-key
    /// slot, as llama.cpp writes them) and returns every token's indexer query
    /// `[d, heads, n]`, normed and rotated. None for dense layers.
    fn qsa_prepare<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let Some(q) = self.qwen4().qsa.as_ref().filter(|q| q.layer(il)) else {
            return Ok(None);
        };
        let (ctx, n) = (pass.ctx, pass.n);
        let d = q.head_dim;
        // ggml's matmuls; from the F16 copy of `h` the mixer wrote, where it did (#774), or the
        // BF16 cast of `h` the step's projections share, where there is one (#1176: a BF16
        // all-gather of `h`, so the widened `h` need not be written for these).
        let mm = |w: Tensor<'a>| match crate::model::core::f16_joined(ctx, w, h) {
            Some((w16, h16)) => ctx.mul_mat(w16, h16),
            None => match crate::model::core::bf16_cast_joined(ctx, w, h) {
                Some(hb) => ctx.mul_mat(w, hb),
                None => ctx.mul_mat(w, h),
            },
        };
        let k_raw = mm(self.lw(il, "indexer.k_proj.weight")?);
        let direct = pass
            .kv
            .direct_idx(il)
            .filter(|&c| norm_rope::idx_append_fits(k_raw, c, pass.cells));
        if let Some(cache) = direct {
            // One launch, the same bits.
            let node = norm_rope::idx_append(ctx, k_raw, cache, pass.cells);
            pass.graph.borrow_mut().expand(node);
        } else {
            let zeros = ctx.fill(ctx.new_tensor(types::F32, &[d, n]), 0.0);
            let rows = ctx.concat(k_raw, zeros, 0);
            pass.kv
                .idx_write(ctx, &mut pass.graph.borrow_mut(), il, rows, pass.cells);
        }

        let qi = mm(self.lw(il, "indexer.q_proj.weight")?);
        let qi = ctx.reshape_3d(qi, d, q.n_head, n);
        let w = self.lw(il, "indexer.q_norm.weight")?;
        Ok(Some(self.norm_rope(ctx, qi, w, pass.pos)))
    }

    /// Group `gi`'s pool keys in sparse-attention layer `il`: those of the pools this step
    /// completes, computed and cached (the graph expands their writes). Returns the pool-key
    /// cache rows.
    fn qsa_pool_keys<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
    ) -> Result<Tensor<'a>, ModelError> {
        let hp = &self.hp;
        let q = self.qwen4().qsa.as_ref().expect("sparse attention");
        let ctx = pass.ctx;
        let inp = QsaInputs::from_slice(&pass.group[gi]);
        let (d, kpool) = (q.head_dim, q.kpool as i64);
        let n_new = pass.groups().nth(gi).expect("group index").1.n_new as i64;
        let (raw, pooled) = pass.kv.idx_rows(ctx, il);

        // Pool keys of the pools this step completes: mean of the members' raw keys, normed,
        // rotated to the first member's position; stored in the last member's row. One launch
        // where it fits (the same bits as the chain; the graph runs its nodes in order, so the
        // keys are in place before any later node reads them).
        let k_norm = self.lw(il, "indexer.k_norm.weight")?;
        let rope = self.rope_params();
        let (members, new_pos, new_rep) = (inp.new_idxs, inp.new_pos, inp.new_rep);
        let written =
            if norm_rope::pool_key_fits(raw, pooled, members, new_pos, new_rep, k_norm, &rope) {
                norm_rope::pool_key(
                    ctx, raw, pooled, members, new_pos, new_rep, k_norm, hp.rms_eps, &rope,
                )
            } else {
                let rows = ctx.reshape_3d(ctx.get_rows(raw, members), d, kpool, n_new);
                let rnb = rows.nb();
                let slice = |i: i64| ctx.view_2d(rows, [d, n_new], rnb[2], i as usize * rnb[1]);
                let mut sum = ctx.cont(slice(0));
                for i in 1..kpool {
                    sum = ctx.add(sum, slice(i));
                }
                let pk = ctx.scale(sum, 1.0 / kpool as f32);
                let pk = ctx.mul(ctx.rms_norm(pk, hp.rms_eps), k_norm);
                let pk = ctx.reshape_3d(pk, d, 1, n_new);
                let pk = ctx.reshape_2d(self.rope(ctx, pk, new_pos), d, n_new);
                ctx.set_rows(pooled, pk, new_rep)
            };
        pass.graph.borrow_mut().expand(written);
        Ok(pooled)
    }

    /// Group `gi`'s pool selection in sparse-attention layer `il`: computes and caches the pool
    /// keys this step completes, scores every listed pool with the indexer queries `iq` and
    /// picks the best `top_k / kpool`. Returns the scores (F32 `[n_pool, tokens]`, -inf for
    /// pools a token does not see) and the picks (I32 `[n_top, tokens]`, in no order). Follows
    /// llama.cpp's `build_qsa_sel` (src/models/qwen4exp.cpp) op for op, except that with
    /// `own_top` (the CUDA paged path) the scores come from [`qsa_attn::scores_exact`] (ggml's
    /// bits, one op) and the picks from [`qsa_attn::top_k`] (one launch; the same pools but for
    /// exact ties at the threshold; `legion serve --qsa-topk-ggml` keeps ggml's `top_k`).
    fn qsa_top<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
        iq: Tensor<'a>,
        own_top: bool,
    ) -> Result<(Tensor<'a>, Tensor<'a>), ModelError> {
        let (off, sh) = pass.groups().nth(gi).expect("group index");
        let q = self.qwen4().qsa.as_ref().expect("sparse attention");
        let ctx = pass.ctx;
        let inp = QsaInputs::from_slice(&pass.group[gi]);
        let (d, kpool) = (q.head_dim, q.kpool as i64);
        let n_pool = sh.n_pool as i64;
        let ng = sh.tokens() as i64;
        let pooled = self.qsa_pool_keys(pass, il, gi)?;
        let keys = || ctx.reshape_3d(ctx.get_rows(pooled, inp.pool_cells), d, 1, n_pool);

        // Score: unweighted sum of the rectified head scores, scaled by 1/sqrt(d).
        let qnb = iq.nb();
        let qg = ctx.view_3d(
            iq,
            [d, q.n_head, ng],
            [qnb[1], qnb[2]],
            off as usize * qnb[2],
        );
        let w = ctx.fill(
            ctx.new_tensor(types::F32, &[q.n_head, ng]),
            1.0 / (d as f32).sqrt(),
        );
        let score = if qsa_attn::scores_fit(q.n_head, d, ng) {
            // On Volta's tensor cores (from Strata-AC922 52a35fe): the key gather, mask and
            // bias in the op.
            qsa_attn::scores(
                ctx,
                qg,
                pooled,
                w,
                inp.pool_vis,
                Some(inp.pool_bias),
                Some(inp.pool_cells),
            )
        } else if own_top && qsa_attn::scores_exact_fit(q.n_head, d, ng) {
            // ggml's CUDA scores, mask and bias, bit for bit, in one op.
            qsa_attn::scores_exact(ctx, qg, keys(), w, inp.pool_vis, inp.pool_bias)
        } else {
            let mask = qsa_attn::pool_mask(ctx, inp.pool_vis, n_pool);
            ctx.add(ctx.lightning_indexer(qg, keys(), w, mask), inp.pool_bias)
        };
        pass.tap.tap("indexer_score", il, score);

        let n_top = n_pool.min(q.top_k as i64 / kpool);
        let top = if own_top && qsa_attn::own_top_k() {
            if crate::model::cluster_select::takes_top_k(score, n_top) {
                // Decode steps: the same picks on thread-block clusters (#554).
                crate::model::cluster_select::top_k(ctx, score, n_top)
            } else {
                qsa_attn::top_k(ctx, score, n_top)
            }
        } else {
            ctx.top_k(score, n_top as i32)
        };
        Ok((score, top))
    }

    /// Group `gi`'s selected positions in sparse-attention layer `il`, for the paged op (see
    /// [`qsa_attn::select`]).
    fn qsa_select<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
        iq: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (off, sh) = pass.groups().nth(gi).expect("group index");
        let inp = QsaInputs::from_slice(&pass.group[gi]);
        let es = pass.pos.element_size();
        // The first section of the positions is the token's position.
        let ng = sh.tokens();
        let pos = pass
            .ctx
            .view_2d(pass.pos, [ng as i64, 1], ng * es, off as usize * es);
        if sh.pick_all && !pass.tap.on {
            // Every token attends to every pool it sees: no indexer query, scores or top_k
            // (the same selection; the pool keys are still cached for later steps).
            let q = self.qwen4().qsa.as_ref().expect("sparse attention");
            self.qsa_pool_keys(pass, il, gi)?;
            let n_top = (sh.n_pool as i64).min((q.top_k / q.kpool) as i64);
            return Ok(qsa_attn::select_all(
                pass.ctx,
                inp.pool_vis,
                n_top,
                inp.pool_idxs,
                inp.tail_idxs,
                pos,
            ));
        }
        let (score, top) = self.qsa_top(pass, il, gi, iq, true)?;
        Ok(qsa_attn::select(
            pass.ctx,
            top,
            score,
            inp.pool_idxs,
            inp.tail_idxs,
            pos,
        ))
    }

    /// The attention mask of group `gi` in sparse-attention layer `il` for the gather path: its
    /// causal mask `mask` plus -inf on every key not selected ([`Self::qsa_top`]'s pools plus
    /// the token's incomplete pool), as llama.cpp's `build_qsa_sel` builds it.
    fn qsa_mask<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
        iq: Tensor<'a>,
        mask: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (score, top) = self.qsa_top(pass, il, gi, iq, false)?;
        let (_, sh) = pass.groups().nth(gi).expect("group index");
        let q = self.qwen4().qsa.as_ref().expect("sparse attention");
        let ctx = pass.ctx;
        let inp = QsaInputs::from_slice(&pass.group[gi]);
        let kpool = q.kpool as i64;
        let n_pool = sh.n_pool as i64;
        let ng = sh.tokens() as i64;
        let n_kv = sh.n_kv as i64;
        let n_top = top.ne()[0];
        // The top pools' members, then the tail (n_kv for missing cells).
        let sel_idx = ctx.get_rows(inp.pool_idxs, ctx.reshape_1d(top, n_top * ng));
        let sel_idx = ctx.reshape_2d(sel_idx, kpool * n_top, ng);
        let sel_idx = ctx.concat(sel_idx, inp.tail_idxs, 0);
        let n_sel = sel_idx.ne()[0];

        // Scatter zeros for the selected cells into an all -inf row; every dead slot (an
        // invisible pool, a missing tail cell) goes to its own dump row n_kv + slot.
        let mask_all = ctx.fill(
            ctx.new_tensor(types::F32, &[n_kv + n_sel, 1, 1, 1]),
            f32::NEG_INFINITY,
        );
        let mask_all = ctx.repeat_4d(mask_all, [n_kv + n_sel, ng, 1, 1]);
        let mask_all = ctx.reshape_3d(mask_all, 1, n_kv + n_sel, ng);
        let zeros = ctx.fill(ctx.new_tensor(types::F32, &[n_sel, 1, 1, 1]), 0.0);
        let zeros = ctx.reshape_3d(ctx.repeat_4d(zeros, [n_sel, ng, 1, 1]), 1, n_sel, ng);
        let top_score = ctx.get_rows(ctx.reshape_3d(score, 1, n_pool, ng), top);
        let live_pool = ctx.clamp(ctx.scale_bias(top_score, 1.0, 1.0), 0.0, 1.0);
        let live_pool = ctx.reshape_2d(
            ctx.repeat_4d(live_pool, [kpool, n_top, ng, 1]),
            kpool * n_top,
            ng,
        );
        let live_tail = ctx.cast(inp.tail_idxs, types::F32);
        let live_tail = ctx.clamp(ctx.scale_bias(live_tail, -1.0, n_kv as f32), 0.0, 1.0);
        let live = ctx.concat(live_pool, live_tail, 0);
        let dump = ctx.scale_bias(ctx.cumsum(ctx.fill(live, 1.0)), 1.0, (n_kv - 1) as f32);
        let idx_f = ctx.cast(sel_idx, types::F32);
        let idx_f = ctx.add(ctx.mul(ctx.sub(idx_f, dump), live), dump);
        let sel_idx = ctx.cast(idx_f, types::I32);
        let sel = ctx.set_rows(mask_all, zeros, ctx.reshape_3d(sel_idx, n_sel, ng, 1));
        let row = sel.nb()[2];
        let [_, mt, m2, m3] = mask.ne();
        let sel = ctx.view_4d(
            sel,
            [n_kv, mt, m2, m3],
            [row, row * mt as usize, row * (mt * m2) as usize],
            0,
        );
        Ok(ctx.add(sel, mask))
    }
}

impl Model {
    /// The qwen4exp hyperparameters of this model.
    fn qwen4(&self) -> &Qwen4Params {
        params(&self.hp).expect("a qwen4exp model")
    }
}

impl Qwen4Params {
    /// The sparse-attention tables of a group's inputs (the first ones).
    fn qsa_inputs_len(&self) -> usize {
        if self.qsa.is_some() { 8 } else { 0 }
    }

    /// The MoE row map of a pass that is one verify group ([`pad_rows`]); `None` otherwise.
    fn moe_rows<'a>(&self, pass: &Pass<'a>) -> Option<Tensor<'a>> {
        match (pass.group.as_slice(), pass.shape) {
            ([g], [sh]) if sh.verify => g.get(self.qsa_inputs_len()).copied(),
            _ => None,
        }
    }
}

impl Arch for Qwen4Params {
    fn name(&self) -> &'static str {
        "qwen4exp"
    }

    fn default_spec_k(&self) -> Option<usize> {
        // MTP verify steps read the experts of every drafted token; two drafts performed
        // better than one, three or four in measured chats.
        Some(2)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn same(&self, other: &dyn Arch) -> bool {
        same_as(self, other)
    }

    fn is_recurrent(&self, hp: &Hparams, il: usize) -> bool {
        crate::model::qwen35::is_delta_net(hp, il)
    }

    fn state_captures(&self) -> bool {
        true
    }

    fn sigmoid_gate(&self) -> bool {
        true
    }

    fn residual_streams(&self) -> i64 {
        self.hc.n
    }

    fn token_rows(&self, _hp: &Hparams, layers: &Range<usize>) -> i64 {
        self.ple
            .as_ref()
            .filter(|p| layers.contains(&p.layer))
            .map_or(0, |p| p.head_dim * p.n_heads() as i64)
    }

    fn token_conv(&self, hp: &Hparams) -> Option<TokenConv> {
        self.ple.as_ref().map(|p| TokenConv {
            layer: p.layer,
            history: p.history(),
            width: self.hc.n * hp.n_embd,
            n_hist: p.ngram - 1,
        })
    }

    fn ngram_table(&self) -> bool {
        self.ple.is_some()
    }

    /// The sparse-attention tables, then the MoE row map ([`pad_rows`], I32 `[tokens]`).
    fn group_inputs_len(&self) -> usize {
        self.qsa_inputs_len() + pad_rows::on() as usize
    }

    fn group_inputs<'a>(&self, ctx: &'a Context, g: &GroupShape, i: usize) -> Vec<Tensor<'a>> {
        let mut v = match &self.qsa {
            Some(q) => QsaInputs::new(ctx, q.kpool, g, i).list().to_vec(),
            None => Vec::new(),
        };
        if pad_rows::on() {
            let rows = ctx.new_tensor(types::I32, &[g.tokens() as i64]);
            v.push(rows.named(&format!("moe_rows-{i}")).as_input());
        }
        v
    }

    fn verify_row_map<'a>(&self, pass: &Pass<'a>) -> Option<Tensor<'a>> {
        self.moe_rows(pass)
    }

    fn group_pools(&self, g: &Group<'_>) -> (usize, usize) {
        let kpool = self.qsa.as_ref().map_or(0, |q| q.kpool);
        (
            step_pools_listed(g, kpool, STEP_POOL_STEPS.load(Ordering::Relaxed)),
            step_pools_new(g, kpool),
        )
    }

    /// A token sees at most its sequence's complete pools, `(n_past + n_tok) / kpool` at the
    /// step's end; when no sequence has more than the indexer picks (`top_k / kpool`: contexts
    /// up to `top_k` positions), `top_k` keeps every visible pool.
    fn group_picks_all(&self, g: &Group<'_>) -> bool {
        let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
        self.qsa.as_ref().is_some_and(|q| {
            qsa_attn::pick_all()
                && q.kpool > 0
                && g.seqs
                    .iter()
                    .all(|s| (s.n_past + n_tok) / q.kpool <= q.top_k / q.kpool)
        })
    }

    /// The sparse-attention tables and the per-layer embedding rows.
    fn fill_inputs(
        &self,
        m: &Model,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        rec: &StatePool,
        kv: &KvPool,
        spec: Option<&SpecPool>,
        sink: &mut dyn InputSink,
    ) {
        if let Some(q) = &self.qsa {
            for ((g, sh), t) in groups.iter().zip(shape).zip(&inputs.group) {
                QsaInputs::from_slice(t).fill(q.kpool, g, sh, kv, sink);
            }
        }
        for ((g, sh), t) in groups.iter().zip(shape).zip(&inputs.group) {
            if let Some(&rows) = t.get(self.qsa_inputs_len()) {
                let map = if sh.verify {
                    pad_rows::map(g.seqs.iter().map(|s| s.tokens), sh.n_tok)
                } else {
                    (0..sh.tokens() as i32).collect()
                };
                sink.i32(rows, &map);
            }
        }
        m.fill_ple(inputs, groups, rec, spec, sink);
    }

    fn index_dim(&self) -> i64 {
        self.qsa.as_ref().map_or(0, |q| q.head_dim)
    }

    fn indexed(&self, il: usize) -> bool {
        self.qsa.as_ref().is_some_and(|q| q.layer(il))
    }

    /// The sparse attention kernels read F16, or q8_0 / q4_0 K and V (#263); dense attention
    /// layers (none in Qwen3.8-Flash-Next) what the paged kernels read.
    fn sparse_reads(&self, hp: &Hparams, types: crate::model::core::KvTypes) -> bool {
        let dense = (0..hp.n_layer)
            .filter(|&il| hp.is_attention(il))
            .any(|il| !self.qsa.as_ref().is_some_and(|q| q.layer(il)));
        qsa_attn::reads_types(types) && (!dense || crate::model::core::paged::reads_types(types))
    }

    fn sparse_keys(&self) -> Option<(usize, usize)> {
        self.qsa.as_ref().map(|q| (q.top_k, q.kpool))
    }

    /// Sparse attention decides per step ([`Self::paged_for`]), so with it this is false.
    fn paged(&self, m: &Model, kv: &KvPool) -> bool {
        if self.qsa.is_some() {
            return false;
        }
        m.paged_default(kv)
    }

    /// For sparse attention the paged path is the [`qsa_attn`] op, which reads each query's
    /// selected keys only; the gather path computes dense attention over the span and masks
    /// it, which needs `span × tokens × heads` scores (gigabytes per layer for a decode batch
    /// at long context). With [`Attention::Auto`] every step pages on CUDA: prompt chunks one
    /// CTA per token and kv head, decode and verify steps with each token's selection split
    /// over several CTAs (see [`qsa_attn::attention`]).
    fn paged_for(&self, m: &Model, kv: &KvPool, _groups: &[Group<'_>]) -> bool {
        let hp = &m.hp;
        let Some(q) = self.qsa.as_ref() else {
            return self.paged(m, kv);
        };
        match m.attention_mode() {
            Attention::Gather => false,
            Attention::Paged => true,
            Attention::Auto => {
                // Dense attention layers (if any) would take the paged kernels.
                let layers_ok = (0..hp.n_layer).filter(|&il| hp.is_attention(il)).all(|il| {
                    q.layer(il)
                        || (crate::model::core::paged::supported(
                            hp.head_dim,
                            hp.n_head,
                            hp.n_head_kv,
                        ) && crate::model::core::paged::reads_types(kv.types()))
                });
                cfg!(feature = "cuda")
                    && !kv.is_host()
                    && layers_ok
                    && qsa_attn::supported(hp.head_dim, hp.n_head, hp.n_head_kv)
                    && q.top_k / q.kpool <= qsa_attn::CUDA_MAX_TOP
            }
        }
    }

    fn index_query<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        m.qsa_prepare(pass, il, h)
    }

    fn sparse_paged<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
        iq: Tensor<'a>,
        [q, k, v]: [Tensor<'a>; 3],
        table: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let sel = m.qsa_select(pass, il, gi, iq)?;
        Ok(qsa_attn::attention(pass.ctx, q, k, v, table, sel))
    }

    fn sparse_mask<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        il: usize,
        gi: usize,
        iq: Tensor<'a>,
        mask: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        m.qsa_mask(pass, il, gi, iq, mask)
    }

    /// The hyper-connection stack ([`Model::hc_layers`]); a capture is the residual streams
    /// of the last layer, which is what the MTP head reads.
    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError> {
        let n_layer = m.hp.n_layer;
        assert!(
            capture.iter().all(|&il| il + 1 == n_layer),
            "hyper-connection models capture the last layer only"
        );
        let last = layers.end == n_layer;
        let (out, all) = m.hc_layers(pass, x, last && !capture.is_empty(), layers)?;
        Ok((out, all.map(|r| (n_layer - 1, r)).into_iter().collect()))
    }

    /// The last layer ends in the final mixer over the output rows.
    fn output_ready(&self) -> bool {
        true
    }

    /// Two DGX Sparks, the NVFP4 checkpoint's BF16 head: the same text split, 1 to 16 users
    /// (#1176).
    fn head_splits(&self) -> bool {
        true
    }

    fn mtp(&self) -> bool {
        true
    }

    fn mtp_head(&self, hp: &Hparams, gguf: &GgufFile) -> Result<ArchRef, ModelError> {
        self.check_mtp_head(hp, gguf)
    }

    fn build_mtp<'a>(
        &self,
        m: &'a Model,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        m.build_mtp_qwen4(ctx, kv, rec, shape)
    }

    fn external_drafters(&self) -> bool {
        false
    }
}
