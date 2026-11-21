//! DFlash: a block-diffusion drafter (Chen, Liang and Liu, "DFlash: Block Diffusion for Flash
//! Speculative Decoding", <https://arxiv.org/abs/2602.06036>). `--spec dflash --spec-draft
//! <checkpoint>`.
//!
//! # How it drafts
//!
//! A small decoder (a few Qwen3-style layers) is conditioned on the target's hidden states from
//! several of the target's layers (named in the drafter's config) and proposes a whole block of
//! tokens in **one** forward pass, instead of one token per pass:
//!
//! 1. The target's hidden states at the fused layers are concatenated per position, projected
//!    by `fc` and RMS-normed (`hidden_norm`). Each drafter layer projects them to K and V and
//!    keeps them in a per-sequence cache ([`cache`]), at the position's RoPE angle.
//! 2. To draft, the noise block `[last token, <mask>, <mask>, ...]` runs through the layers at
//!    the positions after the context. Its attention is not causal: every block position
//!    attends to the cached context and to the whole block, so all masks are denoised at once.
//! 3. The token embeddings and the output head are the target's (loaded from the target GGUF;
//!    a drafter that ships its own, with a reduced vocabulary and a `d2t` map to target ids, is
//!    supported too). The output at block position `i >= 1` is the distribution of draft `i`.
//!    The target's head can be cut down for drafting: its first rows only, or a 4-bit copy
//!    ([`head`], `--spec-draft-vocab`, `--spec-draft-lm-head`). An FP8 head kept raw runs on
//!    Legion's FP8 matmul ([`crate::model::fp8`]).
//!
//! The engine hands the drafter every target step's hidden states ([`ModelDrafter::step`]):
//! prefill chunks fill the cache; after a verify step only the accepted positions arrive, and
//! rows the drafter had seen for positions the target then rejected are dropped from its cache
//! (rollback: the call's `start` is below what the cache holds). The cache is a ring of
//! `window` positions per sequence ([`DflashOptions::window`]), so a very long sequence keeps
//! only its newest context for drafting.
//!
//! # Exactness
//!
//! The drafts are proposals; the core verifies them with the exact acceptance rule
//! ([`crate::spec::accept`]), so the output is the target's own whatever the drafter does.
//! With [`DflashOptions::temperature`] 0 (the default) a draft is the position's most likely
//! token, a point mass (greedy requests accept most of those that match the target's pick, and
//! point masses are exact for sampling requests too). With a temperature above 0 each draft is
//! sampled from its position's softmax and the full distribution is returned in
//! [`Draft::probs`] for exact rejection sampling; the positions are independent (nothing in the
//! block depends on the tokens sampled before it), so each `q_i` is the distribution `d_i` was
//! drawn from.
//!
//! # Checkpoints
//!
//! - GGUF (llama.cpp's `dflash` architecture, from `convert_hf_to_gguf.py`): any quantization
//!   ggml reads.
//! - Hugging Face directory (`config.json` of a `DFlashDraftModel` + safetensors): F32, F16 or
//!   BF16.
//!
//! DFlash 2 (`DFlash2DraftModel`: dynamic convolutions around every attention and MLP, and a
//! candidate selector that chooses the block's drafts in order, [`v2`]) is recognized by its
//! config (`conv_kernel_size`, ...; in a GGUF `dflash.conv_kernel_size`, ...) and run by the
//! same `--spec dflash`. It always runs the whole block. The LiLiCorr and DeepSeek-V4 variants
//! of the `dflash` GGUF architecture are refused with an error.
//! A DSpark drafter (Markov head, [`crate::spec::dspark`]) is the same network and the same
//! architecture name; it is run by `--spec dspark`, and `--spec dflash` says so.
//!
//! # Known result
//!
//! On the AC922 with Nemotron as the target, DFlash's acceptance was 15-30% and decode got
//! slower, not faster: a block pass costs more than it wins at that acceptance. This module
//! only implements the drafter correctly; whether and when to use it is for load-adaptive
//! control (#48).
//!
//! Adapted from llama.cpp's `src/models/dflash.cpp` and `common/speculative.cpp` (MIT) for the
//! network, the block layout and the tensor names.

mod cache;
mod config;
mod fused;
mod head;
mod model;
pub mod tiny;
mod tp_head;
mod v2;
mod weights;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_device;
#[cfg(test)]
mod tests_dspark;
#[cfg(test)]
mod tests_head;
#[cfg(test)]
mod tests_v2;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engine::sampling::Rng;
use crate::ggml::{Backend, ComputeAllocator, Context, Event, gpu_device, types};
use crate::spec::{
    DeviceHidden, Draft, DraftRequest, ModelDrafter, ModelDrafterFactory, SeqId, TargetInfo,
};

pub use config::{Config, ConfigError, Dflash2, Flavor};
use model::{BlockSeq, Feats, INGEST_CHUNK, Model, Rows};
use weights::{Source, load_from_gguf};

/// The take-in's features: `feats` on the host, or rows `idx` of the step's rows on the device.
fn feats_of<'f>(src: Src<'f>, feats: &'f [f32], idx: &'f [i32]) -> Feats<'f> {
    match src {
        Src::Host => Feats::Host(feats),
        Src::Device(d) => Feats::Device {
            layers: d.layers,
            rows: idx,
        },
    }
}

/// Where a step's captured rows are: in the requests' [`DraftRequest::hidden`], or left on the
/// device ([`ModelDrafter::step_device`], #1181).
#[derive(Clone, Copy)]
enum Src<'s> {
    Host,
    Device(&'s DeviceHidden<'s>),
}

/// Settings of a [`DflashDrafter`] beyond the checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DflashOptions {
    /// Positions of context a sequence keeps for drafting (a ring: the newest).
    pub window: usize,
    /// 0: drafts are the most likely tokens (point masses). Above: drafts are sampled at this
    /// temperature and their distributions returned.
    pub temperature: f32,
    /// Seed of the draws when sampling.
    pub seed: u64,
    /// DSpark with a confidence head: a draft block is cut before the first position whose
    /// estimated acceptance is below this (0: never cut).
    pub min_confidence: f32,
    /// The weights it drafts with: as stored, or Q8_0 copies of the matrices
    /// (`--spec-draft-weights`).
    pub weights: crate::spec::DraftWeights,
    /// DFlash with the target's output head: score drafts over its first this many rows only
    /// (`--spec-draft-vocab`; 0: all). The Qwen vocabulary is roughly frequency ordered, so
    /// the first rows are the shortlist.
    pub draft_vocab: usize,
    /// DFlash with the target's output head: those rows as stored or as a 4-bit copy
    /// (`--spec-draft-lm-head`).
    pub lm_head: crate::spec::DraftLmHead,
    /// DFlash 2: the batched draft pass attends every sequence in one flash attention per
    /// layer, over caches that share one pool (`--spec-draft-batched-attn`).
    pub batched_attn: bool,
    /// Tensor parallelism over machines: this rank's share of the drafter's FFN
    /// (`--tp-draft-split`); `None`: all of it.
    pub tp: Option<TpShare>,
}

/// A DFlash drafter's FFN split over the ranks of tensor parallelism over machines
/// (`--tp-draft-split`, #619). Each rank holds its share of every layer's intermediate rows
/// (gate and up rows, down columns) and the rest whole; a draft pass adds the ranks' FFN
/// outputs with one all-reduce per layer, so every rank's residual stream, and with it the
/// drafter's caches, stays the same. Every rank runs every take-in and draft pass; rank 0
/// alone computes the head and picks the drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpShare {
    /// The communicator of the all-reduces ([`crate::tp_net::register`]).
    pub comm: i32,
    pub rank: usize,
    pub n_ranks: usize,
}

impl TpShare {
    /// This rank's intermediate rows of `n_ff`, or why they do not split: every share whole
    /// Q8_0 blocks (32 values) of the down projection's rows.
    pub fn rows(&self, n_ff: usize) -> Result<std::ops::Range<usize>, String> {
        if self.rank >= self.n_ranks || !n_ff.is_multiple_of(32 * self.n_ranks) {
            return Err(format!(
                "--tp-draft-split: the drafter's {n_ff} intermediate rows do not split into {} \
                 shares of whole 32-value blocks",
                self.n_ranks
            ));
        }
        let share = n_ff / self.n_ranks;
        Ok(self.rank * share..(self.rank + 1) * share)
    }

    /// Whether this rank computes the head and picks the drafts (rank 0).
    pub fn leads(&self) -> bool {
        self.rank == 0
    }
}

impl Default for DflashOptions {
    fn default() -> Self {
        Self {
            window: 4096,
            temperature: 0.0,
            seed: 0,
            min_confidence: 0.0,
            weights: Default::default(),
            draft_vocab: 0,
            lm_head: Default::default(),
            batched_attn: false,
            tp: None,
        }
    }
}

/// `--spec dflash`: makes the engine's drafter from `--spec-draft`.
pub fn factory(opts: DflashOptions) -> ModelDrafterFactory {
    Arc::new(move |info: &TargetInfo| {
        DflashDrafter::load(info, opts.clone()).map(|d| Box::new(d) as Box<dyn ModelDrafter>)
    })
}

pub struct DflashDrafter {
    model: Model,
    backend: Backend,
    alloc: Option<ComputeAllocator>,
    /// DFlash 2's batched draft pass's allocator: built while the take-in graph, from
    /// `alloc`, still runs (#1181).
    draft_alloc: Option<ComputeAllocator>,
    opts: DflashOptions,
    rng: Rng,
    n_embd: usize,
    n_target_vocab: usize,
    caches: HashMap<SeqId, cache::SeqCache>,
    /// Caches of finished sequences, to reuse.
    spare: Vec<cache::SeqCache>,
    /// `--spec-draft-batched-attn`: the slots the sequences' caches live in, made on first use.
    /// Declared after `caches`: their tensors are views of it.
    pool: Option<cache::CachePool>,
    /// The most sequences the engine runs at once: the pool's slots.
    max_seqs: usize,
    /// The pool could not be allocated: every sequence has a cache of its own.
    pool_failed: bool,
    /// The CUDA device its backend runs on (`None`: the CPU): a step's rows left in that
    /// device's memory are gathered by its take-in graph ([`ModelDrafter::step_device`]).
    device: Option<usize>,
    /// The host features a take-in fills ([`Self::features`]), kept from one take-in to the
    /// next and grown to the most it needed (#1133). A prompt's piece of [`INGEST_CHUNK`] rows
    /// of every fused layer is tens of MB (512 × 5 × 5,376 floats, 55 MB), more than glibc
    /// keeps on its heap: a fresh vector was mapped, faulted in and unmapped every prompt.
    feats: Vec<f32>,
    /// A take-in of several pieces queues each piece but the last without waiting for it, on
    /// `alloc` and `draft_alloc` in turn ([`ingest_pipe`], #1133).
    pipe: bool,
    /// A take-in of host rows copies them into its graph's input straight from the captured
    /// layers, without gathering them in `feats` ([`ingest_rows`], #1133).
    direct: bool,
    /// Recorded after a queued take-in piece on `alloc` (0) or `draft_alloc` (1), made on first
    /// use; none on the CPU.
    piece_done: Vec<Event>,
}

/// Most idle caches kept for the next sequences.
const SPARE_CACHES: usize = 4;

/// MiB of an allocator's buffer (0 for none).
fn mib(a: &Option<ComputeAllocator>) -> usize {
    a.as_ref().map_or(0, |a| a.buffer_size() >> 20)
}

/// Where a take-in's host time went ([`DflashDrafter::ingest`], logged at debug level).
#[derive(Default)]
struct TakeInTimes {
    pieces: usize,
    /// Gathering host rows into the features buffer.
    gather: Duration,
    /// Writing the pieces' features inputs (host to device).
    upload: Duration,
    /// Waiting for a piece queued before to free its allocator.
    wait: Duration,
}

/// A take-in's host features go to the buffer the drafter keeps ([`DflashDrafter::feats`]);
/// `LEGION_DEBUG_DFLASH_KEEP_FEATS=0`: a fresh vector every take-in, as before (A/B runs only;
/// the same values either way).
fn keep_feats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_DFLASH_KEEP_FEATS").is_none_or(|v| v != "0"))
}

/// A take-in of several pieces queues each piece but the last without waiting for it, the
/// pieces taking two allocators in turn, so that a piece's input is written while the piece
/// before runs (#1133); `LEGION_DEBUG_DFLASH_INGEST_PIPE=0`: waits for every piece, one
/// allocator, as before (A/B runs only; the same graphs in the same order on the same stream,
/// so the same bits either way).
fn ingest_pipe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_DFLASH_INGEST_PIPE").is_none_or(|v| v != "0"))
}

/// A take-in of host rows copies each captured layer's rows into its graph's input, one
/// strided copy each (#1133); `LEGION_DEBUG_DFLASH_INGEST_ROWS=0`: gathers them on the host
/// first and uploads the gathered rows, as before (A/B runs only; the same floats either way).
fn ingest_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_DFLASH_INGEST_ROWS").is_none_or(|v| v != "0"))
}

/// Empties `buf` with room for `n` values. One too small is replaced, not grown: growing would
/// copy the old allocation, whose values are not needed. It never shrinks.
fn empty_with_room(buf: &mut Vec<f32>, n: usize) {
    buf.clear();
    if buf.capacity() < n {
        // The old one goes first: the two are never held together.
        *buf = Vec::new();
        *buf = Vec::with_capacity(n);
    }
}

impl DflashDrafter {
    /// Loads the checkpoint at `target.draft` onto the target's device, and the token embeddings
    /// and output head from the target's GGUF if the checkpoint has none.
    pub fn load(target: &TargetInfo, opts: DflashOptions) -> Result<Self, String> {
        Self::load_as(target, opts, Flavor::Dflash)
    }

    /// Loads a DFlash, or a DSpark drafter (the same network with a Markov head, sometimes a
    /// confidence head and attention sinks; see [`crate::spec::dspark`]).
    pub(crate) fn load_as(
        target: &TargetInfo,
        opts: DflashOptions,
        flavor: Flavor,
    ) -> Result<Self, String> {
        let name = match flavor {
            Flavor::Dflash => "dflash",
            Flavor::Dspark => "dspark",
        };
        let path = target
            .draft
            .as_deref()
            .ok_or_else(|| format!("--spec {name} needs --spec-draft <drafter checkpoint>"))?;
        let source = Source::open(path)?;
        let mut cfg = source
            .config(flavor)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // `--tp-draft-split`: this rank's intermediate rows of every layer's FFN.
        let ffn = match opts.tp {
            Some(_) if flavor != Flavor::Dflash => {
                return Err("--tp-draft-split: a DFlash drafter only".into());
            }
            Some(t) => {
                let rows = t.rows(cfg.n_ff)?;
                let cut = weights::FfnCut {
                    n_ff: cfg.n_ff,
                    rows,
                };
                cfg.n_ff = cut.rows.len();
                Some(cut)
            }
            None => None,
        };
        let names = source.tensor_names();
        if flavor == Flavor::Dflash
            && let Some(n) = names
                .iter()
                .find(|n| n.starts_with("markov_") || n.starts_with("conf_proj"))
        {
            return Err(format!(
                "{}: tensor `{n}` belongs to a DSpark drafter, use --spec dspark",
                path.display()
            ));
        }
        // Layer variants of the architecture (Gemma-style norms, hyper-connections, and the
        // convolutions and selector of a config that is not DFlash 2) that the network here does
        // not compute: refuse, never mis-draft.
        const OTHER_NETWORKS: [&str; 7] = [
            "post_attention_norm",
            "post_ffw_norm",
            "layer_output_scale",
            "rope_freqs",
            "lilicorr",
            "hc_",
            "attn_kv",
        ];
        const DFLASH2: [&str; 2] = ["_conv_", "selector_"];
        let other = |n: &str| {
            OTHER_NETWORKS.iter().any(|v| n.contains(v))
                || (cfg.v2.is_none() && DFLASH2.iter().any(|v| n.contains(v)))
        };
        if let Some(n) = names.iter().find(|n| other(n)) {
            return Err(format!(
                "{}: tensor `{n}` belongs to a drafter network that is not implemented \
                 (only the DFlash and DFlash 2 layers are)",
                path.display()
            ));
        }
        if let Some(&l) = cfg.target_layers.iter().find(|&&l| l >= target.n_layer) {
            return Err(format!(
                "the drafter reads target layer {l}, the target has {} layers",
                target.n_layer
            ));
        }
        let mut backend = if target.gpu {
            gpu_device()
                .ok_or("no CUDA device (is legion built with --features cuda?)")?
                .init()
                .map_err(|e| e.to_string())?
        } else {
            Backend::cpu().map_err(|e| e.to_string())?
        };
        backend.set_threads(target.threads.clamp(1, 8));
        backend.set_use_ref(target.use_ref);
        let buft = backend.buffer_type();
        if opts.weights != crate::spec::DraftWeights::Model && matches!(source, Source::Gguf(_)) {
            tracing::warn!(
                "--spec-draft-weights applies to Hugging Face checkpoints: the GGUF drafter \
                 is read as stored"
            );
        }
        let mut store = source.load_share(buft, opts.weights, ffn.as_ref())?;
        let own = |n: &str| names.iter().any(|t| t == n);
        // `--spec-draft-vocab` / `--spec-draft-lm-head` cut down the target's head a DFlash
        // borrows; a drafter's own head (often a reduced vocabulary with `d2t`) and DSpark's
        // Markov head, which is as wide as the head, stay as they are.
        let cut_head = opts.draft_vocab > 0 || opts.lm_head == crate::spec::DraftLmHead::Q4;
        let borrowed_head = flavor == Flavor::Dflash && !own("output.weight");
        if cut_head && !borrowed_head {
            tracing::warn!(
                "--spec-draft-vocab and --spec-draft-lm-head apply to a DFlash drafter that \
                 uses the target's output head: ignored"
            );
        }
        if cut_head && borrowed_head {
            if !own("token_embd.weight") {
                store.extend(load_from_gguf(
                    &target.model,
                    &[("token_embd.weight", "target.token_embd.weight")],
                    buft,
                )?);
            }
            store.extend(head::load(
                &target.model,
                opts.draft_vocab,
                opts.lm_head,
                buft,
            )?);
        } else if !own("token_embd.weight") || !own("output.weight") {
            store.extend(load_from_gguf(
                &target.model,
                &[
                    ("token_embd.weight", "target.token_embd.weight"),
                    ("output.weight", "target.output.weight"),
                    // An NVFP4 head's per-tensor scale.
                    ("output.scale", "target.output.scale"),
                ],
                buft,
            )?);
        }
        // DFlash 2's selector codebooks stay on the host.
        let codebooks = match &cfg.v2 {
            Some(_) => {
                let book = |n: &str| {
                    source
                        .host_tensor(n)?
                        .ok_or_else(|| format!("the DFlash 2 drafter has no `{n}`"))
                };
                let books = (
                    book("selector_predecessor.weight"),
                    book("selector_successor.weight"),
                );
                match books {
                    (Ok(p), Ok(s)) => Some((p, s)),
                    (Err(e), _) | (_, Err(e)) => return Err(format!("{}: {e}", path.display())),
                }
            }
            None => None,
        };
        // No sequence is longer than the engine's context: its cache needs no more positions.
        let mut window = opts.window.min(target.ctx_size).max(cfg.block_size);
        // Every layer sliding: positions older than the window are never attended (a block on
        // top for the block's later positions, one more for rows a rollback overwrote).
        if let Some((w, pattern)) = &cfg.sliding_window
            && pattern.iter().all(|&s| s)
        {
            window = window.min(w + 2 * cfg.block_size);
        }
        let mut model = Model::new(
            cfg,
            store,
            target.n_embd,
            target.n_vocab,
            window,
            flavor,
            codebooks,
            cut_head && borrowed_head,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        model.set_tp(opts.tp);
        // The head split over the ranks too where it can (#666): greedy drafts only.
        if model.set_tp_head(opts.temperature == 0.0 && tp_head::enabled()) {
            tracing::info!("--tp-draft-split: the drafter's head split over the two ranks");
        }
        if opts.min_confidence > 0.0 && !model.has_confidence() {
            return Err(format!(
                "{}: the drafter has no confidence head to apply a minimum confidence of {}",
                path.display(),
                opts.min_confidence
            ));
        }
        tracing::info!(
            drafter = %path.display(),
            layers = model.cfg.n_layer,
            block_size = model.cfg.block_size,
            target_layers = ?model.cfg.target_layers,
            window,
            head_rows = model.n_vocab,
            markov = model.has_markov(),
            confidence = model.has_confidence(),
            "{name} drafter loaded"
        );
        // ggml-cuda names its devices `CUDA<n>`.
        let device = target
            .gpu
            .then(|| backend.device().name())
            .and_then(|n| n.strip_prefix("CUDA")?.parse().ok());
        Ok(Self {
            n_embd: target.n_embd,
            n_target_vocab: target.n_vocab,
            rng: Rng::new(opts.seed),
            model,
            device,
            backend,
            alloc: None,
            draft_alloc: None,
            opts,
            caches: HashMap::new(),
            spare: Vec::new(),
            pool: None,
            max_seqs: target.max_seqs.max(1),
            pool_failed: false,
            feats: Vec::new(),
            pipe: ingest_pipe(),
            direct: ingest_rows(),
            piece_done: Vec::new(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.model.cfg
    }

    /// Whether the checkpoint has a confidence head (DSpark).
    pub fn has_confidence(&self) -> bool {
        self.model.has_confidence()
    }

    /// The most tokens one pass drafts.
    pub fn max_drafts(&self) -> usize {
        self.model.cfg.block_size - self.model.first_row
    }

    fn cache_for(&mut self, seq: SeqId) -> Result<&mut cache::SeqCache, String> {
        if !self.caches.contains_key(&seq) {
            if let Some(c) = self.pooled_cache()? {
                self.caches.insert(seq, c);
                return Ok(self.caches.get_mut(&seq).expect("inserted"));
            }
            let c = match self.spare.pop() {
                Some(mut c) => {
                    c.clear();
                    c
                }
                None => cache::SeqCache::new(
                    &self.model.cfg,
                    self.backend.buffer_type(),
                    self.model.cfg.block_size,
                    self.model.window,
                )
                .map_err(|e| format!("cannot allocate the drafter's cache: {e}"))?,
            };
            self.caches.insert(seq, c);
        }
        Ok(self.caches.get_mut(&seq).expect("inserted"))
    }

    /// `--spec-draft-batched-attn`: a fresh cache in a free slot of the pool (made on first
    /// use, a slot for each of the engine's sequences); `None` without the flag, for a drafter
    /// without DFlash 2's batched pass, when the pool could not be allocated or with every slot
    /// taken (that sequence then has a cache of its own and attends alone).
    fn pooled_cache(&mut self) -> Result<Option<cache::SeqCache>, String> {
        if !self.opts.batched_attn || self.model.selector().is_none() || self.pool_failed {
            return Ok(None);
        }
        let cfg = &self.model.cfg;
        if self.pool.is_none() {
            let pool = match cache::CachePool::new(
                cfg,
                self.backend.buffer_type(),
                cfg.block_size,
                self.model.window,
                self.max_seqs,
            ) {
                Ok(pool) => pool,
                Err(e) => {
                    tracing::warn!(
                        "cannot allocate the drafter's cache pool ({e}): drafting attends per \
                         sequence"
                    );
                    self.pool_failed = true;
                    return Ok(None);
                }
            };
            tracing::info!(
                slots = pool.slots(),
                cells = pool.cells(),
                "drafter cache pool for batched attention"
            );
            self.pool = Some(pool);
        }
        let pool = self.pool.as_mut().expect("made above");
        let Some(slot) = pool.take() else {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    slots = pool.slots(),
                    "the drafter's cache pool is full: more sequences draft with a cache of \
                     their own and attend alone (warned once)"
                );
            }
            return Ok(None);
        };
        match cache::SeqCache::in_pool(cfg, pool, slot, cfg.block_size, self.model.window) {
            Ok(c) => Ok(Some(c)),
            Err(e) => {
                pool.give(slot);
                Err(format!("cannot place the drafter's cache in its pool: {e}"))
            }
        }
    }

    /// Whether request `i` (`r`) hands over the hidden states of its `ran` rows of every fused
    /// layer: in `r.hidden`, or on this drafter's device.
    fn check_hidden(&self, r: &DraftRequest<'_>, i: usize, src: Src<'_>) -> Result<(), String> {
        let n_layers = self.model.cfg.target_layers.len();
        let ok = match src {
            Src::Host => {
                r.hidden.len() == n_layers
                    && r.hidden.iter().all(|h| h.len() == r.ran * self.n_embd)
            }
            Src::Device(d) => {
                d.layers.len() == n_layers
                    && d.first.get(i).is_some_and(|&f| {
                        d.layers.iter().all(|l| {
                            l.width == self.n_embd && f + r.ran <= l.rows && l.device == self.device
                        })
                    })
            }
        };
        if !ok {
            return Err(format!(
                "sequence {}: expected {n_layers} hidden-state layers of {} rows of {}",
                r.seq, r.ran, self.n_embd
            ));
        }
        Ok(())
    }

    /// Appends rows `rows` of request `i`'s (`r`) fused hidden states: on the host to `feats`
    /// (`n_layers × n_embd` a row), on the device their indices in the step's rows to `idx`.
    fn features(
        &self,
        r: &DraftRequest<'_>,
        i: usize,
        src: Src<'_>,
        rows: std::ops::Range<usize>,
        feats: &mut Vec<f32>,
        idx: &mut Vec<i32>,
    ) {
        match src {
            Src::Host => {
                for row in rows {
                    for h in &r.hidden {
                        feats.extend_from_slice(&h[row * self.n_embd..(row + 1) * self.n_embd]);
                    }
                }
            }
            Src::Device(d) => idx.extend(rows.map(|row| (d.first[i] + row) as i32)),
        }
    }

    /// Takes in the target step's rows of request `i` (`r`), one sequence.
    ///
    /// In pieces of [`INGEST_CHUNK`] rows. Host rows go into each piece's input straight from
    /// the captured layers with [`Self::direct`], else gathered in `feats` first. With
    /// [`Self::pipe`] each piece but the last is queued without waiting, on `alloc` and
    /// `draft_alloc` in turn: a piece's graph is built and its inputs written while the piece
    /// before runs, once the piece two before (on the same allocator) is done (#1133). The same
    /// graphs with the same inputs in the same order on the backend's stream as waiting for
    /// each piece, so the cache gets the same bits; the last piece waits, so the take-in is
    /// done on return either way, and neither allocator is in use.
    fn ingest(&mut self, r: &DraftRequest<'_>, i: usize, src: Src<'_>) -> Result<(), String> {
        self.check_hidden(r, i, src)?;
        let t0 = Instant::now();
        let mut cache = self.caches.remove(&r.seq).expect("made by cache_for");
        cache.rollback(r.start);
        let first = ingest_from(r.tokens.len(), r.start, r.ran, cache.window());
        let mut at = first;
        let direct = self.direct && matches!(src, Src::Host);
        let piece = INGEST_CHUNK.min(r.ran.saturating_sub(at)) * self.feats_width();
        let mut feats = match src {
            Src::Host if !direct && keep_feats() => self.kept_feats(piece),
            _ => Vec::new(),
        };
        let mut idx = Vec::new();
        let mut result = Ok(());
        // Whether a piece queued without waiting for it holds allocator 0 (`alloc`) or 1.
        let mut queued = [false; 2];
        let mut t = TakeInTimes::default();
        while at < r.ran {
            let n = INGEST_CHUNK.min(r.ran - at);
            let last = at + n >= r.ran;
            if !direct {
                feats.clear();
                idx.clear();
                let tg = Instant::now();
                self.features(r, i, src, at..at + n, &mut feats, &mut idx);
                t.gather += tg.elapsed();
            }
            let slot = if self.pipe { t.pieces % 2 } else { 0 };
            if queued[slot] {
                let tw = Instant::now();
                self.wait_piece(slot);
                t.wait += tw.elapsed();
                queued[slot] = false;
            }
            let positions: Vec<usize> = (r.start + at..r.start + at + n).collect();
            let block = !self.pipe || last;
            let feats_in = if direct {
                Feats::Rows {
                    layers: &r.hidden,
                    width: self.n_embd,
                    from: at,
                }
            } else {
                feats_of(src, &feats, &idx)
            };
            let alloc = if slot == 0 {
                &mut self.alloc
            } else {
                &mut self.draft_alloc
            };
            let up = self.model.ingest_many(
                &mut self.backend,
                alloc,
                &mut [&mut cache],
                None,
                feats_in,
                &[&positions],
                block,
            );
            match up {
                Ok(up) => t.upload += up,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
            if !block {
                self.piece_queued(slot);
                queued[slot] = true;
            }
            t.pieces += 1;
            at += n;
        }
        if self.pipe && result.is_err() {
            // Nothing left running after a failed piece (the last piece waits otherwise).
            self.backend.synchronize();
        }
        let rows = r.ran - first.min(r.ran);
        if rows >= INGEST_CHUNK {
            tracing::debug!(
                seq = r.seq,
                rows,
                pieces = t.pieces,
                pipe = self.pipe,
                direct,
                ms = t0.elapsed().as_secs_f64() * 1e3,
                gather_ms = t.gather.as_secs_f64() * 1e3,
                upload_ms = t.upload.as_secs_f64() * 1e3,
                wait_ms = t.wait.as_secs_f64() * 1e3,
                // The two allocators' buffers (the second also DFlash 2's draft pass's).
                alloc_mib = mib(&self.alloc),
                draft_alloc_mib = mib(&self.draft_alloc),
                "dflash take-in"
            );
        }
        self.caches.insert(r.seq, cache);
        self.keep_feats_buf(src, feats);
        result
    }

    /// Marks the end of the take-in piece just queued on allocator `slot` ([`Self::ingest`]).
    fn piece_queued(&mut self, slot: usize) {
        if self.piece_done.is_empty() {
            // Both or none.
            if let (Some(a), Some(b)) = (self.backend.new_event(), self.backend.new_event()) {
                self.piece_done = vec![a, b];
            }
        }
        if let Some(e) = self.piece_done.get(slot) {
            self.backend.record(e);
        }
    }

    /// Waits for the take-in piece queued on allocator `slot`, not for those queued after it
    /// (all of them where the backend has no events).
    fn wait_piece(&mut self, slot: usize) {
        match self.piece_done.get(slot) {
            Some(e) => e.synchronize(),
            None => self.backend.synchronize(),
        }
    }

    /// Floats of one row of [`Self::features`] on the host: every fused layer's.
    fn feats_width(&self) -> usize {
        self.n_embd * self.model.cfg.target_layers.len()
    }

    /// The kept features buffer ([`Self::feats`]), empty, with room for `n` floats.
    fn kept_feats(&mut self, n: usize) -> Vec<f32> {
        let mut f = std::mem::take(&mut self.feats);
        empty_with_room(&mut f, n);
        f
    }

    /// Keeps a take-in's host features buffer for the next one (the larger, when it holds one).
    fn keep_feats_buf(&mut self, src: Src<'_>, f: Vec<f32>) {
        if matches!(src, Src::Host) && keep_feats() && f.capacity() >= self.feats.capacity() {
            self.feats = f;
        }
    }

    /// Takes in the target step's rows of every sequence of `batch`: in one graph when they fit
    /// one (a decode step's few rows each), else one by one.
    fn ingest_all(&mut self, batch: &[DraftRequest<'_>], src: Src<'_>) -> Result<(), String> {
        let take: Vec<usize> = (0..batch.len()).filter(|&i| batch[i].ran > 0).collect();
        let rows: usize = take.iter().map(|&i| batch[i].ran).sum();
        let window = self.model.window;
        // One sequence goes this way only when `ingest` would take in all its rows in one
        // piece (a decode step's): then the graph is `ingest`'s, started without waiting.
        let alone = match take.as_slice() {
            [] => true,
            &[i] => {
                let r = &batch[i];
                ingest_from(r.tokens.len(), r.start, r.ran, window) != 0
            }
            _ => false,
        };
        if alone || rows > INGEST_CHUNK || take.iter().any(|&i| batch[i].ran > window) {
            return take
                .into_iter()
                .try_for_each(|i| self.ingest(&batch[i], i, src));
        }
        let mut feats = match src {
            Src::Host if keep_feats() => self.kept_feats(rows * self.feats_width()),
            Src::Host => Vec::with_capacity(rows * self.feats_width()),
            Src::Device(_) => Vec::new(),
        };
        let mut idx = Vec::with_capacity(rows);
        for &i in &take {
            let r = &batch[i];
            self.check_hidden(r, i, src)?;
            self.features(r, i, src, 0..r.ran, &mut feats, &mut idx);
        }
        let mut caches: Vec<cache::SeqCache> = take
            .iter()
            .map(|&i| {
                let r = &batch[i];
                let mut c = self.caches.remove(&r.seq).expect("made by cache_for");
                c.rollback(r.start);
                c
            })
            .collect();
        let positions: Vec<Vec<usize>> = take
            .iter()
            .map(|&i| (batch[i].start..batch[i].start + batch[i].ran).collect())
            .collect();
        let positions: Vec<&[usize]> = positions.iter().map(Vec::as_slice).collect();
        let mut refs: Vec<&mut cache::SeqCache> = caches.iter_mut().collect();
        // The draft pass that follows (`step`) is built while this runs.
        let result = self
            .model
            .ingest_many(
                &mut self.backend,
                &mut self.alloc,
                &mut refs,
                self.pool.as_ref(),
                feats_of(src, &feats, &idx),
                &positions,
                false,
            )
            .map(|_| ());
        for (&i, c) in take.iter().zip(caches) {
            self.caches.insert(batch[i].seq, c);
        }
        self.keep_feats_buf(src, feats);
        result
    }

    /// [`ModelDrafter::step`] with the step's rows where `src` says.
    fn step_from(&mut self, batch: &mut [DraftRequest<'_>], src: Src<'_>) -> Result<(), String> {
        // DFlash 2 runs the same block length for every sequence: several sequences share one
        // take-in graph and one draft pass, which read the drafter's weights once. One sequence
        // goes the same way, so that its draft pass is built while its take-in runs.
        let asking = batch
            .iter()
            .filter(|r| r.max > 0 && !r.tokens.is_empty())
            .count();
        if self.model.selector().is_some() && asking > 0 {
            for r in batch.iter() {
                self.cache_for(r.seq)?;
            }
            let result = self
                .ingest_all(batch, src)
                .and_then(|()| self.draft_all(batch));
            // The take-in graph may still run when a step fails before the draft pass.
            self.backend.synchronize();
            return result;
        }
        for (i, r) in batch.iter_mut().enumerate() {
            self.cache_for(r.seq)?;
            if r.ran > 0 {
                self.ingest(r, i, src)?;
            }
            if r.max > 0 {
                self.draft(r)?;
            }
        }
        // A lone take-in reads the step's rows on the device: done before they may change.
        if matches!(src, Src::Device(_)) {
            self.backend.synchronize();
        }
        Ok(())
    }

    /// DFlash 2: the drafts of every sequence of `batch` that asks for some, from one pass over
    /// all their blocks (#521).
    fn draft_all(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        let asking: Vec<usize> = (0..batch.len())
            .filter(|&i| batch[i].max > 0 && !batch[i].tokens.is_empty())
            .collect();
        if asking.is_empty() {
            return Ok(());
        }
        let seqs: Vec<BlockSeq<'_>> = asking
            .iter()
            .map(|&i| BlockSeq {
                cache: self.caches.get(&batch[i].seq).expect("made by cache_for"),
                last: *batch[i].tokens.last().expect("checked above"),
                anchor_pos: batch[i].tokens.len() - 1,
            })
            .collect();
        let rows = self.model.draft_many(
            &mut self.backend,
            &mut self.draft_alloc,
            &seqs,
            self.pool.as_ref(),
            self.model.cfg.block_size,
            false,
            self.opts.min_confidence > 0.0,
        )?;
        for (&i, rows) in asking.iter().zip(rows) {
            let last = *batch[i].tokens.last().expect("checked above");
            self.pick_block(&mut batch[i], last, rows);
        }
        Ok(())
    }

    /// DFlash 2: `r`'s drafts, the selector's walk over a block pass's candidates (`last` is
    /// the token before the block).
    fn pick_block(&mut self, r: &mut DraftRequest<'_>, last: u32, rows: Rows) {
        let (Some(selector), Some((candidates, mut unary, proj))) =
            (self.model.selector(), rows.selector)
        else {
            return;
        };
        // The device's logits are the head's output; the checkpoint's logit scale and soft cap
        // apply here.
        let (scale, softcap) = (self.model.cfg.logit_scale, self.model.cfg.logit_softcap);
        unary
            .iter_mut()
            .for_each(|x| *x = logit_transform(*x, scale, softcap));
        let walk = selector.walk(
            last,
            &candidates,
            &unary,
            &proj,
            self.opts.temperature,
            &mut self.rng,
        );
        let keep = r.max.min(walk.tokens.len());
        r.out.tokens = walk.tokens[..keep].to_vec();
        r.out.probs = walk.probs.map_or_else(Vec::new, |rows| {
            rows[..keep]
                .iter()
                .map(|row| {
                    let mut full = vec![0f32; self.n_target_vocab];
                    row.iter().for_each(|&(t, p)| full[t as usize] = p);
                    full
                })
                .collect()
        });
    }

    /// Drafts for one sequence, from the block pass over what its cache holds.
    ///
    /// DFlash: every block position after the first is a draft, chosen on its own. DSpark: the
    /// Markov head makes position `i`'s logits depend on the token chosen at `i - 1` (the last
    /// token for the first), so the positions are chosen in order, each from the block pass's
    /// logits plus the head's bias, and the confidence head can cut the block short.
    fn draft(&mut self, r: &mut DraftRequest<'_>) -> Result<(), String> {
        let first = self.model.first_row;
        let v2 = self.model.selector().is_some();
        // DFlash 2 runs the whole block, as it was trained (the convolutions and the
        // non-causal attention see every position), and keeps the first `max` drafts.
        let n = if v2 {
            self.model.cfg.block_size
        } else {
            (r.max + first).min(self.model.cfg.block_size)
        };
        let Some(&last) = r.tokens.last() else {
            return Ok(());
        };
        if n < first + 1 {
            return Ok(());
        }
        let sampling = self.opts.temperature > 0.0;
        let markov = self.model.has_markov();
        let cache = self.caches.get(&r.seq).expect("made by cache_for");
        let rows = self.model.draft(
            &mut self.backend,
            &mut self.alloc,
            cache,
            last,
            r.tokens.len() - 1,
            n,
            (sampling || markov) && !v2,
            self.opts.min_confidence > 0.0,
        )?;
        // A rank above 0 of `--tp-draft-split` ran its share of the pass: rank 0 picks.
        if self.opts.tp.is_some_and(|t| !t.leads()) {
            return Ok(());
        }
        if v2 {
            // DFlash 2: the block's candidates, chosen in order.
            self.pick_block(r, last, rows);
            return Ok(());
        }
        let to_target = |i: u32| self.model.d2t.as_ref().map_or(i, |m| m[i as usize]);
        if !sampling && !markov {
            r.out.tokens = rows.argmax.iter().map(|&i| to_target(i)).collect();
            r.out.probs.clear();
            return Ok(());
        }
        let v = self.model.n_vocab;
        let n_embd = self.model.cfg.n_embd;
        let (scale, softcap) = (self.model.cfg.logit_scale, self.model.cfg.logit_softcap);
        let mut logits = rows.logits.expect("asked for");
        let mut out = Draft::default();
        let mut prev = last;
        for (i, row) in logits.chunks_exact_mut(v).enumerate() {
            if markov {
                let (bias, w1_row) =
                    self.model
                        .markov_step(&mut self.backend, &mut self.alloc, prev)?;
                row.iter_mut().zip(&bias).for_each(|(l, b)| *l += b);
                if self.opts.min_confidence > 0.0 {
                    let h = &rows.hidden.as_ref().expect("asked for")[i * n_embd..(i + 1) * n_embd];
                    if self.model.confidence(h, &w1_row) < self.opts.min_confidence {
                        break;
                    }
                }
            }
            let pick = if sampling {
                let q = softmax_row(row, self.opts.temperature, scale, softcap);
                let pick = sample(&q, self.rng.next_f64());
                let mut full = vec![0f32; self.n_target_vocab];
                match &self.model.d2t {
                    Some(m) => q
                        .iter()
                        .enumerate()
                        .for_each(|(j, &p)| full[m[j] as usize] = p),
                    // The head's rows are the target's first (all, or a shortlist).
                    None => full[..q.len()].copy_from_slice(&q),
                }
                out.probs.push(full);
                pick
            } else {
                argmax(row)
            };
            prev = to_target(pick as u32);
            out.tokens.push(prev);
        }
        *r.out = out;
        Ok(())
    }

    /// `--tp-draft-split`: the all-reduces the communicator issued before a round. `None`
    /// without the split.
    fn round_start(&self) -> Option<u32> {
        let t = self.opts.tp?;
        Some(crate::tp_net::comm(t.comm)?.all_reduces())
    }

    /// `--tp-draft-split`: a round of `batch` that failed on this rank still runs the FFN sums
    /// it left ([`round_sums`]: the same on every rank), with zeros for its partials, so that
    /// the other ranks' round ends at once and every rank's communicator stays in step (a sum
    /// left out would hold the others until the timeout and fail the communicator for good).
    /// The others' drafts of the round are lost; this rank returns its error. Sums of a failed
    /// device (the communicator's error) are not retried. If the zeros cannot go either, the
    /// communicator fails and cuts its connections ([`crate::tp_net::Comm::abort`]): the
    /// others' sums fail at once rather than at the timeout. A round whose sums failed (the
    /// communicator's error, here or on another rank) fails.
    fn round_end(
        &mut self,
        batch: &[DraftRequest<'_>],
        before: Option<u32>,
        r: Result<(), String>,
    ) -> Result<(), String> {
        let Some(t) = self.opts.tp else {
            return r;
        };
        let comm = crate::tp_net::comm(t.comm);
        if r.is_ok()
            && let Some(e) = comm.as_ref().and_then(|c| c.error())
        {
            return Err(format!(
                "--tp-draft-split: the draft round's sums failed: {e}"
            ));
        }
        let (Err(e), Some(before)) = (&r, before) else {
            return r;
        };
        let Some(comm) = comm.filter(|c| c.error().is_none()) else {
            return r;
        };
        let c = &self.model.cfg;
        let shape = RoundShape {
            v2: self.model.selector().is_some(),
            n_layer: c.n_layer,
            n_embd: c.n_embd,
            block: c.block_size,
            first: self.model.first_row,
            head: self.model.tp_head_split(),
        };
        let sums = round_sums(shape, batch.iter().map(|r| (r.max, !r.tokens.is_empty())));
        let left = sums_left(&sums, comm.all_reduces().wrapping_sub(before));
        match send_zero_sums(&mut self.backend, t.comm, left) {
            Ok(()) => {
                tracing::warn!(
                    error = %e,
                    sums = left.len(),
                    "--tp-draft-split: a draft round failed here, its other sums went as zeros"
                );
                r
            }
            // The others wait for sums this rank cannot send: the communicator fails and cuts
            // its connections, as a rank that died, so that theirs fail now (not at the
            // timeout) and the session ends as it does when a rank goes.
            Err(z) => {
                let e = format!("{e}; its other --tp-draft-split sums failed too: {z}");
                tracing::error!(error = %e, "--tp-draft-split: a draft round failed here");
                comm.abort(e.clone());
                Err(e)
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: [`send_zero_sums`] on this thread fails as if it could not make its graph.
    pub(crate) static FAIL_ZERO_SUMS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What a draft round's FFN sums depend on besides its requests ([`round_sums`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoundShape {
    /// DFlash 2: one pass over every asking sequence's block.
    pub v2: bool,
    pub n_layer: usize,
    pub n_embd: usize,
    pub block: usize,
    /// The block's first row that gives a draft ([`model::first_row`]).
    pub first: usize,
    /// DFlash 2's head split over the ranks ([`tp_head`]), if set.
    pub head: Option<tp_head::Split>,
}

/// `--tp-draft-split`: the values of each sum (one a layer of every draft pass, in order, then
/// the pass's head exchange where its head splits, [`tp_head`]) a round runs for requests
/// `(max, has tokens)`, as [`DflashDrafter::step_from`] runs them: DFlash 2 one pass over the
/// blocks of every request asking for drafts, DFlash one pass per such request of `max` rows
/// after the first (at most the block). Take-ins sum nothing.
pub(crate) fn round_sums(s: RoundShape, reqs: impl Iterator<Item = (usize, bool)>) -> Vec<usize> {
    let asking: Vec<usize> = reqs
        .filter(|&(max, tokens)| max > 0 && tokens)
        .map(|(max, _)| max)
        .collect();
    let passes: Vec<usize> = if s.v2 {
        let rows = s.block * asking.len();
        (rows > 0).then_some(rows).into_iter().collect()
    } else {
        asking
            .iter()
            .map(|&max| (max + s.first).min(s.block))
            .filter(|&n| n > s.first)
            .collect()
    };
    passes
        .iter()
        .flat_map(|&rows| {
            // A DFlash 2 pass's output rows: each block's after its first.
            let out = rows / s.block * (s.block - s.first);
            let head = s
                .head
                .filter(|h| s.v2 && h.takes(out))
                .map(|h| h.values(out));
            std::iter::repeat_n(rows * s.n_embd, s.n_layer).chain(head)
        })
        .collect()
}

/// The sums of a round's `sums` after the first `done` (a failed round's left).
pub(crate) fn sums_left(sums: &[usize], done: u32) -> &[usize] {
    sums.get(done as usize..).unwrap_or_default()
}

/// Runs an all-reduce of `n` zeros on communicator `comm` for each `n` of `sizes`, in order,
/// through the op a draft pass sums with (the same exchanges as the other ranks' pass).
pub(crate) fn send_zero_sums(
    backend: &mut Backend,
    comm: i32,
    sizes: &[usize],
) -> Result<(), String> {
    let Some(&max) = sizes.iter().max() else {
        return Ok(());
    };
    #[cfg(test)]
    if FAIL_ZERO_SUMS.get() {
        return Err("no graph for the zero sums (test)".into());
    }
    let size = 16 + 4 * sizes.len();
    let ctx = Context::new(size, size);
    let zeros = ctx
        .new_tensor(types::F32, &[max as i64])
        .named("zeros")
        .as_input();
    let mut graph = ctx.new_graph(size);
    for &n in sizes {
        let x = ctx.view_1d(zeros, n as i64, 0);
        graph.expand(crate::tp_net::all_reduce(&ctx, comm, x));
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .map_err(|e| e.to_string())?;
    zeros.set_f32(&vec![0.0; max]);
    backend.compute(&graph).map_err(|e| e.to_string())
}

/// The first position whose row a ring of `window` positions still holds when a sequence of
/// `n_tokens` tokens (a request's [`DraftRequest::tokens`]) has had every row before its last
/// token taken in. Rows below it are overwritten before any pass reads the ring: a prompt's
/// chunks come with the whole prompt as `tokens`, and the prompt's last rows arrive (and take
/// those cells) before the first draft; a decode step's rows end at the token before the last.
fn ring_keep_from(n_tokens: usize, window: usize) -> usize {
    n_tokens.saturating_sub(1 + window)
}

/// The first of a step's `ran` rows (positions from `start`, of a sequence of `n_tokens`
/// tokens) that [`DflashDrafter::ingest`] takes in; `ran` when none. A ring holds `window`
/// positions, so a chunk's rows before its last `window` are overwritten; and a piece of
/// [`INGEST_CHUNK`] rows that the sequence's later rows overwrite before any pass reads the
/// ring (a long prompt's early chunks) is left out whole, so the pieces taken in are the same
/// ones, of the same rows, as without the cut. Never grows when `n_tokens` shrinks.
fn ingest_from(n_tokens: usize, start: usize, ran: usize, window: usize) -> usize {
    let keep = ring_keep_from(n_tokens, window);
    let mut at = ran.saturating_sub(window);
    while at < ran {
        let n = INGEST_CHUNK.min(ran - at);
        if start + at + n > keep {
            return at;
        }
        at += n;
    }
    ran
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best
}

/// The checkpoint's logit scale (0: none) and soft cap (0: none) applied to a logit.
fn logit_transform(x: f32, scale: f32, softcap: f32) -> f32 {
    let x = if scale != 0.0 { x * scale } else { x };
    if softcap > 0.0 {
        softcap * (x / softcap).tanh()
    } else {
        x
    }
}

/// `softmax(transform(row) / temperature)`, where the transform is the checkpoint's logit scale
/// and soft cap.
fn softmax_row(row: &[f32], temperature: f32, scale: f32, softcap: f32) -> Vec<f32> {
    let t = |x: f32| logit_transform(x, scale, softcap);
    let max = row.iter().map(|&x| t(x)).fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = row
        .iter()
        .map(|&x| ((t(x) - max) / temperature).exp())
        .collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= sum);
    p
}

/// The index `u` (uniform in `[0, 1)`) falls on in the distribution `p`.
fn sample(p: &[f32], u: f64) -> usize {
    let mut acc = 0f64;
    for (i, &x) in p.iter().enumerate() {
        acc += x as f64;
        if u < acc {
            return i;
        }
    }
    // Rounding: the last token with any mass.
    p.iter().rposition(|&x| x > 0.0).unwrap_or(0)
}

impl ModelDrafter for DflashDrafter {
    fn hidden_layers(&self) -> Vec<usize> {
        self.model.cfg.target_layers.clone()
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        let before = self.round_start();
        let r = self.step_from(batch, Src::Host);
        self.round_end(batch, before, r)
    }

    /// The take-in graph gathers the rows it needs where the target's step left them: the
    /// same floats as from the host, without reading every captured row back (#1181).
    fn reads_device_rows(&self) -> Option<usize> {
        self.device
    }

    fn step_device(
        &mut self,
        batch: &mut [DraftRequest<'_>],
        rows: &DeviceHidden<'_>,
    ) -> Result<(), String> {
        let before = self.round_start();
        let r = if rows.first.len() != batch.len() {
            Err(format!(
                "{} first rows for {} requests",
                rows.first.len(),
                batch.len()
            ))
        } else {
            self.step_from(batch, Src::Device(rows))
        };
        self.round_end(batch, before, r)
    }

    fn release(&mut self, seq: SeqId) {
        let Some(c) = self.caches.remove(&seq) else {
            return;
        };
        if let Some(slot) = c.slot() {
            // Its views go; the slot takes the next sequence.
            drop(c);
            self.pool
                .as_mut()
                .expect("a slot is of the pool")
                .give(slot);
        } else if self.spare.len() < SPARE_CACHES {
            self.spare.push(c);
        }
    }

    /// DFlash 2 runs the whole block whatever the depth.
    fn block_drafts(&self) -> Option<usize> {
        self.model.cfg.v2.is_some().then(|| self.max_drafts())
    }

    /// A lone request takes in its rows from [`ingest_from`] on (`step` runs [`Self::ingest`]
    /// for it).
    fn rows_read_from(&self, n_tokens: usize, start: usize, ran: usize) -> usize {
        ingest_from(n_tokens, start, ran, self.model.window)
    }
}

/// [`ModelDrafter::block_drafts`] of the `--spec dflash` drafter at `path`, from its config
/// alone (no weights read): the ranks above 0 of tensor parallelism over machines (#619), which
/// leave the drafting to rank 0 but must verify as many rows.
pub fn block_drafts_of(path: &std::path::Path) -> Result<Option<usize>, String> {
    let cfg = Source::open(path)?
        .config(Flavor::Dflash)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(cfg
        .v2
        .is_some()
        .then(|| cfg.block_size - model::first_row(Flavor::Dflash, &cfg)))
}
