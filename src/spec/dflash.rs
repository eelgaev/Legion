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
mod head;
mod model;
pub mod tiny;
mod v2;
mod weights;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_dspark;
#[cfg(test)]
mod tests_head;
#[cfg(test)]
mod tests_v2;

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::sampling::Rng;
use crate::ggml::{Backend, ComputeAllocator, gpu_device};
use crate::spec::{Draft, DraftRequest, ModelDrafter, ModelDrafterFactory, SeqId, TargetInfo};

pub use config::{Config, ConfigError, Dflash2, Flavor};
use model::{BlockSeq, INGEST_CHUNK, Model, Rows};
use weights::{Source, load_from_gguf};

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
    opts: DflashOptions,
    rng: Rng,
    n_embd: usize,
    n_target_vocab: usize,
    caches: HashMap<SeqId, cache::SeqCache>,
    /// Caches of finished sequences, to reuse.
    spare: Vec<cache::SeqCache>,
}

/// Most idle caches kept for the next sequences.
const SPARE_CACHES: usize = 4;

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
        let cfg = source
            .config(flavor)
            .map_err(|e| format!("{}: {e}", path.display()))?;
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
        let mut store = source.load(buft, opts.weights)?;
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
        let model = Model::new(
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
        Ok(Self {
            n_embd: target.n_embd,
            n_target_vocab: target.n_vocab,
            rng: Rng::new(opts.seed),
            model,
            backend,
            alloc: None,
            opts,
            caches: HashMap::new(),
            spare: Vec::new(),
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

    /// Whether `r` hands over the hidden states of its `ran` rows of every fused layer.
    fn check_hidden(&self, r: &DraftRequest<'_>) -> Result<(), String> {
        let n_layers = self.model.cfg.target_layers.len();
        if r.hidden.len() != n_layers || r.hidden.iter().any(|h| h.len() != r.ran * self.n_embd) {
            return Err(format!(
                "sequence {}: expected {n_layers} hidden-state layers of {} rows of {}",
                r.seq, r.ran, self.n_embd
            ));
        }
        Ok(())
    }

    /// Appends rows `rows` of `r`'s fused hidden states to `feats` (`n_layers × n_embd` a row).
    fn features(&self, r: &DraftRequest<'_>, rows: std::ops::Range<usize>, feats: &mut Vec<f32>) {
        for row in rows {
            for h in &r.hidden {
                feats.extend_from_slice(&h[row * self.n_embd..(row + 1) * self.n_embd]);
            }
        }
    }

    /// Takes in the target step's rows of one sequence.
    fn ingest(&mut self, r: &DraftRequest<'_>) -> Result<(), String> {
        self.check_hidden(r)?;
        let mut cache = self.caches.remove(&r.seq).expect("made by cache_for");
        cache.rollback(r.start);
        let mut feats = Vec::new();
        let mut at = ingest_from(r.tokens.len(), r.start, r.ran, cache.window());
        let mut result = Ok(());
        while at < r.ran {
            let n = INGEST_CHUNK.min(r.ran - at);
            feats.clear();
            self.features(r, at..at + n, &mut feats);
            let positions: Vec<usize> = (r.start + at..r.start + at + n).collect();
            result = self.model.ingest(
                &mut self.backend,
                &mut self.alloc,
                &mut cache,
                &feats,
                &positions,
            );
            if result.is_err() {
                break;
            }
            at += n;
        }
        self.caches.insert(r.seq, cache);
        result
    }

    /// Takes in the target step's rows of every sequence of `batch`: in one graph when they fit
    /// one (a decode step's few rows each), else one by one.
    fn ingest_all(&mut self, batch: &[DraftRequest<'_>]) -> Result<(), String> {
        let take: Vec<&DraftRequest<'_>> = batch.iter().filter(|r| r.ran > 0).collect();
        let rows: usize = take.iter().map(|r| r.ran).sum();
        let window = self.model.window;
        if take.len() < 2 || rows > INGEST_CHUNK || take.iter().any(|r| r.ran > window) {
            return take.into_iter().try_for_each(|r| self.ingest(r));
        }
        let mut feats = Vec::with_capacity(rows * self.n_embd * self.model.cfg.target_layers.len());
        for r in &take {
            self.check_hidden(r)?;
            self.features(r, 0..r.ran, &mut feats);
        }
        let mut caches: Vec<cache::SeqCache> = take
            .iter()
            .map(|r| {
                let mut c = self.caches.remove(&r.seq).expect("made by cache_for");
                c.rollback(r.start);
                c
            })
            .collect();
        let positions: Vec<Vec<usize>> = take
            .iter()
            .map(|r| (r.start..r.start + r.ran).collect())
            .collect();
        let positions: Vec<&[usize]> = positions.iter().map(Vec::as_slice).collect();
        let mut refs: Vec<&mut cache::SeqCache> = caches.iter_mut().collect();
        let result = self.model.ingest_many(
            &mut self.backend,
            &mut self.alloc,
            &mut refs,
            &feats,
            &positions,
        );
        for (r, c) in take.iter().zip(caches) {
            self.caches.insert(r.seq, c);
        }
        result
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
            &mut self.alloc,
            &seqs,
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
        // DFlash 2 runs the same block length for every sequence: several sequences share one
        // take-in graph and one draft pass, which read the drafter's weights once.
        let asking = batch
            .iter()
            .filter(|r| r.max > 0 && !r.tokens.is_empty())
            .count();
        if self.model.selector().is_some() && asking > 1 {
            for r in batch.iter() {
                self.cache_for(r.seq)?;
            }
            self.ingest_all(batch)?;
            return self.draft_all(batch);
        }
        for r in batch.iter_mut() {
            self.cache_for(r.seq)?;
            if r.ran > 0 {
                self.ingest(r)?;
            }
            if r.max > 0 {
                self.draft(r)?;
            }
        }
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(c) = self.caches.remove(&seq)
            && self.spare.len() < SPARE_CACHES
        {
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
