//! Speculative decoding: drafters propose tokens, the target model verifies them in one batched
//! step, and an exact acceptance rule keeps the output distribution the target's own.
//!
//! # The verify step
//!
//! A decoding sequence with drafts `d1..dk` feeds `[last, d1..dk]` (`last`: the token sampled in
//! the previous step, not yet in the cache) at positions `n_past..=n_past + k` and gets one
//! logits row per position. Every decoding sequence of a step shares one group of the batch
//! (the DeltaNet ops need the same token count per sequence), so all of them feed `1 + max k`
//! tokens; a sequence with fewer drafts is padded with copies of `last`, whose rows are never
//! read (causal attention and the sequential recurrence keep earlier rows independent of them).
//! Prefill chunks of other sequences run in the same step as separate groups. A step in which
//! no sequence drafted anything is a plain decode step (the reused one-token graph).
//!
//! # Acceptance ([`accept`])
//!
//! - Greedy (temperature 0): row `i`'s pick (argmax, after penalties if any) is appended; the
//!   next draft is accepted while it equals that pick. This is the longest matching prefix plus
//!   the bonus token, and with batch-invariant kernels it is bit-identical to plain decoding.
//! - Temperature > 0: speculative rejection sampling (Leviathan et al. 2023, Chen et al. 2023):
//!   accept `d` with probability `min(1, p(d) / q(d))`, else draw from `norm(max(0, p - q))`
//!   and stop; after the last accepted draft, draw the bonus token from `p`. `p` is exactly the
//!   distribution the plain sampler draws from (penalties, temperature, min-p / top-k / top-p),
//!   computed per position with the penalty counts of the tokens accepted before it, so the
//!   output distribution equals plain sampling. Model-free drafters propose point masses
//!   (`q(d) = 1`); then acceptance is `u < p(d)` and the residual is `p` without `d`. With a
//!   seed, the text differs from a plain run with the same seed (the draws are used
//!   differently); its distribution does not.
//!
//! # Rollback
//!
//! After acceptance a sequence keeps `n + 1` of its `k + 1` fed positions (`last` and the `n`
//! accepted drafts); the new `last` is the bonus token.
//!
//! - **Paged KV**: the rejected positions' K/V stay in their cells, invisible (the mask only
//!   shows positions below `n_past`) and overwritten by the next step. The engine only moves
//!   `n_past` and trims its block table ([`crate::kv::BlockTable::truncate`]) to give whole
//!   blocks past the new end back to its reservation. Drafts are written at or past the end of
//!   the prompt, which is never inside a block shared through the radix cache (those are full
//!   prompt blocks), so shared blocks stay intact.
//! - **Recurrent state** (gated DeltaNet delta-rule state and conv history): replay, adapted
//!   from Strata's verify/commit split (MIT, Niko1221/Strata `src/core/verify.cpp`,
//!   `Verifier::capture_commit`). A verify group leaves the sequences' state slots untouched
//!   and instead stores, per recurrent layer and fed token, the delta rule's inputs (`q`, `k`,
//!   `v`, gate and beta, after the conv) and the conv window, into a replay pool
//!   (`crate::model::core::Executor::enable_spec`). Once acceptance is known, a commit graph
//!   (`Executor::commit`) runs `gated_delta_net` over each sequence's first `n + 1` stored
//!   tokens from its unchanged state and copies the matching conv window: the same kernel on
//!   the same inputs, so the state is bit-identical to having decoded those tokens one by one.
//!
//!   The alternatives are per-position state snapshots (`gated_delta_net`'s `K` snapshot
//!   slots; the commit copies the kept one) and recomputing the kept tokens with a full
//!   forward pass. Measured on the CPU (8 threads) at Qwen3.5-4B DeltaNet sizes (24 recurrent
//!   layers, 32 value heads of 128 x 128, so 48 MiB of state per sequence), k = 4, a batch of
//!   4 sequences, the recurrence alone (`spec_rollback_cost` in `tests/spec.rs`; two runs):
//!
//!   | | verify op | rollback | total | extra memory per sequence |
//!   |---|---|---|---|---|
//!   | plain decode, per token | 11-13 ms | - | - | - |
//!   | replay | 13-14 ms | 12-14 ms (3 tokens kept) | ~27 ms | 9.8 MiB (replay inputs, kept) |
//!   | snapshots | 41-42 ms | 11-12 ms (one copy) | ~53 ms | 240 MiB (5 states, per step) |
//!
//!   The recurrence is bound by state traffic: writing five snapshots costs three times the
//!   plain op, and their memory (240 MiB per sequence, 3.8 GiB of compute buffer at 16
//!   sequences) does not fit next to a model on a 24 GB card. Replay costs about one more
//!   plain-decode recurrence and a second, small graph, from 10 MiB per slot. A full
//!   recompute would re-read every weight of the model (gigabytes) to save that. Replay it is.
//!
//! # Drafters
//!
//! - [`TokenDrafter`]: model-free drafters (n-gram, suffix): a function of the token history.
//!   The engine makes one per sequence ([`TokenDrafterFactory`]).
//! - [`ModelDrafter`]: drafters with a forward pass of their own (MTP, a draft model, EAGLE-3,
//!   DFlash, DSpark), one per engine, called once per step for every sequence with the target
//!   step's hidden states at the positions it ran. [`draft::DraftModel`] (`--spec draft`) is
//!   the draft-model one.
//!
//! Metrics: [`SpecCounters`] per sequence and engine-wide (acceptance rate, mean accepted
//! length), in the `timings` object and `/metrics`.

pub mod accept;
pub mod adaptive;
pub mod depth;
pub mod dflash;
pub mod draft;
pub mod dspark;
pub mod eagle3;
pub mod mtp_ngram;
pub mod ngram;
pub(crate) mod q4_head;
pub mod suffix;

use std::path::PathBuf;
use std::sync::Arc;

/// Shared by every model-free drafter.
pub trait TokenDrafter: Send {
    /// Appends up to `max` draft tokens continuing `tokens` (the sequence so far: prompt + generated)
    /// to `out`. Proposing nothing is always allowed (then the step is a normal decode step).
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>);
    /// After verification: the tokens actually appended this step (accepted drafts + the bonus token).
    fn accepted(&mut self, _appended: &[u32]) {}
}

/// Makes the [`TokenDrafter`] of one sequence, when it starts decoding.
pub type TokenDrafterFactory = Arc<dyn Fn() -> Box<dyn TokenDrafter> + Send + Sync>;

/// Makes the engine's [`ModelDrafter`], on the engine's worker thread once the target is
/// loaded.
pub type ModelDrafterFactory =
    Arc<dyn Fn(&TargetInfo) -> Result<Box<dyn ModelDrafter>, String> + Send + Sync>;

/// What a [`ModelDrafter`]'s factory gets to build the drafter from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetInfo {
    /// Decoder layers of the target's main pass.
    pub n_layer: usize,
    /// The length of one hidden-state row ([`DraftRequest::hidden`]).
    pub n_embd: usize,
    pub n_vocab: usize,
    /// The target's GGUF: a drafter that shares tensors with the target (an MTP head reuses
    /// the token embeddings and the output head) loads them from here.
    pub model: PathBuf,
    /// `--spec-draft`: the drafter's own weights (a draft model, an EAGLE-3 / DFlash / DSpark
    /// head, or an MTP head shipped as a separate GGUF).
    pub draft: Option<PathBuf>,
    /// Whether the target runs on the GPU (the drafter should too).
    pub gpu: bool,
    /// CPU threads the target's backend uses.
    pub threads: usize,
    /// The longest sequence the engine runs (prompt + generated tokens).
    pub ctx_size: usize,
    /// Sequences the engine follows at once (decoding and prefilling), so the most a drafter
    /// keeps state for.
    pub max_seqs: usize,
    /// Whether the target's CPU backend runs the reference kernels (tests): a drafter that
    /// must agree with the target bit for bit does too.
    pub use_ref: bool,
}

/// Identifies a sequence across a [`ModelDrafter`]'s calls (unique for the engine's lifetime).
pub type SeqId = u64;

/// Draft tokens for one sequence, and optionally the drafter's distribution at each of them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Draft {
    pub tokens: Vec<u32>,
    /// `probs[i]` is the drafter's full distribution `q` (over the vocabulary, summing to 1) at
    /// draft `i`, which `tokens[i]` was drawn from. Empty: every draft is a point mass (`q = 1`
    /// at the token), as for greedy or model-free drafting. Needed for exact rejection sampling
    /// when the drafter samples.
    pub probs: Vec<Vec<f32>>,
    /// `confidence[i]`: the drafter's probability for `tokens[i]`, when it reports one (MTP
    /// heads; the depth policy calibrates its confidence floor on them, [`depth`]).
    pub confidence: Vec<f32>,
}

/// One sequence's part of a [`ModelDrafter`] call.
pub struct DraftRequest<'a> {
    pub seq: SeqId,
    /// The sequence so far (prompt + generated), as it stands after the step.
    pub tokens: &'a [u32],
    /// Positions the target step just ran for this sequence: `tokens[start..start + ran]`.
    pub start: usize,
    pub ran: usize,
    /// The target's hidden states at those positions, per requested layer
    /// ([`ModelDrafter::hidden_layers`], in that order): `hidden[l]` is `ran` rows of
    /// `n_embd` floats, row `j` being the output of that layer at position `start + j` (for
    /// the last layer: before the final norm). Rows of rejected drafts are not included.
    pub hidden: Vec<&'a [f32]>,
    /// Drafts wanted now, at most (0: only take in the step, e.g. a prefill chunk).
    pub max: usize,
    /// Where the drafts go (cleared by the caller).
    pub out: &'a mut Draft,
}

/// A drafter with a forward pass of its own (MTP #38, draft model #184, EAGLE-3 #185, DFlash
/// #186, DSpark #187). The engine calls [`Self::step`] after every target step that ran any
/// position of a sequence the drafter follows: prefill chunks (to build its own caches from
/// the target's hidden states) and decode / verify steps (after acceptance, with only the
/// accepted positions). The drafter keeps its own per-sequence state (e.g. an MTP layer's KV)
/// and rolls it back itself: rows it saw in an earlier call for positions at or past `start`
/// were rejected.
///
/// The drafter owns its weights and its backend (on the target's device, see
/// [`TargetInfo`]); the engine gives it the target's hidden states on the host. A sequence
/// that resumed from the prefix cache starts at `start > 0`: the target did not run its cached
/// positions, so there are no hidden states for them (an MTP drafter has to cache its own KV
/// alongside, or draft nothing until it has context).
///
/// The target's own MTP head (`--spec mtp`) uses the same requests but is not a
/// `ModelDrafter`: it runs in the target's executor, sharing its backend, embeddings and output
/// head (the engine's `MtpDrafter`, `crate::model::core`'s `mtp.rs`).
pub trait ModelDrafter: Send {
    /// Target layers whose outputs this drafter reads, in the order [`DraftRequest::hidden`]
    /// holds them (e.g. MTP: the last layer, `n_layer - 1`; EAGLE-3: a low, middle and high
    /// layer). Asked once, when the engine starts.
    fn hidden_layers(&self) -> Vec<usize>;

    /// Takes in one target step and proposes the next drafts, batched over the sequences.
    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String>;

    /// The sequence is gone (finished, cancelled or failed); free its state.
    fn release(&mut self, _seq: SeqId) {}

    /// `Some(n)`: one pass drafts a whole block of up to `n` tokens whatever the depth asked
    /// for (DFlash 2). The engine then picks the depth from measured step costs
    /// ([`depth::DepthModel::block`]) instead of from acceptance alone, up to `n` when
    /// `--spec-k` is not given.
    fn block_drafts(&self) -> Option<usize> {
        None
    }

    /// The first row of a request's [`DraftRequest::hidden`] that [`Self::step`] reads, for a
    /// sequence of at least `n_tokens` tokens ([`DraftRequest::tokens`]) whose step ran `ran`
    /// positions from `start`, the step's only request (a prompt chunk). The rows before it
    /// may hold anything: the engine has the executor skip reading them back (#1133). It must
    /// not grow when `n_tokens` shrinks, so a count known before the step (the tokens so far)
    /// is safe. 0 (the default): every row.
    fn rows_read_from(&self, _n_tokens: usize, _start: usize, _ran: usize) -> usize {
        0
    }
}

/// `--spec`: which drafter proposes tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum SpecKind {
    /// No speculative decoding.
    #[default]
    None,
    /// The model's own multi-token-prediction layers (#38).
    Mtp,
    /// Prompt / history n-gram lookup (#182).
    Ngram,
    /// Suffix-tree decoding (#183).
    Suffix,
    /// A separate, smaller draft model (`--spec-draft`, #184).
    Draft,
    /// EAGLE-3 head (`--spec-draft`, #185).
    Eagle3,
    /// DFlash block-diffusion drafter (`--spec-draft`, #186).
    Dflash,
    /// DSpark drafter (`--spec-draft`, #187).
    Dspark,
}

impl SpecKind {
    pub fn name(self) -> &'static str {
        match self {
            SpecKind::None => "none",
            SpecKind::Mtp => "mtp",
            SpecKind::Ngram => "ngram",
            SpecKind::Suffix => "suffix",
            SpecKind::Draft => "draft",
            SpecKind::Eagle3 => "eagle3",
            SpecKind::Dflash => "dflash",
            SpecKind::Dspark => "dspark",
        }
    }

    /// Whether the drafter is a [`ModelDrafter`] (else a [`TokenDrafter`]).
    pub fn is_model(self) -> bool {
        !matches!(self, SpecKind::None | SpecKind::Ngram | SpecKind::Suffix)
    }

    /// Whether the drafter loads weights of its own from `--spec-draft`.
    pub fn needs_weights(self) -> bool {
        matches!(
            self,
            SpecKind::Draft | SpecKind::Eagle3 | SpecKind::Dflash | SpecKind::Dspark
        )
    }

    /// Whether `--spec-draft` is allowed: always when needed, and for MTP, whose head may be
    /// a GGUF of its own (else it is read from the target's).
    pub fn takes_weights(self) -> bool {
        self.needs_weights() || self == SpecKind::Mtp
    }

    /// The model-free drafter for this kind: [`ngram::NgramDrafter`] (default n-gram sizes)
    /// or [`suffix::SuffixDrafter`] over one [`suffix::SuffixCache`] per factory, which every
    /// finished sequence of the engine is added to. `None` for model drafters.
    pub fn token_drafter(self) -> Option<TokenDrafterFactory> {
        match self {
            SpecKind::Ngram => Some(Arc::new(|| {
                Box::new(ngram::NgramDrafter::default()) as Box<dyn TokenDrafter>
            })),
            SpecKind::Suffix => {
                let cache = suffix::SuffixCache::default();
                Some(Arc::new(move || {
                    Box::new(SuffixSeq {
                        drafter: Some(suffix::SuffixDrafter::new(cache.clone())),
                        tokens: Vec::new(),
                    }) as Box<dyn TokenDrafter>
                }))
            }
            _ => None,
        }
    }
}

/// A [`suffix::SuffixDrafter`] that adds its sequence to the shared cache when the engine
/// drops it (the sequence finished).
struct SuffixSeq {
    drafter: Option<suffix::SuffixDrafter>,
    /// The sequence so far, as last proposed from, plus what was accepted since.
    tokens: Vec<u32>,
}

impl TokenDrafter for SuffixSeq {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        if let Some(d) = self.drafter.as_mut() {
            d.propose(tokens, max, out);
        }
    }

    fn accepted(&mut self, appended: &[u32]) {
        self.tokens.extend_from_slice(appended);
        if let Some(d) = self.drafter.as_mut() {
            d.accepted(appended);
        }
    }
}

impl Drop for SuffixSeq {
    fn drop(&mut self) {
        if let Some(d) = self.drafter.take() {
            d.finish(&self.tokens);
        }
    }
}

/// Which weights a DFlash / DSpark drafter drafts with (`--spec-draft-weights`). Drafts only:
/// every token is verified by the target, so the output does not depend on it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DraftWeights {
    /// The checkpoint's weights as stored.
    #[default]
    Model,
    /// Q8_0 copies of a Hugging Face checkpoint's 16- and 32-bit matrices, made at load (the
    /// backbone, `fc` and DFlash 2's selector projection; norms, convolution kernels,
    /// embeddings and head as stored). A GGUF drafter is read as stored.
    #[value(name = "q8_0")]
    Q8_0,
    /// 4-bit copies of the same matrices, made at load: Q4_K (Q4_0 for rows not a multiple of
    /// 256 wide).
    #[value(name = "q4")]
    Q4,
}

/// The output-head rows a drafter that borrows the target's head (DFlash, the MTP heads)
/// scores its drafts with (`--spec-draft-lm-head`). Drafts only: every token is verified by the target with its own
/// head, so the output does not depend on it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DraftLmHead {
    /// `q4` for the qwen4exp MTP head on an integrated GPU when the target's head is 16- or
    /// 32-bit (the NVFP4 checkpoint's BF16 one), `model` otherwise (and always for DFlash).
    #[default]
    Auto,
    /// The target's head as stored (its first `--spec-draft-vocab` rows, if given).
    Model,
    /// A 4-bit copy of those rows, made at load: Q4_K (Q4_0 for rows not a multiple of 256
    /// wide), with the head's scales folded in. NInfer's DFlash 2 drafts from a 4-bit
    /// shortlist head the same way.
    #[value(name = "q4")]
    Q4,
}

/// Speculative decoding settings of an engine.
#[derive(Clone)]
pub struct SpecConfig {
    pub kind: SpecKind,
    /// Max drafts per sequence and step (`--spec-k`).
    pub k: usize,
    /// Disable load and acceptance adaptation (`--spec-fixed`).
    pub fixed: bool,
    /// Disable drafts at this many decoding sequences (`--spec-max-batch`).
    pub max_batch: usize,
    /// Whether `max_batch` also gates the drafters whose depth follows measured costs (MTP
    /// heads, block drafters: [`depth`]) (`--spec-gate-cost-model`; default true).
    pub gate_cost_model: bool,
    /// Whether `--spec-gate-cost-model` was given (the engine then leaves `gate_until` at 0).
    pub gate_cost_model_given: bool,
    /// With `gate_cost_model`: the last batch size whose drafts the gate stops for those
    /// drafters, which draft again past it (0: no end). Set by the engine where the layout
    /// gains from it (`lm::vram::mtp_gate_until_default`, #814).
    pub gate_until: usize,
    /// MTP heads under the depth policy: charge each further draft what its verify row costs
    /// more than padding, measured per verify shape ([`depth::DepthModel::record_fill`];
    /// `--spec-row-cost`, default true).
    pub row_cost: bool,
    /// Verify steps pad every sequence's rows only to the step's longest draft, not to
    /// `1 + k` or the depth policy's depth (`--spec-pad-longest`; default false).
    pub pad_longest: bool,
    /// Drafters under the depth policy: the most rows a verify step takes (each drafting
    /// sequence's token and its drafts); the policy's depth is lowered at large batches to stay
    /// within it ([`adaptive::rows_depth_cap`], `--spec-max-rows`; 0: no cap).
    pub max_rows: usize,
    /// Minimum acceptance probability of the last draft (`--spec-p-floor`).
    pub p_floor: f32,
    /// Drafter weights (`--spec-draft`).
    pub draft: Option<PathBuf>,
    /// Overrides `kind`'s model-free drafter (tests, embedders).
    pub token_drafter: Option<TokenDrafterFactory>,
    /// Overrides `kind`'s model drafter (tests, embedders).
    pub model_drafter: Option<ModelDrafterFactory>,
    /// MTP: stop a sequence's drafting at the first draft the head gives less than this
    /// probability (`--spec-p-min`; 0 drafts `k` every step).
    pub p_min: f32,
    /// DFlash/DSpark draft temperature (`--spec-draft-temperature`; 0: greedy point masses).
    pub draft_temperature: f32,
    /// DSpark confidence threshold (`--spec-min-confidence`; None: never cut a block).
    pub min_confidence: Option<f32>,
    /// DFlash/DSpark: positions of context a sequence keeps for drafting
    /// (`--spec-draft-window`; 0: the drafter's default, 4096).
    pub draft_window: usize,
    /// DFlash/DSpark: the drafter's weights as stored or Q8_0 copies (`--spec-draft-weights`).
    pub draft_weights: DraftWeights,
    /// DFlash and the MTP heads: the output head rows they draft with, as stored or as a 4-bit
    /// copy (`--spec-draft-lm-head`).
    pub draft_lm_head: DraftLmHead,
    /// MTP: draft the continuation of an earlier occurrence of the sequence's last this many
    /// tokens or more, instead of the head's drafts, where that pays ([`mtp_ngram`],
    /// `--spec-mtp-ngram`; 0: off).
    pub mtp_ngram: usize,
    /// With `mtp_ngram`: the most n-gram drafts a step verifies (`--spec-mtp-ngram-k`; 0: `k`).
    /// Above `k`, verify steps may take this many drafts' rows while the head still drafts at
    /// most `k`.
    pub mtp_ngram_k: usize,
}

impl std::fmt::Debug for SpecConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecConfig")
            .field("kind", &self.kind)
            .field("k", &self.k)
            .field("fixed", &self.fixed)
            .field("max_batch", &self.max_batch)
            .field("gate_cost_model", &self.gate_cost_model)
            .field("gate_cost_model_given", &self.gate_cost_model_given)
            .field("gate_until", &self.gate_until)
            .field("row_cost", &self.row_cost)
            .field("pad_longest", &self.pad_longest)
            .field("max_rows", &self.max_rows)
            .field("p_floor", &self.p_floor)
            .field("draft", &self.draft)
            .field("token_drafter", &self.token_drafter.is_some())
            .field("model_drafter", &self.model_drafter.is_some())
            .field("p_min", &self.p_min)
            .field("draft_temperature", &self.draft_temperature)
            .field("min_confidence", &self.min_confidence)
            .field("draft_window", &self.draft_window)
            .field("draft_weights", &self.draft_weights)
            .field("draft_lm_head", &self.draft_lm_head)
            .field("mtp_ngram", &self.mtp_ngram)
            .field("mtp_ngram_k", &self.mtp_ngram_k)
            .finish()
    }
}

impl SpecConfig {
    /// Whether any drafting happens.
    pub fn enabled(&self) -> bool {
        self.k > 0
            && (self.kind != SpecKind::None
                || self.token_drafter.is_some()
                || self.model_drafter.is_some())
    }

    /// The n-gram's draft ceiling with `mtp_ngram`: `mtp_ngram_k`, else `k`.
    pub fn ngram_k(&self) -> usize {
        if self.mtp_ngram_k > 0 {
            self.mtp_ngram_k
        } else {
            self.k
        }
    }

    /// The most drafts a verify step takes: `k`, or the n-gram's ceiling if that is deeper.
    pub fn verify_k(&self) -> usize {
        if self.mtp_ngram > 0 {
            self.k.max(self.ngram_k())
        } else {
            self.k
        }
    }
}

/// Draft and acceptance counts. Acceptance rate is `accepted / drafted`; the mean accepted
/// length is the tokens a verify step appends on average, the bonus token included
/// (`1 + accepted / steps`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct SpecCounters {
    /// Draft tokens verified.
    pub drafted: u64,
    /// Of those, accepted.
    pub accepted: u64,
    /// Verify steps (steps in which the sequence had at least one draft).
    pub steps: u64,
}

impl SpecCounters {
    /// Counts one verify step of `drafted` drafts, `accepted` of them accepted.
    pub fn record(&mut self, drafted: usize, accepted: usize) {
        if drafted > 0 {
            self.drafted += drafted as u64;
            self.accepted += accepted as u64;
            self.steps += 1;
        }
    }

    pub fn add(&mut self, other: &SpecCounters) {
        self.drafted += other.drafted;
        self.accepted += other.accepted;
        self.steps += other.steps;
    }

    /// `accepted / drafted`, or `None` before any draft.
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }

    /// Tokens appended per verify step (accepted drafts + the bonus token), or `None` before
    /// any verify step.
    pub fn mean_accepted_length(&self) -> Option<f64> {
        (self.steps > 0).then(|| 1.0 + self.accepted as f64 / self.steps as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters() {
        let mut c = SpecCounters::default();
        assert_eq!(c.acceptance_rate(), None);
        assert_eq!(c.mean_accepted_length(), None);
        c.record(4, 3);
        c.record(0, 0); // a plain step: not counted
        c.record(4, 1);
        assert_eq!(
            c,
            SpecCounters {
                drafted: 8,
                accepted: 4,
                steps: 2
            }
        );
        assert_eq!(c.acceptance_rate(), Some(0.5));
        assert_eq!(c.mean_accepted_length(), Some(3.0));
    }

    #[test]
    fn kinds() {
        assert!(!SpecKind::Ngram.is_model());
        assert!(SpecKind::Mtp.is_model() && !SpecKind::Mtp.needs_weights());
        assert!(SpecKind::Eagle3.needs_weights());
        assert!(!SpecConfig::default().enabled());
        let c = SpecConfig {
            kind: SpecKind::Ngram,
            k: 4,
            ..Default::default()
        };
        assert!(c.enabled());
        assert!(!SpecConfig { k: 0, ..c }.enabled());
    }
}
