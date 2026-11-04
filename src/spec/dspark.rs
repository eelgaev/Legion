//! DSpark: NVIDIA's semi-autoregressive block drafter. `--spec dspark --spec-draft
//! <checkpoint>`.
//!
//! DSpark is DFlash ([`crate::spec::dflash`]) plus three things, all in the same code:
//!
//! - **A Markov head.** DFlash's block pass gives every position its own logits, independent
//!   of the tokens drafted before it. DSpark adds `markov_w2 · markov_w1[previous token]` to a
//!   position's logits (`vanilla` head: an embedding of `rank` floats per token and a
//!   projection back to the vocabulary), where the previous token is the token drafted just
//!   before it (the last token of the sequence for the first draft). So the positions are
//!   chosen in order, one small projection each, after the one parallel block pass. With
//!   sampling, the chain follows the sampled tokens, and each returned distribution is the one
//!   its draft was drawn from, which is what exact rejection needs.
//! - **A confidence head** (optional): the sigmoid of `proj` applied to the final-normed hidden
//!   state joined with `markov_w1[prev]`, plus a bias: a per-position estimate that the draft is
//!   accepted. Unused by default; with `--spec-min-confidence`
//!   ([`DflashOptions::min_confidence`]) a block is cut before the first position under the
//!   threshold.
//! - **Attention sinks** (optional, some backbones, e.g. Nemotron's): one learned logit per
//!   head that joins the attention softmax as a key with no value. The tensor is
//!   `blk.N.attn_sinks.weight`; conversions that leave it as `blk.N.attn_sinks` are read too.
//!
//! The block layout follows the checkpoint's `sample_from_anchor`: when set (DFlash lineage,
//! the default), the block is `[last token, <mask> x (n - 1)]` and all `n` positions give
//! drafts, the first from the last token's own position, so a block of size `B` drafts `B`
//! tokens; when not (EAGLE-3 lineage, a "bonus anchor"), position 0 is not read and the masks
//! give `B - 1` drafts, as in DFlash.
//!
//! # Checkpoints
//!
//! GGUF in llama.cpp's `dflash` architecture with `markov_w1.weight`, `markov_w2.weight` and
//! optionally `conf_proj.weight` / `conf_proj.bias`, and Hugging Face directories of the
//! DSpark classes (`markov_head.markov_w1.weight`, `confidence_head.proj.weight`, ...). NVIDIA's
//! released drafters are NVFP4: in a GGUF the matrices are `GGML_TYPE_NVFP4`, which ggml
//! multiplies natively (CPU and CUDA), so nothing here is special; NVFP4 safetensors (packed
//! `uint8` plus block scales) are not read, convert them to GGUF with llama.cpp. Only the
//! `vanilla` Markov head is implemented (the `gated` and `rnn` types of NeMo AutoModel are
//! refused), and a reduced draft vocabulary (`d2t`) works as for DFlash.
//!
//! # Known result
//!
//! On the AC922 with Nemotron as the target, DSpark's acceptance was under 10% and decode got
//! slower. This module implements the drafter correctly; whether and when to use it is for
//! load-adaptive control (#48).
//!
//! # Sources
//!
//! The method: the DSpark recipe in NVIDIA NeMo AutoModel (Apache-2.0,
//! `nemo_automodel/components/speculative/dspark/markov_head.py`: `VanillaMarkov`, the block
//! sampling loop that feeds each sampled token to the next position), and llama.cpp
//! (MIT; `src/models/dflash.cpp`, `common/speculative.cpp`, `conversion/qwen.py`) for the
//! checkpoint layout, tensor names, the confidence head's inputs and the anchor handling.

use std::sync::Arc;

use crate::spec::dflash::{DflashDrafter, DflashOptions, Flavor};
use crate::spec::{DraftRequest, ModelDrafter, ModelDrafterFactory, SeqId, TargetInfo};

/// Settings of a [`DsparkDrafter`]: those of DFlash, plus `min_confidence`.
pub type DsparkOptions = DflashOptions;

/// `--spec dspark`: makes the engine's drafter from `--spec-draft`.
pub fn factory(opts: DsparkOptions) -> ModelDrafterFactory {
    Arc::new(move |info: &TargetInfo| {
        DsparkDrafter::load(info, opts.clone()).map(|d| Box::new(d) as Box<dyn ModelDrafter>)
    })
}

pub struct DsparkDrafter(DflashDrafter);

impl DsparkDrafter {
    /// Loads the checkpoint at `target.draft` onto the target's device, and the token embeddings
    /// and output head from the target's GGUF if the checkpoint has none.
    pub fn load(target: &TargetInfo, opts: DsparkOptions) -> Result<Self, String> {
        DflashDrafter::load_as(target, opts, Flavor::Dspark).map(Self)
    }

    pub fn config(&self) -> &crate::spec::dflash::Config {
        self.0.config()
    }

    /// The most tokens one pass drafts.
    pub fn max_drafts(&self) -> usize {
        self.0.max_drafts()
    }

    /// Whether the checkpoint has a confidence head.
    pub fn has_confidence(&self) -> bool {
        self.0.has_confidence()
    }
}

impl ModelDrafter for DsparkDrafter {
    fn hidden_layers(&self) -> Vec<usize> {
        self.0.hidden_layers()
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        self.0.step(batch)
    }

    fn release(&mut self, seq: SeqId) {
        self.0.release(seq)
    }
}
