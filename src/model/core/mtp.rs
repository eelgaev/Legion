//! Multi-token-prediction (MTP) heads: one extra block (index `n_layer`) fed the target's
//! residual, run by the executor for `--spec mtp`. The architecture builds its head's graph
//! ([`Arch::build_mtp`]); the head's weights are in the model's GGUF or in one of their own
//! ([`Model::load_mtp`]).
//!
//! The head's KV cache uses the target's block layout: position `p` of a sequence goes to the
//! same cell as in the target's cache, in a pool holding only the head's layer, so the target's
//! block tables (and prefix sharing) serve it unchanged. A head may instead read the target's
//! own cache and write none ([`Arch::mtp_shares_kv`]).

use crate::ggml::{BufferType, Context, Graph, Tensor};
use crate::gguf::{GgufFile, TensorInfo};
use crate::weights::Weights;

use super::state::KvPool;
use super::{GroupShape, Inputs, Model, ModelError, StatePool};

/// Which weights the MTP head drafts with (`--spec-draft-head-type`). Drafts only: every token
/// is verified by the full model, so the output does not depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum DraftHeadType {
    /// `q8_0` for qwen35 / qwen35moe heads, and qwen4exp heads shipped 16-bit, on an
    /// integrated GPU (their BF16 heads are a large part of a draft pass there), the head as
    /// shipped otherwise.
    #[default]
    Auto,
    /// The head as shipped.
    Model,
    /// A Q8_0 copy of the head's 16- and 32-bit matrices, made at load ([`Model::requant_mtp`]).
    #[value(name = "q8_0")]
    Q8_0,
    /// An F16 copy of the head's BF16 and F32 matrices, made at load ([`Model::requant_mtp`]):
    /// GPUs without BF16 tensor cores (V100) run them as FP16 GEMMs instead of F32 ones.
    #[value(name = "f16")]
    F16,
}

/// What an MTP pass returns, per output row (sequences' last rows, in order).
#[derive(Debug, Clone, Default)]
pub struct MtpOutput {
    /// The head's greedy pick.
    pub argmax: Vec<u32>,
    /// Its probability under the head's softmax.
    pub pmax: Vec<f32>,
    /// The head's output residual, `[rows][hidden width]`: the input for the next draft.
    pub hidden: Vec<f32>,
}

/// The tensors of a built MTP pass.
pub struct MtpGraph<'a> {
    pub inputs: Inputs<'a>,
    /// F32 `[hidden width, n]`: per token, the residual paired with it.
    pub h: Tensor<'a>,
    pub pmax: Option<Tensor<'a>>,
    pub h_out: Option<Tensor<'a>>,
}

impl Model {
    /// Whether the weights include an MTP head (in the model's GGUF, or added by
    /// [`Self::load_mtp`]).
    pub fn has_mtp(&self) -> bool {
        self.hp.arch.has_mtp(self)
    }

    /// Adds the MTP head of a GGUF of its own (`blk.{n_layer}.*`, as the released Qwen3.8 head
    /// ships), placed by `place`. The architecture checks the file ([`Arch::mtp_head`]).
    pub fn load_mtp(
        &mut self,
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
    ) -> Result<(), ModelError> {
        self.hp.arch = self.hp.arch.mtp_head(&self.hp, gguf)?;
        self.mtp = Some(Weights::load(gguf, place)?);
        self.joined = Default::default();
        self.router_bf16 = Default::default();
        if !self.has_mtp() {
            self.mtp = None;
            return Err(ModelError::Metadata(format!(
                "{}: not an MTP head (its tensors are missing)",
                gguf.architecture().unwrap_or_default()
            )));
        }
        Ok(())
    }

    /// The width of the residual the MTP head reads (`n_embd × hyper-connection streams`, see
    /// [`super::Arch::mtp_streams`]).
    pub fn mtp_width(&self) -> usize {
        (self.hp.n_embd * self.hp.arch.mtp_streams()) as usize
    }

    /// Builds one MTP pass over `shape` in `ctx`: per token, its embedding and the residual
    /// in `h`; logits (argmax and its probability) and the output residual for the output rows
    /// (see [`Arch::build_mtp`]).
    pub(super) fn build_mtp<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        self.hp.arch.build_mtp(self, ctx, kv, rec, shape)
    }
}
