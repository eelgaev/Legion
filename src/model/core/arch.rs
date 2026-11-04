//! What an architecture adds to the shared graph builder: its layer kinds, the per-token and
//! per-group inputs it needs beyond tokens and positions, its residual scheme, attention
//! variants and MTP head, as hooks the core calls. Each architecture module (see
//! [`crate::model`]) implements [`Arch`] on its own hyperparameters; [`Hparams::arch`] holds
//! the model's. The defaults describe a plain stack: one residual stream, DeltaNet recurrent
//! layers, dense attention, no extra inputs.

use std::any::Any;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use crate::ggml::{Context, Graph, Tensor};
use crate::gguf::GgufFile;

use super::mtp::MtpGraph;
use super::state::KvPool;
use super::{
    Group, GroupShape, Hparams, InputSink, Inputs, Model, ModelError, Pass, SpecPool, StatePool,
};

/// The architecture of a model, shared by its [`Hparams`] clones; compares by
/// [`Arch::same`].
#[derive(Clone)]
pub struct ArchRef(Arc<dyn Arch>);

impl ArchRef {
    pub fn new(arch: impl Arch) -> Self {
        Self(Arc::new(arch))
    }
}

impl std::ops::Deref for ArchRef {
    type Target = dyn Arch;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl PartialEq for ArchRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.same(&*other.0)
    }
}

impl Debug for ArchRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The mixer of the recurrent layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recurrent {
    /// Gated DeltaNet: state `[head_v, head_v, n_v]` per slot.
    DeltaNet,
    /// Mamba-2's selective scan: state `[d_state, head_dim, n_head]` per slot.
    Mamba2,
    /// Kimi delta attention (GLM-5.3-Flash): state `[head_v, head_v, n_v]` per slot as gated
    /// DeltaNet's, a per-channel decay; its ops are the architecture's own, so a verify step's
    /// commit replays through [`Arch::spec_replay`].
    Kda,
}

/// A causal conv over per-token inputs of one layer, with its history kept per slot next to
/// the recurrent states and a token history of the last `n_hist` tokens per slot (a per-layer
/// n-gram embedding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenConv {
    pub layer: usize,
    /// Conv inputs kept, per slot.
    pub history: i64,
    /// Width of one conv input.
    pub width: i64,
    /// Tokens kept per slot.
    pub n_hist: usize,
}

/// The graph hooks of one architecture. `m` is the model the hook builds for; `self` is its
/// [`Hparams::arch`].
pub trait Arch: Debug + Send + Sync + Any {
    /// The GGUF architecture family, for messages.
    fn name(&self) -> &'static str;

    fn as_any(&self) -> &dyn Any;

    /// Whether `other` is the same architecture with the same hyperparameters.
    fn same(&self, other: &dyn Arch) -> bool;

    /// Drafts per speculative step when no explicit depth is requested; `None` uses the
    /// engine default.
    fn default_spec_k(&self) -> Option<usize> {
        None
    }

    // Layers.

    /// Whether layer `il` is a recurrent layer.
    fn is_recurrent(&self, hp: &Hparams, il: usize) -> bool;

    /// Whether layer `il` is an attention layer (has a KV cache).
    fn is_attention(&self, hp: &Hparams, il: usize) -> bool {
        !self.is_recurrent(hp, il)
    }

    /// `(head size, kv heads)` of attention layer `il`'s KV cache.
    fn kv_dims(&self, hp: &Hparams, _il: usize) -> (i64, i64) {
        (hp.head_dim, hp.n_head_kv)
    }

    /// The layer whose KV cache attention layer `il` reads.
    fn kv_source(&self, _hp: &Hparams, il: usize) -> usize {
        il
    }

    /// Attention layer `il`'s compact cache (#551), if it has one: the rotated key dimensions
    /// per head `r`. Its keys beyond them are its values times a per-dimension weight, so the
    /// cache keeps one granule of `head_dim + r` values per kv head and cell,
    /// `[K rotated (r) | V not rotated (head_dim - r) | V rotated (r)]`, in the V cache's type
    /// (see [`crate::model::core::KvPool`]). 0: a plain K and V cache.
    fn compact_kv(&self, _hp: &Hparams, _il: usize) -> i64 {
        0
    }

    /// Whether attention layer `il`'s cache needs a V of its own. False: nothing reads or
    /// writes its V, so the cache keeps K only, and its V (`cache_v`) is an alias of K's memory
    /// that code listing both still finds (see [`crate::model::core::KvPool`]).
    fn kv_has_v(&self, _hp: &Hparams, _il: usize) -> bool {
        true
    }

    /// Whether attention layer `il` attends over the last [`Self::swa_window`] positions only
    /// (its cache can be a ring, [`crate::model::core::KvRing`]).
    fn sliding_layer(&self, _hp: &Hparams, _il: usize) -> bool {
        false
    }

    /// The recurrent layers' mixer.
    fn recurrent(&self) -> Recurrent {
        Recurrent::DeltaNet
    }

    /// Whether the DeltaNet output is gated with `sigmoid(z)` rather than `silu(z)`.
    fn sigmoid_gate(&self) -> bool {
        false
    }

    // Inputs and state.

    /// Parallel copies of the residual stream (hyper-connections); the residual is
    /// `n_embd × this` wide.
    fn residual_streams(&self) -> i64 {
        1
    }

    /// Width of the per-token embedding rows a pipeline stage running `layers` reads (filled on
    /// the host, [`Inputs::ple`]), 0 for none.
    fn token_rows(&self, _hp: &Hparams, _layers: &Range<usize>) -> i64 {
        0
    }

    /// The per-token conv state kept per slot, if any.
    fn token_conv(&self, _hp: &Hparams) -> Option<TokenConv> {
        None
    }

    /// Whether the rows of [`Self::token_rows`] come from a hashed n-gram table that can be
    /// read through a lookup store ([`crate::residency::lookup_table`]).
    fn ngram_table(&self) -> bool {
        false
    }

    /// Whether a prompt step's graph can save its recurrent states (DeltaNet's, and the
    /// per-token conv's) at points inside the step ([`super::capture`], #46).
    fn state_captures(&self) -> bool {
        false
    }

    /// Whether the recurrent layers read their slots through slot ids ([`Inputs`]'s `seq_ids`).
    fn slot_ids(&self) -> bool {
        false
    }

    /// The sliding window of the sliding-window layers, 0 for none.
    fn swa_window(&self) -> usize {
        0
    }

    /// Input tensors per group (sparse-attention tables), each step; 0 for none.
    fn group_inputs_len(&self) -> usize {
        0
    }

    /// Group `i`'s inputs ([`Self::group_inputs_len`] of them) for its shape `g`.
    fn group_inputs<'a>(&self, _ctx: &'a Context, _g: &GroupShape, _i: usize) -> Vec<Tensor<'a>> {
        Vec::new()
    }

    /// The row map of a pass that is one verify group (I32 `[tokens]`,
    /// [`super::pad_rows::map`]), when the architecture's inputs hold one; `None` otherwise.
    /// The DeltaNet layers' delta rule then skips the padding rows (#836).
    fn verify_row_map<'a>(&self, _pass: &Pass<'a>) -> Option<Tensor<'a>> {
        None
    }

    /// Sparse-attention key pools a group lists and computes (see [`GroupShape`]).
    fn group_pools(&self, _g: &Group<'_>) -> (usize, usize) {
        (0, 0)
    }

    /// Sparse attention: no token of the group sees more pools than the indexer picks, so the
    /// step may select them all without scoring (see [`GroupShape`]).
    fn group_picks_all(&self, _g: &Group<'_>) -> bool {
        false
    }

    /// Fills the architecture's inputs of a step (after [`Inputs::fill`]) through `sink`.
    #[allow(clippy::too_many_arguments)]
    fn fill_inputs(
        &self,
        _m: &Model,
        _inputs: &Inputs<'_>,
        _groups: &[Group<'_>],
        _shape: &[GroupShape],
        _rec: &StatePool,
        _kv: &KvPool,
        _spec: Option<&SpecPool>,
        _sink: &mut dyn InputSink,
    ) {
    }

    // Attention.

    /// Width of a sparse-attention layer's cached indexer keys, 0 without.
    fn index_dim(&self) -> i64 {
        0
    }

    /// Whether attention layer `il` caches indexer keys (a sparse-attention layer).
    fn indexed(&self, _il: usize) -> bool {
        false
    }

    /// Sparse attention's reads (#391's KV bytes): `(keys a token selects, pool size)`; a
    /// paged step's sparse layers read those keys of each token plus its own incomplete pool.
    fn sparse_keys(&self) -> Option<(usize, usize)> {
        None
    }

    /// For an architecture with sparse attention: whether its CUDA kernels (and the paged
    /// kernels, for its dense attention layers) read a cache of `types`.
    fn sparse_reads(&self, _hp: &Hparams, types: super::KvTypes) -> bool {
        types.is_f16()
    }

    /// Whether steps over `kv` use paged attention, whatever their size.
    fn paged(&self, m: &Model, kv: &KvPool) -> bool {
        m.paged_default(kv)
    }

    /// Whether the attention always reads the cache in place through the block tables (its own
    /// ops, whatever `--attention` says): its sliding-window layers can be rings then.
    fn always_paged(&self) -> bool {
        false
    }

    /// Whether the step of `groups` over `kv` uses paged attention.
    fn paged_for(&self, m: &Model, kv: &KvPool, _groups: &[Group<'_>]) -> bool {
        self.paged(m, kv)
    }

    /// For a sparse-attention layer: caches this step's indexer keys and returns every
    /// token's indexer query. `None` for dense layers.
    fn index_query<'a>(
        &self,
        _m: &'a Model,
        _pass: &Pass<'a>,
        _il: usize,
        _h: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        Ok(None)
    }

    /// Sparse attention of group `gi` read in place (paged): the queries `q` over the pools
    /// `k`, `v` through the block `table`, selected with the indexer queries `iq`.
    #[allow(clippy::too_many_arguments)]
    fn sparse_paged<'a>(
        &self,
        _m: &'a Model,
        _pass: &Pass<'a>,
        _il: usize,
        _gi: usize,
        _iq: Tensor<'a>,
        _qkv: [Tensor<'a>; 3],
        _table: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        unreachable!("an architecture with indexer queries has sparse attention")
    }

    /// The gather path's mask of group `gi` for sparse attention: `mask` with the keys the
    /// indexer queries `iq` do not select masked out.
    fn sparse_mask<'a>(
        &self,
        _m: &'a Model,
        _pass: &Pass<'a>,
        _il: usize,
        _gi: usize,
        _iq: Tensor<'a>,
        _mask: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        unreachable!("an architecture with indexer queries has sparse attention")
    }

    // The graph.

    /// The token embeddings as the first layer reads them.
    fn embed<'a>(&self, _m: &'a Model, _ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
        x
    }

    /// The layers `layers` over the residual `x` (the embeddings, or the residual a previous
    /// pipeline stage handed over): returns the last layer's output and the outputs of the
    /// layers in `capture` (see [`Self::output_ready`] and [`Self::capture_normed`]).
    #[allow(clippy::type_complexity)]
    fn layers<'a>(
        &self,
        m: &'a Model,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        capture: &[usize],
        layers: Range<usize>,
    ) -> Result<(Tensor<'a>, Vec<(usize, Tensor<'a>)>), ModelError>;

    /// The embeddings of `tokens` (I32 `[n]`) when the architecture computes them itself
    /// (`None`: the core's `get_rows` of `token_embd.weight`, then [`Self::embed`]).
    fn embed_tokens<'a>(
        &self,
        _m: &'a Model,
        _ctx: &'a Context,
        _tokens: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        Ok(None)
    }

    /// The logits and their argmax of the head's input `x` (the last stage's output, normed
    /// and cropped as [`Self::output_ready`] says) when the architecture computes them itself
    /// (`None`: the core's matmul with the output weights, then [`Self::logits`]).
    #[allow(clippy::type_complexity)]
    fn head<'a>(
        &self,
        _m: &'a Model,
        _ctx: &'a Context,
        _x: Tensor<'a>,
    ) -> Result<Option<(Tensor<'a>, Tensor<'a>)>, ModelError> {
        Ok(None)
    }

    /// Whether the last stage's [`Self::layers`] output is already normed and cropped to the
    /// logits rows (the output head reads it as it is).
    fn output_ready(&self) -> bool {
        false
    }

    /// Whether a capture of the last layer returns its output after the final norm (what the
    /// MTP head reads); the logits then reuse it.
    fn capture_normed(&self) -> bool {
        false
    }

    /// Whether, with an MTP head drafting ([`crate::model::core::Executor::enable_mtp`]), a
    /// capture of the last layer returns its output after the final norm, which that head reads
    /// (Qwen3-Next's); other captures (a draft model's) keep the raw output.
    fn mtp_capture_normed(&self) -> bool {
        false
    }

    /// The logits as the model returns them (a soft-cap, say).
    fn logits<'a>(&self, _ctx: &'a Context, logits: Tensor<'a>) -> Tensor<'a> {
        logits
    }

    // MTP.

    /// Whether the architecture has an MTP head implementation ([`Self::build_mtp`]).
    fn mtp(&self) -> bool {
        false
    }

    /// Checks that `gguf` is an MTP head for a model of `hp` and returns the architecture as
    /// it runs with that head.
    fn mtp_head(&self, _hp: &Hparams, _gguf: &GgufFile) -> Result<ArchRef, ModelError> {
        Err(ModelError::Unsupported(format!(
            "an MTP head in a GGUF of its own for {}",
            self.name()
        )))
    }

    /// Whether `m`'s weights include an MTP head this architecture runs (in the model's GGUF,
    /// or added by [`Model::load_mtp`]). By default: block `n_layer` has `nextn.eh_proj`.
    fn has_mtp(&self, m: &Model) -> bool {
        self.mtp()
            && m.tensor(&format!("blk.{}.nextn.eh_proj.weight", m.hp.n_layer))
                .is_some()
    }

    /// Whether the MTP head reads the target's KV cache instead of keeping one of its own:
    /// the executor then hands [`Self::build_mtp`] the target's pool and allocates none.
    fn mtp_shares_kv(&self) -> bool {
        false
    }

    /// The target layers whose caches an MTP head sharing them ([`Self::mtp_shares_kv`])
    /// reads in every draft pass. A pool whose sliding-window rings spill to host memory
    /// allocates these rings first, so they stay in VRAM ([`super::KvPool::hot_first`]).
    fn mtp_kv_layers(&self, _hp: &Hparams) -> Vec<usize> {
        Vec::new()
    }

    /// Whether gather attention (its MTP head's too) reads a pool of sequence slots in place
    /// ([`super::state::KvPool::slot_span`], [`super::exec::Executor::set_kv_slot_blocks`]).
    fn reads_kv_slots(&self) -> bool {
        false
    }

    /// Whether an MTP pass's query at position `p` sees the key at `p` (the head wrote it). A
    /// head reading the target's cache drafts at a position the target has not written yet:
    /// its queries see the keys before it only.
    fn mtp_sees_own_position(&self) -> bool {
        true
    }

    /// Residual streams of the rows an MTP head reads and returns ([`Model::mtp_width`]): by
    /// default the residual's ([`Self::residual_streams`]). GLM-5.3-Flash's head reads the
    /// streams' normed mean, one stream.
    fn mtp_streams(&self) -> i64 {
        self.residual_streams()
    }

    /// Whether the MTP head's attention is sparse as the target's is: its passes then page by
    /// the target's rule ([`Model::paged_for`]) and take the per-group inputs
    /// ([`Self::group_inputs`], filled by [`Self::fill_inputs`]).
    fn mtp_sparse(&self) -> bool {
        false
    }

    /// The commit of a verify step's kept tokens into recurrent layer `il` of an architecture
    /// whose recurrent layers run its own ops ([`Recurrent::Kda`]): the nodes that advance the
    /// layer's states by them (see [`super::Replay`]).
    fn spec_replay<'a>(
        &self,
        _m: &'a Model,
        _ctx: &'a Context,
        _il: usize,
        _r: super::Replay<'a>,
    ) -> Result<Vec<Tensor<'a>>, ModelError> {
        Err(ModelError::Unsupported(format!(
            "speculative commits of {}'s recurrent layers",
            self.name()
        )))
    }

    /// Builds one MTP pass over `shape` (see [`Model::build_mtp`]).
    fn build_mtp<'a>(
        &self,
        _m: &'a Model,
        _ctx: &'a Context,
        _kv: &'a KvPool,
        _rec: &'a StatePool,
        _shape: &'a [GroupShape],
    ) -> Result<(Graph<'a>, MtpGraph<'a>), ModelError> {
        Err(ModelError::Unsupported(format!(
            "the MTP head of {}",
            self.name()
        )))
    }

    /// Whether external model drafters (a draft model, EAGLE-3, DFlash) can draft for it.
    fn external_drafters(&self) -> bool {
        true
    }

    /// Whether the step's device argmax of a logits row is always the host's
    /// [`crate::engine::sampling::argmax`] of that row (the first of the largest, logits without
    /// NaNs): then a greedy row read in full (`min_tokens` masking) need not be scanned on the
    /// host. Not for the core's head: on CUDA it takes ggml-cuda's `argmax_f32` order among
    /// equal maxima ([`crate::model::cluster_select::argmax`]).
    fn argmax_first_of_ties(&self) -> bool {
        false
    }

    /// Whether decode over an expert cache on GPUs lets each stage GPU's idle NVLink peer fetch
    /// missed experts and stages them on a side stream by default, on any split whose GPUs all
    /// pair up ([`crate::model::core::expert_mv::peer::auto_paired`]), not only on two GPUs.
    fn peer_decode_pairs(&self) -> bool {
        false
    }
}

/// [`Arch::same`] for an architecture whose hyperparameters are `T`.
pub(crate) fn same_as<T: PartialEq + 'static>(this: &T, other: &dyn Arch) -> bool {
    other.as_any().downcast_ref::<T>() == Some(this)
}
