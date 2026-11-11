//! The model-neutral core every architecture runs on: [`Model`] (weights and the graph builder),
//! the [`Executor`] and its pools (paged KV, recurrent states), paged and gather attention, the
//! speculative verify and MTP machinery, [`Linear`] and the expert matvec. The architectures'
//! own parts live in their modules under [`crate::model`], behind the hooks of [`Arch`]: layer
//! kinds, extra inputs, the residual scheme, attention variants, the MTP head. This module
//! builds the shared blocks: gated DeltaNet, gated attention with multi-section RoPE, the MoE
//! layer with a gated shared expert, the dense SwiGLU FFN, and the batch step around the
//! architecture's layer stack.
//!
//! [`Model::forward`] runs a batch of tokens against a [`SeqState`] (F16 KV cache for the attention
//! layers, conv + delta-rule state for the DeltaNet layers): the whole prompt for prefill, then
//! one token per decode step. Graph structure follows llama.cpp's `src/models/qwen35moe.cpp`,
//! `qwen35.cpp` and `delta-net-base.cpp` (MIT), op for op. On the CPU backend the logits are bit-identical to
//! llama.cpp run with flash attention and weight repacking off (see `tests/qwen35.rs`); with
//! llama.cpp's defaults they differ by rounding only, which Q8_0 activation quantization in the
//! quantized matmuls amplifies layer by layer (a 1-ulp input change can flip a rounding).

pub(crate) mod arch;
pub mod capture;
pub(crate) mod decode_fuse;
pub use decode_fuse::set_decode_fuse;
pub(crate) mod node_fuse;
pub use node_fuse::set_node_fuse;
mod draft_head;
mod exec;
pub mod expert_mv;
pub(crate) mod f16_weight;
pub(crate) mod fused;
pub(crate) mod gate_q8;
pub use gate_q8::{set_gate_norm, set_gate_q8};
pub(crate) mod gdn;
pub use fused::{set_delta_prep, set_gated_norm, set_moe_combine, set_router_join, set_shexp_gate};
pub(crate) mod gdn_mma;
pub(crate) mod gdn_scan;
pub(crate) mod graph_size;
#[cfg(feature = "cuda")]
pub use gdn::bench as gdn_bench;
pub use gdn::set_gdn;
pub use gdn::set_gdn_decode;
pub use gdn::set_gdn_pad_skip;
pub use gdn::set_gdn_replay_staged;
pub use gdn_mma::set_gdn_mma;
pub use gdn_scan::set_gdn_scan;
pub(crate) use graph_size::graph_size;
pub use norm_rope::{set_kv_append, set_norm_rope};
pub use paged::qm_bench as paged_qm_bench;
pub use paged::wmma_bench as paged_split_bench;
pub(crate) mod shape_diff;
pub(crate) mod shared_q8;
pub use shared_q8::set_shared_q8;
pub use shared_q8::small::set_shared_q8_small;
mod joined;
pub use joined::{
    joined_matmuls_built, set_joined_matmuls,
    set_prompt_min_tokens as set_joined_prompt_min_tokens, set_prompt_qkv as set_joined_prompt_qkv,
    set_prompt_z as set_joined_prompt_z,
};
pub mod kv_type;
mod linear;
pub mod media;
pub mod mmq_ids;
pub use mmq_ids::set_mmq_ids;
pub(crate) mod mmq_preq;
pub use mmq_preq::set_mmq_preq;
pub(crate) mod norm_rope;
pub mod nvfp4_once;
pub use nvfp4_once::set_nvfp4_once;
pub mod head_mm;
pub use head_mm::set_head_mm;
pub mod nvfp4_mvq;
pub use nvfp4_mvq::set_nvfp4_mvq;
pub mod q4k_head;
pub use q4k_head::set_q4k_head;
pub mod q8_group;
pub mod q8_head;
pub mod q8_mv;
pub(crate) use linear::{bf16_cast_joined, bf16_join_fits, f16_joined, has_precision_hint};
pub use q8_group::set_q8_group;
pub use q8_head::set_q8_head;
pub use q8_mv::set_q8_mv;
pub(crate) mod mtp;
pub mod mtp_moe;
pub mod pad_rows;
pub(crate) mod paged;
pub mod pair;
pub(crate) mod picks;
mod spec;
pub(crate) mod stable_fusions;
pub mod stage_ops;
pub use stable_fusions::set_stable_fusions;
pub mod route_rec;
pub(crate) mod router_bf16;
pub use router_bf16::set_router_bf16;
pub(crate) mod router_mv;
pub use router_mv::set_router_mv;
pub(crate) mod mmvq_glu;
pub use mmvq_glu::set_mmvq_glu;
pub(crate) mod state;
pub mod tp;
pub(crate) mod tp_ep;
pub mod traffic;
pub mod turboquant;
mod window_attn;

pub use arch::{Arch, ArchRef, Recurrent, TokenConv};
pub use exec::{
    ChainOutput, Executor, LENT_SCRATCH, Pool, PromptNext, Remote, StateCapture, StepPart, TpLead,
    default_verify_budget, from_wire, set_mtp_kept_graphs, set_prompt_buffer_reserve,
    set_prompt_build_ahead, set_prompt_step_stats, set_verify_kept_graphs, verify_kept_graphs,
};
pub use kv_type::{KvType, KvTypes};
pub use linear::{FfnFusion, Fp8Act, Linear, Nvfp4A4, Nvfp4Act, Nvfp4Mm};
pub use media::{ImageSpan, Media};
pub use mtp::{DraftHeadType, MtpGraph, MtpOutput};
pub use spec::{
    Replay, SpecPool, set_commit_in_verify, set_commit_slots, set_state_defer,
    set_state_defer_min_seqs, set_state_defer_rows,
};
pub use state::{KV_BLOCK, KvPool, KvRing, KvTier, SeqState, SplitTier, StatePool, fnv1a};

use crate::ggml::{
    Backend, BufferType, ComputeAllocator, Context, GgmlError, Graph, Tensor, types,
};
use crate::gguf::{GgufFile, TensorInfo};
use crate::model::dense_mv;
use crate::model::fp8_act;
use crate::model::moe_ffn;
use crate::residency::cpu_share;
use crate::weights::{LoadError, Weights};
use picks::{Need, Picks, took};

/// `GGML_ROPE_TYPE_IMROPE`: interleaved multi-section RoPE.
pub(crate) const ROPE_IMROPE: i32 = 40;

/// Attention runs over a key span padded to a multiple of this (masked, zero K/V), as llama.cpp's
/// KV cache view is (`n_kv` is padded to 256). The padding changes nothing mathematically, but
/// the soft_max and V·softmax reductions then run over the same length as llama.cpp's, and the
/// SIMD summation order (hence the last bit) depends on that length. One flipped bit there can
/// flip an F16 rounding of a softmax weight and, through Q8_0 activation quantization, grow
/// into ~1 logit of drift over 40 layers.
const ATTN_KV_PAD: i64 = 256;

/// Most bytes of logits rows [`Inputs::out_pin_bytes`] sizes the pinned output buffer for, so
/// that [`Inputs::output_with`] reads runs of rows in one DMA ([`picks::spans`]; a 16-user verify
/// step's 48 rows of a 248K vocabulary are ~48 MB); rows further apart come back in more reads.
const OUT_SPAN_BYTES: usize = 64 << 20;

/// Tensors and nodes a step's graph of up to [`graph_size::GRAPH_SIZE_GROUPS`] groups may hold
/// ([`graph_size()`] sizes bigger steps' graphs).
pub(crate) const GRAPH_SIZE: usize = 16384;

/// Smallest step [`Model::share_bf16`] shares a BF16 cast in: prompt chunks only, so decode and
/// verify graphs (up to 16 sequences of 1 + 3 drafted tokens) stay as they were.
const SHARE_BF16_MIN_N: i64 = 128;

#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    /// Decoder layers run in the main pass (the GGUF's `block_count` minus MTP layers).
    pub n_layer: usize,
    pub n_embd: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_rot: i32,
    pub rope_sections: [i32; 4],
    pub rope_freq_base: f32,
    pub n_ctx_train: i32,
    pub rms_eps: f32,
    /// `qwen35moe` (routed experts + shared expert) rather than dense `qwen35`.
    pub moe: bool,
    /// 0 for the dense model.
    pub n_expert: i64,
    /// 0 for the dense model.
    pub n_expert_used: i64,
    pub full_attention_interval: usize,
    pub ssm_d_conv: i64,
    /// Head size of the DeltaNet keys (= queries).
    pub ssm_d_state: i64,
    /// Number of key heads.
    pub ssm_n_group: i64,
    /// Number of value heads.
    pub ssm_dt_rank: i64,
    pub ssm_d_inner: i64,
    pub n_vocab: i64,
    /// The architecture: its own hyperparameters and graph hooks. For Mamba-2 recurrent
    /// layers the `ssm_*` fields describe the mixer: `ssm_d_state` its state size,
    /// `ssm_n_group` its B/C groups, `ssm_dt_rank` its heads. With per-layer head sizes
    /// `head_dim` and `n_rot` are the global layers'; [`Self::kv_dims`] gives every layer's.
    pub arch: ArchRef,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("architecture `{0}` is not {list}", list = crate::model::ARCHITECTURES)]
    Architecture(String),
    #[error("{0} is not supported")]
    Unsupported(String),
    #[error("missing or invalid metadata `{0}`")]
    Metadata(String),
    #[error("missing tensor `{0}`")]
    Tensor(String),
    #[error("FP8 expert weight `{0}` is not supported yet; pass --hf-fp8 bf16")]
    Fp8Unsupported(String),
    #[error("{0}")]
    Layout(String),
    #[error("{needed} tokens do not fit the {max_ctx}-token context")]
    ContextFull { needed: usize, max_ctx: usize },
    #[error(
        "out of GPU memory on {device}: a {tokens}-token step needs a {need_mib} MiB compute \
         buffer and {free_mib} MiB are free with the other step graphs dropped; lower \
         --max-batch-tokens, or leave more memory free with a smaller --expert-cache-gib"
    )]
    StepOutOfMemory {
        device: String,
        tokens: usize,
        need_mib: usize,
        free_mib: usize,
    },
    /// The link between the nodes of a model split over machines ([`crate::node`]).
    #[error("{0}")]
    Node(String),
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
}

impl Hparams {
    pub fn from_gguf(g: &GgufFile) -> Result<Self, ModelError> {
        let arch = g.architecture().unwrap_or_default();
        crate::model::parse_hparams(g, &arch)
    }

    /// Whether layer `il` is a recurrent layer (gated DeltaNet, or Mamba-2).
    pub fn is_recurrent(&self, il: usize) -> bool {
        self.arch.is_recurrent(self, il)
    }

    /// Whether layer `il` is an attention layer (has a KV cache).
    pub fn is_attention(&self, il: usize) -> bool {
        self.arch.is_attention(self, il)
    }

    /// Whether any layer has a recurrent state (without, a sequence is its KV cache).
    pub fn has_recurrent(&self) -> bool {
        (0..self.n_layer).any(|il| self.is_recurrent(il))
    }

    /// Whether a sequence has state in its slot beyond its KV cache: recurrent layers or a
    /// per-token conv ([`Arch::token_conv`]).
    pub fn has_slot_state(&self) -> bool {
        self.has_recurrent() || self.arch.token_conv(self).is_some()
    }

    /// `(head size, kv heads)` of attention layer `il`'s KV cache: the model's own, unless the
    /// architecture's layers differ (sliding-window and global layers).
    pub fn kv_dims(&self, il: usize) -> (i64, i64) {
        self.arch.kv_dims(self, il)
    }

    /// The layer whose KV cache attention layer `il` reads: itself, but for shared-KV layers.
    pub fn kv_source(&self, il: usize) -> usize {
        self.arch.kv_source(self, il)
    }

    /// Rotated key dimensions per head of attention layer `il`'s compact cache, 0 for a plain
    /// one ([`Arch::compact_kv`]).
    pub fn compact_kv(&self, il: usize) -> i64 {
        self.arch.compact_kv(self, il)
    }

    /// Whether attention layer `il`'s cache has a V of its own ([`Arch::kv_has_v`]).
    pub fn kv_has_v(&self, il: usize) -> bool {
        self.arch.kv_has_v(self, il)
    }

    /// Bytes per cell of attention layer `il`'s own cache, K and V in `types`: a compact
    /// cache's granules ([`Arch::compact_kv`]) in the V type, K only without a V
    /// ([`Arch::kv_has_v`]).
    pub fn kv_cell_bytes(&self, il: usize, types: KvTypes) -> usize {
        let (d, n_kv) = self.kv_dims(il);
        match self.compact_kv(il) {
            0 if !self.kv_has_v(il) => types.k.bytes(d as usize, (d * n_kv) as usize),
            0 => types.row_bytes(d as usize, (d * n_kv) as usize),
            // Never a TurboQuant type: granules are not power-of-two heads.
            r => types.v.bytes((d + r) as usize, ((d + r) * n_kv) as usize),
        }
    }

    /// Whether attention layer `il` is a sliding-window layer ([`Arch::sliding_layer`]).
    pub fn sliding_layer(&self, il: usize) -> bool {
        self.arch.sliding_layer(self, il)
    }

    /// Whether layer `il` writes a KV cache of its own.
    pub fn owns_kv(&self, il: usize) -> bool {
        self.is_attention(il) && self.kv_source(il) == il
    }

    /// The recurrent state of one layer and slot: `[head_v, head_v, n_v]` for gated DeltaNet,
    /// `[d_state, head_dim, n_head]` for Mamba-2. Either way `ssm_d_inner` times some rows.
    pub fn ssm_state_ne(&self) -> [i64; 3] {
        match self.arch.recurrent() {
            Recurrent::Mamba2 => {
                let n_head = self.ssm_dt_rank;
                [self.ssm_d_state, self.ssm_d_inner / n_head, n_head]
            }
            Recurrent::DeltaNet | Recurrent::Kda => {
                let head_v = self.ssm_d_inner / self.ssm_dt_rank;
                [head_v, head_v, self.ssm_dt_rank]
            }
        }
    }
}

/// Last-token logits plus named intermediates (see [`Model::prompt_trace`]).
pub type Trace = (Vec<f32>, Vec<TracedTensor>);

/// One intermediate tensor captured by [`Model::prompt_trace`], named as llama.cpp's eval
/// callback names it (`Qcur-3`, `ffn_moe_topk-3`, `l_out-39`, ...).
#[derive(Debug, Clone)]
pub struct TracedTensor {
    pub name: String,
    /// ggml order, all four dims.
    pub ne: [i64; 4],
    /// Row-major in ggml order (dim 0 fastest). I32 tensors (expert ids) are converted exactly.
    pub data: Vec<f32>,
}

/// Collects named intermediates while a graph is built; does nothing unless enabled.
pub(crate) struct Tap<'a> {
    pub(crate) ctx: &'a Context,
    pub(crate) on: bool,
    pub(crate) taps: std::cell::RefCell<Vec<(String, Tensor<'a>)>>,
}

impl<'a> Tap<'a> {
    pub(crate) fn tap(&self, name: &str, il: usize, t: Tensor<'a>) {
        if self.on {
            let t = if t.is_contiguous() {
                t
            } else {
                self.ctx.cont(t)
            };
            self.taps
                .borrow_mut()
                .push((format!("{name}-{il}"), t.as_output()));
        }
    }
}

pub struct Model {
    /// Tiered routed experts (VRAM cache over host memory), when set up. First, so it is
    /// dropped before the weights its ranges map.
    experts: Option<crate::residency::experts::ExpertResidency>,
    /// Routed experts read from the checkpoint as steps route to them, when set up
    /// ([`crate::residency::ssd_experts`]; GLM-5.3-Flash NVFP4).
    ssd: Option<crate::residency::ssd_experts::SsdExperts>,
    /// Weight pairs read as one matrix by small steps ([`joined`]), made by the first graph
    /// that asks. Before the weights, whose memory they share.
    joined: std::cell::OnceCell<Option<joined::Joined>>,
    /// BF16 copies of the MoE routers whose values are exact in BF16 ([`router_bf16`]), read by
    /// small steps' router ops; made at load ([`Self::prepare_router_bf16`]) or by the first
    /// graph that asks. Each copy is checked against the router it was made from on use.
    router_bf16: std::cell::OnceCell<Option<router_bf16::RouterBf16>>,
    /// Host memory of the weights pinned for GPUs that cannot read pageable memory
    /// ([`Self::keep_host_pins`]). Before the weights, so it is released while they are mapped.
    host_pins: Option<Box<dyn std::any::Any + Send + Sync>>,
    pub hp: Hparams,
    pub(crate) weights: Weights,
    /// An MTP head loaded from a GGUF of its own ([`Self::load_mtp`]).
    mtp: Option<Weights>,
    attention: Attention,
    /// How routed experts run over a batch.
    moe_kernel: MoeKernel,
    /// How the fused expert op keeps per-pair outputs (see [`moe_ffn::PairOut`]).
    moe_pair_out: moe_ffn::PairOut,
    /// MTP drafts over the first this many tokens of the vocabulary only (0: all).
    draft_vocab: usize,
    /// The MTP head attends to the last this many positions only (0: all).
    mtp_window: usize,
    /// Per-tensor scales and precision hints of the weight matmuls (NVFP4 checkpoints).
    pub(crate) linear: Linear,
    /// BF16 copies of NVFP4 weights ([`Self::widen_nvfp4`]), read instead of them.
    widened: Option<Context>,
    /// Requantized copies of the MTP head's matrices ([`Self::requant_mtp`]), read instead of
    /// them.
    requant: Option<Context>,
    /// The 4-bit copy of the output head the MTP head drafts with
    /// ([`Self::quantize_draft_head`]).
    draft_head: Option<Context>,
    /// Where the per-layer n-gram table's rows come from when it is read through a store
    /// ([`crate::residency::lookup_table`]: RAM, page cache or direct SSD reads); `None`: from
    /// the weights' own host memory.
    pub(crate) ngram: Option<crate::residency::lookup_table::LookupTable>,
    /// Whether a step's fill leaves the n-gram rows it reads through `ngram` to its compute,
    /// and the rows the last one left, their reads issued (qwen4exp's `set_ple_late`).
    pub(crate) ple_late: std::sync::Mutex<(bool, Option<crate::model::qwen4::LateRows>)>,
    /// The n-gram rows of a pipelined prompt's next micro-chunk, gathered while the stages
    /// compute (qwen4exp's `gather_ple_ahead`).
    pub(crate) ple_ahead: std::sync::Mutex<crate::model::qwen4::PleAhead>,
    /// Gather attention reads the cache cell by cell instead of block by block (see
    /// [`Self::set_cell_gather`]).
    cell_gather: bool,
    /// The KV pools' host-tier staging slots in blocks (0: none), which sizes prefill groups'
    /// paged-attention scratch (see [`Self::set_kv_stage_blocks`]).
    kv_stage_blocks: usize,
    /// Sequence-slot attention's staging (#1153, see [`Self::set_slot_stage`]):
    /// `(rows, from, upto)`, rows 0 for off.
    slot_stage: (usize, usize, usize),
    /// The communicator of tensor parallelism over machines ([`crate::tp_net`]): this rank's
    /// layers hold its share of the heads and FFN rows, and their partial outputs are summed
    /// over the ranks (see [`Self::set_tp_comm`]).
    tp_comm: Option<i32>,
    /// Tensor parallelism over the two GPUs of an NVLink pair in one process (`--tp-per-gpu`):
    /// the pair's all-reduce and this model's rank in it (see [`Self::set_tp_pair`]).
    tp_pair: Option<(std::sync::Arc<pair::sum::PairSum>, usize)>,
    /// Expert parallelism of the routed experts under tensor parallelism (`--tp-moe-ep`,
    /// [`tp_ep`]): this rank's experts and their lookup tables.
    tp_ep: Option<tp_ep::TpEp>,
    /// GLM's parallel decode (`--glm-parallel-decode`): who does what and the GPUs' exchange.
    pub(crate) glm_par: Option<crate::model::glm5next::par::GlmPar>,
    /// Pair parallelism of the routed experts (`--pair-tp`, [`pair`]).
    pair: Option<pair::PairTp>,
}

/// How the routed experts of a batch run (`--moe-kernel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum MoeKernel {
    /// The fused op on CUDA for batches of at least [`MOE_FFN_MIN_TOKENS`] tokens, where it
    /// supports the expert weights and they are in VRAM (staged, for the expert cache's
    /// uncached ones); ggml's `mul_mat_id` path elsewhere.
    #[default]
    Auto,
    /// Always ggml's `mul_mat_id` path (on the CPU bit-identical to llama.cpp).
    Graph,
    /// The fused op wherever it supports the weights, read in place if not staged; on the CPU
    /// a slow reference, for tests.
    Fused,
}

/// Smallest batch the fused expert op takes in [`MoeKernel::Auto`]: below it, ggml's
/// `mul_mat_id` (its MMQ / MMVQ kernels for small batches) is as fast.
pub const MOE_FFN_MIN_TOKENS: i64 = 64;

/// Where a weight lives when the model runs on a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The backend's own memory.
    Device,
    /// Host memory, mapped from the file. The per-layer embedding table always: only the host
    /// reads it (a few hundred random rows per step), and mapped it costs page cache rather than
    /// 27 GiB that could not be reclaimed. DeepSeek-V4.1-Flash's Engram tables (hashed rows the
    /// host gathers). Routed experts when asked to, for GPUs that read host
    /// memory in place (unified memory: GB10, GH200), so the file's pages are the only copy.
    Host,
}

impl Model {
    /// Placement of tensor `name` on a GPU; `host_experts` keeps the routed experts mapped.
    pub fn placement(name: &str, host_experts: bool) -> Placement {
        // DeepSeek-V4.1-Flash's Engram tables and hash parameters too: the host hashes and
        // gathers their rows.
        let engram = name.starts_with("engram_")
            || name.ends_with(".engram_embd.weight")
            || name.ends_with(".engram_embd.scale");
        if name == "per_layer_token_embd.weight"
            || engram
            || (host_experts && name.contains("_exps."))
        {
            Placement::Host
        } else {
            Placement::Device
        }
    }
}

/// How attention reads the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Attention {
    /// Paged on CUDA when the kernels support the model's head shape, gather elsewhere.
    #[default]
    Auto,
    /// Gather each sequence's blocks into a contiguous span, then mul_mat / soft_max /
    /// mul_mat. On the CPU this is bit-identical to llama.cpp without flash attention.
    Gather,
    /// Read the blocks in place ([`paged`]). On the CPU a slow reference, for tests.
    Paged,
}

/// Which logits a sequence of a batch step returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Logits {
    /// None (a prefill chunk that does not end the prompt).
    None,
    /// The last token's.
    Last,
    /// Only the argmax of the last token's (greedy sampling), computed on the device: no
    /// vocabulary-sized row to copy back and scan.
    Greedy,
    /// Every token's.
    All,
    /// The first `n` tokens' (a speculative verify step: the real tokens before the padding).
    First(usize),
    /// Only the argmaxes of the first `n` tokens' (a greedy verify step).
    FirstGreedy(usize),
    /// Only the greedy pick of the last token's with the stop tokens masked (a greedy sequence
    /// below `min_tokens`, [`picks`]): the argmax, or the host's pick where it may differ.
    Pick,
    /// Only the greedy picks of the first `n` tokens' (a greedy verify step below `min_tokens`,
    /// [`picks`]), the first `m` with the stop tokens masked.
    FirstPick(usize, usize),
}

impl Logits {
    fn rows(self, n_tok: usize) -> usize {
        match self {
            Logits::None => 0,
            Logits::Last | Logits::Greedy | Logits::Pick => 1,
            Logits::All => n_tok,
            Logits::First(n) | Logits::FirstGreedy(n) | Logits::FirstPick(n, _) => n.min(n_tok),
        }
    }

    /// Whether the rows come back whole (for the engine to sample from), not as argmaxes or
    /// picks.
    fn sampled(self) -> bool {
        matches!(self, Logits::Last | Logits::All | Logits::First(_))
    }

    /// Whether the rows are the first tokens' rather than the last ones'.
    fn leading(self) -> bool {
        matches!(
            self,
            Logits::First(_) | Logits::FirstGreedy(_) | Logits::FirstPick(..)
        )
    }

    /// Whether only the last token's row is asked for (or none).
    fn last_only(self) -> bool {
        matches!(
            self,
            Logits::None | Logits::Last | Logits::Greedy | Logits::Pick
        )
    }
}

/// One sequence's part of a batch step.
#[derive(Debug, Clone)]
pub struct SeqInput<'b> {
    pub tokens: &'b [u32],
    /// Position of `tokens[0]` (tokens already in the cache).
    pub n_past: usize,
    /// The sequence's KV blocks in order; they must cover positions `0..n_past + tokens.len()`.
    pub blocks: &'b [u32],
    /// The sequence's ring of the sliding-window layers' caches ([`KvRing`]), when the pool
    /// has rings; ignored otherwise.
    pub ring: usize,
    pub logits: Logits,
    /// The sequence's images ([`Media`]): their tokens' input rows and the positions of every
    /// token. `None`: text only, positions `[p, p, p, 0]`.
    pub media: Option<&'b Media>,
}

/// Sequences of a batch step that run their DeltaNet layers together: the same number of
/// tokens each (as ggml's `ssm_conv` / `gated_delta_net` need), recurrent states in consecutive
/// slots `slot..slot + seqs.len()`. A decode step is one group of one-token sequences; a prefill
/// chunk is a group of one sequence.
#[derive(Debug, Clone)]
pub struct Group<'b> {
    pub slot: usize,
    pub seqs: Vec<SeqInput<'b>>,
    /// A speculative verify step: the recurrent states in the group's slots stay as they are
    /// and the step's replay inputs are stored instead, for [`Executor::commit`] to advance
    /// each state by the tokens it keeps (see `spec.rs`). Needs [`Executor::enable_spec`].
    pub verify: bool,
}

/// What a batch step's graph depends on besides input data: equal shapes, same graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupShape {
    pub(crate) slot: usize,
    pub(crate) n_seqs: usize,
    pub(crate) n_tok: usize,
    /// Gather attention: the span (keys per sequence), padded to ATTN_KV_PAD. 0 for paged
    /// attention, whose graph does not depend on it.
    pub(crate) n_kv: usize,
    /// Paged attention: block-table entries per sequence. 0 for gather attention.
    pub(crate) table: usize,
    /// Paged attention over a pool with rings: blocks per ring ([`KvRing`]), 0 without.
    pub(crate) ring: usize,
    /// Logits rows per sequence.
    logits: Vec<usize>,
    /// No sequence reads an output row other than its last token's (logits none, the last
    /// token's or its argmax): what [`paged::attention_last`] computes is enough.
    pub(crate) last_only: bool,
    pub(crate) verify: bool,
    /// Sparse attention: key pools listed (padded), pools to (re)compute this step (padded);
    /// 0 without (see [`Arch::group_pools`]).
    pub(crate) n_pool: usize,
    pub(crate) n_new: usize,
    /// Sparse attention: every token sees at most the indexer's budget of pools, so it attends
    /// to all of them and the step needs no scores (see [`Arch::group_picks_all`]).
    pub(crate) pick_all: bool,
    /// Gather attention over a pool of sequence slots ([`KvPool::slot_span`]): each
    /// sequence's slot, whose span attention reads in place. Empty otherwise.
    pub(crate) kv_slots: Vec<u32>,
    /// A step of GLM's parallel decode ([`crate::model::glm5next::par`]): every stage's graph
    /// runs every layer, its own share of the work. Kept apart from the pipeline's graphs.
    pub(crate) par: bool,
    /// A decode or verify step under pair parallelism (`--pair-tp`, [`pair`]): the stage's
    /// paired MoE layers share their experts with the other GPU of its NVLink pair. Kept apart
    /// from the other steps' graphs.
    pub(crate) pair: bool,
    /// Image tokens of the group's sequences ([`Media`]), whose input rows replace their token
    /// embeddings.
    pub(crate) img: usize,
    /// A prompt group's recurrent states saved inside the step ([`capture`]), ascending.
    pub(crate) captures: Vec<capture::Capture>,
}

impl GroupShape {
    /// `table`: the block-table width for paged attention, `None` for gather attention.
    /// `arch`: the architecture whose sparse-attention pools the group needs, if any.
    fn of(g: &Group<'_>, table: Option<usize>, arch: Option<&dyn Arch>) -> Self {
        let n_tok = g.seqs.first().map_or(0, |s| s.tokens.len());
        assert!(n_tok > 0, "empty group");
        assert!(
            g.seqs.iter().all(|s| s.tokens.len() == n_tok),
            "sequences of a group need the same number of tokens"
        );
        let end = g.seqs.iter().map(|s| s.n_past + n_tok).max().unwrap_or(0);
        if let Some(width) = table {
            assert!(
                end.div_ceil(KV_BLOCK) <= width,
                "{end} positions do not fit a {width}-block table"
            );
        }
        let (n_pool, n_new) = arch.map_or((0, 0), |a| a.group_pools(g));
        Self {
            slot: g.slot,
            n_seqs: g.seqs.len(),
            n_tok,
            n_kv: match table {
                Some(_) => 0,
                None => end.div_ceil(ATTN_KV_PAD as usize) * ATTN_KV_PAD as usize,
            },
            table: table.unwrap_or(0),
            ring: 0,
            logits: g.seqs.iter().map(|s| s.logits.rows(n_tok)).collect(),
            last_only: g.seqs.iter().all(|s| s.logits.last_only()),
            verify: g.verify,
            n_pool,
            n_new,
            pick_all: arch.is_some_and(|a| a.group_picks_all(g)),
            kv_slots: Vec::new(),
            par: false,
            pair: false,
            img: g
                .seqs
                .iter()
                .map(|s| {
                    s.media
                        .map_or(0, |m| m.image_tokens(s.n_past..s.n_past + n_tok))
                })
                .sum(),
            captures: Vec::new(),
        }
    }

    pub(crate) fn tokens(&self) -> usize {
        self.n_seqs * self.n_tok
    }

    /// A short description of a step's shape for `--stage-graph-stats`' build log: per group
    /// its first slot, sequences x tokens, KV span or table, listed and new pools, logits rows
    /// and flags.
    pub(crate) fn brief(shape: &[GroupShape]) -> String {
        shape
            .iter()
            .map(|g| {
                format!(
                    "s{}:{}x{} kv{} tb{} p{}/{} l{}{}{}{}",
                    g.slot,
                    g.n_seqs,
                    g.n_tok,
                    g.n_kv,
                    g.table,
                    g.n_pool,
                    g.n_new,
                    g.out_rows(),
                    if g.verify { " v" } else { "" },
                    if g.pick_all { " all" } else { "" },
                    if g.kv_slots.is_empty() {
                        String::new()
                    } else {
                        format!(" ks{:?}", g.kv_slots)
                    },
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    pub(crate) fn out_rows(&self) -> usize {
        self.logits.iter().sum()
    }
}

/// The shape of a whole batch step (its groups'), for graph reuse. `table`: the block-table
/// width for paged attention, `None` for gather attention; `arch`: the architecture whose
/// sparse-attention pools the step needs, `None` for none (an MTP pass).
pub fn batch_shape(
    groups: &[Group<'_>],
    table: Option<usize>,
    arch: Option<&dyn Arch>,
) -> Vec<GroupShape> {
    groups
        .iter()
        .map(|g| GroupShape::of(g, table, arch))
        .collect()
}

/// `shape`, of `groups`, reading `kv`: its groups take ring tables (paged) or ring block lists
/// (gather) too when `kv` has rings, and their sequences' slots when `kv` is in slots (the
/// slot of a sequence's first block).
pub(crate) fn with_kv(
    mut shape: Vec<GroupShape>,
    groups: &[Group<'_>],
    kv: &KvPool,
) -> Vec<GroupShape> {
    if let Some(r) = kv.ring() {
        for g in shape.iter_mut() {
            g.ring = r.blocks;
        }
    }
    let per = kv.slot_blocks() as u32;
    if per > 0 && shape.iter().all(|g| g.table == 0) {
        for (sh, g) in shape.iter_mut().zip(groups) {
            sh.kv_slots = g.seqs.iter().map(|s| s.blocks[0] / per).collect();
        }
    }
    shape
}

/// A group's attention inputs.
#[derive(Clone, Copy)]
pub(crate) enum AttnInputs<'a> {
    /// Mask `[n_kv, n_tok, 1, n_seqs]` and the blocks to gather, `[n_kv / KV_BLOCK × n_seqs]`.
    /// With sliding-window layers (Gemma 4) the mask is `[n_kv, n_tok, 1, 2 × n_seqs]`: the
    /// causal masks, then the sliding-window ones (see [`gather_mask`]).
    ///
    /// With rings ([`KvRing`]), `ring` lists, for the sliding-window layers, the ring entries
    /// that hold the positions they read, and `swa` is their mask `[n_swa, n_tok, 1, n_seqs]`
    /// (`mask` then has the causal half only). With `blocks` per block, the same positions as
    /// `blocks` (one entry per block, `n_swa` = `n_kv`): a position in the window is there,
    /// any other one is masked by the window. With `blocks` per cell, only the last `n_swa`
    /// positions of each sequence's span ([`swa_span`]: the window before its tokens), one
    /// entry per cell: a sliding layer gathers its window, not the whole context.
    Gather {
        mask: Tensor<'a>,
        blocks: Tensor<'a>,
        ring: Option<Tensor<'a>>,
        swa: Option<Tensor<'a>>,
    },
    /// Block tables `[table, n_seqs]`, each sequence's first position `[n_seqs]`, the decode
    /// kernel's scratch (never filled), and with rings the ring tables `[ring, n_seqs]` of the
    /// sliding-window layers ([`KvRing`]).
    Paged {
        table: Tensor<'a>,
        n_past: Tensor<'a>,
        scratch: Tensor<'a>,
        ring: Option<Tensor<'a>>,
    },
}

impl<'a> AttnInputs<'a> {
    pub(crate) fn list(self) -> Vec<Tensor<'a>> {
        match self {
            AttnInputs::Gather {
                mask,
                blocks,
                ring,
                swa,
            } => [mask, blocks].into_iter().chain(ring).chain(swa).collect(),
            AttnInputs::Paged {
                table,
                n_past,
                scratch,
                ring,
            } => [table, n_past, scratch].into_iter().chain(ring).collect(),
        }
    }
}

/// A batch step's input tensors, filled after allocation and before compute.
pub struct Inputs<'a> {
    pub(crate) tokens: Tensor<'a>,
    pub(crate) pos: Tensor<'a>,
    /// KV cell of every token (I32 `[n]`).
    pub(crate) cells: Tensor<'a>,
    /// Rows to compute logits for, when not all (I32).
    pub(crate) out_rows: Option<Tensor<'a>>,
    /// With rings: every token's cell in its sequence's ring (I32 `[n]`, [`KvRing::cell`]).
    pub(crate) ring_cells: Option<Tensor<'a>>,
    /// Per group.
    pub(crate) attn: Vec<AttnInputs<'a>>,
    pub(crate) logits: Option<Tensor<'a>>,
    /// Argmax of every logits row (I32).
    pub(crate) argmax: Option<Tensor<'a>>,
    /// Outputs of the captured layers ([`Executor::capture_hidden`]), `[n_embd, n]` each.
    pub(crate) hidden: Vec<Tensor<'a>>,
    /// Every token's per-layer embedding rows, gathered on the host (F32 `[width, n]`, see
    /// [`Arch::token_rows`]).
    pub(crate) ple: Option<Tensor<'a>>,
    /// Per group, the architecture's inputs ([`Arch::group_inputs`]); empty without.
    pub(crate) group: Vec<Vec<Tensor<'a>>>,
    /// `0, 1, ..` (I32, as many as the largest group's sequences), the slot ids of the
    /// selective scan (each group reads its slots through a view of them; [`Arch::slot_ids`]).
    pub(crate) seq_ids: Option<Tensor<'a>>,
    /// A pipeline stage after the first: the residual stream coming in, F32 `[width, n]`
    /// (`n_embd`, times the hyper-connection streams).
    res_in: Option<Tensor<'a>>,
    /// A stage before the last: the residual stream going out, `[width, n]`.
    res_out: Option<Tensor<'a>>,
    /// The first stage of a step with image tokens ([`GroupShape::img`]): their input rows
    /// (F32 `[n_embd, n_img]`) and the step rows they replace (I32 `[n_img]`).
    img: Option<(Tensor<'a>, Tensor<'a>)>,
    /// The sliding window of the gather masks' second half, 0 for none (see [`AttnInputs`]).
    swa: usize,
    /// The gather masks hide each query's own position too: it sees the keys before it only
    /// (an MTP head that reads the target's cache, where that position is not written yet;
    /// see [`Arch::mtp_sees_own_position`]).
    pub(crate) exclusive: bool,
}

/// What a graph of a pipeline stage holds besides the always-present inputs, for
/// [`Inputs::from_list`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputParts {
    /// Slot ids of the selective scan.
    pub seq_ids: bool,
    pub n_hidden: usize,
    pub ple: bool,
    /// Inputs per group.
    pub group: usize,
    /// The stage computes logits (the last stage).
    pub head: bool,
    pub res_in: bool,
    pub res_out: bool,
    /// The sliding window, 0 for none.
    pub swa: usize,
}

/// What a batch step returns, per logits row (in the order of groups, sequences, tokens).
#[derive(Debug, Clone, Default)]
pub struct StepOutput {
    /// Argmax of every row.
    pub argmax: Vec<u32>,
    /// The full row (`n_vocab` floats), or empty for [`Logits::Greedy`] rows.
    pub rows: Vec<Vec<f32>>,
    /// Per captured layer ([`Executor::capture_hidden`]): its output for every token of the
    /// step, `[tokens][n_embd]` in the order of groups, sequences, tokens.
    pub hidden: Vec<Vec<f32>>,
    /// The same rows left on the device instead ([`Executor::set_hidden_on_device`], #1181):
    /// per captured layer, its `[n_embd, tokens]` output in the step graph's buffer, valid
    /// until the executor runs anything again. `hidden` then holds empty vectors. Empty when
    /// the rows were read back.
    pub hidden_dev: Vec<crate::ggml::RowsAt>,
}

impl StepOutput {
    /// Rows concatenated (`[rows][n_vocab]`), for the single-sequence API.
    fn flat(self) -> Vec<f32> {
        self.rows.concat()
    }
}

impl<'a> Inputs<'a> {
    /// All tensors, in a fixed order (see [`Self::from_list`]).
    fn list(&self) -> Vec<Tensor<'a>> {
        let mut v = vec![self.tokens, self.pos, self.cells];
        v.extend(self.out_rows);
        v.extend(self.ring_cells);
        v.extend(self.attn.iter().flat_map(|a| a.list()));
        v.extend(self.logits);
        v.extend(self.argmax);
        v.extend(&self.hidden);
        v.extend(self.ple);
        for g in &self.group {
            v.extend(g);
        }
        v.extend(self.seq_ids);
        v.extend(self.res_in);
        v.extend(self.res_out);
        if let Some((rows, idx)) = self.img {
            v.extend([rows, idx]);
        }
        v
    }

    /// Every input to fill, so a graph keeps them all (and the allocator gives them memory)
    /// even where a stage does not read some.
    pub(crate) fn expand_into(&self, graph: &mut Graph<'a>) {
        for t in self.to_fill() {
            graph.expand(t);
        }
    }

    /// [`Self::expand_into`] for a graph that may read none of some inputs (an MTP take-in
    /// piece's): those are kept as a dependency node's sources ([`Graph::expand_unread`]), so
    /// the allocator sees them grow (#527).
    pub(crate) fn expand_unread_into(&self, graph: &mut Graph<'a>) {
        graph.expand_unread(&self.to_fill());
    }

    fn to_fill(&self) -> Vec<Tensor<'a>> {
        let mut all = vec![self.tokens, self.pos, self.cells];
        all.extend(self.out_rows);
        all.extend(self.ring_cells);
        all.extend(self.attn.iter().flat_map(|a| a.list()));
        all.extend(self.ple);
        for g in &self.group {
            all.extend(g);
        }
        all.extend(self.seq_ids);
        all.extend(self.res_in);
        if let Some((rows, idx)) = self.img {
            all.extend([rows, idx]);
        }
        all
    }

    /// The inputs of a graph built for `shape` (`n_hidden` captured layers; `ple`: the model
    /// has per-layer embedding rows; `group`: inputs per group; `seq_ids`: Mamba-2 slot ids;
    /// `swa`: the sliding window), from [`Self::list`]'s order.
    #[allow(clippy::too_many_arguments)]
    fn from_list(
        shape: &[GroupShape],
        n_hidden: usize,
        ple: bool,
        group: usize,
        seq_ids: bool,
        swa: usize,
        t: impl Fn(usize) -> Tensor<'a>,
    ) -> Self {
        let parts = InputParts {
            n_hidden,
            ple,
            group,
            seq_ids,
            head: true,
            res_in: false,
            res_out: false,
            swa,
        };
        Self::from_parts(shape, parts, t)
    }

    /// [`Self::from_list`] for a graph of any stage.
    fn from_parts(
        shape: &[GroupShape],
        parts: InputParts,
        t: impl Fn(usize) -> Tensor<'a>,
    ) -> Self {
        // A parallel step's graph on every GPU runs all the layers (`Model::glm_par_stage`):
        // no residual in or out; the last stage's has the head and the captured head input.
        let mut parts = parts;
        if shape.first().is_some_and(|g| g.par) {
            parts.res_in = false;
            parts.res_out = false;
            if !parts.head {
                parts.n_hidden = 0;
            }
        }
        let InputParts {
            n_hidden,
            ple,
            group,
            seq_ids,
            head,
            ..
        } = parts;
        let n: usize = shape.iter().map(GroupShape::tokens).sum();
        let n_out: usize = shape.iter().map(GroupShape::out_rows).sum();
        let mut i = 3;
        let mut next = || {
            i += 1;
            t(i - 1)
        };
        let out_rows = (n_out > 0 && n_out < n).then(&mut next);
        let ring_cells = shape.iter().any(|g| g.ring > 0).then(&mut next);
        let attn = shape
            .iter()
            .map(|g| {
                if g.table > 0 {
                    AttnInputs::Paged {
                        table: next(),
                        n_past: next(),
                        scratch: next(),
                        ring: (g.ring > 0).then(&mut next),
                    }
                } else {
                    AttnInputs::Gather {
                        mask: next(),
                        blocks: next(),
                        ring: (g.ring > 0).then(&mut next),
                        swa: (g.ring > 0).then(&mut next),
                    }
                }
            })
            .collect();
        let logits = (head && n_out > 0).then(&mut next);
        let argmax = (head && n_out > 0).then(&mut next);
        let hidden = (0..n_hidden).map(|_| next()).collect();
        let ple = ple.then(&mut next);
        let group = if group > 0 {
            shape
                .iter()
                .map(|_| (0..group).map(|_| next()).collect())
                .collect()
        } else {
            Vec::new()
        };
        let seq_ids = seq_ids.then(&mut next);
        let res_in = parts.res_in.then(&mut next);
        let res_out = parts.res_out.then(&mut next);
        // As [`Model::stage_inputs`] makes them: the first stage's, when images are in the step.
        let img = (!parts.res_in && shape.iter().any(|g| g.img > 0)).then(|| (next(), next()));
        Self {
            res_in,
            res_out,
            img,
            swa: parts.swa,
            exclusive: false,
            tokens: t(0),
            pos: t(1),
            cells: t(2),
            out_rows,
            ring_cells,
            attn,
            logits,
            argmax,
            hidden,
            ple,
            group,
            seq_ids,
        }
    }

    /// Reads the step's results back: every row's argmax, and the rows that want logits.
    fn output(&self, groups: &[Group<'_>], n_vocab: usize) -> StepOutput {
        self.output_with(groups, n_vocab, None, Picks::default(), |t| t.to_f32())
    }

    /// The bytes of the step's argmaxes (0 without them): what [`Self::output_with`] reads
    /// them through needs this much room.
    fn argmax_bytes(&self) -> usize {
        self.argmax.map_or(0, |t| t.nbytes())
    }

    /// The bytes [`Self::output_with`] reads the argmaxes and then the logits rows of `groups`
    /// through: the biggest of [`Self::argmax_bytes`], one row of `n_vocab` F32s and the span
    /// from the first to the last row that comes back whole ([`Logits::sampled`]), when that is
    /// at most [`OUT_SPAN_BYTES`] (0 without logits).
    fn out_pin_bytes(&self, groups: &[Group<'_>], n_vocab: usize) -> usize {
        let Some(_) = self.logits else {
            return self.argmax_bytes();
        };
        let row = n_vocab * 4;
        let (mut first, mut end, mut i) = (None, 0, 0);
        for s in groups.iter().flat_map(|g| &g.seqs) {
            let n = s.logits.rows(s.tokens.len());
            if n > 0 && s.logits.sampled() {
                first.get_or_insert(i);
                end = i + n;
            }
            i += n;
        }
        let span = first.map_or(0, |f| (end - f) * row);
        let span = if span <= OUT_SPAN_BYTES { span } else { 0 };
        self.argmax_bytes().max(row).max(span)
    }

    /// [`Self::output`], reading the captured rows with `read`, and the argmaxes and logits rows
    /// through host tensor `ids` (pinned memory, [`Tensor::to_i32_via`],
    /// [`Tensor::with_f32_range_via`]) when it has room for them ([`Self::out_pin_bytes`]).
    /// Of a greedy pick row ([`Logits::Pick`], [`Logits::FirstPick`]) only what `picks` needs comes
    /// back ([`picks::Need`]): the pick then stands in its argmax's place and its row stays empty.
    /// Rows next to each other come back in one read when they fit ([`picks::spans`]: one DMA,
    /// not one per row).
    fn output_with(
        &self,
        groups: &[Group<'_>],
        n_vocab: usize,
        ids: Option<Tensor<'_>>,
        picks: Picks<'_>,
        mut read: impl FnMut(Tensor<'a>) -> Vec<f32>,
    ) -> StepOutput {
        let hidden = self.hidden.iter().map(|&t| read(t)).collect();
        let (Some(logits), Some(argmax)) = (self.logits, self.argmax) else {
            return StepOutput {
                hidden,
                ..Default::default()
            };
        };
        let argmax = match ids.filter(|v| v.nbytes() >= argmax.nbytes()) {
            Some(via) => argmax.to_i32_via(via),
            None => argmax.to_i32(),
        };
        let mut argmax: Vec<u32> = argmax.into_iter().map(|t| t as u32).collect();
        let mut needs = Vec::with_capacity(argmax.len());
        for s in groups.iter().flat_map(|g| &g.seqs) {
            for i in 0..s.logits.rows(s.tokens.len()) {
                let t = argmax.get(needs.len()).copied().unwrap_or(u32::MAX);
                needs.push(Need::of(s.logits, i, t, n_vocab, picks));
            }
        }
        let mut rows = vec![Vec::new(); needs.len()];
        let wanted: Vec<usize> = (0..needs.len())
            .filter(|&r| needs[r].len(n_vocab) > 0)
            .collect();
        // Each read brings back a run of rows' parts (one DMA through `ids`, [`picks::spans`]).
        let parts: Vec<(usize, usize)> = (wanted.iter())
            .map(|&r| (r * n_vocab, needs[r].len(n_vocab)))
            .collect();
        let cap = ids.map_or(0, |v| v.nbytes() / 4);
        // Picks that need their whole row after all.
        let mut whole = Vec::new();
        for span in picks::spans(&parts, picks::SPAN_GAP, cap) {
            let rs = &wanted[span.clone()];
            let start = parts[span.start].0;
            let (last, len) = parts[span.end - 1];
            let n = last + len - start;
            let mut take = |r: usize, values: &[f32]| {
                if !took(
                    needs[r],
                    values,
                    &mut argmax[r],
                    &mut rows[r],
                    n_vocab,
                    picks,
                ) {
                    whole.push(r);
                }
            };
            match ids.filter(|v| v.nbytes() >= n * 4) {
                // SAFETY: the closure only reads the slice; nothing writes `via` meanwhile.
                Some(via) => unsafe {
                    logits.with_f32_range_via(via, start, n, |v| {
                        for &r in rs {
                            let at = r * n_vocab - start;
                            take(r, &v[at..at + needs[r].len(n_vocab)]);
                        }
                    })
                },
                None => {
                    for &r in rs {
                        take(r, &logits.f32_range(r * n_vocab, needs[r].len(n_vocab)));
                    }
                }
            }
        }
        for r in whole {
            let Need::Pick { masked, .. } = needs[r] else {
                continue;
            };
            let need = Need::Pick {
                len: n_vocab,
                masked,
            };
            let v = match ids.filter(|v| v.nbytes() >= n_vocab * 4) {
                Some(via) => logits.f32_range_via(via, r * n_vocab, n_vocab),
                None => logits.f32_range(r * n_vocab, n_vocab),
            };
            took(need, &v, &mut argmax[r], &mut rows[r], n_vocab, picks);
        }
        StepOutput {
            argmax,
            rows,
            hidden,
            hidden_dev: Vec::new(),
        }
    }

    /// Writes the step's data into the (allocated) inputs.
    fn fill(&self, groups: &[Group<'_>], shape: &[GroupShape]) {
        self.fill_into(groups, shape, &mut SetNow);
    }

    /// [`Self::fill`] through `sink`: the same values, written as `sink` writes them.
    fn fill_into(&self, groups: &[Group<'_>], shape: &[GroupShape], sink: &mut impl InputSink) {
        let n: usize = shape.iter().map(GroupShape::tokens).sum();
        let mut ids = Vec::with_capacity(n);
        let mut positions: Vec<[i32; 4]> = Vec::with_capacity(n);
        let mut img_rows = Vec::new();
        let mut img_idx = Vec::new();
        let mut cells = Vec::with_capacity(n);
        let mut ring_cells = Vec::new();
        let mut out = Vec::new();
        for (g, sh) in groups.iter().zip(shape) {
            let ring = KvRing {
                blocks: sh.ring,
                sets: 1,
            };
            for s in &g.seqs {
                for (i, &t) in s.tokens.iter().enumerate() {
                    let p = s.n_past + i;
                    if let Some(row) = s.media.and_then(|m| m.row(p)) {
                        img_idx.push(ids.len() as i32);
                        img_rows.extend_from_slice(row);
                    }
                    ids.push(t as i32);
                    positions.push(match s.media {
                        Some(m) => m.rope(p),
                        None => [p as i32, p as i32, p as i32, 0],
                    });
                    let b = s.blocks[p / KV_BLOCK] as usize;
                    cells.push((b * KV_BLOCK + p % KV_BLOCK) as i32);
                    if sh.ring > 0 {
                        ring_cells.push(ring.cell(s.ring, p) as i32);
                    }
                }
                let k = s.logits.rows(sh.n_tok);
                let first = if s.logits.leading() {
                    ids.len() - sh.n_tok
                } else {
                    ids.len() - k
                };
                out.extend((first..first + k).map(|r| r as i32));
            }
        }
        sink.i32(self.tokens, &ids);
        // The four RoPE sections one after another: [p, p, p, 0] for text tokens, an image's
        // rows and columns for its tokens ([`Media::rope`]).
        let mut p = Vec::with_capacity(4 * n);
        for s in 0..4 {
            p.extend(positions.iter().map(|x| x[s]));
        }
        sink.i32(self.pos, &p);
        if let Some((rows, idx)) = self.img {
            sink.f32(rows, &img_rows);
            sink.i32(idx, &img_idx);
        }
        sink.i32(self.cells, &cells);
        if let Some(t) = self.ring_cells {
            sink.i32(t, &ring_cells);
        }
        if let Some(t) = self.seq_ids {
            let ids: Vec<i32> = (0..t.ne()[0] as i32).collect();
            sink.i32(t, &ids);
        }
        if let Some(t) = self.out_rows {
            sink.i32(t, &out);
        }
        for ((g, sh), attn) in groups.iter().zip(shape).zip(&self.attn) {
            let (mask, gather, ring, swa_mask) = match *attn {
                AttnInputs::Gather {
                    mask,
                    blocks,
                    ring,
                    swa,
                } => (mask, blocks, ring, swa),
                AttnInputs::Paged {
                    table,
                    n_past,
                    ring,
                    ..
                } => {
                    if let Some(rt) = ring {
                        let r = KvRing {
                            blocks: sh.ring,
                            sets: 1,
                        };
                        let t: Vec<i32> = (g.seqs.iter())
                            .flat_map(|s| (0..sh.ring).map(move |i| r.block(s.ring, i) as i32))
                            .collect();
                        sink.i32(rt, &t);
                    }
                    let mut t = Vec::with_capacity(sh.table * sh.n_seqs);
                    for s in &g.seqs {
                        let used = (s.n_past + sh.n_tok).div_ceil(KV_BLOCK);
                        t.extend(s.blocks[..used].iter().map(|&b| b as i32));
                        // Entries past the span are never read.
                        t.resize(t.len() + sh.table - used, 0);
                    }
                    sink.i32(table, &t);
                    // Exclusive: the kernels' causal limit one position earlier (a query at 0
                    // keeps key 0, as the masks below).
                    let ex = usize::from(self.exclusive);
                    let p: Vec<i32> = (g.seqs.iter())
                        .map(|s| s.n_past.saturating_sub(ex) as i32)
                        .collect();
                    sink.i32(n_past, &p);
                    continue;
                }
            };
            let n_kv = sh.n_kv;
            let n_blk = n_kv / KV_BLOCK;
            // The sliding-window masks in their own tensor (rings), else a second half.
            let halves = if self.swa > 0 && swa_mask.is_none() {
                2
            } else {
                1
            };
            // Read in place from the sequences' slots: no blocks to list (a placeholder).
            let slots = !sh.kv_slots.is_empty();
            let mut m = vec![0f32; n_kv * sh.n_tok * sh.n_seqs * halves];
            // One entry per block, or per cell when the gather reads cells (see
            // `Model::set_cell_gather`).
            let per_cell = gather.ne()[0] as usize == n_kv * sh.n_seqs;
            let mut blk = Vec::with_capacity(n_kv * sh.n_seqs);
            for (si, s) in g.seqs.iter().enumerate() {
                // Causal: token i (at n_past + i) sees keys 0..=n_past + i (0..n_past + i when
                // exclusive, keeping key 0 for a query at 0); padding never.
                let end = |p: usize| if self.exclusive { p.max(1) } else { p + 1 };
                for q in 0..sh.n_tok {
                    let row = &mut m[(si * sh.n_tok + q) * n_kv..][..n_kv];
                    row[end(s.n_past + q)..].fill(f32::NEG_INFINITY);
                }
                // Sliding window: also keys `window` or more positions back (llama.cpp's
                // `is_masked_swa`, standard type).
                if self.swa > 0 && swa_mask.is_none() {
                    let half = n_kv * sh.n_tok * sh.n_seqs;
                    for q in 0..sh.n_tok {
                        let p = s.n_past + q;
                        let row = &mut m[half + (si * sh.n_tok + q) * n_kv..][..n_kv];
                        row[end(p)..].fill(f32::NEG_INFINITY);
                        row[..(p + 1).saturating_sub(self.swa)].fill(f32::NEG_INFINITY);
                    }
                }
                let used = (s.n_past + sh.n_tok).div_ceil(KV_BLOCK);
                // Padding blocks are masked; any block with finite contents will do.
                let block = |b: usize| s.blocks[b.min(used - 1)] as i32;
                if per_cell {
                    let kb = KV_BLOCK as i32;
                    blk.extend((0..n_blk).flat_map(|b| (0..kb).map(move |c| block(b) * kb + c)));
                } else if !slots {
                    blk.extend((0..n_blk).map(block));
                }
            }
            sink.f32(mask, &m);
            // With slots no op reads the placeholder: it has no memory.
            if !slots {
                sink.i32(gather, &blk);
            }
            if let (Some(rt), Some(sm)) = (ring, swa_mask) {
                // The sliding layers' keys through each sequence's ring: the same blocks as the
                // gather (one entry per block), or per cell the last `n_swa` positions of the
                // sequence's span (`swa_span`).
                let r = KvRing {
                    blocks: sh.ring,
                    sets: 1,
                };
                let n_swa = sm.ne()[0] as usize;
                let per_seq = rt.ne()[0] as usize / sh.n_seqs;
                let per_cell = per_seq == n_swa;
                let mut list = Vec::with_capacity(per_seq * sh.n_seqs);
                let mut sm_rows = vec![f32::NEG_INFINITY; n_swa * sh.n_tok * sh.n_seqs];
                let end = |p: usize| if self.exclusive { p.max(1) } else { p + 1 };
                for (si, s) in g.seqs.iter().enumerate() {
                    // Key j of this sequence is at position `first + j`.
                    let first = if per_cell {
                        (s.n_past + sh.n_tok).saturating_sub(n_swa)
                    } else {
                        0
                    };
                    for j in 0..per_seq {
                        list.push(if per_cell {
                            r.cell(s.ring, first + j) as i32
                        } else {
                            r.block(s.ring, j) as i32
                        });
                    }
                    for q in 0..sh.n_tok {
                        let p = s.n_past + q;
                        let row = &mut sm_rows[(si * sh.n_tok + q) * n_swa..][..n_swa];
                        // Positions in (p - window, p] (or before p, exclusive) are seen.
                        let lo = (p + 1).saturating_sub(self.swa).max(first);
                        let hi = end(p).min(first + n_swa);
                        if lo < hi {
                            row[lo - first..hi - first].fill(0.0);
                        }
                    }
                }
                sink.i32(rt, &list);
                sink.f32(sm, &sm_rows);
            }
        }
    }
}

/// Where [`Inputs::fill_into`] writes the inputs' values.
pub trait InputSink {
    fn i32(&mut self, t: Tensor<'_>, values: &[i32]);
    fn f32(&mut self, t: Tensor<'_>, values: &[f32]);
    /// All of `t`, any type.
    fn bytes(&mut self, t: Tensor<'_>, bytes: &[u8]);
}

/// Sets every input right away ([`Tensor::set_i32`], [`Tensor::set_f32`]).
pub(crate) struct SetNow;

impl InputSink for SetNow {
    fn i32(&mut self, t: Tensor<'_>, values: &[i32]) {
        t.set_i32(values);
    }

    fn f32(&mut self, t: Tensor<'_>, values: &[f32]) {
        t.set_f32(values);
    }

    fn bytes(&mut self, t: Tensor<'_>, bytes: &[u8]) {
        t.set_bytes(bytes);
    }
}

/// Graph-building context for one forward pass.
pub struct Pass<'a> {
    pub(crate) ctx: &'a Context,
    pub(crate) graph: std::cell::RefCell<Graph<'a>>,
    pub(crate) tap: Tap<'a>,
    pub(crate) kv: &'a KvPool,
    pub(crate) rec: &'a StatePool,
    /// The saved states, which [`GroupShape::captures`] write.
    pub(crate) saved: Option<&'a StatePool>,
    /// Replay buffers for verify groups.
    pub(crate) spec: Option<&'a SpecPool>,
    pub(crate) shape: &'a [GroupShape],
    pub(crate) pos: Tensor<'a>,
    pub(crate) cells: Tensor<'a>,
    /// With rings: the tokens' ring cells ([`Inputs`]).
    pub(crate) ring_cells: Option<Tensor<'a>>,
    pub(crate) attn: Vec<AttnInputs<'a>>,
    /// Rows to compute logits for, when not all.
    pub(crate) out_rows: Option<Tensor<'a>>,
    /// The per-layer embedding rows (see [`Inputs`]).
    pub(crate) ple: Option<Tensor<'a>>,
    /// The architecture's inputs, per group (see [`Inputs`]).
    pub(crate) group: Vec<Vec<Tensor<'a>>>,
    /// The selective scan's slot ids (see [`Inputs`]).
    pub(crate) seq_ids: Option<Tensor<'a>>,
    /// Tokens in this pass.
    pub(crate) n: i64,
    /// Queries see the keys before their own position only (an MTP head reading the target's
    /// cache, see [`Inputs::exclusive`]).
    pub(crate) exclusive: bool,
    /// Scratch for the MTP head's attention of groups whose last rows alone are read
    /// ([`GroupShape::last_only`], [`paged::attention_last`]); shared by the groups, which run
    /// one after another. `None`: every row computed.
    pub(crate) split: Option<Tensor<'a>>,
}

impl<'a> Pass<'a> {
    /// Whether every group of this pass is a prompt step: more than one token per sequence and
    /// no speculative verify ([`GroupShape::verify`]). Decode and verify groups, also in a
    /// mixed pass, make it false.
    pub(crate) fn prompt_only(&self) -> bool {
        self.shape.iter().all(|g| !g.verify && g.n_tok > 1)
    }

    /// The cells this step's tokens take in layer `il`'s cache: their ring cells if it is a
    /// ring ([`KvRing`]).
    pub(crate) fn write_cells(&self, il: usize) -> Tensor<'a> {
        if self.kv.is_ring(il) {
            self.ring_cells
                .expect("a step over a pool with rings has ring cells")
        } else {
            self.cells
        }
    }

    /// The slot ids `0..n` of a group of `n` sequences (I32 `[n]`).
    pub(crate) fn seq_ids(&self, ctx: &'a Context, n: usize) -> Tensor<'a> {
        let t = self.seq_ids.expect("a Mamba-2 model's pass has slot ids");
        ctx.view_2d(t, [n as i64, 1], t.nb()[1], 0)
    }

    /// (first token, group shape) of every group.
    pub(crate) fn groups(&self) -> impl Iterator<Item = (i64, &GroupShape)> {
        self.shape.iter().scan(0i64, |off, g| {
            let o = *off;
            *off += g.tokens() as i64;
            Some((o, g))
        })
    }
}

/// What a batch step's graph needs besides the pools every step uses.
#[derive(Clone, Copy, Default)]
pub(crate) struct StepOpts<'a> {
    /// Replay buffers, for verify groups.
    pub(crate) spec: Option<&'a SpecPool>,
    /// The saved states, for groups with captures ([`GroupShape::captures`]).
    pub(crate) saved: Option<&'a StatePool>,
    /// Layers whose outputs come back ([`StepOutput::hidden`]).
    hidden: &'a [usize],
    /// An MTP head drafts from the captured rows ([`Arch::mtp_capture_normed`]).
    mtp: bool,
    /// GLM's parallel steps ([`GroupShape::par`]): the GPU (stage) the graph is for.
    pub(crate) par_rank: usize,
}

/// Columns `off..off + n` of the 2-D contiguous `t` (one column per token).
pub(crate) fn cols<'a>(ctx: &'a Context, t: Tensor<'a>, off: i64, n: i64) -> Tensor<'a> {
    let nb1 = t.nb()[1];
    ctx.view_2d(t, [t.ne()[0], n], nb1, off as usize * nb1)
}

/// Keys a sliding-window layer gathers per sequence when it reads cells through its ring:
/// the window before a group's `n_tok` tokens and the tokens, padded as spans are, at most the
/// group's span `n_kv` (see [`AttnInputs::Gather`]).
pub(crate) fn swa_span(n_kv: usize, n_tok: usize, window: usize) -> usize {
    if window == 0 {
        return n_kv;
    }
    let pad = ATTN_KV_PAD as usize;
    ((window + n_tok).div_ceil(pad) * pad).min(n_kv)
}

/// The causal (`swa` false) or sliding-window half of a gather mask built with both (see
/// [`AttnInputs::Gather`]), `[n_kv, n_tok, 1, n_seqs]`; a mask without halves as it is.
pub(crate) fn gather_mask<'a>(
    ctx: &'a Context,
    mask: Tensor<'a>,
    n_seqs: i64,
    swa: bool,
) -> Tensor<'a> {
    let ne = mask.ne();
    if ne[3] == n_seqs {
        assert!(!swa, "a sliding-window layer without its mask");
        return mask;
    }
    let nb = mask.nb();
    let off = if swa { n_seqs as usize * nb[3] } else { 0 };
    ctx.view_4d(
        mask,
        [ne[0], ne[1], ne[2], n_seqs],
        [nb[1], nb[2], nb[3]],
        off,
    )
}

/// Concatenates per-group outputs `[x, n_g]` along the token dim.
pub(crate) fn concat_cols<'a>(ctx: &'a Context, parts: Vec<Tensor<'a>>) -> Tensor<'a> {
    let mut it = parts.into_iter();
    let first = it.next().expect("no groups");
    it.fold(first, |acc, t| ctx.concat(acc, t, 1))
}

/// [`concat_cols`] as a balanced tree: each column is copied about log2(parts) times instead of
/// up to `parts - 1` times.
pub(crate) fn concat_tree<'a>(ctx: &'a Context, mut parts: Vec<Tensor<'a>>) -> Tensor<'a> {
    while parts.len() > 1 {
        parts = parts
            .chunks(2)
            .map(|p| match *p {
                [a, b] => ctx.concat(a, b, 1),
                [a] => a,
                _ => unreachable!("chunks of two"),
            })
            .collect();
    }
    parts.pop().expect("no parts")
}

impl Model {
    pub fn load(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
    ) -> Result<Self, ModelError> {
        let hp = Hparams::from_gguf(gguf)?;
        let weights = Weights::load(gguf, place)?;
        Ok(Self {
            experts: None,
            ssd: None,
            joined: Default::default(),
            router_bf16: Default::default(),
            host_pins: None,
            hp,
            weights,
            mtp: None,
            attention: Attention::Auto,
            moe_kernel: MoeKernel::Auto,
            moe_pair_out: moe_ffn::PairOut::configured(),
            draft_vocab: 0,
            mtp_window: 0,
            linear: Linear::from_gguf(gguf)?,
            widened: None,
            requant: None,
            draft_head: None,
            ngram: None,
            ple_late: Default::default(),
            ple_ahead: Default::default(),
            cell_gather: false,
            kv_stage_blocks: 0,
            slot_stage: (0, 0, 0),
            tp_comm: None,
            tp_pair: None,
            tp_ep: None,
            glm_par: None,
            pair: None,
        })
    }

    /// Like [`Self::load`], with explicit reader options (threads, mmap).
    pub fn load_with(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
        opts: &crate::load_io::Options<'_>,
    ) -> Result<Self, ModelError> {
        Self::load_converted(gguf, place, opts, Default::default())
    }

    /// Like [`Self::load_with`], with weights converted to F16 as `fp16` asks.
    pub fn load_converted(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
        opts: &crate::load_io::Options<'_>,
        fp16: crate::weights::Fp16Load,
    ) -> Result<Self, ModelError> {
        let hp = Hparams::from_gguf(gguf)?;
        let weights = Weights::load_converted(gguf, place, opts, fp16)?;
        Ok(Self {
            experts: None,
            ssd: None,
            joined: Default::default(),
            router_bf16: Default::default(),
            host_pins: None,
            hp,
            weights,
            mtp: None,
            attention: Attention::Auto,
            moe_kernel: MoeKernel::Auto,
            moe_pair_out: moe_ffn::PairOut::configured(),
            draft_vocab: 0,
            mtp_window: 0,
            linear: Linear::from_gguf(gguf)?,
            widened: None,
            requant: None,
            draft_head: None,
            ngram: None,
            ple_late: Default::default(),
            ple_ahead: Default::default(),
            cell_gather: false,
            kv_stage_blocks: 0,
            slot_stage: (0, 0, 0),
            tp_comm: None,
            tp_pair: None,
            tp_ep: None,
            glm_par: None,
            pair: None,
        })
    }

    /// How attention reads the KV cache from now on (default [`Attention::Auto`]).
    /// Makes gather attention read the KV cache by cells (`kv_blocks` inputs then hold one
    /// cell index per key) rather than whole blocks. A block gather reads a block's cells of
    /// every head as one row, which a cache split by heads over a meta device (tensor
    /// parallelism, [`crate::ggml::meta`]) cannot express; a cell is one row of every head.
    pub fn set_cell_gather(&mut self, on: bool) {
        self.cell_gather = on;
    }

    /// Reads the routed experts of the tier's layers through `ssd` from now on (their tensors
    /// must have been left out of the load).
    pub fn set_ssd_experts(&mut self, ssd: crate::residency::ssd_experts::SsdExperts) {
        self.ssd = Some(ssd);
    }

    /// The SSD expert tier, when set up.
    pub fn ssd_experts(&self) -> Option<&crate::residency::ssd_experts::SsdExperts> {
        self.ssd.as_ref()
    }

    /// Sizes prefill groups' paged-attention scratch for KV pools with `n` host-tier staging
    /// slots (#352, [`paged::Tier`]); 0 for none.
    pub fn set_kv_stage_blocks(&mut self, n: usize) {
        self.kv_stage_blocks = n;
    }

    /// Sequence-slot attention (#1153): a step with at least `rows` query rows per sequence
    /// copies each sequence's keys and values into its compute buffers before flash attention
    /// when its span reaches past the slot's first `from` tokens (the ones in VRAM) and is at
    /// most `upto` tokens long (the copies fit in the compute buffers' room), so the host
    /// tier's part crosses the link once per layer instead of once per query tile. The copy
    /// holds the same values in the same layout, so the outputs are the same bits. `rows` 0
    /// turns it off.
    pub fn set_slot_stage(&mut self, rows: usize, from: usize, upto: usize) {
        self.slot_stage = (rows, from, upto);
    }

    /// [`Self::set_slot_stage`]'s `(rows, from, upto)`.
    pub fn slot_stage(&self) -> (usize, usize, usize) {
        self.slot_stage
    }

    /// Sums the attention's and FFN's partial outputs over the ranks of communicator `comm`
    /// ([`crate::tp_net::register`]) from now on: the model was loaded from this rank's share
    /// ([`crate::tp_net::shard::view`]). `None` (the default): no sums.
    pub fn set_tp_comm(&mut self, comm: Option<i32>) {
        self.tp_comm = comm;
    }

    /// Sums the attention's and FFN's partial outputs with the other GPU of a pair through
    /// `sum`, as rank `rank` (0 or 1), from now on: the model was loaded from this rank's share
    /// ([`crate::tp_net::shard::view`]). `None` (the default): no sums.
    pub fn set_tp_pair(&mut self, pair: Option<(std::sync::Arc<pair::sum::PairSum>, usize)>) {
        self.tp_pair = pair;
    }

    /// Runs the routed experts as rank `rank` of `ranks` holding whole experts, its share of
    /// the expert ids (`--tp-moe-ep`, [`tp_ep`]): the model was loaded from such a share
    /// ([`crate::tp_net::shard::qwen4::Experts::Whole`]). Graphs built from now on run only
    /// this rank's (token, expert) pairs. Not with an expert cache. With `place`
    /// (`--tp-moe-ep-place`, #803), the layers it names hold the experts it gives this rank
    /// (the model was loaded from [`crate::tp_net::shard::qwen4::Experts::Placed`]).
    pub fn set_tp_moe_ep(
        &mut self,
        rank: usize,
        ranks: usize,
        place: Option<std::sync::Arc<crate::tp_net::ep_place::Placement>>,
    ) -> Result<(), ModelError> {
        if self.experts.is_some() {
            return Err(ModelError::Unsupported(
                "--tp-moe-ep with an expert cache".into(),
            ));
        }
        let first = (0..self.hp.n_layer)
            .find_map(|il| self.lw(il, "ffn_up_exps.weight").ok())
            .ok_or_else(|| ModelError::Layout("--tp-moe-ep: no routed experts".into()))?;
        let n_expert = self.hp.n_expert as usize;
        self.tp_ep = Some(tp_ep::TpEp::new(
            first,
            n_expert,
            rank,
            ranks,
            place.as_deref(),
        )?);
        tp_ep::ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// The communicator, this rank and the rank count when a step of `n` tokens runs its
    /// per-token work on a share of the tokens per rank (`--tp-seq-par`,
    /// [`crate::tp_net::seq_par`]); `None` without (and on a GPU pair).
    pub(crate) fn tp_seq_par(&self, n: i64) -> Option<(i32, usize, usize)> {
        if self.tp_pair.is_some() {
            return None;
        }
        let c = self.tp_comm?;
        crate::tp_net::seq_par::active(c, n).map(|(rank, ranks)| (c, rank, ranks))
    }

    /// [`Self::tp_sum`], or with `sp` ([`Self::tp_seq_par`]) the sum of this rank's share of
    /// `x`'s columns only.
    pub(crate) fn tp_sum_split<'a>(
        &self,
        ctx: &'a Context,
        x: Tensor<'a>,
        sp: Option<(i32, usize, usize)>,
    ) -> Tensor<'a> {
        match sp {
            Some((c, ..)) => crate::tp_net::seq_par::reduce_scatter(ctx, c, x),
            None => self.tp_sum(ctx, x),
        }
    }

    /// `x` summed over the tensor-parallel ranks ([`Self::set_tp_comm`],
    /// [`Self::set_tp_pair`]); `x` itself without.
    pub(crate) fn tp_sum<'a>(&self, ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
        if let Some((sum, rank)) = &self.tp_pair {
            return pair::sum::all_reduce(ctx, sum, *rank, x);
        }
        match self.tp_comm {
            Some(c) => crate::tp_net::all_reduce(ctx, c, x),
            None => x,
        }
    }

    pub fn set_attention(&mut self, a: Attention) {
        self.attention = a;
    }

    /// MTP drafts from now on over the first `n` tokens of the vocabulary only (0: all; see
    /// `--spec-draft-vocab`).
    pub fn set_draft_vocab(&mut self, n: usize) {
        self.draft_vocab = n;
    }

    /// Keeps `pins` (host memory of the weights registered for the GPUs to read) for as long as
    /// the model: it is released before the weights' memory.
    pub fn keep_host_pins(&mut self, pins: Box<dyn std::any::Any + Send + Sync>) {
        self.host_pins = Some(pins);
    }

    /// The MTP head's attention from now on: over the last `n` positions only (0: all; see
    /// `--mtp-window`). Paged attention (CUDA) only.
    pub fn set_mtp_window(&mut self, n: usize) {
        self.mtp_window = n;
    }

    /// The draft vocabulary set by [`Self::set_draft_vocab`].
    pub(crate) fn draft_vocab(&self) -> usize {
        self.draft_vocab
    }

    /// How the fused expert op keeps per-pair outputs from now on (default:
    /// [`moe_ffn::PairOut::configured`]). Graphs built before keep theirs: an executor forgets them (see
    /// `Executor::set_moe_pair_out`).
    pub fn set_moe_pair_out(&mut self, p: moe_ffn::PairOut) {
        self.moe_pair_out = p;
    }

    /// How the MoE op keeps per-pair outputs (see [`Self::set_moe_pair_out`]).
    pub(crate) fn moe_pair(&self) -> moe_ffn::PairOut {
        self.moe_pair_out
    }

    /// Layer `il`'s tiered expert tensors, when the routed experts run through the expert tier.
    pub(crate) fn expert_tier(
        &self,
        il: usize,
    ) -> Option<crate::residency::experts::TieredLayer<'_>> {
        self.experts.as_ref().and_then(|r| r.layer(il))
    }

    /// How routed experts run from now on (default [`MoeKernel::Auto`]).
    pub fn set_moe_kernel(&mut self, k: MoeKernel) {
        self.moe_kernel = k;
    }

    /// Whether a batch of `n` tokens takes the fused expert op ([`moe_ffn`]) where it supports
    /// the weights: `Some(Fused)` when forced, `Some(Auto)` for big batches on CUDA, `None`
    /// for ggml's `mul_mat_id` path. In auto mode experts in host memory take it only when
    /// they are staged into VRAM first.
    fn fused_moe(&self, n: i64) -> Option<MoeKernel> {
        match self.moe_kernel {
            MoeKernel::Graph => None,
            MoeKernel::Fused => Some(MoeKernel::Fused),
            MoeKernel::Auto => {
                (cfg!(feature = "cuda") && n >= MOE_FFN_MIN_TOKENS).then_some(MoeKernel::Auto)
            }
        }
    }

    /// Whether steps over `kv` use paged attention, whatever their size (the architecture
    /// decides, see [`Arch::paged`]).
    pub fn paged(&self, kv: &KvPool) -> bool {
        self.hp.arch.paged(self, kv)
    }

    /// Whether steps over `kv` use paged attention for an architecture without its own rule:
    /// as [`Self::set_attention`] says, and in [`Attention::Auto`] on CUDA where the kernels
    /// support every attention layer's head shape ([`Hparams::kv_dims`]).
    /// Whether the CUDA paged kernels take every attention layer's head shape.
    pub fn paged_kernels_fit(&self) -> bool {
        let hp = &self.hp;
        (0..hp.n_layer).filter(|&il| hp.is_attention(il)).all(|il| {
            let (d, n_kv) = hp.kv_dims(il);
            paged::supported(d, hp.n_head, n_kv)
        })
    }

    pub(crate) fn paged_default(&self, kv: &KvPool) -> bool {
        let hp = &self.hp;
        match self.attention {
            Attention::Gather => false,
            Attention::Paged => true,
            // The paged kernels read F16, and q8_0 / q4_0 on sm_80+.
            Attention::Auto => {
                cfg!(feature = "cuda")
                    && !kv.is_host()
                    && paged::reads_types(kv.types())
                    && (0..hp.n_layer).filter(|&il| hp.is_attention(il)).all(|il| {
                        let (d, n_kv) = hp.kv_dims(il);
                        paged::supported(d, hp.n_head, n_kv)
                    })
            }
        }
    }

    /// How attention reads the KV cache ([`Self::set_attention`]).
    pub(crate) fn attention_mode(&self) -> Attention {
        self.attention
    }

    /// Whether the step of `groups` over `kv` uses paged attention (an architecture with
    /// sparse attention decides per step, see [`Arch::paged_for`]).
    pub fn paged_for(&self, kv: &KvPool, groups: &[Group<'_>]) -> bool {
        self.hp.arch.paged_for(self, kv, groups)
    }

    /// The loaded weights (for load statistics).
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The MTP head's weights, if loaded from a GGUF of their own.
    pub fn mtp_weights(&self) -> Option<&Weights> {
        self.mtp.as_ref()
    }

    /// The weight tensor `name`, if the model has it.
    pub fn tensor(&self, name: &str) -> Option<Tensor<'_>> {
        self.w(name).ok()
    }

    /// Keeps the activations of every NVFP4 matmul at 8 bits or wider (CUDA otherwise quantizes
    /// those of weights not listed in `prec_a4` to 4 bits on Blackwell). Graphs built after this
    /// call use it.
    pub fn keep_a8(&mut self, on: bool) {
        self.linear.set_all_a8(on);
    }

    /// How FP8 matmuls treat their activations ([`Fp8Act`]); graphs built after this call use it.
    pub fn set_fp8_act(&mut self, a: Fp8Act) {
        self.linear.set_fp8_act(a);
    }

    /// Which kernels run NVFP4 matmuls on CUDA ([`Nvfp4Mm`]); graphs built after this call use
    /// it.
    pub fn set_nvfp4_mm(&mut self, m: Nvfp4Mm) {
        self.linear.set_nvfp4_mm(m);
    }

    /// Which NVFP4 matmuls take 4-bit activations ([`Nvfp4A4`]); graphs built after this call
    /// use it.
    pub fn set_nvfp4_a4(&mut self, a: Nvfp4A4) {
        self.linear.set_nvfp4_a4(a);
        self.probe_nvfp4_a4();
    }

    /// With W4A4 on, runs the FP4 probe (a kernel and a copy on the default stream) now, at
    /// load, on every GPU holding a weight the W4A4 ops could take (as `linear` asks, per
    /// device), rather than at the first W4A4 graph's build, which may come between stage-graph
    /// captures (#1202). Called again by [`Self::load_mtp`] for a head loaded afterwards.
    pub(super) fn probe_nvfp4_a4(&self) {
        if self.linear.nvfp4_a4() == Nvfp4A4::Off {
            return;
        }
        // Each NVFP4 weight's own GPU is probed (cached per device), whatever GPU is current,
        // without changing which kernels any weight routes to.
        let mtp = self.mtp.iter().flat_map(|w| w.tensors());
        for w in self.weights.tensors().chain(mtp) {
            if w.ty() == crate::ggml::types::NVFP4 && !w.is_host() {
                use crate::model::nvfp4_a4::{device_supported, supported_on};
                w.cuda_device().map_or_else(device_supported, supported_on);
            }
        }
    }

    /// Replaces every NVFP4 weight with an exact BF16 copy in buffers of `buft` (each E2M1 code
    /// times its E4M3 block scale fits BF16's mantissa; the per-tensor `.scale` stays separate).
    /// On the CPU the matmuls then round only the activations to BF16: the W4A16 semantics of a
    /// W4A16 checkpoint, as a precision reference. Returns the number of tensors widened.
    pub fn widen_nvfp4(
        &mut self,
        buft: crate::ggml::BufferType,
        infos: &[TensorInfo],
    ) -> Result<usize, ModelError> {
        let picked: Vec<&TensorInfo> = infos
            .iter()
            .filter(|i| i.ggml_type == types::NVFP4 && self.weights.get(&i.name).is_some())
            .collect();
        let ctx = Context::new(picked.len().max(1), 0);
        for info in &picked {
            let shape: Vec<i64> = info.shape.iter().map(|&d| d as i64).collect();
            ctx.new_tensor(types::BF16, &shape).named(&info.name);
        }
        ctx.alloc_tensors(buft)?;
        for info in &picked {
            let src = self.weights.get(&info.name).expect("filtered above");
            let n = src.elements();
            let f = crate::ggml::dequantize(types::NVFP4, &src.bytes(), n);
            let bf16: Vec<u8> = f
                .iter()
                .flat_map(|v| {
                    debug_assert_eq!(v.to_bits() & 0xffff, 0, "not exact in BF16");
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            ctx.tensor(&info.name)
                .expect("created above")
                .set_bytes(&bf16);
        }
        ctx.index_names();
        self.widened = Some(ctx);
        self.joined = Default::default();
        Ok(picked.len())
    }

    /// Copies the MTP head's 16- and 32-bit matrices (block `n_layer`, rows of whole blocks)
    /// as `ty` next to them (same buffer types), read instead of them from now on: the head runs
    /// once per draft, so its bytes are a draft pass's cost (Qwen3.8-27B: 0.85 GB of BF16 a
    /// pass). The head only proposes drafts, so verification keeps the output exact; only
    /// acceptance can change. Returns (matrices, MiB before, MiB after).
    ///
    /// As F16 (`--spec-draft-head-type f16`): the BF16 and F32 matrices only, so GPUs without
    /// BF16 tensor cores (V100) multiply them with FP16 cuBLAS instead of F32 GEMMs (BF16 is
    /// exact in F16; F32 rounds). The hyper-connection inject weights stay as they are (the
    /// fused mixer norm reads F32 ones), and so does a matrix with a value out of F16's range.
    /// Adapted from Strata-AC922's `STRATA_FP16_MTP=1` (`src/core/mtp.cpp`, MIT,
    /// github.com/eelgaev/Strata-AC922).
    pub fn requant_mtp(
        &mut self,
        ty: crate::ggml::ffi::ggml_type,
    ) -> Result<(usize, f64, f64), ModelError> {
        let prefix = format!("blk.{}.", self.hp.n_layer);
        let f16 = ty == types::F16;
        let wide: &[crate::ggml::ffi::ggml_type] = if f16 {
            &[types::BF16, types::F32]
        } else {
            &[types::BF16, types::F16, types::F32]
        };
        let picked: Vec<Tensor<'_>> = self
            .weights
            .tensors()
            .chain(self.mtp.iter().flat_map(|m| m.tensors()))
            .filter(|t| {
                // Whole blocks per row: 32 weights for Q8_0, 256 for the k-quants, 1 for F16.
                let block = match ty {
                    types::Q8_0 => 32,
                    types::F16 => 1,
                    _ => 256,
                };
                t.name().starts_with(&prefix)
                    && wide.contains(&t.ty())
                    && t.ne()[1] > 1
                    && t.ne()[2] == 1
                    && t.ne()[0] % block == 0
                    && !t.is_meta()
                    && !(f16 && crate::weights::fp16::is_hc_inject(&t.name()))
            })
            .filter(|t| {
                !f16 || {
                    let fits = crate::weights::fp16::fits_f16(&mtp_values(*t));
                    if !fits {
                        tracing::warn!(tensor = %t.name(), "MTP head: a value is out of F16's range; kept as stored");
                    }
                    fits
                }
            })
            .collect();
        if picked.is_empty() {
            return Ok((0, 0.0, 0.0));
        }
        // Weights replaced under any graph kept so far: compared in full on their next run.
        crate::ggml::stage_graph::kept_words::words_may_change();
        let ctx = Context::new(picked.len(), 0);
        let mut bufts: Vec<(String, crate::ggml::BufferType)> = Vec::new();
        for t in &picked {
            ctx.new_tensor(ty, &[t.ne()[0], t.ne()[1]]).named(&t.name());
            // SAFETY: an allocated weight's buffer; reads its type.
            let buft = unsafe {
                crate::ggml::BufferType(
                    std::ptr::NonNull::new(crate::ggml::ffi::ggml_backend_buffer_get_type(
                        (*t.raw()).buffer,
                    ))
                    .ok_or_else(|| ModelError::Tensor(format!("{}: no buffer", t.name())))?,
                )
            };
            bufts.push((t.name(), buft));
        }
        // One buffer type for the copies: the head's (all of one stage).
        let buft = bufts[0].1;
        ctx.alloc_tensors(buft)?;
        let (mut before, mut after) = (0usize, 0usize);
        for t in &picked {
            let k = t.ne()[0] as usize;
            let f = mtp_values(*t);
            let threads = std::thread::available_parallelism()
                .map_or(4, |n| n.get())
                .min(16);
            let rows_per = (f.len() / k).div_ceil(threads).max(1);
            let parts: Vec<Vec<u8>> = std::thread::scope(|scope| {
                let hs: Vec<_> = f
                    .chunks(rows_per * k)
                    .map(|c| scope.spawn(move || crate::ggml::quantize(ty, c, k)))
                    .collect();
                hs.into_iter()
                    .map(|h| h.join().expect("quantize thread"))
                    .collect()
            });
            let q = parts.concat();
            before += t.nbytes();
            after += q.len();
            ctx.tensor(&t.name()).expect("created above").set_bytes(&q);
        }
        let n = picked.len();
        // Every weight lookup of a graph build asks it first ([`Self::w`]).
        ctx.index_names();
        self.requant = Some(ctx);
        self.joined = Default::default();
        let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
        Ok((n, mib(before), mib(after)))
    }

    /// Routes the MoE layers through tiered expert tensors (see [`crate::residency`]). Graphs
    /// built before keep the plain tensors: build new ones.
    pub fn set_experts(&mut self, r: Option<crate::residency::experts::ExpertResidency>) {
        self.experts = r;
    }

    /// Sets GLM's parallel decode up (graphs built from now on with [`GroupShape::par`] steps).
    pub fn set_glm_par(&mut self, par: Option<crate::model::glm5next::par::GlmPar>) {
        self.glm_par = par;
    }

    /// Layer `il`'s tiered expert tensors as CUDA device `cuda` serves them (its entry of an
    /// expert-parallel cache, [`crate::residency::experts::ExpertResidency::entry_on`]).
    pub(crate) fn expert_tier_on(
        &self,
        il: usize,
        cuda: usize,
    ) -> Option<crate::residency::experts::TieredLayer<'_>> {
        let r = self.experts.as_ref()?;
        r.layer(r.entry_on(il, cuda)?)
    }

    pub fn experts(&self) -> Option<&crate::residency::experts::ExpertResidency> {
        self.experts.as_ref()
    }

    pub fn experts_mut(&mut self) -> Option<&mut crate::residency::experts::ExpertResidency> {
        self.experts.as_mut()
    }

    pub(crate) fn w(&self, name: &str) -> Result<Tensor<'_>, ModelError> {
        self.requant
            .as_ref()
            .and_then(|c| c.tensor(name))
            .or_else(|| self.widened.as_ref().and_then(|c| c.tensor(name)))
            .or_else(|| self.weights.get(name))
            .or_else(|| self.mtp.as_ref().and_then(|m| m.get(name)))
            .ok_or_else(|| ModelError::Tensor(name.into()))
    }

    /// The addresses of every weight tensor (the model's and its MTP head's), for counting
    /// what graphs read of them ([`crate::ggml::model_bytes`]). Tiered experts' cache and host
    /// tensors and NVFP4's widened copies are not among them.
    pub(crate) fn weight_addrs(&self) -> std::collections::HashSet<usize> {
        self.weights
            .tensors()
            .chain(self.mtp.iter().flat_map(|m| m.tensors()))
            .chain(self.joined().into_iter().flat_map(|j| j.tensors()))
            .chain(
                self.router_bf16
                    .get()
                    .and_then(Option::as_ref)
                    .into_iter()
                    .flat_map(|r| r.tensors()),
            )
            .map(|t| t.raw() as usize)
            .collect()
    }

    /// The BF16 copy of layer `il`'s router `w` ([`router_bf16`]) for small steps' router ops,
    /// the copies made on first use; `None` where they are off, cannot pay (no op on this
    /// device takes them) or `w` has none.
    fn router_bf16<'s>(&'s self, il: usize, w: Tensor<'_>) -> Option<Tensor<'s>> {
        self.router_copies()?
            .get(&format!("blk.{il}{}", router_bf16::ROUTER), w)
    }

    /// Makes the MoE routers' BF16 copies ([`router_bf16`]) now, where small steps' router ops
    /// would read them, rather than in the first graph that asks: the loader calls it before
    /// the caches are sized to the memory left.
    pub fn prepare_router_bf16(&self) {
        self.router_copies();
    }

    /// The routers' BF16 copies ([`Self::router_bf16`]), made on first use.
    fn router_copies(&self) -> Option<&router_bf16::RouterBf16> {
        if !router_bf16::on() || !router_mv::device_supported() {
            return None;
        }
        self.router_bf16
            .get_or_init(|| {
                let names: Vec<String> = self
                    .weights
                    .tensors()
                    .chain(self.mtp.iter().flat_map(|m| m.tensors()))
                    .map(|t| t.name())
                    .filter(|n| n.starts_with("blk.") && n.ends_with(router_bf16::ROUTER))
                    .collect();
                let routers = names
                    .into_iter()
                    .filter_map(|n| self.w(&n).ok().map(|t| (n, t)));
                router_bf16::RouterBf16::build(routers)
            })
            .as_ref()
    }

    /// The weight pairs read as one matrix ([`joined`]), joined on first use.
    fn joined(&self) -> Option<&joined::Joined> {
        self.joined
            .get_or_init(|| {
                let get = |n: &str| {
                    self.weights
                        .get(n)
                        .or_else(|| self.mtp.as_ref().and_then(|m| m.get(n)))
                };
                let all = self
                    .weights
                    .tensors()
                    .chain(self.mtp.iter().flat_map(|m| m.tensors()));
                let j = joined::Joined::build(all, get, false);
                let count = |ty| {
                    let of = |j: &joined::Joined| j.tensors().filter(|t| t.ty() == ty).count();
                    j.as_ref().map_or(0, of)
                };
                let (small, prompt) = (count(types::F32), count(types::BF16));
                tracing::info!(
                    "weight pairs read as one matrix: {small} by small steps, {prompt} by \
                     prompt steps"
                );
                j
            })
            .as_ref()
    }

    /// How many weight pairs small steps read as one matrix ([`joined`]; for tests).
    pub fn joined_pairs(&self) -> usize {
        self.joined().map_or(0, |j| j.tensors().count())
    }

    /// Layer `il`'s `first @ x` and `second @ x` (a pair of [`joined::PAIRS`]) as one matmul,
    /// where the step is small enough and the two are joined ([`joined`]); `None` otherwise
    /// (then run [`Self::lmm`] for each). Plain matmuls only (no scale), with the members'
    /// precision hint (the joined pair's when both have the same).
    pub(crate) fn lmm_joined<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        first: &str,
        second: &str,
        x: Tensor<'a>,
    ) -> Option<(Tensor<'a>, Tensor<'a>)> {
        match self.lmm_joined_run(ctx, il, &[first, second], x)?[..] {
            [a, b] => Some((a, b)),
            _ => None,
        }
    }

    /// [`Self::lmm_joined`] for a pair or a longer run of [`joined::RUNS`] (`names` in storage
    /// order): each member's `w @ x` from one matmul, as row views of its result (rows of the
    /// joined height apart).
    pub(crate) fn lmm_joined_run<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        names: &[&str],
        x: Tensor<'a>,
    ) -> Option<Vec<Tensor<'a>>> {
        let n: i64 = x.ne()[1..].iter().product();
        if !(joined::takes(n) || joined::takes_prompt(n)) || x.ne()[1] != n || names.len() < 2 {
            return None;
        }
        let names: Vec<String> = names.iter().map(|m| format!("blk.{il}.{m}")).collect();
        let plain = |name: &String| {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            linear::scale_name(name).is_none_or(|s| self.w(&s).is_err())
                && self.w(&format!("{base}.input_scale")).is_err()
        };
        let a8 = |name: &String| self.linear.keeps_a8(name);
        if !names.iter().all(plain) || names.iter().any(|m| a8(m) != a8(&names[0])) {
            return None;
        }
        let ws = names
            .iter()
            .map(|m| self.w(m).ok())
            .collect::<Option<Vec<_>>>()?;
        let j = self.joined()?.get_run(&ws)?;
        if !joined::serves(j.ty(), n) {
            return None;
        }
        let r = if j.ty() == types::BF16 {
            // Prompt steps (#836): ggml's cuBLAS GEMM as each member's, on the step's shared
            // BF16 cast of `x` where one was made. GPUs with BF16 tensor cores only.
            if !crate::model::fp8::bf16_mma() {
                return None;
            }
            self.linear.mul_mat(ctx, &names[0], j, None, None, x).ok()?
        } else {
            let r = ctx.mul_mat(j, x);
            // As `Linear::mul_mat` marks each member's matmul (ggml-cuda reads the mark in its
            // quantized kernels only: F32 rows come out the same either way).
            self.linear.hint(ctx, &names[0], j, r);
            r
        };
        let rows: Vec<i64> = ws.iter().map(|w| w.ne()[1]).collect();
        Some(joined::split_run(ctx, r, &rows))
    }

    /// Layer `il`'s projections `names` of `x` (2 to [`dense_mv::MAX_WEIGHTS`] BF16 weights
    /// without scales) in one [`dense_mv`] launch, where the op takes them together
    /// ([`dense_mv::takes_all`]): decode steps of an NVFP4 checkpoint's DeltaNet `qkv` and `z`,
    /// or attention Q, K and V (#836). Q8_0 weights of 1-8 tokens on a V100 take one
    /// [`q8_group`] op instead (where [`q8_group::part`] runs): the input quantized once for all of them, ggml's mmvq bits
    /// (#819); a DeltaNet layer's qkv and z one [`q8_group::mul_mat_each`] node each. Each result has the bits [`Self::lmm`] gives it. `None`: run [`Self::lmm`] (or
    /// [`Self::lmm_q`]) for each.
    pub(crate) fn lmm_dense<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        names: &[&str],
        x: Tensor<'a>,
    ) -> Option<Vec<Tensor<'a>>> {
        let ws = self.dense_weights(il, names)?;
        if dense_mv::takes_all(&ws, x) {
            return Some(dense_mv::mul_mat(ctx, &ws, x));
        }
        // 17-48 tokens (verify steps of 2-5 sequences): wide chains of the same order, e.g. a
        // DeltaNet layer's qkv and z, in one launch (#836); the same bits as one launch each.
        if dense_mv::takes_wide_chains(&ws, x) {
            return Some(dense_mv::mul_mat_wide_chains(ctx, &ws, x));
        }
        let part = match (names[0], ws.len()) {
            ("attn_qkv.weight", 2) => q8_group::PART_DELTA,
            ("attn_qkv.weight", _) => q8_group::PART_DELTA_AB,
            _ => q8_group::PART_ATTN,
        };
        let arch = self.hp.arch.name();
        if !q8_group::part(arch, part) {
            return None;
        }
        // A DeltaNet layer's qkv and z: one node each where the input's Q8_1 blocks are a node
        // already, laid out as ggml's two matmuls (z is read late, by the gated norm); the
        // grouped node only under `--q8-group` (it moves ggml-cuda's layout-dependent fusions).
        if part == q8_group::PART_DELTA {
            if q8_group::takes_each(ctx, &ws, x) {
                return Some(q8_group::mul_mat_each(ctx, &ws, x));
            }
            if q8_group::mode() != q8_group::Mode::All {
                return None;
            }
        }
        q8_group::takes(&ws, x).then(|| q8_group::mul_mat(ctx, &ws, x))
    }

    /// Layer `il`'s weights `names` (2 or more), if every one is a plain weight (no scales) that
    /// [`Self::lmm`] would run as is: [`Self::lmm_dense`]'s candidates.
    fn dense_weights(&self, il: usize, names: &[&str]) -> Option<Vec<Tensor<'_>>> {
        if names.len() < 2 {
            return None;
        }
        let mut ws = Vec::with_capacity(names.len());
        for name in names {
            let name = format!("blk.{il}.{name}");
            let base = name.strip_suffix(".weight").unwrap_or(&name);
            let plain = linear::scale_name(&name).is_none_or(|s| self.w(&s).is_err())
                && self.w(&format!("{base}.input_scale")).is_err();
            if !plain {
                return None;
            }
            ws.push(self.w(&name).ok()?);
        }
        Some(ws)
    }

    /// `mul(rms_norm(x, eps), w)`. In prompt steps of dense models on a GPU with BF16 tensor
    /// cores, the matmuls that read it as BF16 get it from one op with the same bits
    /// ([`fused::rms_norm_bf16`]); the F32 rows are computed only if anything else reads them.
    /// MoE models keep ggml's ops (their graph layout decides ggml-cuda's MoE fusions, #410).
    pub(crate) fn norm_mul<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        w: Tensor<'a>,
    ) -> Tensor<'a> {
        self.norm_mul_with(pass, x, w, false)
    }

    /// [`Self::norm_mul`] for a norm that both BF16 and FP8 matmuls read (`both`, e.g. a gated
    /// DeltaNet layer's input: its beta / alpha projections in BF16, the rest in FP8): with
    /// `--fp8-act a8-dynamic`, the BF16 rows and the FP8 codes come from one op
    /// ([`fp8_act::rms_norm_bf16`]), with the same bits as the two ops.
    pub(crate) fn norm_mul_with<'a>(
        &'a self,
        pass: &Pass<'a>,
        x: Tensor<'a>,
        w: Tensor<'a>,
        both: bool,
    ) -> Tensor<'a> {
        let (ctx, eps) = (pass.ctx, self.hp.rms_eps);
        let h = ctx.mul(ctx.rms_norm(x, eps), w);
        let plain = !pass.tap.on && !self.hp.moe;
        let bf16 = plain && fused::rms_norm_bf16_fits(x, w);
        // `--fp8-act a8-dynamic` prefill: the FP8 matmuls' per-token codes from one op too.
        let fp8 = plain && self.linear.dynamic_batch(x.ne()[1]) && fp8_act::rms_norm_fits(x, w);
        if both && bf16 && fp8 && fp8_act::rms_norm_bf16_fits(x, w) {
            let (q, b) = fp8_act::rms_norm_bf16(ctx, x, w, eps);
            ctx.set_shared_cast(h, types::BF16, b);
            fp8_act::share(ctx, h, q);
            return h;
        }
        if bf16 {
            ctx.set_shared_cast(h, types::BF16, fused::rms_norm_bf16(ctx, x, w, eps));
        }
        if fp8 {
            fp8_act::share(ctx, h, fp8_act::rms_norm(ctx, x, w, eps));
        }
        h
    }

    pub(crate) fn lw(&self, il: usize, name: &str) -> Result<Tensor<'_>, ModelError> {
        self.w(&format!("blk.{il}.{name}"))
    }

    /// `name @ x` as llama.cpp's `build_lora_mm` runs it: with the weight's `.scale` and NVFP4
    /// precision hints (see [`linear`]).
    pub(crate) fn mm<'a>(
        &'a self,
        ctx: &'a Context,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let w = self.w(name)?;
        let scale = linear::scale_name(name).and_then(|s| self.w(&s).ok());
        let input_scale = name
            .strip_suffix(".weight")
            .and_then(|b| self.w(&format!("{b}.input_scale")).ok());
        self.linear.mul_mat(ctx, name, w, scale, input_scale, x)
    }

    /// Whether the LM head `name` (`w`) for `x` runs on [`head_mm`]: [`head_mm::takes`], and
    /// [`Self::mm`] would be a plain matmul of it (no scale, no FP8 activation scale).
    fn head_mm_takes(&self, name: &str, w: Tensor<'_>, x: Tensor<'_>) -> bool {
        head_mm::takes(w, x)
            && linear::scale_name(name).is_none_or(|s| self.w(&s).is_err())
            && name
                .strip_suffix(".weight")
                .is_none_or(|b| self.w(&format!("{b}.input_scale")).is_err())
    }

    /// [`Self::mm`] over the first `rows` rows of `name` only (an MTP head drafting over part
    /// of the vocabulary), or `None` where that cannot be: a scale that is neither per tensor
    /// nor per row ([`linear::first_rows_scale`]).
    pub(crate) fn mm_rows<'a>(
        &'a self,
        ctx: &'a Context,
        name: &str,
        rows: i64,
        x: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let w = self.w(name)?;
        if rows >= w.ne()[1] {
            return Ok(None);
        }
        let scale = match linear::scale_name(name).and_then(|s| self.w(&s).ok()) {
            Some(s) => match linear::first_rows_scale(ctx, s, w.ne()[1], rows) {
                Some(s) => Some(s),
                None => return Ok(None),
            },
            None => None,
        };
        let input_scale = name
            .strip_suffix(".weight")
            .and_then(|b| self.w(&format!("{b}.input_scale")).ok());
        let part = ctx.view_2d(w, [w.ne()[0], rows], w.nb()[1], 0);
        Ok(Some(self.linear.mul_mat(
            ctx,
            name,
            part,
            scale,
            input_scale,
            x,
        )?))
    }

    /// [`Self::mm`] for the weight `name` of layer `il`.
    pub(crate) fn lmm<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.mm(ctx, &format!("blk.{il}.{name}"), x)
    }

    /// [`Self::lmm`] plus `res` (ggml's `add(.., res)`), the add in the matmul's epilogue where
    /// [`linear::Linear::mul_mat_add`] fuses it; just [`Self::lmm`] without `res`.
    pub(crate) fn lmm_add<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
        res: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let Some(res) = res else {
            return self.lmm(ctx, il, name, x);
        };
        let name = format!("blk.{il}.{name}");
        let w = self.w(&name)?;
        let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
        let input_scale = name
            .strip_suffix(".weight")
            .and_then(|b| self.w(&format!("{b}.input_scale")).ok());
        self.linear
            .mul_mat_add(ctx, &name, w, scale, input_scale, x, res)
    }

    /// [`Self::lmm`] of `x · sigmoid(g)` (`k` values for each of `n` tokens) with the gate and
    /// the matmul's quantization as one op ([`gate_q8`], ggml's bytes) where it fits: a Q8_0
    /// weight without a scale on a GB10 prompt step. With `norm`, `x` is normed first
    /// (`mul(rms_norm(x), w)`) in the same op, and it must fit too. `None`: build ggml's ops.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lmm_gated<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
        g: Tensor<'a>,
        norm: Option<gate_q8::Norm<'a>>,
        n: i64,
    ) -> Option<Tensor<'a>> {
        let name = format!("blk.{il}.{name}");
        let w = self.w(&name).ok()?;
        let plain = linear::scale_name(&name).is_none_or(|s| self.w(&s).is_err());
        let norm_ok = norm.is_none_or(|m| gate_q8::norm_fits(x, m.w));
        (plain && norm_ok && gate_q8::fits(w, x, g, n))
            .then(|| gate_q8::mul_mat(ctx, w, x, g, norm, n))
    }

    /// Layer `il`'s Q8_0 weight `name` if [`shared_q8::Shared::mul_mat`] takes it for `n`
    /// tokens: no separate scale, and a weight and step [`mmq_preq::fits`].
    fn shared_weight(&self, il: usize, name: &str, n: i64) -> Option<Tensor<'_>> {
        self.plain_weight(il, name)
            .filter(|&w| mmq_preq::fits(w, n))
    }

    /// Layer `il`'s weight `name` if it has no separate scale ([`Self::lmm`] runs it as is).
    fn plain_weight(&self, il: usize, name: &str) -> Option<Tensor<'_>> {
        let name = format!("blk.{il}.{name}");
        let w = self.w(&name).ok()?;
        let plain = linear::scale_name(&name).is_none_or(|s| self.w(&s).is_err());
        plain.then_some(w)
    }

    /// `x` quantized once for those of layer `il`'s projections `names` that read it on MMQ
    /// ([`shared_q8`]), where at least [`shared_q8::MIN_USERS`] of them do and nothing taps the
    /// step; `None`: each matmul quantizes on its own (ggml's path).
    pub(crate) fn share_q8<'a>(
        &'a self,
        ctx: &'a Context,
        tap: bool,
        il: usize,
        names: &[&str],
        x: Tensor<'a>,
    ) -> Option<shared_q8::Shared<'a>> {
        self.share_bf16(ctx, tap, il, names, x);
        if tap {
            return None;
        }
        // Steps of 9-255 tokens on a GB10: one quantization in a scratch buffer (#836).
        if shared_q8::small::takes_input(x) {
            let users = names
                .iter()
                .filter_map(|name| self.plain_weight(il, name))
                .filter(|&w| shared_q8::small::takes_weight(w, x))
                .count();
            return (users >= shared_q8::MIN_USERS).then(|| shared_q8::small(x));
        }
        if !shared_q8::input_fits(x) {
            return None;
        }
        let (k, n) = (x.ne()[0], x.ne()[1]);
        let users = names
            .iter()
            .filter_map(|name| self.shared_weight(il, name, n))
            .filter(|w| w.ne()[0] == k)
            .count();
        (users >= shared_q8::MIN_USERS).then(|| shared_q8::quantize(ctx, x))
    }

    /// One BF16 cast of `x` for those of layer `il`'s projections `names` that are BF16 weights
    /// read on cuBLAS (the NVFP4 checkpoints' DeltaNet, attention and shared-expert projections,
    /// #469), where at least two of them read it in a prompt chunk of [`SHARE_BF16_MIN_N`]+
    /// tokens and nothing taps the step: [`Context::cast_shared`], which their
    /// [`linear::Linear::mul_mat`] then takes instead of ggml-cuda converting `x` once per matmul.
    /// ggml-cuda rounds `x` with the cast's `__float2bfloat16` and runs the same GEMM on it, so
    /// the results have the same bits. Other matmuls of `x` (a reshape of it too, the NVFP4
    /// experts' input) share the cast.
    fn share_bf16<'a>(
        &'a self,
        ctx: &'a Context,
        tap: bool,
        il: usize,
        names: &[&str],
        x: Tensor<'a>,
    ) {
        if tap || x.ne()[1] < SHARE_BF16_MIN_N {
            return;
        }
        let users = names
            .iter()
            .filter_map(|name| self.w(&format!("blk.{il}.{name}")).ok())
            .filter(|&w| linear::bf16_join_fits(w, x))
            .count();
        if users >= 2 {
            let _ = ctx.cast_shared(x, types::BF16);
        }
    }

    /// [`Self::lmm`] of `x`, from its shared quantization `q` ([`Self::share_q8`]) where it
    /// takes the weight.
    pub(crate) fn lmm_q<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
        q: Option<shared_q8::Shared<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        if let Some(q) = q
            && let Some(w) = self.plain_weight(il, name)
            && q.takes(w)
        {
            return Ok(q.mul_mat(ctx, w));
        }
        self.lmm(ctx, il, name, x)
    }

    /// The expert matmul of layer `il`'s `name`, with per-expert scales (`build_lora_mm_id`).
    pub(crate) fn lmm_id<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        name: &str,
        x: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let name = format!("blk.{il}.{name}");
        let w = self.w(&name)?;
        let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
        self.linear.mul_mat_id(ctx, &name, w, scale, x, ids)
    }

    /// Layer `il`'s routed gate and up matmuls of `x` and their SwiGLU on
    /// [`nvfp4_once::swiglu_id`] (`--nvfp4-once`, #836), when it takes them.
    fn once_glu<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        x: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let weight = |name: &str| -> Result<_, ModelError> {
            let name = format!("blk.{il}.{name}");
            let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
            Ok((self.w(&name)?, scale))
        };
        let (gate, up) = (
            weight("ffn_gate_exps.weight")?,
            weight("ffn_up_exps.weight")?,
        );
        Ok(nvfp4_once::glu_usable(gate, up, x, ids)
            .then(|| nvfp4_once::swiglu_id(ctx, gate, up, x, ids)))
    }

    /// An empty decode state for up to `max_ctx` tokens, in buffers of `buft` (use the
    /// buffer type of the backend that will run [`Self::forward`]).
    pub fn new_state(&self, buft: BufferType, max_ctx: usize) -> Result<SeqState, ModelError> {
        Ok(SeqState::new(&self.hp, buft, max_ctx)?)
    }

    /// Runs `tokens` at positions `state.n_past()..`: appends their K/V to the cache, advances
    /// the recurrent state and returns logits, for the last token or (`all_logits`) for every
    /// one (`[n_tokens][n_vocab]`). Use it for prefill (the whole prompt) and decode (one token).
    ///
    /// `alloc` holds the compute allocator between calls so its buffer is reused; pass the
    /// same one every step (`None` the first time).
    pub fn forward(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        state: &mut SeqState,
        tokens: &[u32],
        all_logits: bool,
    ) -> Result<Vec<f32>, ModelError> {
        Ok(self
            .run(backend, alloc, state, tokens, false, all_logits)?
            .0)
    }

    /// Runs `tokens` as one prompt from an empty state and returns the last token's logits.
    pub fn prompt_logits(
        &self,
        backend: &mut Backend,
        tokens: &[u32],
    ) -> Result<Vec<f32>, ModelError> {
        Ok(self.run_prompt(backend, tokens, false, false)?.0)
    }

    /// Like [`Self::prompt_logits`], but the logits of every position, `[n_tokens][n_vocab]`.
    pub fn prompt_logits_all(
        &self,
        backend: &mut Backend,
        tokens: &[u32],
    ) -> Result<Vec<f32>, ModelError> {
        Ok(self.run_prompt(backend, tokens, false, true)?.0)
    }

    /// Like [`Self::prompt_logits`], plus the main per-layer intermediates in full, named as in
    /// llama.cpp's eval callback, for comparing against it element by element.
    /// Logits are for every position, as in [`Self::prompt_logits_all`].
    pub fn prompt_trace(&self, backend: &mut Backend, tokens: &[u32]) -> Result<Trace, ModelError> {
        self.run_prompt(backend, tokens, true, true)
    }

    fn run_prompt(
        &self,
        backend: &mut Backend,
        tokens: &[u32],
        trace: bool,
        all_logits: bool,
    ) -> Result<Trace, ModelError> {
        let mut state = self.new_state(backend.buffer_type(), tokens.len())?;
        self.run(backend, &mut None, &mut state, tokens, trace, all_logits)
    }

    fn run(
        &self,
        backend: &mut Backend,
        alloc: &mut Option<ComputeAllocator>,
        state: &mut SeqState,
        tokens: &[u32],
        trace: bool,
        all_logits: bool,
    ) -> Result<Trace, ModelError> {
        assert!(!tokens.is_empty(), "empty batch");
        let n_past = state.n_past();
        if n_past + tokens.len() > state.max_ctx() {
            return Err(ModelError::ContextFull {
                needed: n_past + tokens.len(),
                max_ctx: state.max_ctx(),
            });
        }
        let groups = [Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past,
                blocks: &state.blocks,
                ring: 0,
                logits: if all_logits {
                    Logits::All
                } else {
                    Logits::Last
                },
                media: None,
            }],
            verify: false,
        }];
        let table = self
            .paged_for(&state.kv, &groups)
            .then_some(state.blocks.len());
        let shape = batch_shape(&groups, table, Some(&*self.hp.arch));
        let ctx = Context::new(GRAPH_SIZE, GRAPH_SIZE);
        let opts = StepOpts::default();
        let (graph, inputs, taps) = self.build(&ctx, &state.kv, &state.rec, opts, &shape, trace)?;
        let allocator = alloc
            .take()
            .unwrap_or_else(|| ComputeAllocator::new(backend.buffer_type()));
        ctx.alloc_graph(&graph, allocator)?;
        inputs.fill(&groups, &shape);
        self.fill_extra(
            &inputs,
            &groups,
            &shape,
            &state.rec,
            &state.kv,
            None,
            &mut SetNow,
        );
        backend.compute(&graph)?;
        let traced = taps
            .into_iter()
            .map(|(name, t)| TracedTensor {
                name,
                ne: t.ne(),
                data: if t.ty() == types::I32 {
                    t.to_i32().into_iter().map(|v| v as f32).collect()
                } else {
                    t.to_f32()
                },
            })
            .collect();
        let logits = inputs.output(&groups, self.hp.n_vocab as usize).flat();
        *alloc = ctx.into_allocator();
        state.advance(tokens.len());
        Ok((logits, traced))
    }

    /// A step's input tensors for `shape` in `ctx` (allocated with the graph, filled after
    /// it). `extras`: the architecture's per-layer-embedding rows, per-group inputs and slot
    /// ids (an MTP pass has none).
    pub(crate) fn new_inputs<'a>(
        &self,
        ctx: &'a Context,
        shape: &[GroupShape],
        extras: bool,
    ) -> Inputs<'a> {
        self.stage_inputs(ctx, shape, extras, 0..self.hp.n_layer)
    }

    /// [`Self::new_inputs`] for a pipeline stage running `layers`: the per-layer-embedding
    /// rows only if it has that layer, and the incoming residual unless it is the first.
    pub(crate) fn stage_inputs<'a>(
        &self,
        ctx: &'a Context,
        shape: &[GroupShape],
        extras: bool,
        layers: std::ops::Range<usize>,
    ) -> Inputs<'a> {
        let hp = &self.hp;
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let n_out: i64 = shape.iter().map(|g| g.out_rows() as i64).sum();
        // The compute allocator gives inputs memory of their own; filled after it ran.
        let input = |ty, ne: &[i64], name: &str| ctx.new_tensor(ty, ne).named(name).as_input();
        Inputs {
            tokens: input(types::I32, &[n], "tokens"),
            pos: input(types::I32, &[4 * n], "pos"),
            cells: input(types::I32, &[n], "kv_cells"),
            out_rows: (n_out > 0 && n_out < n).then(|| input(types::I32, &[n_out], "out_rows")),
            ring_cells: shape
                .iter()
                .any(|g| g.ring > 0)
                .then(|| input(types::I32, &[n], "kv_ring_cells")),
            attn: shape
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let ns = g.n_seqs as i64;
                    if g.table > 0 {
                        // Sized for the widest heads and the most decode splits (the fewest
                        // kv heads) of any layer; every layer's launch fits in it.
                        let dims: Vec<(i64, i64)> = (0..hp.n_layer)
                            .filter(|&il| hp.is_attention(il))
                            .map(|il| hp.kv_dims(il))
                            .collect();
                        let d = dims.iter().map(|x| x.0).max().unwrap_or(hp.head_dim);
                        let min_kv = dims.iter().map(|x| x.1).min().unwrap_or(hp.n_head_kv);
                        // A verify step's rows split over the key blocks (sm_70 / sm_75).
                        let wmma =
                            paged::wmma_split_len(&dims, hp.n_head, g.n_tok, g.n_seqs, g.table);
                        let scratch = if self.kv_stage_blocks > 0 && g.n_tok > 1 {
                            let n = paged::tier_scratch_len(
                                hp.n_head as usize,
                                g.n_tok,
                                g.n_seqs,
                                g.table,
                                self.kv_stage_blocks,
                            );
                            [n.max(wmma) as i64, 1, 1, 1]
                        } else if wmma > 0 {
                            [wmma as i64, 1, 1, 1]
                        } else {
                            paged::scratch_shape(d, hp.n_head, min_kv, g.n_tok, g.n_seqs)
                        };
                        AttnInputs::Paged {
                            table: input(
                                types::I32,
                                &[g.table as i64, ns],
                                &format!("kv_table-{i}"),
                            ),
                            n_past: input(types::I32, &[ns], &format!("kv_n_past-{i}")),
                            scratch: input(types::F32, &scratch, &format!("attn_scratch-{i}")),
                            ring: (g.ring > 0).then(|| {
                                input(types::I32, &[g.ring as i64, ns], &format!("kv_ring-{i}"))
                            }),
                        }
                    } else {
                        let window = self.swa_window();
                        // With rings the sliding layers' mask is its own tensor.
                        let halves = if window > 0 && g.ring == 0 { 2 } else { 1 };
                        // One entry per block, or per cell (`Self::set_cell_gather`); none
                        // read in place from slots (a placeholder).
                        let blocks = if !g.kv_slots.is_empty() {
                            1
                        } else if self.cell_gather {
                            (g.n_kv * g.n_seqs) as i64
                        } else {
                            (g.n_kv / KV_BLOCK * g.n_seqs) as i64
                        };
                        // The sliding layers read their window only when they gather cells.
                        let n_swa = if self.cell_gather {
                            swa_span(g.n_kv, g.n_tok, window)
                        } else {
                            g.n_kv
                        };
                        let ring_entries = if self.cell_gather {
                            (n_swa * g.n_seqs) as i64
                        } else {
                            blocks
                        };
                        AttnInputs::Gather {
                            mask: input(
                                types::F32,
                                &[g.n_kv as i64, g.n_tok as i64, 1, ns * halves],
                                &format!("kq_mask-{i}"),
                            ),
                            blocks: input(types::I32, &[blocks], &format!("kv_blocks-{i}")),
                            ring: (g.ring > 0).then(|| {
                                input(types::I32, &[ring_entries], &format!("kv_ring_blocks-{i}"))
                            }),
                            swa: (g.ring > 0).then(|| {
                                input(
                                    types::F32,
                                    &[n_swa as i64, g.n_tok as i64, 1, ns],
                                    &format!("kq_mask_swa-{i}"),
                                )
                            }),
                        }
                    }
                })
                .collect(),
            logits: None,
            argmax: None,
            hidden: Vec::new(),
            res_in: (layers.start > 0).then(|| {
                let width = hp.n_embd * hp.arch.residual_streams();
                input(types::F32, &[width, n], "res_in")
            }),
            res_out: None,
            img: {
                let n_img: i64 = shape.iter().map(|g| g.img as i64).sum();
                (layers.start == 0 && n_img > 0).then(|| {
                    (
                        input(types::F32, &[hp.n_embd, n_img], "img_rows"),
                        input(types::I32, &[n_img], "img_idx"),
                    )
                })
            },
            ple: {
                let w = hp.arch.token_rows(hp, &layers);
                (extras && w > 0).then(|| input(types::F32, &[w, n], "ple_rows"))
            },
            group: if extras && hp.arch.group_inputs_len() > 0 {
                shape
                    .iter()
                    .enumerate()
                    .map(|(i, g)| hp.arch.group_inputs(ctx, g, i))
                    .collect()
            } else {
                Vec::new()
            },
            seq_ids: (extras && hp.arch.slot_ids()).then(|| {
                let most = shape.iter().map(|g| g.n_seqs).max().unwrap_or(1);
                input(types::I32, &[most as i64], "seq_ids")
            }),
            swa: self.swa_window(),
            exclusive: false,
        }
    }

    /// The sliding window of the model's sliding-window layers, 0 for none.
    pub fn swa_window(&self) -> usize {
        self.hp.arch.swa_window()
    }

    /// Fills the architecture's inputs of a step (after [`Inputs::fill`]; see
    /// [`Arch::fill_inputs`]) through `sink`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_extra(
        &self,
        inputs: &Inputs<'_>,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        rec: &StatePool,
        kv: &KvPool,
        spec: Option<&SpecPool>,
        sink: &mut dyn InputSink,
    ) {
        self.hp
            .arch
            .fill_inputs(self, inputs, groups, shape, rec, kv, spec, sink);
    }

    /// Builds the graph of one batch step of `shape` in `ctx` (not yet allocated). Also returns
    /// the traced intermediates when `trace` is set.
    #[allow(clippy::type_complexity)]
    fn build<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        opts: StepOpts<'a>,
        shape: &'a [GroupShape],
        trace: bool,
    ) -> Result<(Graph<'a>, Inputs<'a>, Vec<(String, Tensor<'a>)>), ModelError> {
        self.build_stage(ctx, kv, rec, opts, shape, trace, 0..self.hp.n_layer)
    }

    /// [`Self::build`] for one pipeline stage: the layers `layers` only, from the token
    /// embeddings (the first stage) or the incoming residual ([`Inputs::res_in`]), to the
    /// logits (the last stage) or the outgoing residual ([`Inputs::res_out`]). The pools
    /// hold the stage's layers.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn build_stage<'a>(
        &'a self,
        ctx: &'a Context,
        kv: &'a KvPool,
        rec: &'a StatePool,
        opts: StepOpts<'a>,
        shape: &'a [GroupShape],
        trace: bool,
        layers: std::ops::Range<usize>,
    ) -> Result<(Graph<'a>, Inputs<'a>, Vec<(String, Tensor<'a>)>), ModelError> {
        if shape.first().is_some_and(|g| g.par) {
            return self.glm_par_stage(ctx, kv, rec, opts.spec, opts.hidden, shape, opts.par_rank);
        }
        let hp = &self.hp;
        let n: i64 = shape.iter().map(|g| g.tokens() as i64).sum();
        let last = layers.end == hp.n_layer;
        let n_out: i64 = if last {
            shape.iter().map(|g| g.out_rows() as i64).sum()
        } else {
            0
        };
        assert!(n > 0, "empty batch");
        assert!(layers.start < layers.end && layers.end <= hp.n_layer);

        let mut inputs = self.stage_inputs(ctx, shape, true, layers.clone());
        assert!(
            opts.spec.is_some() || shape.iter().all(|g| !g.verify),
            "a verify group needs the replay pool (Executor::enable_spec)"
        );

        let pass = Pass {
            ctx,
            graph: std::cell::RefCell::new(ctx.new_graph(graph_size(shape.len()))),
            tap: Tap {
                ctx,
                on: trace,
                taps: Default::default(),
            },
            kv,
            rec,
            saved: opts.saved,
            spec: opts.spec,
            shape,
            pos: inputs.pos,
            cells: inputs.cells,
            ring_cells: inputs.ring_cells,
            attn: inputs.attn.clone(),
            out_rows: inputs.out_rows,
            ple: inputs.ple,
            group: inputs.group.clone(),
            seq_ids: inputs.seq_ids,
            n,
            exclusive: false,
            split: None,
        };
        let arch = &hp.arch;
        let x = match inputs.res_in {
            Some(r) => r,
            None => {
                let x = match arch.embed_tokens(self, ctx, inputs.tokens)? {
                    Some(x) => x,
                    None => arch.embed(
                        self,
                        ctx,
                        ctx.get_rows(self.w("token_embd.weight")?, inputs.tokens),
                    ),
                };
                // Image tokens take their vision rows instead ([`Media`]).
                match inputs.img {
                    Some((rows, idx)) => ctx.set_rows(x, rows, idx),
                    None => x,
                }
            }
        };
        let (x, mut hidden) = arch.layers(self, &pass, x, opts.hidden, layers.clone())?;
        // Some architectures' MTP heads read the last layer's output after the final norm
        // (llama.cpp's `h_nextn`), so that is what a capture of the last layer returns; logits
        // reuse it.
        let mut final_norm = None;
        let normed = arch.capture_normed() || (opts.mtp && arch.mtp_capture_normed());
        if normed && last && opts.hidden.contains(&(hp.n_layer - 1)) {
            let normed = ctx.mul(ctx.rms_norm(x, hp.rms_eps), self.w("output_norm.weight")?);
            for (il, t) in hidden.iter_mut() {
                if *il + 1 == hp.n_layer {
                    *t = normed;
                }
            }
            final_norm = Some(normed);
        }
        let taps = pass.tap.taps.into_inner();
        let mut graph = pass.graph.into_inner();
        inputs.expand_into(&mut graph);
        for (_, t) in &taps {
            graph.expand(*t);
        }
        if !last {
            // This stage's captured layers (a DFlash drafter's target layers spread over the
            // stages; `Executor::captured`), then the residual (all rows) for the next stage.
            for &il in opts.hidden {
                let (_, t) = hidden
                    .iter()
                    .find(|(l, _)| *l == il)
                    .unwrap_or_else(|| panic!("no layer {il} to capture"));
                let t = t.as_output();
                graph.expand(t);
                inputs.hidden.push(t);
            }
            let out = ctx.cont(x).as_output();
            graph.expand(out);
            inputs.res_out = Some(out);
            return Ok((graph, inputs, taps));
        }
        for &il in opts.hidden {
            let (_, t) = hidden
                .iter()
                .find(|(l, _)| *l == il)
                .unwrap_or_else(|| panic!("no layer {il} to capture"));
            let t = t.as_output();
            graph.expand(t);
            inputs.hidden.push(t);
        }
        if n_out > 0 {
            let x = if arch.output_ready() {
                x
            } else {
                let x = match final_norm {
                    Some(normed) => normed,
                    None => ctx.mul(ctx.rms_norm(x, hp.rms_eps), self.w("output_norm.weight")?),
                };
                match inputs.out_rows {
                    Some(rows) => ctx.get_rows(x, rows),
                    None => x,
                }
            };
            let (logits, argmax) = match arch.head(self, ctx, x)? {
                Some(h) => h,
                None => {
                    // Small models tie the LM head to the embeddings.
                    let head = match self.w("output.weight") {
                        Ok(_) => "output.weight",
                        Err(_) => "token_embd.weight",
                    };
                    let w = self.w(head)?;
                    match arch.logit_softcap() {
                        // A BF16 head of verify steps of many sequences, matmul, soft-cap and
                        // argmax in one pass with ggml's bits (#1181).
                        Some(cap) if self.head_mm_takes(head, w, x) => {
                            head_mm::head(ctx, w, x, cap)
                        }
                        _ => {
                            let logits = self.mm(ctx, head, x)?;
                            let logits = arch.logits(ctx, logits);
                            let argmax = crate::model::cluster_select::argmax_of(ctx, logits, w);
                            (logits, argmax)
                        }
                    }
                }
            };
            let (logits, argmax) = (logits.as_output(), argmax.as_output());
            graph.expand(logits);
            graph.expand(argmax);
            inputs.logits = Some(logits);
            inputs.argmax = Some(argmax);
        } else {
            // Nothing to read back: the step only fills the caches.
            graph.expand(x);
        }
        Ok((graph, inputs, taps))
    }

    pub(crate) fn delta_net<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.delta_net_res(pass, il, h, None)
    }

    /// [`Self::delta_net`] plus the residual `res` (`add(out, res)`), added in the output
    /// projection's epilogue where it can be ([`Self::lmm_add`]). Only for untapped passes (the
    /// tap of the projection's output would see the sum).
    pub(crate) fn delta_net_res<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
        res: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        debug_assert!(
            res.is_none() || !tap.on,
            "a fused residual in a tapped pass"
        );
        let plus_res = |out: Tensor<'a>| match res {
            Some(r) => ctx.add(out, r),
            None => out,
        };
        let hp = &self.hp;
        let head_k = hp.ssm_d_state;
        let n_k = hp.ssm_n_group;
        let n_v = hp.ssm_dt_rank;
        let head_v = hp.ssm_d_inner / n_v;
        let qkv_dim = head_k * n_k * 2 + head_v * n_v;

        // One quantization of h for its MMQ projections on a GB10 (#526).
        let users = [
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
        ];
        let hq = self.share_q8(ctx, tap.on, il, &users, h);
        let (dt_bias, ssm_a) = (self.lw(il, "ssm_dt.bias")?, self.lw(il, "ssm_a")?);
        // On a GPU, one op for the l2 norms, g, beta and a verify step's replay stores
        // (`--delta-prep-ggml`: ggml's chain, as on the CPU).
        let heads = (head_k, n_k, head_v, n_v);
        let prep = !tap.on && fused::delta_prep_fits(heads, dt_bias, ssm_a, n);
        // BF16 decode steps: both projections in one launch (#836), at 9-16 tokens with beta and
        // alpha in it too where ggml would run them on cuBLAS (`dense_mv::fits_chain`). Prompt
        // steps: z, alpha and beta stored back to back as one GEMM (#836, `joined::RUNS`),
        // whose strided rows the gated norm and the op read in place.
        let mut ab = None;
        let zab = (prep && joined::prompt_z() && h.ne()[1] > joined::MAX_TOKENS)
            .then(|| self.lmm_joined_run(ctx, il, joined::RUNS[joined::RUN_Z], h))
            .flatten();
        let (qkv, z) = match (
            self.lmm_dense(ctx, il, &users, h).as_deref(),
            zab.as_deref(),
        ) {
            (Some(&[qkv, z, beta, alpha]), _) => {
                ab = Some((alpha, beta));
                (qkv, z)
            }
            (_, Some(&[z, alpha, beta])) => {
                ab = Some((alpha, beta));
                (self.lmm_q(ctx, il, "attn_qkv.weight", h, hq)?, z)
            }
            _ => match self.lmm_dense(ctx, il, &users[..2], h).as_deref() {
                Some(&[qkv, z]) => (qkv, z),
                _ => (
                    self.lmm_q(ctx, il, "attn_qkv.weight", h, hq)?,
                    self.lmm_q(ctx, il, "attn_gate.weight", h, hq)?,
                ),
            },
        };
        // Small steps: alpha and beta from one matmul (#256), whose strided rows the op reads.
        let (alpha_raw, beta_raw) = match ab.or_else(|| {
            prep.then(|| self.lmm_joined(ctx, il, "ssm_alpha.weight", "ssm_beta.weight", h))
                .flatten()
        }) {
            Some(ab) => ab,
            None => {
                let beta = self.lmm_q(ctx, il, "ssm_beta.weight", h, hq)?;
                (self.lmm_q(ctx, il, "ssm_alpha.weight", h, hq)?, beta)
            }
        };
        let chain = (!prep).then(|| {
            let beta = ctx.sigmoid(beta_raw);
            let alpha = ctx.softplus(ctx.add(alpha_raw, dt_bias));
            (ctx.mul(alpha, ssm_a), beta)
        });

        let mut outs = Vec::new();
        for (off, sh) in pass.groups() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let ng = nt * ns;
            let qkv = ctx.reshape_3d(cols(ctx, qkv, off, ng), qkv_dim, nt, ns);

            // Causal depthwise conv over [history | this chunk], per sequence; the last
            // d_conv - 1 inputs are the next step's history.
            let conv_state = pass.rec.conv_slots(ctx, il, sh.slot, sh.n_seqs);
            let conv_w = self.lw(il, "ssm_conv1d.weight")?;
            // Fused: reads the chunk's rows in place instead of a transposed copy. Every group
            // takes the same op, so a verify row equals the decode step it stands for bit for
            // bit (ggml's silu and the fused op round differently, and ggml's own result
            // differs between optimized and debug builds). A verify group keeps its states
            // (the commit advances them; see spec.rs) and stores the concatenated input for
            // the replay; any other group's history goes to its state now.
            // Prompt groups: the conv normalizes its q and k heads too (`delta_prep`'s bits), so
            // they are read and written once less; the op then computes g and beta only.
            let prep_eps = hp.rms_eps / head_k as f32;
            let conv_norm =
                chain.is_none() && nt > 1 && fused::conv_norm_fits(heads, ng, sh.verify);
            let (conv, history) = if sh.verify {
                // The whole window [history | chunk] for the replay, from the same op.
                let keep = conv_w.ne()[0] - 1 + nt;
                fused::conv_silu_window(ctx, qkv, conv_state, conv_w, keep)
            } else if conv_norm {
                fused::conv_silu_norm(ctx, qkv, conv_state, conv_w, heads, prep_eps)
            } else {
                fused::conv_silu(ctx, qkv, conv_state, conv_w)
            };
            let conv_in = if sh.verify {
                Some(history)
            } else {
                pass.graph.borrow_mut().expand(ctx.cpy(history, conv_state));
                None
            };
            tap.tap("conv_output_silu", il, conv);
            let es = conv.element_size();
            let row = qkv_dim as usize * es;
            let nb = [head_k as usize * es, row, row * nt as usize];
            let q = ctx.view_4d(conv, [head_k, n_k, nt, ns], nb, 0);
            let k = ctx.view_4d(
                conv,
                [head_k, n_k, nt, ns],
                nb,
                (head_k * n_k) as usize * es,
            );
            let v = ctx.view_4d(
                conv,
                [head_v, n_v, nt, ns],
                [head_v as usize * es, row, row * nt as usize],
                (2 * head_k * n_k) as usize * es,
            );
            let (q, k, g, beta, stored) = match chain {
                None if conv_norm => {
                    // q and k: the views of the normalized conv output.
                    let [g, beta] = fused::delta_gates(
                        ctx,
                        conv,
                        cols(ctx, alpha_raw, off, ng),
                        cols(ctx, beta_raw, off, ng),
                        dt_bias,
                        ssm_a,
                        heads,
                        prep_eps,
                    );
                    (q, k, g, beta, false)
                }
                None => {
                    // A pool that defers its commits stores the inputs in the delta rule
                    // (`gdn::rows_replay`), after it reads the last step's kept ones there; a
                    // packed one that does not (no GPU counts) by copies (`store`).
                    let views = sh.verify.then(|| pass.spec.expect("checked in build"));
                    let views = views
                        .filter(|spec| !spec.packed())
                        .map(|spec| spec.token_views(ctx, il, sh.slot, nt, ns));
                    let store = views.map(|[q, k, v_pool, g, beta]| {
                        // v is a view of the conv output: a copy.
                        pass.graph.borrow_mut().expand(ctx.cpy(v, v_pool));
                        [q, k, g, beta]
                    });
                    let [q, k, g, beta] = fused::delta_prep(
                        ctx,
                        conv,
                        cols(ctx, alpha_raw, off, ng),
                        cols(ctx, beta_raw, off, ng),
                        dt_bias,
                        ssm_a,
                        heads,
                        prep_eps,
                        store,
                    );
                    (q, k, g, beta, store.is_some())
                }
                Some((g, beta)) => {
                    let beta = ctx.reshape_4d(cols(ctx, beta, off, ng), [1, n_v, nt, ns]);
                    let g = ctx.reshape_4d(cols(ctx, g, off, ng), [1, n_v, nt, ns]);
                    let q = l2_norm(ctx, q, hp.rms_eps);
                    let k = l2_norm(ctx, k, hp.rms_eps);
                    (q, k, g, beta, false)
                }
            };
            tap.tap("gate", il, g);
            tap.tap("beta_sigmoid", il, beta);
            tap.tap("q_conv_predelta", il, q);
            tap.tap("k_conv_predelta", il, k);

            let ssm_state = pass.rec.ssm_slots(ctx, il, sh.slot, sh.n_seqs);
            if sh.verify {
                // A verify step reads the rows only: its states advance in the commit.
                let spec = pass.spec.expect("checked in build");
                let replay = spec.replay_views(ctx, il, sh.slot, sh.n_seqs);
                let rows = if let Some((keep, replay, arrive)) = replay {
                    // The last commit deferred into this step (#836 part 14): a pool that
                    // defers has no other way to run it. The op then stores this step's inputs
                    // there for the next one (part 39).
                    assert!(
                        gdn::replay_fits(q, k, v, g, beta, ssm_state, keep, replay, arrive),
                        "a verify step that cannot replay its deferred commits"
                    );
                    // Its padding rows are skipped where the step has a row map (#836).
                    let pad = gdn::pad_skip()
                        .then(|| hp.arch.verify_row_map(pass))
                        .flatten()
                        .filter(|&m| gdn::pad_fits(m, v));
                    gdn::rows_replay(ctx, q, k, v, g, beta, ssm_state, keep, replay, arrive, pad)
                } else if gdn::fits(q, k, v, g, beta, ssm_state) {
                    gdn::rows(ctx, q, k, v, g, beta, ssm_state)
                } else {
                    let out = ctx.gated_delta_net(q, k, v, g, beta, ssm_state, 1);
                    ctx.view_2d(out, [head_v * n_v, ng], out.nb()[1], 0)
                };
                let conv_in = conv_in.expect("a verify group's conv input");
                let mut graph = pass.graph.borrow_mut();
                if stored || spec.defers() {
                    // The op wrote q, k, g and beta already, and v is copied; or the delta rule
                    // stored them over the packed rows (`rows_replay`).
                    spec.store_window(ctx, &mut graph, il, sh.slot, conv_in);
                } else {
                    spec.store(ctx, &mut graph, il, sh.slot, conv_in, [q, k, v, g, beta]);
                }
                outs.push(rows);
                continue;
            }
            // Output rows: the tokens' [head_v * n_v], then the final states.
            let delta_rule = |[q, k, v, g, beta]: [Tensor<'a>; 5], state: Tensor<'a>, nt: i64| {
                if gdn_scan::fits(q, v, g, state, nt, true) {
                    gdn_scan::gdn_scan(ctx, q, k, v, g, beta, state)
                } else if gdn_mma::fits(q, k, v, g, beta, state) {
                    // `--gdn-mma` (#781): ggml-cuda's chunked kernel for head counts it keeps
                    // on its per-token one (a tensor parallel rank's share).
                    gdn_mma::gdn_mma(ctx, q, k, v, g, beta, state)
                } else if gdn::decode_fits(q, k, v, g, beta, state) {
                    // One token per sequence (#836): ggml's output and bits, the states moved
                    // in 16-byte vectors.
                    gdn::decode(ctx, q, k, v, g, beta, state)
                } else {
                    ctx.gated_delta_net(q, k, v, g, beta, state, 1)
                }
            };
            let state_of = |out: Tensor<'a>, ng: i64| {
                let es = out.element_size();
                ctx.view_4d(
                    out,
                    ssm_state.ne(),
                    [
                        head_v as usize * es,
                        (head_v * head_v) as usize * es,
                        (head_v * head_v * n_v) as usize * es,
                    ],
                    (head_v * n_v * ng) as usize * es,
                )
            };
            if !sh.captures.is_empty() {
                debug_assert_eq!(ns, 1, "captures in a group of one sequence");
                // States saved inside the step ([`capture`]): the conv histories from the
                // inputs, the delta rule in pieces split where the states are saved.
                let saved = pass
                    .saved
                    .expect("a step with captures has the saved states");
                let d_conv = conv_w.ne()[0];
                let mut state = ssm_state;
                let mut rows = Vec::new();
                for c in &sh.captures {
                    let to = saved.conv_slots(ctx, il, c.saved, 1);
                    capture::history(ctx, &pass.graph, qkv, c.at, d_conv - 1, to);
                }
                for (start, len) in capture::pieces(&sh.captures, nt) {
                    let part = [q, k, v, g, beta].map(|t| capture::tokens(ctx, t, start, len));
                    let out = delta_rule(part, state, len);
                    state = state_of(out, len);
                    rows.push(ctx.view_2d(out, [head_v * n_v, len], out.nb()[1], 0));
                    for c in sh.captures.iter().filter(|c| c.at == start + len) {
                        let to = saved.ssm_slots(ctx, il, c.saved, 1);
                        pass.graph.borrow_mut().expand(ctx.cpy(state, to));
                    }
                }
                pass.graph.borrow_mut().expand(ctx.cpy(state, ssm_state));
                outs.push(concat_tree(ctx, rows));
                continue;
            }
            let out = delta_rule([q, k, v, g, beta], ssm_state, nt);
            let final_state = state_of(out, ng);
            // A decode step's states are updated in place; others are copied back.
            let in_place =
                sh.slot == 0 && sh.n_tok == 1 && pass.rec.decode_output(il, out, ng as usize);
            if !in_place {
                pass.graph
                    .borrow_mut()
                    .expand(ctx.cpy(final_state, ssm_state));
            }
            outs.push(ctx.view_2d(out, [head_v * n_v, ng], out.nb()[1], 0));
        }
        let o = concat_cols(ctx, outs);
        let o = ctx.reshape_4d(o, [head_v, n_v, n, 1]);
        tap.tap("attn_output", il, o);
        tap.tap("z", il, z);
        // Gated RMS norm: norm(o) * silu(z), or sigmoid(z) (see [`Arch::sigmoid_gate`]); on a
        // GPU one op (`--gated-norm`).
        let norm_w = self.lw(il, "ssm_norm.weight")?;
        let o2 = ctx.reshape_2d(o, head_v * n_v, n);
        // `--fp8-act a8-dynamic` prefill: the projection's per-token FP8 codes from one op
        // ([`fp8_act::gated_norm`], the norm's bits), for either output below.
        let fp8_codes = |o: Tensor<'a>| {
            if !tap.on
                && !hp.moe
                && self.linear.dynamic_batch(n)
                && fp8_act::gated_norm_fits(o2, norm_w, z, head_v)
            {
                let silu = !hp.arch.sigmoid_gate();
                let q = fp8_act::gated_norm(ctx, o2, norm_w, z, head_v, hp.rms_eps, silu);
                fp8_act::share(ctx, o, q);
            }
        };
        // `--decode-fuse` (#773): the same op in qwen4exp's decode-sized steps.
        let df = decode_fuse::takes(hp, n) && fused::gated_norm_shapes(o2, norm_w, z, head_v);
        if !tap.on && (df || fused::gated_norm_fits(o2, norm_w, z, head_v)) {
            let silu = !hp.arch.sigmoid_gate();
            let o = fused::gated_norm(ctx, o2, norm_w, z, head_v, hp.rms_eps, silu);
            fp8_codes(o);
            return self.lmm_add(ctx, il, "ssm_out.weight", o, res);
        }
        let z_rows = z;
        let z = if z.is_contiguous() {
            ctx.reshape_4d(z, [head_v, n_v, n, 1])
        } else {
            // Rows of a joined GEMM's result (prompt steps, `joined::RUNS`): the same values in
            // place, read strided.
            let (es, ld) = (z.element_size(), z.nb()[1]);
            ctx.view_4d(
                z,
                [head_v, n_v, n, 1],
                [head_v as usize * es, ld, ld * n as usize],
                0,
            )
        };
        // The norm, the sigmoid gate and the projection's quantization as one op where it fits;
        // else the gate and the quantization (#526).
        let norm = gate_q8::Norm {
            w: norm_w,
            eps: hp.rms_eps,
        };
        if !tap.on
            && hp.arch.sigmoid_gate()
            && let Some(out) = self.lmm_gated(ctx, il, "ssm_out.weight", o, z, Some(norm), n)
        {
            return Ok(plus_res(out));
        }
        let normed = ctx.mul(ctx.rms_norm(o, hp.rms_eps), norm_w);
        if !tap.on
            && hp.arch.sigmoid_gate()
            && let Some(out) = self.lmm_gated(ctx, il, "ssm_out.weight", normed, z, None, n)
        {
            return Ok(plus_res(out));
        }
        let z = if hp.arch.sigmoid_gate() {
            ctx.sigmoid(z)
        } else {
            ctx.silu(z)
        };
        let o = ctx.mul(normed, z);
        let o = ctx.reshape_2d(o, head_v * n_v, n);
        tap.tap("final_output", il, o);
        // Prompt steps of dense models: a projection that reads the output as BF16 gets it from
        // one op with ggml's chain's bits (`o` itself is computed only if anything reads it in
        // F32). MoE models keep the chain, whose graph layout decides ggml-cuda's MoE fusions
        // (#410), except in prompt chunks with a BF16 projection (the NVFP4 checkpoints, #469).
        let bf16_out = || {
            n >= SHARE_BF16_MIN_N
                && self
                    .lw(il, "ssm_out.weight")
                    .is_ok_and(|w| linear::bf16_join_fits(w, o))
        };
        if !tap.on
            && (!hp.moe || bf16_out())
            && fused::gated_norm_bf16_fits(o2, norm_w, z_rows, head_v, n)
        {
            let silu = !hp.arch.sigmoid_gate();
            let ob = fused::gated_norm_bf16(ctx, o2, norm_w, z_rows, head_v, hp.rms_eps, silu);
            ctx.set_shared_cast(o, types::BF16, ob);
        }
        fp8_codes(o);
        let out = self.lmm_add(ctx, il, "ssm_out.weight", o, res)?;
        tap.tap("linear_attn_out", il, out);
        Ok(out)
    }

    pub(crate) fn attention<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.attention_res(pass, il, h, None)
    }

    /// [`Self::attention`] plus the residual `res`, as [`Self::delta_net_res`].
    pub(crate) fn attention_res<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
        res: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let (ctx, tap, n) = (pass.ctx, &pass.tap, pass.n);
        debug_assert!(
            res.is_none() || !tap.on,
            "a fused residual in a tapped pass"
        );
        let hp = &self.hp;
        let d = hp.head_dim;
        // attn_q produces query and output gate interleaved per head: [q_h | gate_h] × n_head.
        // One quantization of h for its MMQ projections on a GB10 (#526).
        let users = ["attn_q.weight", "attn_k.weight", "attn_v.weight"];
        let hq = self.share_q8(ctx, tap.on, il, &users, h);
        // BF16 decode steps: Q, K and V in one launch (#836).
        let joint = self.lmm_dense(ctx, il, &users, h);
        // Prompt steps: Q + gate, K and V stored back to back as one GEMM (#836,
        // `joined::RUNS`), whose rows (the joined height apart) are read strided in place.
        let joint = joint.or_else(|| {
            (!tap.on && joined::prompt_qkv() && h.ne()[1] > joined::MAX_TOKENS)
                .then(|| self.lmm_joined_run(ctx, il, joined::RUNS[joined::RUN_QKV], h))
                .flatten()
        });
        let (qg, kv) = match joint.as_deref() {
            Some(&[qg, k, v]) => (qg, Some((k, v))),
            _ => (self.lmm_q(ctx, il, "attn_q.weight", h, hq)?, None),
        };
        let es = qg.element_size();
        // A token's row: `2 d n_head` values, or the joined height.
        let nb = [(2 * d) as usize * es, qg.nb()[1]];
        let q = ctx.view_3d(qg, [d, hp.n_head, n], nb, 0);
        let gate = ctx.view_3d(qg, [d, hp.n_head, n], nb, d as usize * es);

        let q = self.norm_rope(ctx, q, self.lw(il, "attn_q_norm.weight")?, pass.pos);
        tap.tap("Qcur", il, q);

        // K and V go through the F16 cache (llama.cpp's default cache type too); this step's
        // rows are written first (the graph runs nodes in order), then attention reads them.
        // Gather path: each sequence's span is gathered back and attention is the plain
        // mul_mat / soft_max path, which matches llama.cpp with flash attention off bit for
        // bit. llama.cpp's CPU flash attention (its default) rounds Q to F16 and accumulates
        // V·softmax in F16, which moves the attention output by ~3e-4 relative; that is a
        // llama.cpp kernel choice, not something to copy. Paged path (CUDA): the kernels read
        // the blocks in place and accumulate in F32 (see `paged`).
        if let Some((k, v)) = self.append_kv(pass, il, h, hq, kv)? {
            tap.tap("Kcur", il, k);
            tap.tap("Vcur", il, v);
        }
        // Sparse attention: the indexer query of every token, and this step's raw indexer keys
        // cached.
        let iq = hp.arch.index_query(self, pass, il, h)?;
        let o = self.attend(pass, il, q, iq)?;
        tap.tap("attn_pregate", il, o);

        // The gate and the projection's quantization as one op where it fits, reading the
        // gate in place (#526).
        if !tap.on
            && let Some(out) = self.lmm_gated(ctx, il, "attn_output.weight", o, gate, None, n)
        {
            return Ok(match res {
                Some(r) => ctx.add(out, r),
                None => out,
            });
        }
        let gate_view = gate;
        let gate = ctx.cont_2d(gate, d * hp.n_head, n);
        let o_pre = o;
        let o = ctx.mul(o, ctx.sigmoid(gate));
        tap.tap("attn_gated", il, o);
        // The projection's BF16 input in one kernel, reading the gate in place (sm_80+ prompt
        // chunks of dense models, and of MoE models with a BF16 projection, #469; ggml's bits).
        let bf16_out = || {
            n >= SHARE_BF16_MIN_N
                && self
                    .lw(il, "attn_output.weight")
                    .is_ok_and(|w| linear::bf16_join_fits(w, o))
        };
        if !tap.on && (!hp.moe || bf16_out()) && fused::gate_bf16_fits(gate_view, o_pre) {
            let ob = fused::gate_bf16(ctx, gate_view, o_pre, false);
            ctx.set_shared_cast(o, types::BF16, ob);
        }
        // `--fp8-act a8-dynamic` prefill: its per-token FP8 codes from one op too.
        if !tap.on
            && !hp.moe
            && self.linear.dynamic_batch(n)
            && fp8_act::gate_fits(gate_view, o_pre)
        {
            fp8_act::share(ctx, o, fp8_act::gate(ctx, gate_view, o_pre, false));
        }
        let out = self.lmm_add(ctx, il, "attn_output.weight", o, res)?;
        tap.tap("attn_output", il, out);
        Ok(out)
    }

    /// Writes the step's K and V rows of layer `il` (from its attention input `h`, K normed
    /// and rotated, `[head_dim, n_head_kv, n]` each) to the cache. In one launch where the
    /// caches take it and nothing taps the rows ([`norm_rope::kv_append`], ggml's bits): then
    /// `None`, else the rows. `hq`: `h` quantized for its projections ([`Self::share_q8`]);
    /// `kv`: the K and V projections when the caller has them already ([`Self::lmm_dense`]).
    fn append_kv<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
        hq: Option<shared_q8::Shared<'a>>,
        kv: Option<(Tensor<'a>, Tensor<'a>)>,
    ) -> Result<Option<(Tensor<'a>, Tensor<'a>)>, ModelError> {
        let (ctx, n) = (pass.ctx, pass.n);
        let hp = &self.hp;
        let d = hp.head_dim;
        let (k, v) = match kv {
            Some(kv) => kv,
            None => (
                self.lmm_q(ctx, il, "attn_k.weight", h, hq)?,
                self.lmm_q(ctx, il, "attn_v.weight", h, hq)?,
            ),
        };
        // Rows of a joined GEMM's result (prompt steps, `joined::RUNS`) are viewed in place,
        // the joined height apart.
        let heads = |x: Tensor<'a>| {
            if x.is_contiguous() {
                ctx.reshape_3d(x, d, hp.n_head_kv, n)
            } else {
                let es = x.element_size();
                ctx.view_3d(x, [d, hp.n_head_kv, n], [d as usize * es, x.nb()[1]], 0)
            }
        };
        let (k, v) = (heads(k), heads(v));
        let w = self.lw(il, "attn_k_norm.weight")?;
        let (pos, cells) = (pass.pos, pass.cells);
        let rope = self.rope_params();
        if let Some((kc, vc)) = pass.kv.direct_caches(il).filter(|&(kc, vc)| {
            !pass.tap.on && norm_rope::kv_append_fits(k, w, pos, v, kc, vc, cells, &rope)
        }) {
            let eps = hp.rms_eps;
            let node = norm_rope::kv_append(ctx, k, w, pos, v, kc, vc, cells, eps, &rope);
            pass.graph.borrow_mut().expand(node);
            return Ok(None);
        }
        // The cache writes and ggml's norm chain read contiguous rows.
        let cont = |x: Tensor<'a>| if x.is_contiguous() { x } else { ctx.cont(x) };
        let k = if norm_rope::norm_rope_fits(k, w, pos, &rope) {
            k
        } else {
            cont(k)
        };
        let v = cont(v);
        let k = self.norm_rope(ctx, k, w, pos);
        pass.kv
            .write(ctx, &mut pass.graph.borrow_mut(), il, k, v, cells);
        Ok(Some((k, v)))
    }

    /// The attention layers' RoPE of `mul(rms_norm(x), w)` (`x` `[d, heads, n]`, rows
    /// contiguous), at positions `pos`: one launch ([`norm_rope::norm_rope`]) where it fits,
    /// else ggml's chain; the same bits either way.
    pub(crate) fn norm_rope<'a>(
        &self,
        ctx: &'a Context,
        x: Tensor<'a>,
        w: Tensor<'a>,
        pos: Tensor<'a>,
    ) -> Tensor<'a> {
        let rope = self.rope_params();
        if norm_rope::norm_rope_fits(x, w, pos, &rope) {
            return norm_rope::norm_rope(ctx, x, w, pos, self.hp.rms_eps, &rope);
        }
        self.rope(ctx, ctx.mul(ctx.rms_norm(x, self.hp.rms_eps), w), pos)
    }

    /// [`Self::rope`]'s parameters, for the fused kernels.
    pub(crate) fn rope_params(&self) -> norm_rope::Rope {
        let hp = &self.hp;
        norm_rope::Rope {
            n_dims: hp.n_rot,
            sections: hp.rope_sections,
            mode: ROPE_IMROPE,
            freq_base: hp.rope_freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
        }
    }

    /// Multi-section RoPE of `t` (`[head_dim, heads, n]`) at positions `pos`, as the attention
    /// layers apply it.
    pub(crate) fn rope<'a>(&self, ctx: &'a Context, t: Tensor<'a>, pos: Tensor<'a>) -> Tensor<'a> {
        let hp = &self.hp;
        ctx.rope_multi(
            t,
            pos,
            hp.n_rot,
            hp.rope_sections,
            ROPE_IMROPE,
            hp.n_ctx_train,
            hp.rope_freq_base,
            1.0,
            0.0,
            1.0,
            32.0,
            1.0,
        )
    }

    /// Writes the step's K/V rows of layer `il` (from its attention input `h`) to the cache,
    /// and nothing else: [`Self::attention`] for steps whose rows need no output.
    pub(crate) fn write_kv<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<(), ModelError> {
        let users = ["attn_k.weight", "attn_v.weight"];
        let hq = self.share_q8(pass.ctx, pass.tap.on, il, &users, h);
        self.append_kv(pass, il, h, hq, None)?;
        Ok(())
    }

    /// Attention of every group's queries `q` (`[head_dim, n_head, n]`) over its sequences'
    /// cached K/V of layer `il` (this step's rows already written), causal; `[head_dim ×
    /// n_head, n]`. `iq`: the indexer queries of a sparse-attention layer
    /// ([`Arch::index_query`]).
    pub(crate) fn attend<'a>(
        &'a self,
        pass: &Pass<'a>,
        il: usize,
        q: Tensor<'a>,
        iq: Option<Tensor<'a>>,
    ) -> Result<Tensor<'a>, ModelError> {
        let ctx = pass.ctx;
        let hp = &self.hp;
        let d = hp.head_dim;
        let mut outs = Vec::new();
        for (gi, (off, sh)) in pass.groups().enumerate() {
            let (nt, ns) = (sh.n_tok as i64, sh.n_seqs as i64);
            let qnb = q.nb();
            let q = ctx.view_3d(
                q,
                [d, hp.n_head, nt * ns],
                [qnb[1], qnb[2]],
                off as usize * qnb[2],
            );
            let q = ctx.reshape_4d(q, [d, hp.n_head, nt, ns]);
            let (mask, blocks) = match pass.attn[gi] {
                AttnInputs::Gather { mask, blocks, .. } => (mask, blocks),
                AttnInputs::Paged {
                    table,
                    n_past,
                    scratch,
                    ..
                } => {
                    let (k, v) = pass.kv.pools(il);
                    let tier = pass.kv.paged_tier(il);
                    // TurboQuant caches (#552): their own op (the MTP head's take-in rows all
                    // computed; sparse layers do not read them, which startup checks).
                    if let Some(tq) = pass.kv.tq(il).filter(|_| iq.is_none()) {
                        let window = if il == hp.n_layer { self.mtp_window } else { 0 };
                        let o = paged::tq::attention(
                            ctx,
                            q,
                            k,
                            v,
                            table,
                            n_past,
                            tq,
                            1.0 / (d as f32).sqrt(),
                            window,
                            0,
                        );
                        outs.push(ctx.reshape_2d(o, d * hp.n_head, nt * ns));
                        continue;
                    }
                    let o = match iq {
                        Some(iq) => {
                            hp.arch
                                .sparse_paged(self, pass, il, gi, iq, [q, k, v], table)?
                        }
                        None => match pass
                            .split
                            .filter(|_| il == hp.n_layer && sh.n_tok > 1 && sh.last_only)
                        {
                            // The MTP head's take-in rows only write their K/V: the last
                            // rows' attention alone.
                            Some(split) => paged::attention_last(
                                ctx,
                                q,
                                k,
                                v,
                                table,
                                n_past,
                                scratch,
                                1.0 / (d as f32).sqrt(),
                                self.mtp_window,
                                tier,
                                split,
                            ),
                            // The MTP head (layer n_layer) with a window: the last positions
                            // only.
                            None => paged::attention_tiered(
                                ctx,
                                q,
                                k,
                                v,
                                table,
                                n_past,
                                scratch,
                                1.0 / (d as f32).sqrt(),
                                if il == hp.n_layer { self.mtp_window } else { 0 },
                                0,
                                tier,
                            ),
                        },
                    };
                    outs.push(ctx.reshape_2d(o, d * hp.n_head, nt * ns));
                    continue;
                }
            };
            let (k, vt) = pass.kv.read(ctx, il, blocks, sh.n_kv as i64, ns);
            // [d, tokens, heads, seqs]; GQA via mul_mat broadcasting over dim 2.
            let q = ctx.permute(q, [0, 2, 1, 3]);
            let k = ctx.permute(k, [0, 2, 1, 3]);
            let kq = ctx.mul_mat(k, q);
            let mask = match iq {
                Some(iq) => hp.arch.sparse_mask(self, pass, il, gi, iq, mask)?,
                None => mask,
            };
            let kq = ctx.soft_max_ext(kq, mask, 1.0 / (d as f32).sqrt(), 0.0);
            let kqv = ctx.mul_mat(vt, kq); // [d, tokens, n_head, seqs]
            outs.push(ctx.cont_2d(ctx.permute(kqv, [0, 2, 1, 3]), d * hp.n_head, nt * ns));
        }
        Ok(concat_cols(ctx, outs))
    }

    /// Dense SwiGLU FFN.
    pub(crate) fn ffn<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        h: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        self.ffn_fused(ctx, il, h, linear::FfnFusion::default())
    }

    /// [`Self::ffn`] with `fusion`'s residual added to its output (in the down projection's
    /// epilogue where [`linear::Linear::mul_mat_add`] fuses it) and its norm in the W4A4
    /// SwiGLU's quantization ([`linear::Linear::swiglu_ffn_fused`]); `h` is that norm's output.
    pub(crate) fn ffn_fused<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        h: Tensor<'a>,
        fusion: linear::FfnFusion<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let weight = |name: &str| -> Result<_, ModelError> {
            let name = format!("blk.{il}.{name}");
            let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
            Ok((self.w(&name)?, scale, name))
        };
        let (wg, sg, gn) = weight("ffn_gate.weight")?;
        let (wu, su, un) = weight("ffn_up.weight")?;
        let (wd, sd, dn) = weight("ffn_down.weight")?;
        if let Some(out) = self.linear.swiglu_ffn_fused(
            ctx,
            (&gn, wg, sg),
            (&un, wu, su),
            (&dn, wd, sd),
            h,
            fusion,
        ) {
            return Ok(out);
        }
        // gate, up, then one SwiGLU (llama.cpp's build_ffn): ggml-cuda fuses the three into
        // one matvec kernel in decode steps.
        let gate = self.lmm(ctx, il, "ffn_gate.weight", h)?;
        let up = self.lmm(ctx, il, "ffn_up.weight", h)?;
        let act = ctx.swiglu_split(gate, up);
        // The down projection's BF16 input in one kernel (sm_80+ prompt chunks of dense
        // models; ggml's bits).
        if !self.hp.moe && fused::gate_bf16_fits(gate, up) {
            ctx.set_shared_cast(act, types::BF16, fused::gate_bf16(ctx, gate, up, true));
        }
        // `--fp8-act a8-dynamic` prefill: its per-token FP8 codes from one op too.
        if !self.hp.moe && self.linear.dynamic_batch(gate.ne()[1]) && fp8_act::gate_fits(gate, up) {
            fp8_act::share(ctx, act, fp8_act::gate(ctx, gate, up, true));
        }
        self.lmm_add(ctx, il, "ffn_down.weight", act, fusion.res)
    }

    /// Layer `il`'s GeGLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`) over `h` as
    /// [`linear::Linear::geglu_ffn_fused`] runs it, or `None` where it does not (the caller
    /// builds the unfused chain).
    pub(crate) fn geglu_ffn_fused<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        h: Tensor<'a>,
        norm: Option<(Tensor<'a>, Tensor<'a>, f32)>,
    ) -> Result<Option<Tensor<'a>>, ModelError> {
        let weight = |name: &str| -> Result<_, ModelError> {
            let name = format!("blk.{il}.{name}");
            let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
            Ok((self.w(&name)?, scale, name))
        };
        let (wg, sg, gn) = weight("ffn_gate.weight")?;
        let (wu, su, un) = weight("ffn_up.weight")?;
        let (wd, sd, dn) = weight("ffn_down.weight")?;
        Ok(self
            .linear
            .geglu_ffn_fused(ctx, (&gn, wg, sg), (&un, wu, su), (&dn, wd, sd), h, norm))
    }

    /// Layer `il`'s weights `names` over one input `h`, as
    /// [`linear::Linear::mul_mat_multi`] runs them (one W4A4 op; `norm` as there), or `None`
    /// where it does not (the caller runs [`Self::lmm`] for each).
    pub(crate) fn lmm_multi<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        names: &[&str],
        h: Tensor<'a>,
        norm: Option<(Tensor<'a>, Tensor<'a>, f32)>,
    ) -> Result<Option<Vec<Tensor<'a>>>, ModelError> {
        let mut ws = Vec::new();
        for name in names {
            let name = format!("blk.{il}.{name}");
            let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
            ws.push((self.w(&name)?, scale, name));
        }
        let refs: Vec<_> = ws.iter().map(|(w, s, n)| (n.as_str(), *w, *s)).collect();
        Ok(self.linear.mul_mat_multi(ctx, &refs, h, norm))
    }

    pub(crate) fn moe<'a>(
        &'a self,
        ctx: &'a Context,
        tap: &Tap<'a>,
        graph: &std::cell::RefCell<Graph<'a>>,
        il: usize,
        h: Tensor<'a>,
        n: i64,
        prompt: bool,
    ) -> Result<Tensor<'a>, ModelError> {
        self.moe_rows(ctx, tap, graph, il, h, n, None, prompt)
    }

    /// [`Self::moe`] with a verify step's row map (I32 `[n]`, [`pad_rows::map`]): Legion's
    /// NVFP4 expert kernels route each row's expert work as the row it names (padding rows as
    /// their sequences' first rows), the same bits for every row that names itself; so does
    /// Legion's expert matvec over an expert cache with `--moe-pad-remap-tiered` (opt-in, not
    /// byte-identical end to end). With `--moe-pad-skip-tiered` (opt-in) that matvec skips the
    /// padding rows' pairs of experts no real row chose instead (`pad_rows::skip_tiered`).
    /// `prompt` ([`Pass::prompt_only`]): every group of the pass is a prompt step, the only
    /// passes `--nvfp4-expert-rows-bf16` gives BF16 expert rows (decode and verify rows stay
    /// F32, #1183).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_rows<'a>(
        &'a self,
        ctx: &'a Context,
        tap: &Tap<'a>,
        graph: &std::cell::RefCell<Graph<'a>>,
        il: usize,
        h: Tensor<'a>,
        n: i64,
        rows: Option<Tensor<'a>>,
        prompt: bool,
    ) -> Result<Tensor<'a>, ModelError> {
        let hp = &self.hp;
        let used = hp.n_expert_used;
        let router = !tap.on && hp.n_expert <= fused::ROUTER_MAX_EXPERTS;
        let gate_w = self.lw(il, "ffn_gate_inp_shexp.weight")?;
        let gate_op = !tap.on && fused::shexp_gate_on(gate_w);
        // `--tp-moe-ep` (#798): this rank holds a share of the layer's experts.
        let ep = self.tp_ep.as_ref().filter(|ep| {
            self.lw(il, "ffn_up_exps.weight")
                .is_ok_and(|up| ep.cuts(up))
        });
        if ep.is_some() && self.experts.is_some() {
            return Err(ModelError::Unsupported(
                "--tp-moe-ep with an expert cache".into(),
            ));
        }
        // `--decode-fuse` (#773): the joined router and gate matmul and the layer's tail as one
        // op (`decode_fuse::moe_tail`), or else `shexp_gate`. Not with a share of the experts:
        // the tail op weighs every routed expert.
        let df = ep.is_none()
            && router
            && decode_fuse::takes(hp, n)
            && !gate_w.is_host()
            && !gate_w.is_meta();
        // One quantization of h for its MMQ projections on a GB10 (#526).
        let users = [
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_gate_inp_shexp.weight",
        ];
        let hq = self.share_q8(ctx, tap.on, il, &users, h);
        // The selection where `router_mv` made it with the logits.
        let mut routed = None;
        // Small steps: the router's and the shared expert's gate logits from one matmul (#256),
        // where the fused router and gate ops read their strided rows.
        let (logits, gate_logit) = match (router && (gate_op || df))
            .then(|| {
                let pair = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight");
                self.lmm_joined(ctx, il, pair.0, pair.1, h)
            })
            .flatten()
        {
            Some((l, g)) => (l, Some(g)),
            // Steps of 1-3 tokens: the router's matmul and selection as one op (#836).
            // Both read the router's BF16 copy where it has one ([`router_bf16`]): the same bits.
            None => match self
                .plain_weight(il, "ffn_gate_inp.weight")
                .filter(|&w| router && router_mv::takes(w, h, used, il))
            {
                Some(rw) => {
                    let rw = self
                        .router_bf16(il, rw)
                        .filter(|&b| router_mv::fits(b, h, used))
                        .unwrap_or(rw);
                    let (l, ids, w) = router_mv::router_mv(ctx, rw, h, used, il);
                    routed = Some((ids, w));
                    (l, None)
                }
                // Steps of 4-16 tokens: ggml's F32 matmul as `dense_mv` computes it (#836).
                None => match self
                    .plain_weight(il, "ffn_gate_inp.weight")
                    .filter(|_| router)
                    .and_then(|w| self.router_bf16(il, w))
                    .filter(|&b| dense_mv::takes_wide(b, h))
                {
                    Some(b) => (dense_mv::mul_mat_wide(ctx, b, h), None),
                    None => (self.lmm_q(ctx, il, "ffn_gate_inp.weight", h, hq)?, None),
                },
            },
        };
        tap.tap("ffn_moe_logits", il, logits);
        let (selected, w) = if !router {
            // ggml's chain, whose steps the taps name
            let probs = ctx.soft_max(logits);
            let selected = ctx.argsort_top_k(probs, used as i32);
            tap.tap("ffn_moe_probs", il, probs);
            tap.tap("ffn_moe_topk", il, selected);
            let w = ctx.get_rows(ctx.reshape_3d(probs, 1, hp.n_expert, n), selected);
            let w = ctx.reshape_2d(w, used, n);
            (
                selected,
                ctx.div(w, ctx.clamp(ctx.sum_rows(w), 6.103_515_6e-5, f32::INFINITY)),
            )
        } else if let Some(r) = routed {
            r
        } else {
            // With `--nvfp4-once` the top-k kernel also writes the routing plan of the NVFP4
            // expert ops (#836), which then launch without building one.
            let plan =
                ep.is_none() && self.experts.is_none() && nvfp4_once::plan_fits(gate_w, used, n);
            // Joined top-k and weights (#814): the weights go into the graph below, before
            // the experts pull the ids in (not with `--decode-fuse`, which adds the ids).
            fused::router_planned(ctx, logits, used, plan, !df)
        };
        tap.tap("ffn_moe_weights_norm", il, w);
        let w2 = w;
        let mut w = ctx.reshape_3d(w, 1, used, n);
        // Into the graph now, before the experts pull the ids in. On ggml's chain this keeps
        // llama.cpp's order (softmax, reshape, argsort, view, get_rows, then the norm), which
        // ggml-cuda runs as one fused top-k MoE kernel instead of nine (with
        // `--stable-fusions`, whatever the layout).
        if router {
            // With `--decode-fuse` the tail op finds the weights itself.
            graph.borrow_mut().expand(if df { selected } else { w });
        } else {
            stable_fusions::pin_router(graph, logits, w);
        }
        // `--moe-route-record` (#803): the router's choices, copied out for `legion ep-place`.
        if route_rec::on() {
            let rec = route_rec::record(ctx, il, hp.n_expert as usize, selected);
            graph.borrow_mut().expand(rec);
        }

        let x = ctx.reshape_3d(h, hp.n_embd, 1, n);
        // h's Q8_1 blocks, where the hyper-connection mixer wrote them (#819): the routed
        // experts' SwiGLU op reads them instead of quantizing h again.
        let h_q8_1 = (q8_group::part(hp.arch.name(), q8_group::PART_EXPERTS)
            && q8_group::has_quantized(ctx, h))
        .then(|| q8_group::quantized(ctx, h));
        let fused = self.fused_moe(n);
        // `--mtp-moe-fused-rows` (#858): an MTP head's passes from that many rows on.
        let fused = if mtp_moe::is_head(il, hp.n_layer) {
            let floor = mtp_moe::rows();
            mtp_moe::choose(self.moe_kernel, fused, n, floor, cfg!(feature = "cuda"))
        } else {
            fused
        };
        // The fused op's result is weighted and summed already; the graph path's is per expert.
        let mut summed = None;
        // The shared expert's output and gate logits if built before the experts, and whether
        // the fused op added them.
        let mut shexp_built = None;
        let mut shexp_summed = false;
        // `--hc-fuse` (#774; qwen4exp on a V100, or on sm_80+, #826): the gated
        // shared expert added in the fused op's sum (ggml's sigmoid, mul and add bits, #526),
        // so the op's result is the layer's whole output, which the next hyper-connection norm
        // then makes from the op's pairs.
        // Steps the fused op takes only: smaller (decode) steps keep main's graph.
        let hc_fuse = fused.is_some()
            && !tap.on
            && !gate_op
            && !df
            && crate::model::qwen4::hc_fuse()
            && crate::model::qwen4::params(hp).is_some();
        let experts = if let Some(t) = self.experts.as_ref().and_then(|r| r.layer(il)) {
            // Tiered: cached experts from VRAM slots, the others from host memory in place.
            // With `--moe-pad-remap-tiered`, a verify step whose experts all run on Legion's
            // expert matvec looks up its padding rows as their sequences' first rows
            // (`pad_rows`; not byte-identical end to end, see there).
            // With `--moe-pad-skip-tiered` (opt-in) such a step hands the matvec its row map
            // instead (`skip`), which skips the padding rows' pairs of experts no real row chose
            // and computes the others as without it (see `pad_rows`).
            let mv_rows = rows.filter(|r| {
                r.ne()[0] == selected.ne()[1]
                    && fused.is_none()
                    && !t.gathers(n)
                    && !t.stages(n)
                    && t.cpu_share(n).is_none()
                    && expert_mv::glu_down_fits(t.weight(1), t.weight(0), t.weight(2), x, selected)
            });
            let pad = mv_rows.filter(|_| pad_rows::tiered());
            let skip =
                mv_rows.filter(|&r| pad_rows::skip_tiered() && expert_mv::pad_fits(r, selected));
            let ids = t.remap_rows(ctx, graph, selected, n, pad);
            let refs = |w: [Tensor<'a>; 3]| moe_ffn::ExpertRefs {
                up: w[0],
                gate: w[1],
                down: w[2],
                table: None,
            };
            let ok = |w: [Tensor<'a>; 3]| fused.is_some() && moe_ffn::supported(&refs(w));
            let tiered = [t.weight(0), t.weight(1), t.weight(2)];
            // A big step reads uncached experts from staged VRAM copies.
            if ok(tiered) {
                let shexp = if hc_fuse {
                    let (sh, g) = self.shexp(ctx, il, h, hq, gate_logit)?;
                    shexp_built = Some((sh, g));
                    Some(moe_ffn::Shexp { sh, g }).filter(|s| moe_ffn::shexp_fits(s, hp.n_embd, n))
                } else {
                    None
                };
                // With split waits, the op itself waits for each piece of the layer's copy.
                summed = t.staged_split(ctx, graph, h, n, |x, ws, stage| {
                    let (pair, stage) = (self.moe_pair_out, Some(stage));
                    moe_ffn::moe_ffn_staged_shexp(ctx, x, ids, w2, refs(ws), pair, stage, shexp)
                });
                if summed.is_none() {
                    summed = t.staged(ctx, graph, h, n, |x, ws| {
                        let pair = self.moe_pair_out;
                        moe_ffn::moe_ffn_shexp(ctx, x, ids, w2, refs(ws), pair, shexp)
                    });
                }
                if summed.is_none() && fused == Some(MoeKernel::Fused) {
                    // Forced: in place (on a GPU, host experts over the link).
                    summed = Some(moe_ffn::moe_ffn_shexp(
                        ctx,
                        h,
                        ids,
                        w2,
                        refs(tiered),
                        self.moe_pair_out,
                        shexp,
                    ));
                }
                shexp_summed = summed.is_some() && shexp.is_some();
            }
            if summed.is_some() {
                None
            } else {
                match t.staged_experts(ctx, graph, x, ids, n) {
                    Some(out) => Some(out),
                    None => {
                        // Small steps: Legion's expert matvec reads the misses at the link rate.
                        let mm = |k: usize, x: Tensor<'a>| {
                            let w = t.weight(k);
                            if !t.gathers(n) && expert_mv::fits(w, x, ids) {
                                expert_mv::mul_mat_id_tiered(ctx, w, x, ids, t.n_slots())
                            } else if !t.gathers(n) && mmq_ids::usable(w, x, ids) {
                                // `--expert-mmq` (#814): ggml's MMQ bits, reading each
                                // expert's rows once for only its own tokens.
                                mmq_ids::mul_mat_id(ctx, w, x, ids)
                            } else {
                                t.mul_mat_id(ctx, k, x, ids)
                            }
                        };
                        let (w_up, w_gate) = (t.weight(0), t.weight(1));
                        // The missed experts on a side stream beside the cached ones (#253).
                        let w = [w_gate, w_up, t.weight(2)];
                        // The host computes a share of the missed experts (#268, opt-in).
                        let shared = t.cpu_share(n).and_then(|share| {
                            let (slots, n_expert) = (t.n_slots(), t.n_expert());
                            cpu_share::ops::glu_down(ctx, w, x, ids, slots, n_expert, il, share)
                        });
                        let split = shared.or_else(|| {
                            (!t.gathers(n))
                                .then(|| {
                                    expert_mv::overlap::glu_down_tiered(ctx, w, x, ids, t.n_slots())
                                })
                                .flatten()
                        });
                        split.or_else(|| {
                            let act = if !t.gathers(n) && expert_mv::glu_fits(w_gate, w_up, x, ids)
                            {
                                let preq =
                                    h_q8_1.filter(|&q| expert_mv::glu_preq_fits(w_gate, x, ids, q));
                                match preq {
                                    Some(q) => expert_mv::glu_tiered_preq_pad(
                                        ctx,
                                        w_gate,
                                        w_up,
                                        x,
                                        ids,
                                        t.n_slots(),
                                        q,
                                        skip,
                                    ),
                                    None => expert_mv::glu_tiered_pad(
                                        ctx,
                                        w_gate,
                                        w_up,
                                        x,
                                        ids,
                                        t.n_slots(),
                                        skip,
                                    ),
                                }
                            } else {
                                let up = mm(0, x);
                                let gate = mm(1, x);
                                ctx.mul(ctx.silu(gate), up)
                            };
                            let down = t.weight(2);
                            match skip {
                                Some(_) if !t.gathers(n) && expert_mv::fits(down, act, ids) => {
                                    Some(expert_mv::mul_mat_id_tiered_pad(
                                        ctx,
                                        down,
                                        act,
                                        ids,
                                        t.n_slots(),
                                        skip,
                                    ))
                                }
                                _ => Some(mm(2, act)),
                            }
                        })
                    }
                }
            }
        } else {
            let refs = moe_ffn::ExpertRefs {
                up: self.lw(il, "ffn_up_exps.weight")?,
                gate: self.lw(il, "ffn_gate_exps.weight")?,
                down: self.lw(il, "ffn_down_exps.weight")?,
                // With a share of the experts, the others' pairs are skipped.
                table: ep.map(|ep| ep.table(il)),
            };
            // Per-expert scales (NVFP4 checkpoints) stay on the mul_mat_id path, and so do
            // experts read in place from host memory unless forced.
            let scaled = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"]
                .iter()
                .any(|k| self.w(&format!("blk.{il}.{k}.scale")).is_ok());
            let place_ok = fused == Some(MoeKernel::Fused) || !refs.up.is_host();
            // With a share of the experts on CUDA, decode and verify steps take Legion's expert
            // matvec over the share (the router's ids as they are: the others' pairs skipped,
            // their rows 0), where the row split takes ggml's mmvq; the fused op is slower for
            // 1-3 tokens (#798). A forced fused op (`--moe-kernel fused`) keeps every step.
            let ep_mv = ep.filter(|_| {
                fused.is_none()
                    && !scaled
                    && expert_mv::share_fits(refs.gate, refs.up, refs.down, x, selected)
            });
            // Bigger steps take the fused op: it skips the others' pairs, and ggml-cuda's
            // mul_mat_id cannot take the remapped ids below (its grouping counts an expert once
            // per token).
            let ep_fused = ep.is_some() && ep_mv.is_none() && cfg!(feature = "cuda");
            let fuses = ep_mv.is_none()
                && (fused.is_some() || ep_fused)
                && !scaled
                && place_ok
                && moe_ffn::supported(&refs);
            // Per-expert scales (an NVFP4 checkpoint's experts, #1150) keep the remap below,
            // where every expert matmul of the step takes each (token, slot) pair on its own.
            let ep_pairs = ep_fused
                && scaled
                && ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"]
                    .iter()
                    .all(|k| {
                        let name = format!("blk.{il}.{k}.weight");
                        let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
                        self.w(&name)
                            .is_ok_and(|w| self.linear.mmid_per_pair(&name, w, scale, n))
                    });
            if ep_fused && !fuses && !ep_pairs {
                return Err(ModelError::Unsupported(format!(
                    "--tp-moe-ep: layer {il}'s experts on the fused expert op"
                )));
            }
            // The weight, scale and name of layer `il`'s expert stack `name`.
            let weight = |name: &str| -> Result<_, ModelError> {
                let name = format!("blk.{il}.{name}");
                let scale = linear::scale_name(&name).and_then(|s| self.w(&s).ok());
                Ok((self.w(&name)?, scale, name))
            };
            // Those pairs on Legion's grouped NVFP4 kernels (#1176): the others' ids -1, which
            // the kernels skip (their rows 0), instead of computing them on the rank's first
            // expert. The same bits for the rank's own pairs, and the others' add nothing
            // either way (`--no-tp-moe-ep-skip`: the remap below).
            let ep_skip = ep_pairs && tp_ep::skip() && {
                let (wg, sg, gn) = weight("ffn_gate_exps.weight")?;
                let (wu, su, un) = weight("ffn_up_exps.weight")?;
                let (wd, sd, dn) = weight("ffn_down_exps.weight")?;
                self.linear
                    .takes_skips((&gn, wg, sg), (&un, wu, su), (&dn, wd, sd), n)
            };
            // The paths below compute every routed pair: with a share of the experts (on the
            // CPU), the others' ids point into the share and their weights are zero.
            let selected = match ep {
                Some(ep) if ep_skip => {
                    let (ids, masked) = ep.remap_skips(ctx, il, selected, w2, n);
                    w = ctx.reshape_3d(masked, 1, used, n);
                    ids
                }
                Some(ep) if !fuses && ep_mv.is_none() => {
                    let (ids, masked) = ep.remap(ctx, il, selected, w2, n);
                    w = ctx.reshape_3d(masked, 1, used, n);
                    ids
                }
                _ => selected,
            };
            if let Some(ep) = ep_mv {
                let share = ep.share(il);
                let act = expert_mv::glu_share(ctx, refs.gate, refs.up, x, selected, share);
                Some(expert_mv::mul_mat_id_share(
                    ctx, refs.down, act, selected, share,
                ))
            } else if fuses {
                // On a GB10, the gated shared expert added in the op's sum (#526); with
                // `--hc-fuse` too (above).
                let shexp = if hc_fuse || (!tap.on && !gate_op && moe_ffn::shexp_sum()) {
                    let (sh, g) = self.shexp(ctx, il, h, hq, gate_logit)?;
                    shexp_built = Some((sh, g));
                    Some(moe_ffn::Shexp { sh, g }).filter(|s| moe_ffn::shexp_fits(s, hp.n_embd, n))
                } else {
                    None
                };
                shexp_summed = shexp.is_some();
                summed = Some(moe_ffn::moe_ffn_shexp(
                    ctx,
                    h,
                    selected,
                    w2,
                    refs,
                    self.moe_pair_out,
                    shexp,
                ));
                None
            } else if !scaled
                && !refs.up.is_host()
                && expert_mv::on_device()
                && n >= 2
                && expert_mv::glu_fits(refs.gate, refs.up, x, selected)
            {
                // Verify steps, experts in GPU memory: gate and up in Legion's SwiGLU expert
                // matvec (opt-in, #309), ggml's bits per op (generated text can still differ:
                // layout-dependent ggml-cuda fusions, #472); down stays on ggml.
                let act = expert_mv::glu_tiered(ctx, refs.gate, refs.up, x, selected, 0);
                Some(self.lmm_id(ctx, il, "ffn_down_exps.weight", act, selected)?)
            } else if let Some(out) = {
                // Legion's NVFP4 kernels: gate, up and the SwiGLU as one grouped op (the GEMM at
                // prefill, #483; the grouped SwiGLU mv at decode, its plan shared with the down
                // mv, #836), the same bits as the three ops below. With `ep_skip`, the ids'
                // -1 pairs skipped (`takes_skips` held above, so this path is taken).
                let (wg, sg, gn) = weight("ffn_gate_exps.weight")?;
                let (wu, su, un) = weight("ffn_up_exps.weight")?;
                let (wd, sd, dn) = weight("ffn_down_exps.weight")?;
                let (g, u, d) = ((&*gn, wg, sg), (&*un, wu, su), (&*dn, wd, sd));
                let rows = rows.filter(|r| r.ne()[0] == selected.ne()[1]);
                if ep_skip {
                    let l = &self.linear;
                    l.swiglu_ffn_id_skips(ctx, g, u, d, x, selected, rows)
                } else {
                    let bf16 = prompt && crate::model::nvfp4_mm::expert_rows_bf16(n);
                    self.linear
                        .swiglu_ffn_id(ctx, g, u, d, x, selected, rows, bf16)
                }
            } {
                Some(out)
            } else if let Some(act) = self.once_glu(ctx, il, x, selected)? {
                // Verify steps on a GB10 with `--nvfp4-once`: gate, up, their scales and the
                // SwiGLU in one op reading each routed expert once, the bits of the ops below.
                Some(self.lmm_id(ctx, il, "ffn_down_exps.weight", act, selected)?)
            } else if mmvq_glu::takes(refs.gate, refs.up, n) {
                // Decode and verify steps on ggml's NVFP4 MMVQ: gate, up and the SwiGLU in the
                // order ggml-cuda fuses into one launch, the bits of the ops below (opt-in, #836).
                let gate = self.lmm_id(ctx, il, "ffn_gate_exps.weight", x, selected)?;
                let up = self.lmm_id(ctx, il, "ffn_up_exps.weight", x, selected)?;
                let act = ctx.swiglu_split(gate, up);
                Some(self.lmm_id(ctx, il, "ffn_down_exps.weight", act, selected)?)
            } else {
                let up = self.lmm_id(ctx, il, "ffn_up_exps.weight", x, selected)?;
                let gate = self.lmm_id(ctx, il, "ffn_gate_exps.weight", x, selected)?;
                let act = ctx.mul(ctx.silu(gate), up);
                Some(self.lmm_id(ctx, il, "ffn_down_exps.weight", act, selected)?)
            }
        };
        if df
            && summed.is_none()
            && let Some(experts) = experts
        {
            let (sh, g) = match shexp_built.take() {
                Some(s) => s,
                None => self.shexp(ctx, il, h, hq, gate_logit)?,
            };
            if decode_fuse::moe_tail_fits(experts, logits, selected, sh, g) {
                return Ok(decode_fuse::moe_tail(ctx, experts, logits, selected, sh, g));
            }
            shexp_built = Some((sh, g));
        }
        let out = match (summed, experts) {
            (Some(out), _) => out,
            // One rounding, whatever the layout (`--moe-combine`, #410).
            (None, Some(experts)) if fused::moe_combine_fits(experts, w) => {
                fused::moe_combine(ctx, experts, w)
            }
            (None, Some(experts)) => {
                let weighted = ctx.mul(experts, w); // [n_embd, used, n]
                let nb = weighted.nb();
                let mut out = ctx.view_2d(weighted, [hp.n_embd, n], nb[2], 0);
                for e in 1..used as usize {
                    out = ctx.add(out, ctx.view_2d(weighted, [hp.n_embd, n], nb[2], e * nb[1]));
                }
                // ggml-cuda's one-rounding fusion in every layer (`--stable-fusions`, #472).
                stable_fusions::pin_weighted_sum(graph, weighted, out);
                out
            }
            (None, None) => unreachable!("one of the paths ran"),
        };
        tap.tap("ffn_moe_out", il, out);
        if shexp_summed {
            return Ok(out);
        }

        let (sh, g) = match shexp_built {
            Some(s) => s,
            None => self.shexp(ctx, il, h, hq, gate_logit)?,
        };
        // With `--shexp-gate`, on a GPU, the sigmoid, the scaling and the sum in one op
        // (otherwise ggml's chain, as on the CPU).
        if (gate_op && fused::shexp_gate_fits(out, sh, g, gate_w))
            || (df && fused::shexp_gate_shapes(out, sh, g))
        {
            return Ok(fused::shexp_gate(ctx, out, sh, g));
        }
        // ggml's unary ops read contiguous rows: a joined matmul's are strided.
        let g = if g.is_contiguous() { g } else { ctx.cont(g) };
        let sh = ctx.mul(sh, ctx.sigmoid(g));
        tap.tap("ffn_shexp_gated", il, sh);
        Ok(ctx.add(out, sh))
    }

    /// Layer `il`'s shared expert of `h` (`hq`: `h` quantized for its projections) and its gate
    /// logits (`gate_logit`, if the router's matmul has them already): `(sh, g)`.
    fn shexp<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        h: Tensor<'a>,
        hq: Option<shared_q8::Shared<'a>>,
        gate_logit: Option<Tensor<'a>>,
    ) -> Result<(Tensor<'a>, Tensor<'a>), ModelError> {
        // As in `ffn`: one SwiGLU node, which ggml-cuda fuses with the two matvecs. Steps of
        // 2-16 tokens on a GB10 read BF16 gate and up in one dense matvec: ggml's bits (#860).
        let names = ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"];
        // Q8_0 gate and up of 2-8 tokens on a V100 ([`q8_group::part`]): matvecs and SwiGLU in one launch
        // from h quantized once, ggml's bits (#819). One token stays on ggml, which fuses them
        // the same way, unless h's Q8_1 blocks are there already (the hyper-connection mixer's).
        let sh = match self.dense_weights(il, &names) {
            Some(ws) if dense_mv::takes_small(&ws, h) => {
                let r = dense_mv::mul_mat(ctx, &ws, h);
                ctx.swiglu_split(r[0], r[1])
            }
            Some(ws)
                if q8_group::part(self.hp.arch.name(), q8_group::PART_SHEXP)
                    && (h.ne()[1] > 1 || q8_group::has_quantized(ctx, h))
                    && q8_group::takes_swiglu(ws[0], ws[1], h) =>
            {
                // The down projection reading the SwiGLU's Q8_1 blocks (round 6).
                let down = self
                    .dense_weights(il, &[names[0], names[1], "ffn_down_shexp.weight"])
                    .map(|d| d[2])
                    .filter(|&d| {
                        q8_group::part(self.hp.arch.name(), q8_group::PART_SHEXP_DOWN)
                            && q8_group::takes_swiglu_down(ws[0], ws[1], d, h)
                    });
                if let Some(down) = down {
                    let sh = q8_group::swiglu_down(ctx, ws[0], ws[1], down, h, il);
                    let g = match gate_logit {
                        Some(g) => g,
                        None => self.lmm_q(ctx, il, "ffn_gate_inp_shexp.weight", h, hq)?,
                    };
                    return Ok((sh, g));
                }
                q8_group::swiglu(ctx, ws[0], ws[1], h)
            }
            // Prompt steps: BF16 gate and up stored back to back as one GEMM (#836, `joined`;
            // small steps keep ggml's fused matvecs and SwiGLU).
            _ => match (h.ne()[1] > joined::MAX_TOKENS)
                .then(|| self.lmm_joined(ctx, il, names[0], names[1], h))
                .flatten()
            {
                Some((gate, up)) => ctx.swiglu_split(gate, up),
                None => ctx.swiglu_split(
                    self.lmm_q(ctx, il, names[0], h, hq)?,
                    self.lmm_q(ctx, il, names[1], h, hq)?,
                ),
            },
        };
        let sh = self.lmm(ctx, il, "ffn_down_shexp.weight", sh)?;
        let g = match gate_logit {
            Some(g) => g,
            None => self.lmm_q(ctx, il, "ffn_gate_inp_shexp.weight", h, hq)?,
        };
        Ok((sh, g))
    }
}

/// A 16- or 32-bit weight's values as F32 ([`Model::requant_mtp`]).
fn mtp_values(t: Tensor<'_>) -> Vec<f32> {
    if t.ty() == types::F32 {
        t.to_f32()
    } else {
        crate::ggml::dequantize(t.ty(), &t.bytes(), t.elements())
    }
}

/// L2 normalization along dim 0, written the way llama.cpp's `build_gdn_l2_norm` does.
fn l2_norm<'a>(ctx: &'a Context, x: Tensor<'a>, eps: f32) -> Tensor<'a> {
    let d = x.ne()[0] as f32;
    ctx.scale(ctx.rms_norm(x, eps / d), 1.0 / d.sqrt())
}
