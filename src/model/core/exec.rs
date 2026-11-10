//! [`Executor`]: the model, its backend and the shared decode pools in one owner, running batch
//! steps. A step whose shape equals the previous step of its kind (decode: one token per
//! sequence; mixed: anything with a prefill chunk) reuses that step's graph as is: no rebuild,
//! no re-planning, and the same tensor addresses every run, which lets the CUDA backend replay
//! it as a captured CUDA graph. With paged attention a shape does not include the sequences'
//! lengths, so a decode graph lives as long as the batch keeps its size, and a long prompt's
//! full-size chunks next to a steady decode batch repeat one mixed graph.
//!
//! The layers can be split over several devices (a pipeline of [`Stage`]s, e.g. the two GPUs
//! of an NVLink pair): each stage owns its backend, the KV cache, recurrent states and replay
//! buffers of its layers, and graphs of its own; a step runs the stages in order and hands the
//! residual stream from one to the next (a device-to-device copy). The first stage embeds the
//! tokens, the last computes the logits and runs the MTP head.

use std::ops::Range;

use crate::ggml::{
    Backend, BufferType, ComputeAllocator, Context, Device, GgmlError, KeptGraph, KeptParts,
    SharedAllocator, Tensor, types,
};

use super::kv_type::KvTypes;
use super::mtp::MtpOutput;
use super::spec::SpecPool;
use super::state::{KvPool, KvRing, KvTier, StatePool};
use super::{
    GRAPH_SIZE, Group, GroupShape, Hparams, InputParts, Inputs, KV_BLOCK, Logits, Model,
    ModelError, SeqInput, StepOpts, StepOutput, batch_shape, graph_size, shape_diff, with_kv,
};
use crate::ggml::model_bytes::{Kind, MAX_STAGES};
use crate::ggml::pinned_io::PinnedIo;
pub use capture::StateCapture;

mod after_split;
mod budget;
mod capture;
mod commit_hold;
mod dataflow;
mod early;
mod freeing;
mod headroom;
mod launch_ahead;
mod mtp_ahead;
mod mtp_chain;
mod mtp_graphs;
mod mtp_pipe;
mod mtp_window;
mod pair_step;
mod par_mtp;
mod par_step;
mod pipe;
mod prompt_ahead;
mod prompt_stats;
mod ready_buffers;
mod remote;
mod rows;
mod rows_out;
mod spare;
mod step_fill;
mod step_front;
mod taper;
mod tp_lead;
mod verify_ahead;
mod verify_kept;

use early::{Early, EarlyProgress, round_active};
use freeing::Freeing;
use mtp_ahead::{Ahead, NextPass, Staged};
use mtp_graphs::{MtpGraphs, take_in_waits_for_reuse};
use mtp_pipe::{InFlight, Pass};
use rows::DeferredRows;

pub use mtp_chain::ChainOutput;
pub use mtp_graphs::set_kept as set_mtp_kept_graphs;
pub use prompt_ahead::PromptNext;
pub use prompt_ahead::set_enabled as set_prompt_build_ahead;
pub use prompt_stats::set_logging as set_prompt_step_stats;
pub use ready_buffers::set_enabled as set_prompt_buffer_reserve;
pub use remote::{Remote, from_wire};
pub use tp_lead::TpLead;
pub use verify_kept::default_budget as default_verify_budget;
pub use verify_kept::kept as verify_kept_graphs;
pub use verify_kept::set_kept as set_verify_kept_graphs;

/// Which state pool a slot is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    /// The running sequences' states.
    Seq,
    /// Saved states (prefix-cache checkpoints).
    Saved,
}

/// One device's part of the layer stack.
struct Stage {
    /// The graphs of the last [`DECODE_KEPT`] decode shapes (a batch split into micro-batches
    /// over the stages alternates between its halves) and of the last [`MIXED_KEPT`] mixed
    /// (prefill, verify) shapes (a pipelined prompt alternates between full micro-chunks and its
    /// last, shorter one), each with its own compute buffer, most recent last. Declared first so
    /// they are dropped before the pools they reference (see `KeptGraph::new`).
    decode: Vec<(Vec<GroupShape>, KeptGraph)>,
    mixed: Vec<(Vec<GroupShape>, KeptGraph)>,
    /// The graphs of sequence groups' steps in flight over the stages ([`Executor::pipe_round`]),
    /// most recent last, all in one compute buffer (`piped_shared`): a stage runs one at a time.
    piped: Vec<(Vec<GroupShape>, KeptGraph)>,
    piped_shared: Option<SharedAllocator>,
    /// On a meta backend: the one compute buffer all kept decode graphs share, and the one
    /// the mixed ones share ([`Context::alloc_graph_shared`]).
    shared: [Option<SharedAllocator>; 2],
    /// Speculative commits' graphs (see [`SpecPool::commit`]).
    commits: super::spec::CommitGraphs,
    layers: Range<usize>,
    kv: KvPool,
    /// The buffer type of the KV pool's own context (its rings, and its blocks without a host
    /// tier) when the decode state spills into host memory (#1131,
    /// [`Executor::new_staged_spill`]); `None`: as without.
    kv_spill: Option<BufferType>,
    rec: StatePool,
    saved: StatePool,
    /// Replay buffers for verify steps ([`Executor::enable_spec`]).
    spec: Option<SpecPool>,
    /// Pinned host memory the stage reads big captured rows back through ([`Stage::read_rows`]).
    bounce: Option<Context>,
    /// A second one, for the rows of the round after (see [`DeferredRows`]).
    bounce_b: Option<Context>,
    /// Pinned host memory a step's smaller captured rows (a decode or verify step's) are read
    /// back through ([`read_rows`]).
    bounce_small: Option<Context>,
    /// Pinned host memory a step's argmaxes and logits rows are read back through
    /// ([`Inputs::output_with`]).
    ids_pin: Option<Context>,
    /// Pinned host memory the MTP passes' inputs and outputs go through (the last stage's,
    /// [`mtp_window`]), and a decode or verify step's inputs ([`step_fill`]).
    mtp_io: PinnedIo,
    /// Pinned host memory a long-dataflow step's inputs are queued through
    /// ([`dataflow`]'s `fill_queued`); kept across steps, reserved whole before a step's first
    /// graph (#794).
    flow_io: PinnedIo,
    /// A second one: with `--long-dataflow-async` the micro-chunks take turns, so a stage's
    /// next inputs are queued while the copies of the ones before may still be pending.
    flow_io_b: PinnedIo,
    /// The compute buffer kept through an expert-cache lend for the first graph built after
    /// the restore ([`spare`]).
    spare: Option<ComputeAllocator>,
    /// Compute buffers allocated at load for the kept prompt graphs, taken where a prompt
    /// graph would start a new allocator ([`ready_buffers`]).
    ready: Vec<ComputeAllocator>,
    backend: Backend,
}

impl Stage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        hp: &Hparams,
        backend: Backend,
        layers: Range<usize>,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
        ring: Option<KvRing>,
        types: KvTypes,
        kv_spill: Option<BufferType>,
    ) -> Result<Self, ModelError> {
        let buft = backend.buffer_type();
        let attn = KvPool::attention_layers(hp, layers.clone());
        Ok(Self {
            decode: Vec::new(),
            mixed: Vec::new(),
            piped: Vec::new(),
            piped_shared: None,
            shared: [None, None],
            commits: Vec::new(),
            kv: KvPool::new_ring(hp, kv_spill.unwrap_or(buft), kv_blocks, &attn, ring, types)?,
            kv_spill,
            rec: StatePool::new_layers(hp, buft, slots, layers.clone())?,
            saved: StatePool::new_layers(hp, buft, saved_slots, layers.clone())?,
            spec: None,
            bounce: None,
            bounce_b: None,
            bounce_small: None,
            ids_pin: None,
            mtp_io: PinnedIo::default(),
            flow_io: PinnedIo::default(),
            flow_io_b: PinnedIo::default(),
            spare: None,
            ready: Vec::new(),
            layers,
            backend,
        })
    }

    fn forget_graphs(&mut self) {
        self.decode.clear();
        self.mixed.clear();
        self.piped.clear();
        self.piped_shared = None;
        self.commits.clear();
        self.shared = [None, None];
    }

    /// The kept graph for `shape`, if built.
    fn kept(&self, shape: &[GroupShape]) -> Option<&KeptGraph> {
        kept_in(&self.decode, &self.mixed, shape)
    }
}

/// Captured rows from this size on read back through pinned memory ([`read_rows`]).
const BOUNCE_MIN_BYTES: usize = 4 << 20;

/// Pinned host memory of at least `n` bytes in `bounce` (grown as needed), if the device has a
/// host buffer type and it allocates.
/// Captured rows of a stage's graph already copied into its round's bounce buffer (see
/// [`Executor::stage_output`]): from `src` (device), `bytes`, to `dst` (pinned host).
#[derive(Debug, Clone, Copy)]
struct HiddenRead {
    src: usize,
    dst: usize,
    bytes: usize,
}

fn bounce_for<'b>(
    bounce: &'b mut Option<Context>,
    backend: &Backend,
    n: usize,
) -> Option<Tensor<'b>> {
    let have = bounce
        .as_ref()
        .and_then(|c| c.tensor("bounce"))
        .map_or(0, |b| b.nbytes());
    if have < n {
        *bounce = None;
        let buft = backend.device().host_buffer_type()?;
        let ctx = Context::new(1, 0);
        // Room for a quarter more: the next prompt's rounds may be a little bigger.
        ctx.new_tensor(types::F32, &[(n + n / 4).div_ceil(4) as i64])
            .named("bounce");
        ctx.alloc_tensors(buft).ok()?;
        *bounce = Some(ctx);
    }
    bounce.as_ref().and_then(|c| c.tensor("bounce"))
}

/// Reads captured rows `t` of a stage on `backend` back to the host, through pinned memory:
/// big ones (a prompt round's: ~170 MB per 4096 tokens of Flash-Next) through `bounce`, as
/// reading them into pageable memory ran at ~1.7 GB/s on a V100 over NVLink; smaller ones (a
/// verify step's ~200 KB on Flash-Next) through `small`, a fixed [`BOUNCE_MIN_BYTES`]: on the
/// GB10 the copy into a new pageable vector now and then took 3-12 ms instead of ~30 µs (#506).
/// Big ones go into `spare` when it has room (#363).
fn read_rows(
    bounce: &mut Option<Context>,
    small: &mut Option<Context>,
    spare: &mut rows_out::Spare,
    backend: &Backend,
    t: Tensor<'_>,
) -> Vec<f32> {
    let n = t.nbytes();
    let big = n >= BOUNCE_MIN_BYTES;
    let via = if big {
        bounce_for(bounce, backend, n)
    } else {
        bounce_for(small, backend, BOUNCE_MIN_BYTES)
    };
    match via {
        // Out of the bounce buffer on several threads (#363).
        Some(via) if big => {
            let (src, n) = t.read_via(via);
            rows_out::copy_out(spare, src, n / 4)
        }
        Some(via) => t.to_f32_via(via),
        None => t.to_f32(),
    }
}

/// [`read_rows`] of captured rows `t` (`[rows][width]`) from row `from` on only (#1133): the
/// output keeps every row, those before `from` holding anything ([`rows_out::copy_out_tail`]).
/// A DFlash drafter takes in only a prompt chunk's last rows (its ring): a 16,384-token step of
/// Qwen3.8-27B read 5 x 335 MB back with the GPU idle, now at most its 4,096-row window.
fn read_rows_from(
    bounce: &mut Option<Context>,
    small: &mut Option<Context>,
    spare: &mut rows_out::Spare,
    backend: &Backend,
    t: Tensor<'_>,
    from: usize,
) -> Vec<f32> {
    let width = t.ne()[0] as usize;
    let n = t.nbytes() / 4;
    let skip = (from * width).min(n);
    if skip == 0 {
        return read_rows(bounce, small, spare, backend, t);
    }
    let bytes = (n - skip) * 4;
    let via = if bytes >= BOUNCE_MIN_BYTES {
        bounce_for(bounce, backend, bytes)
    } else {
        bounce_for(small, backend, BOUNCE_MIN_BYTES)
    };
    match via {
        Some(via) => {
            let (src, _) = t.read_tail_via(via, skip * 4);
            rows_out::copy_out_tail(spare, src, n, skip)
        }
        None => t.to_f32(),
    }
}

/// Writes `parts` one after another into input `t` of a stage on `backend`, big ones through
/// pinned memory (see [`read_rows`]), which the parts are copied into directly.
fn write_rows(bounce: &mut Option<Context>, backend: &Backend, t: Tensor<'_>, parts: &[&[f32]]) {
    let n: usize = parts.iter().map(|p| std::mem::size_of_val(*p)).sum();
    match (n >= BOUNCE_MIN_BYTES)
        .then(|| bounce_for(bounce, backend, n))
        .flatten()
    {
        Some(via) => t.fill_f32_via(via, |dst| rows::gather(parts, dst)),
        None if parts.len() == 1 => t.set_f32(parts[0]),
        None => t.set_f32(&parts.concat()),
    }
}

type Kept = (Vec<GroupShape>, KeptGraph);

fn kept_in<'k>(
    decode: &'k [Kept],
    mixed: &'k [Kept],
    shape: &[GroupShape],
) -> Option<&'k KeptGraph> {
    decode
        .iter()
        .chain(mixed)
        .find(|(s, _)| s == shape)
        .map(|(_, g)| g)
}

/// Mixed graphs a stage keeps: a prompt chunk's and a verify batch's two micro-batches (with
/// `--verify-kept-graphs`, the prompt shapes' and apart from the verify ones', [`verify_kept`]).
const MIXED_KEPT: usize = 3;
/// [`MIXED_KEPT`] while the expert cache is lent (see [`Executor::lend_experts`]).
const LENT_MIXED_KEPT: usize = 1;
/// Device memory a step graph leaves free while the expert cache is lent (#490). ggml-cuda
/// allocates its pool as a graph computes and aborts the process when that fails, so a graph
/// that leaves less fails as one that does not fit, before it runs. Qwen3.8-Flash-Next's pool
/// grew 8-58 MiB in 3-12K-token steps on top of what earlier steps had mapped, and an 11.8K-token
/// step with 148 MiB left aborted. Before, a graph over 1.8 GiB that leaves less than this
/// was padded past the free memory (`ComputeAllocator::padded`) and failed anyway.
pub const LENT_SCRATCH: usize = 512 << 20;

/// While the expert cache is lent, after a step graph's compute buffer (`held` bytes on
/// `device`) was allocated: the out-of-memory error the allocation would have been, should it
/// leave the device less than [`LENT_SCRATCH`].
/// The buffer type that spills `stage`'s step compute buffers into host memory
/// (`residency::cuda::spill_type`), on a CUDA device with unified memory.
#[cfg(feature = "cuda")]
fn spill_type(stage: &Stage) -> Option<BufferType> {
    let index = stage.backend.name().strip_prefix("CUDA")?.parse().ok()?;
    crate::residency::cuda::spill_type(index)
}

#[cfg(not(feature = "cuda"))]
fn spill_type(_: &Stage) -> Option<BufferType> {
    None
}

#[cfg(feature = "cuda")]
fn tier_back(slot: usize, tokens: usize, lent: usize) -> Result<(), String> {
    crate::residency::cuda::tp_tier_back(slot, tokens, lent)
}

#[cfg(not(feature = "cuda"))]
fn tier_back(_: usize, _: usize, _: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "cuda")]
fn tier_trim(slot: usize) -> Result<(), String> {
    crate::residency::cuda::tp_tier_trim(slot)
}

#[cfg(not(feature = "cuda"))]
fn tier_trim(_: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "cuda")]
fn tier_lend(chunks: usize) -> usize {
    crate::residency::cuda::tp_tier_lend(chunks)
}

#[cfg(not(feature = "cuda"))]
fn tier_lend(_: usize) -> usize {
    0
}

fn lent_scratch(device: Device, held: usize, buffer: String) -> Result<(), GgmlError> {
    if device.memory().0 >= LENT_SCRATCH {
        Ok(())
    } else {
        Err(GgmlError::Alloc {
            bytes: held + LENT_SCRATCH,
            buffer,
        })
    }
}

/// Decode graphs a stage keeps: a batch's micro-batches ([`Executor::step`]).
const DECODE_KEPT: usize = 2;

/// Graphs of each kind a stage on a meta backend keeps, all in one shared buffer: the shapes
/// a batch alternates between (the gather budget's split steps, verify rows as the draft depth
/// changes, spans crossing a padding boundary).
const META_KEPT: usize = 6;

/// Fewest sequences of a decode or verify batch that several stages split into micro-batches.
const MIN_SPLIT_SEQS: usize = 2;

/// One group of a step measured by [`Executor::measure_groups`]: `n_seqs` sequences of
/// `n_tok` tokens each after `n_past`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepPart {
    pub n_seqs: usize,
    pub n_tok: usize,
    pub n_past: usize,
    /// A speculative verify group (every row's logits); otherwise the last row's.
    pub verify: bool,
}

/// The groups of a measured step: one per part, in consecutive slots, each sequence reading
/// its block list of `blocks` and the first `n_tok` of `toks`.
fn fake_groups<'b>(
    parts: &[StepPart],
    blocks: &'b [Vec<Vec<u32>>],
    toks: &'b [u32],
) -> Vec<Group<'b>> {
    let mut slot = 0;
    parts
        .iter()
        .zip(blocks)
        .map(|(p, blocks)| {
            let logits = if p.verify {
                Logits::First(p.n_tok)
            } else {
                Logits::Last
            };
            let g = Group {
                slot,
                seqs: blocks
                    .iter()
                    .map(|blocks| SeqInput {
                        tokens: &toks[..p.n_tok],
                        n_past: p.n_past,
                        blocks,
                        ring: 0,
                        logits,
                        media: None,
                    })
                    .collect(),
                verify: p.verify,
            };
            slot += p.n_seqs;
            g
        })
        .collect()
}

pub struct Executor {
    /// In layer order; dropped (graphs first) before the model they reference.
    stages: Vec<Stage>,
    /// The graphs of MTP passes (on the last stage): draft passes (one token per sequence) and
    /// take-in passes, the last few shapes of each ([`MtpGraphs`]).
    mtp_decode: MtpGraphs,
    mtp_mixed: MtpGraphs,
    /// `--glm-parallel-mtp`: per GPU, the graphs of MTP passes on every GPU ([`par_mtp`]).
    par_mtp: Vec<par_mtp::ParMtpGraphs>,
    /// The take-in passes' compute buffer kept through an expert-cache lend ([`spare`]).
    mtp_spare: Option<ComputeAllocator>,
    /// Paged attention's block-table width: the most blocks one sequence can have.
    max_seq_blocks: usize,
    /// Layers whose outputs every step returns ([`Self::capture_hidden`]).
    hidden: Vec<usize>,
    /// A captured-rows vector handed back for the next step's copy ([`Self::recycle_rows`]).
    spare_rows: rows_out::Spare,
    /// The MTP head's KV cache ([`Self::enable_mtp`]), on the last stage.
    mtp_kv: Option<KvPool>,
    /// See [`Self::set_pipeline_micro`].
    pipeline_micro: usize,
    /// See [`Self::set_pipeline_riders`].
    pipeline_riders: bool,
    /// See [`Self::set_micro_taper`].
    micro_taper: usize,
    /// Per stage: its KV pool's host tier, if any ([`Self::set_kv_tiers`]).
    kv_tiers: Vec<Option<KvTier>>,
    /// Per stage: the buffer type its KV pool goes in when it has no host tier, in place of
    /// the backend's ([`Self::set_kv_bufts`]).
    kv_bufts: Vec<Option<BufferType>>,
    /// Mixed graphs each stage keeps: [`MIXED_KEPT`], or [`LENT_MIXED_KEPT`] while a long
    /// prompt borrows the expert cache.
    mixed_kept: usize,
    /// Kept mixed graphs set aside while another is built, which count against
    /// [`Self::mixed_kept`] all the same ([`prompt_ahead`]).
    mixed_aside: usize,
    /// The prompt chunk the engine runs after the next step ([`Self::set_prompt_next`]).
    prompt_next: Option<PromptNext>,
    /// The next step's first captured row read back ([`Self::set_hidden_from`]).
    hidden_from: usize,
    /// The CUDA device the next step leaves its captured rows on
    /// ([`Self::set_hidden_on_device`]).
    hidden_on_device: Option<usize>,
    /// The stop tokens a greedy pick below `min_tokens` may not pick ([`Self::set_stop_tokens`]).
    stop_tokens: Vec<u32>,
    /// See [`Self::set_stage_cpus`].
    stage_cpus: Vec<Vec<u32>>,
    /// GLM's parallel decode steps ([`par_step`]).
    par: par_step::ParStats,
    /// Pair parallelism ([`Self::set_pair_tp`]); its helper graphs drop before the model.
    pair: pair_step::PairState,
    /// See [`Self::set_split_batches`].
    split_batches: bool,
    /// See [`Self::set_dataflow`].
    dataflow: bool,
    /// See [`Self::set_dataflow_peer`].
    dataflow_peer: bool,
    /// See [`Self::set_dataflow_async`].
    dataflow_async: bool,
    /// Whether a round gathers its next micro-chunk's n-gram rows while it computes
    /// ([`Model::gather_ple_ahead`]); off only to compare against in tests.
    ple_ahead: bool,
    /// A commit was started on the stages' streams and not waited for ([`Self::commit`]).
    commit_pending: bool,
    /// See [`Self::set_gather_budget`]; 0 without.
    gather_budget: usize,
    /// See [`Self::set_kv_slot_blocks`]; 0 without.
    kv_slot_blocks: usize,
    /// See [`Self::set_kv_slot_host_budget`].
    kv_slot_host_budget: Option<usize>,
    /// See [`Self::set_kv_slot_lend`].
    kv_slot_lend: usize,
    /// See [`Self::set_kv_slot_lend`]: VRAM per GPU the steps still need past the load.
    kv_slot_lend_keep: usize,
    /// See [`Self::enable_ring_saves`]: per stage, its rings' copies (empty: none).
    ring_saves: Vec<Option<crate::ggml::Context>>,
    /// Copies each stage's `ring_saves` hold.
    ring_save_slots: usize,
    /// Expert-cache work deferred to run beside a pass that reads other layers (see
    /// [`Self::set_after_overlap`]): the last target step's (its token count), run while the
    /// next MTP pass computes; the last MTP pass's, run while the next target step computes.
    deferred_target: Option<(usize, crate::ggml::model_bytes::Kind)>,
    /// The share of a target step's deferred cache work left for the last stage's layers
    /// (these layers), after the first MTP pass ran the rest ([`mtp_window::tail_split`]).
    deferred_tail: Option<(usize, crate::ggml::model_bytes::Kind, Range<usize>)>,
    deferred_mtp: Option<usize>,
    after_overlap: bool,
    /// See [`Self::set_launch_ahead`].
    launch_ahead: bool,
    /// See [`Self::set_commit_beside_draft`].
    commit_beside_draft: bool,
    /// See [`Self::set_commit_last_late`].
    commit_last_late: bool,
    /// A verify step's commit held for the next MTP pass ([`commit_hold`]).
    held_commit: commit_hold::HeldCommit,
    /// The host time from an MTP pass's picks to the next step's first launch ([`step_front`]).
    front_clock: step_front::FrontClock,
    /// The last step's host laps, for `--tp-profile` ([`Self::take_step_laps`]).
    step_laps: Option<crate::tp_net::StepLaps>,
    /// The host time the last stage launch spent writing its late n-gram rows (`launch_stage`).
    late_rows_took: std::time::Duration,
    /// See [`Self::set_mtp_ahead`].
    mtp_ahead: bool,
    /// The inputs the last MTP pass queued for the next one ([`mtp_ahead`]), until the next
    /// pass.
    mtp_staged: Option<Staged>,
    /// See [`Self::set_mtp_pipe_rows`].
    mtp_pipe_rows: usize,
    /// [`Self::enable_mtp`] was called.
    mtp_enabled: bool,
    /// The weights' addresses, to count the kept graphs' model bytes (built with the first
    /// graph).
    weight_addrs: Option<std::collections::HashSet<usize>>,
    /// Model bytes and time of the steps run since start (#391).
    bytes: crate::ggml::model_bytes::Totals,
    /// Per stage: what its attention layers read per position (#391).
    kv_rates: Vec<super::traffic::KvRates>,
    /// The expert cache's traffic counted into `bytes` so far: per stage's layers, then the
    /// MTP head's ([`Self::credit_experts`]).
    experts_seen: Vec<crate::ggml::model_bytes::ExpertTraffic>,
    /// See [`Self::set_mtp_head_copy`].
    mtp_head_copy: Option<crate::ggml::ffi::ggml_type>,
    /// See [`Self::set_mtp_lm_head`].
    mtp_lm_head: crate::spec::DraftLmHead,
    /// Sequence groups' steps in flight over the stages ([`Self::pipe_round`]).
    pipe: Vec<pipe::Piece>,
    /// Graphs each stage keeps for them ([`Self::set_pipe_graphs`]).
    pipe_kept: usize,
    /// The groups' rounds' launch threads, one per local stage, made at the first round and
    /// kept ([`Self::pipe_round`], #756).
    pipe_pool: Option<par_step::pool::LaunchPool>,
    /// A round whose exiting step was handed out while other stages may still compute
    /// ([`Self::pipe_round_begin`] until [`Self::pipe_round_end`], #762).
    pipe_run: Option<pipe::PipeRun>,
    /// An error of a round finished on the way ([`Self::wait_commit`] mid-round), for the next
    /// [`Self::pipe_round_end`].
    pipe_failed: Option<String>,
    /// The last round's piece time per local stage (`None`: no piece), and whether its exiting
    /// step was handed out early ([`Self::pipe_last_round`], #765).
    pipe_last: (Vec<Option<std::time::Duration>>, bool),
    /// See [`Self::set_compute_budget`].
    compute_budget: Vec<usize>,
    /// See [`Self::set_verify_budget`].
    verify_budget: Vec<usize>,
    /// The kept graphs the last expert-cache restore handed its slot thread to drop
    /// ([`Self::restore_experts`]), until they are gone.
    freeing: Freeing,
    /// Rank 0 of a model split over two nodes: the layers after the stages' ([`remote`]).
    remote: Option<Remote>,
    /// Rank 0 of tensor parallelism over machines: the other ranks ([`tp_lead`]).
    tp: Option<TpLead>,
    /// See [`Self::no_deferred_commits`].
    no_defer: bool,
    /// Rank 1 of a split model: the residual rows of the step's micro-chunks that came over
    /// the link, which the first stage takes in order ([`Self::step_from`]).
    ext_rows: std::sync::Mutex<std::collections::VecDeque<Vec<f32>>>,
    /// Recurrent states the next step saves as it runs ([`Self::set_state_captures`]), and
    /// the ones the last step took.
    captures: Vec<StateCapture>,
    applied: Vec<StateCapture>,
    model: Model,
}

impl Drop for Executor {
    /// A commit still running reads and writes the state pools: it finishes before they go.
    fn drop(&mut self) {
        self.wait_commit();
        self.freeing.wait();
    }
}

impl Executor {
    /// Pools of `kv_blocks` KV blocks, `slots` running-state slots and `saved_slots` saved-state
    /// slots, in the backend's memory.
    pub fn new(
        model: Model,
        backend: Backend,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
    ) -> Result<Self, ModelError> {
        let n = model.hp.n_layer;
        Self::new_staged(model, vec![(backend, 0..n)], kv_blocks, slots, saved_slots)
    }

    /// Like [`Self::new`] with the layers split over `stages` (each a backend and the layers
    /// it runs, together covering `0..n_layer` in order): every stage holds the pools of its
    /// own layers. The weights must already be placed accordingly.
    pub fn new_staged(
        model: Model,
        stages: Vec<(Backend, Range<usize>)>,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
    ) -> Result<Self, ModelError> {
        Self::new_staged_ring(
            model,
            stages,
            kv_blocks,
            slots,
            saved_slots,
            None,
            KvTypes::default(),
        )
    }

    /// [`Self::new_staged`] with the sliding-window layers' caches in rings `ring`
    /// ([`KvRing`]; their sequences carry ring ids) and the K and V caches in `types`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_staged_ring(
        model: Model,
        stages: Vec<(Backend, Range<usize>)>,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
        ring: Option<KvRing>,
        types: KvTypes,
    ) -> Result<Self, ModelError> {
        let n = stages.len();
        Self::new_staged_spill(
            model,
            stages,
            kv_blocks,
            slots,
            saved_slots,
            ring,
            types,
            vec![None; n],
        )
    }

    /// [`Self::new_staged_ring`] with stage `i`'s KV pool context (its rings, and its blocks
    /// without a host tier) in buffer type `kv_spill[i]` when given (#1131): a decode state
    /// that does not fit beside the weights spills into host memory there, VRAM first.
    #[allow(clippy::too_many_arguments)]
    pub fn new_staged_spill(
        model: Model,
        stages: Vec<(Backend, Range<usize>)>,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
        ring: Option<KvRing>,
        types: KvTypes,
        kv_spill: Vec<Option<BufferType>>,
    ) -> Result<Self, ModelError> {
        let hp = &model.hp;
        // The layers in order; a node of a model split over machines holds its share only
        // ([`remote`]), else every layer.
        let mut next = stages.first().map_or(0, |(_, r)| r.start);
        for (_, r) in &stages {
            assert!(
                r.start == next && r.end > r.start,
                "stages must cover the layers in order"
            );
            next = r.end;
        }
        assert!(next <= hp.n_layer, "stages past the last layer");
        let stages = stages
            .into_iter()
            .enumerate()
            .map(|(i, (b, r))| {
                let spill = kv_spill.get(i).copied().flatten();
                Stage::new(hp, b, r, kv_blocks, slots, saved_slots, ring, types, spill)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_stages(model, stages, kv_blocks)
    }

    /// GLM's parallel steps (`--glm-parallel-decode`, [`crate::model::glm5next::par`]): a
    /// stage per GPU of `backends` (GPU `d` the model's stage `d`), each over every layer: the
    /// latent and indexer caches of every MLA layer, its share of every KDA layer's heads'
    /// states ([`crate::model::glm5next::par::shard_hparams`]). Every step runs on all of them
    /// at once ([`par_step`]).
    #[allow(clippy::too_many_arguments)]
    pub fn new_glm_par(
        model: Model,
        backends: Vec<Backend>,
        kv_blocks: usize,
        slots: usize,
        saved_slots: usize,
        ring: Option<KvRing>,
        types: KvTypes,
    ) -> Result<Self, ModelError> {
        let n_dev = backends.len();
        if model.glm_par.as_ref().is_none_or(|p| p.plan.n_dev != n_dev) {
            return Err(ModelError::Unsupported(
                "parallel steps without a plan for these GPUs".into(),
            ));
        }
        let shard = crate::model::glm5next::par::shard_hparams(&model.hp, n_dev);
        let n_layer = model.hp.n_layer;
        let stages = backends
            .into_iter()
            .map(|b| {
                Stage::new(
                    &shard,
                    b,
                    0..n_layer,
                    kv_blocks,
                    slots,
                    saved_slots,
                    ring,
                    types,
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut exec = Self::with_stages(model, stages, kv_blocks)?;
        exec.par.every_step = true;
        Ok(exec)
    }

    /// The executor over `stages` (built over pools of `kv_blocks` blocks).
    fn with_stages(model: Model, stages: Vec<Stage>, kv_blocks: usize) -> Result<Self, ModelError> {
        let hp = &model.hp;
        for s in &stages {
            s.kv.check_sources(hp, s.layers.clone())
                .map_err(ModelError::Layout)?;
        }
        let kv_rates = stages
            .iter()
            .map(|s| super::traffic::KvRates::new(hp, s.layers.clone(), s.kv.types()))
            .collect();
        let bytes = crate::ggml::model_bytes::Totals {
            n_stages: stages.len(),
            ..Default::default()
        };
        Ok(Self {
            stages,
            mtp_decode: MtpGraphs::new(mtp_graphs::kept()),
            mtp_mixed: MtpGraphs::new(mtp_graphs::kept()),
            par_mtp: Vec::new(),
            mtp_spare: None,
            max_seq_blocks: kv_blocks.max(1),
            hidden: Vec::new(),
            spare_rows: rows_out::Spare::default(),
            mtp_kv: None,
            pipeline_micro: 0,
            pipeline_riders: false,
            micro_taper: 0,
            kv_tiers: Vec::new(),
            kv_bufts: Vec::new(),
            mixed_kept: MIXED_KEPT,
            mixed_aside: 0,
            prompt_next: None,
            hidden_from: 0,
            hidden_on_device: None,
            stop_tokens: Vec::new(),
            stage_cpus: Vec::new(),
            par: Default::default(),
            pair: Default::default(),
            split_batches: false,
            dataflow: false,
            dataflow_peer: false,
            dataflow_async: false,
            ple_ahead: true,
            commit_pending: false,
            gather_budget: 0,
            kv_slot_blocks: 0,
            kv_slot_host_budget: None,
            kv_slot_lend: 0,
            kv_slot_lend_keep: 0,
            ring_saves: Vec::new(),
            ring_save_slots: 0,
            deferred_target: None,
            deferred_tail: None,
            deferred_mtp: None,
            after_overlap: false,
            launch_ahead: false,
            commit_beside_draft: false,
            commit_last_late: false,
            held_commit: Default::default(),
            front_clock: Default::default(),
            step_laps: None,
            late_rows_took: std::time::Duration::ZERO,
            mtp_ahead: false,
            mtp_staged: None,
            mtp_pipe_rows: 0,
            mtp_enabled: false,
            weight_addrs: None,
            bytes,
            kv_rates,
            experts_seen: Vec::new(),
            mtp_head_copy: None,
            mtp_lm_head: Default::default(),
            pipe: Vec::new(),
            pipe_kept: pipe::PIPE_KEPT,
            pipe_pool: None,
            pipe_run: None,
            pipe_failed: None,
            pipe_last: (Vec::new(), false),
            compute_budget: Vec::new(),
            verify_budget: Vec::new(),
            freeing: Freeing::default(),
            remote: None,
            tp: None,
            no_defer: false,
            ext_rows: Default::default(),
            captures: Vec::new(),
            applied: Vec::new(),
            model,
        })
    }

    fn last(&self) -> &Stage {
        self.stages.last().expect("at least one stage")
    }

    /// The weight bytes and time of the steps run since start, by kind of step (#391): what
    /// the graphs must read of the weights ("model bytes"), not measured traffic.
    pub fn model_bytes(&self) -> crate::ggml::model_bytes::Totals {
        self.bytes
    }

    /// Counts `kept`'s model bytes ([`KeptGraph::set_model_bytes`]) against the model's weights.
    fn count_bytes(
        weight_addrs: &mut Option<std::collections::HashSet<usize>>,
        model: &Model,
        kept: &mut KeptGraph,
    ) {
        let addrs = weight_addrs.get_or_insert_with(|| model.weight_addrs());
        kept.set_model_bytes(&|t| addrs.contains(&(t as usize)));
    }

    fn forget_graphs(&mut self) {
        for s in &mut self.stages {
            s.forget_graphs();
        }
        self.pair.forget_graphs();
        self.mtp_decode.clear();
        self.mtp_mixed.clear();
        self.par_mtp
            .iter_mut()
            .for_each(par_mtp::ParMtpGraphs::clear);
    }

    /// The layers each stage runs.
    pub fn stage_layers(&self) -> Vec<Range<usize>> {
        self.stages.iter().map(|s| s.layers.clone()).collect()
    }

    pub fn hp(&self) -> &Hparams {
        &self.model.hp
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    /// How routed experts run from the next step on (see [`crate::model::core::MoeKernel`]);
    /// cached graphs are rebuilt.
    pub fn set_moe_kernel(&mut self, k: crate::model::core::MoeKernel) {
        self.forget_graphs();
        self.model.set_moe_kernel(k);
    }

    /// How the fused expert op keeps per-pair outputs from the next step on (see
    /// [`crate::model::moe_ffn::PairOut`]); cached graphs are rebuilt.
    pub fn set_moe_pair_out(&mut self, p: crate::model::moe_ffn::PairOut) {
        self.forget_graphs();
        self.model.set_moe_pair_out(p);
    }

    /// The expert cache, if the MoE layers are tiered.
    pub fn experts_mut(&mut self) -> Option<&mut crate::residency::experts::ExpertResidency> {
        self.model.experts_mut()
    }

    /// Routes the MoE layers through tiered experts (see [`crate::residency`]) from the next
    /// step on.
    pub fn set_experts(&mut self, r: Option<crate::residency::experts::ExpertResidency>) {
        self.forget_graphs();
        // Deferred work belongs to the cache being replaced.
        self.deferred_target = None;
        self.deferred_tail = None;
        self.deferred_mtp = None;
        // The new cache counts its traffic from zero.
        self.experts_seen.clear();
        self.model.set_experts(r);
    }

    pub fn backend_name(&self) -> String {
        let names: Vec<String> = self.stages.iter().map(|s| s.backend.name()).collect();
        names.join("+")
    }

    /// Room to save `slots` copies of rings ([`Self::save_ring`]): per stage, that many
    /// rings' worth of its ring layers in `bufts[stage]` (pinned host memory). Returns the
    /// bytes taken. After the last [`Self::resize_kv`].
    pub fn enable_ring_saves(
        &mut self,
        bufts: &[crate::ggml::BufferType],
        slots: usize,
    ) -> Result<usize, ModelError> {
        self.tp_enable_ring_saves(slots)?;
        let (mut saves, mut bytes) = (Vec::new(), 0);
        for (s, &buft) in self.stages.iter().zip(bufts) {
            let c = s.kv.ring_copies(buft, slots)?;
            bytes += c.as_ref().map_or(0, |c| c.1);
            saves.push(c.map(|c| c.0));
        }
        self.ring_saves = saves;
        self.ring_save_slots = slots;
        Ok(bytes)
    }

    /// Bytes of one ring's copy over every stage ([`Self::enable_ring_saves`] takes this per
    /// slot); 0 without rings.
    pub fn ring_copy_bytes(&self) -> usize {
        self.stages.iter().map(|s| s.kv.ring_copy_bytes()).sum()
    }

    /// Whether every GPU works on every token (GLM's parallel steps, [`Self::new_glm_par`]).
    pub fn parallel_steps(&self) -> bool {
        self.par.every_step
    }

    /// The copies [`Self::save_ring`] has room for (0: none).
    pub fn ring_save_slots(&self) -> usize {
        if self.ring_saves.is_empty() {
            0
        } else {
            self.ring_save_slots
        }
    }

    /// Saves ring `r`'s positions `0..n` (all of them past its length) in copy `slot`, or
    /// (`restore`) puts that copy back into ring `r`. Between steps: the ring's sequence runs
    /// none meanwhile.
    pub fn save_ring(&mut self, r: usize, slot: usize, n: usize, restore: bool) {
        let t0 = std::time::Instant::now();
        self.tp_save_ring(r, slot, n, restore);
        self.wait_commit();
        for s in &mut self.stages {
            s.backend.synchronize();
        }
        for (s, copies) in self.stages.iter().zip(&self.ring_saves) {
            if let Some(c) = copies {
                s.kv.copy_ring(c, r, slot, n, !restore);
            }
        }
        tracing::debug!(
            ring = r,
            slot,
            n,
            restore,
            ms = t0.elapsed().as_secs_f64() * 1e3,
            "ring copy"
        );
    }

    /// The sliding-window layers' rings, if the pools have them.
    pub fn kv_ring(&self) -> Option<KvRing> {
        self.stages[0].kv.ring()
    }

    pub fn kv_blocks(&self) -> usize {
        self.stages[0].kv.n_blocks()
    }

    /// The KV cache's element types.
    pub fn kv_types(&self) -> crate::model::core::kv_type::KvTypes {
        self.stages[0].kv.types()
    }

    /// KV blocks each stage gives up for `stage_blocks` host-tier staging slots (#352) in its
    /// device memory (the MTP head's own pool's slots with the last stage's): the most over
    /// the stages, as every pool has the same blocks.
    pub fn kv_stage_cost(&self, stage_blocks: usize) -> usize {
        let n = self.stages.len();
        self.stages
            .iter()
            .enumerate()
            .map(|(si, s)| {
                let mtp = self
                    .mtp_kv
                    .as_ref()
                    .filter(|_| si + 1 == n)
                    .map_or(0, |m| m.stage_bytes(stage_blocks));
                (s.kv.stage_bytes(stage_blocks) + mtp).div_ceil(s.kv.bytes_per_block().max(1))
            })
            .max()
            .unwrap_or(0)
    }

    /// Bytes per cell of every KV pool tensor whose rows are cells (the stages' and the MTP
    /// head's own pools, the rings aside), over all the GPUs a tensor-parallel cache splits.
    pub fn kv_cell_tensor_bytes(&self) -> Vec<usize> {
        self.stages
            .iter()
            .map(|s| &s.kv)
            .chain(self.mtp_kv.as_ref())
            .flat_map(KvPool::cell_tensor_bytes)
            .collect()
    }

    /// Bytes another KV block takes over every pool (the stages' and the MTP head's own),
    /// the rings aside.
    pub fn kv_block_bytes(&self) -> usize {
        self.stages
            .iter()
            .map(|s| s.kv.bytes_per_block())
            .chain(self.mtp_kv.as_ref().map(KvPool::bytes_per_block))
            .sum()
    }

    /// The KV host tier's staging slots, in blocks (0 without a tier or staging).
    pub fn kv_stage_blocks(&self) -> usize {
        self.stages[0].kv.stage_blocks()
    }

    /// Per stage: the bytes one KV block takes over its pools (the MTP head's own with the
    /// last stage's), and the bytes its host-tier staging slots take in device memory.
    pub fn kv_stage_bytes(&self) -> Vec<(usize, usize)> {
        let n = self.stages.len();
        self.stages
            .iter()
            .enumerate()
            .map(|(si, s)| {
                let mtp = self.mtp_kv.as_ref().filter(|_| si + 1 == n);
                let block = s.kv.bytes_per_block() + mtp.map_or(0, KvPool::bytes_per_block);
                let blocks = s.kv.stage_blocks();
                let staging =
                    s.kv.stage_bytes(blocks) + mtp.map_or(0, |m| m.stage_bytes(m.stage_blocks()));
                (block, staging)
            })
            .collect()
    }

    /// Replaces every KV pool (the stages' and the MTP head's own) by one of `n_blocks`
    /// blocks, empty, dropping the kept graphs that read the old ones. For sizing the cache
    /// after measuring the compute buffers with a small one.
    pub fn resize_kv(&mut self, n_blocks: usize) -> Result<(), ModelError> {
        self.tp_resize_kv(n_blocks)?;
        self.flush_commits()?;
        self.wait_commit();
        let bufts: Vec<BufferType> = (0..self.stages.len()).map(|si| self.kv_buft(si)).collect();
        let hp = &self.model.hp;
        // The rings an MTP head reads in every draft pass (#1159): first into a spilled pool,
        // so in VRAM rather than in the host memory past it.
        let hot = if self.model.has_mtp() && hp.arch.mtp_shares_kv() {
            hp.arch.mtp_kv_layers(hp)
        } else {
            Vec::new()
        };
        for (si, s) in self.stages.iter_mut().enumerate() {
            s.forget_graphs();
            s.commits.clear();
            let attn = KvPool::attention_layers(hp, s.layers.clone());
            let attn = match s.kv_spill {
                Some(_) => KvPool::hot_first(hp, attn, &hot),
                None => attn,
            };
            let (ring, types) = (s.kv.ring(), s.kv.types());
            let tier = self.kv_tiers.get(si).copied().flatten();
            let buft = match (s.kv_spill, tier) {
                (Some(spill), _) => spill,
                (None, Some(_)) => s.backend.buffer_type(),
                (None, None) => bufts[si],
            };
            // The old pool's memory goes first (an empty pool in its place), then the new one.
            s.kv = KvPool::new_ring(hp, buft, 1, &[], None, types)?;
            s.kv = KvPool::new_tiered(hp, buft, n_blocks, &attn, ring, types, tier)?;
            s.kv.set_slot_blocks(self.kv_slot_blocks);
        }
        self.mtp_decode.clear();
        self.mtp_mixed.clear();
        self.par_mtp
            .iter_mut()
            .for_each(par_mtp::ParMtpGraphs::clear);
        if let Some(types) = self.mtp_kv.as_ref().map(KvPool::types) {
            // The head's own cache runs on the last stage: its tier, or its buffer type.
            let tier = self.kv_tiers.last().copied().flatten();
            let buft = match tier {
                Some(_) => self.last().backend.buffer_type(),
                None => bufts[bufts.len() - 1],
            };
            self.mtp_kv = None;
            self.mtp_kv = Some(KvPool::new_tiered(
                hp,
                buft,
                n_blocks,
                &[hp.n_layer],
                None,
                types,
                tier,
            )?);
        }
        self.max_seq_blocks = self.max_seq_blocks.clamp(1, n_blocks.max(1));
        Ok(())
    }

    /// Gives the stages' KV pools host tiers (#352), one entry per stage (`None`: all in its
    /// device's memory), from the next [`Self::resize_kv`] on.
    pub fn set_kv_tiers(&mut self, tiers: Vec<Option<KvTier>>) {
        let stage = tiers.iter().flatten().map(|t| t.stage_blocks).max();
        self.model.set_kv_stage_blocks(stage.unwrap_or(0));
        self.kv_tiers = tiers;
    }

    /// Puts the stages' KV pools (the MTP head's own with the last stage's) in the buffer
    /// types `bufts`, one entry per stage (`None`: the stage backend's), from the next
    /// [`Self::resize_kv`] on. A stage with a host tier keeps its tier's layout.
    pub fn set_kv_bufts(&mut self, bufts: Vec<Option<BufferType>>) {
        self.kv_bufts = bufts;
    }

    /// Per stage: its KV pool's tensors whose rows are the pool's cells
    /// ([`KvPool::block_tensors`]). The MTP head's own pool is not among them.
    pub fn kv_block_tensors(&self) -> Vec<Vec<Tensor<'_>>> {
        self.stages.iter().map(|s| s.kv.block_tensors()).collect()
    }

    /// Waits until no queued step, commit or deferred pass of any stage still runs: before
    /// memory an earlier step may have read is unmapped (`--kv-vmm`'s lend, #814).
    pub fn quiesce(&mut self) -> Result<(), ModelError> {
        self.wait_commit();
        self.flush_deferred()?;
        self.wait_commit();
        for s in &mut self.stages {
            s.backend.synchronize();
        }
        Ok(())
    }

    /// The buffer type of stage `si`'s KV pool without a host tier.
    fn kv_buft(&self, si: usize) -> BufferType {
        match self.kv_bufts.get(si).copied().flatten() {
            Some(b) => b,
            None => self.stages[si].backend.buffer_type(),
        }
    }

    /// Bytes each stage's compute buffer takes for a prompt step of `tokens` tokens after
    /// `n_past` (as the longest prompt chunks at the end of a long context build it), measured
    /// by building and allocating that step's graphs (and then dropping them). A stage keeps
    /// up to two such graphs at once, each with its own buffer.
    pub fn measure_prompt_step(
        &mut self,
        tokens: usize,
        n_past: usize,
    ) -> Result<Vec<usize>, ModelError> {
        self.measure_step(1, tokens, n_past, false)
    }

    /// Allocates now, on each stage that keeps its prompt graphs' buffers with headroom (an
    /// integrated GPU's), one compute buffer per kept mixed graph, as big as a prompt step of
    /// `tokens` tokens after `n_past` needs ([`Self::measure_prompt_step`]), and writes each
    /// once ([`ready_buffers`]). Per stage: the buffers reserved and the bytes measured
    /// (`(0, 0)` where none is wanted). A buffer that does not fit is skipped with a warning.
    pub fn reserve_prompt_buffers(
        &mut self,
        tokens: usize,
        n_past: usize,
    ) -> Result<Vec<(usize, usize)>, ModelError> {
        let lent = self.mixed_kept == LENT_MIXED_KEPT;
        let on = ready_buffers::enabled();
        // Not where a byte budget bounds the kept graphs (`set_compute_budget`): bigger
        // buffers would count against it.
        let budgeted = |s: usize| self.compute_budget.get(s).is_some_and(|&b| b > 0);
        let wanted: Vec<usize> = (self.stages.iter().enumerate())
            .map(|(s, st)| {
                let kind = st.backend.device().kind();
                let meta = st.backend.is_meta();
                let n = ready_buffers::wanted(on, kind, meta, lent, self.mixed_kept);
                if budgeted(s) { 0 } else { n }
            })
            .collect();
        if wanted.iter().all(|&n| n == 0) {
            return Ok(vec![(0, 0); self.stages.len()]);
        }
        let sizes = self.measure_prompt_step(tokens, n_past)?;
        let mut out = Vec::with_capacity(self.stages.len());
        for (s, st) in self.stages.iter_mut().enumerate() {
            let bytes = sizes.get(s).copied().unwrap_or(0);
            let buft = st.backend.buffer_type();
            while st.ready.len() < wanted[s] && bytes > 0 {
                let free = st.backend.device().memory().0;
                if !ready_buffers::fits(free, bytes) {
                    tracing::warn!(
                        stage = s,
                        mib = bytes >> 20,
                        free_mib = free >> 20,
                        reserved = st.ready.len(),
                        "prompt compute buffer not reserved: too little memory free"
                    );
                    break;
                }
                match ComputeAllocator::headroom(buft).touched(bytes) {
                    Ok(a) => st.ready.push(a),
                    Err(e) => {
                        tracing::warn!(stage = s, error = %e, "prompt compute buffer not reserved");
                        break;
                    }
                }
            }
            out.push((st.ready.len(), if wanted[s] > 0 { bytes } else { 0 }));
        }
        Ok(out)
    }

    /// [`Self::measure_prompt_step`] for a step of `n_seqs` sequences of `n_tok` tokens each
    /// after `n_past` (slots `0..n_seqs`): a decode step (`n_tok` 1) or, with `verify`, a
    /// speculative verify step of a full batch (every row's logits).
    pub fn measure_step(
        &mut self,
        n_seqs: usize,
        n_tok: usize,
        n_past: usize,
        verify: bool,
    ) -> Result<Vec<usize>, ModelError> {
        self.measure_groups(&[StepPart {
            n_seqs,
            n_tok,
            n_past,
            verify,
        }])
    }

    /// [`Self::measure_step`] for a step of several groups at once (a prompt chunk batched
    /// with a verify group, as the scheduler runs them), in slots `0..` in order.
    pub fn measure_groups(&mut self, parts: &[StepPart]) -> Result<Vec<usize>, ModelError> {
        // The buffers reserved for the kept prompt graphs stay out of a measurement (its
        // graphs are dropped, and their buffers with them).
        let ready: Vec<_> = (self.stages.iter_mut())
            .map(|s| std::mem::take(&mut s.ready))
            .collect();
        let out = self.measure_groups_now(parts);
        for (s, r) in self.stages.iter_mut().zip(ready) {
            s.ready = r;
        }
        out
    }

    fn measure_groups_now(&mut self, parts: &[StepPart]) -> Result<Vec<usize>, ModelError> {
        self.wait_commit();
        let blocks = self.fake_blocks(parts);
        let toks = vec![0u32; parts.iter().map(|p| p.n_tok).max().unwrap_or(0)];
        let groups = fake_groups(parts, &blocks, &toks);
        let table = self
            .model
            .paged_for(&self.stages[0].kv, &groups)
            .then_some(self.max_seq_blocks);
        let mut shape = with_kv(
            batch_shape(&groups, table, Some(&*self.model.hp.arch)),
            &groups,
            &self.stages[0].kv,
        );
        // GLM's parallel steps: every GPU's graph is its share of a parallel step's.
        for g in &mut shape {
            g.par = self.par.every_step;
        }
        let mut out = Vec::new();
        for s in 0..self.stages.len() {
            // Built only: making it ready would add per-device views a measurement never computes away.
            self.build_kept(s, &shape)?;
            out.push(
                self.stages[s]
                    .kept(&shape)
                    .map_or(0, |k| k.allocator_size()),
            );
            self.stages[s].forget_graphs();
        }
        Ok(out)
    }

    /// Bytes the MTP head's compute buffer takes for a pass of `n_seqs` sequences of `n_tok`
    /// tokens each after `n_past` (as [`Self::mtp_step`] builds it), measured by building and
    /// allocating that graph (then dropped). Needs [`Self::enable_mtp`].
    pub fn measure_mtp(
        &mut self,
        n_seqs: usize,
        n_tok: usize,
        n_past: usize,
    ) -> Result<usize, ModelError> {
        self.wait_commit();
        let parts = [StepPart {
            n_seqs,
            n_tok,
            n_past,
            verify: false,
        }];
        let blocks = self.fake_blocks(&parts);
        let toks = vec![0u32; n_tok];
        let groups = fake_groups(&parts, &blocks, &toks);
        let shape = self.mtp_shape(&groups);
        let stage = self.stages.last().expect("at least one stage");
        let kv = match &self.mtp_kv {
            _ if self.model.hp.arch.mtp_shares_kv() => &stage.kv,
            Some(kv) => kv,
            None => panic!("measure_mtp without Executor::enable_mtp"),
        };
        let size = graph_size(shape.len());
        let ctx = Context::new(size, size);
        let (graph, _) = self.model.build_mtp(&ctx, kv, &stage.rec, &shape)?;
        ctx.alloc_graph(&graph, ComputeAllocator::new(stage.backend.buffer_type()))?;
        Ok(ctx.allocator_size())
    }

    /// Block lists for [`fake_groups`]: `ceil((n_past + n_tok) / KV_BLOCK)` blocks per
    /// sequence, cycling through the pool (only shapes matter); in slots, every sequence reads
    /// the first slot's blocks (whose span the pool must hold, see [`Self::set_kv_slot_blocks`]).
    fn fake_blocks(&self, parts: &[StepPart]) -> Vec<Vec<Vec<u32>>> {
        if self.kv_slot_blocks > 0 {
            return parts
                .iter()
                .map(|p| {
                    let n_blocks = (p.n_past + p.n_tok).div_ceil(KV_BLOCK) as u32;
                    vec![(0..n_blocks).collect(); p.n_seqs]
                })
                .collect();
        }
        let mut next = 0usize;
        parts
            .iter()
            .map(|p| {
                let n_blocks = (p.n_past + p.n_tok).div_ceil(KV_BLOCK);
                (0..p.n_seqs)
                    .map(|_| {
                        (0..n_blocks)
                            .map(|_| {
                                next += 1;
                                ((next - 1) % self.kv_blocks()) as u32
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// The MTP pass shape of `groups` (see [`Self::mtp_step`]).
    /// Whether the MTP head's attention pages (else it gathers).
    pub fn mtp_paged(&self) -> bool {
        // The head's (dense) attention pages where the kernels take its shape: the gather
        // path's scores grow with tokens x span.
        // A head that reads the target's cache pages with it (a ring is only read in place).
        if self.model.hp.arch.mtp_shares_kv() {
            self.model.paged_default(&self.last().kv)
        } else if self.model.hp.arch.mtp_sparse() {
            // A sparse head reads its cache as the target's sparse layers do.
            self.mtp_kv
                .as_ref()
                .is_some_and(|kv| self.model.paged_for(kv, &[]))
        } else {
            self.mtp_kv
                .as_ref()
                .is_some_and(|kv| self.model.paged_default(kv))
        }
    }

    /// The MTP head's attention window from now on (see [`Model::set_mtp_window`]).
    pub fn set_mtp_window(&mut self, n: usize) {
        self.mtp_decode.clear();
        self.mtp_mixed.clear();
        self.par_mtp
            .iter_mut()
            .for_each(par_mtp::ParMtpGraphs::clear);
        self.model.set_mtp_window(n);
    }

    fn mtp_shape(&self, groups: &[Group<'_>]) -> Vec<GroupShape> {
        let shares = self.model.hp.arch.mtp_shares_kv();
        let paged = self.mtp_paged();
        // A sparse head's groups list their key pools as the target's do.
        let arch = &*self.model.hp.arch;
        let sparse = arch.mtp_sparse().then_some(arch);
        let shape = batch_shape(groups, paged.then_some(self.max_seq_blocks), sparse);
        if shares {
            with_kv(shape, groups, &self.last().kv)
        } else {
            shape
        }
    }

    /// Kept-graph re-initializations on the stages' meta backends so far, and the tensors they
    /// initialized again (see [`crate::ggml::Backend::ready_kept`]).
    pub fn reinit_counts(&self) -> (u64, u64) {
        self.stages.iter().fold((0, 0), |(a, b), s| {
            let (c, n) = s.backend.reinit_counts();
            (a + c, b + n)
        })
    }

    /// Bytes the kept compute buffers of stage `s` and the MTP head's take now (for checking
    /// a measured budget against the steps run).
    pub fn compute_bytes(&self, s: usize) -> usize {
        let st = &self.stages[s];
        // A shared buffer once, else each kept graph's own.
        let steps: usize = if st.shared.iter().any(Option::is_some) {
            st.shared
                .iter()
                .flatten()
                .map(|a| a.lock().unwrap_or_else(|e| e.into_inner()).buffer_size())
                .sum()
        } else {
            st.decode
                .iter()
                .chain(&st.mixed)
                .map(|(_, k)| k.allocator_size())
                .sum()
        };
        let piped = st.piped_shared.as_ref().map_or(0, |a| {
            a.lock().unwrap_or_else(|e| e.into_inner()).buffer_size()
        });
        let mtp = if s + 1 == self.stages.len() {
            self.mtp_decode.buffer_size() + self.mtp_mixed.buffer_size()
        } else {
            0
        } + self
            .par_mtp
            .get(s)
            .map_or(0, par_mtp::ParMtpGraphs::buffer_size);
        // The spares kept through an expert-cache lend ([`spare`]).
        let spares = st.spare.as_ref().map_or(0, ComputeAllocator::buffer_size)
            + st.ready
                .iter()
                .map(ComputeAllocator::buffer_size)
                .sum::<usize>()
            + if s + 1 == self.stages.len() {
                self.mtp_spare
                    .as_ref()
                    .map_or(0, ComputeAllocator::buffer_size)
            } else {
                0
            };
        steps + piped + mtp + spares
    }

    pub fn slots(&self) -> usize {
        self.stages[0].rec.n_slots()
    }

    pub fn saved_slots(&self) -> usize {
        self.stages[0].saved.n_slots()
    }

    /// Moves every stage's saved states ([`Pool::Saved`], prefix caching's) into pinned host
    /// memory (#441), empty: a save or a resume then copies one state over the host link, and
    /// the VRAM they took is free for what is sized after them (the expert cache's slots). A
    /// stage whose device has no pinned host memory keeps its pool. Returns the bytes moved per
    /// stage.
    pub fn saved_states_to_host(&mut self) -> Result<Vec<usize>, ModelError> {
        self.wait_commit();
        // GLM's parallel steps keep each GPU's share of the states ([`Executor::new_glm_par`]):
        // the host pool takes the same shapes, or a save or resume copies across mismatched
        // pools.
        let shard = self
            .par
            .every_step
            .then(|| crate::model::glm5next::par::shard_hparams(&self.model.hp, self.stages.len()));
        let hp = shard.as_ref().unwrap_or(&self.model.hp);
        let mut moved = Vec::with_capacity(self.stages.len());
        for s in &mut self.stages {
            let host = match s.backend.device().host_buffer_type() {
                Some(buft) if !s.saved.is_host() => buft,
                _ => {
                    moved.push(0);
                    continue;
                }
            };
            let n = s.saved.n_slots();
            s.saved = StatePool::new_layers(hp, host, n, s.layers.clone())?;
            moved.push(s.saved.bytes());
        }
        Ok(moved)
    }

    /// Caps a sequence at `tokens` positions (default: the whole pool), which sizes paged
    /// attention's block tables. Steps with a longer sequence panic.
    pub fn set_max_seq_tokens(&mut self, tokens: usize) {
        self.max_seq_blocks = tokens.div_ceil(KV_BLOCK).clamp(1, self.kv_blocks());
    }

    /// Switches how attention reads the KV cache (see [`super::Attention`]); the next steps
    /// build new graphs.
    pub fn set_attention(&mut self, a: super::Attention) {
        for s in &mut self.stages {
            s.forget_graphs();
        }
        self.model.set_attention(a);
    }

    /// Whether steps use paged attention.
    pub fn paged(&self) -> bool {
        self.model.paged(&self.stages[0].kv)
    }

    /// Whether decode and verify steps page their attention (for a sparse-attention model, its
    /// paged sparse op: [`Model::paged_for`] rather than [`Self::paged`]).
    pub fn steps_paged(&self) -> bool {
        self.model.paged_for(&self.stages[0].kv, &[])
    }

    /// Keeps the commits of replay pools made from now on ([`Self::enable_spec`]) in the commit,
    /// not deferred into the next verify step: for an executor that will lead other machines
    /// (tensor parallelism, a split over two nodes), whose commits [`Self::commit`] does not
    /// defer, or that runs another machine's steps.
    pub fn no_deferred_commits(&mut self) {
        self.no_defer = true;
    }

    /// Allocates replay buffers for speculative verify groups ([`Group::verify`]) of up to
    /// `max_tokens` tokens per sequence (drafts + 1), for every running-state slot.
    pub fn enable_spec(&mut self, max_tokens: usize) -> Result<(), ModelError> {
        // The old pool's deferred commits run before it goes.
        self.flush_commits()?;
        self.wait_commit();
        // GLM's parallel steps: every GPU stores its heads' replay inputs and replays through
        // its shards ([`Executor::new_glm_par`]).
        let shard = self
            .par
            .every_step
            .then(|| crate::model::glm5next::par::shard_hparams(&self.model.hp, self.stages.len()));
        let hp = shard.as_ref().unwrap_or(&self.model.hp);
        // Pools defer their commits only where `commit` does (#836 part 38): their inputs are
        // then stored packed for the replay alone.
        let defers = self.defers_commits() && !self.no_defer;
        for (d, s) in self.stages.iter_mut().enumerate() {
            // Kept graphs may reference the pool being replaced.
            s.forget_graphs();
            let buft = s.backend.buffer_type();
            let may_defer = defers && !buft.is_host() && !s.backend.is_meta();
            // The old pool goes first: two at once may not fit.
            s.spec = None;
            let mut pool = SpecPool::new_layers(
                hp,
                buft,
                s.rec.n_slots(),
                max_tokens,
                s.layers.clone(),
                may_defer,
            )?;
            if shard.is_some() {
                pool.set_shard(d);
            }
            s.spec = Some(pool);
        }
        Ok(())
    }

    /// Makes every step return the outputs of `layers` for all its tokens
    /// ([`StepOutput::hidden`]), for drafters that read the target's hidden states. The last
    /// layer's output is the hidden state before the final norm.
    pub fn capture_hidden(&mut self, layers: Vec<usize>) {
        assert!(
            layers.iter().all(|&il| il < self.model.hp.n_layer),
            "capture of a layer the model does not have"
        );
        self.hidden = layers;
        for s in &mut self.stages {
            s.forget_graphs();
        }
    }

    /// Hands a step's captured rows ([`StepOutput::hidden`]) back once read: the next step
    /// copies its rows into one of them instead of a new vector (#363).
    pub fn recycle_rows(&mut self, rows: Vec<Vec<f32>>) {
        self.tp_rows_gone();
        self.spare_rows.keep(rows, BOUNCE_MIN_BYTES);
    }

    /// A step's captured rows the engine keeps past its step (a prompt chunk's take-in left for
    /// the next step, #477), instead of handing them back: with `--tp-mtp-local-rows` (#780)
    /// the next step's MTP passes may name them too, and they go back to the spare rows after
    /// that.
    pub fn hold_rows(&mut self, rows: std::sync::Arc<Vec<Vec<f32>>>) {
        self.tp_hold_rows(rows);
    }

    /// The type [`Self::enable_mtp`] copies the head's wide matrices to (Q8_0 or F16), if any
    /// (`--spec-draft-head-type`).
    pub fn set_mtp_head_copy(&mut self, ty: Option<crate::ggml::ffi::ggml_type>) {
        self.mtp_head_copy = ty;
    }

    /// The output head [`Self::enable_mtp`] has the MTP head draft with: the model's, or a
    /// 4-bit copy of its draft rows (`--spec-draft-lm-head`, [`Model::quantize_draft_head`]).
    pub fn set_mtp_lm_head(&mut self, h: crate::spec::DraftLmHead) {
        self.mtp_lm_head = h;
    }

    /// Allocates the MTP head's KV cache (the model must have the head, see
    /// [`Model::has_mtp`]), laid out like the target's: same block ids.
    pub fn enable_mtp(&mut self) -> Result<(), ModelError> {
        assert!(self.model.has_mtp(), "the model has no MTP head");
        self.mtp_enabled = true;
        // The head's wide matrices requantized, so each draft pass reads fewer bytes (drafts
        // only: the output stays exact).
        if let Some(ty) = self.mtp_head_copy {
            let t = std::time::Instant::now();
            let (n, before, after) = self.model.requant_mtp(ty)?;
            tracing::info!(
                matrices = n,
                mib_before = before,
                mib_after = after,
                seconds = t.elapsed().as_secs_f64(),
                "MTP head requantized to {}",
                crate::ggml::type_name(ty).to_uppercase()
            );
        }
        if self.mtp_lm_head == crate::spec::DraftLmHead::Q4 {
            let t = std::time::Instant::now();
            let (rows, before, after) = self.model.quantize_draft_head()?;
            tracing::info!(
                rows,
                mib_before = before,
                mib_after = after,
                seconds = t.elapsed().as_secs_f64(),
                "MTP drafts with a 4-bit copy of the output head"
            );
        }
        if self.model.hp.arch.mtp_shares_kv() {
            // The head reads the target's cache.
            return Ok(());
        }
        let buft = self.last().backend.buffer_type();
        let hp = &self.model.hp;
        let types = crate::model::core::paged::mtp_head_types(
            self.stages[0].kv.types(),
            !buft.is_host(),
            hp.arch.index_dim() > 0,
        );
        self.mtp_kv = Some(KvPool::new_ring(
            hp,
            buft,
            self.kv_blocks(),
            &[hp.n_layer],
            None,
            types,
        )?);
        Ok(())
    }

    /// MTP pass graphs built so far (both kinds): a pass whose shape is kept reuses its graph.
    pub fn mtp_graph_builds(&self) -> u64 {
        self.mtp_decode.builds() + self.mtp_mixed.builds()
    }

    /// Whether the MTP head reads the target's KV cache (and drafts at the position after the
    /// target's last, see [`crate::model::core::Arch::mtp_shares_kv`]).
    pub fn mtp_shares_kv(&self) -> bool {
        self.model.hp.arch.mtp_shares_kv()
    }

    /// The width of the residual rows an MTP pass reads and returns.
    pub fn mtp_width(&self) -> usize {
        self.model.mtp_width()
    }

    /// One MTP pass (see `mtp.rs`): every group's tokens through the head, each paired with its
    /// row of `h` (`[tokens][mtp_width]`, in the order of groups, sequences, tokens), K/V
    /// written to the head's cache at the sequences' cells (unless the head reads the target's
    /// cache, [`Self::mtp_shares_kv`]). Sequences with [`Logits::Last`]
    /// get the head's greedy pick, its probability and the output residual. Over the gather
    /// budget ([`Self::set_gather_budget`]) it runs as several passes.
    pub fn mtp_step(&mut self, groups: &[Group<'_>], h: &[f32]) -> Result<MtpOutput, ModelError> {
        self.mtp_step_parts(groups, &[h])
    }

    /// [`Self::mtp_step`] with `h` given as `parts`, one after another: a take-in piece's rows
    /// (the previous piece's last row, then a range of the step's output) go straight into the
    /// pinned buffer the head's input is set through, not first into a new vector (#435).
    pub fn mtp_step_parts(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
    ) -> Result<MtpOutput, ModelError> {
        if self.tp.is_some() {
            return self.tp_mtp(groups, parts);
        }
        self.mtp_step_here(groups, parts)
    }

    /// [`Self::mtp_step_parts`] on this machine.
    fn mtp_step_here(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
    ) -> Result<MtpOutput, ModelError> {
        self.mtp_step_ahead(groups, parts, Ahead::default())
    }

    /// [`Self::mtp_step_here`] staging the pass after it, or on staged inputs
    /// ([`mtp_ahead`]), if it runs as one pass.
    fn mtp_step_ahead(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
        ahead: Ahead<'_, '_>,
    ) -> Result<MtpOutput, ModelError> {
        let chunks = self.budget_chunks(groups);
        if chunks.len() < 2 {
            return self.mtp_pass(groups, parts, ahead);
        }
        // Over the gather budget: one pass per chunk, the rows one after another.
        let joined;
        let h = match parts {
            [one] => *one,
            _ => {
                joined = parts.concat();
                &joined[..]
            }
        };
        let w = self.mtp_width();
        let mut out = MtpOutput::default();
        let mut at = 0;
        for c in &chunks {
            let rows: usize = c.iter().flat_map(|g| &g.seqs).map(|s| s.tokens.len()).sum();
            let o = self.mtp_pass(c, &[&h[at * w..(at + rows) * w]], Ahead::default())?;
            at += rows;
            out.argmax.extend(o.argmax);
            out.pmax.extend(o.pmax);
            out.hidden.extend(o.hidden);
        }
        Ok(out)
    }

    /// The keys a gather step reads for a sequence of `end` positions (its span, padded as
    /// its graph is built): the gather budget ([`Self::set_gather_budget`]) that one sequence
    /// of `end` positions fits.
    pub fn gather_span(end: usize) -> usize {
        end.div_ceil(super::ATTN_KV_PAD as usize) * super::ATTN_KV_PAD as usize
    }

    /// Keys a step of `groups` gathers under gather attention: per group, its sequences times
    /// its span (padded, as its graph is built).
    fn gathered(&self, groups: &[Group<'_>]) -> usize {
        batch_shape(groups, None, Some(&*self.model.hp.arch))
            .iter()
            .map(|g| g.n_seqs * g.n_kv)
            .sum()
    }

    /// `groups` as steps that each gather at most the budget ([`Self::set_gather_budget`]):
    /// `[groups]` itself when within it (or without a budget), else each group's sequences in
    /// consecutive runs (their slots consecutive too), as many per run as fit; a sequence over
    /// the budget alone runs alone.
    fn budget_chunks<'g>(&self, groups: &[Group<'g>]) -> Vec<Vec<Group<'g>>> {
        let budget = self.gather_budget;
        if budget == 0 || self.gathered(groups) <= budget {
            return vec![groups.to_vec()];
        }
        let mut out = Vec::new();
        for g in groups {
            let mut start = 0;
            while start < g.seqs.len() {
                let mut end = start + 1;
                while end < g.seqs.len() {
                    let run = Group {
                        slot: g.slot + start,
                        seqs: g.seqs[start..=end].to_vec(),
                        verify: g.verify,
                    };
                    if self.gathered(std::slice::from_ref(&run)) > budget {
                        break;
                    }
                    end += 1;
                }
                out.push(vec![Group {
                    slot: g.slot + start,
                    seqs: g.seqs[start..end].to_vec(),
                    verify: g.verify,
                }]);
                start = end;
            }
        }
        out
    }

    /// Caps the keys one step gathers under gather attention (the sum over its groups of
    /// sequences × span, see [`Self::step`] and [`Self::mtp_step`]): a step over it runs as
    /// several, each within it. The gathered keys and their masks are most of a gather step's
    /// compute buffer, and a batch's span is its longest sequence's: without a cap, four
    /// sequences batched with one long one gather four long contexts. 0 (the default) for none.
    pub fn set_gather_budget(&mut self, keys: usize) {
        self.gather_budget = keys;
    }

    /// Makes the stages' KV pools sequence slots of `n` blocks each from the next
    /// [`Self::resize_kv`] on (0: none): gather attention then reads each sequence's slot in
    /// place ([`KvPool::slot_span`]), with no gathered copy, and the scheduler must give the
    /// `i`-th block of the sequence in slot `s` as block `s × n + i`
    /// ([`crate::kv::BlockPool::slotted`]). A pool smaller than one slot only serves
    /// measurements (whose sequences all read the first slot).
    pub fn set_kv_slot_blocks(&mut self, n: usize) {
        self.kv_slot_blocks = n;
    }

    /// Prompt steps of at least `rows` query rows per sequence copy each sequence slot's span
    /// into the compute buffers before attention when it reaches past the slot's first `from`
    /// tokens and is at most `upto` long (#1153, [`Model::set_slot_stage`]); `rows` 0: never.
    pub fn set_kv_slot_stage(&mut self, rows: usize, from: usize, upto: usize) {
        self.model.set_slot_stage(rows, from, upto);
    }

    /// [`Self::set_kv_slot_stage`]'s `(rows, from, upto)`.
    pub fn kv_slot_stage(&self) -> (usize, usize, usize) {
        self.model.slot_stage()
    }

    /// Whether the model's attention reads sequence slots ([`Self::set_kv_slot_blocks`]).
    pub fn reads_kv_slots(&self) -> bool {
        self.model.hp.arch.reads_kv_slots()
    }

    /// Blocks per sequence slot of the KV pools (see [`Self::set_kv_slot_blocks`]).
    pub fn kv_slot_blocks(&self) -> usize {
        self.kv_slot_blocks
    }

    /// Sequence slots whose host tier is backed as they fill (#1131): the host holds `blocks`
    /// of them at once, not all slots' (`None`: all, mapped with the cache). Takes effect with
    /// the next tensor-parallel tier ([`Self::set_kv_tiers`]); the scheduler backs slots with
    /// [`Self::back_kv_slot`] and gives back idle ones with [`Self::trim_kv_slot`].
    pub fn set_kv_slot_host_budget(&mut self, blocks: Option<usize>) {
        self.kv_slot_host_budget = blocks;
    }

    /// See [`Self::set_kv_slot_host_budget`].
    pub fn kv_slot_host_budget(&self) -> Option<usize> {
        self.kv_slot_host_budget
    }

    /// Sequence slots that share `blocks` of VRAM (whole lazy chunks) past each slot's own
    /// first blocks (#1159): the slots' host tier is then backed as they fill, and a slot's
    /// chunks are VRAM while the shared ones last ([`Self::back_kv_slot`]'s `lent`). Takes
    /// effect with the next tensor-parallel tier ([`Self::set_kv_tiers`]); 0: none.
    /// `keep`: the VRAM per GPU the steps' buffers and their margin take, which
    /// [`Self::lend_kv_slots_more`] leaves free.
    pub fn set_kv_slot_lend(&mut self, blocks: usize, keep: usize) {
        self.kv_slot_lend = blocks;
        self.kv_slot_lend_keep = keep;
    }

    /// See [`Self::set_kv_slot_lend`].
    pub fn kv_slot_lend(&self) -> usize {
        self.kv_slot_lend
    }

    /// See [`Self::set_kv_slot_lend`].
    pub fn kv_slot_lend_keep(&self) -> usize {
        self.kv_slot_lend_keep
    }

    /// Adds up to `blocks` (whole lazy chunks) to the slots' shared VRAM (#1159), as far as
    /// the GPUs have room: the blocks added.
    pub fn lend_kv_slots_more(&mut self, blocks: usize) -> usize {
        let chunk = crate::engine::lm::LAZY_CHUNK_BLOCKS;
        let added = tier_lend(blocks / chunk) * chunk;
        self.kv_slot_lend += added;
        added
    }

    /// Backs sequence slot `slot`'s first `tokens` (rounded up to whole chunks) with host
    /// memory where its host tier has none yet ([`Self::set_kv_slot_host_budget`]), or with
    /// the slots' shared VRAM ([`Self::set_kv_slot_lend`]): `lent` tokens of the slot past its
    /// own VRAM, as far as the shared chunks go, its chunks moving (copied) between VRAM and
    /// the host to match. Only where the rows live changes.
    pub fn back_kv_slot(&self, slot: usize, tokens: usize, lent: usize) -> Result<(), String> {
        tier_back(slot, tokens, lent)
    }

    /// Gives back the host memory and shared VRAM [`Self::back_kv_slot`] gave slot `slot`,
    /// whose sequence left it (waits for the GPUs first).
    pub fn trim_kv_slot(&self, slot: usize) -> Result<(), String> {
        tier_trim(slot)
    }

    /// Whether stage `si`'s KV context spills into host memory (#1131).
    pub fn kv_spilled(&self, si: usize) -> bool {
        self.stages.get(si).is_some_and(|s| s.kv_spill.is_some())
    }

    /// Stage `si`'s KV context in buffer type `buft` from the next [`Self::resize_kv`] on
    /// (a decode state's spill with another share of VRAM, see [`Self::new_staged_spill`]).
    pub fn set_kv_spill(&mut self, si: usize, buft: BufferType) {
        if let Some(s) = self.stages.get_mut(si) {
            s.kv_spill = Some(buft);
        }
    }

    /// One MTP pass of `groups` ([`Self::mtp_step`] within the gather budget).
    fn mtp_pass(
        &mut self,
        groups: &[Group<'_>],
        h: &[&[f32]],
        ahead: Ahead<'_, '_>,
    ) -> Result<MtpOutput, ModelError> {
        let t = std::time::Instant::now();
        let ahead = Ahead {
            pipe: false,
            spec: false,
            ..ahead
        };
        match self.mtp_pass_fly(groups, h, ahead)? {
            Pass::Done(out) => Ok(out),
            Pass::Flying(f) => self.mtp_land(f, t),
            Pass::Skipped => unreachable!("only a pass on staged inputs is skipped"),
        }
    }

    /// [`Self::mtp_pass`], or, with `ahead.pipe`, a pass left running ([`mtp_pipe`]).
    fn mtp_pass_fly(
        &mut self,
        groups: &[Group<'_>],
        h: &[&[f32]],
        ahead: Ahead<'_, '_>,
    ) -> Result<Pass, ModelError> {
        let t = std::time::Instant::now();
        let out = self.mtp_pass_run(groups, h, ahead);
        // The commit launches it started (or, failing early, held): queued before anything
        // else uses their GPUs.
        self.par.join_held();
        let out = out?;
        self.par.launch_result()?;
        if self.par.every_step {
            self.par.gap.add(par_step::GapPhase::Mtp, t.elapsed());
        }
        if !matches!(out, Pass::Done(_)) {
            // Counted when it lands.
            return Ok(out);
        }
        // Model bytes (#391): the pass's graph, as the draft phase's.
        let shape = self.mtp_shape(groups);
        let slot = if shape.iter().all(|g| g.n_tok == 1) {
            &self.mtp_decode
        } else {
            &self.mtp_mixed
        };
        if let Some(kept) = slot.get(&shape) {
            self.bytes
                .draft
                .add(kept.model_bytes(), t.elapsed().as_secs_f64());
        }
        Ok(out)
    }

    /// [`Self::mtp_pass_fly`] itself.
    /// Queues the device's stop check of a pass's picks (`pmax`, F32) on `backend`'s stream
    /// (`crate::ggml::stage_graph::cond::decide`): the round's flag stays up if every row
    /// reached `p_min` and the round went on up to this pass, which it did if the pass ran
    /// unconditionally (not `behind` the flag). False where it was not queued (no CUDA).
    fn mtp_decide(backend: &Backend, pmax: Tensor<'_>, behind: bool, p_min: f32) -> bool {
        #[cfg(feature = "cuda")]
        {
            let Some((at, rows)) = mtp_ahead::f32_span(pmax) else {
                return false;
            };
            let Some(stream) = backend.cuda_launch_stream() else {
                return false;
            };
            // SAFETY: the backend's stream on its device (made current); `pmax` holds `rows`
            // F32 values there, written by the pass queued on it.
            unsafe { crate::ggml::stage_graph::cond::decide(stream, at, rows, p_min, !behind) }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (backend, pmax, behind, p_min);
            false
        }
    }

    fn mtp_pass_run(
        &mut self,
        groups: &[Group<'_>],
        h: &[&[f32]],
        ahead: Ahead<'_, '_>,
    ) -> Result<Pass, ModelError> {
        // What the last pass staged is this pass's or no one's (`mtp_ahead`).
        let staged = self.mtp_staged.take().filter(|_| ahead.staged);
        // Queued ahead (`ahead.spec`), the pass runs on the GPU only if the round went on up
        // to it: the device's flag, which the pass that staged it updated from its picks.
        let behind = ahead.spec && staged.as_ref().is_some_and(|s| s.decided);
        // A pass on staged inputs alone (`mtp_pipe`) runs only as the pass it follows left it:
        // no target work to take.
        if ahead.spec
            && (staged.is_none() || self.deferred_target.is_some() || self.deferred_tail.is_some())
        {
            return Ok(Pass::Skipped);
        }
        // `--glm-parallel-mtp`: a pass with output rows on every GPU.
        if self.mtp_runs_par(groups) {
            return self.mtp_pass_par(groups, h).map(Pass::Done);
        }
        // Under `--glm-mtp-overlap` the last GPU's commit is held too (`commit_par`): queued
        // before this pass uses its GPU.
        if self.mtp_overlap() {
            self.par.join_held();
            self.par.launch_result()?;
        }
        let t_pass = std::time::Instant::now();
        let shares = self.model.hp.arch.mtp_shares_kv();
        let shape = self.mtp_shape(groups);
        // A draft pass's new graphs are captured behind their first run (#827), a prompt's
        // take-in's not.
        let _early = crate::ggml::stage_graph::EarlyScope::enter(
            shape
                .iter()
                .all(|g| g.n_tok <= crate::ggml::model_bytes::DECODE_MAX_TOKENS),
        );
        // A take-in of several sequences is a shape that seldom comes back (its rows follow
        // each one's accepted count): captured only once it runs again (#836).
        let _reuse = crate::ggml::stage_graph::ReuseScope::enter(take_in_waits_for_reuse(&shape));
        // The pass to stage after this one, and its shape.
        let next = (ahead.next)
            .filter(|_| self.mtp_ahead())
            .map(|n| (n, self.mtp_shape(n)));
        let n_layer = self.model.hp.n_layer;
        // The previous pass's cache work touches this pass's layer: before it.
        self.flush_deferred_mtp()?;
        // The target step's cache work to run beside this pass: its first pass's share, or
        // the rest of it (see `mtp_window`).
        let target_work = self.take_target_work()?;
        let target = self.target_entries();
        self.begin_pass(
            shape.iter().map(|g| g.tokens()).sum(),
            |il| il == n_layer,
            Kind::Draft,
        )?;
        // Beside a round's other stages (#762) the peers are not idle.
        let alone = self.stages.len() > 1 && self.pipe_run.is_none();
        let last_stage = self.stages.len() - 1;
        // The other stages, for a verify commit held for this pass (`commit_hold`).
        let (front, back) = self.stages.split_at_mut(last_stage);
        let Stage {
            kv: stage_kv,
            rec,
            backend,
            bounce,
            mtp_io,
            spec: last_spec,
            commits: last_commits,
            ..
        } = &mut back[0];
        // A head that shares the target's cache reads the last stage's pool (it holds the
        // target layers the head reads, the last ones).
        let kv = match &self.mtp_kv {
            _ if shares => &*stage_kv,
            Some(kv) => kv,
            None => panic!("mtp_step without Executor::enable_mtp"),
        };
        let decode = shape.iter().all(|g| g.n_tok == 1);
        let slot = if decode {
            &mut self.mtp_decode
        } else {
            &mut self.mtp_mixed
        };
        // A shape kept from an earlier pass runs as it is (the take-in's rows change with the
        // accepted count, #289).
        let t_pre = t_pass.elapsed();
        let built = !slot.touch(&shape);
        if built && ahead.spec {
            return Ok(Pass::Skipped);
        }
        if built {
            let t_build = std::time::Instant::now();
            // `--stage-graph-stats`' build line, against what is kept before the eviction.
            let kind = if decode { "draft" } else { "take-in" };
            let note = crate::ggml::stage_graph::stats::logging().then(|| {
                shape_diff::note(
                    kind,
                    last_stage,
                    &shape,
                    slot.shapes(),
                    slot.len(),
                    slot.cap(),
                )
            });
            slot.evict_for_one();
            let before = slot.buffer_size();
            let size = graph_size(shape.len());
            let (ctx, parts) = loop {
                let ctx = Context::new(size, size);
                let parts = {
                    let (graph, g) = self.model.build_mtp(&ctx, kv, rec, &shape)?;
                    match slot.alloc(&ctx, &graph, backend.buffer_type()) {
                        // As for the step graphs: the graphs a restore hands its thread.
                        Err(GgmlError::Alloc { .. }) if self.freeing.wait() => continue,
                        r => r?,
                    }
                    let mut list = g.inputs.list();
                    list.push(g.h);
                    list.extend(g.pmax);
                    list.extend(g.h_out);
                    KeptParts::new(&graph, &list)
                };
                break (ctx, parts);
            };
            // SAFETY: as for the step graphs: the graph references the weights and the pool it
            // reads (the MTP head's or the last stage's), which only `resize_kv` replaces, and
            // it drops the kept MTP graphs first.
            let mut kept = unsafe { KeptGraph::new(ctx, parts) };
            crate::ggml::stage_graph::stats::note_build(kind, last_stage, || {
                note.unwrap_or_default()
            });
            Self::count_bytes(&mut self.weight_addrs, &self.model, &mut kept);
            tracing::trace!(
                shape = ?shape.iter().map(|g| (g.n_seqs, g.n_tok)).collect::<Vec<_>>(),
                kept = slot.len() + 1,
                ms = t_build.elapsed().as_secs_f64() * 1e3,
                "mtp graph built"
            );
            crate::ggml::stage_graph::stats::count_split(
                crate::ggml::stage_graph::stats::Split::BuildMtpNs,
                t_build.elapsed().as_nanos() as u64,
            );
            let after = slot.buffer_size();
            if after > before {
                tracing::debug!(
                    tokens = shape.iter().map(GroupShape::tokens).sum::<usize>(),
                    mib = after as f64 / (1u64 << 20) as f64,
                    "mtp compute buffer grew"
                );
            }
            slot.push(shape.clone(), kept);
        }
        let slot = if decode {
            &self.mtp_decode
        } else {
            &self.mtp_mixed
        };
        let t_built = t_pass.elapsed();
        let kept = slot.get(&shape).expect("kept above");
        // Inputs the last pass staged for this one ([`mtp_ahead`]) into this graph, kept since:
        // they hold what filling them would write.
        // A pass on staged inputs alone (`ahead.spec`) does not know its token ids: they are
        // the last pass's picks, copied on the device.
        let staged = !built && staged.is_some_and(|s| s.matches(&shape, groups, kept, ahead.spec));
        // On its own staged inputs only: a pass queued ahead runs on nothing else.
        let behind = behind && staged;
        if ahead.spec && !staged {
            return Ok(Pass::Skipped);
        }
        backend.ready_kept(kept)?;
        let swa = self.model.swa_window();
        // A sparse head's passes take the per-group inputs (`Arch::mtp_sparse`).
        let arch = &self.model.hp.arch;
        let n_group = if arch.mtp_sparse() {
            arch.group_inputs_len()
        } else {
            0
        };
        let mut inputs =
            Inputs::from_list(&shape, 0, false, n_group, false, swa, |i| kept.tensor(i));
        inputs.exclusive = !self.model.hp.arch.mtp_sees_own_position();
        let n_in = inputs.list().len();
        let n_out: usize = shape.iter().map(|g| g.out_rows()).sum();
        // The inputs and outputs as copies queued on the stream (`mtp_window`); the rows of a
        // big take-in through the bounce buffer, as a step's, held and copied by one kernel
        // launch before the graph (`pinned_io::pack`, #814). A pass queued behind one still
        // running (`ahead.spec`) goes on in its transfers: a new start would wait for it.
        if !ahead.spec {
            mtp_io.begin_packed(backend, mtp_window::PINNED_MAX);
        }
        if !staged {
            inputs.fill_into(
                groups,
                &shape,
                &mut mtp_window::Queued {
                    io: mtp_io,
                    backend,
                },
            );
            if n_group > 0 {
                self.model.fill_extra(
                    &inputs,
                    groups,
                    &shape,
                    rec,
                    kv,
                    None,
                    &mut mtp_window::Queued {
                        io: mtp_io,
                        backend,
                    },
                );
            }
            let h_in = kept.tensor(n_in);
            let h_bytes: usize = h.iter().map(|p| std::mem::size_of_val(*p)).sum();
            if h_bytes >= BOUNCE_MIN_BYTES {
                write_rows(bounce, backend, h_in, h);
            } else {
                mtp_io.write_f32_parts(backend, h_in, h);
            }
        }
        mtp_io.flush(backend);
        let t_filled = t_pass.elapsed();
        {
            // The head runs alone on the last stage's GPU: its NVLink peer is idle.
            let _idle = super::expert_mv::peer::Idle::new(alone);
            let _cond = crate::ggml::stage_graph::cond::CondScope::enter(behind);
            if let Err(e) = backend.compute_kept_async(kept) {
                mtp_io.finish(backend);
                return Err(e.into());
            }
        }
        let t_launched = t_pass.elapsed();
        // The other GPUs' commit launches, held for this (`commit_par`): beside the head.
        self.par.start_held();
        let reads = (n_out > 0).then(|| {
            let argmax = inputs.argmax.expect("argmax");
            [argmax, kept.tensor(n_in + 1), kept.tensor(n_in + 2)].map(|t| mtp_io.read(backend, t))
        });
        // The next pass's inputs, behind the reads (`mtp_ahead`): its graph built already, and
        // this pass's outputs read on the stream (not after it, over the copies).
        let mut staged_next = None;
        if let (Some((next, next_shape)), Some(r)) = (&next, &reads)
            && r.iter().all(|r| r.is_queued())
            && let Some(next_kept) = self.mtp_decode.get(next_shape)
        {
            // The next pass, queued ahead, runs behind this one's picks: their check goes on
            // the stream before its inputs, which may reuse their memory.
            let decided =
                ahead.pipe && Self::mtp_decide(backend, kept.tensor(n_in + 1), behind, ahead.p_min);
            staged_next = mtp_ahead::prestage(
                mtp_io,
                backend,
                inputs.argmax.expect("argmax"),
                kept.tensor(n_in + 2),
                NextPass {
                    groups: next,
                    shape: next_shape,
                    kept: next_kept,
                    swa,
                    exclusive: inputs.exclusive,
                },
            );
            if let Some(s) = &mut staged_next {
                s.decided = decided;
            }
        }
        // The target step's cache work while the head computes (it reads no target layer).
        let mut credit = None;
        if let Some((n, kind, layers, last)) = target_work {
            let ran = |il: usize| layers.contains(&il) && target(il);
            let r = match self.model.experts_mut() {
                Some(r) if last => r.after_step_only(n, self.pair.widen(ran)),
                Some(r) => r.after_step_more(n, self.pair.widen(ran)),
                None => Ok(()),
            };
            self.pair.learned(ran);
            credit = Some(kind);
            if let Err(e) = r {
                mtp_io.finish(backend);
                return Err(ModelError::Ggml(GgmlError::External(format!(
                    "expert cache: {e}"
                ))));
            }
        }
        // A verify commit held for this pass (`commit_hold`): the other stages' now, on their
        // idle GPUs, while the head computes; the last stage's behind this pass's reads, or
        // with `commit_last_late` behind the next pass's (`commit_hold::last_now`).
        let mut last_now = true;
        if !self.held_commit.is_empty() {
            let (stages, held) = self.held_commit.take(|s| s != last_stage);
            last_now = commit_hold::last_now(self.commit_last_late, !stages.is_empty());
            for s in stages {
                if let Err(e) = commit_hold::queue_stage(&mut front[s], &held, &self.model) {
                    mtp_io.finish(backend);
                    return Err(e.into());
                }
            }
        }
        let t_cache = t_pass.elapsed();
        let n = shape.iter().map(|g| g.tokens()).sum();
        // Left running (`mtp_pipe`): the next pass's inputs are staged behind its reads, so
        // the next pass can be queued before this one's picks reach the host.
        if ahead.pipe
            && staged_next.is_some()
            && let Some(fence) = mtp_io.fence(backend)
        {
            let reads = reads
                .expect("staged behind the reads")
                .map(|r| r.detach().expect("queued"));
            // A held commit of the last stage behind the reads (the landing waits on `fence`).
            if last_now && !self.held_commit.is_empty() {
                let (_, held) = self.held_commit.take(|_| true);
                if let Err(e) = commit_hold::queue(
                    last_spec.as_ref(),
                    backend,
                    last_commits,
                    rec,
                    &held,
                    &self.model,
                ) {
                    self.held_commit.fail(e);
                }
            }
            self.mtp_staged = staged_next;
            return Ok(Pass::Flying(InFlight {
                fence,
                reads,
                credit,
                n,
                shape,
            }));
        }
        // The last stage's held commit behind the reads, which the host waits for on an event:
        // the commit runs while the host takes the picks and readies the next pass.
        if last_now && !self.held_commit.is_empty() {
            let (_, held) = self.held_commit.take(|_| true);
            let fence = mtp_io.fence(backend);
            if fence.is_none() {
                mtp_io.finish(backend);
            }
            let queued = commit_hold::queue(
                last_spec.as_ref(),
                backend,
                last_commits,
                rec,
                &held,
                &self.model,
            );
            if let Some(f) = fence {
                mtp_io.wait(f);
            }
            if let Err(e) = queued {
                // The pass's outputs stand; the error goes to the next step or commit.
                self.held_commit.fail(e);
            }
        } else {
            mtp_io.finish(backend);
        }
        let t_wait = t_pass.elapsed();
        self.front_clock.picked();
        // The pass's host time by phase (#836): the cache work before it, its graph's build
        // (`fresh`: a shape not kept), the input fill, the launch, the target's cache work
        // beside it, and the wait for its picks. The wait is GPU time: the pass's own, plus
        // whatever the stream still had queued before it (a step's first pass waits for the
        // verify step's commit, ~14 ms at 16 sequences on a GB10).
        tracing::debug!(
            groups = shape.len(),
            tokens = shape.iter().map(GroupShape::tokens).sum::<usize>(),
            fresh = built,
            pre = t_pre.as_secs_f64() * 1e3,
            build = (t_built - t_pre).as_secs_f64() * 1e3,
            fill = (t_filled - t_built).as_secs_f64() * 1e3,
            launch = (t_launched - t_filled).as_secs_f64() * 1e3,
            cache = (t_cache - t_launched).as_secs_f64() * 1e3,
            wait = (t_wait - t_cache).as_secs_f64() * 1e3,
            "mtp pass ms"
        );
        let mut out = MtpOutput::default();
        if let Some([argmax, pmax, hidden]) = reads {
            out.argmax = (mtp_io.take_i32(argmax).into_iter())
                .map(|x| x as u32)
                .collect();
            out.pmax = mtp_io.take_f32(pmax);
            out.hidden = mtp_io.take_f32(hidden);
        }
        if let Some(mut s) = staged_next {
            s.tokens.clone_from(&out.argmax);
            self.mtp_staged = Some(s);
        }
        self.mtp_pass_after(credit, n)?;
        self.front_clock.returned();
        Ok(Pass::Done(out))
    }

    /// The end of an MTP pass of `n` tokens, after its reads: the cache work it took (as
    /// `credit`'s kind of step) and its own.
    fn mtp_pass_after(&mut self, credit: Option<Kind>, n: usize) -> Result<(), ModelError> {
        // The target step's cache work above, as its kind of step.
        if let Some(kind) = credit {
            self.credit_experts(kind);
        }
        let n_layer = self.model.hp.n_layer;
        if self.after_overlap {
            // Run beside the next target step (or before the next pass).
            self.deferred_mtp = Some(n);
        } else {
            self.after_pass(n, |il| il == n_layer, Kind::Draft)?;
        }
        Ok(())
    }

    /// After a step with a verify group: advances each `(slot, keep)` state by the first
    /// `keep` tokens that step fed it (the tokens accepted). Every slot of the verify group
    /// must be committed before its state is used again.
    ///
    /// Returns once the commit graphs are started on every stage, so the stages commit at the
    /// same time and the host goes on to the MTP drafts; steps and MTP passes follow on the
    /// same streams, and [`Self::wait_commit`] waits for it. Under parallel steps the other
    /// GPUs' graphs start beside the next MTP pass instead (`commit_par`). Adapted from
    /// Strata-AC922's async commit on a layer split (eelgaev/Strata-AC922 d775be3, MIT).
    pub fn commit(&mut self, commits: &[(usize, usize)]) -> Result<(), ModelError> {
        let model = &self.model;
        // Without recurrent layers a verify step has nothing to replay: the KV rows past the
        // kept tokens are simply never read again.
        if !model.hp.has_slot_state() {
            return Ok(());
        }
        let wire = || crate::node::Request::TpCommit {
            commits: commits
                .iter()
                .map(|&(slot, keep)| (slot as u32, keep as u32))
                .collect(),
        };
        // Rank 1 of a split model commits its layers' states the same way (#560).
        if let Some(r) = self.remote.as_mut() {
            r.mirror(wire());
        }
        self.tp_mirror_states(wire);
        // Not waited for: every stage's commit runs at once on its own device, and the next
        // step (or MTP pass) runs after it on the same stream. Anything that touches the states
        // another way waits first (`wait_commit`). Pending from the first launch on, so a later
        // stage's failure still leaves the earlier stages' graphs waited for.
        self.commit_pending = true;
        let t = std::time::Instant::now();
        if self.par.every_step && self.par.pool.is_some() {
            let res = self.commit_par(commits);
            self.par.gap.add(par_step::GapPhase::Commit, t.elapsed());
            return res;
        }
        // A commit still held from the step before goes first (its stages' order).
        if let Err(e) = self.queue_all_held() {
            self.wait_commit();
            return Err(e);
        }
        // A layer split with MTP drafting: queued beside the first draft pass (`commit_hold`).
        if self.holds_commit() {
            self.hold_commit(commits);
            return Ok(());
        }
        let model = &self.model;
        if self.defers_commits() {
            let s = &mut self.stages[0];
            let spec = s
                .spec
                .as_ref()
                .expect("commit without Executor::enable_spec");
            // SAFETY: as below; the deferred graphs read the stage's pools only.
            match unsafe { spec.commit_deferred(&mut s.backend, &mut s.commits, &s.rec, commits) } {
                Ok(true) => return Ok(()),
                Ok(false) => {}
                Err(e) => {
                    self.wait_commit();
                    return Err(e.into());
                }
            }
        }
        let mut res = Ok(());
        for (i, s) in self.stages.iter_mut().enumerate() {
            // A stage still computing a round's piece (#762): its commit starts once it is done
            // (`Executor::pipe_round_end`), before its next piece. The piece reads other slots.
            if let Some(run) = &mut self.pipe_run
                && run.hold_commit(i, commits)
            {
                continue;
            }
            let spec = s
                .spec
                .as_ref()
                .expect("commit without Executor::enable_spec");
            // SAFETY: the stage's kept commit graphs are dropped before its pools (field order,
            // and `resize_kv` / `enable_spec` clear them) and the model's weights.
            res = unsafe { spec.commit(&mut s.backend, &mut s.commits, &s.rec, commits, model) };
            if res.is_err() {
                break;
            }
        }
        if res.is_err() {
            self.wait_commit();
        }
        if self.par.every_step {
            self.par.gap.add(par_step::GapPhase::Commit, t.elapsed());
        }
        res.map_err(Into::into)
    }

    /// Whether [`Self::commit`] may defer the delta rule into the next verify step
    /// (`--gdn-commit-in-verify`, #836 part 14): one stage on this machine, no parallel or
    /// pipelined steps (their commits run elsewhere or later).
    fn defers_commits(&self) -> bool {
        self.stages.len() == 1
            && self.tp.is_none()
            && self.remote.is_none()
            && self.par.pool.is_none()
            && !self.par.every_step
            && self.pipe_run.is_none()
    }

    /// Runs every deferred commit's delta rule now (queued, waited for as a commit): before
    /// anything but a verify step of their slots reads or writes the states.
    fn flush_commits(&mut self) -> Result<(), GgmlError> {
        for s in &mut self.stages {
            let Some(spec) = s.spec.as_ref().filter(|p| p.defers_any()) else {
                continue;
            };
            self.commit_pending = true;
            // SAFETY: as in `commit`.
            let res = unsafe { spec.flush(&mut s.backend, &mut s.commits, &s.rec) };
            if let Err(e) = res {
                self.wait_commit();
                return Err(e);
            }
        }
        Ok(())
    }

    /// Before a step's graphs are queued: deferred commits of the slots a non-verify group
    /// runs are flushed, and each verify group's deferred counts are made the device's (its
    /// delta rule replays them).
    fn deferred_before_step(&mut self, groups: &[Group<'_>]) -> Result<(), ModelError> {
        let range = |g: &Group<'_>| g.slot..g.slot + g.seqs.len();
        // Not `defers_any`: with no commit waiting, the device's counts may still hold those a
        // verify step, a flush or a reset slot left there, and a verify step must not replay
        // them again (`before_verify` uploads zeros then).
        let defers = |s: &Stage| s.spec.as_ref().is_some_and(|p| p.defers());
        if !self.stages.iter().any(defers) {
            return Ok(());
        }
        // A step with no token to run runs no graph (`step_here`): its verify groups must not
        // count their slots' deferred commits as replayed.
        if groups
            .iter()
            .all(|g| g.seqs.iter().all(|s| s.tokens.is_empty()))
        {
            return Ok(());
        }
        let touched = self.stages.iter().any(|s| {
            s.spec
                .as_ref()
                .is_some_and(|p| groups.iter().any(|g| !g.verify && p.defers_slots(range(g))))
        });
        if touched {
            self.flush_commits()?;
        }
        for s in &mut self.stages {
            if let Some(p) = s.spec.as_ref() {
                for g in groups.iter().filter(|g| g.verify) {
                    let n_tok = g.seqs.first().map_or(0, |q| q.tokens.len());
                    p.before_verify(&mut s.backend, range(g), n_tok);
                }
            }
        }
        Ok(())
    }

    /// Whether a [`Self::commit`] may still be running on the stages (not waited for yet).
    pub fn commit_in_flight(&self) -> bool {
        self.commit_pending
    }

    /// Host access to the recurrent states: only after the last commit was waited for.
    fn assert_committed(&self) {
        debug_assert!(
            !self.commit_pending,
            "recurrent states touched while a commit may still be running (missing wait_commit)"
        );
    }

    /// Waits for the stages' last [`Self::commit`], if one is still running.
    pub fn wait_commit(&mut self) {
        // A round's stages still computing (#762): done first, their held commits started.
        self.pipe_settle();
        // The other GPUs' launches held for the MTP pass (`commit_par`): queued first. An
        // error of theirs goes to the next step.
        self.par.join_held();
        // A layer split's commit held for the next MTP pass (`commit_hold`): queued first. An
        // error goes to the next step or commit.
        self.queue_all_held_later();
        if std::mem::take(&mut self.commit_pending) {
            for s in &mut self.stages {
                s.backend.synchronize();
            }
        }
    }

    /// Prompt tokens per micro-chunk when several stages pipeline a prefill step: a single
    /// prompt chunk longer than this runs as consecutive micro-chunks, each stage working on
    /// the next one while the stage after it works on the previous one (see [`Self::step`]).
    /// Every micro-chunk reads all the experts it routes to, so this should be what one step
    /// would take without stages. 0 turns pipelining off.
    pub fn set_pipeline_micro(&mut self, tokens: usize) {
        self.pipeline_micro = tokens;
    }

    /// Pipelines a step whose last group is one prompt chunk longer than the micro-chunk and
    /// whose other groups (decoding sequences, other prompts' chunks: the riders) come before
    /// it: the riders run as a micro-chunk of their own ahead of the prompt's micro-chunks
    /// (see [`Self::step`]), so no graph is bigger than the riders or a micro-chunk. Without
    /// it such a step is one graph of all its tokens (#707: an 8K prompt joining a decoding
    /// sequence over the lent expert cache needed a 3.3 GB compute buffer, which spilled into
    /// host memory). Rows stay in the order of groups.
    pub fn set_pipeline_riders(&mut self, on: bool) {
        self.pipeline_riders = on;
    }

    /// The last micro-chunks of a pipelined step ramp down to `tokens`
    /// ([`taper::micro_sizes`]), so the step's last micro-chunk passes the later stages
    /// sooner. 0: no taper. Other chunk boundaries change the float summation order.
    pub fn set_micro_taper(&mut self, tokens: usize) {
        self.micro_taper = tokens;
    }

    /// The pipeline stages (one per device).
    /// See the `ple_ahead` field.
    #[cfg(test)]
    pub(crate) fn set_ple_ahead(&mut self, on: bool) {
        self.ple_ahead = on;
    }

    /// Fills that wrote n-gram rows gathered ahead ([`Model::gather_ple_ahead`]).
    #[cfg(test)]
    pub(crate) fn ple_ahead_hits(&self) -> u64 {
        self.model.ple_ahead_hits()
    }

    /// The n-gram rows (qwen4exp's per-layer embedding) of the micro-chunk the next round
    /// `next` runs on the stage that reads them, gathered on the host now, while the current
    /// round computes: that stage's fill waits for its own graph, and on a long prompt it is
    /// the slowest stage, so the gather (~8 ms per 4,096 tokens on an AC922) used to sit between
    /// its graphs (#439). Only host work moves; the fill writes the same values.
    fn gather_ahead(&self, next: &[(usize, usize)], micros: &[Vec<Group<'_>>]) {
        if !self.ple_ahead {
            return;
        }
        if let Some(&(s, k)) = next.iter().find(|&&(s, _)| self.parts(s).ple) {
            self.model.gather_ple_ahead(&micros[k], &self.stages[s].rec);
        }
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// The CPUs each stage's launch thread runs on (its GPU's NUMA node, see
    /// [`crate::hw::numa::helper_cpus`]); a stage without a list keeps the inherited affinity.
    pub fn set_stage_cpus(&mut self, cpus: Vec<Vec<u32>>) {
        self.stage_cpus = cpus;
    }

    /// With several stages, splits a decode or verify batch of at least two sequences into
    /// one micro-batch per stage, run as a wavefront like a prompt's micro-chunks: stage 0
    /// computes the second half while stage 1 computes the first. Otherwise one stage idles
    /// while the other runs the whole batch.
    pub fn set_split_batches(&mut self, on: bool) {
        self.split_batches = on;
    }

    /// Runs a long prompt's micro-chunks without rounds while the expert cache is lent: each
    /// stage starts its next micro-chunk once what it waits on is done ([`dataflow`]).
    pub fn set_dataflow(&mut self, on: bool) {
        self.dataflow = on;
    }

    /// With [`Self::set_dataflow`] on CUDA: peer access between neighbouring stages' GPUs, so
    /// a residual's hand-off copy goes GPU to GPU instead of through host memory.
    pub fn set_dataflow_peer(&mut self, on: bool) {
        self.dataflow_peer = on;
    }

    /// With [`Self::set_dataflow`] on CUDA: each stage's residual goes to the next GPU on a
    /// copy stream behind an event, which the next stage's graph waits on (not the host), and
    /// a stage that reads nothing back queues its next micro-chunk while its graph runs
    /// ([`dataflow`]'s `hand_off`). The same graphs and bytes.
    pub fn set_dataflow_async(&mut self, on: bool) {
        self.dataflow_async = on;
    }

    /// Runs one batch step and returns its logits rows: every group's tokens through the model,
    /// K/V written to the sequences' blocks, recurrent states advanced in their slots. Rows come
    /// in the order of groups, sequences, tokens (see [`StepOutput`]). The caller keeps `n_past`
    /// and the block tables.
    ///
    /// With several stages, a step that is one prompt chunk longer than the pipeline
    /// micro-chunk ([`Self::set_pipeline_micro`]) runs as a wavefront: in round `r` stage `s`
    /// computes micro-chunk `r - s`, all stages at once on their own devices. With
    /// [`Self::set_pipeline_riders`], the groups before such a chunk run as the first
    /// micro-chunk.
    pub fn step(&mut self, groups: &[Group<'_>]) -> Result<StepOutput, ModelError> {
        self.applied.clear();
        if self.tp.is_some() {
            self.prompt_next = None;
            self.hidden_from = 0;
            self.hidden_on_device = None;
            self.captures.clear();
            return self.tp_step(groups);
        }
        let out = self.step_here(groups);
        self.hidden_from = 0;
        self.hidden_on_device = None;
        out
    }

    /// The next [`Self::step`] may leave its captured rows on the device
    /// ([`StepOutput::hidden_dev`], #1181) instead of reading them back: a drafter that gathers
    /// the rows it takes in on the device (DFlash) reads them there before anything runs on
    /// the executor again. Applied by a step of one micro-chunk on one stage on this machine
    /// whose captured layers are F32 matrices in a buffer of CUDA device `device` (the
    /// drafter's); any other step reads them back as without (`None`: always). Only the next
    /// step reads it.
    pub fn set_hidden_on_device(&mut self, device: Option<usize>) {
        self.hidden_on_device = device;
    }

    /// The stop tokens a greedy pick below `min_tokens` ([`Logits::Pick`],
    /// [`Logits::FirstPick`]) may not pick: the engine's.
    pub fn set_stop_tokens(&mut self, stop: &[u32]) {
        self.stop_tokens = stop.to_vec();
    }

    /// The next [`Self::step`] reads its captured rows ([`StepOutput::hidden`]) back from row
    /// `row` on only: the rows before it keep their length in the output but hold anything (an
    /// earlier step's rows, or zeros). For a drafter that reads only a prompt chunk's last rows
    /// (DFlash's ring, #1133); a step of one micro-chunk applies it, others read every row.
    /// Only the next step reads it.
    pub fn set_hidden_from(&mut self, row: usize) {
        self.hidden_from = row;
    }

    /// The last step's host laps (`--tp-profile`), if it ran as one round of one stage (a
    /// decode or verify step on a tensor-parallel rank); taken, so a later step without them
    /// gives `None`.
    pub fn take_step_laps(&mut self) -> Option<crate::tp_net::StepLaps> {
        self.step_laps.take()
    }

    /// [`Self::step`] on this machine.
    fn step_here(&mut self, groups: &[Group<'_>]) -> Result<StepOutput, ModelError> {
        debug_assert!(self.pipe.is_empty(), "a whole step with groups in flight");
        let t_entry = std::time::Instant::now();
        self.step_laps = None;
        // Read by this step only (`prompt_ahead`).
        let prompt_next = self.prompt_next.take();
        let mut front = self.front_clock.step();
        // The last verify step's commit, if no MTP pass queued it (`commit_hold`): before any
        // stage's graph.
        if let Err(e) = self.queue_all_held() {
            self.wait_commit();
            return Err(e);
        }
        front.lap("held");
        self.deferred_before_step(groups)?;
        front.lap("before_verify");
        if self.par_fits(groups) {
            self.captures.clear();
            return self.step_par(groups);
        }
        // No token to run (every sequence of the step dropped out): nothing to do.
        if groups
            .iter()
            .all(|g| g.seqs.iter().all(|s| s.tokens.is_empty()))
        {
            self.captures.clear();
            return Ok(StepOutput::default());
        }
        let table = self
            .model
            .paged_for(&self.stages[0].kv, groups)
            .then_some(self.max_seq_blocks);
        // A target step reads no MTP layer: the last MTP pass's deferred cache work runs once
        // its first stage is queued (below); anything else deferred runs now.
        if let Some((n, kind)) = self.deferred_target.take() {
            let n_layer = self.model.hp.n_layer;
            self.after_pass_only(n, |il| il < n_layer, kind)?;
        }
        // The last stage's deferred work (an MTP pass's, and the share of the last step's left
        // for the last stage's layers) runs beside the first stage.
        let overlap_mtp =
            (self.deferred_mtp.is_some() || self.deferred_tail.is_some()) && self.stages.len() > 1;
        if !overlap_mtp {
            self.flush_deferred_last()?;
        }
        front.lap("deferred");
        let arch = &*self.model.hp.arch;
        let micros = self.micro_chunks(groups);
        let mut shapes: Vec<Vec<GroupShape>> = micros
            .iter()
            .map(|g| with_kv(batch_shape(g, table, Some(arch)), g, &self.stages[0].kv))
            .collect();
        self.annotate_captures(&micros, &mut shapes);
        let n_stages = self.stages.len();
        let n_micro = micros.len();
        // Model bytes (#391): prompt steps apart from decode and verify steps.
        let t_step = std::time::Instant::now();
        let prompt = groups.iter().any(|g| {
            g.seqs
                .iter()
                .any(|s| s.tokens.len() > crate::ggml::model_bytes::DECODE_MAX_TOKENS)
        });
        let kind = Kind::of_step(prompt);
        // A decode or verify step's new graphs are captured behind their first run (#827).
        let _early = crate::ggml::stage_graph::EarlyScope::enter(!prompt);
        // A decode or verify step's cache work by stage, each share beside the next stage's
        // compute (see `after_split`). With MTP drafting it runs beside the next MTP pass
        // instead (a head in the model's file that does not draft runs no pass).
        let split = (self.after_overlap && n_micro == 1 && !prompt && !self.mtp_enabled)
            .then(|| self.stage_split())
            .flatten();
        let n_tok: usize = shapes.iter().flatten().map(|g| g.tokens()).sum();
        // Pair parallelism (`pair_step`): each stage's GPU pair helps with its experts.
        let paired = self.pair_fits(n_micro, prompt, n_tok);
        if paired {
            shapes.iter_mut().flatten().for_each(|g| g.pair = true);
        }
        // A decode or verify step's inputs as copies queued ahead of each stage's graph
        // (`step_fill`).
        let queued = step_fill::queued(n_micro, prompt, paired, self.remote.is_some());
        let mut phase = crate::ggml::model_bytes::Phase::default();
        // Each stage's part (its busy time as `seconds`).
        let mut stage_phase = [crate::ggml::model_bytes::Phase::default(); MAX_STAGES];
        let paged = table.is_some();
        // Per micro-chunk: the captured layers' rows, in stage order.
        let mut hidden: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); n_micro];
        let mut deferred = DeferredRows::new(
            shapes
                .iter()
                .map(|m| m.iter().map(|g| g.tokens()).sum())
                .collect(),
        );
        let mut out = StepOutput::default();
        // A one-micro-batch step (decode, verify) runs its stages one after another: each
        // later stage's inputs are filled while the stage before it computes (`filled`), and
        // only the residual hand-off waits for it.
        let mut filled = vec![false; n_stages];
        // Stages whose hand-off, pass start and graph were queued while the stage before them
        // computed (`launch_ahead`): their round only waits for them.
        let mut launched = vec![false; n_stages];
        // Rounds of several stages: how far each stage's next micro-chunk got while the round
        // before still computed on other stages (see `early_inputs`).
        let mut early = vec![Early::No; n_stages];
        let mut n_rounds = n_micro + n_stages - 1;
        front.lap("shapes");
        if self.dataflow_for(n_micro) {
            let flow = dataflow::Flow {
                micros: &micros,
                shapes: &shapes,
                kind,
                paged,
                overlap_mtp,
            };
            let sinks = dataflow::Sinks {
                deferred: &mut deferred,
                hidden: &mut hidden,
                out: &mut out,
                phase: &mut phase,
                stage_phase: &mut stage_phase,
            };
            self.step_dataflow(flow, sinks)?;
            n_rounds = 0;
        }
        for round in 0..n_rounds {
            let active = round_active(round, n_stages, n_micro);
            // Round 0 ran it beside its stage (below); the last stage's layers need it first.
            if round > 0 {
                self.flush_deferred_tail()?;
            }
            let t0 = std::time::Instant::now();
            // `--prompt-step-stats`: the counters at the start of a long prompt step, and
            // whether it builds its graph.
            let stats = match active[..] {
                [(s, k)]
                    if prompt_stats::logging()
                        && shapes[k].iter().map(|g| g.tokens()).sum::<usize>()
                            >= prompt_stats::MIN_TOKENS =>
                {
                    let built = self.stages[s].kept(&shapes[k]).is_none();
                    Some((prompt_stats::Snap::take(self.model.ngram_table()), built))
                }
                _ => None,
            };
            let mut ahead_ms = 0.0;
            // Later stages first: a stage's outgoing residual may share memory with its inputs
            // (the allocator reuses an input's bytes once it is consumed), so the next stage
            // takes it before this stage's inputs are refilled for its next micro-chunk; and
            // before this stage prepares that micro-chunk's graph, which may drop the one the
            // residual is in. Stages done early (`early_inputs`) keep that order.
            let mut input_ms = Vec::new();
            let mut t_prepare = std::time::Duration::ZERO;
            for &(s, k) in active.iter().rev() {
                if std::mem::take(&mut filled[s]) {
                    if !launched[s] {
                        input_ms.push((0, self.hand_off(s, &shapes[k])));
                    }
                    continue;
                }
                match std::mem::take(&mut early[s]) {
                    Early::Ready => {}
                    Early::Filled => input_ms.push((0, self.hand_off(s, &shapes[k]))),
                    Early::No => {
                        let t = std::time::Instant::now();
                        self.prepare(s, &shapes[k])?;
                        t_prepare += t.elapsed();
                        front.lap("prepare");
                        // A round of one stage leaves the n-gram rows read from the SSD to its
                        // compute, which reads them while the layers before their first reader
                        // run (see `Model::set_ple_late`).
                        self.model.set_ple_late(active.len() == 1);
                        input_ms.push(self.load_inputs(s, &micros[k], &shapes[k], queued));
                        self.model.set_ple_late(false);
                        front.lap("fill");
                    }
                }
            }
            for &(s, k) in &active {
                let sp = &mut stage_phase[s.min(MAX_STAGES - 1)];
                if let Some(kept) = self.stages[s].kept(&shapes[k]) {
                    phase.add(kept.model_bytes(), 0.0);
                    sp.add_graph(kept.model_bytes());
                }
                // The KV its attention reads (#391).
                let kv = &self.stages[s].kv;
                let tier = (kv.hbm_blocks() < kv.n_blocks()).then_some(kv.hbm_blocks());
                let (dev, host) = self.kv_rates[s].step(&micros[k], paged, tier);
                phase.kv += dev;
                phase.kv_host += host;
                sp.kv += dev;
                sp.kv_host += host;
            }
            let t_inputs = t0.elapsed() - t_prepare;
            let mut stage_done: Vec<f64> = Vec::new();
            let t_go = t0.elapsed().as_secs_f64() * 1e3;
            // Stage GPUs that run nothing this round lend their link to their pair's staging
            // (`--prefill-peer-stage`; a stage's index is its CUDA device's, as for the expert
            // peer fetch).
            let idle = (0..self.stages.len()).filter(|s| active.iter().all(|a| a.0 != *s));
            if let Some(r) = self.model.experts_mut() {
                r.set_idle(idle.collect());
            }
            for &(s, k) in &active {
                if launched[s] {
                    continue;
                }
                let layers = self.stages[s].layers.clone();
                let n = shapes[k].iter().map(|g| g.tokens()).sum();
                self.begin_pass(n, |il| layers.contains(&il), kind)?;
            }
            if let Some(r) = self.model.experts_mut() {
                r.set_idle(Vec::new());
            }
            front.lap("begin_pass");
            let mut t_sync = Vec::new();
            let (mut output_ms, mut early_ms) = (0.0, 0.0);
            let t_launch;
            if active.len() > 1 {
                // One host thread per stage: enqueueing a big graph blocks once the stream's
                // launch queue is full, so from one thread the second stage would only start
                // as the first one ends.
                //
                // The host waits for each stage on its own, not for the whole round: as a
                // stage's graph finishes, its rows are read out and the next round's inputs
                // that no longer wait for a running stage are built, filled and handed over
                // (`early_inputs`), beside the stages still computing. In the spirit of
                // Strata-AC922's prefill expert grouping, which waits on an event for the
                // router's ids rather than on the whole stream (eelgaev/Strata-AC922 1d728c3,
                // MIT); Legion groups experts on the device, so what the host waited on here
                // was the round's slowest stage.
                //
                // Off the engine thread's core (#250), worked out here, where it runs now.
                let stage_cpus: Vec<Vec<u32>> = self
                    .stage_cpus
                    .iter()
                    .map(|c| crate::hw::host_core::off_host_core(c))
                    .collect();
                let mut launchers = Vec::with_capacity(active.len());
                for &(s, k) in &active {
                    let Stage {
                        decode,
                        mixed,
                        backend,
                        ..
                    } = &mut self.stages[s];
                    let kept = kept_in(decode, mixed, &shapes[k]).expect("prepared");
                    // SAFETY: until its thread below has sent that it is done, nothing else
                    // uses this stage's backend or graph: the host reads a stage's rows once
                    // it is done, and `early_inputs` builds and fills only stages whose graphs
                    // are done (and reads the residual of done ones). The scope joins every
                    // thread before the round goes on.
                    launchers.push((s, unsafe { backend.launcher(kept)? }));
                }
                t_launch = t0.elapsed();
                let next = if round + 1 < n_rounds {
                    round_active(round + 1, n_stages, n_micro)
                } else {
                    Vec::new()
                };
                let mut ran = vec![false; n_stages];
                for &(s, _) in &active {
                    ran[s] = true;
                }
                let mut finished = vec![false; n_stages];
                let mut done_at = vec![None; n_stages];
                let (tx, rx) = std::sync::mpsc::channel();
                let result: Result<(), ModelError> = std::thread::scope(|scope| {
                    for (s, launcher) in launchers {
                        let tx = tx.clone();
                        let cpus = stage_cpus.get(s).cloned().unwrap_or_default();
                        scope.spawn(move || {
                            if !cpus.is_empty() {
                                // Best effort: a failed pin leaves the inherited affinity.
                                let _ = crate::hw::numa::pin_current_thread(&cpus);
                            }
                            let r = launcher.run();
                            let _ = tx.send((s, r, std::time::Instant::now()));
                        });
                    }
                    drop(tx);
                    // Before any stage is done (a round of a prompt runs for hundreds of ms).
                    self.gather_ahead(&next, &micros);
                    let mut first_err = None;
                    for (s, r, at) in rx.iter() {
                        done_at[s] = Some(at);
                        if first_err.is_some() {
                            continue;
                        }
                        let k = active.iter().find(|a| a.0 == s).expect("active").1;
                        let t = std::time::Instant::now();
                        if let Err(e) = r {
                            first_err = Some(e.into());
                            continue;
                        }
                        self.stage_output(
                            s,
                            k,
                            round,
                            &micros[k],
                            &shapes[k],
                            n_micro,
                            &mut deferred,
                            &mut hidden,
                            &mut out,
                            None,
                        );
                        finished[s] = true;
                        output_ms += t.elapsed().as_secs_f64() * 1e3;
                        let t = std::time::Instant::now();
                        let progress = EarlyProgress {
                            next: &next,
                            ran: &ran,
                            finished: &finished,
                        };
                        if let Err(e) = self.early_inputs(progress, &mut early, &micros, &shapes) {
                            first_err = Some(e);
                        }
                        early_ms += t.elapsed().as_secs_f64() * 1e3;
                    }
                    first_err.map_or(Ok(()), Err)
                });
                result?;
                for &(s, _) in &active {
                    // When each stage's graph finished: the slower stage sets the round.
                    let at = done_at[s].expect("every launch thread reports");
                    stage_done.push((at - t0).as_secs_f64() * 1e3);
                }
            } else {
                // One stage runs: the other GPUs are idle, so its NVLink peer may fetch missed
                // experts for it (`--expert-peer-fetch`) while its graph is launched.
                let idle = super::expert_mv::peer::Idle::new(n_stages > 1);
                // Pair parallelism: the pair's helper first, then the stage's own graph.
                let helped = match active[..] {
                    [(s, _)] if paired => self.pair_launch(s, n_tok)?,
                    _ => None,
                };
                // The verify shape the next steps move to, built while this one computes
                // (`verify_ahead`).
                let ahead = match active[..] {
                    [(0, 0)] if n_micro == 1 => self.verify_ahead(&micros[0], table, &shapes[0]),
                    _ => None,
                };
                // Whether its launch writes n-gram rows late: the next shape's first run will too.
                let late = ahead.is_some() && self.model.ple_late_pending();
                // The prompt's next chunk graph, built while this chunk computes
                // (`prompt_ahead`).
                let prompt_ahead = match active[..] {
                    [(0, 0)] if n_micro == 1 && ahead.is_none() => {
                        self.prompt_ahead(&micros[0], table, &shapes[0], prompt_next)
                    }
                    _ => None,
                };
                front.lap("ahead");
                let sync = n_stages == 1 && ahead.is_none() && prompt_ahead.is_none();
                for &(s, k) in &active {
                    if !launched[s] {
                        self.launch_stage(s, &shapes[k], sync)?;
                    }
                }
                front.done();
                let t_launched = t0.elapsed();
                // `--tp-profile`'s laps: the next verify graph built, then the GPU waited for.
                let (mut t_ahead, mut ahead_wait) = (t_launched, std::time::Duration::ZERO);
                if let Some(next) = &ahead {
                    self.build_ahead(0, &shapes[0], next, late);
                    t_ahead = t0.elapsed();
                    self.stages[0].backend.synchronize();
                    ahead_wait = t0.elapsed() - t_ahead;
                }
                if let Some(next) = &prompt_ahead {
                    let t = std::time::Instant::now();
                    self.build_prompt_ahead(0, &shapes[0], next);
                    ahead_ms = t.elapsed().as_secs_f64() * 1e3;
                    self.stages[0].backend.synchronize();
                }
                drop(idle);
                // The last MTP pass's cache work, while the target's first stage computes.
                if round == 0
                    && overlap_mtp
                    && let Err(e) = self.flush_deferred_last()
                {
                    for (s, _) in &active {
                        self.stages[*s].backend.synchronize();
                    }
                    return Err(e);
                }
                // The cache work for the stage the last round ran, while this one computes.
                if round > 0
                    && let Some(sp) = &split
                    && let Err(e) = self.after_pass_more(n_tok, sp[round - 1].clone(), kind)
                {
                    for (s, _) in &active {
                        self.stages[*s].backend.synchronize();
                    }
                    return Err(e);
                }
                // The next stage's inputs while this one computes (see `filled`).
                if let [(s, 0)] = active[..]
                    && n_micro == 1
                    && s + 1 < n_stages
                {
                    // Stage `s` is running: on an error, wait for it before returning, so the
                    // caller never reuses its slots and KV blocks while it still writes them.
                    if let Err(e) = self.prepare(s + 1, &shapes[0]) {
                        self.stages[s].backend.synchronize();
                        return Err(e);
                    }
                    self.fill_step_inputs(s + 1, &micros[0], &shapes[0], queued);
                    filled[s + 1] = true;
                    // Its hand-off and graph too, behind this stage's on the GPUs.
                    if self.launches_ahead(queued, s) {
                        if let Err(e) = self.launch_ahead(s + 1, &shapes[0], kind) {
                            self.stages[s].backend.synchronize();
                            self.stages[s + 1].backend.synchronize();
                            return Err(e);
                        }
                        launched[s + 1] = true;
                    }
                }
                if round + 1 < n_rounds {
                    self.gather_ahead(&round_active(round + 1, n_stages, n_micro), &micros);
                }
                t_launch = t0.elapsed();
                for &(s, k) in &active {
                    // With the inputs' queued copies (`step_fill`).
                    self.sync_stage(s);
                    self.pair_wait(helped, s, n_tok)?;
                    t_sync.push(t0.elapsed().as_secs_f64() * 1e3);
                    let t_out = std::time::Instant::now();
                    self.stage_output(
                        s,
                        k,
                        round,
                        &micros[k],
                        &shapes[k],
                        n_micro,
                        &mut deferred,
                        &mut hidden,
                        &mut out,
                        None,
                    );
                    output_ms += t_out.elapsed().as_secs_f64() * 1e3;
                }
                if let [(_, _)] = active[..]
                    && n_rounds == 1
                {
                    let out_d = std::time::Duration::from_secs_f64(output_ms / 1e3);
                    let synced = t0.elapsed().saturating_sub(out_d);
                    self.step_laps = Some(crate::tp_net::StepLaps {
                        before: t0 - t_entry,
                        prepare: t_prepare,
                        fill: t_inputs,
                        launch: t_launched.saturating_sub(t_inputs + t_prepare),
                        rows: self.late_rows_took,
                        ahead: t_ahead - t_launched,
                        beside: t_launch.saturating_sub(t_ahead + ahead_wait),
                        wait: ahead_wait + synced.saturating_sub(t_launch),
                        output: out_d,
                    });
                }
            }
            // The cache learns each micro-batch's routing (its graph's route log holds it until
            // the stage runs the next one), once every stage of the round is done.
            let mut after_ms = 0.0;
            if n_micro > 1 {
                for &(s, k) in &active {
                    let t_after = std::time::Instant::now();
                    let n = shapes[k].iter().map(|g| g.tokens()).sum();
                    let layers = self.stages[s].layers.clone();
                    self.after_pass(n, |il| layers.contains(&il), kind)?;
                    after_ms += t_after.elapsed().as_secs_f64() * 1e3;
                }
            }
            // Each active stage's busy time: from the launches to its graph done (several
            // stages at once) or synchronized.
            for (i, &(s, _)) in active.iter().enumerate() {
                let done = stage_done.get(i).or(t_sync.get(i)).copied().unwrap_or(t_go);
                stage_phase[s.min(MAX_STAGES - 1)].seconds += (done - t_go).max(0.0) / 1e3;
            }
            if shapes.iter().flatten().any(|g| g.tokens() > 64) {
                tracing::debug!(
                    round,
                    active = ?active,
                    prepare_ms = t_prepare.as_secs_f64() * 1e3,
                    inputs_ms = t_inputs.as_secs_f64() * 1e3,
                    launched_ms = t_launch.as_secs_f64() * 1e3,
                    synced_ms = ?t_sync,
                    stage_done_ms = ?stage_done,
                    fill_handoff_ms = ?input_ms,
                    output_ms,
                    early_ms,
                    after_ms,
                    total_ms = t0.elapsed().as_secs_f64() * 1e3,
                    "step round"
                );
            }
            if let Some((snap, built)) = &stats {
                let shape = shapes.iter().flatten();
                snap.log(
                    self.model.ngram_table(),
                    shape.clone().map(|g| g.tokens()).sum(),
                    shape.map(|g| g.n_pool).sum(),
                    *built,
                    t_prepare.as_secs_f64() * 1e3,
                    t_inputs.as_secs_f64() * 1e3,
                    t_launch.as_secs_f64() * 1e3,
                    t_sync.last().copied().unwrap_or(0.0),
                    ahead_ms,
                    output_ms,
                );
            }
        }
        deferred.join(None, &mut hidden);
        // Captured layers in the order asked for, the micro-chunks' rows one after another.
        out.hidden = self
            .hidden
            .iter()
            .map(|il| {
                // Copied in place by the deferred copies (#435).
                if let Some(rows) = deferred.assemble(*il, &mut hidden) {
                    return rows;
                }
                rows_out::take_layer(&mut hidden, *il)
            })
            .collect();
        phase.seconds = t_step.elapsed().as_secs_f64();
        phase.steps = 1;
        self.bytes.phase_mut(kind).merge(&phase);
        for (s, sp) in stage_phase.iter_mut().enumerate().take(n_stages) {
            sp.steps = 1;
            if let Some(into) = self.bytes.stage_mut(s, kind) {
                into.merge(sp);
            }
        }
        // Not run beside the stages above (multi-active rounds): flush what is left.
        self.flush_deferred_last()?;
        // Rank 0 of a split model: the layers on rank 1, and the logits from there.
        if let Some(r) = self.remote.as_mut() {
            let rows = r.take_rows();
            if let Some(o) = r.step(groups, rows)? {
                out.argmax = o.argmax;
                out.rows = o.rows;
            }
        }
        if n_micro == 1 {
            let n_layer = self.model.hp.n_layer;
            let n = n_tok;
            if let Some(mut sp) = split {
                // The last stage's share: beside the next step's first stage, or the next MTP
                // pass (`take_target_work`). It counts the step.
                let last = sp.pop().expect("a split has stages");
                self.deferred_tail = Some((n, kind, last));
            } else if self.after_overlap && self.mtp_enabled {
                // Run beside the next MTP pass (or before the next step).
                self.deferred_target = Some((n, kind));
            } else {
                self.after_pass(n, |il| il < n_layer, kind)?;
            }
        }
        Ok(out)
    }

    /// The step as micro-chunks (see [`Self::step`]): itself, unless it gathers more than the
    /// budget ([`Self::budget_chunks`]), or it is one prompt chunk (one sequence, not a verify)
    /// longer than the micro-chunk on several stages and its logits are the last token's or
    /// none, or such a chunk after other groups ([`Self::set_pipeline_riders`]).
    fn micro_chunks<'g>(&self, groups: &[Group<'g>]) -> Vec<Vec<Group<'g>>> {
        let chunks = self.budget_chunks(groups);
        if chunks.len() > 1 {
            return chunks;
        }
        let micro = self.pipeline_micro;
        let whole = vec![groups.to_vec()];
        // Riders ahead of a long prompt chunk (`set_pipeline_riders`): a micro-chunk of their
        // own, then the prompt's micro-chunks as when it runs alone.
        if self.pipeline_riders
            && let Some((lead, riders)) = groups.split_last()
            && !riders.is_empty()
            && let [seq] = &lead.seqs[..]
            && self.stages.len() > 1
            && micro > 0
            && seq.tokens.len() > micro
            && !lead.verify
            && seq.logits.last_only()
        {
            let mut out = vec![riders.to_vec()];
            out.extend(self.micro_chunks(std::slice::from_ref(lead)));
            return out;
        }
        let [g] = groups else { return whole };
        let n_stages = self.stages.len();
        if self.split_batches
            && n_stages > 1
            && g.seqs.len() >= MIN_SPLIT_SEQS
            && g.seqs
                .iter()
                .all(|s| s.tokens.len() == g.seqs[0].tokens.len())
            && (g.verify || g.seqs[0].tokens.len() == 1)
        {
            // A decode or verify batch: one micro-batch per stage, of consecutive sequences
            // (slots), so the stages work on different ones at once.
            let parts = n_stages.min(g.seqs.len());
            let per = g.seqs.len().div_ceil(parts);
            return g
                .seqs
                .chunks(per)
                .enumerate()
                .map(|(i, seqs)| {
                    vec![Group {
                        slot: g.slot + i * per,
                        seqs: seqs.to_vec(),
                        verify: g.verify,
                    }]
                })
                .collect();
        }
        let [seq] = &g.seqs[..] else { return whole };
        let n = seq.tokens.len();
        let last_only = seq.logits.last_only();
        if self.stages.len() < 2 || micro == 0 || n <= micro || g.verify || !last_only {
            return whole;
        }
        let sizes = taper::micro_sizes(n, micro, self.micro_taper);
        sizes
            .iter()
            .scan(0, |at, &len| {
                *at += len;
                Some((*at - len, *at))
            })
            .map(|(at, end)| {
                vec![Group {
                    slot: g.slot,
                    seqs: vec![SeqInput {
                        tokens: &seq.tokens[at..end],
                        n_past: seq.n_past + at,
                        blocks: seq.blocks,
                        ring: seq.ring,
                        logits: if end == n { seq.logits } else { Logits::None },
                        media: None,
                    }],
                    verify: false,
                }]
            })
            .collect()
    }

    /// The layers stage `s` captures for [`Self::capture_hidden`].
    fn captured(&self, s: usize) -> Vec<usize> {
        // GLM's parallel steps: the last GPU's graph alone ends in the head and its input.
        if self.par.every_step && s + 1 != self.stages.len() {
            return Vec::new();
        }
        let layers = &self.stages[s].layers;
        self.hidden
            .iter()
            .copied()
            .filter(|il| layers.contains(il))
            .collect()
    }

    /// What stage `s`'s graphs hold besides the fixed inputs.
    fn parts(&self, s: usize) -> InputParts {
        let hp = &self.model.hp;
        let layers = &self.stages[s].layers;
        // By the layers (as `Model::build_stage` builds them): a node of a split model starts
        // or ends past its stages ([`remote`]).
        InputParts {
            n_hidden: self.captured(s).len(),
            ple: hp.arch.token_rows(hp, layers) > 0,
            group: hp.arch.group_inputs_len(),
            seq_ids: hp.arch.slot_ids(),
            // GLM's parallel steps: every GPU's stage runs all the layers, the last GPU's
            // graph alone ends in the logits ([`Executor::new_glm_par`]).
            head: layers.end == hp.n_layer && (!self.par.every_step || s + 1 == self.stages.len()),
            res_in: layers.start > 0,
            res_out: layers.end < hp.n_layer,
            swa: self.model.swa_window(),
        }
    }

    /// Builds stage `s`'s graph for `shape` unless it keeps one.
    fn prepare(&mut self, s: usize, shape: &[GroupShape]) -> Result<(), ModelError> {
        self.build_kept(s, shape)?;
        // Ready for its inputs (a meta backend may have dropped their per-device tensors).
        let Stage {
            decode,
            mixed,
            backend,
            ..
        } = &mut self.stages[s];
        if let Some(kept) = kept_in(decode, mixed, shape) {
            backend.ready_kept(kept)?;
        }
        Ok(())
    }

    /// [`Self::prepare`] without making the graph ready.
    fn build_kept(&mut self, s: usize, shape: &[GroupShape]) -> Result<(), ModelError> {
        if self.stages[s].kept(shape).is_some() {
            // Most recent last.
            let st = &mut self.stages[s];
            for list in [&mut st.mixed, &mut st.decode] {
                if let Some(i) = list.iter().position(|(x, _)| x == shape) {
                    let g = list.remove(i);
                    list.push(g);
                }
            }
            return Ok(());
        }
        let captured = self.captured(s);
        let lent = self.mixed_kept == LENT_MIXED_KEPT;
        let verify_budget = self.verify_budget.get(s).copied().unwrap_or(0);
        let stage = &mut self.stages[s];
        // A meta backend (tensor parallelism) re-initializes a kept graph whenever another ran
        // in between anyway, and its compute buffers sit next to a KV cache sized to what is
        // left: the kept graphs of a kind share one buffer (they run one at a time, inputs set
        // each time), and steps that alternate shapes (the gather budget's split steps, the
        // drafted rows) need no rebuilds.
        let meta = stage.backend.is_meta();
        let (decode_kept, mixed_kept) = if meta {
            (META_KEPT, META_KEPT)
        } else {
            // A graph set aside while this one is built counts as kept (`prompt_ahead`).
            (DECODE_KEPT, self.mixed_kept - self.mixed_aside)
        };
        let opts = StepOpts {
            spec: stage.spec.as_ref(),
            saved: Some(&stage.saved),
            hidden: &captured,
            mtp: self.mtp_enabled,
            par_rank: s,
        };
        // Verify steps (several tokens per decoding sequence) count as mixed; with
        // `--verify-kept-graphs` they keep a share of their own ([`verify_kept`]).
        let decode = shape.iter().all(|g| g.n_tok == 1);
        let logging = crate::ggml::stage_graph::stats::logging();
        // `--stage-graph-stats`' build line, against what is kept before the eviction.
        let mut note = None;
        let old = if decode {
            if logging {
                note = Some(shape_diff::note(
                    "decode",
                    s,
                    shape,
                    stage.decode.iter().map(|(x, _)| x.as_slice()),
                    stage.decode.len(),
                    decode_kept,
                ));
            }
            (stage.decode.len() >= decode_kept).then(|| stage.decode.remove(0))
        } else {
            let verify_kept = if meta || lent { 0 } else { verify_kept::kept() };
            let kinds: Vec<bool> = stage
                .mixed
                .iter()
                .map(|(s, _)| verify_kept::is_verify(s))
                .collect();
            let verify = verify_kept::is_verify(shape);
            // Under the verify budget (`--verify-kept-mib`): when the kept verify graphs fill
            // it, the oldest goes now and its buffer holds the new graph.
            let full = verify_kept > 0 && {
                let sizes: Vec<usize> = stage
                    .mixed
                    .iter()
                    .zip(&kinds)
                    .filter(|(_, v)| **v)
                    .map(|((_, k), _)| k.allocator_size())
                    .collect();
                let next = sizes.last().copied().unwrap_or(0);
                verify_kept::at_budget(verify, sizes.iter().sum(), next, verify_budget)
            };
            let oldest_verify = || kinds.iter().position(|&v| v);
            if logging {
                let same = || {
                    (stage.mixed.iter().zip(&kinds))
                        .filter(|(_, v)| **v == verify)
                        .map(|((x, _), _)| x.as_slice())
                };
                // Kept against the cap: the kind's own share, or all mixed graphs.
                let (n_kept, cap) = match verify_kept {
                    0 => (kinds.len(), mixed_kept),
                    n if verify => (same().count(), n),
                    _ => (same().count(), mixed_kept),
                };
                let kind = shape_diff::step_kind(shape, verify);
                let mut n = shape_diff::note(kind, s, shape, same(), n_kept, cap);
                if full && n.why != "new shape" {
                    n.why = "evicted at the byte budget";
                }
                note = Some(n);
            }
            verify_kept::evict(&kinds, verify, mixed_kept, verify_kept)
                .or_else(|| full.then(oldest_verify).flatten())
                .map(|i| stage.mixed.remove(i))
        };
        let buft = stage.backend.buffer_type();
        let roomy = headroom::roomy(stage.backend.device().kind(), decode);
        let shared = meta.then(|| {
            stage.shared[usize::from(!decode)]
                .get_or_insert_with(|| {
                    std::sync::Arc::new(std::sync::Mutex::new(ComputeAllocator::new(buft)))
                })
                .clone()
        });
        let allocator = match shared {
            Some(_) => None,
            None => Some(
                old.and_then(|(_, g)| g.into_allocator())
                    // The first graph after an expert-cache restore: the buffer kept for it.
                    .or_else(|| {
                        let a = if lent { None } else { stage.spare.take() }?;
                        tracing::debug!(
                            stage = s,
                            mib = a.buffer_size() >> 20,
                            "step graph built in the spare compute buffer"
                        );
                        Some(a)
                    })
                    // A prompt graph: a buffer reserved at load, if one is left (#836).
                    .or_else(|| {
                        if lent || decode || verify_kept::is_verify(shape) {
                            return None;
                        }
                        let a = stage.ready.pop()?;
                        tracing::debug!(
                            stage = s,
                            mib = a.buffer_size() >> 20,
                            left = stage.ready.len(),
                            "step graph built in a compute buffer reserved at load"
                        );
                        Some(a)
                    })
                    // Long prompts' graphs grow a little with every chunk: headroom.
                    .unwrap_or_else(|| headroom::step_allocator(buft, lent, roomy)),
            ),
        };
        let mut allocator = allocator;
        let before = allocator.as_ref().map_or(0, ComputeAllocator::buffer_size);
        let t_build = std::time::Instant::now();
        // Out of memory with other graphs kept: once more without them (#430); then, where
        // there is unified memory, once more in a buffer that spills into host memory.
        let mut retry = true;
        let mut spilled = false;
        let size = graph_size(shape.len());
        let (ctx, parts) = loop {
            let ctx = Context::new(size, size);
            let parts = {
                let (graph, inputs, _) = self.model.build_stage(
                    &ctx,
                    &stage.kv,
                    &stage.rec,
                    opts,
                    shape,
                    false,
                    stage.layers.clone(),
                )?;
                match (&shared, allocator.take()) {
                    (Some(shared), _) => {
                        if ctx.alloc_graph_shared(&graph, shared)? {
                            // The buffer moved: the kind's other kept graphs point into the old
                            // one.
                            if decode {
                                stage.decode.clear();
                            } else {
                                stage.mixed.clear();
                            }
                        }
                    }
                    (None, Some(a)) => match ctx.alloc_graph(&graph, a).and_then(|()| {
                        // Only an allocation of its own (the buffer grew): a graph that fits
                        // the buffer it was given allocates nothing.
                        if lent && !spilled && ctx.allocator_size() > before {
                            lent_scratch(stage.backend.device(), ctx.allocator_size(), buft.name())
                        } else {
                            Ok(())
                        }
                    }) {
                        Ok(()) => {}
                        // The graphs a restore hands its thread may still hold the memory.
                        Err(GgmlError::Alloc { .. }) if self.freeing.wait() => {
                            tracing::debug!(
                                stage = s,
                                "out of memory for a step's compute buffer: waited for the \
                                 restore to free the prompt's graphs, allocating again"
                            );
                            allocator = Some(headroom::step_allocator(buft, lent, roomy));
                            continue;
                        }
                        Err(GgmlError::Alloc { bytes, .. }) => {
                            let tokens = shape.iter().map(GroupShape::tokens).sum::<usize>();
                            // The oldest other kept graph first, mixed before decode (the
                            // smaller, hotter ones): its buffer grows for this graph, the rest
                            // stay (a mixed step beside four decoding sequences is a new shape
                            // every step, and dropping them all rebuilt the decode graphs too).
                            let oldest = if stage.mixed.is_empty() {
                                &mut stage.decode
                            } else {
                                &mut stage.mixed
                            };
                            if !oldest.is_empty() {
                                let (_, g) = oldest.remove(0);
                                if let Some(a) = g.into_allocator() {
                                    tracing::debug!(
                                        stage = s,
                                        tokens,
                                        need_mib = bytes >> 20,
                                        mib = a.buffer_size() >> 20,
                                        "out of memory for a step's compute buffer: the stage's \
                                         oldest kept graph dropped, its buffer reused"
                                    );
                                    allocator = Some(a);
                                    continue;
                                }
                            }
                            let others = stage.decode.len() + stage.mixed.len();
                            if retry && others > 0 {
                                tracing::warn!(
                                    stage = s,
                                    tokens,
                                    need_mib = bytes >> 20,
                                    dropped = others,
                                    "out of memory for a step's compute buffer: the stage's \
                                     other kept graphs dropped, allocating again"
                                );
                                stage.decode.clear();
                                stage.mixed.clear();
                                retry = false;
                                allocator = Some(headroom::step_allocator(buft, lent, roomy));
                                continue;
                            }
                            if let Some(spill) = (!spilled).then(|| spill_type(stage)).flatten() {
                                tracing::warn!(
                                    stage = s,
                                    tokens,
                                    need_mib = bytes >> 20,
                                    "out of VRAM for a step's compute buffer: allocating it in \
                                     unified memory (VRAM, then host memory)"
                                );
                                spilled = true;
                                allocator = Some(ComputeAllocator::new(spill));
                                continue;
                            }
                            return Err(ModelError::StepOutOfMemory {
                                device: stage.backend.name(),
                                tokens,
                                need_mib: bytes.div_ceil(1 << 20),
                                // With what the failed graph holds (it is dropped next).
                                free_mib: (stage.backend.device().memory().0
                                    + ctx.allocator_size())
                                    >> 20,
                            });
                        }
                        Err(e) => return Err(e.into()),
                    },
                    (None, None) => unreachable!(),
                }
                KeptParts::new(&graph, &inputs.list())
            };
            break (ctx, parts);
        };
        let after = ctx.allocator_size();
        tracing::trace!(
            stage = s,
            shape = ?shape.iter().map(|g| (g.n_seqs, g.n_tok, g.n_kv)).collect::<Vec<_>>(),
            ms = t_build.elapsed().as_secs_f64() * 1e3,
            "step graph built"
        );
        if after > before {
            tracing::debug!(
                stage = s,
                tokens = shape.iter().map(GroupShape::tokens).sum::<usize>(),
                mib = after as f64 / (1u64 << 20) as f64,
                before_mib = before as f64 / (1u64 << 20) as f64,
                free_mib = stage.backend.device().memory().0 >> 20,
                ms = t_build.elapsed().as_secs_f64() * 1e3,
                "compute buffer grew"
            );
        }
        // SAFETY: the graph references the weights and the stage's pools, owned by self and
        // never replaced while it is kept; the kept graphs are dropped first.
        let mut kept = unsafe { KeptGraph::new(ctx, parts) };
        let split = if verify_ahead::building_ahead() {
            crate::ggml::stage_graph::stats::Split::BuildAheadNs
        } else {
            crate::ggml::stage_graph::stats::Split::BuildStepNs
        };
        crate::ggml::stage_graph::stats::count_split(split, t_build.elapsed().as_nanos() as u64);
        let kind = shape_diff::step_kind(shape, verify_kept::is_verify(shape));
        crate::ggml::stage_graph::stats::note_build(kind, s, || note.unwrap_or_default());
        Self::count_bytes(&mut self.weight_addrs, &self.model, &mut kept);
        let kept = (shape.to_vec(), kept);
        let stage = &mut self.stages[s];
        if decode {
            stage.decode.push(kept);
        } else {
            stage.mixed.push(kept);
            if verify_budget > 0 && !stage.backend.is_meta() && !lent && verify_kept::kept() > 0 {
                self.fit_verify_budget(s, verify_budget);
            }
        }
        self.fit_compute_budget(s, !decode);
        if crate::ggml::stage_graph::stats::logging() {
            // What the kept graphs' own compute buffers hold now (#858: the cost of
            // `--verify-kept-graphs`).
            let st = &self.stages[s];
            tracing::info!(
                stage = s,
                decode = st.decode.len(),
                mixed = st.mixed.len(),
                verify = st
                    .mixed
                    .iter()
                    .filter(|(x, _)| verify_kept::is_verify(x))
                    .count(),
                mib = self.kept_buffer_bytes(s) >> 20,
                verify_mib = self.verify_buffer_bytes(s) >> 20,
                "kept graph buffers (--stage-graph-stats)"
            );
        }
        Ok(())
    }

    /// Fills stage `s`'s inputs for `groups` (as queued copies when `queued`, see
    /// [`step_fill`]), and its incoming residual from what stage `s - 1` computed last for the
    /// same shape. Returns the milliseconds the fill and the hand-off took.
    fn load_inputs(
        &mut self,
        s: usize,
        groups: &[Group<'_>],
        shape: &[GroupShape],
        queued: bool,
    ) -> (u32, u32) {
        let t0 = std::time::Instant::now();
        self.fill_step_inputs(s, groups, shape, queued);
        let filled = t0.elapsed();
        let ms = |d: std::time::Duration| d.as_millis() as u32;
        (ms(filled), self.hand_off(s, shape))
    }

    /// Stage `s`'s own inputs for `groups` (all but the residual from the stage before).
    fn fill_inputs(&self, s: usize, groups: &[Group<'_>], shape: &[GroupShape]) {
        let stage = &self.stages[s];
        let kept = stage.kept(shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, self.parts(s), |i| kept.tensor(i));
        inputs.fill(groups, shape);
        self.model.fill_extra(
            &inputs,
            groups,
            shape,
            &stage.rec,
            &stage.kv,
            stage.spec.as_ref(),
            &mut super::SetNow,
        );
    }

    /// Copies the residual stage `s - 1` handed over into stage `s`'s input (none for the
    /// first stage); returns the milliseconds it took.
    fn hand_off(&self, s: usize, shape: &[GroupShape]) -> u32 {
        let t0 = std::time::Instant::now();
        let stage = &self.stages[s];
        let kept = stage.kept(shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, self.parts(s), |i| kept.tensor(i));
        if let Some(dst) = inputs.res_in
            && s == 0
        {
            // Rank 1 of a split model: the rows that came over the link.
            let rows = self.take_ext_rows().expect("a step from rank 0's rows");
            dst.set_f32(&rows);
        } else if let Some(dst) = inputs.res_in {
            let prev = self.stages[s - 1]
                .kept(shape)
                .expect("the previous stage ran");
            let src = Inputs::from_parts(shape, self.parts(s - 1), |i| prev.tensor(i))
                .res_out
                .expect("a stage before the last hands its residual on");
            dst.copy_from(src);
        }
        t0.elapsed().as_millis() as u32
    }

    /// [`Self::hand_off`] without the host waiting: the copy is queued after stage `s - 1`'s
    /// work and stage `s`'s next graph waits for it. Stage `s - 1` must be synchronized before
    /// the host writes its memory again (its next fill). `false` for the first stage (nothing
    /// to copy).
    fn hand_off_queued(&self, s: usize, shape: &[GroupShape]) -> bool {
        let stage = &self.stages[s];
        let kept = stage.kept(shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, self.parts(s), |i| kept.tensor(i));
        let Some(dst) = inputs.res_in else {
            return false;
        };
        if s == 0 {
            // Rank 1 of a split model: the rows that came over the link.
            let rows = self.take_ext_rows().expect("a step from rank 0's rows");
            dst.set_f32(&rows);
            return true;
        }
        let from = &self.stages[s - 1];
        let prev = from.kept(shape).expect("the previous stage ran");
        let src = Inputs::from_parts(shape, self.parts(s - 1), |i| prev.tensor(i))
            .res_out
            .expect("a stage before the last hands its residual on");
        from.backend.copy_to(&stage.backend, src, dst);
        true
    }

    /// Reads stage `s`'s rows of micro-chunk `k` (`group`, `shape`) once its graph is done:
    /// the captured layers' into `hidden` (one captured layer of a pipelined step is copied out
    /// beside the next round, see `DeferredRows`), and the last stage's logits rows or argmax
    /// into `out`. `read`: the captured rows already in the round's bounce buffer.
    #[allow(clippy::too_many_arguments)]
    fn stage_output(
        &mut self,
        s: usize,
        k: usize,
        round: usize,
        group: &[Group<'_>],
        shape: &[GroupShape],
        n_micro: usize,
        deferred: &mut DeferredRows,
        hidden: &mut [Vec<(usize, Vec<f32>)>],
        out: &mut StepOutput,
        read: Option<HiddenRead>,
    ) {
        let n_stages = self.stages.len();
        let parts = self.parts(s);
        let captured = self.captured(s);
        let n_vocab = self.model.hp.n_vocab as usize;
        let exact = self.model.hp.arch.argmax_first_of_ties();
        // The step's captured rows may stay where the graph wrote them
        // ([`Executor::set_hidden_on_device`]): one stage, one micro-chunk, nothing read ahead.
        let stay_on = self
            .hidden_on_device
            .filter(|_| n_micro == 1 && n_stages == 1 && read.is_none() && self.remote.is_none());
        let Stage {
            decode,
            mixed,
            bounce,
            bounce_b,
            bounce_small,
            ids_pin,
            backend,
            ..
        } = &mut self.stages[s];
        let spare = &mut self.spare_rows;
        let kept = kept_in(decode, mixed, shape).expect("prepared");
        let inputs = Inputs::from_parts(shape, parts, |i| kept.tensor(i));
        let ids = bounce_for(ids_pin, backend, inputs.out_pin_bytes(group, n_vocab));
        let picks = super::Picks {
            stop: &self.stop_tokens,
            exact,
        };
        // Every captured layer an F32 matrix on that device, or all are read back.
        let on_device = stay_on.and_then(|dev| {
            (inputs.hidden.iter())
                .map(|t| t.rows_at().filter(|r| r.device == Some(dev)))
                .collect::<Option<Vec<_>>>()
        });
        let defer = n_micro > 1 && captured.len() == 1;
        // The one micro-chunk's rows are the step's ([`Executor::set_hidden_from`]).
        let from = if n_micro == 1 { self.hidden_from } else { 0 };
        let parity = round % 2;
        let mut started = Vec::new();
        let mut o = inputs.output_with(group, n_vocab, ids, picks, |t| {
            if on_device.is_some() {
                return Vec::new();
            }
            let n = t.nbytes();
            if defer && n >= BOUNCE_MIN_BYTES {
                // Already in the round's bounce buffer (the dataflow stage thread read it).
                // SAFETY: reading the data pointer of a live tensor.
                if let Some(r) = read.filter(|r| r.src == unsafe { (*t.raw()).data } as usize) {
                    started.push((r.dst, r.bytes));
                    return Vec::new();
                }
                let slot = if parity == 0 {
                    &mut *bounce
                } else {
                    &mut *bounce_b
                };
                // The buffer's last rows (two rounds ago) must be out of it first.
                deferred.join(Some(parity), hidden);
                if let Some(via) = bounce_for(slot, backend, n) {
                    started.push(t.read_via(via));
                    return Vec::new();
                }
            }
            read_rows_from(bounce, bounce_small, spare, backend, t, from)
        });
        for (&layer, &(src, n)) in captured.iter().zip(&started) {
            deferred.start(k, layer, parity, src, n);
        }
        hidden[k].extend(captured.into_iter().zip(std::mem::take(&mut o.hidden)));
        if let Some(rows) = on_device {
            out.hidden_dev = rows;
        }
        // Rank 0 of a split model: the residual for rank 1 ([`remote`]).
        if s + 1 == n_stages
            && let Some(r) = self.remote.as_mut()
            && let Some(t) = inputs.res_out
        {
            r.push_rows(read_rows(bounce, bounce_small, spare, backend, t));
        }
        if s + 1 == n_stages {
            // The micro-chunks' rows one after another (a prompt's earlier micro-chunks have
            // none).
            out.argmax.extend(o.argmax);
            out.rows.extend(o.rows);
        }
    }

    /// Lends the expert cache's VRAM to a long prompt ([`ExpertResidency::lend`]): its slots
    /// become staging buffers and room for the prompt's big step buffers. The kept prompt
    /// graphs are dropped first (their buffers are freed; graphs built from now on stage).
    /// `false` when the cache does not stage. `tokens`: the prompt's tokens still to read
    /// ([`ExpertResidency::lend_for`]).
    ///
    /// [`ExpertResidency::lend`]: crate::residency::experts::ExpertResidency::lend
    /// [`ExpertResidency::lend_for`]: crate::residency::experts::ExpertResidency::lend_for
    pub fn lend_experts(&mut self, tokens: usize) -> Result<bool, ModelError> {
        self.wait_commit();
        self.flush_deferred()?;
        let can = self.model.experts().is_some_and(|r| r.can_lend());
        if !can {
            return Ok(false);
        }
        if self.model.experts().is_some_and(|r| r.is_lent()) {
            return Ok(true);
        }
        self.keep_spares();
        self.drop_mixed();
        // A long prompt's steps share one graph shape but the last: one compute buffer per
        // stage, not one per shape kept.
        self.mixed_kept = LENT_MIXED_KEPT;
        self.model
            .experts_mut()
            .expect("checked above")
            .lend_for(tokens)
            .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}"))))?;
        self.alloc_spares();
        Ok(true)
    }

    /// Whether stage `s` keeps a spare compute buffer through a lend ([`spare`]): it runs
    /// verify steps and has buffers of its own per kept graph (not a meta backend's shared one).
    fn wants_spare(&self, s: usize) -> bool {
        let st = &self.stages[s];
        st.spec.is_some() && !st.backend.is_meta()
    }

    /// Whether the MTP head keeps a spare take-in buffer through a lend ([`spare`]).
    fn wants_mtp_spare(&self) -> bool {
        self.mtp_enabled && !self.last().backend.is_meta()
    }

    /// Before a lend drops the kept graphs: keeps a kept verify graph's buffer as its stage's
    /// spare, and the MTP head's take-in buffer as its, where one is big enough ([`spare::pick`]).
    fn keep_spares(&mut self) {
        for s in 0..self.stages.len() {
            if !self.wants_spare(s) || self.stages[s].spare.is_some() {
                continue;
            }
            let st = &mut self.stages[s];
            // Verify steps' graphs (no prompt chunk in them).
            let verify: Vec<usize> = (0..st.mixed.len())
                .filter(|&i| verify_kept::is_verify(&st.mixed[i].0))
                .collect();
            let sizes: Vec<usize> = verify
                .iter()
                .map(|&i| st.mixed[i].1.allocator_size())
                .collect();
            if let Some(k) = spare::pick(&sizes) {
                st.spare = st.mixed.remove(verify[k]).1.into_allocator();
            }
        }
        if self.wants_mtp_spare()
            && self.mtp_spare.is_none()
            && spare::pick(&[self.mtp_mixed.buffer_size()]).is_some()
        {
            self.mtp_spare = self.mtp_mixed.take_buffer();
        }
    }

    /// Once the cache is lent (its VRAM is free): allocates the spares [`Self::keep_spares`]
    /// found none for. Without one (out of memory) the graphs after the restore allocate their
    /// own, as before.
    fn alloc_spares(&mut self) {
        let alloc = |s: usize, buft| {
            ComputeAllocator::reserved(buft, spare::SPARE_BYTES)
                .inspect_err(|e| tracing::debug!(stage = s, error = %e, "no spare compute buffer"))
                .ok()
        };
        for s in 0..self.stages.len() {
            if self.wants_spare(s) && self.stages[s].spare.is_none() {
                self.stages[s].spare = alloc(s, self.stages[s].backend.buffer_type());
            }
        }
        if self.wants_mtp_spare() && self.mtp_spare.is_none() {
            let s = self.stages.len() - 1;
            self.mtp_spare = alloc(s, self.last().backend.buffer_type());
        }
    }

    /// Takes the expert cache's VRAM back after [`Self::lend_experts`]: drops the kept prompt
    /// graphs (they stage, and their big buffers go), then refills the slots in the background.
    /// The graphs are dropped by the restore's thread before it refills ([`freeing`]); the first
    /// graphs built after it allocate from the spares kept through the lend ([`spare`]).
    pub fn restore_experts(&mut self) -> Result<(), ModelError> {
        self.restore_experts_leaving(&[])
    }

    /// [`Self::restore_experts`] that leaves `(CUDA index, bytes)` of each device's lent slots
    /// without VRAM ([`ExpertResidency::restore_leaving`]): the KV pools' tail holds it
    /// (`--kv-vmm-keep`, #814). [`Self::refill_experts`] gives them VRAM again.
    ///
    /// [`ExpertResidency::restore_leaving`]: crate::residency::experts::ExpertResidency::restore_leaving
    pub fn restore_experts_leaving(&mut self, leave: &[(usize, u64)]) -> Result<(), ModelError> {
        self.wait_commit();
        self.flush_deferred()?;
        if !self.model.experts().is_some_and(|r| r.is_lent()) {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let mut graphs: Vec<Box<dyn Send>> = Vec::new();
        for s in &mut self.stages {
            // A meta backend's kept graphs share its buffers: dropped here, as before.
            if s.backend.is_meta() {
                s.mixed.clear();
                s.shared[1] = None;
            } else {
                graphs.extend(s.mixed.drain(..).map(|g| Box::new(g) as Box<dyn Send>));
                if let Some(a) = s.shared[1].take() {
                    graphs.push(Box::new(a));
                }
            }
        }
        // The MTP head's prompt graphs (in the last stage's buffer type) too: freed here, they
        // stalled the same way.
        for g in &mut self.par_mtp {
            graphs.push(Box::new(g.mixed.take()));
        }
        if self.stages.last().is_some_and(|s| s.backend.is_meta()) {
            self.mtp_mixed.clear();
        } else {
            graphs.push(Box::new(self.mtp_mixed.take()));
            // The next take-in passes allocate from the buffer kept through the lend (#491).
            if let Some(a) = self.mtp_spare.take() {
                self.mtp_mixed.start_with(a);
            }
        }
        let (doomed, freeing) = freeing::doomed(graphs);
        tracing::debug!(
            ms = t.elapsed().as_secs_f64() * 1e3,
            handed_off = doomed.len(),
            "prompt graphs dropped for the expert cache restore"
        );
        self.freeing = freeing;
        self.mixed_kept = MIXED_KEPT;
        self.model
            .experts_mut()
            .expect("checked above")
            .restore_leaving(move || drop(doomed), leave)
            .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}"))))
    }

    /// Refills the expert slots [`Self::restore_experts_leaving`] left without VRAM, while the
    /// cache is not lent (the steps' graphs stay: they look experts up in the tables).
    pub fn refill_experts(&mut self) -> Result<(), ModelError> {
        self.wait_commit();
        self.flush_deferred()?;
        match self.model.experts_mut() {
            Some(r) => r
                .refill_unbacked()
                .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}")))),
            None => Ok(()),
        }
    }

    /// Drops every stage's kept prompt and verify graphs and the MTP head's prompt-pass graph
    /// (and their compute buffers).
    fn drop_mixed(&mut self) {
        for s in &mut self.stages {
            s.mixed.clear();
            s.shared[1] = None;
        }
        self.mtp_mixed.clear();
        for g in &mut self.par_mtp {
            g.mixed.clear();
        }
    }

    /// Before a pass of `n` tokens through the layers `ran` selects: starts the expert cache's
    /// prefill staging copies.
    fn begin_pass(
        &mut self,
        n: usize,
        ran: impl Fn(usize) -> bool,
        kind: Kind,
    ) -> Result<(), ModelError> {
        let r = match self.model.experts_mut() {
            Some(r) => r
                .begin_pass(n, ran)
                .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}")))),
            None => return Ok(()),
        };
        self.credit_experts(kind);
        r
    }

    /// Counts the expert cache's traffic since the last call into the model bytes (#391): the
    /// target layers' as `kind` (the pass that just ran or was planned), the MTP head's as
    /// draft passes. A few additions per layer, no device work.
    fn credit_experts(&mut self, kind: Kind) {
        let Some(r) = self.model.experts() else {
            return;
        };
        let n = self.stages.len();
        if self.experts_seen.len() != n + 2 {
            self.experts_seen = vec![Default::default(); n + 2];
        }
        let n_layer = self.model.hp.n_layer;
        // The stages', the MTP head's (draft passes), then the copies of an expert-parallel
        // cache (`--glm-parallel-decode`; none otherwise), which run with the target's steps.
        for si in 0..n + 2 {
            let (layers, k) = match self.stages.get(si) {
                Some(s) => (s.layers.clone(), kind),
                None if si == n => (n_layer..n_layer + 1, Kind::Draft),
                None => (n_layer + 1..usize::MAX, kind),
            };
            let now = r.traffic_in(layers);
            let d = now.since(&self.experts_seen[si]);
            if d == Default::default() {
                continue;
            }
            self.experts_seen[si] = now;
            self.bytes.phase_mut(k).add_experts(&d);
            if si <= n
                && let Some(p) = self.bytes.stage_mut(si, k)
            {
                p.add_experts(&d);
            }
        }
    }

    /// After a pass of `n` tokens through the layers `ran` selects: lets the expert cache learn
    /// the routing and move experts (before the next pass is built or run).
    /// Lets the expert cache's work after a pass (reading its routing, planning swaps,
    /// writing tables: ~1 ms of host time per pass on Qwen3.8-Flash-Next) run while the GPUs
    /// compute a pass that reads other layers, instead of between passes with them idle: a
    /// target step's beside the next MTP pass, an MTP pass's beside the next target step. The
    /// cache's work for those runs confined to their layers (`after_step_only`), so a pass in
    /// flight never sees a slot or a table row of its layers change. Without an MTP head, over
    /// several stages whose cache devices are their own, a decode or verify step's work is
    /// split by stage (`after_split`): each stage's share beside the next stage's compute, the
    /// last stage's beside the next step's first stage.
    pub fn set_after_overlap(&mut self, on: bool) {
        self.after_overlap = on;
    }

    /// Whether a decode or verify step over several stages queues each next stage's residual
    /// hand-off and graph while the stage before it computes, instead of after the host waited
    /// for it ([`launch_ahead`], #814; byte-identical).
    pub fn set_launch_ahead(&mut self, on: bool) {
        self.launch_ahead = on;
    }

    /// Per stage, the target layers of a decode or verify step's cache work split by stage
    /// ([`after_split::stage_split`]); `None` without an expert cache or where it doesn't
    /// split.
    fn stage_split(&self) -> Option<Vec<Range<usize>>> {
        let r = self.model.experts()?;
        let stages: Vec<Range<usize>> = self.stages.iter().map(|s| s.layers.clone()).collect();
        after_split::stage_split(&stages, self.model.hp.n_layer, |il| r.layer_device(il))
    }

    /// The share of a step's cache work for `layers`, all but the last of the step's shares
    /// (`after_step_more`: the step counts in its last share, an [`Self::after_pass_only`]).
    fn after_pass_more(
        &mut self,
        n: usize,
        layers: Range<usize>,
        kind: Kind,
    ) -> Result<(), ModelError> {
        let ran = |il: usize| layers.contains(&il);
        let r = match self.model.experts_mut() {
            Some(r) => r
                .after_step_more(n, self.pair.widen(ran))
                .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}")))),
            None => return Ok(()),
        };
        self.pair.learned(ran);
        self.credit_experts(kind);
        r
    }

    /// Runs any deferred expert-cache work now (see [`Self::set_after_overlap`]).
    fn flush_deferred(&mut self) -> Result<(), ModelError> {
        self.flush_deferred_target()?;
        self.flush_deferred_last()
    }

    /// Runs the last target step's deferred cache work now, if any.
    fn flush_deferred_target(&mut self) -> Result<(), ModelError> {
        if let Some((n, kind)) = self.deferred_target.take() {
            let target = self.target_entries();
            self.after_pass_only(n, target, kind)?;
        }
        Ok(())
    }

    /// The expert cache's entries a target step reads: its layers', and under GLM's parallel
    /// steps also the other GPUs' copies of them (the entries past the MTP head's,
    /// [`crate::model::glm5next::par`]); not the MTP head's (nor its copies,
    /// [`Self::mtp_entries`]).
    fn target_entries(&self) -> impl Fn(usize) -> bool + use<> {
        let n_layer = self.model.hp.n_layer;
        let par = self.par.every_step;
        let copies = self.mtp_copy_entries();
        move |il| {
            if par {
                !par_mtp::is_mtp_entry(il, n_layer, &copies)
            } else {
                il < n_layer
            }
        }
    }

    /// Runs the deferred work of the last stage's device now: the target step's share left
    /// for the last stage's layers, then the last MTP pass's (their order without the split).
    fn flush_deferred_last(&mut self) -> Result<(), ModelError> {
        self.flush_deferred_tail()?;
        self.flush_deferred_mtp()
    }

    /// Runs the share of a target step's cache work left for the last stage's layers now
    /// ([`mtp_window::tail_split`]).
    fn flush_deferred_tail(&mut self) -> Result<(), ModelError> {
        let Some((n, kind, layers)) = self.deferred_tail.take() else {
            return Ok(());
        };
        // The step's last share: it counts the step (and logs the cache's report with all of
        // the step's work in it, as one call over every layer would).
        self.after_pass_only(n, |il| layers.contains(&il), kind)
    }

    /// The target step's deferred cache work to run beside an MTP pass, as `(tokens, kind,
    /// layers, last)`: a step's work not started yet (all its layers, or those before the
    /// last stage's when it splits, [`mtp_window::tail_split`], leaving the rest for the next
    /// pass or step), else the share left for the last stage's layers. `last`: the step's
    /// work ends with this share, which then counts the step as logged.
    #[allow(clippy::type_complexity)]
    fn take_target_work(
        &mut self,
    ) -> Result<Option<(usize, Kind, Range<usize>, bool)>, ModelError> {
        let Some((n, kind)) = self.deferred_target.take() else {
            return Ok(self.deferred_tail.take().map(|(n, k, l)| (n, k, l, true)));
        };
        // An earlier step's share first (a step runs it before its own work is deferred).
        self.flush_deferred_tail()?;
        let n_layer = self.model.hp.n_layer;
        if self.par.every_step {
            // Every GPU runs every layer: no split, the whole step's work beside this pass
            // (`target_entries`: the copies past the MTP head's entry too).
            return Ok(Some((n, kind, 0..usize::MAX, true)));
        }
        let tail = self.model.experts().and_then(|r| {
            let stages: Vec<Range<usize>> = self.stages.iter().map(|s| s.layers.clone()).collect();
            mtp_window::tail_split(&stages, n_layer, |il| r.layer_device(il))
        });
        let head = 0..tail.as_ref().map_or(n_layer, |t| t.start);
        let last = tail.is_none();
        if let Some(t) = tail {
            self.deferred_tail = Some((n, kind, t));
        }
        Ok(Some((n, kind, head, last)))
    }

    fn flush_deferred_mtp(&mut self) -> Result<(), ModelError> {
        if let Some(n) = self.deferred_mtp.take() {
            let draft = self.mtp_entries();
            self.after_pass_only(n, draft, Kind::Draft)?;
        }
        Ok(())
    }

    /// [`Self::after_pass`] confined to the layers `ran` selects (`after_step_only`).
    fn after_pass_only(
        &mut self,
        n: usize,
        ran: impl Fn(usize) -> bool,
        kind: Kind,
    ) -> Result<(), ModelError> {
        let r = match self.model.experts_mut() {
            Some(r) => r
                .after_step_only(n, self.pair.widen(&ran))
                .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}")))),
            None => return Ok(()),
        };
        self.pair.learned(ran);
        self.credit_experts(kind);
        r
    }

    /// `kind`: the pass's kind of step, for its model bytes ([`Self::credit_experts`]).
    fn after_pass(
        &mut self,
        n: usize,
        ran: impl Fn(usize) -> bool,
        kind: Kind,
    ) -> Result<(), ModelError> {
        let r = match self.model.experts_mut() {
            Some(r) => r
                .after_step(n, self.pair.widen(&ran))
                .map_err(|e| ModelError::Ggml(GgmlError::External(format!("expert cache: {e}")))),
            None => return Ok(()),
        };
        self.pair.learned(ran);
        self.credit_experts(kind);
        r
    }

    /// Resets slot `slot` of the running states to the empty state.
    pub fn clear_slot(&mut self, slot: usize) {
        self.wait_commit();
        self.assert_committed();
        for s in &self.stages {
            if let Some(p) = s.spec.as_ref() {
                p.drop_deferred(slot);
            }
            s.rec.clear_slot(slot);
        }
        if let Some(r) = self.remote.as_mut() {
            r.mirror(crate::node::Request::ClearSlot { slot: slot as u32 });
        }
        self.tp_mirror_states(|| crate::node::Request::ClearSlot { slot: slot as u32 });
    }

    /// Copies recurrent states between slots, in order: `(from pool, slot, to pool, slot)`.
    pub fn copy_states(&mut self, copies: &[(Pool, usize, Pool, usize)]) -> Result<(), GgmlError> {
        if !self.model.hp.has_slot_state() {
            return Ok(());
        }
        // Nothing to copy while a round's stages still compute (#762): the round's end waits
        // for the commits instead.
        if copies.is_empty() && self.pipe_run.is_some() {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let copied = self.copy_states_now(copies);
        if self.par.every_step {
            self.par.gap.add(par_step::GapPhase::States, t.elapsed());
        }
        copied
    }

    /// [`Self::copy_states`] of a model with recurrent states.
    fn copy_states_now(&mut self, copies: &[(Pool, usize, Pool, usize)]) -> Result<(), GgmlError> {
        // A running state copied from or over waits for its deferred commit.
        let deferred = copies.iter().any(|&(a, x, b, y)| {
            self.stages.iter().any(|s| {
                s.spec.as_ref().is_some_and(|p| {
                    (a == Pool::Seq && p.defers_slots(x..x + 1))
                        || (b == Pool::Seq && p.defers_slots(y..y + 1))
                })
            })
        });
        if deferred {
            self.flush_commits()?;
        }
        self.wait_commit();
        self.assert_committed();
        let pool = |p| u8::from(p == Pool::Saved);
        let wire = || crate::node::Request::CopyStates {
            copies: copies
                .iter()
                .map(|&(a, x, b, y)| (pool(a), x as u32, pool(b), y as u32))
                .collect(),
        };
        if let Some(r) = self.remote.as_mut() {
            r.mirror(wire());
        }
        self.tp_mirror_states(wire);
        for s in &mut self.stages {
            let pool = |p| match p {
                Pool::Seq => &s.rec,
                Pool::Saved => &s.saved,
            };
            let list: Vec<_> = copies
                .iter()
                .map(|&(a, x, b, y)| (pool(a), x, pool(b), y))
                .collect();
            StatePool::copy_slots(&mut s.backend, &list)?;
        }
        Ok(())
    }
}
