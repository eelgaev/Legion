//! The real engine for Qwen3.5 / 3.6 GGUF models (`qwen35`, `qwen35moe`) and Qwen3.8-Flash-Next
//! (`qwen4exp`) on the CPU or one GPU.
//!
//! [`LmEngine::generate`] renders and tokenizes the prompt on the caller's task (so
//! template errors and context overflows come back as request errors), then queues the request
//! for the worker thread. The worker owns the model, the backend, the compute allocator and the
//! decode state; it prefills the prompt (in chunks of [`PREFILL_CHUNK`] tokens) and then
//! decodes one token per step, streaming [`TokenEvent`]s back.
//!
//! The worker runs continuous batching (#34). Every step it admits waiting requests (in the
//! order of [`super::queue`]: FCFS by default) while there
//! are sequence slots and KV blocks for them, then runs one forward pass over every decoding
//! sequence (one token each) plus prefill chunks of admitted prompts, up to
//! `max_batch_tokens` tokens in all, and hands each sequence its row of logits. Attention K/V
//! live in a block pool ([`crate::kv`]); a prompt that starts with a cached prefix shares its
//! blocks and resumes from a saved recurrent state ([`super::saved_states`]). Decoding
//! sequences keep their recurrent states in slots `0..n`, so a decode step's DeltaNet layers
//! run as one `n`-sequence op and repeat the same graph (see [`Executor`]).
//!
//! Per-sequence work ([`Active`]: sampler, detokenizer, reasoning / tool-call / stop parsers,
//! counters) is separate from the model step.
//!
//! With speculative decoding on ([`LmConfig::spec`], see [`crate::spec`]), decoding
//! sequences feed their last token plus drafts as one verify group; each sequence accepts a
//! prefix of its drafts plus a bonus token, trims its KV block table, and its recurrent state
//! is advanced by the kept tokens only ([`Executor::commit`]).

mod adaptive;
mod bandwidth;
mod captures;
mod chat_format;
mod even;
mod first_rows;
mod fit;
mod host_tier;
mod host_time;
mod kv_vmm;
mod layout;
mod mtp_ngram;
mod node;
mod pair_tp;
mod pipe;
pub mod pipe_split;
mod plan;
mod preempt;
pub mod report;
mod response_format;
mod slo;
mod slot_backing;
mod spec;
mod ssd_tier;
mod stream;
mod tp_node;
pub mod tp_pair;
mod vision;
mod vram;
mod waves;
use chat_format::ChatFormat;
use response_format::{Constraint, GrammarCache, MaskPool};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;

use super::queue::{Scheduling, SchedulingPolicy, Ticket, WaitQueue};
use super::ring_checkpoints::{self, RingCheckpoints};
use super::sampling::Sampler;
use super::saved_states::{Kind as SaveKind, SavedStates, save_points};
use super::stop::StopMatcher;
use super::{
    Engine, EngineError, EngineStats, FinishReason, GenerateRequest, ModelInfo, Prompt, Role,
    SamplingParams, TokenEvent, TokenStream, Usage,
};
use crate::chat_template::ChatTemplate;
use crate::gguf::GgufFile;
use crate::kv::{BlockPool, BlockTable, radix::RadixCache};
use crate::model::core::{
    Executor, Group, KV_BLOCK, KvPool, KvRing, Logits, Media, Model, Pool, PromptNext, SeqInput,
    SpecPool, StatePool,
};
use crate::residency::ExpertMode;
use crate::residency::fallback::known_node;
use crate::spec::accept;
use crate::spec::{
    DeviceHidden, Draft, DraftRequest, SeqId, SpecConfig, SpecCounters, TargetInfo, TokenDrafter,
};
use crate::tokenizer::Tokenizer;
pub use pipe::DEFAULT_MIN_GROUP as DEFAULT_PIPELINE_MIN_GROUP;
pub use preempt::KvAdmission;
use preempt::Waiting;
pub use slo::SloTargets;
pub use spec::set_take_in_sorted as set_mtp_take_in_sorted;
use spec::{Drafting, SeqTarget, draft_cap};

/// Default [`LmConfig::pipeline_prompt`]: eight 512-token micro-chunks per step.
pub const PIPELINE_PROMPT: usize = 4096;

/// Prompt tokens per prefill graph. Bounds the attention scores buffer
/// (`n_kv × chunk × n_head` floats) for long prompts.
pub const PREFILL_CHUNK: usize = 512;

/// `--ring-checkpoint-every` by default.
pub const RING_CHECKPOINT_EVERY: usize = 1024;

/// `--ring-checkpoint-span` by default.
pub const RING_CHECKPOINT_SPAN: usize = 8192;

/// Tokens per forward pass (`--max-batch-tokens`) by default.
pub const MAX_BATCH_TOKENS: usize = 1024;

/// Default [`LmConfig::grammar_threads`]: a cached mask is a lookup, a new grammar state a
/// walk of the vocabulary's trie; two threads keep a few such walks per step off the sampling.
pub const GRAMMAR_THREADS: usize = 2;

/// The default on a GPU that shares its memory with the host (GB10; the owner's choice): a
/// 4K-token prompt chunk reads each routed expert once instead of once per 512 tokens.
/// Qwen3.8-Flash-Next on the GB10 prefilled ~2x faster (with ggml's MMQ experts); 8192 measured
/// the same speed with twice the step buffers. Greedy output differs from 1024-token steps
/// for prompts longer than a chunk (other step sums).
pub const MAX_BATCH_TOKENS_INTEGRATED: usize = 4096;

/// `--max-batch-tokens` when none is given; `integrated`: the engine's GPU shares the host's
/// memory ([`integrated_gpu`]).
pub fn default_max_batch_tokens(device: DeviceChoice, integrated: bool) -> usize {
    match device {
        DeviceChoice::Cuda if integrated => MAX_BATCH_TOKENS_INTEGRATED,
        _ => MAX_BATCH_TOKENS,
    }
}

/// `--prefill-chunk` when none is given: [`PREFILL_CHUNK`], or on an integrated GPU the whole
/// step (sliding-window rings already hold whole steps there, see [`LmConfig::ring_step`]).
pub fn default_prefill_chunk(
    device: DeviceChoice,
    integrated: bool,
    max_batch_tokens: usize,
) -> usize {
    match device {
        DeviceChoice::Cuda if integrated => max_batch_tokens.max(PREFILL_CHUNK),
        _ => PREFILL_CHUNK,
    }
}

/// Whether `--device cuda` runs on an integrated GPU: there is no discrete one, but an
/// integrated one (the engine's own choice, see `layout`).
pub fn integrated_gpu() -> bool {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu).is_none() && device_by_kind(DeviceKind::IntegratedGpu).is_some()
}

/// Whether the load drops the model files' pages from the page cache (#836): on an integrated
/// GPU, unless `--keep-load-cache`. There the weights, the KV cache and the step buffers come
/// out of the same memory as the page cache; allocated while tens of GiB of it sit in memory,
/// they get scattered pages, and about one server start in three ran every prompt kernel
/// 5-15% slower (an 8K-token prompt 4.2 -> 4.6-5.4 s on a DGX Spark). Freed first, they get
/// unfragmented memory. Bytes read later come from the files again: the same output.
pub fn release_load_cache(keep: bool, kind: Option<crate::ggml::DeviceKind>) -> bool {
    !keep && kind == Some(crate::ggml::DeviceKind::IntegratedGpu)
}

/// What `--moe-kernel auto` runs for the routed experts of big steps. The fused op
/// ([`crate::model::moe_ffn`]) on its `mma.m16n8k16` GEMMs (`mma16`: sm_80+) beats ggml's MMQ
/// `mul_mat_id` on the GB10 (Qwen3.8-Flash-Next, 8K-token prompt: 1,062 -> 1,260 tok/s), and
/// `auto` keeps it. Where the op only has Volta's `mma.m8n8k4`, which GPUs from sm_80 on emulate,
/// a GPU sharing the host's memory with every expert in device memory takes ggml's MMQ instead
/// (GB10: 457 vs 744 tok/s). Discrete GPUs keep the op until measured; experts in host memory
/// keep it (it stages them into VRAM). `integrated`, `cc_major`: the engine's GPU.
pub fn resolve_moe_kernel(
    kernel: crate::model::core::MoeKernel,
    mode: ExpertMode,
    integrated: bool,
    cc_major: Option<i32>,
    mma16: bool,
) -> crate::model::core::MoeKernel {
    use crate::model::core::MoeKernel;
    match (kernel, mode, cc_major) {
        (MoeKernel::Auto, ExpertMode::Device, Some(cc)) if integrated && cc >= 8 && !mma16 => {
            MoeKernel::Graph
        }
        (k, _, _) => k,
    }
}

/// Whether `dev`'s step compute buffers can spill into host memory (unified memory).
#[cfg(feature = "cuda")]
fn spills(dev: crate::ggml::Device) -> bool {
    cuda_index(dev)
        .and_then(crate::residency::cuda::spill_type)
        .is_some()
}

#[cfg(not(feature = "cuda"))]
fn spills(_: crate::ggml::Device) -> bool {
    false
}

/// The CUDA index of a ggml CUDA device (`CUDA<n>`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn cuda_index(dev: crate::ggml::Device) -> Option<usize> {
    dev.name().strip_prefix("CUDA")?.parse().ok()
}

/// Refuses a layout with a GPU this binary has no code of its architecture's level for
/// ([`crate::hw::cuda_arch::check`]): the stages' devices and tensor parallelism's GPUs.
#[cfg(feature = "cuda")]
fn check_gpu_code(layout: &layout::Layout) -> Result<(), LoadError> {
    for dev in layout_gpus(layout) {
        if let Some(index) = cuda_index(dev)
            && let Some(e) = crate::hw::cuda_arch::check(index, || dev.description())
        {
            return Err(LoadError::Device(e));
        }
    }
    Ok(())
}

/// The layout's GPUs: the stages' devices and tensor parallelism's.
#[cfg(feature = "cuda")]
fn layout_gpus(layout: &layout::Layout) -> impl Iterator<Item = crate::ggml::Device> + '_ {
    layout
        .stages
        .iter()
        .filter_map(|s| s.0)
        .chain(layout.tp_devices.iter().copied())
}

/// Warns, once per GPU of the layout, when `--nvfp4-a4` asks for the W4A4 kernels and this
/// build has no FP4 code for that GPU ([`crate::model::nvfp4_a4::supported_on`]; an sm_121a
/// binary on an RTX 50, say, #1200): its NVFP4 matmuls stay W4A16 there.
#[cfg(feature = "cuda")]
fn warn_nvfp4_a4(layout: &layout::Layout, a4: crate::model::core::Nvfp4A4) {
    if a4 == crate::model::core::Nvfp4A4::Off {
        return;
    }
    let mut warned = std::collections::BTreeSet::new();
    for dev in layout_gpus(layout) {
        if let Some(index) = cuda_index(dev)
            && warned.insert(index)
            && !crate::model::nvfp4_a4::supported_on(index)
        {
            tracing::warn!(
                gpu = index,
                device = %dev.description(),
                "--nvfp4-a4: W4A4 kernels not in this build for this GPU, using W4A16"
            );
        }
    }
}

/// Whether any of `devs` (a meta device's GPUs) is a V100 (compute capability 7.0).
#[cfg(feature = "cuda")]
fn any_volta(devs: &[crate::ggml::Device]) -> bool {
    devs.iter()
        .filter_map(|&d| cuda_index(d))
        .filter_map(crate::residency::cuda::caps)
        .any(|c| (c.cc_major, c.cc_minor) == (7, 0))
}

#[cfg(not(feature = "cuda"))]
fn any_volta(_: &[crate::ggml::Device]) -> bool {
    false
}

/// VRAM below which a one-GPU or layer-split stage's decode state spills (#1131): it would not
/// fit at all. Under tensor parallelism it spills once it takes the reserve the steps need
/// (their buffers cannot spill there).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const STATE_SPILL_SLACK: usize = 512 << 20;

/// Whether stage `si`'s decode state spills into host memory (#1131), and its KV pool's
/// buffer type then: the state (`state.0` bytes over the stage's GPUs, `state.1` of them its
/// sliding-window rings) on discrete GPUs, when it does not fit beside the weights and
/// `reserve` (`tp`: tensor parallelism, one meta device or a `--tp-per-gpu` rank) or at all.
/// The rings (and the pool's blocks without a host tier) then go in VRAM as far as the reserve
/// and the rest of the state stay free, past that in pinned memory on each GPU's host NUMA
/// node, which the GPU reads over the link. `None` where it fits: as before.
#[cfg(feature = "cuda")]
fn state_spill(
    layout: &layout::Layout,
    hp: &crate::model::core::Hparams,
    si: usize,
    dev: crate::ggml::Device,
    (state, rings): (usize, usize),
    reserve: usize,
    tp: bool,
) -> Result<Option<crate::ggml::BufferType>, String> {
    let gpus = if layout.tp_devices.is_empty() {
        vec![dev]
    } else {
        layout.tp_devices.clone()
    };
    if !spill_needed(
        (state, rings),
        layout.free_for_split(si, 0),
        reserve,
        gpus.len(),
        tp,
    ) || gpus
        .iter()
        .any(|d| d.kind() != crate::ggml::DeviceKind::Gpu)
    {
        return Ok(None);
    }
    let n_dev = gpus.len();
    let keep = reserve + (state - rings) / n_dev;
    let buft = spill_buft(layout, hp, &gpus, keep)?;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    let free = layout.free_for_split(si, 0) / n_dev;
    tracing::warn!(
        state_gib_per_gpu = gib(state / n_dev),
        rings_gib_per_gpu = gib(rings / n_dev),
        free_gib_per_gpu = gib(free),
        reserve_gib = gib(reserve),
        "the decode state does not fit in GPU memory beside the weights and the reserve: the \
         sliding-window rings past {:.2} GiB per GPU go to pinned host memory, read over the \
         link (slower decode); fewer --max-seqs keep them in VRAM",
        gib(free.saturating_sub(keep))
    );
    Ok(Some(buft))
}

/// The buffer type of a decode state spilled over `gpus` ([`state_spill`]), leaving `keep`
/// bytes free per GPU: one GPU's, or a meta device's over the stage's tensor-parallel GPUs.
#[cfg(feature = "cuda")]
fn spill_buft(
    layout: &layout::Layout,
    hp: &crate::model::core::Hparams,
    gpus: &[crate::ggml::Device],
    keep: usize,
) -> Result<crate::ggml::BufferType, String> {
    let idx = gpus
        .iter()
        .map(|d| cuda_index(*d).ok_or_else(|| format!("{} is not a CUDA GPU", d.name())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(if layout.tp_devices.is_empty() {
        crate::residency::cuda::spill_type_keeping(idx[0], keep).ok_or_else(|| {
            format!(
                "CUDA{}'s driver cannot map host NUMA memory (cuMemCreate)",
                idx[0]
            )
        })?
    } else {
        let wrapped = crate::residency::cuda::spill_devices(&idx, keep)?;
        crate::ggml::meta::meta_device(
            &wrapped,
            crate::model::core::tp::megatron_rule(hp, gpus.len()),
        )
        .ok_or("cannot create the tensor-parallel decode state's device")?
        .buffer_type()
    })
}

#[cfg(not(feature = "cuda"))]
fn state_spill(
    _: &layout::Layout,
    _: &crate::model::core::Hparams,
    _: usize,
    _: crate::ggml::Device,
    _: (usize, usize),
    _: usize,
    _: bool,
) -> Result<Option<crate::ggml::BufferType>, String> {
    Ok(None)
}

/// Whether a decode state of `state.0` bytes (`state.1` of them rings) over `n_dev` GPUs with
/// `free` bytes free between them spills ([`state_spill`]): only rings spill, so not without
/// them; under tensor parallelism (`tp`) once it leaves less than `reserve` per GPU, else once
/// it leaves less than [`STATE_SPILL_SLACK`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn spill_needed(
    (state, rings): (usize, usize),
    free: usize,
    reserve: usize,
    n_dev: usize,
    tp: bool,
) -> bool {
    let keep = if tp { reserve } else { STATE_SPILL_SLACK };
    rings > 0 && state > free.saturating_sub(keep * n_dev.max(1))
}

/// Blocks a slotted pool hands out at once over `n_seqs` lazily backed slots (#1131) whose
/// host tier backs `held` blocks: each slot is backed up to its blocks rounded up to a whole
/// chunk ([`crate::residency::cuda::LAZY_CHUNK_ROWS`] rows), so `n_seqs` chunks of the room go
/// to the rounding. Counts a slot's VRAM blocks as host ones (a bound, never short).
pub(crate) fn lazy_slot_cap(held: usize, n_seqs: usize) -> usize {
    held.saturating_sub(n_seqs * LAZY_CHUNK_BLOCKS)
}

/// [`crate::residency::cuda::LAZY_CHUNK_ROWS`] in blocks.
pub(crate) const LAZY_CHUNK_BLOCKS: usize = 4096 / KV_BLOCK;

/// VRAM per GPU kept free past the steps' measured buffers and their margin when the slots'
/// shared chunks take what the load left (#1159).
const LEND_SAFETY: usize = 512 << 20;
#[cfg(feature = "cuda")]
const _: () = assert!(LAZY_CHUNK_BLOCKS * KV_BLOCK == crate::residency::cuda::LAZY_CHUNK_ROWS);

/// Gives a tensor-parallel stage's spilled decode state ([`state_spill`]) a new share of VRAM:
/// what leaves `keep` bytes per GPU free, from the next [`Executor::resize_kv`] on. False when
/// its state did not spill.
#[cfg(feature = "cuda")]
fn respill_tp(exec: &mut Executor, layout: &layout::Layout, keep: usize) -> Result<bool, String> {
    if layout.tp_devices.is_empty() || !exec.kv_spilled(0) {
        return Ok(false);
    }
    let buft = spill_buft(layout, exec.hp(), &layout.tp_devices, keep)?;
    tracing::info!(
        keep_mib = keep >> 20,
        "the decode state's rings leave the step buffers' room in VRAM"
    );
    exec.set_kv_spill(0, buft);
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
fn respill_tp(_: &mut Executor, _: &layout::Layout, _: usize) -> Result<bool, String> {
    Ok(false)
}

/// What `--mtp-window n` comes to for the head: the window it takes (0: none) and, when the
/// flag does not apply or does little, why (a warning).
#[derive(Debug, PartialEq)]
pub(crate) struct MtpWindow {
    pub window: usize,
    pub note: Option<String>,
}

/// `--mtp-window n` for an MTP head of architecture `arch`: it windows the head's dense paged
/// attention (`paged`), not Gemma 4's assistant (which reads the target's cache), not a gather
/// path, not a sparse (QSA) head layer (`head_sparse`); a window of the whole context
/// (`ctx_size`) changes nothing; one below a KV block (64) is allowed but noted.
pub(crate) fn mtp_window_check(
    n: usize,
    arch: &str,
    paged: bool,
    head_sparse: bool,
    ctx_size: usize,
) -> MtpWindow {
    let off = |why: String| MtpWindow {
        window: 0,
        note: Some(why),
    };
    if n == 0 {
        return MtpWindow {
            window: 0,
            note: None,
        };
    }
    if arch == "gemma4" {
        return off(
            "--mtp-window does not apply to Gemma 4's assistant (it reads the target's \
                    cache): ignored"
                .into(),
        );
    }
    if !paged {
        return off(
            "--mtp-window needs the MTP head's paged attention (this run gathers: \
                    --attention gather, a host KV cache, a cache type the paged kernels cannot \
                    read, or --tp > 1): ignored"
                .into(),
        );
    }
    if head_sparse {
        return off("--mtp-window: this MTP head's attention is sparse (QSA): ignored".into());
    }
    if n >= ctx_size {
        return off(format!(
            "--mtp-window {n} covers the whole {ctx_size}-token context: no effect"
        ));
    }
    MtpWindow {
        window: n,
        note: (n < KV_BLOCK).then(|| {
            format!(
                "--mtp-window {n} is under one KV block ({KV_BLOCK}): the head sees very little"
            )
        }),
    }
}

/// The draft ceiling where `--spec-k` is not given: the architecture's (`arch_default`), else 4.
/// qwen4exp on an integrated GPU (the GB10) drafts up to 8: under the depth policy that gained
/// on code, lists and recall (+5-10%) and left prose and the chats flat (#387).
///
/// qwen35 / qwen35moe drafting with their MTP head on an integrated GPU draft up to 8 too: the
/// verify step of the dense Qwen3.8-27B reads its 21 GB whatever its rows, so deep drafts are
/// cheap there (GB10, k=4 -> 8: code 37.2 -> 44.6, recall 45.1 -> 73.2 tok/s, prose flat).
pub(crate) fn default_spec_k(
    arch: &str,
    arch_default: Option<usize>,
    integrated: bool,
    mtp: bool,
) -> usize {
    if integrated && (arch == "qwen4exp" || (mtp && matches!(arch, "qwen35" | "qwen35moe"))) {
        8
    } else {
        arch_default.unwrap_or(4)
    }
}

/// Whether the MTP head drafts with Q8_0 copies of its wide matrices: `--spec-draft-head-type`,
/// with `auto` [`default_mtp_head_q8`]. `wide`: the head ships its dense matrices 16-bit
/// ([`mtp_head_wide`]).
pub(crate) fn draft_head_q8(
    arg: crate::model::core::DraftHeadType,
    arch: &str,
    integrated: bool,
    wide: bool,
) -> bool {
    use crate::model::core::DraftHeadType;
    match arg {
        DraftHeadType::Q8_0 => true,
        DraftHeadType::Model | DraftHeadType::F16 => false,
        DraftHeadType::Auto => default_mtp_head_q8(arch, integrated, wide),
    }
}

/// The type the MTP head's wide matrices are copied to (`--spec-draft-head-type`): F16 for
/// `f16`, Q8_0 where [`draft_head_q8`] says so, else none (the head as shipped).
pub(crate) fn draft_head_copy(
    arg: crate::model::core::DraftHeadType,
    arch: &str,
    integrated: bool,
    wide: bool,
) -> Option<crate::ggml::ffi::ggml_type> {
    if arg == crate::model::core::DraftHeadType::F16 {
        Some(crate::ggml::types::F16)
    } else {
        draft_head_q8(arg, arch, integrated, wide).then_some(crate::ggml::types::Q8_0)
    }
}

/// Whether `--spec-draft-head-type auto` drafts with the Q8_0 copy, on an integrated GPU only:
/// - qwen35 / qwen35moe, whose Qwen3.8-27B head is BF16 (810 MiB a draft pass, 430 as Q8_0;
///   GB10, k=4: cycle 108-113 -> 102-106 ms, code 33.7 -> 37.2, prose 26.2 -> 28.8 tok/s,
///   accepted length unchanged);
/// - qwen4exp when the head is `wide`: the NVFP4 checkpoint's own head (BF16 dense parts, #823;
///   175 -> 92 MiB a draft pass). The released Q8_0 head GGUF is not wide and stays as stored.
///
/// Drafts only: the output stays exact.
pub(crate) fn default_mtp_head_q8(arch: &str, integrated: bool, wide: bool) -> bool {
    integrated && (matches!(arch, "qwen35" | "qwen35moe") || (arch == "qwen4exp" && wide))
}

/// Whether a type is a 16- or 32-bit float one: a matrix [`default_mtp_head_q8`] /
/// [`mtp_lm_head`] would copy to fewer bytes.
pub(crate) fn wide_float(ty: crate::ggml::ffi::ggml_type) -> bool {
    use crate::ggml::types;
    [types::BF16, types::F16, types::F32].contains(&ty)
}

/// Whether the MTP head of `model` ships its dense matrices 16- or 32-bit: its `eh_proj`
/// (qwen4exp's NVFP4 checkpoint: BF16; the released head GGUF: Q8_0).
pub(crate) fn mtp_head_wide(model: &crate::model::core::Model) -> bool {
    let name = format!("blk.{}.nextn.eh_proj.weight", model.hp.n_layer);
    model.w(&name).is_ok_and(|t| wide_float(t.ty()))
}

/// The output head the MTP head drafts with (`--spec-draft-lm-head`): `auto` is `q4` for the
/// qwen4exp head on an integrated GPU when the target's output head is `wide_output` (16- or
/// 32-bit: the NVFP4 checkpoint's BF16 head, 65,536 draft rows 320 -> 90 MiB a draft pass; on
/// one GB10 the draft passes' head time went 2.65 -> 0.75 ms a token, #823), else `model`.
/// Drafts only: every token is verified with the target's own head.
pub(crate) fn mtp_lm_head(
    arg: crate::spec::DraftLmHead,
    arch: &str,
    integrated: bool,
    wide_output: bool,
) -> crate::spec::DraftLmHead {
    use crate::spec::DraftLmHead;
    match arg {
        DraftLmHead::Auto if integrated && arch == "qwen4exp" && wide_output => DraftLmHead::Q4,
        DraftLmHead::Auto => DraftLmHead::Model,
        h => h,
    }
}

/// Kept graph shapes where `--mtp-kept-graphs` / `--verify-kept-graphs` are not given (#858):
/// `(mtp, verify)`, or `None` for the engine-wide defaults (16 and 0). qwen4exp drafting with its
/// MTP head on an integrated GPU drafts up to 8 tokens ([`default_spec_k`]), so 4 sequences take
/// in 2-9 rows each: ~100 take-in shapes in a C=4 run, and ~30 verify shapes (pool counts) that
/// the stage's 3 shared mixed graphs cycled through. Kept, they stop being rebuilt and captured
/// again (GB10, det C=4 run: ~60 MTP + ~20 target builds per 10 s -> 0). The MTP head's kept
/// graphs share one buffer; each kept verify graph has its own (~35-55 MiB on the GB10).
/// The same kernels: the same bits.
pub(crate) fn default_kept_graphs(
    arch: &str,
    mtp: bool,
    integrated: bool,
) -> Option<(usize, usize)> {
    (mtp && integrated && arch == "qwen4exp").then_some((256, 32))
}

/// The bytes a stage's kept verify graphs may hold ([`crate::model::core::Executor::set_verify_budget`]):
/// `--verify-kept-mib` when given (0: no limit), else
/// [`crate::model::core::default_verify_budget`] of the stage device's `free` bytes after the
/// load (#858: 32 kept verify graphs at 16 sequences grew past the 1.4 GiB they took at 4).
pub(crate) fn verify_budget(mib: Option<usize>, free: usize) -> usize {
    match mib {
        Some(m) => m << 20,
        None => crate::model::core::default_verify_budget(free),
    }
}

/// The MTP head's attention window where `--mtp-window` is not given: the last 32,768
/// positions for qwen4exp's head on an integrated GPU (the GB10: at 64K depth the head's dense
/// attention cost 1.4 ms a pass, and the 64K grid cells went 34.1 / 35.1 -> 42.4 / 40.0 tok/s
/// with it; Strata's default too; verification stays exact), else every position (0).
pub(crate) fn default_mtp_window(arch: &str, mtp: bool, integrated: bool) -> usize {
    if mtp && integrated && arch == "qwen4exp" {
        32768
    } else {
        0
    }
}

/// Why `--spec-draft-vocab n` cannot apply, if it cannot: it is for the MTP heads of qwen4exp
/// and qwen35 and for DFlash drafters (`dflash`), and below the vocabulary's size.
pub(crate) fn draft_vocab_unused(
    n: usize,
    mtp: bool,
    dflash: bool,
    arch: &str,
    n_vocab: i64,
) -> Option<String> {
    if n == 0 {
        None
    } else if !dflash && (!mtp || !matches!(arch, "qwen4exp" | "qwen35" | "qwen35moe")) {
        Some(format!(
            "--spec-draft-vocab applies to the MTP heads of qwen4exp and qwen35 (--spec mtp) \
             and to DFlash drafters (--spec dflash) only, not {arch}: ignored"
        ))
    } else if n as i64 >= n_vocab {
        Some(format!(
            "--spec-draft-vocab {n} covers the whole {n_vocab}-token vocabulary: no effect"
        ))
    } else {
        None
    }
}

/// Why `--spec-draft-lm-head q4` does not apply to the MTP head of `arch`, if it does not: only
/// the qwen4exp and qwen35 heads draft with the copy.
pub(crate) fn mtp_lm_head_unused(h: crate::spec::DraftLmHead, arch: &str) -> Option<String> {
    (h == crate::spec::DraftLmHead::Q4 && !matches!(arch, "qwen4exp" | "qwen35" | "qwen35moe"))
        .then(|| {
            format!(
                "--spec-draft-lm-head applies to the MTP heads of qwen4exp and qwen35 only, \
                 not {arch}: ignored"
            )
        })
}

/// Whether the model's own MTP head drafts: `--spec mtp` without a draft model or token drafter.
/// Decided before `--spec-k` gets its default, so `k` 0 means "not given" here (every default
/// is above 0), not "no drafting".
pub fn drafts_with_mtp_head(spec: &SpecConfig) -> bool {
    spec.kind == crate::spec::SpecKind::Mtp
        && spec.model_drafter.is_none()
        && spec.token_drafter.is_none()
}

/// Why `--glm-parallel-decode` cannot draft as `spec` asks (none when it can): no drafting, or
/// GLM's own MTP head (on the last GPU, #669). Other drafters are not wired to the parallel
/// steps.
pub(crate) fn glm_par_spec_refusal(spec: &SpecConfig) -> Option<&'static str> {
    (spec.kind != crate::spec::SpecKind::None && !drafts_with_mtp_head(spec)).then_some(
        "--glm-parallel-decode drafts with the model's MTP head only (--spec mtp) or not at all",
    )
}

/// The most logits rows a parallel step of `max_seqs` decoding sequences outputs: a row per
/// sequence (and prompt slot), times `verify_k + 1` with the MTP head verifying up to
/// `verify_k` drafts; never more than a step's tokens.
pub(crate) fn glm_par_out_rows(
    max_seqs: usize,
    prefill_slots: usize,
    verify_k: Option<usize>,
) -> usize {
    ((max_seqs.max(1) + prefill_slots.max(1)) * (verify_k.unwrap_or(0) + 1))
        .min(crate::model::glm5next::par::MAX_TOKENS)
}

/// The tensors of a GGUF's own MTP layer (`blk.{n_layer}.*`) that a load leaves unread because
/// nothing drafts with them (`mtp` false): GLM-5.3-Flash (`glm5-next`) only, whose layer's
/// routed experts would otherwise take a share of the last GPU's expert cache, and its dense
/// weights VRAM there (#632). Other architectures load theirs as before.
pub(crate) fn unused_mtp_tensors<'a>(
    names: impl IntoIterator<Item = &'a str>,
    arch: &str,
    n_layer: usize,
    mtp: bool,
) -> Vec<String> {
    if mtp || arch != "glm5-next" {
        return Vec::new();
    }
    names
        .into_iter()
        .filter(|n| {
            n.strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|l| l.parse::<usize>().ok())
                .is_some_and(|l| l >= n_layer)
        })
        .map(str::to_owned)
        .collect()
}

/// Whether prefix caching's saved states go to pinned host memory (#441): with the expert cache
/// (`cache`, which takes the VRAM they would) on discrete GPUs, for a model with recurrent state
/// and saved slots, without tensor parallelism, unless `--prefix-states-vram` (`vram`).
pub(crate) fn saved_states_host(
    cache: bool,
    discrete: bool,
    saved_slots: usize,
    slot_state: bool,
    tp: usize,
    vram: Option<bool>,
) -> bool {
    cache && discrete && saved_slots > 0 && slot_state && tp <= 1 && vram != Some(true)
}

/// Recurrent-state slots for sequences being prefilled, above the decode slots: at most this
/// many prompts prefill at once ([`LmConfig::prefill_slots`]'s default).
pub const PREFILL_SLOTS: usize = 4;

/// [`LmConfig::stream_thread_seqs`]'s default.
pub const STREAM_THREAD_SEQS: usize = 8;

/// Device memory left free for compute buffers when the KV pool is sized automatically.
const COMPUTE_HEADROOM: usize = 2 << 30;

/// On a GPU that shares the host's memory (GB10), also left free when the KV pool is sized:
/// the OS and the engine's host side keep this much (`--uma-free-gib`'s default) ...
pub const UMA_HEADROOM: usize = 16 << 30;

/// ... and what comes after the pool: the compute buffers (decode, verify and MTP graphs, ggml's
/// pools) take a fixed part and big prompt steps more per 1024 tokens of `--max-batch-tokens`.
/// Measured on the GB10 with Qwen3.8-Flash-Next (262K-token pool, MTP): after the pool, free
/// memory fell another 9.7 GiB with 4096-token steps and 15.7 GiB with 8192 (an 8K prompt at
/// 64K depth), i.e. ~3.7 GiB + 1.5 GiB per 1K. With these figures 16 GiB stay free. Measured
/// at 64K depth on Qwen3.8-Flash-Next only: other architectures need re-measuring.
const UMA_STEP_FIXED: usize = 2 << 30;
const UMA_STEP_BYTES_PER_1K: usize = 3 << 29;

/// On such a GPU the automatic pool also holds at most this many tokens (or `--ctx-size`, if
/// more): the default `max_seqs x ctx_size` would take all of the memory (#301).
pub(crate) const UMA_KV_TOKENS: usize = 262_144;

/// The automatic KV pool's size before the memory check, in blocks: one context per
/// sequence, at most the larger of a context and [`CACHE_KV_TOKENS`] with the expert cache
/// (`cache`), or of a context and [`UMA_KV_TOKENS`] on a GPU sharing the host's memory.
pub(crate) fn auto_kv_want(max_seqs: usize, ctx_blocks: usize, cache: bool, shared: bool) -> usize {
    let mut want = max_seqs.saturating_mul(ctx_blocks);
    if cache {
        want = want.min(ctx_blocks.max(CACHE_KV_TOKENS.div_ceil(KV_BLOCK)));
    }
    if shared {
        want = want.min(ctx_blocks.max(UMA_KV_TOKENS.div_ceil(KV_BLOCK)));
    }
    want
}

/// A given KV pool (`--kv-cache-tokens`, `blocks`) against what fits next to the reserve
/// (`fit`; `None`: no GPU). On a GPU sharing the host's memory a pool past the reserve is an
/// error: the memory it takes is the OS's (#301). Elsewhere it is the user's call (a discrete
/// GPU fails its allocation instead).
pub(crate) fn kv_explicit(
    blocks: usize,
    fit: Option<usize>,
    shared: bool,
) -> Result<usize, String> {
    match fit {
        Some(f) if shared && blocks > f => Err(format!(
            "--kv-cache-tokens {} does not fit next to the memory this GPU (it shares the host's \
             memory) keeps free for the OS and the prompt steps: room for {} tokens",
            blocks * KV_BLOCK,
            f * KV_BLOCK
        )),
        _ => Ok(blocks),
    }
}

/// The KV pool's blocks given that `fit` blocks fit: at least a context (`ctx_blocks`). A
/// given size (`explicit`) below that is an error. An automatic one is raised to a context,
/// except on a GPU sharing the host's memory (`shared`): there a context that does not fit
/// next to the reserve is an error, since the memory it takes is the OS's (#301).
pub(crate) fn kv_floor(
    fit: usize,
    ctx_blocks: usize,
    explicit: bool,
    shared: bool,
    native: bool,
    types: crate::model::core::kv_type::KvTypes,
) -> Result<usize, String> {
    if fit >= ctx_blocks {
        return Ok(fit);
    }
    let tokens = |b: usize| b * KV_BLOCK;
    if explicit {
        return Err(format!(
            "--kv-cache-tokens must hold at least --ctx-size ({}) tokens",
            tokens(ctx_blocks)
        ));
    }
    if shared {
        return Err(format!(
            "a {}-token context does not fit next to the memory this GPU (it shares the host's \
             memory) keeps free for the OS and the prompt steps: room for {} tokens. Lower \
             --ctx-size, --max-batch-tokens or --uma-free-gib",
            tokens(ctx_blocks),
            tokens(fit)
        ));
    }
    if native {
        return Err(no_room_for_native(tokens(ctx_blocks), tokens(fit), types));
    }
    Ok(ctx_blocks)
}

/// The error when the KV cache cannot hold the model's native context of `native` tokens
/// (the default without --ctx-size): what fits, and the options.
pub(crate) fn no_room_for_native(
    native: usize,
    fits: usize,
    types: crate::model::core::kv_type::KvTypes,
) -> String {
    format!(
        "the model's native context of {native} tokens does not fit: the KV cache has room for \
         {fits} tokens. Options: --ctx-size {fits} (or less); {}more GPUs (--tp); a \
         host-memory KV tier (--kv-host-tokens; automatic on a discrete GPU whose driver maps \
         host NUMA memory)",
        quantized_option(types)
    )
}

/// The quantized-cache option of a context error, unless the cache is quantized already.
fn quantized_option(types: crate::model::core::kv_type::KvTypes) -> &'static str {
    if types.is_f16() {
        "a quantized cache (--cache-type-k q8_0 --cache-type-v q8_0, about half the bytes per \
         token); "
    } else {
        ""
    }
}

/// Device memory the automatic KV pool leaves free on a device of `kind`, with steps of up to
/// `max_batch_tokens`, and on a GPU sharing the host's memory `uma_free` for the OS
/// (`--uma-free-gib`, [`UMA_HEADROOM`] by default). Without the unified-memory reserve the
/// pool took 32.7 GiB on the GB10 (Qwen3.8-Flash-Next, 262K context, 16 sequences) and the
/// box hung (#301).
pub(crate) fn kv_headroom(
    kind: crate::ggml::DeviceKind,
    max_batch_tokens: usize,
    uma_free: usize,
) -> usize {
    match kind {
        crate::ggml::DeviceKind::IntegratedGpu => {
            COMPUTE_HEADROOM
                + uma_free
                + UMA_STEP_FIXED
                + max_batch_tokens
                    .div_ceil(1024)
                    .saturating_mul(UMA_STEP_BYTES_PER_1K)
        }
        _ => COMPUTE_HEADROOM,
    }
}

/// With the expert cache, the default KV pool: at least `--ctx-size`, at most this many tokens
/// (Strata's `--kv-resident` default). Sequences wait for KV blocks when it is full; the VRAM
/// goes to expert slots, which every step reads. `--kv-cache-tokens` overrides it.
const CACHE_KV_TOKENS: usize = 32 * 1024;

/// Where the model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    Cpu,
    /// The first GPU (CUDA0): every weight, the KV cache and the recurrent state live there.
    Cuda,
}

#[derive(Debug, Clone)]
pub struct LmConfig {
    pub model_path: std::path::PathBuf,
    /// `--mmproj`: a vision tower for the model (Qwen3.5 / Qwen3.6, #146); requests may then
    /// carry images.
    pub mmproj: Option<std::path::PathBuf>,
    /// Model id reported by /v1/models; `None` takes `general.name` or the file name.
    pub model_name: Option<String>,
    pub device: DeviceChoice,
    /// Max tokens per sequence (prompt + output); 0 for the model's native context
    /// (`context_length` in the GGUF), which is then an error if the cache cannot hold it.
    pub ctx_size: usize,
    /// CPU threads for the CPU backend.
    pub threads: usize,
    /// Sequences generated at once.
    pub max_seqs: usize,
    /// Tokens per forward pass: one per decoding sequence plus prefill chunks.
    pub max_batch_tokens: usize,
    /// `--prefill-slots`: prompts prefilling at once, each in a recurrent-state slot above the
    /// decode slots ([`PREFILL_SLOTS`] by default).
    pub prefill_slots: usize,
    /// KV cache capacity in tokens over all sequences and the prefix cache (`None`:
    /// `max_seqs × ctx_size`, less if device memory is short).
    pub kv_cache_tokens: Option<usize>,
    /// On a GPU sharing the host's memory: bytes the automatic KV pool leaves free for the OS
    /// beside the step buffers' share (`--uma-free-gib`, [`kv_headroom`]).
    pub uma_free: usize,
    /// Where the KV cache lives ([`KvMemory`]).
    pub kv_memory: KvMemory,
    /// KV cache tokens past what fits in device memory, in a host tier (#352): pinned memory on
    /// each GPU's NUMA node, read over the link (0: none). A layer split or one GPU. `None`:
    /// automatic, a tier only when a context does not fit in a discrete GPU's memory, sized to
    /// the rest of the pool and the nodes' free memory (see `host_tier`).
    pub kv_host_tokens: Option<usize>,
    /// With a host tier: tokens of VRAM staging slots per GPU that prefill copies host blocks
    /// into, so each step reads a host block over the link once instead of once per query
    /// tile; taken from the device blocks. Steps reaching more host tokens run in pieces.
    pub kv_stage_tokens: usize,
    /// Under `--tp`: whether each sequence gets its own slot of the KV cache, which attention
    /// reads in place (no gathered copy), the slot's first part in VRAM and the rest in the
    /// host tier ([`KvSlots`]).
    pub kv_slots: KvSlots,
    /// With sequence slots that reach the host tier: prompt steps of at least this many query
    /// rows per sequence copy a sequence's keys and values into the compute buffers before
    /// attention, so the host part crosses the link once per layer rather than once per tile
    /// of query rows (#1153; the same bits). 0: attention reads the slot in place.
    pub kv_slot_stage_rows: usize,
    /// Saved recurrent states for prefix caching; 0 turns prefix caching off.
    pub prefix_cache_states: usize,
    /// With prefix caching on a model with recurrent state: also save the state every this
    /// many prompt tokens (`--prefix-cache-interval`, [`super::saved_states::save_points`]);
    /// 0: only at branch points and ends.
    pub prefix_cache_interval: usize,
    /// With prefix caching on a model with sliding-window rings (Gemma 4): copies of rings
    /// kept in pinned host memory for the warm sequences ([`RingCheckpoints`]); `None`:
    /// [`ring_checkpoint_slots`]'s default.
    pub ring_checkpoints: Option<usize>,
    /// A ring checkpoint every this many prompt tokens (at the step that ends past it) over the
    /// last [`Self::ring_checkpoint_span`] tokens of a prompt; 0: at prompt ends only.
    pub ring_checkpoint_every: usize,
    pub ring_checkpoint_span: usize,
    /// Log why a prompt resumed nothing from the warm sequences.
    pub ring_checkpoint_log: bool,
    /// `--deterministic`'s admission in waves ([`waves`]): an idle engine gathers requests
    /// until none arrived for this long, admits them in a fixed order, and holds later ones
    /// until it is idle again. `None`: admit requests as they arrive.
    pub deterministic_gather: Option<std::time::Duration>,
    /// `--admit-gather-ms`: an idle engine that gets a request first takes what else arrives
    /// until none has for this long ([`waves::gather`]), then admits them all in arrival
    /// order, so a burst of requests starts its prompts in one step. Nothing is held while
    /// requests run. `None`: admit requests as they arrive.
    pub admit_gather: Option<std::time::Duration>,
    /// `--prefix-capture-cut` (debug): spaced save points end their prompt chunk even where
    /// the step could save the state inside it ([`Executor::can_capture_states`]), as before
    /// in-step saves; for exactness comparisons.
    pub prefix_capture_cut: bool,
    /// `--prefix-state-digests` (debug): log a digest of every saved recurrent state as it is
    /// saved and resumed from ([`crate::model::core::StatePool::slot_digests`]).
    pub prefix_state_digests: bool,
    /// Requests allowed to wait for admission before `generate` fails with `Overloaded`.
    pub max_waiting: usize,
    /// The order waiting requests are admitted in (`--scheduling-policy`, [`super::queue`]).
    pub scheduling: SchedulingPolicy,
    /// Sequences running at once from which new sequences' text goes to a thread of its own
    /// (`--stream-thread-seqs`, [`stream`]); 0: always on the engine thread.
    pub stream_thread_seqs: usize,
    /// How admission sizes a request's KV cache (`--kv-admission`, [`KvAdmission`]).
    pub kv_admission: KvAdmission,
    /// `--itl-target` / `--ttft-target`: each step's prompt budget from a step-time model
    /// ([`slo`]); off by default.
    pub slo: SloTargets,
    /// Speculative decoding (off by default).
    pub spec: SpecConfig,
    /// `--spec mtp`: the head drafts over the first this many tokens only (0: all).
    pub draft_vocab: usize,
    /// `--spec mtp`: the head attends to the last this many positions only (0: all; `None`: the
    /// default for the model and device, [`default_mtp_window`]).
    pub mtp_window: Option<usize>,
    /// `--spec mtp`: which weights the head drafts with (`--spec-draft-head-type`).
    pub draft_head_type: crate::model::core::DraftHeadType,
    /// The CPU backend's reference kernels: slower, but every op gives the same bits whatever
    /// the batch (for tests that compare batched or speculative runs with plain ones).
    pub use_ref: bool,
    /// On a GPU: keep the routed experts in host memory, mapped from the file, and let the
    /// GPU read them in place (unified-memory GPUs only).
    pub host_experts: bool,
    /// How a safetensors checkpoint's FP8 weights load.
    pub hf_fp8: crate::hf::Fp8Mode,
    /// Activation precision of NVFP4 matmuls.
    pub nvfp4_act: crate::model::core::Nvfp4Act,
    /// Activations of FP8 matmuls.
    pub fp8_act: crate::model::core::Fp8Act,
    /// Which kernels run NVFP4 matmuls on CUDA.
    pub nvfp4_mm: crate::model::core::Nvfp4Mm,
    /// Weights converted to F16 at load (`--weights-fp16`).
    pub fp16: crate::weights::Fp16Load,
    /// Which NVFP4 matmuls take 4-bit activations (W4A4 kernels).
    pub nvfp4_a4: crate::model::core::Nvfp4A4,
    /// Where the routed experts live and the VRAM expert cache (see [`crate::residency`]).
    pub residency: crate::residency::ResidencyConfig,
    /// GPUs to split the layers over (CUDA0, CUDA1, ...; one pipeline stage each).
    pub gpus: usize,
    /// Tensor parallelism: run every layer split over this many GPUs (CUDA0, CUDA1, ...) as
    /// one meta device ([`crate::ggml::meta`]); 1 for none. Excludes a layer split.
    pub tp: usize,
    /// Under `tp > 1`: the first of the `tp` GPUs (CUDA<first_gpu>, ...), so one server can run a
    /// copy of the model on each group of GPUs (`--tp-replicas`, [`super::replicas`]); 0 otherwise.
    pub first_gpu: usize,
    /// Prompt tokens per prefill chunk (default [`PREFILL_CHUNK`]). Bigger chunks read each
    /// host-resident expert once for more tokens; attention's scratch grows with them.
    pub prefill_chunk: usize,
    /// A prompt that needs several steps takes them in equal parts (multiples of
    /// [`KV_BLOCK`](crate::model::core::KV_BLOCK)) instead of full steps and a short last one
    /// (`--prefill-even`, [`even`]). Off by default.
    pub prefill_even: bool,
    /// With sliding-window rings: the most prompt tokens a sequence takes per step (its rings
    /// hold the window plus one step). `None`: `max_batch_tokens` on a GPU sharing memory with
    /// the host (GB10: room for the bigger rings, and whole steps prefill faster), else
    /// `prefill_chunk` (small rings for small VRAM).
    pub ring_step: Option<usize>,
    /// With several stages: a prefill chunk longer than this runs as micro-chunks of this
    /// many tokens, pipelined over the stages (0: never). Pair it with a `prefill_chunk` a few
    /// times as long.
    pub pipeline_micro: usize,
    /// With several stages: a prompt prefilling alone (nothing decoding, no other prompt)
    /// takes up to this many tokens per step, run as `pipeline_micro` micro-chunks, so the
    /// stages work at once on consecutive micro-chunks and the pipeline fills and drains once
    /// per step rather than once per chunk (0: steps of a chunk). Each micro-chunk's graph is
    /// at most `pipeline_micro` tokens (with rings at most their step), whatever this is.
    pub pipeline_prompt: usize,
    /// The first layer of every stage after the first (empty: an even split by bytes over
    /// `gpus` stages). On the CPU it makes stages of their own backends, for tests.
    pub layer_split: Vec<usize>,
    /// How attention reads the KV cache (see [`crate::model::core::Attention`]).
    pub attention: crate::model::core::Attention,
    /// How routed experts run over a batch (see [`crate::model::core::MoeKernel`]).
    pub moe_kernel: crate::model::core::MoeKernel,
    /// How a per-layer n-gram table is read on a GPU (see [`crate::residency::lookup_table`]).
    pub ngram: crate::residency::lookup_table::TableOptions,
    /// With several stages, split decode and verify batches into one micro-batch per stage
    /// (see [`crate::model::core::Executor::set_split_batches`]).
    pub split_batches: bool,
    /// With several stages, the decoding sequences as this many groups whose steps are in
    /// flight over the stages at once, a stage running one group's step while the next stage
    /// runs another's ([`crate::model::core::Executor::pipe_round`]; #415). 0 or 1: off (whole
    /// steps). Prompts being read run as whole steps: the groups drain first.
    pub pipeline_groups: usize,
    /// Fewest decoding sequences per group in flight ([`Self::pipeline_groups`]); fewer
    /// sequences make fewer groups. Default [`DEFAULT_PIPELINE_MIN_GROUP`]; 1 lets four
    /// sequences run as four groups over four stages (#752).
    pub pipeline_min_group: usize,
    /// `--pipeline-mtp-split` (#765): with [`Self::pipeline_groups`] and the MTP head, measure
    /// the rounds' stage times ([`LmEngine::mtp_split_hint`], [`pipe_split`]) for a split that
    /// gives the last stage fewer layers. Off: no measuring.
    pub pipeline_mtp_split: bool,
    /// `--split-last-extra` (#814): without a `layer_split`, the automatic split counts the
    /// last stage as also doing this many layers' worth of other work per step (output head,
    /// MTP drafts) and gives it fewer layers ([`pipe_split::mtp_split`]). 0: the split by
    /// bytes.
    pub split_last_extra: f64,
    /// A model split over two machines (#562, [`crate::node`]): this machine's rank, its
    /// peer and the split; `None`: the whole model here.
    pub node: Option<crate::node::NodeConfig>,
    /// Tensor parallelism over machines (#619, [`crate::tp_net`]): this machine's rank and
    /// every rank's address; `None`: the whole model here.
    pub tp_net: Option<crate::tp_net::TpNetConfig>,
    /// With `tp` 2: run the two GPUs as two ranks of this process instead of one meta device
    /// (`--tp-per-gpu`, [`tp_pair`]): each GPU its own executor, graphs and launch thread over
    /// its share of the model, the layers' partial sums added over NVLink.
    pub tp_per_gpu: bool,
    /// Set by `--tp-per-gpu` for each of its ranks' loads (not a setting).
    pub tp_pair: Option<tp_pair::TpPairRank>,
    /// GLM-5.3-Flash over several GPUs: every GPU works on every token (attention by heads,
    /// dense and shared FFNs by rows, routed experts expert-parallel, the LM head by rows; one
    /// all-reduce per sublayer; [`crate::model::glm5next::par`]) instead of one pipeline stage
    /// at a time, prompts and decode alike.
    pub glm_parallel_decode: bool,
    /// qwen4exp over an even number of GPUs: while a stage runs a decode or verify step, the
    /// other GPU of its NVLink pair computes half of each of its MoE layers' routed experts
    /// ([`crate::model::core::pair`]).
    pub pair_tp: bool,
    /// With `glm_parallel_decode`: which GPUs write their exchange payloads straight into
    /// another GPU's VRAM ([`crate::model::glm5next::par::exchange::PeerWrites`]).
    pub glm_exchange_peer: crate::model::glm5next::par::exchange::PeerWrites,
    /// With `glm_parallel_decode`: the tokens of a prompt's step while it borrows the expert
    /// cache (`--glm-prompt-step`, [`crate::model::glm5next::par::step_tokens`]); at most
    /// [`crate::model::glm5next::par::MAX_TOKENS`] keeps every step at that size.
    pub glm_prompt_step: usize,
    /// With `glm_parallel_decode`: which routed experts each GPU owns (`--glm-expert-split`,
    /// [`crate::model::glm5next::par::assign`]).
    pub glm_expert_split: crate::model::glm5next::par::assign::ExpertSplit,
    /// With `glm_parallel_decode`: GPU `d` owns `glm_expert_shares[d]` parts of the routed
    /// experts (`--glm-expert-shares`, [`crate::model::glm5next::par::expert_bounds_shared`]);
    /// empty: an even split.
    pub glm_expert_shares: Vec<u32>,
    /// With `glm_parallel_decode` and the MTP head drafting: its passes on every GPU
    /// (`--glm-parallel-mtp`, [`crate::model::glm5next::par::mtp`]).
    pub glm_parallel_mtp: bool,
    /// With `glm_parallel_mtp`: after steps of at most this many sequences, the commits queued
    /// after the pass and the target step's cache work beside its launches (0: never;
    /// `--glm-mtp-overlap-seqs`, [`crate::model::core::Executor::set_glm_mtp_overlap`]).
    pub glm_mtp_overlap_seqs: usize,
    /// With `glm_parallel_decode`: a decode step's routed experts balanced within each GPU pair
    /// (`--glm-pair-balance`, [`crate::model::glm5next::par::balance`]).
    pub glm_pair_balance: bool,
    /// With `glm_parallel_decode`: the next layer's router guess logged beside the routing and
    /// the prefetch predictors' report (`--glm-prefetch-stats`, #745,
    /// [`crate::residency::ep_prefetch`]).
    pub glm_prefetch_stats: bool,
    /// With `glm_parallel_decode`: the balance over every GPU, uncached experts also moving
    /// across the pairs (`--glm-cross-balance`, #731); implies `glm_pair_balance`.
    pub glm_cross_balance: bool,
    /// With `glm_parallel_decode`: the next layer's guessed uncached experts copied into VRAM
    /// ahead, at most this many per GPU and layer, 0 off (`--glm-prefetch`, #747,
    /// [`crate::model::glm5next::par::prefetch`]).
    pub glm_prefetch: usize,
    /// With `glm_parallel_decode`: a decode step's uncached routed experts split over both
    /// GPUs of their pair by rows (`--glm-pair-split`, [`crate::model::glm5next::par::split`]).
    pub glm_pair_split: bool,
    /// Legion's decode expert matvec on CUDA (default), or ggml's mul_mat_id everywhere.
    pub expert_mv: bool,
    /// Paged prefill attention (and speculative verify steps) on sm_70 GPUs through the m8n8k4
    /// one-pass kernel instead of the WMMA one (other GPUs keep their default). Greedy output
    /// differs from the default by rounding order (P rounded to F16 against the running max,
    /// one pass, 64-wide MMA chains against WMMA's 16-wide).
    pub paged_prefill_m884: bool,
    /// qwen4exp prompt steps: ggml's indexer chain (pool mask, lightning indexer, bias add)
    /// instead of Legion's one-op scores (the same bits), for comparisons.
    pub qsa_scores_ggml: bool,
    /// qwen4exp prompt steps: ggml's `top_k` for the indexer's pools instead of Legion's
    /// one-launch top-k (the same pools but for exact ties at the threshold).
    pub qsa_topk_ggml: bool,
    /// qwen4exp: the generic hc_post_norm kernel instead of the register one on sm_80+ and sm_70
    /// (the same bits), for comparisons.
    pub hc_post_norm_generic: bool,
    /// Integrated GPUs: keep the model files' pages in the page cache through the load
    /// (`--keep-load-cache`; else [`release_load_cache`] drops them before the weights and
    /// before the caches are allocated).
    pub keep_load_cache: bool,
    /// GLM-5.3-Flash: the cached experts' selection bonus in decode and verify steps
    /// ([`crate::model::glm5next::set_route_bias`]; 0: off).
    pub glm_route_bias: f32,
    /// GLM-5.3-Flash: soft pruning, the `(layer, expert)` pairs and their selection penalty
    /// ([`crate::model::glm5next::soft_prune`]).
    pub glm_prune: Option<crate::model::glm5next::SoftPrune>,
    /// How long prompts borrow the expert cache's VRAM (with staging, see [`LongPrompts`]).
    pub long_prompts: LongPrompts,
    /// Keep every position of the sliding-window layers too, instead of a ring of their
    /// window per sequence ([`crate::model::core::KvRing`]).
    pub swa_full: bool,
    /// Rings for the sliding-window layers of a model with a per-sequence state too
    /// (DeepSeek-V4.1), which then has no prefix cache; without it such a model keeps every
    /// position unless prefix caching is off.
    pub swa_ring: bool,
    /// Gemma 4 only: compact caches for the global layers without a value projection
    /// (`--gemma4-compact-global-kv`, [`Model::set_gemma4_compact_global_kv`]).
    pub gemma4_compact_global_kv: bool,
    /// The K and V caches' element types (`--cache-type-k` / `--cache-type-v`).
    pub cache_types: crate::model::core::KvTypes,
    /// Rings with gather attention too (by default only paged attention gets them): for
    /// tensor parallelism, whose attention gathers (ring layers gather through the rings).
    pub ring_gather: bool,
    /// Gather attention by cells, as tensor parallelism does (always under `tp > 1`): for
    /// tests of that path on one device.
    pub cell_gather: bool,
    /// Kernel and load switches ([`crate::engine::switches`]); default: none set.
    pub switches: crate::engine::switches::Switches,
    /// Threads that find grammar-constrained sequences' next token masks while the forward
    /// pass runs (`--grammar-threads`; 0: at sampling, on the engine thread). Started the
    /// first time a sequence is constrained.
    pub grammar_threads: usize,
}

/// Long prompts with an expert cache that stages (`--prefill-staging`): while a prompt with at
/// least `min_tokens` tokens left is being read, the cache lends its VRAM
/// ([`Executor::lend_experts`]): its slots become staging buffers (each layer's experts copied
/// into VRAM before its matmuls) and room for big steps, of up to `batch_tokens` tokens, in
/// pipeline rounds of up to `micro` tokens. Decode keeps the full cache the rest of the time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LongPrompts {
    /// Prompt tokens left from which a prompt borrows the cache (0: never).
    pub min_tokens: usize,
    /// Prompt tokens left from which a shorter prompt borrows the cache for staging only: its
    /// steps keep the chunks of an unlent cache, so the same outputs (0: never). `None` until
    /// the engine loads: then the model's default ([`short_prompt_default`]).
    pub short_min_tokens: Option<usize>,
    pub batch_tokens: usize,
    /// Tokens per step when no sequence is decoding: a prompt alone is read in as few steps as
    /// possible, as the pipeline fills and drains once per step (a 123K prompt in two 64K steps
    /// costs 37 rounds instead of 34 on 4 GPUs: +9% prefill). With decoding sequences the step
    /// stays at `batch_tokens`, so they wait no longer than before.
    pub alone_tokens: usize,
    /// The pipeline micro-chunk, at most (see [`Self::micro_for`]).
    pub micro: usize,
    /// Micro-chunks per pipeline stage a step aims for (0: always [`Self::micro`]).
    pub micro_per_stage: f32,
    /// Cut each step into micro-chunks of about equal size ([`Self::even_micro`]) rather than
    /// full ones and a short last one (`--long-micro-even`; off by default, as other chunk
    /// boundaries change the float summation order).
    pub even: bool,
    /// Ramp the last micro-chunks of a step down to this many tokens (`--long-micro-taper`,
    /// see [`Executor::set_micro_taper`]; 0: off, as other chunk boundaries change the float
    /// summation order).
    pub taper: usize,
    /// Run a step's micro-chunks without rounds (`--long-dataflow`, see
    /// [`Executor::set_dataflow`]).
    pub dataflow: bool,
    /// Peer access between neighbouring stages' GPUs for the dataflow hand-off
    /// (`--long-dataflow-peer`, see [`Executor::set_dataflow_peer`]).
    pub dataflow_peer: bool,
    /// The dataflow hand-off without host waits, and launch-ahead (`--long-dataflow-async`,
    /// see [`Executor::set_dataflow_async`]).
    pub dataflow_async: bool,
}

impl Default for LongPrompts {
    fn default() -> Self {
        Self {
            min_tokens: 2048,
            short_min_tokens: None,
            batch_tokens: 65536,
            alone_tokens: 262144,
            micro: 4096,
            micro_per_stage: 2.0,
            even: false,
            taper: 0,
            dataflow: false,
            dataflow_peer: false,
            dataflow_async: false,
        }
    }
}

/// The smallest adaptive micro-chunk ([`LongPrompts::micro_for`]).
const MIN_MICRO: usize = 2048;

/// The smallest micro-chunk a lend caps long prompts to (#1216, [`Worker::long_micro_cap`]):
/// below it, staging every expert per micro-chunk costs more than a spilled step buffer.
const MIN_LENT_MICRO: usize = 512;

/// VRAM a lend's micro-chunk cap leaves free beside the long step buffer and
/// [`crate::model::core::LENT_SCRATCH`] (#1216).
const LENT_MICRO_MARGIN: usize = 64 << 20;

/// #1216: a stage's long prompt step compute buffer at [`LongPrompts::micro`] tokens, as
/// measured at load after no past and after the longest ([`Worker::long_past`]).
#[derive(Debug, Clone, Copy)]
struct LongStep {
    device: Option<crate::ggml::Device>,
    first: usize,
    last: usize,
}

impl LongStep {
    /// The buffer for a micro-chunk ending at most `end` tokens in: linear in the past
    /// between the two measurements (an indexer's scores grow with it), at most `last`.
    fn at(&self, end: usize, long_past: usize) -> usize {
        let grow = self.last.saturating_sub(self.first) as u128;
        let past = end.min(long_past) as u128;
        self.first + (grow * past / long_past.max(1) as u128) as usize
    }
}

/// [`LongPrompts::even_micro`]'s micro-chunks are whole KV blocks.
const EVEN_STEP: usize = KV_BLOCK;

/// GLM-5.3-Flash over several GPUs with an expert cache: a prompt borrows the cache for staging
/// from this many tokens left ([`LongPrompts::short_min_tokens`]), the fewest a pass stages its
/// experts for ([`crate::residency::STAGE_MIN_TOKENS`]). In place, a short prompt's passes read
/// every uncached expert over the link at the in-place rate; staged, a layer's experts arrive by
/// DMA before its matmuls (#579, Maya-S on 4 V100s). Its steps keep their chunks, and staging
/// only moves where an expert is read from: the same outputs.
pub const GLM_SHORT_PROMPT: usize = crate::residency::STAGE_MIN_TOKENS;

/// [`LongPrompts::short_min_tokens`]' default: [`GLM_SHORT_PROMPT`] for GLM-5.3-Flash over
/// several stages, else 0 (Qwen and Gemma unchanged).
pub fn short_prompt_default(glm: bool, stages: usize) -> usize {
    if glm && stages > 1 {
        GLM_SHORT_PROMPT
    } else {
        0
    }
}

impl LongPrompts {
    /// [`Self::short_min_tokens`] when it is set and below [`Self::min_tokens`].
    pub fn short_from(&self) -> Option<usize> {
        self.short_min_tokens
            .filter(|&n| n > 0 && n < self.min_tokens)
    }

    /// The pipeline micro-chunk for a step of `tokens` prompt tokens over `stages` stages:
    /// about [`Self::micro_per_stage`] micro-chunks per stage, in 256-token steps, at least
    /// 2048 tokens and at most [`Self::micro`]. The pipeline fills and drains for about
    /// `stages - 1` micro-chunk times a step, so a mid-size prompt wants smaller ones and a
    /// long one the full size. Adapted from Strata-AC922's STRATA_PREFILL_ADAPT
    /// (eelgaev/Strata-AC922, 90cdb8b, MIT).
    pub fn micro_for(&self, tokens: usize, stages: usize) -> usize {
        if self.micro_per_stage <= 0.0 || stages < 2 {
            return self.micro;
        }
        let want = (tokens as f64 / (self.micro_per_stage as f64 * stages as f64)).ceil() as usize;
        want.next_multiple_of(256)
            .clamp(MIN_MICRO.min(self.micro), self.micro)
    }

    /// The micro-chunk for a step of `tokens` prompt tokens with [`Self::even`]: as many
    /// micro-chunks as [`Self::micro_for`]'s size needs, of about equal size (rounded up to
    /// whole KV blocks, at most that size). The executor cuts a step into full micro-chunks
    /// and the rest, and a short last micro-chunk still stages about every routed expert of
    /// its layers (a 123K prompt over 2 GPUs: 30 x 4096 tokens, then 312 tokens for ~0.6 s;
    /// equal ones read the same prompt in 31 rounds of ~4000 tokens). Builds on Strata-AC922's
    /// per-prompt chunk (STRATA_PREFILL_ADAPT, eelgaev/Strata-AC922 90cdb8b, MIT), whose last
    /// chunk is the rest as here by default.
    pub fn even_micro(&self, tokens: usize, stages: usize) -> usize {
        let cap = self.micro_for(tokens, stages);
        if cap == 0 || tokens <= cap {
            return cap;
        }
        let n = tokens.div_ceil(cap);
        tokens.div_ceil(n).next_multiple_of(EVEN_STEP).min(cap)
    }
}

impl LmConfig {
    /// Defaults for everything but the model.
    pub fn new(model_path: impl Into<std::path::PathBuf>, device: DeviceChoice) -> Self {
        Self {
            model_path: model_path.into(),
            mmproj: None,
            model_name: None,
            device,
            ctx_size: 8192,
            threads: 8,
            max_seqs: 16,
            max_batch_tokens: MAX_BATCH_TOKENS,
            prefill_slots: PREFILL_SLOTS,
            kv_cache_tokens: None,
            uma_free: UMA_HEADROOM,
            kv_memory: KvMemory::Device,
            kv_host_tokens: None,
            kv_stage_tokens: 16384,
            kv_slots: KvSlots::Auto,
            kv_slot_stage_rows: KV_SLOT_STAGE_ROWS,
            prefix_cache_states: 8,
            prefix_cache_interval: 0,
            ring_checkpoints: None,
            ring_checkpoint_every: RING_CHECKPOINT_EVERY,
            ring_checkpoint_span: RING_CHECKPOINT_SPAN,
            ring_checkpoint_log: false,
            deterministic_gather: None,
            admit_gather: None,
            prefix_capture_cut: false,
            prefix_state_digests: false,
            max_waiting: 1024,
            scheduling: SchedulingPolicy::Fcfs,
            stream_thread_seqs: STREAM_THREAD_SEQS,
            kv_admission: KvAdmission::Reserve,
            slo: SloTargets::default(),
            spec: SpecConfig::default(),
            draft_vocab: 0,
            mtp_window: None,
            draft_head_type: Default::default(),
            use_ref: false,
            host_experts: false,
            hf_fp8: crate::hf::Fp8Mode::Raw,
            nvfp4_act: crate::model::core::Nvfp4Act::A8,
            fp8_act: crate::model::core::Fp8Act::A8,
            nvfp4_mm: crate::model::core::Nvfp4Mm::Auto,
            fp16: Default::default(),
            nvfp4_a4: crate::model::core::Nvfp4A4::Off,
            residency: Default::default(),
            gpus: 1,
            tp: 1,
            first_gpu: 0,
            prefill_chunk: PREFILL_CHUNK,
            prefill_even: false,
            ring_step: None,
            pipeline_micro: PREFILL_CHUNK,
            pipeline_prompt: PIPELINE_PROMPT,
            layer_split: Vec::new(),
            attention: Default::default(),
            moe_kernel: Default::default(),
            ngram: Default::default(),
            split_batches: false,
            pipeline_groups: 0,
            pipeline_min_group: DEFAULT_PIPELINE_MIN_GROUP,
            pipeline_mtp_split: false,
            split_last_extra: 0.0,
            node: None,
            tp_net: None,
            tp_per_gpu: false,
            tp_pair: None,
            glm_parallel_decode: false,
            pair_tp: false,
            glm_exchange_peer: Default::default(),
            glm_prompt_step: crate::model::glm5next::par::PROMPT_TOKENS,
            glm_expert_split: Default::default(),
            glm_expert_shares: Vec::new(),
            glm_parallel_mtp: false,
            glm_mtp_overlap_seqs: 1,
            glm_pair_balance: false,
            glm_prefetch_stats: false,
            glm_cross_balance: false,
            glm_prefetch: 0,
            glm_pair_split: false,
            expert_mv: true,
            qsa_scores_ggml: false,
            qsa_topk_ggml: false,
            hc_post_norm_generic: false,
            keep_load_cache: false,
            glm_route_bias: 0.0,
            glm_prune: None,
            paged_prefill_m884: false,
            long_prompts: LongPrompts::default(),
            swa_full: false,
            swa_ring: false,
            gemma4_compact_global_kv: false,
            ring_gather: false,
            cell_gather: false,
            cache_types: Default::default(),
            switches: Default::default(),
            grammar_threads: GRAMMAR_THREADS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
    #[error(transparent)]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
    #[error(transparent)]
    Template(#[from] crate::chat_template::ChatTemplateError),
    #[error(transparent)]
    Model(#[from] crate::model::core::ModelError),
    #[error(transparent)]
    Ggml(#[from] crate::ggml::GgmlError),
    #[error("{0}")]
    Device(String),
}

pub struct LmEngine {
    info: ModelInfo,
    tokenizer: Arc<Tokenizer>,
    /// `--mmproj`: what turns a request's images into tokens and rows.
    vision: Option<Arc<vision::Vision>>,
    template: Option<ChatTemplate>,
    chat_format: ChatFormat,
    grammars: Mutex<GrammarCache>,
    ctx_size: usize,
    max_waiting: usize,
    /// Requests submitted and not yet admitted.
    queued: Arc<AtomicUsize>,
    stats: Arc<Mutex<EngineStats>>,
    /// `None` only while dropping.
    submit: Option<std::sync::mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
    load_report: report::LoadReport,
    /// `--pipeline-mtp-split`'s measurements, shared with the worker (#765).
    split_probe: Option<Arc<Mutex<pipe_split::Probe>>>,
}

impl Drop for LmEngine {
    /// Stops the worker and waits for it, so device memory is freed before the process exits
    /// (the CUDA runtime aborts if a backend is freed after it shut down).
    fn drop(&mut self) {
        self.submit.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// A tokenized request waiting for the worker.
struct Job {
    grammar: Option<crate::grammar::Grammar>,
    /// The call grammar of strict tools ([`LmEngine::compile_strict_tools`]).
    tool_grammar: Option<crate::grammar::Grammar>,
    prompt: Vec<u32>,
    /// The prompt's images ([`vision`]), and its prefix-cache key with them.
    media: Option<(Arc<Media>, Vec<u32>)>,
    sampling: SamplingParams,
    max_tokens: u32,
    stop: Vec<String>,
    ignore_eos: bool,
    min_tokens: u32,
    /// Split reasoning, and whether the prompt already opened it.
    reasoning: Option<bool>,
    /// The request's tools, for the tool-call parser (none: no parsing).
    tools: Vec<super::Tool>,
    out: mpsc::UnboundedSender<TokenEvent>,
    cancel: CancellationToken,
    /// `GenerateRequest::scheduling`.
    sched: Scheduling,
    /// When `generate` queued it (its queue wait ends at admission).
    queued_at: Instant,
    /// Held while the job waits; dropped when it is admitted or discarded.
    _queued: QueueSlot,
}

/// One place in the waiting queue; frees itself when dropped.
struct QueueSlot(Arc<AtomicUsize>);

impl Drop for QueueSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// VRAM the expert cache leaves beside the measured step compute buffers, for the steps that
/// need a little more than measured.
const COMPUTE_MARGIN: usize = 256 << 20;

/// Tokens of each GPU's KV pool that keep VRAM by default where `--kv-vmm-keep` is on by
/// default (a V100 layer split of 2 GPUs with the expert cache, #814).
const KV_VMM_KEEP_DEFAULT: usize = 32768;

/// How often the worker refreshes [`EngineStats`]' expert cache counters (the interactive
/// status line redraws every 350 ms).
const EXPERT_STATS_EVERY: std::time::Duration = std::time::Duration::from_millis(250);

/// VRAM per GPU the tensor-parallel cache fit keeps free beyond the measured compute buffers,
/// for what grows outside them (ggml's CUDA pools, NCCL's and the meta backend's staging):
/// 512 MiB, and 128 MiB more per 1024 tokens of `--max-batch-tokens` past the first 1024.
/// Measured on Gemma 4 31B over two V100s: at 1024 tokens the peak left ~550 MiB of the 512 MiB
/// margin free; at 4096 (four 8K prompts at once) ~350 MiB.
fn tp_margin(max_batch_tokens: usize) -> usize {
    (512 << 20) + (max_batch_tokens.saturating_sub(1024) >> 10) * (128 << 20)
}

/// Tensor parallelism's cache sizing, as llama.cpp's graph reserve: measure the compute
/// buffers of the steps the engine keeps graphs of (on a meta backend the graphs of a kind
/// share one buffer: the mixed steps, i.e. prompt chunks, verify batches or both batched
/// together; the decode batches; with `mtp`, the MTP head's passes), all on each GPU at once,
/// and give the KV cache what is left on the fullest GPU, minus a margin ([`tp_margin`]). No
/// step gathers more keys than one sequence at the full context (the executor's gather budget
/// splits the others), so the steps measured are the corners of that budget. Nothing then
/// allocates while serving except within those buffers. A context that does not fit is an
/// error naming what it needs and the largest one whose own buffers leave room for its cache
/// ([`fit::fit_context`]); `native`: the context is the model's own (no --ctx-size).
/// `head_pool`: ggml's pool copy of the mirrored LM head that steps of 64+ rows on a V100 take
/// (`vram::head_pool_bytes`), kept free with the margin. Returns the KV blocks.
#[allow(clippy::too_many_arguments)]
fn fit_tp_cache(
    exec: &mut Executor,
    layout: &layout::Layout,
    cfg: &mut LmConfig,
    sched: fit::Sched,
    mtp: bool,
    native: bool,
    tier: TpTier,
    head_pool: usize,
) -> Result<(usize, usize), LoadError> {
    if head_pool > 0 {
        tracing::info!(
            head_pool_mib = head_pool >> 20,
            "ggml's F16 copy of the LM head (cuBLAS at 64+ rows), kept free beside the compute \
             buffers on each GPU"
        );
    }
    let margin = tp_margin(sched.max_batch) + head_pool;
    let spec_k = sched.spec_k;
    let err = |what: String| {
        move |e: crate::model::core::ModelError| {
            LoadError::Device(format!("cannot build {what}: {e}"))
        }
    };
    // The rings (if any) are allocated already, beside the probe's one block.
    let per = exec.kv_block_bytes();
    let mut last = [0usize; 4];
    let asked = cfg.ctx_size;
    // The compute buffers measured at the asked context (for the error if it does not fit).
    let mut at_asked = 0;
    // Free bytes per GPU now, with the probe cache allocated.
    let free = layout.free_for_split(0, 0) / layout.tp_devices.len().max(1);
    let measure_raw = |exec: &mut Executor, ctx: usize| -> Result<[usize; 4], LoadError> {
        // A meta buffer keeps the per-device views that graphs make of its tensors (the
        // cache's) until a compute rotates them out; allocating graph after graph without
        // computing would fill its room for them (a ggml abort). Each measurement starts on
        // a fresh one-block probe cache. With sequence slots, every graph does, on one that
        // holds the span the slot views read (attention pads it to 256 keys): each graph
        // views the cache per sequence and layer.
        let slots = exec.kv_slot_blocks() > 0;
        // With slot prompt copies on (#1153, `fit_tp_slots`), only the buffers of the steps
        // that make them: when decode and verify steps and drafts have too few rows.
        let prompts_only = exec.kv_slot_stage().0 > spec_k + 1;
        let probe = if slots {
            (ctx + spec_k).next_multiple_of(256).div_ceil(KV_BLOCK) + 1
        } else {
            1
        };
        let fresh = |exec: &mut Executor| {
            exec.resize_kv(probe)
                .map_err(|e| LoadError::Device(format!("cannot allocate the probe KV cache: {e}")))
        };
        fresh(exec)?;
        // In slots nothing is gathered: decode steps and drafts with every sequence at the
        // full context.
        let c = if slots {
            fit::slot_corners(&sched, ctx)
        } else {
            fit::corners(&sched, ctx)
        };
        let mut most = |steps: &[Vec<crate::model::core::StepPart>]| -> Result<usize, LoadError> {
            let mut most = 0;
            for parts in steps {
                if slots {
                    fresh(exec)?;
                }
                most = most.max(
                    exec.measure_groups(parts)
                        .map_err(err(format!("a step of {parts:?} at context {ctx}")))?[0],
                );
            }
            Ok(most)
        };
        // A meta backend keeps the graphs of each kind in one shared buffer, grown to the
        // largest: the mixed ones in one, the decode ones in another.
        let mixed = most(&c.mixed)?;
        let decode = if prompts_only { 0 } else { most(&c.decode)? };
        // The MTP head's passes: drafts (one token per sequence); a head with its own cache
        // also takes in what the target ran (up to a piece of a prompt, or a verify's rows).
        let (mtp_decode, mtp_mixed) = if mtp && spec_k > 0 {
            let mut d = 0;
            for &(n, span) in c.drafts.iter().filter(|_| !prompts_only) {
                if slots {
                    fresh(exec)?;
                }
                d = d.max(
                    exec.measure_mtp(n, 1, span.saturating_sub(1))
                        .map_err(err(format!("a {n}-sequence MTP pass")))?,
                );
            }
            let m = if exec.mtp_shares_kv() {
                0
            } else {
                let chunk = sched.chunk.min(sched.max_batch).max(1);
                let piece = exec
                    .measure_mtp(1, spec::TAKE_IN_PIECE.min(chunk), ctx.saturating_sub(chunk))
                    .map_err(err("an MTP take-in piece".into()))?;
                let (n, span) = c.drafts[0];
                let n_tok = spec::TAKE_IN_TAIL + spec_k + 1;
                let rows = exec
                    .measure_mtp(n, n_tok, span.saturating_sub(n_tok))
                    .map_err(err("an MTP take-in batch".into()))?;
                piece.max(rows)
            };
            (d, m)
        } else {
            (0, 0)
        };
        tracing::debug!(
            ctx,
            mixed_mib = mixed >> 20,
            decode_mib = decode >> 20,
            mtp_decode_mib = mtp_decode >> 20,
            mtp_mixed_mib = mtp_mixed >> 20,
            "tensor-parallel compute buffers measured"
        );
        Ok([mixed, decode, mtp_decode, mtp_mixed])
    };
    // The buffers grow about linearly with the context: from two small measurements, a
    // context whose buffers alone would not fit is never measured (allocating them would abort
    // in ggml's meta backend).
    let (c0, c1) = (asked.min(2048), asked.min(4096));
    let line = |exec: &mut Executor, (c0, c1): (usize, usize)| -> Result<Line, LoadError> {
        let first = measure_raw(exec, c0)?;
        let small = first.iter().sum::<usize>();
        if c1 <= c0 {
            return Ok((small, 0.0, first, [0.0; 4]));
        }
        let second = measure_raw(exec, c1)?;
        let grow = |a: usize, b: usize| b.saturating_sub(a) as f64 / (c1 - c0) as f64;
        let slope = grow(small, second.iter().sum());
        let slopes = std::array::from_fn(|i| grow(first[i], second[i]));
        Ok((small, slope, second, slopes))
    };
    let (small, slope, ..) = line(exec, (c0, c1))?;
    let estimate = |ctx: usize| small + (slope * ctx.saturating_sub(c0) as f64) as usize;
    let ctx_blocks = (asked + spec_k).div_ceil(KV_BLOCK);
    let want = sched.max_seqs.max(1) * ctx_blocks;
    // Per-sequence slots: attention reads each sequence's slot in place, so the compute
    // buffers no longer hold a gathered copy of its context.
    let slots = match cfg.kv_slots {
        KvSlots::Off => false,
        _ if !exec.reads_kv_slots() => {
            if cfg.kv_slots == KvSlots::On {
                tracing::warn!("--kv-slots on: this model's attention does not read slots");
            }
            false
        }
        KvSlots::On => true,
        KvSlots::Auto => {
            !matches!(tier, TpTier::Off)
                && layout.free_for_split(0, estimate(asked) + margin) / per.max(1) < want
        }
    };
    if slots {
        // Measured past the sliding layers' windows: their rings' reads stop growing there,
        // and only the masks still grow with the context (decode steps and drafts with every
        // sequence at that span, `fit::slot_corners`).
        let at = (asked.min(16384), asked.min(32768));
        return fit_tp_slots(exec, layout, cfg, sched, tier, margin, at, |exec| {
            line(exec, at)
        });
    }
    let mut measure = |ctx: usize| -> Result<usize, LoadError> {
        let compute = if estimate(ctx) + margin > free {
            tracing::debug!(
                ctx,
                estimate_mib = estimate(ctx) >> 20,
                "tensor-parallel compute buffers estimated (too big to measure)"
            );
            estimate(ctx)
        } else {
            let m = measure_raw(exec, ctx)?;
            last = m;
            m.iter().sum()
        };
        if ctx == asked {
            at_asked = compute;
        }
        Ok(compute + margin)
    };
    // A host tier (#317): the asked context stays, its cache takes the room the compute
    // buffers leave on the GPUs and the tier the rest.
    let mut host = 0;
    let mut tiered = None;
    if !matches!(tier, TpTier::Off) {
        let compute = measure(asked)?;
        // `--kv-cache-tokens` with a tier: the blocks kept in the GPUs, as without --tp.
        let device = (layout.free_for_split(0, compute) / per.max(1))
            .min(want)
            .min(
                cfg.kv_cache_tokens
                    .map_or(usize::MAX, |t| t.div_ceil(KV_BLOCK)),
            );
        host = match &tier {
            TpTier::Blocks(h) => *h,
            TpTier::Auto(nodes) => {
                host_tier::auto_host_blocks(device, ctx_blocks, want, device, nodes)
                    .map_err(LoadError::Device)?
            }
            TpTier::Off => 0,
        };
        if host > 0 {
            if device == 0 || device + host < ctx_blocks {
                return Err(LoadError::Device(format!(
                    "--kv-host-tokens {}: with the {} tokens the GPUs hold next to {} MiB of \
                     compute buffers per GPU, that is short of the {asked}-token context",
                    host * KV_BLOCK,
                    device * KV_BLOCK,
                    compute >> 20
                )));
            }
            tiered = Some(fit::Fitted {
                ctx: asked,
                n_blocks: device,
                compute,
            });
        }
    }
    let fitted = match tiered {
        Some(f) => Ok(f),
        None => fit::fit_context(
            asked,
            spec_k,
            sched.max_seqs,
            per,
            |compute| layout.free_for_split(0, compute),
            measure,
        ),
    }
    .map_err(|e| match e {
        fit::FitError::Measure(e) => e,
        fit::FitError::NoRoom { ctx, compute } => LoadError::Device(format!(
            "no room for a KV cache of {ctx} tokens next to {} MiB of compute buffers per GPU",
            compute >> 20
        )),
    })?;
    if fitted.ctx < asked {
        // Never lowered silently: say what the asked context needs and what fits.
        let n_dev = layout.tp_devices.len().max(1);
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let kv = per * (asked + spec_k).div_ceil(KV_BLOCK) / n_dev;
        let what = if native {
            format!("the model's native context of {asked} tokens")
        } else {
            format!("a {asked}-token context")
        };
        return Err(LoadError::Device(format!(
            "{what} does not fit on {n_dev} GPUs: per GPU it needs {:.1} GiB of KV cache and \
             {:.1} GiB of compute buffers (and a {:.1} GiB margin), next to {:.1} GiB free. \
             The largest context that fits is {}. Options: --ctx-size {} (or less); {}more \
             GPUs (--tp); a host-memory KV tier (--kv-host-tokens)",
            gib(kv),
            gib(at_asked),
            gib(margin),
            gib(free),
            fitted.ctx,
            fitted.ctx,
            quantized_option(exec.kv_types()),
        )));
    }
    let n_blocks = fitted.n_blocks;
    cfg.ctx_size = fitted.ctx;
    exec.resize_kv(n_blocks)
        .map_err(|e| LoadError::Device(format!("cannot allocate {n_blocks} KV blocks: {e}")))?;
    exec.set_max_seq_tokens(cfg.ctx_size + spec_k);
    // What the buffers were measured for: no step gathers more than one full sequence.
    exec.set_gather_budget(fit::budget(cfg.ctx_size, spec_k));
    let least_free = layout
        .tp_devices
        .iter()
        .map(|d| d.memory().0)
        .min()
        .unwrap_or(0);
    let [mixed, decode, mtp_decode, mtp_mixed] = last;
    tracing::info!(
        ctx_size = cfg.ctx_size,
        kv_cache_tokens = n_blocks * KV_BLOCK,
        sequences_at_full_context = n_blocks / (cfg.ctx_size + spec_k).div_ceil(KV_BLOCK).max(1),
        mixed_mib = mixed >> 20,
        decode_mib = decode >> 20,
        mtp_decode_mib = mtp_decode >> 20,
        mtp_mixed_mib = mtp_mixed >> 20,
        headroom_mib = least_free.saturating_sub(fitted.compute - margin) >> 20,
        gpus = layout.tp_devices.len().max(1),
        host_tier_tokens = host * KV_BLOCK,
        "tensor-parallel cache fitted"
    );
    Ok((n_blocks, host))
}

/// Tensor-parallel compute buffers measured at two contexts `c0 < c1`: their sum at `c0`, its
/// growth per position, the buffers (mixed, decode, MTP decode, MTP mixed) at `c1`, and each
/// one's growth per position.
type Line = (usize, f64, [usize; 4], [f64; 4]);

/// [`fit_tp_cache`] with per-sequence slots ([`KvSlots`]): each of `max_seqs` sequences gets a
/// slot of the asked context (rounded up to 64 blocks, a whole number of the tier's mapping
/// granules), its first blocks in VRAM, as many as the room the compute buffers leave allows
/// (`--kv-cache-tokens` caps it), and the rest in the host tier. The buffers are measured by
/// `line` at two lengths past the rings' saturation (a probe cache of the asked context could
/// not be allocated) and extended to the asked context; they hardly grow with it (the mask's).
#[allow(clippy::too_many_arguments)]
fn fit_tp_slots(
    exec: &mut Executor,
    layout: &layout::Layout,
    cfg: &mut LmConfig,
    sched: fit::Sched,
    tier: TpTier,
    margin: usize,
    (c0, c1): (usize, usize),
    mut line: impl FnMut(&mut Executor) -> Result<Line, LoadError>,
) -> Result<(usize, usize), LoadError> {
    const ROUND: usize = 64;
    let spec_k = sched.spec_k;
    let asked = cfg.ctx_size;
    let n_seqs = sched.max_seqs.max(1);
    let slot_blocks = (asked + spec_k).div_ceil(KV_BLOCK).next_multiple_of(ROUND);
    exec.set_kv_slot_blocks(slot_blocks);
    let (small, slope, last, slopes) = line(exec)?;
    let compute = small + (slope * asked.saturating_sub(c0) as f64) as usize;
    // Prompt steps' copies of their spans (#1153): measured again with them, they are made
    // only up to the span whose buffers with them take no more than these buffers sized for
    // the asked context, so they never take room from the cache or the rings. Switched on
    // with the slots' VRAM part (`LmEngine::load`).
    let mut upto = 0;
    if cfg.kv_slot_stage_rows > 0 && c1 > c0 {
        exec.set_kv_slot_stage(cfg.kv_slot_stage_rows, 0, usize::MAX);
        let (_, _, with, with_slopes) = line(exec)?;
        // Every buffer measured again: the mixed steps' and the MTP head's take-ins, and the
        // decode steps' and drafts' when they have as many rows.
        let c = (c0, c1);
        let span =
            |i: usize| slot_stage_span((last[i], slopes[i]), (with[i], with_slopes[i]), c, asked);
        upto = (0..4).filter(|&i| with[i] > 0).map(span).min().unwrap_or(0);
    }
    exec.set_kv_slot_stage(0, 0, upto);
    let per = exec.kv_block_bytes().max(1);
    let most = cfg
        .kv_cache_tokens
        .map_or(usize::MAX, |t| t.div_ceil(KV_BLOCK));
    let room = || (layout.free_for_split(0, compute + margin) / per).min(most);
    let mut device = room();
    // A spilled decode state (#1131) kept the load's reserve free, short of these buffers: its
    // rings take less VRAM, the buffers all they need; the slots then live in the host tier.
    let spilled = device / n_seqs < ROUND && exec.kv_spilled(0);
    if spilled {
        if respill_tp(exec, layout, compute + margin).map_err(LoadError::Device)? {
            exec.resize_kv(1).map_err(|e| {
                LoadError::Device(format!("cannot allocate the probe KV cache: {e}"))
            })?;
        }
        device = room();
    }
    let fixed = (device / n_seqs / ROUND * ROUND).min(slot_blocks);
    // #1159: where the slots do not all fit, the VRAM is not split into a fixed share per slot
    // (idle slots' shares left empty while a long sequence reads most of its context over the
    // link): the slots share it in whole chunks, the running sequences' first, idle warm ones
    // giving theirs up to the host tier. Only where blocks live changes, never what is read.
    let lend = if !spilled && fixed < slot_blocks && lend_fits(exec, layout) {
        (device / ROUND * ROUND).min(n_seqs * slot_blocks)
    } else {
        0
    };
    exec.set_kv_slot_lend(lend, compute + margin);
    let hbm_slot = if lend > 0 { 0 } else { fixed };
    let n_blocks = n_seqs * slot_blocks;
    let dev = n_seqs * hbm_slot;
    let host = n_blocks - dev;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    let n_dev = layout.tp_devices.len().max(1);
    if hbm_slot == 0 && lend == 0 && !(spilled && matches!(tier, TpTier::Auto(_))) {
        return Err(LoadError::Device(format!(
            "--kv-slots: no room for {ROUND} blocks of KV cache per sequence on the GPUs next to \
             {:.1} GiB of compute buffers per GPU; lower --max-seqs",
            gib(compute)
        )));
    }
    if host > 0 {
        match &tier {
            TpTier::Off => {
                return Err(LoadError::Device(format!(
                    "--kv-slots: {n_seqs} slots of {} tokens need {:.1} GiB of KV cache per GPU, \
                     the GPUs have room for {:.1} GiB, and there is no host tier \
                     (--kv-host-tokens 0 or --kv-cache-tokens without it); lower --ctx-size or \
                     --max-seqs",
                    slot_blocks * KV_BLOCK,
                    gib(n_blocks * per / n_dev),
                    gib(device * per / n_dev),
                )));
            }
            TpTier::Auto(nodes) => {
                // One slot's context on the host at least; past what the host holds, the
                // slots' host blocks are backed as they fill (#1131).
                let held = host_tier::auto_host_blocks(
                    hbm_slot,
                    slot_blocks,
                    host + hbm_slot,
                    hbm_slot,
                    nodes,
                )
                .map_err(LoadError::Device)?;
                if held < host {
                    let cap = lazy_slot_cap(held, n_seqs);
                    if cap < slot_blocks {
                        return Err(LoadError::Device(format!(
                            "--kv-slots: the host has room for {} tokens of KV cache past the \
                             GPUs ({}), short of one {}-token context besides the {n_seqs} \
                             sequences' backing; lower --ctx-size or --max-seqs",
                            held * KV_BLOCK,
                            host_tier::room_note(nodes),
                            slot_blocks * KV_BLOCK
                        )));
                    }
                    exec.set_kv_slot_host_budget(Some(held));
                }
            }
            // Given: the slots, not the flag, size it.
            TpTier::Blocks(_) => {}
        }
    }
    // Shared: their VRAM held as the slots' would be until the tier takes it.
    let held = if lend > 0 { lend } else { dev };
    exec.resize_kv(held.max(1))
        .map_err(|e| LoadError::Device(format!("cannot allocate {held} KV blocks: {e}")))?;
    exec.set_max_seq_tokens(asked + spec_k);
    // Nothing is gathered.
    exec.set_gather_budget(0);
    let [mixed, decode, mtp_decode, mtp_mixed] = last;
    tracing::info!(
        ctx_size = asked,
        slots = n_seqs,
        slot_tokens = slot_blocks * KV_BLOCK,
        device_tokens_per_slot = hbm_slot * KV_BLOCK,
        shared_device_tokens = lend * KV_BLOCK,
        mixed_mib = mixed >> 20,
        decode_mib = decode >> 20,
        mtp_decode_mib = mtp_decode >> 20,
        mtp_mixed_mib = mtp_mixed >> 20,
        compute_estimate_mib = compute >> 20,
        prompt_copy_tokens = exec.kv_slot_stage().2,
        gpus = n_dev,
        host_tier_tokens = host * KV_BLOCK,
        host_backed_tokens = exec.kv_slot_host_budget().unwrap_or(host) * KV_BLOCK,
        "tensor-parallel cache fitted in per-sequence slots"
    );
    Ok((dev, host))
}

/// Whether the sequence slots of a tensor-parallel cache can share VRAM in chunks (#1159):
/// every cache tensor's chunk of [`LAZY_CHUNK_BLOCKS`] blocks on each GPU (its heads split
/// evenly) is whole mapping granules (a q8_0 row's 17/16 of a power of two is not).
#[cfg(feature = "cuda")]
fn lend_fits(exec: &Executor, layout: &layout::Layout) -> bool {
    let n = layout.tp_devices.len().max(1);
    let Some(gran) = layout
        .tp_devices
        .first()
        .and_then(|d| cuda_index(*d))
        .and_then(crate::residency::cuda::tp_tier_granularity)
    else {
        return false;
    };
    let rows = LAZY_CHUNK_BLOCKS * KV_BLOCK;
    let bytes = exec.kv_cell_tensor_bytes();
    !bytes.is_empty()
        && bytes
            .iter()
            .all(|&b| b % n == 0 && (b / n * rows) % gran == 0)
}

#[cfg(not(feature = "cuda"))]
fn lend_fits(_: &Executor, _: &layout::Layout) -> bool {
    false
}

/// A tensor-parallel cache's host tier (#317), for [`fit_tp_cache`].
enum TpTier {
    /// None: the cache is what the GPUs hold.
    Off,
    /// `--kv-host-tokens`, in blocks.
    Blocks(usize),
    /// Automatic: only when one context does not fit in the GPUs, sized as
    /// [`host_tier::auto_host_blocks`] does on these nodes.
    Auto(Vec<host_tier::Node>),
}

/// The tier [`fit_tp_cache`] may give a tensor-parallel cache of `block_bytes` per block (over
/// all the GPUs): as given, or automatic where every GPU's driver maps host NUMA memory.
fn tp_tier(cfg: &LmConfig, layout: &layout::Layout, block_bytes: usize) -> Result<TpTier, String> {
    if let Some(t) = cfg.kv_host_tokens {
        return Ok(if t == 0 {
            TpTier::Off
        } else {
            TpTier::Blocks(t.div_ceil(KV_BLOCK))
        });
    }
    if cfg.kv_cache_tokens.is_some() {
        return Ok(TpTier::Off);
    }
    let n_dev = layout.tp_devices.len().max(1);
    let mut nodes: Vec<host_tier::Node> = Vec::new();
    for d in &layout.tp_devices {
        let (Some(device), true) = (cuda_index(*d), d.kind() == crate::ggml::DeviceKind::Gpu)
        else {
            return Ok(TpTier::Off);
        };
        let Some(node) = tier_node(device) else {
            return Ok(TpTier::Off);
        };
        let bytes = (block_bytes / n_dev) as u64;
        match nodes.iter_mut().find(|n| n.node == node) {
            Some(n) => n.block_bytes += bytes,
            None => {
                let mem = host_tier::node_memory(node).ok_or_else(|| {
                    format!("cannot read NUMA node {node}'s memory for the KV host tier")
                })?;
                nodes.push(host_tier::Node {
                    node,
                    block_bytes: bytes,
                    free: mem.free,
                    cache: mem.cache,
                    total: mem.total,
                });
            }
        }
    }
    Ok(TpTier::Auto(nodes))
}

/// Where the KV cache lives (#352).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum KvMemory {
    /// GPU memory, sized to what fits.
    #[default]
    Device,
    /// CUDA managed memory, which may exceed the GPUs' memory (the driver pages it to and from
    /// host memory), as llama.cpp's `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` runs: a capacity
    /// baseline for the host tier. ggml reads that variable at its first allocation and then
    /// makes every allocation managed (weights too), so it must be set in the environment;
    /// `--kv-cache-tokens` then sizes the cache past the GPUs' memory, under `--tp` too.
    Managed,
}

/// Per-sequence KV slots under tensor parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum KvSlots {
    /// On when the model's attention reads slots and the asked context needs a host tier.
    #[default]
    Auto,
    /// On where the model's attention reads slots.
    On,
    /// Off: attention gathers each step's cells into the compute buffers.
    Off,
}

/// Default [`LmConfig::kv_slot_stage_rows`]: prompt steps only (an MTP verify has `k + 1` rows),
/// and enough rows that flash attention reads the span in several tiles.
pub const KV_SLOT_STAGE_ROWS: usize = 64;

/// The longest span sequence-slot attention copies (#1153), from a compute buffer measured at
/// contexts `c0 < c1` without the copies and with them, each as (bytes at `c1`, growth per
/// position): the span at which the buffer with the copies takes no more than the one without
/// them at the `asked` context it is sized for, in whole blocks, and never below `c0` (0: no
/// copies). The copies then take no room from the KV cache or the rings.
pub(crate) fn slot_stage_span(
    (base, slope): (usize, f64),
    (with, with_slope): (usize, f64),
    (c0, c1): (usize, usize),
    asked: usize,
) -> usize {
    let at = |b: usize, s: f64, ctx: usize| b as f64 + s * (ctx as f64 - c1 as f64);
    let room = at(base, slope, asked);
    let span = if with_slope <= 0.0 {
        if with as f64 <= room { asked } else { 0 }
    } else {
        // The line with the copies meets the room.
        let s = c1 as f64 + (room - with as f64) / with_slope;
        s.clamp(0.0, asked as f64) as usize
    };
    if span < c0 {
        return 0;
    }
    span / KV_BLOCK * KV_BLOCK
}

/// The environment variable ggml-cuda reads for managed allocations.
pub const UNIFIED_MEMORY_ENV: &str = "GGML_CUDA_ENABLE_UNIFIED_MEMORY";

/// Checks `--kv-memory managed` against the environment ggml reads (`value`: the variable's).
fn check_kv_memory(m: KvMemory, value: Option<&str>) -> Result<(), String> {
    match (m, value) {
        (KvMemory::Device, _) | (KvMemory::Managed, Some("1")) => Ok(()),
        (KvMemory::Managed, _) => Err(format!(
            "--kv-memory managed needs {UNIFIED_MEMORY_ENV}=1 in the environment (ggml-cuda \
             reads it at its first allocation); start legion with it set"
        )),
    }
}

/// `--kv-memory managed` with `--kv-cache-tokens` under tensor parallelism: that many cache
/// tokens (at least one full context), in managed memory that may exceed the GPUs' memory, so
/// no fit against the compute buffers; the context stays as asked.
fn managed_tp_cache(
    exec: &mut Executor,
    cfg: &LmConfig,
    tokens: usize,
    spec_k: usize,
) -> Result<usize, LoadError> {
    let n_blocks = tokens
        .div_ceil(KV_BLOCK)
        .max((cfg.ctx_size + spec_k).div_ceil(KV_BLOCK));
    exec.resize_kv(n_blocks).map_err(|e| {
        LoadError::Device(format!("cannot allocate {n_blocks} managed KV blocks: {e}"))
    })?;
    exec.set_max_seq_tokens(cfg.ctx_size + spec_k);
    exec.set_gather_budget(fit::budget(cfg.ctx_size, spec_k));
    tracing::info!(
        ctx_size = cfg.ctx_size,
        kv_cache_tokens = n_blocks * KV_BLOCK,
        kv_cache_gib = (n_blocks * exec.kv_block_bytes()) as f64 / (1u64 << 30) as f64,
        "tensor-parallel cache in managed memory (not fitted: the driver pages it)"
    );
    Ok(n_blocks)
}

/// The host tiers of a layout's KV pools (#352): per stage, its GPU's ordinal and NUMA node,
/// the first `hbm_blocks` blocks in device memory. A tensor-parallel cache's one tier, over a
/// meta device, keeps the first `slot.0` blocks of each `slot.1`-block sequence slot in VRAM
/// (`slot.1` 0: no slots, the first `slot.0` of the whole cache), backed as they fill with
/// `slot.2`, sharing `slot.3` more VRAM blocks (#1159).
fn kv_tiers(
    layout: &layout::Layout,
    hp: &crate::model::core::Hparams,
    hbm_blocks: usize,
    stage_blocks: usize,
    slot: (usize, usize, bool, usize),
) -> Result<Vec<Option<crate::model::core::KvTier>>, String> {
    if !layout.tp_devices.is_empty() {
        let devices = layout
            .tp_devices
            .iter()
            .map(|d| cuda_index(*d).ok_or("--kv-host-tokens needs CUDA GPUs"))
            .collect::<Result<Vec<_>, _>>()?;
        return tp_tier_for(hp, &devices, hbm_blocks, slot).map(|t| vec![Some(t)]);
    }
    layout
        .stages
        .iter()
        .map(|(d, _)| {
            let d = d.ok_or("--kv-host-tokens needs a GPU")?;
            let device = cuda_index(d).ok_or("--kv-host-tokens needs CUDA GPUs")?;
            tier_for(device, hbm_blocks, stage_blocks).map(Some)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn tier_for(
    device: usize,
    hbm_blocks: usize,
    stage_blocks: usize,
) -> Result<crate::model::core::KvTier, String> {
    let caps = crate::residency::cuda::caps(device).ok_or("cannot query the GPU")?;
    // A GPU with no host NUMA node: host memory of no node (`Caps::host_node`).
    let Some(node) = caps.host_node.filter(|_| caps.host_numa_vmm) else {
        return Err(format!(
            "--kv-host-tokens: CUDA{device}'s driver cannot map host NUMA memory (cuMemCreate)"
        ));
    };
    Ok(crate::model::core::KvTier {
        device,
        node,
        hbm_blocks,
        stage_blocks,
        split: None,
        alias: None,
    })
}

#[cfg(not(feature = "cuda"))]
fn tier_for(_: usize, _: usize, _: usize) -> Result<crate::model::core::KvTier, String> {
    Err("--kv-host-tokens needs CUDA".into())
}

/// A tensor-parallel cache's tier over CUDA devices `devices` (#317): a meta device split as
/// the model's, whose buffer type keeps each GPU's first `hbm_blocks` blocks in its VRAM and
/// the rest on its host node; with sequence slots of `slot.1` blocks, each slot's first
/// `slot.0` ([`kv_tiers`]), its host blocks backed as it fills with `slot.2`
/// ([`Executor::set_kv_slot_host_budget`]). Prefill reads the host blocks in place (no staging slots: tensor
/// parallelism's attention gathers or reads slots).
#[cfg(feature = "cuda")]
fn tp_tier_for(
    hp: &crate::model::core::Hparams,
    devices: &[usize],
    hbm_blocks: usize,
    (slot_hbm, slot_blocks, lazy, lend): (usize, usize, bool, usize),
) -> Result<crate::model::core::KvTier, String> {
    let split = |alias| -> Result<crate::model::core::SplitTier, String> {
        let tiered = crate::residency::cuda::tp_tier_devices(
            devices,
            slot_hbm * KV_BLOCK,
            slot_blocks * KV_BLOCK,
            alias,
            lazy,
            lend / LAZY_CHUNK_BLOCKS,
        )
        .map_err(|e| format!("--kv-host-tokens: {e}"))?;
        let meta = crate::ggml::meta::meta_device(
            &tiered,
            crate::model::core::tp::megatron_rule(hp, devices.len()),
        )
        .ok_or("cannot create the tensor-parallel KV tier's device")?;
        Ok(crate::model::core::SplitTier(meta.buffer_type()))
    };
    let node = tier_node(devices[0]).ok_or("--kv-host-tokens: no host NUMA node")?;
    Ok(crate::model::core::KvTier {
        device: devices[0],
        node,
        hbm_blocks,
        stage_blocks: 0,
        split: Some(split(false)?),
        // Slots: the MTP head's views through an alias (see `tp_tier_devices`).
        alias: if slot_blocks > 0 {
            Some(split(true)?)
        } else {
            None
        },
    })
}

#[cfg(not(feature = "cuda"))]
fn tp_tier_for(
    _: &crate::model::core::Hparams,
    _: &[usize],
    _: usize,
    _: (usize, usize, bool, usize),
) -> Result<crate::model::core::KvTier, String> {
    Err("--kv-host-tokens needs CUDA".into())
}

/// The automatic host tier's tokens (`--kv-host-tokens` not given): 0 unless a context does
/// not fit in GPU memory (`fit_full` blocks, before the expert cache's share) on a layer split
/// or one discrete CUDA GPU with an automatic pool; then the rest of the automatic pool (`want`)
/// past the `n_blocks` the device holds, capped by the GPUs' NUMA nodes' free memory. Where the
/// tier cannot be had (no host NUMA memory through VMM), 0: the load goes on as without it
/// (and fails as before when a native context does not fit).
#[allow(clippy::too_many_arguments)]
fn auto_host_tokens(
    layout: &layout::Layout,
    cfg: &LmConfig,
    hp: &crate::model::core::Hparams,
    (fit, fit_full): (Option<usize>, Option<usize>),
    (ctx_blocks, n_blocks): (usize, usize),
    want: usize,
    (ring, types, shared): (bool, crate::model::core::kv_type::KvTypes, bool),
) -> Result<usize, String> {
    let (Some(_), Some(full)) = (fit, fit_full) else {
        return Ok(0);
    };
    if cfg.tp > 1 || shared || cfg.kv_cache_tokens.is_some() || full >= ctx_blocks {
        return Ok(0);
    }
    // Per node: the bytes a host block takes over the stages whose GPU is on it.
    let mut nodes: Vec<host_tier::Node> = Vec::new();
    for (d, layers) in &layout.stages {
        let Some(device) = d.and_then(cuda_index) else {
            return Ok(0);
        };
        let Some(node) = tier_node(device) else {
            tracing::warn!(
                "the {}-token context does not fit in CUDA{device}'s memory, and its driver                  cannot map host NUMA memory for a KV host tier",
                cfg.ctx_size
            );
            return Ok(0);
        };
        let bytes = KvPool::block_bytes_paged(hp, layers.clone(), ring, types) as u64;
        // `--tp-per-gpu`: the other rank pins as much for each block, on the same node when
        // its GPU is there.
        let bytes = bytes * pair_ranks_on(cfg.tp_pair.as_ref(), node);
        match nodes.iter_mut().find(|n| n.node == node) {
            Some(n) => n.block_bytes += bytes,
            None => {
                let mem = host_tier::node_memory(node).ok_or_else(|| {
                    format!("cannot read NUMA node {node}'s memory for the KV host tier")
                })?;
                nodes.push(host_tier::Node {
                    node,
                    block_bytes: bytes,
                    free: mem.free,
                    cache: mem.cache,
                    total: mem.total,
                });
            }
        }
    }
    let blocks = host_tier::auto_host_blocks(full, ctx_blocks, want, n_blocks, &nodes)?;
    Ok(blocks * KV_BLOCK)
}

/// The `--tp-per-gpu` ranks whose KV host tiers would sit on NUMA node `node`, this one's
/// included (1 without the pair): each rank sizes its tier alone, from the same free memory.
fn pair_ranks_on(pair: Option<&tp_pair::TpPairRank>, node: u32) -> u64 {
    let Some(p) = pair else { return 1 };
    let other = (p.cuda - p.rank) + (1 - p.rank);
    1 + u64::from(tier_node(other) == Some(node))
}

/// The NUMA node a KV host tier of CUDA device `device` would use, if the driver can map one.
#[cfg(feature = "cuda")]
fn tier_node(device: usize) -> Option<u32> {
    let caps = crate::residency::cuda::caps(device)?;
    caps.host_node.filter(|_| caps.host_numa_vmm)
}

#[cfg(not(feature = "cuda"))]
fn tier_node(_: usize) -> Option<u32> {
    None
}

/// Whether the layout's GPUs reach host memory over NVLink (they use the host's page tables:
/// an AC922, a Grace Hopper), not PCIe.
#[cfg(feature = "cuda")]
fn tier_link_nvlink(layout: &layout::Layout) -> bool {
    let nvlink = |d: Option<crate::ggml::Device>| {
        d.and_then(cuda_index)
            .and_then(crate::residency::cuda::caps)
            .is_some_and(|c| c.pageable_uses_host_page_tables)
    };
    // Tensor parallelism: the GPUs under the meta device (which has no CUDA index).
    if !layout.tp_devices.is_empty() {
        return layout.tp_devices.iter().all(|&d| nvlink(Some(d)));
    }
    layout.stages.iter().all(|(d, _)| nvlink(*d))
}

#[cfg(not(feature = "cuda"))]
fn tier_link_nvlink(_: &layout::Layout) -> bool {
    false
}

/// Per stage, where its rings are saved ([`Executor::enable_ring_saves`]): pinned memory on
/// its GPUs' NUMA node, or plain host memory where this driver cannot place it.
#[cfg(feature = "cuda")]
fn ring_save_bufts(layout: &layout::Layout) -> Vec<crate::ggml::BufferType> {
    let pinned = |devs: Vec<usize>| {
        let node = (devs.first().copied())
            .and_then(crate::residency::cuda::caps)
            .map(|c| c.host_node.unwrap_or(c.host_numa_id.max(0) as u32));
        node.and_then(|n| crate::residency::cuda::host_arena(n, &devs).ok())
            .unwrap_or(layout.host)
    };
    if !layout.tp_devices.is_empty() {
        let devs = layout
            .tp_devices
            .iter()
            .filter_map(|&d| cuda_index(d))
            .collect();
        return vec![pinned(devs)];
    }
    (layout.stages.iter())
        .map(|(d, _)| {
            d.and_then(cuda_index)
                .map_or(layout.host, |i| pinned(vec![i]))
        })
        .collect()
}

#[cfg(not(feature = "cuda"))]
fn ring_save_bufts(layout: &layout::Layout) -> Vec<crate::ggml::BufferType> {
    layout.stages.iter().map(|_| layout.host).collect()
}

/// Free memory on the NUMA node of the first GPU's ring copies ([`ring_save_bufts`]), if known.
#[cfg(feature = "cuda")]
fn ring_save_free(layout: &layout::Layout) -> Option<u64> {
    let dev = (layout.tp_devices.first().copied())
        .or_else(|| layout.stages.first().and_then(|(d, _)| *d))
        .and_then(cuda_index)?;
    let caps = crate::residency::cuda::caps(dev)?;
    let node = u32::try_from(caps.host_numa_id).ok().or(caps.host_node)?;
    host_tier::node_memory(node).map(|m| m.free)
}

#[cfg(not(feature = "cuda"))]
fn ring_save_free(_: &layout::Layout) -> Option<u64> {
    None
}

/// Ring checkpoints kept by default per ring ([`RingCheckpoints`]), memory permitting.
pub const RING_CHECKPOINTS_PER_RING: usize = 3;

/// Checkpoint slots for `rings` rings of `bytes` each: `given` (`--ring-checkpoints`; 0, off:
/// one per ring, the single copy at a prompt's end of [`RingCheckpoints::single`]), else
/// [`RING_CHECKPOINTS_PER_RING`] per ring where a third of the host memory left (`free`)
/// holds them, never fewer than one per ring (a warm sequence's copy at its last prompt's end);
/// one per ring where the free memory is unknown (an integrated GPU's memory is its KV
/// cache's too).
pub fn ring_checkpoint_slots(
    given: Option<usize>,
    rings: usize,
    bytes: usize,
    free: Option<u64>,
) -> usize {
    if let Some(n) = given {
        return if n == 0 { rings } else { n };
    }
    let fit = free.map_or(0, |f| (f / 3) as usize / bytes.max(1));
    fit.min(RING_CHECKPOINTS_PER_RING * rings).max(rings)
}

/// A host tier backed as its sequences grow (#1131) counted the node's free memory at load,
/// which pinned checkpoints then take: `slots` checkpoints of `bytes` each, of which those
/// past one per ring (`rings`, the old single save) come out of the tier's `budget` blocks of
/// `block_bytes`, never below `keep` blocks (one context besides the sequences' backing),
/// fewer checkpoints if they do not fit. The slots and the tier's new budget.
pub fn ring_checkpoints_in_budget(
    slots: usize,
    rings: usize,
    bytes: usize,
    budget: Option<usize>,
    block_bytes: usize,
    keep: usize,
) -> (usize, Option<usize>) {
    let Some(held) = budget else {
        return (slots, None);
    };
    let spare = held.saturating_sub(keep) * block_bytes;
    let extra = slots.saturating_sub(rings).min(spare / bytes.max(1));
    let slots = slots.min(rings) + extra;
    let held = held - (extra * bytes).div_ceil(block_bytes.max(1));
    (slots, Some(held))
}

/// Under tensor parallelism each GPU gathers its share of a cache row (its KV heads) by cells
/// with ggml's `get_rows`, whose CUDA kernel reads IQ4_NL in 256-value super-blocks: every
/// attention layer's per-GPU row must be a multiple of 256 values for an IQ4_NL cache.
fn tp_cache_rows_fit(
    hp: &crate::model::core::Hparams,
    types: crate::model::core::KvTypes,
    tp: usize,
) -> Result<(), String> {
    use crate::model::core::KvType;
    if types.k != KvType::Iq4Nl && types.v != KvType::Iq4Nl {
        return Ok(());
    }
    for il in (0..hp.n_layer).filter(|&il| hp.owns_kv(il)) {
        let (d, n_kv) = hp.kv_dims(il);
        // A compact cache's granules are wider (#551).
        let d = d + hp.compact_kv(il);
        // Uneven KV heads stay whole on every GPU (`tp::uneven_kv_layers`).
        let row = if n_kv % tp as i64 == 0 {
            d * n_kv / tp as i64
        } else {
            d * n_kv
        };
        if row % 256 != 0 {
            return Err(format!(
                "--cache-type iq4_nl with --tp {tp}: layer {il}'s cache row per GPU is {row} \
                 values ({n_kv} KV heads of {d}), and the CUDA gather reads IQ4_NL in 256-value \
                 super-blocks; pick another cache type (q4_0 is the same size)"
            ));
        }
    }
    Ok(())
}

/// `--gemma4-compact-global-kv`: a Gemma 4 model's compact caches
/// ([`Model::set_gemma4_compact_global_kv`]), and a line on what they take per token, or that
/// the model has none (any other model: one line, nothing else changes).
fn gemma4_compact_global_kv(model: &mut Model, types: crate::model::core::KvTypes) {
    let plain: usize = global_kv_layers(&model.hp)
        .map(|il| model.hp.kv_cell_bytes(il, types))
        .sum();
    let layers = model.set_gemma4_compact_global_kv();
    if layers.is_empty() {
        tracing::info!(
            arch = model.hp.arch.name(),
            "--gemma4-compact-global-kv: not a Gemma 4 model with global layers without a value \
             projection, plain KV cache"
        );
        return;
    }
    let hp = &model.hp;
    let compact: usize = global_kv_layers(hp)
        .map(|il| hp.kv_cell_bytes(il, types))
        .sum();
    let all: usize = (0..hp.n_layer)
        .filter(|&il| hp.owns_kv(il))
        .map(|il| hp.kv_cell_bytes(il, types))
        .sum();
    tracing::info!(
        ?layers,
        rotated_dims = hp.compact_kv(layers[0]),
        global_bytes_per_token = compact,
        plain_global_bytes_per_token = plain,
        all_bytes_per_token = all,
        "Gemma 4 compact global KV cache"
    );
}

/// The global (not sliding-window) layers of `hp` with a cache of their own.
fn global_kv_layers(hp: &crate::model::core::Hparams) -> impl Iterator<Item = usize> + '_ {
    (0..hp.n_layer).filter(|&il| hp.owns_kv(il) && !hp.sliding_layer(il))
}

/// [`LmConfig::ring_step`], given whether the first stage's GPU shares memory with the host.
fn ring_step(cfg: &LmConfig, integrated: bool) -> usize {
    cfg.ring_step
        .unwrap_or(if integrated {
            cfg.max_batch_tokens
        } else {
            cfg.prefill_chunk
        })
        .max(1)
}

/// The longest ring step at which `fits`: `step`, else halved (whole KV blocks) down to
/// `floor`; `step` itself when none fits (the KV fit then refuses the load as before).
pub(crate) fn fitting_ring_step(step: usize, floor: usize, fits: impl Fn(usize) -> bool) -> usize {
    let mut s = step;
    loop {
        if fits(s) {
            return s;
        }
        let next = (s / 2 / KV_BLOCK * KV_BLOCK).max(floor);
        if next >= s {
            return step;
        }
        s = next;
    }
}

/// Whether the sliding-window layers of `hp` keep a ring of their window per sequence: with
/// paged attention (`paged`, or an architecture whose attention always reads in place), rings
/// for gather attention or a quantized cache, unless `--swa-full`. A model with a per-sequence
/// state as well (DeepSeek-V4.1's token history) resumes no cached prefix with rings (neither
/// the radix tree nor warm sequences): there only without prefix caching, or with `--swa-ring`.
pub(crate) fn use_rings(hp: &crate::model::core::Hparams, cfg: &LmConfig, paged: bool) -> bool {
    let prefixes_kept = !hp.has_slot_state() || cfg.prefix_cache_states == 0 || cfg.swa_ring;
    hp.arch.swa_window() > 0
        && (paged || hp.arch.always_paged() || cfg.ring_gather || !cfg.cache_types.is_f16())
        && !cfg.swa_full
        && prefixes_kept
}

/// The CUDA indices of the layout's GPUs: its stages', or under tensor parallelism the meta
/// device's.
#[cfg(feature = "cuda")]
fn layout_cuda(layout: &layout::Layout) -> Vec<usize> {
    let mut out: Vec<usize> = (layout.stages.iter().filter_map(|(d, _)| *d))
        .chain(layout.tp_devices.iter().copied())
        .filter_map(cuda_index)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Routed experts the GPUs read in place from plain host memory (the model file's mapping or a
/// CPU buffer, with no pinned expert arena): where every GPU reads pageable memory (an AC922's
/// ATS, the GB10, HMM), nothing to do, as before; else the expert ranges are registered with
/// the driver for the GPUs to read at the same addresses ([`fallback::host_expert_read`]),
/// kept with the model, or the load stops with what to do instead.
#[cfg(feature = "cuda")]
fn register_host_experts(model: &mut Model, layout: &layout::Layout) -> Result<(), LoadError> {
    use crate::residency::fallback::{self, HostExpertRead};
    let gpus = layout_cuda(layout);
    let caps: Vec<_> = gpus
        .iter()
        .filter_map(|&d| crate::residency::cuda::caps(d))
        .collect();
    if caps.is_empty() || caps.len() < gpus.len() {
        return Ok(());
    }
    let read = fallback::host_expert_read(
        caps.iter().all(|c| c.pageable_access),
        // The kernels take the tensors' host pointers: the registered memory must be read there.
        caps.iter()
            .all(|c| c.host_register && c.host_pointer_registered),
        caps.iter().all(|c| c.read_only_register),
    );
    if read == HostExpertRead::InPlace {
        return Ok(());
    }
    // The routed experts in host memory outside a pinned arena, with the padding past their
    // last row the kernels read ([`crate::ggml::mapped_tail_padding`]).
    let host_experts = |w: &crate::weights::Weights| -> Vec<(usize, usize)> {
        w.tensors()
            .filter(|&t| {
                t.name().contains("_exps.")
                    && t.is_host()
                    && crate::residency::arena_node_of(t).is_none()
            })
            .map(|t| {
                // SAFETY: reading the data pointer and the live buffer of an allocated host
                // tensor.
                let (p, end) = unsafe {
                    let buf = (*t.raw()).buffer;
                    let base = crate::ggml::ffi::ggml_backend_buffer_get_base(buf) as usize;
                    let size = crate::ggml::ffi::ggml_backend_buffer_get_size(buf);
                    ((*t.raw()).data as usize, base + size)
                };
                // Never past the buffer (a mapped file's tensors keep their padding inside it).
                let pad = crate::ggml::mapped_tail_padding(t.ty(), t.ne()[0]);
                (p, (t.nbytes() + pad).min(end.saturating_sub(p)))
            })
            .filter(|&(p, n)| p != 0 && n > 0)
            .collect()
    };
    let mut ranges = host_experts(model.weights());
    if let Some(w) = model.mtp_weights() {
        ranges.extend(host_experts(w));
    }
    if ranges.is_empty() {
        return Ok(());
    }
    let gib = ranges.iter().map(|r| r.1).sum::<usize>() as f64 / (1u64 << 30) as f64;
    let stop = |why: &str| {
        LoadError::Device(format!(
            "{gib:.1} GiB of routed experts are in host memory, and {why}: fit them in VRAM \
             (--experts device, with more GPUs or a smaller quant), or run on a driver and kernel \
             with HMM (pageable memory access)"
        ))
    };
    let read_only = match read {
        HostExpertRead::InPlace => return Ok(()),
        HostExpertRead::Refuse(why) => return Err(stop(why)),
        HostExpertRead::Register { read_only } => read_only,
    };
    let start = Instant::now();
    let pins = crate::residency::cuda::register_host(gpus[0], &ranges, read_only)
        .map_err(|e| stop(&format!("registering them with the driver failed ({e})")))?;
    tracing::info!(
        gib,
        read_only,
        seconds = start.elapsed().as_secs_f64(),
        "routed experts in host memory: the GPUs cannot read pageable memory (no ATS or HMM), \
         so they are registered with the driver (cuMemHostRegister) and read in place through it{}",
        if read_only {
            ""
        } else {
            " (not read-only on this driver: the mapped file's pages are copied as they pin)"
        }
    );
    model.keep_host_pins(Box::new(pins));
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn register_host_experts(_: &mut Model, _: &layout::Layout) -> Result<(), LoadError> {
    Ok(())
}

/// The host arenas for the routed experts of a model on CUDA0..`n_gpus`, one per stage: pinned
/// memory on the stage GPU's NUMA node, readable by every GPU of that node (`(buffer type,
/// node)`), so each GPU reads its own layers' experts from its own socket's memory. Without
/// them (no CUDA build, or a driver that cannot place host memory through VMM) the experts
/// stay mapped from the file, and the cache mode falls back to reading them in place.
fn expert_arenas(
    mode: &mut ExpertMode,
    n_gpus: usize,
    split: &mut Option<usize>,
    shares: &[u32],
) -> Option<Vec<(crate::ggml::BufferType, u32)>> {
    #[cfg(feature = "cuda")]
    {
        let nodes: Vec<u32> = (0..n_gpus.max(1))
            .map(|d| {
                crate::residency::cuda::caps(d).map_or(0, |c| {
                    // No host NUMA node: host memory of no node (`Caps::host_node`).
                    c.host_node.unwrap_or(c.host_numa_id.max(0) as u32)
                })
            })
            .collect();
        // Expert parallelism (`--glm-parallel-decode`): one arena every GPU maps, each expert
        // on the node of the GPU that owns it.
        if let Some(n_expert) = *split {
            // The plan's bounds (`glm_par_plan`).
            let bounds =
                crate::model::glm5next::par::expert_bounds_shared(n_expert, nodes.len(), shares);
            let by_expert: Vec<u32> = (0..n_expert as u32)
                .map(|e| nodes[crate::model::glm5next::par::expert_owner(&bounds, e)])
                .collect();
            let devs: Vec<usize> = (0..nodes.len()).collect();
            match crate::residency::cuda::host_arena_split(&by_expert, &devs) {
                Ok(b) => {
                    tracing::info!(
                        ?nodes,
                        "expert arena split by owner: each GPU's experts on its own node"
                    );
                    return Some(nodes.iter().map(|&n| (b, n)).collect());
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no split host arena: no parallel decode");
                    *split = None;
                }
            }
        }
        let made: Result<Vec<_>, String> = nodes
            .iter()
            .map(|&node| {
                let devs: Vec<usize> = (0..nodes.len()).filter(|&d| nodes[d] == node).collect();
                crate::residency::cuda::host_arena(node, &devs).map(|b| (b, node))
            })
            .collect();
        match made {
            Ok(a) => return Some(a),
            Err(e) => tracing::warn!(error = %e, "no pinned host arena for the experts"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (n_gpus, shares);
        *split = None;
    }
    if *mode == ExpertMode::Cache {
        tracing::warn!("the expert cache needs the host arena: reading the experts in place");
        *mode = ExpertMode::Host;
    }
    None
}

/// `--glm-expert-split` for the MoE layers `moe` of `gguf` under `cfg`: interleaved only where
/// the renumbering changes nothing else ([`crate::model::glm5next::par::assign`]). An expert
/// profile and `--glm-prune` name experts by their file ids and `--weights-fp16-gates` converts
/// the router's rows as the file has them, so with any of them, or a layer whose experts cannot
/// be renumbered, the split stays contiguous (said in the log).
fn glm_expert_split(
    gguf: &GgufFile,
    cfg: &LmConfig,
    moe: std::ops::Range<usize>,
) -> crate::model::glm5next::par::assign::ExpertSplit {
    use crate::model::glm5next::par::assign::{ExpertSplit, check_layer};
    if cfg.glm_expert_split == ExpertSplit::Contiguous {
        return ExpertSplit::Contiguous;
    }
    let why = if cfg.residency.profile.is_some() {
        Some("an expert profile names the file's experts".to_string())
    } else if cfg.glm_prune.is_some() {
        Some("--glm-prune names the file's experts".to_string())
    } else if cfg.fp16.mode != crate::weights::WeightsFp16::Off && cfg.fp16.gates {
        Some("--weights-fp16-gates converts the router as the file has it".to_string())
    } else {
        moe.into_iter().find_map(|il| check_layer(gguf, il).err())
    };
    match why {
        Some(why) => {
            tracing::warn!(%why, "--glm-expert-split interleave: keeping the contiguous split");
            ExpertSplit::Contiguous
        }
        None => cfg.glm_expert_split,
    }
}

/// GLM's parallel steps' plan over the layout's GPUs (stage `d` on GPU `d`, which keeps the
/// stage's layers' mirrored weights under their own names) and the tensors it loads instead of
/// the ones it splits.
fn glm_par_plan(
    gguf: &GgufFile,
    layout: &layout::Layout,
    cfg: &LmConfig,
) -> Result<
    (
        crate::model::glm5next::par::ParPlan,
        crate::model::glm5next::par::TpTensors,
    ),
    LoadError,
> {
    use crate::model::glm5next::par::{ParPlan, exchange::MAX_DEV};
    let hp = crate::model::core::Hparams::from_gguf(gguf)?;
    let p = crate::model::glm5next::params(&hp).ok_or_else(|| {
        LoadError::Device("--glm-parallel-decode: not a GLM-5.3-Flash model".into())
    })?;
    let n_dev = layout.stages.len();
    if n_dev > MAX_DEV {
        return Err(LoadError::Device(format!(
            "--glm-parallel-decode: {n_dev} GPUs, at most {MAX_DEV}"
        )));
    }
    let owner: Vec<usize> = (0..hp.n_layer).map(|il| layout.stage_of(il)).collect();
    let split = glm_expert_split(gguf, cfg, p.n_dense..hp.n_layer);
    let shares = &cfg.glm_expert_shares;
    if !shares.is_empty() && shares.len() != n_dev {
        return Err(LoadError::Device(format!(
            "--glm-expert-shares: {} shares for {n_dev} GPUs",
            shares.len()
        )));
    }
    if !shares.is_empty() && split != crate::model::glm5next::par::assign::ExpertSplit::Contiguous {
        return Err(LoadError::Device(
            "--glm-expert-shares needs --glm-expert-split contiguous".into(),
        ));
    }
    let plan = ParPlan::new(n_dev, owner, hp.n_expert as usize)
        .share_experts(shares)
        .split_experts(split);
    // `--glm-parallel-mtp` (#717): with the head drafting, its layer in the file and the
    // experts in the file's order (the head's layer is not renumbered).
    let head_layer = gguf
        .tensor(&format!("blk.{}.ffn_gate_inp.weight", hp.n_layer))
        .is_some();
    let mtp = cfg.glm_parallel_mtp && drafts_with_mtp_head(&cfg.spec) && head_layer;
    if cfg.glm_parallel_mtp && !mtp {
        tracing::warn!("--glm-parallel-mtp ignored: the MTP head is not drafting");
    } else if mtp && plan.order.is_some() {
        tracing::warn!("--glm-parallel-mtp ignored with --glm-expert-split interleave");
    }
    let interleaved = plan.order.is_some();
    let plan = plan
        .split_mtp(mtp && !interleaved)
        .balance_pairs(cfg.glm_pair_balance)
        .predict_next(cfg.glm_prefetch_stats)
        .balance_cross(cfg.glm_cross_balance)
        .prefetch(cfg.glm_prefetch)
        .split_pairs(cfg.glm_pair_split);
    let err = |e: String| LoadError::Device(format!("--glm-parallel-decode: {e}"));
    plan.check(&hp, p).map_err(err)?;
    let tensors = plan.tensors(gguf, p).map_err(err)?;
    Ok((plan, tensors))
}

/// `--glm-parallel-decode`'s steps on `cfg`: every step is a parallel one (no pipeline rounds,
/// no KV host tier) of at most [`MAX_TOKENS`](crate::model::glm5next::par::MAX_TOKENS) tokens,
/// and a prompt of [`GLM_SHORT_PROMPT`] tokens or more borrows the expert cache for staging
/// (each GPU copies its own uncached experts of a layer into VRAM before the layer's matmuls).
/// With prompt steps bigger than that (`--glm-prompt-step`, #686) it borrows the cache as a
/// long prompt ([`LongPrompts`]): steps of up to that many tokens, for which the startup sizes
/// what a lend frees (its measured step buffer); else it keeps its chunks (the short-prompt
/// lend), as before #686.
pub(crate) fn glm_par_steps(cfg: &mut LmConfig) {
    use crate::model::glm5next::par::{MAX_TOKENS, step_tokens};
    let max = MAX_TOKENS;
    cfg.max_batch_tokens = cfg.max_batch_tokens.min(max);
    cfg.prefill_chunk = cfg.prefill_chunk.min(max);
    cfg.pipeline_micro = 0;
    cfg.pipeline_prompt = 0;
    cfg.kv_host_tokens = Some(0);
    cfg.long_prompts.short_min_tokens = Some(GLM_SHORT_PROMPT);
    let step = step_tokens(cfg.glm_prompt_step);
    if step > max {
        cfg.long_prompts.min_tokens = GLM_SHORT_PROMPT;
        cfg.long_prompts.batch_tokens = step;
        cfg.long_prompts.alone_tokens = step;
        cfg.long_prompts.micro = step;
    } else {
        cfg.long_prompts.min_tokens = usize::MAX;
        cfg.long_prompts.micro = cfg.long_prompts.micro.min(max);
    }
}

/// Hands the model its parallel decode plan and the exchange between its GPUs (inboxes in
/// each GPU's own node's pinned memory, slots for steps of up to `max_tokens` tokens).
fn set_glm_par(
    model: &mut Model,
    layout: &layout::Layout,
    plan: crate::model::glm5next::par::ParPlan,
    max_out: usize,
    peers: crate::model::glm5next::par::exchange::PeerWrites,
    max_tokens: usize,
) -> Result<(), LoadError> {
    #[cfg(feature = "cuda")]
    {
        let hp = &model.hp;
        let p = crate::model::glm5next::params(hp)
            .ok_or_else(|| LoadError::Device("--glm-parallel-decode: not GLM".into()))?;
        let nodes: Vec<u32> = layout
            .arenas
            .as_ref()
            .map(|a| a.iter().map(|&(_, n)| n).collect())
            .unwrap_or_else(|| vec![0; plan.n_dev]);
        let slot = crate::model::glm5next::par::slot_bytes(
            hp.n_embd, p.hc, hp.n_vocab, plan.n_dev, max_tokens, max_out,
        );
        // A step of `MAX_TOKENS`'s reduce goes to a peer's VRAM; a longer prompt step's take
        // the host inboxes (VRAM stays with the expert cache).
        let peer_slot = crate::model::glm5next::par::peer_slot_bytes(
            hp.n_embd,
            crate::model::glm5next::par::MAX_TOKENS,
        );
        let ex = crate::model::glm5next::par::exchange::Exchange::pinned(
            &nodes,
            slot,
            &plan.bounds,
            peers,
            peer_slot,
        )
        .map_err(|e| LoadError::Device(format!("--glm-parallel-decode: {e}")))?;
        tracing::info!(
            gpus = plan.n_dev,
            ?nodes,
            bounds = ?plan.bounds,
            experts = if plan.order.is_some() { "interleaved" } else { "contiguous" },
            slot_kib = slot / 1024,
            max_tokens,
            ?peers,
            peer_writer_masks = ?ex.peer_mask(),
            peer_slot_kib = peer_slot / 1024,
            "GLM parallel steps: attention heads, FFN rows and experts split over the GPUs"
        );
        model.set_glm_par(Some(crate::model::glm5next::par::GlmPar {
            plan,
            ex,
            max_tokens,
        }));
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model, layout, plan, max_out, peers, max_tokens);
        Err(LoadError::Device(
            "--glm-parallel-decode needs a CUDA build".into(),
        ))
    }
}

/// Opens an architecture's per-layer n-gram table (`per_layer_token_embd`, gathered on the CPU every
/// step) as a lookup-table store ([`crate::residency::lookup_table`]): locked in RAM, mapped, or
/// read from the SSD (`--ngram-io`, `auto` by default, decided now that the experts' host memory
/// is taken). Its RAM and threads go to the NUMA node of stage 0's GPU, which its rows feed. It
/// never goes to VRAM (the owner's rule: VRAM is for expert slots).
///
/// The rows reach the GPU as a step input, copied on the engine thread's own stream before the
/// step's graph runs: not behind the expert cache's copies, which have a stream of their own.
fn open_ngram_table(
    model: &mut Model,
    gguf: &GgufFile,
    layout: &layout::Layout,
    topo: Option<&crate::hw::numa::Topology>,
    opts: &crate::residency::lookup_table::TableOptions,
) -> Result<Option<report::NgramPlacement>, LoadError> {
    if !model.hp.arch.ngram_table() {
        return Ok(None);
    }
    let Some(file) = Model::ngram_table_info(gguf) else {
        return Ok(None);
    };
    open_lookup_table(model, vec![file], layout, topo, opts, "n-gram table").map(Some)
}

/// Opens the lookup table over `parts` ([`crate::residency::lookup_table::LookupTable::open_parts`])
/// as [`open_ngram_table`] does and reads the model's n-gram rows through it: qwen4exp's
/// per-layer n-gram table, or DeepSeek-V4.1-Flash's Engram tables (`what` names it in the log).
fn open_lookup_table(
    model: &mut Model,
    parts: Vec<crate::residency::lookup_table::TableFile>,
    layout: &layout::Layout,
    topo: Option<&crate::hw::numa::Topology>,
    opts: &crate::residency::lookup_table::TableOptions,
    what: &str,
) -> Result<report::NgramPlacement, LoadError> {
    let bytes: u64 = parts.iter().map(|p| p.bytes()).sum();
    let node = layout
        .arenas
        .as_ref()
        .and_then(|a| a.first().and_then(|&(_, n)| known_node(n)));
    let gpu = layout.stages[0].0;
    #[cfg(feature = "cuda")]
    let node = node.or_else(|| {
        gpu.and_then(cuda_index)
            .and_then(crate::residency::cuda::caps)
            .and_then(|c| u32::try_from(c.host_numa_id).ok())
    });
    let mut opts = opts.clone();
    opts.node = opts.node.or(node);
    // Only a GPU that shares the host's memory (a CPU run has no GPU allocations to come).
    opts.unified |= gpu.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::IntegratedGpu);
    let gib = bytes as f64 / (1u64 << 30) as f64;
    let table = crate::residency::lookup_table::LookupTable::open_parts(parts, &opts, topo)
        .map_err(|e| LoadError::Device(format!("{what}: {e}")))?;
    let c = table.choice();
    tracing::info!(
        io = ?c.io,
        reason = %c.reason,
        gib,
        node = opts.node,
        locked = c.locked,
        ram_cache_gib = c.ram_cache_bytes as f64 / (1u64 << 30) as f64,
        hot_rows = c.hot_rows,
        seconds = c.seconds,
        "{what}"
    );
    let placement = report::NgramPlacement {
        bytes,
        io: report::value_name(&c.io),
        locked: c.locked,
        node: opts.node,
        ram_cache_bytes: c.ram_cache_bytes,
        hot_rows: c.hot_rows,
        hot_bytes: c.hot_bytes,
        reason: c.reason.clone(),
    };
    model.set_ngram_table(table);
    Ok(placement)
}

/// Puts the hottest experts of every tiered MoE layer in its stage's free VRAM (or, on the
/// CPU, in a host copy of `--expert-cache-gib` per stage).
/// `compute`: per stage, the compute buffers its kept graphs take at most (two prompt or verify
/// steps and a decode step, measured); empty for the fixed [`COMPUTE_HEADROOM`]. `cfg.cache_reserve` is kept free on top for
/// the decode and verify graphs, ggml's pools and cuBLAS. `long`: per stage, a long prompt
/// step's compute buffer, measured, for what a lend must free (empty: lends every slot).
/// `pinned`: per stage, what of `compute` stays allocated once taken (ggml's pool copy of the
/// LM head, `vram::head_pool_bytes`), which a lend does not find free.
fn build_expert_cache(
    exec: &mut Executor,
    layout: &layout::Layout,
    cfg: &crate::residency::ResidencyConfig,
    compute: &[usize],
    long: &[usize],
    pinned: &[usize],
) -> Result<(), LoadError> {
    use crate::residency::experts::{ExpertResidency, LayerPlace};
    let hp = exec.hp().clone();
    let gpu = layout.stages[0].0.is_some();
    // Per stage's GPU: the asked size, if it fits, or all that the step buffers leave.
    let budgets = layout
        .stages
        .iter()
        .enumerate()
        .map(|(i, (d, _))| {
            d.map_or(Ok(0), |dev| {
                // Measured: the two biggest prompt steps' buffers and a margin; otherwise the
                // fixed headroom.
                let compute = compute
                    .get(i)
                    .map_or(COMPUTE_HEADROOM, |&b| b + COMPUTE_MARGIN);
                vram::cache_budget(
                    cfg.cache_bytes,
                    dev.memory().0,
                    compute + cfg.cache_reserve as usize,
                    &dev.name(),
                )
            })
        })
        .collect::<Result<Vec<u64>, String>>()
        .map_err(LoadError::Device)?;
    // Per stage's GPU: what a lend must free beside its staging buffers. A long step's buffer,
    // a margin and the reserve, less what the cache leaves free (the step graphs a lend drops
    // among it).
    let mut cfg = cfg.clone();
    // The MTP head (layer n_layer) takes long prompts in as K/V only: no staging.
    cfg.stage_below = Some(hp.n_layer);
    if cfg.lend_keep.is_empty() && !long.is_empty() {
        cfg.lend_need = layout
            .stages
            .iter()
            .enumerate()
            .map(|(i, (d, _))| {
                let free = d.map_or(0, |dev| dev.memory().0) as u64;
                let beside = free.saturating_sub(budgets[i]);
                let need = long.get(i).copied().unwrap_or(0)
                    + 2 * COMPUTE_MARGIN
                    + pinned.get(i).copied().unwrap_or(0);
                (need as u64 + cfg.cache_reserve).saturating_sub(beside)
            })
            .collect();
    }
    // GLM's parallel decode: every GPU caches its own experts of every layer, in an entry per
    // (layer, GPU) beside the owner's ([`crate::model::glm5next::par::ep_layers`]).
    let ep_cuda = exec.model().glm_par.as_ref().map(|par| {
        let (ep, cuda) = crate::model::glm5next::par::ep_layers(&par.plan, hp.n_layer + 1);
        cfg.ep = Some(ep);
        // `--glm-pair-balance`, `--glm-pair-split`: every device keeps the owners' cached
        // experts per layer.
        cfg.ep_cached = par.plan.pair_balance || par.plan.pair_split;
        // `--glm-prefetch-stats`: the next layer's router guess has a log of its own (#745).
        cfg.ep_predict = par.plan.predict_next;
        // `--glm-prefetch`: every GPU keeps a staging ring for the next layer's experts (#747).
        cfg.ep_prefetch = par.plan.prefetch;
        // The host's share computes beside one stage's graph: not with every GPU at once.
        cfg.cpu_share.share = 0.0;
        cuda
    });
    // `--pair-tp`: a paired layer's owner caches its half of the experts, a copy of the layer
    // on the owner's NVLink pair the other ([`crate::model::core::pair::ep_layers`]).
    let ep_cuda = ep_cuda.or_else(|| {
        let p = exec.model().pair_tp()?;
        let (ep, cuda) =
            crate::model::core::pair::ep_layers(&p.owner, hp.n_layer + 1, p.n_expert, |il| {
                layout.stage_of(il)
            });
        cfg.ep = Some(ep);
        // A prompt runs on the owner alone, over every expert: its staging copies the half the
        // owner does not cache too.
        cfg.ep_stage_all = true;
        // Both GPUs of a pair compute in a paired step: not the host's share beside them.
        cfg.cpu_share.share = 0.0;
        Some(cuda)
    });
    let real: Vec<usize> = cfg
        .ep
        .as_ref()
        .map_or_else(|| (0..=hp.n_layer).collect(), |e| e.real.clone());
    let cfg = &cfg;
    // CUDA index of a stage's device: stages run on CUDA0, CUDA1, ... in order.
    let budget = |cuda: Option<usize>| -> u64 {
        match cuda {
            Some(i) => budgets.get(i).copied().unwrap_or(0),
            None => cfg.cache_bytes.unwrap_or(0),
        }
    };
    let model = exec.model();
    let find = |name: &str| {
        // An entry's name: its model layer's tensor.
        let name = match name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(i, rest)| Some((real.get(i.parse::<usize>().ok()?)?, rest)))
        {
            Some((il, rest)) => format!("blk.{il}.{rest}"),
            None => name.to_string(),
        };
        let name = name.as_str();
        let t = model.tensor(name)?;
        // Per-expert scales would need remapping too: those models keep plain experts.
        let scale = name.strip_suffix(".weight").map(|b| format!("{b}.scale"));
        if scale.is_some_and(|s| model.tensor(&s).is_some()) {
            return None;
        }
        (!gpu || crate::residency::arena_node_of(t).is_some()).then_some(t)
    };
    let place = |il: usize| {
        let stage = match &ep_cuda {
            Some(cuda) => *cuda.get(il)?,
            None => layout.stage_of(il),
        };
        let buft = match layout.stages[stage].0 {
            Some(d) => d.buffer_type(),
            None => layout.host,
        };
        Some(LayerPlace {
            cuda: gpu.then_some(stage),
            buft,
        })
    };
    let mut r = ExpertResidency::new(
        real.len(),
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        find,
        place,
        budget,
        cfg,
    )
    .map_err(|e| LoadError::Device(format!("expert cache: {e}")))?;
    // The MTP head (layer n_layer) takes long prompts in as K/V only: no staging.
    r.stage_below(hp.n_layer);
    let s = r.stats();
    tracing::info!(
        layers = s.layers,
        slots = s.slots,
        gib = s.slot_bytes as f64 / (1u64 << 30) as f64,
        profile = ?cfg.profile,
        "expert cache"
    );
    #[cfg(feature = "cuda")]
    if gpu {
        use crate::model::core::expert_mv::{self, peer};
        let nodes = stage_numa_nodes(layout.stages.len());
        let paired = hp.arch.peer_decode_pairs();
        // Expert parallelism: each GPU reads its own experts' misses from its own node, no
        // peer fetch.
        let on = if ep_cuda.is_some() {
            peer::set_auto(false);
            false
        } else {
            peer_fetch_on(&nodes, paired)
        };
        // The side-stream miss copy rides along where the peer fetch is on by choice: alone it
        // measured no gain on GLM, with the peer +10% at C=4.
        expert_mv::overlap::set_auto(paired && on);
        if on && !connect_peers(&r, &nodes) && peer::forced().is_none() {
            // The automatic choice: GPUs that cannot all connect decode as before, without the
            // split pair of ops.
            peer::set_auto(false);
        }
        if cfg.stage_peer && ep_cuda.is_none() {
            stage_peers(&r, &nodes)?;
        }
    }
    exec.set_experts(Some(r));
    Ok(())
}

/// Each stage GPU's host NUMA node (`None` when unknown).
#[cfg(feature = "cuda")]
fn stage_numa_nodes(n_stages: usize) -> Vec<Option<u32>> {
    (0..n_stages)
        .map(|d| crate::residency::cuda::caps(d).and_then(|c| u32::try_from(c.host_numa_id).ok()))
        .collect()
}

/// The expert peer fetch over an expert cache on GPUs: as `--expert-peer-fetch` /
/// `--no-expert-peer-fetch` force it, else [`peer::auto`](crate::model::core::expert_mv::peer::auto)'s
/// choice for these stages (#522), or [`peer::auto_paired`]'s for an architecture that asks for
/// it (`paired`), which graphs built from now on follow. Says what it chose.
#[cfg(feature = "cuda")]
fn peer_fetch_on(nodes: &[Option<u32>], paired: bool) -> bool {
    use crate::model::core::expert_mv::peer;
    if let Some(on) = peer::forced() {
        return on;
    }
    let platform = crate::residency::Platform::of(&crate::hw::HardwareInfo::detect());
    let coherent = platform == crate::residency::Platform::CoherentNvlink;
    let verdict = if paired {
        peer::auto_paired(coherent, nodes)
    } else {
        peer::auto(coherent, nodes)
    };
    if let Err(why) = verdict {
        tracing::debug!(
            ?why,
            "expert peer fetch off (--expert-peer-fetch forces it on)"
        );
    }
    peer::set_auto(verdict.is_ok());
    verdict.is_ok()
}

/// The expert peer fetch: connects each stage GPU to its NVLink peer (the other stage GPU on its
/// host NUMA node), which fetches missed experts for it while it runs alone. Without a peer,
/// or when connecting fails (said once), a GPU decodes as without the switch. Whether every
/// stage GPU got its peer.
#[cfg(feature = "cuda")]
fn connect_peers(r: &crate::residency::experts::ExpertResidency, nodes: &[Option<u32>]) -> bool {
    use crate::model::core::expert_mv::peer;
    let mut all = true;
    for (device, p) in peer::pairs(nodes).into_iter().enumerate() {
        let Some(p) = p else {
            tracing::warn!(
                device,
                "expert peer fetch: no other stage GPU on this GPU's node"
            );
            all = false;
            continue;
        };
        let made = r
            .share_host_parts(device, p)
            .and_then(|shared| peer::connect(device, p, shared));
        match made {
            Ok(()) => tracing::info!(
                device,
                peer = p,
                "expert peer fetch (--no-expert-peer-fetch turns it off)"
            ),
            Err(e) => {
                tracing::warn!(device, peer = p, error = %e, "expert peer fetch: no peer");
                all = false;
            }
        }
    }
    all
}

/// `--prefill-peer-stage`: each stage GPU's NVLink pair (the other stage GPU on its node) reads
/// its host experts and writes into its staged ranges, so a prompt's staging copies split over
/// both GPUs' links while the pair is idle ([`crate::residency::staging`]). As Project Maya's
/// two-GPU prefill staging (mw00/project-maya, MIT). Must run before the cache is first lent.
/// Every peer must reach its GPU's memory (`cuDeviceCanAccessPeer`): a pair without peer access
/// (two PCIe GPUs on different root complexes) stops the load with a clear error rather than
/// faulting in the first prompt's staging.
#[cfg(feature = "cuda")]
fn stage_peers(
    r: &crate::residency::experts::ExpertResidency,
    nodes: &[Option<u32>],
) -> Result<(), LoadError> {
    use crate::model::core::expert_mv::peer;
    let pairs = peer::pairs(nodes);
    let missing = crate::residency::fallback::missing_peer_access(
        &pairs,
        crate::model::core::pair::link::can_access,
    );
    if !missing.is_empty() {
        let list: Vec<String> = (missing.iter())
            .map(|(p, d)| format!("CUDA{p} cannot access CUDA{d}'s memory"))
            .collect();
        return Err(LoadError::Device(format!(
            "--prefill-peer-stage needs peer access between the paired GPUs, and {} (no peer \
             access between them): drop --prefill-peer-stage",
            list.join(", ")
        )));
    }
    for (device, p) in pairs.into_iter().enumerate() {
        let Some(p) = p else {
            tracing::warn!(
                device,
                "prefill peer staging: no other stage GPU on this GPU's node"
            );
            continue;
        };
        match r.share_host_parts(device, p) {
            Ok(_) => {
                crate::residency::cuda::set_stage_peer(device, p);
                tracing::info!(device, peer = p, "prefill peer staging");
            }
            Err(e) => tracing::warn!(device, peer = p, error = %e, "prefill peer staging: no peer"),
        }
    }
    Ok(())
}

/// What [`load_report`] takes from [`LmEngine::load`]'s own decisions.
struct LoadFacts<'a> {
    mode: ExpertMode,
    native_ctx: bool,
    auto_tier: bool,
    /// KV blocks in all, and of them in device memory.
    n_blocks: usize,
    hbm_blocks: usize,
    /// Per stage: its KV host tier's NUMA node (empty without a tier).
    tier_nodes: &'a [Option<u32>],
    bytes_per_token: usize,
    ring: Option<KvRing>,
    /// Per stage's layers: its states' and rings' bytes, as the KV fit counted them.
    state_bytes: &'a dyn Fn(&std::ops::Range<usize>) -> usize,
    /// Per stage: the measured step compute buffers (empty when not measured).
    compute: &'a [usize],
    ngram: Option<report::NgramPlacement>,
    engram: Option<report::NgramPlacement>,
    /// Saved prefix-cache states in pinned host memory, all stages.
    saved_host_bytes: u64,
}

/// Where a load put everything: the layout's stages and weight bytes, the KV pools and host
/// tier as allocated, the expert cache's slots, the n-gram table's resolved tier, and each
/// device's free memory once loaded.
fn load_report(
    layout: &layout::Layout,
    exec: &mut Executor,
    cfg: &LmConfig,
    f: LoadFacts<'_>,
) -> report::LoadReport {
    use report::{HostUse, add_host};
    let hp = exec.hp().clone();
    let weights = layout.stage_bytes_with_mtp(exec.model().weights(), exec.model().mtp_weights());
    let kv = exec.kv_stage_bytes();
    let host_blocks = f.n_blocks - f.hbm_blocks;
    let gpu = layout.stages[0].0.is_some();
    let mut stages: Vec<report::StageMemory> = layout
        .stages
        .iter()
        .enumerate()
        .map(|(si, (d, layers))| {
            let (block, staging) = kv.get(si).copied().unwrap_or_default();
            let mut s = report::StageMemory {
                device: d.map_or_else(|| "CPU".into(), |d| d.name()),
                description: d.map(|d| d.description()).filter(|s| !s.is_empty()),
                layers: layers.clone(),
                weights: weights.get(si).copied().unwrap_or(0),
                kv_tokens: f.hbm_blocks * KV_BLOCK,
                kv_bytes: (f.hbm_blocks * block) as u64,
                kv_staging_bytes: staging as u64,
                states_bytes: (f.state_bytes)(layers) as u64,
                compute_bytes: f.compute.get(si).map(|&b| b as u64),
                ..Default::default()
            };
            if let Some(d) = d {
                s.kv_reserve_bytes =
                    Some(kv_headroom(d.kind(), cfg.max_batch_tokens, cfg.uma_free) as u64);
                if f.mode == ExpertMode::Cache && cfg.residency.cache_bytes.is_none() {
                    // As `build_expert_cache`'s budget: the measured buffers and a margin, or
                    // the fixed headroom, then the reserve.
                    let compute = f
                        .compute
                        .get(si)
                        .map_or(COMPUTE_HEADROOM, |&b| b + COMPUTE_MARGIN);
                    s.expert_reserve_bytes = Some(compute as u64 + cfg.residency.cache_reserve);
                }
                let (free, total) = d.memory();
                s.free_bytes = Some(free as u64);
                s.total_bytes = Some(total as u64);
            }
            s.tp_devices = layout
                .tp_devices
                .iter()
                .map(|d| {
                    let (free, total) = d.memory();
                    (d.name(), free as u64, total as u64)
                })
                .collect();
            s
        })
        .collect();
    // The expert cache's slots, by the stage of their layer.
    let mut experts = None;
    if hp.n_expert > 0 {
        let fetch = report::value_name(&cfg.residency.fetch);
        let mut summary = report::ExpertSummary {
            n_expert: hp.n_expert as usize,
            n_used: hp.n_expert_used.max(0) as usize,
            fetch,
            ..Default::default()
        };
        if let Some(r) = exec.experts_mut() {
            let layers = r.layer_slots();
            for &(il, slots, bytes) in &layers {
                let s = &mut stages[layout.stage_of(il)];
                s.expert_layers += 1;
                s.expert_slots += slots;
                s.expert_slot_bytes += bytes;
            }
            summary.cached_layers = layers.len();
            summary.slots_min = layers.iter().map(|l| l.1).min().unwrap_or(0);
            summary.slots_max = layers.iter().map(|l| l.1).max().unwrap_or(0);
            // Staging VRAM by CUDA index; the stages run on CUDA0, CUDA1, ... in order.
            for (cuda, bytes) in r.stage_bytes() {
                if let Some(s) = stages.get_mut(cuda) {
                    s.expert_staging_bytes = Some(bytes);
                }
            }
        }
        experts = Some(summary);
    }
    // Host memory: the weights the layout keeps off the devices, by where they went.
    let mut host = Vec::new();
    let table = f.ngram.is_some().then_some("per_layer_token_embd.weight");
    let mut place_host = |name: &str, bytes: u64, stage: Option<usize>, il: Option<usize>| {
        if stage.is_some() || Some(name) == table {
            return;
        }
        if name.contains("_exps.") {
            let arena = layout.arenas.as_ref().and_then(|a| {
                let si = il.map_or(layout.stages.len() - 1, |il| layout.stage_of(il));
                a.get(si).map(|&(_, node)| node)
            });
            match arena {
                Some(node) => add_host(&mut host, known_node(node), HostUse::ExpertArena, bytes),
                None => add_host(&mut host, None, HostUse::MappedExperts, bytes),
            }
        } else if layout.shared_host.is_some()
            && matches!(name, "token_embd.weight" | "rope_freqs.weight")
        {
            add_host(&mut host, None, HostUse::SharedWeights, bytes);
        } else {
            add_host(&mut host, None, HostUse::HostWeights, bytes);
        }
    };
    let layer = |name: &str| -> Option<usize> {
        name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
    };
    if gpu {
        for (name, bytes) in exec.model().weights().sizes() {
            place_host(name, bytes, layout.device_stage(name), layer(name));
        }
        for (name, bytes) in exec
            .model()
            .mtp_weights()
            .into_iter()
            .flat_map(|w| w.sizes())
        {
            // The MTP head's experts belong to its layer, on the last stage.
            place_host(name, bytes, layout.mtp_device_stage(name), None);
        }
    }
    // The KV host tier, per node.
    if host_blocks > 0 {
        for (si, node) in f.tier_nodes.iter().enumerate() {
            let bytes = (host_blocks * kv.get(si).map_or(0, |k| k.0)) as u64;
            let node = node.and_then(known_node);
            add_host(&mut host, node, HostUse::KvHostTier, bytes);
            if let Some(h) = host
                .iter_mut()
                .find(|h| h.node == node && h.kind == HostUse::KvHostTier)
            {
                h.tokens = Some(host_blocks * KV_BLOCK);
            }
        }
    }
    add_host(&mut host, None, HostUse::SavedStates, f.saved_host_bytes);
    host.sort_by_key(|h| (h.node.is_none(), h.node, h.kind));
    let types = exec.kv_types();
    report::LoadReport {
        expert_mode: (hp.n_expert > 0).then(|| report::value_name(&f.mode)),
        tp: if layout.tp_devices.is_empty() {
            0
        } else {
            layout.tp_devices.len()
        },
        stages,
        host,
        kv: report::KvSummary {
            ctx_size: cfg.ctx_size,
            native: f.native_ctx,
            device_tokens: f.hbm_blocks * KV_BLOCK,
            host_tokens: host_blocks * KV_BLOCK,
            staging_tokens: exec.kv_stage_blocks() * KV_BLOCK,
            host_auto: f.auto_tier,
            types: format!(
                "{}/{}",
                report::value_name(&types.k),
                report::value_name(&types.v)
            ),
            bytes_per_token: f.bytes_per_token as u64,
            rings: f.ring.map(|r| (r.sets, r.positions())),
        },
        experts,
        ngram: f.ngram,
        engram: f.engram,
    }
}

/// What [`log_memory_plan`] takes from [`LmEngine::load`]'s own decisions.
struct MemoryPlanFacts<'a> {
    ctx_size: usize,
    max_seqs: usize,
    /// KV blocks in all.
    n_blocks: usize,
    kv_lend: Option<&'a kv_vmm::KvLend>,
    /// Per stage: its pools' tail lent to the expert cache at load (empty without a lend).
    kv_lent_bytes: &'a [usize],
    /// Per stage: the VRAM kept free of the expert cache for step compute buffers, and the
    /// prompt steps' room the cache took (empty when the load measured none).
    planned_compute: &'a [usize],
    compute_spill: &'a [usize],
}

/// Logs the memory plan (#1196, [`plan`]) once loaded: per GPU and in host memory, from the
/// load report, the load's own decisions and the driver. Printing only.
fn log_memory_plan(
    layout: &layout::Layout,
    exec: &mut Executor,
    report: &report::LoadReport,
    f: MemoryPlanFacts<'_>,
) {
    let experts = exec
        .experts_mut()
        .map(|r| r.slot_bytes_by_cuda())
        .unwrap_or_default();
    let expert_shrunk = exec
        .experts_mut()
        .map(|r| r.shrunk().to_vec())
        .unwrap_or_default();
    let gpu = |d: crate::ggml::Device| {
        let (free, total) = d.memory();
        plan::Gpu {
            device: d.name(),
            name: d.description(),
            cuda: cuda_index(d),
            total: total as u64,
            free: free as u64,
        }
    };
    let keep_tokens = f.kv_lend.map(|l| l.keep * KV_BLOCK);
    let lent_tokens = exec.kv_slot_lend() * KV_BLOCK;
    let n_stages = exec.stage_layers().len();
    let mut gpus = Vec::new();
    for (si, ((d, _), s)) in layout.stages.iter().zip(&report.stages).enumerate() {
        let Some(d) = d.filter(|d| d.kind() != crate::ggml::DeviceKind::Cpu) else {
            continue;
        };
        // Tensor parallelism: one stage over the meta device's GPUs.
        let devs: Vec<plan::Gpu> = if layout.tp_devices.is_empty() {
            vec![gpu(d)]
        } else {
            layout.tp_devices.iter().map(|&d| gpu(d)).collect()
        };
        let at = |v: &[usize]| v.get(si).map_or(0, |&b| b as u64);
        let facts = plan::StageFacts {
            kv_lent: at(f.kv_lent_bytes),
            // The slots' shared VRAM (#1159) with the tokens the load placed there.
            kv_vram_tokens: keep_tokens.unwrap_or(s.kv_tokens + lent_tokens),
            compute_allocated: if si < n_stages {
                exec.compute_bytes(si) as u64
            } else {
                0
            },
            compute_planned: at(f.planned_compute),
        };
        gpus.extend(plan::stage_gpus(s, &facts, &devs, &experts));
    }
    let mut host = plan::host_from_report(report);
    host.kv_spill_max = f.kv_lend.map_or(0, |l| l.host_budget as u64);
    host.compute_spill_max = f.compute_spill.iter().sum::<usize>() as u64;
    host.available = crate::residency::lookup_table::MemInfo::read().available;
    let tail_tokens = f
        .kv_lend
        .map_or(0, |l| f.n_blocks.saturating_sub(l.keep) * KV_BLOCK);
    let lent = f.kv_lent_bytes.iter().sum::<usize>() as u64;
    plan::log(&plan::MemoryPlan {
        context: plan::ContextPlan {
            ctx_size: f.ctx_size,
            max_seqs: f.max_seqs,
            kv_pool_tokens: f.n_blocks * KV_BLOCK,
            kv_host_tier_tokens: report.kv.host_tokens,
            kv_spill_tokens: plan::spill_tokens(tail_tokens, lent, host.kv_spill_max),
        },
        gpus,
        host,
        expert_shrunk,
    });
}

/// The model's id: `--model-name`, else the checkpoint's `general.name`, else its file stem.
fn model_id(cfg: &LmConfig, gguf: &GgufFile) -> String {
    cfg.model_name.clone().unwrap_or_else(|| {
        gguf.get_str("general.name").unwrap_or_else(|| {
            cfg.model_path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into())
        })
    })
}

impl LmEngine {
    /// Loads the model (blocking: run it off the async runtime) and starts the worker thread.
    pub fn load(mut cfg: LmConfig) -> Result<Self, LoadError> {
        // `--tp-per-gpu`: rank 1 loads on a thread of its own, this call goes on as rank 0.
        if cfg.tp_per_gpu {
            return tp_pair::load(cfg);
        }
        crate::model::core::expert_mv::set_enabled(
            cfg.expert_mv && matches!(cfg.device, DeviceChoice::Cuda),
        );
        crate::model::cluster_select::set_cuda(matches!(cfg.device, DeviceChoice::Cuda));
        // Bit-identical either way; faster on the GB10 (`--glm-linear-plain`: off).
        crate::model::glm5next::ops::set_gb10_linear(
            matches!(cfg.device, DeviceChoice::Cuda) && integrated_gpu(),
        );
        // qwen4exp's MoE tails folded into the next norm (#836): the same bits, prompt steps
        // about 1% faster on the GB10 (`--hc-moe-fold` / `--no-hc-moe-fold` choose).
        crate::model::qwen4::set_moe_fold(
            matches!(cfg.device, DeviceChoice::Cuda) && integrated_gpu(),
        );
        cfg.switches.apply();
        node::check_config(&cfg)?;
        tp_node::check_config(&cfg)?;
        slo::check(cfg.slo, cfg.pipeline_groups, cfg.node.is_some()).map_err(LoadError::Device)?;
        crate::model::core::paged::set_m884_prefill(cfg.paged_prefill_m884);
        crate::model::qwen4::qsa_attn::set_ggml_paths(cfg.qsa_scores_ggml, cfg.qsa_topk_ggml);
        crate::model::qwen4::set_generic_post_norm(cfg.hc_post_norm_generic);
        crate::model::glm5next::set_route_bias(cfg.glm_route_bias);
        check_kv_memory(
            cfg.kv_memory,
            std::env::var(UNIFIED_MEMORY_ENV).ok().as_deref(),
        )
        .map_err(LoadError::Device)?;
        if cfg.kv_memory == KvMemory::Managed {
            tracing::warn!(
                "--kv-memory managed: every ggml CUDA allocation is managed memory (weights too); \
                 keep `legion serve --numa auto` (the default) or name the GPUs' node so \
                 paged-out memory stays local"
            );
        }
        let hf = crate::hf::OpenOptions { fp8: cfg.hf_fp8 };
        let gguf = GgufFile::open_with(&cfg.model_path, &hf)?;
        // Tensor parallelism over machines (#619): this rank's share of every layer, and the
        // connections to the other ranks.
        let (gguf, mut tp_start) = tp_node::start(&cfg, gguf)?;
        // No --ctx-size: the model's native context, in full or not at all.
        let native_ctx = cfg.ctx_size == 0;
        if native_ctx {
            let native = crate::model::core::Hparams::from_gguf(&gguf)?.n_ctx_train;
            cfg.ctx_size = usize::try_from(native)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    LoadError::Device(format!(
                        "the model gives no native context ({native}): pass --ctx-size"
                    ))
                })?;
            tracing::info!(
                ctx_size = cfg.ctx_size,
                "context: the model's native one (--ctx-size lowers it)"
            );
        }
        let sampling = crate::engine::sampling_defaults::ModelSampling::from_gguf(&gguf);
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let chat_format = ChatFormat::detect(&tokenizer);
        let template = match gguf.get_str("tokenizer.chat_template") {
            Some(src) => {
                let text = |id: Option<u32>| {
                    id.and_then(|i| tokenizer.token_text(i))
                        .unwrap_or_default()
                        .to_string()
                };
                Some(ChatTemplate::new(
                    &src,
                    &text(tokenizer.bos_id()),
                    &text(tokenizer.eos_id()),
                )?)
            }
            None => None,
        };
        let id = model_id(&cfg, &gguf);

        let layout = layout::Layout::plan(&cfg, &gguf)?;
        // A GPU this binary has no code of its architecture for fails here, before any weight
        // is read, not mid-serve (#1201).
        #[cfg(feature = "cuda")]
        check_gpu_code(&layout)?;
        let mode = layout.mode;
        let mut backends = layout.backends()?;
        for b in &mut backends {
            b.set_threads(cfg.threads);
            b.set_use_ref(cfg.use_ref);
        }
        let device = layout.stages[0].0;
        let buft = backends[0].buffer_type();
        let t = Instant::now();
        let place = |info: &crate::gguf::TensorInfo| layout.place(info);
        // An arena is filled by its own node's CPUs (the reader pins its workers there).
        let topo = crate::hw::numa::Topology::detect().ok();
        let mut opts = crate::load_io::Options::configured();
        if layout.arenas.is_some() {
            opts.topology = topo.as_ref();
        }
        let head_copy = [self::layout::HEAD_COPY];
        if layout.head_copy {
            opts.aliases = &head_copy;
        }
        // Routed experts that do not fit, read from the checkpoint as routed (#560).
        // Over two machines (#562) every weight of the share is in memory: no SSD tier.
        let ssd_layers = if layout.node.is_some() {
            None
        } else {
            ssd_tier::plan(&cfg, &gguf, &layout)?
        };
        let mut skip = ssd_layers
            .as_deref()
            .map(crate::residency::ssd_experts::skipped_tensors)
            .unwrap_or_default();
        // The other machine's layers and head.
        let hp_file = crate::model::core::Hparams::from_gguf(&gguf)?;
        skip.extend(layout.node_skipped(&gguf, hp_file.n_layer));
        // GLM-5.3-Flash's MTP layer when nothing drafts with it (#632).
        let drafting_mtp = drafts_with_mtp_head(&cfg.spec);
        skip.extend(unused_mtp_tensors(
            gguf.tensors().iter().map(|t| t.name.as_str()),
            hp_file.arch.name(),
            hp_file.n_layer,
            drafting_mtp,
        ));
        // DeepSeek-V4.1-Flash's Engram tables: read through a lookup table, never mapped (or
        // read ahead) as weights (#591).
        let engram = crate::model::deepseek41::engram::lookup_table(&gguf, &hp_file);
        if let Some((_, names)) = &engram {
            skip.extend(names.iter().cloned());
        }
        // GLM's parallel steps: the GPUs' shards and mirrors load with the weights, in place
        // of the tensors they split.
        let glm_plan = if layout.glm_par {
            if ssd_layers.is_some() {
                return Err(LoadError::Device(
                    "--glm-parallel-decode needs every routed expert in memory (no SSD tier)"
                        .into(),
                ));
            }
            if let Some(why) = glm_par_spec_refusal(&cfg.spec) {
                return Err(LoadError::Device(why.into()));
            }
            if cfg.kv_host_tokens.is_some_and(|t| t > 0) {
                return Err(LoadError::Device(
                    "--glm-parallel-decode has no KV host tier yet".into(),
                ));
            }
            glm_par_steps(&mut cfg);
            Some(glm_par_plan(&gguf, &layout, &cfg)?)
        } else {
            None
        };
        let (glm_extra, glm_recipes) = match glm_plan.as_ref() {
            Some((_, t)) => {
                skip.extend(t.skip.iter().cloned());
                (t.extra.clone(), Some(&t.recipes))
            }
            None => (Vec::new(), None),
        };
        opts.skip = &skip;
        opts.extra = &glm_extra;
        opts.extra_recipes = glm_recipes;
        // The page cache an earlier load left behind out of the way of the weights.
        let drop_cache =
            release_load_cache(cfg.keep_load_cache, device.map(crate::ggml::Device::kind));
        if drop_cache {
            gguf.release_page_cache();
        }
        let mut model = Model::load_converted(&gguf, place, &opts, cfg.fp16)?;
        if let Some(s) = &tp_start {
            s.set_sums(&mut model);
        }
        if let Some((rank, ranks, place)) = tp_node::moe_ep(&cfg) {
            model.set_tp_moe_ep(rank, ranks, place)?;
        }
        if let Some(layers) = &ssd_layers {
            let tier = ssd_tier::build(&cfg, &gguf, &layout, &model.hp, layers)?;
            model.set_ssd_experts(tier);
        }
        if cfg.pair_tp {
            pair_tp::set_up(&cfg, &mut model, &layout)?;
        }
        if layout.head_copy
            && let Some(head) = model.weights().get(self::layout::HEAD_COPY.0)
        {
            tracing::info!(
                stage = layout.stages.len() - 1,
                mib = head.nbytes() as f64 / (1u64 << 20) as f64,
                "tied LM head: a copy of the token embeddings on the last GPU"
            );
        }
        if cfg.gemma4_compact_global_kv {
            gemma4_compact_global_kv(&mut model, cfg.cache_types);
        }
        let long_prompts = LongPrompts {
            short_min_tokens: Some(cfg.long_prompts.short_min_tokens.unwrap_or_else(|| {
                short_prompt_default(
                    crate::model::glm5next::params(&model.hp).is_some(),
                    layout.stages.len(),
                )
            })),
            ..cfg.long_prompts
        };
        if let Some(prune) = &cfg.glm_prune {
            if crate::model::glm5next::params(&model.hp).is_none() {
                return Err(LoadError::Device(
                    "--glm-prune: not a GLM-5.3-Flash model".into(),
                ));
            }
            let n = crate::model::glm5next::soft_prune(&model, prune)?;
            tracing::info!(
                pairs = n,
                penalty = prune.penalty,
                "--glm-prune: soft-pruned experts"
            );
        }
        // A cache split by heads is gathered cell by cell (see `Model::set_cell_gather`).
        model.set_cell_gather(cfg.tp > 1 || cfg.cell_gather);
        let integrated = device.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::IntegratedGpu);
        let k_given = cfg.spec.k > 0;
        if cfg.spec.k == 0 {
            cfg.spec.k = default_spec_k(
                model.hp.arch.name(),
                model.hp.arch.default_spec_k(),
                integrated,
                cfg.spec.kind == crate::spec::SpecKind::Mtp,
            );
        }
        if let Some((plan, _)) = &glm_plan {
            let k = drafting_mtp.then(|| cfg.spec.verify_k());
            let max_out = glm_par_out_rows(cfg.max_seqs, cfg.prefill_slots, k);
            set_glm_par(
                &mut model,
                &layout,
                plan.clone(),
                max_out,
                cfg.glm_exchange_peer,
                crate::model::glm5next::par::step_tokens(cfg.glm_prompt_step),
            )?;
        }
        let ngram = if device.is_some() {
            open_ngram_table(&mut model, &gguf, &layout, topo.as_ref(), &cfg.ngram)?
        } else {
            None
        };
        // On the CPU too: the weights left the tables out.
        let engram = match engram {
            Some((parts, _)) => Some(open_lookup_table(
                &mut model,
                parts,
                &layout,
                topo.as_ref(),
                &cfg.ngram,
                "Engram tables",
            )?),
            None => None,
        };
        model.keep_a8(cfg.nvfp4_act == crate::model::core::Nvfp4Act::A8);
        model.set_fp8_act(cfg.fp8_act);
        model.set_nvfp4_mm(cfg.nvfp4_mm);
        model.set_nvfp4_a4(cfg.nvfp4_a4);
        #[cfg(feature = "cuda")]
        warn_nvfp4_a4(&layout, cfg.nvfp4_a4);
        #[cfg(feature = "cuda")]
        let caps = device
            .and_then(cuda_index)
            .and_then(crate::residency::cuda::caps)
            .map(|c| (c.cc_major, c.cc_minor, c.sms));
        #[cfg(not(feature = "cuda"))]
        let caps: Option<(i32, i32, i32)> = None;
        let cc = caps.map(|c| (c.0, c.1));
        let cc_major = cc.map(|c| c.0);
        // V100 (#574): the GB10 prompt kernels too (Maya-S, 8K-token prompt 565 -> 804 tok/s), but
        // the plain Q6_K decode GEMV (the staged one ~11% slower there). Ampere (8.0 to 8.7,
        // #1212) likewise: without them its prompt products run linear_mm, no tensor cores; the
        // decode GEMV stays the plain one. The same bits.
        let volta = cc == Some((7, 0));
        let ampere = cc.is_some_and(crate::hw::per_device::ampere);
        if (volta || ampere) && cfg.switches.glm_gb10_linear.is_none() {
            crate::model::glm5next::ops::set_gb10_linear(true);
        }
        crate::model::glm5next::ops::set_gb10_gemv(!(volta || ampere));
        // #594: long prompt steps' Q6_K products through F16 weights and linear_tc (sm_80+, the
        // same bits)
        crate::model::glm5next::ops::set_q6k_f16(cc_major.is_some_and(|m| m >= 8));
        crate::model::core::gdn_scan::set_gpu(cc_major);
        crate::model::core::gdn_mma::set_gpu(cc_major);
        crate::model::core::gdn::set_gpu(cc_major);
        crate::model::qwen4::set_xn16_gpu(cc);
        crate::model::core::mmq_preq::set_gpu(cc_major);
        crate::model::core::mmq_ids::set_gpu(cc);
        crate::model::core::nvfp4_once::set_gpu(cc);
        crate::model::core::nvfp4_mvq::set_gpu(cc);
        crate::model::qwen4::set_up_mix_gpu(cc_major, caps.map_or(0, |c| c.2));
        crate::model::q8_mmq::set_gpu(cc_major, caps.map_or(0, |c| c.2));
        // #1214: a GTX 16xx's prompt matmuls on dp4a (no tensor cores to run MMQ's MMA on).
        let tu11x = device
            .and_then(cuda_index)
            .is_some_and(crate::hw::per_device::turing_without_tensor_cores);
        crate::model::tu_mmq::set_gpu(if tu11x { caps.map_or(0, |c| c.2) } else { 0 });
        let mma16 = device.is_some() && crate::model::moe_ffn::mma16();
        let moe_kernel = resolve_moe_kernel(cfg.moe_kernel, mode, integrated, cc_major, mma16);
        if moe_kernel != cfg.moe_kernel {
            tracing::info!(?moe_kernel, ?cc_major, mma16, "routed experts of big steps");
        }
        model.set_moe_kernel(moe_kernel);
        let mtp = drafting_mtp;
        // Rank 0 of a model over two nodes: the MTP head runs on rank 1 (#560), which loads it,
        // keeps its cache and drafts there; this rank drafts through the link.
        let head_on_peer = layout.node.as_ref().is_some_and(|s| s.head_on_peer());
        if cfg.mtp_window.is_some_and(|n| n > 0) && !mtp {
            tracing::warn!(
                mtp_window = cfg.mtp_window,
                "--mtp-window applies to an MTP head (--spec mtp) only: ignored"
            );
        }
        // Not given: the default, whose notes (a window past the context, say) are not warnings.
        let mtp_window_given = cfg.mtp_window.is_some();
        let mtp_window = cfg
            .mtp_window
            .unwrap_or_else(|| default_mtp_window(model.hp.arch.name(), mtp, integrated));
        // Before the executor reads them.
        if let Some((m, v)) = default_kept_graphs(model.hp.arch.name(), mtp, integrated) {
            if cfg.switches.mtp_kept_graphs.is_none() {
                crate::model::core::set_mtp_kept_graphs(m);
            }
            if cfg.switches.verify_kept_graphs.is_none() {
                crate::model::core::set_verify_kept_graphs(v);
            }
        }
        let dflash =
            cfg.spec.kind == crate::spec::SpecKind::Dflash && cfg.spec.model_drafter.is_none();
        // The rows of the target's head a DFlash drafter scores its drafts with (0: all).
        let mut dflash_vocab = 0;
        match draft_vocab_unused(
            cfg.draft_vocab,
            mtp,
            dflash,
            model.hp.arch.name(),
            model.hp.n_vocab,
        ) {
            Some(why) => tracing::warn!(draft_vocab = cfg.draft_vocab, "{why}"),
            None if dflash => dflash_vocab = cfg.draft_vocab,
            None => model.set_draft_vocab(cfg.draft_vocab),
        }
        if mtp && !head_on_peer && !model.has_mtp() {
            // The head's GGUF, else a safetensors checkpoint's own head (`mtp.*`), as shipped.
            let head = match cfg.spec.draft.as_ref() {
                Some(path) => GgufFile::open(path)?,
                None if cfg.model_path.is_dir() => {
                    let hf = crate::hf::OpenOptions { fp8: cfg.hf_fp8 };
                    crate::hf::open_mtp(&cfg.model_path, &hf)?
                }
                None => {
                    return Err(LoadError::Device(
                        "the model has no MTP head: pass its GGUF with --spec-draft".into(),
                    ));
                }
            };
            // On the last stage, where the head runs (a Gemma 4 assistant's blocks are
            // numbered from 0, which `place` puts on the first stage).
            let head = tp_node::mtp_share(&cfg, head, &model.hp)?;
            model.load_mtp(&head, |info: &crate::gguf::TensorInfo| {
                layout.place_mtp(info)
            })?;
            if drop_cache {
                head.release_page_cache();
            }
        }
        // GPUs that cannot read pageable memory: their experts in host memory registered.
        if device.is_some() {
            register_host_experts(&mut model, &layout)?;
        }
        // The MoE routers' BF16 copies (#836) before the caches take the memory left.
        model.prepare_router_bf16();
        // What the load read out of the way of the KV cache and the step buffers.
        if drop_cache {
            gguf.release_page_cache();
        }
        if device.is_some() {
            // Each stage's own layers only, converted weights and the MTP head included (Strata-AC922
            // fd96643 logs the same split).
            let gib: Vec<f64> = layout
                .stage_bytes_with_mtp(model.weights(), model.mtp_weights())
                .iter()
                .map(|&b| (b as f64 / (1u64 << 30) as f64 * 100.0).round() / 100.0)
                .collect();
            let layers: Vec<_> = layout.stages.iter().map(|(_, r)| r.clone()).collect();
            tracing::info!(?layers, weights_gib = ?gib, weights_fp16 = ?cfg.fp16.mode, "weights per stage");
        }
        let hp = model.hp.clone();
        let max_seqs = cfg.max_seqs.max(1);
        let prefill_slots = cfg.prefill_slots.max(1);
        let saved_slots = cfg.prefix_cache_states;
        // With the expert cache on discrete GPUs the saved states live in pinned host memory:
        // their VRAM goes to expert slots (#441).
        let saved_host = saved_states_host(
            mode == ExpertMode::Cache,
            layout
                .stages
                .iter()
                .all(|(d, _)| d.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::Gpu)),
            saved_slots,
            hp.has_slot_state(),
            cfg.tp,
            cfg.switches.saved_states_vram,
        );
        let target = TargetInfo {
            n_layer: hp.n_layer,
            n_embd: hp.n_embd as usize,
            n_vocab: hp.n_vocab as usize,
            model: cfg.model_path.clone(),
            draft: cfg.spec.draft.clone(),
            gpu: device.is_some(),
            threads: cfg.threads,
            ctx_size: cfg.ctx_size,
            max_seqs: max_seqs + prefill_slots,
            use_ref: cfg.use_ref,
        };
        let drafting = match tp_node::rank0_drafting(&cfg)? {
            Some(d) => d,
            None => Drafting::new(&cfg.spec, &target, dflash_vocab).map_err(LoadError::Device)?,
        };
        // A block drafter drafts its whole block in one pass: up to all of it without `--spec-k`
        // (the depth policy picks less where verify rows cost).
        if let (false, Some(n)) = (k_given, drafting.block_drafts()) {
            cfg.spec.k = n;
        }
        let spec_k = if drafting.is_on() {
            cfg.spec.verify_k()
        } else {
            0
        };
        if matches!(drafting, Drafting::Model(_)) && !hp.arch.external_drafters() {
            return Err(LoadError::Device(format!(
                "external model drafters are not supported for {} models",
                hp.arch.name()
            )));
        }
        // Sliding-window layers keep a ring of their window per sequence, where attention
        // reads the cache in place (paged); a sequence's step is at most a prompt chunk.
        let window = hp.arch.swa_window();
        // A quantized cache: the paged kernels dequantize q8_0 and q4_0 on sm_80+, the sparse
        // attention kernels on every GPU (K and V both quantized, #263); the gather path every
        // type.
        let types = cfg.cache_types;
        let sparse = hp.arch.index_dim() > 0;
        let gpu_reads = !device.is_some()
            || if sparse {
                hp.arch.sparse_reads(&hp, types)
            } else {
                crate::model::core::paged::reads_types(types)
            };
        if !types.is_f16() {
            if sparse && cfg.attention != crate::model::core::Attention::Gather && !gpu_reads {
                return Err(LoadError::Device(format!(
                    "{} models' sparse attention reads --cache-type-k / --cache-type-v F16, or \
                     q8_0 / q4_0 for both (not {:?} / {:?})",
                    hp.arch.name(),
                    types.k,
                    types.v
                )));
            }
            if cfg.attention == crate::model::core::Attention::Paged && !gpu_reads {
                return Err(LoadError::Device(format!(
                    "--attention paged does not read --cache-type-k {:?} / --cache-type-v {:?} \
                     on this device (F16; q8_0 and q4_0 on sm_80+)",
                    types.k, types.v
                )));
            }
            if cfg.tp > 1 {
                tp_cache_rows_fit(&hp, types, cfg.tp).map_err(LoadError::Device)?;
            }
            crate::model::core::turboquant::check(&hp, types, cfg.tp, cfg.kv_slots == KvSlots::On)
                .map_err(LoadError::Device)?;
        }
        let paged = match cfg.attention {
            crate::model::core::Attention::Gather => false,
            crate::model::core::Attention::Paged => true,
            crate::model::core::Attention::Auto => {
                cfg!(feature = "cuda") && device.is_some() && gpu_reads && model.paged_kernels_fit()
            }
        };
        // Tensor parallelism gathers cells, through the rings too.
        if cfg.tp > 1 {
            cfg.ring_gather = true;
        }
        // A quantized cache (gather attention) is for long contexts: rings there too.
        let mut ring_step = ring_step(
            &cfg,
            device.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::IntegratedGpu),
        );
        let rings_on = use_rings(&hp, &cfg, paged);
        let ring_for = |step: usize| {
            rings_on.then(|| KvRing::for_window(window, step.max(spec_k + 1), max_seqs))
        };
        let mut ring = ring_for(ring_step);
        // Per stage: what its states take, and how many KV blocks fit next to them.
        let ctx_blocks = cfg.ctx_size.div_ceil(KV_BLOCK);
        let shared = layout
            .stages
            .iter()
            .any(|(d, _)| d.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::IntegratedGpu));
        // Per stage: the KV blocks that fit next to the weights, the states and the reserve.
        let cache_split = mode == ExpertMode::Cache && cfg.residency.cache_bytes.is_none();
        let share = match (mode, cfg.residency.cache_bytes) {
            (ExpertMode::Cache, Some(b)) => vram::CacheShare::Bytes(b),
            _ if cache_split && cfg.kv_cache_tokens.is_none() => vram::CacheShare::Half,
            _ => vram::CacheShare::None,
        };
        // The MTP head's own KV pool, on the last stage, has as many blocks as the stages'.
        let mtp_block = match drafting {
            Drafting::Mtp(_) if !hp.arch.mtp_shares_kv() && !head_on_peer => {
                let head =
                    crate::model::core::paged::mtp_head_types(types, device.is_some(), sparse);
                KvPool::layer_block_bytes(&hp, hp.n_layer, head)
            }
            _ => 0,
        };
        // GLM's parallel steps: every GPU keeps every layer's latent cache and its share of
        // every KDA layer's heads' states ([`Executor::new_glm_par`]).
        let par_hp = layout
            .glm_par
            .then(|| crate::model::glm5next::par::shard_hparams(&hp, layout.stages.len()));
        let pool_hp = par_hp.as_ref().unwrap_or(&hp);
        let pool_layers = |layers: &std::ops::Range<usize>| {
            if par_hp.is_some() {
                0..hp.n_layer
            } else {
                layers.clone()
            }
        };
        // Per stage: what its rings take.
        let ring_bytes_with = |layers: &std::ops::Range<usize>, ring: Option<KvRing>| {
            ring.map_or(0, |r| {
                r.sets
                    * r.positions()
                    * KvPool::sliding_token_bytes(pool_hp, pool_layers(layers), types)
            })
        };
        // Per stage: what its recurrent and speculative states and its rings take.
        let vram_saved = if saved_host { 0 } else { saved_slots };
        let state_bytes_with = |layers: &std::ops::Range<usize>, ring: Option<KvRing>| {
            let layers = &pool_layers(layers);
            let hp = pool_hp;
            let mut state_bytes = (max_seqs + prefill_slots + vram_saved)
                * StatePool::slot_bytes_layers(hp, layers.clone());
            if spec_k > 0 {
                state_bytes += (max_seqs + prefill_slots)
                    * SpecPool::slot_bytes_layers(hp, spec_k + 1, layers.clone());
            }
            state_bytes + ring_bytes_with(layers, ring)
        };
        let fit_with = |ring: Option<KvRing>, log: bool| {
            layout
                .stages
                .iter()
                .enumerate()
                .filter_map(|(si, (d, layers))| {
                    let dev = (*d)?;
                    let state_bytes = state_bytes_with(layers, ring);
                    let reserve = kv_headroom(dev.kind(), cfg.max_batch_tokens, cfg.uma_free);
                    let free = layout.free_for_split(si, 0);
                    let room = layout
                        .free_for_split(si, reserve)
                        .saturating_sub(state_bytes);
                    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
                    if log {
                        tracing::info!(
                            device = %dev.name(),
                            free_gib = gib(free),
                            states_gib = gib(state_bytes),
                            reserve_gib = gib(reserve),
                            "KV pool sizing"
                        );
                    }
                    let mtp = if si + 1 == layout.stages.len() {
                        mtp_block
                    } else {
                        0
                    };
                    let per = KvPool::block_bytes_paged(
                        pool_hp,
                        pool_layers(layers),
                        ring.is_some(),
                        types,
                    );
                    Some(vram::kv_fit(room, per + mtp, share))
                })
                .fold(None, |acc: Option<(usize, usize)>, (f, full)| {
                    Some(acc.map_or((f, full), |(a, b)| (a.min(f), b.min(full))))
                })
        };
        // On a GPU sharing the host's memory the rings hold whole batches (`ring_step`): with
        // many sequences of a wide sliding-window model they can leave no room for the
        // context (Gemma 4 31B, 16 sequences at 262144 on a GB10: 63 GiB of rings, #1132).
        // Then shorter ring steps (a sequence's prompt steps get as short), halved down to
        // [`PREFILL_CHUNK`], until the context fits; a load that fits as it is keeps its step.
        if shared
            && cfg.ring_step.is_none()
            && cfg.kv_cache_tokens.is_none()
            && cfg.tp <= 1
            && let Some(r) = ring
        {
            let fits =
                |step: usize| fit_with(ring_for(step), false).is_none_or(|f| f.0 >= ctx_blocks);
            let step = fitting_ring_step(ring_step, PREFILL_CHUNK, fits);
            if step != ring_step {
                let shorter = ring_for(step).expect("rings on");
                tracing::info!(
                    ctx_size = cfg.ctx_size,
                    max_seqs,
                    ring_step_was = ring_step,
                    ring_step = step,
                    ring_positions_was = r.positions(),
                    ring_positions = shorter.positions(),
                    "shorter sliding-window rings so the context fits (a sequence's prompt \
                     steps are at most the ring step)"
                );
                ring_step = step;
                ring = Some(shorter);
            }
        }
        let stage_state_bytes = |layers: &std::ops::Range<usize>| state_bytes_with(layers, ring);
        let stage_ring_bytes = |layers: &std::ops::Range<usize>| ring_bytes_with(layers, ring);
        let fit = fit_with(ring, true);
        let fit_full = fit.map(|f| f.1);
        let fit = fit.map(|f| f.0);
        let mut n_blocks = match cfg.kv_cache_tokens {
            Some(t) => kv_explicit(t.div_ceil(KV_BLOCK), fit, shared).map_err(LoadError::Device)?,
            None => {
                let want = auto_kv_want(max_seqs, ctx_blocks, cache_split, shared);
                fit.map_or(want, |f| want.min(f))
            }
        };
        // The host tier's tokens: as given, or automatic (a discrete GPU whose memory does not
        // hold a context, see `host_tier`).
        let auto_tier = cfg.kv_host_tokens.is_none();
        let mut host_tokens = match cfg.kv_host_tokens {
            Some(t) => t,
            None => auto_host_tokens(
                &layout,
                &cfg,
                &hp,
                (fit, fit_full),
                (ctx_blocks, n_blocks),
                auto_kv_want(max_seqs, ctx_blocks, cache_split, shared),
                (ring.is_some(), types, shared),
            )
            .map_err(LoadError::Device)?,
        };
        // Tensor parallelism: a small cache first, to measure the compute buffers next to it;
        // the cache is then sized to what they leave (see `fit_tp_cache`).
        let tp_fit = cfg.tp > 1;
        if tp_fit {
            n_blocks = 1;
        } else {
            // A host tier (below) holds the rest of a context.
            let host_blocks = host_tokens.div_ceil(KV_BLOCK);
            n_blocks = kv_floor(
                n_blocks,
                ctx_blocks.saturating_sub(host_blocks).max(1),
                cfg.kv_cache_tokens.is_some(),
                shared,
                native_ctx,
                types,
            )
            .map_err(LoadError::Device)?;
        }
        // `--tp-per-gpu` (#1131) and `--tp` over a meta device (the head mirrored, #1159): each
        // GPU holds the whole LM head, which a step of 64+ rows on a V100 runs as cuBLAS over
        // ggml's F16 copy of it (`vram::head_pool_bytes`, 2.6 GiB for Gemma 4's). A spilled
        // decode state leaves that room beside the reserve (`--tp-per-gpu`) or beside the
        // measured compute buffers (`fit_tp_cache`).
        let head_pool = if cfg.tp_pair.is_some() || tp_fit {
            // A fixed depth (`--deterministic`) drafts at every batch size too.
            let ungated = cfg.spec.fixed
                || (crate::spec::adaptive::cost_model_ungated_large(max_seqs, &cfg.spec)
                    && matches!(drafting, Drafting::Mtp(_) | Drafting::Model(_)));
            let rows = vram::max_head_rows(
                max_seqs,
                spec_k,
                ungated,
                cfg.spec.max_batch,
                depth_policy_max_rows(&cfg.spec, &drafting),
            );
            let head = ["output.weight", "token_embd.weight"]
                .into_iter()
                .find_map(|n| model.w(n).ok());
            match head {
                // A meta tensor's shape is each device's: the head is mirrored (`tp::split_of`),
                // and the meta device's GPUs are what runs it.
                Some(w) if !w.is_host() => vram::head_pool_bytes(
                    volta || any_volta(&layout.tp_devices),
                    // SAFETY: a type query.
                    unsafe { crate::ggml::ffi::ggml_is_quantized(w.ty()) },
                    w.elements(),
                    w.ne()[0] as usize,
                    rows,
                ),
                _ => 0,
            }
        } else {
            0
        };
        let pair_head_pool = if cfg.tp_pair.is_some() { head_pool } else { 0 };
        // Per stage: where its decode state goes when it does not fit beside the weights (#1131).
        let kv_spill: Vec<Option<crate::ggml::BufferType>> = layout
            .stages
            .iter()
            .enumerate()
            .map(|(si, (d, layers))| {
                let dev = (*d)?;
                let reserve =
                    kv_headroom(dev.kind(), cfg.max_batch_tokens, cfg.uma_free) + pair_head_pool;
                let spill = state_spill(
                    &layout,
                    &hp,
                    si,
                    dev,
                    (stage_state_bytes(layers), stage_ring_bytes(layers)),
                    reserve,
                    !layout.tp_devices.is_empty() || cfg.tp_pair.is_some(),
                );
                match spill {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("the decode state cannot spill into host memory: {e}");
                        None
                    }
                }
            })
            .collect();
        let made = if par_hp.is_some() {
            Executor::new_glm_par(
                model,
                backends,
                n_blocks,
                max_seqs + prefill_slots,
                saved_slots,
                ring,
                types,
            )
        } else {
            let stages: Vec<_> = backends
                .into_iter()
                .zip(layout.stages.iter().map(|(_, r)| r.clone()))
                .collect();
            Executor::new_staged_spill(
                model,
                stages,
                n_blocks,
                max_seqs + prefill_slots,
                saved_slots,
                ring,
                types,
                kv_spill,
            )
        };
        let mut exec = made
            .and_then(|mut exec| {
                if layout.node.is_some() || tp_start.is_some() {
                    // Their commits are not deferred (`Executor::commit`), so neither are the
                    // pools' (#836 part 38).
                    exec.no_deferred_commits();
                }
                if spec_k > 0 {
                    exec.enable_spec(spec_k + 1)?;
                }
                Ok(exec)
            })
            .map_err(|e| {
                LoadError::Device(format!(
                    "cannot allocate the decode state ({n_blocks} KV blocks, {max_seqs} sequences) \
                 on {}: {e}",
                    buft.name()
                ))
            })?;
        // The stop tokens a pick below `min_tokens` may not take (#1176). Set before the node
        // and TP followers branch off: a layer split's rank 1 makes the step's picks.
        let stop_tokens = chat_format.stop_tokens(&tokenizer);
        exec.set_stop_tokens(&stop_tokens);
        let mut saved_host_bytes = 0u64;
        if saved_host {
            let moved = exec.saved_states_to_host().map_err(|e| {
                LoadError::Device(format!(
                    "cannot allocate the {saved_slots} saved prefix-cache states in pinned host \
                     memory: {e}"
                ))
            })?;
            saved_host_bytes = moved.iter().sum::<usize>() as u64;
            tracing::info!(
                states = saved_slots,
                gib = saved_host_bytes as f64 / (1u64 << 30) as f64,
                "saved prefix-cache states in pinned host memory (--prefix-states-vram keeps them \
                 in VRAM)"
            );
        }
        // Threads off CPU 0, each on its GPU's node: the engine thread (and the gather and
        // refill threads it spawns) with stage 0's, every stage's launches with their own.
        let stage_cpus: Vec<Vec<u32>> = match (&topo, &layout.arenas) {
            (Some(topo), Some(arenas)) => arenas
                .iter()
                .map(|&(_, node)| crate::hw::numa::helper_cpus(topo, node))
                .collect(),
            // `--numa auto` with GPUs on several nodes (#422): no arena to follow, the GPUs' own.
            (Some(topo), None) if crate::hw::numa_bind::per_gpu_threads() => layout
                .stages
                .iter()
                .map(|(d, _)| {
                    d.and_then(cuda_index)
                        .and_then(crate::hw::numa_bind::gpu_node)
                        .map_or_else(Vec::new, |node| crate::hw::numa::helper_cpus(topo, node))
                })
                .collect(),
            _ => Vec::new(),
        };
        // Off the cores of the host's expert share (#268), when it runs.
        let stage_cpus = crate::residency::cpu_share::leave_pool_cores(stage_cpus);
        let engine_cpus = stage_cpus.first().cloned().unwrap_or_default();
        if !engine_cpus.is_empty() {
            tracing::debug!(?stage_cpus, "helper threads pinned per stage");
        }
        exec.set_stage_cpus(stage_cpus);
        // A micro-chunk must fit the rings (their step), as a whole chunk does.
        let micro = match ring {
            Some(_) => cfg.pipeline_micro.min(ring_step),
            None => cfg.pipeline_micro,
        };
        exec.set_pipeline_micro(micro);
        let pipeline_prompt = if layout.stages.len() > 1 && micro > 0 {
            cfg.pipeline_prompt
        } else {
            0
        };
        exec.set_split_batches(cfg.split_batches);
        exec.set_dataflow(cfg.long_prompts.dataflow);
        exec.set_dataflow_peer(cfg.long_prompts.dataflow_peer);
        exec.set_dataflow_async(cfg.long_prompts.dataflow_async);
        // Groups in flight need stages; a group's graphs come back with it (one per slot base
        // and kind: decode and verify).
        // Over two machines rank 1 is one more stage, and groups in flight are the default
        // (two: one on each machine; `--pipeline-groups 1` turns them off).
        let pipeline_groups = if layout.node.is_some() {
            if cfg.pipeline_groups == 0 {
                2
            } else {
                cfg.pipeline_groups
            }
        } else if layout.stages.len() > 1 && cfg.tp <= 1 && !layout.glm_par {
            cfg.pipeline_groups
        } else {
            0
        };
        exec.set_pipe_graphs(2 * pipeline_groups.max(1));
        // The paged kernels are Legion's own ops, which a meta device cannot split: tensor
        // parallelism gathers (ggml ops only).
        exec.set_attention(if cfg.tp > 1 {
            crate::model::core::Attention::Gather
        } else {
            cfg.attention
        });
        // Verify steps write up to k positions past the context.
        exec.set_max_seq_tokens(cfg.ctx_size + spec_k);
        // The expert cache's work after a target step runs beside the next draft pass, and
        // after a draft pass beside the next target step; over several GPUs, a decode or
        // verify step's work by stage beside the next stage's compute, also without MTP
        // (#267; `--no-after-overlap` turns it off).
        exec.set_after_overlap(cfg.switches.after_overlap.unwrap_or(true));
        // Each next stage of a decode or verify step queued while the one before computes
        // (#814; `--no-stage-launch-ahead` turns it off).
        exec.set_launch_ahead(cfg.switches.launch_ahead.unwrap_or(true));
        // A layer split's verify commit queued beside the first MTP draft pass (#814; on by
        // default on compute capability 7.x, `--no-commit-beside-draft` turns it off).
        exec.set_commit_beside_draft(
            cfg.switches
                .commit_beside_draft
                .unwrap_or_else(crate::model::core::q8_head::device_supported),
        );
        // Of that commit, the last stage's queued behind the second draft pass rather than in
        // front of it (#814; opt-in, `--commit-last-late`).
        exec.set_commit_last_late(cfg.switches.commit_last_late.unwrap_or(false));
        // Each stage's verify commit captured as one stage graph (#814; opt-in,
        // `--commit-capture`).
        crate::model::core::stage_ops::set_commit_capture(
            cfg.switches.commit_capture.unwrap_or(false),
        );
        // Each MTP draft pass queues the next one's inputs (#827; opt-in, `--mtp-ahead`).
        exec.set_mtp_ahead(cfg.switches.mtp_ahead.unwrap_or(false));
        // Of those passes, ones of up to N rows are queued before the last pass's picks land,
        // behind the GPU's own stop check (#827; opt-in, `--mtp-pipe-rows N`).
        exec.set_mtp_pipe_rows(cfg.switches.mtp_pipe_rows.unwrap_or(0));
        exec.set_glm_mtp_overlap(cfg.glm_mtp_overlap_seqs);
        match &drafting {
            Drafting::Model(d) => exec.capture_hidden(d.hidden_layers()),
            // Rank 1 holds the head and the last layer: nothing of it here.
            Drafting::Mtp(_) if head_on_peer => {}
            Drafting::Mtp(_) => {
                let wide_head = mtp_head_wide(exec.model());
                exec.set_mtp_head_copy(draft_head_copy(
                    cfg.draft_head_type,
                    hp.arch.name(),
                    integrated,
                    wide_head,
                ));
                let wide_output = exec
                    .model()
                    .w("output.weight")
                    .is_ok_and(|t| wide_float(t.ty()));
                match mtp_lm_head_unused(cfg.spec.draft_lm_head, hp.arch.name()) {
                    Some(why) => tracing::warn!("{why}"),
                    None => exec.set_mtp_lm_head(mtp_lm_head(
                        cfg.spec.draft_lm_head,
                        hp.arch.name(),
                        integrated,
                        wide_output,
                    )),
                }
                exec.enable_mtp().map_err(|e| {
                    LoadError::Device(format!("cannot allocate the MTP head's cache: {e}"))
                })?;
                let head_sparse = hp.arch.mtp_sparse()
                    || crate::model::qwen4::params(&hp)
                        .and_then(|p| p.qsa.as_ref())
                        .is_some_and(|q| q.layer(hp.n_layer));
                let check = mtp_window_check(
                    mtp_window,
                    hp.arch.name(),
                    exec.mtp_paged(),
                    head_sparse,
                    cfg.ctx_size,
                );
                if let Some(why) = &check.note {
                    if mtp_window_given {
                        tracing::warn!(mtp_window, "{why}");
                    } else {
                        tracing::debug!(mtp_window, "default window: {why}");
                    }
                }
                exec.set_mtp_window(check.window);
                exec.capture_hidden(vec![hp.n_layer - 1]);
            }
            _ => {}
        }
        if tp_fit {
            let mtp = matches!(drafting, Drafting::Mtp(_));
            let sched = fit::Sched {
                max_seqs,
                spec_k,
                // As the scheduler cuts prompts: the ring step with rings, else the chunk.
                chunk: if exec.kv_ring().is_some() {
                    ring_step
                } else {
                    cfg.prefill_chunk
                },
                max_batch: cfg.max_batch_tokens,
                prefill_slots,
            };
            n_blocks = match (cfg.kv_memory, cfg.kv_cache_tokens) {
                (KvMemory::Managed, Some(tokens)) => {
                    managed_tp_cache(&mut exec, &cfg, tokens, spec_k)?
                }
                _ => {
                    let tier =
                        tp_tier(&cfg, &layout, exec.kv_block_bytes()).map_err(LoadError::Device)?;
                    let (device, host) = fit_tp_cache(
                        &mut exec, &layout, &mut cfg, sched, mtp, native_ctx, tier, head_pool,
                    )?;
                    host_tokens = host * KV_BLOCK;
                    device
                }
            };
        }
        // `--kv-vmm` (#814): the pools in reserved virtual ranges, before the cache is sized.
        // The default on a V100 with the expert cache over a layer split of 2 GPUs (as measured,
        // the same bits): the pools past the first KV_VMM_KEEP_DEFAULT tokens go to the cache.
        let kv_auto = volta
            && cfg.switches.kv_vmm.is_none()
            && mode == ExpertMode::Cache
            && layout.stages.len() == 2
            && !tp_fit
            && tp_start.is_none()
            && layout.node.is_none()
            && host_tokens == 0;
        // `--spec-gate-cost-model` not given: where MTP drafts pay again past the matvec
        // steps, the gate ends there (#814), before the MTP head's full-batch buffers and the
        // LM head's pool are sized below.
        if !cfg.spec.gate_cost_model_given
            && let Some(until) = vram::mtp_gate_until_default(
                matches!(drafting, Drafting::Mtp(_)),
                cfg.spec.fixed,
                volta,
                mode == ExpertMode::Cache,
                layout.stages.len(),
                tp_fit || tp_start.is_some() || layout.node.is_some(),
                cfg.spec.max_batch,
            )
        {
            cfg.spec.gate_until = until;
            tracing::info!(
                from = cfg.spec.max_batch,
                until,
                "MTP drafts stop only from --spec-max-batch to this many decoding sequences \
                 on this V100 layer split (--spec-gate-cost-model on: at any number above \
                 too; off: never)"
            );
        }
        let kv_keep = cfg
            .switches
            .kv_vmm_keep
            .or(kv_auto.then_some(KV_VMM_KEEP_DEFAULT));
        let mut kv_vmm_all = false;
        if cfg.switches.kv_vmm == Some(true) || kv_auto {
            kv_vmm_all = kv_vmm::apply(&mut exec, &layout, tp_fit)?;
        }
        // `--kv-vmm-keep` (#814): the pools' tails, lent to the expert cache at its sizing.
        let mut kv_lend: Option<kv_vmm::KvLend> = None;
        // Per stage: the bytes of its tail lent at load (for the memory plan).
        let mut kv_lent_bytes: Vec<usize> = Vec::new();
        // A host tier (#352): more blocks, in pinned host memory on each GPU's node, after the
        // ones that fit in device memory. The allocator hands out device blocks first.
        let mut hbm_blocks = n_blocks;
        // Per stage: the NUMA node of its KV host tier (for the startup report).
        let mut tier_nodes: Vec<Option<u32>> = Vec::new();
        // Sequence slots take the tier's buffers (and their alias) even all in VRAM.
        if host_tokens > 0 || exec.kv_slot_blocks() > 0 {
            let mut host_blocks = host_tokens.div_ceil(KV_BLOCK);
            // Tensor parallelism gathers: no staging slots.
            let stage_blocks = if cfg.tp > 1 {
                0
            } else {
                cfg.kv_stage_tokens.div_ceil(KV_BLOCK).min(host_blocks)
            };
            // The staging slots take device blocks' memory; an automatic tier takes them back.
            let cost = exec
                .kv_stage_cost(stage_blocks)
                .min(hbm_blocks.saturating_sub(1));
            hbm_blocks -= cost;
            if auto_tier {
                host_blocks += cost;
            }
            // Sequence slots: each slot's first blocks in VRAM.
            let slot_blocks = exec.kv_slot_blocks();
            let slot_hbm = if slot_blocks > 0 {
                hbm_blocks / max_seqs.max(1)
            } else {
                hbm_blocks
            };
            let tiers = kv_tiers(
                &layout,
                &hp,
                hbm_blocks,
                stage_blocks,
                (
                    slot_hbm,
                    slot_blocks,
                    exec.kv_slot_host_budget().is_some() || exec.kv_slot_lend() > 0,
                    exec.kv_slot_lend(),
                ),
            )
            .map_err(LoadError::Device)?;
            if slot_blocks > 0 {
                // Only spans past a slot's VRAM blocks are copied (#1153).
                let upto = exec.kv_slot_stage().2;
                exec.set_kv_slot_stage(cfg.kv_slot_stage_rows, slot_hbm * KV_BLOCK, upto);
            }
            tier_nodes = tiers.iter().map(|t| t.map(|t| t.node)).collect();
            let nodes: Vec<u32> = tiers.iter().flatten().map(|t| t.node).collect();
            exec.set_kv_tiers(tiers);
            n_blocks = hbm_blocks + host_blocks;
            exec.resize_kv(n_blocks).map_err(|e| {
                LoadError::Device(format!(
                    "cannot allocate a KV host tier of {} tokens: {e}",
                    host_blocks * KV_BLOCK
                ))
            })?;
            // A sequence may now span the device blocks and the host tier.
            exec.set_max_seq_tokens(cfg.ctx_size + spec_k);
            let gib = (host_blocks * exec.kv_block_bytes()) as f64 / (1u64 << 30) as f64;
            if host_blocks > 0 {
                tracing::info!(
                    device_tokens = hbm_blocks * KV_BLOCK,
                    host_tokens = host_blocks * KV_BLOCK,
                    stage_tokens = stage_blocks * KV_BLOCK,
                    host_gib = gib,
                    ?nodes,
                    "KV cache: a host tier past the device blocks"
                );
                if let Some(budget) = exec.kv_slot_host_budget() {
                    let nvlink = tier_link_nvlink(&layout);
                    let held = (budget * exec.kv_block_bytes()) as f64 / (1u64 << 30) as f64;
                    tracing::warn!(
                        "{max_seqs} sequences of {} tokens do not fit in GPU or host memory: \
                        each sequence slot keeps {} tokens in GPU memory and the rest in a KV \
                        host tier on NUMA node(s) {nodes:?}, backed with pinned memory as the \
                        sequences grow, up to {} tokens ({held:.1} GiB) at once; sequences wait \
                        for room past that. {}. Fewer --max-seqs or a shorter --ctx-size keep \
                        every slot backed",
                        cfg.ctx_size,
                        slot_hbm * KV_BLOCK,
                        budget * KV_BLOCK,
                        host_tier::speed_note(nvlink)
                    );
                } else if auto_tier {
                    let nvlink = tier_link_nvlink(&layout);
                    tracing::warn!(
                        "the {}-token context does not fit in GPU memory: a KV host tier of {} \
                        tokens ({gib:.1} GiB of pinned memory on NUMA node(s) {nodes:?}) is on, \
                        after {} tokens in GPU memory. {}. --kv-host-tokens 0 turns it off",
                        cfg.ctx_size,
                        host_blocks * KV_BLOCK,
                        hbm_blocks * KV_BLOCK,
                        host_tier::speed_note(nvlink)
                    );
                }
            }
        }
        // The measured step compute buffers per stage (for the startup report).
        let mut measured_compute: Vec<usize> = Vec::new();
        // Per stage: a long prompt's micro-chunk step buffer as measured (#1216).
        let mut measured_long: Vec<usize> = Vec::new();
        // Per stage, for the memory plan: the VRAM kept free of the expert cache for step
        // compute buffers, and the prompt steps' room the cache took (may spill to host).
        let mut planned_compute: Vec<usize> = Vec::new();
        let mut compute_spill: Vec<usize> = Vec::new();
        if mode == ExpertMode::Cache {
            // The cache takes the VRAM the compute buffers leave: measure what the longest
            // prompt steps (a full chunk at the end of the context) need on each stage. A
            // given size (`--expert-cache-gib`) must leave them as much (#433).
            let gpu = layout.stages[0].0.is_some();
            let (compute, long, measured_prompt) = if gpu {
                let mut tokens = cfg.prefill_chunk.min(cfg.max_batch_tokens).max(1);
                if layout.stages.len() > 1 && cfg.pipeline_micro > 0 {
                    tokens = tokens.min(cfg.pipeline_micro);
                }
                let n_past = cfg.ctx_size.saturating_sub(tokens);
                let err = |what: String| {
                    move |e: crate::model::core::ModelError| {
                        LoadError::Device(format!("cannot build {what}: {e}"))
                    }
                };
                let prompt = exec
                    .measure_prompt_step(tokens, n_past)
                    .map_err(err(format!("a {tokens}-token prompt step")))?;
                // A full decode batch, and with speculation a full verify batch: their graphs are
                // kept beside the prompt step's. With paged attention a batch's buffers grow
                // with the context its sequences read together (sparse attention lists every
                // sequence's pools), which the KV pool bounds: a full batch whose sequences share
                // the pool. Gather attention pads every sequence to the longest: a full batch
                // at the end of the context.
                let past = if exec.steps_paged() {
                    let shared = (n_blocks * KV_BLOCK / max_seqs.max(1)).min(cfg.ctx_size);
                    shared.saturating_sub(spec_k + 1)
                } else {
                    cfg.ctx_size.saturating_sub(spec_k + 1)
                };
                let decode = exec
                    .measure_step(max_seqs, 1, past, false)
                    .map_err(err(format!("a {max_seqs}-sequence decode step")))?;
                let verify = if spec_k > 0 {
                    exec.measure_step(max_seqs, spec_k + 1, past, true)
                        .map_err(err(format!("a {max_seqs}-sequence verify step")))?
                } else {
                    vec![0; prompt.len()]
                };
                // A long prompt's micro-chunk over the lent cache: what a lend must free sizes
                // how much of the cache long prompts keep (`ResidencyConfig::lend_need`).
                let long = if cfg.residency.staging
                    && cfg.residency.lend_keep.is_empty()
                    && cfg.long_prompts.min_tokens > 0
                    && layout.stages.len() > 1
                {
                    let tokens = cfg.long_prompts.micro;
                    exec.measure_prompt_step(tokens, cfg.ctx_size.saturating_sub(tokens))
                        .map_err(err(format!("a {tokens}-token long prompt step")))?
                } else {
                    Vec::new()
                };
                let mib = |v: &[usize]| v.iter().map(|&b| b >> 20).collect::<Vec<_>>();
                tracing::info!(
                    tokens,
                    prompt_mib = ?mib(&prompt),
                    decode_mib = ?mib(&decode),
                    verify_mib = ?mib(&verify),
                    long_mib = ?mib(&long),
                    "step compute buffers"
                );
                // Per stage: the kept mixed graphs (two prompt steps and a verify batch's
                // micro-batches, at most a full verify batch) and the decode graphs (a full
                // batch, whole or split).
                let compute: Vec<usize> = (0..prompt.len())
                    .map(|i| 2 * prompt[i] + verify[i] + decode[i])
                    .collect();
                (compute, long, prompt)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
            // The MTP head's passes at a full batch, on the last stage, when nothing gates its
            // drafts there (`vram::mtp_full_batch`): its draft and take-in buffers, measured.
            let mtp_head = match vram::mtp_full_batch(
                gpu && matches!(drafting, Drafting::Mtp(_)) && !head_on_peer && spec_k > 0,
                !crate::spec::adaptive::cost_model_ungated_large(max_seqs, &cfg.spec),
                max_seqs,
            ) {
                Some(n) => {
                    let past = if exec.steps_paged() {
                        let shared = (n_blocks * KV_BLOCK / n).min(cfg.ctx_size);
                        shared.saturating_sub(spec_k + 1)
                    } else {
                        cfg.ctx_size.saturating_sub(spec_k + 1)
                    };
                    let err = |what: &str| {
                        let what = what.to_string();
                        move |e: crate::model::core::ModelError| {
                            LoadError::Device(format!("cannot build {what}: {e}"))
                        }
                    };
                    let draft = exec
                        .measure_mtp(n, 1, past)
                        .map_err(err("a full batch's MTP draft pass"))?;
                    let take_in = if exec.mtp_shares_kv() {
                        0
                    } else {
                        let n_tok = spec::TAKE_IN_TAIL + spec_k + 1;
                        exec.measure_mtp(n, n_tok, past.saturating_sub(n_tok))
                            .map_err(err("a full batch's MTP take-in pass"))?
                    };
                    tracing::info!(
                        seqs = n,
                        draft_mib = draft >> 20,
                        take_in_mib = take_in >> 20,
                        "MTP head's compute buffers at a full batch (--spec-gate-cost-model off), \
                         kept free of the expert cache on the last GPU"
                    );
                    draft + take_in
                }
                None => 0,
            };
            // ggml-cuda's pool copy of the LM head at the most rows a step asks it for, on the
            // last stage (`vram::head_pool_bytes`): kept free of the cache, and not counted as
            // room a long prompt's lend finds free.
            let head_pool = if gpu {
                let ungated = crate::spec::adaptive::cost_model_ungated_large(max_seqs, &cfg.spec)
                    && matches!(drafting, Drafting::Mtp(_) | Drafting::Model(_));
                let rows = vram::max_head_rows(
                    max_seqs,
                    spec_k,
                    ungated,
                    cfg.spec.max_batch,
                    depth_policy_max_rows(&cfg.spec, &drafting),
                );
                match exec.model().w("output.weight") {
                    Ok(w) if !w.is_host() && !w.is_meta() => vram::head_pool_bytes(
                        volta,
                        // SAFETY: a type query.
                        unsafe { crate::ggml::ffi::ggml_is_quantized(w.ty()) },
                        w.elements(),
                        w.ne()[0] as usize,
                        rows,
                    ),
                    _ => 0,
                }
            } else {
                0
            };
            if head_pool > 0 {
                tracing::info!(
                    head_pool_mib = head_pool >> 20,
                    "ggml's F16 copy of the LM head (cuBLAS at 64+ rows), kept free of the \
                     expert cache on the last GPU"
                );
            }
            // `--expert-cache-fill-prompt-room`: the prompt steps' room goes to the cache where
            // their buffers can spill into host memory. The default on V100s (an AC922).
            let fill_prompt_room = cfg.residency.fills_prompt_room(volta);
            if fill_prompt_room && layout.stages.iter().any(|(d, _)| d.is_some_and(spills)) {
                tracing::info!(
                    "expert cache takes the prompt steps' compute room (prompt buffers may spill \
                     into host memory; --no-expert-cache-fill-prompt-room keeps it)"
                );
            }
            let cache_compute: Vec<usize> = if fill_prompt_room {
                let prompt_room = compute
                    .iter()
                    .zip(&measured_prompt)
                    .map(|(&c, &p)| c.saturating_sub(2 * p));
                layout
                    .stages
                    .iter()
                    .zip(prompt_room.zip(&compute))
                    .map(
                        |((d, _), (lean, &full))| {
                            if d.is_some_and(spills) { lean } else { full }
                        },
                    )
                    .collect()
            } else {
                compute.clone()
            };
            compute_spill = compute
                .iter()
                .zip(&cache_compute)
                .map(|(&c, &l)| c.saturating_sub(l))
                .collect();
            // The head's room stays out of the kept step graphs' budget below: its graphs
            // keep buffers of their own.
            let mut cache_compute = cache_compute;
            if let Some(last) = cache_compute.last_mut() {
                *last += mtp_head + head_pool;
            }
            planned_compute = cache_compute.clone();
            let mut pinned = vec![0; cache_compute.len()];
            if let Some(last) = pinned.last_mut() {
                *last = head_pool;
            }
            // `--kv-vmm-keep` (#814): the pools' blocks past the kept ones give their VRAM to
            // the cache, and a lend must free it again beside a long step's buffers (the
            // blocks are mapped while the cache is lent).
            measured_long.clone_from(&long);
            let mut long = long;
            if let Some(tokens) = kv_keep {
                let plain = hbm_blocks == n_blocks
                    && exec.kv_slot_blocks() == 0
                    && layout.node.is_none()
                    && tp_start.is_none()
                    && cfg.long_prompts.min_tokens > 0
                    && !long.is_empty();
                if kv_vmm_all && plain {
                    let keep = crate::residency::kv_vmm::keep_blocks(tokens, KV_BLOCK, n_blocks);
                    if let Some((lend, bytes)) =
                        kv_vmm::KvLend::at_load(&exec, &layout, keep, n_blocks)?
                    {
                        for (l, b) in long.iter_mut().zip(&bytes) {
                            *l += b;
                        }
                        kv_lent_bytes = bytes;
                        kv_lend = Some(lend);
                    }
                } else if cfg.switches.kv_vmm_keep.is_some() {
                    tracing::warn!(
                        "--kv-vmm-keep: needs --kv-vmm on every stage's GPU, no KV host tier, \
                         sequence slots, tensor parallelism or second node, and long prompts \
                         lending the cache (a layer split); ignored"
                    );
                }
            }
            build_expert_cache(
                &mut exec,
                &layout,
                &cfg.residency,
                &cache_compute,
                &long,
                &pinned,
            )?;
            // The kept step graphs stay within what the cache left them (#430).
            exec.set_compute_budget(compute.iter().map(|&b| b + COMPUTE_MARGIN).collect());
            measured_compute = compute;
        } else if mode == ExpertMode::Host && cfg.tp <= 1 {
            // Host experts leave the KV pool's headroom free: the kept step graphs stay within
            // it, less the reserve, as beside the cache (#433). Discrete GPUs only (a GPU
            // sharing the host's memory has its own reserve).
            let discrete = layout
                .stages
                .iter()
                .all(|(d, _)| d.is_some_and(|d| d.kind() == crate::ggml::DeviceKind::Gpu));
            if discrete {
                let budget: Vec<usize> = layout
                    .stages
                    .iter()
                    .map(|(d, _)| {
                        vram::free_budget(
                            d.map_or(0, |d| d.memory().0),
                            cfg.residency.cache_reserve as usize,
                        )
                    })
                    .collect();
                tracing::info!(
                    budget_mib = ?budget.iter().map(|&b| b >> 20).collect::<Vec<_>>(),
                    "kept step graphs' budget"
                );
                exec.set_compute_budget(budget);
            }
        }
        if cfg.pair_tp {
            if exec.model().experts().is_none() {
                tracing::warn!(
                    "--pair-tp: no expert cache (every expert in VRAM): nothing to pair"
                );
            }
            exec.set_pair_tp(true);
        }
        // An integrated GPU's kept prompt graphs' compute buffers, allocated now at the size
        // of a full chunk at the end of the context, so the first long prompt does not grow
        // them chunk by chunk (#836; `--no-prompt-buffer-reserve`).
        if layout.stages[0].0.is_some() {
            let t_reserve = std::time::Instant::now();
            let mut tokens = cfg.prefill_chunk.min(cfg.max_batch_tokens).max(1);
            if layout.stages.len() > 1 && cfg.pipeline_micro > 0 {
                tokens = tokens.min(cfg.pipeline_micro);
            }
            let n_past = cfg.ctx_size.saturating_sub(tokens);
            let reserved = exec.reserve_prompt_buffers(tokens, n_past).map_err(|e| {
                LoadError::Device(format!("cannot build a {tokens}-token prompt step: {e}"))
            })?;
            if reserved.iter().any(|&(n, _)| n > 0) {
                tracing::info!(
                    tokens,
                    n_past,
                    buffers = ?reserved.iter().map(|r| r.0).collect::<Vec<_>>(),
                    mib_each = ?reserved.iter().map(|r| r.1 >> 20).collect::<Vec<_>>(),
                    seconds = t_reserve.elapsed().as_secs_f64(),
                    "prompt compute buffers reserved"
                );
            }
        }
        // The kept verify graphs' bytes, where they keep a share of their own (#858).
        if crate::model::core::verify_kept_graphs() > 0 {
            let budget: Vec<usize> = layout
                .stages
                .iter()
                .map(|(d, _)| {
                    verify_budget(cfg.switches.verify_kept_mib, d.map_or(0, |d| d.memory().0))
                })
                .collect();
            tracing::info!(
                graphs = crate::model::core::verify_kept_graphs(),
                budget_mib = ?budget.iter().map(|&b| b >> 20).collect::<Vec<_>>(),
                "kept verify graphs' budget per stage (--verify-kept-mib)"
            );
            exec.set_verify_budget(budget);
        }
        tracing::info!(
            model = %id,
            backend = %exec.backend_name(),
            stages = ?exec.stage_layers(),
            experts = ?mode,
            seconds = t.elapsed().as_secs_f64(),
            ctx_size = cfg.ctx_size,
            max_seqs,
            kv_cache_tokens = n_blocks * KV_BLOCK,
            cache_type_k = ?types.k,
            cache_type_v = ?types.v,
            // Per token of context in the pool (the full-attention layers' only, with rings).
            kv_bytes_per_token =
                KvPool::block_bytes_paged(&hp, 0..hp.n_layer, ring.is_some(), types) / KV_BLOCK,
            kv_cache_gib = (n_blocks
                * KvPool::block_bytes_paged(&hp, 0..hp.n_layer, ring.is_some(), types))
                as f64
                / (1u64 << 30) as f64,
            "model loaded"
        );
        // What each GPU holds once loaded (weights, expert cache, KV pool, kept step graphs) and
        // what it leaves free, as the driver reports it. A lend logs its own free VRAM.
        let gpus: Vec<(usize, usize)> = layout
            .stages
            .iter()
            .filter_map(|(d, _)| d.filter(|d| d.kind() == crate::ggml::DeviceKind::Gpu))
            .map(|d| {
                let (free, total) = d.memory();
                (total.saturating_sub(free) >> 20, free >> 20)
            })
            .collect();
        if !gpus.is_empty() {
            tracing::info!(
                used_mib = ?gpus.iter().map(|g| g.0).collect::<Vec<_>>(),
                free_mib = ?gpus.iter().map(|g| g.1).collect::<Vec<_>>(),
                "VRAM after load"
            );
        }
        if let Some(r) = ring {
            let sliding = KvPool::sliding_token_bytes(&hp, 0..hp.n_layer, types);
            let full = KvPool::block_bytes_paged(&hp, 0..hp.n_layer, true, types) / KV_BLOCK;
            tracing::info!(
                kv_bytes_per_token_full_layers = full,
                kv_bytes_per_token_sliding_layers = sliding,
                ring_positions = r.positions(),
                rings = r.sets,
                rings_gib = (r.sets * r.positions() * sliding) as f64 / (1u64 << 30) as f64,
                context_tokens_total = n_blocks * KV_BLOCK,
                context_tokens_per_sequence = (n_blocks * KV_BLOCK).min(cfg.ctx_size),
                "sliding-window layers in rings (prefix cache off; --swa-full keeps every position)"
            );
        }

        // Over two machines (#562): rank 1 runs its layers for rank 0 from here on; rank 0
        // connects to it, and both keep the smaller KV pool.
        if let Some(share) = &layout.node {
            let peer = cfg
                .node
                .as_ref()
                .map_or("", |n| n.peer.as_str())
                .to_string();
            if n_blocks > hbm_blocks {
                return Err(LoadError::Device(
                    "a KV host tier over two nodes: not supported".into(),
                ));
            }
            if share.rank == 1 {
                // The MTP head drafts here, for rank 0 (#560).
                let drafter = match drafting {
                    Drafting::Mtp(d) => Some(d),
                    _ => None,
                };
                return Err(node::follow(exec, drafter, share, &peer, n_blocks));
            }
            let link = crate::node::Link::connect(&peer, std::time::Duration::from_secs(600))
                .map_err(|e| LoadError::Device(format!("--node-peer {peer}: {e}")))?;
            let mut remote = crate::model::core::Remote::new(link);
            let theirs = match remote.call(crate::node::hello(hp.n_layer, share.split, n_blocks)) {
                Ok(crate::node::Reply::Hello { n_blocks }) => n_blocks as usize,
                Ok(r) => {
                    return Err(LoadError::Device(format!(
                        "node rank 1 answered hello with {r:?}"
                    )));
                }
                Err(e) => return Err(LoadError::Device(format!("node rank 1: {e}"))),
            };
            if theirs < n_blocks {
                exec.resize_kv(theirs).map_err(|e| {
                    LoadError::Device(format!("cannot resize the KV pool to rank 1's: {e}"))
                })?;
                n_blocks = theirs;
                hbm_blocks = hbm_blocks.min(theirs);
            }
            tracing::info!(
                peer,
                layers = ?share.layers,
                kv_cache_tokens = n_blocks * KV_BLOCK,
                "node rank 0: rank 1 runs the layers from {}",
                share.split
            );
            exec.set_remote(remote);
        }
        // Tensor parallelism over machines (#619): the ranks above 0 run rank 0's steps from
        // here on; rank 0 connects to them, and every rank keeps the fewest KV blocks.
        if let Some(start) = tp_start.take() {
            // `--tp-per-gpu`'s ranks keep host tiers of their own (#1131): each reads its own
            // blocks past the device's in its steps, as one GPU alone does.
            tp_node::check_exec(&exec, cfg.tp_pair.is_none() && n_blocks > hbm_blocks)?;
            if start.rank() > 0 {
                return Err(tp_node::follow(
                    exec,
                    start,
                    n_blocks,
                    ring_save_bufts(&layout),
                ));
            }
            let blocks = tp_node::lead(&mut exec, start, n_blocks)?;
            n_blocks = n_blocks.min(blocks);
            hbm_blocks = hbm_blocks.min(blocks);
        }

        // #1159: VRAM the fit could not see free (it measured the steps' buffers next to
        // what the load held then) goes to the slots' shared chunks too, the steps' buffers,
        // their margin and LEND_SAFETY left free.
        if exec.kv_slot_lend() > 0 {
            let least = (layout.tp_devices.iter())
                .map(|d| d.memory().0)
                .min()
                .unwrap_or(0);
            let keep = exec.kv_slot_lend_keep() + LEND_SAFETY;
            let n_dev = layout.tp_devices.len().max(1);
            let chunk = (LAZY_CHUNK_BLOCKS * exec.kv_block_bytes() / n_dev).max(1);
            let most = (max_seqs * exec.kv_slot_blocks()).saturating_sub(exec.kv_slot_lend());
            let want = (least.saturating_sub(keep) / chunk * LAZY_CHUNK_BLOCKS).min(most);
            if want > 0 {
                let added = exec.lend_kv_slots_more(want);
                tracing::info!(
                    added_tokens = added * KV_BLOCK,
                    shared_device_tokens = exec.kv_slot_lend() * KV_BLOCK,
                    free_mib = least >> 20,
                    kept_free_mib = keep >> 20,
                    "KV slots: more shared VRAM after the load"
                );
            }
        }
        let load_report = load_report(
            &layout,
            &mut exec,
            &cfg,
            LoadFacts {
                mode,
                native_ctx,
                auto_tier,
                n_blocks,
                hbm_blocks,
                tier_nodes: &tier_nodes,
                bytes_per_token: KvPool::block_bytes_paged(
                    &hp,
                    0..hp.n_layer,
                    ring.is_some(),
                    types,
                ) / KV_BLOCK,
                ring,
                state_bytes: &stage_state_bytes,
                compute: &measured_compute,
                ngram,
                engram,
                saved_host_bytes,
            },
        );
        log_memory_plan(
            &layout,
            &mut exec,
            &load_report,
            MemoryPlanFacts {
                ctx_size: cfg.ctx_size,
                max_seqs,
                n_blocks,
                kv_lend: kv_lend.as_ref(),
                kv_lent_bytes: &kv_lent_bytes,
                planned_compute: &planned_compute,
                compute_spill: &compute_spill,
            },
        );

        let (submit, rx) = std::sync::mpsc::channel();
        let expert_stats = exec.experts_mut().map(|experts| experts.stats());
        let stats = Arc::new(Mutex::new(EngineStats {
            rooflines: bandwidth::stage_rooflines(&layout),
            // Lazily backed slots (#1131): what the pool hands out at once.
            kv_blocks_total: exec
                .kv_slot_host_budget()
                .map_or(n_blocks, |held| lazy_slot_cap(held, max_seqs).min(n_blocks))
                as u32,
            expert_cache_slots: expert_stats.as_ref().map_or(0, |s| s.slots as u32),
            expert_cache_layers: expert_stats.as_ref().map_or(0, |s| s.layers as u32),
            ..Default::default()
        }));
        // A head with its own cache takes in every step's positions even when nothing is drafted;
        // a block drafter drafts any depth in one pass.
        let block = drafting.block_drafts().is_some();
        let depth =
            ((matches!(drafting, Drafting::Mtp(_)) || block) && !cfg.spec.fixed).then(|| {
                if block {
                    crate::spec::depth::DepthModel::block()
                } else {
                    crate::spec::depth::DepthModel::new(!exec.mtp_shares_kv())
                }
            });
        // Sequence slots (`--kv-slots`): one per admitted sequence, its blocks in order.
        let slot_blocks = exec.kv_slot_blocks();
        // Prefix caching where the radix tree cannot serve (rings, slots): warm sequences,
        // each ring saved at the end of its prompt in pinned host memory.
        let warm_on =
            saved_slots > 0 && (ring.is_some() || slot_blocks > 0) && !hp.has_slot_state();
        if warm_on && let Some(r) = ring {
            let bytes = exec.ring_copy_bytes();
            // A host tier backed as it fills holds its budget's memory only later: the
            // checkpoints get a third of what it leaves of the node, at most.
            let budget = exec.kv_slot_host_budget();
            let block_bytes = exec.kv_block_bytes();
            let free = (!integrated).then(|| ring_save_free(&layout)).flatten();
            let free = free.map(|f| budget.map_or(f, |b| f.min((b * block_bytes) as u64)));
            let slots = ring_checkpoint_slots(cfg.ring_checkpoints, r.sets, bytes, free);
            let (slots, budget) = ring_checkpoints_in_budget(
                slots,
                r.sets,
                bytes,
                budget,
                block_bytes,
                slot_blocks + max_seqs * LAZY_CHUNK_BLOCKS,
            );
            match exec.enable_ring_saves(&ring_save_bufts(&layout), slots) {
                Ok(saved) => {
                    if let Some(held) = budget
                        && budget != exec.kv_slot_host_budget()
                    {
                        exec.set_kv_slot_host_budget(budget);
                        if let Ok(mut s) = stats.lock() {
                            s.kv_blocks_total = lazy_slot_cap(held, max_seqs).min(n_blocks) as u32;
                        }
                    }
                    tracing::info!(
                        gib = saved as f64 / (1u64 << 30) as f64,
                        checkpoints = slots,
                        every = cfg.ring_checkpoint_every,
                        span = cfg.ring_checkpoint_span,
                        host_tier_backed_tokens = budget.map(|b| b * KV_BLOCK),
                        "prefix caching: finished sequences stay warm, their rings saved at \
                         the end of each prompt and every `every` tokens over its last `span` \
                         (context checkpoints, in pinned host memory)"
                    )
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    "prefix caching: no room to save the rings; a warm sequence resumes only \
                     while its ring still holds the window"
                ),
            }
        }
        let kv_admission = preempt::resolve(cfg.kv_admission, pipeline_groups, slot_blocks);
        let slo = slo::start(cfg.slo);
        // `--pipeline-mtp-split` (#765): the rounds' stage times, for the server's split.
        let split_probe = (cfg.pipeline_mtp_split
            && pipeline_groups > 1
            && drafts_with_mtp_head(&cfg.spec)
            && exec.stage_layers().len() > 1)
            .then(|| {
                let layers = exec.stage_layers().iter().map(|r| r.len()).collect();
                Arc::new(Mutex::new(pipe_split::Probe::new(layers)))
            });
        // `--mmproj`: the vision tower (#146). Other ranks and nodes get no images' rows.
        let vision = match &cfg.mmproj {
            Some(_) if cfg.tp_net.is_some() || cfg.node.is_some() => {
                return Err(LoadError::Device(
                    "--mmproj: not with --tp-rank or --node-rank (yet)".into(),
                ));
            }
            Some(path) => Some(Arc::new(vision::Vision::load(
                path,
                &gguf.architecture().unwrap_or_default(),
                exec.hp().n_embd as usize,
                &tokenizer,
                cfg.threads,
            )?)),
            None => None,
        };
        let slot_host_budget = exec.kv_slot_host_budget();
        let slot_lend = exec.kv_slot_lend();
        let ring_ckpt_slots = exec.ring_save_slots();
        // #1216: a lend caps the long prompt's micro-chunks to what each stage's GPU has left
        // free. GLM-5.3-Flash over a layer split only (not its parallel decode). Its step
        // buffer grows with the past (the indexer's scores): measured once more at none.
        let long_micro = cfg.long_prompts.micro;
        let long_step = if crate::model::glm5next::params(exec.hp()).is_some()
            && exec.model().glm_par.is_none()
            && measured_long.len() == layout.stages.len()
            && long_micro > 0
        {
            match exec.measure_prompt_step(long_micro, 0) {
                Ok(first) if first.len() == measured_long.len() => {
                    (layout.stages.iter().zip(first).zip(&measured_long))
                        .map(|(((d, _), b0), &b)| LongStep {
                            device: *d,
                            first: b0.min(b),
                            last: b,
                        })
                        .collect()
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let long_past = cfg.ctx_size.saturating_sub(long_micro).max(1);
        let worker = Worker {
            exec,
            tokenizer: tokenizer.clone(),
            chat_format: chat_format.clone(),
            stop_tokens,
            masks: MaskPool::new(cfg.grammar_threads),
            pool: if slot_blocks > 0 {
                let mut pool =
                    BlockPool::slotted(max_seqs as u32, slot_blocks as u32, KV_BLOCK as u32);
                if let Some(held) = slot_host_budget {
                    pool.set_slotted_cap(lazy_slot_cap(held, max_seqs) as u32);
                }
                pool
            } else {
                // The lent tail's blocks come last (`--kv-vmm-keep`).
                let fast = kv_lend
                    .as_ref()
                    .map_or(hbm_blocks, |l| l.keep.min(hbm_blocks));
                BlockPool::with_fast(n_blocks as u32, KV_BLOCK as u32, fast as u32)
            },
            kv_lend,
            // A prefix hit would need the sliding-window layers' ring at the cut point, or
            // a sequence's slot to hold another's blocks: off.
            radix: (saved_slots > 0 && ring.is_none() && slot_blocks == 0)
                .then(|| RadixCache::new(KV_BLOCK as u32)),
            free_rings: ring.map_or_else(Vec::new, |r| (0..r.sets).rev().collect()),
            free_kv_slots: if slot_blocks > 0 {
                (0..max_seqs as u32).rev().collect()
            } else {
                Vec::new()
            },
            // Backed as they fill where the host cannot hold them all, or where they share
            // VRAM (#1159).
            slot_backing: (slot_host_budget.is_some() || slot_lend > 0).then(|| {
                slot_backing::SlotBacking::new(
                    max_seqs,
                    slot_blocks,
                    hbm_blocks / max_seqs.max(1),
                    slot_host_budget.unwrap_or(usize::MAX),
                    slot_lend,
                )
            }),
            warm: Vec::new(),
            warm_on,
            warm_stamp: 0,
            ring_window: window,
            ring_ckpts: if cfg.ring_checkpoints == Some(0) {
                RingCheckpoints::single(ring_ckpt_slots)
            } else {
                RingCheckpoints::new(ring_ckpt_slots)
            },
            ring_ckpt_every: cfg.ring_checkpoint_every,
            ring_ckpt_span: cfg.ring_checkpoint_span,
            ring_ckpts_log: cfg.ring_checkpoint_log,
            saved: SavedStates::new(saved_slots),
            prefix_interval: cfg.prefix_cache_interval,
            gather: cfg.deterministic_gather,
            admit_gather: cfg.admit_gather,
            capture_cut: cfg.prefix_capture_cut,
            state_digests: cfg.prefix_state_digests,
            end_saves: saved_slots > 0 && ring.is_none() && slot_blocks == 0 && hp.has_slot_state(),
            max_seqs,
            prefill_slots,
            scheduling: cfg.scheduling,
            kv_admission,
            kv_keep: 0,
            slo,
            max_batch_tokens: cfg.max_batch_tokens.max(1),
            prefill_chunk: cfg.prefill_chunk.max(1),
            prefill_even: cfg.prefill_even,
            ring_step,
            max_ctx: cfg.ctx_size,
            reserved: 0,
            stats: stats.clone(),
            depth,
            mtp_k: spec_k.min(cfg.spec.k),
            drafting,
            spec_k,
            spec_config: cfg.spec.clone(),
            next_id: 0,
            long: long_prompts,
            micro,
            pipeline_prompt,
            pipeline_groups,
            pipeline_min_group: cfg.pipeline_min_group.max(1),
            split_probe: split_probe.clone(),
            lent: false,
            lent_short: false,
            can_lend: cfg.long_prompts.min_tokens > 0,
            long_step,
            long_past,
            long_cap: 0,
            long_reserve: 0,
            clock: Default::default(),
            outbox: stream::Outbox::new(tokenizer.clone(), cfg.stream_thread_seqs),
        };
        let worker = std::thread::Builder::new()
            .name("legion-qwen35".into())
            .spawn(move || {
                if !engine_cpus.is_empty() {
                    let _ = crate::hw::numa::pin_current_thread(&engine_cpus);
                    // Placed: the threads it starts leave its core (#250).
                    crate::hw::host_core::mark_host_thread();
                }
                worker.run(rx)
            })
            .map_err(|e| LoadError::Device(format!("cannot start the worker thread: {e}")))?;

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(Self {
            info: ModelInfo {
                id,
                owned_by: "legion".into(),
                created,
                context_length: Some(cfg.ctx_size as u32),
                sampling,
                vision: vision.is_some(),
            },
            tokenizer,
            vision,
            template,
            chat_format,
            grammars: Mutex::new(GrammarCache::default()),
            ctx_size: cfg.ctx_size,
            max_waiting: cfg.max_waiting,
            queued: Arc::new(AtomicUsize::new(0)),
            stats,
            submit: Some(submit),
            worker: Some(worker),
            load_report,
            split_probe,
        })
    }

    /// `--pipeline-mtp-split` (#765): what the decoding groups' rounds measured so far and the
    /// layer split it suggests ([`pipe_split::Probe::hint`]); `None` when not measuring or
    /// before enough rounds.
    pub fn mtp_split_hint(&self) -> Option<pipe_split::SplitHint> {
        self.split_probe.as_ref()?.lock().ok()?.hint()
    }

    /// Where the load put the model's weights, caches and tables (`legion serve`'s report).
    pub fn load_report(&self) -> &report::LoadReport {
        &self.load_report
    }

    /// The request's prompt tokens (what `generate` runs and reports as `prompt_tokens`), for
    /// chats whether generation starts inside reasoning, and its images, placed in the tokens
    /// but not yet encoded ([`vision`]).
    #[allow(clippy::type_complexity)]
    fn prompt(
        &self,
        req: &GenerateRequest,
    ) -> Result<(Vec<u32>, Option<bool>, Vec<vision::Pending>), EngineError> {
        if req.model != self.info.id {
            return Err(EngineError::ModelNotFound(req.model.clone()));
        }
        let (text, reasoning) = self.render(req)?;
        let tokens = self.tokenizer.encode_with(&text, true, true);
        let images: Vec<&super::ImageInput> = match &req.prompt {
            Prompt::Chat { messages, .. } => messages.iter().flat_map(|m| &m.images).collect(),
            Prompt::Text(_) => Vec::new(),
        };
        if images.is_empty() {
            return Ok((tokens, reasoning, Vec::new()));
        }
        let vision = self.vision.as_ref().ok_or_else(|| {
            EngineError::InvalidRequest("this model does not take images (no --mmproj)".into())
        })?;
        let (tokens, pending) = vision.expand(tokens, &images)?;
        Ok((tokens, reasoning, pending))
    }

    /// The prompt text, and for chats whether generation starts inside reasoning.
    fn render(&self, req: &GenerateRequest) -> Result<(String, Option<bool>), EngineError> {
        match &req.prompt {
            Prompt::Text(text) => Ok((text.clone(), None)),
            Prompt::Chat {
                system,
                messages,
                tools,
            } => {
                let template = self.template.as_ref().ok_or_else(|| {
                    EngineError::InvalidRequest("this model has no chat template".into())
                })?;
                // The client's chat_template_kwargs, with `thinking` (already resolved from
                // them and reasoning_effort) as the template's `enable_thinking`.
                let mut kwargs = req.template_kwargs.clone();
                if let Some(on) = req.response_format.as_ref().map(|_| false).or(req.thinking) {
                    kwargs.insert("enable_thinking".into(), Value::Bool(on));
                }
                // Qwen templates only accept a system message first. Later ones (Claude Code
                // sends system turns mid-conversation) become user turns at their position,
                // which keeps the conversation's earlier prefix (and its cache) unchanged.
                let messages: Vec<super::Message> = messages
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let mut m = m.clone();
                        if m.role == Some(Role::System) && (i > 0 || system.is_some()) {
                            m.role = Some(Role::User);
                        }
                        m
                    })
                    .collect();
                let text = template
                    .render(system.as_deref(), &messages, tools, true, &kwargs)
                    .map_err(|e| EngineError::InvalidRequest(e.to_string()))?;
                let reasoning = self.chat_format.prompt_reasoning(&text);
                Ok((text, reasoning.filter(|_| req.response_format.is_none())))
            }
        }
    }

    /// `generate`'s first half: the request checked, its grammars compiled and its prompt
    /// tokenized, for [`Self::submit_prepared`] on this engine or on a replica of it
    /// (`--tp-replicas`: every replica loads the same model, so tokenizes alike).
    pub fn prepare(&self, mut req: GenerateRequest) -> Result<Prepared, EngineError> {
        let grammar = req
            .response_format
            .as_ref()
            .map(|f| self.compile_format(f))
            .transpose()
            .map_err(|e| EngineError::InvalidRequest(e.to_string()))?;
        if grammar.is_some() {
            req.thinking = Some(false);
        }
        let (prompt, reasoning, images) = self.prompt(&req)?;
        if prompt.is_empty() {
            return Err(EngineError::InvalidRequest("the prompt is empty".into()));
        }
        // Room for at least one generated token.
        if prompt.len() >= self.ctx_size {
            return Err(EngineError::ContextOverflow {
                prompt: prompt.len() as u32,
                context: self.ctx_size as u32,
            });
        }
        // The images through the vision tower, once the request is known to fit.
        let media = match &self.vision {
            Some(v) if !images.is_empty() => Some(v.encode(&prompt, images)?),
            _ => None,
        };
        let tools = match req.prompt {
            Prompt::Chat { tools, .. } => tools,
            Prompt::Text(_) => Vec::new(),
        };
        // A response format already decides every token.
        let tool_grammar = grammar
            .is_none()
            .then(|| self.compile_strict_tools(&tools))
            .flatten();
        Ok(Prepared {
            grammar,
            tool_grammar,
            prompt,
            media,
            sampling: req.sampling,
            max_tokens: req.max_tokens,
            stop: req.stop,
            ignore_eos: req.ignore_eos,
            min_tokens: req.min_tokens,
            reasoning,
            tools,
            cancel: req.cancel,
            sched: req.scheduling,
        })
    }

    /// `generate`'s second half: queues a [`Prepared`] request. A full waiting queue hands it
    /// back (`Refused::Full`), so a replica router can offer it to another replica.
    pub fn submit_prepared(
        &self,
        p: Prepared,
    ) -> Result<TokenStream, super::replicas::Refused<Prepared>> {
        use super::replicas::Refused;
        if self
            .queued
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.max_waiting).then_some(n + 1)
            })
            .is_err()
        {
            return Err(Refused::Full(Box::new(p)));
        }
        let queued = QueueSlot(self.queued.clone());
        let (out, rx) = mpsc::unbounded_channel();
        self.submit
            .as_ref()
            .ok_or_else(|| {
                Refused::Failed(EngineError::Internal("the engine is shutting down".into()))
            })?
            .send(Job {
                grammar: p.grammar,
                tool_grammar: p.tool_grammar,
                prompt: p.prompt,
                media: p.media,
                sampling: p.sampling,
                max_tokens: p.max_tokens,
                stop: p.stop,
                ignore_eos: p.ignore_eos,
                min_tokens: p.min_tokens,
                reasoning: p.reasoning,
                tools: p.tools,
                out,
                cancel: p.cancel,
                sched: p.sched,
                queued_at: Instant::now(),
                _queued: queued,
            })
            .map_err(|_| {
                Refused::Failed(EngineError::Internal("the model worker stopped".into()))
            })?;
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}

/// A request [`LmEngine::prepare`] checked and tokenized, not yet queued.
pub struct Prepared {
    grammar: Option<crate::grammar::Grammar>,
    tool_grammar: Option<crate::grammar::Grammar>,
    prompt: Vec<u32>,
    media: Option<(Arc<Media>, Vec<u32>)>,
    sampling: SamplingParams,
    max_tokens: u32,
    stop: Vec<String>,
    ignore_eos: bool,
    min_tokens: u32,
    reasoning: Option<bool>,
    tools: Vec<super::Tool>,
    cancel: CancellationToken,
    sched: Scheduling,
}

impl super::replicas::Replica for LmEngine {
    type Prepared = Prepared;

    fn prepare(&self, req: GenerateRequest) -> Result<Prepared, EngineError> {
        LmEngine::prepare(self, req)
    }

    fn prompt(p: &Prepared) -> &[u32] {
        &p.prompt
    }

    fn submit(&self, p: Prepared) -> Result<TokenStream, super::replicas::Refused<Prepared>> {
        self.submit_prepared(p)
    }
}

impl Engine for LmEngine {
    fn models(&self) -> Vec<ModelInfo> {
        vec![self.info.clone()]
    }

    fn check_response_format(
        &self,
        format: &super::ResponseFormat,
    ) -> Result<(), crate::grammar::GrammarError> {
        self.compile_format(format).map(|_| ())
    }

    fn generate(&self, req: GenerateRequest) -> Result<TokenStream, EngineError> {
        let p = self.prepare(req)?;
        self.submit_prepared(p).map_err(|e| match e {
            super::replicas::Refused::Full(_) => EngineError::Overloaded,
            super::replicas::Refused::Failed(e) => e,
        })
    }

    fn count_tokens(&self, req: &GenerateRequest) -> Result<u32, EngineError> {
        Ok(self.prompt(req)?.0.len() as u32)
    }

    fn stats(&self) -> EngineStats {
        self.stats.lock().map(|s| *s).unwrap_or_default()
    }
}

/// Owns everything that touches the device, and the scheduler's bookkeeping.
struct Worker {
    exec: Executor,
    tokenizer: Arc<Tokenizer>,
    stop_tokens: Vec<u32>,
    chat_format: ChatFormat,
    /// Finds constrained sequences' next masks during the forward pass.
    masks: Arc<MaskPool>,
    pool: BlockPool,
    /// Prompt prefixes whose KV blocks are kept for reuse (`None`: prefix caching off).
    radix: Option<RadixCache>,
    saved: SavedStates,
    /// [`LmConfig::prefix_cache_interval`].
    prefix_interval: usize,
    /// [`LmConfig::deterministic_gather`].
    gather: Option<std::time::Duration>,
    /// [`LmConfig::admit_gather`].
    admit_gather: Option<std::time::Duration>,
    /// [`LmConfig::prefix_capture_cut`].
    capture_cut: bool,
    /// [`LmConfig::prefix_state_digests`].
    state_digests: bool,
    /// Whether a finished sequence saves its state past its reply ([`Self::save_end`]): prefix
    /// caching on, through the radix tree, with recurrent state.
    end_saves: bool,
    max_seqs: usize,
    /// [`LmConfig::prefill_slots`].
    prefill_slots: usize,
    /// [`LmConfig::scheduling`].
    scheduling: SchedulingPolicy,
    /// [`LmConfig::kv_admission`], as it runs (see [`preempt::resolve`]).
    kv_admission: KvAdmission,
    /// `on-demand`: free blocks admission leaves for the decoding sequences' growth (one
    /// each, set by [`Self::make_room`]); 0 under `reserve`.
    kv_keep: usize,
    /// `--itl-target` / `--ttft-target`: the step budget's model and state ([`slo`]).
    slo: Option<slo::Slo>,
    max_batch_tokens: usize,
    prefill_chunk: usize,
    /// [`LmConfig::prefill_even`].
    prefill_even: bool,
    /// With rings: the most prompt tokens a sequence takes per step ([`LmConfig::ring_step`]).
    ring_step: usize,
    /// Max tokens per sequence.
    max_ctx: usize,
    /// Blocks promised to admitted sequences and not yet allocated: admission only counts free
    /// blocks beyond these, so a running sequence never runs out.
    reserved: usize,
    stats: Arc<Mutex<EngineStats>>,
    drafting: Drafting,
    /// Max drafts per sequence and step (0: speculation off).
    spec_k: usize,
    spec_config: SpecConfig,
    /// Rings of the sliding-window layers no sequence holds (with rings; empty without).
    free_rings: Vec<usize>,
    /// KV sequence slots no sequence holds (with slots; empty without).
    free_kv_slots: Vec<u32>,
    /// Slots whose host tier is backed as they fill (#1131): what each has backed.
    slot_backing: Option<slot_backing::SlotBacking>,
    /// Finished sequences whose cache is kept for prefix caching where the radix tree cannot
    /// serve (rings, slots); `warm_on`: whether finished sequences go there.
    warm: Vec<Warm>,
    warm_on: bool,
    /// The last [`Warm::stamp`].
    warm_stamp: u64,
    /// The sliding window the rings serve (with rings).
    ring_window: usize,
    /// Saved copies of the rings for the warm sequences ([`RingCheckpoints`]); none without
    /// rings or prefix caching; one per ring at its prompt's end with `--ring-checkpoints 0`
    /// ([`RingCheckpoints::single`]: no interval checkpoints, no guard in [`Self::warm_match`]).
    ring_ckpts: RingCheckpoints,
    /// `--ring-checkpoint-every` and `--ring-checkpoint-span`.
    ring_ckpt_every: usize,
    ring_ckpt_span: usize,
    /// `--ring-checkpoint-log`.
    ring_ckpts_log: bool,
    /// MTP drafting without `--spec-fixed`: the depth policy's measured costs
    /// ([`crate::spec::depth`]).
    depth: Option<crate::spec::depth::DepthModel>,
    /// The depth the drafts in flight were made for: verify steps take `1 +` this many rows per
    /// sequence (with `depth`; otherwise `1 + spec_k`).
    mtp_k: usize,
    /// The next sequence's [`SeqId`].
    next_id: SeqId,
    long: LongPrompts,
    /// Pipeline micro-chunk outside long prompts.
    micro: usize,
    /// Tokens per step of a prompt prefilling alone over several stages
    /// ([`LmConfig::pipeline_prompt`]; 0: off).
    pipeline_prompt: usize,
    /// Decoding groups in flight over the stages ([`LmConfig::pipeline_groups`]; below 2:
    /// off).
    pipeline_groups: usize,
    /// Fewest sequences per group ([`LmConfig::pipeline_min_group`]).
    pipeline_min_group: usize,
    /// `--pipeline-mtp-split`: each round's stage times and exiting work (#765).
    split_probe: Option<Arc<Mutex<pipe_split::Probe>>>,
    /// Whether the expert cache's VRAM is lent to a long prompt now.
    lent: bool,
    /// Whether the cache is lent to short prompts only (`LongPrompts::short_min_tokens`):
    /// steps are planned as with the cache.
    lent_short: bool,
    /// Whether lending is possible (cleared after a failure).
    can_lend: bool,
    /// #1216: per stage, its long prompt step's compute buffers as measured at load (empty:
    /// no cap, see [`Self::long_micro_cap`]).
    long_step: Vec<LongStep>,
    /// The past [`LongStep::last`] was measured after.
    long_past: usize,
    /// The micro-chunk cap of the current lend (0: none).
    long_cap: usize,
    /// VRAM a KV tail mapped during the current lend leaves free for its step buffers.
    long_reserve: usize,
    /// `--kv-vmm-keep` (#814): the KV pools' tail, lent to the expert cache while no sequence
    /// needs it ([`kv_vmm::KvLend`]).
    kv_lend: Option<kv_vmm::KvLend>,
    /// The engine thread's time per step by phase (#47).
    clock: host_time::HostClock,
    /// Sequences' text on the stream thread ([`stream`]): the step's messages.
    outbox: stream::Outbox,
}

/// A finished sequence's cache, kept for the next prompt that starts with its tokens (the next
/// turn of a chat, the same prompt again): with rings the radix tree cannot share blocks (a
/// prefix hit needs the sliding-window layers' ring as it was at the cut point), nor with
/// slots (a slot holds only its own sequence's blocks). Like llama.cpp's slot reuse and
/// context checkpoints. Evicted, least recently finished first, when a sequence needs its
/// ring, slot or blocks.
struct Warm {
    table: BlockTable,
    ring: usize,
    kv_slot: u32,
    /// The tokens whose K/V the cache holds, from position 0.
    tokens: Vec<u32>,
    /// Positions up to here may have been written into the ring (drafts included).
    written: usize,
    /// Its last prompt's length (`tokens` goes on with the reply).
    prompt: usize,
    stamp: u64,
}

/// [`Worker::warm_match`]: the warm sequence's index, the tokens resumed, and the ring
/// checkpoint its ring comes back from (slot, length), if any.
type WarmMatch = (usize, usize, Option<(usize, usize)>);

/// An admitted sequence.
struct Seq<'t> {
    active: Active<'t>,
    prompt: Vec<u32>,
    /// Its images ([`Media`]) and its prefix-cache key ([`Seq::cache_key`]), if it has any.
    media: Option<(Arc<Media>, Vec<u32>)>,
    /// Tokens whose K/V and recurrent state are in the cache.
    n_past: usize,
    table: BlockTable,
    /// `table`'s block ids, as the model takes them.
    blocks: Vec<u32>,
    /// Of `Worker::reserved`, the part this sequence may still allocate.
    reserved: usize,
    /// Recurrent-state slot: `0..` while decoding (its index there), above `max_seqs` while
    /// prefilling.
    slot: usize,
    /// The token to feed next (decoding).
    next: u32,
    /// Prompt positions at which to save the recurrent state for prefix caching, ascending.
    saves: Vec<(usize, SaveKind)>,
    started: Instant,
    /// The engine's model bytes when the sequence was admitted (#391): the request-done line
    /// reports the steps since.
    bytes_at_start: crate::ggml::model_bytes::Totals,
    /// For drafters: the sequence so far (prompt, then every generated token up to `next`).
    /// Empty when speculation is off.
    history: Vec<u32>,
    id: SeqId,
    /// This sequence's model-free drafter.
    drafter: Option<Box<dyn TokenDrafter>>,
    /// Drafts to verify in the next step.
    draft: Draft,
    /// Its ring of the sliding-window layers' caches ([`crate::model::core::KvRing`]); 0
    /// without rings.
    ring: usize,
    /// Its slot of the KV cache, with slots ([`BlockPool::slotted`]); 0 without.
    kv_slot: u32,
    /// The most drafts `draft` was asked for.
    draft_cap: usize,
    adaptive: crate::spec::adaptive::AdaptiveState,
    /// `--spec-mtp-ngram`: drafts from earlier occurrences in the sequence.
    ngram: Option<crate::spec::mtp_ngram::MtpNgram>,
    /// Whether `draft` is the n-gram's, and its match-length class.
    draft_ngram: Option<usize>,
    /// Its place in the waiting queue: fair share's key, and under `--scheduling-policy
    /// priority` the order prompts get the prefill budget in.
    ticket: Ticket,
    /// Blocks its whole length needs (`prompt + max_tokens`, what `reserve` admission takes).
    need: usize,
    /// Preempted while decoding and reading its tokens again ([`preempt`]): once read, it
    /// decodes on from `next` without sampling.
    resuming: bool,
    /// A prompt chunk the MTP head has yet to take in ([`DeferredTakeIn`]).
    take_in: Option<DeferredTakeIn>,
}

impl Seq<'_> {
    /// Its images, for the model's steps.
    fn media(&self) -> Option<&Media> {
        self.media.as_ref().map(|(m, _)| &**m)
    }

    /// What the prefix cache knows its prompt by: the prompt, each image's tokens told apart by
    /// the image ([`vision`]).
    fn cache_key(&self) -> &[u32] {
        self.media.as_ref().map_or(&self.prompt, |(_, key)| key)
    }
}

/// Where an admitted prompt starts ([`Worker::place`]).
struct Placed {
    table: BlockTable,
    /// Blocks reserved for it.
    fresh: usize,
    /// Prompt tokens the prefix cache matched, and those resumed from (in the cache).
    matched: usize,
    resumed: usize,
    ring: usize,
    kv_slot: u32,
}

enum Placement {
    Placed(Placed),
    /// Not enough free KV blocks yet.
    Wait,
    Failed(String),
}

/// The MTP head's take-in of a prompt chunk that left at most [`KV_BLOCK`] prompt tokens (#477).
/// A prompt that is not a whole number of KV blocks stops a step at its last full block (its
/// state is saved there for the prefix cache) and reads the rest in a step of its own; the
/// head's take-in of the long chunk (~0.6 s for a 123K prompt on two V100s) ran between those
/// two steps, before the prompt's first token. It now runs in the next step's
/// [`Worker::finish_step`], once the first token is out and before the rest's own take-in, so
/// the head runs the same passes in the same order.
struct DeferredTakeIn {
    /// The step's captured rows ([`crate::model::core::StepOutput::hidden`]).
    hidden: std::sync::Arc<Vec<Vec<f32>>>,
    /// Positions that step ran (the rows of each captured layer).
    n_step: usize,
    /// The chunk's first row in the step.
    off: usize,
    /// The chunk's first position, and its length.
    start: usize,
    ran: usize,
}

impl DeferredTakeIn {
    /// The chunk's rows of each captured layer.
    fn rows(&self) -> Vec<&[f32]> {
        self.hidden
            .iter()
            .map(|h| {
                let w = h.len() / self.n_step;
                &h[self.off * w..(self.off + self.ran) * w]
            })
            .collect()
    }
}

/// What admission decided about the queue's first job.
enum Admission<'t> {
    Admitted(Box<Seq<'t>>),
    /// Not enough free KV blocks yet.
    Wait(Job),
    Failed(Job, String),
}

impl Worker {
    fn run(mut self, rx: std::sync::mpsc::Receiver<Job>) {
        let tokenizer = self.tokenizer.clone();
        let mut waiting: WaitQueue<Waiting<'_>> = WaitQueue::new(self.scheduling);
        let queue = |waiting: &mut WaitQueue<Waiting<'_>>, job: Job| {
            let sched = job.sched;
            waiting.push(Waiting::New(job), sched);
        };
        let mut prefilling: Vec<Seq<'_>> = Vec::new();
        let mut decoding: Vec<Seq<'_>> = Vec::new();
        // Decoding sequences in groups in flight ([`LmConfig::pipeline_groups`]).
        let mut pipe = pipe::Pipe::default();
        let mut open = true;
        let mut experts_at = Instant::now();
        // `--deterministic` ([`waves`]): requests that arrived since the running wave started.
        let mut held: Vec<Job> = Vec::new();
        loop {
            // The cache gets its VRAM back once no prompt is being read (before idling too).
            // `--kv-vmm-keep`: while the KV pools' tail holds some of the cache's VRAM, all but
            // that much (`restore_cache`).
            if prefilling.is_empty() && !pipe.is_active() {
                self.kv_shrink(&decoding);
            }
            if self.lent && prefilling.is_empty() {
                self.restore_cache();
            }
            // Block for work only when idle; otherwise take what arrived and keep stepping.
            if decoding.is_empty()
                && prefilling.is_empty()
                && !pipe.is_active()
                && waiting.is_empty()
            {
                if let Some(quiet) = self.gather {
                    // A new wave: what arrived meanwhile, and what follows within the window.
                    held.retain(|j| !(j.cancel.is_cancelled() || j.out.is_closed()));
                    if held.is_empty() {
                        if !open {
                            break;
                        }
                        self.outbox.flush();
                        match rx.recv() {
                            Ok(job) => {
                                self.prefetch_first(&job.prompt);
                                held.push(job);
                            }
                            Err(_) => break,
                        }
                    }
                    let first = held.len();
                    open &= waves::gather(&rx, quiet, &mut held);
                    for job in &held[first..] {
                        self.prefetch_first(&job.prompt);
                    }
                    waves::order(&mut held, |j| waves::WaveKey {
                        prompt: j.media.as_ref().map_or(&j.prompt, |(_, key)| key),
                        max_tokens: j.max_tokens,
                        min_tokens: j.min_tokens,
                        ignore_eos: j.ignore_eos,
                    });
                    for job in held.drain(..) {
                        queue(&mut waiting, job);
                    }
                } else {
                    if !open {
                        break;
                    }
                    // Text queued outside a step (a preempted sequence's end) leaves before
                    // idling.
                    self.outbox.flush();
                    match rx.recv() {
                        Ok(job) => {
                            // Its first chunk's table rows, read while it is admitted
                            // ([`first_rows`]).
                            self.prefetch_first(&job.prompt);
                            queue(&mut waiting, job);
                            // `--admit-gather-ms`: the rest of a burst, admitted with it.
                            if let Some(quiet) = self.admit_gather {
                                let mut more = Vec::new();
                                open &= waves::gather(&rx, quiet, &mut more);
                                for job in more {
                                    self.prefetch_first(&job.prompt);
                                    queue(&mut waiting, job);
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
            let mut at = Instant::now();
            loop {
                match rx.try_recv() {
                    Ok(job) if self.gather.is_some() => held.push(job),
                    Ok(job) => queue(&mut waiting, job),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            waiting.retain(|w| w.keep(&mut self.outbox));
            // `on-demand`: room for the decoding sequences' next step, victims preempted.
            if self.on_demand()
                && let Err(e) = self.make_room(&mut decoding, &mut prefilling, &mut waiting)
            {
                self.fail_all(&e, &mut pipe, &mut decoding, &mut prefilling);
            }

            while !waiting.is_empty()
                && decoding.len() + pipe.len() + prefilling.len() < self.max_seqs
                && prefilling.len() < self.prefill_slots
            {
                let slot = (self.max_seqs..self.max_seqs + self.prefill_slots)
                    .find(|s| prefilling.iter().all(|q| q.slot != *s))
                    .expect("a free prefill slot");
                let running = decoding
                    .iter()
                    .chain(pipe.seqs())
                    .chain(prefilling.iter())
                    .map(|s| s.ticket.scheduling().share_key);
                let (item, ticket) = waiting.pop(self.max_seqs, running).unwrap();
                let admitted = match item {
                    Waiting::New(job) => match self.admit(&tokenizer, job, ticket, slot) {
                        Admission::Admitted(seq) => Ok(seq),
                        Admission::Wait(job) => Err(Waiting::New(job)),
                        Admission::Failed(job, e) => {
                            let _ = job.out.send(TokenEvent::Error(e));
                            continue;
                        }
                    },
                    Waiting::Preempted(seq) => match self.resume(seq, slot) {
                        preempt::Resume::Admitted(seq) => Ok(seq),
                        preempt::Resume::Wait(seq) => Err(Waiting::Preempted(seq)),
                        preempt::Resume::Failed => continue,
                    },
                };
                match admitted {
                    Ok(seq) => prefilling.push(*seq),
                    Err(item) => {
                        waiting.put_back(item, ticket);
                        break;
                    }
                }
            }
            // Prefill budget goes to prompts in `prefilling`'s order: the queue's under
            // `priority` (a stable sort, so prompts already in order stay put).
            if self.scheduling == SchedulingPolicy::Priority {
                prefilling.sort_by_key(|s| s.ticket.order());
            }
            if decoding.is_empty() && prefilling.is_empty() && !pipe.is_active() {
                // Nothing admissible: only possible while the KV pool is short, which an idle
                // engine never is. Wait for the next request rather than spin.
                if !waiting.is_empty() {
                    self.outbox.flush();
                    match rx.recv() {
                        Ok(job) if self.gather.is_some() => held.push(job),
                        Ok(job) => queue(&mut waiting, job),
                        Err(_) => open = false,
                    }
                }
                continue;
            }
            self.clock.lap(host_time::Phase::Loop, &mut at);
            let t_step = Instant::now();
            let stepped = if self.pipeline_groups > 1 {
                self.step_piped(&mut pipe, &mut decoding, &mut prefilling)
            } else {
                self.step(&mut decoding, &mut prefilling)
            };
            if let Some(slo) = &mut self.slo {
                // The whole step as a decoding stream waits for it: plan, forward pass,
                // sampling and drafts.
                slo.finished(t_step.elapsed(), stepped.is_ok());
            }
            let mut at = Instant::now();
            if let Err(e) = stepped {
                self.fail_all(&e, &mut pipe, &mut decoding, &mut prefilling);
            }
            // The expert cache's counters walk every layer: at the status line's pace only.
            let expert_stats = (experts_at.elapsed() >= EXPERT_STATS_EVERY)
                .then(|| self.exec.experts_mut().map(|experts| experts.stats()))
                .flatten();
            // The n-gram table's row cache counters, at the same pace.
            let ngram_stats = (experts_at.elapsed() >= EXPERT_STATS_EVERY)
                .then(|| self.exec.model().ngram_table().map(|t| t.stats()))
                .flatten();
            if expert_stats.is_some() || ngram_stats.is_some() {
                experts_at = Instant::now();
            }
            if let Ok(mut s) = self.stats.lock() {
                s.model_bytes = self.exec.model_bytes();
                s.running = (decoding.len() + pipe.len() + prefilling.len()) as u32;
                s.prefilling = prefilling.len() as u32;
                s.decoding = (decoding.len() + pipe.len()) as u32;
                s.waiting = (waiting.len() + held.len()) as u32;
                s.waiting_classes = waiting.classes();
                s.waiting_keys = waiting.waiting_keys();
                s.kv_blocks_used = s.kv_blocks_total - self.pool.free_blocks();
                if let Some(experts) = expert_stats {
                    s.expert_cache_hits = experts.hits;
                    s.expert_cache_misses = experts.misses;
                }
                if let Some(t) = ngram_stats {
                    s.ngram_rows = t.rows;
                    s.ngram_cache_hits = t.cache_hits;
                    s.ngram_hot_rows = t.hot_rows;
                }
            }
            self.outbox.flush();
            self.clock.lap(host_time::Phase::Loop, &mut at);
            self.clock.step(decoding.len() + pipe.len());
        }
    }

    /// Admits `job` into prefill slot `slot`: matches its prompt against the prefix cache,
    /// reserves KV blocks for its whole length (`on-demand`: for its prompt, [`preempt`]) and
    /// sets up its recurrent state.
    fn admit<'t>(
        &mut self,
        tokenizer: &'t Tokenizer,
        job: Job,
        ticket: Ticket,
        slot: usize,
    ) -> Admission<'t> {
        let len = job.prompt.len();
        let max_len = (len + job.max_tokens as usize).min(self.max_ctx);
        // A verify step writes up to k positions past the last one kept.
        let need = (max_len + self.spec_k).div_ceil(KV_BLOCK);
        if need > self.pool_blocks() {
            let e = format!(
                "the sequence needs {} KV cache tokens, the cache holds {}",
                need * KV_BLOCK,
                self.pool_blocks() * KV_BLOCK
            );
            return Admission::Failed(job, e);
        }
        let take = self.admit_blocks(len, need);
        let Placed {
            table,
            fresh,
            matched,
            resumed,
            ring,
            kv_slot,
        } = match self.place(
            job.media.as_ref().map_or(&job.prompt, |(_, key)| key),
            take,
            slot,
        ) {
            Placement::Placed(p) => p,
            Placement::Wait => return Admission::Wait(job),
            Placement::Failed(e) => return Admission::Failed(job, e),
        };
        let stateless = self.stateless();
        let mut seq = Seq {
            active: Active::new(
                tokenizer,
                &self.chat_format,
                &job,
                self.max_ctx,
                &self.masks,
                &mut self.outbox,
                self.next_id,
            ),
            blocks: table.blocks().iter().map(|b| b.0).collect(),
            table,
            reserved: fresh,
            slot,
            next: 0,
            n_past: resumed,
            saves: if self.saved.is_empty() || stateless {
                Vec::new()
            } else {
                save_points(len, matched, resumed, KV_BLOCK, self.prefix_interval)
            },
            started: Instant::now(),
            bytes_at_start: self.exec.model_bytes(),
            history: if self.drafting.is_on() || self.warm_on || self.on_demand() || self.end_saves
            {
                job.prompt.clone()
            } else {
                Vec::new()
            },
            id: self.next_id,
            drafter: self.drafting.for_seq(),
            draft: Draft::default(),
            ring,
            kv_slot,
            draft_cap: 0,
            adaptive: Default::default(),
            ngram: (self.spec_config.mtp_ngram > 0 && matches!(self.drafting, Drafting::Mtp(_)))
                .then(|| crate::spec::mtp_ngram::MtpNgram::new(self.spec_config.mtp_ngram)),
            draft_ngram: None,
            ticket,
            need,
            resuming: false,
            take_in: None,
            prompt: job.prompt,
            media: job.media,
        };
        self.next_id += 1;
        if self.drafting.is_on() {
            seq.active.spec = Some(SpecCounters::default());
        }
        seq.active.cached = resumed as u32;
        if let Ok(mut s) = self.stats.lock() {
            s.prefix_cache_hit_tokens += resumed as u64;
        }
        let _ = seq.active.out.send(TokenEvent::Started {
            prompt_tokens: len as u32,
        });
        Admission::Admitted(Box::new(seq))
    }

    /// Places `prompt` in prefill slot `slot`: matches it against the prefix cache and the
    /// warm sequences, reserves its `take` blocks (less those it resumes from), and sets up
    /// its ring, KV slot and recurrent state.
    fn place(&mut self, prompt: &[u32], take: usize, slot: usize) -> Placement {
        let len = prompt.len();
        // Longest cached prefix with a saved state, block-aligned, leaving at least one prompt
        // token to compute (its logits start the reply). A model without recurrent state
        // (Gemma 4) needs none: its KV blocks are the whole state.
        let (mut matched, mut resumed, mut from) = (0, 0, None);
        // A saved state past a reply, whose partly filled KV block this prompt takes over.
        let mut tail_from = None;
        let mut table = BlockTable::new();
        let stateless = self.stateless();
        if let Some(radix) = &mut self.radix {
            let m = radix.match_prefix(prompt);
            matched = m.num_tokens as usize;
            let found = if stateless {
                let n = matched.min(len - 1) / KV_BLOCK * KV_BLOCK;
                (n > 0).then_some((None, n, None))
            } else {
                self.saved
                    .find(prompt, matched, KV_BLOCK)
                    .map(|f| (Some(f.slot), f.len, f.tail))
            };
            if let Some((saved_slot, n, tail)) = found {
                resumed = n;
                from = saved_slot;
                let full = &m.blocks[..n / KV_BLOCK];
                table = match tail {
                    None => BlockTable::from_shared(&mut self.pool, full),
                    Some(t) => {
                        // The full blocks and the tail, which holds tokens up to `n`.
                        let blocks: Vec<_> = full.iter().copied().chain([t]).collect();
                        let mut t = BlockTable::from_shared(&mut self.pool, &blocks);
                        t.truncate(&mut self.pool, n as u32);
                        tail_from = saved_slot;
                        t
                    }
                };
            }
            radix.unlock(&m);
        }
        // A warm sequence's cache: its blocks, ring (as it is, or put back from its save) and
        // slot.
        let mut warm = None;
        if self.warm_on
            && let Some((i, m, restore)) = self.warm_match(prompt)
        {
            let mut w = self.warm.swap_remove(i);
            w.table.truncate(&mut self.pool, m as u32);
            table = std::mem::take(&mut w.table);
            resumed = m;
            warm = Some((w, restore));
        } else if self.warm_on && !self.warm.is_empty() && self.ring_ckpts_log {
            self.log_warm_miss(prompt);
        }
        let fresh = take.saturating_sub(resumed / KV_BLOCK);
        // `on-demand` leaves the decoding sequences room to grow (0 under `reserve`).
        let want = fresh + self.kv_keep;
        let available = |w: &Self| (w.pool.free_blocks() as usize).saturating_sub(w.reserved);
        if want > available(self) {
            let short = want - available(self);
            if let Some(radix) = &mut self.radix {
                radix.evict(&mut self.pool, short as u32);
            }
            while want > available(self) && self.evict_warm() {}
            if want > available(self) {
                for b in self.saved.drop_tails() {
                    self.pool.decref(b);
                }
            }
            if want > available(self) {
                match warm {
                    Some((mut w, _)) => {
                        w.table = table;
                        w.tokens.truncate(resumed);
                        self.warm.push(w);
                    }
                    None => table.release(&mut self.pool),
                }
                return Placement::Wait;
            }
        }
        self.reserved += fresh;
        // The prompt writes on into the tail block: the saved entry goes (its state is copied
        // out below), and its reference with it.
        if let Some(s) = tail_from
            && let Some(b) = self.saved.take(s)
        {
            self.pool.decref(b);
        }
        let (ring, kv_slot) = match warm {
            Some((w, restore)) => {
                if let Some((ckpt, at)) = restore {
                    self.exec.save_ring(w.ring, ckpt, at, true);
                    self.ring_ckpts.touch(ckpt);
                }
                // Its checkpoints past the resume point hold other tokens.
                self.ring_ckpts.truncate(w.ring, resumed);
                (w.ring, w.kv_slot)
            }
            None => self.take_ring_and_slot(),
        };

        let copied = match from {
            Some(s) => self.exec.copy_states(&[(Pool::Saved, s, Pool::Seq, slot)]),
            None => {
                self.exec.clear_slot(slot);
                Ok(())
            }
        };
        if let Err(e) = copied {
            table.release(&mut self.pool);
            self.free_ring_and_slot(ring, kv_slot);
            self.reserved -= fresh;
            return Placement::Failed(e.to_string());
        }
        if self.state_digests && from.is_some() {
            self.log_state_digest(Pool::Seq, slot, resumed, "resumed from");
        }
        Placement::Placed(Placed {
            table,
            fresh,
            matched,
            resumed,
            ring,
            kv_slot,
        })
    }

    /// Ends every running sequence with error `e`: after a failed step (or `on-demand`'s
    /// state moves) their caches are unknown.
    fn fail_all<'t>(
        &mut self,
        e: &str,
        pipe: &mut pipe::Pipe<'t>,
        decoding: &mut Vec<Seq<'t>>,
        prefilling: &mut Vec<Seq<'t>>,
    ) {
        tracing::error!(error = %e, "batch step failed");
        if self.pipeline_groups > 1 {
            self.exec.pipe_abort();
        }
        let grouped = pipe.take_all();
        for mut seq in decoding
            .drain(..)
            .chain(prefilling.drain(..))
            .chain(grouped)
        {
            seq.active.fail(&mut self.outbox, e.to_string());
            // A failed step leaves its cache unknown.
            self.release_as(&mut seq, false);
        }
    }

    fn pool_blocks(&self) -> usize {
        self.exec.kv_blocks()
    }

    /// The warm sequence ([`Warm`]) to resume `prompt` from: its index, the tokens resumed
    /// (the longest common prefix, leaving one prompt token to compute) and, when its ring
    /// comes back from a checkpoint ([`RingCheckpoints`]), that copy's slot and length.
    ///
    /// A warm sequence that would keep less than half its tokens is left alone while a ring
    /// and KV slot are free: a request that shares only a template's first tokens with a chat
    /// (a title or follow-up request, another chat) would otherwise take the chat's cache over
    /// and cut it down to those tokens. Not with checkpoints off (`--ring-checkpoints 0`).
    fn warm_match(&self, prompt: &[u32]) -> Option<WarmMatch> {
        let ring = self.exec.kv_ring();
        let holds = |m: usize, end: usize| self.ring_holds(m, end);
        let spare = !self.ring_ckpts.is_single() && self.spare_ring_and_slot();
        let mut best: Option<WarmMatch> = None;
        for (i, w) in self.warm.iter().enumerate() {
            let common = (w.tokens.iter().zip(prompt))
                .take_while(|(a, b)| a == b)
                .count();
            let m = common.min(prompt.len().saturating_sub(1));
            let got = if holds(m, w.written) {
                Some((m, None))
            } else if ring.is_some() {
                (self.ring_ckpts.best(w.ring, m, holds)).map(|(m, at, slot)| (m, Some((slot, at))))
            } else {
                None
            };
            // A prompt sharing little of a warm sequence (a side request sharing the chat
            // template or a system prompt) takes a spare ring and slot rather than this
            // sequence's cache; one that covers its whole last prompt (the same question again)
            // resumes it.
            let steals = spare && 2 * m < w.tokens.len() && m + 1 < w.prompt;
            if let Some((m, restore)) = got
                && m > 0
                && !steals
                && best.is_none_or(|b| m > b.1)
            {
                best = Some((i, m, restore));
            }
        }
        best
    }

    /// Whether a ring written up to position `end` still holds the window a token at `m`
    /// reads (each position overwrites the one `positions` before it; a step's drafts run
    /// ahead). Always without rings.
    fn ring_holds(&self, m: usize, end: usize) -> bool {
        (self.exec.kv_ring())
            .is_none_or(|r| m + r.positions() >= end + self.ring_window + self.spec_k + 2)
    }

    /// Whether a new sequence would get a free ring and KV slot without evicting a warm one.
    fn spare_ring_and_slot(&self) -> bool {
        let rings = self.exec.kv_ring().is_some();
        let slots = self.pool.slot_blocks() > 0;
        !(rings && self.free_rings.is_empty()) && !(slots && self.free_kv_slots.is_empty())
    }

    /// Saves ring `ring` as it is after `at` tokens in a checkpoint slot.
    fn save_checkpoint(&mut self, ring: usize, at: usize) {
        if let Some(slot) = self.ring_ckpts.store(ring, at) {
            self.exec.save_ring(ring, slot, at, false);
        }
    }

    /// `--ring-checkpoint-log`: why `prompt` resumed nothing from the warm sequences: per warm
    /// sequence, the tokens it shares and its ring's checkpoints.
    fn log_warm_miss(&self, prompt: &[u32]) {
        for w in &self.warm {
            let common = (w.tokens.iter().zip(prompt))
                .take_while(|(a, b)| a == b)
                .count();
            let ckpts: Vec<usize> = self.ring_ckpts.of(w.ring).iter().map(|c| c.0).collect();
            tracing::info!(
                prompt_tokens = prompt.len(),
                warm_tokens = w.tokens.len(),
                common,
                written = w.written,
                checkpoints = ?ckpts,
                "prefix cache: no warm resume"
            );
        }
    }

    /// Drops the least recently finished warm sequence: its blocks, ring and slot are free
    /// again. False when there is none.
    fn evict_warm(&mut self) -> bool {
        let Some(i) = (0..self.warm.len()).min_by_key(|&i| self.warm[i].stamp) else {
            return false;
        };
        let w = self.warm.swap_remove(i);
        w.table.release(&mut self.pool);
        self.free_ring_and_slot(w.ring, w.kv_slot);
        true
    }

    /// A free ring and KV slot (0 without), a warm sequence's if none is free: at most
    /// max_seqs sequences are admitted, one ring and slot each.
    fn take_ring_and_slot(&mut self) -> (usize, u32) {
        let rings = self.exec.kv_ring().is_some();
        let slots = self.pool.slot_blocks() > 0;
        while ((rings && self.free_rings.is_empty()) || (slots && self.free_kv_slots.is_empty()))
            && self.evict_warm()
        {}
        (
            if rings {
                self.free_rings
                    .pop()
                    .expect("a free ring per admitted sequence")
            } else {
                0
            },
            if slots {
                self.free_kv_slots
                    .pop()
                    .expect("a free KV slot per admitted sequence")
            } else {
                0
            },
        )
    }

    fn free_ring_and_slot(&mut self, ring: usize, kv_slot: u32) {
        if self.exec.kv_ring().is_some() {
            self.ring_ckpts.clear(ring);
            self.free_rings.push(ring);
        }
        if self.pool.slot_blocks() > 0 {
            self.free_kv_slots.push(kv_slot);
            // Shared VRAM goes back at once, for the slots that run (#1159).
            if self.slot_backing.as_ref().is_some_and(|b| b.lends()) {
                match self.exec.trim_kv_slot(kv_slot as usize) {
                    Ok(()) => {
                        if let Some(b) = &mut self.slot_backing {
                            b.trimmed(kv_slot);
                        }
                    }
                    Err(e) => tracing::error!(kv_slot, "cannot give back a KV slot: {e}"),
                }
            }
        }
    }

    /// Whether a sequence's whole state is its KV cache (no recurrent layers): prefix caching
    /// then resumes at any cached block boundary, with no saved state.
    fn stateless(&self) -> bool {
        let hp = self.exec.hp();
        !hp.has_slot_state()
    }

    /// Backs slot `slot`'s first `blocks` blocks where its host tier is backed as it fills
    /// ([`slot_backing`]), idle slots giving back their room first if needed; where the slots
    /// share VRAM, as many of them in it as it has, warm slots' moved out if needed.
    fn back_slot(&mut self, slot: u32, blocks: usize) -> Result<(), String> {
        let Some(b) = &self.slot_backing else {
            return Ok(());
        };
        let mut warm: Vec<(u64, u32)> = self.warm.iter().map(|w| (w.stamp, w.kv_slot)).collect();
        warm.sort_unstable();
        let warm: Vec<u32> = warm.into_iter().map(|w| w.1).collect();
        let Some(plan) = b.plan(slot, blocks, &self.free_kv_slots, &warm)? else {
            return Ok(());
        };
        for &s in &plan.trim {
            self.exec.trim_kv_slot(s as usize)?;
            if let Some(b) = &mut self.slot_backing {
                b.trimmed(s);
            }
        }
        for &(w, keep) in &plan.demote {
            let Some(b) = &mut self.slot_backing else {
                break;
            };
            let (to, _) = b.held(w);
            self.exec
                .back_kv_slot(w as usize, to * KV_BLOCK, keep * KV_BLOCK)?;
            if let Some(b) = &mut self.slot_backing {
                b.backed(w, to, keep);
            }
        }
        self.exec
            .back_kv_slot(slot as usize, plan.to * KV_BLOCK, plan.lent * KV_BLOCK)?;
        if let Some(b) = &mut self.slot_backing {
            b.backed(slot, plan.to, plan.lent);
        }
        Ok(())
    }

    /// Grows `seq`'s block table to hold `n` tokens, from its reservation.
    /// Room for `k` drafted tokens past the kept positions of every decoding sequence not in
    /// `done` and every prompt in `joined` ([`Self::grow`]).
    fn grow_for_drafts(
        &mut self,
        decoding: &mut [Seq<'_>],
        done: &[usize],
        prefilling: &mut [Seq<'_>],
        joined: &[usize],
        k: usize,
    ) -> Result<(), String> {
        for (i, seq) in decoding.iter_mut().enumerate() {
            if !done.contains(&i) {
                self.grow(seq, seq.n_past + k)?;
            }
        }
        for &i in joined {
            let seq = &mut prefilling[i];
            self.grow(seq, seq.n_past + k)?;
        }
        Ok(())
    }

    fn grow(&mut self, seq: &mut Seq<'_>, n: usize) -> Result<(), String> {
        let have = seq.table.num_tokens() as usize;
        if n <= have {
            return Ok(());
        }
        let before = seq.table.blocks().len();
        if let Some(l) = self.kv_lend.as_ref().filter(|l| !l.grown) {
            // #1196: a resumed prefix with blocks the last shrink kept in host memory gets
            // them back in VRAM before this step reads them.
            let keep = l.keep as u32;
            let resume = l.host_held > 0 && seq.blocks.iter().any(|&b| b >= keep);
            let need = n.div_ceil(KV_BLOCK).saturating_sub(before);
            self.kv_make_room(need, resume)?;
        }
        if self.pool.slot_blocks() > 0 {
            seq.table
                .append_in(&mut self.pool, (n - have) as u32, seq.kv_slot)
                .map_err(|e| e.to_string())?;
            self.back_slot(seq.kv_slot, seq.table.blocks().len())?;
        } else if seq.table.append(&mut self.pool, (n - have) as u32).is_err() {
            // Reservations make this unreachable; the cache can still give blocks back.
            if let Some(radix) = &mut self.radix {
                radix.evict(&mut self.pool, (n - have).div_ceil(KV_BLOCK) as u32);
            }
            seq.table
                .append(&mut self.pool, (n - have) as u32)
                .map_err(|e| e.to_string())?;
        }
        let added = seq.table.blocks().len() - before;
        seq.blocks
            .extend(seq.table.blocks()[before..].iter().map(|b| b.0));
        let used = added.min(seq.reserved);
        seq.reserved -= used;
        self.reserved -= used;
        Ok(())
    }

    /// Gives back a finished sequence's blocks and the rest of its reservation; with
    /// `warm_on` its cache stays as a [`Warm`] sequence.
    fn release(&mut self, seq: &mut Seq<'_>) {
        self.release_as(seq, self.warm_on);
    }

    /// [`Self::release`]; `keep`: as a warm sequence (when it has a cache).
    fn release_as(&mut self, seq: &mut Seq<'_>, keep: bool) {
        let table = std::mem::take(&mut seq.table);
        let n = seq.n_past.min(seq.history.len());
        // A warm sequence is matched by its tokens, which do not tell images apart.
        if keep && n > 0 && seq.media.is_none() {
            self.warm_stamp += 1;
            self.warm.push(Warm {
                table,
                ring: seq.ring,
                kv_slot: seq.kv_slot,
                tokens: seq.history[..n].to_vec(),
                written: seq.n_past + self.spec_k + 1,
                prompt: seq.prompt.len().min(n),
                stamp: self.warm_stamp,
            });
        } else {
            table.release(&mut self.pool);
            self.free_ring_and_slot(seq.ring, seq.kv_slot);
        }
        self.reserved -= seq.reserved;
        seq.reserved = 0;
        match &mut self.drafting {
            Drafting::Model(d) => d.release(seq.id),
            Drafting::Mtp(d) => {
                d.release(seq.id);
                // The head's drafter on node rank 1 (#560).
                self.exec.release_remote_drafts(seq.id);
            }
            _ => {}
        }
    }

    /// Saves finished `seq`'s state, which holds its first `state_at` tokens (prompt and reply),
    /// for the next turn of its chat, which resends them ([`super::saved_states`]): their full
    /// blocks go into the radix tree, and the saved entry keeps the partly filled block after
    /// them. Queues the state's copy in `copies`, before any slot move.
    fn save_end(
        &mut self,
        seq: &Seq<'_>,
        state_at: usize,
        copies: &mut Vec<(Pool, usize, Pool, usize)>,
    ) {
        // A sequence with images is cached by a key its reply tokens do not extend.
        if !self.end_saves
            || seq.media.is_some()
            || state_at < KV_BLOCK
            || state_at > seq.history.len()
            || state_at > seq.table.num_tokens() as usize
        {
            return;
        }
        let tokens = &seq.history[..state_at];
        let Some(radix) = &mut self.radix else {
            return;
        };
        if self.saved.contains(tokens) {
            return;
        }
        radix.insert(&mut self.pool, tokens, seq.table.blocks());
        // Past the prompt's partial block (which the radix tree never holds), so only this
        // sequence holds it.
        let tail = (!state_at.is_multiple_of(KV_BLOCK)).then(|| {
            let b = seq.table.blocks()[state_at / KV_BLOCK];
            self.pool.incref(b);
            b
        });
        match self.saved.store(tokens, SaveKind::Anchor, tail) {
            Some(s) => {
                if let Some(b) = s.dropped {
                    self.pool.decref(b);
                }
                copies.push((Pool::Seq, seq.slot, Pool::Saved, s.slot));
            }
            None => {
                if let Some(b) = tail {
                    self.pool.decref(b);
                }
            }
        }
    }

    /// Rolls `seq`'s block table back to `n` tokens after a verify step: whole blocks past
    /// the end go back to the pool and to the sequence's reservation (the next verify step
    /// takes them again). Shared prefix blocks are never past the end (drafts are written
    /// after the prompt), so they stay.
    fn trim(&mut self, seq: &mut Seq<'_>, n: usize) {
        let before = seq.table.blocks().len();
        seq.table.truncate(&mut self.pool, n as u32);
        let freed = before - seq.table.blocks().len();
        seq.blocks.truncate(seq.table.blocks().len());
        seq.reserved += freed;
        self.reserved += freed;
    }

    /// `--kv-vmm-keep` (#814): before `need` more blocks are taken, maps the KV pools' lent
    /// tail if the kept blocks cannot hold them. The prefix cache and warm sequences give kept
    /// blocks back first; then the expert cache is lent (as to a short prompt: steps stay
    /// planned as over the unlent cache), its freed VRAM backing the tail. Once no prompt is
    /// read, [`Self::restore_cache`] restores all of the cache but the tail's share, until
    /// [`Self::kv_shrink`].
    ///
    /// `resume` (#1196): a sequence resumes blocks past the kept ones that the last shrink
    /// kept in host memory: the tail is mapped whatever `need`, and those
    /// blocks are copied back into VRAM before a step reads them.
    fn kv_make_room(&mut self, need: usize, resume: bool) -> Result<(), String> {
        let short = |w: &Self| need > w.pool.free_fast_blocks() as usize;
        if !resume {
            if need == 0 || !short(self) {
                return Ok(());
            }
            // Everything they hold is a kept block while the tail is lent, but blocks the
            // last shrink kept in host memory (#1196).
            let host_held = self.kv_lend.as_ref().is_some_and(|l| l.host_held > 0);
            if let Some(radix) = &mut self.radix {
                let n = need - self.pool.free_fast_blocks() as usize;
                radix.evict(&mut self.pool, n as u32);
                // Evicting an entry on the host-held tail frees no kept block: go on, least
                // recently used first, until enough kept blocks are free.
                while host_held
                    && need > self.pool.free_fast_blocks() as usize
                    && radix.evict(&mut self.pool, 1) > 0
                {}
            }
            while short(self) && self.evict_warm() {}
            if short(self) {
                for b in self.saved.drop_tails() {
                    self.pool.decref(b);
                }
            }
            if !short(self) {
                return Ok(());
            }
        }
        let t = Instant::now();
        if !self.lent && self.can_lend {
            self.lend_cache(0, usize::MAX);
            if self.lent {
                self.lent_short = true;
                self.exec.set_pipeline_micro(self.micro);
                self.exec.set_micro_taper(0);
            }
        }
        let lend = self.kv_lend.as_mut().expect("a lent KV tail");
        let host_held = lend.host_held;
        let reserve = self.long_reserve;
        let (bytes, restored, host) = match lend.grow(&self.exec, reserve) {
            Ok(b) => b,
            Err(e) => {
                // Blocks kept in host memory may have lost their bytes: nothing resumes them.
                if host_held > 0 {
                    let keep = lend.keep as u32;
                    self.kv_evict_tail(keep);
                }
                return Err(format!("--kv-vmm-keep: cannot map the KV pools' tail: {e}"));
            }
        };
        tracing::info!(
            ms = t.elapsed().as_secs_f64() * 1e3,
            mib = bytes >> 20,
            from_host_mib = restored >> 20,
            host_mib = host >> 20,
            cache_lent = self.lent,
            resume,
            "KV pools' tail mapped for a sequence past the kept blocks (--kv-vmm-keep)"
        );
        if host > 0 {
            tracing::warn!(
                host_mib = host >> 20,
                "no VRAM left for part of the KV pools' tail: it is served from host memory \
                 (slower attention past the kept blocks) instead of failing the request"
            );
        }
        Ok(())
    }

    /// Evicts every cached prefix and warm sequence holding a block from `keep` on, then the
    /// saved tails if a block there is still in use: the KV pools' tail gives its blocks back
    /// without keeping them (as before #1196).
    fn kv_evict_tail(&mut self, keep: u32) {
        if let Some(radix) = &mut self.radix {
            radix.evict_from(&mut self.pool, keep);
        }
        let past = |w: &Warm| w.table.blocks().iter().any(|b| b.0 >= keep);
        while let Some(i) = self.warm.iter().position(past) {
            let w = self.warm.swap_remove(i);
            w.table.release(&mut self.pool);
            self.free_ring_and_slot(w.ring, w.kv_slot);
        }
        if self.pool.slow_in_use() > 0 {
            for b in self.saved.drop_tails() {
                self.pool.decref(b);
            }
        }
    }

    /// One eviction of a holder of a tail block (`keep` on) for the host copy's budget
    /// ([`kv_vmm::evict_tail_holder`]; warm sequences least recently finished first).
    fn kv_evict_tail_holder(&mut self, keep: u32) -> bool {
        let warm = &mut self.warm;
        let mut freed = None;
        let evicted = kv_vmm::evict_tail_holder(
            &mut self.pool,
            self.radix.as_mut(),
            &mut self.saved,
            keep,
            |pool| {
                let past = |w: &Warm| w.table.blocks().iter().any(|b| b.0 >= keep);
                let Some(i) = (0..warm.len())
                    .filter(|&i| past(&warm[i]))
                    .min_by_key(|&i| warm[i].stamp)
                else {
                    return false;
                };
                let w = warm.swap_remove(i);
                w.table.release(pool);
                freed = Some((w.ring, w.kv_slot));
                true
            },
        );
        if let Some((ring, slot)) = freed {
            self.free_ring_and_slot(ring, slot);
        }
        evicted
    }

    /// `--kv-vmm-keep` (#814): gives the KV pools' tail back to the expert cache once no
    /// decoding sequence holds a block of it (cached prefixes and warm sequences on it go);
    /// the slots a restore left without VRAM for it are refilled (a lent cache's restore
    /// refills them). Nothing may be in flight (`pipe`).
    ///
    /// #1196: the cached prefixes, warm sequences and saved tails on it stay
    /// as far as the pieces holding their blocks fit in the host budget (least recently used
    /// go first); those pieces are copied into host memory and mapped where they were, and
    /// the next [`Self::kv_make_room`] copies them back.
    fn kv_shrink(&mut self, decoding: &[Seq<'_>]) {
        let Some((keep, budget)) = self
            .kv_lend
            .as_ref()
            .filter(|l| l.grown)
            .map(|l| (l.keep as u32, l.host_budget))
        else {
            return;
        };
        if decoding.iter().any(|s| s.blocks.iter().any(|&b| b >= keep)) {
            return;
        }
        let mut host = None;
        if self.pool.slow_in_use() > 0 {
            if budget > 0 {
                host = kv_vmm::fit_host(
                    budget,
                    self,
                    |w| {
                        let held = w.pool.slow_used();
                        w.kv_lend.as_ref().expect("checked above").host_plan(&held)
                    },
                    |w| w.kv_evict_tail_holder(keep),
                );
            }
            if host.is_none() {
                self.kv_evict_tail(keep);
            }
        }
        if host.is_none() && self.pool.slow_in_use() > 0 {
            return;
        }
        // Nothing held past the kept blocks: a lend as without host memory.
        let host = host.filter(|h| h.bytes > 0);
        let t = Instant::now();
        let lend = self.kv_lend.as_mut().expect("checked above");
        let mut shrunk = lend.shrink(&mut self.exec, host.as_ref());
        if host.is_some()
            && let Err(e) = &shrunk
        {
            let e = e.clone();
            // Some pieces may be without their bytes: evict what the tail holds, then lend
            // it as without host memory (or map it all again if something still holds it).
            tracing::warn!(error = %e, "cannot keep the KV tail in host memory; evicting what it holds");
            self.kv_evict_tail(keep);
            let lend = self.kv_lend.as_mut().expect("checked above");
            shrunk = if self.pool.slow_in_use() == 0 {
                lend.shrink(&mut self.exec, None)
            } else {
                lend.grow(&self.exec, 0).and(Err(e))
            };
        }
        match shrunk {
            Ok(bytes) => {
                let host_mib = self.kv_lend.as_ref().map_or(0, |l| l.host_held >> 20);
                tracing::info!(
                    ms = t.elapsed().as_secs_f64() * 1e3,
                    mib = bytes >> 20,
                    host_mib,
                    "KV pools' tail lent to the expert cache again (--kv-vmm-keep); its \
                     blocks still in use moved to host memory (host_mib)"
                );
                // The slots a restore left without VRAM for the tail (a lent cache gets them
                // back with its restore).
                if !self.lent
                    && let Err(e) = self.exec.refill_experts()
                {
                    tracing::warn!(error = %e, "could not refill the expert cache");
                }
            }
            // The next sequence past the kept blocks maps what is unmapped again.
            Err(e) => {
                tracing::warn!(error = %e, "--kv-vmm-keep: cannot give the KV pools' tail back")
            }
        }
    }

    /// Lends the expert cache's VRAM to the long prompt about to be read (see [`LongPrompts`]);
    /// on failure the cache keeps it and lending stays off. `tokens`: the longest prompt's
    /// tokens still to read.
    fn lend_cache(&mut self, tokens: usize, end: usize) {
        let t = Instant::now();
        match self.exec.lend_experts(tokens) {
            Ok(true) => {
                self.lent = true;
                (self.long_cap, self.long_reserve) = self.long_micro_cap(end);
                self.exec.set_pipeline_micro(self.capped(self.long.micro));
                self.exec.set_micro_taper(self.long.taper);
                tracing::info!(
                    ms = t.elapsed().as_secs_f64() * 1e3,
                    "expert cache lent to a long prompt"
                );
            }
            Ok(false) => self.can_lend = false,
            Err(e) => {
                tracing::warn!(error = %e, "could not lend the expert cache; long prompts run without staging");
                self.can_lend = false;
                if let Err(e) = self.exec.restore_experts() {
                    tracing::warn!(error = %e, "could not restore the expert cache");
                }
            }
        }
    }

    /// Takes the expert cache's VRAM back after a long prompt; the slots refill in the
    /// background while steps run on.
    /// `--kv-vmm-keep` (#814): while the KV pools' tail is mapped, the cache's slots that held
    /// its VRAM stay without (whole layers; their experts read from host memory) until
    /// [`Self::kv_shrink`] gives it back.
    fn restore_cache(&mut self) {
        let t = Instant::now();
        let leave = self
            .kv_lend
            .as_ref()
            .filter(|l| l.grown)
            .map_or(&[][..], |l| &l.leave[..]);
        let kv_tail = !leave.is_empty();
        if let Err(e) = self.exec.restore_experts_leaving(leave) {
            tracing::warn!(error = %e, "could not restore the expert cache");
            self.can_lend = false;
        }
        self.lent = false;
        self.lent_short = false;
        self.long_cap = 0;
        self.long_reserve = 0;
        self.exec.set_pipeline_micro(self.micro);
        self.exec.set_micro_taper(0);
        tracing::info!(
            ms = t.elapsed().as_secs_f64() * 1e3,
            kv_tail,
            "expert cache restored, refilling"
        );
    }

    /// #1216: the longest micro-chunk (in whole KV blocks, at least [`MIN_LENT_MICRO`]) whose
    /// long step buffer for a prompt ending `end` tokens in fits each stage GPU's free VRAM
    /// now (just after a lend) beside [`crate::model::core::LENT_SCRATCH`], and the VRAM a KV
    /// tail mapped during the lend must leave for it; `(0, 0)` (no cap) when the full
    /// [`LongPrompts::micro`] fits, nothing was measured, or not even [`MIN_LENT_MICRO`] fits
    /// (the step then spills as before). The buffer grows about linearly with the micro-chunk
    /// and with the past ([`LongStep::at`]).
    /// Smaller micro-chunks stage every expert more often, but a long step's buffer in host
    /// memory ran prompts ~10x slower (an AC922 with 3 GPUs: GLM's staging ring takes nearly
    /// all of a GPU's lent cache).
    fn long_micro_cap(&self, end: usize) -> (usize, usize) {
        let micro = self.long.micro;
        let mut cap = micro;
        for s in &self.long_step {
            let bytes = s.at(end, self.long_past);
            let Some(d) = s.device.filter(|_| bytes > 0) else {
                continue;
            };
            let keep = crate::model::core::LENT_SCRATCH + LENT_MICRO_MARGIN;
            let free = d.memory().0.saturating_sub(keep);
            let fits = (free as u128 * micro as u128 / bytes as u128) as usize;
            cap = cap.min(fits / KV_BLOCK * KV_BLOCK);
        }
        if cap >= micro || cap < MIN_LENT_MICRO {
            return (0, 0);
        }
        let step = (self.long_step.iter())
            .map(|s| (s.at(end, self.long_past) as u128 * cap as u128 / micro as u128) as usize)
            .max()
            .unwrap_or(0);
        let reserve = step + crate::model::core::LENT_SCRATCH + LENT_MICRO_MARGIN;
        tracing::info!(
            micro = cap,
            prompt_end = end.min(self.long_past),
            reserve_mib = reserve >> 20,
            free_mib = ?self
                .long_step
                .iter()
                .map(|s| s.device.map_or(0, |d| d.memory().0 >> 20))
                .collect::<Vec<_>>(),
            "long prompt micro-chunks capped to what the lend left free (step buffers stay in VRAM)"
        );
        (cap, reserve)
    }

    /// `micro` within the current lend's cap ([`Self::long_micro_cap`]).
    fn capped(&self, micro: usize) -> usize {
        if self.long_cap > 0 {
            micro.min(self.long_cap)
        } else {
            micro
        }
    }

    /// One forward pass: a token for every decoding sequence plus prefill chunks within the
    /// token budget; then sampling, prefix-cache inserts and state-slot moves.
    fn step<'t>(
        &mut self,
        decoding: &mut Vec<Seq<'t>>,
        prefilling: &mut Vec<Seq<'t>>,
    ) -> Result<(), String> {
        let batch = decoding.len();
        let mut at = Instant::now();
        let Some(plan) = self.plan_step(decoding, prefilling, batch)? else {
            return Ok(());
        };
        self.clock.lap(host_time::Phase::Plan, &mut at);
        let asked = self.ask_captures(prefilling, &plan.chunks);
        let fed = fed_tokens(decoding, &plan);
        // Which rows are padding, for the MoE layers' expert kernels (`pad_rows`).
        let pad = crate::model::core::pad_rows::register(
            &fed,
            decoding.iter().map(|s| 1 + s.draft.tokens.len()),
        );
        let mut groups = Vec::new();
        if !decoding.is_empty() {
            groups.push(decode_group(decoding, &fed, &plan, 0));
        }
        for &(i, len) in &plan.chunks {
            let s = &prefilling[i];
            let end = s.n_past + len;
            groups.push(Group {
                slot: s.slot,
                seqs: vec![SeqInput {
                    tokens: &s.prompt[s.n_past..end],
                    n_past: s.n_past,
                    blocks: &s.blocks,
                    ring: s.ring,
                    logits: if end == s.prompt.len() {
                        s.active.logits_kind()
                    } else {
                        Logits::None
                    },
                    media: s.media(),
                }],
                verify: false,
            });
        }
        // A prompt's next chunk: its n-gram table rows are read (from the SSD with
        // `--ngram-io direct`) while this step computes.
        for &(i, len) in &plan.chunks {
            let s = &prefilling[i];
            let end = s.n_past + len;
            let next = &s.prompt[end..(end + len).min(s.prompt.len())];
            if !next.is_empty() {
                self.exec.model().prefetch_ngram(&s.prompt[..end], next);
                self.exec.model().prefetch_engram(&s.prompt[..end], next);
            }
        }
        // A step of one prompt's chunk alone builds the next chunk's graph while it computes
        // (`--no-prompt-build-ahead` turns it off): that chunk, as long as this one or the rest.
        let hint = match plan.chunks[..] {
            [(i, len)] if decoding.is_empty() => {
                let s = &prefilling[i];
                let end = s.n_past + len;
                let tokens = len.min(s.prompt.len() - end);
                (tokens > 0).then(|| PromptNext {
                    tokens,
                    logits: if end + tokens == s.prompt.len() {
                        s.active.logits_kind()
                    } else {
                        Logits::None
                    },
                })
            }
            _ => None,
        };
        self.exec.set_prompt_next(hint);
        // A step of one prompt's chunk alone: its captured rows are read back only from the
        // first one the model drafter takes in (DFlash: its ring's, #1133). `history` holds at
        // most the tokens the drafter's request will name (`finish_step` may add the first
        // sampled token), which can only start the rows read earlier.
        let from = match (&self.drafting, &plan.chunks[..]) {
            (Drafting::Model(d), &[(i, len)]) if decoding.is_empty() => {
                let s = &prefilling[i];
                d.rows_read_from(s.history.len(), s.n_past, len)
            }
            _ => 0,
        };
        self.exec.set_hidden_from(from);
        // A decode or verify step: a drafter that gathers its take-in on the device reads the
        // captured rows where the step left them, not a copy of all of them on the host (#1181).
        let on_device = match &self.drafting {
            Drafting::Model(d) if plan.chunks.is_empty() => d.reads_device_rows(),
            _ => None,
        };
        self.exec.set_hidden_on_device(on_device);
        self.clock.lap(host_time::Phase::Pack, &mut at);
        let t_step = Instant::now();
        let out = self.exec.step(&groups).map_err(|e| e.to_string());
        self.settle_captures(asked, out.is_ok());
        let out = out?;
        let step_ms = t_step.elapsed().as_secs_f64() * 1e3;
        self.clock.lap(host_time::Phase::Exec, &mut at);
        drop(groups);
        drop(pad);
        drop(fed);
        self.finish_step(plan, out, step_ms, decoding, prefilling, 0, batch)
    }

    /// A step's plan: drops cancelled prompts, has the model-free drafters propose, grows the
    /// decoding sequences' block tables and picks the prompt chunks within the token budget.
    /// `batch`: the decoding sequences of the whole batch (with groups in flight, of every
    /// group), which drafting is gated on. `None`: nothing to run.
    fn plan_step<'t>(
        &mut self,
        decoding: &mut [Seq<'t>],
        prefilling: &mut Vec<Seq<'t>>,
        batch: usize,
    ) -> Result<Option<StepPlan>, String> {
        // Cancelled prompts stop before their next chunk.
        let mut i = 0;
        while i < prefilling.len() {
            if prefilling[i].active.cancelled() {
                let mut seq = prefilling.remove(i);
                seq.active
                    .finish(&mut self.outbox, FinishReason::Cancelled, None);
                self.release(&mut seq);
            } else {
                i += 1;
            }
        }
        // Every prompt cancelled and nothing decoding (a shutdown with requests in flight):
        // nothing to run.
        if decoding.is_empty() && prefilling.is_empty() {
            return Ok(None);
        }
        // Drafts: model-free drafters propose now (model drafters did after the last step).
        // Every decoding sequence then feeds `n_tok` tokens: its last one, its drafts, padding.
        let mut n_tok = 1;
        let mut depths = adaptive::StepDepth::default();
        let decode_batch = batch;
        if self.spec_k > 0 {
            for seq in decoding.iter_mut() {
                let cap = if seq.drafter.is_some() {
                    seq.adaptive_cap(&self.spec_config, decode_batch, self.max_ctx, &mut depths)
                } else {
                    // Model drafts were selected after the previous target step. Recheck
                    // load now without advancing the probe cadence a second time. The depth
                    // policy's drafters follow `--spec-gate-cost-model`.
                    let gated = !self.spec_config.fixed
                        && if self.depth.is_some() {
                            crate::spec::adaptive::cost_model_gated(decode_batch, &self.spec_config)
                        } else {
                            decode_batch >= self.spec_config.max_batch
                        };
                    draft_cap(
                        if seq.active.constrained() || gated {
                            0
                        } else {
                            self.spec_k
                        },
                        seq.active.max_tokens,
                        seq.active.generated,
                        self.max_ctx,
                        seq.n_past,
                    )
                };
                if let Some(d) = &mut seq.drafter {
                    seq.draft = Draft::default();
                    d.propose(&seq.history, cap, &mut seq.draft.tokens);
                }
                seq.draft.tokens.truncate(cap);
                seq.active.cut_at_trigger(&mut seq.draft.tokens);
                let n = seq.draft.tokens.len();
                seq.draft.probs.truncate(n);
                n_tok = n_tok.max(1 + seq.draft.tokens.len());
            }
        }
        let verify = n_tok > 1;
        if verify {
            n_tok = verify_rows(
                n_tok - 1,
                self.depth.is_some().then_some(self.mtp_k),
                self.spec_k,
                self.spec_config.pad_longest,
            );
        }
        for seq in decoding.iter_mut() {
            self.grow(seq, seq.n_past + n_tok)?;
        }
        // A long prompt borrows the expert cache's VRAM for staging and big steps; a short one
        // (`LongPrompts::short_min_tokens`) borrows it for staging only: its steps keep the
        // chunks and micro-chunks of an unlent cache, so its outputs stay the same.
        let left = |s: &Seq<'_>| s.prompt.len() - s.n_past;
        let long = self.can_lend && prefilling.iter().any(|s| left(s) >= self.long.min_tokens);
        let short = self.can_lend
            && !long
            && self
                .long
                .short_from()
                .is_some_and(|n| prefilling.iter().any(|s| left(s) >= n));
        if (long || short) && !self.lent {
            let end = prefilling.iter().map(|s| s.prompt.len()).max().unwrap_or(0);
            self.lend_cache(prefilling.iter().map(left).max().unwrap_or(0), end);
            self.lent_short = !long;
            // `--kv-vmm-keep` with `--glm-parallel-decode` (#1174): a long prompt that will
            // pass the kept blocks maps the KV pools' tail now, while the VRAM the lend just
            // freed is free. Mapped mid-prompt, the prompt steps' buffers had taken it first
            // (cuMemCreate out of memory on 4 V100s with 16 sequences). The same blocks
            // and bits, mapped a few steps earlier.
            if long
                && self.lent
                && self.exec.model().glm_par.is_some()
                && self.kv_lend.as_ref().is_some_and(|l| !l.grown)
            {
                let need = prefilling
                    .iter()
                    .map(|s| {
                        (s.prompt.len() + 1)
                            .div_ceil(KV_BLOCK)
                            .saturating_sub(s.table.blocks().len())
                    })
                    .sum();
                // #1196: resumed prefixes held in host memory go back to VRAM with the rest.
                let resume = self.kv_lend.as_ref().is_some_and(|l| {
                    l.host_held > 0
                        && prefilling
                            .iter()
                            .any(|s| s.blocks.iter().any(|&b| b >= l.keep as u32))
                });
                self.kv_make_room(need, resume)?;
            }
        }
        // A long prompt turns a short lend into a long one until the cache is restored (the
        // rest of a long prompt keeps big steps).
        if long {
            self.lent_short = false;
        }
        // Plan as a long prompt: big steps over the borrowed cache.
        let big = self.lent && !self.lent_short;
        if self.lent && !big {
            self.exec.set_pipeline_micro(self.micro);
            self.exec.set_micro_taper(0);
        } else if big {
            self.exec.set_micro_taper(self.long.taper);
        }
        let long_batch = if decoding.is_empty() {
            self.long.alone_tokens.max(self.long.batch_tokens)
        } else {
            self.long.batch_tokens
        };
        let max_batch = if big {
            long_batch.max(self.max_batch_tokens)
        } else {
            self.max_batch_tokens
        };
        // A big step over several stages that holds more than the first prompt's chunk (#707):
        // the first prompt leads, as many tokens as it would take alone, run as micro-chunks;
        // the decoding sequences and the other prompts ride ahead of it as one micro-chunk of
        // at most the micro-chunk's tokens (`Executor::set_pipeline_riders`). One graph of all
        // of them (8K prompts: 3.3 GB per 8K tokens) spilled into host memory. Without rings:
        // their chunks are already small.
        let riders = big && self.exec.n_stages() > 1 && self.exec.kv_ring().is_none();
        self.exec.set_pipeline_riders(riders);
        let mut micro = 0;
        if big {
            let left: usize = match prefilling.first() {
                Some(lead) if riders => lead.prompt.len() - lead.n_past,
                _ => prefilling.iter().map(|s| s.prompt.len() - s.n_past).sum(),
            };
            micro = self.capped(
                self.long
                    .micro_for(left.min(max_batch), self.exec.n_stages()),
            );
            self.exec.set_pipeline_micro(micro);
        }
        // What the riders may take besides the decoding sequences' rows.
        let mut ride = micro.saturating_sub(decoding.len() * n_tok);
        let mut budget = max_batch.saturating_sub(decoding.len() * n_tok);
        // A prompt alone over several stages: a long step of micro-chunks (each a graph of at
        // most the micro-chunk), pipelined over the stages.
        let alone =
            self.pipeline_prompt > 0 && !big && decoding.is_empty() && prefilling.len() == 1;
        if alone {
            budget = budget.max(self.pipeline_prompt);
        }
        // The gather attention path materializes `n_kv × chunk × n_head` scores, so its
        // chunks stay at --prefill-chunk; paged attention does not, and a bigger chunk reads
        // the weights (every routed expert, in a MoE) once for more tokens.
        let chunk = if alone && self.exec.paged() {
            // Micro-chunks of at most the ring step (see above).
            self.pipeline_prompt
        } else if self.exec.kv_ring().is_some() {
            // A step of a sequence must fit its rings (with the window before it).
            self.ring_step
        } else if self.exec.paged() {
            usize::MAX
        } else if big {
            // A borrowed cache stages every expert per step: the longer the better.
            long_batch
        } else {
            self.prefill_chunk
        };
        // `--itl-target` / `--ttft-target`: the budget the targets allow, prompts at risk first.
        if self.slo.is_some() {
            budget = self.slo_budget(decoding, prefilling, n_tok, chunk, budget);
        }
        // Prompts always make some progress, even under a full decode batch.
        if budget == 0 && !prefilling.is_empty() {
            budget = KV_BLOCK;
        }
        let mut chunks = Vec::new();
        for (i, seq) in prefilling.iter_mut().enumerate() {
            let take = if riders && !chunks.is_empty() {
                chunk.min(budget).min(ride)
            } else {
                chunk.min(budget)
            };
            if take == 0 {
                break;
            }
            let take = if self.prefill_even {
                even::even_take(
                    seq.prompt.len() - seq.n_past,
                    take,
                    crate::model::core::KV_BLOCK,
                )
            } else {
                take
            };
            let end = self.save_cut(seq, (seq.n_past + take).min(seq.prompt.len()));
            let len = end - seq.n_past;
            if len == 0 {
                continue;
            }
            self.grow(seq, end)?;
            budget -= len;
            // A lead of at most a micro-chunk runs in one graph with the riders: they share it.
            if !chunks.is_empty() || len <= micro {
                ride = ride.saturating_sub(len);
            }
            chunks.push((i, len));
        }
        // The lead last: its micro-chunks follow the riders' (rows in the order of groups,
        // which are the chunks').
        if riders && chunks.len() > 1 {
            chunks.rotate_left(1);
        }
        // `--long-micro-even`: equal micro-chunks over the prompt chunk as planned (only a
        // lone prompt chunk is cut into micro-chunks).
        if big
            && self.long.even
            && let [(_, len)] = chunks[..]
        {
            let micro = self.capped(self.long.even_micro(len, self.exec.n_stages()));
            self.exec.set_pipeline_micro(micro);
        }
        if self.slo.is_some() {
            self.slo_planned(decoding, prefilling, n_tok, &chunks);
        }
        Ok(Some(StepPlan {
            n_tok,
            verify,
            depths,
            chunks,
        }))
    }

    /// After a step (`out`, `step_ms` long) of the plan `plan`: samples, accepts drafts,
    /// commits, drafts the next step's tokens, moves finished prompts to `decoding` and lets
    /// finished sequences go. `decoding` holds state slots `base..` in order; `batch` is as
    /// for [`Self::plan_step`].
    #[allow(clippy::too_many_arguments)]
    fn finish_step<'t>(
        &mut self,
        plan: StepPlan,
        out: crate::model::core::StepOutput,
        step_ms: f64,
        decoding: &mut Vec<Seq<'t>>,
        prefilling: &mut Vec<Seq<'t>>,
        base: usize,
        batch: usize,
    ) -> Result<(), String> {
        let StepPlan {
            n_tok,
            verify,
            mut depths,
            chunks,
        } = plan;
        let mut at = Instant::now();
        let sampling = at;
        // A decode or verify step alone (no prompt chunk) times the target for the depth policy.
        let timed_rows =
            (chunks.is_empty() && !decoding.is_empty()).then(|| decoding.len() * n_tok);
        // Of a verify step's rows, those that carry drafts (the rest pad its shape).
        let drafted: usize = if verify {
            decoding.iter().map(|s| s.draft.tokens.len()).sum()
        } else {
            0
        };
        // The context the decoding sequences read (the depth policy's costs grow with it).
        let span = decoding.iter().map(|s| s.n_past + n_tok).max().unwrap_or(0);

        // Shared with the take-ins left for the next step ([`DeferredTakeIn`]).
        let hidden = std::sync::Arc::new(out.hidden);
        // Or the same rows still on the device ([`ModelDrafter::step_device`]).
        let hidden_dev = out.hidden_dev;
        let mut rows = out.argmax.into_iter().zip(out.rows);
        let mut copies = Vec::new();
        let mut commits = Vec::new();
        let mut done = Vec::new();
        // Per finished sequence: the tokens its recurrent state holds ([`Self::save_end`]).
        let mut done_at = Vec::new();
        let mut step_spec = SpecCounters::default();
        // Per decoding sequence: (first position run, positions kept).
        let mut ran = Vec::with_capacity(decoding.len());
        // Tokens sampled this step, a sequence's last (its stop, or the length cap) included,
        // as for the prompts that end on their first token below.
        let mut sampled = 0u64;
        // Drafters, warm sequences and `on-demand`'s recompute read the tokens so far.
        let keep_history = self.spec_k > 0 || self.warm_on || self.on_demand() || self.end_saves;
        // Whether a row's device argmax is the host argmax of its logits (#695).
        let exact = self.exec.model().hp.arch.argmax_first_of_ties();
        for (i, seq) in decoding.iter_mut().enumerate() {
            let start = seq.n_past;
            let mut appended = Vec::new();
            let mut finished = false;
            if !verify || seq.active.constrained() {
                seq.n_past += 1;
                let (t, mut row) = rows.next().expect("a logits row per decoding sequence");
                if verify {
                    for _ in 1..n_tok {
                        rows.next();
                    }
                }
                let row = seq.active.row(t, &mut row, exact, &self.stop_tokens);
                match seq
                    .active
                    .advance(row, seq.n_past, &self.stop_tokens, &mut self.outbox)
                {
                    Some(t) => {
                        seq.next = t;
                        appended.push(t);
                    }
                    None => finished = true,
                }
                if verify && !finished {
                    commits.push((base + i, 1));
                    self.trim(seq, seq.n_past);
                }
            } else {
                let drafts = std::mem::take(&mut seq.draft);
                let a = &mut seq.active;
                let mut mine: Vec<_> = rows.by_ref().take(n_tok).collect();
                mine.truncate(1 + drafts.tokens.len());
                let mut target = SeqTarget {
                    keep_raw: a.logprobs.is_some(),
                    device_picks: exact && a.logprobs.is_none() && a.sampler.is_greedy(),
                    rows: mine,
                    sampler: &mut a.sampler,
                    raw: Vec::new(),
                    mask_rows: a.min_tokens.saturating_sub(a.generated) as usize,
                    stop_tokens: &self.stop_tokens,
                };
                let (stop, ignore_eos) = (&self.stop_tokens, a.ignore_eos);
                let o = accept::verify(&mut target, &drafts.tokens, &drafts.probs, |t| {
                    !ignore_eos && stop.contains(&t)
                });
                let mut raw = target.raw;
                // Drafts after an accepted end-of-generation token were never verified (the
                // step ended on an accepted draft: no token was picked after it).
                let verified = if o.appended.len() == o.accepted {
                    o.accepted
                } else {
                    drafts.tokens.len()
                };
                if let Some(c) = &mut seq.active.spec {
                    c.record(verified, o.accepted);
                }
                step_spec.record(verified, o.accepted);
                crate::dashboard::hub::record_spec(verified, o.accepted);
                // The n-gram's drafts teach the n-gram's estimates, not the head's.
                if !seq.ngram_verified(drafts.tokens.len(), o.accepted) {
                    seq.adaptive.record(drafts.tokens.len(), o.accepted);
                    seq.adaptive
                        .record_chain(seq.draft_cap, drafts.tokens.len(), o.accepted);
                }
                if let Some(d) = &mut self.depth {
                    d.record_drafts(&drafts.confidence, o.accepted);
                }
                for (j, &t) in o.appended.iter().enumerate() {
                    seq.n_past += 1;
                    let row = StepRow::Picked(t, raw[j].take());
                    match seq
                        .active
                        .advance(row, seq.n_past, &self.stop_tokens, &mut self.outbox)
                    {
                        Some(t) => {
                            seq.next = t;
                            appended.push(t);
                        }
                        None => {
                            finished = true;
                            break;
                        }
                    }
                }
                if !finished {
                    commits.push((base + i, appended.len()));
                    let n_past = seq.n_past;
                    self.trim(seq, n_past);
                }
            }
            if keep_history && !appended.is_empty() {
                seq.history.extend(&appended);
                if let Some(d) = &mut seq.drafter {
                    d.accepted(&appended);
                }
                if let Some(ng) = &mut seq.ngram {
                    ng.appended(&appended);
                }
            }
            if finished {
                // A verify step left the state where the step started (finished sequences
                // are not committed); a decode step advanced it.
                done.push(i);
                done_at.push(if verify { start } else { seq.n_past });
            }
            sampled += appended.len() as u64 + u64::from(finished);
            ran.push((start, appended.len()));
        }
        let mut joined = Vec::new();
        let mut ended = Vec::new();
        // Preempted sequences that joined again (they sampled nothing).
        let mut rejoined = 0;
        for &(i, len) in &chunks {
            let seq = &mut prefilling[i];
            seq.n_past += len;
            if let Some(&(at, kind)) = seq.saves.first()
                && at == seq.n_past
            {
                seq.saves.remove(0);
                if let Some(s) = self.saved.store(&seq.cache_key()[..seq.n_past], kind, None) {
                    if let Some(b) = s.dropped {
                        self.pool.decref(b);
                    }
                    copies.push((Pool::Seq, seq.slot, Pool::Saved, s.slot));
                }
            }
            // A checkpoint of the ring inside the prompt's last part, where the next prompt of
            // a chat that rewrites its last turns leaves this one ([`RingCheckpoints`]).
            if !self.ring_ckpts.is_empty()
                && !self.ring_ckpts.is_single()
                && seq.media.is_none()
                && ring_checkpoints::due(
                    seq.n_past,
                    seq.prompt.len(),
                    self.ring_ckpts.latest(seq.ring),
                    self.ring_ckpt_every,
                    self.ring_ckpt_span,
                )
            {
                self.save_checkpoint(seq.ring, seq.n_past);
            }
            if seq.n_past < seq.prompt.len() {
                continue;
            }
            if let Some(radix) = &mut self.radix {
                radix.insert(&mut self.pool, seq.cache_key(), seq.table.blocks());
            }
            // The ring at the end of the prompt, for the next turn of a chat (which resumes
            // there: the reply's reasoning is not in its prompt) once the reply has gone
            // past what the ring holds.
            if !self.ring_ckpts.is_empty() {
                self.save_checkpoint(seq.ring, seq.n_past);
            }
            let (t, mut row) = rows.next().expect("a logits row per finished prompt");
            if seq.resuming {
                // Read again after a preemption: `next` was sampled before it.
                seq.resuming = false;
                joined.push(i);
                rejoined += 1;
                continue;
            }
            let row = seq.active.row(t, &mut row, exact, &self.stop_tokens);
            match seq
                .active
                .advance(row, seq.n_past, &self.stop_tokens, &mut self.outbox)
            {
                Some(t) => {
                    seq.next = t;
                    joined.push(i);
                    if keep_history {
                        seq.history.push(t);
                        if let Some(d) = &mut seq.drafter {
                            d.accepted(&[t]);
                        }
                    }
                }
                None => ended.push(i),
            }
        }
        // The step's text, to the stream thread before the device calls below.
        self.outbox.flush();
        self.clock.lap(host_time::Phase::Sample, &mut at);
        // GLM's parallel steps: the depth policy counts the step's host sampling as part of the
        // step (a full logits row, as `min_tokens` or a sampler asks for, cost ~3 ms of host time
        // there: a verify step's extra rows are not free).
        let sample_ms = if self.exec.parallel_steps() {
            sampling.elapsed().as_secs_f64() * 1e3
        } else {
            0.0
        };
        // One lock for the step's counters (the live status line's rates and the drafts').
        if let Ok(mut s) = self.stats.lock() {
            s.prompt_tokens += chunks.iter().map(|(_, len)| *len as u64).sum::<u64>();
            s.generated_tokens += sampled + (joined.len() - rejoined) as u64 + ended.len() as u64;
            if verify {
                s.spec.add(&step_spec);
            }
        }
        if verify {
            // Before any slot moves: the verify group's states are those of slots `base..`.
            self.clock.lap(host_time::Phase::Finish, &mut at);
            let t = Instant::now();
            self.exec.commit(&commits).map_err(|e| e.to_string())?;
            let commit_ms = t.elapsed().as_secs_f64() * 1e3;
            self.clock.lap(host_time::Phase::Device, &mut at);
            tracing::debug!(
                step_ms,
                tokens = n_tok,
                seqs = decoding.len(),
                commit_ms,
                "verify step"
            );
            if let (Some(d), Some(rows)) = (&mut self.depth, timed_rows) {
                d.set_context(span);
                let ms = step_ms + commit_ms + sample_ms;
                d.record_step(rows, ms);
                if self.spec_config.row_cost {
                    d.record_fill(rows, drafted, ms);
                }
            }
        } else if let (Some(d), Some(rows)) = (&mut self.depth, timed_rows) {
            d.set_context(span);
            d.record_step(rows, step_ms + sample_ms);
        }
        if matches!(self.drafting, Drafting::Model(_) | Drafting::Mtp(_)) {
            let k = self.spec_k;
            // These drafts will be verified by the next decode batch, including prompts
            // that just finished and excluding sequences that have ended.
            let draft_batch = batch - done.len() + joined.len();
            // Drafts sit past the kept positions: room for them in the block tables (the
            // next step takes those blocks anyway). A drafter reading the step's rows on the
            // device (#1181) reads them first: growing can lend the expert cache, which drops
            // the kept graphs and with them the buffer those rows are in.
            let grow_after = !hidden_dev.is_empty();
            if !grow_after {
                self.grow_for_drafts(decoding, &done, prefilling, &joined, k)?;
            }
            let max_ctx = self.max_ctx;
            let draft_span = (joined.iter())
                .map(|&i| prefilling[i].n_past)
                .fold(span, usize::max);
            // The depth policy's choice for every sequence that drafts now (one verify shape).
            let chosen = self.depth.as_mut().map(|d| {
                if crate::spec::adaptive::cost_model_gated(draft_batch, &self.spec_config) {
                    return (0, self.spec_config.p_min);
                }
                // The head's ceiling (`--spec-k`; `k` may be the n-gram's deeper one).
                let k_max = (self.spec_config.k.min(k)).min(crate::spec::adaptive::MAX_GAIN_DEPTH);
                // `--spec-max-rows`: the verify step's rows within the cap.
                let k_max = crate::spec::adaptive::rows_depth_cap(
                    draft_batch,
                    k_max,
                    self.spec_config.max_rows,
                );
                let gains: Vec<Vec<f32>> = (decoding.iter().enumerate())
                    .filter(|(i, _)| !done.contains(i))
                    .map(|(_, s)| s)
                    .chain(joined.iter().map(|&i| &prefilling[i]))
                    .map(|s| (0..=k_max).map(|j| s.adaptive.gain(j)).collect())
                    .collect();
                // The next step reads about this step's span, and the prompts joining it.
                d.set_context(draft_span);
                let c = d.choose(&gains, k_max);
                (c, d.confidence_floor(&gains, c, self.spec_config.p_min))
            });
            let chosen = chosen.map(|(c, p_min)| {
                self.mtp_k = c;
                if let Drafting::Mtp(d) = &mut self.drafting {
                    d.set_p_min(p_min);
                }
                c
            });
            let n_step: usize = decoding.len() * n_tok + chunks.iter().map(|c| c.1).sum::<usize>();
            let rows_of = |off: usize, n: usize| -> Vec<&[f32]> {
                hidden
                    .iter()
                    .map(|h| {
                        let w = h.len() / n_step;
                        &h[off * w..(off + n) * w]
                    })
                    .collect()
            };
            // The take-ins the step before left (#477): the prompts' first tokens are out, and
            // each goes before its prompt's chunk of this step.
            if let Drafting::Mtp(d) = &mut self.drafting {
                for &(i, _) in &chunks {
                    let seq = &mut prefilling[i];
                    let Some(t) = seq.take_in.take() else {
                        continue;
                    };
                    let started = Instant::now();
                    let mut scratch = Draft::default();
                    let mut one = [DraftRequest {
                        seq: seq.id,
                        tokens: &seq.history,
                        start: t.start,
                        ran: t.ran,
                        hidden: t.rows(),
                        max: 0,
                        out: &mut scratch,
                    }];
                    d.step(&mut self.exec, &mut one, &[&seq.blocks], &[seq.ring])?;
                    tracing::debug!(
                        positions = t.ran,
                        ms = started.elapsed().as_secs_f64() * 1e3,
                        "mtp deferred take-in"
                    );
                }
            }
            // A chunk that leaves at most a KV block of its prompt is taken in by the next step
            // ([`DeferredTakeIn`]); a head reading the target's cache takes nothing in. Not with
            // the head on node rank 1, which keeps only the last step's rows (#560).
            let defer = matches!(self.drafting, Drafting::Mtp(_))
                && !self.exec.mtp_shares_kv()
                && !self.exec.has_remote();
            let mut off = decoding.len() * n_tok;
            let ngram_step = mtp_ngram::NgramStep {
                depth: self.depth.as_ref(),
                batch: draft_batch,
                gated: !self.spec_config.fixed && draft_batch >= self.spec_config.max_batch,
                take_in: !self.exec.mtp_shares_kv(),
                ceiling: self.spec_config.ngram_k().min(k),
                shape: chosen.unwrap_or(self.spec_config.k.min(k)),
                max_ctx,
            };
            // Whether any sequence drafts from the n-gram (its head request only takes in).
            let mut ngram_drafts = false;
            let mut batch = Vec::new();
            // Each request's KV blocks (the MTP head's cache uses the target's layout).
            let mut blocks: Vec<&[u32]> = Vec::new();
            let mut rings: Vec<usize> = Vec::new();
            // Each request's first row among the step's (the head on node rank 1 reads them
            // there, #560).
            let mut offs: Vec<usize> = Vec::new();
            for (i, seq) in decoding.iter_mut().enumerate() {
                if done.contains(&i) {
                    continue;
                }
                let (start, n) = ran[i];
                let max = match chosen {
                    Some(c) => seq.depth_cap(c, max_ctx, &mut depths),
                    None => seq.adaptive_cap(&self.spec_config, draft_batch, max_ctx, &mut depths),
                };
                seq.draft = Draft::default();
                seq.draft_cap = max;
                let max = seq.ngram_drafts(&self.exec, max, &ngram_step);
                ngram_drafts |= seq.draft_ngram.is_some();
                blocks.push(&seq.blocks);
                rings.push(seq.ring);
                offs.push(i * n_tok);
                batch.push(DraftRequest {
                    seq: seq.id,
                    tokens: &seq.history,
                    start,
                    ran: n,
                    hidden: rows_of(i * n_tok, n),
                    max,
                    out: &mut seq.draft,
                });
            }
            let mut chunk_of = vec![None; prefilling.len()];
            for &(i, len) in &chunks {
                chunk_of[i] = Some((off, len));
                off += len;
            }
            for (i, seq) in prefilling.iter_mut().enumerate() {
                let (Some((off, len)), false) = (chunk_of[i], ended.contains(&i)) else {
                    continue;
                };
                if defer && !joined.contains(&i) && seq.prompt.len() - seq.n_past <= KV_BLOCK {
                    seq.draft = Draft::default();
                    seq.draft_cap = 0;
                    seq.take_in = Some(DeferredTakeIn {
                        hidden: hidden.clone(),
                        n_step,
                        off,
                        start: seq.n_past - len,
                        ran: len,
                    });
                    continue;
                }
                let max = match (joined.contains(&i), chosen) {
                    (false, _) => 0,
                    (true, Some(c)) => seq.depth_cap(c, max_ctx, &mut depths),
                    (true, None) => {
                        seq.adaptive_cap(&self.spec_config, draft_batch, max_ctx, &mut depths)
                    }
                };
                seq.draft = Draft::default();
                seq.draft_cap = max;
                // Prompts still prefilling take in only.
                let max = if joined.contains(&i) {
                    seq.ngram_drafts(&self.exec, max, &ngram_step)
                } else {
                    max
                };
                ngram_drafts |= seq.draft_ngram.is_some();
                blocks.push(&seq.blocks);
                rings.push(seq.ring);
                offs.push(off);
                batch.push(DraftRequest {
                    seq: seq.id,
                    tokens: &seq.history,
                    start: seq.n_past - len,
                    ran: len,
                    hidden: rows_of(off, len),
                    max,
                    out: &mut seq.draft,
                });
            }
            if !batch.is_empty() {
                self.clock.lap(host_time::Phase::Finish, &mut at);
                match &mut self.drafting {
                    Drafting::Model(d) => {
                        let t = Instant::now();
                        if hidden_dev.is_empty() {
                            d.step(&mut batch)?;
                        } else {
                            let rows = DeviceHidden {
                                layers: &hidden_dev,
                                first: &offs,
                            };
                            d.step_device(&mut batch, &rows)?;
                        }
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        tracing::debug!(ms, seqs = batch.len(), "model drafts");
                        // A block drafter's pass, timed without a prompt's take-in.
                        if let (Some(depth), true) = (&mut self.depth, chunks.is_empty()) {
                            depth.set_context(draft_span);
                            depth.record_passes(batch.len(), 1, ms);
                        }
                    }
                    Drafting::Mtp(d) => {
                        let t = Instant::now();
                        let passes = if self.exec.has_remote() {
                            d.step_remote(&mut self.exec, &mut batch, &blocks, &rings, &offs)?
                        } else {
                            d.step(&mut self.exec, &mut batch, &blocks, &rings)?
                        };
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        tracing::debug!(ms, passes, "mtp drafts");
                        // Without a prompt's take-in, the passes time the head alone (and
                        // an n-gram drafter's take-in without logits is not a draft pass).
                        if let (Some(depth), true) =
                            (&mut self.depth, chunks.is_empty() && !ngram_drafts)
                        {
                            depth.set_context(draft_span);
                            depth.record_passes(batch.len(), passes, ms);
                        }
                    }
                    _ => unreachable!(),
                }
                self.clock.lap(host_time::Phase::Device, &mut at);
            }
            if grow_after {
                self.grow_for_drafts(decoding, &done, prefilling, &joined, k)?;
            }
        }
        // Read: the next step copies its captured rows into them (#363), unless a take-in left
        // for the next step still holds them ([`DeferredTakeIn`]).
        match std::sync::Arc::try_unwrap(hidden) {
            Ok(hidden) => self.exec.recycle_rows(hidden),
            Err(held) => self.exec.hold_rows(held),
        }

        if let Some(mean) = depths.mean()
            && let Ok(mut stats) = self.stats.lock()
        {
            stats.spec_mean_k = mean;
        }

        // Finished sequences leave; the last decode slot fills each hole.
        for (&i, &state_at) in done.iter().zip(&done_at).rev() {
            let mut seq = decoding.swap_remove(i);
            self.log_done(&seq);
            self.save_end(&seq, state_at, &mut copies);
            self.release(&mut seq);
            if i < decoding.len() {
                copies.push((Pool::Seq, base + decoding.len(), Pool::Seq, base + i));
                decoding[i].slot = base + i;
            }
        }
        let mut moving: Vec<(usize, bool)> = joined
            .iter()
            .map(|&i| (i, true))
            .chain(ended.iter().map(|&i| (i, false)))
            .collect();
        moving.sort_unstable_by_key(|&(i, _)| std::cmp::Reverse(i));
        let mut joining = Vec::new();
        for (i, joins) in moving {
            let mut seq = prefilling.remove(i);
            if joins {
                joining.push(seq);
            } else {
                self.log_done(&seq);
                let state_at = seq.n_past;
                self.save_end(&seq, state_at, &mut copies);
                self.release(&mut seq);
            }
        }
        for mut seq in joining.into_iter().rev() {
            copies.push((Pool::Seq, seq.slot, Pool::Seq, base + decoding.len()));
            seq.slot = base + decoding.len();
            decoding.push(seq);
        }
        self.clock.lap(host_time::Phase::Finish, &mut at);
        let copied = self.exec.copy_states(&copies).map_err(|e| e.to_string());
        if self.state_digests && copied.is_ok() {
            for &(_, _, to, s) in &copies {
                if to == Pool::Saved
                    && let Some(n) = self.saved.prefix_len(s)
                {
                    self.log_state_digest(Pool::Saved, s, n, "saved after the step");
                }
            }
        }
        self.clock.lap(host_time::Phase::Device, &mut at);
        copied
    }

    fn log_done(&self, seq: &Seq<'_>) {
        let (prefill_s, decode_tok_s) = done_timings(
            seq.started,
            seq.active.first_token,
            Instant::now(),
            seq.active.generated,
        );
        // The bytes the engine's steps had to read while the request ran ("model bytes",
        // #391: weights, routed experts, KV), over those steps' time: a batch's steps count once
        // for each request in it. `mem` is device memory against its roofline, `link` the host
        // link (in-place host reads and staging copies).
        let b = self.exec.model_bytes().since(&seq.bytes_at_start);
        let roofline = self.stats.lock().map(|s| s.roofline()).unwrap_or_default();
        let (pre, dec) = bandwidth::PhaseFigures::of_request(&b, roofline);
        let sched = seq.ticket.scheduling();
        // A short hash, never the API key or user itself.
        let share_key = sched.share_key.map(|k| k.to_string());
        tracing::info!(
            prompt_tokens = seq.active.prompt_tokens,
            cached_tokens = seq.active.cached,
            completion_tokens = seq.active.generated,
            queue_ms = seq.active.queue_ms,
            priority = sched.priority,
            share_key = share_key.as_deref(),
            seconds = seq.started.elapsed().as_secs_f64(),
            prefill_seconds = prefill_s,
            decode_tok_s,
            prefill_model_gb = pre.gb,
            prefill_model_gb_s = b.prefill.gb_s(),
            prefill_mem_gb_s = pre.device_gb_s,
            prefill_mem_roofline_pct = pre.roofline_pct,
            prefill_link_gb_s = pre.link_gb_s,
            decode_model_gb = dec.gb,
            decode_model_gb_s = b.decoding().gb_s(),
            decode_mem_gb_s = dec.device_gb_s,
            decode_mem_roofline_pct = dec.roofline_pct,
            decode_link_gb_s = dec.link_gb_s,
            "request done"
        );
        if let Some(ng) = &seq.ngram {
            // `--spec-mtp-ngram`: the verify steps that drafted from the n-gram (part of the
            // request's spec counters).
            tracing::info!(
                ngram_steps = ng.counters.steps,
                ngram_drafted = ng.counters.drafted,
                ngram_accepted = ng.counters.accepted,
                "mtp n-gram drafts"
            );
        }
    }
}

/// What [`Worker::plan_step`] decided for a step.
struct StepPlan {
    /// Tokens each decoding sequence feeds: 1, or a verify step's `1 +` drafts and padding.
    n_tok: usize,
    verify: bool,
    depths: adaptive::StepDepth,
    /// Prompt chunks: (index in `prefilling`, tokens).
    chunks: Vec<(usize, usize)>,
}

/// The rows each decoding sequence feeds in a verify step whose longest draft is `longest`
/// (at least 1). By default one verify shape whatever was drafted, so the step's graph is
/// built once and reused (padding rows cost little next to rebuilding a graph): `1 + spec_k`,
/// or with the depth policy (`depth`: the depth it chose, which changes rarely) `1 +` that
/// depth or the longest draft if deeper. With `pad_longest` (`--spec-pad-longest`) the step
/// pads only to its longest draft: fewer rows of attention, DeltaNet, dense and LM-head work
/// when every draft stopped early, at the price of a verify graph per row count.
/// `--spec-max-rows` where it holds: the depth policy's (an MTP head or block drafter without a
/// fixed depth, as `depth` in the loader), 0 (no cap) otherwise.
fn depth_policy_max_rows(spec: &crate::spec::SpecConfig, drafting: &Drafting) -> usize {
    let policy = matches!(drafting, Drafting::Mtp(_)) || drafting.block_drafts().is_some();
    if policy && !spec.fixed {
        spec.max_rows
    } else {
        0
    }
}

fn verify_rows(longest: usize, depth: Option<usize>, spec_k: usize, pad_longest: bool) -> usize {
    1 + if pad_longest {
        longest
    } else {
        match depth {
            Some(k) => k.max(longest),
            None => spec_k,
        }
    }
}

/// A verify step's tokens per decoding sequence: [next, drafts.., padding up to n_tok]
/// (none for a decode step).
fn fed_tokens(decoding: &[Seq<'_>], plan: &StepPlan) -> Vec<Vec<u32>> {
    if plan.verify {
        decoding
            .iter()
            .map(|s| {
                let mut f = vec![s.next];
                f.extend(&s.draft.tokens);
                f.resize(plan.n_tok, s.next);
                f
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// The decode or verify group of `decoding` (state slots `slot..`), fed `fed`
/// ([`fed_tokens`]).
fn decode_group<'a>(
    decoding: &'a [Seq<'_>],
    fed: &'a [Vec<u32>],
    plan: &StepPlan,
    slot: usize,
) -> Group<'a> {
    let (verify, n_tok) = (plan.verify, plan.n_tok);
    Group {
        slot,
        seqs: decoding
            .iter()
            .enumerate()
            .map(|(i, s)| SeqInput {
                tokens: if verify {
                    &fed[i]
                } else {
                    std::slice::from_ref(&s.next)
                },
                n_past: s.n_past,
                blocks: &s.blocks,
                ring: s.ring,
                // Every row, padding included, so the step's shape (and graph) is
                // the same whatever each sequence drafted.
                logits: if verify {
                    s.active.verify_logits(n_tok)
                } else {
                    s.active.logits_kind()
                },
                media: s.media(),
            })
            .collect(),
        verify,
    }
}

/// A finished request's prompt time (start to its first token) and decode speed (tokens after
/// the first over the time after it), as the server log reports them.
fn done_timings(
    started: Instant,
    first: Option<Instant>,
    now: Instant,
    generated: u32,
) -> (Option<f64>, Option<f64>) {
    let Some(first) = first else {
        return (None, None);
    };
    let decode = now.duration_since(first).as_secs_f64();
    let rate = (generated > 1 && decode > 0.0).then(|| f64::from(generated - 1) / decode);
    (Some(first.duration_since(started).as_secs_f64()), rate)
}

/// A sequence's share of a step's output.
enum StepRow<'a> {
    Logits(&'a mut [f32]),
    /// The argmax, for sequences that sample greedily ([`Active::logits_kind`]).
    Argmax(u32),
    /// A token speculative acceptance picked (and recorded in the sampler), with its row's raw
    /// logits when log probabilities are asked for.
    Picked(u32, Option<Vec<f32>>),
}

/// What a step returns for a sequence that has `generated` of its `min_tokens` tokens:
/// `greedy` (no grammar, greedy sampling, no log probabilities) only its greedy pick, the
/// device argmax ([`Logits::Greedy`]), or below `min_tokens` the pick with the stop tokens
/// masked ([`Logits::Pick`]: the executor reads back only what that needs, often nothing, #1176);
/// otherwise the full row.
fn logits_kind(greedy: bool, generated: u32, min_tokens: u32) -> Logits {
    if !greedy {
        Logits::Last
    } else if generated >= min_tokens {
        Logits::Greedy
    } else {
        Logits::Pick
    }
}

/// What a verify step returns for `n` rows of a sequence whose [`logits_kind`] is `kind`: rows
/// `0..min_tokens - generated` of a [`Logits::Pick`] sequence are below `min_tokens`.
fn verify_logits(kind: Logits, n: usize, generated: u32, min_tokens: u32) -> Logits {
    match kind {
        Logits::Greedy => Logits::FirstGreedy(n),
        Logits::Pick => {
            Logits::FirstPick(n, (min_tokens.saturating_sub(generated) as usize).min(n))
        }
        _ => Logits::First(n),
    }
}

/// One sequence being generated: everything but the model state.
struct Active<'t> {
    sampler: Sampler,
    constraint: Option<Constraint>,
    /// Its text: detokenizer, parsers, stop strings, sends ([`stream`]).
    text: stream::Text<'t>,
    /// Counts it among the live sequences while it exists ([`stream::Outbox::text`]).
    _live: stream::Live,
    logprobs: Option<u32>,
    max_tokens: u32,
    prompt_tokens: u32,
    generated: u32,
    /// When the first token was generated (the prompt was read).
    first_token: Option<Instant>,
    /// `Job::ignore_eos`: a stop token does not end the sequence.
    ignore_eos: bool,
    /// `Job::min_tokens`: stop tokens are masked and stop strings unchecked below it.
    min_tokens: u32,
    /// Prompt tokens served from the prefix cache.
    cached: u32,
    /// Draft and acceptance counts, when speculation is on.
    spec: Option<SpecCounters>,
    max_ctx: usize,
    out: mpsc::UnboundedSender<TokenEvent>,
    cancel: CancellationToken,
    /// When the sequence was admitted (prefill starts).
    admitted: Instant,
    /// Milliseconds it waited in the queue before admission.
    queue_ms: f64,
    /// When its first token was sampled (decode starts).
    first: Option<Instant>,
}

impl<'t> Active<'t> {
    /// Sequence `id`'s state, its text where `outbox` puts it.
    fn new(
        tokenizer: &'t Tokenizer,
        chat_format: &ChatFormat,
        job: &Job,
        max_ctx: usize,
        masks: &Arc<MaskPool>,
        outbox: &mut stream::Outbox,
        id: SeqId,
    ) -> Self {
        let constraint = match (&job.grammar, &job.tool_grammar, chat_format.strict_tools()) {
            (Some(g), _, _) => Some(Constraint::format(g, masks)),
            (None, Some(g), Some((open, _))) => Some(Constraint::tools(g, open, masks)),
            _ => None,
        };
        let spec = stream::TextSpec {
            grammar: constraint.is_some(),
            reasoning: job
                .reasoning
                .map(|inside| chat_format.reasoning_parser(inside)),
            tools: (job.grammar.is_none() && !job.tools.is_empty())
                .then(|| chat_format.tool_parser(&job.tools)),
            stop: StopMatcher::new(job.stop.clone()),
            logprobs: job.sampling.logprobs,
            out: job.out.clone(),
        };
        let (text, live) = outbox.text(tokenizer, id, spec);
        Self {
            sampler: Sampler::new(&job.sampling),
            constraint,
            text,
            _live: live,
            logprobs: job.sampling.logprobs,
            max_tokens: job.max_tokens,
            prompt_tokens: job.prompt.len() as u32,
            generated: 0,
            first_token: None,
            ignore_eos: job.ignore_eos,
            min_tokens: job.min_tokens,
            cached: 0,
            spec: None,
            max_ctx,
            out: job.out.clone(),
            cancel: job.cancel.clone(),
            admitted: Instant::now(),
            queue_ms: job.queued_at.elapsed().as_secs_f64() * 1e3,
            first: None,
        }
    }

    /// What the step must return for this sequence ([`logits_kind`]).
    fn logits_kind(&self) -> Logits {
        logits_kind(
            !self.constrained() && self.sampler.is_greedy() && self.logprobs.is_none(),
            self.generated,
            self.min_tokens,
        )
    }

    /// What [`Self::advance`] samples from, for a step row with device argmax `t` and `logits`
    /// (empty for an argmax-only row). A full row a greedy sequence asked for only to mask its
    /// stop tokens below `min_tokens` is not scanned on the host when the device argmax is
    /// already the pick (#695): `exact` (the model's device argmax is the host's
    /// [`super::sampling::argmax`], [`crate::model::core::Arch::argmax_first_of_ties`]) and
    /// `t` not one of the masked `stop_tokens`. A scan of GLM's ~155K vocabulary cost ~3 ms of
    /// host time on an AC922.
    fn row<'r>(
        &self,
        t: u32,
        logits: &'r mut [f32],
        exact: bool,
        stop_tokens: &[u32],
    ) -> StepRow<'r> {
        if logits.is_empty() || (exact && self.device_pick(t, stop_tokens)) {
            StepRow::Argmax(t)
        } else {
            StepRow::Logits(logits)
        }
    }

    /// Whether device argmax `t` is what [`Self::advance`] would pick from the full row: greedy
    /// sampling with nothing else to apply (no grammar, no log probabilities) and, below
    /// `min_tokens`, `t` not a stop token (the masked row's argmax is then the row's).
    fn device_pick(&self, t: u32, stop_tokens: &[u32]) -> bool {
        self.sampler.is_greedy()
            && self.logprobs.is_none()
            && !self.constrained()
            && (self.generated >= self.min_tokens || !stop_tokens.contains(&t))
    }

    /// Whether a grammar decides the next token: then it needs its full logits row and does
    /// not draft (strict tools only inside a call).
    fn constrained(&self) -> bool {
        self.constraint.as_ref().is_some_and(Constraint::active)
    }

    /// Drafts stop before the token that opens a strict tool call, so the call's tokens are
    /// sampled under its grammar (the opener is then a step's last token).
    fn cut_at_trigger(&self, drafts: &mut Vec<u32>) {
        if let Some(open) = self.constraint.as_ref().and_then(Constraint::trigger)
            && let Some(at) = drafts.iter().position(|&t| t == open)
        {
            drafts.truncate(at);
        }
    }

    /// What a verify step must return for this sequence's `n` rows ([`verify_logits`]).
    fn verify_logits(&self, n: usize) -> Logits {
        verify_logits(self.logits_kind(), n, self.generated, self.min_tokens)
    }

    /// When the request arrived (queued, before admission).
    fn arrived(&self) -> Instant {
        self.admitted
            .checked_sub(std::time::Duration::from_secs_f64(self.queue_ms / 1e3))
            .unwrap_or(self.admitted)
    }

    fn cancelled(&self) -> bool {
        self.cancel.is_cancelled() || self.out.is_closed()
    }

    /// Picks the next token from `logits` and streams its text (through `outbox` when the text
    /// is away). Returns the token to feed back, or `None` once the sequence has finished (and
    /// sent `Finished`). `n_past` is the sequence length so far, for the context limit.
    fn advance(
        &mut self,
        out: StepRow<'_>,
        n_past: usize,
        stop_tokens: &[u32],
        outbox: &mut stream::Outbox,
    ) -> Option<u32> {
        if self.cancelled() {
            self.finish(outbox, FinishReason::Cancelled, None);
            return None;
        }
        let picked = matches!(out, StepRow::Picked(..));
        let (token, raw) = match out {
            StepRow::Logits(logits) => {
                let raw = self.logprobs.map(|_| logits.to_vec());
                // Below `min_tokens` stop tokens cannot be produced: mask them before sampling.
                // `logits_kind` guarantees the full row whenever masking is needed.
                if self.generated < self.min_tokens {
                    for &t in stop_tokens {
                        if let Some(l) = logits.get_mut(t as usize) {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                }
                if let Some(constraint) = &mut self.constraint
                    && constraint.active()
                    && !constraint.mask(logits)
                {
                    tracing::warn!("no token allowed by the grammar");
                    self.end(outbox, FinishReason::Stop, false);
                    return None;
                }
                (self.sampler.sample(logits), raw)
            }
            StepRow::Argmax(token) => (token, None),
            StepRow::Picked(token, raw) => (token, raw),
        };
        self.first.get_or_insert_with(Instant::now);
        if let Some(constraint) = &mut self.constraint {
            // A sequence with a grammar keeps its text on the engine thread for this.
            let reasoning = self.text.inside_reasoning();
            match constraint.accept(token, reasoning) {
                Ok(false) => {}
                Ok(true) => {
                    self.end(outbox, FinishReason::Stop, false);
                    return None;
                }
                Err(e) => {
                    tracing::warn!(%e, "sampled token rejected by the grammar");
                    self.end(outbox, FinishReason::Stop, false);
                    return None;
                }
            }
        }
        if !self.ignore_eos && stop_tokens.contains(&token) {
            // Tool calls made turn the stop into `ToolCalls`.
            self.end(outbox, FinishReason::Stop, true);
            return None;
        }
        if !picked {
            self.sampler.accept(token);
        }
        self.generated += 1;
        self.first_token.get_or_insert_with(Instant::now);

        // Stop strings are not checked until `min_tokens` tokens exist.
        let checking = self.generated >= self.min_tokens;
        if let Some(stop) = self.text.token(outbox, token, raw, checking) {
            self.finish(outbox, FinishReason::StopSequence, Some(stop));
            return None;
        }
        // The fed-back token needs a cache slot: stop when the context is full.
        if self.generated >= self.max_tokens || n_past >= self.max_ctx {
            self.end(outbox, FinishReason::Length, false);
            return None;
        }
        Some(token)
    }

    /// End of generation: what the parsers still hold, then `Finished` with `reason` (with
    /// `tools`, `ToolCalls` instead of `Stop` when the sequence called tools).
    fn end(&mut self, outbox: &mut stream::Outbox, reason: FinishReason, tools: bool) {
        self.close(outbox, true, reason, None, tools);
    }

    /// `Finished` with `reason`, without releasing what the parsers hold.
    fn finish(
        &mut self,
        outbox: &mut stream::Outbox,
        reason: FinishReason,
        stop_sequence: Option<String>,
    ) {
        self.close(outbox, false, reason, stop_sequence, false);
    }

    fn close(
        &mut self,
        outbox: &mut stream::Outbox,
        flush: bool,
        reason: FinishReason,
        stop_sequence: Option<String>,
        tools: bool,
    ) {
        let end = stream::End {
            flush,
            reason,
            stop_sequence,
            usage: self.usage(),
            tools,
        };
        self.text.end(outbox, end);
    }

    /// The sequence failed: an `Error` event ends its stream.
    fn fail(&mut self, outbox: &mut stream::Outbox, e: String) {
        self.text.fail(outbox, e);
    }

    /// `(prefill_ms, decode_ms)`: admission to the first token, then the first token to now.
    /// A sequence that never produced a token reports the whole span as prefill.
    fn timings(&self) -> (f64, f64) {
        let now = Instant::now();
        let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1e3;
        match self.first {
            Some(first) => (ms(self.admitted, first), ms(first, now)),
            None => (ms(self.admitted, now), 0.0),
        }
    }

    /// The usage `Finished` reports now.
    fn usage(&self) -> Usage {
        let (prefill_ms, decode_ms) = self.timings();
        Usage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.generated,
            cached_tokens: self.cached,
            prefill_ms: Some(prefill_ms),
            decode_ms: Some(decode_ms),
            queue_ms: Some(self.queue_ms),
            spec: self.spec,
        }
    }
}

#[cfg(test)]
mod tests;
