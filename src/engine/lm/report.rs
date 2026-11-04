//! What `legion serve`'s startup report shows of a model: its checkpoint metadata before the
//! load ([`inspect`]), and after it where everything went ([`LoadReport`]), taken from the
//! load's own layout, fit and allocations rather than predicted.

use std::collections::BTreeMap;
use std::ops::Range;

use super::{LmConfig, LoadError};
use crate::engine::sampling_defaults::ModelSampling;
use crate::gguf::GgufFile;
use crate::model::core::Hparams;

/// Checkpoint metadata, read before the weights load.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelFacts {
    pub name: String,
    pub architecture: String,
    /// Bytes per tensor type, by type name.
    pub quantization: Vec<(String, u64)>,
    pub layers: usize,
    pub experts_per_layer: usize,
    pub experts_used: usize,
    pub native_context: usize,
    /// The per-layer n-gram table's bytes, for models with one.
    pub ngram_bytes: Option<u64>,
    pub sampling: ModelSampling,
}

/// Opens checkpoint metadata only. This never maps tensor data or runs a device workload; the
/// layer split is the load's ([`LoadReport::stages`]), not guessed here.
pub fn inspect(cfg: &LmConfig) -> Result<ModelFacts, LoadError> {
    let hf = crate::hf::OpenOptions { fp8: cfg.hf_fp8 };
    let gguf = GgufFile::open_with(&cfg.model_path, &hf)?;
    let hp = Hparams::from_gguf(&gguf)?;
    let mut quantization = BTreeMap::<String, u64>::new();
    let mut ngram_bytes = None;
    for tensor in &gguf.tensors() {
        *quantization.entry(tensor.type_name().into()).or_default() += tensor.size;
        if tensor.name == "per_layer_token_embd.weight" {
            ngram_bytes = Some(tensor.size);
        }
    }
    Ok(ModelFacts {
        name: super::model_id(cfg, &gguf),
        architecture: hp.arch.name().into(),
        quantization: quantization.into_iter().collect(),
        layers: hp.n_layer,
        experts_per_layer: hp.n_expert.max(0) as usize,
        experts_used: hp.n_expert_used.max(0) as usize,
        native_context: hp.n_ctx_train.max(0) as usize,
        ngram_bytes,
        sampling: ModelSampling::from_gguf(&gguf),
    })
}

/// Where a loaded model's memory went.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadReport {
    /// Where the routed experts are: `device`, `host` (read in place) or `cache` (host with
    /// the hottest in VRAM); `None` for dense models.
    pub expert_mode: Option<String>,
    /// Tensor-parallel GPUs (one stage over all of them), else 0.
    pub tp: usize,
    pub stages: Vec<StageMemory>,
    /// Host memory the load took, by NUMA node where it is placed on one.
    pub host: Vec<HostMemory>,
    pub kv: KvSummary,
    pub experts: Option<ExpertSummary>,
    pub ngram: Option<NgramPlacement>,
    /// DeepSeek-V4.1-Flash's Engram tables, read through a lookup table as the n-gram table is.
    pub engram: Option<NgramPlacement>,
}

/// One pipeline stage's device and what it holds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageMemory {
    /// The ggml device name (`CUDA0`), or `CPU`.
    pub device: String,
    /// The device's own description (the GPU's name), if it has one.
    pub description: Option<String>,
    pub layers: Range<usize>,
    /// Weight bytes in this device's memory (converted tensors at their loaded size, the MTP
    /// head with the last stage).
    pub weights: u64,
    /// KV blocks in this device's memory, as tokens, and their bytes.
    pub kv_tokens: usize,
    pub kv_bytes: u64,
    /// The KV host tier's staging slots in this device's memory.
    pub kv_staging_bytes: u64,
    /// Recurrent and speculative states and sliding-window rings.
    pub states_bytes: u64,
    pub expert_layers: usize,
    pub expert_slots: usize,
    pub expert_slot_bytes: u64,
    /// VRAM long prompts borrow from the expert slots to stage experts in.
    pub expert_staging_bytes: Option<u64>,
    /// The step compute buffers measured for the expert cache's budget.
    pub compute_bytes: Option<u64>,
    /// What the KV fit left free for compute buffers.
    pub kv_reserve_bytes: Option<u64>,
    /// What the expert cache's budget left free on top of the measured compute buffers.
    pub expert_reserve_bytes: Option<u64>,
    /// The device's free and total memory once loaded (`None` on the CPU).
    pub free_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    /// Tensor parallelism: each GPU under the stage's meta device, `(name, free, total)`.
    pub tp_devices: Vec<(String, u64, u64)>,
}

/// What a piece of host memory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HostUse {
    /// Routed experts in a pinned arena on the node.
    ExpertArena,
    /// Routed experts mapped from the model file (no arena).
    MappedExperts,
    /// Pinned weights every GPU reads (token embeddings, RoPE factors).
    SharedWeights,
    /// Other host weights.
    HostWeights,
    /// KV blocks past the device's, in pinned memory on the node.
    KvHostTier,
    /// Prefix caching's saved recurrent states, in pinned memory (#441).
    SavedStates,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemory {
    /// `None`: not bound to a node (first touch or the page cache decide).
    pub node: Option<u32>,
    pub kind: HostUse,
    pub bytes: u64,
    /// For the KV host tier: its tokens.
    pub tokens: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvSummary {
    pub ctx_size: usize,
    /// Whether the context is the model's native one (no `--ctx-size`).
    pub native: bool,
    pub device_tokens: usize,
    pub host_tokens: usize,
    pub staging_tokens: usize,
    /// The host tier was sized automatically (no `--kv-host-tokens`).
    pub host_auto: bool,
    /// `f16/f16`, `q8_0/q8_0`, ...: the K and V types.
    pub types: String,
    /// Bytes per token of context over every layer's pool.
    pub bytes_per_token: u64,
    /// Sliding-window rings: their number, and positions each.
    pub rings: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertSummary {
    pub n_expert: usize,
    pub n_used: usize,
    /// MoE layers with a slot cache, and their fewest and most slots.
    pub cached_layers: usize,
    pub slots_min: usize,
    pub slots_max: usize,
    /// How small steps read host experts: `in-place` or `copy`.
    pub fetch: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NgramPlacement {
    pub bytes: u64,
    /// `ram`, `mmap` or `direct` (the resolved tier, never `auto`).
    pub io: String,
    /// The `ram` tier: locked in memory, not just faulted in.
    pub locked: bool,
    pub node: Option<u32>,
    /// The `direct` tier's row cache.
    pub ram_cache_bytes: u64,
    pub reason: String,
}

/// The lower-case name clap gives a value (`q8_0`, `direct`).
pub fn value_name<T: clap::ValueEnum>(v: &T) -> String {
    v.to_possible_value()
        .map_or_else(|| "unknown".into(), |p| p.get_name().to_string())
}

/// Adds `bytes` of `kind` on `node` to `host`, merging with an entry of the same place.
pub(super) fn add_host(host: &mut Vec<HostMemory>, node: Option<u32>, kind: HostUse, bytes: u64) {
    if bytes == 0 {
        return;
    }
    match host.iter_mut().find(|h| h.node == node && h.kind == kind) {
        Some(h) => h.bytes += bytes,
        None => host.push(HostMemory {
            node,
            kind,
            bytes,
            tokens: None,
        }),
    }
}
