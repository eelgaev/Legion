//! Where a model runs: the devices of its pipeline stages, which layers each stage runs, and
//! where every tensor goes (see [`crate::model::core::Executor`] for stages and
//! [`crate::residency`] for the routed experts).
//!
//! - One stage per device. A layer's weights, KV cache and states live on its stage's device;
//!   the output head and the MTP head on the last one.
//! - With several stages the token embeddings go to pinned host memory every GPU reads (the
//!   first stage embeds the prompt, the MTP head on the last embeds its drafts; a few rows per
//!   token), and so do the RoPE frequency factors every attention layer reads. A model whose
//!   LM head is its embeddings (tied, Gemma 4) also gets a copy of them on the last GPU as
//!   `output.weight` ([`HEAD_COPY`]): the head reads every row each step, which over the host
//!   link cost more than the rest of a decode step.
//! - Routed experts stay in VRAM, or go to a pinned host arena on the GPUs' NUMA node
//!   ([`ExpertMode`]); the per-layer n-gram embedding table always stays mapped on the host.

use std::ops::Range;

use super::{COMPUTE_HEADROOM, LmConfig, LoadError};
use crate::ggml::{Backend, BufferType, Device, DeviceKind, device_by_kind, devices};
use crate::gguf::{GgufFile, TensorInfo};
use crate::model::core::{Hparams, Model, Placement};
use crate::residency::ExpertMode;
use crate::weights::Weights;

pub(super) struct Layout {
    /// Per stage: its device (`None`: the CPU) and the layers it runs.
    pub stages: Vec<(Option<Device>, Range<usize>)>,
    pub mode: ExpertMode,
    /// Per stage: the pinned host arena its layers' experts load into, and its NUMA node.
    pub arenas: Option<Vec<(BufferType, u32)>>,
    /// Pinned host memory every GPU reads (the token embeddings of a multi-stage layout).
    pub shared_host: Option<BufferType>,
    pub host: BufferType,
    /// Tensor parallelism: the devices under the one stage's meta device (else empty).
    pub tp_devices: Vec<Device>,
    /// A model whose LM head is its token embeddings (Gemma 4), over several GPUs: a copy of
    /// the embeddings on the last stage's GPU as `output.weight` ([`HEAD_COPY`]), so decode
    /// steps read the head from VRAM, not from the shared pinned host copy over the link.
    pub head_copy: bool,
    /// A model split over two machines (#562): this one's share; its tensors only load.
    pub node: Option<super::node::Share>,
    /// `--glm-parallel-decode` took: GLM over several GPUs, its experts in one arena split by
    /// the owning GPU's node, cached on the GPUs.
    pub glm_par: bool,
}

/// The tied head's device copy for a multi-stage layout: `(name, source)` for the loader
/// ([`crate::load_io::Options::aliases`]).
pub const HEAD_COPY: (&str, &str) = ("output.weight", "token_embd.weight");

/// Tensors outside the layers that layers of any stage read: in memory every GPU reads when
/// there are several stages.
const SHARED: &[&str] = &["token_embd.weight", "rope_freqs.weight"];

/// The layer index of a tensor name (`blk.<il>.`), if it belongs to a layer.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Split points for `n` stages over layers with `bytes[il]` device bytes each, the last stage
/// also holding `last_extra` bytes: each stage takes layers until it reaches its share of the
/// total. Returns the first layer of every stage after the first.
pub(super) fn auto_split(bytes: &[u64], last_extra: u64, n: usize) -> Vec<usize> {
    let n_layer = bytes.len();
    if n <= 1 || n_layer < n {
        return Vec::new();
    }
    let total: u64 = bytes.iter().sum::<u64>() + last_extra;
    let share = total as f64 / n as f64;
    let mut cuts = Vec::new();
    let mut acc = 0u64;
    for (il, &b) in bytes.iter().enumerate() {
        let stage = cuts.len();
        if stage + 1 < n
            && il > cuts.last().copied().unwrap_or(0)
            && (acc + b / 2) as f64 > share * (stage + 1) as f64
            && n_layer - il >= n - stage - 1
        {
            cuts.push(il);
        }
        acc += b;
    }
    // Too few cuts (tiny models): spread the rest evenly at the end.
    while cuts.len() + 1 < n {
        let first = cuts.last().map_or(1, |&c| c + 1);
        cuts.push(first.min(n_layer - (n - 1 - cuts.len())));
    }
    cuts
}

/// The automatic split of a layout over `n` stages: by bytes ([`auto_split`]), or with
/// `last_extra_layers > 0` (`--split-last-extra`) by layers, the last stage counted as also
/// doing that many layers' worth of other work per step ([`super::pipe_split::mtp_split`],
/// which takes the layers as equal).
pub(super) fn auto_cuts(
    bytes: &[u64],
    last_extra: u64,
    n: usize,
    last_extra_layers: f64,
) -> Vec<usize> {
    if last_extra_layers > 0.0 {
        super::pipe_split::mtp_split(bytes.len(), n, last_extra_layers)
    } else {
        auto_split(bytes, last_extra, n)
    }
}

/// The stages' layer ranges for split points `cuts`.
#[cfg(test)]
pub(super) fn ranges(n_layer: usize, cuts: &[usize]) -> Result<Vec<Range<usize>>, String> {
    ranges_in(0..n_layer, cuts)
}

/// The stages' layer ranges over `span` (a node's share, [`super::node`]) for split points
/// `cuts`.
pub(super) fn ranges_in(span: Range<usize>, cuts: &[usize]) -> Result<Vec<Range<usize>>, String> {
    let mut out = Vec::new();
    let mut start = span.start;
    for &c in cuts.iter().chain(std::iter::once(&span.end)) {
        if c <= start || c > span.end {
            return Err(format!(
                "layer split {cuts:?}: stage boundaries must rise within {}..{}",
                span.start + 1,
                span.end
            ));
        }
        out.push(start..c);
        start = c;
    }
    Ok(out)
}

impl Layout {
    pub fn plan(cfg: &LmConfig, gguf: &GgufFile) -> Result<Self, LoadError> {
        let hp = Hparams::from_gguf(gguf)?;
        let host = device_by_kind(DeviceKind::Cpu)
            .ok_or_else(|| LoadError::Device("no CPU device".into()))?
            .buffer_type();
        // Where the routed experts go: `--host-experts` is `--experts host`.
        let mut mode = match cfg.residency.mode {
            ExpertMode::Auto if cfg.host_experts => ExpertMode::Host,
            m => m,
        };
        let n_stages = cfg.layer_split.len() + 1;
        if cfg.tp > 1 {
            let ple = gguf.tensor("per_layer_token_embd.weight").is_some();
            return Self::plan_tp(cfg, &hp, ple, host);
        }
        // Stages and their helpers take CUDA0, CUDA1, ... by stage index.
        if cfg.first_gpu > 0 {
            return Err(LoadError::Device(
                "a replica on other GPUs than CUDA0 needs --tp 2 or more".into(),
            ));
        }
        let devs: Vec<Option<Device>> = match cfg.device {
            super::DeviceChoice::Cpu => {
                // One memory: only the cache mode means anything (its host-copy backing, for
                // tests of the residency path).
                if mode != ExpertMode::Cache {
                    mode = ExpertMode::Device;
                }
                vec![None; n_stages]
            }
            super::DeviceChoice::Cuda => {
                // Discrete GPUs, else ones sharing memory with the host (GB10 reports itself
                // as integrated).
                let mut gpus: Vec<Device> = devices()
                    .into_iter()
                    .filter(|d| d.kind() == DeviceKind::Gpu)
                    .collect();
                if gpus.is_empty() {
                    gpus = devices()
                        .into_iter()
                        .filter(|d| d.kind() == DeviceKind::IntegratedGpu)
                        .collect();
                }
                // A rank of `--tp-per-gpu`: its own GPU of the pair.
                if let Some(p) = &cfg.tp_pair {
                    gpus.drain(..p.cuda.min(gpus.len()));
                }
                let want = cfg.gpus.max(n_stages);
                if gpus.len() < want {
                    return Err(LoadError::Device(format!(
                        "{want} GPUs asked for, {} found (is legion built with --features cuda?)",
                        gpus.len()
                    )));
                }
                gpus.truncate(want);
                gpus.into_iter().map(Some).collect()
            }
        };
        let n_stages = devs.len();
        let tensors = gguf.tensors();
        // Over two machines: this one's share of the layers, and its tensors only.
        let node = match &cfg.node {
            Some(n) => {
                let free = devs.iter().flatten().map(|d| d.memory().0 as u64).sum();
                Some(super::node::Share::plan(n, &hp, &tensors, free, n_stages)?)
            }
            None => None,
        };
        let owned = |t: &TensorInfo| node.as_ref().is_none_or(|s| s.owns(&t.name, hp.n_layer));
        let span = node.as_ref().map_or(0..hp.n_layer, |s| s.layers.clone());
        // A tied head over several GPUs: its copy on the last one counts there.
        let head_copy = (devs[0].is_some() && n_stages > 1)
            .then(|| crate::weights::alias_info(gguf, HEAD_COPY.0, HEAD_COPY.1))
            .flatten();
        let is_dev = |t: &TensorInfo, host_exps: bool| {
            Model::placement(&t.name, host_exps) == Placement::Device
        };
        let gpu = devs[0].is_some();
        let mut arenas = None;
        let mut split = None;
        if gpu {
            let free: u64 = devs.iter().flatten().map(|d| d.memory().0 as u64).sum();
            let all: u64 = tensors
                .iter()
                .filter(|t| is_dev(t, false) && owned(t))
                .map(|t| t.size)
                .sum();
            let platform = crate::residency::Platform::of(&crate::hw::HardwareInfo::detect());
            let headroom = (COMPUTE_HEADROOM * n_stages) as u64;
            mode = mode.resolve(all + headroom <= free, platform);
            if node.is_some() && mode != ExpertMode::Device {
                return Err(LoadError::Device(format!(
                    "this node's share needs every weight in device memory ({:.1} GiB + \
                     headroom, {:.1} GiB free): give it fewer layers with --node-split",
                    all as f64 / (1u64 << 30) as f64,
                    free as f64 / (1u64 << 30) as f64
                )));
            }
            if mode != ExpertMode::Device {
                if cfg.residency.arena {
                    // GLM's parallel decode: every layer's experts split by the socket of
                    // the GPU owning them (`par::expert_bounds`), one arena for all stages.
                    split = (cfg.glm_parallel_decode
                        && n_stages > 1
                        && crate::model::glm5next::params(&hp).is_some())
                    .then_some(hp.n_expert as usize);
                    arenas = super::expert_arenas(
                        &mut mode,
                        n_stages,
                        &mut split,
                        &cfg.glm_expert_shares,
                    );
                } else if mode == ExpertMode::Cache {
                    tracing::warn!("no expert arena: reading every expert in place from the file");
                    mode = ExpertMode::Host;
                }
            }
        }
        let host_exps = gpu && mode != ExpertMode::Device;

        // The layer split: given, or balanced by the bytes each layer puts on its device. With
        // host experts a layer's routed experts count too: a prompt step stages and multiplies
        // all of them on the layer's GPU, so they are most of its work (and the cache's share
        // follows the layers either way).
        let cuts = if !cfg.layer_split.is_empty() || n_stages == 1 {
            cfg.layer_split.clone()
        } else {
            let mut bytes = vec![0u64; hp.n_layer];
            let mut last_extra = 0u64;
            let counted = |t: &&TensorInfo| {
                is_dev(t, host_exps)
                    || (host_exps && layer_of(&t.name).is_some_and(|l| l < hp.n_layer))
            };
            for t in tensors.iter().filter(counted).filter(|t| owned(t)) {
                match layer_of(&t.name) {
                    Some(il) if il < hp.n_layer => bytes[il] += t.size,
                    Some(_) => last_extra += t.size,
                    None if t.name != "token_embd.weight" => last_extra += t.size,
                    None => {}
                }
            }
            last_extra += head_copy.as_ref().map_or(0, |h| h.size);
            // Over two machines this machine's last stage need not hold the head: by bytes.
            let extra_layers = if node.is_some() {
                0.0
            } else {
                cfg.split_last_extra
            };
            auto_cuts(&bytes[span.clone()], last_extra, n_stages, extra_layers)
                .into_iter()
                .map(|c| c + span.start)
                .collect()
        };
        let ranges = ranges_in(span, &cuts).map_err(LoadError::Device)?;
        if ranges.len() != n_stages {
            return Err(LoadError::Device(format!(
                "a layer split of {} stages for {n_stages} devices",
                ranges.len()
            )));
        }
        // Several GPUs read the embeddings: pinned host memory.
        let shared_host = match &devs[0] {
            Some(d) if n_stages > 1 => d.host_buffer_type(),
            _ => None,
        };
        let me = Self {
            stages: devs.into_iter().zip(ranges).collect(),
            mode,
            arenas,
            shared_host,
            host,
            tp_devices: Vec::new(),
            head_copy: head_copy.is_some(),
            node: node.clone(),
            glm_par: split.is_some() && mode == ExpertMode::Cache,
        };
        // Every device's weights must fit.
        for (i, (d, _)) in me.stages.iter().enumerate() {
            let Some(d) = d else { continue };
            let need: u64 = tensors
                .iter()
                .chain(&head_copy)
                .filter(|t| owned(t) && me.place(t) == d.buffer_type())
                .map(|t| t.size)
                .sum();
            let free = d.memory().0 as u64;
            if need > free {
                return Err(LoadError::Device(format!(
                    "{} needs {:.1} GiB for its weights on stage {i}, {} has {:.1} GiB free",
                    cfg.model_path.display(),
                    need as f64 / (1u64 << 30) as f64,
                    d.description(),
                    free as f64 / (1u64 << 30) as f64,
                )));
            }
        }
        Ok(me)
    }

    /// Tensor parallelism: one stage over a meta device of the first `cfg.tp` GPUs, every
    /// tensor split by [`crate::model::core::tp::megatron_rule`]; experts stay with the
    /// layers (dense models are the target).
    /// `ple`: the model has per-layer embeddings (Gemma 4 E2B/E4B).
    fn plan_tp(
        cfg: &LmConfig,
        hp: &Hparams,
        ple: bool,
        host: BufferType,
    ) -> Result<Self, LoadError> {
        // The cache is sized to what the measured compute buffers leave (`fit_tp_cache`); with
        // a host tier, `--kv-cache-tokens` caps its part in the GPUs.
        let tier = cfg.kv_host_tokens.is_some_and(|t| t > 0);
        if cfg.kv_cache_tokens.is_some() && cfg.kv_memory != super::KvMemory::Managed && !tier {
            return Err(LoadError::Device(
                "--tp sizes the KV cache itself: not with --kv-cache-tokens (except with \
                 --kv-memory managed or --kv-host-tokens)"
                    .into(),
            ));
        }
        // The fit measures the target's and the MTP head's buffers, not a drafter model's
        // (which would also need its own split).
        use crate::spec::SpecKind;
        let model_drafter = matches!(
            cfg.spec.kind,
            SpecKind::Draft | SpecKind::Eagle3 | SpecKind::Dflash | SpecKind::Dspark
        ) || cfg.spec.model_drafter.is_some();
        if model_drafter {
            return Err(LoadError::Device(
                "--tp takes MTP or model-free drafting, not a drafter model \
                 (--spec draft|eagle3|dflash|dspark)"
                    .into(),
            ));
        }
        // The split rule and the cell-gather attention are written for dense Gemma 4 (31B)
        // only: other architectures' layers (DeltaNet, sparse attention) have tensors the rule
        // would split wrongly, and routed experts would be mirrored on every GPU.
        if hp.arch.name() != "gemma4" || hp.n_expert > 0 {
            return Err(LoadError::Device(format!(
                "--tp supports dense Gemma 4 models only, not {}{}",
                hp.arch.name(),
                if hp.n_expert > 0 { " with experts" } else { "" }
            )));
        }
        // The E2B/E4B variants (per-layer embeddings, shared-KV layers) are untested.
        if ple {
            return Err(LoadError::Device(
                "--tp supports dense Gemma 4 without per-layer embeddings (31B) only".into(),
            ));
        }
        // The cache fit counts every cache split among the GPUs; a layer whose KV heads do
        // not divide would stay whole on each.
        let uneven = crate::model::core::tp::uneven_kv_layers(hp, cfg.tp);
        if !uneven.is_empty() {
            return Err(LoadError::Device(format!(
                "--tp {}: the KV heads of layers {uneven:?} do not divide among {} GPUs",
                cfg.tp, cfg.tp
            )));
        }
        if cfg.device != super::DeviceChoice::Cuda {
            return Err(LoadError::Device("--tp needs --device cuda".into()));
        }
        if !cfg.layer_split.is_empty() || cfg.gpus > 1 {
            return Err(LoadError::Device(
                "--tp splits every layer: not with --gpus or --layer-split".into(),
            ));
        }
        // A replica's own group (`--tp-replicas`): CUDA<first_gpu> and the next tp - 1.
        let gpus: Vec<Device> = devices()
            .into_iter()
            .filter(|d| d.kind() == DeviceKind::Gpu)
            .skip(cfg.first_gpu)
            .take(cfg.tp)
            .collect();
        if gpus.len() < cfg.tp {
            return Err(LoadError::Device(format!(
                "--tp {} asks for {} GPUs from CUDA{}, {} found",
                cfg.tp,
                cfg.tp,
                cfg.first_gpu,
                gpus.len()
            )));
        }
        let meta = crate::ggml::meta::meta_device(
            &gpus,
            crate::model::core::tp::megatron_rule(hp, gpus.len()),
        )
        .ok_or_else(|| LoadError::Device("cannot create the tensor-parallel device".into()))?;
        tracing::info!(
            devices = ?gpus.iter().map(|d| d.name()).collect::<Vec<_>>(),
            "tensor parallelism"
        );
        Ok(Self {
            stages: vec![(Some(meta), 0..hp.n_layer)],
            mode: ExpertMode::Device,
            arenas: None,
            shared_host: None,
            host,
            tp_devices: gpus,
            // One stage: the embeddings (and so the tied head) are on the meta device.
            head_copy: false,
            node: None,
            glm_par: false,
        })
    }

    /// The tensors of `gguf` another machine runs (#562): left out of the load.
    pub fn node_skipped(&self, gguf: &GgufFile, n_layer: usize) -> Vec<String> {
        let Some(share) = &self.node else {
            return Vec::new();
        };
        gguf.tensors()
            .into_iter()
            .filter(|t| !share.owns(&t.name, n_layer))
            .map(|t| t.name)
            .collect()
    }

    /// The stage running layer `il` (the MTP head's layer, past the last, runs on the last).
    pub fn stage_of(&self, il: usize) -> usize {
        self.stages
            .iter()
            .position(|(_, r)| r.contains(&il))
            .unwrap_or(self.stages.len() - 1)
    }

    fn device_buft(&self, stage: usize) -> BufferType {
        match &self.stages[stage].0 {
            Some(d) => d.buffer_type(),
            None => self.host,
        }
    }

    /// Which stage's device memory holds tensor `name`; `None`: host memory (routed experts
    /// with host experts, the per-layer token embeddings, and with several stages the tensors
    /// every stage reads). A layer's tensors go to its own stage only, whatever their type:
    /// the FP16 load mode converts a tensor in place of the original, so the trim holds for
    /// converted weights too (cf. Strata-AC922 fd96643, where the converted copies had to be
    /// trimmed separately).
    pub fn device_stage(&self, name: &str) -> Option<usize> {
        // GLM's parallel decode: a GPU's own shard or mirror of a weight goes to that GPU.
        if let Some(d) = crate::model::glm5next::par::copy_device(name) {
            return Some(d.min(self.stages.len() - 1));
        }
        let gpu = self.stages[0].0.is_some();
        let host_exps = gpu && self.mode != ExpertMode::Device;
        match Model::placement(name, host_exps) {
            Placement::Host => None,
            Placement::Device => match layer_of(name) {
                Some(il) => Some(self.stage_of(il)),
                // Read by layers of every stage: memory every GPU reads. (The RoPE frequency
                // factors on the last stage's GPU were an illegal access, Xid 31, on the first
                // stage's: Gemma 4's global layers read them.)
                None if SHARED.contains(&name) => self.shared_host.is_none().then_some(0),
                // The output head and its norms.
                None => Some(self.stages.len() - 1),
            },
        }
    }

    /// Which stage's device memory holds tensor `name` of an MTP head's own GGUF
    /// ([`Model::load_mtp`]): the last stage's, whatever its block number, as the head runs on
    /// the last stage only (`Executor::mtp_pass`). A Gemma 4 assistant numbers its blocks from
    /// 0: by [`Self::device_stage`] they went to the first stage's GPU, and the last stage's
    /// kernels read them there, an illegal access (Xid 31) over two GPUs.
    ///
    /// Its own token embeddings too, over several GPUs: only the head reads them (Gemma 4's
    /// assistant as its tied output head, every row on every draft; its embedding lookups take
    /// the target's table), so they go to the last GPU rather than to the host memory every
    /// GPU reads.
    pub fn mtp_device_stage(&self, name: &str) -> Option<usize> {
        let last = self.stages.len() - 1;
        if name == "token_embd.weight" && self.shared_host.is_some() {
            return Some(last);
        }
        self.device_stage(name).map(|_| last)
    }

    /// Where tensor `info` of an MTP head's own GGUF goes: as [`Self::place`], on the last
    /// stage ([`Self::mtp_device_stage`]).
    pub fn place_mtp(&self, info: &TensorInfo) -> BufferType {
        let last = self.stages.len() - 1;
        let name = info.name.as_str();
        match self.mtp_device_stage(name) {
            Some(stage) => self.device_buft(stage),
            None if name.contains("_exps.") => match &self.arenas {
                Some(a) => a[last].0,
                None => self.host,
            },
            None => self.place(info),
        }
    }

    /// Where tensor `info` goes.
    pub fn place(&self, info: &TensorInfo) -> BufferType {
        let name = info.name.as_str();
        match self.device_stage(name) {
            Some(stage) => self.device_buft(stage),
            None if name.contains("_exps.") => match (&self.arenas, layer_of(name)) {
                (Some(a), Some(il)) => a[self.stage_of(il)].0,
                _ => self.host,
            },
            None => match self.shared_host {
                Some(shared) if SHARED.contains(&name) => shared,
                _ => self.host,
            },
        }
    }

    /// The bytes of `weights` each stage's device holds (by [`Self::device_stage`], converted
    /// tensors at their loaded size).
    pub fn stage_bytes<'w>(&self, weights: impl IntoIterator<Item = &'w Weights>) -> Vec<u64> {
        let mut out = vec![0u64; self.stages.len()];
        for w in weights {
            for (name, bytes) in w.sizes() {
                if let Some(s) = self.device_stage(name) {
                    out[s] += bytes;
                }
            }
        }
        out
    }

    /// [`Self::stage_bytes`] of the target's `weights` and an MTP head's own `mtp` (by
    /// [`Self::mtp_device_stage`]).
    pub fn stage_bytes_with_mtp(&self, weights: &Weights, mtp: Option<&Weights>) -> Vec<u64> {
        let mut out = self.stage_bytes([weights]);
        for (name, bytes) in mtp.into_iter().flat_map(|w| w.sizes()) {
            if let Some(s) = self.mtp_device_stage(name) {
                out[s] += bytes;
            }
        }
        out
    }

    /// Free bytes of stage `stage`'s device for what it splits evenly over its GPUs (the KV
    /// cache), with `headroom` kept on each GPU: a meta device's own figure is the sum, but
    /// every GPU must fit its share.
    pub fn free_for_split(&self, stage: usize, headroom: usize) -> usize {
        if self.tp_devices.is_empty() {
            let free = self.stages[stage].0.map_or(usize::MAX, |d| d.memory().0);
            return free.saturating_sub(headroom);
        }
        let least = self
            .tp_devices
            .iter()
            .map(|d| d.memory().0.saturating_sub(headroom))
            .min()
            .unwrap_or(0);
        least * self.tp_devices.len()
    }

    /// Starts one backend per stage.
    pub fn backends(&self) -> Result<Vec<Backend>, LoadError> {
        self.stages
            .iter()
            .map(|(d, _)| match d {
                Some(d) => Ok(d.init()?),
                None => Ok(Backend::cpu()?),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
