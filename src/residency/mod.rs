//! Where a MoE model's routed experts live and how the GPUs reach them.
//!
//! Most of a big MoE model is routed experts (Qwen3.8-Flash-Next UD-Q4_K_XL: 71.7 of ~77 GiB of
//! GPU weights), and a token reads only a few of them. On GPUs that read host memory directly
//! (unified memory: NVLink 2.0 + ATS on the AC922, NVLink-C2C on GH200, one shared memory on
//! GB10, and mapped pinned memory over PCIe elsewhere) the experts can stay in host memory and
//! the VRAM they would take holds the hottest of them instead. This module is that residency
//! layer, written against a small model of the hardware ([`MemoryModel`]) rather than one
//! machine:
//!
//! - **Host arena** ([`cuda::host_arena`]): the experts' bytes in pinned memory on the NUMA node
//!   of the GPUs that read them, filled by that node's CPUs (the loader pins its workers there).
//! - **Tiered tensors** ([`experts::ExpertResidency`]): per MoE layer and expert tensor, one
//!   virtual range `[VRAM slots | the 512 experts' host bytes]`. The graph maps each routed
//!   expert through a per-layer table to a slot (cached) or to its host bytes, and one stock
//!   `mul_mat_id` reads hits from VRAM and misses in place over the link: no copy on the
//!   critical path, no sync, CUDA-graph replayable.
//! - **Policy** ([`policy`]): slot counts per layer from a profile ([`profile`]) and the VRAM
//!   budget; routing heat from every decode step; swaps copied by DMA on a side stream between
//!   steps, never into a slot the graph can read.
//!
//! [`ExpertMode`] picks between all experts in VRAM, host experts read in place, and the VRAM
//! cache in front of them; `auto` decides from the model size, the free VRAM and the platform.

pub mod lookup_table;
pub mod policy;
pub mod profile;
pub mod resident;

pub mod cpu_share;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod ep_balance;
pub mod ep_prefetch;
pub mod experts;
pub mod kv_vmm;
pub mod miss_split;
pub mod split;
pub mod ssd_experts;
#[cfg(feature = "cuda")]
pub mod staging;

use std::fmt::Write as _;

use serde::Serialize;

use crate::ggml::{BufferType, Tensor};
use crate::hw::{HardwareInfo, MemoryKind};

/// The buffer type of the memory a tensor lives in.
pub fn buft_of(t: Tensor<'_>) -> Option<BufferType> {
    // SAFETY: reading the buffer fields of a live tensor.
    unsafe {
        let b = (*t.raw()).buffer;
        if b.is_null() {
            return None;
        }
        std::ptr::NonNull::new(crate::ggml::ffi::ggml_backend_buffer_get_type(b)).map(BufferType)
    }
}

/// The NUMA node of a host arena buffer type (`cuda::host_arena`); `None` for any other.
pub fn arena_node(buft: BufferType) -> Option<u32> {
    #[cfg(feature = "cuda")]
    return cuda::arena_node(buft);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = buft;
        None
    }
}

/// The physical pieces `(bytes, node)` of tensor `t`'s allocation in the host arena `buft`
/// (`cuda::arena_pieces`); `None` when `buft` is no arena.
pub fn arena_pieces(buft: BufferType, t: Tensor<'_>) -> Option<Vec<(usize, u32)>> {
    #[cfg(feature = "cuda")]
    return cuda::arena_pieces(buft, t);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (buft, t);
        None
    }
}

/// The NUMA node of the host arena `t` lives in, if it does.
pub fn arena_node_of(t: Tensor<'_>) -> Option<u32> {
    buft_of(t).and_then(arena_node)
}

/// How the GPUs reach host memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    /// VRAM behind PCIe; host memory only through pinned mappings (RTX cards).
    DiscretePcie,
    /// POWER9 + V100 over NVLink 2.0 with ATS (IBM AC922): coherent host memory, ~68 GB/s per
    /// GPU from its own socket's memory.
    CoherentNvlink,
    /// Grace + Hopper over NVLink-C2C (GH200): coherent, ~450 GB/s each way.
    NvlinkC2c,
    /// One memory for CPU and GPU (GB10, DGX Spark): VRAM is host memory.
    Integrated,
}

/// One device's paths to host memory (GB/s, microseconds). The defaults are nominal figures
/// per platform; [`cuda::probe`] measures the real ones at start-up.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LinkParams {
    /// Kernels reading pinned host memory in place.
    pub in_place_gbs: f64,
    /// DMA copies from pinned host memory to VRAM.
    pub copy_gbs: f64,
    /// Kernels reading pageable (e.g. file-mapped) memory in place; 0 where they cannot.
    pub pageable_gbs: f64,
    /// Kernels reading VRAM.
    pub vram_gbs: f64,
    /// Latency of a host read.
    pub latency_us: f64,
}

impl Platform {
    /// Nominal link figures. AC922: measured on V100s over NVLink 2.0 (pinned in place 68.7,
    /// DMA 67.8, pageable mmap 4-10 GB/s; see `cuda::probe`); the others from the vendors'
    /// figures, to be replaced by the probe.
    pub fn nominal(self) -> LinkParams {
        match self {
            Platform::DiscretePcie => LinkParams {
                in_place_gbs: 25.0,
                copy_gbs: 25.0,
                pageable_gbs: 0.0,
                vram_gbs: 1000.0,
                latency_us: 1.5,
            },
            Platform::CoherentNvlink => LinkParams {
                in_place_gbs: 68.0,
                copy_gbs: 68.0,
                pageable_gbs: 6.0,
                vram_gbs: 850.0,
                latency_us: 1.0,
            },
            Platform::NvlinkC2c => LinkParams {
                in_place_gbs: 400.0,
                copy_gbs: 400.0,
                pageable_gbs: 400.0,
                vram_gbs: 3500.0,
                latency_us: 0.8,
            },
            Platform::Integrated => LinkParams {
                in_place_gbs: 220.0,
                copy_gbs: 220.0,
                pageable_gbs: 220.0,
                vram_gbs: 220.0,
                latency_us: 0.5,
            },
        }
    }

    /// The platform of a machine, from `legion info`'s classification and GPU names.
    pub fn of(hw: &HardwareInfo) -> Self {
        let gh200 = hw.arch == "aarch64"
            && hw
                .gpu
                .as_ref()
                .is_some_and(|r| r.gpus.iter().any(|g| g.name.contains("GH200")));
        match hw.unified_memory.kind {
            MemoryKind::IntegratedShared => Platform::Integrated,
            MemoryKind::CoherentNvlink => Platform::CoherentNvlink,
            MemoryKind::Discrete if gh200 => Platform::NvlinkC2c,
            MemoryKind::Discrete => Platform::DiscretePcie,
        }
    }

    /// Whether VRAM is faster than reading host memory in place by enough for an expert cache
    /// to pay (not on an integrated GPU, where both are the same memory).
    pub fn cache_pays(self) -> bool {
        let l = self.nominal();
        l.vram_gbs > 2.0 * l.in_place_gbs
    }
}

/// Where the routed experts go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertMode {
    /// All in VRAM when they fit; else host memory, with a VRAM cache where it pays.
    #[default]
    Auto,
    /// All in VRAM.
    Device,
    /// Host memory, read in place by the GPU (no cache).
    Host,
    /// Host memory with the hottest experts cached in the free VRAM.
    Cache,
}

impl ExpertMode {
    /// The mode `auto` stands for, given whether the experts fit in VRAM and the platform.
    pub fn resolve(self, experts_fit: bool, platform: Platform) -> ExpertMode {
        match self {
            ExpertMode::Auto if experts_fit => ExpertMode::Device,
            ExpertMode::Auto if platform.cache_pays() => ExpertMode::Cache,
            ExpertMode::Auto => ExpertMode::Host,
            m => m,
        }
    }
}

/// How a step reads the routed experts that are not in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fetch {
    /// The expert matmuls read them in place over the link.
    #[default]
    InPlace,
    /// Small steps (decode, verify) first gather every selected expert into VRAM with a copy
    /// kernel (misses over the link, hits from their slots), then multiply from there, as
    /// Strata's fetch does; big steps read in place.
    Copy,
}

/// [`ResidencyConfig::ring`]'s default: prefill staging buffers per expert kind and device.
pub const PREFILL_RING: usize = 2;
/// Default [`ResidencyConfig::stage_pieces`].
pub const STAGE_PIECES: usize = 0;
/// Default [`ResidencyConfig::stage_min_tokens`].
pub const STAGE_MIN_TOKENS: usize = 128;

/// The ring sizes `--prefill-ring` takes.
pub const PREFILL_RING_RANGE: std::ops::RangeInclusive<usize> = 2..=8;

/// [`ResidencyConfig::profile_seed`]'s default.
pub const PROFILE_SEED: f32 = 16.0;

/// [`ResidencyConfig::cache_reserve`]'s default.
pub const CACHE_RESERVE: u64 = 1 << 30;

/// Settings of the residency layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyConfig {
    pub mode: ExpertMode,
    /// VRAM for the expert cache per device; `None`: what is free after everything else.
    pub cache_bytes: Option<u64>,
    /// VRAM per GPU the expert cache leaves free beside the measured step compute buffers, for
    /// what is allocated after it (ggml's CUDA pools, cuBLAS workspaces); a long prompt's lend
    /// keeps it free too (`--expert-cache-reserve-mib`).
    pub cache_reserve: u64,
    /// The cache also takes the VRAM the measured prompt steps' compute buffers would keep
    /// free, on GPUs whose step buffers can spill into host memory (unified memory); a prompt
    /// step's buffer then takes what VRAM is free and the rest from host memory
    /// (`--expert-cache-fill-prompt-room`, `--no-expert-cache-fill-prompt-room`). `None`: on
    /// for V100s ([`Self::fills_prompt_room`]).
    pub fill_prompt_room: Option<bool>,
    /// Profile to seed the cache from (and to save the learned one to).
    pub profile: Option<std::path::PathBuf>,
    /// The heat the profile's hottest pair starts with (routings): high enough that the first
    /// fill is not swapped out at once, low enough that decode traffic can change it
    /// (`--expert-profile-seed`, for tuning; default [`PROFILE_SEED`]).
    pub profile_seed: f32,
    /// The profile sizes each layer's slots only: the first contents are each layer's lowest
    /// experts, as without a profile, and no heat is seeded (`--expert-profile-split-only`).
    pub profile_split_only: bool,
    /// Per-layer slot counts from a previous run's measured misses ([`miss_split`]): read at
    /// load if the file exists, written on exit (`--expert-split-misses`).
    pub split_misses: Option<std::path::PathBuf>,
    /// Swaps per step and their thresholds; a budget of 0 keeps the first fill.
    pub swaps: policy::SwapRule,
    /// Per-device swap budgets (`--expert-swaps-device`, in the cache's device order),
    /// replacing [`Self::swaps`]' budget on the devices listed; empty: the same everywhere.
    pub device_swaps: Vec<usize>,
    /// The share of each expert's routing heat kept per logged pass (MTP passes included),
    /// in (0, 1].
    pub decay: f32,
    /// The most routing heat an expert gains per logged pass (`--expert-heat-pass-cap`, #814;
    /// 0: no limit, every routed (token, expert) pair adds one).
    pub heat_pass_cap: f32,
    pub fetch: Fetch,
    /// Load host-resident experts into the pinned host-NUMA arena (default). Off, they stay
    /// mapped from the file (page cache), which the GPU also reads in place, but which cannot
    /// sit behind VRAM slots: the cache mode then reads every expert in place.
    pub arena: bool,
    /// Let long prompts borrow the cache's VRAM to stage big steps' uncached experts before
    /// their matmuls (a ring of [`Self::ring`] buffers per device, each its largest staged
    /// layer's experts; see [`experts::ExpertResidency::lend`]). The server turns it on
    /// (`--prefill-staging`).
    pub staging: bool,
    /// Staging buffers per expert kind and device (`--prefill-ring`, at least 2; default
    /// [`PREFILL_RING`]): the copy of a layer's experts can start this many layers ahead.
    pub ring: usize,
    /// Pieces each staged layer's copy is split into (`--prefill-stage-pieces`; 0: one wait
    /// for the whole layer): the fused expert op computes the cached experts at once and each
    /// piece's experts as soon as that piece is in, instead of waiting for the layer's whole
    /// copy before its router (Strata-AC922's prefill waits per streamed expert,
    /// eelgaev/Strata-AC922, MIT). The same outputs.
    pub stage_pieces: usize,
    /// Keep the staging buffers and their staged ranges from one long prompt to the next
    /// (`--prefill-staging-reuse`): made of the lent slots' own allocations, which the slots
    /// map again between long prompts, so a lend only unmaps the slots (no staging buffers
    /// and staged ranges made anew, ~100-150 ms per long prompt on 4 V100s). The same outputs.
    /// Only over whole lent layers (no [`Self::lend_keep`] or [`Self::lend_spread`]).
    pub stage_reuse: bool,
    /// Fewest tokens of a step that stages its experts (`--prefill-stage-min-tokens`, default
    /// [`STAGE_MIN_TOKENS`]); smaller steps read host experts in place over the link.
    pub stage_min_tokens: usize,
    /// While one stage runs a prompt alone, its idle NVLink peer copies half of each staged
    /// layer's experts into it (`--prefill-peer-stage`): a kernel on the peer pulls them from
    /// host memory over its own link and stores them over the pair's NVLink, so the staging
    /// copies come in through two links (Project Maya's two-GPU prefill staging,
    /// mw00/project-maya, MIT). The same bytes in the same buffers, so the same outputs.
    pub stage_peer: bool,
    /// VRAM of expert slots each staging device keeps while its cache is lent
    /// (`--prefill-keep-cache`, 0: lends every slot): whole layers below it keep their cached
    /// experts in their slots, and their prompt chunks copy only the uncached ones in, as
    /// Strata-AC922's prefill streams only the experts its cache misses.
    /// One value per GPU (CUDA0, CUDA1, ...); the last one for the GPUs past the list.
    pub lend_keep: Vec<u64>,
    /// Spread [`Self::lend_keep`] over every staging layer (`--prefill-keep-spread`): each
    /// keeps about the same share of its slots and lends the rest, and the staging buffers
    /// hold only a layer's uncached experts, packed (Strata-AC922's miss-only prefill ring,
    /// eelgaev/Strata-AC922 a3dfde7, MIT), instead of one entry per expert.
    pub lend_spread: bool,
    /// With [`Self::lend_spread`] (or the automatic keep): before a lend for a prompt of at
    /// least this many tokens (0: never), when the slots it keeps hold experts scattered over
    /// the expert ids (decode's swaps put them there), fill them with a window of consecutive
    /// experts instead (`--prefill-keep-repack`), so the prompt's staging copies are a few
    /// runs a layer, not one per gap (#778: after 3 chats, ~120 runs a layer and 0.3-1.3 s
    /// more host time per stage at 123K). The same bytes are read, so the same outputs.
    pub lend_repack: usize,
    /// Per staging GPU (CUDA0, CUDA1, ...): the VRAM a lend must free beside its staging
    /// buffers, for a long prompt's steps. Without a [`Self::lend_keep`], every staging layer
    /// keeps as much of its cache as that leaves, spread as [`Self::lend_spread`] spreads it:
    /// a long prompt reads the experts already cached and copies in only the others. The
    /// engine measures it at load; empty: every slot is lent.
    pub lend_need: Vec<u64>,
    /// Layers from this one read their experts in place during a long prompt and lend all
    /// their slots (an MTP head's; see [`experts::ExpertResidency::stage_below`]); `None`:
    /// every layer stages.
    pub stage_below: Option<usize>,
    /// Whether steps of more than [`experts::LOG_TOKENS`] tokens (long prompts) feed their
    /// routing to the cache: a sample of each step's choices counted on the GPU, so the slots a
    /// long prompt keeps and the ones refilled after it are the experts it routes to most.
    pub learn_long: bool,
    /// The host's share of the host-resident experts in decode steps ([`cpu_share`], opt-in).
    pub cpu_share: cpu_share::CpuShareConfig,
    /// The SSD expert tier ([`ssd_experts`]: experts read from the checkpoint as routed).
    pub ssd: ssd_experts::SsdArgs,
    /// Expert parallelism: more layer entries than the model's layers, each serving a share of
    /// a layer's experts on a GPU of its own (`None`: one entry per layer, any expert). The
    /// engine sets it (GLM's `--glm-parallel-decode`).
    pub ep: Option<EpLayers>,
    /// With [`Self::ep`]: each model layer's own entry computes all of the layer's experts in
    /// the steps its copies take no part in (`--pair-tp`'s prompts), so its prefill staging
    /// copies every uncached expert, not only its owned ones. Off: every entry computes only
    /// its own experts in every step (GLM's parallel decode).
    pub ep_stage_all: bool,
    /// With [`Self::ep`]: every device keeps, per entry, which of its layer's experts their
    /// owners cache ([`experts::TieredLayer::cached_view`]), for GLM's pair balance
    /// (`--glm-pair-balance`, [`ep_balance`]).
    pub ep_cached: bool,
    /// With [`Self::ep`]: every device keeps a second route log that the step graphs fill
    /// with each MoE layer's next-layer router guess, and the cache's periodic report counts
    /// how many misses it and the previous step's choices would have foreseen
    /// (`--glm-prefetch-stats`, [`ep_prefetch`]).
    pub ep_predict: bool,
    /// With [`Self::ep`]: every CUDA device keeps a staging ring for this many of the next
    /// layer's guessed uncached experts per layer, out of the cache's budget
    /// (`--glm-prefetch`, [`ep_prefetch::RingLayout`]); 0 none.
    pub ep_prefetch: usize,
    /// `--deterministic`: with [`Self::ep_cached`], the cache's contents follow the steps alone
    /// ([`Self::settles`]).
    pub deterministic: bool,
}

/// The layer entries of an expert-parallel cache ([`ResidencyConfig::ep`]), by entry index
/// (the model's layers first, then the extra copies).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EpLayers {
    /// The model layer each entry serves.
    pub real: Vec<usize>,
    /// The experts each entry's slots serve (`None`: all).
    pub owned: Vec<Option<std::ops::Range<u32>>>,
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        Self {
            mode: ExpertMode::Auto,
            cache_bytes: None,
            cache_reserve: CACHE_RESERVE,
            fill_prompt_room: None,
            profile: None,
            profile_seed: PROFILE_SEED,
            profile_split_only: false,
            split_misses: None,
            swaps: policy::SwapRule::default(),
            device_swaps: Vec::new(),
            decay: 0.95,
            heat_pass_cap: 0.0,
            fetch: Fetch::InPlace,
            arena: true,
            staging: false,
            ring: PREFILL_RING,
            stage_pieces: STAGE_PIECES,
            stage_reuse: true,
            stage_min_tokens: STAGE_MIN_TOKENS,
            stage_peer: false,
            lend_keep: Vec::new(),
            lend_spread: false,
            lend_repack: 40960,
            lend_need: Vec::new(),
            stage_below: None,
            learn_long: true,
            cpu_share: cpu_share::CpuShareConfig::default(),
            ssd: ssd_experts::SsdArgs::default(),
            ep: None,
            ep_stage_all: false,
            ep_cached: false,
            ep_predict: false,
            ep_prefetch: 0,
            deterministic: false,
        }
    }
}

impl ResidencyConfig {
    /// Whether the cache takes the prompt steps' compute room ([`Self::fill_prompt_room`]):
    /// as given, else on V100s (`volta`: compute capability 7.0). It applies only on GPUs whose
    /// step buffers can spill into host memory, so by default an AC922 (V100s with ATS over
    /// NVLink): its 2-GPU cache gets ~5% more slots (~440 MiB a GPU), the same outputs.
    pub fn fills_prompt_room(&self, volta: bool) -> bool {
        self.fill_prompt_room.unwrap_or(volta)
    }

    /// Whether every step's bookkeeping waits for the cache's own work (#1165): the next
    /// after-step waits for the copies of the swaps and for the slot thread's refill after a
    /// long prompt instead of taking those finished by then, and a long prompt waits for that
    /// refill instead of stopping it, so which experts are cached after each step follows the
    /// steps alone, not their timing. Only with `--deterministic` and [`Self::ep_cached`]
    /// (`--glm-pair-balance`, `--glm-pair-split`): those are the only sums that read which
    /// experts are cached (a GPU pair splits its uncached experts by them); every other path
    /// computes the same bits from VRAM or host memory, so it keeps taking finished copies as
    /// they come.
    pub fn settles(&self) -> bool {
        self.deterministic && self.ep_cached && self.ep.is_some()
    }
}

/// One GPU as the residency layer sees it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeviceMemory {
    /// ggml's CUDA index.
    pub device: usize,
    /// The host NUMA node closest to it.
    pub numa_node: Option<u32>,
    /// Whether the driver allocates pinned host memory on a chosen node through VMM (the host
    /// arena and tiered ranges need it).
    pub host_vmm: bool,
    pub link: LinkParams,
    /// Whether `link` was measured ([`cuda::probe`]) rather than nominal.
    pub measured: bool,
}

/// The machine as the residency layer sees it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryModel {
    pub platform: Platform,
    pub devices: Vec<DeviceMemory>,
}

impl MemoryModel {
    /// Nominal figures for `n_devices` GPUs on `platform`, nodes from the hardware report.
    pub fn nominal(platform: Platform, hw: Option<&HardwareInfo>, n_devices: usize) -> Self {
        let node_of = |i: usize| -> Option<u32> {
            hw.and_then(|h| h.gpu.as_ref())
                .and_then(|r| r.gpus.get(i))
                .and_then(|g| g.numa_node)
        };
        Self {
            platform,
            devices: (0..n_devices)
                .map(|device| DeviceMemory {
                    device,
                    numa_node: node_of(device),
                    host_vmm: false,
                    link: platform.nominal(),
                    measured: false,
                })
                .collect(),
        }
    }

    /// Detects the platform and the CUDA devices' capabilities. With `probe`, measures each
    /// device's link on a small host arena of its node (a few hundred ms per device).
    #[cfg(feature = "cuda")]
    pub fn detect(hw: &HardwareInfo, probe: bool) -> Self {
        let n = crate::ggml::cuda_device_count();
        let mut m = Self::nominal(Platform::of(hw), Some(hw), n);
        for d in &mut m.devices {
            if let Some(c) = cuda::caps(d.device) {
                d.host_vmm = c.host_numa_vmm;
                if c.host_numa_id >= 0 {
                    d.numa_node = Some(c.host_numa_id as u32);
                }
            }
            if probe
                && d.host_vmm
                && let Some(p) = experts::probe_device(d.device, d.numa_node.unwrap_or(0))
            {
                d.link.in_place_gbs = p.in_place_gbs;
                d.link.copy_gbs = p.copy_gbs;
                d.measured = true;
            }
        }
        m
    }

    /// Without CUDA there is nothing to detect: no devices.
    #[cfg(not(feature = "cuda"))]
    pub fn detect(hw: &HardwareInfo, _probe: bool) -> Self {
        Self::nominal(Platform::of(hw), Some(hw), 0)
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Expert residency: platform {:?}", self.platform);
        for d in &self.devices {
            let l = &d.link;
            let _ = writeln!(
                out,
                "  CUDA{}: node {}, host arena {}, host read in place {:.1} GB/s, copy {:.1} GB/s ({})",
                d.device,
                d.numa_node.map_or_else(|| "?".into(), |n| n.to_string()),
                if d.host_vmm {
                    "VMM host-NUMA"
                } else {
                    "unavailable"
                },
                l.in_place_gbs,
                l.copy_gbs,
                if d.measured { "measured" } else { "nominal" },
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_by_fit_and_platform() {
        let p = Platform::CoherentNvlink;
        assert_eq!(ExpertMode::Auto.resolve(true, p), ExpertMode::Device);
        assert_eq!(ExpertMode::Auto.resolve(false, p), ExpertMode::Cache);
        assert_eq!(
            ExpertMode::Auto.resolve(false, Platform::Integrated),
            ExpertMode::Host
        );
        assert_eq!(
            ExpertMode::Auto.resolve(false, Platform::DiscretePcie),
            ExpertMode::Cache
        );
        assert_eq!(ExpertMode::Host.resolve(true, p), ExpertMode::Host);
    }

    #[test]
    fn platform_from_hardware_report() {
        let hw = HardwareInfo::assemble("x86_64", None, Err("none".into()), None);
        assert_eq!(Platform::of(&hw), Platform::DiscretePcie);
        let hw =
            HardwareInfo::assemble("powerpc64", Some("PowerNV 8335-GTH"), Err("x".into()), None);
        assert_eq!(Platform::of(&hw), Platform::CoherentNvlink);
        assert!(Platform::CoherentNvlink.cache_pays());
        assert!(!Platform::Integrated.cache_pays());
        let m = MemoryModel::nominal(Platform::CoherentNvlink, Some(&hw), 2);
        assert_eq!(m.devices.len(), 2);
        assert!(m.render().contains("CUDA1: node ?"), "{}", m.render());
    }
}
