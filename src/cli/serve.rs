use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::ValueEnum;
use legion::engine::Engine;
use legion::engine::lm::{DeviceChoice, KvAdmission, LmConfig, LmEngine, LongPrompts, SloTargets};
use legion::engine::mock::{MockConfig, MockEngine};
use legion::engine::queue::SchedulingPolicy;
use legion::engine::sampling_defaults::{DefaultsMode, SamplingLayer};
use legion::engine::switches::Switches;
use legion::hf::Fp8Mode;
use legion::hw::numa_bind::{NumaMode, Placement};
use legion::model::core::{
    Attention, DraftHeadType, Fp8Act, MoeKernel, Nvfp4A4, Nvfp4Act, Nvfp4Mm,
};
use legion::residency::lookup_table::{self, RamBudget, TableIo};
use legion::residency::policy::SwapRule;
use legion::residency::ssd_experts::{ExpertSsd, SsdArgs, SsdPrefetch};
use legion::residency::{ExpertMode, Fetch, ResidencyConfig};
use legion::server::fair_share::FairShare;
use legion::server::registry::{self, RegistryConfig};
use legion::server::{self, AppState, ServerConfig, TlsConfig};
use legion::spec::{SpecConfig, SpecKind};
use legion::weights::WeightsFp16;

use super::logging::LogFormat;
use super::report::{ColorChoice, StartupReport, StatusLineChoice, TermEnv};
use super::{hfmodel, model_path, statusline, version};

mod adaptive_settings;
mod draft_settings;
mod pipe_split;
mod switches;
mod tui;
mod warmup;

#[derive(clap::Args)]
pub struct ServeArgs {
    #[command(flatten)]
    listen: ListenArgs,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    server: ServerArgs,
}

#[derive(clap::Args)]
struct ListenArgs {
    #[arg(long, env = "LEGION_HOST", default_value = "0.0.0.0")]
    host: String,
    #[arg(long, env = "LEGION_PORT", default_value_t = 8090)]
    port: u16,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EngineKind {
    /// A GPU-free engine that echoes the last user message.
    Mock,
    /// Runs the `--model` GGUF (Qwen3.5 / 3.6, dense or MoE; Qwen3.8-Flash-Next (Qwen4
    /// architecture, `qwen4exp`); Nemotron-H).
    Model,
}

/// `--spec`: the speculative-decoding drafter.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SpecArg {
    /// No speculative decoding.
    None,
    /// The model's own multi-token-prediction layers.
    Mtp,
    /// N-gram lookup in the prompt and output.
    Ngram,
    /// Suffix-tree decoding.
    Suffix,
    /// A smaller draft model (`--spec-draft`).
    Draft,
    /// An EAGLE-3 head (`--spec-draft`).
    Eagle3,
    /// A DFlash or DFlash 2 drafter (`--spec-draft`).
    Dflash,
    /// A DSpark drafter (`--spec-draft`).
    Dspark,
}

impl SpecArg {
    fn kind(self) -> SpecKind {
        match self {
            SpecArg::None => SpecKind::None,
            SpecArg::Mtp => SpecKind::Mtp,
            SpecArg::Ngram => SpecKind::Ngram,
            SpecArg::Suffix => SpecKind::Suffix,
            SpecArg::Draft => SpecKind::Draft,
            SpecArg::Eagle3 => SpecKind::Eagle3,
            SpecArg::Dflash => SpecKind::Dflash,
            SpecArg::Dspark => SpecKind::Dspark,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum DeviceArg {
    Cpu,
    /// CUDA GPUs (`--gpus`); the weights must fit their memory, the routed experts of a MoE
    /// model aside (`--experts`).
    Cuda,
}

/// The flags that say which model to load and where to run it; shared by `serve` and `run`.
#[derive(clap::Args)]
pub(super) struct ModelArgs {
    /// GGUF model file, or a directory holding one (`qwen35`, `qwen35moe`, `qwen4exp` for
    /// Qwen3.8-Flash-Next (Qwen4 architecture), `nemotron_h` or `nemotron_h_moe`); for a split model, any of its shards. A Hugging Face
    /// safetensors checkpoint directory also works (Qwen3.5 / 3.6 dense or MoE, BF16/F16/F32;
    /// ModelOpt NVFP4 / FP8; Nemotron-H).
    #[arg(long, env = "LEGION_MODEL")]
    pub model: Option<PathBuf>,
    /// A Hugging Face repo to load from the local cache instead of `--model`:
    /// `<org/name>[@<revision>]` (revision: a branch, tag or commit hash, default `main`).
    /// Resolves offline to the snapshot a previous `huggingface-cli download <org/name>` (or
    /// `hf download`) left in the cache (`$HF_HUB_CACHE`, else `$HUGGINGFACE_HUB_CACHE`, else
    /// `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), and loads it exactly as `--model
    /// <dir>` would: the safetensors checkpoint, or the snapshot's GGUF files (a split
    /// model: the first shard; several quantizations: an error listing them, pass one with
    /// --model). Never downloads, never reads tokens. The model id defaults to `org/name`.
    #[arg(
        long,
        alias = "hf-model",
        value_name = "ORG/NAME[@REV]",
        conflicts_with = "model"
    )]
    pub hfmodel: Option<String>,
    /// A vision projector (mmproj GGUF, `qwen3vl_merger`, e.g. unsloth's `mmproj-F16.gguf`) for
    /// a Qwen3.5 / Qwen3.6 model: requests may then carry images (all three APIs and the web
    /// UI), and the model advertises vision. Its tower runs on the CPU (up to 8 threads),
    /// seconds per image. Default: none, text only.
    #[arg(long, value_name = "GGUF")]
    pub mmproj: Option<PathBuf>,
    /// Where the model runs.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    pub device: DeviceArg,
    /// Max tokens per sequence (prompt + output). Default: the model's native context, in
    /// full: an error if the KV cache cannot hold it (the message lists what fits).
    #[arg(long)]
    pub ctx_size: Option<usize>,
    /// CPU threads for the CPU backend.
    #[arg(long, default_value_t = 8)]
    pub threads: usize,
    /// With --device cuda: keep the routed experts in host memory, mapped from the file, and
    /// let the GPU read them in place. Only for GPUs that share memory with the host (GB10,
    /// GH200): the weights are then not copied at all.
    #[arg(long, env = "LEGION_HOST_EXPERTS")]
    pub host_experts: bool,
    /// The SSD expert tier (GLM-5.3-Flash NVFP4 safetensors): keep as many routed experts as
    /// fit in memory and read the others from the checkpoint as steps route to them. `auto`:
    /// on a GPU sharing the host's memory (GB10) when the experts do not fit; `on`: always
    /// (also on the CPU, with --expert-ssd-cache-gib); `off`: never.
    #[arg(long, value_enum, default_value_t = ExpertSsd::Auto)]
    pub expert_ssd: ExpertSsd,
    /// SSD expert tier: GiB for the expert cache and its two staging buffers (a layer's
    /// experts each). Default: the memory free after the other weights, the KV cache
    /// (--max-seqs contexts of --ctx-size) and the step buffers, less --expert-ssd-reserve-gib.
    #[arg(long, value_name = "GIB")]
    pub expert_ssd_cache_gib: Option<f64>,
    /// SSD expert tier: GiB an automatic cache size leaves free on top of the KV cache and the
    /// step buffers (which already keep the system's share free).
    #[arg(long, value_name = "GIB", default_value_t = 2.0)]
    pub expert_ssd_reserve_gib: f64,
    /// SSD expert tier: reader threads (expert reads in flight). 16 reads a GB10's NVMe ~7%
    /// faster than 8 in decode.
    #[arg(long, default_value_t = 16)]
    pub expert_ssd_readers: usize,
    /// SSD expert tier: read through the page cache (dropping the pages after each read)
    /// instead of around it with O_DIRECT.
    #[arg(long)]
    pub expert_ssd_buffered: bool,
    /// SSD expert tier: read experts ahead of decode and verify steps. `router`: while a MoE
    /// layer computes, read the experts the next layer's router ranks best on this layer's
    /// post-attention state (each token's routed count plus --ssd-prefetch-margin) into the
    /// next layer's least recently used slots; that layer's fetch then waits only for reads it
    /// still needs and drops the others. `router-post`: the same ranked on this layer's output
    /// (after its MoE): more accurate, but later. `router-both`: start from the post-attention
    /// ranking, then let the post-MoE one cancel the reads it does not name that have not
    /// started and add the ones it names. A step's own reads always go before reads ahead.
    /// Changes only when experts are read, never the output. Default `off`. For
    /// GLM-5.3-Flash NVFP4 on a GB10 whose decode waits on the SSD.
    #[arg(long, value_enum, default_value_t = SsdPrefetch::Off)]
    pub ssd_prefetch: SsdPrefetch,
    /// SSD expert tier, --ssd-prefetch router: experts per token read beyond the routed count
    /// the lookahead ranks best (more hits, more wasted reads). Default 0.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub ssd_prefetch_margin: usize,
    /// SSD expert tier, --ssd-prefetch: read ahead at most each token's best N experts (0: the
    /// routed count plus the margin). Below the routed count, only the surest are read ahead:
    /// fewer wasted reads where the SSD is the bottleneck, more reads the step waits for.
    /// Under router-both it caps only the post-attention lookahead (the post-MoE one reads
    /// every rank).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub ssd_prefetch_ranks: usize,
    /// SSD expert tier, --ssd-prefetch router / router-both: queue the next layer's
    /// post-attention lookahead before this layer's expert fetch instead of after it, so its
    /// reads start as soon as the fetch's own reads leave the readers idle. Default off.
    #[arg(long)]
    pub ssd_prefetch_early: bool,
    /// FP8 weights of a safetensors checkpoint: `raw` keeps them as shipped (Legion's FP8
    /// matmul), `bf16` dequantizes them at load as llama.cpp's converter does.
    #[arg(long, value_enum, env = "LEGION_HF_FP8", default_value_t = Fp8Mode::Raw)]
    pub hf_fp8: Fp8Mode,
    /// Activations of NVFP4 matmuls: `a8` keeps them at 8 bits or wider (as W4A16 checkpoints
    /// are meant to run), `a4` lets CUDA quantize unlisted weights' activations to 4 bits on
    /// Blackwell, as llama.cpp does (faster prefill, lower accuracy).
    #[arg(long, value_enum, env = "LEGION_NVFP4_ACT", default_value_t = Nvfp4Act::A8)]
    pub nvfp4_act: Nvfp4Act,
    /// Activations of FP8 matmuls: `a8` quantizes them to E4M3 with the checkpoint's static
    /// input scale (W8A8, as ModelOpt FP8 checkpoints declare and vLLM runs them), `a16` keeps
    /// them F32/BF16 (W8A16: more precise than shipped, about half the prefill GEMM speed).
    /// `a8-dynamic` is `a8`, plus W8A8 with a per-token scale (amax / 448) for FP8 weights
    /// without a static input scale (e.g. unsloth's), in prefill batches of more than 64 tokens
    /// on sm_89+ (GB10, RTX 40/50): faster prefill, logits move slightly; decode stays W8A16.
    /// Combines with `--nvfp4-a4 prefill`. Default `a8`.
    #[arg(long, value_enum, env = "LEGION_FP8_ACT", default_value_t = Fp8Act::A8)]
    pub fp8_act: Fp8Act,
    /// NVFP4 matmuls on CUDA (weights that keep 8-bit or wider activations): `auto` runs
    /// Legion's W4A16 kernels (BF16 activations) for batches of more than 8 tokens and ggml's
    /// MMVQ (Q8_1 activations) below, `legion` W4A16 at every size (one numeric mode), `ggml`
    /// ggml's kernels everywhere.
    #[arg(long, value_enum, env = "LEGION_NVFP4_MM", default_value_t = Nvfp4Mm::Auto)]
    pub nvfp4_mm: Nvfp4Mm,
    /// FP16 load mode (opt-in, outputs shift slightly): `load` converts BF16 weights to F16 as
    /// they load, so GPUs without BF16 tensor cores (V100) multiply them on FP16 ones. A tensor
    /// with a value out of F16's range keeps its type. Adapted from Strata-AC922.
    #[arg(long, value_enum, env = "LEGION_WEIGHTS_FP16", default_value_t = WeightsFp16::Off)]
    pub weights_fp16: WeightsFp16,
    /// With --weights-fp16 load: also the F32 router and gate weights (ffn_gate_inp,
    /// ffn_gate_inp_shexp, ssm_alpha, ssm_beta) as F16; routing shifts slightly.
    #[arg(long, env = "LEGION_WEIGHTS_FP16_GATES")]
    pub weights_fp16_gates: bool,
    /// Load hyper-connection inject weights stored as BF16 (hc_attn_inject, hc_ffn_inject) as
    /// F32, exactly, so prompt steps take the fused mixer norm instead of ggml's chain (whose
    /// BF16 matmul rounds the activations to BF16). For GGUFs like the IQ2_XS
    /// Qwen3.8-Flash-Next: prefill about 8% faster on the GB10, output moves slightly
    /// (teacher-forced KL 2.5e-3 against the weights as stored). Files with F32 inject weights
    /// are unaffected.
    #[arg(long)]
    pub hc_inject_f32: bool,
    /// NVFP4 matmuls with 4-bit activations on the block-scaled FP4 tensor cores (builds for
    /// sm_120a / sm_121a, e.g. the GB10; W4A16 elsewhere): `off`, `experts` (routed experts'
    /// prefill batches), `prefill` (every NVFP4 matmul of batches over 64 tokens), `all`, or
    /// `experts-decode` (routed experts' prefill batches and their decode steps of 9+ tokens,
    /// e.g. 9+ concurrent sequences, on a W4A4 decode kernel; up to 8 tokens they stay on ggml,
    /// which is faster there). Faster prefill (and, with `experts-decode`, slightly faster
    /// concurrent decode); moves the logits of W4A16 checkpoints.
    #[arg(long, value_enum, env = "LEGION_NVFP4_A4", default_value_t = Nvfp4A4::Off)]
    pub nvfp4_a4: Nvfp4A4,
    /// Where a MoE model's routed experts live with --device cuda: `auto` (VRAM if they fit,
    /// else pinned host memory on the GPU's NUMA node with the hottest cached in VRAM, or read
    /// in place where VRAM is the same memory), `device`, `host` (read in place, no cache) or
    /// `cache`.
    #[arg(long, value_enum, env = "LEGION_EXPERTS", default_value_t = ExpertMode::Auto)]
    pub experts: ExpertMode,
    /// VRAM for the expert cache in GiB, per GPU (default: what the weights, the KV cache and
    /// the compute buffers leave). A given size is taken out before the KV cache is sized; one
    /// that does not fit next to the step compute buffers fails at load.
    #[arg(long, env = "LEGION_EXPERT_CACHE_GIB")]
    pub expert_cache_gib: Option<f64>,
    /// VRAM in MiB per GPU that the expert cache leaves free beside the measured step compute
    /// buffers, for ggml's CUDA pools and cuBLAS workspaces; long prompts keep it free too.
    #[arg(long, value_name = "MIB", default_value_t = legion::residency::CACHE_RESERVE >> 20)]
    pub expert_cache_reserve_mib: u64,
    /// The expert cache also takes the VRAM kept free for prompt steps' compute buffers, on
    /// GPUs with unified memory: a prompt step's buffer then spills into host memory (pinned,
    /// on the GPU's NUMA node) as far as VRAM is short. Decode keeps its buffers in VRAM. The
    /// default on V100s (an AC922: ~5% more slots on 2 GPUs, the same outputs) and on GH200s
    /// (unmeasured); other GPUs keep the room unless this is given.
    #[arg(long)]
    pub expert_cache_fill_prompt_room: bool,
    /// Keep the prompt steps' compute room out of the expert cache on V100s and GH200s too (see
    /// --expert-cache-fill-prompt-room): prompt buffers then stay in VRAM. For comparisons.
    #[arg(long, conflicts_with = "expert_cache_fill_prompt_room")]
    pub no_expert_cache_fill_prompt_room: bool,
    /// Expert profile (Strata's `profile.bin` format) that picks the cache's first contents
    /// and per-layer sizes; the routing seen is saved back to it on exit.
    #[arg(long, env = "LEGION_EXPERT_PROFILE")]
    pub expert_profile: Option<PathBuf>,
    /// With --expert-profile: the routing heat the profile's hottest expert starts with
    /// (default 16): higher keeps the first fill in the cache longer, lower lets decode traffic
    /// replace it sooner. For tuning. Replaces LEGION_EXPERT_PROFILE_SEED.
    #[arg(long, value_name = "ROUTINGS", value_parser = non_negative)]
    pub expert_profile_seed: Option<f32>,
    /// With --expert-profile: take only the per-layer slot counts from the profile (its
    /// hottest pairs over all layers), not its first contents or heat; each layer starts with
    /// its lowest experts as without a profile and decode swaps fill it. Default off. For
    /// testing a measured per-layer split on its own. Placement only: outputs do not change.
    #[arg(long, requires = "expert_profile")]
    pub expert_profile_split_only: bool,
    /// Size each layer's expert-cache slots from the hits and misses a previous run measured
    /// (a JSON file the server writes on exit; if it does not exist yet, this run splits as
    /// usual and writes it). Each GPU's slot bytes then go to the layers whose next slot saves
    /// the most misses (misses taken as inversely proportional to slots), so layers that
    /// missed more get more. Default off. Placement only: outputs do not change. For a
    /// workload that repeats, after a warm-up run.
    #[arg(long, value_name = "PATH", conflicts_with = "expert_profile")]
    pub expert_split_misses: Option<PathBuf>,
    /// Experts the cache may copy in per device, per pass (MTP passes included), replacing
    /// colder ones (0: keep the first fill).
    #[arg(long, env = "LEGION_EXPERT_SWAPS", default_value_t = SwapRule::default().budget)]
    pub expert_swaps: usize,
    /// Per-device copy budgets in place of --expert-swaps, comma-separated in pipeline order
    /// (the GPU holding the first layers first; e.g. `48,24`); devices past the list keep
    /// --expert-swaps. Default: none, --expert-swaps everywhere. For tuning layer-split runs
    /// where one stage's host link has more idle time than another's. Placement only: the
    /// cache contents change, outputs do not.
    #[arg(long, value_name = "N,...", value_delimiter = ',')]
    pub expert_swaps_device: Vec<usize>,
    /// A host expert takes the coldest slot holder's slot when its routing heat is higher by
    /// more than this many routings (finite, >= 0)...
    #[arg(long, env = "LEGION_EXPERT_SWAP_MARGIN", value_parser = non_negative,
        default_value_t = SwapRule::default().margin)]
    pub expert_swap_margin: f32,
    /// ...and by more than this fraction of the holder's heat (finite, >= 0; 0: the margin
    /// alone, as in Strata-AC922's adaptive tier).
    #[arg(long, env = "LEGION_EXPERT_SWAP_RATIO", value_parser = non_negative,
        default_value_t = SwapRule::default().ratio)]
    pub expert_swap_ratio: f32,
    /// The share of each expert's routing heat kept per pass (MTP passes included), in (0, 1]:
    /// lower follows the current conversation faster, 1 never forgets.
    #[arg(long, env = "LEGION_EXPERT_HEAT_DECAY", value_parser = heat_decay,
        default_value_t = ResidencyConfig::default().decay)]
    pub expert_heat_decay: f32,
    /// The most routing heat an expert gains in one pass (0, the default: no limit, each routed
    /// (token, expert) pair adds one). 1 counts an expert once per pass however many of the
    /// pass's tokens chose it, so with MTP at one sequence an expert needs routings in two or
    /// more verify steps to be copied into the cache, not one step's burst. Placement only:
    /// the same outputs. At several sequences it also folds different users' routings.
    #[arg(long, value_name = "ROUTINGS", value_parser = non_negative, default_value_t = 0.0)]
    pub expert_heat_pass_cap: f32,
    /// Hybrid decode (opt-in, POWER9 + CUDA only, ignored elsewhere; decode and verify steps of
    /// up to 8 tokens): the host
    /// computes this share of each MoE layer's missed experts (those read from host memory) on
    /// a pinned worker pool per NUMA node while the GPU computes the rest, cutting the bytes the
    /// GPU reads over the link. 0 (the default) is off. Changes outputs slightly (the host
    /// kernels sum in another order than the GPU's). From Strata-AC922's POWER9 CPU expert
    /// share (`--pcie-frac`).
    #[arg(long, value_parser = share_fraction, default_value_t = 0.0)]
    pub cpu_expert_share: f32,
    /// Worker threads of each NUMA node's `--cpu-expert-share` pool (0: `--cpu-expert-smt`
    /// per core of the node, less two cores left to the engine's threads).
    #[arg(long, default_value_t = 0)]
    pub cpu_expert_workers: usize,
    /// Hardware threads per core the `--cpu-expert-share` pool uses (POWER9 has 4; 2 gave the
    /// most kernel throughput per core in Strata-AC922).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=8))]
    pub cpu_expert_smt: u32,
    /// How decode steps read experts that are not cached: `in-place` over the link, or `copy`
    /// (gather the step's experts into VRAM first, then multiply).
    #[arg(long, value_enum, env = "LEGION_EXPERT_FETCH", default_value_t = Fetch::InPlace)]
    pub expert_fetch: Fetch,
    /// Leave host-resident experts mapped from the file (page cache) instead of loading them
    /// into a pinned arena on the GPUs' NUMA node; the VRAM cache needs the arena.
    #[arg(long, env = "LEGION_NO_EXPERT_ARENA")]
    pub no_expert_arena: bool,
    /// With --experts cache: long prompts (see --long-prompt-tokens) borrow the expert
    /// cache's VRAM and stage every layer's experts into it by DMA before their matmuls; the
    /// cache refills in the background afterwards. `false`: prompts read uncached experts in
    /// place.
    #[arg(long, env = "LEGION_PREFILL_STAGING", default_value_t = true, action = clap::ArgAction::Set)]
    pub prefill_staging: bool,
    /// With --prefill-staging: staging buffers per expert kind on each GPU, each a whole
    /// layer's experts (2-8, default 2). A layer's copy can start this many layers ahead of
    /// its matmuls; every buffer past two takes another layer's experts (~1.9 GB for
    /// Qwen3.8-Flash-Next) from the VRAM a long prompt borrows for its compute buffers. On
    /// 2 and 4 V100s, 3 and 4 read prompts no faster than 2 (each copy already finishes under
    /// the previous layer's compute); try 3 where a profile shows the prompt's expert matmuls
    /// waiting on their copies (faster GPUs, slower links). The same outputs. After
    /// Strata-AC922's --prefill-ring (eelgaev/Strata-AC922, a3dfde7, MIT).
    #[arg(long, value_name = "LAYERS", default_value_t = legion::residency::PREFILL_RING,
        value_parser = prefill_ring)]
    pub prefill_ring: usize,
    /// With --prefill-staging: pieces each layer's staging copy is split into (0: the layer's
    /// expert matmuls wait for its whole copy). The fused expert op computes the experts
    /// already cached in VRAM at once and each piece's experts as soon as that piece is in, so
    /// a long prompt's matmuls overlap their layer's copy (Strata-AC922's prefill waits per
    /// streamed expert, eelgaev/Strata-AC922, MIT). The same outputs. Volta only for now:
    /// newer GPUs wait for the whole layer. Off by default: on 4 V100s with the long-prompt
    /// pipeline the layer copies already finish before their matmuls (no gain measured).
    #[arg(long, value_name = "N", default_value_t = legion::residency::STAGE_PIECES)]
    pub prefill_stage_pieces: usize,
    /// With --prefill-staging: fewest tokens of a forward pass that stages its experts into
    /// VRAM (while the cache is lent, see --long-prompt-tokens); smaller passes read host
    /// experts in place over the link.
    #[arg(long, value_name = "N", default_value_t = legion::residency::STAGE_MIN_TOKENS)]
    pub prefill_stage_min_tokens: usize,
    /// With --prefill-staging: while one stage GPU runs a prompt alone, its idle NVLink peer
    /// (the other GPU on its host NUMA node) copies half of each staged layer's experts into it,
    /// pulling them from host memory over its own link (as Project Maya's two-GPU prefill
    /// staging, mw00/project-maya, MIT). On an AC922 the two links bring ~135 GB/s into one GPU
    /// instead of ~70. The same outputs. Takes a stream and the SMs of the idle peer only.
    #[arg(long)]
    pub prefill_peer_stage: bool,
    /// With --prefill-staging: keep the staging buffers from one long prompt to the next. They
    /// are made of the expert cache's own VRAM, which the cache maps again between long
    /// prompts, so a long prompt starts without making them anew (~100-150 ms on 4 V100s).
    /// The same outputs. Only when long prompts lend whole layers (no --prefill-keep-cache).
    #[arg(long, value_name = "BOOL", default_value_t = true, action = clap::ArgAction::Set)]
    pub prefill_staging_reuse: bool,
    /// With --prefill-staging: MiB of expert cache each GPU keeps while a long prompt borrows
    /// its VRAM (0: the prompt borrows all of it). Whole layers' slots up to this size stay
    /// filled, and those layers copy only the experts they do not hold for each prompt chunk,
    /// so a chunk's copies take less time and smaller --long-pipeline-micro chunks stop
    /// waiting on them. Their VRAM is not left for the prompt's compute buffers. One value for
    /// every GPU, or one per GPU (CUDA0,CUDA1,...; the last one for the rest): a GPU holding
    /// more than its layers (the output head) has less to keep.
    /// By default, over several GPUs, every layer keeps as much of its cache as a long
    /// prompt's steps leave (measured at load), spread as --prefill-keep-spread spreads it:
    /// long prompts read the experts already cached and copy in only the others. Greedy
    /// outputs differ from 0's, as with another expert cache size (what the cache holds when
    /// the prompt ends differs). As Strata-AC922's prefill streams only the experts its cache
    /// misses (eelgaev/Strata-AC922, a3dfde7, MIT).
    #[arg(long, value_name = "MIB", value_delimiter = ',')]
    pub prefill_keep_cache: Vec<u64>,
    /// With --prefill-keep-cache: spread what each GPU keeps over all of its layers instead of
    /// keeping whole layers. Every layer keeps the same share of its slots and lends only the
    /// rest, and the staging buffers hold just a layer's uncached experts, packed, not one
    /// entry per expert: smaller buffers leave more of the cache in VRAM, and every layer's
    /// copies shrink alike. As Strata-AC922's miss-only prefill ring (a3dfde7). The same
    /// outputs as keeping whole layers.
    #[arg(long)]
    pub prefill_keep_spread: bool,
    /// With the spread keep (--prefill-keep-spread, or the default over several GPUs): before
    /// a prompt of at least this many tokens, when the slots each layer keeps hold experts
    /// scattered over the expert ids (decode's cache swaps put them there), fill them with a
    /// window of consecutive experts instead, the cached ones moved within VRAM and the rest
    /// copied from host memory once (~65 ms on 4 V100s). The prompt then copies each layer's
    /// other experts in one or two runs, not one copy per gap: after a few chats ~120 copies
    /// a layer and pass, 0.4-1.0 s more at 123K and up to 0.8 s at 64K on 4 V100s (#778);
    /// 8K-32K prompts did not pay for the gaps. The same outputs. The kept slots then hold
    /// the window, not decode's hottest experts, until decode's swaps bring those back (the
    /// next chat decodes ~0.5% slower). 0: never re-pack.
    #[arg(long, value_name = "TOKENS", default_value_t = 40960)]
    pub prefill_keep_repack: usize,
    /// Do not learn from long prompts. By default a long prompt's steps count a sample of
    /// their routing on the GPU (the first 256 tokens of every chunk) into the expert cache's
    /// heat and the saved --expert-profile, so the slots the next long prompt keeps, and the
    /// ones refilled after it, are the experts prompts route to most (as Strata-AC922 ranks its
    /// resident experts by routed frequency). With this flag only steps of up to 256 tokens
    /// count, and a lend refills the experts it took.
    #[arg(long)]
    pub no_prefill_learn: bool,
    /// With --device cuda: split the layers over this many GPUs (CUDA0, CUDA1, ...), one
    /// pipeline stage each.
    #[arg(long, env = "LEGION_GPUS", default_value_t = 1)]
    pub gpus: usize,
    /// With --device cuda: tensor parallelism over this many GPUs (CUDA0, CUDA1, ...): every
    /// layer's heads and FFN rows split across them, one all-reduce after attention and one
    /// after the FFN (dense models; not with --gpus or --layer-split).
    #[arg(long, env = "LEGION_TP", default_value_t = 1)]
    pub tp: usize,
    /// With --tp 2 (dense Gemma 4): run CUDA0 and CUDA1 as two ranks of this process, each
    /// with its own share of the model, executor and launch thread, instead of one ggml meta
    /// device over both. Each GPU keeps its graphs per step shape (decode, MTP verify and
    /// draft) with fused ops and CUDA graphs, as one GPU alone does, and the layers' partial
    /// sums are added over NVLink. Works with --spec mtp. Off by default (the meta device).
    #[arg(long)]
    pub tp_per_gpu: bool,
    /// With --tp: run this many copies of the model in one server, each a --tp group of its
    /// own GPUs (replica r on CUDA r×tp .. r×tp+tp-1; on an AC922 with --tp 2, one NVLink
    /// pair each), its KV cache sized from those GPUs and, with --numa auto, its threads and
    /// host memory on their NUMA node. A new request goes to the replica with the fewest
    /// requests in flight (with prefix caching on, to the one that most likely holds its
    /// prefix) and stays there; each replica decodes exactly as a single --tp server would.
    /// `auto` takes as many groups as the visible GPUs hold. For throughput at high
    /// concurrency when one group's KV cache or batch is the limit. Default 1 (one copy).
    #[arg(long, value_name = "auto|N", default_value = "1")]
    pub tp_replicas: legion::engine::replicas::ReplicaCount,
    /// The first layer of every stage after the first, e.g. `25` for 0-24 | 25-47 (default:
    /// balanced by the bytes each layer puts on its GPU).
    #[arg(long, env = "LEGION_LAYER_SPLIT", value_delimiter = ',')]
    pub layer_split: Vec<usize>,
    /// How attention reads the KV cache: `auto` (paged kernels on a GPU where they support
    /// the model; qwen4exp's sparse attention pages prompt chunks only), `gather` (each
    /// sequence's span gathered, dense attention plus a mask) or `paged`.
    #[arg(long, value_enum, env = "LEGION_ATTENTION", default_value_t = Attention::Auto)]
    pub attention: Attention,
    /// How routed experts run over a batch: `auto` takes Legion's fused expert op on CUDA for
    /// batches of 64+ tokens (with the expert cache: when the step stages its experts into
    /// VRAM); on sm_80+ GPUs it runs on mma.m16n8k16 (where only Volta's MMA exists for it, a GPU
    /// that shares the host's memory takes ggml's MMQ instead); `graph` always ggml's
    /// mul_mat_id path, `fused` always the op (small steps too). The op's prompt logits differ
    /// from MMQ's (F16 vs Q8_1 activations).
    #[arg(long, value_enum, env = "LEGION_MOE_KERNEL", default_value_t = MoeKernel::Auto)]
    pub moe_kernel: MoeKernel,
    /// Routed-expert matvecs of decode steps on CUDA: `legion` (Legion's kernel: many loads in
    /// flight, each expert read once per step) or `ggml` (ggml's mul_mat_id, for comparisons).
    #[arg(long, value_enum, env = "LEGION_EXPERT_MV", default_value_t = ExpertMv::Legion)]
    pub expert_mv: ExpertMv,
    /// On Volta (sm_70): paged prefill attention, and speculative verify steps, through the
    /// m8n8k4 one-pass flash-attention kernel instead of the WMMA kernel (2.5-3x faster per
    /// layer on a V100). Greedy output differs from the default by rounding order (P rounded
    /// to F16 against the running max, one pass, 64-wide MMA chains against WMMA's 16-wide).
    /// F16 caches only; other GPUs ignore it.
    #[arg(long, env = "LEGION_PAGED_PREFILL_M884")]
    pub paged_prefill_m884: bool,
    /// On Turing GPUs without tensor cores (GTX 16xx, MX450, MX550): prompt attention through a
    /// one-pass kernel on FP32 FMAs instead of the WMMA kernel, whose matrix instructions these
    /// GPUs emulate slowly (a GTX 1660 Ti: attention 3.3x faster, an 8K-token Qwen3.5-4B prompt
    /// 15.3 -> 10.7 s). Not the default's bits: greedy output can differ from it (no F16
    /// rounding of Q and P, one pass, a different summation order). F16 caches only; other GPUs
    /// ignore it.
    #[arg(long)]
    pub tu_attn_ffma: bool,
    /// qwen4exp prompt steps: score the sparse-attention indexer's pools with ggml's ops (pool
    /// mask, lightning indexer, bias add) instead of Legion's one-op kernel, which computes the
    /// same bits faster. For comparisons.
    #[arg(long)]
    pub qsa_scores_ggml: bool,
    /// qwen4exp prompt steps: pick the indexer's best pools with ggml's top_k instead of
    /// Legion's one-launch top-k (the same pools, but for scores tied exactly at the threshold).
    /// For comparisons.
    #[arg(long)]
    pub qsa_topk_ggml: bool,
    /// qwen4exp: the hyper-connection mixer's combine-and-norm through the generic kernel
    /// instead of the one holding the values in registers (sm_80+, and V100s; the same bits).
    /// For comparisons.
    #[arg(long)]
    pub hc_post_norm_generic: bool,
    /// Integrated GPUs (a DGX Spark's GB10): keep the model files' pages in the page cache
    /// through the load. By default the load drops them before it allocates the weights and
    /// again before the KV cache and step buffers, so those get freshly freed, unfragmented
    /// memory; with the cache left in, about one server start in three ran every prompt kernel
    /// 5-15% slower for its whole life. Discrete GPUs never drop. The same output either way;
    /// for comparisons.
    #[arg(long)]
    pub keep_load_cache: bool,
    /// GLM-5.3-Flash with --experts cache: in decode and verify steps, the routed experts the
    /// VRAM cache holds score this much more when the router picks its top 8 (the router's
    /// own selection bias spans ~0.03-0.13; the weights of the chosen experts stay unbiased).
    /// More routings hit the cache, fewer read host memory; prompts route as the model does.
    /// Changes outputs. 0 (the default) is off. After Xolotl's --route-bias
    /// (Indras-Mirror/Xolotl, MIT).
    #[arg(long, value_name = "BONUS", default_value_t = 0.0, value_parser = non_negative)]
    pub glm_route_bias: f32,
    /// GLM-5.3-Flash: soft pruning. A file of `layer expert` lines (Xolotl's --prune lists, e.g.
    /// a REAP saliency ranking's least salient quarter); those experts' routing selection bias
    /// is lowered by --glm-prune-penalty at load, so they win only where the router wants them
    /// by more than that. Changes outputs. Off by default. After Xolotl's --prune with
    /// --prune-penalty (Indras-Mirror/Xolotl, MIT).
    #[arg(long, value_name = "FILE")]
    pub glm_prune: Option<PathBuf>,
    /// With --glm-prune: the selection penalty of the listed experts (Xolotl's GLM setting:
    /// 0.05; ~0.5 acts as a hard prune on GLM's router).
    #[arg(long, value_name = "PENALTY", default_value_t = 0.05, value_parser = non_negative,
        requires = "glm_prune")]
    pub glm_prune_penalty: f32,
    /// How qwen4exp's n-gram table (27 GiB, read a few rows per token on the CPU) is read on a
    /// GPU, and DeepSeek-V4.1-Flash's Engram tables (~110 GB, on any device): `ram` (locked in
    /// memory), `direct` (unbuffered SSD reads plus a RAM row cache; it
    /// never fills the page cache), `mmap` (the page cache) or `auto` (`ram` when it fits with a
    /// margin and may be locked, else `direct`; always `direct` on a GPU that shares the
    /// host's memory, e.g. the GB10, whose KV cache and buffers come later out of that memory).
    /// `mmap` takes no --ngram-ram; `ram` takes `all` or a size at least the table's.
    #[arg(long, value_enum, env = "LEGION_NGRAM_IO", default_value_t = TableIo::Auto)]
    pub ngram_io: TableIo,
    /// How much of the n-gram (or Engram) table lives in RAM; the rest is read from the SSD.
    /// `auto` (default): as --ngram-io decides (`direct` keeps a RAM cache of a quarter of the
    /// memory left free, at most 4 GiB, at most 1 on a GPU that shares the host's memory).
    /// `all`: the whole table, locked (--ngram-io ram). SIZE: a RAM budget, in binary units
    /// with an optional K/M/G/T suffix (8G = 8 GiB; no suffix: GiB; decimals allowed) or a
    /// percentage of the memory available at load (30%). A SIZE at least the table's takes
    /// it whole; a smaller one reads it directly from the SSD with exactly that much RAM, half
    /// of it pinning the rows hit most since start and the rest replacing by use, so rows read
    /// once do not push out rows read again. The same rows either way: outputs do not change.
    #[arg(long, value_name = "auto|all|SIZE", value_parser = lookup_table::parse_ram_budget)]
    pub ngram_ram: Option<RamBudget>,
    /// `--ngram-ram <N>G` (kept for old command lines).
    #[arg(long, env = "LEGION_NGRAM_RAM_GIB", hide = true)]
    pub ngram_ram_gib: Option<f64>,
    /// `--ngram-io direct`: SSD reads in flight (I/O threads).
    #[arg(long, env = "LEGION_NGRAM_INFLIGHT", default_value_t = lookup_table::DEFAULT_INFLIGHT)]
    pub ngram_inflight: usize,
    /// `--ngram-io direct`: batches of at most this many page reads (a decode step's or a draft's
    /// rows, 16 pages per token) go out as Linux AIO from the thread that needs them instead of
    /// through the I/O threads, whose wake-ups from idle cost ~1 ms per batch on the GB10 (#506).
    /// The same bytes either way. 0: I/O threads only.
    #[arg(long, default_value_t = lookup_table::DEFAULT_AIO_MAX, value_name = "PAGES")]
    pub ngram_aio_max: usize,
    /// `--ngram-io direct`: while decode steps fetch rows, read one page of the table whenever no
    /// read went out for this many ms, so the SSD does not drop into a power-saving state between
    /// steps; its wake-up cost ~2 ms per C=16 step on the GB10 (#835). Stops 1 s after the last
    /// fetch; ~100 KB/s at 40. The rows are the same. 0: off (default); try 40 where decode steps
    /// are longer than the drive's idle timeout.
    #[arg(long, default_value_t = 0, value_name = "MS")]
    pub ngram_keep_awake_ms: u64,
    /// With several GPUs, run each decode and verify batch as one micro-batch per stage, the
    /// stages on different halves at once, instead of whole on each stage in turn. Off by
    /// default: on an AC922 (4x V100) it lowered C=16 throughput (halves cost about what whole batches do).
    #[arg(long, env = "LEGION_SPLIT_BATCHES")]
    pub split_batches: bool,
    /// With several GPUs (a layer split), run the decoding sequences as this many groups whose
    /// steps are in flight at once: each GPU runs one group's step while the next GPU runs
    /// another's, instead of every GPU waiting for the others' turn on one whole batch. Needs
    /// at least two decoding sequences per group; prompts drain the groups first. Try as many groups
    /// as GPUs. 0: off (default).
    #[arg(long, default_value_t = 0, value_name = "GROUPS")]
    pub pipeline_groups: usize,
    /// With --pipeline-groups: the fewest decoding sequences per group (default 2); fewer
    /// sequences make fewer groups, and below two groups whole steps run. 1 lets each
    /// sequence be a group of its own (C=4 on four GPUs: four groups of one instead of two of
    /// two). On four V100s with MTP that took C=3 from 79 to 95 tok/s, left C=4 equal and
    /// cost 4% at C=2; without MTP it cost 7% at C=4 (each round's fixed host time outweighs
    /// the stages it fills); on two GPUs two groups of one were slower than whole steps. Each
    /// group runs the steps its sequences would run as a batch of their own.
    #[arg(
        long,
        default_value_t = legion::engine::lm::DEFAULT_PIPELINE_MIN_GROUP,
        value_name = "SEQS",
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pub pipeline_min_group: usize,
    /// With --pipeline-groups and the MTP head (--spec mtp), and no --layer-split: after
    /// loading, decode a short synthetic run as groups, measure each stage's time per layer and
    /// the last stage's extra work per round (output head, sampling, commit, MTP drafts), and if
    /// a split with fewer layers on the last stage makes the rounds shorter, load the model
    /// again with it (the log says what was measured and chosen). Costs a second load at
    /// startup (~25 s on four V100s). Raises throughput at C>=2 but costs C=1 and long prompts:
    /// on four V100s with Qwen3.8-Flash-Next it chose 14/14/14/6 instead of 12 each, C=4 129 ->
    /// 137 tok/s and C=8 171 -> 188, but C=1 -3% and a 32K prompt -33% (the longest stage has
    /// more layers, and the last GPU's long-prompt staging no longer fits its cache slots). Off
    /// by default.
    #[arg(long)]
    pub pipeline_mtp_split: bool,
    /// With several GPUs and no --layer-split: split the layers as if the last GPU also did
    /// this many layers' worth of other work per step (the output head, the MTP drafts), so
    /// it gets fewer layers than the others (on 48 layers over four GPUs: 2 gives 13/12/12/11,
    /// 3 gives 13/13/12/10). The static form of --pipeline-mtp-split, whose log names the
    /// measured value (extra_layers), without its second load. Whole steps and prompts cross
    /// the stages one after another, so it can pay only with --pipeline-groups: on four V100s
    /// without them, 2 cost 1-3% of decode and 32K prefill (the first GPU's 13 layers share
    /// its expert cache, more misses), and greedy outputs differ from the even split's.
    /// 0 (default): the split by bytes.
    #[arg(long, default_value_t = 0.0, value_name = "LAYERS")]
    pub split_last_extra: f64,
    /// Run the model split over two machines (GLM-5.3-Flash; #562), every weight in memory:
    /// rank 0 holds the token embeddings and the layers before --node-split and serves the
    /// HTTP API, rank 1 the layers from there on and the output head. Start rank 1 first;
    /// rank 0 connects to it (and waits up to 10 minutes for it). Only the residual stream
    /// and the steps' positions cross the link. Decode groups are in flight on both machines
    /// (--pipeline-groups 2 unless given; 1 turns that off). Speculative decoding: --spec mtp
    /// only, the model's own MTP head, which runs on rank 1 (it loads the head and a copy of
    /// the token embeddings, and the default split counts them); give both ranks the same
    /// --spec flags. Not with --tp, other --spec modes, host experts or a KV host tier. Off
    /// by default.
    #[arg(long, value_name = "RANK", value_parser = clap::value_parser!(u8).range(0..=1))]
    pub node_rank: Option<u8>,
    /// With --node-rank: rank 1's address, ip:port (e.g. a direct link's). Rank 1 listens
    /// there, rank 0 connects to it.
    #[arg(long, value_name = "IP:PORT", requires = "node_rank")]
    pub node_peer: Option<String>,
    /// With --node-rank: rank 1's first layer (the same on both ranks). Default: the layer
    /// that balances the two machines' weight bytes. A share that does not fit fails at
    /// startup, naming a split that would.
    #[arg(long, value_name = "LAYER", requires = "node_rank")]
    pub node_split: Option<usize>,
    /// Run the model with tensor parallelism over several machines, one GPU each (dense Gemma
    /// 4: GGUFs and NVFP4 safetensors checkpoints, #619; Qwen3.8-Flash-Next: GGUFs, #733):
    /// every rank holds its share of each layer's attention heads (or DeltaNet heads) and FFN
    /// rows (every expert's) and runs every step at once, summing the attention's and FFN's
    /// outputs over the network. This machine's rank: 0 serves the HTTP
    /// API and drives the steps, the others follow it. Start them in any order; each waits up
    /// to 10 minutes for the others. Speculative decoding: --spec mtp (the assistant split like
    /// the target) or --spec dflash|dspark (rank 0 alone drafts; the verify steps are ordinary
    /// steps), the same --spec and --spec-draft on every rank. Not with --tp, --node-rank,
    /// other --spec modes, a KV host tier or pipeline groups. Off by default.
    #[arg(
        long,
        value_name = "RANK",
        requires = "tp_peers",
        conflicts_with = "node_rank"
    )]
    pub tp_rank: Option<usize>,
    /// With --tp-rank: every rank's address, ip:port, in rank order, the same list on every
    /// rank (2 to 8 ranks). Rank r > 0 listens on entry r; rank 0's entry is not listened on.
    #[arg(
        long,
        value_name = "IP:PORT,...",
        value_delimiter = ',',
        requires = "tp_rank"
    )]
    pub tp_peers: Vec<String>,
    /// With --tp-rank: seconds a rank waits for another's share of a sum before it gives the
    /// step up (and the ranks stop).
    #[arg(long, value_name = "SECS", default_value_t = 300)]
    pub tp_timeout: u64,
    /// With --tp-rank: how the shares of the sums travel. auto (default): RDMA writes when
    /// both ends have an RDMA NIC (RoCE v2) on the --tp-peers addresses and run on a GPU, else
    /// the TCP connections; tcp: always TCP; rdma: fail to start without RDMA. On two DGX
    /// Sparks RDMA cut a sum's wait from about 0.6 ms to about 8 µs one way (Gemma 4 31B: 120
    /// sums a token). Every rank must pass the same value.
    #[arg(long, value_name = "MODE", value_enum, default_value_t = legion::tp_net::Transport::Auto)]
    pub tp_transport: legion::tp_net::Transport,
    /// With --tp-rank and RDMA: RDMA rails per peer, 1 (default) or 2. A DGX Spark's QSFP port
    /// is two RDMA devices (two PCIe links of about 108 Gb/s each); 1 uses the one on the
    /// --tp-peers address, 2 also the other (by its link-local RoCE v1 address, no IP needed)
    /// and sends each prompt-step sum of at least 1 MiB in 1 MiB pieces over both, each
    /// device taking the next piece as it frees up. The same bytes arrive: outputs do not
    /// change. Decode-step sums stay on the first. A rank without a second device, or a peer
    /// passing 1, keeps one rail.
    #[arg(long, value_name = "N", default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=2))]
    pub tp_rails: u8,
    /// With --tp-rank: log where the steps' time went (compute, waits for the other ranks'
    /// shares, the step send) every N steps on each rank, to tune a setup. 0 (default): off.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub tp_profile: usize,
    /// With --tp-rank: the type the shares of the sums travel in. f32 (default): as computed;
    /// bf16: half the bytes, for prompts on a slow link (two DGX Sparks: 42 MB a sum per 4096
    /// prompt tokens of Qwen3.8-Flash-Next, about 3.4 ms one way). Only sums of at least 1 MiB
    /// (about 100 tokens) go in BF16; decode steps stay F32. Every rank's share, its own
    /// included, is rounded to BF16 before the F32 sum, so the ranks still agree; the prompt's
    /// output drifts a little from f32's. Every rank must pass the same value.
    #[arg(long, value_name = "TYPE", value_enum, default_value_t = legion::tp_net::Wire::F32)]
    pub tp_wire: legion::tp_net::Wire,
    /// With --tp-rank: prompt steps of at least N tokens split the per-token work between the
    /// sums by tokens as well (sequence parallelism): each rank runs Qwen3.8-Flash-Next's
    /// hyper-connection mixers on its share of the tokens instead of all of them, and each sum
    /// becomes a reduce-scatter plus an all-gather of the same bytes. For long prompts over
    /// two machines (e.g. 1024). 0 (default): off. Every rank must pass the same value.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub tp_seq_par: usize,
    /// With --tp-rank and --spec mtp (Qwen3.8-Flash-Next's head), on rank 0: a draft round
    /// goes to the other ranks as one request (its first MTP pass, each sequence's draft cap and
    /// the confidence floor), and every rank runs the round's further passes on its own: the
    /// head's picks and probabilities are the same bits on every rank, so all ranks make the
    /// same passes. Off (default): one request per pass, each waiting for the other ranks to
    /// receive it. Same drafts and output either way. Two DGX Sparks, Qwen3.8-Flash-Next with
    /// --spec-fixed: an 8-pass round 21.0 -> 16.6 ms, chats +2%, greedy output identical.
    #[arg(long)]
    pub tp_mtp_chain: bool,
    /// With --tp-rank and --spec mtp (Qwen3.8-Flash-Next's head), on rank 0: when the MTP head
    /// takes in a prompt chunk, the other ranks read the chunk's residual rows from their own
    /// copy of the target step's output instead of receiving them from rank 0 (every rank
    /// holds the same bits after the step's last sum). Off (default): rank 0 sends the rows,
    /// about 42 MB per 1024 prompt tokens of Qwen3.8-Flash-Next over the control connection.
    /// Two DGX Sparks, --spec-k 2: the gap between 4096-token prompt steps 208 -> 35 ms, 32K
    /// prefill 2,237 -> 2,436 tok/s, 8K 2,309 -> 2,364; same drafts and output either way.
    /// With --tp-mtp-chain, a decode draft round's rows from the verify step too (#1176).
    #[arg(long)]
    pub tp_mtp_local_rows: bool,
    /// With --tp-rank, Qwen3.8-Flash-Next (qwen4exp): expert parallelism for the routed
    /// experts. Rank r of n holds the experts r*E/n .. (r+1)*E/n whole (the checkpoint's bytes
    /// as shipped, cut by expert index) instead of every expert's 1/n of the rows, and runs
    /// only the token-expert pairs routed to them. The router stays on every rank, the shared
    /// expert, attention and DeltaNet keep their split, and the MoE's partial sums are added
    /// by the same all-reduce (no token exchange). For long prompts: each expert's fixed cost
    /// is paid on one rank instead of both. On CUDA, steps of up to 8 tokens (decode, MTP
    /// verify) run the expert matvec on the rank's share. Two DGX Sparks, TP+MTP k=2: decode
    /// C=1 48.6 -> 51.2, C=4 79.2 -> 84.8 tok/s, 8K prefill 2,475 -> 2,667. A step still waits
    /// on the rank that drew more of the token's experts. The MoE sum's order changes, so output may differ in bits from
    /// the row split. Every rank must pass it (the MTP head is cut the same way). Off by default.
    #[arg(long, requires = "tp_rank")]
    pub tp_moe_ep: bool,
    /// With --tp-rank and --tp-seq-par: hide the sums' wire time in those prompt steps behind
    /// compute by running the per-token work in two micro-batches (half of every rank's
    /// tokens each): while one half's reduce-scatter or all-gather is on the wire the GPU runs
    /// the other half's work. off (default): one exchange at a time, the GPU waiting for each.
    /// mixers: the hyper-connection mixers run per half; ffn: the MoE too (each half reads the
    /// expert weights again). Steps with a half under 1,024 tokens on any rank run whole. Needs
    /// twice the pinned exchange buffers. Every rank must pass the same value. On a GPU smaller
    /// halves' matmuls split K differently from the whole share's (MMQ stream-k) and moved long
    /// prompts' greedy output; with the 1,024-token floor, greedy text at 1,536 and 4,096 prompt
    /// tokens matched off on two DGX Sparks (Qwen3.8-Flash-Next NVFP4, mixers: prefill 8K /
    /// 32K / 123K 2,872 / 3,013 / 2,825 -> 2,966 / 3,022 / 2,859 tok/s). Experimental.
    #[arg(long, value_name = "WHAT", value_enum, default_value_t = legion::tp_net::Overlap::Off)]
    pub tp_overlap: legion::tp_net::Overlap,
    /// With --tp-moe-ep: which rank holds each routed expert, per layer, from a JSON file
    /// written by `legion ep-place` (the router's choices recorded with --moe-route-record,
    /// split so experts often picked together sit on different ranks). Every rank still holds
    /// E/n experts, loaded whole from the checkpoint by expert index; a step waits less on the
    /// rank that drew more of a token's experts. Layers the file does not name keep the
    /// contiguous split. Every rank must pass the same file. Off by default.
    #[arg(long, requires = "tp_moe_ep", value_name = "FILE")]
    pub tp_moe_ep_place: Option<PathBuf>,
    /// Appends every MoE layer's router choices (the top-k expert ids of each token of each
    /// step, decode, prompts and MTP drafts alike) to FILE, for `legion ep-place`. Each step's
    /// ids are copied to the host before the step goes on, so the server runs slower while
    /// recording: for calibration runs only. Off by default.
    #[arg(long, value_name = "FILE")]
    pub moe_route_record: Option<PathBuf>,
    /// GLM-5.3-Flash on several GPUs: every GPU works on every token instead of one pipeline
    /// stage at a time (tensor parallel). Each GPU runs a share of every layer: its attention
    /// heads (with their KV and states), its rows of the dense and shared FFNs, its routed
    /// experts (cached in its VRAM, its misses read from its own socket's memory) and its rows
    /// of the LM head, with one all-reduce per attention and per FFN. Prompts and decode run
    /// this way, in steps of up to 512 tokens (prompts of 128 tokens or more in steps of
    /// --glm-prompt-step). Only GLM's own MTP head drafts; no KV host tier yet. Greedy output differs from the pipeline's by summation order. Off by default.
    #[arg(long)]
    pub glm_parallel_decode: bool,
    /// qwen4exp (Qwen3.8-Flash-Next) over 2 or 4 GPUs in NVLink pairs (CUDA0+1, CUDA2+3, each
    /// pair on one socket): while a stage's GPU runs a decode or verify step of up to 64
    /// tokens, the other GPU of its pair, idle otherwise, computes half of each of its MoE
    /// layers' routed experts, exchanged over NVLink. Each GPU's expert cache holds its half
    /// of its own layers and the other half of its pair's. The layer split, the KV cache and
    /// the MTP head are unchanged. Prompts run on the stage's GPU alone, as before, but with
    /// only half of its layers' experts cached: the other half comes from host memory, so
    /// long prompts are several times slower. Greedy output may differ from the default at
    /// near-ties (two partial sums added). Off by default.
    #[arg(long)]
    pub pair_tp: bool,
    /// With --glm-parallel-decode: which GPUs write their shares of the exchanges (the
    /// all-reduces, the embedding's broadcast) straight into another GPU's memory over a
    /// GPU-GPU link instead of through host memory. pairs (default): the GPUs of one NUMA node
    /// (an AC922's NVLink pairs); all: every GPU that can reach another's memory, across the
    /// sockets too; off: all through host memory. Each such GPU keeps a VRAM inbox (about 16
    /// MiB a writer). The sums keep their order: greedy output is the same in every mode.
    #[arg(long, value_name = "MODE", value_enum, default_value_t = legion::model::glm5next::par::exchange::PeerWrites::Pairs)]
    pub glm_exchange_peer: legion::model::glm5next::par::exchange::PeerWrites,
    /// With --glm-parallel-decode: tokens per step of a prompt of 128 tokens or more, which
    /// borrows the expert cache for the step buffers and the staging while it is read. Each GPU
    /// copies its uncached experts of a layer into VRAM once a step and each expert's matmul
    /// reads its weights once for the step's tokens, so bigger steps read prompts faster; the
    /// sequences decoding beside a prompt wait for whole steps, and the exchange's pinned host
    /// memory grows by 512 KiB a token on each of four GPUs (1 GiB each at 2048). Default 2048; 512
    /// keeps every step at 512 tokens (as before); at most 8192. Decode and verify steps stay
    /// at most 512 tokens.
    #[arg(long, value_name = "TOKENS", default_value_t = legion::model::glm5next::par::PROMPT_TOKENS,
          value_parser = clap::builder::TypedValueParser::map(clap::value_parser!(u64).range(512..=8192), |v| v as usize))]
    pub glm_prompt_step: usize,
    /// With --glm-parallel-decode: which routed experts each GPU owns. contiguous (default):
    /// GPU d owns one contiguous range of the expert ids; interleave: GPU d of n owns the
    /// experts d, d+n, d+2n, ... of every MoE layer (renumbered as the model loads), a sample of
    /// the whole id range. The two add the routed sum in another order, so greedy output
    /// differs at near-ties. On the AC922 interleave measured no faster (#677). An expert
    /// profile, --glm-prune or --weights-fp16-gates keep the contiguous split.
    #[arg(long, value_name = "SPLIT", value_enum, default_value_t = legion::model::glm5next::par::assign::ExpertSplit::Contiguous)]
    pub glm_expert_split: legion::model::glm5next::par::assign::ExpertSplit,
    /// With --glm-parallel-decode and the contiguous split: how many parts of the routed experts
    /// each GPU owns, one number per GPU in CUDA order (e.g. 76,76,72,64 on four GPUs gives
    /// 76/76/72/64 of 288 experts). The last GPU also runs the MTP draft head and keeps the
    /// drafts' LM head, so a smaller share there evens the step's work. Default: an even split.
    /// Another split adds the routed sum in another order, so greedy output differs at
    /// near-ties.
    #[arg(long, value_name = "N,N,...", value_delimiter = ',', num_args = 1..)]
    pub glm_expert_shares: Vec<u32>,
    /// With --glm-parallel-decode and the MTP head drafting (--spec mtp): the head's drafts run
    /// on every GPU instead of the last GPU alone. The last GPU keeps the head's attention and
    /// its cache; every GPU computes its own routed experts of the head's MoE (each caches
    /// them), its rows of the shared expert and its rows of the draft logits, so the last GPU
    /// no longer keeps a full copy of the LM head (about 1 GiB more VRAM for its cache). The
    /// routed sum is added in another order: drafts, and so greedy output, may differ at
    /// near-ties. Ignored with --glm-expert-split interleave. Off by default.
    #[arg(long)]
    pub glm_parallel_mtp: bool,
    /// With --glm-parallel-mtp: less host time between steps (#723) after steps of at most N
    /// sequences. Each GPU queues its verify commit right after its MTP pass instead of before
    /// it (the last GPU's commit was launched from the main thread first, ~0.7 ms at C=1,
    /// while the others waited for its head), and the target step's expert-cache work runs
    /// while the launch threads queue the pass instead of after them. The same bits. Four
    /// V100s, Maya-S, --spec-k 2: C=1 greedy decode 58 -> 61 tok/s per request; at C=4 it lost
    /// ~5% of the total at the same draft depth (#725), so the default, 1, keeps the old order
    /// for steps of two or more sequences. 0 turns it off.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub glm_mtp_overlap_seqs: usize,
    /// With --glm-parallel-mtp: the --glm-mtp-overlap-seqs order after every step, whatever its
    /// sequences (as #723 shipped it; C=4 loses ~5% on four V100s). For comparisons.
    #[arg(long)]
    pub glm_mtp_overlap: bool,
    /// With --glm-parallel-decode: balance a decode step's routed experts within each NVLink
    /// pair of GPUs (0,1 and 2,3). An expert its owner caches stays with the owner; one it
    /// does not (read from host memory, about 12 times as slow) goes to the GPU of the pair
    /// with less work in the step, so the exchanges wait less for the slowest GPU. Both GPUs
    /// read the same node's host memory. Four V100s, Maya-S, --spec-k 2 --glm-parallel-mtp:
    /// C=1 greedy decode 60.5 -> 63 tok/s per request, C=4 the same within noise (#729). The
    /// routed sum is added in another order that follows the cache's contents, so greedy output
    /// may differ at near-ties, also from run to run. Prompts are not changed. Off by default.
    #[arg(long)]
    pub glm_pair_balance: bool,
    /// With --glm-parallel-decode: measure how well each GPU's host-read (uncached) experts of
    /// the next layer could be foreseen, for a prefetch (#745). Every MoE layer's graph also
    /// runs the next layer's router on its own input and logs the guess; the expert cache's
    /// periodic report then adds a line "expert prefetch predictors" with, per GPU and step,
    /// the misses, how many of them that guess or the previous step's choices named, and how
    /// many uncached experts each would have copied. Costs one router matvec per layer and
    /// some host time per step; the output is not changed. Off by default.
    #[arg(long)]
    pub glm_prefetch_stats: bool,
    /// With --glm-parallel-decode: --glm-pair-balance over all the GPUs. An uncached expert may
    /// also go to a GPU of the other pair, which reads it from the owner's socket across the
    /// X-bus at about 2/3 of the speed (18 cached reads instead of 12), when it still finishes
    /// there first. Cached experts stay with their owner. Greedy output may differ at
    /// near-ties, as with --glm-pair-balance. Off by default; implies --glm-pair-balance.
    #[arg(long)]
    pub glm_cross_balance: bool,
    /// With --glm-parallel-decode: copy up to N of each GPU's uncached experts of the next
    /// layer into VRAM while the current layer computes (#747). At every layer of a decode
    /// step of at most 16 tokens, a side stream applies the next layer's router to the current
    /// layer's FFN input on the GPU and copies the guessed experts that miss the GPU's cache
    /// into a small VRAM staging ring (2 x N experts, taken from the expert cache's share of VRAM,
    /// not from the 1 GiB reserve); the next layer reads those of its real choices from VRAM
    /// with the same bits instead of from host memory. With --glm-pair-balance, staged experts
    /// count as cached. The output is not changed beyond what --glm-pair-balance changes.
    /// Experimental: measured about 5% slower at C=1 so far (see docs/serving.md).
    /// 0 (the default): off.
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub glm_prefetch: usize,
    /// With --glm-parallel-decode: split every routed expert of a decode step that its owner
    /// does not cache over both GPUs of the owner's NVLink pair (0,1 and 2,3) by rows: each
    /// computes half of the expert's intermediate rows (SwiGLU) and the down projection over
    /// that half, reading half the expert from its node's host memory, and the exchange adds
    /// the two partial sums. Cached experts stay whole with their owner. Takes precedence over
    /// --glm-pair-balance and --glm-cross-balance for layers whose expert types the split
    /// kernels take (Maya-S's IQ2_XXS/IQ2_S/IQ3_XXS). Four V100s, Maya-S, --spec-k 2
    /// --glm-parallel-mtp: C=1 greedy decode 60.2 -> 62.4 tok/s per request (--glm-pair-balance
    /// 61.9), C=4 63.2 -> 69.1 total (balance 66.5, rounds vary ±5%) (#735). An uncached
    /// expert's output is added in another order that follows the cache's contents, so greedy
    /// output may differ at near-ties, also from run to run. Prompts are not changed. Off by
    /// default.
    #[arg(long)]
    pub glm_pair_split: bool,
    #[command(flatten)]
    pub switches: switches::SwitchArgs,
}

impl ModelArgs {
    /// How the n-gram table is read (`--ngram-io`, `--ngram-ram` or its old spelling
    /// `--ngram-ram-gib`, `--ngram-inflight`, `--ngram-aio-max`, `--ngram-keep-awake-ms`).
    /// Contradicting pairs are errors here, before the load; a size against the table's own
    /// is checked when the table opens.
    pub(super) fn ngram_options(&self) -> anyhow::Result<lookup_table::TableOptions> {
        let ram = match (self.ngram_ram, self.ngram_ram_gib) {
            (Some(_), Some(_)) => {
                anyhow::bail!("--ngram-ram and --ngram-ram-gib (its old spelling) both given")
            }
            (Some(r), None) => r,
            (None, Some(g)) => lookup_table::parse_ram_budget(&format!("{g}G"))
                .map_err(|e| anyhow::anyhow!("--ngram-ram-gib: {e}"))?,
            (None, None) => RamBudget::Auto,
        };
        if let Some(e) = lookup_table::ram_conflict(self.ngram_io, ram) {
            anyhow::bail!(e);
        }
        Ok(lookup_table::TableOptions {
            io: self.ngram_io,
            ram,
            ram_cache_bytes: None,
            inflight: self.ngram_inflight,
            aio_max: self.ngram_aio_max,
            keep_awake_ms: self.ngram_keep_awake_ms,
            node: None,
            unified: false,
        })
    }

    /// The most sequences of a step the #723 overlap applies to: `--glm-mtp-overlap-seqs`, or
    /// every step with `--glm-mtp-overlap`.
    pub(super) fn glm_mtp_overlap_seqs(&self) -> usize {
        if self.glm_mtp_overlap {
            usize::MAX
        } else {
            self.glm_mtp_overlap_seqs
        }
    }

    /// `--glm-prune`'s list, read, with `--glm-prune-penalty`.
    pub(super) fn glm_prune(&self) -> anyhow::Result<Option<legion::model::glm5next::SoftPrune>> {
        let Some(path) = &self.glm_prune else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("--glm-prune: reading {}: {e}", path.display()))?;
        let pairs = legion::model::glm5next::parse_prune_list(&text)
            .map_err(|e| anyhow::anyhow!("--glm-prune {}: {e}", path.display()))?;
        Ok(Some(legion::model::glm5next::SoftPrune {
            pairs,
            penalty: self.glm_prune_penalty,
        }))
    }

    /// Whether graphs run as stage graphs (`--stage-graphs`, on by default since #849): not
    /// turned off with `--no-stage-graphs`, and on one GPU per model part. Tensor parallelism's
    /// meta backend runs ggml's graphs itself (they would run without CUDA graphs at all), and
    /// the spinning exchanges must not wait behind a capture.
    pub(super) fn stage_graphs(&self) -> bool {
        self.switches.stage_graphs() && self.tp <= 1 && !self.glm_parallel_decode && !self.pair_tp
    }

    /// The switches `legion serve` applies: the flags' ([`switches::SwitchArgs::switches`]),
    /// with the stage graphs on wherever they apply (the default since #849; `bind_numa` turned
    /// ggml's own CUDA graphs off for them) and untouched (off) elsewhere.
    pub(super) fn served_switches(&self) -> Switches {
        let mut s = self.switches.switches();
        s.stage_graphs = self.stage_graphs().then_some(true);
        s
    }

    /// `--tp-replicas` resolved: the number of model copies (1 unless asked for). `auto` and
    /// counts past 1 ask the driver how many GPUs are visible.
    pub(super) fn replicas(&self) -> anyhow::Result<usize> {
        use legion::engine::replicas::ReplicaCount;
        let visible = match self.tp_replicas {
            ReplicaCount::Fixed(1) => 0,
            _ if self.tp < 2 => 0,
            _ => legion::hw::numa_bind::gpu_count(),
        };
        self.tp_replicas
            .resolve(self.tp, visible)
            .map_err(anyhow::Error::msg)
    }

    /// Whether a model is named at all (`--model` or `--hfmodel`): `serve` then starts the
    /// model engine, `mock` otherwise.
    pub(super) fn is_named(&self) -> bool {
        self.model.is_some() || self.hfmodel.is_some()
    }

    /// The model file or checkpoint directory to load: `--model`'s path (a GGUF file, or a
    /// directory holding one), or the Hugging Face cache snapshot `--hfmodel` names.
    pub(super) fn resolve_model(&self) -> anyhow::Result<PathBuf> {
        match (&self.model, &self.hfmodel) {
            (Some(_), Some(_)) => anyhow::bail!("--hfmodel and --model are mutually exclusive"),
            (Some(path), None) => model_path::resolve_model_path(path),
            (None, Some(spec)) => hfmodel::resolve(spec),
            (None, None) => anyhow::bail!(
                "no model to load: pass --model <file.gguf|dir> or \
                 --hfmodel <org/name>[@<revision>]"
            ),
        }
    }

    /// The residency settings these flags ask for.
    pub(super) fn residency(&self) -> ResidencyConfig {
        ResidencyConfig {
            mode: self.experts,
            cache_bytes: self
                .expert_cache_gib
                .map(|g| (g.max(0.0) * (1u64 << 30) as f64) as u64),
            cache_reserve: self.expert_cache_reserve_mib << 20,
            fill_prompt_room: if self.no_expert_cache_fill_prompt_room {
                Some(false)
            } else {
                self.expert_cache_fill_prompt_room.then_some(true)
            },
            profile: self.expert_profile.clone(),
            profile_seed: self
                .expert_profile_seed
                .unwrap_or(legion::residency::PROFILE_SEED),
            profile_split_only: self.expert_profile_split_only,
            split_misses: self.expert_split_misses.clone(),
            swaps: SwapRule {
                budget: self.expert_swaps,
                margin: self.expert_swap_margin,
                ratio: self.expert_swap_ratio,
            },
            device_swaps: self.expert_swaps_device.clone(),
            decay: self.expert_heat_decay,
            heat_pass_cap: self.expert_heat_pass_cap,
            fetch: self.expert_fetch,
            arena: !self.no_expert_arena,
            staging: self.prefill_staging,
            ring: self.prefill_ring,
            stage_pieces: self.prefill_stage_pieces,
            stage_reuse: self.prefill_staging_reuse,
            stage_min_tokens: self.prefill_stage_min_tokens,
            stage_peer: self.prefill_peer_stage,
            lend_keep: self.prefill_keep_cache.iter().map(|m| m << 20).collect(),
            lend_spread: self.prefill_keep_spread,
            lend_repack: self.prefill_keep_repack,
            lend_need: Vec::new(),
            stage_below: None,
            learn_long: !self.no_prefill_learn,
            cpu_share: legion::residency::cpu_share::CpuShareConfig {
                share: self.cpu_expert_share,
                workers: self.cpu_expert_workers,
                smt: self.cpu_expert_smt as usize,
            },
            ssd: SsdArgs {
                mode: self.expert_ssd,
                cache_bytes: self
                    .expert_ssd_cache_gib
                    .map(|g| (g.max(0.0) * (1u64 << 30) as f64) as u64),
                reserve: (self.expert_ssd_reserve_gib.max(0.0) * (1u64 << 30) as f64) as u64,
                readers: self.expert_ssd_readers.max(1),
                direct: !self.expert_ssd_buffered,
                prefetch: self.ssd_prefetch,
                prefetch_margin: self.ssd_prefetch_margin,
                prefetch_ranks: self.ssd_prefetch_ranks,
                prefetch_early: self.ssd_prefetch_early,
            },
            ep: None,
            ep_stage_all: false,
            ep_cached: false,
            ep_predict: false,
            ep_prefetch: 0,
            deterministic: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(super) enum ExpertMv {
    Legion,
    Ggml,
}

impl DeviceArg {
    pub(super) fn choice(self) -> DeviceChoice {
        match self {
            DeviceArg::Cpu => DeviceChoice::Cpu,
            DeviceArg::Cuda => DeviceChoice::Cuda,
        }
    }
}

#[derive(clap::Args)]
struct EngineArgs {
    /// Defaults to `model` when `--model` or `--hfmodel` is given, `mock` otherwise.
    #[arg(long, value_enum)]
    engine: Option<EngineKind>,
    #[command(flatten)]
    model: ModelArgs,
    /// Model id reported by /v1/models (default: `mock`; with `--hfmodel`, the repo id
    /// `org/name`; else the GGUF's `general.name`).
    #[arg(long)]
    model_name: Option<String>,
    /// NUMA placement with --device cuda (Linux), set at startup before any large allocation:
    /// `auto` binds the process's CPUs and memory to the NUMA node of the GPUs in use, as
    /// `numactl --cpunodebind=N --membind=N` would (GPUs on several nodes: each GPU's threads
    /// and pinned host buffers on its own node, the rest unbound, never interleaved); `<node>`
    /// or `<node>,<node>` binds to those nodes; `off` leaves placement to the OS. A process
    /// already restricted by numactl, taskset or a cpuset is kept as it is. One-node machines
    /// (desktops, GB10, GH200) are left alone. Turn it off to place the process yourself, or to
    /// let a model whose host memory outgrows the GPUs' node spill to another node.
    #[arg(long, default_value = "auto", value_name = "auto|off|NODE[,NODE...]")]
    numa: NumaMode,
    /// Simulated decode step time of the mock engine.
    #[arg(long, default_value_t = 5)]
    mock_step_ms: u64,
    /// Sequences decoded together.
    #[arg(long, default_value_t = 64)]
    max_batch: usize,
    /// Requests allowed to wait for a batch slot before new ones get 429 (OpenAI) / 529
    /// (Anthropic).
    #[arg(long, default_value_t = 1024)]
    max_waiting: usize,
    /// Order in which waiting requests are admitted to a batch slot. `fcfs` (default):
    /// arrival order, the request's `priority` field is accepted and ignored. `priority`:
    /// the lowest `priority` value first (the default 0; negative values go ahead of it),
    /// ties in arrival order, and prompt prefill in the same order; running requests are
    /// preempted only by --kv-admission on-demand, lowest priority first. Lower-first
    /// matches vLLM; SGLang's default is the opposite (higher first). Use it when some
    /// clients (interactive chat) must not queue behind bulk jobs.
    #[arg(long, value_enum, default_value_t = SchedulingPolicy::Fcfs)]
    scheduling_policy: SchedulingPolicy,
    /// Model engine: how admission sizes a request's KV cache. `reserve` (default): the
    /// whole prompt plus max_tokens at admission, so a running request never runs short, but
    /// a request asking for 32K tokens holds them even if it stops after 200. `on-demand`
    /// (as SGLang and vLLM): the prompt plus a step's headroom; sequences grow as they
    /// decode, and when the cache runs short the cheapest one is preempted (the lowest
    /// --scheduling-policy priority, then the fewest tokens cached), requeued at the front
    /// of its class and recomputed later. More requests run at once; a preempted one pays
    /// its prompt again. Not with --pipeline-groups or KV slots (falls back to `reserve`).
    #[arg(long, value_enum, default_value_t = KvAdmission::Reserve)]
    kv_admission: KvAdmission,
    /// Model engine: inter-token latency target in ms (off by default). While sequences
    /// decode, each step takes only as many prompt tokens (whole KV blocks) as a step-time
    /// model, fitted online from the measured steps, puts under this target, so a long prompt
    /// arriving no longer stalls the chats for a whole --max-batch-tokens chunk per step. With
    /// nothing decoding a prompt still gets the full --max-batch-tokens. Long prompts read
    /// slower meanwhile; pair it with --ttft-target. The fit error is logged every 30 s. Only
    /// step boundaries move. Not with --pipeline-groups above 1 (nor over two machines
    /// without --pipeline-groups 1).
    #[arg(long, value_name = "MS", value_parser = clap::value_parser!(u32).range(1..))]
    itl_target: Option<u32>,
    /// Model engine: time-to-first-token target in ms, from a request's arrival (off by
    /// default). Prompts that would miss it at the planned pace read first (the fewest tokens
    /// left first, within a --scheduling-policy priority class); with --itl-target, a step
    /// takes more prompt tokens than the ITL budget when an at-risk prompt can still make its
    /// target that way. A prompt already past it reads after those still in time and raises
    /// nothing, so one miss does not make the next prompts miss too. Same restrictions as
    /// --itl-target.
    #[arg(long, value_name = "MS", value_parser = clap::value_parser!(u32).range(1..))]
    ttft_target: Option<u32>,
    /// Model engine: sequences generated at once (continuous batching).
    #[arg(long, default_value_t = 16)]
    max_seqs: usize,
    /// Model engine: threads that compute the token masks of grammar-constrained requests
    /// (`response_format` json_object/json_schema, `strict` tools) for the next step while the
    /// model runs the current one, so sampling only applies a ready mask. 0: compute each mask
    /// on the engine thread when sampling. Started only once a request is constrained;
    /// requests without a grammar never use them.
    #[arg(long, default_value_t = legion::engine::lm::GRAMMAR_THREADS)]
    grammar_threads: usize,
    /// Model engine: once this many sequences run at once, new requests' text (detokenizing,
    /// reasoning and tool-call parsing, log probabilities, the streamed sends) is handled on a
    /// thread of its own, beside the next step, instead of on the engine thread between steps.
    /// The streamed text is the same. Requests with stop strings or a grammar
    /// (`response_format`, `strict` tools) stay on the engine thread: a stop string ends the
    /// sequence, and a grammar follows the reasoning parser, before the next step. Smaller
    /// batches keep it there, where it costs less than waking the thread. 0: never.
    #[arg(long, default_value_t = legion::engine::lm::STREAM_THREAD_SEQS)]
    stream_thread_seqs: usize,
    /// Model engine: tokens per forward pass, one per decoding sequence plus prompt chunks
    /// (default 1024; 4096 on a GPU that shares memory with the host, e.g. the GB10).
    #[arg(long)]
    max_batch_tokens: Option<usize>,
    /// Model engine, on a GPU that shares memory with the host (a DGX Spark's GB10): GiB the
    /// automatic KV pool leaves free for the OS and other programs, on top of what it keeps
    /// for the step buffers (default 16). The pool takes the rest, up to max(--ctx-size,
    /// 262144) tokens, and a --ctx-size that does not fit is refused at load. Lower it on a
    /// machine that runs nothing else when the weights leave no room for the context (e.g. 8
    /// for GLM-5.3-Flash Maya-S with MTP and 16 sequences at 131072 on one Spark); below
    /// ~4 GiB the box may swap. No effect on other GPUs or on the output.
    #[arg(long, value_name = "GIB", default_value_t = 16.0, value_parser = non_negative)]
    uma_free_gib: f32,
    /// Model engine: prompts read at once, each in a recurrent-state slot of its own above the
    /// --max-seqs decode slots (states sized for --max-seqs plus this). A burst of requests
    /// with short prompts starts decoding in fewer steps with more slots: at 4, sixteen
    /// requests arriving together are read four per step and decode out of step with each
    /// other; at 16 they start together. Prefill still shares --max-batch-tokens per step.
    /// Batches change, so greedy text can differ from the default.
    #[arg(long, default_value_t = legion::engine::lm::PREFILL_SLOTS,
          value_parser = clap::builder::TypedValueParser::map(clap::value_parser!(u64).range(1..=256), |v| v as usize))]
    prefill_slots: usize,
    /// Model engine: prompt tokens per prefill chunk (at most --max-batch-tokens per step;
    /// default 512, on a GPU sharing memory with the host --max-batch-tokens). With
    /// sliding-window layers in rings, also the most a sequence takes per step (their rings
    /// hold the window plus one step); by default there --max-batch-tokens on a GPU sharing
    /// memory with the host (GB10), 512 elsewhere.
    #[arg(long)]
    prefill_chunk: Option<usize>,
    /// Model engine: a prompt that needs several steps takes them in equal parts (multiples of
    /// 64 tokens) instead of full steps and a short last one, e.g. 8,240 tokens in steps of
    /// 4,096 as 3 x ~2,750 instead of 4,096 + 4,096 + 48. Each step of a MoE reads every
    /// routed expert, so a short last step costs nearly as much as a full one's weights. For
    /// prompts just past a multiple of the step (--max-batch-tokens). Off by default (full
    /// steps).
    #[arg(long)]
    prefill_even: bool,
    /// Model engine: before taking requests, run one prompt of about this many tokens (greedy,
    /// one output token). A step larger than any before it grows its buffers: the step's compute
    /// buffer, ggml-cuda's pool and, under --tp-rank, the pinned exchange buffers (registered
    /// with the NIC again), each allocated anew at the full size, and that step is slower (two
    /// DGX Sparks, the first 8K prompt: 4.1 s, the next 3.3 s). Use 3 x --max-batch-tokens:
    /// three full steps, one for each kept prompt-step graph's buffer; a warmup of smaller steps
    /// leaves the next larger step to grow them again. 0 (default): no warmup.
    #[arg(long, value_name = "TOKENS", default_value_t = 0)]
    warmup_prompt: usize,
    /// Model engine: before taking requests (and before --warmup-prompt), run one short prompt
    /// of each of these sizes, in tokens (greedy, one output token). The first prompt step of a
    /// new size loads the CUDA kernels only it uses (lazy module loading) and grows ggml-cuda's
    /// pool and the step's scratch buffers: ~130 ms once on a DGX Spark, paid by the first
    /// request of that size after start. Sizes above half the context are skipped, and a
    /// failed warmup prompt only logs a warning. The kept buffers are those of these sizes, so
    /// memory stays bounded. 0 turns it off. Default: 1024,256,128.
    #[arg(
        long,
        value_name = "TOKENS,...",
        value_delimiter = ',',
        default_value = "1024,256,128"
    )]
    warmup_shapes: Vec<usize>,
    /// Model engine: keep every position of sliding-window layers (Gemma 4) in the KV cache,
    /// instead of a ring of their window per sequence.
    #[arg(long)]
    swa_full: bool,
    /// Model engine, DeepSeek-V4.1: rings for its sliding-window layers (the layers without
    /// compressed keys of their own) although prefix caching is on, which they turn off: a
    /// model with a per-sequence state resumes no cached prefix with rings. Without it these
    /// layers keep every position unless prefix caching is off (--prefix-cache-states 0,
    /// --deterministic).
    #[arg(long, conflicts_with = "swa_full")]
    swa_ring: bool,
    /// Model engine, Gemma 4: global layers without a value projection (31B) keep only the
    /// keys' RoPE-rotated dimensions in the KV cache: their other key dimensions are the
    /// values times the key norm's weight, which attention applies to the queries instead.
    /// About half the global layers' KV bytes, so more context or sequences fit; the cache
    /// takes --cache-type-v's type for them. Off (default): the plain cache. Other models: logs
    /// one line and changes nothing.
    #[arg(long)]
    gemma4_compact_global_kv: bool,
    /// Model engine: the K cache's element type (llama.cpp's names). Quantized caches are
    /// read by the gather attention path, the paged kernels (q8_0, q4_0 on sm_80+) and
    /// qwen4exp's sparse attention kernels (q8_0, q4_0 for both K and V, every GPU: about half
    /// the KV bytes of F16, which the expert cache takes). tq3 / tq4: TurboQuant (#552, 3 or
    /// 4 bits per value and an F16 norm per head, about 0.20 / 0.26 of F16's bytes), read by
    /// Legion's own paged kernels on every GPU from sm_70 and by the gather path; head sizes
    /// 128, 256 or 512; not with --tp > 1, --kv-slots on or the sparse attention kernels.
    #[arg(long, value_enum, default_value_t, env = "LEGION_CACHE_TYPE_K")]
    cache_type_k: legion::model::core::KvType,
    /// Model engine: the V cache's element type.
    #[arg(long, value_enum, default_value_t, env = "LEGION_CACHE_TYPE_V")]
    cache_type_v: legion::model::core::KvType,
    /// Model engine, several GPUs: a prefill chunk longer than this runs as micro-chunks of
    /// this many tokens, pipelined over the GPUs (0: never).
    #[arg(long, default_value_t = legion::engine::lm::PREFILL_CHUNK)]
    pipeline_micro: usize,
    /// Model engine, several GPUs: tokens per step of a prompt prefilling alone, run as
    /// --pipeline-micro micro-chunks so the GPUs work at once (0: steps of a chunk).
    #[arg(long, default_value_t = legion::engine::lm::PIPELINE_PROMPT)]
    pipeline_prompt: usize,
    /// Model engine with --prefill-staging: a prompt with at least this many tokens left
    /// borrows the expert cache's VRAM while it is read (0: never).
    #[arg(long, default_value_t = LongPrompts::default().min_tokens)]
    long_prompt_tokens: usize,
    /// Model engine with --prefill-staging: a shorter prompt than --long-prompt-tokens with at
    /// least this many tokens left borrows the expert cache too, for staging only: its steps
    /// keep the chunks they have without a lend, so the same outputs (0: never). Default: 128
    /// for GLM-5.3-Flash over several GPUs, where a short prompt's in-place expert reads are
    /// slow, 0 for other models.
    #[arg(long, value_name = "N")]
    short_prompt_tokens: Option<usize>,
    /// Model engine: tokens per forward pass while a long prompt borrows the cache.
    #[arg(long, default_value_t = LongPrompts::default().batch_tokens)]
    long_batch_tokens: usize,
    /// Model engine: tokens per forward pass while a long prompt borrows the cache and no
    /// sequence is decoding (a prompt alone is read in as few passes as possible; set it to
    /// --long-batch-tokens for the earlier behaviour).
    #[arg(long, default_value_t = LongPrompts::default().alone_tokens)]
    long_alone_tokens: usize,
    /// Model engine, several GPUs: the pipeline micro-chunk while a long prompt borrows the
    /// cache.
    #[arg(long, default_value_t = LongPrompts::default().micro)]
    long_pipeline_micro: usize,
    /// Model engine, several GPUs: while a long prompt borrows the cache, a step's
    /// micro-chunks aim at about this many per GPU (smaller ones, down to 2048 tokens, for
    /// mid-size prompts; 0: always --long-pipeline-micro).
    #[arg(long, default_value_t = LongPrompts::default().micro_per_stage)]
    long_micro_per_stage: f32,
    /// Model engine, several GPUs: while a long prompt borrows the cache, cut each step into
    /// micro-chunks of about equal size (whole 64-token blocks) instead of full ones and a
    /// short last one, which still stages nearly every expert (a 123K prompt over 2 GPUs: 31
    /// rounds of ~4000 tokens instead of 30 of 4096 and one of 312). Off by default: other
    /// chunk boundaries change the float summation order, so greedy outputs can differ from
    /// the default's. Use it for faster long-prompt prefill when that is acceptable.
    #[arg(long)]
    long_micro_even: bool,
    /// Model engine, several GPUs: while a long prompt borrows the cache, the last micro-chunks
    /// of a step ramp down from --long-pipeline-micro to this many tokens (halving), so the
    /// last one passes the later GPUs sooner (the pipeline drains in about a micro-chunk time
    /// per GPU); the ones before are of about equal size. 0: off (the default), as other chunk
    /// boundaries change the float summation order, so greedy outputs can differ from the
    /// default's.
    #[arg(long, value_name = "TOKENS", default_value_t = 0)]
    long_micro_taper: usize,
    /// Model engine, several GPUs: while a long prompt borrows the cache, each GPU starts its
    /// next micro-chunk as soon as its own inputs and the residual it takes are ready, rather
    /// than all of them in rounds that wait for the slowest one. Same outputs.
    #[arg(long)]
    long_dataflow: bool,
    /// Model engine, with --long-dataflow on CUDA: turn on peer access between each two
    /// neighbouring stages' GPUs, so a stage's residual goes straight into the next GPU's
    /// mailbox instead of through host memory (driver-staged). Same bytes, same outputs.
    #[arg(long)]
    long_dataflow_peer: bool,
    /// Model engine, with --long-dataflow on CUDA: hand each stage's residual over without
    /// the host waiting for it (a copy stream behind an event, into a second output buffer;
    /// the next stage's graph waits on the event), and queue a stage's next micro-chunk while
    /// its graph still runs (stages that read nothing back). Off by default; use it for long
    /// prompts over 3-4 GPUs. Same kernels and bytes, same outputs. Works with or without
    /// --long-dataflow-peer.
    #[arg(long)]
    long_dataflow_async: bool,
    /// Model engine: KV cache size in tokens, shared by all sequences and the prefix cache
    /// (default: max-seqs x ctx-size, or what fits in device memory; with `--experts cache`
    /// at most the larger of ctx-size and 32768, so VRAM goes to expert slots).
    #[arg(long)]
    kv_cache_tokens: Option<usize>,
    /// Model engine: where the KV cache lives: `device` (GPU memory, sized to fit) or `managed`
    /// (CUDA managed memory that may exceed it, paged by the driver, as llama.cpp's unified
    /// memory: a capacity baseline; needs GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 set, and lets
    /// --kv-cache-tokens size the cache under --tp too).
    #[arg(long, value_enum, default_value_t, env = "LEGION_KV_MEMORY")]
    kv_memory: legion::engine::lm::KvMemory,
    /// Model engine: KV cache tokens past what fits in GPU memory, in a host tier: pinned
    /// memory on each GPU's NUMA node, read over the link (0: none).
    /// Not given: automatic, a tier only when the context (--ctx-size, or the model's native
    /// one) does not fit in a discrete GPU's memory, sized to the rest of the cache and what
    /// the GPUs' NUMA nodes have free (the startup log says so and what it costs). A node's
    /// free memory is its MemFree plus the page cache the kernel can drop for the pinned
    /// tier (Active(file) + Inactive(file) less Mapped, Dirty and Writeback, at least 0), at
    /// most MemTotal, less a margin of 16 GiB or 10% of the node, whichever is more. Long
    /// contexts that reach the tier pay for the link (~68.5 GB/s on an AC922's NVLink, PCIe
    /// about 3x slower): every host block is read once per decoded token.
    #[arg(long, env = "LEGION_KV_HOST_TOKENS")]
    kv_host_tokens: Option<usize>,
    /// Model engine, with --kv-host-tokens: tokens of GPU staging slots (taken from the GPU's
    /// KV blocks) that prompt steps copy the host tier's blocks into, so a step reads each host
    /// block over the link once rather than once per 64 query rows; steps reaching more host
    /// tokens run in pieces of this many (a piece costs a launch and a pass over the output,
    /// little next to the copy, so small slots are cheap). 0: read the host blocks in place.
    #[arg(long, default_value_t = 16384, env = "LEGION_KV_STAGE_TOKENS")]
    kv_stage_tokens: usize,
    /// Model engine, under --tp: per-sequence KV slots. Each of --max-seqs sequences gets its
    /// own --ctx-size range of the cache, its first part in GPU memory and the rest in the
    /// host tier; attention reads it in place instead of gathering it into the compute buffers
    /// (whose size then no longer grows with the context). auto: on when the context needs a
    /// host tier and the model reads slots (Gemma 4).
    #[arg(long, value_enum, default_value_t)]
    kv_slots: legion::engine::lm::KvSlots,
    /// Model engine, with --kv-slots and a KV host tier: prompt steps of at least N tokens per
    /// sequence copy the sequence's keys and values into GPU memory before attention when its
    /// context reaches past the slot's GPU part, so the host tier's part crosses the link once
    /// per layer instead of once per tile of query rows (faster long prompts). Same output
    /// bits and memory: only contexts whose copy fits in the compute buffers sized for the
    /// full --ctx-size are copied (logged as prompt_copy_tokens). Default 64; 0 reads the host
    /// part in place.
    #[arg(long, value_name = "N", default_value_t = legion::engine::lm::KV_SLOT_STAGE_ROWS)]
    kv_slot_stage_rows: usize,
    /// Model engine: saved recurrent states for prefix caching (one per cached prefix point,
    /// ~50 MB each on a 4B model; default 8); 0 turns prefix caching off.
    #[arg(long, value_name = "N")]
    prefix_cache_states: Option<usize>,
    /// Model engine, prefix caching on a model with recurrent layers (Qwen3.5+, Flash-Next,
    /// Nemotron-H, GLM): also save the recurrent state every N prompt tokens (rounded up to
    /// whole 64-token KV blocks), so the second request on a new shared prefix (a system
    /// prompt) resumes at the last point before it branches instead of reading it all again.
    /// On Qwen3.5+ and Flash-Next (one machine, saved states in the GPU's memory) the prefill
    /// step saves the state at a point inside it as it runs, and the steps stay those of a run
    /// without the interval; elsewhere (or at a point not 64 tokens into a step) each point
    /// ends a prefill step, one more step per point. 0 (default): states only at branch points,
    /// prompt ends and reply ends. Try 2048-4096 for many requests sharing a system prompt.
    #[arg(long, value_name = "N", default_value_t = 0)]
    prefix_cache_interval: usize,
    /// Model engine, prefix caching on a model with sliding-window rings (Gemma 4): copies of
    /// sequences' rings (context checkpoints) kept in pinned host memory, so a new prompt
    /// resumes a cached conversation where it leaves it, not only near the end of its last
    /// prompt. Each copy holds one ring (~1.2 GiB on Gemma 4 31B with --tp 2). Default: 3 per
    /// --max-seqs where a third of the free host memory holds them, at least 1 per sequence.
    /// 0: off, prefix caching as before checkpoints (one copy per sequence at its last prompt's
    /// end, no --ring-checkpoint-every copies, a side request may take over a chat's cache, no
    /// host KV tier memory taken).
    #[arg(long, value_name = "N")]
    ring_checkpoints: Option<usize>,
    /// Model engine, with ring checkpoints: also save one every N prompt tokens (where a
    /// prefill step ends past it, so the steps stay those of a run without them) over the last
    /// --ring-checkpoint-span tokens of each prompt, besides every prompt's end. 0: prompt ends
    /// only.
    #[arg(long, value_name = "N", default_value_t = legion::engine::lm::RING_CHECKPOINT_EVERY)]
    ring_checkpoint_every: usize,
    /// Model engine, with ring checkpoints: how far back from a prompt's end
    /// --ring-checkpoint-every saves.
    #[arg(long, value_name = "N", default_value_t = legion::engine::lm::RING_CHECKPOINT_SPAN)]
    ring_checkpoint_span: usize,
    /// Model engine, debug: log, for a prompt that resumes nothing from the finished
    /// sequences kept for prefix caching, the tokens it shares with each and its checkpoints.
    #[arg(long)]
    ring_checkpoint_log: bool,
    /// Model engine, debug (#46): with --prefix-cache-interval, end a prefill step at every
    /// point as before in-step saves (one more step per point), even where the step could save
    /// the state inside it. Off by default; for comparing the two ways' bits.
    #[arg(long)]
    prefix_capture_cut: bool,
    /// Model engine, debug (#46): log a digest of every saved recurrent state (per recurrent
    /// layer its conv history and delta-rule state, and the token history) when it is saved
    /// and when a prompt resumes from it. Each digest reads the state back from the GPU and
    /// waits for it: for exactness checks only. Off by default.
    #[arg(long)]
    prefix_state_digests: bool,
    /// Model engine: the same bytes for the same request, from run to run and whatever the
    /// server served before (greedy or seeded sampling; see docs/serving.md, "Deterministic
    /// output"). Turns prefix caching off (a reused prefix makes the rest of the prompt run in
    /// other steps, so other sums) and fixes the speculative depth as --spec-fixed does (the
    /// adaptive depth follows measured step times); plans every step graph's memory afresh
    /// (a graph's layout otherwise follows the graphs built before it, and ggml-cuda's fused
    /// ops follow the layout, #459); refuses the expert matvec's in-place modes 1 and 2 (their
    /// sums follow the expert cache's contents); with --glm-pair-balance or --glm-pair-split
    /// (which split a GPU pair's work by the cached experts) each step waits for the cache's
    /// copies and refills, so its contents follow the steps, not their timing (the same bytes
    /// then also need a fixed --expert-cache-gib, and the same requests served before). Costs: a
    /// repeated prefix is read again, and the drafts do not adapt. Requests decoded at the same time share steps whose size
    /// changes the sums (#49), so it also admits requests in waves (see
    /// --deterministic-gather-ms): requests sent together give the same bytes every run, and
    /// a request sent while others run waits for them to finish.
    #[arg(long)]
    deterministic: bool,
    /// With --deterministic: an idle server gathers requests until none has arrived for this
    /// many milliseconds (at most 20 times as long), admits them together in a fixed order (by
    /// prompt, then limits; equal requests in arrival order), and holds requests that arrive
    /// while they run until all of them are done. Default 200 ms (each wave's first request
    /// waits that long; a client that sends the next request once one finishes must make it
    /// within the window, ~50 ms measured on a POWER9 host). 0: admit requests as they arrive,
    /// as without --deterministic; then requests decoded together follow their arrival timing.
    #[arg(
        long,
        value_name = "MS",
        default_value_t = 200,
        requires = "deterministic"
    )]
    deterministic_gather_ms: u64,
    /// An idle server that gets a request first waits until no other request has arrived for
    /// this many milliseconds (at most 20 times as long), then admits them all in arrival
    /// order, so requests sent together (a burst of users) start their prompts in the same
    /// step instead of the first one alone and the rest a step later (up to --prefill-slots
    /// per step). Unlike --deterministic it never holds a request that arrives while others
    /// run. Costs each burst's first request up to that long; batches change, so with several
    /// users greedy text can differ. Two DGX Sparks (Qwen3.8-Flash-Next NVFP4 TP, 5 ms): 16
    /// users 231.5 / 234.0 -> 233.0 / 236.7 tok/s, 4 users 110.0 / 111.7 -> 111.5 / 113.9.
    /// 0 (default): admit requests as they arrive.
    #[arg(
        long,
        value_name = "MS",
        default_value_t = 0,
        conflicts_with = "deterministic"
    )]
    admit_gather_ms: u64,
    /// Model engine: speculative decoding; the drafter that proposes tokens for the model to
    /// verify in one step.
    #[arg(long, value_enum, default_value_t = SpecArg::None)]
    spec: SpecArg,
    /// Model engine: max draft tokens per sequence and step (with `--spec`; default: 2 for
    /// qwen4exp, 8 on an integrated GPU such as the GB10; 8 for qwen35 MTP on an integrated GPU;
    /// 8 for Gemma 4; 4 for other models).
    #[arg(long)]
    spec_k: Option<usize>,
    /// Model engine: drafter weights, for `--spec draft|eagle3|dflash|dspark` (and `mtp` when
    /// the MTP head is a GGUF of its own; without it, `--spec mtp` on a Qwen3.8-Flash-Next
    /// safetensors checkpoint runs the checkpoint's own `mtp.*` head as shipped).
    #[arg(long)]
    spec_draft: Option<PathBuf>,
    #[command(flatten)]
    draft_settings: draft_settings::DraftSettings,
    #[command(flatten)]
    adaptive_settings: adaptive_settings::AdaptiveSettings,
    /// Model engine, `--spec mtp`: stop a sequence's drafts at the first one the MTP head
    /// gives less than this probability. Without `--spec-fixed` it is a ceiling: the depth
    /// policy lowers it to what a draft pass costs against the time per token.
    #[arg(long, default_value_t = 0.5)]
    spec_p_min: f32,
    /// Model engine, `--spec mtp`: where the sequence's last N or more tokens occurred earlier
    /// in its context (a prompt it repeats or quotes, code it edits), draft the continuation of
    /// that occurrence instead of the MTP head's drafts, up to --spec-k deep and without head
    /// passes. The depth policy takes it only where its learned acceptance, by match length,
    /// gives more tokens per millisecond than the head's drafts (with --spec-fixed: whenever
    /// there is a match). 0 (default): off. Recommended on a GB10 (Qwen3.8-Flash-Next): 4,
    /// with --spec-mtp-ngram-k 16, where output copies its input (recall 74 -> 135 tok/s,
    /// code edits 78 -> 113, quoting 77 -> 138); prose, code and chats are unchanged. Every draft is verified,
    /// but greedy text is not byte-identical to the head's alone: other drafts in a verify
    /// step can flip near-tied tokens, as the adaptive depth's do (#327).
    #[arg(long, value_name = "N", default_value_t = 0)]
    spec_mtp_ngram: usize,
    /// Model engine, with --spec-mtp-ngram: the most n-gram drafts a step verifies (1-16;
    /// default: --spec-k). Deeper than --spec-k, the head still drafts at most --spec-k and
    /// only steps the n-gram drafts in verify more rows (with --spec-fixed every step does).
    #[arg(long, value_name = "K", value_parser = clap::value_parser!(u8).range(1..=16))]
    spec_mtp_ngram_k: Option<u8>,
    /// Model engine, the MTP heads of qwen4exp and qwen35 (`--spec mtp`) and DFlash drafters that
    /// borrow the target's output head (`--spec dflash`): the head scores only the first this many
    /// tokens of the vocabulary when it drafts (0, the default: all), so a draft pass reads that
    /// many rows of the output head instead of all 248,320 on the Qwen3.8 models (see also
    /// --spec-draft-lm-head). Drafts only: every token is verified by the full model, so the output
    /// does not change; a draft outside the set is never proposed. Recommended: 65536 (GB10:
    /// Flash-Next ~2 ms less per draft pass, chats 38.8 / 37.1 / 37.0 -> 40.8 / 40.5 / 41.8 tok/s;
    /// the 27B cycle 117-127 -> 108-113 ms, +3-9% by prompt class). Smaller sets (32768 and below)
    /// cut acceptance more than they save. The head's probability is then over those tokens
    /// (higher), which --spec-p-min and the depth policy see. See docs/tuning-integrated-gpus.md.
    #[arg(long, env = "LEGION_SPEC_DRAFT_VOCAB", default_value_t = 0)]
    spec_draft_vocab: usize,
    /// Model engine, `--spec mtp`: the MTP head attends to the last this many positions only
    /// (0: every one), as Strata's `--mtp-window` (its default 32768). Long contexts then cost
    /// the head less per draft (Qwen3.8-Flash-Next on the GB10 at 62K: 1.4 ms of dense attention
    /// per pass). Verification stays exact. Paged attention (CUDA) only.
    /// Default: 32768 for qwen4exp on an integrated GPU (the GB10), else 0.
    #[arg(
        long,
        env = "LEGION_MTP_WINDOW",
        value_parser = clap::value_parser!(u32).range(0..=i32::MAX as i64)
    )]
    mtp_window: Option<u32>,
    /// Model engine, `--spec mtp`: which weights the MTP head drafts with. `q8_0` makes a Q8_0
    /// copy of the head's BF16/F16/F32 matrices at load (~1 s) and drafts with it, which halves
    /// what a draft pass reads of a 16-bit head (Qwen3.8-27B-NVFP4: 810 -> 430 MiB; GB10, k=4:
    /// cycle 108-113 -> 102-106 ms, +5-10% by prompt class, acceptance unchanged). Drafts only:
    /// every token is verified by the full model, so the output does not change. Nothing to
    /// gain for a head shipped quantized (Flash-Next's Q8_0 head). `f16` makes an F16 copy of
    /// the head's BF16 and F32 matrices instead (exact for BF16): for GPUs without BF16 tensor
    /// cores (V100), which run BF16/F32 GEMMs on F32 paths; drafts only, opt-in. Flash-Next's
    /// Q8_0 head has only its F32 router and BF16 indexer to copy (8 MiB): no measured change
    /// on 2x V100. `auto` (default): `q8_0` on an integrated GPU for qwen35 / qwen35moe heads
    /// and for a qwen4exp head shipped 16-bit (Flash-Next-NVFP4's own head: 175 -> 92 MiB),
    /// `model` (as shipped) otherwise. Replaces LEGION_MTP_HEAD_TYPE. See
    /// docs/tuning-integrated-gpus.md.
    #[arg(long, value_enum, default_value_t = DraftHeadType::Auto)]
    spec_draft_head_type: DraftHeadType,
}

impl EngineArgs {
    /// Refuses switches that make `--deterministic`'s output follow the expert cache's contents
    /// (`--expert-mv-vram` 1 and 2 sum a cached expert in another order than an uncached one).
    fn check_deterministic(&self, switches: &Switches) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.deterministic || switches.expert_mv_vram.unwrap_or(0) == 0,
            "--deterministic needs --expert-mv-vram 0: modes 1 and 2 make the output follow \
             which experts the cache holds"
        );
        anyhow::ensure!(
            !self.deterministic || self.prefix_cache_states.unwrap_or(0) == 0,
            "--deterministic turns prefix caching off: --prefix-cache-states must be 0 or left out"
        );
        Ok(())
    }

    /// `--deterministic-gather-ms`: the wave admission's gather window, `None` without
    /// `--deterministic` or with 0.
    fn deterministic_gather(&self) -> Option<std::time::Duration> {
        (self.deterministic && self.deterministic_gather_ms > 0)
            .then(|| std::time::Duration::from_millis(self.deterministic_gather_ms))
    }

    /// `--admit-gather-ms`: the idle burst window, `None` at 0.
    fn admit_gather(&self) -> Option<std::time::Duration> {
        (self.admit_gather_ms > 0).then(|| std::time::Duration::from_millis(self.admit_gather_ms))
    }

    /// The expert cache's settings: the model's flags', plus `--deterministic` (with GLM's pair
    /// balance the cache's contents then follow the steps alone, #1165).
    fn residency(&self) -> ResidencyConfig {
        ResidencyConfig {
            deterministic: self.deterministic,
            ..self.model.residency()
        }
    }

    /// `--prefix-cache-states`, 0 with `--deterministic`.
    fn prefix_cache_states(&self) -> usize {
        if self.deterministic {
            0
        } else {
            self.prefix_cache_states.unwrap_or(8)
        }
    }

    fn spec(&self) -> anyhow::Result<SpecConfig> {
        let kind = self.spec.kind();
        anyhow::ensure!(
            self.spec_mtp_ngram == 0 || kind == SpecKind::Mtp,
            "--spec-mtp-ngram needs --spec mtp"
        );
        anyhow::ensure!(
            self.spec_mtp_ngram_k.is_none() || self.spec_mtp_ngram > 0,
            "--spec-mtp-ngram-k needs --spec-mtp-ngram"
        );
        anyhow::ensure!(
            self.draft_settings.spec_draft_weights == legion::spec::DraftWeights::Model
                || matches!(kind, SpecKind::Dflash | SpecKind::Dspark),
            "--spec-draft-weights needs --spec dflash|dspark"
        );
        anyhow::ensure!(
            self.draft_settings.spec_draft_lm_head != legion::spec::DraftLmHead::Q4
                || matches!(kind, SpecKind::Dflash | SpecKind::Mtp),
            "--spec-draft-lm-head needs --spec dflash|mtp"
        );
        anyhow::ensure!(
            !self.draft_settings.spec_draft_batched_attn || kind == SpecKind::Dflash,
            "--spec-draft-batched-attn needs --spec dflash"
        );
        if kind == SpecKind::None {
            anyhow::ensure!(
                self.spec_draft.is_none(),
                "--spec-draft needs --spec mtp|draft|eagle3|dflash|dspark"
            );
            return Ok(SpecConfig::default());
        }
        anyhow::ensure!(self.spec_k != Some(0), "--spec-k must be at least 1");
        anyhow::ensure!(
            !kind.needs_weights() || self.spec_draft.is_some(),
            "--spec {} needs --spec-draft <weights>",
            kind.name()
        );
        anyhow::ensure!(
            kind.takes_weights() || self.spec_draft.is_none(),
            "--spec {} takes no --spec-draft",
            kind.name()
        );
        Ok(SpecConfig {
            kind,
            // 0: the model's default (see the engine).
            k: self.spec_k.unwrap_or(0),
            fixed: self.adaptive_settings.spec_fixed || self.deterministic,
            max_batch: self.adaptive_settings.spec_max_batch,
            gate_cost_model: self.adaptive_settings.spec_gate_cost_model.unwrap_or(true),
            gate_cost_model_given: self.adaptive_settings.spec_gate_cost_model.is_some(),
            row_cost: self.adaptive_settings.spec_row_cost,
            pad_longest: self.adaptive_settings.spec_pad_longest,
            max_rows: self.adaptive_settings.spec_max_rows,
            p_floor: self.adaptive_settings.spec_p_floor,
            draft: self.spec_draft.clone(),
            draft_temperature: self.draft_settings.spec_draft_temperature,
            min_confidence: self.draft_settings.spec_min_confidence,
            // 0: the drafter's default
            draft_window: self
                .draft_settings
                .spec_draft_window
                .map_or(0, |w| w as usize),
            draft_weights: self.draft_settings.spec_draft_weights,
            draft_lm_head: self.draft_settings.spec_draft_lm_head,
            draft_batched_attn: self.draft_settings.spec_draft_batched_attn,
            p_min: self.spec_p_min,
            mtp_ngram: self.spec_mtp_ngram,
            mtp_ngram_k: self.spec_mtp_ngram_k.map_or(0, usize::from),
            ..Default::default()
        })
    }
}

/// Flags that end up in [`ServerConfig`].
#[derive(clap::Args)]
struct ServerArgs {
    /// PEM certificate chain for HTTPS. Requires --tls-key.
    #[arg(long, env = "LEGION_TLS_CERT", value_name = "PEM")]
    tls_cert: Option<PathBuf>,
    /// PEM PKCS#8 or RSA private key for HTTPS. Requires --tls-cert.
    #[arg(long, env = "LEGION_TLS_KEY", value_name = "PEM")]
    tls_key: Option<PathBuf>,
    /// max_tokens for OpenAI chat requests that set none.
    #[arg(long, default_value_t = 4096)]
    default_max_tokens: u32,
    /// Require this API key (`Authorization: Bearer <key>` or `x-api-key: <key>`) on every
    /// route except GET /health. May be given several times; any key matches. Without a key
    /// the server stays open, as before.
    #[arg(long, env = "LEGION_API_KEY")]
    api_key: Vec<String>,
    /// Allow this origin in CORS responses, e.g. `http://localhost:3000` (repeatable); `*`
    /// allows any origin. Browser frontends on another origin need this. Without it the
    /// server sends no CORS headers at all.
    #[arg(long, value_name = "ORIGIN")]
    cors: Vec<String>,
    /// Do not serve the built-in web UI at `/`.
    #[arg(long)]
    no_webui: bool,
    /// Refuse request bodies larger than this many megabytes with 413. Long prompts need
    /// more than axum's built-in 2 MB default.
    #[arg(long, default_value_t = 64)]
    max_request_mb: usize,
    /// On SIGINT/SIGTERM, wait this many seconds for in-flight requests, then exit anyway.
    #[arg(long, default_value_t = 30)]
    shutdown_timeout_secs: u64,
    /// Name to serve the model under; repeatable. Replaces the engine id in `/v1/models`.
    #[arg(long = "served-model-name")]
    served_model_name: Vec<String>,
    /// Defaults for the sampling parameters a request leaves out: `model` takes the model's own
    /// (its GGUF's `general.sampling.*` keys or `generation_config.json`, else its
    /// architecture's published values), then OpenAI's; `openai` takes OpenAI's (temperature 1,
    /// top_p 1, no top-k). A request's own fields always win, `temperature: 0` included. The
    /// startup log lists the effective defaults and their sources.
    #[arg(long, value_enum, env = "LEGION_SAMPLING_DEFAULTS", default_value_t = DefaultsMode::Model)]
    sampling_defaults: DefaultsMode,
    /// Default temperature for requests that set none, over the model's (0..=2; 0 is greedy).
    #[arg(long)]
    default_temperature: Option<f32>,
    /// Default top_p, over the model's (0 < p <= 1).
    #[arg(long)]
    default_top_p: Option<f32>,
    /// Default top_k, over the model's (0 disables it).
    #[arg(long)]
    default_top_k: Option<u32>,
    /// Default min_p, over the model's (0..=1; 0 disables it).
    #[arg(long)]
    default_min_p: Option<f32>,
    /// Default presence_penalty, over the model's (-2..=2).
    #[arg(long, allow_hyphen_values = true)]
    default_presence_penalty: Option<f32>,
    /// Default frequency_penalty, over the model's (-2..=2).
    #[arg(long, allow_hyphen_values = true)]
    default_frequency_penalty: Option<f32>,
    /// Default repetition_penalty, over the model's (> 0; 1 disables it).
    #[arg(long)]
    default_repetition_penalty: Option<f32>,
    /// Without the live status line, log a "server stats" event every N seconds (prefill,
    /// decode and queued sequences, token counts, KV use, expert cache hits, MTP acceptance).
    /// Off by default; the live status line redraws every 350 ms instead.
    #[arg(long, value_name = "SECONDS")]
    stats_interval: Option<u64>,
    /// One live status line under the logs, redrawn in place every 350 ms (sequences in
    /// prefill and decode, queued, prompt and generation tok/s, KV use, expert hits, MTP
    /// acceptance): auto shows it when stdout is a terminal that takes ANSI sequences (not
    /// TERM=dumb); on forces it; never with --log-format json.
    #[arg(long, value_enum, default_value_t)]
    status_line: StatusLineChoice,
    /// The live dashboard in this terminal: requests, throughput, link and memory gauges,
    /// GPUs, speculative decoding, KV and expert caches, latency, and the logs, which go into
    /// its log panel instead of stdout (and the status line is off). `q` closes it and the
    /// server keeps running with plain logs; Ctrl-C stops the server. Without a terminal on
    /// stdout it logs a warning and serves without it. `legion top` shows the same from
    /// another terminal or machine.
    #[arg(long)]
    tui: bool,
    /// Sample the GPUs' hardware counters while serving and export them in /metrics as
    /// legion_gpu_* (NVML: NVLink data bytes over every link, PCIe throughput, memory and SM
    /// busy time), next to the analytic model bytes (legion_engine_model_*), for profiling
    /// runs and `legion bench --workload bandwidth`. A background thread queries NVML every
    /// --hw-counters-interval-ms; nothing runs on the GPUs. Off by default.
    #[arg(long)]
    hw_counters: bool,
    /// How often --hw-counters samples, in milliseconds.
    #[arg(
        long,
        value_name = "MS",
        default_value_t = 200,
        requires = "hw_counters"
    )]
    hw_counters_interval_ms: u64,
    /// Register this server with an LLM proxy (simple-llm-proxy): POST its address, port and
    /// first --api-key to `<URL>/register` (the URL may end in `/register` or not) once the
    /// model is loaded and the server listens, then again every --registry-interval seconds.
    /// Failures are logged and retried with backoff; they never stop the server. Repeatable,
    /// one registration per proxy. Off by default.
    #[arg(long, value_name = "URL", value_parser = registry::register_url)]
    registry_api: Vec<reqwest::Url>,
    /// With --registry-api: the IP address the proxy should reach this server on. Default:
    /// --host when it is a specific address, else the local address of the route to the proxy.
    /// Set it when the proxy must use another address (NAT, a second interface).
    #[arg(long, value_name = "IP", requires = "registry_api")]
    registry_host: Option<IpAddr>,
    /// With --registry-api: seconds between registrations after a success, so a proxy that
    /// restarted without its host list picks the server up again.
    #[arg(
        long,
        value_name = "SECONDS",
        default_value_t = 300,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    registry_interval: u64,
    /// Share batch slots fairly between clients, so one client's flood of requests cannot
    /// starve the others. `off` (default): no fair share. `api-key`: clients are told apart
    /// by their API key (`Authorization: Bearer` or `x-api-key`; with --api-key only accepted
    /// keys, without it any key a client sends); `user`: by the request's `user` field
    /// (Anthropic: `metadata.user_id`). Among waiting requests of equal priority the client
    /// holding the fewest running sequences goes next (round robin on ties), and while
    /// several clients wait none holds more than its share of the slots. Requests without a
    /// key count as one client. Keys are never logged; logs show a short per-process hash.
    #[arg(long, value_enum, default_value_t = FairShare::Off)]
    fair_share: FairShare,
}

impl ServerArgs {
    fn config(&self) -> anyhow::Result<ServerConfig> {
        let tls = match (&self.tls_cert, &self.tls_key) {
            (Some(cert), Some(key)) => Some(TlsConfig {
                cert: cert.clone(),
                key: key.clone(),
            }),
            (Some(_), None) => anyhow::bail!("--tls-cert requires --tls-key"),
            (None, Some(_)) => anyhow::bail!("--tls-key requires --tls-cert"),
            (None, None) => None,
        };
        let check = |flag: &str, v: Option<f32>, ok: fn(f32) -> bool| match v {
            Some(x) if !ok(x) => Err(anyhow::anyhow!("--{flag} {x} is out of range")),
            _ => Ok(v),
        };
        let sampling_overrides = SamplingLayer {
            temperature: check("default-temperature", self.default_temperature, |t| {
                (0.0..=2.0).contains(&t)
            })?,
            top_p: check("default-top-p", self.default_top_p, |p| p > 0.0 && p <= 1.0)?,
            top_k: self.default_top_k,
            min_p: check("default-min-p", self.default_min_p, |p| {
                (0.0..=1.0).contains(&p)
            })?,
            presence_penalty: check(
                "default-presence-penalty",
                self.default_presence_penalty,
                |p| (-2.0..=2.0).contains(&p),
            )?,
            frequency_penalty: check(
                "default-frequency-penalty",
                self.default_frequency_penalty,
                |p| (-2.0..=2.0).contains(&p),
            )?,
            repetition_penalty: check(
                "default-repetition-penalty",
                self.default_repetition_penalty,
                |p| p > 0.0 && p.is_finite(),
            )?,
        };
        Ok(ServerConfig {
            tls,
            default_max_tokens: self.default_max_tokens,
            api_keys: self.api_key.clone(),
            cors_origins: self.cors.clone(),
            webui: !self.no_webui,
            max_request_mb: self.max_request_mb,
            shutdown_timeout_secs: self.shutdown_timeout_secs,
            served_model_names: self.served_model_name.clone(),
            sampling_mode: self.sampling_defaults,
            sampling_overrides,
            hw_counters: self.start_hw_counters(),
            registry: RegistryConfig {
                proxies: self.registry_api.clone(),
                host: self.registry_host,
                interval: Duration::from_secs(self.registry_interval),
                ..Default::default()
            },
            fair_share: self.fair_share,
        })
    }

    /// The hardware counters' sampler, with `--hw-counters` (a warning when NVML cannot).
    fn start_hw_counters(&self) -> Option<Arc<legion::hw::counters::HwCounters>> {
        if !self.hw_counters {
            return None;
        }
        let every = Duration::from_millis(self.hw_counters_interval_ms.max(10));
        match legion::hw::counters::HwCounters::start(every) {
            Ok(c) => {
                tracing::info!(gpus = ?c.gpus(), every = ?every, "hw counters: sampling (NVML index, NVLink links)");
                Some(Arc::new(c))
            }
            Err(e) => {
                tracing::warn!(error = %e, "--hw-counters: no hardware counters");
                None
            }
        }
    }
}

/// Applies `--numa` for a model on CUDA GPUs (#422); `None` for the mock engine and the CPU.
/// Runs before the async runtime starts, so its threads inherit the binding.
pub fn bind_numa(args: &ServeArgs) -> anyhow::Result<Option<Placement>> {
    let e = &args.engine;
    // GLM's parallel decode spins in a kernel on each GPU until the others reach the same
    // exchange. With CUDA's lazy module loading, a GPU's host thread loading a kernel on its
    // first launch can wait behind those spinning kernels: every first step of a new size stalled
    // until the exchange timed out. Loading every module at startup avoids that. Set before
    // anything initializes CUDA, while the process has one thread; an explicit setting wins.
    // `--pair-tp` spins the same way (a stage's GPU and its pair wait on each other).
    let spins = e.model.glm_parallel_decode || e.model.pair_tp;
    if spins && std::env::var_os("CUDA_MODULE_LOADING").is_none() {
        // SAFETY: called from `main` before the runtime starts: no other thread reads the
        // environment.
        unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
    }
    // The same for ggml's CUDA graphs: instantiating one between two exchanges waited behind the
    // spinning kernels on every GPU until the exchange timed out (10 s), and ggml's sweep of
    // graphs unused for 10 s then freed the one it had just instantiated (an assert in
    // ggml-cuda.cu). Run without them, the steps are no slower (25.6 ms a decode step against
    // 29-31 ms with them on GLM-5.3-Flash over 4 GPUs).
    // The stage graphs (#817, on by default since #849 unless `--no-stage-graphs`) capture
    // ggml's segments with the external ops around them, so ggml must launch their kernels
    // itself instead of its own graphs.
    if (spins || e.model.stage_graphs()) && std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_none() {
        // SAFETY: as above.
        unsafe { std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1") };
    }
    let model = e.engine.unwrap_or(if e.model.is_named() {
        EngineKind::Model
    } else {
        EngineKind::Mock
    }) == EngineKind::Model;
    if !model || !matches!(e.model.device, DeviceArg::Cuda) {
        return Ok(None);
    }
    // `off` does not even ask the driver.
    let n_gpus = match e.numa {
        NumaMode::Off => 0,
        _ => e.model.gpus.max(e.model.tp * e.model.replicas()?).max(1),
    };
    let gpus = (0..n_gpus).map(legion::hw::numa_bind::gpu_node).collect();
    Placement::apply(e.numa.clone(), gpus)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

pub async fn run(
    args: ServeArgs,
    numa: Option<Placement>,
    log_format: LogFormat,
    color_choice: ColorChoice,
) -> anyhow::Result<()> {
    tracing::info!(version = version::full(), "legion starting");
    if let Some(numa) = &numa {
        match numa.plan {
            legion::hw::numa_bind::Plan::Skip(_) => tracing::debug!("{}", numa.summary()),
            _ => tracing::info!("{}", numa.summary()),
        }
        for w in numa.warnings() {
            tracing::warn!("{w}");
        }
        // Page cache other processes left in GPU memory (#772): on an AC922 2.1 GiB of one GPU
        // took 2.6 GiB from its long-prompt expert cache.
        if let Ok(topo) = legion::hw::numa::Topology::detect() {
            for (node, bytes) in legion::hw::numa::memory_only_page_cache(&topo, 256 << 20) {
                tracing::warn!(
                    "NUMA node {node} (memory only: on an AC922 a GPU's own memory) holds {} MiB \
                     of page cache from earlier file reads, VRAM the KV pool and the expert cache \
                     cannot have; dropping the page cache (vm.drop_caches) or migrating those \
                     pages to a CPU node gives it back",
                    bytes >> 20
                );
            }
        }
    }
    let server_config = args.server.config()?;
    let e = args.engine;
    let tui = tui::usable(args.server.tui);
    let max_seqs_arg = e.max_seqs as u32;
    // The dashboard header's layout part (the model's load report), its path and drafter.
    let mut dash_layout = legion::dashboard::snapshot::Header::default();
    let mut dash_path: Option<String> = None;
    let mut dash_spec: (&'static str, u32) = ("none", 0);
    let kind = e.engine.unwrap_or(if e.model.is_named() {
        EngineKind::Model
    } else {
        EngineKind::Mock
    });
    let terminal = statusline::stdout_is_terminal();
    let term = TermEnv::from_env();
    let json = log_format.is_json();
    // The dashboard draws the status line's figures itself.
    let live_status = !tui && args.server.status_line.live(terminal, term, json);
    let interactive = !json && color_choice.show_report(terminal);
    let color = !json && color_choice.use_color(terminal, term);
    let mut reported = false;
    let engine: Arc<dyn Engine> = match kind {
        EngineKind::Mock => Arc::new(MockEngine::new(MockConfig {
            model: e.model_name.unwrap_or_else(|| "mock".into()),
            step: Duration::from_millis(e.mock_step_ms),
            max_batch: e.max_batch,
            max_waiting: e.max_waiting,
            scheduling: e.scheduling_policy,
            ..Default::default()
        })),
        EngineKind::Model => {
            let spec = e.spec()?;
            let switches = e.model.served_switches();
            if e.model.switches.stage_graphs_given() && !e.model.stage_graphs() {
                tracing::warn!(
                    "--stage-graphs: not with --tp, --pair-tp or --glm-parallel-decode; off"
                );
            }
            e.check_deterministic(&switches)?;
            let prefix_cache_states = e.prefix_cache_states();
            let deterministic_gather = e.deterministic_gather();
            let admit_gather = e.admit_gather();
            if e.deterministic {
                tracing::info!(
                    gather_ms = e.deterministic_gather_ms,
                    "--deterministic: prefix caching off, speculative depth fixed, step graphs \
                     planned afresh, requests admitted in waves (gather_ms 0: as they arrive; \
                     then requests decoded at the same time follow their timing, #49)"
                );
                // Before the engine allocates its first graph.
                legion::ggml::set_fresh_plans(true);
            }
            let residency = e.residency();
            let model_path = e.model.resolve_model()?;
            anyhow::ensure!(e.model.ctx_size != Some(0), "--ctx-size must be at least 1");
            anyhow::ensure!(e.max_seqs > 0, "--max-seqs must be at least 1");
            let replicas = e.model.replicas()?;
            let moe_ep_place = match &e.model.tp_moe_ep_place {
                Some(p) => Some(Arc::new(
                    legion::tp_net::ep_place::Placement::load(p)
                        .map_err(|err| anyhow::anyhow!("--tp-moe-ep-place: {err}"))?,
                )),
                None => None,
            };
            if let Some(p) = &e.model.moe_route_record {
                legion::model::core::route_rec::open(p)
                    .map_err(|err| anyhow::anyhow!("--moe-route-record: {err}"))?;
            }
            if replicas > 1 {
                anyhow::ensure!(
                    matches!(e.model.device, DeviceArg::Cuda),
                    "--tp-replicas needs --device cuda"
                );
                anyhow::ensure!(
                    e.model.tp_rank.is_none() && e.model.node_rank.is_none(),
                    "--tp-replicas runs every replica in this process: not with --tp-rank or \
                     --node-rank"
                );
            }
            let device = e.model.device.choice();
            let integrated = device == DeviceChoice::Cuda && legion::engine::lm::integrated_gpu();
            let max_batch_tokens = e.max_batch_tokens.unwrap_or_else(|| {
                legion::engine::lm::default_max_batch_tokens(device, integrated)
            });
            anyhow::ensure!(
                max_batch_tokens > 0,
                "--max-batch-tokens must be at least 1"
            );
            let prefill_chunk = e.prefill_chunk.unwrap_or_else(|| {
                legion::engine::lm::default_prefill_chunk(device, integrated, max_batch_tokens)
            });
            let cfg = LmConfig {
                model_path,
                mmproj: e.model.mmproj.clone(),
                model_name: e
                    .model_name
                    .or_else(|| e.model.hfmodel.as_deref().and_then(hfmodel::repo_id)),
                device: e.model.device.choice(),
                ctx_size: e.model.ctx_size.unwrap_or(0),
                threads: e.model.threads,
                max_seqs: e.max_seqs,
                max_batch_tokens,
                prefill_slots: e.prefill_slots,
                kv_cache_tokens: e.kv_cache_tokens,
                uma_free: uma_free_bytes(e.uma_free_gib),
                kv_memory: e.kv_memory,
                kv_host_tokens: e.kv_host_tokens,
                kv_stage_tokens: e.kv_stage_tokens,
                kv_slots: e.kv_slots,
                kv_slot_stage_rows: e.kv_slot_stage_rows,
                prefix_cache_states,
                prefix_cache_interval: e.prefix_cache_interval,
                ring_checkpoints: e.ring_checkpoints,
                ring_checkpoint_every: e.ring_checkpoint_every,
                ring_checkpoint_span: e.ring_checkpoint_span,
                ring_checkpoint_log: e.ring_checkpoint_log,
                deterministic_gather,
                admit_gather,
                prefix_capture_cut: e.prefix_capture_cut,
                prefix_state_digests: e.prefix_state_digests,
                max_waiting: e.max_waiting,
                scheduling: e.scheduling_policy,
                stream_thread_seqs: e.stream_thread_seqs,
                kv_admission: e.kv_admission,
                slo: SloTargets {
                    itl_ms: e.itl_target.map(f64::from),
                    ttft_ms: e.ttft_target.map(f64::from),
                },
                spec,
                draft_vocab: e.spec_draft_vocab,
                mtp_window: e.mtp_window.map(|n| n as usize),
                draft_head_type: e.spec_draft_head_type,
                use_ref: false,
                host_experts: e.model.host_experts,
                hf_fp8: e.model.hf_fp8,
                nvfp4_act: e.model.nvfp4_act,
                fp8_act: e.model.fp8_act,
                nvfp4_mm: e.model.nvfp4_mm,
                fp16: legion::weights::Fp16Load {
                    mode: e.model.weights_fp16,
                    gates: e.model.weights_fp16_gates,
                    hc_inject_f32: e.model.hc_inject_f32,
                },
                nvfp4_a4: e.model.nvfp4_a4,
                residency,
                gpus: e.model.gpus,
                tp: e.model.tp,
                first_gpu: 0,
                prefill_chunk,
                prefill_even: e.prefill_even,
                ring_step: e.prefill_chunk,
                swa_full: e.swa_full,
                swa_ring: e.swa_ring,
                gemma4_compact_global_kv: e.gemma4_compact_global_kv,
                cache_types: legion::model::core::KvTypes {
                    k: e.cache_type_k,
                    v: e.cache_type_v,
                },
                ring_gather: false,
                cell_gather: false,
                grammar_threads: e.grammar_threads,
                pipeline_micro: e.pipeline_micro,
                pipeline_prompt: e.pipeline_prompt,
                layer_split: e.model.layer_split.clone(),
                attention: e.model.attention,
                moe_kernel: e.model.moe_kernel,
                split_batches: e.model.split_batches,
                pipeline_groups: e.model.pipeline_groups,
                pipeline_min_group: e.model.pipeline_min_group,
                pipeline_mtp_split: e.model.pipeline_mtp_split,
                split_last_extra: e.model.split_last_extra,
                node: e.model.node_rank.map(|rank| legion::node::NodeConfig {
                    rank: rank as usize,
                    peer: e.model.node_peer.clone().unwrap_or_default(),
                    split: e.model.node_split,
                }),
                tp_net: e.model.tp_rank.map(|rank| legion::tp_net::TpNetConfig {
                    rank,
                    peers: e.model.tp_peers.clone(),
                    timeout: std::time::Duration::from_secs(e.model.tp_timeout.max(1)),
                    transport: e.model.tp_transport,
                    rails: e.model.tp_rails.into(),
                    profile: e.model.tp_profile,
                    wire: e.model.tp_wire,
                    seq_par: e.model.tp_seq_par,
                    mtp_chain: e.model.tp_mtp_chain,
                    mtp_local_rows: e.model.tp_mtp_local_rows,
                    moe_ep: e.model.tp_moe_ep,
                    overlap: e.model.tp_overlap,
                    moe_ep_place: moe_ep_place.clone(),
                }),
                glm_parallel_decode: e.model.glm_parallel_decode,
                tp_per_gpu: e.model.tp_per_gpu,
                tp_pair: None,
                pair_tp: e.model.pair_tp,
                glm_exchange_peer: e.model.glm_exchange_peer,
                glm_prompt_step: e.model.glm_prompt_step,
                glm_expert_split: e.model.glm_expert_split,
                glm_expert_shares: e.model.glm_expert_shares.clone(),
                glm_parallel_mtp: e.model.glm_parallel_mtp,
                glm_mtp_overlap_seqs: e.model.glm_mtp_overlap_seqs(),
                glm_pair_balance: e.model.glm_pair_balance,
                glm_prefetch_stats: e.model.glm_prefetch_stats,
                glm_cross_balance: e.model.glm_cross_balance,
                glm_prefetch: e.model.glm_prefetch,
                glm_pair_split: e.model.glm_pair_split,
                ngram: e.model.ngram_options()?,
                expert_mv: e.model.expert_mv == ExpertMv::Legion,
                paged_prefill_m884: e.model.paged_prefill_m884,
                tu_attn_ffma: e.model.tu_attn_ffma,
                qsa_scores_ggml: e.model.qsa_scores_ggml,
                qsa_topk_ggml: e.model.qsa_topk_ggml,
                hc_post_norm_generic: e.model.hc_post_norm_generic,
                keep_load_cache: e.model.keep_load_cache,
                glm_route_bias: e.model.glm_route_bias,
                glm_prune: e.model.glm_prune()?,
                long_prompts: LongPrompts {
                    min_tokens: if e.model.prefill_staging {
                        e.long_prompt_tokens
                    } else {
                        0
                    },
                    short_min_tokens: e.short_prompt_tokens,
                    batch_tokens: e.long_batch_tokens,
                    alone_tokens: e.long_alone_tokens,
                    micro: e.long_pipeline_micro,
                    micro_per_stage: e.long_micro_per_stage,
                    even: e.long_micro_even,
                    taper: e.long_micro_taper,
                    dataflow: e.long_dataflow,
                    dataflow_peer: e.long_dataflow_peer,
                    dataflow_async: e.long_dataflow_async,
                },
                switches,
            };
            if interactive {
                let mut report = startup_report_for_model(&cfg, &server_config);
                if let Some(numa) = &numa {
                    report = report.section("NUMA", numa.lines());
                }
                statusline::write_report(&report.render(color))?;
            }
            let max_seqs = cfg.max_seqs;
            dash_path = Some(cfg.model_path.display().to_string());
            dash_spec = (cfg.spec.kind.name(), cfg.spec.k as u32);
            let mut engines = if replicas > 1 {
                tokio::task::spawn_blocking(move || load_replicas(cfg, replicas)).await??
            } else {
                vec![pipe_split::load(cfg).await?]
            };
            dash_layout = tui::layout(engines[0].load_report());
            if interactive {
                let engine = &engines[0];
                let model = engine
                    .models()
                    .first()
                    .map_or_else(String::new, |m| m.id.clone());
                let numa = legion::hw::HardwareInfo::detect().numa_nodes;
                let report = StartupReport::loaded(model).with_load(engine.load_report(), &numa);
                statusline::write_report(&report.render(color))?;
                reported = true;
            }
            if engines.len() == 1 {
                Arc::new(engines.remove(0))
            } else {
                Arc::new(legion::engine::replicas::ReplicaEngine::new(
                    engines,
                    max_seqs,
                    prefix_cache_states > 0,
                ))
            }
        }
    };
    if kind == EngineKind::Model {
        warmup::shapes(engine.as_ref(), &e.warmup_shapes).await;
    }
    if e.warmup_prompt > 0 {
        warmup::run(engine.as_ref(), e.warmup_prompt).await?;
    }
    if interactive {
        if !reported {
            let model = engine
                .models()
                .first()
                .map_or("mock", |m| m.id.as_str())
                .to_string();
            let report = StartupReport::new(model)
                .with_hardware(&legion::hw::HardwareInfo::detect(), 0)
                .section(
                    "Engine",
                    ["mock: no weights, KV cache, experts or n-gram table".to_string()],
                );
            statusline::write_report(&report.render(color))?;
        }
        let model = engine.models().into_iter().next();
        let (name, context) = model
            .as_ref()
            .map_or(("unknown", None), |m| (m.id.as_str(), m.context_length));
        let status_line = if live_status {
            "on"
        } else {
            match args.server.status_line {
                StatusLineChoice::Off => "off (--status-line off)",
                _ if json => "off (JSON logs)",
                _ if term.dumb => "off (TERM=dumb)",
                _ => "off (stdout is not a terminal)",
            }
        };
        statusline::write_report(&super::report::ready_line(
            name,
            context,
            engine.stats(),
            color,
            status_line,
        ))?;
    }
    let stats_interval = args
        .server
        .stats_interval
        .filter(|&seconds| seconds > 0)
        .map(Duration::from_secs);
    let _status = statusline::start(engine.clone(), live_status, stats_interval);
    let state = AppState::with_config(engine, server_config);
    state.stats.set_header(tui::finish(
        dash_layout,
        state
            .engine
            .models()
            .first()
            .map_or_else(|| "unknown".into(), |m| m.id.clone()),
        dash_path.as_deref(),
        max_seqs_arg,
        dash_spec,
        version::full().to_string(),
    ));
    for m in state.engine.models() {
        if let Ok(d) = state.sampling_defaults(&m.id) {
            tracing::info!(model = %m.id, "sampling defaults: {}", d.describe());
        }
    }
    let addr = tokio::net::lookup_host((args.listen.host.as_str(), args.listen.port))
        .await?
        .next()
        .ok_or_else(|| anyhow::anyhow!("cannot resolve {}", args.listen.host))?;
    let dashboard = if tui { tui::start(&state) } else { None };
    let served = server::serve(addr, state).await;
    if let Some(d) = dashboard {
        // The dashboard may still own the terminal (raw mode, alternate screen, logs muted):
        // give it back so the error or the shutdown lines show.
        d.stop();
    }
    served
}

/// `--tp-replicas`: loads `n` copies of the model, replica r on the --tp group from CUDA r×tp,
/// one after another (the meta device's tables are not safe to fill from two threads). Each
/// loads on a thread of its own placed on its GPUs' NUMA node when `--numa auto` plans per
/// GPU: the engine thread and helpers it starts inherit the placement, and the host memory
/// they touch first lands on that node.
fn load_replicas(cfg: LmConfig, n: usize) -> anyhow::Result<Vec<LmEngine>> {
    let per_gpu = legion::hw::numa_bind::per_gpu_threads();
    let mut engines = Vec::with_capacity(n);
    for r in 0..n {
        let mut cfg = cfg.clone();
        cfg.first_gpu = r * cfg.tp;
        let gpus = cfg.first_gpu..cfg.first_gpu + cfg.tp;
        let nodes: Vec<Option<u32>> = gpus.clone().map(legion::hw::numa_bind::gpu_node).collect();
        let node = replica_node(&nodes).filter(|_| per_gpu);
        tracing::info!(replica = r, ?gpus, ?node, "loading replica");
        let engine = std::thread::Builder::new()
            .name(format!("legion-replica{r}"))
            .spawn(move || {
                if let Some(node) = node
                    && let Err(e) = legion::hw::numa_bind::bind_thread_to_node(node)
                {
                    tracing::warn!(
                        replica = r,
                        node,
                        "cannot place the replica on its node: {e}"
                    );
                }
                LmEngine::load(cfg)
            })?
            .join()
            .map_err(|_| anyhow::anyhow!("replica {r}: its load panicked"))??;
        engines.push(engine);
    }
    Ok(engines)
}

/// The NUMA node of a replica whose GPUs all sit on one; `None` when they span nodes or one is
/// unknown.
fn replica_node(nodes: &[Option<u32>]) -> Option<u32> {
    let first = (*nodes.first()?)?;
    nodes.iter().all(|&n| n == Some(first)).then_some(first)
}

/// The report before the load: the hardware, the checkpoint's metadata and the settings the
/// load starts from. Where things go is printed after the load, from the load itself.
fn startup_report_for_model(cfg: &LmConfig, server: &ServerConfig) -> StartupReport {
    use legion::engine::lm::report::value_name;
    let facts = legion::engine::lm::report::inspect(cfg);
    let name = match &facts {
        Ok(f) => f.name.clone(),
        Err(_) => cfg.model_path.file_name().map_or_else(
            || cfg.model_path.display().to_string(),
            |n| n.to_string_lossy().into(),
        ),
    };
    let mut report =
        StartupReport::new(name).with_hardware(&legion::hw::HardwareInfo::detect(), cfg.threads);
    let native = facts.as_ref().ok().map(|f| f.native_context);
    match &facts {
        Ok(f) => {
            let defaults = legion::engine::sampling_defaults::SamplingDefaults::resolve(
                server.sampling_mode,
                &server.sampling_overrides,
                &f.sampling,
            );
            report = report.with_model(f, format!("sampling defaults: {}", defaults.describe()));
        }
        Err(e) => {
            report = report.section("Model", [format!("metadata unreadable: {e}")]);
        }
    }
    let context = match (cfg.ctx_size, native) {
        (0, Some(n)) => format!("{} tokens (model native)", super::report::count(n)),
        (0, None) => "model native (unknown)".into(),
        (n, _) => format!("{} tokens (--ctx-size)", super::report::count(n)),
    };
    let auto = |n: Option<usize>| n.map_or_else(|| "auto".into(), |n| n.to_string());
    let spec = if cfg.spec.kind == SpecKind::None {
        "speculative decoding off".to_string()
    } else {
        let k = if cfg.spec.k == 0 {
            "auto".to_string()
        } else {
            cfg.spec.k.to_string()
        };
        let depth = if cfg.spec.fixed {
            format!("fixed {k}")
        } else {
            format!("adaptive up to {k}")
        };
        let mut line = format!(
            "speculative decoding {} · depth {depth}",
            cfg.spec.kind.name()
        );
        if cfg.spec.kind == SpecKind::Mtp {
            line += &format!(
                " · draft vocabulary {} · MTP window {}",
                if cfg.draft_vocab == 0 {
                    "full".to_string()
                } else {
                    cfg.draft_vocab.to_string()
                },
                auto(cfg.mtp_window)
            );
            if cfg.spec.mtp_ngram > 0 {
                line += &format!(
                    " · n-gram drafts from {}-token matches, up to {}",
                    cfg.spec.mtp_ngram,
                    cfg.spec.ngram_k()
                );
            }
        }
        line
    };
    let mut lines = vec![
        format!(
            "context {context} · up to {} sequences · KV types {}/{}",
            cfg.max_seqs,
            value_name(&cfg.cache_types.k),
            value_name(&cfg.cache_types.v)
        ),
        format!(
            "KV device tokens {} · KV host tier {}",
            auto(cfg.kv_cache_tokens),
            auto(cfg.kv_host_tokens)
        ),
        spec,
        format!(
            "kernels: MoE {} · attention {}",
            value_name(&cfg.moe_kernel),
            value_name(&cfg.attention)
        ),
    ];
    // Precision options away from their defaults.
    let mut moving = Vec::new();
    if cfg.fp16.mode != Default::default() {
        moving.push(format!("--weights-fp16 {}", value_name(&cfg.fp16.mode)));
    }
    if cfg.fp8_act != Default::default() {
        moving.push(format!("--fp8-act {}", value_name(&cfg.fp8_act)));
    }
    if cfg.nvfp4_act != Default::default() {
        moving.push(format!("--nvfp4-act {}", value_name(&cfg.nvfp4_act)));
    }
    if cfg.nvfp4_a4 != Default::default() {
        moving.push(format!("--nvfp4-a4 {}", value_name(&cfg.nvfp4_a4)));
    }
    if !moving.is_empty() {
        lines.push(format!(
            "non-default precision options: {}",
            moving.join(", ")
        ));
    }
    report.section("Settings", lines)
}

/// A swap threshold: finite and not negative (a negative margin with a ratio of -1 or less
/// would let two experts trade a slot back and forth).
/// `--uma-free-gib` in bytes.
fn uma_free_bytes(gib: f32) -> usize {
    (f64::from(gib) * (1u64 << 30) as f64) as usize
}

fn non_negative(value: &str) -> Result<f32, String> {
    let v = value.parse::<f32>().map_err(|e| e.to_string())?;
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(format!("must be a finite number >= 0, got {value}"))
    }
}

/// The host's expert share: in [0, 1].
fn share_fraction(value: &str) -> Result<f32, String> {
    let v = value.parse::<f32>().map_err(|e| e.to_string())?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be in [0, 1], got {value}"))
    }
}

/// The heat decay: in (0, 1] (0 would forget every routing at once, NaN freeze the cache).
fn heat_decay(value: &str) -> Result<f32, String> {
    let v = value.parse::<f32>().map_err(|e| e.to_string())?;
    if v > 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(format!("must be in (0, 1], got {value}"))
    }
}

/// The prefill staging ring: whole buffers per expert kind, within
/// [`legion::residency::PREFILL_RING_RANGE`] (one buffer could not overlap a copy with compute).
fn prefill_ring(value: &str) -> Result<usize, String> {
    let v = value.parse::<usize>().map_err(|e| e.to_string())?;
    let range = legion::residency::PREFILL_RING_RANGE;
    if range.contains(&v) {
        Ok(v)
    } else {
        Err(format!(
            "must be {} to {}, got {value}",
            range.start(),
            range.end()
        ))
    }
}

#[cfg(test)]
mod tests;
