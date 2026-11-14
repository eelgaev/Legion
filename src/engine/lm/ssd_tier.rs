//! Whether a model loads with the SSD expert tier ([`crate::residency::ssd_experts`]) and the
//! tier's size.

use crate::ggml::DeviceKind;
use crate::gguf::GgufFile;
use crate::model::core::{Hparams, KV_BLOCK, KvPool};
use crate::residency::ExpertMode;
use crate::residency::ssd_experts::{ExpertSsd, SsdArgs, SsdExperts, tier_layers};

use super::layout::Layout;
use super::{LmConfig, LoadError};

/// Whether the tier applies: `mode`, the architecture (GLM-5.3-Flash only), one stage on an
/// integrated GPU (the GPU reads the host's memory in place) or, asked for, the CPU, and in
/// `auto` the experts not fitting (`experts_fit`).
pub fn wanted(
    mode: ExpertSsd,
    arch: &str,
    stages: usize,
    tp: usize,
    device: Option<DeviceKind>,
    experts_fit: bool,
) -> bool {
    if arch != "glm5-next" || stages != 1 || tp > 1 {
        return false;
    }
    match mode {
        ExpertSsd::Off => false,
        ExpertSsd::On => matches!(device, None | Some(DeviceKind::IntegratedGpu)),
        ExpertSsd::Auto => device == Some(DeviceKind::IntegratedGpu) && !experts_fit,
    }
}

/// The tier's layers when it applies to this load.
pub fn plan(
    cfg: &LmConfig,
    gguf: &GgufFile,
    layout: &Layout,
) -> Result<Option<Vec<usize>>, LoadError> {
    let hp = Hparams::from_gguf(gguf)?;
    let device = layout.stages[0].0.map(|d| d.kind());
    let fit = layout.mode == ExpertMode::Device;
    let args = &cfg.residency.ssd;
    if !wanted(
        args.mode,
        hp.arch.name(),
        layout.stages.len(),
        cfg.tp,
        device,
        fit,
    ) {
        return Ok(None);
    }
    match tier_layers(gguf, hp.n_layer) {
        Some(l) => Ok(Some(l)),
        None if args.mode == ExpertSsd::On => Err(LoadError::Device(
            "--expert-ssd on: the model's routed experts are not NVFP4 safetensors".into(),
        )),
        None => Ok(None),
    }
}

/// Memory for the tier: as given, or what the device has free less `keep` (the KV cache and
/// the step buffers, [`keep`]) and the reserve.
pub fn budget(args: &SsdArgs, free: Option<u64>, keep: u64) -> Result<u64, LoadError> {
    if let Some(b) = args.cache_bytes {
        return Ok(b);
    }
    let free = free.ok_or_else(|| {
        LoadError::Device("--expert-ssd on the CPU: give --expert-ssd-cache-gib".into())
    })?;
    Ok(free.saturating_sub(keep).saturating_sub(args.reserve))
}

/// What the load sizes next to the tier on a `kind` device: the KV pool's headroom for the
/// step buffers and the blocks it wants for `--max-seqs` contexts.
pub fn keep(cfg: &LmConfig, hp: &Hparams, kind: DeviceKind) -> u64 {
    let ctx_blocks = cfg.ctx_size.div_ceil(KV_BLOCK);
    let shared = kind == DeviceKind::IntegratedGpu;
    let blocks = super::auto_kv_want(cfg.max_seqs.max(1), ctx_blocks, false, shared);
    let per = KvPool::block_bytes_paged(hp, 0..hp.n_layer, false, cfg.cache_types);
    (super::kv_headroom(kind, cfg.max_batch_tokens, cfg.uma_free) + blocks * per) as u64
}

/// Drops the checkpoint's pages from the page cache (the load read them; the tier reads around
/// it), so the memory the tier sizes itself by is free.
pub fn drop_page_cache(gguf: &GgufFile) {
    gguf.release_page_cache();
}

/// Builds the tier over `layers` for the loaded model.
pub fn build(
    cfg: &LmConfig,
    gguf: &GgufFile,
    layout: &Layout,
    hp: &Hparams,
    layers: &[usize],
) -> Result<SsdExperts, LoadError> {
    drop_page_cache(gguf);
    let dev = layout.stages[0].0;
    let free = dev.map(|d| d.memory().0 as u64);
    let keep = dev.map_or(0, |d| keep(cfg, hp, d.kind()));
    let bytes = budget(&cfg.residency.ssd, free, keep)?;
    let buft = match dev {
        Some(d) => d
            .host_buffer_type()
            .ok_or_else(|| LoadError::Device("no pinned host memory for the SSD tier".into()))?,
        None => layout.host,
    };
    let max_tokens = cfg.max_batch_tokens.max(cfg.prefill_chunk);
    SsdExperts::new(
        gguf,
        layers,
        hp.n_layer,
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        bytes,
        buft,
        max_tokens,
        &cfg.residency.ssd,
    )
    .map_err(|e| LoadError::Device(format!("SSD expert tier: {e}")))
}

/// Whether the bounded host expert tier ([`crate::residency::host_tier`], #37) may apply: the
/// VRAM expert cache on one discrete GPU reading a pinned host arena, Qwen3.8-Flash-Next's
/// plain GGUF experts. Not the SSD tier's gate above (GLM-5.3-Flash NVFP4 on an integrated
/// GPU), which stays as it is; whether it does apply is the memory's to say
/// ([`host_tier_plan`]). `cpu` is the host's architecture ([`std::env::consts::ARCH`]): the
/// tier is an x86_64 PC's; a POWER9 host (unified memory behind its GPUs) and every other
/// one load as before, whatever their memory.
pub fn host_tier_wanted(
    cpu: &str,
    mode: ExpertMode,
    arch: &str,
    stages: usize,
    tp: usize,
    device: Option<DeviceKind>,
    arena: bool,
) -> bool {
    cpu == "x86_64"
        && mode == ExpertMode::Cache
        && arch == "qwen4exp"
        && stages == 1
        && tp <= 1
        && device == Some(DeviceKind::Gpu)
        && arena
}

/// The prompt step the bounded host tier's long prompts are sized for: a pass over the lent
/// VRAM cache copies every expert no kept slot holds into VRAM once, those in no RAM slot
/// read from the checkpoint first, so a prompt costs its passes more than its tokens. The
/// cache keeps as many slots through a lend as leave a step of this many tokens its buffers
/// (an 8K prompt is one pass); the kept slots' experts are neither copied nor read.
pub const HOST_TIER_PASS: usize = 8704;

/// From this many prompt tokens left, the bounded host tier's prompts lend the cache (with
/// `--long-prompt-tokens` at its default): a staged pass reads the checkpoint once, shorter
/// prompts' chunks read what they route to. Above the startup warmup's longest prompt
/// (`--warmup-shapes`): the first chats find the VRAM cache as the load filled it.
pub const HOST_TIER_LONG: usize = 1536;

/// The VRAM the expert cache leaves free beside the measured step buffers on the bounded host
/// tier's machine (one discrete GPU, one stage), in place of
/// [`crate::residency::CACHE_RESERVE`]: every 3 MiB of it is a slot, and a slot saves a copy
/// over PCIe each time its expert is routed to. Measured on a 24 GB card (four chats at once,
/// prompts up to 32K, `--deterministic`): about 1 GiB stays free beside the cache with it. A
/// long prompt's lend keeps [`HOST_TIER_LEND_RESERVE`] free instead
/// ([`crate::residency::ResidencyConfig::lend_reserve`]).
pub const HOST_TIER_RESERVE: u64 = 512 << 20;

/// What a long prompt's lend keeps free there in place of [`HOST_TIER_RESERVE`]. Measured: the
/// staged passes then get within 50 MiB of the VRAM they had with the default reserve. With
/// 1 GiB they got 238 MiB more, and four prompts of 3,000 tokens at once went in one pass
/// whose compute buffer did not fit (handled, in unified memory) instead of two.
pub const HOST_TIER_LEND_RESERVE: u64 = 768 << 20;

/// The fewest tokens a staged pass of the bounded host tier is capped to.
pub const HOST_TIER_MIN_PASS: usize = 1024;

/// The most tokens a staged pass of the bounded host tier takes: what fits `free` bytes of
/// VRAM, when a step of `tokens` tokens takes `bytes` (linear in the tokens), in whole
/// `block`s and at least [`HOST_TIER_MIN_PASS`]. 0 (no cap) when nothing was measured.
pub fn host_tier_pass_cap(free: usize, bytes: usize, tokens: usize, block: usize) -> usize {
    if bytes == 0 || tokens == 0 {
        return 0;
    }
    let fits = (free as u128 * tokens as u128 / bytes as u128) as usize;
    (fits / block.max(1) * block.max(1)).max(HOST_TIER_MIN_PASS)
}

/// The bounded host tier of a load: its layers' expert tensors and its memory.
pub struct HostTierPlan {
    pub srcs: Vec<[crate::gguf::TensorInfo; 3]>,
    pub plan: crate::residency::host_tier::Plan,
}

/// [`host_tier_wanted`] for this load (on one machine).
fn host_tier_machine(cfg: &LmConfig, hp: &Hparams, layout: &Layout) -> bool {
    layout.node.is_none()
        && host_tier_wanted(
            std::env::consts::ARCH,
            layout.mode,
            hp.arch.name(),
            layout.stages.len(),
            cfg.tp,
            layout.stages[0].0.map(|d| d.kind()),
            layout.arenas.is_some(),
        )
}

/// Whether this load is the bounded host tier's machine and model ([`host_tier_wanted`], plain
/// GGUF experts) whatever its memory: with every routed expert in host memory
/// ([`host_tier_plan`] gives `None`) it still starts its VRAM slots from the saved routing heat
/// and lends the KV pool's tail to the expert cache, as with the tier.
pub fn host_tier_applies(
    cfg: &LmConfig,
    gguf: &GgufFile,
    layout: &Layout,
) -> Result<bool, LoadError> {
    let hp = Hparams::from_gguf(gguf)?;
    Ok(host_tier_machine(cfg, &hp, layout)
        && crate::residency::host_tier::sources(gguf, hp.n_layer, hp.n_expert as usize).is_some())
}

/// The bounded host tier's plan when it applies to this load: [`host_tier_wanted`], and the
/// routed experts do not fit in the host memory there is beside the n-gram table and the KV
/// host tier ([`crate::residency::host_tier::size`]). `None`: the load is as before.
pub fn host_tier_plan(
    cfg: &LmConfig,
    gguf: &GgufFile,
    layout: &Layout,
) -> Result<Option<HostTierPlan>, LoadError> {
    use crate::residency::host_tier::{self, Sizing, Table};
    use crate::residency::lookup_table::TableIo;
    let hp = Hparams::from_gguf(gguf)?;
    if !host_tier_machine(cfg, &hp, layout) {
        return Ok(None);
    }
    let Some(srcs) = host_tier::sources(gguf, hp.n_layer, hp.n_expert as usize) else {
        return Ok(None);
    };
    let sizes = host_tier::expert_bytes(&srcs, hp.n_expert as usize);
    let per_slot: u64 = sizes.iter().flatten().sum();
    let table = hp
        .arch
        .ngram_table()
        .then(|| crate::model::core::Model::ngram_table_info(gguf))
        .flatten()
        .map(|t| Table {
            bytes: t.bytes(),
            in_ram: matches!(cfg.ngram.io, TableIo::Ram | TableIo::Mmap),
            direct: cfg.ngram.io == TableIo::Direct,
        });
    // A KV host tier asked for by hand; the automatic one takes what the experts leave.
    let per = KvPool::block_bytes_paged(&hp, 0..hp.n_layer, false, cfg.cache_types);
    let kv_spill = cfg
        .kv_host_tokens
        .map_or(0, |t| (t.div_ceil(KV_BLOCK) * per) as u64);
    let mem = host_tier::mem_info();
    let mib = |b: u64| b >> 20;
    match host_tier::size(mem, &sizes, hp.n_expert as usize, table, kv_spill) {
        Sizing::Whole => Ok(None),
        Sizing::Bounded(plan) => {
            tracing::info!(
                part = "host experts",
                ram_slots_per_layer = plan.slots,
                experts_per_layer = hp.n_expert,
                expert_ram_mib = mib(plan.slot_bytes),
                expert_staging_mib = mib(plan.stage_bytes),
                ngram_table_mib = mib(plan.table),
                ngram_io = if table.is_some_and(|t| t.in_ram) {
                    "ram"
                } else {
                    "direct"
                },
                kv_spill_mib = mib(plan.kv_spill),
                margin_mib = mib(plan.margin),
                host_available_mib = mib(plan.available),
                all_experts_mib = mib(per_slot * hp.n_expert as u64),
                "memory plan"
            );
            Ok(Some(HostTierPlan { srcs, plan }))
        }
        Sizing::TooSmall(plan) => Err(LoadError::Device(format!(
            "the routed experts do not fit in host memory: {} MiB available leave {} RAM slots \
             per layer after a {} MiB margin and {} MiB for the n-gram table (at least {} slots \
             are needed)",
            mib(plan.available),
            plan.slots,
            mib(plan.margin),
            mib(plan.table),
            host_tier::MIN_SLOTS
        ))),
    }
}

/// Where the bounded host tier keeps a model's routing heat between runs: a small file (the
/// ranking of every routed layer and expert, some 200 KB) in the user's state directory
/// (`$XDG_STATE_HOME`, else `$HOME/.local/state`), named after the checkpoint's first file and
/// the bytes of all its files, so another checkpoint (or the same one requantized) never reads
/// it. Never beside the model. `None` (nothing saved or read) without a home directory.
pub fn host_tier_heat_file(
    state_home: Option<&str>,
    home: Option<&str>,
    model: &std::path::Path,
    bytes: u64,
) -> Option<std::path::PathBuf> {
    let dir = match (state_home.filter(|s| s.starts_with('/')), home) {
        (Some(s), _) => std::path::PathBuf::from(s),
        (None, Some(h)) if h.starts_with('/') => std::path::Path::new(h).join(".local/state"),
        _ => return None,
    };
    let name = model.file_name()?.to_string_lossy();
    Some(
        dir.join("legion")
            .join("expert-heat")
            .join(format!("{name}.{bytes}.heat")),
    )
}

/// [`host_tier_heat_file`] for the loaded checkpoint.
pub fn host_tier_heat_path(gguf: &GgufFile) -> Option<std::path::PathBuf> {
    let mut shards = gguf.shard_paths();
    let first = shards.next()?.to_path_buf();
    let mut bytes = std::fs::metadata(&first).ok()?.len();
    for p in shards {
        bytes += std::fs::metadata(p).ok()?.len();
    }
    host_tier_heat_file(
        std::env::var("XDG_STATE_HOME").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
        &first,
        bytes,
    )
}

/// Builds the bounded host tier `plan` for the loaded model, its memory in stage 0's arena.
pub fn host_tier_build(
    cfg: &LmConfig,
    gguf: &GgufFile,
    layout: &Layout,
    hp: &Hparams,
    plan: &HostTierPlan,
) -> Result<std::sync::Arc<crate::residency::host_tier::HostTier>, LoadError> {
    let buft = layout
        .arenas
        .as_ref()
        .and_then(|a| a.first())
        .map(|a| a.0)
        .ok_or_else(|| LoadError::Device("host expert tier: no pinned host arena".into()))?;
    let max_tokens = cfg.max_batch_tokens.max(cfg.prefill_chunk);
    crate::residency::host_tier::HostTier::new(
        gguf,
        &plan.srcs,
        hp.n_expert as usize,
        hp.n_expert_used as usize,
        plan.plan,
        buft,
        max_tokens,
    )
    .map_err(|e| LoadError::Device(format!("host expert tier: {e}")))
}

#[cfg(test)]
mod tests;
