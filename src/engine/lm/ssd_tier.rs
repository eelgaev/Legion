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

#[cfg(test)]
mod tests;
