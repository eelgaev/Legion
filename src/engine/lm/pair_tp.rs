//! `--pair-tp` (#668): sets up pair parallelism ([`crate::model::core::pair`]) for a loaded
//! qwen4exp model split over GPUs in NVLink pairs.

use super::{LmConfig, LoadError, layout};
use crate::model::core::Model;

/// Checks that `--pair-tp` applies to this model and layout: qwen4exp, CUDA, an even number of
/// GPU stages (2 or 4), stage `s` on CUDA device `s`, no other parallelism.
pub(super) fn check(cfg: &LmConfig, model: &Model, layout: &layout::Layout) -> Result<(), String> {
    let hp = &model.hp;
    if hp.arch.name() != "qwen4exp" || crate::model::qwen4::params(hp).is_none() {
        return Err(format!("not a qwen4exp model ({})", hp.arch.name()));
    }
    if layout.glm_par || cfg.tp > 1 || layout.node.is_some() || cfg.tp_net.is_some() {
        return Err("not with other parallelism (--tp, --node, --glm-parallel-decode)".into());
    }
    let n = layout.stages.len();
    if n < 2 || !n.is_multiple_of(2) {
        return Err(format!("needs an even number of GPU stages, not {n}"));
    }
    for (s, (d, _)) in layout.stages.iter().enumerate() {
        let cuda = d.and_then(super::cuda_index);
        if cuda != Some(s) {
            return Err(format!("stage {s} is not on CUDA{s}"));
        }
    }
    if hp.n_expert < 2 || hp.n_expert_used < 1 {
        return Err("needs routed experts".into());
    }
    Ok(())
}

/// Sets pair parallelism up on `model` (the exchange's inboxes in each GPU's VRAM, peer access
/// within each pair); the expert cache built later gives each paired layer a copy on the pair.
pub(super) fn set_up(
    cfg: &LmConfig,
    model: &mut Model,
    layout: &layout::Layout,
) -> Result<(), LoadError> {
    let err = |e: String| LoadError::Device(format!("--pair-tp: {e}"));
    check(cfg, model, layout).map_err(err)?;
    #[cfg(feature = "cuda")]
    {
        use crate::model::core::pair;
        let n = layout.stages.len();
        let devices: Vec<crate::ggml::Device> =
            layout.stages.iter().filter_map(|(d, _)| *d).collect();
        // A pair's GPUs on one socket: both read their misses from its memory, and NVLink
        // joins them (on other sockets the exchange would cross the X bus).
        let nodes: Vec<Option<u32>> = (0..n)
            .map(|d| {
                crate::residency::cuda::caps(d).and_then(|c| u32::try_from(c.host_numa_id).ok())
            })
            .collect();
        for s in (0..n).step_by(2) {
            if nodes[s] != nodes[s + 1] {
                return Err(err(format!(
                    "CUDA{s} and CUDA{} are on different host NUMA nodes ({:?}, {:?})",
                    s + 1,
                    nodes[s],
                    nodes[s + 1]
                )));
            }
        }
        let pairs: Vec<(usize, usize)> = (0..n).step_by(2).map(|s| (s, s + 1)).collect();
        pair::link::enable_peers(&pairs).map_err(err)?;
        let bufts: Vec<crate::ggml::BufferType> = devices.iter().map(|d| d.buffer_type()).collect();
        let status = devices[0]
            .host_buffer_type()
            .ok_or_else(|| err("no pinned host memory".into()))?;
        let hp = &model.hp;
        let link = pair::link::PairLink::new(
            &bufts,
            status,
            hp.n_embd,
            hp.n_expert_used,
            pair::MAX_TOKENS,
        )
        .map_err(|e| err(e.to_string()))?;
        let owner = pair::owners(hp.n_layer, |il| layout.stage_of(il));
        let n_expert = hp.n_expert as u32;
        tracing::info!(
            gpus = n,
            ?pairs,
            paired_layers = owner.iter().flatten().count(),
            max_tokens = pair::MAX_TOKENS,
            "pair parallelism: each stage's NVLink pair computes half of its routed experts in \
             decode and verify steps"
        );
        model.set_pair_tp(Some(pair::PairTp {
            owner,
            n_expert,
            link: std::sync::Arc::new(link),
        }));
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model, layout);
        Err(err("needs a CUDA build".into()))
    }
}
