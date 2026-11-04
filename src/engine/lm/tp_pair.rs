//! `--tp 2 --tp-per-gpu` (#691): tensor parallelism over the two GPUs of an NVLink pair as two
//! ranks of this process instead of one ggml meta device. Each GPU loads its share of the
//! model ([`crate::tp_net::shard::view`]) into an executor of its own, with its own kept
//! graphs, fused ops and CUDA graphs per step shape, as one GPU alone would; the layers'
//! partial sums meet through the pair's NVLink all-reduce ([`crate::model::core::pair::sum`]).
//! Rank 0 serves and leads ([`crate::model::core::TpLead::pair`]), rank 1 runs its steps on a
//! thread of its own: the ranks of tensor parallelism over machines ([`super::tp_node`]) with an
//! in-process link instead of the network.

use std::sync::{Arc, Mutex};

use super::{DeviceChoice, LmConfig, LmEngine, LoadError};
use crate::ggml::BufferType;
use crate::model::core::Attention;
use crate::model::core::pair::sum::{PairSum, SLOT_BYTES};
use crate::node::local::{LocalLink, LocalServe, local_link};

/// One rank of `--tp-per-gpu`, in its load's [`LmConfig::tp_pair`].
#[derive(Clone)]
pub struct TpPairRank {
    /// 0 (serves and leads) or 1.
    pub rank: usize,
    /// The GPU this rank loads on (its index among the GPUs).
    pub cuda: usize,
    /// The all-reduce both ranks' graphs sum through.
    pub sum: Arc<PairSum>,
    ends: Arc<Mutex<Ends>>,
}

/// The two ends of the ranks' control link, until each rank's load takes its own.
#[derive(Default)]
struct Ends {
    lead: Option<LocalLink>,
    serve: Option<LocalServe>,
    /// Rank 1's thread, which rank 0's lead joins when it stops (so rank 1 frees its GPU
    /// memory before the process exits).
    rank1: Option<std::thread::JoinHandle<()>>,
}

/// A rank's end of the control link: rank 0's link to rank 1 and rank 1's thread, or rank 1's
/// serving end.
pub(super) enum End {
    Lead(LocalLink, Option<std::thread::JoinHandle<()>>),
    Serve(LocalServe),
}

impl std::fmt::Debug for TpPairRank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpPairRank")
            .field("rank", &self.rank)
            .field("cuda", &self.cuda)
            .finish_non_exhaustive()
    }
}

impl TpPairRank {
    /// The two ranks, over GPUs `first` and `first + 1`.
    pub fn pair(sum: Arc<PairSum>, first: usize) -> [Self; 2] {
        let (lead, serve) = local_link();
        let ends = Arc::new(Mutex::new(Ends {
            lead: Some(lead),
            serve: Some(serve),
            rank1: None,
        }));
        [0, 1].map(|rank| Self {
            rank,
            cuda: first + rank,
            sum: sum.clone(),
            ends: ends.clone(),
        })
    }

    /// This rank's end of the control link; `None` once taken.
    pub(super) fn take_end(&self) -> Option<End> {
        let mut ends = self.ends.lock().unwrap_or_else(|e| e.into_inner());
        match self.rank {
            0 => Some(End::Lead(ends.lead.take()?, ends.rank1.take())),
            _ => ends.serve.take().map(End::Serve),
        }
    }

    /// Hands rank 1's thread to rank 0's end.
    fn set_rank1(&self, h: std::thread::JoinHandle<()>) {
        self.ends.lock().unwrap_or_else(|e| e.into_inner()).rank1 = Some(h);
    }
}

fn err(e: impl std::fmt::Display) -> LoadError {
    LoadError::Device(format!("--tp-per-gpu: {e}"))
}

/// What `--tp-per-gpu` needs of `cfg` (before anything loads).
pub(super) fn check(cfg: &LmConfig) -> Result<(), LoadError> {
    if cfg.tp != 2 {
        return Err(err(format!("needs --tp 2 (got --tp {})", cfg.tp)));
    }
    if cfg.tp_net.is_some() {
        return Err(err("not with --tp-rank (tensor parallelism over machines)"));
    }
    if cfg.node.is_some() {
        return Err(err("not with --node-rank"));
    }
    if cfg.gpus > 1 || !cfg.layer_split.is_empty() {
        return Err(err("not with --gpus or --layer-split (one GPU per rank)"));
    }
    if cfg.pair_tp || cfg.glm_parallel_decode {
        return Err(err("not with --pair-tp or --glm-parallel-decode"));
    }
    // Each rank sizes its cache as one GPU does, for the paged kernels' small compute buffers:
    // gather attention's grow with the prompt (an 8K prompt at --ctx-size 40960 ran out of
    // memory on 16 GB V100s).
    if cfg.device != DeviceChoice::Cpu && cfg.attention == Attention::Gather {
        return Err(err("not with --attention gather (the ranks page)"));
    }
    Ok(())
}

/// Each rank's config: its share of `cfg`, one GPU, the pair's all-reduce. The sliding layers
/// in rings whatever the attention (the CPU gathers), as under the meta device.
pub(super) fn rank_configs(cfg: &LmConfig, sum: Arc<PairSum>, first: usize) -> [LmConfig; 2] {
    TpPairRank::pair(sum, first).map(|r| LmConfig {
        tp: 1,
        tp_per_gpu: false,
        tp_pair: Some(r),
        // The rank's GPU is `r.cuda` (see `layout`), not a group's first.
        first_gpu: 0,
        ring_gather: true,
        ..cfg.clone()
    })
}

/// The inboxes' memory on each GPU (host memory on the CPU) and the host memory both GPUs
/// reach for the error word.
fn memory(cfg: &LmConfig, first: usize) -> Result<([BufferType; 2], BufferType), LoadError> {
    if cfg.device == DeviceChoice::Cpu {
        let cpu = crate::ggml::device_by_kind(crate::ggml::DeviceKind::Cpu)
            .ok_or_else(|| err("no CPU device"))?
            .buffer_type();
        return Ok(([cpu, cpu], cpu));
    }
    #[cfg(feature = "cuda")]
    {
        use crate::ggml::{DeviceKind, devices};
        let gpus: Vec<_> = devices()
            .into_iter()
            .filter(|d| d.kind() == DeviceKind::Gpu)
            .collect();
        if gpus.len() < first + 2 {
            return Err(err(format!(
                "GPUs {first} and {} asked for, {} found",
                first + 1,
                gpus.len()
            )));
        }
        let pair = [gpus[first], gpus[first + 1]];
        let ids = pair.map(super::cuda_index);
        let (Some(a), Some(b)) = (ids[0], ids[1]) else {
            return Err(err("the pair's GPUs are not CUDA devices"));
        };
        crate::model::core::pair::link::enable_peers(&[(a, b)]).map_err(err)?;
        let status = pair[0]
            .host_buffer_type()
            .ok_or_else(|| err("no pinned host memory"))?;
        Ok((pair.map(|d| d.buffer_type()), status))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = first;
        Err(err("needs a CUDA build (or --device cpu)"))
    }
}

/// Loads `cfg` (`--tp 2 --tp-per-gpu`) as two ranks: rank 1 on a thread of its own, which
/// answers rank 0's steps until rank 0 stops; rank 0 here, returned as the engine.
pub(super) fn load(cfg: LmConfig) -> Result<LmEngine, LoadError> {
    check(&cfg)?;
    // A `--tp-replicas` replica's pair starts at its first GPU.
    let first = cfg.first_gpu;
    let (inboxes, status) = memory(&cfg, first)?;
    let sum = Arc::new(PairSum::new(inboxes, status, SLOT_BYTES)?);
    let [cfg0, cfg1] = rank_configs(&cfg, sum, first);
    tracing::info!(
        gpus = ?[first, first + 1],
        slot_mib = SLOT_BYTES >> 20,
        "--tp-per-gpu: two ranks, one per GPU, summing over NVLink"
    );
    let rank0 = cfg0.tp_pair.clone().expect("rank 0");
    let rank1 = std::thread::Builder::new()
        .name("legion-tp-rank1".into())
        .spawn(move || {
            // Returns once rank 0 stops (or rank 1 cannot load: rank 0's load then fails at
            // its hello).
            if let Err(e) = LmEngine::load(cfg1) {
                tracing::warn!(error = %e, "--tp-per-gpu: rank 1 stopped");
            }
        })
        .map_err(err)?;
    rank0.set_rank1(rank1);
    LmEngine::load(cfg0)
}

#[cfg(test)]
mod tests;
