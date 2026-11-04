//! A model split over two machines (#562, [`crate::node`]): each rank's share of the layers
//! and tensors, the fit check that names a split that would fit, and rank 1's side (it runs
//! its executor for rank 0 instead of serving HTTP).

#[cfg(test)]
mod tests;

use std::net::TcpListener;
use std::ops::Range;

use super::{COMPUTE_HEADROOM, LmConfig, LoadError};
use crate::gguf::TensorInfo;
use crate::model::core::{Executor, Hparams, Model, Placement, Pool};
use crate::node::{NodeConfig, Reply, Request};

/// This machine's part of a model split over two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Share {
    pub rank: usize,
    /// Rank 1's first layer.
    pub split: usize,
    /// This rank's layers.
    pub layers: Range<usize>,
}

/// The layer index of a tensor name (`blk.<il>.`), if it belongs to a layer.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Whether rank `rank` of a split at `split` loads tensor `name`: a layer's tensors on the
/// rank running it (the layers past the last, an MTP head's, nowhere: no MTP over two nodes
/// yet), the token embeddings on rank 0, the RoPE factors on both, the rest (the output norm
/// and head) on rank 1.
pub(super) fn owns(rank: usize, split: usize, n_layer: usize, name: &str) -> bool {
    match layer_of(name) {
        Some(il) => il < n_layer && (il < split) == (rank == 0),
        None if name == "token_embd.weight" => rank == 0,
        None if name == "rope_freqs.weight" => true,
        None => rank == 1,
    }
}

/// Device bytes per layer, and those of rank 0's and rank 1's tensors outside the layers.
pub(super) fn byte_shares(tensors: &[TensorInfo], n_layer: usize) -> (Vec<u64>, u64, u64) {
    let mut bytes = vec![0u64; n_layer];
    let (mut first, mut last) = (0u64, 0u64);
    for t in tensors {
        if Model::placement(&t.name, false) != Placement::Device {
            continue;
        }
        match layer_of(&t.name) {
            Some(il) if il < n_layer => bytes[il] += t.size,
            Some(_) => {}
            None => {
                if owns(0, n_layer, n_layer, &t.name) {
                    first += t.size;
                }
                if owns(1, 0, n_layer, &t.name) {
                    last += t.size;
                }
            }
        }
    }
    (bytes, first, last)
}

/// The rank's bytes for a split at `split`.
fn rank_bytes(bytes: &[u64], first: u64, last: u64, split: usize) -> (u64, u64) {
    (
        first + bytes[..split].iter().sum::<u64>(),
        last + bytes[split..].iter().sum::<u64>(),
    )
}

/// The split that balances the two ranks' bytes.
pub(super) fn balanced_split(bytes: &[u64], first: u64, last: u64) -> usize {
    (1..bytes.len())
        .min_by_key(|&s| {
            let (a, b) = rank_bytes(bytes, first, last, s);
            a.max(b)
        })
        .unwrap_or(1)
}

/// The split closest to balanced under which both ranks fit in `room` bytes each (`None`: no
/// split does).
pub(super) fn fitting_split(bytes: &[u64], first: u64, last: u64, room: u64) -> Option<usize> {
    (1..bytes.len())
        .filter(|&s| {
            let (a, b) = rank_bytes(bytes, first, last, s);
            a <= room && b <= room
        })
        .min_by_key(|&s| {
            let (a, b) = rank_bytes(bytes, first, last, s);
            a.max(b)
        })
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

impl Share {
    /// This rank's share of the model's layers: at `--node-split`, else balanced by the
    /// layers' bytes. Every tensor of the share must fit in `free` bytes of device memory
    /// (no SSD or host tier over two nodes): if not, the error names a split that would fit.
    pub fn plan(
        node: &NodeConfig,
        hp: &Hparams,
        tensors: &[TensorInfo],
        free: u64,
        n_stages: usize,
    ) -> Result<Self, LoadError> {
        let n = hp.n_layer;
        if node.rank > 1 {
            return Err(LoadError::Device(format!(
                "--node-rank {}: a model splits over two nodes, ranks 0 and 1",
                node.rank
            )));
        }
        let (bytes, first, last) = byte_shares(tensors, n);
        let split = match node.split {
            Some(s) if s == 0 || s >= n => {
                return Err(LoadError::Device(format!(
                    "--node-split {s}: rank 1's first layer must be within 1..{n}"
                )));
            }
            Some(s) => s,
            None => balanced_split(&bytes, first, last),
        };
        let (b0, b1) = rank_bytes(&bytes, first, last, split);
        let need = if node.rank == 0 { b0 } else { b1 };
        let headroom = (COMPUTE_HEADROOM * n_stages) as u64;
        if need + headroom > free {
            let room = free.saturating_sub(headroom);
            let hint = match fitting_split(&bytes, first, last, room) {
                Some(s) => {
                    let (a, b) = rank_bytes(&bytes, first, last, s);
                    format!(
                        "--node-split {s} would put {:.1} GiB on rank 0 and {:.1} GiB on rank 1 \
                         (if the other node has as much memory free)",
                        gib(a),
                        gib(b)
                    )
                }
                None => format!(
                    "no split fits two nodes of {:.1} GiB: the model needs {:.1} GiB",
                    gib(room),
                    gib(bytes.iter().sum::<u64>() + first + last)
                ),
            };
            return Err(LoadError::Device(format!(
                "rank {}'s share at --node-split {split} needs {:.1} GiB for its weights (+{:.1} \
                 GiB compute headroom), {:.1} GiB free: {hint}",
                node.rank,
                gib(need),
                gib(headroom),
                gib(free)
            )));
        }
        Ok(Self {
            rank: node.rank,
            split,
            layers: node.share(n, split),
        })
    }

    pub fn owns(&self, name: &str, n_layer: usize) -> bool {
        owns(self.rank, self.split, n_layer, name)
    }
}

/// What a split model cannot do (yet): checked before anything loads.
pub(super) fn check_config(cfg: &LmConfig) -> Result<(), LoadError> {
    let Some(node) = &cfg.node else {
        return Ok(());
    };
    let no = |what: &str| {
        Err(LoadError::Device(format!(
            "--node-rank {}: not with {what} (a model over two nodes runs every weight in device \
             memory, without drafting)",
            node.rank
        )))
    };
    if cfg.device != super::DeviceChoice::Cuda {
        return no("--device cpu");
    }
    if cfg.tp > 1 {
        return no("--tp");
    }
    if cfg.spec.enabled() {
        return no("speculative decoding (--spec)");
    }
    if cfg.kv_host_tokens.is_some_and(|t| t > 0) {
        return no("--kv-host-tokens");
    }
    if !matches!(
        cfg.residency.mode,
        crate::residency::ExpertMode::Auto | crate::residency::ExpertMode::Device
    ) || cfg.host_experts
    {
        return no("experts outside device memory (--experts, --host-experts)");
    }
    if node.peer.is_empty() {
        return Err(LoadError::Device(
            "--node-rank needs --node-peer <ip:port> (rank 1 listens there, rank 0 connects)"
                .into(),
        ));
    }
    Ok(())
}

/// Rank 1: answers rank 0's requests with `exec`, one connection after another; returns only
/// when it cannot listen.
pub(super) fn follow(mut exec: Executor, share: &Share, peer: &str, n_blocks: usize) -> LoadError {
    let listener = match TcpListener::bind(peer) {
        Ok(l) => l,
        Err(e) => return LoadError::Device(format!("--node-peer {peer}: cannot listen: {e}")),
    };
    let n_layer = exec.model().hp.n_layer;
    let mut blocks = n_blocks;
    tracing::info!(
        peer,
        layers = ?share.layers,
        kv_cache_tokens = blocks * crate::model::core::KV_BLOCK,
        "node rank 1: waiting for rank 0"
    );
    let r = crate::node::serve(listener, |req| {
        let out = handle(&mut exec, req, (n_layer, share.split), &mut blocks);
        out.unwrap_or_else(Reply::Error)
    });
    LoadError::Device(format!("node rank 1 stopped listening: {r:?}"))
}

fn handle(
    exec: &mut Executor,
    req: Request,
    mine: (usize, usize),
    blocks: &mut usize,
) -> Result<Reply, String> {
    Ok(match req {
        Request::Hello {
            version,
            n_layer,
            split,
            n_blocks,
        } => {
            crate::node::check_hello(version, n_layer, split, mine)?;
            let n = (*blocks).min(n_blocks as usize);
            if n < *blocks {
                exec.resize_kv(n).map_err(|e| e.to_string())?;
                *blocks = n;
            }
            tracing::info!(
                kv_cache_tokens = n * crate::model::core::KV_BLOCK,
                "node rank 0 says hello"
            );
            Reply::Hello { n_blocks: n as u32 }
        }
        Request::Resize { n_blocks } => {
            exec.resize_kv(n_blocks as usize)
                .map_err(|e| e.to_string())?;
            *blocks = n_blocks as usize;
            Reply::Done
        }
        Request::Step { groups, rows } => {
            let t = std::time::Instant::now();
            let groups = crate::model::core::from_wire(&groups)?;
            let out = exec.step_from(&groups, rows).map_err(|e| e.to_string())?;
            Reply::Step {
                argmax: out.argmax,
                rows: out.rows,
                ms: t.elapsed().as_secs_f32() * 1e3,
            }
        }
        Request::ClearSlot { slot } => {
            exec.clear_slot(slot as usize);
            Reply::Done
        }
        Request::CopyStates { copies } => {
            let pool = |p: u8| if p == 1 { Pool::Saved } else { Pool::Seq };
            let copies: Vec<_> = copies
                .into_iter()
                .map(|(a, i, b, j)| (pool(a), i as usize, pool(b), j as usize))
                .collect();
            exec.copy_states(&copies).map_err(|e| e.to_string())?;
            Reply::Done
        }
        Request::TpHello { .. }
        | Request::TpStep { .. }
        | Request::SaveRing { .. }
        | Request::EnableRingSaves
        | Request::TpMtp { .. }
        | Request::TpMtpChain { .. }
        | Request::TpMtpRows { .. }
        | Request::TpCommit { .. } => {
            return Err(
                "a request of tensor parallelism over machines (--tp-rank): this rank \
                        runs a model split by layers (--node-rank)"
                    .into(),
            );
        }
    })
}
