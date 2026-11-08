//! A model split over two machines (#562, [`crate::node`]): each rank's share of the layers
//! and tensors, the fit check that names a split that would fit, and rank 1's side (it runs
//! its executor for rank 0 instead of serving HTTP).
//!
//! With `--spec mtp` (#560) the MTP head runs on rank 1, next to the last layer whose rows it
//! reads and the output head it shares: rank 1 keeps the rows its last step captured and
//! drafts from them when rank 0 asks ([`Follower`]), so a draft round sends only the
//! sequences' positions and gets back their drafts. Rank 0 commits a verify step's states on
//! both ranks and runs the rest of speculative decoding (acceptance, the depth policy) as on
//! one machine.

#[cfg(test)]
mod tests;

use std::net::TcpListener;
use std::ops::Range;

use super::spec::MtpDrafter;
use super::{COMPUTE_HEADROOM, LmConfig, LoadError};
use crate::gguf::TensorInfo;
use crate::model::core::{Executor, Hparams, Model, Placement, Pool};
use crate::node::{NodeConfig, Reply, Request, WireDraft};
use crate::spec::{Draft, DraftRequest};

/// This machine's part of a model split over two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Share {
    pub rank: usize,
    /// Rank 1's first layer.
    pub split: usize,
    /// This rank's layers.
    pub layers: Range<usize>,
    /// The model's MTP head drafts (`--spec mtp`): it runs on rank 1 (#560).
    pub mtp: bool,
}

/// The layer index of a tensor name (`blk.<il>.`), if it belongs to a layer.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Whether rank `rank` of a split at `split` loads tensor `name`: a layer's tensors on the
/// rank running it, the token embeddings on rank 0, the RoPE factors on both, the rest (the
/// output norm and head) on rank 1. The layers past the last (an MTP head's) go to rank 1 when
/// the head drafts (`mtp`), with a copy of the token embeddings, which the head reads for the
/// tokens it drafts from; else nowhere.
pub(super) fn owns(rank: usize, split: usize, n_layer: usize, mtp: bool, name: &str) -> bool {
    match layer_of(name) {
        Some(il) if il >= n_layer => mtp && rank == 1,
        Some(il) => (il < split) == (rank == 0),
        None if name == "token_embd.weight" => rank == 0 || mtp,
        None if name == "rope_freqs.weight" => true,
        None => rank == 1,
    }
}

/// Device bytes per layer, and those of rank 0's and rank 1's tensors outside the layers (with
/// `mtp`, rank 1's include the MTP head's layer and its copy of the embeddings).
pub(super) fn byte_shares(
    tensors: &[TensorInfo],
    n_layer: usize,
    mtp: bool,
) -> (Vec<u64>, u64, u64) {
    let mut bytes = vec![0u64; n_layer];
    let (mut first, mut last) = (0u64, 0u64);
    for t in tensors {
        if Model::placement(&t.name, false) != Placement::Device {
            continue;
        }
        match layer_of(&t.name) {
            Some(il) if il < n_layer => bytes[il] += t.size,
            Some(_) => {
                if owns(1, 0, n_layer, mtp, &t.name) {
                    last += t.size;
                }
            }
            None => {
                if owns(0, n_layer, n_layer, mtp, &t.name) {
                    first += t.size;
                }
                if owns(1, 0, n_layer, mtp, &t.name) {
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
    /// With `mtp` (the MTP head drafts) rank 1's share counts the head's layer and its copy of
    /// the embeddings.
    pub fn plan(
        node: &NodeConfig,
        hp: &Hparams,
        tensors: &[TensorInfo],
        free: u64,
        n_stages: usize,
        mtp: bool,
    ) -> Result<Self, LoadError> {
        let n = hp.n_layer;
        if node.rank > 1 {
            return Err(LoadError::Device(format!(
                "--node-rank {}: a model splits over two nodes, ranks 0 and 1",
                node.rank
            )));
        }
        let (bytes, first, last) = byte_shares(tensors, n, mtp);
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
            mtp,
        })
    }

    pub fn owns(&self, name: &str, n_layer: usize) -> bool {
        owns(self.rank, self.split, n_layer, self.mtp, name)
    }

    /// Rank 0 with the MTP head drafting: the head runs on rank 1 ([`Follower`]).
    pub fn head_on_peer(&self) -> bool {
        self.mtp && self.rank == 0
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
             memory, and drafts with the model's own MTP head only)",
            node.rank
        )))
    };
    if cfg.device != super::DeviceChoice::Cuda {
        return no("--device cpu");
    }
    if cfg.tp > 1 {
        return no("--tp");
    }
    // The MTP head runs on rank 1 beside the head's inputs (#560); no other drafter is wired.
    if cfg.spec.enabled() && !super::drafts_with_mtp_head(&cfg.spec) {
        return no(&format!("--spec {}", cfg.spec.kind.name()));
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
/// when it cannot listen. `drafter`: the MTP head's driver when it drafts (`--spec mtp`).
pub(super) fn follow(
    exec: Executor,
    drafter: Option<MtpDrafter>,
    share: &Share,
    peer: &str,
    n_blocks: usize,
) -> LoadError {
    let listener = match TcpListener::bind(peer) {
        Ok(l) => l,
        Err(e) => return LoadError::Device(format!("--node-peer {peer}: cannot listen: {e}")),
    };
    let n_layer = exec.model().hp.n_layer;
    tracing::info!(
        peer,
        layers = ?share.layers,
        kv_cache_tokens = n_blocks * crate::model::core::KV_BLOCK,
        mtp = drafter.is_some(),
        "node rank 1: waiting for rank 0"
    );
    let mut f = Follower::new(exec, drafter, (n_layer, share.split), n_blocks);
    let r = crate::node::serve(listener, |req| f.handle(req).unwrap_or_else(Reply::Error));
    LoadError::Device(format!("node rank 1 stopped listening: {r:?}"))
}

/// Rank 1's executor and what it keeps between rank 0's requests.
pub(super) struct Follower {
    pub exec: Executor,
    /// The MTP head's driver (`--spec mtp`, #560): it drafts here, from the rows the last step
    /// captured, so neither those rows nor the head cross the link.
    drafter: Option<MtpDrafter>,
    /// The captured rows of the last step ([`crate::model::core::StepOutput::hidden`]).
    last_hidden: Vec<Vec<f32>>,
    /// The model's layer count and the split.
    mine: (usize, usize),
    blocks: usize,
}

impl Follower {
    pub fn new(
        exec: Executor,
        drafter: Option<MtpDrafter>,
        mine: (usize, usize),
        n_blocks: usize,
    ) -> Self {
        Self {
            exec,
            drafter,
            last_hidden: Vec::new(),
            mine,
            blocks: n_blocks,
        }
    }

    /// The MTP drafts of `reqs` from the last step's rows.
    fn drafts(&mut self, reqs: Vec<WireDraft>, p_min: f32) -> Result<Reply, String> {
        let t = std::time::Instant::now();
        let d = self.drafter.as_mut().ok_or(
            "MTP drafts asked for, but this rank does not draft: give both ranks --spec mtp",
        )?;
        d.set_p_min(p_min);
        let w = self.exec.mtp_width();
        let rows = self.last_hidden.first().map_or(&[][..], Vec::as_slice);
        // Each sequence's tokens at their positions (none needed before `base`).
        let tokens: Vec<Vec<u32>> = reqs
            .iter()
            .map(|r| {
                let mut v = vec![0u32; r.base as usize];
                v.extend_from_slice(&r.tokens);
                v
            })
            .collect();
        let mut outs = vec![Draft::default(); reqs.len()];
        let mut batch = Vec::with_capacity(reqs.len());
        for ((r, tokens), out) in reqs.iter().zip(&tokens).zip(&mut outs) {
            let (start, ran, off) = (r.start as usize, r.ran as usize, r.off as usize);
            let fed = start + ran + usize::from(r.max > 0);
            let hidden = rows.get(off * w..(off + ran) * w);
            let (Some(hidden), true) = (hidden, r.base as usize <= start && fed <= tokens.len())
            else {
                return Err(format!(
                    "a draft request for rows {off}..{} of {} (positions {start}..{} of {}): \
                     not what the last step ran",
                    off + ran,
                    rows.len() / w.max(1),
                    start + ran,
                    tokens.len()
                ));
            };
            batch.push(DraftRequest {
                seq: r.seq,
                tokens: tokens.as_slice(),
                start,
                ran,
                hidden: vec![hidden],
                max: r.max as usize,
                out,
            });
        }
        let blocks: Vec<&[u32]> = reqs.iter().map(|r| r.blocks.as_slice()).collect();
        let rings: Vec<usize> = reqs.iter().map(|r| r.ring as usize).collect();
        let passes = d.step(&mut self.exec, &mut batch, &blocks, &rings)?;
        drop(batch);
        let drafts = outs
            .into_iter()
            .map(|o| o.tokens.into_iter().zip(o.confidence).collect())
            .collect();
        Ok(Reply::Drafts {
            drafts,
            passes: passes as u32,
            ms: t.elapsed().as_secs_f32() * 1e3,
        })
    }

    pub fn handle(&mut self, req: Request) -> Result<Reply, String> {
        let exec = &mut self.exec;
        let blocks = &mut self.blocks;
        let mine = self.mine;
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
                // The rows it captured (with the MTP head) for the drafts that follow; the last
                // step's go back to the executor.
                let old = std::mem::replace(&mut self.last_hidden, out.hidden);
                if !old.is_empty() {
                    exec.recycle_rows(old);
                }
                Reply::Step {
                    argmax: out.argmax,
                    rows: out.rows,
                    ms: t.elapsed().as_secs_f32() * 1e3,
                }
            }
            Request::TpCommit { commits } => {
                let commits: Vec<_> = commits
                    .into_iter()
                    .map(|(slot, keep)| (slot as usize, keep as usize))
                    .collect();
                exec.commit(&commits).map_err(|e| e.to_string())?;
                Reply::Done
            }
            Request::Draft { reqs, p_min } => return self.drafts(reqs, f32::from_bits(p_min)),
            Request::DraftRelease { seq } => {
                if let Some(d) = self.drafter.as_mut() {
                    d.release(seq);
                }
                Reply::Done
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
            | Request::EnableRingSaves { .. }
            | Request::TpMtp { .. }
            | Request::TpMtpChain { .. }
            | Request::TpMtpRows { .. } => {
                return Err(
                    "a request of tensor parallelism over machines (--tp-rank): this rank \
                        runs a model split by layers (--node-rank)"
                        .into(),
                );
            }
        })
    }
}
