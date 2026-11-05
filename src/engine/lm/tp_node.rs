//! Tensor parallelism over machines (#619, [`crate::tp_net`]): what it cannot do yet, the
//! connections made before the model loads, this rank's share of the model, and the ranks
//! above 0, which run rank 0's steps instead of serving HTTP.
//!
//! The same ranks run tensor parallelism over the two GPUs of a pair in one process
//! (`--tp-per-gpu`, [`super::tp_pair`]): rank 1 loads on a thread of its own, both ranks reach
//! each other over in-process channels, and their graphs sum over NVLink.

use std::sync::Arc;
use std::time::Duration;

use super::tp_pair::End;
use super::{LmConfig, LoadError};
use crate::ggml::BufferType;
use crate::gguf::GgufFile;
use crate::model::core::pair::sum::PairSum;
use crate::model::core::{Executor, Model, Pool, TpLead};
use crate::node::{Link, Reply, Request, RowsPart};
use crate::tp_net::{self, CommHandle, ControlListener, StepProfile, TpNetConfig};

#[cfg(test)]
mod tests;

/// How long a rank waits for the others to start (and load).
const WAIT: Duration = Duration::from_secs(600);

/// A rank's connections, made before the model loads.
pub(super) struct Start {
    rank: usize,
    via: Via,
}

/// How the ranks reach each other.
enum Via {
    /// Over machines.
    Net {
        cfg: TpNetConfig,
        /// The communicator the model's all-reduces use.
        comm: CommHandle,
        /// Ranks above 0: where rank 0's control connection arrives.
        control: Option<ControlListener>,
    },
    /// The two GPUs of a pair in this process.
    Pair { sum: Arc<PairSum>, end: End },
}

/// What a following rank's steps sum through (its failure and `--tp-profile`).
enum Sums<'a> {
    Net(&'a CommHandle),
    Pair(&'a PairSum),
}

impl Sums<'_> {
    fn error(&self) -> Option<String> {
        match self {
            Self::Net(c) => c.comm().error(),
            Self::Pair(s) => s.error(),
        }
    }

    fn mark_step(&self, profile: &StepProfile) {
        if let Self::Net(c) = self
            && profile.enabled()
        {
            c.comm().mark_step();
        }
    }

    fn record(&self, profile: &mut StepProfile, took: Duration) {
        if let Self::Net(c) = self {
            profile.record(c.comm(), took, Duration::ZERO);
        }
    }
}

impl Start {
    /// This rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Makes `model` sum its layers' partial outputs over the ranks.
    pub fn set_sums(&self, model: &mut Model) {
        match &self.via {
            Via::Net { comm, .. } => model.set_tp_comm(Some(comm.id())),
            Via::Pair { sum, .. } => model.set_tp_pair(Some((sum.clone(), self.rank))),
        }
    }
}

fn device(e: impl std::fmt::Display) -> LoadError {
    LoadError::Device(format!("tensor parallelism over machines: {e}"))
}

/// The ranks of `cfg`'s tensor parallelism and this one's (`None`: none).
fn ranks(cfg: &LmConfig) -> Option<(usize, usize)> {
    match (&cfg.tp_net, &cfg.tp_pair) {
        (Some(t), _) => Some((t.n_ranks(), t.rank)),
        (None, Some(p)) => Some((2, p.rank)),
        (None, None) => None,
    }
}

/// What tensor parallelism over machines (and `--tp-per-gpu`) cannot do (yet): checked before
/// anything loads.
pub(super) fn check_config(cfg: &LmConfig) -> Result<(), LoadError> {
    let what = match (&cfg.tp_net, &cfg.tp_pair) {
        (Some(t), _) => {
            t.check().map_err(device)?;
            format!("--tp-rank {}", t.rank)
        }
        (None, Some(_)) => "--tp-per-gpu".to_string(),
        (None, None) => return Ok(()),
    };
    let no =
        |n: &str| -> Result<(), LoadError> { Err(device(format!("{what}: not with {n} (yet)"))) };
    if cfg.tp > 1 {
        return no("--tp (one GPU per rank)");
    }
    if cfg.node.is_some() {
        return no("--node-rank");
    }
    if cfg.tp_pair.is_some() {
        // Rank 1 is a thread here: nothing drafts on rank 0 alone.
        if cfg.spec.enabled() && !super::drafts_with_mtp_head(&cfg.spec) {
            return no("speculative decoding other than --spec mtp");
        }
    } else if cfg.spec.enabled() && !super::drafts_with_mtp_head(&cfg.spec) && !drafts_on_rank0(cfg)
    {
        // The MTP head is split like the target (`tp_net::shard::assistant_view`); DFlash and
        // DSpark run on rank 0 alone ([`rank0_drafting`]); other drafters are not wired to the
        // other ranks.
        return no("speculative decoding other than --spec mtp|dflash|dspark");
    }
    if cfg.kv_host_tokens.is_some_and(|n| n > 0) {
        return no("--kv-host-tokens");
    }
    if cfg.pipeline_groups > 1 {
        return no("--pipeline-groups");
    }
    if cfg.gemma4_compact_global_kv {
        return no("--gemma4-compact-global-kv");
    }
    if moe_ep(cfg).is_some() && cfg.residency.mode == crate::residency::ExpertMode::Cache {
        // The cache's slots and remapped ids index the whole stack of experts.
        return no("--tp-moe-ep and --experts cache");
    }
    if let Some(t) = &cfg.tp_net
        && t.transport == tp_net::Transport::Rdma
        && cfg.device == super::DeviceChoice::Cpu
    {
        return no("--tp-transport rdma on the CPU (RDMA carries a GPU's sums)");
    }
    Ok(())
}

/// This rank's share of `gguf` and its connections to the other ranks (`None`: no tensor
/// parallelism over machines, `gguf` itself).
pub(super) fn start(
    cfg: &LmConfig,
    gguf: GgufFile,
) -> Result<(GgufFile, Option<Start>), LoadError> {
    if let Some(p) = &cfg.tp_pair {
        let share = tp_net::shard::view(&gguf, p.rank, 2).map_err(device)?;
        let end = p
            .take_end()
            .ok_or_else(|| device("--tp-per-gpu: this rank's link was taken already"))?;
        tracing::info!(
            rank = p.rank,
            cuda = p.cuda,
            "--tp-per-gpu: loading this GPU's share"
        );
        let via = Via::Pair {
            sum: p.sum.clone(),
            end,
        };
        return Ok((share, Some(Start { rank: p.rank, via })));
    }
    let Some(t) = &cfg.tp_net else {
        return Ok((gguf, None));
    };
    let n = t.n_ranks();
    let share = if t.moe_ep {
        if gguf.architecture().as_deref() != Some("qwen4exp") {
            return Err(device("--tp-moe-ep: Qwen3.8-Flash-Next (qwen4exp) only"));
        }
        tp_net::shard::qwen4::view_with(&gguf, t.rank, n, experts(cfg))
    } else {
        tp_net::shard::view(&gguf, t.rank, n)
    }
    .map_err(device)?;
    tracing::info!(
        rank = t.rank,
        ranks = n,
        "tensor parallelism over machines: connecting to the other ranks"
    );
    let mut mesh = tp_net::connect(t, WAIT).map_err(device)?;
    // RDMA carries the CUDA path's pinned buffers only.
    let transport = match cfg.device {
        super::DeviceChoice::Cpu => tp_net::Transport::Tcp,
        super::DeviceChoice::Cuda => t.transport,
    };
    let data = std::mem::take(&mut mesh.data);
    let comm =
        tp_net::Comm::with_rails(t.rank, data, t.timeout, transport, t.rails).map_err(device)?;
    comm.set_collectives(t.wire, t.seq_par);
    comm.set_overlap(t.overlap);
    tracing::info!(
        rank = t.rank,
        ranks = n,
        transports = %comm.transports(),
        "tensor parallelism over machines: connected; loading this rank's share"
    );
    Ok((
        share,
        Some(Start {
            rank: t.rank,
            via: Via::Net {
                cfg: t.clone(),
                comm: tp_net::register(comm),
                control: mesh.control,
            },
        }),
    ))
}

/// The MTP assistant `gguf` to load: this rank's share of it under tensor parallelism over
/// machines (its queries over this rank's share of the cache of `hp`, the target's share),
/// else `gguf` itself.
pub(super) fn mtp_share(
    cfg: &LmConfig,
    gguf: GgufFile,
    hp: &crate::model::core::Hparams,
) -> Result<GgufFile, LoadError> {
    let Some((n, rank)) = ranks(cfg) else {
        return Ok(gguf);
    };
    if gguf.architecture().as_deref() == Some("qwen4exp") {
        // qwen4exp's head is one more block of the target's shape: cut as the target's are.
        return tp_net::shard::qwen4::view_with(&gguf, rank, n, experts(cfg))
            .map_err(|e| device(format!("--spec-draft: {e}")));
    }
    if hp.n_layer < 2 {
        return Err(device(
            "an MTP assistant for a model of fewer than two layers",
        ));
    }
    // All ranks' kv heads: this rank's share times the ranks.
    let full = |il: usize| {
        let (d, kv) = hp.kv_dims(il);
        (d as u64, kv as u64 * n as u64)
    };
    tp_net::shard::assistant_view(&gguf, full(hp.n_layer - 2), full(hp.n_layer - 1), rank, n)
        .map_err(|e| device(format!("--spec-draft: {e}")))
}

/// How `cfg`'s ranks share the routed experts out (`--tp-moe-ep`, #798).
fn experts(cfg: &LmConfig) -> tp_net::shard::qwen4::Experts {
    match &cfg.tp_net {
        Some(t) if t.moe_ep => match &t.moe_ep_place {
            Some(p) => tp_net::shard::qwen4::Experts::Placed(p.clone()),
            None => tp_net::shard::qwen4::Experts::Whole,
        },
        _ => tp_net::shard::qwen4::Experts::Rows,
    }
}

/// The routed experts this rank holds under `--tp-moe-ep` (#798): `(rank, ranks, placement)`,
/// the model's expert ids shared out evenly in rank order, or per layer as the placement
/// (`--tp-moe-ep-place`, #803) says; `None` without.
pub(super) fn moe_ep(
    cfg: &LmConfig,
) -> Option<(
    usize,
    usize,
    Option<std::sync::Arc<tp_net::ep_place::Placement>>,
)> {
    cfg.tp_net
        .as_ref()
        .filter(|t| t.moe_ep)
        .map(|t| (t.rank, t.n_ranks(), t.moe_ep_place.clone()))
}

/// Whether `cfg` drafts with a model drafter that runs on rank 0 alone (`--spec dflash|dspark`).
/// It reads the target's hidden states, which every rank has whole after the all-reduces: rank
/// 0 drafts with them and sends the drafts in its verify steps, ordinary steps of k + 1 rows
/// per sequence. Splitting the drafter too would halve its weight reads per pass, at the price
/// of every drafter state change sent to the others.
fn drafts_on_rank0(cfg: &LmConfig) -> bool {
    use crate::spec::SpecKind;
    matches!(cfg.spec.kind, SpecKind::Dflash | SpecKind::Dspark)
        && cfg.spec.model_drafter.is_none()
        && cfg.spec.token_drafter.is_none()
}

/// A rank above 0's drafting when rank 0 alone drafts ([`drafts_on_rank0`]): the drafter's
/// whole block read from its config (rank 0 sizes its verify rows by it), its weights left
/// unread. `None`: this rank makes its drafter as usual.
pub(super) fn rank0_drafting(cfg: &LmConfig) -> Result<Option<super::spec::Drafting>, LoadError> {
    match &cfg.tp_net {
        Some(t) if t.rank > 0 && drafts_on_rank0(cfg) => {}
        _ => return Ok(None),
    }
    let block = match cfg.spec.kind {
        // DSpark drafts no whole block (its confidence head may cut it).
        crate::spec::SpecKind::Dspark => None,
        _ => {
            let path = cfg.spec.draft.as_deref().ok_or_else(|| {
                device("--spec dflash needs --spec-draft <drafter checkpoint> on every rank")
            })?;
            crate::spec::dflash::block_drafts_of(path)
                .map_err(|e| device(format!("--spec-draft: {e}")))?
        }
    };
    Ok(Some(super::spec::Drafting::Rank0(block)))
}

/// What every rank's executor must be (checked after the load).
pub(super) fn check_exec(exec: &Executor, host_tier: bool) -> Result<(), LoadError> {
    if exec.n_stages() != 1 {
        return Err(device(format!(
            "{} stages on this rank: one GPU per rank",
            exec.n_stages()
        )));
    }
    if host_tier {
        return Err(device("a KV host tier: not supported"));
    }
    Ok(())
}

/// Rank 0: connects to the other ranks once they have loaded and makes `exec` lead them.
/// Returns the KV blocks every rank keeps (the fewest of any).
pub(super) fn lead(exec: &mut Executor, start: Start, n_blocks: usize) -> Result<usize, LoadError> {
    let n_layer = exec.model().hp.n_layer;
    let mut blocks = n_blocks;
    let hello = |r: Result<Reply, String>, to: usize, blocks: &mut usize| match r {
        Ok(Reply::Hello { n_blocks }) => {
            *blocks = (*blocks).min(n_blocks as usize);
            Ok(())
        }
        Ok(r) => Err(device(format!("rank {to} answered hello with {r:?}"))),
        Err(e) => Err(device(format!("rank {to}: {e}"))),
    };
    let n = match start.via {
        Via::Net { cfg, comm, .. } => {
            let n = cfg.n_ranks();
            let mut links = Vec::with_capacity(n - 1);
            for to in 1..n {
                let s = tp_net::dial_control(&cfg, to, WAIT)
                    .map_err(|e| device(format!("rank {to}: {e}")))?;
                let mut link = Link::over(s).map_err(|e| device(format!("rank {to}: {e}")))?;
                hello(
                    link.call(crate::node::tp_hello(n, to, n_layer, n_blocks)),
                    to,
                    &mut blocks,
                )?;
                links.push(link);
            }
            let mut lead = TpLead::new(links, comm, cfg.profile);
            lead.set_mtp_chains(cfg.mtp_chain);
            lead.set_mtp_local_rows(cfg.mtp_local_rows);
            exec.set_tp_lead(lead);
            n
        }
        Via::Pair { sum, end } => {
            let End::Lead(mut link, rank1) = end else {
                return Err(device("--tp-per-gpu: rank 0 has no link"));
            };
            // Rank 1 answers once it has loaded (its load error is logged on its thread).
            hello(
                link.call(crate::node::tp_hello(2, 1, n_layer, n_blocks)),
                1,
                &mut blocks,
            )?;
            exec.set_tp_lead(TpLead::pair(link, sum, rank1));
            2
        }
    };
    if blocks < n_blocks {
        // Every rank to the fewest blocks (rank 0's resize goes to the others first).
        exec.resize_kv(blocks).map_err(|e| {
            device(format!(
                "cannot resize the KV pool to the ranks' {blocks}: {e}"
            ))
        })?;
    }
    tracing::info!(
        ranks = n,
        kv_cache_tokens = blocks * crate::model::core::KV_BLOCK,
        "tensor parallelism over machines: rank 0 leads"
    );
    Ok(blocks)
}

/// A rank above 0: answers rank 0's requests with `exec` until rank 0 disconnects (the
/// connections are made once: the rank then stops). `ring_bufts`: where its rings are saved.
pub(super) fn follow(
    mut exec: Executor,
    start: Start,
    n_blocks: usize,
    ring_bufts: Vec<BufferType>,
) -> LoadError {
    let rank = start.rank;
    tracing::info!(
        rank,
        kv_cache_tokens = n_blocks * crate::model::core::KV_BLOCK,
        "tensor parallelism: loaded, waiting for rank 0"
    );
    let mut blocks = n_blocks;
    let (cfg, comm, control) = match start.via {
        Via::Net { cfg, comm, control } => (cfg, comm, control),
        Via::Pair { sum, end } => {
            let End::Serve(serve) = end else {
                return device("--tp-per-gpu: rank 1 has no link");
            };
            let mine = (2, rank, exec.model().hp.n_layer);
            let mut profile = StepProfile::new(0);
            let mut rows = Default::default();
            serve.serve(&mut |req| {
                let state = (&mut blocks, &mut profile, &mut rows);
                handle(&mut exec, req, mine, state, &ring_bufts, &Sums::Pair(&sum))
                    .unwrap_or_else(Reply::Error)
            });
            return device(format!("--tp-per-gpu: rank {rank}: rank 0 stopped"));
        }
    };
    let Some(mut control) = control else {
        return device("rank 0 has no control connection to take");
    };
    let mine = (cfg.n_ranks(), rank, exec.model().hp.n_layer);
    let stream = match control.accept() {
        Ok(s) => s,
        Err(e) => return device(format!("rank {rank}: waiting for rank 0: {e}")),
    };
    let mut profile = StepProfile::new(cfg.profile);
    let mut rows = Default::default();
    let r = crate::node::serve_one(stream, &mut |req| {
        let state = (&mut blocks, &mut profile, &mut rows);
        handle(&mut exec, req, mine, state, &ring_bufts, &Sums::Net(&comm))
            .unwrap_or_else(Reply::Error)
    });
    device(format!(
        "rank {rank}: rank 0 disconnected ({})",
        match r {
            Ok(()) => "its connection ended".to_string(),
            Err(e) => e.to_string(),
        }
    ))
}

/// What a rank above 0 keeps between requests: its KV blocks, its `--tp-profile`, and the
/// rows its last two steps captured ([`crate::model::core::StepOutput::hidden`], the last
/// first, which [`Request::TpMtpRows`] reads).
type FollowState<'s> = (
    &'s mut usize,
    &'s mut StepProfile,
    &'s mut [Vec<Vec<f32>>; 2],
);

/// An MTP pass's residual rows from `parts` (see [`RowsPart`]): the sent ones from `h` in
/// order, the others from `steps`, the captured layers of this rank's last steps (the last
/// first). A range outside them, `h` not used up, or (debug builds) rows whose bits differ
/// from rank 0's fail.
fn rows_of<'a>(
    parts: &[RowsPart],
    h: &'a [f32],
    steps: &'a [Vec<Vec<f32>>],
) -> Result<Vec<&'a [f32]>, String> {
    let mut at = 0usize;
    let out = parts
        .iter()
        .map(|p| match *p {
            RowsPart::Sent { len } => {
                let s = (at.checked_add(len as usize))
                    .and_then(|end| h.get(at..end))
                    .ok_or("MTP rows: more sent than the request holds")?;
                at += s.len();
                Ok(s)
            }
            RowsPart::Step {
                back,
                layer,
                start,
                len,
                check,
            } => {
                let rows = steps.get(back as usize).map_or(&[][..], |r| &r[..]);
                let s = rows
                    .get(layer as usize)
                    .and_then(|r| {
                        r.get(start as usize..(start as usize).checked_add(len as usize)?)
                    })
                    .ok_or_else(|| {
                        format!(
                            "MTP rows: floats {start}+{len} of captured layer {layer} are not \
                             in this rank's step {back} back ({} layers)",
                            rows.len()
                        )
                    })?;
                if check != 0 && crate::node::rows_check(s) != check {
                    return Err(format!(
                        "MTP rows: floats {start}+{len} of captured layer {layer} of step \
                         {back} back differ from rank 0's"
                    ));
                }
                Ok(s)
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    if at != h.len() {
        return Err(format!("MTP rows: {} sent floats, {at} used", h.len()));
    }
    Ok(out)
}

fn handle(
    exec: &mut Executor,
    req: Request,
    mine: (usize, usize, usize),
    (blocks, profile, rows): FollowState<'_>,
    ring_bufts: &[BufferType],
    comm: &Sums<'_>,
) -> Result<Reply, String> {
    Ok(match req {
        Request::TpHello {
            version,
            n_ranks,
            rank,
            n_layer,
            n_blocks,
        } => {
            crate::node::check_tp_hello((version, n_ranks, rank, n_layer), mine)?;
            let n = (*blocks).min(n_blocks as usize);
            if n < *blocks {
                exec.resize_kv(n).map_err(|e| e.to_string())?;
                *blocks = n;
            }
            tracing::info!(
                kv_cache_tokens = n * crate::model::core::KV_BLOCK,
                "tensor parallelism over machines: rank 0 says hello"
            );
            Reply::Hello { n_blocks: n as u32 }
        }
        Request::TpStep {
            groups,
            micro,
            taper,
            gather_budget,
        } => {
            exec.set_pipeline_micro(micro as usize);
            exec.set_micro_taper(taper as usize);
            exec.set_gather_budget(gather_budget as usize);
            let groups = crate::model::core::from_wire(&groups)?;
            let t0 = std::time::Instant::now();
            comm.mark_step(profile);
            // The rows of the step before the last are not read any more (rank 0 names at most
            // the last two steps' rows): this step copies into them.
            rows.swap(0, 1);
            exec.recycle_rows(std::mem::take(&mut rows[0]));
            rows[0] = exec.step(&groups).map_err(|e| e.to_string())?.hidden;
            comm.record(profile, t0.elapsed());
            if let Some(e) = comm.error() {
                return Err(e);
            }
            Reply::Done
        }
        Request::TpMtp {
            groups,
            gather_budget,
            h,
        } => {
            exec.set_gather_budget(gather_budget as usize);
            let groups = crate::model::core::from_wire(&groups)?;
            let t0 = std::time::Instant::now();
            comm.mark_step(profile);
            exec.mtp_step(&groups, &h).map_err(|e| e.to_string())?;
            comm.record(profile, t0.elapsed());
            if let Some(e) = comm.error() {
                return Err(e);
            }
            Reply::Done
        }
        Request::TpMtpRows {
            groups,
            gather_budget,
            parts,
            h,
        } => {
            // A prompt chunk's take-in reads this rank's copy of the step's rows (#780).
            exec.set_gather_budget(gather_budget as usize);
            let groups = crate::model::core::from_wire(&groups)?;
            let parts = rows_of(&parts, &h, &rows[..])?;
            let t0 = std::time::Instant::now();
            comm.mark_step(profile);
            exec.mtp_step_parts(&groups, &parts)
                .map_err(|e| e.to_string())?;
            comm.record(profile, t0.elapsed());
            if let Some(e) = comm.error() {
                return Err(e);
            }
            Reply::Done
        }
        Request::TpMtpChain {
            groups,
            gather_budget,
            h,
            caps,
            p_min,
        } => {
            // Rank 0's whole draft round: the same passes from the same picks (#771).
            exec.set_gather_budget(gather_budget as usize);
            let groups = crate::model::core::from_wire(&groups)?;
            let t0 = std::time::Instant::now();
            comm.mark_step(profile);
            let first: Vec<usize> = (0..caps.len()).collect();
            exec.mtp_chain(
                &groups,
                &[&h],
                &caps,
                &first,
                f32::from_bits(p_min),
                |_, _, _| {},
            )
            .map_err(|e| e.to_string())?;
            comm.record(profile, t0.elapsed());
            if let Some(e) = comm.error() {
                return Err(e);
            }
            Reply::Done
        }
        Request::TpCommit { commits } => {
            let commits: Vec<_> = commits
                .into_iter()
                .map(|(slot, keep)| (slot as usize, keep as usize))
                .collect();
            exec.commit(&commits).map_err(|e| e.to_string())?;
            Reply::Done
        }
        Request::SaveRing {
            ring,
            slot,
            n,
            restore,
        } => {
            exec.save_ring(ring as usize, slot as usize, n as usize, restore);
            Reply::Done
        }
        Request::EnableRingSaves { slots } => {
            let bytes = exec
                .enable_ring_saves(ring_bufts, slots as usize)
                .map_err(|e| e.to_string())?;
            tracing::info!(
                gib = bytes as f64 / (1u64 << 30) as f64,
                "tensor parallelism over machines: rings saved for prefix caching"
            );
            Reply::Done
        }
        Request::Resize { n_blocks } => {
            exec.resize_kv(n_blocks as usize)
                .map_err(|e| e.to_string())?;
            *blocks = n_blocks as usize;
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
        Request::Hello { .. } | Request::Step { .. } => {
            return Err(
                "a request of a model split by layers (--node-rank): this rank runs \
                        tensor parallelism over machines (--tp-rank)"
                    .into(),
            );
        }
    })
}
