//! Rank 0 of tensor parallelism over machines (#619, [`crate::tp_net`]): every step this
//! executor runs (and every MTP pass), every other rank runs too, at once, with its own share of
//! the heads and FFN rows; the graphs' all-reduces meet over the data connections. Rank 0 sends
//! each step (and each state change the others must mirror) on its control connection to each
//! rank before it runs the step itself, and reads the answers later: a rank's failure shows in
//! the communicator at once (the all-reduces stop), and in its answer at the next step.
//!
//! The same lead runs tensor parallelism over the two GPUs of a pair in one process
//! (`--tp-per-gpu`, #691): rank 1 is an executor on another thread of this process, reached
//! over a [`LocalLink`], and the graphs sum through the pair's NVLink all-reduce
//! ([`crate::model::core::pair::sum`]).

use std::sync::Arc;

use super::Executor;
use crate::model::core::pair::sum::PairSum;
use crate::model::core::{Group, ModelError, MtpOutput, StepOutput, head_split};
use crate::node::local::LocalLink;
use crate::node::{Link, Reply, Request, RowsPart};
use crate::tp_net::{CommHandle, StepProfile};

#[cfg(test)]
mod tests;

/// Steps sent before rank 0 reads the answers (the others finish each step with rank 0: their
/// answers are there by then).
const STEPS_AHEAD: usize = 2;

/// Answers left unread before a mirrored state change reads the oldest ones.
const MIRRORS_AHEAD: usize = 16;

/// The prompt tokens before a read-ahead chunk sent with it ([`Executor::tp_prefetch_rows`]).
const PREFETCH_BEFORE: usize = 32;

/// A control connection to another rank: over the network, or to a thread of this process.
enum RankLink {
    Net(Link),
    Local(LocalLink),
}

impl RankLink {
    fn send(&mut self, req: Request) -> Result<(), String> {
        match self {
            Self::Net(l) => l.send(req),
            Self::Local(l) => l.send(req),
        }
    }

    fn wait(&mut self) -> Result<Reply, String> {
        match self {
            Self::Net(l) => l.wait(),
            Self::Local(l) => l.wait(),
        }
    }

    fn owed(&self) -> usize {
        match self {
            Self::Net(l) => l.owed(),
            Self::Local(l) => l.owed(),
        }
    }
}

/// What the ranks' graphs sum through.
enum LeadComm {
    /// The network communicator of tensor parallelism over machines.
    Net(CommHandle),
    /// The NVLink all-reduce of a pair of GPUs in this process.
    Pair(Arc<PairSum>),
}

impl LeadComm {
    fn error(&self) -> Option<String> {
        match self {
            Self::Net(c) => c.comm().error(),
            Self::Pair(s) => s.error(),
        }
    }

    /// The network communicator, which `--tp-profile` reads.
    fn net(&self) -> Option<&CommHandle> {
        match self {
            Self::Net(c) => Some(c),
            Self::Pair(_) => None,
        }
    }
}

/// Rank 0's control connections to the other ranks, and the communicator its graphs use.
pub struct TpLead {
    links: Vec<RankLink>,
    comm: LeadComm,
    profile: StepProfile,
    /// A pair's rank 1 thread: joined once its link is closed.
    follower: Option<std::thread::JoinHandle<()>>,
    /// Draft rounds go to the other ranks as one request each (`--tp-mtp-chain`,
    /// [`Executor::mtp_chain`]).
    chains: bool,
    /// MTP passes name the rows of the last step's output instead of sending them
    /// (`--tp-mtp-local-rows`): the other ranks read their own copy.
    local_rows: bool,
    /// With `local_rows`: where the last step's captured layers are (address and floats, per
    /// layer of [`StepOutput::hidden`]) while the engine reads them; empty once handed back
    /// ([`Executor::recycle_rows`]) or before a step returned.
    step_rows: Vec<(usize, usize)>,
    /// The last step's rows, kept by the engine past its step ([`Executor::hold_rows`]).
    held: Option<Arc<Vec<Vec<f32>>>>,
    /// Where the rows of the step before the last are while `prev_held` keeps them.
    prev_rows: Vec<(usize, usize)>,
    prev_held: Option<Arc<Vec<Vec<f32>>>>,
}

impl Drop for TpLead {
    fn drop(&mut self) {
        // Rank 1 stops when its link closes; wait for it to free its GPU's memory, which it
        // cannot do once the process has begun to exit.
        self.links.clear();
        if let Some(h) = self.follower.take() {
            let _ = h.join();
        }
    }
}

fn err(e: String) -> ModelError {
    ModelError::Node(format!("tensor parallel: {e}"))
}

/// `parts` as [`RowsPart`]s: the ones within one of `steps[back]` (address and floats of each
/// captured layer of the last step, `back` 0, and the step before it) as a range of it, the
/// others sent (their values appended to `h`).
fn rows_parts(steps: &[&[(usize, usize)]], parts: &[&[f32]], h: &mut Vec<f32>) -> Vec<RowsPart> {
    const F: usize = std::mem::size_of::<f32>();
    parts
        .iter()
        .map(|p| {
            let (a, len) = (p.as_ptr() as usize, p.len());
            let within = (steps.iter().enumerate()).find_map(|(back, rows)| {
                (rows.iter().enumerate())
                    .find(|&(_, &(base, n))| len > 0 && a >= base && a + len * F <= base + n * F)
                    .map(|(layer, &(base, _))| (back, layer, base))
            });
            match within {
                Some((back, layer, base)) => RowsPart::Step {
                    back: back as u32,
                    layer: layer as u32,
                    start: ((a - base) / F) as u64,
                    len: len as u64,
                    check: if cfg!(debug_assertions) {
                        crate::node::rows_check(p)
                    } else {
                        0
                    },
                },
                None => {
                    h.extend_from_slice(p);
                    RowsPart::Sent { len: len as u64 }
                }
            }
        })
        .collect()
}

impl TpLead {
    /// `links`: to ranks 1.. in order, past their hello. `profile`: log the steps' timing
    /// every this many steps (0: never; `--tp-profile`).
    pub fn new(links: Vec<Link>, comm: CommHandle, profile: usize) -> Self {
        Self {
            links: links.into_iter().map(RankLink::Net).collect(),
            comm: LeadComm::Net(comm),
            profile: StepProfile::new(profile),
            follower: None,
            chains: false,
            local_rows: false,
            step_rows: Vec::new(),
            held: None,
            prev_rows: Vec::new(),
            prev_held: None,
        }
    }

    /// The lead of a pair of GPUs in this process: `link` to rank 1 (on the other GPU, past
    /// its hello), `sum` the all-reduce both ranks' graphs use, `follower` rank 1's thread.
    pub fn pair(
        link: LocalLink,
        sum: Arc<PairSum>,
        follower: Option<std::thread::JoinHandle<()>>,
    ) -> Self {
        Self {
            links: vec![RankLink::Local(link)],
            comm: LeadComm::Pair(sum),
            profile: StepProfile::new(0),
            follower,
            chains: false,
            local_rows: false,
            step_rows: Vec::new(),
            held: None,
            prev_rows: Vec::new(),
            prev_held: None,
        }
    }

    /// Sends each draft round to the other ranks as one request, whose further passes every
    /// rank runs on its own (`--tp-mtp-chain`, [`Executor::mtp_chain`]).
    pub fn set_mtp_chains(&mut self, on: bool) {
        self.chains = on;
    }

    /// See [`Self::set_mtp_chains`].
    pub(super) fn chains(&self) -> bool {
        self.chains
    }

    /// An MTP pass's residual rows that are a range of the last step's output (a prompt
    /// chunk's take-in) go to the other ranks as that range, not as values: every rank
    /// captured the same bits in that step (the layer is after the last sum, which adds the
    /// partials in rank order), so each reads its own copy (`--tp-mtp-local-rows`, #780).
    pub fn set_mtp_local_rows(&mut self, on: bool) {
        self.local_rows = on;
    }

    /// Queues `req` to every other rank.
    fn send(&mut self, req: &Request) -> Result<(), ModelError> {
        for (i, l) in self.links.iter_mut().enumerate() {
            l.send(req.clone())
                .map_err(|e| err(format!("rank {}: {e}", i + 1)))?;
        }
        Ok(())
    }

    /// Reads every answer owed by ranks with at least `owed` of them.
    fn settle(&mut self, owed: usize) -> Result<(), ModelError> {
        for (i, l) in self.links.iter_mut().enumerate() {
            if l.owed() >= owed.max(1) {
                l.wait().map_err(|e| err(format!("rank {}: {e}", i + 1)))?;
            }
        }
        Ok(())
    }

    /// Sends `req` to every other rank and waits for all the answers.
    pub fn call(&mut self, req: &Request) -> Result<Vec<Reply>, ModelError> {
        self.settle(1)?;
        self.send(req)?;
        self.links
            .iter_mut()
            .enumerate()
            .map(|(i, l)| l.wait().map_err(|e| err(format!("rank {}: {e}", i + 1))))
            .collect()
    }

    /// Queues a state change the others mirror; its answers are read with a later step's.
    fn mirror(&mut self, req: &Request) {
        // A step reads one answer before it sends: mirrors (one per new sequence on a model
        // with recurrent states) would leave answers behind for good. Old ones are read here,
        // never the last steps' (which keeps the steps sent ahead).
        for (i, l) in self.links.iter_mut().enumerate() {
            while l.owed() > MIRRORS_AHEAD {
                if let Err(e) = l.wait() {
                    tracing::error!(error = %e, rank = i + 1, "tensor parallel: a rank failed");
                    break;
                }
            }
        }
        if let Err(e) = self.send(req) {
            tracing::error!(error = %e, "tensor parallel: a state change was not sent");
        }
    }

    /// The communicator's failure, if any.
    fn failed(&self) -> Result<(), ModelError> {
        match self.comm.error() {
            Some(e) => Err(err(e)),
            None => Ok(()),
        }
    }
}

impl Executor {
    /// Makes this executor rank 0 of tensor parallelism over machines: the other ranks run its
    /// steps from now on.
    pub fn set_tp_lead(&mut self, lead: TpLead) {
        assert_eq!(
            self.stages.len(),
            1,
            "tensor parallelism over machines: one stage per rank"
        );
        self.tp = Some(lead);
    }

    /// The other ranks of tensor parallelism over machines, if this executor leads them.
    pub fn tp_lead(&mut self) -> Option<&mut TpLead> {
        self.tp.as_mut()
    }

    /// [`Self::step`] on every rank (see the module docs).
    pub(super) fn tp_step(&mut self, groups: &[Group<'_>]) -> Result<StepOutput, ModelError> {
        // A greedy decode or verify step's output head, half the vocabulary on each of two
        // ranks (`--tp-head-split`).
        let split = self.head_split
            && self.tp.as_ref().is_some_and(|l| l.links.len() == 1)
            && head_split::fits(groups, crate::ggml::model_bytes::DECODE_MAX_TOKENS);
        let req = Request::TpStep {
            groups: super::remote::to_wire(groups),
            micro: self.pipeline_micro as u32,
            taper: self.micro_taper as u32,
            gather_budget: self.gather_budget as u64,
            head: if split { 2 } else { 0 },
        };
        let lead = self.tp.as_mut().expect("tensor-parallel lead");
        // The last step's rows become the step before's while the engine holds them (the
        // other ranks keep theirs too); the older ones go back to the spare rows.
        let older = lead.prev_held.take();
        lead.prev_rows.clear();
        if let Some(held) = lead.held.take() {
            lead.prev_rows = std::mem::take(&mut lead.step_rows);
            lead.prev_held = Some(held);
        }
        lead.step_rows.clear();
        if let Some(rows) = older.and_then(|o| Arc::try_unwrap(o).ok()) {
            self.spare_rows.keep(rows, super::BOUNCE_MIN_BYTES);
        }
        self.set_head_half(u8::from(split));
        let out = self.tp_run(&req, |e| e.step_here(groups));
        self.head_half = 0;
        let mut out = out?;
        if split {
            self.tp_join_head(&mut out)?;
        }
        let lead = self.tp.as_mut().expect("tensor-parallel lead");
        if lead.local_rows {
            lead.step_rows = (out.hidden.iter())
                .map(|v| (v.as_ptr() as usize, v.len()))
                .collect();
        }
        Ok(out)
    }

    /// After this rank's share of a split head ([`head_split`]): waits for rank 1's answer to
    /// the step (the last request sent: state changes go out between steps) and puts each
    /// row's pick over both shares in `out`'s argmaxes.
    fn tp_join_head(&mut self, out: &mut StepOutput) -> Result<(), ModelError> {
        let mine = self.take_head_vals();
        let lead = self.tp.as_mut().expect("tensor-parallel lead");
        let reply = lead.links[0]
            .wait()
            .map_err(|e| err(format!("rank 1: {e}")))?;
        let Reply::Step { argmax, rows, .. } = reply else {
            return Err(err("rank 1 answered a split step without its share".into()));
        };
        let theirs = rows.first().map_or(&[][..], |v| v.as_slice());
        out.argmax = head_split::join((&out.argmax[..], &mine[..]), (&argmax[..], theirs))
            .ok_or_else(|| err("rank 1's share of a split step has other rows".into()))?;
        Ok(())
    }

    /// The last step's rows are handed back ([`Self::recycle_rows`]): MTP passes send their
    /// rows from now on, until the next step.
    pub(super) fn tp_rows_gone(&mut self) {
        if let Some(lead) = self.tp.as_mut() {
            lead.step_rows.clear();
            lead.held = None;
        }
    }

    /// [`Self::hold_rows`]: keeps `rows` if they are the last step's (named by MTP passes until
    /// the step after the next), forgets the last step's otherwise.
    pub(super) fn tp_hold_rows(&mut self, rows: Arc<Vec<Vec<f32>>>) {
        let Some(lead) = self.tp.as_mut() else {
            return;
        };
        let same = !lead.step_rows.is_empty()
            && rows.len() == lead.step_rows.len()
            && (rows.iter().zip(&lead.step_rows))
                .all(|(v, &(a, n))| v.as_ptr() as usize == a && v.len() == n);
        if same {
            lead.held = Some(rows);
        } else {
            lead.step_rows.clear();
            lead.held = None;
        }
    }

    /// [`Self::mtp_step_parts`] on every rank: the head is split like the target
    /// (`tp_net::shard::assistant_view`), its attention reading each rank's share of the cache.
    pub(super) fn tp_mtp(
        &mut self,
        groups: &[Group<'_>],
        parts: &[&[f32]],
    ) -> Result<MtpOutput, ModelError> {
        let lead = self.tp.as_ref().expect("tensor-parallel lead");
        let req = if lead.local_rows {
            let (parts, h) = self.tp_rows_wire(parts);
            Request::TpMtpRows {
                groups: super::remote::to_wire(groups),
                gather_budget: self.gather_budget as u64,
                parts,
                h,
            }
        } else {
            Request::TpMtp {
                groups: super::remote::to_wire(groups),
                gather_budget: self.gather_budget as u64,
                h: parts.concat(),
            }
        };
        self.tp_run(&req, |e| e.mtp_step_here(groups, parts))
    }

    /// An MTP pass's residual rows `parts` as the other ranks read them: with
    /// `--tp-mtp-local-rows` the ones within the last two steps' rows as ranges of their own
    /// copies, the rest sent (the floats to send second); else all sent, as one part.
    pub(super) fn tp_rows_wire(&self, parts: &[&[f32]]) -> (Vec<RowsPart>, Vec<f32>) {
        let lead = self.tp.as_ref().expect("tensor-parallel lead");
        if lead.local_rows {
            let mut h = Vec::new();
            let parts = rows_parts(&[&lead.step_rows, &lead.prev_rows], parts, &mut h);
            (parts, h)
        } else {
            let h = parts.concat();
            let parts = vec![RowsPart::Sent {
                len: h.len() as u64,
            }];
            (parts, h)
        }
    }

    /// Sends `req` to the other ranks, then runs it here with `run` (a pass of `--tp-profile`).
    pub(super) fn tp_run<T>(
        &mut self,
        req: &Request,
        run: impl FnOnce(&mut Self) -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        let lead = self.tp.as_mut().expect("tensor-parallel lead");
        lead.failed()?;
        lead.settle(STEPS_AHEAD)?;
        let t0 = std::time::Instant::now();
        let profiling = lead.profile.enabled();
        if profiling && let Some(c) = lead.comm.net() {
            c.comm().mark_step();
        }
        lead.send(req)?;
        let send = t0.elapsed();
        let out = run(self);
        let lead = self.tp.as_mut().expect("tensor-parallel lead");
        if profiling && let Some(c) = lead.comm.net() {
            let comm = c.comm().clone();
            let laps = self.step_laps.take();
            lead.profile.record(&comm, t0.elapsed(), send, laps);
        }
        lead.failed()?;
        out
    }

    /// The other ranks' mirror of [`Self::save_ring`].
    pub(super) fn tp_save_ring(&mut self, r: usize, slot: usize, n: usize, restore: bool) {
        if let Some(lead) = self.tp.as_mut() {
            lead.mirror(&Request::SaveRing {
                ring: r as u32,
                slot: slot as u32,
                n: n as u32,
                restore,
            });
        }
    }

    /// The other ranks' mirror of a change to the recurrent states (clearing a slot, copying
    /// slots, a commit), queued before this rank makes it: their next step runs after it.
    pub(super) fn tp_mirror_states(&mut self, req: impl FnOnce() -> Request) {
        if let Some(lead) = self.tp.as_mut() {
            lead.mirror(&req());
        }
    }

    /// The other ranks' read-ahead of a prompt chunk's table rows (`tokens`, after `before`,
    /// the prompt's last tokens before them), as this rank reads its own (`next`: the next
    /// chunk while this one's step computes; else the first one on its arrival): without it
    /// every other rank read each prompt step's rows from the SSD inside that step, which rank
    /// 0 then waited for at the sums (#1176). Nothing without a table read through a store.
    pub fn tp_prefetch_rows(&mut self, before: &[u32], tokens: &[u32], next: bool) {
        if tokens.is_empty() || self.model.ngram.is_none() {
            return;
        }
        if let Some(lead) = self.tp.as_mut() {
            // A row's n-gram reaches back a few tokens: 32 is more than any table's, and too
            // few would only read other rows (it is a hint).
            let before = &before[before.len().saturating_sub(PREFETCH_BEFORE)..];
            lead.mirror(&Request::TpPrefetch {
                before: before.to_vec(),
                tokens: tokens.to_vec(),
                next,
            });
        }
    }

    /// The other ranks' [`Self::resize_kv`], before this rank's.
    pub(super) fn tp_resize_kv(&mut self, n_blocks: usize) -> Result<(), ModelError> {
        match self.tp.as_mut() {
            Some(lead) => lead
                .call(&Request::Resize {
                    n_blocks: n_blocks as u32,
                })
                .map(|_| ()),
            None => Ok(()),
        }
    }

    /// The other ranks' [`Self::enable_ring_saves`], before this rank's: all of them or none.
    pub(super) fn tp_enable_ring_saves(&mut self, slots: usize) -> Result<(), ModelError> {
        match self.tp.as_mut() {
            Some(lead) => lead
                .call(&Request::EnableRingSaves {
                    slots: slots as u32,
                })
                .map(|_| ()),
            None => Ok(()),
        }
    }
}
