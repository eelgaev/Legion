//! A model's tiered expert tensors and the VRAM cache over them (see the module docs of
//! [`super`] for the design).
//!
//! Per MoE layer and expert tensor (`ffn_{up,gate,down}_exps`), a tensor of
//! `n_slots + n_expert` experts: slots `0..n_slots` hold cached copies, `n_slots + e` is expert
//! `e` itself. Per device, an I32 table `[n_expert, layers]` maps each layer's experts to the
//! index the graph uses ([`TieredLayer::remap`]), and a route log receives those indices for
//! every step of at most [`LOG_TOKENS`] tokens, so [`ExpertResidency::after_step`] learns what
//! was routed and whether VRAM served it without any extra kernel.
//!
//! Two backings:
//! - **CUDA** (`feature = "cuda"`): the slots are VRAM and the experts are the host arena's
//!   bytes, mapped right after the slots (`cuda::tiered`); swaps are DMA copies on a side
//!   stream.
//! - **Host copy**: one host buffer holding the slots and a copy of the experts. The CPU
//!   backend reads it; it exists so the whole path (remap, logging, swaps) runs in CPU tests,
//!   where results must be bit-identical to the plain expert tensors.

#[cfg(feature = "cuda")]
use std::collections::VecDeque;
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::policy::{
    LayerSlots, LendLayer, MAX_FIT_TRIES, SwapRule, allocate_slots_in, auto_keep_carved,
    layers_left_for, lowest_experts, plan_swaps, shrunk_budget,
};
use super::profile::Profile;
use super::{Fetch, ResidencyConfig};
use crate::ggml::model_bytes::ExpertTraffic;
use crate::ggml::{BufferType, Context, GgmlError, Tensor, types};

/// Tokens a step may have for its routing to be logged (decode and verify steps).
pub const LOG_TOKENS: usize = 256;

/// Logged steps between the cache's periodic hit-rate reports.
const REPORT_EVERY: u64 = 512;

/// The expert tensors of a MoE layer, in the order [`TieredLayer::weight`] takes.
pub const KINDS: [&str; 3] = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"];

/// Where one MoE layer's tiered tensors and tables go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlace {
    /// The CUDA device (ggml index) that runs the layer; `None` for the CPU backend.
    pub cuda: Option<usize>,
    /// The memory its table and route log live in (the device's own).
    pub buft: BufferType,
}

#[cfg(feature = "cuda")]
type Copier = super::cuda::Copier;
#[cfg(feature = "cuda")]
type CopyEvent = super::cuda::CopyEvent;

/// Copies queued on a device and the slots they fill.
#[cfg(feature = "cuda")]
struct InFlight {
    event: CopyEvent,
    slots: Vec<(usize, u32)>,
    /// A batch of the bounded host tier's swaps ([`super::host_tier::HostTier::swap`]) it
    /// has not been told the end of yet.
    tier: bool,
}

struct Device {
    place: LayerPlace,
    /// `expert_table` `[n_expert, rows]` and `expert_log` `[n_used × LOG_TOKENS × rows]`, I32,
    /// and with [`ResidencyConfig::learn_long`] the tally ([`Self::tally`]).
    ctx: Context,
    rows: usize,
    /// [`Fetch::Copy`]: where each expert kind's gathered experts start in `expert_fetch`.
    fetch_at: [usize; 3],
    #[cfg(feature = "cuda")]
    copier: Option<Copier>,
    #[cfg(feature = "cuda")]
    inflight: VecDeque<InFlight>,
    /// Prefill staging buffers, while the device's slots are lent ([`ExpertResidency::lend`]),
    /// and between lends while [`Self::kept`].
    #[cfg(feature = "cuda")]
    staging: Option<super::staging::Staging>,
    /// Whether the staging buffers and staged ranges stay for the next lend
    /// ([`ResidencyConfig::stage_reuse`]): made of the slots' own allocations, which the slots
    /// map again between lends ([`Layer::shared`]).
    #[cfg(feature = "cuda")]
    kept: bool,
    /// Whether the kept staging is parked (the slots are not lent): no step stages.
    #[cfg(feature = "cuda")]
    parked: bool,
    /// Per kind, the staging buffer size the device lends its slots for, if it stages.
    #[cfg(feature = "cuda")]
    lend_bytes: Option<[usize; 3]>,
    /// Whether the whole table must be uploaded (rows of single layers: [`Layer::dirty`]).
    dirty: bool,
    /// [`ResidencyConfig::ep_prefetch`]: the staging ring's layout (`expert_prefetch_ring`).
    ring: Option<super::ep_prefetch::RingLayout>,
}

impl Device {
    /// The `--glm-prefetch` staging ring ([`Self::ring`]).
    fn prefetch_ring(&self) -> Option<(Tensor<'_>, super::ep_prefetch::RingLayout)> {
        Some((self.ctx.tensor("expert_prefetch_ring")?, self.ring?))
    }

    /// The staging buffers, while the slots are lent.
    #[cfg(feature = "cuda")]
    fn stage(&self) -> Option<&super::staging::Staging> {
        self.staging.as_ref().filter(|_| !self.parked)
    }

    #[cfg(feature = "cuda")]
    fn stage_mut(&mut self) -> Option<&mut super::staging::Staging> {
        if self.parked {
            return None;
        }
        self.staging.as_mut()
    }

    fn table(&self) -> Tensor<'_> {
        self.ctx.tensor("expert_table").expect("table")
    }

    /// The table of packed staged passes ([`Layer::stage_ids`]), `[n_expert, rows]` I32.
    fn stage_table(&self) -> Tensor<'_> {
        self.ctx.tensor("expert_stage_table").expect("stage table")
    }

    fn log(&self) -> Tensor<'_> {
        self.ctx.tensor("expert_log").expect("log")
    }

    /// The next-layer router guesses, laid out as [`Self::log`] with raw expert ids (`-1`
    /// where no guess was written) ([`ResidencyConfig::ep_predict`]).
    fn pred_log(&self) -> Option<Tensor<'_>> {
        self.ctx.tensor("expert_pred_log")
    }

    /// Per entry, which of its layer's experts their owners cache (1) or not (0), `[n_expert,
    /// rows]` I32 ([`ResidencyConfig::ep_cached`]).
    fn cached(&self) -> Option<Tensor<'_>> {
        self.ctx.tensor("expert_cached")
    }

    /// Long steps' routing counts per expert, `[n_expert, rows]` F32, and the one-hot rows
    /// they are counted with, `[n_expert, n_expert]` ([`ResidencyConfig::learn_long`]).
    fn tally(&self) -> Option<(Tensor<'_>, Tensor<'_>)> {
        Some((
            self.ctx.tensor("expert_tally")?,
            self.ctx.tensor("expert_eye")?,
        ))
    }
}

struct Layer {
    dev: usize,
    row: usize,
    slots: LayerSlots,
    /// One context per kind, each holding the tiered tensor (and, for CUDA, its range).
    tensors: Vec<Context>,
    /// Bytes of one expert per kind.
    expert_bytes: [usize; 3],
    /// Routed (token, expert) pairs per expert since the start, for the saved profile.
    counts: Vec<u64>,
    /// The bytes of routed experts read since the start, by where (#391; see [`Self::read`]).
    traffic: ExpertTraffic,
    /// Per looked-up id (slot or `n_slots + expert`): the pass that last read it, to count each
    /// once per pass.
    seen: Vec<u32>,
    pass: u32,
    /// Prefill staging: per kind, a tensor over the same slots followed by a staging buffer
    /// (see [`super::staging`]); empty without staging.
    #[cfg(feature = "cuda")]
    staged: Vec<Context>,
    /// Per kind, the slots' own allocations `(vram, tail)` that the device's kept staging
    /// buffers are made of ([`Device::kept`]): a restore maps them back into the slots.
    #[cfg(feature = "cuda")]
    shared: [Option<(u64, u64)>; 3],
    /// Per kind, the staged range kept with the device's staging ([`Device::kept`]); each lend
    /// makes [`Self::staged`] views of them.
    #[cfg(feature = "cuda")]
    ranges: Vec<super::cuda::StagedRange>,
    /// Whether the slots have VRAM: not while lent ([`ExpertResidency::lend`]), nor after a
    /// restore that found no room (retried at the next one). No swaps into a layer without.
    backed: bool,
    /// Whether the layer's table row changed since its last upload.
    dirty: bool,
    /// [`ResidencyConfig::lend_spread`]: the slots `0..keep` keep their experts while lent
    /// (the slots' VRAM is two allocations, the rest of it is lent: [`super::cuda::unback_tail`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the CUDA lend paths only
    keep: Option<usize>,
    /// Its host part is the bounded host tier's ([`super::host_tier`]): RAM slots and staging
    /// places, not every expert's bytes.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the CUDA lend paths only
    tier: bool,
    /// Its tail is lent and not refilled yet (the slots before it still have VRAM).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the CUDA lend paths only
    tail_lent: bool,
    /// While lent with packed staging: per expert, its id in the staged tensors (a kept slot,
    /// or `stage_first + i` for the `i`-th uncached expert, `stage_miss[i]`); empty otherwise.
    stage_ids: Vec<i32>,
    stage_first: usize,
    stage_miss: Vec<u32>,
    /// Staging copies every uncached expert, owned or not ([`stages_all`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))] // read by the CUDA staging paths only
    stage_all: bool,
    /// [`ResidencyConfig::ep_predict`]: the owned experts its previous logged pass chose.
    prev_chosen: Vec<u32>,
}

/// Counts entry `l`'s logged pass (`mine`, its looked-up ids) on CUDA device `gpu` against
/// `guess`, the next-layer router's raw expert ids logged for it, and the entry's previous
/// pass, which it then replaces ([`super::ep_prefetch`]). Returns whether the entry had a
/// guess (an entry whose layer follows no MoE layer has none, and is not counted).
fn predict_entry(
    stats: &mut super::ep_prefetch::PrefetchStats,
    l: &mut Layer,
    gpu: usize,
    mine: &[i32],
    guess: &[i32],
) -> bool {
    let chosen = l.owned_chosen(mine);
    let had = guess.iter().any(|&id| id >= 0);
    if had {
        let next = super::ep_prefetch::owned_distinct(guess, |e| l.slots.owns(e));
        stats.add(
            &super::ep_prefetch::EntryPredicted {
                gpu,
                chosen: &chosen,
                next: &next,
                prev: &l.prev_chosen,
            },
            |e| !l.slots.serves(e),
        );
    }
    l.prev_chosen = chosen.into_iter().map(|(e, _)| e).collect();
    had
}

/// Bytes per staging copy at most (see [`Layer::stage_copies`]).
#[cfg(feature = "cuda")]
const STAGE_PIECE: usize = 32 << 20;

/// `n` uncached experts (in staging order) in `pieces` groups of near-equal size, at most one
/// per expert (one group for `pieces` 0).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn piece_groups(n: usize, pieces: usize) -> Vec<std::ops::Range<usize>> {
    let g = pieces.max(1).min(n);
    (0..g).map(|i| i * n / g..(i + 1) * n / g).collect()
}

/// Whether cache entry `il` stages while lent: its model layer (`real`, by entry, for expert
/// parallelism's copies; `None`: the entry itself) is below `limit`. With
/// [`super::ResidencyConfig::ep_stage_all`] (`all`), a copy past the model's layers never
/// stages: its layer's prompts run on the layer's own entry alone (`--pair-tp`, #681), so the
/// copy lends all of its slots, for the owner's staging buffers and the prompt's compute.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn stages_entry(real: Option<&[usize]>, il: usize, limit: usize, all: bool) -> bool {
    let r = real.and_then(|r| r.get(il).copied()).unwrap_or(il);
    r < limit && !(all && r != il)
}

/// How many experts a staging entry may copy in a pass at most, for the staging buffers' size
/// ([`auto_keep_carved`]): every expert of the layer when it computes them all ([`stages_all`],
/// `all`), else those it owns (`owned`; `None`: all).
fn stage_width(owned: Option<&std::ops::Range<u32>>, all: bool, n_expert: usize) -> usize {
    match owned {
        Some(r) if !all => r.len(),
        _ => n_expert,
    }
}

/// Whether cache entry `il`'s staging copies every uncached expert, not only those it owns:
/// with [`super::ResidencyConfig::ep_stage_all`] (`on`), a model layer's own entry (`real`, by
/// entry: the layer it serves), which computes all of the layer's experts in the steps no other
/// entry takes part in. A copy past the model's layers computes only its own share.
fn stages_all(real: Option<&[usize]>, il: usize, on: bool) -> bool {
    on && real.and_then(|r| r.get(il)).is_some_and(|&r| r == il)
}

/// Each entry's row on its device (`dev_of`, by entry; `usize::MAX`: no entry): a device's
/// entries numbered in the order its graphs run them, by model layer (`real` as in
/// [`stages_entry`]: an expert-parallel copy takes its layer's place among the device's own
/// layers). Staging's ring gives a device's rows its buffers in turn, in row order. Without
/// copies, the entries' own order.
fn run_rows(dev_of: &[usize], real: Option<&[usize]>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dev_of.len())
        .filter(|&il| dev_of[il] != usize::MAX)
        .collect();
    order.sort_by_key(|&il| (real.and_then(|r| r.get(il).copied()).unwrap_or(il), il));
    let mut next: Vec<usize> = Vec::new();
    let mut rows = vec![usize::MAX; dev_of.len()];
    for il in order {
        let d = dev_of[il];
        if next.len() <= d {
            next.resize(d + 1, 0);
        }
        rows[il] = next[d];
        next[d] += 1;
    }
    rows
}

/// Runs of consecutive experts in `miss`: `(first index, first expert, length)`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn runs_of(miss: &[u32]) -> Vec<(usize, u32, usize)> {
    let mut runs: Vec<(usize, u32, usize)> = Vec::new();
    for (i, &e) in miss.iter().enumerate() {
        match runs.last_mut() {
            Some((_, start, len)) if *start + *len as u32 == e => *len += 1,
            _ => runs.push((i, e, 1)),
        }
    }
    runs
}

/// [`super::policy::LayerSlots::repack_kept`]'s moves `(slot, from)` as runs `(slot, from,
/// len)`: consecutive slots filled from consecutive sources on the same side of `n_slots`
/// (slots, or host experts), one copy per run and kind.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn move_runs(moves: &[(u32, u32)], n_slots: u32) -> Vec<(u32, u32, u32)> {
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for &(s, from) in moves {
        match runs.last_mut() {
            Some((s0, f0, len))
                if *s0 + *len == s && *f0 + *len == from && (*f0 < n_slots) == (from < n_slots) =>
            {
                *len += 1
            }
            _ => runs.push((s, from, 1)),
        }
    }
    runs
}

/// A layer's staging copies by kind: the pads before and after its experts, and each group's
/// ([`piece_groups`]) runs.
#[cfg(feature = "cuda")]
struct Parts {
    head: [Vec<super::staging::Copy>; 3],
    body: [Vec<Vec<super::staging::Copy>>; 3],
    tail: [Vec<super::staging::Copy>; 3],
}

#[cfg(feature = "cuda")]
impl Parts {
    fn new(groups: usize) -> Self {
        Self {
            head: Default::default(),
            body: std::array::from_fn(|_| vec![Vec::new(); groups]),
            tail: Default::default(),
        }
    }

    /// In order: kind by kind (head, groups, tail); or `split`, every pad, the mark of the
    /// cached ids (up to `cached`), then group by group its kinds and its mark (its ids up to
    /// `ends[g]`).
    fn copies(self, split: bool, cached: usize, ends: &[usize]) -> Vec<super::staging::Copy> {
        use super::staging::{CACHED_MARK, MARK, PIECE_MARK};
        let mut out = Vec::new();
        if !split {
            for k in 0..3 {
                out.extend_from_slice(&self.head[k]);
                self.body[k].iter().for_each(|g| out.extend_from_slice(g));
                out.extend_from_slice(&self.tail[k]);
            }
            return out;
        }
        for k in 0..3 {
            out.extend_from_slice(&self.head[k]);
            out.extend_from_slice(&self.tail[k]);
        }
        out.push((MARK, cached, CACHED_MARK));
        for (g, &end) in ends.iter().enumerate() {
            for k in 0..3 {
                out.extend_from_slice(&self.body[k][g]);
            }
            out.push((MARK, end, PIECE_MARK));
        }
        out
    }
}

impl Layer {
    fn tensor(&self, k: usize) -> Tensor<'_> {
        self.tensors[k].tensor(KINDS[k]).expect("tiered tensor")
    }

    /// The copies that fill this layer's staging buffers: every uncached expert's bytes, in
    /// runs of consecutive experts, to the same offset of the staged range. With `pieces`
    /// (`--prefill-stage-pieces`), the uncached experts in that many groups, each group's
    /// kinds together and followed by a mark ([`super::staging::MARK`]), after the pads and
    /// the cached experts' mark.
    #[cfg(feature = "cuda")]
    fn stage_copies(&self, pieces: usize) -> Vec<super::staging::Copy> {
        if !self.stage_ids.is_empty() {
            return self.packed_copies(pieces);
        }
        let n_slots = self.slots.n_slots();
        let table = self.slots.table();
        // Expert parallelism: only this entry's own experts (the others' choices never read
        // its staged copies), unless it computes them all ([`stages_all`]).
        let miss: Vec<u32> = (0..table.len() as u32)
            .filter(|&e| {
                table[e as usize] as usize >= n_slots && (self.stage_all || self.slots.owns(e))
            })
            .collect();
        let groups = piece_groups(miss.len(), pieces);
        let n_expert = table.len();
        let mut parts = Parts::new(groups.len());
        for (k, kind) in KINDS.iter().enumerate() {
            // SAFETY: reading the data pointer of a live tensor.
            let staged =
                unsafe { (*self.staged[k].tensor(kind).expect("staged").raw()).data } as usize;
            let b = self.expert_bytes[k];
            // ggml-cuda's quantized matmuls read up to a padded row past an expert's last row
            // when K is not a multiple of 512. In a staging buffer, what follows a copied run
            // can be a cached expert's entry (never copied), holding another layer's bytes of
            // another type: the run's copy takes the first bytes of the next expert's host
            // part along, so those reads stay finite. So does the entry after the last slot
            // (expert 0's) and the tail, zeroed when nothing is copied there.
            let pad = (self.tensor(k).ne()[0] % 512 != 0).then_some(4096.min(b));
            for (g, r) in groups.iter().enumerate() {
                for (_, e, len) in runs_of(&miss[r.clone()]) {
                    let e = e as usize;
                    let at = (n_slots + e) * b;
                    let src = self.addr(k, n_slots + e);
                    let more = pad.filter(|_| e + len < n_expert).unwrap_or(0);
                    let total = len * b + more;
                    // In pieces: the copy engine takes other streams' copies (a step's inputs)
                    // between them instead of behind a whole layer's.
                    for off in (0..total).step_by(STAGE_PIECE) {
                        let n = STAGE_PIECE.min(total - off);
                        parts.body[k][g].push((staged + at + off, src + off, n));
                    }
                }
            }
            if let Some(pad) = pad {
                if table.first().is_some_and(|&id| (id as usize) < n_slots) {
                    parts.tail[k].push((staged + n_slots * b, 0, pad));
                }
                parts.tail[k].push((staged + (n_slots + n_expert) * b, 0, pad));
            }
        }
        let ends = groups
            .iter()
            .map(|r| n_slots + miss[r.end - 1] as usize + 1)
            .collect::<Vec<_>>();
        parts.copies(pieces > 0, n_slots, &ends)
    }

    /// The copies `(dst, src, bytes)` of a re-pack's moves ([`super::policy::LayerSlots::repack_kept`]),
    /// every kind.
    #[cfg(feature = "cuda")]
    fn repack_copies(&self, moves: &[(u32, u32)]) -> Vec<(usize, usize, usize)> {
        let runs = move_runs(moves, self.slots.n_slots() as u32);
        let mut out = Vec::with_capacity(3 * runs.len());
        for k in 0..3 {
            for &(s, from, len) in &runs {
                let n = len as usize * self.expert_bytes[k];
                out.push((self.addr(k, s as usize), self.addr(k, from as usize), n));
            }
        }
        out
    }

    /// [`Self::stage_copies`] with packed staging ([`Self::stage_ids`]): the uncached experts
    /// back to back from id `stage_first` on, in runs of consecutive experts.
    #[cfg(feature = "cuda")]
    fn packed_copies(&self, pieces: usize) -> Vec<super::staging::Copy> {
        let groups = piece_groups(self.stage_miss.len(), pieces);
        let n_slots = self.slots.n_slots();
        let mut parts = Parts::new(groups.len());
        for (k, kind) in KINDS.iter().enumerate() {
            // SAFETY: reading the data pointer of a live tensor.
            let staged =
                unsafe { (*self.staged[k].tensor(kind).expect("staged").raw()).data } as usize;
            let b = self.expert_bytes[k];
            let (at, split) = super::cuda::tiered_split(self.tensor(k)).unwrap_or((0, 0));
            // The quantized matmuls' reads past an expert's last row (see `stage_copies`):
            // the last kept slot's run into the staging buffer's head (up to the first packed
            // id), and the last packed expert's into its tail, both zeroed.
            let pad = (self.tensor(k).ne()[0] % 512 != 0).then_some(4096.min(b));
            if let Some(pad) = pad {
                let head = (at + self.stage_first * b).saturating_sub(split);
                if head > 0 {
                    parts.head[k].push((staged - at + split, 0, pad.min(head)));
                }
            }
            for (g, r) in groups.iter().enumerate() {
                for (i, e, len) in runs_of(&self.stage_miss[r.clone()]) {
                    let dst = staged + (self.stage_first + r.start + i) * b;
                    let src = self.addr(k, n_slots + e as usize);
                    let total = len * b;
                    for off in (0..total).step_by(STAGE_PIECE) {
                        let n = STAGE_PIECE.min(total - off);
                        parts.body[k][g].push((dst + off, src + off, n));
                    }
                }
            }
            if let Some(pad) = pad {
                let end = self.stage_first + self.stage_miss.len();
                parts.tail[k].push((staged + end * b, 0, pad));
            }
        }
        let ends = groups
            .iter()
            .map(|r| self.stage_first + r.end)
            .collect::<Vec<_>>();
        parts.copies(pieces > 0, self.stage_first, &ends)
    }

    /// The experts a staged pass copies into this layer's staging buffers, each with its id in
    /// the staged tensors: the packed ids ([`Self::stage_ids`]), or `n_slots + expert` for
    /// every expert the table reads from host memory.
    #[cfg(feature = "cuda")]
    fn stage_misses(&self) -> Vec<(u32, usize)> {
        if !self.stage_ids.is_empty() {
            let first = self.stage_first;
            let ids = self.stage_miss.iter().enumerate();
            return ids.map(|(i, &e)| (e, first + i)).collect();
        }
        let n_slots = self.slots.n_slots();
        let table = self.slots.table();
        (0..table.len() as u32)
            .filter(|&e| table[e as usize] as usize >= n_slots)
            .map(|e| (e, n_slots + e as usize))
            .collect()
    }

    /// [`Self::stage_copies`] with the bounded host tier: expert `i` of [`Self::stage_misses`]
    /// comes from `srcs[i]` ([`super::host_tier::HostTier::stage_pass`]). In order: the zeroed
    /// pads, the experts in RAM slots, then each batch of experts read from the checkpoint
    /// between the calls that wait for its reads and give its places back
    /// ([`super::staging::CALL`]). Copies of neighbours on both sides are one copy.
    #[cfg(feature = "cuda")]
    fn tier_copies(&self, srcs: &[super::host_tier::PassSrc]) -> Vec<super::staging::Copy> {
        use super::host_tier::{PassSrc, release_arg, wait_arg};
        use super::staging::CALL;
        let n_slots = self.slots.n_slots();
        let n_expert = self.slots.n_expert();
        let miss = self.stage_misses();
        debug_assert_eq!(miss.len(), srcs.len());
        let packed = !self.stage_ids.is_empty();
        let staged: [usize; 3] = std::array::from_fn(|k| {
            // SAFETY: reading the data pointer of a live tensor.
            unsafe { (*self.staged[k].tensor(KINDS[k]).expect("staged").raw()).data as usize }
        });
        let mut out: Vec<super::staging::Copy> = Vec::new();
        // The quantized matmuls' reads past an expert's last row (see `stage_copies`): zeroes
        // after every run of copied ids that no copy follows, and where the ids start.
        for k in 0..3 {
            let b = self.expert_bytes[k];
            let Some(pad) = (self.tensor(k).ne()[0] % 512 != 0).then_some(4096.min(b)) else {
                continue;
            };
            if packed {
                let (at, split) = super::cuda::tiered_split(self.tensor(k)).unwrap_or((0, 0));
                let head = (at + self.stage_first * b).saturating_sub(split);
                if head > 0 {
                    out.push((staged[k] - at + split, 0, pad.min(head)));
                }
            } else if miss.first().is_none_or(|m| m.0 != 0) {
                out.push((staged[k] + n_slots * b, 0, pad));
            }
            for (i, &(_, id)) in miss.iter().enumerate() {
                if miss.get(i + 1).is_none_or(|m| m.1 != id + 1) {
                    out.push((staged[k] + (id + 1) * b, 0, pad));
                }
            }
        }
        debug_assert!(packed || miss.last().is_none_or(|m| m.1 < n_slots + n_expert));
        // One kind's copies of the experts `picked` (positions in `miss`, from index `at`
        // past the slots), neighbours joined up to a staging piece.
        let copy = |out: &mut Vec<super::staging::Copy>, picked: &[(usize, usize)]| {
            for k in 0..3 {
                let b = self.expert_bytes[k];
                let start = out.len();
                for &(i, at) in picked {
                    let (dst, src) = (staged[k] + miss[i].1 * b, self.addr(k, n_slots + at));
                    match out[start..].last_mut() {
                        Some(c) if c.0 + c.2 == dst && c.1 + c.2 == src && c.2 < STAGE_PIECE => {
                            c.2 += b;
                        }
                        _ => out.push((dst, src, b)),
                    }
                }
            }
        };
        let ram: Vec<(usize, usize)> = srcs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                PassSrc::Ram(at) => Some((i, *at)),
                PassSrc::Read { .. } => None,
            })
            .collect();
        copy(&mut out, &ram);
        let mut read: Vec<(usize, usize, usize)> = srcs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                PassSrc::Read { batch, at } => Some((*batch, i, *at)),
                PassSrc::Ram(_) => None,
            })
            .collect();
        read.sort_unstable();
        for batch in read.chunk_by(|a, b| a.0 == b.0) {
            let picked: Vec<(usize, usize)> = batch.iter().map(|r| (r.1, r.2)).collect();
            out.push((CALL, wait_arg(batch[0].0), 0));
            copy(&mut out, &picked);
            out.push((CALL, release_arg(batch[0].0), 0));
        }
        out
    }

    /// Plans packed staging for this lend ([`Self::stage_ids`]) from the current table: the
    /// first packed id is the first whole expert past every kind's kept slots. Returns the
    /// staging buffer bytes per kind it needs (one expert more for the matmuls' reads past
    /// the last row).
    #[cfg(feature = "cuda")]
    fn pack(&mut self) -> [usize; 3] {
        let n_slots = self.slots.n_slots();
        let geometry: Vec<(usize, usize)> = (0..3)
            .map(|k| super::cuda::tiered_split(self.tensor(k)).unwrap_or((0, 0)))
            .collect();
        self.stage_first = (0..3)
            .map(|k| {
                geometry[k]
                    .1
                    .saturating_sub(geometry[k].0)
                    .div_ceil(self.expert_bytes[k])
            })
            .max()
            .unwrap_or(0);
        self.stage_miss.clear();
        self.stage_ids = self.slots.table();
        // Expert parallelism: only this entry's own experts are staged (the others' choices
        // never read their ids), unless it computes them all ([`stages_all`]).
        for (e, id) in self.stage_ids.iter_mut().enumerate() {
            if *id as usize >= n_slots && (self.stage_all || self.slots.owns(e as u32)) {
                *id = (self.stage_first + self.stage_miss.len()) as i32;
                self.stage_miss.push(e as u32);
            }
        }
        let n_ids = self.stage_first + self.stage_miss.len() + 1;
        std::array::from_fn(|k| {
            (geometry[k].0 + n_ids * self.expert_bytes[k]).saturating_sub(geometry[k].1)
        })
    }

    /// The ids of a pass logged with packed staging as the tiered tensors' ([`Self::stage_ids`]).
    fn unpack(&self, ids: &[i32]) -> Vec<i32> {
        let n_slots = self.slots.n_slots() as i32;
        let first = self.stage_first as i32;
        ids.iter()
            .map(|&id| match id.checked_sub(first) {
                Some(i) if i >= 0 => self
                    .stage_miss
                    .get(i as usize)
                    .map_or(-1, |&e| n_slots + e as i32),
                _ => id,
            })
            .collect()
    }

    /// Counts a logged pass's reads (#391): `ids`, its looked-up ids, as distinct experts (each
    /// read once by the pass's `mul_mat_id`), or every pair (`gathered`: [`Fetch::Copy`]'s
    /// gather copies each choice), from the slots (VRAM) or host memory, or all from VRAM in a
    /// `staged` pass (its uncached experts were copied in first).
    fn read(&mut self, ids: &[i32], gathered: bool, staged: bool) {
        let n_slots = self.slots.n_slots();
        let per = self.expert_bytes.iter().sum::<usize>() as u64;
        self.pass = self.pass.wrapping_add(1);
        if self.pass == 0 {
            self.seen.fill(0);
            self.pass = 1;
        }
        let (mut device, mut host) = (0u64, 0u64);
        for &id in ids {
            // Expert parallelism: another GPU computes the experts this layer does not own.
            if id >= n_slots as i32 && !self.slots.owns((id - n_slots as i32) as u32) {
                continue;
            }
            let Some(seen) = self.seen.get_mut(id as usize) else {
                continue;
            };
            if !gathered {
                if *seen == self.pass {
                    continue;
                }
                *seen = self.pass;
            }
            if (id as usize) < n_slots {
                device += per;
            } else {
                host += per;
            }
        }
        if staged {
            device += std::mem::take(&mut host);
        }
        self.traffic.device += device;
        self.traffic.host += host;
        self.traffic.host_passes += u64::from(host > 0);
    }

    /// The distinct experts of a logged pass's looked-up `ids` this layer owns, ascending, each
    /// with whether VRAM served it (#729, [`super::ep_balance`]).
    fn owned_chosen(&self, ids: &[i32]) -> Vec<(u32, bool)> {
        let mut v: Vec<(u32, bool)> = ids
            .iter()
            .filter_map(|&id| self.slots.decode(id))
            .filter(|&(e, _)| self.slots.owns(e))
            .collect();
        v.sort_unstable_by_key(|c| c.0);
        v.dedup_by_key(|c| c.0);
        v
    }

    /// Counts an unlogged pass's reads (#391): at most `pairs` different experts, from VRAM in
    /// the share the ready slots hold (all of them in a `staged` pass), the rest from host.
    fn read_at_most(&mut self, pairs: usize, staged: bool) {
        let n_expert = self.slots.n_expert().max(1);
        let read = pairs.min(n_expert) as u64;
        let per = self.expert_bytes.iter().sum::<usize>() as u64;
        let device = if staged {
            read
        } else {
            read * self.slots.ready_count() as u64 / n_expert as u64
        };
        self.traffic.device += device * per;
        self.traffic.host += (read - device) * per;
    }

    /// Address of slot `s` (`s >= n_slots`: expert `s - n_slots`) of kind `k`.
    fn addr(&self, k: usize, s: usize) -> usize {
        // SAFETY: reading the data pointer of a live tensor.
        let base = unsafe { (*self.tensor(k).raw()).data as usize };
        base + s * self.expert_bytes[k]
    }
}

/// Where the cache stands.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ResidencyStats {
    pub layers: usize,
    pub slots: usize,
    pub slot_bytes: u64,
    pub hits: u64,
    pub misses: u64,
    pub swaps: u64,
    pub logged_steps: u64,
}

impl ResidencyStats {
    /// Share of routed (token, expert) pairs served from VRAM.
    pub fn hit_rate(&self) -> f64 {
        self.hits as f64 / (self.hits + self.misses).max(1) as f64
    }
}

/// The tiered experts of a model.
pub struct ExpertResidency {
    /// Per layer id: its tiered state, `None` for layers whose experts are not tiered.
    layers: Vec<Option<Layer>>,
    devices: Vec<Device>,
    n_expert: usize,
    n_used: usize,
    rule: SwapRule,
    /// Per-device swap budgets over `rule`'s (`--expert-swaps-device`; empty: `rule`'s).
    device_swaps: Vec<usize>,
    decay: f32,
    /// Most heat an expert gains per logged pass (0: no limit; `--expert-heat-pass-cap`).
    heat_pass_cap: f32,
    profile: Option<PathBuf>,
    /// Where the per-layer hits and misses go on exit (`--expert-split-misses`).
    split_misses: Option<PathBuf>,
    swaps: u64,
    logged: u64,
    /// Hits and misses at the last periodic report.
    reported: (u64, u64),
    fetch: Fetch,
    /// While the slots are lent to a long prompt ([`Self::lend`]): per layer, the experts its
    /// slots held, refilled by [`Self::restore`].
    lent: Option<Vec<Vec<u32>>>,
    /// After [`Self::restore`]: the thread giving the slots their VRAM and experts back, with
    /// per layer whether it did.
    #[cfg(feature = "cuda")]
    restoring: Option<SlotThread>,
    /// Set by [`Self::lend`] to stop the restoring thread at its next layer (#447).
    #[cfg(feature = "cuda")]
    stop_refill: Arc<AtomicBool>,
    /// The finished thread's per-layer results not applied yet: each layer's are applied by
    /// the next after-step that covers it ([`Self::after_step_only`] covers only its own).
    #[cfg(feature = "cuda")]
    refilled: Vec<(usize, Result<(), String>)>,
    /// Layers at or past it are never staged ([`Self::stage_below`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage_limit: usize,
    /// Staging buffers per expert kind and device ([`ResidencyConfig::ring`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    ring: usize,
    /// [`ResidencyConfig::stage_pieces`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage_pieces: usize,
    /// [`ResidencyConfig::stage_reuse`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage_reuse: bool,
    /// [`ResidencyConfig::lend_repack`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    lend_repack: usize,
    /// CUDA devices idle during the passes [`Self::begin_pass`] starts ([`Self::set_idle`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    idle: Vec<usize>,
    /// [`ResidencyConfig::stage_min_tokens`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage_min: i64,
    /// [`ResidencyConfig::lend_keep`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    lend_keep: Vec<u64>,
    /// [`ResidencyConfig::settles`]: after-steps wait for the copies and the refill, and a lend
    /// lets the refill finish (`--deterministic` with the pair balance, #1165).
    settle: bool,
    /// The host's share threshold ([`super::cpu_share::select`]; 0: off).
    share: u32,
    /// Expert parallelism ([`ResidencyConfig::ep`]): the model layer of each entry.
    ep_real: Option<Vec<usize>>,
    /// Expert parallelism: the logged steps' routed work per GPU since the last report (#729).
    balance: super::ep_balance::BalanceStats,
    /// [`ResidencyConfig::ep_predict`]: the prefetch predictors' counts since the last report.
    predict: super::ep_prefetch::PrefetchStats,
    /// [`ResidencyConfig::ep_cached`]: per device, its `expert_cached` rows as last written.
    cached_rows: Vec<Vec<i32>>,
    /// [`ResidencyConfig::ep_stage_all`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    ep_stage_all: bool,
    /// The CUDA devices whose slots did not all fit at load (#1208), [`ExpertResidency::shrunk`].
    shrunk: Vec<Shrunk>,
    /// The bounded host tier (#37) the layers' host parts are, if any: RAM slots and a staging
    /// buffer in place of every expert's host bytes. Last: dropped after the tiered ranges
    /// that map its memory.
    host: Option<std::sync::Arc<super::host_tier::HostTier>>,
}

/// A CUDA device whose expert slots did not all find VRAM at load (#1208): the cache planned
/// again in less and took `slots` of the `planned` ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shrunk {
    pub cuda: usize,
    pub planned: usize,
    pub slots: usize,
}

/// The thread of [`ExpertResidency::restore`]: per layer, whether its slots came back.
#[cfg(feature = "cuda")]
type SlotThread = std::thread::JoinHandle<Vec<(usize, Result<(), String>)>>;

/// One layer's view for graph building.
pub struct TieredLayer<'a> {
    layer: &'a Layer,
    dev: &'a Device,
    n_expert: usize,
    fetch: Fetch,
    share: u32,
    /// [`ResidencyConfig::stage_pieces`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pieces: usize,
    /// [`ResidencyConfig::stage_min_tokens`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage_min: i64,
    /// The bounded host tier and this layer in it ([`ExpertResidency::host_tier`]).
    host: Option<(&'a super::host_tier::HostTier, usize)>,
}

/// Steps of at most this many tokens gather their experts first with [`Fetch::Copy`].
const COPY_MAX_TOKENS: i64 = 8;

impl<'a> TieredLayer<'a> {
    /// The tiered tensor of expert kind `k` ([`KINDS`] order).
    pub fn weight(&self, k: usize) -> Tensor<'a> {
        self.layer.tensor(k)
    }

    /// The host's share of this layer's host-resident experts in a step of `n` tokens
    /// ([`super::cpu_share`], `--cpu-expert-share`): the threshold of
    /// [`super::cpu_share::select`], `None` when the step does not take part (off, the CPU
    /// backend, a step too big for the exchange buffer, or [`Fetch::Copy`] gathering it).
    pub fn cpu_share(&self, n: i64) -> Option<u32> {
        (self.share > 0
            && self.dev.place.cuda.is_some()
            && n as usize <= super::cpu_share::ops::MAX_TOKENS
            && !self.gathers(n))
        .then_some(self.share)
    }

    /// The experts of the layer (host-resident: [`Self::remap`]'s ids past [`Self::n_slots`]).
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// The layer's cache slots: [`Self::remap`]'s indices below this are VRAM.
    pub fn n_slots(&self) -> usize {
        self.layer.slots.n_slots()
    }

    /// The layer's lookup table as uploaded (I32 `[n_expert]`): expert `e`'s VRAM slot, or
    /// `n_slots + e` where host memory holds it (`--glm-route-bias` reads which are cached).
    pub fn table_view(&self, ctx: &'a Context) -> Tensor<'a> {
        let t = self.dev.table();
        ctx.view_1d(
            t,
            self.n_expert as i64,
            self.layer.row * self.n_expert * t.nb()[0],
        )
    }

    /// Which of the layer's experts their owners cache, I32 `[n_expert]` (1 cached, 0 not),
    /// the same on every device ([`ResidencyConfig::ep_cached`]); `None` without it.
    /// The device's `--glm-prefetch` staging ring and its layout
    /// ([`ResidencyConfig::ep_prefetch`]), `None` without one.
    pub fn prefetch_ring(&self) -> Option<(Tensor<'a>, super::ep_prefetch::RingLayout)> {
        self.dev.prefetch_ring()
    }

    pub fn cached_view(&self, ctx: &'a Context) -> Option<Tensor<'a>> {
        let t = self.dev.cached()?;
        Some(ctx.view_1d(
            t,
            self.n_expert as i64,
            self.layer.row * self.n_expert * t.nb()[0],
        ))
    }

    /// Whether [`Self::mul_mat_id`] gathers a step of `n` tokens' experts into VRAM first
    /// ([`Fetch::Copy`]).
    pub fn gathers(&self, n: i64) -> bool {
        self.fetch == Fetch::Copy && n <= COPY_MAX_TOKENS
    }

    /// Maps the router's choices `selected` (I32 `[n_used, n]`) to indices into the tiered
    /// tensors, `[n_used, n]`. Steps of at most [`LOG_TOKENS`] tokens write the result straight
    /// into the route log (no extra kernel), for [`ExpertResidency::after_step`]; longer ones
    /// count a sample into the tally ([`ResidencyConfig::learn_long`]).
    pub fn remap(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        selected: Tensor<'a>,
        n: i64,
    ) -> Tensor<'a> {
        self.remap_rows(ctx, graph, selected, n, None)
    }

    /// [`Self::remap`] with a verify step's MoE row map (`rows`, I32 `[n]`): row `t` takes row
    /// `rows[t]`'s choices ([`crate::model::core::pad_rows::routed_ids`]), so a padding row
    /// looks up, logs and reads its sequence's first row's experts. With
    /// `--moe-pad-remap-probe` (a diagnostic, [`crate::model::core::pad_rows::Probe`]) only the
    /// kernels' ids or only the log's take the map.
    pub fn remap_rows(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        selected: Tensor<'a>,
        n: i64,
        rows: Option<Tensor<'a>>,
    ) -> Tensor<'a> {
        use crate::model::core::pad_rows::{self, Probe};
        let used = selected.ne()[0];
        let chosen = match rows {
            Some(r) => pad_rows::routed_ids(ctx, selected, r),
            None => ctx.cont(selected),
        };
        // `--moe-pad-remap-probe`: one of the lookup's uses takes the router's own ids.
        let probe = rows.map_or(Probe::Both, |_| pad_rows::probe());
        let own = (probe != Probe::Both).then(|| ctx.reshape_1d(ctx.cont(selected), used * n));
        let flat = ctx.reshape_1d(chosen, used * n);
        let (flat, kernel_flat) = match (probe, own) {
            (Probe::Kernels, Some(o)) => (o, Some(flat)),
            (Probe::Log, Some(o)) => (flat, Some(o)),
            _ => (flat, None),
        };
        // A staged pass with packed staging reads the staged tensors' ids.
        let t = if self.packs(n) {
            self.dev.stage_table()
        } else {
            self.dev.table()
        };
        let table = ctx.view_2d(
            t,
            [1, self.n_expert as i64],
            t.nb()[0],
            self.layer.row * self.n_expert * t.nb()[0],
        );
        let ids = ctx.get_rows(table, flat);
        if n as usize <= LOG_TOKENS {
            let off = self.layer.row * used as usize * LOG_TOKENS * 4;
            // SAFETY: a fresh op result placed in the persistent log, inside its bounds (rows ×
            // n_used × LOG_TOKENS entries); nothing else writes that range during the step.
            unsafe { ids.place_in(self.dev.log(), off) };
        } else if let Some((tally, eye)) = self.dev.tally() {
            // A long step: its first LOG_TOKENS tokens' choices, one-hot, summed per expert
            // into the layer's tally row (read and cleared by `ExpertResidency::learn`).
            let e = self.n_expert as i64;
            let sample = ctx.view_1d(flat, used * LOG_TOKENS as i64, 0);
            let hot = ctx.get_rows(eye, sample);
            let per = ctx.sum_rows(ctx.cont(ctx.transpose(hot)));
            let row = ctx.view_1d(tally, e, self.layer.row * self.n_expert * 4);
            let sum = ctx.add(row, ctx.reshape_1d(per, e));
            graph.borrow_mut().expand(ctx.cpy(sum, row));
        }
        let ids = kernel_flat.map_or(ids, |k| ctx.get_rows(table, k));
        let ids = ctx.reshape_2d(ids, used, n);
        // The bounded host tier: a miss (`n_slots + expert`, as logged) reads the RAM slot or
        // staging place the tier's fetch puts the expert in. Not in a staged pass, whose
        // copies bring every miss into VRAM ([`ExpertResidency::begin_pass`]).
        match self.host.filter(|_| !self.stages(n)) {
            Some((h, il)) => h
                .fetch(ctx, il, ids, self.n_slots())
                .unwrap_or_else(|e| panic!("host expert tier: {e}")),
            None => ids,
        }
    }

    /// Whether the layer's host part is the bounded host tier's (RAM slots and staging, not
    /// every expert's bytes).
    pub fn bounded(&self) -> bool {
        self.host.is_some()
    }

    /// Logs `guess` (I32 `[n_used, n]`, raw expert ids: the router of this entry's layer
    /// applied to the previous layer's input) beside the route log, for the prefetch
    /// predictors' report ([`ResidencyConfig::ep_predict`]). Nothing without that log or for
    /// a step of more than [`LOG_TOKENS`] tokens.
    pub fn log_predicted(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        guess: Tensor<'a>,
        n: i64,
    ) {
        let Some(log) = self.dev.pred_log() else {
            return;
        };
        if n as usize > LOG_TOKENS {
            return;
        }
        let used = guess.ne()[0];
        let ids = ctx.cont(guess);
        let off = self.layer.row * used as usize * LOG_TOKENS * 4;
        // SAFETY: a fresh op result placed in the persistent log, inside its bounds (rows ×
        // n_used × LOG_TOKENS entries); nothing else writes that range during the step.
        unsafe { ids.place_in(log, off) };
        graph.borrow_mut().expand(ids);
    }

    /// Whether a step of `n` tokens stages this layer ([`Self::staged`]).
    pub fn stages(&self, n: i64) -> bool {
        #[cfg(feature = "cuda")]
        return self.dev.stage().is_some() && n >= self.stage_min && !self.layer.staged.is_empty();
        #[cfg(not(feature = "cuda"))]
        {
            let _ = n;
            false
        }
    }

    /// Whether a step of `n` tokens stages this layer with packed staging ids.
    fn packs(&self, n: i64) -> bool {
        self.stages(n) && !self.layer.stage_ids.is_empty()
    }

    /// The routed experts' outputs `down(silu(gate x) * up x)` (`[n_embd, n_used, n]`) from
    /// staged copies, for a step of `n` tokens big enough to stage (see [`super::staging`]).
    /// `None` when the step is small or the layer does not stage.
    pub fn staged_experts(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        x: Tensor<'a>,
        ids: Tensor<'a>,
        n: i64,
    ) -> Option<Tensor<'a>> {
        self.staged(ctx, graph, x, n, |x, w| {
            let up = ctx.mul_mat_id(w[0], x, ids);
            let gate = ctx.mul_mat_id(w[1], x, ids);
            let act = ctx.mul(ctx.silu(gate), up);
            ctx.mul_mat_id(w[2], act, ids)
        })
    }

    /// `f(x, weights)` over this layer's staged tensors (up, gate, down: the VRAM slots
    /// followed by the staged copies of the uncached experts, indexed like the tiered tensors),
    /// for a step of `n` tokens big enough to stage (see [`super::staging`]): waits for this
    /// layer's copy before `f`'s products and frees its buffer after them. `None` when the step
    /// is small or the layer does not stage.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn staged(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        x: Tensor<'a>,
        n: i64,
        f: impl FnOnce(Tensor<'a>, [Tensor<'a>; 3]) -> Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        #[cfg(feature = "cuda")]
        {
            use super::staging::{DONE, WAIT};
            let st = self.dev.stage()?;
            if !self.stages(n) {
                return None;
            }
            let w = |k: usize| self.layer.staged[k].tensor(KINDS[k]).expect("staged");
            let (wait_name, done_name) = st.op_names(self.layer.row);
            let wait = ctx
                .external(&WAIT, types::F32, [1, 1, 1, 1], &[x])
                .named(&wait_name);
            let x = ctx.add(x, wait);
            let out = f(x, [w(0), w(1), w(2)]);
            let done = ctx
                .external(&DONE, types::F32, [1, 1, 1, 1], &[out])
                .named(&done_name);
            graph.borrow_mut().expand(done);
            Some(out)
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    /// [`Self::staged`] for the fused expert op with split waits
    /// ([`ResidencyConfig::stage_pieces`]): no wait before `f`, which gets the `(staging id,
    /// row)` its op passes to [`super::staging::split_wait`] (the op then waits for each piece
    /// of the layer's copy itself), and the buffer freed after it. `None` as [`Self::staged`],
    /// or with split waits off.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn staged_split(
        &self,
        ctx: &'a Context,
        graph: &std::cell::RefCell<crate::ggml::Graph<'a>>,
        x: Tensor<'a>,
        n: i64,
        f: impl FnOnce(Tensor<'a>, [Tensor<'a>; 3], (usize, usize)) -> Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        #[cfg(feature = "cuda")]
        {
            let st = self.dev.stage()?;
            if self.pieces == 0 || !self.stages(n) {
                return None;
            }
            let w = |k: usize| self.layer.staged[k].tensor(KINDS[k]).expect("staged");
            let (_, done_name) = st.op_names(self.layer.row);
            let out = f(x, [w(0), w(1), w(2)], (st.id(), self.layer.row));
            let done = ctx
                .external(&super::staging::DONE, types::F32, [1, 1, 1, 1], &[out])
                .named(&done_name);
            graph.borrow_mut().expand(done);
            Some(out)
        }
        #[cfg(not(feature = "cuda"))]
        None
    }

    /// `w_k @ x` for the experts `ids` (from [`Self::remap`]): `mul_mat_id` over the tiered
    /// tensor, or with [`Fetch::Copy`] in a small step, over a copy of just the selected
    /// experts gathered into VRAM first (a `get_rows` of their raw bytes).
    pub fn mul_mat_id(
        &self,
        ctx: &'a Context,
        k: usize,
        x: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Tensor<'a> {
        let w = self.weight(k);
        let [used, n, ..] = ids.ne();
        if !self.gathers(n) {
            return ctx.mul_mat_id(w, x, ids);
        }
        let ne = w.ne();
        let row = w.nb()[2];
        let rows = ctx.reinterpret(w, types::I32, [row as i64 / 4, ne[2], 1, 1]);
        let picked = ctx.get_rows(rows, ctx.reshape_1d(ids, used * n));
        // Into the device's fetch buffer (weight memory, not the compute buffer: the matmuls
        // then leave its padding alone and accept a view of it).
        let fetch = self.dev.ctx.tensor("expert_fetch").expect("fetch buffer");
        // SAFETY: a fresh op result placed inside the fetch buffer, in kind k's region, which
        // holds COPY_MAX_TOKENS tokens' experts; layers run one after another on the device.
        unsafe { picked.place_in(fetch, self.dev.fetch_at[k]) };
        let gathered = as_type(ctx, picked, w.ty(), [ne[0], ne[1], used * n]);
        // Entry j of token t is gathered expert t * used + j.
        let iota = ctx.cast(ctx.arange(0.0, (used * n) as f32, 1.0), types::I32);
        ctx.mul_mat_id(gathered, x, ctx.reshape_2d(iota, used, n))
    }
}

/// The contiguous bytes of `t` seen as a tensor of `ty` and shape `ne` (a view).
fn as_type<'a>(
    ctx: &'a Context,
    t: Tensor<'a>,
    ty: crate::ggml::ffi::ggml_type,
    ne: [i64; 3],
) -> Tensor<'a> {
    use crate::ggml::ffi;
    assert!(t.is_contiguous());
    let row = crate::ggml::row_size(ty, ne[0] as usize);
    assert_eq!(
        row * (ne[1] * ne[2]) as usize,
        t.nbytes(),
        "as_type changes the size"
    );
    // SAFETY: a fresh view node of this context covering exactly `t`'s bytes, retyped.
    unsafe {
        let v = ffi::ggml_view_1d(ctx.raw(), t.raw(), t.elements() as i64, 0);
        (*v).type_ = ty;
        (*v).ne = [ne[0], ne[1], ne[2], 1];
        (*v).nb = [
            ffi::ggml_type_size(ty),
            row,
            row * ne[1] as usize,
            row * (ne[1] * ne[2]) as usize,
        ];
        ctx.wrap(v)
    }
}

impl ExpertResidency {
    /// Builds the tiered tensors for every layer `il` with `place(il)` and all three expert
    /// tensors (`find`), and fills the slots: per device (`None` = CPU) up to `budget(device)`
    /// bytes of slots, split over its layers by the profile in `cfg` (if it exists) and then
    /// round robin. `n_layers` counts every layer id `find` may know (MTP layers included).
    pub fn new<'w>(
        n_layers: usize,
        n_expert: usize,
        n_used: usize,
        find: impl Fn(&str) -> Option<Tensor<'w>>,
        place: impl Fn(usize) -> Option<LayerPlace>,
        budget: impl Fn(Option<usize>) -> u64,
        cfg: &ResidencyConfig,
    ) -> Result<Self, String> {
        // Which layers, on which device, with how many bytes per expert.
        let mut found: Vec<Option<(LayerPlace, [Tensor<'w>; 3])>> = Vec::new();
        for il in 0..n_layers {
            let ts: Option<Vec<Tensor<'w>>> = KINDS
                .iter()
                .map(|k| find(&format!("blk.{il}.{k}.weight")))
                .collect();
            // With the bounded host tier (#37) a layer's host tensor is its RAM slots.
            let whole = |t: &Tensor<'w>| {
                t.ne()[2] == n_expert as i64 || super::host_tier::HostTier::of(*t).is_some()
            };
            let entry = match (place(il), ts) {
                (Some(p), Some(ts)) if ts.iter().all(whole) => Some((p, [ts[0], ts[1], ts[2]])),
                _ => None,
            };
            found.push(entry);
        }
        // The bounded host tier: experts are in RAM slots or read as routed, so nothing may
        // read an expert's host bytes at `n_slots + expert`: no host share, no re-pack of the
        // kept slots, and a long prompt's staged passes take every expert from where the tier
        // has it ([`Layer::tier_copies`]: one plan a row, kept across lends).
        let host = found
            .iter()
            .flatten()
            .find_map(|(_, ts)| super::host_tier::HostTier::of(ts[0]))
            .map(|(t, _)| t);
        let bounded;
        let cfg = match &host {
            Some(_) => {
                bounded = ResidencyConfig {
                    lend_repack: 0,
                    lend_keep: Vec::new(),
                    stage_pieces: 0,
                    stage_reuse: true,
                    cpu_share: super::cpu_share::CpuShareConfig {
                        share: 0.0,
                        ..cfg.cpu_share.clone()
                    },
                    ..cfg.clone()
                };
                &bounded
            }
            None => cfg,
        };
        let mut devices: Vec<Device> = Vec::new();
        let mut dev_of = vec![usize::MAX; n_layers];
        for (il, f) in found.iter().enumerate() {
            let Some((p, _)) = f else { continue };
            let d = match devices.iter().position(|d| d.place == *p) {
                Some(d) => d,
                None => {
                    devices.push(Device {
                        place: *p,
                        ctx: Context::new(7, 0),
                        rows: 0,
                        fetch_at: [0; 3],
                        #[cfg(feature = "cuda")]
                        copier: None,
                        #[cfg(feature = "cuda")]
                        inflight: VecDeque::new(),
                        #[cfg(feature = "cuda")]
                        staging: None,
                        #[cfg(feature = "cuda")]
                        kept: false,
                        #[cfg(feature = "cuda")]
                        parked: false,
                        #[cfg(feature = "cuda")]
                        lend_bytes: None,
                        dirty: true,
                        ring: None,
                    });
                    devices.len() - 1
                }
            };
            dev_of[il] = d;
            devices[d].rows += 1;
        }

        let profile = match &cfg.profile {
            Some(p) if p.exists() => {
                let prof = Profile::read(p).map_err(|e| e.to_string())?;
                if prof.n_expert != n_expert {
                    return Err(format!(
                        "{}: a profile for {} experts, the model has {n_expert}",
                        p.display(),
                        prof.n_expert
                    ));
                }
                Some(prof)
            }
            _ => None,
        };
        let mut ranked: Vec<(usize, u32)> =
            profile.as_ref().map_or_else(Vec::new, |p| p.ranked.clone());
        // Expert parallelism: a layer's ranking goes to every entry serving it, each taking its
        // own experts.
        let ep = cfg.ep.as_ref();
        let real = |il: usize| ep.and_then(|e| e.real.get(il).copied()).unwrap_or(il);
        let owned = |il: usize| ep.and_then(|e| e.owned.get(il).cloned().flatten());
        if ep.is_some() {
            let mut by_real: Vec<Vec<usize>> = vec![Vec::new(); n_layers];
            for il in 0..n_layers {
                if let Some(v) = by_real.get_mut(real(il)) {
                    v.push(il);
                }
            }
            ranked = ranked
                .iter()
                .flat_map(|&(l, e)| {
                    let entries = by_real.get(l).cloned().unwrap_or_default();
                    entries
                        .into_iter()
                        .filter(move |&il| owned(il).is_none_or(|r| r.contains(&e)))
                        .map(move |il| (il, e))
                })
                .collect();
        }
        let owned_all: Vec<Option<std::ops::Range<u32>>> = (0..n_layers).map(owned).collect();
        // `--expert-split-misses`: a previous run's per-layer misses size the layers.
        let measured = match &cfg.split_misses {
            Some(p) if p.exists() => Some(
                super::miss_split::MissStats::read(p)
                    .map_err(|e| format!("{}: {e}", p.display()))?,
            ),
            _ => None,
        };

        // Per device: the bytes its slots may take.
        let mut dev_budget: Vec<u64> = devices
            .iter()
            .enumerate()
            .map(|(d, dev)| {
                // Each tiered range rounds its VRAM part up to the mapping granule (2 MiB).
                let rounding = match dev.place.cuda {
                    Some(_) => 3 * (2u64 << 20) * dev.rows as u64,
                    None => 0,
                };
                // Staging takes no VRAM of its own: long prompts borrow the slots' (see
                // `Self::lend`).
                // `--glm-prefetch`: its staging ring comes out of the cache's share, not the
                // reserve's.
                let ring = prefetch_ring(cfg, &found, &dev_of, d, dev.place.cuda, n_expert)
                    .map_or(0, |r| r.bytes() as u64);
                budget(dev.place.cuda)
                    .saturating_sub(rounding)
                    .saturating_sub(ring)
            })
            .collect();
        // Which experts each of device d's layers caches first in `b` bytes of slots (empty
        // lists for the other devices' layers).
        let plan_first = |d: usize, b: u64| -> Vec<Vec<u32>> {
            let dev = &devices[d];
            let slot_bytes: Vec<u64> = (0..n_layers)
                .map(|il| match &found[il] {
                    Some((_, ts)) if dev_of[il] == d => ts.iter().map(|t| t.nb()[2] as u64).sum(),
                    _ => 0,
                })
                .collect();
            let mut alloc = allocate_slots_in(&ranked, &slot_bytes, &owned_all, n_expert, b);
            if cfg.profile_split_only && profile.is_some() {
                alloc = lowest_experts(&alloc, &owned_all, n_expert);
            }
            if let Some(m) = &measured {
                alloc =
                    super::miss_split::split_measured(alloc, &slot_bytes, m, &owned_all, n_expert);
                tracing::info!(
                    device = ?dev.place.cuda,
                    slots = ?(0..n_layers)
                        .filter(|&il| slot_bytes[il] > 0)
                        .map(|il| alloc[il].len())
                        .collect::<Vec<_>>(),
                    "expert cache: layers sized by the measured misses"
                );
            }
            alloc
        };
        let mut first: Vec<Vec<u32>> = vec![Vec::new(); n_layers];
        for (d, &b) in dev_budget.iter().enumerate() {
            for (il, experts) in plan_first(d, b).into_iter().enumerate() {
                if !experts.is_empty() {
                    first[il] = experts;
                }
            }
        }

        // `lend_spread`: each staging layer keeps the same share of its slots, the device's
        // `lend_keep` bytes over all of its slots' bytes. Without a `lend_keep`, the most that
        // leaves the device's `lend_need` free ([`auto_keep_carved`]).
        let auto = cfg.lend_keep.is_empty() && !cfg.lend_need.is_empty();
        let spread = (cfg.lend_spread || auto) && cfg.staging && cfg!(feature = "cuda");
        // Device d's (layer, slots kept) for the slots of `first`.
        let plan_keep = |d: usize, first: &[Vec<u32>]| -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            let dev = &devices[d];
            {
                let Some(c) = dev.place.cuda else {
                    return out;
                };
                let mine: Vec<usize> = (0..n_layers)
                    .filter(|&il| dev_of[il] == d && found[il].is_some())
                    .collect();
                let slot_bytes = |il: usize| -> u64 {
                    let ts = &found[il].as_ref().expect("found").1;
                    ts.iter().map(|t| t.nb()[2] as u64).sum()
                };
                if auto {
                    let Some(&need) = cfg.lend_need.get(c) else {
                        return out;
                    };
                    let below = cfg.stage_below.unwrap_or(usize::MAX);
                    let reals = ep.map(|e| e.real.as_slice());
                    let stages = |il: usize| stages_entry(reals, il, below, cfg.ep_stage_all);
                    let staged: Vec<usize> =
                        mine.iter().copied().filter(|&il| stages(il)).collect();
                    // In the device's layer order, as a lend unmaps their slots.
                    let layers: Vec<LendLayer> = mine
                        .iter()
                        .map(|&il| {
                            let ts = &found[il].as_ref().expect("found").1;
                            LendLayer {
                                slots: first[il].len(),
                                bytes: std::array::from_fn(|k| ts[k].nb()[2]),
                                over: std::array::from_fn(|k| {
                                    crate::ggml::mapped_tail_padding(ts[k].ty(), ts[k].ne()[0])
                                }),
                                staged: stages(il),
                            }
                        })
                        .collect();
                    // Expert parallelism: an entry stages only the experts it owns, or all of
                    // them when it computes them all (`--pair-tp`'s prompts, #681: sized for
                    // its owned share only, the buffers took the prompt's compute room).
                    let n_staged = staged
                        .iter()
                        .map(|&il| {
                            let all = stages_all(reals, il, cfg.ep_stage_all);
                            stage_width(owned(il).as_ref(), all, n_expert)
                        })
                        .max()
                        .unwrap_or(n_expert);
                    let (keep, by_bytes) =
                        auto_keep_carved(&layers, n_staged, cfg.ring, need, cfg.stage_reuse);
                    let bytes_of = |keep: &[usize]| -> u64 {
                        staged
                            .iter()
                            .zip(keep)
                            .map(|(&il, &k)| k as u64 * slot_bytes(il))
                            .sum()
                    };
                    let kept = bytes_of(&keep);
                    if keep != by_bytes {
                        tracing::info!(
                            device = c,
                            kept_mib = kept >> 20,
                            by_bytes_mib = bytes_of(&by_bytes) >> 20,
                            "expert cache: long prompts keep less, so the staging buffers made of \
                             the lent slots leave the prompt its VRAM"
                        );
                    }
                    tracing::info!(
                        device = c,
                        kept_mib = kept >> 20,
                        need_mib = need >> 20,
                        "expert cache: long prompts keep the hottest cached experts"
                    );
                    if keep.iter().any(|&k| k > 0) {
                        out.extend(staged.iter().copied().zip(keep));
                    }
                    return out;
                }
                let want = cfg
                    .lend_keep
                    .get(c)
                    .or(cfg.lend_keep.last())
                    .copied()
                    .unwrap_or(0);
                let bytes = |il: usize| first[il].len() as u128 * slot_bytes(il) as u128;
                let total: u128 = mine.iter().map(|&il| bytes(il)).sum();
                for &il in &mine {
                    let n = first[il].len() as u128;
                    out.push((il, (n * want as u128 / total.max(1)).min(n) as usize));
                }
            }
            out
        };
        let mut keep_of: Vec<Option<usize>> = vec![None; n_layers];
        if spread {
            for d in 0..devices.len() {
                for (il, k) in plan_keep(d, &first) {
                    keep_of[il] = Some(k);
                }
            }
        }

        let mut layers: Vec<Option<Layer>> = (0..n_layers).map(|_| None).collect();
        let present: Vec<usize> = (0..n_layers)
            .map(|il| found[il].as_ref().map_or(usize::MAX, |_| dev_of[il]))
            .collect();
        let rows = run_rows(&present, ep.map(|e| e.real.as_slice()));
        // #1208: a CUDA device whose slots find no VRAM (free memory drifted since the plan)
        // drops its layers built so far and plans again in less. Per device: the free VRAM
        // its plan was made against (read before its first layer), the failures so far, the
        // slots first planned.
        let mut free_at: Vec<Option<u64>> = vec![None; devices.len()];
        let mut failures = vec![0u32; devices.len()];
        let slots_on = |d: usize, first: &[Vec<u32>]| -> usize {
            (0..n_layers)
                .filter(|&il| dev_of[il] == d)
                .map(|il| first[il].len())
                .sum()
        };
        let planned: Vec<usize> = (0..devices.len()).map(|d| slots_on(d, &first)).collect();
        let mut il = 0;
        while il < n_layers {
            let Some((p, ts)) = found[il].as_ref().filter(|_| layers[il].is_none()) else {
                il += 1;
                continue;
            };
            let d = dev_of[il];
            #[cfg(feature = "cuda")]
            if let (Some(c), None) = (p.cuda, free_at[d]) {
                free_at[d] = super::cuda::free_vram(c).ok().map(|f| f as u64);
            }
            let n_slots = first[il].len();
            let mut tensors = Vec::with_capacity(3);
            let mut no_vram = None;
            for (k, t) in ts.iter().enumerate() {
                let tail = host.as_deref().map(|h| (h.stage_tensor(il, k), h.n_host()));
                match tiered_tensor(p, *t, n_slots, keep_of[il], KINDS[k], n_expert, tail) {
                    Ok(c) => tensors.push(c),
                    Err(e) if e.no_vram && n_slots > 0 && failures[d] < MAX_FIT_TRIES => {
                        no_vram = Some(e.msg);
                        break;
                    }
                    Err(e) => return Err(e.msg),
                }
            }
            if let Some(msg) = no_vram {
                // Every slot of the device given back, then what others took since the plan:
                // the free VRAM the plan saw less what is free now.
                drop(tensors);
                for (l, _) in layers.iter_mut().zip(&dev_of).filter(|(_, o)| **o == d) {
                    *l = None;
                }
                #[cfg(feature = "cuda")]
                let free_now = p
                    .cuda
                    .and_then(|c| super::cuda::free_vram(c).ok())
                    .map(|f| f as u64);
                #[cfg(not(feature = "cuda"))]
                let free_now: Option<u64> = None;
                let drift = match (free_at[d], free_now) {
                    (Some(was), Some(now)) => was.saturating_sub(now),
                    _ => 0,
                };
                let slot: u64 = ts.iter().map(|t| t.nb()[2] as u64).sum();
                let b = shrunk_budget(dev_budget[d], drift, slot, failures[d]);
                tracing::warn!(
                    device = ?p.cuda,
                    layer = il,
                    error = %msg,
                    drift_mib = drift >> 20,
                    budget_mib = dev_budget[d] >> 20,
                    to_mib = b >> 20,
                    "expert cache: slots found no VRAM, planning the device again in less"
                );
                failures[d] += 1;
                dev_budget[d] = b;
                free_at[d] = free_now;
                for j in (0..n_layers).filter(|&j| dev_of[j] == d) {
                    first[j].clear();
                    keep_of[j] = None;
                }
                for (j, experts) in plan_first(d, b).into_iter().enumerate() {
                    if !experts.is_empty() {
                        first[j] = experts;
                    }
                }
                if spread {
                    for (j, k) in plan_keep(d, &first) {
                        keep_of[j] = Some(k);
                    }
                }
                il = (0..n_layers).find(|&j| dev_of[j] == d).unwrap_or(il);
                continue;
            }
            // The slots wholly before every kind's tail, the matmuls' reads past them included.
            #[cfg(feature = "cuda")]
            let keep = keep_of[il].map(|_| {
                (0..3)
                    .map(|k| {
                        let t = tensors[k].tensor(KINDS[k]).expect("tiered tensor");
                        let (at, split) = super::cuda::tiered_split(t).unwrap_or((0, 0));
                        let over = crate::ggml::mapped_tail_padding(t.ty(), t.ne()[0]);
                        super::policy::kept_slots(at, split, ts[k].nb()[2], over, n_slots)
                    })
                    .min()
                    .unwrap_or(0)
            });
            #[cfg(not(feature = "cuda"))]
            let keep = None;
            let mut slots = LayerSlots::new(n_expert, n_slots);
            if let Some(r) = owned(il) {
                slots.set_owned(r);
            }
            slots.fill(&first[il]);
            if let Some(prof) = profile
                .as_ref()
                .filter(|p| !cfg.profile_split_only && real(il) < p.n_layers)
            {
                // Seed the heat so the profile's residents are not swapped out at once.
                slots.add_heat(&prof.heat(cfg.profile_seed)[real(il)]);
            }
            layers[il] = Some(Layer {
                dev: d,
                row: rows[il],
                slots,
                tensors,
                expert_bytes: [ts[0].nb()[2], ts[1].nb()[2], ts[2].nb()[2]],
                counts: vec![0; n_expert],
                traffic: ExpertTraffic::default(),
                seen: vec![0; n_slots + n_expert],
                pass: 0,
                #[cfg(feature = "cuda")]
                staged: Vec::new(),
                #[cfg(feature = "cuda")]
                shared: [None; 3],
                #[cfg(feature = "cuda")]
                ranges: Vec::new(),
                backed: true,
                dirty: true,
                keep,
                tier: host.is_some(),
                tail_lent: false,
                stage_ids: Vec::new(),
                stage_first: 0,
                stage_miss: Vec::new(),
                stage_all: stages_all(
                    cfg.ep.as_ref().map(|e| e.real.as_slice()),
                    il,
                    cfg.ep_stage_all,
                ),
                prev_chosen: Vec::new(),
            });
            il += 1;
        }
        // For the memory plan's line (`Self::shrunk`).
        let shrunk: Vec<Shrunk> = (0..devices.len())
            .filter(|&d| failures[d] > 0)
            .filter_map(|d| {
                Some(Shrunk {
                    cuda: devices[d].place.cuda?,
                    planned: planned[d],
                    slots: slots_on(d, &first),
                })
            })
            .collect();

        for (d, dev) in devices.iter_mut().enumerate() {
            let r = dev.rows as i64;
            if cfg.fetch == Fetch::Copy {
                // Room for a small step's gathered experts of each kind (the largest layer's
                // size), plus a zero tail the quantized matmuls may read past the last row.
                let mut at = 0;
                for k in 0..3 {
                    dev.fetch_at[k] = at;
                    let most = layers
                        .iter()
                        .flatten()
                        .filter(|l| l.dev == d)
                        .map(|l| l.expert_bytes[k])
                        .max()
                        .unwrap_or(0);
                    at += (most * n_used * COPY_MAX_TOKENS as usize).div_ceil(256) * 256;
                }
                dev.ctx
                    .new_tensor(types::I32, &[(at as i64 + (1 << 20)) / 4])
                    .named("expert_fetch");
            }
            dev.ctx
                .new_tensor(types::I32, &[n_expert as i64, r])
                .named("expert_table");
            dev.ctx
                .new_tensor(types::I32, &[n_expert as i64, r])
                .named("expert_stage_table");
            dev.ctx
                .new_tensor(types::I32, &[(n_used * LOG_TOKENS) as i64 * r])
                .named("expert_log");
            if cfg.ep_cached && cfg.ep.is_some() {
                dev.ctx
                    .new_tensor(types::I32, &[n_expert as i64, r])
                    .named("expert_cached");
            }
            if cfg.ep_predict && cfg.ep.is_some() {
                dev.ctx
                    .new_tensor(types::I32, &[(n_used * LOG_TOKENS) as i64 * r])
                    .named("expert_pred_log");
            }
            dev.ring = prefetch_ring(cfg, &found, &dev_of, d, dev.place.cuda, n_expert);
            if let Some(ring) = dev.ring {
                dev.ctx
                    .new_tensor(types::I8, &[ring.bytes() as i64])
                    .named("expert_prefetch_ring");
            }
            if cfg.learn_long {
                dev.ctx
                    .new_tensor(types::F32, &[n_expert as i64, r])
                    .named("expert_tally");
                dev.ctx
                    .new_tensor(types::F32, &[n_expert as i64, n_expert as i64])
                    .named("expert_eye");
            }
            dev.ctx
                .alloc_tensors(dev.place.buft)
                .map_err(|e| e.to_string())?;
            if let Some(t) = dev.ctx.tensor("expert_fetch") {
                t.memset(0, t.nbytes(), 0);
            }
            if let Some(t) = dev.cached() {
                t.memset(0, t.nbytes(), 0);
            }
            if let Some((t, _)) = dev.prefetch_ring() {
                // The route kernel's CTA counter starts at zero.
                t.memset(0, t.nbytes(), 0);
            }
            if let Some(t) = dev.pred_log() {
                // -1 everywhere: rows no graph writes (a layer without a previous MoE layer)
                // hold no guess.
                t.memset(0, t.nbytes(), 0xff);
            }
            if let Some((tally, eye)) = dev.tally() {
                tally.memset(0, tally.nbytes(), 0);
                let mut one = vec![0f32; n_expert * n_expert];
                for e in 0..n_expert {
                    one[e * n_expert + e] = 1.0;
                }
                eye.set_f32(&one);
            }
            #[cfg(feature = "cuda")]
            if let Some(c) = dev.place.cuda {
                dev.copier = Some(Copier::new(c)?);
                dev.lend_bytes = staging_bytes(cfg, Some(c), &found, &dev_of, d);
            }
        }

        let mut me = Self {
            layers,
            devices,
            n_expert,
            n_used,
            rule: cfg.swaps,
            device_swaps: cfg.device_swaps.clone(),
            decay: cfg.decay,
            heat_pass_cap: cfg.heat_pass_cap,
            profile: cfg.profile.clone(),
            split_misses: cfg.split_misses.clone(),
            swaps: 0,
            logged: 0,
            reported: (0, 0),
            fetch: cfg.fetch,
            lent: None,
            #[cfg(feature = "cuda")]
            restoring: None,
            #[cfg(feature = "cuda")]
            stop_refill: Arc::default(),
            #[cfg(feature = "cuda")]
            refilled: Vec::new(),
            stage_limit: cfg.stage_below.unwrap_or(usize::MAX),
            ring: cfg.ring.max(super::PREFILL_RING),
            stage_pieces: cfg.stage_pieces,
            stage_reuse: cfg.stage_reuse,
            lend_repack: cfg.lend_repack,
            stage_min: cfg.stage_min_tokens as i64,
            idle: Vec::new(),
            lend_keep: cfg.lend_keep.clone(),
            settle: cfg.settles(),
            share: 0,
            ep_real: cfg.ep.as_ref().map(|e| e.real.clone()),
            balance: Default::default(),
            predict: Default::default(),
            cached_rows: Vec::new(),
            ep_stage_all: cfg.ep_stage_all,
            shrunk,
            host,
        };
        #[cfg(feature = "cuda")]
        if super::cpu_share::threshold(cfg.cpu_share.share) > 0 {
            me.share = share_setup(cfg, &me, &found)?;
        }
        // The first fill: every ready slot gets its expert's bytes.
        for il in 0..n_layers {
            let Some(l) = &me.layers[il] else { continue };
            let n_slots = l.slots.n_slots();
            let fills: Vec<(u32, u32)> = l
                .slots
                .ready()
                .iter()
                .enumerate()
                .filter_map(|(s, e)| e.map(|e| (s as u32, e)))
                .collect();
            let _ = n_slots;
            me.copy_from_host(il, &fills)?;
        }
        me.sync_copies()?;
        me.upload_tables(None)?;
        Ok(me)
    }

    /// Before a pass of `n` tokens through the layers `ran` selects (one device's): big passes
    /// queue the first staging copies (see [`super::staging`]). The previous pass must be
    /// complete.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn begin_pass(&mut self, n: usize, ran: impl Fn(usize) -> bool) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        if n as i64 >= self.stage_min {
            for d in 0..self.devices.len() {
                let plan = match self.host.clone() {
                    // The bounded host tier: the pass's copies with the tier's sources, its
                    // checkpoint reads started; the pass before's copies are done first, as
                    // they read the staging places these reads fill.
                    Some(h) => {
                        match self.devices[d].stage_mut() {
                            Some(st) => st.sync()?,
                            None => continue,
                        }
                        let (plan, hook) = self.tier_plan(d, &ran, &h)?;
                        if let (Some(st), Some(hook)) = (self.devices[d].stage_mut(), hook) {
                            st.set_hook(hook)?;
                        }
                        plan
                    }
                    None => self.stage_plan(d, &ran),
                };
                let idle = &self.idle;
                if let (Some(st), false) = (self.devices[d].stage_mut(), plan.is_empty()) {
                    // A staging peer idle while this pass runs takes half of its copies
                    // (`--prefill-peer-stage`).
                    let help = st.peer().is_some_and(|p| idle.contains(&p));
                    // Model bytes (#391): the experts copied over the link (not the zeroed
                    // pads, whose source is 0).
                    for (il, _, copies) in &plan {
                        let bytes: usize = copies.iter().filter(|c| c.1 != 0).map(|c| c.2).sum();
                        if let Some(l) = self.layers[*il].as_mut() {
                            l.traffic.staged += bytes as u64;
                        }
                    }
                    st.begin(plan.into_iter().map(|(_, row, c)| (row, c)).collect(), help)?;
                }
            }
        }
        Ok(())
    }

    /// The CUDA devices that run nothing during the passes [`Self::begin_pass`] starts until the
    /// next call: their staging help is free ([`super::staging::Staging::peer`]). Empty: none.
    pub fn set_idle(&mut self, devices: Vec<usize>) {
        self.idle = devices;
    }

    /// While a pass runs: plans the next pass of `n` tokens through the same layers, so the
    /// running pass's last layers queue its first staging copies ([`super::staging::Staging::chain`]).
    /// Its [`Self::begin_pass`] then queues nothing. Only while the slots are lent: the next
    /// pass's copies must not depend on what the running pass caches.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn stage_ahead(&mut self, n: usize, ran: impl Fn(usize) -> bool) -> Result<(), String> {
        // Not with the bounded host tier: a pass's checkpoint reads take the staging places
        // the running pass's copies still read.
        #[cfg(feature = "cuda")]
        if self.is_lent() && n as i64 >= self.stage_min && self.host.is_none() {
            for d in 0..self.devices.len() {
                let plan = self.stage_plan(d, &ran);
                if let (Some(st), false) = (self.devices[d].stage_mut(), plan.is_empty()) {
                    st.chain(plan.into_iter().map(|(_, row, c)| (row, c)).collect())?;
                }
            }
        }
        Ok(())
    }

    /// Device `d`'s staged layers among `ran`, in order: `(layer, row, copies)`.
    #[cfg(feature = "cuda")]
    fn stage_plan(
        &self,
        d: usize,
        ran: &impl Fn(usize) -> bool,
    ) -> Vec<(usize, usize, Vec<super::staging::Copy>)> {
        let mut plan = Vec::new();
        for (il, l) in self.layers.iter().enumerate() {
            let Some(l) = l.as_ref().filter(|l| l.dev == d && ran(il)) else {
                continue;
            };
            if l.staged.is_empty() {
                continue;
            }
            plan.push((il, l.row, l.stage_copies(self.stage_pieces)));
        }
        // In the order the graphs run the rows (an expert-parallel copy's entry comes after
        // the model's layers, its row at its layer's place, [`run_rows`]).
        plan.sort_by_key(|p| p.1);
        plan
    }

    /// [`Self::stage_plan`] with the bounded host tier `h`, and the calls its copies make
    /// ([`super::host_tier::HostTier::stage_pass`]): none when no layer is staged.
    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    fn tier_plan(
        &self,
        d: usize,
        ran: &impl Fn(usize) -> bool,
        h: &super::host_tier::HostTier,
    ) -> Result<
        (
            Vec<(usize, usize, Vec<super::staging::Copy>)>,
            Option<super::host_tier::PassHook>,
        ),
        String,
    > {
        let mut rows: Vec<(usize, &Layer)> = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(il, l)| Some((il, l.as_ref()?)))
            .filter(|(il, l)| l.dev == d && ran(*il) && !l.staged.is_empty())
            .collect();
        if rows.is_empty() {
            return Ok((Vec::new(), None));
        }
        rows.sort_by_key(|r| r.1.row);
        let misses: Vec<(usize, Vec<u32>)> = rows
            .iter()
            .map(|(il, l)| (*il, l.stage_misses().iter().map(|m| m.0).collect()))
            .collect();
        let (srcs, hook) = h.stage_pass(&misses)?;
        let plan = rows
            .iter()
            .zip(&srcs)
            .map(|((il, l), s)| (*il, l.row, l.tier_copies(s)))
            .collect();
        Ok((plan, Some(hook)))
    }

    /// The bytes of routed experts the layers `layers` read since the start, by where (#391's
    /// model bytes): distinct experts per logged pass, at most every choice a different one
    /// for bigger passes.
    pub fn traffic_in(&self, layers: std::ops::Range<usize>) -> ExpertTraffic {
        let mut t = ExpertTraffic::default();
        for l in self
            .layers
            .get(layers.start.min(self.layers.len())..layers.end.min(self.layers.len()))
            .unwrap_or_default()
            .iter()
            .flatten()
        {
            t.add(&l.traffic);
        }
        t
    }

    /// The traffic of the layers cached on CUDA device `cuda` (expert parallelism: what that
    /// GPU read, its misses from its own node's memory included).
    pub fn traffic_on(&self, cuda: usize) -> ExpertTraffic {
        let mut t = ExpertTraffic::default();
        for l in self
            .layers
            .iter()
            .flatten()
            .filter(|l| self.devices[l.dev].place.cuda == Some(cuda))
        {
            t.add(&l.traffic);
        }
        t
    }

    /// Maps the host parts of the tiered tensors on CUDA device `device` for CUDA device `peer`
    /// too ([`super::cuda::share_host_part`]): `peer` may fetch their uncached experts for
    /// `device` (#252). Returns the tensors' addresses.
    #[cfg(feature = "cuda")]
    pub fn share_host_parts(&self, device: usize, peer: usize) -> Result<Vec<usize>, String> {
        let mut at = Vec::new();
        for l in self
            .layers
            .iter()
            .flatten()
            .filter(|l| self.devices[l.dev].place.cuda == Some(device))
        {
            for k in 0..KINDS.len() {
                at.push(super::cuda::share_host_part(l.tensor(k), peer)?);
            }
        }
        Ok(at)
    }

    /// The graph's view of layer `il`, if its experts are tiered.
    pub fn layer(&self, il: usize) -> Option<TieredLayer<'_>> {
        let l = self.layers.get(il)?.as_ref()?;
        Some(TieredLayer {
            layer: l,
            dev: &self.devices[l.dev],
            n_expert: self.n_expert,
            fetch: self.fetch,
            share: self.share,
            pieces: self.stage_pieces,
            stage_min: self.stage_min,
            host: self.host.as_deref().map(|h| (h, il)),
        })
    }

    pub fn stats(&self) -> ResidencyStats {
        let mut s = ResidencyStats {
            swaps: self.swaps,
            logged_steps: self.logged,
            ..Default::default()
        };
        for l in self.layers.iter().flatten() {
            s.layers += 1;
            s.slots += l.slots.n_slots();
            s.slot_bytes += l.slots.n_slots() as u64 * l.expert_bytes.iter().sum::<usize>() as u64;
            s.hits += l.slots.hits;
            s.misses += l.slots.misses;
        }
        s
    }

    /// Per tiered layer: (layer id, slots, bytes its slots take).
    pub fn layer_slots(&self) -> Vec<(usize, usize, u64)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(il, l)| {
                let l = l.as_ref()?;
                let n = l.slots.n_slots();
                Some((
                    il,
                    n,
                    n as u64 * l.expert_bytes.iter().sum::<usize>() as u64,
                ))
            })
            .collect()
    }

    /// Per CUDA device (ggml index) with tiered layers: the bytes its slots take, ascending
    /// by index (for the load's memory plan). Layers on the CPU backend are left out.
    pub fn slot_bytes_by_cuda(&self) -> Vec<(usize, u64)> {
        let mut out: std::collections::BTreeMap<usize, u64> = Default::default();
        for (il, _, bytes) in self.layer_slots() {
            let dev = self.layers[il].as_ref().map(|l| l.dev);
            if let Some(cuda) = dev.and_then(|d| self.devices[d].place.cuda) {
                *out.entry(cuda).or_default() += bytes;
            }
        }
        out.into_iter().collect()
    }

    /// The CUDA devices whose slots the load shrank to fit the VRAM it found (#1208).
    pub fn shrunk(&self) -> &[Shrunk] {
        &self.shrunk
    }

    /// Per tiered layer: (layer id, slots, hits, misses).
    pub fn layer_stats(&self) -> Vec<(usize, usize, u64, u64)> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(il, l)| {
                l.as_ref()
                    .map(|l| (il, l.slots.n_slots(), l.slots.hits, l.slots.misses))
            })
            .collect()
    }

    /// The table entries of layer `il` (for tests).
    pub fn table(&self, il: usize) -> Option<Vec<i32>> {
        self.layers[il].as_ref().map(|l| l.slots.table())
    }

    /// Copies slot or expert `src` of layer `il` into slot `dst`, all kinds.
    fn copy_expert(&mut self, il: usize, dst: u32, src: usize) -> Result<(), String> {
        let l = self.layers[il].as_ref().expect("tiered layer");
        let dev = &mut self.devices[l.dev];
        for k in 0..3 {
            let (to, from, n) = (l.addr(k, dst as usize), l.addr(k, src), l.expert_bytes[k]);
            match dev.place.cuda {
                #[cfg(feature = "cuda")]
                Some(_) => {
                    let c = dev.copier.as_mut().expect("copier");
                    // SAFETY: both ranges are in the layer's tiered range; the slot is out of
                    // the table (or the first fill runs before any step).
                    unsafe { c.copy(to, from, n)? };
                }
                #[cfg(not(feature = "cuda"))]
                Some(_) => unreachable!("CUDA placement without CUDA"),
                None => {
                    // SAFETY: disjoint ranges (a slot and an expert) of the same host buffer.
                    unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, n) };
                }
            }
        }
        Ok(())
    }

    /// Copies the host bytes of experts into slots of layer `il` (`(slot, expert)`): from
    /// `n_slots + expert`, or with the bounded host tier from where it has them (a RAM slot,
    /// or the expert's staging place after a read), waiting for the copies there, as those
    /// bytes only stay until the next step.
    fn copy_from_host(&mut self, il: usize, fills: &[(u32, u32)]) -> Result<(), String> {
        let n_slots = self.layers[il]
            .as_ref()
            .expect("tiered layer")
            .slots
            .n_slots();
        let Some(h) = self.host.clone() else {
            for &(s, e) in fills {
                self.copy_expert(il, s, n_slots + e as usize)?;
            }
            return Ok(());
        };
        if fills.is_empty() {
            return Ok(());
        }
        // As many at a time as the tier has staging places.
        for part in fills.chunks(h.places().max(1)) {
            let experts: Vec<u32> = part.iter().map(|f| f.1).collect();
            let at = h.sources(il, &experts)?;
            for (&(s, _), a) in part.iter().zip(at) {
                self.copy_expert(il, s, n_slots + a)?;
            }
            self.sync_copies()?;
        }
        Ok(())
    }

    /// The bounded host tier the layers' host parts are, if any.
    pub fn host_tier(&self) -> Option<&std::sync::Arc<super::host_tier::HostTier>> {
        self.host.as_ref()
    }

    fn sync_copies(&mut self) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        for d in &mut self.devices {
            if let Some(c) = d.copier.as_mut() {
                c.sync()?;
            }
        }
        Ok(())
    }

    /// Writes the changed tables: whole device tables (`scope` None), or only the changed rows
    /// of the layers `scope` selects, leaving every other row's bytes alone.
    fn upload_tables(&mut self, scope: Option<&dyn Fn(usize) -> bool>) -> Result<(), String> {
        let n_expert = self.n_expert;
        for (d, dev) in self.devices.iter_mut().enumerate() {
            let mine = self
                .layers
                .iter_mut()
                .enumerate()
                .filter_map(|(il, l)| l.as_mut().map(|l| (il, l)))
                .filter(|(_, l)| l.dev == d);
            match scope {
                None => {
                    let mine: Vec<_> = mine.collect();
                    if !dev.dirty && !mine.iter().any(|(_, l)| l.dirty) {
                        continue;
                    }
                    let mut table = vec![0i32; n_expert * dev.rows];
                    for (_, l) in mine {
                        table[l.row * n_expert..][..n_expert].copy_from_slice(&l.slots.table());
                        l.dirty = false;
                    }
                    dev.table().set_i32(&table);
                    dev.dirty = false;
                }
                Some(scope) => {
                    let (inside, outside): (Vec<_>, Vec<_>) = mine.partition(|(il, _)| scope(*il));
                    if !inside.iter().any(|(_, l)| l.dirty) {
                        continue;
                    }
                    let lo = inside.iter().map(|(_, l)| l.row).min().unwrap_or(0);
                    let hi = inside.iter().map(|(_, l)| l.row + 1).max().unwrap_or(0);
                    if outside.iter().all(|(_, l)| l.row < lo || l.row >= hi) {
                        // One write over the selected layers' rows (unchanged ones keep their
                        // bytes).
                        let mut table = vec![0i32; n_expert * (hi - lo)];
                        for (_, l) in inside {
                            table[(l.row - lo) * n_expert..][..n_expert]
                                .copy_from_slice(&l.slots.table());
                            l.dirty = false;
                        }
                        dev.table().set_i32_at(lo * n_expert, &table);
                    } else {
                        for (_, l) in inside.into_iter().filter(|(_, l)| l.dirty) {
                            dev.table().set_i32_at(l.row * n_expert, &l.slots.table());
                            l.dirty = false;
                        }
                    }
                }
            }
        }
        self.upload_cached();
        Ok(())
    }

    /// Writes every device's `expert_cached` rows ([`ResidencyConfig::ep_cached`]) where they
    /// changed: each entry's row says which of its model layer's experts the entries serving
    /// that layer hold in a ready slot (each expert's owner). Every device gets the same row
    /// for a layer in the same call, before the next pass starts, so the GPUs of a pair agree
    /// on the pair balance's assignment ([`super::ep_balance`]) whatever the tables say.
    fn upload_cached(&mut self) {
        let Some(real) = &self.ep_real else { return };
        if self.devices.iter().all(|d| d.cached().is_none()) {
            return;
        }
        let n_expert = self.n_expert;
        let n_real = real.iter().max().map_or(0, |&r| r + 1);
        let mut by_real = vec![vec![0i32; n_expert]; n_real];
        for (il, l) in self.layers.iter().enumerate() {
            let (Some(l), Some(&r)) = (l, real.get(il)) else {
                continue;
            };
            let n_slots = l.slots.n_slots();
            for (e, &t) in l.slots.table().iter().enumerate() {
                if (t as usize) < n_slots {
                    by_real[r][e] = 1;
                }
            }
        }
        self.cached_rows.resize(self.devices.len(), Vec::new());
        for (d, dev) in self.devices.iter().enumerate() {
            let Some(t) = dev.cached() else { continue };
            let mut rows = vec![0i32; n_expert * dev.rows];
            for (il, l) in self.layers.iter().enumerate() {
                let (Some(l), Some(&r)) = (l, real.get(il)) else {
                    continue;
                };
                if l.dev == d {
                    rows[l.row * n_expert..][..n_expert].copy_from_slice(&by_real[r]);
                }
            }
            if rows != self.cached_rows[d] {
                t.set_i32(&rows);
                self.cached_rows[d] = rows;
            }
        }
    }

    /// After a pass of `n` tokens through the layers `ran` says (computed, synchronized): marks
    /// finished copies, learns the pass's routing if it was logged, plans and starts swaps, and
    /// writes changed tables, over every layer. Must run before the next pass starts.
    pub fn after_step(&mut self, n: usize, ran: impl Fn(usize) -> bool) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        self.tier_settled()?;
        #[cfg(feature = "cuda")]
        self.refill(None, false)?;
        self.learn(n, &ran, None, true)
    }

    /// The bounded host tier: waits for the devices before the after-step work. Without the
    /// wait a kernel of a step soon after a restore read unmapped memory (an illegal access,
    /// in both runs that lent the cache); with it none did. What the kernel raced with is
    /// not known yet; decode measured the same with the wait.
    #[cfg(feature = "cuda")]
    fn tier_settled(&self) -> Result<(), String> {
        if self.host.is_none() {
            return Ok(());
        }
        for c in self.devices.iter().filter_map(|d| d.place.cuda) {
            super::cuda::sync_device(c)?;
        }
        Ok(())
    }

    /// [`Self::after_step`] that touches only the layers `ran` says: it completes their
    /// finished copies, learns their routing, plans and starts swaps among their slots and
    /// writes their table rows, so a pass through other layers may run meanwhile (another
    /// layer's copies stay pending until its own call). Once the refill after
    /// [`Self::restore`] is done, it points these layers' tables at their slots again.
    pub fn after_step_only(&mut self, n: usize, ran: impl Fn(usize) -> bool) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        self.tier_settled()?;
        #[cfg(feature = "cuda")]
        self.refill(Some(&ran), false)?;
        self.learn(n, &ran, Some(&ran), true)
    }

    /// [`Self::after_step_only`] for some layers of a pass whose work is split over several
    /// calls (#362), all but the last: the same work for these layers, but the pass counts as
    /// a logged step (and logs the periodic report) only in its last call, an
    /// [`Self::after_step_only`]. Planning is per device, so as long as each device's layers
    /// are in one call, the calls plan the swaps one call over all of them would.
    pub fn after_step_more(&mut self, n: usize, ran: impl Fn(usize) -> bool) -> Result<(), String> {
        #[cfg(feature = "cuda")]
        self.tier_settled()?;
        #[cfg(feature = "cuda")]
        self.refill(Some(&ran), false)?;
        self.learn(n, &ran, Some(&ran), false)
    }

    /// Adds the tallies of the layers `ran` selects (long steps' routing samples,
    /// [`ResidencyConfig::learn_long`]) to their heat and routing counts, and clears them.
    fn learn_tally(&mut self, ran: &dyn Fn(usize) -> bool) {
        let n_expert = self.n_expert;
        let decay = self.decay;
        for d in 0..self.devices.len() {
            let Some((tally, _)) = self.devices[d].tally() else {
                continue;
            };
            let (lo, hi) = self
                .layers
                .iter()
                .enumerate()
                .filter_map(|(il, l)| l.as_ref().filter(|l| l.dev == d && ran(il)))
                .fold((usize::MAX, 0), |(lo, hi), l| {
                    (lo.min(l.row), hi.max(l.row + 1))
                });
            if lo >= hi {
                continue;
            }
            let vals = tally.f32_range(lo * n_expert, (hi - lo) * n_expert);
            tracing::debug!(
                device = d,
                rows = ?(lo, hi),
                routed = vals.iter().sum::<f32>(),
                "expert cache: long-step tally"
            );
            for (_, l) in self
                .layers
                .iter_mut()
                .enumerate()
                .filter_map(|(il, l)| l.as_mut().map(|l| (il, l)))
                .filter(|(il, l)| l.dev == d && ran(*il))
            {
                let seen = &vals[(l.row - lo) * n_expert..][..n_expert];
                if seen.iter().all(|&x| x == 0.0) {
                    continue;
                }
                l.slots.decay(decay);
                l.slots.add_heat(seen);
                for (c, &x) in l.counts.iter_mut().zip(seen) {
                    *c += x as u64;
                }
                tally.memset(l.row * n_expert * 4, n_expert * 4, 0);
            }
        }
    }

    /// The device (index into the cache's devices) of layer `il`'s slots, if it is tiered.
    pub fn layer_device(&self, il: usize) -> Option<usize> {
        self.layers.get(il)?.as_ref().map(|l| l.dev)
    }

    fn learn(
        &mut self,
        n: usize,
        ran: &dyn Fn(usize) -> bool,
        scope: Option<&dyn Fn(usize) -> bool>,
        count: bool,
    ) -> Result<(), String> {
        self.complete_copies(scope)?;
        // Model bytes (#391): whether the pass gathered its experts or read staged copies.
        let gathered = self.fetch == Fetch::Copy && n as i64 <= COPY_MAX_TOKENS;
        #[cfg(feature = "cuda")]
        let devices = &self.devices;
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let staged = |l: &Layer| -> bool {
            #[cfg(feature = "cuda")]
            return devices[l.dev].stage().is_some()
                && n as i64 >= self.stage_min
                && !l.staged.is_empty();
            #[cfg(not(feature = "cuda"))]
            false
        };
        if n > LOG_TOKENS {
            let pairs = n * self.n_used;
            for (_, l) in self
                .layers
                .iter_mut()
                .enumerate()
                .filter_map(|(il, l)| l.as_mut().map(|l| (il, l)))
                .filter(|(il, _)| ran(*il))
            {
                let st = staged(l);
                l.read_at_most(pairs, st);
            }
        }
        if n > 0 && n <= LOG_TOKENS {
            if count {
                self.logged += 1;
            }
            let per = self.n_used * n;
            let cap = self.n_used * LOG_TOKENS;
            // Expert parallelism: every entry's owned experts of the pass (#729).
            let mut passes: Option<Vec<super::ep_balance::EntryPass>> =
                self.ep_real.as_ref().map(|_| Vec::new());
            // `--glm-prefetch-stats`: whether an entry of the pass had a next-layer guess.
            let mut predicted = false;
            for d in 0..self.devices.len() {
                let dev = &self.devices[d];
                // Only the rows of the layers that ran: others may be written meanwhile.
                let rows = self
                    .layers
                    .iter()
                    .enumerate()
                    .filter_map(|(il, l)| l.as_ref().filter(|l| l.dev == d && ran(il)))
                    .map(|l| l.row);
                let (lo, hi) = rows.fold((usize::MAX, 0), |(lo, hi), r| (lo.min(r), hi.max(r + 1)));
                if lo >= hi {
                    continue;
                }
                let ids = dev.log().i32_range(lo * cap, cap * (hi - lo));
                let guesses = dev
                    .pred_log()
                    .map(|t| t.i32_range(lo * cap, cap * (hi - lo)));
                let gpu = dev.place.cuda.unwrap_or(d);
                for (il, l) in self
                    .layers
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(il, l)| l.as_mut().map(|l| (il, l)))
                    .filter(|(il, l)| l.dev == d && ran(*il))
                {
                    let mine = &ids[(l.row - lo) * cap..][..per];
                    let unpacked;
                    let mine = if staged(l) && !l.stage_ids.is_empty() {
                        unpacked = l.unpack(mine);
                        &unpacked[..]
                    } else {
                        mine
                    };
                    l.read(mine, gathered, staged(l));
                    if let Some(g) = &guesses {
                        let guess = &g[(l.row - lo) * cap..][..per];
                        predicted |= predict_entry(&mut self.predict, l, gpu, mine, guess);
                    }
                    if let Some(p) = passes.as_mut() {
                        p.push(super::ep_balance::EntryPass {
                            real: self
                                .ep_real
                                .as_ref()
                                .and_then(|r| r.get(il).copied())
                                .unwrap_or(il),
                            gpu,
                            chosen: l.owned_chosen(mine),
                        });
                    }
                    l.slots.record_capped(mine, self.heat_pass_cap);
                    for &id in mine {
                        if let Some((e, _)) = l.slots.decode(id) {
                            l.counts[e as usize] += 1;
                        }
                    }
                    l.slots.decay(self.decay);
                }
            }
            if let Some(p) = passes {
                self.balance.add_pass(&p);
            }
            if predicted {
                self.predict.end_pass();
            }
            // No swaps into lent slots: their VRAM is gone until `restore`.
            if self.lent.is_none() {
                for d in 0..self.devices.len() {
                    if self.rule.on_device(&self.device_swaps, d).budget > 0 {
                        self.start_swaps(d, scope)?;
                    }
                }
            }
            if count && self.logged.is_multiple_of(REPORT_EVERY) {
                let s = self.stats();
                let (h, m) = (s.hits - self.reported.0, s.misses - self.reported.1);
                tracing::info!(
                    hit_rate = h as f64 / (h + m).max(1) as f64,
                    swaps = s.swaps,
                    logged_steps = s.logged_steps,
                    "expert cache, last {REPORT_EVERY} logged steps"
                );
                self.reported = (s.hits, s.misses);
                self.report_balance();
            }
        }
        // While lent, a long prompt's tallies wait for `restore`: reading one here would wait
        // for the device's next micro-chunk, which is already running.
        if n > LOG_TOKENS && self.lent.is_none() {
            self.learn_tally(ran);
        }
        self.upload_tables(scope)
    }

    /// Logs the expert-parallel steps' routed work since the last report (#729): per GPU its
    /// chosen experts per step (VRAM hits, host misses), and per step the sum over the layers
    /// of the busiest GPU's excess over the mean, in misses and in work units
    /// ([`super::ep_balance::MISS_COST`]), as owned and as the pair and cross (#731) balances
    /// would give it.
    fn report_balance(&mut self) {
        self.report_predict();
        let b = self.balance.take();
        if b.steps == 0 {
            return;
        }
        let per = |x: f64| (x / b.steps as f64 * 100.0).round() / 100.0;
        tracing::info!(
            steps = b.steps,
            layers_per_step = per(b.layers as f64),
            hits_per_step = ?b.hits.iter().map(|&x| per(x as f64)).collect::<Vec<_>>(),
            misses_per_step = ?b.misses.iter().map(|&x| per(x as f64)).collect::<Vec<_>>(),
            miss_excess = per(b.miss_excess),
            cost_excess = per(b.cost_excess),
            pair_cost_excess = per(b.pair_excess),
            cross_cost_excess = per(b.cross_excess),
            miss_cost = super::ep_balance::MISS_COST,
            cross_miss_cost = super::ep_balance::CROSS_MISS_COST,
            "expert parallelism: routed work per step (busiest GPU over the mean, summed over layers)"
        );
    }

    /// Logs the prefetch predictors since the last report (`--glm-prefetch-stats`, #745): per
    /// GPU and logged step (summed over the layers that follow a MoE layer), its misses, how
    /// many of them the next-layer router guess (`next_hits`), the previous step's choices
    /// (`prev_hits`) or either named, how many uncached experts each would copy, and the share
    /// of all chosen experts the guess named (`next_recall`).
    fn report_predict(&mut self) {
        let s = self.predict.take();
        if s.steps == 0 {
            return;
        }
        let per = |x: u64| (x as f64 / s.steps as f64 * 100.0).round() / 100.0;
        let col = |f: fn(&super::ep_prefetch::GpuCounts) -> u64| -> Vec<f64> {
            s.gpus.iter().map(|g| per(f(g))).collect()
        };
        let share = |a: u64, b: u64| (a as f64 / b.max(1) as f64 * 1000.0).round() / 1000.0;
        let sum =
            |f: fn(&super::ep_prefetch::GpuCounts) -> u64| -> u64 { s.gpus.iter().map(f).sum() };
        let misses = sum(|g| g.misses);
        tracing::info!(
            steps = s.steps,
            misses = ?col(|g| g.misses),
            next_hits = ?col(|g| g.next_hits),
            prev_hits = ?col(|g| g.prev_hits),
            either_hits = ?col(|g| g.either_hits),
            next_copies = ?col(|g| g.next_copies),
            prev_copies = ?col(|g| g.prev_copies),
            next_miss_rate = share(sum(|g| g.next_hits), misses),
            prev_miss_rate = share(sum(|g| g.prev_hits), misses),
            either_miss_rate = share(sum(|g| g.either_hits), misses),
            next_recall = share(sum(|g| g.next_chosen), sum(|g| g.chosen)),
            "expert prefetch predictors (per GPU and step: misses foreseen, uncached copies)"
        );
    }

    /// Plans and starts device `d`'s swaps among the layers `scope` selects (all: None).
    fn start_swaps(
        &mut self,
        d: usize,
        scope: Option<&dyn Fn(usize) -> bool>,
    ) -> Result<(), String> {
        let swaps = {
            let mut view: Vec<Option<&mut LayerSlots>> = self
                .layers
                .iter_mut()
                .enumerate()
                .map(|(il, l)| {
                    l.as_mut()
                        .filter(|l| l.dev == d && l.backed && scope.is_none_or(|f| f(il)))
                        .map(|l| &mut l.slots)
                })
                .collect();
            plan_swaps(&mut view, self.rule.on_device(&self.device_swaps, d))
        };
        if swaps.is_empty() {
            return Ok(());
        }
        self.swaps += swaps.len() as u64;
        let tier = self.host.clone();
        for s in &swaps {
            self.layers[s.layer].as_mut().unwrap().dirty = true;
            match &tier {
                // The bounded host tier: the dropped expert's bytes go to a RAM slot first,
                // the new one's come from its RAM slot (or, read first, its staging place:
                // those bytes only stay until the next read, so that copy is waited for).
                Some(h) => {
                    let n_slots = self.layers[s.layer].as_ref().unwrap().slots.n_slots();
                    let at = h.swap(s.layer, s.new, s.old)?;
                    if let Some(back) = at.back {
                        self.copy_expert(s.layer, (n_slots + back) as u32, s.slot as usize)?;
                    }
                    self.copy_expert(s.layer, s.slot, n_slots + at.src)?;
                    if at.staged {
                        self.sync_copies()?;
                    }
                }
                None => self.copy_from_host(s.layer, &[(s.slot, s.new)])?,
            }
        }
        if let Some(h) = &tier {
            h.swaps_queued();
            if self.devices[d].place.cuda.is_none() {
                h.swaps_landed();
            }
        }
        let filled: Vec<(usize, u32)> = swaps.iter().map(|s| (s.layer, s.slot)).collect();
        let dev = &mut self.devices[d];
        match dev.place.cuda {
            #[cfg(feature = "cuda")]
            Some(_) => {
                let event = dev.copier.as_mut().expect("copier").mark()?;
                dev.inflight.push_back(InFlight {
                    event,
                    slots: filled,
                    tier: tier.is_some(),
                });
            }
            _ => {
                // Host copies are done already.
                for (il, slot) in filled {
                    self.layers[il].as_mut().unwrap().slots.complete(slot);
                }
            }
        }
        Ok(())
    }

    /// Marks the slots of finished copies filled, of the layers `scope` selects (all: None);
    /// others stay queued.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn complete_copies(&mut self, scope: Option<&dyn Fn(usize) -> bool>) -> Result<(), String> {
        // `settle` (#1165): every copy queued so far counts, finished or not yet.
        if self.settle {
            self.sync_copies()?;
        }
        #[cfg(feature = "cuda")]
        for dev in &mut self.devices {
            let mut i = 0;
            // One copy stream per device: its copies finish in order.
            while i < dev.inflight.len() {
                if !dev.inflight[i].event.done()? {
                    break;
                }
                if std::mem::take(&mut dev.inflight[i].tier)
                    && let Some(h) = &self.host
                {
                    h.swaps_landed();
                }
                let layers = &mut self.layers;
                dev.inflight[i].slots.retain(|&(il, slot)| {
                    if !scope.is_none_or(|f| f(il)) {
                        return true;
                    }
                    let l = layers[il].as_mut().unwrap();
                    l.slots.complete(slot);
                    l.dirty = true;
                    false
                });
                if dev.inflight[i].slots.is_empty() {
                    dev.inflight.remove(i);
                } else {
                    i += 1;
                }
            }
        }
        Ok(())
    }

    /// Slots of layer `il` whose copies are queued or not yet marked filled (for tests).
    pub fn copies_pending(&self, il: usize) -> usize {
        #[cfg(feature = "cuda")]
        return self
            .devices
            .iter()
            .flat_map(|d| &d.inflight)
            .flat_map(|f| &f.slots)
            .filter(|&&(l, _)| l == il)
            .count();
        #[cfg(not(feature = "cuda"))]
        {
            let _ = il;
            0
        }
    }

    /// The table row of layer `il` as uploaded to its device (for tests).
    pub fn uploaded_table(&self, il: usize) -> Option<Vec<i32>> {
        let l = self.layers.get(il)?.as_ref()?;
        Some(
            self.devices[l.dev]
                .table()
                .i32_range(l.row * self.n_expert, self.n_expert),
        )
    }

    /// Replaces the swap rule.
    pub fn set_swap_rule(&mut self, rule: SwapRule) {
        self.rule = rule;
    }

    /// Whether long prompts can borrow the slots' VRAM for staging ([`Self::lend`]).
    pub fn can_lend(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.devices.iter().any(|d| d.lend_bytes.is_some());
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Stages only the layers below `n` while lent: the rest (an MTP head's layer, whose prompt
    /// take-in writes its K/V only) read their experts from host memory. Its experts may be
    /// wider than the trunk's (Q8_0 next to Q4_K), and staging sizes every buffer of a device
    /// for its widest layer.
    pub fn stage_below(&mut self, n: usize) {
        self.stage_limit = n;
        #[cfg(feature = "cuda")]
        self.size_staging();
    }

    /// Whether entry `il` stages while lent: its model layer is below [`Self::stage_below`]
    /// (an expert-parallel copy past the model's layers stages as its layer does, unless its
    /// layer's own entry computes every expert in prompts: [`stages_entry`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn stages_entry(&self, il: usize) -> bool {
        stages_entry(
            self.ep_real.as_deref(),
            il,
            self.stage_limit,
            self.ep_stage_all,
        )
    }

    /// Sizes each staging device's buffers for its layers below [`Self::stage_below`].
    #[cfg(feature = "cuda")]
    fn size_staging(&mut self) {
        let n_expert = self.n_expert;
        let staged: Vec<bool> = (0..self.layers.len())
            .map(|il| self.stages_entry(il))
            .collect();
        for (d, dev) in self.devices.iter_mut().enumerate() {
            if dev.lend_bytes.is_none() {
                continue;
            }
            let mut bytes = [0usize; 3];
            let mine = self.layers.iter().zip(&staged).filter(|(_, s)| **s);
            for l in mine.filter_map(|(l, _)| l.as_ref()).filter(|l| l.dev == d) {
                for (b, e) in bytes.iter_mut().zip(l.expert_bytes) {
                    // One more expert's worth, as in [`staging_bytes`].
                    *b = (*b).max((n_expert + 1) * e);
                }
            }
            dev.lend_bytes = (bytes.iter().sum::<usize>() > 0).then_some(bytes);
        }
    }

    /// Whether the slots are lent out now.
    pub fn is_lent(&self) -> bool {
        self.lent.is_some()
    }

    /// VRAM a device needs from its lent slots for staging, by CUDA index.
    pub fn stage_bytes(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "cuda")]
        return self
            .devices
            .iter()
            .filter_map(|d| Some((d.place.cuda?, stage_footprint(d.lend_bytes?, self.ring))))
            .collect();
        #[cfg(not(feature = "cuda"))]
        Vec::new()
    }

    /// Lends the VRAM slots of every staging device to a long prompt: their experts are read
    /// from host memory (the tables point there), the slots' memory is freed, and staging
    /// buffers are allocated in its place, so big steps copy each layer's experts into VRAM
    /// before its matmuls (see [`super::staging`]). The compute buffers of the prompt's big
    /// steps get the rest. No step may run while it lends; graphs built before it must be
    /// rebuilt (their staged ops are new). [`Self::restore`] undoes it.
    ///
    /// A refill after the last restore that is still under way stops at its next layer rather
    /// than finish (#447: waiting for it cost a long prompt 0.6-2 s): the layers it did not
    /// reach have no VRAM yet and keep the experts they held, which the next restore refills.
    /// Every expert is read from host memory or staged while lent either way.
    pub fn lend(&mut self) -> Result<(), String> {
        self.lend_for(usize::MAX)
    }

    /// [`Self::lend`] for a prompt of `tokens` tokens still to read: kept slots are re-packed
    /// ([`ResidencyConfig::lend_repack`]) only for a prompt of at least that many.
    pub fn lend_for(&mut self, tokens: usize) -> Result<(), String> {
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let repack_kept = self.lend_repack > 0 && tokens >= self.lend_repack;
        if self.lent.is_some() || !self.can_lend() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        {
            // `settle` (#1165): the refill runs to its end, so the layers it gave back do not
            // follow when the prompt came.
            if !self.settle {
                self.stop_refill.store(true, Ordering::Relaxed);
            }
            self.refill(None, true)?;
        }
        self.sync_copies()?;
        self.complete_copies(None)?;
        #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
        let t_sync = t0.elapsed();
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut held: Vec<Vec<u32>> = vec![Vec::new(); self.layers.len()];
        #[cfg(feature = "cuda")]
        {
            let staging: Vec<bool> = self
                .devices
                .iter()
                .map(|d| d.lend_bytes.is_some())
                .collect();
            // Layers a stopped refill did not reach.
            let unrefilled = self
                .layers
                .iter()
                .flatten()
                .filter(|l| staging[l.dev] && !l.backed)
                .count();
            // Layers that keep their slots (`lend_keep`): staged ones with VRAM, while their
            // slots fit the device's budget, every other layer first (from the last one down),
            // then the rest: a kept layer copies only its uncached experts, and between two
            // layers that copy all of theirs it lets the staging ring catch up (each layer's
            // copies overlap the layer before's compute; a run of whole-layer copies outlasts it).
            let mut mode = vec![Lend::Whole; self.layers.len()];
            let staged: Vec<bool> = (0..self.layers.len())
                .map(|il| self.stages_entry(il))
                .collect();
            let spread = self.layers.iter().flatten().any(|l| l.keep.is_some());
            let mut budget: Vec<u64> = self
                .devices
                .iter()
                .map(|d| {
                    let keep = &self.lend_keep;
                    let i = d.place.cuda.unwrap_or(0);
                    keep.get(i).or(keep.last()).copied().unwrap_or(0)
                })
                .collect();
            let order = (0..self.layers.len())
                .rev()
                .filter(|il| il % 2 == 1)
                .chain((0..self.layers.len()).rev().filter(|il| il % 2 == 0));
            for il in order.filter(|_| !spread) {
                let Some(l) = self.layers[il].as_ref() else {
                    continue;
                };
                if !staging[l.dev] || !l.backed || !staged[il] {
                    continue;
                }
                let bytes = (0..3)
                    .map(|k| (l.slots.n_slots() * l.expert_bytes[k]) as u64)
                    .sum::<u64>();
                if bytes > 0 && bytes <= budget[l.dev] {
                    budget[l.dev] -= bytes;
                    mode[il] = Lend::Keep;
                }
            }
            // `lend_spread`: every staged layer whose kept slots still have VRAM lends only
            // its tail.
            for (il, l) in self.layers.iter().enumerate() {
                let Some(l) = l.as_ref().filter(|l| spread && staging[l.dev]) else {
                    continue;
                };
                if l.keep.is_some() && (l.backed || l.tail_lent) && staged[il] {
                    mode[il] = Lend::Tail;
                }
            }
            let kept = mode.iter().filter(|m| **m == Lend::Keep).count();
            // Staging kept across lends: not with whole kept layers (`lend_keep`).
            let reuse = self.stage_reuse && kept == 0;
            // `lend_repack` (#778): per device, the copies that re-pack its layers' kept slots,
            // and how many experts came from host memory and from VRAM.
            let n_dev = self.devices.len();
            let mut repack: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_dev];
            let mut repacked = vec![(0usize, 0usize); n_dev];
            for (il, l) in self.layers.iter_mut().enumerate() {
                let Some(l) = l.as_mut().filter(|l| staging[l.dev]) else {
                    continue;
                };
                match mode[il] {
                    Lend::Keep => continue,
                    Lend::Tail => {
                        let keep = l.keep.unwrap_or(0);
                        if repack_kept {
                            // The tail's bytes are still there unless a stopped refill left it
                            // without VRAM.
                            let moves = l.slots.repack_kept(keep, l.backed);
                            let n_slots = l.slots.n_slots() as u32;
                            let host = moves.iter().filter(|m| m.1 >= n_slots).count();
                            repacked[l.dev].0 += host;
                            repacked[l.dev].1 += moves.len() - host;
                            repack[l.dev].extend(l.repack_copies(&moves));
                        }
                        held[il] = l.slots.clear_from(keep);
                    }
                    Lend::Whole => held[il] = l.slots.clear(),
                }
                self.devices[l.dev].dirty = true;
            }
            // Packed staging: per staged layer, its uncached experts' ids past its kept slots,
            // and per device, the buffer size of its most uncached layer.
            let mut packed_bytes: Vec<Option<[usize; 3]>> = vec![None; self.devices.len()];
            // Per device: the runs of consecutive experts its staged layers copy each pass.
            let mut runs = vec![0usize; n_dev];
            for (il, l) in self.layers.iter_mut().enumerate() {
                let Some(l) = l.as_mut().filter(|l| spread && staging[l.dev]) else {
                    continue;
                };
                if !staged[il] {
                    continue;
                }
                let need = l.pack();
                runs[l.dev] += runs_of(&l.stage_miss).len();
                let bytes = packed_bytes[l.dev].get_or_insert([0; 3]);
                for (b, n) in bytes.iter_mut().zip(need) {
                    *b = (*b).max(n);
                }
                self.devices[l.dev]
                    .stage_table()
                    .set_i32_at(l.row * self.n_expert, &l.stage_ids);
            }
            // The bounded host tier: the experts the slots gave up are in the checkpoint only.
            if let Some(h) = &self.host {
                for (il, gone) in held.iter().enumerate() {
                    h.dropped(il, gone);
                }
            }
            // Every lookup reads host memory before any slot is freed.
            self.upload_tables(None)?;
            self.lent = Some(held);
            let t_tables = t0.elapsed();
            // Per staging device: its slots unmapped, then its staging buffers made of the
            // slots' own allocations (`Staging::new`) and mapped behind the holes. #439: a
            // 4-GPU lend took ~320 ms of every long prompt, ~200 ms of it creating new VRAM
            // (the driver clears it); over the slots' allocations it takes ~120 ms. Each
            // device on a thread of its own: the driver runs most VMM calls one at a time,
            // but each device's memsets and syncs overlap the others' calls. Every device
            // only unmaps and maps its own VRAM, in the same order.
            // Per device: its staged layers' rows and bytes per kind ([`ring_bytes`]).
            let (ring, n_expert) = (self.ring, self.n_expert);
            let mut ring_rows: Vec<Vec<(usize, [usize; 3])>> = vec![Vec::new(); n_dev];
            for (il, l) in self.layers.iter().enumerate() {
                if let Some(l) = l.as_ref().filter(|_| staged[il]) {
                    let b = l.expert_bytes.map(|e| (n_expert + 1) * e);
                    ring_rows[l.dev].push((l.row, b));
                }
            }
            let mut work: Vec<(usize, LendWork<'_>)> = self
                .devices
                .iter_mut()
                .enumerate()
                .filter(|(d, _)| staging[*d])
                .map(|(d, dev)| {
                    let parked = dev.staging.take().filter(|_| dev.kept);
                    let bytes = if spread {
                        packed_bytes[d]
                    } else {
                        dev.lend_bytes
                    };
                    // #1216: each buffer sized for the layers whose rows use it (packed
                    // staging: every buffer its most uncached layer's size, as before).
                    let need = bytes
                        .map(|b| ring_bytes(b, ring, if spread { &[] } else { &ring_rows[d][..] }));
                    let stage = dev.place.cuda.zip(need);
                    let stage = stage.map(|(c, need)| (c, need, dev.rows));
                    let layers = Vec::new();
                    tracing::debug!(
                        device = ?dev.place.cuda,
                        runs = runs[d],
                        repacked_from_host = repacked[d].0,
                        repacked_in_vram = repacked[d].1,
                        repack_copies = repack[d].len(),
                        "expert cache lend: staged runs a pass"
                    );
                    let moves = dev.place.cuda.map(|c| (c, std::mem::take(&mut repack[d])));
                    (
                        d,
                        LendWork {
                            stage,
                            layers,
                            parked,
                            repack: moves,
                        },
                    )
                })
                .collect();
            for (il, l) in self.layers.iter_mut().enumerate() {
                let Some(l) = l.as_mut() else { continue };
                if let Some((_, w)) = work.iter_mut().find(|(d, _)| *d == l.dev) {
                    w.layers.push((staged[il], mode[il], l));
                }
            }
            let (ring, pieces) = (self.ring, self.stage_pieces);
            let done = per_device_thread("legion-lend", work, |_, w| w.lend(ring, pieces, reuse));
            let mut first_err = None;
            let (mut unback_ms, mut alloc_ms, mut repack_ms) = (0f64, 0f64, 0f64);
            for (d, r) in done {
                let r = r.and_then(|r| {
                    unback_ms = unback_ms.max(r.unback_ms);
                    alloc_ms = alloc_ms.max(r.alloc_ms);
                    repack_ms = repack_ms.max(r.repack_ms);
                    r.staging
                });
                let dev = &mut self.devices[d];
                dev.kept = false;
                dev.parked = false;
                match r {
                    Ok((st, kept)) => {
                        dev.staging = st;
                        dev.kept = kept;
                    }
                    Err(e) => {
                        first_err.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = first_err {
                return Err(e);
            }
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            tracing::debug!(
                unrefilled,
                kept,
                sync_ms = ms(t_sync),
                tables_ms = ms(t_tables - t_sync),
                devices_ms = ms(t0.elapsed() - t_tables),
                unback_max_ms = unback_ms,
                staging_alloc_max_ms = alloc_ms,
                repack_max_ms = repack_ms,
                "expert cache lend"
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.lent = Some(held);
        }
        Ok(())
    }

    /// Takes the slots back after [`Self::lend`]: a thread per device frees its staged ranges
    /// and staging buffers, then gives the slots VRAM again (mapping fresh memory takes ~0.4 s
    /// for a V100's cache) and copies the experts they held back in on a copy stream of its own. Steps run
    /// on meanwhile, every expert read from host memory; once the thread is done
    /// ([`Self::after_step`] looks), the table entries point at the refilled slots. A
    /// [`Self::lend`] meanwhile stops the thread at its next layer.
    pub fn restore(&mut self) -> Result<(), String> {
        self.restore_after(|| ())
    }

    /// [`Self::restore`], whose thread first runs `free_first` (it frees the long prompt's
    /// compute buffers, #466: a `cudaFree` of a 1.5-2 GiB buffer sometimes took the engine
    /// thread 40-60 ms) before it maps the slots' VRAM again. [`Self::lend`] waits for it with
    /// the thread. Without a restore to run (or without CUDA), `free_first` runs right away.
    pub fn restore_after(
        &mut self,
        free_first: impl FnOnce() + Send + 'static,
    ) -> Result<(), String> {
        self.restore_leaving(free_first, &[])
    }

    /// [`Self::restore_after`] that leaves, per CUDA device `(index, bytes)` in `leave`, staged
    /// layers whose lent slots hold at least `bytes` without VRAM (their lent experts read from
    /// host memory, as after a refill a lend stopped): memory something else holds now (the
    /// KV pools' tail of `--kv-vmm-keep`, #814). The device's staging goes rather than parks.
    /// [`Self::refill_unbacked`] refills them once that memory is free again.
    pub fn restore_leaving(
        &mut self,
        free_first: impl FnOnce() + Send + 'static,
        leave: &[(usize, u64)],
    ) -> Result<(), String> {
        let Some(held) = self.lent.take() else {
            free_first();
            return Ok(());
        };
        self.restore_held(held, true, free_first, leave)?;
        // Leaving layers drops the staging instead of parking it: its frees and the slots'
        // fresh VRAM on the slot thread broke a CUDA graph the next step captured meanwhile
        // (2 V100s, #814). Done before the next step (once per sequence past the kept blocks).
        #[cfg(feature = "cuda")]
        if !leave.is_empty() {
            self.refill(None, true)?;
        }
        Ok(())
    }

    /// Refills, while not lent, the layers [`Self::restore_leaving`] left without VRAM (after a
    /// restore still under way). Nothing to do when every layer has VRAM.
    pub fn refill_unbacked(&mut self) -> Result<(), String> {
        if self.lent.is_some() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        self.refill(None, true)?;
        if self.layers.iter().flatten().all(|l| l.backed) {
            return Ok(());
        }
        let held = vec![Vec::new(); self.layers.len()];
        self.restore_held(held, false, || (), &[])?;
        // Fresh VRAM on the slot thread while steps capture graphs: done first (see
        // `restore_leaving`).
        #[cfg(feature = "cuda")]
        self.refill(None, true)?;
        Ok(())
    }

    /// Which layers [`Self::restore_leaving`] leaves without VRAM for `leave`: per device, its
    /// staged layers without VRAM from the last one down, until their lent slots' bytes (the
    /// tail past [`Layer::keep`], or every slot) reach the device's bytes.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn layers_left(&self, leave: &[(usize, u64)]) -> Vec<bool> {
        let mut left = vec![false; self.layers.len()];
        for &(cuda, bytes) in leave {
            let Some(d) = self.devices.iter().position(|d| d.place.cuda == Some(cuda)) else {
                continue;
            };
            let lent: Vec<Option<u64>> = (0..self.layers.len())
                .map(|il| {
                    let l = self.layers[il].as_ref()?;
                    if l.dev != d || l.backed || !self.stages_entry(il) {
                        return None;
                    }
                    let kept = if l.tail_lent { l.keep.unwrap_or(0) } else { 0 };
                    let n = l.slots.n_slots().saturating_sub(kept);
                    Some((0..3).map(|k| (n * l.expert_bytes[k]) as u64).sum())
                })
                .collect();
            for (x, y) in left.iter_mut().zip(layers_left_for(&lent, bytes)) {
                *x |= y;
            }
        }
        left
    }

    /// [`Self::restore_leaving`] of the slots `held` held when lent.
    fn restore_held(
        &mut self,
        held: Vec<Vec<u32>>,
        was_lent: bool,
        free_first: impl FnOnce() + Send + 'static,
        leave: &[(usize, u64)],
    ) -> Result<(), String> {
        // The long prompt's routing, before the slots are refilled by it.
        self.learn_tally(&|_| true);
        #[cfg(feature = "cuda")]
        {
            let t0 = std::time::Instant::now();
            // `leave`: layers kept without VRAM, and the devices whose staging goes (parked,
            // it would hold their lent slots' memory).
            let left = self.layers_left(leave);
            let leaving: Vec<bool> = (0..self.devices.len())
                .map(|d| {
                    self.layers
                        .iter()
                        .zip(&left)
                        .any(|(l, &x)| x && l.as_ref().is_some_and(|l| l.dev == d))
                })
                .collect();
            // Per device: its staged ranges and staging buffers, which its refill thread frees
            // first (#450: their ~150 driver calls took the engine thread 2-3.5 s whenever
            // another process held the driver's lock). Not lent ([`Self::refill_unbacked`]):
            // none, and kept staging stays parked as it is.
            let mut lent = Vec::new();
            for (d, dev) in self.devices.iter_mut().enumerate().filter(|_| was_lent) {
                let Some(c) = dev.place.cuda else { continue };
                if let Some(st) = dev.staging.as_ref().filter(|_| !dev.parked) {
                    let s = st.take_stats();
                    tracing::debug!(
                        device = c,
                        rows = s.rows,
                        copies = s.copies,
                        queue_ms = s.queue_ms,
                        lock_wait_ms = s.lock_wait_ms,
                        "expert cache staging queued"
                    );
                }
                // Kept staging stays, parked by the refill thread before the slots map its
                // buffers' memory again.
                let kept = dev.kept && dev.staging.is_some() && !leaving[d];
                let staged: Vec<Context> = self
                    .layers
                    .iter_mut()
                    .flatten()
                    .filter(|l| l.dev == d)
                    .flat_map(|l| {
                        l.stage_ids.clear();
                        l.stage_miss.clear();
                        std::mem::take(&mut l.staged)
                    })
                    .collect();
                let ranges: Vec<super::cuda::StagedRange> = self
                    .layers
                    .iter_mut()
                    .flatten()
                    .filter(|l| l.dev == d && !kept)
                    .flat_map(|l| {
                        l.shared = [None; 3];
                        std::mem::take(&mut l.ranges)
                    })
                    .collect();
                if kept {
                    dev.parked = true;
                    let park = dev.staging.as_ref().map(|st| st.parker());
                    lent.push((
                        c,
                        Lent {
                            staged,
                            ranges,
                            staging: None,
                            park,
                        },
                    ));
                } else if !staged.is_empty() || dev.staging.is_some() || !ranges.is_empty() {
                    dev.kept = false;
                    lent.push((
                        c,
                        Lent {
                            staged,
                            ranges,
                            staging: dev.staging.take(),
                            park: None,
                        },
                    ));
                }
            }
            // Per layer to restore: its device, slot ranges and the copies that refill them.
            let mut work = Vec::new();
            for (il, (l, held)) in self.layers.iter_mut().zip(&held).enumerate() {
                // Not lent: only devices without parked staging (it would hold their memory).
                let Some(l) = l
                    .as_mut()
                    .filter(|l| !l.backed && !left[il] && (was_lent || !self.devices[l.dev].kept))
                else {
                    continue;
                };
                // Kept staging: the slots map their own allocations back.
                let kept = self.devices[l.dev].kept;
                let slots = (0..3)
                    .map(|k| {
                        let s = super::cuda::Slots::of(l.tensor(k))?;
                        Ok(s.with_parts(l.shared[k].filter(|_| kept)))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                // The experts routed to most that the kept slots do not hold (the kept ones stay:
                // a lend copies the others in runs of consecutive ids); before any routing was
                // seen, those it took.
                // The bounded host tier has no host bytes of every expert: its layers' slots
                // only take VRAM here, and `refill` fills them from the tier's RAM slots.
                let fills = match l.slots.ranked() {
                    _ if l.tier => Vec::new(),
                    Some(ranked) => l.slots.refill(&ranked),
                    None => l.slots.refill(held),
                };
                let n_slots = l.slots.n_slots();
                let mut copies = Vec::with_capacity(3 * fills.len());
                for &(slot, e) in &fills {
                    for k in 0..3 {
                        let n = l.expert_bytes[k];
                        copies.push((l.addr(k, slot as usize), l.addr(k, n_slots + e as usize), n));
                    }
                }
                let device = self.devices[l.dev].place.cuda.expect("a CUDA layer");
                work.push((il, (device, (slots, copies))));
            }
            // Started by the engine thread: off its core (#250).
            let pin = crate::hw::host_core::off_host_core(&[]);
            let stop = Arc::new(AtomicBool::new(false));
            self.stop_refill = stop.clone();
            let lent = std::sync::Mutex::new(lent);
            let thread = std::thread::Builder::new()
                .name("legion-slots".into())
                .spawn(move || {
                    crate::hw::host_core::pin_helper(&pin);
                    // Before any slot takes VRAM again: what it needs may be in there.
                    let t = std::time::Instant::now();
                    free_first();
                    tracing::debug!(
                        ms = t.elapsed().as_secs_f64() * 1e3,
                        "expert cache restore: freed first"
                    );
                    // One thread per device, all at once (#450): mapping fresh VRAM takes
                    // ~0.4 s per V100 cache, which one thread paid device after device.
                    let done = refill_per_device(by_device(work), |device, work| {
                        crate::hw::host_core::pin_helper(&pin);
                        let freed = take_device(&lent, device).map(|l| l.free(device));
                        refill_device(device, work, freed, &stop)
                    });
                    // A device with nothing to refill.
                    for (device, l) in lent.into_inner().unwrap_or_else(|e| e.into_inner()) {
                        l.free(device);
                    }
                    done
                })
                .map_err(|e| format!("cannot start the slot thread: {e}"))?;
            self.restoring = Some(thread);
            tracing::debug!(
                ms = t0.elapsed().as_secs_f64() * 1e3,
                "expert cache restore"
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (leave, was_lent);
            drop(held);
            free_first();
        }
        Ok(())
    }

    /// After [`Self::restore`]: once its thread is done (`wait`: waits for it), points the table
    /// entries of the layers `scope` selects (all: None) at their refilled slots; the other
    /// layers' wait for a call that covers them (nothing to do when no restore is under way).
    #[cfg(feature = "cuda")]
    fn refill(&mut self, scope: Option<&dyn Fn(usize) -> bool>, wait: bool) -> Result<(), String> {
        // `settle` (#1165): the next after-step takes the whole refill, not what is done by then.
        let wait = wait || self.settle;
        if self
            .restoring
            .as_ref()
            .is_some_and(|h| wait || h.is_finished())
        {
            let thread = self.restoring.take().expect("checked");
            let done = thread.join().map_err(|_| "the slot thread panicked")?;
            self.refilled.extend(done);
        }
        if self.refilled.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let (now, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.refilled)
            .into_iter()
            .partition(|(il, _)| scope.is_none_or(|f| f(*il)));
        self.refilled = later;
        let mut n = 0;
        let mut tier = Vec::new();
        for (il, r) in now {
            let l = self.layers[il].as_mut().expect("tiered layer");
            match r {
                Ok(()) => {
                    l.backed = true;
                    l.tail_lent = false;
                    for s in 0..l.slots.n_slots() as u32 {
                        l.slots.complete(s);
                    }
                    if l.tier {
                        tier.push(il);
                    }
                    n += 1;
                }
                Err(e) => {
                    // The layer's slots stay empty (its experts read from host memory) until a
                    // later restore finds room.
                    tracing::warn!(layer = il, error = %e, "expert cache: slots not restored");
                    l.slots.clear();
                }
            }
            l.dirty = true;
        }
        // The bounded host tier: the emptied slots take the most routed experts its RAM
        // slots hold (copies over the link, no checkpoint read); the swaps bring the others.
        let mut from_ram = 0;
        if let Some(h) = self.host.clone() {
            for il in tier {
                let in_ram = h.in_ram(il);
                let l = self.layers[il].as_mut().expect("tiered layer");
                let Some(ranked) = l.slots.ranked() else {
                    continue;
                };
                let hot: Vec<u32> = ranked.into_iter().filter(|&e| in_ram[e as usize]).collect();
                let fills = l.slots.refill(&hot);
                from_ram += fills.len();
                self.copy_from_host(il, &fills)?;
                let l = self.layers[il].as_mut().expect("tiered layer");
                for &(s, _) in &fills {
                    l.slots.complete(s);
                }
            }
        }
        tracing::debug!(
            waited_ms = t0.elapsed().as_secs_f64() * 1e3,
            layers = n,
            from_ram,
            "expert cache refilled"
        );
        Ok(())
    }

    /// The routing counted so far as a profile (every routed pair, most routed first).
    pub fn learned_profile(&self) -> Profile {
        let mut heat: Vec<Vec<f32>> = self
            .layers
            .iter()
            .map(|l| {
                l.as_ref().map_or_else(
                    || vec![0.0; self.n_expert],
                    |l| l.counts.iter().map(|&c| c as f32).collect(),
                )
            })
            .collect();
        // Expert parallelism: each copy's counts are its layer's (of its own experts).
        if let Some(real) = &self.ep_real {
            let n = (0..heat.len()).filter(|&i| real.get(i) == Some(&i)).count();
            for i in n..heat.len() {
                let Some(&r) = real.get(i).filter(|&&r| r < n) else {
                    continue;
                };
                let add = std::mem::take(&mut heat[i]);
                for (h, a) in heat[r].iter_mut().zip(add) {
                    *h += a;
                }
            }
            heat.truncate(n);
        }
        Profile::from_heat(&heat)
    }

    /// Expert parallelism: the model layer each entry serves ([`super::EpLayers::real`]).
    pub fn ep_real(&self) -> Option<&[usize]> {
        self.ep_real.as_deref()
    }

    /// The entry index of model layer `il`'s copy on CUDA device `cuda` (expert parallelism),
    /// or `il` itself when that entry is on `cuda`.
    pub fn entry_on(&self, il: usize, cuda: usize) -> Option<usize> {
        let on = |i: usize| {
            self.layers
                .get(i)?
                .as_ref()
                .filter(|l| self.devices[l.dev].place.cuda == Some(cuda))
                .map(|_| i)
        };
        if let Some(i) = on(il) {
            return Some(i);
        }
        let real = self.ep_real.as_ref()?;
        (0..real.len())
            .filter(|&i| i != il && real[i] == il)
            .find_map(on)
    }
}

impl Drop for ExpertResidency {
    fn drop(&mut self) {
        // The share's pools read the host experts below: stop them first.
        if self.share > 0 {
            super::cpu_share::ops::clear();
        }
        // The slot thread maps memory into ranges the tensors below own.
        #[cfg(feature = "cuda")]
        if let Some(thread) = self.restoring.take() {
            let _ = thread.join();
        }
        let _ = self.sync_copies();
        // Raw routing counts per layer and expert, for offline cache studies. Debug output
        // (docs/debug-env.md), not a setting, so an environment variable rather than a flag
        // (#737).
        if let Ok(path) = std::env::var("LEGION_EXPERT_COUNTS") {
            let counts: Vec<Vec<u64>> = self
                .layers
                .iter()
                .map(|l| l.as_ref().map_or_else(Vec::new, |l| l.counts.clone()))
                .collect();
            if let Ok(text) = serde_json::to_string(&counts) {
                let _ = std::fs::write(path, text);
            }
        }
        self.learn_tally(&|_| true);
        let s = self.stats();
        if s.logged_steps > 0 {
            tracing::info!(
                hit_rate = s.hit_rate(),
                hits = s.hits,
                misses = s.misses,
                swaps = s.swaps,
                logged_steps = s.logged_steps,
                "expert cache"
            );
            // Per layer, for sizing the split (`--expert-profile-split-only`).
            let per = self.layer_stats();
            tracing::info!(
                slots = ?per.iter().map(|l| l.1).collect::<Vec<_>>(),
                hits = ?per.iter().map(|l| l.2).collect::<Vec<_>>(),
                misses = ?per.iter().map(|l| l.3).collect::<Vec<_>>(),
                "expert cache per layer"
            );
            if let Some(path) = &self.split_misses {
                let stats = super::miss_split::MissStats::from_layer_stats(&per);
                if let Err(e) = stats.write(path) {
                    tracing::warn!(path = %path.display(), error = %e, "could not save the per-layer misses");
                }
            }
        }
        // Long prompts' routing counts too (their steps log none), as Strata's prefill saves
        // its profile.
        let routed = self
            .layers
            .iter()
            .flatten()
            .any(|l| l.counts.iter().any(|&c| c > 0));
        if let Some(path) = &self.profile
            && routed
        {
            // Merge with what the file held: the routing seen before counts too.
            let mut learned = self.learned_profile();
            if let Ok(old) = Profile::read(path) {
                let mut seen = std::collections::HashSet::new();
                learned.ranked.retain(|p| seen.insert(*p));
                for p in old.ranked {
                    if seen.insert(p) {
                        learned.ranked.push(p);
                    }
                }
            }
            if let Err(e) = learned.write(path) {
                tracing::warn!(path = %path.display(), error = %e, "could not save the expert profile");
            }
        }
    }
}

/// Sets up the host's expert share ([`super::cpu_share`]) for the tiered layers on CUDA
/// devices, from their host-resident expert tensors; returns the threshold.
#[cfg(feature = "cuda")]
fn share_setup(
    cfg: &ResidencyConfig,
    me: &ExpertResidency,
    found: &[Option<(LayerPlace, [Tensor<'_>; 3])>],
) -> Result<u32, String> {
    use super::cpu_share::{DevicePlan, HostExperts, format_of, setup, threshold};
    // POWER9 (AC922) only: its host link is the one the share relieves; elsewhere a no-op.
    if !cfg!(all(target_arch = "powerpc64", target_endian = "little")) {
        tracing::warn!("--cpu-expert-share is POWER9 (ppc64le) only; ignored on this host");
        return Ok(0);
    }
    let mut plans: Vec<DevicePlan> = Vec::new();
    for (il, f) in found.iter().enumerate() {
        let Some((p, ts)) = f else { continue };
        if me.layers.get(il).is_none_or(Option::is_none) {
            continue;
        }
        let Some(device) = p.cuda else { continue };
        let (Some(up_gate), Some(down)) = (format_of(ts[1].ty()), format_of(ts[2].ty())) else {
            tracing::warn!(
                layer = il,
                "cpu expert share: unsupported expert type, layer left to the GPU"
            );
            continue;
        };
        if ts[0].ty() != ts[1].ty() {
            continue;
        }
        // SAFETY: reading the data pointers of the live host expert tensors.
        let base = ts.map(|t| unsafe { (*t.raw()).data as usize });
        let he = HostExperts {
            base,
            expert_bytes: [ts[0].nb()[2], ts[1].nb()[2], ts[2].nb()[2]],
            up_gate,
            down,
            n_embd: ts[1].ne()[0] as usize,
            n_ff: ts[1].ne()[1] as usize,
            n_expert: ts[1].ne()[2] as usize,
        };
        let plan = match plans.iter_mut().find(|d| d.device == device) {
            Some(d) => d,
            None => {
                let node = super::cuda::caps(device).map_or(0, |c| c.host_numa_id.max(0) as u32);
                plans.push(DevicePlan {
                    device,
                    node,
                    layers: Vec::new(),
                    n_embd: he.n_embd,
                });
                plans.last_mut().unwrap()
            }
        };
        if plan.layers.len() <= il {
            plan.layers.resize(il + 1, None);
        }
        plan.layers[il] = Some(he);
    }
    setup(&cfg.cpu_share, plans)?;
    Ok(threshold(cfg.cpu_share.share))
}

/// Why [`tiered_tensor`] failed: `no_vram` when a CUDA layer's slots found no VRAM (the cache
/// then shrinks to fit, #1208); any other failure ends the load.
struct TierError {
    msg: String,
    no_vram: bool,
}

impl From<String> for TierError {
    fn from(msg: String) -> Self {
        Self {
            msg,
            no_vram: false,
        }
    }
}

/// The tiered tensor of `host` (an expert tensor) for a layer placed at `p`.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn tiered_tensor(
    p: &LayerPlace,
    host: Tensor<'_>,
    n_slots: usize,
    keep: Option<usize>,
    name: &str,
    n_expert: usize,
    tail: Option<(Tensor<'_>, usize)>,
) -> Result<Context, TierError> {
    #[cfg(not(feature = "cuda"))]
    let _ = tail;
    match p.cuda {
        #[cfg(feature = "cuda")]
        Some(dev) => match tail {
            // The bounded host tier: RAM slots, then the staging buffer's places.
            Some((tail, n_host)) => {
                super::cuda::tiered_tail(dev, host, tail, n_host, n_slots, keep, name)
            }
            None => super::cuda::tiered(dev, host, n_slots, keep, name),
        }
        .map_err(|e| TierError {
            msg: e.to_string(),
            no_vram: e.no_vram,
        }),
        #[cfg(not(feature = "cuda"))]
        Some(_) => Err("tiered experts on a GPU need a CUDA build"
            .to_string()
            .into()),
        None => host_copy(host, n_slots, name, n_expert, p.buft).map_err(|e| e.to_string().into()),
    }
}

/// The host-copy backing: `[n_slots | the experts]` in one buffer of `buft`.
fn host_copy(
    host: Tensor<'_>,
    n_slots: usize,
    name: &str,
    n_expert: usize,
    buft: BufferType,
) -> Result<Context, GgmlError> {
    let ne = host.ne();
    let ctx = Context::new(1, 0);
    let t = ctx
        .new_tensor(host.ty(), &[ne[0], ne[1], (n_slots + n_expert) as i64])
        .named(name);
    ctx.alloc_tensors(buft)?;
    let bytes = host.bytes();
    t.set_bytes_at(n_slots * host.nb()[2], &bytes);
    Ok(ctx)
}

/// Bandwidth of `device` reading a host arena on `node` (filled by the node's CPUs): a ggml
/// `get_rows` of random 3 MiB rows (an expert's size) read in place by the GPU, and DMA copies
/// of the same rows.
#[cfg(feature = "cuda")]
pub fn probe_device(device: usize, node: u32) -> Option<super::cuda::Probe> {
    use crate::hw::numa::{Topology, run_on_node};
    const ROW: usize = 3 << 20;
    const ROWS: usize = 170; // ~510 MiB
    const PICK: usize = 64;
    let buft = super::cuda::host_arena(node, &[device]).ok()?;
    let weights = Context::new(1, 0);
    let t = weights.new_tensor(types::I32, &[(ROW / 4) as i64, ROWS as i64]);
    weights.alloc_tensors(buft).ok()?;
    // SAFETY: reading the data pointer of the tensor just allocated.
    let addr = unsafe { (*t.raw()).data as usize };
    let fill = move || {
        // SAFETY: the arena tensor is ROW x ROWS bytes of host-writable memory.
        unsafe { std::ptr::write_bytes(addr as *mut u8, 0x5a, ROW * ROWS) }
    };
    match Topology::detect() {
        Ok(topo)
            if topo
                .nodes
                .iter()
                .any(|n| n.id == node && !n.cpus.is_empty()) =>
        {
            run_on_node(&topo, node, fill)
        }
        _ => fill(),
    }
    let mut backend = crate::ggml::backend_by_name(&format!("CUDA{device}")).ok()?;
    let ctx = Context::new(8, 8);
    let ids = ctx.new_tensor(types::I32, &[PICK as i64]).as_input();
    let out = ctx.get_rows(t, ids).as_output();
    let mut graph = ctx.new_graph(8);
    graph.expand(out);
    ctx.alloc_graph(
        &graph,
        crate::ggml::ComputeAllocator::new(backend.buffer_type()),
    )
    .ok()?;
    let picks: Vec<i32> = (0..PICK).map(|i| ((i * 37 + 11) % ROWS) as i32).collect();
    ids.set_i32(&picks);
    backend.compute(&graph).ok()?;
    let runs = 4;
    let start = std::time::Instant::now();
    for _ in 0..runs {
        backend.compute(&graph).ok()?;
    }
    let in_place = (runs * PICK * ROW) as f64 / start.elapsed().as_secs_f64() / 1e9;
    // SAFETY: reading the data pointer of the graph's output, a VRAM range of PICK rows.
    let vram = unsafe { (*out.raw()).data as usize };
    let copy = super::cuda::probe_copy(device, addr, ROW * ROWS, vram, ROW).ok()?;
    Some(super::cuda::Probe {
        in_place_gbs: in_place,
        copy_gbs: copy,
    })
}

/// The `--glm-prefetch` staging ring of device `d` ([`ResidencyConfig::ep_prefetch`]): slots
/// of its largest layer's expert (gate, up and down), FFN inputs of the up projection's width.
/// `None` off, on the CPU or without expert parallelism.
fn prefetch_ring(
    cfg: &ResidencyConfig,
    found: &[Option<(LayerPlace, [Tensor<'_>; 3])>],
    dev_of: &[usize],
    d: usize,
    cuda: Option<usize>,
    n_expert: usize,
) -> Option<super::ep_prefetch::RingLayout> {
    if cfg.ep_prefetch == 0 || cfg.ep.is_none() || cuda.is_none() {
        return None;
    }
    let mine = found
        .iter()
        .enumerate()
        .filter(|&(il, _)| dev_of[il] == d)
        .filter_map(|(_, f)| f.as_ref().map(|(_, ts)| ts));
    let (slot, k) = mine.fold((0, 0), |(s, k), ts| {
        let bytes: usize = ts.iter().map(|t| t.nb()[2]).sum();
        (s.max(bytes), k.max(ts[0].ne()[0] as usize))
    });
    (slot > 0).then(|| super::ep_prefetch::RingLayout::new(cfg.ep_prefetch, slot, k, n_expert))
}

/// Per kind, the staging buffer size a device needs (its largest layer's host part plus a
/// padded row), if `cfg` stages on that (CUDA) device.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables, dead_code))]
fn staging_bytes(
    cfg: &ResidencyConfig,
    cuda: Option<usize>,
    found: &[Option<(LayerPlace, [Tensor<'_>; 3])>],
    dev_of: &[usize],
    d: usize,
) -> Option<[usize; 3]> {
    if !cfg.staging || cuda.is_none() || !cfg!(feature = "cuda") {
        return None;
    }
    let mut bytes = [0usize; 3];
    for (il, f) in found.iter().enumerate() {
        let Some((_, ts)) = f else { continue };
        if dev_of[il] != d {
            continue;
        }
        for k in 0..3 {
            // One more expert's worth covers the matmuls' reads past the last row.
            bytes[k] = bytes[k].max(ts[k].nbytes() + ts[k].nb()[2]);
        }
    }
    (bytes.iter().sum::<usize>() > 0).then_some(bytes)
}

/// Each of a device's staging buffers' bytes (#1216), for [`plan_staging_sized`]: buffer `i`
/// is kind `i / ring`'s buffer `i % ring`, which only rows `row % ring == i % ring` use
/// ([`super::staging::Staging::buffer`]), so it takes the biggest of those rows' layers
/// (`rows`: per staged layer, its row and its bytes per kind, as [`staging_bytes`] counts
/// them), never more than `bytes`. A buffer no row uses keeps `bytes`. With one layer much
/// bigger than the rest (a GLM layer of higher quants on an AC922 with 3 GPUs: 5 GiB against
/// 4.2), the other buffers no longer take its size.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn ring_bytes(
    bytes: [usize; 3],
    ring: usize,
    rows: &[(usize, [usize; 3])],
) -> Vec<usize> {
    let mut at: Vec<Option<[usize; 3]>> = vec![None; ring.max(1)];
    for &(row, b) in rows {
        let n = at.len();
        let a = at[row % n].get_or_insert([0; 3]);
        for (x, y) in a.iter_mut().zip(b) {
            *x = (*x).max(y);
        }
    }
    (0..3 * at.len())
        .map(|i| {
            let k = i / at.len();
            at[i % at.len()].map_or(bytes[k], |a| a[k].min(bytes[k]))
        })
        .collect()
}

/// VRAM a device's staging buffers take (`ring` per kind, each rounded to the 2 MiB granule).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn stage_footprint(bytes: [usize; 3], ring: usize) -> u64 {
    ring as u64 * bytes.iter().map(|&b| b as u64 + (2 << 20)).sum::<u64>()
}

/// The slot thread's loop ([`ExpertResidency::restore`]): `refill` gives each layer of `work`
/// its slots back, in order, until `stop` is set. Layers not reached are left out of the
/// result: they stay without VRAM and keep their held experts for the next restore.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn refill_until<W>(
    work: Vec<(usize, W)>,
    stop: &AtomicBool,
    mut refill: impl FnMut(W) -> Result<(), String>,
) -> Vec<(usize, Result<(), String>)> {
    let mut done = Vec::with_capacity(work.len());
    for (il, w) in work {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        done.push((il, refill(w)));
    }
    done
}

/// The work of [`refill_per_device`]: per device, its layers (in their order) with their work.
type DeviceWork<W> = Vec<(usize, Vec<(usize, W)>)>;

/// Groups `(layer, (device, work))` by device, keeping the layers' order within each.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn by_device<W>(work: Vec<(usize, (usize, W))>) -> DeviceWork<W> {
    let mut groups: DeviceWork<W> = Vec::new();
    for (il, (device, w)) in work {
        match groups.iter_mut().find(|(d, _)| *d == device) {
            Some((_, g)) => g.push((il, w)),
            None => groups.push((device, vec![(il, w)])),
        }
    }
    groups
}

/// Takes `device`'s entry out of `list`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn take_device<T>(list: &std::sync::Mutex<Vec<(usize, T)>>, device: usize) -> Option<T> {
    let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
    let i = list.iter().position(|(d, _)| *d == device)?;
    Some(list.swap_remove(i).1)
}

/// Runs `refill` over every device's layers ([`by_device`]), each device on a thread of its own,
/// all at once (one device: on the caller's thread); the per-layer results of every device, in
/// the order of `groups`. A device whose thread panicked reports each of its layers failed.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn refill_per_device<W: Send>(
    groups: DeviceWork<W>,
    refill: impl Fn(usize, Vec<(usize, W)>) -> Vec<(usize, Result<(), String>)> + Sync,
) -> Vec<(usize, Result<(), String>)> {
    if groups.len() <= 1 {
        return groups
            .into_iter()
            .flat_map(|(device, work)| refill(device, work))
            .collect();
    }
    let refill = &refill;
    std::thread::scope(|s| {
        let threads: Vec<_> = groups
            .into_iter()
            .map(|(device, work)| {
                let layers: Vec<usize> = work.iter().map(|(il, _)| *il).collect();
                let thread = std::thread::Builder::new()
                    .name(format!("legion-slots{device}"))
                    .spawn_scoped(s, move || refill(device, work))
                    .map_err(|e| format!("cannot start the slot thread: {e}"));
                (layers, thread)
            })
            .collect();
        threads
            .into_iter()
            .flat_map(|(layers, thread)| {
                let failed = |e: String| layers.iter().map(|&il| (il, Err(e.clone()))).collect();
                match thread {
                    Ok(t) => t
                        .join()
                        .unwrap_or_else(|_| failed("the slot thread panicked".into())),
                    Err(e) => failed(e),
                }
            })
            .collect()
    })
}

/// Which of the lent slot allocations (by size) a staging buffer of `need` bytes maps
/// ([`super::cuda::DeviceMem::from_parts`]): the subset with the largest total that is not over
/// `need`, so the fewest bytes are allocated new and none are wasted; indices ascending. Exact
/// (a subset sum over the sizes' common unit) up to `MAX_UNITS` units, greedy (largest first)
/// beyond.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn pick_parts(sizes: &[usize], need: usize) -> Vec<usize> {
    const MAX_UNITS: usize = 1 << 16;
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let unit = sizes.iter().fold(0, |g, &s| gcd(g, s));
    if unit == 0 || need < unit {
        return Vec::new();
    }
    let cap = need / unit;
    if cap > MAX_UNITS {
        let mut order: Vec<usize> = (0..sizes.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(sizes[i]));
        let mut left = need;
        let mut picked: Vec<usize> = order
            .into_iter()
            .filter(|&i| {
                let fits = sizes[i] > 0 && sizes[i] <= left;
                if fits {
                    left -= sizes[i];
                }
                fits
            })
            .collect();
        picked.sort_unstable();
        return picked;
    }
    // reached[v]: the part that first reached a total of `v` units and the total before it
    // (every part counted once: totals are visited downwards, as in a 0/1 knapsack).
    let mut reached: Vec<Option<(usize, usize)>> = vec![None; cap + 1];
    let mut any = vec![false; cap + 1];
    any[0] = true;
    for (i, &s) in sizes.iter().enumerate() {
        let u = s / unit;
        if u == 0 || u > cap {
            continue;
        }
        for v in (u..=cap).rev() {
            if any[v - u] && !any[v] {
                any[v] = true;
                reached[v] = Some((i, v - u));
            }
        }
    }
    let mut v = (0..=cap).rev().find(|&v| any[v]).unwrap_or(0);
    let mut picked = Vec::new();
    while let Some((i, before)) = reached[v] {
        picked.push(i);
        v = before;
    }
    picked.sort_unstable();
    picked
}

/// The parts of `sizes` (allocation sizes) whose total is the smallest at least `need`
/// (all of them if they fall short): a staging buffer made of lent slots only
/// ([`super::staging::Staging::new`]'s `cover`), with as little left over as they allow.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // used by `staging` (CUDA only) and tests
pub(super) fn pick_cover(sizes: &[usize], need: usize) -> Vec<usize> {
    const MAX_UNITS: usize = 1 << 16;
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let total: usize = sizes.iter().sum();
    if total <= need {
        return (0..sizes.len()).filter(|&i| sizes[i] > 0).collect();
    }
    let unit = sizes.iter().fold(0, |g, &s| gcd(g, s));
    let want = need.div_ceil(unit);
    let cap = want + sizes.iter().max().copied().unwrap_or(0) / unit;
    if cap > MAX_UNITS {
        // Too fine to search: the most that fits, then the smallest part that covers the rest.
        let mut picked = pick_parts(sizes, need);
        let have: usize = picked.iter().map(|&i| sizes[i]).sum();
        let rest = (0..sizes.len())
            .filter(|i| !picked.contains(i) && sizes[*i] > 0)
            .filter(|&i| have + sizes[i] >= need)
            .min_by_key(|&i| sizes[i]);
        picked.extend(rest);
        picked.sort_unstable();
        return picked;
    }
    // As in `pick_parts`: reached[v] is the part that first reached `v` units and the total
    // before it; the smallest total reached from `want` up wins.
    let mut reached: Vec<Option<(usize, usize)>> = vec![None; cap + 1];
    let mut any = vec![false; cap + 1];
    any[0] = true;
    for (i, &s) in sizes.iter().enumerate() {
        let u = s / unit;
        if u == 0 || u > cap {
            continue;
        }
        for v in (u..=cap).rev() {
            if any[v - u] && !any[v] {
                any[v] = true;
                reached[v] = Some((i, v - u));
            }
        }
    }
    let Some(mut v) = (want..=cap).find(|&v| any[v]) else {
        return (0..sizes.len()).filter(|&i| sizes[i] > 0).collect();
    };
    let mut picked = Vec::new();
    while let Some((i, before)) = reached[v] {
        picked.push(i);
        v = before;
    }
    picked.sort_unstable();
    picked
}

/// Which lent allocations (`sizes`, in the order the lend unmapped them) each of a device's
/// staging buffers maps ([`super::staging::Staging::new`]): `ring` buffers of `bytes[k]` per
/// kind `k` (buffer `i` is kind `i / ring`'s), the biggest first, each of the parts
/// [`pick_cover`] (`cover`) or [`pick_parts`] picks from those still left. Per buffer, in that
/// order: its index and its parts' indices into `sizes`, in the order it takes them. What no
/// buffer takes is freed; a buffer its parts fall short of gets new VRAM for the rest.
pub(super) fn plan_staging(
    sizes: &[usize],
    bytes: [usize; 3],
    ring: usize,
    cover: bool,
) -> Vec<(usize, Vec<usize>)> {
    let need: Vec<usize> = (0..3 * ring).map(|i| bytes[i / ring]).collect();
    plan_staging_sized(sizes, &need, cover)
}

/// [`plan_staging`] for buffers of their own sizes: `need[i]` bytes for buffer `i` (kind
/// `i / ring`'s buffer `i % ring`, [`ring_bytes`]).
pub(super) fn plan_staging_sized(
    sizes: &[usize],
    need: &[usize],
    cover: bool,
) -> Vec<(usize, Vec<usize>)> {
    let mut order: Vec<usize> = (0..need.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(need[i]));
    let mut pool: Vec<usize> = (0..sizes.len()).collect();
    order
        .into_iter()
        .map(|i| {
            let need = need[i];
            let left: Vec<usize> = pool.iter().map(|&p| sizes[p]).collect();
            let mut picked = if cover {
                pick_cover(&left, need)
            } else {
                pick_parts(&left, need)
            };
            // Taken from the back, so the indices still to take stay where they are.
            picked.reverse();
            (i, picked.into_iter().map(|p| pool.swap_remove(p)).collect())
        })
        .collect()
}

/// Runs `run` over every device's work, each device on a thread named `{name}{device}` of its
/// own, all at once (one device: on the caller's thread); each device's result, in the order of
/// `work`. A device whose thread could not start or panicked gets an error.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn per_device_thread<T: Send, R: Send>(
    name: &str,
    work: Vec<(usize, T)>,
    run: impl Fn(usize, T) -> R + Sync,
) -> Vec<(usize, Result<R, String>)> {
    if work.len() <= 1 {
        return work
            .into_iter()
            .map(|(device, w)| (device, Ok(run(device, w))))
            .collect();
    }
    let run = &run;
    std::thread::scope(|s| {
        let threads: Vec<_> = work
            .into_iter()
            .map(|(device, w)| {
                let thread = std::thread::Builder::new()
                    .name(format!("{name}{device}"))
                    .spawn_scoped(s, move || run(device, w))
                    .map_err(|e| format!("cannot start the {name} thread: {e}"));
                (device, thread)
            })
            .collect();
        threads
            .into_iter()
            .map(|(device, thread)| {
                let joined = thread
                    .and_then(|t| t.join().map_err(|_| format!("the {name} thread panicked")));
                (device, joined)
            })
            .collect()
    })
}

/// The slot work of one device ([`ExpertResidency::restore`]): per layer, its slot ranges and
/// the copies `(dst, src, bytes)` that refill them.
#[cfg(feature = "cuda")]
type SlotWork = (Vec<super::cuda::Slots>, Vec<(usize, usize, usize)>);

/// The lend of one device ([`ExpertResidency::lend`]): `(CUDA index, each staging buffer's
/// bytes ([`ring_bytes`]), table rows)` if it stages, and its layers, each with whether it
/// stages (below the stage limit) and whether it keeps its slots
/// ([`ResidencyConfig::lend_keep`]).
#[cfg(feature = "cuda")]
struct LendWork<'a> {
    stage: Option<(usize, Vec<usize>, usize)>,
    layers: Vec<(bool, Lend, &'a mut Layer)>,
    /// The device's staging kept from the last lend ([`Device::kept`]).
    parked: Option<super::staging::Staging>,
    /// `(CUDA index, copies)` that re-pack the kept slots ([`LayerSlots::repack_kept`]),
    /// made before any slot is unmapped (some read slots about to be lent).
    repack: Option<(usize, Copies)>,
}

/// Device copies `(dst, src, bytes)` ([`Layer::repack_copies`]).
#[cfg(feature = "cuda")]
type Copies = Vec<(usize, usize, usize)>;

/// What a layer lends ([`ExpertResidency::lend`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lend {
    /// Nothing: whole layers' slots up to `lend_keep`.
    Keep,
    /// The slots past [`Layer::keep`] (`lend_spread`).
    Tail,
    /// All of its slots.
    Whole,
}

/// What a device's [`LendWork`] made: its staging buffers (none if it does not stage) and
/// whether they are kept for the next lend, and the milliseconds its unback and its staging
/// allocation took.
#[cfg(feature = "cuda")]
struct LendDone {
    staging: Result<(Option<super::staging::Staging>, bool), String>,
    unback_ms: f64,
    alloc_ms: f64,
    repack_ms: f64,
}

#[cfg(feature = "cuda")]
impl LendWork<'_> {
    /// Unmaps the layers' slots, then makes the staging buffers (over the slots' own VRAM
    /// where it fits, #439) and the staged ranges over the holes.
    ///
    /// `reuse` ([`ResidencyConfig::stage_reuse`]): buffers made of the lent slots' allocations
    /// only are kept for the next lend, with their staged ranges ([`Layer::ranges`]), and the
    /// slots map those allocations again in between ([`Layer::shared`]). A lend with kept
    /// buffers big enough only unmaps the slots, zeroes the buffers and wraps the ranges in
    /// tensors again. The same buffers and ranges, so the same outputs.
    fn lend(self, ring: usize, pieces: usize, reuse: bool) -> LendDone {
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let LendWork {
            stage,
            mut layers,
            mut parked,
            repack,
        } = self;
        let t = std::time::Instant::now();
        let repacked = match repack {
            Some((c, copies)) if !copies.is_empty() => copy_and_wait(c, &copies),
            _ => Ok(()),
        };
        let repack_ms = ms(t);
        if let Err(e) = repacked {
            return LendDone {
                staging: Err(format!("re-packing the kept slots: {e}")),
                unback_ms: 0.0,
                alloc_ms: 0.0,
                repack_ms,
            };
        }
        let reused = reuse
            && match (&parked, &stage) {
                (Some(st), Some((_, need, _))) => st.fits(need),
                _ => false,
            };
        let stages = stage.is_some();
        let t = std::time::Instant::now();
        // The slots' VRAM, kept for the staging buffers (freed at once without staging).
        let mut pool = Vec::new();
        let unback = layers
            .iter_mut()
            .filter(|(_, mode, _)| *mode != Lend::Keep)
            .try_for_each(|(_, mode, l)| {
                l.backed = false;
                l.tail_lent = *mode == Lend::Tail;
                (0..3).try_for_each(|k| {
                    if !reused {
                        // What the slots could map again of the buffers about to be made.
                        let tail = *mode == Lend::Tail;
                        l.shared[k] =
                            super::cuda::slot_handles(l.tensor(k), tail).filter(|_| reuse);
                    }
                    let mem = if *mode == Lend::Tail {
                        super::cuda::unback_tail(l.tensor(k))?
                    } else {
                        super::cuda::unback_keep(l.tensor(k))?
                    };
                    if stages && !reused {
                        pool.extend(mem);
                    }
                    Ok::<(), String>(())
                })
            });
        let unback_ms = ms(t);
        let mut alloc_ms = 0.0;
        let staging = unback.and_then(|()| {
            let Some((c, need, rows)) = stage else {
                return Ok((None, false));
            };
            let t = std::time::Instant::now();
            let (mut st, fresh, pooled) = if reused {
                let mut st = parked.take().expect("checked");
                st.reuse()?;
                (st, false, Default::default())
            } else {
                // Kept staging too small for this lend: it goes, with its ranges.
                for (_, _, l) in layers.iter_mut() {
                    l.ranges.clear();
                }
                drop(parked.take());
                let pooled: std::collections::HashSet<u64> =
                    pool.iter().flat_map(|m| m.handles()).map(|h| h.0).collect();
                let st = super::staging::Staging::new(c, &need, rows, ring, pool, reuse)?;
                (st, true, pooled)
            };
            alloc_ms = ms(t);
            for (_, mode, l) in layers.iter_mut().filter(|(stages, _, _)| *stages) {
                let n_ids = l.stage_first + l.stage_miss.len().max(1);
                let packed = !l.stage_ids.is_empty();
                for (k, kind) in KINDS.iter().enumerate() {
                    // A kept layer's range maps its slots again: cached experts read them.
                    let buf = st.buffer(k, l.row);
                    let ctx = if reuse && *mode != Lend::Keep {
                        if !l
                            .ranges
                            .get(k)
                            .is_some_and(|r| r.serves(l.tensor(k), packed))
                        {
                            let r = super::cuda::staged_range(c, l.tensor(k), buf, packed)?;
                            if k < l.ranges.len() {
                                l.ranges[k] = r;
                            } else {
                                l.ranges.push(r);
                            }
                        }
                        // The bounded host tier's tensor ends after its RAM slots and
                        // staging places: the staged ids go on to the last expert's.
                        let all = l.slots.n_slots() + l.slots.n_expert();
                        let n = packed.then_some(n_ids).or(l.tier.then_some(all));
                        l.ranges[k].view(l.tensor(k), n, kind)
                    } else if packed {
                        super::cuda::staged_packed(c, l.tensor(k), buf, n_ids, kind)
                    } else if *mode == Lend::Keep {
                        super::cuda::staged(c, l.tensor(k), buf, kind)
                    } else {
                        super::cuda::staged_over_hole(c, l.tensor(k), buf, kind)
                    };
                    let ctx = ctx.map_err(|e| e.to_string())?;
                    l.staged.push(ctx);
                }
                if fresh {
                    st.add_row(l.row, pieces)?;
                }
            }
            let kept = if fresh {
                // Kept only if every buffer is the slots' memory (a new allocation would hold
                // VRAM the refilled cache needs).
                let parts = st.handles();
                let new: usize = parts
                    .iter()
                    .filter(|h| !pooled.contains(&h.0))
                    .map(|h| h.1)
                    .sum();
                let kept = reuse && new == 0;
                let mine: std::collections::HashSet<u64> = parts.iter().map(|h| h.0).collect();
                let ours = |h: u64| h == 0 || mine.contains(&h);
                for (_, _, l) in layers.iter_mut() {
                    for s in &mut l.shared {
                        *s = s.filter(|&(v, t)| kept && ours(v) && ours(t));
                    }
                }
                if kept {
                    st.keep()?;
                }
                tracing::debug!(
                    device = c,
                    kept,
                    new_mib = new >> 20,
                    total_mib = parts.iter().map(|h| h.1).sum::<usize>() >> 20,
                    "expert cache staging made"
                );
                kept
            } else {
                true
            };
            tracing::info!(
                device = c,
                kept,
                reused,
                free_mib = super::cuda::free_vram(c).unwrap_or(0) >> 20,
                "expert cache lent"
            );
            Ok((Some(st), kept))
        });
        if staging.is_err() || !stages {
            for (_, _, l) in layers.iter_mut() {
                l.shared = [None; 3];
            }
        }
        LendDone {
            staging,
            unback_ms,
            alloc_ms,
            repack_ms,
        }
    }
}

/// Queues `copies` `(dst, src, bytes)` on a copy stream of CUDA device `device` and waits
/// for them.
#[cfg(feature = "cuda")]
fn copy_and_wait(device: usize, copies: &[(usize, usize, usize)]) -> Result<(), String> {
    let mut c = Copier::new(device)?;
    for &(dst, src, n) in copies {
        // SAFETY: the caller's ranges, inside one layer's tiered range: kept slots that
        // nothing reads until the lend is done, and slots or host experts not written meanwhile.
        unsafe { c.copy(dst, src, n)? };
    }
    c.sync()
}

/// What a lent device's refill frees first ([`ExpertResidency::restore`]): the staged ranges,
/// then the staging buffers they map.
#[cfg(feature = "cuda")]
struct Lent {
    staged: Vec<Context>,
    /// Kept ranges the staged views were made of, when the staging is not kept after all.
    ranges: Vec<super::cuda::StagedRange>,
    staging: Option<super::staging::Staging>,
    /// Kept staging to park instead ([`Device::kept`]).
    park: Option<super::staging::Parker>,
}

#[cfg(feature = "cuda")]
impl Lent {
    /// Frees it all; the milliseconds it took.
    fn free(self, device: usize) -> f64 {
        let t = std::time::Instant::now();
        if let Some(Err(e)) = self.park.map(|p| p.park()) {
            tracing::warn!(device, error = %e, "expert cache: parking the staging failed");
        }
        if let Err(e) = super::cuda::free_staged(device, self.staged) {
            tracing::warn!(device, error = %e, "expert cache: staged ranges freed one by one");
        }
        drop(self.ranges);
        drop(self.staging);
        t.elapsed().as_secs_f64() * 1e3
    }
}

/// Gives one device's layers their VRAM and experts back, in order, until `stop` ([`refill_until`]),
/// on a copy stream of the device's own; waits for the copies before it reports. `free_ms`: how
/// long freeing the device's staging took first (for the log).
#[cfg(feature = "cuda")]
fn refill_device(
    device: usize,
    work: Vec<(usize, SlotWork)>,
    free_ms: Option<f64>,
    stop: &AtomicBool,
) -> Vec<(usize, Result<(), String>)> {
    let t0 = std::time::Instant::now();
    let mut copier: Option<super::cuda::Copier> = None;
    let (mut map_ms, mut map_max_ms) = (0f64, 0f64);
    let mut done = refill_until(work, stop, |(slots, copies)| {
        let t = std::time::Instant::now();
        slots.iter().try_for_each(|s| s.reback())?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        map_ms += ms;
        map_max_ms = map_max_ms.max(ms);
        let c = match &mut copier {
            Some(c) => c,
            None => copier.insert(super::cuda::Copier::new(device)?),
        };
        for &(dst, src, n) in &copies {
            // SAFETY: a slot of the layer's new memory and an expert's host bytes; the table
            // does not point at the slot before the copy is done.
            unsafe { c.copy(dst, src, n)? };
        }
        Ok(())
    });
    let t_queued = t0.elapsed();
    if let Some(Err(e)) = copier.as_mut().map(|c| c.sync()) {
        for d in &mut done {
            d.1 = Err(e.clone());
        }
    }
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    tracing::debug!(
        device,
        layers = done.len(),
        free_ms = free_ms.unwrap_or(0.0),
        ms = ms(t0.elapsed()),
        map_ms,
        map_max_ms,
        copy_wait_ms = ms(t0.elapsed() - t_queued),
        "expert cache refill thread"
    );
    done
}

#[cfg(test)]
mod tests;
