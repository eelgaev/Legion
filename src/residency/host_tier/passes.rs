//! Staged prompt passes over the bounded host tier (#37).
//!
//! A long prompt lends the VRAM cache ([`crate::residency::experts::ExpertResidency::lend`])
//! and its big steps copy every expert a layer has in no VRAM slot into a VRAM staging buffer
//! before the layer's matmuls ([`crate::residency::staging`]). Without the tier each copy's
//! source is the expert's own host bytes; with it an expert is in one of three places:
//!
//! - a kept **VRAM slot**: read there, no copy;
//! - a **RAM slot**: copied from the slot by DMA;
//! - the **checkpoint**: read into a pinned staging place first, then copied from there.
//!
//! The checkpoint reads of a whole pass are known when it starts ([`HostTier::stage_pass`]):
//! a reader thread takes them in the order the pass runs its layers, in batches of at most
//! one set of places, and only stops while every set holds a batch the copy stream has not
//! copied out yet. So the reads run as many layers ahead of the matmuls as the pinned sets
//! hold, and the SSD is busy from the start of the pass. The copy stream calls back before
//! and after a batch's copies ([`PassHook`]): it waits for the batch's reads, and gives its
//! places back.
//!
//! The batches take the places as a ring ([`ring_places`], #37 round 31): each the places
//! right behind the bytes the batch before took, from the start again where a layer's last
//! place is passed, and the reader waits only for the batches whose bytes it overwrites. A
//! layer that reads more experts than a set holds then takes the places behind its first
//! batch's, not a whole set of its own: with two sets, such a layer let the reader run only
//! one layer ahead, and a 64 GB machine's SSD stood still for 0.7 s of a 4.5 s 8K pass.
//! The same experts reach the same VRAM, so the same outputs.
//!
//! The ring never makes the reader wait for more batches than the two sets would (round 32):
//! where layers of different expert sizes meet, a layer's places can end before the bytes
//! behind the batch before, and its first batch, back at the start, overwrote the batch just
//! read (on a 32 GB machine, 35 ms a pass the reader stood still where the two sets wait for
//! nothing). So a batch takes the first of these starts that waits for no more batches than
//! the sets would: right behind the batch before, the layer's first place, its last places,
//! or where its set would begin, and one that leaves the next batch such a start. A pass the
//! ring cannot plan that way takes the two sets. `LEGION_DEBUG_TIER_PASS_RING=0` keeps a set per batch, as before.
//!
//! A batch's reads of experts that follow one another in the checkpoint and in their places
//! go as one read of up to [`PASS_MERGE`] bytes (round 33): the drive gives a few percent more
//! for 4 MiB reads than for one 1 MiB expert matrix, and a 32 GB machine's pass is the SSD's
//! reads.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use super::{GAP, HostTier, Inner, To, fatal, plan_step};
use crate::residency::ssd_experts::io::{Class, Group, Job, merge_raw};

/// Where a staged pass copies an expert from: an index past the layer's VRAM slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassSrc {
    /// A RAM slot.
    Ram(usize),
    /// A staging place that read `batch` of the pass fills.
    Read { batch: usize, at: usize },
}

/// The copy stream's calls of a pass, in the stream's order: [`wait_arg`] before the copies
/// out of a batch's places, [`release_arg`] after them.
pub type PassHook = Arc<dyn Fn(usize) + Send + Sync>;

/// A routing-gated pass's message to its reader ([`HostTier::stage_pass`] with a [`PassGate`]):
/// the pass's layer `l` (its position in the pass) routes its tokens to the experts marked in
/// the first mask (per expert the layer copies, in the pass's order; empty: all of them), and
/// the copy stream already copies those of the second (ahead of the routing, from their RAM
/// slots). The reader reads only the routed ones from the checkpoint. Returns `(expert, slot)`
/// for those it reads into a RAM slot instead of a staging place ([`slot_learn`]): the copy
/// stream copies them from index `slot` past the VRAM slots (after their batch, as before).
pub type PassRoute = Arc<dyn Fn(usize, &[bool], &[bool]) -> Vec<(usize, usize)> + Send + Sync>;

/// A gated pass's RAM-slot learning for one layer (#37 round 42): its RAM slots the pass may
/// give up, `(expert, slot)` in the order a step's fetch evicts them
/// ([`crate::residency::ssd_experts::cache::SlotCache::evict_order`]), and the experts it
/// reads from the checkpoint, hottest first (positions in the layer's list).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Learn {
    pub il: usize,
    pub experts: Vec<u32>,
    pub victims: Vec<(usize, usize)>,
    pub incoming: Vec<usize>,
}

/// Which of a gated layer's checkpoint reads go into RAM slots ([`Learn`]): the routed ones
/// (`routed`; shorter: routed) that no reader took ahead into a place (`claimed`), hottest
/// first, each into the next slot whose expert no token routes to and the copy stream does not
/// copy ahead (`busy`). As a step's fetch, which reads a miss into the coldest slot the step
/// does not use, so the RAM slots follow what the prompts route to. `(expert, slot)`.
pub(crate) fn learn_pairs(
    learn: &Learn,
    routed: &[bool],
    busy: &[bool],
    claimed: &[bool],
) -> Vec<(usize, usize)> {
    let at = |v: &[bool], i: usize, d: bool| v.get(i).copied().unwrap_or(d);
    let victims = learn
        .victims
        .iter()
        .filter(|&&(i, _)| !at(routed, i, true) && !at(busy, i, false))
        .map(|&(_, s)| s);
    let incoming = learn
        .incoming
        .iter()
        .copied()
        .filter(|&i| at(routed, i, true) && !at(claimed, i, false));
    incoming.zip(victims).collect()
}

/// `LEGION_DEBUG_SLOT_LEARN=0`: a gated pass reads every routed expert without a RAM slot into
/// a staging place and gives up no RAM slot, and the bounded host tier's short prompts borrow
/// the cache only where [`crate::engine::lm::short_gate_ram`] holds, as before round 42.
pub fn slot_learn() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("LEGION_DEBUG_SLOT_LEARN").is_none_or(|v| v != "0"))
}

/// What a routing-gated pass ([`crate::residency::route_gate`]) gives its reader: per layer of
/// the pass, per expert it copies, its routing heat (which it reads ahead first), and how many
/// bytes per batch it may read ahead of the layer's router. A pass that would read less than
/// `min_read` bytes from the checkpoint is not gated: the SSD does not hold it up, and waiting
/// for each layer's routing would.
#[derive(Debug, Clone, PartialEq)]
pub struct PassGate {
    pub heat: Vec<Vec<f32>>,
    pub ahead: u64,
    pub min_read: u64,
}

/// How many of a gated batch's `places` experts (hottest first, `bytes` each) its reader reads
/// ahead of the layer's routing: the share the pass's batches before routed (`hit` of
/// `seen`, half before any), rounded up, at most `budget` bytes.
pub(crate) fn ahead_count(
    places: usize,
    seen: usize,
    hit: usize,
    bytes: u64,
    budget: u64,
) -> usize {
    let share = (places * (2 * hit + 1)).div_ceil(2 * seen + 2);
    let fit = usize::try_from(budget.checked_div(bytes).unwrap_or(0)).unwrap_or(usize::MAX);
    share.min(places).min(fit)
}

/// [`PassHook`]'s argument that waits for read `batch`.
pub fn wait_arg(batch: usize) -> usize {
    2 * batch
}

/// [`PassHook`]'s argument that gives read `batch`'s places back.
pub fn release_arg(batch: usize) -> usize {
    2 * batch + 1
}

/// Where a pass finds an expert it copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Held {
    /// In this RAM slot.
    Ram(usize),
    /// In the checkpoint; this RAM slot is free for it (it stays there after the pass).
    Load(usize),
    /// In the checkpoint, and no RAM slot is free: through a staging place.
    Stage,
}

/// One read of a pass: `(layer, expert, to)`, `layer` and `expert` as positions in the pass's
/// lists, into RAM slot or staging place `to`.
pub(crate) type Batch = Vec<(usize, usize, To)>;

/// A pass's sources and reads. `layers`: per layer in the pass's order, per expert it copies,
/// where it is. Those in the checkpoint are read in batches of one layer's experts, at most
/// `places` each, in order; batch `b` takes set `b % sets` (`places` places a set) for the
/// experts without a RAM slot.
pub(crate) fn plan_pass(
    layers: &[Vec<Held>],
    slots: usize,
    places: usize,
    sets: usize,
) -> Result<(Vec<Vec<PassSrc>>, Vec<Batch>), String> {
    let mut batches: Vec<Batch> = Vec::new();
    let mut srcs = Vec::with_capacity(layers.len());
    for (l, experts) in layers.iter().enumerate() {
        let mut open = false;
        // Places taken in the open batch.
        let mut taken = 0;
        let mut out = Vec::with_capacity(experts.len());
        for (i, held) in experts.iter().enumerate() {
            if let Held::Ram(s) = *held {
                out.push(PassSrc::Ram(s));
                continue;
            }
            if places == 0 || sets == 0 {
                return Err("the host expert tier has no staging places".into());
            }
            if !open || batches.last().is_none_or(|b| b.len() == places) {
                batches.push(Vec::new());
                open = true;
                taken = 0;
            }
            let b = batches.len() - 1;
            let (to, at) = match *held {
                Held::Load(s) => (To::Slot(s), s),
                _ => {
                    let place = (b % sets) * places + taken;
                    taken += 1;
                    (To::Place(place), slots + GAP + place)
                }
            };
            batches[b].push((l, i, to));
            out.push(PassSrc::Read { batch: b, at });
        }
        srcs.push(out);
    }
    Ok((srcs, batches))
}

/// Where a layer's staging places lie, for [`ring_places`]: its place indices past the gap
/// run from `first` up to `end` (both sets and the room between them, [`super::stage_layout`]),
/// and index `q`'s bytes of kind `k` start at `stage[k] + q * bytes[k]` (host addresses, or
/// any one origin for every layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Geo {
    pub first: usize,
    pub end: usize,
    pub stage: [usize; 3],
    pub bytes: [usize; 3],
}

impl Geo {
    /// The bytes of places `q..q + n`, per kind.
    fn span(&self, q: usize, n: usize) -> [(usize, usize); 3] {
        std::array::from_fn(|k| {
            let at = self.stage[k] + q * self.bytes[k];
            (at, at + n * self.bytes[k])
        })
    }

    /// The first place index whose bytes start at or past `ends` in every kind.
    fn after(&self, ends: &[usize; 3]) -> usize {
        (0..3)
            .map(|k| {
                ends[k]
                    .saturating_sub(self.stage[k])
                    .div_ceil(self.bytes[k].max(1))
            })
            .max()
            .unwrap_or(0)
            .max(self.first)
    }

    /// Where a batch of `n` places may start, in [`ring_places`]'s order: right behind the
    /// bytes that end at `ends` (the first place where that passes `end`), the first place, the
    /// last `n`, and the second set's start (`places` a set).
    fn starts(&self, n: usize, ends: &[usize; 3], places: usize) -> Vec<usize> {
        let mut ring = self.after(ends);
        if ring + n > self.end {
            ring = self.first;
        }
        let last = self.end.saturating_sub(n);
        let second = self.end.saturating_sub(places);
        let mut out = Vec::with_capacity(4);
        for q in [ring, self.first, last, second] {
            if q >= self.first && q + n <= self.end && !out.contains(&q) {
                out.push(q);
            }
        }
        out
    }
}

/// A batch's bytes per kind, in [`ring_places`].
type Span = [(usize, usize); 3];

/// The batches before one with bytes `s` that the reader must see given back: up to the last
/// of `spans` (one per batch, `None` for one without places) that `s` overlaps in any kind.
fn need_of(spans: &[Option<Span>], s: &Span) -> usize {
    let overlaps = |o: &Span| (0..3).any(|k| o[k].0 < s[k].1 && s[k].0 < o[k].1);
    spans
        .iter()
        .rposition(|o| o.as_ref().is_some_and(overlaps))
        .map_or(0, |b| b + 1)
}

/// [`ring_places`]' plan so far.
struct Ring<'a> {
    geo: &'a [Geo],
    /// Per batch, its layer and places (`None`: reads into RAM slots only).
    want: Vec<Option<(usize, usize)>>,
    places: usize,
    sets: usize,
    /// Per batch planned, its bytes.
    spans: Vec<Option<Span>>,
}

impl Ring<'_> {
    /// The start for batch `b` (one with places), the batches it waits for and its bytes,
    /// after batches `..b` ([`Self::spans`]) whose last ends at `ends`: the first of
    /// [`Geo::starts`] that waits for no more batches than the sets would (`b + 1 - sets`).
    /// `look`: only a start after which the next batch with places finds one too.
    ///
    /// The first, not the one that waits for the fewest: on a 64 GB machine that one made the
    /// reader wait 0.33 s a pass instead of 0.04, as it leaves the batches after it less room.
    fn pick(&mut self, b: usize, ends: &[usize; 3], look: bool) -> Option<(usize, usize, Span)> {
        let (l, n) = self.want[b]?;
        let g = self.geo[l];
        let most = (b + 1).saturating_sub(self.sets);
        let next = (b + 1..self.want.len()).find(|&j| self.want[j].is_some());
        for q in g.starts(n, ends, self.places) {
            let s = g.span(q, n);
            let need = need_of(&self.spans, &s);
            if need > most {
                continue;
            }
            if look && let Some(j) = next {
                let planned = self.spans.len();
                self.spans.push(Some(s));
                self.spans.resize(j, None);
                let room = self.pick(j, &s.map(|r| r.1), false).is_some();
                self.spans.truncate(planned);
                if !room {
                    continue;
                }
            }
            return Some((q, need, s));
        }
        None
    }
}

/// [`plan_pass`]'s batches over the places as a ring: per batch, in order, the place indices
/// (past the gap, of the batch's layer `geo[l]`) its experts read into ([`To::Place`] ones, in
/// their order), and the batches the reader must have seen given back before it reads this
/// one (`need[b]`: every batch before it whose bytes this one's overlap, in any kind).
///
/// A batch takes the first start of [`Geo::starts`] (right behind the batch before, where it
/// fits) that waits for no more batches than `sets` sets of `places` would (`b + 1 - sets`),
/// and where it can, one that leaves the next batch such a start. `None` where a batch finds
/// none: the pass takes the sets.
pub(crate) fn ring_places(
    batches: &[Batch],
    geo: &[Geo],
    places: usize,
    sets: usize,
) -> Result<Option<(Vec<Vec<usize>>, Vec<usize>)>, String> {
    let want = batches
        .iter()
        .map(|batch| {
            let n = batch.iter().filter(|r| matches!(r.2, To::Place(_))).count();
            let Some(&(l, _, _)) = batch.first().filter(|_| n > 0) else {
                // Only reads into RAM slots: no places.
                return Ok(None);
            };
            let g = geo.get(l).ok_or("a staged pass layer without its places")?;
            if n > g.end.saturating_sub(g.first) {
                return Err(
                    "a staged pass batch has more experts than its layer has places".into(),
                );
            }
            Ok(Some((l, n)))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut ring = Ring {
        geo,
        want,
        places,
        sets,
        spans: Vec::with_capacity(batches.len()),
    };
    let mut ends = [0usize; 3];
    let mut at = Vec::with_capacity(batches.len());
    let mut need = Vec::with_capacity(batches.len());
    for b in 0..batches.len() {
        let Some((_, n)) = ring.want[b] else {
            at.push(Vec::new());
            need.push(0);
            ring.spans.push(None);
            continue;
        };
        let Some((q, w, s)) = ring
            .pick(b, &ends, true)
            .or_else(|| ring.pick(b, &ends, false))
        else {
            return Ok(None);
        };
        ends = s.map(|r| r.1);
        ring.spans.push(Some(s));
        at.push((q..q + n).collect());
        need.push(w);
    }
    Ok(Some((at, need)))
}

/// Whether staged passes take their places as a ring ([`ring_places`]):
/// `LEGION_DEBUG_TIER_PASS_RING=0` gives every batch a set of its own, as before (A/B runs).
fn ring_mode() -> bool {
    static RING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RING.get_or_init(|| std::env::var("LEGION_DEBUG_TIER_PASS_RING").as_deref() != Ok("0"))
}

/// Bytes a staged pass's read takes at most: a batch's reads of experts that follow one
/// another in the checkpoint and in their places are joined ([`merge_raw`]) up to this.
const PASS_MERGE: u64 = 4 << 20;

/// [`PASS_MERGE`], or `LEGION_DEBUG_TIER_PASS_MERGE_KIB` (0: one read per expert matrix, as
/// before), for A/B runs.
fn pass_merge() -> u64 {
    static MERGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MERGE.get_or_init(|| {
        std::env::var("LEGION_DEBUG_TIER_PASS_MERGE_KIB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(PASS_MERGE, |kib| kib << 10)
    })
}

#[derive(Default)]
struct Progress {
    /// Batches read, and batches whose places the copy stream gave back.
    read: usize,
    released: usize,
    stop: bool,
    err: Option<String>,
    /// The reads' time, the reader's wait for a free set, the copy stream's wait for reads.
    read_ns: u64,
    set_wait_ns: u64,
    gate_ns: u64,
    /// A gated pass: per layer, the experts its tokens route to, once its staging wait read
    /// them ([`PassRoute`]).
    routed: Vec<Option<Vec<bool>>>,
    /// A gated pass's reads: experts and bytes read into places (ahead included), experts read
    /// ahead and those of them no token routed to, and the reader's wait for the routing.
    gated_experts: usize,
    gated_bytes: u64,
    ahead_experts: usize,
    ahead_unused: usize,
    route_wait_ns: u64,
    /// RAM-slot learning ([`Learn`]): per layer, the experts the reader took ahead into
    /// places before the routing came, and the `(expert, slot)` the routing sends into RAM
    /// slots; the `(layer, slot, expert)` read so far, for the slots' caches
    /// ([`Inner::settle`]).
    claimed: Vec<Vec<bool>>,
    to_slots: Vec<Vec<(usize, usize)>>,
    learned: Vec<(usize, usize, usize)>,
}

/// A gated pass's batch for its reader: its layer (position in the pass), the reads into RAM
/// slots (read whatever the routing), and per expert read into a place its position in the
/// layer's list and its reads, hottest first.
pub(super) struct GatedBatch {
    layer: usize,
    slots: Vec<Job>,
    places: Vec<(usize, [Job; 3])>,
    /// One expert's bytes.
    bytes: u64,
}

/// A pass's reads under way.
pub(super) struct Pass {
    progress: Mutex<Progress>,
    moved: Condvar,
    /// Per batch, the batches the copy stream must have given back before it is read: those
    /// whose places it overwrites, and every one before them.
    need: Vec<usize>,
    /// The batches took the places as a ring ([`ring_places`]), not a set each.
    ring: bool,
    batches: usize,
    layers: usize,
    ram_experts: usize,
    read_experts: usize,
    /// Reads the readers ran for them (after [`merge_raw`]).
    reads: usize,
    ram_bytes: u64,
    read_bytes: u64,
    started: Instant,
    /// Routing-gated ([`PassGate`]): the bytes a batch may read ahead of its layer's router.
    gated: Option<u64>,
    /// Per layer, its RAM-slot learning (empty: none, [`slot_learn`]).
    learn: Vec<Learn>,
}

impl Pass {
    /// [`PassRoute`]: layer `l` routes to the experts in `mask`, the copy stream copies those
    /// in `busy` ahead; the experts read into RAM slots.
    fn route(&self, l: usize, mask: &[bool], busy: &[bool]) -> Vec<(usize, usize)> {
        let mut p = self.progress.lock().unwrap();
        let p = &mut *p;
        let mut out = Vec::new();
        if let Some(r) = p.routed.get_mut(l)
            && r.is_none()
        {
            if let Some(learn) = self.learn.get(l) {
                let claimed = p.claimed.get(l).map_or(&[][..], |c| &c[..]);
                out = learn_pairs(learn, mask, busy, claimed);
                p.to_slots[l] = out.clone();
            }
            *r = Some(mask.to_vec());
        }
        self.moved.notify_all();
        out
    }

    /// Layer `l`'s routing if known; `Err` once the pass stops. Otherwise its experts at
    /// `ahead` (positions in its list) are the reader's to read into places
    /// ([`learn_pairs`]'s `claimed`).
    fn routing(&self, l: usize, ahead: &[usize]) -> Result<Option<Vec<bool>>, ()> {
        let mut p = self.progress.lock().unwrap();
        match p.routed.get(l) {
            Some(Some(m)) => Ok(Some(m.clone())),
            _ if p.stop => Err(()),
            _ => {
                if let Some(c) = p.claimed.get_mut(l) {
                    for &i in ahead {
                        if let Some(x) = c.get_mut(i) {
                            *x = true;
                        }
                    }
                }
                Ok(None)
            }
        }
    }

    /// Where the routing sends layer `l`'s reads into RAM slots: per position in its list,
    /// the slot (`n` positions).
    fn to_slots(&self, l: usize, n: usize) -> Vec<Option<usize>> {
        let p = self.progress.lock().unwrap();
        let mut out = vec![None; n];
        for &(i, s) in p.to_slots.get(l).into_iter().flatten() {
            if let Some(o) = out.get_mut(i) {
                *o = Some(s);
            }
        }
        out
    }

    /// Waits for the batches `b` overwrites to be given back; whether the pass stopped.
    fn wait_need(&self, b: usize) -> bool {
        let t0 = Instant::now();
        let mut p = self.progress.lock().unwrap();
        while !p.stop && p.released < self.need[b] {
            p = self.moved.wait(p).unwrap();
        }
        p.set_wait_ns += t0.elapsed().as_nanos() as u64;
        p.stop
    }

    /// Layer `l`'s routing, once its staging wait sends it ([`Self::route`]); `None` if the
    /// pass stops first. The wait's time is added to `ns`.
    fn wait_routing(&self, l: usize, ns: &mut u64) -> Option<Vec<bool>> {
        let t0 = Instant::now();
        let mut p = self.progress.lock().unwrap();
        while p.routed[l].is_none() && !p.stop {
            p = self.moved.wait(p).unwrap();
        }
        *ns += t0.elapsed().as_nanos() as u64;
        p.routed[l].clone()
    }

    /// The reader of a gated pass: batch by batch, the reads into RAM slots, then those into
    /// places of the experts the layer's tokens route to. While the layer's staging wait has
    /// not told it the routing yet ([`Self::route`]), the readers read the layer's hottest
    /// experts ahead (each its own group of [`Class::Ahead`], up to `ahead` bytes a batch);
    /// the routing then drops those not routed that no reader took yet. A pass given up reads
    /// only the RAM slots' experts.
    fn read_gated(&self, inner: &Inner, batches: Vec<GatedBatch>, ahead: u64) {
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        let merge = pass_merge();
        // The pass's batches so far: experts read into places and how many of them routed.
        let (mut seen, mut hit) = (0usize, 0usize);
        for (b, gb) in batches.into_iter().enumerate() {
            let stopped = self.wait_need(b);
            let t0 = Instant::now();
            let mut groups: Vec<Arc<Group>> = Vec::new();
            if !gb.slots.is_empty() {
                groups.push(inner.pool.submit(gb.slots));
            }
            // Per place read: its group if read ahead.
            let mut sent: Vec<Option<Arc<Group>>> = vec![None; gb.places.len()];
            let mut route_ns = 0u64;
            let mut mask = None;
            if !stopped && !gb.places.is_empty() {
                // As many of the hottest as the batches before routed (half at first): more
                // are mostly not routed, and would hold up the SSD when the routing comes.
                let n = ahead_count(gb.places.len(), seen, hit, gb.bytes, ahead);
                let claim: Vec<usize> = gb.places.iter().take(n).map(|p| p.0).collect();
                mask = match self.routing(gb.layer, &claim) {
                    Ok(Some(m)) => Some(m),
                    Err(()) => None,
                    Ok(None) => {
                        for (s, (_, reads)) in sent.iter_mut().zip(&gb.places).take(n) {
                            *s = Some(inner.pool.submit_as(reads.to_vec(), Class::Ahead));
                        }
                        self.wait_routing(gb.layer, &mut route_ns)
                    }
                };
            }
            if let Some(m) = &mask {
                seen += gb.places.len();
                hit += gb
                    .places
                    .iter()
                    .filter(|(pos, _)| m.get(*pos).copied().unwrap_or(true))
                    .count();
            }
            let routed = |m: &Option<Vec<bool>>, pos: usize| {
                m.as_ref()
                    .is_some_and(|m| m.get(pos).copied().unwrap_or(true))
            };
            let mut jobs = Vec::new();
            let mut rest = 0usize;
            // The routed experts the routing sent into RAM slots ([`Learn`]).
            let learn = self.learn.get(gb.layer).filter(|_| mask.is_some());
            let slot_of = learn.map(|l| self.to_slots(gb.layer, l.experts.len()));
            let mut learned = Vec::new();
            for ((pos, reads), s) in gb.places.iter().zip(&sent) {
                match (s, routed(&mask, *pos)) {
                    (None, true) => {
                        let to = slot_of
                            .as_ref()
                            .and_then(|v| v.get(*pos).copied().flatten());
                        match (learn, to) {
                            (Some(l), Some(slot)) => {
                                let e = l.experts[*pos] as usize;
                                jobs.extend(inner.reads(l.il, e, To::Slot(slot)));
                                learned.push((l.il, slot, e));
                            }
                            _ => jobs.extend(*reads),
                        }
                        rest += 1;
                    }
                    (Some(g), false) => g.cancel(),
                    _ => {}
                }
            }
            if !jobs.is_empty() {
                groups.push(inner.pool.submit(merge_raw(jobs, merge)));
            }
            let mut err = None;
            for g in groups.iter().chain(sent.iter().flatten()) {
                if let Err(e) = g.wait() {
                    err.get_or_insert(e);
                }
            }
            // Read ahead: those no cancel dropped a job of.
            let (mut ahead_n, mut unused) = (0usize, 0usize);
            for ((pos, _), s) in gb.places.iter().zip(&sent) {
                if s.as_ref().is_some_and(|g| !g.skipped()) {
                    ahead_n += 1;
                    unused += usize::from(!routed(&mask, *pos));
                }
            }
            let n = ahead_n + rest;
            let slots = learned.len();
            inner.counts.ram_loads.fetch_add(slots as u64, relaxed);
            inner
                .counts
                .stage_loads
                .fetch_add((n - slots) as u64, relaxed);
            let mut p = self.progress.lock().unwrap();
            // The reads ahead run while it waits for the routing.
            p.read_ns += t0.elapsed().as_nanos() as u64;
            p.gated_experts += n;
            p.gated_bytes += n as u64 * gb.bytes;
            p.ahead_experts += ahead_n;
            p.ahead_unused += unused;
            p.route_wait_ns += route_ns;
            match err {
                None => {
                    p.read = b + 1;
                    p.learned.append(&mut learned);
                }
                Some(e) => p.err = Some(e),
            }
            self.moved.notify_all();
            if p.err.is_some() {
                return;
            }
        }
    }

    fn call(&self, arg: usize) {
        let (batch, release) = (arg / 2, arg % 2 == 1);
        let mut p = self.progress.lock().unwrap();
        if release {
            p.released = p.released.max(batch + 1);
            self.moved.notify_all();
            return;
        }
        let t0 = Instant::now();
        while p.read <= batch && !p.stop && p.err.is_none() {
            p = self.moved.wait(p).unwrap();
        }
        p.gate_ns += t0.elapsed().as_nanos() as u64;
        if let Some(e) = &p.err {
            fatal(e);
        }
    }

    /// The reader: `batches` in order, each `(into RAM slots, into places)`. A pass given up
    /// ([`Self::stop`]) still reads what its RAM slots were promised.
    fn read(&self, inner: &Inner, batches: Vec<(Vec<Job>, Vec<Job>)>) {
        for (b, (mut jobs, places)) in batches.into_iter().enumerate() {
            let stopped = {
                let t0 = Instant::now();
                let mut p = self.progress.lock().unwrap();
                while !p.stop && p.released < self.need[b] {
                    p = self.moved.wait(p).unwrap();
                }
                p.set_wait_ns += t0.elapsed().as_nanos() as u64;
                p.stop
            };
            if !stopped {
                jobs.extend(places);
            }
            let t0 = Instant::now();
            let done = inner.pool.run(jobs);
            let mut p = self.progress.lock().unwrap();
            p.read_ns += t0.elapsed().as_nanos() as u64;
            match done {
                Ok(()) => p.read = b + 1,
                Err(e) => p.err = Some(e),
            }
            self.moved.notify_all();
            if p.err.is_some() {
                return;
            }
        }
    }

    fn stop(&self) {
        self.progress.lock().unwrap().stop = true;
        self.moved.notify_all();
    }

    fn log(&self) {
        let p = self.progress.lock().unwrap();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let s = |ns: u64| ns as f64 / 1e9;
        // A gated pass reads fewer than it planned.
        let read_bytes = match self.gated {
            Some(_) => p.gated_bytes,
            None => self.read_bytes,
        };
        tracing::info!(
            layers = self.layers,
            ram_experts = self.ram_experts,
            read_experts = self.read_experts,
            reads = self.reads,
            ram_gib = gib(self.ram_bytes),
            read_gib = gib(self.read_bytes),
            batches = self.batches,
            ring = self.ring,
            read = p.read,
            read_s = s(p.read_ns),
            read_gib_s = gib(read_bytes) / s(p.read_ns).max(1e-9),
            reader_waited_for_a_set_s = s(p.set_wait_ns),
            copies_waited_for_reads_s = s(p.gate_ns),
            pass_s = self.started.elapsed().as_secs_f64(),
            "host expert tier: staged pass"
        );
        if self.gated.is_some() {
            tracing::info!(
                planned_read_experts = self.read_experts,
                read_experts = p.gated_experts,
                read_gib = gib(p.gated_bytes),
                read_ahead = p.ahead_experts,
                read_ahead_unused = p.ahead_unused,
                read_into_ram_slots = p.learned.len(),
                reader_waited_for_routing_s = s(p.route_wait_ns),
                "host expert tier: routing-gated pass read what the prompt's tokens route to"
            );
        }
    }
}

/// The pass under way: its state and its reader.
pub(super) type Running = (Arc<Pass>, JoinHandle<()>);

impl Inner {
    /// Ends the staged pass under way, if any: its reader is done once the copy stream ran
    /// the pass's calls (a pass given up before that stops here). The staging places are
    /// free again. Returns the `(layer, slot, expert)` the pass read into RAM slots in place of
    /// what they held ([`Learn`]), for [`Self::settle_into`].
    pub(super) fn settle(&self) -> Vec<(usize, usize, usize)> {
        let Some((pass, reader)) = self.pass.lock().unwrap().take() else {
            return Vec::new();
        };
        pass.stop();
        if reader.join().is_err() {
            fatal("the staged pass reader panicked");
        }
        pass.log();
        std::mem::take(&mut pass.progress.lock().unwrap().learned)
    }

    /// [`Self::settle`], and the RAM slots' caches take what the pass read into them.
    pub(super) fn settle_into(&self, st: &mut super::State) {
        for (il, s, e) in self.settle() {
            st.caches[il].learn(e, s);
        }
    }
}

impl HostTier {
    /// Plans a staged pass and starts its reads. `layers`: `(layer, experts)` in the order
    /// the pass runs them, the experts each copies into its staging buffer (those in none of
    /// its VRAM slots). Returns every expert's source, in the same order, and the calls the
    /// copy stream makes around each read batch's copies. The copy stream must have run the
    /// pass before's calls, and nothing else may read or swap experts until this pass's
    /// copies are done: the RAM slots keep their experts meanwhile.
    ///
    /// With `gate` (a routing-gated pass, [`crate::residency::route_gate`]) the reader reads
    /// a layer's experts into staging places only once the returned [`PassRoute`] names the
    /// ones its tokens route to (reading the hottest ahead meanwhile); every layer of the pass
    /// must be routed that way (all experts, where its copies are not gated), or the copy
    /// stream's waits for its reads never end. Reads into RAM slots are never gated.
    ///
    /// A gated pass's RAM slots also learn what the prompt routes to (#37 round 42,
    /// [`slot_learn`]): the routed experts it reads from the checkpoint go into the RAM slots of
    /// the layer's experts no token routes to and the copy stream does not copy ahead (coldest
    /// first, as a step's fetch evicts), not into staging places, and stay there; the copy
    /// stream copies them from those slots ([`PassRoute`]). Only those slots change, which no
    /// copy of the pass reads; the slots' caches take them when the pass ends.
    pub fn stage_pass(
        &self,
        layers: &[(usize, Vec<u32>)],
        gate: Option<PassGate>,
    ) -> Result<(Vec<Vec<PassSrc>>, PassHook, Option<PassRoute>), String> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let st = &mut *st;
        inner.take_ahead(st, None)?;
        // The pass's reads take the SSD: refills not started are dropped (#37 round 43).
        inner.cancel_refills(st)?;
        // Before the pass marks its RAM slots used: the order they go out in ([`Learn`]).
        let learning = gate.is_some() && slot_learn();
        let evict: Vec<Vec<usize>> = layers
            .iter()
            .map(|(il, _)| match learning {
                true => st.caches[*il].evict_order(),
                false => Vec::new(),
            })
            .collect();
        // The experts in the checkpoint take the RAM slots the pass does not use (free ones,
        // then the least recently used, as a step's fetch fills them), and stay there.
        let held: Vec<Vec<Held>> = layers
            .iter()
            .map(|(il, experts)| {
                let need: Vec<usize> = experts.iter().map(|&e| e as usize).collect();
                let t = plan_step(&mut st.caches[*il], inner.n_expert, &need);
                need.iter()
                    .map(|&e| match t.index[e] {
                        s if s >= inner.slots => Held::Stage,
                        s if t.cached[e] => Held::Ram(s),
                        s => Held::Load(s),
                    })
                    .collect()
            })
            .collect();
        let (mut srcs, mut batches) = plan_pass(&held, inner.slots, inner.places, inner.sets)?;
        let first = inner.slots + GAP;
        let ring = if ring_mode() {
            let geo: Vec<Geo> = layers
                .iter()
                .map(|&(il, _)| {
                    let mem = &inner.layers[il];
                    Geo {
                        first: mem.first,
                        end: mem.second + inner.places,
                        stage: mem.stage,
                        bytes: mem.bytes,
                    }
                })
                .collect();
            ring_places(&batches, &geo, inner.places, inner.sets)?
        } else {
            None
        };
        let in_ring = ring.is_some();
        let need = if let Some((at, need)) = ring {
            for (b, (batch, at)) in batches.iter_mut().zip(at).enumerate() {
                let mut at = at.into_iter();
                for (l, i, to) in batch.iter_mut() {
                    if let To::Place(_) = to {
                        let q = at.next().ok_or("a staged pass ring lost a place")?;
                        *to = To::At(q);
                        srcs[*l][*i] = PassSrc::Read {
                            batch: b,
                            at: first + q,
                        };
                    }
                }
            }
            need
        } else {
            // A place's index is the layer's own ([`Inner::place_at`]).
            for ((il, _), s) in layers.iter().zip(&mut srcs) {
                for src in s.iter_mut() {
                    if let PassSrc::Read { at, .. } = src
                        && let Some(p) = at.checked_sub(first)
                    {
                        *at = first + inner.place_at(*il, p);
                    }
                }
            }
            // Batch `b` takes the set batch `b - sets` took.
            (0..batches.len())
                .map(|b| (b + 1).saturating_sub(inner.sets))
                .collect()
        };
        let per = |il: usize| inner.layers[il].bytes.iter().sum::<usize>() as u64;
        let (mut ram_experts, mut ram_bytes) = (0usize, 0u64);
        for ((il, _), s) in layers.iter().zip(&srcs) {
            let n = s.iter().filter(|s| matches!(s, PassSrc::Ram(_))).count();
            ram_experts += n;
            ram_bytes += n as u64 * per(*il);
        }
        let mut read_bytes = 0u64;
        let mut loads = 0u64;
        let merge = pass_merge();
        let jobs: Vec<(Vec<Job>, Vec<Job>)> = batches
            .iter()
            .map(|b| {
                let (mut slots, mut places) = (Vec::new(), Vec::new());
                for &(l, i, to) in b {
                    let (il, experts) = &layers[l];
                    read_bytes += per(*il);
                    let reads = inner.reads(*il, experts[i] as usize, to);
                    match to {
                        To::Slot(_) => {
                            loads += 1;
                            slots.extend(reads);
                        }
                        To::Place(_) | To::At(_) => places.extend(reads),
                    }
                }
                (merge_raw(slots, merge), merge_raw(places, merge))
            })
            .collect();
        let reads: usize = jobs.iter().map(|(s, p)| s.len() + p.len()).sum();
        let read_experts: usize = batches.iter().map(Vec::len).sum();
        let c = &inner.counts;
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        c.ram_loads.fetch_add(loads, relaxed);
        let gate = gate.filter(|g| read_bytes >= g.min_read);
        // A gated pass's reader counts the reads into places it ran.
        if gate.is_none() {
            c.stage_loads
                .fetch_add(read_experts as u64 - loads, relaxed);
        }
        // A gated pass's reader: per batch the reads into places expert by expert, hottest
        // first, to pick the routed ones (and merge them) as it reads.
        let gated: Option<Vec<GatedBatch>> = gate.as_ref().map(|g| {
            batches
                .iter()
                .map(|b| {
                    let layer = b.first().map_or(0, |r| r.0);
                    let il = layers[layer].0;
                    let heat = |i: usize| g.heat.get(layer).and_then(|h| h.get(i)).copied();
                    let mut slots = Vec::new();
                    let mut places: Vec<(usize, [Job; 3])> = Vec::new();
                    for &(l, i, to) in b {
                        let reads = inner.reads(layers[l].0, layers[l].1[i] as usize, to);
                        match to {
                            To::Slot(_) => slots.extend(reads),
                            To::Place(_) | To::At(_) => places.push((i, reads)),
                        }
                    }
                    let h = |p: &(usize, [Job; 3])| heat(p.0).unwrap_or(0.0);
                    places.sort_by(|a, b| h(b).total_cmp(&h(a)));
                    GatedBatch {
                        layer,
                        slots: merge_raw(slots, merge),
                        places,
                        bytes: per(il),
                    }
                })
                .collect()
        });
        // RAM-slot learning: per layer the RAM slots of the experts it copies, in the order
        // they go out, and its experts in the checkpoint, hottest first (the reader's order).
        let learn: Vec<Learn> = match &gate {
            Some(g) if learning => layers
                .iter()
                .enumerate()
                .map(|(l, (il, experts))| {
                    let mut pos_of = vec![None; inner.slots];
                    let mut incoming = Vec::new();
                    for (i, h) in held[l].iter().enumerate() {
                        match *h {
                            Held::Ram(s) => pos_of[s] = Some(i),
                            Held::Stage => incoming.push(i),
                            Held::Load(_) => {}
                        }
                    }
                    let heat = |i: usize| g.heat.get(l).and_then(|h| h.get(i)).copied();
                    let h = |i: usize| heat(i).unwrap_or(0.0);
                    incoming.sort_by(|&a, &b| h(b).total_cmp(&h(a)));
                    let victims = evict[l]
                        .iter()
                        .filter_map(|&s| Some((pos_of.get(s).copied().flatten()?, s)))
                        .collect();
                    Learn {
                        il: *il,
                        experts: experts.clone(),
                        victims,
                        incoming,
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let progress = Progress {
            routed: vec![None; if gate.is_some() { layers.len() } else { 0 }],
            claimed: learn.iter().map(|l| vec![false; l.experts.len()]).collect(),
            to_slots: vec![Vec::new(); learn.len()],
            ..Progress::default()
        };
        let pass = Arc::new(Pass {
            progress: Mutex::new(progress),
            moved: Condvar::new(),
            need,
            ring: in_ring,
            batches: batches.len(),
            layers: layers.len(),
            ram_experts,
            read_experts,
            reads,
            ram_bytes,
            read_bytes,
            started: Instant::now(),
            gated: gate.as_ref().map(|g| g.ahead),
            learn,
        });
        let (p, i) = (pass.clone(), inner.clone());
        let ahead = gate.as_ref().map_or(0, |g| g.ahead);
        let reader = std::thread::Builder::new()
            .name("legion-tier-pass".into())
            .spawn(move || match gated {
                Some(batches) => p.read_gated(&i, batches, ahead),
                None => p.read(&i, jobs),
            })
            .map_err(|e| format!("cannot start the staged pass reader: {e}"))?;
        *inner.pass.lock().unwrap() = Some((pass.clone(), reader));
        let route: Option<PassRoute> = gate.is_some().then(|| {
            let p = pass.clone();
            Arc::new(move |l: usize, mask: &[bool], busy: &[bool]| p.route(l, mask, busy))
                as PassRoute
        });
        let hook: PassHook = Arc::new(move |arg| pass.call(arg));
        Ok((srcs, hook, route))
    }

    /// The VRAM cache dropped `experts` of layer `il` without a swap (a lend unmapped their
    /// slots): they are in the checkpoint only, unless a RAM slot holds them.
    pub fn dropped(&self, il: usize, experts: &[u32]) {
        let mut st = self.inner.state.lock().unwrap();
        for &e in experts {
            st.in_vram[il][e as usize] = false;
        }
    }

    /// Per expert of layer `il`: whether a RAM slot holds it (a staged pass done by now has
    /// ended, [`Inner::settle_into`]).
    pub fn in_ram(&self, il: usize) -> Vec<bool> {
        let mut st = self.inner.state.lock().unwrap();
        self.inner.settle_into(&mut st);
        let cache = &st.caches[il];
        (0..self.inner.n_expert)
            .map(|e| cache.slot(e).is_some())
            .collect()
    }
}
