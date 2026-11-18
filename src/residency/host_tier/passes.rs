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
//! `LEGION_DEBUG_TIER_PASS_RING=0` keeps a set per batch, as before.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use super::{GAP, HostTier, Inner, To, fatal, plan_step};
use crate::residency::ssd_experts::io::Job;

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
}

/// [`plan_pass`]'s batches over the places as a ring: per batch, in order, the place indices
/// (past the gap, of the batch's layer `geo[l]`) its experts read into ([`To::Place`] ones, in
/// their order), and the batches the reader must have seen given back before it reads this
/// one (`need[b]`: every batch before it whose bytes this one's overlap, in any kind). A batch
/// takes the places right behind the bytes the batch before took, or its layer's first ones
/// when they would pass `end`.
pub(crate) fn ring_places(
    batches: &[Batch],
    geo: &[Geo],
) -> Result<(Vec<Vec<usize>>, Vec<usize>), String> {
    let mut ends = [0usize; 3];
    let mut spans: Vec<Option<[(usize, usize); 3]>> = Vec::with_capacity(batches.len());
    let mut at = Vec::with_capacity(batches.len());
    let mut need = Vec::with_capacity(batches.len());
    for batch in batches {
        let n = batch.iter().filter(|r| matches!(r.2, To::Place(_))).count();
        let Some(&(l, _, _)) = batch.first().filter(|_| n > 0) else {
            // Only reads into RAM slots: no places.
            at.push(Vec::new());
            spans.push(None);
            need.push(0);
            continue;
        };
        let g = geo.get(l).ok_or("a staged pass layer without its places")?;
        if n > g.end.saturating_sub(g.first) {
            return Err("a staged pass batch has more experts than its layer has places".into());
        }
        let mut q = g.after(&ends);
        if q + n > g.end {
            q = g.first;
        }
        let s = g.span(q, n);
        ends = s.map(|r| r.1);
        let overlaps = |o: &[(usize, usize); 3]| (0..3).any(|k| o[k].0 < s[k].1 && s[k].0 < o[k].1);
        need.push(
            spans
                .iter()
                .rposition(|o| o.as_ref().is_some_and(overlaps))
                .map_or(0, |b| b + 1),
        );
        spans.push(Some(s));
        at.push((q..q + n).collect());
    }
    Ok((at, need))
}

/// Whether staged passes take their places as a ring ([`ring_places`]):
/// `LEGION_DEBUG_TIER_PASS_RING=0` gives every batch a set of its own, as before (A/B runs).
fn ring_mode() -> bool {
    static RING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RING.get_or_init(|| std::env::var("LEGION_DEBUG_TIER_PASS_RING").as_deref() != Ok("0"))
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
}

/// A pass's reads under way.
pub(super) struct Pass {
    progress: Mutex<Progress>,
    moved: Condvar,
    /// Per batch, the batches the copy stream must have given back before it is read: those
    /// whose places it overwrites, and every one before them.
    need: Vec<usize>,
    batches: usize,
    layers: usize,
    ram_experts: usize,
    read_experts: usize,
    ram_bytes: u64,
    read_bytes: u64,
    started: Instant,
}

impl Pass {
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
        tracing::info!(
            layers = self.layers,
            ram_experts = self.ram_experts,
            read_experts = self.read_experts,
            ram_gib = gib(self.ram_bytes),
            read_gib = gib(self.read_bytes),
            batches = self.batches,
            read = p.read,
            read_s = s(p.read_ns),
            read_gib_s = gib(self.read_bytes) / s(p.read_ns).max(1e-9),
            reader_waited_for_a_set_s = s(p.set_wait_ns),
            copies_waited_for_reads_s = s(p.gate_ns),
            pass_s = self.started.elapsed().as_secs_f64(),
            "host expert tier: staged pass"
        );
    }
}

/// The pass under way: its state and its reader.
pub(super) type Running = (Arc<Pass>, JoinHandle<()>);

impl Inner {
    /// Ends the staged pass under way, if any: its reader is done once the copy stream ran
    /// the pass's calls (a pass given up before that stops here). The staging places are
    /// free again.
    pub(super) fn settle(&self) {
        let Some((pass, reader)) = self.pass.lock().unwrap().take() else {
            return;
        };
        pass.stop();
        if reader.join().is_err() {
            fatal("the staged pass reader panicked");
        }
        pass.log();
    }
}

impl HostTier {
    /// Plans a staged pass and starts its reads. `layers`: `(layer, experts)` in the order
    /// the pass runs them, the experts each copies into its staging buffer (those in none of
    /// its VRAM slots). Returns every expert's source, in the same order, and the calls the
    /// copy stream makes around each read batch's copies. The copy stream must have run the
    /// pass before's calls, and nothing else may read or swap experts until this pass's
    /// copies are done: the RAM slots keep their experts meanwhile.
    pub fn stage_pass(
        &self,
        layers: &[(usize, Vec<u32>)],
    ) -> Result<(Vec<Vec<PassSrc>>, PassHook), String> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap();
        let st = &mut *st;
        inner.take_ahead(st, None)?;
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
        let need = if ring_mode() {
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
            let (at, need) = ring_places(&batches, &geo)?;
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
                (slots, places)
            })
            .collect();
        let read_experts: usize = batches.iter().map(Vec::len).sum();
        let c = &inner.counts;
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        c.ram_loads.fetch_add(loads, relaxed);
        c.stage_loads
            .fetch_add(read_experts as u64 - loads, relaxed);
        let pass = Arc::new(Pass {
            progress: Mutex::new(Progress::default()),
            moved: Condvar::new(),
            need,
            batches: batches.len(),
            layers: layers.len(),
            ram_experts,
            read_experts,
            ram_bytes,
            read_bytes,
            started: Instant::now(),
        });
        let (p, i) = (pass.clone(), inner.clone());
        let reader = std::thread::Builder::new()
            .name("legion-tier-pass".into())
            .spawn(move || p.read(&i, jobs))
            .map_err(|e| format!("cannot start the staged pass reader: {e}"))?;
        *inner.pass.lock().unwrap() = Some((pass.clone(), reader));
        let hook: PassHook = Arc::new(move |arg| pass.call(arg));
        Ok((srcs, hook))
    }

    /// The VRAM cache dropped `experts` of layer `il` without a swap (a lend unmapped their
    /// slots): they are in the checkpoint only, unless a RAM slot holds them.
    pub fn dropped(&self, il: usize, experts: &[u32]) {
        let mut st = self.inner.state.lock().unwrap();
        for &e in experts {
            st.in_vram[il][e as usize] = false;
        }
    }

    /// Per expert of layer `il`: whether a RAM slot holds it.
    pub fn in_ram(&self, il: usize) -> Vec<bool> {
        let st = self.inner.state.lock().unwrap();
        let cache = &st.caches[il];
        (0..self.inner.n_expert)
            .map(|e| cache.slot(e).is_some())
            .collect()
    }
}
