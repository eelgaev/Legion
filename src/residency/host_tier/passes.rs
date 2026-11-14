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
//! set back for the batch that many sets later.

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
    sets: usize,
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
                while !p.stop && b >= p.released + self.sets {
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
        let (srcs, batches) = plan_pass(&held, inner.slots, inner.places, inner.sets)?;
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
                        To::Place(_) => places.extend(reads),
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
            sets: inner.sets,
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
