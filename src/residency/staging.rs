//! Prefill staging: a prompt chunk touches nearly every expert of every layer, and reading
//! host experts in place from a GEMM runs far below the link (the #43 measurements on a V100:
//! 235 ms per layer in place against 22 ms staged, at 8K tokens), so big steps first copy each
//! layer's uncached experts into VRAM by DMA, overlapped with the previous layer's compute.
//!
//! Per device, a ring of [`super::PREFILL_RING`] staging buffers per expert kind by default (`--prefill-ring`),
//! each the size of the device's largest layer's host part. A layer's *staged tensor*
//! ([`super::cuda::staged`]) maps the layer's VRAM slots again, followed by buffer
//! `row % ring`: the same ids that read an uncached expert from
//! host memory in the tiered tensor read its staged copy here, and cached experts stay in their
//! slots (only uncached experts are copied).
//!
//! Ordering is stream-ordered and host-sync free, through two tiny graph ops per layer (no
//! kernels: CUDA API calls on the compute stream when the graph runs):
//! - `wait` before the layer's expert matmuls: the compute stream waits for the layer's copy.
//! - `done` after them: records that the layer's buffer is free, makes the copy stream wait for
//!   that, and queues the copy of the layer `ring` ahead (same buffer).
//!
//! [`Staging::begin`] queues the first `ring` layers' copies before a pass starts;
//! [`Staging::chain`] instead plans the next pass behind the running one, so the `done` ops of
//! its last layers queue the next pass's first copies and the next pass's first layers need not
//! wait for a whole layer's copy (as Strata-AC922's prefetch runs into the next chunk). A deeper ring
//! lets the copy stream run further ahead of the compute, for layers whose copy takes longer
//! than the previous layer's matmuls; it takes more of the lent VRAM from the prompt's compute
//! buffers. Sizing the ring follows Strata-AC922's `--prefill-ring` (eelgaev/Strata-AC922,
//! a3dfde7, MIT), whose ring counts expert slots; here a slot is a whole layer's experts.
//!
//! Split waits (`--prefill-stage-pieces`): a plan with [marks](MARK) between its copies
//! records an event per piece, and the fused expert op ([`split_wait`]) computes the experts
//! the layer has cached in VRAM at once and each piece's experts once that piece is in, with no
//! `wait` op: as Strata-AC922's prefill waits per streamed expert (eelgaev/Strata-AC922,
//! `src/prefill/prefill.cpp`, MIT).
//!
//! Peer staging (`--prefill-peer-stage`): while a stage runs a prompt alone, the other GPU of
//! its NVLink pair is idle, and so is that GPU's own link to host memory. A [`PeerPuller`] on
//! it then takes every other piece of each row's copies: a kernel there reads them from host
//! memory and stores them into this device's buffer over the pair's NVLink, after what the
//! copy stream waited for (the buffer's `done`), and the copy stream waits for it before the
//! row's marks and its `copied` event. As Project Maya's two-GPU prefill staging
//! (mw00/project-maya, MIT). The same bytes in the same places.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::cuda::{Copier, DeviceMem, Event, Mapping, PeerPuller, zero_async};
use crate::ggml::{ExternalOp, ffi};

/// One planned copy: `(dst, src, bytes)` device-addressable addresses; `src == 0` zeroes
/// `bytes` at `dst` instead (the tail past a layer's last expert, which the quantized matmuls
/// read and the other buffer's layer may have filled with experts of another type).
pub(super) type Copy = (usize, usize, usize);

/// A plan mark instead of a copy: `(MARK, bound, CACHED_MARK)` after the zeroed pads, with
/// `bound` the end of the expert ids the layer reads from VRAM slots (nothing to wait for but
/// the pads); `(MARK, bound, PIECE_MARK)` after each piece's copies, its ids ending at `bound`
/// (each piece's start is the previous bound).
pub(super) const MARK: usize = usize::MAX;
pub(super) const CACHED_MARK: usize = 1;
pub(super) const PIECE_MARK: usize = 0;

/// A row's split waits, from the marks of its last queued plan.
#[derive(Default)]
struct Split {
    /// Whether the plan had marks.
    marked: bool,
    /// After the pads: the cached experts' compute waits for it.
    pre: Option<Event>,
    /// The cached experts' ids end.
    first: usize,
    /// Each piece's ids end, and the pieces' events (one per piece at most).
    cuts: Vec<usize>,
    evs: Vec<Event>,
}

struct Row {
    copied: Event,
    done: Event,
    split: Split,
}

struct Core {
    copier: Copier,
    rows: Vec<Option<Row>>,
    /// The rows of the current pass (and a chained one), in order, each with the copies that
    /// fill its buffer, and how many of them were queued.
    plan: Vec<(usize, Vec<Copy>)>,
    queued: usize,
    /// Where the pass [`Staging::chain`] planned starts in `plan`.
    ahead: Option<usize>,
    /// `done` ops that found no row left to queue: their buffers are free for a chained pass.
    missed: usize,
    /// The idle peer's copies ([`Staging::peer`]), and whether this pass uses it.
    peer: Option<PeerPuller>,
    help: bool,
    /// Since the last [`Staging::take_stats`].
    stats: StageStats,
}

/// What a device's staging queued since the last [`Staging::take_stats`] (#778): rows,
/// copies (and zeroed pads) queued, the host time queueing them took, and the time
/// [`Staging::begin`] and [`Staging::chain`] waited for the staging lock.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageStats {
    pub rows: u64,
    pub copies: u64,
    pub queue_ms: f64,
    pub lock_wait_ms: f64,
}

// SAFETY: CUDA handles, used under the mutex.
unsafe impl Send for Core {}

impl Core {
    /// Queues the copies of the next planned row (after the compute that last used its buffer).
    fn queue_next(&mut self) -> Result<(), String> {
        let Some((row, copies)) = self.plan.get(self.queued) else {
            self.missed += 1;
            return Ok(());
        };
        self.queued += 1;
        let t0 = std::time::Instant::now();
        self.stats.rows += 1;
        self.stats.copies += copies.iter().filter(|c| c.0 != MARK).count() as u64;
        let r = self.rows[*row].as_mut().expect("planned rows stage");
        let s = &mut r.split;
        s.marked = false;
        s.cuts.clear();
        // With the peer: it copies a piece whenever it has queued fewer bytes than the copy
        // stream, after what the copy stream waited for so far.
        let mut peer = self.peer.as_mut().filter(|_| self.help);
        if let Some(p) = peer.as_mut() {
            p.follow(&mut self.copier)?;
        }
        let (mut own, mut theirs, mut pending) = (0usize, 0usize, false);
        for &(dst, src, n) in copies {
            if dst == MARK {
                if let (Some(p), true) = (peer.as_mut(), pending) {
                    p.join(&mut self.copier)?;
                    pending = false;
                }
                s.marked = true;
                // The events come from `add_row`: made here, on whichever thread queues, they
                // would switch its current device.
                let ev = if n == CACHED_MARK {
                    s.first = src;
                    s.pre.as_ref().ok_or("staging marks without events")?
                } else {
                    s.cuts.push(src);
                    s.evs
                        .get(s.cuts.len() - 1)
                        .ok_or("more staging pieces than events")?
                };
                self.copier.record(ev)?;
                continue;
            }
            // SAFETY: inside the row's staged range and the arena; the buffer is free (the
            // copy stream waited for the compute that read it last).
            unsafe {
                if src == 0 {
                    self.copier.zero(dst, n)?;
                } else if let Some(p) = peer.as_mut().filter(|_| theirs < own) {
                    p.pull(dst, src, n)?;
                    theirs += n;
                    pending = true;
                } else {
                    self.copier.copy(dst, src, n)?;
                    own += n;
                }
            }
        }
        if let (Some(p), true) = (peer, pending) {
            p.join(&mut self.copier)?;
        }
        let recorded = self.copier.record(&r.copied);
        self.stats.queue_ms += t0.elapsed().as_secs_f64() * 1e3;
        recorded
    }
}

/// A device's staging buffers and the state of the current pass.
pub struct Staging {
    id: usize,
    device: usize,
    core: Arc<Mutex<Core>>,
    /// Mappings of the buffers while they are kept across lends ([`Self::keep`]), to zero them
    /// again.
    kept: Vec<Mapping>,
    /// Per kind, the ring's buffers (the staged ranges map them).
    bufs: Vec<Vec<DeviceMem>>,
}

/// Parks a kept [`Staging`] from another thread ([`Staging::parker`]).
pub struct Parker(Arc<Mutex<Core>>);

impl Parker {
    /// Waits for the staging copies and forgets the passes: the buffers' memory may serve the
    /// cache slots until the next lend ([`Staging::reuse`]).
    pub fn park(&self) -> Result<(), String> {
        let mut c = self.0.lock().unwrap();
        c.copier.sync()?;
        c.plan.clear();
        c.queued = 0;
        c.ahead = None;
        c.missed = 0;
        Ok(())
    }
}

static REGISTRY: Mutex<Vec<(usize, Arc<Mutex<Core>>)>> = Mutex::new(Vec::new());
static NEXT_ID: Mutex<usize> = Mutex::new(1);

impl Drop for Staging {
    fn drop(&mut self) {
        REGISTRY.lock().unwrap().retain(|(id, _)| *id != self.id);
        if let Ok(mut c) = self.core.lock() {
            let _ = c.copier.sync();
        }
    }
}

impl Staging {
    /// Buffers of `bytes[k]` per kind (`ring` each, at least two) on `device`, for `rows` rows.
    ///
    /// The buffers map the allocations of `pool` where they fit (the lent slots' VRAM,
    /// [`super::cuda::unback_keep`]; [`super::experts::pick_parts`] chooses), the biggest
    /// buffers first, and new VRAM for the rest; what is left of `pool` is freed. `cover`:
    /// allocations of `pool` for the rest too, the fewest bytes over that cover it
    /// ([`super::experts::pick_cover`], a buffer may come out bigger), so that buffers made of
    /// the pool only can be kept ([`Self::keep`]).
    pub fn new(
        device: usize,
        bytes: [usize; 3],
        rows: usize,
        ring: usize,
        pool: Vec<DeviceMem>,
        cover: bool,
    ) -> Result<Self, String> {
        let ring = ring.max(super::PREFILL_RING);
        let sizes: Vec<usize> = pool.iter().map(DeviceMem::size).collect();
        let plan = super::experts::plan_staging(&sizes, bytes, ring, cover);
        let mut pool: Vec<Option<DeviceMem>> = pool.into_iter().map(Some).collect();
        let mut made: Vec<Option<DeviceMem>> = (0..3 * ring).map(|_| None).collect();
        for (i, picked) in plan {
            let parts = picked
                .into_iter()
                .map(|p| pool[p].take().expect("each part once"))
                .collect();
            made[i] = Some(DeviceMem::from_parts(device, bytes[i / ring], parts)?);
        }
        drop(pool);
        let mut made = made.into_iter().map(|m| m.expect("every buffer made"));
        let bufs = (0..3).map(|_| made.by_ref().take(ring).collect()).collect();
        let peer = super::cuda::stage_peer(device)
            .map(|p| PeerPuller::new(device, p))
            .transpose()?;
        let core = Arc::new(Mutex::new(Core {
            copier: Copier::new(device)?,
            rows: (0..rows).map(|_| None).collect(),
            plan: Vec::new(),
            queued: 0,
            ahead: None,
            missed: 0,
            peer,
            help: false,
            stats: StageStats::default(),
        }));
        let id = {
            let mut n = NEXT_ID.lock().unwrap();
            *n += 1;
            *n
        };
        REGISTRY.lock().unwrap().push((id, core.clone()));
        Ok(Self {
            id,
            device,
            core,
            kept: Vec::new(),
            bufs,
        })
    }

    /// The GPU that takes half of the copies of the passes [`Self::begin`] starts with `help`:
    /// the device's staging peer ([`super::cuda::set_stage_peer`]) when it had one as the
    /// buffers were made.
    pub fn peer(&self) -> Option<usize> {
        self.core
            .lock()
            .unwrap()
            .peer
            .as_ref()
            .map(PeerPuller::peer)
    }

    /// Whether its buffers hold `bytes` per kind.
    pub fn fits(&self, bytes: [usize; 3]) -> bool {
        self.bufs
            .iter()
            .zip(bytes)
            .all(|(b, n)| b.iter().all(|m| m.size() >= n))
    }

    /// The allocations its buffers are made of, with their sizes.
    pub fn handles(&self) -> Vec<(u64, usize)> {
        self.bufs
            .iter()
            .flatten()
            .flat_map(|b| b.handles())
            .collect()
    }

    /// Keeps the buffers across lends: maps each once, so [`Self::reuse`] can zero them again.
    pub fn keep(&mut self) -> Result<(), String> {
        if self.kept.is_empty() {
            self.kept = self
                .bufs
                .iter()
                .flatten()
                .map(DeviceMem::map)
                .collect::<Result<_, _>>()?;
        }
        Ok(())
    }

    /// A [`Parker`] for the refill thread of a restore.
    pub fn parker(&self) -> Parker {
        Parker(self.core.clone())
    }

    /// Ready for the next lend after [`Parker::park`]: the buffers zeroed on the copy stream
    /// (the slots' experts filled them meanwhile; a new buffer starts zeroed), ahead of any
    /// staging copy.
    pub fn reuse(&mut self) -> Result<(), String> {
        let mut c = self.core.lock().unwrap();
        for m in &self.kept {
            let (at, n) = m.range();
            // SAFETY: the buffer's own mapping; nothing reads it until the copies after this.
            unsafe { c.copier.zero(at, n)? };
        }
        Ok(())
    }

    /// Kind `k`'s buffer for `row`.
    pub fn buffer(&self, k: usize, row: usize) -> &DeviceMem {
        // Consecutive rows take turns: the copy queued when a row's matmuls finish (the row
        // `ring` ahead) fills the buffer they just read.
        &self.bufs[k][row % self.bufs[k].len()]
    }

    /// Declares that `row` stages (once, when its staged tensors exist), its copy in up to
    /// `pieces` pieces with split waits (0: none).
    pub fn add_row(&mut self, row: usize, pieces: usize) -> Result<(), String> {
        let mut c = self.core.lock().unwrap();
        let split = Split {
            pre: (pieces > 0).then(|| Event::new(self.device)).transpose()?,
            evs: (0..pieces)
                .map(|_| Event::new(self.device))
                .collect::<Result<_, _>>()?,
            ..Split::default()
        };
        c.rows[row] = Some(Row {
            copied: Event::new(self.device)?,
            done: Event::new(self.device)?,
            split,
        });
        Ok(())
    }

    /// Starts a pass over `plan` (rows in the order the graph runs them), each row with the
    /// copies that fill its buffer: queues the first `ring` rows' copies now, or none if
    /// [`Self::chain`] planned this pass and the previous one queued them. The previous pass
    /// must be complete. `help`: the peer ([`Self::peer`]) is idle during this pass and
    /// takes half of its copies.
    pub fn begin(&mut self, plan: Vec<(usize, Vec<Copy>)>, help: bool) -> Result<(), String> {
        let ring = self.ring();
        let mut c = self.lock_timed();
        c.help = help;
        if let Some(start) = c.ahead.take()
            && c.plan[start..] == plan[..]
        {
            c.plan.drain(..start);
            c.queued = c.queued.saturating_sub(start);
            c.missed = 0;
            // Normally all of them already are.
            while c.queued < ring.min(c.plan.len()) {
                c.queue_next()?;
            }
            return Ok(());
        }
        // Copies a chained pass that did not come queued wrote buffers this pass fills again
        // after them, in the copy stream's order.
        c.plan = plan;
        c.queued = 0;
        c.missed = 0;
        for _ in 0..ring {
            c.queue_next()?;
        }
        Ok(())
    }

    /// Plans the pass after the running one (as [`Self::begin`] would): the `done` ops of the
    /// running pass's last rows queue its first rows' copies, each after the compute that last
    /// read its buffer. Does nothing if a pass is already chained, or where a row would not
    /// follow a row of the same buffer `ring` rows earlier (then the next `begin` queues).
    pub fn chain(&mut self, plan: Vec<(usize, Vec<Copy>)>) -> Result<(), String> {
        let ring = self.ring();
        let mut c = self.lock_timed();
        let start = c.plan.len();
        if c.ahead.is_some() || start < ring || plan.is_empty() {
            return Ok(());
        }
        let row_at = |i: usize| {
            if i < start {
                c.plan[i].0
            } else {
                plan[i - start].0
            }
        };
        let same = (start..start + ring.min(plan.len()))
            .all(|i| row_at(i) % ring == row_at(i - ring) % ring);
        if !same {
            return Ok(());
        }
        c.ahead = Some(start);
        c.plan.extend(plan);
        // Rows whose `done` ran before this: the copy stream already waited for them.
        for _ in 0..std::mem::take(&mut c.missed) {
            c.queue_next()?;
        }
        Ok(())
    }

    fn ring(&self) -> usize {
        self.bufs.first().map_or(super::PREFILL_RING, Vec::len)
    }

    /// The staging lock, the wait for it counted ([`StageStats::lock_wait_ms`]).
    fn lock_timed(&self) -> std::sync::MutexGuard<'_, Core> {
        let t0 = std::time::Instant::now();
        let mut c = self.core.lock().unwrap();
        c.stats.lock_wait_ms += t0.elapsed().as_secs_f64() * 1e3;
        c
    }

    /// What it queued since the last call ([`StageStats`]), and starts counting anew.
    pub fn take_stats(&self) -> StageStats {
        std::mem::take(&mut self.core.lock().unwrap().stats)
    }

    /// The registry id that [`split_wait`] takes.
    pub fn id(&self) -> usize {
        self.id
    }

    /// The names of `row`'s wait and done ops ([`WAIT`], [`DONE`]).
    pub fn op_names(&self, row: usize) -> (String, String) {
        (
            format!("stage-{}-w-{row}", self.id),
            format!("stage-{}-d-{row}", self.id),
        )
    }
}

/// `(registry id, done?, row)` of a staging op's output tensor.
fn parse(dst: &ffi::ggml_tensor) -> Option<(usize, bool, usize)> {
    // SAFETY: ggml tensor names are NUL-terminated.
    let name = unsafe { std::ffi::CStr::from_ptr(dst.name.as_ptr()) }
        .to_str()
        .ok()?;
    let mut it = name.strip_prefix("stage-")?.split('-');
    let id = it.next()?.parse().ok()?;
    let done = it.next()? == "d";
    let row = it.next()?.parse().ok()?;
    Some((id, done, row))
}

fn core_of(id: usize) -> Option<Arc<Mutex<Core>>> {
    REGISTRY
        .lock()
        .unwrap()
        .iter()
        .find(|(i, _)| *i == id)
        .map(|(_, c)| c.clone())
}

/// Runs `run(lo, hi)` for staging `id`'s `row` over expert ids `[0, n_ids)` in ranges, each
/// on `stream` after the copies it reads: the ids cached in VRAM slots after the pads only,
/// then each piece after its copies (the last range runs to `n_ids`). Without marks in the
/// row's plan, or with `whole`, one range after the layer's whole copy.
///
/// # Safety
/// `stream` must be a live stream of the staging device.
pub unsafe fn split_wait(
    id: usize,
    row: usize,
    n_ids: usize,
    whole: bool,
    stream: *mut c_void,
    mut run: impl FnMut(usize, usize) -> Result<(), String>,
) -> Result<(), String> {
    let core = core_of(id).ok_or("staging is gone")?;
    let c = core.lock().unwrap();
    let r = c.rows[row].as_ref().ok_or("row does not stage")?;
    let s = &r.split;
    if whole || !s.marked {
        // SAFETY: the caller's stream.
        unsafe { r.copied.wait_on(stream)? };
        return run(0, n_ids);
    }
    if let Some(pre) = &s.pre {
        // SAFETY: as above.
        unsafe { pre.wait_on(stream)? };
    }
    let mut lo = if s.cuts.is_empty() {
        n_ids
    } else {
        s.first.min(n_ids)
    };
    run(0, lo)?;
    for (i, &hi) in s.cuts.iter().enumerate() {
        // SAFETY: as above.
        unsafe { s.evs[i].wait_on(stream)? };
        let hi = if i + 1 == s.cuts.len() {
            n_ids
        } else {
            hi.clamp(lo, n_ids)
        };
        run(lo, hi)?;
        lo = hi;
    }
    Ok(())
}

unsafe fn cpu_noop(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith == 0 && !dst.data.is_null() {
        // SAFETY: the op's own F32 output.
        unsafe { *(dst.data as *mut f32) = 0.0 };
    }
}

unsafe fn cuda_stage(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let (id, done, row) = parse(dst).ok_or("not a staging op")?;
    let core = core_of(id).ok_or("staging is gone")?;
    let mut c = core.lock().unwrap();
    if done {
        {
            let r = c.rows[row].as_ref().ok_or("row does not stage")?;
            // SAFETY: the compute stream ggml hands the op.
            unsafe { r.done.record(stream)? };
            let ev: *const Event = &r.done;
            // SAFETY: the event lives in `c.rows`, which `wait` does not touch.
            c.copier.wait(unsafe { &*ev })?;
        }
        c.queue_next()?;
    } else {
        let r = c.rows[row].as_ref().ok_or("row does not stage")?;
        // SAFETY: as above.
        unsafe { r.copied.wait_on(stream)? };
    }
    // SAFETY: the op's own 4-byte output on the compute stream: added to the activations as
    // a zero, it orders the graph around this op.
    unsafe { zero_async(dst.data, 4, stream) }
}

/// Before a layer's expert matmuls: wait for its staged copy.
pub static WAIT: ExternalOp = ExternalOp {
    name: "stage_wait",
    cpu: cpu_noop,
    cuda: Some(cuda_stage),
};

/// After a layer's expert matmuls: free its buffer, queue the layer `ring` ahead.
pub static DONE: ExternalOp = ExternalOp {
    name: "stage_done",
    cpu: cpu_noop,
    cuda: Some(cuda_stage),
};
