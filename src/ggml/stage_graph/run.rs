//! The CUDA side of stage graphs: the captures kept per graph and by its words ([`super::store`]),
//! and a graph's run through them.

use std::collections::HashMap;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::{Duration, Instant};

use super::super::backend::LateInput;
use super::super::external::{cuda, op_of};
use super::super::{GgmlError, ffi};
use super::cond::{self, CondExec};
use super::kept_words::{self, KeptRun, KeptSlot, LateAt};
use super::store::{Kept, Store, hash};
use super::{Piece, Unit, plan, props, props_same, props_same_diff, srcs_are_nodes, stats, units};

mod cluster_nodes;
mod copy_node;
mod sig;

unsafe extern "C" {
    fn cudaStreamBeginCapture(stream: *mut c_void, mode: c_int) -> c_int;
    fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> c_int;
    fn cudaGraphInstantiateWithFlags(
        exec: *mut *mut c_void,
        graph: *mut c_void,
        flags: u64,
    ) -> c_int;
    fn cudaGraphExecUpdate(
        exec: *mut c_void,
        graph: *mut c_void,
        info: *mut UpdateResultInfo,
    ) -> c_int;
    fn cudaGraphGetNodes(graph: *mut c_void, nodes: *mut *mut c_void, num: *mut usize) -> c_int;
    fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaGraphDestroy(graph: *mut c_void) -> c_int;
    fn cudaGraphExecDestroy(exec: *mut c_void) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
}

/// `cudaDevAttrComputeCapabilityMajor`, `...Minor`, `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;
const ATTR_SMEM_OPTIN: c_int = 97;

/// Per device: 0 not checked yet, 1 [`mmq_ids`] holds, 2 it does not.
static MMQ_IDS: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];

/// Whether the current device's expert matmuls (`mul_mat_id`) of an MMQ type under
/// [`super::MMQ_IDS_MAX_TOKENS`] tokens may be captured ([`super::ggml_capturable`]): compute
/// capability 7.5 or newer with at least 48 KiB of shared memory a block. ggml-cuda takes mmvq
/// or MMQ for them there (`ggml_cuda_should_use_mmq`: Turing's MMA, or under
/// `MMQ_DP4A_MAX_BATCH_SIZE` tokens whatever it was compiled for), never its synchronizing
/// path. The rule holds from 6.0 on, but older GPUs (the AC922's V100s) keep their captures as they
/// were until measured there.
fn mmq_ids() -> bool {
    let mut dev = 0;
    // SAFETY: plain queries of the current device.
    unsafe {
        if cudaGetDevice(&mut dev) != 0 || !(0..64).contains(&dev) {
            cudaGetLastError();
            return false;
        }
        let slot = &MMQ_IDS[dev as usize];
        match slot.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        let (mut major, mut minor, mut smem) = (0, 0, 0);
        let ok = cudaDeviceGetAttribute(&mut major, ATTR_CC_MAJOR, dev) == 0
            && cudaDeviceGetAttribute(&mut minor, ATTR_CC_MINOR, dev) == 0
            && cudaDeviceGetAttribute(&mut smem, ATTR_SMEM_OPTIN, dev) == 0;
        if !ok {
            cudaGetLastError();
        }
        let on = ok && major * 10 + minor >= 75 && smem >= 48 * 1024;
        slot.store(if on { 1 } else { 2 }, Ordering::Relaxed);
        on
    }
}

/// `cudaStreamCaptureModeRelaxed`, as ggml-cuda captures: other threads (and ggml's memory
/// pool on this one) may make calls a stricter mode would refuse during the capture.
const CAPTURE_RELAXED: c_int = 2;

/// Captures unused this long are dropped (ggml-cuda drops its own graphs after 10 s too).
const UNUSED: Duration = Duration::from_secs(10);
/// How often the cache looks for unused captures.
const SWEEP: Duration = Duration::from_secs(5);

/// `cudaGraphExecUpdateResultInfo` (driver_types.h, CUDA 12+), written by CUDA.
#[repr(C)]
struct UpdateResultInfo {
    result: c_int,
    error_node: *mut c_void,
    error_from_node: *mut c_void,
}

/// An instantiated capture, shared with the registry of big ones ([`LIVE`]).
type Exec = Arc<ExecCell>;

/// An instantiated capture: its executable graph (taken out when a new capture of the same
/// topology takes it over, [`steal`]), the device it was made on, its CUDA graph's node count,
/// where its kernels on clusters are ([`super::spare::cluster_key`]) and when it was last
/// launched (for [`super::spare`]).
struct ExecCell {
    x: Mutex<Option<Handle>>,
    dev: c_int,
    nodes: usize,
    clusters: u64,
    used: AtomicU64,
}

impl ExecCell {
    /// A capture of `nodes` CUDA graph nodes on device `dev` with its kernels on clusters at
    /// `clusters`, registered for [`steal`] when it is big enough.
    fn new(x: *mut c_void, dev: c_int, nodes: usize, clusters: u64) -> Exec {
        let cell = Arc::new(ExecCell {
            x: Mutex::new(Some(Handle(x))),
            dev,
            nodes,
            clusters,
            used: AtomicU64::new(now_ns()),
        });
        if dev >= 0 && nodes >= super::spare::MIN_NODES && super::exec_steal_ms() > 0 {
            let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            if live.len() % 64 == 63 {
                live.retain(|w| w.strong_count() > 0);
            }
            live.push(Arc::downgrade(&cell));
        }
        cell
    }

    /// Whether its executable graph was taken over ([`steal`]): the set captures it again.
    fn taken(&self) -> bool {
        self.x.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }

    /// Launches its executable graph on `stream`; `Ok(false)` if it was taken over.
    ///
    /// # Safety
    /// `stream` is a stream of the cell's device.
    unsafe fn launch(&self, stream: *mut c_void) -> Result<bool, c_int> {
        let x = self.x.lock().unwrap_or_else(|e| e.into_inner());
        let Some(h) = x.as_ref() else {
            return Ok(false);
        };
        self.used.store(now_ns(), Ordering::Relaxed);
        // SAFETY: an executable graph of this device's kernels; the caller's stream.
        match unsafe { cudaGraphLaunch(h.0, stream) } {
            0 => Ok(true),
            status => Err(status),
        }
    }
}

impl Drop for ExecCell {
    /// A big one becomes a spare ([`super::spare`]) that a capture of the same topology may
    /// update; the rest (and spares past the cap) are destroyed.
    fn drop(&mut self) {
        let Some(h) = self.x.get_mut().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let gone = SPARES.lock().unwrap_or_else(|e| e.into_inner()).put(
            self.dev,
            self.nodes,
            self.clusters,
            h,
            super::spare_execs(),
        );
        // Destroyed outside the lock.
        drop(gone);
    }
}

/// An executable graph, destroyed when dropped.
struct Handle(*mut c_void);

// SAFETY: an executable graph is a handle any thread may update, launch or destroy; a handle
// is used by one thread at a time (under its cell's lock, or taken out of the pool).
unsafe impl Send for Handle {}

impl Handle {
    /// The handle, no longer destroyed by this value.
    fn into_raw(self) -> *mut c_void {
        let x = self.0;
        std::mem::forget(self);
        x
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: an executable graph made by cudaGraphInstantiateWithFlags, destroyed once. A
        // launch still running is freed by CUDA when it completes.
        unsafe { cudaGraphExecDestroy(self.0) };
    }
}

/// The spare executable graphs of every device ([`super::spare`]).
static SPARES: LazyLock<Mutex<super::spare::Spares<Handle>>> = LazyLock::new(Mutex::default);

/// The big captures still held by a set ([`ExecCell::new`]), which [`steal`] may take over.
static LIVE: LazyLock<Mutex<Vec<Weak<ExecCell>>>> = LazyLock::new(Mutex::default);

/// Nanoseconds since the first call (the cells' last launch).
fn now_ns() -> u64 {
    static START: LazyLock<Instant> = LazyLock::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

/// Takes over the executable graph of the big capture of device `dev`, `nodes` CUDA graph
/// nodes and kernels on clusters at `clusters` that was launched longest ago, if that was at
/// least `min_idle_ns` ago ([`super::spare::pick_idle`]). Its set captures it again if it runs
/// again ([`ExecCell::taken`]).
fn steal(dev: c_int, nodes: usize, clusters: u64, min_idle_ns: u64) -> Option<Handle> {
    let cells: Vec<Exec> = {
        let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        live.retain(|w| w.strong_count() > 0);
        live.iter()
            .filter_map(Weak::upgrade)
            // A taken cell stays registered while its set lives, idle forever: skipped, or
            // it would be picked again and again with nothing left to take.
            .filter(|c| c.dev == dev && c.nodes == nodes && c.clusters == clusters && !c.taken())
            .collect()
    };
    let now = now_ns();
    let idle = |c: &Exec| now.saturating_sub(c.used.load(Ordering::Relaxed));
    let i = super::spare::pick_idle(cells.iter().map(idle), min_idle_ns)?;
    let c = &cells[i];
    let mut x = c.x.lock().unwrap_or_else(|e| e.into_inner());
    // Launched again since (or taken by another capture): left alone.
    if idle(c) < min_idle_ns {
        return None;
    }
    x.take()
}

/// What the cache keeps of one graph on one backend.
#[derive(Default)]
struct Entry {
    /// The graph's words ([`props_same`]) at its last run, in the shorter form when `nodes`.
    words: Vec<u64>,
    nodes: bool,
    /// Whether the shorter form was weighed since the words last changed: once a graph runs
    /// unchanged, it is (a graph that changes every run, a prompt chunk's, never pays for it).
    weighed: bool,
    /// Its units at its last run ([`units`]).
    units: Vec<Unit>,
    /// Runs in a row with the same words before this one (0: the first run or changed).
    same: u32,
    /// Its set of captures in the store ([`Cache::sets`]), once planned: the hash of its words
    /// (full form) and the set's id.
    set: Option<(u64, u64)>,
    /// A capture of it failed: it runs directly until its words change.
    refused: bool,
    /// Its units behind a device condition ([`cond`]), once a run asked for it.
    cond: Option<CondExec>,
    /// Building `cond` failed: it runs plainly until its words change.
    cond_refused: bool,
    /// The kept graph and epoch of its last full words compare ([`kept_words`]).
    seen: Option<kept_words::Seen>,
    /// Where a kept graph's late input was first read ([`LateAt`]).
    late_at: Option<LateAt>,
    used: Option<Instant>,
}

/// A kept graph's entry and set of captures, held in its [`KeptSlot`] between runs.
pub(super) struct Held {
    /// The entry's key in the shared cache (backend, graph).
    key: (usize, usize),
    entry: Entry,
    set: Option<Taken>,
}

impl Drop for Held {
    /// The graph is dropped (or its entry goes back to the shared cache, the set taken out
    /// before): its captures go back into the store, as a run's [`put`] leaves them.
    fn drop(&mut self) {
        if let Some((h, k)) = self.set.take() {
            let now = Instant::now();
            if kept_words::hold_all() {
                // A set made while held never went through the store's cap: it does now.
                cache()
                    .sets
                    .insert(self.key.0, h, k, now, super::cache_sets());
            } else {
                cache().sets.put(self.key.0, h, k, now);
            }
        }
    }
}

/// The pieces of a captured run and their captures.
#[derive(Default)]
struct Captured {
    pieces: Vec<Piece>,
    execs: Vec<Option<Exec>>,
}

// SAFETY: executable graphs are handles any thread may launch or destroy; a set is used by one
// thread at a time (taken out of the cache meanwhile).
unsafe impl Send for Captured {}

/// A set of captures taken out of the store for a run: its words' hash and the set.
type Taken = (u64, Kept<Captured>);

#[derive(Default)]
struct Cache {
    entries: HashMap<(usize, usize), Entry>,
    /// The captures, per backend and words ([`super::store`]).
    sets: Store<Captured>,
    swept: Option<Instant>,
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Mutex::default);

fn cache() -> std::sync::MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Takes the entry of graph `g` on backend `raw` out of the cache (a new one if none) with its
/// set of captures (if planned and still kept), and drops the entries and captures unused for
/// [`UNUSED`]. The flag: the entry had a set planned that is no longer kept.
fn take(key: (usize, usize)) -> (Entry, Option<Taken>, bool) {
    let mut c = cache();
    let now = Instant::now();
    if c.swept.is_none_or(|t| now - t >= SWEEP) {
        c.swept = Some(now);
        c.entries
            .retain(|_, e| e.used.is_some_and(|t| now - t < UNUSED));
        c.sets.sweep(now, UNUSED);
    }
    let mut e = c.entries.remove(&key).unwrap_or_default();
    let set = e
        .set
        .and_then(|(h, id)| Some((h, c.sets.take_id(key.0, h, id)?)));
    let lost = set.is_none() && e.set.is_some();
    if set.is_none() {
        // Dropped meanwhile (unused, or past the cap): planned again on its next captured run.
        e.set = None;
    }
    (e, set, lost)
}

/// Takes the entry of graph `g` on backend `raw` ([`take`]'s `key`): from its kept graph's
/// slot if it is held there for the same key, else from the shared cache ([`take`]).
fn take_kept(key: (usize, usize), kept: Option<KeptRun<'_>>) -> (Entry, Option<Taken>, bool) {
    if let Some(k) = kept {
        let held = k.slot.held.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(mut h) = held {
            let (e, set) = (std::mem::take(&mut h.entry), h.set.take());
            if h.key == key {
                return (e, set, false);
            }
            // Run on another backend now: the old key's entry goes back to the shared cache.
            put(h.key, e, set, false);
        }
    }
    take(key)
}

/// Holds entry `e` of `key` and its set of captures in kept graph `slot` until its next run.
fn hold(slot: &KeptSlot, key: (usize, usize), e: Entry, set: Option<Taken>) {
    let held = Held { key, entry: e, set };
    let old = slot
        .held
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(held);
    // None: the run took it out (outside the lock: a set goes back into the store).
    drop(old);
}

/// Puts entry `e` of `key` back, with its set of captures (`fresh`: new, kept now).
fn put(key: (usize, usize), e: Entry, set: Option<Taken>, fresh: bool) {
    let now = Instant::now();
    let mut c = cache();
    c.entries.insert(key, e);
    if let Some((h, k)) = set {
        if fresh {
            c.sets.insert(key.0, h, k, now, super::cache_sets());
        } else {
            c.sets.put(key.0, h, k, now);
        }
    }
}

/// After a graph's words changed on backend `dev`: puts back the set of its last words
/// (`old`; dropped when the store keeps none, as before #858), and takes the set kept for its
/// new `words` (full form), if any.
fn swap(dev: usize, old: Option<Taken>, words: &[u64]) -> Option<Taken> {
    if super::cache_sets() == 0 {
        drop(old);
        return None;
    }
    let h = hash(words);
    let now = Instant::now();
    let mut c = cache();
    if let Some((oh, k)) = old {
        c.sets.put(dev, oh, k, now);
    }
    let found = c.sets.take_words(dev, h, words)?;
    drop(c);
    stats::count_reuse();
    Some((h, found))
}

/// Runs graph `g` on CUDA backend `raw` with stage graphs: its units ([`units`]) directly on
/// its first run and after a change, captured on its second run in a row unchanged (or right
/// behind the first one, [`super::capture_early`]), the captures launched after that. `late`
/// (taken) is written before its first reader's segment. To completion when `sync`. `kept`:
/// the kept graph `g` is ([`crate::ggml::KeptGraph`]), whose unchanged runs may skip the words
/// check and keep their cache entry with the graph ([`kept_words`]).
///
/// # Safety
/// As for `run_graph` (backend.rs): `raw` is a live CUDA backend, `g` a graph allocated in
/// memory it computes in, and nothing else uses either until this returns. ggml's own CUDA
/// graphs are off ([`super::set_enabled`]). With `kept`, `g` is that kept graph's.
pub(in crate::ggml) unsafe fn compute(
    raw: *mut ffi::ggml_backend,
    g: *mut ffi::ggml_cgraph,
    sync: bool,
    late: &mut Option<LateInput<'_>>,
    kept: Option<KeptRun<'_>>,
) -> Result<(), GgmlError> {
    // The check's host time, for --stage-graph-stats only.
    let check_t0 = stats::logging().then(Instant::now);
    let key = (raw as usize, g as usize);
    let (mut e, mut set, lost) = take_kept(key, kept);
    // Whether this kept graph ran before (for the stats' [`stats::Why`]).
    let ran = kept.is_some_and(|k| k.slot.ran.swap(true, Ordering::Relaxed));
    let serial = kept.map(|k| k.serial);
    let (on, epoch) = (kept_words::enabled(), kept_words::epoch());
    // SAFETY: the caller's: a live graph, node indices in range, sources live or null.
    let (n, late_at) = unsafe {
        let n = ffi::ggml_graph_n_nodes(g);
        // The late input's first reader (the graph's start if none reads it): a kept graph's
        // nodes never change, so found once per kept graph and input.
        let late_at = late.as_ref().map(|l| {
            let t = l.tensor.raw();
            if let Some(at) = LateAt::again(e.late_at, on, serial, epoch, t as usize, n) {
                return at;
            }
            let at = (0..n)
                .find(|&i| (*ffi::ggml_graph_node(g, i)).src.contains(&t))
                .unwrap_or(0);
            e.late_at = serial.map(|serial| LateAt {
                serial,
                epoch,
                tensor: t as usize,
                nodes: n,
                at,
            });
            at
        });
        (n, late_at)
    };
    // Where the late input is written splits the units too.
    let tail = late_at.map_or(u64::MAX, |at| at as u64);
    // After a change: whether the entry was new, and its last words' count of runs (taken
    // before the compare below writes this run's words).
    let (fresh, last) = (e.words.is_empty(), e.same);
    let mut changed = None;
    // An unchanged kept graph's words are this run's already ([`kept_words`]).
    let last_tail = e.words.last().copied();
    let skip = kept_words::skip(kept_words::Check {
        on,
        kept: serial,
        epoch,
        seen: e.seen,
        weighed: e.weighed,
        same: e.same,
        last_tail,
        tail,
    });
    let (same, diff) = if skip {
        (true, None)
    } else {
        // SAFETY: the caller's.
        unsafe { props_same_diff(g, &mut e.words, tail, e.nodes) }
    };
    if !same && !fresh {
        stats::count_reword(n, e.nodes, diff);
        let skippable = e.seen.is_some() && e.seen == kept_words::seen(serial, epoch);
        if skippable && e.weighed && last_tail == Some(tail) {
            // Words a skip would have taken as unchanged: something rewrote a kept graph's
            // tensors without `words_may_change`.
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                tracing::warn!(
                    "a kept stage graph's words changed without a rebuild; \
                     run with --no-stage-graph-words-skip and report it"
                )
            });
        }
    }
    if !skip {
        e.seen = kept_words::seen(serial, epoch);
    }
    if same {
        // The same units as the last run: they follow from the node count, which nodes are
        // external ops (each node's op and parameters, the op's own) and the late input's
        // reader, all in the words.
        e.same = e.same.saturating_add(1);
        // A graph that runs again unchanged compares in the shorter form from its next run
        // on, where that form holds for it: this run's words in it.
        if !e.weighed {
            e.weighed = true;
            // SAFETY: the caller's.
            if unsafe { srcs_are_nodes(g) } {
                e.nodes = true;
                // SAFETY: the caller's.
                unsafe { props_same(g, &mut e.words, tail, true) };
            }
        }
    } else {
        changed = Some((fresh, last));
        // Changed (or new): back to the full form until it repeats (the shorter one may no
        // longer hold), this run's words in it.
        if e.nodes {
            e.nodes = false;
            // SAFETY: the caller's.
            unsafe { props_same(g, &mut e.words, tail, false) };
        }
        e.weighed = false;
        // SAFETY: the caller's.
        let ext: Vec<i32> = unsafe {
            (0..n)
                .filter(|&i| op_of(ffi::ggml_graph_node(g, i)).is_some())
                .collect()
        };
        e.units = units(n, &ext, late_at);
        e.same = 0;
        // A prompt step holding an op that makes it run directly (#1176): as a refused capture,
        // directly until its words change.
        // SAFETY: the caller's; `ext` are nodes of `g`.
        e.refused = unsafe { super::prompt_direct(g, &ext) };
        // The last words' captures stay kept under them; these words' own, made by a graph
        // with the same words before (a kept step graph built again), are launched from this
        // run on.
        set = swap(key.0, set.take(), &e.words);
        e.set = set.as_ref().map(|(h, k)| (*h, k.id));
        e.cond = None;
        e.cond_refused = false;
    }
    let units = std::mem::take(&mut e.units);
    let why = match changed {
        _ if kept.is_none() => stats::Why::Unkept,
        Some((false, _)) => stats::Why::Changed,
        Some((true, _)) if ran => stats::Why::Lost,
        _ if lost => stats::Why::Lost,
        _ => stats::Why::New,
    };
    if let Some(t0) = check_t0 {
        stats::count_check(t0.elapsed());
    }
    // A set planned in this run, kept from now on.
    let mut made = false;
    // SAFETY: the caller's; every kernel and capture goes on the backend's stream, in order.
    let done = unsafe {
        // The external ops launch on the current device, which a graph starting with one has
        // not set yet on this thread.
        cuda::legion_cuda_set_device(raw);
        let stream = cuda::legion_cuda_stream(raw);
        let _backend = cuda::BackendScope::enter(raw, g);
        let run = Run {
            raw,
            g,
            stream,
            units: &units,
            captured: std::cell::Cell::new(0),
        };
        // A graph marked to wait for its reuse, unchanged since its first run and not
        // captured yet: run directly once more, captured behind it (`capture_on_reuse`).
        let second =
            super::second_now(mmq_ids(), e.same, set.is_some(), e.refused, cond::current());
        let done = if (e.same == 0 && set.is_none()) || e.refused || second {
            if let Some(k) = kept {
                let split = if mtp(k) {
                    stats::Split::DirectMtp
                } else {
                    stats::Split::DirectStep
                };
                stats::count_split(split, 1);
            }
            let r = units.iter().try_for_each(|u| run.unit(*u, late));
            // On the graph's device (set above).
            let early = second
                || changed.is_some_and(|(fresh, last)| super::early_now(mmq_ids(), fresh, last));
            if early && r.is_ok() {
                // Captured now, behind the run just queued, rather than on the next run with
                // the GPU waiting (`capture_early`).
                let mut s = new_set(g, &units, &e, tail);
                run.capture_pieces(&mut s.1.val, &mut e.refused);
                e.set = Some((s.0, s.1.id));
                set = Some(s);
                made = true;
            }
            r
        } else {
            if set.is_none() {
                let s = new_set(g, &units, &e, tail);
                e.set = Some((s.0, s.1.id));
                set = Some(s);
                made = true;
            }
            let (_, s) = set.as_mut().expect("planned above");
            match cond::current() {
                // Behind the device's condition where every unit is captured (no fill, no
                // unsafe op between them).
                true if !e.cond_refused
                    && (s.val.pieces.iter()).all(|p| matches!(p, Piece::Captured(_))) =>
                {
                    run.cond(&mut e, &mut s.val)
                }
                true if !e.cond_refused => {
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::info!(
                            "an mtp pass queued ahead has units run directly: it runs plainly"
                        )
                    });
                    run.pieces(&mut s.val, &mut e.refused, late)
                }
                _ => run.pieces(&mut s.val, &mut e.refused, late),
            }
        };
        stats::count_why(why, run.captured.get());
        if why == stats::Why::New && kept.is_some_and(mtp) {
            stats::count_split(stats::Split::NewMtp, run.captured.get());
        }
        done
    };
    e.units = units;
    let all = on && kept_words::hold_all();
    match kept {
        // Unchanged (or any run, `all`): with the graph until its next run, without the
        // shared cache.
        Some(k) if kept_words::hold(true, skip, made, all) => hold(k.slot, key, e, set),
        _ => {
            e.used = Some(Instant::now());
            put(key, e, set, made);
        }
    }
    stats::count_run();
    if sync || done.is_err() {
        // SAFETY: the caller's.
        unsafe { ffi::ggml_backend_synchronize(raw) };
    }
    done
}

/// Whether kept graph `k` is the MTP head's ([`crate::ggml::KeptGraph::mark_mtp`]).
fn mtp(k: KeptRun<'_>) -> bool {
    k.slot.mtp.load(Ordering::Relaxed)
}

/// A new set of captures for graph `g` of `units` (none captured yet): its pieces planned, and
/// the words it is kept under, in the full form (`e`'s own when they are in it; the shorter
/// form names the graph's nodes, which a graph built again does not share), with the late
/// input's place `tail` last, as [`props_same`] has them.
///
/// # Safety
/// As for [`compute`], on the graph's device.
unsafe fn new_set(g: *mut ffi::ggml_cgraph, units: &[Unit], e: &Entry, tail: u64) -> Taken {
    let words = if e.nodes {
        let mut w = Vec::with_capacity(2 * e.words.len());
        // SAFETY: the caller's.
        unsafe { props(g, &mut w) };
        w.push(tail);
        w
    } else {
        e.words.clone()
    };
    // SAFETY: the caller's.
    let pieces = unsafe { plan(g, units, mmq_ids()) };
    let execs = pieces.iter().map(|_| None).collect();
    let h = hash(&words);
    (
        h,
        Kept::new(words, Captured { pieces, execs }, Instant::now()),
    )
}

/// Captures graph `g`'s pieces on CUDA backend `raw` without running any of them (#836), and
/// keeps them in the store under the words its first run will have, `late` being the input
/// that run fills late (if any): that run finds them ([`swap`]) and launches them, rather than
/// running its units directly and capturing them behind it. Only where that first run would
/// capture early ([`super::capture_early`]: the setting on, a device whose small steps capture
/// whole) and the store keeps sets. Whether a set was captured (none: already kept, or CUDA
/// refused a piece; the first run then goes as before).
///
/// The same units planned into the same pieces and recorded on the same stream as a capture
/// behind the first run: the same kernels with the same arguments, so the same bits. What the
/// captures read and write is the graph's memory, as when it runs.
///
/// # Safety
/// As for [`compute`]: `raw` is a live CUDA backend and `g` a graph allocated in memory it
/// computes in, nothing else using either until this returns; work queued on the stream
/// before (another graph computing) may still run. `late` is a tensor of `g` or null.
pub(in crate::ggml) unsafe fn precapture(
    raw: *mut ffi::ggml_backend,
    g: *mut ffi::ggml_cgraph,
    late: Option<*mut ffi::ggml_tensor>,
) -> bool {
    if super::cache_sets() == 0 || !super::EARLY.load(Ordering::Relaxed) {
        return false;
    }
    // Planned under the step mark as it is now ([`super::small_step`]): the graph built ahead
    // is the running step's shape a few positions on, so its first run gets the same mark (a
    // verify step with a sequence over the decode limit counts as a prompt step, and keeps the
    // ops a capture holds in small steps only between its pieces, as on main).
    // SAFETY: the caller's; node indices in range, sources live or null.
    unsafe {
        cuda::legion_cuda_set_device(raw);
        if !mmq_ids() {
            return false;
        }
        let n = ffi::ggml_graph_n_nodes(g);
        // As `compute` finds the late input's first reader and the words' tail.
        let late_at = late.map(|t| {
            (0..n)
                .find(|&i| (*ffi::ggml_graph_node(g, i)).src.contains(&t))
                .unwrap_or(0)
        });
        let tail = late_at.map_or(u64::MAX, |at| at as u64);
        let mut words = Vec::new();
        props(g, &mut words);
        words.push(tail);
        let h = hash(&words);
        if cache().sets.has(raw as usize, h, &words) {
            return false;
        }
        let ext: Vec<i32> = (0..n)
            .filter(|&i| op_of(ffi::ggml_graph_node(g, i)).is_some())
            .collect();
        // Its first run goes directly (#1176): nothing to capture for it.
        if super::prompt_direct(g, &ext) {
            return false;
        }
        let units = units(n, &ext, late_at);
        let pieces = plan(g, &units, true);
        let mut val = Captured {
            execs: pieces.iter().map(|_| None).collect(),
            pieces,
        };
        let stream = cuda::legion_cuda_stream(raw);
        let _backend = cuda::BackendScope::enter(raw, g);
        let run = Run {
            raw,
            g,
            stream,
            units: &units,
            captured: std::cell::Cell::new(0),
        };
        let mut refused = false;
        run.capture_pieces(&mut val, &mut refused);
        if refused {
            return false;
        }
        stats::count_precapture();
        let now = Instant::now();
        cache().sets.insert(
            raw as usize,
            h,
            Kept::new(words, val, now),
            now,
            super::cache_sets(),
        );
        true
    }
}

/// One run of a graph's units on its backend's stream.
struct Run<'u> {
    raw: *mut ffi::ggml_backend,
    g: *mut ffi::ggml_cgraph,
    stream: *mut c_void,
    units: &'u [Unit],
    /// Pieces this run captured (for [`stats::count_why`]).
    captured: std::cell::Cell<u64>,
}

impl Run<'_> {
    /// Counts one piece captured in `took`.
    fn counted(&self, took: Duration) {
        stats::count_capture(took);
        self.captured.set(self.captured.get() + 1);
    }

    /// Runs (or, during a capture, records) unit `u`.
    ///
    /// # Safety
    /// As for [`compute`], on the graph's device.
    unsafe fn unit(&self, u: Unit, late: &mut Option<LateInput<'_>>) -> Result<(), GgmlError> {
        // SAFETY: the caller's; node indices in range.
        unsafe {
            match u {
                Unit::Segment(i0, i1) => {
                    let status = cuda::legion_cuda_compute_range(self.raw, self.g, i0, i1);
                    if status == ffi::ggml_status_GGML_STATUS_SUCCESS {
                        Ok(())
                    } else {
                        Err(GgmlError::Compute(status))
                    }
                }
                Unit::External(i) => {
                    let node = ffi::ggml_graph_node(self.g, i);
                    let op = op_of(node).expect("an external op's unit");
                    let launch = op.cuda.ok_or_else(|| {
                        GgmlError::External(format!("{}: no CUDA kernel", op.name))
                    })?;
                    launch(&*node, self.stream).map_err(|e| {
                        let name = std::ffi::CStr::from_ptr((*node).name.as_ptr());
                        let e = format!("{} (node {i} `{}`): {e}", op.name, name.to_string_lossy());
                        // A sticky CUDA error (an illegal address) aborts the sync that
                        // follows before this error is seen: name the op first.
                        tracing::error!("stage graph: {e}");
                        GgmlError::External(e)
                    })
                }
                Unit::Fill => {
                    // Its write returns once the bytes are on the device, before the reader
                    // is queued.
                    if let Some(l) = late.take() {
                        (l.fill)();
                    }
                    Ok(())
                }
            }
        }
    }

    /// Units `us`' node range and its first and last nodes' names, for an error.
    ///
    /// # Safety
    /// As for [`Self::unit`].
    unsafe fn describe(&self, us: &[Unit]) -> String {
        let node = |u: &Unit, last: bool| match *u {
            Unit::Segment(i0, i1) => Some(if last { i1 - 1 } else { i0 }),
            Unit::External(i) => Some(i),
            Unit::Fill => None,
        };
        let i0 = us.iter().find_map(|u| node(u, false));
        let i1 = us.iter().rev().find_map(|u| node(u, true));
        let (Some(i0), Some(i1)) = (i0, i1) else {
            return "no nodes".into();
        };
        // SAFETY: the caller's; node indices in range.
        let name = |i| unsafe {
            let n = ffi::ggml_graph_node(self.g, i);
            std::ffi::CStr::from_ptr((*n).name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        format!("nodes {i0}..={i1} `{}`..`{}`", name(i0), name(i1))
    }

    /// Runs the planned pieces: each capture launched (captured first if it is not yet), the
    /// other units directly.
    ///
    /// # Safety
    /// As for [`Self::unit`].
    unsafe fn pieces(
        &self,
        e: &mut Captured,
        refused: &mut bool,
        late: &mut Option<LateInput<'_>>,
    ) -> Result<(), GgmlError> {
        for (k, piece) in e.pieces.iter().enumerate() {
            let range = match piece {
                Piece::Direct(u) => {
                    // SAFETY: the caller's.
                    unsafe { self.unit(self.units[*u], late)? };
                    continue;
                }
                Piece::Captured(r) => r.clone(),
            };
            // None yet, or taken over by a capture of the same topology ([`steal`]).
            if !*refused && e.execs[k].as_ref().is_none_or(|x| x.taken()) {
                let t = Instant::now();
                // SAFETY: the caller's.
                match unsafe { self.capture(&self.units[range.clone()]) } {
                    Some(x) => {
                        self.counted(t.elapsed());
                        e.execs[k] = Some(x);
                    }
                    None => *refused = true,
                }
            }
            let launched = match &e.execs[k] {
                // SAFETY: a capture of this device's kernels; its stream.
                Some(x) => unsafe { x.launch(self.stream) }.map_err(|status| {
                    // SAFETY: the caller's; node indices in range.
                    let e = format!(
                        "stage graph launch (piece {k}, {}): CUDA error {status}",
                        unsafe { self.describe(&self.units[range.clone()]) }
                    );
                    // Named before the sync that follows aborts on a sticky error.
                    tracing::error!("{e}");
                    GgmlError::External(e)
                })?,
                None => false,
            };
            if !launched {
                // Not captured (refused), or taken over since the check above: the same units
                // directly.
                for &u in &self.units[range] {
                    // SAFETY: the caller's.
                    unsafe { self.unit(u, late)? };
                }
            }
        }
        Ok(())
    }

    /// Launches the units behind the device's round flag ([`cond`]), building the entry's conditional graph the
    /// first time; where CUDA refuses it, runs the pieces as before (unconditionally) and
    /// marks the entry so.
    ///
    /// # Safety
    /// As for [`Self::unit`].
    unsafe fn cond(&self, e: &mut Entry, s: &mut Captured) -> Result<(), GgmlError> {
        if e.cond.is_none() {
            let t = Instant::now();
            let mut none = None;
            // SAFETY: the caller's; the stream is the backend's and captures nothing now.
            let built = unsafe {
                CondExec::build(self.stream, || {
                    self.units.iter().try_for_each(|u| self.unit(*u, &mut none))
                })
            };
            match built {
                Some(x) => {
                    self.counted(t.elapsed());
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::info!("mtp passes queued ahead run behind a device condition")
                    });
                    e.cond = Some(x);
                }
                None => {
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::warn!(
                            "CUDA refused a conditional graph: mtp passes queued ahead run plainly"
                        )
                    });
                    e.cond_refused = true;
                    // SAFETY: the caller's.
                    return unsafe { self.pieces(s, &mut e.refused, &mut None) };
                }
            }
        }
        let x = e.cond.as_mut().expect("built above");
        // SAFETY: built on this stream; the caller queued the decide before it.
        unsafe { x.launch(self.stream) }.map_err(|status| {
            GgmlError::External(format!("conditional stage graph: CUDA error {status}"))
        })
    }

    /// Captures each planned piece that is a capture, without launching any ([`Self::pieces`]
    /// launches them on the next run). A refused capture marks the entry refused, as there.
    ///
    /// # Safety
    /// As for [`Self::unit`].
    unsafe fn capture_pieces(&self, e: &mut Captured, refused: &mut bool) {
        for (k, piece) in e.pieces.iter().enumerate() {
            let Piece::Captured(r) = piece else { continue };
            let t = Instant::now();
            // SAFETY: the caller's.
            match unsafe { self.capture(&self.units[r.clone()]) } {
                Some(x) => {
                    self.counted(t.elapsed());
                    e.execs[k] = Some(x);
                }
                None => {
                    *refused = true;
                    return;
                }
            }
        }
    }

    /// Captures `units` (none of them a fill) on the stream as one executable graph: a spare
    /// of the same topology updated with it ([`update_spare`]), else its instantiation. `None`
    /// when CUDA refused the capture or its instantiation, or a unit failed while recorded
    /// (nothing ran then; the caller runs the units directly).
    ///
    /// # Safety
    /// As for [`Self::unit`].
    unsafe fn capture(&self, units: &[Unit]) -> Option<Exec> {
        // SAFETY: the caller's; the stream is the backend's and captures nothing else now (the
        // backend is this thread's until `compute` returns).
        unsafe {
            if cudaStreamBeginCapture(self.stream, CAPTURE_RELAXED) != 0 {
                cudaGetLastError();
                return None;
            }
            let mut none = None;
            let recorded = units.iter().try_for_each(|u| self.unit(*u, &mut none));
            let mut graph = std::ptr::null_mut();
            let end = cudaStreamEndCapture(self.stream, &mut graph);
            let refused = end != 0 || graph.is_null();
            if refused {
                cudaGetLastError();
            }
            if recorded.is_err() || refused {
                // A unit that failed while recording ran nothing either: the direct run after
                // this reports it if it fails there too.
                cudaGetLastError();
                if !graph.is_null() {
                    cudaGraphDestroy(graph);
                }
                return None;
            }
            let (mut dev, mut nodes) = (-1, 0usize);
            if cudaGetDevice(&mut dev) != 0
                || cudaGraphGetNodes(graph, std::ptr::null_mut(), &mut nodes) != 0
            {
                cudaGetLastError();
                // Neither kept as a spare nor matched with one.
                (dev, nodes) = (-1, 0);
            }
            let big = dev >= 0 && nodes >= super::spare::MIN_NODES;
            let mut clusters = 0;
            if big && (super::spare_execs() > 0 || super::exec_steal_ms() > 0) {
                // A copy's node as a kernel node, which an update may move (node count kept).
                let (made, left) = copy_node::convert(graph, dev);
                if stats::logging() && made + left > 0 {
                    super::topo::log_copies(dev, nodes, made, left);
                }
                // Updates pair only captures with their cluster kernels at the same nodes.
                clusters = cluster_nodes::key(graph, dev);
            }
            if stats::logging() && big {
                super::topo::note_capture(dev, sig::of(graph));
            }
            if let Some(x) = update_spare(dev, nodes, clusters, graph) {
                cudaGraphDestroy(graph);
                return Some(ExecCell::new(x, dev, nodes, clusters));
            }
            let mut exec = std::ptr::null_mut();
            let status = cudaGraphInstantiateWithFlags(&mut exec, graph, 0);
            cudaGraphDestroy(graph);
            if status != 0 || exec.is_null() {
                cudaGetLastError();
                return None;
            }
            Some(ExecCell::new(exec, dev, nodes, clusters))
        }
    }
}

/// An executable graph of device `dev` with `nodes` CUDA graph nodes and its kernels on
/// clusters at `clusters` ([`super::spare::cluster_key`]: CUDA would update a plain node with a
/// cluster kernel, which then runs without its cluster) updated with captured `graph`'s node
/// parameters, if one is at hand and CUDA performs the update: launched, it
/// runs `graph`'s nodes with their parameters, as an instantiation of `graph` would. First a
/// spare ([`super::spare`]), else (or if CUDA refused the spare's update) one taken over from
/// a capture unlaunched for `--stage-graph-exec-steal-ms` ([`steal`]). One whose update CUDA
/// refuses (another topology, or a change it cannot apply; the exec is left as it was) is
/// destroyed.
///
/// # Safety
/// `graph` is a live CUDA graph of device `dev`, the current one.
unsafe fn update_spare(
    dev: c_int,
    nodes: usize,
    clusters: u64,
    graph: *mut c_void,
) -> Option<*mut c_void> {
    if dev < 0 || nodes < super::spare::MIN_NODES {
        return None;
    }
    let spare = (super::spare_execs() > 0)
        .then(|| {
            SPARES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take(dev, nodes, clusters)
        })
        .flatten();
    if let Some(h) = spare {
        // SAFETY: the caller's.
        if let Some(x) = unsafe { update(h, graph, false) } {
            return Some(x);
        }
    }
    let ms = super::exec_steal_ms();
    if ms == 0 {
        return None;
    }
    let h = steal(dev, nodes, clusters, ms.saturating_mul(1_000_000))?;
    stats::count_steal();
    // SAFETY: the caller's.
    unsafe { update(h, graph, true) }
}

/// Executable graph `h` updated with `graph`'s node parameters, or `None` (and `h` destroyed)
/// when CUDA refused the update; logged with its reason while `--stage-graph-stats` logs
/// ([`super::topo`]; `stolen`: `h` was taken over from a kept capture).
///
/// # Safety
/// `h` and `graph` are of the current device, `h` used by no other thread.
unsafe fn update(h: Handle, graph: *mut c_void, stolen: bool) -> Option<*mut c_void> {
    let mut info = UpdateResultInfo {
        result: 0,
        error_node: std::ptr::null_mut(),
        error_from_node: std::ptr::null_mut(),
    };
    // SAFETY: the caller's; launches of `h` still queued are not affected.
    let status = unsafe { cudaGraphExecUpdate(h.0, graph, &mut info) };
    if status != 0 {
        // SAFETY: clears the refused update's error.
        unsafe { cudaGetLastError() };
        stats::count_update(false);
        if stats::logging() {
            // SAFETY: the caller's graph; the nodes CUDA named.
            let node = unsafe { sig::error_node(graph, info.error_node, info.error_from_node) };
            super::topo::log_refused(status, info.result, stolen, &node);
        }
        return None;
    }
    stats::count_update(true);
    Some(h.into_raw())
}
