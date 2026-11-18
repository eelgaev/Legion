//! The host side of the expert share: the worker pool of a NUMA node that runs the layers' jobs
//! as the GPUs publish them ([`Pool`]), and what a job computes ([`reference`] is the same on
//! one thread).
//!
//! As Strata-AC922's `ExpertPool` (eelgaev/Strata-AC922 `src/kernels/cpu/pool.cpp`, MIT): one
//! coordinating thread and pinned helpers that spin between jobs; a job's work is split by
//! rows over all its experts per phase, threads claiming tasks from a shared counter; each
//! expert runs once for all the tokens that chose it (its weights are read once per step).

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::HostExperts;
use crate::cpu::{Q8Activations, matmul_q8};

/// One layer's share of one step.
#[derive(Debug, Clone, Copy)]
pub struct Job {
    /// The step's sequence number: the "ready" flag reaches it when the inputs are published,
    /// the pool sets "done" to it when the outputs are written.
    pub seq: u32,
    pub layer: u32,
    pub n_tok: usize,
    pub n_used: usize,
    pub experts: HostExperts,
    /// The device's exchange buffer the step's inputs and outputs are in.
    pub exchange: Exchange,
}

/// Host memory a GPU and the pool share for one device: the step's inputs and outputs and the
/// two flags. Pointers into a buffer that outlives the pool's use of it.
#[derive(Debug, Clone, Copy)]
pub struct Exchange {
    pub ready: *const AtomicU32,
    pub done: *const AtomicU32,
    /// Raised by the GPU when it reaches the step's `collect` (with debug logging on only):
    /// [`stats`] counts the steps it then waited for the host.
    pub arrived: *const AtomicU32,
    /// `[n_tok][n_used]`: the host's expert for the pair, or -1.
    pub ids: *const i32,
    /// `[n_tok][n_embd]`.
    pub x: *const f32,
    /// `[n_tok][n_used][n_embd]`: the host's rows.
    pub out: *mut f32,
    pub max_tok: usize,
    pub max_used: usize,
    pub n_embd: usize,
    /// The exact share's part (x86, see the `cpu_share` module docs): `x` and the host kernels
    /// are not used then.
    pub exact: Option<ExactExchange>,
}

/// The exact share's part of an exchange buffer (`cpu_share::ExactLayout`).
#[derive(Debug, Clone, Copy)]
pub struct ExactExchange {
    /// Raised by the pool when its pairs' gate and up sums are written.
    pub gu_done: *const AtomicU32,
    /// Raised by the GPU when the host pairs' SwiGLU blocks are written (`activate`).
    pub act_ready: *const AtomicU32,
    /// Raised by the GPU when it reaches `activate` (with debug logging on only).
    pub act_arrived: *const AtomicU32,
    /// The step's distinct host experts (`publish`).
    pub n_miss: *const AtomicU32,
    /// `[n_layers]`: each layer's share of 1024, which `publish` reads.
    pub thr: *const AtomicU32,
    /// `[n_layers]`: the pool's running mean of microseconds per expert (0: none yet).
    pub cost: *mut f32,
    pub n_layers: usize,
    /// `[n_tok][n_embd / 32]` block_q8_1: x as the expert kernels quantize it.
    pub xq: *const u8,
    /// `[n_tok][n_used][2][n_ff]`: the host pairs' raw gate and up sums (the layer's `n_ff`).
    pub gu: *mut f32,
    /// `[n_tok][n_used][n_ff / 32]` block_q8_1: the host pairs' SwiGLU (the layer's `n_ff`).
    pub aq: *const u8,
    /// The most rows of an expert the buffer holds.
    pub n_ff: usize,
    /// Whether the pool sets each layer's share from the costs (else `thr` stays).
    pub auto: bool,
    /// The most share `auto` gives a layer, of 1024.
    pub cap: u32,
    /// The link's time per byte of an expert the GPU reads over it.
    pub link_us_per_byte: f32,
    /// `publish` gives the pool every host expert of a step with at most this many (0: none
    /// past the share), and `auto` learns only from the other jobs ([`all_upto`]).
    pub all_upto: u32,
}

// SAFETY: plain addresses of a pinned buffer; the flags order every access (see `Pool`).
unsafe impl Send for Exchange {}
unsafe impl Sync for Exchange {}

/// The experts of a step the host computes, each with the pairs `(token, slot)` that chose it,
/// in the order of their first pair.
pub fn groups(ids: &[i32], n_used: usize) -> Vec<(usize, Vec<(usize, usize)>)> {
    let mut out: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for (i, &e) in ids.iter().enumerate() {
        if e < 0 {
            continue;
        }
        let pair = (i / n_used, i % n_used);
        match out.iter_mut().find(|g| g.0 == e as usize) {
            Some(g) => g.1.push(pair),
            None => out.push((e as usize, vec![pair])),
        }
    }
    out
}

/// SwiGLU of the gate and up rows.
fn swiglu(g: f32, u: f32) -> f32 {
    g / (1.0 + (-g).exp()) * u
}

/// The activations of `pairs`' tokens, token after token.
fn gather(x: &[f32], n_embd: usize, pairs: &[(usize, usize)]) -> Vec<f32> {
    pairs
        .iter()
        .flat_map(|&(t, _)| &x[t * n_embd..][..n_embd])
        .copied()
        .collect()
}

/// A job on one thread: for every pair of `ids` (`[n_tok][n_used]`, the host's expert or -1),
/// `down(swiglu(gate x, up x))` into its row of `out` (`[n_tok][n_used][n_embd]`), from `x`
/// (`[n_tok][n_embd]`). Rows of other pairs are left alone. [`Pool`] computes the same, bit
/// for bit.
///
/// # Safety
/// `experts` must describe live host memory (see [`HostExperts::expert`]).
pub unsafe fn reference(
    experts: &HostExperts,
    n_used: usize,
    ids: &[i32],
    x: &[f32],
    out: &mut [f32],
) {
    let (ne, nf) = (experts.n_embd, experts.n_ff);
    for (e, pairs) in groups(ids, n_used) {
        let k = pairs.len();
        let xa = Q8Activations::quantize(&gather(x, ne, &pairs), ne, k).expect("x shape");
        // SAFETY: the caller's guarantee.
        let (up, gate, down) = unsafe {
            (
                experts.expert(0, e),
                experts.expert(1, e),
                experts.expert(2, e),
            )
        };
        let (mut g, mut u) = (vec![0.0f32; k * nf], vec![0.0f32; k * nf]);
        matmul_q8(experts.up_gate, gate, &xa, nf, &mut g).expect("gate shape");
        matmul_q8(experts.up_gate, up, &xa, nf, &mut u).expect("up shape");
        let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu(g, u)).collect();
        let aa = Q8Activations::quantize(&act, nf, k).expect("act shape");
        let mut o = vec![0.0f32; k * ne];
        matmul_q8(experts.down, down, &aa, ne, &mut o).expect("down shape");
        for (q, &(t, j)) in pairs.iter().enumerate() {
            out[(t * n_used + j) * ne..][..ne].copy_from_slice(&o[q * ne..][..ne]);
        }
    }
}

/// How long a pool keeps spinning after its last job before it naps between checks.
const SPIN_FOR: Duration = Duration::from_millis(50);
const NAP: Duration = Duration::from_micros(50);
/// How often the pool warns while a job waits for its "ready" flag. It never gives up on a
/// step: "done" without the rows would let the GPU's `collect` read the previous step's rows
/// (silently wrong hidden states), so a GPU stalled ahead of a publish (a long prompt chunk on
/// the device, a debugger) only costs warnings. Short in tests, which wait past it.
#[cfg(not(test))]
const READY_WARN_EVERY: Duration = Duration::from_secs(10);
#[cfg(test)]
pub(super) const READY_WARN_EVERY: Duration = Duration::from_millis(20);
/// The fewest rows a task takes, and the tasks per thread aimed for.
const MIN_ROWS: usize = 16;
const TASKS_PER_THREAD: usize = 3;

/// How a phase's rows are cut into tasks: about `tasks_per_thread` tasks per thread over all
/// the job's experts, at least `min_rows` rows each. The tasks go expert after expert and
/// thread `i` starts with task `i`, so at one task per thread each thread takes one
/// contiguous range and a job of several experts runs them side by side on a part of the
/// threads each (`legion cpu-bench pool --tasks-per-thread` compares them). Placement of work
/// only: every row is the same sum either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    pub tasks_per_thread: usize,
    pub min_rows: usize,
}

impl Default for Split {
    fn default() -> Self {
        Self {
            tasks_per_thread: TASKS_PER_THREAD,
            min_rows: MIN_ROWS,
        }
    }
}

/// Rows per task for `rows` rows of each of `groups` experts over `threads` threads.
pub fn rows_per_task(split: Split, threads: usize, groups: usize, rows: usize) -> usize {
    (rows * groups)
        .div_ceil(threads.max(1) * split.tasks_per_thread.max(1))
        .max(split.min_rows.max(1))
        .min(rows)
        .max(1)
}

/// POWER's SMT priority hints (`or 1,1,1`: low, `or 2,2,2`: medium): a spinning thread
/// gives its core's dispatch slots to the core's other threads. Elsewhere a spin hint.
#[inline(always)]
fn spin_low() {
    #[cfg(target_arch = "powerpc64")]
    // SAFETY: a priority hint; no memory, registers or flags change.
    unsafe {
        std::arch::asm!("or 1,1,1", options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "powerpc64"))]
    std::hint::spin_loop();
}

#[inline(always)]
fn spin_done() {
    #[cfg(target_arch = "powerpc64")]
    // SAFETY: as `spin_low`.
    unsafe {
        std::arch::asm!("or 2,2,2", options(nomem, nostack, preserves_flags));
    }
}

/// One expert of the current job: its pairs, activations and intermediate rows.
struct Group {
    expert: usize,
    pairs: Vec<(usize, usize)>,
    /// Phase 0's output: the expert's tokens' activations.
    xa: UnsafeCell<Option<Q8Activations>>,
    /// Phase 1's output: `[k][n_ff]` SwiGLU values.
    act: UnsafeCell<Vec<f32>>,
    /// `act`'s data, which phase 1's tasks write row ranges of.
    act_ptr: *mut f32,
    /// Phase 2's output: `act` quantized.
    aa: UnsafeCell<Option<Q8Activations>>,
}

/// A row range of a group, one task of phase 1 or 3.
#[derive(Clone, Copy)]
struct Task {
    group: usize,
    r0: usize,
    r1: usize,
}

/// The current job, written by the coordinator between phases only; in a phase, each task's
/// thread writes its own group's fields (phases 0, 2) or rows (1, 3) through raw pointers.
#[derive(Default)]
struct Current {
    job: Option<Job>,
    /// An exact job ([`run_exact`]): phases 1 and 3 only, with the GPU kernels' bits.
    exact: bool,
    groups: Vec<Group>,
    tasks: Vec<Task>,
    n_tasks: usize,
}

struct Cell(UnsafeCell<Current>);

// SAFETY: see `Current`; its raw pointers point into its own vectors or the exchange buffer.
unsafe impl Sync for Cell {}
unsafe impl Send for Cell {}

struct Shared {
    /// Per device, its jobs in order.
    queues: Mutex<Vec<VecDeque<(Job, Instant)>>>,
    pending: AtomicUsize,
    /// Bumped for every phase the helpers run (the phase in the low two bits).
    phase_gen: AtomicU64,
    /// The next task to claim, and the threads done with the phase.
    next: AtomicUsize,
    finished: AtomicUsize,
    current: Cell,
    /// Set by [`Pool::stop`]: the coordinator returns between jobs.
    stop: AtomicBool,
    /// Set when the coordinator returned: the helpers return (never in the middle of a job,
    /// whose phases need every thread).
    quit: AtomicBool,
    threads: usize,
    split: Split,
}

/// Raises its flag when dropped (the coordinator's `quit`, also when it panics).
struct RaiseOnDrop<'a>(&'a AtomicBool);

impl Drop for RaiseOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A NUMA node's pool: `threads` threads pinned one per CPU of `cpus` (cycled), the first
/// coordinating. Jobs of each device run in order; of several devices, whichever is ready.
pub struct Pool {
    shared: Arc<Shared>,
    handles: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Pool {
    pub fn new(cpus: &[u32], threads: usize) -> Self {
        Self::with_split(cpus, threads, Split::default())
    }

    /// [`Self::new`] with another cut of the phases into tasks (the pool benchmark's).
    pub fn with_split(cpus: &[u32], threads: usize, split: Split) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queues: Mutex::new(Vec::new()),
            pending: AtomicUsize::new(0),
            phase_gen: AtomicU64::new(0),
            next: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            current: Cell(UnsafeCell::new(Current::default())),
            stop: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            threads,
            split,
        });
        let handles = (0..threads)
            .map(|i| {
                let s = shared.clone();
                let cpu = (!cpus.is_empty()).then(|| cpus[i % cpus.len()]);
                std::thread::Builder::new()
                    .name(format!("cpu-share-{i}"))
                    .spawn(move || {
                        if let Some(c) = cpu {
                            // Best effort: unpinned, the thread still works.
                            let _ = crate::hw::numa::pin_current_thread(&[c]);
                        }
                        if i == 0 {
                            let _quit = RaiseOnDrop(&s.quit);
                            coordinate(&s);
                        } else {
                            help(&s, i);
                        }
                    })
                    .expect("spawning a cpu-share thread")
            })
            .collect();
        Self {
            shared,
            handles: Mutex::new(handles),
        }
    }

    /// Stops the threads (a job they run finishes first), then raises "done" for every job
    /// still queued, rows not computed: the one case the pool lets a step through without its
    /// rows, so that a GPU stream still waiting at a `collect` drains instead of hanging the
    /// teardown (the share is going away; nothing reads those steps' outputs). Idempotent;
    /// the queued jobs' exchange buffers must still be live, so the share stops its pools
    /// before it frees them (`ops::clear`).
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
        for h in self.handles.lock().unwrap().drain(..) {
            let _ = h.join();
        }
        let mut released = 0usize;
        for queue in self.shared.queues.lock().unwrap().iter_mut() {
            for (job, _) in queue.drain(..) {
                // SAFETY: the exchange buffer is live (the caller's order, see above).
                unsafe {
                    if let Some(e) = job.exchange.exact {
                        (*e.gu_done).store(job.seq, Ordering::Release);
                    }
                    (*job.exchange.done).store(job.seq, Ordering::Release);
                }
                released += 1;
            }
        }
        self.shared.pending.store(0, Ordering::Release);
        if released > 0 {
            tracing::warn!(
                jobs = released,
                "cpu share: stopped with queued steps; released them without their rows"
            );
        }
    }

    /// A new job queue (one per device).
    pub fn add_queue(&self) -> usize {
        let mut q = self.shared.queues.lock().unwrap();
        q.push(VecDeque::new());
        q.len() - 1
    }

    /// Queues `job` on `queue`; it runs once the GPU raises its "ready" flag.
    pub fn submit(&self, queue: usize, job: Job) {
        self.shared.queues.lock().unwrap()[queue].push_back((job, Instant::now()));
        self.shared.pending.fetch_add(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Whether `flag` reached `seq` (sequence numbers wrap).
fn reached(flag: &AtomicU32, seq: u32) -> bool {
    flag.load(Ordering::Acquire).wrapping_sub(seq) < u32::MAX / 2
}

/// Spins (at low priority) while recently busy, naps otherwise, until `ready` says go; false
/// when `stop` is raised first.
fn wait_for(stop: &AtomicBool, mut ready: impl FnMut() -> bool) -> bool {
    let busy = Instant::now();
    let mut spins = 0u32;
    loop {
        if ready() {
            spin_done();
            return true;
        }
        spins = spins.wrapping_add(1);
        if spins.is_multiple_of(1024) {
            if stop.load(Ordering::Acquire) {
                spin_done();
                return false;
            }
            if busy.elapsed() >= SPIN_FOR {
                std::thread::sleep(NAP);
                // `ready` may have acted (taken a job), so a yes here is final.
                if ready() {
                    spin_done();
                    return true;
                }
                continue;
            }
        }
        spin_low();
    }
}

/// The coordinator: takes the first queue's head job that is ready, runs it, raises "done".
/// A job runs only once its step is published, however long that takes (with a warning every
/// [`READY_WARN_EVERY`]); "done" always follows its rows.
fn coordinate(s: &Shared) {
    // The last "still waiting" warning (one per interval, of all queues).
    let mut warned = Instant::now();
    loop {
        if !wait_for(&s.stop, || s.pending.load(Ordering::Acquire) > 0) {
            return;
        }
        let mut picked = None;
        if !wait_for(&s.stop, || {
            let mut q = s.queues.lock().unwrap();
            for queue in q.iter_mut() {
                let Some(&(job, at)) = queue.front() else {
                    continue;
                };
                // SAFETY: the flags live in the exchange buffer while jobs use it.
                let ready = unsafe { &*job.exchange.ready };
                if reached(ready, job.seq) {
                    queue.pop_front();
                    picked = Some(job);
                    return true;
                }
                let waited = at.elapsed();
                if waited >= READY_WARN_EVERY && warned.elapsed() >= READY_WARN_EVERY {
                    warned = Instant::now();
                    tracing::warn!(
                        seq = job.seq,
                        layer = job.layer,
                        waited_s = waited.as_secs(),
                        "cpu share: still waiting for the GPU to publish a step"
                    );
                }
            }
            false
        }) {
            return;
        }
        let job = picked.expect("a picked job");
        s.pending.fetch_sub(1, Ordering::AcqRel);
        // SAFETY: as above.
        let (ready, done) = unsafe { (&*job.exchange.ready, &*job.exchange.done) };
        // The GPU publishes the next step of a device only after this one's `collect`, which
        // waits for this job: "ready" past the job's step means the host missed a step.
        let published = ready.load(Ordering::Acquire);
        if published != job.seq {
            tracing::error!(
                seq = job.seq,
                layer = job.layer,
                published,
                "cpu share: the GPU published a later step than the host's next job"
            );
            debug_assert_eq!(published, job.seq, "cpu share: a step was skipped");
        }
        if job.exchange.exact.is_some() {
            let st = run_exact(s, &job);
            done.store(job.seq, Ordering::Release);
            laps::record(job.seq, job.layer, st.experts, st.laps);
            exact_stats(&st);
            continue;
        }
        let start = Instant::now();
        let pairs = run_job(s, &job);
        let compute = start.elapsed();
        // SAFETY: as above (null when the buffer has no such flag: tests).
        let waited = unsafe { job.exchange.arrived.as_ref() }.is_some_and(|a| reached(a, job.seq));
        done.store(job.seq, Ordering::Release);
        stats(compute, pairs, waited);
    }
}

/// Debug statistics (`RUST_LOG=legion::residency::cpu_share=debug`), per 4,096 jobs: the mean
/// time per job, host pairs per job, and the share of jobs the GPU had to wait for.
fn stats(compute: Duration, pairs: usize, waited: bool) {
    static ACC: Mutex<(u64, f64, u64, u64)> = Mutex::new((0, 0.0, 0, 0));
    let mut a = ACC.lock().unwrap();
    a.0 += 1;
    a.1 += compute.as_secs_f64() * 1e6;
    a.2 += pairs as u64;
    a.3 += u64::from(waited);
    if a.0.is_multiple_of(4096) {
        let n = a.0 as f64;
        tracing::debug!(
            jobs = a.0,
            compute_us = a.1 / n,
            pairs = a.2 as f64 / n,
            gpu_waited = a.3 as f64 / n,
            "cpu share jobs"
        );
        *a = (0, 0.0, 0, 0);
    }
}

/// What an exact job did ([`run_exact`]), for [`exact_stats`].
#[derive(Debug, Default)]
struct ExactStats {
    /// The step's distinct host experts, and those the pool computed (with their pairs).
    miss: u32,
    experts: usize,
    pairs: usize,
    gate_up: Duration,
    /// From "gate/up done" to "act ready".
    wait: Duration,
    down: Duration,
    /// Whether the GPU reached `activate` / `collect` before the pool was done (`activate`'s
    /// with `auto` or debug logging, `collect`'s with debug logging only).
    gpu_waited_act: bool,
    gpu_waited: bool,
    /// The layer's share after the job.
    thr: u32,
    /// [`laps`]' clock (0 when off) when the job was picked, at "gate/up done", when "act
    /// ready" was seen and when the down rows were written.
    laps: [u64; 4],
    /// The experts the pool computed for this layer in its previous job, and how many of
    /// them it computes again ([`repeats`]).
    repeat_of: usize,
    repeated: usize,
}

/// Per layer, the experts of the pool's last job (debug logging only, [`repeats`]).
static LAST_EXPERTS: Mutex<Vec<Vec<usize>>> = Mutex::new(Vec::new());

/// With debug logging on: the experts the pool computed for `layer` in its previous job and
/// how many of them `groups` has again, i.e. how well "the same experts as the last step"
/// would guess this job's experts ahead of it (#37 round 21). `(0, 0)` otherwise.
fn repeats(layer: usize, groups: &[(usize, Vec<(usize, usize)>)]) -> (usize, usize) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return (0, 0);
    }
    let mut last = LAST_EXPERTS.lock().unwrap();
    if last.len() <= layer {
        last.resize(layer + 1, Vec::new());
    }
    let prev = std::mem::replace(&mut last[layer], groups.iter().map(|g| g.0).collect());
    let again = prev
        .iter()
        .filter(|e| groups.iter().any(|g| g.0 == **e))
        .count();
    (prev.len(), again)
}

/// Whether the pool prefetches the down rows while it waits for the GPU's SwiGLU blocks (on;
/// `LEGION_DEBUG_SHARE_PREFETCH=0` turns it off, for A/B runs). The same rows and sums either
/// way.
fn prefetch_down() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGION_DEBUG_SHARE_PREFETCH").map_or(true, |v| v != "0"))
}

/// Per-job timestamps of the exact share for a decode attribution (`LEGION_DEBUG_SHARE_LAPS=
/// <file>`): one line a job, `seq layer experts picked gu_done act_seen down_done`, in
/// CLOCK_REALTIME nanoseconds (the clock a CUPTI trace's host stamps use), appended in batches.
pub(super) mod laps {
    use std::io::Write;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    fn path() -> Option<&'static str> {
        static P: OnceLock<Option<String>> = OnceLock::new();
        P.get_or_init(|| {
            std::env::var("LEGION_DEBUG_SHARE_LAPS")
                .ok()
                .filter(|p| !p.is_empty())
        })
        .as_deref()
    }

    /// The clock in nanoseconds, or 0 when the laps are off.
    pub fn now() -> u64 {
        if path().is_none() {
            return 0;
        }
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    }

    /// Records one job (when on), writing the batch out every 16,384 jobs.
    pub fn record(seq: u32, layer: u32, experts: usize, t: [u64; 4]) {
        let Some(p) = path() else {
            return;
        };
        static BUF: Mutex<Vec<[u64; 7]>> = Mutex::new(Vec::new());
        let mut b = BUF.lock().unwrap();
        b.push([
            u64::from(seq),
            u64::from(layer),
            experts as u64,
            t[0],
            t[1],
            t[2],
            t[3],
        ]);
        if b.len() >= 16_384 {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
            {
                let mut s = String::with_capacity(b.len() * 80);
                for r in b.iter() {
                    s.push_str(&format!(
                        "{} {} {} {} {} {} {}\n",
                        r[0], r[1], r[2], r[3], r[4], r[5], r[6]
                    ));
                }
                let _ = f.write_all(s.as_bytes());
            }
            b.clear();
        }
    }
}

/// How fast a layer's cost per expert follows its jobs (a running mean over about 16).
const COST_EMA: f64 = 1.0 / 16.0;

/// How far one job moves an `auto` layer's share, of [`super::THRESHOLD_ONE`].
pub const NUDGE: u32 = 4;

/// An `auto` layer's next share after a job whose misses were split at `thr`: less when the
/// GPU reached `activate` before the pool's gate and up sums were done (it waited for the
/// pool), more when the pool finished first (the GPU's side was the later one). The first
/// split ([`split`]) prices the link from a timed copy, which leaves out the GPU's own work
/// on its hits and how fast its kernels read mapped memory, so after it the race itself sets
/// the share: it settles where each side waits for the other about as often. Only jobs the
/// pool took experts in move it (a job without any says nothing about the race), so it stays
/// at [`AUTO_MIN`] or more: a share near 0 would hardly ever give the pool a job to learn from.
pub fn nudge(thr: u32, gpu_waited: bool, cap: u32) -> u32 {
    let next = if gpu_waited {
        thr.saturating_sub(NUDGE)
    } else {
        thr + NUDGE
    };
    next.max(AUTO_MIN).min(cap)
}

/// The least share [`nudge`] leaves a layer, of [`super::THRESHOLD_ONE`] (3%).
pub const AUTO_MIN: u32 = 32;

/// [`ExactExchange::all_upto`] with `auto` where every routed expert is in host memory (no
/// bounded host tier, `tiered` false): a decode step's layer has 2-3 host experts, and the
/// pool computing all of them beat splitting them with the link (the GPU reading the rest
/// over PCIe while the pool reads the same DRAM) on a 4090 with 128 GB. Steps with more (four
/// chats at once) race as before. 0 elsewhere. `LEGION_DEBUG_SHARE_ALL_UPTO=<n>` sets it (0:
/// off), for A/B runs.
pub fn all_upto(auto: bool, tiered: bool) -> u32 {
    let debug = std::env::var("LEGION_DEBUG_SHARE_ALL_UPTO")
        .ok()
        .and_then(|v| v.parse().ok());
    all_upto_with(auto, tiered, debug)
}

/// [`all_upto`] with the debug value given.
pub fn all_upto_with(auto: bool, tiered: bool, debug: Option<u32>) -> u32 {
    match debug {
        Some(n) if auto => n,
        _ if auto && !tiered => ALL_UPTO,
        _ => 0,
    }
}

/// [`all_upto`]'s default.
pub const ALL_UPTO: u32 = 3;

/// Runs an exact job (see the `cpu_share` module docs): the pairs' gate and up sums, "gate/up
/// done", the wait for the GPU's SwiGLU blocks, the down rows. The caller raises "done".
fn run_exact(s: &Shared, job: &Job) -> ExactStats {
    let ex = job.exchange;
    let e = ex.exact.expect("an exact exchange");
    let layer = job.layer as usize;
    let (n_tok, n_used) = (job.n_tok.min(ex.max_tok), job.n_used.min(ex.max_used));
    // SAFETY: the GPU wrote the ids and the count before raising "ready" (acquired); nothing
    // writes them until this job's "done". The flags live in the buffer.
    let (ids, gu_done, act_ready, miss) = unsafe {
        (
            std::slice::from_raw_parts(ex.ids, n_tok * n_used),
            &*e.gu_done,
            &*e.act_ready,
            (*e.n_miss).load(Ordering::Acquire),
        )
    };
    let thr_of = |il: usize| {
        // SAFETY: a word of the buffer's `thr` (in range).
        (il < e.n_layers).then(|| unsafe { &*e.thr.add(il) }.load(Ordering::Relaxed))
    };
    let mut st = ExactStats {
        miss,
        thr: thr_of(layer).unwrap_or(0),
        laps: [laps::now(), 0, 0, 0],
        ..Default::default()
    };
    let groups = groups(ids, n_used);
    (st.repeat_of, st.repeated) = repeats(layer, &groups);
    if groups.is_empty() {
        gu_done.store(job.seq, Ordering::Release);
        return st;
    }
    st.experts = groups.len();
    st.pairs = groups.iter().map(|g| g.1.len()).sum();
    let (ne, nf) = (job.experts.n_embd, job.experts.n_ff);
    // SAFETY: no phase runs (the helpers wait for `phase_gen`): this thread has it alone.
    unsafe {
        let c = &mut *s.current.0.get();
        c.job = Some(*job);
        c.exact = true;
        c.groups = groups
            .into_iter()
            .map(|(expert, pairs)| Group {
                act_ptr: std::ptr::null_mut(),
                act: UnsafeCell::new(Vec::new()),
                expert,
                pairs,
                xa: UnsafeCell::new(None),
                aa: UnsafeCell::new(None),
            })
            .collect();
    }
    let t = Instant::now();
    split_rows(s, nf);
    run_phase(s, 1);
    st.gate_up = t.elapsed();
    // SAFETY: the flag lives in the buffer.
    st.gpu_waited_act = reached(unsafe { &*e.act_arrived }, job.seq);
    gu_done.store(job.seq, Ordering::Release);
    st.laps[1] = laps::now();
    let t = Instant::now();
    split_rows(s, ne);
    // While the GPU makes the host pairs' SwiGLU blocks, every thread pulls the down rows of
    // its first task into its cache ([`prefetch_down`]); placement of reads only.
    if prefetch_down() {
        run_phase(s, 2);
    }
    // The GPU's `activate` follows its SwiGLU op at once; stopping leaves the rows undone (the
    // share is going away, see `Pool::stop`).
    if !wait_for(&s.stop, || reached(act_ready, job.seq)) {
        return st;
    }
    st.wait = t.elapsed();
    st.laps[2] = laps::now();
    let t = Instant::now();
    run_phase(s, 3);
    st.down = t.elapsed();
    st.laps[3] = laps::now();
    // SAFETY: as above (null when the buffer has no such flag: tests).
    st.gpu_waited = unsafe { ex.arrived.as_ref() }.is_some_and(|a| reached(a, job.seq));
    // A step the pool takes whole ([`ExactExchange::all_upto`]) says nothing about the race.
    if e.auto && layer < e.n_layers && miss > e.all_upto {
        let now = (st.gate_up + st.down).as_secs_f64() * 1e6 / st.experts as f64;
        // SAFETY: the layer's words of the buffer; only this thread writes `cost`.
        unsafe {
            let cost = e.cost.add(layer);
            let prev = f64::from(*cost);
            let c = if prev > 0.0 {
                prev + (now - prev) * COST_EMA
            } else {
                now
            };
            *cost = c as f32;
            let thr = if prev > 0.0 {
                nudge(st.thr, st.gpu_waited_act, e.cap)
            } else {
                let bytes: usize = job.experts.expert_bytes.iter().sum();
                split(f64::from(e.link_us_per_byte) * bytes as f64, c, e.cap)
            };
            (*e.thr.add(layer)).store(thr, Ordering::Relaxed);
            st.thr = thr;
        }
    }
    st
}

/// The share of 1024 where `c` microseconds a pool expert and `l` a link expert finish together
/// (`l / (l + c)` of the misses on the pool), at most `cap`.
pub fn split(l: f64, c: f64, cap: u32) -> u32 {
    if !(l > 0.0) || !(c >= 0.0) {
        return 0;
    }
    ((l / (l + c) * super::THRESHOLD_ONE as f64).round() as u32).min(cap)
}

/// Debug statistics of the exact share (`RUST_LOG=legion::residency::cpu_share=debug`), per
/// 4,096 jobs: misses per job and the pool's part of them, the pool's phases and its wait for
/// the GPU's SwiGLU blocks, microseconds per expert, the jobs the GPU waited for (debug only)
/// and the mean share.
fn exact_stats(st: &ExactStats) {
    #[derive(Default)]
    struct Acc {
        jobs: u64,
        busy: u64,
        miss: u64,
        experts: u64,
        pairs: u64,
        gate_up: f64,
        wait: f64,
        down: f64,
        waited_act: u64,
        waited: u64,
        thr: u64,
        repeat_of: u64,
        repeated: u64,
    }
    static ACC: Mutex<Option<Acc>> = Mutex::new(None);
    let mut g = ACC.lock().unwrap();
    let a = g.get_or_insert_with(Acc::default);
    a.jobs += 1;
    a.busy += u64::from(st.experts > 0);
    a.miss += u64::from(st.miss);
    a.experts += st.experts as u64;
    a.pairs += st.pairs as u64;
    a.gate_up += st.gate_up.as_secs_f64() * 1e6;
    a.wait += st.wait.as_secs_f64() * 1e6;
    a.down += st.down.as_secs_f64() * 1e6;
    a.waited_act += u64::from(st.gpu_waited_act);
    a.waited += u64::from(st.gpu_waited);
    a.thr += u64::from(st.thr);
    a.repeat_of += st.repeat_of as u64;
    a.repeated += st.repeated as u64;
    if a.jobs.is_multiple_of(4096) {
        let n = a.jobs as f64;
        let busy = a.busy.max(1) as f64;
        tracing::debug!(
            jobs = a.jobs,
            misses = a.miss as f64 / n,
            pool_experts = a.experts as f64 / n,
            pool_pairs = a.pairs as f64 / n,
            gate_up_us = a.gate_up / busy,
            act_wait_us = a.wait / busy,
            down_us = a.down / busy,
            us_per_expert = (a.gate_up + a.down) / a.experts.max(1) as f64,
            gpu_waited_act = a.waited_act as f64 / n,
            gpu_waited = a.waited as f64 / n,
            share = a.thr as f64 / n / super::THRESHOLD_ONE as f64,
            // Of the layer's previous job's experts, the share computed again (precision of
            // that guess), and the share of this job's experts it names (recall).
            repeat_precision = a.repeated as f64 / a.repeat_of.max(1) as f64,
            repeat_recall = a.repeated as f64 / a.experts.max(1) as f64,
            "cpu share exact jobs"
        );
        *a = Acc::default();
    }
}

/// Runs `job` over the pool; returns its host pairs.
fn run_job(s: &Shared, job: &Job) -> usize {
    let ex = job.exchange;
    let (n_tok, n_used) = (job.n_tok.min(ex.max_tok), job.n_used.min(ex.max_used));
    // SAFETY: the GPU wrote the ids before raising "ready" (acquired); nothing writes them
    // until this job's "done".
    let ids = unsafe { std::slice::from_raw_parts(ex.ids, n_tok * n_used) };
    let groups = groups(ids, n_used);
    if groups.is_empty() {
        return 0;
    }
    let pairs = groups.iter().map(|g| g.1.len()).sum();
    let (ne, nf) = (job.experts.n_embd, job.experts.n_ff);
    // SAFETY: no phase runs (the helpers wait for `phase_gen`): this thread has it alone.
    unsafe {
        let c = &mut *s.current.0.get();
        c.job = Some(*job);
        c.exact = false;
        c.groups = groups
            .into_iter()
            .map(|(expert, pairs)| {
                let mut act = vec![0.0; pairs.len() * nf];
                Group {
                    act_ptr: act.as_mut_ptr(),
                    act: UnsafeCell::new(act),
                    expert,
                    pairs,
                    xa: UnsafeCell::new(None),
                    aa: UnsafeCell::new(None),
                }
            })
            .collect();
        c.n_tasks = c.groups.len();
    }
    run_phase(s, 0);
    split_rows(s, nf);
    run_phase(s, 1);
    // SAFETY: as above.
    unsafe { (*s.current.0.get()).n_tasks = (*s.current.0.get()).groups.len() };
    run_phase(s, 2);
    split_rows(s, ne);
    run_phase(s, 3);
    pairs
}

/// Splits every group's `rows` into tasks of [`rows_per_task`] rows (the pool's [`Split`]).
fn split_rows(s: &Shared, rows: usize) {
    // SAFETY: between phases (see `run_job`).
    let c = unsafe { &mut *s.current.0.get() };
    let per = rows_per_task(s.split, s.threads, c.groups.len(), rows);
    c.tasks.clear();
    for g in 0..c.groups.len() {
        for r0 in (0..rows).step_by(per) {
            c.tasks.push(Task {
                group: g,
                r0,
                r1: rows.min(r0 + per),
            });
        }
    }
    c.n_tasks = c.tasks.len();
}

/// Runs one phase on every thread (this one included) and returns when all are done.
fn run_phase(s: &Shared, phase: u64) {
    s.finished.store(0, Ordering::Relaxed);
    s.next.store(s.threads, Ordering::Relaxed);
    let g = s.phase_gen.load(Ordering::Relaxed);
    s.phase_gen
        .store((((g >> 2) + 1) << 2) | phase, Ordering::Release);
    work(s, 0, phase);
    s.finished.fetch_add(1, Ordering::AcqRel);
    while s.finished.load(Ordering::Acquire) < s.threads {
        spin_low();
    }
    spin_done();
}

fn help(s: &Shared, i: usize) {
    let mut seen = 0u64;
    loop {
        if !wait_for(&s.quit, || s.phase_gen.load(Ordering::Acquire) != seen) {
            return;
        }
        seen = s.phase_gen.load(Ordering::Acquire);
        work(s, i, seen & 3);
        s.finished.fetch_add(1, Ordering::AcqRel);
    }
}

/// Thread `i`'s tasks of the current phase: task `i` first, then claimed ones.
fn work(s: &Shared, i: usize, phase: u64) {
    // SAFETY: in a phase every thread only reads the current job; outputs go through raw
    // pointers to fields or rows of the task's own.
    let c = unsafe { &*s.current.0.get() };
    if phase == 2 && c.exact {
        // The prefetch: each thread the task it takes first in phase 3 (task `i`), no claims.
        if i < c.n_tasks {
            prefetch_task(c, i);
        }
        return;
    }
    let mut task = i;
    while task < c.n_tasks {
        // SAFETY: the task is this thread's alone (claimed once).
        unsafe { run_task(c, phase, task) };
        task = s.next.fetch_add(1, Ordering::Relaxed);
    }
}

/// Task `task` of phase `phase`: 0 quantizes a group's activations, 1 computes a row range of
/// its gate, up and SwiGLU, 2 quantizes the SwiGLU values, 3 computes a row range of down into
/// the exchange buffer.
///
/// # Safety
/// The task must be the calling thread's alone in the phase.
unsafe fn run_task(c: &Current, phase: u64, task: usize) {
    let job = c.job.as_ref().expect("a job");
    let ex = &job.experts;
    let (ne, nf) = (ex.n_embd, ex.n_ff);
    if c.exact {
        // SAFETY: the caller's guarantee.
        unsafe { run_exact_task(c, job, phase, task) };
        return;
    }
    match phase {
        0 | 2 => {
            let g = &c.groups[task];
            let k = g.pairs.len();
            let q = if phase == 0 {
                // SAFETY: the GPU wrote x before raising "ready".
                let x = unsafe { std::slice::from_raw_parts(job.exchange.x, job.n_tok * ne) };
                Q8Activations::quantize(&gather(x, ne, &g.pairs), ne, k).expect("x shape")
            } else {
                // SAFETY: phase 1 (which wrote it) is over.
                let act = unsafe { &*g.act.get() };
                Q8Activations::quantize(act, nf, k).expect("act shape")
            };
            let slot = if phase == 0 { &g.xa } else { &g.aa };
            // SAFETY: the group's field is this task's alone; no one reads it in this phase.
            unsafe { *slot.get() = Some(q) };
        }
        1 => {
            let Task { group, r0, r1 } = c.tasks[task];
            let g = &c.groups[group];
            let (k, n) = (g.pairs.len(), r1 - r0);
            let rb = ex.up_gate.row_bytes(ne).expect("gate rows");
            // SAFETY: live host experts (the pool's owner keeps them mapped).
            let (gate, up) = unsafe { (ex.expert(1, g.expert), ex.expert(0, g.expert)) };
            // SAFETY: phase 0 (which wrote it) is over.
            let xa = unsafe { &*g.xa.get() }.as_ref().expect("phase 0 ran");
            let (mut gv, mut uv) = (vec![0.0f32; k * n], vec![0.0f32; k * n]);
            matmul_q8(ex.up_gate, &gate[r0 * rb..r1 * rb], xa, n, &mut gv).expect("gate");
            matmul_q8(ex.up_gate, &up[r0 * rb..r1 * rb], xa, n, &mut uv).expect("up");
            let act = g.act_ptr;
            for q in 0..k {
                for r in 0..n {
                    let v = swiglu(gv[q * n + r], uv[q * n + r]);
                    // SAFETY: rows `r0..r1` of the group's act are this task's alone.
                    unsafe { *act.add(q * nf + r0 + r) = v };
                }
            }
        }
        _ => {
            let Task { group, r0, r1 } = c.tasks[task];
            let g = &c.groups[group];
            let (k, n) = (g.pairs.len(), r1 - r0);
            let rb = ex.down.row_bytes(nf).expect("down rows");
            // SAFETY: as above.
            let down = unsafe { ex.expert(2, g.expert) };
            // SAFETY: phase 2 (which wrote it) is over.
            let aa = unsafe { &*g.aa.get() }.as_ref().expect("phase 2 ran");
            let mut o = vec![0.0f32; k * n];
            matmul_q8(ex.down, &down[r0 * rb..r1 * rb], aa, n, &mut o).expect("down");
            for (q, &(t, j)) in g.pairs.iter().enumerate() {
                // SAFETY: rows `r0..r1` of pair (t, j)'s output row are this task's alone;
                // the GPU reads them only after "done".
                unsafe {
                    let dst = job.exchange.out.add((t * job.n_used + j) * ne + r0);
                    std::ptr::copy_nonoverlapping(o.as_ptr().add(q * n), dst, n);
                }
            }
        }
    }
}

/// An exact job's prefetch (phase 2): the down rows of phase 3's task `task` into this thread's
/// caches, a cache line at a time, until done or the GPU raised "act ready" (then phase 3 starts
/// at once and reads the rest from memory as without it). Reads only: no row or sum changes.
fn prefetch_task(c: &Current, task: usize) {
    let Some(job) = c.job.as_ref() else {
        return;
    };
    let Some(e) = job.exchange.exact else {
        return;
    };
    let ex = &job.experts;
    let Task { group, r0, r1 } = c.tasks[task];
    let Some(rb) = ex.down.row_bytes(ex.n_ff) else {
        return;
    };
    // SAFETY: live host experts (the pool's owner keeps them mapped); the flag lives in the
    // exchange buffer.
    let (down, act_ready) = unsafe { (ex.expert(2, c.groups[group].expert), &*e.act_ready) };
    let rows = &down[r0 * rb..r1 * rb];
    const LINE: usize = 64;
    const CHECK_EVERY: usize = 32;
    for (n, at) in (0..rows.len()).step_by(LINE).enumerate() {
        if n % CHECK_EVERY == 0 && reached(act_ready, job.seq) {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a prefetch hint of an address inside `rows`; it never faults or writes.
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T0>(rows.as_ptr().add(at).cast());
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = at;
    }
}

/// Task `task` of an exact job's phase `phase`: 1 computes a row range of a group's gate and up
/// sums for each of its pairs from x's blocks, 3 a row range of its down rows from the pairs'
/// SwiGLU blocks, into the exchange buffer.
///
/// # Safety
/// The task must be the calling thread's alone in the phase.
unsafe fn run_exact_task(c: &Current, job: &Job, phase: u64, task: usize) {
    use crate::cpu::gpu_exact::{Q8_1_BLOCK_BYTES, matvec};
    let ex = &job.experts;
    let e = job.exchange.exact.expect("an exact exchange");
    let (ne, nf) = (ex.n_embd, ex.n_ff);
    let Task { group, r0, r1 } = c.tasks[task];
    let g = &c.groups[group];
    let n = r1 - r0;
    let (xq_bytes, aq_bytes) = (ne / 32 * Q8_1_BLOCK_BYTES, nf / 32 * Q8_1_BLOCK_BYTES);
    if phase == 1 {
        let rb = ex.up_gate.row_bytes(ne).expect("gate rows");
        // SAFETY: live host experts (the pool's owner keeps them mapped).
        let (gate, up) = unsafe { (ex.expert(1, g.expert), ex.expert(0, g.expert)) };
        for &(t, j) in &g.pairs {
            let pair = t * job.n_used + j;
            // SAFETY: the GPU wrote x's blocks before raising "ready"; rows `r0..r1` of the
            // pair's sums are this task's alone, and the GPU reads them after "gate/up done".
            unsafe {
                let xq = std::slice::from_raw_parts(e.xq.add(t * xq_bytes), xq_bytes);
                let sums = e.gu.add(pair * 2 * nf);
                let g_out = std::slice::from_raw_parts_mut(sums.add(r0), n);
                matvec(ex.up_gate, &gate[r0 * rb..r1 * rb], xq, ne, g_out).expect("gate");
                let u_out = std::slice::from_raw_parts_mut(sums.add(nf + r0), n);
                matvec(ex.up_gate, &up[r0 * rb..r1 * rb], xq, ne, u_out).expect("up");
            }
        }
    } else {
        let rb = ex.down.row_bytes(nf).expect("down rows");
        // SAFETY: as above.
        let down = unsafe { ex.expert(2, g.expert) };
        for &(t, j) in &g.pairs {
            let pair = t * job.n_used + j;
            // SAFETY: the GPU wrote the pair's blocks before raising "act ready"; rows `r0..r1`
            // of the pair's output row are this task's alone; the GPU reads them after "done".
            unsafe {
                let aq = std::slice::from_raw_parts(e.aq.add(pair * aq_bytes), aq_bytes);
                let out = std::slice::from_raw_parts_mut(job.exchange.out.add(pair * ne + r0), n);
                matvec(ex.down, &down[r0 * rb..r1 * rb], aq, nf, out).expect("down");
            }
        }
    }
}

/// Runs `job` on `pool` as the GPU would (raises "ready", waits for "done"), with the job's
/// exchange buffer in plain host memory: the unit tests and `legion cpu-bench pool`.
///
/// # Safety
/// The job's exchange buffer and experts must outlive the call.
pub unsafe fn run_now(pool: &Pool, queue: usize, job: Job) {
    // SAFETY: the caller's guarantee.
    let (ready, done) = unsafe { (&*job.exchange.ready, &*job.exchange.done) };
    pool.submit(queue, job);
    ready.store(job.seq, Ordering::Release);
    let start = Instant::now();
    while !reached(done, job.seq) {
        assert!(start.elapsed() < Duration::from_secs(30), "pool stuck");
        std::hint::spin_loop();
    }
}
