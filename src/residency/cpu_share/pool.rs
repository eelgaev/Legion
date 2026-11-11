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
                unsafe { &*job.exchange.done }.store(job.seq, Ordering::Release);
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

/// Splits every group's `rows` into tasks: about [`TASKS_PER_THREAD`] per thread, at least
/// [`MIN_ROWS`] rows each.
fn split_rows(s: &Shared, rows: usize) {
    // SAFETY: between phases (see `run_job`).
    let c = unsafe { &mut *s.current.0.get() };
    let total = rows * c.groups.len();
    let per = total
        .div_ceil(s.threads * TASKS_PER_THREAD)
        .max(MIN_ROWS)
        .min(rows);
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
