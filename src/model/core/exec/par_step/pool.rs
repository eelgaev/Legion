//! The parallel steps' launch threads: one per GPU, kept for the executor's life (#669).
//!
//! A thread spawned per GPU per step started each GPU's graph about 0.9 ms after the one before
//! (the spawns are serial and slow on a process mapping this much memory), so the last GPU
//! began some 2.7 ms after the first and every exchange of the step waited for it. Kept threads
//! take their graphs within microseconds of each other.

use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::ggml::GgmlError;

/// How long the calling thread spins for the launch threads' results before it sleeps
/// ([`recv_spinning`]): longer than a decode or verify step (30-60 ms on four V100s).
const SPIN_FOR: Duration = Duration::from_millis(250);

/// A launch thread's result: polled without sleeping until `deadline`, then waited for (none
/// if the thread is gone). The engine thread waits through a whole step (about 30 ms) for
/// the threads that launch and synchronize its graphs; asleep that long, its CPU went idle
/// deep enough that it woke about 0.55 ms after the last GPU was done, every step (#676,
/// nsys on POWER9). Spinning, it reads the step's outputs at once.
pub(super) fn recv_spinning<T>(rx: &Receiver<T>, deadline: Instant) -> Option<T> {
    loop {
        match rx.try_recv() {
            Ok(v) => return Some(v),
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) if Instant::now() < deadline => std::hint::spin_loop(),
            Err(TryRecvError::Empty) => return rx.recv().ok(),
        }
    }
}

/// A graph's launch ([`crate::ggml::Launcher::run`]).
pub(in crate::model::core::exec) type Launch = Box<dyn FnOnce() -> Result<(), GgmlError> + Send>;

/// A launch and the CPUs to run it on (empty: the thread's own).
type Job = (Vec<u32>, Launch);

struct Worker {
    jobs: Option<Sender<Job>>,
    done: Receiver<Result<(), GgmlError>>,
    thread: Option<JoinHandle<()>>,
    /// A job started ([`LaunchPool::start`]) whose result is not taken yet.
    busy: std::cell::Cell<bool>,
}

/// One launch thread per GPU ([`Self::run`], [`Self::run_beside`], or [`Self::start`] then
/// [`Self::join`]).
pub(in crate::model::core::exec) struct LaunchPool {
    workers: Vec<Worker>,
}

impl LaunchPool {
    /// `n` launch threads, idle until [`Self::run`].
    pub(in crate::model::core::exec) fn new(n: usize) -> std::io::Result<Self> {
        let mut workers = Vec::with_capacity(n);
        for d in 0..n {
            let (jobs, rx) = channel::<Job>();
            let (tx, done) = channel();
            let thread = std::thread::Builder::new()
                .name(format!("legion-par-{d}"))
                .spawn(move || {
                    let mut pinned: Vec<u32> = Vec::new();
                    while let Ok((cpus, launch)) = rx.recv() {
                        if !cpus.is_empty() && cpus != pinned {
                            // Best effort: a failed pin leaves the affinity it had.
                            let _ = crate::hw::numa::pin_current_thread(&cpus);
                            pinned = cpus;
                        }
                        if tx.send(launch()).is_err() {
                            break;
                        }
                    }
                })?;
            workers.push(Worker {
                jobs: Some(jobs),
                done,
                thread: Some(thread),
                busy: std::cell::Cell::new(false),
            });
        }
        Ok(Self { workers })
    }

    /// Launch threads it has.
    pub(in crate::model::core::exec) fn len(&self) -> usize {
        self.workers.len()
    }

    /// Runs `jobs[d]` on thread `d`, all at once, and returns once every one has: their
    /// results in order. Panics (after the others are done) if a thread panicked.
    pub(in crate::model::core::exec) fn run(&self, jobs: Vec<Job>) -> Vec<Result<(), GgmlError>> {
        assert!(jobs.len() <= self.workers.len(), "more graphs than threads");
        assert!(!self.pending(), "a run while started jobs are not joined");
        // A dead thread hands its job back unrun; the others' still run and are waited for.
        let sent: Vec<bool> = self
            .workers
            .iter()
            .zip(jobs)
            .map(|(w, job)| w.jobs.as_ref().is_some_and(|j| j.send(job).is_ok()))
            .collect();
        let deadline = Instant::now() + SPIN_FOR;
        let results: Vec<_> = self
            .workers
            .iter()
            .zip(&sent)
            .map(|(w, &sent)| sent.then(|| recv_spinning(&w.done, deadline)).flatten())
            .collect();
        results
            .into_iter()
            .map(|r| r.expect("a launch thread panicked"))
            .collect()
    }

    /// [`Self::run`] with `work` on the calling thread while the threads launch (#723): the
    /// jobs' results in order and `work`'s. The jobs start before `work` and are waited for
    /// after it, whatever it returns.
    pub(in crate::model::core::exec) fn run_beside<T>(
        &self,
        jobs: Vec<Job>,
        work: impl FnOnce() -> T,
    ) -> (Vec<Result<(), GgmlError>>, T) {
        assert!(jobs.len() <= self.workers.len(), "more graphs than threads");
        assert!(!self.pending(), "a run while started jobs are not joined");
        for (d, job) in jobs.into_iter().enumerate() {
            self.start(d, job);
        }
        let out = work();
        (self.join(), out)
    }

    /// Starts `job` on thread `d` and returns at once; [`Self::join`] takes its result. A dead
    /// thread's job is not run, and [`Self::join`] panics for it.
    pub(in crate::model::core::exec) fn start(&self, d: usize, job: Job) {
        let w = &self.workers[d];
        assert!(!w.busy.get(), "launch thread {d} started twice");
        let sent = w.jobs.as_ref().is_some_and(|j| j.send(job).is_ok());
        assert!(sent, "launch thread {d} is gone");
        w.busy.set(true);
    }

    /// Whether a job [`Self::start`]ed has not been joined yet.
    pub(in crate::model::core::exec) fn pending(&self) -> bool {
        self.workers.iter().any(|w| w.busy.get())
    }

    /// Waits for every job [`Self::start`]ed since the last join: their results in thread
    /// order (none when nothing was started). Panics (after the others are done) if a thread
    /// panicked.
    pub(in crate::model::core::exec) fn join(&self) -> Vec<Result<(), GgmlError>> {
        let deadline = Instant::now() + SPIN_FOR;
        let results: Vec<_> = self
            .workers
            .iter()
            .filter(|w| w.busy.replace(false))
            .map(|w| recv_spinning(&w.done, deadline))
            .collect();
        results
            .into_iter()
            .map(|r| r.expect("a launch thread panicked"))
            .collect()
    }
}

impl LaunchPool {
    /// Waits for whichever job [`Self::start`]ed and not joined yet is done first: its thread
    /// and result, `None` when none is left (#756: a round of groups in flight reads each
    /// stage's output as soon as that stage is done). Spins as [`recv_spinning`] does, then
    /// waits for the first such thread in order. Panics if a thread panicked.
    pub(in crate::model::core::exec) fn join_next(&self) -> Option<(usize, Result<(), GgmlError>)> {
        let deadline = Instant::now() + SPIN_FOR;
        loop {
            let mut first = None;
            for (d, w) in self.workers.iter().enumerate() {
                if !w.busy.get() {
                    continue;
                }
                first.get_or_insert(d);
                match w.done.try_recv() {
                    Ok(r) => {
                        w.busy.set(false);
                        return Some((d, r));
                    }
                    Err(TryRecvError::Disconnected) => panic!("a launch thread panicked"),
                    Err(TryRecvError::Empty) => {}
                }
            }
            let d = first?;
            if Instant::now() >= deadline {
                let w = &self.workers[d];
                w.busy.set(false);
                return Some((d, w.done.recv().expect("a launch thread panicked")));
            }
            std::hint::spin_loop();
        }
    }
}

impl std::fmt::Debug for LaunchPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LaunchPool({} threads)", self.workers.len())
    }
}

impl Drop for LaunchPool {
    fn drop(&mut self) {
        for w in &mut self.workers {
            w.jobs = None;
        }
        for w in &mut self.workers {
            if let Some(t) = w.thread.take() {
                let _ = t.join();
            }
        }
    }
}
