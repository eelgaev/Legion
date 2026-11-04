//! The tier's reader threads: each job reads one expert matrix from the checkpoint (its packed
//! codes and its sub-block scales, two reads) and repacks it straight into its slot, or copies
//! an expert matrix between slots and staging buffers.
//!
//! Reads go around the page cache (`O_DIRECT`, through an aligned buffer per thread), so the
//! cache's memory and the page cache never hold the same experts twice. Where a file cannot be
//! opened for direct reads, it is read through the page cache and its pages dropped again
//! (`POSIX_FADV_DONTNEED`) after each read.
//!
//! Jobs queue by [`Class`] (#615): the readers take a step's own reads before reads ahead, a
//! group can move up to them ([`Pool::promote`]), and a cancelled group leaves the queues at
//! once ([`Pool::cancel`]).

use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::gguf::{FileRange, Nvfp4Source};

/// Direct reads' alignment (offset, length and buffer).
pub const ALIGN: u64 = 4096;

/// One unit of work.
#[derive(Debug, Clone, Copy)]
pub enum Job {
    /// Read expert matrix `src` (`rows` x `k`) and repack it as ggml NVFP4 rows at `dst`
    /// (`rows * k / 64 * 36` bytes of host memory).
    Read {
        src: Nvfp4Source,
        rows: usize,
        k: usize,
        dst: usize,
    },
    /// Copy `len` bytes from `src` to `dst` (host memory, not overlapping).
    Copy { src: usize, dst: usize, len: usize },
}

/// A batch of jobs: waits for all of them, or cancels the ones not started.
pub struct Group {
    left: Mutex<usize>,
    done: Condvar,
    cancel: AtomicBool,
    /// A job was dropped by [`Self::cancel`] (its destination is incomplete).
    skipped: AtomicBool,
    err: Mutex<Option<String>>,
}

impl Group {
    fn finish(&self, r: Result<(), String>) {
        if let Err(e) = r {
            self.err.lock().unwrap().get_or_insert(e);
        }
        self.finish_n(1);
    }

    fn finish_n(&self, n: usize) {
        let mut left = self.left.lock().unwrap();
        *left -= n;
        if *left == 0 {
            self.done.notify_all();
        }
    }

    /// Whether every job ran or was dropped (without waiting).
    pub fn done(&self) -> bool {
        *self.left.lock().unwrap() == 0
    }

    /// Blocks until every job ran (or was cancelled); the first error, if any.
    pub fn wait(&self) -> Result<(), String> {
        let mut left = self.left.lock().unwrap();
        while *left > 0 {
            left = self.done.wait(left).unwrap();
        }
        match self.err.lock().unwrap().take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Drops the jobs no thread took yet; [`Self::wait`] then returns once the running ones
    /// end.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Whether a cancel dropped one of the jobs (after [`Self::wait`]: some did not run).
    pub fn skipped(&self) -> bool {
        self.skipped.load(Ordering::Relaxed)
    }
}

/// How soon the readers take a group's jobs: every queued job of a lower class first. A step's
/// own reads are [`Class::Demand`]; reads ahead of a step never hold one up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// What a step waits for now (and the staging reads of prompts, as before #615).
    Demand = 0,
    /// Reads ahead of the next step: the experts its routing most likely takes.
    Ahead = 1,
    /// Reads ahead the readers take only when nothing else waits (a prediction's margin).
    Extra = 2,
}

const CLASSES: usize = 3;

type Queued = (Job, Arc<Group>);

struct Shared {
    /// One queue per [`Class`].
    queue: Mutex<[VecDeque<Queued>; CLASSES]>,
    work: Condvar,
    stop: AtomicBool,
    files: Vec<File>,
    direct: bool,
    /// Bytes read from the files.
    read: AtomicU64,
}

/// The reader threads.
pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

/// Opens `path` for reading, around the page cache when `direct` (and the file system allows
/// it). Returns whether it is direct.
fn open(path: &PathBuf, direct: bool) -> std::io::Result<(File, bool)> {
    use std::os::unix::fs::OpenOptionsExt;
    if direct
        && let Ok(f) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
    {
        return Ok((f, true));
    }
    Ok((File::open(path)?, false))
}

impl Pool {
    /// `threads` readers over `files` (the checkpoint's shards, by [`FileRange::file`]).
    pub fn new(files: &[PathBuf], threads: usize, direct: bool) -> Result<Self, String> {
        Self::with_threads(files, threads.max(1), direct)
    }

    /// [`Self::new`] with exactly `threads` readers: none leaves every job queued (tests of
    /// the queues' order).
    pub(super) fn with_threads(
        files: &[PathBuf],
        threads: usize,
        direct: bool,
    ) -> Result<Self, String> {
        let mut opened = Vec::with_capacity(files.len());
        let mut all_direct = true;
        for p in files {
            let (f, d) = open(p, direct).map_err(|e| format!("{}: {e}", p.display()))?;
            all_direct &= d;
            opened.push(f);
        }
        if direct && !all_direct {
            // One mode for all: the files are on one file system.
            opened.clear();
            for p in files {
                opened.push(File::open(p).map_err(|e| format!("{}: {e}", p.display()))?);
            }
        }
        let shared = Arc::new(Shared {
            queue: Mutex::new(Default::default()),
            work: Condvar::new(),
            stop: AtomicBool::new(false),
            files: opened,
            direct: direct && all_direct,
            read: AtomicU64::new(0),
        });
        let threads = (0..threads)
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("ssd-experts-{i}"))
                    .spawn(move || worker(&s))
                    .map_err(|e| format!("reader thread: {e}"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { shared, threads })
    }

    /// The groups of the queued jobs, in the order the readers would take them (one entry per
    /// job). For tests.
    #[cfg(test)]
    pub(super) fn queued(&self) -> Vec<Arc<Group>> {
        let mut q = self.shared.queue.lock().unwrap().clone();
        std::iter::from_fn(|| pop(&mut q).map(|(_, g)| g)).collect()
    }

    /// Whether the files are read around the page cache.
    pub fn direct(&self) -> bool {
        self.shared.direct
    }

    /// Bytes read from the files so far.
    pub fn bytes_read(&self) -> u64 {
        self.shared.read.load(Ordering::Relaxed)
    }

    /// Queues `jobs` (in order, reads split into [`split`] pieces) as one group of
    /// [`Class::Demand`].
    pub fn submit(&self, jobs: Vec<Job>) -> Arc<Group> {
        self.submit_as(jobs, Class::Demand)
    }

    /// Queues `jobs` as one group of `class`.
    pub fn submit_as(&self, jobs: Vec<Job>, class: Class) -> Arc<Group> {
        let jobs: Vec<Job> = jobs.into_iter().flat_map(split).collect();
        let g = Arc::new(Group {
            left: Mutex::new(jobs.len()),
            done: Condvar::new(),
            cancel: AtomicBool::new(false),
            skipped: AtomicBool::new(false),
            err: Mutex::new(None),
        });
        if !jobs.is_empty() {
            let mut q = self.shared.queue.lock().unwrap();
            q[class as usize].extend(jobs.into_iter().map(|j| (j, g.clone())));
            self.shared.work.notify_all();
        }
        g
    }

    /// Moves `g`'s queued jobs of a later class to the end of the [`Class::Demand`] queue (a
    /// step now waits for them).
    pub fn promote(&self, g: &Arc<Group>) {
        let mut q = self.shared.queue.lock().unwrap();
        let (demand, later) = q.split_at_mut(1);
        for l in later {
            if !l.iter().any(|(_, x)| Arc::ptr_eq(x, g)) {
                continue;
            }
            let (mine, rest): (VecDeque<Queued>, VecDeque<Queued>) =
                l.drain(..).partition(|(_, x)| Arc::ptr_eq(x, g));
            *l = rest;
            demand[0].extend(mine);
        }
    }

    /// Cancels `g` and takes its queued jobs off the queues at once (they count as skipped),
    /// so [`Group::wait`] returns as soon as the jobs already running end.
    pub fn cancel(&self, g: &Arc<Group>) {
        g.cancel();
        let mut dropped = 0;
        {
            let mut q = self.shared.queue.lock().unwrap();
            for l in q.iter_mut() {
                let before = l.len();
                l.retain(|(_, x)| !Arc::ptr_eq(x, g));
                dropped += before - l.len();
            }
        }
        if dropped > 0 {
            g.skipped.store(true, Ordering::Relaxed);
            g.finish_n(dropped);
        }
    }

    /// Runs `jobs` and waits for them.
    pub fn run(&self, jobs: Vec<Job>) -> Result<(), String> {
        self.submit(jobs).wait()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Under the queue's lock: a reader checks the flag and waits with the lock held, so it
        // either sees the flag or is already waiting when the notify comes (#643: set outside
        // the lock, a reader between its check and its wait missed it and the join hung).
        {
            let _q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        self.shared.work.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Packed codes a read job takes at most: a decode step misses a few experts per layer, and
/// whole matrices (4 MiB) would leave most reader threads idle and the drive's queue short.
pub const PIECE: usize = 1 << 20;

/// `job` as reads of at most [`PIECE`] codes each (whole rows: a row's codes, scales and
/// repacked bytes are contiguous); other jobs as they are.
pub fn split(job: Job) -> Vec<Job> {
    let Job::Read { src, rows, k, dst } = job else {
        return vec![job];
    };
    let per = (PIECE / (k / 2).max(1)).max(1);
    (0..rows.div_ceil(per))
        .map(|i| {
            let r0 = i * per;
            let r = per.min(rows - r0);
            let sub = |f: FileRange, row: usize| FileRange {
                file: f.file,
                offset: f.offset + (r0 * row) as u64,
                len: (r * row) as u64,
            };
            Job::Read {
                src: Nvfp4Source {
                    weight: sub(src.weight, k / 2),
                    scale: sub(src.scale, k / 16),
                    scale2: src.scale2,
                },
                rows: r,
                k,
                dst: dst + r0 * (k / 64 * 36),
            }
        })
        .collect()
}

/// A page-aligned buffer that grows as needed.
struct Aligned {
    ptr: *mut u8,
    cap: usize,
}

impl Aligned {
    fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    fn get(&mut self, len: usize) -> &mut [u8] {
        if len > self.cap {
            self.free();
            let layout = std::alloc::Layout::from_size_align(len, ALIGN as usize).unwrap();
            // SAFETY: a non-zero size (len > cap >= 0).
            self.ptr = unsafe { std::alloc::alloc(layout) };
            assert!(
                !self.ptr.is_null(),
                "out of memory for a {len}-byte read buffer"
            );
            self.cap = len;
        }
        // SAFETY: `cap >= len` bytes owned by this buffer; reads fill them before use.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len) }
    }

    fn free(&mut self) {
        if !self.ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.cap, ALIGN as usize).unwrap();
            // SAFETY: allocated above with this layout.
            unsafe { std::alloc::dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        self.free();
    }
}

/// The aligned span around `r`: (start, length).
pub fn aligned_span(r: FileRange) -> (u64, usize) {
    let start = r.offset / ALIGN * ALIGN;
    let end = (r.offset + r.len).div_ceil(ALIGN) * ALIGN;
    (start, (end - start) as usize)
}

/// Reads `r` into `buf`, returning the range's bytes.
fn read_range<'b>(s: &Shared, r: FileRange, buf: &'b mut Aligned) -> Result<&'b [u8], String> {
    let f = &s.files[r.file];
    let (start, want, skip) = if s.direct {
        let (start, len) = aligned_span(r);
        (start, len, (r.offset - start) as usize)
    } else {
        (r.offset, r.len as usize, 0)
    };
    let need = skip + r.len as usize;
    let b = buf.get(want);
    let mut got = 0;
    while got < need {
        // SAFETY: `b[got..]` is writable; the fd is open.
        let n = unsafe {
            libc::pread(
                f.as_raw_fd(),
                b[got..].as_mut_ptr().cast(),
                want - got,
                (start + got as u64) as libc::off_t,
            )
        };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(format!("expert read at {}+{}: {e}", r.offset, r.len));
        }
        if n == 0 {
            return Err(format!(
                "expert read at {}+{}: end of file",
                r.offset, r.len
            ));
        }
        got += n as usize;
    }
    s.read.fetch_add(r.len, Ordering::Relaxed);
    if !s.direct {
        let _ = nix::fcntl::posix_fadvise(
            f.as_raw_fd(),
            r.offset as i64,
            r.len as i64,
            nix::fcntl::PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        );
    }
    Ok(&b[skip..need])
}

fn run(s: &Shared, job: Job, codes: &mut Aligned, scales: &mut Aligned) -> Result<(), String> {
    match job {
        Job::Read { src, rows, k, dst } => {
            let w = read_range(s, src.weight, codes)?;
            let sc = read_range(s, src.scale, scales)?;
            let bytes = rows * k / 64 * 36;
            // SAFETY: the tier hands each job a destination of its own, `bytes` long, in host
            // memory that outlives the job.
            let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes) };
            crate::hf::nvfp4::repack_nvfp4_into(w, sc, rows, k, out);
            Ok(())
        }
        Job::Copy { src, dst, len } => {
            // SAFETY: as above; source and destination are distinct buffers.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
            Ok(())
        }
    }
}

/// The next job a reader takes: the first of the most urgent class that has one.
fn pop(q: &mut [VecDeque<Queued>; CLASSES]) -> Option<Queued> {
    q.iter_mut().find_map(|l| l.pop_front())
}

fn worker(s: &Shared) {
    let (mut codes, mut scales) = (Aligned::new(), Aligned::new());
    loop {
        let (job, g) = {
            let mut q = s.queue.lock().unwrap();
            loop {
                if s.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(j) = pop(&mut q) {
                    break j;
                }
                q = s.work.wait(q).unwrap();
            }
        };
        let r = if g.cancelled() {
            g.skipped.store(true, Ordering::Relaxed);
            Ok(())
        } else {
            run(s, job, &mut codes, &mut scales)
        };
        g.finish(r);
    }
}
