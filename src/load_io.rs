//! Parallel positioned reads of file ranges straight into their destinations.
//!
//! The caller lists `(file, offset, len, destination)` jobs (sorted by file offset); a small pool
//! of worker threads pulls work in that order and reads it with `pread` ([`FileExt::read_exact_at`]).
//! A destination is either host memory, which is read into directly with no copy, or a
//! [`DeviceSink`] (VRAM), which is fed through a reusable per-worker staging buffer. Large jobs
//! are cut into chunks so one huge tensor still uses every worker.
//!
//! Buffered reads are the default so a warm page cache stays fast; each file gets
//! `posix_fadvise(SEQUENTIAL)` and the start of its range `WILLNEED`. No O_DIRECT. A NUMA
//! node's host pieces whose pages are not cached yet (`mincore`) may add page cache only up to
//! what the node has free once its arena is allocated: pieces past that drop their page cache
//! (`DONTNEED`) once read, else the new cache fills the node and spills onto other nodes, on an
//! AC922 the GPUs' own memory, which the KV cache then finds taken. Cached pieces cost nothing
//! new, so a warm reload (e.g. 186 GB of GLM-5.3-Flash on an AC922) keeps its cache.
//! `--load-page-cache keep` / `drop` ([`PageCache`]) never / always drop it instead.
//!
//! Device destinations are written by a single worker unless `--load-parallel-device`
//! ([`Options::parallel_device`]). The parallel device path is unverified on a GPU. What the
//! source says: in `third_party/ggml/src/ggml-cuda/ggml-cuda.cu`,
//! `ggml_backend_cuda_buffer_set_tensor` does `ggml_cuda_set_device(ctx->device)`, then
//! `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, cudaStreamPerThread)` and
//! `cudaStreamSynchronize(cudaStreamPerThread)`. The per-thread default stream and the per-call
//! sync suggest concurrent calls from different threads are independent, but nobody has run it
//! here, and other backends make no such promise, so it is opt-in. Host jobs always use the
//! whole pool; the device worker also helps with host jobs once its device queue is empty.
//!
//! NUMA hook: with an [`Options::topology`] and a [`Job::node`], the pool keeps one queue per
//! node and its workers pin themselves to that node's CPUs ([`pin_current_thread`]), so a
//! node-local arena is first-touched and filled by node-local threads. Where the machine has
//! memory-only nodes (an AC922's GPU memory), every worker also binds its memory to the nodes
//! with CPUs (`MPOL_BIND`, nearest first), so the page cache its reads add never lands in a
//! GPU's memory, where it would outlive the process and shrink every later one's VRAM (#772).

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use nix::fcntl::{PosixFadviseAdvice, posix_fadvise};

use crate::hw::numa::{Topology, pin_current_thread};

pub const DEFAULT_LOAD_THREADS: usize = 4;
/// Machine-sharing rule: no CPU-heavy work uses more than 8 threads.
pub const MAX_LOAD_THREADS: usize = 8;
/// Largest piece of one job read at a time; also the size of a worker's staging buffer.
pub const DEFAULT_CHUNK: usize = 16 << 20;
/// How much of each file's range gets `WILLNEED` up front.
const WILLNEED_BYTES: u64 = 256 << 20;

static LOAD_THREADS: AtomicUsize = AtomicUsize::new(DEFAULT_LOAD_THREADS);
static PARALLEL_DEVICE: AtomicBool = AtomicBool::new(false);
static MMAP: AtomicBool = AtomicBool::new(true);
static PAGE_CACHE: AtomicU8 = AtomicU8::new(PageCache::Auto as u8);

/// Sets what [`Options::configured`] gives from now on, for what is `Some` (`legion serve
/// --load-threads`, `--load-parallel-device`, `--no-mmap`). Threads are clamped to 1..=8.
pub fn configure(
    threads: Option<usize>,
    parallel_device: Option<bool>,
    mmap: Option<bool>,
    page_cache: Option<PageCache>,
) {
    if let Some(n) = threads {
        LOAD_THREADS.store(n.clamp(1, MAX_LOAD_THREADS), Ordering::Relaxed);
    }
    if let Some(on) = parallel_device {
        PARALLEL_DEVICE.store(on, Ordering::Relaxed);
    }
    if let Some(on) = mmap {
        MMAP.store(on, Ordering::Relaxed);
    }
    if let Some(p) = page_cache {
        PAGE_CACHE.store(p as u8, Ordering::Relaxed);
    }
}

/// What the loader does with the page cache of the host pieces it reads
/// (`--load-page-cache`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum PageCache {
    /// Pieces not cached yet keep their cache while it fits in the free memory of their NUMA
    /// node (with an arena per node); the rest drop it once read (see the module docs).
    #[default]
    Auto,
    /// Leave the page cache alone.
    Keep,
    /// Drop every host piece's page cache once read.
    Drop,
}

impl PageCache {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Keep,
            2 => Self::Drop,
            _ => Self::Auto,
        }
    }
}

/// The loader's worker count ([`configure`]; default 4).
pub fn load_threads() -> usize {
    LOAD_THREADS.load(Ordering::Relaxed)
}

/// A destination that is not host-addressable and takes bytes at an offset (a device tensor).
/// Called from several worker threads at once, always for disjoint ranges.
pub trait DeviceSink: Send + Sync {
    fn write_at(&self, offset: usize, bytes: &[u8]);
}

pub enum Dest<'a> {
    /// Read straight into this memory; its length must equal the job's `len`.
    Host(&'a mut [u8]),
    /// Read into a staging buffer, then [`DeviceSink::write_at`] each chunk.
    Device(Arc<dyn DeviceSink + 'a>),
}

pub struct Job<'a> {
    /// Index into the `files` slice given to [`read_all`].
    pub file: usize,
    pub offset: u64,
    pub len: u64,
    pub dest: Dest<'a>,
    /// NUMA node whose threads should do this read (used only when a topology is given).
    pub node: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Options<'t> {
    pub threads: usize,
    pub chunk: usize,
    /// Let every worker write device destinations (default: one worker per queue does).
    pub parallel_device: bool,
    /// The page cache of host pieces ([`PageCache`]).
    pub page_cache: PageCache,
    pub topology: Option<&'t Topology>,
    /// Map the file and point plain-CPU tensors into it instead of reading them
    /// (default true; `--no-mmap` turns it off).
    pub mmap: bool,
    /// Extra tensors loaded from another tensor's bytes, `(name, source)`: a second copy in
    /// whatever buffer the placement gives `name` (the tied LM head on the last GPU).
    pub aliases: &'t [(&'t str, &'t str)],
    /// Tensors left out of the load (the SSD expert tier reads them itself,
    /// [`crate::residency::ssd_experts`]).
    pub skip: &'t [String],
    /// Extra tensors read from ranges of the file their infos give (GLM's parallel decode:
    /// row shards and per-GPU mirrors), placed like any other tensor.
    pub extra: &'t [crate::gguf::TensorInfo],
    /// How to read those of the extra tensors that are no plain range of the file (GLM's
    /// tensor parallelism: column shards). Never converted to F16.
    pub extra_recipes: Option<&'t std::collections::HashMap<String, crate::gguf::Recipe>>,
}

impl Options<'_> {
    /// Default chunking, no pinning, and the workers, device writers and mapping [`configure`]
    /// set.
    pub fn configured() -> Self {
        let mut o = Self::with_threads(load_threads());
        o.parallel_device = PARALLEL_DEVICE.load(Ordering::Relaxed);
        o.mmap = MMAP.load(Ordering::Relaxed);
        o.page_cache = PageCache::from_u8(PAGE_CACHE.load(Ordering::Relaxed));
        o
    }

    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.clamp(1, MAX_LOAD_THREADS),
            chunk: DEFAULT_CHUNK,
            parallel_device: false,
            page_cache: PageCache::Auto,
            topology: None,
            mmap: true,
            aliases: &[],
            skip: &[],
            extra: &[],
            extra_recipes: None,
        }
    }
}

/// A failed read: which job (index into the `jobs` vector) and why.
#[derive(Debug)]
pub struct ReadError {
    pub job: usize,
    pub source: io::Error,
}

enum UnitDest<'a> {
    Host(&'a mut [u8]),
    Device {
        sink: Arc<dyn DeviceSink + 'a>,
        base: usize,
        len: usize,
    },
}

#[derive(Default)]
struct Queues<'a> {
    host: VecDeque<Unit<'a>>,
    device: VecDeque<Unit<'a>>,
}

impl<'a> Queues<'a> {
    fn first_job(&self) -> usize {
        self.host
            .front()
            .or(self.device.front())
            .map_or(0, |u| u.job)
    }

    /// Device work first (only for workers allowed to do it), then host work.
    fn pop(&mut self, device_ok: bool) -> Option<Unit<'a>> {
        if device_ok && let Some(u) = self.device.pop_front() {
            return Some(u);
        }
        self.host.pop_front()
    }
}

struct Unit<'a> {
    job: usize,
    file: usize,
    offset: u64,
    dest: UnitDest<'a>,
    /// Drop the piece's page cache once read (see the module docs).
    drop_cache: bool,
}

/// Reads every job. Returns the error of the first failing read (other workers stop early).
pub fn read_all(files: &[File], jobs: Vec<Job<'_>>, opts: &Options<'_>) -> Result<(), ReadError> {
    let chunk = opts.chunk.max(1);
    let mut ranges: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    let mut buckets: BTreeMap<Option<u32>, Queues<'_>> = BTreeMap::new();
    for (job, j) in jobs.into_iter().enumerate() {
        assert!(j.file < files.len(), "job {job}: no such file {}", j.file);
        if j.len == 0 {
            continue;
        }
        let r = ranges.entry(j.file).or_insert((j.offset, 0));
        r.0 = r.0.min(j.offset);
        r.1 = r.1.max(j.offset + j.len);
        let node = j.node.filter(|_| opts.topology.is_some());
        let entry = buckets.entry(node).or_default();
        match j.dest {
            Dest::Host(mut rest) => {
                let queue = &mut entry.host;
                assert_eq!(rest.len() as u64, j.len, "job {job}: destination size");
                let mut offset = j.offset;
                while !rest.is_empty() {
                    let (head, tail) = rest.split_at_mut(chunk.min(rest.len()));
                    let n = head.len() as u64;
                    queue.push_back(Unit {
                        job,
                        file: j.file,
                        offset,
                        dest: UnitDest::Host(head),
                        drop_cache: false,
                    });
                    offset += n;
                    rest = tail;
                }
            }
            Dest::Device(sink) => {
                let queue = &mut entry.device;
                let mut base = 0usize;
                while (base as u64) < j.len {
                    let len = chunk.min(j.len as usize - base);
                    queue.push_back(Unit {
                        job,
                        file: j.file,
                        offset: j.offset + base as u64,
                        dest: UnitDest::Device {
                            sink: sink.clone(),
                            base,
                            len,
                        },
                        drop_cache: false,
                    });
                    base += len;
                }
            }
        }
    }
    // Before `advise`, whose readahead would count as cached.
    if opts.page_cache == PageCache::Drop {
        for queue in buckets.values_mut() {
            queue.host.iter_mut().for_each(|u| u.drop_cache = true);
        }
    } else if opts.page_cache == PageCache::Auto && opts.topology.is_some() {
        for (node, queue) in buckets.iter_mut() {
            let Some(free) = node.and_then(crate::hw::numa::node_mem_free) else {
                continue;
            };
            let misses: Vec<u64> = queue
                .host
                .iter()
                .map(|u| {
                    let len = match &u.dest {
                        UnitDest::Host(buf) => buf.len() as u64,
                        UnitDest::Device { len, .. } => *len as u64,
                    };
                    uncached_bytes(&files[u.file], u.offset, len).unwrap_or(len)
                })
                .collect();
            for (u, drop) in queue.host.iter_mut().zip(drop_cache_plan(&misses, free)) {
                u.drop_cache = drop;
            }
        }
    }
    for (&f, &(start, end)) in &ranges {
        advise(&files[f], start, end);
    }
    if buckets.is_empty() {
        return Ok(());
    }

    // CPUs per node, checked before any thread starts.
    let mut cpus: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    if let Some(topo) = opts.topology {
        for (&node, queue) in &buckets {
            let Some(node) = node else { continue };
            match topo.nodes.iter().find(|n| n.id == node) {
                Some(n) if !n.cpus.is_empty() => {
                    cpus.insert(node, n.cpus.clone());
                }
                _ => {
                    return Err(ReadError {
                        job: queue.first_job(),
                        source: io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("NUMA node {node} unknown or has no CPUs"),
                        ),
                    });
                }
            }
        }
    }

    // The page cache the workers' reads add stays on the nodes with CPUs (#772): spilled onto
    // an AC922's GPU memory, it outlives the process and takes VRAM from every later one.
    let bind = opts
        .topology
        .and_then(crate::hw::numa::cpu_nodes_beside_memory_only);
    let bind = bind.as_deref();

    let nb = buckets.len();
    let total = opts.threads.max(nb);
    let abort = AtomicBool::new(false);
    let first_error: Mutex<Option<ReadError>> = Mutex::new(None);
    let queues: Vec<(Option<u32>, Mutex<Queues<'_>>)> = buckets
        .into_iter()
        .map(|(node, q)| (node, Mutex::new(q)))
        .collect();

    std::thread::scope(|scope| {
        for (i, (node, queue)) in queues.iter().enumerate() {
            let workers = total / nb + usize::from(i < total % nb);
            for w in 0..workers {
                let device_ok = opts.parallel_device || w == 0;
                let (abort, first_error) = (&abort, &first_error);
                let pin = node.map(|n| cpus[&n].as_slice());
                scope.spawn(move || {
                    let mut staging = Vec::new();
                    if let Some(pin) = pin
                        && let Err(source) = pin_current_thread(pin)
                    {
                        fail(first_error, abort, 0, source);
                        return;
                    }
                    // Only a thread on the default policy: a `--numa` or numactl bind stays.
                    let default_policy = || {
                        crate::hw::numa_bind::thread_policy()
                            .is_ok_and(|p| p.mode == crate::hw::numa_bind::MPOL_DEFAULT)
                    };
                    if let Some(nodes) = bind.filter(|_| default_policy())
                        && let Err(e) = crate::hw::numa_bind::bind_thread_memory(nodes)
                    {
                        // Best effort: the default policy only risks the spill.
                        tracing::debug!(error = %e, "load worker: cannot bind its page cache");
                    }
                    while !abort.load(Ordering::Relaxed) {
                        let Some(unit) = queue.lock().unwrap().pop(device_ok) else {
                            break;
                        };
                        let job = unit.job;
                        if let Err(source) = run(&files[unit.file], unit, &mut staging) {
                            fail(first_error, abort, job, source);
                        }
                    }
                });
            }
        }
    });
    match first_error.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn fail(slot: &Mutex<Option<ReadError>>, abort: &AtomicBool, job: usize, source: io::Error) {
    abort.store(true, Ordering::Relaxed);
    slot.lock()
        .unwrap()
        .get_or_insert(ReadError { job, source });
}

fn run(file: &File, unit: Unit<'_>, staging: &mut Vec<u8>) -> io::Result<()> {
    match unit.dest {
        UnitDest::Host(buf) => {
            file.read_exact_at(buf, unit.offset)?;
            if unit.drop_cache {
                let _ = posix_fadvise(
                    file.as_raw_fd(),
                    unit.offset as i64,
                    buf.len() as i64,
                    PosixFadviseAdvice::POSIX_FADV_DONTNEED,
                );
            }
            Ok(())
        }
        UnitDest::Device { sink, base, len } => {
            if staging.len() < len {
                staging.resize(len, 0);
            }
            file.read_exact_at(&mut staging[..len], unit.offset)?;
            sink.write_at(base, &staging[..len]);
            Ok(())
        }
    }
}

/// Which of a node's host pieces drop their page cache as they are read, given each piece's
/// bytes not cached yet (`misses`, in read order) and the node's `free` memory with its arena
/// allocated: pieces keep their cache while the new cache fits, less [`CACHE_MARGIN`].
fn drop_cache_plan(misses: &[u64], free: u64) -> Vec<bool> {
    let mut budget = free.saturating_sub(CACHE_MARGIN);
    misses
        .iter()
        .map(|&m| {
            if m <= budget {
                budget -= m;
                false
            } else {
                true
            }
        })
        .collect()
}

/// Free memory a node keeps when the loader adds page cache to it.
const CACHE_MARGIN: u64 = 2 << 30;

/// The bytes of `file`'s `[offset, offset + len)` not in the page cache (`mincore` over a
/// read-only mapping), `None` if that cannot be told.
fn uncached_bytes(file: &File, offset: u64, len: u64) -> Option<u64> {
    if len == 0 {
        return Some(0);
    }
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page = u64::try_from(page).ok().filter(|&p| p > 0)?;
    let start = offset / page * page;
    let map_len = usize::try_from(offset + len - start).ok()?;
    let pages = map_len.div_ceil(page as usize);
    let mut vec = vec![0u8; pages];
    // SAFETY: a fresh shared read-only mapping of the file, only passed to mincore (its pages
    // are never touched, so nothing is read in) and unmapped before returning.
    unsafe {
        let p = libc::mmap(
            std::ptr::null_mut(),
            map_len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            start as libc::off_t,
        );
        if p == libc::MAP_FAILED {
            return None;
        }
        let rc = libc::mincore(p, map_len, vec.as_mut_ptr());
        libc::munmap(p, map_len);
        if rc != 0 {
            return None;
        }
    }
    let missing = vec.iter().filter(|&&v| v & 1 == 0).count() as u64;
    Some((missing * page).min(len))
}

/// Advisory only: failures (unsupported file systems, pipes) are ignored.
fn advise(file: &File, start: u64, end: u64) {
    let fd = file.as_raw_fd();
    let len = (end - start) as i64;
    let _ = posix_fadvise(
        fd,
        start as i64,
        len,
        PosixFadviseAdvice::POSIX_FADV_SEQUENTIAL,
    );
    let ahead = (end - start).min(WILLNEED_BYTES) as i64;
    let _ = posix_fadvise(
        fd,
        start as i64,
        ahead,
        PosixFadviseAdvice::POSIX_FADV_WILLNEED,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn drop_cache_only_for_new_cache_past_free_memory() {
        let gib = 1u64 << 30;
        // A warm reload: everything cached, nothing new, whatever the node has free.
        assert_eq!(drop_cache_plan(&[0; 3], 0), vec![false; 3]);
        // Cold: pieces keep their cache until the free memory (less the margin) is used up.
        let free = 5 * gib + CACHE_MARGIN;
        assert_eq!(
            drop_cache_plan(&[2 * gib, 2 * gib, 2 * gib, gib, 0], free),
            vec![false, false, true, false, false]
        );
    }

    #[test]
    fn mincore_counts_pages_not_cached() {
        let dir = std::env::temp_dir().join(format!("legion-mincore-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        std::fs::write(&path, pattern(1 << 20)).unwrap();
        let f = File::open(&path).unwrap();
        // Just written: cached (tmpfs or not), so nothing is missing.
        assert_eq!(uncached_bytes(&f, 12345, 500_000), Some(0));
        assert_eq!(uncached_bytes(&f, 0, 0), Some(0));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8)
            .collect()
    }

    fn file_with(data: &[u8]) -> (tempfile::NamedTempFile, File) {
        let mut t = tempfile::NamedTempFile::new().unwrap();
        t.write_all(data).unwrap();
        let f = File::open(t.path()).unwrap();
        (t, f)
    }

    #[test]
    fn host_jobs_fill_buffers_in_chunks() {
        let data = pattern(100_000);
        let (_t, f) = file_with(&data);
        let mut a = vec![0u8; 70_001];
        let mut b = vec![0u8; 5];
        let mut c = vec![0u8; 0];
        let jobs = vec![
            Job {
                file: 0,
                offset: 10,
                len: 70_001,
                dest: Dest::Host(&mut a),
                node: None,
            },
            Job {
                file: 0,
                offset: 80_000,
                len: 5,
                dest: Dest::Host(&mut b),
                node: None,
            },
            Job {
                file: 0,
                offset: 0,
                len: 0,
                dest: Dest::Host(&mut c),
                node: None,
            },
        ];
        let opts = Options {
            threads: 4,
            page_cache: PageCache::Auto,
            parallel_device: false,
            chunk: 4096,
            topology: None,
            mmap: true,
            aliases: &[],
            skip: &[],
            extra: &[],
            extra_recipes: None,
        };
        read_all(std::slice::from_ref(&f), jobs, &opts).unwrap();
        assert_eq!(a, data[10..70_011]);
        assert_eq!(b, data[80_000..80_005]);
    }

    struct Capture(Mutex<Vec<u8>>, AtomicBool);
    impl DeviceSink for Capture {
        fn write_at(&self, offset: usize, bytes: &[u8]) {
            self.1.store(true, Ordering::Relaxed);
            self.0.lock().unwrap()[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
    }

    /// A host buffer pretending to be device memory: only reachable through `write_at`.
    #[test]
    fn device_jobs_go_through_staging() {
        let data = pattern(50_000);
        let (_t, f) = file_with(&data);
        let sink = Arc::new(Capture(Mutex::new(vec![0; 30_000]), AtomicBool::new(false)));
        let jobs = vec![Job {
            file: 0,
            offset: 1234,
            len: 30_000,
            dest: Dest::Device(sink.clone()),
            node: None,
        }];
        let opts = Options {
            threads: 3,
            page_cache: PageCache::Auto,
            parallel_device: false,
            chunk: 7000,
            topology: None,
            mmap: true,
            aliases: &[],
            skip: &[],
            extra: &[],
            extra_recipes: None,
        };
        read_all(std::slice::from_ref(&f), jobs, &opts).unwrap();
        assert!(sink.1.load(Ordering::Relaxed));
        assert_eq!(*sink.0.lock().unwrap(), data[1234..31_234]);
    }

    struct Ids(Mutex<std::collections::HashSet<std::thread::ThreadId>>);
    impl DeviceSink for Ids {
        fn write_at(&self, _offset: usize, _bytes: &[u8]) {
            std::thread::sleep(std::time::Duration::from_millis(2));
            self.0.lock().unwrap().insert(std::thread::current().id());
        }
    }

    fn device_threads(parallel_device: bool) -> usize {
        let (_t, f) = file_with(&pattern(64_000));
        let sink = Arc::new(Ids(Mutex::new(Default::default())));
        let jobs = vec![Job {
            file: 0,
            offset: 0,
            len: 64_000,
            dest: Dest::Device(sink.clone()),
            node: None,
        }];
        let opts = Options {
            threads: 4,
            page_cache: PageCache::Auto,
            chunk: 1000,
            parallel_device,
            topology: None,
            mmap: true,
            aliases: &[],
            skip: &[],
            extra: &[],
            extra_recipes: None,
        };
        read_all(std::slice::from_ref(&f), jobs, &opts).unwrap();
        let ids = sink.0.lock().unwrap();
        ids.len()
    }

    #[test]
    fn device_jobs_use_one_worker_unless_opted_in() {
        assert_eq!(device_threads(false), 1);
        assert!(device_threads(true) > 1);
    }

    #[test]
    fn short_file_reports_the_failing_job() {
        let (_t, f) = file_with(&pattern(100));
        let mut a = vec![0u8; 50];
        let mut b = vec![0u8; 50];
        let jobs = vec![
            Job {
                file: 0,
                offset: 0,
                len: 50,
                dest: Dest::Host(&mut a),
                node: None,
            },
            Job {
                file: 0,
                offset: 80,
                len: 50,
                dest: Dest::Host(&mut b),
                node: None,
            },
        ];
        let err = read_all(std::slice::from_ref(&f), jobs, &Options::with_threads(1)).unwrap_err();
        assert_eq!(err.job, 1);
        assert_eq!(err.source.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn node_pinned_workers_read_their_jobs() {
        let topo = Topology::detect().unwrap();
        let node = topo.nodes.iter().find(|n| !n.cpus.is_empty()).unwrap().id;
        let data = pattern(40_000);
        let (_t, f) = file_with(&data);
        let mut a = vec![0u8; 40_000];
        let jobs = vec![Job {
            file: 0,
            offset: 0,
            len: 40_000,
            dest: Dest::Host(&mut a),
            node: Some(node),
        }];
        let opts = Options {
            threads: 2,
            page_cache: PageCache::Auto,
            parallel_device: false,
            chunk: 9000,
            topology: Some(&topo),
            mmap: true,
            aliases: &[],
            skip: &[],
            extra: &[],
            extra_recipes: None,
        };
        read_all(std::slice::from_ref(&f), jobs, &opts).unwrap();
        assert_eq!(a, data);

        let mut b = vec![0u8; 4];
        let jobs = vec![Job {
            file: 0,
            offset: 0,
            len: 4,
            dest: Dest::Host(&mut b),
            node: Some(9999),
        }];
        let err = read_all(std::slice::from_ref(&f), jobs, &opts).unwrap_err();
        assert_eq!(err.source.kind(), io::ErrorKind::InvalidInput);
    }
}
