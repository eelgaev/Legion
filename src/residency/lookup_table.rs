//! A big host lookup table read a few rows at a time: qwen4exp's per-layer n-gram embedding
//! (`per_layer_token_embd`, 26.8 GiB of 90-byte IQ4_NL rows, 16 rows per token), and hashed
//! n-gram memories like DeepSeek's Engram. The table is a row range of a model file, or several
//! ranges of one file numbered on from one to the next ([`LookupTable::open_parts`]: Engram's
//! per-layer tables). Every step the CPU fetches the rows its tokens hash to; the model then
//! dequantizes them and adds them to the residual stream. The table never goes to VRAM: VRAM is for expert slots.
//!
//! Where the rows come from ([`TableIo`]):
//! - **ram:** the whole table faulted in and locked ([`super::resident`]) on one NUMA node.
//! - **mmap:** the file mapped as is: rows not in the page cache are disk reads in the step.
//!   Kept for A/B runs.
//! - **direct:** unbuffered `O_DIRECT` reads of the rows' 4 KiB pages, so the table never enters
//!   RAM or the page cache. A bounded RAM cache keeps rows already fetched (set-associative,
//!   round-robin within a set; `--ngram-ram-gib`). A pool of I/O threads keeps up to
//!   `--ngram-inflight` reads in flight, with each batch's pages deduplicated and sorted. Small
//!   batches (a decode step's or a draft's rows, up to `--ngram-aio-max` pages) go out as Linux
//!   AIO from the asking thread instead ([`aio`]): no thread to wake per read. With
//!   `--ngram-keep-awake-ms`, a helper thread keeps the SSD out of its power-saving states while
//!   fetches come ([`keep_awake`]).
//! - **auto:** ram when the table fits in available memory with a margin and may be locked;
//!   otherwise direct, with a RAM cache sized to what is free. On a GPU that shares the host's
//!   memory (GB10) always direct: the GPU's KV cache and buffers come later out of the same RAM
//!   (Strata keeps the table on the SSD there too).
//!
//! The API is split ([`LookupTable::issue`], then [`LookupTable::collect`]), so a caller can start
//! a step's reads as soon as its tokens are known and wait only when it needs the rows.
//!
//! Adapted from Strata-AC922's `--ple-io direct|mmap|ram` (MIT): `PleReader` in
//! `src/ngram/ple_reader.cpp` (issue/collect, page dedup, the set-associative row cache) and
//! `PleTable::open` in `src/kernels/ngram.cpp`.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use serde::Serialize;

use crate::hw::numa::{Topology, pin_current_thread};

mod aio;
mod keep_awake;

/// How a lookup table's rows are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TableIo {
    /// `ram` when the table fits in available memory with a margin and may be locked, else
    /// `direct`.
    #[default]
    Auto,
    /// The whole table faulted in and locked in RAM.
    Ram,
    /// The file mapped as is (page cache; for comparisons).
    Mmap,
    /// Unbuffered SSD reads, with a bounded RAM cache of rows already fetched.
    Direct,
}

/// Where a table is: rows `0..n_rows` of `row_bytes` each, from byte `offset` of `path`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFile {
    pub path: PathBuf,
    pub offset: u64,
    pub n_rows: u64,
    pub row_bytes: usize,
}

impl TableFile {
    pub fn bytes(&self) -> u64 {
        self.n_rows * self.row_bytes as u64
    }
}

/// How to open a table.
#[derive(Debug, Clone)]
pub struct TableOptions {
    pub io: TableIo,
    /// The direct tier's row cache in bytes (default: a quarter of the memory left free, at most
    /// 4 GiB).
    pub ram_cache_bytes: Option<u64>,
    /// The direct tier's reads in flight (I/O threads; default [`DEFAULT_INFLIGHT`]).
    pub inflight: usize,
    /// The direct tier's batches of at most this many page reads go out as Linux AIO instead of
    /// through the I/O threads ([`aio`]; default [`DEFAULT_AIO_MAX`]); 0: never.
    pub aio_max: usize,
    /// The direct tier reads a page whenever no read went out for this many ms, while fetches
    /// come ([`keep_awake`]); 0: never.
    pub keep_awake_ms: u64,
    /// The NUMA node whose memory and CPUs the table uses (the rows' GPU's).
    pub node: Option<u32>,
    /// The GPU shares this memory (GB10): its KV cache and compute buffers are allocated after
    /// the table opens, out of the same RAM, so `auto` never takes the table whole.
    pub unified: bool,
}

impl Default for TableOptions {
    fn default() -> Self {
        Self {
            io: TableIo::default(),
            ram_cache_bytes: None,
            inflight: 0,
            aio_max: DEFAULT_AIO_MAX,
            keep_awake_ms: 0,
            node: None,
            unified: false,
        }
    }
}

/// Reads in flight by default with `direct` (a SATA SSD's queue, NCQ, holds 32; NVMe takes
/// more). A prompt's first chunk reads its rows' pages while only the layers before the n-gram
/// layer run (#815): on the GB10, a fresh 8K prompt's first step took 158 ms longer with 16 and
/// 70 ms less with 256 than with 64 (+1.5% prefill tok/s at 8K; decode batches go out as AIO).
pub const DEFAULT_INFLIGHT: usize = 256;

/// Biggest batch (page reads) sent as AIO by default: a decode step's rows of a few sequences
/// (16 per token). Prompt chunks' gathers (thousands) stay on the I/O threads, which do not
/// hold up the caller for the submit.
pub const DEFAULT_AIO_MAX: usize = 128;

const PAGE: u64 = 4096;
/// Most threads of the RAM tiers' gather (rows spread over tens of GB).
const GATHER_THREADS: usize = 8;

/// What [`LookupTable::open`] picked, and why.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableChoice {
    pub io: TableIo,
    pub reason: String,
    /// The direct tier's row cache, in bytes.
    pub ram_cache_bytes: u64,
    /// The ram tier: the table is locked (not just faulted in).
    pub locked: bool,
    pub seconds: f64,
}

/// Counters of the direct tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TableStats {
    pub rows: u64,
    pub cache_hits: u64,
    pub reads: u64,
    pub read_bytes: u64,
    /// Pages read only to keep the SSD awake ([`keep_awake`]).
    pub keep_awake_reads: u64,
}

/// A host lookup table (see the module docs).
pub struct LookupTable {
    /// The table's file, its first part's offset and every part's rows.
    file: TableFile,
    parts: Parts,
    tier: Tier,
    choice: TableChoice,
    cpus: Vec<u32>,
    /// [`Self::prefetch`]es not yet taken into the direct tier's cache.
    prefetched: Mutex<Vec<Ticket>>,
}

enum Tier {
    /// ram and mmap: the table's bytes, mapped (the map covers whole pages around them);
    /// `base`: the file offset of the map's first byte.
    Mapped {
        map: memmap2::Mmap,
        base: u64,
    },
    Direct(Direct),
}

/// Where each row of a table is in its file: per part its first row, byte offset and rows, the
/// rows numbered on from one part to the next.
#[derive(Debug, Clone)]
struct Parts {
    row_bytes: u64,
    starts: Vec<(u64, u64, u64)>,
}

impl Parts {
    fn new(parts: &[TableFile]) -> Self {
        let mut first = 0;
        let starts = parts
            .iter()
            .map(|p| {
                let s = (first, p.offset, p.n_rows);
                first += p.n_rows;
                s
            })
            .collect();
        Self {
            row_bytes: parts.first().map_or(0, |p| p.row_bytes as u64),
            starts,
        }
    }

    /// The file offset of row `r`.
    fn at(&self, r: u32) -> u64 {
        let r = r as u64;
        let (first, offset, _) = match self.starts.as_slice() {
            [one] => *one,
            s => s[s.partition_point(|&(first, _, _)| first <= r) - 1],
        };
        offset + (r - first) * self.row_bytes
    }

    /// Each part's bytes: (file offset, length).
    fn ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.starts
            .iter()
            .map(|&(_, offset, n)| (offset, n * self.row_bytes))
    }
}

/// A batch of rows being fetched ([`LookupTable::issue`]).
pub struct Ticket {
    rows: Vec<u32>,
    pending: Option<Arc<Pending>>,
    /// Rows the direct tier read from the file (not the cache): (index in `rows`, row).
    missed: Vec<(usize, u32)>,
}

impl LookupTable {
    /// Opens `file` with `opts`, deciding `auto` from the memory available now (call it after
    /// the experts' host memory is allocated) and placing RAM on `opts.node`.
    pub fn open(file: TableFile, opts: &TableOptions, topo: Option<&Topology>) -> io::Result<Self> {
        Self::open_parts(vec![file], opts, topo)
    }

    /// [`Self::open`] for a table in several row ranges of one file (`parts`, all of one row
    /// size): row `r` of the table is row `r` of the first part, then of the next part on, and
    /// so on. One tier, row cache and set of I/O threads serve them all; `auto` decides on
    /// their total size.
    pub fn open_parts(
        parts: Vec<TableFile>,
        opts: &TableOptions,
        topo: Option<&Topology>,
    ) -> io::Result<Self> {
        let invalid = |what: String| io::Error::new(io::ErrorKind::InvalidInput, what);
        let Some(head) = parts.first() else {
            return Err(invalid("a lookup table of no rows".into()));
        };
        if let Some(p) = parts
            .iter()
            .find(|p| p.path != head.path || p.row_bytes != head.row_bytes)
        {
            return Err(invalid(format!(
                "lookup table parts of other files or row sizes: {} ({} B rows) and {} ({} B)",
                head.path.display(),
                head.row_bytes,
                p.path.display(),
                p.row_bytes
            )));
        }
        let n_rows: u64 = parts.iter().map(|p| p.n_rows).sum();
        if n_rows > u32::MAX as u64 + 1 {
            return Err(invalid(format!("{n_rows} rows: rows are numbered by u32")));
        }
        let file = TableFile {
            path: head.path.clone(),
            offset: head.offset,
            n_rows,
            row_bytes: head.row_bytes,
        };
        let parts = Parts::new(&parts);
        let t = std::time::Instant::now();
        // The table node's helper CPUs: off CPU 0's core, within the process's own affinity (an
        // outer taskset holds; none of the node's: not pinned).
        let cpus: Vec<u32> = match (topo, opts.node) {
            (Some(topo), Some(id)) => crate::hw::numa::helper_cpus(topo, id),
            _ => Vec::new(),
        };
        let mem = MemInfo::read();
        let (io, reason) = match opts.io {
            TableIo::Auto if opts.unified => (
                TableIo::Direct,
                "the GPU shares this memory: its KV cache and buffers are allocated after the table"
                    .to_string(),
            ),
            TableIo::Auto => auto_choice(file.bytes(), &mem, memlock_allowed(file.bytes())),
            io => (io, "asked for".to_string()),
        };
        let mut choice = TableChoice {
            io,
            reason,
            ram_cache_bytes: 0,
            locked: false,
            seconds: 0.0,
        };
        let tier = match io {
            TableIo::Ram | TableIo::Mmap | TableIo::Auto => {
                let (map, base) = map_parts(&file.path, &parts)?;
                if io == TableIo::Ram {
                    let mut error = None;
                    choice.locked = true;
                    for (offset, len) in parts.ranges() {
                        let bytes = &map[(offset - base) as usize..][..len as usize];
                        let r = super::resident::make_resident(bytes, topo.zip(opts.node));
                        choice.locked &= r.locked;
                        error = error.or(r.lock_error);
                    }
                    if let Some(e) = error {
                        choice.reason += &format!("; not locked ({e}): its pages may be reclaimed");
                    }
                }
                Tier::Mapped { map, base }
            }
            TableIo::Direct => {
                let cache = opts.ram_cache_bytes.unwrap_or_else(|| {
                    let c = default_cache(&mem);
                    // Shared with the GPU: what the OS keeps free is all that is left.
                    if opts.unified {
                        c.min(UNIFIED_CACHE)
                    } else {
                        c
                    }
                });
                choice.ram_cache_bytes = cache;
                if is_rotational(&file.path) == Some(true) {
                    tracing::warn!(
                        path = %file.path.display(),
                        "direct reads of a lookup table from a rotational disk: every step waits \
                         on seeks; use an SSD or --ngram-io ram"
                    );
                }
                let inflight = if opts.inflight == 0 {
                    DEFAULT_INFLIGHT
                } else {
                    opts.inflight
                };
                let keep_awake = (opts.keep_awake_ms > 0).then(|| {
                    let (start, len) = parts.ranges().next().unwrap_or_default();
                    KeepAwakeAt {
                        start,
                        len,
                        interval: std::time::Duration::from_millis(opts.keep_awake_ms),
                    }
                });
                match Direct::open(&file, cache, inflight, opts.aio_max, keep_awake, &cpus) {
                    Ok(d) => Tier::Direct(d),
                    // `auto` chose direct: where the file cannot be read that way, map it.
                    Err(e) if opts.io == TableIo::Auto => {
                        tracing::warn!(
                            path = %file.path.display(),
                            error = %e,
                            "direct reads of the lookup table unavailable: mapping it instead"
                        );
                        choice.io = TableIo::Mmap;
                        choice.reason += &format!("; direct reads failed ({e}): mmap");
                        choice.ram_cache_bytes = 0;
                        let (map, base) = map_parts(&file.path, &parts)?;
                        Tier::Mapped { map, base }
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        choice.seconds = t.elapsed().as_secs_f64();
        Ok(Self {
            file,
            parts,
            tier,
            choice,
            cpus,
            prefetched: Mutex::new(Vec::new()),
        })
    }

    pub fn choice(&self) -> &TableChoice {
        &self.choice
    }

    pub fn row_bytes(&self) -> usize {
        self.file.row_bytes
    }

    pub fn n_rows(&self) -> u64 {
        self.file.n_rows
    }

    /// The direct tier's counters (zeros for the RAM tiers).
    pub fn stats(&self) -> TableStats {
        match &self.tier {
            Tier::Direct(d) => TableStats {
                keep_awake_reads: d.keep_awake.as_ref().map_or(0, |k| k.reads()),
                ..*d.stats.lock().unwrap_or_else(|e| e.into_inner())
            },
            Tier::Mapped { .. } => TableStats::default(),
        }
    }

    /// Starts fetching `rows` (each below [`Self::n_rows`]). The direct tier serves rows it has
    /// cached at once and queues reads of the others' pages.
    pub fn issue(&self, rows: &[u32]) -> Ticket {
        assert!(
            rows.iter().all(|&r| (r as u64) < self.file.n_rows),
            "row outside the {}-row table",
            self.file.n_rows
        );
        match &self.tier {
            Tier::Mapped { .. } => Ticket {
                rows: rows.to_vec(),
                pending: None,
                missed: Vec::new(),
            },
            Tier::Direct(d) => {
                self.take_prefetched(d);
                d.issue(&self.parts, rows)
            }
        }
    }

    /// Starts reading rows a later [`Self::issue`] will ask for (the next prompt chunk's) into
    /// the direct tier's row cache, so that issue finds them there. A hint: the RAM tiers and a
    /// direct tier without a cache ignore it.
    pub fn prefetch(&self, rows: &[u32]) {
        let Tier::Direct(d) = &self.tier else {
            return;
        };
        if d.cache.lock().unwrap_or_else(|e| e.into_inner()).sets == 0 {
            return;
        }
        let mut rows: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&r| (r as u64) < self.file.n_rows)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        let t = d.issue(&self.parts, &rows);
        self.prefetched
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(t);
    }

    /// Waits for the prefetches still out and puts their rows in the cache.
    fn take_prefetched(&self, d: &Direct) {
        let tickets =
            std::mem::take(&mut *self.prefetched.lock().unwrap_or_else(|e| e.into_inner()));
        for t in tickets {
            // A failed prefetch only leaves rows uncached: their own read reports the error.
            let _ = d.collect(t, self.file.row_bytes);
        }
    }

    /// Waits for `t`'s rows: their bytes, row after row (`rows.len() × row_bytes`).
    pub fn collect(&self, t: Ticket) -> io::Result<Vec<u8>> {
        let rb = self.file.row_bytes;
        match &self.tier {
            Tier::Mapped { map, base } => {
                let parts = &self.parts;
                let mut out = vec![0u8; t.rows.len() * rb];
                // Rows not resident (mmap) are page faults: several threads wait on them at
                // once; for a few rows starting threads costs more.
                let threads = (t.rows.len() / 64).clamp(1, GATHER_THREADS);
                let per = t.rows.len().div_ceil(threads).max(1);
                let cpus = self.cpus.as_slice();
                let copy = |rows: &[u32], out: &mut [u8]| {
                    for (&r, dst) in rows.iter().zip(out.chunks_mut(rb)) {
                        dst.copy_from_slice(&map[(parts.at(r) - base) as usize..][..rb]);
                    }
                };
                if threads == 1 {
                    // A few rows (a decode step's): on the calling thread (the engine's, on its
                    // GPU's node when the layout pins it); a new thread and its pin cost more
                    // than the copies.
                    copy(&t.rows, &mut out);
                } else {
                    // The table's node, off the engine thread's core (#250).
                    let pin = &crate::hw::host_core::off_host_core(cpus);
                    std::thread::scope(|scope| {
                        for (rows, out) in t.rows.chunks(per).zip(out.chunks_mut(per * rb)) {
                            scope.spawn(move || {
                                crate::hw::host_core::pin_helper(pin);
                                copy(rows, out);
                            });
                        }
                    });
                }
                Ok(out)
            }
            Tier::Direct(d) => d.collect(t, rb),
        }
    }

    /// [`Self::issue`] then [`Self::collect`].
    pub fn fetch(&self, rows: &[u32]) -> io::Result<Vec<u8>> {
        self.collect(self.issue(rows))
    }
}

/// Maps the whole pages around `parts` of `path` (from the first part's start to the last
/// one's end); returns the map and the file offset of its first byte.
fn map_parts(path: &Path, parts: &Parts) -> io::Result<(memmap2::Mmap, u64)> {
    let f = File::open(path)?;
    let lo = parts.ranges().map(|(o, _)| o).min().unwrap_or(0);
    let hi = parts.ranges().map(|(o, n)| o + n).max().unwrap_or(0);
    let start = lo - lo % PAGE;
    // SAFETY: a read-only private mapping of a model file nothing writes while it is open.
    let map = unsafe {
        memmap2::MmapOptions::new()
            .offset(start)
            .len((hi - start) as usize)
            .map(&f)?
    };
    Ok((map, start))
}

/// `/proc/meminfo`'s totals in bytes (zeros where unreadable).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
}

impl MemInfo {
    pub fn read() -> Self {
        std::fs::read_to_string("/proc/meminfo")
            .map(|s| Self::parse(&s))
            .unwrap_or_default()
    }

    fn parse(s: &str) -> Self {
        let kib = |key: &str| {
            s.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
                .map_or(0, |v| v << 10)
        };
        Self {
            total: kib("MemTotal:"),
            available: kib("MemAvailable:"),
        }
    }
}

/// Memory kept free beside a table taken whole into RAM: 8 GiB or a tenth of RAM.
fn margin(mem: &MemInfo) -> u64 {
    (8u64 << 30).max(mem.total / 10)
}

/// `auto`: ram when `bytes` plus the margin fits in available memory and the table may be locked;
/// otherwise direct. Returns the choice and why.
fn auto_choice(bytes: u64, mem: &MemInfo, lockable: bool) -> (TableIo, String) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let need = bytes + margin(mem);
    if mem.available < need {
        (
            TableIo::Direct,
            format!(
                "{:.1} GiB table + {:.1} GiB margin > {:.1} GiB available",
                gib(bytes),
                gib(margin(mem)),
                gib(mem.available)
            ),
        )
    } else if !lockable {
        (
            TableIo::Direct,
            "the table fits but may not be locked (no CAP_IPC_LOCK, RLIMIT_MEMLOCK too low)"
                .to_string(),
        )
    } else {
        (
            TableIo::Ram,
            format!(
                "{:.1} GiB table + {:.1} GiB margin fit in {:.1} GiB available",
                gib(bytes),
                gib(margin(mem)),
                gib(mem.available)
            ),
        )
    }
}

/// The direct tier's row cache by default at most, where the GPU shares the memory.
const UNIFIED_CACHE: u64 = 1 << 30;

/// The direct tier's row cache by default: a quarter of what the margin leaves, 64 MiB to 4 GiB.
fn default_cache(mem: &MemInfo) -> u64 {
    (mem.available.saturating_sub(margin(mem)) / 4).clamp(64 << 20, 4 << 30)
}

/// Whether this process may lock `bytes`: CAP_IPC_LOCK, or RLIMIT_MEMLOCK at least that.
fn memlock_allowed(bytes: u64) -> bool {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let cap = status
        .lines()
        .find_map(|l| l.strip_prefix("CapEff:"))
        .and_then(|v| u64::from_str_radix(v.trim(), 16).ok())
        .unwrap_or(0);
    const CAP_IPC_LOCK: u64 = 1 << 14;
    if cap & CAP_IPC_LOCK != 0 {
        return true;
    }
    let limits = std::fs::read_to_string("/proc/self/limits").unwrap_or_default();
    limits
        .lines()
        .find(|l| l.starts_with("Max locked memory"))
        .and_then(|l| l.split_whitespace().nth(3))
        .is_some_and(|soft| soft == "unlimited" || soft.parse::<u64>().is_ok_and(|b| b >= bytes))
}

/// Whether the disk holding `path` spins (sysfs `queue/rotational`); `None` when unknown.
fn is_rotational(path: &Path) -> Option<bool> {
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path).ok()?.dev();
    let (major, minor) = (
        ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff),
        (dev & 0xff) | ((dev >> 12) & !0xff),
    );
    let base = PathBuf::from(format!("/sys/dev/block/{major}:{minor}"));
    // A partition's queue is its disk's.
    [
        base.join("queue/rotational"),
        base.join("../queue/rotational"),
    ]
    .iter()
    .find_map(|p| std::fs::read_to_string(p).ok())
    .map(|s| s.trim() == "1")
}

// ---------------------------------------------------------------------------------------------
// The direct tier.

/// Ways per set of the row cache.
const WAYS: usize = 8;
const EMPTY: u32 = u32::MAX;

/// Rows already fetched: set-associative, round-robin within a set; bounded by construction
/// (Strata's `RowCache`).
struct RowCache {
    sets: usize,
    rb: usize,
    keys: Vec<u32>,
    data: Vec<u8>,
    next: Vec<u8>,
}

impl RowCache {
    fn new(bytes: u64, rb: usize) -> Self {
        let sets = (bytes as usize / rb) / WAYS;
        Self {
            sets,
            rb,
            keys: vec![EMPTY; sets * WAYS],
            data: vec![0; sets * WAYS * rb],
            next: vec![0; sets],
        }
    }

    fn set_of(&self, row: u32) -> usize {
        let x = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((x ^ (x >> 29)) % self.sets as u64) as usize
    }

    fn find(&self, row: u32) -> Option<&[u8]> {
        if self.sets == 0 {
            return None;
        }
        let s = self.set_of(row);
        (0..WAYS)
            .find(|w| self.keys[s * WAYS + w] == row)
            .map(|w| &self.data[(s * WAYS + w) * self.rb..][..self.rb])
    }

    fn insert(&mut self, row: u32, bytes: &[u8]) {
        if self.sets == 0 || self.find(row).is_some() {
            return;
        }
        let s = self.set_of(row);
        let w = self.next[s] as usize;
        self.next[s] = ((w + 1) % WAYS) as u8;
        self.keys[s * WAYS + w] = row;
        self.data[(s * WAYS + w) * self.rb..][..self.rb].copy_from_slice(bytes);
    }
}

/// A ticket's rows still to arrive.
struct Pending {
    state: Mutex<PendingState>,
    done: Condvar,
}

struct PendingState {
    jobs: usize,
    /// Of `jobs`, the reads in flight as AIO ([`aio::Aio::wait`] reaps them).
    aio_jobs: usize,
    /// The rows, one after another.
    out: Vec<u8>,
    row_bytes: usize,
    error: Option<String>,
}

/// One read: `len` bytes at the page-aligned `offset`, and where its rows go.
struct Job {
    offset: u64,
    len: usize,
    /// (index in the ticket's rows, byte offset of the row in the read).
    uses: Vec<(usize, usize)>,
    pending: Arc<Pending>,
}

struct Queue {
    jobs: Mutex<(VecDeque<Job>, bool)>,
    ready: Condvar,
}

struct Direct {
    cache: Mutex<RowCache>,
    queue: Arc<Queue>,
    workers: Vec<std::thread::JoinHandle<()>>,
    stats: Mutex<TableStats>,
    /// Small batches' reads ([`aio`]); dropped after the workers, waiting for its reads.
    aio: Option<Box<aio::Aio>>,
    /// Batches of at most this many pages go to `aio`.
    aio_max: usize,
    keep_awake: Option<keep_awake::KeepAwake>,
}

/// What [`keep_awake`] reads: a page of the bytes `start..start + len`, every `interval`.
struct KeepAwakeAt {
    start: u64,
    len: u64,
    interval: std::time::Duration,
}

/// A page-aligned buffer for `O_DIRECT` reads.
struct Aligned {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl Aligned {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, PAGE as usize).expect("layout");
        // SAFETY: a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory");
        Self { ptr, layout }
    }

    fn slice(&mut self, len: usize) -> &mut [u8] {
        // SAFETY: `len` is within the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, len.min(self.layout.size())) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// Opens `path` for unbuffered reads (`O_DIRECT`); where the filesystem refuses it (tmpfs),
/// buffered, with a warning. The flag: opened `O_DIRECT`.
fn open_direct(path: &Path) -> io::Result<(File, bool)> {
    let direct = nix::fcntl::OFlag::O_DIRECT.bits();
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(direct)
        .open(path)
    {
        Ok(f) => Ok((f, true)),
        Err(e) if e.raw_os_error() == Some(nix::libc::EINVAL) => {
            tracing::warn!(
                path = %path.display(),
                "O_DIRECT refused by the filesystem: reading the lookup table through the page cache"
            );
            Ok((File::open(path)?, false))
        }
        Err(e) => Err(e),
    }
}

impl Direct {
    fn open(
        file: &TableFile,
        cache_bytes: u64,
        inflight: usize,
        aio_max: usize,
        keep_awake: Option<KeepAwakeAt>,
        cpus: &[u32],
    ) -> io::Result<Self> {
        assert!(
            file.row_bytes as u64 <= PAGE,
            "rows longer than a page are not supported"
        );
        let (f, direct) = open_direct(&file.path)?;
        let f = Arc::new(f);
        // Buffered AIO would read inside io_submit: only for O_DIRECT.
        let aio = (aio_max > 0 && direct)
            .then(|| aio::Aio::new(f.clone()))
            .flatten()
            .map(Box::new);
        let queue = Arc::new(Queue {
            jobs: Mutex::new((VecDeque::new(), false)),
            ready: Condvar::new(),
        });
        let workers = (0..inflight.max(1))
            .map(|i| {
                let (f, queue, cpus) = (f.clone(), queue.clone(), cpus.to_vec());
                std::thread::Builder::new()
                    .name(format!("table-io-{i}"))
                    .spawn(move || {
                        if !cpus.is_empty() {
                            let _ = pin_current_thread(&cpus);
                        }
                        io_worker(&f, &queue)
                    })
            })
            .collect::<io::Result<Vec<_>>>()?;
        let keep_awake = keep_awake
            .map(|k| {
                keep_awake::KeepAwake::start(
                    f.clone(),
                    k.start,
                    k.len,
                    k.interval,
                    keep_awake::WINDOW,
                    cpus,
                )
            })
            .transpose()?;
        // The cache's pages are first touched on the table's node.
        let rb = file.row_bytes;
        let cache = match cpus {
            [] => RowCache::new(cache_bytes, rb),
            cpus => {
                let cpus = cpus.to_vec();
                std::thread::spawn(move || {
                    let _ = pin_current_thread(&cpus);
                    RowCache::new(cache_bytes, rb)
                })
                .join()
                .map_err(|_| io::Error::other("row cache allocation failed"))?
            }
        };
        Ok(Self {
            cache: Mutex::new(cache),
            queue,
            workers,
            stats: Mutex::new(TableStats::default()),
            aio,
            aio_max: aio_max.min(aio::DEPTH),
            keep_awake,
        })
    }

    fn issue(&self, parts: &Parts, rows: &[u32]) -> Ticket {
        let rb = parts.row_bytes as usize;
        let mut out = vec![0u8; rows.len() * rb];
        let mut missed = Vec::new();
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &r) in rows.iter().enumerate() {
                match cache.find(r) {
                    Some(b) => out[i * rb..][..rb].copy_from_slice(b),
                    None => missed.push((i, r)),
                }
            }
        }
        // One read per page run a row needs (two pages when it straddles a boundary), each page
        // read once per batch, in file order.
        let mut by_page: HashMap<u64, (usize, Vec<(usize, usize)>)> = HashMap::new();
        for &(i, r) in &missed {
            let at = parts.at(r);
            let page = at - at % PAGE;
            let len = (at + rb as u64).next_multiple_of(PAGE) - page;
            let e = by_page.entry(page).or_insert((0, Vec::new()));
            e.0 = e.0.max(len as usize);
            e.1.push((i, (at - page) as usize));
        }
        let mut pages: Vec<_> = by_page.into_iter().collect();
        pages.sort_unstable_by_key(|(p, _)| *p);
        {
            let mut s = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            s.rows += rows.len() as u64;
            s.cache_hits += (rows.len() - missed.len()) as u64;
            s.reads += pages.len() as u64;
            s.read_bytes += pages.iter().map(|(_, (len, _))| *len as u64).sum::<u64>();
        }
        if let Some(k) = &self.keep_awake {
            k.touch(!pages.is_empty());
        }
        let aio = self.aio.as_ref().filter(|_| pages.len() <= self.aio_max);
        let pending = Arc::new(Pending {
            state: Mutex::new(PendingState {
                jobs: pages.len(),
                aio_jobs: if aio.is_some() { pages.len() } else { 0 },
                out,
                row_bytes: rb,
                error: None,
            }),
            done: Condvar::new(),
        });
        let jobs: Vec<Job> = pages
            .into_iter()
            .map(|(offset, (len, uses))| Job {
                offset,
                len,
                uses,
                pending: pending.clone(),
            })
            .collect();
        // What AIO does not take (no free slots) goes to the threads.
        let jobs = match aio {
            Some(aio) => aio.submit(jobs),
            None => jobs,
        };
        if !jobs.is_empty() {
            let mut q = self.queue.jobs.lock().unwrap_or_else(|e| e.into_inner());
            q.0.extend(jobs);
            self.queue.ready.notify_all();
        }
        Ticket {
            rows: rows.to_vec(),
            pending: Some(pending),
            missed,
        }
    }

    fn collect(&self, t: Ticket, rb: usize) -> io::Result<Vec<u8>> {
        let pending = t.pending.expect("a direct ticket");
        let mut st = pending.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.jobs > 0 {
            match &self.aio {
                // Reads in flight as AIO: this thread reaps them (and any other ticket's).
                Some(aio) if st.aio_jobs > 0 => {
                    drop(st);
                    aio.wait(&pending);
                    st = pending.state.lock().unwrap_or_else(|e| e.into_inner());
                }
                _ => st = pending.done.wait(st).unwrap_or_else(|e| e.into_inner()),
            }
        }
        if let Some(e) = st.error.take() {
            return Err(io::Error::other(e));
        }
        let out = std::mem::take(&mut st.out);
        drop(st);
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        for &(i, r) in &t.missed {
            cache.insert(r, &out[i * rb..][..rb]);
        }
        Ok(out)
    }
}

impl Drop for LookupTable {
    fn drop(&mut self) {
        if let Tier::Direct(_) = self.tier {
            let s = self.stats();
            tracing::info!(
                rows = s.rows,
                cache_hit_rate = s.cache_hits as f64 / s.rows.max(1) as f64,
                reads = s.reads,
                read_mib = s.read_bytes as f64 / (1u64 << 20) as f64,
                "lookup table direct reads"
            );
        }
    }
}

impl Drop for Direct {
    fn drop(&mut self) {
        {
            let mut q = self.queue.jobs.lock().unwrap_or_else(|e| e.into_inner());
            q.1 = true;
        }
        self.queue.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// An I/O thread: takes reads off the queue until the table closes.
fn io_worker(file: &File, queue: &Queue) {
    let mut buf = Aligned::new(2 * PAGE as usize);
    loop {
        let job = {
            let mut q = queue.jobs.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(j) = q.0.pop_front() {
                    break j;
                }
                if q.1 {
                    return;
                }
                q = queue.ready.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        let read = read_full(file, buf.slice(job.len), job.offset);
        deliver(&job, read, buf.slice(job.len), false);
    }
}

/// Puts `job`'s rows from `data` (its read: `read` bytes of it, or the error) into its ticket and
/// counts the job done (an AIO one: also off `aio_jobs`).
fn deliver(job: &Job, read: io::Result<usize>, data: &[u8], aio: bool) {
    let mut st = job.pending.state.lock().unwrap_or_else(|e| e.into_inner());
    match read {
        Ok(n) => {
            let data = &data[..n.min(data.len())];
            let row_bytes = st.row_bytes;
            for &(i, at) in &job.uses {
                match data.get(at..at + row_bytes) {
                    Some(src) => st.out[i * row_bytes..][..row_bytes].copy_from_slice(src),
                    None => {
                        st.error.get_or_insert_with(|| {
                            format!("short read at byte {} of the table file", job.offset)
                        });
                    }
                }
            }
        }
        Err(e) => {
            st.error.get_or_insert_with(|| e.to_string());
        }
    }
    st.jobs -= 1;
    if aio {
        st.aio_jobs -= 1;
    }
    if st.jobs == 0 {
        job.pending.done.notify_all();
    }
}

/// `pread` until `buf` is full or the file ends; returns the bytes read.
fn read_full(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match file.read_at(&mut buf[n..], offset + n as u64) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests;
