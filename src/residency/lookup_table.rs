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
//!   round-robin within a set). With a RAM budget (`--ngram-ram SIZE`, [`RamBudget`]) the
//!   cache is that size and keeps the rows that recur instead: a share of every set pins the
//!   rows hit most so far (the hot set, counted per row as the cache serves, whatever a row
//!   index means: token n-grams or Engram hashes), and the rest replaces by CLOCK with a small
//!   use counter, so rows read once cannot push out rows read again. A pool of I/O threads keeps up to
//!   `--ngram-inflight` reads in flight, with each batch's pages deduplicated and sorted. Small
//!   batches (a decode step's or a draft's rows, up to `--ngram-aio-max` pages) go out as Linux
//!   AIO from the asking thread instead ([`aio`]): no thread to wake per read. With
//!   `--ngram-keep-awake-ms`, a helper thread keeps the SSD out of its power-saving states while
//!   fetches come ([`keep_awake`]).
//! - **auto:** ram when the table fits in available memory with a margin and may be locked;
//!   otherwise direct, with a RAM cache sized to what is free. On a GPU that shares the host's
//!   memory (GB10) always direct: the GPU's KV cache and buffers come later out of the same RAM
//!   (Strata keeps the table on the SSD there too). `--ngram-ram` overrides it: `all` is ram, a
//!   SIZE at least the table's is ram, a smaller one direct with that RAM budget.
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

/// How much of a table lives in RAM (`--ngram-ram`).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RamBudget {
    /// As [`TableIo`] decides (today's tiers, cache sizes and replacement).
    #[default]
    Auto,
    /// The whole table (the ram tier).
    All,
    /// This many bytes: the ram tier when the table fits in it, else direct with a RAM budget of
    /// exactly this.
    Bytes(u64),
    /// This share of MemAvailable at load, in percent (then as [`Self::Bytes`]).
    Percent(f64),
}

impl RamBudget {
    /// The budget in bytes (a percentage of `available`); `None` for `Auto` and `All`.
    pub fn bytes(&self, available: u64) -> Option<u64> {
        match *self {
            Self::Auto | Self::All => None,
            Self::Bytes(b) => Some(b),
            Self::Percent(p) => Some((available as f64 * p / 100.0) as u64),
        }
    }
}

impl std::fmt::Display for RamBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Bytes(b) => f.write_str(&size(b)),
            Self::Percent(p) => write!(f, "{p}% of available memory"),
        }
    }
}

/// `b` bytes for people: GiB from 1 GiB up, else MiB.
fn size(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64)
    }
}

/// Parses `--ngram-ram`: `auto`, `all`, a size (a number, decimals allowed, with an optional
/// `K`/`M`/`G`/`T` suffix in binary units, `iB` optional; no suffix: GiB) or a percentage of
/// MemAvailable (`30%`). A size must be above 0.
pub fn parse_ram_budget(s: &str) -> Result<RamBudget, String> {
    let lower = s.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" => return Ok(RamBudget::Auto),
        "all" => return Ok(RamBudget::All),
        _ => {}
    }
    let bad = || format!("{s:?}: want auto, all, a size such as 8G, 512M or 1.5 (GiB), or 30%");
    let number = |n: &str| {
        n.trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(bad)
    };
    if let Some(p) = lower.strip_suffix('%') {
        let p = number(p)?;
        if p <= 0.0 || p > 100.0 {
            return Err(format!("{s:?}: a percentage above 0 and at most 100"));
        }
        return Ok(RamBudget::Percent(p));
    }
    let body = lower.strip_suffix("ib").unwrap_or(&lower);
    let (n, shift) = match body.as_bytes().last() {
        Some(b'k') => (&body[..body.len() - 1], 10),
        Some(b'm') => (&body[..body.len() - 1], 20),
        Some(b'g') => (&body[..body.len() - 1], 30),
        Some(b't') => (&body[..body.len() - 1], 40),
        // `8iB` has no unit.
        _ if body.len() != lower.len() => return Err(bad()),
        _ => (body, 30),
    };
    let bytes = number(n)? * (1u64 << shift) as f64;
    if bytes < 1.0 {
        return Err(format!("{s:?}: the RAM budget must be above 0"));
    }
    Ok(RamBudget::Bytes(bytes as u64))
}

/// The `--ngram-io` and `--ngram-ram` pairs that contradict each other whatever the table's
/// size: `mmap` with `all` or a size, `direct` with `all`.
pub fn ram_conflict(io: TableIo, ram: RamBudget) -> Option<String> {
    match (io, ram) {
        (_, RamBudget::Auto) => None,
        (TableIo::Mmap, r) => Some(format!(
            "--ngram-io mmap reads through the page cache: it takes no --ngram-ram ({r})"
        )),
        (TableIo::Direct, RamBudget::All) => Some(
            "--ngram-io direct with --ngram-ram all: all is the ram tier (--ngram-io ram)".into(),
        ),
        _ => None,
    }
}

/// What `io` and `ram` ask of a table of `table` bytes with `mem` free now: the tier (`Auto`:
/// decided as without `--ngram-ram`) and, for direct with a size, its RAM budget in bytes.
/// Pairs that contradict each other are errors.
pub fn resolve_ram(
    io: TableIo,
    ram: RamBudget,
    table: u64,
    mem: &MemInfo,
) -> Result<(TableIo, Option<u64>), String> {
    if let Some(e) = ram_conflict(io, ram) {
        return Err(e);
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    match (ram, ram.bytes(mem.available)) {
        (RamBudget::Auto, _) => Ok((io, None)),
        (RamBudget::All, _) => Ok((TableIo::Ram, None)),
        (_, Some(0)) | (_, None) => Err(format!(
            "--ngram-ram {ram}: that is no memory ({:.1} GiB available)",
            gib(mem.available)
        )),
        (_, Some(b)) if b >= table => match io {
            TableIo::Direct => Ok((TableIo::Direct, Some(b))),
            _ => Ok((TableIo::Ram, None)),
        },
        (_, Some(b)) => match io {
            TableIo::Ram => Err(format!(
                "--ngram-io ram with --ngram-ram {ram} ({:.2} GiB): the table is {:.2} GiB; \
                 give a size at least that, all, or drop --ngram-io",
                gib(b),
                gib(table)
            )),
            _ => Ok((TableIo::Direct, Some(b))),
        },
    }
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
    /// How much of the table lives in RAM (`--ngram-ram`); with a size it decides the tier with
    /// `io` ([`resolve_ram`]) and replaces `ram_cache_bytes`.
    pub ram: RamBudget,
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
            ram: RamBudget::Auto,
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
    /// The direct tier's row cache, in bytes (with `--ngram-ram SIZE`: that budget).
    pub ram_cache_bytes: u64,
    /// The direct tier with a RAM budget: the most rows the hot set pins, and their bytes (0
    /// without a budget: today's round-robin cache).
    pub hot_rows: u64,
    pub hot_bytes: u64,
    /// The ram tier: the table is locked (not just faulted in).
    pub locked: bool,
    pub seconds: f64,
}

/// Counters of the direct tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TableStats {
    /// Rows asked for (fetches and prefetches), and of them those the row cache served.
    pub rows: u64,
    pub cache_hits: u64,
    /// Rows the hot set pins now.
    pub hot_rows: u64,
    pub reads: u64,
    pub read_bytes: u64,
    /// Pages read only to keep the SSD awake ([`keep_awake`]).
    pub keep_awake_reads: u64,
}

impl TableStats {
    /// Rows the row cache did not hold.
    pub fn cache_misses(&self) -> u64 {
        self.rows.saturating_sub(self.cache_hits)
    }

    /// Hits over rows asked for; `None` before any.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.rows > 0).then(|| self.cache_hits as f64 / self.rows as f64)
    }
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
    /// [`Self::prefetch_next`]es that the next [`Self::issue`] leaves for the one after.
    next: Mutex<Vec<Ticket>>,
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
    /// A [`LookupTable::prefetch`]: its rows are asked for later (their first hit is that ask).
    prefetch: bool,
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
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let (asked, budget) =
            resolve_ram(opts.io, opts.ram, file.bytes(), &mem).map_err(invalid)?;
        let (io, reason) = match (asked, budget) {
            (_, Some(b)) => (
                TableIo::Direct,
                format!(
                    "--ngram-ram {}: a {} RAM budget for the {:.1} GiB table",
                    opts.ram,
                    size(b),
                    gib(file.bytes())
                ),
            ),
            (TableIo::Ram, None) if opts.ram != RamBudget::Auto => (
                TableIo::Ram,
                format!("--ngram-ram {}: the whole table", opts.ram),
            ),
            (TableIo::Auto, None) if opts.unified => (
                TableIo::Direct,
                "the GPU shares this memory: its KV cache and buffers are allocated after the table"
                    .to_string(),
            ),
            (TableIo::Auto, None) => {
                auto_choice(file.bytes(), &mem, memlock_allowed(file.bytes()))
            }
            (io, None) => (io, "asked for".to_string()),
        };
        let mut choice = TableChoice {
            io,
            reason,
            ram_cache_bytes: 0,
            hot_rows: 0,
            hot_bytes: 0,
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
                let cache = match budget {
                    // No more than holds every row.
                    Some(b) => CacheSpec::budget(
                        b.min(file.n_rows * (file.row_bytes as u64 + HOT_WAY_BYTES)),
                        old_policy(),
                    ),
                    None => CacheSpec::plain(opts.ram_cache_bytes.unwrap_or_else(|| {
                        let c = default_cache(&mem);
                        // Shared with the GPU: what the OS keeps free is all that is left.
                        if opts.unified {
                            c.min(UNIFIED_CACHE)
                        } else {
                            c
                        }
                    })),
                };
                choice.ram_cache_bytes = cache.bytes;
                if cache.budget && cache.replace == Replace::RoundRobin {
                    choice.reason += "; LEGION_DEBUG_NGRAM_OLD_POLICY: round-robin, no hot set";
                }
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
                    Ok(d) => {
                        let hot = d.cache.lock().unwrap_or_else(|e| e.into_inner()).hot_max();
                        choice.hot_rows = hot;
                        choice.hot_bytes = hot * file.row_bytes as u64;
                        Tier::Direct(d)
                    }
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
            next: Mutex::new(Vec::new()),
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
                hot_rows: d.cache.lock().unwrap_or_else(|e| e.into_inner()).pinned,
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
                prefetch: false,
            },
            Tier::Direct(d) => {
                self.take_prefetched(d);
                d.issue(&self.parts, rows, false)
            }
        }
    }

    /// Starts reading rows a later [`Self::issue`] will ask for (the next prompt chunk's) into
    /// the direct tier's row cache, so that issue finds them there. A hint: the RAM tiers and a
    /// direct tier without a cache ignore it.
    pub fn prefetch(&self, rows: &[u32]) {
        self.prefetch_into(rows, &self.prefetched);
    }

    /// [`Self::prefetch`] of rows that the issue after the next one asks for (the next prompt
    /// chunk's, issued before the current chunk's step fills): the next issue does not wait
    /// for them, it leaves them for the one after. Waiting there held a Flash-Next prompt
    /// step's fill (with the GPU idle) for ~35 ms on about a third of the 4,096-token steps of
    /// a 123K prompt on a DGX Spark (GB10), the steps whose next chunk read from the SSD
    /// (#1183).
    pub fn prefetch_next(&self, rows: &[u32]) {
        self.prefetch_into(rows, &self.next);
    }

    fn prefetch_into(&self, rows: &[u32], into: &Mutex<Vec<Ticket>>) {
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
        let t = d.issue(&self.parts, &rows, true);
        into.lock().unwrap_or_else(|e| e.into_inner()).push(t);
    }

    /// Waits for the prefetches still out and puts their rows in the cache; the
    /// [`Self::prefetch_next`]es become prefetches the next call waits for.
    fn take_prefetched(&self, d: &Direct) {
        let next = std::mem::take(&mut *self.next.lock().unwrap_or_else(|e| e.into_inner()));
        let tickets = std::mem::replace(
            &mut *self.prefetched.lock().unwrap_or_else(|e| e.into_inner()),
            next,
        );
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

/// Of a set's ways, the most the hot set pins: half the budget. On a simulated serving trace of
/// Qwen3.8-Flash-Next's n-gram rows (the real row hash over 4.45M tokens of docs and code,
/// 1000 requests of 500-30K prompt tokens plus 256-1000 generated, four users interleaved), with
/// budgets of 5-50% of the rows the trace touches, the hit rate was (round-robin / CLOCK / this):
/// 60.1/64.2/65.8% at 5%, 65.7/69.7/70.9% at 10%, 73.9/77.3/77.7% at 25%, 80.6/82.8/82.9% at
/// 50%. Pinning 6 of 8 gained ~0.4 point more on steady traffic but lost up to 0.2 against CLOCK
/// when the traffic moved to other documents half way; 2 of 8 gained half as much.
const HOT_WAYS: usize = 4;
/// Hits a row needs since it came in to join the hot set (1 pinned rows that recur once and
/// held up worse when the traffic moved; 4 gained less).
const HOT_MIN_HITS: u16 = 2;
/// The CLOCK use counter's top: a row hit this often survives this many sweeps of its set.
const USE_MAX: u8 = 3;
/// A way's flags in [`RowCache::meta`]: pinned (the hot set), and fresh (a prefetch's, not yet
/// asked for: its first hit is that ask and does not count).
const PINNED: u8 = 0x80;
const FRESH: u8 = 0x40;
const USE: u8 = 0x03;
/// What a way costs in a RAM budget besides its row: its key, flags and hit count.
const HOT_WAY_BYTES: u64 = 4 + 1 + 2;

/// How [`RowCache`] picks the row a new one replaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Replace {
    /// Round-robin within a set (Strata's): `auto` and `--ngram-io direct` without a size.
    RoundRobin,
    /// A RAM budget (`--ngram-ram SIZE`): up to [`HOT_WAYS`] ways of a set pin its rows hit
    /// most, the others replace by CLOCK.
    Hot,
}

/// The row cache's size and policy.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CacheSpec {
    bytes: u64,
    /// `bytes` is a RAM budget: the rows and their bookkeeping fit in it.
    budget: bool,
    replace: Replace,
}

impl CacheSpec {
    /// Today's cache: `bytes` of rows, round-robin.
    fn plain(bytes: u64) -> Self {
        Self {
            bytes,
            budget: false,
            replace: Replace::RoundRobin,
        }
    }

    /// A RAM budget of `bytes`: the hot set and CLOCK, or (`old`, for A/B runs) round-robin over
    /// as many rows.
    fn budget(bytes: u64, old: bool) -> Self {
        Self {
            bytes,
            budget: true,
            replace: if old {
                Replace::RoundRobin
            } else {
                Replace::Hot
            },
        }
    }
}

/// `LEGION_DEBUG_NGRAM_OLD_POLICY=1`: a RAM budget's cache replaces round-robin, without a hot
/// set (old against new policy at one size; debug only).
fn old_policy() -> bool {
    std::env::var_os("LEGION_DEBUG_NGRAM_OLD_POLICY").is_some_and(|v| v == "1")
}

/// Rows already fetched: set-associative, bounded by construction (Strata's `RowCache`).
/// Without a budget round-robin within a set; with one ([`Replace::Hot`]) every hit counts for
/// its row, a row hit [`HOT_MIN_HITS`] times joins its set's hot ways (up to [`HOT_WAYS`],
/// taking the place of the hot row hit least when they are taken and it was hit more), and a
/// new row replaces by CLOCK among the other ways. Hot rows are never evicted: a row that loses
/// its hot place stays cached as a recently used one. Nothing here reads a row index as a
/// token: the counts are per row, whatever the row stands for.
struct RowCache {
    sets: usize,
    rb: usize,
    keys: Vec<u32>,
    data: Vec<u8>,
    /// Per set: the way round-robin replaces next, or CLOCK's hand.
    next: Vec<u8>,
    replace: Replace,
    /// [`Replace::Hot`], per way: [`PINNED`], [`FRESH`] and the use counter ([`USE`]).
    meta: Vec<u8>,
    /// [`Replace::Hot`], per way: its row's hits since it came in (saturating).
    hits: Vec<u16>,
    /// Rows pinned now.
    pinned: u64,
}

impl RowCache {
    fn new(spec: CacheSpec, rb: usize) -> Self {
        let sets = if spec.budget {
            (spec.bytes / (WAYS as u64 * (rb as u64 + HOT_WAY_BYTES) + 1)) as usize
        } else {
            (spec.bytes as usize / rb) / WAYS
        };
        let hot = spec.replace == Replace::Hot;
        Self {
            sets,
            rb,
            keys: vec![EMPTY; sets * WAYS],
            data: vec![0; sets * WAYS * rb],
            next: vec![0; sets],
            replace: spec.replace,
            meta: vec![0; if hot { sets * WAYS } else { 0 }],
            hits: vec![0; if hot { sets * WAYS } else { 0 }],
            pinned: 0,
        }
    }

    /// The most rows the hot set pins.
    fn hot_max(&self) -> u64 {
        match self.replace {
            Replace::Hot => (self.sets * HOT_WAYS) as u64,
            Replace::RoundRobin => 0,
        }
    }

    fn set_of(&self, row: u32) -> usize {
        let x = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((x ^ (x >> 29)) % self.sets as u64) as usize
    }

    /// The way holding `row`, as an index into `keys`.
    fn slot(&self, row: u32) -> Option<usize> {
        if self.sets == 0 {
            return None;
        }
        let s = self.set_of(row) * WAYS;
        (s..s + WAYS).find(|&i| self.keys[i] == row)
    }

    /// `row`'s bytes if cached. `ask`: a fetch's (counts for the replacement), not a prefetch's.
    fn get(&mut self, row: u32, ask: bool) -> Option<&[u8]> {
        let i = self.slot(row)?;
        if ask && self.replace == Replace::Hot {
            self.hit(i);
        }
        Some(&self.data[i * self.rb..][..self.rb])
    }

    /// Counts a hit of way `i`: its use counter and hits, and the hot set.
    fn hit(&mut self, i: usize) {
        let m = self.meta[i];
        if m & FRESH != 0 {
            // A prefetched row's first ask.
            self.meta[i] = m & !FRESH;
            return;
        }
        self.meta[i] = (m & !USE) | ((m & USE) + 1).min(USE_MAX);
        self.hits[i] = self.hits[i].saturating_add(1);
        if m & PINNED == 0 && self.hits[i] >= HOT_MIN_HITS {
            self.promote(i);
        }
    }

    /// Pins way `i` if its set has a hot way free, or in place of its hot way hit least when
    /// `i` was hit more; that row stays cached, as a recently used one.
    fn promote(&mut self, i: usize) {
        let s = i / WAYS * WAYS;
        let mut pinned = 0;
        let mut least: Option<usize> = None;
        for j in s..s + WAYS {
            if self.meta[j] & PINNED != 0 {
                pinned += 1;
                let lower = match least {
                    Some(l) => self.hits[j] < self.hits[l],
                    None => true,
                };
                if lower {
                    least = Some(j);
                }
            }
        }
        if pinned < HOT_WAYS {
            self.meta[i] |= PINNED;
            self.pinned += 1;
        } else if let Some(l) = least.filter(|&l| self.hits[i] > self.hits[l]) {
            self.meta[l] = (self.meta[l] & !(PINNED | USE)) | USE_MAX;
            self.meta[i] |= PINNED;
        }
    }

    /// The way of set `s` a new row takes: an empty one, else CLOCK over the ways not pinned
    /// (each sweep lowers their use counters; the first at 0 goes).
    fn victim(&mut self, s: usize) -> usize {
        let base = s * WAYS;
        if let Some(w) = (0..WAYS).find(|&w| self.keys[base + w] == EMPTY) {
            return w;
        }
        // Ends: at most HOT_WAYS < WAYS ways are pinned, and every pass lowers the others.
        let mut hand = self.next[s] as usize;
        loop {
            let (w, m) = (hand, self.meta[base + hand]);
            hand = (hand + 1) % WAYS;
            if m & PINNED != 0 {
                continue;
            }
            if m & USE == 0 {
                self.next[s] = hand as u8;
                return w;
            }
            self.meta[base + w] = m - 1;
        }
    }

    /// Caches `row`'s `bytes`; `prefetched`: read ahead of the fetch that will ask for it.
    fn insert(&mut self, row: u32, bytes: &[u8], prefetched: bool) {
        if self.sets == 0 || self.slot(row).is_some() {
            return;
        }
        let s = self.set_of(row);
        let w = match self.replace {
            Replace::RoundRobin => {
                let w = self.next[s] as usize;
                self.next[s] = ((w + 1) % WAYS) as u8;
                w
            }
            Replace::Hot => self.victim(s),
        };
        let i = s * WAYS + w;
        if self.replace == Replace::Hot {
            // `victim` never takes a pinned way.
            self.meta[i] = if prefetched { FRESH } else { 0 };
            self.hits[i] = 0;
        }
        self.keys[i] = row;
        self.data[i * self.rb..][..self.rb].copy_from_slice(bytes);
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
        cache: CacheSpec,
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
            [] => RowCache::new(cache, rb),
            cpus => {
                let cpus = cpus.to_vec();
                std::thread::spawn(move || {
                    let _ = pin_current_thread(&cpus);
                    RowCache::new(cache, rb)
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

    /// `prefetch`: rows a later issue asks for ([`LookupTable::prefetch`]); the hot set counts
    /// that ask, not this one.
    fn issue(&self, parts: &Parts, rows: &[u32], prefetch: bool) -> Ticket {
        let rb = parts.row_bytes as usize;
        let mut out = vec![0u8; rows.len() * rb];
        let mut missed = Vec::new();
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &r) in rows.iter().enumerate() {
                match cache.get(r, !prefetch) {
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
            prefetch,
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
            cache.insert(r, &out[i * rb..][..rb], t.prefetch);
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
                hot_rows = s.hot_rows,
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
