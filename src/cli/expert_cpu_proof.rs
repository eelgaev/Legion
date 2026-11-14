//! `legion expert-cpu-proof`: whether the host can compute a routed expert with the GPU decode
//! kernels' bits (#37; a development tool).
//!
//! `check` recomputes, from the GGUF's bytes, every expert product of a
//! `LEGION_DEBUG_EXPERT_DUMP` file (what the kernels read and wrote in real steps) with
//! `legion::cpu::gpu_exact` and compares bit for bit. `time` measures those host kernels on real
//! experts, on one thread and on pools. `selftest` checks the vector kernels against the scalar
//! reference.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, anyhow, bail};
use clap::{Args, Subcommand};
use legion::cpu::QuantFormat;
use legion::cpu::gpu_exact::{self, Q8_1_BLOCK_BYTES};
use legion::ggml::types;
use legion::gguf::GgufFile;
use legion::model::core::expert_mv::dump::{self, PRINT, Record};

#[derive(Args)]
pub struct ExpertCpuProofArgs {
    #[command(subcommand)]
    what: What,
}

#[derive(Subcommand)]
enum What {
    /// The vector kernels against the scalar reference on random blocks, bit for bit.
    Selftest {
        #[arg(long, default_value_t = 200)]
        rounds: usize,
    },
    /// Recompute a dump's expert products on the host and compare with the GPU's, bit for bit.
    Check {
        /// The GGUF the dump's server ran.
        #[arg(long)]
        model: PathBuf,
        /// Dump files (`LEGION_DEBUG_EXPERT_DUMP`).
        #[arg(long, required = true, num_args = 1..)]
        dump: Vec<PathBuf>,
        /// Also compute every n-th product with the scalar reference (0: never).
        #[arg(long, default_value_t = 64)]
        scalar_every: usize,
        /// Print each layer's line.
        #[arg(long)]
        layers: bool,
    },
    /// Time the host kernels on real experts of a GGUF, held in memory.
    Time {
        #[arg(long)]
        model: PathBuf,
        /// Distinct experts in memory (3 MB each; more than the caches hold).
        #[arg(long, default_value_t = 420)]
        experts: usize,
        /// CPUs the threads are pinned to, in order, comma separated.
        #[arg(long, value_delimiter = ',', required = true)]
        cpus: Vec<u32>,
        /// Pool sizes, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 4, 8, 12, 16])]
        threads: Vec<usize>,
        /// Experts of a step (the misses a decode step has), for the per-step column.
        #[arg(long, default_value_t = 140)]
        step: usize,
        /// Passes over the experts per measurement.
        #[arg(long, default_value_t = 3)]
        passes: usize,
        /// Memory traffic beside the pool, GB/s in all (0: none): reader threads stream their
        /// own buffers at this rate, as the GPU's reads of host experts over the link would.
        #[arg(long, default_value_t = 0.0)]
        load_gbs: f64,
        /// CPUs of the reader threads (one each), comma separated; not the pool's.
        #[arg(long, value_delimiter = ',')]
        load_cpus: Vec<u32>,
    },
}

const KINDS: [&str; 3] = ["gate", "up", "down"];

fn format_of(ty: u32) -> Option<QuantFormat> {
    [
        (types::Q4_K, QuantFormat::Q4K),
        (types::Q5_K, QuantFormat::Q5K),
        (types::Q5_1, QuantFormat::Q5_1),
        (types::Q8_0, QuantFormat::Q8_0),
    ]
    .into_iter()
    .find_map(|(t, f)| (t as u32 == ty).then_some(f))
}

/// One layer's stack of experts of one matrix.
struct Stack {
    layer: usize,
    kind: usize,
    format: Option<QuantFormat>,
    ty: u32,
    k: usize,
    rows: usize,
    n_expert: usize,
    file: usize,
    offset: usize,
    expert_bytes: usize,
}

/// A model read with plain positioned reads: page faults at random places of a large model file
/// are slow on some file systems, sequential reads of whole stacks are not.
struct Model {
    files: Vec<File>,
    stacks: Vec<Stack>,
    layers: usize,
}

/// Shard `index` of a split GGUF whose first shard is `first` (`...-00001-of-0000N.gguf`).
fn shard_path(first: &Path, index: usize) -> PathBuf {
    let name = first.to_string_lossy();
    PathBuf::from(name.replace("-00001-of-", &format!("-{:05}-of-", index + 1)))
}

impl Model {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let gguf = GgufFile::open(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let mut stacks = Vec::new();
        let mut layers = 0;
        'layers: loop {
            for (kind, name) in KINDS.iter().enumerate() {
                let Some(t) = gguf.tensor(&format!("blk.{layers}.ffn_{name}_exps.weight")) else {
                    break 'layers;
                };
                if t.shape.len() != 3 {
                    bail!("{}: not a stack of experts", t.name);
                }
                let n_expert = t.shape[2] as usize;
                stacks.push(Stack {
                    layer: layers,
                    kind,
                    format: format_of(t.ggml_type as u32),
                    ty: t.ggml_type as u32,
                    k: t.shape[0] as usize,
                    rows: t.shape[1] as usize,
                    n_expert,
                    file: t.file_index,
                    offset: t.offset as usize,
                    expert_bytes: t.size as usize / n_expert,
                });
            }
            layers += 1;
        }
        if layers == 0 {
            bail!("{}: no blk.N.ffn_*_exps.weight tensors", path.display());
        }
        let files = stacks.iter().map(|s| s.file).max().unwrap_or(0) + 1;
        let files = (0..files)
            .map(|i| {
                let p = shard_path(gguf.path(), i);
                File::open(&p).with_context(|| p.display().to_string())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let lens = files
            .iter()
            .map(|f| f.metadata().map(|m| m.len() as usize))
            .collect::<std::io::Result<Vec<_>>>()?;
        for s in &stacks {
            if s.offset + s.n_expert * s.expert_bytes > lens[s.file] {
                bail!(
                    "blk.{} {}: outside its shard (is shard {} this model's?)",
                    s.layer,
                    KINDS[s.kind],
                    s.file + 1
                );
            }
        }
        Ok(Self {
            files,
            stacks,
            layers,
        })
    }

    /// Experts `first..first + count` of a stack into `buf`, then drops them from the page cache
    /// (a pass over the model reads more than the RAM others need holds).
    fn read(&self, s: &Stack, first: usize, count: usize, buf: &mut Vec<u8>) -> anyhow::Result<()> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::FileExt;
        let (at, len) = (s.offset + first * s.expert_bytes, count * s.expert_bytes);
        buf.resize(len, 0);
        let f = &self.files[s.file];
        f.read_exact_at(buf, at as u64)
            .with_context(|| format!("blk.{} {}: read", s.layer, KINDS[s.kind]))?;
        // SAFETY: a valid descriptor; the advice only affects caching.
        unsafe {
            libc::posix_fadvise(
                f.as_raw_fd(),
                at as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        Ok(())
    }
}

/// A product the dump recorded: record, matrix of the op, pair (token * n_used + slot).
type Want = (u32, u8, u16);

#[derive(Default, Clone)]
struct Tally {
    products: u64,
    equal: u64,
    /// Products of experts the kernel read outside VRAM (a host tier slot).
    host: u64,
}

impl Tally {
    fn add(&mut self, equal: bool, host: bool) {
        self.products += 1;
        self.equal += u64::from(equal);
        self.host += u64::from(host);
    }
}

fn check(
    model: &Path,
    dumps: &[PathBuf],
    scalar_every: usize,
    per_layer: bool,
) -> anyhow::Result<()> {
    let m = Model::open(model)?;
    let mut recs: Vec<Record> = Vec::new();
    for path in dumps {
        let mut r = BufReader::new(File::open(path).with_context(|| path.display().to_string())?);
        loop {
            let rec = match Record::read(&mut r) {
                Ok(Some(rec)) => rec,
                Ok(None) => break,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    println!("{}: ends inside a record (ignored)", path.display());
                    break;
                }
                Err(e) => return Err(e).with_context(|| path.display().to_string()),
            };
            recs.push(rec);
        }
    }
    let records = recs.len() as u64;

    // The products the records want, by the first bytes of the expert each read: those bytes
    // name an expert (checked below: the same bytes twice would be ambiguous).
    let (mut dead, mut bad_id, mut shape) = (0u64, 0u64, 0u64);
    let (mut act_pairs, mut act_equal) = (0u64, 0u64);
    let mut steps: BTreeMap<usize, u64> = BTreeMap::new();
    let mut wants: HashMap<[u8; PRINT], Vec<Want>> = HashMap::new();
    for (ri, rec) in recs.iter().enumerate() {
        *steps.entry(rec.n_tok).or_default() += 1;
        for (mat, prints) in rec.prints.iter().enumerate() {
            for pair in 0..rec.pairs() {
                let (t, j) = (pair / rec.n_used, pair % rec.n_used);
                if rec.glu() && mat == 0 && !rec.dead(t, j) {
                    act_pairs += 1;
                    act_equal += u64::from(rec.act_equal[pair] == 1);
                }
                if rec.dead(t, j) {
                    dead += 1;
                    continue;
                }
                let id = rec.ids[pair];
                if id < 0 || id as u32 >= rec.n_w {
                    bad_id += 1;
                    continue;
                }
                let print: [u8; PRINT] =
                    prints[pair * PRINT..][..PRINT].try_into().expect("a print");
                wants
                    .entry(print)
                    .or_default()
                    .push((ri as u32, mat as u8, pair as u16));
            }
        }
    }
    let wanted: u64 = wants.values().map(|v| v.len() as u64).sum();

    // One sequential pass over the model's experts, a chunk at a time.
    const CHUNK: usize = 64;
    let mut by_kind: BTreeMap<(String, usize, usize), Tally> = BTreeMap::new();
    let mut by_layer: Vec<[Tally; 3]> = vec![Default::default(); m.layers];
    let (mut scalar_done, mut scalar_equal) = (0u64, 0u64);
    let mut seen: HashMap<[u8; PRINT], (usize, usize)> = HashMap::new();
    let (mut ambiguous, mut matched) = (0u64, 0u64);
    let mut distinct = std::collections::HashSet::new();
    let mut first_bad: Option<String> = None;
    let mut out = Vec::new();
    let mut reference = Vec::new();
    let mut buf = Vec::new();
    let mut n = 0usize;
    let started = Instant::now();
    for (si, s) in m.stacks.iter().enumerate() {
        if s.kind == 0 && s.layer % 8 == 0 {
            eprintln!(
                "layer {} of {} ({:.0} s)",
                s.layer,
                m.layers,
                started.elapsed().as_secs_f64()
            );
        }
        for first in (0..s.n_expert).step_by(CHUNK) {
            let count = CHUNK.min(s.n_expert - first);
            m.read(s, first, count, &mut buf)?;
            for e in first..first + count {
                let w = &buf[(e - first) * s.expert_bytes..][..s.expert_bytes];
                let print: [u8; PRINT] = w[..PRINT].try_into().expect("a print");
                if seen.insert(print, (si, e)).is_some() {
                    ambiguous += 1;
                }
                let Some(list) = wants.remove(&print) else {
                    continue;
                };
                for (ri, mat, pair) in list {
                    let (rec, mat, pair) = (&recs[ri as usize], mat as usize, pair as usize);
                    let (t, j) = (pair / rec.n_used, pair % rec.n_used);
                    let (id, blocks) = (rec.ids[pair], rec.k / 32);
                    matched += 1;
                    let Some(format) = s
                        .format
                        .filter(|_| s.ty == rec.ty && s.k == rec.k && s.rows == rec.rows)
                    else {
                        shape += 1;
                        continue;
                    };
                    let col = t * rec.n_x + if rec.n_x == 1 { 0 } else { j };
                    let xq =
                        &rec.xq[col * blocks * Q8_1_BLOCK_BYTES..][..blocks * Q8_1_BLOCK_BYTES];
                    out.clear();
                    out.resize(rec.rows, 0.0f32);
                    gpu_exact::matvec(format, w, xq, rec.k, &mut out)
                        .map_err(|e| anyhow!("{e}"))?;
                    let bytes: Vec<u8> =
                        out.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                    let equal = dump::digest(&bytes) == rec.digests[mat][pair];
                    let host = id >= rec.vram_below;
                    distinct.insert((si, e));
                    by_kind
                        .entry((format!("{format:?}"), s.kind, rec.n_tok))
                        .or_default()
                        .add(equal, host);
                    by_layer[s.layer][s.kind].add(equal, host);
                    n += 1;
                    if scalar_every != 0 && n.is_multiple_of(scalar_every) {
                        reference.clear();
                        reference.resize(rec.rows, 0.0f32);
                        gpu_exact::matvec_scalar(format, w, xq, rec.k, &mut reference)
                            .map_err(|e| anyhow!("{e}"))?;
                        scalar_done += 1;
                        scalar_equal += u64::from(
                            out.iter()
                                .zip(&reference)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                        );
                    }
                    if !equal && first_bad.is_none() {
                        let at = rec.full.get(mat).and_then(|full| {
                            let gpu = &full[pair * rec.rows..][..rec.rows];
                            (0..rec.rows)
                                .find(|&r| gpu[r].to_bits() != out[r].to_bits())
                                .map(|r| {
                                    let differing = (0..rec.rows)
                                        .filter(|&r| gpu[r].to_bits() != out[r].to_bits())
                                        .count();
                                    format!(
                                        "row {r}: GPU {:#010x} ({:e}), host {:#010x} ({:e}); {differing} of {} rows differ",
                                        gpu[r].to_bits(),
                                        gpu[r],
                                        out[r].to_bits(),
                                        out[r],
                                        rec.rows
                                    )
                                })
                        });
                        first_bad = Some(format!(
                            "record {} ({} tokens), layer {} {} ({format:?}) expert {e}, token {t} pair {j}: {}",
                            rec.seq,
                            rec.n_tok,
                            s.layer,
                            KINDS[s.kind],
                            at.unwrap_or_else(
                                || "digests differ (a record without its rows)".into()
                            )
                        ));
                    }
                }
            }
        }
    }
    let unknown = wanted - matched;
    println!(
        "model: {} layers, {} expert matrices, {ambiguous} with first {PRINT} bytes seen before (not identifiable); read in {:.0} s",
        m.layers,
        seen.len() as u64 + ambiguous,
        started.elapsed().as_secs_f64()
    );

    println!(
        "records: {records} ({})",
        steps
            .iter()
            .map(|(t, n)| format!("{n} of {t} token{}", if *t == 1 { "" } else { "s" }))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "{:<6} {:<5} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "type", "op", "tokens", "products", "bit-equal", "differ", "host-slot"
    );
    let mut total = Tally::default();
    for ((format, kind, n_tok), t) in &by_kind {
        println!(
            "{:<6} {:<5} {:>6} {:>10} {:>10} {:>10} {:>10}",
            format,
            KINDS[*kind],
            n_tok,
            t.products,
            t.equal,
            t.products - t.equal,
            t.host
        );
        total.products += t.products;
        total.equal += t.equal;
        total.host += t.host;
    }
    if per_layer {
        println!("layer  gate products/equal   up products/equal   down products/equal");
        for (l, k) in by_layer.iter().enumerate() {
            let ty = |kind: usize| {
                m.stacks
                    .iter()
                    .find(|s| s.layer == l && s.kind == kind)
                    .and_then(|s| s.format)
                    .map_or("?".to_string(), |f| format!("{f:?}"))
            };
            println!(
                "{l:>5}  {:<5} {:>7}/{:<7}  {:<5} {:>7}/{:<7}  {:<5} {:>7}/{:<7}",
                ty(0),
                k[0].products,
                k[0].equal,
                ty(1),
                k[1].products,
                k[1].equal,
                ty(2),
                k[2].products,
                k[2].equal
            );
        }
    }
    let layers_all = by_layer
        .iter()
        .filter(|k| k.iter().all(|t| t.products > 0 && t.products == t.equal))
        .count();
    println!(
        "layers with every recorded gate, up and down product bit-equal: {layers_all} of {}",
        m.layers
    );
    println!(
        "total: {} expert products ({} distinct expert matrices, {} read from a host slot), {} bit-equal, {} differ",
        total.products,
        distinct.len(),
        total.host,
        total.equal,
        total.products - total.equal
    );
    println!(
        "not compared: {dead} dead padding pairs, {bad_id} ids outside the stack, {unknown} experts not identified, {shape} of another shape or type"
    );
    println!(
        "SwiGLU of the raw gate and up sums equal to the op's output: {act_equal} of {act_pairs} pairs"
    );
    if scalar_done > 0 {
        println!(
            "scalar reference equal to the vector kernel: {scalar_equal} of {scalar_done} products"
        );
    }
    if let Some(bad) = &first_bad {
        println!("first difference: {bad}");
    }
    let yes = total.products > 0
        && ambiguous == 0
        && total.products == total.equal
        && act_pairs == act_equal
        && scalar_done == scalar_equal;
    println!("BIT-EQUAL: {}", if yes { "yes" } else { "NO" });
    Ok(())
}

/// One expert's three matrices, copied out of the file.
struct Expert {
    w: [Vec<u8>; 3],
    format: [QuantFormat; 3],
}

/// [`time`]'s memory traffic beside the pool.
struct Load {
    gbs: f64,
    cpus: Vec<u32>,
}

/// Streams a buffer of its own at `gbs` GB/s on `cpu` until `stop`; adds the bytes it read to
/// `read`.
fn stream(cpu: u32, gbs: f64, stop: &std::sync::atomic::AtomicBool, read: &AtomicU64) {
    const CHUNK: usize = 1 << 20;
    if let Err(e) = legion::hw::numa::pin_current_thread(&[cpu]) {
        eprintln!("CPU {cpu}: {e}");
    }
    // 1 GiB: well past the caches, so every pass comes from DRAM.
    let buf: Vec<u64> = (0..(1usize << 30) / 8).map(|i| i as u64).collect();
    let started = Instant::now();
    let (mut bytes, mut acc) = (0u64, 0u64);
    'run: loop {
        for chunk in buf.chunks(CHUNK / 8) {
            if stop.load(Ordering::Relaxed) {
                break 'run;
            }
            acc = acc.wrapping_add(chunk.iter().fold(0u64, |a, &v| a ^ v));
            bytes += CHUNK as u64;
            // Keep to the rate: wait while ahead of it.
            while bytes as f64 > gbs * 1e9 * started.elapsed().as_secs_f64() {
                if stop.load(Ordering::Relaxed) {
                    break 'run;
                }
                std::hint::spin_loop();
            }
        }
    }
    std::hint::black_box(acc);
    read.fetch_add(bytes, Ordering::Relaxed);
}

fn time(
    model: &Path,
    n_experts: usize,
    cpus: &[u32],
    threads: &[usize],
    step: usize,
    passes: usize,
    load: &Load,
) -> anyhow::Result<()> {
    if load.gbs > 0.0 && load.cpus.is_empty() {
        bail!("--load-gbs needs --load-cpus");
    }
    if let Some(c) = load.cpus.iter().find(|c| cpus.contains(c)) {
        bail!("CPU {c} is both the pool's and a reader's");
    }
    let m = Model::open(model)?;
    let usable: Vec<usize> = (0..m.layers)
        .filter(|&l| {
            m.stacks[3 * l..3 * l + 3]
                .iter()
                .all(|s| s.format.is_some())
        })
        .collect();
    if usable.is_empty() || n_experts == 0 {
        bail!("no layers of Q4_K, Q5_K, Q5_1 or Q8_0 experts");
    }
    let (k, ffn) = (m.stacks[0].k, m.stacks[0].rows);
    let experts: Vec<Expert> = (0..n_experts)
        .map(|i| -> anyhow::Result<Expert> {
            let l = usable[i * 7 % usable.len()];
            let s = &m.stacks[3 * l..3 * l + 3];
            let e = i * 37 % s[0].n_expert;
            let mut w: [Vec<u8>; 3] = Default::default();
            for (kind, w) in w.iter_mut().enumerate() {
                m.read(&s[kind], e, 1, w)?;
            }
            Ok(Expert {
                w,
                format: [0, 1, 2].map(|kind| s[kind].format.expect("usable")),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    let bytes: usize = experts
        .iter()
        .map(|e| e.w.iter().map(Vec::len).sum::<usize>())
        .sum();
    println!(
        "{} experts of {} layers in memory ({:.0} MB, {:.2} MB each); rows of {k} (gate, up) and {ffn} (down) values; vector kernels: {}",
        experts.len(),
        usable.len(),
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / experts.len() as f64,
        gpu_exact::simd()
    );
    // Activation blocks of plausible scales (the time does not depend on their values).
    let acts = |values: usize, seed: u64| -> Vec<u8> {
        let mut s = seed;
        let mut v = vec![0u8; values / 32 * Q8_1_BLOCK_BYTES];
        for b in v.chunks_exact_mut(Q8_1_BLOCK_BYTES) {
            for q in &mut b[4..] {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *q = (s >> 56) as u8;
            }
            b[0..2].copy_from_slice(&gpu_exact::f32_to_f16(0.01).to_le_bytes());
            b[2..4].copy_from_slice(&gpu_exact::f32_to_f16(0.3).to_le_bytes());
        }
        v
    };
    let (x_in, x_act) = (acts(k, 1), acts(ffn, 2));
    // Phase 0: gate and up (the SwiGLU op's sums); phase 1: down.
    let run = |e: &Expert, phase: usize, out: &mut [f32]| -> u32 {
        if phase == 0 {
            let (g, u) = out[..2 * ffn].split_at_mut(ffn);
            gpu_exact::matvec(e.format[0], &e.w[0], &x_in, k, g).expect("shape");
            gpu_exact::matvec(e.format[1], &e.w[1], &x_in, k, u).expect("shape");
            g[0].to_bits() ^ u[0].to_bits()
        } else {
            gpu_exact::matvec(e.format[2], &e.w[2], &x_act, ffn, &mut out[..k]).expect("shape");
            out[0].to_bits()
        }
    };
    println!(
        "{:>7} {:>14} {:>14} {:>14} {:>16} {:>12}",
        "threads",
        "gate+up us",
        "down us",
        "expert us",
        &format!("{step} experts ms"),
        "MB/s read"
    );
    let jobs = passes * experts.len();
    let (stop, read) = (std::sync::atomic::AtomicBool::new(false), AtomicU64::new(0));
    let mut load_secs = 0.0;
    std::thread::scope(|bg| {
        if load.gbs > 0.0 {
            let gbs = load.gbs / load.cpus.len() as f64;
            for &cpu in &load.cpus {
                let (stop, read) = (&stop, &read);
                bg.spawn(move || stream(cpu, gbs, stop, read));
            }
            // The readers fill their buffers first.
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        let load_started = Instant::now();
        measure(
            &experts,
            &run,
            cpus,
            threads,
            jobs,
            step,
            bytes,
            k.max(2 * ffn),
        );
        load_secs = load_started.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
    });
    if load.gbs > 0.0 {
        println!(
            "readers beside the pool on CPUs {:?}: {:.1} GB/s on average (asked {:.1}; includes their 2 s head start)",
            load.cpus,
            read.load(Ordering::Relaxed) as f64 / 1e9 / (load_secs + 2.0),
            load.gbs
        );
    }
    println!(
        "(wall time per expert with all the pool's threads busy: a step's experts take the per-step column once its threads are awake)"
    );
    Ok(())
}

/// [`time`]'s table: each pool size's time per expert, gate and up then down.
#[allow(clippy::too_many_arguments)]
fn measure(
    experts: &[Expert],
    run: &(impl Fn(&Expert, usize, &mut [f32]) -> u32 + Sync),
    cpus: &[u32],
    threads: &[usize],
    jobs: usize,
    step: usize,
    bytes: usize,
    out_len: usize,
) {
    for &t in threads {
        if t == 0 || t > cpus.len() {
            println!("{t:>7} (skipped: {} CPUs given)", cpus.len());
            continue;
        }
        let mut wall = [0.0f64; 2];
        let sink = AtomicU64::new(0);
        for (phase, wall) in wall.iter_mut().enumerate() {
            let next = AtomicUsize::new(0);
            let ready = std::sync::Barrier::new(t + 1);
            let mut started = Instant::now();
            std::thread::scope(|scope| {
                for &cpu in &cpus[..t] {
                    let (next, ready, sink, experts, run) = (&next, &ready, &sink, &experts, &run);
                    scope.spawn(move || {
                        if let Err(e) = legion::hw::numa::pin_current_thread(&[cpu]) {
                            eprintln!("CPU {cpu}: {e}");
                        }
                        let mut out = vec![0.0f32; out_len];
                        // Warm the code and this thread's buffers before the clock starts.
                        let mut acc = run(&experts[0], phase, &mut out);
                        ready.wait();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= jobs {
                                break;
                            }
                            acc ^= run(&experts[i % experts.len()], phase, &mut out);
                        }
                        sink.fetch_xor(u64::from(acc), Ordering::Relaxed);
                    });
                }
                ready.wait();
                started = Instant::now();
            });
            *wall = started.elapsed().as_secs_f64();
        }
        let us = |w: f64| w / jobs as f64 * 1e6;
        let per_expert = us(wall[0]) + us(wall[1]);
        println!(
            "{t:>7} {:>14.1} {:>14.1} {:>14.1} {:>16.2} {:>12.0}",
            us(wall[0]),
            us(wall[1]),
            per_expert,
            per_expert * step as f64 / 1e3,
            bytes as f64 / experts.len() as f64 / per_expert,
        );
        std::hint::black_box(sink.load(Ordering::Relaxed));
    }
}

pub fn run(args: ExpertCpuProofArgs) -> anyhow::Result<()> {
    match args.what {
        What::Selftest { rounds } => {
            let rows = gpu_exact::self_check(rounds, 0x37).map_err(anyhow::Error::msg)?;
            println!(
                "bit-exact: {rows} rows, vector kernels {} against the scalar reference",
                if gpu_exact::simd() {
                    "(AVX2, FMA, F16C)"
                } else {
                    "absent: the reference"
                }
            );
            Ok(())
        }
        What::Check {
            model,
            dump,
            scalar_every,
            layers,
        } => check(&model, &dump, scalar_every, layers),
        What::Time {
            model,
            experts,
            cpus,
            threads,
            step,
            passes,
            load_gbs,
            load_cpus,
        } => {
            let load = Load {
                gbs: load_gbs,
                cpus: load_cpus,
            };
            time(&model, experts, &cpus, &threads, step, passes, &load)
        }
    }
}
