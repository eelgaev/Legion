//! Hybrid decode (#268): the host computes a share of each MoE layer's host-resident experts
//! (the cache's misses, whose bytes otherwise cross the link to the GPU) on its own cores while
//! the GPU computes the rest. Opt-in (`--cpu-expert-share`): the host kernels (`crate::cpu`)
//! sum in another order than the GPU's, so outputs change; with the share at 0 nothing here
//! runs and the graph is unchanged. CUDA only (the GPU must read host memory coherently); steps
//! of at most [`ops::MAX_TOKENS`] tokens (decode and MTP verify steps).
//!
//! Adapted from Strata-AC922's POWER9 CPU expert share (eelgaev/Strata-AC922 bd1a96d, 1c9c0ba,
//! c77c2b0, MIT): `--pcie-frac` there leaves the GPU a share of each layer's distinct misses and
//! the CPU computes the others on a worker pool per NUMA node, two hardware threads per core.
//!
//! # Which experts the host computes
//!
//! Per layer and step, of the distinct host-resident experts the step routes to (`n` of them),
//! the host takes `c = floor(share * n + d)`, `d` a fixed per-layer dither in `[0, 1)` (so the
//! layers together take `share` of the misses, and no layer's count strays by more than one).
//! It takes the experts fewest tokens use (a multi-token expert costs the host per token, the
//! GPU per byte), ties by expert id. The rule is [`select`]; the GPU computes it in the
//! `publish` kernel (`src/cuda/cpu_share.cu`), so nothing waits for a plan.
//!
//! # Per layer, in the stage's stream
//!
//! 1. `publish` (an external op after the router): computes the split, copies the activations
//!    and the host's expert ids into the device's exchange buffer (host memory both sides
//!    read), raises the "ready" flag to the step's sequence number with a stream memory
//!    operation, and outputs the ids with the host's pairs marked [`HOST_PAIR`] (the expert
//!    matvecs skip ids outside the stack). At enqueue time it queues the job for the host pool.
//! 2. The expert matvecs (`crate::model::core::expert_mv`) compute the other pairs.
//! 3. The pool waits for "ready", computes gate, up, SwiGLU and down of its experts with the
//!    host kernels (each expert once for all its tokens), writes the rows into the exchange
//!    buffer and sets "done", never before the rows are written (it waits for "ready" as long
//!    as it takes, warning now and then). A layer whose split gives the host nothing is done
//!    as soon as the pool sees "ready".
//! 4. `collect` (a kernel the down projection's op launches after its own): writes the host's
//!    rows into the op's output once "done" reached the sequence number. The kernel's blocks
//!    of the host's pairs, and its first block whatever its pair, poll the flag (a
//!    system-scope acquire load, a short sleep apart) while the other blocks end at once
//!    (their rows are there), so the kernel ends after "done" as a stream wait would. `LEGION_DEBUG_SHARE_STREAM_WAIT=1`: the stream
//!    waits for the flag before the launch (a stream memory operation), the way before.
//!
//! Layers follow each other in a device's stream, so one exchange buffer per device suffices:
//! a layer's `publish` runs after the previous layer's `collect`, which ran after the host was
//! done with it.
//!
//! # The exact share (x86 with a discrete GPU, #37)
//!
//! On x86 the host computes its experts' matvecs with the GPU kernels' own bits
//! ([`crate::cpu::gpu_exact`]), so the share changes no output. What the host cannot reproduce
//! (the activations' Q8_1 quantization and SwiGLU, which the GPU computes with its function
//! unit's approximations) stays on the GPU, so each layer exchanges twice:
//!
//! 1. `publish` writes x as the expert kernels' Q8_1 blocks; the host takes the first
//!    `floor(share * n + d)` distinct host experts in routing order (the link reads the last
//!    ones, as Strata's `--pcie-frac` leaves them).
//! 2. The pool computes its pairs' raw gate and up sums ([`crate::cpu::gpu_exact::matvec`]) into
//!    the exchange buffer and raises "gate/up done" while the GPU's SwiGLU op runs the others.
//! 3. `activate` (a kernel the SwiGLU op launches after its own): waits for "gate/up done" as
//!    `collect` waits for "done", makes the host pairs' SwiGLU from the sums (the kernels'
//!    `swiglu`) and their Q8_1 blocks, and raises "act ready".
//! 4. The pool computes its pairs' down rows from those blocks while the GPU's down op runs the
//!    others; `collect` is as above.
//!
//! The share of each layer is fixed (`--cpu-expert-share <x>`) or follows the measured costs
//! (`--cpu-expert-share auto`): a layer's first job splits at `L / (L + C)`, with `L` the link's
//! time for an expert's bytes (a copy timed at setup) and `C` the pool's time per expert; after
//! that each job the pool takes part in moves the share a little toward the side that waited
//! ([`pool::nudge`]: the GPU flags its arrival at `activate`), so it settles where the GPU and the
//! pool wait for each other about as often. Steps of 5 or 7 tokens leave every expert to the
//! GPU (the kernels' bits at those widths were not checked).
//!
//! # The graph's layout
//!
//! ggml-cuda fuses some MoE ops (the weighted sum of the experts, #472) only where the
//! tensors' memory ranges allow, so a node more or less in the step graph's memory moves which
//! layers fuse, and with it the bits. The share adds no memory of its own: `publish` writes
//! its ids at a place of the layer's own outside the graph (`expert_share_ids`,
//! [`super::experts::TieredLayer::share_ids`]); `activate` and `collect` are no nodes but
//! kernels the SwiGLU and down ops launch after theirs (ops wrapping the share-off ones, the
//! same nodes, sources and params); the SwiGLU op reads the same pre-quantized x as without
//! the share; and the down op keeps the router's ids as an unread source, so they are freed
//! where the share-off down op frees them. So the graph lays out its memory as the share-off
//! one, node for node. (In-place nodes of their own did not do: ggml's allocator never lets a
//! view's reader take the view's memory, so the weighted sum's `mul` got memory of its own.)
//!
//! # The bounded host tier
//!
//! With the bounded host tier ([`super::host_tier`]) a layer's tiered tensor is
//! `[S VRAM slots | R RAM slots | gap | staging places]`, and the tier's fetch op (before
//! `publish` in the stream) has turned every miss `S + expert` into `S + i`: `i < R` a RAM
//! slot, `i >= R + gap` a staging place. The share's host tensors are then the RAM slots
//! (`n_expert = R`, base the first RAM slot), so the host takes only misses in a RAM slot
//! (`publish` tests `i < R`); a staging place (an expert read this step with no RAM slot left
//! for it, or the next layer's read ahead of a prompt chunk) stays on the link. An expert the
//! fetch read from the checkpoint into a RAM slot this step is the host's like any other: the
//! fetch returns after its reads are done.
//!
//! Why nothing writes a RAM slot the pool still reads. The pool reads layer `l`'s slots only
//! between `l`'s `publish` and its "done", and `l`'s `collect` waits for "done" in the stream.
//! The writers of a layer's RAM slots:
//!
//! 1. Layer `l`'s fetch, which reads missed experts into the slots the step does not use
//!    (least recently used first). The same step's fetch runs before `publish`; the next
//!    step's runs after `cudaStreamSynchronize` (or as a host function in the stream), so
//!    after this step's `collect`, after "done". Other layers' fetches write only their own
//!    layers' slots.
//! 2. A long prompt's staged pass loads (`Held::Load`) and the load-time prefill: steps of
//!    more than [`ops::MAX_TOKENS`] tokens, or before any step, which the share never takes
//!    part in; the after-step before them waits for the device (`tier_settled`), so for every
//!    `collect` and every "done" before it.
//! 3. A VRAM swap's write-back (a device-to-host copy of the expert the VRAM slot gives up)
//!    lands in a slot `reserve` emptied: it holds no expert and stays pinned until the copy
//!    landed, so no fetch maps a miss to it nor takes it as a victim, and the pool reads only
//!    slots a fetch mapped. Swaps are planned in the after-step, behind the device wait.
//! 4. A swap's source slot (the RAM copy of the expert a VRAM slot takes) is only read by its
//!    host-to-device copy and pinned until it lands; it is dropped when the swaps landed, in
//!    an after-step, with the device idle.
//!
//! The staging sets the layers take in turn (`il % sets`), the reads ahead into them and the
//! mixed model's zeroed pads write staging places only, which the pool never reads.

pub mod bench;
pub mod ops;
pub mod pool;

use std::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::QuantFormat;
use crate::ggml::types;

/// [`threshold`]'s scale: a share of 1.0.
pub const THRESHOLD_ONE: u32 = 1024;

/// The id the `publish` op puts in place of a pair the host computes: outside every expert
/// stack, so the GPU's expert matvecs skip it (and leave its output rows alone).
pub const HOST_PAIR: i32 = -2;

/// A share in `0..=1` as the integer threshold [`select`] works with.
pub fn threshold(share: f32) -> u32 {
    if share.is_nan() {
        return 0;
    }
    (share.clamp(0.0, 1.0) * THRESHOLD_ONE as f32).round() as u32
}

/// A layer's dither in `0..THRESHOLD_ONE` (see the module docs). The CUDA kernel computes the
/// same function.
pub fn dither(layer: u32) -> u32 {
    let mut h = layer.wrapping_mul(0x9E37_79B1) ^ 0x85EB_CA77;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h & (THRESHOLD_ONE - 1)
}

/// The split of one layer's step: for each pair of `ids` (tiered ids, `[n_tok][n_used]`
/// dense; `n_slots` VRAM slots, then `n_expert` host experts), whether the host computes it.
/// Every pair of an expert gets the same answer. The CUDA `publish` kernel computes the same.
pub fn select(ids: &[i32], n_slots: i32, n_expert: i32, threshold: u32, layer: u32) -> Vec<bool> {
    let host = |id: i32| id >= n_slots && id - n_slots < n_expert;
    let users = |id: i32| ids.iter().filter(|&&o| o == id).count() as u64;
    // Distinct host experts, as (users, expert) keys.
    let mut keys: Vec<(u64, i32)> = ids
        .iter()
        .enumerate()
        .filter(|&(i, &id)| host(id) && !ids[..i].contains(&id))
        .map(|(_, &id)| (users(id), id))
        .collect();
    keys.sort_unstable();
    let n = keys.len() as u32;
    let c = ((threshold.min(THRESHOLD_ONE) * n + dither(layer)) / THRESHOLD_ONE) as usize;
    let taken = &keys[..c.min(keys.len())];
    ids.iter()
        .map(|&id| host(id) && taken.iter().any(|k| k.1 == id))
        .collect()
}

/// The host kernels' format for a ggml type, if they take it.
pub fn format_of(ty: u32) -> Option<QuantFormat> {
    match ty {
        types::Q4_K => Some(QuantFormat::Q4K),
        types::Q5_K => Some(QuantFormat::Q5K),
        types::Q6_K => Some(QuantFormat::Q6K),
        types::Q8_0 => Some(QuantFormat::Q8_0),
        types::Q5_1 => Some(QuantFormat::Q5_1),
        _ => None,
    }
}

/// One MoE layer's experts in host memory, as the host kernels read them: expert `e` of kind
/// `k` (up, gate, down) is `expert_bytes[k]` bytes at `base[k] + e * expert_bytes[k]`.
#[derive(Debug, Clone, Copy)]
pub struct HostExperts {
    pub base: [usize; 3],
    pub expert_bytes: [usize; 3],
    pub up_gate: QuantFormat,
    pub down: QuantFormat,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_expert: usize,
}

impl HostExperts {
    /// The bytes of expert `e` of kind `k`.
    ///
    /// # Safety
    /// `base` must point at `n_expert` experts of host memory that stays mapped and unchanged
    /// while the result is used.
    pub unsafe fn expert(&self, k: usize, e: usize) -> &[u8] {
        assert!(e < self.n_expert, "expert {e} of {}", self.n_expert);
        // SAFETY: the caller's guarantee, `e` in range.
        unsafe {
            std::slice::from_raw_parts(
                (self.base[k] + e * self.expert_bytes[k]) as *const u8,
                self.expert_bytes[k],
            )
        }
    }
}

/// The settings of the share (`--cpu-expert-share`, `--cpu-expert-workers`,
/// `--cpu-expert-smt`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuShareConfig {
    /// Fraction of each layer's host-resident experts the host computes (0: off). With
    /// [`Self::auto`] the most it takes.
    pub share: f32,
    /// The exact share (x86): each layer's share from the measured link and pool costs (see the
    /// module docs), at most [`Self::share`]. Elsewhere the share is off.
    pub auto: bool,
    /// Worker threads per NUMA node's pool (0: [`Self::smt`] per core of the node's cores the
    /// process may use, less [`RESERVED_CORES`]).
    pub workers: usize,
    /// Hardware threads per core the pool uses (POWER9: SMT2 gave 1.95x one thread's kernel
    /// throughput in Strata-AC922).
    pub smt: usize,
}

/// Cores of each node the pool leaves to the engine thread, the stages' launch threads and
/// the rest of the process (with the default worker count).
pub const RESERVED_CORES: usize = 2;

impl Default for CpuShareConfig {
    fn default() -> Self {
        Self {
            share: 0.0,
            auto: false,
            workers: 0,
            smt: 2,
        }
    }
}

/// A step's sequence numbers, per device: `publish` takes the next one, `collect` waits for the
/// last one taken.
#[derive(Debug, Default)]
pub struct Sequence(AtomicU32);

impl Sequence {
    pub fn next(&self) -> u32 {
        self.0.fetch_add(1, Ordering::Relaxed).wrapping_add(1)
    }

    pub fn last(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }
}

/// The pool's CPUs on a node: of `cpus` (the node's CPUs the process may use), cores in order,
/// the first `smt` CPUs of each, less the last `reserved` cores; `siblings` gives a CPU's core
/// (its SMT siblings, itself included; empty when unknown: every CPU its own core).
pub fn pool_cpus(
    cpus: &[u32],
    smt: usize,
    reserved: usize,
    siblings: impl Fn(u32) -> Vec<u32>,
) -> Vec<u32> {
    let mut cores: Vec<Vec<u32>> = Vec::new();
    for &c in cpus {
        let sib = siblings(c);
        if let Some(core) = cores.iter_mut().find(|k| k.iter().any(|o| sib.contains(o))) {
            core.push(c);
        } else {
            cores.push(vec![c]);
        }
    }
    let keep = cores.len().saturating_sub(reserved).max(1);
    cores
        .into_iter()
        .take(keep)
        .flat_map(|core| core.into_iter().take(smt.max(1)))
        .collect()
}

/// Page-locked host memory mapped for a device (`cudaHostAlloc`, mapped and portable),
/// allocated by a thread on the device's NUMA node so its pages land there; zeroed.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct HostBuffer {
    pub host: usize,
    pub device: u64,
}

#[cfg(feature = "cuda")]
mod host_alloc {
    use std::ffi::{c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn cudaSetDevice(device: c_int) -> c_int;
        pub fn cudaHostAlloc(p: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
        pub fn cudaHostGetDevicePointer(
            d: *mut *mut c_void,
            h: *mut c_void,
            flags: c_uint,
        ) -> c_int;
        pub fn cudaFreeHost(p: *mut c_void) -> c_int;
    }

    /// `cudaHostAllocPortable | cudaHostAllocMapped`.
    pub const FLAGS: c_uint = 1 | 2;
}

#[cfg(feature = "cuda")]
impl HostBuffer {
    fn new(
        device: usize,
        node: u32,
        topo: &crate::hw::numa::Topology,
        bytes: usize,
    ) -> Result<Self, String> {
        use host_alloc::*;
        let (host, dev) =
            crate::hw::numa::run_on_node(topo, node, move || -> Result<_, String> {
                let mut h = std::ptr::null_mut();
                let mut d = std::ptr::null_mut();
                // SAFETY: plain allocation calls; the memory is zeroed before anything reads it.
                unsafe {
                    if cudaSetDevice(device as std::ffi::c_int) != 0
                        || cudaHostAlloc(&mut h, bytes, FLAGS) != 0
                        || cudaHostGetDevicePointer(&mut d, h, 0) != 0
                    {
                        return Err(format!(
                            "cpu share: cannot allocate a {bytes}-byte exchange buffer"
                        ));
                    }
                    std::ptr::write_bytes(h as *mut u8, 0, bytes);
                }
                Ok((h as usize, d as u64))
            })?;
        Ok(Self { host, device: dev })
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        // SAFETY: memory from `cudaHostAlloc`, freed once; nothing uses it after the share.
        unsafe {
            host_alloc::cudaFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// One device's part of a share: its CUDA index, the NUMA node of its host memory, and per
/// layer (index = layer id) the experts of the layers it runs.
pub struct DevicePlan {
    pub device: usize,
    pub node: u32,
    pub layers: Vec<Option<HostExperts>>,
    pub n_embd: usize,
    /// The exact share (x86, see the module docs): the host computes the GPU kernels' bits.
    pub exact: bool,
    /// The layers' host experts are the bounded host tier's RAM slots (not every expert).
    pub tiered: bool,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// The exact share's part of a device's exchange buffer, after [`Layout`]'s: its flags (a cache
/// line each; "act arrived" for debug statistics), the step's distinct host experts (from
/// `publish`), each layer's share of 1024 (the host writes it, `publish` reads it) and the
/// pool's cost per expert, x's Q8_1 blocks, the host pairs' raw gate and up sums and the
/// SwiGLU's Q8_1 blocks.
pub(crate) struct ExactLayout {
    pub gu_done: usize,
    pub act_ready: usize,
    pub act_arrived: usize,
    pub n_miss: usize,
    pub thr: usize,
    pub cost: usize,
    pub xq: usize,
    pub gu: usize,
    pub aq: usize,
    pub n_ff: usize,
    pub bytes: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl ExactLayout {
    /// After the first `start` bytes, for experts of `n_ff` rows at most and `n_layers` layers.
    pub(crate) fn new(start: usize, n_embd: usize, n_ff: usize, n_layers: usize) -> Self {
        let pairs = ops::MAX_TOKENS * ops::MAX_USED;
        let align = |b: usize| b.next_multiple_of(256);
        let q8 = |k: usize| k / 32 * crate::cpu::gpu_exact::Q8_1_BLOCK_BYTES;
        let gu_done = align(start);
        let thr = gu_done + 512;
        let cost = thr + align(n_layers * 4);
        let xq = cost + align(n_layers * 4);
        let gu = xq + align(ops::MAX_TOKENS * q8(n_embd));
        let aq = gu + align(pairs * 2 * n_ff * 4);
        Self {
            gu_done,
            act_ready: gu_done + 128,
            act_arrived: gu_done + 256,
            n_miss: gu_done + 384,
            thr,
            cost,
            xq,
            gu,
            aq,
            n_ff,
            bytes: aq + pairs * q8(n_ff),
        }
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// Byte offsets in a device's exchange buffer: the flags (a cache line each; "arrived" for
/// debug statistics), the host's expert ids, the activations and the host's output rows.
pub(crate) struct Layout {
    pub ready: usize,
    pub done: usize,
    pub arrived: usize,
    pub ids: usize,
    pub x: usize,
    pub out: usize,
    pub bytes: usize,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl Layout {
    pub(crate) fn new(n_embd: usize) -> Self {
        let (max_tok, max_used) = (ops::MAX_TOKENS, ops::MAX_USED);
        let align = |b: usize| b.next_multiple_of(256);
        let ids = 256;
        let x = ids + align(max_tok * max_used * 4);
        let out = x + align(max_tok * n_embd * 4);
        Self {
            ready: 0,
            done: 128,
            arrived: 192,
            ids,
            x,
            out,
            bytes: out + max_tok * max_used * n_embd * 4,
        }
    }
}

/// Every CPU of the cores the pools run on (SMT siblings included): the engine's own threads
/// keep off them ([`leave_pool_cores`]).
static POOL_CORES: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

/// The engine's thread placement (per stage, its CPUs) without the share's pool cores, where
/// that leaves a stage any CPU. Unchanged when the share is off.
pub fn leave_pool_cores(stage_cpus: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    let pool = POOL_CORES.lock().unwrap();
    if pool.is_empty() {
        return stage_cpus;
    }
    stage_cpus
        .into_iter()
        .map(|cpus| {
            let off: Vec<u32> = cpus.iter().copied().filter(|c| !pool.contains(c)).collect();
            if off.is_empty() { cpus } else { off }
        })
        .collect()
}

/// `cpus` as a Linux CPU list (`80-81,84-85`).
pub fn cpulist(cpus: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        out.push(if j == i {
            cpus[i].to_string()
        } else {
            format!("{}-{}", cpus[i], cpus[j])
        });
        i = j + 1;
    }
    out.join(",")
}

/// Sets up the share on `devices`: an exchange buffer per device (host memory of its node,
/// mapped for it) and a worker pool per node, registered for the graph ops ([`ops`]).
#[cfg(feature = "cuda")]
pub fn setup(cfg: &CpuShareConfig, devices: Vec<DevicePlan>) -> Result<(), String> {
    use std::collections::HashMap;
    use std::sync::Arc;

    ops::clear();
    POOL_CORES.lock().unwrap().clear();
    if threshold(cfg.share) == 0 {
        return Ok(());
    }
    let topo = crate::hw::numa::Topology::detect().map_err(|e| format!("cpu share: {e}"))?;
    let siblings = |c: u32| crate::hw::host_core::siblings(c).to_vec();
    let mut pools: HashMap<u32, Arc<pool::Pool>> = HashMap::new();
    for d in devices {
        let pool = match pools.get(&d.node) {
            Some(p) => p.clone(),
            None => {
                let cpus = crate::hw::numa::helper_cpus(&topo, d.node);
                if cpus.is_empty() {
                    return Err(format!("cpu share: no CPUs on node {}", d.node));
                }
                let picked = pool_cpus(&cpus, cfg.smt, RESERVED_CORES, siblings);
                let n = match cfg.workers {
                    0 => picked.len(),
                    n => n,
                };
                let used: Vec<u32> = picked.iter().copied().cycle().take(n).collect();
                {
                    let mut cores = POOL_CORES.lock().unwrap();
                    for &c in &used {
                        let sib = siblings(c);
                        cores.extend(if sib.is_empty() { vec![c] } else { sib });
                    }
                    cores.sort_unstable();
                    cores.dedup();
                }
                let mut sorted = used.clone();
                sorted.sort_unstable();
                sorted.dedup();
                tracing::info!(
                    node = d.node,
                    threads = n,
                    smt = cfg.smt,
                    cpus = %cpulist(&sorted),
                    "cpu expert share: worker pool (Strata-AC922's POWER9 CPU expert share)"
                );
                let p = Arc::new(pool::Pool::new(&used, n));
                pools.insert(d.node, p.clone());
                p
            }
        };
        let lay = Layout::new(d.n_embd);
        let n_ff = d.layers.iter().flatten().map(|l| l.n_ff).max().unwrap_or(0);
        let exact_lay = d
            .exact
            .then(|| ExactLayout::new(lay.bytes, d.n_embd, n_ff, d.layers.len()));
        let bytes = exact_lay.as_ref().map_or(lay.bytes, |e| e.bytes);
        let buf = HostBuffer::new(d.device, d.node, &topo, bytes)?;
        let base = buf.host;
        let exact = match &exact_lay {
            None => None,
            Some(e) => {
                let first = d.layers.iter().flatten().next().copied();
                let us_per_byte = first
                    .and_then(|l| {
                        link::us_per_byte(d.device, l.base[0], l.expert_bytes[0] * l.n_expert)
                    })
                    .unwrap_or(link::FALLBACK_US_PER_BYTE);
                let thr = threshold(cfg.share);
                // Where the two sides take as long: a pool expert costs a link expert at first.
                let start = if cfg.auto {
                    thr.min(THRESHOLD_ONE / 2)
                } else {
                    thr
                };
                for il in 0..d.layers.len() {
                    // SAFETY: the buffer's own words, nothing else uses it yet.
                    unsafe {
                        (*((base + e.thr + il * 4) as *const AtomicU32))
                            .store(start, Ordering::Relaxed);
                        *((base + e.cost + il * 4) as *mut f32) = 0.0;
                    }
                }
                let per_expert = first.map_or(0.0, |l| {
                    us_per_byte * l.expert_bytes.iter().sum::<usize>() as f64
                });
                tracing::info!(
                    device = d.device,
                    link_gb_s = 1e-3 / us_per_byte,
                    link_us_per_expert = per_expert,
                    share = %if cfg.auto {
                        "auto".to_string()
                    } else {
                        cfg.share.to_string()
                    },
                    "cpu expert share: exact (the GPU kernels' bits on the host)"
                );
                Some(pool::ExactExchange {
                    gu_done: (base + e.gu_done) as *const AtomicU32,
                    act_ready: (base + e.act_ready) as *const AtomicU32,
                    act_arrived: (base + e.act_arrived) as *const AtomicU32,
                    n_miss: (base + e.n_miss) as *const AtomicU32,
                    thr: (base + e.thr) as *const AtomicU32,
                    cost: (base + e.cost) as *mut f32,
                    n_layers: d.layers.len(),
                    xq: (base + e.xq) as *const u8,
                    gu: (base + e.gu) as *mut f32,
                    aq: (base + e.aq) as *const u8,
                    n_ff,
                    auto: cfg.auto,
                    cap: thr,
                    link_us_per_byte: us_per_byte as f32,
                    all_upto: pool::all_upto(cfg.auto, d.tiered),
                })
            }
        };
        let exchange = pool::Exchange {
            ready: (base + lay.ready) as *const AtomicU32,
            done: (base + lay.done) as *const AtomicU32,
            arrived: (base + lay.arrived) as *const AtomicU32,
            ids: (base + lay.ids) as *const i32,
            x: (base + lay.x) as *const f32,
            out: (base + lay.out) as *mut f32,
            max_tok: ops::MAX_TOKENS,
            max_used: ops::MAX_USED,
            n_embd: d.n_embd,
            exact,
        };
        let queue = pool.add_queue();
        ops::register(
            d.device,
            Arc::new(ops::DeviceShare {
                exchange,
                device_base: buf.device,
                layout: lay,
                exact: exact_lay,
                seq: Sequence::default(),
                layers: d.layers,
                pool,
                queue,
                _buffer: buf,
            }),
        );
    }
    Ok(())
}

/// The link's time per byte from host memory to a device: what the exact share's split weighs
/// a link expert by.
#[cfg(feature = "cuda")]
mod link {
    use std::ffi::{c_int, c_void};
    use std::time::Instant;

    /// When the timing fails: about a PCIe 4.0 x16 link's 25.6 GB/s.
    pub const FALLBACK_US_PER_BYTE: f64 = 1.0 / 25_600.0;
    /// Bytes a timed copy takes at most.
    const PROBE: usize = 32 << 20;

    unsafe extern "C" {
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, size: usize) -> c_int;
        fn cudaFree(p: *mut c_void) -> c_int;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> c_int;
        fn cudaDeviceSynchronize() -> c_int;
    }

    /// `cudaMemcpyHostToDevice`.
    const H2D: c_int = 1;

    /// The best of three copies of up to [`PROBE`] bytes at `src` (pinned host memory of
    /// `bytes`, a layer's expert tensor) to `device`, in microseconds per byte.
    pub fn us_per_byte(device: usize, src: usize, bytes: usize) -> Option<f64> {
        let n = bytes.min(PROBE);
        if n < (1 << 20) {
            return None;
        }
        let mut was = 0;
        let mut dst = std::ptr::null_mut();
        // SAFETY: plain runtime calls on a buffer allocated here; `src` is `n` live bytes.
        unsafe {
            if cudaGetDevice(&mut was) != 0 || cudaSetDevice(device as c_int) != 0 {
                return None;
            }
            let mut best = f64::MAX;
            if cudaMalloc(&mut dst, n) == 0 {
                for i in 0..4 {
                    let t = Instant::now();
                    if cudaMemcpy(dst, src as *const c_void, n, H2D) != 0
                        || cudaDeviceSynchronize() != 0
                    {
                        best = f64::MAX;
                        break;
                    }
                    if i > 0 {
                        best = best.min(t.elapsed().as_secs_f64() * 1e6);
                    }
                }
                cudaFree(dst);
            }
            cudaSetDevice(was);
            (best < f64::MAX).then(|| best / n as f64)
        }
    }
}

#[cfg(test)]
mod tests;
