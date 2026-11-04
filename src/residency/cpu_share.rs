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
//!    buffer and sets "done". A layer whose split gives the host nothing sets "done" in the
//!    publish kernel already.
//! 4. `collect` (an external op on the down projection's output): the stream waits for "done"
//!    to equal the sequence number (a stream memory operation: the GPU stalls, it does not
//!    spin), then copies the host's rows into the output.
//!
//! Layers follow each other in a device's stream, so one exchange buffer per device suffices:
//! a layer's `publish` runs after the previous layer's `collect`, which ran after the host was
//! done with it.

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
    /// Fraction of each layer's host-resident experts the host computes (0: off).
    pub share: f32,
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
        let buf = HostBuffer::new(d.device, d.node, &topo, lay.bytes)?;
        let base = buf.host;
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
        };
        let queue = pool.add_queue();
        ops::register(
            d.device,
            Arc::new(ops::DeviceShare {
                exchange,
                device_base: buf.device,
                layout: lay,
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

#[cfg(test)]
mod tests;
