//! Communication micro-benchmarks for multi-GPU layouts (`legion comm-bench`, #668): how long
//! one all-reduce of a hidden-size vector takes through GLM's exchange
//! ([`crate::model::glm5next::par::exchange::reduce`]), a plain `cudaMemcpyPeer` baseline, and
//! how fast each GPU reads host memory of each NUMA node. The numbers decide between flat tensor
//! parallelism over all GPUs and tensor parallelism inside each NVLink pair with a layer split
//! across the sockets.
//!
//! The device set is the process's CUDA devices: choose it with `CUDA_VISIBLE_DEVICES` (0,1 a
//! pair, 0,2 across the sockets, 0,1,2,3 all of them) and the CPU side with `numactl`.

use std::sync::{Arc, Barrier};
use std::time::Instant;

use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, types};
use crate::model::glm5next::par::exchange::{self, Exchange};

/// One reduce shape: `e` values per token, `n` tokens (`[e, n]` F32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub e: i64,
    pub n: i64,
}

/// What a timed graph chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// `len` exchange reduces, each ordered after the one before (as a layer's are).
    Reduce,
    /// `len` adds of the same tensors: the cost of a graph node without communication.
    Add,
}

/// One shape's timing over all the devices.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeResult {
    pub shape: Shape,
    pub chain: Chain,
    /// Nodes per graph.
    pub len: usize,
    /// Per timed graph run: the slowest device's wall time, in microseconds.
    pub runs_us: Vec<f64>,
}

impl ExchangeResult {
    /// The payload of one reduce from one device, in bytes.
    pub fn bytes(&self) -> usize {
        (self.shape.e * self.shape.n) as usize * 4
    }

    /// Quantile `q` (0..=1) of the per-node time, in microseconds.
    pub fn per_node_us(&self, q: f64) -> f64 {
        quantile(&self.runs_us, q) / self.len as f64
    }
}

/// Quantile `q` (0..=1, nearest rank) of `v`; NaN when empty.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let i = ((s.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    s[i]
}

/// The devices a benchmark runs on.
pub enum Devices {
    /// `n` CPU "GPUs" (threads), exchanging through plain host memory (tests).
    Cpu(usize),
    /// CUDA devices `0..nodes.len()`, device `d`'s exchange inbox on NUMA node `nodes[d]`; the
    /// GPUs `peers` names write into each other's VRAM inboxes (#671), slots of `peer_slot`
    /// bytes (None: as big as the host ones); reduces of `scatter_min` bytes or more go as a
    /// reduce-scatter and an all-gather (#686).
    #[cfg(feature = "cuda")]
    Cuda {
        nodes: Vec<u32>,
        peers: exchange::PeerWrites,
        peer_slot: Option<usize>,
        scatter_min: usize,
    },
}

impl Devices {
    fn n_dev(&self) -> usize {
        match self {
            Devices::Cpu(n) => *n,
            #[cfg(feature = "cuda")]
            Devices::Cuda { nodes, .. } => nodes.len(),
        }
    }

    /// The exchange, its VRAM inbox slots as big as its host ones unless given (every shape may
    /// go GPU to GPU).
    fn exchange(&self, slot_bytes: usize) -> Result<Arc<Exchange>, String> {
        let n_dev = self.n_dev();
        let bounds: Vec<u32> = (0..=n_dev as u32).collect();
        match self {
            Devices::Cpu(_) => Ok(Exchange::host(n_dev, slot_bytes, &bounds)),
            #[cfg(feature = "cuda")]
            Devices::Cuda {
                nodes,
                peers,
                peer_slot,
                scatter_min,
            } => {
                let peer_slot = peer_slot.unwrap_or(slot_bytes);
                let mut ex = Exchange::pinned(nodes, slot_bytes, &bounds, *peers, peer_slot)?;
                Arc::get_mut(&mut ex)
                    .expect("a new exchange")
                    .set_scatter_min(*scatter_min);
                Ok(ex)
            }
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn backend(&self, d: usize) -> Result<Backend, String> {
        match self {
            Devices::Cpu(_) => {
                let mut b = Backend::cpu().map_err(|e| e.to_string())?;
                b.set_threads(1);
                Ok(b)
            }
            #[cfg(feature = "cuda")]
            Devices::Cuda { .. } => {
                crate::ggml::backend_by_name(&format!("CUDA{d}")).map_err(|e| e.to_string())
            }
        }
    }
}

/// Times graphs of `len` chained nodes ([`Chain`]) of every shape on every device at once, one
/// host thread per device as the parallel decode runs them: `warmup` untimed runs, then `iters`
/// timed ones, all devices starting each run together.
pub fn exchange_bench(
    devs: &Devices,
    shapes: &[Shape],
    chains: &[Chain],
    len: usize,
    warmup: usize,
    iters: usize,
) -> Result<Vec<ExchangeResult>, String> {
    let n_dev = devs.n_dev();
    if n_dev < 2 || len == 0 || len >= 1 << 16 {
        return Err("comm-bench: two devices or more, 1..65535 nodes a graph".into());
    }
    let slot = shapes
        .iter()
        .map(|s| exchange::reduce_bytes(s.e, s.n))
        .max()
        .unwrap_or(16)
        .next_multiple_of(4096);
    let ex = devs.exchange(slot)?;
    let mut backends = (0..n_dev)
        .map(|d| devs.backend(d))
        .collect::<Result<Vec<_>, _>>()?;
    let mut step = 0u64;
    let mut out = Vec::new();
    for &shape in shapes {
        for &chain in chains {
            let barrier = Barrier::new(n_dev);
            let first_step = step + 1;
            step += (warmup + iters) as u64;
            let per_dev: Vec<Result<Vec<f64>, String>> = std::thread::scope(|s| {
                let handles: Vec<_> = backends
                    .iter_mut()
                    .enumerate()
                    .map(|(d, backend)| {
                        let (ex, barrier) = (&ex, &barrier);
                        s.spawn(move || {
                            run_device(
                                backend, ex, barrier, d, shape, chain, len, warmup, iters,
                                first_step,
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("a device thread panicked"))
                    .collect()
            });
            let mut times = Vec::with_capacity(n_dev);
            for r in per_dev {
                times.push(r?);
            }
            if ex.failed() {
                return Err(format!(
                    "comm-bench: an exchange timed out ({shape:?}, {chain:?})"
                ));
            }
            let runs_us = (0..iters)
                .map(|i| times.iter().map(|t| t[i]).fold(0.0, f64::max))
                .collect();
            out.push(ExchangeResult {
                shape,
                chain,
                len,
                runs_us,
            });
        }
    }
    Ok(out)
}

/// One device's thread of [`exchange_bench`]: builds its graph, then runs it in step with the
/// others. After an error it keeps meeting the barriers (without running) so the others finish.
#[allow(clippy::too_many_arguments)]
fn run_device(
    backend: &mut Backend,
    ex: &Exchange,
    barrier: &Barrier,
    d: usize,
    shape: Shape,
    chain: Chain,
    len: usize,
    warmup: usize,
    iters: usize,
    first_step: u64,
) -> Result<Vec<f64>, String> {
    let ctx = Context::new(len + 4, len + 4);
    let x = ctx.new_tensor(types::F32, &[shape.e, shape.n]).as_input();
    let mut prev: Option<Tensor<'_>> = None;
    for _ in 0..len {
        prev = Some(match chain {
            Chain::Reduce => exchange::reduce(&ctx, ex, d, x, prev),
            Chain::Add => ctx.add(x, prev.unwrap_or(x)),
        });
    }
    let last = prev.expect("len > 0").as_output();
    let mut graph = ctx.new_graph(len + 4);
    graph.expand(last);
    let mut err = ctx
        .alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .err()
        .map(|e| e.to_string());
    if err.is_none() {
        x.set_f32(&bench_values(d, shape));
    }
    let mut times = Vec::with_capacity(iters);
    for i in 0..warmup + iters {
        barrier.wait();
        if d == 0 {
            ex.begin_step(first_step + i as u64);
        }
        barrier.wait();
        if err.is_some() {
            continue;
        }
        let t0 = Instant::now();
        if let Err(e) = backend.compute(&graph) {
            err = Some(e.to_string());
            continue;
        }
        if i >= warmup {
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    if err.is_none() && chain == Chain::Reduce && !ex.failed() {
        let want = reduce_sum(ex.n_dev(), shape);
        let got = last.to_f32();
        if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
            err = Some(format!(
                "the reduce's value {i} is {}, not {} ({shape:?})",
                got[i], want[i]
            ));
        }
    }
    match err {
        Some(e) => Err(format!("device {d}: {e}")),
        None => Ok(times),
    }
}

/// Device `d`'s part of a timed reduce of `shape`.
fn bench_values(d: usize, shape: Shape) -> Vec<f32> {
    (0..shape.e * shape.n)
        .map(|i| (d as f32 + 1.0) * ((i % 7) as f32 - 3.0) / 8.0)
        .collect()
}

/// What a reduce of `n_dev` devices' [`bench_values`] gives: their sum in device order.
pub fn reduce_sum(n_dev: usize, shape: Shape) -> Vec<f32> {
    let parts: Vec<Vec<f32>> = (0..n_dev).map(|d| bench_values(d, shape)).collect();
    (0..parts[0].len())
        .map(|i| parts[1..].iter().fold(parts[0][i], |acc, p| acc + p[i]))
        .collect()
}

/// One-way `cudaMemcpyPeer` from `src` to `dst`.
#[derive(Clone, Debug, PartialEq)]
pub struct PeerResult {
    pub src: usize,
    pub dst: usize,
    pub bytes: usize,
    /// `cudaDeviceCanAccessPeer(src, dst)`: a direct link (else the driver stages the copy).
    pub direct: bool,
    /// Back-to-back async copies on one stream, timed by events: per copy, microseconds.
    pub async_us: f64,
    /// A synchronous `cudaMemcpyPeer`, host-timed: median, microseconds.
    pub sync_us: f64,
}

/// Every device sending `bytes` to every other one at once (async copies on each source's
/// stream), then the host waiting for all: a host-driven exchange round.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundResult {
    pub bytes: usize,
    pub runs_us: Vec<f64>,
}

#[cfg(feature = "cuda")]
mod sys {
    use std::os::raw::{c_char, c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn cudaSetDevice(d: c_int) -> c_int;
        pub fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        pub fn cudaFree(p: *mut c_void) -> c_int;
        pub fn cudaMemset(p: *mut c_void, v: c_int, n: usize) -> c_int;
        pub fn cudaDeviceCanAccessPeer(can: *mut c_int, d: c_int, peer: c_int) -> c_int;
        pub fn cudaDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> c_int;
        pub fn cudaGetLastError() -> c_int;
        pub fn cudaMemcpyPeer(
            dst: *mut c_void,
            dst_dev: c_int,
            src: *const c_void,
            src_dev: c_int,
            n: usize,
        ) -> c_int;
        pub fn cudaMemcpyPeerAsync(
            dst: *mut c_void,
            dst_dev: c_int,
            src: *const c_void,
            src_dev: c_int,
            n: usize,
            s: *mut c_void,
        ) -> c_int;
        pub fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> c_int;
        pub fn cudaStreamDestroy(s: *mut c_void) -> c_int;
        pub fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
        pub fn cudaEventCreate(e: *mut *mut c_void) -> c_int;
        pub fn cudaEventDestroy(e: *mut c_void) -> c_int;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
        pub fn cudaEventSynchronize(e: *mut c_void) -> c_int;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> c_int;
        pub fn cudaGetErrorString(e: c_int) -> *const c_char;
    }

    pub fn rt(e: c_int, what: &str) -> Result<(), String> {
        if e == 0 {
            return Ok(());
        }
        // SAFETY: a static NUL-terminated string.
        let msg = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(e)) }.to_string_lossy();
        Err(format!("{what}: {msg}"))
    }
}

/// `cudaErrorPeerAccessAlreadyEnabled`.
#[cfg(feature = "cuda")]
const PEER_ALREADY_ENABLED: std::os::raw::c_int = 704;

/// Per device: a send buffer, a receive buffer per peer, a stream.
#[cfg(feature = "cuda")]
struct PeerBufs {
    send: Vec<*mut std::os::raw::c_void>,
    recv: Vec<Vec<*mut std::os::raw::c_void>>,
    stream: Vec<*mut std::os::raw::c_void>,
}

#[cfg(feature = "cuda")]
impl Drop for PeerBufs {
    fn drop(&mut self) {
        use sys::*;
        // SAFETY: what `peer_bench` allocated, each on its device; nothing runs any more.
        unsafe {
            for d in 0..self.send.len() {
                cudaSetDevice(d as i32);
                cudaFree(self.send[d]);
                for &r in &self.recv[d] {
                    cudaFree(r);
                }
                cudaStreamDestroy(self.stream[d]);
            }
        }
    }
}

/// The `cudaMemcpyPeer` baseline over CUDA devices `0..n_dev`: per ordered pair and size a
/// [`PeerResult`], per size a [`RoundResult`].
#[cfg(feature = "cuda")]
pub fn peer_bench(
    n_dev: usize,
    sizes: &[usize],
    iters: usize,
) -> Result<(Vec<PeerResult>, Vec<RoundResult>), String> {
    use std::ptr::null_mut;
    use sys::*;
    let max = sizes.iter().copied().max().unwrap_or(0).max(16);
    let iters = iters.max(1);
    let mut b = PeerBufs {
        send: vec![null_mut(); n_dev],
        recv: vec![vec![null_mut(); n_dev]; n_dev],
        stream: vec![null_mut(); n_dev],
    };
    let mut direct = vec![vec![false; n_dev]; n_dev];
    // SAFETY: plain runtime calls on buffers this function owns (freed by `b`'s drop).
    unsafe {
        for (d, can_reach) in direct.iter_mut().enumerate() {
            rt(cudaSetDevice(d as i32), "cudaSetDevice")?;
            rt(cudaMalloc(&mut b.send[d], max), "cudaMalloc")?;
            rt(cudaMemset(b.send[d], 1, max), "cudaMemset")?;
            for p in 0..n_dev {
                rt(cudaMalloc(&mut b.recv[d][p], max), "cudaMalloc")?;
            }
            rt(
                cudaStreamCreateWithFlags(&mut b.stream[d], 1),
                "cudaStreamCreate",
            )?;
            for p in (0..n_dev).filter(|&p| p != d) {
                let mut can = 0;
                rt(
                    cudaDeviceCanAccessPeer(&mut can, d as i32, p as i32),
                    "cudaDeviceCanAccessPeer",
                )?;
                if can != 0 {
                    let e = cudaDeviceEnablePeerAccess(p as i32, 0);
                    if e == PEER_ALREADY_ENABLED {
                        cudaGetLastError();
                    } else {
                        rt(e, "cudaDeviceEnablePeerAccess")?;
                    }
                }
                can_reach[p] = can != 0;
            }
        }
        let (mut e0, mut e1) = (null_mut(), null_mut());
        let mut pairs = Vec::new();
        let mut rounds = Vec::new();
        for &bytes in sizes {
            for (src, dst) in (0..n_dev).flat_map(|a| (0..n_dev).map(move |c| (a, c))) {
                if src == dst {
                    continue;
                }
                rt(cudaSetDevice(src as i32), "cudaSetDevice")?;
                rt(cudaEventCreate(&mut e0), "cudaEventCreate")?;
                rt(cudaEventCreate(&mut e1), "cudaEventCreate")?;
                let s = b.stream[src];
                let copy = || {
                    cudaMemcpyPeerAsync(
                        b.recv[dst][src],
                        dst as i32,
                        b.send[src],
                        src as i32,
                        bytes,
                        s,
                    )
                };
                for _ in 0..8 {
                    rt(copy(), "cudaMemcpyPeerAsync")?;
                }
                rt(cudaEventRecord(e0, s), "cudaEventRecord")?;
                for _ in 0..iters {
                    rt(copy(), "cudaMemcpyPeerAsync")?;
                }
                rt(cudaEventRecord(e1, s), "cudaEventRecord")?;
                rt(cudaEventSynchronize(e1), "cudaEventSynchronize")?;
                let mut ms = 0f32;
                rt(
                    cudaEventElapsedTime(&mut ms, e0, e1),
                    "cudaEventElapsedTime",
                )?;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                let mut sync = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    rt(
                        cudaMemcpyPeer(
                            b.recv[dst][src],
                            dst as i32,
                            b.send[src],
                            src as i32,
                            bytes,
                        ),
                        "cudaMemcpyPeer",
                    )?;
                    sync.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                pairs.push(PeerResult {
                    src,
                    dst,
                    bytes,
                    direct: direct[src][dst],
                    async_us: ms as f64 * 1e3 / iters as f64,
                    sync_us: quantile(&sync, 0.5),
                });
            }
            let mut runs = Vec::with_capacity(iters);
            for i in 0..iters + 8 {
                let t0 = Instant::now();
                for src in 0..n_dev {
                    for dst in (0..n_dev).filter(|&p| p != src) {
                        rt(
                            cudaMemcpyPeerAsync(
                                b.recv[dst][src],
                                dst as i32,
                                b.send[src],
                                src as i32,
                                bytes,
                                b.stream[src],
                            ),
                            "cudaMemcpyPeerAsync",
                        )?;
                    }
                }
                for &s in &b.stream {
                    rt(cudaStreamSynchronize(s), "cudaStreamSynchronize")?;
                }
                if i >= 8 {
                    runs.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }
            rounds.push(RoundResult {
                bytes,
                runs_us: runs,
            });
        }
        Ok((pairs, rounds))
    }
}

/// One [`host_read_bench`] measurement: `(device, node, rep, Some((in place GB/s, copy
/// GB/s)))`, `None` where the probe could not run.
#[cfg(feature = "cuda")]
pub type HostRead = (usize, u32, usize, Option<(f64, f64)>);

/// Bandwidth of each CUDA device `0..n_dev` reading host memory of each node in `nodes`
/// ([`crate::residency::experts::probe_device`]: random 3 MiB expert-sized rows read in place by
/// a kernel, and DMA copies of them), `reps` times each: `(device, node, rep, in place GB/s,
/// copy GB/s)`.
#[cfg(feature = "cuda")]
pub fn host_read_bench(n_dev: usize, nodes: &[u32], reps: usize) -> Vec<HostRead> {
    let mut out = Vec::new();
    for d in 0..n_dev {
        for &node in nodes {
            for r in 0..reps {
                let p = crate::residency::experts::probe_device(d, node);
                out.push((d, node, r, p.map(|p| (p.in_place_gbs, p.copy_gbs))));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
