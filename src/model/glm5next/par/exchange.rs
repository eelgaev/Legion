//! The exchange between the GPUs of a parallel decode step ([`super`]): an external op
//! ([`crate::ggml::ExternalOp`], the CUDA kernel in `src/cuda/glm_exchange.cu`) through which
//! each GPU hands its part of a split tensor to the others, so that every GPU then holds all of
//! it. Our own exchange rather than NCCL's all-gather: a step makes a few hundred small ones
//! (a few KiB per GPU each), where the cost is the latency of the handshake, not bandwidth.
//!
//! **Memory.** Every GPU has an inbox in pinned host memory on its own NUMA node, mapped for
//! all GPUs: a header (per parity, writer and chunk a flag, per writer the time this GPU waited
//! for it, an error word) and per parity and writer a payload slot. Writers write payloads and
//! flags across NVLink (to the CPU of their own socket) or across the sockets' link; the reader
//! polls its own node's memory. A GPU that peers write to directly ([`PeerWrites`], #671: by
//! default the other GPU of its NVLink pair) also has an inbox in its own VRAM: the same flags,
//! and per parity and such writer a slot of `peer_slot` bytes. Those writers' payloads that fit
//! go there, over the GPU-GPU link, and the reader polls its own VRAM; bigger ones (a long
//! prompt step's broadcast of the streams) take the host inbox.
//!
//! **Protocol.** Every exchange of a step gets a value `v`, the same on every GPU (they make the
//! same exchanges in the same order; [`Exchange::begin_step`] restarts the count at
//! `step << 16`). A GPU writes its payload into each reader's slot of parity `v & 1` in chunks
//! (#671, [`CHUNK_MIN`] bytes or more, at most [`MAX_CHUNKS`]): per chunk it copies, fences and
//! sets the chunk's flag there to `v`. Then it waits for the others' chunks in its own inboxes
//! and copies (or adds) each as its flag comes up. Many blocks share the chunks, so a big payload
//! moves at the links' bandwidth, not one block's. Every exchange but a gather to one GPU is a
//! barrier: nobody leaves it before everybody arrived. A slot of parity `p` is written again two
//! exchanges later, by which time every reader finished reading it (the writer has the readers'
//! payloads of the exchange in between, which they send only after finishing the one before).
//! The last exchange of a step (the LM head's gather to the last GPU) is no barrier, but the
//! host waits for every GPU at the end of a step.
//!
//! **Kinds** ([`Kind`]):
//! - *gather*: up to four segments, each a tensor split by rows over the GPUs ([`super::rows`]).
//!   GPU `d` has its rows of each, `[rows_d, n]` F32; the result is the segments one after
//!   another, each `[total, n]`. To all GPUs, or to one (`root`).
//! - *bcast*: one GPU (the owner) has a whole `[total, n]` tensor; afterwards all do.
//! - *select*: the routed experts' outputs. GPU `d` computed the columns (token, choice) whose
//!   expert it owns ([`super::expert_owner`]) in its `[e, n_used, n]` tensor (the others are
//!   untouched garbage), and one GPU (`sh`) the shared expert's `[e, n]`. The result,
//!   `[e, n_used * n + n]`, holds every chosen expert's column from its owner and the shared
//!   expert's output after them.
//! - *reduce*: every GPU has a partial sum `[e, n]` of the same tensor (its heads' share of an
//!   attention output projection, its rows' share of an FFN's down projection, its experts');
//!   the result, on every GPU, is the sum of all of them, added in GPU order (`0 + 1 + 2 + ..`,
//!   F32), so every GPU gets the same bits, whichever chunk or link the parts came by. A reduce
//!   of a chunk or more per GPU on three GPUs or more ([`Exchange::scatters`], #686) goes as a
//!   reduce-scatter and an all-gather: chunk `c` is GPU `c % n_dev`'s, which alone gets the
//!   others' parts of it, adds them in GPU order and sends the sum to the others, into the
//!   places of their slots the first sends leave alone (the owner's own chunks). Half the bytes
//!   on four GPUs, the same bits.
//!
//! A select goes as before #671: one chunk, host inboxes, a block per GPU.
//!
//! Gather, bcast and select copy the values as they are: the result is bitwise what one GPU
//! computing everything would have. A reduce adds them up in another order than one GPU's
//! matmul would (the tensor-parallel layers, [`super`]).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::ggml::{Context, EXTERNAL_PARAMS, ExternalOp, Tensor, external_params, ffi, types};

static EXCHANGE: ExternalOp = ExternalOp {
    name: "glm_exchange",
    cpu: cpu_exchange,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_exchange),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The most GPUs an exchange connects.
pub const MAX_DEV: usize = 8;
/// The most segments of a gather.
pub const MAX_SEG: usize = 4;
/// The most chunks a payload goes in.
pub const MAX_CHUNKS: usize = 64;
/// The smallest chunk (bytes): payloads up to it go in one, as one decode token's reduce.
pub const CHUNK_MIN: usize = 32 << 10;
/// The smallest reduce (bytes) that goes as a reduce-scatter and an all-gather (#686), on three
/// GPUs or more and once every GPU owns a chunk of it (four chunks of [`CHUNK_MIN`] or less on
/// four GPUs: 7 tokens of GLM): on n GPUs `2 / n` of the bytes every GPU sending its whole part to
/// every other one moves, for a second hop's latency. On four V100s (`comm-bench exchange`)
/// that wins from 12 tokens (90 -> 57 us) to a 2048-token step (7.3 -> 3.6 ms), so no other
/// threshold.
pub const SCATTER_MIN: usize = 0;
/// The most blocks of a launch (all resident at once on any GPU this runs on: no block of
/// one GPU spins while a block another GPU waits for cannot start).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const MAX_BLOCKS: usize = 64;
/// Bytes between two flags (each in a cache line of its own).
const FLAG_SEP: usize = 128;
/// The flags' bytes: per parity, writer and chunk.
const FLAGS: usize = 2 * MAX_DEV * MAX_CHUNKS * FLAG_SEP;
/// Where the per-writer waits (u64 ns) start in the host inbox's header.
const WAIT_AT: usize = FLAGS;
/// Where the error word (u32, set when a wait timed out) is in the host inbox's header.
const ERR_AT: usize = FLAGS + 512;
/// The host inbox header's size: the flags, the waits, the error word.
const HDR: usize = FLAGS + 4096;
/// The VRAM inbox header's size: the flags.
const PHDR: usize = FLAGS;
/// How long a GPU waits for another before it gives up on the exchange (the step then fails
/// instead of hanging): no healthy GPU is a second late.
const TIMEOUT: Duration = Duration::from_secs(10);

/// What an exchange does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Kind {
    Gather = 0,
    Bcast = 1,
    Select = 2,
    Reduce = 3,
}

/// Which GPUs write their payloads straight into another GPU's VRAM over a GPU-GPU link
/// (`--glm-exchange-peer`, #671) instead of into its inbox in host memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum PeerWrites {
    /// None: every payload goes through host memory (as before #671).
    Off,
    /// GPUs of one NUMA node that can reach each other's memory: an AC922's NVLink pairs.
    #[default]
    Pairs,
    /// Every GPU that can reach another's memory, across the sockets too.
    All,
}

/// Per reader `r`: bit `w` set when GPU `w` writes into `r`'s VRAM inbox, for GPUs on NUMA
/// nodes `nodes` where `w` can reach `r`'s memory when `can(w, r)`. GPUs of no known node
/// ([`crate::residency::fallback::NO_NODE`]) are never a pair: no shared node is known.
pub fn peer_masks(
    mode: PeerWrites,
    nodes: &[u32],
    can: impl Fn(usize, usize) -> bool,
) -> [u32; MAX_DEV] {
    let mut m = [0u32; MAX_DEV];
    let n = nodes.len().min(MAX_DEV);
    for (r, mask) in m.iter_mut().enumerate().take(n) {
        for w in (0..n).filter(|&w| w != r) {
            let on = match mode {
                PeerWrites::Off => false,
                PeerWrites::Pairs => {
                    crate::residency::fallback::known_node(nodes[r]).is_some()
                        && nodes[w] == nodes[r]
                        && can(w, r)
                }
                PeerWrites::All => can(w, r),
            };
            if on {
                *mask |= 1 << w;
            }
        }
    }
    m
}

/// The exchange memory and state of one engine's GPUs.
pub struct Exchange {
    n_dev: usize,
    slot_bytes: usize,
    /// [`super::expert_bounds`], for selects.
    bounds: [i32; MAX_DEV + 1],
    /// Per GPU: its inbox (host memory every GPU reaches at this address).
    inbox: Vec<usize>,
    /// Per GPU: its VRAM inbox (0: none), its writers ([`peer_masks`]), and the bytes of a slot
    /// there.
    peer_in: Vec<usize>,
    peer_mask: [u32; MAX_DEV],
    peer_slot: usize,
    /// The smallest chunk ([`CHUNK_MIN`]; tests make it small).
    chunk_min: usize,
    /// The smallest scattered reduce ([`SCATTER_MIN`]; tests and `comm-bench` change it).
    scatter_min: usize,
    /// Per GPU: the exchanges it made so far this step (its next `v`).
    count: [AtomicU64; MAX_DEV],
    /// Owns the inboxes' memory.
    _mem: Mem,
}

enum Mem {
    /// Plain host memory (CPU graphs, tests); the VRAM inboxes as host memory too.
    Host(#[allow(dead_code)] Vec<Vec<u64>>),
    /// The contexts holding the buffers: pinned host memory on each GPU's node and the VRAM
    /// inboxes.
    #[cfg(feature = "cuda")]
    Pinned(#[allow(dead_code)] std::sync::Mutex<Vec<Context>>),
}

// SAFETY: the inboxes are plain memory the GPUs and threads synchronize on through the flags;
// the rest is atomics or read-only after construction.
unsafe impl Send for Exchange {}
unsafe impl Sync for Exchange {}

impl Exchange {
    /// The inbox size of `n_dev` GPUs' slots of `slot_bytes`.
    pub fn inbox_bytes(n_dev: usize, slot_bytes: usize) -> usize {
        HDR + 2 * n_dev * slot_bytes
    }

    /// The VRAM inbox size of a GPU with `writers` peers writing to it, slots of `peer_slot`.
    pub fn peer_inbox_bytes(writers: usize, peer_slot: usize) -> usize {
        PHDR + 2 * writers * peer_slot
    }

    #[allow(clippy::too_many_arguments)]
    fn with(
        n_dev: usize,
        slot_bytes: usize,
        bounds: &[u32],
        inbox: Vec<usize>,
        peer_in: Vec<usize>,
        peer_mask: [u32; MAX_DEV],
        peer_slot: usize,
        mem: Mem,
    ) -> Self {
        assert!((1..=MAX_DEV).contains(&n_dev) && bounds.len() == n_dev + 1);
        assert!(slot_bytes.is_multiple_of(16) && peer_slot.is_multiple_of(16));
        assert_eq!(peer_in.len(), n_dev);
        let mut b = [i32::MAX; MAX_DEV + 1];
        for (o, &v) in b.iter_mut().zip(bounds) {
            *o = v as i32;
        }
        Self {
            n_dev,
            slot_bytes,
            bounds: b,
            inbox,
            peer_in,
            peer_mask,
            peer_slot,
            chunk_min: CHUNK_MIN,
            scatter_min: SCATTER_MIN,
            count: Default::default(),
            _mem: mem,
        }
    }

    /// An exchange in plain host memory, for graphs on the CPU.
    pub fn host(n_dev: usize, slot_bytes: usize, bounds: &[u32]) -> Arc<Self> {
        Arc::new(Self::host_with(
            n_dev,
            slot_bytes,
            bounds,
            [0; MAX_DEV],
            0,
            CHUNK_MIN,
        ))
    }

    /// [`Self::host`] with "VRAM" inboxes (host memory too) for the writers `peer_mask` names,
    /// slots of `peer_slot`, and chunks of `chunk_min` bytes or more: the CPU twin of the
    /// peers' and the chunks' paths, for tests.
    pub fn host_with(
        n_dev: usize,
        slot_bytes: usize,
        bounds: &[u32],
        peer_mask: [u32; MAX_DEV],
        peer_slot: usize,
        chunk_min: usize,
    ) -> Self {
        assert!(chunk_min >= 16 && chunk_min.is_multiple_of(16));
        let words = Self::inbox_bytes(n_dev, slot_bytes).div_ceil(8);
        let mut bufs: Vec<Vec<u64>> = (0..n_dev).map(|_| vec![0u64; words]).collect();
        let inbox = bufs.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
        let mut peer_in = vec![0; n_dev];
        for (r, p) in peer_in.iter_mut().enumerate() {
            let writers = peer_mask[r].count_ones() as usize;
            if writers > 0 {
                let words = Self::peer_inbox_bytes(writers, peer_slot).div_ceil(8);
                let mut buf = vec![0u64; words];
                *p = buf.as_mut_ptr() as usize;
                bufs.push(buf);
            }
        }
        let mut ex = Self::with(
            n_dev,
            slot_bytes,
            bounds,
            inbox,
            peer_in,
            peer_mask,
            peer_slot,
            Mem::Host(bufs),
        );
        ex.chunk_min = chunk_min;
        ex
    }

    /// An exchange for CUDA devices `0..nodes.len()`: GPU `d`'s inbox in pinned host memory on
    /// NUMA node `nodes[d]`, mapped for every GPU, and with `peers` a VRAM inbox of slots of
    /// `peer_slot` bytes on every GPU others write to directly (peer access enabled for them).
    #[cfg(feature = "cuda")]
    pub fn pinned(
        nodes: &[u32],
        slot_bytes: usize,
        bounds: &[u32],
        peers: PeerWrites,
        peer_slot: usize,
    ) -> Result<Arc<Self>, String> {
        let n_dev = nodes.len();
        let devices: Vec<usize> = (0..n_dev).collect();
        let bytes = Self::inbox_bytes(n_dev, slot_bytes);
        let mut ctxs = Vec::new();
        let mut inbox = Vec::new();
        for &node in nodes {
            let buft = crate::residency::cuda::host_arena(node, &devices)?;
            let ctx = Context::new(1, 0);
            let t = ctx.new_tensor(types::I8, &[bytes as i64]);
            ctx.alloc_tensors(buft)
                .map_err(|e| format!("exchange inbox on node {node}: {e}"))?;
            // SAFETY: just allocated; zeroed before any GPU reads it.
            let p = unsafe { (*t.raw()).data as *mut u8 };
            unsafe { std::ptr::write_bytes(p, 0, bytes) };
            inbox.push(p as usize);
            ctxs.push(ctx);
        }
        let peer_slot = peer_slot.next_multiple_of(16);
        // The pairs by the GPUs' own nodes (the inboxes' `nodes` are all 0 under `--numa off`).
        let topo: Vec<u32> = (0..n_dev)
            .map(|d| crate::residency::cuda::device_numa_node(d).unwrap_or(nodes[d]))
            .collect();
        // Peer access both ways (the pair exchange's setup), so the masks are symmetric.
        use crate::model::core::pair::link;
        let mask = if peer_slot == 0 || n_dev < 2 {
            [0; MAX_DEV]
        } else {
            peer_masks(peers, &topo, |w, r| {
                link::can_access(w, r) && link::can_access(r, w)
            })
        };
        let pairs: Vec<(usize, usize)> = (0..n_dev)
            .flat_map(|r| (0..r).map(move |w| (w, r)))
            .filter(|&(w, r)| mask[r] >> w & 1 == 1)
            .collect();
        link::enable_peers(&pairs).map_err(|e| format!("glm exchange: {e}"))?;
        let mut peer_in = vec![0; n_dev];
        for r in 0..n_dev {
            let writers = mask[r].count_ones() as usize;
            if writers == 0 {
                continue;
            }
            let bytes = Self::peer_inbox_bytes(writers, peer_slot);
            // SAFETY: ggml's static buffer type of CUDA device `r`.
            let buft =
                std::ptr::NonNull::new(unsafe { ffi::ggml_backend_cuda_buffer_type(r as i32) })
                    .map(crate::ggml::BufferType)
                    .ok_or_else(|| format!("glm exchange: no buffer type for CUDA{r}"))?;
            let ctx = Context::new(1, 0);
            let t = ctx.new_tensor(types::I8, &[bytes as i64]);
            ctx.alloc_tensors(buft)
                .map_err(|e| format!("exchange VRAM inbox on CUDA{r}: {e}"))?;
            ctx.clear_buffers(0);
            // SAFETY: just allocated; its address on device `r`.
            peer_in[r] = unsafe { (*t.raw()).data as usize };
            ctxs.push(ctx);
        }
        Ok(Arc::new(Self::with(
            n_dev,
            slot_bytes,
            bounds,
            inbox,
            peer_in,
            mask,
            peer_slot,
            Mem::Pinned(std::sync::Mutex::new(ctxs)),
        )))
    }

    pub fn n_dev(&self) -> usize {
        self.n_dev
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Per reader: the GPUs writing into its VRAM inbox ([`peer_masks`]).
    pub fn peer_mask(&self) -> &[u32] {
        &self.peer_mask[..self.n_dev]
    }

    /// The bytes of a VRAM inbox slot: payloads up to this go to a peer's VRAM.
    pub fn peer_slot(&self) -> usize {
        self.peer_slot
    }

    /// Reduces of `bytes` or more go as a reduce-scatter and an all-gather ([`SCATTER_MIN`] by
    /// default; `usize::MAX`: none). The sums' bits are the same either way.
    pub fn set_scatter_min(&mut self, bytes: usize) {
        self.scatter_min = bytes;
    }

    /// Whether a reduce of `bytes` goes as a reduce-scatter and an all-gather: on three GPUs or
    /// more (on two it moves as many bytes as the plain one), of [`Self::set_scatter_min`]'s
    /// bytes or more, and a chunk for every GPU to own.
    pub fn scatters(&self, bytes: usize) -> bool {
        self.n_dev > 2
            && bytes >= self.scatter_min
            && bytes.div_ceil(self.chunk_bytes(bytes)) >= self.n_dev
    }

    /// The bytes of a chunk of an exchange whose biggest payload is `most` bytes, the same on
    /// every GPU: [`CHUNK_MIN`] (or the tests' `chunk_min`) or more, so the biggest payload goes
    /// in [`MAX_CHUNKS`] at most; a multiple of 16.
    fn chunk_bytes(&self, most: usize) -> usize {
        self.chunk_min
            .max(most.div_ceil(MAX_CHUNKS))
            .next_multiple_of(16)
    }

    /// Starts step `step` (every GPU idle): the exchanges' values restart at `step << 16`, and
    /// the error words clear.
    pub fn begin_step(&self, step: u64) {
        for c in &self.count[..self.n_dev] {
            c.store(step << 16, Ordering::Relaxed);
        }
        for d in 0..self.n_dev {
            self.err(d).store(0, Ordering::Relaxed);
        }
    }

    /// Whether a wait timed out since [`Self::begin_step`] (every GPU idle).
    pub fn failed(&self) -> bool {
        (0..self.n_dev).any(|d| self.err(d).load(Ordering::Relaxed) != 0)
    }

    /// Per GPU: the time it waited for the others since the last call (every GPU idle), in ns.
    pub fn take_waits(&self) -> Vec<u64> {
        self.take_wait_matrix()
            .iter()
            .map(|row| row.iter().sum())
            .collect()
    }

    /// Per GPU and writer: the time the GPU waited for that writer's flags since the last call
    /// (every GPU idle), in ns; `[reader][writer]`. A column's sum is how late the writer was.
    pub fn take_wait_matrix(&self) -> Vec<Vec<u64>> {
        (0..self.n_dev)
            .map(|d| {
                (0..self.n_dev)
                    .map(|w| self.wait(d, w).swap(0, Ordering::Relaxed))
                    .collect()
            })
            .collect()
    }

    fn header(&self, d: usize, at: usize) -> *mut u8 {
        (self.inbox[d] + at) as *mut u8
    }

    /// The flag of `writer`'s chunk `c` of parity `parity` in the inbox (host or VRAM) at `base`.
    fn flag_at(&self, base: usize, parity: usize, writer: usize, c: usize) -> &AtomicU64 {
        let at = ((parity * MAX_DEV + writer) * MAX_CHUNKS + c) * FLAG_SEP;
        // SAFETY: inside an inbox's header, 8-byte aligned; the memory lives as long as self.
        unsafe { &*((base + at) as *const AtomicU64) }
    }

    /// Where writer `w`'s payload of `bytes` for reader `r` goes, `(flags' base, slot)`: `r`'s
    /// VRAM inbox when `w` writes there and it fits, else `r`'s host inbox (the kernel's `route`).
    fn route(&self, w: usize, r: usize, parity: usize, bytes: usize) -> (usize, usize) {
        let m = self.peer_mask[r];
        if m >> w & 1 == 1 && self.peer_in[r] != 0 && bytes <= self.peer_slot {
            let idx = (m & ((1 << w) - 1)).count_ones() as usize;
            let np = m.count_ones() as usize;
            let base = self.peer_in[r];
            return (base, base + PHDR + (parity * np + idx) * self.peer_slot);
        }
        (self.inbox[r], self.slot(r, parity, w) as usize)
    }

    fn wait(&self, d: usize, writer: usize) -> &AtomicU64 {
        // SAFETY: as in `flag`.
        unsafe { &*(self.header(d, WAIT_AT + writer * 8) as *const AtomicU64) }
    }

    fn err(&self, d: usize) -> &AtomicU32 {
        // SAFETY: as in `flag`.
        unsafe { &*(self.header(d, ERR_AT) as *const AtomicU32) }
    }

    fn slot(&self, d: usize, parity: usize, writer: usize) -> *mut f32 {
        self.header(d, HDR + (parity * self.n_dev + writer) * self.slot_bytes) as *mut f32
    }

    /// GPU `me`'s next exchange value.
    fn next_v(&self, me: usize) -> u64 {
        self.count[me].fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The GPU owning expert `e`.
    fn owner(&self, e: i32) -> usize {
        (0..self.n_dev)
            .find(|&d| e < self.bounds[d + 1])
            .unwrap_or(self.n_dev - 1)
    }
}

/// Values per payload segment, padded so the next one starts 16-byte aligned.
fn padded(n: i64) -> i64 {
    (n + 3) / 4 * 4
}

/// The bytes GPU `d`'s gather payload takes.
pub fn gather_bytes(totals: &[i64], n: i64, n_dev: usize, d: usize) -> usize {
    totals
        .iter()
        .map(|&t| padded(super::rows(t, n_dev, d).count() as i64 * n) as usize * 4)
        .sum()
}

/// The bytes a select's payload can take at most (a GPU owning every chosen expert and running
/// the shared one): `e` values per column, `n_used` choices, `n` tokens.
pub fn select_bytes(e: i64, n_used: i64, n: i64) -> usize {
    ((n_used * n + n) * e * 4) as usize
}

/// The exchange's address as two params.
fn ex_params(ex: &Exchange) -> [i32; 2] {
    let p = ex as *const Exchange as u64;
    [p as u32 as i32, (p >> 32) as u32 as i32]
}

/// Parameters: kind, me, root / owner / sh (-1: none), segments, four totals, the exchange.
const P_KIND: usize = 0;
const P_ME: usize = 1;
const P_PEER: usize = 2;
const P_SEG: usize = 3;
const P_TOTAL: usize = 4;
const P_EX: usize = 8;
const _: () = assert!(P_EX + 2 <= EXTERNAL_PARAMS);

#[allow(clippy::too_many_arguments)]
fn node<'a>(
    ctx: &'a Context,
    ex: &Exchange,
    kind: Kind,
    me: usize,
    peer: Option<usize>,
    totals: &[i64],
    ne: [i64; 4],
    args: &[Tensor<'a>],
) -> Tensor<'a> {
    let mut p = [0i32; P_EX + 2];
    p[P_KIND] = kind as i32;
    p[P_ME] = me as i32;
    p[P_PEER] = peer.map_or(-1, |d| d as i32);
    p[P_SEG] = totals.len() as i32;
    for (i, &t) in totals.iter().enumerate() {
        p[P_TOTAL + i] = t as i32;
    }
    p[P_EX..].copy_from_slice(&ex_params(ex));
    ctx.external_with(&EXCHANGE, types::F32, ne, args, &p)
}

/// Gathers `parts` (GPU `me`'s rows of each segment, contiguous F32 `[rows, n]`; totals
/// `totals`) to every GPU, or to `root` only. The result: one segment `[total, n]`; several,
/// 1D, the segments one after another, each `[total, n]` ([`segment`]). `after`: a tensor the exchange is ordered after (the
/// previous exchange's result, say), so every GPU makes its exchanges in the same order.
pub fn gather<'a>(
    ctx: &'a Context,
    ex: &Exchange,
    me: usize,
    parts: &[Tensor<'a>],
    totals: &[i64],
    root: Option<usize>,
    after: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(!parts.is_empty() && parts.len() <= MAX_SEG && parts.len() == totals.len());
    let n = parts[0].ne()[1];
    for (p, &t) in parts.iter().zip(totals) {
        assert!(
            p.ty() == types::F32 && p.is_contiguous(),
            "gather: F32 contiguous parts"
        );
        assert_eq!(p.ne()[0], super::rows(t, ex.n_dev, me).count() as i64);
        assert_eq!(p.ne()[1] * p.ne()[2] * p.ne()[3], n);
    }
    for d in 0..ex.n_dev {
        assert!(
            gather_bytes(totals, n, ex.n_dev, d) <= ex.slot_bytes,
            "gather: slot too small"
        );
    }
    // One segment: `[total, n]`; several: 1D.
    let ne = match totals {
        [t] => [*t, n, 1, 1],
        _ => [totals.iter().map(|t| t * n).sum(), 1, 1, 1],
    };
    let mut args = parts.to_vec();
    args.extend(after);
    node(ctx, ex, Kind::Gather, me, root, totals, ne, &args)
}

/// Segment `s` of a [`gather`]'s result, `[totals[s], n]`.
pub fn segment<'a>(
    ctx: &'a Context,
    out: Tensor<'a>,
    totals: &[i64],
    n: i64,
    s: usize,
) -> Tensor<'a> {
    let off: i64 = totals[..s].iter().map(|t| t * n).sum();
    ctx.view_2d(
        out,
        [totals[s], n],
        totals[s] as usize * 4,
        off as usize * 4,
    )
}

/// Hands `x` (`[total, n]` F32 contiguous, on `owner` only: `None` on the others) to every GPU.
#[allow(clippy::too_many_arguments)]
pub fn bcast<'a>(
    ctx: &'a Context,
    ex: &Exchange,
    me: usize,
    owner: usize,
    x: Option<Tensor<'a>>,
    total: i64,
    n: i64,
    after: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert_eq!(x.is_some(), me == owner, "bcast: the owner has the tensor");
    if let Some(x) = x {
        assert!(x.ty() == types::F32 && x.is_contiguous() && x.elements() as i64 == total * n);
    }
    assert!(
        padded(total * n) as usize * 4 <= ex.slot_bytes,
        "bcast: slot too small"
    );
    let mut args: Vec<Tensor<'a>> = x.into_iter().collect();
    args.extend(after);
    node(
        ctx,
        ex,
        Kind::Bcast,
        me,
        Some(owner),
        &[total],
        [total, n, 1, 1],
        &args,
    )
}

/// The routed experts' outputs `y` (`[e, n_used, n]` F32 contiguous; GPU `me`'s columns valid)
/// by the router's choices `ids` (`[n_used, n]` I32 contiguous, the same on every GPU), and the
/// shared expert's output `shared` (`[e, n]`, on GPU `sh` only), to every GPU: `[e, n_used * n
/// + n]`, the shared expert's columns last.
#[allow(clippy::too_many_arguments)]
pub fn select<'a>(
    ctx: &'a Context,
    ex: &Exchange,
    me: usize,
    y: Tensor<'a>,
    ids: Tensor<'a>,
    sh: usize,
    shared: Option<Tensor<'a>>,
    after: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let [e, n_used, n, _] = y.ne();
    assert!(y.ty() == types::F32 && y.is_contiguous(), "select: y");
    assert!(ids.ty() == types::I32 && ids.is_contiguous() && ids.ne()[..2] == [n_used, n]);
    assert_eq!(
        shared.is_some(),
        me == sh,
        "select: the shared expert's GPU has it"
    );
    if let Some(s) = shared {
        assert!(s.ty() == types::F32 && s.is_contiguous() && s.ne()[..2] == [e, n]);
    }
    assert!(
        select_bytes(e, n_used, n) <= ex.slot_bytes,
        "select: slot too small"
    );
    let mut args = vec![y, ids];
    args.extend(shared);
    args.extend(after);
    node(
        ctx,
        ex,
        Kind::Select,
        me,
        Some(sh),
        &[],
        [e, n_used * n + n, 1, 1],
        &args,
    )
}

/// The bytes of a reduce's payload: `[e, n]` F32.
pub fn reduce_bytes(e: i64, n: i64) -> usize {
    (padded(e * n) * 4) as usize
}

/// The sum of every GPU's `part` (F32 contiguous, the same shape on every GPU), added in GPU
/// order, on every GPU: the same bits everywhere. `after`: as for [`gather`].
pub fn reduce<'a>(
    ctx: &'a Context,
    ex: &Exchange,
    me: usize,
    part: Tensor<'a>,
    after: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        part.ty() == types::F32 && part.is_contiguous(),
        "reduce: an F32 contiguous part"
    );
    let ne = part.ne();
    assert!(
        reduce_bytes(ne[0], ne[1] * ne[2] * ne[3]) <= ex.slot_bytes,
        "reduce: slot too small"
    );
    let mut args = vec![part];
    args.extend(after);
    node(ctx, ex, Kind::Reduce, me, None, &[], ne, &args)
}

/// What one node exchanges, read from it.
struct Op<'x> {
    ex: &'x Exchange,
    kind: Kind,
    me: usize,
    peer: Option<usize>,
    totals: Vec<i64>,
    n: i64,
    /// select: values per column, choices.
    e: i64,
    n_used: i64,
}

/// # Safety
/// `dst` is a node made by this module's builders, its exchange alive.
unsafe fn op_of(dst: &ffi::ggml_tensor) -> Op<'static> {
    let p = external_params(dst);
    let ex = (p[P_EX] as u32 as u64) | ((p[P_EX + 1] as u32 as u64) << 32);
    // SAFETY: the address the builder stored; the engine keeps the exchange alive while
    // graphs using it exist.
    let ex = unsafe { &*(ex as usize as *const Exchange) };
    let kind = match p[P_KIND] {
        0 => Kind::Gather,
        1 => Kind::Bcast,
        2 => Kind::Select,
        _ => Kind::Reduce,
    };
    let peer = (p[P_PEER] >= 0).then_some(p[P_PEER] as usize);
    let totals = p[P_TOTAL..P_TOTAL + p[P_SEG] as usize]
        .iter()
        .map(|&t| t as i64)
        .collect();
    let (n, e, n_used) = match kind {
        Kind::Gather => {
            // SAFETY: src[0] is the first part.
            let s = unsafe { &*dst.src[0] };
            (s.ne[1] * s.ne[2] * s.ne[3], 0, 0)
        }
        Kind::Bcast => (dst.ne[1], 0, 0),
        Kind::Select => {
            // SAFETY: src[0] is y.
            let y = unsafe { &*dst.src[0] };
            (y.ne[2], y.ne[0], y.ne[1])
        }
        // All the values as one column.
        Kind::Reduce => (1, dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3], 0),
    };
    Op {
        ex,
        kind,
        me: p[P_ME] as usize,
        peer,
        totals,
        n,
        e,
        n_used,
    }
}

impl Op<'_> {
    /// GPU `d`'s rows `(start, len)` of a gather / bcast segment of `total` rows.
    fn rows(&self, total: i64, d: usize) -> (i64, i64) {
        if self.kind == Kind::Bcast {
            return (0, if Some(d) == self.peer { total } else { 0 });
        }
        let r = super::rows(total, self.ex.n_dev, d);
        (r.start, r.end - r.start)
    }

    /// The columns of a select GPU `d` owns, in order.
    fn owned_cols(&self, ids: &[i32], d: usize) -> Vec<usize> {
        (0..ids.len())
            .filter(|&q| self.ex.owner(ids[q]) == d)
            .collect()
    }

    /// Whether this GPU sends to `b` / receives from others.
    fn sends_to(&self, b: usize) -> bool {
        self.kind != Kind::Gather || self.peer.is_none_or(|r| r == b)
    }

    fn receives(&self) -> bool {
        self.kind != Kind::Gather || self.peer.is_none_or(|r| r == self.me)
    }

    /// The floats of GPU `w`'s payload (not a select): per segment its rows of every token,
    /// each segment padded to 4 floats; a reduce's part.
    fn payload_floats(&self, w: usize) -> i64 {
        if self.kind == Kind::Reduce {
            return self.e;
        }
        self.totals
            .iter()
            .map(|&t| padded(self.rows(t, w).1 * self.n))
            .sum()
    }

    /// The bytes of a chunk, the same on every GPU: [`CHUNK_MIN`] or more, so the biggest
    /// payload goes in [`MAX_CHUNKS`] at most; a multiple of 16.
    fn chunk_bytes(&self) -> usize {
        let most = (0..self.ex.n_dev)
            .map(|w| self.payload_floats(w) as usize * 4)
            .max()
            .unwrap_or(0);
        self.ex.chunk_bytes(most)
    }

    /// The chunks of GPU `w`'s payload (one at least: its flag says it arrived).
    fn n_chunks(&self, w: usize) -> usize {
        let cf = (self.chunk_bytes() / 4) as i64;
        (self.payload_floats(w).max(1) + cf - 1).div_euclid(cf) as usize
    }

    /// Whether this is a reduce that goes as a reduce-scatter and an all-gather
    /// ([`Exchange::scatters`]).
    fn scatter(&self) -> bool {
        self.kind == Kind::Reduce && self.ex.scatters(self.payload_floats(self.me) as usize * 4)
    }

    /// The chunks of a scattered reduce's GPU `d` owns: `d`, `d + n_dev`, ..
    fn owned_chunks(&self, d: usize) -> usize {
        self.n_chunks(d).saturating_sub(d).div_ceil(self.ex.n_dev)
    }

    /// A reduce's sum blocks per chunk: enough for one per GPU at least (one when scattered:
    /// every GPU has a chunk).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn sub(&self) -> usize {
        if self.kind == Kind::Reduce {
            self.ex.n_dev.div_ceil(self.n_chunks(self.me))
        } else {
            1
        }
    }

    /// The readers this GPU sends to, the next GPU first.
    fn readers(&self) -> Vec<usize> {
        let n = self.ex.n_dev;
        (1..n)
            .map(|i| (self.me + i) % n)
            .filter(|&b| self.sends_to(b))
            .collect()
    }

    /// The kernel's blocks: a select one per GPU; else as many as the larger of its send items
    /// (readers times chunks) and its receive items, [`MAX_BLOCKS`] at most.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn blocks(&self) -> usize {
        if self.kind == Kind::Select {
            return self.ex.n_dev;
        }
        if self.scatter() {
            // The others' chunks out and back in, the own ones summed.
            let own = self.owned_chunks(self.me);
            return (self.n_chunks(self.me) - own).max(own).clamp(1, MAX_BLOCKS);
        }
        let send = self.readers().len() * self.n_chunks(self.me);
        let recv = match self.kind {
            _ if !self.receives() => 0,
            Kind::Reduce => self.n_chunks(self.me) * self.sub(),
            _ => (0..self.ex.n_dev).map(|w| self.n_chunks(w)).sum(),
        };
        send.max(recv).clamp(1, MAX_BLOCKS)
    }

    /// This GPU's payload's floats `lo..hi` into `out` (a reader's slot, at the same offsets).
    ///
    /// # Safety
    /// `out` holds the payload; the sources are this GPU's host tensors `src`.
    unsafe fn pack_range(
        &self,
        src: &dyn Fn(usize) -> *const f32,
        out: *mut f32,
        lo: i64,
        hi: i64,
    ) {
        if self.kind == Kind::Reduce {
            // SAFETY: as the caller promises.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src(0).add(lo as usize),
                    out.add(lo as usize),
                    (hi - lo) as usize,
                )
            };
            return;
        }
        let mut at = 0;
        for (s, &total) in self.totals.iter().enumerate() {
            let l = self.rows(total, self.me).1 * self.n;
            let (a, b) = (lo.max(at), hi.min(at + l));
            if b > a {
                // SAFETY: as the caller promises.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src(s).add((a - at) as usize),
                        out.add(a as usize),
                        (b - a) as usize,
                    )
                };
            }
            at += padded(l);
        }
    }

    /// GPU `w`'s gather / bcast payload's floats `lo..hi` into the result `out`: `from` its slot,
    /// or None for this GPU's own sources. Segment by segment, token row by token row.
    ///
    /// # Safety
    /// As for [`Self::pack_range`]; `out` is the node's result.
    unsafe fn unpack_range(
        &self,
        src: &dyn Fn(usize) -> *const f32,
        out: *mut f32,
        w: usize,
        from: Option<*const f32>,
        lo: i64,
        hi: i64,
    ) {
        let (mut at, mut o) = (0i64, 0i64);
        for (s, &total) in self.totals.iter().enumerate() {
            let (start, len) = self.rows(total, w);
            let l = len * self.n;
            let (a, b) = (lo.max(at) - at, hi.min(at + l) - at);
            if b > a {
                let base = from.map_or_else(|| src(s), |f| f.wrapping_add(at as usize));
                let mut t = a / len;
                while t * len < b {
                    let (k0, k1) = (a.max(t * len), b.min((t + 1) * len));
                    let to = o + t * total + start + (k0 - t * len);
                    // SAFETY: as the caller promises.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(k0 as usize),
                            out.add(to as usize),
                            (k1 - k0) as usize,
                        )
                    };
                    t += 1;
                }
            }
            at += padded(l);
            o += total * self.n;
        }
    }
}

/// The CPU op: one thread does it all (copies are small; a parallel copy is not worth the
/// barrier). The kernel's protocol: the same chunks, slots, flags and order of additions.
unsafe fn cpu_exchange(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node of ours; its sources are host tensors.
    unsafe {
        let op = op_of(dst);
        let ex = op.ex;
        let v = ex.next_v(op.me);
        let parity = (v & 1) as usize;
        let src = |i: usize| (*dst.src[i]).data as *const f32;
        let out = dst.data as *mut f32;
        // Waits for the flag `fl` of writer `w`, adding the wait to this GPU's count when `count`.
        let arrive = |fl: &AtomicU64, w: usize, count: bool| {
            let t0 = Instant::now();
            while fl.load(Ordering::Acquire) != v {
                if t0.elapsed() > TIMEOUT {
                    ex.err(op.me).store(1, Ordering::Relaxed);
                    break;
                }
                std::hint::spin_loop();
            }
            if count {
                ex.wait(op.me, w)
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        };
        if op.kind == Kind::Select {
            select_cpu(&op, dst, v, &arrive);
            return;
        }
        if op.scatter() {
            scatter_cpu(&op, dst, v, &arrive);
            return;
        }
        let cf = (op.chunk_bytes() / 4) as i64;
        let mine = op.payload_floats(op.me);
        let span = |c: usize, all: i64| (c as i64 * cf, (c as i64 * cf + cf).min(all));
        for b in op.readers() {
            let (fl, slot) = ex.route(op.me, b, parity, mine as usize * 4);
            for c in 0..op.n_chunks(op.me) {
                let (lo, hi) = span(c, mine);
                if hi > lo {
                    op.pack_range(&src, slot as *mut f32, lo, hi);
                }
                ex.flag_at(fl, parity, op.me, c).store(v, Ordering::Release);
            }
        }
        if !op.receives() {
            return;
        }
        if op.kind == Kind::Reduce {
            let routes: Vec<(usize, usize)> = (0..ex.n_dev)
                .map(|w| ex.route(w, op.me, parity, op.e as usize * 4))
                .collect();
            let parts: Vec<*const f32> = (0..ex.n_dev)
                .map(|w| {
                    if w == op.me {
                        src(0)
                    } else {
                        routes[w].1 as *const f32
                    }
                })
                .collect();
            for c in 0..op.n_chunks(op.me) {
                for w in (0..ex.n_dev).filter(|&w| w != op.me) {
                    arrive(ex.flag_at(routes[w].0, parity, w, c), w, c == 0);
                }
                let (lo, hi) = span(c, op.e);
                if hi > lo {
                    let at: Vec<*const f32> = parts.iter().map(|p| p.add(lo as usize)).collect();
                    reduce_into(out.add(lo as usize), &at, (hi - lo) as usize);
                }
            }
            return;
        }
        for w in 0..ex.n_dev {
            let all = op.payload_floats(w);
            let (fl, slot) = ex.route(w, op.me, parity, all as usize * 4);
            for c in 0..op.n_chunks(w) {
                let from = if w == op.me {
                    None
                } else {
                    arrive(ex.flag_at(fl, parity, w, c), w, c == 0);
                    Some(slot as *const f32)
                };
                let (lo, hi) = span(c, all);
                if hi > lo {
                    op.unpack_range(&src, out, w, from, lo, hi);
                }
            }
        }
    }
}

/// A big reduce on the CPU (#686), as the kernel's `scatter_reduce`: chunk `c` is GPU
/// `c % n_dev`'s. Every GPU sends the others' chunks of its part to them; each adds its own
/// chunks' parts in GPU order (the bits of every GPU adding them all) and sends the sums to the
/// others, into their slots' places of the owner's chunks (which the first sends leave alone);
/// then every GPU copies the others' sums.
///
/// # Safety
/// `dst` is `op`'s node; its sources are host tensors.
unsafe fn scatter_cpu(
    op: &Op<'_>,
    dst: &ffi::ggml_tensor,
    v: u64,
    arrive: &dyn Fn(&AtomicU64, usize, bool),
) {
    let ex = op.ex;
    let n = ex.n_dev;
    let parity = (v & 1) as usize;
    let cf = (op.chunk_bytes() / 4) as i64;
    let bytes = op.e as usize * 4;
    let span = |c: usize| (c as i64 * cf, (c as i64 * cf + cf).min(op.e));
    let chunks = 0..op.n_chunks(op.me);
    // SAFETY: as the caller promises; the slots hold a whole part.
    unsafe {
        let part = (*dst.src[0]).data as *const f32;
        let out = dst.data as *mut f32;
        let copy = |from: *const f32, to: *mut f32, c: usize| {
            let (lo, hi) = span(c);
            if hi > lo {
                std::ptr::copy_nonoverlapping(
                    from.add(lo as usize),
                    to.add(lo as usize),
                    (hi - lo) as usize,
                );
            }
        };
        for c in chunks.clone().filter(|c| c % n != op.me) {
            let (fl, slot) = ex.route(op.me, c % n, parity, bytes);
            copy(part, slot as *mut f32, c);
            ex.flag_at(fl, parity, op.me, c).store(v, Ordering::Release);
        }
        let routes: Vec<(usize, usize)> =
            (0..n).map(|w| ex.route(w, op.me, parity, bytes)).collect();
        let parts: Vec<*const f32> = (0..n)
            .map(|w| {
                if w == op.me {
                    part
                } else {
                    routes[w].1 as *const f32
                }
            })
            .collect();
        for c in chunks.clone().filter(|c| c % n == op.me) {
            for w in (0..n).filter(|&w| w != op.me) {
                arrive(ex.flag_at(routes[w].0, parity, w, c), w, c == op.me);
            }
            let (lo, hi) = span(c);
            if hi > lo {
                let at: Vec<*const f32> = parts.iter().map(|p| p.add(lo as usize)).collect();
                reduce_into(out.add(lo as usize), &at, (hi - lo) as usize);
            }
            for b in (0..n).filter(|&b| b != op.me) {
                let (fl, slot) = ex.route(op.me, b, parity, bytes);
                copy(out as *const f32, slot as *mut f32, c);
                ex.flag_at(fl, parity, op.me, c).store(v, Ordering::Release);
            }
        }
        for c in chunks.filter(|c| c % n != op.me) {
            let o = c % n;
            arrive(ex.flag_at(routes[o].0, parity, o, c), o, c == o);
            copy(routes[o].1 as *const f32, out, c);
        }
    }
}

/// A select on the CPU, as before #671: one chunk, the host inboxes.
///
/// # Safety
/// `dst` is `op`'s node; its sources are host tensors.
unsafe fn select_cpu(
    op: &Op<'_>,
    dst: &ffi::ggml_tensor,
    v: u64,
    arrive: &dyn Fn(&AtomicU64, usize, bool),
) {
    let ex = op.ex;
    let parity = (v & 1) as usize;
    // SAFETY: as the caller promises.
    unsafe {
        let src = |i: usize| (*dst.src[i]).data as *const f32;
        let out = dst.data as *mut f32;
        let ids = std::slice::from_raw_parts(
            (*dst.src[1]).data as *const i32,
            (op.n_used * op.n) as usize,
        );
        let e = op.e as usize;
        // Packs this GPU's payload into `to`.
        let pack = |to: *mut f32| {
            let mut k = 0;
            for q in op.owned_cols(ids, op.me) {
                std::ptr::copy_nonoverlapping(src(0).add(q * e), to.add(k * e), e);
                k += 1;
            }
            if Some(op.me) == op.peer {
                std::ptr::copy_nonoverlapping(src(2), to.add(k * e), e * op.n as usize);
            }
        };
        // GPU w's payload (`from`; None: this GPU's own sources) into the result.
        let unpack = |w: usize, from: Option<*const f32>| {
            let pairs = (op.n_used * op.n) as usize;
            let mut k = 0;
            for q in op.owned_cols(ids, w) {
                let s = from.map_or(src(0).add(q * e), |f| f.add(k * e));
                std::ptr::copy_nonoverlapping(s, out.add(q * e), e);
                k += 1;
            }
            if Some(w) == op.peer {
                let s = from.map_or_else(|| src(2), |f| f.add(k * e));
                std::ptr::copy_nonoverlapping(s, out.add(pairs * e), e * op.n as usize);
            }
        };
        for b in (0..ex.n_dev).filter(|&b| b != op.me) {
            pack(ex.slot(b, parity, op.me));
            ex.flag_at(ex.inbox[b], parity, op.me, 0)
                .store(v, Ordering::Release);
        }
        unpack(op.me, None);
        for b in (0..ex.n_dev).filter(|&b| b != op.me) {
            arrive(ex.flag_at(ex.inbox[op.me], parity, b, 0), b, true);
            unpack(b, Some(ex.slot(op.me, parity, b)));
        }
    }
}

/// `out[i] = parts[0][i] + parts[1][i] + ..`, added in that order (a reduce's sum, as the
/// CUDA kernel adds it).
///
/// # Safety
/// Every part holds `len` values; `out` has room for them.
unsafe fn reduce_into(out: *mut f32, parts: &[*const f32], len: usize) {
    for i in 0..len {
        // SAFETY: as the caller promises.
        unsafe {
            let mut acc = *parts[0].add(i);
            for p in &parts[1..] {
                acc += *p.add(i);
            }
            *out.add(i) = acc;
        }
    }
}

/// `legion_exchange` (`src/cuda/glm_exchange.cu`).
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Kernel {
    inbox: [*mut u8; MAX_DEV],
    peer_in: [*mut u8; MAX_DEV],
    src: [*const u8; MAX_SEG],
    ids: *const i32,
    dst: *mut u8,
    slot_bytes: i64,
    peer_slot: i64,
    chunk: i64,
    v: u64,
    total: [i64; MAX_SEG],
    n: i64,
    e: i64,
    timeout_ns: i64,
    bounds: [i32; MAX_DEV + 1],
    peer_mask: [u32; MAX_DEV],
    n_dev: i32,
    me: i32,
    mode: i32,
    n_seg: i32,
    root: i32,
    sh: i32,
    n_used: i32,
    sub: i32,
    scatter: i32,
}

/// `dim3`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    static legion_exchange_kernel: *const c_void;
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_exchange(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of ours on the stream's device; the kernel reads `k` from the launch's copy.
    unsafe {
        let op = op_of(dst);
        let ex = op.ex;
        let select = op.kind == Kind::Select;
        let mut k = Kernel {
            inbox: [std::ptr::null_mut(); MAX_DEV],
            peer_in: [std::ptr::null_mut(); MAX_DEV],
            src: [std::ptr::null(); MAX_SEG],
            ids: std::ptr::null(),
            dst: dst.data as *mut u8,
            slot_bytes: ex.slot_bytes as i64,
            peer_slot: ex.peer_slot as i64,
            chunk: if select { 0 } else { op.chunk_bytes() as i64 },
            v: ex.next_v(op.me),
            total: [0; MAX_SEG],
            n: op.n,
            e: op.e,
            timeout_ns: TIMEOUT.as_nanos() as i64,
            bounds: ex.bounds,
            peer_mask: ex.peer_mask,
            n_dev: ex.n_dev as i32,
            me: op.me as i32,
            mode: op.kind as i32,
            n_seg: op.totals.len() as i32,
            root: -1,
            sh: -1,
            n_used: op.n_used as i32,
            sub: op.sub() as i32,
            scatter: op.scatter() as i32,
        };
        for d in 0..ex.n_dev {
            k.inbox[d] = ex.inbox[d] as *mut u8;
            k.peer_in[d] = ex.peer_in[d] as *mut u8;
        }
        for (s, &t) in op.totals.iter().enumerate() {
            k.total[s] = t;
        }
        let data = |i: usize| (*dst.src[i]).data as *const u8;
        match op.kind {
            Kind::Gather => {
                k.root = op.peer.map_or(-1, |d| d as i32);
                for s in 0..op.totals.len() {
                    k.src[s] = data(s);
                }
            }
            Kind::Bcast => {
                k.root = op.peer.map_or(-1, |d| d as i32);
                if op.peer == Some(op.me) {
                    k.src[0] = data(0);
                }
            }
            Kind::Select => {
                k.sh = op.peer.map_or(-1, |d| d as i32);
                k.src[0] = data(0);
                k.ids = data(1) as *const i32;
                if op.peer == Some(op.me) {
                    k.src[1] = data(2);
                }
            }
            // `e`: the values of the part.
            Kind::Reduce => k.src[0] = data(0),
        }
        let grid = Dim3 {
            x: op.blocks() as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 { x: 512, y: 1, z: 1 };
        let mut args = [(&raw mut k).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_exchange_kernel,
            grid,
            block,
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_exchange: CUDA error {e}")),
        }
    }
}

static EP_MASK: ExternalOp = ExternalOp {
    name: "glm_ep_mask",
    cpu: cpu_ep_mask,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ep_mask),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Expert parallelism: the tiered ids `mapped` (`[n_used, n]` I32, the cache's remap of the
/// router's choices `sel`) with -1 where the chosen expert is not in `owned`, the GPU's share,
/// so the expert matvec computes this GPU's choices only.
pub fn ep_mask<'a>(
    ctx: &'a Context,
    mapped: Tensor<'a>,
    sel: Tensor<'a>,
    owned: std::ops::Range<u32>,
) -> Tensor<'a> {
    assert!(mapped.ty() == types::I32 && sel.ty() == types::I32);
    assert_eq!(
        mapped.elements(),
        sel.elements(),
        "ep_mask: one id per choice"
    );
    let cont = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
    let p = [owned.start as i32, owned.end as i32];
    ctx.external_with(
        &EP_MASK,
        types::I32,
        mapped.ne(),
        &[cont(mapped), cont(sel)],
        &p,
    )
}

unsafe fn cpu_ep_mask(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    // SAFETY: a node of `ep_mask`: two contiguous I32 sources of dst's size, host tensors.
    unsafe {
        let n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as usize;
        let mapped = std::slice::from_raw_parts((*dst.src[0]).data as *const i32, n);
        let sel = std::slice::from_raw_parts((*dst.src[1]).data as *const i32, n);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut i32, n);
        for i in 0..n {
            out[i] = if (p[0]..p[1]).contains(&sel[i]) {
                mapped[i]
            } else {
                -1
            };
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    static legion_ep_mask_kernel: *const c_void;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ep_mask(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let p = external_params(dst);
    // SAFETY: a node of `ep_mask` on the stream's device; the launch copies the arguments.
    unsafe {
        let mut mapped = (*dst.src[0]).data as *const i32;
        let mut sel = (*dst.src[1]).data as *const i32;
        let mut out = dst.data as *mut i32;
        let mut n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as i32;
        let (mut lo, mut hi) = (p[0], p[1]);
        let mut args = [
            (&raw mut mapped).cast::<c_void>(),
            (&raw mut sel).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
            (&raw mut lo).cast::<c_void>(),
            (&raw mut hi).cast::<c_void>(),
        ];
        let block = 128u32;
        let grid = Dim3 {
            x: (n as u32).div_ceil(block).max(1),
            y: 1,
            z: 1,
        };
        match cudaLaunchKernel(
            legion_ep_mask_kernel,
            grid,
            Dim3 {
                x: block,
                y: 1,
                z: 1,
            },
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_ep_mask: CUDA error {e}")),
        }
    }
}

static EP_SUM: ExternalOp = ExternalOp {
    name: "glm_ep_sum",
    cpu: cpu_ep_sum,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ep_sum),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Expert parallelism's share of a MoE layer's sum: per token, the router's weights `w`
/// (`[n_used, n]` F32) times the experts' outputs `y` (`[e, n_used, n]` F32) over the choices
/// whose expert (`sel`, `[n_used, n]` I32) is in `owned`, in choice order, plus `add`
/// (`[e, n]`, this GPU's share of the shared expert). `[e, n]` F32. The other columns of `y`
/// are never read: the expert matvec leaves them unwritten ([`ep_mask`]).
pub fn ep_sum<'a>(
    ctx: &'a Context,
    y: Tensor<'a>,
    w: Tensor<'a>,
    sel: Tensor<'a>,
    owned: std::ops::Range<u32>,
    add: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let cont = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
    let [e, n_used, n, _] = y.ne();
    assert!(y.ty() == types::F32 && w.ty() == types::F32 && sel.ty() == types::I32);
    assert!(w.elements() as i64 == n_used * n && sel.elements() as i64 == n_used * n);
    let mut args = vec![cont(y), cont(w), cont(sel)];
    if let Some(a) = add {
        assert!(
            a.ty() == types::F32 && a.elements() as i64 == e * n,
            "ep_sum: add"
        );
        args.push(cont(a));
    }
    let p = [owned.start as i32, owned.end as i32, add.is_some() as i32];
    ctx.external_with(&EP_SUM, types::F32, [e, n, 1, 1], &args, &p)
}

unsafe fn cpu_ep_sum(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let p = external_params(dst);
    // SAFETY: a node of `ep_sum`: contiguous F32 / I32 host sources of the shapes it checked.
    unsafe {
        let y = &*dst.src[0];
        let (e, n_used, n) = (y.ne[0] as usize, y.ne[1] as usize, y.ne[2] as usize);
        let yv = y.data as *const f32;
        let w = (*dst.src[1]).data as *const f32;
        let sel = (*dst.src[2]).data as *const i32;
        let add = (p[2] != 0).then(|| (*dst.src[3]).data as *const f32);
        let out = dst.data as *mut f32;
        for t in (ith..n).step_by(nth) {
            for i in 0..e {
                let mut acc = 0.0f32;
                for j in 0..n_used {
                    let s = *sel.add(t * n_used + j);
                    if (p[0]..p[1]).contains(&s) {
                        acc += *w.add(t * n_used + j) * *yv.add((t * n_used + j) * e + i);
                    }
                }
                if let Some(a) = add {
                    acc += *a.add(t * e + i);
                }
                *out.add(t * e + i) = acc;
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    static legion_ep_sum_kernel: *const c_void;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ep_sum(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let p = external_params(dst);
    // SAFETY: a node of `ep_sum` on the stream's device; the launch copies the arguments.
    unsafe {
        let y = &*dst.src[0];
        let (mut e, mut n_used, n) = (y.ne[0], y.ne[1] as i32, y.ne[2]);
        let mut yv = y.data as *const f32;
        let mut w = (*dst.src[1]).data as *const f32;
        let mut sel = (*dst.src[2]).data as *const i32;
        let mut add = if p[2] != 0 {
            (*dst.src[3]).data as *const f32
        } else {
            std::ptr::null()
        };
        let mut out = dst.data as *mut f32;
        let (mut lo, mut hi) = (p[0], p[1]);
        let mut args = [
            (&raw mut yv).cast::<c_void>(),
            (&raw mut w).cast::<c_void>(),
            (&raw mut sel).cast::<c_void>(),
            (&raw mut add).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut e).cast::<c_void>(),
            (&raw mut n_used).cast::<c_void>(),
            (&raw mut lo).cast::<c_void>(),
            (&raw mut hi).cast::<c_void>(),
        ];
        let block = 256u32;
        let grid = Dim3 {
            x: (e as u32).div_ceil(block).max(1),
            y: n as u32,
            z: 1,
        };
        match cudaLaunchKernel(
            legion_ep_sum_kernel,
            grid,
            Dim3 {
                x: block,
                y: 1,
                z: 1,
            },
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_ep_sum: CUDA error {e}")),
        }
    }
}
