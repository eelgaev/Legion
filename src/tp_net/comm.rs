//! The all-reduce's transport: per peer a writer and a reader thread, fed in all-reduce order
//! by [`Comm::all_reduce_host`] (CPU graphs) or by the CUDA op's launch (`op::cuda::launch`),
//! which hands them pinned buffers and flags instead of owned bytes. A peer's partials go over
//! its TCP data connection, or as RDMA writes into the pinned buffers ([`super::rdma`]) when both
//! ends have RDMA to each other (`--tp-transport`, [`super::Transport`]); the TCP connection
//! then carries only the setup.
//!
//! A TCP frame is `[seq u32][len u64]` and `len` bytes of f32 partial. Sequence numbers count the
//! all-reduces from 1 and wrap; every rank runs the same graphs, so the reader knows the frame
//! it must get next and refuses any other (the ranks are out of step).
//!
//! Failure (a peer gone, a frame refused, a timeout) is final for the communicator: it keeps the
//! first error ([`Comm::error`]), wakes everything waiting (the GPU's flag waits included, so a
//! stream never hangs) and refuses further all-reduces.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::rdma::{self, Endpoint, RemoteRegion};
use super::{Overlap, Transport, Wire};

/// A waiting thread spins this long after the last progress before it naps between polls (or
/// blocks, waiting for its next job).
const SPIN_FOR: Duration = Duration::from_millis(5);
const NAP: Duration = Duration::from_micros(50);

/// How long [`Comm::shutdown`] gives the writers to send what is still queued before it closes
/// the connections (at most the communicator's timeout).
const DRAIN_FOR: Duration = Duration::from_secs(5);

/// Whether flag value `v` reached `seq` (sequence numbers wrap; as `CU_STREAM_WAIT_VALUE_GEQ`).
pub(super) fn reached(v: u32, seq: u32) -> bool {
    v.wrapping_sub(seq) < 1 << 31
}

/// The next job on `rx`: spins for [`SPIN_FOR`] before it blocks, so a thread woken for each
/// all-reduce of a step is already running (a sleeping thread's wake-up costs tens of µs, more
/// from a deep idle state). `None` once the queue is closed.
fn next_job<T>(rx: &mpsc::Receiver<T>) -> Option<T> {
    let t0 = Instant::now();
    let mut spins = 0u32;
    loop {
        match rx.try_recv() {
            Ok(j) => return Some(j),
            Err(mpsc::TryRecvError::Disconnected) => return None,
            Err(mpsc::TryRecvError::Empty) => {
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(64) && t0.elapsed() >= SPIN_FOR {
                    return rx.recv().ok();
                }
                std::hint::spin_loop();
            }
        }
    }
}

/// How a peer's partials travel.
enum PeerLink {
    Tcp,
    /// RDMA, with a second rail (`--tp-rails 2`) or not.
    Rdma(Arc<Endpoint>, Option<Arc<Endpoint>>),
}

/// An RDMA peer's view of this rank's pinned buffers: this rank's local key and the peer's
/// region (where this rank writes), made by [`Comm::register_region`]; the same on the second
/// rail (the same memory registered with its device).
#[derive(Clone, Copy)]
struct PeerRegion {
    lkey: u32,
    remote: RemoteRegion,
    rail: Option<(u32, RemoteRegion)>,
}

/// With a second rail: exchanges of at least this many bytes to one peer go in [`RAIL_CHUNK`]
/// pieces over both rails (smaller ones, decode steps, as one write on the first).
const RAIL_MIN: usize = 1 << 20;
/// The piece of a split exchange: each rail takes the next piece when one of its writes
/// completes, so the faster rail carries more.
const RAIL_CHUNK: usize = 1 << 20;
/// Writes in flight per rail.
const RAIL_DEPTH: usize = 4;

/// Whether an exchange of `len` bytes to a peer goes over both rails.
fn split_rails(len: usize, rails: bool) -> bool {
    rails && len >= RAIL_MIN
}

/// An RDMA write's keys: the local key of the source, the peer's address and key.
#[derive(Clone, Copy)]
struct RdmaDst {
    lkey: u32,
    raddr: u64,
    rkey: u32,
}

/// What a writer sends next.
enum SendJob {
    /// These bytes.
    Owned { seq: u32, data: Arc<Vec<u8>> },
    /// `len` bytes at host address `ptr`, once the GPU raised the flag at `ready` to `seq`; an
    /// RDMA peer's write goes to `rdma` (and, split, to `rail` on the second rail).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Pinned {
        seq: u32,
        ptr: usize,
        len: usize,
        ready: usize,
        rdma: Option<RdmaDst>,
        rail: Option<RdmaDst>,
    },
}

/// Where a reader puts the next frame.
enum RecvJob {
    /// A new buffer, handed back on `out`.
    Owned {
        seq: u32,
        len: usize,
        out: mpsc::Sender<Vec<u8>>,
    },
    /// `len` bytes at host address `ptr` (pinned), after the GPU's flag at `consumed` says it
    /// is done with what was there (all-reduce `free_after`, the buffer's last user: `seq - 2`,
    /// or `seq - 4` with `--tp-overlap`); then the flag at `flag` is raised to `seq`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Pinned {
        seq: u32,
        ptr: usize,
        len: usize,
        flag: usize,
        consumed: usize,
        free_after: u32,
    },
}

impl RecvJob {
    fn seq_len(&self) -> (u32, usize) {
        match *self {
            RecvJob::Owned { seq, len, .. } | RecvJob::Pinned { seq, len, .. } => (seq, len),
        }
    }
}

/// Where the all-reduces' time went, summed since the communicator started (see
/// [`Comm::stats`]). A wait is measured on the host: from when this rank's partial was ready
/// (the GPU raised its flag, or the CPU op began) to when the last peer's had arrived. Summed
/// over two ranks, the signed waits of the same all-reduces are twice the one-way transport
/// time: the skew between the ranks cancels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArStats {
    /// All-reduces measured.
    pub count: u64,
    /// The sum of the signed waits (negative: the peers' partials were there first).
    pub wait_ns: i64,
    /// The sum of the positive waits: how long this rank's partials waited for the peers'.
    pub stall_ns: u64,
    /// The positive waits of the first all-reduce of the steps marked with
    /// [`Comm::mark_step`] (a step's first one also waits for the ranks that started the step
    /// later), and how many there were.
    pub first_stall_ns: u64,
    pub firsts: u64,
}

/// One in-flight all-reduce's timestamps (ns since the communicator's start).
#[derive(Clone, Copy, Default)]
struct TimeSlot {
    seq: u32,
    ready: Option<u64>,
    recv: u64,
    left: u32,
}

/// The all-reduces' timing ([`ArStats`]): the writers report when the partial was ready, the
/// readers when a peer's arrived; the last of the `2 (n - 1)` reports adds the wait.
struct Timing {
    epoch: Instant,
    events: u32,
    slots: Mutex<[TimeSlot; 4]>,
    totals: Mutex<ArStats>,
    /// The sequence number of the marked step's first all-reduce (0: none marked).
    first: AtomicU32,
}

impl Timing {
    fn new(n: usize) -> Self {
        Self {
            epoch: Instant::now(),
            events: 2 * (n.max(2) as u32 - 1),
            slots: Mutex::new([TimeSlot::default(); 4]),
            totals: Mutex::new(ArStats::default()),
            first: AtomicU32::new(0),
        }
    }

    fn now(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    /// A report for all-reduce `seq`: its partial ready (`ready`) or a peer's arrived.
    fn report(&self, seq: u32, ready: bool) {
        let t = self.now();
        let done = {
            let mut slots = self.slots.lock().expect("timing lock");
            let s = &mut slots[seq as usize % 4];
            if s.seq != seq || s.left == 0 {
                *s = TimeSlot {
                    seq,
                    ready: None,
                    recv: 0,
                    left: self.events,
                };
            }
            if ready {
                s.ready.get_or_insert(t);
            } else {
                s.recv = s.recv.max(t);
            }
            s.left -= 1;
            (s.left == 0).then(|| s.recv as i64 - s.ready.unwrap_or(s.recv) as i64)
        };
        if let Some(w) = done {
            self.add(seq, w);
        }
    }

    /// Adds all-reduce `seq`'s wait (ns).
    fn add(&self, seq: u32, wait: i64) {
        let mut t = self.totals.lock().expect("timing lock");
        t.count += 1;
        t.wait_ns += wait;
        t.stall_ns += wait.max(0) as u64;
        if seq != 0 && self.first.load(Ordering::Acquire) == seq {
            t.first_stall_ns += wait.max(0) as u64;
            t.firsts += 1;
        }
    }
}

/// What the threads share with the communicator.
struct Shared {
    failed: AtomicBool,
    error: Mutex<Option<String>>,
    /// The last sequence number issued.
    issued: AtomicU32,
    /// Host addresses of the flags the GPU waits on, released on failure.
    gpu_flags: Mutex<Vec<usize>>,
    /// Per rank: the last sequence number its writer finished sending (unused at this rank).
    sent: Vec<AtomicU32>,
    timeout: Duration,
    timing: Timing,
}

impl Shared {
    fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Records the first error and wakes every waiter: GPU flag waits pass (far ahead of any
    /// sequence number issued), host waits see `failed`.
    fn fail(&self, e: String) {
        {
            let mut err = self.error.lock().expect("error lock");
            if err.is_none() {
                tracing::error!(error = %e, "tensor parallel: the all-reduce failed");
                *err = Some(e);
            }
        }
        self.failed.store(true, Ordering::Release);
        let ahead = self.issued.load(Ordering::Acquire).wrapping_add(1 << 30);
        for &f in self.gpu_flags.lock().expect("flags lock").iter() {
            // SAFETY: registered flags are u32s in pinned memory that outlives the
            // communicator's use of them (see `Pinned`).
            unsafe { (*(f as *const AtomicU32)).store(ahead, Ordering::Release) };
        }
    }

    /// Waits until the u32 flag at host address `addr` reaches `seq`; false on failure or
    /// timeout (which it records as `what`).
    fn wait_flag(&self, addr: usize, seq: u32, what: impl Fn() -> String) -> bool {
        // SAFETY: the caller's address of a u32 in pinned memory, as for `fail`.
        let flag = unsafe { &*(addr as *const AtomicU32) };
        let poll = || Ok(reached(flag.load(Ordering::Acquire), seq).then_some(()));
        self.poll_until(poll, what).is_some()
    }

    /// Polls `poll` until it gives a value: spinning, then napping between polls after
    /// [`SPIN_FOR`]. `None` on failure (an error from `poll` fails the communicator), or on
    /// timeout (recorded as `what`).
    fn poll_until<T>(
        &self,
        mut poll: impl FnMut() -> Result<Option<T>, String>,
        what: impl Fn() -> String,
    ) -> Option<T> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        loop {
            match poll() {
                Ok(Some(v)) => return (!self.failed()).then_some(v),
                Ok(None) => {}
                Err(e) => {
                    self.fail(e);
                    return None;
                }
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(256) {
                if self.failed() {
                    return None;
                }
                let waited = t0.elapsed();
                if waited >= self.timeout {
                    self.fail(format!("{} (waited {:.1} s)", what(), waited.as_secs_f64()));
                    return None;
                }
                if waited >= SPIN_FOR {
                    std::thread::sleep(NAP);
                    continue;
                }
            }
            std::hint::spin_loop();
        }
    }
}

/// `x` rounded to BF16, nearest even (CUDA's `__float2bfloat16_rn`): the high half of its bits.
pub(super) fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Values as they travel: little-endian F32 or BF16.
pub(super) fn encode(x: &[f32], wire: Wire) -> Vec<u8> {
    match wire {
        Wire::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Wire::Bf16 => x.iter().flat_map(|&v| bf16_bits(v).to_le_bytes()).collect(),
    }
}

/// Value `i` of the bytes `b` of [`encode`].
pub(super) fn decode(b: &[u8], i: usize, wire: Wire) -> f32 {
    match wire {
        Wire::F32 => f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().expect("4 bytes")),
        Wire::Bf16 => {
            let h = u16::from_le_bytes(b[2 * i..2 * i + 2].try_into().expect("2 bytes"));
            f32::from_bits((h as u32) << 16)
        }
    }
}

/// `dst[i]`: the sum over the ranks of value `i` of their bytes (`parts`, this rank's `mine`
/// where `parts` has none), added in rank order one at a time: the CUDA kernels' order, the
/// same bits on every rank.
fn sum_parts(parts: &[Option<Vec<u8>>], rank: usize, mine: &[u8], wire: Wire, dst: &mut [f32]) {
    let part = |r: usize| {
        if r == rank {
            mine
        } else {
            parts[r].as_deref().unwrap_or(mine)
        }
    };
    for (i, d) in dst.iter_mut().enumerate() {
        let mut acc = decode(part(0), i, wire);
        for r in 1..parts.len() {
            acc += decode(part(r), i, wire);
        }
        *d = acc;
    }
}

/// One all-reduce frame's header.
fn header(seq: u32, len: usize) -> [u8; 12] {
    let mut h = [0u8; 12];
    h[..4].copy_from_slice(&seq.to_le_bytes());
    h[4..].copy_from_slice(&(len as u64).to_le_bytes());
    h
}

fn write_frame(s: &mut TcpStream, seq: u32, data: &[u8]) -> std::io::Result<()> {
    let h = header(seq, data.len());
    crate::dashboard::hub::tp_sent(h.len() + data.len());
    if data.len() <= 64 << 10 {
        // One segment for a small frame (a decode step's partials).
        let mut b = Vec::with_capacity(h.len() + data.len());
        b.extend_from_slice(&h);
        b.extend_from_slice(data);
        s.write_all(&b)
    } else {
        s.write_all(&h)?;
        s.write_all(data)
    }
}

fn writer(mut s: TcpStream, rx: mpsc::Receiver<SendJob>, sh: Arc<Shared>, peer: usize) {
    while let Some(job) = next_job(&rx) {
        if sh.failed() {
            return;
        }
        let (seq, r) = match job {
            SendJob::Owned { seq, data } => {
                sh.timing.report(seq, true);
                (seq, write_frame(&mut s, seq, &data))
            }
            SendJob::Pinned {
                seq,
                ptr,
                len,
                ready,
                ..
            } => {
                let what = || format!("the GPU did not reach all-reduce {seq}");
                if !sh.wait_flag(ready, seq, what) {
                    return;
                }
                sh.timing.report(seq, true);
                // SAFETY: the send buffer of `seq`, which the GPU does not touch again before
                // every peer has received it (see `op::cuda::launch`).
                let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                (seq, write_frame(&mut s, seq, data))
            }
        };
        match r {
            Ok(()) => sh.sent[peer].store(seq, Ordering::Release),
            Err(e) => {
                sh.fail(format!("sending all-reduce {seq} to rank {peer}: {e}"));
                return;
            }
        }
    }
}

fn reader(mut s: TcpStream, rx: mpsc::Receiver<RecvJob>, sh: Arc<Shared>, peer: usize) {
    if let Err(e) = s.set_read_timeout(Some(sh.timeout)) {
        sh.fail(format!("rank {peer}'s connection: {e}"));
        return;
    }
    while let Some(job) = next_job(&rx) {
        if sh.failed() {
            return;
        }
        let (seq, len) = job.seq_len();
        let r = (|| -> Result<(), String> {
            let io = |e: std::io::Error| format!("reading all-reduce {seq} from rank {peer}: {e}");
            let mut h = [0u8; 12];
            s.read_exact(&mut h).map_err(io)?;
            crate::dashboard::hub::tp_received(h.len() + len);
            let got_seq = u32::from_le_bytes(h[..4].try_into().expect("4 bytes"));
            let got_len = u64::from_le_bytes(h[4..].try_into().expect("8 bytes"));
            if got_seq != seq || got_len != len as u64 {
                return Err(format!(
                    "rank {peer} sent all-reduce {got_seq} of {got_len} bytes, this rank waits \
                     for {seq} of {len}: the ranks are out of step"
                ));
            }
            match job {
                RecvJob::Owned { out, .. } => {
                    let mut v = vec![0u8; len];
                    s.read_exact(&mut v).map_err(io)?;
                    sh.timing.report(seq, false);
                    // A receiver gone: its all-reduce gave up (and failed the communicator).
                    let _ = out.send(v);
                }
                RecvJob::Pinned {
                    ptr,
                    flag,
                    consumed,
                    free_after,
                    ..
                } => {
                    let what = || format!("the GPU did not finish all-reduce {free_after}");
                    if !sh.wait_flag(consumed, free_after, what) {
                        return Ok(());
                    }
                    // SAFETY: rank `peer`'s receive buffer for `seq`, which the GPU reads only
                    // after the flag below.
                    let buf = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
                    s.read_exact(buf).map_err(io)?;
                    sh.timing.report(seq, false);
                    // SAFETY: a u32 flag in pinned memory (see `Shared::fail`).
                    unsafe { (*(flag as *const AtomicU32)).store(seq, Ordering::Release) };
                }
            }
            Ok(())
        })();
        if let Err(e) = r {
            sh.fail(e);
            return;
        }
    }
}

/// The RDMA transport carries the pinned buffers of the CUDA path only.
const RDMA_PINNED_ONLY: &str = "the RDMA transport carries a GPU's all-reduces only: \
                                --tp-transport tcp for a CPU run";

/// A writer to an RDMA peer: each partial as one write with immediate (its sequence number)
/// into the peer's receive buffer, waited for until it completed. With a second rail
/// (`rail`), a partial of at least [`RAIL_MIN`] bytes goes in pieces over both
/// ([`send_split`]).
fn rdma_writer(
    ep: Arc<Endpoint>,
    rail: Option<Arc<Endpoint>>,
    rx: mpsc::Receiver<SendJob>,
    sh: Arc<Shared>,
    peer: usize,
) {
    while let Some(job) = next_job(&rx) {
        if sh.failed() {
            return;
        }
        let SendJob::Pinned {
            seq,
            ptr,
            len,
            ready,
            rdma: Some(dst),
            rail: rail_dst,
        } = job
        else {
            sh.fail(RDMA_PINNED_ONLY.into());
            return;
        };
        let what = || format!("the GPU did not reach all-reduce {seq}");
        if !sh.wait_flag(ready, seq, what) {
            return;
        }
        sh.timing.report(seq, true);
        if let (Some(r), Some(rd)) = (&rail, rail_dst)
            && split_rails(len, true)
        {
            if !send_split([&ep, r], [dst, rd], (ptr, len), seq, &sh, peer) {
                return;
            }
            sh.sent[peer].store(seq, Ordering::Release);
            continue;
        }
        if let Err(e) = ep.post_write((ptr, len, dst.lkey), (dst.raddr, dst.rkey), seq) {
            sh.fail(format!("sending all-reduce {seq} to rank {peer}: {e}"));
            return;
        }
        let what = || format!("all-reduce {seq} was not delivered to rank {peer}");
        match sh.poll_until(|| ep.poll_send(), what) {
            Some(done) if done == seq => sh.sent[peer].store(seq, Ordering::Release),
            Some(done) => {
                sh.fail(format!(
                    "rank {peer}: all-reduce {done} completed, this rank sent {seq}"
                ));
                return;
            }
            None => return,
        }
    }
}

/// Sends the `len` bytes at host address `ptr` to `peer` in [`RAIL_CHUNK`] pieces over both
/// rails (`eps`, each with its keys in `dsts`), each a write with immediate `seq` at the same
/// offset of the peer's buffer: up to [`RAIL_DEPTH`] writes in flight per rail, the next piece
/// to the rail whose write completed, so each rail carries what it can. Returns once every
/// piece completed; false on failure (recorded).
fn send_split(
    eps: [&Endpoint; 2],
    dsts: [RdmaDst; 2],
    (ptr, len): (usize, usize),
    seq: u32,
    sh: &Shared,
    peer: usize,
) -> bool {
    let pieces = len.div_ceil(RAIL_CHUNK);
    let (mut next, mut out) = (0usize, [0usize; 2]);
    let what = || format!("all-reduce {seq} was not delivered to rank {peer}");
    sh.poll_until(
        || {
            for q in 0..2 {
                while let Some(done) = eps[q].poll_send()? {
                    if done != seq || out[q] == 0 {
                        return Err(format!(
                            "rank {peer}: all-reduce {done} completed on rail {q}, this rank \
                             sent {seq}"
                        ));
                    }
                    out[q] -= 1;
                }
                while out[q] < RAIL_DEPTH && next < pieces {
                    let off = next * RAIL_CHUNK;
                    let n = RAIL_CHUNK.min(len - off);
                    let d = dsts[q];
                    eps[q]
                        .post_write((ptr + off, n, d.lkey), (d.raddr + off as u64, d.rkey), seq)
                        .map_err(|e| format!("sending all-reduce {seq} to rank {peer}: {e}"))?;
                    out[q] += 1;
                    next += 1;
                }
            }
            Ok((next == pieces && out == [0, 0]).then_some(()))
        },
        what,
    )
    .is_some()
}

/// A reader of an RDMA peer: waits for the write with immediate `seq` (the data is in the
/// receive buffer then) and raises the GPU's flag. With a second rail, for the writes of
/// `seq` on both rails ([`recv_rails`]).
fn rdma_reader(
    ep: Arc<Endpoint>,
    rail: Option<Arc<Endpoint>>,
    rx: mpsc::Receiver<RecvJob>,
    sh: Arc<Shared>,
    peer: usize,
) {
    // With two rails: pieces of later exchanges seen while this one's were still coming.
    let mut early = Vec::new();
    while let Some(job) = next_job(&rx) {
        if sh.failed() {
            return;
        }
        let RecvJob::Pinned { seq, len, flag, .. } = job else {
            sh.fail(RDMA_PINNED_ONLY.into());
            return;
        };
        if let Some(r) = &rail {
            if !recv_rails([&ep, r], (seq, len), &mut early, &sh, peer) {
                return;
            }
        } else {
            let what = || format!("no all-reduce {seq} from rank {peer} in time");
            let Some((got_seq, got_len)) = sh.poll_until(|| ep.poll_recv(), what) else {
                return;
            };
            if got_seq != seq || got_len != len {
                sh.fail(format!(
                    "rank {peer} sent all-reduce {got_seq} of {got_len} bytes, this rank waits \
                     for {seq} of {len}: the ranks are out of step"
                ));
                return;
            }
        }
        sh.timing.report(seq, false);
        // SAFETY: a u32 flag in pinned memory (see `Shared::fail`).
        unsafe { (*(flag as *const AtomicU32)).store(seq, Ordering::Release) };
    }
}

/// Waits until the writes with immediate `seq` on both rails (`eps`) add up to `len` bytes:
/// one write on the first rail, or [`send_split`]'s pieces. The rails complete out of order,
/// so a piece of a later exchange can come first; it waits in `early` for its own. An
/// exchange too small to split polls the first rail only, as with one rail. False on failure
/// (recorded).
fn recv_rails(
    eps: [&Endpoint; 2],
    (seq, len): (u32, usize),
    early: &mut Vec<(u32, usize)>,
    sh: &Shared,
    peer: usize,
) -> bool {
    let mut got = 0usize;
    early.retain(|&(s, b)| {
        let mine = s == seq;
        if mine {
            got += b;
        }
        !mine
    });
    let what = || format!("no all-reduce {seq} from rank {peer} in time");
    let done = got >= len
        || sh
            .poll_until(
                || {
                    let n = if split_rails(len, true) { 2 } else { 1 };
                    for ep in &eps[..n] {
                        if let Some((s, b)) = ep.poll_recv()? {
                            if s == seq {
                                got += b;
                            } else if reached(s, seq) {
                                early.push((s, b));
                            } else {
                                return Err(format!(
                                    "rank {peer} sent all-reduce {s}, this rank waits for {seq}: \
                                     the ranks are out of step"
                                ));
                            }
                        }
                    }
                    Ok((got >= len).then_some(()))
                },
                what,
            )
            .is_some();
    if done && got != len {
        sh.fail(format!(
            "rank {peer} sent {got} bytes of all-reduce {seq}, this rank waits for {len}: the \
             ranks are out of step"
        ));
        return false;
    }
    done
}

/// Picks how the partials travel to `peer` over its data connection `s`: both ends swap their
/// offers (whether they have an RDMA device on the connection's addresses, and its queue pair;
/// none for `Tcp`) and take RDMA when both do. With `rails` 2, an RDMA link also gets a second
/// rail when both ends have one ([`add_rail`]).
fn choose_link(
    s: &TcpStream,
    rank: usize,
    peer: usize,
    (transport, rails): (Transport, usize),
    timeout: Duration,
) -> std::io::Result<PeerLink> {
    let fail = |e: String| super::invalid(format!("rank {peer}: {e}"));
    let ep = match transport {
        Transport::Tcp => None,
        _ => match Endpoint::open(s.local_addr()?.ip(), rank) {
            Ok(ep) => ep,
            Err(e) if transport == Transport::Auto => {
                tracing::warn!(peer, error = %e, "tensor parallel: no RDMA, the all-reduce takes TCP");
                None
            }
            Err(e) => return Err(fail(e)),
        },
    };
    let mine = ep.as_ref().map(Endpoint::offer).unwrap_or_default();
    let mut s = s.try_clone()?;
    let old = s.read_timeout()?;
    s.set_read_timeout(Some(timeout))?;
    let far = rdma::swap_offers(&mut s, &mine).map_err(fail)?;
    let link = match ep {
        Some(ep) if far.rdma => {
            ep.connect(&far).map_err(fail)?;
            let rail = add_rail(&mut s, &ep, rank, peer, rails).map_err(fail)?;
            Ok(PeerLink::Rdma(Arc::new(ep), rail.map(Arc::new)))
        }
        _ if transport == Transport::Rdma => Err(fail(format!(
            "--tp-transport rdma, but {} has no RDMA device on the data connection's address",
            if mine.rdma { "the peer" } else { "this rank" }
        ))),
        _ => Ok(PeerLink::Tcp),
    };
    s.set_read_timeout(old)?;
    link
}

/// The second rail of the RDMA link `ep` to `peer` over its data connection `s`: both ends
/// swap whether they want one (`rails` 2), then, when both do, their offers for a queue pair
/// on their NIC port's second RDMA device ([`Endpoint::open_rail`]; none without one), and
/// take it when both have one and connected it. `None`: the link has one rail.
fn add_rail(
    s: &mut TcpStream,
    ep: &Endpoint,
    rank: usize,
    peer: usize,
    rails: usize,
) -> Result<Option<Endpoint>, String> {
    let swap = |s: &mut TcpStream, yes: bool| -> Result<bool, String> {
        let io = |e: std::io::Error| format!("second rail: {e}");
        s.write_all(&[yes as u8]).map_err(io)?;
        let mut b = [0u8; 1];
        s.read_exact(&mut b).map_err(io)?;
        Ok(yes && b[0] == 1)
    };
    if !swap(s, rails >= 2)? {
        if rails >= 2 {
            tracing::warn!(
                peer,
                "tensor parallel: --tp-rails 2, but the peer runs one rail"
            );
        }
        return Ok(None);
    }
    let rail = ep.open_rail(rank).unwrap_or_else(|e| {
        tracing::warn!(peer, error = %e, "tensor parallel: no second RDMA rail");
        None
    });
    let mine = rail.as_ref().map(Endpoint::offer).unwrap_or_default();
    let far = rdma::swap_offers(s, &mine)?;
    let rail = match rail {
        Some(r) if far.rdma => match r.connect(&far) {
            Ok(()) => Some(r),
            Err(e) => {
                tracing::warn!(peer, error = %e, "tensor parallel: no second RDMA rail");
                None
            }
        },
        _ => None,
    };
    if !swap(s, rail.is_some())? {
        tracing::warn!(
            peer,
            "tensor parallel: --tp-rails 2, but a rank has no second rail"
        );
        return Ok(None);
    }
    Ok(rail)
}

/// An RDMA peer: its endpoint, its second rail's and its data connection.
type RdmaPeer = (Arc<Endpoint>, Option<Arc<Endpoint>>, Mutex<TcpStream>);

/// A peer's writer and reader queues.
type Queues = (mpsc::Sender<SendJob>, mpsc::Sender<RecvJob>);

/// One rank's all-reduce over its data connections (see the module docs).
pub struct Comm {
    rank: usize,
    n: usize,
    shared: Arc<Shared>,
    /// Per rank: its writer's and reader's job queues (`None` at this rank; all `None` once
    /// shut down).
    queues: Mutex<Vec<Option<Queues>>>,
    /// The connections, to shut them down (which ends blocked reads).
    streams: Vec<TcpStream>,
    /// The writer threads, the reader threads.
    writers: Mutex<Vec<JoinHandle<()>>>,
    readers: Mutex<Vec<JoinHandle<()>>>,
    /// Disconnects once every writer thread ended (each holds a sender); `None` once shut down.
    writers_done: Mutex<Option<mpsc::Receiver<()>>>,
    /// Per rank: the RDMA peer's endpoint, its second rail's (`--tp-rails 2`) and the data
    /// connection (which then carries only the regions of [`Comm::register_region`]); `None`
    /// for a TCP peer.
    rdma: Vec<Option<RdmaPeer>>,
    /// Per rank: the RDMA peer's region for this rank's writes (the latest registered).
    regions: Mutex<Vec<Option<PeerRegion>>>,
    /// `--tp-wire`: the smallest exchange (in values of the whole tensor) that travels in BF16
    /// (`usize::MAX`: none), and `--tp-seq-par` ([`Comm::set_collectives`]).
    bf16_min: AtomicUsize,
    seq_par: AtomicUsize,
    /// `--tp-overlap` ([`Comm::set_overlap`]).
    overlap: AtomicU8,
    /// The pinned buffers of the CUDA path, made by its first all-reduce.
    #[cfg(feature = "cuda")]
    pub(super) pinned: Mutex<Option<super::op::cuda::Pinned>>,
    /// The CUDA path's exchanges started ([`super::micro`]) and not finished yet, in order.
    #[cfg(feature = "cuda")]
    pub(super) in_flight: Mutex<std::collections::VecDeque<u32>>,
    /// The CUDA path's reduce-scatter whose sum its reader makes ([`super::seq_par::defer_sum`]),
    /// until that reader ran.
    #[cfg(feature = "cuda")]
    pub(super) deferred: Mutex<Option<super::op::cuda::Deferred>>,
}

impl Comm {
    /// The communicator of rank `rank` over `data` (one connection per other rank, `None` at
    /// `rank`): starts a writer and a reader thread per peer. `timeout`: how long anything
    /// waits for a peer (or this rank's GPU) before the communicator fails.
    pub fn new(
        rank: usize,
        data: Vec<Option<TcpStream>>,
        timeout: Duration,
    ) -> std::io::Result<Arc<Self>> {
        Self::with_transport(rank, data, timeout, Transport::Tcp)
    }

    /// [`Comm::new`] with the partials over `transport`: each data connection starts with both
    /// ends' transport offers. `Auto` takes RDMA to each peer when both ends have it, else TCP;
    /// `Rdma` fails without it; `Tcp` offers none. RDMA carries the CUDA path only.
    pub fn with_transport(
        rank: usize,
        data: Vec<Option<TcpStream>>,
        timeout: Duration,
        transport: Transport,
    ) -> std::io::Result<Arc<Self>> {
        Self::with_rails(rank, data, timeout, transport, 1)
    }

    /// [`Comm::with_transport`] with `rails` RDMA rails per peer (`--tp-rails`, 1 or 2): with 2,
    /// each RDMA link also takes its NIC port's second RDMA device when both ends have one, and
    /// exchanges of at least 1 MiB go in pieces over both (the same bytes: the results do not
    /// change). Every rank must pass the same.
    pub fn with_rails(
        rank: usize,
        data: Vec<Option<TcpStream>>,
        timeout: Duration,
        transport: Transport,
        rails: usize,
    ) -> std::io::Result<Arc<Self>> {
        let n = data.len();
        // The dashboard's network gauge (#1198) shows this exchange's bytes.
        crate::dashboard::hub::tp_network_used();
        let shared = Arc::new(Shared {
            failed: AtomicBool::new(false),
            error: Mutex::new(None),
            issued: AtomicU32::new(0),
            gpu_flags: Mutex::new(Vec::new()),
            sent: (0..n).map(|_| AtomicU32::new(0)).collect(),
            timeout,
            timing: Timing::new(n),
        });
        let mut queues = Vec::with_capacity(n);
        let mut streams = Vec::new();
        let (mut writers, mut readers) = (Vec::new(), Vec::new());
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let mut links = Vec::with_capacity(n);
        for (peer, s) in data.into_iter().enumerate() {
            let Some(s) = s else {
                queues.push(None);
                links.push(None);
                continue;
            };
            if peer == rank {
                return Err(super::invalid("a data connection to this rank itself"));
            }
            let link = choose_link(&s, rank, peer, (transport, rails), timeout)?;
            let (wtx, wrx) = mpsc::channel();
            let (rtx, rrx) = mpsc::channel();
            let sh = shared.clone();
            let w = std::thread::Builder::new().name(format!("legion-tp-tx{peer}"));
            let sh2 = shared.clone();
            let r = std::thread::Builder::new().name(format!("legion-tp-rx{peer}"));
            let done = done_tx.clone();
            match &link {
                PeerLink::Tcp => {
                    let (ws, rs) = (s.try_clone()?, s.try_clone()?);
                    writers.push(w.spawn(move || {
                        let _done = done;
                        writer(ws, wrx, sh, peer)
                    })?);
                    readers.push(r.spawn(move || reader(rs, rrx, sh2, peer))?);
                    links.push(None);
                }
                PeerLink::Rdma(ep, rail) => {
                    let (we, re) = (ep.clone(), ep.clone());
                    let (wr, rr) = (rail.clone(), rail.clone());
                    writers.push(w.spawn(move || {
                        let _done = done;
                        rdma_writer(we, wr, wrx, sh, peer)
                    })?);
                    readers.push(r.spawn(move || rdma_reader(re, rr, rrx, sh2, peer))?);
                    links.push(Some((ep.clone(), rail.clone(), Mutex::new(s.try_clone()?))));
                }
            }
            streams.push(s);
            queues.push(Some((wtx, rtx)));
        }
        if queues.iter().filter(|q| q.is_some()).count() + 1 != n || rank >= n {
            return Err(super::invalid(
                "a data connection to every other rank is needed",
            ));
        }
        Ok(Arc::new(Self {
            rank,
            n,
            shared,
            queues: Mutex::new(queues),
            streams,
            writers: Mutex::new(writers),
            readers: Mutex::new(readers),
            writers_done: Mutex::new(Some(done_rx)),
            rdma: links,
            regions: Mutex::new((0..n).map(|_| None).collect()),
            bf16_min: AtomicUsize::new(usize::MAX),
            seq_par: AtomicUsize::new(0),
            overlap: AtomicU8::new(Overlap::Off as u8),
            #[cfg(feature = "cuda")]
            pinned: Mutex::new(None),
            #[cfg(feature = "cuda")]
            in_flight: Mutex::new(std::collections::VecDeque::new()),
            #[cfg(feature = "cuda")]
            deferred: Mutex::new(None),
        }))
    }

    /// Sets the type the partials travel in (`--tp-wire`) and the smallest step that runs the
    /// per-token work on a share of the tokens (`--tp-seq-par`, 0: none; see
    /// [`super::seq_par`]), before the first exchange. Every rank must set the same.
    pub fn set_collectives(&self, wire: Wire, seq_par: usize) {
        let min = match wire {
            Wire::F32 => usize::MAX,
            Wire::Bf16 => super::BF16_MIN_VALUES,
        };
        self.set_bf16_min(min);
        self.seq_par.store(seq_par, Ordering::Release);
    }

    /// Exchanges of at least `values` values (of the whole tensor) travel in BF16 from now on
    /// (`usize::MAX`: none). [`Comm::set_collectives`] sets [`super::BF16_MIN_VALUES`] for
    /// `--tp-wire bf16`; tests set less.
    pub fn set_bf16_min(&self, values: usize) {
        self.bf16_min.store(values, Ordering::Release);
    }

    /// The type the partials of an exchange of a `full`-value tensor travel in (the same on
    /// every rank).
    pub fn wire(&self, full: usize) -> Wire {
        if full >= self.bf16_min.load(Ordering::Acquire) {
            Wire::Bf16
        } else {
            Wire::F32
        }
    }

    /// `--tp-seq-par` (0: off).
    pub fn seq_par(&self) -> usize {
        self.seq_par.load(Ordering::Acquire)
    }

    /// Sets `--tp-overlap` ([`super::micro`]), before the first exchange. Every rank must set
    /// the same.
    pub fn set_overlap(&self, overlap: Overlap) {
        self.overlap.store(overlap as u8, Ordering::Release);
    }

    /// `--tp-overlap`.
    pub fn overlap(&self) -> Overlap {
        match self.overlap.load(Ordering::Acquire) {
            x if x == Overlap::Mixers as u8 => Overlap::Mixers,
            x if x == Overlap::Ffn as u8 => Overlap::Ffn,
            _ => Overlap::Off,
        }
    }

    /// How the partials travel to each peer, e.g. `rank 1: rdma (rocep1s0f1)`, or with a second
    /// rail `rank 1: rdma (rocep1s0f1 + roceP2p1s0f1)`.
    pub fn transports(&self) -> String {
        (0..self.n)
            .filter(|&p| p != self.rank)
            .map(|p| match &self.rdma[p] {
                Some((ep, None, _)) => format!("rank {p}: rdma ({})", ep.device_name()),
                Some((ep, Some(r), _)) => format!(
                    "rank {p}: rdma ({} + {})",
                    ep.device_name(),
                    r.device_name()
                ),
                None => format!("rank {p}: tcp"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Whether any peer's partials travel by RDMA.
    #[cfg(test)]
    pub(super) fn has_rdma(&self) -> bool {
        self.rdma.iter().any(Option::is_some)
    }

    /// The all-reduces' timing so far (see [`ArStats`]).
    pub fn stats(&self) -> ArStats {
        *self.shared.timing.totals.lock().expect("timing lock")
    }

    /// Marks the start of a step: the next all-reduce issued counts as the step's first.
    pub fn mark_step(&self) {
        let next = self.all_reduces().wrapping_add(1);
        self.shared.timing.first.store(next, Ordering::Release);
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn n_ranks(&self) -> usize {
        self.n
    }

    /// The first failure, if any (the communicator refuses all-reduces since).
    pub fn error(&self) -> Option<String> {
        self.shared.error.lock().expect("error lock").clone()
    }

    /// Fails the communicator with `e` (unless it failed already; see the module docs).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) fn fail(&self, e: String) {
        self.shared.fail(e);
    }

    /// All-reduces issued so far (wraps).
    pub fn all_reduces(&self) -> u32 {
        self.shared.issued.load(Ordering::Acquire)
    }

    /// The next all-reduce's sequence number.
    pub(super) fn next_seq(&self) -> u32 {
        self.shared
            .issued
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1)
    }

    /// Gives the writer of each peer `send(peer)` and its reader `recv(peer)`.
    fn queue(
        &self,
        mut send: impl FnMut(usize) -> SendJob,
        mut recv: impl FnMut(usize) -> RecvJob,
    ) -> Result<(), String> {
        let q = self.queues.lock().expect("queues lock");
        for (peer, pq) in q.iter().enumerate() {
            if peer == self.rank {
                continue;
            }
            let Some((w, r)) = pq else {
                return Err("the communicator is shut down".into());
            };
            if w.send(send(peer)).is_err() || r.send(recv(peer)).is_err() {
                return Err(self
                    .error()
                    .unwrap_or_else(|| "a comm thread stopped".into()));
            }
        }
        Ok(())
    }

    /// The sum over the ranks of every rank's `src`, in rank order, into `dst` (host memory;
    /// the CPU op). On failure `dst` gets `src` and the communicator keeps the error.
    pub fn all_reduce_host(&self, src: &[f32], dst: &mut [f32]) -> Result<(), String> {
        assert_eq!(src.len(), dst.len(), "all-reduce lengths");
        let r: Result<(), String> = (|| {
            let wire = self.wire(src.len());
            let mine = Arc::new(encode(src, wire));
            let parts = self.exchange_host(|_| mine.clone(), |_| mine.len())?;
            sum_parts(&parts, self.rank, &mine, wire, dst);
            Ok(())
        })();
        if let Err(e) = &r {
            self.shared.fail(e.clone());
            dst.copy_from_slice(src);
        }
        r
    }

    /// The reduce-scatter of [`super::seq_par`] (host memory; the CPU op): `src` holds
    /// `src.len() / row` columns of `row` values; `dst` gets the sum over the ranks of this
    /// rank's share of the columns ([`super::seq_par::share`]), every rank's added in rank
    /// order. On failure `dst` gets this rank's own columns and the communicator keeps the error.
    pub fn reduce_scatter_host(
        &self,
        src: &[f32],
        dst: &mut [f32],
        row: usize,
    ) -> Result<(), String> {
        let cols = src.len() / row;
        let piece = |r: usize| {
            let (a, n) = super::seq_par::share(cols, r, self.n);
            &src[a * row..(a + n) * row]
        };
        assert_eq!(piece(self.rank).len(), dst.len(), "reduce-scatter lengths");
        let r: Result<(), String> = (|| {
            let wire = self.wire(src.len());
            let mine = encode(piece(self.rank), wire);
            let parts =
                self.exchange_host(|peer| Arc::new(encode(piece(peer), wire)), |_| mine.len())?;
            sum_parts(&parts, self.rank, &mine, wire, dst);
            Ok(())
        })();
        if let Err(e) = &r {
            self.shared.fail(e.clone());
            dst.copy_from_slice(piece(self.rank));
        }
        r
    }

    /// The all-gather of [`super::seq_par`] (host memory; the CPU op): `dst` holds
    /// `dst.len() / row` columns of `row` values and gets every rank's share of them
    /// ([`super::seq_par::share`]); `src` is this rank's. With `--tp-wire bf16` every share is
    /// rounded, this rank's too. On failure the other ranks' columns are zeros.
    pub fn all_gather_host(&self, src: &[f32], dst: &mut [f32], row: usize) -> Result<(), String> {
        let cols = dst.len() / row;
        let range = |r: usize| {
            let (a, n) = super::seq_par::share(cols, r, self.n);
            a * row..(a + n) * row
        };
        assert_eq!(range(self.rank).len(), src.len(), "all-gather lengths");
        let r: Result<(), String> = (|| {
            let wire = self.wire(dst.len());
            let mine = Arc::new(encode(src, wire));
            let parts =
                self.exchange_host(|_| mine.clone(), |peer| range(peer).len() * wire.bytes())?;
            for (r, part) in parts.iter().enumerate() {
                let b = part.as_deref().unwrap_or(&mine);
                for (i, d) in dst[range(r)].iter_mut().enumerate() {
                    *d = decode(b, i, wire);
                }
            }
            Ok(())
        })();
        if let Err(e) = &r {
            self.shared.fail(e.clone());
            dst.fill(0.0);
            dst[range(self.rank)].copy_from_slice(src);
        }
        r
    }

    /// An exchange of one micro-batch of [`super::micro`] on the host (its CPU op). Rank `r`'s
    /// piece is the columns `pieces(r) = (first, count)` (of `row` values each) of `src` for a
    /// reduce-scatter (`dst` gets the sum of this rank's piece) or of `dst` for an all-gather
    /// (`src` is this rank's piece). The wire type goes by `wire_values`, the values of the
    /// step's whole tensor, as the plain exchange's would.
    pub(super) fn micro_host(
        &self,
        gather: bool,
        src: &[f32],
        dst: &mut [f32],
        row: usize,
        pieces: &dyn Fn(usize) -> (usize, usize),
        wire_values: usize,
    ) -> Result<(), String> {
        let range = |r: usize| {
            let (a, c) = pieces(r);
            a * row..(a + c) * row
        };
        let wire = self.wire(wire_values);
        let r: Result<(), String> = (|| {
            if gather {
                assert_eq!(range(self.rank).len(), src.len(), "all-gather lengths");
                let mine = Arc::new(encode(src, wire));
                let parts =
                    self.exchange_host(|_| mine.clone(), |p| range(p).len() * wire.bytes())?;
                for (r, part) in parts.iter().enumerate() {
                    let b = part.as_deref().unwrap_or(&mine);
                    for (i, d) in dst[range(r)].iter_mut().enumerate() {
                        *d = decode(b, i, wire);
                    }
                }
            } else {
                assert_eq!(range(self.rank).len(), dst.len(), "reduce-scatter lengths");
                let mine = encode(&src[range(self.rank)], wire);
                let parts =
                    self.exchange_host(|p| Arc::new(encode(&src[range(p)], wire)), |_| mine.len())?;
                sum_parts(&parts, self.rank, &mine, wire, dst);
            }
            Ok(())
        })();
        if let Err(e) = &r {
            self.shared.fail(e.clone());
            if gather {
                dst[range(self.rank)].copy_from_slice(src);
            } else {
                dst.copy_from_slice(&src[range(self.rank)]);
            }
        }
        r
    }

    /// One exchange of host bytes: `send(peer)` goes to each peer, `recv_len(peer)` bytes come
    /// from each; the peers' bytes by rank (`None` at this rank).
    fn exchange_host(
        &self,
        send: impl Fn(usize) -> Arc<Vec<u8>>,
        recv_len: impl Fn(usize) -> usize,
    ) -> Result<Vec<Option<Vec<u8>>>, String> {
        if let Some(e) = self.error() {
            return Err(e);
        }
        let seq = self.next_seq();
        let mut rxs: Vec<Option<mpsc::Receiver<Vec<u8>>>> = (0..self.n).map(|_| None).collect();
        self.queue(
            |peer| SendJob::Owned {
                seq,
                data: send(peer),
            },
            |peer| {
                let (tx, rx) = mpsc::channel();
                rxs[peer] = Some(rx);
                RecvJob::Owned {
                    seq,
                    len: recv_len(peer),
                    out: tx,
                }
            },
        )?;
        let t0 = Instant::now();
        let mut parts: Vec<Option<Vec<u8>>> = (0..self.n).map(|_| None).collect();
        for (peer, rx) in rxs.iter().enumerate() {
            let Some(rx) = rx else { continue };
            loop {
                match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(v) => {
                        parts[peer] = Some(v);
                        break;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Some(e) = self.error() {
                            return Err(e);
                        }
                        if t0.elapsed() >= self.shared.timeout {
                            return Err(format!("no all-reduce {seq} from rank {peer} in time"));
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(self
                            .error()
                            .unwrap_or_else(|| format!("rank {peer}'s reader stopped")));
                    }
                }
            }
        }
        Ok(parts)
    }

    /// Stops the threads. The writers first send what is still queued, for up to
    /// [`DRAIN_FOR`] (unless the communicator failed): a rank's all-reduce ends once the peers'
    /// partials arrived, maybe before its own went out, and a peer still reading it would see
    /// the connection cut. Then blocked waits end (on failure), and blocked reads and writes
    /// with the connections.
    pub fn shutdown(&self) {
        // Closed queues: each writer ends after the jobs it has.
        self.queues
            .lock()
            .expect("queues lock")
            .iter_mut()
            .for_each(|q| *q = None);
        let done = self.writers_done.lock().expect("writers lock").take();
        if let Some(done) = done
            && !self.shared.failed()
        {
            // Nothing is sent on it: it disconnects once every writer ended.
            let _ = done.recv_timeout(DRAIN_FOR.min(self.shared.timeout));
        }
        // Blocked waits end on failure.
        if !self.shared.failed() {
            self.shared.failed.store(true, Ordering::Release);
        }
        for s in &self.streams {
            let _ = s.shutdown(std::net::Shutdown::Both);
        }
        let writers = std::mem::take(&mut *self.writers.lock().expect("threads lock"));
        let readers = std::mem::take(&mut *self.readers.lock().expect("threads lock"));
        for t in writers.into_iter().chain(readers) {
            let _ = t.join();
        }
    }

    /// Records the host addresses of flags the GPU waits on (released on failure).
    #[cfg(feature = "cuda")]
    pub(super) fn add_gpu_flags(&self, flags: &[usize]) {
        self.shared
            .gpu_flags
            .lock()
            .expect("flags lock")
            .extend_from_slice(flags);
    }

    /// Waits until every writer finished sending all-reduce `seq` (or the communicator failed).
    #[cfg(feature = "cuda")]
    pub(super) fn wait_sent(&self, seq: u32) -> Result<(), String> {
        for peer in (0..self.n).filter(|&p| p != self.rank) {
            let addr = &self.shared.sent[peer] as *const AtomicU32 as usize;
            let what = || format!("all-reduce {seq} was not sent to rank {peer}");
            if !self.shared.wait_flag(addr, seq, what) {
                return Err(self.error().unwrap_or_else(|| "shut down".into()));
            }
        }
        Ok(())
    }

    /// Registers this rank's pinned buffers (`len` bytes at host address `host`) with every RDMA
    /// peer's endpoint and swaps the regions with them over the data connections, as all-reduce
    /// `seq` (every rank registers at the same all-reduce: the buffers grow with the same
    /// graphs). Nothing without RDMA peers. The regions returned must be dropped before the
    /// memory is freed; the peers write into it until the next registration.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) fn register_region(
        &self,
        host: usize,
        len: usize,
        seq: u32,
    ) -> Result<Vec<rdma::Mr>, String> {
        let mut mrs = Vec::new();
        let mut keys: Vec<Option<(u32, Option<u32>)>> = vec![None; self.n];
        for (peer, link) in self.rdma.iter().enumerate() {
            let Some((ep, rail, s)) = link else { continue };
            let mut s = s.lock().expect("stream lock");
            let mut lkeys = Vec::new();
            for e in std::iter::once(ep).chain(rail) {
                let mr = e.register(host, len)?;
                let mine = RemoteRegion {
                    addr: host as u64,
                    rkey: mr.rkey(),
                    len: len as u64,
                };
                lkeys.push(mr.lkey());
                mrs.push(mr);
                s.write_all(&mine.encode(seq))
                    .map_err(|e| format!("sending a memory region to rank {peer}: {e}"))?;
            }
            keys[peer] = Some((lkeys[0], lkeys.get(1).copied()));
        }
        let mut regions = self.regions.lock().expect("regions lock");
        for (peer, link) in self.rdma.iter().enumerate() {
            let (Some((_, _, s)), Some((lkey, rail_lkey))) = (link, keys[peer]) else {
                continue;
            };
            let mut s = s.lock().expect("stream lock");
            let mut read = || {
                let mut b = [0u8; 28];
                s.set_read_timeout(Some(self.shared.timeout))
                    .and_then(|_| s.read_exact(&mut b))
                    .map_err(|e| format!("rank {peer}'s memory region: {e}"))?;
                let remote =
                    RemoteRegion::decode(&b, seq).map_err(|e| format!("rank {peer}: {e}"))?;
                if remote.len != len as u64 {
                    return Err(format!(
                        "rank {peer}'s buffers are {} bytes, this rank's {len}: the ranks differ",
                        remote.len
                    ));
                }
                Ok(remote)
            };
            let remote = read()?;
            let rail = match rail_lkey {
                Some(l) => Some((l, read()?)),
                None => None,
            };
            regions[peer] = Some(PeerRegion { lkey, remote, rail });
        }
        Ok(mrs)
    }

    /// Queues the CUDA path's jobs for exchange `seq` (`seqs.0`): each peer's writer sends
    /// `send(peer) = (host address, bytes)` once the flag `ready` reaches `seq`; each peer's
    /// reader fills the buffer of `recv(peer) = (buffer, flag, bytes)` once `consumed` reaches
    /// `seqs.1` (the exchange that used the buffer last) and raises that flag (an all-reduce
    /// sends and receives the same bytes; a reduce-scatter or all-gather of [`super::seq_par`]
    /// a different share per peer).
    /// An RDMA peer's write goes to offset `remote_off` of the peer's region
    /// ([`Comm::register_region`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) fn queue_pinned(
        &self,
        seqs: (u32, u32),
        send: impl Fn(usize) -> (usize, usize),
        ready: usize,
        recv: impl Fn(usize) -> (usize, usize, usize),
        consumed: usize,
        remote_off: usize,
    ) -> Result<(), String> {
        let (seq, free_after) = seqs;
        let regions = self.regions.lock().expect("regions lock");
        for (peer, r) in regions.iter().enumerate() {
            let Some(r) = r else { continue };
            let len = send(peer).1;
            if (remote_off + len) as u64 > r.remote.len {
                return Err(format!(
                    "exchange {seq}: {len} bytes at offset {remote_off} overrun rank {peer}'s \
                     region of {}",
                    r.remote.len
                ));
            }
        }
        self.queue(
            |peer| {
                let (ptr, len) = send(peer);
                SendJob::Pinned {
                    seq,
                    ptr,
                    len,
                    ready,
                    rdma: regions[peer].map(|r| RdmaDst {
                        lkey: r.lkey,
                        raddr: r.remote.addr + remote_off as u64,
                        rkey: r.remote.rkey,
                    }),
                    rail: regions[peer]
                        .and_then(|r| r.rail)
                        .map(|(lkey, remote)| RdmaDst {
                            lkey,
                            raddr: remote.addr + remote_off as u64,
                            rkey: remote.rkey,
                        }),
                }
            },
            |peer| {
                let (ptr, flag, len) = recv(peer);
                RecvJob::Pinned {
                    seq,
                    ptr,
                    len,
                    flag,
                    consumed,
                    free_after,
                }
            },
        )
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        self.shutdown();
    }
}
