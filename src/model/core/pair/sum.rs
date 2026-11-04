//! The all-reduce of tensor parallelism over the two GPUs of an NVLink pair in one process
//! (`--tp-per-gpu`, #691): an external op ([`all_reduce`]) each GPU's graph runs where a
//! layer's partial sums meet, over NVLink peer writes with the kernel of [`super::link`].
//! Model-neutral: any F32 tensor, the same on both GPUs.
//!
//! **Memory.** Every GPU has an inbox in its own VRAM: a header (a flag per slot, its sends'
//! block counter) and two slots of [`PairSum::slot_bytes`] each. A message goes into slot `v %
//! 2` of the receiver's inbox, `v` its value: the count of messages its sender has sent, one per
//! piece of a sum (a sum bigger than a slot goes in pieces). Both GPUs make the same sums in the
//! same order, so both count the same `v` for a piece, and each sends and receives one message
//! per piece.
//!
//! **Protocol.** Per piece, each GPU writes its part into the other's slot, then raises the
//! other's flag of that slot to `v`, then waits for its own flag of the slot to read `v` and
//! adds the other's part to its own. Two slots are enough: a GPU writes slot `v % 2` again with
//! message `v + 2` only after it took message `v + 1`, which the other sent after it took `v`.
//! An F32 sum of two terms is the same bits in either order, so both GPUs hold the same sum.
//!
//! **Host order.** Each GPU's graph launches on its own thread. Before either thread enqueues
//! a sum's kernels, both have queued everything before it (`arrived`), and neither goes past
//! it before the other has queued its kernels too (`queued`). A GPU therefore only ever waits
//! in this kernel for work already queued on the other GPU, never for a host thread that might
//! itself wait for the GPU (a CUDA graph's instantiation, a module load or a copy waits behind
//! a spinning kernel). CUDA graphs stay on: the op launches between captured segments.
//!
//! A wait that times out (a GPU that never sent) sets the error word: the step fails, and every
//! later sum fails at once (the counts no longer pair up).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::ggml::{
    BufferType, Context, EXTERNAL_PARAMS, ExternalOp, GgmlError, Tensor, external_params, ffi,
    types,
};

static SUM: ExternalOp = ExternalOp {
    name: "pair_sum",
    cpu: cpu_sum,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_sum),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The inbox header's size.
const HDR: usize = 512;
/// Where each slot's flag (u64) is in the header.
const FLAG_AT: [usize; 2] = [0, 128];
/// Where the block counter (u32) of this GPU's sends is in the header.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const DONE_AT: usize = 256;
/// How long a GPU waits for the other's part: the other has queued it, so it only runs what
/// it queued before (a sublayer at most).
const TIMEOUT: Duration = Duration::from_secs(10);
/// How long a thread waits for the other GPU's thread to reach the same sum: it may be building
/// a graph or allocating its compute buffer first.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const HOST_TIMEOUT: Duration = Duration::from_secs(120);
/// The default slot: a 512-token prompt chunk of a 5376-wide model (11 MB) in one piece.
pub const SLOT_BYTES: usize = 16 << 20;

/// One GPU's inbox.
struct Inbox {
    _ctx: Context,
    hdr: usize,
    slots: usize,
}

/// The inboxes and counts of the two GPUs (ranks 0 and 1).
pub struct PairSum {
    slot: usize,
    inbox: [Inbox; 2],
    /// Holds the error word (u32 at its start).
    _status: Context,
    err: usize,
    /// Per rank: the messages it has sent (pieces).
    sent: [AtomicU64; 2],
    /// Per rank: the sums its thread has reached, and has queued (GPU sums).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    arrived: [AtomicU64; 2],
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    queued: [AtomicU64; 2],
}

// SAFETY: the contexts are only read after construction (their tensors' addresses); the
// inboxes are memory the GPUs (or threads) synchronize on through the flags; the rest is atomics.
unsafe impl Send for PairSum {}
unsafe impl Sync for PairSum {}

fn data(t: Tensor<'_>) -> usize {
    // SAFETY: an allocated tensor of ours.
    unsafe { (*t.raw()).data as usize }
}

impl PairSum {
    /// The inboxes of ranks 0 and 1 in `bufts` (each GPU's own memory; host memory for graphs
    /// on the CPU), slots of `slot_bytes` (a multiple of 16). `status`: memory the host and
    /// both GPUs reach (the error word).
    pub fn new(
        bufts: [BufferType; 2],
        status: BufferType,
        slot_bytes: usize,
    ) -> Result<Self, GgmlError> {
        assert!(slot_bytes >= 16 && slot_bytes.is_multiple_of(16));
        let inbox = bufts.map(|buft| {
            let ctx = Context::new(2, 0);
            let hdr = ctx.new_tensor(types::I8, &[HDR as i64]).named("hdr");
            let slots = ctx
                .new_tensor(types::I8, &[2 * slot_bytes as i64])
                .named("slots");
            ctx.alloc_tensors(buft)?;
            ctx.clear_buffers(0);
            Ok::<_, GgmlError>(Inbox {
                hdr: data(hdr),
                slots: data(slots),
                _ctx: ctx,
            })
        });
        let [a, b] = inbox;
        let inbox = [a?, b?];
        let st = Context::new(1, 0);
        let word = st.new_tensor(types::I32, &[64]);
        st.alloc_tensors(status)?;
        st.clear_buffers(0);
        Ok(Self {
            slot: slot_bytes,
            inbox,
            err: data(word),
            _status: st,
            sent: Default::default(),
            arrived: Default::default(),
            queued: Default::default(),
        })
    }

    /// The bytes of one slot: a bigger sum goes in pieces of this.
    pub fn slot_bytes(&self) -> usize {
        self.slot
    }

    /// Whether a wait timed out: every sum since has failed.
    pub fn failed(&self) -> bool {
        self.err_word().load(Ordering::Relaxed) != 0
    }

    /// The failure, if any, as an error message.
    pub fn error(&self) -> Option<String> {
        self.failed().then(|| {
            "the GPUs' all-reduce timed out (one GPU stopped sending its partial sums)".into()
        })
    }

    fn err_word(&self) -> &AtomicU32 {
        // SAFETY: the start of the status tensor (`self._status`), which lives as long as self.
        unsafe { &*(self.err as *const AtomicU32) }
    }

    fn flag(&self, rank: usize, v: u64) -> &AtomicU64 {
        // SAFETY: inside the header, 8-byte aligned; host memory whenever the CPU op runs.
        unsafe { &*((self.inbox[rank].hdr + FLAG_AT[(v & 1) as usize]) as *const AtomicU64) }
    }

    /// Rank `rank`'s slot for message `v`.
    fn slot_of(&self, rank: usize, v: u64) -> usize {
        self.inbox[rank].slots + (v & 1) as usize * self.slot
    }

    /// The next message's value of `rank`.
    fn next(&self, rank: usize) -> u64 {
        self.sent[rank].fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The pieces of a sum of `bytes`: `(offset, bytes)`.
    fn pieces(&self, bytes: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..bytes)
            .step_by(self.slot)
            .map(move |off| (off, self.slot.min(bytes - off)))
    }

    /// Rank `me`'s thread is at its `k`-th sum in `at` (`arrived` or `queued`): waits until the
    /// other's thread is too.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn meet(&self, at: &[AtomicU64; 2], me: usize, k: u64) -> Result<(), String> {
        at[me].store(k, Ordering::Release);
        let other = &at[1 - me];
        let t0 = Instant::now();
        let mut spins = 0u32;
        while other.load(Ordering::Acquire) < k {
            if self.failed() {
                return Err(self.error().unwrap_or_default());
            }
            if t0.elapsed() > HOST_TIMEOUT {
                self.err_word().store(1, Ordering::Relaxed);
                return Err(format!(
                    "the other GPU's thread did not reach all-reduce {k} in {HOST_TIMEOUT:?}"
                ));
            }
            spins += 1;
            if spins < 4096 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        Ok(())
    }
}

const P_ME: usize = 0;
const P_SUM: usize = 1;
const _: () = assert!(P_SUM + 2 <= EXTERNAL_PARAMS);

/// Rank `me`'s `x` (F32) plus the other rank's at the same point of its graph: the same bits on
/// both. `sum` must outlive every graph built with it.
pub fn all_reduce<'a>(ctx: &'a Context, sum: &PairSum, me: usize, x: Tensor<'a>) -> Tensor<'a> {
    assert!(me < 2 && x.ty() == types::F32, "pair_sum: rank {me}, F32");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let p = sum as *const PairSum as u64;
    let mut params = [0i32; P_SUM + 2];
    params[P_ME] = me as i32;
    params[P_SUM] = p as u32 as i32;
    params[P_SUM + 1] = (p >> 32) as u32 as i32;
    ctx.external_with(&SUM, types::F32, x.ne(), &[x], &params)
}

/// # Safety
/// `dst` is a node made by [`all_reduce`], its sum alive.
unsafe fn node_of(dst: &ffi::ggml_tensor) -> (&'static PairSum, usize) {
    let p = external_params(dst);
    let addr = (p[P_SUM] as u32 as u64) | ((p[P_SUM + 1] as u32 as u64) << 32);
    // SAFETY: the address the builder stored; the engine keeps the sum alive while graphs using
    // it exist.
    (
        unsafe { &*(addr as usize as *const PairSum) },
        p[P_ME] as usize,
    )
}

/// The CPU op (tests, and a CPU twin of the protocol): one thread does it all.
unsafe fn cpu_sum(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node of ours; its source, its data and the inboxes are host memory.
    unsafe {
        let (s, me) = node_of(dst);
        let other = 1 - me;
        let x = (*dst.src[0]).data as *const u8;
        let out = dst.data as *mut u8;
        for (off, bytes) in s.pieces(ffi::ggml_nbytes(dst)) {
            let v = s.next(me);
            std::ptr::copy_nonoverlapping(x.add(off), s.slot_of(other, v) as *mut u8, bytes);
            s.flag(other, v).store(v, Ordering::Release);
            let t0 = Instant::now();
            while s.flag(me, v).load(Ordering::Acquire) != v {
                if s.failed() || t0.elapsed() > TIMEOUT {
                    s.err_word().store(1, Ordering::Relaxed);
                    break;
                }
                std::hint::spin_loop();
            }
            let own = x.add(off) as *const f32;
            let part = s.slot_of(me, v) as *const f32;
            let o = out.add(off) as *mut f32;
            for i in 0..bytes / 4 {
                *o.add(i) = *own.add(i) + *part.add(i);
            }
        }
    }
}

#[cfg(feature = "cuda")]
const MAX_BLOCKS: u32 = 80;

#[cfg(feature = "cuda")]
unsafe fn cuda_sum(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    use super::link::{Kernel, THREADS};
    let blocks = |values: usize, per_thread: usize| -> u32 {
        let per_block = THREADS as usize * per_thread;
        (values.div_ceil(per_block) as u32).clamp(1, MAX_BLOCKS)
    };
    // SAFETY: a node of ours on the stream's device; its source and data are device memory,
    // the inboxes VRAM of this GPU and of its peer (peer access on).
    unsafe {
        let (s, me) = node_of(dst);
        if let Some(e) = s.error() {
            return Err(e);
        }
        let other = 1 - me;
        let k = s.arrived[me].load(Ordering::Relaxed) + 1;
        s.meet(&s.arrived, me, k)?;
        let x = (*dst.src[0]).data as *const u8;
        let out = dst.data as *mut u8;
        let err = s.err as *mut u32;
        let mut queued = Ok(());
        for (off, bytes) in s.pieces(ffi::ggml_nbytes(dst)) {
            let v = s.next(me);
            let mut send = Kernel::empty(v, err, TIMEOUT);
            send.seg_dst[0] = s.slot_of(other, v) as *mut u8;
            send.seg_src[0] = x.add(off);
            send.seg_bytes[0] = bytes as i64;
            send.flag = s.flag(other, v).as_ptr();
            send.done = (s.inbox[me].hdr + DONE_AT) as *mut u32;
            // 16 bytes a thread, a few loads each.
            queued = send.launch(blocks(bytes, 64), stream);
            if queued.is_err() {
                break;
            }
            let mut add = Kernel::empty(v, err, TIMEOUT);
            add.mode = 2;
            add.flag = s.flag(me, v).as_ptr();
            add.from = s.slot_of(me, v) as *const u8;
            add.own = x.add(off) as *const f32;
            add.dst = out.add(off);
            add.n = (bytes / 4) as i64;
            queued = add.launch(blocks(bytes / 4, 4), stream);
            if queued.is_err() {
                break;
            }
        }
        // Even after a failed launch: the other thread must not wait here for two minutes.
        let met = s.meet(&s.queued, me, k);
        queued.map_err(|e| format!("pair_sum: {e}"))?;
        met
    }
}

#[cfg(test)]
mod tests;
