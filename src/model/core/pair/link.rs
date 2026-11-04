//! The exchange between the two GPUs of an NVLink pair (#668): external ops
//! ([`crate::ggml::ExternalOp`], the CUDA kernel in `src/cuda/pair_link.cu`) through which one
//! GPU hands the other a task (a MoE layer's input rows, the router's choices and weights) and
//! gets back its part of the result (a partial sum), over NVLink peer writes. Model-neutral: the
//! payload is a residual-width F32 block plus two `[n_used, n]` blocks.
//!
//! **Memory.** Every GPU has an inbox in its own VRAM: a header (a flag per kind of message, and
//! per kind the block counter of its own sends), then the payload tensors `x` F32
//! `[n_embd, max]`, `ids` I32 `[n_used, max]`, `w` F32 `[n_used, max]` (a task) and `part` F32
//! `[n_embd, max]` (a part). A sender writes straight into the receiver's inbox (peer access),
//! fences, and raises the receiver's flag; the receiver polls its own memory. An error word in
//! host memory every GPU reaches says a wait timed out (the step then fails instead of hanging).
//!
//! **Protocol.** Each message gets a value `v`: per receiving GPU and kind, the senders count
//! what they posted and the receiver what it took ([`PairLink::begin`] restarts both at
//! `round << 20`, above every value of an earlier round), so both sides name the same message
//! the same `v` as long as they make their exchanges in the same order. One inbox slot per kind
//! is enough: a task's sender posts the next task only after it took the part the receiver sends
//! once done with the task's payload, and a part's sender posts its next part only after the
//! next task, which its receiver posts after taking the part before.
//!
//! **Sums.** [`recv_add`] adds the other GPU's part to this GPU's: one F32 addition per value,
//! whose result does not depend on which GPU holds which term.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::ggml::{
    BufferType, Context, EXTERNAL_PARAMS, ExternalOp, GgmlError, Tensor, external_params, ffi,
    types,
};

static LINK: ExternalOp = ExternalOp {
    name: "pair_link",
    cpu: cpu_link,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_link),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The inbox header's size.
const HDR: usize = 512;
/// Where each kind's flag (u64) is in the header.
const FLAG_AT: [usize; 2] = [0, 128];
/// Where each kind's block counter of this GPU's sends (u32) is in the header.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const DONE_AT: [usize; 2] = [256, 384];
/// How long a GPU waits for its pair before it gives up: no healthy GPU is a second late.
const TIMEOUT: Duration = Duration::from_secs(10);
/// Bits of a message value below the round ([`PairLink::begin`]).
const ROUND_SHIFT: u32 = 20;

/// A kind of message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The input rows, the router's choices and weights, to the GPU that helps.
    Task = 0,
    /// A partial sum, back to the GPU that asked.
    Part = 1,
}

/// One GPU's inbox: its tensors (the header first) in one context.
struct Inbox {
    ctx: Context,
    hdr: usize,
    x: usize,
    ids: usize,
    w: usize,
    part: usize,
}

/// The inboxes and counters of a set of GPUs exchanging in pairs.
pub struct PairLink {
    n_embd: i64,
    n_used: i64,
    max_tokens: i64,
    inbox: Vec<Inbox>,
    /// Holds the error word (u32 at its start).
    _status: Context,
    err: usize,
    /// Per receiving GPU and kind: messages posted to it, messages it took.
    posted: Vec<[AtomicU64; 2]>,
    taken: Vec<[AtomicU64; 2]>,
}

// SAFETY: the contexts are only read after construction (their tensors' addresses); the
// inboxes are memory the GPUs (or threads) synchronize on through the flags; the rest is atomics.
unsafe impl Send for PairLink {}
unsafe impl Sync for PairLink {}

fn data(t: Tensor<'_>) -> usize {
    // SAFETY: an allocated tensor of ours.
    unsafe { (*t.raw()).data as usize }
}

impl PairLink {
    /// A link between GPUs whose inboxes go in `bufts` (one per GPU, its own memory; host memory
    /// for graphs on the CPU), for steps of up to `max_tokens` tokens of width `n_embd` with
    /// `n_used` experts a token. `status`: memory the host and every GPU reach (the error word).
    pub fn new(
        bufts: &[BufferType],
        status: BufferType,
        n_embd: i64,
        n_used: i64,
        max_tokens: i64,
    ) -> Result<Self, GgmlError> {
        assert!(n_embd % 4 == 0 && n_used > 0 && max_tokens > 0);
        let inbox = bufts
            .iter()
            .map(|&buft| {
                let ctx = Context::new(5, 0);
                let hdr = ctx.new_tensor(types::I8, &[HDR as i64]).named("hdr");
                let x = ctx.new_tensor(types::F32, &[n_embd, max_tokens]).named("x");
                let ids = ctx
                    .new_tensor(types::I32, &[n_used, max_tokens])
                    .named("ids");
                let w = ctx.new_tensor(types::F32, &[n_used, max_tokens]).named("w");
                let part = ctx
                    .new_tensor(types::F32, &[n_embd, max_tokens])
                    .named("part");
                ctx.alloc_tensors(buft)?;
                ctx.clear_buffers(0);
                Ok(Inbox {
                    hdr: data(hdr),
                    x: data(x),
                    ids: data(ids),
                    w: data(w),
                    part: data(part),
                    ctx,
                })
            })
            .collect::<Result<Vec<_>, GgmlError>>()?;
        let st = Context::new(1, 0);
        let word = st.new_tensor(types::I32, &[64]);
        st.alloc_tensors(status)?;
        st.clear_buffers(0);
        let err = data(word);
        let n = bufts.len();
        Ok(Self {
            n_embd,
            n_used,
            max_tokens,
            inbox,
            _status: st,
            err,
            posted: (0..n).map(|_| Default::default()).collect(),
            taken: (0..n).map(|_| Default::default()).collect(),
        })
    }

    /// The most tokens of a message.
    pub fn max_tokens(&self) -> i64 {
        self.max_tokens
    }

    /// Starts round `round` (> every earlier round's; the GPUs involved idle): every count
    /// restarts at `round << 20`, and the error word clears.
    pub fn begin(&self, round: u64) {
        let base = round << ROUND_SHIFT;
        for c in self.posted.iter().chain(&self.taken).flatten() {
            c.store(base, Ordering::Relaxed);
        }
        self.err_word().store(0, Ordering::Relaxed);
    }

    /// Whether a wait timed out since [`Self::begin`] (the GPUs involved idle).
    pub fn failed(&self) -> bool {
        self.err_word().load(Ordering::Relaxed) != 0
    }

    fn err_word(&self) -> &AtomicU32 {
        // SAFETY: the start of the status tensor (`self._status`), which lives as long as self.
        unsafe { &*(self.err as *const AtomicU32) }
    }

    fn flag(&self, d: usize, kind: Kind) -> &AtomicU64 {
        // SAFETY: inside the header, 8-byte aligned; host memory whenever the CPU op runs.
        unsafe { &*((self.inbox[d].hdr + FLAG_AT[kind as usize]) as *const AtomicU64) }
    }

    /// The task's input rows in GPU `d`'s inbox, `[n_embd, n]` (valid once [`recv_task`] ran).
    pub fn task_x<'a>(&'a self, ctx: &'a Context, d: usize, n: i64) -> Tensor<'a> {
        let x = self.inbox[d].ctx.tensor("x").expect("inbox x");
        ctx.view_2d(x, [self.n_embd, n], self.n_embd as usize * 4, 0)
    }

    /// The task's router weights in GPU `d`'s inbox, `[n_used, n]` (as [`Self::task_x`]).
    pub fn task_w<'a>(&'a self, ctx: &'a Context, d: usize, n: i64) -> Tensor<'a> {
        let w = self.inbox[d].ctx.tensor("w").expect("inbox w");
        ctx.view_2d(w, [self.n_used, n], self.n_used as usize * 4, 0)
    }

    fn params(&self, op: Op, me: usize, peer: usize, own: usize) -> [i32; P_LINK + 2] {
        let p = self as *const Self as u64;
        let mut out = [0i32; P_LINK + 2];
        out[P_OP] = op as i32;
        out[P_ME] = me as i32;
        out[P_PEER] = peer as i32;
        out[P_OWN] = own as i32;
        out[P_LINK] = p as u32 as i32;
        out[P_LINK + 1] = (p >> 32) as u32 as i32;
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    SendTask = 0,
    RecvTask = 1,
    SendPart = 2,
    RecvAdd = 3,
}

const P_OP: usize = 0;
const P_ME: usize = 1;
const P_PEER: usize = 2;
const P_OWN: usize = 3;
const P_LINK: usize = 4;
const _: () = assert!(P_LINK + 2 <= EXTERNAL_PARAMS);

fn cont<'a>(ctx: &'a Context, t: Tensor<'a>) -> Tensor<'a> {
    if t.is_contiguous() { t } else { ctx.cont(t) }
}

/// GPU `me` hands GPU `peer` a task: `x` (`[n_embd, n]` F32), the router's choices `ids`
/// (`[n_used, n]` I32) and weights `w` (`[n_used, n]` F32). The result is a token (one value,
/// never written) to order later ops after the send.
pub fn send_task<'a>(
    ctx: &'a Context,
    link: &PairLink,
    me: usize,
    peer: usize,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
) -> Tensor<'a> {
    let n = x.elements() as i64 / link.n_embd;
    assert!(x.ty() == types::F32 && x.elements() as i64 == link.n_embd * n);
    assert!(ids.ty() == types::I32 && ids.elements() as i64 == link.n_used * n);
    assert!(w.ty() == types::F32 && w.elements() as i64 == link.n_used * n);
    assert!(n <= link.max_tokens, "send_task: {n} tokens");
    let args = [cont(ctx, x), cont(ctx, ids), cont(ctx, w)];
    let p = link.params(Op::SendTask, me, peer, 0);
    ctx.external_with(&LINK, types::I32, [1, 1, 1, 1], &args, &p)
}

/// GPU `me` takes the next task of `n` tokens from its inbox: the router's choices, `[n_used,
/// n]` I32 (the rows and weights stay in the inbox: [`PairLink::task_x`], [`PairLink::task_w`]).
/// `after`: ops it is ordered after (the previous part's send).
pub fn recv_task<'a>(
    ctx: &'a Context,
    link: &PairLink,
    me: usize,
    n: i64,
    after: &[Tensor<'a>],
) -> Tensor<'a> {
    assert!(n <= link.max_tokens, "recv_task: {n} tokens");
    let p = link.params(Op::RecvTask, me, me, 0);
    ctx.external_with(&LINK, types::I32, [link.n_used, n, 1, 1], after, &p)
}

/// GPU `me` hands GPU `peer` its part of a sum, `part` (`[n_embd, n]` F32). A token as
/// [`send_task`]'s.
pub fn send_part<'a>(
    ctx: &'a Context,
    link: &PairLink,
    me: usize,
    peer: usize,
    part: Tensor<'a>,
) -> Tensor<'a> {
    let n = part.elements() as i64 / link.n_embd;
    assert!(part.ty() == types::F32 && part.elements() as i64 == link.n_embd * n);
    assert!(n <= link.max_tokens, "send_part: {n} tokens");
    let p = link.params(Op::SendPart, me, peer, 0);
    ctx.external_with(&LINK, types::I32, [1, 1, 1, 1], &[cont(ctx, part)], &p)
}

/// GPU `me`'s part `own` (`[n_embd, n]` F32) plus the part its pair sends: the whole sum, the
/// same bits whichever GPU holds which part. Ordered after `first` (the task's send, say: the
/// graph runs them before `own`'s ops) and `then` (ops to run before the wait).
pub fn recv_add<'a>(
    ctx: &'a Context,
    link: &PairLink,
    me: usize,
    first: &[Tensor<'a>],
    own: Tensor<'a>,
    then: &[Tensor<'a>],
) -> Tensor<'a> {
    let n = own.elements() as i64 / link.n_embd;
    assert!(own.ty() == types::F32 && own.elements() as i64 == link.n_embd * n);
    assert!(n <= link.max_tokens, "recv_add: {n} tokens");
    let mut args = first.to_vec();
    args.push(cont(ctx, own));
    args.extend_from_slice(then);
    let p = link.params(Op::RecvAdd, me, me, first.len());
    ctx.external_with(&LINK, types::F32, [link.n_embd, n, 1, 1], &args, &p)
}

struct Node {
    link: &'static PairLink,
    op: Op,
    me: usize,
    peer: usize,
    own: usize,
}

/// # Safety
/// `dst` is a node made by this module's builders, its link alive.
unsafe fn node_of(dst: &ffi::ggml_tensor) -> Node {
    let p = external_params(dst);
    let addr = (p[P_LINK] as u32 as u64) | ((p[P_LINK + 1] as u32 as u64) << 32);
    // SAFETY: the address the builder stored; the engine keeps the link alive while graphs
    // using it exist.
    let link = unsafe { &*(addr as usize as *const PairLink) };
    let op = match p[P_OP] {
        0 => Op::SendTask,
        1 => Op::RecvTask,
        2 => Op::SendPart,
        _ => Op::RecvAdd,
    };
    Node {
        link,
        op,
        me: p[P_ME] as usize,
        peer: p[P_PEER] as usize,
        own: p[P_OWN] as usize,
    }
}

impl Node {
    /// The kind of message, its receiver and its value (counted here: once per op run).
    fn message(&self) -> (Kind, usize, u64) {
        let l = self.link;
        let (kind, to, counts) = match self.op {
            Op::SendTask => (Kind::Task, self.peer, &l.posted),
            Op::SendPart => (Kind::Part, self.peer, &l.posted),
            Op::RecvTask => (Kind::Task, self.me, &l.taken),
            Op::RecvAdd => (Kind::Part, self.me, &l.taken),
        };
        let v = counts[to][kind as usize].fetch_add(1, Ordering::Relaxed) + 1;
        (kind, to, v)
    }

    /// The receiver's payload segments of a send: (destination, source index, bytes).
    fn segments(&self, dst: &ffi::ggml_tensor) -> Vec<(usize, usize, usize)> {
        let l = self.link;
        let to = &l.inbox[self.peer];
        // SAFETY: the node's sources, as its builder made them.
        let bytes = |i: usize| unsafe { ffi::ggml_nbytes(dst.src[i]) };
        match self.op {
            Op::SendTask => vec![
                (to.x, 0, bytes(0)),
                (to.ids, 1, bytes(1)),
                (to.w, 2, bytes(2)),
            ],
            Op::SendPart => vec![(to.part, 0, bytes(0))],
            _ => Vec::new(),
        }
    }
}

/// The CPU op: one thread does it all.
unsafe fn cpu_link(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node of ours; its sources and the inboxes are host memory.
    unsafe {
        let node = node_of(dst);
        let l = node.link;
        let (kind, to, v) = node.message();
        let src = |i: usize| (*dst.src[i]).data as *const u8;
        match node.op {
            Op::SendTask | Op::SendPart => {
                for (at, i, bytes) in node.segments(dst) {
                    std::ptr::copy_nonoverlapping(src(i), at as *mut u8, bytes);
                }
                l.flag(to, kind).store(v, Ordering::Release);
            }
            Op::RecvTask | Op::RecvAdd => {
                let t0 = Instant::now();
                while l.flag(to, kind).load(Ordering::Acquire) != v {
                    if l.failed() || t0.elapsed() > TIMEOUT {
                        l.err_word().store(1, Ordering::Relaxed);
                        break;
                    }
                    std::hint::spin_loop();
                }
                let n = (dst.ne[0] * dst.ne[1]) as usize;
                if node.op == Op::RecvTask {
                    let from = l.inbox[to].ids as *const i32;
                    std::ptr::copy_nonoverlapping(from, dst.data as *mut i32, n);
                } else {
                    let own = src(node.own) as *const f32;
                    let part = l.inbox[to].part as *const f32;
                    let out = dst.data as *mut f32;
                    for i in 0..n {
                        *out.add(i) = *own.add(i) + *part.add(i);
                    }
                }
            }
        }
    }
}

/// `legion_pair_link` (`src/cuda/pair_link.cu`); also launched by [`super::sum`].
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Kernel {
    pub seg_dst: [*mut u8; 3],
    pub seg_src: [*const u8; 3],
    pub seg_bytes: [i64; 3],
    pub flag: *mut u64,
    pub done: *mut u32,
    pub err: *mut u32,
    pub from: *const u8,
    pub own: *const f32,
    pub dst: *mut u8,
    pub n: i64,
    pub v: u64,
    pub timeout_ns: i64,
    pub mode: i32,
    pub pad: i32,
}

#[cfg(feature = "cuda")]
impl Kernel {
    /// A message of value `v` (the rest null: [`Self::launch`]'s caller fills its mode's fields).
    pub(super) fn empty(v: u64, err: *mut u32, timeout: Duration) -> Self {
        Self {
            seg_dst: [std::ptr::null_mut(); 3],
            seg_src: [std::ptr::null(); 3],
            seg_bytes: [0; 3],
            flag: std::ptr::null_mut(),
            done: std::ptr::null_mut(),
            err,
            from: std::ptr::null(),
            own: std::ptr::null(),
            dst: std::ptr::null_mut(),
            n: 0,
            v,
            timeout_ns: timeout.as_nanos() as i64,
            mode: 0,
            pad: 0,
        }
    }

    /// Enqueues the kernel with `grid` blocks of [`THREADS`] on `stream`.
    ///
    /// # Safety
    /// The pointers are valid on the stream's device for the kernel's mode.
    pub(super) unsafe fn launch(mut self, grid: u32, stream: *mut c_void) -> Result<(), String> {
        let mut args = [(&raw mut self).cast::<c_void>()];
        // SAFETY: the caller's; the runtime copies the parameter block at launch.
        match unsafe {
            cudaLaunchKernel(
                legion_pair_link_kernel,
                Dim3 {
                    x: grid,
                    y: 1,
                    z: 1,
                },
                Dim3 {
                    x: THREADS,
                    y: 1,
                    z: 1,
                },
                args.as_mut_ptr(),
                0,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("CUDA error {e}")),
        }
    }
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
    static legion_pair_link_kernel: *const c_void;
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
pub(super) const THREADS: u32 = 256;
#[cfg(feature = "cuda")]
const MAX_BLOCKS: u32 = 32;

#[cfg(feature = "cuda")]
unsafe fn cuda_link(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of ours on the stream's device; the kernel reads `k` from the launch's copy.
    unsafe {
        let node = node_of(dst);
        let l = node.link;
        let (kind, to, v) = node.message();
        let mut k = Kernel::empty(v, l.err as *mut u32, TIMEOUT);
        k.flag = (l.inbox[to].hdr + FLAG_AT[kind as usize]) as *mut u64;
        k.dst = dst.data as *mut u8;
        k.n = dst.ne[0] * dst.ne[1];
        let blocks = |values: i64, per_thread: i64| -> u32 {
            let per_block = THREADS as i64 * per_thread;
            ((values.max(0) as u64).div_ceil(per_block as u64) as u32).clamp(1, MAX_BLOCKS)
        };
        let grid = match node.op {
            Op::SendTask | Op::SendPart => {
                let segs = node.segments(dst);
                for (s, &(at, i, bytes)) in segs.iter().enumerate() {
                    k.seg_dst[s] = at as *mut u8;
                    k.seg_src[s] = (*dst.src[i]).data as *const u8;
                    k.seg_bytes[s] = bytes as i64;
                }
                k.done = (l.inbox[node.me].hdr + DONE_AT[kind as usize]) as *mut u32;
                let total: usize = segs.iter().map(|s| s.2).sum();
                // 16 bytes a thread, a few loads each.
                blocks(total as i64, 16 * 4)
            }
            Op::RecvTask => {
                k.mode = 1;
                k.from = l.inbox[to].ids as *const u8;
                1
            }
            Op::RecvAdd => {
                k.mode = 2;
                k.from = l.inbox[to].part as *const u8;
                k.own = (*dst.src[node.own]).data as *const f32;
                blocks(k.n, 4)
            }
        };
        k.launch(grid, stream)
            .map_err(|e| format!("pair_link: {e}"))
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaGetDevice(d: *mut i32) -> i32;
    fn cudaSetDevice(d: i32) -> i32;
    fn cudaDeviceCanAccessPeer(can: *mut i32, d: i32, peer: i32) -> i32;
    fn cudaDeviceEnablePeerAccess(peer: i32, flags: u32) -> i32;
    fn cudaGetLastError() -> i32;
}

/// Whether CUDA device `d` can access CUDA device `p`'s memory directly.
#[cfg(feature = "cuda")]
pub fn can_access(d: usize, p: usize) -> bool {
    let mut can = 0;
    // SAFETY: a plain runtime call writing its out-parameter.
    let e = unsafe { cudaDeviceCanAccessPeer(&mut can, d as i32, p as i32) };
    e == 0 && can != 0
}

/// Lets each CUDA device of `pairs` write into the other's memory (peer access both ways).
#[cfg(feature = "cuda")]
pub fn enable_peers(pairs: &[(usize, usize)]) -> Result<(), String> {
    const ALREADY_ENABLED: i32 = 704; // cudaErrorPeerAccessAlreadyEnabled
    let mut current = 0;
    // SAFETY: plain runtime calls writing their out-parameters; the current device is put back.
    unsafe {
        if cudaGetDevice(&mut current) != 0 {
            return Err("cudaGetDevice failed".into());
        }
        let made = (|| {
            for &(a, b) in pairs {
                for (d, p) in [(a as i32, b as i32), (b as i32, a as i32)] {
                    if !can_access(d as usize, p as usize) {
                        return Err(format!("CUDA{d} cannot access CUDA{p}'s memory"));
                    }
                    if cudaSetDevice(d) != 0 {
                        return Err(format!("cudaSetDevice({d}) failed"));
                    }
                    match cudaDeviceEnablePeerAccess(p, 0) {
                        0 => {}
                        ALREADY_ENABLED => {
                            cudaGetLastError();
                        }
                        e => return Err(format!("CUDA{d} -> CUDA{p}: peer access error {e}")),
                    }
                }
            }
            Ok(())
        })();
        cudaSetDevice(current);
        made
    }
}

#[cfg(test)]
mod tests;
