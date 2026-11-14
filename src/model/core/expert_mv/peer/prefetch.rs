//! The peer fetch one layer early (#519): while a layer computes, the idle NVLink peer also
//! copies a guess of the *next* layer's missed experts into the running GPU's VRAM, so that the
//! next layer finds them there instead of reading them over the link inside its own kernels.
//!
//! The guess is the next layer's router over this layer's input (GLM-5.3-Flash's prefetch
//! guesses the same way): measured on Qwen3.8-Flash-Next over two V100s it names 60-67% of the
//! next layer's missed experts, where the previous step's choices of that layer name 19-31%.
//! The peer computes it: the router's rows (the running GPU's memory) over a copy of the
//! layer's input, the largest logits per token looked up in the next layer's table of tiered
//! ids. Nothing is added to the step's graph, whose layout the text depends on.
//!
//! Per device, next to the peer's staging slots ([`Layout`]):
//! - two halves of [`SLOTS`] prefetch slots: a layer reads one while the peer fills the other
//!   for the layer after it; per half the tiered id each slot holds and a stamp per slot and
//!   copy CTA;
//! - the guess's ids, the ids left for the peer's copy inside the layer ("rest"), the running
//!   GPU's progress word, the layer's input and the guess's logits.
//!
//! The prefetch never makes the running GPU wait (round 1 did: the next FORK waited for the
//! copy, and the next layer's own peer copies queued behind it). It runs on a stream of its own
//! on the peer, and a copy is used only if it is done in time:
//! - a FORK op of the peer alone ([`fork`]): one small kernel (`stage_prepare`) on the running
//!   GPU writes the progress word (the peer's copy for this layer stops there), takes the
//!   prefetch slots whose stamps all say "done for this layer", matches the layer's ids against
//!   the ids they hold and writes the slot table (an expert a done slot holds: that slot; the
//!   others as before, the peer's share ranked over the ids left), the ids left and the copy of
//!   the layer's input. The peer then copies its share of the experts left as before, on its
//!   own stream;
//! - behind that ([`ahead`]), on the prefetch stream: once the running GPU's FORK is in, the
//!   guess for the layer whose FORK followed this one's last time (a layer says where its router
//!   and table are when its graph is built, [`register`]); once the running GPU is past this
//!   layer's experts (its JOIN op, [`join`]: so the copy reads the host's memory while the running
//!   GPU does not, and the peer's own copies for the layer are done), the copy of the guessed host
//!   experts into the free half (`prefetch_copy`),
//!   which stamps each slot when its bytes are in and stops as soon as the progress word says the
//!   next layer has started. It also learns per layer how long it has until that FORK and how long
//!   a slot takes, and does not start a slot that would end after it. So the copy only uses the
//!   peer while it is idle, and a slot that is late is read from host memory as without the
//!   prefetch. The device's first layer is not
//!   prefetched from its last one: the peer runs its own layers between them.
//!
//! The guess reads the layer's input from the buffer without holding the next FORK op back: if
//! that op's `stage_prepare` overwrote it first, the guess is just worse.
//!
//! Outputs are the same bits: a done prefetch slot holds the host bytes of the very tiered id
//! the layer asks for (ids are compared exactly, the stamps follow a system fence behind the
//! bytes, and a tiered tensor's host part never changes), and every pair is computed by the same
//! tile code as a peer-staged one. A wrong or late guess costs the idle peer a copy; the layer
//! then runs as without the prefetch.
//!
//! Opt-in (`legion serve --expert-peer-prefetch`), and only where the peer fetch is on by the
//! automatic choice for a layer split over one NVLink pair ([`super::auto`]: two V100s of an
//! AC922). Architectures with their own prefetch (`peer_decode_pairs`) do not use it.

// Without CUDA only the switches and the registry are used (and the tests).
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(feature = "cuda")]
use std::cell::Cell;
#[cfg(feature = "cuda")]
use std::os::raw::{c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

static WANTED: AtomicBool = AtomicBool::new(false);
static ALLOWED: AtomicBool = AtomicBool::new(false);

/// `legion serve --expert-peer-prefetch`: whether the prefetch is wanted where it can run.
pub fn set_enabled(on: bool) {
    WANTED.store(on, Ordering::Relaxed);
}

/// The load's choice: whether it can run ([`super::auto`] chose the peer fetch for a plain layer
/// split).
pub fn set_auto(on: bool) {
    ALLOWED.store(on, Ordering::Relaxed);
}

/// Whether FORK ops launched from now on prefetch for the layer after them.
pub fn enabled() -> bool {
    ALLOWED.load(Ordering::Relaxed) && WANTED.load(Ordering::Relaxed)
}

/// Prefetch slots per half: the guessed experts copied for one layer at most (the likeliest
/// uncached ones first; at most 8, `prefetch_copy`'s bound).
pub const SLOTS: usize = 4;

/// CTAs of the prefetch copy (at most 64, the stamps' bound). The peer is idle when it runs, so
/// it can take as many as its copies inside a layer do.
pub const CTAS: usize = 24;

/// Bytes of a half's stamps: a u64 per slot and copy CTA, 256-byte aligned.
const STAMPS: usize = (SLOTS * CTAS * 8).next_multiple_of(256);

/// Layers per device whose order is kept.
pub const MAX_LAYERS: usize = 64;

/// Pairs of a step at most (`stage_copy`'s).
const MAX_PAIRS: usize = 512;

/// Tokens of a step whose next layer is guessed, and the widest input and router (the guess
/// kernels' bounds).
const MAX_TOKENS: usize = 16;
const MAX_EMBD: usize = 8192;
const MAX_EXPERTS: usize = 1024;

/// Bytes of a list of ids or slots, one i32 per pair.
const LIST: usize = MAX_PAIRS * 4;

/// Where a layer's router and table of tiered ids are ([`register`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Router {
    /// The router's address: `n_expert` contiguous rows of `n_embd`, in the memory of the GPU
    /// that runs the layer.
    pub w: usize,
    /// Its rows are F16 (`--weights-fp16-gates`), else F32.
    pub f16: bool,
    pub n_embd: usize,
    pub n_expert: usize,
    /// The address of the layer's table: an i32 tiered id per expert, same GPU.
    pub table: usize,
}

static ROUTERS: std::sync::Mutex<Vec<(usize, Router)>> = std::sync::Mutex::new(Vec::new());

/// The layer whose gate stack is at `gate` routes with `router` (said when its graph is built;
/// the last word counts). Without it nothing is prefetched for the layer.
pub fn register(gate: usize, router: Router) {
    let mut all = ROUTERS.lock().unwrap();
    match all.iter_mut().find(|(g, _)| *g == gate) {
        Some((_, r)) => *r = router,
        None => all.push((gate, router)),
    }
}

/// The router [`register`]ed for the gate stack at `gate`, if the guess kernels take it.
fn router_of(gate: usize) -> Option<Router> {
    let all = ROUTERS.lock().unwrap();
    all.iter()
        .find(|(g, _)| *g == gate)
        .map(|(_, r)| *r)
        .filter(|r| (1..=MAX_EMBD).contains(&r.n_embd) && (1..=MAX_EXPERTS).contains(&r.n_expert))
}

/// The device's staging buffer with the prefetch: the slot table, the peer's slots, the two
/// halves of prefetch slots, then the lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Layout {
    /// The bytes from a slot to the next: the largest slot of the device's layers so far (their
    /// experts' bytes differ where their quantizations do).
    pub slot: usize,
    /// The peer's slots (for the layer's own misses).
    pub peer_slots: usize,
}

impl Layout {
    /// The slot table's bytes, whatever the step's pairs (so the slots never move).
    pub const HDR: usize = LIST;

    /// The first slot of prefetch half `half`.
    pub fn half(&self, half: usize) -> usize {
        self.peer_slots + half * SLOTS
    }

    /// Where slot `s` starts.
    pub fn slot_at(&self, s: usize) -> usize {
        Self::HDR + s * self.slot
    }

    fn lists(&self) -> usize {
        self.slot_at(self.half(2)).next_multiple_of(256)
    }

    /// The ids left for the peer's copy inside the layer.
    pub fn rest(&self) -> usize {
        self.lists()
    }

    /// The guess's tiered ids (the peer's scratch; one guess is made at a time).
    pub fn guess(&self) -> usize {
        self.lists() + LIST
    }

    /// The stamps of prefetch half `half`: a u64 per slot and copy CTA.
    pub fn stamps(&self, half: usize) -> usize {
        self.lists() + 2 * LIST + half * STAMPS
    }

    /// The tiered id each slot of prefetch half `half` holds (an i32 per slot).
    pub fn pf_ids(&self, half: usize) -> usize {
        self.stamps(2) + half * 256
    }

    /// The running GPU's progress word (u64): the stamp of the last FORK op's prefetch.
    pub fn progress(&self) -> usize {
        self.pf_ids(2)
    }

    /// The counters (u64): four of the running GPU's, then 32 bytes on three of the prefetch
    /// copy's.
    pub fn count(&self) -> usize {
        self.progress() + 256
    }

    /// The times the prefetch copy learned for the device's layer `layer` (two u64: from the
    /// copy's start to the layer's FORK, and a slot's copy).
    pub fn timing(&self, layer: usize) -> usize {
        self.count() + 256 + layer * 16
    }

    /// The words zeroed with a new buffer (stamps, ids, progress, counters, times): from
    /// `stamps(0)` this many bytes.
    pub fn state_bytes(&self) -> usize {
        self.x() - self.stamps(0)
    }

    /// The copy of the layer's input ([`MAX_TOKENS`] columns of [`MAX_EMBD`] floats).
    pub fn x(&self) -> usize {
        self.timing(MAX_LAYERS)
    }

    /// The guess's logits ([`MAX_TOKENS`] rows of [`MAX_EXPERTS`] floats).
    pub fn logits(&self) -> usize {
        self.x() + MAX_TOKENS * MAX_EMBD * 4
    }

    pub fn bytes(&self) -> usize {
        self.logits() + MAX_TOKENS * MAX_EXPERTS * 4
    }
}

/// What a layer's ids index: its expert stacks (gate, up, down; their addresses), an expert's
/// bytes in the gate (and up) stack and in the down stack, and the tiered ids' bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub stacks: [usize; 3],
    pub nb2: [usize; 2],
    pub vram_below: i32,
    pub n_w: usize,
}

struct Layer {
    shape: Shape,
    /// The layer whose FORK followed this one's last time.
    next: Option<usize>,
}

/// A FORK op's part of the prefetch ([`Chain::visit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Visit {
    pub layer: usize,
    /// The copy the peer was asked to make for this layer: its stamp and the prefetch half.
    pub guess: Option<(u64, usize)>,
}

/// The copy to queue on the peer for the next layer ([`Chain::ahead`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ahead {
    pub layer: usize,
    pub shape: Shape,
    pub half: usize,
}

/// A device's layers in the order their FORK ops come, and what the peer last prefetched.
#[derive(Default)]
pub(super) struct Chain {
    layers: Vec<Layer>,
    last: Option<usize>,
    /// The layer the last prefetch was for, the shape it was copied with, its stamp and the
    /// half.
    pf: Option<(usize, Shape, u64, usize)>,
    /// The half the last prefetch wrote (the next one takes the other: the layer that may read
    /// that one runs meanwhile).
    half: usize,
}

impl Chain {
    /// The buffer is new or was written by others: nothing prefetched. The order of the layers
    /// stays.
    pub fn reset(&mut self) {
        self.pf = None;
    }

    /// The FORK op of the layer over `shape`: which guess it matches. `None`: too many layers.
    pub fn visit(&mut self, shape: Shape) -> Option<Visit> {
        let pf = self.pf.take();
        let ix = match self
            .layers
            .iter()
            .position(|l| l.shape.stacks == shape.stacks)
        {
            Some(ix) => ix,
            None if self.layers.len() < MAX_LAYERS => {
                self.layers.push(Layer { shape, next: None });
                self.layers.len() - 1
            }
            None => return None,
        };
        if let Some(prev) = self.last
            && prev != ix
        {
            self.layers[prev].next = Some(ix);
        }
        self.last = Some(ix);
        self.layers[ix].shape = shape;
        // A copy made for this layer, with the id bounds it has now.
        let guess = match pf {
            Some((p, s, seq, half)) if p == ix && s == shape => Some((seq, half)),
            _ => None,
        };
        Some(Visit { layer: ix, guess })
    }

    /// The layer after `layer` if it comes later in the order: the device's first layer after
    /// its last one runs once the peer has run its own layers, so a copy for it would share the
    /// peer with them (it is not made).
    fn after(&self, layer: usize) -> Option<usize> {
        self.layers[layer].next.filter(|&nx| nx > layer)
    }

    /// The prefetch to queue while `layer` computes: a guess of `n` ids for the layer after it,
    /// stamped `seq`, into the half the last prefetch did not write (the one `layer` may read).
    /// `None` without a next layer ([`Self::next`]) or ids.
    pub fn ahead(&mut self, layer: usize, seq: u64, n: usize) -> Option<Ahead> {
        let nx = self.after(layer)?;
        let shape = self.layers[nx].shape;
        if n == 0 {
            return None;
        }
        let half = 1 - self.half;
        self.half = half;
        self.pf = Some((nx, shape, seq, half));
        Some(Ahead {
            layer: nx,
            shape,
            half,
        })
    }

    /// The layer after `layer`, as [`Self::ahead`] would take it.
    pub fn next(&self, layer: usize) -> Option<Shape> {
        Some(self.layers[self.after(layer)?].shape)
    }
}

/// FORK ops between two reports of a device ([`report`]).
#[cfg(feature = "cuda")]
const REPORT_EVERY: u64 = 16384;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaSetDevice(d: c_int) -> c_int;
    fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
    fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: c_uint) -> c_int;
    fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int;
    fn cudaStreamCreateWithPriority(s: *mut *mut c_void, flags: c_uint, priority: c_int) -> c_int;
    fn cudaMemsetAsync(p: *mut c_void, v: c_int, bytes: usize, s: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
    fn legion_expert_prepare_launch(p: *const PrepareParams, stream: *mut c_void) -> c_int;
    fn legion_expert_guess_launch(p: *const GuessParams, stream: *mut c_void) -> c_int;
    fn legion_expert_prefetch_launch(p: *const PrefetchParams, stream: *mut c_void) -> c_int;
}

/// `legion_expert_prepare` (src/cuda/expert_mv/common.cuh).
#[cfg(feature = "cuda")]
#[repr(C)]
struct PrepareParams {
    ids: *const i32,
    ids_nb1: i64,
    n_used: i32,
    n_tok: i32,
    vram_below: i32,
    n_w: i32,
    table: *mut i32,
    rest: *mut i32,
    stamps: *const u64,
    pf_ids: *const i32,
    pf_slots: i32,
    pf_ctas: i32,
    seq: u64,
    now: u64,
    progress: *mut u64,
    pf_base: i32,
    cap: i32,
    t_every: i32,
    t_take: i32,
    k: i32,
    x: *const u8,
    x_nb2: i64,
    x_keep: *mut f32,
    count: *mut u64,
}

/// `legion_expert_guess` (src/cuda/expert_mv/common.cuh).
#[cfg(feature = "cuda")]
#[repr(C)]
struct GuessParams {
    w: *const c_void,
    x: *const f32,
    k: i32,
    n_expert: i32,
    n_tok: i32,
    n_used: i32,
    w_f16: i32,
    logits: *mut f32,
    table: *const i32,
    guess: *mut i32,
}

/// `legion_expert_prefetch` (src/cuda/expert_mv/common.cuh).
#[cfg(feature = "cuda")]
#[repr(C)]
struct PrefetchParams {
    guess: *const i32,
    n_guess: i32,
    vram_below: i32,
    n_w: i32,
    cap: i32,
    slots: *mut u8,
    slot: i64,
    n_seg: i32,
    src: [*const u8; 3],
    nb2: [i64; 3],
    off: [i64; 3],
    stamps: *mut u64,
    pf_ids: *mut i32,
    seq: u64,
    progress: *const u64,
    count: *mut u64,
    timing: *mut u64,
    ctas: i32,
}

/// Counters of the running GPU's (`stage_prepare`, four) and of the prefetch copy (three, 32
/// bytes on), in the buffer.
#[cfg(feature = "cuda")]
const COUNTERS: usize = 7;

#[cfg(feature = "cuda")]
struct Dev {
    device: c_int,
    buf: usize,
    layout: Layout,
    /// On the peer: the prefetch's own stream (the lowest priority, the peer stream's), and an
    /// event recorded on it after each prefetch.
    stream: usize,
    event: usize,
    /// On the device: recorded on its stream by a JOIN op, waited for by the prefetch's copy.
    joined: usize,
    /// The copy [`ahead`] guessed for, which the FORK op's JOIN op queues ([`join`]).
    pending: Option<(Ahead, u64, usize)>,
    /// Whether a prefetch was ever queued (the event recorded).
    launched: bool,
    /// The last stamp handed out (0: none yet; the stamps start zeroed).
    seq: u64,
    chain: Chain,
    visits: u64,
    /// Prefetches queued since the last report.
    queued: u64,
    seen: [u64; COUNTERS],
}

#[cfg(feature = "cuda")]
impl Dev {
    /// `main` waits for the prefetch in flight: before a pair of ops that runs without the
    /// prefetch over the same buffer, or one that may move the buffer. Not in the steady state.
    ///
    /// # Safety
    /// `main` is a live stream of the device.
    unsafe fn quiesce(&self, main: *mut c_void) -> Result<(), String> {
        if !self.launched {
            return Ok(());
        }
        // SAFETY: live handles; a stream may wait for another device's event.
        super::rt(
            unsafe { cudaStreamWaitEvent(main, self.event as *mut c_void, 0) },
            "cudaStreamWaitEvent",
        )
    }
}

#[cfg(feature = "cuda")]
static DEVS: std::sync::Mutex<Vec<Dev>> = std::sync::Mutex::new(Vec::new());

/// A FORK op with the prefetch: its staging buffer and where its parts are.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(in crate::model::core::expert_mv) struct Forked {
    pub buf: *mut u8,
    /// The slot table's bytes.
    pub hdr: usize,
    /// The bytes from a slot to the next.
    pub slot: usize,
    pub bytes: usize,
    /// The ids the peer's copy inside the layer ranks, `n_used * 4` bytes per token.
    pub rest: *const i32,
    layer: usize,
    /// The guess to make for the next layer: its router, this step's tokens and experts per
    /// token (the layer's input is in the buffer).
    guess: Option<(Router, usize, usize)>,
}

#[cfg(feature = "cuda")]
thread_local! {
    /// The last FORK op's on this thread, for its JOIN op.
    static FORKED: Cell<Option<Forked>> = const { Cell::new(None) };
}

/// A FORK op of the peer alone, as [`fork`] takes it.
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) struct Op {
    pub ids: *const i32,
    pub ids_nb1: usize,
    pub n_used: usize,
    pub n_tok: usize,
    pub stacks: [*const u8; 3],
    /// An expert's bytes in the gate (and up) stack and in the down stack.
    pub nb2: [usize; 2],
    pub vram_below: i32,
    pub n_w: usize,
    /// A slot's bytes, and the peer's slots for this step.
    pub slot: usize,
    pub cap: usize,
    /// The layer's input: `n_tok` columns of `k` F32 `x_nb2` bytes apart, or null when it is
    /// not one column per token.
    pub x: *const u8,
    pub x_nb2: usize,
    pub k: usize,
}

/// The FORK op's part on the running GPU, before the peer is forked (see the module docs):
/// `stage_prepare` on `main`, which waits for nothing on the peer. `None`: this pair of ops runs
/// without the prefetch (off, or a step it does not take), on [`super::buffer`] as before; then
/// (and when the buffer may move) `main` waits for the copy in flight.
///
/// # Safety
/// `main` is a live stream of the current device, `l`'s; `op`'s pointers are that device's
/// memory of the sizes it gives (the stacks: tiered tensors whose host parts the peer reads).
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) unsafe fn fork(
    main: *mut c_void,
    l: super::Link,
    op: &Op,
) -> Result<Option<Forked>, String> {
    FORKED.with(|f| f.set(None));
    if !enabled() {
        return Ok(None);
    }
    let mut devs = DEVS.lock().unwrap();
    let d = match devs.iter().position(|d| d.device == l.device) {
        Some(i) => &mut devs[i],
        None => {
            const NON_BLOCKING: c_uint = 1;
            const DISABLE_TIMING: c_uint = 2;
            let (mut event, mut stream) = (std::ptr::null_mut(), std::ptr::null_mut());
            let (mut least, mut greatest) = (0, 0);
            // SAFETY: plain runtime calls; the stream and the event are the peer's, the current
            // device is put back.
            unsafe {
                super::rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
                let made = super::rt(
                    cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest),
                    "cudaDeviceGetStreamPriorityRange",
                )
                .and_then(|()| {
                    super::rt(
                        cudaStreamCreateWithPriority(&mut stream, NON_BLOCKING, least),
                        "cudaStreamCreateWithPriority",
                    )
                })
                .and_then(|()| {
                    super::rt(
                        cudaEventCreateWithFlags(&mut event, DISABLE_TIMING),
                        "cudaEventCreate",
                    )
                });
                super::rt(cudaSetDevice(l.device), "cudaSetDevice")?;
                made?;
            }
            let mut joined = std::ptr::null_mut();
            // SAFETY: a plain runtime call on the current device, `l`'s.
            super::rt(
                unsafe { cudaEventCreateWithFlags(&mut joined, DISABLE_TIMING) },
                "cudaEventCreate",
            )?;
            devs.push(Dev {
                device: l.device,
                buf: 0,
                layout: Layout::default(),
                stream: stream as usize,
                event: event as usize,
                joined: joined as usize,
                pending: None,
                launched: false,
                seq: 0,
                chain: Chain::default(),
                visits: 0,
                queued: 0,
                seen: [0; COUNTERS],
            });
            devs.last_mut().unwrap()
        }
    };
    // A guess whose JOIN op never came is not copied.
    d.pending = None;
    let n_pairs = op.n_used * op.n_tok;
    let layout = Layout {
        slot: d.layout.slot.max(op.slot),
        peer_slots: super::slots(super::super::overlap::stage_pairs()),
    };
    if n_pairs == 0
        || n_pairs > MAX_PAIRS
        || op.slot != 2 * op.nb2[0] + op.nb2[1]
        || !op.slot.is_multiple_of(16)
        || op.cap > layout.peer_slots
    {
        // The pair runs as before, over the same buffer (its slots may reach the prefetch
        // halves, and it may grow it): the copy in flight is done first, and nothing here is
        // valid after it.
        d.chain.reset();
        // SAFETY: `main` is live.
        unsafe { d.quiesce(main)? };
        return Ok(None);
    }
    if layout != d.layout {
        // The buffer may grow (the old one is freed once `main` is done): the copy in flight
        // into it is done first.
        // SAFETY: `main` is live.
        unsafe { d.quiesce(main)? };
    }
    // SAFETY: as the caller promises.
    let buf = unsafe { super::buffer(main, l, layout.bytes())? };
    let at = |off: usize| buf.wrapping_add(off);
    if buf as usize != d.buf || layout != d.layout {
        (d.buf, d.layout, d.seen) = (buf as usize, layout, [0; COUNTERS]);
        d.chain.reset();
        // Stamps, ids, the progress word and the counters start at zero (the stamps handed out
        // go on from `d.seq`, so none of them is ever 0).
        // SAFETY: the words are inside the buffer.
        super::rt(
            unsafe {
                cudaMemsetAsync(
                    at(layout.stamps(0)) as *mut c_void,
                    0,
                    layout.state_bytes(),
                    main,
                )
            },
            "cudaMemsetAsync",
        )?;
    }
    let shape = Shape {
        stacks: op.stacks.map(|p| p as usize),
        nb2: op.nb2,
        vram_below: op.vram_below,
        n_w: op.n_w,
    };
    let Some(v) = d.chain.visit(shape) else {
        return Ok(None);
    };
    // The next layer's guess: its router over this layer's input, kept for the peer.
    let guess = d
        .chain
        .next(v.layer)
        .and_then(|next| router_of(next.stacks[0]))
        .filter(|r| !op.x.is_null() && r.n_embd == op.k && op.n_tok <= MAX_TOKENS)
        .map(|r| (r, op.n_tok, op.n_used.min(r.n_expert)));
    let (every, take) = super::share();
    let (stamps, pf_ids, seq, pf_base) = match v.guess {
        Some((seq, half)) => (
            at(layout.stamps(half)),
            at(layout.pf_ids(half)),
            seq,
            layout.half(half),
        ),
        None => (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0),
    };
    let p = PrepareParams {
        ids: op.ids,
        ids_nb1: op.ids_nb1 as i64,
        n_used: op.n_used as i32,
        n_tok: op.n_tok as i32,
        vram_below: op.vram_below,
        n_w: op.n_w as i32,
        table: buf as *mut i32,
        rest: at(layout.rest()) as *mut i32,
        stamps: stamps as *const u64,
        pf_ids: pf_ids as *const i32,
        pf_slots: SLOTS as i32,
        pf_ctas: CTAS as i32,
        seq,
        // Every copy handed out so far stops here (the one for this layer included).
        now: d.seq,
        progress: at(layout.progress()) as *mut u64,
        pf_base: pf_base as i32,
        cap: op.cap as i32,
        t_every: every,
        t_take: take,
        k: op.k as i32,
        x: op.x,
        x_nb2: op.x_nb2 as i64,
        x_keep: match guess {
            Some(_) => at(layout.x()) as *mut f32,
            None => std::ptr::null_mut(),
        },
        count: at(layout.count()) as *mut u64,
    };
    // SAFETY: the lists are inside the buffer, the ids and the input the caller's.
    match unsafe { legion_expert_prepare_launch(&p, main) } {
        0 => {}
        e => return Err(format!("CUDA error {e} (expert prefetch prepare)")),
    }
    d.visits += 1;
    if d.visits.is_multiple_of(REPORT_EVERY) {
        // SAFETY: `main` is live; the counters are inside the buffer.
        unsafe { report(d, main)? };
    }
    let f = Forked {
        buf,
        hdr: Layout::HDR,
        slot: layout.slot,
        bytes: layout.bytes(),
        rest: at(layout.rest()) as *const i32,
        layer: v.layer,
        guess,
    };
    FORKED.with(|c| c.set(Some(f)));
    Ok(Some(f))
}

/// The FORK op's last part on the peer, after [`super::fork`], on the prefetch stream: the guess
/// for the layer after it, then once the layer's own copies are done (the peer is idle) the
/// guessed host experts' copy into the free half (gate, up and down rows), and an event. Nothing
/// on the running GPU or the peer's stream waits for it.
///
/// # Safety
/// As [`fork`], which returned `f` for this op; the current device is `l`'s. The router and
/// table [`register`]ed for the next layer are live memory of this device.
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) unsafe fn ahead(
    l: super::Link,
    f: &Forked,
) -> Result<(), String> {
    let Some((r, n_tok, n_used)) = f.guess else {
        return Ok(());
    };
    let mut devs = DEVS.lock().unwrap();
    let Some(d) = devs.iter_mut().find(|d| d.device == l.device) else {
        return Ok(());
    };
    let layout = d.layout;
    // The next layer's experts fit the slots (they do once its FORK op came with this layout).
    if d.chain
        .next(f.layer)
        .is_none_or(|s| 2 * s.nb2[0] + s.nb2[1] > layout.slot)
    {
        return Ok(());
    }
    let seq = d.seq + 1;
    let Some(a) = d.chain.ahead(f.layer, seq, n_tok * n_used) else {
        return Ok(());
    };
    let at = |off: usize| f.buf.wrapping_add(off);
    let g = GuessParams {
        w: r.w as *const c_void,
        x: at(layout.x()) as *const f32,
        k: r.n_embd as i32,
        n_expert: r.n_expert as i32,
        n_tok: n_tok as i32,
        n_used: n_used as i32,
        w_f16: i32::from(r.f16),
        logits: at(layout.logits()) as *mut f32,
        table: r.table as *const i32,
        guess: at(layout.guess()) as *mut i32,
    };
    let (stream, event) = (d.stream as *mut c_void, d.event as *mut c_void);
    let fork = l.fork as *mut c_void;
    // SAFETY: live handles; the kernel goes to the prefetch stream with the peer current, and the
    // current device is put back whatever happens. The peer reaches the buffer, the router and
    // the table (this device's memory).
    unsafe {
        super::rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
        let queued = (|| {
            // The layer's input is in once its FORK is (recorded just before by this FORK op).
            // The guess runs beside the peer's own copies for the layer: it reads the router
            // over the link the other way.
            super::rt(cudaStreamWaitEvent(stream, fork, 0), "cudaStreamWaitEvent")?;
            match legion_expert_guess_launch(&g, stream) {
                0 => {}
                e => return Err(format!("CUDA error {e} (expert prefetch guess)")),
            }
            super::rt(cudaEventRecord(event, stream), "cudaEventRecord")
        })();
        let back = super::rt(cudaSetDevice(l.device), "cudaSetDevice");
        d.launched = true;
        d.seq = seq;
        d.pending = Some((a, seq, n_tok * n_used));
        queued.and(back)
    }
}

/// The JOIN op's part, after its last kernel on `main`: the copy of the guess [`ahead`] made for
/// this layer, on the prefetch stream once `main` is past the layer's experts (so it does not
/// share the host's memory with the running GPU's own reads of its missed experts, and the
/// peer's copies for the layer are done: `main` waited for them), and an event.
///
/// # Safety
/// `main` is a live stream of the current device, `l`'s, which ran the FORK op of this pair with
/// the prefetch; the stacks' host parts are mapped for the peer.
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) unsafe fn join(
    main: *mut c_void,
    l: super::Link,
) -> Result<(), String> {
    let mut devs = DEVS.lock().unwrap();
    let Some(d) = devs.iter_mut().find(|d| d.device == l.device) else {
        return Ok(());
    };
    let Some((a, seq, n_guess)) = d.pending.take() else {
        return Ok(());
    };
    let layout = d.layout;
    let at = |off: usize| (d.buf + off) as *mut u8;
    let [gate, down] = a.shape.nb2.map(|b| b as i64);
    let c = PrefetchParams {
        // Rank-major: the first slots go to every token's likeliest experts.
        guess: at(layout.guess()) as *const i32,
        n_guess: n_guess as i32,
        vram_below: a.shape.vram_below,
        n_w: a.shape.n_w as i32,
        cap: SLOTS as i32,
        slots: at(layout.slot_at(layout.half(a.half))),
        slot: layout.slot as i64,
        n_seg: 3,
        src: a.shape.stacks.map(|p| p as *const u8),
        nb2: [gate, gate, down],
        off: [0, gate, 2 * gate],
        stamps: at(layout.stamps(a.half)) as *mut u64,
        pf_ids: at(layout.pf_ids(a.half)) as *mut i32,
        seq,
        progress: at(layout.progress()) as *const u64,
        count: at(layout.count() + 32) as *mut u64,
        timing: at(layout.timing(a.layer)) as *mut u64,
        ctas: CTAS as i32,
    };
    let (stream, event, joined) = (
        d.stream as *mut c_void,
        d.event as *mut c_void,
        d.joined as *mut c_void,
    );
    // SAFETY: live handles (`joined` is the current device's); the kernel goes to the prefetch
    // stream with the peer current, and the current device is put back whatever happens. The
    // peer reaches the buffer (this device's memory) and the stacks' host parts.
    unsafe {
        super::rt(cudaEventRecord(joined, main), "cudaEventRecord")?;
        super::rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
        let queued = (|| {
            super::rt(
                cudaStreamWaitEvent(stream, joined, 0),
                "cudaStreamWaitEvent",
            )?;
            match legion_expert_prefetch_launch(&c, stream) {
                0 => {}
                e => return Err(format!("CUDA error {e} (expert prefetch copy)")),
            }
            super::rt(cudaEventRecord(event, stream), "cudaEventRecord")
        })();
        let back = super::rt(cudaSetDevice(l.device), "cudaSetDevice");
        d.queued += 1;
        queued.and(back)
    }
}

/// The FORK op's [`Forked`] for its JOIN op on this thread.
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) fn forked() -> Option<Forked> {
    FORKED.with(|f| f.get())
}

/// Says what the prefetch did since the last report: of the layers' missed experts, how many a
/// done prefetch slot served (`served`); of the slots the copies took, how many were done in
/// time (`in_time`) and how many served a miss (`used`); how many copies stopped early.
///
/// # Safety
/// `main` is a live stream of `d`'s device, whose buffer holds the counters.
#[cfg(feature = "cuda")]
unsafe fn report(d: &mut Dev, main: *mut c_void) -> Result<(), String> {
    const DEVICE_TO_HOST: c_int = 2;
    let mut now = [0u64; COUNTERS];
    // SAFETY: the counters are inside the buffer: the running GPU's four, read once its stream
    // is done, and the prefetch copy's two 32 bytes on (read while a copy may run: a report).
    unsafe {
        super::rt(cudaStreamSynchronize(main), "cudaStreamSynchronize")?;
        super::rt(
            cudaMemcpy(
                now.as_mut_ptr() as *mut c_void,
                (d.buf + d.layout.count()) as *const c_void,
                COUNTERS * 8,
                DEVICE_TO_HOST,
            ),
            "cudaMemcpy",
        )?;
    }
    let [missed, prefetched, done, layers, taken, stopped, skipped] =
        std::array::from_fn(|i| now[i].wrapping_sub(d.seen[i]));
    d.seen = now;
    let (queued, routers) = (std::mem::take(&mut d.queued), ROUTERS.lock().unwrap().len());
    tracing::info!(
        device = d.device,
        layers,
        queued,
        routers,
        missed,
        prefetched,
        taken,
        done,
        stopped,
        skipped,
        served = prefetched as f64 / missed.max(1) as f64,
        in_time = done as f64 / taken.max(1) as f64,
        used = prefetched as f64 / taken.max(1) as f64,
        "expert peer prefetch"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
