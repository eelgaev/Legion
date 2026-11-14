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
//!   for the layer after it;
//! - per half the guess's ids and the slots they got; the ids left for the peer's copy inside
//!   the layer ("rest"); the layer's input and the guess's logits.
//!
//! A FORK op of the peer alone ([`fork`]): the running GPU's stream waits for the prefetch in
//! flight, then one small kernel (`stage_prepare`) matches the layer's ids against the guess the
//! peer copied for it and writes the slot table (an expert the guess holds: its prefetch slot;
//! the others as before, the peer's share ranked over the ids left), the ids left and the copy
//! of the layer's input. The peer then copies its share of the experts left as before and,
//! behind that ([`ahead`]), guesses and copies for the layer whose FORK followed this one's last
//! time (a layer says where its router and table are when its graph is built, [`register`]).
//!
//! Outputs are the same bits: a prefetch slot holds the host bytes of the very tiered id the
//! layer asks for (ids are compared exactly, and a tiered tensor's host part never changes), and
//! every pair is computed by the same tile code as a peer-staged one. A wrong guess costs the
//! idle peer a copy; the layer then runs as without the prefetch.
//!
//! Opt-in (`legion serve --expert-peer-prefetch`), and only where the peer fetch is on by the
//! automatic choice for a layer split over one NVLink pair ([`super::auto`]: two V100s of an
//! AC922). Architectures with their own prefetch (`peer_decode_pairs`) do not use it. Measured
//! there it loses: the peer's copies of the guess sit in the step's path (the next layer waits
//! for them, and its own peer copies queue behind them), so one user decodes 10% slower with 4
//! slots and 18% with 8, while the copies serve 33-49% of the misses.

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
/// uncached ones first). 8 copied ~5 a layer of which about half were used, and the next layer
/// waited for them.
pub const SLOTS: usize = 4;

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

    /// The guess in prefetch half `half`: its tiered ids.
    pub fn guess(&self, half: usize) -> usize {
        self.lists() + (1 + 2 * half) * LIST
    }

    /// The slot table of the guess in prefetch half `half`.
    pub fn guess_slots(&self, half: usize) -> usize {
        self.guess(half) + LIST
    }

    /// The counters (four u64).
    pub fn count(&self) -> usize {
        self.lists() + 5 * LIST
    }

    /// The copy of the layer's input ([`MAX_TOKENS`] columns of [`MAX_EMBD`] floats).
    pub fn x(&self) -> usize {
        self.count() + 256
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
    /// The guess the peer copied for this layer: its ids' count and the prefetch half.
    pub guess: Option<(usize, usize)>,
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
    /// The layer the last prefetch was for, the shape it was copied with, the guess's ids and
    /// the half.
    pf: Option<(usize, Shape, usize, usize)>,
    /// The half the layer last visited reads (the other one is free).
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
        // A guess copied for this layer, with the id bounds it has now.
        let guess = match pf {
            Some((p, s, n, half)) if p == ix && s == shape => Some((n, half)),
            _ => None,
        };
        if let Some((_, half)) = guess {
            self.half = half;
        }
        Some(Visit { layer: ix, guess })
    }

    /// The prefetch to queue while `layer` computes: a guess of `n` ids for the layer after it,
    /// into the half `layer` does not read. `None` without a next layer.
    pub fn ahead(&mut self, layer: usize, n: usize) -> Option<Ahead> {
        let nx = self.layers[layer].next.filter(|&nx| nx != layer)?;
        let shape = self.layers[nx].shape;
        if n == 0 {
            return None;
        }
        let half = 1 - self.half;
        self.pf = Some((nx, shape, n, half));
        Some(Ahead {
            layer: nx,
            shape,
            half,
        })
    }

    /// The layer after `layer`, as [`Self::ahead`] would take it.
    pub fn next(&self, layer: usize) -> Option<Shape> {
        let nx = self.layers[layer].next.filter(|&nx| nx != layer)?;
        Some(self.layers[nx].shape)
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
    fn cudaMemsetAsync(p: *mut c_void, v: c_int, bytes: usize, s: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
    fn legion_expert_prepare_launch(p: *const PrepareParams, stream: *mut c_void) -> c_int;
    fn legion_expert_guess_launch(p: *const GuessParams, stream: *mut c_void) -> c_int;
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
    guess: *const i32,
    guess_slot: *const i32,
    n_guess: i32,
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

#[cfg(feature = "cuda")]
struct Dev {
    device: c_int,
    buf: usize,
    layout: Layout,
    /// On the peer: recorded on its stream after each prefetch.
    event: usize,
    /// Whether a prefetch was ever queued (the event recorded).
    launched: bool,
    chain: Chain,
    visits: u64,
    /// Prefetches queued since the last report.
    queued: u64,
    seen: [u64; 4],
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
/// `main` waits for the prefetch in flight and runs `stage_prepare`. `None`: this pair of ops
/// runs without the prefetch (off, or a step it does not take), on [`super::buffer`] as before.
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
            const DISABLE_TIMING: c_uint = 2;
            let mut event = std::ptr::null_mut();
            // SAFETY: plain runtime calls; the event is the peer's, the current device is put back.
            unsafe {
                super::rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
                let made = super::rt(
                    cudaEventCreateWithFlags(&mut event, DISABLE_TIMING),
                    "cudaEventCreate",
                );
                super::rt(cudaSetDevice(l.device), "cudaSetDevice")?;
                made?;
            }
            devs.push(Dev {
                device: l.device,
                buf: 0,
                layout: Layout::default(),
                event: event as usize,
                launched: false,
                chain: Chain::default(),
                visits: 0,
                queued: 0,
                seen: [0; 4],
            });
            devs.last_mut().unwrap()
        }
    };
    if d.launched {
        // The prefetch in flight reads the lists and writes the buffer: done before this op
        // rewrites either (or the buffer grows).
        // SAFETY: live handles; a stream may wait for another device's event.
        super::rt(
            unsafe { cudaStreamWaitEvent(main, d.event as *mut c_void, 0) },
            "cudaStreamWaitEvent",
        )?;
    }
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
        // The pair runs as before, over the same buffer: nothing here is valid after it.
        d.chain.reset();
        return Ok(None);
    }
    // SAFETY: as the caller promises.
    let buf = unsafe { super::buffer(main, l, layout.bytes())? };
    let at = |off: usize| buf.wrapping_add(off);
    if buf as usize != d.buf || layout != d.layout {
        (d.buf, d.layout, d.seen) = (buf as usize, layout, [0; 4]);
        d.chain.reset();
        // SAFETY: the counters are inside the buffer.
        super::rt(
            unsafe { cudaMemsetAsync(at(layout.count()) as *mut c_void, 0, 32, main) },
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
    let (guess_ids, guess_slot, n_guess, pf_base) = match v.guess {
        Some((n, half)) => (
            at(layout.guess(half)),
            at(layout.guess_slots(half)),
            n,
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
        guess: guess_ids as *const i32,
        guess_slot: guess_slot as *const i32,
        n_guess: n_guess as i32,
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

/// The FORK op's last part on the peer, after [`super::fork`]: behind the layer's own copies,
/// the guess for the layer after it, the guessed host experts' copy into the free half (gate,
/// up and down rows), and an event.
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
    let Some(a) = d.chain.ahead(f.layer, n_tok * n_used) else {
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
        guess: at(layout.guess(a.half)) as *mut i32,
    };
    let [gate, down] = a.shape.nb2.map(|b| b as i64);
    let s = super::super::overlap::StageParams {
        // Rank-major: stage_copy's first-come slots go to every token's likeliest experts.
        ids: g.guess,
        ids_nb1: (n_tok * 4) as i64,
        n_used: n_tok as i32,
        n_tok: n_used as i32,
        vram_below: a.shape.vram_below,
        n_w: a.shape.n_w as i32,
        // stage_copy writes its slot table at `stage` and the slots `hdr` behind it.
        stage: at(layout.guess_slots(a.half)),
        hdr: layout.slot_at(layout.half(a.half)) as i64 - layout.guess_slots(a.half) as i64,
        slot: layout.slot as i64,
        cap: SLOTS as i32,
        table: 1,
        n_seg: 3,
        src: a.shape.stacks.map(|p| p as *const u8),
        nb2: [gate, gate, down],
        off: [0, gate, 2 * gate],
        ctas: super::ctas(),
        t_every: 1,
        t_take: 1,
        c_every: 1,
        c_take: 1,
        c_flip: 0,
    };
    let (stream, event) = (l.stream as *mut c_void, d.event as *mut c_void);
    // SAFETY: live handles; the kernels go to the peer's stream with the peer current, and the
    // current device is put back whatever happens. The peer reaches the buffer, the router and
    // the table (this device's memory) and the stacks' host parts (mapped for it when it was
    // connected).
    unsafe {
        super::rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
        let queued = match legion_expert_guess_launch(&g, stream) {
            0 => super::super::overlap::stage(&s, stream),
            e => Err(format!("CUDA error {e} (expert prefetch guess)")),
        }
        .and_then(|()| super::rt(cudaEventRecord(event, stream), "cudaEventRecord"));
        let back = super::rt(cudaSetDevice(l.device), "cudaSetDevice");
        d.launched = true;
        d.queued += 1;
        queued.and(back)
    }
}

/// The FORK op's [`Forked`] for its JOIN op on this thread.
#[cfg(feature = "cuda")]
pub(in crate::model::core::expert_mv) fn forked() -> Option<Forked> {
    FORKED.with(|f| f.get())
}

/// Says what the prefetch did since the last report: of the layers' missed experts, how many
/// the peer had copied early.
///
/// # Safety
/// `main` is a live stream of `d`'s device, whose buffer holds the counters.
#[cfg(feature = "cuda")]
unsafe fn report(d: &mut Dev, main: *mut c_void) -> Result<(), String> {
    const DEVICE_TO_HOST: c_int = 2;
    let mut now = [0u64; 4];
    // SAFETY: 32 bytes inside the buffer, read once the stream's kernels are done.
    unsafe {
        super::rt(cudaStreamSynchronize(main), "cudaStreamSynchronize")?;
        super::rt(
            cudaMemcpy(
                now.as_mut_ptr() as *mut c_void,
                (d.buf + d.layout.count()) as *const c_void,
                32,
                DEVICE_TO_HOST,
            ),
            "cudaMemcpy",
        )?;
    }
    let [missed, prefetched, copied, layers] =
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
        copied,
        served = prefetched as f64 / missed.max(1) as f64,
        used = prefetched as f64 / copied.max(1) as f64,
        "expert peer prefetch"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
