//! Decode steps over an expert cache: the missed experts brought over the link on a side stream
//! while the cached experts are computed (#253).
//!
//! Adapted from Strata-AC922 (MIT; eelgaev/Strata-AC922 commit 756ff34, the verifier's
//! `fetch_s_`): there the copy kernel that brings a layer's missed experts over the link runs on
//! a stream of its own, forked once the layer's routing is in and joined before the missed
//! experts' products, on few CTAs so that the cached experts computed meanwhile keep the SMs.
//!
//! Legion reads missed experts in place (host memory over NVLink2 / ATS, #236), in the same
//! kernels as the cached ones, one CTA per tile of a (token, expert) pair. On a V100 a layer's
//! missed experts alone take most of the layer's expert time (link-bound) and the cached ones
//! add a good part on top, though they could hide under the link time. Splitting the same tile
//! kernels over two streams does not hide them: the in-place host tiles keep far more loads in
//! flight than the link needs, and those slow the VRAM tiles beside them (measured in
//! `tests/expert_mv_overlap_cuda.rs`). A copy kernel on a few CTAs fills the link with far
//! fewer loads in flight.
//!
//! So here a layer's SwiGLU (gate/up) op and its down op become a pair ([`glu_down_tiered`]):
//! - the SwiGLU op quantizes the activations, forks a side stream (an event) and launches
//!   `stage_copy` there twice on [`ctas`] CTAs: the step's missed experts' gate and up rows into a
//!   VRAM staging buffer (each expert once, at most [`stage_pairs`] of them), an event, then their
//!   down rows, another event. On the compute stream it computes the cached experts, waits for the
//!   first event and computes the staged experts' SwiGLU from VRAM;
//! - the down op computes the cached experts' products, waits for the second event, then the
//!   staged experts' products.
//!
//! Every pair is computed by the same tile code, from the same bytes, as without the split, so
//! outputs are the same bits. The staging buffer is a source of both ops, so ggml's allocator
//! keeps it alive until the down op; everything else the side stream reads (the ids, the weights)
//! outlives the pair too, and the compute stream has waited for the side stream by the end of
//! the down op.
//!
//! The same pair of ops lets the running GPU's idle NVLink peer fill the staging buffer
//! ([`super::peer`], `--expert-peer-fetch`): the fourth parameter says what fills it
//! ([`MODE_SIDE`], [`MODE_PEER`] or both), the fifth gives a slot's bytes. The peer alone fills
//! a buffer outside the graph ([`super::peer::buffer`]), so the pair has no staging node then.

#[cfg(feature = "cuda")]
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

/// The staging buffer's node: allocates it, computes nothing (the FORK op fills it).
static STAGE_BUFFER: ExternalOp = ExternalOp {
    name: "expert_mv_stage",
    cpu: cpu_nothing,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_nothing),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

unsafe fn cpu_nothing(_: &ffi::ggml_tensor, _: usize, _: usize) {}

#[cfg(feature = "cuda")]
unsafe fn cuda_nothing(_: &ffi::ggml_tensor, _: *mut c_void) -> Result<(), String> {
    Ok(())
}

/// An op's role in the split (its second parameter): none.
pub(super) const NONE: i32 = 0;
/// The SwiGLU op that forks the side stream.
pub(super) const FORK: i32 = 1;
/// The down op that joins it.
pub(super) const JOIN: i32 = 2;

/// What fills a split pair's staging buffer (its fourth parameter, bits): a side stream of the
/// running GPU (this module, `--expert-miss-overlap`) ...
pub(super) const MODE_SIDE: i32 = 1;
/// ... and/or the running GPU's idle NVLink peer ([`super::peer`], `--expert-peer-fetch`).
pub(super) const MODE_PEER: i32 = 2;

/// What the flags asked for: 0 nothing, 1 on, 2 off.
static FORCED: AtomicU8 = AtomicU8::new(0);
/// The automatic choice ([`set_auto`]), used while nothing is forced.
static AUTO_ON: AtomicBool = AtomicBool::new(false);

/// Forces whether graphs built from now on stage a tiered layer's missed experts on a side
/// stream (`legion serve --expert-miss-overlap` / `--no-expert-miss-overlap`).
pub fn set_enabled(on: bool) {
    FORCED.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Sets the automatic choice, used while nothing is forced (off until a load makes one).
pub fn set_auto(on: bool) {
    AUTO_ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on stage on a side stream: as forced, else the automatic choice.
pub fn enabled() -> bool {
    match FORCED.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => AUTO_ON.load(Ordering::Relaxed),
    }
}

/// CTAs of the copy kernel by default: 8 x 256 threads with 8 16-byte loads in flight each fill
/// a V100's NVLink2 link (68 GB/s) on their own, and the fewer CTAs, the less they slow the
/// cached experts beside them (8 measured best of 8, 16 and 32).
pub const CTAS: usize = 8;

static CTAS_SET: AtomicUsize = AtomicUsize::new(0);

/// The copy kernel's CTAs from now on (0: [`CTAS`]); for tuning.
pub fn set_ctas(n: usize) {
    CTAS_SET.store(n, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn ctas() -> i32 {
    match CTAS_SET.load(Ordering::Relaxed) {
        0 => CTAS as i32,
        n => n.min(1 << 16) as i32,
    }
}

/// Missed experts a layer's step stages by default (more are read in place, after the staged
/// ones): the staging buffer takes this many experts' bytes of VRAM (Qwen3.8-Flash-Next: 3.07 MB
/// each).
pub const STAGE_PAIRS: usize = 16;

static STAGE_PAIRS_SET: AtomicUsize = AtomicUsize::new(0);

/// The staged experts per layer and step in graphs built from now on (0: [`STAGE_PAIRS`]); for
/// tuning.
pub fn set_stage_pairs(n: usize) {
    STAGE_PAIRS_SET.store(n, Ordering::Relaxed);
}

fn stage_pairs() -> usize {
    match STAGE_PAIRS_SET.load(Ordering::Relaxed) {
        0 => STAGE_PAIRS,
        n => n,
    }
}

/// Most pairs (tokens x experts per token) of a step `stage_copy` takes.
const MAX_PAIRS: i64 = 512;

/// Bytes of the staging buffer's slot table: an i32 per pair, rounded up so that the slots start
/// 256-byte aligned.
pub(super) fn stage_header(n_pairs: usize) -> usize {
    (n_pairs * 4).next_multiple_of(256)
}

/// `down(silu(gate x) * up x)` per (token, expert) pair over tiered expert tensors, as
/// [`super::glu_tiered`] followed by [`super::mul_mat_id_tiered`] compute it, with the missed
/// experts staged on a side stream (see the module docs). `None` when the split is off, the
/// layer has no cache slots, or either op would not run on Legion's matvec.
pub fn glu_down_tiered<'a>(
    ctx: &'a Context,
    w: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
) -> Option<Tensor<'a>> {
    let fits = super::glu_fits(w[0], w[1], x, ids);
    glu_down_split(ctx, w, x, ids, vram_slots, 0.0, (fits, false))
}

/// [`glu_down_tiered`] with the SwiGLU clamped at `limit` ([`super::glu_tiered_clamped`]), for
/// gate and up rows down to [`super::glu_fits_narrow`]'s (GLM-5.3-Flash), in steps of up to
/// `max_tokens` tokens ([`super::fits_glm_up_to`]).
pub fn glu_down_tiered_clamped<'a>(
    ctx: &'a Context,
    w: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    (vram_slots, limit): (usize, f32),
    max_tokens: i64,
) -> Option<Tensor<'a>> {
    let fits = super::glu_fits_narrow_up_to(w[0], w[1], x, ids, max_tokens);
    glu_down_split(ctx, w, x, ids, vram_slots, limit, (fits, true))
}

fn glu_down_split<'a>(
    ctx: &'a Context,
    [gate, up, down]: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    limit: f32,
    (glu_fits, glm): (bool, bool),
) -> Option<Tensor<'a>> {
    let mode =
        if enabled() { MODE_SIDE } else { 0 } | if super::peer::enabled() { MODE_PEER } else { 0 };
    let ok = mode != 0
        && vram_slots > 0
        && glu_fits
        && super::stack_fits_as(down, glm)
        && down.ne()[0] == gate.ne()[1]
        && down.ne()[2] == gate.ne()[2]
        && ids.ne()[0] * ids.ne()[1] <= MAX_PAIRS;
    ok.then(|| glu_down_node(ctx, [gate, up, down], x, ids, (vram_slots, limit), mode))
}

/// [`glu_down_tiered`]'s two nodes for `mode` ([`MODE_SIDE`], [`MODE_PEER`] or both), whatever
/// the switches (CPU tests build them directly).
pub(super) fn glu_down_node<'a>(
    ctx: &'a Context,
    [gate, up, down]: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    (vram_slots, limit): (usize, f32),
    mode: i32,
) -> Tensor<'a> {
    let n_pairs = (ids.ne()[0] * ids.ne()[1]) as usize;
    // The peer alone stages only its share of the missed experts (the running GPU reads the
    // others in place).
    let staged = if mode & MODE_SIDE != 0 {
        stage_pairs()
    } else {
        super::peer::slots(stage_pairs())
    };
    let slots = staged.min(n_pairs).max(1);
    // A slot: the expert's gate, up and down bytes, in that order.
    let slot = gate.nb()[2] + up.nb()[2] + down.nb()[2];
    // The output of a node that does nothing, not a leaf: ggml's allocator keeps every leaf for
    // the whole graph (a stage of 24 layers held 24 buffers, ~1.2 GB per GPU at 3 tokens), a node
    // only from itself to its last consumer (the JOIN node), so the layers share the bytes. The
    // side stream writes it after the FORK node is reached and the compute stream waits for that
    // before the JOIN node ends, so nothing else uses it meanwhile.
    let bytes = stage_header(n_pairs) + slots * slot;
    // The peer alone stages into its device's own buffer (see `super::peer`): no node, so the
    // pair lays out as the plain ops do (#522).
    let stage = if mode & MODE_SIDE != 0 {
        Some(ctx.external(
            &STAGE_BUFFER,
            types::I8,
            [bytes as i64, 1, 1, 1],
            &[gate, ids],
        ))
    } else {
        super::peer::reserve(
            stage_header(MAX_PAIRS as usize) + super::peer::slots(stage_pairs()) * slot,
        );
        None
    };
    let slot = i32::try_from(slot).expect("an expert's bytes fit an i32");
    let params = [
        vram_slots as i32,
        FORK,
        slots as i32,
        mode,
        slot,
        limit.to_bits() as i32,
    ];
    let glu_xq = super::q8_1_scratch(ctx, gate, x, ids);
    let more: Vec<Tensor<'a>> = stage.into_iter().chain([down]).collect();
    let act = super::glu_node(ctx, [gate, up], x, ids, glu_xq, &more, &params);
    let xq = super::q8_1_scratch(ctx, down, act, ids);
    let (rows, n_used, n_tok) = (down.ne()[1], ids.ne()[0], ids.ne()[1]);
    let srcs: Vec<Tensor<'a>> = [down, act, ids, xq].into_iter().chain(stage).collect();
    ctx.external_with(
        &super::EXPERT_MV,
        types::F32,
        [rows, n_used, n_tok, 1],
        &srcs,
        &[vram_slots as i32, JOIN, slots as i32, mode, slot],
    )
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaGetDevice(d: *mut c_int) -> c_int;
    fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int;
    fn cudaStreamCreateWithPriority(s: *mut *mut c_void, flags: c_uint, priority: c_int) -> c_int;
    fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
    fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: c_uint) -> c_int;
    fn cudaGetErrorString(e: c_int) -> *const c_char;
}

#[cfg(feature = "cuda")]
fn rt(e: c_int, what: &str) -> Result<(), String> {
    if e == 0 {
        return Ok(());
    }
    // SAFETY: a static NUL-terminated string.
    let msg = unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(e)) }.to_string_lossy();
    Err(format!("{what}: {msg}"))
}

static HIGH_PRIORITY: AtomicBool = AtomicBool::new(true);

/// Whether side streams created from now on take the device's highest priority (the default)
/// or the lowest, which is also the compute stream's; for tuning.
pub fn set_high_priority(on: bool) {
    HIGH_PRIORITY.store(on, Ordering::Relaxed);
}

/// A compute stream's side stream and the events between them (CUDA handles, kept for the
/// process's life).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct Side {
    device: c_int,
    main: usize,
    high: bool,
    stream: usize,
    /// Recorded on `main` and waited for by the side stream: [`fork`].
    fork: usize,
    /// Recorded on the side stream ([`mark`]) and waited for by `main` ([`wait`]).
    marks: [usize; 3],
}

#[cfg(feature = "cuda")]
static SIDES: std::sync::Mutex<Vec<Side>> = std::sync::Mutex::new(Vec::new());

/// The side stream of the compute stream `main` on the current device, made on first use.
#[cfg(feature = "cuda")]
fn side_of(main: *mut c_void) -> Result<Side, String> {
    const NON_BLOCKING: c_uint = 1;
    const DISABLE_TIMING: c_uint = 2;
    let high = HIGH_PRIORITY.load(Ordering::Relaxed);
    let mut device = 0;
    // SAFETY: plain runtime calls writing their out-parameters.
    unsafe {
        rt(cudaGetDevice(&mut device), "cudaGetDevice")?;
        let mut sides = SIDES.lock().unwrap();
        let key = |s: &&Side| s.device == device && s.main == main as usize && s.high == high;
        if let Some(s) = sides.iter().find(key) {
            return Ok(*s);
        }
        let (mut least, mut greatest) = (0, 0);
        rt(
            cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest),
            "cudaDeviceGetStreamPriorityRange",
        )?;
        let mut stream = std::ptr::null_mut();
        rt(
            cudaStreamCreateWithPriority(
                &mut stream,
                NON_BLOCKING,
                if high { greatest } else { least },
            ),
            "cudaStreamCreateWithPriority",
        )?;
        let mut events = [std::ptr::null_mut(); 4];
        for e in &mut events {
            rt(
                cudaEventCreateWithFlags(e, DISABLE_TIMING),
                "cudaEventCreate",
            )?;
        }
        let s = Side {
            device,
            main: main as usize,
            high,
            stream: stream as usize,
            fork: events[0] as usize,
            marks: [events[1] as usize, events[2] as usize, events[3] as usize],
        };
        sides.push(s);
        Ok(s)
    }
}

/// Forks `main`'s side stream: it waits for what `main` has queued so far. Returns it.
///
/// # Safety
/// `main` is a live stream of the current device.
#[cfg(feature = "cuda")]
pub(super) unsafe fn fork(main: *mut c_void) -> Result<*mut c_void, String> {
    let s = side_of(main)?;
    let (stream, fork) = (s.stream as *mut c_void, s.fork as *mut c_void);
    // SAFETY: live handles of the current device.
    unsafe {
        rt(cudaEventRecord(fork, main), "cudaEventRecord")?;
        rt(cudaStreamWaitEvent(stream, fork, 0), "cudaStreamWaitEvent")?;
    }
    Ok(stream)
}

/// The side stream's event `which` (0: the staged gate and up rows are in, 1: the down rows; 2:
/// an op's PTX VRAM pairs are done, [`super::ptx`]'s side launches):
/// recorded on the side stream after what it has queued so far.
///
/// # Safety
/// As [`fork`].
#[cfg(feature = "cuda")]
pub(super) unsafe fn mark(main: *mut c_void, which: usize) -> Result<(), String> {
    let s = side_of(main)?;
    // SAFETY: live handles of the current device.
    unsafe {
        rt(
            cudaEventRecord(s.marks[which] as *mut c_void, s.stream as *mut c_void),
            "cudaEventRecord",
        )
    }
}

/// `main` waits for the side stream's event `which` ([`mark`]).
///
/// # Safety
/// As [`fork`].
#[cfg(feature = "cuda")]
pub(super) unsafe fn wait(main: *mut c_void, which: usize) -> Result<(), String> {
    let s = side_of(main)?;
    // SAFETY: live handles of the current device.
    unsafe {
        rt(
            cudaStreamWaitEvent(main, s.marks[which] as *mut c_void, 0),
            "cudaStreamWaitEvent",
        )
    }
}

/// `stage_copy`'s parameters (`legion_expert_stage` in `src/cuda/expert_mv/`).
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct StageParams {
    pub ids: *const i32,
    pub ids_nb1: i64,
    pub n_used: i32,
    pub n_tok: i32,
    pub vram_below: i32,
    pub n_w: i32,
    pub stage: *mut u8,
    pub hdr: i64,
    pub slot: i64,
    pub cap: i32,
    pub table: i32,
    pub n_seg: i32,
    pub src: [*const u8; 3],
    pub nb2: [i64; 3],
    pub off: [i64; 3],
    pub ctas: i32,
    /// Which experts take a slot: rank `r` (the step's missed experts in the order of their
    /// first pairs) when `r % t_every < t_take` (see `stage_copy`).
    pub t_every: i32,
    pub t_take: i32,
    /// Which slots this launch copies: `(slot % c_every < c_take) != c_flip`.
    pub c_every: i32,
    pub c_take: i32,
    pub c_flip: i32,
}

#[cfg(feature = "cuda")]
impl StageParams {
    /// Every missed expert takes a slot (while slots last), and this launch copies them all.
    pub(crate) const ALL: (i32, i32, i32, i32, i32) = (1, 1, 1, 1, 0);

    /// `self` taking slots by `rule` (`(t_every, t_take, c_every, c_take, c_flip)`).
    pub(super) fn rule(
        self,
        (t_every, t_take, c_every, c_take, c_flip): (i32, i32, i32, i32, i32),
    ) -> Self {
        Self {
            t_every,
            t_take,
            c_every,
            c_take,
            c_flip,
            ..self
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_expert_stage_launch(s: *const StageParams, stream: *mut c_void) -> c_int;
}

/// Launches `stage_copy` on `stream`.
///
/// # Safety
/// The pointers in `s` are live device-readable memory of the sizes it gives, `stream` a live
/// stream of the current device.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn stage(s: &StageParams, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as the caller promises.
    rt(
        unsafe { legion_expert_stage_launch(s, stream) },
        "expert stage copy",
    )
}

#[cfg(test)]
mod tests;
