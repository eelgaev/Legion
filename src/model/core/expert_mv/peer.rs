//! Decode steps over an expert cache on a layer split: the running GPU's idle NVLink peer
//! brings a share of the step's missed experts into the running GPU's VRAM (#252).
//!
//! Adapted from Strata-AC922 (MIT; eelgaev/Strata-AC922 commit 94d7556, `partner_fetch`): on a
//! layer split the stages run one after another, so while one GPU runs its stage its peer on the
//! same socket (an IBM AC922's NVLink pairs 0-1 and 2-3) is idle, and so is the peer's own link
//! to host memory. There the peer copies every other missed expert of a verify window from host
//! memory straight into the running GPU's staging buffer (peer writes over the pair's NVLink)
//! while the running GPU copies the others; Strata-AC922 measured 55 GB/s from RAM into one busy
//! GPU alone and 104-106 GB/s with its peer forwarding.
//!
//! On by default where it was measured to help ([`auto`], #522): a layer split over the two GPUs
//! of one NVLink pair on an AC922. `--expert-peer-fetch` forces it on elsewhere (all four GPUs:
//! no gain measured), `--no-expert-peer-fetch` off.
//!
//! Here the split pair of ops of [`super::overlap`] (FORK/JOIN) gets a
//! stream on the peer:
//! - the FORK (SwiGLU) op first records an event on the running GPU's stream, which the peer's
//!   stream waits for (the routing ids are in by then); the peer then runs `stage_copy` twice:
//!   its share of the missed experts' gate and up rows into the running GPU's staging buffer, an
//!   event, their down rows, another event;
//! - alone, the running GPU writes the buffer's slot table (one small launch), computes every
//!   pair whose expert the peer does not stage (cached experts and the other missed ones, read in
//!   place as without the switch), waits for the peer's first event, then computes the peer's
//!   experts from VRAM; the down op the same with the second event;
//! - with `--expert-miss-overlap` too (Strata-AC922's arrangement), the running GPU's side
//!   stream copies the even slots and the peer the odd ones, and the staged products wait for
//!   both.
//!
//! The peer takes [`SHARE`] of the missed experts (ranked in the order of their first pairs),
//! at most [`slots`] of them. It pays only while the peer is idle: the executor says so
//! ([`Idle`]) while one stage runs alone (decode and verify steps on a split, the MTP head);
//! otherwise the pair runs as without the switch. Outputs are the same bits either way: every
//! pair is computed by the same tile code from the same bytes.
//!
//! What the peer touches: the routing ids and the staging buffer are the running GPU's memory
//! (peer access, [`connect`]); the missed experts are the host part of the running GPU's tiered
//! tensors, which the expert cache maps for the peer too
//! ([`crate::residency::experts::ExpertResidency::share_host_parts`]).
//!
//! The peer alone stages into a buffer of the running GPU's own ([`buffer`]), not into a node of
//! the step graph as the side stream does: a graph node moves the graph's memory layout, and with
//! it which layers ggml-cuda fuses the MoE weighted sum in (#472), so the text would move with the
//! switch. Without the node the pair of ops lays out as the plain ops (#522).

#[cfg(feature = "cuda")]
use std::cell::Cell;
#[cfg(feature = "cuda")]
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// What the flags asked for: [`AUTO`], [`FORCED_ON`] or [`FORCED_OFF`].
static FORCED: AtomicU8 = AtomicU8::new(AUTO);
const AUTO: u8 = 0;
const FORCED_ON: u8 = 1;
const FORCED_OFF: u8 = 2;

/// The automatic choice ([`set_auto`]), used while nothing is forced.
static AUTO_ON: AtomicBool = AtomicBool::new(false);

/// Forces whether graphs built from now on let the idle NVLink peer fetch a share of a tiered
/// layer's missed experts (`legion serve --expert-peer-fetch` / `--no-expert-peer-fetch`),
/// whatever [`auto`] decides. Takes effect on devices [`connect`]ed.
pub fn set_enabled(on: bool) {
    FORCED.store(if on { FORCED_ON } else { FORCED_OFF }, Ordering::Relaxed);
}

/// What the flags forced ([`set_enabled`]), or `None` for the automatic choice.
pub fn forced() -> Option<bool> {
    forced_from(FORCED.load(Ordering::Relaxed))
}

fn forced_from(v: u8) -> Option<bool> {
    match v {
        FORCED_ON => Some(true),
        FORCED_OFF => Some(false),
        _ => None,
    }
}

/// Sets the automatic choice (the load's [`auto`] verdict), used while nothing is forced.
pub fn set_auto(on: bool) {
    AUTO_ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on use the peer: as forced, else the automatic choice (off
/// until the load makes one).
pub fn enabled() -> bool {
    forced().unwrap_or_else(|| AUTO_ON.load(Ordering::Relaxed))
}

/// Why [`auto`] leaves the peer fetch off, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoOff {
    /// The GPUs do not read host memory over NVLink 2.0 with ATS (an IBM AC922's V100s): the
    /// only platform it was measured on.
    Platform,
    /// Not a layer split over exactly two GPUs: one GPU has no idle peer, and on all four of an
    /// AC922 it measured no gain (C=1 a tie, C=4/8 1-4% below in the noise; #522).
    Stages,
    /// The two GPUs are not on the same host NUMA node (not an NVLink pair).
    NotPaired,
}

/// The automatic choice (#522): on where it was measured to help, a layer split over exactly two
/// GPUs that read host memory over NVLink 2.0 with ATS and are each other's NVLink peer (the same
/// host NUMA node: an AC922's GPUs 0, 1 or 2, 3). `coherent_nvlink`: the platform is
/// [`crate::residency::Platform::CoherentNvlink`]; `nodes`: each stage GPU's host NUMA node.
pub fn auto(coherent_nvlink: bool, nodes: &[Option<u32>]) -> Result<(), AutoOff> {
    if !coherent_nvlink {
        return Err(AutoOff::Platform);
    }
    if nodes.len() != 2 {
        return Err(AutoOff::Stages);
    }
    if pairs(nodes) != [Some(1), Some(0)] {
        return Err(AutoOff::NotPaired);
    }
    Ok(())
}

/// [`auto`] for an architecture whose decode gains from the peer on every NVLink pair
/// ([`crate::model::core::arch::Arch::peer_decode_pairs`]): any split whose GPUs all pair up
/// (an AC922's 2 or 4), over NVLink 2.0 with ATS.
pub fn auto_paired(coherent_nvlink: bool, nodes: &[Option<u32>]) -> Result<(), AutoOff> {
    if !coherent_nvlink {
        return Err(AutoOff::Platform);
    }
    if nodes.len() < 2 || !nodes.len().is_multiple_of(2) {
        return Err(AutoOff::Stages);
    }
    if pairs(nodes).iter().any(Option::is_none) {
        return Err(AutoOff::NotPaired);
    }
    Ok(())
}

static IDLE: AtomicBool = AtomicBool::new(false);

/// While alive: the ops launched from now on may use the peers (one stage runs alone, so its
/// peer is idle). Dropping it ends that, also on an early return.
pub struct Idle(());

impl Idle {
    /// `idle`: whether the peers are idle (false changes nothing).
    pub fn new(idle: bool) -> Self {
        IDLE.store(idle, Ordering::Relaxed);
        Self(())
    }
}

impl Drop for Idle {
    fn drop(&mut self) {
        IDLE.store(false, Ordering::Relaxed);
    }
}

/// The peer's share of a layer's missed experts by default: `take` of every `every`, in rank
/// order (Strata-AC922's every other one).
pub const SHARE: (usize, usize) = (1, 2);

static SHARE_TAKE: AtomicUsize = AtomicUsize::new(0);
static SHARE_EVERY: AtomicUsize = AtomicUsize::new(0);

/// The peer's share from now on: `take` of every `every` missed experts (0, 0: [`SHARE`]).
/// Without `--expert-miss-overlap` only; with it the peer copies every other slot.
/// `legion serve --expert-peer-share`.
pub fn set_share(take: usize, every: usize) {
    SHARE_TAKE.store(take, Ordering::Relaxed);
    SHARE_EVERY.store(every, Ordering::Relaxed);
}

/// A share of a layer's missed experts for the peer, `TAKE/EVERY` on the command line
/// (`legion serve --expert-peer-share`): `take` of every `every` in rank order,
/// `1 <= take <= every <= 64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Share {
    pub take: usize,
    pub every: usize,
}

impl std::str::FromStr for Share {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let bad = || format!("{s:?}: expected TAKE/EVERY with 1 <= TAKE <= EVERY <= 64, e.g. 2/3");
        let (take, every) = s.split_once('/').ok_or_else(bad)?;
        let take: usize = take.trim().parse().map_err(|_| bad())?;
        let every: usize = every.trim().parse().map_err(|_| bad())?;
        if take == 0 || take > every || every > 64 {
            return Err(bad());
        }
        Ok(Self { take, every })
    }
}

/// `(every, take)` of [`set_share`], valid (`1 <= take <= every <= 64`).
pub(super) fn share() -> (i32, i32) {
    let (take, every) = match (
        SHARE_TAKE.load(Ordering::Relaxed),
        SHARE_EVERY.load(Ordering::Relaxed),
    ) {
        (0, _) | (_, 0) => SHARE,
        s => s,
    };
    let every = every.clamp(1, 64);
    (every as i32, take.clamp(1, every) as i32)
}

/// Staging slots of a pair of ops the peer fills alone: its share of `pairs` slots.
pub(super) fn slots(pairs: usize) -> usize {
    let (every, take) = share();
    (pairs * take as usize).div_ceil(every as usize).max(1)
}

/// CTAs of the peer's copy kernel by default: the peer is idle, so it can take more than the
/// running GPU's side stream does (Strata-AC922's `partner_fetch` runs on 24).
pub const CTAS: usize = 24;

static CTAS_SET: AtomicUsize = AtomicUsize::new(0);

/// The peer's copy CTAs from now on (0: [`CTAS`]); `legion serve --expert-peer-ctas`.
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

/// Each device's NVLink peer among `nodes` (the host NUMA node of each CUDA device, `None` when
/// unknown): the first other device on the same node. On an AC922 (GPUs 0, 1 on node 0, 2, 3 on
/// node 8) that is the NVLink pair.
pub fn pairs(nodes: &[Option<u32>]) -> Vec<Option<usize>> {
    nodes
        .iter()
        .enumerate()
        .map(|(d, n)| {
            let n = (*n)?;
            (0..nodes.len()).find(|&o| o != d && nodes[o] == Some(n))
        })
        .collect()
}

/// Bytes the pairs of ops built so far stage at most ([`reserve`]).
static RESERVED: AtomicUsize = AtomicUsize::new(0);

/// A pair of ops the peer fills alone was built that stages up to `bytes` (slot table and slots):
/// the [`buffer`]s grow to that from their next use, so the graphs built before a step need one
/// allocation between them.
pub(super) fn reserve(bytes: usize) {
    RESERVED.fetch_max(bytes, Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaGetDevice(d: *mut c_int) -> c_int;
    fn cudaSetDevice(d: c_int) -> c_int;
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(p: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(s: *mut c_void) -> c_int;
    fn cudaDeviceCanAccessPeer(can: *mut c_int, d: c_int, peer: c_int) -> c_int;
    fn cudaDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> c_int;
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

/// A device and its peer, with the CUDA handles between them (kept for the process's life).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
pub(super) struct Link {
    device: c_int,
    peer: c_int,
    /// On the peer.
    stream: usize,
    /// On `device`: recorded on its compute stream, waited for by `stream`.
    fork: usize,
    /// On the peer: recorded on `stream` after its gate/up copy and its down copy.
    marks: [usize; 2],
}

/// A link, the expert stacks its peer can read (their addresses, sorted) and the device's staging
/// buffer for the peer alone ([`buffer`]; 0 until first used).
#[cfg(feature = "cuda")]
struct Entry {
    link: Link,
    shared: Vec<usize>,
    buf: usize,
    buf_bytes: usize,
}

#[cfg(feature = "cuda")]
static LINKS: std::sync::Mutex<Vec<Entry>> = std::sync::Mutex::new(Vec::new());

/// Lets CUDA device `peer` fetch missed experts for `device`: peer access from `peer` to
/// `device`'s memory, a stream on `peer` and the events between them. `shared`: the addresses
/// of `device`'s tiered expert stacks whose host parts the caller has mapped for `peer`
/// ([`crate::residency::experts::ExpertResidency::share_host_parts`]); ops over other stacks
/// (prefill's staged ranges) run without the peer. Connecting a device again adds to them.
#[cfg(feature = "cuda")]
pub fn connect(device: usize, peer: usize, mut shared: Vec<usize>) -> Result<(), String> {
    const NON_BLOCKING: c_uint = 1;
    const DISABLE_TIMING: c_uint = 2;
    const ALREADY_ENABLED: c_int = 704; // cudaErrorPeerAccessAlreadyEnabled
    let (d, p) = (device as c_int, peer as c_int);
    if d == p {
        return Err(format!("CUDA{device} cannot be its own peer"));
    }
    let mut links = LINKS.lock().unwrap();
    if let Some(e) = links.iter_mut().find(|e| e.link.device == d) {
        if e.link.peer != p {
            return Err(format!(
                "CUDA{device} already fetches through CUDA{}",
                e.link.peer
            ));
        }
        e.shared.append(&mut shared);
        e.shared.sort_unstable();
        e.shared.dedup();
        return Ok(());
    }
    shared.sort_unstable();
    shared.dedup();
    let mut current = 0;
    // SAFETY: plain runtime calls writing their out-parameters; the current device is put
    // back before returning.
    unsafe {
        rt(cudaGetDevice(&mut current), "cudaGetDevice")?;
        let made = (|| {
            let mut can = 0;
            rt(
                cudaDeviceCanAccessPeer(&mut can, p, d),
                "cudaDeviceCanAccessPeer",
            )?;
            if can == 0 {
                return Err(format!("CUDA{peer} cannot access CUDA{device}'s memory"));
            }
            let mut fork = std::ptr::null_mut();
            rt(cudaSetDevice(d), "cudaSetDevice")?;
            rt(
                cudaEventCreateWithFlags(&mut fork, DISABLE_TIMING),
                "cudaEventCreate",
            )?;
            rt(cudaSetDevice(p), "cudaSetDevice")?;
            match cudaDeviceEnablePeerAccess(d, 0) {
                ALREADY_ENABLED => {
                    cudaGetLastError();
                }
                e => rt(e, "cudaDeviceEnablePeerAccess")?,
            }
            let mut stream = std::ptr::null_mut();
            rt(
                cudaStreamCreateWithFlags(&mut stream, NON_BLOCKING),
                "cudaStreamCreate",
            )?;
            let mut marks = [std::ptr::null_mut(); 2];
            for m in &mut marks {
                rt(
                    cudaEventCreateWithFlags(m, DISABLE_TIMING),
                    "cudaEventCreate",
                )?;
            }
            Ok(Link {
                device: d,
                peer: p,
                stream: stream as usize,
                fork: fork as usize,
                marks: [marks[0] as usize, marks[1] as usize],
            })
        })();
        cudaSetDevice(current);
        links.push(Entry {
            link: made?,
            shared,
            buf: 0,
            buf_bytes: 0,
        });
    }
    Ok(())
}

/// The peer [`connect`]ed to CUDA device `device`, if any.
#[cfg(feature = "cuda")]
pub fn peer_of(device: usize) -> Option<usize> {
    let links = LINKS.lock().unwrap();
    links
        .iter()
        .find(|e| e.link.device == device as c_int)
        .map(|e| e.link.peer as usize)
}

#[cfg(feature = "cuda")]
thread_local! {
    /// The link the last FORK op on this thread used, for its JOIN op (both are launched from
    /// the same thread, one after the other).
    static FORKED: Cell<Option<Link>> = const { Cell::new(None) };
}

/// The FORK op's peer: the current device's link when the switch is on, the peers are idle
/// ([`Idle`]), the device has one and it can read the expert `stacks` (gate, up, down) the op
/// stages from. Kept for the JOIN op ([`forked`]); `None` for a pair of ops without the peer.
#[cfg(feature = "cuda")]
pub(super) fn link(stacks: [*const u8; 3]) -> Result<Option<Link>, String> {
    let link = if enabled() && IDLE.load(Ordering::Relaxed) {
        let mut device = 0;
        // SAFETY: an out-parameter.
        rt(unsafe { cudaGetDevice(&mut device) }, "cudaGetDevice")?;
        LINKS
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.link.device == device)
            .filter(|e| {
                stacks
                    .iter()
                    .all(|&p| e.shared.binary_search(&(p as usize)).is_ok())
            })
            .map(|e| e.link)
    } else {
        None
    };
    FORKED.with(|f| f.set(link));
    Ok(link)
}

/// The staging buffer of `l`'s device for a pair of ops the peer fills alone, at least `bytes`
/// (and what [`reserve`] asked for so far) and 256-byte aligned. Growing it waits for `main`
/// first: the ops queued before may still read the old one (and the peer's copies into it are
/// done once `main` has waited for them, by the end of their JOIN op).
///
/// # Safety
/// `main` is a live stream of `l`'s device, the current device; only ops on `main` use the
/// buffer.
#[cfg(feature = "cuda")]
pub(super) unsafe fn buffer(main: *mut c_void, l: Link, bytes: usize) -> Result<*mut u8, String> {
    let mut links = LINKS.lock().unwrap();
    let e = links
        .iter_mut()
        .find(|e| e.link.device == l.device)
        .ok_or_else(|| format!("CUDA{} has no peer link", l.device))?;
    if e.buf_bytes < bytes {
        let want = bytes.max(RESERVED.load(Ordering::Relaxed));
        // SAFETY: plain runtime calls; the old buffer is freed once nothing queued uses it.
        unsafe {
            if e.buf != 0 {
                rt(cudaStreamSynchronize(main), "cudaStreamSynchronize")?;
                rt(cudaFree(e.buf as *mut c_void), "cudaFree")?;
                (e.buf, e.buf_bytes) = (0, 0);
            }
            let mut p = std::ptr::null_mut();
            rt(cudaMalloc(&mut p, want), "cudaMalloc (expert peer staging)")?;
            (e.buf, e.buf_bytes) = (p as usize, want);
        }
    }
    Ok(e.buf as *mut u8)
}

/// The FORK op's part on `l`'s peer ([`link`]): the peer's stream waits for what `main` has
/// queued, then copies `gate_up`'s and `down`'s slots, recording an event after each.
///
/// # Safety
/// `main` is a live stream of the current device, `l`'s; the pointers in `gate_up` and `down`
/// are memory the peer can reach (the current device's, or host memory mapped for the peer) of
/// the sizes they give, and stay so until `main` has waited for both events.
#[cfg(feature = "cuda")]
pub(super) unsafe fn fork(
    main: *mut c_void,
    l: Link,
    gate_up: &super::overlap::StageParams,
    down: &super::overlap::StageParams,
) -> Result<(), String> {
    let (stream, fork) = (l.stream as *mut c_void, l.fork as *mut c_void);
    // SAFETY: live handles (the fork event is the current device's, the stream and the marks
    // the peer's); kernels go to the peer's stream with the peer current, and the current
    // device is put back whatever happens.
    unsafe {
        rt(cudaEventRecord(fork, main), "cudaEventRecord")?;
        rt(cudaSetDevice(l.peer), "cudaSetDevice")?;
        let queued = (|| {
            rt(cudaStreamWaitEvent(stream, fork, 0), "cudaStreamWaitEvent")?;
            for (s, mark) in [gate_up, down].into_iter().zip(l.marks) {
                super::overlap::stage(s, stream)?;
                rt(
                    cudaEventRecord(mark as *mut c_void, stream),
                    "cudaEventRecord",
                )?;
            }
            Ok(())
        })();
        let back = rt(cudaSetDevice(l.device), "cudaSetDevice");
        queued.and(back)
    }
}

/// The link the last FORK op on this thread got from [`link`] (the JOIN op's).
#[cfg(feature = "cuda")]
pub(super) fn forked() -> Option<Link> {
    FORKED.with(|f| f.get())
}

/// `main` waits for the peer's event `which` (0: its gate and up rows are in, 1: its down
/// rows).
///
/// # Safety
/// `main` is a live stream of the link's device.
#[cfg(feature = "cuda")]
pub(super) unsafe fn wait(main: *mut c_void, l: Link, which: usize) -> Result<(), String> {
    // SAFETY: live handles; a stream may wait for another device's event.
    rt(
        unsafe { cudaStreamWaitEvent(main, l.marks[which] as *mut c_void, 0) },
        "cudaStreamWaitEvent",
    )
}

#[cfg(test)]
mod tests;
