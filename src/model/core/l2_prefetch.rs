//! A weight's first bytes read into L2 ahead of the matmul that reads it, as an external op that
//! writes nothing (#689 round 11; kernel in `src/cuda/l2_prefetch.cu`).
//!
//! In a dense gated DeltaNet layer's one-user verify step on a GB10 (Qwen3.8-27B NVFP4, DFlash 2)
//! the qkv projection is followed by ~60-70 us of small kernels (conv, alpha/beta, delta prep,
//! the delta rule, the gated norm's norm) with DRAM mostly idle, and the z projection after them
//! ran ~20 us over the time its bytes take; out_proj takes that cost instead when z goes first.
//! [`ahead`] after the qkv projection reads z's first rows into L2 meanwhile. Nothing reads the
//! node's result and no tensor changes, so every output keeps its bits.
//!
//! On the compute stream it saves z what it costs elsewhere (round 11): the kernel holds the
//! stream until its prefetches are served (4 MiB ~10 us, 8 MiB ~24 us, z -8 / -18 us); issued
//! before its wait it takes the qkv projection's tail bandwidth (qkv +9 us, z -7 us); issued
//! from the conv kernel it lengthens the conv by as much.
//!
//! So it runs on a side stream (#689 round 12), forked from the compute stream right after the
//! qkv projection (an event recorded on the compute stream, waited for on the side stream): no
//! kernel of the compute stream waits for it, and a [`join`] node after the layer's output
//! projection makes the compute stream wait for the side stream's done event. A stage graph
//! captures the pair as a forked branch of the same capture (both ops pure, every op between
//! them pure in small steps), so the capture stays one graph; a direct run does the same on the
//! two streams. One CTA issues it paced, 128 KiB every [`GDN_Z_PACE_NS`]: an unpaced burst
//! queues the small kernels' own reads behind it (measured, the conv ran as long as the
//! prefetch). Measured on a GB10 (8 MiB over ~79 us): z 154 -> 134 us (its back-to-back
//! speed), the small kernels +10 us, about -0.4 ms a verify step.
//!
//! Where it applies ([`takes`]: CUDA, compute capability 12.x, stage graphs on, dense models,
//! steps of up to [`MAX_TOKENS`] tokens) it is on by default ([`GDN_Z_MIB`]); probes, read once:
//! `LEGION_DEBUG_GDN_PF=<MiB>` (0: off), `LEGION_DEBUG_GDN_PF_FORK=0` (on the compute stream,
//! unpaced), `LEGION_DEBUG_GDN_PF_PACE=<ns>` (0: unpaced), `LEGION_DEBUG_GDN_PF_FAR=1` (as many
//! bytes of the LM head instead, rows nobody reads before they leave L2: DRAM traffic without
//! the data; z does not move, so the cost is z's data, not a DRAM ramp).
//!
//! Round 13: where it applies (an FP8 z, [`takes`]), the z projection's GEMM itself runs on the
//! side stream instead, forked after the qkv projection, beside the small kernels, and a
//! [`side_join`] before the gated norm brings it back ([`z_side`], [`on_side`]); the z prefetch
//! is then left out (z starts right after qkv). The stage graph gives every kernel of such a
//! capture but the side stream's the greatest priority (`stage_graph::prio`), so the small
//! kernels' blocks go ahead of z's queued ones instead of waiting for z's last wave. Measured
//! on a GB10: z 134 -> 166 us beside the small kernels, out_proj 135 -> 152 us (it now pays the
//! cold start), the layer from qkv's end to out_proj's end 345 -> 330 us, -0.5 to -1.0 ms a
//! verify step. `LEGION_DEBUG_GDN_Z_SIDE=0` goes back to the paced z prefetch (an A/B probe).

use std::sync::OnceLock;

#[cfg(any(test, feature = "cuda"))]
use crate::ggml::external_params;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;
pub mod tp;

static L2_PREFETCH: ExternalOp = ExternalOp {
    name: "l2_prefetch",
    cpu: cpu_l2_prefetch,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_l2_prefetch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static L2_JOIN: ExternalOp = ExternalOp {
    name: "l2_join",
    cpu: cpu_l2_prefetch,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_l2_join),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SIDE_JOIN: ExternalOp = ExternalOp {
    name: "side_join",
    cpu: cpu_l2_prefetch,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_side_join),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Bytes of one thread's bulk prefetch; offsets and sizes are whole chunks.
pub const CHUNK: usize = 4096;

/// Most tokens a step may have for [`takes`]: one user's verify step (a DFlash 2 block is 8
/// tokens), the case measured. With 64 here, 4 users' steps ran 2% lower total in one det run
/// (within that run's noise, but unmeasured otherwise), so they keep the plain chain.
pub const MAX_TOKENS: i64 = 8;

/// The default size ahead of a DeltaNet layer's z projection, in MiB (0: off): with the pace
/// below, the prefetch spans the ~75 us of small kernels before z.
pub const GDN_Z_MIB: usize = 8;

/// The default pace of a side-stream prefetch: one 128 KiB step (32 threads x 4 KiB) every this
/// many ns (~130 GB/s, about half the DRAM's), next to the small kernels' own reads.
pub const GDN_Z_PACE_NS: u32 = 1000;

/// Bytes ahead of a DeltaNet layer's z projection: `LEGION_DEBUG_GDN_PF` in MiB (0: off), else
/// [`GDN_Z_MIB`]; read once.
pub fn gdn_z_bytes() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| gdn_z_bytes_from(std::env::var("LEGION_DEBUG_GDN_PF").ok().as_deref()))
}

/// [`gdn_z_bytes`] for the switch's value `v`: whole MiB up to 64, else the default.
pub fn gdn_z_bytes_from(v: Option<&str>) -> usize {
    let mib = v
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&m| m <= 64)
        .unwrap_or(GDN_Z_MIB);
    mib << 20
}

/// `LEGION_DEBUG_GDN_PF_FAR=1` (a probe, read once): the LM head's rows instead of z's.
pub fn far() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_FAR").is_ok_and(|v| v.trim() == "1"))
}

/// `LEGION_DEBUG_GDN_PF_LATE=1` (an A/B probe, read once): the prefetch kernel issues its
/// prefetches before it waits for the qkv projection (during its last CTAs) instead of after.
pub fn late() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_LATE").is_ok_and(|v| v.trim() == "1"))
}

/// `LEGION_DEBUG_GDN_PF_CONV=1` (an A/B probe, read once): no prefetch node; the layer's
/// `conv_silu` kernel issues the prefetch at its start (see [`fused::conv_silu_window_pf`]).
///
/// [`fused::conv_silu_window_pf`]: super::fused::conv_silu_window_pf
pub fn in_conv() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_CONV").is_ok_and(|v| v.trim() == "1"))
}

/// The side-stream prefetch's pace: one CTA issues it a block's chunks at a time (`_SHAPE`'s
/// threads x chunk, 128 KiB by default), one step every this many ns, instead of every chunk
/// at once (0). `LEGION_DEBUG_GDN_PF_PACE=<ns>` (an A/B probe, at most 1 ms), else
/// [`GDN_Z_PACE_NS`]; read once.
pub fn pace_ns() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| pace_from(std::env::var("LEGION_DEBUG_GDN_PF_PACE").ok().as_deref()))
}

/// [`pace_ns`] for the switch's value `v`.
pub fn pace_from(v: Option<&str>) -> u32 {
    v.and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&ns| ns <= 1_000_000)
        .unwrap_or(GDN_Z_PACE_NS)
}

/// Whether a DeltaNet layer's z projection itself runs on the side stream, forked right after
/// the qkv projection, beside the small kernels between the two, a [`side_join`] before the
/// gated norm bringing it back (#689 round 13): yes unless `LEGION_DEBUG_GDN_Z_SIDE=0` (an A/B
/// probe: the paced z prefetch instead); read once.
pub fn z_side() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| z_side_from(std::env::var("LEGION_DEBUG_GDN_Z_SIDE").ok().as_deref()))
}

/// [`z_side`] for the switch's value `v`.
pub fn z_side_from(v: Option<&str>) -> bool {
    !v.is_some_and(|v| v.trim() == "0")
}

/// Whether the prefetch runs on the forked side stream, joined after the layer's output
/// projection (see the module doc): yes unless `LEGION_DEBUG_GDN_PF_FORK=0` (an A/B probe: the
/// compute stream, unpaced); read once.
pub fn forked() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !std::env::var("LEGION_DEBUG_GDN_PF_FORK").is_ok_and(|v| v.trim() == "0"))
}

/// Whether a step of `n` tokens may prefetch `w`: a CUDA build with stage graphs on, a GPU of
/// compute capability 12.x, `w` in that GPU's memory and 16-byte aligned, `n` up to
/// [`MAX_TOKENS`].
pub fn takes(w: Tensor<'_>, n: i64) -> bool {
    cfg!(feature = "cuda")
        && (1..=MAX_TOKENS).contains(&n)
        && crate::ggml::stage_graph::enabled()
        && !w.is_host()
        && !w.is_meta()
        && w.nbytes() >= CHUNK
        && super::q8_group::sm12x()
}

/// Whether node `t` reads `w` itself or through a source of its own (a matmul followed by its
/// scale's `mul`, say).
pub fn reads(t: Tensor<'_>, w: Tensor<'_>) -> bool {
    /// # Safety
    /// `t` is a valid tensor whose sources are null or valid tensors.
    unsafe fn within(t: *const ffi::ggml_tensor, w: *const ffi::ggml_tensor, depth: u32) -> bool {
        // SAFETY: the caller's; each source is null or a valid tensor.
        unsafe {
            (*t).src.iter().any(|&s| {
                !s.is_null() && (std::ptr::eq(s, w) || (depth > 0 && within(s, w, depth - 1)))
            })
        }
    }
    // SAFETY: a valid node.
    unsafe { within(t.raw(), w.raw(), 1) }
}

/// Chunks of a prefetch of `bytes` from byte `offset` of a `total`-byte tensor: (first chunk,
/// chunks), within the tensor, at least one; whole chunks only (a partial one is dropped), so a
/// tensor under one chunk gets none.
pub fn span(offset: usize, bytes: usize, total: usize) -> (usize, usize) {
    let all = total / CHUNK;
    if all == 0 {
        return (0, 0);
    }
    let first = (offset / CHUNK).min(all.saturating_sub(1));
    let n = (bytes / CHUNK).clamp(1, all - first);
    (first, n)
}

/// A node that, once `after` has run, reads `bytes` of `w` from byte `offset` into L2 (whole
/// [`CHUNK`]s, see [`span`]). Its F32 `[1]` result holds nothing: expand it into the graph right
/// after `after` is built so it runs next to it. `side`: on the forked side stream, which a
/// [`join`] of the node must bring back to the compute stream later in the same graph.
pub fn ahead<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    offset: usize,
    bytes: usize,
    after: Tensor<'a>,
    side: bool,
) -> Tensor<'a> {
    let (first, n) = span(offset, bytes, w.nbytes());
    ctx.external_with(
        &L2_PREFETCH,
        types::F32,
        [1, 1, 1, 1],
        &[w, after],
        &[first as i32, n as i32, i32::from(side)],
    )
}

/// A node after which the compute stream waits for side-stream prefetch `pf` ([`ahead`] with
/// `side`), placed after `after`: expanded, it pulls in `after`'s chain first (the order the
/// graph would have anyway) and runs right behind it. Its F32 `[1]` result holds nothing; for
/// a prefetch on the compute stream it launches nothing.
pub fn join<'a>(ctx: &'a Context, pf: Tensor<'a>, after: Tensor<'a>) -> Tensor<'a> {
    ctx.external(&L2_JOIN, types::F32, [1, 1, 1, 1], &[after, pf])
}

/// A node after which the compute stream waits for the side stream's last op (one launched
/// through [`on_side`]), placed after `after` like [`join`]. `keep` are that op's node and its
/// inputs: as this node's sources they stay allocated until it, so no tensor of the compute
/// stream's work in between takes their memory while the side stream still reads or writes it.
pub fn side_join<'a>(ctx: &'a Context, after: Tensor<'a>, keep: &[Tensor<'a>]) -> Tensor<'a> {
    let mut srcs = vec![after];
    srcs.extend_from_slice(keep);
    ctx.external(&SIDE_JOIN, types::F32, [1, 1, 1, 1], &srcs)
}

/// Runs `launch` on the side stream forked from `stream` at this point, then records the side
/// stream's done event for a [`side_join`]. In a capture, the node(s) `launch` captured keep the
/// least priority (`stage_graph::prio`).
#[cfg(feature = "cuda")]
pub(crate) fn on_side(
    stream: *mut std::os::raw::c_void,
    launch: impl FnOnce(*mut std::os::raw::c_void) -> Result<(), String>,
) -> Result<(), String> {
    let s = side::fork(stream)?;
    launch(s)?;
    crate::ggml::stage_graph::prio::mark_side(s);
    side::done(s)
}

#[cfg(feature = "cuda")]
unsafe fn cuda_side_join(
    _dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    side::join(stream)
}

unsafe fn cpu_l2_prefetch(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {
    // Nothing to compute: the result is never read.
}

#[cfg(feature = "cuda")]
unsafe fn cuda_l2_join(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `join`: src[1] a node made by `ahead`.
    match unsafe { plan(&*dst.src[1]) } {
        Some((.., true)) => side::join(stream),
        _ => Ok(()),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_l2_prefetch(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `ahead`.
    let Some((base, bytes, on_side)) = (unsafe { plan(dst) }) else {
        return Ok(());
    };
    if on_side {
        let s = side::fork(stream)?;
        launch::launch(base, bytes, s, false, pace_ns())?;
        return side::done(s);
    }
    launch::launch(base, bytes, stream, true, 0)
}

/// A prefetch node's (device address, bytes, on the side stream), or `None` when it launches
/// nothing: the bulk prefetch takes 16-byte aligned addresses, and an empty span has nothing
/// to read. Its [`join`] forks or waits only when this is `Some` with the side stream.
///
/// # Safety
/// `dst` is a node made by [`ahead`]: src[0] the weight, in the stream's device's memory.
#[cfg(feature = "cuda")]
unsafe fn plan(dst: &ffi::ggml_tensor) -> Option<(usize, usize, bool)> {
    // SAFETY: the caller's.
    let w = unsafe { &*dst.src[0] };
    let [first, n, on_side, ..] = external_params(dst);
    let base = w.data as usize + first as usize * CHUNK;
    (base % 16 == 0 && n > 0).then_some((base, n as usize * CHUNK, on_side != 0))
}

/// The side stream of [`forked`] prefetches: one per device, with its fork and done events.
#[cfg(feature = "cuda")]
mod side {
    use std::os::raw::{c_int, c_uint, c_void};

    unsafe extern "C" {
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: c_uint) -> c_int;
        fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> c_int;
        fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: c_uint) -> c_int;
    }

    /// A device's side stream, its fork event (recorded on the compute stream) and its done
    /// event (recorded on the side stream behind the prefetch).
    #[derive(Clone, Copy)]
    struct Side {
        dev: c_int,
        stream: *mut c_void,
        fork: *mut c_void,
        done: *mut c_void,
    }

    // SAFETY: CUDA streams and events may be used from any host thread.
    unsafe impl Send for Side {}

    static SIDES: std::sync::Mutex<Vec<Side>> = std::sync::Mutex::new(Vec::new());

    fn rt(e: c_int, what: &str) -> Result<(), String> {
        match e {
            0 => Ok(()),
            e => Err(format!("l2_prefetch side stream: {what}: CUDA error {e}")),
        }
    }

    /// The current device's side stream, made on first use (a direct run, before any capture).
    fn get() -> Result<Side, String> {
        let mut dev = 0;
        // SAFETY: an out pointer.
        rt(unsafe { cudaGetDevice(&mut dev) }, "device")?;
        let mut sides = SIDES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = sides.iter().find(|s| s.dev == dev) {
            return Ok(*s);
        }
        let event = || {
            let mut e = std::ptr::null_mut();
            // SAFETY: an out pointer; 2 = cudaEventDisableTiming.
            rt(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, "event")?;
            Ok::<_, String>(e)
        };
        let mut stream = std::ptr::null_mut();
        // SAFETY: an out pointer; 1 = cudaStreamNonBlocking.
        rt(
            unsafe { cudaStreamCreateWithFlags(&mut stream, 1) },
            "stream",
        )?;
        let s = Side {
            dev,
            stream,
            fork: event()?,
            done: event()?,
        };
        sides.push(s);
        Ok(s)
    }

    /// Forks the side stream from `stream` (it runs what follows once `stream` reached this
    /// point) and returns it.
    pub(super) fn fork(stream: *mut c_void) -> Result<*mut c_void, String> {
        let s = get()?;
        // SAFETY: live events and streams of the current device.
        unsafe {
            rt(cudaEventRecord(s.fork, stream), "fork")?;
            rt(cudaStreamWaitEvent(s.stream, s.fork, 0), "fork")?;
        }
        Ok(s.stream)
    }

    /// Records the done event on side stream `side` behind what it holds.
    pub(super) fn done(side: *mut c_void) -> Result<(), String> {
        let s = get()?;
        // SAFETY: a live event and stream of the current device.
        rt(unsafe { cudaEventRecord(s.done, side) }, "done")
    }

    /// Makes `stream` wait for the side stream's last done event.
    pub(super) fn join(stream: *mut c_void) -> Result<(), String> {
        let s = get()?;
        // SAFETY: a live event and stream of the current device.
        rt(unsafe { cudaStreamWaitEvent(stream, s.done, 0) }, "join")
    }
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `cudaLaunchAttribute`: the id, padding to 8 bytes, a 64-byte value union.
    #[repr(C, align(8))]
    struct LaunchAttribute {
        id: c_int,
        pad: [u8; 4],
        val: [u32; 16],
    }

    /// `cudaLaunchConfig_t`.
    #[repr(C)]
    struct LaunchConfig {
        grid: Dim3,
        block: Dim3,
        dynamic_smem: usize,
        stream: *mut c_void,
        attrs: *mut LaunchAttribute,
        num_attrs: u32,
    }

    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

    /// Threads a CTA, one chunk each.
    const THREADS: u32 = 32;

    /// `LEGION_DEBUG_GDN_PF_SHAPE=<KiB>x<threads>` (an A/B probe, read once): bytes per bulk
    /// prefetch (a multiple of 4 KiB, up to 1 MiB) and threads per CTA (1-32); else 4 KiB x 32.
    fn shape() -> (usize, u32) {
        static V: std::sync::OnceLock<(usize, u32)> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("LEGION_DEBUG_GDN_PF_SHAPE")
                .ok()
                .and_then(|s| {
                    let (k, t) = s.trim().split_once('x')?;
                    let k = k.parse::<usize>().ok()?;
                    let t = t.parse::<u32>().ok()?;
                    ((4..=1024).contains(&k) && k % 4 == 0 && (1..=32).contains(&t))
                        .then_some((k << 10, t))
                })
                .unwrap_or((super::CHUNK, THREADS))
        })
    }

    unsafe extern "C" {
        static legion_l2_prefetch_kernels: [*const c_void; 1];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
    }

    /// Launches the prefetch of `bytes` (whole chunks) from device address `base` on `stream`,
    /// as a programmatic dependent launch where `pdl` (the GPUs [`super::takes`] gives it are
    /// 12.x), else a plain one (the side stream: it starts once the fork's point is complete,
    /// and its `griddepcontrol.wait` returns at once). Paced, one step every `pace_ns` (0: all
    /// at once), on the side stream only: on the compute stream a paced kernel would hold every
    /// kernel after it.
    pub(super) fn launch(
        base: usize,
        bytes: usize,
        stream: *mut c_void,
        pdl: bool,
        pace_ns: u32,
    ) -> Result<(), String> {
        let (size, threads) = shape();
        let chunks = bytes.div_ceil(size);
        let mut p = base as *const u8;
        let mut total = bytes as i64;
        let mut chunk = size as i32;
        let mut late = i32::from(super::late());
        let mut pace = if pdl { 0 } else { pace_ns as i32 };
        let mut argv = [
            (&mut p as *mut *const u8).cast::<c_void>(),
            (&mut total as *mut i64).cast::<c_void>(),
            (&mut chunk as *mut i32).cast::<c_void>(),
            (&mut late as *mut i32).cast::<c_void>(),
            (&mut pace as *mut i32).cast::<c_void>(),
        ];
        // Paced: one CTA walks the whole span.
        let ctas = match pace {
            0 => chunks.div_ceil(threads as usize) as u32,
            _ => 1,
        };
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid: Dim3 {
                x: ctas,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: threads,
                y: 1,
                z: 1,
            },
            dynamic_smem: 0,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(pdl),
        };
        // SAFETY: the table's kernel takes (const char *, int64_t, int32_t, int32_t, int32_t),
        // given here by pointers to the launch's copies; `base` is device memory of the
        // stream's device.
        match unsafe {
            cudaLaunchKernelExC(&config, legion_l2_prefetch_kernels[0], argv.as_mut_ptr())
        } {
            0 => Ok(()),
            e => Err(format!("l2_prefetch: CUDA error {e}")),
        }
    }
}
