//! The share's graph ops and per-device state: [`publish`] before a MoE layer's expert matvecs
//! and the share's wraps of them (see the module docs of `cpu_share`), [`glu_down`] that builds
//! the layer's experts with them, and the registry of devices that share.
//!
//! On the CPU backend `publish` passes the ids through and the wrapped matvecs compute as
//! unwrapped (the graph never takes them there: the share is CUDA-only, [`glu_down`] checks).

use std::sync::{Arc, Mutex};

use super::HostExperts;
use super::pool::Pool;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, retarget, types};
use crate::model::core::expert_mv;

static PUBLISH: ExternalOp = ExternalOp {
    name: "cpu_share_publish",
    cpu: cpu_pass,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::publish),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The exact share's SwiGLU op: `expert_mv_glu`'s launches, then the host pairs' rows and
/// blocks made from the host's gate and up sums (`activate`), in place.
static SHARE_GLU: ExternalOp = ExternalOp {
    name: "cpu_share_glu",
    cpu: expert_mv::cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::glu),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// [`SHARE_GLU`] of a pre-quantized SwiGLU op ([`expert_mv::glu_tiered_preq`]).
static SHARE_GLU_PREQ: ExternalOp = ExternalOp {
    name: "cpu_share_glu",
    cpu: expert_mv::cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::glu_preq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The share's down op: `expert_mv`'s launches, then the host's rows for its pairs once they
/// are done (`collect`), in place.
static SHARE_DOWN: ExternalOp = ExternalOp {
    name: "cpu_share_down",
    cpu: expert_mv::cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::down),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The share's down op's source holding the router's ids, which it does not read: ggml's
/// allocator keeps them to the down op as without the share (the matvec reads sources 0 to 4).
const ROUTER_SRC: usize = ffi::GGML_MAX_SRC as usize - 1;

/// Tokens a step may have for the share to take part (larger steps leave every expert to the
/// GPU): the expert matvecs' default limit, and the exchange buffer's capacity.
pub const MAX_TOKENS: usize = expert_mv::MAX_TOKENS as usize;
/// Experts per token the exchange buffer holds (the CUDA kernel's limit: 128 pairs).
pub const MAX_USED: usize = 16;
/// (token, expert) pairs of a step the share takes at most: the bytes of a layer's
/// [`publish`] output are at most 4 times this (its place, [`glu_down`]'s `ids_at`).
pub const MAX_PAIRS: usize = MAX_TOKENS * MAX_USED;

/// A layer of the share: its tiered tensors' VRAM slots (`n_slots`), the host experts the host
/// may take (`n_expert`), its layer id and the share's threshold ([`super::select`]).
#[derive(Debug, Clone, Copy)]
pub struct ShareLayer {
    pub n_slots: usize,
    pub n_expert: usize,
    pub layer: usize,
    pub threshold: u32,
}

/// A MoE layer's routed experts with the host's share: `silu(gate x) * (up x)` then `down`, per
/// (token, expert) pair, F32 `[n_embd, n_used, n]`, the host computing the pairs [`super::select`]
/// gives it. `w` is the tiered `[gate, up, down]` (`n_slots` VRAM slots, then `n_expert` host
/// experts), `x` F32 `[n_embd, 1, n]`, `ids` the tiered ids, `xq` x's Q8_1 blocks where an op
/// wrote them (the SwiGLU op then reads them, [`expert_mv::glu_tiered_preq`], as the share-off
/// path does), `ids_at` device memory of at least `4 * MAX_PAIRS` bytes (a base tensor and a
/// byte offset in it) the layer's [`publish`] output goes to. `None` when the expert matvecs
/// cannot take the step (the caller's other paths compute it).
///
/// The nodes lay out the step graph's memory as the share-off path's SwiGLU and down ops do
/// ([`share_nodes`]), so ggml-cuda's layout-dependent MoE fusions (#472) fire in the same
/// layers with the share on and off.
pub fn glu_down<'a>(
    ctx: &'a Context,
    w: [Tensor<'a>; 3],
    (x, ids, xq): (Tensor<'a>, Tensor<'a>, Option<Tensor<'a>>),
    ids_at: (Tensor<'a>, usize),
    share: ShareLayer,
) -> Option<Tensor<'a>> {
    let [gate, up, down] = w;
    let [n_used, n, ..] = ids.ne();
    if !cfg!(feature = "cuda")
        || share.threshold == 0
        || n as usize > MAX_TOKENS
        || n_used as usize > MAX_USED
        || !expert_mv::glu_down_fits(gate, up, down, x, ids)
    {
        return None;
    }
    let exact = registered(share.layer)?;
    if exact && !EXACT_WIDTHS.contains(&(n as usize)) {
        return None;
    }
    Some(share_nodes(ctx, w, (x, ids, xq), ids_at, share, exact))
}

/// [`glu_down`]'s nodes, whatever the backend (CPU tests build them), `exact` the exact
/// share's (its SwiGLU op wrapped too).
///
/// The share-off path's two ops (`glu_tiered_preq` or `glu_tiered`, then
/// `mul_mat_id_tiered`) reading the published ids, wrapped: the same nodes, so ggml's
/// allocator gives each tensor the share-off offset and a tensor read only by the down op's
/// output (the weighted sum's `mul`) may take its memory in place, as without the share.
/// In-place nodes of their own would not do: ggml's allocator never lets a view's reader
/// take over the view's memory, so the `mul` after them got fresh memory and the fusions
/// changed.
/// - [`publish`]'s output goes to `ids_at`, outside the graph's memory, so it takes no
///   offset;
/// - the down op keeps the router's `ids` as an unread source ([`ROUTER_SRC`]), so they stay
///   allocated to it as the share-off ops keep them.
pub(super) fn share_nodes<'a>(
    ctx: &'a Context,
    [gate, up, down]: [Tensor<'a>; 3],
    (x, ids, xq): (Tensor<'a>, Tensor<'a>, Option<Tensor<'a>>),
    ids_at: (Tensor<'a>, usize),
    share: ShareLayer,
    exact: bool,
) -> Tensor<'a> {
    let n_slots = share.n_slots;
    let shared = publish(ctx, x, ids, ids_at, share);
    let act = match xq {
        Some(q) => expert_mv::glu_tiered_preq(ctx, gate, up, x, shared, n_slots, q),
        None => expert_mv::glu_tiered(ctx, gate, up, x, shared, n_slots),
    };
    let glu = if xq.is_some() {
        &SHARE_GLU_PREQ
    } else {
        &SHARE_GLU
    };
    let out = expert_mv::mul_mat_id_tiered(ctx, down, act, shared, n_slots);
    // SAFETY: nodes just made by the ops the wraps run first, in no graph yet; the down op's
    // sources end at its scratch (4 at most), so `ROUTER_SRC` is free.
    unsafe {
        if exact {
            retarget(act.raw(), glu);
        }
        retarget(out.raw(), &SHARE_DOWN);
        let node = &mut *out.raw();
        assert!(node.src[ROUTER_SRC - 1..].iter().all(|s| s.is_null()));
        node.src[ROUTER_SRC] = ids.raw();
    }
    out
}

/// The steps' widths the exact share takes: those whose kernel bits the host's were checked
/// against on a real model (`legion expert-cpu-proof`); 5 and 7 tokens never came up.
pub const EXACT_WIDTHS: [usize; 6] = [1, 2, 3, 4, 6, 8];

/// The router's `ids` (I32 `[n_used, n]`, tiered) of the share's layer, published with the
/// activations `x` (F32 `[n_embd, 1, n]`) for the host: the ids with the host's pairs replaced
/// by [`super::HOST_PAIR`], written at `ids_at` (outside the graph's memory, so the node takes
/// no place in the step graph's layout).
fn publish<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    (base, offset): (Tensor<'a>, usize),
    share: ShareLayer,
) -> Tensor<'a> {
    let [n_used, n, _, _] = ids.ne();
    let t = ctx.external_with(
        &PUBLISH,
        types::I32,
        [n_used, n, 1, 1],
        &[ids, x],
        &[
            share.layer as i32,
            share.n_slots as i32,
            share.n_expert as i32,
            share.threshold as i32,
        ],
    );
    // SAFETY: a fresh node; the place is the layer's own (no other node writes it), and the
    // layer's nodes that read it run in the stream before the next step's `publish` of it.
    unsafe { t.place_in(base, offset) };
    t
}

/// [`publish`] on the CPU: the ids (`src[0]`) copied to the output.
unsafe fn cpu_pass(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `publish`; host tensors of the output's size.
    unsafe {
        let src = &*dst.src[0];
        let n = ffi::ggml_nbytes(dst);
        assert!(ffi::ggml_is_contiguous(src) && ffi::ggml_nbytes(src) == n);
        std::ptr::copy_nonoverlapping(src.data as *const u8, dst.data as *mut u8, n);
    }
}

/// A device's share state.
pub struct DeviceShare {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) exchange: super::pool::Exchange,
    /// The exchange buffer's device address (the layout's offsets apply to it).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) device_base: u64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) layout: super::Layout,
    /// The exact share's part of the buffer (x86), if the device runs it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) exact: Option<super::ExactLayout>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) seq: super::Sequence,
    /// Per layer (index = layer id), its experts in host memory, if the device runs it.
    pub(super) layers: Vec<Option<HostExperts>>,
    pub(super) pool: Arc<Pool>,
    /// The device's job queue in [`Self::pool`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) queue: usize,
    /// Frees the exchange buffer (page-locked, mapped host memory).
    pub(super) _buffer: super::HostBuffer,
}

// SAFETY: the exchange is plain addresses of the buffer this owns; the pool is Sync.
unsafe impl Send for DeviceShare {}
unsafe impl Sync for DeviceShare {}

/// The devices that share, by CUDA device index.
static DEVICES: Mutex<Vec<Option<Arc<DeviceShare>>>> = Mutex::new(Vec::new());

/// Registers `share` for CUDA device `device` (replacing any earlier one).
pub fn register(device: usize, share: Arc<DeviceShare>) {
    let mut d = DEVICES.lock().unwrap();
    if d.len() <= device {
        d.resize(device + 1, None);
    }
    d[device] = Some(share);
}

/// Unregisters every device (the residency that set them up is gone). Stops the pools before
/// any exchange buffer is freed: a pool serves every device of its node and may still hold
/// their queued jobs, which [`Pool::stop`] releases.
pub fn clear() {
    let shares = std::mem::take(&mut *DEVICES.lock().unwrap());
    for share in shares.iter().flatten() {
        share.pool.stop();
    }
    drop(shares);
}

/// Whether a device registered layer `layer` (graphs are built before the step knows its
/// device; a layer belongs to one device): `Some(exact)`, whether its share is the exact one.
fn registered(layer: usize) -> Option<bool> {
    DEVICES
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .find(|d| d.layers.get(layer).is_some_and(Option::is_some))
        .map(|d| d.exact.is_some())
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn device(index: usize) -> Option<Arc<DeviceShare>> {
    DEVICES.lock().unwrap().get(index).cloned().flatten()
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::{c_int, c_void};

    use super::super::pool::Job;
    use super::device;
    use crate::ggml::{external_params, ffi};
    use crate::model::core::expert_mv;

    #[repr(C)]
    struct Publish {
        ids: *const i32,
        ids_nb1: i64,
        x: *const f32,
        x_nb1: i64,
        n_used: i32,
        n_tok: i32,
        k: i32,
        n_slots: i32,
        n_expert: i32,
        threshold: u32,
        dither: u32,
        ids_host: *mut i32,
        x_host: *mut f32,
        ids_dev: *mut i32,
        exact: i32,
        thr: *const u32,
        xq_host: *mut c_void,
        n_miss: *mut u32,
        all_upto: u32,
    }

    #[repr(C)]
    struct Activate {
        ids: *const i32,
        n_used: i32,
        n_tok: i32,
        rows: i32,
        src: *const f32,
        src_nb1: i64,
        src_nb2: i64,
        gu: *const f32,
        aq: *mut c_void,
        dst: *mut f32,
        limit: f32,
        wait: *const u32,
        seq: u32,
    }

    #[repr(C)]
    struct Collect {
        ids: *const i32,
        n_used: i32,
        n_tok: i32,
        rows: i32,
        host: *const f32,
        src: *const f32,
        src_nb1: i64,
        src_nb2: i64,
        dst: *mut f32,
        wait: *const u32,
        seq: u32,
    }

    /// `CU_STREAM_WAIT_VALUE_GEQ`: `(int32_t)(*addr - value) >= 0`, so wrapping is fine.
    const WAIT_GEQ: u32 = 0;

    /// How `activate` and `collect` wait for the host's flag. By default the host pairs' blocks
    /// of the op's kernel poll it (a system-scope acquire load a few hundred nanoseconds
    /// apart), so the launch and the GPU pairs' copies overlap the host's work and the kernel
    /// sees the flag about as soon as the host raises it. `LEGION_DEBUG_SHARE_STREAM_WAIT=1`:
    /// the stream waits for it before the launch (`cuStreamWaitValue32`, the way before), for
    /// A/B runs. The same results either way.
    fn stream_wait() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("LEGION_DEBUG_SHARE_STREAM_WAIT").is_ok_and(|v| v == "1"))
    }

    /// Waits in `stream` for the host's flag at `addr` to reach `seq` when the stream waits
    /// ([`stream_wait`]); otherwise gives the flag's device address for the kernel to poll.
    ///
    /// # Safety
    /// `addr` is a flag's device address in a device's exchange buffer.
    unsafe fn wait_for(stream: *mut c_void, addr: u64, seq: u32) -> Result<*const u32, String> {
        if !stream_wait() {
            return Ok(addr as *const u32);
        }
        // SAFETY: the caller's.
        match unsafe { cuStreamWaitValue32_v2(stream, addr, seq, WAIT_GEQ) } {
            0 => Ok(std::ptr::null()),
            e => Err(format!("cuStreamWaitValue32: error {e}")),
        }
    }

    unsafe extern "C" {
        fn legion_cpu_share_publish_launch(p: *const Publish, stream: *mut c_void) -> c_int;
        fn legion_cpu_share_collect_launch(p: *const Collect, stream: *mut c_void) -> c_int;
        fn legion_cpu_share_activate_launch(p: *const Activate, stream: *mut c_void) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cuStreamWriteValue32_v2(stream: *mut c_void, addr: u64, value: u32, flags: u32)
        -> c_int;
        fn cuStreamWaitValue32_v2(stream: *mut c_void, addr: u64, value: u32, flags: u32) -> c_int;
    }

    fn current_device() -> Result<usize, String> {
        let mut d = 0;
        // SAFETY: an out-parameter query.
        match unsafe { cudaGetDevice(&mut d) } {
            0 => Ok(d as usize),
            e => Err(format!("cudaGetDevice: error {e}")),
        }
    }

    pub(super) unsafe fn publish(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [layer, n_slots, n_expert, threshold, ..] = external_params(dst);
        let share = device(current_device()?).ok_or("cpu share: device not registered")?;
        // SAFETY: a node made by `publish`: ids I32 [n_used, n], x F32 [n_embd, 1, n].
        let (ids, x) = unsafe { (&*dst.src[0], &*dst.src[1]) };
        let (n_used, n_tok, k) = (ids.ne[0] as usize, ids.ne[1] as usize, x.ne[0] as usize);
        let ex = share.exchange;
        if n_tok > ex.max_tok || n_used > ex.max_used || k != ex.n_embd {
            return Err(format!(
                "cpu share: a step of {n_tok} x {n_used} (k {k}) does not fit the exchange"
            ));
        }
        let experts = share
            .layers
            .get(layer as usize)
            .copied()
            .flatten()
            .ok_or("cpu share: layer not registered on this device")?;
        // The pool reads ids below `n_expert` at its experts' addresses (the bounded host
        // tier: its RAM slots, so no more than those).
        if n_expert < 0 || n_expert as usize > experts.n_expert {
            return Err(format!(
                "cpu share: layer {layer} takes {n_expert} host ids, its host tensor holds {}",
                experts.n_expert
            ));
        }
        let lay = &share.layout;
        let base = share.device_base;
        let el = share.exact.as_ref();
        let n_layers = ex.exact.map_or(0, |x| x.n_layers);
        if el.is_some() && (k % 32 != 0 || layer as usize >= n_layers) {
            return Err(format!(
                "cpu share: layer {layer} (k {k}) does not fit the exchange"
            ));
        }
        let p = Publish {
            ids: ids.data as *const i32,
            ids_nb1: ids.nb[1] as i64,
            x: x.data as *const f32,
            x_nb1: x.nb[2] as i64,
            n_used: n_used as i32,
            n_tok: n_tok as i32,
            k: k as i32,
            n_slots,
            n_expert,
            threshold: threshold as u32,
            dither: super::super::dither(layer as u32),
            ids_host: (base + lay.ids as u64) as *mut i32,
            x_host: (base + lay.x as u64) as *mut f32,
            ids_dev: dst.data as *mut i32,
            exact: i32::from(el.is_some()),
            thr: el.map_or(std::ptr::null(), |e| {
                (base + (e.thr + layer as usize * 4) as u64) as *const u32
            }),
            xq_host: el.map_or(std::ptr::null_mut(), |e| {
                (base + e.xq as u64) as *mut c_void
            }),
            n_miss: el.map_or(std::ptr::null_mut(), |e| {
                (base + e.n_miss as u64) as *mut u32
            }),
            all_upto: ex.exact.map_or(0, |x| x.all_upto),
        };
        // SAFETY: device tensors of the stream's device; the exchange is mapped for it.
        let e = unsafe { legion_cpu_share_publish_launch(&p, stream) };
        if e != 0 {
            return Err(format!("cpu share publish: error {e}"));
        }
        let seq = share.seq.next();
        // Ordered after the kernel's writes (no `CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`).
        // SAFETY: the ready flag's device address, in the exchange buffer.
        let e = unsafe {
            cuStreamWriteValue32_v2(stream, share.device_base + lay.ready as u64, seq, 0)
        };
        if e != 0 {
            return Err(format!("cuStreamWriteValue32: error {e}"));
        }
        share.pool.submit(
            share.queue,
            Job {
                seq,
                layer: layer as u32,
                n_tok,
                n_used,
                experts,
                exchange: ex,
            },
        );
        Ok(())
    }

    /// [`super::SHARE_GLU`]: the SwiGLU op's launches, then [`activate`].
    pub(super) unsafe fn glu(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: the caller's; a node made by `glu_tiered`.
        unsafe {
            expert_mv::cuda_expert_mv(dst, stream)?;
            activate(dst, stream)
        }
    }

    /// [`super::SHARE_GLU_PREQ`]: the pre-quantized SwiGLU op's launches, then [`activate`].
    pub(super) unsafe fn glu_preq(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: the caller's; a node made by `glu_tiered_preq`.
        unsafe {
            expert_mv::cuda_expert_mv_preq(dst, stream)?;
            activate(dst, stream)
        }
    }

    /// [`super::SHARE_DOWN`]: the down op's launches, then [`collect`].
    pub(super) unsafe fn down(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: the caller's; a node made by `mul_mat_id_tiered`.
        unsafe {
            expert_mv::cuda_expert_mv(dst, stream)?;
            collect(dst, stream)?;
            // The layer's share of the expert cache's swap copies starts behind it (#37).
            crate::residency::cuda::swap_pace_layer(current_device()?, stream)
        }
    }

    /// The exact share's SwiGLU output `act` (the node, F32 `[n_ff, n_used, n]`, the published
    /// ids its source 2): waits for the host's gate and up sums, writes its pairs' SwiGLU rows
    /// (which the SwiGLU op skipped) and blocks, raises "act ready".
    unsafe fn activate(act: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let share = device(current_device()?).ok_or("cpu share: device not registered")?;
        let e = share
            .exact
            .as_ref()
            .ok_or("cpu share: activate without the exact share")?;
        // SAFETY: a SwiGLU op's node: ids I32 dense.
        let ids = unsafe { &*act.src[2] };
        let rows = act.ne[0] as usize;
        if rows > e.n_ff || rows % 32 != 0 {
            return Err(format!(
                "cpu share: {rows} rows do not fit the exchange ({})",
                e.n_ff
            ));
        }
        let base = share.device_base;
        let seq = share.seq.last();
        // `auto` steers the share by who waited ([`super::pool::nudge`]); otherwise only the
        // debug statistics read the flag.
        if share.exchange.exact.is_some_and(|x| x.auto)
            || tracing::enabled!(target: "legion::residency::cpu_share", tracing::Level::DEBUG)
        {
            // SAFETY: the flag's device address, in the exchange buffer.
            let r = unsafe { cuStreamWriteValue32_v2(stream, base + e.act_arrived as u64, seq, 0) };
            if r != 0 {
                return Err(format!("cuStreamWriteValue32: error {r}"));
            }
        }
        // SAFETY: the flag's device address, in the exchange buffer.
        let wait = unsafe { wait_for(stream, base + e.gu_done as u64, seq)? };
        let p = Activate {
            ids: ids.data as *const i32,
            n_used: ids.ne[0] as i32,
            n_tok: ids.ne[1] as i32,
            rows: rows as i32,
            src: act.data as *const f32,
            src_nb1: act.nb[1] as i64,
            src_nb2: act.nb[2] as i64,
            gu: (base + e.gu as u64) as *const f32,
            aq: (base + e.aq as u64) as *mut c_void,
            dst: act.data as *mut f32,
            limit: 0.0,
            wait,
            seq,
        };
        // SAFETY: device tensors; the exchange is mapped for the device.
        let r = unsafe { legion_cpu_share_activate_launch(&p, stream) };
        if r != 0 {
            return Err(format!("cpu share activate: error {r}"));
        }
        // Ordered after the kernel's writes, as `publish`'s flag.
        // SAFETY: the flag's device address, in the exchange buffer.
        match unsafe { cuStreamWriteValue32_v2(stream, base + e.act_ready as u64, seq, 0) } {
            0 => Ok(()),
            r => Err(format!("cuStreamWriteValue32: error {r}")),
        }
    }

    /// The down op's output `experts` (the node, F32 `[n_embd, n_used, n]`, the published ids
    /// its source 2) with the host's rows for its pairs (which the down op skipped), once they
    /// are done.
    unsafe fn collect(experts: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let share = device(current_device()?).ok_or("cpu share: device not registered")?;
        // SAFETY: a down op's node: ids I32 dense.
        let ids = unsafe { &*experts.src[2] };
        let lay = &share.layout;
        let seq = share.seq.last();
        if tracing::enabled!(target: "legion::residency::cpu_share", tracing::Level::DEBUG) {
            // SAFETY: the arrived flag's device address, in the exchange buffer.
            let e = unsafe {
                cuStreamWriteValue32_v2(stream, share.device_base + lay.arrived as u64, seq, 0)
            };
            if e != 0 {
                return Err(format!("cuStreamWriteValue32: error {e}"));
            }
        }
        // SAFETY: the done flag's device address, in the exchange buffer.
        let wait = unsafe { wait_for(stream, share.device_base + lay.done as u64, seq)? };
        let p = Collect {
            ids: ids.data as *const i32,
            n_used: ids.ne[0] as i32,
            n_tok: ids.ne[1] as i32,
            rows: experts.ne[0] as i32,
            host: (share.device_base + lay.out as u64) as *const f32,
            src: experts.data as *const f32,
            src_nb1: experts.nb[1] as i64,
            src_nb2: experts.nb[2] as i64,
            dst: experts.data as *mut f32,
            wait,
            seq,
        };
        // SAFETY: device tensors; the exchange's output rows are mapped for the device.
        match unsafe { legion_cpu_share_collect_launch(&p, stream) } {
            0 => Ok(()),
            e => Err(format!("cpu share collect: error {e}")),
        }
    }
}
