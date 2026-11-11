//! The share's graph ops and per-device state: [`publish`] and [`collect`] around a MoE layer's
//! expert matvecs (see the module docs of `cpu_share`), [`glu_down`] that builds the layer's
//! experts with them, and the registry of devices that share.
//!
//! On the CPU backend both ops pass their input through (the graph never takes them there: the
//! share is CUDA-only, [`glu_down`] checks).

use std::sync::{Arc, Mutex};

use super::HostExperts;
use super::pool::Pool;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::expert_mv;

static PUBLISH: ExternalOp = ExternalOp {
    name: "cpu_share_publish",
    cpu: cpu_pass,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::publish),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static COLLECT: ExternalOp = ExternalOp {
    name: "cpu_share_collect",
    cpu: cpu_pass,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::collect),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Tokens a step may have for the share to take part (larger steps leave every expert to the
/// GPU): the expert matvecs' default limit, and the exchange buffer's capacity.
pub const MAX_TOKENS: usize = expert_mv::MAX_TOKENS as usize;
/// Experts per token the exchange buffer holds (the CUDA kernel's limit: 128 pairs).
pub const MAX_USED: usize = 16;

/// A MoE layer's routed experts with the host's share: `silu(gate x) * (up x)` then `down`, per
/// (token, expert) pair, F32 `[n_embd, n_used, n]`, the host computing the pairs [`super::select`]
/// gives it. `w` is the tiered `[gate, up, down]` (`n_slots` VRAM slots, then `n_expert` host
/// experts), `x` F32 `[n_embd, 1, n]`, `ids` the tiered ids. `None` when the expert matvecs
/// cannot take the step (the caller's other paths compute it).
#[allow(clippy::too_many_arguments)]
pub fn glu_down<'a>(
    ctx: &'a Context,
    [gate, up, down]: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    n_slots: usize,
    n_expert: usize,
    layer: usize,
    threshold: u32,
) -> Option<Tensor<'a>> {
    let [n_used, n, ..] = ids.ne();
    if !cfg!(feature = "cuda")
        || threshold == 0
        || n as usize > MAX_TOKENS
        || n_used as usize > MAX_USED
        || !expert_mv::glu_fits(gate, up, x, ids)
        || !registered(layer)
    {
        return None;
    }
    let shared = publish(ctx, x, ids, layer, n_slots, n_expert, threshold);
    let act = expert_mv::glu_tiered(ctx, gate, up, x, shared, n_slots);
    if !expert_mv::fits(down, act, shared) {
        return None;
    }
    let out = expert_mv::mul_mat_id_tiered(ctx, down, act, shared, n_slots);
    Some(collect(ctx, out, shared))
}

/// The router's `ids` (I32 `[n_used, n]`, tiered) of layer `layer`, published with the
/// activations `x` (F32 `[n_embd, 1, n]`) for the host: the ids with the host's pairs replaced
/// by [`super::HOST_PAIR`].
fn publish<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    layer: usize,
    n_slots: usize,
    n_expert: usize,
    threshold: u32,
) -> Tensor<'a> {
    let [n_used, n, _, _] = ids.ne();
    ctx.external_with(
        &PUBLISH,
        types::I32,
        [n_used, n, 1, 1],
        &[ids, x],
        &[
            layer as i32,
            n_slots as i32,
            n_expert as i32,
            threshold as i32,
        ],
    )
}

/// The down projection's output `experts` (F32 `[n_embd, n_used, n]`) with the host's rows for
/// its pairs, once it is done. `ids` is [`publish`]'s result.
fn collect<'a>(ctx: &'a Context, experts: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
    ctx.external_with(&COLLECT, types::F32, experts.ne(), &[experts, ids], &[])
}

/// Both ops on the CPU: `src[0]` copied to the output.
unsafe fn cpu_pass(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `publish` or `collect`; host tensors of the output's size.
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
/// device; a layer belongs to one device).
fn registered(layer: usize) -> bool {
    DEVICES
        .lock()
        .unwrap()
        .iter()
        .flatten()
        .any(|d| d.layers.get(layer).is_some_and(Option::is_some))
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
    }

    /// `CU_STREAM_WAIT_VALUE_GEQ`: `(int32_t)(*addr - value) >= 0`, so wrapping is fine.
    const WAIT_GEQ: u32 = 0;

    unsafe extern "C" {
        fn legion_cpu_share_publish_launch(p: *const Publish, stream: *mut c_void) -> c_int;
        fn legion_cpu_share_collect_launch(p: *const Collect, stream: *mut c_void) -> c_int;
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
        let lay = &share.layout;
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
            ids_host: (share.device_base + lay.ids as u64) as *mut i32,
            x_host: (share.device_base + lay.x as u64) as *mut f32,
            ids_dev: dst.data as *mut i32,
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

    pub(super) unsafe fn collect(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let share = device(current_device()?).ok_or("cpu share: device not registered")?;
        // SAFETY: a node made by `collect`: experts F32 [n_embd, n_used, n], ids I32 dense.
        let (experts, ids) = unsafe { (&*dst.src[0], &*dst.src[1]) };
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
        let e = unsafe {
            cuStreamWaitValue32_v2(stream, share.device_base + lay.done as u64, seq, WAIT_GEQ)
        };
        if e != 0 {
            return Err(format!("cuStreamWaitValue32: error {e}"));
        }
        let p = Collect {
            ids: ids.data as *const i32,
            n_used: ids.ne[0] as i32,
            n_tok: ids.ne[1] as i32,
            rows: dst.ne[0] as i32,
            host: (share.device_base + lay.out as u64) as *const f32,
            src: experts.data as *const f32,
            src_nb1: experts.nb[1] as i64,
            src_nb2: experts.nb[2] as i64,
            dst: dst.data as *mut f32,
        };
        // SAFETY: device tensors; the exchange's output rows are mapped for the device.
        match unsafe { legion_cpu_share_collect_launch(&p, stream) } {
            0 => Ok(()),
            e => Err(format!("cpu share collect: error {e}")),
        }
    }
}
