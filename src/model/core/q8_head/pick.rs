//! The MTP draft pick behind the one-column Q8_0 draft LM head (#814): the draft token
//! (ggml-cuda's `argmax_f32` pick) and its probability (ggml-cuda's `soft_max` read at the pick
//! by `get_rows`, the value the draft gate compares with `--spec-p-min`), with the head's
//! logits read once instead of three more times.
//!
//! After [`super::mul_mat`] a draft pass ran the split argmax (`cluster_select::argmax`, two
//! launches over the 248,320 logits), ggml's `soft_max` (its cooperative
//! `soft_max_f32_parallelize_cols` on one row: three passes, the whole row of probabilities
//! written) and a `get_rows` of the one probability: ~22 us a draft pass on a V100 (nsys, #814).
//! Here ([`head`]):
//!
//! - the head's kernel also keeps, per CTA, the best (value, tie rank) of the rows it computed
//!   and their largest value, in a buffer of the stream (`src/cuda/q8_head.cu`);
//! - an argmax node joins those partials in one CTA: the same pick, ties included (ggml-cuda's
//!   tie order is a total order, see [`crate::model::cluster_select`]), and the row's largest
//!   value (exact in any order);
//! - a probability node sums `expf(x - max)` with `soft_max_f32_parallelize_cols`'s terms and
//!   order (one CTA of 256 threads per SM, its `block_reduce`s), and divides the pick's term by
//!   the sum as ggml's soft_max does: the same bits.
//!
//! The partials are not a graph tensor (one buffer per device and stream, made at the first
//! launch, as the split argmax's): a tensor would move the graph's memory layout, and with it
//! which ggml-cuda fusions fire (#472). The three nodes run in order on one stream, so the
//! buffer holds the head's partials when the argmax node reads them.
//!
//! Taken where the head takes [`super::mul_mat`] (compute capability 7.x, one drafting
//! sequence) and ggml's soft_max would run its cooperative kernel there; on by default,
//! `legion serve --no-draft-head-pick` keeps the split argmax, soft_max and get_rows
//! ([`set_pick`]). The CPU implementations are references, not ggml's bits.
//!
//! A GB10's Q4_K draft head ([`crate::model::core::q4k_head::head`], #836) writes the same
//! partials and takes the same argmax and probability nodes ([`tail`]); its switches are
//! `--no-draft-head-q4k-pick` and, for 2-7 drafting sequences, `--no-draft-head-q4k-pick-cols`.
//! There the buffer holds a slice per column ([`SLICES`] of [`buffer_bytes`]), and the argmax
//! and probability kernels run a CTA row per column: ggml's cooperative soft_max loops over the
//! rows with one grid, each row summed as a single row's ([`coop`]).

#[cfg(feature = "cuda")]
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static ARGMAX: ExternalOp = ExternalOp {
    name: "q8_head_argmax",
    cpu: cpu_argmax,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::argmax),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static PMAX: ExternalOp = ExternalOp {
    name: "q8_head_pmax",
    cpu: cpu_pmax,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::pmax),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Byte offset of the partials in the buffer (`PICK_PARTS` in `src/cuda/q8_head.cu`): after
/// a 16-byte header and the probability CTAs' 256 sums.
pub const PARTS_OFFSET: usize = 2048;

/// Bytes of a partial (best value bits, its tie rank, the largest value bits, padding).
pub const PART_BYTES: usize = 16;

/// Partials per SM the buffer holds: more CTAs than any GPU keeps resident on one SM.
pub const PARTS_PER_SM: usize = 32;

/// Most SMs the probability kernel takes (ggml's cooperative soft_max needs fewer CTAs than
/// its 256 threads).
pub const MAX_SMS: i32 = 255;

/// Bytes of one column's slice of the buffer for a GPU of `sms` SMs.
pub fn buffer_bytes(sms: usize) -> usize {
    PARTS_OFFSET + sms * PARTS_PER_SM * PART_BYTES
}

/// Column slices the buffer holds, each of [`buffer_bytes`] (`head_pick.cuh`): the most
/// columns a pick takes ([`MAX_COLS`]) and one to spare.
pub const SLICES: usize = 8;

/// Most columns (drafting sequences) the pick takes: ggml's soft_max runs its cooperative
/// kernel, whose sums the probability node follows, for rows of more than [`COOP_COLS`] values
/// per row of the tensor, so up to 7 rows of a 65,536-value draft head ([`coop`]).
pub const MAX_COLS: i64 = 7;

/// ggml-cuda's `soft_max_f32_cuda` takes `soft_max_f32_parallelize_cols` (when the row does not
/// fit a CTA's shared memory) only while `ncols / nrows > COOP_COLS`.
pub const COOP_COLS: i64 = 8192;

/// Whether ggml's soft_max of `rows` values in each of `n` rows runs the cooperative kernel
/// (given rows too long for a CTA's shared memory) and the pick takes that many columns.
pub fn coop(rows: i64, n: i64) -> bool {
    (1..=MAX_COLS).contains(&n) && rows / n > COOP_COLS
}

/// Whether graphs built from now on run the pick on the head's partials ([`set_pick`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on take the draft pick ops after a one-column Q8_0 draft
/// LM head on [`super::mul_mat`] (on by default; `legion serve --no-draft-head-pick` turns it
/// off).
pub fn set_pick(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether the draft pass's head `w @ x` takes [`head`]: the head takes the op
/// ([`super::takes`]), the pick is switched on, and the current GPU runs ggml's cooperative
/// soft_max on at most [`MAX_SMS`] SMs.
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed) && super::takes(w, x) && device_supported()
}

/// Whether the current GPU takes the pick: cooperative launches (what makes ggml's soft_max
/// take the kernel whose sums the probability node follows) and at most [`MAX_SMS`] SMs.
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::sm_count().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// The draft head and pick for `w` and `x` that [`super::fits`]: the logits (F32 `[rows,
/// 1]`, [`super::mul_mat`]'s), the pick (I32 `[1]`, `cluster_select::argmax`'s) and its
/// probability (F32 `[1]`, ggml's `soft_max` read at the pick). The three nodes must run
/// together, in this order (one graph, nothing between them on the stream).
pub fn head<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let logits = super::mul_mat_pick(ctx, w, x);
    let (argmax, pmax) = tail(ctx, logits);
    (logits, argmax, pmax)
}

/// The pick nodes after a head node that left its partials in the stream's buffer (F32
/// `[rows, n]` logits, 1 to [`MAX_COLS`] columns: [`super::mul_mat_pick`]'s or
/// `q4k_head::head`'s): each column's pick (I32 `[n]`) and its probability (F32 `[n]`).
/// Nothing may run on the stream between the head and these.
pub(in crate::model::core) fn tail<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>) {
    let n = logits.ne()[1];
    assert!(
        (1..=MAX_COLS).contains(&n),
        "q8_head pick: {n} columns, at most {MAX_COLS}"
    );
    let argmax = ctx.external(&ARGMAX, types::I32, [n, 1, 1, 1], &[logits]);
    let pmax = ctx.external(&PMAX, types::F32, [n, 1, 1, 1], &[logits, argmax]);
    (argmax, pmax)
}

/// Column `j` of the logits of a node made by [`head`] (host memory).
///
/// # Safety
/// `t` is a host F32 tensor of contiguous columns, `j` one of them.
unsafe fn values(t: &ffi::ggml_tensor, j: usize) -> &[f32] {
    // SAFETY: as the function's contract.
    unsafe {
        let at = (t.data as *const u8).add(j * t.nb[1]) as *const f32;
        std::slice::from_raw_parts(at, t.ne[0] as usize)
    }
}

unsafe fn cpu_argmax(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `head`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        for j in 0..dst.ne[0] as usize {
            *(dst.data as *mut i32).add(j) = crate::model::cluster_select::argmax_row(values(x, j));
        }
    }
}

/// The probability of `x[pick]` in the softmax of `x`, in plain sequential F32 (a reference: the
/// GPU follows ggml-cuda's sum order); NaN for no pick.
pub(crate) fn pick_probability(x: &[f32], pick: i32) -> f32 {
    let Ok(a) = usize::try_from(pick) else {
        return f32::NAN;
    };
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    (x[a] - max).exp() / sum
}

unsafe fn cpu_pmax(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `head`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        for j in 0..dst.ne[0] as usize {
            let pick = *((*dst.src[1]).data as *const i32).add(j);
            *(dst.data as *mut f32).add(j) = pick_probability(values(x, j), pick);
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `legion_q8_head_argmax` (`src/cuda/q8_head.cu`).
    #[repr(C)]
    struct Argmax {
        pick: *const c_void,
        dst: *mut i32,
        stride: i64,
    }

    /// `legion_q8_head_pmax` (`src/cuda/q8_head.cu`).
    #[repr(C)]
    struct Pmax {
        x: *const f32,
        pick: *const i32,
        dst: *mut f32,
        buf: *mut c_void,
        n: i32,
        x_ld: i64,
        stride: i64,
    }

    unsafe extern "C" {
        static legion_q8_head_argmax_kernel: *const c_void;
        static legion_q8_head_pmax_kernel: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
    }

    /// `cudaDevAttrMultiProcessorCount`.
    const ATTR_SMS: c_int = 16;
    /// `cudaDevAttrCooperativeLaunch`.
    const ATTR_COOPERATIVE: c_int = 95;

    /// Threads of the argmax and probability kernels (`PMAX_T`, ggml's soft_max block).
    const THREADS: u32 = 256;

    /// The current device's ordinal.
    fn device() -> Option<c_int> {
        let mut dev = 0;
        // SAFETY: a plain query of the current device.
        (unsafe { cudaGetDevice(&mut dev) } == 0).then_some(dev)
    }

    /// The current GPU's SM count where it takes the pick (cooperative launches, at most
    /// [`MAX_SMS`] SMs).
    pub(super) fn sm_count() -> Option<i32> {
        let dev = device()?;
        let (mut sms, mut coop) = (0, 0);
        // SAFETY: plain queries of a device's attributes.
        let ok = unsafe {
            cudaDeviceGetAttribute(&mut sms, ATTR_SMS, dev) == 0
                && cudaDeviceGetAttribute(&mut coop, ATTR_COOPERATIVE, dev) == 0
        };
        (ok && coop != 0 && (1..=MAX_SMS).contains(&sms)).then_some(sms)
    }

    /// A stream's buffer: its device memory, the partials a column's slice holds, the device's
    /// SM count and the bytes between the [`SLICES`] column slices.
    #[derive(Clone, Copy)]
    pub(in crate::model::core) struct Buffer {
        pub ptr: *mut c_void,
        pub parts: i32,
        pub sms: i32,
        pub stride: i64,
    }

    /// A buffer made so far: its device and stream, its address, partials and SMs.
    type Entry = (c_int, usize, usize, i32, i32);

    static BUFFERS: std::sync::Mutex<Vec<Entry>> = std::sync::Mutex::new(Vec::new());

    /// The pick's buffer for `stream` on the current device, made at its first use (a stage
    /// graph's capture runs in relaxed mode, where `cudaMalloc` is allowed). It needs no
    /// initial contents: the head writes the partials and their count, the argmax node the
    /// largest value and the probability CTAs' ticket. Launches on one stream run in order,
    /// so they share it; a captured graph replays on its stream.
    pub(in crate::model::core) fn buffer(stream: *mut c_void) -> Result<Buffer, String> {
        let dev = device().ok_or("q8_head pick: cudaGetDevice failed")?;
        let mut list = BUFFERS.lock().unwrap();
        if let Some(&(_, _, p, parts, sms)) = list
            .iter()
            .find(|&&(d, s, ..)| d == dev && s == stream as usize)
        {
            return Ok(Buffer {
                ptr: p as *mut c_void,
                parts,
                sms,
                stride: buffer_bytes(sms as usize) as i64,
            });
        }
        let sms = sm_count().ok_or("q8_head pick: a device without the pick")?;
        let mut p = std::ptr::null_mut();
        let stride = buffer_bytes(sms as usize);
        // SAFETY: a plain allocation on the current device, kept for the process.
        match unsafe { cudaMalloc(&mut p, SLICES * stride) } {
            0 => {
                let parts = sms * PARTS_PER_SM as i32;
                list.push((dev, stream as usize, p as usize, parts, sms));
                Ok(Buffer {
                    ptr: p,
                    parts,
                    sms,
                    stride: stride as i64,
                })
            }
            e => Err(format!("q8_head pick: cudaMalloc: CUDA error {e}")),
        }
    }

    /// Launches `func` on `grid` x `cols` CTAs of [`THREADS`] with `p` (`blockIdx.y`: the
    /// column).
    unsafe fn launch<P>(
        name: &str,
        func: *const c_void,
        grid: u32,
        cols: u32,
        p: &mut P,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut args = [(p as *mut P).cast::<c_void>()];
        let (g, b) = (
            Dim3 {
                x: grid,
                y: cols,
                z: 1,
            },
            Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            },
        );
        // SAFETY: `func` takes one argument of type `P`; the launch copies it.
        match unsafe { cudaLaunchKernel(func, g, b, args.as_mut_ptr(), 0, stream) } {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }

    pub(super) unsafe fn argmax(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let b = buffer(stream)?;
        let cols = columns(dst)?;
        let mut p = Argmax {
            pick: b.ptr,
            dst: dst.data as *mut i32,
            stride: b.stride,
        };
        // SAFETY: the kernel's host stub; device tensors of the stream's device.
        unsafe {
            launch(
                "q8_head_argmax",
                legion_q8_head_argmax_kernel,
                1,
                cols,
                &mut p,
                stream,
            )
        }
    }

    /// The columns of a pick node (`[n]`), at most the buffer's [`SLICES`].
    fn columns(dst: &ffi::ggml_tensor) -> Result<u32, String> {
        match dst.ne[0] {
            n @ 1..=MAX_COLS if n as usize <= SLICES => Ok(n as u32),
            n => Err(format!("q8_head pick: {n} columns")),
        }
    }

    pub(super) unsafe fn pmax(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let b = buffer(stream)?;
        let cols = columns(dst)?;
        // SAFETY: a node made by `head`: the logits and the picks, device tensors.
        unsafe {
            let (x, pick) = (&*dst.src[0], &*dst.src[1]);
            if x.ne[0] > i32::MAX as i64 {
                return Err("q8_head_pmax: too many logits".into());
            }
            let mut p = Pmax {
                x: x.data as *const f32,
                pick: pick.data as *const i32,
                dst: dst.data as *mut f32,
                buf: b.ptr,
                n: x.ne[0] as i32,
                x_ld: (x.nb[1] / 4) as i64,
                stride: b.stride,
            };
            launch(
                "q8_head_pmax",
                legion_q8_head_pmax_kernel,
                b.sms as u32,
                cols,
                &mut p,
                stream,
            )
        }
    }
}

#[cfg(feature = "cuda")]
pub(in crate::model::core) use cuda::buffer;

#[cfg(test)]
mod tests;
