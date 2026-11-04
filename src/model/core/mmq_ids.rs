//! Expert-grid matmuls with ggml's MMQ bits (#836): `mul_mat_id` of Q4_K and Q5_1 expert stacks
//! for steps of [`MIN_TOKENS`]..[`MAX_TOKENS`] tokens on a GB10, as an external op running
//! `src/cuda/mmq_ids.cu` in place of ggml-cuda's MMQ.
//!
//! In those steps (16-sequence decode with MTP drafts) ggml's MMQ streams the experts at about
//! 200 GB/s: a CTA per (expert, 128 rows) tile, the tile loaded to shared memory and multiplied on
//! tensor cores against at least as many columns as the step has tokens, though an expert sees
//! one or two of them. The op computes each output with MMQ's own chain (the activations
//! quantized as ggml quantizes them, then per 32-value chunk in K order the same two
//! multiply-adds of the same factors; see the kernel file), a thread per weight row streaming
//! the row from memory with many loads in flight, so its result is ggml's bit for bit.
//!
//! Opt-in (`legion serve --expert-mmq`, [`set_mmq_ids`]), on a compute capability 12.x GPU or a
//! V100 ([`set_gpu`]): `legion mmq-ids-bench` on a GB10 found every output equal to ggml's bits but
//! the op 8-19% slower than MMQ on gate/up (Q4_K) and 21-46% slower on down (Q5_1) at 9-63 tokens
//! (#836), so ggml's MMQ stays the default. On a V100 ggml's MMQ has no int8 tensor cores and
//! takes its dp4a path, a different chain, which the kernel computes there instead (#814); the
//! expert-cache path (tiered experts, `--experts cache`) takes the op too. The CPU implementation
//! is a plain f32 reference, for tests of the wiring.
//!
//! The node has `mul_mat_id`'s shape and sources (weights, activations, ids) and nothing else:
//! the quantized activations and the grouping plan go to a scratch buffer of the GPU outside the
//! graphs ([`SCRATCH_BYTES`] per device and stream, as ggml's MMQ takes its pool). A scratch
//! tensor in the graph moved the step graph's memory layout, and ggml-cuda turns some of its
//! fusions on or off by whether outputs overlap inputs (#819), so the greedy text moved although
//! every output of the op had ggml's bits (#814).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static MMQ_IDS: ExternalOp = ExternalOp {
    name: "mmq_ids",
    cpu: cpu_mmq_ids,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmq_ids),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest tokens of a step the op takes: ggml-cuda keeps steps of up to 8 tokens of these types
/// on its MMVQ (its own bits), and MMQ takes the bigger ones.
pub const MIN_TOKENS: i64 = 9;

/// Steps of this many tokens or more stay on ggml (MMQ with its pool scratch, or the fused MoE op
/// from 64 tokens up: [`crate::ggml::stage_graph::MMQ_IDS_MAX_TOKENS`]).
pub const MAX_TOKENS: i64 = 64;

/// Bytes of a scratch buffer (one per device and stream): the quantized activations and the plan
/// of the widest step the op takes ([`scratch_layout`]; 0.5 MB for Qwen3.8-Flash-Next's down
/// projection at 63 tokens). Bigger steps stay on ggml.
pub(crate) const SCRATCH_BYTES: usize = 4 << 20;

static ON: AtomicBool = AtomicBool::new(false);
static GPU: AtomicU8 = AtomicU8::new(CHAIN_NONE);

/// [`GPU`]: no GPU the op knows MMQ's chain on ...
const CHAIN_NONE: u8 = 0;
/// ... a compute capability 12.x GPU (GB10), where MMQ runs on int8 tensor cores ...
const CHAIN_MMA: u8 = 1;
/// ... a V100 (7.0), where MMQ takes its dp4a path (#814).
const CHAIN_DP4A: u8 = 2;

/// Whether graphs built from now on take the op where it fits (`legion serve --expert-mmq`) or
/// ggml's MMQ (the default).
pub fn set_mmq_ids(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU (`cc`: compute capability): the op is used on 12.x (GB10) and 7.0 (V100),
/// each with MMQ's chain on that GPU, where its bits were checked against ggml's MMQ.
pub fn set_gpu(cc: Option<(i32, i32)>) {
    GPU.store(chain_for(cc), Ordering::Relaxed);
}

/// Which of MMQ's chains the op computes on a GPU of compute capability `cc`.
fn chain_for(cc: Option<(i32, i32)>) -> u8 {
    match cc {
        Some((12, _)) => CHAIN_MMA,
        Some((7, 0)) => CHAIN_DP4A,
        _ => CHAIN_NONE,
    }
}

/// Whether the op takes `mul_mat_id(w, x, ids)`: switched on, a 12.x GPU, and [`fits_shape`].
pub fn usable(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && GPU.load(Ordering::Relaxed) != CHAIN_NONE
        && cfg!(feature = "cuda")
        && fits_shape(w, x, ids)
}

/// Whether the op can compute `mul_mat_id(w, x, ids)`: a Q4_K or Q5_1 expert stack in one
/// device's memory (rows of a multiple of 256 or 128 values, a multiple of 128 rows), F32
/// activations shared by the slots (`[k, 1, n]`) or one per slot (`[k, n_used, n]`) with 16-byte
/// aligned rows, I32 ids `[n_used, n]`, [`MIN_TOKENS`] <= n < [`MAX_TOKENS`], and a scratch of at
/// most [`SCRATCH_BYTES`].
pub fn fits_shape(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    let [k, rows, n_expert, w3] = w.ne();
    let [xk, n_x, n, x3] = x.ne();
    let [n_used, n_ids, i2, i3] = ids.ne();
    let wnb = w.nb();
    let xnb = x.nb();
    let row_mult = match w.ty() {
        t if t == types::Q4_K => 256,
        t if t == types::Q5_1 => 128,
        _ => return false,
    };
    k > 0
        && k % row_mult == 0
        && xk == k
        && rows > 0
        && rows % 128 == 0
        && w3 == 1
        && x3 == 1
        && (n_x == 1 || n_x == n_used)
        && n_ids == n
        && i2 == 1
        && i3 == 1
        && (MIN_TOKENS..MAX_TOKENS).contains(&n)
        && (1..=4096).contains(&n_expert)
        && x.ty() == types::F32
        && ids.ty() == types::I32
        && xnb[0] == 4
        && xnb[1].is_multiple_of(16)
        && xnb[2].is_multiple_of(16)
        && wnb[1].is_multiple_of(16)
        && wnb[2].is_multiple_of(16)
        && !w.is_host()
        && !w.is_meta()
        && scratch_layout(k as usize, n_x as usize, n_used as usize, n as usize).2 <= SCRATCH_BYTES
}

/// The scratch layout for `n_x * n` activation rows of `k` values and `n_used * n` pairs: the
/// offsets of the half2 scales and of the grouping plan, and the total bytes.
pub(crate) fn scratch_layout(
    k: usize,
    n_x: usize,
    n_used: usize,
    n: usize,
) -> (usize, usize, usize) {
    let rows = n_x * n;
    let pairs = n_used * n;
    let ds_off = (rows * k).next_multiple_of(16);
    let plan_off = (ds_off + rows * (k / 32) * 4).next_multiple_of(16);
    let plan_ints = 1 + pairs + (pairs + 1) + pairs;
    (ds_off, plan_off, plan_off + plan_ints * 4)
}

/// `mul_mat_id(w, x, ids)` (F32 `[rows, n_used, n]`) on the op; the caller checked [`usable`].
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    let rows = w.ne()[1];
    let n = x.ne()[2];
    let n_used = ids.ne()[0];
    ctx.external(&MMQ_IDS, types::F32, [rows, n_used, n, 1], &[w, x, ids])
}

struct Operands {
    ty: ffi::ggml_type,
    w: *const u8,
    w_nb: [usize; 4],
    x: *const u8,
    x_nb: [usize; 4],
    n_x: usize,
    ids: *const u8,
    ids_nb1: usize,
    dst: *mut u8,
    dst_nb: [usize; 4],
    k: usize,
    rows: usize,
    n_used: usize,
    n_tok: usize,
    n_expert: usize,
}

impl Operands {
    /// # Safety
    /// `dst` is a node made by [`mul_mat_id`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        unsafe {
            let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
            Self {
                ty: w.type_,
                w: w.data as *const u8,
                w_nb: w.nb,
                x: x.data as *const u8,
                x_nb: x.nb,
                n_x: x.ne[1] as usize,
                ids: ids.data as *const u8,
                ids_nb1: ids.nb[1],
                dst: dst.data as *mut u8,
                dst_nb: dst.nb,
                k: w.ne[0] as usize,
                rows: w.ne[1] as usize,
                n_used: ids.ne[0] as usize,
                n_tok: ids.ne[1] as usize,
                n_expert: w.ne[2] as usize,
            }
        }
    }
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`], all tensors in host memory.
unsafe fn cpu_mmq_ids(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id`, all tensors in host memory.
    let o = unsafe { Operands::of(dst) };
    let row_bytes = crate::ggml::row_size(o.ty, o.k);
    for pair in (ith..o.n_used * o.n_tok).step_by(nth) {
        let (t, j) = (pair / o.n_used, pair % o.n_used);
        // SAFETY: in-bounds reads of ids, x and the expert's rows; writes to dst's row.
        unsafe {
            let e = *(o.ids.add(t * o.ids_nb1 + j * 4) as *const i32);
            let Some(e) = usize::try_from(e).ok().filter(|&e| e < o.n_expert) else {
                continue;
            };
            let out = o.dst.add(t * o.dst_nb[2] + j * o.dst_nb[1]) as *mut f32;
            let xj = if o.n_x == 1 { 0 } else { j };
            let x = std::slice::from_raw_parts(
                o.x.add(t * o.x_nb[2] + xj * o.x_nb[1]) as *const f32,
                o.k,
            );
            for r in 0..o.rows {
                let row =
                    std::slice::from_raw_parts(o.w.add(e * o.w_nb[2] + r * o.w_nb[1]), row_bytes);
                let w = crate::ggml::dequantize(o.ty, row, o.k);
                *out.add(r) = w.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchParams {
    w: *const u8,
    w_nb1: i64,
    w_nb2: i64,
    x: *const u8,
    x_nb1: i64,
    x_nb2: i64,
    n_x: i32,
    ids: *const u8,
    ids_nb1: i64,
    dst: *mut u8,
    dst_nb1: i64,
    dst_nb2: i64,
    k: i32,
    rows: i32,
    n_used: i32,
    n_tok: i32,
    n_expert: i32,
    scratch: *mut u8,
    ds_off: i64,
    plan_off: i64,
    dp4a: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_mmq_ids_launch(ty: i32, p: *const LaunchParams, stream: *mut c_void) -> i32;
    fn cudaGetDevice(device: *mut std::os::raw::c_int) -> std::os::raw::c_int;
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> std::os::raw::c_int;
    fn cudaGetLastError() -> std::os::raw::c_int;
}

/// The scratch buffer of each (device, stream), allocated on first use and kept: captured graphs
/// replay its address. The op's nodes on one stream run one after the other, so they share it.
#[cfg(feature = "cuda")]
static SCRATCH: std::sync::Mutex<Option<std::collections::HashMap<(i32, usize), usize>>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "cuda")]
fn scratch(stream: *mut c_void) -> Result<*mut u8, String> {
    let mut dev = 0;
    // SAFETY: plain queries and an allocation on the current device (the stream's).
    unsafe {
        if cudaGetDevice(&mut dev) != 0 {
            cudaGetLastError();
            return Err("mmq_ids: no current device".into());
        }
        let mut map = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(std::collections::HashMap::new);
        if let Some(&p) = map.get(&(dev, stream as usize)) {
            return Ok(p as *mut u8);
        }
        let mut p = std::ptr::null_mut();
        if cudaMalloc(&mut p, SCRATCH_BYTES) != 0 {
            cudaGetLastError();
            return Err(format!("mmq_ids: cudaMalloc of {SCRATCH_BYTES} bytes"));
        }
        map.insert((dev, stream as usize), p as usize);
        Ok(p as *mut u8)
    }
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`], its tensors in device memory; `stream` is the
/// compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_mmq_ids(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat_id`.
    let o = unsafe { Operands::of(dst) };
    let (ds_off, plan_off, bytes) = scratch_layout(o.k, o.n_x, o.n_used, o.n_tok);
    if bytes > SCRATCH_BYTES {
        return Err(format!(
            "mmq_ids launch refused: {bytes} bytes of scratch (at most {SCRATCH_BYTES})"
        ));
    }
    let s = scratch(stream)?;
    let a16 = |p: *const u8| (p as usize).is_multiple_of(16);
    if !a16(o.w) || !a16(o.x) || !a16(s) {
        return Err(format!(
            "mmq_ids launch refused: weights, activations or scratch not 16-byte aligned \
             (w {:p}, x {:p}, scratch {:p})",
            o.w, o.x, s
        ));
    }
    let p = LaunchParams {
        w: o.w,
        w_nb1: o.w_nb[1] as i64,
        w_nb2: o.w_nb[2] as i64,
        x: o.x,
        x_nb1: o.x_nb[1] as i64,
        x_nb2: o.x_nb[2] as i64,
        n_x: o.n_x as i32,
        ids: o.ids,
        ids_nb1: o.ids_nb1 as i64,
        dst: o.dst,
        dst_nb1: o.dst_nb[1] as i64,
        dst_nb2: o.dst_nb[2] as i64,
        k: o.k as i32,
        rows: o.rows as i32,
        n_used: o.n_used as i32,
        n_tok: o.n_tok as i32,
        n_expert: o.n_expert as i32,
        scratch: s,
        ds_off: ds_off as i64,
        plan_off: plan_off as i64,
        dp4a: i32::from(GPU.load(Ordering::Relaxed) == CHAIN_DP4A),
    };
    // SAFETY: operands checked by `fits_shape` when the node was built and above.
    match unsafe { legion_mmq_ids_launch(o.ty as i32, &p, stream) } {
        0 => Ok(()),
        e => Err(format!(
            "CUDA error {e} (mmq_ids {} k {} rows {}, {} x {} pairs)",
            crate::ggml::type_name(o.ty),
            o.k,
            o.rows,
            o.n_used,
            o.n_tok
        )),
    }
}
