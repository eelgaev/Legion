//! A Q8_0 matvec of one column with ggml-cuda's bits, read faster (#814): the MTP draft pass's LM
//! head at one drafting sequence on a V100.
//!
//! At C=1 every MTP draft pass of Qwen3.8-Flash-Next reads the whole LM head (248,320 x 2560
//! Q8_0, 680 MB) through ggml-cuda's `mul_mat`: `quantize_q8_1` and `mul_mat_vec_q<Q8_0, 1>`,
//! which on compute capability 7.x runs one 4-warp CTA per row and reads it at ~745 GB/s
//! (908 us, half the draft pass; nsys on an AC922 with 4 GPUs, #814), where the same head at 3 columns runs at
//! ~850. Here one external op ([`crate::ggml::ExternalOp`], `src/cuda/q8_head.cu`) replaces
//! the `mul_mat` node (the same inputs and output, so the graph keeps its nodes and layout): one
//! warp a row, resident CTAs walking the rows, 16-byte weight loads, the input quantized by
//! every lane for the blocks it reads, as ggml's `quantize_q8_1` does. Every sum is ggml's
//! expression over the same values in the same order, so every output has ggml's bits: the draft
//! tokens and their probabilities do not change.
//!
//! Taken where ggml would run that kernel the same way: a GPU of compute capability 7.x
//! ([`device_supported`]), a Q8_0 weight of a row length the kernel is built for ([`K`]) and one
//! F32 column ([`fits`]). On by default there; `legion serve --no-draft-head-mv` keeps ggml's
//! `mul_mat` ([`set_q8_head`]). The CPU implementation is the reference (the same quantization
//! and sums, in plain F32).
//!
//! The same kernel runs the one-column Q8_0 projections of [`K_OUT`] values a row ([`takes_out`],
//! [`mul_mat_out`]: an attention layer's output and a DeltaNet layer's `ssm_out`, 2,560 rows on
//! Qwen3.8-Flash-Next, in place of their `mul_mat` nodes), with the input quantized once per CTA
//! in shared memory, opt-in with `legion serve --q8-out-row-mv` (11-34% slower than ggml's
//! mmvq on a V100, so off by default).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static Q8_HEAD: ExternalOp = ExternalOp {
    name: "q8_head",
    cpu: cpu_q8_head,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_head),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The op for the 6,144-value projections ([`mul_mat_out`]): the same kernel and node, a name of
/// its own so that stage graphs may capture it ([`crate::model::core::stage_ops`]).
static Q8_HEAD_OUT: ExternalOp = ExternalOp {
    name: "q8_head_out",
    cpu: cpu_q8_head,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_head),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The op that also writes the draft pick's partials ([`pick`]): the same outputs.
static Q8_HEAD_PICK: ExternalOp = ExternalOp {
    name: "q8_head_pick",
    cpu: cpu_q8_head,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_head_pick),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Row lengths the op takes, those where it beats ggml on a V100 (`legion q8-head-bench`, #814):
/// 2,560 (80 blocks: ggml's 32-block steps leave a CTA's last step 3/8 idle; 955 -> 814 us on
/// the whole head) and 1,024 (32 blocks, one step a CTA: 415 -> 335 us). At 4,096 ggml reads
/// the head at ~880 GB/s and the op is 5% slower.
pub const K: [i64; 2] = [1024, 2560];

/// Fewest rows the op takes: a vocabulary-sized head. With fewer, too few rows per warp hide the
/// resident CTAs' start (10,240 rows: 42 -> 55 us); at 65,536 (`--spec-draft-vocab 65536`) it
/// still gains 4%.
pub const MIN_ROWS: i64 = 65536;

/// Bytes of a Q8_0 block: an F16 scale, 32 int8.
const Q8_0_BYTES: usize = 34;

/// Values per Q8_0 / Q8_1 block.
const GROUP: usize = 32;

/// ggml's mmvq K step for Q8_0 at one column on compute capability 7.x: 4 warps, each lane two
/// ints (8 values) of a block, so 32 blocks a step; virtual thread `t` of the CTA takes blocks
/// `t / 4 + 32 i`.
const CTA_THREADS: usize = 128;

/// Whether graphs built from now on may take the op ([`set_q8_head`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether [`crate::model::core::q8_group`]'s one-column matvecs run on the kernel ([`row_mv`]).
static ROW_MV: AtomicBool = AtomicBool::new(true);

/// The row length of the projections [`mul_mat_out`] takes: an attention layer's output and a
/// DeltaNet layer's `ssm_out` on Qwen3.8-Flash-Next (24 heads of 256, 48 value heads of 128).
/// The kernel keeps the input's 192 blocks in shared memory there, quantized once per CTA.
pub const K_OUT: i64 = 6144;

/// The fewest rows [`takes_out`] takes (Qwen3.8-Flash-Next's projections have 2,560).
pub const OUT_MIN_ROWS: i64 = 2048;

/// Whether graphs built from now on may run the 6,144-value projections on the op
/// ([`set_out_mv`]).
static OUT_MV: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on run one-column Q8_0 projections of [`K_OUT`] values a
/// row (an attention layer's output, a DeltaNet layer's `ssm_out`) on the op where
/// [`takes_out`] (off by default: slower than ggml's `mul_mat` on a V100; `legion serve
/// --q8-out-row-mv` takes it, the same bits).
pub fn set_out_mv(on: bool) {
    OUT_MV.store(on, Ordering::Relaxed);
}

/// [`row_mv`] launches so far (for the bench's check that the kernel ran).
static ROW_LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// How many times [`row_mv`] launched the kernel so far.
pub fn row_mv_launches() -> u64 {
    ROW_LAUNCHES.load(Ordering::Relaxed)
}

/// Most weights of one [`row_mv`] launch.
pub const XQ_WEIGHTS: usize = 4;

/// The fewest rows (all weights) [`row_mv`] takes: below that the head's kernel only ties
/// q8_group's (640 and 512 rows, `legion q8-head-bench --group`).
pub const XQ_MIN_ROWS: i64 = 4096;

/// Sets whether [`crate::model::core::q8_group`]'s one-column Q8_0 matvecs (a DeltaNet layer's
/// qkv and gate, an attention layer's Q, K and V at one token) run on this op's kernel where
/// [`row_mv_fits`] (on by default; `legion serve --no-q8-group-row-mv` keeps them on q8_group's
/// copy of ggml's mmvq, the same bits). Read at each launch.
pub fn set_row_mv(on: bool) {
    ROW_MV.store(on, Ordering::Relaxed);
}

/// Whether [`set_row_mv`] is on.
pub fn row_mv_on() -> bool {
    ROW_MV.load(Ordering::Relaxed)
}

/// Whether the kernel takes a one-column matvec of `k` values per row from Q8_1 blocks, for
/// weights at `addrs` with `row_bytes` between rows: `k` one of [`K`], 1 to [`XQ_WEIGHTS`]
/// weights, 16-byte aligned rows (the kernel's 16-byte loads).
pub fn row_mv_fits(k: i64, weights: &[(usize, usize)]) -> bool {
    K.contains(&k)
        && (1..=XQ_WEIGHTS).contains(&weights.len())
        && weights.iter().all(|&(addr, row_bytes)| {
            addr.is_multiple_of(16)
                && row_bytes.is_multiple_of(16)
                && row_bytes >= k as usize / GROUP * Q8_0_BYTES
        })
}

/// `ws[i] @ x` into `dsts[i]` for one column whose Q8_1 blocks are `xq` (ggml's `quantize_q8_1`
/// of `x`, `k` values), on `stream`, with ggml's one-column mmvq bits: [`row_mv_on`], a GPU
/// that runs the kernel ([`device_supported`]) and [`row_mv_fits`]. `ws[i]` is `(data, row bytes,
/// rows)`. `Ok(false)` where it does not apply (nothing launched: the caller runs its own).
///
/// # Safety
/// Device pointers of the stream's device: each weight `rows` rows of Q8_0 `k` values, each
/// `dsts[i]` `rows` F32, `xq` `k / 32` Q8_1 blocks.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn row_mv(
    ws: &[(*const c_void, usize, i64)],
    dsts: &[*mut f32],
    xq: *const c_void,
    k: i64,
    stream: *mut c_void,
) -> Result<bool, String> {
    let mut shapes = [(0usize, 0usize); XQ_WEIGHTS];
    for (s, &(w, ld, _)) in shapes.iter_mut().zip(ws) {
        *s = (w as usize, ld);
    }
    let shapes = &shapes[..ws.len().min(XQ_WEIGHTS)];
    let rows: i64 = ws.iter().map(|w| w.2).sum();
    if !row_mv_on()
        || ws.len() > XQ_WEIGHTS
        || rows < XQ_MIN_ROWS
        || !row_mv_fits(k, shapes)
        || !device_supported()
    {
        return Ok(false);
    }
    // SAFETY: as the caller promises.
    let taken = unsafe { cuda::launch_xq(ws, dsts, xq, k, stream) }?;
    if taken {
        ROW_LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    Ok(taken)
}

/// Sets whether graphs built from now on run a one-column Q8_0 MTP draft LM head on the op where
/// it applies (on by default; `legion serve --no-draft-head-mv` turns it off).
pub fn set_q8_head(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether `w @ x` takes the op: switched on, a GPU that runs the kernel ([`device_supported`]),
/// [`fits`], at least [`MIN_ROWS`] rows, a weight in one GPU's memory.
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && fits(w, x)
        && w.ne()[1] >= MIN_ROWS
        && !w.is_host()
        && !w.is_meta()
        && device_supported()
}

/// Whether a one-column `w @ x` takes [`mul_mat_out`]: switched on ([`set_out_mv`]), a GPU that
/// runs the kernel ([`device_supported`]), [`fits_out`], at least [`OUT_MIN_ROWS`] rows, a
/// weight in one GPU's memory.
pub fn takes_out(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    OUT_MV.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && fits_out(w, x)
        && w.ne()[1] >= OUT_MIN_ROWS
        && !w.is_host()
        && !w.is_meta()
        && device_supported()
}

/// [`fits`] for the 6,144-value projections: the same rules with `k` [`K_OUT`].
pub fn fits_out(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    fits_k(w, x, &[K_OUT])
}

/// Whether the op computes `w @ x` as ggml-cuda's one-column mmvq would: a Q8_0 matrix `w`
/// (`[k, rows]`, `k` one of [`K`], rows 16-byte aligned) and a contiguous F32 `x` of one column.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    fits_k(w, x, &K)
}

/// [`fits`] with the row lengths `ks`.
fn fits_k(w: Tensor<'_>, x: Tensor<'_>, ks: &[i64]) -> bool {
    let [k, rows, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let wnb = w.nb();
    w.ty() == types::Q8_0
        && x.ty() == types::F32
        && x.is_contiguous()
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && ks.contains(&k)
        && rows > 0
        && n == 1
        && wnb[0] == Q8_0_BYTES
        && wnb[1].is_multiple_of(16)
        && wnb[1] >= k as usize / GROUP * Q8_0_BYTES
}

/// Whether the current GPU runs the kernel (compute capability 7.x).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q8_head_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_q8_head_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Launches every op from now on with `(warps per CTA, most CTAs per SM, rows a warp step)`:
/// warps 4, 8 or 16, CTAs 0 for as many as fit, rows 1 or 2; `None` for the default (4, 0, 2).
/// For tests and tuning, the same bits either way. False where the kernel lacks the variant (or
/// without CUDA).
#[doc(hidden)]
pub fn force_variant(v: Option<(i32, i32, i32)>) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q8_head_force(warps: i32, ctas: i32, rows: i32) -> i32;
        }
        let (w, c, r) = v.unwrap_or((0, 0, 1));
        // SAFETY: stores the choice in an atomic.
        unsafe { legion_q8_head_force(w, c, r) == 0 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = v;
        false
    }
}

/// `w @ x` (F32 `[rows, 1]`) for `w` and `x` that [`fits`] takes, as one node with ggml's
/// `mul_mat`'s inputs and output.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q8_head: w {:?}, x {:?}", w.ne(), x.ne());
    ctx.external(&Q8_HEAD, types::F32, [w.ne()[1], 1, 1, 1], &[w, x])
}

/// `w @ x` (F32 `[rows, 1]`) for `w` and `x` that [`fits_out`] takes, as one node with ggml's
/// `mul_mat`'s inputs and output (so the graph's layout does not move), ggml's bits.
pub fn mul_mat_out<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits_out(w, x), "q8_head: w {:?}, x {:?}", w.ne(), x.ne());
    ctx.external(&Q8_HEAD_OUT, types::F32, [w.ne()[1], 1, 1, 1], &[w, x])
}

/// [`mul_mat`] that also leaves the draft pick's partials in its stream's buffer, for
/// [`pick::head`]'s argmax and probability nodes right after it.
fn mul_mat_pick<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q8_head: w {:?}, x {:?}", w.ne(), x.ne());
    ctx.external(&Q8_HEAD_PICK, types::F32, [w.ne()[1], 1, 1, 1], &[w, x])
}

/// ggml's `quantize_q8_1` of one 32-value block: the F16-rounded scale (as F32) and the int8
/// values.
fn quantize_block(x: &[f32]) -> (f32, [i8; GROUP]) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 127.0;
    let mut q = [0i8; GROUP];
    if amax != 0.0 {
        for (o, v) in q.iter_mut().zip(x) {
            *o = (v / d).round() as i8;
        }
    }
    // SAFETY: plain conversions.
    let dh = unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(d)) };
    (dh, q)
}

/// One output as ggml's one-column mmvq sums it: each virtual thread's products in its block
/// order (`d_w * d_x * idot` added with one rounding, as the GPU's fused multiply-add), the four
/// warps' sums of a lane in order, then the warp's butterfly sum.
fn row_dot(row: &[u8], xq: &[(f32, [i8; GROUP])]) -> f32 {
    let nb = xq.len();
    let mut lanes = [0f32; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let mut sum = 0f32;
        for t in (l..CTA_THREADS).step_by(32) {
            let mut tmp = 0f32;
            let q = t % 4;
            for b in (t / 4..nb).step_by(CTA_THREADS / 4) {
                let blk = &row[b * Q8_0_BYTES..][..Q8_0_BYTES];
                // SAFETY: a plain conversion.
                let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                let (dx, qx) = &xq[b];
                let idot: i32 = (8 * q..8 * q + 8)
                    .map(|i| i32::from(blk[2 + i] as i8) * i32::from(qx[i]))
                    .sum();
                tmp = (dw * dx).mul_add(idot as f32, tmp);
            }
            sum = if t < 32 { tmp } else { sum + tmp };
        }
        *lane = sum;
    }
    let mut width = 16;
    while width > 0 {
        let prev = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v += prev[l ^ width];
        }
        width /= 2;
    }
    lanes[0]
}

unsafe fn cpu_q8_head(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors: the weight and one F32 column.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows) = (w.ne[0] as usize, w.ne[1] as usize);
        let xs = std::slice::from_raw_parts(x.data as *const f32, k);
        let xq: Vec<_> = xs.chunks(GROUP).map(quantize_block).collect();
        let out = dst.data as *mut f32;
        let row_bytes = k / GROUP * Q8_0_BYTES;
        for r in (ith..rows).step_by(nth) {
            let row = std::slice::from_raw_parts((w.data as *const u8).add(r * w.nb[1]), row_bytes);
            *out.add(r) = row_dot(row, &xq);
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// `legion_q8_head` of `src/cuda/q8_head.cu`.
    #[repr(C)]
    pub(super) struct Params {
        w: *const c_void,
        w_ld: i64,
        rows: i64,
        x: *const f32,
        dst: *mut f32,
        k: i32,
        pick: *mut c_void,
        pick_cap: i32,
        xq: *const c_void,
        nw: i32,
        ws: [*const c_void; XQ_WEIGHTS],
        lds: [i64; XQ_WEIGHTS],
        dsts: [*mut f32; XQ_WEIGHTS],
        ends: [i64; XQ_WEIGHTS],
    }

    unsafe extern "C" {
        fn legion_q8_head_launch(p: *const Params, stream: *mut c_void) -> i32;
    }

    /// The decode matvecs ([`super::row_mv`]): `ws[i]` (`(data, row bytes, rows)`) times the
    /// Q8_1 blocks `xq` of `k` values into `dsts[i]`, one launch on `stream`. `Ok(false)` where
    /// the kernel does not take them (nothing launched).
    pub(super) unsafe fn launch_xq(
        ws: &[(*const c_void, usize, i64)],
        dsts: &[*mut f32],
        xq: *const c_void,
        k: i64,
        stream: *mut c_void,
    ) -> Result<bool, String> {
        if ws.is_empty() || ws.len() > XQ_WEIGHTS || ws.len() != dsts.len() {
            return Ok(false);
        }
        let mut p = Params {
            w: std::ptr::null(),
            w_ld: 0,
            rows: 0,
            x: std::ptr::null(),
            dst: std::ptr::null_mut(),
            k: k as i32,
            pick: std::ptr::null_mut(),
            pick_cap: 0,
            xq,
            nw: ws.len() as i32,
            ws: [std::ptr::null(); XQ_WEIGHTS],
            lds: [0; XQ_WEIGHTS],
            dsts: [std::ptr::null_mut(); XQ_WEIGHTS],
            ends: [0; XQ_WEIGHTS],
        };
        for (i, (&(w, ld, rows), &d)) in ws.iter().zip(dsts).enumerate() {
            p.rows += rows;
            p.ws[i] = w;
            p.lds[i] = ld as i64;
            p.dsts[i] = d;
            p.ends[i] = p.rows;
        }
        // SAFETY: device pointers of the stream's device, as the caller promises.
        match unsafe { legion_q8_head_launch(&p, stream) } {
            0 => Ok(true),
            1 => Ok(false),
            e => Err(format!("q8_head: CUDA error {e}")),
        }
    }

    /// `pick`: also write the draft pick's partials into the stream's buffer.
    pub(super) unsafe fn launch(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
        pick: bool,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat` or `mul_mat_pick`, device tensors of the stream's
        // device.
        unsafe {
            let (w, x) = (&*dst.src[0], &*dst.src[1]);
            let (pick, pick_cap) = if pick {
                let b = super::pick::buffer(stream)?;
                (b.ptr, b.parts)
            } else {
                (std::ptr::null_mut(), 0)
            };
            let p = Params {
                w: w.data as *const c_void,
                w_ld: w.nb[1] as i64,
                rows: w.ne[1],
                x: x.data as *const f32,
                dst: dst.data as *mut f32,
                k: w.ne[0] as i32,
                pick,
                pick_cap,
                xq: std::ptr::null(),
                nw: 0,
                ws: [std::ptr::null(); XQ_WEIGHTS],
                lds: [0; XQ_WEIGHTS],
                dsts: [std::ptr::null_mut(); XQ_WEIGHTS],
                ends: [0; XQ_WEIGHTS],
            };
            match legion_q8_head_launch(&p, stream) {
                0 => Ok(()),
                1 => Err("q8_head: shape, alignment or device the kernel does not take".into()),
                e => Err(format!("q8_head: CUDA error {e}")),
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q8_head(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::launch`.
    unsafe { cuda::launch(dst, stream, false) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q8_head_pick(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::launch`.
    unsafe { cuda::launch(dst, stream, true) }
}

pub mod bench;
pub mod pick;

#[cfg(test)]
mod tests;
