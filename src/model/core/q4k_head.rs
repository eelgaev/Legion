//! A Q4_K matvec of 1-8 columns with ggml-cuda's bits, read faster (#836): the MTP draft pass's
//! LM head on a GB10.
//!
//! qwen4exp's NVFP4 checkpoint on an integrated GPU drafts with a Q4_K copy of the head's first
//! `--spec-draft-vocab` rows (`--spec-draft-lm-head` auto, [`super::Model::quantize_draft_head`]:
//! 65,536 x 2560, 94 MB). Every draft pass reads it through ggml-cuda's `mul_mat`:
//! `quantize_q8_1` and `mul_mat_vec_q<Q4_K, n>`, which on a GB10 runs one 8-warp CTA per row at
//! one column (a CTA of 256 threads, 160 with a block to read) and reads the head at ~199 GB/s
//! (474 us a pass, a third of a C=1 draft pass; nsys on a DGX Spark, #836). Here one external op
//! ([`crate::ggml::ExternalOp`], `src/cuda/q4k_head.cu`) replaces the `mul_mat` node (the same
//! inputs and output, so the graph keeps its nodes and layout): one warp a row, resident CTAs
//! walking the rows, 16-byte weight loads, the input quantized by every CTA as ggml's
//! `quantize_q8_1` does. Every sum is ggml's expression over the same values in the same order,
//! so every output has ggml's bits: the draft tokens and their probabilities do not change.
//!
//! Taken where ggml would run that kernel the same way: a GB10 (compute capability 12.1, ggml's
//! `MMVQ_PARAMETERS_GB10` table, [`device_supported`]), a Q4_K weight of a row length the kernel
//! is built for ([`K`]), at least [`MIN_ROWS`] rows, and 1 to [`MAX_COLS`] F32 columns
//! ([`fits`]). On by default there; `legion serve --no-draft-head-q4k-mv` keeps ggml's `mul_mat`
//! ([`set_q4k_head`]). The CPU implementation is the reference (the same quantization and sums,
//! in plain F32; the GPU's fused multiply-adds as `mul_add`).
//!
//! The draft pick (#836, [`head`]): the op also keeps each CTA's best (value, tie rank) and
//! largest value of every column in q8_head's per-stream buffer (a slice per column), and
//! q8_head's pick nodes ([`super::q8_head::pick`]) give argmax_f32's pick and its probability
//! with the bits of ggml's soft_max + get_rows, instead of the split argmax, soft_max and
//! get_rows reading the 65,536 logits of each column three more times. On by default where the
//! op runs at 1 to [`super::q8_head::pick::MAX_COLS`] drafting sequences while ggml's soft_max
//! would take its cooperative kernel ([`pick_takes`]); `legion serve --no-draft-head-q4k-pick`
//! keeps the old tail ([`set_pick`]), `--no-draft-head-q4k-pick-cols` keeps it from two
//! sequences ([`set_pick_cols`]).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static Q4K_HEAD: ExternalOp = ExternalOp {
    name: "q4k_head",
    cpu: cpu_q4k_head,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q4k_head),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The op that also writes the draft pick's partials ([`head`]): the same outputs.
static Q4K_HEAD_PICK: ExternalOp = ExternalOp {
    name: "q4k_head_pick",
    cpu: cpu_q4k_head,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q4k_head_pick),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Row lengths the op takes: Qwen3.8-Flash-Next's 2,560 (10 Q4_K blocks).
pub const K: [i64; 1] = [2560];

/// Fewest rows the op takes: a draft head (`--spec-draft-vocab 65536` or the whole vocabulary).
/// With far fewer, the resident CTAs' start (each quantizes the input) is not hidden.
pub const MIN_ROWS: i64 = 32768;

/// Most columns the op takes: ggml-cuda runs MMVQ up to 8 (`MMVQ_MAX_BATCH_SIZE`).
pub const MAX_COLS: i64 = 8;

/// Bytes of a Q4_K block: F16 `d` and `dmin`, 12 bytes of 6-bit scales and mins, 128 of nibbles.
const Q4_K_BYTES: usize = 144;

/// Values per Q4_K block.
const QK_K: usize = 256;

/// Values per Q8_1 block of the input.
const GROUP: usize = 32;

/// Whether graphs built from now on may take the op ([`set_q4k_head`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on run a Q4_K MTP draft LM head of 1-8 columns on the op
/// where it applies (on by default; `legion serve --no-draft-head-q4k-mv` turns it off).
pub fn set_q4k_head(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on run the draft pick on the op's partials ([`set_pick`]).
static PICK: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on take [`head`] (the draft pick from the op's partials)
/// for a one-column Q4_K draft LM head on the op (on by default; `legion serve
/// --no-draft-head-q4k-pick` keeps the split argmax, soft_max and get_rows after the op).
pub fn set_pick(on: bool) {
    PICK.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on run the pick for 2 or more columns ([`set_pick_cols`]).
static PICK_COLS: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on take [`head`] for a Q4_K draft LM head of 2 to
/// [`super::q8_head::pick::MAX_COLS`] columns on the op (on by default; `legion serve
/// --no-draft-head-q4k-pick-cols` keeps the split argmax, soft_max and get_rows there, one
/// column still takes the pick).
pub fn set_pick_cols(on: bool) {
    PICK_COLS.store(on, Ordering::Relaxed);
}

/// Whether the draft pass's head `w @ x` takes [`head`]: the head takes the op ([`takes`]), the
/// pick is switched on for its column count ([`set_pick`], [`set_pick_cols`]), ggml's soft_max
/// of the logits would run its cooperative kernel ([`super::q8_head::pick::coop`]: at most 7
/// columns of a 65,536-row head), and the current GPU runs that kernel on at most
/// [`super::q8_head::pick::MAX_SMS`] SMs ([`super::q8_head::pick::device_supported`]).
pub fn pick_takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let n = x.ne()[1];
    PICK.load(Ordering::Relaxed)
        && (n == 1 || PICK_COLS.load(Ordering::Relaxed))
        && super::q8_head::pick::coop(w.ne()[1], n)
        && takes(w, x)
        && super::q8_head::pick::device_supported()
}

/// The draft head and pick for `w` and `x` that [`fits`] (1 to
/// [`super::q8_head::pick::MAX_COLS`] columns `n`): the logits (F32 `[rows, n]`, [`mul_mat`]'s),
/// each column's pick (I32 `[n]`, `cluster_select::argmax`'s) and its probability (F32 `[n]`,
/// ggml's `soft_max` read at the pick), with q8_head's pick nodes
/// ([`super::q8_head::pick::tail`]). The three nodes must run together, in this order.
pub fn head<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let n = x.ne()[1];
    assert!(
        fits(w, x) && n <= super::q8_head::pick::MAX_COLS,
        "q4k_head pick: w {:?}, x {:?}",
        w.ne(),
        x.ne()
    );
    let logits = ctx.external(&Q4K_HEAD_PICK, types::F32, [w.ne()[1], n, 1, 1], &[w, x]);
    let (argmax, pmax) = super::q8_head::pick::tail(ctx, logits);
    (logits, argmax, pmax)
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

/// Whether the op computes `w @ x` as ggml-cuda's mmvq would: a Q4_K matrix `w` (`[k, rows]`,
/// `k` one of [`K`], rows 16-byte aligned) and a contiguous F32 `x` of 1 to [`MAX_COLS`]
/// columns.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [k, rows, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let wnb = w.nb();
    w.ty() == types::Q4_K
        && x.ty() == types::F32
        && x.is_contiguous()
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && K.contains(&k)
        && rows > 0
        && (1..=MAX_COLS).contains(&n)
        && wnb[0] == Q4_K_BYTES
        && wnb[1].is_multiple_of(16)
        && wnb[1] >= k as usize / QK_K * Q4_K_BYTES
}

/// Whether the current GPU runs the kernel (a GB10 in a build for compute capability 12.1).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q4k_head_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_q4k_head_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Launches every op from now on with `(warps per CTA, most CTAs per SM, rows a warp step)`:
/// warps 4, 8 or 16, CTAs 0 for as many as fit, rows 1, 2 or 4; `None` for the default. For
/// tests and tuning, the same bits either way. False where the kernel lacks the variant (or
/// without CUDA).
#[doc(hidden)]
pub fn force_variant(v: Option<(i32, i32, i32)>) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q4k_head_force(warps: i32, ctas: i32, rows: i32) -> i32;
        }
        let (w, c, r) = v.unwrap_or((0, 0, 1));
        // SAFETY: stores the choice in an atomic.
        unsafe { legion_q4k_head_force(w, c, r) == 0 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = v;
        false
    }
}

/// `w @ x` (F32 `[rows, n]`) for `w` and `x` that [`fits`] takes, as one node with ggml's
/// `mul_mat`'s inputs and output.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q4k_head: w {:?}, x {:?}", w.ne(), x.ne());
    ctx.external(&Q4K_HEAD, types::F32, [w.ne()[1], x.ne()[1], 1, 1], &[w, x])
}

/// ggml's warps per CTA for Q4_K at `n` columns on a GB10 for rows of `nb` blocks
/// (`calc_nwarps` with `MMVQ_PARAMETERS_GB10`): 4 for 1-4 columns, 2 for 5-8, doubled for one
/// column when that halves the K loop with at most an eighth of it idle (`should_halve_iters`)
/// and the row is not short (`should_use_small_k`). Only this decides a row's sum order.
pub(crate) fn ggml_warps(n: usize, nb: usize) -> usize {
    let generic = if n <= 4 { 4 } else { 2 };
    if n != 1 {
        return generic;
    }
    // Blocks a warp takes a K step: two ints a lane, 16 lanes a block.
    let step = generic * 2;
    let small_k = nb < step;
    let iters = nb.div_ceil(step);
    let wide = nb.div_ceil(2 * step);
    let idle = 2 * wide - iters;
    if !small_k && idle * 8 <= 2 * wide {
        2 * generic
    } else {
        generic
    }
}

/// ggml's `quantize_q8_1` of one 32-value block: the F16-rounded scale (as F32) and the int8
/// values (its F16 sum is not read by the Q4_K dot product).
pub(crate) fn quantize_block(x: &[f32]) -> (f32, [i8; GROUP]) {
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

fn f16(b: &[u8]) -> f32 {
    // SAFETY: a plain conversion.
    unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([b[0], b[1]])) }
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// `dp4a`: the four signed byte products of `a` and `b`, added to `c`.
fn dp4a(a: i32, b: i32, c: i32) -> i32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    (0..4).fold(c, |s, i| s + i32::from(a[i] as i8) * i32::from(b[i] as i8))
}

/// ggml's `vec_dot_q4_K_q8_1` (vecdotq.cuh) of the Q4_K block `blk` and the input's 8 blocks
/// `xq` under it, for `iqs` (0, 2, .., 30): two ints of nibbles of two of the block's 32-value
/// groups.
pub(crate) fn vec_dot(blk: &[u8], xq: &[(f32, [i8; GROUP])], iqs: usize) -> f32 {
    let bq8_offset = 2 * ((iqs / 2) / 4);
    let qs = &blk[16..];
    let q4 = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    let v = [i32_at(qs, q4), i32_at(qs, q4 + 16)];
    let scales = |i: usize| u32::from(u16::from_le_bytes([blk[4 + 2 * i], blk[5 + 2 * i]]));
    let j = bq8_offset / 2;
    let jm = j & 1;
    let (s0, s2, s4) = (scales(jm), scales(jm + 2), scales(jm + 4));
    let (aux0, aux1) = if j < 2 {
        (s0 & 0x3f3f, s2 & 0x3f3f)
    } else {
        (
            (s4 & 0x0f0f) | ((s0 & 0xc0c0) >> 2),
            ((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2),
        )
    };
    let sc = [(aux0 & 0xff) as i32, (aux0 >> 8) as i32];
    let m = [(aux1 & 0xff) as i32, (aux1 >> 8) as i32];
    let int8 = |q: &[i8; GROUP], at: usize| {
        i32::from_le_bytes([
            q[at] as u8,
            q[at + 1] as u8,
            q[at + 2] as u8,
            q[at + 3] as u8,
        ])
    };
    let o = 4 * ((iqs / 2) % 4);
    let (mut sumf_d, mut sumf_m) = (0f32, 0f32);
    for (i, ((d8, q8), (s, mn))) in xq[bq8_offset..bq8_offset + 2]
        .iter()
        .zip(sc.iter().zip(&m))
        .enumerate()
    {
        let u = [int8(q8, o), int8(q8, o + 16)];
        let v0 = (v[0] >> (4 * i)) & 0x0F0F_0F0F;
        let v1 = (v[1] >> (4 * i)) & 0x0F0F_0F0F;
        let dot1 = dp4a(v1, u[1], dp4a(v0, u[0], 0));
        let dot2 = dp4a(0x0101_0101, u[1], dp4a(0x0101_0101, u[0], 0));
        sumf_d = d8.mul_add((dot1 * s) as f32, sumf_d);
        sumf_m = d8.mul_add((dot2 * mn) as f32, sumf_m);
    }
    let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
    d.mul_add(sumf_d, -(dmin * sumf_m))
}

/// One output as ggml's mmvq sums it with `g` warps a CTA: each virtual thread's dot products in
/// its block order, the warps' sums of a lane in order, then the warp's butterfly sum.
pub(crate) fn row_dot(row: &[u8], xq: &[(f32, [i8; GROUP])], g: usize) -> f32 {
    let nb = row.len() / Q4_K_BYTES;
    let mut lanes = [0f32; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        let mut sum = 0f32;
        for w in 0..g {
            let t = l + 32 * w;
            let mut tmp = 0f32;
            for kbx in (t / 16..nb).step_by(2 * g) {
                let blk = &row[kbx * Q4_K_BYTES..][..Q4_K_BYTES];
                tmp += vec_dot(blk, &xq[kbx * QK_K / GROUP..], 2 * (t % 16));
            }
            sum = if w == 0 { tmp } else { sum + tmp };
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

unsafe fn cpu_q4k_head(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors: the weight and the F32 columns.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows, n) = (w.ne[0] as usize, w.ne[1] as usize, x.ne[1] as usize);
        let g = ggml_warps(n, k / QK_K);
        let xq: Vec<Vec<_>> = (0..n)
            .map(|j| {
                let col = std::slice::from_raw_parts((x.data as *const f32).add(j * k), k);
                col.chunks(GROUP).map(quantize_block).collect()
            })
            .collect();
        let out = dst.data as *mut f32;
        let row_bytes = k / QK_K * Q4_K_BYTES;
        for r in (ith..rows).step_by(nth) {
            let row = std::slice::from_raw_parts((w.data as *const u8).add(r * w.nb[1]), row_bytes);
            for (j, xq) in xq.iter().enumerate() {
                *out.add(j * rows + r) = row_dot(row, xq, g);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// `legion_q4k_head` of `src/cuda/q4k_head.cu`.
    #[repr(C)]
    struct Params {
        w: *const c_void,
        w_ld: i64,
        rows: i64,
        x: *const f32,
        x_ld: i64,
        dst: *mut f32,
        dst_ld: i64,
        k: i32,
        n: i32,
        pick: *mut c_void,
        pick_cap: i32,
        pick_stride: i64,
    }

    unsafe extern "C" {
        fn legion_q4k_head_launch(p: *const Params, stream: *mut c_void) -> i32;
    }

    /// `pick`: also write the draft pick's partials into the stream's buffer.
    pub(super) unsafe fn launch(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
        pick: bool,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat` or `head`, device tensors of the stream's device.
        unsafe {
            let (w, x) = (&*dst.src[0], &*dst.src[1]);
            let (pick, pick_cap, pick_stride) = if pick {
                let b = crate::model::core::q8_head::pick::buffer(stream)?;
                (b.ptr, b.parts, b.stride)
            } else {
                (std::ptr::null_mut(), 0, 0)
            };
            let p = Params {
                w: w.data as *const c_void,
                w_ld: w.nb[1] as i64,
                rows: w.ne[1],
                x: x.data as *const f32,
                x_ld: (x.nb[1] / 4) as i64,
                dst: dst.data as *mut f32,
                dst_ld: (dst.nb[1] / 4) as i64,
                k: w.ne[0] as i32,
                n: x.ne[1] as i32,
                pick,
                pick_cap,
                pick_stride,
            };
            match legion_q4k_head_launch(&p, stream) {
                0 => Ok(()),
                1 => Err("q4k_head: shape, alignment or device the kernel does not take".into()),
                e => Err(format!("q4k_head: CUDA error {e}")),
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q4k_head(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::launch`.
    unsafe { cuda::launch(dst, stream, false) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q4k_head_pick(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::launch`.
    unsafe { cuda::launch(dst, stream, true) }
}

pub mod bench;

#[cfg(test)]
mod tests;
