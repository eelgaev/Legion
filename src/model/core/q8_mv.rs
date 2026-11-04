//! A Q8_0 matmul of 9-16 columns with ggml-cuda's MMQ bits, read faster (#836 part 13): the
//! MTP draft pass's LM head on a GB10.
//!
//! An MTP draft pass at 16 sequences runs the draft LM head (the first `--spec-draft-vocab`
//! rows of `output.weight`, 65,536 x 2560 Q8_0, 178 MB) through ggml-cuda's `mul_mat`: its
//! quantization of the input (`quantize_mmq_q8_1`) and MMQ, which loads each tile of the weight
//! with 2-byte loads between barriers and reads it at ~188-193 GB/s there, the pass's kernel
//! furthest from the GPU's ~235 (nsys, part 13). Here the input is quantized by ggml's own kernel
//! ([`shared_q8::blocks`], the same bytes) and an external op ([`crate::ggml::ExternalOp`],
//! `src/cuda/q8_mv.cu`) streams the weight through shared memory with `cp.async` (8 blocks of
//! each of a warp's 32 rows per stage, 16-byte copies), adding each
//! row's 32-value groups in ggml's order with ggml's expression (`(float) idot * dA * dB`): every
//! output has ggml's bits, so the draft tokens and their probabilities do not change.
//!
//! The bits hold where ggml runs MMQ without its stream-k fix-up: then each output is one
//! CTA's sum over the whole row ([`ggml_whole_rows`]). ggml takes MMVQ below 9 columns
//! (`MMVQ_MAX_BATCH_SIZE`), a different sum, so those steps keep ggml's path.
//!
//! Off by default: on a GB10 (compute capability 12.x, [`mmq_preq::gb10`], the only GPU it
//! takes) it has ggml's bits at every shape measured, but reads the head at ~172-178 GB/s
//! against MMQ's ~190-193 (65,536 and 248,320 rows, 9-16 columns: 7-10% slower, about as fast
//! at any of the stage shapes `legion q8-mv-bench` compares). `legion serve --draft-head-q8-mv`
//! takes it, for comparisons and further work. The CPU implementation is the reference (the
//! same sums in plain F32).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::{mmq_preq, shared_q8};
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static Q8_MV: ExternalOp = ExternalOp {
    name: "q8_mv",
    cpu: cpu_q8_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the op takes: ggml-cuda runs MMVQ up to 8.
pub const MIN_COLS: i64 = 9;

/// Most columns the op takes (one MMQ tile of 16 columns in ggml).
pub const MAX_COLS: i64 = 16;

/// Rows of an MMQ tile for Q8_0 at 16 columns on a GB10 (`mmq-config-ampere.cuh`: `I` = 128).
const MMQ_ROWS: i64 = 128;

/// Bytes of a Q8_0 block: an F16 scale, 32 int8.
const Q8_0_BYTES: usize = 34;

/// Values per Q8_0 block (and per scale of the activations).
const GROUP: usize = 32;

/// Whether graphs built from now on may take the op ([`set_q8_mv`]).
static ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on run the MTP draft LM head of 9-16 columns on the op
/// where it applies (`legion serve --draft-head-q8-mv`); off by default: ggml's `mul_mat` reads
/// the head faster on a GB10 (see the module docs).
pub fn set_q8_mv(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether ggml-cuda's MMQ computes every output of a `rows`-row matmul of up to 16 columns as
/// one CTA's sum over the whole row on a GPU of `sms` SMs: its stream-k launch
/// (`launch_mul_mat_q`) gives each CTA one tile when the tiles fill 90% of their waves, else
/// one CTA per SM, and adds partial sums (the fix-up) unless the tiles divide evenly.
pub fn ggml_whole_rows(rows: i64, sms: i64) -> bool {
    if rows <= 0 || sms <= 0 {
        return false;
    }
    let tiles = (rows + MMQ_ROWS - 1) / MMQ_ROWS;
    let waves = (tiles + sms - 1) / sms;
    let grid = if 100 * tiles / (sms * waves) >= 90 {
        tiles
    } else {
        sms
    };
    tiles % grid == 0
}

/// Whether `w @ x` takes the op: switched on, a compute capability 12.x GPU
/// ([`mmq_preq::gb10`]) that runs the kernel ([`device_supported`]), [`fits`], a weight in one
/// GPU's memory, and a row count ggml's MMQ sums whole ([`ggml_whole_rows`]).
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && mmq_preq::gb10()
        && fits(w, x)
        && !w.is_host()
        && !w.is_meta()
        && device_supported()
        && ggml_whole_rows(w.ne()[1], sms())
}

/// Whether the op computes `w @ x` as ggml-cuda's MMQ would: a Q8_0 matrix `w` (`[k, rows]`,
/// `k` a multiple of 512, `rows` of 32, rows 4-byte aligned) and a contiguous F32 `x`
/// (`[k, n]`) of [`MIN_COLS`] to [`MAX_COLS`] columns.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
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
        && k % 512 == 0
        && rows % 32 == 0
        && wnb[0] == Q8_0_BYTES
        && wnb[1].is_multiple_of(4)
        && wnb[1] >= k as usize / GROUP * Q8_0_BYTES
        && (MIN_COLS..=MAX_COLS).contains(&n)
}

/// Whether the current GPU runs the kernel (compute capability 8.0+, the kernel in this build).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q8_mv_supported() -> i32;
        }
        // SAFETY: queries the current device.
        unsafe { legion_q8_mv_supported() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// The current GPU's SM count (ggml-cuda's `nsm`); 0 when unknown or without CUDA.
fn sms() -> i64 {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q8_mv_sms() -> i32;
        }
        // SAFETY: queries the current device.
        i64::from(unsafe { legion_q8_mv_sms() })
    }
    #[cfg(not(feature = "cuda"))]
    0
}

/// Launches every op from now on with `(blocks per stage, warps per CTA, stages)` of (8, 4, 2),
/// (8, 2, 2), (4, 8, 2), (4, 4, 2), (2, 8, 2) or (2, 16, 2), or the default choice for `None`: for tests and tuning, the same bits either way. False
/// where the kernel lacks the variant (or without CUDA).
#[doc(hidden)]
pub fn force_variant(v: Option<(i32, i32, i32)>) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_q8_mv_force(cb: i32, warps: i32, stages: i32) -> i32;
        }
        let (cb, w, s) = v.unwrap_or((0, 0, 0));
        // SAFETY: stores the choice in an atomic.
        unsafe { legion_q8_mv_force(cb, w, s) == 0 }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = v;
        false
    }
}

/// `w @ x` (F32 `[rows, n]`) for `w` and `x` that [`fits`] takes: ggml's quantization of `x`,
/// then the op.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q8_mv: w {:?}, x {:?}", w.ne(), x.ne());
    let xq = shared_q8::blocks(ctx, x);
    ctx.external(&Q8_MV, types::F32, [w.ne()[1], x.ne()[1], 1, 1], &[w, xq])
}

unsafe fn cpu_q8_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors: the weight, the blocks of `x` in MMQ's
    // layout (`mmq_preq::q8_1_bytes`).
    unsafe {
        let (w, xq) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = dst.ne[1] as usize;
        let per = mmq_preq::BLOCK as usize;
        let xb = xq.data as *const u8;
        let out = dst.data as *mut f32;
        for r in (ith..rows).step_by(nth) {
            let row = (w.data as *const u8).add(r * w.nb[1]);
            for j in 0..n {
                let mut sum = 0f32;
                for g in 0..k / GROUP {
                    let wb = row.add(g * Q8_0_BYTES);
                    let da = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*wb, *wb.add(1)]));
                    let blk = xb.add(((g * GROUP / per) * n + j) * mmq_preq::BLOCK_BYTES);
                    let gi = g % (per / GROUP);
                    let mut s4 = [0u8; 4];
                    std::ptr::copy_nonoverlapping(blk.add(gi * 4), s4.as_mut_ptr(), 4);
                    let db = f32::from_ne_bytes(s4);
                    let mut idot = 0i32;
                    for i in 0..GROUP {
                        let a = *wb.add(2 + i) as i8 as i32;
                        let b = *blk.add(16 + gi * GROUP + i) as i8 as i32;
                        idot += a * b;
                    }
                    sum += idot as f32 * da * db;
                }
                *out.add(j * rows + r) = sum;
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// `legion_q8_mv` of `src/cuda/q8_mv.cu`.
    #[repr(C)]
    struct Params {
        w: *const c_void,
        w_ld: i64,
        rows: i64,
        yq: *const c_void,
        dst: *mut f32,
        k: i64,
        n: i32,
    }

    unsafe extern "C" {
        fn legion_q8_mv_launch(p: *const Params, stream: *mut c_void) -> i32;
    }

    pub(super) unsafe fn launch(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`, device tensors of the stream's device.
        unsafe {
            let (w, xq) = (&*dst.src[0], &*dst.src[1]);
            let (k, n) = (w.ne[0], dst.ne[1]);
            if ffi::ggml_nbytes(xq) < mmq_preq::q8_1_bytes(k, n) {
                return Err("q8_mv: activations short".into());
            }
            let p = Params {
                w: w.data as *const c_void,
                w_ld: w.nb[1] as i64,
                rows: w.ne[1],
                yq: xq.data as *const c_void,
                dst: dst.data as *mut f32,
                k,
                n: n as i32,
            };
            match legion_q8_mv_launch(&p, stream) {
                0 => Ok(()),
                1 => Err("q8_mv: shape, alignment or device the kernel does not take".into()),
                e => Err(format!("q8_mv: CUDA error {e}")),
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q8_mv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::launch`.
    unsafe { cuda::launch(dst, stream) }
}

pub mod bench;

#[cfg(test)]
mod tests;
