//! A Q8_0 matmul through ggml-cuda's MMQ on activations a Legion kernel has already quantized
//! (#363): an external op ([`crate::ggml::ExternalOp`]) whose CUDA side calls ggml's
//! `mul_mat_q_case` through the glue in `src/cuda/mmq_preq.cu`.
//!
//! For a Q8_0 weight and more than 8 columns, ggml-cuda's `mul_mat` quantizes its F32 input to
//! Q8_1 in MMQ's layout (`quantize_mmq_q8_1`, a pass over the whole input) and then runs MMQ.
//! When the kernel that writes the input can quantize it on the way out, as qwen4exp's
//! hyper-connection norm does for its 10240 -> 320 down matmul (`hc_post_norm` in
//! `src/cuda/fused.cu`), the matmul here takes those blocks and skips ggml's pass. The kernel
//! writes ggml's bytes (its arithmetic, same compiler flags) and MMQ runs as ggml runs it (the
//! same tile width and arguments): ggml's output, bit for bit.
//!
//! The layout ([`q8_1_bytes`]): for `k` values per column (a multiple of 512, as ggml-cuda's
//! row padding needs) and `n` columns, block `b · n + j` (144 bytes: 4 F32 scales, then 128
//! int8) holds column `j`'s values `128 b .. 128 b + 127`, each group of 32 with its own scale
//! (`MMQ_Q8_1_DS_LAYOUT_D4`, the layout ggml uses for Q8_0 weights); then [`PAD_BYTES`] that
//! MMQ's last tile may read.
//!
//! Chosen on compute capability 12.x GPUs (GB10), where it was measured; `legion serve
//! --hc-quant-ggml` keeps ggml's quantization everywhere.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static MMQ_PREQ: ExternalOp = ExternalOp {
    name: "mmq_preq",
    cpu: cpu_mmq_preq,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmq_preq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Values per block (`QK8_1_MMQ`).
pub(crate) const BLOCK: i64 = 128;

/// Bytes per block (`sizeof(block_q8_1_mmq)`): 4 F32 scales, 128 int8.
pub(crate) const BLOCK_BYTES: usize = 144;

/// Values per scale.
const GROUP: usize = 32;

/// Bytes after the blocks MMQ may read (ggml-cuda pads its own buffer the same way): at most
/// 128 columns of a block each, rounded up to the tile's threads; the glue checks it.
pub(crate) const PAD_BYTES: usize = 20480;

/// Fewest columns [`mul_mat`] is chosen for. ggml-cuda takes MMQ from 9 (fewer: MMVQ,
/// `MMVQ_MAX_BATCH_SIZE`), but on a GB10 the GPU time saved (ggml's quantization less the norm
/// kernel's extra work, nsys, 10240 -> 320) is -3 us at 9 tokens, 1 at 48, 4 at 256, 43 at
/// 512 and 690 at 4096, so decode and MTP verify steps keep ggml's path.
const MIN_COLS: i64 = 256;

/// Whether graphs built from now on may use [`mul_mat`] (`legion serve --hc-quant-ggml` turns it
/// off for comparisons).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether the engine's GPU is one the op is chosen for (compute capability 12.x), set when the
/// engine loads ([`set_gpu`]).
static GPU: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on hand MMQ activations quantized by the kernel that
/// writes them where it fits (on by default); `false` keeps ggml's own quantization.
pub fn set_mmq_preq(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: [`mul_mat`] is used on compute capability 12.x (`cc_major`: GB10), where
/// it was measured faster than ggml's quantization pass; elsewhere ggml's `mul_mat` stays.
pub fn set_gpu(cc_major: Option<i32>) {
    GPU.store(cc_major == Some(12), Ordering::Relaxed);
}

/// Whether the engine's GPU is compute capability 12.x ([`set_gpu`]): where the #526 ops were
/// measured.
pub(crate) fn gb10() -> bool {
    GPU.load(Ordering::Relaxed)
}

/// Bytes of the quantized activations of `n` columns of `k` values, with MMQ's padding.
pub(crate) fn q8_1_bytes(k: i64, n: i64) -> usize {
    (k / BLOCK * n) as usize * BLOCK_BYTES + PAD_BYTES
}

/// Whether [`mul_mat`] takes `w` times `n` columns: switched on, a 12.x GPU ([`set_gpu`]),
/// a Q8_0 matrix in one device's memory with rows of a multiple of 512 values, and enough
/// columns for ggml-cuda to take MMQ.
pub(crate) fn fits(w: Tensor<'_>, n: i64) -> bool {
    let [k, _, n2, n3] = w.ne();
    ON.load(Ordering::Relaxed)
        && GPU.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && w.ty() == types::Q8_0
        && k % 512 == 0
        && n2 == 1
        && n3 == 1
        && n >= MIN_COLS
        && w.is_contiguous()
        && !w.is_host()
        && !w.is_meta()
}

/// `mul_mat(w, x)` (F32 `[rows, n]`) for `w` Q8_0 `[k, rows]` and `x` F32 `[k, n]`, from `xq`:
/// `x` quantized in MMQ's layout (see the module docs), [`q8_1_bytes`] at least, by the op that
/// wrote both. `x` itself is only read by the tests' check ([`set_check`]).
pub(crate) fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    x: Tensor<'a>,
) -> Tensor<'a> {
    let [k, rows, _, _] = w.ne();
    let n = x.ne()[1];
    assert_eq!(w.ty(), types::Q8_0, "mmq_preq: Q8_0 weight");
    assert_eq!(k % 512, 0, "mmq_preq: row size");
    assert_eq!(x.ne()[0], k, "mmq_preq: activations");
    assert!(
        xq.nbytes() >= q8_1_bytes(k, n),
        "mmq_preq: {} bytes of activations",
        xq.nbytes()
    );
    ctx.external(&MMQ_PREQ, types::F32, [rows, n, 1, 1], &[w, xq, x])
}

/// [`mul_mat`] of `n` columns where the op that wrote `xq` did not write `x` as F32 (the tests'
/// check then has nothing to compare).
pub(crate) fn mul_mat_cols<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    n: i64,
) -> Tensor<'a> {
    let [k, rows, _, _] = w.ne();
    assert_eq!(w.ty(), types::Q8_0, "mmq_preq: Q8_0 weight");
    assert_eq!(k % 512, 0, "mmq_preq: row size");
    assert!(
        xq.nbytes() >= q8_1_bytes(k, n),
        "mmq_preq: {} bytes of activations",
        xq.nbytes()
    );
    ctx.external(&MMQ_PREQ, types::F32, [rows, n, 1, 1], &[w, xq])
}

/// Quantizes `x` (`n` columns of `k` values) into `out` (at least [`q8_1_bytes`]) in MMQ's
/// layout with ggml's expressions (`quantize_mmq_q8_1`): the CPU reference. ggml-cuda's
/// fast-math division can round the scales differently.
pub(crate) fn quantize_ref(x: &[f32], k: usize, n: usize, out: &mut [u8]) {
    assert_eq!(x.len(), k * n);
    assert_eq!(k % BLOCK as usize, 0);
    let per = BLOCK as usize;
    for j in 0..n {
        for b in 0..k / per {
            let blk = &mut out[(b * n + j) * BLOCK_BYTES..][..BLOCK_BYTES];
            for g in 0..per / GROUP {
                let v = &x[j * k + b * per + g * GROUP..][..GROUP];
                let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
                let d_inv = 127.0 / amax;
                for (i, x) in v.iter().enumerate() {
                    // NaN (an all-zero group) converts to 0, as on the GPU.
                    blk[16 + g * GROUP + i] = ((x * d_inv).round() as i8) as u8;
                }
                blk[g * 4..g * 4 + 4].copy_from_slice(&(1.0 / d_inv).to_ne_bytes());
            }
        }
    }
}

unsafe fn cpu_mmq_preq(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors. The reference for tests: MMQ's sums per
    // group of 32 (integer products times both scales), in a plain order.
    unsafe {
        let (w, xq) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = dst.ne[1] as usize;
        let per = BLOCK as usize;
        let xb = xq.data as *const u8;
        let out = dst.data as *mut f32;
        // block_q8_0: an F16 scale, 32 int8.
        const Q8_0: usize = 34;
        for r in (ith..rows).step_by(nth) {
            let row = (w.data as *const u8).add(r * w.nb[1]);
            for j in 0..n {
                let mut sum = 0f32;
                for g in 0..k / GROUP {
                    let wb = row.add(g * Q8_0);
                    let dw = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*wb, *wb.add(1)]));
                    let blk = xb.add(((g * GROUP / per) * n + j) * BLOCK_BYTES);
                    let gi = g % (per / GROUP);
                    let mut s4 = [0u8; 4];
                    std::ptr::copy_nonoverlapping(blk.add(gi * 4), s4.as_mut_ptr(), 4);
                    let dx = f32::from_ne_bytes(s4);
                    let mut isum = 0i32;
                    for i in 0..GROUP {
                        let a = *wb.add(2 + i) as i8 as i32;
                        let b = *blk.add(16 + gi * GROUP + i) as i8 as i32;
                        isum += a * b;
                    }
                    sum += dw * dx * isum as f32;
                }
                *out.add(j * rows + r) = sum;
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// ggml's MMQ of a Q8_0 `src0` on blocks already in MMQ's layout (and
    /// [`super::shared_q8::small`]).
    pub(crate) fn legion_mmq_preq(
        backend: ffi::ggml_backend_t,
        src0: *const ffi::ggml_tensor,
        yq: *const c_void,
        yq_bytes: usize,
        ne11: i64,
        dst: *mut f32,
        nb1: i64,
    ) -> i32;
    /// ggml's own `quantize_mmq_q8_1` of a contiguous `x` for a Q8_0 weight (the tests' check,
    /// and [`super::shared_q8`]).
    pub(crate) fn legion_ggml_quantize_mmq_q8_0(
        x: *const f32,
        vy: *mut c_void,
        ne10: i64,
        ne11: i64,
        stream: *mut c_void,
    ) -> i32;
}

/// Tests: whether every [`mul_mat`] launched from now on also has ggml quantize its `x`
/// (`quantize_mmq_q8_1`) and compares that with `xq` byte for byte ([`checked`]).
#[cfg(all(test, feature = "cuda"))]
static CHECK: AtomicBool = AtomicBool::new(false);

/// Tests: the checks' counts since [`set_check`]: launches, bytes compared, bytes different.
#[cfg(all(test, feature = "cuda"))]
static CHECKED: std::sync::Mutex<(u64, u64, u64)> = std::sync::Mutex::new((0, 0, 0));

/// Tests: see [`CHECK`]; turning it on resets the counts.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn set_check(on: bool) {
    if on {
        *CHECKED.lock().unwrap() = (0, 0, 0);
    }
    CHECK.store(on, Ordering::Relaxed);
}

/// Tests: the checks' counts (launches, bytes compared, bytes different).
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn checked() -> (u64, u64, u64) {
    *CHECKED.lock().unwrap()
}

/// Tests: ggml's quantization of `x` against `xq` (see [`CHECK`]), synchronizing `stream`.
#[cfg(all(test, feature = "cuda"))]
unsafe fn check(
    backend: ffi::ggml_backend_t,
    x: &ffi::ggml_tensor,
    xq: &ffi::ggml_tensor,
    stream: *mut c_void,
) -> Result<(), String> {
    unsafe extern "C" {
        fn cudaStreamSynchronize(s: *mut c_void) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: i32) -> i32;
    }
    const DEFAULT: i32 = 4;
    // SAFETY: device tensors of the backend's device; the scratch buffer outlives its uses.
    unsafe {
        let (k, n) = (x.ne[0], x.ne[1]);
        if x.nb[1] != k as usize * 4 {
            return Err("check: x not contiguous".into());
        }
        let data = (k / BLOCK * n) as usize * BLOCK_BYTES;
        let buf = ffi::ggml_backend_alloc_buffer(backend, data + PAD_BYTES);
        let base = ffi::ggml_backend_buffer_get_base(buf);
        let e = legion_ggml_quantize_mmq_q8_0(x.data as *const f32, base, k, n, stream);
        let mut ours = vec![0u8; data];
        let mut ggml = vec![0u8; data];
        let s = cudaStreamSynchronize(stream);
        let c1 = cudaMemcpy(ours.as_mut_ptr().cast(), xq.data, data, DEFAULT);
        let c2 = cudaMemcpy(ggml.as_mut_ptr().cast(), base, data, DEFAULT);
        ffi::ggml_backend_buffer_free(buf);
        if e != 0 || s != 0 || c1 != 0 || c2 != 0 {
            return Err(format!("check: CUDA errors {e} {s} {c1} {c2}"));
        }
        let diff = ours.iter().zip(&ggml).filter(|(a, b)| a != b).count();
        let mut c = CHECKED.lock().unwrap();
        c.0 += 1;
        c.1 += data as u64;
        c.2 += diff as u64;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
unsafe fn cuda_mmq_preq(dst: &ffi::ggml_tensor, _stream: *mut c_void) -> Result<(), String> {
    let backend = crate::ggml::current_cuda_backend();
    if backend.is_null() {
        return Err("no CUDA backend computing".into());
    }
    // SAFETY: a node made by `mul_mat`, device tensors of the backend's device; the glue
    // enqueues on the backend's stream (the one the graph's segments and `_stream` run on).
    unsafe {
        let (w, xq) = (&*dst.src[0], &*dst.src[1]);
        #[cfg(test)]
        if CHECK.load(Ordering::Relaxed) && !dst.src[2].is_null() {
            check(backend, &*dst.src[2], xq, _stream)?;
        }
        let bytes = ffi::ggml_nbytes(xq);
        let n = dst.ne[1];
        match legion_mmq_preq(
            backend,
            w,
            xq.data,
            bytes,
            n,
            dst.data as *mut f32,
            dst.nb[1] as i64,
        ) {
            0 => Ok(()),
            -1 => Err("not a matmul ggml-cuda runs with MMQ".into()),
            -2 => Err(format!("{bytes} bytes of activations: MMQ's padding short")),
            e => Err(format!("CUDA error {e}")),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
