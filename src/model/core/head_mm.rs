//! The target's BF16 LM head in verify steps of many sequences (#1181): `w @ x`, the final logit
//! soft-cap and the argmax of every column in one pass over the weights, with ggml-cuda's bits.
//!
//! Gemma 4 31B's head is BF16 `[5376, 262144]` (2.8 GB). A DFlash 2 verify step of 16 sequences
//! reads it for 128 rows through ggml-cuda: the BF16 conversion of `x`, cuBLAS's BF16 GEMM, the
//! fused soft-cap (`softcap_f32`, another read and write of the 134 MB of logits) and
//! `argmax_f32` (another read). On a DGX Spark (GB10) cuBLAS picks a kernel of 128 x 208 tiles
//! there (80 of its 208 columns padding) that streams the head at ~206 GB/s: ~13.7 ms of a
//! ~215 ms step, the soft-cap and argmax another ~1.7 ms.
//!
//! Here two external ops ([`crate::ggml::ExternalOp`], kernels in `src/cuda/head_mm.cu`)
//! replace those nodes ([`head`]):
//! - `head_mm` (the logits, F32 `[m, n]`): `x` rounded to BF16 as ggml rounds it, then one CTA
//!   per 128 rows of the weights (staged 64 `k` values, 128 bytes a row, at a time) computing
//!   every output as one chain of m16n8k16 BF16 products with F32 sums over `k` in order from 0
//!   (the order of cuBLAS's kernel there, so its bits), the soft-cap as `softcap_f32` computes
//!   it, the logits written once, and the CTA's best (value, tie rank) of each column into a
//!   per-stream buffer. On a GB10 the three nodes take ~15.5 ms at 128 rows, the op ~13.8 ms
//!   (~205 GB/s);
//! - `head_pick` (the argmax, I32 `[n]`): one CTA per column joins those partials with
//!   `argmax_f32`'s pick, ties included (its tie order: `src/cuda/head_pick.cuh`,
//!   [`crate::model::cluster_select::argmax_row`]).
//!
//! The partials buffer and `x`'s BF16 copy are not graph tensors (one buffer per device and
//! stream, made at the first launch, as [`crate::model::cluster_select`]'s split argmax keeps
//! its partials): tensors would move the graph's memory layout and with it which ggml-cuda
//! fusions fire (#472). `head_pick` reads what the `head_mm` before it on its stream wrote; the
//! graph runs them back to back.
//!
//! Taken only where that gives cuBLAS's bits ([`takes`]): by default the head shapes and column
//! counts checked bit for bit on a GB10 ([`BITWISE`], `legion head-mm-bench --check`); `legion
//! serve --no-head-mm` keeps ggml's nodes ([`set_head_mm`]). The CPU implementation is a
//! reference (BF16 products summed in F32 in order, the soft-cap in F32).

use std::sync::atomic::{AtomicU8, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static HEAD_MM: ExternalOp = ExternalOp {
    name: "head_mm",
    cpu: cpu_head_mm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::head_mm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static HEAD_PICK: ExternalOp = ExternalOp {
    name: "head_pick",
    cpu: cpu_head_pick,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::head_pick),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the op takes: below, BF16 heads run on the dense ops
/// ([`crate::model::dense_mv`], up to 48 tokens).
pub const MIN_COLS: i64 = 49;

/// Most columns the op takes (16 sequences of 8 verified tokens).
pub const MAX_COLS: i64 = 128;

/// Weight rows per CTA (the head's rows a multiple of it).
pub const ROWS: i64 = 128;

/// `k` values per pipeline stage (the row length a multiple of it).
pub const BK: i64 = 64;

/// Longest rows the op takes (the size of `x`'s BF16 copy).
pub const MAX_K: i64 = 16384;

/// Most CTAs (row blocks of [`ROWS`]) the op takes: the partials buffer's size.
pub const MAX_PARTS: i64 = 4096;

/// Fewest rows: `argmax_f32` runs 1,024 threads only from about this many (its tie order is
/// that of 1,024 threads).
pub const MIN_ROWS: i64 = crate::model::cluster_select::MIN_ARGMAX_COLS;

/// The compute capability (10 * major + minor) [`BITWISE`] was measured on: a GB10.
pub const BITWISE_CC: i32 = 121;

/// The BF16 head shapes `(k, m)` (`w` is `[k, m]`) whose head the op computes with the bits of
/// ggml-cuda's cuBLAS GEMM, soft-cap and argmax on a GB10 ([`BITWISE_CC`]), and the column
/// counts (bit `n - MIN_COLS` for `n`). Measured with `legion head-mm-bench --check` (random
/// weights and inputs, ties among repeated rows; every logit's and pick's bits compared).
/// cuBLAS picks its kernel by shape, so a cuBLAS that picks others needs that check again.
pub const BITWISE: &[(i64, i64, u128)] = &[
    // Gemma 4 31B (tied embeddings, BF16).
    (5376, 262144, (1u128 << (MAX_COLS - MIN_COLS + 1)) - 1),
];

/// [`MODE`]: the shapes of cuBLAS's bits ([`BITWISE`]).
const MODE_BITWISE: u8 = 0;
/// [`MODE`]: every head that fits (tools and tests).
const MODE_ALL: u8 = 1;
/// [`MODE`]: none.
const MODE_OFF: u8 = 2;

static MODE: AtomicU8 = AtomicU8::new(MODE_BITWISE);

/// Whether graphs built from now on run the BF16 LM head of verify steps of [`MIN_COLS`] to
/// [`MAX_COLS`] rows on the op where it gives cuBLAS's bits ([`BITWISE`]; on by default);
/// `legion serve --no-head-mm` keeps ggml's matmul, soft-cap and argmax (for comparisons).
pub fn set_head_mm(on: bool) {
    MODE.store(if on { MODE_BITWISE } else { MODE_OFF }, Ordering::Relaxed);
}

/// Takes every head that [`fits`] from now on (`true`), or only [`BITWISE`]'s again: for
/// `legion head-mm-bench` and tests, which compare the bits themselves.
#[doc(hidden)]
pub fn force(all: bool) {
    MODE.store(if all { MODE_ALL } else { MODE_BITWISE }, Ordering::Relaxed);
}

/// Whether `(k, m)` at `n` columns is one of [`BITWISE`]'s.
pub fn bitwise(k: i64, m: i64, n: i64) -> bool {
    (MIN_COLS..=MAX_COLS).contains(&n)
        && BITWISE
            .iter()
            .any(|&(bk, bm, mask)| bk == k && bm == m && (mask >> (n - MIN_COLS)) & 1 != 0)
}

/// Whether the head `w @ x` (then the soft-cap and the argmax) takes the op: [`fits`], a
/// weight in one GPU's memory on a CUDA engine, that GPU running the kernels ([`supported_on`]),
/// and by default a shape and column count of cuBLAS's bits on a GB10 ([`bitwise`]) as that GPU.
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let mode = MODE.load(Ordering::Relaxed);
    let [k, m, ..] = w.ne();
    mode != MODE_OFF
        && cfg!(feature = "cuda")
        && fits(w, x)
        && !w.is_host()
        && !w.is_meta()
        && crate::model::cluster_select::on_cuda()
        && w.cuda_device().is_some_and(|dev| {
            supported_on(dev)
                && (mode == MODE_ALL
                    || (bitwise(k, m, x.ne()[1])
                        && crate::hw::per_device::compute_capability(dev) == Some(BITWISE_CC)))
        })
}

/// Whether the kernels compute `w @ x`: a BF16 matrix `w` (`[k, m]`, rows 16-byte aligned,
/// `k` a multiple of [`BK`] up to [`MAX_K`], `m` a multiple of [`ROWS`] from [`MIN_ROWS`] up
/// to [`MAX_PARTS`] CTAs) and a contiguous F32 `x` of [`MIN_COLS`] to [`MAX_COLS`] columns.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    shape_fits(w.ty(), w.ne(), w.nb(), x.ty(), x.ne(), x.is_contiguous())
}

/// The shape test of [`fits`].
pub(crate) fn shape_fits(
    wty: ffi::ggml_type,
    wne: [i64; 4],
    wnb: [usize; 4],
    xty: ffi::ggml_type,
    xne: [i64; 4],
    x_contiguous: bool,
) -> bool {
    let [k, m, w2, w3] = wne;
    let [xk, n, x2, x3] = xne;
    wty == types::BF16
        && xty == types::F32
        && x_contiguous
        && [w2, w3, x2, x3] == [1, 1, 1, 1]
        && xk == k
        && k > 0
        && k % BK == 0
        && k <= MAX_K
        && m >= MIN_ROWS
        && m % ROWS == 0
        && m / ROWS <= MAX_PARTS
        && (MIN_COLS..=MAX_COLS).contains(&n)
        && wnb[0] == 2
        && wnb[1].is_multiple_of(16)
        && wnb[1] >= 2 * k as usize
}

/// Whether the current GPU runs the kernels: compute capability 8.0 or newer (BF16 tensor
/// cores, `cp.async`), code for it in this build, and the GEMM's shared memory granted.
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::supported()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether CUDA device `dev` runs the kernels ([`device_supported`] asked of that device): the
/// head weight's GPU, which need not be the current one when graphs are built.
pub fn supported_on(dev: usize) -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::supported_on(dev)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = dev;
        false
    }
}

/// The logits of the head `w` (BF16 `[k, m]`) for `x` (F32 `[k, n]`) soft-capped by `cap`
/// (`cap * tanh(v / cap)` as ggml-cuda's fused soft-cap computes it; none for `cap <= 0`), and
/// their argmax: F32 `[m, n]` and I32 `[n]`, ggml's `mul_mat`, soft-cap and `argmax` bit for bit
/// where [`takes`] holds. [`fits`] must hold.
pub fn head<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    cap: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    assert!(fits(w, x), "head_mm: w {:?}, x {:?}", w.ne(), x.ne());
    let (scale, cap) = if cap > 0.0 {
        (1.0 / cap, cap)
    } else {
        (1.0, 0.0)
    };
    let n = x.ne()[1];
    let logits = ctx.external_with(
        &HEAD_MM,
        types::F32,
        [w.ne()[1], n, 1, 1],
        &[w, x],
        &[scale.to_bits() as i32, cap.to_bits() as i32],
    );
    let argmax = ctx.external(&HEAD_PICK, types::I32, [n, 1, 1, 1], &[logits]);
    (logits, argmax)
}

/// The op's soft-cap parameters of a `head_mm` node: `(scale, cap)`.
fn params(dst: &ffi::ggml_tensor) -> (f32, f32) {
    let p = crate::ggml::external_params(dst);
    (f32::from_bits(p[0] as u32), f32::from_bits(p[1] as u32))
}

/// `v` rounded to BF16 (to nearest, ties to even) and back.
pub(crate) fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    if v.is_nan() {
        return v;
    }
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// One logit as the CPU reference computes it: the BF16 row `w` times `x` rounded to BF16,
/// summed in F32 in order, then the soft-cap.
pub(crate) fn cpu_logit(w: &[u16], x: &[f32], scale: f32, cap: f32) -> f32 {
    let s = w.iter().zip(x).fold(0.0f32, |s, (&wv, &xv)| {
        f32::from_bits(u32::from(wv) << 16).mul_add(bf16_round(xv), s)
    });
    if cap > 0.0 {
        (scale * s).tanh() * cap
    } else {
        s
    }
}

unsafe fn cpu_head_mm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `head`, host tensors.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, m, n) = (w.ne[0] as usize, w.ne[1] as usize, x.ne[1] as usize);
        let (scale, cap) = params(dst);
        for c in (ith..n).step_by(nth) {
            let xs =
                std::slice::from_raw_parts((x.data as *const u8).add(c * x.nb[1]) as *const f32, k);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(c * dst.nb[1]) as *mut f32,
                m,
            );
            for (r, o) in out.iter_mut().enumerate() {
                let ws = std::slice::from_raw_parts(
                    (w.data as *const u8).add(r * w.nb[1]) as *const u16,
                    k,
                );
                *o = cpu_logit(ws, xs, scale, cap);
            }
        }
    }
}

unsafe fn cpu_head_pick(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `head`, host tensors.
    unsafe {
        let l = &*dst.src[0];
        let (m, n) = (l.ne[0] as usize, l.ne[1] as usize);
        for c in (ith..n).step_by(nth) {
            let row =
                std::slice::from_raw_parts((l.data as *const u8).add(c * l.nb[1]) as *const f32, m);
            *(dst.data as *mut i32).add(c) = crate::model::cluster_select::argmax_row(row);
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The kernels' launches: `cudaLaunchKernelExC` from Rust (no host code in the `.cu`).

    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicU8, Ordering};

    use super::{BK, MAX_COLS, MAX_K, MAX_PARTS, MIN_COLS, ROWS, ffi};

    /// `legion_head_mm` (`src/cuda/head_mm.cu`).
    #[repr(C)]
    struct Params {
        w: *const c_void,
        x: *const f32,
        dst: *mut f32,
        xb: *mut c_void,
        part: *mut c_void,
        argmax: *mut i32,
        w_ld: i64,
        x_ld: i64,
        dst_ld: i64,
        m: i64,
        k: i64,
        n: i32,
        ncols: i32,
        parts: i32,
        scale: f32,
        cap: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `cudaLaunchAttribute`: the id, padding to 8 bytes, a 64-byte value union (none set).
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

    /// `cudaFuncAttributes`, its fields up to `binaryVersion`, padded past the struct's size.
    #[repr(C)]
    struct FuncAttributes {
        shared: usize,
        constant: usize,
        local: usize,
        max_threads: c_int,
        regs: c_int,
        ptx_version: c_int,
        binary_version: c_int,
        rest: [u8; 256],
    }

    const DEV_ATTR_CC_MAJOR: c_int = 75;
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize`.
    const FUNC_ATTR_MAX_DYNAMIC_SMEM: c_int = 8;

    unsafe extern "C" {
        /// The conversion, the GEMM for 64, 96 and 128 columns, the pick.
        static legion_head_mm_kernels: [*const c_void; 5];
        /// The GEMM kernels' dynamic shared memory.
        static legion_head_mm_smem: [c_int; 3];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
    }

    /// Devices checked by [`supported`]: 0 unknown, 1 yes, 2 no.
    static CHECKED: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];

    pub(super) fn supported() -> bool {
        let mut dev = 0;
        // SAFETY: plain queries of the current device; the attribute set is the kernels' own.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..64).contains(&dev) {
                cudaGetLastError();
                return false;
            }
            let slot = &CHECKED[dev as usize];
            match slot.load(Ordering::Relaxed) {
                1 => return true,
                2 => return false,
                _ => {}
            }
            let mut major = 0;
            let mut ok =
                cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0 && major >= 8;
            for (i, &f) in legion_head_mm_kernels.iter().enumerate() {
                if !ok {
                    break;
                }
                let mut a: FuncAttributes = std::mem::zeroed();
                ok = cudaFuncGetAttributes(&mut a, f) == 0 && a.binary_version >= 80;
                // The GEMM kernels (1-3) need more than the default 48 KiB.
                if ok && (1..=3).contains(&i) {
                    ok = cudaFuncSetAttribute(
                        f,
                        FUNC_ATTR_MAX_DYNAMIC_SMEM,
                        legion_head_mm_smem[i - 1],
                    ) == 0;
                }
            }
            if !ok {
                cudaGetLastError();
            }
            slot.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }

    /// [`supported`] asked of device `dev`, made current for the queries and switched back.
    pub(super) fn supported_on(dev: usize) -> bool {
        let Ok(dev) = c_int::try_from(dev) else {
            return false;
        };
        let mut was = 0;
        // SAFETY: device queries; the current device is put back before returning.
        unsafe {
            if cudaGetDevice(&mut was) != 0 {
                cudaGetLastError();
                return false;
            }
            if was == dev {
                return supported();
            }
            if cudaSetDevice(dev) != 0 {
                cudaGetLastError();
                return false;
            }
            let ok = supported();
            cudaSetDevice(was);
            ok
        }
    }

    /// Bytes of `x`'s BF16 copy: [`MAX_COLS`] columns of [`MAX_K`].
    const XB_BYTES: usize = (MAX_COLS * MAX_K * 2) as usize;

    /// Bytes of the partials: (value bits, tie rank) per column and CTA.
    const PART_BYTES: usize = (MAX_COLS * MAX_PARTS * 8) as usize;

    /// The scratch buffers: (device, stream, buffer of [`XB_BYTES`] + [`PART_BYTES`]).
    static SCRATCH: std::sync::Mutex<Vec<(c_int, usize, usize)>> =
        std::sync::Mutex::new(Vec::new());

    /// The scratch buffer for `stream` on the current device, made at its first use (a stage
    /// graph's capture runs in relaxed mode, where `cudaMalloc` is allowed). Launches on one
    /// stream run in order, so they share it; a captured graph replays on its stream.
    fn scratch(stream: *mut c_void) -> Result<*mut c_void, String> {
        let mut dev = 0;
        // SAFETY: a plain query of the current device.
        if unsafe { cudaGetDevice(&mut dev) } != 0 {
            return Err("head_mm: cudaGetDevice failed".into());
        }
        let mut bufs = SCRATCH.lock().unwrap();
        if let Some(&(_, _, p)) = bufs
            .iter()
            .find(|&&(d, s, _)| d == dev && s == stream as usize)
        {
            return Ok(p as *mut c_void);
        }
        let mut p = std::ptr::null_mut();
        // SAFETY: a plain allocation on the current device, kept for the process.
        match unsafe { cudaMalloc(&mut p, XB_BYTES + PART_BYTES) } {
            0 => {
                bufs.push((dev, stream as usize, p as usize));
                Ok(p)
            }
            e => Err(format!("head_mm: cudaMalloc: CUDA error {e}")),
        }
    }

    /// Launches `func` on `grid` CTAs of `threads` threads with `smem` bytes of dynamic shared
    /// memory and `p`.
    unsafe fn launch(
        func: *const c_void,
        grid: u32,
        threads: u32,
        smem: usize,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let config = LaunchConfig {
            grid: Dim3 {
                x: grid,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: threads,
                y: 1,
                z: 1,
            },
            dynamic_smem: smem,
            stream,
            attrs: std::ptr::null_mut(),
            num_attrs: 0,
        };
        let mut args = [(p as *mut Params).cast::<c_void>()];
        // SAFETY: `func` takes one argument of type `Params`; the launch copies it.
        match unsafe { cudaLaunchKernelExC(&config, func, args.as_mut_ptr()) } {
            0 => Ok(()),
            e => Err(format!("head_mm: CUDA error {e}")),
        }
    }

    /// The launch parameters of the `head_mm` node `dst` (or of the `head_pick` node's source
    /// `dst`) on `stream`, and the GEMM kernel's index in the tables (NT 4, 6, 8: 0-2).
    unsafe fn params(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(Params, usize), String> {
        // SAFETY: a node made by `head`, device tensors of the stream's device.
        unsafe {
            let (w, x) = (&*dst.src[0], &*dst.src[1]);
            let (k, m, n) = (w.ne[0], w.ne[1], x.ne[1]);
            let ok = k > 0
                && k % BK == 0
                && k <= MAX_K
                && m % ROWS == 0
                && m / ROWS <= MAX_PARTS
                && (MIN_COLS..=MAX_COLS).contains(&n)
                && w.nb[0] == 2
                && w.nb[1] % 16 == 0
                && x.nb[0] == 4
                && x.nb[1] % 8 == 0
                && dst.nb[1] % 4 == 0
                && (w.data as usize) % 16 == 0
                && (x.data as usize) % 8 == 0;
            if !ok {
                return Err("head_mm: shape or alignment the kernels do not take".into());
            }
            // Also grants the GEMM's shared memory on this device.
            if !supported() {
                return Err("head_mm: this GPU does not run the kernels".into());
            }
            let nt = match n {
                ..=64 => 4,
                65..=96 => 6,
                _ => 8,
            };
            let buf = scratch(stream)?;
            let (scale, cap) = super::params(dst);
            Ok((
                Params {
                    w: w.data as *const c_void,
                    x: x.data as *const f32,
                    dst: dst.data as *mut f32,
                    xb: buf,
                    part: buf.cast::<u8>().add(XB_BYTES).cast(),
                    argmax: std::ptr::null_mut(),
                    w_ld: (w.nb[1] / 2) as i64,
                    x_ld: (x.nb[1] / 4) as i64,
                    dst_ld: (dst.nb[1] / 4) as i64,
                    m,
                    k,
                    n: n as i32,
                    ncols: 16 * nt,
                    parts: (m / ROWS) as i32,
                    scale,
                    cap,
                },
                (nt as usize - 4) / 2,
            ))
        }
    }

    pub(super) unsafe fn head_mm(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `head`; the kernels read and write within the shapes checked.
        unsafe {
            let (mut p, i) = params(dst, stream)?;
            let pairs = i64::from(p.ncols) * p.k / 2;
            let grid = ((pairs + 255) / 256) as u32;
            launch(legion_head_mm_kernels[0], grid, 256, 0, &mut p, stream)?;
            // The GEMM kernels follow the conversion in the table.
            let smem = legion_head_mm_smem[i] as usize;
            launch(
                legion_head_mm_kernels[1 + i],
                p.parts as u32,
                256,
                smem,
                &mut p,
                stream,
            )
        }
    }

    pub(super) unsafe fn head_pick(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `head`: its source is the `head_mm` node that ran before it on
        // this stream and left its partials in the stream's buffer.
        unsafe {
            let (mut p, _) = params(&*dst.src[0], stream)?;
            p.argmax = dst.data as *mut i32;
            launch(
                legion_head_mm_kernels[4],
                p.n as u32,
                256,
                0,
                &mut p,
                stream,
            )
        }
    }
}

pub mod bench;

#[cfg(test)]
mod tests;
