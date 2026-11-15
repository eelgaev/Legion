//! Q4_K matvecs of 1-8 columns with ggml-cuda's bits on a GB10 (#689): the DFlash drafter's
//! Q4_K matmuls of a draft pass (`--spec-draft-weights q4`, `--spec-draft-lm-head q4`).
//!
//! A DFlash 2 draft pass of one sequence runs its 5 layers and its 131,072-row head on 8 columns
//! (the draft block). ggml-cuda runs every one of those Q4_K matmuls as `quantize_q8_1` plus
//! `mul_mat_vec_q<Q4_K, 8>`: CTAs of 2 warps and 2 rows, each lane reading 4-byte pieces of the
//! weights and decoding its blocks again for every column. On Qwen3.8-27B's drafter that reads
//! the weights at ~200 GB/s (the Q/K/V matvec at ~120; nsys on a DGX Spark, #689 r6), the
//! largest part of the drafter's time a cycle above its weight floor. Here one external op
//! ([`crate::ggml::ExternalOp`], `src/cuda/q4k_mv.cu`) replaces the `mul_mat` node (the same
//! inputs and output): resident CTAs whose warps walk the rows, one warp a row, 16-byte weight
//! loads a row ahead, each block decoded once for all columns, the first rows' weights read
//! before the kernel waits for the one ahead (a programmatic dependent launch), and the input
//! quantized by every CTA as ggml's `quantize_q8_1` does. Every sum is ggml's expression over the
//! same values in the same order (the order of [`super::q4k_head`], #836), so every output has
//! ggml's bits.
//!
//! Taken where ggml would run that kernel the same way: a GB10 ([`super::q4k_head::device_supported`]),
//! a Q4_K weight of a row length the kernel is built for ([`K`]) and 1 to [`MAX_COLS`] F32 columns
//! ([`fits`]). The DFlash drafter takes it for its Q4_K matmuls of up to 8 columns
//! (`LEGION_DEBUG_DRAFT_Q4K_MV=0` keeps ggml's `mul_mat`, for A/B runs). The CPU implementation is
//! [`super::q4k_head`]'s reference.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static Q4K_MV: ExternalOp = ExternalOp {
    name: "q4k_mv",
    cpu: cpu_q4k_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q4k_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Row lengths the op takes: the drafter's hidden size (5,120: Q/K/V, gate/up, the head) and its
/// attention width (4,096: the attention output), 20 and 16 Q4_K blocks.
pub const K: [i64; 2] = [4096, 5120];

/// Most columns the op takes: ggml-cuda runs MMVQ up to 8 (`MMVQ_MAX_BATCH_SIZE`).
pub const MAX_COLS: i64 = 8;

/// Bytes of a Q4_K block.
const Q4_K_BYTES: usize = 144;

/// Values per Q4_K block.
const QK_K: usize = 256;

/// Whether the op runs the drafter's Q4_K matmuls: on unless `LEGION_DEBUG_DRAFT_Q4K_MV=0` (A/B
/// runs), `head` for the head only, `layers` for the layers only. Read once.
pub fn draft_mode() -> DraftMode {
    static V: std::sync::OnceLock<DraftMode> = std::sync::OnceLock::new();
    *V.get_or_init(
        || match std::env::var("LEGION_DEBUG_DRAFT_Q4K_MV").as_deref() {
            Ok("0") => DraftMode::Off,
            Ok("head") => DraftMode::Head,
            Ok("layers") => DraftMode::Layers,
            _ => DraftMode::All,
        },
    )
}

/// Which of the drafter's Q4_K matmuls take the op ([`draft_mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftMode {
    Off,
    Head,
    Layers,
    All,
}

impl DraftMode {
    /// Whether the head (`head` true) or a layer's matmul takes the op.
    pub fn on(self, head: bool) -> bool {
        match self {
            DraftMode::Off => false,
            DraftMode::Head => head,
            DraftMode::Layers => !head,
            DraftMode::All => true,
        }
    }
}

/// Whether `w @ x` takes the op: a GPU that runs the kernel ([`super::q4k_head::device_supported`]),
/// [`fits`], a weight in one GPU's memory.
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    cfg!(feature = "cuda")
        && fits(w, x)
        && !w.is_host()
        && !w.is_meta()
        && super::q4k_head::device_supported()
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
        && rows <= i32::MAX as i64
        && (1..=MAX_COLS).contains(&n)
        && wnb[0] == Q4_K_BYTES
        && wnb[1].is_multiple_of(16)
        && wnb[1] >= k as usize / QK_K * Q4_K_BYTES
}

/// `w @ x` (F32 `[rows, n]`) for `w` and `x` that [`fits`] takes, as one node with ggml's
/// `mul_mat`'s inputs and output.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q4k_mv: w {:?}, x {:?}", w.ne(), x.ne());
    ctx.external(&Q4K_MV, types::F32, [w.ne()[1], x.ne()[1], 1, 1], &[w, x])
}

/// A launch shape: warps per CTA (8 or 16) and rows a warp step (1 or 2). Every shape gives the
/// same bits (only which warp computes a row, and when its weights are read, change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub warps: usize,
    pub rows: usize,
}

/// The default shape for `n` columns (GB10, the drafter's shapes; `legion q4k-mv-bench` times
/// the others).
pub fn default_shape(n: usize) -> Shape {
    if n <= 4 {
        Shape { warps: 8, rows: 2 }
    } else {
        Shape { warps: 16, rows: 1 }
    }
}

/// The forced shape: warps | rows << 8, 0 for the default.
static FORCED: AtomicU32 = AtomicU32::new(0);

/// Launches from now on take `shape` (tuning; `None` for the default). False for a shape the
/// kernel lacks.
#[doc(hidden)]
pub fn force_shape(shape: Option<Shape>) -> bool {
    match shape {
        None => {
            FORCED.store(0, Ordering::Relaxed);
            true
        }
        Some(s) if [8, 16].contains(&s.warps) && [1, 2].contains(&s.rows) => {
            FORCED.store((s.warps | s.rows << 8) as u32, Ordering::Relaxed);
            true
        }
        Some(_) => false,
    }
}

/// The shape launches take for `n` columns ([`force_shape`], else [`default_shape`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn shape(n: usize) -> Shape {
    match FORCED.load(Ordering::Relaxed) as usize {
        0 => default_shape(n),
        f => Shape {
            warps: f & 0xff,
            rows: f >> 8,
        },
    }
}

/// The kernel of `legion_q4k_mv_kernels` for rows of `k` values, `n` columns and `shape`.
pub fn kernel_index(k: usize, n: usize, shape: Shape) -> Option<usize> {
    let nb = match k {
        4096 => 0,
        5120 => 1,
        _ => return None,
    };
    if !(1..=MAX_COLS as usize).contains(&n) {
        return None;
    }
    let nw = match shape.warps {
        8 => 0,
        16 => 1,
        _ => return None,
    };
    let r = match shape.rows {
        1 => 0,
        2 => 1,
        _ => return None,
    };
    Some(((nb * 8 + n - 1) * 2 + nw) * 2 + r)
}

/// Dynamic shared memory of a launch: the input (16 bytes of Q8_1 ints and 4 of scale per 32
/// values a column) and each warp's rows of weights.
pub fn smem_bytes(k: usize, n: usize, shape: Shape) -> usize {
    let nb = k / QK_K;
    n * nb * (16 * 16 + 8 * 4) + shape.warps * shape.rows * nb * Q4_K_BYTES
}

/// # Safety
/// `dst` is a node made by [`mul_mat`], all tensors in host memory.
unsafe fn cpu_q4k_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    use super::q4k_head::{ggml_warps, quantize_block, row_dot};
    // SAFETY: a node made by `mul_mat`, host tensors: the weight and the F32 columns.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows, n) = (w.ne[0] as usize, w.ne[1] as usize, x.ne[1] as usize);
        let g = ggml_warps(n, k / QK_K);
        let xq: Vec<Vec<_>> = (0..n)
            .map(|j| {
                let col = std::slice::from_raw_parts((x.data as *const f32).add(j * k), k);
                col.chunks(32).map(quantize_block).collect()
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

/// # Safety
/// `dst` is a node made by [`mul_mat`] (its shapes passed [`fits`]), its tensors in device
/// memory of the stream's device; `stream` is the compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_q4k_mv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat`, device tensors of the stream's device.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        if !(w.data as usize).is_multiple_of(16) {
            return Err("q4k_mv: weight not 16-byte aligned".into());
        }
        let (k, n) = (w.ne[0] as usize, x.ne[1] as usize);
        let shape = shape(n);
        let kernel = kernel_index(k, n, shape)
            .ok_or_else(|| format!("q4k_mv: no kernel for k {k}, {n} columns, {shape:?}"))?;
        let mut p = launch::Params {
            w: w.data as *const c_void,
            w_ld: w.nb[1] as i64,
            rows: w.ne[1],
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            dst: dst.data as *mut f32,
            dst_ld: (dst.nb[1] / 4) as i64,
            k: k as i32,
            n: n as i32,
        };
        launch::launch(
            kernel,
            shape.warps,
            smem_bytes(k, n, shape),
            (w.ne[1] as usize).div_ceil(shape.warps * shape.rows),
            &mut p,
            stream,
        )
    }
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI32, Ordering};

    /// `legion_q4k_mv` of `src/cuda/q4k_mv.cu`.
    #[repr(C)]
    pub(super) struct Params {
        pub w: *const c_void,
        pub w_ld: i64,
        pub rows: i64,
        pub x: *const f32,
        pub x_ld: i64,
        pub dst: *mut f32,
        pub dst_ld: i64,
        pub k: i32,
        pub n: i32,
    }

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

    /// Entries of the kernel table.
    const KERNELS: usize = 64;

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs), by
        /// [`super::kernel_index`].
        static legion_q4k_mv_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            blocks: *mut c_int,
            func: *const c_void,
            block_size: c_int,
            dynamic_smem: usize,
        ) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const DEV_ATTR_SM_COUNT: c_int = 16;
    const FUNC_ATTR_MAX_DYNAMIC_SMEM: c_int = 8;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// Each device's SM count; 0 not yet read, -1 unknown.
    static SMS: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];
    /// CTAs an SM holds per (device, kernel) with its shared memory opted in; 0 not yet known.
    static FIT: [[AtomicI32; KERNELS]; DEVICES] =
        [const { [const { AtomicI32::new(0) }; KERNELS] }; DEVICES];

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream`: `warps` warps a CTA,
    /// `smem` bytes of dynamic shared memory, as many CTAs as fit on the GPU at once (at most
    /// `need`); as a programmatic dependent launch, as ggml-cuda launches its own kernels on a
    /// GB10 (the kernel waits for the one ahead before it reads the input).
    pub(super) fn launch(
        kernel: usize,
        warps: usize,
        smem: usize,
        need: usize,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return Err("q4k_mv: no current device".into());
            }
        }
        let d = dev as usize;
        let func = unsafe { legion_q4k_mv_kernels[kernel] };
        let threads = (warps * 32) as c_int;
        let mut sms = SMS[d].load(Ordering::Relaxed);
        if sms == 0 {
            // SAFETY: a plain attribute query.
            sms = if unsafe { cudaDeviceGetAttribute(&mut sms, DEV_ATTR_SM_COUNT, dev) } == 0
                && sms > 0
            {
                sms
            } else {
                unsafe { cudaGetLastError() };
                -1
            };
            SMS[d].store(sms, Ordering::Relaxed);
        }
        if sms <= 0 {
            return Err("q4k_mv: SM count unknown".into());
        }
        let mut fit = FIT[d][kernel].load(Ordering::Relaxed);
        if fit == 0 {
            // SAFETY: an attribute of a kernel of the table, then an occupancy query.
            unsafe {
                if cudaFuncSetAttribute(func, FUNC_ATTR_MAX_DYNAMIC_SMEM, smem as c_int) != 0
                    || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mut fit, func, threads, smem)
                        != 0
                    || fit <= 0
                {
                    cudaGetLastError();
                    return Err(format!("q4k_mv: kernel {kernel} does not fit {smem} B"));
                }
            }
            FIT[d][kernel].store(fit, Ordering::Relaxed);
        }
        let grid = need.min(sms as usize * fit as usize).max(1) as u32;
        let mut argv = [(p as *mut Params).cast::<c_void>()];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid: Dim3 {
                x: grid,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: threads as u32,
                y: 1,
                z: 1,
            },
            dynamic_smem: smem,
            stream,
            attrs: &mut attr,
            num_attrs: 1,
        };
        // SAFETY: a kernel of the table taking `legion_q4k_mv` by value, from the launch's copy
        // of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe { cudaLaunchKernelExC(&config, func, argv.as_mut_ptr()) } {
            0 => Ok(()),
            e => Err(format!("q4k_mv: CUDA error {e}")),
        }
    }
}
