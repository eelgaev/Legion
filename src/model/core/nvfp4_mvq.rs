//! NVFP4 dense matvecs of decode and verify steps on tensor cores with ggml's bits (#1133):
//! `w @ x` of a 2D NVFP4 weight for 1-[`MAX_N`] columns, times a one-value scale when the
//! checkpoint has one, as an external op running `src/cuda/sm120/nvfp4_mvq.cu` in place of
//! ggml-cuda's quantize_q8_1 + `mul_mat_vec_q` pair and the `mul` after it.
//!
//! ggml's MMVQ does the NVFP4 sub-block dot products on the CUDA cores (E2M1 table lookups,
//! dp4a, then the float scale math): on Qwen3.8-27B's verify steps (8 columns) it is close to
//! half the GPU time of a C=1 round, and the chip draws more for it than for the weight reads.
//! The op takes each sub-block's integer sum from an F16 tensor-core MMA (exact: small
//! integers) and does the float part in ggml's order, so every output keeps ggml's bits (see
//! the kernel file).
//!
//! Off by default: on a GB10 it is 1.17-1.25x slower than ggml's MMVQ per matmul (its 16-row
//! weight reads stream about 20% slower), though at 8 columns it draws 48 W against ggml's
//! 79-91 W. `legion serve --nvfp4-mvq` ([`set_nvfp4_mvq`]) takes it on compute capability
//! 12.x (GB10, RTX 50; [`set_gpu`]) where the binary has the sm_120a/sm_121a code, for steps
//! of [`MIN_N`]..=[`MAX_N`] columns. `legion
//! nvfp4-mvq-bench` checks the bits against ggml's and times both ([`bench`]). The CPU
//! implementation is a plain f32 reference, for tests of the wiring.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static NVFP4_MVQ: ExternalOp = ExternalOp {
    name: "nvfp4_mvq",
    cpu: cpu_nvfp4_mvq,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_nvfp4_mvq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the model's matmuls take the op at: at one column every MMA would carry one
/// useful column of 8, and ggml's one-column MMVQ is cheaper.
pub const MIN_N: i64 = 2;

/// Most columns: ggml-cuda runs NVFP4 matmuls on MMVQ up to 8 ([`super::linear::NVFP4_GGML_MAX_N`]),
/// one MMA's columns.
pub const MAX_N: i64 = 8;

/// Longest rows (values) of the device-global activation scratch (the kernel's `MAX_K`).
pub const MAX_K: i64 = 65536;

/// The weights' rows are a multiple of this: a CTA's rows.
pub const ROW_MULTIPLE: i64 = 16;

/// The weights' rows (values) are a multiple of this: 4 NVFP4 blocks, so the kernel's
/// stages of 16 blocks end on whole 16-byte loads.
const K_MULTIPLE: i64 = 256;

/// Bytes per NVFP4 block.
const BLOCK_BYTES: usize = 36;

static ON: AtomicBool = AtomicBool::new(false);
static GPU: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on take the op where it fits (`legion serve --nvfp4-mvq`) or
/// ggml's MMVQ (the default).
pub fn set_nvfp4_mvq(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: the op is used on compute capability 12.x (GB10, RTX 50; the SM it was
/// checked and measured on, #1133).
pub fn set_gpu(cc: Option<(i32, i32)>) {
    GPU.store(gpu_takes(cc), Ordering::Relaxed);
}

/// Whether compute capability `cc` (major, minor) takes the op ([`set_gpu`]).
pub fn gpu_takes(cc: Option<(i32, i32)>) -> bool {
    cc.is_some_and(|(major, _)| major == 12)
}

fn on() -> bool {
    ON.load(Ordering::Relaxed) && GPU.load(Ordering::Relaxed) && cfg!(feature = "cuda")
}

/// Whether [`super::linear::Linear::mul_mat`] takes `w @ x` times `scale` on the op: switched
/// on, a 12.x GPU whose code here has the FP4 conversions, [`MIN_N`]..=[`MAX_N`] columns and
/// [`fits`].
pub fn takes(w: Tensor<'_>, x: Tensor<'_>, scale: Option<Tensor<'_>>) -> bool {
    on() && (MIN_N..=MAX_N).contains(&x.ne()[1])
        && fits(w, x, scale)
        && w.cuda_device()
            .is_some_and(crate::model::nvfp4_a4::supported_on)
}

/// Whether the op can compute `w @ x` (times `scale`) with ggml's MMVQ bits: a 2D NVFP4 weight
/// in device memory (rows of a multiple of 256 values, at most [`MAX_K`], 16-byte strides, a
/// multiple of [`ROW_MULTIPLE`] rows), F32 activations `[k, n]` of 1-[`MAX_N`] columns, and no scale or an
/// F32 one of one value.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>, scale: Option<Tensor<'_>>) -> bool {
    let [k, rows, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let (wnb, xnb) = (w.nb(), x.nb());
    w.ty() == types::NVFP4
        && k > 0
        && k % K_MULTIPLE == 0
        && k <= MAX_K
        && xk == k
        && rows > 0
        && rows % ROW_MULTIPLE == 0
        && rows <= i32::MAX as i64
        && w2 == 1
        && w3 == 1
        && (1..=MAX_N).contains(&n)
        && x2 == 1
        && x3 == 1
        && x.ty() == types::F32
        && xnb[0] == 4
        && xnb[1].is_multiple_of(4)
        && wnb[0] == BLOCK_BYTES
        && wnb[1].is_multiple_of(16)
        && !w.is_host()
        && !w.is_meta()
        && scale.is_none_or(|s| {
            s.ty() == types::F32 && s.ne() == [1, 1, 1, 1] && !s.is_host() && !s.is_meta()
        })
}

/// `w @ x` (F32 `[rows, n]`), times `scale` when there is one, on the op; the caller checked
/// [`fits`] (and [`takes`] for a served graph).
pub fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let mut args = vec![w, x];
    args.extend(scale);
    ctx.external_with(
        &NVFP4_MVQ,
        types::F32,
        [w.ne()[1], x.ne()[1], 1, 1],
        &args,
        &[i32::from(scale.is_some())],
    )
}

struct Operands {
    w: *const u8,
    w_nb1: usize,
    x: *const u8,
    x_nb1: usize,
    scale: *const u8,
    dst: *mut u8,
    dst_nb1: usize,
    k: usize,
    rows: usize,
    n: usize,
}

impl Operands {
    /// # Safety
    /// `dst` is a node made by [`mul_mat`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        let [scaled, ..] = external_params(dst);
        unsafe {
            let (w, x) = (&*dst.src[0], &*dst.src[1]);
            Self {
                w: w.data as *const u8,
                w_nb1: w.nb[1],
                x: x.data as *const u8,
                x_nb1: x.nb[1],
                scale: if scaled != 0 {
                    (*dst.src[2]).data as *const u8
                } else {
                    std::ptr::null()
                },
                dst: dst.data as *mut u8,
                dst_nb1: dst.nb[1],
                k: w.ne[0] as usize,
                rows: w.ne[1] as usize,
                n: x.ne[1] as usize,
            }
        }
    }
}

/// # Safety
/// `dst` is a node made by [`mul_mat`], all tensors in host memory.
unsafe fn cpu_nvfp4_mvq(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, all tensors in host memory.
    let o = unsafe { Operands::of(dst) };
    let row_bytes = crate::ggml::row_size(types::NVFP4, o.k);
    // SAFETY: the scale (one F32) when there is one.
    let scale = (!o.scale.is_null()).then(|| unsafe { *(o.scale as *const f32) });
    for r in (ith..o.rows).step_by(nth) {
        // SAFETY: in-bounds reads of row r and of x's columns; writes to dst's column entries.
        unsafe {
            let row = std::slice::from_raw_parts(o.w.add(r * o.w_nb1), row_bytes);
            let w = crate::ggml::dequantize(types::NVFP4, row, o.k);
            for j in 0..o.n {
                let x = std::slice::from_raw_parts(o.x.add(j * o.x_nb1) as *const f32, o.k);
                let v: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                *(o.dst.add(j * o.dst_nb1) as *mut f32).add(r) = scale.map_or(v, |s| v * s);
            }
        }
    }
}

/// The matvec of the kernel table for `n` columns: ggml's warps for the column count (4 up to
/// 4 columns, 2 above).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn kernel_index(n: usize) -> usize {
    1 + usize::from(n > 4)
}

/// `legion_nvfp4_mvq` of `src/cuda/sm120/nvfp4_mvq.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Params {
    x: *const f32,
    x_nb1: i64,
    w: *const c_void,
    w_nb1: i64,
    dst: *mut f32,
    dst_nb1: i64,
    scale: *const f32,
    k: i32,
    rows: i32,
    n: i32,
}

/// # Safety
/// `dst` is a node made by [`mul_mat`] (its shapes passed [`fits`]), its tensors in device
/// memory of the stream's device; `stream` is the compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_nvfp4_mvq(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat`.
    let o = unsafe { Operands::of(dst) };
    let mut p = Params {
        x: o.x as *const f32,
        x_nb1: (o.x_nb1 / 4) as i64,
        w: o.w as *const c_void,
        w_nb1: o.w_nb1 as i64,
        dst: o.dst as *mut f32,
        dst_nb1: (o.dst_nb1 / 4) as i64,
        scale: o.scale as *const f32,
        k: o.k as i32,
        rows: o.rows as i32,
        n: o.n as i32,
    };
    let quantize = launch::Dim3 {
        x: o.k.div_ceil(256) as u32,
        y: o.n as u32,
        z: 1,
    };
    launch::launch(
        0,
        quantize,
        launch::Dim3 { x: 256, y: 1, z: 1 },
        &mut p,
        stream,
    )?;
    if !(o.w as usize).is_multiple_of(16) {
        return Err("nvfp4_mvq: weight not 16-byte aligned".into());
    }
    let grid = launch::Dim3 {
        x: (o.rows / ROW_MULTIPLE as usize) as u32,
        y: 1,
        z: 1,
    };
    let block = launch::Dim3 { x: 128, y: 1, z: 1 };
    launch::launch(kernel_index(o.n), grid, block, &mut p, stream)
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI32, Ordering};

    /// `dim3`.
    #[repr(C)]
    pub(super) struct Dim3 {
        pub x: u32,
        pub y: u32,
        pub z: u32,
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

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs): the quantization,
        /// then the matvecs (`super::kernel_index`).
        static legion_nvfp4_mvq_kernels: [*const c_void; 3];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const DEV_ATTR_CC_MAJOR: c_int = 75;

    /// Devices the cache below covers.
    const DEVICES: usize = 64;

    /// Each device's compute capability major; 0 not yet read, -1 unknown.
    static MAJOR: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// The current device's compute capability major, read once per device.
    fn major() -> Option<i32> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return None;
            }
            let slot = &MAJOR[dev as usize];
            let mut m = slot.load(Ordering::Relaxed);
            if m == 0 {
                m = if cudaDeviceGetAttribute(&mut m, DEV_ATTR_CC_MAJOR, dev) == 0 {
                    m
                } else {
                    cudaGetLastError();
                    -1
                };
                slot.store(m, Ordering::Relaxed);
            }
            (m > 0).then_some(m)
        }
    }

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream`; on compute
    /// capability 12.x as a programmatic dependent launch, as ggml-cuda launches its own
    /// kernels there (each kernel waits for the one ahead before it reads activations).
    pub(super) fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        p: &mut super::Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut super::Params).cast::<c_void>()];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid,
            block,
            dynamic_smem: 0,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(major() == Some(12)),
        };
        // SAFETY: a kernel of the table taking `legion_nvfp4_mvq` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe {
            cudaLaunchKernelExC(&config, legion_nvfp4_mvq_kernels[kernel], argv.as_mut_ptr())
        } {
            0 => Ok(()),
            e => Err(format!("nvfp4_mvq: CUDA error {e}")),
        }
    }
}
