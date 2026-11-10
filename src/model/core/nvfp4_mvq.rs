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
//! On by default on compute capability 12.x (GB10, RTX 50; [`set_gpu`]) where the binary has
//! the sm_120a/sm_121a code, for steps of [`MIN_N`]..=[`MAX_N`] columns; `legion serve
//! --no-nvfp4-mvq` ([`set_nvfp4_mvq`]) keeps ggml's MMVQ. On a GB10 the matmul streams the
//! weights at about ggml's speed (Qwen3.8-27B's FFN shapes at 8 columns: within 2% of MMVQ)
//! while the tensor cores take the arithmetic. `legion nvfp4-mvq-bench` checks the bits
//! against ggml's and times both ([`bench`]). The CPU implementation is a plain f32
//! reference, for tests of the wiring.

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
/// useful column of 8, and ggml's one-column MMVQ is cheaper; at 2-4 the op is still 4-7%
/// slower per matmul on a GB10, so those stay on MMVQ too.
pub const MIN_N: i64 = 5;

/// Most columns: ggml-cuda runs NVFP4 matmuls on MMVQ up to 8 ([`super::linear::NVFP4_GGML_MAX_N`]),
/// one MMA's columns.
pub const MAX_N: i64 = 8;

/// Longest rows (values) of the device-global activation scratch (the kernel's `MAX_K`).
pub const MAX_K: i64 = 65536;

/// The weights' rows are a multiple of this: the kernel's groups of rows (one MMA's weights).
pub const ROW_MULTIPLE: i64 = 4;

/// The weights' rows (values) are a multiple of this: 4 NVFP4 blocks, so a row is whole
/// 16-byte pieces (the bulk copies' unit).
const K_MULTIPLE: i64 = 256;

/// Bytes per NVFP4 block.
const BLOCK_BYTES: usize = 36;

/// Shared memory a CTA may take on compute capability 12.x (99 KiB; the launch asks the device).
const SMEM_12X: usize = 99 * 1024;

/// The kernel's warps (`MV_WARPS`).
const WARPS: usize = 8;

/// Rows of a CTA's region (`MV_R`).
const REGION: usize = 16;

/// Pieces the ring holds beyond a chunk of a region (the next chunk's first rows in flight;
/// 12 measured best of 4-16 on a GB10 at 8 columns).
const AHEAD: usize = 12;

/// A launch's shape: each row in `chunks` pieces along K, `ns` piece slots in a CTA's ring,
/// `smem` bytes of dynamic shared memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub chunks: usize,
    pub ns: usize,
    pub smem: usize,
}

impl Plan {
    /// The plan for rows of `k` values in `chunks` pieces with `ns` slots, if it fits in `smem`
    /// bytes, the pieces are whole 16-byte units and each spans at most two of ggml's 64-block
    /// steps (the kernel holds a piece's activations in registers: `MV_ITERS`).
    pub fn with(k: usize, chunks: usize, ns: usize, smem: usize) -> Option<Plan> {
        let bpr = k / 64;
        if chunks == 0 || !bpr.is_multiple_of(4 * chunks) || ns <= REGION {
            return None;
        }
        let cb = bpr / chunks;
        if (0..chunks).any(|c| c * cb % 64 + cb > 128) {
            return None;
        }
        let pitch = bpr / chunks * BLOCK_BYTES + 16;
        let part = WARPS * 2 * REGION * 8 * 4;
        let bytes = ns * (pitch + 8) + part;
        (bytes <= smem).then_some(Plan {
            chunks,
            ns,
            smem: bytes,
        })
    }
}

/// The kernel's shape for rows of `k` values on a device with `smem` bytes of shared memory a
/// CTA: the fewest pieces a row (1, 2, 4, 8 or 16) whose ring holds a chunk of a region and
/// [`AHEAD`] more pieces; `None` when none fits (or the rows are not whole groups).
pub fn plan(k: usize, rows: usize, smem: usize) -> Option<Plan> {
    if !rows.is_multiple_of(ROW_MULTIPLE as usize) {
        return None;
    }
    [1, 2, 4, 8, 16]
        .into_iter()
        .find_map(|c| Plan::with(k, c, REGION + exp_env("LEGION_MVQ_AHEAD", AHEAD), smem))
}

/// EXPERIMENT (temporary): an env knob read once per name.
fn exp_env(name: &'static str, default: usize) -> usize {
    use std::sync::OnceLock;
    static A: OnceLock<usize> = OnceLock::new();
    static E: OnceLock<usize> = OnceLock::new();
    let cell = if name == "LEGION_MVQ_AHEAD" { &A } else { &E };
    *cell.get_or_init(|| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    })
}

static ON: AtomicBool = AtomicBool::new(true);
static GPU: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on take the op where it fits (the default) or ggml's MMVQ
/// (`legion serve --no-nvfp4-mvq`).
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
        && plan(k as usize, rows as usize, SMEM_12X).is_some()
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
    ns: i32,
    chunks: i32,
    exp: i32,
}

/// # Safety
/// `dst` is a node made by [`mul_mat`] (its shapes passed [`fits`]), its tensors in device
/// memory of the stream's device; `stream` is the compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_nvfp4_mvq(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat`.
    let o = unsafe { Operands::of(dst) };
    if !(o.w as usize).is_multiple_of(16) {
        return Err("nvfp4_mvq: weight not 16-byte aligned".into());
    }
    let dev = launch::device()?;
    let plan = plan(o.k, o.rows, dev.smem.min(SMEM_12X))
        .ok_or_else(|| format!("nvfp4_mvq: no ring for k {} rows {}", o.k, o.rows))?;
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
        ns: plan.ns as i32,
        chunks: plan.chunks as i32,
        exp: exp_env("LEGION_MVQ_EXP", 0) as i32,
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
        0,
        &mut p,
        stream,
    )?;
    // One CTA per SM, each taking a run of rows.
    let grid = launch::Dim3 {
        x: (o.rows / ROW_MULTIPLE as usize).min(dev.sms) as u32,
        y: 1,
        z: 1,
    };
    let ws = exp_env("LEGION_MVQ_EXP", 0) & 1024 != 0;
    let block = launch::Dim3 {
        x: (WARPS * 32 + if ws { 32 } else { 0 }) as u32,
        y: 1,
        z: 1,
    };
    launch::launch(kernel_index(o.n), grid, block, plan.smem + if ws { plan.ns * 8 } else { 0 }, &mut p, stream)
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
        static legion_nvfp4_mvq_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    /// Entries of the kernel table.
    const KERNELS: usize = 3;
    const DEV_ATTR_SM_COUNT: c_int = 16;
    const DEV_ATTR_SMEM_OPTIN: c_int = 97;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const FUNC_ATTR_MAX_DYNAMIC_SMEM: c_int = 8;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// Each device's compute capability major, SM count and opt-in shared memory per
    /// block; 0 not yet read, -1 unknown.
    static MAJOR: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];
    static SMS: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];
    static SMEM: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];
    /// The dynamic shared memory each kernel was opted into per device.
    static OPTED: [[AtomicI32; KERNELS]; DEVICES] =
        [const { [const { AtomicI32::new(0) }; KERNELS] }; DEVICES];

    /// What the launch needs of the current device.
    pub(super) struct Device {
        pub id: usize,
        pub major: i32,
        pub sms: usize,
        pub smem: usize,
    }

    /// Reads `attr` of `dev` once into `slot`.
    ///
    /// # Safety
    /// `dev` is a valid device ordinal.
    unsafe fn cached(slot: &AtomicI32, attr: c_int, dev: c_int) -> i32 {
        let mut m = slot.load(Ordering::Relaxed);
        if m == 0 {
            // SAFETY: a plain attribute query.
            m = if unsafe { cudaDeviceGetAttribute(&mut m, attr, dev) } == 0 && m > 0 {
                m
            } else {
                // SAFETY: clears the error the failed query left.
                unsafe { cudaGetLastError() };
                -1
            };
            slot.store(m, Ordering::Relaxed);
        }
        m
    }

    /// The current device, its attributes read once per device.
    pub(super) fn device() -> Result<Device, String> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return Err("nvfp4_mvq: no current device".into());
            }
            let i = dev as usize;
            let major = cached(&MAJOR[i], DEV_ATTR_CC_MAJOR, dev);
            let sms = cached(&SMS[i], DEV_ATTR_SM_COUNT, dev);
            let smem = cached(&SMEM[i], DEV_ATTR_SMEM_OPTIN, dev);
            if sms <= 0 || smem <= 0 {
                return Err("nvfp4_mvq: device attributes unknown".into());
            }
            Ok(Device {
                id: i,
                major,
                sms: sms as usize,
                smem: smem as usize,
            })
        }
    }

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream` with `smem` bytes
    /// of dynamic shared memory; on compute capability 12.x as a programmatic dependent
    /// launch, as ggml-cuda launches its own kernels there (each kernel waits for the one
    /// ahead before it reads activations).
    pub(super) fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        smem: usize,
        p: &mut super::Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let dev = device()?;
        let func = unsafe { legion_nvfp4_mvq_kernels[kernel] };
        let opted = &OPTED[dev.id][kernel];
        if smem as i32 > opted.load(Ordering::Relaxed) {
            // SAFETY: an attribute of a kernel of the table, within the device's opt-in.
            if unsafe { cudaFuncSetAttribute(func, FUNC_ATTR_MAX_DYNAMIC_SMEM, smem as c_int) } != 0
            {
                unsafe { cudaGetLastError() };
                return Err(format!("nvfp4_mvq: cannot opt into {smem} B shared memory"));
            }
            opted.store(smem as i32, Ordering::Relaxed);
        }
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
            dynamic_smem: smem,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(dev.major == 12),
        };
        // SAFETY: a kernel of the table taking `legion_nvfp4_mvq` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe { cudaLaunchKernelExC(&config, func, argv.as_mut_ptr()) } {
            0 => Ok(()),
            e => Err(format!("nvfp4_mvq: CUDA error {e}")),
        }
    }
}
