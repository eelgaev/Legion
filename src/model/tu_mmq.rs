//! Prompt matmuls on Turing GPUs without tensor cores (TU11x: GTX 16xx, MX450, MX550) with
//! ggml-cuda's MMQ bits (#1214): `w @ x` for a Q4_K, Q5_K, Q6_K or Q8_0 weight and
//! [`MIN_TOKENS`] to [`MAX_TOKENS`] F32 columns, as an external op
//! ([`crate::ggml::ExternalOp`]).
//!
//! On compute capability 7.5 ggml-cuda runs these matmuls on MMQ's int8 MMA path, which TU11x
//! chips execute without tensor cores: a GTX 1660 Ti runs `mma.sync` m8n8k16 at ~2.4 int8 TOPS
//! and dp4a at ~17 (a microbenchmark of both), and a 1024-token Q4_K prompt matmul at ~2.4
//! TFLOPS. The op (`src/cuda/tu_mmq.cu`) computes the same per-output chain on dp4a: ggml's
//! quantization of the input, ggml's tile loader for the weight, the chunks' exact int dots,
//! and ggml's float operations on them in ggml's order. That chain is an output's whole value
//! where ggml's stream-k grid gives every CTA one whole tile (no partial sums to add up:
//! [`ggml_unsplit`]). Where ggml splits the tiles' K blocks over its SMs instead (short prompt
//! chunks, 1024-row K/V projections), the op runs ggml's stream-k grid itself: the same runs of
//! K blocks per CTA and ggml's fixup, which adds the parts in ggml's order ([`plan`]; for
//! ggml's 128-column tiles only, [`ggml_tile_width`]); other plans stay on ggml's MMQ. The same
//! bits, so it is on by default on these GPUs (`legion serve --tu-mmq-ggml` keeps ggml's MMQ).
//!
//! The node is ggml's `mul_mat`'s shape (`[rows, n]` F32, the input and the weight its sources).
//! Its scratch (the quantized input) is a buffer per device and stream outside the graphs, as
//! ggml's pool is. The CPU implementation is a reference for the tests (the dequantized
//! product, not the GPU's bits).

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;

static TU_MMQ: ExternalOp = ExternalOp {
    name: "tu_mmq",
    cpu: cpu_tu_mmq,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::tu_mmq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the op takes (prompt chunks; smaller steps keep ggml's MMQ and MMVQ).
pub const MIN_TOKENS: i64 = 128;

/// Most columns the op takes.
pub const MAX_TOKENS: i64 = 8192;

/// Rows per output tile (`I` of ggml's MMQ config for these types).
pub const TILE_ROWS: i64 = 128;

/// Values per K iteration of the tile loop (`MMQ_ITER_K`): `k` a multiple of it.
const ITER_K: i64 = 256;

/// Bytes of a `block_q8_1_mmq` (128 values: 4 scales, 128 int8).
const Q8_1_MMQ_BYTES: usize = 144;

/// Values per `block_q8_1_mmq`.
const Q8_1_MMQ_VALUES: i64 = 128;

/// Bytes of the quantized input the scratch holds.
pub const MAX_QUANT_BYTES: usize = 16 << 20;

/// Most SMs a stream-k plan runs on (the fixup scratch holds a 128 x 128 tile per SM).
pub const MAX_STREAM_K_SMS: i64 = 64;

/// Bytes of the stream-k fixup scratch: a 128 x 128 F32 tile per CTA.
const FIXUP_BYTES: usize = MAX_STREAM_K_SMS as usize * (TILE_ROWS * TILE_ROWS) as usize * 4;

/// ggml-cuda's MMQ tile widths for rows a multiple of 128 (`mmq-config-ampere.cuh`, which
/// compute capability 7.5 takes, the same for the four types).
const GGML_WIDTHS: [i64; 11] = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Whether graphs built from now on may take the op ([`set_tu_mmq`]).
static ON: AtomicBool = AtomicBool::new(true);

/// The engine GPU's SMs where it is one the op runs on (a TU11x), else 0.
static GPU_SMS: AtomicI32 = AtomicI32::new(0);

/// Sets whether graphs built from now on run the prompt matmuls the op takes on it (`true`, the
/// default); `false` (`legion serve --tu-mmq-ggml`) keeps ggml's MMQ for all.
pub fn set_tu_mmq(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: `sms` streaming multiprocessors when it is a Turing GPU without tensor
/// cores ([`crate::hw::per_device::turing_without_tensor_cores`]), else 0. Only weights whose
/// device a tensor doesn't name go by it; the rest ask their own device.
pub fn set_gpu(sms: i32) {
    GPU_SMS.store(sms.max(0), Ordering::Relaxed);
}

/// The SMs [`set_gpu`] stored.
pub fn gpu_sms() -> i32 {
    GPU_SMS.load(Ordering::Relaxed)
}

/// The SMs of `w`'s GPU where the op runs there, else 0: asked of the weight's own device (a
/// process may hold engines on GPUs of different kinds, so the last engine loaded doesn't
/// decide for the others); the engine's GPU ([`gpu_sms`]) only where the device is unknown.
fn sms_for(w: Tensor<'_>) -> i32 {
    w.cuda_device().map_or_else(gpu_sms, |d| {
        if crate::hw::per_device::turing_without_tensor_cores(d) {
            crate::hw::per_device::sm_count(d)
        } else {
            0
        }
    })
}

/// ggml-cuda's MMQ tile width for `n` columns (`ggml_cuda_mul_mat_q`, rows a multiple of 128):
/// the first width, from 8 up, with the fewest column tiles.
pub fn ggml_tile_width(n: i64) -> i64 {
    let (mut best_j, mut best) = (0, i64::MAX);
    for j in GGML_WIDTHS {
        let ntx = (n + j - 1) / j;
        if ntx < best {
            (best_j, best) = (j, ntx);
        }
        if best == 1 {
            break;
        }
    }
    best_j
}

/// Whether ggml-cuda's MMQ gives every CTA of its grid one whole tile for `rows` (a multiple
/// of 128) times `n` columns on `sms` SMs (`launch_mul_mat_q`: a CTA per tile when the tiles
/// fill their waves to 90%, else `sms` CTAs sharing the tiles' K blocks, whose partial sums a
/// fixup adds up). Only then is each output the chain the op computes.
pub fn ggml_unsplit(rows: i64, n: i64, sms: i64) -> bool {
    if sms <= 0 {
        return false;
    }
    let j = ggml_tile_width(n);
    let ntiles = (n + j - 1) / j * (rows / TILE_ROWS);
    let waves = (ntiles + sms - 1) / sms;
    100 * ntiles / (sms * waves) >= 90
}

/// How the op runs `rows` times `n` columns on `sms` SMs: `(tile width, stream_k)`. Where
/// ggml runs the plan unsplit, a CTA per tile of [`tile_width`] (the chains don't depend on the
/// tiling); else ggml's stream-k grid of `sms` CTAs on its 128-column tiles, which only mirrors
/// ggml where ggml's tiles are 128 wide (`None` otherwise, or past [`MAX_STREAM_K_SMS`]).
pub fn plan(rows: i64, n: i64, sms: i64) -> Option<(i64, bool)> {
    if sms <= 0 {
        None
    } else if ggml_unsplit(rows, n, sms) {
        Some((tile_width(rows, n, sms), false))
    } else if ggml_tile_width(n) == TILE_ROWS && sms <= MAX_STREAM_K_SMS {
        Some((TILE_ROWS, true))
    } else {
        None
    }
}

/// The op's tile width for `rows` times `n` columns on `sms` SMs: 128 or 64 columns, the one
/// with the fewest column-waves (waves of CTAs times the width), 128 on a tie.
pub fn tile_width(rows: i64, n: i64, sms: i64) -> i64 {
    let cost = |j: i64| {
        let ctas = (n + j - 1) / j * (rows / TILE_ROWS);
        (ctas + sms - 1) / sms * j
    };
    if cost(64) < cost(128) { 64 } else { 128 }
}

/// Bytes of the quantized input of `n` columns of `k` values for tiles `j` wide (the columns
/// padded to whole tiles).
pub fn quant_bytes(k: i64, n: i64, j: i64) -> usize {
    let n_pad = (n + j - 1) / j * j;
    (k / Q8_1_MMQ_VALUES * n_pad) as usize * Q8_1_MMQ_BYTES
}

/// The weight types the op takes, with their bytes per block and values per block.
fn block(ty: ffi::ggml_type) -> Option<(usize, i64)> {
    match ty {
        t if t == types::Q4_K => Some((144, 256)),
        t if t == types::Q5_K => Some((176, 256)),
        t if t == types::Q6_K => Some((210, 256)),
        t if t == types::Q8_0 => Some((34, 32)),
        _ => None,
    }
}

/// The index of `ty` in the kernel table's matmuls (`legion_tu_mmq_kernels`).
fn type_index(ty: ffi::ggml_type) -> usize {
    match ty {
        t if t == types::Q4_K => 0,
        t if t == types::Q5_K => 1,
        t if t == types::Q6_K => 2,
        _ => 3,
    }
}

/// Whether `ty`'s input quantization carries the block sums (`MMQ_Q8_1_DS_LAYOUT_DS4`: Q4_K,
/// Q5_K) or only the scales (`_D4`: Q6_K, Q8_0).
fn ds4(ty: ffi::ggml_type) -> bool {
    ty == types::Q4_K || ty == types::Q5_K
}

/// Whether `w @ x` takes the op: switched on, a TU11x GPU holding the weight in its memory,
/// shapes the op takes ([`fits`]) and a plan ggml runs unsplit ([`ggml_unsplit`]).
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    if !ON.load(Ordering::Relaxed) || !cfg!(feature = "cuda") || w.is_host() || w.is_meta() {
        return false;
    }
    let sms = sms_for(w) as i64;
    sms > 0 && fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb()), sms)
}

/// [`takes`] on shapes: a Q4_K, Q5_K, Q6_K or Q8_0 weight `[k, rows]` with contiguous rows,
/// `k` a multiple of 256 and `rows` of 128; an F32 `x` `[k, n]`, [`MIN_TOKENS`] to
/// [`MAX_TOKENS`] columns, contiguous values; a [`plan`] on `sms` SMs; at most
/// [`MAX_QUANT_BYTES`] of quantized input.
pub fn fits(
    w: (ffi::ggml_type, [i64; 4], [usize; 4]),
    x: (ffi::ggml_type, [i64; 4], [usize; 4]),
    sms: i64,
) -> bool {
    let (wty, [k, rows, w2, w3], wnb) = w;
    let (xty, [xk, n, x2, x3], xnb) = x;
    let Some((bytes, qk)) = block(wty) else {
        return false;
    };
    xty == types::F32
        && k > 0
        && k % ITER_K == 0
        && xk == k
        && rows > 0
        && rows % TILE_ROWS == 0
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && wnb[0] == bytes
        && wnb[1] % bytes == 0
        && wnb[1] >= (k / qk) as usize * bytes
        && rows * ((wnb[1] / bytes) as i64) < 1 << 31
        && xnb[0] == 4
        && xnb[1] % 4 == 0
        && (MIN_TOKENS..=MAX_TOKENS).contains(&n)
        && plan(rows, n, sms).is_some_and(|(j, _)| quant_bytes(k, n, j) <= MAX_QUANT_BYTES)
}

/// `w @ x` (see [`takes`]) on the op: F32 `[rows, n]`, ggml's `mul_mat`'s shape, with its bits.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    let sms = sms_for(w) as i64;
    mul_mat_sms(ctx, w, x, sms)
}

/// [`mul_mat`] planned for `sms` SMs (tests).
pub fn mul_mat_sms<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, sms: i64) -> Tensor<'a> {
    assert!(
        fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb()), sms),
        "tu_mmq: shapes"
    );
    let (rows, n) = (w.ne()[1], x.ne()[1]);
    let (j, stream_k) = plan(rows, n, sms).expect("tu_mmq: plan");
    ctx.external_with(
        &TU_MMQ,
        types::F32,
        [rows, n, 1, 1],
        &[x, w],
        &[j as i32, i32::from(stream_k), sms as i32],
    )
}

unsafe fn cpu_tu_mmq(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let (k, n, rows) = (x.ne[0] as usize, x.ne[1] as usize, w.ne[1] as usize);
        let row_bytes = crate::ggml::row_size(w.type_, k);
        let out = dst.data as *mut f32;
        for r in (ith..rows).step_by(nth) {
            let src = std::slice::from_raw_parts((w.data as *const u8).add(r * w.nb[1]), row_bytes);
            let wr = crate::ggml::dequantize(w.type_, src, k);
            for c in 0..n {
                let col = (x.data as *const u8).add(c * x.nb[1]) as *const f32;
                let xc = std::slice::from_raw_parts(col, k);
                *out.add(c * rows + r) = wr.iter().zip(xc).map(|(a, b)| a * b).sum();
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::collections::HashMap;
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI8, Ordering};

    use super::*;

    /// `legion_tu_mmq` of `src/cuda/tu_mmq.cu`.
    #[repr(C)]
    struct Params {
        w: *const c_void,
        x: *const f32,
        yq: *mut c_void,
        dst: *mut f32,
        fixup: *mut f32,
        x_ld: i64,
        dst_ld: i64,
        w_ld: i32,
        k: i32,
        n: i32,
        n_pad: i32,
        rows: i32,
    }

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// Kernels in `legion_tu_mmq_kernels`: the quantization (D4, DS4), then per weight type
    /// the matmul with tiles 128 and 64 columns wide, the stream-k matmul and its fixup.
    const KERNELS: usize = 18;

    unsafe extern "C" {
        static legion_tu_mmq_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;

    /// Devices the shared-memory cache covers.
    const DEVICES: usize = 64;

    /// Each device's shared-memory opt-in of each kernel: 0 not yet, 1 done, -1 refused.
    static SMEM: [[AtomicI8; KERNELS]; DEVICES] =
        [const { [const { AtomicI8::new(0) }; KERNELS] }; DEVICES];

    /// The quantized input of each (device, stream), allocated on first use at
    /// [`MAX_QUANT_BYTES`] (then the stream-k fixup's [`FIXUP_BYTES`]) and kept: graphs
    /// captured with the pointer replay it. Ops on one stream run one after the other, so they
    /// share it as ggml's MMQ shares its pool.
    static SCRATCH: Mutex<Option<HashMap<(c_int, usize), usize>>> = Mutex::new(None);

    fn scratch(dev: c_int, stream: *mut c_void) -> Result<usize, String> {
        let mut map = SCRATCH.lock().unwrap();
        let map = map.get_or_insert_with(HashMap::new);
        if let Some(&s) = map.get(&(dev, stream as usize)) {
            return Ok(s);
        }
        let mut q = std::ptr::null_mut();
        // SAFETY: a plain allocation on the current device.
        if unsafe { cudaMalloc(&mut q, MAX_QUANT_BYTES + FIXUP_BYTES) } != 0 {
            // SAFETY: clears the error the failed call left.
            unsafe { cudaGetLastError() };
            return Err(format!(
                "tu_mmq: cudaMalloc of {} bytes",
                MAX_QUANT_BYTES + FIXUP_BYTES
            ));
        }
        map.insert((dev, stream as usize), q as usize);
        Ok(q as usize)
    }

    fn opt_in(dev: c_int, kernel: usize, bytes: usize) -> Result<(), String> {
        let slot = &SMEM[dev as usize][kernel];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            state = unsafe {
                if cudaFuncSetAttribute(
                    legion_tu_mmq_kernels[kernel],
                    FUNC_ATTR_MAX_DYNAMIC_SHARED,
                    bytes as c_int,
                ) == 0
                {
                    1
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(state, Ordering::Relaxed);
        }
        if state < 0 {
            return Err(format!("tu_mmq: {bytes} bytes of shared memory"));
        }
        Ok(())
    }

    fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        shared: usize,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut Params).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `legion_tu_mmq` by value, from the launch's copy
        // of `p`; its pointers are device memory of the stream's device.
        match unsafe {
            cudaLaunchKernel(
                legion_tu_mmq_kernels[kernel],
                grid,
                block,
                argv.as_mut_ptr(),
                shared,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("tu_mmq: CUDA error {e}")),
        }
    }

    /// The matmul's dynamic shared memory for tiles `j` wide: the input tile (`j` columns of
    /// 36 words), then the weight tile (128 rows of 76 words, ggml's MMA layout).
    pub(super) fn smem(j: i64) -> usize {
        (j as usize * 36 + TILE_ROWS as usize * 76) * 4
    }

    pub(super) unsafe fn tu_mmq(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat` (its shapes passed `fits`), device tensors of the
        // stream's device, the current one.
        unsafe {
            let (x, w) = (&*dst.src[0], &*dst.src[1]);
            let params = crate::ggml::external_params(dst);
            let (j, stream_k, sms) = (params[0] as i64, params[1] != 0, params[2] as u32);
            let mut dev: c_int = 0;
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                return Err("tu_mmq: no current device".into());
            }
            let yq = scratch(dev, stream)?;
            let (k, n, rows) = (x.ne[0], x.ne[1], w.ne[1]);
            let (bytes, _) = block(w.type_).ok_or("tu_mmq: weight type")?;
            let n_pad = (n + j - 1) / j * j;
            let mut a = Params {
                w: w.data as *const c_void,
                x: x.data as *const f32,
                yq: yq as *mut c_void,
                dst: dst.data as *mut f32,
                fixup: (yq + MAX_QUANT_BYTES) as *mut f32,
                x_ld: (x.nb[1] / 4) as i64,
                dst_ld: (dst.nb[1] / 4) as i64,
                w_ld: (w.nb[1] / bytes) as i32,
                k: k as i32,
                n: n as i32,
                n_pad: n_pad as i32,
                rows: rows as i32,
            };
            let quant_grid = Dim3 {
                x: n as u32,
                y: ((k + 511) / 512) as u32,
                z: 1,
            };
            let quant_block = Dim3 { x: 128, y: 1, z: 1 };
            launch(
                ds4(w.type_) as usize,
                quant_grid,
                quant_block,
                0,
                &mut a,
                stream,
            )?;
            let base = 2 + 4 * type_index(w.type_);
            let kernel = base + if stream_k { 2 } else { (j == 64) as usize };
            let shared = smem(j);
            opt_in(dev, kernel, shared)?;
            let block = Dim3 { x: 32, y: 8, z: 1 };
            if !stream_k {
                let grid = Dim3 {
                    x: (rows / TILE_ROWS) as u32,
                    y: (n_pad / j) as u32,
                    z: 1,
                };
                return launch(kernel, grid, block, shared, &mut a, stream);
            }
            // ggml's stream-k grid: a CTA per SM, then the fixup (a CTA per SM and 32 rows).
            let grid = Dim3 { x: sms, y: 1, z: 1 };
            launch(kernel, grid, block, shared, &mut a, stream)?;
            let grid = Dim3 {
                x: sms,
                y: (TILE_ROWS / 32) as u32,
                z: 1,
            };
            let block = Dim3 { x: 32, y: 4, z: 1 };
            launch(base + 3, grid, block, 0, &mut a, stream)
        }
    }
}
