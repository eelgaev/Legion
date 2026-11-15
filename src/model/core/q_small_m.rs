//! Quantized dense matmuls of 2-8 columns on a V100 with ggml-cuda's bits (#1210): MTP verify
//! steps and small decode batches of a GGUF checkpoint's projections.
//!
//! ggml-cuda runs `mul_mat(w, x)` of a quantized `w` and up to 8 columns on a V100 as
//! `quantize_q8_1` + `mul_mat_vec_q` (MMVQ) where `ggml_cuda_should_use_mmvq` says so (Q4_K up to
//! 5 columns, Q3_K and Q5_K 6, Q6_K 7, the other types 8; MMQ above). MMVQ's CTAs of 2 rows split
//! each row's blocks over 2-4 warps and reduce them through shared memory, with few weight loads
//! in flight per thread: on a V100 a Q4_K `ffn_gate` (17,408 x 5,120) takes ~1.5x a one-column
//! step's time at 4 columns and ~2.6x at 8 (1Cat-vLLM's table), so a verify step grows with its
//! rows and deeper `--spec-k` does not pay on the 27B.
//!
//! Here two external ops ([`crate::ggml::ExternalOp`], kernels in `src/cuda/q_small_m.cu`)
//! replace the pair: `q_small_m_quantize` (ggml's quantize_q8_1, value for value) and
//! `q_small_m` in one of [`VARIANTS`]: each warp whole rows (a lane does the work of the virtual
//! threads ggml's CTA gives that lane's position in each of its warps, each into a sum of its
//! own, in ggml's block order, then adds them in ggml's warp order and runs ggml's warp
//! reduction), or ggml's own CTA layout over 4 or 8 rows instead of 2. Each product is ggml's
//! own `vec_dot_*_q8_1` on the same values, so every output has ggml's bits.
//!
//! Measured on a V100 (ggml 0.27's MMVQ, the 27B's shapes) the op beats MMVQ on IQ4_XS only
//! (up to ~20% at 5-8 columns) and is 0-15% slower on Q4_K, Q5_K and Q6_K: MMVQ there is bound
//! by the dot products' integer work, which the bits fix, not by the loads in flight. So it is
//! opt-in (`legion serve --q-small-m`): compute capability 7.0, a type of [`TYPES`], weights in
//! one GPU's memory, and 2 columns up to ggml's MMVQ bound for the type ([`ggml_mmvq_max`]).
//! `--q-small-m-wide` also takes the steps ggml runs on MMQ, up to 8 columns: MMVQ's sums
//! there, not MMQ's, so the outputs move. One column stays on ggml (already at the bandwidth).
//! `legion q-small-m-bench` times the op against ggml's and checks its bits ([`bench`]). The
//! CPU implementation is a plain F32 reference, for tests of the wiring.
//!
//! Ideas from 1Cat-vLLM (Apache-2.0): `csrc/sm70_turbomind/ops/qpn_pair_sm70.cuh`,
//! `gguf_native_pair_readers_sm70.cuh`, `docs/design/sm70_awq_qpn_m1.md`,
//! `docs/design/gguf_native_sm70.md` (PR #966), and from v100-skinny (MIT). No code is taken
//! from them.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static QUANTIZE: ExternalOp = ExternalOp {
    name: "q_small_m_quantize",
    cpu: cpu_quantize,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_quantize),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static MV: ExternalOp = ExternalOp {
    name: "q_small_m",
    cpu: cpu_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The weight types the op takes, in the kernel table's order (`src/cuda/q_small_m.cu`).
pub const TYPES: [ffi::ggml_type; 7] = [
    types::Q4_K,
    types::Q5_K,
    types::Q6_K,
    types::IQ4_XS,
    types::Q8_0,
    types::IQ4_NL,
    types::Q3_K,
];

/// Fewest columns the op takes: one column stays on ggml's MMVQ.
pub const MIN_COLS: i64 = 2;

/// Most columns the op takes: ggml's `MMVQ_MAX_BATCH_SIZE`.
pub const MAX_COLS: i64 = 8;

/// The matvec variants, in the kernel table's order: `R > 0` the kernel `mv` with `R` rows a
/// warp, `-R` the kernel `mv_cta` (ggml's own CTA layout) with `R` rows a CTA.
pub const VARIANTS: [i32; 5] = [1, 2, 4, -4, -8];

/// Bytes of a Q8_1 block: the half2 (d, sum), 32 int8.
const Q8_1_BYTES: usize = 36;

/// Values per Q8_1 block.
const QK8_1: usize = 32;

/// Warps of a matvec CTA (the kernel's `NW`).
#[cfg(feature = "cuda")]
const NW: i64 = 4;

static ON: AtomicBool = AtomicBool::new(false);
static WIDE: AtomicBool = AtomicBool::new(false);

/// The variant ([`VARIANTS`]) of graphs built from now on, 0 for the default choice
/// ([`variant`]): for the bench and tests, the same bits either way.
static FORCED_R: AtomicI32 = AtomicI32::new(0);

/// Sets whether graphs built from now on run the matmuls the op takes on it (`legion serve
/// --q-small-m`; off by default, ggml's kernels).
pub fn set_q_small_m(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Sets whether graphs built from now on also run steps of up to [`MAX_COLS`] columns that
/// ggml runs on MMQ on the op (`legion serve --q-small-m-wide`, off by default: those outputs
/// move).
pub fn set_wide(on: bool) {
    WIDE.store(on, Ordering::Relaxed);
}

/// Forces the variant (one of [`VARIANTS`]; `None` the default choice). False for another
/// value.
#[doc(hidden)]
pub fn force_variant(r: Option<i32>) -> bool {
    match r {
        None => {
            FORCED_R.store(0, Ordering::Relaxed);
            true
        }
        Some(r) if VARIANTS.contains(&r) => {
            FORCED_R.store(r, Ordering::Relaxed);
            true
        }
        Some(_) => false,
    }
}

/// The most columns ggml-cuda runs on MMVQ for a weight of type `ty` on a V100
/// (`ggml_cuda_should_use_mmvq`, `cc == GGML_CUDA_CC_VOLTA`); MMQ above.
pub fn ggml_mmvq_max(ty: ffi::ggml_type) -> i64 {
    match ty {
        t if t == types::Q2_K => 4,
        t if t == types::Q4_K => 5,
        t if t == types::Q3_K || t == types::Q5_K => 6,
        t if t == types::Q6_K => 7,
        _ => MAX_COLS,
    }
}

/// The type's index in [`TYPES`].
fn type_index(ty: ffi::ggml_type) -> Option<usize> {
    TYPES.iter().position(|&t| t == ty)
}

/// The variant ([`VARIANTS`]) for a weight of type `ty` and `n` columns: the forced one, else
/// the fastest in `legion q-small-m-bench` on a V100 over the 27B's shapes. IQ4_XS: 2 rows a
/// warp up to 3 columns, 4 rows a CTA at 4-5, 4 rows a warp above. The other types: 4 rows a
/// CTA, 2 rows a warp at 8 columns.
pub fn variant(ty: ffi::ggml_type, n: i64) -> i32 {
    match FORCED_R.load(Ordering::Relaxed) {
        0 if ty == types::IQ4_XS => match n {
            ..=3 => 2,
            4 | 5 => -4,
            _ => 4,
        },
        0 if n >= 8 => 2,
        0 => -4,
        r => r,
    }
}

/// The most columns the op takes for `ty` on a V100 now: ggml's MMVQ bound, or [`MAX_COLS`]
/// with [`set_wide`].
pub fn max_cols(ty: ffi::ggml_type) -> i64 {
    if WIDE.load(Ordering::Relaxed) {
        MAX_COLS
    } else {
        ggml_mmvq_max(ty).min(MAX_COLS)
    }
}

/// Whether graphs built now run `w @ x` on the op: switched on, a CUDA build, [`fits`], a
/// weight in the memory of a compute capability 7.0 GPU (not host memory, not split), and a
/// column count up to [`max_cols`].
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && fits(w, x)
        && x.ne()[1] <= max_cols(w.ty())
        && !w.is_host()
        && !w.is_meta()
        && w.cuda_device()
            .and_then(crate::hw::per_device::compute_capability)
            == Some(70)
}

/// Whether the op computes `w @ x`: `w` a matrix of a type of [`TYPES`] (`[k, rows]`, rows
/// `nb[1]` bytes apart), `x` a contiguous F32 `[k, n]` with `n` of [`MIN_COLS`] to [`MAX_COLS`],
/// `k` a multiple of the type's block.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [k, rows, w2, w3] = w.ne();
    let [xk, n, x2, x3] = x.ne();
    let ty = w.ty();
    type_index(ty).is_some()
        && x.ty() == types::F32
        && x.is_contiguous()
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && xk == k
        && k > 0
        && k <= i32::MAX as i64
        && rows > 0
        && rows <= i32::MAX as i64
        && (k as usize).is_multiple_of(crate::ggml::block_size(ty))
        && w.nb()[0] == crate::ggml::row_size(ty, crate::ggml::block_size(ty))
        && w.nb()[1] >= crate::ggml::row_size(ty, k as usize)
        && (MIN_COLS..=MAX_COLS).contains(&n)
}

/// Bytes of `n` columns of `k` values as Q8_1 blocks.
fn q8_1_bytes(k: i64, n: i64) -> usize {
    (k as usize / QK8_1) * Q8_1_BYTES * n as usize
}

/// `w @ x` (F32 `[rows, n]`) for `w` and `x` that [`fits`] takes: `x` quantized as ggml's
/// `quantize_q8_1`, then the matvec.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(fits(w, x), "q_small_m: w {:?}, x {:?}", w.ne(), x.ne());
    let [k, n, _, _] = x.ne();
    let floats = q8_1_bytes(k, n).div_ceil(4) as i64;
    let xq = ctx.external(&QUANTIZE, types::F32, [floats, 1, 1, 1], &[x]);
    // The variant, fixed in the node at build time (a launch reads nothing that changes).
    let r = variant(w.ty(), n);
    ctx.external_with(&MV, types::F32, [w.ne()[1], n, 1, 1], &[w, xq], &[r])
}

/// `x` (`k` values a column, `n` columns) as ggml's Q8_1 blocks, column after column.
pub(crate) fn quantize_ref(xs: &[f32], k: usize, n: usize, out: &mut [u8]) {
    for j in 0..n {
        for b in 0..k / QK8_1 {
            let vals = &xs[j * k + b * QK8_1..j * k + (b + 1) * QK8_1];
            let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
            let sum: f32 = vals.iter().sum();
            let d = amax / 127.0;
            let blk = &mut out[(j * (k / QK8_1) + b) * Q8_1_BYTES..][..Q8_1_BYTES];
            // SAFETY: plain conversions.
            let (dh, sh) = unsafe { (ffi::ggml_fp32_to_fp16(d), ffi::ggml_fp32_to_fp16(sum)) };
            blk[0..2].copy_from_slice(&dh.to_le_bytes());
            blk[2..4].copy_from_slice(&sh.to_le_bytes());
            for (i, v) in vals.iter().enumerate() {
                let q = if amax == 0.0 {
                    0
                } else {
                    (v / d).round() as i8
                };
                blk[4 + i] = q as u8;
            }
        }
    }
}

unsafe fn cpu_quantize(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // SAFETY: a node made by `mul_mat`: a contiguous host `x`, room for its blocks behind `dst`.
    unsafe {
        if ith != 0 {
            return;
        }
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let xs = std::slice::from_raw_parts(x.data as *const f32, k * n);
        let out =
            std::slice::from_raw_parts_mut(dst.data as *mut u8, q8_1_bytes(k as i64, n as i64));
        quantize_ref(xs, k, n, out);
    }
}

unsafe fn cpu_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors: the weight, the blocks of `x`.
    unsafe {
        let (w, xq) = (&*dst.src[0], &*dst.src[1]);
        let (k, rows) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = dst.ne[1] as usize;
        let xb = std::slice::from_raw_parts(xq.data as *const u8, q8_1_bytes(k as i64, n as i64));
        let row_bytes = crate::ggml::row_size(w.type_, k);
        let out = dst.data as *mut f32;
        // x as the blocks hold it: q * f16(d).
        let xs: Vec<f32> = (0..n * k / QK8_1)
            .flat_map(|b| {
                let blk = &xb[b * Q8_1_BYTES..][..Q8_1_BYTES];
                let d = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]]));
                (0..QK8_1).map(move |i| blk[4 + i] as i8 as f32 * d)
            })
            .collect();
        for r in (ith..rows).step_by(nth) {
            let src = std::slice::from_raw_parts((w.data as *const u8).add(r * w.nb[1]), row_bytes);
            let wr = crate::ggml::dequantize(w.type_, src, k);
            for j in 0..n {
                let s: f32 = wr
                    .iter()
                    .zip(&xs[j * k..(j + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum();
                *out.add(j * rows + r) = s;
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// `legion_q_small_m` of `src/cuda/q_small_m.cu`.
    #[repr(C)]
    pub(super) struct Params {
        pub w: *const c_void,
        pub w_ld: i64,
        pub rows: i32,
        pub k: i32,
        pub x: *const f32,
        pub x_ld: i64,
        pub xq: *mut c_void,
        pub dst: *mut f32,
        pub dst_ld: i64,
        pub n: i32,
    }

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// Kernels of the table: the quantization, then the matvecs.
    const KERNELS: usize = 1 + TYPES.len() * 7 * VARIANTS.len();

    /// ggml's warps a CTA for `n` columns (`calc_nwarps`, the kernels' `ggml_warps`).
    fn ggml_warps(n: i64) -> u32 {
        if n <= 4 { 4 } else { 2 }
    }

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs): the quantization,
        /// then the matvec of each type of [`TYPES`], column count 2-8 and variant of
        /// [`VARIANTS`], at `1 + (type * 7 + n - 2) * 5 + variant_index`.
        static legion_q_small_m_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> i32;
    }

    fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut args = [(p as *mut Params).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `legion_q_small_m` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe {
            cudaLaunchKernel(
                legion_q_small_m_kernels[kernel],
                grid,
                block,
                args.as_mut_ptr(),
                0,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("q_small_m: CUDA error {e}")),
        }
    }

    /// The table index of the matvec for `ty`, `n` columns and variant `r`.
    pub(super) fn kernel_index(ty: ffi::ggml_type, n: i64, r: i32) -> Option<usize> {
        let t = type_index(ty)?;
        let ri = VARIANTS.iter().position(|&v| v == r)?;
        (MIN_COLS..=MAX_COLS)
            .contains(&n)
            .then(|| 1 + (t * 7 + (n - 2) as usize) * VARIANTS.len() + ri)
    }

    pub(super) unsafe fn quantize(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`: `x` contiguous in the stream's device memory, room
        // for its blocks behind `dst`.
        unsafe {
            let x = &*dst.src[0];
            let (k, n) = (x.ne[0], x.ne[1]);
            if ffi::ggml_nbytes(dst) < q8_1_bytes(k, n) {
                return Err("q_small_m: blocks short".into());
            }
            let mut p = Params {
                w: std::ptr::null(),
                w_ld: 0,
                rows: 0,
                k: k as i32,
                x: x.data as *const f32,
                x_ld: (x.nb[1] / 4) as i64,
                xq: dst.data,
                dst: std::ptr::null_mut(),
                dst_ld: 0,
                n: n as i32,
            };
            let grid = Dim3 {
                x: (k as u64).div_ceil(256) as u32,
                y: n as u32,
                z: 1,
            };
            let block = Dim3 { x: 256, y: 1, z: 1 };
            launch(0, grid, block, &mut p, stream)
        }
    }

    pub(super) unsafe fn mv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`, device tensors of the stream's device.
        unsafe {
            let (w, xq) = (&*dst.src[0], &*dst.src[1]);
            let (k, rows, n) = (w.ne[0], w.ne[1], dst.ne[1]);
            if ffi::ggml_nbytes(xq) < q8_1_bytes(k, n) {
                return Err("q_small_m: blocks short".into());
            }
            let r = crate::ggml::external_params(dst)[0];
            let Some(kernel) = kernel_index(w.type_, n, r) else {
                return Err(format!("q_small_m: no kernel for {} columns", n));
            };
            let mut p = Params {
                w: w.data as *const c_void,
                w_ld: w.nb[1] as i64,
                rows: rows as i32,
                k: k as i32,
                x: std::ptr::null(),
                x_ld: 0,
                xq: xq.data,
                dst: dst.data as *mut f32,
                dst_ld: rows,
                n: n as i32,
            };
            // `mv`: NW warps of r rows; `mv_cta`: ggml's G warps over -r rows.
            let (per_cta, block) = if r > 0 {
                (
                    NW * i64::from(r),
                    Dim3 {
                        x: NW as u32 * 32,
                        y: 1,
                        z: 1,
                    },
                )
            } else {
                (
                    -i64::from(r),
                    Dim3 {
                        x: 32,
                        y: ggml_warps(n),
                        z: 1,
                    },
                )
            };
            let grid = Dim3 {
                x: ((rows + per_cta - 1) / per_cta) as u32,
                y: 1,
                z: 1,
            };
            launch(kernel, grid, block, &mut p, stream)
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_quantize(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::quantize`.
    unsafe { cuda::quantize(dst, stream) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_mv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: as `cuda::mv`.
    unsafe { cuda::mv(dst, stream) }
}
