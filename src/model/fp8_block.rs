//! The expert matmul of block-scaled FP8 weights, kept as shipped.
//!
//! DeepSeek-style `FP8_BLOCK_SCALES` checkpoints (Qwen3.8-Flash-Next's MTP head experts) store
//! each weight as E4M3 bytes with one scale per `block_k × block_m` tile (128 × 128),
//! `weight_scale_inv`, which multiplies: `w[r][c] = e4m3(q[r][c]) · s[r / block_m][c / block_k]`.
//! Here a stack of experts is an **I8** tensor `[k, m, n_expert]` of the E4M3 bytes (as
//! [`crate::model::fp8`] keeps FP8 weights) plus an F32 scale grid `[k / block_k, m / block_m,
//! n_expert]` under the weight's `.scale` name; [`mul_mat_id`] is ggml's `mul_mat_id` for them.
//!
//! - CUDA (`src/cuda/fp8_block.cu`): a grouped GEMV, one warp per output row, each 16-byte
//!   weight load inside one scale block (decode and an MTP head's output rows; the head's MoE
//!   runs on those alone, so prompts never reach it with many rows).
//! - CPU: the reference (exact E4M3 values, F32 sums per scale block, each block's sum times its
//!   scale), for tests and CPU runs.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static FP8_BLOCK_MMID: ExternalOp = ExternalOp {
    name: "fp8_block_mmid",
    cpu: cpu_mmid,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmid),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The row bytes one CUDA lane reads at a time; a scale block's width must be a multiple.
const CHUNK: i64 = 16;

/// Whether `w` and its `scale` are a block-scaled FP8 expert stack that [`mul_mat_id`] takes:
/// I8 `[k, m, n_expert]` (more than one expert) and F32 `[k / block_k, m / block_m, n_expert]`
/// with `block_k` a multiple of 16.
pub fn takes(w: Tensor<'_>, scale: Option<Tensor<'_>>) -> bool {
    let Some(s) = scale else {
        return false;
    };
    let [k, m, e, w3] = w.ne();
    let [kb, mb, se, s3] = s.ne();
    w.ty() == types::I8
        && s.ty() == types::F32
        && s.is_contiguous()
        && e > 1
        && se == e
        && w3 == 1
        && s3 == 1
        && kb > 0
        && mb > 0
        && k % kb == 0
        && m % mb == 0
        && (k / kb) % CHUNK == 0
}

/// `mul_mat_id(w, x, ids)` for a block-scaled FP8 expert stack `w` with scales `scale` (see
/// [`takes`]): `x` F32 `[k, 1 or n_used, n_tok]`, `ids` I32 `[n_used, n_tok]`; F32 `[m, n_used,
/// n_tok]`. A row whose expert id is out of range is 0.
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Tensor<'a>,
) -> Tensor<'a> {
    assert!(
        takes(w, Some(scale)),
        "fp8_block: not a block-scaled FP8 expert stack: {:?} with scales {:?}",
        w.ne(),
        scale.ne()
    );
    assert_eq!(x.ty(), types::F32, "fp8_block: activations are F32");
    assert_eq!(ids.ty(), types::I32, "fp8_block: expert ids are I32");
    let [k, m, _, _] = w.ne();
    let [xk, x_n1, n_tok, x3] = x.ne();
    let [n_used, id_tok, _, _] = ids.ne();
    assert!(
        xk == k && x3 == 1 && id_tok == n_tok && (x_n1 == 1 || x_n1 == n_used),
        "fp8_block: x {:?} and ids {:?} for k = {k}",
        x.ne(),
        ids.ne()
    );
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.external(
        &FP8_BLOCK_MMID,
        types::F32,
        [m, n_used, n_tok, 1],
        &[w, x, ids, scale],
    )
}

/// The op's operands as both backends read them.
struct Shape {
    k: usize,
    m: usize,
    n_expert: usize,
    w_nb2: usize,
    bk: usize,
    bm: usize,
    x_n1: usize,
    n_used: usize,
    n_tok: usize,
    ids_nb: [usize; 2],
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`].
unsafe fn shape(dst: &ffi::ggml_tensor) -> Shape {
    // SAFETY: the node's sources are w, x, ids and the scales.
    let (w, x, ids, s) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]) };
    let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
    Shape {
        k,
        m,
        n_expert: w.ne[2] as usize,
        w_nb2: w.nb[2],
        bk: k / s.ne[0] as usize,
        bm: m / s.ne[1] as usize,
        x_n1: x.ne[1] as usize,
        n_used: ids.ne[0] as usize,
        n_tok: ids.ne[1] as usize,
        ids_nb: [ids.nb[0], ids.nb[1]],
    }
}

unsafe fn cpu_mmid(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id`, host tensors; x and the scales contiguous, dst
    // contiguous as ggml allocates it.
    unsafe {
        let p = shape(dst);
        let (w, x, ids, s) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let lut: [f32; 256] = std::array::from_fn(|b| super::fp8::e4m3_to_f32(b as u8));
        let (kb, mb) = (p.k / p.bk, p.m / p.bm);
        let xs = std::slice::from_raw_parts(x.data as *const f32, p.k * p.x_n1 * p.n_tok);
        let ss = std::slice::from_raw_parts(s.data as *const f32, kb * mb * p.n_expert);
        let out = dst.data as *mut f32;
        let rows = p.n_used * p.n_tok * p.m;
        for j in (ith..rows).step_by(nth) {
            let (r, slot) = (j % p.m, j / p.m);
            let (u, t) = (slot % p.n_used, slot / p.n_used);
            let id = (ids.data as *const u8).add(u * p.ids_nb[0] + t * p.ids_nb[1]);
            let e = *(id as *const i32);
            if e < 0 || e as usize >= p.n_expert {
                *out.add(j) = 0.0;
                continue;
            }
            let e = e as usize;
            let wr =
                std::slice::from_raw_parts((w.data as *const u8).add(e * p.w_nb2 + r * p.k), p.k);
            let xi = t * p.x_n1 + if p.x_n1 == 1 { 0 } else { u };
            let xr = &xs[xi * p.k..(xi + 1) * p.k];
            let sr = &ss[(e * mb + r / p.bm) * kb..][..kb];
            let mut acc = 0f32;
            for (b, &sc) in sr.iter().enumerate() {
                let c = b * p.bk..(b + 1) * p.bk;
                let part: f32 = wr[c.clone()]
                    .iter()
                    .zip(&xr[c])
                    .map(|(&q, &v)| lut[q as usize] * v)
                    .sum();
                acc += sc * part;
            }
            *out.add(j) = acc;
        }
    }
}

/// `legion_fp8_block_args` of `src/cuda/fp8_block.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Args {
    w: *const u8,
    s: *const f32,
    x: *const f32,
    ids: *const u8,
    dst: *mut f32,
    k: i64,
    m: i64,
    n_expert: i64,
    w_nb2: i64,
    bk: i64,
    bm: i64,
    x_n1: i64,
    n_used: i64,
    ids_nb0: i64,
    ids_nb1: i64,
}

/// `dim3`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs' addresses).
    static legion_fp8_block_kernels: [*const std::os::raw::c_void; 1];
    fn cudaLaunchKernel(
        func: *const std::os::raw::c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut std::os::raw::c_void,
        shared: usize,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_mmid(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    use std::os::raw::c_void;
    // SAFETY: a node made by `mul_mat_id`, all tensors in device memory of the stream's device.
    unsafe {
        let p = shape(dst);
        let mut a = Args {
            w: (*dst.src[0]).data as *const u8,
            s: (*dst.src[3]).data as *const f32,
            x: (*dst.src[1]).data as *const f32,
            ids: (*dst.src[2]).data as *const u8,
            dst: dst.data as *mut f32,
            k: p.k as i64,
            m: p.m as i64,
            n_expert: p.n_expert as i64,
            w_nb2: p.w_nb2 as i64,
            bk: p.bk as i64,
            bm: p.bm as i64,
            x_n1: p.x_n1 as i64,
            n_used: p.n_used as i64,
            ids_nb0: p.ids_nb[0] as i64,
            ids_nb1: p.ids_nb[1] as i64,
        };
        let slots = p.n_used * p.n_tok;
        let groups = p.m.div_ceil(4);
        if slots == 0 || groups == 0 {
            return Ok(());
        }
        if groups > u16::MAX as usize {
            return Err(format!("fp8_block_mmid: {} rows are too many", p.m));
        }
        let grid = Dim3 {
            x: slots as u32,
            y: groups as u32,
            z: 1,
        };
        let block = Dim3 { x: 128, y: 1, z: 1 };
        let mut argv = [(&raw mut a).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_fp8_block_kernels[0],
            grid,
            block,
            argv.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("fp8_block_mmid: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
