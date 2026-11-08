//! The expert matmul of BF16 weights, kept as shipped.
//!
//! GLM-5.3-Flash's NVFP4 release keeps its MTP layer in BF16, its 288 routed experts too (#560).
//! A stack of them is a BF16 tensor `[k, m, n_expert]`; [`mul_mat_id`] is ggml's `mul_mat_id`
//! for it with F32 activations and F32 sums.
//!
//! - CUDA (`src/cuda/bf16_experts.cu`): a grouped GEMV, one warp per output row, each lane
//!   reading 8 weights (16 bytes) at a time (an MTP head's MoE runs on its output rows alone,
//!   so it only ever sees a few rows).
//! - CPU: the reference (exact BF16 values, F32 sums in order), for tests and CPU runs.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static BF16_EXPERTS_MMID: ExternalOp = ExternalOp {
    name: "bf16_experts_mmid",
    cpu: cpu_mmid,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmid),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The weights one CUDA lane reads at a time; rows must be a multiple.
const CHUNK: i64 = 8;

/// Whether `w` is a BF16 expert stack that [`mul_mat_id`] takes: `[k, m, n_expert]` (more than
/// one expert) with contiguous rows of a multiple of [`CHUNK`] values.
pub fn takes(w: Tensor<'_>) -> bool {
    let [k, _, e, w3] = w.ne();
    w.ty() == types::BF16
        && e > 1
        && w3 == 1
        && k % CHUNK == 0
        && w.nb()[0] == 2
        && w.nb()[1] as i64 == 2 * k
}

/// `mul_mat_id(w, x, ids)` for a BF16 expert stack `w` (see [`takes`]): `x` F32 `[k, 1 or
/// n_used, n_tok]`, `ids` I32 `[n_used, n_tok]`; F32 `[m, n_used, n_tok]`. A row whose expert
/// id is out of range is 0.
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    assert!(
        takes(w),
        "bf16_experts: not a BF16 expert stack: {:?}",
        w.ne()
    );
    assert_eq!(x.ty(), types::F32, "bf16_experts: activations are F32");
    assert_eq!(ids.ty(), types::I32, "bf16_experts: expert ids are I32");
    let [k, m, _, _] = w.ne();
    let [xk, x_n1, n_tok, x3] = x.ne();
    let [n_used, id_tok, _, _] = ids.ne();
    assert!(
        xk == k && x3 == 1 && id_tok == n_tok && (x_n1 == 1 || x_n1 == n_used),
        "bf16_experts: x {:?} and ids {:?} for k = {k}",
        x.ne(),
        ids.ne()
    );
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.external(
        &BF16_EXPERTS_MMID,
        types::F32,
        [m, n_used, n_tok, 1],
        &[w, x, ids],
    )
}

/// The op's operands as both backends read them.
struct Shape {
    k: usize,
    m: usize,
    n_expert: usize,
    w_nb2: usize,
    x_n1: usize,
    n_used: usize,
    n_tok: usize,
    ids_nb: [usize; 2],
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`].
unsafe fn shape(dst: &ffi::ggml_tensor) -> Shape {
    // SAFETY: the node's sources are w, x and ids.
    let (w, x, ids) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2]) };
    Shape {
        k: w.ne[0] as usize,
        m: w.ne[1] as usize,
        n_expert: w.ne[2] as usize,
        w_nb2: w.nb[2],
        x_n1: x.ne[1] as usize,
        n_used: ids.ne[0] as usize,
        n_tok: ids.ne[1] as usize,
        ids_nb: [ids.nb[0], ids.nb[1]],
    }
}

unsafe fn cpu_mmid(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id`, host tensors; x contiguous, dst contiguous as ggml
    // allocates it.
    unsafe {
        let p = shape(dst);
        let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let xs = std::slice::from_raw_parts(x.data as *const f32, p.k * p.x_n1 * p.n_tok);
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
            let wr = std::slice::from_raw_parts(
                (w.data as *const u8).add(e as usize * p.w_nb2 + r * p.k * 2) as *const u16,
                p.k,
            );
            let xi = t * p.x_n1 + if p.x_n1 == 1 { 0 } else { u };
            let xr = &xs[xi * p.k..(xi + 1) * p.k];
            let mut acc = 0f32;
            for (&q, &v) in wr.iter().zip(xr) {
                acc += f32::from_bits((q as u32) << 16) * v;
            }
            *out.add(j) = acc;
        }
    }
}

/// `legion_bf16_experts_args` of `src/cuda/bf16_experts.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Args {
    w: *const u8,
    x: *const f32,
    ids: *const u8,
    dst: *mut f32,
    k: i64,
    m: i64,
    n_expert: i64,
    w_nb2: i64,
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
    static legion_bf16_experts_kernels: [*const std::os::raw::c_void; 1];
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
    // SAFETY: a node made by `mul_mat_id`, all tensors readable by the stream's device.
    unsafe {
        let p = shape(dst);
        let mut a = Args {
            w: (*dst.src[0]).data as *const u8,
            x: (*dst.src[1]).data as *const f32,
            ids: (*dst.src[2]).data as *const u8,
            dst: dst.data as *mut f32,
            k: p.k as i64,
            m: p.m as i64,
            n_expert: p.n_expert as i64,
            w_nb2: p.w_nb2 as i64,
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
        if groups > u16::MAX as usize || slots > i32::MAX as usize {
            return Err(format!(
                "bf16_experts_mmid: {} rows of {slots} slots are too many",
                p.m
            ));
        }
        let grid = Dim3 {
            x: slots as u32,
            y: groups as u32,
            z: 1,
        };
        let block = Dim3 { x: 128, y: 1, z: 1 };
        let mut argv = [(&raw mut a).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_bf16_experts_kernels[0],
            grid,
            block,
            argv.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("bf16_experts_mmid: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
