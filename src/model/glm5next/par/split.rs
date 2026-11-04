//! `--glm-pair-split` (#735): a decode step's uncached routed experts split over both GPUs of
//! their NVLink pair by rows, instead of computed whole by one GPU.
//!
//! A GPU reads a chosen expert its cache does not hold in place from host memory, about 12
//! times the time of a cached one, and the step's exchanges wait at every MoE layer for the GPU
//! with the most work. With one to three tokens a step a pair has only a few such misses per
//! layer, so even [`super::balance`]'s pair balance leaves one GPU a whole miss more than its
//! partner in most layers (#731). Both GPUs of a pair read the same node's host memory, so here
//! each computes half of every uncached expert of the pair's range: the SwiGLU op the rows of
//! its half of the intermediate (gate and up rows `0..n_ff / 2` on the pair's first GPU,
//! `n_ff / 2..n_ff` on the second), the down op every output row over the matching half of K,
//! a partial sum the step's exchange adds to the partner's ([`expert_mv`]). A cached expert
//! stays whole with its owner. [`ep_split`] gives the matvec its ids (the half ones marked with
//! [`expert_mv::HALF_BIT`]), and the same ids select the choices [`super::exchange::ep_sum`]
//! adds.
//!
//! Both GPUs of a pair compute the ids from the same inputs (the router's choices and the
//! owners' cached experts, [`crate::residency::experts::TieredLayer::cached_view`]), so they
//! agree without an exchange. An uncached expert's output becomes two partial sums added in
//! the exchange, in another order than one GPU's sum: greedy output may differ at near-ties,
//! and which experts are split follows the cache's contents, hence opt-in.

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};
use crate::model::core::expert_mv;
use crate::residency::ep_balance::pair;

/// The parameters of an ids node: the pair's first expert, the second GPU's first expert, the
/// pair's end, this GPU's index in the pair, the bit marking a half.
const P_LO: usize = 0;
const P_MID: usize = 1;
const P_HI: usize = 2;
const P_SIDE: usize = 3;
const P_HALF: usize = 4;

static EP_SPLIT: ExternalOp = ExternalOp {
    name: "glm_ep_split",
    cpu: cpu_ep_split,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ep_split),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// GPU `me`'s index in its pair (which half of a split expert it computes), `None` for a GPU
/// without a pair (the last of an odd count).
pub fn side(me: usize, n_dev: usize) -> Option<usize> {
    let p = pair(me, n_dev);
    (p.len() == 2).then(|| me - p.start)
}

/// GPU `me`'s expert matvec ids under the pair split: for each of the router's choices `sel`
/// (`[n_used, n]` I32, contiguous) with tiered id `mapped` (the same shape), `mapped` where this
/// GPU computes the whole expert (its own, cached), `mapped | HALF_BIT` where it computes half
/// (an uncached expert of its pair) and -1 elsewhere, given `cached` (I32 `[n_expert]`, the
/// owners' cached experts). `None` for a GPU without a pair.
pub fn ep_split<'a>(
    ctx: &'a Context,
    sel: Tensor<'a>,
    cached: Tensor<'a>,
    mapped: Tensor<'a>,
    bounds: &[u32],
    me: usize,
) -> Option<Tensor<'a>> {
    let n_dev = bounds.len() - 1;
    let s = side(me, n_dev)?;
    let p = pair(me, n_dev);
    assert!(sel.ty() == types::I32 && cached.ty() == types::I32 && mapped.ty() == types::I32);
    assert_eq!(
        sel.elements(),
        mapped.elements(),
        "ep_split: one id per choice"
    );
    let cont = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
    let params = [
        bounds[p.start] as i32,
        bounds[p.start + 1] as i32,
        bounds[p.end] as i32,
        s as i32,
        expert_mv::HALF_BIT,
    ];
    Some(ctx.external_with(
        &EP_SPLIT,
        types::I32,
        sel.ne(),
        &[cont(sel), cont(cached), cont(mapped)],
        &params,
    ))
}

/// [`ep_split`]'s ids on the host for its parameters `p`: the CUDA kernel's (`ep_split_kernel`
/// in `src/cuda/glm_exchange.cu`) value for value.
pub fn split_host(sel: &[i32], cached: &[i32], mapped: &[i32], p: &[i32]) -> Vec<i32> {
    let (lo, mid, hi) = (p[P_LO], p[P_MID], p[P_HI]);
    sel.iter()
        .zip(mapped)
        .map(|(&e, &m)| {
            if e < lo || e >= hi {
                -1
            } else if cached[e as usize] == 0 {
                m | p[P_HALF]
            } else if (e >= mid) == (p[P_SIDE] != 0) {
                m
            } else {
                -1
            }
        })
        .collect()
}

unsafe fn cpu_ep_split(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    // SAFETY: a node of `ep_split`: contiguous I32 host sources, `sel` and `mapped` of dst's
    // size and `cached` one value per expert.
    unsafe {
        let n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as usize;
        let c = &*dst.src[1];
        let n_expert = (c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3]) as usize;
        let sel = std::slice::from_raw_parts((*dst.src[0]).data as *const i32, n);
        let cached = std::slice::from_raw_parts(c.data as *const i32, n_expert);
        let mapped = std::slice::from_raw_parts((*dst.src[2]).data as *const i32, n);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut i32, n);
        out.copy_from_slice(&split_host(sel, cached, mapped, &p[..5]));
    }
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
    static legion_ep_split_kernel: *const c_void;
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ep_split(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let p = external_params(dst);
    // SAFETY: a node of `ep_split` on the stream's device; the launch copies the arguments.
    unsafe {
        let mut sel = (*dst.src[0]).data as *const i32;
        let mut cached = (*dst.src[1]).data as *const i32;
        let mut mapped = (*dst.src[2]).data as *const i32;
        let mut out = dst.data as *mut i32;
        let mut n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as i32;
        let (mut lo, mut mid, mut hi) = (p[P_LO], p[P_MID], p[P_HI]);
        let (mut side, mut half) = (p[P_SIDE], p[P_HALF]);
        let mut args = [
            (&raw mut sel).cast::<c_void>(),
            (&raw mut cached).cast::<c_void>(),
            (&raw mut mapped).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
            (&raw mut lo).cast::<c_void>(),
            (&raw mut mid).cast::<c_void>(),
            (&raw mut hi).cast::<c_void>(),
            (&raw mut side).cast::<c_void>(),
            (&raw mut half).cast::<c_void>(),
        ];
        match cudaLaunchKernel(
            legion_ep_split_kernel,
            Dim3 {
                x: (n as u32).div_ceil(256).max(1),
                y: 1,
                z: 1,
            },
            Dim3 { x: 256, y: 1, z: 1 },
            args.as_mut_ptr(),
            0,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_ep_split: CUDA error {e}")),
        }
    }
}

/// A step's routed experts of `[up, gate, down]` (tiered: `slots` VRAM cache slots) under the
/// pair split, for the matvec ids `ids` of [`ep_split`] and GPU side `side`: the per-choice
/// outputs (F32 `[n_embd, n_used, n]`, whole experts' or this GPU's half's partial sums),
/// SwiGLU clamped at `limit`. `None` when the kernels do not take the weights or the step
/// ([`expert_mv::split_fits`], steps of up to `max` tokens).
pub fn expert_mv<'a>(
    ctx: &'a Context,
    (slots, max): (usize, i64),
    [up, gate, down]: [Tensor<'a>; 3],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    limit: f32,
    side: usize,
) -> Option<Tensor<'a>> {
    let n = x.ne()[1];
    let x = ctx.reshape_3d(x, x.ne()[0], 1, n);
    if !expert_mv::split_fits([gate, up, down], x, ids, max) {
        return None;
    }
    let act = expert_mv::glu_split_clamped(ctx, gate, up, x, ids, slots, limit, side);
    Some(expert_mv::mul_mat_id_split(
        ctx, down, act, ids, slots, side,
    ))
}
