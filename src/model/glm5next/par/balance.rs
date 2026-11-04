//! `--glm-pair-balance` (#729): the routed experts of a decode step balanced within each GPU
//! pair instead of computed by their owners.
//!
//! A GPU's experts that miss its cache are read in place from host memory, about 12 times the
//! time of a cached one, and the step's exchanges wait at every MoE layer for the GPU with the
//! most work. Both GPUs of an NVLink pair read the same node's host memory ([`super`]'s
//! contiguous split puts a pair's experts on its own node), so either can compute a chosen
//! expert neither caches: [`ep_assign`] gives each such expert to the GPU of the owner's pair
//! with less work so far ([`crate::residency::ep_balance::assign_pair`]); a cached expert stays
//! with its owner. Its output replaces the router's choices as [`super::exchange::ep_mask`]'s
//! and [`super::exchange::ep_sum`]'s selection, so the matvec and the sum take this GPU's
//! assigned choices with no other change.
//!
//! Both GPUs of a pair compute the assignment from the same inputs (the router's choices and
//! the owners' cached experts, [`crate::residency::experts::TieredLayer::cached_view`]), so
//! they agree without an exchange. An expert computed by its pair's other GPU lands in that
//! GPU's partial sum: the routed sum adds in another order, and greedy output may differ at
//! near-ties, hence opt-in.
//!
//! `--glm-cross-balance` (#731, [`ep_assign_cross`]) does the same over every GPU: an uncached
//! expert may also go to the other pair, read across the sockets at about 2/3 of the speed
//! ([`crate::residency::ep_balance::assign_cross`]), when it still finishes there first. Each
//! GPU's tiered table names every expert it does not cache by its host copy, so any GPU can
//! compute any expert.

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};
#[cfg(feature = "cuda")]
use crate::residency::ep_balance::{CROSS_MISS_COST, HIT_COST, MISS_COST};
use crate::residency::ep_balance::{assign_cross, assign_pair, pair};

/// The parameters of an assignment node: the pair's first expert, the second GPU's first
/// expert, the pair's end, this GPU's index in the pair, the value its choices get.
const P_LO: usize = 0;
const P_MID: usize = 1;
const P_HI: usize = 2;
const P_SIDE: usize = 3;
const P_MINE: usize = 4;

static EP_ASSIGN: ExternalOp = ExternalOp {
    name: "glm_ep_assign",
    cpu: cpu_ep_assign,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ep_assign),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// GPU `me`'s choices under the pair balance: the router's choices `sel` (`[n_used, n]` I32,
/// contiguous) with `bounds[me]` (in its owned range) where this GPU computes the chosen
/// expert and -1 elsewhere, given `cached` (I32 `[n_expert]`, the owners' cached experts).
/// `None` for a GPU without a pair (the last of an odd count): its owned choices as before.
pub fn ep_assign<'a>(
    ctx: &'a Context,
    sel: Tensor<'a>,
    cached: Tensor<'a>,
    bounds: &[u32],
    me: usize,
) -> Option<Tensor<'a>> {
    let n_dev = bounds.len() - 1;
    let p = pair(me, n_dev);
    if p.len() != 2 {
        return None;
    }
    assert!(sel.ty() == types::I32 && cached.ty() == types::I32);
    let cont = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
    let params = [
        bounds[p.start] as i32,
        bounds[p.start + 1] as i32,
        bounds[p.end] as i32,
        (me - p.start) as i32,
        bounds[me] as i32,
    ];
    Some(ctx.external_with(
        &EP_ASSIGN,
        types::I32,
        sel.ne(),
        &[cont(sel), cont(cached)],
        &params,
    ))
}

/// The most GPUs [`ep_assign_cross`] takes: its node keeps the GPU count, this GPU and every
/// bound in [`crate::ggml::EXTERNAL_PARAMS`] (the kernel's `EP_MAX_DEV` is 8).
pub const CROSS_MAX_DEV: usize = 7;

static EP_ASSIGN_CROSS: ExternalOp = ExternalOp {
    name: "glm_ep_assign_cross",
    cpu: cpu_ep_assign_cross,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_ep_assign_cross),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// GPU `me`'s choices under the cross balance (`--glm-cross-balance`, #731): as [`ep_assign`]
/// with every GPU taking part. `None` past [`CROSS_MAX_DEV`] GPUs or for one GPU.
pub fn ep_assign_cross<'a>(
    ctx: &'a Context,
    sel: Tensor<'a>,
    cached: Tensor<'a>,
    bounds: &[u32],
    me: usize,
) -> Option<Tensor<'a>> {
    let n_dev = bounds.len() - 1;
    if !(2..=CROSS_MAX_DEV).contains(&n_dev) {
        return None;
    }
    assert!(sel.ty() == types::I32 && cached.ty() == types::I32);
    let cont = |t: Tensor<'a>| if t.is_contiguous() { t } else { ctx.cont(t) };
    let mut params = vec![n_dev as i32, me as i32];
    params.extend(bounds.iter().map(|&b| b as i32));
    Some(ctx.external_with(
        &EP_ASSIGN_CROSS,
        types::I32,
        sel.ne(),
        &[cont(sel), cont(cached)],
        &params,
    ))
}

/// [`ep_assign_cross`]'s node on the host, from its parameters `p` (the GPU count, this GPU,
/// the bounds): [`assign_cross`] over the step's distinct chosen experts.
pub fn assign_cross_host(sel: &[i32], cached: &[i32], p: &[i32]) -> Vec<i32> {
    let (n_dev, me) = (p[0] as usize, p[1] as usize);
    let bounds = &p[2..3 + n_dev];
    let owner = |e: i32| (0..n_dev).rfind(|&d| e >= bounds[d]).unwrap_or(0);
    let mut chosen: Vec<(u32, usize, bool)> = sel
        .iter()
        .filter(|&&e| e >= 0 && e < bounds[n_dev])
        .map(|&e| (e as u32, owner(e), cached[e as usize] != 0))
        .collect();
    chosen.sort_unstable_by_key(|c| c.0);
    chosen.dedup_by_key(|c| c.0);
    let who = assign_cross(&chosen, n_dev);
    sel.iter()
        .map(|&e| {
            let mine = chosen
                .binary_search_by_key(&(e as u32), |c| c.0)
                .is_ok_and(|i| e >= 0 && who[i] == me);
            if mine { bounds[me] } else { -1 }
        })
        .collect()
}

unsafe fn cpu_ep_assign_cross(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    // SAFETY: a node of `ep_assign_cross`: contiguous I32 host sources, `sel` of dst's size and
    // `cached` one value per expert.
    unsafe {
        let n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as usize;
        let c = &*dst.src[1];
        let n_expert = (c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3]) as usize;
        let sel = std::slice::from_raw_parts((*dst.src[0]).data as *const i32, n);
        let cached = std::slice::from_raw_parts(c.data as *const i32, n_expert);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut i32, n);
        out.copy_from_slice(&assign_cross_host(sel, cached, &p));
    }
}

/// The assignment of `sel` for the pair's GPU `side` ([`ep_assign`]'s parameters), on the host:
/// [`assign_pair`] over the pair's distinct chosen experts.
pub fn assign_host(sel: &[i32], cached: &[i32], p: &[i32]) -> Vec<i32> {
    let (lo, mid, hi) = (p[P_LO], p[P_MID], p[P_HI]);
    let mut chosen: Vec<(u32, usize, bool)> = sel
        .iter()
        .filter(|&&e| e >= lo && e < hi)
        .map(|&e| (e as u32, usize::from(e >= mid), cached[e as usize] != 0))
        .collect();
    chosen.sort_unstable_by_key(|c| c.0);
    chosen.dedup_by_key(|c| c.0);
    let who = assign_pair(&chosen, 2);
    sel.iter()
        .map(|&e| {
            let mine = chosen
                .binary_search_by_key(&(e as u32), |c| c.0)
                .is_ok_and(|i| (e >= lo && e < hi) && who[i] == p[P_SIDE] as usize);
            if mine { p[P_MINE] } else { -1 }
        })
        .collect()
}

unsafe fn cpu_ep_assign(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    // SAFETY: a node of `ep_assign`: contiguous I32 host sources, `sel` of dst's size and
    // `cached` one value per expert.
    unsafe {
        let n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as usize;
        let c = &*dst.src[1];
        let n_expert = (c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3]) as usize;
        let sel = std::slice::from_raw_parts((*dst.src[0]).data as *const i32, n);
        let cached = std::slice::from_raw_parts(c.data as *const i32, n_expert);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut i32, n);
        out.copy_from_slice(&assign_host(sel, cached, &p[..5]));
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

/// The kernel's `ep_bounds`: GPU `d` owns `b[d]..b[d + 1]`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct EpBounds {
    b: [i32; 9],
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    static legion_ep_assign_kernel: *const c_void;
    static legion_ep_assign_cross_kernel: *const c_void;
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
unsafe fn cuda_ep_assign(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let p = external_params(dst);
    // SAFETY: a node of `ep_assign` on the stream's device; the launch copies the arguments.
    unsafe {
        let mut sel = (*dst.src[0]).data as *const i32;
        let mut cached = (*dst.src[1]).data as *const i32;
        let mut out = dst.data as *mut i32;
        let mut n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as i32;
        let (mut lo, mut mid, mut hi) = (p[P_LO], p[P_MID], p[P_HI]);
        let (mut side, mut mine) = (p[P_SIDE], p[P_MINE]);
        let (mut hit, mut miss) = (HIT_COST as i32, MISS_COST as i32);
        let mut args = [
            (&raw mut sel).cast::<c_void>(),
            (&raw mut cached).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
            (&raw mut lo).cast::<c_void>(),
            (&raw mut mid).cast::<c_void>(),
            (&raw mut hi).cast::<c_void>(),
            (&raw mut side).cast::<c_void>(),
            (&raw mut mine).cast::<c_void>(),
            (&raw mut hit).cast::<c_void>(),
            (&raw mut miss).cast::<c_void>(),
        ];
        // Who computes each of the pair's experts, then the uncached chosen ones.
        let shared = 2 * (hi - lo).max(1) as usize * 4;
        match cudaLaunchKernel(
            legion_ep_assign_kernel,
            Dim3 { x: 1, y: 1, z: 1 },
            Dim3 { x: 256, y: 1, z: 1 },
            args.as_mut_ptr(),
            shared,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_ep_assign: CUDA error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_ep_assign_cross(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let p = external_params(dst);
    let n_dev = p[0] as usize;
    let mut bounds = EpBounds { b: [0; 9] };
    bounds.b[..=n_dev].copy_from_slice(&p[2..3 + n_dev]);
    // SAFETY: a node of `ep_assign_cross` on the stream's device; the launch copies the
    // arguments.
    unsafe {
        let mut sel = (*dst.src[0]).data as *const i32;
        let mut cached = (*dst.src[1]).data as *const i32;
        let mut out = dst.data as *mut i32;
        let mut n = (dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3]) as i32;
        let (mut n_dev_arg, mut me) = (p[0], p[1]);
        let (mut hit, mut miss, mut cross) =
            (HIT_COST as i32, MISS_COST as i32, CROSS_MISS_COST as i32);
        let mut args = [
            (&raw mut sel).cast::<c_void>(),
            (&raw mut cached).cast::<c_void>(),
            (&raw mut out).cast::<c_void>(),
            (&raw mut n).cast::<c_void>(),
            (&raw mut n_dev_arg).cast::<c_void>(),
            (&raw mut me).cast::<c_void>(),
            (&raw mut hit).cast::<c_void>(),
            (&raw mut miss).cast::<c_void>(),
            (&raw mut cross).cast::<c_void>(),
            (&raw mut bounds).cast::<c_void>(),
        ];
        // Who computes each expert of the layer, then the uncached chosen ones.
        let shared = 2 * bounds.b[n_dev].max(1) as usize * 4;
        match cudaLaunchKernel(
            legion_ep_assign_cross_kernel,
            Dim3 { x: 1, y: 1, z: 1 },
            Dim3 { x: 256, y: 1, z: 1 },
            args.as_mut_ptr(),
            shared,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("glm_ep_assign_cross: CUDA error {e}")),
        }
    }
}
