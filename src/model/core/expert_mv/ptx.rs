//! The decode expert matvec's VRAM experts through hand-written sm_70 PTX (#703, opt-in:
//! `legion serve --expert-mv-ptx`).
//!
//! `src/cuda/ptx/expert_mv_vram.ptx` (written by `gen_expert_mv_vram.py` beside it) holds the
//! kernels, loaded through [`crate::ptx`]: the SwiGLU op over Q4_K gate and up
//! (`legion_ptx_q4k_glu`) and the plain op over Q5_1 (`legion_ptx_q51_mv`), Qwen3.8-Flash-Next's
//! expert formats, each in a one-token form and a form for steps of several tokens (`_users`,
//! #710) that reads an expert's rows once for every pair that chose it, as the tile kernel does.
//! When the switch is on and an op has such experts in VRAM cache slots, the nvcc launcher runs
//! with `vram_kernel` 2 (its tile kernel leaves the VRAM pairs alone, and its own VRAM kernel
//! stays off) and the PTX kernel computes those pairs right after it on the same stream, reading
//! each expert's rows in place; with `--expert-mv-ptx-side` ([`set_side`]) it runs on the side
//! stream beside the tile kernel instead. Every (row, pair) sum takes the tile kernel's units,
//! lanes and additions in the same order (see the .ptx header), so the output bits do not
//! change. Other formats, host experts and staged copies keep the tile kernel.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{ffi, types};

/// The kernels' PTX text.
pub static MODULE: crate::ptx::Module = crate::ptx::Module::new(
    "expert_mv_vram",
    include_str!("../../../cuda/ptx/expert_mv_vram.ptx"),
);

/// The SwiGLU op over Q4_K gate and up.
pub const Q4K_GLU: &str = "legion_ptx_q4k_glu";
/// The plain op over Q5_1.
pub const Q51_MV: &str = "legion_ptx_q51_mv";
/// [`Q4K_GLU`] for steps of several tokens: each expert's rows once for all its pairs.
pub const Q4K_GLU_USERS: &str = "legion_ptx_q4k_glu_users";
/// [`Q51_MV`] for steps of several tokens.
pub const Q51_MV_USERS: &str = "legion_ptx_q51_mv_users";

/// Threads per CTA: four warps (the kernels' `.maxntid`).
pub const THREADS: u32 = 128;

/// Whether ops with experts the kernels take read their VRAM experts through them, for launches
/// from now on.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// See [`set_enabled`].
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether, with [`enabled`], the PTX kernel runs on the compute stream's side stream (#710),
/// beside the tile kernel's host pairs instead of after them, for launches from now on.
pub fn set_side(on: bool) {
    SIDE.store(on, Ordering::Relaxed);
}

/// See [`set_side`].
pub fn side() -> bool {
    SIDE.load(Ordering::Relaxed)
}

static SIDE: AtomicBool = AtomicBool::new(false);

/// The kernel for an op over `ty` experts (`glu`: gate and up through SwiGLU), if there is one.
pub fn kernel_for(ty: ffi::ggml_type, glu: bool) -> Option<&'static str> {
    match (ty, glu) {
        (types::Q4_K, true) => Some(Q4K_GLU),
        (types::Q5_1, false) => Some(Q51_MV),
        _ => None,
    }
}

/// [`kernel_for`] in a step of `n_tok` tokens: the `_users` form from 2 tokens on (a token picks
/// an expert at most once, so one token gives each expert one pair).
pub fn entry_for(ty: ffi::ggml_type, glu: bool, n_tok: usize) -> Option<&'static str> {
    let one = kernel_for(ty, glu)?;
    Some(match (one, n_tok > 1) {
        (Q4K_GLU, true) => Q4K_GLU_USERS,
        (Q51_MV, true) => Q51_MV_USERS,
        _ => one,
    })
}

/// vec_dot units per row of `k` values: Q4_K 16 per 256-value block (vdr 2 of qi 32), Q5_1 2
/// per 32-value block (vdr 2 of qi 4); expert_mv.cu's `units_per_row`.
pub fn units_per_row(ty: ffi::ggml_type, k: usize) -> usize {
    match ty {
        types::Q4_K => k / 256 * 16,
        types::Q5_1 => k / 32 * 2,
        _ => 0,
    }
}

/// Lanes sharing a row: 32, halved while a lane would get fewer than 4 units (expert_mv.cu's
/// `lanes_per_row`, which fixes the order the sums keep).
pub fn lanes_per_row(units: usize) -> usize {
    let mut l = 32;
    while l > 1 && units / l < 4 {
        l /= 2;
    }
    l
}

/// The kernels' one parameter (the offsets the .ptx reads; checked in the tests).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtxParams {
    pub w: *const u8,
    pub w2: *const u8,
    pub w_nb1: i64,
    pub w_nb2: i64,
    pub xq: *const u8,
    pub xq_col: i64,
    pub ids: *const u8,
    pub ids_nb1: i64,
    pub dst: *mut u8,
    pub dst_nb1: i64,
    pub dst_nb2: i64,
    pub rows: i32,
    pub n_used: i32,
    pub n_x: i32,
    pub vram_below: i32,
    pub n_w: i32,
    pub part: i32,
    pub units: i32,
    pub lshift: i32,
    pub limit: f32,
    pub n_tok: i32,
}

/// What a launch needs besides the parameter: the entry, and the grid (row blocks, pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub entry: &'static str,
    pub grid: [u32; 3],
}

/// The shape facts the kernels assume, checked on the host: rows of whole blocks, the header
/// loads aligned (Q4_K 16 bytes, Q5_1 8), the activation columns whole Q8_1 blocks.
pub fn shape_ok(ty: ffi::ggml_type, k: usize, w_nb1: usize, w_nb2: usize, w: usize) -> bool {
    let (qk, block, align) = match ty {
        types::Q4_K => (256, 144, 16),
        types::Q5_1 => (32, 24, 8),
        _ => return false,
    };
    k > 0
        && k.is_multiple_of(qk)
        && w_nb1 == k / qk * block
        && w_nb1.is_multiple_of(align)
        && w_nb2.is_multiple_of(align)
        && w.is_multiple_of(align)
}

/// The launch for `p` (over `ty`, `glu`), or `None` when the kernels do not take it: another
/// format, no VRAM experts, a part of a split without VRAM pairs (2, 4), or an empty grid.
pub fn plan(ty: ffi::ggml_type, glu: bool, p: &PtxParams, pairs: usize) -> Option<Plan> {
    let entry = entry_for(ty, glu, p.n_tok.max(1) as usize)?;
    if p.vram_below <= 0 || p.part == 2 || p.part == 4 || p.rows <= 0 || pairs == 0 {
        return None;
    }
    let rows_per_cta = (THREADS as usize / 32) * (32 >> p.lshift);
    let blocks = (p.rows as usize).div_ceil(rows_per_cta);
    Some(Plan {
        entry,
        grid: [blocks as u32, u32::try_from(pairs).ok()?, 1],
    })
}

/// The parameter for op `o` when the switch is on and the kernels take its experts and shape
/// (`None`: the nvcc kernels as without the switch).
#[cfg(feature = "cuda")]
pub(super) fn params_for(o: &super::Operands) -> Option<PtxParams> {
    let glu = !o.w2.is_null();
    kernel_for(o.ty, glu)?;
    let aligned = |w: *const u8| shape_ok(o.ty, o.k, o.w_nb[1], o.w_nb[2], w as usize);
    if !enabled() || o.vram_below <= 0 || !aligned(o.w) || (glu && !aligned(o.w2)) {
        return None;
    }
    let units = units_per_row(o.ty, o.k);
    Some(PtxParams {
        w: o.w,
        w2: o.w2,
        w_nb1: o.w_nb[1] as i64,
        w_nb2: o.w_nb[2] as i64,
        xq: o.xq,
        xq_col: super::q8_1_column_bytes(o.k) as i64,
        ids: o.ids,
        ids_nb1: o.ids_nb1 as i64,
        dst: o.dst,
        dst_nb1: o.dst_nb[1] as i64,
        dst_nb2: o.dst_nb[2] as i64,
        rows: i32::try_from(o.rows).ok()?,
        n_used: o.n_used as i32,
        n_x: o.n_x as i32,
        vram_below: o.vram_below,
        n_w: o.n_w as i32,
        part: 0,
        units: units as i32,
        lshift: lanes_per_row(units).trailing_zeros() as i32,
        limit: o.limit,
        n_tok: i32::try_from(o.n_tok).ok()?,
    })
}

/// The PTX kernel for the VRAM pairs of `part` of op `o` (no launch for the host parts).
///
/// # Safety
/// As [`launch`]; `p` from [`params_for`] of `o`.
#[cfg(feature = "cuda")]
pub(super) unsafe fn launch_part(
    o: &super::Operands,
    p: PtxParams,
    part: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), String> {
    let p = PtxParams { part, ..p };
    match plan(o.ty, !o.w2.is_null(), &p, o.n_used * o.n_tok) {
        // SAFETY: as the caller promises.
        Some(plan) => unsafe { launch(&plan, &p, stream) },
        None => Ok(()),
    }
}

/// Launches `p` on `stream` (the op's stream, current device), loading the module on the first
/// launch on a device.
///
/// # Safety
/// Every pointer in `p` must be valid for the kernel over `plan`'s grid: the experts below
/// `vram_below` of `w` (and `w2`) in device memory, the Q8_1 activations already written on
/// this stream, `dst` and `ids` of the op.
#[cfg(feature = "cuda")]
pub unsafe fn launch(
    plan: &Plan,
    p: &PtxParams,
    stream: *mut std::ffi::c_void,
) -> Result<(), String> {
    let f = crate::ptx::function(&MODULE, plan.entry)?;
    let mut copy = *p;
    let mut args = [&mut copy as *mut PtxParams as *mut std::ffi::c_void];
    // SAFETY: the kernel takes one parameter, a `PtxParams` (the .ptx's `P[128]`); the
    // addresses as the caller promises.
    unsafe { crate::ptx::launch(f, plan.grid, [THREADS, 1, 1], 0, stream, &mut args) }
}

pub mod bench;

#[cfg(test)]
mod tests;
