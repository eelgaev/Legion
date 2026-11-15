//! Paged attention prefill on Hopper's warpgroup MMA (`wgmma`) fed by the tensor memory
//! accelerator, FlashAttention-3 style (#1215; the kernel is `src/cuda/sm90/paged_attn_wgmma.cu`,
//! sm_90a only). **Untested on hardware**: written and compiled without a Hopper GPU, so it runs
//! only under `legion serve --hopper-attn` ([`set_enabled`], off by default), only on a device of
//! compute capability 9.0 whose binary holds the sm_90a bodies
//! ([`crate::model::wgmma_gemm::supported`], the same build), and only for the ops it covers
//! ([`covers`]): prefill groups (more than one token per sequence) over F16 caches with heads
//! of 128 or 256, no compact cache, no host tier, no key-block split. Every other op, and every
//! other GPU, keeps the default kernels: the dispatch is one branch before
//! `legion_paged_attn_launch`, with the same inputs.
//!
//! The kernel computes what the sm_80+ prefill kernel (`paged_prefill_mma`) does (the same
//! masks, F16 Q and P, F32 softmax and output), with other MMA shapes and sum orders: other
//! last bits, so it stays an opt-in until it is measured.
//!
//! The shared-memory plan and the wgmma matrix descriptors' fixed bits live here as plain
//! functions the CPU tests check; the kernel asserts the same sizes and takes the descriptor
//! bits from [`desc_k_hi`] and [`desc_v_hi`].

// Without CUDA only the CPU tests use the plan.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use super::KV_BLOCK;

/// Query rows per CTA (64 per consumer warpgroup): `ROWS / G` tokens of the G query heads of one
/// kv head.
pub const ROWS: usize = 128;
/// Threads per CTA: a producer warpgroup and two consumer warpgroups.
pub const THREADS: u32 = 384;
/// Bytes of a 128-byte swizzle span: 64 F16 values, the TMA box's inner size.
pub const SPAN: usize = 128;

/// The head sizes the kernel is instantiated for.
pub const HEAD_SIZES: [usize; 2] = [128, 256];

/// K stages of the ring, and as many V stages: four for heads of 128, two for 256 (the
/// shared memory a CTA can have, [`SMEM_LIMIT`]).
pub const fn stages(d: usize) -> usize {
    if d == 128 { 4 } else { 2 }
}

/// The kernel's dynamic shared memory for heads of `d`: Q (128 rows), the K and V stages (one
/// 64-cell block each), 1 KiB to align them on swizzle atoms, four mbarriers per stage
/// (`smem_bytes` in the kernel file, which asserts the same numbers).
pub const fn smem_bytes(d: usize) -> usize {
    let q = ROWS * d * 2;
    let kv = KV_BLOCK * d * 2;
    q + 2 * stages(d) * kv + 1024 + 4 * stages(d) * 8
}

/// The shared memory a block may opt into on compute capability 9.0.
pub const SMEM_LIMIT: usize = crate::model::wgmma_gemm::SMEM_LIMIT;

/// Tokens per CTA for `g` query heads per kv head (rows past `tokens * g` are padding when `g`
/// does not divide [`ROWS`]).
pub const fn tokens_per_cta(g: usize) -> usize {
    ROWS / g
}

/// The fixed bits of the K-major descriptors (Q and K: rows of 128 bytes, 64 dims, in the
/// 128-byte swizzle): the GEMM's ([`crate::model::wgmma_gemm::desc_hi`]).
pub const fn desc_k_hi() -> u64 {
    crate::model::wgmma_gemm::desc_hi()
}

/// The fixed bits of V's MN-major descriptor (PTX ISA, "Matrix Descriptor Format"; the B
/// operand of `O += P V`, read with the transpose bit): V's shared tile is one 128-byte row per
/// key holding 64 output dims, 8-key swizzle atoms of 1,024 bytes, one 8 KiB column of 64 keys
/// per 64 dims. Leading-dimension byte offset (bits 16-29, 16-byte units) = 8,192, from one
/// 64-dim column to the next (the MN direction); stride byte offset (bits 32-45) = 1,024, from
/// one 8-key atom to the next (the K direction); base offset 0; swizzle mode 1 = 128 bytes
/// (bits 62-63). The kernel ORs in the start address (+2,048 bytes per 16-key slice).
pub const fn desc_v_hi() -> u64 {
    let lbo = ((KV_BLOCK * SPAN) as u64) >> 4;
    let sbo = ((8 * SPAN) as u64) >> 4;
    (lbo << 16) | (sbo << 32) | (1u64 << 62)
}

/// Index of the kernel for heads of `d` in `legion_hopper_attn_kernels`.
pub const fn kernel_index(d: usize) -> Option<usize> {
    match d {
        128 => Some(0),
        256 => Some(1),
        _ => None,
    }
}

/// What [`covers`] looks at in an op.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub d: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_tok: usize,
    /// Both caches F16.
    pub f16: bool,
    /// A Gemma 4 compact cache.
    pub compact: bool,
    /// The pool has a host tier.
    pub tiered: bool,
    /// Key-block splits per tile (`--paged-verify-split`'s scratch): 1 without.
    pub max_split: usize,
    /// Bytes per cell of K and V.
    pub k_cell: usize,
    pub v_cell: usize,
    /// Cells of the K and V pools.
    pub k_cells: usize,
    pub v_cells: usize,
}

/// Whether the kernel takes an op of `s`: prefill (more than one token per sequence), F16
/// caches, heads of 128 or 256 over a GQA ratio that leaves a whole token per CTA, no compact
/// cache, no host tier, no split, rows the TMA accepts (multiples of 16 bytes) and cells it can
/// address (`i32`).
pub fn covers(s: &Shape) -> bool {
    HEAD_SIZES.contains(&s.d)
        && s.n_tok > 1
        && s.f16
        && !s.compact
        && !s.tiered
        && s.max_split <= 1
        && s.n_head_kv > 0
        && s.n_head % s.n_head_kv == 0
        && (1..=ROWS).contains(&(s.n_head / s.n_head_kv))
        && s.k_cell >= s.d * s.n_head_kv * 2
        && s.v_cell >= s.d * s.n_head_kv * 2
        && s.k_cell % 16 == 0
        && s.v_cell % 16 == 0
        && s.k_cells <= i32::MAX as usize
        && s.v_cells <= i32::MAX as usize
}

/// Whether graphs built from now on run paged prefill on the kernel ([`set_enabled`]).
static ON: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// `set_enabled_here`: the calling thread's own setting (tests), over [`ON`].
    static HERE: Cell<Option<bool>> = const { Cell::new(None) };
}

#[cfg(feature = "cuda")]
thread_local! {
    /// Launches of the kernel from this thread.
    static LAUNCHES: Cell<i64> = const { Cell::new(0) };
}

/// Whether paged attention prefill runs on the Hopper wgmma kernel where it covers the op, on a
/// device of compute capability 9.0 with an sm_90a binary (`legion serve --hopper-attn`). Off by
/// default: untested on hardware, and its sums run in another order than the default kernel's.
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// [`set_enabled`]'s state for the calling thread.
pub fn enabled() -> bool {
    HERE.with(|h| h.get())
        .unwrap_or_else(|| ON.load(Ordering::Relaxed))
}

/// [`set_enabled`] for the calling thread only (tests); `None` follows the process setting.
#[cfg(test)]
pub(crate) fn set_enabled_here(on: Option<bool>) {
    HERE.with(|h| h.set(on));
}

/// Launches of the kernel from the calling thread so far.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn launches() -> i64 {
    LAUNCHES.with(|n| n.get())
}

/// The kernel's argument block (`legion_hopper_attn` in the kernel file).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Args {
    pub q: *const f32,
    pub q_nb1: i64,
    pub q_nb2: i64,
    pub q_nb3: i64,
    pub table: *const i32,
    pub n_past: *const i32,
    pub dst: *mut f32,
    pub desc_k: u64,
    pub desc_v: u64,
    pub table_width: i32,
    pub n_head: i32,
    pub n_head_kv: i32,
    pub n_tok: i32,
    pub window: i32,
    pub ring: i32,
    pub scale: f32,
    pub pad: i32,
}

/// The op of `p` (the default kernels' parameters) on the kernel, on `stream`, when it is on
/// ([`enabled`]), covers the op ([`covers`]) and the device runs it. `Ok(false)`: not launched,
/// nothing written (the caller launches the default kernel).
///
/// # Safety
/// `p` and `shape` describe an op of `cuda_paged_attn` on the current device: its pointers are
/// device memory, the pools `shape.k_cells` / `shape.v_cells` cells of `k_cell` / `v_cell` bytes.
#[cfg(feature = "cuda")]
pub(super) unsafe fn run(
    p: &super::LaunchParams,
    shape: &Shape,
    stream: *mut std::os::raw::c_void,
) -> Result<bool, String> {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::AtomicI8;

    use crate::model::wgmma_gemm::launch::{
        Dim3, FUNC_ATTR_MAX_DYNAMIC_SHARED, TENSOR_MAP_F16, TensorMap, cudaFuncSetAttribute,
        cudaGetLastError, cudaLaunchKernel, current, map_2d,
    };

    /// `legion_hopper_attn_maps`: the K and V pools' tensor maps.
    #[repr(C, align(64))]
    struct Maps {
        k: TensorMap,
        v: TensorMap,
    }

    unsafe extern "C" {
        /// The kernels' host stubs: heads of 128, 256.
        static legion_hopper_attn_kernels: [*const c_void; 2];
    }

    const DEVICES: usize = 64;
    /// Per device and kernel: the shared-memory opt-in, 0 not yet, 1 done, -1 refused.
    static SMEM: [[AtomicI8; 2]; DEVICES] = [const { [const { AtomicI8::new(0) }; 2] }; DEVICES];

    if !enabled() || !covers(shape) || !crate::model::wgmma_gemm::supported() {
        return Ok(false);
    }
    let (Some(i), Some(dev)) = (kernel_index(shape.d), current()) else {
        return Ok(false);
    };
    let d = shape.d;
    if smem_bytes(d) > SMEM_LIMIT {
        return Ok(false);
    }
    let slot = &SMEM[dev][i];
    let mut s = slot.load(Ordering::Relaxed);
    if s == 0 {
        // SAFETY: a kernel of the table and its shared memory size.
        s = unsafe {
            if cudaFuncSetAttribute(
                legion_hopper_attn_kernels[i],
                FUNC_ATTR_MAX_DYNAMIC_SHARED,
                smem_bytes(d) as c_int,
            ) == 0
            {
                1
            } else {
                cudaGetLastError();
                -1
            }
        };
        slot.store(s, Ordering::Relaxed);
    }
    if s < 0 {
        return Ok(false);
    }
    // Pools of kv_row elements x cells, boxes of 64 dims x one 64-cell block.
    let kv_row = (d * shape.n_head_kv) as u64;
    let box_inner = (SPAN / 2) as u32;
    let (Some(mk), Some(mv)) = (
        map_2d(
            TENSOR_MAP_F16,
            p.k.cast(),
            kv_row,
            shape.k_cells as u64,
            shape.k_cell as u64,
            box_inner,
            KV_BLOCK as u32,
        ),
        map_2d(
            TENSOR_MAP_F16,
            p.v.cast(),
            kv_row,
            shape.v_cells as u64,
            shape.v_cell as u64,
            box_inner,
            KV_BLOCK as u32,
        ),
    ) else {
        return Ok(false);
    };
    let mut maps = Maps { k: mk, v: mv };
    let mut a = Args {
        q: p.q,
        q_nb1: p.q_nb1,
        q_nb2: p.q_nb2,
        q_nb3: p.q_nb3,
        table: p.table,
        n_past: p.n_past,
        dst: p.dst,
        desc_k: desc_k_hi(),
        desc_v: desc_v_hi(),
        table_width: p.table_width,
        n_head: p.n_head,
        n_head_kv: p.n_head_kv,
        n_tok: p.n_tok,
        window: p.window,
        ring: p.ring,
        scale: p.scale,
        pad: 0,
    };
    let g = shape.n_head / shape.n_head_kv;
    let qt = tokens_per_cta(g);
    let grid = Dim3 {
        x: shape.n_tok.div_ceil(qt) as u32,
        y: p.n_head_kv as u32,
        z: p.n_seqs as u32,
    };
    let block = Dim3 {
        x: THREADS,
        y: 1,
        z: 1,
    };
    let mut argv = [
        (&mut a as *mut Args).cast::<c_void>(),
        (&mut maps as *mut Maps).cast::<c_void>(),
    ];
    // SAFETY: the kernel's parameters (the argument block and the maps, copied at launch); the
    // pointers are the op's device memory.
    match unsafe {
        cudaLaunchKernel(
            legion_hopper_attn_kernels[i],
            grid,
            block,
            argv.as_mut_ptr(),
            smem_bytes(d),
            stream,
        )
    } {
        0 => {
            LAUNCHES.with(|n| n.set(n.get() + 1));
            Ok(true)
        }
        e => Err(format!("hopper paged attention: CUDA launch error {e}")),
    }
}

#[cfg(test)]
mod tests;
