//! The CUDA side of the ops: the kernels of `src/cuda/deepseek41.cu`, launched from here with
//! `cudaLaunchKernel`, each taking its node described generically (data, shape, byte strides,
//! type) and its external params.

use std::os::raw::c_void;

use crate::ggml::{external_params, ffi};

const MAX_SRC: usize = 10;
const _: () = assert!(ffi::GGML_MAX_SRC as usize == MAX_SRC);

/// `legion_ds4_args` of `src/cuda/deepseek41.cu`.
#[repr(C)]
struct Args {
    dst: *mut c_void,
    ne: [i64; 4],
    nb: [i64; 4],
    src: [*const c_void; MAX_SRC],
    src_ne: [[i64; 4]; MAX_SRC],
    src_nb: [[i64; 4]; MAX_SRC],
    src_type: [i32; MAX_SRC],
    n_src: i32,
    params: [i32; 8],
}

/// `dim3`.
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

fn dim(x: i64, y: i64) -> Dim3 {
    Dim3 {
        x: x as u32,
        y: y as u32,
        z: 1,
    }
}

unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs' addresses), in
    /// [`Kernel`] order.
    static legion_ds4_kernels: [*const c_void; 15];
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// The kernels' order in `legion_ds4_kernels`.
#[derive(Clone, Copy)]
enum Kernel {
    Rope,
    Collapse,
    Engram,
    CacheWrite,
    Pool,
    IndexScores,
    CandMask,
    Topk,
    Attn,
    RouterIds,
    RouterW,
    /// `index_scores_keys_kernel<128>`.
    IndexScoresKeys,
    /// `attn_tiled_kernel<1, 16, 4, 32>` and `<8, 4, 4, 16>`.
    AttnTiled1,
    AttnTiled8,
    /// `attn_scored_kernel<2, 8, 8, 32>`.
    AttnScored,
}

/// Pairs the rope kernel's shared memory holds.
const MAX_PAIRS: i64 = 128;
/// The attention kernel's threads and values per thread (`ATTN_THREADS`, `ATTN_PER`).
const ATTN_D: i64 = 128 * 8;
/// The widest head the tiled attention kernels take (`PER` 4).
const ATTN_TILED_D: i64 = 128 * 4;
/// Heads per block of the wide tiled attention kernel (`HB`), and the fewest queries it takes:
/// fewer take the scored kernel (GB10, 32K context: 1 query 79 µs there, 182 in a block per
/// head; 4 queries 224 vs 530; at 8 the wide tiled kernel's 290 beats its 448).
const ATTN_HB: i64 = 8;
const ATTN_HB_MIN_QUERIES: i64 = 8;
/// The most keys (window and picks) a tiled attention kernel takes (`ATTN_KEYS`).
const ATTN_KEYS: i64 = 1024;
/// The scored attention kernel's head width (`SCORED_D`), heads per block (`HB`) and threads.
const SCORED_D: i64 = 512;
const SCORED_HB: i64 = 2;
const SCORED_THREADS: u32 = 256;
/// The index key width `index_scores_keys_kernel` holds in registers, its positions and queries
/// per block (`IDX_KEYS`, `IDX_TQ`).
const IDX_DI: i64 = 128;
const IDX_KEYS: i64 = 128;
const IDX_TQ: i64 = 32;

/// The node `dst` as the kernels read it.
unsafe fn args(dst: &ffi::ggml_tensor) -> Args {
    let mut a = Args {
        dst: dst.data,
        ne: dst.ne,
        nb: dst.nb.map(|b| b as i64),
        src: [std::ptr::null(); MAX_SRC],
        src_ne: [[0; 4]; MAX_SRC],
        src_nb: [[0; 4]; MAX_SRC],
        src_type: [0; MAX_SRC],
        n_src: 0,
        params: [0; 8],
    };
    for (i, &s) in dst.src.iter().enumerate() {
        if s.is_null() {
            break;
        }
        // SAFETY: the node's sources are valid tensors.
        let s = unsafe { &*s };
        a.src[i] = s.data;
        a.src_ne[i] = s.ne;
        a.src_nb[i] = s.nb.map(|b| b as i64);
        a.src_type[i] = s.type_ as i32;
        a.n_src = i as i32 + 1;
    }
    let p = external_params(dst);
    let n = p.len().min(8);
    a.params[..n].copy_from_slice(&p[..n]);
    a
}

/// Enqueues kernel `k` for the node `dst` on `stream`; nothing for an empty grid.
unsafe fn launch(
    k: Kernel,
    name: &str,
    dst: &ffi::ggml_tensor,
    grid: Dim3,
    block: u32,
    shared: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    if grid.x == 0 || grid.y == 0 {
        return Ok(());
    }
    // SAFETY: device tensors of the stream's device; the kernel reads `a` from the launch's
    // copy of its argument.
    unsafe {
        let mut a = args(dst);
        let mut argv = [(&raw mut a).cast::<c_void>()];
        match cudaLaunchKernel(
            legion_ds4_kernels[k as usize],
            grid,
            dim(block as i64, 1),
            argv.as_mut_ptr(),
            shared,
            stream,
        ) {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }
}

/// Source `i` of `dst`.
unsafe fn src(dst: &ffi::ggml_tensor, i: usize) -> &ffi::ggml_tensor {
    // SAFETY: the ops' builders give these sources.
    unsafe { &*dst.src[i] }
}

pub(super) unsafe fn rope(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    if external_params(dst)[0] as i64 / 2 > MAX_PAIRS {
        return Err("ds4_rope: more rotated pairs than the kernel holds".into());
    }
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Rope,
            "ds4_rope",
            dst,
            dim(dst.ne[2], 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn collapse(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Collapse,
            "ds4_collapse",
            dst,
            dim(dst.ne[1], 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn engram(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Engram,
            "ds4_engram",
            dst,
            dim(dst.ne[1], dst.ne[2]),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn cache_write(
    dst: &ffi::ggml_tensor,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        let n = src(dst, 1).ne[1];
        launch(
            Kernel::CacheWrite,
            "ds4_cache_write",
            dst,
            dim(n, 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn pool(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Pool,
            "ds4_pool",
            dst,
            dim(dst.ne[1], 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn index_scores(
    dst: &ffi::ggml_tensor,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        let iq = src(dst, 0);
        let shared = ((iq.ne[0] + 1) * iq.ne[1]) as usize * 4;
        if shared > 48 * 1024 {
            return Err("ds4_index_scores: indexer heads too wide for shared memory".into());
        }
        if iq.ne[0] == IDX_DI {
            // Queries of one sequence per block: tiles of each sequence's tokens.
            let n_tok = external_params(dst)[0] as i64;
            let tiles = (n_tok + IDX_TQ - 1) / IDX_TQ;
            return launch(
                Kernel::IndexScoresKeys,
                "ds4_index_scores",
                dst,
                dim(
                    (dst.ne[0] + IDX_KEYS - 1) / IDX_KEYS,
                    dst.ne[1] / n_tok * tiles,
                ),
                IDX_KEYS as u32,
                shared,
                stream,
            );
        }
        launch(
            Kernel::IndexScores,
            "ds4_index_scores",
            dst,
            dim((dst.ne[0] + 127) / 128, dst.ne[1]),
            128,
            shared,
            stream,
        )
    }
}

pub(super) unsafe fn cand_mask(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::CandMask,
            "ds4_cand_mask",
            dst,
            dim(dst.ne[1], 1),
            1024,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn topk(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Topk,
            "ds4_topk",
            dst,
            dim(dst.ne[1], 1),
            1024,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    if dst.ne[0] > ATTN_D {
        return Err("ds4_attn: head wider than the kernel holds".into());
    }
    let (d, nh, ng) = (dst.ne[0], dst.ne[1], dst.ne[2]);
    let p = external_params(dst);
    // SAFETY: the op's builder gives the selection as source 6.
    let picks = if p[6] != 0 {
        unsafe { src(dst, 6) }.ne[0]
    } else {
        0
    };
    if d <= ATTN_TILED_D && p[2] as i64 + picks <= ATTN_KEYS {
        // SAFETY: the op's builder gives the caches as sources 1 and 5.
        let (kown, kc) = unsafe { (src(dst, 1), src(dst, 5)) };
        // The scored kernel reads the keys' rows 4 values (8 bytes) at a time.
        let aligned =
            |t: &ffi::ggml_tensor| (t.data as usize).is_multiple_of(8) && t.nb[1].is_multiple_of(8);
        if d == SCORED_D
            && nh % SCORED_HB == 0
            && ng < ATTN_HB_MIN_QUERIES
            && aligned(kown)
            && aligned(kc)
            && p[4] % 4 == 0
        {
            // SAFETY: a node of this op, in device memory.
            return unsafe {
                launch(
                    Kernel::AttnScored,
                    "ds4_attn",
                    dst,
                    dim(nh / SCORED_HB, ng),
                    SCORED_THREADS,
                    0,
                    stream,
                )
            };
        }
        let wide = nh % ATTN_HB == 0 && ng >= ATTN_HB_MIN_QUERIES;
        let (k, hb) = if wide {
            (Kernel::AttnTiled8, ATTN_HB)
        } else {
            (Kernel::AttnTiled1, 1)
        };
        // SAFETY: a node of this op, in device memory.
        return unsafe { launch(k, "ds4_attn", dst, dim(nh / hb, ng), 128, 0, stream) };
    }
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Attn,
            "ds4_attn",
            dst,
            dim(dst.ne[1], dst.ne[2]),
            128,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn router_ids(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::RouterIds,
            "ds4_router_ids",
            dst,
            dim(dst.ne[1], 1),
            32,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn router_w(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::RouterW,
            "ds4_router_w",
            dst,
            dim((dst.ne[1] + 127) / 128, 1),
            128,
            0,
            stream,
        )
    }
}
