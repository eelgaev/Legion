//! Decode-sized selections on thread-block clusters (#554), compute capability 9.0+: the QSA
//! indexer's top-k ([`top_k`]) and the greedy pick over the vocabulary ([`argmax`]) with each
//! row split over a cluster of [`CLUSTER`] CTAs that combine their partial results through
//! distributed shared memory (kernels in `src/cuda/sm90/cluster_select.cu`).
//!
//! A decode step selects over a handful of rows. The one-CTA-per-row kernels (Legion's
//! `qsa_topk`, ggml-cuda's `argmax_f32`) then run on a few of the GPU's SMs, each reading a
//! whole row; on a cluster every row is read by [`CLUSTER`] SMs at once.
//!
//! Both ops are bitwise: [`top_k`] gives `qsa_topk`'s output index for index (the same k-th
//! largest key, every index above it ascending, then the lowest indexes at it), and [`argmax`]
//! gives ggml-cuda's pick, ties included: among equal maxima `argmax_f32` keeps the lowest
//! `(bitrev5(warp), bitrev5(lane), column / 1024)` of its 1,024 threads' scan
//! ([`argmax_row`]), not the lowest column, and the kernel reproduces that order. Larger
//! batches stay on the one-CTA-per-row kernels ([`MAX_TOPK_ROWS`], [`MAX_ARGMAX_ROWS`]), as do
//! short top-k rows ([`MIN_TOPK_COLS`], where one CTA a row is faster), GPUs older than
//! compute capability 9.0 (no clusters; top-k only) and the CPU.
//!
//! On GPUs without clusters (the V100s, #814) [`argmax`] splits each row over up to
//! [`MAX_ARGMAX_PARTS`] CTAs ([`argmax_parts`]) that write their best (value, tie rank) to a
//! buffer of their own, and a second launch joins them, one warp a row: the same pick, since
//! the tie order is a total order. The buffer is not a graph tensor (one 8 KiB buffer per device
//! and stream, made at the first launch): a tensor would move the graph's memory layout, and
//! with it which ggml-cuda fusions fire (#472), so the greedy text.
//!
//! On by default where it applies; `legion serve --no-cluster-select` keeps the one-CTA-per-row
//! kernels. Design after Basalt (github.com/jesdga95/basalt, MIT: `qsa_select.cu`'s
//! `block_topk_cluster_kernel`, `sampler.cu`'s `sampler_greedy_cluster_kernel`; some of its
//! kernels are adapted from Infernix, Apache-2.0); the top-k's radix passes and tie rule are
//! Strata-AC922's (`qsa_attn.cu`'s `topk_kernel`).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static TOPK: ExternalOp = ExternalOp {
    name: "cluster_topk",
    cpu: cpu_topk,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::topk),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ARGMAX: ExternalOp = ExternalOp {
    name: "cluster_argmax",
    cpu: cpu_argmax,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::argmax),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// CTAs per row: the portable cluster size.
pub const CLUSTER: u32 = 8;

/// Most rows [`top_k`] takes (rows of [`MIN_TOPK_COLS_8`] or more scores); more rows fill the
/// GPU with one CTA each.
pub const MAX_TOPK_ROWS: i64 = 8;

/// Fewest scores a row [`top_k`] takes: shorter rows (under 32K tokens of context at 4 a pool) are
/// faster on one CTA each (GB10: 8.3 against 10.6 us at 4,096, 14.0 against 12.3 at 8,192).
pub const MIN_TOPK_COLS: i64 = 8192;

/// Fewest scores a row [`top_k`] takes more than 4 rows of (GB10, 8 rows: 14.3 against 14.4 us
/// at 8,192, 34.9 against 30.8 at 25,000).
pub const MIN_TOPK_COLS_8: i64 = 16384;

/// Most rows [`argmax`] takes; more rows fill the GPU with one CTA each.
pub const MAX_ARGMAX_ROWS: i64 = 16;

/// Fewest columns [`argmax`] takes: `argmax_f32` runs 1,024 threads only from about this many
/// (its tie order is that of 1,024 threads).
pub const MIN_ARGMAX_COLS: i64 = 1024;

/// Most CTAs a row of the cluster-free [`argmax`].
pub const MAX_ARGMAX_PARTS: i64 = 64;

/// Columns a CTA of the cluster-free [`argmax`] takes at least (8 a thread).
pub const ARGMAX_PART_COLS: i64 = 4096;

/// CTAs a row of the cluster-free [`argmax`] for rows of `n` columns.
pub fn argmax_parts(n: i64) -> i64 {
    ((n.max(1) + ARGMAX_PART_COLS - 1) / ARGMAX_PART_COLS).min(MAX_ARGMAX_PARTS)
}

/// Whether graphs built from now on may take the ops ([`set_cluster_select`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether the engine computes on CUDA ([`set_cuda`]); the CPU keeps ggml's argmax (its own tie
/// rule).
static CUDA: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on run decode-sized top-k and argmax on clusters where they
/// apply (on by default); off keeps the one-CTA-per-row kernels (`legion serve
/// --no-cluster-select`, for comparisons).
pub fn set_cluster_select(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether the engine's graphs compute on CUDA (set at load).
pub fn set_cuda(on: bool) {
    CUDA.store(on, Ordering::Relaxed);
}

/// Whether the engine's graphs compute on CUDA ([`set_cuda`]).
pub(crate) fn on_cuda() -> bool {
    CUDA.load(Ordering::Relaxed)
}

/// Whether the ops are on, the engine on CUDA, and the current GPU runs them
/// ([`device_supported`]).
fn active() -> bool {
    cfg!(feature = "cuda")
        && ON.load(Ordering::Relaxed)
        && CUDA.load(Ordering::Relaxed)
        && device_supported()
}

/// Whether [`argmax`] may take its op: on, and the engine on CUDA (any GPU: without clusters
/// it runs the split form).
fn argmax_active() -> bool {
    cfg!(feature = "cuda") && ON.load(Ordering::Relaxed) && CUDA.load(Ordering::Relaxed)
}

/// Whether [`top_k`] takes `k` of each row of `score`: F32 `[n, rows]` (values contiguous), `0 < k
/// <= n`, [`MIN_TOPK_COLS`] or more scores a row, at most 4 rows ([`MAX_TOPK_ROWS`] from
/// [`MIN_TOPK_COLS_8`] scores), and the ops active.
pub fn takes_top_k(score: Tensor<'_>, k: i64) -> bool {
    topk_fits(score.ty(), score.ne(), score.nb(), k) && active()
}

/// The shape test of [`takes_top_k`].
pub(crate) fn topk_fits(ty: ffi::ggml_type, ne: [i64; 4], nb: [usize; 4], k: i64) -> bool {
    ty == types::F32
        && nb[0] == 4
        && ne[2] == 1
        && ne[3] == 1
        && k > 0
        && k <= ne[0]
        && (MIN_TOPK_COLS..=i32::MAX as i64).contains(&ne[0])
        && (1..=if ne[0] >= MIN_TOPK_COLS_8 {
            MAX_TOPK_ROWS
        } else {
            4
        })
            .contains(&ne[1])
}

/// The shape test of [`argmax`]: F32 `[n, rows]`, contiguous (as ggml's argmax requires),
/// [`MIN_ARGMAX_COLS`] or more columns, at most [`MAX_ARGMAX_ROWS`] rows.
pub(crate) fn argmax_fits(ty: ffi::ggml_type, ne: [i64; 4], nb: [usize; 4]) -> bool {
    ty == types::F32
        && nb[0] == 4
        && nb[1] as i64 == 4 * ne[0]
        && ne[2] == 1
        && ne[3] == 1
        && (MIN_ARGMAX_COLS..=i32::MAX as i64).contains(&ne[0])
        && (1..=MAX_ARGMAX_ROWS).contains(&ne[1])
}

/// The indexes of each row's `k` largest scores ([`takes_top_k`] must hold): I32 `[k, rows]`,
/// `qsa_attn::top_k`'s output exactly.
pub fn top_k<'a>(ctx: &'a Context, score: Tensor<'a>, k: i64) -> Tensor<'a> {
    let rows = score.ne()[1];
    ctx.external(&TOPK, types::I32, [k, rows, 1, 1], &[score])
}

/// The index of each row's largest value: I32 `[rows]`, `ctx.argmax(x)` on ggml-cuda bit for
/// bit, when the ops are on and `x` fits: on a cluster where the GPU has them
/// ([`device_supported`]), else split over CTAs and joined in a second launch; otherwise
/// ggml's argmax.
pub fn argmax<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    if argmax_fits(x.ty(), x.ne(), x.nb()) && argmax_active() {
        ctx.external(&ARGMAX, types::I32, [x.ne()[1], 1, 1, 1], &[x])
    } else {
        ctx.argmax(x)
    }
}

/// [`argmax`] of logits made with the head weight `w`: ggml's argmax when `w` is split over a
/// tensor-parallel meta device, whose backend hands every node to ggml-cuda (no external ops
/// there); the same bits either way.
pub fn argmax_of<'a>(ctx: &'a Context, x: Tensor<'a>, w: Tensor<'_>) -> Tensor<'a> {
    if w.is_meta() {
        ctx.argmax(x)
    } else {
        argmax(ctx, x)
    }
}

/// `argmax_f32`'s order among equal maxima for column `col` (lower wins): thread `t = col %
/// 1024` scans columns `t, t + 1024, ...` keeping the first strictly larger value, then two
/// xor-butterflies keep the larger value and otherwise their own, so the pick is the lowest
/// `(bitrev5(warp), bitrev5(lane), col / 1024)`.
pub(crate) fn argmax_rank(col: usize) -> u64 {
    let t = (col % 1024) as u32;
    let rev5 = |v: u32| v.reverse_bits() >> 27;
    ((rev5(t >> 5) as u64) << 37) | ((rev5(t & 31) as u64) << 32) | (col / 1024) as u64
}

/// One row of ggml-cuda's `argmax_f32` (rows of 1,024 columns or more): the pick among the
/// values above `-f32::MAX` (NaN never taken), ties by [`argmax_rank`]; -1 if there is none.
pub(crate) fn argmax_row(row: &[f32]) -> i32 {
    let mut best: Option<(f32, u64, usize)> = None;
    for (c, &v) in row.iter().enumerate() {
        if v > -f32::MAX {
            let r = argmax_rank(c);
            let better = match best {
                None => true,
                Some((bv, br, _)) => v > bv || (v == bv && r < br),
            };
            if better {
                best = Some((v, r, c));
            }
        }
    }
    best.map_or(-1, |(_, _, c)| c as i32)
}

unsafe fn cpu_topk(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `top_k`, host tensors.
    unsafe {
        let score = &*dst.src[0];
        let (n, k, rows) = (score.ne[0] as usize, dst.ne[0] as usize, dst.ne[1] as usize);
        for r in (ith..rows).step_by(nth) {
            let row = std::slice::from_raw_parts(
                (score.data as *const u8).add(r * score.nb[1]) as *const f32,
                n,
            );
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(r * dst.nb[1]) as *mut i32,
                k,
            );
            super::qwen4::qsa_attn::top_k_row(row, out);
        }
    }
}

unsafe fn cpu_argmax(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `argmax`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (n, rows) = (x.ne[0] as usize, x.ne[1] as usize);
        for r in (ith..rows).step_by(nth) {
            let row =
                std::slice::from_raw_parts((x.data as *const u8).add(r * x.nb[1]) as *const f32, n);
            *(dst.data as *mut i32).add(r) = argmax_row(row);
        }
    }
}

/// Whether the current GPU runs the ops: compute capability 9.0 or newer, cluster launch, and
/// code for the kernels in this build (checked once per device).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::supported()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// The current GPU's cluster limits for the kernels, for the record (#554): the largest
/// cluster the top-k kernel can launch, and how many clusters of [`CLUSTER`] CTAs fit at once.
#[cfg(feature = "cuda")]
pub fn cluster_limits() -> Option<(i32, i32)> {
    cuda::limits()
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The kernels' launches: `cudaLaunchKernelExC` with a cluster dimension, from Rust (no host
    //! code in the `.cu`).

    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicU8, Ordering};

    use super::{CLUSTER, ffi};

    /// `legion_topk_cluster` (`src/cuda/sm90/cluster_select.cu`).
    #[repr(C)]
    pub(super) struct TopK {
        pub score: *const f32,
        pub score_nb1: i64,
        pub dst: *mut i32,
        pub dst_nb1: i64,
        pub n: i32,
        pub k: i32,
        pub rows: i32,
    }

    /// `legion_argmax_cluster`.
    #[repr(C)]
    pub(super) struct Argmax {
        pub x: *const f32,
        pub x_nb1: i64,
        pub dst: *mut i32,
        pub n: i32,
        pub rows: i32,
    }

    /// `legion_argmax_split`.
    #[repr(C)]
    pub(super) struct ArgmaxSplit {
        pub x: *const f32,
        pub x_nb1: i64,
        pub dst: *mut i32,
        pub part: *mut c_void,
        pub n: i32,
        pub rows: i32,
        pub parts: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
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

    /// `cudaFuncAttributes`, its fields up to `binaryVersion`, padded past the struct's size.
    #[repr(C)]
    struct FuncAttributes {
        shared: usize,
        constant: usize,
        local: usize,
        max_threads: c_int,
        regs: c_int,
        ptx_version: c_int,
        binary_version: c_int,
        rest: [u8; 256],
    }

    const ATTR_CLUSTER_DIMENSION: c_int = 4;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CLUSTER_LAUNCH: c_int = 120;

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs). Top-k: PER = 4, 8,
        /// 16, 32, 64, then 0.
        static legion_topk_cluster_kernels: [*const c_void; 6];
        static legion_argmax_cluster_kernel: *const c_void;
        static legion_argmax_split_kernel: *const c_void;
        static legion_argmax_join_kernel: *const c_void;
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> c_int;
        fn cudaOccupancyMaxPotentialClusterSize(
            size: *mut c_int,
            func: *const c_void,
            config: *const LaunchConfig,
        ) -> c_int;
        fn cudaOccupancyMaxActiveClusters(
            count: *mut c_int,
            func: *const c_void,
            config: *const LaunchConfig,
        ) -> c_int;
    }

    /// Threads per CTA of both kernels.
    const THREADS: u32 = 512;

    /// Devices checked by [`supported`]: 0 unknown, 1 yes, 2 no.
    static CHECKED: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];

    pub(super) fn supported() -> bool {
        let mut dev = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..64).contains(&dev) {
                return false;
            }
            let slot = &CHECKED[dev as usize];
            match slot.load(Ordering::Relaxed) {
                1 => return true,
                2 => return false,
                _ => {}
            }
            let (mut major, mut cluster) = (0, 0);
            let mut ok = cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0
                && cudaDeviceGetAttribute(&mut cluster, DEV_ATTR_CLUSTER_LAUNCH, dev) == 0
                && major >= 9
                && cluster != 0;
            // Code for this device's architecture (an sm_90+ binary) for every kernel.
            let kernels = legion_topk_cluster_kernels
                .iter()
                .chain(std::iter::once(&legion_argmax_cluster_kernel));
            for &f in kernels {
                if !ok {
                    break;
                }
                let mut a: FuncAttributes = std::mem::zeroed();
                ok = cudaFuncGetAttributes(&mut a, f) == 0 && a.binary_version >= 90;
            }
            slot.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }

    /// The top-k kernel for `per` entries a thread.
    fn topk_kernel(per: usize) -> *const c_void {
        let i = match per {
            0..=4 => 0,
            5..=8 => 1,
            9..=16 => 2,
            17..=32 => 3,
            33..=64 => 4,
            _ => 5,
        };
        // SAFETY: a static of the CUDA object.
        unsafe { legion_topk_cluster_kernels[i] }
    }

    /// Entries per thread of the top-k kernel for rows of `n`.
    pub(super) fn topk_per(n: usize) -> usize {
        let slice = n.div_ceil(CLUSTER as usize);
        slice.div_ceil(THREADS as usize)
    }

    /// Launches `func` on `rows` clusters of [`CLUSTER`] CTAs of [`THREADS`] threads with `p`.
    unsafe fn launch<P>(
        name: &str,
        func: *const c_void,
        rows: u32,
        p: &mut P,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut attr = LaunchAttribute {
            id: ATTR_CLUSTER_DIMENSION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[..3].copy_from_slice(&[CLUSTER, 1, 1]);
        let config = LaunchConfig {
            grid: Dim3 {
                x: CLUSTER * rows,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            },
            dynamic_smem: 0,
            stream,
            attrs: &mut attr,
            num_attrs: 1,
        };
        let mut args = [(p as *mut P).cast::<c_void>()];
        // SAFETY: `func` takes one argument of type `P`; the launch copies it.
        match unsafe { cudaLaunchKernelExC(&config, func, args.as_mut_ptr()) } {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }

    /// The split argmax's buffers: (device, stream, buffer of [`PART_BYTES`]).
    static PARTS: std::sync::Mutex<Vec<(c_int, usize, usize)>> = std::sync::Mutex::new(Vec::new());

    /// Bytes of a split-argmax buffer: (value, rank) pairs for the most parts of the most rows.
    const PART_BYTES: usize = (super::MAX_ARGMAX_ROWS * super::MAX_ARGMAX_PARTS * 8) as usize;

    /// The split argmax's buffer for `stream` on the current device, made at its first use (a
    /// stage graph's capture runs in relaxed mode, where `cudaMalloc` is allowed). Launches on
    /// one stream run in order, so they share it; a captured graph replays on its stream.
    fn part_buffer(stream: *mut c_void) -> Result<*mut c_void, String> {
        let mut dev = 0;
        // SAFETY: a plain query of the current device.
        if unsafe { cudaGetDevice(&mut dev) } != 0 {
            return Err("cluster_argmax: cudaGetDevice failed".into());
        }
        let mut parts = PARTS.lock().unwrap();
        if let Some(&(_, _, p)) = parts
            .iter()
            .find(|&&(d, s, _)| d == dev && s == stream as usize)
        {
            return Ok(p as *mut c_void);
        }
        let mut p = std::ptr::null_mut();
        // SAFETY: a plain allocation on the current device, kept for the process.
        match unsafe { cudaMalloc(&mut p, PART_BYTES) } {
            0 => {
                parts.push((dev, stream as usize, p as usize));
                Ok(p)
            }
            e => Err(format!("cluster_argmax: cudaMalloc: CUDA error {e}")),
        }
    }

    /// Launches `func` on a `grid` of CTAs of `threads` threads with `p`, no cluster.
    unsafe fn launch_plain<P>(
        name: &str,
        func: *const c_void,
        grid: Dim3,
        threads: u32,
        p: &mut P,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let config = LaunchConfig {
            grid,
            block: Dim3 {
                x: threads,
                y: 1,
                z: 1,
            },
            dynamic_smem: 0,
            stream,
            attrs: std::ptr::null_mut(),
            num_attrs: 0,
        };
        let mut args = [(p as *mut P).cast::<c_void>()];
        // SAFETY: `func` takes one argument of type `P`; the launch copies it.
        match unsafe { cudaLaunchKernelExC(&config, func, args.as_mut_ptr()) } {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }

    pub(super) unsafe fn topk(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `top_k`, device tensors of the stream's device.
        unsafe {
            let score = &*dst.src[0];
            let (n, k, rows) = (score.ne[0], dst.ne[0], dst.ne[1]);
            if score.nb[0] != 4 || k <= 0 || k > n || n > i32::MAX as i64 || rows <= 0 {
                return Err("cluster_topk: unsupported shape".into());
            }
            let mut p = TopK {
                score: score.data as *const f32,
                score_nb1: score.nb[1] as i64,
                dst: dst.data as *mut i32,
                dst_nb1: dst.nb[1] as i64,
                n: n as i32,
                k: k as i32,
                rows: rows as i32,
            };
            let f = topk_kernel(topk_per(n as usize));
            launch("cluster_topk", f, rows as u32, &mut p, stream)
        }
    }

    pub(super) unsafe fn argmax(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `argmax`, device tensors of the stream's device.
        unsafe {
            let x = &*dst.src[0];
            let (n, rows) = (x.ne[0], x.ne[1]);
            if x.nb[0] != 4 || n < super::MIN_ARGMAX_COLS || n > i32::MAX as i64 || rows <= 0 {
                return Err("cluster_argmax: unsupported shape".into());
            }
            if !supported() {
                if rows > super::MAX_ARGMAX_ROWS {
                    return Err("cluster_argmax: too many rows".into());
                }
                let parts = super::argmax_parts(n);
                let mut p = ArgmaxSplit {
                    x: x.data as *const f32,
                    x_nb1: x.nb[1] as i64,
                    dst: dst.data as *mut i32,
                    part: part_buffer(stream)?,
                    n: n as i32,
                    rows: rows as i32,
                    parts: parts as i32,
                };
                let grid = Dim3 {
                    x: parts as u32,
                    y: rows as u32,
                    z: 1,
                };
                launch_plain(
                    "cluster_argmax",
                    legion_argmax_split_kernel,
                    grid,
                    THREADS,
                    &mut p,
                    stream,
                )?;
                let grid = Dim3 {
                    x: rows as u32,
                    y: 1,
                    z: 1,
                };
                return launch_plain(
                    "cluster_argmax",
                    legion_argmax_join_kernel,
                    grid,
                    32,
                    &mut p,
                    stream,
                );
            }
            let mut p = Argmax {
                x: x.data as *const f32,
                x_nb1: x.nb[1] as i64,
                dst: dst.data as *mut i32,
                n: n as i32,
                rows: rows as i32,
            };
            launch(
                "cluster_argmax",
                legion_argmax_cluster_kernel,
                rows as u32,
                &mut p,
                stream,
            )
        }
    }

    pub(super) fn limits() -> Option<(i32, i32)> {
        if !supported() {
            return None;
        }
        let mut attr = LaunchAttribute {
            id: ATTR_CLUSTER_DIMENSION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[..3].copy_from_slice(&[CLUSTER, 1, 1]);
        let mut config = LaunchConfig {
            grid: Dim3 {
                x: CLUSTER * 64,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            },
            dynamic_smem: 0,
            stream: std::ptr::null_mut(),
            attrs: &mut attr,
            num_attrs: 1,
        };
        let f = topk_kernel(1);
        let (mut size, mut active) = (0, 0);
        // SAFETY: occupancy queries of a kernel of this build.
        unsafe {
            config.num_attrs = 0;
            if cudaOccupancyMaxPotentialClusterSize(&mut size, f, &config) != 0 {
                return None;
            }
            config.num_attrs = 1;
            if cudaOccupancyMaxActiveClusters(&mut active, f, &config) != 0 {
                return None;
            }
        }
        Some((size, active))
    }
}

#[cfg(test)]
mod tests;
