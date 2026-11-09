//! Hopper prompt GEMMs on warpgroup MMA (`wgmma`) fed by the tensor memory accelerator (#1215;
//! CUDA kernels in `src/cuda/sm90/wgmma_gemm.cu`, sm_90a only). **Untested on hardware**: they
//! were written and compiled without a Hopper GPU, so they run only under
//! `legion serve --hopper-gemm` ([`set_enabled`], off by default) and only on a device of
//! compute capability 9.0 whose binary holds their sm_90a bodies ([`supported`]).
//!
//! `dst[t * m + r] = scale(r, t) * sum_k W[r][k] * X[t][k]` (+ a residual): W `m` rows and X `n`
//! tokens of `k` values, both K-major, BF16 or FP8 E4M3 ([`Kind`]); F32 out, rows fastest, with
//! `src/cuda/fp8_mm.cu`'s W8A8 scale rules. Today the FP8 kind takes the W8A8 prompt GEMM
//! (`crate::model::fp8`, more than 64 tokens) when the flag is on; the BF16 kind has no graph
//! caller yet (BF16 prompt matmuls run on cuBLAS, which has its own wgmma kernels on Hopper) and
//! is reached from the tests.
//!
//! The tile shapes, the shared-memory plan, the tile order and the wgmma matrix descriptor's
//! fixed bits live here, as plain functions the CPU tests check; the kernel takes the descriptor
//! bits from [`desc_hi`] and mirrors [`tile_of`].

use std::sync::atomic::{AtomicBool, Ordering};

/// The element type of both operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// BF16 x BF16, `wgmma .m64n128k16` (128 x 256 tiles).
    Bf16,
    /// FP8 E4M3 x E4M3, `wgmma .m64n128k32` (128 x 128 tiles; each 128-deep K block's sum
    /// promoted into F32 registers).
    E4m3,
}

/// Weight rows per CTA (64 per consumer warpgroup).
pub const BM: i64 = 128;
/// Bytes of one K step of one row: the 128-byte swizzle span.
pub const ROW_BYTES: i64 = 128;
/// Threads per CTA: a producer warpgroup and two consumer warpgroups.
pub const THREADS: u32 = 384;
/// Row tiles per group of [`tile_of`]'s order.
pub const GROUP_M: i32 = 8;
/// The shared memory a block may opt into on compute capability 9.0 (227 KiB), less the 1 KiB
/// the kernels reserve statically.
pub const SMEM_LIMIT: usize = 227 * 1024 - 1024;

impl Kind {
    /// Bytes of one value.
    pub const fn elem_bytes(self) -> i64 {
        match self {
            Kind::Bf16 => 2,
            Kind::E4m3 => 1,
        }
    }

    /// Tokens per CTA.
    pub const fn bn(self) -> i64 {
        match self {
            Kind::Bf16 => 256,
            Kind::E4m3 => 128,
        }
    }

    /// Values of K per stage (one 128-byte row of each operand).
    pub const fn bk(self) -> i64 {
        ROW_BYTES / self.elem_bytes()
    }

    /// Shared-memory stages of the ring.
    pub const fn stages(self) -> usize {
        match self {
            Kind::Bf16 => 4,
            Kind::E4m3 => 6,
        }
    }

    /// Bytes of one stage: the weight tile and the token tile.
    pub const fn stage_bytes(self) -> usize {
        ((BM + self.bn()) * ROW_BYTES) as usize
    }

    /// The kernel's dynamic shared memory: the stages, 1 KiB to align them on swizzle atoms, two
    /// mbarriers per stage (`smem_bytes` in the kernel file, which asserts the same numbers).
    pub const fn smem_bytes(self) -> usize {
        self.stages() * self.stage_bytes() + 1024 + 2 * self.stages() * 8
    }

    /// The `CUtensorMapDataType` of the operands.
    pub const fn tensor_map_type(self) -> u32 {
        match self {
            Kind::Bf16 => 9, // CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            Kind::E4m3 => 0, // CU_TENSOR_MAP_DATA_TYPE_UINT8
        }
    }

    /// Index of the kernel in `legion_wgmma_gemm_kernels`.
    pub const fn kernel_index(self, res: bool) -> usize {
        (match self {
            Kind::Bf16 => 0,
            Kind::E4m3 => 2,
        }) + res as usize
    }
}

/// The fixed bits of a wgmma shared-memory matrix descriptor for a K-major operand in the
/// 128-byte swizzle (PTX ISA, "Matrix Descriptor Format"): leading-dimension byte offset 16
/// (bits 16-29, in 16-byte units; unused for a swizzled K-major operand whose K slice lies
/// inside one swizzle span), stride byte offset 1,024 (bits 32-45: from one 8-row group to the
/// next, 8 rows x 128 bytes), base offset 0 (bits 49-51: stages start on 1,024-byte atoms),
/// swizzle mode 1 = 128 bytes (bits 62-63). The kernel ORs in the start address (bits 0-13).
pub const fn desc_hi() -> u64 {
    let lbo = 16u64 >> 4;
    let sbo = (8 * ROW_BYTES as u64) >> 4;
    (lbo << 16) | (sbo << 32) | (1u64 << 62)
}

/// The descriptor the kernel builds for shared address `addr` ([`desc_hi`] plus the start
/// address in 16-byte units).
pub const fn desc(addr: u32) -> u64 {
    desc_hi() | ((addr as u64 & 0x3_FFFF) >> 4)
}

/// Where the TMA's 128-byte swizzle puts byte `byte` (0-127) of row `row` of a tile, as a byte
/// offset from the tile's start (1,024-byte aligned): 16-byte chunk `c` of row `r` lands at chunk
/// `c ^ (r % 8)`. The layout wgmma's 128-byte swizzle mode reads.
pub const fn swizzle_128b(row: usize, byte: usize) -> usize {
    row * ROW_BYTES as usize + (((byte / 16) ^ (row % 8)) * 16) + byte % 16
}

/// Tile `pid` of a launch of `tiles_m` x `tiles_n` tiles as (row tile, token tile): groups of
/// [`GROUP_M`] row tiles, walked token tile by token tile, so that a wave's CTAs share weight
/// rows in L2. The kernel's `tile_of`.
pub fn tile_of(pid: i32, tiles_m: i32, tiles_n: i32) -> (i32, i32) {
    let in_group = GROUP_M * tiles_n;
    let first_m = pid / in_group * GROUP_M;
    let group = (tiles_m - first_m).min(GROUP_M);
    (first_m + pid % in_group % group, pid % in_group / group)
}

/// The launch's tiles `(tiles_m, tiles_n)` for `m` rows and `n` tokens.
pub fn tiles(kind: Kind, m: i64, n: i64) -> (i64, i64) {
    ((m + BM - 1) / BM, (n + kind.bn() - 1) / kind.bn())
}

/// Whether the kernels take a GEMM of `k` values, `m` rows and `n` tokens: whole K steps, row
/// strides the TMA accepts (multiples of 16 bytes), coordinates and the tile count in `i32`.
pub fn fits(kind: Kind, k: i64, m: i64, n: i64) -> bool {
    let (tm, tn) = tiles(kind, m, n);
    k > 0
        && m > 0
        && n > 0
        && k % kind.bk() == 0
        && (k * kind.elem_bytes()) % 16 == 0
        && k <= i32::MAX as i64
        && m <= i32::MAX as i64
        && n <= i32::MAX as i64
        && tm.checked_mul(tn).is_some_and(|t| t <= i32::MAX as i64)
}

/// Whether graphs built from now on run the W8A8 prompt GEMM on these kernels ([`set_enabled`]).
static ON: AtomicBool = AtomicBool::new(false);

/// Whether the W8A8 FP8 prompt GEMMs (more than 64 tokens) run on the Hopper wgmma kernels on a
/// device of compute capability 9.0 with an sm_90a binary (`legion serve --hopper-gemm`). Off by
/// default: untested on hardware, and the sums run in another order than the default GEMM's.
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// [`set_enabled`]'s state.
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The kernels' argument block (`legion_wgmma_gemm` in the kernel file).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Args {
    pub dst: *mut f32,
    pub res: *const f32,
    pub scale: *const f32,
    pub wscale: *const f32,
    pub k: i64,
    pub m: i64,
    pub n: i64,
    pub scale_stride: i64,
    pub wscale_stride: i64,
    pub desc_hi: u64,
    pub tiles_m: i32,
    pub tiles_n: i32,
}

/// Whether the current device runs the kernels: compute capability 9.0 and an sm_90a binary.
pub fn supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        launch::supported()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// `dst = scale * (w . x) (+ res)` on the current device and `stream`: `w` `m` rows and `x` `n`
/// tokens of `k` values of `kind` (device pointers, rows `k` values apart), `a`'s outputs and
/// scales. `Ok(false)`: not launched (the device, the binary or the shape does not allow it;
/// nothing was written).
///
/// # Safety
/// The pointers are device memory of the current device, large enough for the shapes.
#[cfg(feature = "cuda")]
pub unsafe fn run(
    kind: Kind,
    w: *const u8,
    x: *const u8,
    a: &mut Args,
    stream: *mut std::os::raw::c_void,
) -> Result<bool, String> {
    // SAFETY: the caller's contract.
    unsafe { launch::run(kind, w, x, a, stream) }
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI8, Ordering};

    use super::{Args, BM, Kind, ROW_BYTES, THREADS, desc_hi, fits, tiles};

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `CUtensorMap`: 128 opaque bytes, 64-byte aligned.
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    struct TensorMap([u64; 16]);

    /// `legion_wgmma_maps`: the weights' and the activations' tensor maps.
    #[repr(C, align(64))]
    struct Maps {
        w: TensorMap,
        x: TensorMap,
    }

    /// The head of `cudaFuncAttributes`.
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

    unsafe extern "C" {
        /// The kernels' host stubs: BF16, BF16 + residual, E4M3, E4M3 + residual.
        static legion_wgmma_gemm_kernels: [*const c_void; 4];
        /// Static shared memory only in its sm_90a body (see the kernel file).
        static legion_wgmma_gemm_probe: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cuTensorMapEncodeTiled(
            map: *mut TensorMap,
            dtype: u32,
            rank: u32,
            addr: *mut c_void,
            dims: *const u64,
            strides: *const u64,
            box_dim: *const u32,
            elem_strides: *const u32,
            interleave: u32,
            swizzle: u32,
            l2: u32,
            oob: u32,
        ) -> c_int;
    }

    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CC_MINOR: c_int = 76;
    const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;
    const SWIZZLE_128B: u32 = 3;
    const L2_PROMOTION_256B: u32 = 3;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// Per device: 0 not yet asked, 1 runs the kernels, -1 does not.
    static SUPPORTED: [AtomicI8; DEVICES] = [const { AtomicI8::new(0) }; DEVICES];
    /// Per device and kernel: the shared-memory opt-in, 0 not yet, 1 done, -1 refused.
    static SMEM: [[AtomicI8; 4]; DEVICES] = [const { [const { AtomicI8::new(0) }; 4] }; DEVICES];

    fn current() -> Option<usize> {
        let mut dev: c_int = 0;
        // SAFETY: a plain query.
        let ok = unsafe { cudaGetDevice(&mut dev) } == 0;
        (ok && (0..DEVICES as c_int).contains(&dev)).then_some(dev as usize)
    }

    /// Compute capability 9.0 exactly (wgmma and setmaxnreg are sm_90a instructions, which
    /// no later GPU runs) and the sm_90a bodies in the binary (the probe's static shared memory;
    /// a plain sm_90 build has trap stubs there). Asked once per device, no launch.
    pub(super) fn supported() -> bool {
        let Some(dev) = current() else {
            return false;
        };
        let slot = &SUPPORTED[dev];
        let mut s = slot.load(Ordering::Relaxed);
        if s == 0 {
            // SAFETY: plain queries of the current device and of a kernel of this binary.
            let ok = unsafe {
                let (mut major, mut minor) = (0, 0);
                let mut fa: FuncAttributes = std::mem::zeroed();
                let ok = cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev as c_int) == 0
                    && cudaDeviceGetAttribute(&mut minor, DEV_ATTR_CC_MINOR, dev as c_int) == 0
                    && (major, minor) == (9, 0)
                    && cudaFuncGetAttributes(&mut fa, legion_wgmma_gemm_probe) == 0
                    && fa.shared > 0;
                if !ok {
                    cudaGetLastError();
                }
                ok
            };
            s = if ok { 1 } else { -1 };
            slot.store(s, Ordering::Relaxed);
        }
        s > 0
    }

    fn opt_in(dev: usize, i: usize, bytes: usize) -> bool {
        let slot = &SMEM[dev][i];
        let mut s = slot.load(Ordering::Relaxed);
        if s == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            s = unsafe {
                if cudaFuncSetAttribute(
                    legion_wgmma_gemm_kernels[i],
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
            slot.store(s, Ordering::Relaxed);
        }
        s > 0
    }

    /// The tensor map of `rows` rows of `k` values of `kind` at `base`, in boxes of one 128-byte
    /// K step x `box_rows` rows, 128-byte swizzle, zeros past the edges; `None` where the driver
    /// refuses it.
    fn map(kind: Kind, base: *const u8, k: i64, rows: i64, box_rows: i64) -> Option<TensorMap> {
        if (base as usize) % 16 != 0 {
            return None;
        }
        let mut m = TensorMap([0; 16]);
        let dims = [k as u64, rows as u64];
        let strides = [(k * kind.elem_bytes()) as u64];
        let box_dim = [kind.bk() as u32, box_rows as u32];
        let elem = [1u32, 1];
        // SAFETY: arrays of the rank's lengths; the driver only encodes the address.
        let r = unsafe {
            cuTensorMapEncodeTiled(
                &mut m,
                kind.tensor_map_type(),
                2,
                base as *mut c_void,
                dims.as_ptr(),
                strides.as_ptr(),
                box_dim.as_ptr(),
                elem.as_ptr(),
                0,
                SWIZZLE_128B,
                L2_PROMOTION_256B,
                0,
            )
        };
        (r == 0).then_some(m)
    }

    pub(super) unsafe fn run(
        kind: Kind,
        w: *const u8,
        x: *const u8,
        a: &mut Args,
        stream: *mut c_void,
    ) -> Result<bool, String> {
        debug_assert_eq!(ROW_BYTES, kind.bk() * kind.elem_bytes());
        if !fits(kind, a.k, a.m, a.n) || !supported() {
            return Ok(false);
        }
        let Some(dev) = current() else {
            return Ok(false);
        };
        let i = kind.kernel_index(!a.res.is_null());
        if !opt_in(dev, i, kind.smem_bytes()) {
            return Ok(false);
        }
        let (Some(mw), Some(mx)) = (
            map(kind, w, a.k, a.m, BM),
            map(kind, x, a.k, a.n, kind.bn()),
        ) else {
            return Ok(false);
        };
        let mut maps = Maps { w: mw, x: mx };
        let (tm, tn) = tiles(kind, a.m, a.n);
        a.desc_hi = desc_hi();
        a.tiles_m = tm as i32;
        a.tiles_n = tn as i32;
        let grid = Dim3 {
            x: (tm * tn) as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: THREADS,
            y: 1,
            z: 1,
        };
        let mut argv = [
            (a as *mut Args).cast::<c_void>(),
            (&mut maps as *mut Maps).cast::<c_void>(),
        ];
        // SAFETY: the kernel's parameters (the argument block and the maps, copied at launch);
        // the pointers are the caller's device memory.
        match unsafe {
            cudaLaunchKernel(
                legion_wgmma_gemm_kernels[i],
                grid,
                block,
                argv.as_mut_ptr(),
                kind.smem_bytes(),
                stream,
            )
        } {
            0 => Ok(true),
            e => Err(format!("wgmma_gemm: CUDA launch error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
