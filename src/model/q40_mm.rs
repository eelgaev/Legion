//! Q4_0 prompt matmuls on a GB10 with ggml-cuda's MMQ bits: `w @ x` for one Q4_0 weight and
//! [`MIN_TOKENS`] or more F32 columns, as an external op ([`crate::ggml::ExternalOp`]).
//!
//! Gemma 4 31B QAT GGUFs hold every matmul weight as Q4_0. On a prompt step ggml-cuda runs
//! them as MMQ: a quantization of the input to q8_1 blocks, then 128-row tiles of 128 columns,
//! one CTA each, whose loop loads a 256-value slice of the weight's and the input's blocks into
//! shared memory, waits, multiplies it, waits, and loads the next (no copy overlaps the math),
//! converting each block's integer dot product to F32 (`I2F`, a quarter-rate instruction) before
//! scaling it. On a DGX Spark (GB10) its 4096-token FFN matmuls run at ~60 TFLOPS.
//!
//! The op (`src/cuda/q40_mm.cu`) quantizes the input as ggml does (the same values, the scale as
//! the F16 ggml stores) and multiplies 64 x 128 tiles ([`TILE_ROWS`] x [`TILE`]) from a two-stage
//! `cp.async` ring, two CTAs of 4 warps an SM, each block's dot product an `mma.sync` m16n8k32
//! s8 one, scaled without the conversion: the MMA's accumulator starts at the float 1.5 * 2^23,
//! so its result read as a float is that plus the integer exactly, and one FMA takes the offset
//! back out while scaling, rounding once to ggml's `fl(C * dA)`. The sums are ggml's:
//! `sum = fma(fl(C * dA), dB, sum)` over the K blocks in order from 0, each output's chain in one
//! thread, as in ggml's CTA per tile. Only plans on which ggml runs a CTA per tile
//! ([`ggml_tiles`]) are taken: ggml's stream-k split adds the pieces of a tile in another order.
//! Tiles run in groups of [`GROUP`] column tiles (row tiles slowest in a group, [`tile_of`]), so
//! a wave of CTAs reads a few input columns from DRAM once for all the weight's rows.
//!
//! Taken on compute capability 12.x (the GB10, whose ggml MMQ config the plan copies, see
//! [`crate::model::q8_mmq::plan`]) for every Q4_0 weight in one GPU's memory, 16-byte aligned,
//! with rows a multiple of 128 and K of 256, at [`MIN_TOKENS`] or more columns ([`fits`]); on by default,
//! `legion serve --q40-mm-ggml` keeps ggml's MMQ (the same output). Decode and verify steps
//! (fewer columns) stay on ggml's MMVQ / MMQ. Consecutive op nodes on the same input quantize it
//! once ([`quant_run`]). The CPU implementation is a reference for the tests: the same sums
//! (scalar `mul_add`), not necessarily the GPU's bits (the GPU's division and reciprocal of the
//! quantization are fast-math ones).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;

static Q40_MM: ExternalOp = ExternalOp {
    name: "q40_mm",
    cpu: cpu_q40_mm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::q40_mm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest columns the op takes: prompt steps (verify and decode steps keep ggml's kernels).
pub const MIN_TOKENS: i64 = 512;

/// Tile columns (and the first version's tile rows; the weight's rows are a multiple of it).
pub const TILE: i64 = 128;

/// The op's tile rows.
pub const TILE_ROWS: i64 = 64;

/// Column tiles of a raster group of the matmul ([`tile_of`]): a GB10's wave of 96 CTAs (two on
/// each of its 48 SMs) is 24 row tiles x 4 column tiles.
pub const GROUP: i64 = 4;

/// Row tiles of a raster group of the first version's matmul (one CTA an SM), which the timing
/// test runs beside the op's.
pub const GROUP_V1: i64 = 8;

/// Values per Q4_0 / q8_1 block.
const QK: i64 = 32;

/// Bytes of a Q4_0 block: the F16 scale, 16 bytes of nibbles.
const BLOCK_BYTES: usize = 18;

/// K values per quantization CTA; `k` is a multiple of it.
const K_STEP: i64 = 256;

/// Most bytes of quantized input a node takes: 4096 columns of K 21504 (Gemma 4 31B's FFN down)
/// take 99 MiB.
pub const MAX_QUANT_BYTES: usize = 256 << 20;

/// The matmul kernel's dynamic shared memory: two stages of 64 weight rows of 20 words, 4 x 128
/// columns of 8 quant words and 4 x 4 rows of 32 scales 36 words apart.
pub const SMEM_BYTES: usize = 2 * (64 * 20 + 4 * 128 * 8 + 4 * 4 * 36) * 4;

/// The first version's: three stages of 128 weight rows of 20 words, 4 x 128 columns of 8 quant
/// words and 4 x 128 scales.
pub const SMEM_BYTES_V1: usize = 3 * (128 * 20 + 4 * 128 * 8 + 4 * 128) * 4;

/// Whether the op runs where it applies ([`takes`]).
static ON: AtomicBool = AtomicBool::new(true);

/// The matmul of `src/cuda/q40_mm.cu`'s table the op runs: 0 (the op's) but in the timing and
/// bit tests, which run them all (the same bits).
#[cfg(feature = "cuda")]
static VARIANT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether the op runs where it applies ([`takes`]; `true`, the default); `false`
/// (`legion serve --q40-mm-ggml`) keeps ggml's MMQ for all.
pub fn set_q40_mm(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The SMs of `w`'s GPU where the op runs there (compute capability 12.x), else 0 (the
/// engine's GPU's, [`crate::model::q8_mmq::gpu_sms`], where the weight names no device).
fn sms_for(w: Tensor<'_>) -> i32 {
    crate::hw::per_device::sms_if_cc12_on(w.cuda_device(), crate::model::q8_mmq::gpu_sms())
}

/// Whether ggml-cuda runs the matmul of a Q4_0 weight of `rows` rows of `k` values times `n`
/// columns on `sms` SMs as one CTA per tile, each output's whole K chain in one thread (its
/// Q4_0 MMQ config is Q8_0's: the same tile widths and rows, stream-k), not split over CTAs.
pub fn ggml_tiles(rows: i64, k: i64, n: i64, sms: i64) -> bool {
    sms > 0 && !crate::model::q8_mmq::plan(rows, k, n, sms).split()
}

/// True when `w @ x` runs on the op: on, CUDA built, a compute capability 12.x GPU holding the
/// weight in its memory 16-byte aligned, shapes the op takes ([`fits`]) and ggml's plan a CTA
/// per tile ([`ggml_tiles`]).
pub fn takes(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    if !(ON.load(Ordering::Relaxed) && cfg!(feature = "cuda")) {
        return false;
    }
    let sms = sms_for(w) as i64;
    // SAFETY: a valid tensor's data pointer (null before allocation: the op checks it again).
    let aligned = (unsafe { (*w.raw()).data } as usize).is_multiple_of(16);
    sms > 0
        && !w.is_host()
        && !w.is_meta()
        && aligned
        && fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb()))
        && ggml_tiles(w.ne()[1], w.ne()[0], x.ne()[1], sms)
}

/// [`takes`] on shapes: a Q4_0 weight `[k, rows]` with contiguous rows 16-byte aligned (`k` a
/// multiple of 256, `rows` of 128); an F32 `x` `[k, n]` with contiguous values, `n` at least
/// [`MIN_TOKENS`]; at most [`MAX_QUANT_BYTES`] of quantized input.
pub fn fits(
    w: (ffi::ggml_type, [i64; 4], [usize; 4]),
    x: (ffi::ggml_type, [i64; 4], [usize; 4]),
) -> bool {
    let (wty, [k, rows, w2, w3], wnb) = w;
    let (xty, [xk, n, x2, x3], xnb) = x;
    wty == types::Q4_0
        && xty == types::F32
        && k > 0
        && k % K_STEP == 0
        && xk == k
        && rows > 0
        && rows % TILE == 0
        && w2 == 1
        && w3 == 1
        && x2 == 1
        && x3 == 1
        && wnb[0] == BLOCK_BYTES
        && wnb[1] % 16 == 0
        && wnb[1] >= (k / QK) as usize * BLOCK_BYTES
        && xnb[0] == 4
        && xnb[1] % 4 == 0
        && n >= MIN_TOKENS
        && n < 1 << 24
        && rows < 1 << 24
        && grid(rows, n).0 * grid(rows, n).1 < 1 << 31
        && quant_bytes(k, n) <= MAX_QUANT_BYTES
}

/// Column tiles and row tiles of the op's matmul of `rows` x `n`.
pub fn grid(rows: i64, n: i64) -> (i64, i64) {
    ((n + TILE - 1) / TILE, (rows + TILE_ROWS - 1) / TILE_ROWS)
}

/// Columns of the quantized input: `n` padded to whole tiles.
pub fn n_padded(n: i64) -> i64 {
    grid(0, n).0 * TILE
}

/// Bytes of the quantized input of `k` x `n`: per block of 32 values and padded column, 32
/// quants and an F32 scale.
pub fn quant_bytes(k: i64, n: i64) -> usize {
    (k / QK) as usize * n_padded(n) as usize * 36
}

/// The tile (row tile, column tile) CTA `b` of a grid of `ntx` x `nty` tiles computes, as the
/// kernels pick it: with `cols` (the op's) groups of `group` column tiles ([`GROUP`], the last
/// one shorter), each group's tiles column tile fastest; else (the first version's) groups of
/// `group` row tiles ([`GROUP_V1`]), row tile fastest.
pub fn tile_of(b: i64, ntx: i64, nty: i64, group: i64, cols: bool) -> (i64, i64) {
    let (major, minor) = if cols { (ntx, nty) } else { (nty, ntx) };
    let per_group = group * minor;
    let g = b / per_group;
    let r0 = g * group;
    let size = group.min(major - r0);
    let r = b - g * per_group;
    if cols {
        (r / size, r0 + r % size)
    } else {
        (r0 + r % size, r / size)
    }
}

/// `w @ x` (see [`takes`]) on the op: F32 `[rows, n]`, ggml's `mul_mat`'s shape, with its bits.
pub fn mul_mat<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(
        fits((w.ty(), w.ne(), w.nb()), (x.ty(), x.ne(), x.nb())),
        "q40_mm: shapes"
    );
    ctx.external(&Q40_MM, types::F32, [w.ne()[1], x.ne()[1], 1, 1], &[x, w])
}

/// Whether `node` is one of the op's.
///
/// # Safety
/// `node` is a valid tensor.
unsafe fn is_ours(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe { crate::ggml::op_of(node).is_some_and(|o| std::ptr::eq(o, &Q40_MM)) }
}

/// Whether the op's node `nodes[at]` quantizes its input: not when the last node of the op
/// before it in `nodes` (the graph's order, in which the nodes run on one stream) read the same
/// input, whose blocks are in the stream's scratch already (Gemma's FFN gate and up, the
/// attention's q, k and v). The same graph gives the same answer on every run.
///
/// # Safety
/// Every pointer of `nodes` is a valid tensor; `nodes[at]` is a node made by [`mul_mat`].
pub unsafe fn quant_run(nodes: &[*mut ffi::ggml_tensor], at: usize) -> bool {
    // SAFETY: the caller's.
    unsafe {
        let x = (*nodes[at]).src[0];
        !nodes[..at]
            .iter()
            .rev()
            .find(|&&n| is_ours(n))
            .is_some_and(|&prev| (*prev).src[0] == x)
    }
}

/// A column of `x` quantized as ggml-cuda's `quantize_mmq_q8_1` does for a Q4_0 weight (layout
/// DS4): per block of 32, the quants and the scale as the F16 ggml stores, in F32.
pub fn quantize_col(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / QK as usize);
    for v in x.chunks(QK as usize) {
        let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let d_inv = 127.0 / amax;
        // NaN (an all-zero block) converts to 0, as on the GPU.
        q.extend(v.iter().map(|x| (x * d_inv).round() as i8));
        // SAFETY: plain conversions.
        d.push(unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(1.0 / d_inv)) });
    }
    (q, d)
}

/// Row `r`'s value of column `(q, d)` ([`quantize_col`]) for Q4_0 weight row `row` of `blocks`
/// blocks: ggml's chain `sum = fma(fl(C * dA), dB, sum)`.
fn row_dot(row: &[u8], blocks: usize, q: &[i8], d: &[f32]) -> f32 {
    let mut s = 0f32;
    for b in 0..blocks {
        let wb = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
        // SAFETY: a plain conversion.
        let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([wb[0], wb[1]])) };
        let qx = &q[b * QK as usize..];
        let mut c = 0i32;
        for (i, &byte) in wb[2..].iter().enumerate() {
            c += ((byte & 0xF) as i32 - 8) * qx[i] as i32;
            c += ((byte >> 4) as i32 - 8) * qx[i + 16] as i32;
        }
        s = (c as f32 * dw).mul_add(d[b], s);
    }
    s
}

unsafe fn cpu_q40_mm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let (k, n, rows) = (x.ne[0] as usize, x.ne[1] as usize, w.ne[1] as usize);
        let blocks = k / QK as usize;
        let out = dst.data as *mut f32;
        for c in (ith..n).step_by(nth) {
            let col = (x.data as *const u8).add(c * x.nb[1]) as *const f32;
            let (q, d) = quantize_col(std::slice::from_raw_parts(col, k));
            for r in 0..rows {
                let row = std::slice::from_raw_parts(
                    (w.data as *const u8).add(r * w.nb[1]),
                    blocks * BLOCK_BYTES,
                );
                *out.add(c * rows + r) = row_dot(row, blocks, &q, &d);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI8, Ordering};

    use super::*;

    /// `legion_q40_mm` of `src/cuda/q40_mm.cu`.
    #[repr(C)]
    struct Params {
        w: *const c_void,
        x: *const f32,
        yq: *mut i32,
        yd: *mut f32,
        dst: *mut f32,
        x_ld: i64,
        w_nb1: i64,
        dst_ld: i64,
        k: i32,
        n: i32,
        n_pad: i32,
        rows: i32,
        ntx: i32,
        nty: i32,
    }

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `legion_q40_kernel` of `src/cuda/q40_mm.cu`: a matmul of the table.
    #[repr(C)]
    pub(super) struct Kernel {
        quant: *const c_void,
        gemm: *const c_void,
        smem: c_int,
        threads: c_int,
        bm: c_int,
        bn: c_int,
        /// The timing test's name for it.
        #[cfg_attr(not(test), allow(dead_code))]
        name: *const std::os::raw::c_char,
    }

    /// Most matmuls the shared-memory cache covers.
    const KERNELS: usize = 8;

    unsafe extern "C" {
        /// The first of `legion_q40_mm_table_len` entries.
        static legion_q40_mm_table: Kernel;
        static legion_q40_mm_table_len: c_int;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;
    const FUNC_ATTR_PREFERRED_SHARED_CARVEOUT: c_int = 9;

    /// Devices the shared-memory cache covers.
    const DEVICES: usize = 64;

    /// Each device's shared-memory opt-in of each matmul of the table: 0 not yet, 1 done, -1
    /// refused.
    static SMEM: [[AtomicI8; KERNELS]; DEVICES] =
        [const { [const { AtomicI8::new(0) }; KERNELS] }; DEVICES];

    /// Matmuls in the table.
    pub(super) fn kernels() -> usize {
        // SAFETY: a constant of src/cuda/q40_mm.cu.
        (unsafe { legion_q40_mm_table_len } as usize).min(KERNELS)
    }

    /// The table's matmul `i` (below [`kernels`]).
    pub(super) fn kernel(i: usize) -> &'static Kernel {
        assert!(i < kernels());
        // SAFETY: the table holds `legion_q40_mm_table_len` entries from its first.
        unsafe { &*std::ptr::addr_of!(legion_q40_mm_table).add(i) }
    }

    /// The name of the table's matmul `i` (the timing test's).
    #[cfg(test)]
    pub(super) fn kernel_name(i: usize) -> String {
        // SAFETY: a string literal of the table.
        unsafe { std::ffi::CStr::from_ptr(kernel(i).name) }
            .to_string_lossy()
            .into_owned()
    }

    /// The matmul the op runs: the table's first, or the tests' pick.
    fn variant() -> usize {
        VARIANT.load(Ordering::Relaxed)
    }

    /// One stream's quantized input buffer and its size.
    #[derive(Clone, Copy)]
    struct Scratch {
        ptr: usize,
        bytes: usize,
    }

    /// The scratch of each (device, stream), allocated on first use at the node's size rounded
    /// up to a power of two (32 MiB at least) and grown the same way when a node needs more. A
    /// buffer is never freed: graphs captured with its pointer may replay it. Ops on one stream
    /// run one after the other, so they share it as ggml's MMQ shares its pool.
    static SCRATCH: Mutex<Option<HashMap<(c_int, usize), Scratch>>> = Mutex::new(None);

    fn scratch(dev: c_int, stream: *mut c_void, need: usize) -> Result<Scratch, String> {
        let mut map = SCRATCH.lock().unwrap();
        let map = map.get_or_insert_with(HashMap::new);
        if let Some(s) = map.get(&(dev, stream as usize))
            && s.bytes >= need
        {
            return Ok(*s);
        }
        let bytes = need.next_power_of_two().max(32 << 20);
        let mut p = std::ptr::null_mut();
        // SAFETY: a plain allocation on the current device (a relaxed capture allows it).
        if unsafe { cudaMalloc(&mut p, bytes) } != 0 {
            // SAFETY: clears the error the failed call left.
            unsafe { cudaGetLastError() };
            return Err(format!("q40_mm: cudaMalloc of {bytes} bytes"));
        }
        let s = Scratch {
            ptr: p as usize,
            bytes,
        };
        map.insert((dev, stream as usize), s);
        Ok(s)
    }

    /// The shared memory of the table's matmul `i` on `dev`, and all of an SM's for shared
    /// memory (a hint: the GPU may keep another split).
    fn opt_in(dev: c_int, i: usize) -> Result<(), String> {
        let k = kernel(i);
        let slot = &SMEM[dev as usize][i];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a matmul of the table and its shared memory size.
            state = unsafe {
                if cudaFuncSetAttribute(k.gemm, FUNC_ATTR_MAX_DYNAMIC_SHARED, k.smem) == 0 {
                    if cudaFuncSetAttribute(k.gemm, FUNC_ATTR_PREFERRED_SHARED_CARVEOUT, 100) != 0 {
                        cudaGetLastError();
                    }
                    1
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(state, Ordering::Relaxed);
        }
        if state < 0 {
            return Err(format!("q40_mm: {} bytes of shared memory", k.smem));
        }
        Ok(())
    }

    fn launch(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        shared: usize,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut Params).cast::<c_void>()];
        // SAFETY: a kernel of the table taking `legion_q40_mm` by value, from the launch's copy
        // of `p`; its pointers are device memory of the stream's device.
        match unsafe { cudaLaunchKernel(func, grid, block, argv.as_mut_ptr(), shared, stream) } {
            0 => Ok(()),
            e => Err(format!("q40_mm: CUDA error {e}")),
        }
    }

    thread_local! {
        /// The graph and index of the last node of the op this thread launched: the next one
        /// is usually a little after it, so [`quant_in`] looks there first.
        static LAST: Cell<(usize, usize)> = const { Cell::new((0, usize::MAX)) };
    }

    /// [`quant_run`] for `dst` in graph `g`; a node outside a graph (`g` null or not holding
    /// it) quantizes.
    ///
    /// # Safety
    /// `g` is null or a valid graph; `dst` a node made by [`mul_mat`].
    unsafe fn quant_in(g: *mut ffi::ggml_cgraph, dst: &ffi::ggml_tensor) -> bool {
        // SAFETY: the caller's: the graph's node array holds `n` valid tensor pointers.
        unsafe {
            if g.is_null() {
                return true;
            }
            let n = ffi::ggml_graph_n_nodes(g) as usize;
            let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(g), n);
            let node = dst as *const ffi::ggml_tensor;
            let (lg, li) = LAST.with(Cell::get);
            let from = if lg == g as usize && li < n && is_ours(nodes[li]) {
                li
            } else {
                0
            };
            let at = (from..n)
                .find(|&i| std::ptr::eq(nodes[i], node))
                .or_else(|| (0..from).find(|&i| std::ptr::eq(nodes[i], node)));
            let Some(at) = at else { return true };
            LAST.with(|c| c.set((g as usize, at)));
            quant_run(nodes, at)
        }
    }

    pub(super) unsafe fn q40_mm(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat` (its shapes passed `fits`), device tensors of the
        // stream's device, the current one.
        unsafe {
            let (x, w) = (&*dst.src[0], &*dst.src[1]);
            let (k, n, rows) = (x.ne[0], x.ne[1], w.ne[1]);
            if !(w.data as usize).is_multiple_of(16) {
                return Err("q40_mm: weight not 16-byte aligned".into());
            }
            let mut dev: c_int = 0;
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                return Err("q40_mm: no current device".into());
            }
            let quant = quant_in(crate::ggml::current_cuda_graph(), dst);
            let v = variant();
            let kern = kernel(v);
            let (bm, bn) = (kern.bm as i64, kern.bn as i64);
            if rows % bm != 0 {
                return Err(format!("q40_mm: {rows} rows, tiles of {bm}"));
            }
            let (ntx, nty) = ((n + bn - 1) / bn, rows / bm);
            let n_pad = ntx * bn;
            let s = scratch(dev, stream, (k / QK) as usize * n_pad as usize * 36)?;
            let yd_off = (k / QK) as usize * n_pad as usize * 32;
            let mut a = Params {
                w: w.data as *const c_void,
                x: x.data as *const f32,
                yq: s.ptr as *mut i32,
                yd: (s.ptr + yd_off) as *mut f32,
                dst: dst.data as *mut f32,
                x_ld: (x.nb[1] / 4) as i64,
                w_nb1: w.nb[1] as i64,
                dst_ld: (dst.nb[1] / 4) as i64,
                k: k as i32,
                n: n as i32,
                n_pad: n_pad as i32,
                rows: rows as i32,
                ntx: ntx as i32,
                nty: nty as i32,
            };
            if quant {
                let quant_grid = Dim3 {
                    x: n as u32,
                    y: (k / K_STEP) as u32,
                    z: 1,
                };
                let quant_block = Dim3 { x: 64, y: 1, z: 1 };
                launch(kern.quant, quant_grid, quant_block, 0, &mut a, stream)?;
            }
            opt_in(dev, v)?;
            let grid = Dim3 {
                x: (ntx * nty) as u32,
                y: 1,
                z: 1,
            };
            let block = Dim3 {
                x: kern.threads as u32,
                y: 1,
                z: 1,
            };
            launch(kern.gemm, grid, block, kern.smem as usize, &mut a, stream)
        }
    }
}
