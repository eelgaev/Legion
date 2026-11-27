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

/// The quantization of a matmul's input as its own node ([`quantize`], [`quantize_norm`],
/// [`quantize_geglu`]): the matmul's layout of the blocks, the bytes the op's own quantization
/// writes, read by [`mul_mat_q`] nodes.
static Q40_QUANT: ExternalOp = ExternalOp {
    name: "q40_quant",
    cpu: cpu_q40_quant,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::q40_quant),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// What a [`Q40_QUANT`] node quantizes (its first parameter): its F32 input; ggml's
/// `mul(rms_norm(x, eps), w)` of its inputs `x`, `w`; ggml's `geglu_split(gate, up)`.
const QUANT_PLAIN: i32 = 0;
const QUANT_NORM: i32 = 1;
const QUANT_GEGLU: i32 = 2;

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

/// Most bytes between weight rows: the kernel offsets a thread's copies within a tile (up to 128
/// rows) in 32-bit ints, and 127 rows of 16 MiB plus a block's bytes stay below `i32::MAX`.
pub const MAX_ROW_BYTES: usize = 16 << 20;

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
    takes_shape(w, (x.ty(), x.ne(), x.nb()))
}

/// [`takes`] for an `x` of type, shape and strides `x` (one not built yet).
pub fn takes_shape(w: Tensor<'_>, x: (ffi::ggml_type, [i64; 4], [usize; 4])) -> bool {
    if !(ON.load(Ordering::Relaxed) && cfg!(feature = "cuda")) {
        return false;
    }
    let sms = sms_for(w) as i64;
    // SAFETY: a valid tensor's data pointer. Null before allocation: the address it gets then
    // is unknown, so such a weight stays on the generic path.
    let data = unsafe { (*w.raw()).data } as usize;
    let aligned = data != 0 && data.is_multiple_of(16);
    sms > 0
        && !w.is_host()
        && !w.is_meta()
        && aligned
        && fits((w.ty(), w.ne(), w.nb()), x)
        && ggml_tiles(w.ne()[1], w.ne()[0], x.1[1], sms)
}

/// The type, shape and strides of a contiguous F32 tensor of `ne` (`[k, n]`).
pub fn f32_shape(ne: [i64; 2]) -> (ffi::ggml_type, [i64; 4], [usize; 4]) {
    let row = ne[0] as usize * 4;
    let nb2 = row * ne[1] as usize;
    (types::F32, [ne[0], ne[1], 1, 1], [4, row, nb2, nb2])
}

/// Whether [`quantize_norm`] takes `mul(rms_norm(x, eps), w)`: an F32 `x` of contiguous
/// columns 16-byte apart (`[k, n]`, `k` a multiple of 256) and a contiguous F32 `w` of `k`,
/// both in GPU memory.
pub fn norm_fits(x: Tensor<'_>, w: Tensor<'_>) -> bool {
    let [k, n, x2, x3] = x.ne();
    let nb = x.nb();
    x.ty() == types::F32
        && w.ty() == types::F32
        && k > 0
        && k % K_STEP == 0
        && n > 0
        && x2 == 1
        && x3 == 1
        && nb[0] == 4
        && nb[1] % 16 == 0
        && w.ne() == [k, 1, 1, 1]
        && w.nb()[0] == 4
        && !x.is_host()
        && !w.is_host()
        && !x.is_meta()
        && !w.is_meta()
}

/// Whether [`quantize_geglu`] takes `geglu_split(gate, up)`: F32 `[k, n]` of the same shape,
/// contiguous columns 16 bytes apart, `k` a multiple of 256.
pub fn geglu_fits(gate: Tensor<'_>, up: Tensor<'_>) -> bool {
    let [k, n, g2, g3] = gate.ne();
    let ok = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4 && t.nb()[1] % 16 == 0;
    ok(gate)
        && ok(up)
        && !gate.is_meta()
        && !up.is_meta()
        && up.ne() == gate.ne()
        && k > 0
        && k % K_STEP == 0
        && n > 0
        && g2 == 1
        && g3 == 1
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
        && wnb[1] <= MAX_ROW_BYTES
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

/// [`mul_mat`] of `w` over `n` columns already quantized in `xq` (a [`quantize`],
/// [`quantize_norm`] or [`quantize_geglu`] node of `w`'s `k`, or [`quant_bytes`] I8 bytes in
/// that layout): the same output, without the op's own quantization. Where [`takes`] takes
/// `w` for the F32 `[k, n]` columns.
pub fn mul_mat_q<'a>(ctx: &'a Context, w: Tensor<'a>, xq: Tensor<'a>, n: i64) -> Tensor<'a> {
    let k = w.ne()[0];
    assert!(
        fits((w.ty(), w.ne(), w.nb()), f32_shape([k, n]))
            && xq.ty() == types::I8
            && xq.ne()[0] as usize >= quant_bytes(k, n),
        "q40_mm: shapes"
    );
    ctx.external(&Q40_MM, types::F32, [w.ne()[1], n, 1, 1], &[xq, w])
}

/// The bytes of a [`Q40_QUANT`] node of `k` x `n` values: the blocks, and with `y` (the tests')
/// after them the F32 values it quantized (`[k, n]`).
fn quant_node_bytes(k: i64, n: i64, y: bool) -> usize {
    quant_bytes(k, n) + if y { (k * n) as usize * 4 } else { 0 }
}

#[allow(clippy::too_many_arguments)]
fn quant_node<'a>(
    ctx: &'a Context,
    kind: i32,
    srcs: &[Tensor<'a>],
    k: i64,
    n: i64,
    eps: f32,
    y: bool,
) -> Tensor<'a> {
    assert!(
        k > 0 && k % K_STEP == 0 && n > 0 && n < 1 << 24 && quant_bytes(k, n) <= MAX_QUANT_BYTES,
        "q40_quant: shapes"
    );
    let bytes = quant_node_bytes(k, n, y) as i64;
    let params = [kind, eps.to_bits() as i32, y as i32, k as i32, n as i32];
    ctx.external_with(&Q40_QUANT, types::I8, [bytes, 1, 1, 1], srcs, &params)
}

/// The op's quantization of the F32 `x` (`[k, n]`, as [`fits`] takes it) as its own node, for
/// [`mul_mat_q`]: the same bytes the op's own quantization writes.
pub fn quantize<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let [k, n, _, _] = x.ne();
    assert!(x.ty() == types::F32 && x.nb()[0] == 4, "q40_quant: x");
    quant_node(ctx, QUANT_PLAIN, &[x], k, n, 0.0, false)
}

/// [`quantize`] of ggml's `mul(rms_norm(x, eps), w)` ([`norm_fits`]) in one pass over `x`: the
/// bytes [`quantize`] writes for the F32 values ggml-cuda's fused norm and mul write, without
/// those values. `y` (the tests'): the node keeps them after the blocks too.
pub fn quantize_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    y: bool,
) -> Tensor<'a> {
    assert!(norm_fits(x, w), "q40_quant: norm shapes");
    let [k, n, _, _] = x.ne();
    quant_node(ctx, QUANT_NORM, &[x, w], k, n, eps, y)
}

/// [`quantize`] of ggml's `geglu_split(gate, up)` ([`geglu_fits`]), each value as ggml-cuda's
/// kernel computes it, without writing them. `y` as in [`quantize_norm`].
pub fn quantize_geglu<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    y: bool,
) -> Tensor<'a> {
    assert!(geglu_fits(gate, up), "q40_quant: geglu shapes");
    let [k, n, _, _] = gate.ne();
    quant_node(ctx, QUANT_GEGLU, &[gate, up], k, n, 0.0, y)
}

/// Where the quantization (layout 1, `src/cuda/q40_mm.cu`) puts column `col`'s block `kb`: the
/// byte of its word `w` (values 4s .. 4s + 3 of slot s, `w = 2s` for s < 4, `2(s - 4) + 1`
/// else) among the blocks, and the index of its scale among the F32 scales after them.
pub fn quant_at(col: usize, kb: usize, w: usize, n_pad: usize) -> (usize, usize) {
    let tile = kb * n_pad + (col & !127);
    let (nt, g) = ((col & 127) / 8, col % 8);
    let word = tile * 8 + ((nt / 2 * 8 + g) * 4 + w / 2) * 4 + nt % 2 * 2 + w % 2;
    (word * 4, tile + g / 2 * 32 + 2 * nt + g % 2)
}

/// The word of slot `s` of a block.
fn slot_word(s: usize) -> usize {
    if s < 4 { 2 * s } else { 2 * (s - 4) + 1 }
}

/// Writes column `col`'s quantization `(q, d)` ([`quantize_col`]) into `buf` (the blocks of
/// `k` x `n_pad`, then the scales).
///
/// # Safety
/// `buf` holds [`quant_bytes`] of `k` x (any `n` padding to `n_pad`) bytes.
unsafe fn put_col(buf: *mut u8, k: usize, n_pad: usize, col: usize, q: &[i8], d: &[f32]) {
    let scales = (k / QK as usize * n_pad * 32) as isize;
    // SAFETY: the caller's: every offset is within the blocks or the scales.
    unsafe {
        for kb in 0..k / QK as usize {
            for s in 0..8 {
                let (at, sd) = quant_at(col, kb, slot_word(s), n_pad);
                for b in 0..4 {
                    *buf.add(at + b) = q[kb * 32 + 4 * s + b] as u8;
                }
                if s == 0 {
                    *(buf.offset(scales) as *mut f32).add(sd) = d[kb];
                }
            }
        }
    }
}

/// Column `col`'s quantization `(q, d)` from `buf` (as [`put_col`] writes it).
pub fn get_col(buf: &[u8], k: usize, n_pad: usize, col: usize) -> (Vec<i8>, Vec<f32>) {
    let scales = k / QK as usize * n_pad * 32;
    let mut q = vec![0i8; k];
    let mut d = vec![0f32; k / QK as usize];
    for kb in 0..k / QK as usize {
        for s in 0..8 {
            let (at, sd) = quant_at(col, kb, slot_word(s), n_pad);
            for b in 0..4 {
                q[kb * 32 + 4 * s + b] = buf[at + b] as i8;
            }
            if s == 0 {
                let o = scales + sd * 4;
                d[kb] = f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            }
        }
    }
    (q, d)
}

/// ggml's tanh GELU (a reference: not necessarily ggml's bits on the CPU).
fn gelu_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

unsafe fn cpu_q40_quant(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let p = crate::ggml::external_params(dst);
    let (kind, eps, y, k, n) = (
        p[0],
        f32::from_bits(p[1] as u32),
        p[2] != 0,
        p[3] as usize,
        p[4] as usize,
    );
    let n_pad = n_padded(n as i64) as usize;
    let out = dst.data as *mut u8;
    // SAFETY: a node made by `quant_node`: host tensors of the shapes it checked.
    unsafe {
        let col_of = |t: *const ffi::ggml_tensor, c: usize| {
            let t = &*t;
            std::slice::from_raw_parts((t.data as *const u8).add(c * t.nb[1]) as *const f32, k)
        };
        for c in (ith..n).step_by(nth) {
            let x = col_of(dst.src[0], c);
            let v: Vec<f32> = match kind {
                QUANT_NORM => {
                    let w = std::slice::from_raw_parts((*dst.src[1]).data as *const f32, k);
                    let ss = x.iter().fold(0f32, |s, v| v.mul_add(*v, s));
                    let scale = 1.0 / (ss / k as f32 + eps).sqrt();
                    x.iter().zip(w).map(|(x, w)| scale * x * w).collect()
                }
                QUANT_GEGLU => {
                    let u = col_of(dst.src[1], c);
                    x.iter().zip(u).map(|(g, u)| gelu_ref(*g) * u).collect()
                }
                _ => x.to_vec(),
            };
            if y {
                let yo = out.add(quant_bytes(k as i64, n as i64)) as *mut f32;
                std::ptr::copy_nonoverlapping(v.as_ptr(), yo.add(c * k), k);
            }
            let (q, d) = quantize_col(&v);
            put_col(out, k, n_pad, c, &q, &d);
        }
    }
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
        let (k, n, rows) = (w.ne[0] as usize, dst.ne[1] as usize, w.ne[1] as usize);
        let blocks = k / QK as usize;
        let out = dst.data as *mut f32;
        // A `mul_mat_q` node's input: the blocks already.
        let pre = x.type_ == types::I8;
        let xq = pre.then(|| {
            std::slice::from_raw_parts(x.data as *const u8, quant_bytes(k as i64, n as i64))
        });
        let n_pad = n_padded(n as i64) as usize;
        for c in (ith..n).step_by(nth) {
            let (q, d) = match xq {
                Some(buf) => get_col(buf, k, n_pad, c),
                None => {
                    let col = (x.data as *const u8).add(c * x.nb[1]) as *const f32;
                    quantize_col(std::slice::from_raw_parts(col, k))
                }
            };
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

    /// `legion_q40_quant` of `src/cuda/q40_mm.cu`.
    #[repr(C)]
    struct QuantParams {
        x: *const f32,
        x2: *const f32,
        y: *mut f32,
        yq: *mut i32,
        yd: *mut f32,
        x_ld: i64,
        x2_ld: i64,
        eps: f32,
        k: i32,
        n: i32,
        n_pad: i32,
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
        /// The fused quantizations: the GEGLU's, the norm's below 1024 values and from 1024.
        static legion_q40_quant_kernels: [*const c_void; 3];
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

    fn launch<P>(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        shared: usize,
        p: &mut P,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut P).cast::<c_void>()];
        // SAFETY: a kernel of src/cuda/q40_mm.cu taking its arguments struct (`Params` or
        // `QuantParams`, the one it is launched with) by value, from the launch's copy of `p`;
        // its pointers are device memory of the stream's device.
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
            let (k, n, rows) = (w.ne[0], dst.ne[1], w.ne[1]);
            if !(w.data as usize).is_multiple_of(16) {
                return Err("q40_mm: weight not 16-byte aligned".into());
            }
            let mut dev: c_int = 0;
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                return Err("q40_mm: no current device".into());
            }
            // A `mul_mat_q` node: its input holds the blocks (layout 1) already.
            let pre = x.type_ == types::I8;
            let quant = !pre && quant_in(crate::ggml::current_cuda_graph(), dst);
            let v = variant();
            let kern = kernel(v);
            if pre && kern.quant != kernel(0).quant {
                return Err(format!("q40_mm: matmul {v} reads another layout"));
            }
            let (bm, bn) = (kern.bm as i64, kern.bn as i64);
            if rows % bm != 0 {
                return Err(format!("q40_mm: {rows} rows, tiles of {bm}"));
            }
            let (ntx, nty) = ((n + bn - 1) / bn, rows / bm);
            let n_pad = ntx * bn;
            let yd_off = (k / QK) as usize * n_pad as usize * 32;
            let base = if pre {
                if !(x.data as usize).is_multiple_of(16) {
                    return Err("q40_mm: quantized input not 16-byte aligned".into());
                }
                x.data as usize
            } else {
                scratch(dev, stream, (k / QK) as usize * n_pad as usize * 36)?.ptr
            };
            let mut a = Params {
                w: w.data as *const c_void,
                x: x.data as *const f32,
                yq: base as *mut i32,
                yd: (base + yd_off) as *mut f32,
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

    /// A [`Q40_QUANT`] node: the op's quantization (the table's first, layout 1) of its input,
    /// or one of the fused quantizations.
    pub(super) unsafe fn q40_quant(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let p = crate::ggml::external_params(dst);
        let (kind, eps, y, k, n) = (
            p[0],
            f32::from_bits(p[1] as u32),
            p[2] != 0,
            p[3] as i64,
            p[4] as i64,
        );
        let n_pad = n_padded(n);
        // SAFETY: a node made by `quant_node` (its shapes checked there), device tensors of
        // the stream's device; the node holds `quant_node_bytes` bytes.
        unsafe {
            let base = dst.data as usize;
            if !base.is_multiple_of(16) {
                return Err("q40_quant: output not 16-byte aligned".into());
            }
            let yq = base as *mut i32;
            let yd = (base + (k / QK) as usize * n_pad as usize * 32) as *mut f32;
            let yo = if y {
                (base + quant_bytes(k, n)) as *mut f32
            } else {
                std::ptr::null_mut()
            };
            let x = &*dst.src[0];
            let wide = Dim3 {
                x: n as u32,
                y: (k / K_STEP) as u32,
                z: 1,
            };
            let block64 = Dim3 { x: 64, y: 1, z: 1 };
            if kind == QUANT_PLAIN {
                let mut a = Params {
                    w: std::ptr::null(),
                    x: x.data as *const f32,
                    yq,
                    yd,
                    dst: std::ptr::null_mut(),
                    x_ld: (x.nb[1] / 4) as i64,
                    w_nb1: 0,
                    dst_ld: 0,
                    k: k as i32,
                    n: n as i32,
                    n_pad: n_pad as i32,
                    rows: 0,
                    ntx: 0,
                    nty: 0,
                };
                return launch(kernel(0).quant, wide, block64, 0, &mut a, stream);
            }
            let x2 = &*dst.src[1];
            let mut a = QuantParams {
                x: x.data as *const f32,
                x2: x2.data as *const f32,
                y: yo,
                yq,
                yd,
                x_ld: (x.nb[1] / 4) as i64,
                x2_ld: (x2.nb[1] / 4) as i64,
                eps,
                k: k as i32,
                n: n as i32,
                n_pad: n_pad as i32,
            };
            let kernels = &*std::ptr::addr_of!(legion_q40_quant_kernels);
            if kind == QUANT_GEGLU {
                launch(kernels[0], wide, block64, 0, &mut a, stream)
            } else {
                let grid = Dim3 {
                    x: n as u32,
                    y: 1,
                    z: 1,
                };
                let block = Dim3 { x: 256, y: 1, z: 1 };
                let f = if k < 1024 { kernels[1] } else { kernels[2] };
                launch(f, grid, block, 0, &mut a, stream)
            }
        }
    }
}
