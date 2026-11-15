//! Quantized routed-expert matvecs that stream each picked expert's rows once per step (#1176):
//! `mul_mat_id(w, x, ids)` of a Q4_K, Q5_K, Q6_K, Q5_1 or Q8_0 expert stack for steps of 1 to
//! [`MAX_TOKENS`] tokens (decode and MTP verify steps), as an external op running
//! `src/cuda/mmvq_once.cu` in place of ggml-cuda's quantize_q8_1 + MMVQ pair, with ggml's bits.
//!
//! On a GB10 ggml's MMVQ reads Qwen3.8-Flash-Next UD-Q4_K_XL's routed experts at 153-176 GB/s
//! at one token (a CTA per row and slot, 1.4 KB each, for the Q4_K gate and up) and streams an
//! expert's rows again for every pair that picked it at 2-8 tokens. The op splits the step's
//! distinct experts' rows into one equal contiguous range per resident CTA and streams each range
//! through a shared-memory ring with cp.async, adding every pair of a row's expert from the staged
//! row: ggml's own vec_dot in ggml's thread layout and summing order (its 4-warp 1-token kernel's
//! or its warp-per-pair MoE kernel's, see the kernel file), so each output keeps ggml's bits.
//!
//! Two ops: [`mul_mat_id`] (one matmul) and [`swiglu_id`] (the gate and up matmuls of the same
//! experts on one quantization of `x`, then `silu(gate) * up`), the bits of the graph nodes they
//! replace (`mul_mat_id`, and `mul_mat_id`, `mul_mat_id`, `silu`, `mul`).
//!
//! Off by default (opt-in for A/B runs, [`Mode`]): on compute capability 12.x (GB10, RTX 50;
//! [`set_gpu`]) `LEGION_DEBUG_MMVQ_ONCE=1` takes the SwiGLU op for steps of up to
//! [`GLU_TOKENS`] tokens, where it measured faster than ggml's MMVQ on a GB10, and `=all` (or
//! `split`) adds the one-matmul op (the down matmul) and longer steps, which measured slower.
//! Each op keeps ggml's bits, but swapping its graph nodes moves ggml's allocation and with it
//! ggml-cuda's layout-dependent MoE fusions (#472), so served greedy text can change. `legion mmvq-once-bench` times the ops against ggml's and
//! checks their bits ([`bench`]). The CPU implementation is a plain f32 reference, for tests of
//! the wiring.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static MMVQ_ONCE: ExternalOp = ExternalOp {
    name: "mmvq_once",
    cpu: cpu_mmvq_once,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmvq_once),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static MMVQ_ONCE_GLU: ExternalOp = ExternalOp {
    name: "mmvq_once_glu",
    cpu: cpu_mmvq_once,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_mmvq_once),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most tokens of a step the op takes: ggml-cuda runs quantized `mul_mat_id` on its MMVQ up to 8
/// tokens on Ada and newer (`get_mmvq_mmid_max_batch`), on MMQ above.
pub const MAX_TOKENS: i64 = 8;

/// Most (token, slot) pairs of a step: the routing plan's (`nvfp4_plan.cuh`, a thread per pair).
pub const MAX_PAIRS: i64 = 128;

/// Q8_1 blocks of the device-global activation scratch (`n_x * n * k / 32` at most).
pub const SCRATCH_BLOCKS: i64 = 8192;

/// Stages of a CTA's shared-memory ring (the kernel's `STAGES`).
pub const STAGES: usize = 3;

/// Most bytes of a weight row: a SwiGLU's ring of [`STAGES`] stages of one row of each matmul
/// fits the 99 KB a CTA may ask for on sm_120.
pub const MAX_ROW_BYTES: usize = 16 * 1024;

/// Most bytes of a CTA's ring: within the 99 KB a CTA may ask for on sm_120.
pub const RING_BYTES: usize = 96 * 1024;

/// The weight types, in the order of `legion_mmvq_once_kernels`.
pub const TYPES: [ffi::ggml_type; 5] = [
    types::Q4_K,
    types::Q5_K,
    types::Q6_K,
    types::Q5_1,
    types::Q8_0,
];

/// Most tokens of a step [`swiglu_id`] takes under [`Mode::On`]: in serving op profiles
/// on a GB10 it beat ggml's MMVQ at 1-3 tokens and was slower at 4-8 (#1176).
pub const GLU_TOKENS: i64 = 3;

static MODE: AtomicU8 = AtomicU8::new(Mode::Off as u8);
static GPU: AtomicBool = AtomicBool::new(false);

/// What `LEGION_DEBUG_MMVQ_ONCE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// `1` (`on`, `true`, `yes`): [`swiglu_id`] for the routed gate and up matmuls of steps of
    /// up to [`GLU_TOKENS`] tokens, ggml's MMVQ for the rest. On a GB10 the one-matmul op (the
    /// down matmul) and steps of 4-8 tokens measured slower than ggml's MMVQ (#1176).
    On,
    /// Unset (or anything else): ggml's MMVQ.
    Off,
    /// `all`: both ops at every step they fit ([`MAX_TOKENS`]).
    All,
    /// `split`: [`mul_mat_id`] for every matmul, the gate and up ones apart, no [`swiglu_id`].
    Split,
}

/// `LEGION_DEBUG_MMVQ_ONCE`'s value `v` as a [`Mode`].
pub fn mode_of(v: Option<&str>) -> Mode {
    match v.map(str::trim) {
        Some("1" | "on" | "true" | "yes") => Mode::On,
        Some("all") => Mode::All,
        Some("split") => Mode::Split,
        _ => Mode::Off,
    }
}

/// The mode [`set_gpu`] or [`set_mode`] set.
fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::On,
        2 => Mode::All,
        3 => Mode::Split,
        _ => Mode::Off,
    }
}

/// Whether `mode` takes the one-matmul op ([`mul_mat_id`]).
fn mode_takes_mul_mat(mode: Mode) -> bool {
    matches!(mode, Mode::All | Mode::Split)
}

/// Whether `mode` takes the SwiGLU op ([`swiglu_id`]) for a step of `n` tokens.
fn mode_takes_glu(mode: Mode, n: i64) -> bool {
    match mode {
        Mode::On => n <= GLU_TOKENS,
        Mode::All => true,
        Mode::Off | Mode::Split => false,
    }
}

/// The engine's GPU: the ops are used on compute capability 12.x (#1176: the GB10, where their
/// bits were checked against ggml's MMVQ and where they were measured; the RTX 50 shares its SM).
/// `LEGION_DEBUG_MMVQ_ONCE` (read here, [`mode_of`]) turns them off, on everywhere they fit, or
/// splits the SwiGLU op for A/B runs.
pub fn set_gpu(cc: Option<(i32, i32)>) {
    GPU.store(gpu_takes(cc), Ordering::Relaxed);
    set_mode(mode_of(
        std::env::var("LEGION_DEBUG_MMVQ_ONCE").ok().as_deref(),
    ));
}

/// Sets the mode directly ([`set_gpu`] reads it from the environment).
#[doc(hidden)]
pub fn set_mode(mode: Mode) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

/// Whether compute capability `cc` (major, minor) takes the ops ([`set_gpu`]).
pub fn gpu_takes(cc: Option<(i32, i32)>) -> bool {
    cc.is_some_and(|(major, _)| major == 12)
}

/// Whether the GPU of device `dev` (the weights') takes the ops, asked per device; the engine's
/// answer where `dev` is unknown.
fn gpu_for(dev: Option<usize>) -> bool {
    gpu_for_by(dev, GPU.load(Ordering::Relaxed), crate::hw::per_device::cc)
}

/// [`gpu_for`] with the engine's answer `engine` and the compute capabilities `cc_of` gives.
fn gpu_for_by(
    dev: Option<usize>,
    engine: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, gpu_takes, cc_of)
}

/// Whether the ops are on for the weight `w` (its device's answer, [`gpu_for`]).
pub(crate) fn on(w: Tensor<'_>) -> bool {
    mode() != Mode::Off && cfg!(feature = "cuda") && gpu_for(w.cuda_device())
}

/// Whether [`mul_mat_id`] takes `mul_mat_id(w, x, ids)` (no per-expert scale): under
/// `LEGION_DEBUG_MMVQ_ONCE=all` or `split` only ([`Mode`]), on ([`on`]) and [`fits_shape`].
pub fn usable(w: Tensor<'_>, scale: Option<Tensor<'_>>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    mode_takes_mul_mat(mode()) && scale.is_none() && fits_shape(w, x, ids) && on(w)
}

/// Whether [`swiglu_id`] takes the gate and up matmuls of `x`: the mode takes the step's tokens
/// ([`Mode`]), on, and both weights fit ([`fits_shape`]) with the same type, shapes and strides.
pub fn glu_usable(gate: Tensor<'_>, up: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    mode_takes_glu(mode(), ids.ne()[1]) && glu_fits(gate, up, x, ids) && on(gate)
}

/// [`glu_usable`]'s shape checks.
pub fn glu_fits(gate: Tensor<'_>, up: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    gate.ty() == up.ty()
        && gate.ne() == up.ne()
        && gate.nb() == up.nb()
        && fits_shape(gate, x, ids)
        && fits_shape(up, x, ids)
}

/// The index of `ty` in [`TYPES`].
fn type_index(ty: ffi::ggml_type) -> Option<usize> {
    TYPES.iter().position(|&t| t == ty)
}

/// Whether the op can compute `mul_mat_id(w, x, ids)` with ggml's MMVQ bits: an expert stack of
/// one of [`TYPES`] in one device's memory with contiguous rows of at most [`MAX_ROW_BYTES`],
/// F32 activations shared by the slots (`[k, 1, n]`) or one per slot (`[k, n_used, n]`), I32 ids
/// `[n_used, n]`, 1 <= n <= [`MAX_TOKENS`], and the pairs and the activations' blocks within
/// [`MAX_PAIRS`] and [`SCRATCH_BLOCKS`].
pub fn fits_shape(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    let [k, rows, n_expert, w3] = w.ne();
    let [xk, n_x, n, x3] = x.ne();
    let [n_used, n_ids, i2, i3] = ids.ne();
    let (wnb, xnb, inb) = (w.nb(), x.nb(), ids.nb());
    let Some(_) = type_index(w.ty()) else {
        return false;
    };
    let qk = crate::ggml::block_size(w.ty()) as i64;
    if k <= 0 || k % qk != 0 || k > i32::MAX as i64 {
        return false;
    }
    let row_bytes = crate::ggml::row_size(w.ty(), k as usize);
    xk == k
        && rows > 0
        && rows <= i32::MAX as i64
        && w3 == 1
        && x3 == 1
        && n_used > 0
        && (n_x == 1 || n_x == n_used)
        && n_ids == n
        && i2 == 1
        && i3 == 1
        && (1..=MAX_TOKENS).contains(&n)
        && n_used * n <= MAX_PAIRS
        && n_x * n * (k / 32) <= SCRATCH_BLOCKS
        && (1..=1 << 16).contains(&n_expert)
        && x.ty() == types::F32
        && ids.ty() == types::I32
        && xnb[0] == 4
        && xnb[1].is_multiple_of(4)
        && xnb[2].is_multiple_of(4)
        && inb[0] == 4
        && inb[1].is_multiple_of(4)
        && row_bytes <= MAX_ROW_BYTES
        && row_bytes.is_multiple_of(4)
        && wnb[1] == row_bytes
        && wnb[2].is_multiple_of(4)
        && wnb[2] >= row_bytes * rows as usize
        && !w.is_host()
        && !w.is_meta()
}

/// `mul_mat_id(w, x, ids)` (F32 `[rows, n_used, n]`) on the op; the caller checked [`usable`].
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    let rows = w.ne()[1];
    let [n_used, n, _, _] = ids.ne();
    ctx.external_with(
        &MMVQ_ONCE,
        types::F32,
        [rows, n_used, n, 1],
        &[w, x, ids],
        &[0],
    )
}

/// `silu(gate x) * (up x)` of the routed experts on one op: F32 `[rows, n_used, n]`; the caller
/// checked [`glu_usable`].
pub fn swiglu_id<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    let rows = up.ne()[1];
    let [n_used, n, _, _] = ids.ne();
    ctx.external_with(
        &MMVQ_ONCE_GLU,
        types::F32,
        [rows, n_used, n, 1],
        &[up, x, ids, gate],
        &[1],
    )
}

/// A launch's shape choices (every one gives the same bits): bytes of a stage's rows per matmul
/// (its rows: as many whole rows as fit, at least one) and warps per CTA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub stage_bytes: usize,
    pub warps: u32,
}

/// The defaults at one token: 16 KB stages (a SwiGLU's ring of Flash-Next's Q4_K gate and up
/// rows is 3 x 2 x 8 rows, 68 KB, one CTA an SM) and 8 warps, the best of `legion
/// mmvq-once-bench`'s tunings for the SwiGLU op on a GB10 (#1176: 1-3% over 8 KB and 4 warps,
/// and faster in a serving op profile).
pub const DEFAULT_TUNING: Tuning = Tuning {
    stage_bytes: 16 * 1024,
    warps: 8,
};

/// The defaults at 2-8 tokens: 8 KB stages (3 x 2 x 4 rows, 34 KB, two CTAs an SM) and 4 warps.
/// The bench put them within 1-2% of [`DEFAULT_TUNING`] at 2-4 tokens; in a serving op profile
/// of four users the SwiGLU op took 211-258 us with them and 261 us with [`DEFAULT_TUNING`]
/// (#1176).
pub const DEFAULT_TUNING_STEPS: Tuning = Tuning {
    stage_bytes: 8 * 1024,
    warps: 4,
};

/// The default tuning of a step of `n_tok` tokens.
pub fn default_tuning(n_tok: usize) -> Tuning {
    if n_tok <= 1 {
        DEFAULT_TUNING
    } else {
        DEFAULT_TUNING_STEPS
    }
}

/// The forced tuning (stage bytes and warps packed, 0: the defaults).
static FORCED: AtomicUsize = AtomicUsize::new(0);

/// Launches every op from now on with `t`, or the defaults for `None`: for the bench (the same
/// bits either way).
#[doc(hidden)]
pub fn force_tuning(t: Option<Tuning>) {
    FORCED.store(packed(t), Ordering::Relaxed);
}

/// A forced tuning as [`FORCED`] holds it (warps 1-8).
fn packed(t: Option<Tuning>) -> usize {
    t.map_or(0, |t| {
        (t.stage_bytes.max(1) << 8) | t.warps.clamp(1, 8) as usize
    })
}

/// The tuning [`FORCED`]'s value `v` stands for (`None`: the defaults).
fn unpacked(v: usize) -> Option<Tuning> {
    match v {
        0 => None,
        v => Some(Tuning {
            stage_bytes: v >> 8,
            warps: (v & 0xff) as u32,
        }),
    }
}

/// The tuning of a launch of `n_tok` tokens: forced or [`default_tuning`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn tuning(n_tok: usize) -> Tuning {
    unpacked(FORCED.load(Ordering::Relaxed)).unwrap_or_else(|| default_tuning(n_tok))
}

/// Rows of a stage: as many rows of `row_bytes` as fit `t.stage_bytes` (at least one), a
/// multiple of the warps when there are more rows than warps, at most `rows`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn stage_rows(t: Tuning, row_bytes: usize, rows: usize) -> usize {
    let fit = (t.stage_bytes / row_bytes.max(1)).max(1);
    let warps = t.warps as usize;
    let fit = if fit > warps {
        fit / warps * warps
    } else {
        fit
    };
    fit.min(rows).max(1)
}

/// The largest cp.async size (16, 8 or 4 bytes) that divides every one of `v` (addresses and
/// strides); 0 if 4 does not.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn copy_unit(v: &[usize]) -> usize {
    [16, 8, 4]
        .into_iter()
        .find(|u| v.iter().all(|x| x.is_multiple_of(*u)))
        .unwrap_or(0)
}

/// The kernel of `legion_mmvq_once_kernels` for type `ti` (into [`TYPES`]), the op (`glu`) and
/// the step's tokens (1: ggml's 1-token order, else its MoE kernel's).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn kernel_index(ti: usize, glu: bool, n_tok: usize) -> usize {
    1 + 4 * ti + 2 * usize::from(glu) + usize::from(n_tok > 1)
}

struct Operands {
    glu: bool,
    ty: ffi::ggml_type,
    w: *const u8,
    wg: *const u8,
    w_nb: [usize; 4],
    x: *const u8,
    x_nb: [usize; 4],
    n_x: usize,
    ids: *const u8,
    ids_nb1: usize,
    dst: *mut u8,
    dst_nb: [usize; 4],
    k: usize,
    rows: usize,
    n_used: usize,
    n_tok: usize,
    n_expert: usize,
}

impl Operands {
    /// # Safety
    /// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        let [glu, ..] = external_params(dst);
        let glu = glu != 0;
        unsafe {
            let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
            // A SwiGLU node has the up weights first and the gate weights fourth.
            let wg = if glu {
                (*dst.src[3]).data as *const u8
            } else {
                std::ptr::null()
            };
            Self {
                glu,
                ty: w.type_,
                w: w.data as *const u8,
                wg,
                w_nb: w.nb,
                x: x.data as *const u8,
                x_nb: x.nb,
                n_x: x.ne[1] as usize,
                ids: ids.data as *const u8,
                ids_nb1: ids.nb[1],
                dst: dst.data as *mut u8,
                dst_nb: dst.nb,
                k: w.ne[0] as usize,
                rows: w.ne[1] as usize,
                n_used: ids.ne[0] as usize,
                n_tok: ids.ne[1] as usize,
                n_expert: w.ne[2] as usize,
            }
        }
    }
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`], all tensors in host memory.
unsafe fn cpu_mmvq_once(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id` or `swiglu_id`, all tensors in host memory.
    let o = unsafe { Operands::of(dst) };
    let row_bytes = crate::ggml::row_size(o.ty, o.k);
    for pair in (ith..o.n_used * o.n_tok).step_by(nth) {
        let (t, j) = (pair / o.n_used, pair % o.n_used);
        // SAFETY: in-bounds reads of ids, x and the expert's rows; writes to dst's row.
        unsafe {
            let e = *(o.ids.add(t * o.ids_nb1 + j * 4) as *const i32);
            let Some(e) = usize::try_from(e).ok().filter(|&e| e < o.n_expert) else {
                continue;
            };
            let out = o.dst.add(t * o.dst_nb[2] + j * o.dst_nb[1]) as *mut f32;
            let xj = if o.n_x == 1 { 0 } else { j };
            let x = std::slice::from_raw_parts(
                o.x.add(t * o.x_nb[2] + xj * o.x_nb[1]) as *const f32,
                o.k,
            );
            let dot = |w: *const u8, r: usize| {
                let row =
                    std::slice::from_raw_parts(w.add(e * o.w_nb[2] + r * o.w_nb[1]), row_bytes);
                let w = crate::ggml::dequantize(o.ty, row, o.k);
                w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
            };
            for r in 0..o.rows {
                let mut v = dot(o.w, r);
                if o.glu {
                    let g = dot(o.wg, r);
                    v *= g / (1.0 + (-g).exp());
                }
                *out.add(r) = v;
            }
        }
    }
}

/// `legion_mmvq_once` of `src/cuda/mmvq_once.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Params {
    x: *const f32,
    x_nb1: i64,
    x_nb2: i64,
    n_x: i32,
    w: *const c_void,
    wg: *const c_void,
    w_nb2: i64,
    row_bytes: i32,
    ids: *const i32,
    ids_nb1: i64,
    dst: *mut f32,
    dst_nb1: i64,
    dst_nb2: i64,
    k: i32,
    rows: i32,
    n_used: i32,
    n_tok: i32,
    n_expert: i32,
    rt: i32,
    unit: i32,
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`] (its shapes passed [`fits_shape`]),
/// its tensors in device memory of the stream's device; `stream` is the compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_mmvq_once(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat_id` or `swiglu_id`.
    let o = unsafe { Operands::of(dst) };
    let ti = type_index(o.ty).ok_or("mmvq_once: weight type")?;
    let row_bytes = crate::ggml::row_size(o.ty, o.k);
    let mut addrs = vec![o.w as usize, o.w_nb[2], row_bytes];
    if o.glu {
        addrs.push(o.wg as usize);
    }
    let unit = copy_unit(&addrs);
    if unit == 0 {
        return Err("mmvq_once: weights not 4-byte aligned".into());
    }
    let t = tuning(o.n_tok);
    let mats = 1 + usize::from(o.glu);
    // The ring within the 96 KB a CTA may ask for on sm_120 (a forced tuning may want more).
    let max_rt = (RING_BYTES / (STAGES * mats * row_bytes)).max(1);
    let rt = stage_rows(t, row_bytes, o.rows).min(max_rt);
    let smem = STAGES * mats * rt * row_bytes;
    let mut p = Params {
        x: o.x as *const f32,
        x_nb1: (o.x_nb[1] / 4) as i64,
        x_nb2: (o.x_nb[2] / 4) as i64,
        n_x: o.n_x as i32,
        w: o.w as *const c_void,
        wg: o.wg as *const c_void,
        w_nb2: o.w_nb[2] as i64,
        row_bytes: row_bytes as i32,
        ids: o.ids as *const i32,
        ids_nb1: (o.ids_nb1 / 4) as i64,
        dst: o.dst as *mut f32,
        dst_nb1: (o.dst_nb[1] / 4) as i64,
        dst_nb2: (o.dst_nb[2] / 4) as i64,
        k: o.k as i32,
        rows: o.rows as i32,
        n_used: o.n_used as i32,
        n_tok: o.n_tok as i32,
        n_expert: o.n_expert as i32,
        rt: rt as i32,
        unit: unit as i32,
    };
    // The quantization, one more z slice for the routing plan.
    let quantize = launch::Dim3 {
        x: o.k.div_ceil(256) as u32,
        y: o.n_x as u32,
        z: o.n_tok as u32 + 1,
    };
    launch::launch(
        0,
        quantize,
        launch::Dim3 { x: 256, y: 1, z: 1 },
        0,
        &mut p,
        stream,
    )?;
    let kernel = kernel_index(ti, o.glu, o.n_tok);
    let threads = 32 * t.warps;
    // One wave of resident CTAs, no more than stages of the step's rows (an expert per pair at
    // most).
    let stages = (o.rows.div_ceil(rt) * o.n_used * o.n_tok).max(1);
    let grid = launch::resident(kernel, threads, smem)?.min(stages).max(1);
    let block = launch::Dim3 {
        x: 32,
        y: t.warps,
        z: 1,
    };
    let grid = launch::Dim3 {
        x: grid as u32,
        y: 1,
        z: 1,
    };
    launch::launch(kernel, grid, block, smem, &mut p, stream)
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI32, Ordering};

    /// `dim3`.
    #[repr(C)]
    pub(super) struct Dim3 {
        pub x: u32,
        pub y: u32,
        pub z: u32,
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

    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

    /// Kernels of the table: the quantization, then 4 matvecs per type.
    const KERNELS: usize = 1 + 4 * super::TYPES.len();

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs), in the order of
        /// `super::kernel_index`.
        static legion_mmvq_once_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            blocks: *mut c_int,
            func: *const c_void,
            block_size: c_int,
            dynamic_smem: usize,
        ) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const DEV_ATTR_SM_COUNT: c_int = 16;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CC_MINOR: c_int = 76;
    /// `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
    const DEV_ATTR_SMEM_OPTIN: c_int = 97;
    /// `cudaFuncAttributeMaxDynamicSharedMemorySize`.
    const FUNC_ATTR_MAX_DYNAMIC_SMEM: c_int = 8;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// Each device's compute capability (10 * major + minor); 0 not yet read, -1 unknown.
    static CC: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// (device, kernel, threads, shared memory) -> resident CTAs, and the (device, kernel) pairs
    /// whose shared-memory limit was raised.
    #[allow(clippy::type_complexity)]
    static RESIDENT: Mutex<(
        Vec<((usize, usize, u32, usize), usize)>,
        Vec<(usize, usize)>,
    )> = Mutex::new((Vec::new(), Vec::new()));

    fn device() -> Option<usize> {
        let mut dev: c_int = 0;
        // SAFETY: a plain query of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return None;
            }
        }
        Some(dev as usize)
    }

    /// The current device's compute capability, read once per device.
    fn compute_capability() -> Option<i32> {
        let dev = device()?;
        let slot = &CC[dev];
        let mut cc = slot.load(Ordering::Relaxed);
        if cc == 0 {
            let (mut major, mut minor) = (0, 0);
            // SAFETY: plain queries of the current device.
            cc = unsafe {
                if cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev as c_int) == 0
                    && cudaDeviceGetAttribute(&mut minor, DEV_ATTR_CC_MINOR, dev as c_int) == 0
                {
                    10 * major + minor
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(cc, Ordering::Relaxed);
        }
        (cc > 0).then_some(cc)
    }

    /// CTAs of `threads` threads and `smem` bytes of dynamic shared memory of kernel `kernel` the
    /// current device holds at once (one wave), read once per device and shape; the kernel's
    /// shared-memory limit raised to the device's first.
    pub(super) fn resident(kernel: usize, threads: u32, smem: usize) -> Result<usize, String> {
        let dev = device().ok_or("mmvq_once: no current CUDA device")?;
        let mut cache = RESIDENT.lock().unwrap_or_else(|e| e.into_inner());
        let key = (dev, kernel, threads, smem);
        if let Some(&(_, n)) = cache.0.iter().find(|(k, _)| *k == key) {
            return Ok(n);
        }
        // SAFETY: plain queries of the current device and of a kernel of the table.
        let n = unsafe {
            let func = legion_mmvq_once_kernels[kernel];
            if !cache.1.contains(&(dev, kernel)) {
                let mut optin = 0;
                if cudaDeviceGetAttribute(&mut optin, DEV_ATTR_SMEM_OPTIN, dev as c_int) != 0
                    || cudaFuncSetAttribute(func, FUNC_ATTR_MAX_DYNAMIC_SMEM, optin) != 0
                {
                    cudaGetLastError();
                    return Err("mmvq_once: shared-memory limit".into());
                }
                cache.1.push((dev, kernel));
            }
            let (mut sms, mut per_sm) = (0, 0);
            if cudaDeviceGetAttribute(&mut sms, DEV_ATTR_SM_COUNT, dev as c_int) != 0
                || cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mut per_sm,
                    func,
                    threads as c_int,
                    smem,
                ) != 0
                || per_sm < 1
            {
                cudaGetLastError();
                return Err(format!("mmvq_once: no CTA of {smem} B shared memory fits"));
            }
            (sms * per_sm) as usize
        };
        cache.0.push((key, n));
        Ok(n)
    }

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream` with `smem` bytes of
    /// dynamic shared memory; on compute capability 12.x as a programmatic dependent launch, as
    /// ggml-cuda launches its own kernels there (each kernel waits for the one ahead before it
    /// reads activations, ids or the plan, `src/cuda/mmvq_once.cu`).
    pub(super) fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        smem: usize,
        p: &mut super::Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut super::Params).cast::<c_void>()];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid,
            block,
            dynamic_smem: smem,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(compute_capability().is_some_and(|cc| cc / 10 == 12)),
        };
        // SAFETY: a kernel of the table taking `legion_mmvq_once` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe {
            cudaLaunchKernelExC(&config, legion_mmvq_once_kernels[kernel], argv.as_mut_ptr())
        } {
            0 => Ok(()),
            e => Err(format!("mmvq_once: CUDA error {e}")),
        }
    }
}
