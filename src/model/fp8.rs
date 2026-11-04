//! FP8 W8A16 matmul: weights kept as the checkpoint's own FP8 E4M3 bytes, F32 activations.
//!
//! ggml has no FP8 type, so an FP8 weight is a ggml **I8** tensor `[k, m]` holding the raw E4M3
//! bytes (output row `j` is bytes `j·k .. (j+1)·k`), and its `weight_scale` is a separate F32
//! tensor that the matmuls multiply into their result in the same op: `[1]` for a per-tensor
//! scale (ModelOpt), `[m]` for one per output row (compressed-tensors' channel-wise FP8).
//! [`mul_mat`] (W8A16) and [`quantize`] + [`mul_mat_a8`] (W8A8) are external ops
//! ([`crate::ggml::ExternalOp`]).
//!
//! - CUDA (`src/cuda/fp8_mm.cu`): a GEMV for up to 4 tokens (decode; memory-bound, one 128-bit
//!   weight load per lane and step, F32 math, each token the bits it gets alone), and a
//!   tensor-core GEMM above that (prefill; the weight tile converted to BF16 exactly, the
//!   activations rounded to BF16, F32 accumulation). [`set_small_batch_gemv`] moves 5-9 tokens
//!   (verify steps) to the GEMV on sm_80+, and [`set_sliced_k`] 5-64 tokens (the rest of them,
//!   with the GEMV on) to the sliced-K tensor-core GEMM; before sm_80 the GEMV takes every batch.
//! - CPU: a plain reference (exact E4M3 values, F32 dot products), for tests and CPU runs.
//!
//! E4M3 here is the `fn` variant every FP8 checkpoint uses: 4 exponent bits (bias 7), 3 mantissa
//! bits, subnormals, no infinities, `0x7f` / `0xff` NaN, largest finite 448.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use std::sync::atomic::{AtomicBool, Ordering};

static FP8_MM: ExternalOp = ExternalOp {
    name: "fp8_mm",
    cpu: cpu_fp8_mm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_fp8_mm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs built from now on run W8A16 matmuls of 5-9 tokens on the GEMV
/// ([`set_small_batch_gemv`]).
static SMALL_BATCH_GEMV: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on run W8A16 matmuls of 5-9 tokens (a speculative verify
/// step's rows, or a decode step of 5-9 sequences) on GPUs with BF16 tensor cores on the GEMV,
/// which gives every token the bits it gets alone in a one-token decode step, instead of the
/// tensor-core GEMM (`legion serve --fp8-small-batch-gemv`). Off by default: it changes those
/// steps' bits.
pub fn set_small_batch_gemv(on: bool) {
    SMALL_BATCH_GEMV.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on run W8A16 matmuls of 5-64 tokens ([`set_sliced_k`]).
static SLICED_K: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on run W8A16 matmuls of 5-64 tokens (speculative verify steps,
/// decode steps of that many sequences, short prompt chunks) on GPUs with BF16 tensor cores on
/// the sliced-K GEMM (`src/cuda/sm80/fp8_sliced_k.cuh`, after NInfer: 16 weight rows per CTA, K
/// split over its warps, which streams the weights near the GEMV's bandwidth) instead of the
/// prefill-shaped GEMMs (`legion serve --fp8-sliced-k`). The 5-9 token steps that
/// [`set_small_batch_gemv`] gives the GEMV stay there. Off by default: the sums round in another
/// order, so those steps' bits change. Rows whose length is not a multiple of 256 keep the GEMMs.
pub fn set_sliced_k(on: bool) {
    SLICED_K.store(on, Ordering::Relaxed);
}

/// `w · x` for an FP8 E4M3 weight `w` (I8 `[k, m]`, see the module docs) and F32 activations `x`
/// (`[k, n, ..]`, made contiguous here if it is not): F32 `[m, n, ..]`, times the weight's scale
/// `wscale` (F32 `[1]` per tensor or `[m]` per row) when given. `k` must be a multiple of 32.
pub fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let small = SmallBatch {
        gemv: SMALL_BATCH_GEMV.load(Ordering::Relaxed),
        sliced_k: SLICED_K.load(Ordering::Relaxed),
        sliced_k_tile: None,
    };
    mul_mat_small(ctx, w, x, wscale, small)
}

/// [`mul_mat`] with the choice [`set_small_batch_gemv`] makes given here: `small_batch_gemv`
/// runs 5-9 tokens on the GEMV (sm_80+; every token gets its one-token bits). The sliced-K GEMM
/// stays off.
pub fn mul_mat_with<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
    small_batch_gemv: bool,
) -> Tensor<'a> {
    let small = SmallBatch {
        gemv: small_batch_gemv,
        ..SmallBatch::default()
    };
    mul_mat_small(ctx, w, x, wscale, small)
}

/// The kernels that take a W8A16 matmul's small batches on GPUs with BF16 tensor cores
/// ([`mul_mat_small`]). The default (all off) is the tensor-core GEMM from 5 tokens on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SmallBatch {
    /// 5-9 tokens on the GEMV ([`set_small_batch_gemv`]).
    pub gemv: bool,
    /// 5-64 tokens the GEMV does not take on the sliced-K GEMM ([`set_sliced_k`]).
    pub sliced_k: bool,
    /// Tuning only: the sliced-K GEMM's tile `i` (`SLICED_KW` / `SLICED_TOK` in
    /// `src/cuda/fp8_mm.cu`) instead of the one it picks by the token count.
    #[doc(hidden)]
    pub sliced_k_tile: Option<u8>,
}

/// [`mul_mat`] with the small-batch kernels chosen here instead of by [`set_small_batch_gemv`]
/// and [`set_sliced_k`].
pub fn mul_mat_small<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
    small: SmallBatch,
) -> Tensor<'a> {
    assert_eq!(
        w.ty(),
        types::I8,
        "an FP8 weight is an I8 tensor of E4M3 bytes"
    );
    assert_eq!(x.ty(), types::F32, "FP8 matmul activations are F32");
    let [k, m, w2, w3] = w.ne();
    assert!(w2 == 1 && w3 == 1, "FP8 weights are 2-D");
    if let Some(s) = wscale {
        assert!(
            s.ty() == types::F32
                && s.is_contiguous()
                && (s.elements() == 1 || s.elements() as i64 == m),
            "an FP8 weight scale is F32, one value or one per row ({m}), not {:?}",
            s.ne()
        );
    }
    assert_eq!(
        x.ne()[0],
        k,
        "FP8 matmul: x rows of {} for k = {k}",
        x.ne()[0]
    );
    assert!(k % 32 == 0, "FP8 matmul needs k a multiple of 32, not {k}");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let [_, n1, n2, n3] = x.ne();
    let n = n1 * n2 * n3;
    let sliced = !w.is_host() && sliced_k_takes(small, n, k) && bf16_mma();
    // Large batches on the GPU, and the sliced-K GEMM's: the tensor-core GEMMs take BF16
    // activations, rounded once here (ggml's cast, round to nearest even) instead of by every
    // CTA that reads them, and once for all the matmuls of the same activations.
    let x = if sliced || (!w.is_host() && n > GEMM_BF16_MIN_N && k % 64 == 0 && bf16_mma()) {
        ctx.cast_shared(x, types::BF16)
    } else {
        x
    };
    let mut args = vec![w, x];
    args.extend(wscale);
    let gemv_max_n = if small.gemv {
        SMALL_BATCH_GEMV_MAX_N
    } else {
        0
    };
    let sliced_k = match small.sliced_k_tile {
        _ if !sliced => 0,
        None => 1,
        Some(t) => 2 + t as i32,
    };
    ctx.external_with(
        &FP8_MM,
        types::F32,
        [m, n1, n2, n3],
        &args,
        &[gemv_max_n, sliced_k],
    )
}

/// Whether the sliced-K GEMM takes a matmul of `n` tokens and row length `k` (on a GPU with
/// BF16 tensor cores): `small.sliced_k` asks for it, `n` is in its range and not the GEMV's,
/// and `k` is a multiple of its smallest K group.
pub(crate) fn sliced_k_takes(small: SmallBatch, n: i64, k: i64) -> bool {
    let gemv = small.gemv && (SMALL_BATCH_GEMV_MIN_N..=SMALL_BATCH_GEMV_MAX_N as i64).contains(&n);
    small.sliced_k
        && !gemv
        && (SLICED_K_MIN_N..=SLICED_K_MAX_N).contains(&n)
        && k % SLICED_K_ALIGN == 0
}

/// The fewest tokens [`set_small_batch_gemv`] moves to the GEMV (fewer take it by default).
const SMALL_BATCH_GEMV_MIN_N: i64 = 5;

/// The most tokens [`set_small_batch_gemv`] runs on the GEMV; it must match `ROWS_MAX_N` in
/// `src/cuda/fp8_mm.cu`.
const SMALL_BATCH_GEMV_MAX_N: i32 = 9;

/// The tokens the sliced-K GEMM takes ([`set_sliced_k`]) and the multiple of 4 warps x 64 its
/// rows' length must be; they must match `SLICED_MIN_N`, `SLICED_MAX_N` and `SLICED_K_ALIGN` in
/// `src/cuda/fp8_mm.cu`.
const SLICED_K_MIN_N: i64 = 5;
const SLICED_K_MAX_N: i64 = 64;
const SLICED_K_ALIGN: i64 = 256;

/// Whether the current GPU has BF16 tensor cores (sm_80+), for the BF16-activation GEMMs.
pub(crate) fn bf16_mma() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_fp8_mm_bf16_gemm() -> i32;
        }
        unsafe { legion_fp8_mm_bf16_gemm() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Above this many tokens a GPU matmul gets BF16 activations (`src/cuda/fp8_mm.cu`'s
/// large-batch GEMM); it must match `BF16_MIN_N` there.
const GEMM_BF16_MIN_N: i64 = 32;

/// The E4M3 (`fn`) byte as f32, exactly.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0x0f && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    sign * if exp == 0 {
        man / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// The E8M0 scale byte (OCP microscaling: `2^(b - 127)`, 255 is NaN) as f32, exactly.
pub fn e8m0_to_f32(b: u8) -> f32 {
    match b {
        0 => f32::from_bits(1 << 22),
        255 => f32::NAN,
        b => f32::from_bits((b as u32) << 23),
    }
}

/// The nearest E4M3 value of `x` (round to nearest even, saturating at ±448, as CUDA's
/// `__nv_cvt_float2_to_fp8x2(.., __NV_SATFINITE, __NV_E4M3)`); NaN stays NaN.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7e;
    }
    // Normal numbers have 3 mantissa bits from 2^-6 up; below, steps of 2^-9.
    let e = if a < 2f32.powi(-6) {
        -6
    } else {
        a.log2().floor() as i32
    };
    let step = 2f32.powi(e.max(-6) - 3);
    let q = a / step; // exact: a power-of-two division
    let r = q.round_ties_even() * step;
    // Encode r (the rounding may have carried into the next binade).
    if r == 0.0 {
        return sign;
    }
    let e = r.log2().floor() as i32;
    let code = if e < -6 {
        (r / 2f32.powi(-9)) as u8
    } else {
        let man = (r / 2f32.powi(e) * 8.0) as u8 - 8;
        (((e + 7) as u8) << 3) | man
    };
    sign | code
}

/// `e4m3(x / scale)` for every activation of `x` (F32 `[k, n, ..]`): the FP8 activations of a
/// W8A8 matmul with the checkpoint's static per-tensor `input_scale` (`scale`, F32 `[1]`).
/// Result: I8 `[k, n, ..]` of E4M3 bytes.
pub fn quantize<'a>(ctx: &'a Context, x: Tensor<'a>, scale: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "FP8 quantize takes F32 activations");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.external(&FP8_QUANT, types::I8, x.ne(), &[x, scale])
}

/// The W8A8 matmul `scale · (w · xq)` of an FP8 weight `w` (I8 `[k, m]`) and activations
/// quantized by [`quantize`] with input scale `scale`: F32 `[m, n, ..]`, times the weight's own
/// per-tensor scale `wscale` (F32 `[1]`) when given. `k` must be a multiple of 64.
pub fn mul_mat_a8<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    scale: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert_eq!(w.ty(), types::I8);
    assert_eq!(xq.ty(), types::I8);
    assert!(
        wscale.is_none_or(|s| s.elements() == 1),
        "W8A8 takes a per-tensor weight scale"
    );
    let [k, m, _, _] = w.ne();
    assert_eq!(xq.ne()[0], k);
    assert!(k % 64 == 0, "W8A8 matmul needs k a multiple of 64, not {k}");
    let [_, n1, n2, n3] = xq.ne();
    let mut args = vec![w, xq, scale];
    args.extend(wscale);
    ctx.external(&FP8_MM8, types::F32, [m, n1, n2, n3], &args)
}

/// One token's dynamic FP8 quantization (`--fp8-act a8-dynamic`): its scale `s = amax / 448`
/// (`amax = max |x|`, NaNs ignored; 0 for an all-zero token) and codes `e4m3(x · (1 / s))`
/// (0 when `s` is 0), in IEEE F32 with round to nearest: the CUDA kernel's bits.
pub fn quantize_token(x: &[f32]) -> (Vec<u8>, f32) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let s = if amax > 0.0 { amax / 448.0 } else { 0.0 };
    let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
    (x.iter().map(|&v| f32_to_e4m3(v * inv)).collect(), s)
}

/// Where [`quantize_dynamic`] puts `n` tokens of `k`: the codes `[k, n]` at 0, the F32 scales
/// `[n]` at the returned offset, and the total bytes.
pub fn dynamic_layout(k: i64, n: i64) -> (usize, usize) {
    let codes = (k * n) as usize;
    (codes, codes + 4 * n as usize)
}

/// W8A8 activations with one scale per token ([`quantize_token`] for each token of `x`, F32
/// `[k, n, ..]`, `k` a multiple of 4): I8 `[dynamic_layout(k, n).1]` holding the E4M3 codes
/// `[k, n]` and then the tokens' F32 scales. For [`mul_mat_a8_dynamic`].
pub fn quantize_dynamic<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "FP8 quantize takes F32 activations");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let [k, n1, n2, n3] = x.ne();
    assert!(k % 4 == 0, "dynamic FP8 quantize needs k a multiple of 4");
    let (_, total) = dynamic_layout(k, n1 * n2 * n3);
    ctx.external(&FP8_QUANT_DYN, types::I8, [total as i64, 1, 1, 1], &[x])
}

/// The W8A8 matmul of an FP8 weight `w` (I8 `[k, m]`) and activations `xq` quantized by
/// [`quantize_dynamic`] from an `x` of shape `x_ne`: F32 `[m, x_ne[1..]]`, element (j, t)
/// `(w_j · xq_t) · (s_t · wscale_j)` (`wscale` F32 `[1]` or `[m]`; `s_t` alone without one).
/// `k` must be a multiple of 64.
pub fn mul_mat_a8_dynamic<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    x_ne: [i64; 4],
    wscale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    a8_dynamic(ctx, w, xq, x_ne, wscale, None)
}

/// [`mul_mat_a8_dynamic`] plus `res` (F32, contiguous, the output's shape) added in the GEMM's
/// epilogue with one rounding, `fl(fl(w · x) + res)`: bit-identical to the matmul followed by
/// `ctx.add(.., res)`. Only the large-batch GEMM (sm_89+, more than 64 tokens) fuses it: other
/// shapes report an error, so callers check [`dynamic_add_fits`] first.
pub fn mul_mat_a8_dynamic_add<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    x_ne: [i64; 4],
    wscale: Option<Tensor<'a>>,
    res: Tensor<'a>,
) -> Tensor<'a> {
    a8_dynamic(ctx, w, xq, x_ne, wscale, Some(res))
}

/// Whether [`mul_mat_a8_dynamic_add`] runs for `n` tokens on this GPU (the large-batch FP8 GEMM).
pub(crate) fn dynamic_add_fits(n: i64) -> bool {
    n > 64 && fp8_mma()
}

fn a8_dynamic<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    xq: Tensor<'a>,
    x_ne: [i64; 4],
    wscale: Option<Tensor<'a>>,
    res: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert_eq!(w.ty(), types::I8);
    assert_eq!(xq.ty(), types::I8);
    let [k, m, _, _] = w.ne();
    let [xk, n1, n2, n3] = x_ne;
    assert_eq!(xk, k);
    assert!(k % 64 == 0, "W8A8 matmul needs k a multiple of 64, not {k}");
    assert_eq!(
        xq.elements(),
        dynamic_layout(k, n1 * n2 * n3).1,
        "not quantize_dynamic's output for {x_ne:?}"
    );
    if let Some(s) = wscale {
        assert!(
            s.ty() == types::F32
                && s.is_contiguous()
                && (s.elements() == 1 || s.elements() as i64 == m),
            "an FP8 weight scale is F32, one value or one per row ({m})"
        );
    }
    let mut args = vec![w, xq];
    args.extend(wscale);
    let mut res_at = 0;
    if let Some(r) = res {
        assert!(
            r.ty() == types::F32 && r.is_contiguous() && r.ne() == [m, n1, n2, n3],
            "an FP8 matmul residual is contiguous F32 of the output's shape"
        );
        res_at = args.len() as i32;
        args.push(r);
    }
    ctx.external_with(&FP8_MM8, types::F32, [m, n1, n2, n3], &args, &[1, res_at])
}

/// The [`Context::shared_node`] key of an activation's [`quantize_dynamic`] (or of the op that
/// [`crate::model::fp8_act`] gives it instead).
pub const DYNAMIC_KEY: &str = "fp8_quant_dyn";

/// `w · x` as W8A8 with per-token activation scales (`--fp8-act a8-dynamic`): `x` (F32 `[k, n,
/// ..]`) quantized by [`quantize_dynamic`], once per `x` in a graph for all the matmuls that
/// read it ([`Context::shared_node`]; or the op the model gave `x`, see
/// [`crate::model::fp8_act`]), then [`mul_mat_a8_dynamic`] with the weight's scale.
pub fn mul_mat_dynamic<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let xq = ctx.shared_node(x, DYNAMIC_KEY, || quantize_dynamic(ctx, x));
    mul_mat_a8_dynamic(ctx, w, xq, x.ne(), wscale)
}

/// [`mul_mat_dynamic`] with `res` added in the GEMM's epilogue ([`mul_mat_a8_dynamic_add`]).
pub fn mul_mat_dynamic_add<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    wscale: Option<Tensor<'a>>,
    res: Tensor<'a>,
) -> Tensor<'a> {
    let xq = ctx.shared_node(x, DYNAMIC_KEY, || quantize_dynamic(ctx, x));
    mul_mat_a8_dynamic_add(ctx, w, xq, x.ne(), wscale, res)
}

/// Whether the current GPU multiplies E4M3 on tensor cores (sm_89+), where the W8A8 GEMMs run.
pub(crate) fn fp8_mma() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn legion_fp8_mm_fp8_gemm() -> i32;
        }
        unsafe { legion_fp8_mm_fp8_gemm() != 0 }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

static FP8_QUANT_DYN: ExternalOp = ExternalOp {
    name: "fp8_quant_dyn",
    cpu: cpu_fp8_quant_dyn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_fp8_quant_dyn),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static FP8_QUANT: ExternalOp = ExternalOp {
    name: "fp8_quant",
    cpu: cpu_fp8_quant,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_fp8_quant),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static FP8_MM8: ExternalOp = ExternalOp {
    name: "fp8_mm8",
    cpu: cpu_fp8_mm8,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_fp8_mm8),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

unsafe fn cpu_fp8_quant(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `quantize`, host tensors, x contiguous.
    unsafe {
        let (x, s) = (&*dst.src[0], &*dst.src[1]);
        let n = (x.ne[0] * x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let inv = 1.0 / *(s.data as *const f32);
        let xs = std::slice::from_raw_parts(x.data as *const f32, n);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut u8, n);
        let chunk = n.div_ceil(nth);
        for i in ith * chunk..((ith + 1) * chunk).min(n) {
            out[i] = f32_to_e4m3(xs[i] * inv);
        }
    }
}

unsafe fn cpu_fp8_quant_dyn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `quantize_dynamic`, host tensors, x contiguous.
    unsafe {
        let x = &*dst.src[0];
        let k = x.ne[0] as usize;
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let xs = std::slice::from_raw_parts(x.data as *const f32, k * n);
        let (codes, _) = dynamic_layout(k as i64, n as i64);
        let out = dst.data as *mut u8;
        let scales = out.add(codes) as *mut f32;
        for t in (ith..n).step_by(nth) {
            let (q, s) = quantize_token(&xs[t * k..(t + 1) * k]);
            std::ptr::copy_nonoverlapping(q.as_ptr(), out.add(t * k), k);
            scales.add(t).write_unaligned(s);
        }
    }
}

unsafe fn cpu_fp8_mm8(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_a8` or `mul_mat_a8_dynamic`, host tensors, contiguous.
    unsafe {
        let [mode, res_at, ..] = crate::ggml::external_params(dst);
        let dynamic = mode == 1;
        let res = (res_at != 0).then(|| (*dst.src[res_at as usize]).data as *const f32);
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = (dst.ne[1] * dst.ne[2] * dst.ne[3]) as usize;
        let lut: [f32; 256] = std::array::from_fn(|b| e4m3_to_f32(b as u8));
        let wb = std::slice::from_raw_parts(w.data as *const u8, k * m);
        let xb = std::slice::from_raw_parts(x.data as *const u8, k * n);
        // Static: one input scale times the per-tensor weight scale for every element.
        // Dynamic: the token's scale times the row's weight scale.
        let (static_scale, token_scales) = if dynamic {
            let (codes, _) = dynamic_layout(k as i64, n as i64);
            let s = (x.data as *const u8).add(codes) as *const f32;
            let ts: Vec<f32> = (0..n).map(|t| s.add(t).read_unaligned()).collect();
            (0.0, ts)
        } else {
            let s = &*dst.src[2];
            (*(s.data as *const f32) * wscale(dst, 3), Vec::new())
        };
        let out = dst.data as *mut f32;
        for j in (ith..m).step_by(nth) {
            let row = &wb[j * k..(j + 1) * k];
            for t in 0..n {
                let col = &xb[t * k..(t + 1) * k];
                let d: f32 = row
                    .iter()
                    .zip(col)
                    .map(|(&a, &b)| lut[a as usize] * lut[b as usize])
                    .sum();
                let scale = if dynamic {
                    match dst.src[2].as_ref() {
                        Some(_) if res_at != 2 => token_scales[t] * wscale_of_row(dst, 2, j),
                        _ => token_scales[t],
                    }
                } else {
                    static_scale
                };
                let v = d * scale;
                *out.add(t * m + j) = match res {
                    Some(r) => v + *r.add(t * m + j),
                    None => v,
                };
            }
        }
    }
}

/// The CPU implementation: rows of `dst` split over threads; each element an F32 dot product
/// of exact weight values and the activations, in order of k.
unsafe fn cpu_fp8_mm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors, x and dst contiguous.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let lut: [f32; 256] = std::array::from_fn(|b| e4m3_to_f32(b as u8));
        let wb = std::slice::from_raw_parts(w.data as *const u8, k * m);
        // Activations are F32, or BF16 when the graph rounded them for the GPU GEMM.
        let widened: Vec<f32>;
        let xs: &[f32] = if x.type_ == types::BF16 {
            let raw = std::slice::from_raw_parts(x.data as *const u16, k * n);
            widened = raw
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect();
            &widened
        } else {
            std::slice::from_raw_parts(x.data as *const f32, k * n)
        };
        let out = dst.data as *mut f32;
        let mut row = vec![0f32; k];
        for j in (ith..m).step_by(nth) {
            let ws = wscale_of_row(dst, 2, j);
            for (r, &b) in row.iter_mut().zip(&wb[j * k..(j + 1) * k]) {
                *r = lut[b as usize];
            }
            for t in 0..n {
                let col = &xs[t * k..(t + 1) * k];
                let s: f32 = row.iter().zip(col).map(|(a, b)| a * b).sum();
                *out.add(t * m + j) = s * ws;
            }
        }
    }
}

/// The weight scale in `dst.src[i]`, or 1 when the op has none.
///
/// # Safety
/// `dst.src[i]`, when not null, is an F32 tensor in host memory.
unsafe fn wscale(dst: &ffi::ggml_tensor, i: usize) -> f32 {
    unsafe { dst.src[i].as_ref().map_or(1.0, |s| *(s.data as *const f32)) }
}

/// Output row `j`'s weight scale in `dst.src[i]` (one value for all rows, or one per row), or 1
/// when the op has none.
///
/// # Safety
/// As [`wscale`], with `j` below the scale's length when it has more than one value.
unsafe fn wscale_of_row(dst: &ffi::ggml_tensor, i: usize, j: usize) -> f32 {
    unsafe {
        dst.src[i].as_ref().map_or(1.0, |s| {
            let per_row = s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3] > 1;
            *(s.data as *const f32).add(if per_row { j } else { 0 })
        })
    }
}

/// The scale stride over output rows the CUDA kernels take: 1 for a scale per row, else 0.
#[cfg(feature = "cuda")]
fn wscale_stride(dst: &ffi::ggml_tensor, i: usize) -> i64 {
    // SAFETY: `dst.src[i]`, when not null, is a tensor of the graph.
    unsafe {
        dst.src[i]
            .as_ref()
            .map_or(0, |s| (s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3] > 1) as i64)
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LaunchParams {
    w: *const u8,
    x: *const f32,
    dst: *mut f32,
    k: i64,
    m: i64,
    n: i64,
    x_ld: i64,
    x_bf16: i32,
    wscale: *const f32,
    wscale_stride: i64,
    gemv_max_n: i64,
    sliced_k: i64,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_fp8_mm_launch(p: *const LaunchParams, stream: *mut std::os::raw::c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_fp8_mm(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat`, all tensors in device memory of the stream's device.
    let p = unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        LaunchParams {
            w: w.data as *const u8,
            x: x.data as *const f32,
            dst: dst.data as *mut f32,
            k: w.ne[0],
            m: w.ne[1],
            n: x.ne[1] * x.ne[2] * x.ne[3],
            x_ld: (x.nb[1] / x.nb[0]) as i64,
            x_bf16: (x.type_ == types::BF16) as i32,
            wscale: dst.src[2]
                .as_ref()
                .map_or(std::ptr::null(), |s| s.data as *const f32),
            wscale_stride: wscale_stride(dst, 2),
            gemv_max_n: crate::ggml::external_params(dst)[0] as i64,
            sliced_k: crate::ggml::external_params(dst)[1] as i64,
        }
    };
    match unsafe { legion_fp8_mm_launch(&p, stream) } {
        0 => Ok(()),
        1 => Err(format!("unsupported FP8 matmul shape: k = {}", p.k)),
        e => Err(format!("CUDA launch failed ({e})")),
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LaunchParams8 {
    w: *const u8,
    x: *const u8,
    scale: *const f32,
    dst: *mut f32,
    k: i64,
    m: i64,
    n: i64,
    wscale: *const f32,
    scale_stride: i64,
    wscale_stride: i64,
    res: *const f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_fp8_quant_dyn_launch(
        x: *const f32,
        x_ld: i64,
        out: *mut u8,
        scales: *mut f32,
        k: i64,
        n: i64,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
    fn legion_fp8_quant_launch(
        x: *const f32,
        x_ld: i64,
        out: *mut u8,
        k: i64,
        n: i64,
        scale: *const f32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
    fn legion_fp8_mm8_launch(p: *const LaunchParams8, stream: *mut std::os::raw::c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_fp8_quant(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `quantize`, device tensors.
    let r = unsafe {
        let (x, s) = (&*dst.src[0], &*dst.src[1]);
        legion_fp8_quant_launch(
            x.data as *const f32,
            x.ne[0],
            dst.data as *mut u8,
            x.ne[0],
            x.ne[1] * x.ne[2] * x.ne[3],
            s.data as *const f32,
            stream,
        )
    };
    match r {
        0 => Ok(()),
        1 => Err("FP8 quantize: odd row length".into()),
        e => Err(format!("CUDA launch failed ({e})")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_fp8_quant_dyn(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `quantize_dynamic`, device tensors, x contiguous.
    let r = unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0], x.ne[1] * x.ne[2] * x.ne[3]);
        let (codes, _) = dynamic_layout(k, n);
        let out = dst.data as *mut u8;
        let scales = out.add(codes) as *mut f32;
        // One pass where the row fits in registers (the same bits), else read twice.
        match crate::model::fp8_act::quantize_plain(
            x.data as *const f32,
            k,
            out,
            scales,
            k,
            n,
            stream,
        ) {
            Some(r) => r,
            None => legion_fp8_quant_dyn_launch(x.data as *const f32, k, out, scales, k, n, stream),
        }
    };
    match r {
        0 => Ok(()),
        1 => Err("dynamic FP8 quantize: row length not a multiple of 4".into()),
        e => Err(format!("CUDA launch failed ({e})")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_fp8_mm8(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    let [mode, res_at, ..] = crate::ggml::external_params(dst);
    let dynamic = mode == 1;
    // SAFETY: a node made by `mul_mat_a8` (src: w, xq, the input scale, the weight's
    // per-tensor scale or nothing) or `mul_mat_a8_dynamic` (src: w, the packed codes and token
    // scales, the weight's scale or nothing, then the residual at `res_at` or nothing), device
    // tensors.
    let p = unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let (k, n) = (w.ne[0], dst.ne[1] * dst.ne[2] * dst.ne[3]);
        let (scale, ws) = if dynamic {
            let (codes, _) = dynamic_layout(k, n);
            ((x.data as *const u8).add(codes) as *const f32, 2)
        } else {
            ((*dst.src[2]).data as *const f32, 3)
        };
        let has_ws = ws != res_at as usize && !dst.src[ws].is_null();
        LaunchParams8 {
            w: w.data as *const u8,
            x: x.data as *const u8,
            scale,
            dst: dst.data as *mut f32,
            k,
            m: w.ne[1],
            n,
            wscale: if has_ws {
                (*dst.src[ws]).data as *const f32
            } else {
                std::ptr::null()
            },
            scale_stride: dynamic as i64,
            wscale_stride: if dynamic && has_ws {
                wscale_stride(dst, ws)
            } else {
                0
            },
            res: if res_at != 0 {
                (*dst.src[res_at as usize]).data as *const f32
            } else {
                std::ptr::null()
            },
        }
    };
    match unsafe { legion_fp8_mm8_launch(&p, stream) } {
        0 => Ok(()),
        1 if !p.res.is_null() => Err(format!(
            "unsupported W8A8 matmul with a residual: n = {} (needs the sm_89+ GEMM, n > 64)",
            p.n
        )),
        1 => Err(format!("unsupported W8A8 matmul shape: k = {}", p.k)),
        e => Err(format!("CUDA launch failed ({e})")),
    }
}

#[cfg(test)]
mod tests;
