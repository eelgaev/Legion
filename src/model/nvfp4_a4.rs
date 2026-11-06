//! W4A4 NVFP4 matmuls on block-scaled FP4 tensor cores (`src/cuda/sm120/nvfp4_a4.cu`): the
//! activations are quantized to NVFP4 in the op (one F32 decode scale per token row, one UE4M3
//! scale per 16 elements, E2M1 codes), then multiplied with the NVFP4 weights as loaded by
//! `mma.sync ... kind::mxf4nvf4.block_scale` (F32 sums), times the weight's per-tensor (or
//! per-expert) scale and the row's decode scale. Same ops as [`super::nvfp4_mm`]:
//! - [`mul_mat_act`]: a dense weight `[k, m]`, with the epilogues of [`Act`];
//! - [`mul_mat_id_act`]: stacked experts, grouped by expert on the device;
//! - [`swiglu`]: the fused gate / up pair of a SwiGLU FFN, BF16 out;
//! - [`swiglu_id`]: the same for routed experts, grouped by expert, F32 out;
//! - [`mul_mat_add`] and [`swiglu_norm`]: a residual add in the dense GEMM's epilogue and an
//!   input RMS norm in the SwiGLU's quantization, each with the unfused ops' bits (#749).
//!
//! The grouped ops take one of two kernels ([`Grouped`]): a GEMM through shared-memory tiles
//! for prefill batches, and for batches of up to [`MV_MAX_TOKENS`] tokens (decode steps, a few
//! tokens per expert) a matvec bound by the weight stream, which copies each row's blocks in
//! long runs and multiplies them as stored, one MMA per 64 weights.
//!
//! Only on GPUs whose build has the FP4 kernels ([`device_supported`]: sm_120a / sm_121a
//! targets, e.g. the GB10); the policy that picks these over the W4A16 kernels is
//! [`super::core::Nvfp4A4`]. 4-bit activations shift the outputs: W4A16 checkpoints are not
//! calibrated for them.
//!
//! Quantization of a row `x` of `k`: `amax = max |x|`, decode scale `d = amax / 2688` (6 · 448:
//! the row's largest sub-block scale becomes 448); per 16 elements, scale `s = e4m3(amax16 ·
//! (448 / amax))` or one of the two codes either side of it, whichever quantizes the sub-block
//! with the smallest squared error (as ggml's activation quantizer), and codes `e2m1(x · (2688
//! / amax) / s)`, rounded to nearest even and saturated; all in IEEE F32 (the CUDA kernel's
//! `__fdiv_rn` / `__fmul_rn`): [`quantize`] is its bit-exact CPU reference. Codes are packed
//! as ggml packs NVFP4 weights (element `j` of a 16-element sub-block in the low, `j + 8` in
//! the high nibble of byte `j`).
//!
//! The CPU implementations quantize the activations the same way and sum the dequantized
//! products in F32: references for the CUDA kernels.

use super::nvfp4_mm::Act;
use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

static A4_MM: ExternalOp = ExternalOp {
    name: "nvfp4_a4_mm",
    cpu: cpu_mm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_MM_MULTI: ExternalOp = ExternalOp {
    name: "nvfp4_a4_mm_multi",
    cpu: cpu_mm_multi,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat_multi),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_MM_ID: ExternalOp = ExternalOp {
    name: "nvfp4_a4_mm_id",
    cpu: cpu_mm_id,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat_id),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_SWIGLU: ExternalOp = ExternalOp {
    name: "nvfp4_a4_swiglu",
    cpu: cpu_swiglu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::swiglu),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_MM_ID_W: ExternalOp = ExternalOp {
    name: "nvfp4_a4_mm_id_weighted",
    cpu: cpu_mm_id_weighted,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat_id_weighted),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_SWIGLU_ID: ExternalOp = ExternalOp {
    name: "nvfp4_a4_swiglu_id",
    cpu: cpu_swiglu_id,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::swiglu_id),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static A4_QUANT: ExternalOp = ExternalOp {
    name: "nvfp4_a4_quant",
    cpu: cpu_quant,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::quant),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Bytes per 64-weight NVFP4 block: four sub-block scales and 32 bytes of codes.
const BLOCK_BYTES: i64 = 36;

/// Whether this process's GPU runs the W4A4 kernels (probed once, on the current device).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SUPPORTED.get_or_init(|| {
            unsafe extern "C" {
                fn legion_nvfp4_a4_supported() -> i32;
            }
            // SAFETY: a device query (one probe kernel).
            unsafe { legion_nvfp4_a4_supported() != 0 }
        })
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether `w · x` can take these ops: an NVFP4 weight as [`super::nvfp4_mm::supports`]
/// takes, on a GPU with the FP4 kernels.
pub fn supports(w: Tensor<'_>) -> bool {
    super::nvfp4_mm::supports(w) && !w.is_host() && device_supported()
}

/// A quantized activation row (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Quantized {
    /// E2M1 codes, `k / 2` bytes in ggml's NVFP4 nibble order.
    pub codes: Vec<u8>,
    /// UE4M3 scales, one per 16 elements.
    pub scales: Vec<u8>,
    /// The row's decode scale.
    pub dec: f32,
}

/// `v` (>= 0) as E4M3, rounded to nearest even, saturated to 448 (`__NV_SATFINITE`).
pub fn e4m3(v: f32) -> u8 {
    if v.is_nan() || v >= 448.0 {
        return 0x7e;
    }
    // Units of the smallest subnormal (2^-9) below 2^-6, of 2^(e - 3) above (v is a normal
    // F32 there: e is its exponent).
    let e = if v < 2f32.powi(-6) {
        -6
    } else {
        ((v.to_bits() >> 23) & 0xff) as i32 - 127
    };
    let ulp = 2f32.powi(e - 3);
    let q = v / ulp;
    let r = q.round_ties_even();
    let mant = r as u32; // 8..=16 for normals, 0..=8 below
    let bits = if e == -6 && mant < 8 {
        mant
    } else if mant == 16 {
        ((e + 1 + 7) as u32) << 3
    } else {
        ((e + 7) as u32) << 3 | (mant - 8)
    };
    bits.min(0x7e) as u8
}

/// The value of an E4M3 (sign ignored, 0x7f as 0, as ggml decodes NVFP4 scales).
pub fn e4m3_value(b: u8) -> f32 {
    let b = b & 0x7f;
    if b == 0x7f {
        return 0.0;
    }
    let (e, m) = ((b >> 3) as i32, (b & 7) as f32);
    if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// `v` as an E2M1 code (sign in bit 3), rounded to nearest even, saturated to +-6.
pub fn e2m1(v: f32) -> u8 {
    let a = v.abs();
    // Midpoints between 0, 0.5, 1, 1.5, 2, 3, 4 and 6; at a tie the even code wins.
    let code = (a > 0.25) as u8
        + (a >= 0.75) as u8
        + (a > 1.25) as u8
        + (a >= 1.75) as u8
        + (a > 2.5) as u8
        + (a >= 3.5) as u8
        + (a > 5.0) as u8;
    code | if v.is_sign_negative() { 8 } else { 0 }
}

/// The value of an E2M1 code.
pub fn e2m1_value(c: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(c & 7) as usize];
    if c & 8 != 0 { -v } else { v }
}

/// Quantizes a row (`x.len()` a multiple of 16) exactly as the CUDA kernel does.
pub fn quantize(x: &[f32]) -> Quantized {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let live = amax > 0.0;
    let enc = if live { 2688.0 / amax } else { 0.0 };
    let enc16 = if live { 448.0 / amax } else { 0.0 };
    let mut codes = Vec::with_capacity(x.len() / 2);
    let mut scales = Vec::with_capacity(x.len() / 16);
    // A sub-block's squared error with scale code `c`, in the row's scale units, summed in the
    // order j, j + 8 for j = 0..7 (as the kernel sums it).
    let error = |sb: &[f32], c: u8| {
        let sf = e4m3_value(c);
        let qs = if sf > 0.0 { enc / sf } else { 0.0 };
        let mut err = 0f32;
        for j in 0..8 {
            for v in [sb[j], sb[j + 8]] {
                let d = v.abs() * enc - e2m1_value(e2m1(v * qs) & 7) * sf;
                err += d * d;
            }
        }
        err
    };
    for sb in x.as_chunks::<16>().0 {
        let bmax = sb.iter().fold(0f32, |m, v| m.max(v.abs()));
        // The scale rounded to nearest, then the two codes either side if one gives a smaller
        // squared error (as ggml's `quantize_mmq_nvfp4`).
        let first = e4m3(bmax * enc16);
        let (mut s, mut best) = (first, error(sb, first));
        for off in [-1, 1, -2, 2] {
            let c = first as i32 + off;
            if (0..=0x7e).contains(&c) {
                let e = error(sb, c as u8);
                if e < best {
                    (s, best) = (c as u8, e);
                }
            }
        }
        let sf = e4m3_value(s);
        let qs = if sf > 0.0 { enc / sf } else { 0.0 };
        scales.push(s);
        for j in 0..8 {
            codes.push(e2m1(sb[j] * qs) | e2m1(sb[j + 8] * qs) << 4);
        }
    }
    Quantized {
        codes,
        scales,
        dec: amax / 2688.0,
    }
}

/// The values a quantized row stands for, in element order: code · scale · decode scale.
pub fn dequantize(q: &Quantized) -> Vec<f32> {
    let mut out = vec![0f32; q.codes.len() * 2];
    for (sb, &s) in q.scales.iter().enumerate() {
        let sf = e4m3_value(s);
        for j in 0..8 {
            let b = q.codes[sb * 8 + j];
            out[sb * 16 + j] = e2m1_value(b & 15) * sf * q.dec;
            out[sb * 16 + j + 8] = e2m1_value(b >> 4) * sf * q.dec;
        }
    }
    out
}

/// Row strides of the quantized activations in the op's scratch: (code bytes, scale bytes),
/// k padded to a multiple of 256 (whole pipeline stages of the GEMM).
fn q_strides(k: i64) -> (i64, i64) {
    let sb = (k + 255) / 256 * 16;
    (sb * 8, sb)
}

/// Byte offsets in a node: the result (`out` bytes), then the codes, scales and decode scales
/// of `rows` activation rows, then `plan` I32s; each part 256-byte aligned. Returns the four
/// offsets after the result and the total.
fn layout(out: usize, rows: i64, k: i64, plan: usize) -> ([usize; 4], usize) {
    let up = |v: usize| v.div_ceil(256) * 256;
    let (q_ld, s_ld) = q_strides(k);
    let codes = up(out);
    let scales = codes + up((rows * q_ld) as usize);
    let dec = scales + up((rows * s_ld) as usize);
    let plan_at = dec + up(rows as usize * 4);
    ([codes, scales, dec, plan_at], plan_at + plan * 4)
}

/// Most CTAs a stream-K dense GEMM runs with (`legion_nvfp4_a4`'s `sk_max`).
const SK_MAX: usize = 64;
/// F32 accumulators one stream-K CTA hands on (`LEGION_NVFP4_A4_SK_ACC`).
const SK_ACC: usize = 256 * 64;

/// Scratch bytes for the stream-K split of a dense GEMM of `m` rows over `rows` tokens (the
/// accumulators handed on, then the flags and the ticket), or 0 when the split cannot apply:
/// it takes the 128 x 128 tile (over 64 tokens) with fewer tiles than [`SK_MAX`].
fn sk_bytes(m: i64, rows: i64) -> usize {
    let tiles = (m + 127) / 128 * ((rows + 127) / 128);
    if rows > 64 && (tiles as usize) < SK_MAX {
        SK_MAX * SK_ACC * 4 + (SK_MAX + 1) * 4
    } else {
        0
    }
}

/// A node of `total` bytes with `sk` stream-K scratch bytes after them: (the scratch's offset,
/// the new total).
fn with_sk(total: usize, sk: usize) -> (usize, usize) {
    if sk == 0 {
        return (total, total);
    }
    let at = total.div_ceil(256) * 256;
    (at, at + sk)
}

/// The ops' activation quantization alone (tests): `x` `[k, rows]` (F32 or BF16) to I8 bytes
/// laid out as [`quant_layout`] says.
#[doc(hidden)]
pub fn quantize_op<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let x = input(ctx, x);
    let [k, n1, n2, n3] = x.ne();
    let (_, total) = layout(0, n1 * n2 * n3, k, 0);
    ctx.external(&A4_QUANT, types::I8, [total as i64, 1, 1, 1], &[x])
}

/// Where [`quantize_op`] puts `rows` quantized rows of `k`: (code bytes per row, scale bytes
/// per row, offsets of the codes, the scales and the F32 decode scales).
#[doc(hidden)]
pub fn quant_layout(k: i64, rows: i64) -> (i64, i64, [usize; 3]) {
    let ([codes, scales, dec, _], _) = layout(0, rows, k, 0);
    let (q_ld, s_ld) = q_strides(k);
    (q_ld, s_ld, [codes, scales, dec])
}

/// `x` as the op takes it: contiguous F32 or BF16.
fn input<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    if x.ty() == types::F32 || x.ty() == types::BF16 {
        x
    } else {
        ctx.cast(x, types::F32)
    }
}

fn act_code(act: Act) -> i32 {
    (act == Act::ReluSqr) as i32
}

fn out_type(act: Act) -> ffi::ggml_type {
    if act == Act::ReluSqr {
        types::BF16
    } else {
        types::F32
    }
}

/// `scale · (w · x)` with 4-bit activations, through the epilogue `act`: `w` an NVFP4 `[k, m]`
/// (see [`supports`]), `x` `[k, n, ..]` (F32 or BF16), `scale` an optional F32 `[1]`; out
/// `[m, n, ..]` of `act`'s type.
pub fn mul_mat_act<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    act: Act,
) -> Tensor<'a> {
    dense(ctx, w, x, scale, act, None)
}

/// [`mul_mat`] plus `res` (F32, contiguous, the output's shape `[m, n, ..]`) added in the
/// GEMM's epilogue with one rounding, `fl(fl(scale · (w · x)) + res)`: bit-identical to the
/// matmul followed by `ctx.add(.., res)`.
pub fn mul_mat_add<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    res: Tensor<'a>,
) -> Tensor<'a> {
    dense(ctx, w, x, scale, Act::None, Some(res))
}

fn dense<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    act: Act,
    res: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let [k, m, w2, _] = w.ne();
    assert!(
        super::nvfp4_mm::supports(w) && w2 == 1,
        "unsupported NVFP4 weight {:?}",
        w.ne()
    );
    assert_eq!(
        x.ne()[0],
        k,
        "NVFP4 W4A4: x rows of {} for k = {k}",
        x.ne()[0]
    );
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [1, 1, 1, 1]),
            "a per-tensor scale"
        );
    }
    let x = input(ctx, x);
    let [_, n1, n2, n3] = x.ne();
    let rows = n1 * n2 * n3;
    let ty = out_type(act);
    let esize = crate::ggml::row_size(ty, 1);
    let (_, total) = layout((m * rows) as usize * esize, rows, k, 0);
    let (_, total) = with_sk(total, sk_bytes(m, rows));
    let mut args = vec![w, x];
    args.extend(scale);
    if let Some(r) = res {
        assert!(
            act == Act::None
                && r.ty() == types::F32
                && r.is_contiguous()
                && r.ne() == [m, n1, n2, n3],
            "an NVFP4 W4A4 residual is contiguous F32 of the output's shape"
        );
        args.push(r);
    }
    let node = ctx.external_with(
        &A4_MM,
        ty,
        [(total / esize) as i64, 1, 1, 1],
        &args,
        &[scale.is_some() as i32, act_code(act), res.is_some() as i32],
    );
    let e = esize;
    ctx.view_4d(
        node,
        [m, n1, n2, n3],
        [
            m as usize * e,
            (m * n1) as usize * e,
            (m * n1 * n2) as usize * e,
        ],
        0,
    )
}

/// [`mul_mat_act`] without an epilogue: F32 out.
pub fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    mul_mat_act(ctx, w, x, scale, Act::None)
}

/// How many weights [`mul_mat_multi`] takes.
pub const MULTI_MAX: usize = 3;

/// [`mul_mat`] of up to [`MULTI_MAX`] weights `ws` (each with its optional per-tensor scale)
/// over one input as one op: the activations are quantized once, and with `norm` (`(nw,
/// eps)`) the input is `mul(rms_norm(x, eps), nw)` normed in that quantization as in
/// [`swiglu_norm`] ([`rms_norm_fits`] on CUDA). F32 outs `[m_i, n, ..]`, each the bits of its
/// own [`mul_mat`] (attention's q, k and v projections: one quantization instead of three).
pub fn mul_mat_multi<'a>(
    ctx: &'a Context,
    ws: &[(Tensor<'a>, Option<Tensor<'a>>)],
    x: Tensor<'a>,
    norm: Option<(Tensor<'a>, f32)>,
) -> Vec<Tensor<'a>> {
    assert!(
        !ws.is_empty() && ws.len() <= MULTI_MAX,
        "{} weights for one NVFP4 W4A4 input",
        ws.len()
    );
    let k = x.ne()[0];
    for &(w, s) in ws {
        assert!(
            super::nvfp4_mm::supports(w) && w.ne()[2] == 1 && w.ne()[0] == k,
            "unsupported NVFP4 weight {:?} for k = {k}",
            w.ne()
        );
        if let Some(s) = s {
            assert_eq!(
                (s.ty(), s.ne()),
                (types::F32, [1, 1, 1, 1]),
                "a per-tensor scale"
            );
        }
    }
    let x = input(ctx, x);
    let [_, n1, n2, n3] = x.ne();
    let rows = n1 * n2 * n3;
    let m_sum: i64 = ws.iter().map(|(w, _)| w.ne()[1]).sum();
    let (_, total) = layout((m_sum * rows) as usize * 4, rows, k, 0);
    let m_min = ws.iter().map(|(w, _)| w.ne()[1]).min().unwrap_or(0);
    let (_, total) = with_sk(total, sk_bytes(m_min, rows));
    let mut args: Vec<Tensor<'a>> = ws.iter().map(|&(w, _)| w).collect();
    args.push(x);
    let mut mask = 0;
    for (i, &(_, s)) in ws.iter().enumerate() {
        if let Some(s) = s {
            args.push(s);
            mask |= 1 << i;
        }
    }
    let mut params = vec![ws.len() as i32, mask];
    if let Some((nw, eps)) = norm {
        assert!(
            x.ty() == types::F32 && nw.ty() == types::F32 && nw.ne() == [k, 1, 1, 1],
            "an NVFP4 W4A4 norm takes F32 x and an F32 [k] weight"
        );
        args.push(nw);
        params.extend([1, eps.to_bits() as i32]);
    }
    let node = ctx.external_with(
        &A4_MM_MULTI,
        types::F32,
        [(total / 4) as i64, 1, 1, 1],
        &args,
        &params,
    );
    let mut off = 0usize;
    ws.iter()
        .map(|&(w, _)| {
            let m = w.ne()[1];
            let out = ctx.view_4d(
                node,
                [m, n1, n2, n3],
                [
                    m as usize * 4,
                    (m * n1) as usize * 4,
                    (m * n1 * n2) as usize * 4,
                ],
                off,
            );
            off += (m * rows) as usize * 4;
            out
        })
        .collect()
}

/// The CPU implementation of [`mul_mat_multi`].
unsafe fn cpu_mm_multi(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [count, mask, has_norm, eps, ..] = external_params(dst);
    let count = count as usize;
    // SAFETY: a node made by `mul_mat_multi`: host tensors, x contiguous.
    unsafe {
        let x = &*dst.src[count];
        let k = x.ne[0] as usize;
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let n_scales = (mask as u32).count_ones() as usize;
        let xq = if has_norm != 0 {
            let nw = values(&*dst.src[count + 1 + n_scales], k);
            let eps = f32::from_bits(eps as u32);
            values(x, k * n)
                .chunks_exact(k)
                .map(|r| dequantize(&quantize(&rms_norm_row(r, &nw, eps))))
                .collect()
        } else {
            quantized_rows(x, k, n)
        };
        let mut out = dst.data as *mut f32;
        let mut si = count + 1;
        for i in 0..count {
            let w = &*dst.src[i];
            let scale = if mask & (1 << i) != 0 {
                si += 1;
                *((*dst.src[si - 1]).data as *const f32)
            } else {
                1.0
            };
            let m = w.ne[1] as usize;
            for j in (ith..m).step_by(nth) {
                let row = weight_row(w, 0, j);
                for (t, xr) in xq.iter().enumerate() {
                    *out.add(t * m + j) = dot(&row, xr) * scale;
                }
            }
            out = out.add(m * n);
        }
    }
}

/// The expert matmul with 4-bit activations, as [`super::nvfp4_mm::mul_mat_id_act`]: experts
/// `w` `[k, m, n_expert]`, `x` `[k, n_x, n_tok]` (`n_x` 1 or `n_used`), `ids` I32 `[n_used,
/// n_tok]`, per-expert F32 `scale` `[n_expert]`; out `[m, n_used, n_tok]` of `act`'s type.
pub fn mul_mat_id_act<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    act: Act,
) -> Tensor<'a> {
    assert!(
        super::nvfp4_mm::supports(w),
        "unsupported NVFP4 experts {:?}",
        w.ne()
    );
    let [k, m, n_expert, _] = w.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, n_x, xt, _] = x.ne();
    assert_eq!(xk, k, "NVFP4 W4A4 mul_mat_id: x rows of {xk} for k = {k}");
    assert!(
        (n_x == 1 || n_x == n_used) && xt == n_tok,
        "NVFP4 W4A4 mul_mat_id: x {:?} for ids {:?}",
        x.ne(),
        ids.ne()
    );
    assert!(
        n_expert <= 1024,
        "NVFP4 W4A4 mul_mat_id: {n_expert} experts"
    );
    assert_eq!(ids.ty(), types::I32);
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [n_expert, 1, 1, 1]),
            "a per-expert scale"
        );
    }
    let x = input(ctx, x);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let ty = out_type(act);
    let esize = crate::ggml::row_size(ty, 1);
    let plan = super::nvfp4_mm::plan_len(n_expert, n_used * n_tok);
    let (_, total) = layout((m * n_used * n_tok) as usize * esize, n_x * n_tok, k, plan);
    let mut args = vec![w, x, ids];
    args.extend(scale);
    let node = ctx.external_with(
        &A4_MM_ID,
        ty,
        [(total / esize) as i64, 1, 1, 1],
        &args,
        &[scale.is_some() as i32, act_code(act)],
    );
    ctx.view_3d(
        node,
        [m, n_used, n_tok],
        [m as usize * esize, (m * n_used) as usize * esize],
        0,
    )
}

/// [`mul_mat_id_act`] without an epilogue: F32 out.
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    mul_mat_id_act(ctx, w, x, ids, scale, Act::None)
}

/// Where [`mul_mat_id_weighted`]'s node keeps the BF16 expert outputs: after the result
/// (`out` bytes), the quantized activations of `rows` rows of `k` and the plan, 256-aligned.
fn weighted_mid(out: usize, rows: i64, k: i64, n_expert: i64, pairs: i64) -> usize {
    let (_, end) = layout(out, rows, k, super::nvfp4_mm::plan_len(n_expert, pairs));
    end.div_ceil(256) * 256
}

/// A MoE layer's routed experts, weighted and summed: `Σ_s weights[s, t] · scale[e] · (w_e ·
/// x)` over the `n_used` slots `s` of each token `t` (expert `e = ids[s, t]`), in slot order,
/// with 4-bit activations. `w`, `x`, `ids`, `scale` as [`mul_mat_id_act`]; `weights` F32 with
/// `n_used · n_tok` elements (`[1, n_used, n_tok]` as the router makes them). F32 `[m,
/// n_tok]`. The weighted expert outputs are rounded to BF16 before the sum (as vLLM's MoE
/// sums them), which halves what the GEMM writes and the sum reads.
pub fn mul_mat_id_weighted<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    weights: Tensor<'a>,
) -> Tensor<'a> {
    assert!(
        super::nvfp4_mm::supports(w),
        "unsupported NVFP4 experts {:?}",
        w.ne()
    );
    let [k, m, n_expert, _] = w.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, n_x, xt, _] = x.ne();
    assert_eq!(xk, k, "NVFP4 W4A4 experts: x rows of {xk} for k = {k}");
    assert!((n_x == 1 || n_x == n_used) && xt == n_tok);
    assert!(n_expert <= 1024 && ids.ty() == types::I32);
    assert_eq!(weights.ty(), types::F32);
    assert_eq!(
        weights.ne().iter().product::<i64>(),
        n_used * n_tok,
        "routing weights"
    );
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [n_expert, 1, 1, 1]),
            "a per-expert scale"
        );
    }
    let x = input(ctx, x);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let weights = if weights.is_contiguous() {
        weights
    } else {
        ctx.cont(weights)
    };
    let out = (m * n_tok) as usize * 4;
    let total = weighted_mid(out, n_x * n_tok, k, n_expert, n_used * n_tok)
        + (m * n_used * n_tok) as usize * 2;
    let mut args = vec![w, x, ids, weights];
    args.extend(scale);
    let node = ctx.external_with(
        &A4_MM_ID_W,
        types::F32,
        [total.div_ceil(4) as i64, 1, 1, 1],
        &args,
        &[scale.is_some() as i32],
    );
    ctx.view_2d(node, [m, n_tok], m as usize * 4, 0)
}

/// `silu(sg · (wg · x)) ⊙ (su · (wu · x))` with 4-bit activations, as
/// [`super::nvfp4_mm::swiglu`]: BF16 `[m, n, ..]`.
pub fn swiglu<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_with(ctx, wg, wu, x, sg, su, None, false)
}

/// [`swiglu`] with the tanh GELU of the gate (as ggml's `geglu_split`) for silu: Gemma 4's
/// GeGLU FFN, BF16 `[m, n, ..]`.
pub fn geglu<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_with(ctx, wg, wu, x, sg, su, None, true)
}

/// [`geglu`] of `mul(rms_norm(x, eps), nw)`, the norm in the quantization as [`swiglu_norm`].
#[allow(clippy::too_many_arguments)]
pub fn geglu_norm<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    nw: Tensor<'a>,
    eps: f32,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_with(ctx, wg, wu, x, sg, su, Some((nw, eps)), true)
}

/// [`swiglu`] of `mul(rms_norm(x, eps), nw)` (ggml's ops; `x` F32 `[k, n, ..]`, `nw` F32 `[k]`)
/// with the norm in the activations' quantization: one kernel instead of ggml's norm and the
/// quantization of its output, the same bits ([`rms_norm_fits`] on CUDA).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_norm<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    nw: Tensor<'a>,
    eps: f32,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_with(ctx, wg, wu, x, sg, su, Some((nw, eps)), false)
}

/// The longest row [`swiglu_norm`]'s CUDA kernel normalizes (its rows sit in shared memory).
pub const RMS_MAX_K: i64 = 8192;

/// Whether [`swiglu_norm`] takes `x` normed by `nw` on this GPU: F32, contiguous, rows of `k`
/// a multiple of 64 up to [`RMS_MAX_K`], `nw` F32 `[k]` in device memory.
pub fn rms_norm_fits(x: Tensor<'_>, nw: Tensor<'_>) -> bool {
    let k = x.ne()[0];
    device_supported()
        && x.ty() == types::F32
        && x.is_contiguous()
        && k % 64 == 0
        && k <= RMS_MAX_K
        && nw.ty() == types::F32
        && nw.is_contiguous()
        && nw.ne() == [k, 1, 1, 1]
        && !nw.is_host()
}

/// Params: has_sg, has_su, has_norm, the norm's eps bits, gelu.
#[allow(clippy::too_many_arguments)]
fn swiglu_with<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
    norm: Option<(Tensor<'a>, f32)>,
    gelu: bool,
) -> Tensor<'a> {
    assert!(
        super::nvfp4_mm::supports(wg)
            && super::nvfp4_mm::supports(wu)
            && wg.ne() == wu.ne()
            && wg.nb() == wu.nb()
            && wg.ne()[2] == 1,
        "unsupported NVFP4 gate / up {:?} {:?}",
        wg.ne(),
        wu.ne()
    );
    let [k, m, _, _] = wg.ne();
    assert_eq!(
        x.ne()[0],
        k,
        "NVFP4 W4A4 SwiGLU: x rows of {} for k = {k}",
        x.ne()[0]
    );
    for s in [sg, su].into_iter().flatten() {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [1, 1, 1, 1]),
            "a per-tensor scale"
        );
    }
    let x = input(ctx, x);
    let [_, n1, n2, n3] = x.ne();
    let rows = n1 * n2 * n3;
    let (_, total) = layout((m * rows) as usize * 2, rows, k, 0);
    let mut args = vec![wg, wu, x];
    args.extend(sg);
    args.extend(su);
    let mut params = vec![sg.is_some() as i32, su.is_some() as i32];
    if let Some((nw, eps)) = norm {
        assert!(
            x.ty() == types::F32 && nw.ty() == types::F32 && nw.ne() == [k, 1, 1, 1],
            "an NVFP4 W4A4 SwiGLU norm takes F32 x and an F32 [k] weight"
        );
        args.push(nw);
        params.extend([1, eps.to_bits() as i32]);
    } else {
        params.extend([0, 0]);
    }
    params.push(gelu as i32);
    let node = ctx.external_with(
        &A4_SWIGLU,
        types::BF16,
        [(total / 2) as i64, 1, 1, 1],
        &args,
        &params,
    );
    ctx.view_4d(
        node,
        [m, n1, n2, n3],
        [
            m as usize * 2,
            (m * n1) as usize * 2,
            (m * n1 * n2) as usize * 2,
        ],
        0,
    )
}

/// [`swiglu`] for routed experts, with 4-bit activations: `silu(sg[e] · (wg[e] · x)) ⊙ (su[e] ·
/// (wu[e] · x))` for each token and its experts `e` from `ids` (I32 `[n_used, n_tok]`), NVFP4
/// `wg` and `wu` of the same shape `[k, m, n_expert]`, `x` `[k, n_x, n_tok]` (`n_x` 1 or
/// `n_used`) and optional F32 `[n_expert]` per-expert scales: F32 `[m, n_used, n_tok]`, the down
/// [`mul_mat_id`]'s input. On CUDA one grouped GEMM that quantizes `x` once; the same bits as
/// the two [`mul_mat_id`]s, ggml-cuda's `silu` and its `mul` (each output is the same F32 sum
/// and the same epilogue arithmetic).
pub fn swiglu_id<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        super::nvfp4_mm::supports(wg)
            && super::nvfp4_mm::supports(wu)
            && wg.ne() == wu.ne()
            && wg.nb() == wu.nb(),
        "unsupported NVFP4 gate / up experts {:?} {:?}",
        wg.ne(),
        wu.ne()
    );
    let [k, m, n_expert, _] = wg.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, n_x, xt, _] = x.ne();
    assert_eq!(
        xk, k,
        "NVFP4 W4A4 SwiGLU experts: x rows of {xk} for k = {k}"
    );
    assert!(
        (n_x == 1 || n_x == n_used) && xt == n_tok,
        "NVFP4 W4A4 SwiGLU experts: x {:?} for ids {:?}",
        x.ne(),
        ids.ne()
    );
    assert!(
        n_expert <= 1024,
        "NVFP4 W4A4 SwiGLU experts: {n_expert} experts"
    );
    assert_eq!(ids.ty(), types::I32);
    for s in [sg, su].into_iter().flatten() {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [n_expert, 1, 1, 1]),
            "a per-expert scale"
        );
    }
    let x = input(ctx, x);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let plan = super::nvfp4_mm::plan_len(n_expert, n_used * n_tok);
    let (_, total) = layout((m * n_used * n_tok) as usize * 4, n_x * n_tok, k, plan);
    let mut args = vec![wg, wu, x, ids];
    args.extend(sg);
    args.extend(su);
    let node = ctx.external_with(
        &A4_SWIGLU_ID,
        types::F32,
        [total.div_ceil(4) as i64, 1, 1, 1],
        &args,
        &[sg.is_some() as i32, su.is_some() as i32],
    );
    ctx.view_3d(
        node,
        [m, n_used, n_tok],
        [m as usize * 4, (m * n_used) as usize * 4],
        0,
    )
}

/// Rows of a contiguous F32 or BF16 host tensor as F32, `len` elements.
///
/// # Safety
/// `x` is such a tensor with at least `len` elements.
unsafe fn values(x: &ffi::ggml_tensor, len: usize) -> Vec<f32> {
    // SAFETY: as documented.
    unsafe {
        if x.type_ == types::BF16 {
            std::slice::from_raw_parts(x.data as *const u16, len)
                .iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect()
        } else {
            std::slice::from_raw_parts(x.data as *const f32, len).to_vec()
        }
    }
}

/// Every row of `x` (`rows` of `k`) quantized and dequantized.
///
/// # Safety
/// As [`values`].
unsafe fn quantized_rows(x: &ffi::ggml_tensor, k: usize, rows: usize) -> Vec<Vec<f32>> {
    // SAFETY: as documented.
    let xs = unsafe { values(x, k * rows) };
    xs.chunks_exact(k)
        .map(|r| dequantize(&quantize(r)))
        .collect()
}

/// `w`'s row `j` of expert `e`, dequantized.
///
/// # Safety
/// `w` is an NVFP4 host tensor with that row.
unsafe fn weight_row(w: &ffi::ggml_tensor, e: usize, j: usize) -> Vec<f32> {
    let k = w.ne[0] as usize;
    // SAFETY: as documented.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (w.data as *const u8).add(e * w.nb[2] + j * w.nb[1]),
            k / 64 * BLOCK_BYTES as usize,
        )
    };
    crate::ggml::dequantize(types::NVFP4, bytes, k)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// `v` rounded to the nearest BF16 (ties to even), as `__float2bfloat16_rn`.
fn bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Stores `v` through the epilogue `act` (1: relu² as BF16) at element `i` of `out`.
///
/// # Safety
/// `out` holds `i + 1` elements of the epilogue's output type.
unsafe fn store(out: *mut u8, i: usize, v: f32, act: i32) {
    // SAFETY: as documented.
    unsafe {
        if act == 1 {
            *(out as *mut u16).add(i) = bf16_bits(v.max(0.0) * v.max(0.0));
        } else {
            *(out as *mut f32).add(i) = v;
        }
    }
}

/// The CPU implementation of [`quantize_op`] (one thread; padding stays zero).
unsafe fn cpu_quant(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `quantize_op`: host tensors, x contiguous.
    unsafe {
        let x = &*dst.src[0];
        let k = x.ne[0];
        let rows = x.ne[1] * x.ne[2] * x.ne[3];
        let (q_ld, s_ld, [codes, scales, dec]) = quant_layout(k, rows);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut u8, dst.ne[0] as usize);
        out.fill(0);
        let xs = values(x, (k * rows) as usize);
        for (r, row) in xs.chunks_exact(k as usize).enumerate() {
            let q = quantize(row);
            let c = codes + r * q_ld as usize;
            out[c..c + q.codes.len()].copy_from_slice(&q.codes);
            let s = scales + r * s_ld as usize;
            out[s..s + q.scales.len()].copy_from_slice(&q.scales);
            out[dec + 4 * r..dec + 4 * r + 4].copy_from_slice(&q.dec.to_ne_bytes());
        }
    }
}

/// The CPU implementation of [`mul_mat_act`]: every thread quantizes the activations (the op
/// has no barrier between the two phases), then takes every `nth` weight row.
unsafe fn cpu_mm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [has_scale, act, has_res, ..] = external_params(dst);
    // SAFETY: a node made by `mul_mat_act` or `mul_mat_add`: host tensors, x and the residual
    // contiguous.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let scale = if has_scale != 0 {
            *((*dst.src[2]).data as *const f32)
        } else {
            1.0
        };
        let res = (has_res != 0).then(|| (*dst.src[2 + has_scale as usize]).data as *const f32);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let xq = quantized_rows(x, k, n);
        let out = dst.data as *mut u8;
        for j in (ith..m).step_by(nth) {
            let row = weight_row(w, 0, j);
            for (t, xr) in xq.iter().enumerate() {
                let v = dot(&row, xr) * scale;
                match res {
                    Some(r) => *(out as *mut f32).add(t * m + j) = v + *r.add(t * m + j),
                    None => store(out, t * m + j, v, act),
                }
            }
        }
    }
}

/// `mul(rms_norm(x, eps), w)` of one row as ggml's CPU ops compute it (F64 sum of the F32
/// squares).
fn rms_norm_row(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect()
}

/// The CPU implementation of [`mul_mat_id_act`].
unsafe fn cpu_mm_id(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [has_scale, act, ..] = external_params(dst);
    // SAFETY: a node made by `mul_mat_id_act`: contiguous host tensors as checked there.
    unsafe {
        let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let scale = (has_scale != 0).then(|| (*dst.src[3]).data as *const f32);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let (n_used, n_tok) = (ids.ne[0] as usize, ids.ne[1] as usize);
        let n_x = x.ne[1] as usize;
        let xq = quantized_rows(x, k, n_x * n_tok);
        let out = dst.data as *mut u8;
        for j in (ith..m).step_by(nth) {
            for t in 0..n_tok {
                for s in 0..n_used {
                    let e = *(ids.data as *const i32).add(t * n_used + s) as usize;
                    let sc = scale.map_or(1.0, |p| *p.add(e));
                    let xr = &xq[t * n_x + if n_x == 1 { 0 } else { s }];
                    let v = dot(&weight_row(w, e, j), xr) * sc;
                    store(out, (t * n_used + s) * m + j, v, act);
                }
            }
        }
    }
}

/// The CPU implementation of [`mul_mat_id_weighted`].
unsafe fn cpu_mm_id_weighted(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [has_scale, ..] = external_params(dst);
    // SAFETY: a node made by `mul_mat_id_weighted`: contiguous host tensors as checked there.
    unsafe {
        let (w, x, ids, weights) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let scale = (has_scale != 0).then(|| (*dst.src[4]).data as *const f32);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let (n_used, n_tok) = (ids.ne[0] as usize, ids.ne[1] as usize);
        let n_x = x.ne[1] as usize;
        let xq = quantized_rows(x, k, n_x * n_tok);
        let out = dst.data as *mut f32;
        let pw = weights.data as *const f32;
        for j in (ith..m).step_by(nth) {
            for t in 0..n_tok {
                let mut sum = 0f32;
                for s in 0..n_used {
                    let q = t * n_used + s;
                    let e = *(ids.data as *const i32).add(q) as usize;
                    let sc = scale.map_or(1.0, |p| *p.add(e));
                    let xr = &xq[t * n_x + if n_x == 1 { 0 } else { s }];
                    let v = dot(&weight_row(w, e, j), xr) * sc * *pw.add(q);
                    sum += f32::from_bits((bf16_bits(v) as u32) << 16);
                }
                *out.add(t * m + j) = sum;
            }
        }
    }
}

/// The CPU implementation of [`swiglu`].
unsafe fn cpu_swiglu(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [has_g, has_u, has_norm, eps, gelu, ..] = external_params(dst);
    // SAFETY: a node made by `swiglu_with`: host tensors, x contiguous.
    unsafe {
        let (wg, wu, x) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let value = |on: i32, i: usize| {
            if on != 0 {
                *((*dst.src[i]).data as *const f32)
            } else {
                1.0
            }
        };
        let sg = value(has_g, 3);
        let su = value(has_u, 3 + has_g as usize);
        let (k, m) = (wg.ne[0] as usize, wg.ne[1] as usize);
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let xq = if has_norm != 0 {
            let nw = values(&*dst.src[3 + (has_g + has_u) as usize], k);
            let eps = f32::from_bits(eps as u32);
            values(x, k * n)
                .chunks_exact(k)
                .map(|r| dequantize(&quantize(&rms_norm_row(r, &nw, eps))))
                .collect()
        } else {
            quantized_rows(x, k, n)
        };
        let out = dst.data as *mut u16;
        for j in (ith..m).step_by(nth) {
            let (rg, ru) = (weight_row(wg, 0, j), weight_row(wu, 0, j));
            for (t, xr) in xq.iter().enumerate() {
                let g = dot(&rg, xr) * sg;
                let a = if gelu != 0 {
                    super::nvfp4_mm::gelu_tanh(g)
                } else {
                    g / (1.0 + (-g).exp())
                };
                *out.add(t * m + j) = bf16_bits(a * (dot(&ru, xr) * su));
            }
        }
    }
}

/// The CPU implementation of [`swiglu_id`]: per (token, slot), the gate and up sums as
/// [`cpu_mm_id`] makes them, then `silu(gate) · up` in F32.
unsafe fn cpu_swiglu_id(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let [has_g, has_u, ..] = external_params(dst);
    // SAFETY: a node made by `swiglu_id`: contiguous host tensors as checked there.
    unsafe {
        let (wg, wu, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let sg = (has_g != 0).then(|| (*dst.src[4]).data as *const f32);
        let su = (has_u != 0).then(|| (*dst.src[4 + has_g as usize]).data as *const f32);
        let (k, m) = (wg.ne[0] as usize, wg.ne[1] as usize);
        let (n_used, n_tok) = (ids.ne[0] as usize, ids.ne[1] as usize);
        let n_x = x.ne[1] as usize;
        let xq = quantized_rows(x, k, n_x * n_tok);
        let out = dst.data as *mut f32;
        for j in (ith..m).step_by(nth) {
            for t in 0..n_tok {
                for s in 0..n_used {
                    let q = t * n_used + s;
                    let e = *(ids.data as *const i32).add(q) as usize;
                    let xr = &xq[t * n_x + if n_x == 1 { 0 } else { s }];
                    let g = dot(&weight_row(wg, e, j), xr) * sg.map_or(1.0, |p| *p.add(e));
                    let u = dot(&weight_row(wu, e, j), xr) * su.map_or(1.0, |p| *p.add(e));
                    *out.add(q * m + j) = g / (1.0 + (-g).exp()) * u;
                }
            }
        }
    }
}

/// Grouped matmuls ([`mul_mat_id_act`], [`swiglu_id`], [`mul_mat_id_weighted`]) of up to this
/// many tokens take the decode kernel ([`Grouped::Mv`]), larger ones the GEMM. A batch this
/// small has a few tokens per expert (Qwen3.8-Flash-Next, 64 tokens: 640 pairs over 512
/// experts), so the matmul is bound by the weight stream.
pub const MV_MAX_TOKENS: i64 = 64;

/// The kernel of a grouped matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grouped {
    /// The tiled GEMM (prefill).
    Gemm,
    /// The decode kernel: 16 · `rt` weight rows (`rt` 1 or 2; the SwiGLU: 2 or 4, `rt / 2` row
    /// tiles of the gate and the same rows of the up weight) and 8, 16 or 32 tokens of one
    /// expert per CTA, `kw` (2, 4 or 8) warps splitting k. Its sums depend on k and `kw` only,
    /// so the grouped SwiGLU keeps the bits of its two matmuls and the activation.
    Mv { rt: i32, kw: i32 },
}

/// The decode kernel's shape for experts with rows of `k` (`glu`: the grouped SwiGLU), as
/// measured on the GB10 (`tests/nvfp4_a4.rs`, `cuda_decode_timings`, Qwen3.8-Flash-Next's
/// experts): rows of up to 16 blocks (its down matmul, k 640) on 2 warps, longer rows (its
/// gate and up, k 2560) on 8. `kw` depends on `k` alone (see [`Grouped::Mv`]).
pub fn mv_shape(k: i64, glu: bool) -> Grouped {
    let kw = if k <= 16 * 64 { 2 } else { 8 };
    Grouped::Mv {
        rt: if glu { 2 } else { 1 },
        kw,
    }
}

/// The decode kernel's token tile for `per` tokens per expert: 8 · `nt` tokens, `nt` 1, 2 or 4.
pub fn mv_tokens_tile(per: i64) -> i32 {
    match per {
        ..=8 => 1,
        9..=16 => 2,
        _ => 4,
    }
}

/// Runs every grouped matmul from now on on `kernel` (the decode kernel where its weights
/// allow it), or the default choice for `None`: for tests and tuning. A token tile forced by
/// [`force_tokens_tile`] also forces the GEMM.
#[doc(hidden)]
pub fn force_grouped(kernel: Option<Grouped>) {
    #[cfg(feature = "cuda")]
    cuda::FORCED_GROUPED.store(
        match kernel {
            None => 0,
            Some(Grouped::Gemm) => 1,
            Some(Grouped::Mv { rt, kw }) => 2 | (rt as u64) << 8 | (kw as u64) << 16,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
    #[cfg(not(feature = "cuda"))]
    let _ = kernel;
}

/// Token tile of the GEMM for `n` tokens (per expert when grouped): 16 · `nt` tokens, `nt` 2,
/// 4 or 8.
pub fn tokens_tile(n: i64) -> i32 {
    #[cfg(feature = "cuda")]
    {
        let f = cuda::FORCED_NT.load(std::sync::atomic::Ordering::Relaxed);
        if f != 0 {
            return f;
        }
    }
    match n {
        ..=32 => 2,
        33..=64 => 4,
        _ => 8,
    }
}

/// Launches every W4A4 GEMM from now on with token tile `nt` (2, 4 or 8; grouped also
/// [`TWO_CTAS_4`]), or the default choice for `None`: for tests and tuning.
#[doc(hidden)]
pub fn force_tokens_tile(nt: Option<i32>) {
    #[cfg(feature = "cuda")]
    cuda::FORCED_NT.store(nt.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "cuda"))]
    let _ = nt;
}

/// Launches every dense W4A4 GEMM from now on with one CTA per tile (`true`) instead of the
/// stream-K split of GEMMs with fewer tiles than SMs (the default, `false`): for tests (the
/// split gives the same bits) and tuning.
#[doc(hidden)]
pub fn force_no_stream_k(off: bool) {
    #[cfg(feature = "cuda")]
    cuda::NO_SK.store(off, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "cuda"))]
    let _ = off;
}

/// The grouped tile of 64 tokens sized for two CTAs per SM (`legion_nvfp4_a4_gemm_launch`'s
/// `LEGION_NVFP4_A4_TWO_CTAS | 4`), also for [`force_tokens_tile`].
#[doc(hidden)]
pub const TWO_CTAS_4: i32 = 0x104;

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::c_void;
    use std::sync::atomic::AtomicI32;

    /// A token tile forced by [`super::force_tokens_tile`]; 0 is none.
    pub(super) static FORCED_NT: AtomicI32 = AtomicI32::new(0);

    /// The stream-K split turned off by [`super::force_no_stream_k`].
    pub(super) static NO_SK: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// A grouped kernel forced by [`super::force_grouped`]: 0 none, 1 the GEMM, `2 | rt << 8 |
    /// kw << 16` the decode kernel.
    pub(super) static FORCED_GROUPED: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);

    /// The grouped kernel for `n_tok` tokens of experts with rows of `k`.
    fn pick_grouped(n_tok: i64, k: i64, glu: bool) -> Grouped {
        use std::sync::atomic::Ordering::Relaxed;
        match FORCED_GROUPED.load(Relaxed) {
            1 => return Grouped::Gemm,
            0 => {}
            f => {
                return Grouped::Mv {
                    rt: (f >> 8 & 0xff) as i32,
                    kw: (f >> 16 & 0xff) as i32,
                };
            }
        }
        if FORCED_NT.load(Relaxed) != 0 || n_tok > MV_MAX_TOKENS {
            Grouped::Gemm
        } else {
            mv_shape(k, glu)
        }
    }

    use super::*;

    /// The dense GLU's GeGLU epilogue (`act`): the tanh GELU of the gate, BF16 out.
    const ACT_GEGLU: i32 = 3;

    /// `legion_nvfp4_a4` of `src/cuda/sm120/nvfp4_a4.cu`.
    #[repr(C)]
    struct Params {
        w: *const u8,
        xq: *const u8,
        xs: *const u8,
        xd: *const f32,
        scale: *const f32,
        dst: *mut c_void,
        k: i64,
        m: i64,
        n: i64,
        xq_ld: i64,
        xs_ld: i64,
        w_ld: i64,
        w_e: i64,
        tiles_m: i32,
        tiles_n: i32,
        group_m: i32,
        plan: *const i32,
        n_expert: i32,
        n_used: i32,
        n_x: i32,
        w2: *const u8,
        scale2: *const f32,
        act: i32,
        pw: *const f32,
        res: *const f32,
        sk_acc: *mut f32,
        sk_flags: *mut i32,
        sk_max: i32,
    }

    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn legion_nvfp4_a4_quant_launch(
            x: *const c_void,
            x_bf16: i32,
            x_ld: i64,
            k: i64,
            rows: i64,
            q: *mut u8,
            q_ld: i64,
            s: *mut u8,
            s_ld: i64,
            d: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn legion_nvfp4_a4_rms_quant_launch(
            x: *const f32,
            x_ld: i64,
            w: *const f32,
            eps: f32,
            k: i64,
            rows: i64,
            q: *mut u8,
            q_ld: i64,
            s: *mut u8,
            s_ld: i64,
            d: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_a4_gemm_launch(p: *const Params, nt: i32, stream: *mut c_void) -> i32;
        fn legion_nvfp4_a4_mv_launch(
            p: *const Params,
            nt: i32,
            rt: i32,
            kw: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_a4_combine_launch(
            y: *const u16,
            m: i64,
            n_used: i32,
            n_tok: i64,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_plan_launch(
            ids: *const i32,
            ids_ld: i64,
            rows: *const i32,
            skip_pad: i32,
            n_used: i32,
            n_tokens: i64,
            n_expert: i32,
            plan: *mut i32,
            stream: *mut c_void,
        ) -> i32;
    }

    /// The grouped GEMM's tile for token tile `nt` ([`tokens_tile`]): the SwiGLU's and short-k
    /// (the routed experts' down matmul) tiles of 64 or 128 tokens as 64-token tiles for two
    /// CTAs per SM, faster on the GB10 for Qwen3.8-Flash-Next's experts (the launcher). A
    /// forced tile stays as forced. The sums are the same either way (k in the same order).
    fn grouped_tile(nt: i32, glu: bool, k: i64) -> i32 {
        let forced = FORCED_NT.load(std::sync::atomic::Ordering::Relaxed) != 0;
        if !forced && nt >= 4 && (glu || k <= 1024) {
            TWO_CTAS_4
        } else {
            nt
        }
    }

    /// Weight rows per CTA (half as many output rows for SwiGLU) and the dense tile order's
    /// group.
    const BM: i64 = 128;
    const GROUP_M: i32 = 8;

    /// Quantizes `rows` rows of `x` into the node's scratch; returns the codes, scales and
    /// decode scales' addresses.
    fn quantize(
        dst: &ffi::ggml_tensor,
        x: &ffi::ggml_tensor,
        k: i64,
        rows: i64,
        out_bytes: usize,
        stream: *mut c_void,
    ) -> Result<(*const u8, *const u8, *const f32), String> {
        let x_ld = (x.nb[1] / x.nb[0]) as i64;
        if k % 64 != 0 || !(x.nb[1]).is_multiple_of(16) || !(x.data as usize).is_multiple_of(16) {
            return Err(format!(
                "unsupported NVFP4 W4A4 input: k = {k}, x_ld = {x_ld}"
            ));
        }
        let ([codes, scales, dec, _], _) = layout(out_bytes, rows, k, 0);
        let (q_ld, s_ld) = q_strides(k);
        let base = dst.data as *mut u8;
        // SAFETY: offsets inside the node (see `layout`).
        let (q, s, d) = unsafe { (base.add(codes), base.add(scales), base.add(dec) as *mut f32) };
        // SAFETY: device pointers of the node's input and scratch.
        let code = unsafe {
            legion_nvfp4_a4_quant_launch(
                x.data as *const c_void,
                (x.type_ == types::BF16) as i32,
                x_ld,
                k,
                rows,
                q,
                q_ld,
                s,
                s_ld,
                d,
                stream,
            )
        };
        if code != 0 {
            return Err(format!("nvfp4 a4 quant: CUDA error {code}"));
        }
        Ok((q, s, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn params(
        w: &ffi::ggml_tensor,
        (xq, xs, xd): (*const u8, *const u8, *const f32),
        scale: *const f32,
        dst: *mut c_void,
        (k, m, n): (i64, i64, i64),
        tiles: (i64, i64),
        act: i32,
    ) -> Params {
        let (q_ld, s_ld) = q_strides(k);
        Params {
            w: w.data as *const u8,
            xq,
            xs,
            xd,
            scale,
            dst,
            k,
            m,
            n,
            xq_ld: q_ld,
            xs_ld: s_ld,
            w_ld: w.nb[1] as i64,
            w_e: w.nb[2] as i64,
            tiles_m: tiles.0 as i32,
            tiles_n: tiles.1 as i32,
            group_m: GROUP_M,
            plan: std::ptr::null(),
            n_expert: 0,
            n_used: 0,
            n_x: 0,
            w2: std::ptr::null(),
            scale2: std::ptr::null(),
            act,
            pw: std::ptr::null(),
            res: std::ptr::null(),
            sk_acc: std::ptr::null_mut(),
            sk_flags: std::ptr::null_mut(),
            sk_max: 0,
        }
    }

    /// Points `p` at the node's stream-K scratch when it has one (see [`super::sk_bytes`]):
    /// `total` is the node's size without it, `(m_min, n)` what the builder reserved it for.
    fn set_sk(p: &mut Params, dst: &ffi::ggml_tensor, total: usize, (m_min, n): (i64, i64)) {
        let sk = super::sk_bytes(m_min, n);
        if sk == 0 || NO_SK.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let (at, _) = super::with_sk(total, sk);
        let base = dst.data as *mut u8;
        // SAFETY: the scratch lies inside the node (see `dense` and `mul_mat_multi`).
        unsafe {
            p.sk_acc = base.add(at) as *mut f32;
            p.sk_flags = base.add(at + super::SK_MAX * super::SK_ACC * 4) as *mut i32;
        }
        p.sk_max = super::SK_MAX as i32;
    }

    fn launch(p: &Params, nt: i32, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: parameters checked by the callers; the node's stream.
        match unsafe { legion_nvfp4_a4_gemm_launch(p, nt, stream) } {
            0 => Ok(()),
            -1 => Err(format!("nvfp4 a4: no kernel for nt = {nt}")),
            e => Err(format!("nvfp4 a4 gemm: CUDA error {e}")),
        }
    }

    fn scale_ptr(dst: &ffi::ggml_tensor, on: i32, i: usize) -> *const f32 {
        if on == 0 {
            return std::ptr::null();
        }
        // SAFETY: src[i] of a node made in this module, flagged as present.
        unsafe { (*dst.src[i]).data as *const f32 }
    }

    pub(super) unsafe fn quant(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `quantize_op`, tensors in device memory of the stream's device.
        let x = unsafe { &*dst.src[0] };
        quantize(dst, x, x.ne[0], x.ne[1] * x.ne[2] * x.ne[3], 0, stream).map(|_| ())
    }

    pub(super) unsafe fn mul_mat(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [has_scale, act, has_res, ..] = external_params(dst);
        // SAFETY: a node made by `mul_mat_act` or `mul_mat_add`, tensors in device memory of
        // the stream's device.
        let (w, x) = unsafe { (&*dst.src[0], &*dst.src[1]) };
        let (k, m, n) = (w.ne[0], w.ne[1], x.ne[1] * x.ne[2] * x.ne[3]);
        let esize = if act == 1 { 2 } else { 4 };
        let xq = quantize(dst, x, k, n, (m * n) as usize * esize, stream)?;
        let nt = tokens_tile(n);
        let tiles = (
            (m + BM - 1) / BM,
            (n + 16 * nt as i64 - 1) / (16 * nt as i64),
        );
        let mut p = params(
            w,
            xq,
            scale_ptr(dst, has_scale, 2),
            dst.data,
            (k, m, n),
            tiles,
            act,
        );
        p.res = scale_ptr(dst, has_res, 2 + has_scale as usize);
        let (_, total) = super::layout((m * n) as usize * esize, n, k, 0);
        set_sk(&mut p, dst, total, (m, n));
        launch(&p, nt, stream)
    }

    pub(super) unsafe fn mul_mat_multi(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [count, mask, has_norm, eps, ..] = external_params(dst);
        let count = count as usize;
        // SAFETY: a node made by `mul_mat_multi`, tensors in device memory of the stream's
        // device.
        let x = unsafe { &*dst.src[count] };
        let (k, n) = (x.ne[0], x.ne[1] * x.ne[2] * x.ne[3]);
        // SAFETY: the node's first `count` sources are its weights.
        let ws: Vec<&ffi::ggml_tensor> = (0..count).map(|i| unsafe { &*dst.src[i] }).collect();
        let out_bytes = ws.iter().map(|w| (w.ne[1] * n) as usize * 4).sum();
        let n_scales = (mask as u32).count_ones() as usize;
        let xq = if has_norm != 0 {
            // SAFETY: the norm weight follows the scales (see `mul_mat_multi`).
            let nw = unsafe { &*dst.src[count + 1 + n_scales] };
            let eps = f32::from_bits(eps as u32);
            rms_norm_quantize(dst, x, nw, eps, (k, n), out_bytes, stream)?
        } else {
            quantize(dst, x, k, n, out_bytes, stream)?
        };
        let nt = tokens_tile(n);
        let (_, total) = super::layout(out_bytes, n, k, 0);
        let m_min = ws.iter().map(|w| w.ne[1]).min().unwrap_or(0);
        let mut out = dst.data as *mut u8;
        let mut si = count + 1;
        for (i, w) in ws.iter().enumerate() {
            let m = w.ne[1];
            let scale = if mask & (1 << i) != 0 {
                si += 1;
                scale_ptr(dst, 1, si - 1)
            } else {
                std::ptr::null()
            };
            let tiles = (
                (m + BM - 1) / BM,
                (n + 16 * nt as i64 - 1) / (16 * nt as i64),
            );
            let mut p = params(w, xq, scale, out as *mut c_void, (k, m, n), tiles, 0);
            set_sk(&mut p, dst, total, (m_min, n));
            launch(&p, nt, stream)?;
            // SAFETY: the outputs lie back to back at the start of the node (see `layout`).
            out = unsafe { out.add((m * n) as usize * 4) };
        }
        Ok(())
    }

    /// `x`'s rows normed by `nw` (`mul(rms_norm(x, eps), nw)`, ggml-cuda's bits) and quantized
    /// into the node's scratch, as [`quantize`].
    fn rms_norm_quantize(
        dst: &ffi::ggml_tensor,
        x: &ffi::ggml_tensor,
        nw: &ffi::ggml_tensor,
        eps: f32,
        (k, rows): (i64, i64),
        out_bytes: usize,
        stream: *mut c_void,
    ) -> Result<(*const u8, *const u8, *const f32), String> {
        let ([codes, scales, dec, _], _) = layout(out_bytes, rows, k, 0);
        let (q_ld, s_ld) = q_strides(k);
        let base = dst.data as *mut u8;
        // SAFETY: offsets inside the node (see `layout`).
        let (q, s, d) = unsafe { (base.add(codes), base.add(scales), base.add(dec) as *mut f32) };
        // SAFETY: device pointers of the node's inputs and scratch; x contiguous F32 (checked
        // by `swiglu_norm`).
        let code = unsafe {
            legion_nvfp4_a4_rms_quant_launch(
                x.data as *const f32,
                k,
                nw.data as *const f32,
                eps,
                k,
                rows,
                q,
                q_ld,
                s,
                s_ld,
                d,
                stream,
            )
        };
        match code {
            0 => Ok((q, s, d)),
            1 => Err(format!("unsupported NVFP4 W4A4 norm: k = {k}")),
            e => Err(format!("nvfp4 a4 rms quant: CUDA error {e}")),
        }
    }

    pub(super) unsafe fn swiglu(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let [has_g, has_u, has_norm, eps, gelu, ..] = external_params(dst);
        // SAFETY: a node made by `swiglu_with`, tensors in device memory of the stream's
        // device.
        let (wg, wu, x) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2]) };
        let (k, m, n) = (wg.ne[0], wg.ne[1], x.ne[1] * x.ne[2] * x.ne[3]);
        let out_bytes = (m * n) as usize * 2;
        let xq = if has_norm != 0 {
            // SAFETY: the norm weight follows the scales (see `swiglu_norm`).
            let nw = unsafe { &*dst.src[3 + (has_g + has_u) as usize] };
            let eps = f32::from_bits(eps as u32);
            rms_norm_quantize(dst, x, nw, eps, (k, n), out_bytes, stream)?
        } else {
            quantize(dst, x, k, n, out_bytes, stream)?
        };
        let nt = tokens_tile(n);
        let tiles = (
            (m + BM / 2 - 1) / (BM / 2),
            (n + 16 * nt as i64 - 1) / (16 * nt as i64),
        );
        let mut p = params(
            wg,
            xq,
            scale_ptr(dst, has_g, 3),
            dst.data,
            (k, m, n),
            tiles,
            if gelu != 0 { ACT_GEGLU } else { 2 },
        );
        p.w2 = wu.data as *const u8;
        p.scale2 = scale_ptr(dst, has_u, 3 + has_g as usize);
        launch(&p, nt, stream)
    }

    /// A grouped GEMM of the node's experts (src 0), activations (1) and ids (2), its output
    /// (`out_bytes`) at `out` in the node; `pw`: routing weights for act 2.
    #[allow(clippy::too_many_arguments)]
    fn grouped(
        dst: &ffi::ggml_tensor,
        scale: *const f32,
        act: i32,
        out: *mut c_void,
        out_bytes: usize,
        pw: *const f32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat_id_act` or `mul_mat_id_weighted`, tensors in device
        // memory of the stream's device.
        let (w, x, ids) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2]) };
        grouped_on(
            dst,
            (w, None),
            x,
            ids,
            (scale, std::ptr::null()),
            act,
            out,
            out_bytes,
            pw,
            stream,
        )
    }

    /// [`grouped`] on given tensors: experts `w.0` (and the SwiGLU's up experts `w.1`, scales
    /// `scales.1`), activations `x`, `ids`.
    #[allow(clippy::too_many_arguments)]
    fn grouped_on(
        dst: &ffi::ggml_tensor,
        w: (&ffi::ggml_tensor, Option<&ffi::ggml_tensor>),
        x: &ffi::ggml_tensor,
        ids: &ffi::ggml_tensor,
        scales: (*const f32, *const f32),
        act: i32,
        out: *mut c_void,
        out_bytes: usize,
        pw: *const f32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let (w, w2) = w;
        let (scale, scale2) = scales;
        let (k, m, n_expert) = (w.ne[0], w.ne[1], w.ne[2]);
        let (n_used, n_tok) = (ids.ne[0], ids.ne[1]);
        let n_x = x.ne[1];
        let xq = quantize(dst, x, k, n_x * n_tok, out_bytes, stream)?;
        let ([.., plan_at], _) = layout(out_bytes, n_x * n_tok, k, 0);
        // SAFETY: the plan's I32s follow the scratch in the node (see `layout`).
        let plan = unsafe { (dst.data as *mut u8).add(plan_at) } as *mut i32;
        // SAFETY: device pointers of the node's inputs and scratch.
        let code = unsafe {
            legion_nvfp4_plan_launch(
                ids.data as *const i32,
                (ids.nb[1] / 4) as i64,
                std::ptr::null(),
                0,
                n_used as i32,
                n_tok,
                n_expert as i32,
                plan,
                stream,
            )
        };
        if code != 0 {
            return Err(format!("nvfp4 plan: CUDA error {code}"));
        }
        let pairs = n_used * n_tok;
        let per = pairs / pairs.min(n_expert).max(1);
        if let Grouped::Mv { rt, kw } = pick_grouped(n_tok, k, w2.is_some()) {
            let nt = mv_tokens_tile(per);
            let rows = if w2.is_some() { 8 } else { 16 } * rt as i64;
            let tiles = (
                (m + rows - 1) / rows,
                pairs / (8 * nt as i64) + pairs.min(n_expert),
            );
            let mut p = params(w, xq, scale, out, (k, m, pairs), tiles, act);
            if let Some(w2) = w2 {
                p.w2 = w2.data as *const u8;
                p.scale2 = scale2;
            }
            p.plan = plan;
            p.n_expert = n_expert as i32;
            p.n_used = n_used as i32;
            p.n_x = n_x as i32;
            p.pw = pw;
            // SAFETY: parameters checked above; the node's stream.
            match unsafe { legion_nvfp4_a4_mv_launch(&p, nt, rt, kw, stream) } {
                0 => return Ok(()),
                -1 => {} // weights the kernel cannot copy: the GEMM below
                e => return Err(format!("nvfp4 a4 mv: CUDA error {e}")),
            }
        }
        let nt = grouped_tile(tokens_tile(per), w2.is_some(), k);
        let cols = 16 * (nt & 0xff) as i64;
        // The full tiles, plus at most one partial tile per expert; SwiGLU tiles are half as
        // many output rows.
        let bm = if w2.is_some() { BM / 2 } else { BM };
        let tiles = ((m + bm - 1) / bm, pairs / cols + pairs.min(n_expert));
        let mut p = params(w, xq, scale, out, (k, m, pairs), tiles, act);
        if let Some(w2) = w2 {
            p.w2 = w2.data as *const u8;
            p.scale2 = scale2;
        }
        p.plan = plan;
        p.n_expert = n_expert as i32;
        p.n_used = n_used as i32;
        p.n_x = n_x as i32;
        p.pw = pw;
        launch(&p, nt, stream)
    }

    pub(super) unsafe fn mul_mat_id(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [has_scale, act, ..] = external_params(dst);
        // SAFETY: a node made by `mul_mat_id_act`.
        let (m, ids) = unsafe { ((*dst.src[0]).ne[1], &*dst.src[2]) };
        let esize = if act == 1 { 2 } else { 4 };
        let out = (m * ids.ne[0] * ids.ne[1]) as usize * esize;
        let scale = scale_ptr(dst, has_scale, 3);
        grouped(dst, scale, act, dst.data, out, std::ptr::null(), stream)
    }

    pub(super) unsafe fn swiglu_id(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [has_g, has_u, ..] = external_params(dst);
        // SAFETY: a node made by `swiglu_id`, tensors in device memory of the stream's device.
        let (wg, wu, x, ids) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]) };
        let out = (wg.ne[1] * ids.ne[0] * ids.ne[1]) as usize * 4;
        let scales = (
            scale_ptr(dst, has_g, 4),
            scale_ptr(dst, has_u, 4 + has_g as usize),
        );
        grouped_on(
            dst,
            (wg, Some(wu)),
            x,
            ids,
            scales,
            0,
            dst.data,
            out,
            std::ptr::null(),
            stream,
        )
    }

    pub(super) unsafe fn mul_mat_id_weighted(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let [has_scale, ..] = external_params(dst);
        // SAFETY: a node made by `mul_mat_id_weighted`.
        let (w, x, ids, weights) =
            unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]) };
        let (k, m, n_expert) = (w.ne[0], w.ne[1], w.ne[2]);
        let (n_used, n_tok) = (ids.ne[0], ids.ne[1]);
        let out = (m * n_tok) as usize * 4;
        let mid = weighted_mid(out, x.ne[1] * n_tok, k, n_expert, n_used * n_tok);
        // SAFETY: the BF16 expert outputs at the end of the node (see `weighted_mid`).
        let y = unsafe { (dst.data as *mut u8).add(mid) } as *mut c_void;
        let scale = scale_ptr(dst, has_scale, 4);
        grouped(dst, scale, 2, y, out, weights.data as *const f32, stream)?;
        // SAFETY: device pointers inside the node.
        let code = unsafe {
            legion_nvfp4_a4_combine_launch(
                y as *const u16,
                m,
                n_used as i32,
                n_tok,
                dst.data as *mut f32,
                stream,
            )
        };
        match code {
            0 => Ok(()),
            e => Err(format!("nvfp4 a4 combine: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
