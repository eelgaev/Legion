//! GLM-5.3-Flash's own ops ([`crate::ggml::ExternalOp`]): every computation of the
//! `glm5-next` graph that is not an NVFP4 / FP8 matmul ([`crate::model::nvfp4_mm`],
//! [`crate::model::fp8`]), a routed-expert FFN ([`crate::model::moe_ffn`]) or a sparse
//! selection step ([`crate::model::qwen4::qsa_attn`]). ggml only holds the tensors, views and
//! reshapes; nothing here is a ggml compute op.
//!
//! The CPU implementations are plain references (F32 math, sums in order, norms' sums in f64
//! as ggml's), for tests and the CPU backend; the CUDA kernels are `src/cuda/glm5next.cu`
//! ([`cuda`]). Formulas follow llama.cpp's `src/models/glm5-next.cpp` (MIT) unless a doc says
//! otherwise.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU16, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub mod lin_bench;

macro_rules! external_op {
    ($static:ident, $name:literal, $cpu:ident, $cuda:ident) => {
        static $static: ExternalOp = ExternalOp {
            name: $name,
            cpu: $cpu,
            #[cfg(feature = "cuda")]
            cuda: Some(cuda::$cuda),
            #[cfg(not(feature = "cuda"))]
            cuda: None,
        };
    };
}

external_op!(EMBED, "glm_embed", cpu_embed, embed);
external_op!(LINEAR, "glm_linear", cpu_linear, linear);
external_op!(BF16, "glm_bf16", cpu_bf16, bf16);
external_op!(RMS_NORM, "glm_rms_norm", cpu_rms_norm, rms_norm);
external_op!(LAYER_NORM, "glm_layer_norm", cpu_layer_norm, layer_norm);
external_op!(HC_PRE, "glm_hc_pre", cpu_hc_pre, hc_pre);
external_op!(HC_POST, "glm_hc_post", cpu_hc_post, hc_post);
external_op!(HC_HEAD, "glm_hc_head", cpu_hc_head, hc_head);
external_op!(KDA_CONV, "glm_kda_conv", cpu_kda_conv, kda_conv);
external_op!(KDA_GATE, "glm_kda_gate", cpu_kda_gate, kda_gate);
external_op!(SIGMOID, "glm_sigmoid", cpu_sigmoid, sigmoid);
external_op!(KDA_SCAN, "glm_kda_scan", cpu_kda_scan, kda_scan);
external_op!(GATED_RMS, "glm_gated_rms", cpu_gated_rms, gated_rms);
external_op!(GLU, "glm_glu", cpu_glu, glu);
external_op!(ROUTER_IDS, "glm_router_ids", cpu_router_ids, router_ids);
external_op!(ROUTER_W, "glm_router_w", cpu_router_w, router_w);
external_op!(
    WEIGHTED_SUM,
    "glm_weighted_sum",
    cpu_weighted_sum,
    weighted_sum
);
external_op!(CACHE_ROWS, "glm_cache_rows", cpu_cache_rows, cache_rows);
external_op!(POOL_KEYS, "glm_pool_keys", cpu_pool_keys, pool_keys);
external_op!(MLA_ATTN, "glm_mla_attn", cpu_mla_attn, mla_attn);
external_op!(JOIN_COLS, "glm_join_cols", cpu_join_cols, join_cols);
external_op!(ARGMAX, "glm_argmax", cpu_argmax, argmax);
external_op!(PMAX, "glm_pmax", cpu_pmax, pmax);
external_op!(NEXTN_IN, "glm_nextn_in", cpu_nextn_in, nextn_in);
external_op!(GET_COLS, "glm_get_cols", cpu_get_cols, get_cols);
external_op!(ADD, "glm_add", cpu_add, add);
external_op!(F16, "glm_f16", cpu_f16, f16);
external_op!(WF16, "glm_wf16", cpu_wf16, wf16);
external_op!(LINEAR_SET, "glm_linear_set", cpu_linear_set, linear_set);
external_op!(GLU_MV, "glm_glu_mv", cpu_glu_mv, glu_mv);

// ---- graph builders ----

/// The hyper-connection streams of `tokens` (I32 `[n]`): each row of `w` (`[n_embd, n_vocab]`,
/// any type ggml can convert) `hc` times, F32 `[n_embd, hc, n]`.
pub fn embed<'a>(ctx: &'a Context, w: Tensor<'a>, tokens: Tensor<'a>, hc: i64) -> Tensor<'a> {
    let n = tokens.ne()[0];
    ctx.external(&EMBED, types::F32, [w.ne()[0], hc, n, 1], &[w, tokens])
}

/// `scale · ws · (w @ x)`: `w` `[k, m]` (any type ggml can convert to F32, rows contiguous)
/// and `x` F32 `[k, n]` give `[m, n]`; batched, `w` `[k, m, h]` and `x` `[k, h, n]` give
/// `[m, h, n]` (head `i` of every token times matrix `i`). `x` may be a strided view with
/// contiguous rows. `ws`: a checkpoint's weight scale (F32, one value or one per row of `m`).
pub fn linear<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: f32,
    ws: Option<Tensor<'a>>,
) -> Tensor<'a> {
    linear_h(ctx, w, x, None, scale, ws)
}

/// [`linear`] with `xh`, `x` rounded to F16 by [`f16`] (one matrix), for the CUDA kernels to
/// read in place of `x` where they would round it the same way themselves (the same bits):
/// linear_mm_mma, ~15% faster on F16 activations (#571). Worth it for 2+ products of one `x`.
pub fn linear_h<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    xh: Option<Tensor<'a>>,
    scale: f32,
    ws: Option<Tensor<'a>>,
) -> Tensor<'a> {
    linear_tc_kernel(ctx, w, x, xh, scale, ws, LinTc::Auto)
}

/// The CUDA kernel of a [`linear`]'s F16 / BF16 prompt GEMM (linear_tc's products: sm_80+ with
/// the GB10 kernels, F16 or BF16 weights, Q6_K ones through [`wf16`]). All of them give the same
/// bits; they differ in speed only (#862).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinTc {
    /// The launcher's choice: linear_tc_gb's measured shape on compute capability 12.x (the
    /// GB10) and Ampere (8.0 to 8.7, a table of its own, #1212), linear_tc on the other GPUs.
    Auto,
    /// `linear_tc`: 128 x 128 tiles, K steps of 32, three stages.
    Tc,
    /// Shape `v` of `linear_tc_gb` (`src/cuda/sm80/linear_tc_gb10.cuh`, `tc_gb_shape`).
    Gb(u8),
}

impl LinTc {
    /// The node's `params[4]`.
    fn param(self) -> i32 {
        match self {
            LinTc::Auto => 0,
            LinTc::Tc => 1,
            LinTc::Gb(v) => 2 + v as i32,
        }
    }
}

/// [`linear_h`] on an explicit [`LinTc`] kernel (`legion lin-bench`).
pub fn linear_tc_kernel<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    xh: Option<Tensor<'a>>,
    scale: f32,
    ws: Option<Tensor<'a>>,
    kernel: LinTc,
) -> Tensor<'a> {
    linear_io(ctx, w, x, xh, scale, ws, kernel, false)
}

/// [`linear_tc_kernel`] that may also take `x` as F16 rows (an earlier product's `out_f16`
/// output, with no `xh`) and give F16 outputs (`out_f16`: each F32 output rounded to nearest,
/// [`f16`]'s rounding) for a next product that reads F16 rows (#862). Either one is linear_tc's
/// only (the CUDA launcher fails the node otherwise): use it where [`f16_chain_takes`].
#[allow(clippy::too_many_arguments)]
pub fn linear_io<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    xh: Option<Tensor<'a>>,
    scale: f32,
    ws: Option<Tensor<'a>>,
    kernel: LinTc,
    out_f16: bool,
) -> Tensor<'a> {
    let [k, m, h, _] = w.ne();
    assert_eq!(x.ne()[0], k, "glm_linear: x rows");
    let es = x.element_size();
    assert!(
        x.ty() == types::F32 || (x.ty() == types::F16 && xh.is_none()),
        "glm_linear: x is F32 (or F16 rows without xh)"
    );
    assert_eq!(x.nb()[0], es, "glm_linear: x rows contiguous");
    let ne = if h == 1 {
        let n: i64 = x.ne()[1..].iter().product();
        assert!(x.ne()[2] == 1 || x.ne()[1] == 1 || x.is_contiguous());
        [m, n, 1, 1]
    } else {
        assert_eq!(x.ne()[1], h, "glm_linear: one input row per matrix");
        [m, h, x.ne()[2], 1]
    };
    if let Some(s) = ws {
        assert!(
            s.ty() == types::F32
                && s.is_contiguous()
                && (s.elements() == 1 || s.elements() as i64 == m),
            "glm_linear: weight scale"
        );
    }
    if let Some(xh) = xh {
        assert!(
            h == 1 && xh.ty() == types::F16 && xh.is_contiguous() && xh.ne() == x.ne(),
            "glm_linear: xh is x in F16"
        );
    }
    let n = if h == 1 { ne[1] } else { ne[2] };
    let gb10 = gb10_kernels(
        GB10_LINEAR.load(Ordering::Relaxed),
        GB10_GEMV.load(Ordering::Relaxed),
        n,
    );
    let w = if w.ty() == types::Q6_K && q6k_f16_takes(gb10, Q6K_F16.load(Ordering::Relaxed), n) {
        wf16(ctx, w)
    } else {
        w
    };
    let mut args = vec![w, x];
    args.extend(ws);
    args.extend(xh);
    ctx.external_with(
        &LINEAR,
        if out_f16 { types::F16 } else { types::F32 },
        ne,
        &args,
        &[
            scale.to_bits() as i32,
            ws.is_some() as i32,
            gb10 as i32,
            xh.is_some() as i32,
            kernel.param(),
            out_f16 as i32,
            // the GB10 GEMV's alone (above 8 tokens params[2] is on without it)
            if GB10_GEMV.load(Ordering::Relaxed) {
                bf16_gemv() as i32
            } else {
                0
            },
            if GB10_GEMV.load(Ordering::Relaxed) && w.ty() == types::Q6_K {
                q6k_mv_param(q6k_mv(), k, m, w.nb()[1] as i64)
            } else if GB10_GEMV.load(Ordering::Relaxed) && w.ty() == types::F32 {
                (small_mv() & SMALL_MV_F32) as i32
            } else {
                0
            },
        ],
    )
}

/// [`small_mv`] before the first read: the environment's choice (outside u8: every u8 is a mode).
const SMALL_MV_UNSET: u16 = u16::MAX;

static SMALL_MV: AtomicU16 = AtomicU16::new(SMALL_MV_UNSET);

/// [`small_mv`]: F32 weights' decode GEMV (the router) with the tokens' inputs copied to shared
/// memory once per block (linear_mv_f32_xs).
pub(crate) const SMALL_MV_F32: u8 = 1;
/// [`small_mv`]: [`linear_set`]'s 128-value Q8_0 rows (`ssm_f_b`, `ssm_g_b`) staged through
/// shared memory with 16-byte loads, two rows a warp (linear_set_q8_rw).
pub(crate) const SMALL_MV_SET: u8 = 2;
/// [`small_mv`]: [`hc_pre_norm`]'s F32 mix rows with the tokens in one block, the weights read
/// once (hc_mix_tok_kernel).
pub(crate) const SMALL_MV_HC: u8 = 4;
/// [`small_mv`] (#1172 r5): [`hc_pre_norm`]'s finish (weights, `y` and `h`) launched while the
/// mix drains, its streams and norm weight loaded before it waits for the mix rows
/// (hc_mix_tok_kernel's early trigger and hc_finish_wide_kernel's PRE; with [`SMALL_MV_HC`]).
pub(crate) const SMALL_MV_HCF: u8 = 8;
/// [`small_mv`] (#1172 r5): [`glu_mv`] on these GPUs at 1-4 tokens for gate and up rows of one or
/// two whole stages (the shared expert's 4096 x 2048), each row loaded before the wait and summed
/// as linear_mv_q6k_st sums it (glu_mv_q6k_sp): one launch instead of two GEMVs and the GLU.
pub(crate) const SMALL_MV_GLU: u8 = 16;
/// [`small_mv`] (#1172 r6): the kernels ahead of MLA's kv_a and q_a and of the shared expert's
/// down (hc_finish_wide_kernel, pool_keys_kernel, glu_mv_q6k_sp) let the next kernel launch once
/// each of their blocks is past its own wait (griddepcontrol.launch_dependents), so its blocks
/// are resident and its weights loading while they drain. The next kernel still waits for them
/// before it reads anything: the same bits.
pub(crate) const SMALL_MV_TRIG: u8 = 32;
/// [`small_mv`] (#1172 r6): [`linear_sp`]'s Q6_K products (kv_a, q_a, shared expert down) at 1-4
/// tokens load each warp's whole row (one or two stages) before the wait for the kernel ahead
/// and sum it as linear_mv_q6k_st sums it (linear_mv_q6k_sp): the same bits.
pub(crate) const SMALL_MV_SP: u8 = 64;
/// [`small_mv`] (#1172 r6): the routed experts' kernels back to back ([`moe_pdl`]).
pub(crate) const SMALL_MV_MOE: u8 = 128;

/// The small decode products of GLM's 1-4 token steps on [`set_gb10_gemv`]'s GPUs (#1172 r4-r6):
/// [`SMALL_MV_F32`], [`SMALL_MV_SET`], [`SMALL_MV_HC`], [`SMALL_MV_HCF`], [`SMALL_MV_GLU`],
/// [`SMALL_MV_TRIG`], [`SMALL_MV_SP`] and [`SMALL_MV_MOE`], each summing the same values in the
/// same order as the kernels it replaces (the same bits) and only reading its memory differently
/// or launching sooner. `LEGION_DEBUG_GLM_SMALL_MV` picks the bits for A/B runs (0: the kernels
/// before r4, 7: r4's, 31: r5's).
pub fn small_mv() -> u8 {
    let v = SMALL_MV.load(Ordering::Relaxed);
    if v != SMALL_MV_UNSET {
        return v as u8;
    }
    let v = small_mv_of(std::env::var("LEGION_DEBUG_GLM_SMALL_MV").ok().as_deref());
    SMALL_MV.store(v as u16, Ordering::Relaxed);
    v
}

/// [`small_mv`] given the environment's value.
pub(crate) fn small_mv_of(env: Option<&str>) -> u8 {
    match env.map(str::trim).and_then(|s| s.parse::<u8>().ok()) {
        Some(v) => v & SMALL_MV_BITS,
        None => SMALL_MV_DEFAULT,
    }
}

/// [`small_mv`]'s default: every bit but [`SMALL_MV_SP`] (31 = r5's kernels). The pre-wait rows
/// gave nothing in the model with the early triggers on (#1172 r6: 191 and 255 the same step) and
/// are 4-21% slower alone, so they stay opt-in.
pub(crate) const SMALL_MV_DEFAULT: u8 = SMALL_MV_BITS & !SMALL_MV_SP;

/// The bits [`small_mv`] uses.
const SMALL_MV_BITS: u8 = SMALL_MV_F32
    | SMALL_MV_SET
    | SMALL_MV_HC
    | SMALL_MV_HCF
    | SMALL_MV_GLU
    | SMALL_MV_TRIG
    | SMALL_MV_SP
    | SMALL_MV_MOE;

/// Sets [`small_mv`] for nodes built from now on (`legion lin-bench --small`).
pub fn set_small_mv(v: u8) {
    SMALL_MV.store((v & SMALL_MV_BITS) as u16, Ordering::Relaxed);
}

/// [`small_mv`] as the nodes' parameter: its bits on [`set_gb10_gemv`]'s GPUs, else 0.
fn small_mv_param() -> i32 {
    if GB10_GEMV.load(Ordering::Relaxed) {
        small_mv() as i32
    } else {
        0
    }
}

/// Whether GLM's routed experts launch their decode steps' kernels back to back
/// ([`SMALL_MV_MOE`], `crate::model::moe_ffn::glm_pdl`): on [`set_gb10_gemv`]'s GPUs.
pub(crate) fn moe_pdl() -> bool {
    small_mv_param() as u8 & SMALL_MV_MOE != 0
}

/// `linear_mv_q6k_sp`'s flag in a Q6_K [`linear`]'s `params[7]` (above [`q6k_mv_param`]'s bits).
const Q6K_SP: i32 = 1 << 10;

/// Marks `t`, a [`linear`] (or [`linear_io`]) result, for linear_mv_q6k_sp ([`SMALL_MV_SP`]) where
/// its weights are Q6_K rows of one or two whole stages, contiguous, and [`q6k_mv`] left it on
/// linear_mv_q6k_st (#1172 r6: MLA's kv_a and q_a and the shared expert's down, whose kernels
/// ahead trigger early with [`SMALL_MV_TRIG`]). CUDA takes it at 1-4 tokens on the GB10 kernels
/// and keeps linear_mv_q6k_st otherwise. Any other tensor is returned as is.
pub(crate) fn linear_sp<'a>(t: Tensor<'a>) -> Tensor<'a> {
    if small_mv_param() as u8 & SMALL_MV_SP == 0 {
        return t;
    }
    let node = t.raw();
    // SAFETY: `t` is a valid tensor; a LINEAR node's first source is its weights and its
    // params are linear_io's.
    unsafe {
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &LINEAR)) {
            return t;
        }
        let w = &*(*node).src[0];
        let p = crate::ggml::external_params(&*node);
        if w.type_ == types::Q6_K
            && w.ne[2] == 1
            && p[2] != 0
            && p[7] == 0
            && sp_shape(w.ne[0], w.nb[1] as i64)
        {
            crate::ggml::set_external_param(node, 7, Q6K_SP);
        }
    }
    t
}

/// [`linear_sp`]'s rows: `k` values in one or two whole stages (2048 or 4096), `nb1` bytes apart.
pub(crate) fn sp_shape(k: i64, nb1: i64) -> bool {
    k > 0 && k % (256 * Q6K_STAGE) == 0 && k / (256 * Q6K_STAGE) <= 2 && nb1 == k / 256 * 210
}

/// [`q6k_mv`] before the first read: the environment's choice.
const Q6K_MV_UNSET: u8 = u8::MAX;

static Q6K_MV: AtomicU8 = AtomicU8::new(Q6K_MV_UNSET);

/// The Q6_K decode GEMV of [`set_gb10_gemv`]'s steps (8 tokens or fewer, #1172 r3; the node's
/// `params[7]`, [`q6k_mv_param`]). 0: linear_mv_q6k_st on rows of whole stages (2048 values),
/// linear_mv_q6k (2-byte loads) on the others. Bit 0: linear_mv_q6k_rw on those others at 1-4 tokens,
/// linear_mv_q6k_st's lanes over several contiguous rows a warp (MLA's q_b, k_b and v_b), its first
/// stages loaded before the wait for the kernel ahead; bit 1: two stages in flight; bit 2: the
/// next kernel launched once every block is past its wait; bit 3: the output head's rows too
/// (#1172 r8); bits 4-6: log2 of more rows a warp.
/// Every lane sums the same values in the same order in all of them: the same bits.
/// `LEGION_DEBUG_GLM_Q6K_MV` picks one for A/B runs (0: as before).
pub fn q6k_mv() -> u8 {
    let v = Q6K_MV.load(Ordering::Relaxed);
    if v != Q6K_MV_UNSET {
        return v;
    }
    let v = q6k_mv_of(std::env::var("LEGION_DEBUG_GLM_Q6K_MV").ok().as_deref());
    Q6K_MV.store(v, Ordering::Relaxed);
    v
}

/// [`q6k_mv`] given the environment's value.
pub(crate) fn q6k_mv_of(env: Option<&str>) -> u8 {
    match env.map(str::trim).and_then(|s| s.parse::<u8>().ok()) {
        Some(v) => v & Q6K_MV_BITS,
        None => Q6K_MV_DEFAULT,
    }
}

/// [`q6k_mv`]'s default: linear_mv_q6k_rw on MLA's q_b, k_b and v_b (a DGX Spark at one user:
/// those three 141 / 66 / 85 GB/s -> 174 / 128 / 126 at 3 tokens, ~1 ms off an 85 ms verify
/// step). Bits 1 and 2 gained nothing there.
pub(crate) const Q6K_MV_DEFAULT: u8 = 1;

/// The bits [`q6k_mv`] uses.
const Q6K_MV_BITS: u8 = 0x7F;

/// Sets [`q6k_mv`] for [`linear`]s built from now on (`legion lin-bench --q6k`).
pub fn set_q6k_mv(v: u8) {
    Q6K_MV.store(v & Q6K_MV_BITS, Ordering::Relaxed);
}

/// Super-blocks a linear_mv_q6k_rw stage holds (the kernel file's KQ_STAGE).
const Q6K_STAGE: i64 = 8;

/// Rows from which a Q6_K product of whole-stage rows counts as the output head for
/// [`q6k_mv`]'s bit 3 (Maya-S: 154,880, or 66,067 with `--spec-draft-vocab 65536`; its widest
/// other Q6_K matrix has 16,384).
const Q6K_HEAD_ROWS: i64 = 32768;

/// A Q6_K [`linear`]'s `params[7]` for [`q6k_mv`] mode `mode`, weights `[k, m]` (or `[k, m, h]`)
/// with rows `nb1` bytes apart: linear_mv_q6k_rw's rows a warp (bits 0-7: the fewest that make
/// whole stages, times 2^(mode bits 4-6)), two stages in flight (bit 8), early launch (bit 9);
/// 0 (the staged or plain kernel) when the mode is off, the rows are whole stages already, not
/// contiguous, or `m` is no multiple of the rows a warp.
pub(crate) fn q6k_mv_param(mode: u8, k: i64, m: i64, nb1: i64) -> i32 {
    if mode & 1 == 0 || k <= 0 || k % 256 != 0 || m <= 0 || nb1 != k / 256 * 210 {
        return 0;
    }
    let nbr = k / 256;
    let mut a = nbr;
    let mut b = Q6K_STAGE;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    // Rows of whole stages stay on linear_mv_q6k_st: the same bytes, and its fixed row loop is
    // 10-25% faster at 2-4 columns (`lin-bench --q6k`). Bit 3 (#1172 r8) moves the output head's
    // (Q6K_HEAD_ROWS or more rows) to linear_mv_q6k_rw too, 2^(bits 4-6) rows a warp.
    if a == Q6K_STAGE && (mode & 8 == 0 || m < Q6K_HEAD_ROWS) {
        return 0;
    }
    let rows = (Q6K_STAGE / a) << ((mode >> 4) & 7);
    if rows > 255 || m % rows != 0 {
        return 0;
    }
    rows as i32 | (((mode >> 1) & 1) as i32) << 8 | (((mode >> 2) & 1) as i32) << 9
}

/// Whether a prompt step of `n` tokens on a compute capability 12.x GPU (`cc12`, the GB10)
/// hands F16 rows from one Q6_K product to the next where the next one rounds its input to F16
/// anyway (#862): GLM's MLA attn_q_b -> attn_k_b and attn_v_b -> attn_output. All of them take
/// linear_tc then ([`q6k_f16_takes`]), which rounds F32 rows with `__float2half_rn` as [`f16`]
/// does, so the same bits with half the bytes between them and no [`f16`] copy before
/// attn_output (Maya-S: ~190 KB per token and MLA layer less memory traffic).
pub(crate) fn f16_chain_takes(cc12: bool, n: i64) -> bool {
    let gb10 = gb10_kernels(
        GB10_LINEAR.load(Ordering::Relaxed),
        GB10_GEMV.load(Ordering::Relaxed),
        n,
    );
    f16_chain_pays(
        cc12,
        q6k_f16_takes(gb10, Q6K_F16.load(Ordering::Relaxed), n),
    )
}

/// [`f16_chain_takes`] given whether the Q6_K products take linear_tc (`tc`).
pub(crate) fn f16_chain_pays(cc12: bool, tc: bool) -> bool {
    cc12 && tc
}

/// Tokens from which a prompt step's Q6_K products go through [`wf16`] (see [`set_q6k_f16`]):
/// the F16 copy costs about as much as ~1000 tokens of the faster GEMM gain (GB10, Maya-S).
pub(crate) const Q6K_F16_MIN: i64 = 1024;

/// Whether a [`linear`] of `n` tokens with Q6_K weights takes [`wf16`] first: the GB10 kernels
/// (`gb10`, [`gb10_kernels`]), sm_80+ (`on`, [`set_q6k_f16`]) and [`Q6K_F16_MIN`]+ tokens.
pub(crate) fn q6k_f16_takes(gb10: bool, on: bool, n: i64) -> bool {
    gb10 && on && n >= Q6K_F16_MIN
}

/// Weight rows from which [`xh_one_takes`] rounds a lone product's `x` to F16: with the copy,
/// Maya-S's 512-row attn_kv_a_mqa took 1.4-1.7x as long, its 1536-row attn_q_a 0.92-0.97x.
pub(crate) const XH_ONE_MIN_ROWS: i64 = 1024;

/// Whether a [`linear`] of `n` tokens with Q6_K weights `[k, m]` takes its F32 `x`, used by it
/// alone, in F16 from [`f16`] first, as 2+ products of one `x` do ([`linear_h`]; #862). On a
/// compute capability 12.x GPU (`cc12`, the GB10) linear_tc_gb reads F16 rows faster than
/// linear_tc rounds the F32 ones in its K loop, copy included (`legion lin-bench --convert`,
/// Maya-S at 2048-4096 tokens: the attention outputs 0.69-0.84x, attn_q_b ~0.8x, the FFN downs
/// 0.79-0.98x); both round with `__float2half_rn`, so the same bits. Only for products that take
/// linear_tc ([`q6k_f16_takes`] of the switches, `n`): others would round `x` for nothing.
pub(crate) fn xh_one_takes(cc12: bool, k: i64, m: i64, n: i64) -> bool {
    let gb10 = gb10_kernels(
        GB10_LINEAR.load(Ordering::Relaxed),
        GB10_GEMV.load(Ordering::Relaxed),
        n,
    );
    xh_one_pays(
        cc12,
        q6k_f16_takes(gb10, Q6K_F16.load(Ordering::Relaxed), n),
        k,
        m,
    )
}

/// [`xh_one_takes`] given whether the product takes linear_tc (`tc`).
pub(crate) fn xh_one_pays(cc12: bool, tc: bool, k: i64, m: i64) -> bool {
    cc12 && tc && m >= XH_ONE_MIN_ROWS && k % 512 == 0
}

static Q6K_F16: AtomicBool = AtomicBool::new(false);

/// Whether, with [`set_gb10_linear`]'s kernels, the Q6_K products of long prompt steps take
/// their weights through [`wf16`] (F16, the values the Q6_K GEMMs round to themselves) and then
/// linear_tc, the F16 GEMM (#594): the same bits; on the GB10 (Maya-S, 2048-token steps) a 5.7K
/// prompt 554 -> 583 tok/s with the copies included. The engine turns it on for sm_80+ (linear_tc).
pub fn set_q6k_f16(on: bool) {
    Q6K_F16.store(on, Ordering::Relaxed);
}

/// Q6_K weights `w` (`[k, m, h]`, `k` a multiple of 256) as F16: each value `d · scale` in F32
/// times `q - 32`, rounded to nearest, as the GEMMs round them in shared memory.
pub fn wf16<'a>(ctx: &'a Context, w: Tensor<'a>) -> Tensor<'a> {
    assert!(
        w.ty() == types::Q6_K && w.ne()[0] % 256 == 0 && w.ne()[3] == 1,
        "glm_wf16: Q6_K [k, m, h] in"
    );
    ctx.external(&WF16, types::F16, w.ne(), &[w])
}

/// Whether a [`linear`] of `n` tokens takes the GB10 kernels: `gb10` ([`set_gb10_linear`]), but a
/// decode GEMV (8 tokens or fewer) only with `gemv` ([`set_gb10_gemv`]).
pub(crate) fn gb10_kernels(gb10: bool, gemv: bool, n: i64) -> bool {
    gb10 && (gemv || n > 8)
}

static GB10_LINEAR: AtomicBool = AtomicBool::new(false);

/// Whether [`linear`]s built from now on take the CUDA kernels tuned on the GB10 (#567), each
/// computing the same values in the same order as the plain ones (the same bits): the Q6_K GEMV
/// staging each warp's weight rows through shared memory with 16-byte loads (rows 16-byte
/// aligned, `k` a multiple of 2048; ~165 -> ~220 GB/s on Maya's weights), and for steps of 256+
/// tokens the GEMM on 128 x 128 tiles, double-buffered (~2x), and on sm_80+ for Q6_K weights of
/// 1024+ rows on 128 x 256 tiles through cp.async and mma.sync (#571, ~1.3x more; [`linear_h`]
/// for inputs shared by several); and
/// [`mla_attn`] over 64+ tokens with 16 heads a block, each staged row read by 4x the heads
/// (#571: a warp's lanes over the positions, ~1.8x more); and [`hc_pre`]'s F32 mix rows over
/// 64+ tokens 4 tokens a block, each weight and stream read once; and the KDA conv and scan over
/// 64+ tokens without a stored state, 64 state rows a block (#571, ~3x); and the routed experts'
/// GEMMs on IQ2 weights ([`crate::model::moe_ffn::glm_kernels`]). The engine turns them on for
/// integrated GPUs, V100s and Ampere GPUs (compute capability 8.0 to 8.7, #1212; on those two
/// without the GEMV, [`set_gb10_gemv`]); `legion serve --glm-linear-plain` keeps the plain
/// kernels.
pub fn set_gb10_linear(on: bool) {
    GB10_LINEAR.store(on, Ordering::Relaxed);
}

/// See [`set_gb10_linear`].
pub fn gb10_linear() -> bool {
    GB10_LINEAR.load(Ordering::Relaxed)
}

static GB10_GEMV: AtomicBool = AtomicBool::new(true);

/// Whether [`set_gb10_linear`]'s kernels include the staged Q6_K decode GEMV (steps of 8 tokens or
/// fewer; on by default). Off on V100 (#574): there the staged GEMV takes ~11% longer than the
/// plain one (ncu, Maya-S decode), while the prompt kernels gain (8K-token prompt 565 -> 804 tok/s).
/// The same bits either way.
pub fn set_gb10_gemv(on: bool) {
    GB10_GEMV.store(on, Ordering::Relaxed);
}

/// [`bf16_gemv`] before the first read: the environment's choice.
const BF16_GEMV_UNSET: u8 = u8::MAX;

static BF16_GEMV: AtomicU8 = AtomicU8::new(BF16_GEMV_UNSET);

/// The BF16 decode GEMV of [`set_gb10_gemv`]'s steps (8 tokens or fewer, BF16 weights, the
/// node's `params[6]`; #560 r6, r8): 0 linear_mv; 1 linear_mv_bf16 (the warp's weight bytes
/// loaded contiguously a few steps ahead and handed to the summing lane through shared memory,
/// x in float4s); 2, the default, linear_mv_bf16 at 1 token and linear_mv_bf16x at 2-8 (x
/// copied to shared memory once per block, conflict free, instead of each warp reading it from
/// L1). Every lane sums the same values in the same order in all three: the same bits.
/// 3 (an opt-in) is 2 plus linear_mv_bf16x at 9-16 tokens on one matrix of 2048+-value rows,
/// which otherwise take the GEMM (x rounded to F16, sums by K tiles): outputs move.
/// `LEGION_DEBUG_GLM_BF16_GEMV` picks one for A/B runs.
pub fn bf16_gemv() -> u8 {
    let v = BF16_GEMV.load(Ordering::Relaxed);
    if v != BF16_GEMV_UNSET {
        return v;
    }
    let v = bf16_gemv_of(std::env::var("LEGION_DEBUG_GLM_BF16_GEMV").ok().as_deref());
    BF16_GEMV.store(v, Ordering::Relaxed);
    v
}

/// [`bf16_gemv`] given the environment's value.
pub(crate) fn bf16_gemv_of(env: Option<&str>) -> u8 {
    match env.map(str::trim).and_then(|s| s.parse::<u8>().ok()) {
        Some(v) if v <= BF16_GEMV_MAX => v,
        _ => BF16_GEMV_DEFAULT,
    }
}

/// [`bf16_gemv`]'s default.
pub(crate) const BF16_GEMV_DEFAULT: u8 = 2;

/// [`bf16_gemv`]'s largest value.
pub(crate) const BF16_GEMV_MAX: u8 = 3;

/// Sets [`bf16_gemv`] for [`linear`]s built from now on (`legion lin-bench --gemv`).
pub fn set_bf16_gemv(v: u8) {
    BF16_GEMV.store(v.min(BF16_GEMV_MAX), Ordering::Relaxed);
}

/// `x` (F32, contiguous) rounded to BF16 (to nearest, ties to even, as ggml's
/// `ggml_compute_fp32_to_bf16`): the input of the NVFP4 matmuls.
pub fn bf16<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let x = contiguous(x);
    assert_eq!(x.ty(), types::F32, "glm_bf16: F32 in");
    ctx.external(&BF16, types::BF16, x.ne(), &[x])
}

/// `x` (F32, contiguous) rounded to F16 (to nearest, ties to even): [`linear_h`]'s `xh`.
pub fn f16<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    assert!(
        x.ty() == types::F32 && x.is_contiguous(),
        "glm_f16: contiguous F32 in"
    );
    ctx.external(&F16, types::F16, x.ne(), &[x])
}

/// RMS norm over dim 0 of `x` (F32, contiguous), times `w` (`[ne0]`) if given.
pub fn rms_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Option<Tensor<'a>>,
    eps: f32,
) -> Tensor<'a> {
    let x = contiguous(x);
    let mut args = vec![x];
    args.extend(w);
    ctx.external_with(
        &RMS_NORM,
        types::F32,
        x.ne(),
        &args,
        &[eps.to_bits() as i32],
    )
}

/// Layer norm over dim 0 of `x` (F32, contiguous): `(x - mean) / sqrt(var + eps) · w + b`.
pub fn layer_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    b: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let x = contiguous(x);
    ctx.external_with(
        &LAYER_NORM,
        types::F32,
        x.ne(),
        &[x, w, b],
        &[eps.to_bits() as i32],
    )
}

/// What [`hc_pre`] returns: the mixed input of a sublayer and the weights [`hc_post`] reads.
#[derive(Clone, Copy)]
pub struct HcPre<'a> {
    /// F32 `[n_embd, n]`.
    pub y: Tensor<'a>,
    /// F32 `[hc, n]`: each stream's weight in `y`.
    pub pre: Tensor<'a>,
    /// F32 `[hc, n]`: each stream's weight of the sublayer output.
    pub post: Tensor<'a>,
    /// F32 `[hc, hc, n]`: `comb[d, s]` is the weight of stream `s` in new stream `d`
    /// (Sinkhorn-normalized).
    pub comb: Tensor<'a>,
}

/// The hyper-connection mixer before a sublayer (llama.cpp's `build_hc_pre` with its Sinkhorn
/// iterations): `x` F32 `[n_embd, hc, n]`, `f` the `[hc · n_embd, (2 + hc) · hc]` mix weights,
/// `scale` `[3]`, `base` `[(2 + hc) · hc]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    f: Tensor<'a>,
    scale: Tensor<'a>,
    base: Tensor<'a>,
    rms_eps: f32,
    hc_eps: f32,
    iters: u32,
) -> HcPre<'a> {
    let [e, hc, n, _] = x.ne();
    assert!(x.is_contiguous(), "glm_hc_pre: x contiguous");
    assert_eq!(f.ne()[..2], [e * hc, (2 + hc) * hc], "glm_hc_pre: weights");
    let total = e * n + (2 * hc + hc * hc) * n;
    let node = ctx.external_with(
        &HC_PRE,
        types::F32,
        [total, 1, 1, 1],
        &[x, f, scale, base],
        &[
            rms_eps.to_bits() as i32,
            hc_eps.to_bits() as i32,
            iters as i32,
            GB10_LINEAR.load(Ordering::Relaxed) as i32,
        ],
    );
    let fb = 4usize;
    let at = |i: i64| i as usize * fb;
    HcPre {
        y: ctx.view_2d(node, [e, n], at(e), 0),
        pre: ctx.view_2d(node, [hc, n], at(hc), at(e * n)),
        post: ctx.view_2d(node, [hc, n], at(hc), at(e * n + hc * n)),
        comb: ctx.view_3d(
            node,
            [hc, hc, n],
            [at(hc), at(hc * hc)],
            at(e * n + 2 * hc * n),
        ),
    }
}

/// Tokens up to which [`hc_pre_norm`] makes one node (else [`hc_pre`] and [`rms_norm`]), its F32
/// weights then taking CUDA's hc_mix_split_kernel, and that kernel's warps a row.
const HC_SPLIT_N: i64 = 8;
const HC_MIX_W: i64 = 8;

/// Where [`hc_pre_norm`]'s `h` starts in its node, in F32 elements: past `y` and the weights,
/// at a 32-element boundary. Its mix scratch follows `h`.
fn hc_h_off(e: i64, hc: i64, n: i64) -> i64 {
    (e * n + (2 * hc + hc * hc) * n + 31) / 32 * 32
}

/// [`hc_pre`] with the sublayer's norm in the same node (#606): also returns
/// `h = rms_norm(y, norm, norm_eps)`, the bits of [`rms_norm`] on [`HcPre::y`]. On CUDA one
/// kernel then does the weights, `y` and `h` (instead of three, two of them for one token).
/// Steps of more than [`HC_SPLIT_N`] tokens take the two ops as before: the one node keeps `y`
/// and `h` until [`hc_post`], which would raise a long step's compute buffer (and so take
/// expert cache slots, which shifts the cache's contents and the greedy text).
#[allow(clippy::too_many_arguments)]
pub fn hc_pre_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    f: Tensor<'a>,
    scale: Tensor<'a>,
    base: Tensor<'a>,
    rms_eps: f32,
    hc_eps: f32,
    iters: u32,
    norm: Tensor<'a>,
    norm_eps: f32,
) -> (HcPre<'a>, Tensor<'a>) {
    let [e, hc, n, _] = x.ne();
    assert!(x.is_contiguous(), "glm_hc_pre: x contiguous");
    assert_eq!(f.ne()[..2], [e * hc, (2 + hc) * hc], "glm_hc_pre: weights");
    assert_eq!(norm.ne()[0], e, "glm_hc_pre: norm weight");
    if n > HC_SPLIT_N {
        let p = hc_pre(ctx, x, f, scale, base, rms_eps, hc_eps, iters);
        let h = rms_norm(ctx, p.y, Some(norm), norm_eps);
        return (p, h);
    }
    let h_off = hc_h_off(e, hc, n);
    // CUDA's mix scratch past h: a row a token, or for F32 weights and a few tokens each row's
    // sums of hc_mix_split_kernel's HC_MIX_W warps, after theirs of squares (doubles).
    let split = f.ty() == types::F32 && n <= HC_SPLIT_N;
    let rows = (2 + hc) * hc;
    let stride = if split {
        HC_MIX_W * (rows + 2)
    } else {
        rows + 1
    };
    let total = (h_off + e * n + 31) / 32 * 32 + stride * n;
    let node = ctx.external_with(
        &HC_PRE,
        types::F32,
        [total, 1, 1, 1],
        &[x, f, scale, base, norm],
        &[
            rms_eps.to_bits() as i32,
            hc_eps.to_bits() as i32,
            iters as i32,
            GB10_LINEAR.load(Ordering::Relaxed) as i32,
            norm_eps.to_bits() as i32,
            split as i32,
            small_mv_param(),
        ],
    );
    let at = |i: i64| i as usize * 4;
    let pre = HcPre {
        y: ctx.view_2d(node, [e, n], at(e), 0),
        pre: ctx.view_2d(node, [hc, n], at(hc), at(e * n)),
        post: ctx.view_2d(node, [hc, n], at(hc), at(e * n + hc * n)),
        comb: ctx.view_3d(
            node,
            [hc, hc, n],
            [at(hc), at(hc * hc)],
            at(e * n + 2 * hc * n),
        ),
    };
    (pre, ctx.view_2d(node, [e, n], at(e), at(h_off)))
}

/// The hyper-connection update after a sublayer (llama.cpp's `build_hc_post`): new stream `d`
/// is `out · post[d] + Σ_s res[s] · comb[d, s]`, with `out` the sublayer output (F32
/// `[n_embd, n]`) plus `extra` if given (the shared expert's, added first). Returns F32
/// `[n_embd, hc, n]`.
pub fn hc_post<'a>(
    ctx: &'a Context,
    out: Tensor<'a>,
    extra: Option<Tensor<'a>>,
    res: Tensor<'a>,
    pre: &HcPre<'a>,
) -> Tensor<'a> {
    assert!(out.nb()[0] == 4 && res.is_contiguous());
    let mut args = vec![out, res, pre.post, pre.comb];
    args.extend(extra);
    ctx.external_with(
        &HC_POST,
        types::F32,
        res.ne(),
        &args,
        &[extra.is_some() as i32],
    )
}

/// The output head's input (llama.cpp's `build_hc_mean`, the crop and the final norm): the
/// mean of the streams of `x` (F32 `[n_embd, hc, n]`) at the tokens `rows` (I32 `[n_out]`,
/// all if `None`), RMS-normed times `w`. F32 `[n_embd, n_out]`.
pub fn hc_head<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    rows: Option<Tensor<'a>>,
    w: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let [e, _, n, _] = x.ne();
    let n_out = rows.map_or(n, |r| r.ne()[0]);
    let mut args = vec![x, w];
    args.extend(rows);
    ctx.external_with(
        &HC_HEAD,
        types::F32,
        [e, n_out, 1, 1],
        &args,
        &[eps.to_bits() as i32, rows.is_some() as i32],
    )
}

/// One of KDA's causal short convolutions (q, k or v) of a group of `n_seqs` sequences of
/// `n_tok` tokens each, then SiLU: `x` F32 `[c, n_seqs · n_tok]` or `[c, n_tok, n_seqs]` (rows
/// contiguous), `w` F32 `[d_conv, 1, c]`, `state` F32 `[d_conv - 1, channels, n_seqs]` (the
/// group's conv states, this projection's channels from `ch_off`), advanced in place. F32
/// `[c, n_seqs · n_tok]`.
pub fn kda_conv<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    state: Tensor<'a>,
    ch_off: i64,
    n_tok: i64,
) -> Tensor<'a> {
    conv_op(ctx, x, w, state, ch_off, n_tok, None)
}

/// [`kda_conv`] of a speculative verify group: the same outputs, the conv states left as they
/// were, and every token's input `x` stored in `store` (F32 `[c, n_tok, n_seqs]`, rows
/// contiguous) for the commit to replay.
pub fn kda_conv_verify<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    state: Tensor<'a>,
    ch_off: i64,
    n_tok: i64,
    store: Tensor<'a>,
) -> Tensor<'a> {
    conv_op(ctx, x, w, state, ch_off, n_tok, Some(store))
}

fn conv_op<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    state: Tensor<'a>,
    ch_off: i64,
    n_tok: i64,
    store: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let [c, a, b, _] = x.ne();
    let n = a * b;
    assert_eq!(x.nb()[0], 4, "glm_kda_conv: x rows contiguous");
    assert!(b == 1 || a == n_tok, "glm_kda_conv: x [c, n_tok, n_seqs]");
    assert_eq!(w.ne()[0], 4, "glm_kda_conv: width 4");
    assert_eq!(state.ne()[0], 3);
    assert_eq!(n % n_tok, 0);
    assert_eq!(
        n / n_tok,
        state.ne()[2],
        "glm_kda_conv: one state per sequence"
    );
    assert!(ch_off + c <= state.ne()[1]);
    let mut args = vec![x, w, state];
    if let Some(s) = store {
        assert!(
            s.nb()[0] == 4 && s.ne()[..3] == [c, n_tok, n / n_tok],
            "glm_kda_conv: store [c, n_tok, n_seqs]"
        );
        args.push(s);
    }
    ctx.external_with(
        &KDA_CONV,
        types::F32,
        [c, n, 1, 1],
        &args,
        &[
            ch_off as i32,
            n_tok as i32,
            store.is_some() as i32,
            GB10_LINEAR.load(Ordering::Relaxed) as i32,
            0,
        ],
    )
}

/// [`kda_conv`] of q, k and v at once (#676: one launch for three in a verify commit's
/// replay, each output the same): `x` and `w` the three projections' (each as
/// [`kda_conv`] takes it, the channels of `x[j]` from `j · c` in `state`). F32 `[c,
/// n_seqs · n_tok, 3]`.
pub fn kda_conv3<'a>(
    ctx: &'a Context,
    x: [Tensor<'a>; 3],
    w: [Tensor<'a>; 3],
    state: Tensor<'a>,
    n_tok: i64,
) -> Tensor<'a> {
    let [c, a, b, _] = x[0].ne();
    let n = a * b;
    for (xj, wj) in x.iter().zip(&w) {
        assert!(
            xj.ne() == x[0].ne() && xj.nb()[0] == 4,
            "glm_kda_conv: x rows contiguous, one shape"
        );
        assert_eq!(wj.ne()[0], 4, "glm_kda_conv: width 4");
    }
    assert!(b == 1 || a == n_tok, "glm_kda_conv: x [c, n_tok, n_seqs]");
    assert_eq!(state.ne()[0], 3);
    assert_eq!(n % n_tok, 0);
    assert_eq!(
        n / n_tok,
        state.ne()[2],
        "glm_kda_conv: one state per sequence"
    );
    assert!(3 * c <= state.ne()[1]);
    let [xq, xk, xv] = x;
    let [wq, wk, wv] = w;
    ctx.external_with(
        &KDA_CONV,
        types::F32,
        [c, n, 3, 1],
        &[xq, wq, state, xk, wk, xv, wv],
        &[0, n_tok as i32, 0, 0, 1],
    )
}

/// KDA's decay gate (lower-bounded): `lower · sigmoid(-(z + dt_bias) · a[head])`, F32 like
/// `z` (`[head_dim · n_head, n]`), `a` `[n_head]` (`-exp(A_log)`).
pub fn kda_gate<'a>(
    ctx: &'a Context,
    z: Tensor<'a>,
    dt_bias: Tensor<'a>,
    a: Tensor<'a>,
    lower: f32,
) -> Tensor<'a> {
    let z = contiguous(z);
    ctx.external_with(
        &KDA_GATE,
        types::F32,
        z.ne(),
        &[z, dt_bias, a],
        &[lower.to_bits() as i32],
    )
}

/// What [`linear_set`] does to one product's outputs.
#[derive(Clone, Copy)]
pub enum SetEpi<'a> {
    Plain,
    /// [`sigmoid`]'s.
    Sigmoid,
    /// [`kda_gate`]'s with these `dt_bias`, `a` and `lower`.
    KdaGate(Tensor<'a>, Tensor<'a>, f32),
}

/// Most products in one [`linear_set`] (the kernel's `LS_MAX`).
pub const SET_MAX: usize = 3;

/// Whether [`linear_set`] takes these weights over `n` tokens: Q8_0 matrices without a scale,
/// up to 8 tokens (decode and verify steps).
pub fn linear_set_takes(ws: &[Tensor<'_>], n: i64) -> bool {
    (1..=8).contains(&n)
        && ws
            .iter()
            .all(|w| w.ty() == types::Q8_0 && w.ne()[2] == 1 && w.ne()[0] % 32 == 0)
}

/// Whether [`linear_set`] takes these BF16 weights over `n` tokens with plain epilogues the way
/// a [`linear`] each would run (#1221 r7): one token, BF16 matrices without a scale, rows
/// 16-byte aligned, on the GB10 kernels with their BF16 GEMV ([`set_gb10_gemv`],
/// [`bf16_gemv`]), whose linear_mv_bf16 the set sums each row with; or 2-8 tokens with
/// [`bf16_gemv`] 2+ (#1221 r9), summed as linear_mv_bf16x sums them (the same bits as
/// linear_mv_bf16). The caller checks there is no weight scale.
pub fn linear_set_bf16_takes(ws: &[Tensor<'_>], n: i64) -> bool {
    (n == 1 || ((2..=8).contains(&n) && bf16_gemv() >= 2))
        && GB10_LINEAR.load(Ordering::Relaxed)
        && GB10_GEMV.load(Ordering::Relaxed)
        && bf16_gemv() >= 1
        && ws.iter().all(|w| bf16_set_matrix(*w))
}

/// [`linear_set_bf16_takes`]'s matrices: BF16, one of them, `k` a multiple of 32, rows 16-byte
/// aligned.
fn bf16_set_matrix(w: Tensor<'_>) -> bool {
    w.ty() == types::BF16 && w.ne()[2] == 1 && w.ne()[0] % 32 == 0 && w.nb()[1] % 16 == 0
}

/// Up to [`SET_MAX`] products `w @ x` (Q8_0 `w`, F32 `x` of `n` tokens, see
/// [`linear_set_takes`]; or BF16 `w` with plain epilogues, [`linear_set_bf16_takes`]) in one
/// node, each then through its epilogue: the bits of
/// [`linear`] (scale 1, no weight scale) and then [`sigmoid`] or [`kda_gate`] on its output, in
/// one launch on CUDA (#606). Returns F32 `[m, n]` per product.
pub fn linear_set<'a>(
    ctx: &'a Context,
    set: &[(Tensor<'a>, Tensor<'a>, SetEpi<'a>)],
) -> Vec<Tensor<'a>> {
    linear_set_with(ctx, set, small_mv_param())
}

/// [`linear_set`] with [`small_mv`]'s default bits, whatever `LEGION_DEBUG_GLM_SMALL_MV` says:
/// for other models' sets (Inkling-2, #1221), which GLM's A/B knob must not reach.
pub fn linear_set_default<'a>(
    ctx: &'a Context,
    set: &[(Tensor<'a>, Tensor<'a>, SetEpi<'a>)],
) -> Vec<Tensor<'a>> {
    linear_set_with(ctx, set, small_mv_default())
}

/// [`small_mv`]'s default bits on this GPU, whatever `LEGION_DEBUG_GLM_SMALL_MV` says.
fn small_mv_default() -> i32 {
    if GB10_GEMV.load(Ordering::Relaxed) {
        SMALL_MV_DEFAULT as i32
    } else {
        0
    }
}

/// How [`linear_set_deep`] lays Q8_0 rows over a warp (#1221 r10): `rows` 1 or 2 rows a warp,
/// `loads` 2 or 4 of a lane's blocks of each row loaded before it sums any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeepQ8 {
    pub rows: u8,
    pub loads: u8,
}

/// [`linear_set`] of Q8_0 weights with plain epilogues on CUDA's linear_set_q8d (#1221 r10,
/// Inkling-2's attention and head): more of a lane's loads in flight ([`DeepQ8`]), each row
/// summed as linear_mv_q8 sums it, the same bits. One product alone is a set too.
pub fn linear_set_deep<'a>(
    ctx: &'a Context,
    set: &[(Tensor<'a>, Tensor<'a>, SetEpi<'a>)],
    deep: DeepQ8,
) -> Vec<Tensor<'a>> {
    assert!(
        set.iter()
            .all(|s| s.0.ty() == types::Q8_0 && matches!(s.2, SetEpi::Plain)),
        "glm_linear_set: deep, Q8_0 and plain"
    );
    assert!(
        matches!(deep.rows, 1 | 2) && matches!(deep.loads, 2 | 4),
        "glm_linear_set: 1-2 rows a warp, 2 or 4 loads"
    );
    let bits = LS_DEEP
        | if deep.rows == 2 { LS_DEEP_ROWS2 } else { 0 }
        | if deep.loads == 4 { LS_DEEP_LOADS4 } else { 0 }
        | if set.iter().all(|s| leaf_data(s.0)) {
            LS_DEEP_EARLY
        } else {
            0
        };
    linear_set_with(ctx, set, small_mv_default() | bits)
}

/// The CUDA node's params[6] bits of [`linear_set_deep`] (above [`small_mv`]'s bits): on, two
/// rows a warp, four loads a lane, and the matrices read before the wait for the kernel ahead.
const LS_DEEP: i32 = 1 << 16;
const LS_DEEP_ROWS2: i32 = 1 << 17;
const LS_DEEP_LOADS4: i32 = 1 << 18;
const LS_DEEP_EARLY: i32 = 1 << 19;

/// Whether `t` is a leaf's data (the leaf or a view of one): a model weight in
/// [`linear_set_deep`]'s graphs (Inkling-2's attention and head), which no kernel writes, so its
/// kernel loads it before it waits for the kernel ahead. A computed matrix waits.
fn leaf_data(t: Tensor<'_>) -> bool {
    // SAFETY: a tensor of a live context, whose view source is null or a live tensor.
    unsafe {
        let r = &*t.raw();
        let leaf = |r: &ffi::ggml_tensor| r.op == ffi::ggml_op_GGML_OP_NONE;
        leaf(r)
            && (r.view_src.is_null() || (leaf(&*r.view_src) && (*r.view_src).view_src.is_null()))
    }
}

/// [`linear_set`] with `small` as the node's [`small_mv`] bits.
fn linear_set_with<'a>(
    ctx: &'a Context,
    set: &[(Tensor<'a>, Tensor<'a>, SetEpi<'a>)],
    small: i32,
) -> Vec<Tensor<'a>> {
    assert!(
        (1..=SET_MAX).contains(&set.len()),
        "glm_linear_set: 1-3 products"
    );
    let n: i64 = set[0].1.ne()[1..].iter().product();
    let ws: Vec<_> = set.iter().map(|s| s.0).collect();
    let bf16 = (1..=8).contains(&n) && ws.iter().all(|w| bf16_set_matrix(*w));
    assert!(
        linear_set_takes(&ws, n) || (bf16 && set.iter().all(|s| matches!(s.2, SetEpi::Plain))),
        "glm_linear_set: Q8_0 or BF16 (plain), up to 8 tokens"
    );
    let mut args = Vec::new();
    let mut params = [set.len() as i32, 0, 0, 0, 0, n as i32, small];
    let mut gate = None;
    for (g, &(w, x, epi)) in set.iter().enumerate() {
        assert_eq!(x.ne()[0], w.ne()[0], "glm_linear_set: x rows");
        assert!(
            x.ty() == types::F32 && x.nb()[0] == 4,
            "glm_linear_set: x F32 rows"
        );
        assert_eq!(
            x.ne()[1..].iter().product::<i64>(),
            n,
            "glm_linear_set: tokens"
        );
        assert!(x.ne()[2] == 1 || x.ne()[1] == 1 || x.is_contiguous());
        args.extend([w, x]);
        params[1 + g] = match epi {
            SetEpi::Plain => 0,
            SetEpi::Sigmoid => 1,
            SetEpi::KdaGate(dt, a, lower) => {
                assert!(gate.is_none(), "glm_linear_set: one kda_gate");
                assert_eq!(dt.ne()[0], w.ne()[1], "glm_linear_set: ssm_dt rows");
                gate = Some((dt, a));
                params[4] = lower.to_bits() as i32;
                2
            }
        };
    }
    if let Some((dt, a)) = gate {
        args.extend([dt, a]);
    }
    let total: i64 = set.iter().map(|s| s.0.ne()[1] * n).sum();
    let node = ctx.external_with(&LINEAR_SET, types::F32, [total, 1, 1, 1], &args, &params);
    let mut off = 0usize;
    set.iter()
        .map(|s| {
            let m = s.0.ne()[1];
            let v = ctx.view_2d(node, [m, n], m as usize * 4, off);
            off += (m * n) as usize * 4;
            v
        })
        .collect()
}

/// Whether [`glu_mv`] takes the gate and up weights `gate`, `up` over `n` tokens: Q6_K
/// matrices of one shape and layout, `k` a multiple of 256, up to 8 tokens (decode and verify
/// steps), and the plain kernels (not the GB10 ones, [`set_gb10_linear`]), whose
/// linear_mv_q6k it sums with; on the GB10 ones with [`SMALL_MV_GLU`], rows of one or two
/// whole stages at 1-4 tokens ([`glu_sp_takes`]), summed as their linear_mv_q6k_st.
pub fn glu_mv_takes(gate: Tensor<'_>, up: Tensor<'_>, n: i64) -> bool {
    glu_mv_fits(gate, up, n)
        && (!gb10_kernels(
            GB10_LINEAR.load(Ordering::Relaxed),
            GB10_GEMV.load(Ordering::Relaxed),
            n,
        ) || glu_sp_takes(gate, n))
}

/// Whether [`glu_mv`] of `gate`'s shape at `n` tokens takes CUDA's glu_mv_q6k_sp on the GB10
/// kernels: [`SMALL_MV_GLU`] in [`small_mv`] (on [`set_gb10_gemv`]'s GPUs), 1-4 tokens, rows
/// of one or two whole stages, contiguous, and at most [`GLU_SP_ROWS`] of them.
fn glu_sp_takes(gate: Tensor<'_>, n: i64) -> bool {
    let k = gate.ne()[0];
    gb10_kernels(
        GB10_LINEAR.load(Ordering::Relaxed),
        GB10_GEMV.load(Ordering::Relaxed),
        n,
    ) && small_mv_param() as u8 & SMALL_MV_GLU != 0
        && (1..=4).contains(&n)
        && glu_sp_shape(k, gate.ne()[1], gate.nb()[1] as i64)
}

/// The most rows [`glu_sp_takes`] takes: the shared expert's 2048 with room; the leading dense
/// FFNs' 12288 (four stages a row) stay on linear_mv_q6k_st.
const GLU_SP_ROWS: i64 = 4096;

/// [`glu_sp_takes`]'s shapes: `k` values a row in one or two whole stages, `m` rows of at most
/// [`GLU_SP_ROWS`], `nb1` bytes apart (contiguous).
pub(crate) fn glu_sp_shape(k: i64, m: i64, nb1: i64) -> bool {
    k > 0
        && k % (256 * Q6K_STAGE) == 0
        && k / (256 * Q6K_STAGE) <= 2
        && (1..=GLU_SP_ROWS).contains(&m)
        && nb1 == k / 256 * 210
}

/// [`glu_mv_takes`]'s shapes and types.
fn glu_mv_fits(gate: Tensor<'_>, up: Tensor<'_>, n: i64) -> bool {
    (1..=8).contains(&n)
        && [gate, up].iter().all(|w| w.ty() == types::Q6_K)
        && gate.ne() == up.ne()
        && gate.nb() == up.nb()
        && gate.ne()[2] == 1
        && gate.ne()[0] % 256 == 0
}

/// [`glu`] of [`linear`]`(gate, x)` and [`linear`]`(up, x)` (scale 1, no weight scale) in one
/// node, one launch on CUDA (#606, a dense FFN's or the shared expert's gate and up of a decode
/// step), the same bits; see [`glu_mv_takes`]. F32 `[m, n]` from `x` F32 `[k, n]`, contiguous.
pub fn glu_mv<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    limit: f32,
) -> Tensor<'a> {
    let n: i64 = x.ne()[1..].iter().product();
    assert!(glu_mv_fits(gate, up, n), "glm_glu_mv: Q6_K, up to 8 tokens");
    assert_eq!(x.ne()[0], gate.ne()[0], "glm_glu_mv: x rows");
    assert!(
        x.ty() == types::F32 && x.is_contiguous(),
        "glm_glu_mv: x F32, contiguous"
    );
    ctx.external_with(
        &GLU_MV,
        types::F32,
        [gate.ne()[1], n, 1, 1],
        &[gate, up, x],
        &[
            limit.to_bits() as i32,
            glu_sp_takes(gate, n) as i32,
            // the next kernel (the shared expert's down) launched early (#1172 r6)
            (small_mv_param() as u8 & SMALL_MV_TRIG != 0) as i32,
        ],
    )
}

/// Elementwise logistic sigmoid of `x` (F32, contiguous).
pub fn sigmoid<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let x = contiguous(x);
    ctx.external(&SIGMOID, types::F32, x.ne(), &[x])
}

/// KDA's gated delta rule over a group of `n_seqs` sequences of `n_tok` tokens: `q`, `k`, `v`
/// (after the convolutions), `g` (decay logs) F32 `[d, n_head, n_seqs · n_tok]`, `beta` F32
/// `[n_head, n_seqs · n_tok]`, `state` F32 `[d, d, n_head, n_seqs]` advanced in place. q and k
/// are l2-normed inside (`x / sqrt(Σx² + 1e-6)`, FLA's), q times `1 / sqrt(d)`. Per token and
/// head, with `S[j][i]` (value `j`, key `i`): `S *= exp(g)` over keys, `S += ((v - S k) β) ⊗ k`,
/// `out = S q`. F32 `[d, n_head, n_seqs · n_tok]`.
///
/// `g` may also be a view `[d, n_head, n_tok, n_seqs]` and `beta` one of `[n_head, n_tok,
/// n_seqs]` (rows contiguous; a commit's stored inputs, see [`kda_scan_verify`]).
#[allow(clippy::too_many_arguments)]
pub fn kda_scan<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
    n_tok: i64,
) -> Tensor<'a> {
    scan_op(ctx, [q, k, v, g, beta], state, n_tok, None)
}

/// [`kda_scan`] of a speculative verify group: the same outputs, the states left as they
/// were, and every token's decay logs and beta stored in `store` (`g` F32 `[d · n_head,
/// n_tok, n_seqs]`, `beta` F32 `[n_head, n_tok, n_seqs]`, rows contiguous) for the commit.
#[allow(clippy::too_many_arguments)]
pub fn kda_scan_verify<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
    n_tok: i64,
    store: [Tensor<'a>; 2],
) -> Tensor<'a> {
    scan_op(ctx, [q, k, v, g, beta], state, n_tok, Some(store))
}

fn scan_op<'a>(
    ctx: &'a Context,
    [q, k, v, g, beta]: [Tensor<'a>; 5],
    state: Tensor<'a>,
    n_tok: i64,
    store: Option<[Tensor<'a>; 2]>,
) -> Tensor<'a> {
    let [d, h, n, _] = q.ne();
    for t in [q, k, v] {
        assert!(
            t.is_contiguous() && t.ne()[..3] == [d, h, n],
            "glm_kda_scan: shapes"
        );
    }
    let n_seqs = n / n_tok;
    // `lead` values per token (rows contiguous), then the tokens: n of them, or n_tok of each
    // of n_seqs sequences (the kernels step by the strides).
    let per_token = |t: Tensor<'_>, lead: &[i64]| {
        let (ne, k) = (t.ne(), lead.len());
        ne[..k] == *lead
            && t.nb()[0] == 4
            && (k == 1 || t.nb()[1] == d as usize * 4)
            && ((ne[k] == n && ne[k + 1..].iter().all(|&x| x == 1))
                || (ne[k] == n_tok && ne[k + 1] == n_seqs))
    };
    assert!(per_token(g, &[d, h]), "glm_kda_scan: g");
    assert!(per_token(beta, &[h]), "glm_kda_scan: beta");
    assert_eq!(state.ne(), [d, d, h, n_seqs], "glm_kda_scan: state");
    assert_eq!(
        state.nb()[1],
        d as usize * 4,
        "glm_kda_scan: state rows contiguous"
    );
    let mut args = vec![q, k, v, g, beta, state];
    if let Some([sg, sb]) = store {
        assert!(
            sg.nb()[0] == 4 && sg.ne()[..3] == [d * h, n_tok, n_seqs],
            "glm_kda_scan: g store"
        );
        assert!(
            sb.nb()[0] == 4 && sb.ne()[..3] == [h, n_tok, n_seqs],
            "glm_kda_scan: beta store"
        );
        args.extend([sg, sb]);
    }
    ctx.external_with(
        &KDA_SCAN,
        types::F32,
        [d, h, n, 1],
        &args,
        &[
            n_tok as i32,
            store.is_some() as i32,
            GB10_LINEAR.load(Ordering::Relaxed) as i32,
            kda_split_param(),
        ],
    )
}

/// [`kda_split`] before the first read: the environment's choice (outside 0-2: every one is a
/// mode).
const KDA_SPLIT_UNSET: u8 = u8::MAX;

static KDA_SPLIT: AtomicU8 = AtomicU8::new(KDA_SPLIT_UNSET);

/// How many blocks [`kda_scan`]'s decode kernel (steps of fewer than 64 tokens: a 1-user verify
/// step's scan and every commit's, #1172 r8) gives each head on [`set_gb10_gemv`]'s GPUs, as a
/// log2: 0 one (1,024 threads, the kernel before r8), 1 two, 2 four (256 threads). Each block
/// holds its share of the head's state rows and computes the token's q and k norms itself, from
/// the same values in the same order: the same bits. `LEGION_DEBUG_GLM_KDA_SPLIT` picks it for
/// A/B runs.
pub fn kda_split() -> u8 {
    let v = KDA_SPLIT.load(Ordering::Relaxed);
    if v != KDA_SPLIT_UNSET {
        return v;
    }
    let v = kda_split_of(std::env::var("LEGION_DEBUG_GLM_KDA_SPLIT").ok().as_deref());
    KDA_SPLIT.store(v, Ordering::Relaxed);
    v
}

/// [`kda_split`] given the environment's value: 0-2, else the default.
pub(crate) fn kda_split_of(env: Option<&str>) -> u8 {
    match env.map(str::trim).and_then(|s| s.parse::<u8>().ok()) {
        Some(v) if v <= 2 => v,
        _ => KDA_SPLIT_DEFAULT,
    }
}

/// [`kda_split`]'s default: four blocks a head (Maya-S's 64 heads as 256 blocks, one wave on a
/// DGX Spark's 48 SMs instead of two).
pub(crate) const KDA_SPLIT_DEFAULT: u8 = 2;

/// Sets [`kda_split`] for nodes built from now on (`legion lin-bench --kda`).
pub fn set_kda_split(v: u8) {
    KDA_SPLIT.store(v.min(2), Ordering::Relaxed);
}

/// [`kda_split`] as the node's parameter: on the GPUs that take the GB10 decode kernels
/// ([`set_gb10_linear`] and [`set_gb10_gemv`]), else 0 (a V100's 80 SMs hold the 64 blocks).
fn kda_split_param() -> i32 {
    if GB10_LINEAR.load(Ordering::Relaxed) && GB10_GEMV.load(Ordering::Relaxed) {
        kda_split() as i32
    } else {
        0
    }
}

/// RMS norm of each `ne0` row of `o` (F32, contiguous) times `w`, times `sigmoid(gate)` if
/// given (same shape as `o`).
pub fn gated_rms<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    w: Tensor<'a>,
    gate: Option<Tensor<'a>>,
    eps: f32,
) -> Tensor<'a> {
    let o = contiguous(o);
    let mut args = vec![o, w];
    args.extend(gate.map(contiguous));
    ctx.external_with(
        &GATED_RMS,
        types::F32,
        o.ne(),
        &args,
        &[eps.to_bits() as i32, gate.is_some() as i32],
    )
}

/// SwiGLU with GLM's limit (ggml's `swiglu_clamp`): `silu(min(g, lim)) · clamp(u, -lim,
/// lim)`; `limit` 0 is plain SwiGLU. `g`, `u` F32 of one shape, rows contiguous.
pub fn glu<'a>(ctx: &'a Context, g: Tensor<'a>, u: Tensor<'a>, limit: f32) -> Tensor<'a> {
    let (g, u) = (contiguous(g), contiguous(u));
    assert_eq!(g.ne(), u.ne(), "glm_glu: shapes");
    ctx.external_with(&GLU, types::F32, g.ne(), &[g, u], &[limit.to_bits() as i32])
}

/// The router's choices: the `k` experts of the highest `sigmoid(logits) + bias` per token
/// (ties to the lower id), I32 `[k, n]`, from `logits` F32 `[n_expert, n]`, `bias` `[n_expert]`.
pub fn router_ids<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    bias: Tensor<'a>,
    k: i64,
) -> Tensor<'a> {
    let logits = contiguous(logits);
    ctx.external(
        &ROUTER_IDS,
        types::I32,
        [k, logits.ne()[1], 1, 1],
        &[logits, bias],
    )
}

/// [`router_ids`] with `bonus` added to the selection score of the experts the expert cache
/// holds: those whose entry in `table` (I32 `[n_expert]`, the layer's lookup table) is a VRAM
/// slot, below `n_slots`. `--glm-route-bias`, after Xolotl's `--route-bias`
/// (Indras-Mirror/Xolotl, MIT): more routings hit the cache, the weights
/// ([`router_w`]) stay unbiased.
pub fn router_ids_resident<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    bias: Tensor<'a>,
    table: Tensor<'a>,
    n_slots: usize,
    bonus: f32,
    k: i64,
) -> Tensor<'a> {
    let logits = contiguous(logits);
    ctx.external_with(
        &ROUTER_IDS,
        types::I32,
        [k, logits.ne()[1], 1, 1],
        &[logits, bias, table],
        &[n_slots as i32, bonus.to_bits() as i32],
    )
}

/// The router weights of the choices `ids`: `sigmoid(logits)` at them, divided by their sum
/// (at least 2^-14) if `norm`, times `scale`. F32 `[k, n]`.
pub fn router_w<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    ids: Tensor<'a>,
    norm: bool,
    scale: f32,
) -> Tensor<'a> {
    ctx.external_with(
        &ROUTER_W,
        types::F32,
        ids.ne(),
        &[contiguous(logits), ids],
        &[norm as i32, scale.to_bits() as i32],
    )
}

/// `Σ_j y[:, j, t] · w[j, t]` in order of `j`: `y` F32 `[e, k, n]`, `w` F32 `[k, n]`; F32
/// `[e, n]` (the routed experts' outputs weighted and summed).
pub fn weighted_sum<'a>(ctx: &'a Context, y: Tensor<'a>, w: Tensor<'a>) -> Tensor<'a> {
    let y = contiguous(y);
    let [e, _, n, _] = y.ne();
    ctx.external(&WEIGHTED_SUM, types::F32, [e, n, 1, 1], &[y, contiguous(w)])
}

/// Writes the rows of `x` (F32 `[d, n]`, rows contiguous) as F16 into rows `cells` (I32 `[n]`)
/// of `cache` (F16 `[w, rows]`) from element `off` on. Returns a token node (I32 `[1]`) to
/// order readers after the write.
pub fn cache_rows<'a>(
    ctx: &'a Context,
    cache: Tensor<'a>,
    x: Tensor<'a>,
    cells: Tensor<'a>,
    off: i64,
) -> Tensor<'a> {
    assert_eq!(cache.ty(), types::F16, "glm_cache_rows: F16 cache");
    assert!(off + x.ne()[0] <= cache.ne()[0]);
    assert_eq!(x.nb()[0], 4);
    assert_eq!(cells.ne()[0], x.ne()[1]);
    ctx.external_with(
        &CACHE_ROWS,
        types::I32,
        [1, 1, 1, 1],
        &[cache, x, cells],
        &[off as i32],
    )
}

/// The indexer's pool keys of the pools a step completes (llama.cpp's k-pool compressor):
/// for pool `p` with member cells `members[p·kpool..]` and channel `c`, the softmax over the
/// members of `gate[m][c] + ape[c, k]` weighs their keys `key[m][c]`; the result goes as F16
/// into the pooled slot (from element `2d`) of row `rep[p]` of `cache` (F16 `[3d, rows]`:
/// key | gate | pooled). Also returns the keys, F32 `[d, n_new]`. `after`: nodes that must run
/// first (the key and gate writes).
pub fn pool_keys<'a>(
    ctx: &'a Context,
    cache: Tensor<'a>,
    members: Tensor<'a>,
    rep: Tensor<'a>,
    ape: Tensor<'a>,
    after: &[Tensor<'a>],
) -> Tensor<'a> {
    let [d, kpool, ..] = ape.ne();
    assert_eq!(
        cache.ne()[0],
        3 * d,
        "glm_pool_keys: cache rows key | gate | pooled"
    );
    assert_eq!(rep.ty(), types::I64);
    let n_new = rep.ne()[0];
    assert_eq!(members.ne()[0], kpool * n_new);
    let mut args = vec![cache, members, rep, ape];
    args.extend_from_slice(after);
    // params[0]: the next kernel (MLA's q_a) launched once every block is past its wait
    // ([`SMALL_MV_TRIG`], #1172 r6)
    ctx.external_with(
        &POOL_KEYS,
        types::F32,
        [d, n_new, 1, 1],
        &args,
        &[(small_mv_param() as u8 & SMALL_MV_TRIG != 0) as i32],
    )
}

/// Sparse MLA attention of a group: query `t` (`q` F32 `[d, n_head, n_seqs · n_tok]`,
/// absorbed into the latent space) attends to the positions of selection row `t` of `sel`
/// (I32 `[1 + cap, n]`: count, positions), whose latents (keys and values both) are the F16
/// rows of `cache` (`[d, cells]`) found through the block table `table` (I32 `[width,
/// n_seqs]`). Softmax scale `scale`. F32 `[d, n_head, n]`. `after`: the cache write.
#[allow(clippy::too_many_arguments)]
pub fn mla_attn<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    cache: Tensor<'a>,
    table: Tensor<'a>,
    sel: Tensor<'a>,
    n_tok: i64,
    scale: f32,
    after: Tensor<'a>,
) -> Tensor<'a> {
    let flags = [
        GB10_LINEAR.load(Ordering::Relaxed),
        MLA_TC.load(Ordering::Relaxed),
    ];
    mla_node(
        ctx,
        [q, cache, table, sel, after],
        n_tok,
        scale,
        flags,
        MlaKernel::Auto,
    )
}

/// The CUDA kernel of [`mla_attn`]'s prompt steps (64+ tokens, with the GB10 kernels of
/// [`set_gb10_linear`] on and without [`set_mla_tc`]). `Tiled` and `Gb` give the same bits;
/// they differ in speed only (#862).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MlaKernel {
    /// The launcher's choice: [`MlaKernel::Gb`] on compute capability 12.x (the GB10, RTX 50)
    /// and Ampere (8.0 to 8.7, #1212), [`MlaKernel::Tiled`] on the others.
    Auto = 0,
    /// `mla_tiled_kernel`: 8 warps a block, each scoring 2 heads and then summing values.
    Tiled = 1,
    /// `mla_gb_kernel` (sm_80+): 4 score warps and 4 value warps a chunk apart, ~3x fewer
    /// shared-memory wavefronts.
    Gb = 2,
    /// `mla_tc_kernel` (sm_80+), as [`set_mla_tc`] takes it: q rounded to F16, other bits (for
    /// `legion mla-bench` only; the model's nodes take it from [`set_mla_tc`]).
    Tc = 3,
}

/// [`mla_attn`] as a prompt step takes it with the GB10 kernels on, on `kernel` (the
/// tensor-core kernel only for [`MlaKernel::Tc`]) (`legion
/// mla-bench` and the tests): `srcs` are q, cache, table, sel and after.
pub fn mla_attn_prompt<'a>(
    ctx: &'a Context,
    srcs: [Tensor<'a>; 5],
    n_tok: i64,
    scale: f32,
    kernel: MlaKernel,
) -> Tensor<'a> {
    mla_node(
        ctx,
        srcs,
        n_tok,
        scale,
        [true, kernel == MlaKernel::Tc],
        kernel,
    )
}

/// The [`mla_attn`] node; `flags`: the GB10 kernels, the tensor-core kernel.
fn mla_node<'a>(
    ctx: &'a Context,
    srcs: [Tensor<'a>; 5],
    n_tok: i64,
    scale: f32,
    flags: [bool; 2],
    kernel: MlaKernel,
) -> Tensor<'a> {
    let [q, cache, _, sel, _] = srcs;
    let [d, h, n, _] = q.ne();
    assert!(
        q.nb()[0] == 4 && q.nb()[1] == d as usize * 4,
        "glm_mla_attn: q rows"
    );
    assert_eq!(cache.ne()[0], d);
    assert_eq!(cache.ty(), types::F16);
    assert_eq!(sel.ne()[1], n);
    assert_eq!(n % n_tok, 0);
    ctx.external_with(
        &MLA_ATTN,
        types::F32,
        [d, h, n, 1],
        &srcs,
        &[
            n_tok as i32,
            scale.to_bits() as i32,
            crate::model::core::KV_BLOCK as i32,
            flags[0] as i32,
            flags[1] as i32,
            kernel as i32,
        ],
    )
}

static MLA_TC: AtomicBool = AtomicBool::new(false);

/// Whether [`mla_attn`]s built from now on run their prompt steps (64+ tokens, with the GB10
/// kernels of [`set_gb10_linear`] on) on tensor cores on sm_80+ (#575, `legion serve
/// --glm-mla-tc`): q rounded to F16, the scores and the weighted sums by `mma.sync` with F32
/// sums, the softmax weights rounded to F16. Not the default kernels' bits, so off by default.
pub fn set_mla_tc(on: bool) {
    MLA_TC.store(on, Ordering::Relaxed);
}

/// See [`set_mla_tc`].
pub fn mla_tc() -> bool {
    MLA_TC.load(Ordering::Relaxed)
}

static EXPERT_Q8: AtomicBool = AtomicBool::new(false);

/// Whether MoE layers built from now on run their prompt steps' IQ2 expert GEMMs (more than 16
/// tokens, with the GB10 kernels of [`set_gb10_linear`] on) on int8 tensor cores on sm_80+
/// (#603, `legion serve --glm-expert-q8`; [`crate::model::moe_ffn::glm_q8`]): the expert inputs
/// rounded to 8 bits with a scale per 32 values. Not the default kernels' bits, so off by
/// default.
pub fn set_expert_q8(on: bool) {
    EXPERT_Q8.store(on, Ordering::Relaxed);
}

/// See [`set_expert_q8`].
pub fn expert_q8() -> bool {
    EXPERT_Q8.load(Ordering::Relaxed)
}

/// The columns of `parts` (F32 `[d, n_i]` each, rows contiguous) side by side: `[d, Σ n_i]`.
pub fn join_cols<'a>(ctx: &'a Context, parts: &[Tensor<'a>]) -> Tensor<'a> {
    match parts {
        [one] => *one,
        // ggml takes fewer than GGML_MAX_SRC sources a node: 10 parts (e.g. 10 sequence groups
        // of a decode step) split too (#1174: aborted at 10 of 16 users).
        _ if parts.len() >= ffi::GGML_MAX_SRC as usize => {
            let (a, b) = parts.split_at(parts.len() / 2);
            join_cols(ctx, &[join_cols(ctx, a), join_cols(ctx, b)])
        }
        _ => {
            let d = parts[0].ne()[0];
            let n: i64 = parts
                .iter()
                .map(|p| p.ne()[1..].iter().product::<i64>())
                .sum();
            ctx.external(&JOIN_COLS, types::F32, [d, n, 1, 1], parts)
        }
    }
}

/// Each column's largest entry's index (the first of ties): I32 `[n]` from F32 `[v, n]`.
pub fn argmax<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    ctx.external(&ARGMAX, types::I32, [x.ne()[1], 1, 1, 1], &[contiguous(x)])
}

/// Chunks a column of [`argmax_wide`] is split into on CUDA (argmax_part_kernel's CTAs).
const ARGMAX_SPLIT: i64 = 32;

/// [`argmax`] with CUDA's search spread over [`ARGMAX_SPLIT`] CTAs a column, then their bests
/// compared (#606): one CTA took ~190 us over GLM's vocabulary on a V100. The node is I32
/// `[n, 1 + 2 · ARGMAX_SPLIT]`, row 0 the result (returned as a view), the rest the chunks'
/// values and indices. The same index as [`argmax`] for logits without NaNs (the first of the
/// largest).
pub fn argmax_wide<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    argmax_wide_ids(ctx, x, None)
}

/// [`argmax_wide`] of logits over the rows `ids` (I32 `[v]`, ascending token ids: the MTP
/// draft head's picked rows, #1172) where given: each column's pick is `ids[i]` for its largest
/// entry `i` (the first of ties, so the lowest id, as over the whole vocabulary).
pub fn argmax_wide_ids<'a>(ctx: &'a Context, x: Tensor<'a>, ids: Option<Tensor<'a>>) -> Tensor<'a> {
    let n = x.ne()[1];
    let x = contiguous(x);
    let node = match ids {
        Some(ids) => {
            assert!(
                ids.ty() == types::I32 && ids.elements() as i64 == x.ne()[0],
                "argmax ids: {:?} for {:?}",
                ids.ne(),
                x.ne()
            );
            ctx.external_with(
                &ARGMAX,
                types::I32,
                [n, 1 + 2 * ARGMAX_SPLIT, 1, 1],
                &[x, ids],
                &[ARGMAX_SPLIT as i32],
            )
        }
        None => ctx.external_with(
            &ARGMAX,
            types::I32,
            [n, 1 + 2 * ARGMAX_SPLIT, 1, 1],
            &[x],
            &[ARGMAX_SPLIT as i32],
        ),
    };
    ctx.view_1d(node, n, 0)
}

/// Each column's largest softmax probability, `1 / Σ exp(x - max)`: F32 `[n]` from F32
/// `[v, n]` (an MTP draft's confidence).
pub fn pmax<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    ctx.external(&PMAX, types::F32, [x.ne()[1], 1, 1, 1], &[contiguous(x)])
}

/// The MTP head's input (llama.cpp's `graph_mtp`): per token, the RMS-normed embedding `e`
/// times `enorm` stacked over the RMS-normed hidden row `h` times `hnorm`, F32 `[2 · d, n]`
/// from F32 `[d, n]` each (contiguous).
pub fn nextn_in<'a>(
    ctx: &'a Context,
    e: Tensor<'a>,
    h: Tensor<'a>,
    enorm: Tensor<'a>,
    hnorm: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let [d, n, ..] = e.ne();
    assert_eq!(h.ne()[..2], [d, n], "glm_nextn_in: shapes");
    ctx.external_with(
        &NEXTN_IN,
        types::F32,
        [2 * d, n, 1, 1],
        &[contiguous(e), contiguous(h), enorm, hnorm],
        &[eps.to_bits() as i32],
    )
}

/// Columns `rows` (I32 `[m]`) of `x` (F32 `[d, n]`, rows contiguous): F32 `[d, m]`.
pub fn get_cols<'a>(ctx: &'a Context, x: Tensor<'a>, rows: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.nb()[0], 4, "glm_get_cols: rows contiguous");
    ctx.external(
        &GET_COLS,
        types::F32,
        [x.ne()[0], rows.ne()[0], 1, 1],
        &[x, rows],
    )
}

/// `a + b`, elementwise (F32, contiguous, same shape).
pub fn add<'a>(ctx: &'a Context, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(a.ne(), b.ne(), "glm_add: shapes");
    ctx.external(&ADD, types::F32, a.ne(), &[contiguous(a), contiguous(b)])
}

/// `t` itself when contiguous; ops here never take a ggml copy, so a strided input is a bug.
fn contiguous(t: Tensor<'_>) -> Tensor<'_> {
    assert!(t.is_contiguous(), "glm5next op input must be contiguous");
    t
}

// ---- CPU references ----

/// The byte stride between sequences of `t`, whose tokens are dim `k` (stride `nb[k]`):
/// `[.., n_tok, n_seqs]` views step by `nb[k + 1]`, token-major `[.., n]` ones by `n_tok`
/// tokens. (The CUDA kernels take the same rule.)
fn seq_stride(t: &ffi::ggml_tensor, k: usize, n_tok: usize, n_seqs: usize) -> usize {
    if k < 3 && t.ne[k] as usize == n_tok && t.ne[k + 1] as usize == n_seqs {
        t.nb[k + 1]
    } else {
        n_tok * t.nb[k]
    }
}

/// The rows `ith` of `nth` threads take out of `n`.
fn part(n: usize, ith: usize, nth: usize) -> std::ops::Range<usize> {
    let per = n.div_ceil(nth.max(1));
    (ith * per).min(n)..((ith + 1) * per).min(n)
}

fn f32_param(p: i32) -> f32 {
    f32::from_bits(p as u32)
}

/// Element `i` of `t`'s data as `T` (contiguous tensors).
unsafe fn at<T: Copy>(t: &ffi::ggml_tensor, i: usize) -> T {
    unsafe { *(t.data as *const T).add(i) }
}

/// `n` F32 values at byte offset `off` of `t`'s data.
unsafe fn f32s(t: &ffi::ggml_tensor, off: usize, n: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts((t.data as *const u8).add(off) as *const f32, n) }
}

/// The whole (contiguous) F32 data of `t`.
unsafe fn all_f32(t: &ffi::ggml_tensor) -> &[f32] {
    let n = (0..4).map(|i| t.ne[i] as usize).product();
    unsafe { f32s(t, 0, n) }
}

/// The destination's data as F32 (`dst` as a pointer: the slice is ggml's buffer, not `dst`).
unsafe fn out_f32<'a>(dst: *const ffi::ggml_tensor, n: usize) -> &'a mut [f32] {
    unsafe { std::slice::from_raw_parts_mut((*dst).data as *mut f32, n) }
}

/// Row `r` of matrix `h` of `w` as F32 (`ne0` values), whatever its type.
unsafe fn weight_row(w: &ffi::ggml_tensor, r: usize, h: usize) -> Vec<f32> {
    let k = w.ne[0] as usize;
    unsafe {
        let p = (w.data as *const u8).add(r * w.nb[1] + h * w.nb[2]);
        if w.type_ == types::F32 {
            return std::slice::from_raw_parts(p as *const f32, k).to_vec();
        }
        let bytes = std::slice::from_raw_parts(p, crate::ggml::row_size(w.type_, k));
        crate::ggml::dequantize(w.type_, bytes, k)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigm(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `1 / sqrt(mean(x²) + eps)`, the sum in f64 as ggml's.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    1.0 / ((sum / x.len() as f64) as f32 + eps).sqrt()
}

fn f16_to_f32(h: u16) -> f32 {
    // SAFETY: plain conversion.
    unsafe { ffi::ggml_fp16_to_fp32(h) }
}

fn f32_to_f16(x: f32) -> u16 {
    // SAFETY: plain conversion.
    unsafe { ffi::ggml_fp32_to_fp16(x) }
}

/// `n` F16 values at byte offset `off` of `t`'s data, as F32.
unsafe fn f16_row(t: &ffi::ggml_tensor, off: usize, n: usize) -> Vec<f32> {
    unsafe {
        let p = (t.data as *const u8).add(off) as *const u16;
        (0..n).map(|i| f16_to_f32(*p.add(i))).collect()
    }
}

unsafe fn cpu_embed(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (w, tok) = (&*dst.src[0], &*dst.src[1]);
        let (e, hc, n) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let out = out_f32(dst, e * hc * n);
        for t in part(n, ith, nth) {
            let row = weight_row(w, at::<i32>(tok, t) as usize, 0);
            for s in 0..hc {
                out[(t * hc + s) * e..][..e].copy_from_slice(&row);
            }
        }
    }
}

unsafe fn cpu_linear(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let p = external_params(dst);
        let scale = f32_param(p[0]);
        let ws = (p[1] != 0).then(|| all_f32(&*dst.src[2]));
        let (k, m, h) = (w.ne[0] as usize, w.ne[1] as usize, w.ne[2] as usize);
        // Input row of (matrix i, token t).
        let (n, xrow): (usize, Box<dyn Fn(usize, usize) -> usize>) = if h == 1 {
            let n = dst.ne[1] as usize;
            let nb1 = x.nb[1];
            (n, Box::new(move |_, t| t * nb1))
        } else {
            let (nb1, nb2) = (x.nb[1], x.nb[2]);
            (dst.ne[2] as usize, Box::new(move |i, t| i * nb1 + t * nb2))
        };
        // linear_io's F16 rows in and F16 outputs out (#862)
        let x16 = x.type_ == types::F16;
        let out16 = dst.type_ == types::F16;
        // One weight row dequantized once, against every token.
        for idx in part(m * h, ith, nth) {
            let (i, r) = (idx / m, idx % m);
            let wr = weight_row(w, r, i);
            let s = match ws {
                Some(ws) if ws.len() > 1 => ws[r] * scale,
                Some(ws) => ws[0] * scale,
                None => scale,
            };
            for t in 0..n {
                let y = if x16 {
                    let xr = std::slice::from_raw_parts(
                        (x.data as *const u8).add(xrow(i, t)) as *const u16,
                        k,
                    );
                    let xr: Vec<f32> = xr.iter().map(|&v| f16_to_f32(v)).collect();
                    dot(&wr, &xr) * s
                } else {
                    dot(&wr, f32s(x, xrow(i, t), k)) * s
                };
                let o = (t * h + i) * m + r;
                if out16 {
                    *(dst.data as *mut u16).add(o) = f32_to_f16(y);
                } else {
                    *(dst.data as *mut f32).add(o) = y;
                }
            }
        }
    }
}

unsafe fn cpu_linear_set(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let p = external_params(dst);
        let (nseg, n) = (p[0] as usize, p[5] as usize);
        let gate = (1..=nseg).any(|g| p[g] == 2).then(|| {
            (
                all_f32(&*dst.src[2 * nseg]),
                all_f32(&*dst.src[2 * nseg + 1]),
            )
        });
        let lower = f32_param(p[4]);
        let out = dst.data as *mut f32;
        let mut off = 0usize;
        for g in 0..nseg {
            let (w, x) = (&*dst.src[2 * g], &*dst.src[2 * g + 1]);
            let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
            // cpu_linear's token rows (h = 1).
            let nbt = if x.ne[1] == 1 && x.ne[2] > 1 {
                x.nb[2]
            } else {
                x.nb[1]
            };
            for r in part(m, ith, nth) {
                let wr = weight_row(w, r, 0);
                for t in 0..n {
                    let z = dot(&wr, f32s(x, t * nbt, k));
                    *out.add(off + t * m + r) = match p[1 + g] {
                        1 => sigm(z),
                        2 => {
                            let (dt, a) = gate.unwrap();
                            sigm(-((z + dt[r]) * a[r / (dt.len() / a.len())])) * lower
                        }
                        _ => z,
                    };
                }
            }
            off += m * n;
        }
    }
}

/// ggml's `ggml_compute_fp32_to_bf16`.
pub(crate) fn to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    if u & 0x7fff_ffff > 0x7f80_0000 {
        // NaN: keep it quiet.
        return ((u >> 16) | 64) as u16;
    }
    ((u + (0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

unsafe fn cpu_bf16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = all_f32(&*dst.src[0]);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut u16, x.len());
        for i in part(x.len(), ith, nth) {
            out[i] = to_bf16(x[i]);
        }
    }
}

unsafe fn cpu_f16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = all_f32(&*dst.src[0]);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut ffi::ggml_fp16_t, x.len());
        for i in part(x.len(), ith, nth) {
            out[i] = ffi::ggml_fp32_to_fp16(x[i]);
        }
    }
}

unsafe fn cpu_wf16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let w = &*dst.src[0];
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let rows = m * w.ne[2] as usize;
        let out = std::slice::from_raw_parts_mut(dst.data as *mut ffi::ggml_fp16_t, rows * k);
        for r in part(rows, ith, nth) {
            let v = weight_row(w, r % m, r / m);
            for (o, x) in out[r * k..][..k].iter_mut().zip(v) {
                *o = ffi::ggml_fp32_to_fp16(x);
            }
        }
    }
}

/// [`rms_norm`]'s CPU reference, which Inkling-2's `ink_rms` shares (the same math).
pub(crate) unsafe fn cpu_rms_norm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = &*dst.src[0];
        let w = (!dst.src[1].is_null()).then(|| all_f32(&*dst.src[1]));
        let eps = f32_param(external_params(dst)[0]);
        let d = x.ne[0] as usize;
        let rows = all_f32(x).len() / d;
        let (xs, out) = (all_f32(x), out_f32(dst, rows * d));
        for r in part(rows, ith, nth) {
            let xr = &xs[r * d..][..d];
            let s = rms_scale(xr, eps);
            for (i, o) in out[r * d..][..d].iter_mut().enumerate() {
                *o = xr[i] * s * w.map_or(1.0, |w| w[i]);
            }
        }
    }
}

unsafe fn cpu_layer_norm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, w, b) = (
            all_f32(&*dst.src[0]),
            all_f32(&*dst.src[1]),
            all_f32(&*dst.src[2]),
        );
        let eps = f32_param(external_params(dst)[0]);
        let d = w.len();
        let rows = x.len() / d;
        let out = out_f32(dst, x.len());
        for r in part(rows, ith, nth) {
            let xr = &x[r * d..][..d];
            let mean = (xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64) as f32;
            let var: f64 = xr.iter().map(|&v| ((v - mean) as f64).powi(2)).sum();
            let s = 1.0 / ((var / d as f64) as f32 + eps).sqrt();
            for i in 0..d {
                out[r * d + i] = (xr[i] - mean) * s * w[i] + b[i];
            }
        }
    }
}

unsafe fn cpu_hc_pre(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, f) = (&*dst.src[0], &*dst.src[1]);
        let (scale, base) = (all_f32(&*dst.src[2]), all_f32(&*dst.src[3]));
        let p = external_params(dst);
        let (rms_eps, hc_eps, iters) = (f32_param(p[0]), f32_param(p[1]), p[2] as usize);
        let (e, hc, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let n_mix = (2 + hc) * hc;
        // hc_pre_norm's norm weight and h (the mix scratch past h is the CUDA kernels').
        let norm = (!dst.src[4].is_null()).then(|| (all_f32(&*dst.src[4]), f32_param(p[4])));
        let h_off = hc_h_off(e as i64, hc as i64, n as i64) as usize;
        let len = if norm.is_some() {
            h_off + e * n
        } else {
            e * n + (2 * hc + hc * hc) * n
        };
        let out = out_f32(dst, len);
        let (out, h_all) = out.split_at_mut(e * n + (2 * hc + hc * hc) * n);
        let h_skip = h_off - out.len();
        let (y, rest) = out.split_at_mut(e * n);
        let (pre_all, rest) = rest.split_at_mut(hc * n);
        let (post_all, comb_all) = rest.split_at_mut(hc * n);
        let xs = all_f32(x);
        let range = part(n, ith, nth);
        if range.is_empty() {
            return;
        }
        let rows: Vec<Vec<f32>> = (0..n_mix).map(|r| weight_row(f, r, 0)).collect();
        for t in range {
            let flat = &xs[t * e * hc..][..e * hc];
            let s = rms_scale(flat, rms_eps);
            let normed: Vec<f32> = flat.iter().map(|v| v * s).collect();
            let mix: Vec<f32> = rows.iter().map(|r| dot(r, &normed)).collect();
            let pre = &mut pre_all[t * hc..][..hc];
            let post = &mut post_all[t * hc..][..hc];
            for h in 0..hc {
                pre[h] = sigm(mix[h] * scale[0] + base[h]) + hc_eps;
                post[h] = sigm(mix[hc + h] * scale[1] + base[hc + h]) * 2.0;
            }
            // comb[d + hc·s]: the weight of source stream s in destination d.
            let comb = &mut comb_all[t * hc * hc..][..hc * hc];
            for (i, c) in comb.iter_mut().enumerate() {
                *c = mix[2 * hc + i] * scale[2] + base[2 * hc + i];
            }
            sinkhorn(comb, hc, hc_eps, iters);
            let yt = &mut y[t * e..][..e];
            for (i, v) in yt.iter_mut().enumerate() {
                let mut acc = flat[i] * pre[0];
                for h in 1..hc {
                    acc += flat[h * e + i] * pre[h];
                }
                *v = acc;
            }
            if let Some((w, eps)) = norm {
                // cpu_rms_norm's arithmetic.
                let s = rms_scale(yt, eps);
                let ht = &mut h_all[h_skip + t * e..][..e];
                for (i, o) in ht.iter_mut().enumerate() {
                    *o = yt[i] * s * w[i];
                }
            }
        }
    }
}

/// llama.cpp's `build_hc_sinkhorn` on one token's `comb` (`[dst, src]`, dst fastest): softmax
/// over dst per src, `+ eps`, normalize each dst over the srcs, then `iters - 1` rounds of
/// (each src over the dsts, each dst over the srcs), each sum `+ eps`.
pub(crate) fn sinkhorn(c: &mut [f32], hc: usize, eps: f32, iters: usize) {
    for s in 0..hc {
        let col = &mut c[s * hc..][..hc];
        let max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f64;
        for v in col.iter_mut() {
            *v = (*v - max).exp();
            sum += *v as f64;
        }
        let inv = (1.0 / sum) as f32;
        for v in col.iter_mut() {
            *v = *v * inv + eps;
        }
    }
    let norm_cols = |c: &mut [f32]| {
        for d in 0..hc {
            let sum: f32 = (0..hc).map(|s| c[d + hc * s]).sum::<f32>() + eps;
            for s in 0..hc {
                c[d + hc * s] /= sum;
            }
        }
    };
    let norm_rows = |c: &mut [f32]| {
        for s in 0..hc {
            let sum: f32 = c[s * hc..][..hc].iter().sum::<f32>() + eps;
            for v in &mut c[s * hc..][..hc] {
                *v /= sum;
            }
        }
    };
    norm_cols(c);
    for _ in 1..iters {
        norm_rows(c);
        norm_cols(c);
    }
}

unsafe fn cpu_hc_post(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (o, res, post, comb) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let extra = (external_params(dst)[0] != 0).then(|| &*dst.src[4]);
        let (e, hc, n) = (res.ne[0] as usize, res.ne[1] as usize, res.ne[2] as usize);
        let rs = all_f32(res);
        let out = out_f32(dst, e * hc * n);
        for t in part(n, ith, nth) {
            let ot = f32s(o, t * o.nb[1], e);
            let xt = extra.map(|x| f32s(x, t * x.nb[1], e));
            let pw = f32s(post, t * post.nb[1], hc);
            let cw = f32s(comb, t * comb.nb[2], hc * hc);
            for d in 0..hc {
                let dst_row = &mut out[(t * hc + d) * e..][..e];
                for (i, v) in dst_row.iter_mut().enumerate() {
                    let a = match xt {
                        Some(x) => ot[i] + x[i],
                        None => ot[i],
                    };
                    let mut acc = a * pw[d];
                    for s in 0..hc {
                        acc += rs[(t * hc + s) * e + i] * cw[d + hc * s];
                    }
                    *v = acc;
                }
            }
        }
    }
}

unsafe fn cpu_hc_head(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, w) = (&*dst.src[0], all_f32(&*dst.src[1]));
        let p = external_params(dst);
        let eps = f32_param(p[0]);
        let rows = (p[1] != 0).then(|| &*dst.src[2]);
        let (e, hc) = (x.ne[0] as usize, x.ne[1] as usize);
        let n_out = dst.ne[1] as usize;
        let xs = all_f32(x);
        let out = out_f32(dst, e * n_out);
        let inv = 1.0 / hc as f32;
        for r in part(n_out, ith, nth) {
            let t = rows.map_or(r, |rw| at::<i32>(rw, r) as usize);
            let mean: Vec<f32> = (0..e)
                .map(|i| {
                    let mut acc = xs[t * hc * e + i];
                    for s in 1..hc {
                        acc += xs[(t * hc + s) * e + i];
                    }
                    acc * inv
                })
                .collect();
            let s = rms_scale(&mean, eps);
            for i in 0..e {
                out[r * e + i] = mean[i] * s * w[i];
            }
        }
    }
}

unsafe fn cpu_kda_conv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, w, st) = (&*dst.src[0], all_f32(&*dst.src[1]), &*dst.src[2]);
        let p = external_params(dst);
        let (ch_off, n_tok, verify) = (p[0] as usize, p[1] as usize, p[2] != 0);
        let (c, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        if p[4] != 0 {
            // q, k and v at once (kda_conv3): each projection as its own op computes it.
            for j in 0..3 {
                let (xi, wi) = if j == 0 {
                    (0, 1)
                } else {
                    (2 * j + 1, 2 * j + 2)
                };
                conv_rows(
                    (&*dst.src[xi], all_f32(&*dst.src[wi]), st),
                    (j * c, n_tok, None),
                    (dst.data as *mut f32).add(j * c * n),
                    (c, n),
                    (ith, nth),
                );
            }
            return;
        }
        let store = verify.then(|| &*dst.src[3]);
        conv_rows(
            (x, w, st),
            (ch_off, n_tok, store),
            dst.data as *mut f32,
            (c, n),
            (ith, nth),
        );
    }
}

/// [`cpu_kda_conv`] of one projection: `x`'s `c` channels to `out` (`[c, n]`), the state's
/// from `ch_off`.
#[allow(clippy::type_complexity)]
unsafe fn conv_rows(
    (x, w, st): (&ffi::ggml_tensor, &[f32], &ffi::ggml_tensor),
    (ch_off, n_tok, store): (usize, usize, Option<&ffi::ggml_tensor>),
    out: *mut f32,
    (c, n): (usize, usize),
    (ith, nth): (usize, usize),
) {
    unsafe {
        let verify = store.is_some();
        let n_seqs = n / n_tok;
        let seq_nb = seq_stride(x, 1, n_tok, n_seqs);
        for ch in part(c, ith, nth) {
            let wc = &w[ch * 4..][..4];
            for s in 0..n_seqs {
                let sp = |k: usize| {
                    (st.data as *mut u8).add(k * st.nb[0] + (ch_off + ch) * st.nb[1] + s * st.nb[2])
                        as *mut f32
                };
                let mut win: Vec<f32> = (0..3).map(|k| *sp(k)).collect();
                for t in 0..n_tok {
                    let col = s * n_tok + t;
                    let xv = *((x.data as *const u8).add(s * seq_nb + t * x.nb[1]) as *const f32)
                        .add(ch);
                    win.push(xv);
                    if let Some(sx) = store {
                        *((sx.data as *mut u8).add(s * sx.nb[2] + t * sx.nb[1]) as *mut f32)
                            .add(ch) = xv;
                    }
                    let mut acc = 0f32;
                    for k in 0..4 {
                        acc += win[t + k] * wc[k];
                    }
                    *out.add(col * c + ch) = silu(acc);
                }
                // A verify group leaves the state for the commit.
                if !verify {
                    for k in 0..3 {
                        *sp(k) = win[n_tok + k];
                    }
                }
            }
        }
    }
}

unsafe fn cpu_kda_gate(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (z, dt, a) = (
            all_f32(&*dst.src[0]),
            all_f32(&*dst.src[1]),
            all_f32(&*dst.src[2]),
        );
        let lower = f32_param(external_params(dst)[0]);
        let c = dt.len();
        let head = c / a.len();
        let out = out_f32(dst, z.len());
        for i in part(z.len(), ith, nth) {
            let ch = i % c;
            out[i] = sigm(-((z[i] + dt[ch]) * a[ch / head])) * lower;
        }
    }
}

unsafe fn cpu_sigmoid(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = all_f32(&*dst.src[0]);
        let out = out_f32(dst, x.len());
        for i in part(x.len(), ith, nth) {
            out[i] = sigm(x[i]);
        }
    }
}

/// `x / sqrt(Σx² + 1e-6)` (llama.cpp's `build_gdn_l2_norm`, FLA's l2 norm).
fn l2_normed(x: &[f32], scale: f32) -> Vec<f32> {
    let sum: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let s = scale / ((sum as f32) + 1e-6).sqrt();
    x.iter().map(|v| v * s).collect()
}

unsafe fn cpu_kda_scan(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let src = |i: usize| all_f32(&*dst.src[i]);
        let (q, k, v) = (src(0), src(1), src(2));
        let (g, beta) = (&*dst.src[3], &*dst.src[4]);
        let st = &*dst.src[5];
        let p = external_params(dst);
        let (n_tok, verify) = (p[0] as usize, p[1] != 0);
        let (d, h, n) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let n_seqs = n / n_tok;
        let (g_seq, b_seq) = (
            seq_stride(g, 2, n_tok, n_seqs),
            seq_stride(beta, 1, n_tok, n_seqs),
        );
        let store = verify.then(|| (&*dst.src[6], &*dst.src[7]));
        let qs = 1.0 / (d as f32).sqrt();
        let out = dst.data as *mut f32;
        for idx in part(n_seqs * h, ith, nth) {
            let (s, hd) = (idx / h, idx % h);
            let sp = (st.data as *mut u8).add(hd * st.nb[2] + s * st.nb[3]) as *mut f32;
            // A verify group works on a copy: the state stays for the commit.
            let mut copy;
            let state = if verify {
                copy = std::slice::from_raw_parts(sp, d * d).to_vec();
                &mut copy[..]
            } else {
                std::slice::from_raw_parts_mut(sp, d * d)
            };
            for t in 0..n_tok {
                let col = s * n_tok + t;
                let off = (col * h + hd) * d;
                let qn = l2_normed(&q[off..][..d], qs);
                let kn = l2_normed(&k[off..][..d], 1.0);
                let gr = f32s(g, s * g_seq + t * g.nb[2] + hd * g.nb[1], d);
                let decay: Vec<f32> = gr.iter().map(|x| x.exp()).collect();
                let b = *f32s(beta, s * b_seq + t * beta.nb[1] + hd * 4, 1)
                    .first()
                    .unwrap();
                if let Some((sg, sb)) = store {
                    let at = |t: &ffi::ggml_tensor, i: usize| {
                        (t.data as *mut u8).add(s * t.nb[2] + (col - s * n_tok) * t.nb[1] + i * 4)
                            as *mut f32
                    };
                    std::ptr::copy_nonoverlapping(gr.as_ptr(), at(sg, hd * d), d);
                    *at(sb, hd) = b;
                }
                for j in 0..d {
                    let row = &mut state[j * d..][..d];
                    for i in 0..d {
                        row[i] *= decay[i];
                    }
                    let delta = (v[off + j] - dot(row, &kn)) * b;
                    for i in 0..d {
                        row[i] += delta * kn[i];
                    }
                    *out.add(off + j) = dot(row, &qn);
                }
            }
        }
    }
}

unsafe fn cpu_gated_rms(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (o, w) = (all_f32(&*dst.src[0]), all_f32(&*dst.src[1]));
        let p = external_params(dst);
        let eps = f32_param(p[0]);
        let gate = (p[1] != 0).then(|| all_f32(&*dst.src[2]));
        let d = w.len();
        let out = out_f32(dst, o.len());
        for r in part(o.len() / d, ith, nth) {
            let s = rms_scale(&o[r * d..][..d], eps);
            for (i, &wi) in w.iter().enumerate() {
                let j = r * d + i;
                out[j] = o[j] * s * wi * gate.map_or(1.0, |g| sigm(g[j]));
            }
        }
    }
}

unsafe fn cpu_glu(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (g, u) = (all_f32(&*dst.src[0]), all_f32(&*dst.src[1]));
        let lim = f32_param(external_params(dst)[0]);
        let out = out_f32(dst, g.len());
        for i in part(g.len(), ith, nth) {
            out[i] = if lim > 0.0 {
                silu(g[i].min(lim)) * u[i].clamp(-lim, lim)
            } else {
                silu(g[i]) * u[i]
            };
        }
    }
}

unsafe fn cpu_glu_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (gate, up, x) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let lim = f32_param(external_params(dst)[0]);
        let (k, m, n) = (gate.ne[0] as usize, gate.ne[1] as usize, dst.ne[1] as usize);
        // cpu_linear's token rows (h = 1).
        let nbt = if x.ne[1] == 1 && x.ne[2] > 1 {
            x.nb[2]
        } else {
            x.nb[1]
        };
        let out = dst.data as *mut f32;
        for r in part(m, ith, nth) {
            let (wg, wu) = (weight_row(gate, r, 0), weight_row(up, r, 0));
            for t in 0..n {
                let xr = f32s(x, t * nbt, k);
                let (g, u) = (dot(&wg, xr), dot(&wu, xr));
                *out.add(t * m + r) = if lim > 0.0 {
                    silu(g.min(lim)) * u.clamp(-lim, lim)
                } else {
                    silu(g) * u
                };
            }
        }
    }
}

unsafe fn cpu_router_ids(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (l, b) = (all_f32(&*dst.src[0]), all_f32(&*dst.src[1]));
        let (k, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let e = b.len();
        let out = dst.data as *mut i32;
        // `router_ids_resident`: the cached experts' bonus.
        let bonus: Vec<f32> = if dst.src[2].is_null() {
            vec![0.0; e]
        } else {
            let p = external_params(dst);
            let (n_slots, rb) = (p[0], f32_param(p[1]));
            (0..e)
                .map(|i| {
                    if at::<i32>(&*dst.src[2], i) < n_slots {
                        rb
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        for t in part(n, ith, nth) {
            let score: Vec<f32> = (0..e)
                .map(|i| sigm(l[t * e + i]) + b[i] + bonus[i])
                .collect();
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &c| score[c].total_cmp(&score[a]).then(a.cmp(&c)));
            for (j, &i) in order[..k].iter().enumerate() {
                *out.add(t * k + j) = i as i32;
            }
        }
    }
}

unsafe fn cpu_router_w(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (l, ids) = (all_f32(&*dst.src[0]), &*dst.src[1]);
        let p = external_params(dst);
        let (norm, scale) = (p[0] != 0, f32_param(p[1]));
        let (k, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let e = l.len() / n;
        let out = out_f32(dst, k * n);
        for t in part(n, ith, nth) {
            let w: Vec<f32> = (0..k)
                .map(|j| sigm(l[t * e + at::<i32>(ids, t * k + j) as usize]))
                .collect();
            let sum = if norm {
                w.iter().sum::<f32>().max(6.103_515_6e-5)
            } else {
                1.0
            };
            for j in 0..k {
                out[t * k + j] = w[j] / sum * scale;
            }
        }
    }
}

unsafe fn cpu_weighted_sum(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (y, w) = (all_f32(&*dst.src[0]), all_f32(&*dst.src[1]));
        let (e, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let k = w.len() / n;
        let out = out_f32(dst, e * n);
        for t in part(n, ith, nth) {
            for i in 0..e {
                let mut acc = y[(t * k) * e + i] * w[t * k];
                for j in 1..k {
                    acc += y[(t * k + j) * e + i] * w[t * k + j];
                }
                out[t * e + i] = acc;
            }
        }
    }
}

unsafe fn cpu_cache_rows(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (cache, x, cells) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let off = external_params(dst)[0] as usize;
        let d = x.ne[0] as usize;
        for t in part(x.ne[1] as usize, ith, nth) {
            let row = f32s(x, t * x.nb[1], d);
            let cell = at::<i32>(cells, t) as usize;
            let p = (cache.data as *mut u8).add(cell * cache.nb[1]) as *mut u16;
            for (i, v) in row.iter().enumerate() {
                *p.add(off + i) = f32_to_f16(*v);
            }
        }
    }
}

unsafe fn cpu_pool_keys(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (cache, members, rep, ape) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let (d, kpool) = (ape.ne[0] as usize, ape.ne[1] as usize);
        let ape = all_f32(ape);
        let n_new = dst.ne[1] as usize;
        let out = dst.data as *mut f32;
        for p in part(n_new, ith, nth) {
            let rows: Vec<Vec<f32>> = (0..kpool)
                .map(|k| {
                    let cell = at::<i32>(members, p * kpool + k) as usize;
                    f16_row(cache, cell * cache.nb[1], 2 * d)
                })
                .collect();
            let dst_cell = at::<i64>(rep, p) as usize;
            let w = (cache.data as *mut u8).add(dst_cell * cache.nb[1]) as *mut u16;
            for c in 0..d {
                let logit: Vec<f32> = (0..kpool)
                    .map(|k| rows[k][d + c] + ape[c + d * k])
                    .collect();
                let max = logit.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = logit.iter().map(|l| (l - max).exp()).collect();
                let sum: f64 = ex.iter().map(|&v| v as f64).sum();
                let inv = (1.0 / sum) as f32;
                let mut acc = 0f32;
                for k in 0..kpool {
                    acc += ex[k] * inv * rows[k][c];
                }
                *out.add(p * d + c) = acc;
                *w.add(2 * d + c) = f32_to_f16(acc);
            }
        }
    }
}

unsafe fn cpu_mla_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (q, cache, table, sel) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let p = external_params(dst);
        let (n_tok, scale, block) = (p[0] as usize, f32_param(p[1]), p[2] as usize);
        let (d, h, n) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let width = table.ne[0] as usize;
        let cap = sel.ne[0] as usize;
        let out = dst.data as *mut f32;
        for idx in part(n * h, ith, nth) {
            let (t, hd) = (idx / h, idx % h);
            let s = t / n_tok;
            let qr = f32s(q, t * q.nb[2] + hd * q.nb[1], d);
            let count = at::<i32>(sel, t * cap) as usize;
            let mut keys = Vec::with_capacity(count);
            let mut score = Vec::with_capacity(count);
            for j in 0..count {
                let pos = at::<i32>(sel, t * cap + 1 + j) as usize;
                let blk = at::<i32>(table, s * width + pos / block) as usize;
                let row = f16_row(cache, (blk * block + pos % block) * cache.nb[1], d);
                score.push(dot(qr, &row) * scale);
                keys.push(row);
            }
            let max = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = score.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = ex.iter().sum();
            let o = std::slice::from_raw_parts_mut(out.add((t * h + hd) * d), d);
            o.fill(0.0);
            for (row, e) in keys.iter().zip(&ex) {
                let w = e / sum;
                for i in 0..d {
                    o[i] += row[i] * w;
                }
            }
        }
    }
}

unsafe fn cpu_join_cols(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    if ith != 0 {
        return;
    }
    let _ = nth;
    unsafe {
        let d = dst.ne[0] as usize;
        let mut col = 0;
        let out = dst.data as *mut f32;
        for i in 0..ffi::GGML_MAX_SRC as usize {
            let s = dst.src[i];
            if s.is_null() {
                break;
            }
            let s = &*s;
            let cols = (s.ne[1] * s.ne[2] * s.ne[3]) as usize;
            for c in 0..cols {
                let r = f32s(s, c * s.nb[1], d);
                std::ptr::copy_nonoverlapping(r.as_ptr(), out.add((col + c) * d), d);
            }
            col += cols;
        }
    }
}

unsafe fn cpu_argmax(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = all_f32(&*dst.src[0]);
        let n = dst.ne[0] as usize;
        let v = x.len() / n;
        let out = dst.data as *mut i32;
        for t in part(n, ith, nth) {
            let r = &x[t * v..][..v];
            let mut best = 0;
            for i in 1..v {
                if r[i] > r[best] {
                    best = i;
                }
            }
            // Over picked rows (`argmax_wide_ids`): the row's token id.
            *out.add(t) = if dst.src[1].is_null() {
                best as i32
            } else {
                *((*dst.src[1]).data as *const i32).add(best)
            };
        }
    }
}

unsafe fn cpu_pmax(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = all_f32(&*dst.src[0]);
        let n = dst.ne[0] as usize;
        let v = x.len() / n;
        let out = dst.data as *mut f32;
        for t in part(n, ith, nth) {
            let r = &x[t * v..][..v];
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f64 = r.iter().map(|&y| ((y - m) as f64).exp()).sum();
            *out.add(t) = (1.0 / sum) as f32;
        }
    }
}

unsafe fn cpu_nextn_in(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let src = |i: usize| all_f32(&*dst.src[i]);
        let (e, h, we, wh) = (src(0), src(1), src(2), src(3));
        let eps = f32_param(external_params(dst)[0]);
        let d = we.len();
        let n = dst.ne[1] as usize;
        let out = dst.data as *mut f32;
        for t in part(n, ith, nth) {
            for (half, (x, w)) in [(e, we), (h, wh)].into_iter().enumerate() {
                let r = &x[t * d..][..d];
                let s = rms_scale(r, eps);
                for i in 0..d {
                    *out.add(t * 2 * d + half * d + i) = r[i] * s * w[i];
                }
            }
        }
    }
}

unsafe fn cpu_get_cols(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, rows) = (&*dst.src[0], &*dst.src[1]);
        let (d, m) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let out = dst.data as *mut f32;
        for j in part(m, ith, nth) {
            let r = at::<i32>(rows, j) as usize;
            let src = f32s(x, r * x.nb[1], d);
            std::ptr::copy_nonoverlapping(src.as_ptr(), out.add(j * d), d);
        }
    }
}

unsafe fn cpu_add(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (a, b) = (all_f32(&*dst.src[0]), all_f32(&*dst.src[1]));
        let out = dst.data as *mut f32;
        for i in part(a.len(), ith, nth) {
            *out.add(i) = a[i] + b[i];
        }
    }
}

#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(test)]
mod tests;
