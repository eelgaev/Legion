//! qwen4exp's hyper-connection mix in prompt steps on a V100: ggml's gated `hc_pre`
//! (`mixed = scale · Σ_c xn[c] · sigmoid(gate[c])`) as an external op ([`crate::ggml::ExternalOp`],
//! CPU reference plus the kernel in `src/cuda/hc_mix16.cu`) that also writes `mixed` rounded to
//! F16 (#774). ggml-cuda runs every quantized projection of a prompt step on a V100 as cuBLAS
//! F16 GEMMs and converts its F32 input to F16 for each one of them: the block input feeds the
//! Q8_0 projections of a DeltaNet layer (qkv, gate) or an attention layer (q, k, v and the
//! indexer's two), the FFN input the shared expert's two. With the F16 copy registered as the
//! input's shared cast ([`Context::set_shared_cast`]), those matmuls take it with an F16 copy
//! of their weight from ggml's own converter ([`crate::model::core::f16_joined`]): the same
//! GEMMs on the same F16 values, so the same bits, without the input's conversions.
//!
//! After [`super::hc_norm::hc_post_norm_rebuilt16`] the op also rebuilds `xn` itself
//! ([`mix_rebuilt`]): the norm then writes each stream's scale instead of `xn` (4 floats a token,
//! not `hc · n_embd`), and `xn = (res · scale) · gamma` here is the norm's own two rounded
//! multiplies, so the same bits; `legion serve --hc-xn-written` keeps `xn` written.
//!
//! Opt-in (`legion serve --hc-mix-f16`): bit-exact op for op, but the smaller graph moves ggml's
//! allocation, which can flip ggml-cuda's layout-gated MoE fusions and so greedy text (#472).
//!
//! On sm_80+ with BF16 mixer weights (the NVFP4 checkpoints) [`mix_rebuilt_f32`] is the same op
//! without the F16 copy, after [`super::hc_norm::hc_post_norm_rebuilt_xnb`]: the norm writes the
//! streams' scales and the BF16 `xn` the down matmul reads, not the F32 `xn` (40 KB a token
//! written and read back), and the op rebuilds `xn` (#836). Opt-in (`legion serve
//! --hc-mix-rebuild-bf16`): bit-exact op for op, but greedy output moved in serving tests (the
//! smaller graph moves ggml's allocation, as above), and still moved with the norm's node
//! kept in `hc_post_norm`'s size and layout, the BF16 `xn` at its usual offset. Under
//! `--deterministic --stable-fusions` the greedy output equals the default's (GB10, C=1, C=4
//! and depth 16K), so the move is the fusions', not the op's.
//!
//! [`hc_pre`] is ggml's `hc_pre_gated` itself on a GPU (no F16 copy, `xn` read): one node of
//! ggml's shape with ggml's sources in ggml's order, so the graph and its allocation stay
//! ggml's, and the kernel's expressions are ggml's, so the same bits (#836: no ggml-cuda
//! compute). On by default; `legion serve --hc-pre-ggml` keeps ggml's op.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static HC_MIX16: ExternalOp = ExternalOp {
    name: "hc_mix16",
    cpu: cpu_hc_mix16,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_mix16),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Whether graphs built from now on may use [`mix`] / [`mix_rebuilt`] (off by default).
static ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on run the mixer's gated mix as this op where it fits
/// (`legion serve --hc-mix-f16`; off by default: ggml's `hc_pre` and each matmul's own F16
/// conversion, the same bits op for op, but a different graph layout).
pub fn set_mix16(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_mix16`].
pub(super) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Whether graphs built from now on may use [`mix_rebuilt_f32`] (off by default).
static REBUILD_BF16: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on have the norm of an sm_80+ GPU's BF16 mixers write the
/// streams' scales instead of the F32 `xn` and the gated mix rebuild it (`legion serve
/// --hc-mix-rebuild-bf16`, #836; off by default: the same bits op for op, but a different graph
/// layout, and greedy output moved in serving tests).
pub fn set_rebuild_bf16(on: bool) {
    REBUILD_BF16.store(on, Ordering::Relaxed);
}

/// See [`set_rebuild_bf16`].
pub(super) fn rebuild_bf16() -> bool {
    REBUILD_BF16.load(Ordering::Relaxed)
}

/// Whether graphs built from now on run ggml's gated `hc_pre` as this op ([`hc_pre`]; on by
/// default).
static PRE: AtomicBool = AtomicBool::new(true);

/// Sets whether graphs built from now on run the mixer's gated mix on a GPU as [`hc_pre`] where
/// ggml's `hc_pre_gated` would run (on by default, the same bits; `legion serve --hc-pre-ggml`
/// turns it off, for comparisons).
pub fn set_hc_pre(on: bool) {
    PRE.store(on, Ordering::Relaxed);
}

/// See [`set_hc_pre`].
pub(super) fn pre_on() -> bool {
    PRE.load(Ordering::Relaxed)
}

/// Whether [`hc_pre`] takes these operands of ggml's `hc_pre_gated(x, gate, _)`: on a GPU
/// (`on_gpu`; the CPU graph stays ggml's, the reference), F32 `[e, hc, n]` both, contiguous.
pub(super) fn pre_fits(x: Tensor<'_>, gate: Tensor<'_>, on_gpu: bool) -> bool {
    pre_on()
        && cfg!(feature = "cuda")
        && on_gpu
        && x.ne() == gate.ne()
        && x.ne()[3] == 1
        && (1..=64).contains(&x.ne()[1])
        && [x, gate]
            .iter()
            .all(|t| t.ty() == types::F32 && t.is_contiguous())
}

/// ggml's `hc_pre_gated(x, gate, scale)` as this op (#836: no ggml-cuda compute): `x` the normed
/// streams and `gate` the up matmul's output, F32 `[e, hc, n]` contiguous, as ggml takes them;
/// `mixed = scale · Σ_c x[c] · sigmoid(gate[c])`, F32 `[e, n]`, ggml's expressions in ggml's
/// order (the same bits), one node as ggml's, of its shape.
pub(super) fn hc_pre<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    gate: Tensor<'a>,
    scale: f32,
) -> Tensor<'a> {
    let [e, hc, n, _] = x.ne();
    assert_eq!(gate.ne(), x.ne(), "hc_pre: gate");
    for t in [gate, x] {
        assert_eq!(t.ty(), types::F32, "hc_pre: F32 operands");
        assert!(t.is_contiguous(), "hc_pre: contiguous operands");
    }
    ctx.external_with(
        &HC_MIX16,
        types::F32,
        [e, n, 1, 1],
        &[x, gate],
        &[scale.to_bits() as i32, hc as i32, 0, 1],
    )
}

/// A node's gate, its `xn` (or streams) and `(e, n)`: [`hc_pre`]'s (param 3 set: `[x, gate]`,
/// ggml's order, `[e, hc, n]` both) or the others' (`[gate, xn or res, ..]`, `[hc · e, n]`).
///
/// # Safety
/// `dst` is a node made by this module.
unsafe fn operands<'t>(
    dst: &'t ffi::ggml_tensor,
    params: &[i32],
) -> (&'t ffi::ggml_tensor, &'t ffi::ggml_tensor, (usize, usize)) {
    // SAFETY: the caller's: two or more live sources.
    let (a, b) = unsafe { (&*dst.src[0], &*dst.src[1]) };
    if params[3] != 0 {
        (b, a, (b.ne[0] as usize, b.ne[2] as usize))
    } else {
        let hc = params[1] as usize;
        (a, b, (a.ne[0] as usize / hc, a.ne[1] as usize))
    }
}

/// Where the op's parts sit in its F32 node, in bytes: `[mixed | mixed16]`, the F16 part 256
/// bytes aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Layout {
    /// `mixed` as F16, `[n][e]`.
    pub mixed16: usize,
    /// The node's size in floats.
    pub floats: usize,
}

impl Layout {
    pub(super) fn new(e: usize, n: usize) -> Self {
        let mixed16 = (e * n * 4).next_multiple_of(256);
        Layout {
            mixed16,
            floats: (mixed16 + e * n * 2).div_ceil(4),
        }
    }
}

/// The node's size in floats: [`Layout`]'s with the F16 part, `mixed` alone without.
fn node_floats(e: usize, n: usize, f16: bool) -> usize {
    if f16 { Layout::new(e, n).floats } else { e * n }
}

/// `mixed = scale · Σ_c xn[c] · sigmoid(gate[c])` as ggml's `hc_pre_gated(xn, gate, scale)`
/// computes it, `gate` and `xn` F32 `[hc · e, n]`, contiguous: `(mixed, mixed16)`, `mixed` F32
/// `[e, n]` and the same values rounded to F16 as ggml's F32 to F16 conversion rounds them.
pub(super) fn mix<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    xn: Tensor<'a>,
    hc: i64,
    scale: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    let [rows, n, _, _] = gate.ne();
    assert_eq!(xn.ne(), [rows, n, 1, 1], "hc_mix16: xn");
    for t in [gate, xn] {
        assert_eq!(t.ty(), types::F32, "hc_mix16: F32 operands");
        assert!(t.is_contiguous(), "hc_mix16: contiguous operands");
    }
    let node = node(ctx, &[gate, xn], rows / hc, n, hc, scale, true);
    (node.0, node.1.expect("the F16 part"))
}

/// [`mix`] with `xn` rebuilt from what [`super::hc_norm::hc_post_norm_rebuilt16`] writes in its
/// place, `xn = (res · scales) · gamma` per value as the norm computes it (the same bits): `res`
/// the new streams (F32 `[e, hc, n]`), `scales` each stream's norm scale (F32 `[hc, n]`),
/// `gamma` the norm's weight (F32 `[e · hc]`), all contiguous.
pub(super) fn mix_rebuilt<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    res: Tensor<'a>,
    scales: Tensor<'a>,
    gamma: Tensor<'a>,
    hc: i64,
    scale: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    let (e, n) = rebuilt_shape(gate, res, scales, gamma, hc);
    let node = node(ctx, &[gate, res, scales, gamma], e, n, hc, scale, true);
    (node.0, node.1.expect("the F16 part"))
}

/// [`mix_rebuilt`] without the F16 copy: `mixed` alone, F32 `[e, n]`, the bits of ggml's
/// `hc_pre_gated` on the `xn` the norm would have written (#836).
pub(super) fn mix_rebuilt_f32<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    res: Tensor<'a>,
    scales: Tensor<'a>,
    gamma: Tensor<'a>,
    hc: i64,
    scale: f32,
) -> Tensor<'a> {
    let (e, n) = rebuilt_shape(gate, res, scales, gamma, hc);
    node(ctx, &[gate, res, scales, gamma], e, n, hc, scale, false).0
}

/// [`mix_rebuilt`]'s operand checks: `(e, n)`.
fn rebuilt_shape(
    gate: Tensor<'_>,
    res: Tensor<'_>,
    scales: Tensor<'_>,
    gamma: Tensor<'_>,
    hc: i64,
) -> (i64, i64) {
    let [rows, n, _, _] = gate.ne();
    let e = rows / hc;
    assert_eq!(rows, e * hc, "hc_mix16: gate rows");
    assert_eq!(res.ne(), [e, hc, n, 1], "hc_mix16: streams");
    assert_eq!(scales.ne(), [hc, n, 1, 1], "hc_mix16: scales");
    assert_eq!(gamma.elements(), rows as usize, "hc_mix16: gamma");
    for t in [gate, res, scales, gamma] {
        assert_eq!(t.ty(), types::F32, "hc_mix16: F32 operands");
        assert!(t.is_contiguous(), "hc_mix16: contiguous operands");
    }
    (e, n)
}

/// The op's node for `src` (`[gate, xn]` or `[gate, res, scales, gamma]`) and its views:
/// `mixed`, and with `f16` `mixed16`.
fn node<'a>(
    ctx: &'a Context,
    src: &[Tensor<'a>],
    e: i64,
    n: i64,
    hc: i64,
    scale: f32,
    f16: bool,
) -> (Tensor<'a>, Option<Tensor<'a>>) {
    assert!(!f16 || e % 2 == 0, "hc_mix16: F16 rows of an odd width");
    let at = Layout::new(e as usize, n as usize);
    let node = ctx.external_with(
        &HC_MIX16,
        types::F32,
        [node_floats(e as usize, n as usize, f16) as i64, 1, 1, 1],
        src,
        &[scale.to_bits() as i32, hc as i32, i32::from(f16), 0],
    );
    let mixed = ctx.view_2d(node, [e, n], e as usize * 4, 0);
    let mixed16 = f16.then(|| {
        ctx.reinterpret(
            ctx.view_1d(node, e * n / 2, at.mixed16),
            types::F16,
            [e, n, 1, 1],
        )
    });
    (mixed, mixed16)
}

/// For a node made by [`mix_rebuilt`], its scales and gamma (`src[2]`, `src[3]`; `src[1]` is
/// then the streams); `None` for [`mix`]'s (`src[1]` is `xn`).
fn rebuilt(dst: &ffi::ggml_tensor) -> Option<(*const ffi::ggml_tensor, *const ffi::ggml_tensor)> {
    (!dst.src[2].is_null()).then(|| (dst.src[2] as *const _, dst.src[3] as *const _))
}

unsafe fn cpu_hc_mix16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mix`, `mix_rebuilt` or `hc_pre`, host tensors. ggml's gated mix, then the
    // F16 rounding.
    unsafe {
        let params = crate::ggml::external_params(dst);
        let (gate, xn, (e, n)) = operands(dst, &params);
        let scale = f32::from_bits(params[0] as u32);
        let hc = params[1] as usize;
        let rows = e * hc;
        let at = Layout::new(e, n);
        let gp = gate.data as *const f32;
        // `xn`, or (`mix_rebuilt`) the streams with the scales and gamma to rebuild it from.
        let xp = xn.data as *const f32;
        let rebuilt = rebuilt(dst).map(|(s, g)| ((*s).data as *const f32, (*g).data as *const f32));
        let out = dst.data as *mut f32;
        // Past the node without the F16 part.
        let out16 = (params[2] != 0).then(|| (dst.data as *mut u8).add(at.mixed16) as *mut u16);
        for t in (ith..n).step_by(nth) {
            for i in 0..e {
                let mut sum = 0f32;
                for c in 0..hc {
                    let at = t * rows + c * e + i;
                    let mut xv = *xp.add(at);
                    if let Some((sc, g)) = rebuilt {
                        xv = (xv * *sc.add(t * hc + c)) * *g.add(c * e + i);
                    }
                    sum += xv * (1.0 / (1.0 + (-*gp.add(at)).exp()));
                }
                let m = scale * sum;
                *out.add(t * e + i) = m;
                if let Some(o16) = out16 {
                    *o16.add(t * e + i) = ffi::ggml_fp32_to_fp16(m);
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct HcMix16 {
    gate: *const f32,
    xn: *const f32,
    res: *const f32,
    sc: *const f32,
    g: *const f32,
    mixed: *mut f32,
    mixed16: *mut u16,
    e: i64,
    hc: i64,
    n: i64,
    scale: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_mix16_launch(p: *const HcMix16, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_mix16(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mix`, `mix_rebuilt` or `hc_pre`, device tensors of the stream's device.
    unsafe {
        let params = crate::ggml::external_params(dst);
        let (gate, xn, (e, n)) = operands(dst, &params);
        let hc = params[1] as i64;
        let at = Layout::new(e, n);
        let (e, n) = (e as i64, n as i64);
        let null = std::ptr::null::<f32>();
        let (x, res, sc, g) = match rebuilt(dst) {
            Some((s, g)) => (
                null,
                xn.data as *const f32,
                (*s).data as *const f32,
                (*g).data as *const f32,
            ),
            None => (xn.data as *const f32, null, null, null),
        };
        let p = HcMix16 {
            gate: gate.data as *const f32,
            xn: x,
            res,
            sc,
            g,
            mixed: dst.data as *mut f32,
            mixed16: if params[2] != 0 {
                (dst.data as *mut u8).add(at.mixed16) as *mut u16
            } else {
                std::ptr::null_mut()
            },
            e,
            hc,
            n,
            scale: f32::from_bits(params[0] as u32),
        };
        match legion_hc_mix16_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("hc_mix16: error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
