//! qwen4exp's hyper-connection mixer front as one external op ([`crate::ggml::ExternalOp`],
//! CPU reference plus a CUDA kernel in `src/cuda/fused.cu`): every residual stream RMS-normed
//! and times its gamma, and the per-stream inject weights from the normed streams (an F32 matmul
//! with 4 outputs that only reads them), in one pass per token instead of ggml's `rms_norm`,
//! `mul` and an F32 `mul_mat`.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};
use crate::model::core::mmq_preq;
use crate::model::moe_ffn;

static HC_POST_NORM: ExternalOp = ExternalOp {
    name: "hc_post_norm",
    cpu: cpu_hc_post_norm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_post_norm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static HC_NORM: ExternalOp = ExternalOp {
    name: "hc_norm",
    cpu: cpu_hc_norm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_hc_norm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The [`hc_post_norm_rebuilt_xnb`] node's parameter set when its `out` is a deferred
/// reduce-scatter ([`crate::tp_net::seq_par::defer_sum`], #1176); 7 and 8 are a folded MoE
/// tail's.
const P_RS: usize = 9;

/// Most streams and inject outputs [`hc_norm`]'s CUDA kernel takes.
pub(super) const HC_NORM_MAX: i64 = 8;

/// The hyper-connection mixer's front for `x` (F32 `[e, hc, n]`): `xn = rms_norm(x) · gamma`
/// with each stream (`e` values) normed on its own and `gamma` F32 `[e, hc]`, and
/// `inject = w^T xn` with `w` F32 `[e · hc, n_inj]` (as `mul_mat(w, xn)`). Returns `xn` (F32
/// `[e · hc, n]`) and `inject` (F32 `[n_inj, n]`).
pub(super) fn hc_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    let [e, hc, n, _] = x.ne();
    let [k, n_inj, _, _] = w.ne();
    assert_eq!(k, e * hc, "hc_norm: inject rows");
    assert!(
        hc <= HC_NORM_MAX && n_inj <= HC_NORM_MAX,
        "hc_norm: {hc} streams, {n_inj} outputs"
    );
    assert_eq!(gamma.elements(), (e * hc) as usize, "hc_norm: gamma");
    for t in [x, gamma, w] {
        assert_eq!(t.ty(), types::F32, "hc_norm: F32 tensors");
    }
    assert!(
        gamma.is_contiguous() && w.is_contiguous(),
        "hc_norm: weights contiguous"
    );
    assert_eq!(x.nb()[1], (e * 4) as usize, "hc_norm: streams contiguous");
    let node = ctx.external_with(
        &HC_NORM,
        types::F32,
        [e * hc * n + n_inj * n, 1, 1, 1],
        &[x, gamma, w],
        &[eps.to_bits() as i32],
    );
    let xn = ctx.view_2d(node, [e * hc, n], (e * hc) as usize * 4, 0);
    let inject = ctx.view_2d(
        node,
        [n_inj, n],
        n_inj as usize * 4,
        (e * hc * n) as usize * 4,
    );
    (xn, inject)
}

/// [`hc_norm`] of the streams after `hc_combine`, in one pass per token (prompt steps: the
/// streams are read once, not written by one op and read back by the next): `res` (F32
/// `[e, hc, n]`, streams contiguous) plus the block output `out` (F32 `[e, n]`, rows contiguous)
/// times `2 sigmoid(inj / hc)` per stream (`inj` F32 `[hc, n]`), as ggml's `scale`, `sigmoid`,
/// `scale` and `hc_post` compute it, then normed and injected as [`hc_norm`]. Returns the new
/// streams (`[e, hc, n]`), `xn` (`[e · hc, n]`) and `inject` (`[n_inj, n]`). With a Q8_0
/// `down` (contiguous `[e · hc, m]`), also `xn` rounded to F16 (`[e · hc, n]`, each value as
/// ggml's F32 to F16 conversion rounds it) and `down` as F16 (ggml-cuda's own Q8_0 to F16
/// conversion on CUDA), the two inputs ggml-cuda's cuBLAS path makes for `mul_mat(down, xn)`.
/// With a BF16 `down`, `xn` rounded to BF16 instead (as ggml-cuda rounds it for its cuBLAS BF16
/// GEMM, which then reads it as it is, #469; [`xnb_fits`]). With `quant`, also `xn` quantized
/// for ggml-cuda's MMQ of a Q8_0 weight ([`mmq_preq`]'s layout, [`mmq_preq::q8_1_bytes`];
/// CUDA on sm_80+, `e · hc` a multiple of 512), written by
/// the same kernel (#363). With `legion serve --hc-fuse`, where `out` is the fused MoE op's
/// whole sum ([`moe_sum_fits`]), the op makes `out` from the MoE op's per-pair rows itself and
/// the MoE op skips its sum kernel (#774).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_norm<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    down: Option<Tensor<'a>>,
    quant: bool,
) -> PostNorm<'a> {
    // (The kernels that make the sum write no BF16 xn.)
    let moe = !down.is_some_and(|d| d.ty() == types::BF16) && moe_sum_fits(res, out, w);
    let kind = PostNormKind {
        down,
        quant,
        rebuild: false,
        moe,
        streams16: false,
        padded: false,
    };
    let (node, at) = post_norm_node(ctx, res, out, inj, gamma, w, eps, kind);
    let [e, hc, n, _] = res.ne();
    let [k, n_inj, _, _] = w.ne();
    let m = down.map_or(0, |d| d.ne()[1]);
    let res_new = ctx.view_3d(node, [e, hc, n], [e as usize * 4, k as usize * 4], 0);
    let xn = ctx.view_2d(node, [k, n], k as usize * 4, at.xn * 4);
    let f16 = |floats: usize, from: usize, ne: [i64; 4]| {
        ctx.reinterpret(ctx.view_1d(node, floats as i64, from * 4), types::F16, ne)
    };
    let bf16 = down.filter(|d| d.ty() == types::BF16);
    let xnb = bf16.map(|_| {
        let v = ctx.view_1d(node, (k * n) / 2, at.xn16 * 4);
        ctx.reinterpret(v, types::BF16, [k, n, 1, 1])
    });
    let down = down.filter(|d| d.ty() == types::Q8_0);
    let xn16 = down.map(|_| f16((k * n) as usize / 2, at.xn16, [k, n, 1, 1]));
    let down16 = down.map(|_| f16((k * m) as usize / 2, at.down16, [k, m, 1, 1]));
    let inject = ctx.view_2d(node, [n_inj, n], n_inj as usize * 4, at.inject * 4);
    let xq = quant.then(|| ctx.view_1d(node, (at.floats - at.xq / 4) as i64, at.xq));
    PostNorm {
        res: res_new,
        xn,
        xn16,
        down16,
        xnb,
        inject,
        xq,
    }
}

/// [`hc_post_norm`] with `quant` for a reader that rebuilds `xn` itself
/// ([`super::hc_up_mix::up_mix_rebuilt`]): instead of `xn` (`e · hc` floats a token) the op
/// writes each stream's norm scale (`hc` floats), and `xn = (res · scale) · gamma` per value is
/// the op's own arithmetic, so the reader gets the same bits (#526). `xq` is written as by
/// [`hc_post_norm`]. Needs [`rebuild_fits`].
///
/// `bf16` ([`set_streams_bf16`], opt-in, #826): the new streams are rounded to BF16 and
/// written as BF16 (half the bytes), and the rounded values are the ones normed, quantized and
/// injected; `res` may then be BF16 too (the previous op's streams). Moves output.
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_norm_rebuilt<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    bf16: bool,
) -> PostNormRebuilt<'a> {
    let [e, hc, n, _] = res.ne();
    assert!(rebuild_fits(e * hc, hc, n), "hc_post_norm: {n} tokens");
    let (node, at) = post_norm_node(
        ctx,
        res,
        out,
        inj,
        gamma,
        w,
        eps,
        PostNormKind {
            down: None,
            quant: true,
            rebuild: true,
            // `--hc-fuse` on sm_80+ (#826): F32 streams only.
            moe: !bf16 && moe_sum_fits(res, out, w),
            streams16: bf16,
            padded: false,
        },
    );
    let [k, n_inj, _, _] = w.ne();
    let res_new = if bf16 {
        let v = ctx.view_1d(node, (k * n) / 2, 0);
        ctx.reinterpret(v, types::BF16, [e, hc, n, 1])
    } else {
        ctx.view_3d(node, [e, hc, n], [e as usize * 4, k as usize * 4], 0)
    };
    PostNormRebuilt {
        res: res_new,
        scales: ctx.view_2d(node, [hc, n], hc as usize * 4, at.xn * 4),
        inject: ctx.view_2d(node, [n_inj, n], n_inj as usize * 4, at.inject * 4),
        xq: ctx.view_1d(node, (at.floats - at.xq / 4) as i64, at.xq),
    }
}

/// [`hc_post_norm`] with the F16 copies for a V100's down matmul ([`xn16_fits`]) and for a
/// reader that rebuilds `xn` itself ([`super::hc_mix16::mix_rebuilt`]): each stream's norm scale
/// instead of `xn`, as [`hc_post_norm_rebuilt`] (#774). On CUDA, needs [`rebuild16_fits`].
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_norm_rebuilt16<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    down: Tensor<'a>,
) -> PostNormRebuilt16<'a> {
    let [e, hc, n, _] = res.ne();
    let [k, n_inj, _, _] = w.ne();
    // The CUDA kernel takes rebuild16_fits' shape only; the CPU reference any.
    assert!(rebuild_fits(k, hc, n), "hc_post_norm: {n} tokens");
    let (node, at) = post_norm_node(
        ctx,
        res,
        out,
        inj,
        gamma,
        w,
        eps,
        PostNormKind {
            down: Some(down),
            quant: false,
            rebuild: true,
            moe: false,
            streams16: false,
            padded: false,
        },
    );
    let m = down.ne()[1];
    let f16 = |floats: usize, from: usize, ne: [i64; 4]| {
        ctx.reinterpret(ctx.view_1d(node, floats as i64, from * 4), types::F16, ne)
    };
    PostNormRebuilt16 {
        res: ctx.view_3d(node, [e, hc, n], [e as usize * 4, k as usize * 4], 0),
        scales: ctx.view_2d(node, [hc, n], hc as usize * 4, at.xn * 4),
        xn16: f16((k * n) as usize / 2, at.xn16, [k, n, 1, 1]),
        down16: f16((k * m) as usize / 2, at.down16, [k, m, 1, 1]),
        inject: ctx.view_2d(node, [n_inj, n], n_inj as usize * 4, at.inject * 4),
    }
}

/// [`hc_post_norm`] with the BF16 `xn` for an sm_80+ GPU's BF16 down matmul ([`xnb_fits`]) and
/// for a reader that rebuilds `xn` itself ([`super::hc_mix16::mix_rebuilt_f32`]): each stream's
/// norm scale instead of the F32 `xn`, as [`hc_post_norm_rebuilt16`] (#836). On CUDA, needs
/// [`rebuild16_fits`]'s shape (the register kernel's).
#[allow(clippy::too_many_arguments)]
pub(super) fn hc_post_norm_rebuilt_xnb<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    down: Tensor<'a>,
) -> PostNormRebuiltXnb<'a> {
    let [e, hc, n, _] = res.ne();
    let [k, n_inj, _, _] = w.ne();
    assert_eq!(down.ty(), types::BF16, "hc_post_norm: BF16 down");
    assert!(rebuild_fits(k, hc, n), "hc_post_norm: {n} tokens");
    let (node, at) = post_norm_node(
        ctx,
        res,
        out,
        inj,
        gamma,
        w,
        eps,
        PostNormKind {
            down: Some(down),
            quant: false,
            rebuild: true,
            moe: false,
            streams16: false,
            // xn's whole place kept: every part of the node where hc_post_norm has it (#836).
            padded: true,
        },
    );
    // `out` the reduce-scatter of a `--tp-seq-par` step with BF16 partials (#1176): the norm
    // adds the two ranks' partials itself, as the exchange's sum kernel adds them, so the sum
    // is not written and read back.
    if crate::tp_net::seq_par::deferrable(out) {
        crate::tp_net::seq_par::defer_sum(out);
        // SAFETY: the node just made, an external op node.
        unsafe { crate::ggml::set_external_param(node.raw(), P_RS, 1) };
    }
    let xnb = ctx.view_1d(node, (k * n) / 2, at.xn16 * 4);
    PostNormRebuiltXnb {
        res: ctx.view_3d(node, [e, hc, n], [e as usize * 4, k as usize * 4], 0),
        scales: ctx.view_2d(node, [hc, n], hc as usize * 4, at.xn * 4),
        xnb: ctx.reinterpret(xnb, types::BF16, [k, n, 1, 1]),
        inject: ctx.view_2d(node, [n_inj, n], n_inj as usize * 4, at.inject * 4),
    }
}

/// [`hc_post_norm_rebuilt_xnb`]'s outputs.
pub(super) struct PostNormRebuiltXnb<'a> {
    /// The new streams, F32 `[e, hc, n]`.
    pub res: Tensor<'a>,
    /// Each stream's norm scale, F32 `[hc, n]`.
    pub scales: Tensor<'a>,
    /// `xn` rounded to BF16 (`[e · hc, n]`).
    pub xnb: Tensor<'a>,
    /// The per-stream weights, F32 `[n_inj, n]`.
    pub inject: Tensor<'a>,
}

/// [`hc_post_norm_rebuilt16`]'s outputs.
pub(super) struct PostNormRebuilt16<'a> {
    /// The new streams, F32 `[e, hc, n]`.
    pub res: Tensor<'a>,
    /// Each stream's norm scale, F32 `[hc, n]`.
    pub scales: Tensor<'a>,
    /// `xn` rounded to F16 (`[e · hc, n]`).
    pub xn16: Tensor<'a>,
    /// The down weight as F16 (`[e · hc, m]`).
    pub down16: Tensor<'a>,
    /// The per-stream weights, F32 `[n_inj, n]`.
    pub inject: Tensor<'a>,
}

/// Whether [`hc_post_norm_rebuilt16`] (and [`hc_post_norm_rebuilt_xnb`]) takes `n` tokens of `e` values in `hc` streams with
/// `n_inj` inject outputs: the CUDA kernel's (`hc_post_norm_v70_kernel`) qwen4exp shape, 4 x
/// 2560 values and 4 outputs, and the scales' place in the node within its 32-bit offset.
pub(super) fn rebuild16_fits(e: i64, hc: i64, n_inj: i64, n: i64) -> bool {
    e == 2560 && hc == 4 && n_inj == 4 && rebuild_fits(e * hc, hc, n)
}

/// Whether [`hc_post_norm_rebuilt_xnb`] takes this step for a reader that rebuilds `xn` with
/// the register kernel's arithmetic: [`rebuild16_fits`], and not [`set_generic_post_norm`]
/// (the generic kernel writes `xn`; its comparisons keep it written and read).
pub(super) fn rebuild_xnb_fits(e: i64, hc: i64, n_inj: i64, n: i64) -> bool {
    !GENERIC.load(std::sync::atomic::Ordering::Relaxed) && rebuild16_fits(e, hc, n_inj, n)
}

/// Whether [`hc_post_norm_rebuilt`] takes `n` tokens of `k` values in `hc` streams: the
/// scales' place in the node, in floats, fits the kernel's 32-bit offset.
pub(super) fn rebuild_fits(k: i64, hc: i64, n: i64) -> bool {
    (k + hc) * n <= i64::from(i32::MAX)
}

/// [`hc_post_norm_rebuilt`]'s outputs.
pub(super) struct PostNormRebuilt<'a> {
    /// The new streams, F32 `[e, hc, n]` (BF16 with `bf16`).
    pub res: Tensor<'a>,
    /// Each stream's norm scale, F32 `[hc, n]`.
    pub scales: Tensor<'a>,
    /// The per-stream weights, F32 `[n_inj, n]`.
    pub inject: Tensor<'a>,
    /// `xn` quantized for ggml-cuda's MMQ ([`mmq_preq::mul_mat_cols`]'s `xq`).
    pub xq: Tensor<'a>,
}

/// What [`post_norm_node`] writes besides the streams and the inject weights.
struct PostNormKind<'a> {
    /// The down weight to write F16 copies (or a BF16 `xn`) for.
    down: Option<Tensor<'a>>,
    /// `xn` quantized for MMQ.
    quant: bool,
    /// The streams' scales instead of `xn` ([`hc_post_norm_rebuilt`]).
    rebuild: bool,
    /// `out` is a MoE op's deferred sum.
    moe: bool,
    /// BF16 streams (with `rebuild`, #826).
    streams16: bool,
    /// With `rebuild`: the scales keep all of `xn`'s place (`e · hc` floats a token, the rest
    /// unwritten), so every other part sits where [`hc_post_norm`] puts it and the node has its
    /// size (#836). The BF16 `xn` then starts where [`hc_post_norm`]'s does, as aligned (a
    /// cuBLAS GEMM reads it), and ggml's allocator plans the graph as it plans main's.
    padded: bool,
}

/// The node of [`hc_post_norm`] (`rebuild` false) or [`hc_post_norm_rebuilt`] (true; `quant`,
/// no `down`) and where its parts sit. `moe`: `out` is a [`moe_ffn::deferrable`] sum the op
/// takes over ([`moe_ffn::defer_sum`]; on CUDA the op then reads the MoE op's pairs, see
/// [`moe_sum_fits`]).
#[allow(clippy::too_many_arguments)]
fn post_norm_node<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    gamma: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    kind: PostNormKind<'a>,
) -> (Tensor<'a>, PostNormLayout) {
    let PostNormKind {
        down,
        quant,
        rebuild,
        moe,
        streams16,
        padded,
    } = kind;
    assert!(
        !padded || rebuild,
        "hc_post_norm: padded scales in the rebuilt form only"
    );
    let [e, hc, n, _] = res.ne();
    let [k, n_inj, _, _] = w.ne();
    assert!(
        !rebuild || quant != down.is_some(),
        "hc_post_norm: rebuild needs xq or the F16 copies"
    );
    assert!(
        !streams16 || (rebuild && !moe),
        "hc_post_norm: BF16 streams in the rebuilt form only"
    );
    assert_eq!(k, e * hc, "hc_post_norm: inject rows");
    assert!(!quant || k % 512 == 0, "hc_post_norm: quantized rows");
    assert!(
        !quant || down.is_none(),
        "hc_post_norm: F16 copies or xq, not both"
    );
    if let Some(d) = down {
        assert!(
            d.ty() == types::Q8_0 || d.ty() == types::BF16,
            "hc_post_norm: Q8_0 or BF16 down"
        );
        assert!(d.is_contiguous(), "hc_post_norm: down contiguous");
        assert_eq!(d.ne()[0], k, "hc_post_norm: down width");
        assert!(k % 2 == 0, "hc_post_norm: F16 xn of an odd width");
    }
    assert!(
        hc <= HC_NORM_MAX && n_inj <= HC_NORM_MAX,
        "hc_post_norm: {hc} streams, {n_inj} outputs"
    );
    assert_eq!(out.ne()[..2], [e, n], "hc_post_norm: out");
    assert_eq!(inj.ne()[..2], [hc, n], "hc_post_norm: inject of the block");
    for t in [out, inj, gamma, w] {
        assert_eq!(t.ty(), types::F32, "hc_post_norm: F32 tensors");
    }
    assert!(
        res.ty() == types::F32 || (streams16 && res.ty() == types::BF16),
        "hc_post_norm: F32 streams (or BF16 into BF16)"
    );
    assert!(
        gamma.is_contiguous() && w.is_contiguous(),
        "hc_post_norm: weights contiguous"
    );
    let es = if res.ty() == types::BF16 { 2 } else { 4 };
    assert_eq!(
        (res.nb()[0], res.nb()[1]),
        (es, e as usize * es),
        "hc_post_norm: streams contiguous"
    );
    assert_eq!(out.nb()[0], 4, "hc_post_norm: out rows contiguous");
    assert_eq!(inj.nb()[0], 4, "hc_post_norm: inject rows contiguous");
    let m = down.map_or(0, |d| d.ne()[1]);
    let at = PostNormLayout::new(
        k as usize,
        n as usize,
        n_inj as usize,
        down16_values(down.map(|d| d.ty()), (k * m) as usize),
        down.is_some(),
        quant,
        rebuild.then_some((if padded { k as usize } else { hc as usize }, streams16)),
    );
    if moe {
        moe_ffn::defer_sum(out);
    }
    let mut src = vec![res, out, inj, gamma, w];
    src.extend(down);
    if moe {
        // read by the kernel through the op's node; kept alive until it ran
        src.extend(moe_ffn::sum_srcs(out));
    }
    let node = ctx.external_with(
        &HC_POST_NORM,
        types::F32,
        [at.floats as i64, 1, 1, 1],
        &src,
        &[
            eps.to_bits() as i32,
            i32::from(down.is_some()),
            i32::from(quant),
            i32::from(rebuild),
            i32::from(moe),
            i32::from(streams16),
            i32::from(padded),
        ],
    );
    (node, at)
}

/// Where [`hc_post_norm`]'s outputs sit in its F32 node, in floats: `[res_new | xn | xn16 |
/// down16 | inject | xq]`, the F16 parts (empty unless asked for) each starting 16 bytes
/// aligned, as the F16 copies ggml makes for a GEMM do, and the quantized `xq` (empty unless
/// asked for) 256 bytes aligned. For [`hc_post_norm_rebuilt`] the `xn` part holds the
/// streams' scales (`hc` floats a token), and with BF16 streams (#826) `res_new` takes half
/// the floats.
#[derive(Clone, Copy)]
struct PostNormLayout {
    xn: usize,
    xn16: usize,
    down16: usize,
    inject: usize,
    /// In bytes.
    xq: usize,
    /// The node's size.
    floats: usize,
}

impl PostNormLayout {
    /// `k` values per token, `n` tokens, `n_inj` inject outputs, `down_values` in the down
    /// weight; `f16`: with the F16 copies; `quant`: with `xq`; `scales`: `Some((hc, bf16))` for
    /// the streams' scales in place of `xn` (`hc` floats a token; `e · hc` for the padded form,
    /// [`PostNormKind::padded`]), `bf16` for BF16 streams.
    fn new(
        k: usize,
        n: usize,
        n_inj: usize,
        down_values: usize,
        f16: bool,
        quant: bool,
        scales: Option<(usize, bool)>,
    ) -> Self {
        let floats = |halves: usize| if f16 { halves.div_ceil(8) * 4 } else { 0 };
        let streams = match scales {
            Some((_, true)) => (k * n).div_ceil(2),
            _ => k * n,
        };
        let xn16 = streams + scales.map_or(k, |(hc, _)| hc) * n;
        let down16 = xn16 + floats(k * n);
        let inject = down16 + floats(down_values);
        let end = inject + n_inj * n;
        let xq = (end * 4).next_multiple_of(256);
        PostNormLayout {
            xn: streams,
            xn16,
            down16,
            inject,
            xq,
            floats: if quant {
                (xq + mmq_preq::q8_1_bytes(k as i64, n as i64)).div_ceil(4)
            } else {
                end
            },
        }
    }
}

/// How many values of the down weight `down` (of type `ty`, `values` in all) [`hc_post_norm`]'s
/// node holds as F16: all of a Q8_0 one, none of a BF16 one (its matmul reads it as it is; only
/// `xn` is rounded, to BF16, #469) or of none.
fn down16_values(ty: Option<ffi::ggml_type>, values: usize) -> usize {
    if ty == Some(types::Q8_0) { values } else { 0 }
}

/// [`hc_post_norm`]'s outputs.
pub(super) struct PostNorm<'a> {
    /// The new streams, F32 `[e, hc, n]`.
    pub res: Tensor<'a>,
    /// The normed streams, F32 `[e · hc, n]`.
    pub xn: Tensor<'a>,
    /// `xn` rounded to F16 (`[e · hc, n]`), if asked for.
    pub xn16: Option<Tensor<'a>>,
    /// The down weight as F16 (`[e · hc, m]`), if asked for.
    pub down16: Option<Tensor<'a>>,
    /// `xn` rounded to BF16 (`[e · hc, n]`), for a BF16 down weight.
    pub xnb: Option<Tensor<'a>>,
    /// The per-stream weights, F32 `[n_inj, n]`.
    pub inject: Tensor<'a>,
    /// `xn` quantized for ggml-cuda's MMQ ([`mmq_preq::mul_mat`]'s `xq`), if asked for.
    pub xq: Option<Tensor<'a>>,
}

unsafe fn cpu_hc_post_norm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `hc_post_norm`, host tensors.
    unsafe {
        let (res, out, inj, g, w) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let (e, hc, n) = (res.ne[0] as usize, res.ne[1] as usize, res.ne[2] as usize);
        let n_inj = w.ne[1] as usize;
        let params = crate::ggml::external_params(dst);
        let eps = f32::from_bits(params[0] as u32);
        let k = e * hc;
        let (gp, wp) = (g.data as *const f32, w.data as *const f32);
        let down = (params[1] != 0).then(|| &*dst.src[5]);
        let m = down.map_or(0, |d| d.ne[1] as usize);
        let quant = params[2] != 0;
        let rebuild = params[3] != 0;
        // BF16 streams (#826): each new value rounded to BF16 before it is used.
        let streams16 = params[5] != 0;
        let at = PostNormLayout::new(
            k,
            n,
            n_inj,
            down16_values(down.map(|d| d.type_), k * m),
            down.is_some(),
            quant,
            rebuild.then_some((if params[6] != 0 { k } else { hc }, streams16)),
        );
        let rn = dst.data as *mut f32;
        let rn16 = dst.data as *mut u16;
        let res16 = res.type_ == types::BF16;
        let mut srow = vec![0f32; k];
        // `xn`, or (`rebuild`) the streams' scales.
        let xn = rn.add(at.xn);
        let mut row = vec![0f32; k];
        // A BF16 down weight: `xn` as BF16 (#469); a Q8_0 one: `xn` and the weight as F16.
        let xnb = down
            .filter(|d| d.type_ == types::BF16)
            .map(|_| rn.add(at.xn16) as *mut u16);
        let down = down.filter(|d| d.type_ == types::Q8_0);
        let xn16 = down.map(|_| rn.add(at.xn16) as *mut u16);
        let inject = rn.add(at.inject);
        if let Some(d) = down {
            // Q8_0 blocks of 32: an F16 scale, then 32 int8 values; d · q is exact in F32.
            const QK: usize = 32;
            const BLOCK: usize = 2 + QK;
            let (q8, d16) = (d.data as *const u8, rn.add(at.down16) as *mut u16);
            for b in (ith..k * m / QK).step_by(nth) {
                let block = q8.add(b * BLOCK);
                let scale = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*block, *block.add(1)]));
                for j in 0..QK {
                    let q = *(block.add(2 + j) as *const i8);
                    *d16.add(b * QK + j) = ffi::ggml_fp32_to_fp16(f32::from(q) * scale);
                }
            }
        }
        // Past the node unless `quant`.
        let xq = (dst.data as *mut u8).wrapping_add(at.xq);
        // A folded MoE tail (#836): the token's block output made here.
        let folded = params[7] != 0;
        let mut fold_row = vec![0f32; if folded { e } else { 0 }];
        for t in (ith..n).step_by(nth) {
            let rr = (res.data as *const u8).add(t * res.nb[2]);
            let res_at = |i: usize| {
                if res16 {
                    f32::from_bits(u32::from(*(rr as *const u16).add(i)) << 16)
                } else {
                    *(rr as *const f32).add(i)
                }
            };
            let orow = if folded {
                cpu_fold_row(dst, t, &mut fold_row);
                fold_row.as_ptr()
            } else {
                (out.data as *const u8).add(t * out.nb[1]) as *const f32
            };
            let irow = (inj.data as *const u8).add(t * inj.nb[1]) as *const f32;
            let mut acc = vec![0f32; n_inj];
            for h in 0..hc {
                let v = *irow.add(h) * (1.0 / hc as f32);
                let wh = 1.0 / (1.0 + (-v).exp()) * 2.0;
                let mut ss = 0f32;
                for i in 0..e {
                    let mut x = *orow.add(i) * wh + res_at(h * e + i);
                    if streams16 {
                        let b = crate::model::nvfp4_mm::bf16_bits(x);
                        *rn16.add(t * k + h * e + i) = b;
                        x = f32::from_bits(u32::from(b) << 16);
                    } else {
                        *rn.add(t * k + h * e + i) = x;
                    }
                    srow[h * e + i] = x;
                    ss += x * x;
                }
                let scale = 1.0 / (ss / e as f32 + eps).sqrt();
                if rebuild {
                    *xn.add(t * hc + h) = scale;
                }
                for i in 0..e {
                    let at = h * e + i;
                    let y = (srow[at] * scale) * *gp.add(at);
                    row[at] = y;
                    if !rebuild {
                        *xn.add(t * k + at) = y;
                    }
                    if let Some(x16) = xn16 {
                        *x16.add(t * k + at) = ffi::ggml_fp32_to_fp16(y);
                    }
                    if let Some(xb) = xnb {
                        *xb.add(t * k + at) = crate::model::nvfp4_mm::bf16_bits(y);
                    }
                    for (j, a) in acc.iter_mut().enumerate() {
                        *a += y * *wp.add(at + k * j);
                    }
                }
            }
            for (j, a) in acc.iter().enumerate() {
                *inject.add(t * n_inj + j) = *a;
            }
            // hc_post_norm_q: the token's xn quantized too, into its column of every block row.
            if quant {
                let mut col = vec![0u8; k / mmq_preq::BLOCK as usize * mmq_preq::BLOCK_BYTES];
                mmq_preq::quantize_ref(&row, k, 1, &mut col);
                let (blocks, _) = col.as_chunks::<{ mmq_preq::BLOCK_BYTES }>();
                for (b, blk) in blocks.iter().enumerate() {
                    let to = xq.add((b * n + t) * mmq_preq::BLOCK_BYTES);
                    std::ptr::copy_nonoverlapping(blk.as_ptr(), to, mmq_preq::BLOCK_BYTES);
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct HcPostNorm {
    res: *const f32,
    res_ld: i64,
    out: *const f32,
    out_ld: i64,
    inj: *const f32,
    inj_ld: i64,
    g: *const f32,
    w: *const f32,
    res_new: *mut f32,
    xn: *mut f32,
    inject: *mut f32,
    xn16: *mut u16,
    e: i32,
    hc: i32,
    n_inj: i32,
    n_tok: i32,
    eps: f32,
    /// 0, or (with `xq`) where the streams' scales go instead of `xn`, in floats from `res_new`
    /// ([`hc_post_norm_rebuilt`]). In the C struct's padding after `eps`.
    scale_at: i32,
    xq: *mut i8,
    /// Null, or where `xn` rounded to BF16 goes (a BF16 down weight, #469).
    xnb: *mut u16,
    /// Null, or (`legion_hc_post_norm_rs_launch`, #1176) device memory holding the addresses of
    /// rank 0's and rank 1's BF16 partials whose sum is `out`, laid out as `out` (`out_ld`),
    /// which the exchange writes there on the stream ([`crate::tp_net::seq_par::sum_slot`]).
    rs: *const u64,
}

/// `legion_moe_fold` (src/cuda/fused.cu): a folded MoE tail's inputs ([`fold_moe`]).
#[cfg(feature = "cuda")]
#[repr(C)]
struct MoeFold {
    x: *const f32,
    w: *const f32,
    sh: *const f32,
    g: *const f32,
    g_ld: i64,
    k: i32,
    fma: i32,
    /// Null, or the experts' rows as BF16 (`x` null; `--nvfp4-expert-rows-bf16`, #1183).
    x16: *const u16,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_post_norm_fold_launch(
        p: *const HcPostNorm,
        f: *const MoeFold,
        stream: *mut c_void,
    ) -> i32;
    fn legion_hc_post_norm_launch(p: *const HcPostNorm, stream: *mut c_void) -> i32;
    fn legion_hc_post_norm_rs_launch(p: *const HcPostNorm, stream: *mut c_void) -> i32;
    fn legion_hc_post_norm_s16_launch(p: *const HcPostNorm, in16: i32, stream: *mut c_void) -> i32;
    fn legion_hc_post_norm_moe_launch(
        p: *const HcPostNorm,
        s: *const moe_ffn::PairSum,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    fn legion_hc_post_norm_force_generic(on: i32);
    fn legion_hc_post_norm_set_generic(on: i32);
    /// ggml-cuda's `to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type)` (`ggml-cuda/convert.cuh`,
    /// a C++ symbol; `--wrap`ped as for ggml's own calls, see `build/cuda_wrap.rs`).
    #[link_name = "_Z21ggml_get_to_fp16_cuda9ggml_type"]
    fn ggml_get_to_fp16_cuda(ty: ffi::ggml_type) -> Option<ToF16>;
}

/// ggml-cuda's `to_fp16_cuda_t`: converts `k` values at `x` to F16 at `y` on `stream`.
#[cfg(feature = "cuda")]
type ToF16 = unsafe extern "C" fn(x: *const c_void, y: *mut u16, k: i64, stream: *mut c_void);

/// Whether the hyper-connection combine and the next norm run as [`hc_post_norm`] where it fits
/// (on by default).
static FUSED_POST_NORM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether graphs built from now on fuse the hyper-connection combine and norm; off keeps
/// ggml's combine and the separate norm (for A/B runs, `legion serve --hc-post-norm-ggml`).
pub fn set_fused_post_norm(on: bool) {
    FUSED_POST_NORM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// See [`set_fused_post_norm`].
pub(super) fn fused_post_norm() -> bool {
    FUSED_POST_NORM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether CUDA graphs fold MoE tails into [`hc_post_norm_rebuilt_xnb`] ([`set_moe_fold`]).
static MOE_FOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether CUDA graphs allocated from now on have a MoE layer's tail (the routed experts'
/// weighted sum and the shared expert's gate) made by the next [`hc_post_norm_rebuilt_xnb`]
/// from the tail's inputs instead of written by the tail's kernels and read back ([`fold_moe`],
/// #836, src/model/core/node_fuse/fold.rs; the same bits). Off until set (the engine sets it on
/// an integrated GPU; `legion serve --hc-moe-fold` / `--no-hc-moe-fold`).
pub fn set_moe_fold(on: bool) {
    MOE_FOLD.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_moe_fold`] is on.
pub(crate) fn moe_fold() -> bool {
    MOE_FOLD.load(std::sync::atomic::Ordering::Relaxed)
}

/// How a folded MoE tail ([`fold_moe`]) sums the routed experts' rows, as the layer's own nodes
/// round it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FoldSum {
    /// The router weights' products, each rounded, added in order (ggml's `mul` and adds).
    Plain,
    /// The first product, then one FMA per expert (ggml-cuda's fused weighted sum).
    Fma,
}

/// `params[7]` of a folded node: 0 unfolded, else the [`FoldSum`]'s code.
fn fold_code(f: FoldSum) -> i32 {
    match f {
        FoldSum::Plain => 1,
        FoldSum::Fma => 2,
    }
}

/// Whether `node` is an unfolded [`hc_post_norm_rebuilt_xnb`] node a MoE tail may be folded
/// into ([`fold_moe`]): F32 streams, a BF16 down weight, and on CUDA (`cuda`) qwen4exp's 4 x
/// 2560 streams with 4 inject outputs (the fold kernel's shape).
///
/// # Safety
/// `node` is a valid tensor.
pub(crate) unsafe fn fold_target(node: *const ffi::ggml_tensor, cuda: bool) -> bool {
    // SAFETY: the caller's; an hc_post_norm node's sources are valid tensors.
    unsafe {
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &HC_POST_NORM)) {
            return false;
        }
        let n = &*node;
        let p = crate::ggml::external_params(n);
        // eps, down, quant, rebuild, moe, streams16, padded, fold
        if p[1..8] != [1, 0, 1, 0, 0, 1, 0]
            || p[P_RS] != 0
            || n.src[5].is_null()
            || n.src[6..].iter().any(|s| !s.is_null())
        {
            return false;
        }
        let (res, w, down) = (&*n.src[0], &*n.src[4], &*n.src[5]);
        res.type_ == types::F32
            && down.type_ == types::BF16
            && (!cuda || (res.ne[0] == 2560 && res.ne[1] == 4 && w.ne[1] == 4))
    }
}

/// Makes the [`fold_target`] `node` compute its block output itself: per token, the `k` rows
/// of `x` (F32 `[e, k, n]`, contiguous) times the router weights `w` (F32 `[1, k, n]`,
/// contiguous), summed as `sum` says, plus the shared expert's output
/// `sh` (F32 `[e, n]`, contiguous) times the sigmoid of its gate logit `g` (F32 `[1, n]`, any
/// row stride), each step rounded as ggml rounds it (#836). The node then reads those tensors
/// at its own position, not the tail's output.
///
/// # Safety
/// As [`crate::ggml::make_external`]; `node` passed [`fold_target`], the inputs fit.
pub(crate) unsafe fn fold_moe(
    node: *mut ffi::ggml_tensor,
    x: *mut ffi::ggml_tensor,
    w: *mut ffi::ggml_tensor,
    k: usize,
    sum: FoldSum,
    sh: *mut ffi::ggml_tensor,
    g: *mut ffi::ggml_tensor,
) {
    // SAFETY: the caller's.
    unsafe {
        let n = &*node;
        let mut params = crate::ggml::external_params(n);
        params[7] = fold_code(sum);
        params[8] = k as i32;
        let mut srcs = n.src[..6].to_vec();
        srcs[1] = x;
        srcs.extend([sh, g, w]);
        crate::ggml::make_external(node, &HC_POST_NORM, &srcs, &params[..9]);
    }
}

/// A folded node's ([`fold_moe`]) block output for token `t` into `out` (`e` values), as the
/// tail's ggml nodes compute it on the CPU (the FMA sum as ggml-cuda's fused kernel).
///
/// # Safety
/// `dst` is a folded node of host tensors, `t` one of its tokens.
unsafe fn cpu_fold_row(dst: &ffi::ggml_tensor, t: usize, out: &mut [f32]) {
    // SAFETY: the caller's.
    unsafe {
        let params = crate::ggml::external_params(dst);
        let (x, sh, g) = (&*dst.src[1], &*dst.src[6], &*dst.src[7]);
        let w = (*dst.src[8]).data as *const f32;
        let fma = params[7] == 2;
        let (e, k) = (out.len(), params[8] as usize);
        // BF16 rows (`--nvfp4-expert-rows-bf16`, #1183): read as F32, exactly.
        let x16 = x.type_ == types::BF16;
        let xp = x.data as *const f32;
        let xh = x.data as *const u16;
        let gv = *((g.data as *const u8).add(t * g.nb[1]) as *const f32);
        let s = 1.0 / (1.0 + (-gv).exp());
        let shp = (sh.data as *const f32).add(t * e);
        for (col, o) in out.iter_mut().enumerate() {
            let v = |j: usize| {
                let i = (t * k + j) * e + col;
                if x16 {
                    f32::from_bits((*xh.add(i) as u32) << 16)
                } else {
                    *xp.add(i)
                }
            };
            let mut sum = v(0) * *w.add(t * k);
            for j in 1..k {
                sum = if fma {
                    v(j).mul_add(*w.add(t * k + j), sum)
                } else {
                    sum + v(j) * *w.add(t * k + j)
                };
            }
            *o = sum + *shp.add(col) * s;
        }
    }
}

/// `legion serve --hc-streams-bf16` ([`set_streams_bf16`]; off by default).
static STREAMS_BF16: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether graphs built from now on keep qwen4exp's residual streams in BF16 between the
/// mixers that take [`hc_post_norm_rebuilt`] (prompt steps on a GB10 with Q8_0 mixers, #826):
/// half the bytes of the streams written and read twice per mixer, but each new stream value
/// is rounded to BF16, so output moves (opt-in). Every other reader of the streams gets them
/// converted back to F32.
pub fn set_streams_bf16(on: bool) {
    STREAMS_BF16.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`hc_post_norm_rebuilt`] of the streams `res` with the inject weights `w` writes
/// BF16 streams: [`set_streams_bf16`] on and qwen4exp's 4 x 2560 streams with 4 inject outputs
/// (the shape of the kernel, `hc_post_norm_s16_kernel` in `src/cuda/fused.cu`).
pub(super) fn streams_bf16_fits(res: Tensor<'_>, w: Tensor<'_>) -> bool {
    STREAMS_BF16.load(std::sync::atomic::Ordering::Relaxed)
        && res.ne()[..2] == [2560, 4]
        && w.ne()[1] == 4
}

/// Whether [`hc_post_norm`] also writes `xn` as F16 for the mixer's down projection where
/// [`xn16_fits`] (on by default).
static XN16: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the engine's GPU is one where that gives ggml's bits (set when the engine loads).
static XN16_GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Smallest step [`xn16_fits`] takes: ggml-cuda's `MMQ_DP4A_MAX_BATCH_SIZE`; below it ggml runs
/// a quantized `mul_mat` on V100 as MMQ (its own rounding of `xn`), not cuBLAS.
pub(super) const XN16_MIN_TOKENS: i64 = 64;

/// Whether graphs built from now on hand the mixer's down projection an F16 `xn` written by
/// [`hc_post_norm`]; off keeps ggml's conversion inside the matmul (for A/B runs,
/// `legion serve --hc-xn-f32`). The same bits either way.
pub fn set_xn16(on: bool) {
    XN16.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The engine's GPU (compute capability major, minor). Only on a V100 (7.0) does ggml-cuda run
/// a quantized `mul_mat` of [`XN16_MIN_TOKENS`] or more columns as one cuBLAS F16 GEMM with F32
/// output after rounding `xn` to F16 (`ggml_cuda_mul_mat_cublas`), which is what [`xn16_fits`]
/// replaces; sm_75+ GPUs take MMQ there. Off, too, when `GGML_CUDA_CUBLAS_COMPUTE_TYPE` changes
/// that GEMM's compute type.
pub fn set_xn16_gpu(cc: Option<(i32, i32)>) {
    V70.store(cc == Some((7, 0)), std::sync::atomic::Ordering::Relaxed);
    SM80.store(
        cc.is_some_and(|(major, _)| major >= 8),
        std::sync::atomic::Ordering::Relaxed,
    );
    let ggml_default = std::env::var("GGML_CUDA_CUBLAS_COMPUTE_TYPE").map_or(true, |v| {
        matches!(v.to_lowercase().as_str(), "auto" | "f16" | "fp16")
    });
    XN16_CUBLAS.store(ggml_default, std::sync::atomic::Ordering::Relaxed);
    XN16_GPU.store(
        xn16_takes(cc, ggml_default),
        std::sync::atomic::Ordering::Relaxed,
    );
    // A BF16 weight's GEMM computes in BF16 unless the variable asks for F32 or F16.
    let bf16_default = std::env::var("GGML_CUDA_CUBLAS_COMPUTE_TYPE").map_or(true, |v| {
        matches!(v.to_lowercase().as_str(), "auto" | "bf16")
    });
    XNB_CUBLAS.store(bf16_default, std::sync::atomic::Ordering::Relaxed);
    XNB_GPU.store(
        xnb_takes(cc, bf16_default),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Whether `GGML_CUDA_CUBLAS_COMPUTE_TYPE` leaves the F16 GEMM [`xn16_fits`] replaces as it is
/// (set with [`set_xn16_gpu`]).
static XN16_CUBLAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether `GGML_CUDA_CUBLAS_COMPUTE_TYPE` leaves the BF16 GEMM [`xnb_fits`] feeds as it is (set
/// with [`set_xn16_gpu`]).
static XNB_CUBLAS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a GPU of compute capability `cc` takes [`xn16_fits`] (a V100), the cuBLAS compute
/// type ggml's (`cublas`).
fn xn16_takes(cc: Option<(i32, i32)>, cublas: bool) -> bool {
    cc == Some((7, 0)) && cublas
}

/// Whether a GPU of compute capability `cc` takes [`xnb_fits`] (sm_80+), the cuBLAS compute type
/// ggml's (`cublas`).
fn xnb_takes(cc: Option<(i32, i32)>, cublas: bool) -> bool {
    cc.is_some_and(|(major, _)| major >= 8) && cublas
}

/// The answer of `takes` (with the cuBLAS setting `cublas`) for the GPU of device `dev` (the
/// down weight's), asked per device (#1201: a box may hold GPUs of different architectures);
/// `engine` (the engine's GPU's, [`set_xn16_gpu`]) where `dev` is unknown.
fn gpu_for_by(
    dev: Option<usize>,
    engine: bool,
    takes: fn(Option<(i32, i32)>, bool) -> bool,
    cublas: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, |cc| takes(cc, cublas), cc_of)
}

/// [`xn16_fits`]'s GPU check for the down weight `down`.
fn xn16_gpu(down: Tensor<'_>) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    gpu_for_by(
        down.cuda_device(),
        XN16_GPU.load(Relaxed),
        xn16_takes,
        XN16_CUBLAS.load(Relaxed),
        crate::hw::per_device::cc,
    )
}

/// [`xnb_fits`]'s GPU check for the down weight `down`.
fn xnb_gpu(down: Tensor<'_>) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    gpu_for_by(
        down.cuda_device(),
        XNB_GPU.load(Relaxed),
        xnb_takes,
        XNB_CUBLAS.load(Relaxed),
        crate::hw::per_device::cc,
    )
}

/// Whether a prompt step of `n` tokens hands `mul_mat(down, xn)` an F16 `xn` from
/// [`hc_post_norm`] (with `down` as F16 from it too): switched on, at least [`XN16_MIN_TOKENS`]
/// tokens and Q8_0 down weights in the memory of one V100 ([`xn16_gpu`]), so ggml-cuda would
/// run the same GEMM on the same F16 inputs.
pub(super) fn xn16_fits(down: Tensor<'_>, n: i64) -> bool {
    XN16.load(std::sync::atomic::Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && n >= XN16_MIN_TOKENS
        && down.ty() == types::Q8_0
        && down.is_contiguous()
        && down.ne()[2] == 1
        && down.ne()[3] == 1
        && down.ne()[0] % 2 == 0
        && !down.is_host()
        && !down.is_meta()
        && xn16_gpu(down)
}

/// Whether the engine's GPU runs ggml-cuda's BF16 `mul_mat` as a cuBLAS BF16 GEMM that reads a
/// BF16 `xn` as it is (sm_80+, the cuBLAS compute type not overridden; set when the engine
/// loads, [`set_xn16_gpu`]).
static XNB_GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Smallest step [`xnb_fits`] takes: prompt chunks only, so decode and verify graphs (up to
/// 16 sequences of 1 + 3 drafted tokens) stay as they were. ggml-cuda runs a BF16 weight's
/// `mul_mat` of more than 16 columns on cuBLAS whether `xn` comes as F32 or BF16.
pub(super) const XNB_MIN_TOKENS: i64 = 128;

/// Whether a prompt step of `n` tokens hands `mul_mat(down, xn)` a BF16 `xn` from
/// [`hc_post_norm`] (#469): switched on ([`set_xn16`]), at least [`XNB_MIN_TOKENS`] tokens and a
/// BF16 down weight (the NVFP4 checkpoints' mixers) in the memory of one sm_80+ GPU
/// ([`xnb_gpu`]). ggml-cuda would round `xn` to BF16 as the op does (`__float2bfloat16`) and
/// run the same cuBLAS GEMM on it, so the result has the same bits, minus the conversion pass.
pub(super) fn xnb_fits(down: Tensor<'_>, n: i64) -> bool {
    XN16.load(std::sync::atomic::Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && n >= XNB_MIN_TOKENS
        && down.ty() == types::BF16
        && down.is_contiguous()
        && down.ne()[2] == 1
        && down.ne()[3] == 1
        && down.ne()[0] % 2 == 0
        && !down.is_host()
        && !down.is_meta()
        && xnb_gpu(down)
}

/// Whether the engine's GPU is a V100 (compute capability 7.0; set with [`set_xn16_gpu`]).
static V70: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the engine's GPU is compute capability 8.0 or newer (a GB10; set with
/// [`set_xn16_gpu`]).
static SM80: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `legion serve --hc-fuse` ([`set_hc_fuse`]).
static HC_FUSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`set_generic_post_norm`]'s setting.
static GENERIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether graphs built from now on fuse qwen4exp's hyper-connection glue around the MoE
/// op (`legion serve --hc-fuse`, off by default; #774 on a V100, #826 on sm_80+ GPUs such as
/// the GB10, where it also takes the rebuilt form with `xq`): the layer's gated shared expert
/// added in the MoE op's per-token sum (as on the GB10, #526, ggml's sigmoid, mul and add
/// bits), and that sum made by [`hc_post_norm`] from the op's per-pair rows instead of written
/// by the op's sum kernel and read back. The same bits per op as without; the graph changes,
/// so ggml-cuda's layout-dependent fusions (#472) can move greedy output.
pub fn set_hc_fuse(on: bool) {
    HC_FUSE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_hc_fuse`] is on and takes this engine: a CUDA build on a V100 or an sm_80+
/// GPU, with [`hc_post_norm`]'s register kernels (not [`set_generic_post_norm`]).
pub fn hc_fuse() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let gpu = V70.load(Relaxed) || SM80.load(Relaxed);
    cfg!(feature = "cuda") && HC_FUSE.load(Relaxed) && gpu && !GENERIC.load(Relaxed)
}

/// Whether [`hc_post_norm`] of `res` takes `out` over as the MoE op's sum ([`set_hc_fuse`]):
/// on, qwen4exp's 4 x 2560 streams and 4 inject outputs (the kernel's shape), and `out` the
/// whole sum of a fused MoE op ([`moe_ffn::deferrable`]), whose only reader is the norm.
pub(super) fn moe_sum_fits(res: Tensor<'_>, out: Tensor<'_>, w: Tensor<'_>) -> bool {
    hc_fuse() && res.ne()[..2] == [2560, 4] && w.ne()[1] == 4 && moe_ffn::deferrable(out)
}

/// Every [`hc_post_norm`] from now on takes the generic kernel rather than the one holding the
/// values in registers (`legion serve --hc-post-norm-generic`, for comparisons).
pub fn set_generic_post_norm(on: bool) {
    GENERIC.store(on, std::sync::atomic::Ordering::Relaxed);
    #[cfg(feature = "cuda")]
    // SAFETY: sets a flag the launches read.
    unsafe {
        legion_hc_post_norm_set_generic(i32::from(on))
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// Tests: [`hc_post_norm`] launches from the calling thread take the generic kernel (`true`)
/// rather than the one holding the values in registers.
#[cfg(all(test, feature = "cuda"))]
fn force_generic_post_norm(on: bool) {
    force_post_norm_kernel(i32::from(on));
}

/// Tests: [`hc_post_norm`] launches from the calling thread take the generic kernel (1), the
/// register kernel `hc_post_norm_reg_kernel` (2) or the default choice (0).
#[cfg(all(test, feature = "cuda"))]
pub(super) fn force_post_norm_kernel(mode: i32) {
    // SAFETY: sets a thread-local flag the launches read.
    unsafe { legion_hc_post_norm_force_generic(mode) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_post_norm(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `hc_post_norm`, device tensors of the stream's device.
    unsafe {
        let (res, out, inj, g, w) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let (e, hc, n) = (res.ne[0], res.ne[1], res.ne[2]);
        let params = crate::ggml::external_params(dst);
        let down = (params[1] != 0).then(|| &*dst.src[5]);
        let m = down.map_or(0, |d| d.ne[1]);
        let quant = params[2] != 0;
        let rebuild = params[3] != 0;
        let streams16 = params[5] != 0;
        let at = PostNormLayout::new(
            (e * hc) as usize,
            n as usize,
            w.ne[1] as usize,
            down16_values(down.map(|d| d.type_), (e * hc * m) as usize),
            down.is_some(),
            quant,
            rebuild.then_some((
                if params[6] != 0 {
                    (e * hc) as usize
                } else {
                    hc as usize
                },
                streams16,
            )),
        );
        // A Q8_0 down weight: xn and the weight as F16; a BF16 one: xn as BF16 (#469).
        let bf16 = down.is_some_and(|d| d.type_ == types::BF16);
        let down = down.filter(|d| d.type_ == types::Q8_0);
        let rn = dst.data as *mut f32;
        let res16 = res.type_ == types::BF16;
        let mut p = HcPostNorm {
            res: res.data as *const f32,
            // In the streams' own values (BF16 ones for BF16 streams).
            res_ld: (res.nb[2] / if res16 { 2 } else { 4 }) as i64,
            out: out.data as *const f32,
            out_ld: (out.nb[1] / 4) as i64,
            inj: inj.data as *const f32,
            inj_ld: (inj.nb[1] / 4) as i64,
            g: g.data as *const f32,
            w: w.data as *const f32,
            res_new: rn,
            xn: rn.add(at.xn),
            inject: rn.add(at.inject),
            xn16: match down {
                Some(_) => rn.add(at.xn16) as *mut u16,
                None => std::ptr::null_mut(),
            },
            e: e as i32,
            hc: hc as i32,
            n_inj: w.ne[1] as i32,
            n_tok: n as i32,
            eps: f32::from_bits(params[0] as u32),
            // Below i32::MAX: `rebuild_fits`.
            scale_at: if rebuild { at.xn as i32 } else { 0 },
            xq: if quant {
                (dst.data as *mut i8).add(at.xq)
            } else {
                std::ptr::null_mut()
            },
            xnb: if bf16 {
                rn.add(at.xn16) as *mut u16
            } else {
                std::ptr::null_mut()
            },
            rs: std::ptr::null(),
        };
        // `out` a deferred reduce-scatter (#1176): the two ranks' BF16 partials added here,
        // their addresses read on the GPU from the slot the exchange wrote them to, so a
        // stage graph may capture this launch. The exchange frees its buffers later.
        let code = if params[P_RS] != 0 && crate::tp_net::seq_par::deferred(out) {
            p.rs = crate::tp_net::seq_par::sum_slot(out)? as *const u64;
            legion_hc_post_norm_rs_launch(&p, stream)
        } else if params[7] != 0 {
            // A folded MoE tail (#836): `out` (the experts' rows) is not read as a row.
            let (sh, g) = (&*dst.src[6], &*dst.src[7]);
            let x16 = out.type_ == types::BF16;
            let f = MoeFold {
                x: if x16 {
                    std::ptr::null()
                } else {
                    out.data as *const f32
                },
                x16: if x16 {
                    out.data as *const u16
                } else {
                    std::ptr::null()
                },
                w: (*dst.src[8]).data as *const f32,
                sh: sh.data as *const f32,
                g: g.data as *const f32,
                g_ld: (g.nb[1] / 4) as i64,
                k: params[8],
                fma: i32::from(params[7] == 2),
            };
            legion_hc_post_norm_fold_launch(&p, &f, stream)
        } else if streams16 {
            // BF16 streams (#826).
            legion_hc_post_norm_s16_launch(&p, i32::from(res16), stream)
        } else if params[4] != 0 {
            // `out` is a MoE op's deferred sum: made here from the op's pairs (#774).
            let s = moe_ffn::pair_sum(&*out.view_src);
            legion_hc_post_norm_moe_launch(&p, &s, stream)
        } else {
            legion_hc_post_norm_launch(&p, stream)
        };
        if code != 0 {
            return Err(format!("hc_post_norm: error {code}"));
        }
        if let Some(d) = down {
            // The converter ggml-cuda's cuBLAS matmul path dequantizes the weight with.
            let Some(to_f16) = ggml_get_to_fp16_cuda(d.type_) else {
                return Err("hc_post_norm: no F16 conversion of the down weight".into());
            };
            let values = d.ne[0] * d.ne[1];
            to_f16(d.data, rn.add(at.down16) as *mut u16, values, stream);
        }
        Ok(())
    }
}

unsafe fn cpu_hc_norm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `hc_norm`, host tensors.
    unsafe {
        let (x, g, w) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (e, hc, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let n_inj = w.ne[1] as usize;
        let eps = f32::from_bits(crate::ggml::external_params(dst)[0] as u32);
        let k = e * hc;
        let (gp, wp) = (g.data as *const f32, w.data as *const f32);
        let xn = dst.data as *mut f32;
        let inject = xn.add(k * n);
        for t in (ith..n).step_by(nth) {
            let xr = (x.data as *const u8).add(t * x.nb[2]) as *const f32;
            let mut acc = vec![0f32; n_inj];
            for h in 0..hc {
                let row = std::slice::from_raw_parts(xr.add(h * e), e);
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let scale = 1.0 / (ss / e as f32 + eps).sqrt();
                for (i, v) in row.iter().enumerate() {
                    let at = h * e + i;
                    let y = (v * scale) * *gp.add(at);
                    *xn.add(t * k + at) = y;
                    for (j, a) in acc.iter_mut().enumerate() {
                        *a += y * *wp.add(at + k * j);
                    }
                }
            }
            for (j, a) in acc.iter().enumerate() {
                *inject.add(t * n_inj + j) = *a;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct HcNorm {
    x: *const f32,
    x_ld: i64,
    g: *const f32,
    w: *const f32,
    xn: *mut f32,
    inject: *mut f32,
    e: i32,
    hc: i32,
    n_inj: i32,
    n_tok: i32,
    eps: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_hc_norm_launch(p: *const HcNorm, stream: *mut c_void) -> i32;
    fn legion_hc_norm_set_fma(on: i32);
    #[cfg(test)]
    fn legion_hc_norm_force_fma(on: i32);
}

/// Every [`hc_norm`] from now on on a compute capability 7.x GPU adds each square and inject
/// term with one FMA, as [`hc_post_norm`] does there, so the two give the same bits for the
/// same streams (`legion serve --hc-norm-fma`, #409). Off (the default), nvcc's contraction
/// adds every fourth square with a separate multiply and add, an ulp apart at times; it moves
/// prompt steps' layer-0 mixer (and the MTP head's) by that ulp, so it stays opt-in. No effect
/// on other GPUs, where the two kernels already agree.
pub fn set_norm_fma(on: bool) {
    #[cfg(feature = "cuda")]
    // SAFETY: sets a flag the launches read.
    unsafe {
        legion_hc_norm_set_fma(i32::from(on))
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// Tests: [`hc_norm`] launches from the calling thread take the FMA form (`true`), as
/// [`set_norm_fma`] does for every thread.
#[cfg(all(test, feature = "cuda"))]
fn force_norm_fma(on: bool) {
    // SAFETY: sets a thread-local flag the launches read.
    unsafe { legion_hc_norm_force_fma(i32::from(on)) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_hc_norm(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `hc_norm`, device tensors of the stream's device.
    unsafe {
        let (x, g, w) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (e, hc, n) = (x.ne[0], x.ne[1], x.ne[2]);
        let xn = dst.data as *mut f32;
        let p = HcNorm {
            x: x.data as *const f32,
            x_ld: (x.nb[2] / 4) as i64,
            g: g.data as *const f32,
            w: w.data as *const f32,
            xn,
            inject: xn.add((e * hc * n) as usize),
            e: e as i32,
            hc: hc as i32,
            n_inj: w.ne[1] as i32,
            n_tok: n as i32,
            eps: f32::from_bits(crate::ggml::external_params(dst)[0] as u32),
        };
        match legion_hc_norm_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("hc_norm: error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
