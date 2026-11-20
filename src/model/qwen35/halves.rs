//! A dense Qwen3.5+ prompt step under tensor parallelism over machines with each layer's sums
//! in two halves of its tokens (#1133), as Gemma 4's (#619): the mixer (gated attention or
//! gated DeltaNet) runs whole, its output projection's partial is summed per half, and each
//! half's FFN runs as soon as its sum is in, so that the halves' transfers run while the GPU
//! computes: half 1's mixer sum during half 0's FFN, half 0's FFN sum during half 1's FFN.
//!
//! The output is the plain layer's bit for bit: the halves' work is per token (the residual
//! adds, the RMS norms and the FFN, whose matmuls must all run on the W4A4 dense kernels at the
//! halves' and the whole step's batch, [`Model::a4_dense`]: every token the same bits at any of
//! them), and every half's sum gives each value the plain all-reduce's bits
//! ([`micro::all_reduce_start`]). A layer whose FFN does not fit (e.g. an FP8 MLP) runs whole.

#[cfg(test)]
mod tests;

use crate::ggml::Tensor;
use crate::model::core::{FfnFusion, Model, ModelError, Pass, cols};
use crate::tp_net::micro;

/// Tokens each half of a prompt step needs (a shorter step has little wire time to hide; the
/// same floor as Gemma 4's).
pub(super) const TP_HALF_MIN: i64 = 1024;

/// The FFN weights of a layer that runs in halves ([`ffn_fits`]).
const FFN: [&str; 3] = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];

/// Whether prompt steps may run their layers' sums in halves: on, the same bits as whole;
/// `LEGION_DEBUG_QWEN35_TP_HALVES=0` keeps them whole. Read once.
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        on_of(
            std::env::var("LEGION_DEBUG_QWEN35_TP_HALVES")
                .ok()
                .as_deref(),
        )
    })
}

/// [`on`] for the variable's value `v`.
fn on_of(v: Option<&str>) -> bool {
    v != Some("0")
}

/// The communicator this pass's layers sum their halves over: a prompt step of at least twice
/// [`TP_HALF_MIN`] tokens of a dense model under tensor parallelism over machines (not
/// sequence-parallel), without taps, and [`on`]. The same answer on every rank.
pub(super) fn comm(m: &Model, pass: &Pass<'_>) -> Option<i32> {
    if !on() || pass.tap.on || m.hp.moe {
        return None;
    }
    let comm = m.tp_net_comm()?;
    let n = pass.n;
    if n / 2 < TP_HALF_MIN || m.tp_seq_par(n).is_some() {
        return None;
    }
    Some(comm)
}

/// Whether layer `il` of a step of `n` tokens runs in halves: every FFN matmul on the W4A4
/// dense kernels at both halves' batches and the whole step's.
pub(super) fn ffn_fits(m: &Model, il: usize, n: i64) -> bool {
    let n0 = n / 2;
    FFN.iter()
        .all(|name| [n0, n - n0, n].iter().all(|&c| m.a4_dense(il, name, c)))
}

/// The halves of the layer input `x` (`[n_embd, n]`), put into the graph at the layer's start:
/// placed later, a view could sit between two ops ggml-cuda fuses in the plain layer.
pub(super) fn views<'a>(pass: &Pass<'a>, x: Tensor<'a>) -> [Tensor<'a>; 2] {
    let (ctx, n) = (pass.ctx, pass.n);
    let xs = [cols(ctx, x, 0, n / 2), cols(ctx, x, n / 2, n - n / 2)];
    xs.iter().for_each(|&v| pass.graph.borrow_mut().expand(v));
    xs
}

/// The rest of layer `il` from its mixer's partial output `attn` (this rank's, `[n_embd, n]`)
/// and the halves `xs` of its input ([`views`]): per half the mixer's sum, the residual add,
/// the post-attention norm, the FFN, its sum and the residual add, the sums started over
/// `comm` in the order that overlaps them with the other half's FFN. Returns the layer's
/// output, both halves' FFN sums written into one tensor (#1133 r2: no concatenation).
pub(super) fn tail<'a>(
    m: &'a Model,
    pass: &Pass<'a>,
    il: usize,
    xs: [Tensor<'a>; 2],
    attn: Tensor<'a>,
    comm: i32,
    next: Option<crate::tp_net::Norm<'a>>,
) -> Result<(Tensor<'a>, Option<crate::tp_net::Normed<'a>>), ModelError> {
    let (ctx, n) = (pass.ctx, pass.n);
    let (n0, n1) = (n / 2, n - n / 2);
    let whole = n as usize;
    let add = |t: Tensor<'a>| pass.graph.borrow_mut().expand(t);
    // Its columns as views ([`cols`]): a copy where the projection's output is strided.
    let attn = if attn.is_contiguous() {
        attn
    } else {
        ctx.cont_2d(attn, attn.ne()[0], n)
    };
    let start = |t: Tensor<'a>| {
        let s = micro::all_reduce_start(ctx, comm, t, whole);
        add(s.node);
        s
    };
    let s0 = start(cols(ctx, attn, 0, n0));
    let s1 = start(cols(ctx, attn, n0, n1));
    let norm_w = m.lw(il, "post_attention_norm.weight")?;
    // A half's mixer sum finished with its residual added (#1133 r2: in the sum's kernel),
    // and its FFN's sum started. The post-attention norm rides in the W4A4 SwiGLU's
    // quantization, as on one GPU (ggml's bits).
    let ffn = |s: micro::Started<'a>, x: Tensor<'a>, width: i64| -> Result<_, ModelError> {
        let x_attn = micro::finish_add(ctx, s, x, width, 0, None);
        add(x_attn);
        let h = m.norm_mul(pass, x_attn, norm_w);
        let fusion = FfnFusion {
            res: None,
            norm: Some((x_attn, norm_w, m.hp.rms_eps)),
        };
        Ok((x_attn, start(m.ffn_fused(ctx, il, h, fusion)?)))
    };
    let (xa0, f0) = ffn(s0, xs[0], n0)?;
    let (xa1, f1) = ffn(s1, xs[1], n1)?;
    // The FFN sums with their residuals, both halves into one `[n_embd, n]` output; with
    // `next`, the next layer's input norm in the same kernels (its sums' bits, #1133).
    if let Some(nm) = next {
        let y0 = micro::finish_add_norm(ctx, f0, xa0, n, 0, None, nm);
        add(y0);
        let y = micro::finish_add_norm(ctx, f1, xa1, n, n0, Some(y0), nm);
        add(y);
        let nd = crate::tp_net::normed(ctx, y, xa0.ne()[0], n, nm.bf16);
        return Ok((nd.y, Some(nd)));
    }
    let y0 = micro::finish_add(ctx, f0, xa0, n, 0, None);
    add(y0);
    let y = micro::finish_add(ctx, f1, xa1, n, n0, Some(y0));
    add(y);
    Ok((y, None))
}
