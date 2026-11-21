//! A dense Qwen3.5+ prompt step under tensor parallelism over machines with each layer's sums
//! in parts of its tokens (#1133), as Gemma 4's halves (#619): the mixer (gated attention or
//! gated DeltaNet) runs whole, its output projection's partial is summed per half, and each
//! FFN part runs as soon as its half's sum is in, so that the transfers run while the GPU
//! computes: half 1's mixer sum during half 0's FFN, each FFN part's sum during the next
//! part's FFN.
//!
//! The FFN runs on half 0 whole and on half 1 in [`plan`]'s parts, the last of them small: the
//! last part's sum travels while the GPU has nothing left to do in the layer, so a small last
//! part leaves less of the wire time in sight (an 8K step over two DGX Sparks: half 1 in
//! 2,560 + 1,024 + 512 tokens, its last sum an eighth of the halves' last one). The mixer sum stays
//! in two halves: the output projection writes both straight into the send buffer
//! ([`micro::send_gemm`]), which needs every start of it right behind it, and at most two
//! exchanges are in flight.
//!
//! The output is the plain layer's bit for bit: the parts' work is per token (the residual
//! adds, the RMS norms and the FFN, whose matmuls must all run on the W4A4 dense kernels at
//! every part's and the whole step's batch, [`Model::a4_dense`]: every token the same bits at
//! any of them), and every part's sum gives each value the plain all-reduce's bits
//! ([`micro::all_reduce_start`]). A layer whose FFN does not fit (e.g. an FP8 MLP) runs whole.

#[cfg(test)]
mod tests;

use crate::ggml::Tensor;
use crate::model::core::{FfnFusion, Model, ModelError, Pass, cols};
use crate::tp_net::micro;

/// Tokens each half of a prompt step needs (a shorter step has little wire time to hide; the
/// same floor as Gemma 4's).
pub(super) const TP_HALF_MIN: i64 = 1024;

/// Tokens the smallest FFN part of half 1 needs ([`plan`]): its W4A4 matmuls are the GEMM's
/// (more than 64 tokens: the same kernel and tiles as the halves', [`crate::model::nvfp4_a4`]'s
/// `tokens_tile`), and a smaller one would add launches for little wire time.
pub(super) const TP_PART_MIN: i64 = 512;

/// The most FFN parts a step takes ([`parts`]).
pub(super) const TP_PARTS_MAX: usize = 4;

/// FFN parts by default ([`parts`]): half 0, and half 1 in three (5/8, 1/4 and 1/8 of it).
/// Over two DGX Sparks an 8K prompt step's prefill gains ~6.7% on the halves, a 32K prompt's
/// ~6.4%, and half 1 in two (`3`) ~5.8% and ~5.1%.
const TP_PARTS: usize = 4;

/// The FFN weights of a layer that runs in parts ([`ffn_fits`]).
const FFN: [&str; 3] = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];

/// Whether prompt steps may run their layers' sums in parts: on, the same bits as whole;
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

/// The most FFN parts a step's layers take ([`plan`]): `LEGION_DEBUG_QWEN35_TP_PARTS` (2 the
/// two halves, as before; up to [`TP_PARTS_MAX`]), else [`TP_PARTS`]. Read once.
fn parts() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        parts_of(
            std::env::var("LEGION_DEBUG_QWEN35_TP_PARTS")
                .ok()
                .as_deref(),
        )
    })
}

/// [`parts`] for the variable's value `v`.
fn parts_of(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .map_or(TP_PARTS, |p| p.clamp(2, TP_PARTS_MAX))
}

/// The FFN parts of a step of `n` tokens (at least twice [`TP_HALF_MIN`]) with at most `parts`
/// of them, `(first token, tokens)` in token order: half 0 (`n / 2` tokens) whole, then half 1
/// in `k` parts, `k` up to `parts - 1`: after its first (the rest of it) `n1 >> 2`, ...,
/// `n1 >> k` tokens (`n1` half 1's); `k` drops while the last would be under [`TP_PART_MIN`].
/// Two parts are the halves. The same answer on every rank.
pub(super) fn plan(n: i64, parts: usize) -> Vec<(i64, i64)> {
    let (n0, n1) = (n / 2, n - n / 2);
    let mut k = parts.saturating_sub(1).max(1) as u32;
    while k > 1 && (n1 >> k) < TP_PART_MIN {
        k -= 1;
    }
    // Half 1's parts after its first, smallest last.
    let rest: Vec<i64> = (2..=k).map(|s| n1 >> s).collect();
    let first = n1 - rest.iter().sum::<i64>();
    let mut out = vec![(0, n0), (n0, first)];
    for c in rest {
        let (a, l) = out[out.len() - 1];
        out.push((a + l, c));
    }
    out
}

/// The communicator this pass's layers sum their parts over: a prompt step of at least twice
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

/// Whether layer `il` of a step of `n` tokens runs in parts: every FFN matmul on the W4A4
/// dense kernels at every part's batch ([`plan`]) and the whole step's.
pub(super) fn ffn_fits(m: &Model, il: usize, n: i64) -> bool {
    let parts = plan(n, parts());
    let mut batches = parts.iter().map(|&(_, c)| c).chain([n]);
    batches.all(|c| FFN.iter().all(|name| m.a4_dense(il, name, c)))
}

/// The halves of the layer input `x` (`[n_embd, n]`), put into the graph at the layer's start:
/// placed later, a view could sit between two ops ggml-cuda fuses in the plain layer.
pub(super) fn views<'a>(pass: &Pass<'a>, x: Tensor<'a>) -> [Tensor<'a>; 2] {
    let (ctx, n) = (pass.ctx, pass.n);
    let xs = [cols(ctx, x, 0, n / 2), cols(ctx, x, n / 2, n - n / 2)];
    xs.iter().for_each(|&v| pass.graph.borrow_mut().expand(v));
    xs
}

/// One step of [`tail`]'s order: what the graph runs next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    /// Half `h`'s mixer sum finished (with its residual): its FFN parts may run.
    MixerFinish(usize),
    /// FFN part `j`, its sum started right behind its down matmul (whose send-buffer rewrite
    /// needs nothing but views between the two).
    Ffn(usize),
    /// FFN part `j`'s sum finished into the layer's output.
    FfnFinish(usize),
}

/// The order of [`tail`]'s work after the two mixer sums' starts for FFN parts of half `half`
/// each (0 or 1, in token order, part 0 half 0 whole): the starts and finishes in one order (the
/// finishes on a GPU run in the order of the starts) with at most two exchanges in flight at
/// every start, each FFN part after its half's mixer sum. Two parts give the halves' order
/// (#1133 r1).
pub(super) fn order(half: &[usize]) -> Vec<Op> {
    use std::collections::VecDeque;
    /// Finishes the oldest exchange in flight (`Op::FfnFinish` an FFN part's, else a mixer
    /// half's `Op::MixerFinish`).
    fn finish(flight: &mut VecDeque<Op>, mixed: &mut [bool; 2], out: &mut Vec<Op>) {
        if let Some(op) = flight.pop_front() {
            if let Op::MixerFinish(h) = op {
                mixed[h] = true;
            }
            out.push(op);
        }
    }
    let mut flight: VecDeque<Op> = [Op::MixerFinish(0), Op::MixerFinish(1)].into();
    let mut out = Vec::new();
    let mut mixed = [false; 2];
    for (j, &h) in half.iter().enumerate() {
        while !mixed[h] {
            finish(&mut flight, &mut mixed, &mut out);
        }
        // Room for the part's start, made before the part (its start follows its matmul).
        while flight.len() > 1 {
            finish(&mut flight, &mut mixed, &mut out);
        }
        out.push(Op::Ffn(j));
        flight.push_back(Op::FfnFinish(j));
    }
    while !flight.is_empty() {
        finish(&mut flight, &mut mixed, &mut out);
    }
    out
}

/// The rest of layer `il` from its mixer's partial output `attn` (this rank's, `[n_embd, n]`)
/// and the halves `xs` of its input ([`views`]): per half the mixer's sum and the residual add,
/// per FFN part ([`plan`]) the post-attention norm, the FFN, its sum and the residual add, in
/// [`order`]. Returns the layer's output, every part's FFN sum written into one tensor (#1133
/// r2: no concatenation).
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
    let mut mixer = [
        Some(start(cols(ctx, attn, 0, n0))),
        Some(start(cols(ctx, attn, n0, n1))),
    ];
    let norm_w = m.lw(il, "post_attention_norm.weight")?;
    let parts = plan(n, parts());
    let half: Vec<usize> = parts.iter().map(|&(a, _)| (a >= n0) as usize).collect();
    // Per half its mixer sum finished with its residual added (#1133 r2: in the sum's kernel).
    let mut x_attn: [Option<Tensor<'a>>; 2] = [None, None];
    // Per FFN part its sum's start and residual, until finished.
    let mut ffn: Vec<Option<(micro::Started<'a>, Tensor<'a>)>> = vec![None; parts.len()];
    // The layer's output: the last part finished so far (the first one's node holds the
    // memory, every later one writes into it).
    let mut y: Option<Tensor<'a>> = None;
    for op in order(&half) {
        match op {
            Op::MixerFinish(h) => {
                let s = mixer[h].take().expect("a mixer half's start");
                let width = if h == 0 { n0 } else { n1 };
                let t = micro::finish_add(ctx, s, xs[h], width, 0, None);
                add(t);
                x_attn[h] = Some(t);
            }
            Op::Ffn(j) => {
                let (at, c) = parts[j];
                let h = half[j];
                let xa = x_attn[h].expect("the part's mixer sum");
                let off = if h == 0 { at } else { at - n0 };
                let width = if h == 0 { n0 } else { n1 };
                let x = if c == width {
                    xa
                } else {
                    cols(ctx, xa, off, c)
                };
                // The post-attention norm rides in the W4A4 SwiGLU's quantization, as on one
                // GPU (ggml's bits).
                let hn = m.norm_mul(pass, x, norm_w);
                let fusion = FfnFusion {
                    res: None,
                    norm: Some((x, norm_w, m.hp.rms_eps)),
                };
                ffn[j] = Some((start(m.ffn_fused(ctx, il, hn, fusion)?), x));
            }
            // The FFN sums with their residuals, every part into one `[n_embd, n]` output;
            // with `next`, the next layer's input norm in the same kernels (its sums' bits,
            // #1133).
            Op::FfnFinish(j) => {
                let (s, res) = ffn[j].take().expect("an FFN part's start");
                let at = parts[j].0;
                let t = match next {
                    Some(nm) => micro::finish_add_norm(ctx, s, res, n, at, y, nm),
                    None => micro::finish_add(ctx, s, res, n, at, y),
                };
                add(t);
                y = Some(t);
            }
        }
    }
    let y = y.expect("the layer's output");
    if let Some(nm) = next {
        let nd = crate::tp_net::normed(ctx, y, xs[0].ne()[0], n, nm.bf16);
        return Ok((nd.y, Some(nd)));
    }
    Ok((y, None))
}
