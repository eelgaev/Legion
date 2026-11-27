//! A dense Qwen3.5+ prompt step under tensor parallelism over machines with each layer's sums
//! in parts of its tokens (#1133), as Gemma 4's halves (#619): the mixer's output projection's
//! partial is summed per half, and each FFN part runs as soon as its half's sum is in, so that
//! the transfers run while the GPU computes: half 1's mixer sum during half 0's FFN, each FFN
//! part's sum during the next part's FFN. Gated DeltaNet runs its delta rule, gated norm and
//! output projection per half where it can ([`Model::delta_net_halves`], [`mixer_on`]), and
//! gated attention its attention, gate and output projection ([`Model::attention_halves`],
//! [`attn_on`]), so half 0's mixer sum travels during half 1's mixer too.
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
//! every part's and the whole step's batch, [`Model::a4_dense`], or all on the FP8 W8A8 GEMM
//! with per-token activation scales, [`Model::fp8_dense`]: every token the same bits at any of
//! them), and every part's sum gives each value the plain all-reduce's bits
//! ([`micro::all_reduce_start`]). An NVFP4 checkpoint's FP8 MLPs (Qwen3.8-27B's last 8 layers)
//! take the FP8 kind (#1133 r3); a layer whose FFN fits neither runs whole.

#[cfg(test)]
mod tests;

use crate::ggml::Tensor;
use crate::model::core::{FfnFusion, Mixed, Model, ModelError, Pass, cols};
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

/// Whether the gated DeltaNet layers of those steps run their mixer per half of the tokens
/// ([`Model::delta_net_halves`]), so that half 0's sum is on the wire while half 1's delta
/// rule, norm and output projection run: on, the same bits as whole;
/// `LEGION_DEBUG_QWEN35_TP_MIXER_HALVES=0` keeps the mixer whole (both halves' sums start
/// after it). Read once ([`mixer`]).
pub(super) fn mixer_on() -> bool {
    mixer() >= 1
}

/// Whether the gated attention layers of those steps run per half of the tokens from the
/// attention on ([`Model::attention_halves`]: the attention in two launches, then per half the
/// gate and the output projection), so that half 0's sum is on the wire while half 1's
/// attention and projection run: on, the same bits as whole;
/// `LEGION_DEBUG_QWEN35_TP_ATTN_HALVES=0` keeps the layer whole. Read once.
pub(super) fn attn_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = attn_of(
            std::env::var("LEGION_DEBUG_QWEN35_TP_ATTN_HALVES")
                .ok()
                .as_deref(),
        );
        if !on {
            tracing::info!(
                "attention parts (#1133): off, the attention layers run whole \
                 (LEGION_DEBUG_QWEN35_TP_ATTN_HALVES=0)"
            );
        }
        on
    })
}

/// [`attn_on`] for the variable's value `v`.
fn attn_of(v: Option<&str>) -> bool {
    v.map(str::trim) != Some("0")
}

/// Whether those halves run the mixer's input projections and conv per half too
/// ([`Model::delta_net_halves`]'s `inputs`, #1133), so half 0's delta rule and sum start
/// before half 1's inputs are made: on, the same bits;
/// `LEGION_DEBUG_QWEN35_TP_MIXER_HALVES=1` keeps them whole (the halves from the delta rule
/// on only). Read once ([`mixer`]).
pub(super) fn mixer_inputs_on() -> bool {
    mixer() >= 2
}

/// How much of the DeltaNet mixer runs per half: `LEGION_DEBUG_QWEN35_TP_MIXER_HALVES` 0
/// (none), 1 (the delta rule, norm and output projection) or 2 (also the input projections
/// and conv; the default, and any other value). Read once.
fn mixer() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        mixer_of(
            std::env::var("LEGION_DEBUG_QWEN35_TP_MIXER_HALVES")
                .ok()
                .as_deref(),
        )
    })
}

/// [`mixer`] for the variable's value `v`.
fn mixer_of(v: Option<&str>) -> u8 {
    match v.map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    }
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

/// The most FFN parts of the layers whose FFN runs on the FP8 W8A8 GEMM ([`Model::fp8_dense`],
/// #1133 r3): `LEGION_DEBUG_QWEN35_TP_FP8_PARTS=0` keeps them whole (as before), `2` up to
/// [`TP_PARTS_MAX`] that many, else as many as the other layers ([`parts`]). `None`: whole.
/// Read once.
fn fp8_parts() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        fp8_parts_of(
            std::env::var("LEGION_DEBUG_QWEN35_TP_FP8_PARTS")
                .ok()
                .as_deref(),
            parts(),
        )
    })
}

/// [`fp8_parts`] for the variable's value `v` and the other layers' `parts`.
fn fp8_parts_of(v: Option<&str>, parts: usize) -> Option<usize> {
    match v.map(str::trim) {
        Some("0") => None,
        Some(s) => Some(
            s.parse::<usize>()
                .map_or(parts, |p| p.clamp(2, TP_PARTS_MAX)),
        ),
        None => Some(parts),
    }
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

/// The most FFN parts layer `il` of a step of `n` tokens takes ([`plan`]), or `None` where it
/// runs whole: every FFN matmul on the W4A4 dense kernels ([`Model::a4_dense`], [`parts`]), or
/// every one on the FP8 W8A8 GEMM with per-token activation scales ([`Model::fp8_dense`],
/// [`fp8_parts`]), at every part's batch and the whole step's.
pub(super) fn ffn_parts(m: &Model, il: usize, n: i64) -> Option<usize> {
    let fits = |p: usize, dense: &dyn Fn(&str, i64) -> bool| {
        let parts = plan(n, p);
        let mut batches = parts.iter().map(|&(_, c)| c).chain([n]);
        batches.all(|c| FFN.iter().all(|name| dense(name, c)))
    };
    let p = parts();
    if fits(p, &|name, c| m.a4_dense(il, name, c)) {
        return Some(p);
    }
    let p = fp8_parts()?;
    fits(p, &|name, c| m.fp8_dense(il, name, c)).then_some(p)
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

/// The rest of layer `il` from its mixer's partial output `attn` (this rank's, `[n_embd, n]`
/// whole or per half, [`Mixed`]) and the halves `xs` of its input ([`views`]): per half the
/// mixer's sum (half 0's started before half 1's mixer runs where the mixer has halves) and
/// the residual add,
/// per FFN part ([`plan`]) the post-attention norm, the FFN, its sum and the residual add, in
/// [`order`], summed over `comm` in at most `parts` FFN parts ([`ffn_parts`]). Returns the
/// layer's output, every part's FFN sum written into one tensor (#1133 r2: no concatenation).
pub(super) fn tail<'a>(
    m: &'a Model,
    pass: &Pass<'a>,
    il: usize,
    xs: [Tensor<'a>; 2],
    attn: Mixed<'a>,
    (comm, parts): (i32, usize),
    next: Option<crate::tp_net::Norm<'a>>,
) -> Result<(Tensor<'a>, Option<crate::tp_net::Normed<'a>>), ModelError> {
    let (ctx, n) = (pass.ctx, pass.n);
    let (n0, n1) = (n / 2, n - n / 2);
    let whole = n as usize;
    let add = |t: Tensor<'a>| pass.graph.borrow_mut().expand(t);
    let start = |t: Tensor<'a>| {
        let s = micro::all_reduce_start(ctx, comm, t, whole);
        add(s.node);
        s
    };
    let (halves, after) = match attn {
        Mixed::Whole(attn) => {
            // Its columns as views ([`cols`]): a copy where the projection's output is
            // strided.
            let attn = if attn.is_contiguous() {
                attn
            } else {
                ctx.cont_2d(attn, attn.ne()[0], n)
            };
            ([cols(ctx, attn, 0, n0), cols(ctx, attn, n0, n1)], [None; 2])
        }
        Mixed::Halves { out, after } => (out, after),
    };
    // Half 0's start goes into the graph first, behind its own mixer half only (each start
    // right behind its projection, [`micro::send_gemm`]); half 1's mixer follows it.
    let mut mixer = [Some(start(halves[0])), Some(start(halves[1]))];
    // The delta rule's final states (and the conv history) into the pool, behind half 1's
    // projection's start.
    for t in after.into_iter().flatten() {
        add(t);
    }
    let norm_w = m.lw(il, "post_attention_norm.weight")?;
    let parts = plan(n, parts);
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
                // GPU (ggml's bits); on FP8 MLPs one op writes its per-token FP8 codes, the
                // bits the whole layer's sum computes them with (`Model::norm_fold`).
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
