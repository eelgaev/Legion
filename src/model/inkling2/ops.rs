//! Inkling-2's own ops ([`crate::ggml::ExternalOp`]): the steps of the `inkling2_loop` graph
//! that GLM-5.3-Flash's ops ([`crate::model::glm5next::ops`]: linear, RMS norm, the router,
//! the weighted expert sum, ...) and DeepSeek-V4.1's cache writes do not cover, and an RMS norm
//! with `glm_rms_norm`'s bits ([`rms`]). ggml only holds the tensors, views and reshapes;
//! nothing here is a ggml compute op.
//!
//! The CPU implementations are plain references (F32 math, sums in order, norms' sums in f64),
//! for tests and the CPU backend; the CUDA kernels are `src/cuda/inkling2.cu` ([`cuda`]),
//! launched from Rust. The math follows llama.cpp's `src/models/inkling2.cpp` and
//! `inkling2-loop.cpp` (MIT, ggml-org/llama.cpp PRs #32694 and #32704).

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};
use crate::model::glm5next::ops::{self as glm_ops, cpu_rms_norm};

#[cfg(feature = "cuda")]
mod cuda;

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

external_op!(MIX, "ink_mix", cpu_mix, mix);
external_op!(RMS, "ink_rms", cpu_rms_norm, rms);
external_op!(CTRL_IN, "ink_ctrl_in", cpu_ctrl_in, ctrl_in);
external_op!(CTRL_OUT, "ink_ctrl_out", cpu_ctrl_out, ctrl_out);
external_op!(CTRL_GATE, "ink_ctrl_gate", cpu_ctrl_gate, ctrl_gate);
external_op!(CONV, "ink_conv", cpu_conv, conv);
external_op!(CLIP, "ink_clip", cpu_clip, clip);
external_op!(ATTN, "ink_attn", cpu_attn, attn);
external_op!(SIGMUL, "ink_sigmul", cpu_sigmul, sigmul);
external_op!(SITU, "ink_situ", cpu_situ, situ);
external_op!(MM_ID, "ink_mm_id", cpu_mm_id, mm_id);
external_op!(SOFTCAP, "ink_softcap", cpu_softcap, softcap);

/// The most states [`mix`] takes (the depth blocks and the current residual): with its query,
/// the 9 sources ggml takes.
pub const MAX_STATES: usize = 8;
/// The most conv taps [`conv`] takes.
pub const MAX_TAPS: usize = 8;

/// The depth attention ("attention residuals", llama.cpp's `mix`): every state (F32 `[e, n]`,
/// contiguous; the earlier blocks' inputs, then the current residual) RMS-normed without a
/// weight (`eps`), scored per token against each query column of `query` (F32 `[e, nq]`) as
/// `normed · q / sqrt(e)`, and the normed states summed with the softmax of their scores. F32
/// `[e, n, nq]`: query `j`'s mix of every token is the contiguous slice `j`.
pub fn mix<'a>(ctx: &'a Context, states: &[Tensor<'a>], query: Tensor<'a>, eps: f32) -> Tensor<'a> {
    assert!(
        !states.is_empty() && states.len() <= MAX_STATES,
        "ink_mix: {} states",
        states.len()
    );
    let [e, n, _, _] = states[0].ne();
    for s in states {
        assert_eq!(s.ne()[..2], [e, n], "ink_mix: state shapes");
        assert_eq!(s.ty(), types::F32);
        contiguous(*s);
    }
    let query = contiguous(query);
    assert_eq!(query.ty(), types::F32, "ink_mix: F32 queries");
    assert_eq!(query.ne()[0], e, "ink_mix: query width");
    let nq = query.ne()[1];
    let mut args = states.to_vec();
    args.push(query);
    ctx.external_with(
        &MIX,
        types::F32,
        [e, n, nq, 1],
        &args,
        &[states.len() as i32, eps.to_bits() as i32],
    )
}

/// Whether `t` is a leaf with data of its own (no view): in Inkling-2's graphs, an RMS norm's
/// weight, never a cache or a state, so [`rms`]'s kernel reads it before waiting for the kernel
/// ahead (#1221 r6). A computed or viewed weight waits.
fn weight(t: Tensor<'_>) -> bool {
    // SAFETY: a tensor of a live context.
    unsafe {
        let r = &*t.raw();
        r.op == ffi::ggml_op_GGML_OP_NONE && r.view_src.is_null()
    }
}

/// The widest rows [`rms`] takes as its own op; wider ones are `glm_rms_norm`.
pub const RMS_MAX: i64 = 4096;

/// RMS norm over dim 0 of `x` (F32, contiguous), times the weight `w` (F32 `[ne0]`) if given:
/// `glm_rms_norm`'s ([`glm_ops::rms_norm`]) math and, on CUDA, its sums in its order (the same
/// bits), with a thread's elements loaded at once and a weight read before the kernel ahead
/// completes (#1221 r6). Rows wider than [`RMS_MAX`] are `glm_rms_norm` nodes.
pub fn rms<'a>(ctx: &'a Context, x: Tensor<'a>, w: Option<Tensor<'a>>, eps: f32) -> Tensor<'a> {
    if x.ne()[0] > RMS_MAX {
        return glm_ops::rms_norm(ctx, x, w, eps);
    }
    let x = contiguous(x);
    let early = w.is_some_and(weight);
    let mut args = vec![x];
    args.extend(w.map(contiguous));
    ctx.external_with(
        &RMS,
        types::F32,
        x.ne(),
        &args,
        &[eps.to_bits() as i32, i32::from(early)],
    )
}

/// A loop controller's input: `[rms(cur) | rms(msg)]` (no weight, `eps`), F32 `[2e, n]` from
/// F32 `[e, n]` each (contiguous).
pub fn ctrl_in<'a>(ctx: &'a Context, cur: Tensor<'a>, msg: Tensor<'a>, eps: f32) -> Tensor<'a> {
    let (cur, msg) = (contiguous(cur), contiguous(msg));
    let [e, n, _, _] = cur.ne();
    assert_eq!(msg.ne()[..2], [e, n], "ink_ctrl_in: shapes");
    ctx.external_with(
        &CTRL_IN,
        types::F32,
        [2 * e, n, 1, 1],
        &[cur, msg],
        &[eps.to_bits() as i32],
    )
}

/// A loop controller's output: `cur - tanh(forget) · cur + tanh(gain) · msg`, all F32 `[e, n]`
/// (contiguous; `gain` and `forget` the controller's projections before the tanh).
pub fn ctrl_out<'a>(
    ctx: &'a Context,
    cur: Tensor<'a>,
    msg: Tensor<'a>,
    gain: Tensor<'a>,
    forget: Tensor<'a>,
) -> Tensor<'a> {
    let args = [cur, msg, gain, forget].map(contiguous);
    for a in &args[1..] {
        assert_eq!(a.ne(), cur.ne(), "ink_ctrl_out: shapes");
    }
    ctx.external(&CTRL_OUT, types::F32, cur.ne(), &args)
}

/// Most tokens [`ctrl_gate`] takes: small steps.
pub const CTRL_TOKENS: i64 = 8;
/// The widest rank (gain and forget rows) [`ctrl_gate`] takes.
pub const CTRL_RANK_MAX: i64 = 512;

/// Whether [`ctrl_gate`] takes the F32 weights `gain` and `forget` (`[rank, e]`, no scales) over
/// `n` tokens: up to [`CTRL_TOKENS`], the rank a multiple of 32 up to [`CTRL_RANK_MAX`], rows
/// contiguous and 16-byte strided. Prompt steps keep `glm_linear` and [`ctrl_out`]
/// (`glm_linear`'s prompt kernels sum in another order).
pub fn ctrl_gate_takes(gain: Tensor<'_>, forget: Tensor<'_>, n: i64) -> bool {
    let [rank, e, _, _] = gain.ne();
    let rows = |w: Tensor<'_>| {
        w.ty() == types::F32 && w.ne() == [rank, e, 1, 1] && w.nb()[0] == 4 && w.nb()[1] % 16 == 0
    };
    (1..=CTRL_TOKENS).contains(&n)
        && e > 0
        && rank % 32 == 0
        && (32..=CTRL_RANK_MAX).contains(&rank)
        && rows(gain)
        && rows(forget)
}

/// A controller's output from its down projection `d` (F32 `[rank, n]`): [`ctrl_out`] of
/// `gain @ d` and `forget @ d` in one node, the bits of two `glm_linear`s (scale 1) then
/// [`ctrl_out`] at the small steps [`ctrl_gate_takes`] takes (#1221 r5).
pub fn ctrl_gate<'a>(
    ctx: &'a Context,
    cur: Tensor<'a>,
    msg: Tensor<'a>,
    d: Tensor<'a>,
    gain: Tensor<'a>,
    forget: Tensor<'a>,
) -> Tensor<'a> {
    let (cur, msg, d) = (contiguous(cur), contiguous(msg), contiguous(d));
    let [e, n, _, _] = cur.ne();
    assert_eq!(msg.ne()[..2], [e, n], "ink_ctrl_gate: shapes");
    assert_eq!(d.ne()[1], n, "ink_ctrl_gate: tokens");
    for w in [gain, forget] {
        assert_eq!(w.ne()[..2], [d.ne()[0], e], "ink_ctrl_gate: weights");
    }
    ctx.external_with(
        &CTRL_GATE,
        types::F32,
        cur.ne(),
        &[cur, msg, d, gain, forget],
        &[
            1f32.to_bits() as i32,
            1f32.to_bits() as i32,
            0f32.to_bits() as i32,
        ],
    )
}

/// The causal short conv with its residual of a group of `n_seqs · n_tok` tokens (`x` F32
/// `[c, ng]`, contiguous, sequence-major): per sequence `X` = its `K - 1` kept inputs (oldest
/// first) then its tokens' rows, `out[c, t] = Σ_j w[j, c] · X[t + j]` (`w` F32 `[K, c]`); with
/// `dynamic` (F32 `[K · heads, ng]`, element `lag + K · head`) also `Σ_lag dynamic[lag, head(c),
/// t] · X[t + K - 1 - lag]`, `head(c) = c / (c_total / heads)`. Returns `(x + out) · scale`,
/// F32 `[c, ng]`, and stores each sequence's last `K - 1` rows of `X` back into `state`
/// (F32 `[K - 1, width, n_seqs]`, the slots' conv history) at channels `off ..`.
#[allow(clippy::too_many_arguments)]
pub fn conv<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    state: Tensor<'a>,
    dynamic: Option<Tensor<'a>>,
    n_tok: i64,
    off: i64,
    scale: f32,
) -> Tensor<'a> {
    let x = contiguous(x);
    let [c, ng, _, _] = x.ne();
    let k = w.ne()[0];
    assert!((2..=MAX_TAPS as i64).contains(&k), "ink_conv: {k} taps");
    assert_eq!(
        (w.ty(), w.ne()[1]),
        (types::F32, c),
        "ink_conv: F32 [K, c] taps"
    );
    assert_eq!(w.nb()[1], k as usize * 4, "ink_conv: taps contiguous");
    assert_eq!(state.ty(), types::F32);
    assert_eq!(state.ne()[0], k - 1, "ink_conv: K - 1 kept inputs");
    assert!(off + c <= state.ne()[1], "ink_conv: state width");
    assert_eq!(ng % n_tok, 0);
    assert_eq!(state.ne()[2], ng / n_tok, "ink_conv: a state per sequence");
    let mut args = vec![x, w, state];
    if let Some(d) = dynamic {
        let d = contiguous(d);
        let heads = d.ne()[0] / k;
        assert!(
            d.ne()[0] == k * heads && heads > 0 && c % heads == 0 && d.ne()[1] == ng,
            "ink_conv: dynamic taps {:?} for {c} channels",
            d.ne()
        );
        args.push(d);
    }
    ctx.external_with(
        &CONV,
        types::F32,
        [c, ng, 1, 1],
        &args,
        &[
            n_tok as i32,
            off as i32,
            dynamic.is_some() as i32,
            scale.to_bits() as i32,
        ],
    )
}

/// The log-length scale of the global layers at position `p`: `1 + alpha · ln(max(1, (p + 1)
/// / floor))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogScale {
    pub floor: f32,
    pub alpha: f32,
}

impl LogScale {
    pub fn at(&self, p: i32) -> f32 {
        let v = ((p as f32 + 1.0) * (1.0 / self.floor)).max(1.0);
        v.ln() * self.alpha + 1.0
    }
}

/// Each row (`[d]` of `x` F32 `[d, h, n]`, contiguous) times `scale` (F32 `[d, h]`) if given,
/// then scaled down to an L2 norm of at most `thr` (`· clamp(thr / (‖row‖ + 1e-6), 0, 1)`), then
/// times the log-length scale of the token's position `pos[t]` when `tau` is given. F32
/// `[d, h, n]`.
pub fn clip<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    thr: f32,
    tau: Option<(Tensor<'a>, LogScale)>,
) -> Tensor<'a> {
    let x = contiguous(x);
    let [d, h, n, _] = x.ne();
    let mut args = vec![x];
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()[..2].to_vec()),
            (types::F32, vec![d, h]),
            "ink_clip: F32 [d, h] scale"
        );
        args.push(contiguous(s));
    }
    let (floor, alpha) = match tau {
        Some((pos, ls)) => {
            assert_eq!(
                (pos.ty(), pos.ne()[0]),
                (types::I32, n),
                "ink_clip: positions"
            );
            args.push(contiguous(pos));
            (ls.floor, ls.alpha)
        }
        None => (0.0, 0.0),
    };
    ctx.external_with(
        &CLIP,
        types::F32,
        x.ne(),
        &args,
        &[
            thr.to_bits() as i32,
            scale.is_some() as i32,
            tau.is_some() as i32,
            floor.to_bits() as i32,
            alpha.to_bits() as i32,
        ],
    )
}

/// Inkling-2's attention of a group of `n_seqs · n_tok` tokens: query `t` (`q` F32 `[d, nh,
/// ng]`, contiguous) head `h` attends to the keys and values (F16 `kc`, `vc` `[d · nkv, cells]`,
/// kv head `h / (nh / nkv)`) of positions `lo ..= p` (`p = pos[t]`; `lo = max(0, p - window +
/// 1)` for a window, else 0), with logits `q · k · scale + bias(p - j)`: `bias(dist) = Σ_r r[r,
/// h, t] · rel[r, dist]` for `dist` below the bias table's extent (`rel` F32 `[nr, extent]`),
/// else 0; and the per-token sink logit `sinks[h, t]` (F32 `[nh, ng]`, a zero value) in the
/// denominator. Cells through the block `table` (I32 `[width, n_seqs]`), or through `ring`
/// (its table I32 `[entries, n_seqs]` and entries: position `p` in entry `(p / KV_BLOCK) %
/// entries`). F32 `[d, nh, ng]`. `after`: the writes the reads need.
#[allow(clippy::too_many_arguments)]
pub fn attn<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    kc: Tensor<'a>,
    vc: Tensor<'a>,
    table: Tensor<'a>,
    ring: Option<(Tensor<'a>, usize)>,
    pos: Tensor<'a>,
    r: Tensor<'a>,
    rel: Tensor<'a>,
    sinks: Tensor<'a>,
    n_tok: i64,
    scale: f32,
    window: usize,
    after: Tensor<'a>,
) -> Tensor<'a> {
    let q = contiguous(q);
    let [d, nh, ng, _] = q.ne();
    assert_eq!(
        (kc.ty(), vc.ty()),
        (types::F16, types::F16),
        "ink_attn: F16 caches"
    );
    let kvw = kc.ne()[0];
    assert!(
        kvw % d == 0 && nh % (kvw / d) == 0,
        "ink_attn: {nh} heads over {kvw} kv values"
    );
    assert_eq!(vc.ne()[0], kvw);
    let r = contiguous(r);
    let nr = r.ne()[0];
    assert_eq!(r.ne()[1..3], [nh, ng], "ink_attn: r rows");
    assert_eq!(
        (rel.ty(), rel.ne()[0]),
        (types::F32, nr),
        "ink_attn: F32 [nr, extent] bias"
    );
    assert!(rel.is_contiguous());
    let sinks = contiguous(sinks);
    assert_eq!(sinks.ne()[..2], [nh, ng], "ink_attn: sinks");
    assert_eq!(pos.ne()[0], ng);
    assert_eq!(ng % n_tok, 0);
    let (wtable, entries) = ring.unwrap_or((table, 0));
    if entries > 0 {
        assert_eq!(wtable.ne()[0], entries as i64, "ink_attn: ring table");
        let kb = crate::model::core::KV_BLOCK;
        assert!(
            window > 0 && entries > (window + n_tok as usize).div_ceil(kb),
            "ink_attn: a ring shorter than the window and the step"
        );
    }
    ctx.external_with(
        &ATTN,
        types::F32,
        [d, nh, ng, 1],
        // The cells come through `wtable` alone (the ring's table, else the block table):
        // ggml takes at most 9 sources.
        &[q, kc, vc, wtable, pos, r, rel, sinks, after],
        &[
            n_tok as i32,
            scale.to_bits() as i32,
            window as i32,
            crate::model::core::KV_BLOCK as i32,
            entries as i32,
        ],
    )
}

/// `x · sigmoid(g)`, elementwise (F32, contiguous, one shape).
pub fn sigmul<'a>(ctx: &'a Context, x: Tensor<'a>, g: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.ne(), g.ne(), "ink_sigmul: shapes");
    ctx.external(&SIGMUL, types::F32, x.ne(), &[contiguous(x), contiguous(g)])
}

/// The experts' SITU activation of the gate | up rows `y` (F32 `[2f, k, n]`, contiguous; gate
/// the first `f` values of a row): `beta · tanh(g / beta) · sigmoid(g) · lb · tanh(u / lb)`
/// (`lb` 0: `u` as it is). F32 `[f, k, n]`.
pub fn situ<'a>(ctx: &'a Context, y: Tensor<'a>, beta: f32, lb: f32) -> Tensor<'a> {
    let y = contiguous(y);
    let [f2, k, n, _] = y.ne();
    assert_eq!(f2 % 2, 0, "ink_situ: gate | up rows");
    ctx.external_with(
        &SITU,
        types::F32,
        [f2 / 2, k, n, 1],
        &[y],
        &[beta.to_bits() as i32, lb.to_bits() as i32],
    )
}

/// Whether [`mm_id`] takes the expert stack `w` (`[k, m, n_expert]`) with `scale`: Q8_0, F32,
/// F16 or BF16 without one, or the E4M3 bytes of an MXFP8 stack (I8) with its F32 scale grid
/// `[k / 32, m, n_expert]` (a scale per 32 values of a row).
pub fn mm_id_takes(w: Tensor<'_>, scale: Option<Tensor<'_>>) -> bool {
    let [k, m, e, w3] = w.ne();
    // The readers take a row's values (blocks) as adjacent; rows and experts may be padded.
    if w3 != 1 || k % 32 != 0 || w.nb()[0] != w.element_size() {
        return false;
    }
    match (w.ty(), scale) {
        (types::Q8_0 | types::F32 | types::F16 | types::BF16, None) => true,
        (types::I8, Some(s)) => {
            s.ty() == types::F32 && s.is_contiguous() && s.ne()[..3] == [k / 32, m, e]
        }
        _ => false,
    }
}

/// `mul_mat_id(w, x, ids)` for the stacks [`mm_id_takes`] takes: `x` F32 `[k, 1 or n_used,
/// n_tok]` (contiguous), `ids` I32 `[n_used, n_tok]`; F32 `[m, n_used, n_tok]`, sums in F32 in
/// K order. A row whose id is out of range is 0.
pub fn mm_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    assert!(
        mm_id_takes(w, scale),
        "ink_mm_id: an expert stack of type {} {:?}",
        crate::ggml::type_name(w.ty()),
        w.ne()
    );
    let x = contiguous(x);
    let ids = contiguous(ids);
    let [k, m, _, _] = w.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, nx, xt, _] = x.ne();
    assert!(
        xk == k && xt == n_tok && (nx == 1 || nx == n_used) && x.ty() == types::F32,
        "ink_mm_id: x {:?} for ids {:?}",
        x.ne(),
        ids.ne()
    );
    let mut args = vec![w, x, ids];
    args.extend(scale);
    ctx.external_with(
        &MM_ID,
        types::F32,
        [m, n_used, n_tok, 1],
        &args,
        &[scale.is_some() as i32],
    )
}

/// The head's logits `cap · tanh(x / cap)` (`cap` 0: as they are) of the first `n_out` rows
/// of `x` (F32 `[v, n]`, contiguous): F32 `[n_out, n]`.
pub fn softcap<'a>(ctx: &'a Context, x: Tensor<'a>, cap: f32, n_out: i64) -> Tensor<'a> {
    let x = contiguous(x);
    assert!(n_out <= x.ne()[0], "ink_softcap: rows");
    ctx.external_with(
        &SOFTCAP,
        types::F32,
        [n_out, x.ne()[1], 1, 1],
        &[x],
        &[cap.to_bits() as i32],
    )
}

/// `t` itself when contiguous; ops here never take a ggml copy, so a strided input is a bug.
fn contiguous(t: Tensor<'_>) -> Tensor<'_> {
    assert!(t.is_contiguous(), "inkling2 op input must be contiguous");
    t
}

// ---- CPU references ----

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

/// `n` F32 values at byte offset `off` of `dst`'s data, to write.
#[allow(clippy::mut_from_ref)]
unsafe fn f32s_mut(t: &ffi::ggml_tensor, off: usize, n: usize) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut((t.data as *mut u8).add(off) as *mut f32, n) }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `1 / sqrt(mean(x²) + eps)`, the sum in f64.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    1.0 / ((sum / x.len() as f64) as f32 + eps).sqrt()
}

fn f16_to_f32(h: u16) -> f32 {
    // SAFETY: plain conversion.
    unsafe { ffi::ggml_fp16_to_fp32(h) }
}

/// `n` F16 values at byte offset `off` of `t`'s data, as F32.
unsafe fn f16_row(t: &ffi::ggml_tensor, off: usize, n: usize) -> Vec<f32> {
    unsafe {
        let p = (t.data as *const u8).add(off) as *const u16;
        (0..n).map(|i| f16_to_f32(*p.add(i))).collect()
    }
}

/// The cache cell of position `p` of sequence `s` through the block table.
unsafe fn cell(table: &ffi::ggml_tensor, s: usize, p: usize, block: usize) -> usize {
    // Sequence `s`'s row at its byte stride, as the CUDA kernel reads it.
    unsafe {
        let row = (table.data as *const u8).add(s * table.nb[1]) as *const i32;
        *row.add(p / block) as usize * block + p % block
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

unsafe fn cpu_mix(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let p = external_params(dst);
        let (ns, eps) = (p[0] as usize, f32_param(p[1]));
        let query = &*dst.src[ns];
        let (e, n, nq) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let inv = 1.0 / (e as f32).sqrt();
        for t in part(n, ith, nth) {
            let normed: Vec<Vec<f32>> = (0..ns)
                .map(|s| {
                    let x = f32s(&*dst.src[s], t * e * 4, e);
                    let r = rms_scale(x, eps);
                    x.iter().map(|v| v * r).collect()
                })
                .collect();
            for j in 0..nq {
                let qv = f32s(query, j * query.nb[1], e);
                let score: Vec<f32> = normed.iter().map(|x| dot(x, qv) * inv).collect();
                let m = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = score.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = ex.iter().sum();
                let o = f32s_mut(dst, (j * n + t) * e * 4, e);
                o.fill(0.0);
                for (x, w) in normed.iter().zip(&ex) {
                    let w = w / sum;
                    for i in 0..e {
                        o[i] += w * x[i];
                    }
                }
            }
        }
    }
}

unsafe fn cpu_ctrl_in(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let eps = f32_param(external_params(dst)[0]);
        let e = dst.ne[0] as usize / 2;
        for t in part(dst.ne[1] as usize, ith, nth) {
            for (h, s) in [&*dst.src[0], &*dst.src[1]].into_iter().enumerate() {
                let x = f32s(s, t * e * 4, e);
                let r = rms_scale(x, eps);
                let o = f32s_mut(dst, (t * 2 * e + h * e) * 4, e);
                for i in 0..e {
                    o[i] = x[i] * r;
                }
            }
        }
    }
}

unsafe fn cpu_ctrl_out(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let n = (dst.ne[0] * dst.ne[1]) as usize;
        let src = |i: usize| &*dst.src[i];
        for i in part(n, ith, nth) {
            let (c, m) = (at::<f32>(src(0), i), at::<f32>(src(1), i));
            let (g, f) = (at::<f32>(src(2), i).tanh(), at::<f32>(src(3), i).tanh());
            *(dst.data as *mut f32).add(i) = c - f * c + g * m;
        }
    }
}

/// Row `r` of the F32 matrix `w` (`k` values).
unsafe fn f32_row(w: &ffi::ggml_tensor, r: usize, k: usize) -> &[f32] {
    unsafe { f32s(w, r * w.nb[1], k) }
}

/// `glm_linear`'s CPU reference for one row and input: the dot in order, times the scale.
fn linear_row(w: &[f32], x: &[f32], scale: f32) -> f32 {
    dot(w, x) * scale
}

unsafe fn cpu_ctrl_gate(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let p = external_params(dst);
        let (sg, sf) = (f32_param(p[0]), f32_param(p[1]));
        let (cur, msg, d) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (gw, fw) = (&*dst.src[3], &*dst.src[4]);
        let (e, k) = (dst.ne[0] as usize, d.ne[0] as usize);
        let n = dst.ne[1] as usize;
        for r in part(e, ith, nth) {
            let (wg, wf) = (f32_row(gw, r, k), f32_row(fw, r, k));
            for t in 0..n {
                let x = f32s(d, t * d.nb[1], k);
                let i = t * e + r;
                let (c, m) = (at::<f32>(cur, i), at::<f32>(msg, i));
                // cpu_ctrl_out of the two products as cpu_linear stores them
                let (g, f) = (linear_row(wg, x, sg).tanh(), linear_row(wf, x, sf).tanh());
                *(dst.data as *mut f32).add(i) = c - f * c + g * m;
            }
        }
    }
}

unsafe fn cpu_conv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, w, state) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let p = external_params(dst);
        let (n_tok, off, has_dyn, scale) =
            (p[0] as usize, p[1] as usize, p[2] != 0, f32_param(p[3]));
        let dynamic = has_dyn.then(|| &*dst.src[3]);
        let k = w.ne[0] as usize;
        let (c, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let n_seqs = ng / n_tok;
        let per_head = dynamic.map_or(1, |d| c / (d.ne[0] as usize / k));
        for sc in part(n_seqs * c, ith, nth) {
            let (s, ch) = (sc / c, sc % c);
            let hist =
                (state.data as *mut u8).add(s * state.nb[2] + (off + ch) * state.nb[1]) as *mut f32;
            let mut xs: Vec<f32> = (0..k - 1).map(|j| *hist.add(j)).collect();
            for i in 0..n_tok {
                xs.push(at::<f32>(x, (s * n_tok + i) * c + ch));
            }
            let taps = f32s(w, ch * w.nb[1], k);
            for i in 0..n_tok {
                let t = s * n_tok + i;
                let mut out: f32 = (0..k).map(|j| taps[j] * xs[i + j]).sum();
                if let Some(d) = dynamic {
                    let dr = f32s(d, t * d.nb[1], d.ne[0] as usize);
                    let hh = ch / per_head;
                    for lag in 0..k {
                        out += dr[lag + k * hh] * xs[i + k - 1 - lag];
                    }
                }
                *(dst.data as *mut f32).add(t * c + ch) = (xs[k - 1 + i] + out) * scale;
            }
            for j in 0..k - 1 {
                *hist.add(j) = xs[n_tok + j];
            }
        }
    }
}

unsafe fn cpu_clip(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = &*dst.src[0];
        let p = external_params(dst);
        let (thr, has_scale, has_tau) = (f32_param(p[0]), p[1] != 0, p[2] != 0);
        let ls = LogScale {
            floor: f32_param(p[3]),
            alpha: f32_param(p[4]),
        };
        let scale = has_scale.then(|| &*dst.src[1]);
        let pos = has_tau.then(|| &*dst.src[1 + has_scale as usize]);
        let (d, h) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let rows = h * dst.ne[2] as usize;
        for row in part(rows, ith, nth) {
            let (hd, t) = (row % h, row / h);
            let mut v = f32s(x, row * d * 4, d).to_vec();
            if let Some(s) = scale {
                let sr = f32s(s, hd * d * 4, d);
                for i in 0..d {
                    v[i] *= sr[i];
                }
            }
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt() + 1e-6;
            let mut f = (thr / norm).clamp(0.0, 1.0);
            if let Some(pos) = pos {
                f *= ls.at(at::<i32>(pos, t));
            }
            let o = f32s_mut(dst, row * d * 4, d);
            for i in 0..d {
                o[i] = v[i] * f;
            }
        }
    }
}

unsafe fn cpu_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let s = |i: usize| &*dst.src[i];
        let (q, kc, vc, wtable, pos, r, rel, sinks) =
            (s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7));
        let p = external_params(dst);
        let (n_tok, scale, window, block) =
            (p[0] as usize, f32_param(p[1]), p[2] as usize, p[3] as usize);
        let wrap = match p[4] as usize {
            0 => usize::MAX,
            e => e * block,
        };
        let (d, nh, ng) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let nkv = kc.ne[0] as usize / d;
        let per_kv = nh / nkv;
        let (nr, extent) = (rel.ne[0] as usize, rel.ne[1] as usize);
        for th in part(ng * nh, ith, nth) {
            let (t, h) = (th / nh, th % nh);
            let sq = t / n_tok;
            let pt = at::<i32>(pos, t) as usize;
            let lo = if window > 0 {
                pt.saturating_sub(window - 1)
            } else {
                0
            };
            let g = h / per_kv;
            let qr = f32s(q, (t * nh + h) * d * 4, d);
            let rr = f32s(r, (t * nh + h) * nr * 4, nr);
            let mut score = Vec::with_capacity(pt + 1 - lo);
            let mut vals = Vec::with_capacity(pt + 1 - lo);
            for j in lo..=pt {
                let c = cell(wtable, sq, j % wrap, block);
                let k = f16_row(kc, c * kc.nb[1] + g * d * 2, d);
                let dist = pt - j;
                let bias = if dist < extent {
                    dot(rr, f32s(rel, dist * nr * 4, nr))
                } else {
                    0.0
                };
                score.push(dot(qr, &k) * scale + bias);
                vals.push(f16_row(vc, c * vc.nb[1] + g * d * 2, d));
            }
            let sink = at::<f32>(sinks, t * nh + h);
            let m = score.iter().copied().fold(sink, f32::max);
            let ex: Vec<f32> = score.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = ex.iter().sum::<f32>() + (sink - m).exp();
            let o = f32s_mut(dst, (t * nh + h) * d * 4, d);
            o.fill(0.0);
            for (v, e) in vals.iter().zip(&ex) {
                let w = e / sum;
                for i in 0..d {
                    o[i] += v[i] * w;
                }
            }
        }
    }
}

unsafe fn cpu_sigmul(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let n = (0..4).map(|i| dst.ne[i] as usize).product();
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        for i in part(n, ith, nth) {
            *(dst.data as *mut f32).add(i) = at::<f32>(x, i) * sigmoid(at::<f32>(g, i));
        }
    }
}

/// SITU of one gate and up value.
fn situ_one(g: f32, u: f32, beta: f32, lb: f32) -> f32 {
    let a = beta * (g / beta).tanh() * sigmoid(g);
    let u = if lb > 0.0 { lb * (u / lb).tanh() } else { u };
    a * u
}

unsafe fn cpu_situ(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let y = &*dst.src[0];
        let p = external_params(dst);
        let (beta, lb) = (f32_param(p[0]), f32_param(p[1]));
        let f = dst.ne[0] as usize;
        let rows = (dst.ne[1] * dst.ne[2]) as usize;
        for row in part(rows, ith, nth) {
            let src = f32s(y, row * 2 * f * 4, 2 * f);
            let o = f32s_mut(dst, row * f * 4, f);
            for i in 0..f {
                o[i] = situ_one(src[i], src[f + i], beta, lb);
            }
        }
    }
}

/// Row `r` of expert `e` of the stack `w` as F32 (`k` values), with an MXFP8 stack's scales.
unsafe fn expert_row(
    w: &ffi::ggml_tensor,
    scale: Option<&ffi::ggml_tensor>,
    e: usize,
    r: usize,
) -> Vec<f32> {
    let k = w.ne[0] as usize;
    unsafe {
        let p = (w.data as *const u8).add(e * w.nb[2] + r * w.nb[1]);
        match scale {
            Some(s) => {
                let m = w.ne[1] as usize;
                let bytes = std::slice::from_raw_parts(p, k);
                let sr = f32s(s, ((e * m + r) * (k / 32)) * 4, k / 32);
                bytes
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| crate::model::fp8::e4m3_to_f32(b) * sr[i / 32])
                    .collect()
            }
            None if w.type_ == types::F32 => {
                std::slice::from_raw_parts(p as *const f32, k).to_vec()
            }
            None => {
                let bytes = std::slice::from_raw_parts(p, crate::ggml::row_size(w.type_, k));
                crate::ggml::dequantize(w.type_, bytes, k)
            }
        }
    }
}

unsafe fn cpu_mm_id(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let scale = (external_params(dst)[0] != 0).then(|| &*dst.src[3]);
        let (k, n_expert) = (w.ne[0] as usize, w.ne[2] as usize);
        let (m, n_used, n_tok) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let nx = x.ne[1] as usize;
        // The pairs by expert, so a dequantized row serves all of an expert's pairs.
        let mut pairs: Vec<(usize, Option<usize>)> = (0..n_tok * n_used)
            .map(|pr| {
                let e = at::<i32>(ids, pr);
                (
                    pr,
                    (e >= 0 && (e as usize) < n_expert).then_some(e as usize),
                )
            })
            .collect();
        pairs.sort_by_key(|&(pr, e)| (e, pr));
        for r in part(m, ith, nth) {
            let mut cached: Option<(usize, Vec<f32>)> = None;
            for &(pr, e) in &pairs {
                let out = (dst.data as *mut f32).add(pr * m + r);
                let Some(e) = e else {
                    *out = 0.0;
                    continue;
                };
                if cached.as_ref().is_none_or(|(ce, _)| *ce != e) {
                    cached = Some((e, expert_row(w, scale, e, r)));
                }
                let row = &cached.as_ref().unwrap().1;
                let (t, j) = (pr / n_used, pr % n_used);
                let xr = f32s(x, (t * nx + if nx == 1 { 0 } else { j }) * k * 4, k);
                *out = dot(row, xr);
            }
        }
    }
}

unsafe fn cpu_softcap(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let x = &*dst.src[0];
        let cap = f32_param(external_params(dst)[0]);
        let (v, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        for t in part(n, ith, nth) {
            let src = f32s(x, t * x.nb[1], v);
            let o = f32s_mut(dst, t * v * 4, v);
            for i in 0..v {
                o[i] = if cap > 0.0 {
                    cap * (src[i] / cap).tanh()
                } else {
                    src[i]
                };
            }
        }
    }
}
