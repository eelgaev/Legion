//! Fused elementwise ops of the qwen35 graph, as external ops ([`crate::ggml::ExternalOp`]):
//! a CPU implementation (the reference, for tests) and a CUDA kernel (`src/cuda/fused.cu`)
//! each. They replace chains of ggml nodes that each read and write a whole activation, so a
//! prompt step moves less memory and launches fewer kernels.
//!
//! - [`conv_silu`]: the gated DeltaNet's causal depthwise conv over `[history | this step]`
//!   and its SiLU, reading the step's rows where they are (ggml's `ssm_conv` needs them
//!   concatenated behind the history and transposed: a copy of the whole input), and the next
//!   step's history.
//! - [`router`]: a MoE layer's routing from the router logits, as `soft_max`, `argsort_top_k`,
//!   `get_rows`, `sum_rows`, `clamp` and `div` compute it (a full sort of every token's logits
//!   among them): the `k` best experts in descending order of logit (ties: the lower id first)
//!   and their softmax probabilities normalized over the `k`.
//! - [`shexp_gate`]: a MoE layer's output plus its shared expert's scaled by the sigmoid gate,
//!   one launch for ggml's sigmoid, broadcast mul and add.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static CONV_SILU: ExternalOp = ExternalOp {
    name: "conv_silu",
    cpu: cpu_conv_silu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_conv_silu),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static DELTA_PREP: ExternalOp = ExternalOp {
    name: "delta_prep",
    cpu: cpu_delta_prep,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_delta_prep),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static GATED_NORM: ExternalOp = ExternalOp {
    name: "gated_norm",
    cpu: cpu_gated_norm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gated_norm),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static RMS_NORM_BF16: ExternalOp = ExternalOp {
    name: "rms_norm_bf16",
    cpu: cpu_rms_norm_bf16,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_rms_norm_bf16),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static GATE_BF16: ExternalOp = ExternalOp {
    name: "gate_bf16",
    cpu: cpu_gate_bf16,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gate_bf16),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static MOE_COMBINE: ExternalOp = ExternalOp {
    name: "moe_combine",
    cpu: cpu_moe_combine,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_moe_combine),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SHEXP_GATE: ExternalOp = ExternalOp {
    name: "shexp_gate",
    cpu: cpu_shexp_gate,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_shexp_gate),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ROUTER_TOPK: ExternalOp = ExternalOp {
    name: "router_topk",
    cpu: cpu_router_topk,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_router_topk),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ROUTER_WEIGHTS: ExternalOp = ExternalOp {
    name: "router_weights",
    cpu: cpu_router_weights,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_router_weights),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most experts [`router`] takes (the CUDA kernel holds a token's logits in one warp).
pub(super) const ROUTER_MAX_EXPERTS: i64 = 512;

/// The routing of `logits` (F32 `[n_expert, n]`): the `k` best experts per token (I32 `[k, n]`,
/// descending logit, ties to the lower id) and their weights (F32 `[k, n]`): softmax
/// probabilities normalized over the `k`, `e^(l_j - l_max) / Σ_k e^(l_i - l_max)` (the softmax
/// denominator cancels).
pub(super) fn router<'a>(ctx: &'a Context, logits: Tensor<'a>, k: i64) -> (Tensor<'a>, Tensor<'a>) {
    router_planned(ctx, logits, k, false, false)
}

/// Whether graphs built from now on let [`router_planned`] callers join the two CUDA launches
/// ([`set_router_join`]).
static ROUTER_JOIN_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Sets whether graphs built from now on run the router's top-k and weights as one CUDA launch
/// where the caller allows it (#814): on by default (the same bits, one launch fewer per MoE
/// layer); `legion serve --no-router-join` keeps two (for comparisons).
pub fn set_router_join(on: bool) {
    ROUTER_JOIN_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// [`router`], and with `plan` (the caller checked [`super::nvfp4_once::plan_fits`]; steps of
/// another shape ignore it) the ids node's CUDA kernel also writes the routing plan of the
/// step's NVFP4 expert ops (#836), so they launch without building one. With `join` (the
/// caller puts the weights into the graph before any other user of the ids) and without a
/// plan, the weights node's CUDA kernel writes the ids too and the ids node launches nothing
/// ([`set_router_join`], #814). The same ids and weights either way.
pub(super) fn router_planned<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    k: i64,
    plan: bool,
    join: bool,
) -> (Tensor<'a>, Tensor<'a>) {
    let [n_expert, n, _, _] = logits.ne();
    assert!(
        n_expert <= ROUTER_MAX_EXPERTS && k <= n_expert,
        "router: {k} of {n_expert}"
    );
    assert_eq!(logits.ty(), types::F32, "router: F32 logits");
    assert_eq!(logits.nb()[0], 4, "router: rows contiguous");
    let plan = plan && super::nvfp4_once::plan_shape(k, n);
    let join = join && !plan && ROUTER_JOIN_ON.load(std::sync::atomic::Ordering::Relaxed);
    // Params: the ids node [plan, joined], the weights node [joined].
    let ids = ctx.external_with(
        &ROUTER_TOPK,
        types::I32,
        [k, n, 1, 1],
        &[logits],
        &[i32::from(plan), i32::from(join)],
    );
    let w = ctx.external_with(
        &ROUTER_WEIGHTS,
        types::F32,
        [k, n, 1, 1],
        &[logits, ids],
        &[i32::from(join)],
    );
    (ids, w)
}

/// Whether `node` is the ids of a [`router_planned`] whose kernel writes the routing plan.
///
/// # Safety
/// `node` points to a valid tensor.
pub(super) unsafe fn writes_plan(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: a valid tensor; an external node's params are its own.
    unsafe {
        crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &ROUTER_TOPK))
            && crate::ggml::external_params(&*node)[0] == 1
    }
}

unsafe fn cpu_router_topk(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `router`, host tensors.
    unsafe {
        let l = &*dst.src[0];
        let (n_expert, n) = (l.ne[0] as usize, l.ne[1] as usize);
        let k = dst.ne[0] as usize;
        for t in (ith..n).step_by(nth) {
            let row = std::slice::from_raw_parts(
                (l.data as *const u8).add(t * l.nb[1]) as *const f32,
                n_expert,
            );
            let mut order: Vec<usize> = (0..n_expert).collect();
            // descending value, ties to the lower id (a stable sort keeps id order)
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            let out = (dst.data as *mut i32).add(t * k);
            for (j, &e) in order[..k].iter().enumerate() {
                *out.add(j) = e as i32;
            }
        }
    }
}

unsafe fn cpu_router_weights(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `router`, host tensors.
    unsafe {
        let (l, ids) = (&*dst.src[0], &*dst.src[1]);
        let n = l.ne[1] as usize;
        let k = dst.ne[0] as usize;
        for t in (ith..n).step_by(nth) {
            let row = (l.data as *const u8).add(t * l.nb[1]) as *const f32;
            let ids = (ids.data as *const i32).add(t * k);
            let m = *row.add(*ids as usize);
            let mut sum = 0f32;
            for j in 0..k {
                sum += (*row.add(*ids.add(j) as usize) - m).exp();
            }
            let out = (dst.data as *mut f32).add(t * k);
            for j in 0..k {
                *out.add(j) = (*row.add(*ids.add(j) as usize) - m).exp() / sum;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct Router {
    logits: *const f32,
    ld: i64,
    ids_in: *const i32,
    ids: *mut i32,
    w: *mut f32,
    n_expert: i32,
    k: i32,
    n_tok: i32,
    /// The routing plan the top-k kernel writes (nvfp4_plan.cuh), or null.
    plan: *mut c_void,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_router_topk_launch(p: *const Router, stream: *mut c_void) -> i32;
    fn legion_router_weights_launch(p: *const Router, stream: *mut c_void) -> i32;
    fn legion_router_select_launch(p: *const Router, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn router_args(dst: &ffi::ggml_tensor) -> Router {
    // SAFETY: a node made by `router`.
    unsafe {
        let l = &*dst.src[0];
        let topk = dst.src[1].is_null();
        Router {
            logits: l.data as *const f32,
            ld: (l.nb[1] / 4) as i64,
            ids_in: if topk {
                std::ptr::null()
            } else {
                (*dst.src[1]).data as *const i32
            },
            ids: if topk {
                dst.data as *mut i32
            } else {
                std::ptr::null_mut()
            },
            w: if topk {
                std::ptr::null_mut()
            } else {
                dst.data as *mut f32
            },
            n_expert: l.ne[0] as i32,
            k: dst.ne[0] as i32,
            n_tok: l.ne[1] as i32,
            plan: std::ptr::null_mut(),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_router_topk(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let params = crate::ggml::external_params(dst);
    if params[1] == 1 {
        // joined: the weights node's kernel writes the ids
        return Ok(());
    }
    // SAFETY: a node made by `router`.
    let mut p = unsafe { router_args(dst) };
    if params[0] == 1 {
        p.plan = super::nvfp4_once::plan_ptr()?;
    }
    // SAFETY: device tensors of the stream's device; the plan is that device's.
    match unsafe { legion_router_topk_launch(&p, stream) } {
        0 => Ok(()),
        e => Err(format!("router_topk: error {e}")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_router_weights(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `router`.
    let mut p = unsafe { router_args(dst) };
    let r = if crate::ggml::external_params(dst)[0] == 1 {
        // joined: the ids (src[1], the ids node's output) written here too
        p.ids = p.ids_in as *mut i32;
        p.ids_in = std::ptr::null();
        // SAFETY: device tensors of the stream's device; nothing reads the ids before this node.
        unsafe { legion_router_select_launch(&p, stream) }
    } else {
        // SAFETY: device tensors of the stream's device.
        unsafe { legion_router_weights_launch(&p, stream) }
    };
    match r {
        0 => Ok(()),
        e => Err(format!("router_weights: error {e}")),
    }
}

/// The causal depthwise conv of `x` (F32 `[C, nt, ns]`: `nt` tokens of `ns` sequences, rows
/// contiguous) with history `state` (F32 `[d - 1, C, ns]`, the last `d - 1` inputs before
/// `x`) and kernel `w` (F32 `[d, C]`), then SiLU:
///
/// `out[c, t, s] = silu(Σ_k in[t + k, c, s] · w[k, c])`, `in = [state | x]` along time,
///
/// as ggml's `silu(ssm_conv(concat(state, transpose(x)), w))`. Returns `out` (F32 `[C, nt,
/// ns]`) and the next history (F32 `[d - 1, C, ns]`, the last `d - 1` inputs), which the
/// caller copies into the state.
pub(super) fn conv_silu<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    state: Tensor<'a>,
    w: Tensor<'a>,
) -> (Tensor<'a>, Tensor<'a>) {
    let d = w.ne()[0];
    conv_silu_window(ctx, x, state, w, d - 1)
}

/// [`conv_silu`] with the last `keep` inputs of `[state | x]` as the second result (F32
/// `[keep, C, ns]`, `d - 1 <= keep <= d - 1 + nt`): with `d - 1 + nt`, the whole window,
/// `concat(state, transpose(x))` (what an MTP verify step stores for the replay), without a
/// separate copy of it.
pub(super) fn conv_silu_window<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    state: Tensor<'a>,
    w: Tensor<'a>,
    keep: i64,
) -> (Tensor<'a>, Tensor<'a>) {
    conv_silu_node(ctx, x, state, w, keep, None)
}

/// Threads of a `conv_silu` CUDA block: one q or k head of [`conv_silu_norm`] each.
const CONV_BLOCK: i64 = 128;

/// Groups of more tokens than this take [`conv_silu_norm`] (prompt steps; decode and verify
/// steps keep [`conv_silu`] and [`delta_prep`]).
const CONV_NORM_MIN_TOKENS: i64 = 64;

/// Whether a group of `n` tokens of a layer [`delta_prep`] takes ([`delta_prep_fits`]) runs
/// [`conv_silu_norm`] and [`delta_gates`] instead of [`conv_silu`] and [`delta_prep`]: a
/// non-verify prompt group with heads of [`CONV_BLOCK`].
pub(super) fn conv_norm_fits(heads: DeltaHeads, n: i64, verify: bool) -> bool {
    !verify && heads.0 == CONV_BLOCK && n > CONV_NORM_MIN_TOKENS
}

/// [`conv_silu`] with the q and k heads of its output (`n_k` of `head_k` each, then as many
/// k heads, at the start of every row) l2-normalized per token as [`delta_prep`] does it, bit
/// for bit: the conv output's q and k rows are then [`delta_prep`]'s q and k, without the pass
/// that reads and writes them again ([`delta_gates`] computes the rest). `eps` is
/// [`delta_prep`]'s. Heads of [`CONV_BLOCK`] only ([`conv_norm_fits`]).
pub(super) fn conv_silu_norm<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    state: Tensor<'a>,
    w: Tensor<'a>,
    heads: DeltaHeads,
    eps: f32,
) -> (Tensor<'a>, Tensor<'a>) {
    let (head_k, n_k, head_v, n_v) = heads;
    assert_eq!(head_k, CONV_BLOCK, "conv_silu_norm: head size");
    assert_eq!(
        x.ne()[0],
        2 * head_k * n_k + head_v * n_v,
        "conv_silu_norm: conv rows"
    );
    let d = w.ne()[0];
    conv_silu_node(ctx, x, state, w, d - 1, Some((2 * n_k, head_k, eps)))
}

/// The `conv_silu` node; `norm`: (heads normalized, head size, eps).
fn conv_silu_node<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    state: Tensor<'a>,
    w: Tensor<'a>,
    keep: i64,
    norm: Option<(i64, i64, f32)>,
) -> (Tensor<'a>, Tensor<'a>) {
    let [c, nt, ns, _] = x.ne();
    let [d, wc, _, _] = w.ne();
    assert_eq!(wc, c, "conv_silu: kernel channels");
    assert_eq!(state.ne()[..3], [d - 1, c, ns], "conv_silu: state shape");
    assert!(
        (d - 1..=d - 1 + nt).contains(&keep),
        "conv_silu: keep {keep} of {} inputs",
        d - 1 + nt
    );
    for t in [x, state, w] {
        assert_eq!(t.ty(), types::F32, "conv_silu: F32 tensors");
        assert_eq!(t.nb()[0], 4, "conv_silu: rows contiguous");
    }
    assert!(w.is_contiguous(), "conv_silu: kernel contiguous");
    let n_out = c * nt * ns;
    let n_hist = keep * c * ns;
    let (norm_heads, norm_head, eps) = norm.unwrap_or((0, 0, 0.0));
    let node = ctx.external_with(
        &CONV_SILU,
        types::F32,
        [n_out + n_hist, 1, 1, 1],
        &[x, state, w],
        &[
            keep as i32,
            norm_heads as i32,
            norm_head as i32,
            eps.to_bits() as i32,
        ],
    );
    let out = ctx.view_3d(
        node,
        [c, nt, ns],
        [c as usize * 4, (c * nt) as usize * 4],
        0,
    );
    let hist = ctx.view_3d(
        node,
        [keep, c, ns],
        [keep as usize * 4, (keep * c) as usize * 4],
        n_out as usize * 4,
    );
    (out, hist)
}

/// Strides in floats of dims 1 and 2.
fn strides(t: &ffi::ggml_tensor) -> (usize, usize) {
    (t.nb[1] / 4, t.nb[2] / 4)
}

unsafe fn cpu_conv_silu(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `conv_silu`, host tensors.
    unsafe {
        let (x, state, w) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (c_n, nt, ns) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let d = w.ne[0] as usize;
        let params = crate::ggml::external_params(dst);
        let keep = params[0] as usize;
        // conv_silu_norm: these first channels' outputs are written by head below.
        let (norm_heads, norm_head) = (params[1] as usize, params[2] as usize);
        let norm_c = norm_heads * norm_head;
        let (x_ld, x_sd) = strides(x);
        let (st_ld, st_sd) = strides(state);
        let (xp, sp, wp) = (
            x.data as *const f32,
            state.data as *const f32,
            w.data as *const f32,
        );
        let out = dst.data as *mut f32;
        let hist = out.add(c_n * nt * ns);
        let input = |j: usize, c: usize, s: usize| -> f32 {
            if j < d - 1 {
                *sp.add(j + c * st_ld + s * st_sd)
            } else {
                *xp.add(c + (j + 1 - d) * x_ld + s * x_sd)
            }
        };
        let conv = |t: usize, c: usize, s: usize| -> f32 {
            let mut acc = 0f32;
            for k in 0..d {
                acc += input(t + k, c, s) * *wp.add(k + c * d);
            }
            acc / (1.0 + (-acc).exp())
        };
        // Channels split over the threads: each reads and writes its own.
        for c in (ith..c_n).step_by(nth) {
            for s in 0..ns {
                if c >= norm_c {
                    for t in 0..nt {
                        *out.add(c + t * c_n + s * c_n * nt) = conv(t, c, s);
                    }
                }
                for j in 0..keep {
                    *hist.add(j + c * keep + s * keep * c_n) = input(nt + d - 1 - keep + j, c, s);
                }
            }
        }
        // The normalized heads split over the threads: `cpu_delta_prep`'s l2 norm of each.
        let eps = f32::from_bits(params[3] as u32);
        let scale_out = 1.0 / (norm_head as f32).sqrt();
        let mut row = vec![0f32; norm_head];
        for h in (ith..norm_heads).step_by(nth) {
            for s in 0..ns {
                for t in 0..nt {
                    for (i, v) in row.iter_mut().enumerate() {
                        *v = conv(t, h * norm_head + i, s);
                    }
                    let mean = row.iter().map(|v| v * v).sum::<f32>() / norm_head as f32;
                    let scale = 1.0 / (mean + eps).sqrt();
                    for (i, v) in row.iter().enumerate() {
                        *out.add(h * norm_head + i + t * c_n + s * c_n * nt) =
                            scale_out * (scale * v);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct ConvSilu {
    x: *const f32,
    x_ld: i64,
    x_sd: i64,
    state: *const f32,
    st_ld: i64,
    st_sd: i64,
    w: *const f32,
    out: *mut f32,
    hist: *mut f32,
    c: i32,
    nt: i32,
    ns: i32,
    d: i32,
    keep: i32,
    chunk: i32,
    norm_blocks: i32,
    norm_head: i32,
    norm_eps: f32,
    norm_scale: f32,
    grid: [u32; 3],
    block: u32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_conv_silu_launch(p: *const ConvSilu, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_conv_silu(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `conv_silu`, device tensors of the stream's device.
    unsafe {
        let (x, state, w) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (c, nt, ns) = (x.ne[0], x.ne[1], x.ne[2]);
        let (x_ld, x_sd) = strides(x);
        let (st_ld, st_sd) = strides(state);
        // Runs of 32 tokens a thread: enough blocks for a big step, few re-read inputs.
        const BLOCK: u32 = CONV_BLOCK as u32;
        const CHUNK: i64 = 32;
        let params = crate::ggml::external_params(dst);
        let out = dst.data as *mut f32;
        let p = ConvSilu {
            x: x.data as *const f32,
            x_ld: x_ld as i64,
            x_sd: x_sd as i64,
            state: state.data as *const f32,
            st_ld: st_ld as i64,
            st_sd: st_sd as i64,
            w: w.data as *const f32,
            out,
            hist: out.add((c * nt * ns) as usize),
            c: c as i32,
            nt: nt as i32,
            ns: ns as i32,
            d: w.ne[0] as i32,
            keep: params[0],
            chunk: CHUNK as i32,
            norm_blocks: params[1],
            norm_head: params[2],
            norm_eps: f32::from_bits(params[3] as u32),
            norm_scale: if params[2] > 0 {
                1.0 / (params[2] as f32).sqrt()
            } else {
                0.0
            },
            grid: [
                (c as u32).div_ceil(BLOCK),
                (nt as u32).div_ceil(CHUNK as u32),
                ns as u32,
            ],
            block: BLOCK,
        };
        match legion_conv_silu_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err(format!("conv_silu: kernel width {} unsupported", w.ne[0])),
            -2 => Err(format!(
                "conv_silu: normalized heads of {} unsupported",
                params[2]
            )),
            e => Err(format!("conv_silu: CUDA error {e}")),
        }
    }
}

/// The shapes of a gated DeltaNet layer's heads: `(head_k, n_k, head_v, n_v)`.
pub(super) type DeltaHeads = (i64, i64, i64, i64);

/// Whether graphs built from now on use [`delta_prep`].
static DELTA_PREP_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Sets whether graphs built from now on use [`delta_prep`] where it fits (A/B tests). On by
/// default (ggml-cuda's chain bit for bit, ~11 kernels fewer per recurrent layer);
/// `legion serve --delta-prep-ggml` turns it off.
pub fn set_delta_prep(on: bool) {
    DELTA_PREP_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`delta_prep`] takes this layer: on a GPU only (the CPU graph keeps ggml's chain,
/// the reference), heads of at most 256, F32 weights in device memory, and switched on
/// ([`set_delta_prep`]).
pub(super) fn delta_prep_fits(
    heads: DeltaHeads,
    dt_bias: Tensor<'_>,
    a: Tensor<'_>,
    n: i64,
) -> bool {
    let on = DELTA_PREP_ON.load(std::sync::atomic::Ordering::Relaxed);
    let (head_k, _, _, n_v) = heads;
    on && cfg!(feature = "cuda")
        && head_k <= 256
        && n <= 65535
        && [dt_bias, a].iter().all(|t| {
            t.ty() == types::F32
                && !t.is_host()
                && t.is_contiguous()
                && t.elements() == n_v as usize
        })
}

/// A gated DeltaNet layer's delta-rule inputs from its conv output `conv` (F32 `[qkv_dim, nt,
/// ns]`, contiguous: per token the q heads, k heads and v heads), the `alpha` and `beta`
/// projections (F32 `[n_v, nt * ns]`, rows contiguous) and the weights `dt_bias`, `a`
/// (`[n_v]`), as ggml's chain computes them:
///
/// - `q`, `k`: `l2_norm` of each head (`rms_norm(x, eps / head_k) / sqrt(head_k)`),
///   `[head_k, n_k, nt, ns]`;
/// - `g = softplus(alpha + dt_bias) * a`, `beta = sigmoid(beta)`: `[1, n_v, nt, ns]`.
///
/// With `store` (a verify step's replay pool views of q, k, g and beta, `[a, b, nt, ns]` each
/// with rows contiguous), the op writes those too, instead of four copies. (ggml takes nine
/// sources; v, a view of `conv`, is the caller's to copy.)
#[allow(clippy::too_many_arguments)]
pub(super) fn delta_prep<'a>(
    ctx: &'a Context,
    conv: Tensor<'a>,
    alpha: Tensor<'a>,
    beta: Tensor<'a>,
    dt_bias: Tensor<'a>,
    a: Tensor<'a>,
    heads: DeltaHeads,
    eps: f32,
    store: Option<[Tensor<'a>; 4]>,
) -> [Tensor<'a>; 4] {
    let (head_k, n_k, _, n_v) = heads;
    let [_, nt, ns, _] = conv.ne();
    let n = nt * ns;
    let node = delta_prep_node(ctx, conv, alpha, beta, dt_bias, a, heads, eps, store, false);
    let qk = head_k * n_k * n;
    let heads_view = |at: i64| {
        ctx.view_4d(
            node,
            [head_k, n_k, nt, ns],
            [
                head_k as usize * 4,
                (head_k * n_k) as usize * 4,
                (head_k * n_k * nt) as usize * 4,
            ],
            at as usize * 4,
        )
    };
    let [g, beta] = gate_views(ctx, node, n_v, nt, ns, 2 * qk);
    [heads_view(0), heads_view(qk), g, beta]
}

/// [`delta_prep`]'s `g` and `beta` only, for a conv output whose q and k heads
/// [`conv_silu_norm`] normalized already (they are its q and k).
#[allow(clippy::too_many_arguments)]
pub(super) fn delta_gates<'a>(
    ctx: &'a Context,
    conv: Tensor<'a>,
    alpha: Tensor<'a>,
    beta: Tensor<'a>,
    dt_bias: Tensor<'a>,
    a: Tensor<'a>,
    heads: DeltaHeads,
    eps: f32,
) -> [Tensor<'a>; 2] {
    let [_, nt, ns, _] = conv.ne();
    let node = delta_prep_node(ctx, conv, alpha, beta, dt_bias, a, heads, eps, None, true);
    gate_views(ctx, node, heads.3, nt, ns, 0)
}

/// `g` and `beta` (`[1, n_v, nt, ns]` each) of a `delta_prep` node, from float `at` on.
fn gate_views<'a>(
    ctx: &'a Context,
    node: Tensor<'a>,
    n_v: i64,
    nt: i64,
    ns: i64,
    at: i64,
) -> [Tensor<'a>; 2] {
    let gb = n_v * nt * ns;
    [at, at + gb].map(|at| {
        ctx.view_4d(
            node,
            [1, n_v, nt, ns],
            [4, n_v as usize * 4, (n_v * nt) as usize * 4],
            at as usize * 4,
        )
    })
}

/// The `delta_prep` node: q, k, g and beta, or with `gates_only` g and beta.
#[allow(clippy::too_many_arguments)]
fn delta_prep_node<'a>(
    ctx: &'a Context,
    conv: Tensor<'a>,
    alpha: Tensor<'a>,
    beta: Tensor<'a>,
    dt_bias: Tensor<'a>,
    a: Tensor<'a>,
    heads: DeltaHeads,
    eps: f32,
    store: Option<[Tensor<'a>; 4]>,
    gates_only: bool,
) -> Tensor<'a> {
    let (head_k, n_k, head_v, n_v) = heads;
    let [qkv_dim, nt, ns, _] = conv.ne();
    assert_eq!(
        qkv_dim,
        2 * head_k * n_k + head_v * n_v,
        "delta_prep: conv rows"
    );
    assert!(
        conv.is_contiguous() && conv.ty() == types::F32,
        "delta_prep: conv"
    );
    let n = nt * ns;
    for t in [alpha, beta] {
        assert_eq!(t.ne()[..2], [n_v, n], "delta_prep: alpha / beta shape");
        assert!(
            t.ty() == types::F32 && t.nb()[0] == 4,
            "delta_prep: alpha / beta rows"
        );
    }
    let mut args = vec![conv, alpha, beta, dt_bias, a];
    if let Some(st) = store {
        for (t, ab) in st
            .iter()
            .zip([[head_k, n_k], [head_k, n_k], [1, n_v], [1, n_v]])
        {
            assert_eq!(t.ne(), [ab[0], ab[1], nt, ns], "delta_prep: store shape");
            assert!(
                t.ty() == types::F32 && t.nb()[0] == 4 && t.nb()[1] == ab[0] as usize * 4,
                "delta_prep: store rows contiguous"
            );
        }
        // Written as side effects into the replay pool (memory of its own).
        args.extend(st);
    }
    let qk = if gates_only { 0 } else { head_k * n_k * n };
    let gb = n_v * n;
    ctx.external_with(
        &DELTA_PREP,
        types::F32,
        [2 * qk + 2 * gb, 1, 1, 1],
        &args,
        &[
            head_k as i32,
            n_k as i32,
            head_v as i32,
            n_v as i32,
            eps.to_bits() as i32,
            gates_only as i32,
        ],
    )
}

unsafe fn cpu_delta_prep(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `delta_prep`, host tensors. The CPU graphs keep ggml's chain;
    // this is the op's reference for tests (sums in f32, not ggml's order).
    unsafe {
        let [head_k, n_k, head_v, n_v, eps] = delta_params(dst);
        let eps = f32::from_bits(eps as u32);
        let (head_k, n_k, head_v, n_v) =
            (head_k as usize, n_k as usize, head_v as usize, n_v as usize);
        let conv = &*dst.src[0];
        let (alpha, beta, dt_bias, a) = (&*dst.src[1], &*dst.src[2], &*dst.src[3], &*dst.src[4]);
        let (nt, ns) = (conv.ne[1] as usize, conv.ne[2] as usize);
        let n = nt * ns;
        let qk = head_k * n_k;
        let qkv_dim = 2 * qk + head_v * n_v;
        let cp = conv.data as *const f32;
        let out = dst.data as *mut f32;
        let scale_out = 1.0 / (head_k as f32).sqrt();
        let store = |i: usize| dst.src[5 + i].as_ref();
        // `delta_gates`: g and beta only, from the output's start.
        let gates_only = crate::ggml::external_params(dst)[5] != 0;
        let (n_qk, at_g) = if gates_only { (0, 0) } else { (2, 2 * qk * n) };
        for t in (ith..n).step_by(nth) {
            let (ts, s) = (t % nt, t / nt);
            let row = cp.add(t * qkv_dim);
            for which in 0..n_qk {
                for h in 0..n_k {
                    let off = which * qk + h * head_k;
                    let x = std::slice::from_raw_parts(row.add(off), head_k);
                    let mean = x.iter().map(|v| v * v).sum::<f32>() / head_k as f32;
                    let scale = 1.0 / (mean + eps).sqrt();
                    for (i, v) in x.iter().enumerate() {
                        let y = scale_out * (scale * v);
                        *out.add(which * qk * n + t * qk + h * head_k + i) = y;
                        if let Some(st) = store(which) {
                            let p = (st.data as *mut u8).add(ts * st.nb[2] + s * st.nb[3]);
                            *(p as *mut f32).add(h * head_k + i) = y;
                        }
                    }
                }
            }
            let at = |t2: &ffi::ggml_tensor, i: usize| {
                *((t2.data as *const u8).add(t * t2.nb[1]) as *const f32).add(i)
            };
            for i in 0..n_v {
                let x = at(alpha, i) + *(dt_bias.data as *const f32).add(i);
                let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                let gv = sp * *(a.data as *const f32).add(i);
                let bv = 1.0 / (1.0 + (-at(beta, i)).exp());
                *out.add(at_g + t * n_v + i) = gv;
                *out.add(at_g + n_v * n + t * n_v + i) = bv;
                for (k, v) in [(2, gv), (3, bv)] {
                    if let Some(st) = store(k) {
                        let p = (st.data as *mut u8).add(ts * st.nb[2] + s * st.nb[3]);
                        *(p as *mut f32).add(i) = v;
                    }
                }
            }
        }
    }
}

/// `delta_prep`'s params: head_k, n_k, head_v, n_v and the eps's bits.
unsafe fn delta_params(dst: &ffi::ggml_tensor) -> [i32; 5] {
    let p = crate::ggml::external_params(dst);
    [p[0], p[1], p[2], p[3], p[4]]
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct DeltaPrep {
    conv: *const f32,
    alpha: *const f32,
    alpha_ld: i64,
    beta_in: *const f32,
    beta_ld: i64,
    dt_bias: *const f32,
    a: *const f32,
    q: *mut f32,
    k: *mut f32,
    g: *mut f32,
    beta: *mut f32,
    st: [*mut f32; 4],
    st_ld: [i64; 4],
    st_sd: [i64; 4],
    head_k: i32,
    n_k: i32,
    head_v: i32,
    n_v: i32,
    nt: i32,
    n_tok: i32,
    eps: f32,
    scale_out: f32,
    gates_only: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_delta_prep_launch(p: *const DeltaPrep, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_delta_prep(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `delta_prep`, device tensors of the stream's device.
    unsafe {
        let [head_k, n_k, head_v, n_v, eps] = delta_params(dst);
        let conv = &*dst.src[0];
        let (alpha, beta) = (&*dst.src[1], &*dst.src[2]);
        let (nt, ns) = (conv.ne[1], conv.ne[2]);
        let n = nt * ns;
        // `delta_gates`: no q and k in the output.
        let gates_only = crate::ggml::external_params(dst)[5];
        let qk = if gates_only != 0 {
            0
        } else {
            (head_k * n_k) as i64 * n
        };
        let gb = n_v as i64 * n;
        let out = dst.data as *mut f32;
        let mut st = [std::ptr::null_mut(); 4];
        let mut st_ld = [0i64; 4];
        let mut st_sd = [0i64; 4];
        for i in 0..4 {
            if let Some(t) = dst.src[5 + i].as_ref() {
                st[i] = t.data as *mut f32;
                st_ld[i] = (t.nb[2] / 4) as i64;
                st_sd[i] = (t.nb[3] / 4) as i64;
            }
        }
        let p = DeltaPrep {
            conv: conv.data as *const f32,
            alpha: alpha.data as *const f32,
            alpha_ld: (alpha.nb[1] / 4) as i64,
            beta_in: beta.data as *const f32,
            beta_ld: (beta.nb[1] / 4) as i64,
            dt_bias: (*dst.src[3]).data as *const f32,
            a: (*dst.src[4]).data as *const f32,
            q: out,
            k: out.add(qk as usize),
            g: out.add(2 * qk as usize),
            beta: out.add((2 * qk + gb) as usize),
            st,
            st_ld,
            st_sd,
            head_k,
            n_k,
            head_v,
            n_v,
            nt: nt as i32,
            n_tok: n as i32,
            eps: f32::from_bits(eps as u32),
            scale_out: 1.0 / (head_k as f32).sqrt(),
            gates_only,
        };
        match legion_delta_prep_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err(format!(
                "delta_prep: head size {head_k} or {n} tokens unsupported"
            )),
            e => Err(format!("delta_prep: CUDA error {e}")),
        }
    }
}

/// Whether graphs built from now on use [`gated_norm`].
static GATED_NORM_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether graphs built from now on use [`gated_norm`] where it fits (A/B tests; off by
/// default until measured, `legion serve --gated-norm` turns it on).
pub fn set_gated_norm(on: bool) {
    GATED_NORM_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`gated_norm`] takes this layer: on a GPU only (the CPU graph keeps ggml's chain),
/// heads of at most 256, an F32 weight in device memory, rows contiguous, and switched on.
pub(super) fn gated_norm_fits(o: Tensor<'_>, w: Tensor<'_>, z: Tensor<'_>, head_v: i64) -> bool {
    GATED_NORM_ON.load(std::sync::atomic::Ordering::Relaxed) && gated_norm_shapes(o, w, z, head_v)
}

/// Whether the CUDA op takes these tensors (see [`gated_norm_fits`]).
pub(super) fn gated_norm_shapes(o: Tensor<'_>, w: Tensor<'_>, z: Tensor<'_>, head_v: i64) -> bool {
    let rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
    cfg!(feature = "cuda")
        && head_v <= 256
        && o.ne()[1] <= 65535
        && rows(o)
        && rows(z)
        && w.ty() == types::F32
        && !w.is_host()
        && !w.is_meta()
        && w.is_contiguous()
        && w.elements() == head_v as usize
}

/// Whether [`gated_norm_bf16`] takes this layer's output norm for a prompt step of `n` tokens
/// (more than [`NORM_BF16_MIN_N`]) on a GPU with BF16 tensor cores (whatever `--gated-norm`
/// says).
pub(super) fn gated_norm_bf16_fits(
    o: Tensor<'_>,
    w: Tensor<'_>,
    z: Tensor<'_>,
    head_v: i64,
    n: i64,
) -> bool {
    n > NORM_BF16_MIN_N && gated_norm_shapes(o, w, z, head_v) && crate::model::fp8::bf16_mma()
}

/// The norms that write BF16 for the matmuls ([`gated_norm_bf16`], [`rms_norm_bf16`]) take
/// steps of more than this many tokens: prompt chunks, where the matmuls read BF16
/// activations (the FP8 GEMM above 32 tokens, the NVFP4 GEMMs).
pub(super) const NORM_BF16_MIN_N: i64 = 32;

/// A gated DeltaNet layer's output norm: per token and `head_v`-wide head of `o` (F32
/// `[head_v * n_v, n]`, rows contiguous), `rms_norm(o, eps) * w * gate(z)` with `gate`
/// sigmoid or (`silu`) SiLU of `z` (same shape), as ggml's chain computes it: F32 `[head_v *
/// n_v, n]`.
pub(super) fn gated_norm<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    w: Tensor<'a>,
    z: Tensor<'a>,
    head_v: i64,
    eps: f32,
    silu: bool,
) -> Tensor<'a> {
    gated_norm_as(ctx, o, w, z, head_v, eps, silu, types::F32)
}

/// [`gated_norm`] rounded to BF16 as ggml's cast rounds it: `cast(gated_norm(..), BF16)`'s
/// values, without the F32 rows.
pub(super) fn gated_norm_bf16<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    w: Tensor<'a>,
    z: Tensor<'a>,
    head_v: i64,
    eps: f32,
    silu: bool,
) -> Tensor<'a> {
    gated_norm_as(ctx, o, w, z, head_v, eps, silu, types::BF16)
}

#[allow(clippy::too_many_arguments)]
fn gated_norm_as<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    w: Tensor<'a>,
    z: Tensor<'a>,
    head_v: i64,
    eps: f32,
    silu: bool,
    ty: ffi::ggml_type,
) -> Tensor<'a> {
    let [c, n, _, _] = o.ne();
    assert_eq!(z.ne()[..2], [c, n], "gated_norm: z shape");
    assert_eq!(c % head_v, 0, "gated_norm: heads");
    ctx.external_with(
        &GATED_NORM,
        ty,
        [c, n, 1, 1],
        &[o, w, z],
        &[head_v as i32, i32::from(silu), eps.to_bits() as i32],
    )
}

/// `mul(rms_norm(x, eps), w)` rounded to BF16 as ggml's cast rounds it, for each row of `x` (F32
/// `[ncols, n]`, rows contiguous; `w` F32 `[ncols]`): `cast(mul(rms_norm(x), w), BF16)`'s
/// values on a GPU (ggml-cuda's reduction order), without the F32 rows. BF16 `[ncols, n]`.
pub(crate) fn rms_norm_bf16<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let [c, n, _, _] = x.ne();
    assert_eq!(w.elements(), c as usize, "rms_norm_bf16: weight size");
    ctx.external_with(
        &RMS_NORM_BF16,
        types::BF16,
        [c, n, 1, 1],
        &[x, w],
        &[eps.to_bits() as i32],
    )
}

/// Whether [`rms_norm_bf16`] takes `x` and `w` for a prompt step: more than
/// [`NORM_BF16_MIN_N`] rows of F32 `x` (2-D, rows contiguous) and an F32 `w` in device memory,
/// on a GPU with BF16 tensor cores.
pub(crate) fn rms_norm_bf16_fits(x: Tensor<'_>, w: Tensor<'_>) -> bool {
    let [c, n, x2, x3] = x.ne();
    cfg!(feature = "cuda")
        && n > NORM_BF16_MIN_N
        && n <= i32::MAX as i64
        && x2 == 1
        && x3 == 1
        && x.ty() == types::F32
        && x.nb()[0] == 4
        && w.ty() == types::F32
        && w.is_contiguous()
        && w.elements() == c as usize
        && !w.is_host()
        && !w.is_meta()
        && crate::model::fp8::bf16_mma()
}

unsafe fn cpu_rms_norm_bf16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `rms_norm_bf16`, host tensors. The CPU graphs keep ggml's chain;
    // this is the op's reference for tests (sums in f32, not ggml's order).
    unsafe {
        let eps = f32::from_bits(crate::ggml::external_params(dst)[0] as u32);
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let (c, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let w = std::slice::from_raw_parts(w.data as *const f32, c);
        let out = dst.data as *mut u16;
        for t in (ith..n).step_by(nth) {
            let row = (x.data as *const u8).add(t * x.nb[1]) as *const f32;
            let row = std::slice::from_raw_parts(row, c);
            let mean = row.iter().map(|v| v * v).sum::<f32>() / c as f32;
            let scale = 1.0 / (mean + eps).sqrt();
            for (i, v) in row.iter().enumerate() {
                *out.add(t * c + i) = crate::model::nvfp4_mm::bf16_bits(scale * v * w[i]);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct RmsNormBf16 {
    x: *const f32,
    x_ld: i64,
    w: *const f32,
    out: *mut c_void,
    ncols: i32,
    n_rows: i32,
    eps: f32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_rms_norm_bf16_launch(p: *const RmsNormBf16, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_rms_norm_bf16(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `rms_norm_bf16`, device tensors of the stream's device.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let p = RmsNormBf16 {
            x: x.data as *const f32,
            x_ld: (x.nb[1] / 4) as i64,
            w: w.data as *const f32,
            out: dst.data,
            ncols: x.ne[0] as i32,
            n_rows: x.ne[1] as i32,
            eps: f32::from_bits(crate::ggml::external_params(dst)[0] as u32),
        };
        match legion_rms_norm_bf16_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err(format!("rms_norm_bf16: shape {:?} unsupported", x.ne)),
            e => Err(format!("rms_norm_bf16: CUDA error {e}")),
        }
    }
}

/// `op(x) * g` rounded to BF16 as ggml's cast rounds it, with `op` sigmoid or (`silu`) SiLU:
/// `cast(mul(g, sigmoid(x)), BF16)` or `cast(swiglu_split(x, g), BF16)`'s values on a GPU,
/// without the F32 rows. `x` is F32 `[d, n_head, n]` (columns contiguous, any head and token
/// strides: the gate half of attn_q's interleaved heads) or `[d * n_head, n]`; `g` F32 `[d *
/// n_head, n]`, rows contiguous. BF16 `[d * n_head, n]`.
pub(crate) fn gate_bf16<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    g: Tensor<'a>,
    silu: bool,
) -> Tensor<'a> {
    let [k, n, _, _] = g.ne();
    assert_eq!(x.elements(), (k * n) as usize, "gate_bf16: shapes");
    ctx.external_with(
        &GATE_BF16,
        types::BF16,
        [k, n, 1, 1],
        &[x, g],
        &[i32::from(silu)],
    )
}

/// Whether [`gate_bf16`] takes `x` and `g` for a prompt step: more than [`NORM_BF16_MIN_N`]
/// tokens of F32, `x` 2-D or 3-D with contiguous columns and `g` 2-D with contiguous rows of
/// the same size, on a GPU with BF16 tensor cores.
pub(crate) fn gate_bf16_fits(x: Tensor<'_>, g: Tensor<'_>) -> bool {
    let [k, n, g2, g3] = g.ne();
    let [d, xh, xn, x3] = x.ne();
    let x_ok = if xn == 1 && x3 == 1 {
        d == k && xh == n
    } else {
        x3 == 1 && xn == n && d * xh == k
    };
    cfg!(feature = "cuda")
        && n > NORM_BF16_MIN_N
        && n <= 65535
        && g2 == 1
        && g3 == 1
        && x_ok
        && x.ty() == types::F32
        && g.ty() == types::F32
        && x.nb()[0] == 4
        && g.nb()[0] == 4
        && k <= i32::MAX as i64
        && crate::model::fp8::bf16_mma()
}

/// `x`'s (head stride, token stride) in elements, as [`gate_bf16`] reads it.
fn gate_strides(x: &ffi::ggml_tensor) -> (usize, usize) {
    if x.ne[2] == 1 && x.ne[3] == 1 {
        (0, x.nb[1] / 4)
    } else {
        (x.nb[1] / 4, x.nb[2] / 4)
    }
}

unsafe fn cpu_gate_bf16(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `gate_bf16`, host tensors. The CPU graphs keep ggml's ops; this is
    // the op's reference for tests.
    unsafe {
        let silu = crate::ggml::external_params(dst)[0] != 0;
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let (k, n) = (g.ne[0] as usize, g.ne[1] as usize);
        let d = x.ne[0] as usize;
        let (x_hd, x_ld) = gate_strides(x);
        let (xp, gp, out) = (
            x.data as *const f32,
            g.data as *const u8,
            dst.data as *mut u16,
        );
        for t in (ith..n).step_by(nth) {
            let grow = gp.add(t * g.nb[1]) as *const f32;
            for i in 0..k {
                let xv = *xp.add(t * x_ld + (i / d) * x_hd + i % d);
                let e = 1.0 + (-xv).exp();
                let a = if silu { xv / e } else { 1.0 / e };
                *out.add(t * k + i) = crate::model::nvfp4_mm::bf16_bits(a * *grow.add(i));
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct GateBf16 {
    x: *const f32,
    x_ld: i64,
    x_hd: i64,
    g: *const f32,
    g_ld: i64,
    out: *mut c_void,
    d: i32,
    n_head: i32,
    n_tok: i32,
    silu: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_gate_bf16_launch(p: *const GateBf16, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gate_bf16(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gate_bf16`, device tensors of the stream's device.
    unsafe {
        let (x, g) = (&*dst.src[0], &*dst.src[1]);
        let (x_hd, x_ld) = gate_strides(x);
        let d = x.ne[0];
        let p = GateBf16 {
            x: x.data as *const f32,
            x_ld: x_ld as i64,
            x_hd: x_hd as i64,
            g: g.data as *const f32,
            g_ld: (g.nb[1] / 4) as i64,
            out: dst.data,
            d: d as i32,
            n_head: (g.ne[0] / d) as i32,
            n_tok: g.ne[1] as i32,
            silu: crate::ggml::external_params(dst)[0],
        };
        match legion_gate_bf16_launch(&p, stream) {
            0 => Ok(()),
            -1 => Err(format!("gate_bf16: shape {:?} unsupported", g.ne)),
            e => Err(format!("gate_bf16: CUDA error {e}")),
        }
    }
}

unsafe fn cpu_gated_norm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `gated_norm`, host tensors. The CPU graphs keep ggml's chain;
    // this is the op's reference for tests (sums in f32, not ggml's order).
    unsafe {
        let p = crate::ggml::external_params(dst);
        let (head_v, silu, eps) = (p[0] as usize, p[1] != 0, f32::from_bits(p[2] as u32));
        let (o, w, z) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (c, n) = (o.ne[0] as usize, o.ne[1] as usize);
        let wp = w.data as *const f32;
        let bf16 = dst.type_ == types::BF16;
        for t in (ith..n).step_by(nth) {
            let op = (o.data as *const u8).add(t * o.nb[1]) as *const f32;
            let zp = (z.data as *const u8).add(t * z.nb[1]) as *const f32;
            for h in 0..c / head_v {
                let x = std::slice::from_raw_parts(op.add(h * head_v), head_v);
                let mean = x.iter().map(|v| v * v).sum::<f32>() / head_v as f32;
                let scale = 1.0 / (mean + eps).sqrt();
                for (i, v) in x.iter().enumerate() {
                    let zv = *zp.add(h * head_v + i);
                    let g = if silu {
                        zv / (1.0 + (-zv).exp())
                    } else {
                        1.0 / (1.0 + (-zv).exp())
                    };
                    let y = scale * v * *wp.add(i) * g;
                    let at = t * c + h * head_v + i;
                    if bf16 {
                        *(dst.data as *mut u16).add(at) = crate::model::nvfp4_mm::bf16_bits(y);
                    } else {
                        *(dst.data as *mut f32).add(at) = y;
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct GatedNorm {
    o: *const f32,
    o_ld: i64,
    w: *const f32,
    z: *const f32,
    z_ld: i64,
    out: *mut c_void,
    head_v: i32,
    n_v: i32,
    n_tok: i32,
    silu: i32,
    eps: f32,
    out_bf16: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_gated_norm_launch(p: *const GatedNorm, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gated_norm(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gated_norm`, device tensors of the stream's device.
    unsafe {
        let p = crate::ggml::external_params(dst);
        let (o, w, z) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let head_v = p[0];
        let g = GatedNorm {
            o: o.data as *const f32,
            o_ld: (o.nb[1] / 4) as i64,
            w: w.data as *const f32,
            z: z.data as *const f32,
            z_ld: (z.nb[1] / 4) as i64,
            out: dst.data,
            head_v,
            n_v: (o.ne[0] / head_v as i64) as i32,
            n_tok: o.ne[1] as i32,
            silu: p[1],
            eps: f32::from_bits(p[2] as u32),
            out_bf16: i32::from(dst.type_ == types::BF16),
        };
        match legion_gated_norm_launch(&g, stream) {
            0 => Ok(()),
            -1 => Err(format!(
                "gated_norm: head size {head_v} or {} tokens unsupported",
                o.ne[1]
            )),
            e => Err(format!("gated_norm: CUDA error {e}")),
        }
    }
}

/// Whether graphs built from now on use [`moe_combine`].
static MOE_COMBINE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether graphs built from now on use [`moe_combine`] (A/B tests; off by default,
/// `legion serve --moe-combine` turns it on).
pub fn set_moe_combine(on: bool) {
    MOE_COMBINE_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`moe_combine`] takes these experts' outputs: on a GPU only (the CPU graph keeps
/// ggml's chain), contiguous F32, and switched on ([`set_moe_combine`]).
pub(super) fn moe_combine_fits(experts: Tensor<'_>, w: Tensor<'_>) -> bool {
    let on = MOE_COMBINE_ON.load(std::sync::atomic::Ordering::Relaxed);
    on && cfg!(feature = "cuda")
        && [experts, w]
            .iter()
            .all(|t| t.ty() == types::F32 && t.is_contiguous())
}

/// A MoE layer's output from its experts' (F32 `[n_embd, n_used, n]`, contiguous) and their
/// weights (F32 `[1, n_used, n]`): `Σ_e experts[:, e, t] · w[e, t]`, `[n_embd, n]`, always
/// rounded as ggml-cuda's fused `moe_weighted_reduction` rounds it (the first product, then
/// one fma per expert). ggml runs that fusion only when the graph's tensors do not overlap in
/// memory and a mul and adds otherwise (two roundings per expert), so the same values came
/// out different whenever a change elsewhere moved the allocation (#410).
pub(super) fn moe_combine<'a>(ctx: &'a Context, experts: Tensor<'a>, w: Tensor<'a>) -> Tensor<'a> {
    let [e, used, n, _] = experts.ne();
    assert_eq!(w.elements(), (used * n) as usize, "moe_combine: weights");
    ctx.external(&MOE_COMBINE, types::F32, [e, n, 1, 1], &[experts, w])
}

unsafe fn cpu_moe_combine(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `moe_combine`, host tensors, contiguous.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let (e, used, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let (xp, wp, out) = (
            x.data as *const f32,
            w.data as *const f32,
            dst.data as *mut f32,
        );
        for t in (ith..n).step_by(nth) {
            for c in 0..e {
                let mut sum = *xp.add(t * used * e + c) * *wp.add(t * used);
                for k in 1..used {
                    sum = (*xp.add((t * used + k) * e + c)).mul_add(*wp.add(t * used + k), sum);
                }
                *out.add(t * e + c) = sum;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct MoeCombine {
    experts: *const f32,
    w: *const f32,
    out: *mut f32,
    n_embd: i64,
    n_used: i32,
    n_tok: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_moe_combine_launch(p: *const MoeCombine, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_moe_combine(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `moe_combine`, device tensors of the stream's device.
    unsafe {
        let (x, w) = (&*dst.src[0], &*dst.src[1]);
        let p = MoeCombine {
            experts: x.data as *const f32,
            w: w.data as *const f32,
            out: dst.data as *mut f32,
            n_embd: x.ne[0],
            n_used: x.ne[1] as i32,
            n_tok: x.ne[2] as i32,
        };
        match legion_moe_combine_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("moe_combine: CUDA error {e}")),
        }
    }
}

/// Whether graphs built from now on use [`shexp_gate`].
static SHEXP_GATE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether graphs built from now on use [`shexp_gate`] where it fits. Off by default:
/// the op is ggml-cuda's chain bit for bit, but the graph's new layout can flip ggml-cuda's
/// layout-dependent MoE reduction fusion and so move speculative outputs (#256, #410);
/// `legion serve --shexp-gate` turns it on.
pub fn set_shexp_gate(on: bool) {
    SHEXP_GATE_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a MoE layer whose shared-expert gate weight is `w` may take [`shexp_gate`]: on a
/// GPU only (the CPU graph keeps ggml's chain, the reference), not split over devices, and
/// switched on ([`set_shexp_gate`]).
pub(super) fn shexp_gate_on(w: Tensor<'_>) -> bool {
    SHEXP_GATE_ON.load(std::sync::atomic::Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && !w.is_host()
        && !w.is_meta()
}

/// Whether [`shexp_gate`] takes these: [`shexp_gate_on`], F32, `moe` and `sh` contiguous of one
/// shape `[e, n]`, `g` `[1, n]`.
pub(super) fn shexp_gate_fits(
    moe: Tensor<'_>,
    sh: Tensor<'_>,
    g: Tensor<'_>,
    w: Tensor<'_>,
) -> bool {
    shexp_gate_on(w) && shexp_gate_shapes(moe, sh, g)
}

/// Whether the op takes these tensors (see [`shexp_gate_fits`], whatever the switch says).
pub(super) fn shexp_gate_shapes(moe: Tensor<'_>, sh: Tensor<'_>, g: Tensor<'_>) -> bool {
    let [e, n, s2, s3] = moe.ne();
    cfg!(feature = "cuda")
        && [moe, sh, g].iter().all(|t| t.ty() == types::F32)
        && moe.is_contiguous()
        && sh.is_contiguous()
        && sh.ne() == [e, n, s2, s3]
        && s2 * s3 == 1
        && g.ne() == [1, n, 1, 1]
        && n <= i32::MAX as i64
}

/// A MoE layer's output `moe` plus its shared expert's `sh` (F32 `[e, n]` each, contiguous)
/// gated by the logits `g` (F32 `[1, n]`, any row stride): `moe + sh · sigmoid(g)`, `[e, n]`,
/// rounded as ggml's `add(moe, mul(sh, sigmoid(g)))`: each step on its own.
pub(super) fn shexp_gate<'a>(
    ctx: &'a Context,
    moe: Tensor<'a>,
    sh: Tensor<'a>,
    g: Tensor<'a>,
) -> Tensor<'a> {
    let [e, n, _, _] = moe.ne();
    assert_eq!(sh.ne()[..2], [e, n], "shexp_gate: shared expert shape");
    assert_eq!(g.ne()[..2], [1, n], "shexp_gate: gate shape");
    ctx.external(&SHEXP_GATE, types::F32, [e, n, 1, 1], &[moe, sh, g])
}

unsafe fn cpu_shexp_gate(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `shexp_gate`, host tensors.
    unsafe {
        let (moe, sh, g) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (e, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let (mp, sp, out) = (
            moe.data as *const f32,
            sh.data as *const f32,
            dst.data as *mut f32,
        );
        for t in (ith..n).step_by(nth) {
            let gv = *((g.data as *const u8).add(t * g.nb[1]) as *const f32);
            let s = 1.0 / (1.0 + (-gv).exp());
            for c in 0..e {
                let i = t * e + c;
                let scaled = *sp.add(i) * s;
                *out.add(i) = *mp.add(i) + scaled;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct ShexpGate {
    moe: *const f32,
    sh: *const f32,
    g: *const f32,
    g_ld: i64,
    out: *mut f32,
    n_embd: i64,
    n_tok: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_shexp_gate_launch(p: *const ShexpGate, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_shexp_gate(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `shexp_gate`, device tensors of the stream's device.
    unsafe {
        let (moe, sh, g) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let p = ShexpGate {
            moe: moe.data as *const f32,
            sh: sh.data as *const f32,
            g: g.data as *const f32,
            g_ld: (g.nb[1] / 4) as i64,
            out: dst.data as *mut f32,
            n_embd: dst.ne[0],
            n_tok: dst.ne[1] as i32,
        };
        match legion_shexp_gate_launch(&p, stream) {
            0 => Ok(()),
            e => Err(format!("shexp_gate: CUDA error {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
