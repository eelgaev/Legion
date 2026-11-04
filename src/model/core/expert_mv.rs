//! Decode MoE expert matvecs on CUDA: an external op ([`crate::ggml::ExternalOp`]) running
//! `src/cuda/expert_mv/` in place of ggml's `mul_mat_id` for steps of a few tokens (decode, MTP
//! verify). It reads each selected expert once per step, whichever tokens chose it, and keeps
//! many loads in flight, which matters when the experts are host memory read in place over
//! NVLink2 / ATS (`--host-experts`): ggml's mmvq waits on one load per lane there.
//!
//! **Where experts live.** Expert `id` is `w->data + id * nb[2]`, as for `mul_mat_id`. That
//! address may be VRAM or host memory the GPU reads in place; the kernel doesn't care which. A
//! VRAM expert cache passes a tiered tensor (`[ne0, ne1, slots + n_expert]`: cache slots, then the
//! host copies) and remapped ids (#37).
//!
//! Numerics are ggml's: activations quantized to Q8_1 as `quantize_q8_1` does, then ggml's own
//! Q4_K / Q5_K / Q5_1 / Q6_K / Q8_0 dot products, so results match `mul_mat_id` up to summation order. The CPU
//! implementation is a plain f32 reference, for tests of the wiring. GLM-5.3-Flash's graphs also
//! run Project Maya's quants on it ([`maya_type`]: Q2_K, Q3_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S,
//! IQ4_XS, #559), with ggml's per-user vec_dot only.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static EXPERT_MV: ExternalOp = ExternalOp {
    name: "expert_mv",
    cpu: cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_expert_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The SwiGLU variant: `silu(gate x) * (up x)` in one op ([`glu_tiered`]).
static EXPERT_MV_GLU: ExternalOp = ExternalOp {
    name: "expert_mv_glu",
    cpu: cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_expert_mv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// [`EXPERT_MV_GLU`] from activations quantized already ([`glu_tiered_preq`]): no `quantize_x`
/// launch. The same name, so stage graphs judge it as the SwiGLU op.
static EXPERT_MV_GLU_PREQ: ExternalOp = ExternalOp {
    name: "expert_mv_glu",
    cpu: cpu_expert_mv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_expert_mv_preq),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most tokens in a step the op takes by default (one sequence's decode and MTP verify steps);
/// bigger steps stay on ggml's `mul_mat_id` (and #232's fused op). The op computes steps of up
/// to 64 tokens exactly (`--expert-mv-max-tokens`): that halved short prompts' time to the
/// first token on an AC922, but 16-sequence decode batches, whose experts are mostly VRAM cache
/// hits, ran faster on ggml's MMQ (79.1 against 74.8 tok/s).
pub const MAX_TOKENS: i64 = 8;

/// Most tokens of a step the op computes exactly (its kernels' 512 (token, expert) pairs at 8
/// experts a token); [`set_max_tokens`] and [`fits_glm_up_to`] stay within it.
pub const EXACT_MAX_TOKENS: i64 = 64;

/// [`MAX_TOKENS`], or what [`set_max_tokens`] set (for A/B runs).
pub fn max_tokens() -> i64 {
    match MAX_TOKENS_SET.load(Ordering::Relaxed) {
        0 => MAX_TOKENS,
        n => n as i64,
    }
}

static MAX_TOKENS_SET: AtomicUsize = AtomicUsize::new(0);

/// The most tokens of a step the op takes in graphs built from now on (0: [`MAX_TOKENS`]; the
/// op computes up to 64 exactly). For A/B runs: `legion serve --expert-mv-max-tokens`.
pub fn set_max_tokens(n: usize) {
    MAX_TOKENS_SET.store(n, Ordering::Relaxed);
}

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on use the op where it applies, or ggml's mul_mat_id (the
/// default). The engine turns it on for models it runs on CUDA: a graph doesn't know its backend
/// when it is built, and on the CPU the op would only be a slow reference.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

static GLU: AtomicBool = AtomicBool::new(true);

/// Whether gate and up take the SwiGLU op (the default) or two ops (for A/B runs: `legion serve
/// --expert-mv-split-glu`), in graphs built from now on.
pub fn set_glu(on: bool) {
    GLU.store(on, Ordering::Relaxed);
}

/// Sets [`DEVICE`] for graphs built from now on (`legion serve --expert-mv-device`).
pub fn set_device(on: bool) {
    DEVICE.store(on, Ordering::Relaxed);
}

/// `--expert-mv-device`: the gate and up projections of experts in GPU memory (not
/// tiered) take the SwiGLU op in verify steps (#309): one launch for both matrices, each distinct
/// expert read once. On Q4_K's 2560-wide rows it sums as ggml's mmvq does (a warp per row, the
/// same lanes), so for 2 to 8 tokens the op's result is ggml's bit for bit. The model's output is
/// not byte-identical end to end: the changed graph layout can flip ggml-cuda's layout-dependent
/// fusions, so greedy text can differ from the first multi-token verify step (#410, #472).
static DEVICE: AtomicBool = AtomicBool::new(false);

/// Whether experts in GPU memory take the op ([`DEVICE`], with the op on: [`set_enabled`]).
pub fn on_device() -> bool {
    ENABLED.load(Ordering::Relaxed) && DEVICE.load(Ordering::Relaxed)
}

/// Whether `mul_mat_id(w, x, ids)` runs on [`mul_mat_id`]: [`fits`], with the expert stack in
/// host memory (read in place by the GPU). Experts in VRAM stay on ggml's mmvq, which is faster
/// there (V100: ~460-550 GB/s against this kernel's ~250-320).
pub fn usable(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    fits(w, x, ids) && w.is_host()
}

/// Whether the op can compute `mul_mat_id(w, x, ids)` and is enabled: a Q4_K / Q5_K / Q5_1 /
/// Q6_K / Q8_0 expert stack with rows of a multiple of 4 bytes (at most a quarter tile), F32
/// activations with contiguous columns, and at most [`MAX_TOKENS`] tokens, wherever the experts
/// live.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    fits_as(w, x, ids, false, max_tokens())
}

/// [`fits`] for GLM-5.3-Flash's graphs, which also take [`maya_type`] stacks.
pub fn fits_glm(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    fits_glm_up_to(w, x, ids, max_tokens())
}

/// [`fits_glm`] for steps of up to `max` tokens instead of [`max_tokens`] (at most
/// [`EXACT_MAX_TOKENS`]): GLM-5.3-Flash's verify steps over an expert cache (#569).
pub fn fits_glm_up_to(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>, max: i64) -> bool {
    fits_as(w, x, ids, true, max.min(EXACT_MAX_TOKENS))
}

fn fits_as(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>, glm: bool, max: i64) -> bool {
    let (wn, xn, idn) = (w.ne(), x.ne(), ids.ne());
    stack_fits_as(w, glm)
        && x.ty() == types::F32
        && x.nb()[0] == 4
        && xn[0] == wn[0]
        && (xn[1] == 1 || xn[1] == idn[0])
        && xn[2] == idn[1]
        && xn[3] == 1
        && ids.ty() == types::I32
        && idn[1] <= max
}

/// GLM-5.3-Flash's Maya quants (#559), which the kernels take with ggml's per-user dot products
/// only. Only GLM's graphs ([`fits_glm`], [`glu_fits_narrow`]) route them here: other models
/// keep ggml's `mul_mat_id` for these types, as before.
pub fn maya_type(ty: ffi::ggml_type) -> bool {
    matches!(
        ty,
        types::Q2_K
            | types::Q3_K
            | types::IQ2_XXS
            | types::IQ2_S
            | types::IQ3_XXS
            | types::IQ3_S
            | types::IQ4_XS
    )
}

/// [`fits`]' conditions on the expert stack `w` alone (and that the op is enabled); with `glm`
/// also [`maya_type`] stacks.
fn stack_fits_as(w: Tensor<'_>, glm: bool) -> bool {
    ENABLED.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && (matches!(
            w.ty(),
            types::Q4_K | types::Q5_K | types::Q5_1 | types::Q6_K | types::Q8_0
        ) || (glm && maya_type(w.ty())))
        && w.nb()[1].is_multiple_of(4)
        && w.nb()[1] * MIN_ROWS <= TILE_BYTES
        && w.nb()[2].is_multiple_of(16)
        && (w.nb()[1] * w.ne()[1] as usize).is_multiple_of(16)
        && w.ne()[0] % 32 == 0
}

/// `mul_mat_id(w, x, ids)`: `w` `[k, rows, n_expert]`, `x` F32 `[k, 1 or n_used, n_tok]`, `ids`
/// I32 `[n_used, n_tok]`; F32 `[rows, n_used, n_tok]`.
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    mul_mat_id_tiered(ctx, w, x, ids, 0)
}

/// [`mul_mat_id`] over a tiered expert tensor (`crate::residency`): ids below `vram_slots` are
/// VRAM cache slots, the others experts read in place from host memory. The kernel reads each
/// tier its own way.
pub fn mul_mat_id_tiered<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
) -> Tensor<'a> {
    mul_mat_id_tiered_pad(ctx, w, x, ids, vram_slots, None)
}

/// The external param holding the source index of a verify step's row map (`pad_rows::map`,
/// 0: none; [`mul_mat_id_tiered_pad`], [`glu_tiered_pad`], [`glu_tiered_preq_pad`]). Only on
/// ops of no split ([`overlap::NONE`]), where param 3 is otherwise unused.
const P_PAD: usize = 3;

/// [`mul_mat_id_tiered`] with a verify step's row map `pad` (I32 `[n_tok]`, `pad_rows::map`;
/// `--moe-pad-skip-tiered`, #814): a padding row's pair whose expert no real row's pair chose (a
/// dead padding pair) is skipped and its rows written 0, so experts only padding rows chose are
/// never read. Every expert a real row chose is computed as without the map, for all its pairs
/// (padding ones too): the same bits for every real row. `None` is [`mul_mat_id_tiered`]'s
/// node exactly.
pub fn mul_mat_id_tiered_pad<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    pad: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, w, x, ids);
    let (rows, n_used, n_tok) = (w.ne()[1], ids.ne()[0], ids.ne()[1]);
    let mut srcs = vec![w, x, ids, xq];
    let mut params = vec![vram_slots as i32, overlap::NONE];
    if let Some(p) = pad {
        assert!(pad_fits(p, ids), "expert_mv: row map shape");
        params.extend([0, srcs.len() as i32]);
        srcs.push(p);
    }
    ctx.external_with(
        &EXPERT_MV,
        types::F32,
        [rows, n_used, n_tok, 1],
        &srcs,
        &params,
    )
}

/// Whether `pad` is a row map for `ids` (I32 `[n_used, n_tok]`): I32 `[n_tok]`, contiguous.
pub fn pad_fits(pad: Tensor<'_>, ids: Tensor<'_>) -> bool {
    pad.ty() == types::I32
        && pad.ne()[0] == ids.ne()[1]
        && pad.elements() as i64 == ids.ne()[1]
        && pad.is_contiguous()
}

/// `silu(gate x) * (up x)` for the experts `ids`, as [`mul_mat_id_tiered`] computes each
/// product, in one op: the activations are quantized once and each tile holds both matrices'
/// rows. `gate` and `up` must have the same type and layout, and [`fits`] both.
pub fn glu_tiered<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
) -> Tensor<'a> {
    glu_tiered_clamped(ctx, gate, up, x, ids, vram_slots, 0.0)
}

/// [`glu_tiered`] with a verify step's row map `pad` (see [`mul_mat_id_tiered_pad`]); `None` is
/// [`glu_tiered`]'s node exactly.
pub fn glu_tiered_pad<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    pad: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let Some(p) = pad else {
        return glu_tiered(ctx, gate, up, x, ids, vram_slots);
    };
    assert!(pad_fits(p, ids), "expert_mv_glu: row map shape");
    let xq = q8_1_scratch(ctx, gate, x, ids);
    let mut params = [0; P_LIMIT + 1];
    params[0] = vram_slots as i32;
    params[1] = overlap::NONE;
    params[P_PAD] = 5;
    glu_node(ctx, [gate, up], x, ids, xq, &[p], &params)
}

/// [`glu_tiered`] with a clamped SwiGLU, `silu(min(g, limit)) · clamp(u, -limit, limit)`
/// (GLM-5's `swiglu_limit`); `limit` 0 is [`glu_tiered`].
pub fn glu_tiered_clamped<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    limit: f32,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, gate, x, ids);
    glu_node(
        ctx,
        [gate, up],
        x,
        ids,
        xq,
        &[],
        &[
            vram_slots as i32,
            overlap::NONE,
            0,
            0,
            0,
            limit.to_bits() as i32,
        ],
    )
}

/// [`glu_tiered`] reading the activations' Q8_1 blocks from `xq` (#819): blocks an op wrote
/// already, `quantize_q8_1`'s bits of `x` (the caller's promise; [`glu_preq_fits`]), such as
/// those qwen4exp's hyper-connection mixer writes beside the layer input
/// ([`super::q8_group::set_quantized`]). The op then skips its `quantize_x` launch, whose blocks
/// are the same bits: `quantize_x` and ggml's `quantize_q8_1` take the same warp max and sum.
///
/// The node keeps the scratch [`glu_tiered`] would quantize into as a source of its own, unread,
/// so ggml's graph allocator lays the graph out as before: the greedy text depends on that
/// layout through ggml-cuda's layout-dependent MoE fusions (#819 round 3).
pub fn glu_tiered_preq<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    xq: Tensor<'a>,
) -> Tensor<'a> {
    glu_tiered_preq_pad(ctx, gate, up, x, ids, vram_slots, xq, None)
}

/// [`glu_tiered_preq`] with a verify step's row map `pad` (see [`mul_mat_id_tiered_pad`]); `None`
/// is [`glu_tiered_preq`]'s node exactly.
#[allow(clippy::too_many_arguments)]
pub fn glu_tiered_preq_pad<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    xq: Tensor<'a>,
    pad: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        preq_shapes(gate, x, ids, xq),
        "expert_mv_glu: pre-quantized activations"
    );
    let scratch = q8_1_scratch(ctx, gate, x, ids);
    assert!(
        gate.ty() == up.ty() && gate.ne() == up.ne() && gate.nb() == up.nb(),
        "expert_mv_glu: gate and up differ"
    );
    let (rows, n_used, n_tok) = (gate.ne()[1], ids.ne()[0], ids.ne()[1]);
    let mut srcs = vec![gate, x, ids, xq, up, scratch];
    let mut params = [vram_slots as i32, overlap::NONE, 0, 0, 0, 0];
    if let Some(p) = pad {
        assert!(pad_fits(p, ids), "expert_mv_glu: row map shape");
        params[P_PAD] = srcs.len() as i32;
        srcs.push(p);
    }
    ctx.external_with(
        &EXPERT_MV_GLU_PREQ,
        types::F32,
        [rows, n_used, n_tok, 1],
        &srcs,
        &params,
    )
}

/// Whether [`glu_tiered_preq`] takes the activations' Q8_1 blocks `xq` (an F32 node of
/// [`super::q8_group::quantized_elements`]) for `x` (`[k, 1, n_tok]`, one column per token) and
/// gate stack `gate`: block_q8_1 columns as `quantize_x` writes them for these weights (not the
/// IQ types' split layout, not the Q4_K per-unit activations of `--expert-mv-q4k-units`), and
/// `xq` holding them all.
pub fn glu_preq_fits(gate: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>, xq: Tensor<'_>) -> bool {
    !Q4K_UNITS.load(Ordering::Relaxed) && preq_shapes(gate, x, ids, xq)
}

/// [`glu_preq_fits`] but for the switches: the shapes and types alone.
fn preq_shapes(gate: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>, xq: Tensor<'_>) -> bool {
    let (k, n_tok) = (x.ne()[0], ids.ne()[1]);
    !maya_type(gate.ty())
        && x.ne()[1] == 1
        && x.ne()[2] == n_tok
        && xq.ty() == types::F32
        && xq.is_contiguous()
        && xq.elements() * 4 >= q8_1_column_bytes(k as usize) * n_tok as usize
}

/// [`glu_tiered`]'s node, with its activations' scratch `xq`, sources after the first five
/// (`more`) and `params` (the VRAM slots, the role in a split: [`overlap::NONE`] or
/// [`overlap::FORK`], ..., at [`P_LIMIT`] the SwiGLU clamp's f32 bits).
fn glu_node<'a>(
    ctx: &'a Context,
    [gate, up]: [Tensor<'a>; 2],
    x: Tensor<'a>,
    ids: Tensor<'a>,
    xq: Tensor<'a>,
    more: &[Tensor<'a>],
    params: &[i32],
) -> Tensor<'a> {
    assert!(
        gate.ty() == up.ty() && gate.ne() == up.ne() && gate.nb() == up.nb(),
        "expert_mv_glu: gate and up differ"
    );
    let (rows, n_used, n_tok) = (gate.ne()[1], ids.ne()[0], ids.ne()[1]);
    let srcs: Vec<Tensor<'a>> = [gate, x, ids, xq, up]
        .into_iter()
        .chain(more.iter().copied())
        .collect();
    ctx.external_with(
        &EXPERT_MV_GLU,
        types::F32,
        [rows, n_used, n_tok, 1],
        &srcs,
        params,
    )
}

/// Whether [`glu_tiered`] can take `gate` and `up`: both [`fits`], alike, and two rows of each
/// fit a quarter tile.
pub fn glu_fits(gate: Tensor<'_>, up: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    GLU.load(Ordering::Relaxed)
        && fits(gate, x, ids)
        && gate.ty() == up.ty()
        && gate.ne() == up.ne()
        && gate.nb() == up.nb()
        && 2 * MIN_ROWS * gate.nb()[1] <= TILE_BYTES
}

/// Whether a step's whole routed FFN of `x` runs on the op: [`glu_fits`] for gate and up, and
/// [`fits`] for the down stack on their result (`[gate rows, n_used, n_tok]`).
pub fn glu_down_fits(
    gate: Tensor<'_>,
    up: Tensor<'_>,
    down: Tensor<'_>,
    x: Tensor<'_>,
    ids: Tensor<'_>,
) -> bool {
    glu_fits(gate, up, x, ids)
        && stack_fits_as(down, false)
        && down.ne()[0] == gate.ne()[1]
        && down.ne()[2] == gate.ne()[2]
}

/// [`glu_fits`] down to tiles of [`NARROW_ROWS`] rows, for gate and up rows too wide for two
/// tiles of [`MIN_ROWS`] (GLM-5.3-Flash's Q4_K at 4096: 2 x 4 x 2304 bytes past 16 KiB), whose
/// two rows are a whole number of 16-byte loads. Shapes [`glu_fits`] takes get the same tiles.
pub fn glu_fits_narrow(gate: Tensor<'_>, up: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    glu_fits_narrow_up_to(gate, up, x, ids, max_tokens())
}

/// [`glu_fits_narrow`] for steps of up to `max` tokens ([`fits_glm_up_to`]).
pub fn glu_fits_narrow_up_to(
    gate: Tensor<'_>,
    up: Tensor<'_>,
    x: Tensor<'_>,
    ids: Tensor<'_>,
    max: i64,
) -> bool {
    GLU.load(Ordering::Relaxed)
        && fits_glm_up_to(gate, x, ids, max)
        && gate.ty() == up.ty()
        && gate.ne() == up.ne()
        && gate.nb() == up.nb()
        && (NARROW_ROWS * gate.nb()[1]).is_multiple_of(16)
        && 2 * NARROW_ROWS * gate.nb()[1] <= TILE_BYTES
}

/// Scratch for the activations as Q8_1, quantized once per step and column; with
/// [`set_q4k_units`] also the Q4_K per-unit layout ([`scratch_bytes`]).
fn q8_1_scratch<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
    let cols = (x.ne()[1] * ids.ne()[1]) as usize;
    let k = w.ne()[0] as usize;
    let bytes = if Q4K_UNITS.load(Ordering::Relaxed) {
        scratch_bytes(w.ty(), k, cols)
    } else {
        q8_1_column_bytes(k) * cols
    };
    (ctx.new_tensor(types::I8, &[bytes as i64]))
        .named("expert_mv_scratch")
        .as_input()
}

static Q4K_UNITS: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on give the Q4_K gate/up kernel at 3-4 tokens a step its
/// per-unit activations (`legion serve --expert-mv-q4k-units`, #816): quantize_x writes each
/// unit's ints, scales and quant sums beside the Q8_1 columns and the kernel reads them in two
/// 16-byte loads per user and unit. The same bits per op; the bigger scratch moves the graph's
/// memory layout, which can move greedy text through ggml-cuda's layout-dependent fusions
/// (identical with --deterministic --stable-fusions), so it is off by default.
pub fn set_q4k_units(on: bool) {
    Q4K_UNITS.store(on, Ordering::Relaxed);
}

/// Bytes of the activation scratch for `cols` columns of `k` against weights of type `ty`: the
/// Q8_1 columns, and for Q4_K rows also the per-unit layout the gate/up kernel reads at 3-4
/// tokens a step ([`q4k_units_at`], #816).
fn scratch_bytes(ty: ffi::ggml_type, k: usize, cols: usize) -> usize {
    let q8 = q8_1_column_bytes(k) * cols;
    if ty == types::Q4_K && k.is_multiple_of(256) {
        q4k_units_at(k, cols) + q4k_units_column_bytes(k) * cols
    } else {
        q8
    }
}

/// Where the Q4_K per-unit activations start in the scratch: after the Q8_1 columns, 16-aligned.
fn q4k_units_at(k: usize, cols: usize) -> usize {
    (q8_1_column_bytes(k) * cols).next_multiple_of(16)
}

/// Bytes of one column's Q4_K per-unit activations: for each of the k / 16 units of a row (a
/// lane's 2 x 2 ints of a 256-block, vec_dot_q4_K_q8_1's), its 4 activation ints, then (k / 16
/// x 16 bytes on) its 2 Q8_1 scales as floats and 2 quant sums.
fn q4k_units_column_bytes(k: usize) -> usize {
    k / 16 * 32
}

/// Whether a launch has quantize_x write the Q4_K per-unit activations and the gate/up kernel
/// read them: Q4_K SwiGLU at 3-4 tokens a step (the steps that kernel takes), with the scratch
/// holding them ([`scratch_bytes`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn q4k_units_for(o: &Operands) -> bool {
    o.ty == types::Q4_K
        && !o.w2.is_null()
        && (3..=4).contains(&o.n_tok)
        && o.k.is_multiple_of(256)
        && o.xq_bytes >= scratch_bytes(o.ty, o.k, o.n_x * o.n_tok)
}

/// The external param holding the SwiGLU clamp ([`glu_tiered_clamped`]), f32 bits.
pub(super) const P_LIMIT: usize = 5;
/// ... the first expert id of the stack ([`Share`]; 0 without).
const P_ID_BASE: usize = 6;
/// ... and whether skipped pairs' rows are written 0 ([`Share`]; 0: left as they are).
const P_ZERO: usize = 7;
/// ... and the source holding [`Share::map`] (0: none).
const P_MAP: usize = 8;

/// A stack that holds a share of the router's experts (`--tp-moe-ep`, #798): experts
/// `first..first + w.ne[2]`, or with `map` (`--tp-moe-ep-place`, #803) the experts whose
/// entry of `map` (I32 `[n_expert]`, the router's id to its slot in the stack, -1 for experts
/// held elsewhere) is a slot. The ops take the router's ids as they are, skip the pairs of the
/// other experts and write their rows 0, so the weighted sum over the used experts is this
/// share's part of the MoE.
#[derive(Debug, Clone, Copy)]
pub struct Share<'a> {
    pub first: usize,
    pub map: Option<Tensor<'a>>,
}

impl Share<'_> {
    fn params(&self, map_at: usize) -> [i32; P_MAP + 1] {
        let mut params = [0; P_MAP + 1];
        params[1] = overlap::NONE;
        params[P_ID_BASE] = self.first as i32;
        params[P_ZERO] = 1;
        params[P_MAP] = if self.map.is_some() { map_at as i32 } else { 0 };
        params
    }
}

/// [`mul_mat_id`] over a stack holding the router's experts `share.first..` (see [`Share`]):
/// `ids` are the router's, other experts' rows of the result are 0.
pub fn mul_mat_id_share<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    share: Share<'a>,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, w, x, ids);
    let (rows, n_used, n_tok) = (w.ne()[1], ids.ne()[0], ids.ne()[1]);
    let mut srcs = vec![w, x, ids, xq];
    srcs.extend(share.map);
    ctx.external_with(
        &EXPERT_MV,
        types::F32,
        [rows, n_used, n_tok, 1],
        &srcs,
        &share.params(4),
    )
}

/// Whether [`glu_share`] and then [`mul_mat_id_share`] take a share's routed experts `gate`,
/// `up` (`[k, n_ff, n]`) and `down` (`[n_ff, k, n]`) for `x` (`[k, 1, n_tok]`) and the router's
/// `ids`: [`glu_fits`], and the down stack [`fits`] the SwiGLU's result.
pub fn share_fits(
    gate: Tensor<'_>,
    up: Tensor<'_>,
    down: Tensor<'_>,
    x: Tensor<'_>,
    ids: Tensor<'_>,
) -> bool {
    glu_fits(gate, up, x, ids)
        && stack_fits_as(down, false)
        && down.ne()[0] == gate.ne()[1]
        && down.ne()[2] == gate.ne()[2]
}

/// [`glu_tiered`] over stacks holding the router's experts `share.first..` (see [`Share`]).
pub fn glu_share<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    share: Share<'a>,
) -> Tensor<'a> {
    let xq = q8_1_scratch(ctx, gate, x, ids);
    let more: Vec<Tensor<'a>> = share.map.into_iter().collect();
    glu_node(ctx, [gate, up], x, ids, xq, &more, &share.params(5))
}

/// The external param holding the half a split op computes ([`glu_split_clamped`],
/// [`mul_mat_id_split`]): 0 none (every other op), 1 the first, 2 the second. The last one
/// ([`crate::ggml::EXTERNAL_PARAMS`] is 10): 6 to 8 are [`Share`]'s (a staged-ahead op's
/// ring offsets, [`ahead`]).
const P_HALVES: usize = 9;

/// In a split op's ids ([`glu_split_clamped`]): the bit marking a (token, expert) pair as half
/// of its expert's work (`HALF_BIT` in the kernel).
pub const HALF_BIT: i32 = 1 << 30;

/// `--glm-pair-split` (#735): [`glu_tiered_clamped`] where the ids carrying [`HALF_BIT`] are
/// half of their expert: only rows `side * rows / 2..(side + 1) * rows / 2` of their output
/// columns are computed (the rest stay unwritten), with the same code and bits as the whole
/// expert's rows. The NVLink peer computes the other half. Every pair goes through the tile
/// kernel (VRAM ones too, [`set_vram_kernel`]'s mode 0), without [`set_prefetch`].
#[allow(clippy::too_many_arguments)]
pub fn glu_split_clamped<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    limit: f32,
    side: usize,
) -> Tensor<'a> {
    assert!(side < 2, "a pair has two halves");
    let xq = q8_1_scratch(ctx, gate, x, ids);
    glu_node(
        ctx,
        [gate, up],
        x,
        ids,
        xq,
        &[],
        &[
            vram_slots as i32,
            overlap::NONE,
            0,
            0,
            0,
            limit.to_bits() as i32,
            0,
            0,
            0,
            side as i32 + 1,
        ],
    )
}

/// The down op after [`glu_split_clamped`]: a pair with [`HALF_BIT`] is every row of its
/// expert over K's half `side` only (the SwiGLU half's values), a partial sum the NVLink peer's
/// other half completes; the others whole.
pub fn mul_mat_id_split<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    vram_slots: usize,
    side: usize,
) -> Tensor<'a> {
    assert!(side < 2, "a pair has two halves");
    let xq = q8_1_scratch(ctx, w, x, ids);
    let (rows, n_used, n_tok) = (w.ne()[1], ids.ne()[0], ids.ne()[1]);
    ctx.external_with(
        &EXPERT_MV,
        types::F32,
        [rows, n_used, n_tok, 1],
        &[w, x, ids, xq],
        &[
            vram_slots as i32,
            overlap::NONE,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            side as i32 + 1,
        ],
    )
}

/// Whether [`glu_split_clamped`] and [`mul_mat_id_split`] take `[gate, up, down]` for `x` and
/// `ids` in steps of up to `max` tokens: the Maya quants ([`maya_type`], the kernels' per-user
/// dot products), gate and up as [`glu_fits_narrow_up_to`] takes them with their halves whole
/// tiles, and down's K halves whole blocks of 8-byte multiples.
pub fn split_fits(w: [Tensor<'_>; 3], x: Tensor<'_>, ids: Tensor<'_>, max: i64) -> bool {
    let [gate, up, down] = w;
    let rows = gate.ne()[1] as usize;
    let k = down.ne()[0] as usize;
    let nb1 = down.nb()[1];
    maya_type(gate.ty())
        && maya_type(down.ty())
        && glu_fits_narrow_up_to(gate, up, x, ids, max)
        && stack_fits_as(down, true)
        && down.ne()[0] == gate.ne()[1]
        && down.ne()[2] == gate.ne()[2]
        && rows.is_multiple_of(2)
        && (rows / 2).is_multiple_of(rows_per_cta(gate.nb()[1], 2))
        && k.is_multiple_of(2 * crate::ggml::block_size(down.ty()))
        && nb1 == crate::ggml::row_size(down.ty(), k)
        && (nb1 / 2).is_multiple_of(8)
}

/// What the kernels read from a node made by [`mul_mat_id`].
struct Operands {
    ty: ffi::ggml_type,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    xq: *mut u8,
    /// The scratch's bytes ([`scratch_bytes`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    xq_bytes: usize,
    w: *const u8,
    /// [`glu_tiered`]'s up projection (null for [`mul_mat_id`]).
    w2: *const u8,
    w_nb: [usize; 4],
    x: *const u8,
    x_nb: [usize; 4],
    n_x: usize,
    ids: *const u8,
    ids_nb1: usize,
    dst: *mut u8,
    dst_nb: [usize; 4],
    k: usize,
    rows: usize,
    n_used: usize,
    n_tok: usize,
    /// Experts `w` holds (`ne[2]`): ids outside `0..n_w` are skipped (their rows of dst zeroed on
    /// the CPU, left alone on the GPU), never read out of bounds.
    n_w: usize,
    /// Ids below this are VRAM (cache slots of a tiered tensor).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    vram_below: i32,
    /// The op's role in a split ([`overlap::NONE`], [`overlap::FORK`], [`overlap::JOIN`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    split: i32,
    /// In a split: what fills the staging buffer ([`overlap::MODE_SIDE`], [`overlap::MODE_PEER`]
    /// bits).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    mode: i32,
    /// In a split: the staging buffer and its bytes, its slots, and (the SwiGLU op) the down
    /// stack whose rows it stages too.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    stage: Staging,
    /// The SwiGLU op's clamp (0: none).
    limit: f32,
    /// The router's id of the stack's first expert ([`Share`]): ids are taken minus this.
    id_base: i32,
    /// Skipped pairs' rows written 0 on the GPU too ([`Share`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    zero_skipped: bool,
    /// [`Share::map`]'s data (null without): the router's id to its slot, -1 elsewhere.
    map: *const i32,
    /// Its entries.
    n_map: usize,
    /// [`ahead`]'s roles: the pairs staged a layer ahead.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    ahead: Option<StagedAhead>,
    /// A split op's half ([`P_HALVES`]): 0 none, 1 the first, 2 the second.
    halves: i32,
    /// A verify step's row map ([`P_PAD`]'s source, I32 `[n_tok]`, `pad_rows::map`; null
    /// without): a padding row's pair whose expert no real row's pair chose is skipped, its rows
    /// 0.
    pad: *const i32,
}

/// See [`Operands::ahead`]: the slot table (`LaunchParams::stage`), the slots' offset from it,
/// a slot's bytes, the matrices' offsets in a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct StagedAhead {
    table: *const u8,
    hdr: i64,
    slot: i64,
    w: i64,
    w2: i64,
}

/// See [`Operands::stage`].
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Staging {
    /// Null for the peer alone: its device's buffer ([`peer::buffer`]).
    buf: *mut u8,
    bytes: usize,
    slots: usize,
    /// A slot's bytes: the expert's gate, up and down rows.
    slot: usize,
    down: *const u8,
    down_nb2: usize,
}

impl Operands {
    /// # Safety
    /// `dst` must be a node made by [`mul_mat_id_tiered`], [`glu_tiered`] or
    /// [`overlap::glu_down_tiered`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        unsafe {
            let (w, x, ids, xq) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
            let params = crate::ggml::external_params(dst);
            let [vram_below, split, slots, mode, slot, ..] = params;
            let src = |i: usize| (!dst.src[i].is_null()).then(|| &*dst.src[i]);
            // A staging node only with the side stream; the peer alone stages outside the graph.
            let node = mode & overlap::MODE_SIDE != 0;
            let buf = match split {
                overlap::FORK if node => src(5),
                overlap::JOIN if node => src(4),
                _ => None,
            };
            let down = (split == overlap::FORK)
                .then(|| src(if node { 6 } else { 5 }))
                .flatten();
            let stage = Staging {
                buf: buf.map_or(std::ptr::null_mut(), |b| b.data as *mut u8),
                bytes: buf.map_or(0, |b| b.ne[0] as usize),
                slots: slots.max(0) as usize,
                slot: slot.max(0) as usize,
                down: down.map_or(std::ptr::null(), |d| d.data as *const u8),
                down_nb2: down.map_or(0, |d| d.nb[2]),
            };
            let ring = match split {
                ahead::GLU => src(5),
                ahead::DOWN => src(4),
                _ => None,
            };
            let ahead = ring.map(|r| StagedAhead {
                table: (r.data as *const u8).add(params[ahead::P_TABLE] as usize),
                hdr: params[ahead::P_HDR] as i64,
                slot: params[ahead::P_SLOT] as i64,
                w: params[ahead::P_W] as i64,
                w2: params[ahead::P_W2] as i64,
            });
            // `--tp-moe-ep-place`'s id map's source ([`P_MAP`], 0: none); a staged-ahead op's
            // param 8 is [`ahead::P_W2`] instead.
            let map_at = if ahead.is_some() {
                0
            } else {
                params[P_MAP] as usize
            };
            // The row map's source ([`P_PAD`], 0: none): an op of no split, no half.
            let pad_at = if split == overlap::NONE && params[P_HALVES] == 0 {
                mode.max(0) as usize
            } else {
                0
            };
            Self {
                ty: w.type_,
                xq: xq.data as *mut u8,
                xq_bytes: xq.ne[0] as usize,
                w: w.data as *const u8,
                // A joining (or staged-ahead) down op's fifth source is the staging buffer, a
                // placed share's down op's its id map ([`Share::map`]).
                w2: if dst.src[4].is_null()
                    || split == overlap::JOIN
                    || split == ahead::DOWN
                    || map_at == 4
                    || pad_at == 4
                {
                    std::ptr::null()
                } else {
                    (*dst.src[4]).data as *const u8
                },
                w_nb: w.nb,
                x: x.data as *const u8,
                x_nb: x.nb,
                n_x: x.ne[1] as usize,
                ids: ids.data as *const u8,
                ids_nb1: ids.nb[1],
                dst: dst.data as *mut u8,
                dst_nb: dst.nb,
                k: w.ne[0] as usize,
                rows: w.ne[1] as usize,
                n_used: ids.ne[0] as usize,
                n_tok: ids.ne[1] as usize,
                n_w: w.ne[2] as usize,
                vram_below,
                split,
                mode: if pad_at == 0 { mode } else { 0 },
                stage,
                limit: f32::from_bits(params[P_LIMIT] as u32),
                // A staged-ahead op's params 6 to 8 are its ring offsets ([`ahead::P_HDR`],
                // [`ahead::P_W`], [`ahead::P_W2`]), not a [`Share`]'s: it never takes a share.
                id_base: if ahead.is_some() {
                    0
                } else {
                    params[P_ID_BASE]
                },
                zero_skipped: ahead.is_none() && params[P_ZERO] != 0,
                map: match map_at {
                    0 => std::ptr::null(),
                    at => (*dst.src[at]).data as *const i32,
                },
                n_map: match map_at {
                    0 => 0,
                    at => (*dst.src[at]).ne[0] as usize,
                },
                ahead,
                halves: params[P_HALVES],
                pad: match pad_at {
                    0 => std::ptr::null(),
                    at => (*dst.src[at]).data as *const i32,
                },
            }
        }
    }
}

/// Whether a stage graph (`--stage-graphs`, #817) may capture this op's CUDA launch: not one of
/// a split pair (its staging waits on host copies, the peer's or the side stream's), and not
/// with the PTX kernel on the side stream (a fork and join of streams). Otherwise the launch is
/// kernels on the compute stream chosen from the node and the startup switches alone.
///
/// # Safety
/// `dst` must be a node made by [`mul_mat_id_tiered`], [`glu_tiered`] or
/// [`overlap::glu_down_tiered`].
pub unsafe fn capture_safe(dst: &ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    let o = unsafe { Operands::of(dst) };
    if matches!(o.split, overlap::FORK | overlap::JOIN) {
        return false;
    }
    #[cfg(feature = "cuda")]
    if ptx::side() && ptx::params_for(&o).is_some() {
        return false;
    }
    true
}

/// Whether token `t`'s pair of expert `raw` is a dead padding pair of a verify step: `t` a
/// padding row of `o.pad` (non-null; `pad[t] != t`) and no real row's pair chose `raw`
/// (`expert_of` in the kernels).
///
/// # Safety
/// `o.pad` holds `o.n_tok` entries and `o.ids` the step's ids.
unsafe fn pad_pair_dead(o: &Operands, t: usize, raw: i32) -> bool {
    unsafe {
        let real = |r: usize| *o.pad.add(r) == r as i32;
        if real(t) {
            return false;
        }
        !(0..o.n_tok).filter(|&r| real(r)).any(|r| {
            (0..o.n_used).any(|j| *(o.ids.add(r * o.ids_nb1 + j * 4) as *const i32) == raw)
        })
    }
}

unsafe fn cpu_expert_mv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id`, all tensors in host memory.
    let o = unsafe { Operands::of(dst) };
    for pair in (ith..o.n_used * o.n_tok).step_by(nth) {
        let (t, j) = (pair / o.n_used, pair % o.n_used);
        // SAFETY: in-bounds reads of ids, x and the expert's rows; writes to dst's row.
        unsafe {
            let raw = *(o.ids.add(t * o.ids_nb1 + j * 4) as *const i32);
            let e = if o.map.is_null() {
                raw.wrapping_sub(o.id_base)
            } else {
                usize::try_from(raw)
                    .ok()
                    .filter(|&r| r < o.n_map)
                    .map_or(-1, |r| *o.map.add(r))
            };
            let out = o.dst.add(t * o.dst_nb[2] + j * o.dst_nb[1]) as *mut f32;
            // A split op's half pair ([`HALF_BIT`]): the rows (SwiGLU) or the K values (down)
            // of half `o.halves - 1` only; the other rows zero here.
            let half = o.halves != 0 && e >= 0 && e & HALF_BIT != 0;
            let e = if half { e & !HALF_BIT } else { e };
            // A dead padding pair ([`Operands::pad`]): skipped, its rows 0, as the kernels do.
            let e = if !o.pad.is_null()
                && !super::pad_rows::skip_layout_only()
                && pad_pair_dead(&o, t, raw)
            {
                -1
            } else {
                e
            };
            let Some(e) = usize::try_from(e).ok().filter(|&e| e < o.n_w) else {
                std::slice::from_raw_parts_mut(out, o.rows).fill(0.0);
                continue;
            };
            let side = (o.halves.max(1) - 1) as usize;
            let (k0, kn) = if half && o.w2.is_null() {
                (side * o.k / 2, o.k / 2)
            } else {
                (0, o.k)
            };
            let rows = if half && !o.w2.is_null() {
                side * o.rows / 2..(side + 1) * o.rows / 2
            } else {
                0..o.rows
            };
            let xj = if o.n_x == 1 { 0 } else { j };
            let x = std::slice::from_raw_parts(
                o.x.add(t * o.x_nb[2] + xj * o.x_nb[1]) as *const f32,
                o.k,
            );
            let x = &x[k0..k0 + kn];
            let (skip, take) = (
                crate::ggml::row_size(o.ty, k0),
                crate::ggml::row_size(o.ty, kn),
            );
            let dot = |m: *const u8, r: usize| -> f32 {
                let row =
                    std::slice::from_raw_parts(m.add(e * o.w_nb[2] + r * o.w_nb[1] + skip), take);
                let w = crate::ggml::dequantize(o.ty, row, kn);
                w.iter().zip(x).map(|(a, b)| a * b).sum()
            };
            for r in 0..o.rows {
                if !rows.contains(&r) {
                    *out.add(r) = 0.0;
                    continue;
                }
                let s = if o.w2.is_null() {
                    dot(o.w, r)
                } else {
                    let (mut g, mut u) = (dot(o.w, r), dot(o.w2, r));
                    if o.limit > 0.0 {
                        g = g.min(o.limit);
                        u = u.clamp(-o.limit, o.limit);
                    }
                    g / (1.0 + (-g).exp()) * u
                };
                *out.add(r) = s;
            }
        }
    }
}

/// Bytes of a tile's loads in flight per CTA: 128 threads x 16 loads x 16 bytes
/// (`MAX_LOADS` in the kernel).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const TILE_BYTES: usize = 16 * 1024;

/// A tile size other than [`TILE_BYTES`] for launches from now on (0: the default); for tuning.
pub fn set_tile_bytes(bytes: usize) {
    TILE_OVERRIDE.store(bytes, Ordering::Relaxed);
}

static TILE_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// How VRAM experts (a tiered tensor's cache slots) are read: 0 (the default) through tiles as
/// host experts are, 1 by a kernel of their own reading them in place as ggml's mmvq does, 2 the
/// same for the gate/up (SwiGLU) op only (the down matrix keeps the tiles, where that kernel is
/// slower). Not byte-identical to 0 (mmvq's rounding), so opt-in: `--expert-mv-vram`. A mode
/// other than 0 sums VRAM and host experts in different orders, so with the cache's
/// timing-dependent residency greedy output may not reproduce from run to run.
pub fn set_vram_kernel(mode: usize) {
    VRAM_KERNEL.store(mode, Ordering::Relaxed);
}

static VRAM_KERNEL: AtomicUsize = AtomicUsize::new(0);

/// How the tile kernel computes its dot products (`legion serve --expert-mv-dot`). Every mode
/// gives the same bits: each (row, user) sum takes the same units in the same order and meets
/// in the same addition tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Dot {
    /// Where `once` was faster on V100: Q4_K and Q5_K SwiGLU pairs at every step size, Q5_1
    /// rows in steps of 3 tokens or more, Q8_0 rows in steps of 3 and 4. Elsewhere `per-user`.
    #[default]
    Auto,
    /// One ggml vec_dot per user, as ggml's mmvq calls it (the kernel before #288).
    PerUser,
    /// Each weight unit loaded and decoded once for every user, each user's activations once
    /// for both SwiGLU matrices (adapted from upstream Strata's decoded-once tables).
    Once,
    /// `once` with rows of 32 lanes computed by 16, two rows per warp (upstream Strata's
    /// 16-lane sub-warp kernels). Slower on V100.
    Sub16,
}

impl Dot {
    /// The kernel's `DOT_*` value.
    pub fn code(self) -> usize {
        match self {
            Dot::Auto => 0,
            Dot::PerUser => 1,
            Dot::Once => 2,
            Dot::Sub16 => 3,
        }
    }
}

/// The tile kernel's dot products for launches from now on.
pub fn set_dot(dot: Dot) {
    DOT.store(dot.code(), Ordering::Relaxed);
}

static DOT: AtomicUsize = AtomicUsize::new(0);

/// Whether the tile kernel reads experts in VRAM (cache slots with [`set_vram_kernel`] 0, and
/// staged copies) in place with each lane's weight loads issued up front, instead of through its
/// shared-memory tile (`--expert-mv-prefetch`; after Strata-AC922's S28 kernels). Only experts
/// one (token, slot) pair uses, not Q8_0 weights, and steps of at most 2 tokens; the others keep
/// the tile. The same lanes
/// and summation order as the tile path, so the same output bits; launches from now on.
pub fn set_prefetch(on: bool) {
    PREFETCH.store(on, Ordering::Relaxed);
}

/// See [`set_prefetch`].
pub fn prefetch() -> bool {
    PREFETCH.load(Ordering::Relaxed)
}

static PREFETCH: AtomicBool = AtomicBool::new(false);

/// Tiles of one expert a CTA of the tile kernel takes, software pipelined (`legion serve
/// --expert-mv-pipe`, #816): the next tile's loads are in flight while the current one is
/// computed from shared memory, so HBM no longer idles while an SM's CTAs all compute at once.
/// 0: the tile kernel as before (a tile per CTA, loaded then computed); `None` (the default):
/// [`auto_pipe_tiles`]. Every tile is computed by the same code from the same shared-memory
/// layout: the same output bits whatever the value. Not with [`set_prefetch`] (whose steps keep
/// the in-place kernel); launches from now on.
pub fn set_pipe_tiles(tiles: Option<usize>) {
    PIPE_TILES.store(tiles.unwrap_or(PIPE_AUTO), Ordering::Relaxed);
}

/// The tiles per CTA of an op on `ty` (gate and up through the SwiGLU op if `glu`) at `n_tok`
/// tokens a step; see [`set_pipe_tiles`].
pub fn pipe_tiles(ty: ffi::ggml_type, glu: bool, n_tok: usize) -> usize {
    match PIPE_TILES.load(Ordering::Relaxed) {
        PIPE_AUTO if n_tok == 1 && half_tiles_for(ty, glu, n_tok) => HALF_PIPE_ONE,
        PIPE_AUTO => auto_pipe_tiles(ty, glu, n_tok),
        n => n,
    }
}

/// The default tiles per CTA (`legion expert-mv-ptx-bench` on a V100, #816): Qwen3.8-Flash-Next's
/// gate/up Q4_K 1 at one token a step (0.85x the tile kernel's time), 4 at 2-4 (0.79-0.91x);
/// its down Q5_1 1 at one token (0.89x), 4 at 2-4 (0.78-0.93x); its MTP layer's gate, up and
/// down Q8_0 4 at 1-3 (0.71-0.86x). 0 (the tile kernel) elsewhere: over 4 tokens (the kernels
/// compiled for 8 users spill), Q8_0 at 4, and the types not measured.
pub fn auto_pipe_tiles(ty: ffi::ggml_type, glu: bool, n_tok: usize) -> usize {
    match (ty, glu, n_tok) {
        (types::Q4_K, true, 1) | (types::Q5_1, false, 1) => 1,
        (types::Q4_K, true, 2..=4) | (types::Q5_1, false, 2..=4) | (types::Q8_0, false, 1..=3) => 4,
        _ => 0,
    }
}

/// [`PIPE_TILES`]: [`auto_pipe_tiles`].
const PIPE_AUTO: usize = usize::MAX;

static PIPE_TILES: AtomicUsize = AtomicUsize::new(PIPE_AUTO);

/// Link worker CTAs by default ([`set_link_ctas`]): none. 64 (tiles of 11.5-15 KB in flight,
/// about three times what a V100's NVLink2 read needs, `stage_copy` filling it from 256 KB) was
/// measured on 2 V100s with MTP (#710): greedy chats about 5% slower, so it stays an opt-in.
/// Their text changed too, which the same tile code should not cause; MTP's adaptive depth
/// follows timing (#712, #780), the likely cause, but no per-op GPU check proves it.
pub const LINK_CTAS: usize = 0;

/// How many CTAs of the tile kernel read the host experts of a tiered op (an expert cache's
/// misses read in place over the link, not staged) for launches from now on; 0 (the default)
/// turns it off (`legion serve --expert-mv-link-ctas`, #710). Those CTAs come first in the grid
/// and go over the host pairs' tiles; the other CTAs take the VRAM pairs only. Each tile goes
/// through the same code as before; only when the link reads start and how many SM slots they
/// hold changes. Experimental: greedy text differed from the default's in the #710 bench.
/// Kernels without a decoded-once form (GLM-5.3-Flash's Maya quants), `--expert-mv-prefetch`
/// launches and ops without host experts keep the plain grid.
pub fn set_link_ctas(n: usize) {
    LINK_CTAS_SET.store(n.min(1 << 16), Ordering::Relaxed);
}

/// See [`set_link_ctas`].
pub fn link_ctas() -> usize {
    LINK_CTAS_SET.load(Ordering::Relaxed)
}

static LINK_CTAS_SET: AtomicUsize = AtomicUsize::new(LINK_CTAS);

/// The link workers a launch over `vram_below` cache slots of `n_w` experts gets: [`link_ctas`]
/// when the op has host experts beside VRAM ones and the same tile launch takes both (`part` 0,
/// or 3 of a split: all but the staged experts; `vram_kernel` 0: the VRAM experts not handed to
/// another kernel; not with `prefetch`), else 0. A launch of host experts alone keeps all its
/// CTAs on them.
pub fn link_ctas_for(
    vram_below: i32,
    n_w: usize,
    part: i32,
    vram_kernel: i32,
    prefetch: bool,
) -> i32 {
    let mixed = vram_below > 0 && (vram_below as usize) < n_w;
    if mixed && !prefetch && vram_kernel == 0 && (part == 0 || part == 3) {
        link_ctas() as i32
    } else {
        0
    }
}
/// Whether the pipelined tile kernel for Q4_K gate and up at 3-4 tokens a step keeps each
/// lane's activations of its first units in registers over the CTA's tiles instead of
/// reloading them per tile (#816 rounds 6 and 8; on by default, off with `legion serve
/// --no-expert-mv-q4k-hold`). The same values in the same sums, so the same output bits;
/// launches from now on. Not with the per-unit layout ([`set_q4k_units`]), which takes precedence.
pub fn set_q4k_hold(on: bool) {
    Q4K_HOLD.store(on, Ordering::Relaxed);
}

/// See [`set_q4k_hold`].
pub fn q4k_hold() -> bool {
    Q4K_HOLD.load(Ordering::Relaxed)
}

static Q4K_HOLD: AtomicBool = AtomicBool::new(true);

/// Whether the pipelined tile kernel's Q4_K gate/up at one token spreads the step's tiles evenly
/// over as many CTAs as the GPU holds at once (`expert_mv_tile_bal`, #814; on by default, off with
/// `legion serve --no-expert-mv-balance`) instead of a CTA per tile, so each CTA has its next
/// tile's loads in flight while it computes the current one, across expert boundaries too (V100:
/// 48.3 -> 42.5 us in `legion expert-mv-ptx-bench`). The other ops and step sizes keep their
/// groups of [`pipe_tiles`] tiles (balanced, they measured 5-50% slower). Each tile is computed by
/// the same code from the same shared-memory layout, so the same output bits; launches from now on.
pub fn set_balance(on: bool) {
    BALANCE.store(on, Ordering::Relaxed);
}

/// See [`set_balance`].
pub fn balance() -> bool {
    BALANCE.load(Ordering::Relaxed)
}

static BALANCE: AtomicBool = AtomicBool::new(true);

/// Whether the pipelined tile kernel's Q5_1 down projection at 1-3 tokens a step takes tiles of
/// half the rows (16 of Qwen3.8-Flash-Next's 480-byte rows, not 32) in twice as many CTAs
/// (#814; on by default, off with `legion serve --no-expert-mv-half-tiles`): at 2-3 tokens each
/// CTA keeps its chain of [`pipe_tiles`] tiles (4 by default), at one token the default chain is
/// [`HALF_PIPE_ONE`] half tiles instead of one full one. V100 (`legion expert-mv-ptx-bench`,
/// alternating with main): 34.8 -> 30.8 us at one token, 50 -> 40 at 2, 51 -> 49 at 3; 4 tokens
/// and the Q8_0 down got slower, so they keep the full tiles. Every row is computed by the same
/// lanes from the same shared-memory layout, so the same output bits; launches from now on.
pub fn set_half_tiles(on: bool) {
    HALF_TILES.store(on, Ordering::Relaxed);
}

/// See [`set_half_tiles`].
pub fn half_tiles() -> bool {
    HALF_TILES.load(Ordering::Relaxed)
}

static HALF_TILES: AtomicBool = AtomicBool::new(true);

/// The default half tiles a CTA takes at one token ([`set_half_tiles`]; the V100 bench: 2 32.1
/// us, 4 34.1, 8 30.7, the full tiles' best 34.3).
pub const HALF_PIPE_ONE: usize = 8;

/// The ops [`set_half_tiles`] takes: Q5_1 single-matrix ops (the down projection) at 1-3 tokens.
pub fn half_tiles_at(ty: ffi::ggml_type, glu: bool, n_tok: usize) -> bool {
    ty == types::Q5_1 && !glu && (1..=3).contains(&n_tok)
}

/// [`half_tiles_at`] with [`set_half_tiles`] on.
fn half_tiles_for(ty: ffi::ggml_type, glu: bool, n_tok: usize) -> bool {
    half_tiles() && half_tiles_at(ty, glu, n_tok)
}

/// Fewest rows of a tile: with rows of a multiple of 4 bytes and experts a multiple of 16 bytes
/// apart ([`fits`]), every tile starts 16-byte aligned and spans a multiple of 16 bytes.
const MIN_ROWS: usize = 4;

/// Rows of a tile where [`MIN_ROWS`] do not fit: only shapes [`glu_fits_narrow`] lets in.
const NARROW_ROWS: usize = 2;

/// Rows per CTA for `mats` matrices with rows of `row_bytes`: the most of 32, 16, 8, 4 whose
/// tile (each matrix's rows 16-byte aligned) fits [`TILE_BYTES`] (Q4_K 2560-wide rows: 8, or 4
/// for gate and up together; Q5_1 640-wide: 32).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn rows_per_cta(row_bytes: usize, mats: usize) -> usize {
    let tile = match TILE_OVERRIDE.load(Ordering::Relaxed) {
        0 => TILE_BYTES,
        t => t.min(TILE_BYTES),
    };
    [32, 16, 8, MIN_ROWS, NARROW_ROWS]
        .into_iter()
        .find(|r| mats * (r * row_bytes).next_multiple_of(16) <= tile)
        .unwrap_or(MIN_ROWS)
}

/// Bytes of one Q8_1 activation column of `k` elements (`block_q8_1`: 32 values + 2 halves).
fn q8_1_column_bytes(k: usize) -> usize {
    k / 32 * 36
}

/// What the kernels assume of a launch, checked on the host: a violation is an error here
/// rather than a misaligned or out-of-bounds access on the GPU (an Xid 13 or 31 that can take
/// the device down with it). [`fits`] keeps graphs within these; this guards every launch.
/// - `expert_mv_tile` loads each tile in 16-byte pieces from `w + e * nb2 + r0 * nb1` (and
///   `w2`'s): both bases 16-aligned, experts 16 bytes apart, rows 4 bytes apart (so a tile of
///   `rows_per_cta` rows, and the expert's last partial one, are 16-byte multiples).
/// - Both matrices' tiles together fit the loads the kernel keeps in flight ([`TILE_BYTES`]):
///   a bigger tile would be computed from shared memory never loaded.
/// - `quantize_x` gives a warp a 32-value block: K a multiple of 32; X rows of floats.
/// - The IQ types' activation quants are read 16 bytes at a time: the scratch 16-byte aligned
///   (a K multiple of 32 keeps every column so).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn launch_check(o: &Operands, rows_per_cta: usize, smem: usize) -> Result<(), String> {
    let a16 = |p: *const u8| (p as usize).is_multiple_of(16);
    let [_, nb1, nb2, _] = o.w_nb;
    let problem = if !a16(o.w) || !(o.w2.is_null() || a16(o.w2)) {
        "weights not 16-byte aligned"
    } else if !nb2.is_multiple_of(16) || !nb1.is_multiple_of(4) {
        "expert or row stride breaks 16-byte tile loads"
    } else if !(rows_per_cta * nb1).is_multiple_of(16) || !(o.rows * nb1).is_multiple_of(16) {
        "a tile is not a whole number of 16-byte loads"
    } else if smem > TILE_BYTES {
        "tile larger than the loads the kernel keeps in flight"
    } else if o.k == 0 || !o.k.is_multiple_of(32) || o.x_nb[0] != 4 {
        "K not a multiple of 32, or X not rows of floats"
    } else if !a16(o.xq) {
        "activation scratch not 16-byte aligned"
    } else if o.n_w == 0 || o.n_w > i32::MAX as usize {
        "expert count out of range"
    } else {
        return Ok(());
    };
    Err(format!(
        "expert_mv launch refused: {problem} (w {:p}, w2 {:p}, nb1 {nb1}, nb2 {nb2}, rows {}, \
         rows per tile {rows_per_cta}, k {})",
        o.w, o.w2, o.rows, o.k
    ))
}

/// What `expert_mv_split` assumes of a split op ([`glu_split_clamped`], [`mul_mat_id_split`])
/// beyond [`launch_check`]: no split pair of ops around it, the half 1 or 2, a type with the
/// per-user dot products only ([`maya_type`]); with SwiGLU each half whole tiles of rows, else
/// K halves of whole blocks whose bytes are a multiple of 8 (its 8-byte loads), the rows packed.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn split_check(o: &Operands, rows_per_cta: usize, mats: usize) -> Result<(), String> {
    let nb1 = o.w_nb[1];
    let problem = if o.split != overlap::NONE || !(1..=2).contains(&o.halves) {
        "not a plain op of one half"
    } else if !maya_type(o.ty) {
        "a type with decoded-once dot products"
    } else if mats == 2 && !(o.rows.is_multiple_of(2) && (o.rows / 2).is_multiple_of(rows_per_cta))
    {
        "the SwiGLU rows' halves are not whole tiles"
    } else if mats == 1
        && !(o.k.is_multiple_of(2 * crate::ggml::block_size(o.ty))
            && nb1 == crate::ggml::row_size(o.ty, o.k)
            && (nb1 / 2).is_multiple_of(8))
    {
        "K's halves are not whole blocks of 8-byte multiples"
    } else {
        return Ok(());
    };
    Err(format!(
        "expert_mv split launch refused: {problem} (half {}, rows {}, k {}, nb1 {nb1}, rows per \
         tile {rows_per_cta})",
        o.halves, o.rows, o.k
    ))
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchParams {
    w: *const u8,
    w2: *const u8,
    w_nb1: i64,
    w_nb2: i64,
    x: *const f32,
    x_nb1: i64,
    x_nb2: i64,
    n_x: i32,
    ids: *const i32,
    ids_nb1: i64,
    dst: *mut f32,
    dst_nb1: i64,
    dst_nb2: i64,
    k: i32,
    rows: i32,
    n_used: i32,
    n_tok: i32,
    rows_per_cta: i32,
    xq: *mut u8,
    vram_below: i32,
    vram_kernel: i32,
    n_w: i32,
    /// The pairs this launch computes: 0 all, 1 VRAM ones, 2 host ones.
    part: i32,
    /// 1: quantize the activations, 2: the expert kernels, 3: both.
    steps: i32,
    /// Non-null: host pairs with a staging slot read their expert there (see the kernel).
    stage: *const u8,
    stage_hdr: i64,
    stage_slot: i64,
    stage_w: i64,
    stage_w2: i64,
    /// The tile kernel's dot products ([`set_dot`]).
    dot: i32,
    /// Non-zero: VRAM experts read in place, loads up front ([`set_prefetch`]).
    vram_pf: i32,
    /// The SwiGLU clamp (0: none).
    limit: f32,
    /// The router's id of the stack's first expert ([`Share`]).
    id_base: i32,
    /// Non-zero: skipped pairs' rows of dst written 0 ([`Share`]).
    zero_skipped: i32,
    /// > 0: tiles per CTA, pipelined ([`set_pipe_tiles`]).
    pipe_tiles: i32,
    /// Non-zero: the scratch holds the Q4_K per-unit activations ([`q4k_units_for`]) at this
    /// byte offset.
    q4k_units: i64,
    /// Non-null: the router's id to its slot, -1 for experts held elsewhere ([`Share::map`];
    /// `id_base` is 0 then), `n_map` entries.
    map: *const i32,
    n_map: i32,
    /// Link worker CTAs ([`link_ctas_for`]).
    link_ctas: i32,
    /// A split op's half ([`P_HALVES`]): 0 none, 1 or 2 (the kernel's `expert_mv_split`).
    halves: i32,
    /// Non-zero: Q4_K gate/up activations held in registers ([`set_q4k_hold`]).
    q4k_hold: i32,
    /// Non-zero: the piped tiles balanced over the resident CTAs ([`set_balance`]).
    balance: i32,
    /// Non-zero: the piped Q5_1 down on tiles of half the rows ([`set_half_tiles`]).
    half_tiles: i32,
    /// Non-null: a verify step's row map ([`Operands::pad`]); its dead padding pairs skipped.
    pad_rows: *const i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_expert_mv_launch(
        ty: i32,
        p: *const LaunchParams,
        smem: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_expert_mv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: the caller's.
    unsafe { cuda_expert_mv_as(dst, stream, false) }
}

/// [`cuda_expert_mv`] of a [`glu_tiered_preq`] node: its activations' Q8_1 blocks are there.
#[cfg(feature = "cuda")]
unsafe fn cuda_expert_mv_preq(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: the caller's.
    unsafe { cuda_expert_mv_as(dst, stream, true) }
}

/// The launches of an expert op; `preq`: its `xq` source holds the activations' Q8_1 blocks
/// already ([`glu_tiered_preq`]), so no `quantize_x`.
#[cfg(feature = "cuda")]
unsafe fn cuda_expert_mv_as(
    dst: &ffi::ggml_tensor,
    stream: *mut c_void,
    preq: bool,
) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat_id`; weights in memory the GPU can read.
    let o = unsafe { Operands::of(dst) };
    let mats = if o.w2.is_null() { 1 } else { 2 };
    let rows_per_cta = rows_per_cta(o.w_nb[1], mats);
    let smem = mats * (rows_per_cta * o.w_nb[1]).next_multiple_of(16);
    launch_check(&o, rows_per_cta, smem)?;
    let p = LaunchParams {
        w: o.w,
        w2: o.w2,
        w_nb1: o.w_nb[1] as i64,
        w_nb2: o.w_nb[2] as i64,
        x: o.x as *const f32,
        x_nb1: o.x_nb[1] as i64,
        x_nb2: o.x_nb[2] as i64,
        n_x: o.n_x as i32,
        ids: o.ids as *const i32,
        ids_nb1: o.ids_nb1 as i64,
        dst: o.dst as *mut f32,
        dst_nb1: o.dst_nb[1] as i64,
        dst_nb2: o.dst_nb[2] as i64,
        k: o.k as i32,
        rows: o.rows as i32,
        n_used: o.n_used as i32,
        n_tok: o.n_tok as i32,
        rows_per_cta: rows_per_cta as i32,
        xq: o.xq,
        vram_below: o.vram_below,
        vram_kernel: match VRAM_KERNEL.load(Ordering::Relaxed) {
            0 => 0,
            2 => i32::from(mats == 2),
            _ => 1,
        },
        n_w: o.n_w as i32,
        part: 0,
        steps: if preq { 2 } else { 3 },
        // The pairs staged a layer ahead, read from VRAM by the same code ([`ahead`]).
        stage: o.ahead.map_or(std::ptr::null(), |a| a.table),
        stage_hdr: o.ahead.map_or(0, |a| a.hdr),
        stage_slot: o.ahead.map_or(0, |a| a.slot),
        stage_w: o.ahead.map_or(0, |a| a.w),
        stage_w2: o.ahead.map_or(0, |a| a.w2),
        dot: DOT.load(Ordering::Relaxed) as i32,
        vram_pf: i32::from(prefetch()),
        limit: o.limit,
        id_base: o.id_base,
        zero_skipped: i32::from(o.zero_skipped),
        pipe_tiles: pipe_tiles(o.ty, mats == 2, o.n_tok).min(i32::MAX as usize) as i32,
        q4k_units: if !preq && q4k_units_for(&o) {
            q4k_units_at(o.k, o.n_x * o.n_tok) as i64
        } else {
            0
        },
        map: o.map,
        n_map: o.n_map as i32,
        link_ctas: 0,
        halves: 0,
        q4k_hold: i32::from(q4k_hold()),
        balance: i32::from(balance()),
        half_tiles: i32::from(half_tiles_for(o.ty, mats == 2, o.n_tok)),
        pad_rows: std::ptr::null(),
    };
    if o.halves != 0 {
        // A split op (`--glm-pair-split`): every pair through `expert_mv_split`'s tiles, in one
        // launch on this stream.
        split_check(&o, rows_per_cta, mats)?;
        let p = LaunchParams {
            vram_kernel: 0,
            vram_pf: 0,
            halves: o.halves,
            ..p
        };
        // SAFETY: operands checked above; `stream` is the compute stream.
        return match unsafe { legion_expert_mv_launch(o.ty as i32, &p, smem as i32, stream) } {
            0 => Ok(()),
            e => Err(format!(
                "CUDA error {e} (split, {} k {} rows {}, {} x {} pairs)",
                crate::ggml::type_name(o.ty),
                o.k,
                o.rows,
                o.n_used,
                o.n_tok
            )),
        };
    }
    // `--expert-mv-ptx`: the VRAM pairs through the PTX kernels (vram_kernel 2 hands them over).
    let pp = ptx::params_for(&o);
    // A verify step's dead padding pairs skipped (`--moe-pad-skip-tiered`): only where the tile
    // kernels compute every pair (not the PTX or VRAM kernels, which do not know of them).
    let pad = !o.pad.is_null()
        && pp.is_none()
        && p.vram_kernel == 0
        && !super::pad_rows::skip_layout_only();
    let p = LaunchParams {
        vram_kernel: if pp.is_some() { 2 } else { p.vram_kernel },
        pad_rows: if pad { o.pad } else { std::ptr::null() },
        zero_skipped: if pad { 1 } else { p.zero_skipped },
        ..p
    };
    let nvcc = |p: LaunchParams, stream: *mut c_void| {
        let p = LaunchParams {
            link_ctas: link_ctas_for(o.vram_below, o.n_w, p.part, p.vram_kernel, p.vram_pf != 0),
            ..p
        };
        // SAFETY: operands checked above; `stream` is the compute stream or its side stream.
        match unsafe { legion_expert_mv_launch(o.ty as i32, &p, smem as i32, stream) } {
            0 => Ok(()),
            e => Err(format!(
                "CUDA error {e} ({} k {} rows {}, {} x {} pairs)",
                crate::ggml::type_name(o.ty),
                o.k,
                o.rows,
                o.n_used,
                o.n_tok
            )),
        }
    };
    let launch = |p: LaunchParams, stream: *mut c_void| {
        nvcc(p, stream)?;
        match pp {
            // SAFETY: as above; the activations were quantized on this stream (steps 1 of this
            // launch or an earlier one).
            Some(pp) if p.steps & 2 != 0 => unsafe { ptx::launch_part(&o, pp, p.part, stream) },
            _ => Ok(()),
        }
    };
    match o.split {
        overlap::FORK | overlap::JOIN => {
            // SAFETY: the compute stream ggml hands the op, on the current device; the staging
            // buffer (with the side stream; the peer alone: the device's own) and the down stack
            // are this pair of ops' sources.
            unsafe { split_launch(&o, p, launch, stream) }
        }
        _ => match pp {
            // The PTX kernel on the side stream, beside the tile kernel's host pairs: the
            // activations first, then the side stream forks, the tile kernel goes on the
            // compute stream and the PTX kernel on the side one, and the compute stream waits.
            // SAFETY: as above; the side stream reads what the compute stream had queued when
            // it forked (the activations), and the compute stream waits for it before the op
            // ends.
            Some(pp) if ptx::side() => unsafe {
                if !preq {
                    nvcc(LaunchParams { steps: 1, ..p }, stream)?;
                }
                let side = overlap::fork(stream)?;
                nvcc(LaunchParams { steps: 2, ..p }, stream)?;
                ptx::launch_part(&o, pp, p.part, side)?;
                overlap::mark(stream, 2)?;
                overlap::wait(stream, 2)
            },
            _ => launch(p, stream),
        },
    }
}

/// One op of a split pair (see [`overlap`] and [`peer`]): the cached experts (and with the peer
/// alone the missed ones it does not stage), the wait for the staged experts, then those.
///
/// # Safety
/// As [`cuda_expert_mv`].
#[cfg(feature = "cuda")]
unsafe fn split_launch(
    o: &Operands,
    p: LaunchParams,
    launch: impl Fn(LaunchParams, *mut c_void) -> Result<(), String>,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut st = o.stage;
    let n_pairs = o.n_used * o.n_tok;
    let hdr = overlap::stage_header(n_pairs);
    let nb2 = o.w_nb[2];
    let fork = o.split == overlap::FORK;
    let side = o.mode & overlap::MODE_SIDE != 0;
    let with_peer = o.mode & overlap::MODE_PEER != 0;
    // Slots of the gate, up and down bytes (the SwiGLU op's w, w2; the down op's w).
    let slot = st.slot;
    // The peer, if it helps this pair: the SwiGLU op picks it, the down op follows.
    let link = match (with_peer, fork) {
        (false, _) => None,
        (true, true) => peer::link([o.w, o.w2, st.down])?,
        (true, false) => peer::forked(),
    };
    if !side {
        let Some(l) = link else {
            // No idle peer: the pair of ops as without the switch.
            return launch(p, stream);
        };
        st.bytes = hdr + st.slots * slot;
        // SAFETY: `stream` is the current device's, the link's; only this stream's ops use it.
        st.buf = unsafe { peer::buffer(stream, l, st.bytes)? };
    }
    let ok = !st.buf.is_null()
        && st.slots > 0
        && st.bytes >= hdr + st.slots * slot
        && (!fork || !(st.down.is_null() || o.w2.is_null()))
        && (!fork || slot == 2 * nb2 + st.down_nb2)
        && slot >= nb2
        && (st.buf as usize).is_multiple_of(16)
        && slot.is_multiple_of(16)
        && n_pairs <= 512
        && o.mode & (overlap::MODE_SIDE | overlap::MODE_PEER) != 0;
    if !ok {
        return Err(format!(
            "expert_mv split refused: staging buffer {:p} of {} bytes, {} slots of {slot}, \
             mode {}",
            st.buf, st.bytes, st.slots, o.mode
        ));
    }
    let staged = LaunchParams {
        part: if side { 2 } else { 4 },
        steps: 2,
        stage: st.buf,
        stage_hdr: hdr as i64,
        stage_slot: slot as i64,
        // FORK: gate at 0, up after it; JOIN: down last.
        stage_w: if fork { 0 } else { (slot - nb2) as i64 },
        stage_w2: nb2 as i64,
        ..p
    };
    // With the side stream: the VRAM experts, then every host one (staged or in place). The
    // peer alone: all but its staged experts, then those.
    let first = if side {
        LaunchParams {
            part: 1,
            steps: 2,
            ..p
        }
    } else {
        LaunchParams { part: 3, ..staged }
    };
    if fork {
        // The peer alone takes its share of the missed experts' ranks and fills every slot;
        // beside the side stream every rank takes a slot and the peer fills the odd ones.
        let (every, take) = peer::share();
        let ranks = if side { (1, 1) } else { (every, take) };
        let gate_up = overlap::StageParams {
            ids: o.ids as *const i32,
            ids_nb1: o.ids_nb1 as i64,
            n_used: o.n_used as i32,
            n_tok: o.n_tok as i32,
            vram_below: o.vram_below,
            n_w: o.n_w as i32,
            stage: st.buf,
            hdr: hdr as i64,
            slot: slot as i64,
            cap: st.slots as i32,
            table: 0,
            n_seg: 2,
            src: [o.w, o.w2, std::ptr::null()],
            nb2: [nb2 as i64, nb2 as i64, 0],
            off: [0, nb2 as i64, 0],
            ctas: peer::ctas(),
            t_every: ranks.0,
            t_take: ranks.1,
            c_every: if side { 2 } else { 1 },
            c_take: 1,
            c_flip: i32::from(side),
        };
        let down = overlap::StageParams {
            n_seg: 1,
            src: [st.down, std::ptr::null(), std::ptr::null()],
            nb2: [st.down_nb2 as i64, 0, 0],
            off: [2 * nb2 as i64, 0, 0],
            ..gate_up
        };
        if let Some(l) = link {
            // SAFETY: as the caller promises; the peer reaches the ids and the buffer (this
            // device's memory) and the stacks' host parts (mapped for it when it was connected).
            unsafe { peer::fork(stream, l, &gate_up, &down)? };
        }
        // The activations, all columns.
        launch(LaunchParams { steps: 1, ..p }, stream)?;
        let mine = overlap::StageParams {
            table: 1,
            ctas: overlap::ctas(),
            c_flip: 0,
            ..gate_up
        };
        if side {
            // SAFETY: as the caller promises.
            let s = unsafe { overlap::fork(stream)? };
            let mine = mine.rule(if link.is_some() {
                (1, 1, 2, 1, 0)
            } else {
                overlap::StageParams::ALL
            });
            // SAFETY: the stacks hold `n_w` experts `nb2` apart; the buffer the slots and table.
            unsafe {
                overlap::stage(&mine, s)?;
                overlap::mark(stream, 0)?;
                let down = overlap::StageParams {
                    table: 0,
                    n_seg: 1,
                    src: [st.down, std::ptr::null(), std::ptr::null()],
                    nb2: [st.down_nb2 as i64, 0, 0],
                    off: [2 * nb2 as i64, 0, 0],
                    ..mine
                };
                overlap::stage(&down, s)?;
                overlap::mark(stream, 1)?;
            }
        } else {
            // Only the slot table, on this stream, for the launches below.
            let table = overlap::StageParams {
                n_seg: 0,
                ctas: 1,
                ..mine
            };
            // SAFETY: as above.
            unsafe { overlap::stage(&table, stream)? };
        }
    } else {
        launch(LaunchParams { steps: 1, ..p }, stream)?;
    }
    launch(first, stream)?;
    let which = if fork { 0 } else { 1 };
    // SAFETY: as the caller promises; the SwiGLU op recorded the events.
    unsafe {
        if side {
            overlap::wait(stream, which)?;
        }
        if let Some(l) = link {
            peer::wait(stream, l, which)?;
        }
    }
    launch(staged, stream)
}

pub mod ahead;
pub mod overlap;
pub mod peer;
pub mod ptx;

#[cfg(test)]
mod tests;
