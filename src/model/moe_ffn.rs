//! Routed-expert FFN for a batch of tokens in one op (prefill and big decode batches):
//!
//! `out[:, t] = Σ_j w[j, t] · down_e(silu(gate_e · x_t) ⊙ (up_e · x_t))`, `e = ids[j, t]`
//!
//! It replaces the graph's three `mul_mat_id`s, the SwiGLU, the router-weight multiply and
//! the sum over the used experts. ggml's CUDA `mul_mat_id` copies the routing to the host and
//! runs each expert's rows on their own; this op groups the tokens by expert on the device and
//! runs every expert of the layer in two launches on the tensor cores, with each expert's
//! weight tiles dequantized from their GGUF blocks in shared memory (`src/cuda/moe_gemm.cu`).
//!
//! Expert `e`'s bytes of a weight are at `w.data + table[e] · w.nb[2]` (`table`: I32
//! `[n_expert]`, absent for `table[e] = e`). With the residency layer's tiered experts the ids
//! are already remapped (no table), and the weights are a layer's *staged* tensors: VRAM slots
//! followed by the layer's uncached experts, copied into VRAM by DMA ahead of the op
//! ([`crate::residency::staging`]). Read in place over the link instead, the tiles would fetch
//! each expert's weights several times at a fraction of the copy's speed (#43: 235 ms per
//! layer at 8K tokens against 22 ms staged).
//!
//! The op's result is a view at the start of a bigger node: the CUDA kernels keep their
//! grouping, the SwiGLU rows and the per-expert outputs behind it, so the compute allocator
//! plans that scratch with the rest of the graph and reuses it across layers.
//!
//! The CPU implementation is a plain reference (weights dequantized to F32, activations F32,
//! F32 sums), for tests.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static MOE_FFN: ExternalOp = ExternalOp {
    name: "moe_ffn",
    cpu: cpu_moe_ffn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::launch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// One MoE layer's routed experts: `[k, m, n_slots]` each (up and gate `[n_embd, n_ff, ..]`,
/// down `[n_ff, n_embd, ..]`), and the slot of each expert.
#[derive(Clone, Copy)]
pub struct ExpertRefs<'a> {
    pub up: Tensor<'a>,
    pub gate: Tensor<'a>,
    pub down: Tensor<'a>,
    /// I32 `[n_expert]`: expert -> index along dim 2 of the weights; `None` is the identity.
    pub table: Option<Tensor<'a>>,
}

/// Where the op keeps whether it has a table in the node's `op_params`, after the custom
/// op's own (`ggml_custom_op_params`, 24 bytes).
const P_TABLE: usize = 8;
/// ... and how it stores the per-pair outputs ([`PairOut::code`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const P_PAIR: usize = 9;
/// ... and whether it adds a gated shared expert ([`Shexp`]; bit 0), whose two tensors then
/// follow the table in the sources, and whether a reader sums the pairs itself ([`defer_sum`];
/// bit 1).
const P_SHEXP: usize = 10;
/// [`P_SHEXP`]'s bits.
const SHEXP_ADDED: i32 = 1;
const SUM_DEFERRED: i32 = 2;

/// A sigmoid-gated shared expert the op adds to its sum (#526): `out[:, t] += sh[:, t] ·
/// sigmoid(g[t])`, each step rounded as ggml's `sigmoid`, `mul` and `add` round it, so the
/// same bits as those three ops after the op, without their passes over `[n_embd, n]`.
#[derive(Clone, Copy)]
pub struct Shexp<'a> {
    /// F32 `[n_embd, n]`, contiguous rows.
    pub sh: Tensor<'a>,
    /// F32 `[1, n]`: the gate logit of each token.
    pub g: Tensor<'a>,
}

/// Whether graphs built from now on may add the shared expert in the op ([`shexp_sum`];
/// `legion serve --shexp-sum-ggml` turns it off for comparisons).
static SHEXP_SUM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Sets whether graphs built from now on add a gated shared expert in the op's per-token sum
/// where it fits (on by default); `false` keeps ggml's sigmoid, mul and add after the op.
pub fn set_shexp_sum(on: bool) {
    SHEXP_SUM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the op's GEMMs on sm_80+ may stage weight rows through shared memory with cp.async
/// ([`set_raw_staging`]; `legion serve --no-moe-raw-staging` turns it off for comparisons).
static RAW_STAGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Sets whether the op's GEMMs on sm_80+ take the kernels that copy weight row blocks into
/// shared memory with cp.async first (on by default: `gemm_raw_kernel`, and on the GB10 the
/// fragment kernels); `false` keeps `gemm_kernel`, which loads each K step's weight bytes from
/// global memory. The same output bits either way. Read at each launch.
pub fn set_raw_staging(on: bool) {
    RAW_STAGING.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the op's sm_80+ GEMMs may take the cp.async-staged kernels ([`set_raw_staging`]).
pub fn raw_staging() -> bool {
    RAW_STAGING.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the op's down GEMM makes the per-token sums itself ([`set_down_sum`]; `legion serve
/// --moe-down-sum` turns it on).
static DOWN_SUM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets whether the CUDA op's down GEMM sums each token's pairs itself (#826; off by default)
/// where it runs the GB10's fragment down kernel (Q5_1 / Q8_0 down, F32 pair rows): the last
/// block to finish a token's pair sums the token's pairs, in slot order as the sum kernel does,
/// and that kernel does not run. The same output bits; `false` keeps the sum kernel. Opt-in:
/// prefill gained within noise on a GB10. Read at each launch.
pub fn set_down_sum(on: bool) {
    DOWN_SUM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the op's down GEMM may make the per-token sums itself ([`set_down_sum`]).
pub fn down_sum() -> bool {
    DOWN_SUM.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a model adds its gated shared expert in the op ([`Shexp`]): switched on, on a
/// compute capability 12.x GPU (GB10), where it was measured.
pub fn shexp_sum() -> bool {
    SHEXP_SUM.load(std::sync::atomic::Ordering::Relaxed) && crate::model::core::mmq_preq::gb10()
}

/// Whether [`moe_ffn_shexp`] takes `s` for `n_embd` values and `n` tokens: F32 operands of those
/// shapes with contiguous rows.
pub fn shexp_fits(s: &Shexp<'_>, n_embd: i64, n: i64) -> bool {
    let f32_rows = |t: Tensor<'_>| t.ty() == types::F32 && t.nb()[0] == 4;
    f32_rows(s.sh)
        && f32_rows(s.g)
        && s.sh.ne() == [n_embd, n, 1, 1]
        && s.g.ne() == [1, n, 1, 1]
        && s.sh.nb()[1].is_multiple_of(4)
        && s.g.nb()[1].is_multiple_of(4)
}
/// ... and, with split staging waits, the staging id (0: none) and row it waits on
/// ([`crate::residency::staging::split_wait`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const P_STAGE: usize = 11;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const P_STAGE_ROW: usize = 12;
/// ... and the SwiGLU clamp of [`moe_ffn_clamped`] (f32 bits; 0: none).
const P_CLAMP: usize = 13;
/// ... and whether the GEMMs may take GLM-5.3-Flash's IQ2 kernels ([`glm_kernels`]; 0: no).
const P_GLM: usize = 14;
/// ... and whether those may run on int8 tensor cores in prompt steps ([`glm_q8`]; 0: no).
const P_GLM_Q8: usize = 15;

/// How the CUDA op keeps each (token, expert) pair's weighted output before the per-token sum
/// (`legion serve --moe-pair-out`). F32 by default. F16 / BF16 (opt-in) halve the op's largest memory
/// stream (written by the down GEMM, read by the sum) and round each pair's output once:
/// output-moving. The sums stay F32, in slot order. The CPU reference ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum PairOut {
    #[default]
    F32,
    F16,
    Bf16,
}

/// [`PairOut::configured`]'s value, as [`PairOut::code`].
static DEFAULT_PAIR_OUT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

impl PairOut {
    /// What models loaded from now on start with ([`Self::configured`]; `legion serve
    /// --moe-pair-out`).
    pub fn set_default(p: Self) {
        DEFAULT_PAIR_OUT.store(p.code(), std::sync::atomic::Ordering::Relaxed);
    }

    /// What a model starts with: F32 unless [`Self::set_default`] said otherwise.
    pub fn configured() -> Self {
        match DEFAULT_PAIR_OUT.load(std::sync::atomic::Ordering::Relaxed) {
            1 => Self::F16,
            2 => Self::Bf16,
            _ => Self::F32,
        }
    }

    /// The kernels' `d_fmt`.
    fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }
}

/// The compiled tile of the CUDA kernels (`src/cuda/moe_gemm.cu`): pairs (token, expert slot)
/// per tile and weight rows per tile.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const TILE_PAIRS: i64 = 64;
const TILE_ROWS: i64 = 128;
/// K step of the kernels.
const TILE_K: i64 = 64;

/// Whether the CUDA kernels take these expert weights (types and shapes).
pub fn supported(e: &ExpertRefs<'_>) -> bool {
    supported_as(e, false)
}

/// [`supported`] for GLM-5.3-Flash and DeepSeek-V4.1-Flash, which also take Project Maya's quants
/// ([`crate::model::core::expert_mv::maya_type`]: Q2_K, Q3_K, IQ3_XXS, IQ3_S and IQ4_XS on top of
/// the IQ2 formats, #559) and IQ2_XS (DeepSeek-V4.1-Flash GSQ-RCO GGUFs mix it into their
/// IQ2_XXS / IQ1_M experts, #584). Other models keep their checks as before.
pub fn supported_glm(e: &ExpertRefs<'_>) -> bool {
    supported_as(e, true)
}

fn supported_as(e: &ExpertRefs<'_>, glm: bool) -> bool {
    let [n_embd, n_ff, _, _] = e.up.ne();
    let maya = |t| glm && (crate::model::core::expert_mv::maya_type(t) || t == types::IQ2_XS);
    let ok_type = |t| {
        maya(t)
            || matches!(
                t,
                types::Q4_K
                    | types::Q5_K
                    | types::Q6_K
                    | types::Q5_1
                    | types::Q8_0
                    | types::IQ2_XXS
                    | types::IQ2_S
                    | types::IQ1_M
                    | types::Q2_0
            )
    };
    // Formats of 256-value blocks (Q2_0's are 64 values: a K step holds whole ones).
    let kquant = |t| {
        maya(t)
            || matches!(
                t,
                types::Q4_K
                    | types::Q5_K
                    | types::Q6_K
                    | types::IQ2_XXS
                    | types::IQ2_S
                    | types::IQ1_M
            )
    };
    // A K step reads whole 32-value blocks, and K-quant super-blocks of 256 whole.
    let k_ok = |t, k: i64| k % if kquant(t) { 256 } else { TILE_K } == 0;
    e.gate.ty() == e.up.ty()
        && e.gate.ne() == e.up.ne()
        && e.gate.nb() == e.up.nb()
        && ok_type(e.up.ty())
        && ok_type(e.down.ty())
        && e.down.ne()[0] == n_ff
        && e.down.ne()[1] == n_embd
        && k_ok(e.up.ty(), n_embd)
        && k_ok(e.down.ty(), n_ff)
        && (2 * n_ff) % TILE_ROWS == 0
        && n_embd % TILE_ROWS == 0
}

/// Where the CUDA kernels' scratch lives behind the result, in bytes from its start (each
/// part 256-byte aligned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Scratch {
    xh: usize,
    count: usize,
    first: usize,
    cursor: usize,
    tile0: usize,
    list: usize,
    h: usize,
    d: usize,
    total: usize,
}

impl Scratch {
    fn new(n_expert: i64, n_used: i64, n_embd: i64, n_ff: i64, n_tok: i64) -> Self {
        let pairs = (n_tok * n_used) as usize;
        let (ne, n_embd, n_ff, n_tok) = (
            n_expert as usize,
            n_embd as usize,
            n_ff as usize,
            n_tok as usize,
        );
        let mut off = 0;
        let mut take = |bytes: usize| {
            let at = off;
            off += bytes.div_ceil(256) * 256;
            at
        };
        Self {
            xh: take(2 * n_tok * n_embd),
            count: take(4 * ne),
            first: take(4 * (ne + 1)),
            cursor: take(4 * ne),
            tile0: take(4 * (ne + 1)),
            list: take(4 * pairs),
            h: take(2 * pairs * n_ff),
            d: take(4 * pairs * n_embd),
            total: off,
        }
    }
}

/// The routed-expert FFN of `x` (F32 `[n_embd, n]`) with router choices `ids` (I32
/// `[n_used, n]`) and weights `w` (F32 `[n_used, n]`): F32 `[n_embd, n]`. `pair`: how the CUDA
/// kernels keep the per-pair outputs ([`PairOut`]).
pub fn moe_ffn<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
) -> Tensor<'a> {
    moe_ffn_with(ctx, x, ids, w, e, pair, None, None)
}

/// [`moe_ffn`], plus the gated shared expert `shexp` if given (see [`Shexp`], [`shexp_fits`]).
pub fn moe_ffn_shexp<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
    shexp: Option<Shexp<'a>>,
) -> Tensor<'a> {
    moe_ffn_with(ctx, x, ids, w, e, pair, None, shexp)
}

/// [`moe_ffn`] with a clamped SwiGLU, `silu(min(g, limit)) · clamp(u, -limit, limit)` (GLM-5's
/// `swiglu_limit`).
pub fn moe_ffn_clamped<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
    limit: f32,
) -> Tensor<'a> {
    clamp(moe_ffn_staged(ctx, x, ids, w, e, pair, None), limit)
}

/// Sets the SwiGLU clamp of `out`, a result of [`moe_ffn`] or [`moe_ffn_staged`] (see
/// [`moe_ffn_clamped`]); `limit` 0 leaves it plain.
pub fn clamp<'a>(out: Tensor<'a>, limit: f32) -> Tensor<'a> {
    // SAFETY: `out` is a view of the fresh node `moe_ffn_with` made.
    unsafe {
        let node = (*out.raw()).view_src;
        assert!(!node.is_null(), "moe_ffn::clamp: not a moe_ffn result");
        (*node).op_params[P_CLAMP] = limit.to_bits() as i32;
    }
    out
}

/// Sets whether the op of `out`, a result of [`moe_ffn`] or [`moe_ffn_staged`], may run its
/// GEMMs on GLM-5.3-Flash's kernels for Project Maya's IQ2 experts (#571, sm_80+, IQ2_XXS
/// gate/up and IQ2_S down; `legion_moe_gemm_glm` in `src/cuda/moe_gemm.cu`): each warp builds
/// its weight fragments from the raw blocks in registers, the same halves and the same MMAs in
/// the same order as the shared kernels (the same bits), ~1.2x faster on the GB10. Other types
/// and GPUs take the shared kernels either way.
pub fn glm_kernels<'a>(out: Tensor<'a>, on: bool) -> Tensor<'a> {
    // SAFETY: `out` is a view of the fresh node `moe_ffn_with` made.
    unsafe {
        let node = (*out.raw()).view_src;
        assert!(
            !node.is_null(),
            "moe_ffn::glm_kernels: not a moe_ffn result"
        );
        (*node).op_params[P_GLM] = on as i32;
    }
    out
}

/// Sets whether the op of `out` (marked by [`glm_kernels`]) may run its prompt steps' IQ2 GEMMs
/// on int8 tensor cores (#603, `legion serve --glm-expert-q8`; `sm80/moe_gemm_glm_q8.cuh`): x
/// and the SwiGLU rows rounded to Q8 blocks (int8 values, an F32 scale per 32, as ggml's
/// `quantize_q8_1`), each 32-value block's integer dot product scaled by the weight block's and
/// the activation block's scales. Output-moving, so opt-in. Steps of 16 tokens or fewer, other
/// types and GPUs before sm_80 keep the F16 kernels; down takes the int8 GEMM only when it is
/// IQ2_S or IQ3_XXS (#594; gate/up then writes the SwiGLU rows as Q8).
pub fn glm_q8<'a>(out: Tensor<'a>, on: bool) -> Tensor<'a> {
    // SAFETY: `out` is a view of the fresh node `moe_ffn_with` made.
    unsafe {
        let node = (*out.raw()).view_src;
        assert!(!node.is_null(), "moe_ffn::glm_q8: not a moe_ffn result");
        (*node).op_params[P_GLM_Q8] = on as i32;
    }
    out
}

/// [`moe_ffn`] over staged weights whose copy is still running: `stage` is the `(staging id,
/// row)` of [`crate::residency::experts::TieredLayer::staged_split`], and the op computes each range of
/// experts once its part of the copy is in (on Volta; newer GPUs wait for the whole copy).
/// The same outputs as waiting first.
pub fn moe_ffn_staged<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
    stage: Option<(usize, usize)>,
) -> Tensor<'a> {
    moe_ffn_with(ctx, x, ids, w, e, pair, stage, None)
}

/// [`moe_ffn_staged`] plus the gated shared expert `shexp` if given (see [`moe_ffn_shexp`]).
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_staged_shexp<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
    stage: Option<(usize, usize)>,
    shexp: Option<Shexp<'a>>,
) -> Tensor<'a> {
    moe_ffn_with(ctx, x, ids, w, e, pair, stage, shexp)
}

/// Whether `out` is the whole per-token sum of a [`moe_ffn`] op (a view at the start of its
/// node, as made) with F32 pair rows and no reader taking it over yet: what [`defer_sum`]
/// takes.
pub fn deferrable(out: Tensor<'_>) -> bool {
    // SAFETY: `out` and its view source are valid tensors of the context.
    unsafe {
        let t = &*out.raw();
        let node = t.view_src;
        if node.is_null() || t.op != ffi::ggml_op_GGML_OP_VIEW || t.view_offs != 0 {
            return false;
        }
        let ours = crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &MOE_FFN));
        if !ours {
            return false;
        }
        let n = &*node;
        let (x, ids, up) = (&*n.src[0], &*n.src[1], &*n.src[3]);
        // the reader's pair mask holds 16 pairs per token (`legion_hc_post_norm_moe_launch`)
        n.op_params[P_PAIR] == PairOut::F32.code()
            && ids.ne[0] <= 16
            && n.op_params[P_SHEXP] & SUM_DEFERRED == 0
            && t.ne[..2] == [up.ne[0], x.ne[1]]
            && t.nb[1] == up.ne[0] as usize * 4
    }
}

/// Hands the per-token sum of `out` ([`deferrable`]) to a reader that makes it from the op's
/// per-pair rows itself (`hc_post_norm` with `legion serve --hc-fuse`, #774): the CUDA op then
/// skips its sum kernel and leaves `out` unwritten, so that reader must be its only one. The
/// CPU op writes it either way.
pub fn defer_sum(out: Tensor<'_>) {
    assert!(
        deferrable(out),
        "moe_ffn::defer_sum: not a whole moe_ffn sum"
    );
    // SAFETY: `out` is a view of a node `moe_ffn_with` made (checked above).
    unsafe { (*(*out.raw()).view_src).op_params[P_SHEXP] |= SUM_DEFERRED };
}

/// What a reader of the deferred sum `out` ([`defer_sum`]) reads besides the op's node: the
/// routing ids, the slot table (if any), the shared expert and its gate logits (if the op adds
/// them). The reader takes them as its own sources too, so that the graph allocator keeps them
/// until it ran (else the op would be their last reader and their memory free for the nodes in
/// between).
pub fn sum_srcs(out: Tensor<'_>) -> Vec<Tensor<'_>> {
    assert!(sum_deferred(out), "moe_ffn::sum_srcs: not a deferred sum");
    // a view's first source is the viewed node
    let node = out.srcs()[0];
    let src = node.srcs();
    // SAFETY: a node `moe_ffn_with` made (`sum_deferred`).
    let (table, shexp) = unsafe {
        let p = &(*node.raw()).op_params;
        (p[P_TABLE] != 0, p[P_SHEXP] & SHEXP_ADDED != 0)
    };
    // sources: x, ids, w, up, gate, down, [table], [sh, g]
    let at = 6 + usize::from(table);
    let mut v = vec![src[1]];
    if table {
        v.push(src[6]);
    }
    if shexp {
        v.extend([src[at], src[at + 1]]);
    }
    v
}

/// Whether the op of `out`, a result of [`moe_ffn`] and the like, was handed to a reader by
/// [`defer_sum`].
pub fn sum_deferred(out: Tensor<'_>) -> bool {
    // SAFETY: `out` is a view of a node `moe_ffn_with` made.
    unsafe {
        let node = (*out.raw()).view_src;
        assert!(
            !node.is_null(),
            "moe_ffn::sum_deferred: not a moe_ffn result"
        );
        (*node).op_params[P_SHEXP] & SUM_DEFERRED != 0
    }
}

#[allow(clippy::too_many_arguments)]
fn moe_ffn_with<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    w: Tensor<'a>,
    e: ExpertRefs<'a>,
    pair: PairOut,
    stage: Option<(usize, usize)>,
    shexp: Option<Shexp<'a>>,
) -> Tensor<'a> {
    assert!(supported_glm(&e), "moe_ffn: unsupported expert weights");
    assert!(
        stage.is_none() || e.table.is_none(),
        "moe_ffn: split staging waits index experts by slot"
    );
    let [n_embd, n_ff, n_slots, _] = e.up.ne();
    let n = x.ne()[1];
    let n_used = ids.ne()[0];
    assert_eq!(x.ne()[0], n_embd, "moe_ffn: x rows");
    assert_eq!(x.ty(), types::F32, "moe_ffn: x is F32");
    assert_eq!(ids.ne()[1], n, "moe_ffn: ids columns");
    assert_eq!(w.ne()[..2], [n_used, n], "moe_ffn: router weights shape");
    let n_expert = e.table.map_or(n_slots, |t| t.ne()[0]);
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let w = if w.is_contiguous() { w } else { ctx.cont(w) };
    let out = (n_embd * n) as usize;
    let scratch = Scratch::new(n_expert, n_used, n_embd, n_ff, n)
        .total
        .div_ceil(4);
    // The scratch starts 256-byte aligned behind the result.
    let total = out.div_ceil(64) * 64 + scratch;
    let mut args = vec![x, ids, w, e.up, e.gate, e.down];
    args.extend(e.table);
    if let Some(s) = shexp {
        assert!(shexp_fits(&s, n_embd, n), "moe_ffn: shared expert");
        args.extend([s.sh, s.g]);
    }
    let node = ctx.external(&MOE_FFN, types::F32, [total as i64, 1, 1, 1], &args);
    // SAFETY: a fresh node; indices past the custom op's own parameters (see P_TABLE).
    unsafe {
        (*node.raw()).op_params[P_TABLE] = e.table.is_some() as i32;
        (*node.raw()).op_params[P_PAIR] = pair.code();
        (*node.raw()).op_params[P_SHEXP] = if shexp.is_some() { SHEXP_ADDED } else { 0 };
        if let Some((id, row)) = stage {
            (*node.raw()).op_params[P_STAGE] = id as i32;
            (*node.raw()).op_params[P_STAGE_ROW] = row as i32;
        }
    }
    ctx.view_2d(node, [n_embd, n], n_embd as usize * 4, 0)
}

/// The CPU implementation: tokens split over threads; per used expert, its weight rows
/// dequantized to F32 and F32 dot products in order of k; the experts' outputs summed in
/// slot order.
unsafe fn cpu_moe_ffn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `moe_ffn`: contiguous host tensors as checked there.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (x, ids, w, up, gate, down) = (src(0), src(1), src(2), src(3), src(4), src(5));
        let table = (dst.op_params[P_TABLE] != 0).then(|| src(6));
        let (n_embd, n_ff) = (up.ne[0] as usize, up.ne[1] as usize);
        let n_used = ids.ne[0] as usize;
        let n = x.ne[1] as usize;
        let lim = f32::from_bits(dst.op_params[P_CLAMP] as u32);
        let row = |t: &ffi::ggml_tensor, slot: usize, r: usize, k: usize| -> Vec<f32> {
            let bytes = std::slice::from_raw_parts(
                (t.data as *const u8).add(slot * t.nb[2] + r * t.nb[1]),
                crate::ggml::row_size(t.type_, k),
            );
            crate::ggml::dequantize(t.type_, bytes, k)
        };
        let out = dst.data as *mut f32;
        let mut h = vec![0f32; n_ff];
        for t in (ith..n).step_by(nth) {
            let xt = std::slice::from_raw_parts((x.data as *const f32).add(t * n_embd), n_embd);
            let mut acc = vec![0f32; n_embd];
            for j in 0..n_used {
                let e = *(ids.data as *const i32).add(t * n_used + j);
                let wt = *(w.data as *const f32).add(t * n_used + j);
                // A routing entry the weights don't hold is left out (as the CUDA kernels do),
                // never read out of bounds.
                let n_expert = table.map_or(up.ne[2], |tb| tb.ne[0]);
                let Some(e) = usize::try_from(e).ok().filter(|&e| (e as i64) < n_expert) else {
                    continue;
                };
                let slot = table.map_or(e as i64, |tb| *(tb.data as *const i32).add(e) as i64);
                let Some(slot) = usize::try_from(slot)
                    .ok()
                    .filter(|&s| (s as i64) < up.ne[2])
                else {
                    continue;
                };
                let dot =
                    |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(p, q)| p * q).sum() };
                for (r, hr) in h.iter_mut().enumerate() {
                    let g = dot(&row(gate, slot, r, n_embd), xt);
                    let u = dot(&row(up, slot, r, n_embd), xt);
                    let (g, u) = if lim > 0.0 {
                        (g.min(lim), u.clamp(-lim, lim))
                    } else {
                        (g, u)
                    };
                    *hr = g / (1.0 + (-g).exp()) * u;
                }
                for (r, a) in acc.iter_mut().enumerate() {
                    *a += dot(&row(down, slot, r, n_ff), &h) * wt;
                }
            }
            // (The CPU writes the sum even when a reader takes it over: `SUM_DEFERRED`.)
            if dst.op_params[P_SHEXP] & SHEXP_ADDED != 0 {
                // The shared expert: ggml's sigmoid, then the product and the sum rounded apart.
                let at = 6 + table.is_some() as usize;
                let (sh, g) = (src(at), src(at + 1));
                let gt = *(g.data as *const u8).add(t * g.nb[1]).cast::<f32>();
                let s = 1.0 / (1.0 + (-gt).exp());
                let shr = (sh.data as *const u8).add(t * sh.nb[1]).cast::<f32>();
                for (r, a) in acc.iter_mut().enumerate() {
                    *a += *shr.add(r) * s;
                }
            }
            std::slice::from_raw_parts_mut(out.add(t * n_embd), n_embd).copy_from_slice(&acc);
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub(crate) use cuda::{PairSum, pair_sum};

/// Whether the op's GEMMs run on `mma.m16n8k16` on the current CUDA device (sm_80 and newer
/// with code for it; else Volta's m8n8k4, which newer GPUs emulate). False without CUDA.
pub fn mma16() -> bool {
    #[cfg(feature = "cuda")]
    return cuda::mma16_supported();
    #[cfg(not(feature = "cuda"))]
    false
}

#[cfg(test)]
mod tests;
