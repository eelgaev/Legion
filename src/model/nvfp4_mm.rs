//! W4A16 NVFP4 matmuls: ggml `NVFP4` weights times BF16 activations, F32 sums, times the
//! weight's per-tensor (or per-expert) scale in the same op. This is how NVIDIA's `W4A16_NVFP4`
//! checkpoints are meant to run; ggml's own CUDA kernels quantize the activations to 8 (Q8_1)
//! or 4 bits instead.
//!
//! - [`mul_mat`]: a dense weight `[k, m]`.
//! - [`mul_mat_id`]: stacked experts `[k, m, n_expert]` (`mul_mat_id`), the pairs of tokens and
//!   their experts grouped by expert on the device, so each expert's weights are read once per
//!   step.
//!
//! CUDA (`src/cuda/nvfp4_mm/`): each weight is decoded exactly to BF16 in registers (an E2M1
//! code times its UE4M3 sub-block scale has at most 6 significant bits), `mma.sync` m16n8k16
//! with F32 accumulation, then the scale. Two kernels, picked here by the batch:
//! - up to [`MV_MAX_N`] tokens (decode), `nvfp4_mv`: bound by the weight stream, so each warp
//!   loads its weights straight into registers, several blocks ahead;
//! - more (prefill), `nvfp4_gemm`: 128- or 256-row tiles through shared memory.
//!
//! CPU: the references (ggml's dequantization, BF16 activations widened, F32 dot products).
//!
//! The kernel file holds kernel bodies only; the shape checks, the kernel, tile and grid choice
//! and the scratch layout are here.

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static NVFP4_MM: ExternalOp = ExternalOp {
    name: "nvfp4_mm",
    cpu: cpu_nvfp4_mm,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static NVFP4_MM_ID: ExternalOp = ExternalOp {
    name: "nvfp4_mm_id",
    cpu: cpu_nvfp4_mm_id,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::mul_mat_id),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Bytes per 64-weight NVFP4 block: four sub-block scales and 32 bytes of codes.
const BLOCK_BYTES: i64 = 36;

/// Batches up to this many tokens (per expert, for [`mul_mat_id`]) take the mv kernel.
pub const MV_MAX_N: i64 = 32;

/// Dense batches up to this many tokens take the mv kernel under [`set_wide_mv`].
pub const WIDE_MV_MAX_N: i64 = 64;

/// Wider dense batches on the mv kernel take two row tiles (32 rows) per CTA.
const WIDE_MV_RTW2_N: i64 = 40;

/// Dense batches of this many tokens up to [`MV_MAX_N`] (one tile of 32 tokens) run four row
/// tiles one after the other per CTA: the same sums, a quarter of the CTAs (a DGX Spark (GB10),
/// Gemma 4 31B's shapes at 32 tokens, a speculative verify step of 4 sequences: 0.78 to 0.95 of
/// the time, gate/up 409 -> 329 us; at 16 tokens it is slower, so not below 24).
const MV_RPT4_MIN_N: i64 = 24;

static WIDE_MV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `legion serve --nvfp4-wide-mv`: dense matmuls (and fused SwiGLUs) of 33 to
/// [`WIDE_MV_MAX_N`] tokens on the mv kernel, 64 tokens per CTA, instead of the GEMM, whose
/// tiles read the weights at about half the stream's bandwidth there (GB10: ~115 against
/// ~215 GB/s, #664). Each token then gets the bits the mv kernel gives it at 1-32 tokens (its
/// sums do not depend on the other tokens), so outputs move where such steps occur (MTP verify
/// at 4+ sequences, short prompt chunks).
pub fn set_wide_mv(on: bool) {
    WIDE_MV.store(on, std::sync::atomic::Ordering::Relaxed);
}

static EXPERT_PDL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether the down mv of a routed SwiGLU FFN at decode batches ([`mul_mat_id`] on a
/// [`swiglu_id`]'s plan) is a programmatic dependent launch of the grouped SwiGLU mv on GPUs
/// that have it (sm_90+ code: the GB10): it finds its tile, copies its first weight stages and
/// prefetches while the SwiGLU mv's last CTAs run, and waits for them only before it reads
/// their output. The same bits either way. On by default; `legion serve
/// --no-nvfp4-expert-pdl` turns it off for comparisons.
pub fn set_expert_pdl(on: bool) {
    EXPERT_PDL.store(on, std::sync::atomic::Ordering::Relaxed);
}

static EXPERT_TILE_PREFETCH: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Whether the decode expert mvs of a routed SwiGLU FFN at one token per expert tile prefetch
/// whole row tiles into L2 ([`GLU_MV_TILE`], [`DOWN_MV_TILE`]) instead of each row a few
/// stages ahead: each expert's rows are fetched in long contiguous runs. GB10,
/// Qwen3.8-Flash-Next, 48 layers over 68 GB of experts (`legion nvfp4-mv-bench --copies 48`):
/// 16 tokens 1151 -> 1113 us per layer (224 -> 232 GB/s; a pure read of the same bytes
/// reaches ~239), 1 token 153 -> 143 us. The same bits either way. On by default; `legion
/// serve --no-nvfp4-expert-tile-prefetch` turns it off for comparisons.
pub fn set_expert_tile_prefetch(on: bool) {
    EXPERT_TILE_PREFETCH.store(on, std::sync::atomic::Ordering::Relaxed);
}

static EXPERT_TMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the routed experts' fused SwiGLU GEMM at prompt steps ([`swiglu_id`] on the grouped
/// GEMM, default stages) fills its stages with the TMA (one tensor copy per 128 weight rows, one
/// bulk copy per token, mbarrier stages, the last warp out of a stage refilling it) instead of
/// per-thread cp.async copies and a CTA barrier per block of k, on GPUs with sm_90+ code (the
/// GB10) and 16-byte aligned weight rows (k a multiple of 256). The same MMAs in the same order:
/// the same bits either way. `legion serve --nvfp4-expert-tma` (#1183).
pub fn set_expert_tma(on: bool) {
    EXPERT_TMA.store(on, std::sync::atomic::Ordering::Relaxed);
}

static EXPERT_TILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static EXPERT_TILED_MODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// Micro harness only (`nvfp4-gemm-bench`, #1183): the grouped SwiGLU GEMM at default stages
/// reads its gate/up experts from `packed` (a device address of the tiled copy, see
/// `src/cuda/nvfp4_mm/gemm_glu_tile.cuh`) with its stages filled in `mode` (0-3); `None`: off.
#[doc(hidden)]
pub fn set_expert_tiled(tiled: Option<(usize, i32)>) {
    use std::sync::atomic::Ordering::Relaxed;
    let (packed, mode) = tiled.unwrap_or((0, -1));
    EXPERT_TILED.store(packed, Relaxed);
    EXPERT_TILED_MODE.store(mode, Relaxed);
}

/// The tiled copy of the gate (`gate`) and up (`up`) experts, `[k, m, n_expert]` NVFP4 bytes
/// (rows `k / 64 * 36` bytes, packed), in `gemm_glu_tile.cuh`'s layout: per expert, per 128 rows,
/// per block of k, 9,216 bytes in the order the GEMM's lanes read them.
#[doc(hidden)]
pub fn tile_experts(gate: &[u8], up: &[u8], k: usize, m: usize, n_expert: usize) -> Vec<u8> {
    const SPAN: usize = 9216;
    let (row, blocks, tiles_m) = (k / 64 * 36, k / 64, m.div_ceil(128));
    let mut out = vec![0u8; n_expert * tiles_m * blocks * SPAN];
    for e in 0..n_expert {
        for mt in 0..tiles_m {
            for b in 0..blocks {
                let span = &mut out[((e * tiles_m + mt) * blocks + b) * SPAN..][..SPAN];
                for w in 0..8 {
                    for lane in 0..32 {
                        let (g, t) = (lane / 4, lane % 4);
                        for (i, src) in [gate, up].into_iter().enumerate() {
                            for h in 0..2 {
                                let r = mt * 128 + w * 16 + h * 8 + g;
                                if r >= m {
                                    continue;
                                }
                                let blk = &src[(e * m + r) * row + b * 36..][..36];
                                for s in 0..4 {
                                    let at = ((w * 2 + i) * 32 + lane) * 16 + s * 4 + h * 2;
                                    span[at..at + 2]
                                        .copy_from_slice(&blk[4 + 8 * s + 2 * t..][..2]);
                                }
                                span[8192 + (w * 32 + lane) * 4 + i * 2 + h] = blk[t];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

static PAD_SKIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether a routed SwiGLU FFN on the grouped mv kernels ([`swiglu_id_rows`] with a row map,
/// then the down [`mul_mat_id`] on its plan) skips the pairs of a verify step's padding rows
/// (the rows the map names another row for, see [`crate::model::core::pad_rows`]) instead of
/// routing them as the row they name: their results are zeros, and no expert tile holds them,
/// so a first row's experts stay one tile when its sequence has many padding rows (at 8 tokens
/// per tile, a sequence with one draft of 8 filled its first row's tiles; any other row of the
/// step on the same expert took a second tile, a second read of its weights). The real rows'
/// bits do not change (each tile column is summed on its own). On by default; `legion serve
/// --no-moe-pad-skip` turns it off for comparisons.
pub fn set_pad_skip(on: bool) {
    PAD_SKIP.store(on, std::sync::atomic::Ordering::Relaxed);
}

static EXPERT_ROWS_BF16: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Prompt steps of more tokens than this take [`set_expert_rows_bf16`]'s BF16 rows (decode and
/// verify steps, and steps that batch a prompt with them, stay F32).
pub const EXPERT_ROWS_BF16_MIN_N: i64 = 64;

/// `legion serve --nvfp4-expert-rows-bf16` (#1183; off by default): the routed experts' down
/// matmul of prompt steps of more than [`EXPERT_ROWS_BF16_MIN_N`] tokens on Legion's W4A16
/// kernels stores its per-pair rows as BF16 ([`Act::Bf16`]) instead of F32, and the weighted
/// sum reads them back as F32 (the hyper-connection norm a qwen4exp MoE tail folds into reads
/// the BF16 rows itself). Half the bytes of those rows written and read; each row value is
/// rounded to BF16 before the sum, so output moves (opt-in).
pub fn set_expert_rows_bf16(on: bool) {
    EXPERT_ROWS_BF16.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a routed-expert down matmul of `n_tok` tokens stores BF16 rows
/// ([`set_expert_rows_bf16`]).
pub fn expert_rows_bf16(n_tok: i64) -> bool {
    n_tok > EXPERT_ROWS_BF16_MIN_N && EXPERT_ROWS_BF16.load(std::sync::atomic::Ordering::Relaxed)
}

static EXPERT_MV_MAX: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(MV_MAX_N);

/// `legion serve --nvfp4-expert-mv-max <N>`: routed-expert matmuls ([`mul_mat_id`],
/// [`swiglu_id`]) of up to `per` pairs per used expert (if the routing were even) take the
/// grouped mv kernel, more the grouped GEMM ([`pick_grouped`]); [`MV_MAX_N`] by default. A
/// lower value moves shorter prompt steps to the GEMM, whose sums round in another order than
/// the mv kernel's, so outputs move where such steps occur.
pub fn set_expert_mv_max(per: i64) {
    EXPERT_MV_MAX.store(per, std::sync::atomic::Ordering::Relaxed);
}

/// The most pairs per used expert on the grouped mv kernel ([`set_expert_mv_max`]).
pub fn expert_mv_max() -> i64 {
    EXPERT_MV_MAX.load(std::sync::atomic::Ordering::Relaxed)
}

static EXPERT_MV_NT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(4);

/// `legion serve --nvfp4-expert-mv-tile <NT>`: the grouped mv kernel's widest token tile, `8 *
/// nt` tokens of one expert per CTA (1, 2 or 4; 4 by default). A token's sums do not depend on
/// the other tokens of its tile, so every value gives the same bits; a narrower tile does fewer
/// MMAs on the columns an expert with few pairs leaves empty, and reads the weights of an
/// expert with more pairs than one tile holds once per tile.
pub fn set_expert_mv_tile(nt: i32) {
    assert!(
        matches!(nt, 1 | 2 | 4),
        "grouped mv token tiles: 1, 2 or 4, not {nt}"
    );
    EXPERT_MV_NT.store(nt, std::sync::atomic::Ordering::Relaxed);
}

/// The grouped mv kernel's widest token tile ([`set_expert_mv_tile`]).
pub fn expert_mv_tile() -> i32 {
    EXPERT_MV_NT.load(std::sync::atomic::Ordering::Relaxed)
}

/// How [`set_expert_mv_max`]'s cutoff applies (`legion serve --nvfp4-expert-mv-split`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum MvSplit {
    /// A step's pairs per used expert (if the routing were even) pick one kernel for all its
    /// experts (the default).
    Step,
    /// Prompt steps of [`SPLIT_MIN_TOKENS`] or more tokens and up to [`MV_MAX_N`] pairs per
    /// expert: the experts with more pairs than the cutoff on the grouped GEMM, the others on
    /// the grouped mv kernel, two launches on one plan ([`Grouped::Split`]).
    Expert,
    /// The cutoff for the down projections (and the other non-SwiGLU expert matmuls) only:
    /// the fused gate/up keeps the default's choice and bits.
    Down,
    /// [`MvSplit::Expert`] for the down projections only, the fused gate/up as the default's.
    ExpertDown,
}

impl MvSplit {
    fn code(self) -> u8 {
        match self {
            MvSplit::Step => 0,
            MvSplit::Expert => 1,
            MvSplit::Down => 2,
            MvSplit::ExpertDown => 3,
        }
    }

    fn from_code(v: u8) -> Self {
        match v {
            1 => MvSplit::Expert,
            2 => MvSplit::Down,
            3 => MvSplit::ExpertDown,
            _ => MvSplit::Step,
        }
    }
}

static EXPERT_MV_SPLIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `legion serve --nvfp4-expert-mv-split <MODE>` ([`MvSplit`]); [`MvSplit::Step`] by default.
/// Every mode but the default moves bits where the GEMM takes experts the mv kernel had (each
/// expert's outputs are the bits of the kernel it runs on, whatever the other experts run on).
pub fn set_expert_mv_split(mode: MvSplit) {
    EXPERT_MV_SPLIT.store(mode.code(), std::sync::atomic::Ordering::Relaxed);
}

/// [`set_expert_mv_split`]'s mode.
pub fn expert_mv_split() -> MvSplit {
    MvSplit::from_code(EXPERT_MV_SPLIT.load(std::sync::atomic::Ordering::Relaxed))
}

/// [`MvSplit::Expert`] splits steps of at least this many tokens: decode and verify steps (up
/// to 16 sequences of 9 tokens) keep the step-wide choice.
pub const SPLIT_MIN_TOKENS: i64 = 256;

/// The token tiles of a grouped launch over `pairs` pairs at `cols` per tile: the full tiles,
/// at most one partial tile per expert, and with `spare` one more past them (the CTAs that
/// store a skipping plan's padding pairs, which find no tile of their own).
pub fn grouped_tiles(pairs: i64, cols: i64, n_expert: i64, spare: bool) -> i32 {
    (pairs / cols + pairs.min(n_expert) + spare as i64) as i32
}

/// The largest dense batch on the mv kernel: [`MV_MAX_N`], or [`WIDE_MV_MAX_N`] under
/// [`set_wide_mv`].
pub fn dense_mv_max_n() -> i64 {
    if WIDE_MV.load(std::sync::atomic::Ordering::Relaxed) {
        WIDE_MV_MAX_N
    } else {
        MV_MAX_N
    }
}

/// Whether `w · x` can take these ops on `w`'s device: an NVFP4 weight (one matrix, or stacked
/// experts) with `k` a multiple of 64, on the CPU or on a GPU with BF16 tensor cores (sm_80+).
pub fn supports(w: Tensor<'_>) -> bool {
    let [k, _, _, w3] = w.ne();
    w.ty() == types::NVFP4
        && k % 64 == 0
        && w3 == 1
        && w.nb()[1] as i64 == k / 64 * BLOCK_BYTES
        && (w.is_host() || device_supported())
}

/// Whether this process's GPU runs the kernels on device-resident weights (BF16 tensor cores,
/// sm_80+; probed on the current device): [`supports`] is false for them elsewhere.
pub fn device_supported() -> bool {
    crate::model::fp8::bf16_mma()
}

/// `x` (F32 or BF16) as contiguous BF16, the kernels' input (one cast shared by the matmuls of
/// the same `x`).
fn bf16_input<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    if x.ty() == types::BF16 && x.is_contiguous() {
        return x;
    }
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.cast_shared(x, types::BF16)
}

/// What the kernels' epilogue applies to `scale · (w · x)` before it stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    /// Nothing: F32 out.
    None,
    /// `relu(v)²`, BF16 out: the up matmul of a squared-ReLU FFN (Nemotron-H), stored as the
    /// down matmul's input, which it would round to BF16 anyway. Replaces the graph's relu, sqr
    /// and cast and their passes over the hidden activations.
    ReluSqr,
    /// Nothing, BF16 out (each value rounded to nearest): the routed experts' down matmul of
    /// prompt steps under `legion serve --nvfp4-expert-rows-bf16` (#1183), half the bytes of the
    /// per-pair rows written and read back.
    Bf16,
}

impl Act {
    fn out_type(self) -> ffi::ggml_type {
        match self {
            Act::None => types::F32,
            Act::ReluSqr | Act::Bf16 => types::BF16,
        }
    }

    /// `legion_nvfp4_mm::act` (2 is the GLU launches' GELU, [`GLU_GELU`]).
    fn code(self) -> i32 {
        match self {
            Act::None => 0,
            Act::ReluSqr => 1,
            Act::Bf16 => 3,
        }
    }
}

/// Where the nodes keep their [`Act`] (as `legion_nvfp4_mm::act`) in `op_params`.
const P_ACT: usize = 10;

fn act_of(dst: &ffi::ggml_tensor) -> Act {
    match dst.op_params[P_ACT] {
        1 => Act::ReluSqr,
        3 => Act::Bf16,
        _ => Act::None,
    }
}

/// `v` through the epilogue `act`, stored at `out[i]` (F32 or BF16 as `act` says).
///
/// # Safety
/// `out` holds `i + 1` elements of `act`'s output type.
unsafe fn store_act(out: *mut u8, i: usize, v: f32, act: Act) {
    // SAFETY: as documented.
    unsafe {
        match act {
            Act::None => *(out as *mut f32).add(i) = v,
            Act::ReluSqr => *(out as *mut u16).add(i) = bf16_bits(v.max(0.0) * v.max(0.0)),
            Act::Bf16 => *(out as *mut u16).add(i) = bf16_bits(v),
        }
    }
}

/// `scale · (w · x)` for an NVFP4 weight `w` (`[k, m]`, see [`supports`]), activations `x`
/// (`[k, n, ..]`, F32 rounded to BF16 here, or BF16) and an optional F32 `[1]` per-tensor
/// `scale`: F32 `[m, n, ..]`.
pub fn mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    mul_mat_act(ctx, w, x, scale, Act::None)
}

/// [`mul_mat`] with the epilogue `act`: `[m, n, ..]` of `act`'s type (F32 or BF16).
pub fn mul_mat_act<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    act: Act,
) -> Tensor<'a> {
    assert!(
        supports(w) && w.ne()[2] == 1,
        "unsupported NVFP4 weight {:?}",
        w.ne()
    );
    let [k, m, _, _] = w.ne();
    assert_eq!(
        x.ne()[0],
        k,
        "NVFP4 matmul: x rows of {} for k = {k}",
        x.ne()[0]
    );
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [1, 1, 1, 1]),
            "a per-tensor scale"
        );
    }
    let x = bf16_input(ctx, x);
    let [_, n1, n2, n3] = x.ne();
    let mut args = vec![w, x];
    args.extend(scale);
    let node = ctx.external(&NVFP4_MM, act.out_type(), [m, n1, n2, n3], &args);
    // SAFETY: a fresh node; an index past the custom op's own parameters (see P_SCALE).
    unsafe { (*node.raw()).op_params[P_ACT] = act.code() };
    node
}

/// Where [`mul_mat_id`]'s node keeps whether it has a scale, in `op_params` after the custom
/// op's own (`ggml_custom_op_params`, 24 bytes).
const P_SCALE: usize = 8;

/// The I32s after the pairs in a grouped plan (`nvfp4_plan`): its token tiles of 8, 16, 32, 64
/// and 128 pairs over every expert (each mv and GEMM token tile), so the grouped mv's and GEMM's
/// CTAs past the routing's last tile exit at once.
pub(crate) const PLAN_TILES: i64 = 5;

/// The I32s of a grouped plan of `pairs` pairs over `n_expert` experts: each expert's first pair
/// (`n_expert + 1`), the pairs grouped by expert, then [`PLAN_TILES`].
pub(crate) fn plan_len(n_expert: i64, pairs: i64) -> usize {
    (n_expert + 1 + pairs + PLAN_TILES) as usize
}

/// Bytes of the result before the plan (`esize` per element, 256-byte aligned) and the plan's
/// I32s ([`plan_len`]).
fn id_layout(m: i64, n_used: i64, n_tok: i64, n_expert: i64, esize: usize) -> (usize, usize) {
    let out = (m * n_used * n_tok) as usize * esize;
    (out.div_ceil(256) * 256, plan_len(n_expert, n_used * n_tok))
}

/// Where [`mul_mat_id`]'s node keeps whether its `x` is a whole [`swiglu_id`] result of the same
/// routing ([`plan_of_swiglu`]): the down matmul of a SwiGLU FFN, which then reads that node's
/// plan of the pairs (CUDA) instead of making the same one again.
const P_PLAN: usize = 12;

/// Whether `x` is the whole result of a [`swiglu_id`] node of the routing `ids` (the same tensor,
/// or a copy of the same one) over `n_expert` experts.
fn plan_of_swiglu(x: Tensor<'_>, ids: Tensor<'_>, n_expert: i64) -> bool {
    // SAFETY: `x`, `ids` and their sources are valid tensors of the context.
    unsafe {
        let t = &*x.raw();
        let node = t.view_src;
        if node.is_null() || t.op != ffi::ggml_op_GGML_OP_VIEW || t.view_offs != 0 {
            return false;
        }
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &NVFP4_SWIGLU_ID)) {
            return false;
        }
        let n = &*node;
        let (wg, theirs, ours) = (&*n.src[0], n.src[3], ids.raw());
        let copy_of = |a: *mut ffi::ggml_tensor| {
            (*a).op == ffi::ggml_op_GGML_OP_CONT && !(*a).src[0].is_null()
        };
        let same_ids = std::ptr::eq(theirs, ours)
            || (copy_of(theirs) && copy_of(ours) && (*theirs).src[0] == (*ours).src[0]);
        same_ids && wg.ne[2] == n_expert && t.ne[..3] == [wg.ne[1], (*ours).ne[0], (*ours).ne[1]]
    }
}

/// The plan of the pairs a [`swiglu_id`] node left after its result, when `x` (a
/// [`mul_mat_id`]'s input flagged by [`plan_of_swiglu`]) still is a view of that node where it
/// runs (an input from another backend is a copy).
#[cfg(feature = "cuda")]
unsafe fn swiglu_plan(x: &ffi::ggml_tensor, n_used: i64, n_tok: i64, n_expert: i64) -> *mut i32 {
    let node = x.view_src;
    if node.is_null() || x.view_offs != 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: `x`'s view source is a valid tensor; a swiglu_id node's plan follows its result
    // (see `id_layout`).
    unsafe {
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &NVFP4_SWIGLU_ID)) {
            return std::ptr::null_mut();
        }
        let (out, _) = id_layout(x.ne[0], n_used, n_tok, n_expert, 2);
        ((*node).data as *mut u8).add(out) as *mut i32
    }
}

/// The expert matmul `mul_mat_id(w, x, ids)` of NVFP4 experts `w` (`[k, m, n_expert]`, see
/// [`supports`]) with activations `x` (`[k, n_x, n_tok]`, `n_x` 1 or `n_used`; F32 rounded to
/// BF16 here, or BF16) and the router's choices `ids` (I32 `[n_used, n_tok]`), each row times
/// its expert's entry of `scale` (F32 `[n_expert]`) when given: F32 `[m, n_used, n_tok]`.
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    mul_mat_id_act(ctx, w, x, ids, scale, Act::None)
}

/// [`mul_mat_id`] with the epilogue `act`: `[m, n_used, n_tok]` of `act`'s type.
pub fn mul_mat_id_act<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
    act: Act,
) -> Tensor<'a> {
    assert!(supports(w), "unsupported NVFP4 experts {:?}", w.ne());
    let [k, m, n_expert, _] = w.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, n_x, xt, _] = x.ne();
    assert_eq!(xk, k, "NVFP4 mul_mat_id: x rows of {xk} for k = {k}");
    assert!(
        (n_x == 1 || n_x == n_used) && xt == n_tok,
        "NVFP4 mul_mat_id: x {:?} for ids {:?}",
        x.ne(),
        ids.ne()
    );
    assert!(n_expert <= 1024, "NVFP4 mul_mat_id: {n_expert} experts");
    assert_eq!(ids.ty(), types::I32);
    if let Some(s) = scale {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [n_expert, 1, 1, 1]),
            "a per-expert scale"
        );
    }
    let x = bf16_input(ctx, x);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    let esize = crate::ggml::row_size(act.out_type(), 1);
    let (out, plan) = id_layout(m, n_used, n_tok, n_expert, esize);
    let shared = plan_of_swiglu(x, ids, n_expert);
    let mut args = vec![w, x, ids];
    args.extend(scale);
    let node = ctx.external(
        &NVFP4_MM_ID,
        act.out_type(),
        [((out + 4 * plan) / esize) as i64, 1, 1, 1],
        &args,
    );
    // SAFETY: a fresh node; indices past the custom op's own parameters (see P_SCALE).
    unsafe {
        (*node.raw()).op_params[P_SCALE] = scale.is_some() as i32;
        (*node.raw()).op_params[P_ACT] = act.code();
        (*node.raw()).op_params[P_PLAN] = shared as i32;
    }
    ctx.view_3d(
        node,
        [m, n_used, n_tok],
        [m as usize * esize, (m * n_used) as usize * esize],
        0,
    )
}

static NVFP4_SWIGLU: ExternalOp = ExternalOp {
    name: "nvfp4_swiglu",
    cpu: cpu_nvfp4_swiglu,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::swiglu),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// `silu(sg · (wg · x)) ⊙ (su · (wu · x))` in one op: the gate and up matmuls of a SwiGLU FFN
/// with the activation, for NVFP4 `wg` and `wu` of the same shape `[k, m]` (see [`supports`]),
/// `x` `[k, n, ..]` (F32 rounded to BF16 here, or BF16) and optional F32 `[1]` per-tensor
/// scales: BF16 `[m, n, ..]`, ready for the down matmul's [`mul_mat`] (which would round it to
/// BF16 anyway). The same CTA reads a row range of both weights (the mv kernel up to
/// [`MV_MAX_N`] tokens, the GEMM above), the activations once, and the two results never leave
/// registers.
pub fn swiglu<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    glu(ctx, wg, wu, x, sg, su, false)
}

/// [`swiglu`] with the tanh GELU in place of silu: `gelu(sg · (wg · x)) ⊙ (su · (wu · x))`,
/// a GeGLU FFN's gate and up (Gemma 4), BF16 out. Each output is the bits of the two matmuls,
/// ggml-cuda's `geglu_split` (its GELU expression) and the down matmul's cast to BF16.
pub fn geglu<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    glu(ctx, wg, wu, x, sg, su, true)
}

/// The `act` of a GLU node whose gate takes the tanh GELU (`NVFP4_GLU_GELU` in
/// `src/cuda/nvfp4_mm/common.cuh`); 0 is silu.
const GLU_GELU: i32 = 2;

fn glu<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
    gelu: bool,
) -> Tensor<'a> {
    assert!(
        supports(wg) && supports(wu) && wg.ne() == wu.ne() && wg.ne()[2] == 1,
        "unsupported NVFP4 gate / up {:?} {:?}",
        wg.ne(),
        wu.ne()
    );
    let [k, m, _, _] = wg.ne();
    assert_eq!(
        x.ne()[0],
        k,
        "NVFP4 SwiGLU: x rows of {} for k = {k}",
        x.ne()[0]
    );
    for s in [sg, su].into_iter().flatten() {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [1, 1, 1, 1]),
            "a per-tensor scale"
        );
    }
    let x = bf16_input(ctx, x);
    let [_, n1, n2, n3] = x.ne();
    let mut args = vec![wg, wu, x];
    args.extend(sg);
    args.extend(su);
    let node = ctx.external(&NVFP4_SWIGLU, types::BF16, [m, n1, n2, n3], &args);
    // SAFETY: a fresh node; indices past the custom op's own parameters (see P_SCALE).
    unsafe {
        (*node.raw()).op_params[P_SCALE] = sg.is_some() as i32;
        (*node.raw()).op_params[P_SCALE + 1] = su.is_some() as i32;
        (*node.raw()).op_params[P_ACT] = if gelu { GLU_GELU } else { 0 };
    }
    node
}

/// The gate and up scales of a node made by [`swiglu`] (`first`: 3) or [`swiglu_id`] (4), the
/// sources from `first` on.
///
/// # Safety
/// `dst` is such a node.
unsafe fn swiglu_scales(
    dst: &ffi::ggml_tensor,
    first: usize,
) -> (Option<&ffi::ggml_tensor>, Option<&ffi::ggml_tensor>) {
    unsafe {
        let has_g = dst.op_params[P_SCALE] != 0;
        let has_u = dst.op_params[P_SCALE + 1] != 0;
        let sg = has_g.then(|| &*dst.src[first]);
        let su = has_u.then(|| &*dst.src[first + has_g as usize]);
        (sg, su)
    }
}

/// The row map of a node made by [`swiglu_id_rows`] (its source after the scales), if any.
///
/// # Safety
/// `dst` is a node made by [`swiglu_id_full`].
unsafe fn swiglu_rows(dst: &ffi::ggml_tensor) -> Option<&ffi::ggml_tensor> {
    unsafe {
        let first =
            4 + (dst.op_params[P_SCALE] != 0) as usize + (dst.op_params[P_SCALE + 1] != 0) as usize;
        (dst.op_params[P_ROWS] != 0).then(|| &*dst.src[first])
    }
}

static NVFP4_SWIGLU_ID: ExternalOp = ExternalOp {
    name: "nvfp4_swiglu_id",
    cpu: cpu_nvfp4_swiglu_id,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::swiglu_id),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// [`swiglu`] for experts: `silu(sg[e] · (wg[e] · x)) ⊙ (su[e] · (wu[e] · x))` for each token
/// and its experts `e` from `ids` (I32 `[n_used, n_tok]`), NVFP4 `wg` and `wu` of the same shape
/// `[k, m, n_expert]`, `x` `[k, n_x, n_tok]` (`n_x` 1 or `n_used`; F32 rounded to BF16 here, or
/// BF16) and optional F32 `[n_expert]` per-expert scales: BF16 `[m, n_used, n_tok]`, the down
/// [`mul_mat_id`]'s input. The same values, bit for bit, as the two [`mul_mat_id`]s, ggml's
/// `silu` and `mul` and the down matmul's rounding to BF16. CUDA: where [`pick_grouped`] picks
/// the GEMM (prefill), one grouped GEMM; where it picks the mv kernel (decode), the grouped
/// SwiGLU mv ([`GLU_MV`]: 8 gate and 8 up rows per row tile, each row's sums the plain grouped
/// mv's). A down [`mul_mat_id`] on the whole result with the same `ids` reads this op's plan of
/// the pairs instead of making its own.
pub fn swiglu_id<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_id_clamped(ctx, wg, wu, x, ids, sg, su, 0.0)
}

/// Where [`swiglu_id_clamped`]'s node keeps its limit (F32 bits) in `op_params`.
const P_LIMIT: usize = 11;

/// [`swiglu_id`] with GLM-5.3-Flash's clamped SwiGLU for a `limit` above 0:
/// `silu(min(g, limit)) ⊙ clamp(u, -limit, limit)` of the scaled gate and up sums `g` and `u`
/// (the same bits as the two [`mul_mat_id`]s and `glm5next::ops::glu` rounded to BF16).
#[allow(clippy::too_many_arguments)]
pub fn swiglu_id_clamped<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
    limit: f32,
) -> Tensor<'a> {
    swiglu_id_full(ctx, wg, wu, x, ids, sg, su, limit, None)
}

/// Where [`swiglu_id_rows`]'s node keeps whether it has a row map (its last source).
const P_ROWS: usize = 13;

/// Where a [`mul_mat_id`] or [`swiglu_id`] node keeps whether its ids may hold -1 ([`skips`]).
const P_SKIP: usize = 14;

/// Marks the result of a [`mul_mat_id`] or [`swiglu_id`] (the view they return) as one whose
/// router ids may be -1: such a pair goes to no expert (no tile computes it) and its result is
/// zero. `--tp-moe-ep` (#1176) hands the grouped kernels each rank's slot table this way, so
/// a rank computes only the pairs of the experts it holds instead of the others' on its first
/// expert. The other pairs' bits are the same: each tile's sums do not depend on the other
/// tiles. On CUDA such a node keeps the step-wide kernel choice (not a split's) and one spare
/// token tile, whose CTAs store the zeros. The W4A4 grouped ops (`nvfp4_a4::mul_mat_id_act`,
/// `nvfp4_a4::swiglu_id`) take the mark too: they run the GEMM then, never the mv.
pub fn skips<'a>(t: Tensor<'a>) -> Tensor<'a> {
    // SAFETY: a view made by `mul_mat_id_act` or `swiglu_id_full` (or their W4A4 twins);
    // flags its fresh node.
    unsafe {
        let node = (*t.raw()).view_src;
        assert!(
            !node.is_null()
                && crate::ggml::op_of(node).is_some_and(|op| {
                    std::ptr::eq(op, &NVFP4_MM_ID)
                        || std::ptr::eq(op, &NVFP4_SWIGLU_ID)
                        || super::nvfp4_a4::is_grouped(op, false)
                        || super::nvfp4_a4::is_grouped(op, true)
                }),
            "nvfp4 skips: not a grouped NVFP4 result"
        );
        (*node).op_params[P_SKIP] = 1;
    }
    t
}

/// Whether [`skips`] marked the node `dst`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn skipping(dst: &ffi::ggml_tensor) -> bool {
    dst.op_params[P_SKIP] != 0
}

/// Where a [`skips`] node keeps whether nothing reads its skipped pairs' rows ([`leave_unread`]).
const P_UNREAD: usize = 15;

/// Whether [`leave_unread`] marked the node `dst`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn unread(dst: &ffi::ggml_tensor) -> bool {
    dst.op_params[P_UNREAD] != 0
}

/// The grouped NVFP4 nodes whose skipped pairs' rows nothing reads, given `down`, the node of
/// a down [`mul_mat_id`] marked by [`skips`] whose rows only a MoE tail reads that skips the
/// rows of zero router weights (the fold's moe_sum_gate under `--tp-moe-ep`, #1176): `down`,
/// and the [`swiglu_id`] whose plan it reads if that is marked too (the down matmul never
/// reads a skipped pair's input). `read_only_by_down`: whether that SwiGLU's result is read
/// by `down` alone.
/// Without a row map the only skipped pairs are the ids of -1 (the other ranks' experts),
/// whose router weights are zero; their zeros need not be stored. Empty when `down` is not
/// such a node, or when its plan has padding rows. The W4A4 twins (`nvfp4_a4`) likewise, the
/// SwiGLU when it routes by the down matmul's ids.
///
/// # Safety
/// `down` is a valid tensor whose sources are valid tensors.
pub(crate) unsafe fn unread_nodes(
    down: *mut ffi::ggml_tensor,
    read_only_by_down: impl Fn(*mut ffi::ggml_tensor) -> bool,
) -> Vec<*mut ffi::ggml_tensor> {
    let is = |t: *mut ffi::ggml_tensor, op: &ExternalOp| {
        // SAFETY: the caller's.
        unsafe { crate::ggml::op_of(t).is_some_and(|o| std::ptr::eq(o, op)) }
    };
    // SAFETY: the caller's; a node of `NVFP4_MM_ID` has its sources (`mul_mat_id_act`).
    unsafe {
        let a4 = crate::ggml::op_of(down).is_some_and(|o| super::nvfp4_a4::is_grouped(o, false));
        if !(a4 || is(down, &NVFP4_MM_ID)) || !skipping(&*down) {
            return Vec::new();
        }
        let mut x = (*down).src[1];
        while !(*x).view_src.is_null() {
            x = (*x).view_src;
        }
        if a4 {
            // The W4A4 ops: each runs its own plan of its ids (sources 2 and 3), and the down
            // matmul's GEMM reads no skipped pair's input row (its quantization reads them all,
            // whatever they hold, and nothing uses their codes).
            let glu = crate::ggml::op_of(x)
                .is_some_and(|o| super::nvfp4_a4::is_grouped(o, true))
                .then_some(x)
                .filter(|&g| skipping(&*g) && (*g).src[3] == (*down).src[2]);
            let mut out = vec![down];
            out.extend(glu.filter(|&g| read_only_by_down(g)));
            return out;
        }
        let glu = (is(x, &NVFP4_SWIGLU_ID) && skipping(&*x)).then_some(x);
        // A down matmul on a SwiGLU's plan has its padding pairs too.
        if glu.is_some_and(|g| swiglu_rows(&*g).is_some()) {
            return Vec::new();
        }
        let shared = (*down).op_params[P_PLAN] != 0;
        if shared && glu.is_none() {
            return Vec::new();
        }
        let mut out = vec![down];
        // Only on the SwiGLU's own plan is the down matmul's routing surely the same.
        out.extend(glu.filter(|&g| shared && read_only_by_down(g)));
        out
    }
}

/// Marks a node of [`unread_nodes`]: on CUDA its skipped pairs get no zeros stored.
///
/// # Safety
/// `node` is one [`unread_nodes`] returned.
pub(crate) unsafe fn leave_unread(node: *mut ffi::ggml_tensor) {
    // SAFETY: the caller's.
    unsafe { (*node).op_params[P_UNREAD] = 1 };
}

/// [`swiglu_id`] whose plan of the pairs routes each token's pairs as token `rows[t]`'s
/// (`rows` I32 `[n_tok]`; a verify step's padding rows as their sequences' first rows, see
/// [`crate::model::core::pad_rows`]): token `t`'s results are those of `rows[t]`'s experts on
/// its own input, so every token with `rows[t] == t` gets the same bits as without the map.
/// A down [`mul_mat_id`] reading this node's plan routes the same way.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_id_rows<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
    rows: Option<Tensor<'a>>,
) -> Tensor<'a> {
    swiglu_id_full(ctx, wg, wu, x, ids, sg, su, 0.0, rows)
}

#[allow(clippy::too_many_arguments)]
fn swiglu_id_full<'a>(
    ctx: &'a Context,
    wg: Tensor<'a>,
    wu: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    sg: Option<Tensor<'a>>,
    su: Option<Tensor<'a>>,
    limit: f32,
    rows: Option<Tensor<'a>>,
) -> Tensor<'a> {
    assert!(
        supports(wg) && supports(wu) && wg.ne() == wu.ne() && wg.nb() == wu.nb(),
        "unsupported NVFP4 gate / up experts {:?} {:?}",
        wg.ne(),
        wu.ne()
    );
    let [k, m, n_expert, _] = wg.ne();
    let [n_used, n_tok, _, _] = ids.ne();
    let [xk, n_x, xt, _] = x.ne();
    assert_eq!(xk, k, "NVFP4 SwiGLU experts: x rows of {xk} for k = {k}");
    assert!(
        (n_x == 1 || n_x == n_used) && xt == n_tok,
        "NVFP4 SwiGLU experts: x {:?} for ids {:?}",
        x.ne(),
        ids.ne()
    );
    assert!(n_expert <= 1024, "NVFP4 SwiGLU experts: {n_expert} experts");
    assert_eq!(ids.ty(), types::I32);
    for s in [sg, su].into_iter().flatten() {
        assert_eq!(
            (s.ty(), s.ne()),
            (types::F32, [n_expert, 1, 1, 1]),
            "a per-expert scale"
        );
    }
    let x = bf16_input(ctx, x);
    let ids = if ids.is_contiguous() {
        ids
    } else {
        ctx.cont(ids)
    };
    if let Some(r) = rows {
        assert!(
            r.ty() == types::I32 && r.ne() == [n_tok, 1, 1, 1] && r.is_contiguous(),
            "NVFP4 SwiGLU experts: row map {:?} for ids {:?}",
            r.ne(),
            ids.ne()
        );
    }
    let (out, plan) = id_layout(m, n_used, n_tok, n_expert, 2);
    let mut args = vec![wg, wu, x, ids];
    args.extend(sg);
    args.extend(su);
    args.extend(rows);
    let node = ctx.external(
        &NVFP4_SWIGLU_ID,
        types::BF16,
        [((out + 4 * plan) / 2) as i64, 1, 1, 1],
        &args,
    );
    // SAFETY: a fresh node; indices past the custom op's own parameters (see P_SCALE).
    unsafe {
        (*node.raw()).op_params[P_SCALE] = sg.is_some() as i32;
        (*node.raw()).op_params[P_SCALE + 1] = su.is_some() as i32;
        (*node.raw()).op_params[P_LIMIT] = limit.to_bits() as i32;
        (*node.raw()).op_params[P_ROWS] = rows.is_some() as i32;
    }
    ctx.view_3d(
        node,
        [m, n_used, n_tok],
        [m as usize * 2, (m * n_used) as usize * 2],
        0,
    )
}

/// The CPU implementation of [`swiglu_id`]: as [`cpu_nvfp4_swiglu`] per (token, slot).
unsafe fn cpu_nvfp4_swiglu_id(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `swiglu_id`: contiguous host tensors as checked there.
    unsafe {
        let (wg, wu, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let (sg, su) = swiglu_scales(dst, 4);
        let rows = swiglu_rows(dst);
        let limit = f32::from_bits(dst.op_params[P_LIMIT] as u32);
        let (k, m) = (wg.ne[0] as usize, wg.ne[1] as usize);
        let (n_used, n_tok) = (ids.ne[0] as usize, ids.ne[1] as usize);
        // The token whose experts token `t`'s pairs take.
        let routed = |t: usize| rows.map_or(t, |r| *(r.data as *const i32).add(t) as usize);
        let n_x = x.ne[1] as usize;
        let xs = bf16_values(x, k * n_x * n_tok);
        let out = dst.data as *mut u16;
        let value = |s: Option<&ffi::ggml_tensor>, e: usize| {
            s.map_or(1.0, |s| *(s.data as *const f32).add(e))
        };
        for j in (ith..m).step_by(nth) {
            for t in 0..n_tok {
                for s in 0..n_used {
                    let e = *(ids.data as *const i32).add(routed(t) * n_used + s);
                    // A skipped pair ([`skips`]): no expert, a zero.
                    let Ok(e) = usize::try_from(e) else {
                        *out.add((t * n_used + s) * m + j) = bf16_bits(0.0);
                        continue;
                    };
                    let xr = (t * n_x + if n_x == 1 { 0 } else { s }) * k;
                    let col = &xs[xr..xr + k];
                    let mut g = dot(&weight_row(wg, e, j), col) * value(sg, e);
                    let mut u = dot(&weight_row(wu, e, j), col) * value(su, e);
                    if limit > 0.0 {
                        g = g.min(limit);
                        u = u.clamp(-limit, limit);
                    }
                    *out.add((t * n_used + s) * m + j) = bf16_bits(g / (1.0 + (-g).exp()) * u);
                }
            }
        }
    }
}

/// The CPU implementation of [`swiglu`]: as [`cpu_nvfp4_mm`] for both weights, then the
/// activation in F32, rounded to BF16.
unsafe fn cpu_nvfp4_swiglu(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `swiglu`, host tensors, x and dst contiguous.
    unsafe {
        let (wg, wu, x) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (sg, su) = swiglu_scales(dst, 3);
        let value = |s: Option<&ffi::ggml_tensor>| s.map_or(1.0, |s| *(s.data as *const f32));
        let (sg, su) = (value(sg), value(su));
        let (k, m) = (wg.ne[0] as usize, wg.ne[1] as usize);
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let xs = bf16_values(x, k * n);
        let out = dst.data as *mut u16;
        let gelu = dst.op_params[P_ACT] == GLU_GELU;
        for j in (ith..m).step_by(nth) {
            let (rg, ru) = (weight_row(wg, 0, j), weight_row(wu, 0, j));
            for t in 0..n {
                let col = &xs[t * k..(t + 1) * k];
                let g = dot(&rg, col) * sg;
                let a = if gelu {
                    gelu_tanh(g)
                } else {
                    g / (1.0 + (-g).exp())
                };
                *out.add(t * m + j) = bf16_bits(a * (dot(&ru, col) * su));
            }
        }
    }
}

/// The tanh GELU as ggml computes it (`ggml_gelu_f32`, MIT): the CPU reference of the GeGLU
/// kernels' gate.
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    const GELU_COEF_A: f32 = 0.044715;
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
}

/// Activations of a host node as F32 (BF16 widened).
///
/// # Safety
/// `x` is a contiguous BF16 host tensor of `len` elements.
unsafe fn bf16_values(x: &ffi::ggml_tensor, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(x.data as *const u16, len) }
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

/// `w`'s row `j` of expert `e`, dequantized.
///
/// # Safety
/// `w` is an NVFP4 host tensor with that row.
unsafe fn weight_row(w: &ffi::ggml_tensor, e: usize, j: usize) -> Vec<f32> {
    let k = w.ne[0] as usize;
    // SAFETY: as documented.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (w.data as *const u8).add(e * w.nb[2] + j * w.nb[1]),
            k / 64 * BLOCK_BYTES as usize,
        )
    };
    crate::ggml::dequantize(types::NVFP4, bytes, k)
}

/// `v` rounded to the nearest BF16 (ties to even), as `__float2bfloat16_rn`; NaN stays NaN.
pub(crate) fn bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The CPU implementation: rows of `dst` split over threads; each element an F32 dot product of
/// the dequantized weights and the BF16 activations in order of k, times the scale.
unsafe fn cpu_nvfp4_mm(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors, x and dst contiguous.
    unsafe {
        let (w, x) = (&*dst.src[0], &*dst.src[1]);
        let scale = dst.src[2].as_ref().map_or(1.0, |s| *(s.data as *const f32));
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let n = (x.ne[1] * x.ne[2] * x.ne[3]) as usize;
        let xs = bf16_values(x, k * n);
        let (out, act) = (dst.data as *mut u8, act_of(dst));
        for j in (ith..m).step_by(nth) {
            let row = weight_row(w, 0, j);
            for t in 0..n {
                store_act(
                    out,
                    t * m + j,
                    dot(&row, &xs[t * k..(t + 1) * k]) * scale,
                    act,
                );
            }
        }
    }
}

/// The CPU implementation of [`mul_mat_id`]: rows split over threads, as [`cpu_nvfp4_mm`] per
/// (token, slot).
unsafe fn cpu_nvfp4_mm_id(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id`: contiguous host tensors as checked there.
    unsafe {
        let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let scale = (dst.op_params[P_SCALE] != 0).then(|| &*dst.src[3]);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let (n_used, n_tok) = (ids.ne[0] as usize, ids.ne[1] as usize);
        let n_x = x.ne[1] as usize;
        let xs = bf16_values(x, k * n_x * n_tok);
        let (out, act) = (dst.data as *mut u8, act_of(dst));
        for j in (ith..m).step_by(nth) {
            for t in 0..n_tok {
                for s in 0..n_used {
                    let e = *(ids.data as *const i32).add(t * n_used + s);
                    // A skipped pair ([`skips`]): no expert, a zero.
                    let Ok(e) = usize::try_from(e) else {
                        store_act(out, (t * n_used + s) * m + j, 0.0, act);
                        continue;
                    };
                    let sc = scale.map_or(1.0, |sc| *(sc.data as *const f32).add(e));
                    let xr = (t * n_x + if n_x == 1 { 0 } else { s }) * k;
                    let row = weight_row(w, e, j);
                    let v = dot(&row, &xs[xr..xr + k]) * sc;
                    store_act(out, (t * n_used + s) * m + j, v, act);
                }
            }
        }
    }
}

/// A kernel and its tile, as `src/cuda/nvfp4_mm/` compiles them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `nvfp4_mv`: 16 * `rtw` rows x 8 * `nt` tokens per CTA, `kw` warps splitting k; `rpt`
    /// such row tiles one after the other per CTA (the same sums, fewer CTAs).
    Mv {
        nt: i32,
        rtw: i32,
        kw: i32,
        rpt: i32,
    },
    /// `nvfp4_gemm`: 128 * `mi` rows x 8 * `nt` tokens per CTA.
    Gemm { mi: i32, nt: i32 },
}

impl Kernel {
    /// Rows and tokens of a tile.
    pub fn tile(self) -> (i64, i64) {
        match self {
            Kernel::Mv { nt, rtw, rpt, .. } => (16 * (rtw * rpt) as i64, 8 * nt as i64),
            Kernel::Gemm { mi, nt } => (128 * mi as i64, 8 * nt as i64),
        }
    }
}

/// The smallest compiled n8 tile count covering `n` tokens, up to `max`.
fn tokens_tile(n: i64, max: i32) -> i32 {
    let mut nt = 1;
    while nt < max && 8 * (nt as i64) < n {
        nt *= 2;
    }
    nt
}

/// The kernel for a dense `[k, m]` weight times `n` tokens on a GPU with `sms` SMs, batches
/// up to `mv_max_n` tokens ([`dense_mv_max_n`]) on the mv kernel.
pub fn pick_dense(_k: i64, m: i64, n: i64, sms: i64, mv_max_n: i64) -> Kernel {
    if n <= mv_max_n {
        // Above 40 tokens (--nvfp4-wide-mv) 32 rows per CTA: each weight stage serves twice
        // the rows for the same activations (GB10, 27B MLP shapes at 64 tokens: 17408 -> 5120
        // 461 -> 379 us, 5120 -> 17408 446 -> 324 us; 336 against 288 us at 33).
        return Kernel::Mv {
            nt: tokens_tile(n, 8),
            rtw: if n > WIDE_MV_RTW2_N { 2 } else { 1 },
            kw: 8,
            rpt: if (MV_RPT4_MIN_N..=MV_MAX_N).contains(&n) {
                4
            } else {
                1
            },
        };
    }
    // 256-row tiles while they give every SM three CTAs, 128-row tiles below that (fewer
    // waves' tails: 68 CTAs of 256 rows on 48 SMs run at ~0.7 of 136 of 128 rows).
    let nt = tokens_tile(n, 16);
    let tiles_n = (n + 8 * nt as i64 - 1) / (8 * nt as i64);
    let mi = if (m + 255) / 256 * tiles_n >= 3 * sms {
        2
    } else {
        1
    };
    Kernel::Gemm { mi, nt }
}

/// Row tiles per CTA of the grouped mv kernel.
const GROUPED_MV_RPT: i32 = 8;

/// A grouped mv kernel variant (`legion_nvfp4_mvg_launch`): `rpt` row tiles per CTA of 16 *
/// `rtw` rows (the SwiGLU: 8 gate and 8 up rows), stages of `sb` blocks. Every variant gives
/// the same bits: each block goes to warp `block % 8` in every one of them. `pft` 0: an L2
/// prefetch of each row a few stages ahead; above 0: of whole row tiles, `pft` tiles ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedMv {
    pub rtw: i32,
    pub rpt: i32,
    pub sb: i32,
    pub pft: i32,
}

/// The grouped SwiGLU mv kernel's variant ([`swiglu_id`] at decode batches).
pub const GLU_MV: GroupedMv = GroupedMv {
    rtw: 1,
    rpt: 8,
    sb: 16,
    pft: 0,
};

/// [`GLU_MV`] at one token per tile under [`set_expert_tile_prefetch`]: 4 row tiles per CTA,
/// each whole row tile prefetched into L2 one tile ahead.
pub const GLU_MV_TILE: GroupedMv = GroupedMv {
    rtw: 1,
    rpt: 4,
    sb: 16,
    pft: 1,
};

/// The down mv on a [`swiglu_id`]'s plan at one token per tile under
/// [`set_expert_tile_prefetch`]: the plain grouped mv's tiles, each whole row tile prefetched
/// into L2 two tiles ahead.
pub const DOWN_MV_TILE: GroupedMv = GroupedMv {
    rtw: 1,
    rpt: GROUPED_MV_RPT,
    sb: 16,
    pft: 2,
};

/// The kernel for `pairs` (token, expert) pairs over `n_expert` experts of `[k, m]`.
pub fn pick_grouped(k: i64, m: i64, pairs: i64, n_expert: i64) -> Kernel {
    pick_grouped_at(k, m, pairs, n_expert, expert_mv_max(), expert_mv_tile())
}

/// [`pick_grouped`] with up to `mv_max` pairs per expert on the mv kernel ([`set_expert_mv_max`])
/// at token tiles of up to `8 * mv_nt` ([`set_expert_mv_tile`]).
pub fn pick_grouped_at(
    k: i64,
    _m: i64,
    pairs: i64,
    n_expert: i64,
    mv_max: i64,
    mv_nt: i32,
) -> Kernel {
    // Pairs per expert used, if the routing is even.
    let per = pairs / pairs.min(n_expert).max(1);
    if per <= mv_max {
        // Eight row tiles per CTA (the same bits as one): a grid of one CTA per 16 rows and
        // expert tile is mostly CTAs that only look up an empty tile, and a 16-row down
        // projection tile (k = 640) copies one 6 KB stage. GB10, Qwen3.8-Flash-Next at 16 decode
        // tokens on 56 experts: down 362 -> 256 us, gate+up 579 -> 498 us; Nemotron 3.5's
        // shapes +3-7% (tests/nvfp4_mm.rs, cuda_swiglu_id_decode_timings and
        // cuda_mul_mat_id_variants).
        return Kernel::Mv {
            nt: tokens_tile(per, mv_nt),
            rtw: 1,
            kw: 8,
            rpt: GROUPED_MV_RPT,
        };
    }
    if wide_down(k, n_expert) {
        // 256 rows by 64 tokens, two CTAs per SM (grouped_gemm_tune): GB10, Qwen3.8-Flash-Next's
        // down projection (k = 640, 512 experts) at 2048-4096 tokens 10-27% faster than 128
        // rows by 128 tokens, the same bits (legion nvfp4-gemm-bench, #824).
        return Kernel::Gemm { mi: 2, nt: 8 };
    }
    Kernel::Gemm {
        mi: 1,
        nt: tokens_tile(per, 16),
    }
}

/// The launches of one grouped matmul ([`plan_grouped`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grouped {
    /// Every expert on one kernel.
    One(Kernel),
    /// The experts with at most `cut` pairs on `mv` (a [`Kernel::Mv`] whose tile holds `cut`
    /// tokens, up to [`set_expert_mv_tile`]'s), the others on `gemm`: two launches on the same
    /// plan, each finding only its experts' tiles.
    Split { cut: i32, mv: Kernel, gemm: Kernel },
}

/// The launches for `pairs` (token, expert) pairs of `n_tok` tokens over `n_expert` experts of
/// `[k, m]`: `glu` for the fused gate/up ([`swiglu_id`]), `step_only` where the step-wide
/// choice must hold (a plan with a row map, whose padding pairs only the mv kernel stores, and
/// a clamped SwiGLU, which the mv kernel does not have).
pub fn plan_grouped(
    k: i64,
    m: i64,
    pairs: i64,
    n_tok: i64,
    n_expert: i64,
    glu: bool,
    step_only: bool,
) -> Grouped {
    plan_grouped_at(
        k,
        m,
        [pairs, n_tok, n_expert],
        glu,
        step_only,
        expert_mv_max(),
        expert_mv_tile(),
        expert_mv_split(),
    )
}

/// [`plan_grouped`] at the cutoff `mv_max`, widest mv tile `mv_nt` and `mode`; `shape` is
/// `[pairs, n_tok, n_expert]`.
#[allow(clippy::too_many_arguments)]
pub fn plan_grouped_at(
    k: i64,
    m: i64,
    shape: [i64; 3],
    glu: bool,
    step_only: bool,
    mv_max: i64,
    mv_nt: i32,
    mode: MvSplit,
) -> Grouped {
    let [pairs, n_tok, n_expert] = shape;
    let down_only = matches!(mode, MvSplit::Down | MvSplit::ExpertDown);
    let per_expert = matches!(mode, MvSplit::Expert | MvSplit::ExpertDown);
    if step_only || mode == MvSplit::Step {
        return Grouped::One(pick_grouped_at(k, m, pairs, n_expert, mv_max, mv_nt));
    }
    if glu && down_only {
        // The fused gate/up as the default's.
        return Grouped::One(pick_grouped_at(k, m, pairs, n_expert, MV_MAX_N, mv_nt));
    }
    let per = pairs / pairs.min(n_expert).max(1);
    if !per_expert || n_tok < SPLIT_MIN_TOKENS || per > MV_MAX_N || mv_max < 1 {
        return Grouped::One(pick_grouped_at(k, m, pairs, n_expert, mv_max, mv_nt));
    }
    let cut = mv_max.min(i32::MAX as i64) as i32;
    let mv = Kernel::Mv {
        nt: tokens_tile(cut as i64, mv_nt),
        rtw: 1,
        kw: 8,
        rpt: GROUPED_MV_RPT,
    };
    // The busy experts: the step's GEMM (the fused gate/up's tile is `glu_gemm_tile`'s).
    let gemm = if glu {
        Kernel::Gemm {
            mi: 2,
            nt: glu_gemm_tile(k, n_expert, per).0,
        }
    } else {
        pick_grouped_at(k, m, pairs, n_expert, 0, mv_nt)
    };
    Grouped::Split { cut, mv, gemm }
}

/// Token tiles of a [`Grouped::Split`]'s launches at `cols` tokens per tile, an upper bound as
/// [`grouped_tiles`]'s: on the GEMM (`busy`) the full tiles and one partial tile per expert of
/// more than `cut` pairs; on the mv kernel `ceil(cut / cols)` tiles per expert of at most `cut`.
pub fn split_tiles(pairs: i64, cols: i64, n_expert: i64, cut: i32, busy: bool) -> i32 {
    let cut = cut as i64;
    let all = grouped_tiles(pairs, cols, n_expert, false) as i64;
    let bound = if busy {
        pairs / cols + n_expert.min(pairs / (cut + 1))
    } else {
        pairs.min(n_expert) * ((cut + cols - 1) / cols)
    };
    all.min(bound) as i32
}

/// A grouped projection over a short k (one with few blocks per row, like a down projection)
/// and many experts: the measured shape of the 256-row, two-CTA GEMM (#824).
fn wide_down(k: i64, n_expert: i64) -> bool {
    k <= 1024 && n_expert >= 256
}

/// A stage count and occupancy of the GEMM (`nvfp4_gemm`): `stages` cp.async stages of one
/// NVFP4 block (64 of k), registers limited for `ctas` CTAs per SM. Every variant gives the same
/// bits (each output is the same MMAs over k in the same order); the non-default ones are
/// compiled for grouped tiles of 64 and 128 tokens (#824).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmTune {
    pub stages: i32,
    pub ctas: i32,
}

impl GemmTune {
    /// The launch functions' `variant`.
    pub fn code(self) -> i32 {
        self.stages * 10 + self.ctas
    }

    /// The tune of a launch function's `variant`.
    pub fn from_code(v: i32) -> Self {
        Self {
            stages: v / 10,
            ctas: v % 10,
        }
    }
}

/// The GEMM's default: three stages, one CTA per SM.
pub const GEMM_DEFAULT: GemmTune = GemmTune { stages: 3, ctas: 1 };

/// Whether the GEMM has `tune` for tiles of `128 * mi` rows by `8 * nt` tokens: the default
/// always; the others at 64 and 128 tokens where `ctas` CTAs' stages fit an sm_120 / sm_121 SM's
/// 100 KiB of shared memory (`gemm_tuned` in `src/cuda/nvfp4_mm/common.cuh`).
pub fn gemm_tune_fits(mi: i32, nt: i32, tune: GemmTune) -> bool {
    if tune == GEMM_DEFAULT {
        return true;
    }
    // One stage: 48 bytes per weight row, 144 per token row; then x_off/d_off and the CTA's
    // reserved KiB.
    let smem = tune.stages * (128 * mi * 48 + 8 * nt * 144);
    (nt == 8 || nt == 16)
        && smem <= 99 * 1024
        && tune.ctas * (smem + 16 * 8 * nt + 16 + 1024) <= 100 * 1024
}

/// Two stages, two CTAs per SM: [`pick_grouped`]'s 256-row by 64-token down projection tile.
pub const GEMM_DOWN: GemmTune = GemmTune { stages: 2, ctas: 2 };

/// The variant of a grouped GEMM (`glu`: the fused SwiGLU's) of `128 * mi` rows by `8 * nt`
/// tokens over `k` on `n_expert` experts.
pub fn grouped_gemm_tune(k: i64, n_expert: i64, mi: i32, nt: i32, glu: bool) -> GemmTune {
    if !glu && mi == 2 && nt == 8 && wide_down(k, n_expert) {
        GEMM_DOWN
    } else {
        GEMM_DEFAULT
    }
}

/// Four stages, one CTA per SM: the grouped SwiGLU GEMM's tile of 64 tokens at up to
/// [`MV_MAX_N`] pairs per expert (the GEMM there only under [`set_expert_mv_max`]).
pub const GEMM_GLU_SHORT: GemmTune = GemmTune { stages: 4, ctas: 1 };

/// The grouped SwiGLU GEMM's token tile (`8 * nt` tokens) and its variant at `per` pairs per
/// expert. Up to [`MV_MAX_N`] pairs (short prompt steps, on the GEMM only under
/// [`set_expert_mv_max`]): 64-token tiles in four stages. GB10, Qwen3.8-Flash-Next's gate/up at
/// 256-1,536-token steps (`legion nvfp4-gemm-bench`, #836): 0.70-1.01 of the 32-token
/// three-stage tile's time, the best variant tried (or within 2% of it) where the tokens pick from 160 or all 512
/// experts (128-token tiles gain up to 14% more where they pick from 64). More pairs: as
/// before (#824).
pub fn glu_gemm_tile(k: i64, n_expert: i64, per: i64) -> (i32, GemmTune) {
    if per <= MV_MAX_N {
        return (8, GEMM_GLU_SHORT);
    }
    let nt = tokens_tile(per, 16).max(4);
    (nt, grouped_gemm_tune(k, n_expert, 2, nt, true))
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::c_void;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// `legion_nvfp4_mm` of `src/cuda/nvfp4_mm/common.cuh`.
    #[repr(C)]
    pub(super) struct Params {
        w: *const u8,
        x: *const u16,
        scale: *const f32,
        dst: *mut f32,
        k: i64,
        m: i64,
        n: i64,
        x_ld: i64,
        w_ld: i64,
        w_e: i64,
        tiles_m: i32,
        tiles_n: i32,
        group_m: i32,
        plan: *const i32,
        n_expert: i32,
        n_used: i32,
        n_x: i32,
        w2: *const u8,
        scale2: *const f32,
        act: i32,
        limit: f32,
        pdl: i32,
        sel_lo: i32,
        sel_hi: i32,
        unread_skips: i32,
    }

    /// `Params::sel_lo`, `sel_hi` for every expert.
    const ALL: (i32, i32) = (0, i32::MAX);

    unsafe extern "C" {
        fn legion_nvfp4_gemm_launch(
            p: *const Params,
            mi: i32,
            nt: i32,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_mv_launch(
            p: *const Params,
            nt: i32,
            rtw: i32,
            kw: i32,
            rpt: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_plan_launch(
            ids: *const i32,
            ids_ld: i64,
            rows: *const i32,
            skip_pad: i32,
            n_used: i32,
            n_tokens: i64,
            n_expert: i32,
            plan: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_swiglu_mv_launch(p: *const Params, nt: i32, stream: *mut c_void) -> i32;
        fn legion_nvfp4_swiglu_gemm_launch(
            p: *const Params,
            nt: i32,
            variant: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_swiglu_gemm_tma_launch(
            p: *const Params,
            nt: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_swiglu_gemm_tile_launch(
            p: *const Params,
            packed: *const u8,
            nt: i32,
            mode: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_nvfp4_mvg_launch(
            p: *const Params,
            nt: i32,
            rtw: i32,
            rpt: i32,
            sb: i32,
            pft: i32,
            stream: *mut c_void,
        ) -> i32;
        fn legion_cuda_sm_count() -> i32;
    }

    /// Dense GEMM tile order: CTAs walk the token tiles of `GROUP_M` row tiles at a time, so
    /// the CTAs resident at once (one per SM) share weight rows and activations in L2.
    const GROUP_M: i32 = 8;

    /// A kernel forced by [`super::force_kernel`] (tests and tuning), encoded; 0 is none.
    pub(super) static FORCED: AtomicU64 = AtomicU64::new(0);

    pub(super) fn encode(k: Kernel) -> u64 {
        match k {
            Kernel::Mv { nt, rtw, kw, rpt } => {
                1 | (nt as u64) << 8 | (rtw as u64) << 16 | (kw as u64) << 24 | (rpt as u64) << 32
            }
            Kernel::Gemm { mi, nt } => 2 | (nt as u64) << 8 | (mi as u64) << 16,
        }
    }

    fn forced() -> Option<Kernel> {
        let v = FORCED.load(Ordering::Relaxed);
        let f = |s: u32| ((v >> s) & 0xff) as i32;
        match v & 0xff {
            1 => Some(Kernel::Mv {
                nt: f(8),
                rtw: f(16),
                kw: f(24),
                rpt: f(32),
            }),
            2 => Some(Kernel::Gemm {
                mi: f(16),
                nt: f(8),
            }),
            _ => None,
        }
    }

    fn sms() -> i64 {
        // SAFETY: a device query.
        unsafe { legion_cuda_sm_count() as i64 }
    }

    fn launch(p: &mut Params, kernel: Kernel, stream: *mut c_void) -> Result<(), String> {
        let (rows, _) = kernel.tile();
        p.tiles_m = ((p.m + rows - 1) / rows) as i32;
        let tune = match kernel {
            Kernel::Gemm { mi, nt } if !p.plan.is_null() => forced_tune()
                .unwrap_or_else(|| grouped_gemm_tune(p.k, p.n_expert as i64, mi, nt, false)),
            _ => forced_tune().unwrap_or(GEMM_DEFAULT),
        };
        // SAFETY: parameters checked by the callers; the node's stream.
        let code = unsafe {
            match kernel {
                Kernel::Mv { nt, rtw, kw, rpt } => {
                    legion_nvfp4_mv_launch(p, nt, rtw, kw, rpt, stream)
                }
                Kernel::Gemm { mi, nt } => legion_nvfp4_gemm_launch(p, mi, nt, tune.code(), stream),
            }
        };
        match code {
            0 => Ok(()),
            -1 => Err(format!("nvfp4: no kernel {kernel:?} {tune:?}")),
            e => Err(format!("nvfp4 {kernel:?} {tune:?}: CUDA error {e}")),
        }
    }

    /// The GEMM variant forced by [`super::force_gemm_tune`] (`stages * 10 + ctas`), and the
    /// grouped SwiGLU GEMM's forced tokens per tile (`nt`); 0 is none.
    pub(super) static FORCED_TUNE: AtomicU64 = AtomicU64::new(0);
    pub(super) static FORCED_GLU_NT: AtomicU64 = AtomicU64::new(0);

    fn forced_tune() -> Option<GemmTune> {
        let v = FORCED_TUNE.load(Ordering::Relaxed) as i32;
        (v != 0).then(|| GemmTune::from_code(v))
    }

    fn check(k: i64, x_ld: i64, x: *const u16) -> Result<(), String> {
        if k % 64 != 0 || x_ld % 8 != 0 || !(x as usize).is_multiple_of(16) {
            return Err(format!("unsupported NVFP4 matmul: k = {k}, x_ld = {x_ld}"));
        }
        Ok(())
    }

    pub(super) unsafe fn mul_mat(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`, all tensors in device memory of the stream's device.
        let (w, x) = unsafe { (&*dst.src[0], &*dst.src[1]) };
        let (k, m, n) = (w.ne[0], w.ne[1], x.ne[1] * x.ne[2] * x.ne[3]);
        let x_ld = (x.nb[1] / x.nb[0]) as i64;
        check(k, x_ld, x.data as *const u16)?;
        let kernel = forced().unwrap_or_else(|| pick_dense(k, m, n, sms(), dense_mv_max_n()));
        let (_, cols) = kernel.tile();
        let mut p = Params {
            w: w.data as *const u8,
            x: x.data as *const u16,
            // SAFETY: src[2], when present, is the F32 [1] scale.
            scale: unsafe { dst.src[2].as_ref() }
                .map_or(std::ptr::null(), |s| s.data as *const f32),
            dst: dst.data as *mut f32,
            k,
            m,
            n,
            x_ld,
            w_ld: w.nb[1] as i64,
            w_e: 0,
            tiles_m: 0,
            tiles_n: ((n + cols - 1) / cols) as i32,
            group_m: GROUP_M,
            plan: std::ptr::null(),
            n_expert: 0,
            n_used: 0,
            n_x: 0,
            w2: std::ptr::null(),
            scale2: std::ptr::null(),
            act: dst.op_params[P_ACT],
            limit: 0.0,
            pdl: 0,
            sel_lo: ALL.0,
            sel_hi: ALL.1,
            unread_skips: 0,
        };
        launch(&mut p, kernel, stream)
    }

    pub(super) unsafe fn swiglu(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `swiglu`, all tensors in device memory of the stream's device.
        let (wg, wu, x) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2]) };
        let (k, m, n) = (wg.ne[0], wg.ne[1], x.ne[1] * x.ne[2] * x.ne[3]);
        let x_ld = (x.nb[1] / x.nb[0]) as i64;
        check(k, x_ld, x.data as *const u16)?;
        // SAFETY: the scales' sources as `swiglu` laid them out.
        let (sg, su) = unsafe { swiglu_scales(dst, 3) };
        let ptr =
            |s: Option<&ffi::ggml_tensor>| s.map_or(std::ptr::null(), |s| s.data as *const f32);
        // Decode batches: the mv kernel; more: the GEMM, 128 output rows per CTA.
        let (mv, nt) = if n <= dense_mv_max_n() {
            (true, tokens_tile(n, 8))
        } else {
            (false, tokens_tile(n, 16).max(4))
        };
        let rows = if mv { 16 } else { 128 };
        let p = Params {
            w: wg.data as *const u8,
            x: x.data as *const u16,
            scale: ptr(sg),
            dst: dst.data as *mut f32,
            k,
            m,
            n,
            x_ld,
            w_ld: wg.nb[1] as i64,
            w_e: 0,
            tiles_m: ((m + rows - 1) / rows) as i32,
            tiles_n: ((n + 8 * nt as i64 - 1) / (8 * nt as i64)) as i32,
            group_m: GROUP_M,
            plan: std::ptr::null(),
            n_expert: 0,
            n_used: 0,
            n_x: 0,
            w2: wu.data as *const u8,
            scale2: ptr(su),
            // The gate's activation (silu, or `GLU_GELU`).
            act: dst.op_params[P_ACT],
            limit: 0.0,
            pdl: 0,
            sel_lo: ALL.0,
            sel_hi: ALL.1,
            unread_skips: 0,
        };
        // SAFETY: parameters checked above; the node's stream.
        let code = unsafe {
            if mv {
                legion_nvfp4_swiglu_mv_launch(&p, nt, stream)
            } else {
                let tune = forced_tune().unwrap_or(GEMM_DEFAULT);
                legion_nvfp4_swiglu_gemm_launch(&p, nt, tune.code(), stream)
            }
        };
        match code {
            0 => Ok(()),
            e => Err(format!("nvfp4 swiglu: CUDA error {e}")),
        }
    }

    pub(super) unsafe fn mul_mat_id(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat_id`, all tensors in device memory of the stream's
        // device.
        let (w, x, ids) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2]) };
        let scale = (dst.op_params[P_SCALE] != 0).then(|| dst.src[3]);
        let (k, m, n_expert) = (w.ne[0], w.ne[1], w.ne[2]);
        let (n_used, n_tok) = (ids.ne[0], ids.ne[1]);
        let x_ld = (x.nb[1] / x.nb[0]) as i64;
        check(k, x_ld, x.data as *const u16)?;
        // The down matmul of a SwiGLU FFN reads the gate and up op's plan (the same routing).
        let shared = if dst.op_params[P_PLAN] != 0 {
            // SAFETY: `x` is a valid tensor; see `swiglu_plan`.
            unsafe { swiglu_plan(x, n_used, n_tok, n_expert) }
        } else {
            std::ptr::null_mut()
        };
        let plan = if shared.is_null() {
            let (out, _) = id_layout(m, n_used, n_tok, n_expert, dst.nb[0]);
            // SAFETY: the plan's I32s follow the result in the node (see `id_layout`).
            let plan = unsafe { (dst.data as *mut u8).add(out) } as *mut i32;
            // SAFETY: device pointers of the node's inputs and scratch.
            let code = unsafe {
                legion_nvfp4_plan_launch(
                    ids.data as *const i32,
                    (ids.nb[1] / 4) as i64,
                    std::ptr::null(),
                    0,
                    n_used as i32,
                    n_tok,
                    n_expert as i32,
                    plan,
                    stream,
                )
            };
            if code != 0 {
                return Err(format!("nvfp4 plan: CUDA error {code}"));
            }
            plan
        } else {
            shared
        };
        let pairs = n_used * n_tok;
        // On the plan of a SwiGLU with a row map, a spare tile for the padding pairs that plan
        // may skip (see `swiglu_id`); so for ids that may be -1 ([`skips`]).
        // SAFETY: a shared plan's `x` is a view of a `swiglu_id_full` node (`swiglu_plan`).
        let spare =
            skipping(dst) || (!shared.is_null() && unsafe { swiglu_rows(&*x.view_src) }.is_some());
        // A split (--nvfp4-expert-mv-split): the mv kernel for the experts of up to `cut` pairs
        // first (the kernel ahead stays the SwiGLU mv, see `pdl` below), then the GEMM.
        let (kernel, split) = match forced()
            .map(Grouped::One)
            .unwrap_or_else(|| plan_grouped(k, m, pairs, n_tok, n_expert, false, spare))
        {
            Grouped::One(kernel) => (kernel, None),
            Grouped::Split { cut, mv, gemm } => (mv, Some((cut, gemm))),
        };
        // A forced token tile (tests and tuning) on the SwiGLU's down mv, as on the SwiGLU.
        let forced_nt = forced_grouped_nt().filter(|_| !shared.is_null() && split.is_none());
        let kernel = match (kernel, forced_nt) {
            (
                Kernel::Mv {
                    nt,
                    rtw: 1,
                    kw: 8,
                    rpt,
                },
                Some(f),
            ) if rpt == GROUPED_MV_RPT => Kernel::Mv {
                nt: nt.max(f),
                rtw: 1,
                kw: 8,
                rpt,
            },
            (kernel, _) => kernel,
        };
        // The SwiGLU's down mv at one token per tile (or a forced tile): whole row tiles
        // prefetched (the same bits as the plain grouped mv).
        let tile_prefetch = !shared.is_null()
            && EXPERT_TILE_PREFETCH.load(Ordering::Relaxed)
            && matches!(kernel, Kernel::Mv { nt, rtw: 1, kw: 8, rpt }
                if rpt == GROUPED_MV_RPT && (nt == 1 || forced_nt.is_some()));
        let grouped = forced_grouped().or(tile_prefetch.then_some(DOWN_MV_TILE));
        let (_, cols) = kernel.tile();
        let mut p = Params {
            w: w.data as *const u8,
            x: x.data as *const u16,
            // SAFETY: src[3], when flagged, is the F32 [n_expert] scale.
            scale: scale.map_or(std::ptr::null(), |s| unsafe { (*s).data } as *const f32),
            dst: dst.data as *mut f32,
            k,
            m,
            n: pairs,
            x_ld,
            w_ld: w.nb[1] as i64,
            w_e: w.nb[2] as i64,
            tiles_m: 0,
            // The full tiles, plus at most one partial tile per expert (and the spare tile).
            tiles_n: match split {
                Some((cut, _)) => split_tiles(pairs, cols, n_expert, cut, false),
                None => grouped_tiles(pairs, cols, n_expert, spare),
            },
            group_m: GROUP_M,
            plan,
            n_expert: n_expert as i32,
            n_used: n_used as i32,
            n_x: x.ne[1] as i32,
            w2: std::ptr::null(),
            scale2: std::ptr::null(),
            act: dst.op_params[P_ACT],
            limit: 0.0,
            // On the SwiGLU's plan, the kernel ahead is the grouped SwiGLU mv (or its GEMM):
            // the mv starts while its last CTAs run (the same bits).
            pdl: (!shared.is_null() && EXPERT_PDL.load(Ordering::Relaxed)) as i32,
            sel_lo: ALL.0,
            sel_hi: split.map_or(ALL.1, |(cut, _)| cut),
            unread_skips: unread(dst) as i32,
        };
        if let Some(t) = forced_tiles() {
            p.tiles_n = p.tiles_n.min(t);
        }
        let first = match (kernel, grouped) {
            (Kernel::Mv { nt, .. }, Some(g)) => launch_grouped(&mut p, nt, g, stream),
            _ => launch(&mut p, kernel, stream),
        };
        let Some((cut, gemm)) = split else {
            return first;
        };
        first?;
        // The experts of more than `cut` pairs, after the mv in the stream.
        let (_, cols) = gemm.tile();
        p.tiles_n = split_tiles(pairs, cols, n_expert, cut, true);
        p.pdl = 0;
        (p.sel_lo, p.sel_hi) = (cut, ALL.1);
        launch(&mut p, gemm, stream)
    }

    /// The grouped mv kernel's variant `g` (`p.w2` set: the SwiGLU) at `8 * nt` tokens per tile.
    fn launch_grouped(
        p: &mut Params,
        nt: i32,
        g: GroupedMv,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // Rows per row tile: 16 * rtw, the SwiGLU's 8 (8 gate and 8 up rows).
        let tile = if p.w2.is_null() { 16 * g.rtw } else { 8 };
        let rows = (tile * g.rpt) as i64;
        p.tiles_m = ((p.m + rows - 1) / rows) as i32;
        // SAFETY: parameters checked by the callers; the node's stream.
        match unsafe { legion_nvfp4_mvg_launch(p, nt, g.rtw, g.rpt, g.sb, g.pft, stream) } {
            0 => Ok(()),
            -1 => Err(format!("nvfp4: no grouped mv kernel {g:?} at nt = {nt}")),
            e => Err(format!("nvfp4 grouped mv {g:?}: CUDA error {e}")),
        }
    }

    /// Variants forced by [`super::force_grouped_mv`] and [`super::force_glu_mv`], encoded; 0
    /// is none.
    pub(super) static FORCED_GROUPED: AtomicU64 = AtomicU64::new(0);
    pub(super) static FORCED_GLU: AtomicU64 = AtomicU64::new(0);

    pub(super) fn encode_grouped(g: Option<GroupedMv>) -> u64 {
        g.map_or(0, |g| {
            1 | (g.rtw as u64) << 8
                | (g.rpt as u64) << 16
                | (g.sb as u64) << 24
                | (g.pft as u64) << 32
        })
    }

    fn decode_grouped(v: u64) -> Option<GroupedMv> {
        let f = |s: u32| ((v >> s) & 0xff) as i32;
        (v & 0xff == 1).then(|| GroupedMv {
            rtw: f(8),
            rpt: f(16),
            sb: f(24),
            pft: f(32),
        })
    }

    fn forced_grouped() -> Option<GroupedMv> {
        decode_grouped(FORCED_GROUPED.load(Ordering::Relaxed))
    }

    /// The grouped decode mvs' least token tile and their token tiles' cap, forced by
    /// [`super::force_grouped_tiles`]; 0 is none.
    pub(super) static FORCED_GROUPED_NT: AtomicU64 = AtomicU64::new(0);
    pub(super) static FORCED_TILES: AtomicU64 = AtomicU64::new(0);

    fn forced_grouped_nt() -> Option<i32> {
        let v = FORCED_GROUPED_NT.load(Ordering::Relaxed) as i32;
        (v != 0).then_some(v)
    }

    fn forced_tiles() -> Option<i32> {
        let v = FORCED_TILES.load(Ordering::Relaxed) as i32;
        (v != 0).then_some(v)
    }

    pub(super) unsafe fn swiglu_id(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `swiglu_id`, all tensors in device memory of the stream's
        // device.
        let (wg, wu, x, ids) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]) };
        // SAFETY: the scales' sources as `swiglu_id` laid them out.
        let (sg, su) = unsafe { swiglu_scales(dst, 4) };
        let ptr =
            |s: Option<&ffi::ggml_tensor>| s.map_or(std::ptr::null(), |s| s.data as *const f32);
        let (k, m, n_expert) = (wg.ne[0], wg.ne[1], wg.ne[2]);
        let (n_used, n_tok) = (ids.ne[0], ids.ne[1]);
        let x_ld = (x.nb[1] / x.nb[0]) as i64;
        check(k, x_ld, x.data as *const u16)?;
        let (out, _) = id_layout(m, n_used, n_tok, n_expert, 2);
        // SAFETY: the plan's I32s follow the result in the node (see `id_layout`).
        let plan = unsafe { (dst.data as *mut u8).add(out) } as *mut i32;
        // SAFETY: a node made by `swiglu_id_full`.
        let rows = unsafe { swiglu_rows(dst) }.map_or(std::ptr::null(), |r| r.data as *const i32);
        let pairs = n_used * n_tok;
        // Decode batches (the grouped mv's): the grouped SwiGLU mv, the tokens per tile of the
        // plain one's. More: the GEMM, 128 output rows (128 gate and 128 up rows) per CTA,
        // tokens per tile as the grouped GEMM's. A split (--nvfp4-expert-mv-split): both, the
        // experts of up to `cut` pairs on the mv. A row map's plan and a clamp keep the
        // step-wide choice (the padding pairs and the clamp are the mv's or the GEMM's alone).
        let limit = f32::from_bits(dst.op_params[P_LIMIT] as u32);
        // Ids that may be -1 ([`skips`]): the step-wide choice too, and a spare tile below.
        let skips = skipping(dst);
        let step_only = !rows.is_null() || limit > 0.0 || skips;
        let (mv, split) = match plan_grouped(k, m, pairs, n_tok, n_expert, true, step_only) {
            Grouped::One(Kernel::Mv { nt, .. }) => (Some(nt), None),
            Grouped::One(Kernel::Gemm { .. }) => (None, None),
            Grouped::Split { cut, mv, .. } => (Some((mv.tile().1 / 8) as i32), Some(cut)),
        };
        // A forced token tile (tests and tuning): the down mv on this plan takes it too.
        let forced_mv_nt = forced_grouped_nt().filter(|_| split.is_none() && mv.is_some());
        let mv = mv.map(|nt| forced_mv_nt.map_or(nt, |f| nt.max(f)));
        // Padding rows skipped: on the mv kernels only (the down mv on this plan is one too:
        // the same pair count and experts, unless a test forces another kernel).
        let skip = !rows.is_null()
            && mv.is_some()
            && forced().is_none()
            && PAD_SKIP.load(Ordering::Relaxed);
        // SAFETY: device pointers of the node's inputs and scratch.
        let code = unsafe {
            legion_nvfp4_plan_launch(
                ids.data as *const i32,
                (ids.nb[1] / 4) as i64,
                rows,
                skip as i32,
                n_used as i32,
                n_tok,
                n_expert as i32,
                plan,
                stream,
            )
        };
        if code != 0 {
            return Err(format!("nvfp4 plan: CUDA error {code}"));
        }
        let per = pairs / pairs.min(n_expert).max(1);
        let forced_nt = FORCED_GLU_NT.load(Ordering::Relaxed) as i32;
        let (gemm_nt, gemm_tune) = glu_gemm_tile(k, n_expert, per);
        let g_nt = if forced_nt != 0 { forced_nt } else { gemm_nt };
        let nt = mv.unwrap_or(g_nt);
        let cols = 8 * nt as i64;
        let tune_at = |nt: i32| {
            forced_tune().unwrap_or(if nt == gemm_nt {
                gemm_tune
            } else {
                grouped_gemm_tune(k, n_expert, 2, nt, true)
            })
        };
        let mut p = Params {
            w: wg.data as *const u8,
            x: x.data as *const u16,
            scale: ptr(sg),
            dst: dst.data as *mut f32,
            k,
            m,
            n: pairs,
            x_ld,
            w_ld: wg.nb[1] as i64,
            w_e: wg.nb[2] as i64,
            tiles_m: ((m + 127) / 128) as i32,
            // The full tiles, plus at most one partial tile per expert (and the skipped
            // pairs' spare tile).
            tiles_n: match split {
                Some(cut) => split_tiles(pairs, cols, n_expert, cut, false),
                None => grouped_tiles(pairs, cols, n_expert, skip || skips),
            },
            group_m: GROUP_M,
            plan,
            n_expert: n_expert as i32,
            n_used: n_used as i32,
            n_x: x.ne[1] as i32,
            w2: wu.data as *const u8,
            scale2: ptr(su),
            act: 0,
            limit,
            pdl: 0,
            sel_lo: ALL.0,
            sel_hi: split.unwrap_or(ALL.1),
            unread_skips: unread(dst) as i32,
        };
        if let Some(cut) = split {
            // The experts of more than `cut` pairs on the GEMM first, so the grouped SwiGLU mv
            // stays the kernel ahead of the down mv (its programmatic dependent launch).
            let mv_tiles = p.tiles_n;
            p.tiles_n = split_tiles(pairs, 8 * g_nt as i64, n_expert, cut, true);
            (p.sel_lo, p.sel_hi) = (cut, ALL.1);
            glu_gemm(&p, g_nt, tune_at(g_nt), stream)?;
            p.tiles_n = mv_tiles;
            (p.sel_lo, p.sel_hi) = (ALL.0, cut);
        }
        if let Some(nt) = mv {
            if let Some(t) = forced_tiles() {
                p.tiles_n = p.tiles_n.min(t);
            }
            let g = decode_grouped(FORCED_GLU.load(Ordering::Relaxed)).unwrap_or(
                if (nt == 1 || forced_mv_nt.is_some())
                    && EXPERT_TILE_PREFETCH.load(Ordering::Relaxed)
                {
                    GLU_MV_TILE
                } else {
                    GLU_MV
                },
            );
            return launch_grouped(&mut p, nt, g, stream);
        }
        if let Some(t) = forced_tiles().filter(|_| split.is_none()) {
            p.tiles_n = p.tiles_n.min(t);
        }
        glu_gemm(&p, nt, tune_at(nt), stream)
    }

    /// The grouped SwiGLU GEMM at `8 * nt` tokens per tile.
    fn glu_gemm(p: &Params, nt: i32, tune: GemmTune, stream: *mut c_void) -> Result<(), String> {
        let tiled = EXPERT_TILED.load(Ordering::Relaxed);
        if tune == GEMM_DEFAULT && tiled != 0 {
            let mode = EXPERT_TILED_MODE.load(Ordering::Relaxed);
            // SAFETY: as below, `tiled` a device copy of these experts (the micro harness's);
            // -1 (nothing launched) where the kernel does not apply.
            match unsafe {
                legion_nvfp4_swiglu_gemm_tile_launch(p, tiled as *const u8, nt, mode, stream)
            } {
                0 => return Ok(()),
                -1 => {}
                e => return Err(format!("nvfp4 swiglu_id (tiled): CUDA error {e}")),
            }
        }
        if tune == GEMM_DEFAULT && EXPERT_TMA.load(Ordering::Relaxed) {
            // SAFETY: as below; -1 (nothing launched) where the TMA kernel does not apply.
            match unsafe { legion_nvfp4_swiglu_gemm_tma_launch(p, nt, stream) } {
                0 => return Ok(()),
                -1 => {}
                e => return Err(format!("nvfp4 swiglu_id (TMA): CUDA error {e}")),
            }
        }
        // SAFETY: parameters checked by the caller; the node's stream.
        match unsafe { legion_nvfp4_swiglu_gemm_launch(p, nt, tune.code(), stream) } {
            0 => Ok(()),
            -1 => Err(format!("nvfp4 swiglu_id: no GEMM at nt = {nt} {tune:?}")),
            e => Err(format!("nvfp4 swiglu_id: CUDA error {e}")),
        }
    }
}

/// Launches every NVFP4 matmul from now on with `kernel` (where it fits), or the default
/// choice for `None`: for tests and tuning.
#[doc(hidden)]
pub fn force_kernel(kernel: Option<Kernel>) {
    #[cfg(feature = "cuda")]
    cuda::FORCED.store(
        kernel.map_or(0, cuda::encode),
        std::sync::atomic::Ordering::Relaxed,
    );
    #[cfg(not(feature = "cuda"))]
    let _ = kernel;
}

/// Launches every NVFP4 GEMM from now on with the stages and occupancy `tune` (where it is
/// compiled; see [`GemmTune`]), and every prefill [`swiglu_id`] with tiles of `8 * glu_nt`
/// tokens; `None`: the defaults. For tests and tuning (the same bits either way).
#[doc(hidden)]
pub fn force_gemm_tune(tune: Option<GemmTune>, glu_nt: Option<i32>) {
    #[cfg(feature = "cuda")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        cuda::FORCED_TUNE.store(tune.map_or(0, |t| t.code() as u64), Relaxed);
        cuda::FORCED_GLU_NT.store(glu_nt.map_or(0, |n| n as u64), Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (tune, glu_nt);
}

/// Launches every grouped mv [`mul_mat_id`] from now on as the variant `g`, or the default for
/// `None`: for tests and tuning (the same bits either way).
#[doc(hidden)]
pub fn force_grouped_mv(g: Option<GroupedMv>) {
    #[cfg(feature = "cuda")]
    cuda::FORCED_GROUPED.store(
        cuda::encode_grouped(g),
        std::sync::atomic::Ordering::Relaxed,
    );
    #[cfg(not(feature = "cuda"))]
    let _ = g;
}

/// Launches every decode [`swiglu_id`] from now on as the variant `g`, or [`GLU_MV`] for `None`:
/// for tests and tuning (the same bits either way).
#[doc(hidden)]
pub fn force_glu_mv(g: Option<GroupedMv>) {
    #[cfg(feature = "cuda")]
    cuda::FORCED_GLU.store(
        cuda::encode_grouped(g),
        std::sync::atomic::Ordering::Relaxed,
    );
    #[cfg(not(feature = "cuda"))]
    let _ = g;
}

/// Launches every decode [`swiglu_id`] on the grouped mv, and the down [`mul_mat_id`] on its
/// plan, from now on with token tiles of at least `8 * nt` tokens (1, 2 or 4; the whole-tile
/// prefetch variants where [`set_expert_tile_prefetch`] holds), and every grouped mv launch (and
/// unsplit grouped GEMM launch) with at most `tiles` token tiles (the caller's exact count at
/// the launch's own tile, for measuring the CTAs that find no tile); `None`: the defaults. For tests and tuning (the same bits: a token's sums do not
/// depend on its tile's other tokens).
#[doc(hidden)]
pub fn force_grouped_tiles(nt: Option<i32>, tiles: Option<i32>) {
    #[cfg(feature = "cuda")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        cuda::FORCED_GROUPED_NT.store(nt.map_or(0, |n| n as u64), Relaxed);
        cuda::FORCED_TILES.store(tiles.map_or(0, |n| n as u64), Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (nt, tiles);
}

pub mod bench;
pub mod dense_bench;
pub mod gemm_bench;

#[cfg(test)]
mod tests;
