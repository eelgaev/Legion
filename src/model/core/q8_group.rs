//! Q8_0 matvecs that read one input, quantized once (#819): `w_i @ x` for 2 to [`MAX_WEIGHTS`]
//! Q8_0 weights and one F32 input `x` of 1 to [`MAX_TOKENS`] columns, as one external op
//! ([`crate::ggml::ExternalOp`], `src/cuda/q8_group.cu`) of two kernels. ggml-cuda runs each of
//! these matmuls as its own `quantize_q8_1` + `mul_mat_vec_q` pair, so a DeltaNet layer's input
//! is quantized twice (`attn_qkv`, `attn_gate`), an attention layer's three times (`attn_q`,
//! `attn_k`, `attn_v`) and a shared expert's twice (`ffn_gate_shexp`, `ffn_up_shexp`), each to the
//! same bytes, and every matmul is a launch of its own. On a V100, at C=1 decode with MTP, those
//! launches are a few microseconds each against ~25-50 us matvecs.
//!
//! `x` is quantized once, as ggml's `quantize_q8_1` does (one thread a value, ggml's warp
//! reductions and rounding), by an op of its own ([`quantized`]: one node per input, shared by
//! every group that reads it, or the Q8_1 blocks qwen4exp's hyper-connection mixer writes beside
//! its output, [`set_quantized`]), then the op runs ggml's
//! `mul_mat_vec_q<Q8_0, n>` without fusion for every weight in one launch: ggml's warps and rows
//! per CTA for compute capability 7.x-8.x, its K loop, its `vec_dot_q8_0_q8_1` (vecdotq.cuh) and
//! its cross-warp and warp sums, so every output has ggml's bits (mmvq.cu, quantize.cu, MIT).
//! ggml's row split does not depend on the other rows, so CTAs of several weights side by side
//! compute what each weight's own launch would.
//!
//! Taken on NVIDIA compute capability 7.x and 8.x and on a GB10 ([`device_supported`]; #311:
//! ggml's GB10 mmvq is the same launch at 2-8 columns plus an L2 prefetch of the weights, which
//! the kernels issue too, and a programmatic dependent launch, which the op's launches are there;
//! one column, where ggml widens the CTA, stays on ggml, [`device_takes_cols`], and so do the
//! hyper-connection mixer's parts, [`GB10_NO_PARTS`]; on a GB10 only with `--q8-group`
//! ([`Mode::All`]): its det decode was flat, so the default leaves it on ggml; #994: the default's
//! parts there are bitwise but flat too, +0.4% at C=1 and +0.8% at C=4 under `--deterministic`, and
//! `--q8-group`'s one DeltaNet qkv-and-gate node moves the greedy text once a verify step is 8
//! tokens or fewer), inside stage graphs only (between ggml's CUDA graphs each op would add a
//! graph split), where every weight has an even row count (ggml's
//! two rows per CTA from two columns) and, at one column, where ggml would not take its
//! small-K variant. A gate and up pair with its SwiGLU ([`swiglu`]) is one launch too, the
//! gate's and up's rows of a CTA together and `up * silu(gate)` written, as ggml's fused matvec
//! does at one column and its SwiGLU op after the two matvecs does at more.
//!
//! Every kernel has ggml's bits, on its own (`legion q8-group-check`) and in the model: with
//! `--stable-fusions` Qwen3.8-Flash-Next's greedy text with all parts equals ggml's matmuls'
//! (#819 round 3). Without it, one node for a DeltaNet layer's qkv and gate moves the text all
//! the same: qkv and z as views of it, z read late by the gated norm, z's memory is taken from
//! the qkv matmul on, which changes where ggml-alloc puts the layer's later tensors, and
//! ggml-cuda turns its MoE fusions on or off by whether their outputs overlap their inputs
//! (`ggml_cuda_check_fusion_memory_ranges`), which `--stable-fusions` pins. So those two take a
//! node each ([`mul_mat_each`], round 4), sized and placed as ggml's two matmuls, both reading
//! the Q8_1 blocks the hyper-connection mixer wrote: one quantize launch fewer per matmul and
//! ggml's layout. A MoE layer's routed gate and up read the mixer's blocks too
//! ([`PART_EXPERTS`], round 5): Legion's expert SwiGLU op quantizes its input with ggml's
//! `quantize_q8_1` arithmetic, so it skips that launch and keeps its scratch in the graph, unread,
//! for the layout. A shared expert's SwiGLU writes its output's Q8_1 blocks for the down
//! projection ([`swiglu_down`], [`PART_SHEXP_DOWN`], round 6): to a per-layer slot of device
//! memory outside the graph, so both nodes keep the sizes and inputs of [`swiglu`]'s and ggml's
//! `mul_mat`. By default ([`Mode::Default`]) qwen4exp, the architecture whose text was
//! checked, runs every part ([`DEFAULT_PARTS`]); `legion serve --q8-group` runs all parts on
//! every architecture (the DeltaNet pair as one node where no Q8_1 node exists),
//! `--no-q8-group` none. The CPU
//! implementation is the reference (the same quantization, F32 sums of each block's products
//! in row order).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static Q8_GROUP: ExternalOp = ExternalOp {
    name: "q8_group",
    cpu: cpu_q8_group,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_group),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static Q8_QUANTIZE: ExternalOp = ExternalOp {
    name: "q8_quantize",
    cpu: cpu_q8_quantize,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_q8_quantize),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The [`Context::shared_node`] key of an input's Q8_1 blocks ([`quantized`]).
const Q8_1_KEY: &str = "q8_group.q8_1";

/// Most weights one op reads (all with the same input).
pub const MAX_WEIGHTS: usize = 4;

/// Fewest weights the op takes: with one it would only move ggml's pair of launches.
pub const MIN_WEIGHTS: usize = 2;

/// Most columns of `x` (ggml-cuda's `MMVQ_MAX_BATCH_SIZE`; more go to MMQ).
pub const MAX_TOKENS: i64 = 8;

/// Values per Q8_0 / Q8_1 block.
const GROUP: i64 = 32;

/// Bytes of a Q8_0 block: an F16 scale, 32 int8.
const Q8_0_BYTES: usize = 34;

/// Bytes of a Q8_1 block: F16 scale and sum, 32 int8.
const Q8_1_BYTES: usize = 36;

/// ggml's mmvq blocks per warp and K step for Q8_0 (`vdr * warp_size / qi` = 2 * 32 / 8): with
/// fewer blocks per row than its CTA's warps take in one step, ggml runs one column on its
/// small-K variant (more rows per CTA), which the op does not have.
const BLOCKS_PER_WARP_STEP: i64 = 8;

/// Which graphs take the op ([`set_q8_group`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// [`DEFAULT_PARTS`], on the architectures whose greedy text was checked with them.
    Default = 0,
    /// Every part, on every architecture (`--q8-group`).
    All = 1,
    /// None (`--no-q8-group`).
    Off = 2,
}

/// The [`Mode`] graphs built from now on take.
static MODE: AtomicU8 = AtomicU8::new(Mode::Default as u8);

/// The parts of the op that run ([`set_parts`]), [`NO_MASK`] when not set: bits of
/// [`PART_DELTA`] and the others.
static PARTS: AtomicU8 = AtomicU8::new(NO_MASK);

/// [`PARTS`] not set: [`Mode`]'s parts run.
const NO_MASK: u8 = u8::MAX;

/// A DeltaNet layer's qkv and gate: one [`mul_mat_each`] node each where the input's Q8_1
/// blocks are a node already, else (under [`Mode::All`] only) one [`mul_mat`] node.
pub const PART_DELTA: u8 = 1;
/// A DeltaNet layer's beta and alpha in the same launch as its qkv and gate (Q8_0 beta and
/// alpha only: Qwen3.8-Flash-Next's UD GGUFs keep them in F32, so it never runs there).
pub const PART_DELTA_AB: u8 = 2;
/// An attention layer's Q, K and V.
pub const PART_ATTN: u8 = 4;
/// A shared expert's gate and up with its SwiGLU.
pub const PART_SHEXP: u8 = 8;
/// The hyper-connection mixer's Q8_1 blocks of the block input.
pub const PART_HC: u8 = 16;
/// A MoE layer's routed experts' gate and up (Legion's expert SwiGLU op) reading the mixer's
/// Q8_1 blocks of their input instead of quantizing it again
/// ([`crate::model::core::expert_mv::glu_tiered_preq`]).
pub const PART_EXPERTS: u8 = 32;

/// A shared expert's SwiGLU ([`PART_SHEXP`]) writing its output's Q8_1 blocks for the down
/// projection, which reads them instead of quantizing again ([`swiglu_down`], round 6).
pub const PART_SHEXP_DOWN: u8 = 64;

/// The parts [`Mode::Default`] runs: all of them, [`PART_DELTA`] as one node per result
/// ([`mul_mat_each`], see the module docs).
pub const DEFAULT_PARTS: u8 =
    PART_DELTA | PART_DELTA_AB | PART_ATTN | PART_SHEXP | PART_HC | PART_EXPERTS | PART_SHEXP_DOWN;

/// Slots of [`swiglu_down`]'s Q8_1 blocks on each GPU (`src/cuda/q8_group.cu`'s device
/// globals): one per layer (its index modulo this), so that two layers' pairs never share one.
pub const SLOTS: usize = 64;

/// Q8_1 blocks a slot holds: `rows / 32` for each of the SwiGLU's columns.
pub const SLOT_BLOCKS: i64 = 256;

/// 32-row blocks of a SwiGLU output a slot counts (one counter each).
pub const SLOT_ROW_BLOCKS: i64 = 64;

/// Node kinds ([`crate::ggml::external_params`]'s first value) of the [`Q8_GROUP`] op.
const KIND_GROUP: i32 = 0;
/// [`swiglu`].
const KIND_SWIGLU: i32 = 1;
/// [`swiglu_down`]'s SwiGLU: its Q8_1 blocks written to the slot too.
const KIND_SWIGLU_Q8: i32 = 2;
/// [`swiglu_down`]'s down projection: its input's Q8_1 blocks read from the slot.
const KIND_SLOT_IN: i32 = 3;

/// Sets which parts of the op run (`legion serve --q8-group-parts`, a diagnostic): in place of
/// [`Mode`]'s, which still decides the architectures.
pub fn set_parts(mask: u8) {
    PARTS.store(mask, Ordering::Relaxed);
}

/// Sets whether graphs built from now on run the Q8_0 matvecs of one input on the op where it
/// applies: `true` every part on every architecture (`legion serve --q8-group`), `false` none
/// (`--no-q8-group`). Without a call, [`Mode::Default`].
pub fn set_q8_group(on: bool) {
    set_mode(if on { Mode::All } else { Mode::Off });
}

/// Sets the [`Mode`] graphs built from now on take.
pub fn set_mode(mode: Mode) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

/// The [`Mode`] graphs built now take.
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        1 => Mode::All,
        2 => Mode::Off,
        _ => Mode::Default,
    }
}

/// Whether part `bit` runs in a model of architecture `arch` ([`crate::model::core::Arch`]'s
/// name) under `mode` with the [`set_parts`] mask `mask` ([`NO_MASK`]: none).
pub fn part_in(mode: Mode, mask: u8, arch: &str, bit: u8) -> bool {
    let parts = match mode {
        Mode::Off => 0,
        Mode::All => u8::MAX,
        Mode::Default if arch == "qwen4exp" => DEFAULT_PARTS,
        Mode::Default => 0,
    };
    let parts = if mask == NO_MASK || mode == Mode::Off || parts == 0 {
        parts
    } else {
        mask
    };
    parts & bit != 0
}

/// Whether part `bit` of the op runs in a model of architecture `arch` ([`part_in`] with the
/// current [`Mode`] and mask), on a GB10 without [`GB10_NO_PARTS`].
pub fn part(arch: &str, bit: u8) -> bool {
    part_in(mode(), PARTS.load(Ordering::Relaxed), arch, bit)
        && !((bit & GB10_NO_PARTS) != 0 && gb10())
}

/// The parts a GB10 never runs (#311): its hyper-connection mixer has kernels of its own, which
/// do not write the block input's Q8_1 blocks ([`PART_HC`]), so neither can the routed experts
/// read them ([`PART_EXPERTS`]); a DeltaNet layer's qkv and gate then take the op only as one
/// node ([`PART_DELTA`] under [`Mode::All`]).
pub const GB10_NO_PARTS: u8 = PART_HC | PART_EXPERTS;

/// Whether the current GPU is a GB10 (compute capability 12.1), whose ggml mmvq the op copies
/// at 2-8 columns ([`device_takes_cols`]).
pub fn gb10() -> bool {
    #[cfg(feature = "cuda")]
    {
        launch::compute_capability() == Some(121)
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether the op runs `n` columns as ggml's mmvq does on a GPU that is a GB10 (`gb10`) or not:
/// on a GB10, 2 to 8 (ggml's one-column launch there takes a wider CTA when that halves its K
/// loop, `MMVQ_PARAMETERS_GB10`, which the op does not have); elsewhere 1 to 8.
pub fn cols_on(gb10: bool, n: i64) -> bool {
    (1..=MAX_TOKENS).contains(&n) && (n > 1 || !gb10)
}

/// [`cols_on`] for the current GPU.
pub fn device_takes_cols(n: i64) -> bool {
    cols_on(gb10(), n)
}

/// ggml-cuda's warps per CTA for a Q8_0 mmvq of `n` columns (compute capability 7.x-8.x).
pub fn ggml_warps(n: i64) -> i64 {
    if n <= 4 { 4 } else { 2 }
}

/// ggml-cuda's rows per CTA for a Q8_0 mmvq of `n` columns (without its small-K variant).
pub fn ggml_rows_per_block(n: i64) -> i64 {
    if n == 1 { 1 } else { 2 }
}

/// Whether the op computes `w @ x` as ggml-cuda's mmvq would: a Q8_0 matrix `w` (`[k, rows]`,
/// whole blocks between rows, `rows` a multiple of [`ggml_rows_per_block`]) and an F32 `x`
/// (`[k, n]`, values contiguous) of 1 to [`MAX_TOKENS`] columns, `k` a multiple of 32; one
/// column only where ggml would not take its small-K variant.
pub fn fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [xk, n, x2, x3] = x.ne();
    let xnb = x.nb();
    x.ty() == types::F32
        && x2 == 1
        && x3 == 1
        && weight_fits(w, xk, n)
        && xnb[0] == 4
        && xnb[1].is_multiple_of(4)
}

/// [`fits`]'s rules for the weight `w`, read with an input of `n` columns of `k` values.
fn weight_fits(w: Tensor<'_>, k: i64, n: i64) -> bool {
    let [wk, rows, w2, w3] = w.ne();
    let wnb = w.nb();
    w.ty() == types::Q8_0
        && w2 == 1
        && w3 == 1
        && wk == k
        && k > 0
        && k % GROUP == 0
        && k <= i32::MAX as i64
        && rows > 0
        && rows <= i32::MAX as i64
        && (1..=MAX_TOKENS).contains(&n)
        && rows % ggml_rows_per_block(n) == 0
        && (n > 1 || k / GROUP >= ggml_warps(1) * BLOCKS_PER_WARP_STEP)
        && wnb[0] == Q8_0_BYTES
        && wnb[1].is_multiple_of(Q8_0_BYTES)
        && wnb[1] >= k as usize / GROUP as usize * Q8_0_BYTES
}

/// Whether `w @ x` for every weight of `ws` ([`MIN_WEIGHTS`] to [`MAX_WEIGHTS`]) takes one op:
/// [`enabled`], each [`fits`], weights in one GPU's memory.
pub fn takes(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    fits_all(ws, x)
        && ws.iter().all(|w| !w.is_host() && !w.is_meta())
        && enabled()
        && device_takes_cols(x.ne()[1])
}

/// Whether graphs built now may take the op where the shapes fit (and [`part`] runs): not
/// [`Mode::Off`], inside stage graphs, and a GPU of compute capability 7.x or 8.x, or a GB10
/// under [`Mode::All`] ([`device_supported`], [`mode_on_device`]).
pub fn enabled() -> bool {
    cfg!(feature = "cuda")
        && crate::ggml::stage_graph::enabled()
        && device_supported()
        && mode_on_device(mode(), gb10())
}

/// Whether `mode` runs the op on a device (`gb10` or not): never under [`Mode::Off`], and on a
/// GB10 only under [`Mode::All`] (`--q8-group`; #311 and #994: the default's parts are bitwise
/// there, but det decode was flat, so the default stays on ggml).
pub fn mode_on_device(mode: Mode, gb10: bool) -> bool {
    match mode {
        Mode::Off => false,
        Mode::All => true,
        Mode::Default => !gb10,
    }
}

/// Whether [`fits`] takes every weight of `ws` ([`MIN_WEIGHTS`] to [`MAX_WEIGHTS`]) with `x`.
pub fn fits_all(ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    (MIN_WEIGHTS..=MAX_WEIGHTS).contains(&ws.len()) && ws.iter().all(|&w| fits(w, x))
}

/// Whether the current GPU runs the op as ggml's mmvq: NVIDIA compute capability 7.x or 8.x
/// (Volta to Ada, whose Q8_0 mmvq launch it copies), or a GB10 (12.1, #311: the same launch at
/// 2-8 columns, with ggml's L2 prefetch, [`device_takes_cols`]).
pub fn device_supported() -> bool {
    #[cfg(feature = "cuda")]
    {
        launch::compute_capability().is_some_and(|cc| (70..90).contains(&cc) || cc == 121)
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Bytes of the quantized input of `n` columns of `k` values (Q8_1 blocks, `k / 32` a column,
/// columns one after the other).
pub fn quantized_bytes(k: i64, n: i64) -> usize {
    (k / GROUP * n) as usize * Q8_1_BYTES
}

/// The F32 elements of a node holding [`quantized_bytes`].
pub fn quantized_elements(k: i64, n: i64) -> i64 {
    quantized_bytes(k, n).div_ceil(4) as i64
}

/// `x` (F32 `[k, n]`, as [`fits`] takes it) as ggml's `quantize_q8_1` writes it for its mmvq:
/// a node of [`quantized_elements`] F32 holding the Q8_1 blocks, made once per `x` in `ctx`.
pub fn quantized<'a>(ctx: &'a Context, x: Tensor<'a>) -> Tensor<'a> {
    let (k, n) = (x.ne()[0], x.ne()[1]);
    ctx.shared_node(x, Q8_1_KEY, || {
        ctx.external(
            &Q8_QUANTIZE,
            types::F32,
            [quantized_elements(k, n), 1, 1, 1],
            &[x],
        )
    })
}

/// Whether `x`'s Q8_1 blocks are a node in `ctx` already ([`quantized`], [`set_quantized`]).
pub fn has_quantized(ctx: &Context, x: Tensor<'_>) -> bool {
    ctx.has_shared_node(x, Q8_1_KEY)
}

/// Makes `q` what [`quantized`] returns for `x` from now on: an op that writes `x`'s Q8_1
/// blocks itself, the bits `quantize_q8_1` gives (the caller's promise), so the quantization
/// is not a launch of its own.
pub fn set_quantized<'a>(ctx: &'a Context, x: Tensor<'a>, q: Tensor<'a>) {
    let (k, n) = (x.ne()[0], x.ne()[1]);
    assert!(
        q.ty() == types::F32 && q.elements() as i64 == quantized_elements(k, n),
        "q8_group: Q8_1 blocks of {k} x {n}"
    );
    ctx.set_shared_node(x, Q8_1_KEY, q);
}

/// `w @ x` for each weight of `ws` (each [`fits`] with `x`, [`MIN_WEIGHTS`] to [`MAX_WEIGHTS`])
/// in one op: F32 `[rows_i, n]` results, contiguous views of one node.
pub fn mul_mat<'a>(ctx: &'a Context, ws: &[Tensor<'a>], x: Tensor<'a>) -> Vec<Tensor<'a>> {
    assert!(fits_all(ws, x), "q8_group: shapes");
    let n = x.ne()[1];
    let mut args = vec![x, quantized(ctx, x)];
    args.extend_from_slice(ws);
    let total: i64 = ws.iter().map(|w| w.ne()[1]).sum();
    let node = ctx.external_with(
        &Q8_GROUP,
        types::F32,
        [total * n, 1, 1, 1],
        &args,
        &[KIND_GROUP],
    );
    let mut off = 0;
    ws.iter()
        .map(|w| {
            let m = w.ne()[1];
            let v = ctx.view_2d(node, [m, n], m as usize * 4, off);
            off += (m * n) as usize * 4;
            v
        })
        .collect()
}

/// Whether [`mul_mat_each`] takes `ws` (1 to [`MAX_WEIGHTS`] weights) with `x`: each [`fits`]
/// with `x`, in one GPU's memory, [`enabled`], and `x`'s Q8_1 blocks a node of `ctx` already
/// ([`has_quantized`]: a quantization node of their own would change the graph's layout).
pub fn takes_each(ctx: &Context, ws: &[Tensor<'_>], x: Tensor<'_>) -> bool {
    (1..=MAX_WEIGHTS).contains(&ws.len())
        && ws
            .iter()
            .all(|&w| fits(w, x) && !w.is_host() && !w.is_meta())
        && enabled()
        && device_takes_cols(x.ne()[1])
        && has_quantized(ctx, x)
}

/// `w @ x` for each weight of `ws` ([`takes_each`]) as an op node of its own, reading `x`'s
/// Q8_1 blocks: F32 `[rows_i, n]`, ggml's mmvq bits without its quantize launch.
///
/// Each node has the size of ggml's `mul_mat` node it stands for and reads the same input, so
/// ggml's graph allocator lays the graph out as with ggml's matmuls, wherever each result is
/// first read. [`mul_mat`]'s one node for a DeltaNet layer's qkv and z kept z's memory from the
/// qkv matmul to the gated norm that reads z, which moved the layout ggml-cuda's MoE fusions
/// depend on and with it the greedy text (#819).
pub fn mul_mat_each<'a>(ctx: &'a Context, ws: &[Tensor<'a>], x: Tensor<'a>) -> Vec<Tensor<'a>> {
    assert!(
        (1..=MAX_WEIGHTS).contains(&ws.len()) && ws.iter().all(|&w| fits(w, x)),
        "q8_group: shapes"
    );
    let (n, q) = (x.ne()[1], quantized(ctx, x));
    ws.iter()
        .map(|&w| {
            ctx.external_with(
                &Q8_GROUP,
                types::F32,
                [w.ne()[1], n, 1, 1],
                &[x, q, w],
                &[KIND_GROUP],
            )
        })
        .collect()
}

/// Whether [`swiglu`] takes `gate` and `up` with `x`: [`takes`] both, the same rows.
pub fn takes_swiglu(gate: Tensor<'_>, up: Tensor<'_>, x: Tensor<'_>) -> bool {
    gate.ne() == up.ne() && takes(&[gate, up], x)
}

/// `silu(gate @ x) * (up @ x)` in one op (gate and up [`fits`] with `x`, the same rows): F32
/// `[rows, n]`, the bits of ggml's two mmvq launches and its SwiGLU (fused or not).
pub fn swiglu<'a>(ctx: &'a Context, gate: Tensor<'a>, up: Tensor<'a>, x: Tensor<'a>) -> Tensor<'a> {
    assert!(
        gate.ne() == up.ne() && fits_all(&[gate, up], x),
        "q8_group: shapes"
    );
    let (m, n) = (gate.ne()[1], x.ne()[1]);
    let args = [x, quantized(ctx, x), gate, up];
    let node = ctx.external_with(
        &Q8_GROUP,
        types::F32,
        [m * n, 1, 1, 1],
        &args,
        &[KIND_SWIGLU],
    );
    ctx.view_2d(node, [m, n], m as usize * 4, 0)
}

/// Whether [`swiglu_down`] takes `gate`, `up` and `down` with `x`: [`takes_swiglu`], and `down`
/// [`fits`] the SwiGLU's `[rows, n]` output in one GPU's memory, its Q8_1 blocks within a slot.
pub fn takes_swiglu_down(
    gate: Tensor<'_>,
    up: Tensor<'_>,
    down: Tensor<'_>,
    x: Tensor<'_>,
) -> bool {
    let (m, n) = (gate.ne()[1], x.ne()[1]);
    takes_swiglu(gate, up, x)
        && weight_fits(down, m, n)
        && !down.is_host()
        && !down.is_meta()
        && m / GROUP <= SLOT_ROW_BLOCKS
        && m / GROUP * n <= SLOT_BLOCKS
}

/// `down @ (silu(gate @ x) * (up @ x))` ([`takes_swiglu_down`]) as two op nodes, the SwiGLU's
/// and the down projection's, each of the size and inputs of the ggml node it stands for
/// ([`swiglu`]'s node and ggml's `mul_mat`), so the graph's layout is [`swiglu`]'s: the SwiGLU
/// also writes its output's Q8_1 blocks, as ggml's `quantize_q8_1` would, to slot `layer` of
/// [`SLOTS`] on the GPU, and the down projection reads them there instead of quantizing its
/// input: ggml's bits without the quantize launch. Layer `layer`'s pair is the only one using
/// that slot, and the graph runs the two nodes in order on one stream.
pub fn swiglu_down<'a>(
    ctx: &'a Context,
    gate: Tensor<'a>,
    up: Tensor<'a>,
    down: Tensor<'a>,
    x: Tensor<'a>,
    layer: usize,
) -> Tensor<'a> {
    let (m, n) = (gate.ne()[1], x.ne()[1]);
    assert!(
        gate.ne() == up.ne() && fits_all(&[gate, up], x) && weight_fits(down, m, n),
        "q8_group: shapes"
    );
    let slot = (layer % SLOTS) as i32;
    let args = [x, quantized(ctx, x), gate, up];
    let node = ctx.external_with(
        &Q8_GROUP,
        types::F32,
        [m * n, 1, 1, 1],
        &args,
        &[KIND_SWIGLU_Q8, slot],
    );
    let sh = ctx.view_2d(node, [m, n], m as usize * 4, 0);
    // The input twice: src[1] stands where the other nodes' Q8_1 node is (the slot's blocks
    // are read instead), so `weights` and `out_offset` read the node as theirs.
    ctx.external_with(
        &Q8_GROUP,
        types::F32,
        [down.ne()[1], n, 1, 1],
        &[sh, sh, down],
        &[KIND_SLOT_IN, slot],
    )
}

/// The weights of a node made by [`mul_mat`] or [`swiglu`].
fn weights(dst: &ffi::ggml_tensor) -> usize {
    dst.src[2..]
        .iter()
        .take(MAX_WEIGHTS)
        .take_while(|s| !s.is_null())
        .count()
}

/// Whether a node is [`swiglu`]'s or [`swiglu_down`]'s SwiGLU (else a matvec of each weight).
fn is_swiglu(dst: &ffi::ggml_tensor) -> bool {
    matches!(kind(dst), KIND_SWIGLU | KIND_SWIGLU_Q8)
}

/// A node's kind ([`KIND_GROUP`] and the others).
fn kind(dst: &ffi::ggml_tensor) -> i32 {
    crate::ggml::external_params(dst)[0]
}

/// The [`SLOTS`] slot of a [`swiglu_down`] node.
#[cfg(feature = "cuda")]
fn slot(dst: &ffi::ggml_tensor) -> i32 {
    crate::ggml::external_params(dst)[1]
}

/// The `[rows, n]` result of weight `i` of a node made by [`mul_mat`]: its offset in
/// `dst.data` ([`swiglu`]: 0, its one result).
fn out_offset(dst: &ffi::ggml_tensor, i: usize) -> usize {
    // SAFETY: a node made by `mul_mat`: src[0] the input, src[1] its Q8_1, src[2..] the weights.
    unsafe {
        let n = (*dst.src[0]).ne[1] as usize;
        (0..i)
            .map(|j| (*dst.src[2 + j]).ne[1] as usize * n * 4)
            .sum()
    }
}

/// ggml's `quantize_q8_1` of 32 values: the F16 scale's F32 value and the quants.
pub(crate) fn quantize_block(x: &[f32]) -> (f32, [i8; 32]) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let mut q = [0i8; 32];
    if amax != 0.0 {
        for (q, &v) in q.iter_mut().zip(x) {
            *q = (v / d).round() as i8;
        }
    }
    // SAFETY: plain conversions.
    let d16 = unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(d)) };
    (d16, q)
}

/// ggml's `quantize_q8_1` of `x` (`[k, n]`) into `dst` (Q8_1 blocks, a column after another).
unsafe fn cpu_q8_quantize(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `quantized`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        write_q8_1(
            |t| std::slice::from_raw_parts((x.data as *const u8).add(t * x.nb[1]) as *const f32, k),
            k,
            n,
            dst.data as *mut u8,
        );
    }
}

/// Writes the Q8_1 blocks of `n` columns of `k` values (`col(t)`) to `out` as ggml's
/// `quantize_q8_1` lays them out: F16 scale and sum, 32 int8, `k / 32` blocks a column.
///
/// # Safety
/// `out` holds [`quantized_bytes`]`(k, n)` bytes.
pub(crate) unsafe fn write_q8_1<'x>(
    col: impl Fn(usize) -> &'x [f32],
    k: usize,
    n: usize,
    out: *mut u8,
) {
    let blocks = k / GROUP as usize;
    for t in 0..n {
        let c = col(t);
        for b in 0..blocks {
            let v = &c[b * 32..b * 32 + 32];
            let (_, q) = quantize_block(v);
            let amax = v.iter().fold(0f32, |m, v| m.max(v.abs()));
            let sum: f32 = v.iter().sum();
            // SAFETY: block (t, b) lies within `out` (the caller's promise); plain conversions.
            unsafe {
                let o = out.add((t * blocks + b) * Q8_1_BYTES);
                let d = ffi::ggml_fp32_to_fp16(amax / 127.0).to_le_bytes();
                let s = ffi::ggml_fp32_to_fp16(sum).to_le_bytes();
                std::ptr::copy_nonoverlapping([d[0], d[1], s[0], s[1]].as_ptr(), o, 4);
                std::ptr::copy_nonoverlapping(q.as_ptr() as *const u8, o.add(4), 32);
            }
        }
    }
}

unsafe fn cpu_q8_group(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat` or `swiglu`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (k, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let blocks = k / GROUP as usize;
        // Each thread quantizes the input for itself (the same values as the Q8_1 node's).
        let mut xd = vec![0f32; blocks * n];
        let mut xq = vec![0i8; k * n];
        for t in 0..n {
            let col =
                std::slice::from_raw_parts((x.data as *const u8).add(t * x.nb[1]) as *const f32, k);
            for b in 0..blocks {
                let (d, q) = quantize_block(&col[b * 32..b * 32 + 32]);
                xd[t * blocks + b] = d;
                xq[t * k + b * 32..t * k + b * 32 + 32].copy_from_slice(&q);
            }
        }
        let dot = |w: &ffi::ggml_tensor, r: usize, t: usize| {
            let row = (w.data as *const u8).add(r * w.nb[1]);
            let mut s = 0f32;
            for b in 0..blocks {
                let wb = row.add(b * Q8_0_BYTES);
                let dw = ffi::ggml_fp16_to_fp32(u16::from_le_bytes([*wb, *wb.add(1)]));
                let mut idot = 0i32;
                for v in 0..32 {
                    idot += (*wb.add(2 + v) as i8 as i32) * xq[t * k + b * 32 + v] as i32;
                }
                s += dw * xd[t * blocks + b] * idot as f32;
            }
            s
        };
        if is_swiglu(dst) {
            let (gate, up) = (&*dst.src[2], &*dst.src[3]);
            let m = gate.ne[1] as usize;
            let out = dst.data as *mut f32;
            for r in (ith..m).step_by(nth) {
                for t in 0..n {
                    let g = dot(gate, r, t);
                    // ggml's silu: x / (1 + exp(-x)).
                    *out.add(t * m + r) = dot(up, r, t) * (g / (1.0 + (-g).exp()));
                }
            }
            return;
        }
        for i in 0..weights(dst) {
            let w = &*dst.src[2 + i];
            let m = w.ne[1] as usize;
            let out = (dst.data as *mut u8).add(out_offset(dst, i)) as *mut f32;
            for r in (ith..m).step_by(nth) {
                for t in 0..n {
                    *out.add(t * m + r) = dot(w, r, t);
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI32, Ordering};

    /// `legion_q8_group` of `src/cuda/q8_group.cu`.
    #[repr(C)]
    pub(super) struct Params {
        pub x: *const f32,
        pub x_ld: i64,
        pub xq: *mut c_void,
        pub w: [*const c_void; super::MAX_WEIGHTS],
        pub w_ld: [i64; super::MAX_WEIGHTS],
        pub dst: [*mut f32; super::MAX_WEIGHTS],
        pub rows: [i32; super::MAX_WEIGHTS],
        pub start: [i32; super::MAX_WEIGHTS + 1],
        pub k: i32,
        pub nw: i32,
        pub slot: i32,
    }

    /// What a matvec launch reads and writes ([`mv`]): the kernel table's ranges.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) enum Io {
        /// Each weight's matvec from `xq`.
        Plain,
        /// Gate and up from `xq`, their SwiGLU written to `dst[0]`.
        Glu,
        /// [`Io::Glu`], and the SwiGLU's Q8_1 blocks written to slot `slot` too.
        GluQ8Out,
        /// The matvec reading its input's Q8_1 blocks from slot `slot`.
        SlotIn,
    }

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `cudaLaunchAttribute`: the id, padding to 8 bytes, a 64-byte value union.
    #[repr(C, align(8))]
    struct LaunchAttribute {
        id: c_int,
        pad: [u8; 4],
        val: [u32; 16],
    }

    /// `cudaLaunchConfig_t`.
    #[repr(C)]
    struct LaunchConfig {
        grid: Dim3,
        block: Dim3,
        dynamic_smem: usize,
        stream: *mut c_void,
        attrs: *mut LaunchAttribute,
        num_attrs: u32,
    }

    /// `cudaLaunchAttributeProgrammaticStreamSerialization`.
    const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs): the quantization,
        /// then the matvec of 1-8 columns, then the same with the SwiGLU, with the SwiGLU
        /// writing its Q8_1 blocks to a slot, and reading the input's blocks from a slot.
        static legion_q8_group_kernels: [*const c_void; 33];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CC_MINOR: c_int = 76;

    /// Devices the cache below covers.
    const DEVICES: usize = 64;

    /// Each device's compute capability (10 * major + minor); 0 not yet read, -1 unknown.
    static CC: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// ggml's `quantize_q8_1` block size (`CUDA_QUANTIZE_BLOCK_SIZE`).
    const QUANTIZE_THREADS: i64 = 256;

    /// The current device's compute capability, read once per device.
    pub(super) fn compute_capability() -> Option<i32> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return None;
            }
            let slot = &CC[dev as usize];
            let mut cc = slot.load(Ordering::Relaxed);
            if cc == 0 {
                let (mut major, mut minor) = (0, 0);
                cc = if cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0
                    && cudaDeviceGetAttribute(&mut minor, DEV_ATTR_CC_MINOR, dev) == 0
                {
                    10 * major + minor
                } else {
                    cudaGetLastError();
                    -1
                };
                slot.store(cc, Ordering::Relaxed);
            }
            (cc > 0).then_some(cc)
        }
    }

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream`; on a GB10 as a
    /// programmatic dependent launch, as ggml-cuda launches its own kernels there (each kernel
    /// waits for the one ahead before it touches activations, `src/cuda/q8_group.cu`).
    fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        p: &mut Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut Params).cast::<c_void>()];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid,
            block,
            dynamic_smem: 0,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(compute_capability() == Some(121)),
        };
        // SAFETY: a kernel of the table taking `legion_q8_group` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the nodes).
        match unsafe {
            cudaLaunchKernelExC(&config, legion_q8_group_kernels[kernel], argv.as_mut_ptr())
        } {
            0 => Ok(()),
            e => Err(format!("q8_group: CUDA error {e}")),
        }
    }

    /// Enqueues the quantization of `p.x` (`n` columns) into `p.xq` on `stream`.
    pub(super) fn quantize(p: &mut Params, n: i64, stream: *mut c_void) -> Result<(), String> {
        let k = p.k as i64;
        let grid = Dim3 {
            x: ((k + QUANTIZE_THREADS - 1) / QUANTIZE_THREADS) as u32,
            y: n as u32,
            z: 1,
        };
        let block = Dim3 {
            x: QUANTIZE_THREADS as u32,
            y: 1,
            z: 1,
        };
        launch(0, grid, block, p, stream)
    }

    /// Enqueues the matvecs of `n` columns from `p.xq` on `stream` as `io` says (with a SwiGLU:
    /// weights 0 and 1 a gate and up pair, their SwiGLU written to `p.dst[0]`).
    pub(super) fn mv(p: &mut Params, n: i64, io: Io, stream: *mut c_void) -> Result<(), String> {
        let glu = matches!(io, Io::Glu | Io::GluQ8Out);
        let nw = if glu { 1 } else { p.nw as usize };
        let warps = super::ggml_warps(n);
        let rpb = super::ggml_rows_per_block(n);
        p.start[0] = 0;
        for i in 0..nw {
            let ctas = p.rows[i] as i64 / rpb;
            let end = p.start[i] as i64 + ctas;
            if end > i32::MAX as i64 {
                return Err("q8_group: too many rows".into());
            }
            p.start[i + 1] = end as i32;
        }
        let grid = Dim3 {
            x: p.start[nw] as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: 32,
            y: warps as u32,
            z: 1,
        };
        let range = match io {
            Io::Plain => 0,
            Io::Glu => 1,
            Io::GluQ8Out => 2,
            Io::SlotIn => 3,
        };
        let kernel = n as usize + range * super::MAX_TOKENS as usize;
        launch(kernel, grid, block, p, stream)
    }
}

#[cfg(feature = "cuda")]
fn params(x: &ffi::ggml_tensor, xq: *mut c_void) -> launch::Params {
    launch::Params {
        x: x.data as *const f32,
        x_ld: (x.nb[1] / 4) as i64,
        xq,
        w: [std::ptr::null(); MAX_WEIGHTS],
        w_ld: [0; MAX_WEIGHTS],
        dst: [std::ptr::null_mut(); MAX_WEIGHTS],
        rows: [0; MAX_WEIGHTS],
        start: [0; MAX_WEIGHTS + 1],
        k: x.ne[0] as i32,
        nw: 0,
        slot: 0,
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q8_quantize(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `quantized` (its input passed `fits`), device tensors of the
    // stream's device.
    unsafe {
        let x = &*dst.src[0];
        let mut p = params(x, dst.data);
        launch::quantize(&mut p, x.ne[1], stream)
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_q8_group(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat`, `mul_mat_each`, `swiglu` or `swiglu_down` (its shapes
    // passed `fits`), device tensors of the stream's device.
    unsafe {
        let x = &*dst.src[0];
        let nw = weights(dst);
        let io = match kind(dst) {
            KIND_SWIGLU => launch::Io::Glu,
            KIND_SWIGLU_Q8 => launch::Io::GluQ8Out,
            KIND_SLOT_IN => launch::Io::SlotIn,
            _ => launch::Io::Plain,
        };
        let mut p = params(x, (*dst.src[1]).data);
        p.nw = nw as i32;
        if matches!(io, launch::Io::GluQ8Out | launch::Io::SlotIn) {
            p.slot = slot(dst);
        }
        for i in 0..nw {
            let w = &*dst.src[2 + i];
            p.w[i] = w.data as *const c_void;
            p.w_ld[i] = (w.nb[1] / Q8_0_BYTES) as i64;
            p.dst[i] = (dst.data as *mut u8).add(out_offset(dst, i)) as *mut f32;
            p.rows[i] = w.ne[1] as i32;
        }
        // One column on a V100: a warp a row, the same bits (#814, `q8_head::row_mv`).
        if io == launch::Io::Plain && x.ne[1] == 1 && nw <= super::q8_head::XQ_WEIGHTS {
            let mut ws = [(std::ptr::null(), 0, 0); super::q8_head::XQ_WEIGHTS];
            for (i, s) in ws.iter_mut().enumerate().take(nw) {
                *s = (p.w[i], (*dst.src[2 + i]).nb[1], i64::from(p.rows[i]));
            }
            if super::q8_head::row_mv(&ws[..nw], &p.dst[..nw], p.xq, x.ne[0], stream)? {
                return Ok(());
            }
        }
        launch::mv(&mut p, x.ne[1], io, stream)
    }
}

pub mod check;

#[cfg(test)]
mod tests;
