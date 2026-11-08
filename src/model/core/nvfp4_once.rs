//! NVFP4 expert matvecs that read each routed expert's weights once per step (#836):
//! `mul_mat_id(w, x, ids)` of an NVFP4 expert stack for steps of [`MIN_TOKENS`]..=[`MAX_TOKENS`]
//! tokens (an MTP verify step at one sequence), as an external op running
//! `src/cuda/nvfp4_once.cu` in place of ggml-cuda's quantize_q8_1 + MMVQ pair, with ggml's bits.
//!
//! ggml's MoE matvec (`mul_mat_vec_q_moe`) runs a warp per (token, slot) pair, so when a verify
//! step's tokens pick the same expert (most of them do) each pair streams the expert's rows again:
//! on Qwen3.8-Flash-Next's C=1 MTP profile the routed experts took 31% of a round (PR #999). The
//! op groups the step's pairs by expert on the device: one work item per (expert, rows) tile
//! loads the rows' weights once (several K steps at a time, so the loads are in flight together)
//! and adds every pair's products from the same registers, each pair's sum in ggml's warp layout
//! and order (see the kernel file), so each output keeps ggml's bits.
//!
//! Two ops: [`mul_mat_id`] (one matmul, times the per-expert scale of NVFP4 checkpoints when
//! there is one) and [`swiglu_id`] (the gate and up matmuls of the same experts on one
//! quantization of `x`, each times its scale, then `silu(gate) * up`): the bits of the graph
//! nodes they replace (`mul_mat_id`, the scale chain `reshape`, `repeat`, `get_rows`, `mul`,
//! and `silu` and `mul`).
//!
//! Opt-in on compute capability 12.x (GB10, RTX 50; [`set_gpu`]) with
//! `legion serve --nvfp4-once` ([`set_nvfp4_once`]). `legion nvfp4-once-bench` times the ops
//! against ggml's pair and checks their bits ([`bench`]). The CPU implementation is a plain f32
//! reference, for tests of the wiring.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

pub mod bench;
#[cfg(test)]
mod tests;

static NVFP4_ONCE: ExternalOp = ExternalOp {
    name: "nvfp4_once",
    cpu: cpu_nvfp4_once,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_nvfp4_once),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static NVFP4_ONCE_GLU: ExternalOp = ExternalOp {
    name: "nvfp4_once_glu",
    cpu: cpu_nvfp4_once,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_nvfp4_once),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Fewest tokens of a step the op takes: at one token no two pairs share an expert.
pub const MIN_TOKENS: i64 = 2;

/// Most tokens of a step the op takes: ggml-cuda runs NVFP4 `mul_mat_id` on its MMVQ up to 8
/// tokens (`get_mmvq_mmid_max_batch`), on MMQ above, and Legion's own NVFP4 kernels take those
/// ([`super::linear::NVFP4_GGML_MAX_N`]).
pub const MAX_TOKENS: i64 = 8;

/// Most (token, slot) pairs of a step: the kernel's shared-memory id table (a thread per pair).
pub const MAX_PAIRS: i64 = 128;

/// Q8_1 blocks of the device-global activation scratch (`n_x * n * k / 32` at most): 8192 * 36
/// bytes per device.
pub const SCRATCH_BLOCKS: i64 = 8192;

/// Warps of a matvec CTA (the kernel's `NW`).
pub const WARPS: i64 = 4;

/// The weights' rows are a multiple of this: a CTA's rows at the biggest tile.
pub const ROW_MULTIPLE: i64 = 32;

/// Values per NVFP4 block and its bytes.
const QK: i64 = 64;
const BLOCK_BYTES: usize = 36;

static ON: AtomicBool = AtomicBool::new(false);
static GPU: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on take the ops where they fit (`legion serve --nvfp4-once`)
/// or ggml's MMVQ (the default).
pub fn set_nvfp4_once(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: the ops are used on compute capability 12.x (#1200: the GB10, where their
/// bits were checked against ggml's MMVQ and where they were measured, and the RTX 50, whose SM
/// is the GB10's; its bits there want a `legion nvfp4-once-bench` check before anyone relies
/// on them). Opt-in either way ([`set_nvfp4_once`]).
pub fn set_gpu(cc: Option<(i32, i32)>) {
    GPU.store(gpu_takes(cc), Ordering::Relaxed);
}

/// Whether compute capability `cc` (major, minor) takes the ops ([`set_gpu`]).
pub fn gpu_takes(cc: Option<(i32, i32)>) -> bool {
    cc.is_some_and(|(major, _)| major == 12)
}

pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed) && GPU.load(Ordering::Relaxed) && cfg!(feature = "cuda")
}

/// Whether a router of `n_used` experts for `n` tokens writes the routing plan of the ops'
/// warp kernels with its ids ([`super::fused::router_planned`]): switched on, a GB10, and a
/// step the ops take (the plan holds every pair). The ops then launch without building it.
pub fn plan_fits(n_used: i64, n: i64) -> bool {
    on() && plan_shape(n_used, n)
}

/// [`plan_fits`]'s shape check: a step of [`MIN_TOKENS`]..=[`MAX_TOKENS`] tokens and at most
/// [`MAX_PAIRS`] pairs.
pub fn plan_shape(n_used: i64, n: i64) -> bool {
    n_used > 0 && (MIN_TOKENS..=MAX_TOKENS).contains(&n) && n_used * n <= MAX_PAIRS
}

/// Whether [`mul_mat_id`] takes `mul_mat_id(w, x, ids)` times `scale`: switched on, a GB10,
/// [`fits_shape`] and [`scale_fits`].
pub fn usable(w: Tensor<'_>, scale: Option<Tensor<'_>>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    on() && fits_shape(w, x, ids) && scale_fits(w, scale)
}

/// Whether [`swiglu_id`] takes the gate and up matmuls `(weight, scale)` of `x`: switched on, a
/// GB10, and both weights as [`usable`] with the same shapes and strides.
pub fn glu_usable(
    gate: (Tensor<'_>, Option<Tensor<'_>>),
    up: (Tensor<'_>, Option<Tensor<'_>>),
    x: Tensor<'_>,
    ids: Tensor<'_>,
) -> bool {
    on() && glu_fits(gate, up, x, ids)
}

/// [`glu_usable`]'s shape checks: both weights fit ([`fits_shape`], [`scale_fits`]) with the
/// same shapes and strides, and both have a scale or neither does.
pub fn glu_fits(
    gate: (Tensor<'_>, Option<Tensor<'_>>),
    up: (Tensor<'_>, Option<Tensor<'_>>),
    x: Tensor<'_>,
    ids: Tensor<'_>,
) -> bool {
    gate.0.ne() == up.0.ne()
        && gate.0.nb() == up.0.nb()
        && gate.1.is_some() == up.1.is_some()
        && fits_shape(gate.0, x, ids)
        && fits_shape(up.0, x, ids)
        && scale_fits(gate.0, gate.1)
        && scale_fits(up.0, up.1)
}

/// Whether `scale` is a per-expert scale the kernels read: none, or a contiguous F32 `[n_expert]`
/// in device memory.
pub fn scale_fits(w: Tensor<'_>, scale: Option<Tensor<'_>>) -> bool {
    scale.is_none_or(|s| {
        s.ty() == types::F32
            && s.ne() == [w.ne()[2], 1, 1, 1]
            && s.is_contiguous()
            && !s.is_host()
            && !s.is_meta()
    })
}

/// Whether the op can compute `mul_mat_id(w, x, ids)` with ggml's MMVQ bits: an NVFP4 expert
/// stack in one device's memory (rows of a multiple of 64 values, a multiple of
/// [`ROW_MULTIPLE`] rows), F32 activations shared by the slots (`[k, 1, n]`) or one per slot
/// (`[k, n_used, n]`), I32 ids `[n_used, n]`, [`MIN_TOKENS`] <= n <= [`MAX_TOKENS`], and the
/// pairs and the activations' blocks within [`MAX_PAIRS`] and [`SCRATCH_BLOCKS`].
pub fn fits_shape(w: Tensor<'_>, x: Tensor<'_>, ids: Tensor<'_>) -> bool {
    let [k, rows, n_expert, w3] = w.ne();
    let [xk, n_x, n, x3] = x.ne();
    let [n_used, n_ids, i2, i3] = ids.ne();
    let (wnb, xnb, inb) = (w.nb(), x.nb(), ids.nb());
    w.ty() == types::NVFP4
        && k > 0
        && k % QK == 0
        && k <= i32::MAX as i64
        && xk == k
        && rows > 0
        && rows % ROW_MULTIPLE == 0
        && rows <= i32::MAX as i64
        && w3 == 1
        && x3 == 1
        && n_used > 0
        && (n_x == 1 || n_x == n_used)
        && n_ids == n
        && i2 == 1
        && i3 == 1
        && (MIN_TOKENS..=MAX_TOKENS).contains(&n)
        && n_used * n <= MAX_PAIRS
        && n_x * n * (k / 32) <= SCRATCH_BLOCKS
        && (1..=1 << 16).contains(&n_expert)
        && x.ty() == types::F32
        && ids.ty() == types::I32
        && xnb[0] == 4
        && xnb[1].is_multiple_of(4)
        && xnb[2].is_multiple_of(4)
        && inb[0] == 4
        && inb[1].is_multiple_of(4)
        && wnb[0] == BLOCK_BYTES
        && wnb[1].is_multiple_of(BLOCK_BYTES)
        && wnb[2].is_multiple_of(BLOCK_BYTES)
        && wnb[1] / BLOCK_BYTES * rows as usize <= i32::MAX as usize
        && !w.is_host()
        && !w.is_meta()
}

/// `mul_mat_id(w, x, ids)` (F32 `[rows, n_used, n]`), each row times its expert's entry of
/// `scale` when there is one, on the op; the caller checked [`usable`].
pub fn mul_mat_id<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
    scale: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let rows = w.ne()[1];
    let [n_used, n, _, _] = ids.ne();
    let mut args = vec![w, x, ids];
    args.extend(scale);
    let params = [0, i32::from(scale.is_some())];
    ctx.external_with(
        &NVFP4_ONCE,
        types::F32,
        [rows, n_used, n, 1],
        &args,
        &params,
    )
}

/// `silu(gate) * up` of the routed experts' gate and up matmuls of `x` (each `(weight, scale)`,
/// as [`mul_mat_id`]) on one op: F32 `[rows, n_used, n]`; the caller checked [`glu_usable`].
pub fn swiglu_id<'a>(
    ctx: &'a Context,
    gate: (Tensor<'a>, Option<Tensor<'a>>),
    up: (Tensor<'a>, Option<Tensor<'a>>),
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    let rows = up.0.ne()[1];
    let [n_used, n, _, _] = ids.ne();
    let mut args = vec![up.0, x, ids, gate.0];
    let scaled = if let (Some(su), Some(sg)) = (up.1, gate.1) {
        args.extend([su, sg]);
        1
    } else {
        0
    };
    let params = [1, scaled];
    ctx.external_with(
        &NVFP4_ONCE_GLU,
        types::F32,
        [rows, n_used, n, 1],
        &args,
        &params,
    )
}

/// The matvec's tiles: rows per warp (4 warps a CTA). Every one gives the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub rows: u32,
}

/// The tiles of `legion_nvfp4_once_kernels`, in table order.
pub const TILES: [Tile; 4] = [
    Tile { rows: 1 },
    Tile { rows: 2 },
    Tile { rows: 4 },
    Tile { rows: 8 },
];

/// The tile of [`swiglu_id`] when none is forced (two matmuls' weights in registers).
pub const DEFAULT_GLU_TILE: usize = 1;

/// The tile of [`mul_mat_id`] when none is forced: 2 rows a warp, as for [`swiglu_id`] (the
/// best of the tiles for Flash-Next's down projection on a GB10 too).
pub fn default_tile(_k: usize) -> usize {
    1
}

/// The tile every launch takes from now on (index into [`TILES`] plus 1; 0: the defaults).
static FORCED: AtomicUsize = AtomicUsize::new(0);

/// Launches every op from now on with tile `TILES[i]`, or the defaults for `None`: for the
/// bench and tests (the same bits either way).
#[doc(hidden)]
pub fn force_tile(i: Option<usize>) {
    FORCED.store(
        i.map_or(0, |i| i.min(TILES.len() - 1) + 1),
        Ordering::Relaxed,
    );
}

/// The tile of a launch: forced, or the default for the op (`glu`) and row length `k`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn tile(glu: bool, k: usize) -> usize {
    match FORCED.load(Ordering::Relaxed) {
        0 if glu => DEFAULT_GLU_TILE,
        0 => default_tile(k),
        i => i - 1,
    }
}

/// The K steps every launch loads at once from now on (0: by the row length).
static FORCED_ITERS: AtomicUsize = AtomicUsize::new(0);

/// Launches every op from now on loading `i` K steps at once (1 or 3; others take 3), or by
/// the row length for `None`: for the bench (the same bits either way).
#[doc(hidden)]
pub fn force_iters(i: Option<usize>) {
    FORCED_ITERS.store(
        i.map_or(0, |i| if i == 1 { 1 } else { 3 }),
        Ordering::Relaxed,
    );
}

/// K steps (16 blocks of 64 values per warp) a warp loads before its first dot when none are
/// forced: one (3, all of a 2560-wide row, needs more registers and was no faster on a GB10).
const DEFAULT_ITERS: usize = 1;

/// K steps a warp loads before its first dot: forced, or [`DEFAULT_ITERS`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn iters(_k: usize) -> usize {
    match FORCED_ITERS.load(Ordering::Relaxed) {
        0 => DEFAULT_ITERS,
        i => i,
    }
}

/// Pairs of one expert a warp adds per pass over its rows (the kernels' USERS): the smallest of
/// 2, 4 and 8 that holds a pair per token of the step (an expert takes a pair per token at
/// most), as an index 0..3.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn users(n_tok: usize) -> usize {
    match n_tok {
        0..=2 => 0,
        3..=4 => 1,
        _ => 2,
    }
}

/// The kernel of `legion_nvfp4_once_kernels` for `tile`, the op, `iters` and `users`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn kernel_index(tile: usize, glu: bool, iters: usize, users: usize) -> usize {
    1 + 3 * (4 * tile + 2 * usize::from(glu) + usize::from(iters == 3)) + users
}

/// A warp-item matvec (the kernels' `mvw`): every warp takes its own (expert, 2 rows) items
/// from a counter, each expert's pairs `users` per pass over its rows, the weights of a K step
/// decoded once per pass (`pre`) or per pair, at 7 CTAs of 4 warps an SM (`occ7`, at most 72
/// registers a thread) or 8. Every one gives the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Warp {
    /// Pairs per pass: 1, 2, 4 or 8.
    pub users: u32,
    pub pre: bool,
    pub occ7: bool,
}

/// Pairs per pass of the [`Warp`] kernels, in table order.
pub const WARP_USERS: [u32; 4] = [1, 2, 4, 8];

/// Rows of a warp's work item in the [`Warp`] kernels (the kernels' `RPW_W`, ggml's rows per
/// block of `mul_mat_vec_q_moe`).
pub const WARP_ROWS: i64 = 2;

/// The [`Warp`] kernels, in table order.
pub fn warps() -> Vec<Warp> {
    WARP_USERS
        .iter()
        .flat_map(|&users| {
            [false, true].into_iter().flat_map(move |pre| {
                [false, true]
                    .into_iter()
                    .map(move |occ7| Warp { users, pre, occ7 })
            })
        })
        .collect()
}

impl Warp {
    /// Its kernel in `legion_nvfp4_once_kernels` for the op (`glu`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn kernel(self, glu: bool) -> usize {
        let u = WARP_USERS
            .iter()
            .position(|&u| u == self.users)
            .unwrap_or(WARP_USERS.len() - 1);
        1 + 12 * TILES.len()
            + 16 * usize::from(glu)
            + 4 * u
            + 2 * usize::from(self.pre)
            + usize::from(self.occ7)
    }

    /// Short name for the bench: `w u2 pre o12`.
    pub fn name(self) -> String {
        format!(
            "w u{}{} o{}",
            self.users,
            if self.pre { " pre" } else { "" },
            if self.occ7 { 7 } else { 8 }
        )
    }
}

/// A hybrid warp kernel (the kernels' `mvh`): as a [`Warp`] kernel, but each (expert, 2 rows)
/// item picks its pass on the device from its expert's pairs in the routing plan: one pair on
/// the one-pair body, more on 2, 4 or 8 pairs a pass up to `umax`, so a step with singleton and
/// shared experts runs each kind on its own body in one launch, at the register budget of `occ`
/// CTAs of 4 warps an SM. Every one gives the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hybrid {
    /// Most pairs per pass: 2, 4 or 8.
    pub umax: u32,
    /// CTAs an SM the registers are budgeted for: 8, 6 or 4.
    pub occ: u32,
}

/// Most pairs per pass and CTAs an SM of the [`Hybrid`] kernels, in table order.
pub const HYBRID_UMAX: [u32; 3] = [2, 4, 8];
pub const HYBRID_OCC: [u32; 3] = [8, 6, 4];

/// The [`Hybrid`] kernels, in table order.
pub fn hybrids() -> Vec<Hybrid> {
    HYBRID_UMAX
        .iter()
        .flat_map(|&umax| HYBRID_OCC.iter().map(move |&occ| Hybrid { umax, occ }))
        .collect()
}

impl Hybrid {
    /// Its kernel in `legion_nvfp4_once_kernels` for the op (`glu`): after the [`Warp`] ones.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn kernel(self, glu: bool) -> usize {
        let pos = |a: &[u32], v| a.iter().position(|&x| x == v).unwrap_or(a.len() - 1);
        let n = HYBRID_UMAX.len() * HYBRID_OCC.len();
        1 + 12 * TILES.len()
            + 2 * 4 * WARP_USERS.len()
            + n * usize::from(glu)
            + HYBRID_OCC.len() * pos(&HYBRID_UMAX, self.umax)
            + pos(&HYBRID_OCC, self.occ)
    }

    /// Short name for the bench: `h u4 o6`.
    pub fn name(self) -> String {
        format!("h u{} o{}", self.umax, self.occ)
    }
}

/// The kernel family of a launch: the tiles, a [`Warp`] or a [`Hybrid`] kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Tiles,
    Warp(Warp),
    Hybrid(Hybrid),
}

/// The kernel family of a launch when none is forced. The SwiGLU op: the hybrid kernel at
/// most 4 pairs a pass, 4 CTAs an SM (its 122 registers a thread let a warp keep more loads in
/// flight than at 8 or 6 CTAs). The down projection (one K step of 640 values a row, so short
/// items where more warps an SM count): the warp-item kernel at 8 CTAs an SM, one pair a pass
/// at 2 tokens, two above. Per layer 0.78-0.97x ggml at 6-8 tokens, 0.82-1.01x at 4, and
/// 0.96-1.01x at 2 (`legion nvfp4-once-bench` on a GB10, #836), where the tiles were
/// 0.77-1.21x and #1030's default 0.82-1.10x.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn default_family(glu: bool, n_tok: usize) -> Family {
    if glu {
        Family::Hybrid(Hybrid { umax: 4, occ: 4 })
    } else {
        Family::Warp(Warp {
            users: if n_tok <= 2 { 1 } else { 2 },
            pre: false,
            occ7: false,
        })
    }
}

/// The kernel family every launch takes from now on: 0 the defaults, 1 the tiles, 2 + i
/// `warps()[i]`, 2 + `warps().len()` + i `hybrids()[i]`.
static FORCED_FAMILY: AtomicUsize = AtomicUsize::new(0);

/// The same for the [`mul_mat_id`] launches only (0: as [`FORCED_FAMILY`]).
static FORCED_MATMUL: AtomicUsize = AtomicUsize::new(0);

/// A family as [`FORCED_FAMILY`] stores it.
fn encode(f: Option<Family>) -> usize {
    match f {
        None => 0,
        Some(Family::Tiles) => 1,
        Some(Family::Warp(w)) => 2 + warps().iter().position(|&x| x == w).unwrap_or(0),
        Some(Family::Hybrid(h)) => {
            2 + warps().len() + hybrids().iter().position(|&x| x == h).unwrap_or(0)
        }
    }
}

/// [`encode`]'s family back (`None` for 0).
fn decode(v: usize) -> Option<Family> {
    let n_warps = warps().len();
    match v {
        0 => None,
        1 => Some(Family::Tiles),
        i if i - 2 < n_warps => Some(Family::Warp(warps()[i - 2])),
        i => Some(
            hybrids()
                .get(i - 2 - n_warps)
                .copied()
                .map_or(Family::Tiles, Family::Hybrid),
        ),
    }
}

/// Launches every op from now on on kernel family `Some(f)`, or as by default (`None`): for the
/// bench and tests (the same bits either way).
#[doc(hidden)]
pub fn force_family(f: Option<Family>) {
    FORCED_FAMILY.store(encode(f), Ordering::Relaxed);
}

/// Launches every [`mul_mat_id`] op (the down projection) from now on on kernel family
/// `Some(f)`, or as [`force_family`] says (`None`): for the bench.
#[doc(hidden)]
pub fn force_matmul_family(f: Option<Family>) {
    FORCED_MATMUL.store(encode(f), Ordering::Relaxed);
}

/// The kernel family of a launch: forced, or [`default_family`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn family(glu: bool, n_tok: usize) -> Family {
    let matmul = (!glu)
        .then(|| decode(FORCED_MATMUL.load(Ordering::Relaxed)))
        .flatten();
    matmul
        .or_else(|| decode(FORCED_FAMILY.load(Ordering::Relaxed)))
        .unwrap_or_else(|| default_family(glu, n_tok))
}

/// Whether the ops leave a router's routing plan unused and build their own.
static IGNORE_ROUTER_PLAN: AtomicBool = AtomicBool::new(false);

/// Launches every op from now on building its own routing plan even where the router wrote one
/// (`true`), or taking the router's (`false`, the default): for the bench (the same bits).
#[doc(hidden)]
pub fn force_own_plan(own: bool) {
    IGNORE_ROUTER_PLAN.store(own, Ordering::Relaxed);
}

struct Operands {
    glu: bool,
    /// The ids are a router's that wrote the routing plan for this stack's experts.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    planned: bool,
    w: *const u8,
    wg: *const u8,
    w_nb: [usize; 4],
    x: *const u8,
    x_nb: [usize; 4],
    n_x: usize,
    ids: *const u8,
    ids_nb1: usize,
    s_w: *const u8,
    s_g: *const u8,
    dst: *mut u8,
    dst_nb: [usize; 4],
    k: usize,
    rows: usize,
    n_used: usize,
    n_tok: usize,
    n_expert: usize,
}

impl Operands {
    /// # Safety
    /// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        let [glu, scaled, ..] = external_params(dst);
        let (glu, scaled) = (glu != 0, scaled != 0);
        unsafe {
            let (w, x, ids) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
            let data = |i: usize| (*dst.src[i]).data as *const u8;
            // A SwiGLU node has the gate weights fourth, then the two scales; a matmul node its
            // scale fourth.
            let none = std::ptr::null();
            let (wg, s_w, s_g) = match (glu, scaled) {
                (true, true) => (data(3), data(4), data(5)),
                (true, false) => (data(3), none, none),
                (false, true) => (none, data(3), none),
                (false, false) => (none, none, none),
            };
            // The plan's experts are the router's logits rows: this stack's only if they match.
            let planned = super::fused::writes_plan(dst.src[2])
                && (*ids.src[0]).ne[0] == w.ne[2]
                && ids.ne[0] * ids.ne[1] <= MAX_PAIRS;
            Self {
                glu,
                planned,
                w: w.data as *const u8,
                wg,
                w_nb: w.nb,
                x: x.data as *const u8,
                x_nb: x.nb,
                n_x: x.ne[1] as usize,
                ids: ids.data as *const u8,
                ids_nb1: ids.nb[1],
                s_w,
                s_g,
                dst: dst.data as *mut u8,
                dst_nb: dst.nb,
                k: w.ne[0] as usize,
                rows: w.ne[1] as usize,
                n_used: ids.ne[0] as usize,
                n_tok: ids.ne[1] as usize,
                n_expert: w.ne[2] as usize,
            }
        }
    }
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`], all tensors in host memory.
unsafe fn cpu_nvfp4_once(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat_id` or `swiglu_id`, all tensors in host memory.
    let o = unsafe { Operands::of(dst) };
    let row_bytes = crate::ggml::row_size(types::NVFP4, o.k);
    for pair in (ith..o.n_used * o.n_tok).step_by(nth) {
        let (t, j) = (pair / o.n_used, pair % o.n_used);
        // SAFETY: in-bounds reads of ids, x, the scales and the expert's rows; writes to dst's
        // row.
        unsafe {
            let e = *(o.ids.add(t * o.ids_nb1 + j * 4) as *const i32);
            let Some(e) = usize::try_from(e).ok().filter(|&e| e < o.n_expert) else {
                continue;
            };
            let out = o.dst.add(t * o.dst_nb[2] + j * o.dst_nb[1]) as *mut f32;
            let xj = if o.n_x == 1 { 0 } else { j };
            let x = std::slice::from_raw_parts(
                o.x.add(t * o.x_nb[2] + xj * o.x_nb[1]) as *const f32,
                o.k,
            );
            let scale = |s: *const u8| (!s.is_null()).then(|| *(s as *const f32).add(e));
            let dot = |w: *const u8, s: *const u8, r: usize| {
                let row =
                    std::slice::from_raw_parts(w.add(e * o.w_nb[2] + r * o.w_nb[1]), row_bytes);
                let w = crate::ggml::dequantize(types::NVFP4, row, o.k);
                let v: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                scale(s).map_or(v, |s| v * s)
            };
            for r in 0..o.rows {
                let mut v = dot(o.w, o.s_w, r);
                if o.glu {
                    let g = dot(o.wg, o.s_g, r);
                    v *= g / (1.0 + (-g).exp());
                }
                *out.add(r) = v;
            }
        }
    }
}

/// `legion_nvfp4_once` of `src/cuda/nvfp4_once.cu`.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Params {
    x: *const f32,
    x_nb1: i64,
    x_nb2: i64,
    n_x: i32,
    w: *const c_void,
    w_nb1: i64,
    w_nb2: i64,
    ids: *const i32,
    ids_nb1: i64,
    dst: *mut f32,
    dst_nb1: i64,
    dst_nb2: i64,
    k: i32,
    rows: i32,
    n_used: i32,
    n_tok: i32,
    n_expert: i32,
    wg: *const c_void,
    s_w: *const f32,
    s_g: *const f32,
}

/// This device's routing plan in `src/cuda/nvfp4_once.cu` (for [`super::fused::router_planned`]'s
/// kernel to write), read once per device.
#[cfg(feature = "cuda")]
pub(super) fn plan_ptr() -> Result<*mut c_void, String> {
    launch::plan()
}

/// # Safety
/// `dst` is a node made by [`mul_mat_id`] or [`swiglu_id`] (its shapes passed [`fits_shape`]),
/// its tensors in device memory of the stream's device; `stream` is the compute stream.
#[cfg(feature = "cuda")]
unsafe fn cuda_nvfp4_once(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `mul_mat_id` or `swiglu_id`.
    let o = unsafe { Operands::of(dst) };
    let mut p = Params {
        x: o.x as *const f32,
        x_nb1: (o.x_nb[1] / 4) as i64,
        x_nb2: (o.x_nb[2] / 4) as i64,
        n_x: o.n_x as i32,
        w: o.w as *const c_void,
        w_nb1: (o.w_nb[1] / BLOCK_BYTES) as i64,
        w_nb2: (o.w_nb[2] / BLOCK_BYTES) as i64,
        ids: o.ids as *const i32,
        ids_nb1: (o.ids_nb1 / 4) as i64,
        dst: o.dst as *mut f32,
        dst_nb1: (o.dst_nb[1] / 4) as i64,
        dst_nb2: (o.dst_nb[2] / 4) as i64,
        k: o.k as i32,
        rows: o.rows as i32,
        n_used: o.n_used as i32,
        n_tok: o.n_tok as i32,
        n_expert: o.n_expert as i32,
        wg: o.wg as *const c_void,
        s_w: o.s_w as *const f32,
        s_g: o.s_g as *const f32,
    };
    let family = family(o.glu, o.n_tok);
    // The warp kernels' routing plan: the router's if it wrote one, else one more z slice of
    // the quantization, whose first CTA writes it.
    let own_plan =
        family != Family::Tiles && (!o.planned || IGNORE_ROUTER_PLAN.load(Ordering::Relaxed));
    let quantize = launch::Dim3 {
        x: o.k.div_ceil(256) as u32,
        y: o.n_x as u32,
        z: o.n_tok as u32 + u32::from(own_plan),
    };
    launch::launch(
        0,
        quantize,
        launch::Dim3 { x: 256, y: 1, z: 1 },
        &mut p,
        stream,
    )?;
    let block = launch::Dim3 {
        x: 32,
        y: WARPS as u32,
        z: 1,
    };
    let warp_kernel = match family {
        Family::Tiles => None,
        Family::Warp(w) => Some(w.kernel(o.glu)),
        Family::Hybrid(h) => Some(h.kernel(o.glu)),
    };
    if let Some(kernel) = warp_kernel {
        // Work items: (distinct expert, 2 rows), at most one expert per pair; one wave of
        // resident CTAs, no more warps than items.
        let items = o.rows / WARP_ROWS as usize * o.n_used * o.n_tok;
        let grid = launch::Dim3 {
            x: items
                .div_ceil(WARPS as usize)
                .min(launch::resident(kernel, 32 * WARPS as u32))
                .max(1) as u32,
            y: 1,
            z: 1,
        };
        return launch::launch(kernel, grid, block, &mut p, stream);
    }
    let ti = tile(o.glu, o.k);
    let kernel = kernel_index(ti, o.glu, iters(o.k), users(o.n_tok));
    // Work items: (distinct expert, tile), at most one expert per pair; one wave of resident
    // CTAs takes them in turn.
    let n_tiles = o.rows / (WARPS as usize * TILES[ti].rows as usize);
    let items = n_tiles * o.n_used * o.n_tok;
    let grid = launch::Dim3 {
        x: items
            .min(launch::resident(kernel, 32 * WARPS as u32))
            .max(1) as u32,
        y: 1,
        z: 1,
    };
    launch::launch(kernel, grid, block, &mut p, stream)
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    /// `dim3`.
    #[repr(C)]
    pub(super) struct Dim3 {
        pub x: u32,
        pub y: u32,
        pub z: u32,
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

    /// Kernels of the table: the quantization, the tiles' matvecs, the warp-item matvecs, the
    /// hybrid warp matvecs.
    const KERNELS: usize = 1
        + 4 * 3 * super::TILES.len()
        + 2 * super::WARP_USERS.len() * 4
        + 2 * super::HYBRID_UMAX.len() * super::HYBRID_OCC.len();

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs): the quantization,
        /// then the matvecs of each of [`super::TILES`] (`super::kernel_index`), then the
        /// warp-item matvecs ([`super::Warp::kernel`]), then the hybrid ones
        /// ([`super::Hybrid::kernel`]).
        static legion_nvfp4_once_kernels: [*const c_void; KERNELS];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            blocks: *mut c_int,
            func: *const c_void,
            block_size: c_int,
            dynamic_smem: usize,
        ) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn legion_nvfp4_once_plan(ptr: *mut *mut c_void) -> c_int;
    }

    /// Each device's routing plan address; 0 not yet read.
    static PLAN: [AtomicUsize; DEVICES] = [const { AtomicUsize::new(0) }; DEVICES];

    /// The current device's routing plan (`plan` in `src/cuda/nvfp4_once.cu`), read once per
    /// device.
    pub(super) fn plan() -> Result<*mut c_void, String> {
        let dev = device().ok_or("nvfp4_once: no current CUDA device")?;
        let mut a = PLAN[dev].load(Ordering::Relaxed);
        if a == 0 {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: the address of a device global of this device.
            match unsafe { legion_nvfp4_once_plan(&mut ptr) } {
                0 if !ptr.is_null() => a = ptr as usize,
                e => {
                    // SAFETY: clears the error just reported.
                    unsafe { cudaGetLastError() };
                    return Err(format!("nvfp4_once: plan address, CUDA error {e}"));
                }
            }
            PLAN[dev].store(a, Ordering::Relaxed);
        }
        Ok(a as *mut c_void)
    }

    const DEV_ATTR_SM_COUNT: c_int = 16;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const DEV_ATTR_CC_MINOR: c_int = 76;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// Each device's compute capability (10 * major + minor); 0 not yet read, -1 unknown.
    static CC: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// Each device's resident CTAs of each kernel (SMs times CTAs per SM); 0 not yet read.
    static RESIDENT: [[AtomicI32; KERNELS]; DEVICES] =
        [const { [const { AtomicI32::new(0) }; KERNELS] }; DEVICES];

    fn device() -> Option<usize> {
        let mut dev: c_int = 0;
        // SAFETY: a plain query of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                cudaGetLastError();
                return None;
            }
        }
        Some(dev as usize)
    }

    /// The current device's compute capability, read once per device.
    fn compute_capability() -> Option<i32> {
        let dev = device()?;
        let slot = &CC[dev];
        let mut cc = slot.load(Ordering::Relaxed);
        if cc == 0 {
            let (mut major, mut minor) = (0, 0);
            // SAFETY: plain queries of the current device.
            cc = unsafe {
                if cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev as c_int) == 0
                    && cudaDeviceGetAttribute(&mut minor, DEV_ATTR_CC_MINOR, dev as c_int) == 0
                {
                    10 * major + minor
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(cc, Ordering::Relaxed);
        }
        (cc > 0).then_some(cc)
    }

    /// CTAs of `threads` threads of kernel `kernel` the current device holds at once (one wave),
    /// read once per device and kernel; 48 (a GB10's SMs, one CTA each) if unknown.
    pub(super) fn resident(kernel: usize, threads: u32) -> usize {
        let Some(dev) = device() else {
            return 48;
        };
        let slot = &RESIDENT[dev][kernel];
        let mut n = slot.load(Ordering::Relaxed);
        if n == 0 {
            let (mut sms, mut per_sm) = (0, 0);
            // SAFETY: plain queries of the current device and of a kernel of the table.
            n = unsafe {
                if cudaDeviceGetAttribute(&mut sms, DEV_ATTR_SM_COUNT, dev as c_int) == 0
                    && cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &mut per_sm,
                        legion_nvfp4_once_kernels[kernel],
                        threads as c_int,
                        0,
                    ) == 0
                {
                    (sms * per_sm).max(1)
                } else {
                    cudaGetLastError();
                    48
                }
            };
            slot.store(n, Ordering::Relaxed);
        }
        n as usize
    }

    /// Launches kernel `kernel` of the table on `p` (by value) on `stream`; on compute
    /// capability 12.x (GB10, RTX 50) as a programmatic dependent launch, as ggml-cuda launches
    /// its own kernels there (each kernel waits for the one ahead before it reads activations,
    /// `src/cuda/nvfp4_once.cu`).
    pub(super) fn launch(
        kernel: usize,
        grid: Dim3,
        block: Dim3,
        p: &mut super::Params,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut argv = [(p as *mut super::Params).cast::<c_void>()];
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
            num_attrs: u32::from(compute_capability().is_some_and(|cc| cc / 10 == 12)),
        };
        // SAFETY: a kernel of the table taking `legion_nvfp4_once` by value, from the launch's
        // copy of `p`; its pointers are device memory of the stream's device (the node's).
        match unsafe {
            cudaLaunchKernelExC(
                &config,
                legion_nvfp4_once_kernels[kernel],
                argv.as_mut_ptr(),
            )
        } {
            0 => Ok(()),
            e => Err(format!("nvfp4_once: CUDA error {e}")),
        }
    }
}
