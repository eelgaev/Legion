//! Paged attention: attention that reads each sequence's K/V blocks in place, through its block
//! table, instead of gathering them into a contiguous span first ([`super::KvPool::read`]).
//!
//! One external op ([`crate::ggml::ExternalOp`]) per attention layer and group. On CUDA it runs
//! the kernels of `src/cuda/paged_attn/` (flash-decoding for one-token groups, a tensor-core
//! kernel for prefill chunks); on the CPU a plain reference implementation, used by tests to
//! check the graph wiring (the CPU engine keeps the gather path, which matches llama.cpp bit
//! for bit).
//!
//! Inputs, besides the query: the group's block tables (I32 `[table_width, n_seqs]`, each
//! sequence's blocks in order) and positions (I32 `[n_seqs]`, the position of each sequence's
//! first token). Neither the span length nor a mask is part of the graph, so a step's graph
//! depends on the batch's shape only, not on how long its sequences are.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

use super::KV_BLOCK;

/// Upper bound on CTAs a decode launch aims for (flash-decoding splits fill the rest).
const DECODE_TARGET_CTAS: usize = 512;
/// Most splits per sequence.
const MAX_SPLIT: usize = 32;

static PAGED_ATTN: ExternalOp = ExternalOp {
    name: "paged_attn",
    cpu: cpu_paged_attn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_paged_attn),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Splits per sequence a decode launch may use for `n_seqs` sequences with `n_head_kv` kv heads.
pub fn decode_splits(n_seqs: usize, n_head_kv: usize) -> usize {
    (DECODE_TARGET_CTAS / (n_seqs * n_head_kv).max(1)).clamp(1, MAX_SPLIT)
}

/// Most tokens per sequence of a group that [`set_verify_split`] splits over the key blocks.
pub const VERIFY_SPLIT_TOKENS: usize = 16;

/// Whether graphs built from now on split small multi-token groups over the key blocks
/// ([`set_verify_split`]).
static VERIFY_SPLIT: AtomicBool = AtomicBool::new(false);

/// Whether graphs built from now on run the attention of groups of 2 to
/// [`VERIFY_SPLIT_TOKENS`] tokens per sequence (speculative verify steps, short prompt chunks)
/// split over the key blocks on sm_80+ GPUs, as one-token groups are (flash-decoding: the sm_80+
/// prefill kernel per split, then a combine kernel), instead of one CTA per query tile and kv
/// head walking the whole context (`legion serve --paged-verify-split`, #697). Off by default:
/// the rows' sums are taken in another order. Not for pools with a host tier; sm_70 / sm_75
/// keep their kernels.
pub fn set_verify_split(on: bool) {
    VERIFY_SPLIT.store(on, Ordering::Relaxed);
}

/// Whether a group of `n_tok` tokens per sequence takes the split ([`set_verify_split`]) on the
/// current device.
fn verify_split(n_tok: usize) -> bool {
    if !(2..=VERIFY_SPLIT_TOKENS).contains(&n_tok) || !VERIFY_SPLIT.load(Ordering::Relaxed) {
        return false;
    }
    if !quant_ok() {
        static NOTE: std::sync::Once = std::sync::Once::new();
        NOTE.call_once(|| {
            tracing::warn!("--paged-verify-split needs a compute capability 8.0+ GPU; ignored")
        });
        return false;
    }
    true
}

/// [`attention_tiered`] without a tier (tests).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn attention_with<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    n_past: Tensor<'a>,
    scratch: Tensor<'a>,
    scale: f32,
    window: usize,
    ring: usize,
) -> Tensor<'a> {
    attention_tiered(
        ctx, q, k, v, table, n_past, scratch, scale, window, ring, None,
    )
}

/// The host tier of a pool a layer reads (#352): blocks from `hbm_blocks` on are in host
/// memory, and prefill copies them into the staging slots `k_stage` / `v_stage` (as many
/// blocks of the layer's K and V rows as `stage_blocks`) before reading them, a piece of at
/// most that many host blocks at a time (see `src/cuda/paged_attn/`).
#[derive(Clone, Copy)]
pub struct Tier<'a> {
    pub hbm_blocks: usize,
    pub stage_blocks: usize,
    pub k_stage: Tensor<'a>,
    pub v_stage: Tensor<'a>,
}

/// Attention of `q` (F32 `[d, n_head, n_tok, n_seqs]`, may be a strided view) over the pools
/// `k`, `v` (F16 `[d × n_head_kv, cells]`), causal from each sequence's position `n_past`,
/// with softmax scale `scale`. `scratch` holds the decode splits' partial results (see
/// [`scratch_shape`]). Returns F32 `[d, n_head, n_tok, n_seqs]`, contiguous. With
/// `window > 0`, a sliding window: a query at position `p` sees keys `p - window + 1 ..= p`
/// only (sliding-window attention). With `ring > 0` each table is a ring of `ring` blocks (a
/// sliding-window layer's ring cache, [`super::KvRing`]): position `p` is in entry
/// `(p / KV_BLOCK) % ring`.
///
/// Over a pool with a host tier `tier` (see [`Tier`]), prefill groups' `scratch` must have the
/// room [`tier_scratch_len`] gives.
#[allow(clippy::too_many_arguments)]
pub fn attention_tiered<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    n_past: Tensor<'a>,
    scratch: Tensor<'a>,
    scale: f32,
    window: usize,
    ring: usize,
    tier: Option<Tier<'a>>,
) -> Tensor<'a> {
    build(
        ctx,
        [q, k, v, table, n_past, scratch],
        scale,
        window,
        ring,
        tier,
        None,
        0,
    )
}

/// [`attention_tiered`] over a compact cache (#551, [`super::KvPool`]): `k` and `v` are views
/// of the same granules, `head_stride` values per kv head apart within a cell (`kv_row` =
/// `head_stride × n_head_kv`). Not a ring.
#[allow(clippy::too_many_arguments)]
pub fn attention_granules<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    n_past: Tensor<'a>,
    scratch: Tensor<'a>,
    scale: f32,
    window: usize,
    tier: Option<Tier<'a>>,
    head_stride: usize,
) -> Tensor<'a> {
    build(
        ctx,
        [q, k, v, table, n_past, scratch],
        scale,
        window,
        0,
        tier,
        None,
        head_stride,
    )
}

/// [`attention_tiered`] for a group of which only the last token of each sequence is read
/// afterwards (an MTP head's take-in pass: its earlier rows only write their K/V). `split`
/// (F32, [`split_floats`] elements for the group) is scratch for computing those rows split
/// over the key blocks where the CUDA prefill kernel would walk them in one or two CTAs per kv
/// head (sm_70 / sm_75's WMMA kernel, [`split_ok`]): the same bits for those rows, the other
/// rows' output left unwritten. Elsewhere (decode groups, other kernels, the CPU) it computes
/// what [`attention_tiered`] does.
#[allow(clippy::too_many_arguments)]
pub fn attention_last<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    n_past: Tensor<'a>,
    scratch: Tensor<'a>,
    scale: f32,
    window: usize,
    tier: Option<Tier<'a>>,
    split: Tensor<'a>,
) -> Tensor<'a> {
    build(
        ctx,
        [q, k, v, table, n_past, scratch],
        scale,
        window,
        0,
        tier,
        Some(split),
        0,
    )
}

/// The op's node. Parameters: scale, window, ring, the tier's (hbm_blocks, stage_blocks) (0 and
/// 0 without), the `src` index of the split scratch (0 without), and with a compact cache the
/// head stride (absent: the head size).
#[allow(clippy::too_many_arguments)]
fn build<'a>(
    ctx: &'a Context,
    args: [Tensor<'a>; 6],
    scale: f32,
    window: usize,
    ring: usize,
    tier: Option<Tier<'a>>,
    split: Option<Tensor<'a>>,
    head_stride: usize,
) -> Tensor<'a> {
    let q = args[0];
    let mut args = args.to_vec();
    let mut params = vec![scale.to_bits() as i32, window as i32, ring as i32, 0, 0, 0];
    if head_stride > 0 {
        params.push(head_stride as i32);
    }
    if let Some(t) = tier.filter(|t| t.stage_blocks > 0 && ring == 0) {
        args.extend([t.k_stage, t.v_stage]);
        params[3] = t.hbm_blocks as i32;
        params[4] = t.stage_blocks as i32;
    }
    if let Some(s) = split {
        params[5] = args.len() as i32;
        args.push(s);
    }
    ctx.external_with(&PAGED_ATTN, types::F32, q.ne(), &args, &params)
}

/// Rows of a split scratch's query tile (sm70/paged_split.cuh): the last token's query heads of
/// one kv head, padded to one WMMA tile.
const SPLIT_ROWS: usize = 16;

/// Sequences a split scratch ([`attention_last`]) needs room for at most: a group of more runs
/// them in turns (each sequence's lane is 8.6 MB per kv head at a 128K context).
pub const SPLIT_SEQS: usize = 2;

/// F32 elements of the split scratch ([`attention_last`]) for `n_seqs` sequences at a time over
/// `n_head_kv` kv heads and `table_width`-block tables: per sequence and kv head, every block's
/// scores (then weights) of the last token's rows, their max and sum per block, and their final
/// max and 1 / sum. A smaller scratch than a group's sequences need runs them in turns (at
/// least one sequence's room, else the op runs as [`attention_tiered`]).
pub fn split_floats(n_seqs: usize, n_head_kv: usize, table_width: usize) -> usize {
    let lane = table_width * (SPLIT_ROWS * KV_BLOCK + 2 * SPLIT_ROWS) + 2 * SPLIT_ROWS;
    n_seqs * n_head_kv * lane
}

/// Whether [`attention_last`] computes its rows split over the key blocks on the current
/// device (sm_70 / sm_75 prefill through the WMMA kernel; not with `--paged-prefill-m884`), so
/// that a graph needs its scratch.
pub fn split_ok() -> bool {
    #[cfg(feature = "cuda")]
    // SAFETY: queries the current device and a flag.
    return unsafe { legion_paged_attn_split_ok() != 0 };
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether graphs built from now on give groups of 2 to [`VERIFY_SPLIT_TOKENS`] tokens per
/// sequence (speculative verify steps, short prompt chunks) the scratch that splits all their
/// rows over the key blocks on sm_70 / sm_75 ([`split_ok`]; `src/cuda/sm70/paged_vsplit.cuh`,
/// #1211): the WMMA kernel's result, bit for bit, from CTAs spread over the whole GPU instead
/// of one per query tile and kv head walking the whole context twice. Off by default
/// (`legion serve --paged-verify-split-wmma` turns it on): at a short context its launches
/// cost more than the walk they split (a V100, 128 keys, 4 sequences: 0.13 ms against 0.10).
static WMMA_SPLIT: AtomicBool = AtomicBool::new(false);

/// Sets [`WMMA_SPLIT`].
pub fn set_verify_split_wmma(on: bool) {
    WMMA_SPLIT.store(on, Ordering::Relaxed);
}

/// The weights of at least this many key blocks at a time (or the whole table) in a scratch
/// of [`wmma_split_floats`]: fewer would take more launches than the split saves.
const WMMA_SPLIT_MIN_CHUNK: usize = 64;

/// F32 elements a group's scratch for [`WMMA_SPLIT`] takes at most when its sequences' whole
/// tables do not fit (64 MiB: one sequence of Qwen3.8-27B's verify step at a 256K context in two
/// chunks of weights); a group of more then runs its sequences in turns or its weights in chunks
/// ([`wmma_split_plan`]).
const WMMA_SPLIT_BUDGET: usize = 16 << 20;

/// Most CTAs (query tiles times kv heads times sequences) of the WMMA kernel a group may have
/// to take [`WMMA_SPLIT`]: more already spread over a V100's 80 SMs, and the split would run in
/// more turns than it saves.
const WMMA_SPLIT_MAX_CTAS: usize = 32;

/// Query tiles of a group of `n_tok` tokens per sequence with `g` query heads per kv head
/// (the WMMA kernel's CTAs per sequence and kv head: 64 / g tokens each), and the rows of the
/// largest, padded to whole 16-row WMMA tiles (`vsplit::tiles`, `vsplit::rows16`).
fn wmma_rows(g: usize, n_tok: usize) -> (usize, usize) {
    let qt = 64 / g;
    (n_tok.div_ceil(qt), (n_tok.min(qt) * g).next_multiple_of(16))
}

/// F32 elements of one (sequence, query tile, kv head) lane of the [`WMMA_SPLIT`] scratch: per
/// table entry each row's max and sum, each row's final max and 1 / sum, the output fragments
/// carried between chunks, and `chunk` blocks of F16 weights (`vsplit::lane`).
fn wmma_lane_floats(d: usize, nr: usize, table_width: usize, chunk: usize) -> usize {
    2 * table_width * nr + 2 * nr + nr * d + chunk * nr * KV_BLOCK / 2
}

/// F32 elements of the [`WMMA_SPLIT`] scratch for a layer of head size `d` and `n_head` query
/// heads over `n_head_kv` kv heads, for a group of `n_tok` tokens per sequence of `n_seqs`
/// sequences with `table_width`-entry tables: all of it at once if that fits in
/// [`WMMA_SPLIT_BUDGET`], else that budget or (if more) one sequence's room with the smallest
/// chunk. 0 when the group does not take the split.
pub fn wmma_split_floats(
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_seqs: usize,
    table_width: usize,
) -> usize {
    if !(2..=VERIFY_SPLIT_TOKENS).contains(&n_tok)
        || n_head_kv == 0
        || !n_head.is_multiple_of(n_head_kv)
    {
        return 0;
    }
    let (tiles, nr) = wmma_rows(n_head / n_head_kv, n_tok);
    let units = tiles * n_head_kv;
    if n_seqs * units > WMMA_SPLIT_MAX_CTAS {
        return 0;
    }
    let full = n_seqs * units * wmma_lane_floats(d, nr, table_width, table_width);
    let least = units * wmma_lane_floats(d, nr, table_width, WMMA_SPLIT_MIN_CHUNK.min(table_width));
    full.min(WMMA_SPLIT_BUDGET.max(least))
}

/// The [`WMMA_SPLIT`] scratch a group's attention layers share (the most any of them, of head
/// size and kv heads `dims`, needs; [`wmma_split_floats`]), or 0 when the group does not take
/// the split: off, not 2 to [`VERIFY_SPLIT_TOKENS`] tokens per sequence, or a device that does
/// not run the WMMA kernel.
pub fn wmma_split_len(
    dims: &[(i64, i64)],
    n_head: i64,
    n_tok: usize,
    n_seqs: usize,
    table_width: usize,
) -> usize {
    if !(2..=VERIFY_SPLIT_TOKENS).contains(&n_tok)
        || !WMMA_SPLIT.load(Ordering::Relaxed)
        || !split_ok()
    {
        return 0;
    }
    dims.iter()
        .map(|&(d, kv)| {
            wmma_split_floats(
                d as usize,
                n_head as usize,
                kv as usize,
                n_tok,
                n_seqs,
                table_width,
            )
        })
        .max()
        .unwrap_or(0)
}

/// How a launch uses a [`WMMA_SPLIT`] scratch of `len` elements: (sequences per run, key blocks
/// of weights per chunk), the pair with the fewest kernel launches (each run takes 4 plus 2 per
/// chunk), or `None` when not even one sequence with the smallest chunk fits (the op then runs
/// the WMMA kernel).
pub fn wmma_split_plan(
    len: usize,
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_seqs: usize,
    table_width: usize,
) -> Option<(usize, usize)> {
    if !(2..=VERIFY_SPLIT_TOKENS).contains(&n_tok)
        || n_head_kv == 0
        || !n_head.is_multiple_of(n_head_kv)
        || table_width == 0
    {
        return None;
    }
    let (tiles, nr) = wmma_rows(n_head / n_head_kv, n_tok);
    let units = tiles * n_head_kv;
    if n_seqs * units > WMMA_SPLIT_MAX_CTAS {
        return None;
    }
    let fixed = wmma_lane_floats(d, nr, table_width, 0);
    let per_block = nr * KV_BLOCK / 2;
    let min_chunk = WMMA_SPLIT_MIN_CHUNK.min(table_width);
    (1..=n_seqs)
        .filter_map(|s| {
            let lane = len / (s * units);
            let chunk = (lane.checked_sub(fixed)? / per_block).min(table_width);
            (chunk >= min_chunk).then_some((s, chunk))
        })
        .min_by_key(|&(s, chunk)| {
            (
                n_seqs.div_ceil(s) * (4 + 2 * table_width.div_ceil(chunk)),
                s,
            )
        })
}

/// F32 elements of a prefill group's scratch over a tiered pool with `stage_blocks` staging
/// slots: the rows' running (max, sum), then the pieces' I32 walk arrays (blocks and entries
/// per table entry, a count per sequence, the staged count) and the slots' pool blocks.
pub fn tier_scratch_len(
    n_head: usize,
    n_tok: usize,
    n_seqs: usize,
    table_width: usize,
    stage_blocks: usize,
) -> usize {
    2 * n_head * n_tok * n_seqs + 2 * table_width * n_seqs + n_seqs + 1 + stage_blocks
}

/// Shape of the scratch input for a group: `[d + 2, splits, n_head, n_seqs]` for one-token
/// groups (per split: max, sum, unnormalized output); with [`set_verify_split`] on a device that
/// takes it, [`split_scratch_shape`] for groups of up to [`VERIFY_SPLIT_TOKENS`] tokens; a
/// minimal placeholder otherwise.
pub fn scratch_shape(d: i64, n_head: i64, n_head_kv: i64, n_tok: usize, n_seqs: usize) -> [i64; 4] {
    if n_tok == 1 {
        let splits = decode_splits(n_seqs, n_head_kv as usize) as i64;
        [d + 2, splits, n_head, n_seqs as i64]
    } else if verify_split(n_tok) {
        split_scratch_shape(d, n_head, n_head_kv, n_tok, n_seqs)
    } else {
        [1, 1, 1, 1]
    }
}

/// The scratch that splits a group of `n_tok` tokens per sequence over the key blocks
/// ([`set_verify_split`]; the CUDA op takes the split on sm_80+ GPUs when it is given one):
/// `[d + 2, splits, n_head, n_tok × n_seqs]` with as many splits as a decode step of `n_seqs`
/// sequences ([`decode_splits`]), or the placeholder when that is one.
pub fn split_scratch_shape(
    d: i64,
    n_head: i64,
    n_head_kv: i64,
    n_tok: usize,
    n_seqs: usize,
) -> [i64; 4] {
    let splits = decode_splits(n_seqs, n_head_kv as usize) as i64;
    if splits > 1 {
        [d + 2, splits, n_head, (n_tok * n_seqs) as i64]
    } else {
        [1, 1, 1, 1]
    }
}

/// The op's operands, read from the node.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Operands {
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_seqs: usize,
    q: *const u8,
    q_nb: [usize; 4],
    k: *const u16,
    v: *const u16,
    /// Elements per cell: head_stride × n_head_kv.
    kv_row: usize,
    /// Elements from one kv head to the next in a cell: d, or a compact cache's granule
    /// ([`attention_granules`]).
    head_stride: usize,
    /// A Gemma 4 compact cache ([`attention_granules`]): V lies in K's cells, a staged piece
    /// copies K's cells only.
    compact: bool,
    /// The caches' types and bytes per cell (row).
    k_ty: ffi::ggml_type,
    v_ty: ffi::ggml_type,
    k_cell: usize,
    v_cell: usize,
    table: *const i32,
    table_width: usize,
    n_past: *const i32,
    scratch: *mut f32,
    max_split: usize,
    dst: *mut f32,
    scale: f32,
    /// Sliding window, 0 for none.
    window: usize,
    /// Table entries in a ring, 0 for none.
    ring: usize,
    /// The pool's host tier: (hbm_blocks, stage_blocks, k_stage, v_stage).
    tier: Option<(usize, usize, *const u8, *const u8)>,
    /// Elements of `scratch`.
    scratch_len: usize,
    /// Bytes of the smaller staging tensor (0 without a tier).
    stage_bytes: usize,
    /// The split scratch ([`attention_last`]) and its elements.
    split: Option<(*mut f32, usize)>,
}

impl Operands {
    /// # Safety
    /// `dst` must be a node made by [`attention_tiered`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        unsafe {
            let src = |i: usize| &*dst.src[i];
            let (q, k, v, table, n_past, scratch) =
                (src(0), src(1), src(2), src(3), src(4), src(5));
            let params = crate::ggml::external_params(dst);
            let d = q.ne[0] as usize;
            let compact = params[6] > 0;
            let head_stride = if compact { params[6] as usize } else { d };
            Self {
                d,
                n_head: q.ne[1] as usize,
                n_head_kv: k.ne[0] as usize / head_stride,
                head_stride,
                compact,
                n_tok: q.ne[2] as usize,
                n_seqs: q.ne[3] as usize,
                q: q.data as *const u8,
                q_nb: q.nb,
                k: k.data as *const u16,
                v: v.data as *const u16,
                kv_row: k.ne[0] as usize,
                k_ty: k.type_,
                v_ty: v.type_,
                k_cell: k.nb[1],
                v_cell: v.nb[1],
                table: table.data as *const i32,
                table_width: table.ne[0] as usize,
                n_past: n_past.data as *const i32,
                scratch: scratch.data as *mut f32,
                max_split: scratch.ne[1] as usize,
                dst: dst.data as *mut f32,
                scale: f32::from_bits(params[0] as u32),
                window: params[1] as usize,
                ring: params[2] as usize,
                tier: (params[4] > 0).then(|| {
                    (
                        params[3] as usize,
                        params[4] as usize,
                        src(6).data as *const u8,
                        src(7).data as *const u8,
                    )
                }),
                scratch_len: scratch.ne.iter().product::<i64>() as usize,
                stage_bytes: if params[4] > 0 {
                    ffi::ggml_nbytes(dst.src[6]).min(ffi::ggml_nbytes(dst.src[7]))
                } else {
                    0
                },
                split: (params[5] > 0).then(|| {
                    let t = src(params[5] as usize);
                    (t.data as *mut f32, t.ne.iter().product::<i64>() as usize)
                }),
            }
        }
    }
}

/// Rounds through F16, as ggml's CPU mul_mat does with the F32 operand of an F16 matmul.
fn round_f16(x: &mut [f32]) {
    let mut h = vec![0u16; x.len()];
    // SAFETY: equal lengths.
    unsafe {
        ffi::ggml_fp32_to_fp16_row(x.as_ptr(), h.as_mut_ptr(), x.len() as i64);
        ffi::ggml_fp16_to_fp32_row(h.as_ptr(), x.as_mut_ptr(), x.len() as i64);
    }
}

/// The CPU implementation: per (sequence, token, head) row, scores against every visible key,
/// softmax with the max subtracted, weighted sum of V, all in F32 except that the query and
/// the softmax weights are rounded to F16 first, as the gather path's F16 matmuls round them.
/// So it agrees with the gather path to summation order.
unsafe fn cpu_paged_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: called by ggml for a node made by `attention`, with host tensors.
    unsafe {
        let o = Operands::of(dst);
        let g = o.n_head / o.n_head_kv;
        let scale = o.scale;
        let rows = o.n_seqs * o.n_tok * o.n_head;
        let mut kf = vec![0f32; o.d];
        let mut q = vec![0f32; o.d];
        let mut scores = Vec::new();
        let mut out = vec![0f32; o.d];
        for row in (ith..rows).step_by(nth) {
            let h = row % o.n_head;
            let t = row / o.n_head % o.n_tok;
            let s = row / (o.n_head * o.n_tok);
            let kvh = h / g;
            q.copy_from_slice(std::slice::from_raw_parts(
                o.q.add(s * o.q_nb[3] + t * o.q_nb[2] + h * o.q_nb[1]) as *const f32,
                o.d,
            ));
            round_f16(&mut q);
            let visible = *o.n_past.add(s) as usize + t + 1;
            let first = if o.window > 0 {
                visible.saturating_sub(o.window)
            } else {
                0
            };
            let cell = |p: usize| {
                let entry = if o.ring > 0 {
                    p / KV_BLOCK % o.ring
                } else {
                    p / KV_BLOCK
                };
                let b = *o.table.add(s * o.table_width + entry) as usize;
                b * KV_BLOCK + p % KV_BLOCK
            };
            // Head kvh's values of a cell of the cache at `base` (type `ty`, `bytes` per cell).
            let head = |base: *const u16, ty, bytes: usize, c: usize, out: &mut [f32]| {
                if ty == types::F16 {
                    ffi::ggml_fp16_to_fp32_row(
                        base.add(c * o.kv_row + kvh * o.head_stride),
                        out.as_mut_ptr(),
                        o.d as i64,
                    );
                } else {
                    let raw = std::slice::from_raw_parts((base as *const u8).add(c * bytes), bytes);
                    let all = crate::ggml::dequantize(ty, raw, o.kv_row);
                    let at = kvh * o.head_stride;
                    out.copy_from_slice(&all[at..at + o.d]);
                }
            };
            scores.clear();
            for p in first..visible {
                head(o.k, o.k_ty, o.k_cell, cell(p), &mut kf);
                let dot: f32 = q.iter().zip(&kf).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in &mut scores {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in &mut scores {
                *x /= sum;
            }
            round_f16(&mut scores);
            out.fill(0.0);
            for (p, w) in (first..visible).zip(&scores) {
                head(o.v, o.v_ty, o.v_cell, cell(p), &mut kf);
                for (acc, x) in out.iter_mut().zip(&kf) {
                    *acc += w * x;
                }
            }
            std::slice::from_raw_parts_mut(o.dst.add(row * o.d), o.d).copy_from_slice(&out);
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct LaunchParams {
    q: *const f32,
    q_nb1: i64,
    q_nb2: i64,
    q_nb3: i64,
    k: *const u16,
    v: *const u16,
    kv_row: i64,
    table: *const i32,
    table_width: i32,
    n_past: *const i32,
    dst: *mut f32,
    scratch: *mut f32,
    max_split: i32,
    d: i32,
    n_head: i32,
    n_head_kv: i32,
    n_tok: i32,
    n_seqs: i32,
    scale: f32,
    window: i32,
    ring: i32,
    k_type: i32,
    v_type: i32,
    k_cell: i64,
    v_cell: i64,
    piece_blk: *const i32,
    piece_ent: *const i32,
    piece_n: *const i32,
    piece_cnt: *const i32,
    k_stage: *const u8,
    v_stage: *const u8,
    ml: *mut f32,
    piece: i32,
    split: *mut f32,
    split_seqs: i32,
    /// A Gemma 4 compact cache (#551): only the kernels instantiated for it read these.
    compact: i32,
    /// Elements from one kv head to the next in a compact cache's cell.
    head_stride: i64,
    /// With `split`: every row of a 2-16-token group split over the key blocks, the weights of
    /// this many blocks at a time ([`wmma_split_plan`]); 0: the last token's rows only.
    vsplit_chunk: i32,
}

/// The kernels' code for a cache type (`LEGION_KV_*` in paged_attn.cu), if they read it.
#[cfg(feature = "cuda")]
pub(crate) fn kernel_type(ty: ffi::ggml_type) -> Option<i32> {
    match ty {
        t if t == types::F16 => Some(0),
        t if t == types::Q8_0 => Some(1),
        t if t == ffi::ggml_type_GGML_TYPE_Q4_0 => Some(2),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_paged_attn_launch(p: *const LaunchParams, stream: *mut c_void) -> i32;
    fn legion_paged_attn_piece(
        p: *const LaunchParams,
        hbm_blocks: i32,
        cap: i32,
        piece: i32,
        src: *mut i32,
        stream: *mut c_void,
    ) -> i32;
    fn legion_paged_attn_quant_ok() -> i32;
    fn legion_paged_attn_split_ok() -> i32;
    fn legion_paged_attn_set_prefill(kernel: i32, this_thread: i32);
    #[cfg(test)]
    fn legion_paged_attn_m884_launches() -> i64;
    #[cfg(test)]
    fn legion_paged_attn_ffma_launches() -> i64;
    fn legion_paged_attn_vsplit_launches() -> i64;
    fn legion_paged_attn_set_qm(on: i32, this_thread: i32);
    fn legion_paged_attn_qm_launches() -> i64;
}

/// Prefill groups with heads of 256 or 512 on compute capability 12.x devices (a DGX Spark's
/// GB10, RTX 50) through the kernels that take the queries as the MMA's M dimension
/// (`src/cuda/sm80/paged_attn_qm.cuh`, #1133, #1181) where they cover them, or (`false`) through
/// `paged_prefill_mma` as before. The same bits either way; on by default (`legion serve
/// --no-paged-prefill-qm` turns it off). Other GPUs and ops keep their kernels. For every thread.
pub fn set_qm_prefill(on: bool) {
    #[cfg(feature = "cuda")]
    // SAFETY: sets a flag the launches read.
    unsafe {
        legion_paged_attn_set_qm(i32::from(on), 0)
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// [`set_qm_prefill`] for launches from the calling thread only (benches and tests); `None`
/// follows the setting for every thread again.
pub fn set_qm_prefill_here(on: Option<bool>) {
    #[cfg(feature = "cuda")]
    // SAFETY: sets a thread-local flag the launches read.
    unsafe {
        legion_paged_attn_set_qm(on.map_or(-1, i32::from), 1)
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// Launches of the queries-as-M prefill kernel ([`set_qm_prefill`]) from the calling thread so
/// far (0 without CUDA).
pub fn qm_launches() -> i64 {
    #[cfg(feature = "cuda")]
    // SAFETY: reads a thread-local counter.
    return unsafe { legion_paged_attn_qm_launches() };
    #[cfg(not(feature = "cuda"))]
    0
}

/// Prefill chunks (and speculative verify steps, several tokens per sequence) on sm_70 devices
/// through the m8n8k4 one-pass kernel (F16 caches), or (`false`) the default: the WMMA kernel
/// there. Other GPUs keep their default (the sm_80+ kernel; WMMA on sm_75). For every thread.
pub fn set_m884_prefill(on: bool) {
    #[cfg(feature = "cuda")]
    // SAFETY: sets a flag the launches read.
    unsafe {
        legion_paged_attn_set_prefill(i32::from(on), 0)
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// Prefill chunks on sm_75 devices through the one-pass FP32 FMA kernel (F16 caches; the
/// caller sets it for Turing GPUs without tensor cores, whose emulated HMMA runs the default
/// WMMA kernel slowly), or (`false`) the default. Not the WMMA kernel's bits. Overrides
/// [`set_m884_prefill`] (one choice for every thread; m8n8k4 applies on sm_70 only anyway).
pub fn set_ffma_prefill(on: bool) {
    #[cfg(feature = "cuda")]
    // SAFETY: sets a flag the launches read.
    unsafe {
        legion_paged_attn_set_prefill(if on { 2 } else { 0 }, 0)
    };
    #[cfg(not(feature = "cuda"))]
    let _ = on;
}

/// [`set_ffma_prefill`] for launches from the calling thread only, on any GPU (tests).
#[cfg(all(test, feature = "cuda"))]
fn set_ffma_prefill_here(on: bool) {
    // SAFETY: sets a thread-local flag the launches read.
    unsafe { legion_paged_attn_set_prefill(if on { 2 } else { 0 }, 1) };
}

/// Launches of the FP32 FMA prefill kernel from the calling thread so far (tests).
#[cfg(all(test, feature = "cuda"))]
fn ffma_launches() -> i64 {
    // SAFETY: reads a thread-local counter.
    unsafe { legion_paged_attn_ffma_launches() }
}

/// The calling thread reads `kernel` (`LEGION_PREFILL_*`, -1 to stop) in place of the choice
/// for every thread, with its per-architecture limits; other threads keep theirs (tests of
/// [`set_m884_prefill`] and [`set_ffma_prefill`] that run beside other GPU tests).
#[cfg(all(test, feature = "cuda"))]
fn set_prefill_for_all_here(kernel: i32) {
    // SAFETY: sets a thread-local flag the launches read.
    unsafe { legion_paged_attn_set_prefill(kernel, 2) };
}

/// [`set_m884_prefill`] for launches from the calling thread only (tests).
#[cfg(all(test, feature = "cuda"))]
fn set_m884_prefill_here(on: bool) {
    // SAFETY: sets a thread-local flag the launches read.
    unsafe { legion_paged_attn_set_prefill(i32::from(on), 1) };
}

/// Launches of the WMMA kernel's all-row split ([`WMMA_SPLIT`]) from the calling thread so far
/// (0 without CUDA).
pub(crate) fn vsplit_launches() -> i64 {
    #[cfg(feature = "cuda")]
    // SAFETY: reads a thread-local counter.
    return unsafe { legion_paged_attn_vsplit_launches() };
    #[cfg(not(feature = "cuda"))]
    0
}

/// Launches of the m8n8k4 prefill kernel from the calling thread so far (tests).
#[cfg(all(test, feature = "cuda"))]
fn m884_launches() -> i64 {
    // SAFETY: reads a thread-local counter.
    unsafe { legion_paged_attn_m884_launches() }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_paged_attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `attention`, all tensors in device memory of the stream's device.
    let o = unsafe { Operands::of(dst) };
    let (Some(k_type), Some(v_type)) = (kernel_type(o.k_ty), kernel_type(o.v_ty)) else {
        return Err(format!(
            "the paged kernels do not read {} / {} caches",
            crate::ggml::type_name(o.k_ty),
            crate::ggml::type_name(o.v_ty)
        ));
    };
    let mut p = LaunchParams {
        q: o.q as *const f32,
        q_nb1: o.q_nb[1] as i64,
        q_nb2: o.q_nb[2] as i64,
        q_nb3: o.q_nb[3] as i64,
        k: o.k,
        v: o.v,
        kv_row: o.kv_row as i64,
        table: o.table,
        table_width: o.table_width as i32,
        n_past: o.n_past,
        dst: o.dst,
        scratch: o.scratch,
        max_split: o.max_split as i32,
        d: o.d as i32,
        n_head: o.n_head as i32,
        n_head_kv: o.n_head_kv as i32,
        n_tok: o.n_tok as i32,
        n_seqs: o.n_seqs as i32,
        scale: o.scale,
        window: o.window as i32,
        ring: o.ring as i32,
        k_type,
        v_type,
        k_cell: o.k_cell as i64,
        v_cell: o.v_cell as i64,
        piece_blk: std::ptr::null(),
        piece_ent: std::ptr::null(),
        piece_n: std::ptr::null(),
        piece_cnt: std::ptr::null(),
        k_stage: std::ptr::null(),
        v_stage: std::ptr::null(),
        ml: std::ptr::null_mut(),
        piece: 0,
        split: std::ptr::null_mut(),
        split_seqs: 0,
        compact: i32::from(o.compact),
        head_stride: o.head_stride as i64,
        vsplit_chunk: 0,
    };
    // A multi-token group runs split over the key blocks (sm_80+) only with a scratch made for
    // it (split_scratch_shape); the placeholder and a tier's scratch have one split.
    if o.n_tok > 1 && o.scratch_len < (o.d + 2) * o.max_split * o.n_head * o.n_tok * o.n_seqs {
        p.max_split = 1;
    }
    // A prefill group's last rows split over the key blocks, in runs of as many sequences as
    // the scratch holds (the kernel takes it on sm_70 / sm_75's WMMA path only, and not for
    // staged pieces).
    if let Some((ptr, len)) = o.split.filter(|_| o.n_tok > 1) {
        let fit = len / split_floats(1, o.n_head_kv, o.table_width);
        if fit > 0 {
            p.split = ptr;
            p.split_seqs = fit.min(o.n_seqs) as i32;
        }
    } else if WMMA_SPLIT.load(Ordering::Relaxed)
        && !(o.tier.is_some() && staged(&o))
        && let Some((seqs, chunk)) = wmma_split_plan(
            o.scratch_len,
            o.d,
            o.n_head,
            o.n_head_kv,
            o.n_tok,
            o.n_seqs,
            o.table_width,
        )
    {
        // Every row of a 2-16-token group split over the key blocks (sm_70 / sm_75's WMMA path
        // only; the other kernels ignore it), given a scratch made for it (wmma_split_len).
        p.split = o.scratch;
        p.split_seqs = seqs as i32;
        p.vsplit_chunk = chunk as i32;
    }
    let check = |rc: i32| match rc {
        0 => Ok(()),
        1 => Err(format!(
            "unsupported shape: head size {}, {} heads over {} kv heads",
            o.d, o.n_head, o.n_head_kv
        )),
        e => Err(format!("CUDA launch failed ({e})")),
    };
    let Some((hbm, cap, k_stage, v_stage)) = o.tier.filter(|_| staged(&o)) else {
        // `--hopper-attn` (#1215): prefill on Hopper's wgmma kernel where it covers the op.
        let shape = hopper::Shape {
            d: o.d,
            n_head: o.n_head,
            n_head_kv: o.n_head_kv,
            n_tok: o.n_tok,
            f16: k_type == 0 && v_type == 0,
            compact: o.compact,
            tiered: o.tier.is_some(),
            max_split: p.max_split.max(1) as usize,
            k_cell: o.k_cell,
            v_cell: o.v_cell,
            // SAFETY: the op's K and V pools.
            k_cells: unsafe { (*dst.src[1]).ne[1] } as usize,
            v_cells: unsafe { (*dst.src[2]).ne[1] } as usize,
        };
        // SAFETY: the op's parameters and pools, on the stream's device.
        if unsafe { hopper::run(&p, &shape, stream) }? {
            return Ok(());
        }
        return check(unsafe { legion_paged_attn_launch(&p, stream) });
    };
    // Pieces (#352): at most `cap` host blocks each, as many as the step can reach.
    let pool_blocks = unsafe { (*dst.src[1]).ne[1] } as usize / KV_BLOCK;
    let reach = (pool_blocks.saturating_sub(hbm)).min(o.table_width * o.n_seqs);
    let pieces = reach.div_ceil(cap).max(1);
    let len = tier_scratch_len(o.n_head, o.n_tok, o.n_seqs, o.table_width, cap);
    if o.stage_bytes < cap * KV_BLOCK * o.k_cell.max(o.v_cell) {
        return Err(format!(
            "{cap} staging blocks of {}-byte cells need more than {} bytes",
            o.k_cell.max(o.v_cell),
            o.stage_bytes
        ));
    }
    if o.scratch_len < len {
        return Err(format!(
            "tiered prefill scratch of {} elements, needs {len}",
            o.scratch_len
        ));
    }
    let rows = 2 * o.n_head * o.n_tok * o.n_seqs;
    let ints = unsafe { o.scratch.add(rows) } as *mut i32;
    let tw = o.table_width * o.n_seqs;
    p.ml = o.scratch;
    p.piece_blk = ints;
    p.piece_ent = unsafe { ints.add(tw) };
    p.piece_n = unsafe { ints.add(2 * tw) };
    p.piece_cnt = unsafe { ints.add(2 * tw + o.n_seqs) };
    let src = unsafe { ints.add(2 * tw + o.n_seqs + 1) };
    p.k_stage = k_stage;
    p.v_stage = if o.compact {
        // V's cells are K's, from an offset: so are their staged copies.
        unsafe { k_stage.add(o.v as usize - o.k as usize) }
    } else {
        v_stage
    };
    for j in 0..pieces {
        p.piece = j as i32;
        // SAFETY: the arrays are in the scratch (checked above), on the op's device.
        check(unsafe {
            legion_paged_attn_piece(&p, hbm as i32, cap as i32, j as i32, src, stream)
        })?;
        check(unsafe { legion_paged_attn_launch(&p, stream) })?;
    }
    Ok(())
}

/// Whether a group reads a tiered pool through staging slots: prefill with more than one
/// query tile (64 query rows, `n_tok × G`), each of which would read every host block over the
/// link (one tile reads them once anyway).
#[cfg(feature = "cuda")]
fn staged(o: &Operands) -> bool {
    o.n_tok > 1 && o.ring == 0 && o.n_tok * (o.n_head / o.n_head_kv.max(1)) > 64
}

/// Whether the CUDA paged kernels read caches of `types` on the current device: F16 always;
/// q8_0 and q4_0 (K and V each) with the sm_80+ prefill kernel.
pub fn reads_types(types: super::KvTypes) -> bool {
    use super::KvType::*;
    let quantized = |t| matches!(t, Q8_0 | Q4_0);
    types.is_f16()
        || (matches!(types.k, F16 | Q8_0 | Q4_0)
            && matches!(types.v, F16 | Q8_0 | Q4_0)
            && (quantized(types.k) || quantized(types.v))
            && quant_ok())
        || tq::reads(types)
}

/// The cache types of an MTP head's own pool (its dense attention layer) next to a target cache
/// of `types`, on a GPU (`gpu`) for a model with sparse attention (`sparse`, whose kernels read
/// q8_0 / q4_0 on every GPU, #263): F16 where the paged kernels do not read `types` on this
/// device (sm_70), so the head pages: one layer's cache, against a gather over the context per
/// draft. Otherwise `types`.
pub fn mtp_head_types(types: super::KvTypes, gpu: bool, sparse: bool) -> super::KvTypes {
    if gpu && sparse && !reads_types(types) {
        super::KvTypes::default()
    } else {
        types
    }
}

#[cfg(feature = "cuda")]
fn quant_ok() -> bool {
    // SAFETY: queries the current device.
    unsafe { legion_paged_attn_quant_ok() != 0 }
}

#[cfg(not(feature = "cuda"))]
fn quant_ok() -> bool {
    false
}

/// Whether the CUDA kernels support a model's attention shape: head sizes 128, 256 and 512, and
/// the GQA ratios (query heads per kv head) the kernels are instantiated for, `launch_d` in
/// `src/cuda/paged_attn/` (6: Qwen3.5 / 3.6 / 3.8-27B, 24 query heads over 4 kv heads).
pub fn supported(head_dim: i64, n_head: i64, n_head_kv: i64) -> bool {
    matches!(head_dim, 128 | 256 | 512)
        && n_head_kv > 0
        && n_head % n_head_kv == 0
        && matches!(n_head / n_head_kv, 1 | 2 | 4 | 6 | 8 | 12 | 16)
}

#[cfg(all(test, feature = "cuda"))]
mod ffma_tests;
#[cfg(test)]
mod granule_tests;
pub mod hopper;
#[cfg(all(test, feature = "cuda"))]
mod m884_tests;
pub mod qm_bench;
#[cfg(test)]
mod quant_tests;
#[cfg(test)]
mod ring_tests;
#[cfg(test)]
mod split_tests;
#[cfg(test)]
mod tests;
#[cfg(all(test, feature = "cuda"))]
mod tier_tests;
pub mod tq;
#[cfg(test)]
mod vsplit_tests;
pub mod wmma_bench;
#[cfg(test)]
mod wmma_split_tests;
