//! qwen4exp's sparse (QSA) attention as it runs in big steps: every query attends to its own
//! selection only (the `top_k / kpool` best pools plus its incomplete pool), read in place from
//! the KV pool through the sequence's block table. The gather path ([`crate::model::core::Model::qsa_mask`])
//! computes the same thing as dense attention over the whole span plus a mask, which costs
//! `span × tokens × heads` scores and masks of that size: fine for decode and as the CPU
//! oracle, out of memory for long prompts.
//!
//! External ops ([`crate::ggml::ExternalOp`]), each with a CPU implementation (tests) and
//! a CUDA kernel (`src/cuda/qsa_attn.cu`):
//!
//! - [`pool_mask`]: the indexer's pool mask, F16 `[n_pool, tokens]`, built on the device from
//!   each token's range of visible pools (two integers per token) instead of being filled on
//!   the host (126M entries per 4K chunk at 123K context).
//! - [`select`]: from the indexer's `top_k` result, each token's selected positions in
//!   ascending order. The selected pools are sorted by position, their members follow in
//!   order, then the token's own incomplete pool. The order of `top_k`'s output does not
//!   matter, so the attention sums in the same order on every backend and for every `top_k`
//!   budget.
//! - [`top_k`]: the indexer's best pools, one launch (ggml's `top_k` as a single histogram
//!   pass; the same set as ggml's but for ties at the threshold).
//! - [`attention`]: attention of every query head over its token's selected positions.
//!
//! Selection rows (I32 `[1 + capacity, tokens]`): the count, then that many positions, then -1.
//! A pool is selected when its score is finite (an invisible pool scores -inf through the
//! mask); a tail position when it is at or before the token (the padding sentinel never is).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

use crate::model::core::KV_BLOCK;

static POOL_MASK: ExternalOp = ExternalOp {
    name: "qsa_pool_mask",
    cpu: cpu_pool_mask,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_pool_mask),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SELECT: ExternalOp = ExternalOp {
    name: "qsa_select",
    cpu: cpu_select,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_select),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SELECT_INTO: ExternalOp = ExternalOp {
    name: "qsa_select_into",
    cpu: cpu_select_into,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_select_into),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SELECT_ALL: ExternalOp = ExternalOp {
    name: "qsa_select_all",
    cpu: cpu_select_all,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_select_all),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SELECT_VIS: ExternalOp = ExternalOp {
    name: "qsa_select_vis",
    cpu: cpu_select_vis,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_select_vis),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SCORES: ExternalOp = ExternalOp {
    name: "qsa_scores",
    cpu: cpu_scores,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_scores),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static SCORES_EXACT: ExternalOp = ExternalOp {
    name: "qsa_scores_exact",
    cpu: cpu_scores_exact,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_scores_exact),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static TOPK: ExternalOp = ExternalOp {
    name: "qsa_topk",
    cpu: cpu_topk,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_topk),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ATTN: ExternalOp = ExternalOp {
    name: "qsa_attn",
    cpu: cpu_attn,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_attn),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static ATTN_CODES: ExternalOp = ExternalOp {
    name: "qsa_attn_codes",
    cpu: cpu_attn_codes,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_attn_codes),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// F16 0 and -inf.
const KEEP: u16 = 0x0000;
const DROP: u16 = 0xfc00;

/// Most pools a token may select on CUDA (the select kernel sorts them in shared memory).
pub const CUDA_MAX_TOP: usize = 1024;

/// The pool mask for the lightning indexer: F16 `[n_pool, tokens]`, 0 where `first <= j <
/// end` and -inf elsewhere, from `vis` (I32 `[2, tokens]`: each token's `first, end`).
pub fn pool_mask<'a>(ctx: &'a Context, vis: Tensor<'a>, n_pool: i64) -> Tensor<'a> {
    let ng = vis.ne()[1];
    ctx.external(&POOL_MASK, types::F16, [n_pool, ng, 1, 1], &[vis])
}

/// Whether [`scores_fit`] was asked for ([`set_scores_tc`]).
static SCORES_TC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Lets [`scores`] take prompt steps' pool scores from now on where it fits (`legion serve
/// --qsa-scores-tc`; off by default until its selection shift is measured).
pub fn set_scores_tc(on: bool) {
    SCORES_TC.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`scores`] takes a step's pool scores: asked for ([`set_scores_tc`]), on CUDA devices
/// of compute capability 7.x only (the hi / lo split pays for Volta's F16-only tensor cores;
/// newer GPUs keep ggml's indexer), the indexer shape the kernel is written for (4 heads of 128
/// dims) and a prompt chunk (at least a 16-query tile of tokens).
pub fn scores_fit(n_head: i64, head_dim: i64, tokens: i64) -> bool {
    static DEVICES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    SCORES_TC.load(std::sync::atomic::Ordering::Relaxed)
        && *DEVICES.get_or_init(|| {
            let fit = scores_devices();
            if !fit {
                tracing::info!(
                    "--qsa-scores-tc ignored: not every CUDA device is compute capability 7.x"
                );
            }
            fit
        })
        && n_head == SCORE_HEADS
        && head_dim == SCORE_DIM
        && tokens >= 16
}

/// Every visible CUDA device is compute capability 7.x.
#[cfg(feature = "cuda")]
fn scores_devices() -> bool {
    // SAFETY: reads device attributes only.
    unsafe { cuda::legion_qsa_scores_sm7x() == 1 }
}

#[cfg(not(feature = "cuda"))]
fn scores_devices() -> bool {
    false
}

/// The indexer shape [`scores`] takes.
const SCORE_HEADS: i64 = 4;
const SCORE_DIM: i64 = 128;

/// The lightning indexer's pool scores with the pool mask applied, as ggml's
/// `lightning_indexer(q, keys, w, pool_mask(vis))`: F32 `[n_pool, tokens]`, `sum_h max(q_h .
/// key_j, 0) * w_h` for each token's visible pools (`first <= j < end` of `vis`), -inf for the
/// others. `q` F32 `[d, n_head, tokens]` (rows contiguous, a strided view allowed), `w` F32
/// `[n_head, tokens]`. `keys` F32 `[d, 1, n_pool]`, or with `cells` (I32 `[n_pool]`) F32 or F16
/// `[d, rows]` with pool j's key in row `cells[j]` (`get_rows(keys, cells)` in the op). `bias` (F32
/// `[n_pool]`) is added to the visible scores (`add(scores, bias)` in the op). On CUDA the
/// scores run on Volta's tensor cores (FP32 operands split into FP16 high and low parts,
/// adapted from Strata-AC922 52a35fe): FP32-level, not bitwise ggml's.
pub fn scores<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    keys: Tensor<'a>,
    w: Tensor<'a>,
    vis: Tensor<'a>,
    bias: Option<Tensor<'a>>,
    cells: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let n_pool = cells.map_or(keys.ne()[2], |c| c.ne()[0]);
    let mut args = vec![q, keys, w, vis];
    args.extend(bias);
    args.extend(cells);
    ctx.external_with(
        &SCORES,
        types::F32,
        [n_pool, q.ne()[2], 1, 1],
        &args,
        &[bias.is_some() as i32, cells.is_some() as i32],
    )
}

/// A [`scores`] node's optional sources: (bias, cells).
///
/// # Safety
/// `dst` is a node made by [`scores`].
unsafe fn scores_extra(
    dst: &ffi::ggml_tensor,
) -> (Option<&ffi::ggml_tensor>, Option<&ffi::ggml_tensor>) {
    let [has_bias, has_cells, ..] = crate::ggml::external_params(dst);
    // SAFETY: `scores` put the bias, then the cells, after its four sources.
    unsafe {
        let bias = (has_bias != 0).then(|| &*dst.src[4]);
        let cells = (has_cells != 0).then(|| &*dst.src[4 + has_bias as usize]);
        (bias, cells)
    }
}

/// ggml's ops instead of [`scores_exact`] and [`top_k`] (`legion serve --qsa-scores-ggml`,
/// `--qsa-topk-ggml`), for comparisons.
static SCORES_GGML: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TOPK_GGML: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Makes the paged path take ggml's indexer chain (`scores`) and ggml's `top_k` (`topk`)
/// instead of [`scores_exact`] and [`top_k`], for every model from now on.
pub fn set_ggml_paths(scores: bool, topk: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    SCORES_GGML.store(scores, Relaxed);
    TOPK_GGML.store(topk, Relaxed);
}

/// Whether [`scores_exact`] takes a step's pool scores: the indexer shape its kernel is written
/// for (4 heads of 128 dims) and a prompt chunk (at least ggml's 8-token tile, below which ggml
/// takes another kernel with another summation order), unless [`set_ggml_paths`] asked for
/// ggml's chain (mask, lightning indexer, bias add).
pub fn scores_exact_fit(n_head: i64, head_dim: i64, tokens: i64) -> bool {
    !SCORES_GGML.load(std::sync::atomic::Ordering::Relaxed)
        && n_head == SCORE_HEADS
        && head_dim == SCORE_DIM
        && tokens >= 8
}

/// `add(lightning_indexer(q, keys, w, pool_mask(vis)), bias)` bit for bit as ggml computes it on
/// CUDA for a prompt chunk (its tile kernel: keys rounded to F16, each head's dot product an FMA
/// chain over the dims in order, then `score += relu(qk) · w_h` over the heads in order, plus
/// the mask, plus the bias), in one op: no mask tensor, no separate add, more of the products
/// per thread. Shapes as [`scores`], `bias` F32 `[n_pool]`.
pub fn scores_exact<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    keys: Tensor<'a>,
    w: Tensor<'a>,
    vis: Tensor<'a>,
    bias: Tensor<'a>,
) -> Tensor<'a> {
    let (n_pool, ng) = (keys.ne()[2], q.ne()[2]);
    ctx.external(
        &SCORES_EXACT,
        types::F32,
        [n_pool, ng, 1, 1],
        &[q, keys, w, vis, bias],
    )
}

unsafe fn cpu_scores_exact(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `scores_exact`, host tensors.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (q, keys, w, vis, bias) = (src(0), src(1), src(2), src(3), src(4));
        let (d, n_head, ng) = (q.ne[0] as usize, q.ne[1] as usize, q.ne[2] as usize);
        let n_pool = dst.ne[0] as usize;
        let row = |base: *const u8, off: usize, n: usize| {
            std::slice::from_raw_parts(base.add(off) as *const f32, n)
        };
        let bias = row(bias.data as *const u8, 0, n_pool);
        let mut key = vec![0f32; d];
        for t in (ith..ng).step_by(nth) {
            let first = at_i32(vis.data as *const u8, 2 * t) as usize;
            let end = at_i32(vis.data as *const u8, 2 * t + 1) as usize;
            let wt = row(w.data as *const u8, t * w.nb[1], n_head);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut f32,
                n_pool,
            );
            for (j, o) in out.iter_mut().enumerate() {
                key.copy_from_slice(row(keys.data as *const u8, j * keys.nb[2], d));
                round_f16(&mut key);
                let mut score = 0f32;
                for (h, &wh) in wt.iter().enumerate() {
                    let qh = row(q.data as *const u8, t * q.nb[2] + h * q.nb[1], d);
                    let qk = qh
                        .iter()
                        .zip(&key)
                        .fold(0f32, |acc, (a, b)| b.mul_add(*a, acc));
                    score = qk.max(0.0).mul_add(wh, score);
                }
                let m = if j >= first && j < end {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
                *o = (score + m) + bias[j];
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_scores_exact(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `scores_exact`, device tensors of the stream's device.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (q, keys, w, vis, bias) = (src(0), src(1), src(2), src(3), src(4));
        if q.ne[0] != SCORE_DIM || q.ne[1] != SCORE_HEADS || q.nb[0] != 4 || keys.nb[0] != 4 {
            return Err(format!(
                "qsa_scores_exact: {} heads of {} dims (the kernel takes {SCORE_HEADS} of {SCORE_DIM})",
                q.ne[1], q.ne[0]
            ));
        }
        let p = cuda::ScoresExact {
            q: q.data as *const f32,
            q_nb1: q.nb[1] as i64,
            q_nb2: q.nb[2] as i64,
            keys: keys.data as *const f32,
            k_nb: keys.nb[2] as i64,
            w: w.data as *const f32,
            w_nb1: w.nb[1] as i64,
            vis: vis.data as *const i32,
            bias: bias.data as *const f32,
            dst: dst.data as *mut f32,
            dst_nb1: dst.nb[1] as i64,
            n_pool: dst.ne[0] as i32,
            ng: dst.ne[1] as i32,
        };
        launched(
            "qsa_scores_exact",
            cuda::legion_qsa_scores_exact_launch(&p, stream),
        )
    }
}

unsafe fn cpu_scores(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `scores`, host tensors.
    unsafe {
        let (q, keys, w, vis) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let (bias, cells) = scores_extra(dst);
        let (d, n_head, ng) = (q.ne[0] as usize, q.ne[1] as usize, q.ne[2] as usize);
        let n_pool = dst.ne[0] as usize;
        let row = |base: *const u8, off: usize| {
            std::slice::from_raw_parts(base.add(off) as *const f32, d)
        };
        let mut k = vec![0f32; d];
        // pool j's key: row cells[j] of [d, rows], or pool j of [d, 1, n_pool]
        let key_off = |j: usize| match cells {
            Some(c) => at_i32(c.data as *const u8, j) as usize * keys.nb[1],
            None => j * keys.nb[2],
        };
        for t in (ith..ng).step_by(nth) {
            let first = at_i32(vis.data as *const u8, 2 * t) as usize;
            let end = at_i32(vis.data as *const u8, 2 * t + 1) as usize;
            let wt = std::slice::from_raw_parts(
                (w.data as *const u8).add(t * w.nb[1]) as *const f32,
                n_head,
            );
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut f32,
                n_pool,
            );
            for (j, o) in out.iter_mut().enumerate() {
                if j < first || j >= end {
                    *o = f32::NEG_INFINITY;
                    continue;
                }
                let at = (keys.data as *const u8).add(key_off(j));
                if keys.type_ == types::F16 {
                    ffi::ggml_fp16_to_fp32_row(at as *const u16, k.as_mut_ptr(), d as i64);
                } else {
                    k.copy_from_slice(row(at, 0));
                }
                let mut s = 0f32;
                for (h, &wh) in wt.iter().enumerate() {
                    let qh = row(q.data as *const u8, t * q.nb[2] + h * q.nb[1]);
                    let dot: f32 = qh.iter().zip(&k).map(|(a, b)| a * b).sum();
                    s += dot.max(0.0) * wh;
                }
                if let Some(b) = bias {
                    s += *(b.data as *const f32).add(j);
                }
                *o = s;
            }
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_scores(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `scores`, device tensors of the stream's device.
    unsafe {
        let (q, keys, w, vis) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let (bias, cells) = scores_extra(dst);
        if q.ne[0] != SCORE_DIM || q.ne[1] % SCORE_HEADS != 0 || q.ne[1] == 0 {
            return Err(format!(
                "qsa_scores: {} heads of {} dims (the kernel takes a multiple of {SCORE_HEADS} \
                 of {SCORE_DIM})",
                q.ne[1], q.ne[0]
            ));
        }
        let k_f16 = keys.type_ == types::F16;
        if q.nb[0] != 4 || keys.nb[0] != if k_f16 { 2 } else { 4 } {
            return Err("qsa_scores: q and keys must be contiguous rows".into());
        }
        let p = cuda::Scores {
            q: q.data as *const f32,
            q_nb1: q.nb[1] as i64,
            q_nb2: q.nb[2] as i64,
            keys: keys.data as *const f32,
            k_nb: if cells.is_some() {
                keys.nb[1]
            } else {
                keys.nb[2]
            } as i64,
            w: w.data as *const f32,
            w_nb1: w.nb[1] as i64,
            vis: vis.data as *const i32,
            dst: dst.data as *mut f32,
            dst_nb1: dst.nb[1] as i64,
            n_pool: dst.ne[0] as i32,
            ng: dst.ne[1] as i32,
            bias: bias.map_or(std::ptr::null(), |b| b.data as *const f32),
            rows: cells.map_or(std::ptr::null(), |c| c.data as *const i32),
            k_f16: k_f16 as i32,
            n_head: q.ne[1] as i32,
        };
        launched("qsa_scores", cuda::legion_qsa_scores_launch(&p, stream))
    }
}

/// Each token's selected positions (see the module docs): I32 `[1 + n_top × kpool + kpool - 1,
/// tokens]`. `top` (I32 `[n_top, tokens]`) are the selected pools (indices into the listing),
/// `score` (F32 `[n_pool, tokens]`) their scores, `pool_idxs` (I32 `[kpool, n_pool]`) every
/// listed pool's member positions, `tail` (I32 `[kpool - 1, tokens]`) the positions of each
/// token's own incomplete pool, `pos` (I32 `[tokens]`) each token's position.
pub fn select<'a>(
    ctx: &'a Context,
    top: Tensor<'a>,
    score: Tensor<'a>,
    pool_idxs: Tensor<'a>,
    tail: Tensor<'a>,
    pos: Tensor<'a>,
) -> Tensor<'a> {
    select_rows(ctx, top, score, pool_idxs, tail, pos, top.ne()[1])
}

/// [`select`] of the first `top.ne[1]` of a step's `tokens` tokens, in a node with a row for
/// every token: [`select_into`] writes the later tokens' rows over it in place (#1183: a
/// prompt step's selections chunk by chunk without a concat). The rows past `top`'s tokens
/// are the node's until then.
pub fn select_rows<'a>(
    ctx: &'a Context,
    top: Tensor<'a>,
    score: Tensor<'a>,
    pool_idxs: Tensor<'a>,
    tail: Tensor<'a>,
    pos: Tensor<'a>,
    tokens: i64,
) -> Tensor<'a> {
    let n_top = top.ne()[0];
    assert!(top.ne()[1] <= tokens, "qsa_select: more tokens than rows");
    let kpool = pool_idxs.ne()[0];
    let cap = n_top * kpool + kpool - 1;
    ctx.external(
        &SELECT,
        types::I32,
        [1 + cap, tokens, 1, 1],
        &[top, score, pool_idxs, tail, pos],
    )
}

/// [`select`] of tokens `first..first + top.ne[1]` written over those rows of `rows` (a
/// [`select_rows`] or [`select_into`] node of the same picks per token, read by nothing else)
/// in place, the rest of its rows left as they are.
#[allow(clippy::too_many_arguments)]
pub fn select_into<'a>(
    ctx: &'a Context,
    rows: Tensor<'a>,
    first: i64,
    top: Tensor<'a>,
    score: Tensor<'a>,
    pool_idxs: Tensor<'a>,
    tail: Tensor<'a>,
    pos: Tensor<'a>,
) -> Tensor<'a> {
    let [n_top, ng, ..] = top.ne();
    let kpool = pool_idxs.ne()[0];
    assert_eq!(
        rows.ne()[0],
        1 + n_top * kpool + kpool - 1,
        "qsa_select_into: row length"
    );
    assert!(
        first >= 0 && first + ng <= rows.ne()[1],
        "qsa_select_into: rows {first}..{} of {}",
        first + ng,
        rows.ne()[1]
    );
    ctx.external_inplace(
        &SELECT_INTO,
        rows,
        &[top, score, pool_idxs, tail, pos],
        &[first as i32],
    )
}

/// Whether [`select_all`] may stand in for the scores, `top_k` and [`select`] (on by default;
/// `legion serve --qsa-always-score` turns it off).
static PICK_ALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Lets steps whose tokens each see at most the indexer's budget of pools skip the scoring
/// ([`select_all`]), for every model from now on.
pub fn set_pick_all(on: bool) {
    PICK_ALL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`set_pick_all`] left [`select_all`] on.
pub fn pick_all() -> bool {
    PICK_ALL.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`select`] for a step in which no token sees more than `n_top` pools (the indexer's early
/// exit, after upstream Strata's #783 series, MIT): `top_k` would pick every pool a token sees
/// whatever their scores, and every visible pool scores above -inf, so the selection is each
/// token's visible pools (`vis`, I32 `[2, tokens]`: `first..end` in the listing) with no
/// indexer query, scores or `top_k` at all. The same rows as `select(top_k(score, n_top),
/// score, ..)` (the tests compare them), in the same shape.
pub fn select_all<'a>(
    ctx: &'a Context,
    vis: Tensor<'a>,
    n_top: i64,
    pool_idxs: Tensor<'a>,
    tail: Tensor<'a>,
    pos: Tensor<'a>,
) -> Tensor<'a> {
    let ng = vis.ne()[1];
    let kpool = pool_idxs.ne()[0];
    let cap = n_top * kpool + kpool - 1;
    ctx.external(
        &SELECT_ALL,
        types::I32,
        [1 + cap, ng, 1, 1],
        &[vis, pool_idxs, tail, pos],
    )
}

/// [`select`] from the picks `top` without their scores (#560 r12: GLM's `--tp-seq-par`
/// ranks exchange each token's picks, a quarter of its selection row's bytes, and select
/// for every token themselves): a pick is live when its pool is in the token's visible range
/// `vis` (I32 `[2, tokens]`, as [`select_all`] takes it). The indexer's scores mask every
/// other pool to -inf and every visible one is finite, so these are `select(top, score, ..)`'s
/// rows (the tests compare them).
pub fn select_vis<'a>(
    ctx: &'a Context,
    top: Tensor<'a>,
    vis: Tensor<'a>,
    pool_idxs: Tensor<'a>,
    tail: Tensor<'a>,
    pos: Tensor<'a>,
) -> Tensor<'a> {
    let [n_top, ng, ..] = top.ne();
    assert_eq!(vis.ne()[1], ng, "qsa select_vis: a range a token");
    let kpool = pool_idxs.ne()[0];
    let cap = n_top * kpool + kpool - 1;
    ctx.external(
        &SELECT_VIS,
        types::I32,
        [1 + cap, ng, 1, 1],
        &[top, vis, pool_idxs, tail, pos],
    )
}

/// Attention of `q` (F32 `[d, n_head, n_tok, n_seqs]`, may be a strided view) over the pools
/// `k`, `v` (`[d × n_head_kv, cells]`, F16, or q8_0 / q4_0 as [`reads_types`] says): token
/// `t` of sequence `s` attends to the positions of selection row `s × n_tok + t` of `sel` (from
/// [`select`]), whose cells are found through the block table `table` (I32 `[table_width,
/// n_seqs]`). Returns F32 `[d, n_head, n_tok, n_seqs]`, contiguous; softmax scale
/// `1 / sqrt(d)`.
///
/// Steps of at most [`DECODE_TOKENS`] tokens (decode and verify) run the CUDA kernel that
/// splits each token's selection into [`DECODE_SPLITS`] parts: the node then holds the parts'
/// partial results behind the output.
pub fn attention<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    table: Tensor<'a>,
    sel: Tensor<'a>,
) -> Tensor<'a> {
    let [d, n_head, n_tok, n_seqs] = q.ne();
    let tokens = n_tok * n_seqs;
    if tokens > DECODE_TOKENS {
        return ctx.external(&ATTN, types::F32, q.ne(), &[q, k, v, table, sel]);
    }
    let n_head_kv = k.ne()[0] / d;
    let out = d * n_head * tokens;
    let parts = decode_part_floats(d, n_head, n_head_kv, tokens);
    let node = ctx.external(
        &ATTN,
        types::F32,
        [out + parts, 1, 1, 1],
        &[q, k, v, table, sel],
    );
    ctx.view_4d(
        node,
        q.ne(),
        [
            d as usize * 4,
            (d * n_head) as usize * 4,
            (d * n_head * n_tok) as usize * 4,
        ],
        0,
    )
}

/// `--dense-fp8-prompt` (#1183): [`crate::model::fp8_act::gate`]`(gate, o, false)` (the
/// attn_output GEMM's packed FP8 activations, [`crate::model::fp8::dynamic_layout`]) from the
/// attention op itself, for `o` a prompt step's [`attention`] output (one op, as
/// `[d * n_head, tokens]`) and `gate` F32 `[d, n_head, tokens]` (columns contiguous, any head
/// and token strides). The codes and scales have that op's bits: on a GB10 the prompt kernel
/// gates and quantizes in its epilogue with `src/cuda/fp8_act_row.cuh`'s expressions, so the
/// output is not written and read back (`attn_gw_kernel`'s CODES); elsewhere the op runs the
/// attention, then the gate's kernel. `None` where `o` is not such an output.
pub(crate) fn gated_codes<'a>(
    ctx: &'a Context,
    o: Tensor<'a>,
    gate: Tensor<'a>,
) -> Option<Tensor<'a>> {
    // SAFETY: the graph's live tensors.
    unsafe {
        let raw = o.raw();
        let node = if (*raw).op == ffi::ggml_op_GGML_OP_RESHAPE {
            (*raw).src[0]
        } else {
            raw
        };
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &ATTN)) {
            return None;
        }
        let q = &*(*node).src[0];
        let [d, n_head, n_tok, n_seqs] = q.ne;
        let (k, tokens) = (d * n_head, n_tok * n_seqs);
        // The prompt form (the decode form holds the parts behind the output).
        if (*node).ne != q.ne || tokens <= DECODE_TOKENS || o.ne() != [k, tokens, 1, 1] {
            return None;
        }
        if gate.ty() != types::F32 || gate.nb()[0] != 4 || gate.ne() != [d, n_head, tokens, 1] {
            return None;
        }
        let mut args: Vec<Tensor<'a>> = (0..5).map(|i| ctx.wrap((*node).src[i])).collect();
        args.push(gate);
        let (_, total) = crate::model::fp8::dynamic_layout(k, tokens);
        let bytes = codes_scratch(total) + 4 * (k * tokens) as usize + codes_sync(tokens);
        let node = ctx.external(&ATTN_CODES, types::I8, [bytes as i64, 1, 1, 1], &args);
        Some(ctx.view_1d(node, total as i64, 0))
    }
}

/// Whether `node` is made by [`gated_codes`] (`node_fuse`'s count of producers that write an
/// FP8 GEMM's codes themselves).
///
/// # Safety
/// `node` must point to a valid tensor.
pub(crate) unsafe fn is_gated_codes(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe { crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &ATTN_CODES)) }
}

/// The query's head norm and RoPE as ggml's nodes compute them (`rms_norm`, `mul` by the
/// weight, `rope_multi` without frequency factors, `ext_factor` 0), for [`fold_q`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct QNormRope {
    /// The query before the norm: F32 `[d, n_head, tokens]`, columns contiguous.
    pub raw: *mut ffi::ggml_tensor,
    /// The norm's weight: F32 `[d]`.
    pub w: *mut ffi::ggml_tensor,
    /// The RoPE's positions: I32 `[4 * tokens]`.
    pub pos: *mut ffi::ggml_tensor,
    pub eps: f32,
    pub n_dims: i32,
    /// ggml's mrope sections, each at most 255.
    pub sections: [i32; 4],
    /// `GGML_ROPE_TYPE_IMROPE`, else `GGML_ROPE_TYPE_MROPE`.
    pub imrope: bool,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub attn_factor: f32,
    /// The attention's first token among `raw`'s (and the positions').
    pub off: i32,
}

/// A [`gated_codes`] or [`is_prompt_attn`] node's parameter 0 after [`fold_q`]: 1 (MROPE) or 2
/// (IMROPE); 0 before.
const P_FOLD: usize = 0;

/// Whether `node` is a prompt step's [`attention`] (more than [`DECODE_TOKENS`] tokens, the
/// output as the query's shape), the node [`fold_q`] takes without codes (#1183: the
/// attention of prompt steps without `--dense-fp8-prompt`).
///
/// # Safety
/// `node` is a valid tensor.
pub(crate) unsafe fn is_prompt_attn(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's; an `attention` node's query is a valid tensor.
    unsafe {
        if !crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &ATTN)) {
            return false;
        }
        let q = &*(*node).src[0];
        (*node).ne == q.ne && q.ne[2] * q.ne[3] > DECODE_TOKENS
    }
}

/// Where [`fold_q`] puts the fold's sources (the query rows, the weight, the positions) in a
/// node it takes: past the node's own (a [`gated_codes`] node's gate is its sixth).
///
/// # Safety
/// `node` is a valid tensor.
unsafe fn fold_srcs(node: *const ffi::ggml_tensor) -> usize {
    // SAFETY: the caller's.
    if unsafe { is_gated_codes(node) } {
        6
    } else {
        5
    }
}

/// Whether [`fold_q`] made `node` (a [`gated_codes`] or [`is_prompt_attn`] node) read its query
/// before the norm and RoPE.
///
/// # Safety
/// `node` is a valid tensor.
pub(crate) unsafe fn is_q_folded(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe {
        (is_gated_codes(node) || is_prompt_attn(node))
            && crate::ggml::external_params(&*node)[P_FOLD] != 0
    }
}

/// `LEGION_DEBUG_QSA_Q_FOLD=0`: prompt steps without `--dense-fp8-prompt` keep ggml's query
/// norm and RoPE nodes (the way before #1183's fold of them into the plain prompt attention),
/// for A/B runs (the same bits either way).
#[cfg(feature = "cuda")]
fn plain_fold_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("LEGION_DEBUG_QSA_Q_FOLD").is_some_and(|v| v == "0"))
}

/// Why [`fold_q`] does not take `node`, a [`gated_codes`] or [`is_prompt_attn`] node (`None`:
/// it does): the fold is `attn_gw_kernel`'s (the launcher's own pick on a compute capability
/// 12.x GPU, with codes or without), at head size 256 over F16 K and V.
///
/// # Safety
/// `node` is an allocated [`gated_codes`] or [`is_prompt_attn`] node.
pub(crate) unsafe fn fold_q_off(node: *const ffi::ggml_tensor) -> Option<&'static str> {
    #[cfg(feature = "cuda")]
    // SAFETY: the caller's.
    unsafe {
        if !is_gated_codes(node) && plain_fold_off() {
            return Some("LEGION_DEBUG_QSA_Q_FOLD=0");
        }
        let (q, k, v) = (&*(*node).src[0], &*(*node).src[1], &*(*node).src[2]);
        if k.type_ != types::F16 || v.type_ != types::F16 {
            return Some("a quantized KV cache (the fold reads F16 K and V)");
        }
        if q.ne[0] != 256 || !supported(q.ne[0], q.ne[1], k.ne[0] / q.ne[0]) {
            return Some("head size not 256");
        }
        if prompt_kernel() != 0 {
            return Some("a forced prompt kernel (LEGION_DEBUG_QSA_PROMPT_KERNEL)");
        }
        if gw_cg() {
            return Some("LEGION_DEBUG_QSA_GW_L1=0");
        }
        let cc = crate::ggml::cuda_device_of(k).and_then(crate::hw::per_device::cc);
        if cc.map(|c| c.0) != Some(12) {
            return Some("not a compute capability 12.x GPU (the fold is attn_gw_kernel's)");
        }
        None
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = node;
        Some("a build without CUDA")
    }
}

/// Makes `node`, a [`gated_codes`] or [`is_prompt_attn`] node, read its query before the head
/// norm and RoPE `f` (`node_fuse`, after allocation, with ggml's nodes of them turned off): its
/// kernel computes them as ggml-cuda does ([`fold_q_off`] says where).
///
/// # Safety
/// `node` is a [`gated_codes`] or [`is_prompt_attn`] node [`fold_q_off`] takes, `f`'s tensors
/// valid and allocated, and `f` the chain whose output `node`'s query reads, `f.off` tokens in.
pub(crate) unsafe fn fold_q(node: *mut ffi::ggml_tensor, f: &QNormRope) {
    assert!(
        f.sections.iter().all(|s| (0..256).contains(s)),
        "qsa_attn: RoPE sections {:?}",
        f.sections
    );
    let packed = f
        .sections
        .iter()
        .enumerate()
        .fold(0u32, |a, (i, &s)| a | ((s as u32) << (8 * i)));
    // SAFETY: the caller's.
    unsafe {
        let (op, base) = if is_gated_codes(node) {
            (&ATTN_CODES, 6)
        } else {
            (&ATTN, 5)
        };
        let mut srcs: Vec<*mut ffi::ggml_tensor> = (0..base).map(|i| (*node).src[i]).collect();
        srcs.extend([f.raw, f.w, f.pos]);
        let params = [
            1 + f.imrope as i32,
            f.n_dims,
            packed as i32,
            f.eps.to_bits() as i32,
            f.freq_base.to_bits() as i32,
            f.freq_scale.to_bits() as i32,
            f.attn_factor.to_bits() as i32,
            f.off,
        ];
        crate::ggml::make_external(node, op, &srcs, &params);
    }
}

/// [`fold_q`]'s parameters of a node, `None` where it did not fold the node's query.
///
/// # Safety
/// `node` is a [`gated_codes`] node or a prompt step's [`attention`] node.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
unsafe fn folded(node: &ffi::ggml_tensor) -> Option<QNormRope> {
    let p = crate::ggml::external_params(node);
    if p[P_FOLD] == 0 {
        return None;
    }
    // SAFETY: the caller's.
    let base = unsafe { fold_srcs(node) };
    let f = |i: usize| f32::from_bits(p[i] as u32);
    Some(QNormRope {
        raw: node.src[base],
        w: node.src[base + 1],
        pos: node.src[base + 2],
        eps: f(3),
        n_dims: p[1],
        sections: std::array::from_fn(|i| (((p[2] as u32) >> (8 * i)) & 0xff) as i32),
        imrope: p[P_FOLD] == 2,
        freq_base: f(4),
        freq_scale: f(5),
        attn_factor: f(6),
        off: p[7],
    })
}

/// Where a [`gated_codes`] node's F32 rows start: after the packed activations (`total`
/// bytes), aligned to 256 bytes. Then `tokens` rows of `d * n_head` values (attention's output
/// or, on the GB10 kernel, the gated values of the CTAs that finish a token first), then the
/// kernel's per-token counters ([`codes_sync`]).
fn codes_scratch(total: usize) -> usize {
    total.next_multiple_of(256)
}

/// Bytes of the GB10 kernel's per-token counters: a ticket, a done count and the amax bits.
fn codes_sync(tokens: i64) -> usize {
    12 * tokens as usize
}

/// Most tokens a step may have for the split decode kernel.
pub const DECODE_TOKENS: i64 = 64;
/// Parts each token's selection is split into in decode and verify steps.
pub const DECODE_SPLITS: i64 = 16;

/// `legion serve --qsa-attn-chain` ([`set_attn_chain`]; opt-in: it changes the sums' rounding):
/// the prompt kernel accumulates the hi and the lo m8n8k4 products in their own chains across
/// the K steps (Strata-AC922 ea91c64) instead of one zero-started product per step, 17% faster
/// at 32K.
///
/// The flag applies to the Volta prompt kernel, on sm_70/75 (sm_80+ run the mma.m16n8k16
/// kernel, which has no chain, and say so once). The chain's accuracy is measured on sm_70 only. Were it
/// ever the default, it would be decided per device (compute capability 7.0, cached per device
/// ordinal), not once per process.
#[cfg(feature = "cuda")]
fn prompt_chain() -> bool {
    #[cfg(test)]
    match CHAIN_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    ATTN_CHAIN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the Volta prompt kernel accumulates in chains ([`prompt_chain`]).
static ATTN_CHAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets [`prompt_chain`] for launches from now on (`legion serve --qsa-attn-chain`).
pub fn set_attn_chain(on: bool) {
    ATTN_CHAIN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Tests: 0 follows [`set_attn_chain`], 1 forces the chain off, 2 on.
#[cfg(all(test, feature = "cuda"))]
static CHAIN_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The F16 prompt kernel on compute capability 8.0+ (#828). All kernels give the same bits;
/// they differ in speed only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PromptKernel {
    /// The launcher's choice: [`PromptKernel::Gw`] on compute capability 12.x (the GB10),
    /// [`PromptKernel::Gb`] on Ampere (8.0 to 8.7, #1212) and Ada (8.9, #1213 round 30),
    /// [`PromptKernel::Mma`] on the others. `LEGION_DEBUG_QSA_PROMPT_KERNEL` overrides it
    /// ([`prompt_kernel_named`]).
    Auto = 0,
    /// `attn_mma_kernel`: one CTA of 4 warps per SM (two stages of K and V, q in shared memory).
    Mma = 1,
    /// `attn_gb_kernel`: the same arithmetic in ~39 KB, two CTAs per SM (q in registers, one K
    /// and one V buffer filled out of phase).
    Gb = 2,
    /// `attn_gw_kernel` (#1183): `attn_gb_kernel` with warp-local copies (each warp refills the
    /// K rows and V dims it reads itself), two CTA barriers per chunk instead of three; the
    /// copies go through L1 (`cp.async.ca`, #1176 round 42) unless `LEGION_DEBUG_QSA_GW_L1=0`.
    Gw = 3,
    /// `attn_gw_kernel`'s variants for `legion qsa-bench` (#1176 round 42): the same values in
    /// the same order. `GwCg`: the copies with `cp.async.cg` (L2 only), the kernel before.
    GwCg = 4,
    /// `GwCg` at one CTA per SM (half the bytes in flight).
    GwOne = 5,
    /// `GwCg` with consecutive CTAs taking tokens far apart, so no neighbours share a wave.
    GwFar = 6,
    /// [`PromptKernel::GwFar`] with the copies through L1.
    GwFarL1 = 7,
    /// K copies only through L1.
    GwL1K = 8,
    /// V copies only through L1.
    GwL1V = 9,
    /// [`PromptKernel::GwOne`] with the copies through L1 (no CTA beside it to share lines).
    GwOneL1 = 10,
}

/// The prompt kernel launches take from now on ([`PromptKernel`]; `legion qsa-bench`). A graph
/// whose query [`fold_q`] folded (built under [`PromptKernel::Auto`]) then fails to compute
/// rather than run another kernel: build it again after forcing one.
#[cfg(feature = "cuda")]
pub fn set_prompt_kernel(k: PromptKernel) {
    PROMPT_KERNEL.store(k as u8, std::sync::atomic::Ordering::Relaxed);
}

/// [`set_prompt_kernel`]'s choice.
#[cfg(feature = "cuda")]
static PROMPT_KERNEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The prompt kernel a `LEGION_DEBUG_QSA_PROMPT_KERNEL` value names (#1213 round 30, A/B runs):
/// `mma`, `gb` or `gw` force that kernel on any sm_80+ GPU; unset or anything else leaves the
/// launcher's choice ([`PromptKernel::Auto`]). The same bits either way.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn prompt_kernel_named(v: Option<&str>) -> PromptKernel {
    match v {
        Some("mma") => PromptKernel::Mma,
        Some("gb") => PromptKernel::Gb,
        Some("gw") => PromptKernel::Gw,
        _ => PromptKernel::Auto,
    }
}

/// The launches' prompt kernel: [`set_prompt_kernel`]'s choice, else
/// `LEGION_DEBUG_QSA_PROMPT_KERNEL`'s ([`prompt_kernel_named`], read once).
#[cfg(feature = "cuda")]
fn prompt_kernel() -> u8 {
    static ENV: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match PROMPT_KERNEL.load(std::sync::atomic::Ordering::Relaxed) {
        0 => *ENV.get_or_init(|| {
            let v = std::env::var("LEGION_DEBUG_QSA_PROMPT_KERNEL").ok();
            prompt_kernel_named(v.as_deref()) as u8
        }),
        k => k,
    }
}

/// `LEGION_DEBUG_QSA_GW_L1=0`: `attn_gw_kernel` copies K and V with `cp.async.cg` (L2 only),
/// the way before #1176 round 42, for A/B runs (the same bits either way).
#[cfg(feature = "cuda")]
fn gw_cg() -> bool {
    static CG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CG.get_or_init(|| std::env::var_os("LEGION_DEBUG_QSA_GW_L1").is_some_and(|v| v == "0"))
}

/// The decode kernel's head groups ([`cuda::Attn::hgroups`]): 0, the launcher's choice, except
/// in tests ([`DECODE_HG_OVERRIDE`]).
#[cfg(feature = "cuda")]
fn decode_hgroups() -> i32 {
    #[cfg(test)]
    {
        DECODE_HG_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    0
}

/// Tests: the decode kernel's head groups (0 the launcher's choice, -1 the reference kernel).
#[cfg(all(test, feature = "cuda"))]
static DECODE_HG_OVERRIDE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Floats of the decode kernel's partial results: per token, kv head and part, each query
/// head's max, sum and weighted V.
fn decode_part_floats(d: i64, n_head: i64, n_head_kv: i64, tokens: i64) -> i64 {
    tokens * n_head_kv * DECODE_SPLITS * (n_head / n_head_kv) * (2 + d)
}

/// Whether the CUDA kernels read K and V of `types` (#263): F16, or q8_0 / q4_0 each (both
/// quantized), dequantized to F16 values as the gather path reads them. The CPU op reads every
/// type.
pub fn reads_types(types: crate::model::core::KvTypes) -> bool {
    use crate::model::core::KvType::{Q4_0, Q8_0};
    types.is_f16() || (matches!(types.k, Q8_0 | Q4_0) && matches!(types.v, Q8_0 | Q4_0))
}

/// Whether the CUDA kernel supports an attention shape.
pub fn supported(head_dim: i64, n_head: i64, n_head_kv: i64) -> bool {
    matches!(head_dim, 128 | 256)
        && n_head_kv > 0
        && n_head % n_head_kv == 0
        && n_head / n_head_kv <= 16
}

/// `i32` element `i` of a row at `base`.
///
/// # Safety
/// In bounds of a live tensor.
unsafe fn at_i32(base: *const u8, i: usize) -> i32 {
    unsafe { *(base as *const i32).add(i) }
}

unsafe fn cpu_pool_mask(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `pool_mask`, host tensors.
    unsafe {
        let vis = &*dst.src[0];
        let (n_pool, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        for t in (ith..ng).step_by(nth) {
            let v = (vis.data as *const u8).add(t * vis.nb[1]);
            let (first, end) = (at_i32(v, 0) as i64, at_i32(v, 1) as i64);
            let row = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut u16,
                n_pool,
            );
            for (j, m) in row.iter_mut().enumerate() {
                *m = if (first..end).contains(&(j as i64)) {
                    KEEP
                } else {
                    DROP
                };
            }
        }
    }
}

/// One token's selection: the live pools (finite score) sorted by first member (ties by pool
/// index), their members, then the live tail positions in ascending order. Written as the
/// count, the positions, then -1 up to `out.len()`. The CUDA kernel follows the same steps.
#[allow(clippy::too_many_arguments)]
fn select_row(
    top: &[i32],
    score: impl Fn(usize) -> f32,
    member: impl Fn(usize, usize) -> i32,
    kpool: usize,
    tail: &[i32],
    pos: i32,
    out: &mut [i32],
) {
    let mut pools: Vec<(i32, i32)> = top
        .iter()
        .filter(|&&p| score(p as usize) > f32::NEG_INFINITY)
        .map(|&p| (member(p as usize, 0), p))
        .collect();
    pools.sort_unstable();
    let mut n = 0;
    for &(_, p) in &pools {
        for k in 0..kpool {
            out[1 + n] = member(p as usize, k);
            n += 1;
        }
    }
    // The tail lists the token first, then backwards.
    for &p in tail.iter().rev() {
        if p <= pos && p >= 0 {
            out[1 + n] = p;
            n += 1;
        }
    }
    out[0] = n as i32;
    out[1 + n..].fill(-1);
}

unsafe fn cpu_select(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `select_rows`.
    unsafe { cpu_select_at(dst, 0, 0, ith, nth) }
}

unsafe fn cpu_select_into(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    let first = crate::ggml::external_params(dst)[0] as usize;
    // SAFETY: a node made by `select_into`: its operands follow the rows it writes over.
    unsafe { cpu_select_at(dst, 1, first, ith, nth) }
}

/// [`select`]'s rows on the host for the operands at `dst.src[s0..s0 + 5]`, token `t` of them
/// into row `first + t` of `dst`.
///
/// # Safety
/// A node made by [`select_rows`] (`s0` 0) or [`select_into`] (1), host tensors.
unsafe fn cpu_select_at(dst: &ffi::ggml_tensor, s0: usize, first: usize, ith: usize, nth: usize) {
    // SAFETY: the caller's.
    unsafe {
        let src = |i: usize| &*dst.src[s0 + i];
        let (top, score, pidx, tail, pos) = (src(0), src(1), src(2), src(3), src(4));
        let (n_top, ng) = (top.ne[0] as usize, top.ne[1] as usize);
        let kpool = pidx.ne[0] as usize;
        let row_len = dst.ne[0] as usize;
        for t in (ith..ng).step_by(nth) {
            let top_row = std::slice::from_raw_parts(
                (top.data as *const u8).add(t * top.nb[1]) as *const i32,
                n_top,
            );
            let score_row = (score.data as *const u8).add(t * score.nb[1]) as *const f32;
            let tail_row = std::slice::from_raw_parts(
                (tail.data as *const u8).add(t * tail.nb[1]) as *const i32,
                kpool - 1,
            );
            let p = at_i32((pos.data as *const u8).add(t * pos.nb[0]), 0);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add((first + t) * dst.nb[1]) as *mut i32,
                row_len,
            );
            select_row(
                top_row,
                |j| *score_row.add(j),
                |j, k| at_i32((pidx.data as *const u8).add(j * pidx.nb[1]), k),
                kpool,
                tail_row,
                p,
                out,
            );
        }
    }
}

unsafe fn cpu_select_vis(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `select_vis`, host tensors.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (top, vis, pidx, tail, pos) = (src(0), src(1), src(2), src(3), src(4));
        let (n_top, ng) = (top.ne[0] as usize, top.ne[1] as usize);
        let kpool = pidx.ne[0] as usize;
        let row_len = dst.ne[0] as usize;
        for t in (ith..ng).step_by(nth) {
            let top_row = std::slice::from_raw_parts(
                (top.data as *const u8).add(t * top.nb[1]) as *const i32,
                n_top,
            );
            let v = (vis.data as *const u8).add(t * vis.nb[1]);
            let (first, end) = (at_i32(v, 0) as usize, at_i32(v, 1) as usize);
            let tail_row = std::slice::from_raw_parts(
                (tail.data as *const u8).add(t * tail.nb[1]) as *const i32,
                kpool - 1,
            );
            let p = at_i32((pos.data as *const u8).add(t * pos.nb[0]), 0);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut i32,
                row_len,
            );
            select_row(
                top_row,
                |j| {
                    if (first..end).contains(&j) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                },
                |j, k| at_i32((pidx.data as *const u8).add(j * pidx.nb[1]), k),
                kpool,
                tail_row,
                p,
                out,
            );
        }
    }
}

unsafe fn cpu_select_all(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `select_all`, host tensors.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (vis, pidx, tail, pos) = (src(0), src(1), src(2), src(3));
        let ng = vis.ne[1] as usize;
        let kpool = pidx.ne[0] as usize;
        let row_len = dst.ne[0] as usize;
        let n_top = row_len / kpool - 1;
        for t in (ith..ng).step_by(nth) {
            let v = (vis.data as *const u8).add(t * vis.nb[1]);
            let (first, end) = (at_i32(v, 0), at_i32(v, 1));
            let picks: Vec<i32> = (first..end.max(first)).collect();
            assert!(
                picks.len() <= n_top,
                "qsa_select_all: {} pools over {n_top}",
                picks.len()
            );
            let tail_row = std::slice::from_raw_parts(
                (tail.data as *const u8).add(t * tail.nb[1]) as *const i32,
                kpool - 1,
            );
            let p = at_i32((pos.data as *const u8).add(t * pos.nb[0]), 0);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut i32,
                row_len,
            );
            select_row(
                &picks,
                |_| 0.0,
                |j, k| at_i32((pidx.data as *const u8).add(j * pidx.nb[1]), k),
                kpool,
                tail_row,
                p,
                out,
            );
        }
    }
}

/// The indexes of each row's `k` largest scores, for [`select`]: I32 `[k, rows]` from F32
/// `[n, rows]` (rows contiguous). Floats compare as ggml's `top_k` orders them (by sign-flipped
/// bits); the `k`-th largest is the threshold, and the result is every index above it, then the
/// lowest indexes at it (Strata-AC922's rule), each run ascending. ggml's CUDA `top_k` picks the
/// same set but for ties at the threshold, which it takes in whatever order its threads come,
/// and costs a dozen launches; this is one.
pub fn top_k<'a>(ctx: &'a Context, score: Tensor<'a>, k: i64) -> Tensor<'a> {
    let rows = score.ne()[1];
    ctx.external(&TOPK, types::I32, [k, rows, 1, 1], &[score])
}

/// Whether the paged path takes [`top_k`] (default) rather than ggml's ([`set_ggml_paths`]).
pub fn own_top_k() -> bool {
    !TOPK_GGML.load(std::sync::atomic::Ordering::Relaxed)
}

/// ggml's order key of a float for `top_k` (`top_k_float_to_ordered`).
fn topk_key(v: f32) -> u32 {
    let bits = v.to_bits();
    let mask = (0u32.wrapping_sub(bits >> 31)) | 0x8000_0000;
    bits ^ mask
}

/// One row of [`top_k`]: `out.len()` indexes of `row`.
pub(crate) fn top_k_row(row: &[f32], out: &mut [i32]) {
    let k = out.len();
    let mut keys: Vec<u32> = row.iter().map(|&v| topk_key(v)).collect();
    keys.sort_unstable_by(|a, b| b.cmp(a));
    let thr = keys[k - 1];
    let above = keys.iter().take_while(|&&x| x > thr).count();
    let mut a = 0;
    let mut e = above;
    for (i, &v) in row.iter().enumerate() {
        let x = topk_key(v);
        if x > thr {
            out[a] = i as i32;
            a += 1;
        } else if x == thr && e < k {
            out[e] = i as i32;
            e += 1;
        }
    }
}

unsafe fn cpu_topk(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `top_k`, host tensors.
    unsafe {
        let score = &*dst.src[0];
        let (n, k, rows) = (score.ne[0] as usize, dst.ne[0] as usize, dst.ne[1] as usize);
        for r in (ith..rows).step_by(nth) {
            let row = std::slice::from_raw_parts(
                (score.data as *const u8).add(r * score.nb[1]) as *const f32,
                n,
            );
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(r * dst.nb[1]) as *mut i32,
                k,
            );
            top_k_row(row, out);
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_topk(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `top_k`, device tensors of the stream's device.
    unsafe {
        let score = &*dst.src[0];
        if score.nb[0] != 4 {
            return Err("qsa_topk: score rows must be contiguous".into());
        }
        let p = cuda::TopK {
            score: score.data as *const f32,
            score_nb1: score.nb[1] as i64,
            dst: dst.data as *mut i32,
            dst_nb1: dst.nb[1] as i64,
            n: score.ne[0] as i32,
            k: dst.ne[0] as i32,
            rows: dst.ne[1] as i32,
        };
        launched("qsa_topk", cuda::legion_qsa_topk_launch(&p, stream))
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

/// The attention op's operands, read from the node.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Operands {
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_seqs: usize,
    q: *const u8,
    q_nb: [usize; 4],
    k: *const u8,
    v: *const u8,
    /// Values per cell (`d × n_head_kv`).
    kv_row: usize,
    k_type: ffi::ggml_type,
    v_type: ffi::ggml_type,
    /// Bytes per cell.
    k_nb1: usize,
    v_nb1: usize,
    table: *const i32,
    table_width: usize,
    sel: *const i32,
    sel_row: usize,
    dst: *mut f32,
}

impl Operands {
    /// # Safety
    /// `dst` must be a node made by [`attention`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        unsafe {
            let src = |i: usize| &*dst.src[i];
            let (q, k, v, table, sel) = (src(0), src(1), src(2), src(3), src(4));
            let d = q.ne[0] as usize;
            Self {
                d,
                n_head: q.ne[1] as usize,
                n_head_kv: k.ne[0] as usize / d,
                n_tok: q.ne[2] as usize,
                n_seqs: q.ne[3] as usize,
                q: q.data as *const u8,
                q_nb: q.nb,
                k: k.data as *const u8,
                v: v.data as *const u8,
                kv_row: k.ne[0] as usize,
                k_type: k.type_,
                v_type: v.type_,
                k_nb1: k.nb[1],
                v_nb1: v.nb[1],
                table: table.data as *const i32,
                table_width: table.ne[0] as usize,
                sel: sel.data as *const i32,
                sel_row: sel.ne[0] as usize,
                dst: dst.data as *mut f32,
            }
        }
    }
}

/// Head `kvh`'s `out.len()` values at `cell` of a pool at `base` (`ty`, `nb1` bytes per cell):
/// F16 as stored, a quantized type dequantized and rounded to F16 (the gather path's
/// `get_rows` to F32, then F16).
///
/// # Safety
/// The cell is in bounds of a live tensor of `ty`.
unsafe fn read_head(
    base: *const u8,
    ty: ffi::ggml_type,
    nb1: usize,
    cell: usize,
    kvh: usize,
    out: &mut [f32],
) {
    let d = out.len();
    // SAFETY: in bounds (the caller).
    unsafe {
        let row = base.add(cell * nb1);
        if ty == types::F16 {
            ffi::ggml_fp16_to_fp32_row(
                (row as *const u16).add(kvh * d),
                out.as_mut_ptr(),
                d as i64,
            );
        } else {
            let bytes = crate::ggml::row_size(ty, d);
            let raw = std::slice::from_raw_parts(row.add(kvh * bytes), bytes);
            out.copy_from_slice(&crate::ggml::dequantize(ty, raw, d));
            round_f16(out);
        }
    }
}

/// The CPU implementation: per (token, head) row, scores against the selected keys in their
/// order, softmax with the max subtracted, weighted sum of V, all in F32 except that the query
/// and the softmax weights are rounded to F16 first, as the gather path's F16 matmuls round
/// them. So it agrees with the gather path to summation order. A quantized cache's values are
/// read as the gather path reads them ([`read_head`]).
unsafe fn cpu_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: called by ggml for a node made by `attention`, with host tensors.
    unsafe {
        // `fold_q` takes CUDA nodes only.
        assert!(
            crate::ggml::external_params(dst)[P_FOLD] == 0,
            "qsa_attn: a folded query on the CPU"
        );
        let o = Operands::of(dst);
        let rows = o.n_seqs * o.n_tok * o.n_head;
        let mut buf = RowBuf::new(o.d);
        for row in (ith..rows).step_by(nth) {
            cpu_row(&o, row, &mut buf);
        }
    }
}

/// [`cpu_row`]'s scratch.
struct RowBuf {
    kf: Vec<f32>,
    q: Vec<f32>,
    scores: Vec<f32>,
    out: Vec<f32>,
}

impl RowBuf {
    fn new(d: usize) -> Self {
        Self {
            kf: vec![0f32; d],
            q: vec![0f32; d],
            scores: Vec::new(),
            out: vec![0f32; d],
        }
    }
}

/// [`cpu_attn`]'s output row `row` (token `row / n_head`, head `row % n_head`), written to
/// `o.dst`.
///
/// # Safety
/// `o` read from a node made by [`attention`] or [`gated_codes`], with host tensors.
unsafe fn cpu_row(o: &Operands, row: usize, buf: &mut RowBuf) {
    // SAFETY: the caller's.
    unsafe {
        let g = o.n_head / o.n_head_kv;
        let scale = 1.0 / (o.d as f32).sqrt();
        let h = row % o.n_head;
        let tok = row / o.n_head;
        let (s, t) = (tok / o.n_tok, tok % o.n_tok);
        let kvh = h / g;
        buf.q.copy_from_slice(std::slice::from_raw_parts(
            o.q.add(s * o.q_nb[3] + t * o.q_nb[2] + h * o.q_nb[1]) as *const f32,
            o.d,
        ));
        round_f16(&mut buf.q);
        let sel = o.sel.add(tok * o.sel_row);
        let n = *sel as usize;
        let ids = std::slice::from_raw_parts(sel.add(1), n);
        let cell = |p: i32| {
            let p = p as usize;
            let b = *o.table.add(s * o.table_width + p / KV_BLOCK) as usize;
            b * KV_BLOCK + p % KV_BLOCK
        };
        buf.scores.clear();
        for &p in ids {
            read_head(o.k, o.k_type, o.k_nb1, cell(p), kvh, &mut buf.kf);
            let dot: f32 = buf.q.iter().zip(&buf.kf).map(|(a, b)| a * b).sum();
            buf.scores.push(dot * scale);
        }
        let max = buf.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in &mut buf.scores {
            *x = (*x - max).exp();
            sum += *x;
        }
        for x in &mut buf.scores {
            *x /= sum;
        }
        round_f16(&mut buf.scores);
        buf.out.fill(0.0);
        for (&p, w) in ids.iter().zip(&buf.scores) {
            read_head(o.v, o.v_type, o.v_nb1, cell(p), kvh, &mut buf.kf);
            for (acc, x) in buf.out.iter_mut().zip(&buf.kf) {
                *acc += w * x;
            }
        }
        std::slice::from_raw_parts_mut(o.dst.add(row * o.d), o.d).copy_from_slice(&buf.out);
    }
}

/// A [`gated_codes`] node's pieces: the attention operands (their output in the node's F32
/// rows), the gate (`src[5]`: its data, head and token strides in floats), the codes and the
/// scales.
struct CodesOperands {
    attn: Operands,
    gate: *const f32,
    gate_hd: usize,
    gate_ld: usize,
    codes: *mut u8,
    scales: *mut f32,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    sync: *mut u32,
}

impl CodesOperands {
    /// # Safety
    /// `dst` must be a node made by [`gated_codes`].
    unsafe fn of(dst: &ffi::ggml_tensor) -> Self {
        // SAFETY: the caller's; the pieces lie in the node as `gated_codes` sized it.
        unsafe {
            let mut attn = Operands::of(dst);
            let (k, tokens) = (attn.d * attn.n_head, attn.n_tok * attn.n_seqs);
            let (codes, total) = crate::model::fp8::dynamic_layout(k as i64, tokens as i64);
            let base = dst.data as *mut u8;
            let scratch = codes_scratch(total);
            attn.dst = base.add(scratch) as *mut f32;
            let gate = &*dst.src[5];
            Self {
                attn,
                gate: gate.data as *const f32,
                gate_hd: gate.nb[1] / 4,
                gate_ld: gate.nb[2] / 4,
                codes: base,
                scales: base.add(codes) as *mut f32,
                sync: base.add(scratch + 4 * k * tokens) as *mut u32,
            }
        }
    }
}

/// The CPU implementation of [`gated_codes`]: per token, [`cpu_attn`]'s rows of every head,
/// then [`crate::model::fp8_act`]'s gate and [`crate::model::fp8::quantize_token`] of them.
unsafe fn cpu_attn_codes(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: called by ggml for a node made by `gated_codes`, with host tensors.
    unsafe {
        // `fold_q` takes CUDA nodes only.
        assert!(
            crate::ggml::external_params(dst)[P_FOLD] == 0,
            "qsa_attn_codes: a folded query on the CPU"
        );
        let c = CodesOperands::of(dst);
        let o = &c.attn;
        let (d, k, tokens) = (o.d, o.d * o.n_head, o.n_tok * o.n_seqs);
        let mut buf = RowBuf::new(d);
        let mut y = vec![0f32; k];
        for tok in (ith..tokens).step_by(nth) {
            for h in 0..o.n_head {
                cpu_row(o, tok * o.n_head + h, &mut buf);
            }
            let rows = std::slice::from_raw_parts(o.dst.add(tok * k), k);
            for (i, v) in y.iter_mut().enumerate() {
                let x = *c.gate.add(tok * c.gate_ld + (i / d) * c.gate_hd + i % d);
                *v = crate::model::fp8_act::gated_host(x, rows[i], false);
            }
            let (q, s) = crate::model::fp8::quantize_token(&y);
            std::ptr::copy_nonoverlapping(q.as_ptr(), c.codes.add(tok * k), k);
            c.scales.add(tok).write_unaligned(s);
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! Launch parameters of `src/cuda/qsa_attn.cu`. Grids, block sizes and shared memory are
    //! chosen here; the launchers only launch.

    use std::os::raw::c_void;

    #[repr(C)]
    pub struct PoolMask {
        pub vis: *const i32,
        pub vis_nb1: i64,
        pub dst: *mut u16,
        pub dst_nb1: i64,
        pub n_pool: i32,
        pub n_tok: i32,
        pub grid: [u32; 2],
        pub block: u32,
    }

    #[repr(C)]
    pub struct Select {
        pub top: *const i32,
        pub top_nb1: i64,
        pub score: *const f32,
        pub score_nb1: i64,
        pub pool_idxs: *const i32,
        pub pool_nb1: i64,
        pub tail: *const i32,
        pub tail_nb1: i64,
        pub pos: *const i32,
        pub pos_nb0: i64,
        pub dst: *mut i32,
        pub dst_nb1: i64,
        pub n_top: i32,
        pub kpool: i32,
        pub row_len: i32,
        pub n_tok: i32,
        /// Sort width: a power of two >= n_top.
        pub sort_n: i32,
        pub block: u32,
        /// [`super::select_all`]: each token's visible pools (`top` and `score` unused); null
        /// for [`super::select`].
        pub vis: *const i32,
        pub vis_nb1: i64,
    }

    #[repr(C)]
    pub struct Attn {
        pub q: *const f32,
        pub q_nb1: i64,
        pub q_nb2: i64,
        pub q_nb3: i64,
        pub k: *const u8,
        pub v: *const u8,
        pub kv_row: i64,
        pub table: *const i32,
        pub table_width: i32,
        pub sel: *const i32,
        pub sel_row: i32,
        pub dst: *mut f32,
        pub head_dim: i32,
        pub n_head: i32,
        pub n_head_kv: i32,
        pub n_tok: i32,
        pub scale_log2: f32,
        pub grid: [u32; 2],
        pub part: *mut f32,
        pub splits: i32,
        pub chain: i32,
        /// 0 F16, 1 q8_0, 2 q4_0 (the paged kernels' codes).
        pub k_type: i32,
        pub v_type: i32,
        /// Bytes per cell.
        pub k_nb1: i64,
        pub v_nb1: i64,
        /// Decode steps: CTAs per kv head's query heads, 0 the launcher's choice (tests: -1
        /// the reference kernel).
        pub hgroups: i32,
        /// The F16 prompt kernel on sm_80+ ([`super::PromptKernel`] as i32): 0 the launcher's
        /// choice, 1 `attn_mma_kernel`, 2 `attn_gb_kernel`, 3 `attn_gw_kernel`, 4-10 its
        /// variants for `legion qsa-bench`.
        pub mma_kernel: i32,
        /// 1: `attn_gw_kernel` copies with `cp.async.cg` (`LEGION_DEBUG_QSA_GW_L1=0`), 0
        /// through L1.
        pub gw_cg: i32,
        /// [`super::gated_codes`] (null: off): the gate (strides in floats), the codes and
        /// scales written instead of the output, and the kernel's zeroed per-token counters.
        pub gate: *const f32,
        pub gate_ld: i64,
        pub gate_hd: i64,
        pub codes: *mut u8,
        pub scales: *mut f32,
        pub sync: *mut u32,
        /// [`super::fold_q`] (null: off): `q` is the query before its head norm (weight `qn_w`)
        /// and RoPE (positions `qn_pos`, section `s` of token `t` at `t + s * qn_pos_n`), which
        /// the GB10 kernel computes as ggml-cuda does.
        pub qn_w: *const f32,
        pub qn_pos: *const i32,
        pub qn_pos_n: i32,
        pub qn_eps: f32,
        pub qn_rot: i32,
        pub qn_sections: [i32; 4],
        pub qn_imrope: i32,
        pub qn_freq_base: f32,
        pub qn_freq_scale: f32,
        pub qn_attn_factor: f32,
        /// Set by the launcher.
        pub qn_theta_scale: f32,
    }

    #[repr(C)]
    pub struct Scores {
        pub q: *const f32,
        pub q_nb1: i64,
        pub q_nb2: i64,
        pub keys: *const f32,
        pub k_nb: i64,
        pub w: *const f32,
        pub w_nb1: i64,
        pub vis: *const i32,
        pub dst: *mut f32,
        pub dst_nb1: i64,
        pub n_pool: i32,
        pub ng: i32,
        /// Optional (null for none): added to the visible scores ([n_pool]).
        pub bias: *const f32,
        /// Optional (null for none): pool j's key is row rows[j] of keys ([n_pool]).
        pub rows: *const i32,
        /// Keys are F16 (else F32).
        pub k_f16: i32,
        /// Indexer heads, a multiple of 4.
        pub n_head: i32,
    }

    #[repr(C)]
    pub struct ScoresExact {
        pub q: *const f32,
        pub q_nb1: i64,
        pub q_nb2: i64,
        pub keys: *const f32,
        pub k_nb: i64,
        pub w: *const f32,
        pub w_nb1: i64,
        pub vis: *const i32,
        pub bias: *const f32,
        pub dst: *mut f32,
        pub dst_nb1: i64,
        pub n_pool: i32,
        pub ng: i32,
    }

    #[repr(C)]
    pub struct TopK {
        pub score: *const f32,
        pub score_nb1: i64,
        pub dst: *mut i32,
        pub dst_nb1: i64,
        pub n: i32,
        pub k: i32,
        pub rows: i32,
    }

    unsafe extern "C" {
        pub fn legion_qsa_topk_launch(p: *const TopK, stream: *mut c_void) -> i32;
        pub fn legion_qsa_scores_exact_launch(p: *const ScoresExact, stream: *mut c_void) -> i32;
        pub fn legion_qsa_scores_launch(p: *const Scores, stream: *mut c_void) -> i32;
        pub fn legion_qsa_scores_sm7x() -> i32;
        pub fn legion_qsa_pool_mask_launch(p: *const PoolMask, stream: *mut c_void) -> i32;
        pub fn legion_qsa_select_launch(p: *const Select, stream: *mut c_void) -> i32;
        pub fn legion_qsa_attn_launch(p: *const Attn, stream: *mut c_void) -> i32;
        pub fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, stream: *mut c_void) -> i32;
    }
}

#[cfg(feature = "cuda")]
fn launched(name: &str, code: i32) -> Result<(), String> {
    match code {
        0 => Ok(()),
        1 => Err(format!("{name}: unsupported shape")),
        e => Err(format!("{name}: CUDA launch failed ({e})")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_pool_mask(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `pool_mask`, device tensors of the stream's device.
    unsafe {
        let vis = &*dst.src[0];
        let (n_pool, ng) = (dst.ne[0], dst.ne[1]);
        const BLOCK: u32 = 256;
        let p = cuda::PoolMask {
            vis: vis.data as *const i32,
            vis_nb1: vis.nb[1] as i64,
            dst: dst.data as *mut u16,
            dst_nb1: dst.nb[1] as i64,
            n_pool: n_pool as i32,
            n_tok: ng as i32,
            grid: [(n_pool as u32).div_ceil(BLOCK), ng as u32],
            block: BLOCK,
        };
        launched(
            "qsa_pool_mask",
            cuda::legion_qsa_pool_mask_launch(&p, stream),
        )
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_select(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `select_rows`.
    unsafe { cuda_select_at(dst, 0, 0, stream) }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_select_into(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let first = crate::ggml::external_params(dst)[0] as usize;
    // SAFETY: a node made by `select_into`: its operands follow the rows it writes over.
    unsafe { cuda_select_at(dst, 1, first, stream) }
}

/// [`cpu_select_at`] on CUDA.
///
/// # Safety
/// A node made by [`select_rows`] (`s0` 0) or [`select_into`] (1), device tensors of the
/// stream's device.
#[cfg(feature = "cuda")]
unsafe fn cuda_select_at(
    dst: &ffi::ggml_tensor,
    s0: usize,
    first: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: the caller's.
    unsafe {
        let src = |i: usize| &*dst.src[s0 + i];
        let (top, score, pidx, tail, pos) = (src(0), src(1), src(2), src(3), src(4));
        let n_top = top.ne[0] as usize;
        if n_top > CUDA_MAX_TOP {
            return Err(format!(
                "qsa_select: {n_top} pools per token (at most {CUDA_MAX_TOP})"
            ));
        }
        let p = cuda::Select {
            top: top.data as *const i32,
            top_nb1: top.nb[1] as i64,
            score: score.data as *const f32,
            score_nb1: score.nb[1] as i64,
            pool_idxs: pidx.data as *const i32,
            pool_nb1: pidx.nb[1] as i64,
            tail: tail.data as *const i32,
            tail_nb1: tail.nb[1] as i64,
            pos: pos.data as *const i32,
            pos_nb0: pos.nb[0] as i64,
            dst: (dst.data as *mut u8).add(first * dst.nb[1]) as *mut i32,
            dst_nb1: dst.nb[1] as i64,
            n_top: n_top as i32,
            kpool: pidx.ne[0] as i32,
            row_len: dst.ne[0] as i32,
            n_tok: top.ne[1] as i32,
            sort_n: n_top.next_power_of_two().max(32) as i32,
            block: 256,
            vis: std::ptr::null(),
            vis_nb1: 0,
        };
        launched("qsa_select", cuda::legion_qsa_select_launch(&p, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_select_all(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `select_all`, device tensors of the stream's device.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (vis, pidx, tail, pos) = (src(0), src(1), src(2), src(3));
        let kpool = pidx.ne[0] as usize;
        let n_top = dst.ne[0] as usize / kpool - 1;
        if n_top > CUDA_MAX_TOP {
            return Err(format!(
                "qsa_select_all: {n_top} pools per token (at most {CUDA_MAX_TOP})"
            ));
        }
        let p = cuda::Select {
            top: std::ptr::null(),
            top_nb1: 0,
            score: std::ptr::null(),
            score_nb1: 0,
            pool_idxs: pidx.data as *const i32,
            pool_nb1: pidx.nb[1] as i64,
            tail: tail.data as *const i32,
            tail_nb1: tail.nb[1] as i64,
            pos: pos.data as *const i32,
            pos_nb0: pos.nb[0] as i64,
            dst: dst.data as *mut i32,
            dst_nb1: dst.nb[1] as i64,
            n_top: n_top as i32,
            kpool: kpool as i32,
            row_len: dst.ne[0] as i32,
            n_tok: vis.ne[1] as i32,
            sort_n: n_top.next_power_of_two().max(32) as i32,
            block: 256,
            vis: vis.data as *const i32,
            vis_nb1: vis.nb[1] as i64,
        };
        launched("qsa_select_all", cuda::legion_qsa_select_launch(&p, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_select_vis(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `select_vis`, device tensors of the stream's device.
    unsafe {
        let src = |i: usize| &*dst.src[i];
        let (top, vis, pidx, tail, pos) = (src(0), src(1), src(2), src(3), src(4));
        let n_top = top.ne[0] as usize;
        if n_top > CUDA_MAX_TOP {
            return Err(format!(
                "qsa_select_vis: {n_top} pools per token (at most {CUDA_MAX_TOP})"
            ));
        }
        // The select kernel with picks and no scores: the pick's liveness from `vis`.
        let p = cuda::Select {
            top: top.data as *const i32,
            top_nb1: top.nb[1] as i64,
            score: std::ptr::null(),
            score_nb1: 0,
            pool_idxs: pidx.data as *const i32,
            pool_nb1: pidx.nb[1] as i64,
            tail: tail.data as *const i32,
            tail_nb1: tail.nb[1] as i64,
            pos: pos.data as *const i32,
            pos_nb0: pos.nb[0] as i64,
            dst: dst.data as *mut i32,
            dst_nb1: dst.nb[1] as i64,
            n_top: n_top as i32,
            kpool: pidx.ne[0] as i32,
            row_len: dst.ne[0] as i32,
            n_tok: top.ne[1] as i32,
            sort_n: n_top.next_power_of_two().max(32) as i32,
            block: 256,
            vis: vis.data as *const i32,
            vis_nb1: vis.nb[1] as i64,
        };
        launched("qsa_select_vis", cuda::legion_qsa_select_launch(&p, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `attention`, device tensors of the stream's device.
    let o = unsafe { Operands::of(dst) };
    let mut p = attn_params(&o)?;
    // SAFETY: an `attention` node; `fold_q`'s tensors are the node's, allocated.
    if let Some(f) = unsafe { folded(dst) } {
        // SAFETY: as above.
        unsafe { fold_params(&mut p, &f, &o, "qsa_attn")? };
    }
    let r = launched("qsa_attn", unsafe {
        cuda::legion_qsa_attn_launch(&p, stream)
    });
    if r.is_ok() {
        // SAFETY: the node's operands.
        unsafe { after_attn(&o, stream) };
    }
    r
}

/// [`gated_codes`] on CUDA: the prompt kernel writes the codes where it can (the GB10's
/// `attn_gw_kernel`); where another kernel ran (it wrote the output to the node's F32 rows),
/// [`crate::model::fp8_act::gate`]'s kernel quantizes those rows.
#[cfg(feature = "cuda")]
unsafe fn cuda_attn_codes(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gated_codes`, device tensors of the stream's device.
    let c = unsafe { CodesOperands::of(dst) };
    let o = &c.attn;
    let (k, tokens) = ((o.d * o.n_head) as i64, (o.n_tok * o.n_seqs) as i64);
    let mut p = attn_params(o)?;
    p.gate = c.gate;
    p.gate_hd = c.gate_hd as i64;
    p.gate_ld = c.gate_ld as i64;
    p.codes = c.codes;
    p.scales = c.scales;
    p.sync = c.sync;
    // SAFETY: a `gated_codes` node; `fold_q`'s tensors are the node's, allocated.
    if let Some(f) = unsafe { folded(dst) } {
        // SAFETY: as above.
        unsafe { fold_params(&mut p, &f, o, "qsa_attn_codes")? };
    }
    // SAFETY: the node's counters, `codes_sync(tokens)` bytes.
    let e = unsafe { cuda::cudaMemsetAsync(c.sync as *mut c_void, 0, codes_sync(tokens), stream) };
    if e != 0 {
        return Err(format!("qsa_attn_codes: cudaMemsetAsync failed ({e})"));
    }
    // SAFETY: the parameters' device pointers.
    let r = match unsafe { cuda::legion_qsa_attn_launch(&p, stream) } {
        3 => {
            // SAFETY: the node's gate, rows, codes and scales, as `gated_codes` sized them.
            let r = unsafe {
                crate::model::fp8_act::gate_raw(
                    (c.gate, c.gate_hd as i64, c.gate_ld as i64, o.d as i32),
                    (o.dst as *const f32, k),
                    c.codes,
                    c.scales,
                    (k, tokens),
                    false,
                    stream,
                )
            };
            launched("qsa_attn_codes (fp8_act gate)", r)
        }
        r => launched("qsa_attn_codes", r),
    };
    if r.is_ok() {
        // SAFETY: the node's operands.
        unsafe { after_attn(o, stream) };
    }
    r
}

/// The launch parameters `p` of a node whose query [`fold_q`] folded (`f`; `o` the node's
/// operands): the kernel reads the query rows before the norm and makes the norm and RoPE.
///
/// # Safety
/// `f` read from a live, allocated node whose operands are `o`.
#[cfg(feature = "cuda")]
unsafe fn fold_params(
    p: &mut cuda::Attn,
    f: &QNormRope,
    o: &Operands,
    name: &str,
) -> Result<(), String> {
    // ggml's norm and RoPE nodes are off, so only the fold's kernel makes this query: a kernel
    // forced after the rewrite is refused, not silently replaced by it.
    if p.mma_kernel != 0 || p.gw_cg != 0 {
        return Err(format!(
            "{name}: a prompt kernel was forced after node_fuse folded the query under the \
             launcher's choice; build the graph again"
        ));
    }
    // SAFETY: `fold_q`'s sources: the query before the norm (`f.off` tokens in, as the
    // attention's query viewed the RoPE's output), the weight and the positions.
    unsafe {
        let (raw, pos) = (&*f.raw, &*f.pos);
        p.q = (raw.data as *const u8).add(f.off as usize * raw.nb[2]) as *const f32;
        p.q_nb1 = raw.nb[1] as i64;
        p.q_nb2 = raw.nb[2] as i64;
        p.q_nb3 = (raw.nb[2] * o.n_tok) as i64;
        p.qn_w = (*f.w).data as *const f32;
        p.qn_pos = (pos.data as *const i32).add(f.off as usize);
        p.qn_pos_n = (pos.ne[0] / 4) as i32;
    }
    p.qn_eps = f.eps;
    p.qn_rot = f.n_dims;
    p.qn_sections = f.sections;
    p.qn_imrope = f.imrope as i32;
    p.qn_freq_base = f.freq_base;
    p.qn_freq_scale = f.freq_scale;
    p.qn_attn_factor = f.attn_factor;
    Ok(())
}

/// What follows a prompt step's attention launch: the selections' dump where it is on.
///
/// # Safety
/// `o` read from a live node on the stream's device.
#[cfg(feature = "cuda")]
unsafe fn after_attn(o: &Operands, stream: *mut c_void) {
    let tokens = o.n_tok * o.n_seqs;
    if tokens as i64 > DECODE_TOKENS && sel_dump::enabled() {
        // SAFETY: the node's selection rows, `tokens` of `sel_row` I32s.
        unsafe { sel_dump::after(o.sel, tokens, o.sel_row, o.n_tok, stream) };
    }
}

/// The launch parameters of the attention op whose operands are `o` (no codes).
#[cfg(feature = "cuda")]
fn attn_params(o: &Operands) -> Result<cuda::Attn, String> {
    if !supported(o.d as i64, o.n_head as i64, o.n_head_kv as i64) {
        return Err(format!(
            "qsa_attn: unsupported shape: head size {}, {} heads over {} kv heads",
            o.d, o.n_head, o.n_head_kv
        ));
    }
    let (Some(k_type), Some(v_type)) = (
        crate::model::core::paged::kernel_type(o.k_type),
        crate::model::core::paged::kernel_type(o.v_type),
    ) else {
        return Err("qsa_attn: the CUDA kernels read F16, q8_0 and q4_0 caches".into());
    };
    if (k_type == 0) != (v_type == 0) {
        return Err("qsa_attn: the CUDA kernels read F16 K and V, or both quantized".into());
    }
    let p = cuda::Attn {
        q: o.q as *const f32,
        q_nb1: o.q_nb[1] as i64,
        q_nb2: o.q_nb[2] as i64,
        q_nb3: o.q_nb[3] as i64,
        k: o.k,
        v: o.v,
        kv_row: o.kv_row as i64,
        table: o.table,
        table_width: o.table_width as i32,
        sel: o.sel,
        sel_row: o.sel_row as i32,
        dst: o.dst,
        head_dim: o.d as i32,
        n_head: o.n_head as i32,
        n_head_kv: o.n_head_kv as i32,
        n_tok: o.n_tok as i32,
        scale_log2: std::f32::consts::LOG2_E / (o.d as f32).sqrt(),
        // One CTA per (token, kv head) (and per part in decode steps).
        grid: [(o.n_tok * o.n_seqs) as u32, o.n_head_kv as u32],
        part: if (o.n_tok * o.n_seqs) as i64 <= DECODE_TOKENS {
            // SAFETY: the parts follow the output in the node (see `attention`).
            unsafe { o.dst.add(o.d * o.n_head * o.n_tok * o.n_seqs) }
        } else {
            std::ptr::null_mut()
        },
        splits: DECODE_SPLITS as i32,
        chain: prompt_chain() as i32,
        k_type,
        v_type,
        k_nb1: o.k_nb1 as i64,
        v_nb1: o.v_nb1 as i64,
        hgroups: decode_hgroups(),
        mma_kernel: prompt_kernel() as i32,
        gw_cg: gw_cg() as i32,
        gate: std::ptr::null(),
        gate_ld: 0,
        gate_hd: 0,
        codes: std::ptr::null_mut(),
        scales: std::ptr::null_mut(),
        sync: std::ptr::null_mut(),
        qn_w: std::ptr::null(),
        qn_pos: std::ptr::null(),
        qn_pos_n: 0,
        qn_eps: 0.0,
        qn_rot: 0,
        qn_sections: [0; 4],
        qn_imrope: 0,
        qn_freq_base: 0.0,
        qn_freq_scale: 0.0,
        qn_attn_factor: 0.0,
        qn_theta_scale: 0.0,
    };
    Ok(p)
}

#[cfg(any(test, feature = "cuda"))]
pub mod bench;
#[cfg(test)]
mod codes_tests;
#[cfg(all(test, feature = "cuda"))]
mod decode_tests;
#[cfg(test)]
mod fold_tests;
#[cfg(test)]
mod quant_tests;
pub mod sel_dump;
#[cfg(test)]
mod tests;
