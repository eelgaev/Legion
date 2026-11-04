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

static SELECT_ALL: ExternalOp = ExternalOp {
    name: "qsa_select_all",
    cpu: cpu_select_all,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_select_all),
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
    let [n_top, ng, ..] = top.ne();
    let kpool = pool_idxs.ne()[0];
    let cap = n_top * kpool + kpool - 1;
    ctx.external(
        &SELECT,
        types::I32,
        [1 + cap, ng, 1, 1],
        &[top, score, pool_idxs, tail, pos],
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

/// The F16 prompt kernel on compute capability 8.0+ (#828). Both kernels give the same bits;
/// they differ in speed only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PromptKernel {
    /// The launcher's choice: [`PromptKernel::Gb`] on compute capability 12.x (the GB10),
    /// [`PromptKernel::Mma`] on the others.
    Auto = 0,
    /// `attn_mma_kernel`: one CTA of 4 warps per SM (two stages of K and V, q in shared memory).
    Mma = 1,
    /// `attn_gb_kernel`: the same arithmetic in ~39 KB, two CTAs per SM (q in registers, one K
    /// and one V buffer filled out of phase).
    Gb = 2,
}

/// The prompt kernel launches take from now on ([`PromptKernel`]; `legion qsa-bench`).
#[cfg(feature = "cuda")]
pub fn set_prompt_kernel(k: PromptKernel) {
    PROMPT_KERNEL.store(k as u8, std::sync::atomic::Ordering::Relaxed);
}

/// [`set_prompt_kernel`]'s choice.
#[cfg(feature = "cuda")]
static PROMPT_KERNEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

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
    // SAFETY: a node made by `select`, host tensors.
    unsafe {
        let src = |i: usize| &*dst.src[i];
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
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut i32,
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
        let o = Operands::of(dst);
        let g = o.n_head / o.n_head_kv;
        let scale = 1.0 / (o.d as f32).sqrt();
        let rows = o.n_seqs * o.n_tok * o.n_head;
        let mut kf = vec![0f32; o.d];
        let mut q = vec![0f32; o.d];
        let mut scores = Vec::new();
        let mut out = vec![0f32; o.d];
        for row in (ith..rows).step_by(nth) {
            let h = row % o.n_head;
            let tok = row / o.n_head;
            let (s, t) = (tok / o.n_tok, tok % o.n_tok);
            let kvh = h / g;
            q.copy_from_slice(std::slice::from_raw_parts(
                o.q.add(s * o.q_nb[3] + t * o.q_nb[2] + h * o.q_nb[1]) as *const f32,
                o.d,
            ));
            round_f16(&mut q);
            let sel = o.sel.add(tok * o.sel_row);
            let n = *sel as usize;
            let ids = std::slice::from_raw_parts(sel.add(1), n);
            let cell = |p: i32| {
                let p = p as usize;
                let b = *o.table.add(s * o.table_width + p / KV_BLOCK) as usize;
                b * KV_BLOCK + p % KV_BLOCK
            };
            scores.clear();
            for &p in ids {
                read_head(o.k, o.k_type, o.k_nb1, cell(p), kvh, &mut kf);
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
            for (&p, w) in ids.iter().zip(&scores) {
                read_head(o.v, o.v_type, o.v_nb1, cell(p), kvh, &mut kf);
                for (acc, x) in out.iter_mut().zip(&kf) {
                    *acc += w * x;
                }
            }
            std::slice::from_raw_parts_mut(o.dst.add(row * o.d), o.d).copy_from_slice(&out);
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
        /// choice, 1 `attn_mma_kernel`, 2 `attn_gb_kernel`.
        pub mma_kernel: i32,
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
    // SAFETY: a node made by `select`, device tensors of the stream's device.
    unsafe {
        let src = |i: usize| &*dst.src[i];
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
            dst: dst.data as *mut i32,
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
unsafe fn cuda_attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `attention`, device tensors of the stream's device.
    let o = unsafe { Operands::of(dst) };
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
        mma_kernel: PROMPT_KERNEL.load(std::sync::atomic::Ordering::Relaxed) as i32,
    };
    launched("qsa_attn", unsafe {
        cuda::legion_qsa_attn_launch(&p, stream)
    })
}

#[cfg(any(test, feature = "cuda"))]
pub mod bench;
#[cfg(all(test, feature = "cuda"))]
mod decode_tests;
#[cfg(test)]
mod quant_tests;
#[cfg(test)]
mod tests;
