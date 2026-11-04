//! DeepSeek-V4.1-Flash's own ops ([`crate::ggml::ExternalOp`]): the steps of the `deepseek41`
//! graph that GLM-5.3-Flash's ops ([`crate::model::glm5next::ops`]: linear, RMS norm, the
//! hyper-connection mixer, SwiGLU, ...) do not cover. ggml only holds the tensors, views and
//! reshapes; nothing here is a ggml compute op.
//!
//! The CPU implementations are plain references (F32 math, sums in order, norms' sums in f64),
//! for tests and the CPU backend; the CUDA kernels are `src/cuda/deepseek41.cu` ([`cuda`]),
//! launched from Rust. Formulas follow DeepSeek's reference `inference/model.py` and
//! `engram.py` (MIT) for DeepSeek-V4.1-Flash, without the simulated FP8 / FP4 activation
//! quantization of its training kernels (`act_quant`, `fp4_act_quant`): values stay F32, and the
//! cached keys F16.
//!
//! The KV cache of a layer is one F16 row per position (cell), `[window latent (d) | compressed
//! latent (d) | index key (di)]` in K; a ratio-2 compressor's raw rows `[kv (d) | gate (d)]` in
//! V. The compressed latent and index key of group `j` (positions `j·r .. j·r + r - 1`) sit in
//! the cell of the group's last position.

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

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

external_op!(ROPE, "ds4_rope", cpu_rope, rope);
external_op!(COLLAPSE, "ds4_collapse", cpu_collapse, collapse);
external_op!(ENGRAM, "ds4_engram", cpu_engram, engram);
external_op!(CACHE_WRITE, "ds4_cache_write", cpu_cache_write, cache_write);
external_op!(POOL, "ds4_pool", cpu_pool, pool);
external_op!(
    INDEX_SCORES,
    "ds4_index_scores",
    cpu_index_scores,
    index_scores
);
external_op!(CAND_MASK, "ds4_cand_mask", cpu_cand_mask, cand_mask);
external_op!(TOPK, "ds4_topk", cpu_topk, topk);
external_op!(ATTN, "ds4_attn", cpu_attn, attn);
external_op!(ROUTER_IDS, "ds4_router_ids", cpu_router_ids, router_ids);
external_op!(ROUTER_W, "ds4_router_w", cpu_router_w, router_w);

/// The RoPE of a layer: the last `n_rot` dimensions of each head rotate in adjacent pairs, pair
/// `i` at `1 / base^(2i / n_rot)` per position, with YaRN's ramp between `low` and `high`
/// (dimensions past `high` divided by `factor`) when `yarn`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
    pub n_rot: i64,
    pub base: f32,
    pub yarn: bool,
    pub factor: f32,
    pub low: i32,
    pub high: i32,
}

impl Rope {
    pub fn plain(n_rot: i64, base: f32) -> Self {
        Self {
            n_rot,
            base,
            yarn: false,
            factor: 1.0,
            low: 0,
            high: 0,
        }
    }

    /// `precompute_freqs_cis`'s YaRN: the ramp's ends are the dimensions whose wavelength turns
    /// `beta_fast` and `beta_slow` times over the `orig` training context.
    pub fn yarn(
        n_rot: i64,
        base: f32,
        factor: f32,
        orig: f64,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Self {
        let dim = n_rot as f64;
        let cd = |r: f64| {
            dim * (orig / (r * 2.0 * std::f64::consts::PI)).ln() / (2.0 * (base as f64).ln())
        };
        let low = cd(beta_fast).floor().max(0.0) as i32;
        let high = cd(beta_slow).ceil().min(dim - 1.0) as i32;
        Self {
            n_rot,
            base,
            yarn: true,
            factor,
            low,
            high,
        }
    }

    /// Pair `i`'s angle per position.
    pub fn freq(&self, i: usize) -> f32 {
        freq(
            self.n_rot as usize,
            self.base,
            self.yarn,
            self.factor,
            self.low,
            self.high,
            i,
        )
    }

    fn params(&self, shift: i32, inverse: bool) -> [i32; 8] {
        [
            self.n_rot as i32,
            self.base.to_bits() as i32,
            self.yarn as i32,
            self.factor.to_bits() as i32,
            self.low,
            self.high,
            inverse as i32,
            shift,
        ]
    }
}

fn freq(n_rot: usize, base: f32, yarn: bool, factor: f32, low: i32, high: i32, i: usize) -> f32 {
    let f = 1.0 / base.powf((2 * i) as f32 / n_rot as f32);
    if !yarn {
        return f;
    }
    let ramp = ((i as f32 - low as f32) / ((high - low) as f32).max(1e-3)).clamp(0.0, 1.0);
    let smooth = 1.0 - ramp;
    f / factor * (1.0 - smooth) + f * smooth
}

/// `x` (F32 `[d, h, n]`, contiguous) with the last `n_rot` dimensions of every head rotated by
/// the angles of position `pos[t] + shift` (`pos` I32 `[n]`); backwards with `inverse`.
pub fn rope<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    pos: Tensor<'a>,
    r: &Rope,
    shift: i32,
    inverse: bool,
) -> Tensor<'a> {
    let x = contiguous(x);
    let [d, _, n, _] = x.ne();
    assert!(r.n_rot % 2 == 0 && r.n_rot <= d, "ds4_rope: n_rot");
    assert_eq!(pos.ne()[0], n, "ds4_rope: one position per token");
    ctx.external_with(
        &ROPE,
        types::F32,
        x.ne(),
        &[x, pos],
        &r.params(shift, inverse),
    )
}

/// The streams of `res` (F32 `[e, hc, n]`) collapsed into one per token: weighted by `pre`
/// (F32 `[hc, n]`, rows contiguous), or stream 0 alone without it (the first layer's identity
/// mix); at the tokens `rows` (I32 `[n_out]`) only, if given. F32 `[e, n_out]`.
pub fn collapse<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    pre: Option<Tensor<'a>>,
    rows: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let res = contiguous(res);
    let [e, _, n, _] = res.ne();
    let n_out = rows.map_or(n, |r| r.ne()[0]);
    let mut args = vec![res];
    args.extend(pre);
    args.extend(rows);
    ctx.external_with(
        &COLLAPSE,
        types::F32,
        [e, n_out, 1, 1],
        &args,
        &[pre.is_some() as i32, rows.is_some() as i32],
    )
}

/// Engram's update of the streams `res` (F32 `[e, hc, n]`) from the projected n-gram rows `kv`
/// (F32 `[(hc + 1) · e, n]`: a key per stream, then the value): stream `h` gains `gate · value`,
/// `gate = sigmoid(signed_sqrt(Σ res·q[h]·k[h]·key · rstd / sqrt(e)))` with `rstd` the RMS
/// scales of the stream and the key. `q`, `k`: `[e, hc]` (F32 or BF16).
pub fn engram<'a>(
    ctx: &'a Context,
    res: Tensor<'a>,
    kv: Tensor<'a>,
    q: Tensor<'a>,
    k: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    let res = contiguous(res);
    let [e, hc, n, _] = res.ne();
    assert_eq!(kv.ne()[..2], [(hc + 1) * e, n], "ds4_engram: kv rows");
    assert_eq!(kv.nb()[0], 4);
    assert_eq!(q.ne()[..2], [e, hc], "ds4_engram: q weight");
    assert_eq!(k.ne()[..2], [e, hc], "ds4_engram: k weight");
    ctx.external_with(
        &ENGRAM,
        types::F32,
        res.ne(),
        &[res, kv, q, k],
        &[eps.to_bits() as i32],
    )
}

/// Writes row `t` of `x` (F32 `[w, n]`, rows contiguous) as F16 into row `cells[t]` of `cache`
/// (F16 `[row, cells]`) from element `off` on, for the tokens whose position `pos[t]` ends a
/// group of `ratio` (all for `ratio` ≤ 1). Returns a token node (I32 `[1]`) that readers take to
/// run after the write. `after`: nodes that must run first.
#[allow(clippy::too_many_arguments)]
pub fn cache_write<'a>(
    ctx: &'a Context,
    cache: Tensor<'a>,
    x: Tensor<'a>,
    cells: Tensor<'a>,
    pos: Tensor<'a>,
    off: i64,
    ratio: usize,
    after: &[Tensor<'a>],
) -> Tensor<'a> {
    assert_eq!(cache.ty(), types::F16, "ds4_cache_write: F16 cache");
    assert!(
        off + x.ne()[0] <= cache.ne()[0],
        "ds4_cache_write: row width"
    );
    assert_eq!(x.nb()[0], 4);
    assert_eq!(cells.ne()[0], x.ne()[1]);
    assert_eq!(pos.ne()[0], x.ne()[1]);
    let mut args = vec![cache, x, cells, pos];
    args.extend_from_slice(after);
    ctx.external_with(
        &CACHE_WRITE,
        types::I32,
        [1, 1, 1, 1],
        &args,
        &[off as i32, ratio as i32],
    )
}

/// A ratio-`ratio` compressor's pooling for a group of `n_seqs · n_tok` tokens (positions `pos`
/// I32 `[ng]`): for a token that ends a group, per channel the softmax over the group's
/// positions of their gate rows weighs their raw rows, read from `vcache` (F16 `[≥ 2d, cells]`:
/// raw | gate) through the block `table` (I32 `[width, n_seqs]`); zeros for the other tokens.
/// F32 `[d, ng]`. `after`: the raw rows' write.
#[allow(clippy::too_many_arguments)]
pub fn pool<'a>(
    ctx: &'a Context,
    vcache: Tensor<'a>,
    table: Tensor<'a>,
    pos: Tensor<'a>,
    n_tok: i64,
    ratio: usize,
    d: i64,
    after: Tensor<'a>,
) -> Tensor<'a> {
    assert_eq!(vcache.ty(), types::F16, "ds4_pool: F16 cache");
    assert!(vcache.ne()[0] >= 2 * d && ratio >= 2);
    let ng = pos.ne()[0];
    assert_eq!(ng % n_tok, 0);
    ctx.external_with(
        &POOL,
        types::F32,
        [d, ng, 1, 1],
        &[vcache, table, pos, after],
        &[
            n_tok as i32,
            ratio as i32,
            crate::model::core::KV_BLOCK as i32,
        ],
    )
}

/// The lightning indexer's scores of a group: for query `t` (`iq` F32 `[di, ih, ng]`, head
/// weights `w` F32 `[ih, ng]`) and compressed position `j < (pos[t] + 1) / ratio`,
/// `Σ_h relu(iq[h] · key_j) · w[h]` (plus `mask[j, t]` if given), `-inf` past them. `key_j`: the
/// F16 values from element `koff` of the cell of position `j · ratio + ratio - 1` in `kcache`
/// (through `table`). F32 `[nc, ng]`. `after`: the keys' writes.
#[allow(clippy::too_many_arguments)]
pub fn index_scores<'a>(
    ctx: &'a Context,
    iq: Tensor<'a>,
    w: Tensor<'a>,
    kcache: Tensor<'a>,
    table: Tensor<'a>,
    pos: Tensor<'a>,
    mask: Option<Tensor<'a>>,
    n_tok: i64,
    ratio: usize,
    koff: i64,
    nc: i64,
    after: Tensor<'a>,
) -> Tensor<'a> {
    let [di, ih, ng, _] = iq.ne();
    assert!(
        iq.nb()[0] == 4 && iq.nb()[1] == di as usize * 4,
        "ds4_index_scores: iq rows"
    );
    assert_eq!(w.ne()[..2], [ih, ng]);
    assert_eq!(w.nb()[0], 4);
    assert_eq!(kcache.ty(), types::F16);
    assert!(koff + di <= kcache.ne()[0]);
    assert_eq!(pos.ne()[0], ng);
    let mask = mask.inspect(|m| assert_eq!(m.ne()[..2], [nc, ng]));
    // Fixed source slots for the kernels: the mask's is `pos` when there is none.
    ctx.external_with(
        &INDEX_SCORES,
        types::F32,
        [nc, ng, 1, 1],
        &[iq, w, kcache, table, pos, mask.unwrap_or(pos), after],
        &[
            n_tok as i32,
            ratio as i32,
            koff as i32,
            crate::model::core::KV_BLOCK as i32,
            mask.is_some() as i32,
        ],
    )
}

/// The candidate blocks of the indexer's first level (`select_candidate_blocks`): per query,
/// blocks of `bs` compressed positions scored by their best score in `score` (F32 `[nc, ng]`,
/// `-inf` where not visible), the block of the query's newest visible position pinned in, the
/// best `tb` reachable ones kept (ties to the lower block). F32 `[nc, ng]`: 0 in kept blocks,
/// `-inf` elsewhere (a mask the later indexers add).
pub fn cand_mask<'a>(
    ctx: &'a Context,
    score: Tensor<'a>,
    pos: Tensor<'a>,
    ratio: usize,
    bs: usize,
    tb: usize,
) -> Tensor<'a> {
    let score = contiguous(score);
    assert_eq!(pos.ne()[0], score.ne()[1]);
    ctx.external_with(
        &CAND_MASK,
        types::F32,
        score.ne(),
        &[score, pos],
        &[ratio as i32, bs as i32, tb as i32],
    )
}

/// The indexer's selection: per query the `k` best finite scores of `score` (F32 `[nc, ng]`;
/// ties to the lower position), as compressed positions in ascending order, `-1` after them.
/// I32 `[k, ng]`.
pub fn topk<'a>(ctx: &'a Context, score: Tensor<'a>, k: usize) -> Tensor<'a> {
    let score = contiguous(score);
    ctx.external(&TOPK, types::I32, [k as i64, score.ne()[1], 1, 1], &[score])
}

/// Where a compressed attention reads its other keys: the source layer's cache, the selection
/// (I32 `[k, ng]` of compressed positions, `-1` none), the ratio and the latents' offset in the
/// rows.
#[derive(Clone, Copy)]
pub struct Compressed<'a> {
    pub kcache: Tensor<'a>,
    pub sel: Tensor<'a>,
    pub ratio: usize,
    pub off: i64,
}

/// DeepSeek-V4's sparse attention of a group (`sparse_attn`): query `t` (`q` F32
/// `[d, nh, ng]`) attends to the latents (keys and values both) of the window of positions
/// `max(0, p - window + 1) ..= p` (`p = pos[t]`), from element 0 of the rows of `kown` (F16), and
/// to the compressed latents `comp` selects; softmax scale `scale` and a per-head sink logit
/// (`sinks` F32 `[nh]`) in the denominator. Cells through the block `table`; the window's
/// through `ring` instead when `kown` is a ring ([`crate::model::core::KvRing`]): its table
/// (I32 `[entries, n_seqs]`) and entries, position `p` in entry `(p / KV_BLOCK) % entries`.
/// F32 `[d, nh, ng]`. `after`: the writes the reads need.
#[allow(clippy::too_many_arguments)]
pub fn attn<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    kown: Tensor<'a>,
    table: Tensor<'a>,
    ring: Option<(Tensor<'a>, usize)>,
    pos: Tensor<'a>,
    sinks: Tensor<'a>,
    comp: Option<Compressed<'a>>,
    n_tok: i64,
    scale: f32,
    window: usize,
    after: &[Tensor<'a>],
) -> Tensor<'a> {
    let [d, nh, ng, _] = q.ne();
    assert!(
        q.nb()[0] == 4 && q.nb()[1] == d as usize * 4,
        "ds4_attn: q rows"
    );
    assert_eq!(kown.ty(), types::F16);
    assert!(kown.ne()[0] >= d);
    assert_eq!(sinks.ne()[0], nh);
    assert_eq!(pos.ne()[0], ng);
    assert_eq!(ng % n_tok, 0);
    let (kc, sel, ratio, off) = match comp {
        Some(c) => {
            assert_eq!(c.kcache.ty(), types::F16);
            assert!(c.off + d <= c.kcache.ne()[0]);
            assert_eq!(c.sel.ne()[1], ng);
            (c.kcache, c.sel, c.ratio, c.off)
        }
        None => (kown, pos, 0, 0),
    };
    // Fixed source slots for the kernels: the window's table is `table` without a ring.
    let (wtable, entries) = ring.unwrap_or((table, 0));
    if entries > 0 {
        assert_eq!(wtable.ne()[0], entries as i64, "ds4_attn: ring table");
        // As `KvRing::for_window`: the window before the step and the step's own positions.
        let kb = crate::model::core::KV_BLOCK;
        assert!(
            entries > (window + n_tok as usize).div_ceil(kb),
            "ds4_attn: a ring shorter than the window and the step"
        );
    }
    let mut args = vec![q, kown, table, pos, sinks, kc, sel, wtable];
    args.extend_from_slice(after);
    ctx.external_with(
        &ATTN,
        types::F32,
        [d, nh, ng, 1],
        &args,
        &[
            n_tok as i32,
            scale.to_bits() as i32,
            window as i32,
            ratio as i32,
            off as i32,
            crate::model::core::KV_BLOCK as i32,
            comp.is_some() as i32,
            entries as i32,
        ],
    )
}

/// The router's choices: the `k` experts of the highest `sqrt(softplus(logits)) + bias` per
/// token (ties to the lower id), I32 `[k, n]`, from `logits` F32 `[n_expert, n]`.
pub fn router_ids<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    bias: Tensor<'a>,
    k: i64,
) -> Tensor<'a> {
    let logits = contiguous(logits);
    ctx.external(
        &ROUTER_IDS,
        types::I32,
        [k, logits.ne()[1], 1, 1],
        &[logits, bias],
    )
}

/// The router weights of the choices `ids`: `sqrt(softplus(logits))` at them, divided by their
/// sum plus 1e-20 if `norm` (and more than one is chosen), times `scale`. F32 `[k, n]`.
pub fn router_w<'a>(
    ctx: &'a Context,
    logits: Tensor<'a>,
    ids: Tensor<'a>,
    norm: bool,
    scale: f32,
) -> Tensor<'a> {
    ctx.external_with(
        &ROUTER_W,
        types::F32,
        ids.ne(),
        &[contiguous(logits), ids],
        &[norm as i32, scale.to_bits() as i32],
    )
}

/// `t` itself when contiguous; ops here never take a ggml copy, so a strided input is a bug.
fn contiguous(t: Tensor<'_>) -> Tensor<'_> {
    assert!(t.is_contiguous(), "deepseek41 op input must be contiguous");
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

/// Row `r` of `w` as F32 (`ne0` values), whatever its type.
unsafe fn weight_row(w: &ffi::ggml_tensor, r: usize) -> Vec<f32> {
    let k = w.ne[0] as usize;
    unsafe {
        let p = (w.data as *const u8).add(r * w.nb[1]);
        if w.type_ == types::F32 {
            return std::slice::from_raw_parts(p as *const f32, k).to_vec();
        }
        let bytes = std::slice::from_raw_parts(p, crate::ggml::row_size(w.type_, k));
        crate::ggml::dequantize(w.type_, bytes, k)
    }
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

fn f32_to_f16(x: f32) -> u16 {
    // SAFETY: plain conversion.
    unsafe { ffi::ggml_fp32_to_fp16(x) }
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
    let width = table.ne[0] as usize;
    unsafe { at::<i32>(table, s * width + p / block) as usize * block + p % block }
}

/// The number of compressed positions visible at position `p`.
fn visible(p: usize, ratio: usize) -> usize {
    (p + 1) / ratio
}

unsafe fn cpu_rope(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (x, pos) = (&*dst.src[0], &*dst.src[1]);
        let p = external_params(dst);
        let (n_rot, base, yarn, factor) =
            (p[0] as usize, f32_param(p[1]), p[2] != 0, f32_param(p[3]));
        let (low, high, inverse, shift) = (p[4], p[5], p[6] != 0, p[7]);
        let (d, h, n) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let src = f32s(x, 0, d * h * n);
        let out = std::slice::from_raw_parts_mut(dst.data as *mut f32, d * h * n);
        let freqs: Vec<f32> = (0..n_rot / 2)
            .map(|i| freq(n_rot, base, yarn, factor, low, high, i))
            .collect();
        for t in part(n, ith, nth) {
            let ps = (at::<i32>(pos, t) + shift) as f32;
            let rot: Vec<(f32, f32)> = freqs.iter().map(|f| (ps * f).sin_cos()).collect();
            for hd in 0..h {
                let r = (t * h + hd) * d;
                out[r..r + d].copy_from_slice(&src[r..r + d]);
                let tail = &mut out[r + d - n_rot..r + d];
                for (i, &(s, c)) in rot.iter().enumerate() {
                    let s = if inverse { -s } else { s };
                    let (a, b) = (tail[2 * i], tail[2 * i + 1]);
                    tail[2 * i] = a * c - b * s;
                    tail[2 * i + 1] = a * s + b * c;
                }
            }
        }
    }
}

unsafe fn cpu_collapse(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let res = &*dst.src[0];
        let p = external_params(dst);
        let (has_pre, has_rows) = (p[0] != 0, p[1] != 0);
        let pre = has_pre.then(|| &*dst.src[1]);
        let rows = has_rows.then(|| &*dst.src[1 + has_pre as usize]);
        let (e, hc) = (res.ne[0] as usize, res.ne[1] as usize);
        let n_out = dst.ne[1] as usize;
        let out = dst.data as *mut f32;
        for j in part(n_out, ith, nth) {
            let t = rows.map_or(j, |r| at::<i32>(r, j) as usize);
            let o = std::slice::from_raw_parts_mut(out.add(j * e), e);
            match pre {
                None => o.copy_from_slice(f32s(res, t * res.nb[2], e)),
                Some(pre) => {
                    let w = f32s(pre, t * pre.nb[1], hc);
                    o.fill(0.0);
                    for (s, ws) in w.iter().enumerate() {
                        let r = f32s(res, t * res.nb[2] + s * res.nb[1], e);
                        for i in 0..e {
                            o[i] += ws * r[i];
                        }
                    }
                }
            }
        }
    }
}

unsafe fn cpu_engram(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (res, kv, q, k) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let eps = f32_param(external_params(dst)[0]);
        let (e, hc, n) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        // q · k per stream, as the reference only ever uses their product.
        let qk: Vec<Vec<f32>> = (0..hc)
            .map(|h| {
                let (qr, kr) = (weight_row(q, h), weight_row(k, h));
                qr.iter().zip(&kr).map(|(a, b)| a * b).collect()
            })
            .collect();
        let inv = 1.0 / (e as f32).sqrt();
        let out = dst.data as *mut f32;
        for t in part(n, ith, nth) {
            let kvr = f32s(kv, t * kv.nb[1], (hc + 1) * e);
            let value = &kvr[hc * e..];
            for h in 0..hc {
                let x = f32s(res, t * res.nb[2] + h * res.nb[1], e);
                let key = &kvr[h * e..(h + 1) * e];
                let rstd = rms_scale(x, eps) * rms_scale(key, eps);
                let s: f32 = (0..e).map(|i| x[i] * qk[h][i] * key[i]).sum();
                let dot = s * rstd * inv;
                let g = dot.abs().max(1e-6).sqrt().copysign(dot);
                let gate = 1.0 / (1.0 + (-g).exp());
                let o = std::slice::from_raw_parts_mut(out.add((t * hc + h) * e), e);
                for i in 0..e {
                    o[i] = x[i] + gate * value[i];
                }
            }
        }
    }
}

unsafe fn cpu_cache_write(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (cache, x, cells, pos) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let p = external_params(dst);
        let (off, ratio) = (p[0] as usize, p[1].max(1) as usize);
        let w = x.ne[0] as usize;
        for t in part(x.ne[1] as usize, ith, nth) {
            if !(at::<i32>(pos, t) as usize + 1).is_multiple_of(ratio) {
                continue;
            }
            let row = f32s(x, t * x.nb[1], w);
            let c = at::<i32>(cells, t) as usize;
            let dp = (cache.data as *mut u8).add(c * cache.nb[1]) as *mut u16;
            for (i, v) in row.iter().enumerate() {
                *dp.add(off + i) = f32_to_f16(*v);
            }
        }
    }
}

unsafe fn cpu_pool(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (v, table, pos) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let p = external_params(dst);
        let (n_tok, ratio, block) = (p[0] as usize, p[1] as usize, p[2] as usize);
        let (d, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let out = dst.data as *mut f32;
        for t in part(ng, ith, nth) {
            let o = std::slice::from_raw_parts_mut(out.add(t * d), d);
            o.fill(0.0);
            let pt = at::<i32>(pos, t) as usize;
            if !(pt + 1).is_multiple_of(ratio) {
                continue;
            }
            let s = t / n_tok;
            let rows: Vec<Vec<f32>> = (pt + 1 - ratio..=pt)
                .map(|q| f16_row(v, cell(table, s, q, block) * v.nb[1], 2 * d))
                .collect();
            for c in 0..d {
                let m = rows
                    .iter()
                    .map(|r| r[d + c])
                    .fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = rows.iter().map(|r| (r[d + c] - m).exp()).collect();
                let sum: f32 = ex.iter().sum();
                o[c] = rows.iter().zip(&ex).map(|(r, e)| r[c] * (e / sum)).sum();
            }
        }
    }
}

unsafe fn cpu_index_scores(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (iq, w, kc, table, pos) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let p = external_params(dst);
        let (n_tok, ratio, koff, block, has_mask) = (
            p[0] as usize,
            p[1] as usize,
            p[2] as usize,
            p[3] as usize,
            p[4] != 0,
        );
        let mask = has_mask.then(|| &*dst.src[5]);
        let (di, ih) = (iq.ne[0] as usize, iq.ne[1] as usize);
        let (nc, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let out = dst.data as *mut f32;
        for t in part(ng, ith, nth) {
            let s = t / n_tok;
            let n_vis = visible(at::<i32>(pos, t) as usize, ratio).min(nc);
            let o = std::slice::from_raw_parts_mut(out.add(t * nc), nc);
            o.fill(f32::NEG_INFINITY);
            let wt = f32s(w, t * w.nb[1], ih);
            for (j, oj) in o.iter_mut().enumerate().take(n_vis) {
                let c = cell(table, s, j * ratio + ratio - 1, block);
                let key = f16_row(kc, c * kc.nb[1] + koff * 2, di);
                let mut sc = 0f32;
                for (h, wh) in wt.iter().enumerate() {
                    let qh = f32s(iq, t * iq.nb[2] + h * iq.nb[1], di);
                    sc += dot(qh, &key).max(0.0) * wh;
                }
                if let Some(m) = mask {
                    sc += *f32s(m, t * m.nb[1] + j * 4, 1).first().unwrap();
                }
                *oj = sc;
            }
        }
    }
}

/// The best `k` of `vals` by value (ties to the lower index), keeping only those `keep` takes.
fn best(vals: &[f32], k: usize, keep: impl Fn(f32) -> bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).filter(|&j| keep(vals[j])).collect();
    // A stable sort: equal values keep the lower index first.
    idx.sort_by(|&a, &b| vals[b].total_cmp(&vals[a]));
    idx.truncate(k);
    idx
}

unsafe fn cpu_cand_mask(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (score, pos) = (&*dst.src[0], &*dst.src[1]);
        let p = external_params(dst);
        let (ratio, bs, tb) = (p[0] as usize, p[1] as usize, p[2] as usize);
        let (nc, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let nb = nc.div_ceil(bs);
        let out = dst.data as *mut f32;
        for t in part(ng, ith, nth) {
            let sc = f32s(score, t * score.nb[1], nc);
            let mut blocks: Vec<f32> = (0..nb)
                .map(|b| {
                    sc[b * bs..((b + 1) * bs).min(nc)]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();
            let n_vis = visible(at::<i32>(pos, t) as usize, ratio);
            if n_vis > 0 && (n_vis - 1) / bs < nb {
                blocks[(n_vis - 1) / bs] = f32::INFINITY;
            }
            let o = std::slice::from_raw_parts_mut(out.add(t * nc), nc);
            o.fill(f32::NEG_INFINITY);
            for b in best(&blocks, tb, |v| v > f32::NEG_INFINITY) {
                o[b * bs..((b + 1) * bs).min(nc)].fill(0.0);
            }
        }
    }
}

unsafe fn cpu_topk(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let score = &*dst.src[0];
        let nc = score.ne[0] as usize;
        let (k, ng) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let out = dst.data as *mut i32;
        for t in part(ng, ith, nth) {
            let sc = f32s(score, t * score.nb[1], nc);
            let mut sel = best(sc, k, f32::is_finite);
            sel.sort_unstable();
            for i in 0..k {
                *out.add(t * k + i) = sel.get(i).map_or(-1, |&j| j as i32);
            }
        }
    }
}

unsafe fn cpu_attn(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (q, kown, table, pos, sinks) = (
            &*dst.src[0],
            &*dst.src[1],
            &*dst.src[2],
            &*dst.src[3],
            &*dst.src[4],
        );
        let (kc, sel, wtable) = (&*dst.src[5], &*dst.src[6], &*dst.src[7]);
        let p = external_params(dst);
        let (n_tok, scale, window, ratio, coff, block, has_comp) = (
            p[0] as usize,
            f32_param(p[1]),
            p[2] as usize,
            p[3] as usize,
            p[4] as usize,
            p[5] as usize,
            p[6] != 0,
        );
        // A ring's positions wrap after its entries' blocks.
        let wrap = match p[7] as usize {
            0 => usize::MAX,
            e => e * block,
        };
        let (d, nh, ng) = (dst.ne[0] as usize, dst.ne[1] as usize, dst.ne[2] as usize);
        let out = dst.data as *mut f32;
        for t in part(ng, ith, nth) {
            let s = t / n_tok;
            let pt = at::<i32>(pos, t) as usize;
            // The keys: the window's, then the compressed ones.
            let mut keys: Vec<Vec<f32>> = (pt.saturating_sub(window - 1)..=pt)
                .map(|w| f16_row(kown, cell(wtable, s, w % wrap, block) * kown.nb[1], d))
                .collect();
            if has_comp {
                let k = sel.ne[0] as usize;
                for i in 0..k {
                    let j = at::<i32>(sel, t * k + i);
                    if j < 0 {
                        continue;
                    }
                    let c = cell(table, s, j as usize * ratio + ratio - 1, block);
                    keys.push(f16_row(kc, c * kc.nb[1] + coff * 2, d));
                }
            }
            for h in 0..nh {
                let qr = f32s(q, t * q.nb[2] + h * q.nb[1], d);
                let score: Vec<f32> = keys.iter().map(|k| dot(qr, k) * scale).collect();
                let m = score.iter().copied().fold(-1e30f32, f32::max);
                let ex: Vec<f32> = score.iter().map(|x| (x - m).exp()).collect();
                let sum: f32 = ex.iter().sum::<f32>() + (at::<f32>(sinks, h) - m).exp();
                let o = std::slice::from_raw_parts_mut(out.add((t * nh + h) * d), d);
                o.fill(0.0);
                for (k, e) in keys.iter().zip(&ex) {
                    let w = e / sum;
                    for i in 0..d {
                        o[i] += k[i] * w;
                    }
                }
            }
        }
    }
}

/// `sqrt(softplus(x))`, softplus as torch's (`x` itself past 20).
fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
    sp.sqrt()
}

unsafe fn cpu_router_ids(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (logits, bias) = (&*dst.src[0], &*dst.src[1]);
        let n_exp = logits.ne[0] as usize;
        let (k, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let b = weight_row(bias, 0);
        let out = dst.data as *mut i32;
        for t in part(n, ith, nth) {
            let l = f32s(logits, t * n_exp * 4, n_exp);
            let sel: Vec<f32> = (0..n_exp).map(|e| sqrt_softplus(l[e]) + b[e]).collect();
            for (i, e) in best(&sel, k, |_| true).into_iter().enumerate() {
                *out.add(t * k + i) = e as i32;
            }
        }
    }
}

unsafe fn cpu_router_w(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    unsafe {
        let (logits, ids) = (&*dst.src[0], &*dst.src[1]);
        let p = external_params(dst);
        let (norm, scale) = (p[0] != 0, f32_param(p[1]));
        let n_exp = logits.ne[0] as usize;
        let (k, n) = (dst.ne[0] as usize, dst.ne[1] as usize);
        let out = dst.data as *mut f32;
        for t in part(n, ith, nth) {
            let l = f32s(logits, t * n_exp * 4, n_exp);
            let w: Vec<f32> = (0..k)
                .map(|i| sqrt_softplus(l[at::<i32>(ids, t * k + i) as usize]))
                .collect();
            let div = if norm && k > 1 {
                w.iter().sum::<f32>() + 1e-20
            } else {
                1.0
            };
            for (i, v) in w.iter().enumerate() {
                *out.add(t * k + i) = v / div * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests;
