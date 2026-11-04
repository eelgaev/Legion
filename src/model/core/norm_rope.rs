//! An attention head's RMS norm, weight and multi-section RoPE as one external op
//! ([`crate::ggml::ExternalOp`], kernels in `src/cuda/norm_rope.cu`): upstream Strata #783 PR-f
//! ("fused rms_norm + rope", by stuchapin909, MIT, via Strata-AC922).
//!
//! - [`norm_rope`]: `rope_multi(mul(rms_norm(x), w), pos)`, for the attention layers' q and k
//!   and qwen4exp's indexer queries: one launch instead of two (ggml fuses the norm with the
//!   weight, but not with a multi-section RoPE).
//! - [`pool_key`]: qwen4exp's pool keys, from the members' raw indexer keys to the F16 store
//!   (`get_rows`, `cont`, `add`s, `scale`, the norm, the RoPE, `set_rows`): one launch instead
//!   of about nine.
//! - [`kv_append`] and [`idx_append`] (upstream Strata #783's batched K/V and indexer appends,
//!   same author and licence): a layer's step K (normed and rotated) and V into the F16 caches
//!   in one launch, for norm, mul, rope and two `set_rows`; the indexer rows (raw key, zero
//!   pool-key half) in one, for `fill`, `concat` and `set_rows`.
//!
//! The CUDA kernels compute what ggml-cuda's kernels compute, operation for operation (the
//! norm's 256-thread reduction order, the RoPE's angle and rotation expressions), so the
//! outputs are ggml's bit for bit (`norm_rope/tests.rs`). All are opt-in. [`norm_rope`]
//! (`legion serve --norm-rope`) keeps the output but gained no decode time on V100. The three
//! cache writers (`legion serve --kv-append`) are faster, but taking ggml's nodes out of the
//! graph moves the compute buffer's layout, and ggml-cuda decides some fusions (whose rounding
//! differs from the unfused ops) by layout, so the model's output moves although each writer's
//! bits do not.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static NORM_ROPE: ExternalOp = ExternalOp {
    name: "norm_rope",
    cpu: cpu_norm_rope,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_norm_rope),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static POOL_KEY: ExternalOp = ExternalOp {
    name: "pool_key",
    cpu: cpu_pool_key,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_pool_key),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static KV_APPEND: ExternalOp = ExternalOp {
    name: "kv_append",
    cpu: cpu_kv_append,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_kv_append),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static IDX_APPEND: ExternalOp = ExternalOp {
    name: "idx_append",
    cpu: cpu_idx_append,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_idx_append),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// ggml's `GGML_ROPE_TYPE_MROPE` (sections in order); IMROPE interleaves them.
const ROPE_MROPE: i32 = 8;
use super::ROPE_IMROPE;

/// Most columns per row (ggml's rms_norm takes its 256-thread block below 1024).
const MAX_COLS: i64 = 1023;

/// Whether graphs built from now on use [`norm_rope`].
static ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on use [`norm_rope`] where it fits; `legion serve
/// --norm-rope` turns it on. Off by default: ggml's chains bit for bit, but on V100 one external
/// launch costs what ggml's two fused, graph-captured launches cost (and splits the graph), so
/// decode gained nothing.
pub fn set_norm_rope(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Whether graphs built from now on use [`kv_append`], [`idx_append`] and [`pool_key`].
static APPEND_ON: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on use [`kv_append`], [`idx_append`] and [`pool_key`]
/// where they fit; `legion serve --kv-append` turns them on. Off by default: the kernels write
/// ggml's bits, but dropping ggml's nodes moves the compute buffer's layout, and ggml-cuda
/// picks some of its fusions (which round differently) by that layout, so greedy output
/// moves.
pub fn set_kv_append(on: bool) {
    APPEND_ON.store(on, Ordering::Relaxed);
}

/// A multi-section RoPE as ggml's `rope_multi` takes it, for the cases the kernels do: no
/// frequency factors, `freq_scale` and `attn_factor` 1, `ext_factor` 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rope {
    pub n_dims: i32,
    pub sections: [i32; 4],
    /// `GGML_ROPE_TYPE_IMROPE` (interleaved sections) or `GGML_ROPE_TYPE_MROPE`.
    pub mode: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
}

impl Rope {
    fn fits(&self, cols: i64) -> bool {
        (self.mode == ROPE_IMROPE || self.mode == ROPE_MROPE)
            && self.freq_scale == 1.0
            && self.ext_factor == 0.0
            && self.attn_factor == 1.0
            && self.n_dims > 0
            && self.n_dims % 2 == 0
            && self.n_dims as i64 <= cols
            && self.sections.iter().all(|&s| s >= 0)
            && self.sections.iter().sum::<i32>() > 0
    }

    fn params(&self) -> [i32; 7] {
        [
            self.n_dims,
            self.sections[0],
            self.sections[1],
            self.sections[2],
            self.sections[3],
            (self.mode == ROPE_IMROPE) as i32,
            self.freq_base.to_bits() as i32,
        ]
    }
}

fn on_gpu(ts: &[Tensor<'_>]) -> bool {
    ON.load(Ordering::Relaxed) && device_only(ts)
}

fn device_only(ts: &[Tensor<'_>]) -> bool {
    cfg!(feature = "cuda") && ts.iter().all(|t| !t.is_host() && !t.is_meta())
}

/// A step's F32 rows `[d, heads, n]` with contiguous rows.
fn f32_heads(x: Tensor<'_>, d: i64, h: i64, n: i64) -> bool {
    x.ty() == types::F32 && x.nb()[0] == 4 && x.ne() == [d, h, n, 1]
}

/// An F16 cache `[row, cells]` with contiguous rows.
fn f16_cache(c: Tensor<'_>, row: i64) -> bool {
    c.ty() == types::F16 && c.nb()[0] == 2 && c.ne()[0] == row && c.ne()[2] == 1 && c.ne()[3] == 1
}

/// Whether [`kv_append`] takes these tensors (see there): on a GPU, switched on, F16 caches,
/// rows of at most 1023 even columns, a RoPE the kernel does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_append_fits(
    k: Tensor<'_>,
    w: Tensor<'_>,
    pos: Tensor<'_>,
    v: Tensor<'_>,
    k_cache: Tensor<'_>,
    v_cache: Tensor<'_>,
    cells: Tensor<'_>,
    rope: &Rope,
) -> bool {
    APPEND_ON.load(Ordering::Relaxed)
        && device_only(&[k, w, pos, v, k_cache, v_cache, cells])
        && kv_append_shapes(k, w, pos, v, k_cache, v_cache, cells, rope)
}

#[allow(clippy::too_many_arguments)]
fn kv_append_shapes(
    k: Tensor<'_>,
    w: Tensor<'_>,
    pos: Tensor<'_>,
    v: Tensor<'_>,
    k_cache: Tensor<'_>,
    v_cache: Tensor<'_>,
    cells: Tensor<'_>,
    rope: &Rope,
) -> bool {
    let [d, h, n, _] = k.ne();
    norm_rope_shapes(k, w, pos, rope)
        && f32_heads(v, d, h, n)
        && f16_cache(k_cache, d * h)
        && f16_cache(v_cache, d * h)
        && h <= 32767
        && cells.ty() == types::I32
        && cells.is_contiguous()
        && cells.elements() as i64 == n
}

/// Appends a layer's step K (`k` F32 `[d, heads, n]` before its norm) and V (`v`, the same
/// shape) to the F16 caches `k_cache` and `v_cache` (`[d × heads, cells]`) at cells `cells`
/// (I32 `[n]`, distinct): K as `rope_multi(mul(rms_norm(k), w), pos)`. ggml's bits (norm, mul,
/// rope, two `set_rows`). The returned node (one float, never written) is what the graph
/// expands; the caches are written as a side effect, so it must be expanded before what reads
/// them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_append<'a>(
    ctx: &'a Context,
    k: Tensor<'a>,
    w: Tensor<'a>,
    pos: Tensor<'a>,
    v: Tensor<'a>,
    k_cache: Tensor<'a>,
    v_cache: Tensor<'a>,
    cells: Tensor<'a>,
    eps: f32,
    rope: &Rope,
) -> Tensor<'a> {
    assert!(
        kv_append_shapes(k, w, pos, v, k_cache, v_cache, cells, rope),
        "kv_append: shapes"
    );
    let mut params = vec![eps.to_bits() as i32];
    params.extend(rope.params());
    ctx.external_with(
        &KV_APPEND,
        types::F32,
        [1, 1, 1, 1],
        &[k, w, pos, v, k_cache, v_cache, cells],
        &params,
    )
}

/// Whether [`idx_append`] takes these tensors: on a GPU, switched on, `raw` F32 `[d, n]` with
/// contiguous rows, `cache` F16 `[2 d, cells]`, `cells` I32 `[n]`.
pub(crate) fn idx_append_fits(raw: Tensor<'_>, cache: Tensor<'_>, cells: Tensor<'_>) -> bool {
    APPEND_ON.load(Ordering::Relaxed)
        && device_only(&[raw, cache, cells])
        && idx_append_shapes(raw, cache, cells)
}

fn idx_append_shapes(raw: Tensor<'_>, cache: Tensor<'_>, cells: Tensor<'_>) -> bool {
    let [d, n, ..] = raw.ne();
    f32_heads(raw, d, n, 1)
        && f16_cache(cache, 2 * d)
        && cells.ty() == types::I32
        && cells.is_contiguous()
        && cells.elements() as i64 == n
}

/// Writes the indexer cache rows of a step: for token `t`, row `cells[t]` of `cache` (F16
/// `[2 d, cells]`) becomes `raw`'s row `t` (F32 `[d, n]`) and `d` zeros (ggml's `fill`,
/// `concat` and `set_rows`, bit for bit). Returns the node to expand, as [`kv_append`].
pub(crate) fn idx_append<'a>(
    ctx: &'a Context,
    raw: Tensor<'a>,
    cache: Tensor<'a>,
    cells: Tensor<'a>,
) -> Tensor<'a> {
    assert!(idx_append_shapes(raw, cache, cells), "idx_append: shapes");
    ctx.external_with(
        &IDX_APPEND,
        types::F32,
        [1, 1, 1, 1],
        &[raw, cache, cells],
        &[],
    )
}

/// Whether [`norm_rope`] takes `x` (F32 `[d, heads, n]`, rows contiguous), `w` (F32 `[d]`) and
/// `pos` (I32 `[4 n]`): on a GPU (the CPU graph keeps ggml's chain), switched on, rows of at
/// most 1023 even columns, a RoPE the kernel does ([`Rope`]).
pub(crate) fn norm_rope_fits(x: Tensor<'_>, w: Tensor<'_>, pos: Tensor<'_>, rope: &Rope) -> bool {
    on_gpu(&[x, w, pos]) && norm_rope_shapes(x, w, pos, rope)
}

fn norm_rope_shapes(x: Tensor<'_>, w: Tensor<'_>, pos: Tensor<'_>, rope: &Rope) -> bool {
    let [d, h, n, rest] = x.ne();
    x.ty() == types::F32
        && x.nb()[0] == 4
        && rest == 1
        && d % 2 == 0
        && (2..=MAX_COLS).contains(&d)
        && h >= 1
        && (1..=65535).contains(&n)
        && w.ty() == types::F32
        && w.is_contiguous()
        && w.elements() as i64 == d
        && pos.ty() == types::I32
        && pos.elements() as i64 == 4 * n
        && rope.fits(d)
}

/// `rope_multi(mul(rms_norm(x, eps), w), pos)` as ggml computes it, F32 `[d, heads, n]`
/// contiguous.
pub(crate) fn norm_rope<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    pos: Tensor<'a>,
    eps: f32,
    rope: &Rope,
) -> Tensor<'a> {
    assert!(norm_rope_shapes(x, w, pos, rope), "norm_rope: shapes");
    let mut params = vec![eps.to_bits() as i32];
    params.extend(rope.params());
    ctx.external_with(&NORM_ROPE, types::F32, x.ne(), &[x, w, pos], &params)
}

/// Whether [`pool_key`] takes these tensors (see there): on a GPU, switched on, F16 cache
/// halves of at most 1023 even columns with one row stride, a RoPE the kernel does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_key_fits(
    raw: Tensor<'_>,
    pooled: Tensor<'_>,
    members: Tensor<'_>,
    pos: Tensor<'_>,
    rep: Tensor<'_>,
    w: Tensor<'_>,
    rope: &Rope,
) -> bool {
    APPEND_ON.load(Ordering::Relaxed)
        && device_only(&[raw, pooled, members, pos, rep, w])
        && pool_key_shapes(raw, pooled, members, pos, rep, w, rope)
}

fn pool_key_shapes(
    raw: Tensor<'_>,
    pooled: Tensor<'_>,
    members: Tensor<'_>,
    pos: Tensor<'_>,
    rep: Tensor<'_>,
    w: Tensor<'_>,
    rope: &Rope,
) -> bool {
    let d = raw.ne()[0];
    let n_new = rep.elements() as i64;
    [raw, pooled]
        .iter()
        .all(|t| t.ty() == types::F16 && t.nb()[0] == 2 && t.ne()[0] == d && t.nb()[1] % 2 == 0)
        && raw.nb()[1] == pooled.nb()[1]
        && d % 2 == 0
        && (2..=MAX_COLS).contains(&d)
        && n_new >= 1
        && members.ty() == types::I32
        && members.is_contiguous()
        && members.elements() as i64 % n_new == 0
        && pos.ty() == types::I32
        && pos.elements() as i64 == 4 * n_new
        && rep.ty() == types::I64
        && rep.is_contiguous()
        && w.ty() == types::F32
        && w.is_contiguous()
        && w.elements() as i64 == d
        && rope.fits(d)
}

/// qwen4exp's pool keys: for pool `j` of `rep` (I64 `[n_new]`), the mean of the raw indexer
/// keys of its members (rows `members[j * kpool ..][..kpool]` of `raw`, F16 `[d, rows]`),
/// normed, weighted by `w`, rotated to position `pos` (I32 `[4 n_new]`) and stored as F16 in
/// row `rep[j]` of `pooled` (F16 `[d, rows]`, the same row stride): ggml's `get_rows`, `cont`,
/// `add`s, `scale(1 / kpool)`, `rms_norm`, `mul`, `rope_multi` and `set_rows`, bit for bit.
/// The rows of `rep` must differ. The returned node (one float, never written) is what the
/// graph expands; `pooled` is written as a side effect.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pool_key<'a>(
    ctx: &'a Context,
    raw: Tensor<'a>,
    pooled: Tensor<'a>,
    members: Tensor<'a>,
    pos: Tensor<'a>,
    rep: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
    rope: &Rope,
) -> Tensor<'a> {
    assert!(
        pool_key_shapes(raw, pooled, members, pos, rep, w, rope),
        "pool_key: shapes"
    );
    let mut params = vec![eps.to_bits() as i32];
    params.extend(rope.params());
    ctx.external_with(
        &POOL_KEY,
        types::F32,
        [1, 1, 1, 1],
        &[raw, pooled, members, pos, rep, w],
        &params,
    )
}

/// The params: eps, then [`Rope::params`].
struct Params {
    eps: f32,
    n_dims: usize,
    sections: [i32; 4],
    imrope: bool,
    freq_base: f32,
}

fn params(dst: &ffi::ggml_tensor) -> Params {
    let p = crate::ggml::external_params(dst);
    Params {
        eps: f32::from_bits(p[0] as u32),
        n_dims: p[1] as usize,
        sections: [p[2], p[3], p[4], p[5]],
        imrope: p[6] != 0,
        freq_base: f32::from_bits(p[7] as u32),
    }
}

/// The CPU reference: norm (f32 sums in column order) and rotation of one row `x` into `out`,
/// token `t` of `n_pos` positions.
fn cpu_row(x: &[f32], w: &[f32], out: &mut [f32], pos: &[i32], t: usize, n_pos: usize, p: &Params) {
    let d = x.len();
    let mean = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
    let scale = 1.0 / (mean + p.eps).sqrt();
    let y: Vec<f32> = x.iter().zip(w).map(|(v, w)| scale * v * w).collect();
    let theta_scale = p.freq_base.powf(-2.0 / p.n_dims as f32);
    let s = p.sections;
    let sect_dims = s.iter().sum::<i32>() as usize;
    let half = p.n_dims / 2;
    out.copy_from_slice(&y);
    for i in 0..half {
        let sector = i % sect_dims;
        let sec = |k: usize| s[k] as usize;
        let row = if p.imrope {
            match sector % 3 {
                1 if sector < 3 * sec(1) => 1,
                2 if sector < 3 * sec(2) => 2,
                0 if sector < 3 * sec(0) => 0,
                _ => 3,
            }
        } else if sector < sec(0) {
            0
        } else if sector < sec(0) + sec(1) {
            1
        } else if sector < sec(0) + sec(1) + sec(2) {
            2
        } else {
            3
        };
        let theta = pos[t + row * n_pos] as f32 * theta_scale.powf(i as f32);
        let (sin, cos) = theta.sin_cos();
        let (x0, x1) = (y[i], y[i + half]);
        out[i] = x0 * cos - x1 * sin;
        out[i + half] = x0 * sin + x1 * cos;
    }
}

unsafe fn cpu_norm_rope(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `norm_rope`, host tensors. The CPU graphs keep ggml's chain; this
    // is the op's reference for tests.
    unsafe {
        let p = params(dst);
        let (x, w, pos) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let [d, h, n] = [x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize];
        let w = std::slice::from_raw_parts(w.data as *const f32, d);
        let pos = std::slice::from_raw_parts(pos.data as *const i32, 4 * n);
        for r in (ith..h * n).step_by(nth) {
            let (hh, t) = (r % h, r / h);
            let xr = (x.data as *const u8).add(hh * x.nb[1] + t * x.nb[2]) as *const f32;
            let xr = std::slice::from_raw_parts(xr, d);
            let out = std::slice::from_raw_parts_mut((dst.data as *mut f32).add(r * d), d);
            cpu_row(xr, w, out, pos, t, n, &p);
        }
    }
}

unsafe fn cpu_pool_key(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `pool_key`, host tensors (the reference for tests).
    unsafe {
        let p = params(dst);
        let src = |i: usize| &*dst.src[i];
        let (raw, pooled, members, pos, rep, w) = (src(0), src(1), src(2), src(3), src(4), src(5));
        let d = raw.ne[0] as usize;
        let n_new = rep.ne[0] as usize;
        let kpool = (members.ne[0] as usize) / n_new;
        let w = std::slice::from_raw_parts(w.data as *const f32, d);
        let pos = std::slice::from_raw_parts(pos.data as *const i32, 4 * n_new);
        let members = std::slice::from_raw_parts(members.data as *const i32, kpool * n_new);
        let rep = std::slice::from_raw_parts(rep.data as *const i64, n_new);
        let row = |t: &ffi::ggml_tensor, r: usize| (t.data as *mut u8).add(r * t.nb[1]) as *mut u16;
        for j in (ith..n_new).step_by(nth) {
            let mut x = vec![0f32; d];
            for (k, &m) in members[j * kpool..(j + 1) * kpool].iter().enumerate() {
                let src = std::slice::from_raw_parts(row(raw, m as usize), d);
                for (xi, &h) in x.iter_mut().zip(src) {
                    let v = ffi::ggml_fp16_to_fp32(h);
                    *xi = if k == 0 { v } else { *xi + v };
                }
            }
            let inv = 1.0 / kpool as f32;
            x.iter_mut().for_each(|v| *v *= inv);
            let mut out = vec![0f32; d];
            cpu_row(&x, w, &mut out, pos, j, n_new, &p);
            let dst = std::slice::from_raw_parts_mut(row(pooled, rep[j] as usize), d);
            for (o, v) in dst.iter_mut().zip(out) {
                *o = ffi::ggml_fp32_to_fp16(v);
            }
        }
    }
}

unsafe fn cpu_kv_append(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `kv_append`, host tensors (the reference for tests).
    unsafe {
        let p = params(dst);
        let src = |i: usize| &*dst.src[i];
        let (k, w, pos, v, kc, vc, cells) =
            (src(0), src(1), src(2), src(3), src(4), src(5), src(6));
        let [d, h, n] = [k.ne[0] as usize, k.ne[1] as usize, k.ne[2] as usize];
        let w = std::slice::from_raw_parts(w.data as *const f32, d);
        let pos = std::slice::from_raw_parts(pos.data as *const i32, 4 * n);
        let cells = std::slice::from_raw_parts(cells.data as *const i32, n);
        let at = |t: &ffi::ggml_tensor, hh: usize, tt: usize| {
            let x = (t.data as *const u8).add(hh * t.nb[1] + tt * t.nb[2]) as *const f32;
            std::slice::from_raw_parts(x, d)
        };
        let out = |c: &ffi::ggml_tensor, hh: usize, tt: usize| {
            let row = (c.data as *mut u8).add(cells[tt] as usize * c.nb[1]) as *mut u16;
            std::slice::from_raw_parts_mut(row.add(hh * d), d)
        };
        for r in (ith..h * n).step_by(nth) {
            let (hh, t) = (r % h, r / h);
            let mut y = vec![0f32; d];
            cpu_row(at(k, hh, t), w, &mut y, pos, t, n, &p);
            for (o, x) in out(kc, hh, t).iter_mut().zip(y) {
                *o = ffi::ggml_fp32_to_fp16(x);
            }
            for (o, &x) in out(vc, hh, t).iter_mut().zip(at(v, hh, t)) {
                *o = ffi::ggml_fp32_to_fp16(x);
            }
        }
    }
}

unsafe fn cpu_idx_append(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `idx_append`, host tensors (the reference for tests).
    unsafe {
        let (raw, cache, cells) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (d, n) = (raw.ne[0] as usize, raw.ne[1] as usize);
        let cells = std::slice::from_raw_parts(cells.data as *const i32, n);
        for t in (ith..n).step_by(nth) {
            let x = (raw.data as *const u8).add(t * raw.nb[1]) as *const f32;
            let x = std::slice::from_raw_parts(x, d);
            let row = (cache.data as *mut u8).add(cells[t] as usize * cache.nb[1]) as *mut u16;
            let row = std::slice::from_raw_parts_mut(row, 2 * d);
            for (i, o) in row.iter_mut().enumerate() {
                *o = ffi::ggml_fp32_to_fp16(if i < d { x[i] } else { 0.0 });
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct RopeC {
    n_dims: i32,
    sections: [i32; 4],
    imrope: i32,
    freq_base: f32,
    freq_scale: f32,
    attn_factor: f32,
}

#[cfg(feature = "cuda")]
impl RopeC {
    fn of(p: &Params) -> Self {
        Self {
            n_dims: p.n_dims as i32,
            sections: p.sections,
            imrope: p.imrope as i32,
            freq_base: p.freq_base,
            freq_scale: 1.0,
            attn_factor: 1.0,
        }
    }
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct NormRopeC {
    x: *const f32,
    x_s1: i64,
    x_s2: i64,
    w: *const f32,
    pos: *const i32,
    out: *mut f32,
    ncols: i32,
    n_head: i32,
    n_tok: i32,
    eps: f32,
    rope: RopeC,
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct PoolKeyC {
    raw: *const u16,
    members: *const i32,
    pos: *const i32,
    rep: *const i64,
    pooled: *mut u16,
    ld: i64,
    w: *const f32,
    ncols: i32,
    kpool: i32,
    n_new: i32,
    inv_kpool: f32,
    eps: f32,
    rope: RopeC,
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct KvAppendC {
    k: *const f32,
    k_s1: i64,
    k_s2: i64,
    w: *const f32,
    pos: *const i32,
    v: *const f32,
    v_s1: i64,
    v_s2: i64,
    k_cache: *mut u16,
    v_cache: *mut u16,
    k_ld: i64,
    v_ld: i64,
    cells: *const i32,
    ncols: i32,
    n_head: i32,
    n_tok: i32,
    eps: f32,
    rope: RopeC,
}

#[cfg(feature = "cuda")]
#[repr(C)]
struct IdxAppendC {
    raw: *const f32,
    raw_s1: i64,
    cache: *mut u16,
    ld: i64,
    cells: *const i32,
    d: i32,
    n_tok: i32,
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_norm_rope_launch(p: *const NormRopeC, stream: *mut c_void) -> i32;
    fn legion_pool_key_launch(p: *const PoolKeyC, stream: *mut c_void) -> i32;
    fn legion_kv_append_launch(p: *const KvAppendC, stream: *mut c_void) -> i32;
    fn legion_idx_append_launch(p: *const IdxAppendC, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
fn launched(name: &str, rc: i32) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        -1 => Err(format!("{name}: shape unsupported")),
        e => Err(format!("{name}: CUDA error {e}")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_norm_rope(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `norm_rope`, device tensors of the stream's device.
    unsafe {
        let p = params(dst);
        let (x, w, pos) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let c = NormRopeC {
            x: x.data as *const f32,
            x_s1: (x.nb[1] / 4) as i64,
            x_s2: (x.nb[2] / 4) as i64,
            w: w.data as *const f32,
            pos: pos.data as *const i32,
            out: dst.data as *mut f32,
            ncols: x.ne[0] as i32,
            n_head: x.ne[1] as i32,
            n_tok: x.ne[2] as i32,
            eps: p.eps,
            rope: RopeC::of(&p),
        };
        launched("norm_rope", legion_norm_rope_launch(&c, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_pool_key(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `pool_key`, device tensors of the stream's device.
    unsafe {
        let p = params(dst);
        let src = |i: usize| &*dst.src[i];
        let (raw, pooled, members, pos, rep, w) = (src(0), src(1), src(2), src(3), src(4), src(5));
        let n_new = rep.ne[0] as i32;
        let kpool = members.ne[0] as i32 / n_new;
        let c = PoolKeyC {
            raw: raw.data as *const u16,
            members: members.data as *const i32,
            pos: pos.data as *const i32,
            rep: rep.data as *const i64,
            pooled: pooled.data as *mut u16,
            ld: (raw.nb[1] / 2) as i64,
            w: w.data as *const f32,
            ncols: raw.ne[0] as i32,
            kpool,
            n_new,
            // As qsa_top passes it to ggml's scale.
            inv_kpool: 1.0 / kpool as f32,
            eps: p.eps,
            rope: RopeC::of(&p),
        };
        launched("pool_key", legion_pool_key_launch(&c, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_kv_append(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `kv_append`, device tensors of the stream's device.
    unsafe {
        let p = params(dst);
        let src = |i: usize| &*dst.src[i];
        let (k, w, pos, v, kc, vc, cells) =
            (src(0), src(1), src(2), src(3), src(4), src(5), src(6));
        let c = KvAppendC {
            k: k.data as *const f32,
            k_s1: (k.nb[1] / 4) as i64,
            k_s2: (k.nb[2] / 4) as i64,
            w: w.data as *const f32,
            pos: pos.data as *const i32,
            v: v.data as *const f32,
            v_s1: (v.nb[1] / 4) as i64,
            v_s2: (v.nb[2] / 4) as i64,
            k_cache: kc.data as *mut u16,
            v_cache: vc.data as *mut u16,
            k_ld: (kc.nb[1] / 2) as i64,
            v_ld: (vc.nb[1] / 2) as i64,
            cells: cells.data as *const i32,
            ncols: k.ne[0] as i32,
            n_head: k.ne[1] as i32,
            n_tok: k.ne[2] as i32,
            eps: p.eps,
            rope: RopeC::of(&p),
        };
        launched("kv_append", legion_kv_append_launch(&c, stream))
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_idx_append(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `idx_append`, device tensors of the stream's device.
    unsafe {
        let (raw, cache, cells) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let c = IdxAppendC {
            raw: raw.data as *const f32,
            raw_s1: (raw.nb[1] / 4) as i64,
            cache: cache.data as *mut u16,
            ld: (cache.nb[1] / 2) as i64,
            cells: cells.data as *const i32,
            d: raw.ne[0] as i32,
            n_tok: raw.ne[1] as i32,
        };
        launched("idx_append", legion_idx_append_launch(&c, stream))
    }
}

#[cfg(test)]
mod tests;
