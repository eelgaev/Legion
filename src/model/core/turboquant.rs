//! TurboQuant KV caches (`--cache-type-k/v tq3|tq4`, #552): Zandieh et al., "TurboQuant:
//! Online Vector Quantization with Near-optimal Distortion Rate", arXiv 2504.19874, the MSE
//! stage (TurboQuant_mse).
//!
//! Each (cell, kv head) vector `x` of `d` values (`d` a power of two) is stored as its norm
//! `n = ||x||` (F16) and `b` bits per value: the unit vector `x / n` is rotated by a randomized
//! Hadamard transform `R = H D / sqrt(d)` (`H` the ±1 Walsh-Hadamard matrix, `D` a diagonal of
//! ±1 signs from a fixed seed per layer, [`sign`]), which makes every coordinate close to
//! `N(0, 1 / d)` whatever `x` is, and each coordinate is quantized to the nearest Lloyd-Max
//! centroid `c_k = g_k / sqrt(d)` of that distribution ([`lloyd_max`], computed here by Lloyd's
//! iteration on the standard normal density). So `R x ≈ n · g[idx] / sqrt(d)`.
//!
//! Attention never leaves the rotated basis: `q · k = (R q) · (R k)` for an orthogonal `R`, so
//! the kernels rotate the query once, take dot products against the stored centroids, sum the
//! weighted V rows in the rotated basis too and rotate the result back (`R^T`) once. K and V of a
//! layer share one `R` (one seed: the cache's layer index).
//!
//! The paper's second stage (a 1-bit QJL transform of the residual, TurboQuant_prod, for
//! unbiased inner products) and its outlier-channel split are not implemented: QJL would add,
//! per head, the residual's norm and `d` sign bits after the indices (see [`cell_bytes`]), and a
//! correction term to the kernels' dot products.
//!
//! These are not ggml types: the cache tensors are raw bytes (I8 `[cell_bytes, cells]`), written
//! by [`write`] (in place of `set_rows`) and read by Legion's kernels only: the paged attention op
//! of [`super::paged::tq`], and [`rows_f16`] for the gather path (dequantized to F16 in the
//! original basis).
//!
//! A cell: the kv heads' indices, then their norms (F16), padded to 16 bytes. A head's indices
//! are bit planes per group of 32 values: `b` u32 words per group, bit `l` of word `p` is bit `p`
//! of the index of value `32 × group + l`.

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::OnceLock;

use crate::ggml::{Context, ExternalOp, Graph, Tensor, ffi, types};

/// Whether a head of `d` values can be stored: a power of two the kernels take.
pub fn head_ok(d: usize) -> bool {
    matches!(d, 64 | 128 | 256 | 512)
}

/// Bytes of one head's indices at `bits` per value.
pub fn head_bytes(d: usize, bits: u32) -> usize {
    d * bits as usize / 8
}

/// Bytes of a cache cell of `n_kv` heads of `d` values at `bits` per value: the indices, the F16
/// norms, padded to 16 bytes (the staging copies move 16-byte words).
pub fn cell_bytes(d: usize, n_kv: usize, bits: u32) -> usize {
    (n_kv * (head_bytes(d, bits) + 2)).next_multiple_of(16)
}

/// The standard normal density.
fn pdf(x: f64) -> f64 {
    if x.is_infinite() {
        0.0
    } else {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

/// erf(x) for x >= 0: `2/sqrt(pi) · e^{-x²} · Σ 2^n x^{2n+1} / (1·3·…·(2n+1))`, a series of
/// positive terms (no cancellation), to double precision for the arguments used here.
fn erf_pos(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    let mut n = 0.0;
    while term > sum * 1e-18 {
        n += 1.0;
        term *= 2.0 * x * x / (2.0 * n + 1.0);
        sum += term;
        if n > 2000.0 {
            break;
        }
    }
    (2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp() * sum).min(1.0)
}

/// The standard normal distribution function.
fn cdf(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    let e = erf_pos(x.abs() / std::f64::consts::SQRT_2);
    if x >= 0.0 {
        0.5 + 0.5 * e
    } else {
        0.5 - 0.5 * e
    }
}

/// Lloyd-Max centroids of the standard normal for `2^bits` levels, ascending, iterated until no
/// centroid moves by more than 1e-12 (so they are exact to 1e-9 and better).
pub fn lloyd_max(bits: u32) -> Vec<f64> {
    let k = 1usize << bits;
    // Start at the levels' quantiles' middles.
    let mut g: Vec<f64> = (0..k)
        .map(|i| {
            let p = (i as f64 + 0.5) / k as f64;
            // A rough inverse of the cdf by bisection.
            let (mut lo, mut hi) = (-10.0, 10.0);
            for _ in 0..100 {
                let mid: f64 = 0.5 * (lo + hi);
                if cdf(mid) < p { lo = mid } else { hi = mid }
            }
            0.5 * (lo + hi)
        })
        .collect();
    for _ in 0..1_000_000 {
        let t: Vec<f64> = std::iter::once(f64::NEG_INFINITY)
            .chain(g.windows(2).map(|w| 0.5 * (w[0] + w[1])))
            .chain(std::iter::once(f64::INFINITY))
            .collect();
        let mut moved = 0f64;
        for i in 0..k {
            let (a, b) = (t[i], t[i + 1]);
            let next = (pdf(a) - pdf(b)) / (cdf(b) - cdf(a));
            moved = moved.max((next - g[i]).abs());
            g[i] = next;
        }
        if moved < 1e-12 {
            break;
        }
    }
    g
}

/// MSE of quantizing a standard normal value with `2^bits` Lloyd-Max levels: `1 - Σ p_k g_k²`
/// (the centroids are the cells' conditional means). Per unit vector, TurboQuant's MSE is this
/// over a coordinate's variance `1 / d` times `d` coordinates: the same number.
pub fn lloyd_max_mse(bits: u32) -> f64 {
    let g = lloyd_max(bits);
    let t: Vec<f64> = std::iter::once(f64::NEG_INFINITY)
        .chain(g.windows(2).map(|w| 0.5 * (w[0] + w[1])))
        .chain(std::iter::once(f64::INFINITY))
        .collect();
    1.0 - (0..g.len())
        .map(|i| (cdf(t[i + 1]) - cdf(t[i])) * g[i] * g[i])
        .sum::<f64>()
}

/// The tables the kernels take: per bit width (3, 4), the centroids `g_k` and the boundaries
/// between them (midpoints), as F32. Laid out as `TqTables` in `src/cuda/turboquant.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tables {
    pub g3: [f32; 8],
    pub t3: [f32; 7],
    pub g4: [f32; 16],
    pub t4: [f32; 15],
}

impl Tables {
    pub fn get() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(|| {
            let mut t = Tables {
                g3: [0.0; 8],
                t3: [0.0; 7],
                g4: [0.0; 16],
                t4: [0.0; 15],
            };
            let fill = |bits: u32, g: &mut [f32], b: &mut [f32]| {
                let c = lloyd_max(bits);
                for (o, x) in g.iter_mut().zip(&c) {
                    *o = *x as f32;
                }
                for (o, w) in b.iter_mut().zip(c.windows(2)) {
                    *o = (0.5 * (w[0] + w[1])) as f32;
                }
            };
            fill(3, &mut t.g3, &mut t.t3);
            fill(4, &mut t.g4, &mut t.t4);
            t
        })
    }

    /// The centroids and boundaries for `bits` (3 or 4).
    pub fn of(&self, bits: u32) -> (&[f32], &[f32]) {
        match bits {
            3 => (&self.g3, &self.t3),
            4 => (&self.g4, &self.t4),
            _ => panic!("TurboQuant takes 3 or 4 bits, not {bits}"),
        }
    }
}

/// The sign `D_jj` of coordinate `j` for the layer seed `seed`: a 32-bit integer hash (the same
/// in the kernels, `tq_sign`).
pub fn sign(seed: u32, j: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9E37_79B1) ^ j.wrapping_mul(0x85EB_CA77) ^ 0x5bd1_e995;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    if h & 1 == 0 { 1.0 } else { -1.0 }
}

/// The unnormalized fast Walsh-Hadamard transform of `x` (a power of two long), in place: the
/// butterflies in the kernels' order (strides 1, 2, 4, ...).
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// `1 / sqrt(d)` as F32 (rounded once from F64): the transform's scale, which the kernels take
/// as a parameter, so that they multiply by the same number.
pub fn inv_sqrt(d: usize) -> f32 {
    (1.0 / (d as f64).sqrt()) as f32
}

/// `R x` (`H D x / sqrt(d)`).
pub fn rotate(seed: u32, x: &[f32]) -> Vec<f32> {
    let d = x.len();
    let mut y: Vec<f32> = (0..d).map(|j| sign(seed, j as u32) * x[j]).collect();
    fwht(&mut y);
    let s = inv_sqrt(d);
    y.iter_mut().for_each(|v| *v *= s);
    y
}

/// `R^T y` (`D H y / sqrt(d)`).
pub fn unrotate(seed: u32, y: &[f32]) -> Vec<f32> {
    let d = y.len();
    let mut x = y.to_vec();
    fwht(&mut x);
    let s = inv_sqrt(d);
    (0..d).map(|j| sign(seed, j as u32) * x[j] * s).collect()
}

/// `||x||` in the kernels' order: 32 partial sums of the squares (lane `l` of values `l`,
/// `l + 32`, ...), then their sum.
fn norm(x: &[f32]) -> f32 {
    let mut part = [0f32; 32];
    for (l, p) in part.iter_mut().enumerate() {
        for v in x.iter().skip(l).step_by(32) {
            *p += v * v;
        }
    }
    part.iter().fold(0f32, |s, p| s + p).sqrt()
}

fn f16(x: f32) -> u16 {
    // SAFETY: a pure conversion.
    unsafe { ffi::ggml_fp32_to_fp16(x) }
}

fn f32_of(h: u16) -> f32 {
    // SAFETY: a pure conversion.
    unsafe { ffi::ggml_fp16_to_fp32(h) }
}

/// Encodes one head `x` (`d` values) at `bits` per value: its indices into `words` (`d × bits /
/// 32` u32s, the bit planes) and its norm, as F16 bits.
pub fn encode_head(seed: u32, bits: u32, x: &[f32], words: &mut [u32]) -> u16 {
    let d = x.len();
    let (_, bounds) = Tables::get().of(bits);
    let norm = norm(x);
    let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    let mut z: Vec<f32> = (0..d).map(|j| sign(seed, j as u32) * x[j]).collect();
    fwht(&mut z);
    words.fill(0);
    for (i, &v) in z.iter().enumerate() {
        // Coordinates of the rotated unit vector times sqrt(d): about N(0, 1).
        let zi = v * inv;
        let idx = bounds.iter().filter(|&&t| zi > t).count() as u32;
        let (grp, lane) = (i / 32, i % 32);
        for p in 0..bits {
            words[grp * bits as usize + p as usize] |= ((idx >> p) & 1) << lane;
        }
    }
    f16(norm)
}

/// The index of value `i` of a head's bit planes `words`.
pub fn index_of(words: &[u32], bits: u32, i: usize) -> usize {
    let (grp, lane) = (i / 32, i % 32);
    (0..bits).fold(0, |acc, p| {
        acc | ((((words[grp * bits as usize + p as usize] >> lane) & 1) as usize) << p)
    })
}

/// A head decoded in the rotated basis: `n · g[idx] / sqrt(d)` (≈ `R x`).
pub fn decode_head_rotated(bits: u32, words: &[u32], norm: u16, d: usize) -> Vec<f32> {
    let (g, _) = Tables::get().of(bits);
    let s = f32_of(norm) * inv_sqrt(d);
    (0..d).map(|i| s * g[index_of(words, bits, i)]).collect()
}

/// A head decoded in the original basis (≈ `x`).
pub fn decode_head(seed: u32, bits: u32, words: &[u32], norm: u16, d: usize) -> Vec<f32> {
    unrotate(seed, &decode_head_rotated(bits, words, norm, d))
}

/// One cache cell's layout: `n_kv` heads of `d` values at `bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub d: usize,
    pub n_kv: usize,
    pub bits: u32,
}

impl Layout {
    pub fn bytes(&self) -> usize {
        cell_bytes(self.d, self.n_kv, self.bits)
    }

    fn words(&self) -> usize {
        self.d * self.bits as usize / 32
    }

    /// Encodes `x` (`d × n_kv` values, head after head) into `cell`.
    pub fn encode(&self, seed: u32, x: &[f32], cell: &mut [u8]) {
        let hb = head_bytes(self.d, self.bits);
        let mut w = vec![0u32; self.words()];
        cell.fill(0);
        for h in 0..self.n_kv {
            let n = encode_head(seed, self.bits, &x[h * self.d..][..self.d], &mut w);
            for (i, word) in w.iter().enumerate() {
                cell[h * hb + 4 * i..][..4].copy_from_slice(&word.to_le_bytes());
            }
            cell[self.n_kv * hb + 2 * h..][..2].copy_from_slice(&n.to_le_bytes());
        }
    }

    /// Head `h` of `cell`: its index words and F16 norm.
    pub fn head(&self, cell: &[u8], h: usize) -> (Vec<u32>, u16) {
        let hb = head_bytes(self.d, self.bits);
        let w = (0..self.words())
            .map(|i| u32::from_le_bytes(cell[h * hb + 4 * i..][..4].try_into().unwrap()))
            .collect();
        let n = u16::from_le_bytes(cell[self.n_kv * hb + 2 * h..][..2].try_into().unwrap());
        (w, n)
    }

    /// `cell` decoded, `d × n_kv` values, in the rotated basis or (`rotated` false) the
    /// original one.
    pub fn decode(&self, seed: u32, cell: &[u8], rotated: bool) -> Vec<f32> {
        (0..self.n_kv)
            .flat_map(|h| {
                let (w, n) = self.head(cell, h);
                if rotated {
                    decode_head_rotated(self.bits, &w, n, self.d)
                } else {
                    decode_head(seed, self.bits, &w, n, self.d)
                }
            })
            .collect()
    }
}

/// A TurboQuant cache's description, for the ops: its cell layout and seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cache {
    pub layout: Layout,
    pub seed: u32,
}

/// The startup checks of TurboQuant caches in `types` for a model of `hp`, with `--tp tp` and
/// sequence slots forced on (`slots`): every attention layer's head size 128, 256 or 512 (the
/// attention kernels'; the format and the ops also take 64, for the tests); no tensor
/// parallelism (its caches are split by heads over a meta device, whose ops are ggml's:
/// Legion's kernels do not run there); no sequence slots (read in place by ggml's flash
/// attention, which does not read TurboQuant); no TurboQuant V under a compact cache.
pub fn check(
    hp: &super::Hparams,
    types: super::KvTypes,
    tp: usize,
    slots: bool,
) -> Result<(), String> {
    if !types.has_tq() {
        return Ok(());
    }
    let name = format!(
        "--cache-type-k {:?} / --cache-type-v {:?}",
        types.k, types.v
    );
    if tp > 1 {
        return Err(format!(
            "{name}: TurboQuant caches do not run with --tp {tp} (the split caches are read by \
             ggml's kernels, which do not read them); use a layer split over the GPUs (no --tp) \
             or another cache type"
        ));
    }
    if slots {
        return Err(format!(
            "{name}: --kv-slots on reads the caches with ggml's flash attention, which does not \
             read TurboQuant; leave --kv-slots at auto"
        ));
    }
    for il in (0..hp.n_layer).filter(|&il| hp.owns_kv(il)) {
        let (d, _) = hp.kv_dims(il);
        if hp.compact_kv(il) > 0 && types.v.tq_bits().is_some() {
            return Err(format!(
                "{name}: --gemma4-compact-global-kv stores layer {il}'s granules in the V type, \
                 which TurboQuant cannot (they are not power-of-two heads); use another V type \
                 or drop the flag"
            ));
        }
        if !matches!(d, 128 | 256 | 512) {
            return Err(format!(
                "{name}: layer {il}'s head size is {d}; TurboQuant caches need 128, 256 or 512"
            ));
        }
    }
    Ok(())
}

static TQ_WRITE: ExternalOp = ExternalOp {
    name: "tq_write",
    cpu: cpu_write,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_write),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static TQ_ROWS: ExternalOp = ExternalOp {
    name: "tq_rows",
    cpu: cpu_rows,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_rows),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Adds to `graph` the write of `x` (F32 `[d × n_kv, n]`, contiguous) to cells `cells` (I32
/// `[n]`) of the TurboQuant cache `cache` (I8 `[cell bytes, cells]`), in place of `set_rows`.
/// Negative cells are skipped.
pub fn write<'a>(
    ctx: &'a Context,
    graph: &mut Graph<'a>,
    c: Cache,
    cache: Tensor<'a>,
    x: Tensor<'a>,
    cells: Tensor<'a>,
) {
    assert_eq!(
        cache.ne()[0] as usize,
        c.layout.bytes(),
        "tq_write: cell bytes"
    );
    assert_eq!(x.ty(), types::F32, "tq_write: F32 rows");
    let node = ctx.external_with(
        &TQ_WRITE,
        types::F32,
        [1, 1, 1, 1],
        &[cache, x, cells],
        &params(c, 0),
    );
    graph.expand(node);
}

/// Rows `ids` (I32 `[n]`) of the TurboQuant cache `cache` dequantized to F16 in the original
/// basis: `[d × n_kv × per_id, n]`, `per_id` consecutive cells from cell `id × per_id` per id
/// (64: a block's; 1: a cell's).
pub fn rows_f16<'a>(
    ctx: &'a Context,
    c: Cache,
    cache: Tensor<'a>,
    ids: Tensor<'a>,
    per_id: usize,
) -> Tensor<'a> {
    let row = (c.layout.d * c.layout.n_kv * per_id) as i64;
    ctx.external_with(
        &TQ_ROWS,
        types::F16,
        [row, ids.ne()[0], 1, 1],
        &[cache, ids],
        &params(c, per_id),
    )
}

fn params(c: Cache, per_id: usize) -> Vec<i32> {
    vec![
        c.layout.d as i32,
        c.layout.n_kv as i32,
        c.layout.bits as i32,
        c.seed as i32,
        per_id as i32,
    ]
}

fn cache_of(params: &[i32]) -> (Cache, usize) {
    (
        Cache {
            layout: Layout {
                d: params[0] as usize,
                n_kv: params[1] as usize,
                bits: params[2] as u32,
            },
            seed: params[3] as u32,
        },
        params[4] as usize,
    )
}

unsafe fn cpu_write(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `write`, host tensors.
    unsafe {
        let (c, _) = cache_of(&crate::ggml::external_params(dst));
        let (cache, x, cells) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let row = c.layout.d * c.layout.n_kv;
        let n = cells.ne[0] as usize;
        let bytes = c.layout.bytes();
        for t in (ith..n).step_by(nth) {
            let cell = *(cells.data as *const i32).add(t);
            if cell < 0 {
                continue;
            }
            let xs = std::slice::from_raw_parts(
                (x.data as *const u8).add(t * x.nb[1]) as *const f32,
                row,
            );
            let out = std::slice::from_raw_parts_mut(
                (cache.data as *mut u8).add(cell as usize * cache.nb[1]),
                bytes,
            );
            c.layout.encode(c.seed, xs, out);
        }
    }
}

unsafe fn cpu_rows(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `rows_f16`, host tensors.
    unsafe {
        let (c, per_id) = cache_of(&crate::ggml::external_params(dst));
        let (cache, ids) = (&*dst.src[0], &*dst.src[1]);
        let row = c.layout.d * c.layout.n_kv;
        let n = ids.ne[0] as usize;
        let bytes = c.layout.bytes();
        let mut h = vec![0u16; row];
        for i in (ith..n * per_id).step_by(nth) {
            let (id, j) = (i / per_id, i % per_id);
            let cell = *(ids.data as *const i32).add(id) as usize * per_id + j;
            let src = std::slice::from_raw_parts(
                (cache.data as *const u8).add(cell * cache.nb[1]),
                bytes,
            );
            let x = c.layout.decode(c.seed, src, false);
            ffi::ggml_fp32_to_fp16_row(x.as_ptr(), h.as_mut_ptr(), row as i64);
            let out = (dst.data as *mut u8).add(id * dst.nb[1] + j * row * 2) as *mut u16;
            std::ptr::copy_nonoverlapping(h.as_ptr(), out, row);
        }
    }
}

/// The kernels' description of a cache (`TqCache` in `src/cuda/turboquant.cu`).
#[cfg(feature = "cuda")]
#[repr(C)]
pub(crate) struct KernelCache {
    pub data: *const u8,
    pub cell: i64,
    pub d: i32,
    pub n_kv: i32,
    pub bits: i32,
    pub seed: u32,
    pub rsd: f32,
}

#[cfg(feature = "cuda")]
impl KernelCache {
    pub(crate) fn of(c: Cache, data: *const u8, cell: usize) -> Self {
        Self {
            data,
            cell: cell as i64,
            d: c.layout.d as i32,
            n_kv: c.layout.n_kv as i32,
            bits: c.layout.bits as i32,
            seed: c.seed,
            rsd: inv_sqrt(c.layout.d),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_tq_write(
        c: *const KernelCache,
        t: *const Tables,
        x: *const f32,
        x_nb1: i64,
        cells: *const i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    fn legion_tq_rows(
        c: *const KernelCache,
        t: *const Tables,
        ids: *const i32,
        n: i32,
        per_id: i32,
        out: *mut u16,
        out_nb1: i64,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(feature = "cuda")]
fn launched(rc: i32, what: &str) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        1 => Err(format!("{what}: unsupported shape")),
        e => Err(format!("{what}: CUDA launch failed ({e})")),
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_write(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `write`, device tensors.
    unsafe {
        let (c, _) = cache_of(&crate::ggml::external_params(dst));
        let (cache, x, cells) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let kc = KernelCache::of(c, cache.data as *const u8, cache.nb[1]);
        launched(
            legion_tq_write(
                &kc,
                Tables::get(),
                x.data as *const f32,
                x.nb[1] as i64,
                cells.data as *const i32,
                cells.ne[0] as i32,
                stream,
            ),
            "tq_write",
        )
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_rows(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `rows_f16`, device tensors.
    unsafe {
        let (c, per_id) = cache_of(&crate::ggml::external_params(dst));
        let (cache, ids) = (&*dst.src[0], &*dst.src[1]);
        let kc = KernelCache::of(c, cache.data as *const u8, cache.nb[1]);
        launched(
            legion_tq_rows(
                &kc,
                Tables::get(),
                ids.data as *const i32,
                ids.ne[0] as i32,
                per_id as i32,
                dst.data as *mut u16,
                dst.nb[1] as i64,
                stream,
            ),
            "tq_rows",
        )
    }
}

#[cfg(test)]
mod tests;
