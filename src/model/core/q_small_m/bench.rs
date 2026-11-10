//! The small-M quantized matmul in isolation (#1210): `w @ x` for a GGUF-quantized `[k, rows]`
//! and 1-16 columns on one GPU, as ggml's `mul_mat` (the reference: MMVQ up to its bound, MMQ
//! above) and as the op in each of its variants ([`super::VARIANTS`]), timed
//! (the weight bytes over the time). Each timed graph reads distinct copies of the weight, so
//! that the GPU's L2 cache holds none of it. [`check`] compares the bits in one graph that holds
//! both. Used by the hidden `legion q-small-m-bench` subcommand (a build tree's binary on the GPU
//! box).

use super::{MAX_COLS, MIN_COLS, VARIANTS, force_variant, ggml_mmvq_max, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Weight types.
    pub types: Vec<ffi::ggml_type>,
    /// Weight shapes: values per row, rows.
    pub shapes: Vec<(usize, usize)>,
    /// Columns per step.
    pub cols: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the weight, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a weight a graph reads.
pub const MAX_COPIES: usize = 16;

impl Default for Config {
    /// The dense 27B's (Qwen3.8-27B) FFN gate/up, FFN down, attention/DeltaNet out, DeltaNet
    /// qkv and its gate, in the types of its UD-Q4_K_XL GGUF.
    fn default() -> Self {
        Self {
            types: vec![types::Q4_K, types::Q5_K, types::Q6_K, types::IQ4_XS],
            shapes: vec![
                (5120, 17408),
                (17408, 5120),
                (6144, 5120),
                (5120, 10240),
                (5120, 6144),
            ],
            cols: vec![1, 2, 3, 4, 5, 6, 8, 16],
            rounds: 5,
            min_bytes: 512 << 20,
        }
    }
}

/// One variant of one weight at one step: `r` 0 for ggml's `mul_mat`, else the op in variant
/// `r` ([`super::VARIANTS`]).
#[derive(Clone, Debug)]
pub struct Row {
    pub ty: ffi::ggml_type,
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    pub r: i32,
    /// Best mean time per copy of the weight over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes.
    pub bytes: f64,
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// `rows` rows of `k` values of type `ty`: up to 64 distinct rows (ggml's CPU quantization of
/// noise), repeated; `seed` picks them.
fn weights(ty: ffi::ggml_type, k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let distinct = rows.min(64);
    let pattern = crate::ggml::quantize(ty, &noise(distinct * k, seed), k);
    let bytes = rows * crate::ggml::row_size(ty, k);
    pattern.iter().copied().cycle().take(bytes).collect()
}

/// The graph of one matmul per weight of `ws`, on the op (`op`) or ggml's `mul_mat`: the input
/// and the graph.
fn build<'a>(
    ctx: &'a Context,
    op: bool,
    ws: &[Tensor<'a>],
    k: usize,
    cols: usize,
) -> (Tensor<'a>, Graph<'a>) {
    let x = ctx
        .new_tensor(types::F32, &[k as i64, cols as i64])
        .as_input();
    let mut graph = ctx.new_graph(8 * ws.len() + 16);
    for &w in ws {
        let y = if op {
            mul_mat(ctx, w, x)
        } else {
            ctx.mul_mat(w, x)
        };
        graph.expand(y.as_output());
    }
    (x, graph)
}

fn check_shape(ty: ffi::ggml_type, k: usize, rows: usize) -> Result<(), String> {
    if super::type_index(ty).is_none() {
        return Err(format!(
            "type {}: not one the op takes",
            crate::ggml::type_name(ty)
        ));
    }
    if k == 0 || !k.is_multiple_of(512) || rows == 0 {
        return Err(format!("k {k}, rows {rows}: k a multiple of 512"));
    }
    Ok(())
}

fn op_cols(cols: usize) -> bool {
    (MIN_COLS as usize..=MAX_COLS as usize).contains(&cols)
}

/// Every variant for every weight and step of `cfg` on `gpu` (the op only for 2-8 columns);
/// the variants take turns within a round, the best round counts. The rows-a-warp choice is
/// back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    for &ty in &cfg.types {
        for &(k, rows) in &cfg.shapes {
            check_shape(ty, k, rows)?;
            let bytes = rows * crate::ggml::row_size(ty, k);
            let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
            let wctx = Context::new(copies + 1, 0);
            let ws: Vec<Tensor<'_>> = (0..copies)
                .map(|_| wctx.new_tensor(ty, &[k as i64, rows as i64]))
                .collect();
            wctx.alloc_tensors(gpu.buffer_type())
                .map_err(|e| e.to_string())?;
            for (i, w) in ws.iter().enumerate() {
                w.set_bytes(&weights(ty, k, rows, 3 + i as u64));
            }
            for &cols in &cfg.cols {
                // Variant 0 ggml, then the op in each variant.
                let rs: Vec<i32> = std::iter::once(0)
                    .chain(VARIANTS.iter().copied().filter(|_| op_cols(cols)))
                    .collect();
                let ctxs: Vec<Context> = rs
                    .iter()
                    .map(|_| Context::new(16 * copies + 16, 8 * copies + 16))
                    .collect();
                let mut graphs = Vec::new();
                for (&r, ctx) in rs.iter().zip(&ctxs) {
                    force_variant((r != 0).then_some(r));
                    graphs.push(build(ctx, r != 0, &ws, k, cols));
                }
                force_variant(None);
                let xs = noise(k * cols, 5 + cols as u64);
                for (ctx, (x, graph)) in ctxs.iter().zip(&graphs) {
                    ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                        .map_err(|e| e.to_string())?;
                    x.set_f32(&xs);
                }
                let mut best = vec![f64::MAX; rs.len()];
                // One warm-up pass, then the timed rounds.
                for round in 0..=cfg.rounds {
                    for (vi, (_, graph)) in graphs.iter().enumerate() {
                        let t = std::time::Instant::now();
                        gpu.compute(graph).map_err(|e| {
                            format!("{} {k}x{rows} {cols} cols r{}: {e}", type_label(ty), rs[vi])
                        })?;
                        let secs = t.elapsed().as_secs_f64();
                        if round > 0 {
                            best[vi] = best[vi].min(secs / copies as f64);
                        }
                    }
                }
                for (vi, &r) in rs.iter().enumerate() {
                    out.push(Row {
                        ty,
                        k,
                        rows,
                        cols,
                        r,
                        secs: best[vi],
                        bytes: bytes as f64,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// The type's name for tables.
pub fn type_label(ty: ffi::ggml_type) -> &'static str {
    crate::ggml::type_name(ty)
}

/// One weight at one step in [`check`]: outputs whose bits differ from ggml's over every
/// rows-a-warp variant, whether ggml runs MMVQ there (`ggml_mmvq`: the bits must match; else
/// MMQ, other sums), the largest relative difference, and the outputs per variant.
#[derive(Clone, Debug)]
pub struct Check {
    pub ty: ffi::ggml_type,
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    pub ggml_mmvq: bool,
    pub differing: usize,
    pub max_rel: f64,
    pub variants: usize,
    pub outputs: usize,
}

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every weight and
/// step of 2-8 columns of `cfg` on `gpu`.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for &ty in &cfg.types {
        for &(k, rows) in &cfg.shapes {
            check_shape(ty, k, rows)?;
            for &cols in cfg.cols.iter().filter(|&&c| op_cols(c)) {
                let ctx = Context::new(64, 64);
                let x = ctx
                    .new_tensor(types::F32, &[k as i64, cols as i64])
                    .as_input();
                let w = ctx.new_tensor(ty, &[k as i64, rows as i64]).as_input();
                let reference = ctx.mul_mat(w, x).as_output();
                let mut graph = ctx.new_graph(64);
                graph.expand(reference);
                let ops: Vec<Tensor<'_>> = VARIANTS
                    .iter()
                    .map(|&r| {
                        force_variant(Some(r));
                        let y = mul_mat(&ctx, w, x).as_output();
                        graph.expand(y);
                        y
                    })
                    .collect();
                force_variant(None);
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                // Some all-zero groups and one large value: ggml's quantization's corner cases.
                let mut xs = noise(k * cols, 1000 + cols as u64);
                xs[32..64].fill(0.0);
                xs[k + 7] = 300.0;
                x.set_f32(&xs);
                w.set_bytes(&weights(ty, k, rows, 2000 + rows as u64));
                gpu.compute(&graph)
                    .map_err(|e| format!("{} {k}x{rows} {cols} cols: {e}", type_label(ty)))?;
                let b = reference.to_f32();
                let (mut differing, mut max_rel) = (0, 0f64);
                for op in &ops {
                    let a = op.to_f32();
                    for (p, q) in a.iter().zip(&b) {
                        if p.to_bits() != q.to_bits() {
                            differing += 1;
                            let rel = f64::from((p - q).abs()) / f64::from(q.abs()).max(1e-6);
                            max_rel = max_rel.max(rel);
                        }
                    }
                }
                out.push(Check {
                    ty,
                    k,
                    rows,
                    cols,
                    ggml_mmvq: cols as i64 <= ggml_mmvq_max(ty),
                    differing,
                    max_rel,
                    variants: ops.len(),
                    outputs: rows * cols,
                });
            }
        }
    }
    Ok(out)
}
