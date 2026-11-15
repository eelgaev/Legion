//! The Q4_K matvec op in isolation (#689): `w @ x` for the DFlash drafter's Q4_K shapes and 1-8
//! columns on one GPU, as ggml's `mul_mat` (its quantization and mmvq: the reference bits) and as
//! the op in its launch shapes ([`super::force_shape`]), timed (the weight bytes over the time).
//! Each timed graph reads distinct copies of the weight, so that the GPU's L2 cache holds none of
//! it. [`check`] compares the bits in one graph that holds both. Used by the hidden
//! `legion q4k-mv-bench` subcommand (a build tree's binary on the GPU box).

use super::{K, MAX_COLS, Q4_K_BYTES, QK_K, Shape, force_shape, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};
use crate::model::core::q4k_head::bench::{hash, noise, weights};

/// One variant: its name and the launch shape forced (`None`: the default); ggml's `mul_mat` for
/// `op` false.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub op: bool,
    pub force: Option<Shape>,
}

const fn v(name: &'static str, op: bool, force: Option<(usize, usize)>) -> Variant {
    let force = match force {
        Some((warps, rows)) => Some(Shape { warps, rows }),
        None => None,
    };
    Variant { name, op, force }
}

/// The variants, the reference (ggml) first: warps per CTA, rows a warp step.
pub const VARIANTS: [Variant; 6] = [
    v("ggml", false, None),
    v("op", true, None),
    v("op w8r1", true, Some((8, 1))),
    v("op w8r2", true, Some((8, 2))),
    v("op w16r1", true, Some((16, 1))),
    v("op w16r2", true, Some((16, 2))),
];

/// The drafter's Q4_K matmuls of Qwen3.8-27B's DFlash 2 drafter (`k`, rows): Q/K/V fused, the
/// take-in's K/V fused, gate/up fused, the attention output, the 131,072-row head.
pub const DRAFTER_SHAPES: [(usize, usize); 5] = [
    (5120, 6144),
    (5120, 2048),
    (5120, 34816),
    (4096, 5120),
    (5120, 131072),
];

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// The weights: (values per row, rows).
    pub shapes: Vec<(usize, usize)>,
    /// Column counts of the input.
    pub cols: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the weight, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a weight a graph reads.
pub const MAX_COPIES: usize = 32;

impl Default for Config {
    /// The drafter's shapes at 1-8 columns.
    fn default() -> Self {
        Self {
            shapes: DRAFTER_SHAPES.to_vec(),
            cols: (1..=8).collect(),
            rounds: 5,
            min_bytes: 1 << 30,
        }
    }
}

/// One variant of one weight and column count.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    /// Best mean time per copy of the weight over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes.
    pub bytes: f64,
}

/// The graph of one matmul per weight of `ws`, on the op or ggml's `mul_mat`: the input and the
/// graph.
fn build<'a>(
    ctx: &'a Context,
    op: bool,
    ws: &[Tensor<'a>],
    k: usize,
    n: usize,
) -> (Tensor<'a>, Graph<'a>) {
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
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

fn check_shape(k: usize, rows: usize, n: usize) -> Result<(), String> {
    if !K.contains(&(k as i64)) || rows == 0 || n == 0 || n as i64 > MAX_COLS {
        return Err(format!(
            "k {k}, rows {rows}, cols {n}: k one of {K:?}, rows > 0, 1-{MAX_COLS} columns"
        ));
    }
    Ok(())
}

/// Every variant for every weight and column count of `cfg` on `gpu`; the variants take turns
/// within a round, the best round counts. The launch shape is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    for &(k, rows) in &cfg.shapes {
        check_shape(k, rows, 1)?;
        let bytes = rows * (k / QK_K) * Q4_K_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q4_K, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), k, 3 + i as u64));
        }
        for &n in &cfg.cols {
            check_shape(k, rows, n)?;
            let kinds = [false, true];
            let ctxs = kinds.map(|_| Context::new(16 * copies + 16, 8 * copies + 16));
            let graphs: Vec<_> = kinds
                .iter()
                .zip(&ctxs)
                .map(|(&op, ctx)| build(ctx, op, &ws, k, n))
                .collect();
            let xs = noise(k * n, 5);
            for (ctx, (x, graph)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                x.set_f32(&xs);
            }
            let mut live = [true; VARIANTS.len()];
            let order: Vec<usize> = (0..VARIANTS.len())
                .chain((0..cfg.rounds).flat_map(|_| 0..VARIANTS.len()))
                .collect();
            let mut best = [f64::MAX; VARIANTS.len()];
            for (i, &vi) in order.iter().enumerate() {
                let var = &VARIANTS[vi];
                if !live[vi] {
                    continue;
                }
                let (_, graph) = &graphs[usize::from(var.op)];
                if var.op && !force_shape(var.force) {
                    live[vi] = false;
                    continue;
                }
                let t = std::time::Instant::now();
                let r = gpu.compute(graph);
                let secs = t.elapsed().as_secs_f64();
                force_shape(None);
                if let Err(e) = r {
                    if var.force.is_none() {
                        return Err(format!("k {k} rows {rows} cols {n} {}: {e}", var.name));
                    }
                    // A shape whose shared memory does not fit this column count.
                    live[vi] = false;
                    continue;
                }
                if i >= VARIANTS.len() {
                    best[vi] = best[vi].min(secs / copies as f64);
                }
            }
            for (vi, var) in VARIANTS.iter().enumerate() {
                if live[vi] {
                    out.push(Row {
                        variant: var.name,
                        k,
                        rows,
                        cols: n,
                        secs: best[vi],
                        bytes: bytes as f64,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// One weight and column count in [`check`]: outputs whose bits differ from ggml's over every op
/// variant, the variants that ran, the outputs per variant, and an FNV-1a hash of the op's
/// outputs (the last variant's).
#[derive(Clone, Debug)]
pub struct Check {
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    pub differing: usize,
    pub variants: usize,
    pub outputs: usize,
    pub hash: u64,
}

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every weight and
/// column count of `cfg` on `gpu`, with a few inputs (one with an all-zero block and a large
/// value: ggml's quantization's corner cases; one at a residual's scale).
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for &(k, rows) in &cfg.shapes {
        for &n in &cfg.cols {
            check_shape(k, rows, n)?;
            let ctx = Context::new(32, 32);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let w = ctx
                .new_tensor(types::Q4_K, &[k as i64, rows as i64])
                .as_input();
            let reference = ctx.mul_mat(w, x).as_output();
            let op = mul_mat(&ctx, w, x).as_output();
            let mut graph = ctx.new_graph(32);
            graph.expand(reference);
            graph.expand(op);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            w.set_bytes(&weights(w.nbytes(), k, 2000 + rows as u64));
            let (mut differing, mut variants, mut h) = (0, 0, 0);
            for seed in 0..3u64 {
                let mut xs = noise(k * n, 1000 + seed);
                if seed == 1 {
                    xs[32..64].fill(0.0);
                    xs[7] = 300.0;
                }
                if seed == 2 {
                    xs.iter_mut().for_each(|v| *v *= 40.0);
                }
                x.set_f32(&xs);
                for var in VARIANTS.iter().filter(|v| v.op) {
                    if !force_shape(var.force) {
                        continue;
                    }
                    let r = gpu.compute(&graph);
                    force_shape(None);
                    if r.is_err() && var.force.is_some() {
                        // A shape whose shared memory does not fit this column count.
                        continue;
                    }
                    r.map_err(|e| format!("k {k} rows {rows} cols {n} {}: {e}", var.name))?;
                    variants += 1;
                    let (a, b) = (op.to_f32(), reference.to_f32());
                    differing += a
                        .iter()
                        .zip(&b)
                        .filter(|(p, q)| p.to_bits() != q.to_bits())
                        .count();
                    h = hash(&a);
                }
            }
            out.push(Check {
                k,
                rows,
                cols: n,
                differing,
                variants,
                outputs: rows * n,
                hash: h,
            });
        }
    }
    Ok(out)
}
