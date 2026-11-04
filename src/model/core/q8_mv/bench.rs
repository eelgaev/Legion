//! The Q8_0 draft LM head in isolation (#836 part 13): `w @ x` for a Q8_0 `[k, rows]` and 9-16
//! columns on one GPU, as ggml's `mul_mat` (its quantization and MMQ: the reference bits) and as
//! the op (ggml's quantization and the kernel) in its variants ([`super::force_variant`]), timed
//! (the weight bytes over the time). Each timed graph reads distinct copies of the weight, so
//! that the GPU's L2 cache holds none of it. [`check`] compares the bits in one graph that holds
//! both. Used by the hidden `legion q8-mv-bench` subcommand (a build tree's binary on the GPU
//! box).

use super::{GROUP, MAX_COLS, MIN_COLS, Q8_0_BYTES, force_variant, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

/// One variant: its name and the kernel variant forced (`None`: the default); ggml's `mul_mat`
/// for `op` false.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub op: bool,
    pub force: Option<(i32, i32, i32)>,
}

const fn v(name: &'static str, op: bool, force: Option<(i32, i32, i32)>) -> Variant {
    Variant { name, op, force }
}

/// The variants, the reference (ggml) first.
pub const VARIANTS: [Variant; 7] = [
    v("ggml", false, None),
    v("op c8w4", true, Some((8, 4, 2))),
    v("op c8w2", true, Some((8, 2, 2))),
    v("op c4w8", true, Some((4, 8, 2))),
    v("op c4w4", true, Some((4, 4, 2))),
    v("op c2w8", true, Some((2, 8, 2))),
    v("op c2w16", true, Some((2, 16, 2))),
];

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per row.
    pub k: usize,
    /// Row counts of the weights.
    pub rows: Vec<usize>,
    /// Columns per step.
    pub cols: Vec<usize>,
    pub rounds: usize,
    /// Weight bytes a timed graph reads at least (copies of the weight, at most [`MAX_COPIES`]).
    pub min_bytes: usize,
}

/// Most copies of a weight a graph reads.
pub const MAX_COPIES: usize = 16;

impl Default for Config {
    /// Qwen3.8-Flash-Next's MTP draft head with `--spec-draft-vocab 65536`, and the whole head.
    fn default() -> Self {
        Self {
            k: 2560,
            rows: vec![65536, 248320],
            cols: vec![9, 12, 16],
            rounds: 5,
            min_bytes: 512 << 20,
        }
    }
}

/// One variant of one weight at one step.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub rows: usize,
    pub cols: usize,
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

/// `bytes` (whole Q8_0 blocks) of Q8_0 weights: up to 1 MiB of blocks (int8 in -127..=127,
/// scales in [0.001, 0.011)), repeated; `seed` picks them.
fn weights(bytes: usize, seed: u64) -> Vec<u8> {
    let blocks = (bytes / Q8_0_BYTES).min((1 << 20) / Q8_0_BYTES);
    let qs = noise(blocks * GROUP, seed);
    let ds = noise(blocks, seed + 1);
    let mut pattern = Vec::with_capacity(blocks * Q8_0_BYTES);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        pattern.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * GROUP..(b + 1) * GROUP] {
            pattern.push(((q * 127.0).round() as i8) as u8);
        }
    }
    pattern.iter().copied().cycle().take(bytes).collect()
}

/// The graph of one matmul per weight of `ws`, on the op or ggml's `mul_mat`: the input and the
/// graph.
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

fn check_shape(k: usize, rows: usize) -> Result<(), String> {
    if k == 0 || !k.is_multiple_of(512) || rows == 0 || !rows.is_multiple_of(32) {
        return Err(format!(
            "k {k}, rows {rows}: k a multiple of 512, rows of 32"
        ));
    }
    Ok(())
}

fn cols_ok(cols: usize) -> bool {
    (MIN_COLS as usize..=MAX_COLS as usize).contains(&cols)
}

/// Every variant for every weight and step of `cfg` on `gpu`; the variants take turns within a
/// round, the best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_shape(k, rows)?;
        let bytes = rows * (k / GROUP) * Q8_0_BYTES;
        let copies = cfg.min_bytes.div_ceil(bytes).clamp(1, MAX_COPIES);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q8_0, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(w.nbytes(), 3 + i as u64));
        }
        for &cols in cfg.cols.iter().filter(|&&c| cols_ok(c)) {
            let kinds = [false, true];
            let ctxs = kinds.map(|_| Context::new(16 * copies + 16, 8 * copies + 16));
            let graphs: Vec<_> = kinds
                .iter()
                .zip(&ctxs)
                .map(|(&op, ctx)| build(ctx, op, &ws, k, cols))
                .collect();
            let xs = noise(k * cols, 5 + cols as u64);
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
                if var.op && !force_variant(var.force) {
                    live[vi] = false;
                    continue;
                }
                let t = std::time::Instant::now();
                let r = gpu.compute(graph);
                let secs = t.elapsed().as_secs_f64();
                force_variant(None);
                if let Err(e) = r {
                    if vi == 0 {
                        return Err(format!("{rows} rows {}: {e}", var.name));
                    }
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
                        rows,
                        cols,
                        secs: best[vi],
                        bytes: bytes as f64,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// One weight at one step in [`check`]: outputs whose bits differ from ggml's over every op
/// variant, the variants that ran, and the outputs per variant.
#[derive(Clone, Debug)]
pub struct Check {
    pub rows: usize,
    pub cols: usize,
    pub differing: usize,
    pub variants: usize,
    pub outputs: usize,
}

/// The bits of every op variant against ggml's `mul_mat` in one graph, for every weight and step
/// of `cfg` on `gpu`.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    let k = cfg.k;
    for &rows in &cfg.rows {
        check_shape(k, rows)?;
        for &cols in cfg.cols.iter().filter(|&&c| cols_ok(c)) {
            let ctx = Context::new(32, 32);
            let x = ctx
                .new_tensor(types::F32, &[k as i64, cols as i64])
                .as_input();
            let w = ctx
                .new_tensor(types::Q8_0, &[k as i64, rows as i64])
                .as_input();
            let reference = ctx.mul_mat(w, x).as_output();
            let op = mul_mat(&ctx, w, x).as_output();
            let mut graph = ctx.new_graph(32);
            graph.expand(reference);
            graph.expand(op);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            // Some all-zero groups and one large value: ggml's quantization's corner cases.
            let mut xs = noise(k * cols, 1000 + cols as u64);
            xs[32..64].fill(0.0);
            xs[k + 7] = 300.0;
            x.set_f32(&xs);
            w.set_bytes(&weights(w.nbytes(), 2000 + rows as u64));
            let (mut differing, mut variants) = (0, 0);
            for var in VARIANTS.iter().filter(|v| v.op) {
                if !force_variant(var.force) {
                    continue;
                }
                let r = gpu.compute(&graph);
                force_variant(None);
                r.map_err(|e| format!("{rows} rows {}: {e}", var.name))?;
                variants += 1;
                let (a, b) = (op.to_f32(), reference.to_f32());
                differing += a
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
            }
            out.push(Check {
                rows,
                cols,
                differing,
                variants,
                outputs: rows * cols,
            });
        }
    }
    Ok(out)
}
