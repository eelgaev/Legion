//! The op's bits and time against ggml's `mul_mat` on one GPU (#836): for each Q8_0 weight shape
//! and step size, one graph holds ggml's `mul_mat` and the op's result (the bits compared), then
//! graphs of copies of the weight on either side are timed. Used by the hidden
//! `legion q8-mmq-check` subcommand (a build tree's binary on the GPU box).

pub use super::Fixup;
use super::{BLOCK_BYTES, QK, fits, mul_mat_fast, plan};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

/// One weight at one step size.
#[derive(Clone, Debug)]
pub struct Check {
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    /// Whether ggml splits some tile over several CTAs (its fixup kernel runs).
    pub split: bool,
    pub outputs: usize,
    pub differing: usize,
    /// Largest absolute difference.
    pub max_diff: f32,
    /// Best time per matmul, ggml's and the op's (seconds; 0 when not timed).
    pub ggml_secs: f64,
    pub op_secs: f64,
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

/// A Q8_0 matrix of `blocks` blocks: int8 in -127..=127, scales in [0.001, 0.011).
fn weights(blocks: usize, seed: u64) -> Vec<u8> {
    let qs = noise(blocks * QK as usize, seed);
    let ds = noise(blocks, seed + 1);
    let mut out = Vec::with_capacity(blocks * BLOCK_BYTES);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        out.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * QK as usize..(b + 1) * QK as usize] {
            out.push(((q * 127.0).round() as i8) as u8);
        }
    }
    out
}

fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter().zip(b).fold((0, 0f32), |(n, m), (p, q)| {
        (
            n + usize::from(p.to_bits() != q.to_bits()),
            m.max((p - q).abs()),
        )
    })
}

/// The input of `n` columns of `k` values for weight shape `si`: noise with all-zero blocks and
/// one large value (ggml's quantization's corner cases).
fn input(k: usize, n: usize, si: usize) -> Vec<f32> {
    let mut xs = noise(k * n, 1000 + (si * 256 + n) as u64);
    xs[32..64].fill(0.0);
    xs[k / 2 + 7] = 300.0;
    xs
}

/// Every weight shape of `shapes` (`(k, rows)`) at every step size of `cols` on `gpu` with
/// `sms` SMs: the op's bits against ggml's `mul_mat`, and with `rounds` > 0 the best time of
/// each over graphs of copies of the weight (at least `min_bytes` of weights a graph). `fast`:
/// the op's kernel ([`super::set_q8_mmq_fast`]); `fix`: its split-tile add-up
/// ([`super::set_q8_mmq_fixup`]).
#[allow(clippy::too_many_arguments)]
pub fn check(
    gpu: &mut Backend,
    sms: i64,
    fast: i32,
    fix: Fixup,
    shapes: &[(usize, usize)],
    cols: &[usize],
    rounds: usize,
    min_bytes: usize,
) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    for (si, &(k, rows)) in shapes.iter().enumerate() {
        let bytes = rows * (k / QK as usize) * BLOCK_BYTES;
        let copies = min_bytes.div_ceil(bytes).clamp(1, 64);
        let wctx = Context::new(copies + 1, 0);
        let ws: Vec<Tensor<'_>> = (0..copies)
            .map(|_| wctx.new_tensor(types::Q8_0, &[k as i64, rows as i64]))
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(bytes / BLOCK_BYTES, 2000 + (si * 64 + i) as u64));
        }
        for &n in cols {
            let wsh = (types::Q8_0, ws[0].ne(), ws[0].nb());
            let xsh = (
                types::F32,
                [k as i64, n as i64, 1, 1],
                [4, 4 * k, 4 * k * n, 4 * k * n],
            );
            if !fits(wsh, xsh, sms) {
                return Err(format!(
                    "k {k}, rows {rows}, {n} columns: the op does not take them"
                ));
            }
            let xs = input(k, n, si);
            // The bits: both in one graph.
            let ctx = Context::new(16, 16);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let theirs = ctx.mul_mat(ws[0], x).as_output();
            let ours = mul_mat_fast(&ctx, ws[0], x, sms, fast, fix).as_output();
            // A second node on the same input: it reuses the first one's quantized blocks and
            // the counters the first one left at zero (#858).
            let again = mul_mat_fast(&ctx, ws[0], x, sms, fast, fix).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(theirs);
            graph.expand(ours);
            graph.expand(again);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&xs);
            gpu.compute(&graph).map_err(|e| e.to_string())?;
            let (a, b) = (ours.to_f32(), theirs.to_f32());
            let (d1, m1) = compare(&a, &b);
            let (d2, m2) = compare(&again.to_f32(), &b);
            let (differing, max_diff) = (d1 + d2, m1.max(m2));
            // The time: copies of the weight, each side its own graph, taking turns.
            let (mut ggml_secs, mut op_secs) = (0f64, 0f64);
            if rounds > 0 {
                let ctxs = [false, true].map(|_| Context::new(4 * copies + 16, 4 * copies + 16));
                // An input per copy: each matmul quantizes its own, as ggml's does (the op's
                // nodes reuse the blocks of a node before them with the same input, #858).
                let graphs: Vec<(Vec<Tensor<'_>>, Graph<'_>)> = [false, true]
                    .iter()
                    .zip(&ctxs)
                    .map(|(&op, ctx)| {
                        let mut g = ctx.new_graph(4 * copies + 16);
                        let xs: Vec<Tensor<'_>> = ws
                            .iter()
                            .map(|&w| {
                                let x =
                                    ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                                let y = if op {
                                    mul_mat_fast(ctx, w, x, sms, fast, fix)
                                } else {
                                    ctx.mul_mat(w, x)
                                };
                                g.expand(y.as_output());
                                x
                            })
                            .collect();
                        (xs, g)
                    })
                    .collect();
                for (ctx, (inputs, g)) in ctxs.iter().zip(&graphs) {
                    ctx.alloc_graph(g, ComputeAllocator::new(gpu.buffer_type()))
                        .map_err(|e| e.to_string())?;
                    for x in inputs {
                        x.set_f32(&xs);
                    }
                }
                let mut best = [f64::MAX; 2];
                // One warm-up turn each, then the timed rounds.
                for r in 0..=rounds {
                    for (side, (_, g)) in graphs.iter().enumerate() {
                        let t = std::time::Instant::now();
                        gpu.compute(g).map_err(|e| e.to_string())?;
                        let secs = t.elapsed().as_secs_f64() / copies as f64;
                        if r > 0 {
                            best[side] = best[side].min(secs);
                        }
                    }
                }
                (ggml_secs, op_secs) = (best[0], best[1]);
            }
            out.push(Check {
                k,
                rows,
                cols: n,
                split: plan(rows as i64, k as i64, n as i64, sms).split(),
                outputs: a.len(),
                differing,
                max_diff,
                ggml_secs,
                op_secs,
            });
        }
    }
    Ok(out)
}
