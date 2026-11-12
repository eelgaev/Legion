//! The op against ggml in isolation (#689): a DeltaNet layer's alpha and beta (BF16 `[k, 48]`
//! each by default, Qwen3.8-27B's) over 1-8 tokens on one GPU, once as ggml's two `mul_mat`s
//! (`mul_mat_vec_f`) and once as the op. Checks that every output has ggml's bits and times
//! both per layer. Used by `legion gdn-bench --ab` (a build tree's binary on the GPU box).

use super::{fits, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, types};

/// Shapes and runs.
#[derive(Clone, Debug)]
pub struct Config {
    /// Row lengths (the model's hidden size).
    pub ks: Vec<usize>,
    /// Rows of each weight of a layer (alpha's, beta's).
    pub rows: Vec<usize>,
    /// Tokens per step.
    pub tokens: Vec<usize>,
    /// Layers, each with its own weights (more than the L2 cache holds, so the reads stream).
    pub layers: usize,
    /// Timed rounds (the best counts; ggml and the op take turns within a round).
    pub rounds: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ks: vec![5120],
            rows: vec![48, 48],
            tokens: (1..=8).collect(),
            layers: 48,
            rounds: 5,
        }
    }
}

/// One variant at one shape and token count.
#[derive(Clone, Debug)]
pub struct Row {
    /// `ggml` or the op (`mvf`).
    pub variant: &'static str,
    pub k: usize,
    pub tokens: usize,
    /// Best mean time per layer over the rounds (seconds), the graph's launches included.
    pub secs: f64,
    /// Output values whose bits differ from ggml's.
    pub differ: usize,
    /// Output values compared.
    pub outputs: usize,
}

/// xorshift values in [-1, 1).
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

/// BF16 bytes of `v` (truncated).
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

const VARIANTS: [&str; 2] = ["ggml", "mvf"];

/// ggml and the op at every shape and token count of `cfg` on `gpu`.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let nw = cfg.rows.len();
    for &k in &cfg.ks {
        let wctx = Context::new(nw * cfg.layers + 2, 0);
        let ws: Vec<Vec<Tensor<'_>>> = (0..cfg.layers)
            .map(|_| {
                cfg.rows
                    .iter()
                    .map(|&m| wctx.new_tensor(types::BF16, &[k as i64, m as i64]))
                    .collect()
            })
            .collect();
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (l, lw) in ws.iter().enumerate() {
            for (i, w) in lw.iter().enumerate() {
                let m = cfg.rows[i];
                let v: Vec<f32> = noise(k * m, (1 + 7 * l + i) as u64)
                    .iter()
                    .map(|v| v * 0.04)
                    .collect();
                w.set_bytes(&bf16_bytes(&v));
            }
        }
        for &n in &cfg.tokens {
            // Per layer: the matmuls or the op and its views, and the copies.
            let nodes = cfg.layers * (2 * nw + 2) + 8;
            let ctxs = [(); 2].map(|_| Context::new(nodes + 4, nodes + 4));
            let mut graphs = Vec::new();
            for (vi, ctx) in ctxs.iter().enumerate() {
                let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                let mut graph = ctx.new_graph(nodes);
                if vi == 1 && !fits(&ws[0], x) {
                    return Err(format!("the op does not take k {k} at {n} tokens"));
                }
                let mut outs = Vec::new();
                for lw in &ws {
                    let ys = if vi == 0 {
                        lw.iter().map(|&w| ctx.mul_mat(w, x)).collect()
                    } else {
                        mul_mat(ctx, lw, x)
                    };
                    for y in ys {
                        // A contiguous copy, so both read back the same layout (the op's
                        // results are row views of one node); the same copy in both graphs.
                        let y = ctx.cont(y).as_output();
                        graph.expand(y);
                        outs.push(y);
                    }
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                graphs.push((x, graph, outs));
            }
            let xs = noise(k * n, (99 + n) as u64);
            let mut best = [f64::MAX; 2];
            let mut reference: Vec<u32> = Vec::new();
            let mut differ = [0usize; 2];
            for round in 0..=cfg.rounds.max(1) {
                let lead = round % 2;
                for vi in [lead, 1 - lead] {
                    let (x, graph, outs) = &graphs[vi];
                    x.set_f32(&xs);
                    let t = std::time::Instant::now();
                    gpu.compute(graph)
                        .map_err(|e| format!("{}: {e}", VARIANTS[vi]))?;
                    let secs = t.elapsed().as_secs_f64();
                    if round > 0 {
                        best[vi] = best[vi].min(secs / cfg.layers as f64);
                        continue;
                    }
                    let got: Vec<u32> = outs
                        .iter()
                        .flat_map(|o| o.to_f32())
                        .map(f32::to_bits)
                        .collect();
                    if vi == 0 {
                        reference = got;
                    } else {
                        differ[vi] = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
                    }
                }
            }
            for (vi, name) in VARIANTS.iter().enumerate() {
                out.push(Row {
                    variant: name,
                    k,
                    tokens: n,
                    secs: best[vi],
                    differ: differ[vi],
                    outputs: reference.len(),
                });
            }
        }
    }
    Ok(out)
}
