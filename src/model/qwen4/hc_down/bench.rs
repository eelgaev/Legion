//! The mixer's combine-and-norm and down matmul of a prompt step with BF16 streams (#1183), timed
//! on one GPU: main's path (the norm writes `xnb`, ggml-cuda's cuBLAS GEMM reads it) against
//! the norm without `xnb` and [`super::mul_mat`] (each token tile, and the one chosen by step
//! size), and each norm alone, so the GEMMs' own times are the differences. Each timed graph
//! chains [`Config::copies`] mixers (each one's new streams the next one's) with down weights of
//! their own, as a step's layers. [`check`] compares the rebuilt B operand with `xnb` value for
//! value (one-hot down weights) and the output with cuBLAS's. Used by the hidden `legion
//! hc-down-bench` subcommand (a build tree's binary on the GPU box).

use super::super::hc_norm;
use super::{M, force_nt, mul_mat};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};
use crate::model::nvfp4_mm::bf16_bits;

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per stream (qwen4exp: 2,560 in 4 streams).
    pub e: i64,
    /// Tokens of the step.
    pub tokens: Vec<i64>,
    /// Mixers a graph chains, each with its own down weight.
    pub copies: usize,
    pub rounds: usize,
}

impl Default for Config {
    /// Qwen3.8-Flash-Next's mixer at prompt steps of 4,096 tokens and a tail step's 3,971.
    fn default() -> Self {
        Self {
            e: 2560,
            tokens: vec![4096, 3971],
            copies: 8,
            rounds: 10,
        }
    }
}

/// The graphs [`run`] times.
const VARIANTS: [&str; 7] = ["main", "rebuilt", "nt2", "nt3", "nt4", "norm_xnb", "norm"];

/// One variant at one token count.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub tokens: i64,
    /// Best mean time per mixer over the rounds (seconds).
    pub secs: f64,
}

/// One comparison of [`check`].
#[derive(Clone, Debug)]
pub struct Check {
    pub tokens: i64,
    pub what: &'static str,
    pub values: usize,
    /// Values that differ (bit for bit, or for `vs_cublas` none: see `rel`).
    pub differing: usize,
    /// Relative L2 distance (`vs_cublas`; 0 elsewhere).
    pub rel: f64,
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

fn bf16_bytes(x: impl Iterator<Item = f32>) -> Vec<u8> {
    x.flat_map(|v| bf16_bits(v).to_ne_bytes()).collect()
}

/// A mixer's weights: gamma, inject weights and down weight.
struct Weights<'a> {
    sets: Vec<[Tensor<'a>; 3]>,
}

fn weights<'a>(ctx: &'a Context, gpu: &Backend, cfg: &Config) -> Result<Weights<'a>, String> {
    let k = cfg.e * 4;
    let sets: Vec<[Tensor<'a>; 3]> = (0..cfg.copies)
        .map(|_| {
            [
                ctx.new_tensor(types::F32, &[k]),
                ctx.new_tensor(types::F32, &[k, 4]),
                ctx.new_tensor(types::BF16, &[k, M]),
            ]
        })
        .collect();
    ctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (i, [g, w, down]) in sets.iter().enumerate() {
        let s = 10 * i as u64;
        g.set_f32(
            &noise(k as usize, s + 1)
                .iter()
                .map(|v| 1.0 + 0.2 * v)
                .collect::<Vec<_>>(),
        );
        w.set_f32(
            &noise((k * 4) as usize, s + 2)
                .iter()
                .map(|v| v * 0.01)
                .collect::<Vec<_>>(),
        );
        down.set_bytes(&bf16_bytes(
            noise((k * M) as usize, s + 3).into_iter().map(|v| 0.02 * v),
        ));
    }
    Ok(Weights { sets })
}

/// A step's inputs: the BF16 streams, the block output and its inject weights.
struct Inputs<'a> {
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
}

/// The inputs, also marked as outputs: ggml's allocator would otherwise hand their memory to
/// later nodes once read, and the graph runs again without refilling them.
fn inputs<'a>(ctx: &'a Context, cfg: &Config, n: i64) -> Inputs<'a> {
    Inputs {
        res: ctx
            .new_tensor(types::BF16, &[cfg.e, 4, n])
            .as_input()
            .as_output(),
        out: ctx
            .new_tensor(types::F32, &[cfg.e, n])
            .as_input()
            .as_output(),
        inj: ctx.new_tensor(types::F32, &[4, n]).as_input().as_output(),
    }
}

fn fill(i: &Inputs<'_>, cfg: &Config, n: i64) {
    let n = n as usize;
    let e = cfg.e as usize;
    i.res.set_bytes(&bf16_bytes(
        noise(e * 4 * n, 91).into_iter().map(|v| 4.0 * v),
    ));
    i.out.set_f32(&noise(e * n, 92));
    i.inj.set_f32(&noise(4 * n, 93));
}

/// A graph chaining every copy's mixer front: `variant` of [`VARIANTS`].
fn build<'a>(ctx: &'a Context, w: &Weights<'a>, i: &Inputs<'a>, variant: &str) -> Graph<'a> {
    let mut graph = ctx.new_graph(16 * w.sets.len() + 16);
    let mut res = i.res;
    for &[g, inj_w, down] in &w.sets {
        let xnb = matches!(variant, "main" | "norm_xnb");
        let (next, lo) = if xnb {
            let p = hc_norm::hc_post_norm_rebuilt_xnb(
                ctx, res, i.out, i.inj, g, inj_w, 1e-6, down, true,
            );
            let lo = (variant == "main").then(|| ctx.mul_mat(down, p.xnb));
            (p.res, lo.unwrap_or(p.xnb))
        } else {
            let p = hc_norm::hc_post_norm_b16_streams(ctx, res, i.out, i.inj, g, inj_w, 1e-6, down);
            let lo = (variant != "norm").then(|| mul_mat(ctx, down, p.res, p.scales, g));
            (p.res, lo.unwrap_or(p.scales))
        };
        graph.expand(lo.as_output());
        res = next;
    }
    graph.expand(res.as_output());
    graph
}

fn check_config(cfg: &Config) -> Result<(), String> {
    if cfg.copies == 0
        || cfg.rounds == 0
        || cfg.e % super::BK != 0
        || cfg.tokens.iter().any(|&n| n < 1)
    {
        return Err(format!(
            "copies > 0, rounds > 0, e a multiple of 32, tokens > 0: {cfg:?}"
        ));
    }
    Ok(())
}

/// Every variant at every token count of `cfg` on `gpu`; the variants take turns within a
/// round, the best round counts. The token tile is back to its choice by step size afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    check_config(cfg)?;
    let wctx = Context::new(3 * cfg.copies + 4, 0);
    let w = weights(&wctx, gpu, cfg)?;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        let mut graphs = Vec::new();
        let ctxs: Vec<Context> = VARIANTS
            .iter()
            .map(|_| Context::new(32 * cfg.copies + 32, 16 * cfg.copies + 16))
            .collect();
        for (ctx, v) in ctxs.iter().zip(VARIANTS) {
            let i = inputs(ctx, cfg, n);
            let graph = build(ctx, &w, &i, v);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            fill(&i, cfg, n);
            graphs.push(graph);
        }
        let mut best = vec![f64::MAX; VARIANTS.len()];
        for round in 0..=cfg.rounds {
            for (vi, v) in VARIANTS.iter().enumerate() {
                force_nt(match *v {
                    "nt2" => 2,
                    "nt3" => 3,
                    "nt4" => 4,
                    _ => 0,
                });
                let t = std::time::Instant::now();
                let r = gpu.compute(&graphs[vi]);
                let secs = t.elapsed().as_secs_f64();
                force_nt(0);
                r.map_err(|e| format!("{n} tokens {v}: {e}"))?;
                if round > 0 {
                    best[vi] = best[vi].min(secs / cfg.copies as f64);
                }
            }
        }
        for (vi, v) in VARIANTS.iter().enumerate() {
            rows.push(Row {
                variant: v,
                tokens: n,
                secs: best[vi],
            });
        }
    }
    Ok(rows)
}

/// At every token count of `cfg`, one mixer: the norm without `xnb` writes main's streams,
/// scales and inject weights (bit for bit); [`super::mul_mat`] with one-hot down weights (row
/// `r` of weight `j` reads column `j · 320 + r`) gives `xnb`'s values for every column and token
/// (each token tile forced in turn); with a random down weight, its distance to cuBLAS's GEMM
/// of `xnb`.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    check_config(cfg)?;
    let k = cfg.e * 4;
    let hot_n = ((k + M - 1) / M) as usize;
    let wctx = Context::new(hot_n + 8, 0);
    let w = weights(
        &wctx,
        gpu,
        &Config {
            copies: 1,
            ..cfg.clone()
        },
    )?;
    let hctx = Context::new(hot_n + 1, 0);
    let hot: Vec<Tensor<'_>> = (0..hot_n)
        .map(|_| hctx.new_tensor(types::BF16, &[k, M]))
        .collect();
    hctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (j, h) in hot.iter().enumerate() {
        let mut v = vec![0f32; (k * M) as usize];
        for r in 0..M {
            v[(r * k + (j as i64 * M + r) % k) as usize] = 1.0;
        }
        h.set_bytes(&bf16_bytes(v.into_iter()));
    }
    let [g, inj_w, down] = w.sets[0];
    let mut out = Vec::new();
    for &n in &cfg.tokens {
        let ctx = Context::new(4 * hot_n + 64, 4 * hot_n + 64);
        let i = inputs(&ctx, cfg, n);
        let a = hc_norm::hc_post_norm_rebuilt_xnb(
            &ctx, i.res, i.out, i.inj, g, inj_w, 1e-6, down, true,
        );
        let b = hc_norm::hc_post_norm_b16_streams(&ctx, i.res, i.out, i.inj, g, inj_w, 1e-6, down);
        let f32_of = |t| ctx.cont(ctx.cast(t, types::F32)).as_output();
        let parts = [
            ("streams", f32_of(a.res), f32_of(b.res)),
            ("scales", f32_of(a.scales), f32_of(b.scales)),
            ("inject", f32_of(a.inject), f32_of(b.inject)),
        ];
        let xnb = f32_of(a.xnb);
        let hot_lo: Vec<Tensor<'_>> = hot
            .iter()
            .map(|&h| mul_mat(&ctx, h, b.res, b.scales, g).as_output())
            .collect();
        let lo = mul_mat(&ctx, down, b.res, b.scales, g).as_output();
        let lo_ref = ctx.mul_mat(down, a.xnb).as_output();
        let mut graph = ctx.new_graph(4 * hot_n + 64);
        for (_, x, y) in &parts {
            graph.expand(*x);
            graph.expand(*y);
        }
        for t in hot_lo.iter().chain([&xnb, &lo, &lo_ref]) {
            graph.expand(*t);
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        fill(&i, cfg, n);
        for nt in [0, 2, 3, 4] {
            force_nt(nt);
            let r = gpu.compute(&graph);
            force_nt(0);
            r.map_err(|e| format!("{n} tokens: {e}"))?;
            if nt == 0 {
                for (what, x, y) in &parts {
                    let (x, y) = (x.bytes(), y.bytes());
                    out.push(Check {
                        tokens: n,
                        what: *what,
                        values: x.len() / 4,
                        differing: x
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .zip(y.as_chunks::<4>().0)
                            .filter(|(p, q)| p != q)
                            .count(),
                        rel: 0.0,
                    });
                }
                let (x, y) = (lo.to_f32(), lo_ref.to_f32());
                let (num, den) = x.iter().zip(&y).fold((0f64, 0f64), |(s, d), (&p, &q)| {
                    let diff = f64::from(p) - f64::from(q);
                    (s + diff * diff, d + f64::from(q) * f64::from(q))
                });
                out.push(Check {
                    tokens: n,
                    what: "vs_cublas",
                    values: x.len(),
                    differing: x.iter().zip(&y).filter(|(p, q)| p != q).count(),
                    rel: (num / den.max(f64::MIN_POSITIVE)).sqrt(),
                });
            }
            let want = xnb.to_f32();
            let mut differing = 0;
            let mut values = 0;
            for (j, h) in hot_lo.iter().enumerate() {
                let got = h.to_f32();
                for t in 0..n {
                    for r in 0..M {
                        let col = (j as i64 * M + r) % k;
                        values += 1;
                        // Equal values: a one-hot sum turns -0 into +0.
                        if got[(t * M + r) as usize] != want[(t * k + col) as usize] {
                            differing += 1;
                        }
                    }
                }
            }
            out.push(Check {
                tokens: n,
                what: ["xnb", "xnb_nt2", "xnb_nt3", "xnb_nt4"][(nt.max(1) - 1) as usize],
                values,
                differing,
                rel: 0.0,
            });
        }
    }
    Ok(out)
}
