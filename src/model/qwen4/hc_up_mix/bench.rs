//! qwen4exp's mixer up matmul and gated mix with BF16 up weights (the NVFP4 checkpoints) at a
//! prompt step's shape, timed on one GPU in each of its kernels (#1183): `up_mix_bf16` (1), the
//! default (0) and `up_mix_bf16_reg` (2, [`super::force_kernel`]). Each timed graph
//! runs [`Config::copies`] mixers, each with an `up` weight of its own, over the same streams
//! (far larger than L2). [`check`] compares every kernel's bits with `up_mix_bf16`'s. Used by the
//! hidden `legion hc-up-mix-bench` subcommand (a build tree's binary on the GPU box).

use super::{force_kernel, last_kernel, set_mix_cast, up_mix, up_mix_rebuilt};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};
use crate::model::nvfp4_mm::bf16_bits;

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per stream (qwen4exp: 2,560 in 4 streams).
    pub e: i64,
    /// The low rank (qwen4exp: 320).
    pub rank: i64,
    /// Tokens of the step.
    pub tokens: Vec<i64>,
    /// Kernels timed or checked ([`super::force_kernel`]'s numbers).
    pub kernels: Vec<i32>,
    /// Mixers a timed graph runs, each with its own `up` weight.
    pub copies: usize,
    pub rounds: usize,
}

impl Default for Config {
    /// Qwen3.8-Flash-Next's mixer at prompt steps of 2K, 4K and 8K tokens and a tail step's
    /// 3,971; `up_mix_bf16`, the default and `up_mix_bf16_reg`.
    fn default() -> Self {
        Self {
            e: 2560,
            rank: 320,
            tokens: vec![2048, 4096, 8192, 3971],
            kernels: vec![1, 0, 2],
            copies: 8,
            rounds: 8,
        }
    }
}

/// The mixer's forms: F32 streams writing the F32 values and their BF16 copy (`f32`), F32
/// streams writing the copy only (`bf16`: the node fusion's mix cast), BF16 streams writing the
/// copy only (`s16`: `legion serve --hc-bf16`), `xn` read (`xn`, F32 values only).
pub const MODES: [&str; 4] = ["f32", "bf16", "s16", "xn"];

/// One kernel in one mode at one token count.
#[derive(Clone, Debug)]
pub struct Row {
    pub mode: &'static str,
    pub kernel: i32,
    /// The kernel that ran ([`super::last_kernel`]).
    pub ran: i32,
    pub tokens: i64,
    /// Best mean time per mixer over the rounds (seconds), `lo`'s BF16 rounding included.
    pub secs: f64,
    /// The bytes a mixer must move (streams, scales, `lo` as BF16, the outputs).
    pub bytes: f64,
}

/// One comparison of [`check`].
#[derive(Clone, Debug)]
pub struct Check {
    pub mode: &'static str,
    pub kernel: i32,
    pub ran: i32,
    pub tokens: i64,
    pub values: usize,
    /// Values whose bits differ from `up_mix_bf16`'s (F32 values and BF16 copy together).
    pub differing: usize,
    /// FNV-1a of the outputs' bytes.
    pub hash: u64,
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

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The bytes one mixer of `cfg` moves at `n` tokens in `mode`.
fn bytes(cfg: &Config, mode: &str, n: i64) -> f64 {
    let (e, n) = (cfg.e as f64, n as f64);
    let streams = 4.0 * e * if mode == "s16" { 2.0 } else { 4.0 };
    let scales = if mode == "xn" { 0.0 } else { 16.0 };
    let out = match mode {
        "f32" => 6.0 * e,
        "xn" => 4.0 * e,
        _ => 2.0 * e,
    };
    n * (streams + scales + 2.0 * cfg.rank as f64 + out)
}

/// A step's inputs (also marked as outputs, so ggml's allocator keeps their memory).
struct Inputs<'a> {
    lo: Tensor<'a>,
    res: Tensor<'a>,
    sc: Tensor<'a>,
    g: Tensor<'a>,
}

fn inputs<'a>(ctx: &'a Context, cfg: &Config, mode: &str, n: i64) -> Inputs<'a> {
    let k = 4 * cfg.e;
    let res = match mode {
        "s16" => ctx.new_tensor(types::BF16, &[cfg.e, 4, n]),
        "xn" => ctx.new_tensor(types::F32, &[k, n]),
        _ => ctx.new_tensor(types::F32, &[cfg.e, 4, n]),
    };
    Inputs {
        lo: ctx
            .new_tensor(types::F32, &[cfg.rank, n])
            .as_input()
            .as_output(),
        res: res.as_input().as_output(),
        sc: ctx.new_tensor(types::F32, &[4, n]).as_input().as_output(),
        g: ctx.new_tensor(types::F32, &[k]).as_input().as_output(),
    }
}

fn fill(i: &Inputs<'_>, cfg: &Config, mode: &str, n: i64) {
    let (k, nu) = ((4 * cfg.e) as usize, n as usize);
    // lo as a SiLU output would be: mostly small, some larger.
    i.lo.set_f32(
        &noise(cfg.rank as usize * nu, 81)
            .iter()
            .map(|v| v * 1.5)
            .collect::<Vec<_>>(),
    );
    let r = noise(k * nu, 82).into_iter().map(|v| 4.0 * v);
    if mode == "s16" {
        i.res.set_bytes(&bf16_bytes(r));
    } else {
        i.res.set_f32(&r.collect::<Vec<_>>());
    }
    if mode == "xn" {
        // No scales or gamma in the graph (not allocated).
        return;
    }
    i.sc.set_f32(
        &noise(4 * nu, 83)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<_>>(),
    );
    i.g.set_f32(
        &noise(k, 84)
            .iter()
            .map(|v| 1.0 + 0.2 * v)
            .collect::<Vec<_>>(),
    );
}

/// `copies` BF16 `up` weights (`[rank, 4 · e]`) and the BF16 copies' outputs (`[e, n]`).
struct Weights<'a> {
    ups: Vec<Tensor<'a>>,
    casts: Vec<Tensor<'a>>,
}

fn weights<'a>(
    ctx: &'a Context,
    gpu: &Backend,
    cfg: &Config,
    n: i64,
) -> Result<Weights<'a>, String> {
    let ups: Vec<Tensor<'a>> = (0..cfg.copies)
        .map(|_| ctx.new_tensor(types::BF16, &[cfg.rank, 4 * cfg.e]))
        .collect();
    let casts: Vec<Tensor<'a>> = (0..cfg.copies)
        .map(|_| ctx.new_tensor(types::BF16, &[cfg.e, n]))
        .collect();
    ctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (j, up) in ups.iter().enumerate() {
        up.set_bytes(&bf16_bytes(
            noise((cfg.rank * 4 * cfg.e) as usize, 100 + j as u64)
                .into_iter()
                .map(|v| 0.25 * v),
        ));
    }
    Ok(Weights { ups, casts })
}

/// A graph of one mixer per weight in `mode`, allocated, its BF16 copies set; the mixers.
fn build<'a>(
    ctx: &'a Context,
    gpu: &Backend,
    w: &Weights<'a>,
    i: &Inputs<'a>,
    mode: &str,
) -> Result<(Graph<'a>, Vec<Tensor<'a>>), String> {
    let mut graph = ctx.new_graph(8 * w.ups.len() + 16);
    let mut outs = Vec::new();
    for &up in &w.ups {
        let y = if mode == "xn" {
            up_mix(ctx, up, i.lo, i.res, 4, 0.25)
        } else {
            up_mix_rebuilt(ctx, up, i.lo, i.res, i.sc, i.g, 4, 0.25)
        };
        let y = y.as_output();
        graph.expand(y);
        outs.push(y);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .map_err(|e| e.to_string())?;
    if mode != "xn" {
        for (y, c) in outs.iter().zip(&w.casts) {
            // SAFETY: nodes of the graph just allocated, which has not run; the copy holds as
            // many BF16 values as the mixer has F32 ones.
            unsafe { set_mix_cast(y.raw(), c.raw(), mode != "f32") };
        }
    }
    Ok((graph, outs))
}

fn check_config(cfg: &Config) -> Result<(), String> {
    if cfg.copies == 0
        || cfg.rounds == 0
        || cfg.e % 64 != 0
        || cfg.rank % 16 != 0
        || cfg.tokens.iter().any(|&n| n < 1)
    {
        return Err(format!(
            "copies > 0, rounds > 0, e a multiple of 64, rank of 16, tokens > 0: {cfg:?}"
        ));
    }
    Ok(())
}

/// Every kernel of `cfg` in every mode of `modes` at every token count on `gpu`; the kernels take
/// turns within a round, the best round counts. The default kernel is back afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config, modes: &[&'static str]) -> Result<Vec<Row>, String> {
    check_config(cfg)?;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        let wctx = Context::new(2 * cfg.copies + 4, 0);
        let w = weights(&wctx, gpu, cfg, n)?;
        for &mode in modes {
            let ctx = Context::new(16 * cfg.copies + 16, 8 * cfg.copies + 16);
            let i = inputs(&ctx, cfg, mode, n);
            let (graph, _) = build(&ctx, gpu, &w, &i, mode)?;
            fill(&i, cfg, mode, n);
            let mut best = vec![f64::MAX; cfg.kernels.len()];
            let mut ran = vec![0; cfg.kernels.len()];
            for round in 0..=cfg.rounds {
                for (ki, &k) in cfg.kernels.iter().enumerate() {
                    force_kernel(k);
                    let t = std::time::Instant::now();
                    let r = gpu.compute(&graph);
                    let secs = t.elapsed().as_secs_f64();
                    force_kernel(0);
                    r.map_err(|e| format!("{n} tokens {mode} kernel {k}: {e}"))?;
                    ran[ki] = last_kernel();
                    if round > 0 {
                        best[ki] = best[ki].min(secs / cfg.copies as f64);
                    }
                }
            }
            for (ki, &k) in cfg.kernels.iter().enumerate() {
                rows.push(Row {
                    mode,
                    kernel: k,
                    ran: ran[ki],
                    tokens: n,
                    secs: best[ki],
                    bytes: bytes(cfg, mode, n),
                });
            }
        }
    }
    Ok(rows)
}

/// At every token count of `cfg` and in every mode, one mixer per kernel: its F32 values (where
/// it writes them) and BF16 copy against `up_mix_bf16`'s, bit for bit. The outputs are filled
/// with NaN before each run, so a value no kernel wrote cannot match.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    check_config(cfg)?;
    let one = Config {
        copies: 1,
        ..cfg.clone()
    };
    let mut out = Vec::new();
    for &n in &cfg.tokens {
        let wctx = Context::new(8, 0);
        let w = weights(&wctx, gpu, &one, n)?;
        for mode in MODES {
            let ctx = Context::new(64, 32);
            let i = inputs(&ctx, cfg, mode, n);
            let (graph, outs) = build(&ctx, gpu, &w, &i, mode)?;
            fill(&i, cfg, mode, n);
            let (y, cast) = (outs[0], w.casts[0]);
            let values = (cfg.e * n) as usize;
            let mut get = |k: i32| -> Result<(Vec<u8>, i32), String> {
                y.set_f32(&vec![f32::NAN; values]);
                cast.set_bytes(&vec![0xff; values * 2]);
                force_kernel(k);
                let r = gpu.compute(&graph);
                force_kernel(0);
                r.map_err(|e| format!("{n} tokens {mode} kernel {k}: {e}"))?;
                let mut b = Vec::new();
                if mode == "f32" || mode == "xn" {
                    b.extend(y.bytes());
                }
                if mode != "xn" {
                    b.extend(cast.bytes());
                }
                Ok((b, last_kernel()))
            };
            let (want, _) = get(1)?;
            for &k in &cfg.kernels {
                let (got, ran) = get(k)?;
                // Values: 4-byte F32 ones first, then 2-byte BF16 ones.
                let f32_len = if mode == "f32" || mode == "xn" {
                    values * 4
                } else {
                    0
                };
                let (wa, wb) = want.split_at(f32_len);
                let (ga, gb) = got.split_at(f32_len);
                let differing = wa
                    .chunks(4)
                    .zip(ga.chunks(4))
                    .filter(|(p, q)| p != q)
                    .count()
                    + wb.chunks(2)
                        .zip(gb.chunks(2))
                        .filter(|(p, q)| p != q)
                        .count();
                out.push(Check {
                    mode,
                    kernel: k,
                    ran,
                    tokens: n,
                    values: wa.len() / 4 + wb.len() / 2,
                    differing,
                    hash: fnv(&got),
                });
            }
        }
    }
    Ok(out)
}
