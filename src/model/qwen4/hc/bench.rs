//! The hyper-connection mixer op in isolation (#814): [`super::combine_mix`] at a step's shape on
//! one GPU, as the stage graphs of a decode or MTP verify step run it (the previous block's
//! combine, the inject rows and the block input's Q8_1 blocks), in its kernel variants. Each
//! timed graph holds [`Config::copies`] mixers with weights of their own, so that the GPU's L2
//! holds none of a mixer's weights when it starts, as in a step. [`check`] compares every
//! variant's bytes with main's kernels' (`ahead`). Used by the hidden `legion
//! hc-mix-bench` subcommand (a build tree's binary on the GPU box; `ncu` can profile it).

use super::{combine_mix, set_mix_loads_ahead};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};
use crate::model::core::q8_group;

/// One kernel variant: its name, whether the loads go ahead, and the kernel choice it forces
/// ([`super::force_variant`]; negative: the defaults).
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub ahead: bool,
    pub force: i32,
}

/// The variants, the reference (the kernels before #814's variants: main's) first; `default`
/// is what `legion serve` runs. The one-unit-at-a-time kernels
/// cannot write the block input's Q8_1 blocks, so they are not among them (the GPU test
/// `cuda_loads_ahead_is_bit_for_bit` compares them with the default).
pub const VARIANTS: [Variant; 8] = [
    Variant {
        name: "ahead",
        ahead: true,
        force: 0,
    },
    Variant {
        name: "normreg",
        ahead: true,
        force: 2,
    },
    Variant {
        name: "split",
        ahead: true,
        force: 4,
    },
    Variant {
        name: "both",
        ahead: true,
        force: 6,
    },
    // The norm in registers with down_hoist (16), up_mix_stage (32), both (the defaults on a
    // V100 since #814's second round).
    Variant {
        name: "hoist",
        ahead: true,
        force: 2 | 16,
    },
    Variant {
        name: "stage",
        ahead: true,
        force: 2 | 32,
    },
    Variant {
        name: "hstage",
        ahead: true,
        force: 2 | 16 | 32,
    },
    Variant {
        name: "default",
        ahead: true,
        force: -1,
    },
];

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    pub n_embd: i64,
    pub hc: i64,
    pub rank: i64,
    /// Tokens of the step.
    pub tokens: Vec<i64>,
    /// Mixers a graph runs, each with its own weights.
    pub copies: usize,
    pub rounds: usize,
}

impl Default for Config {
    /// Qwen3.8-Flash-Next's mixer: 2,560 values, 4 streams, rank 320; decode, MTP k=1 and k=2.
    fn default() -> Self {
        Self {
            n_embd: 2560,
            hc: 4,
            rank: 320,
            tokens: vec![1, 2, 3],
            copies: 32,
            rounds: 20,
        }
    }
}

/// One variant at one token count.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub tokens: i64,
    /// Best mean time per mixer over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes a mixer reads (down, up, inject).
    pub bytes: f64,
}

/// One variant's bytes against the reference's at one token count.
#[derive(Clone, Debug)]
pub struct Check {
    pub variant: &'static str,
    pub tokens: i64,
    pub bytes: usize,
    pub differing: usize,
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

/// A Q8_0 matrix of `rows` rows of `k` values (int8 in -127..=127, scales in [0.001, 0.011)).
fn q8_0(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let blocks = k / 32 * rows;
    let qs = noise(blocks * 32, seed);
    let ds = noise(blocks, seed + 1);
    let mut out = Vec::with_capacity(blocks * 34);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.006 + 0.005 * d) };
        out.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * 32..(b + 1) * 32] {
            out.push(((q * 127.0).round() as i8) as u8);
        }
    }
    out
}

/// The mixers' weights: (norm, down, up, inject) per copy.
struct Weights<'a> {
    sets: Vec<[Tensor<'a>; 4]>,
}

fn weights<'a>(ctx: &'a Context, gpu: &Backend, cfg: &Config) -> Result<Weights<'a>, String> {
    let k = cfg.n_embd * cfg.hc;
    let sets: Vec<[Tensor<'a>; 4]> = (0..cfg.copies)
        .map(|_| {
            [
                ctx.new_tensor(types::F32, &[k]),
                ctx.new_tensor(types::Q8_0, &[k, cfg.rank]),
                ctx.new_tensor(types::Q8_0, &[cfg.rank, k]),
                ctx.new_tensor(types::F32, &[k, cfg.hc]),
            ]
        })
        .collect();
    ctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (i, [norm, down, up, inj]) in sets.iter().enumerate() {
        let s = 10 * i as u64;
        norm.set_f32(
            &noise(k as usize, s + 1)
                .iter()
                .map(|v| 1.0 + 0.2 * v)
                .collect::<Vec<_>>(),
        );
        down.set_bytes(&q8_0(k as usize, cfg.rank as usize, s + 2));
        up.set_bytes(&q8_0(cfg.rank as usize, k as usize, s + 4));
        inj.set_f32(
            &noise((k * cfg.hc) as usize, s + 6)
                .iter()
                .map(|v| v * 0.01)
                .collect::<Vec<_>>(),
        );
    }
    Ok(Weights { sets })
}

/// The inputs of a step: x, the previous block's output and its inject.
struct Inputs<'a> {
    x: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
}

/// A graph of every copy's mixer for `n` tokens; with `keep`, every output part is a graph
/// output (for [`check`]).
fn build<'a>(
    ctx: &'a Context,
    w: &Weights<'a>,
    cfg: &Config,
    n: i64,
    keep: bool,
) -> (Inputs<'a>, Graph<'a>, Vec<Tensor<'a>>) {
    let (e, hc) = (cfg.n_embd, cfg.hc);
    let x = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let mut graph = ctx.new_graph(16 * cfg.copies + 16);
    let mut kept = Vec::new();
    for &[norm, down, up, inj_w] in &w.sets {
        let (mixed, inject, res) =
            combine_mix(ctx, x, Some((out, inj)), norm, down, up, Some(inj_w), 1e-6);
        let mut parts = vec![mixed, inject.expect("inject"), res];
        if q8_group::has_quantized(ctx, mixed) {
            parts.push(q8_group::quantized(ctx, mixed));
        }
        for t in parts {
            if keep {
                kept.push(t.as_output());
            }
            graph.expand(t);
        }
        if !keep {
            graph.expand(mixed.as_output());
        }
    }
    (Inputs { x, out, inj }, graph, kept)
}

fn fill(i: &Inputs<'_>, cfg: &Config, n: i64) {
    let (e, hc) = (cfg.n_embd as usize, cfg.hc as usize);
    let n = n as usize;
    i.x.set_f32(
        &noise(e * hc * n, 91)
            .iter()
            .map(|v| v * 4.0)
            .collect::<Vec<_>>(),
    );
    i.out.set_f32(&noise(e * n, 92));
    i.inj.set_f32(&noise(hc * n, 93));
}

/// Sets `v`'s kernels; back to the defaults with `None`.
fn select(v: Option<&Variant>) {
    set_mix_loads_ahead(v.is_none_or(|v| v.ahead));
    super::force_variant(v.map_or(-1, |v| v.force));
}

fn check_config(cfg: &Config) -> Result<(), String> {
    if cfg.copies == 0
        || cfg
            .tokens
            .iter()
            .any(|&n| !(1..=super::MAX_TOKENS).contains(&n))
    {
        return Err(format!(
            "copies > 0 and tokens 1..={}: {cfg:?}",
            super::MAX_TOKENS
        ));
    }
    Ok(())
}

/// Every variant at every token count of `cfg` on `gpu`; the variants take turns within a
/// round, the best round counts. The kernels are back at their defaults afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    check_config(cfg)?;
    super::force_quant(true);
    let wctx = Context::new(4 * cfg.copies + 4, 0);
    let w = weights(&wctx, gpu, cfg)?;
    let k = (cfg.n_embd * cfg.hc) as f64;
    let bytes = 2.0 * k / 32.0 * 34.0 * cfg.rank as f64 + 4.0 * k * cfg.hc as f64;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        // A graph per variant: the op's choices are read when the graph is computed, but its
        // quant output is fixed at build time, the same for all.
        let ctx = Context::new(64 * cfg.copies + 64, 16 * cfg.copies + 16);
        let (inputs, graph, _) = build(&ctx, &w, cfg, n, false);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        fill(&inputs, cfg, n);
        let mut best = vec![f64::MAX; VARIANTS.len()];
        for round in 0..=cfg.rounds {
            for (vi, v) in VARIANTS.iter().enumerate() {
                select(Some(v));
                let t = std::time::Instant::now();
                let r = gpu.compute(&graph);
                let secs = t.elapsed().as_secs_f64();
                select(None);
                r.map_err(|e| format!("{n} tokens {}: {e}", v.name))?;
                if round > 0 {
                    best[vi] = best[vi].min(secs / cfg.copies as f64);
                }
            }
        }
        for (vi, v) in VARIANTS.iter().enumerate() {
            rows.push(Row {
                variant: v.name,
                tokens: n,
                secs: best[vi],
                bytes,
            });
        }
    }
    super::force_quant(false);
    Ok(rows)
}

/// Every variant's output bytes (mixed, inject, the combined streams and the block input's
/// Q8_1 blocks of every copy) against [`VARIANTS`]`[0]`'s (the default's), at every token count of `cfg`.
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    check_config(cfg)?;
    super::force_quant(true);
    let wctx = Context::new(4 * cfg.copies + 4, 0);
    let w = weights(&wctx, gpu, cfg)?;
    let mut out = Vec::new();
    for &n in &cfg.tokens {
        let ctx = Context::new(64 * cfg.copies + 64, 16 * cfg.copies + 16);
        let (inputs, graph, kept) = build(&ctx, &w, cfg, n, true);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        fill(&inputs, cfg, n);
        let mut reference: Vec<u8> = Vec::new();
        for v in &VARIANTS {
            select(Some(v));
            let r = gpu.compute(&graph);
            select(None);
            r.map_err(|e| format!("{n} tokens {}: {e}", v.name))?;
            let bytes: Vec<u8> = kept.iter().flat_map(|t| t.bytes()).collect();
            if reference.is_empty() {
                reference = bytes.clone();
            }
            let differing = bytes.iter().zip(&reference).filter(|(a, b)| a != b).count();
            let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
            });
            out.push(Check {
                variant: v.name,
                tokens: n,
                bytes: bytes.len(),
                differing,
                hash,
            });
        }
    }
    super::force_quant(false);
    Ok(out)
}
