//! The prefill expert GEMMs of NVFP4 checkpoints in isolation (#824): Qwen3.8-Flash-Next's
//! routed experts (gate and up `[2560, 640]`, down `[640, 2560]`, 512 experts, 10 per token) at
//! prompt chunks on one GPU, the fused gate/up/SwiGLU GEMM and the down GEMM, each in the GEMM's
//! tile shapes, stage counts and occupancies. Checks that every variant gives the default's bits
//! and times them. Used by the hidden `legion nvfp4-gemm-bench` subcommand (a build tree's binary
//! on the GPU box).

use super::bench::routing_tiles;
use super::{
    GemmTune, Grouped, Kernel, MV_MAX_N, bf16_bits, force_expert_l2, force_gemm_tune,
    force_grouped_tiles, force_kernel, glu_gemm_tile, mul_mat_id, pick_grouped, plan_grouped,
    set_expert_mv_max, set_expert_mv_tile, swiglu_id,
};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};

/// Which expert GEMM a variant runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// The fused gate/up/SwiGLU GEMM (`swiglu_id`, BF16 out).
    Glu,
    /// The down GEMM (`mul_mat_id` on the BF16 SwiGLU output, F32 out).
    Down,
}

/// Which kernel a variant runs on (#836: short prompt steps, where the default is the grouped
/// mv kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// The default choice ([`super::pick_grouped`]).
    Default,
    /// The grouped GEMM at any size (`--nvfp4-expert-mv-max 0`), in `tile` / `tune` or, with
    /// neither, the GEMM's own choice.
    Gemm,
    /// The grouped mv kernel at any size, token tiles up to `8 * nt`
    /// (`--nvfp4-expert-mv-tile nt`).
    Mv(i32),
}

/// One variant: the op, its tile (`mi`, `nt`; the SwiGLU's `mi` is always 2) and its stages and
/// occupancy; `None`s are the defaults. A `tile` or `tune` is the GEMM's. `exact`: the default
/// with a grid of the routing's own token tile count instead of the host's upper bound (#836:
/// what the CTAs that find no tile cost). `l2` (#1183): the default GEMM with this L2
/// prefetch run and these cache hints ([`super::force_expert_l2`]).
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub op: Op,
    pub tile: Option<(i32, i32)>,
    pub tune: Option<GemmTune>,
    pub family: Family,
    pub exact: bool,
    pub l2: Option<(u8, u8)>,
}

impl Variant {
    /// The op's default choice, the reference of the bit checks.
    pub fn is_default(&self) -> bool {
        self.family == Family::Default
            && self.tile.is_none()
            && self.tune.is_none()
            && !self.exact
            && self.l2.is_none()
    }

    /// The kernel family this variant runs on at `per` pairs per expert.
    fn runs_mv(&self, per: i64) -> bool {
        match self.family {
            Family::Default => self.tile.is_none() && self.tune.is_none() && per <= MV_MAX_N,
            Family::Gemm => false,
            Family::Mv(_) => true,
        }
    }

    pub fn name(&self) -> String {
        let op = match self.op {
            Op::Glu => "glu",
            Op::Down => "down",
        };
        if self.exact {
            return format!("{op} exact");
        }
        if let Some((pf, hint)) = self.l2 {
            return format!("{op} pf{pf} h{hint}");
        }
        match self.family {
            Family::Mv(nt) => return format!("{op} mv nt<={nt}"),
            Family::Gemm if self.tile.is_none() && self.tune.is_none() => {
                return format!("{op} gemm auto");
            }
            _ => {}
        }
        match (self.tile, self.tune) {
            (None, None) => format!("{op} default"),
            (t, u) => {
                let (mi, nt) = t.unwrap_or((0, 0));
                let u = u.unwrap_or(super::GEMM_DEFAULT);
                format!("{op} mi{mi} nt{nt} s{}c{}", u.stages, u.ctas)
            }
        }
    }
}

/// The stage counts and occupancies tried (the default first).
const TUNES: [(i32, i32); 5] = [(3, 1), (2, 1), (2, 2), (3, 2), (4, 1)];

/// The variants: each op's default first, then its tiles in every tune, then (#836) the GEMM's
/// own choice and its small tiles at any size and the mv kernel at each widest token tile; each
/// default's exact grid right after it.
pub fn variants() -> Vec<Variant> {
    let tune = |(stages, ctas): (i32, i32)| Some(GemmTune { stages, ctas });
    let v = |op, tile, tune, family| Variant {
        op,
        tile,
        tune,
        family,
        exact: false,
        l2: None,
    };
    let exact = |op| Variant {
        exact: true,
        ..v(op, None, None, Family::Default)
    };
    // #1183: the default GEMM's L2 prefetch run (blocks) and cache hints.
    let l2 = |op, pf, hint| Variant {
        l2: Some((pf, hint)),
        ..v(op, None, None, Family::Default)
    };
    const L2: [(u8, u8); 9] = [
        (4, 0),
        (8, 0),
        (16, 0),
        (24, 0),
        (32, 1),
        (32, 2),
        (32, 3),
        (8, 1),
        (8, 3),
    ];
    let mut out = vec![v(Op::Glu, None, None, Family::Default), exact(Op::Glu)];
    out.extend(L2.map(|(pf, hint)| l2(Op::Glu, pf, hint)));
    for nt in [8, 16] {
        for t in TUNES {
            out.push(v(Op::Glu, Some((2, nt)), tune(t), Family::Default));
        }
    }
    out.push(v(Op::Glu, None, None, Family::Gemm));
    out.push(v(Op::Glu, Some((2, 4)), tune((3, 1)), Family::Gemm));
    for nt in [1, 2, 4] {
        out.push(v(Op::Glu, None, None, Family::Mv(nt)));
    }
    out.push(v(Op::Down, None, None, Family::Default));
    out.push(exact(Op::Down));
    out.extend(L2.map(|(pf, hint)| l2(Op::Down, pf, hint)));
    for (mi, nt) in [(1, 8), (1, 16), (2, 8), (2, 16)] {
        for t in TUNES {
            out.push(v(Op::Down, Some((mi, nt)), tune(t), Family::Default));
        }
    }
    out.push(v(Op::Down, None, None, Family::Gemm));
    for (mi, nt) in [(1, 2), (1, 4), (2, 4)] {
        out.push(v(Op::Down, Some((mi, nt)), tune((3, 1)), Family::Gemm));
    }
    for nt in [1, 2, 4] {
        out.push(v(Op::Down, None, None, Family::Mv(nt)));
    }
    out
}

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub k: usize,
    pub m: usize,
    pub n_expert: usize,
    pub n_used: usize,
    /// Layers a timed graph holds (the same weights, routed differently).
    pub layers: usize,
    /// Tokens per prompt chunk.
    pub toks: Vec<usize>,
    /// Each token picks its experts from this many of them.
    pub pools: Vec<usize>,
    /// Routings per round.
    pub sets: usize,
    pub rounds: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k: 2560,
            m: 640,
            n_expert: 512,
            n_used: 10,
            layers: 4,
            toks: vec![4096],
            pools: vec![512, 160],
            sets: 2,
            rounds: 3,
        }
    }
}

/// One variant at one chunk size and pool.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: String,
    pub op: Op,
    pub n_tok: usize,
    pub pool: usize,
    /// Best mean time per layer over the rounds (seconds).
    pub secs: f64,
    /// Multiply-adds times two per layer.
    pub flops: f64,
    /// Output bytes whose bits differ from the op's default; `None` where there is no kernel.
    pub differ: Option<usize>,
    /// Whether the variant runs on the default's kernel family (the mv kernel or the GEMM), so
    /// it should give the default's bits.
    pub bitwise: bool,
    /// FNV-1a of the checked outputs' bytes (0 where there is no kernel): equal across
    /// binaries where they give the same bits.
    pub hash: u64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-2, 2).
    fn value(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

/// Random NVFP4 blocks: sub-block scales of moderate size, codes uniform.
fn weights(bytes: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(bytes);
    while out.len() < bytes {
        let r = rng.next();
        for i in 0..4 {
            out.push((((3 + (r >> (8 * i)) % 6) << 3) | (r >> (32 + 3 * i)) & 7) as u8);
        }
        for _ in 0..4 {
            out.extend(rng.next().to_le_bytes());
        }
    }
    out.truncate(bytes);
    out
}

/// Each token's `n_used` distinct experts from a random pool of `pool`.
fn routing(n_used: usize, n_tok: usize, n_expert: usize, pool: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1);
    let mut all: Vec<i32> = (0..n_expert as i32).collect();
    for i in 0..pool.min(n_expert) {
        let j = i + (rng.next() % (n_expert - i) as u64) as usize;
        all.swap(i, j);
    }
    let pool = &all[..pool.min(n_expert).max(n_used)];
    let mut out = Vec::with_capacity(n_used * n_tok);
    for _ in 0..n_tok {
        let mut chosen = Vec::with_capacity(n_used);
        while chosen.len() < n_used {
            let e = pool[(rng.next() % pool.len() as u64) as usize];
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        out.extend(chosen);
    }
    out
}

/// The graph of `layers` of `op` on the weights `w` (gate, up, down) and their scales, each
/// layer with its own routing: the input, the routings, the graph and the outputs.
#[allow(clippy::type_complexity)]
fn build<'a>(
    ctx: &'a Context,
    op: Op,
    [wg, wu, wd, sg, su, sd]: [Tensor<'a>; 6],
    cfg: &Config,
    n_tok: usize,
) -> (Tensor<'a>, Vec<Tensor<'a>>, Graph<'a>, Vec<Tensor<'a>>) {
    let (k, m, n_used) = (cfg.k as i64, cfg.m as i64, cfg.n_used as i64);
    let x = match op {
        Op::Glu => ctx.new_tensor(types::BF16, &[k, 1, n_tok as i64]),
        Op::Down => ctx.new_tensor(types::BF16, &[m, n_used, n_tok as i64]),
    }
    .as_input();
    let ids: Vec<_> = (0..cfg.layers)
        .map(|_| {
            ctx.new_tensor(types::I32, &[n_used, n_tok as i64])
                .as_input()
        })
        .collect();
    let mut graph = ctx.new_graph(16 * cfg.layers + 16);
    let outs = ids
        .iter()
        .map(|&id| {
            let y = match op {
                Op::Glu => swiglu_id(ctx, wg, wu, x, id, Some(sg), Some(su)),
                Op::Down => mul_mat_id(ctx, wd, x, id, Some(sd)),
            }
            .as_output();
            graph.expand(y);
            y
        })
        .collect();
    (x, ids, graph, outs)
}

/// The token tiles of the default launch of `op` for the routing set `set` (one routing per
/// layer): the most of any layer, at the tile the default picks (call after [`force`]).
fn exact_tiles(op: Op, cfg: &Config, n_tok: usize, set: &[Vec<i32>]) -> i32 {
    let (k, m, n_expert) = (cfg.k as i64, cfg.m as i64, cfg.n_expert as i64);
    let pairs = (cfg.n_used * n_tok) as i64;
    let cols = match op {
        Op::Glu => match plan_grouped(k, m, pairs, n_tok as i64, n_expert, true, false) {
            Grouped::One(Kernel::Mv { nt, .. }) => 8 * nt,
            _ => 8 * glu_gemm_tile(k, n_expert, pairs / pairs.min(n_expert).max(1)).0,
        },
        Op::Down => pick_grouped(m, k, pairs, n_expert).tile().1 as i32,
    };
    set.iter()
        .map(|ids| routing_tiles(ids, cols as usize))
        .max()
        .unwrap_or(1) as i32
}

/// FNV-1a of `bytes`: the outputs' bits in one number, to compare across binaries.
fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn force(v: Option<&Variant>) {
    force_grouped_tiles(None, None);
    // A GEMM tile or tune, or the GEMM family: the GEMM at any size; the mv family: the mv
    // kernel at any size.
    let (mv_max, mv_nt) = match v {
        Some(Variant {
            family: Family::Mv(nt),
            ..
        }) => (i64::MAX, *nt),
        Some(v) if v.family == Family::Gemm || v.tile.is_some() || v.tune.is_some() => (0, 4),
        _ => (MV_MAX_N, 4),
    };
    set_expert_mv_max(mv_max);
    set_expert_mv_tile(mv_nt);
    let tile = v.and_then(|v| v.tile);
    let down = v.is_some_and(|v| v.op == Op::Down);
    force_kernel(
        tile.filter(|_| down)
            .map(|(mi, nt)| Kernel::Gemm { mi, nt }),
    );
    let glu_nt = tile.filter(|_| !down).map(|(_, nt)| nt);
    force_gemm_tune(v.and_then(|v| v.tune), glu_nt);
    let l2 = v.and_then(|v| v.l2);
    force_expert_l2(true, l2.filter(|_| !down));
    force_expert_l2(false, l2.filter(|_| down));
}

/// Every variant at every chunk size and pool of `cfg` on `gpu`; the variants take turns within
/// a round, the best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let (k, m, n_expert) = (cfg.k, cfg.m, cfg.n_expert);
    let blocks = |k: usize, m: usize| k / 64 * 36 * m;
    let weights_ctx = Context::new(8, 0);
    let shape = |a: usize, b: usize| [a as i64, b as i64, n_expert as i64];
    let wg = weights_ctx.new_tensor(types::NVFP4, &shape(k, m));
    let wu = weights_ctx.new_tensor(types::NVFP4, &shape(k, m));
    let wd = weights_ctx.new_tensor(types::NVFP4, &shape(m, k));
    let sg = weights_ctx.new_tensor(types::F32, &[n_expert as i64]);
    let su = weights_ctx.new_tensor(types::F32, &[n_expert as i64]);
    let sd = weights_ctx.new_tensor(types::F32, &[n_expert as i64]);
    weights_ctx
        .alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (i, w) in [wg, wu, wd].into_iter().enumerate() {
        w.set_bytes(&weights(blocks(k, m) * n_expert, 3 + i as u64));
    }
    for (i, s) in [sg, su, sd].into_iter().enumerate() {
        let mut rng = Rng(17 + i as u64);
        let v: Vec<f32> = (0..n_expert)
            .map(|_| 0.002 + (rng.next() % 1000) as f32 * 1e-5)
            .collect();
        s.set_f32(&v);
    }
    let variants = variants();
    let down0 = variants
        .iter()
        .position(|v| v.op == Op::Down)
        .expect("a down default");
    let mut rows = Vec::new();
    for &n_tok in &cfg.toks {
        for &pool in &cfg.pools {
            let sets: Vec<Vec<Vec<i32>>> = (0..cfg.sets as u64)
                .map(|i| {
                    (0..cfg.layers as u64)
                        .map(|l| routing(cfg.n_used, n_tok, n_expert, pool, 1 + 64 * i + l))
                        .collect()
                })
                .collect();
            let ops = [Op::Glu, Op::Down];
            let ctxs = ops.map(|_| Context::new(16 + 24 * cfg.layers, 16 * cfg.layers + 16));
            let graphs: Vec<_> = ops
                .into_iter()
                .zip(&ctxs)
                .map(|(op, ctx)| build(ctx, op, [wg, wu, wd, sg, su, sd], cfg, n_tok))
                .collect();
            for (ctx, (x, _, graph, _)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                let mut rng = Rng(5 + n_tok as u64);
                let n = x.nbytes() / 2;
                let xs: Vec<u8> = (0..n)
                    .flat_map(|_| bf16_bits(rng.value()).to_le_bytes())
                    .collect();
                x.set_bytes(&xs);
            }
            // A warm-up run of each variant (and its check), then the rounds.
            let nv = variants.len();
            let order: Vec<usize> = (0..nv).chain((0..cfg.rounds).flat_map(|_| 0..nv)).collect();
            let mut best = vec![f64::MAX; nv];
            let mut differ: Vec<Option<usize>> = vec![Some(0); nv];
            let mut hashes = vec![0u64; nv];
            let mut reference: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
            for (i, &vi) in order.iter().enumerate() {
                let var = &variants[vi];
                if differ[vi].is_none() {
                    continue;
                }
                let gi = (var.op == Op::Down) as usize;
                let (_, ids, graph, outs) = &graphs[gi];
                force(Some(var));
                let mut total = 0.0;
                for (si, set) in sets.iter().enumerate() {
                    for (it, ids) in ids.iter().zip(set) {
                        it.set_i32(ids);
                    }
                    if var.exact {
                        force_grouped_tiles(None, Some(exact_tiles(var.op, cfg, n_tok, set)));
                    }
                    let t = std::time::Instant::now();
                    let r = gpu.compute(graph);
                    total += t.elapsed().as_secs_f64();
                    if let Err(e) = r {
                        if var.is_default() {
                            force(None);
                            return Err(format!("{}: {e}", var.name()));
                        }
                        differ[vi] = None;
                        break;
                    }
                    // The check: the warm-up run's last routing against the op's default.
                    if i < nv && si == sets.len() - 1 {
                        let got: Vec<u8> = outs.iter().flat_map(|o| o.bytes()).collect();
                        hashes[vi] = hash(&got);
                        if var.is_default() {
                            reference[gi] = got;
                        } else {
                            differ[vi] = Some(
                                got.iter()
                                    .zip(&reference[gi])
                                    .filter(|(a, b)| a != b)
                                    .count(),
                            );
                        }
                    }
                }
                if i >= nv && differ[vi].is_some() {
                    best[vi] = best[vi].min(total / (sets.len() * cfg.layers) as f64);
                }
            }
            force(None);
            let pairs = (cfg.n_used * n_tok) as f64;
            let per = (cfg.n_used * n_tok / (cfg.n_used * n_tok).clamp(1, n_expert)) as i64;
            for (vi, var) in variants.iter().enumerate() {
                let default = &variants[if var.op == Op::Glu { 0 } else { down0 }];
                let mults = if var.op == Op::Glu { 2.0 } else { 1.0 };
                rows.push(Row {
                    variant: var.name(),
                    op: var.op,
                    n_tok,
                    pool,
                    secs: best[vi],
                    flops: mults * 2.0 * (k * m) as f64 * pairs,
                    differ: differ[vi],
                    bitwise: var.runs_mv(per) == default.runs_mv(per),
                    hash: hashes[vi],
                });
            }
        }
    }
    Ok(rows)
}
