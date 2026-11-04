//! The NVFP4 expert FFN of a verify step in isolation (#836): Qwen3.8-Flash-Next's routed experts
//! (gate and up `[2560, 640]`, down `[640, 2560]`, 10 per token) at 2-8 tokens on one GPU, each
//! matmul with its per-expert scale (`reshape`, `repeat`, `get_rows`, `mul`, as
//! [`crate::model::core`] builds it), once on ggml's `mul_mat_id` (quantize_q8_1 + MMVQ, the scale
//! chain fused into it where ggml-cuda can, then `silu` and `mul`) and once on the ops (gate, up
//! and the SwiGLU in [`swiglu_id`], the down matmul in [`mul_mat_id`], the scales in both) at
//! their default tiles and in each of their [`TILES`]. Checks that every variant gives ggml's
//! bits and times them per layer. The tokens pick their experts from
//! a pool: a pool of 10 is a step whose tokens all pick the same experts, a big pool one whose
//! tokens share few. Used by the hidden `legion nvfp4-once-bench` subcommand (a build tree's binary
//! on the GPU box).

use super::{
    Family, Hybrid, TILES, Warp, force_family, force_iters, force_matmul_family, force_own_plan,
    force_tile, mul_mat_id, swiglu_id,
};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};

/// Shapes and runs.
#[derive(Clone, Debug)]
pub struct Config {
    /// The model's width (gate and up: `[k, m]`, down: `[m, k]`).
    pub k: usize,
    pub m: usize,
    /// Experts per layer stack (each layer its own stack, so no layer finds another's weights in
    /// the L2 cache).
    pub n_expert: usize,
    pub n_used: usize,
    pub layers: usize,
    /// Tokens per step.
    pub toks: Vec<usize>,
    /// Each step's tokens pick their experts from this many of them.
    pub pools: Vec<usize>,
    /// Routings per round.
    pub sets: usize,
    pub rounds: usize,
    /// Each matmul with its per-expert scale chain (as served); without it, the matmuls alone.
    pub scale: bool,
    /// Fill the input once, after the graph's allocation, instead of before every run (the
    /// bench's old behaviour, kept to show the stale read it caused): ggml's allocator frees an
    /// input after its last consumer like any other tensor, and with the scale chain the last
    /// layer's gate and down scale `repeat` and `get_rows` nodes are placed over the start of
    /// it (1-4 KiB at 2-8 tokens, inside token 0's 10 KiB), so every run after the first read
    /// a token 0 that was not the one set (#836).
    pub input_once: bool,
    /// The ids from the model's router top-k over logits set to give each routing (in both
    /// graphs, as served; the ops' graph with the routing plan written by the top-k kernel,
    /// #836), instead of set as an input.
    pub router: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k: 2560,
            m: 640,
            n_expert: 128,
            n_used: 10,
            layers: 8,
            toks: vec![2, 4, 6, 8],
            pools: vec![10, 16, 24, 128],
            sets: 4,
            rounds: 5,
            scale: true,
            input_once: false,
            router: true,
        }
    }
}

/// One variant at one step size and pool.
#[derive(Clone, Debug)]
pub struct Row {
    /// `ggml` or the op's tile (`once 4x2`: 4 warps of 2 rows).
    pub variant: String,
    pub n_tok: usize,
    pub pool: usize,
    /// Distinct experts per layer (mean).
    pub experts: f64,
    /// Best mean time per layer's FFN over the rounds (seconds).
    pub secs: f64,
    /// Output values whose bits differ from ggml's.
    pub differ: usize,
    /// Output values compared.
    pub outputs: usize,
    /// Layer 0, token 0, slot 0, the first [`REF_ROWS`] outputs of the first round's first
    /// routing against an F64 host reference of the same FFN: the largest difference over the
    /// largest reference value (the GPU quantizes the activations to Q8_1, so about 1e-2 is
    /// expected; a stale input is off by the whole value).
    pub ref_err: f64,
    /// ggml only: the same for its warm-up run (the first run after the allocation).
    pub warm_ref_err: Option<f64>,
    /// The graph's nodes placed over (part of) the input `x` by ggml's allocator.
    pub over_input: Vec<String>,
}

/// Outputs checked against the host reference (see [`Row::ref_err`]).
pub const REF_ROWS: usize = 8;

/// The nodes of the allocated `graph` whose memory overlaps `t`'s (views of `t` aside): ggml's
/// allocator frees an input after its last consumer, so a later node may be placed over it.
fn placed_over(graph: &Graph<'_>, t: Tensor<'_>) -> Vec<String> {
    let raw = t.raw();
    // SAFETY: an allocated graph; every node and `t` are live tensors with their data set.
    unsafe {
        let start = (*raw).data as usize;
        let end = start + t.nbytes();
        (0..ffi::ggml_graph_n_nodes(graph.raw()))
            .map(|i| ffi::ggml_graph_node(graph.raw(), i))
            .filter(|&n| n != raw && (*n).view_src != raw && !(*n).data.is_null())
            .filter(|&n| {
                let s = (*n).data as usize;
                s < end && start < s + ffi::ggml_nbytes(n)
            })
            .map(|n| {
                let op = std::ffi::CStr::from_ptr(ffi::ggml_op_desc(n)).to_string_lossy();
                let name = std::ffi::CStr::from_ptr((*n).name.as_ptr()).to_string_lossy();
                let at = (*n).data as isize - start as isize;
                format!("{op} '{name}' at {at:+} B, {} B", ffi::ggml_nbytes(n))
            })
            .collect()
    }
}

/// The F64 host reference of one token's FFN through one expert (layer 0's weights): the first
/// `rows` outputs of `sd[e] * down_e(silu(sg[e] * gate_e(x)) * su[e] * up_e(x))`, the weights
/// dequantized by ggml's own NVFP4 decoder.
fn host_reference(
    w: &[Vec<u8>; 3],
    s: &[Vec<f32>; 3],
    (k, m): (usize, usize),
    e: usize,
    x: &[f32],
    rows: usize,
) -> Vec<f64> {
    let row = |i: usize, r: usize, n: usize, n_rows: usize| {
        let bytes = n / 64 * 36;
        let off = (e * n_rows + r) * bytes;
        crate::ggml::dequantize(types::NVFP4, &w[i][off..off + bytes], n)
    };
    let dot = |a: &[f32], b: &[f64]| a.iter().zip(b).map(|(&a, &b)| a as f64 * b).sum::<f64>();
    let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let act: Vec<f64> = (0..m)
        .map(|j| {
            let g = s[0][e] as f64 * dot(&row(0, j, k, m)[..], &x);
            let u = s[1][e] as f64 * dot(&row(1, j, k, m)[..], &x);
            g / (1.0 + (-g).exp()) * u
        })
        .collect();
    (0..rows)
        .map(|r| s[2][e] as f64 * dot(&row(2, r, m, k)[..], &act))
        .collect()
}

/// The largest difference of `got` from `want` over the largest `want`.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let top = want
        .iter()
        .fold(0f64, |a, &b| a.max(b.abs()))
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .fold(0f64, |a, (&g, &w)| a.max((g as f64 - w).abs()))
        / top
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Random NVFP4 blocks: sub-block scales of moderate size, codes uniform.
pub(crate) fn nvfp4_bytes(bytes: usize, seed: u64) -> Vec<u8> {
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
pub(crate) fn routing(
    n_used: usize,
    n_tok: usize,
    n_expert: usize,
    pool: usize,
    seed: u64,
) -> Vec<i32> {
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

/// One expert matmul with its per-expert scale, as the model builds it on ggml.
fn expert<'a>(
    ctx: &'a Context,
    scale: bool,
    w: Tensor<'a>,
    s: Tensor<'a>,
    x: Tensor<'a>,
    ids: Tensor<'a>,
) -> Tensor<'a> {
    let res = ctx.mul_mat_id(w, x, ids);
    if !scale {
        return res;
    }
    let n_expert = s.ne()[0];
    let n_tokens = x.ne()[2];
    let s = ctx.reshape_3d(s, 1, n_expert, 1);
    let s = ctx.repeat_4d(s, [1, n_expert, n_tokens, 1]);
    ctx.mul(res, ctx.get_rows(s, ids))
}

/// The graph of `layers` expert FFNs (layer `l` on weights `w[l]`: gate, up, down and their
/// scales), each layer with its own routing: the input, the routings (the ids, or with `router`
/// the router logits `[n_expert, n_tok]` the ids are picked from), the graph and the outputs.
#[allow(clippy::type_complexity)]
fn build<'a>(
    ctx: &'a Context,
    (once, scale, router): (bool, bool, bool),
    w: &[[Tensor<'a>; 6]],
    k: usize,
    n_used: usize,
    n_tok: usize,
) -> (Tensor<'a>, Vec<Tensor<'a>>, Graph<'a>, Vec<Tensor<'a>>) {
    let x = ctx
        .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
        .as_input();
    let routes: Vec<_> = w
        .iter()
        .map(|lw| {
            if router {
                ctx.new_tensor(types::F32, &[lw[0].ne()[2], n_tok as i64])
            } else {
                ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
            }
            .as_input()
        })
        .collect();
    let mut graph = ctx.new_graph(32 * w.len() + 16);
    let outs = routes
        .iter()
        .zip(w)
        .map(|(&route, &[wg, wu, wd, sg, su, sd])| {
            let id = if router {
                crate::model::core::fused::router_planned(ctx, route, n_used as i64, once, false).0
            } else {
                route
            };
            let y = if once {
                let s = |t: Tensor<'a>| scale.then_some(t);
                let act = swiglu_id(ctx, (wg, s(sg)), (wu, s(su)), x, id);
                mul_mat_id(ctx, wd, act, id, s(sd))
            } else {
                let up = expert(ctx, scale, wu, su, x, id);
                let gate = expert(ctx, scale, wg, sg, x, id);
                let act = ctx.mul(ctx.silu(gate), up);
                expert(ctx, scale, wd, sd, act, id)
            }
            .as_output();
            graph.expand(y);
            y
        })
        .collect();
    (x, routes, graph, outs)
}

/// A variant's forced choices: the tile, the K steps, the kernel family ([`force_family`]) and
/// the down projection's own ([`force_matmul_family`]), then whether the ops build their
/// own routing plan ([`force_own_plan`]).
#[derive(Clone, Copy, Default)]
struct Forced {
    tile: Option<usize>,
    iters: Option<usize>,
    family: Option<Family>,
    matmul: Option<Family>,
    own_plan: bool,
}

const fn families(family: Option<Family>, matmul: Option<Family>) -> Forced {
    Forced {
        tile: None,
        iters: None,
        family,
        matmul,
        own_plan: false,
    }
}

const fn warp(users: u32, pre: bool, occ7: bool) -> Family {
    Family::Warp(Warp { users, pre, occ7 })
}

const fn hybrid(umax: u32, occ: u32) -> Family {
    Family::Hybrid(Hybrid { umax, occ })
}

/// The name of a family for the bench.
fn family_name(f: Family) -> String {
    match f {
        Family::Tiles => "tiles".to_string(),
        Family::Warp(w) => w.name(),
        Family::Hybrid(h) => h.name(),
    }
}

/// The kernel families the bench times on both ops: the best of #836's warp kernels at 2-4
/// tokens (one pair a pass, decoded once, 7 CTAs an SM) and at 5-8 (two pairs, 7 CTAs), and
/// the hybrids.
const BENCH_FAMILIES: [Family; 5] = [
    warp(1, true, true),
    warp(2, false, true),
    hybrid(2, 4),
    hybrid(4, 4),
    hybrid(4, 6),
];

/// The (SwiGLU, down) family pairs the bench times: the hybrids with more registers on the
/// SwiGLU and, on the short down rows (one K step), kernels for more warps an SM.
const BENCH_PAIRS: [(Family, Family); 6] = [
    (hybrid(4, 4), hybrid(2, 8)),
    (hybrid(4, 4), hybrid(2, 6)),
    (hybrid(4, 4), warp(1, false, false)),
    (hybrid(4, 4), warp(2, false, false)),
    (hybrid(2, 4), hybrid(2, 8)),
    (hybrid(4, 6), hybrid(2, 8)),
];

/// The variants: ggml, the ops at their defaults, in the 2-row tiles, on [`BENCH_FAMILIES`],
/// then on [`BENCH_PAIRS`]: each name with its forced choices.
fn variants() -> Vec<(String, Forced)> {
    let def = Forced::default();
    let mut v = vec![
        ("ggml".to_string(), def),
        ("once def".to_string(), def),
        // The defaults with the plan built by the ops' own quantize launch (the extra CTA,
        // #1035) where the router wrote one.
        (
            "once own plan".to_string(),
            Forced {
                own_plan: true,
                ..def
            },
        ),
        (
            format!("once r{}", TILES[1].rows),
            Forced {
                tile: Some(1),
                ..families(Some(Family::Tiles), None)
            },
        ),
    ];
    v.extend(
        BENCH_FAMILIES
            .iter()
            .map(|&f| (family_name(f), families(Some(f), None))),
    );
    v.extend(BENCH_PAIRS.iter().map(|&(g, d)| {
        (
            format!("{}/{}", family_name(g), family_name(d)),
            families(Some(g), Some(d)),
        )
    }));
    v
}

/// Forces a variant's choices; `None` puts the defaults back.
fn force(f: Option<Forced>) {
    let f = f.unwrap_or_default();
    force_tile(f.tile);
    force_iters(f.iters);
    force_family(f.family);
    force_matmul_family(f.matmul);
    force_own_plan(f.own_plan);
}

/// The variants' names, ggml first.
pub fn variant_names() -> Vec<String> {
    variants().into_iter().map(|v| v.0).collect()
}

/// ggml and every tile at every step size and pool of `cfg` on `gpu`; the variants take turns
/// within a round, the best round counts. The tile choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let (k, m, n_expert) = (cfg.k, cfg.m, cfg.n_expert);
    let blocks = |k: usize, m: usize| k / 64 * 36 * m;
    let weights_ctx = Context::new(8 * cfg.layers, 0);
    let shape = |a: usize, b: usize| [a as i64, b as i64, n_expert as i64];
    let w: Vec<[Tensor<'_>; 6]> = (0..cfg.layers)
        .map(|_| {
            [
                weights_ctx.new_tensor(types::NVFP4, &shape(k, m)),
                weights_ctx.new_tensor(types::NVFP4, &shape(k, m)),
                weights_ctx.new_tensor(types::NVFP4, &shape(m, k)),
                weights_ctx.new_tensor(types::F32, &[n_expert as i64]),
                weights_ctx.new_tensor(types::F32, &[n_expert as i64]),
                weights_ctx.new_tensor(types::F32, &[n_expert as i64]),
            ]
        })
        .collect();
    weights_ctx
        .alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    // Layer 0's weights and scales, kept for the host reference.
    let mut w0: [Vec<u8>; 3] = Default::default();
    let mut s0: [Vec<f32>; 3] = Default::default();
    for (l, lw) in w.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = if i < 2 { (k, m) } else { (m, k) };
            let bytes = nvfp4_bytes(blocks(a, b) * n_expert, (3 + 8 * l + i) as u64);
            lw[i].set_bytes(&bytes);
            let mut rng = Rng(17 + (8 * l + i) as u64);
            let v: Vec<f32> = (0..n_expert)
                .map(|_| 0.002 + (rng.next() % 1000) as f32 * 1e-5)
                .collect();
            lw[3 + i].set_f32(&v);
            if l == 0 {
                w0[i] = bytes;
                s0[i] = v;
            }
        }
    }
    let names = variant_names();
    let forced = variants();
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
            let used: f64 = sets
                .iter()
                .flatten()
                .map(|ids| ids.iter().collect::<std::collections::HashSet<_>>().len() as f64)
                .sum::<f64>()
                / (cfg.sets * cfg.layers) as f64;
            let ctxs =
                [false, true].map(|_| Context::new(8 + 32 * cfg.layers, 32 * cfg.layers + 16));
            let graphs: Vec<_> = [false, true]
                .into_iter()
                .zip(&ctxs)
                .map(|(once, ctx)| {
                    let modes = (once, cfg.scale, cfg.router);
                    build(ctx, modes, &w, k, cfg.n_used, n_tok)
                })
                .collect();
            // With the router: per layer, logits that rank each token's experts first in
            // routing order (2.0, 1.99, ...), every other expert below (distinct values).
            let logits = |ids: &[i32]| {
                let mut l = Vec::with_capacity(n_expert * n_tok);
                for t in ids.chunks(cfg.n_used) {
                    let base = l.len();
                    l.extend((0..n_expert).map(|e| -1.0 - 1e-4 * e as f32));
                    for (j, &e) in t.iter().enumerate() {
                        l[base + e as usize] = 2.0 - 0.01 * j as f32;
                    }
                }
                l
            };
            let routes: Vec<Vec<Vec<f32>>> = sets
                .iter()
                .map(|set| set.iter().map(|ids| logits(ids)).collect())
                .collect();
            let mut rng = Rng(5 + n_tok as u64);
            let xs: Vec<f32> = (0..k * n_tok)
                .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0)
                .collect();
            let mut over = Vec::new();
            for (ctx, (x, _, graph, _)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                over.push(placed_over(graph, *x));
                x.set_f32(&xs);
            }
            // Token 0's first expert in layer 0 (first routing): the host reference. Without
            // the scale chain the graph has no scales: the reference neither.
            let e0 = sets[0][0][0] as usize;
            let ones = || vec![1.0; n_expert];
            let scales = if cfg.scale {
                s0.clone()
            } else {
                [ones(), ones(), ones()]
            };
            let want = host_reference(&w0, &scales, (k, m), e0, &xs[..k], REF_ROWS);
            let mut ref_err = vec![f64::NAN; names.len()];
            let mut warm_ref_err = None;
            // A warm-up run of each variant, then the rounds. The check: every routing of the
            // first round against ggml's in that round; ggml's own row counts its outputs that
            // differ between its warm-up run and that round.
            let n_var = names.len();
            let order: Vec<usize> = (0..n_var)
                .chain((0..cfg.rounds.max(1)).flat_map(|_| 0..n_var))
                .collect();
            let mut best = vec![f64::MAX; n_var];
            let mut differ = vec![0usize; n_var];
            // Differing outputs per token of the step (each output's token: its index within
            // a layer's [rows][n_used][n_tok] block).
            let mut by_tok = vec![vec![0usize; n_tok]; n_var];
            // Each variant's first differing output: (index, got, reference).
            let mut first: Vec<Option<(usize, f32, f32)>> = vec![None; n_var];
            let per_tok = k * cfg.n_used;
            let mut warm: Vec<Vec<f32>> = Vec::new();
            let mut reference: Vec<Vec<f32>> = Vec::new();
            let mut outputs = 0;
            for (i, &vi) in order.iter().enumerate() {
                let (x, ids, graph, outs) = &graphs[(vi > 0) as usize];
                force(Some(forced[vi].1));
                let mut total = 0.0;
                for (si, set) in sets.iter().enumerate() {
                    // Every input before every run, as a served step fills its inputs: ggml's
                    // allocator may place a later node over an input after its last use.
                    if !cfg.input_once {
                        x.set_f32(&xs);
                    }
                    for ((it, ids), l) in ids.iter().zip(set).zip(&routes[si]) {
                        if cfg.router {
                            it.set_f32(l);
                        } else {
                            it.set_i32(ids);
                        }
                    }
                    let t = std::time::Instant::now();
                    let r = gpu.compute(graph);
                    total += t.elapsed().as_secs_f64();
                    if let Err(e) = r {
                        force(None);
                        return Err(format!("{}: {e}", names[vi]));
                    }
                    let first_round = (n_var..2 * n_var).contains(&i);
                    if i == 0 || first_round {
                        let got: Vec<f32> = outs.iter().flat_map(|o| o.to_f32()).collect();
                        // Layer 0's output first, token 0 slot 0 its first rows.
                        if si == 0 {
                            let err = rel_err(&got[..REF_ROWS], &want);
                            if i == 0 {
                                warm_ref_err = Some(err);
                            } else {
                                ref_err[vi] = err;
                            }
                        }
                        let count = |a: &[f32], b: &[f32]| {
                            a.iter()
                                .zip(b)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count()
                        };
                        let mut tally = |vi: usize, a: &[f32], b: &[f32]| {
                            for (o, (a, b)) in a.iter().zip(b).enumerate() {
                                if a.to_bits() != b.to_bits() {
                                    by_tok[vi][o / per_tok % n_tok] += 1;
                                    first[vi].get_or_insert((o, *a, *b));
                                }
                            }
                        };
                        if i == 0 {
                            outputs += got.len();
                            warm.push(got);
                        } else if vi == 0 {
                            differ[0] += count(&got, &warm[si]);
                            tally(0, &got, &warm[si]);
                            reference.push(got);
                        } else {
                            differ[vi] += count(&got, &reference[si]);
                            tally(vi, &got, &reference[si]);
                        }
                    }
                }
                if i >= n_var {
                    best[vi] = best[vi].min(total / (sets.len() * cfg.layers) as f64);
                }
            }
            force(None);
            for ((name, t), f) in names.iter().zip(&by_tok).zip(&first) {
                if let Some((o, a, b)) = f {
                    let (layer, t0) = (o / (per_tok * n_tok), o / per_tok % n_tok);
                    let (j, r) = (o / k % cfg.n_used, o % k);
                    eprintln!(
                        "toks {n_tok} pool {pool} {name}: differing outputs by token {t:?}; \
                         first: layer {layer} token {t0} slot {j} row {r}: {a:e} vs {b:e} \
                         ({:#010x} vs {:#010x})",
                        a.to_bits(),
                        b.to_bits()
                    );
                }
            }
            for (vi, ((name, secs), differ)) in names.iter().zip(best).zip(differ).enumerate() {
                rows.push(Row {
                    variant: name.clone(),
                    n_tok,
                    pool,
                    experts: used,
                    secs,
                    differ,
                    outputs,
                    ref_err: ref_err[vi],
                    warm_ref_err: (vi == 0).then_some(warm_ref_err).flatten(),
                    over_input: over[(vi > 0) as usize].clone(),
                });
            }
        }
    }
    Ok(rows)
}
