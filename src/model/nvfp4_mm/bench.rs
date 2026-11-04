//! The decode expert FFN of NVFP4 checkpoints in isolation (#836): Qwen3.8-Flash-Next's
//! routed experts (gate and up `[2560, 640]`, down `[640, 2560]`, 512 experts, 10 per token) at
//! decode batches on one GPU, run as the graph's unfused chain (two grouped mvs, ggml's silu,
//! mul and cast, the down mv) and as the grouped SwiGLU mv plus the down mv on its plan, each in
//! the grouped mv kernel's variants. Checks that every variant gives the chain's bits and times
//! them (the weight bytes of the experts used over the time: the stream's bandwidth). Used by
//! the hidden `legion nvfp4-mv-bench` subcommand (a build tree's binary on the GPU box).

use super::{
    GroupedMv, force_glu_mv, force_grouped_mv, force_grouped_tiles, mul_mat_id, set_expert_pdl,
    set_expert_tile_prefetch, swiglu_id,
};
use crate::ggml::{Backend, ComputeAllocator, Context, ExternalOp, Graph, Tensor, ffi, types};

/// How a variant builds the FFN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    /// up, gate (grouped mvs), silu, mul, the down mv on the F32 product (its BF16 cast).
    Chain,
    /// The grouped SwiGLU mv (BF16 out), the down mv reading its plan.
    Glu,
}

/// One variant: the FFN and the grouped mv variants it forces (`None`: the default, as served).
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub ffn: Ffn,
    pub glu: Option<GroupedMv>,
    pub mv: Option<GroupedMv>,
    /// The down mv on the SwiGLU's plan as a programmatic dependent launch ([`set_expert_pdl`]).
    pub pdl: bool,
    /// The defaults' whole-tile L2 prefetch ([`set_expert_tile_prefetch`]).
    pub tile: bool,
    /// The SwiGLU mv's and its down mv's least token tile, `8 * nt` tokens (0: the default's).
    pub nt: i32,
    /// Grids of the routing's own token tiles (the most of any layer of a routing set) instead
    /// of the host's upper bound, so no CTA only finds out it has no tile: what those cost.
    pub exact: bool,
}

const fn g(rtw: i32, rpt: i32, sb: i32, pft: i32) -> Option<GroupedMv> {
    Some(GroupedMv { rtw, rpt, sb, pft })
}

const fn v(name: &'static str, ffn: Ffn, glu: Option<GroupedMv>, mv: Option<GroupedMv>) -> Variant {
    Variant {
        name,
        ffn,
        glu,
        mv,
        pdl: true,
        tile: true,
        nt: 0,
        exact: false,
    }
}

const fn no_pdl(v: Variant) -> Variant {
    Variant { pdl: false, ..v }
}

const fn tile_nt(nt: i32, v: Variant) -> Variant {
    Variant { nt, ..v }
}

const fn exact(v: Variant) -> Variant {
    Variant { exact: true, ..v }
}

const fn no_tile(v: Variant) -> Variant {
    Variant { tile: false, ..v }
}

/// The variants, the reference (the default chain) first; a variant that forces one of the
/// SwiGLU mv and the down mv leaves the other at its default. The non-default row tiles and
/// stages are compiled for one token tile (`nt` 1: up to 8 pairs per expert). "t": whole-tile
/// L2 prefetch (`pft` tiles ahead); "d": the down mv; "nopdl": the down mv a plain launch;
/// "notile": the per-row prefetch defaults ("glu notile nopdl": the path before #836 part 8).
/// "nt2", "nt4": the defaults at token tiles of at least 16 or 32 tokens; "exact": grids of the
/// routing's own token tiles.
pub const VARIANTS: [Variant; 21] = [
    v("chain", Ffn::Chain, None, None),
    v("chain p4", Ffn::Chain, None, g(1, 4, 16, 0)),
    v("glu", Ffn::Glu, None, None),
    no_pdl(v("glu nopdl", Ffn::Glu, None, None)),
    no_tile(v("glu notile", Ffn::Glu, None, None)),
    no_tile(no_pdl(v("glu notile nopdl", Ffn::Glu, None, None))),
    v("glu p8", Ffn::Glu, g(1, 8, 16, 0), None),
    v("glu p16", Ffn::Glu, g(1, 16, 16, 0), None),
    v("glu s40", Ffn::Glu, g(1, 8, 40, 0), None),
    v("glu t1", Ffn::Glu, g(1, 8, 16, 1), None),
    v("glu t2", Ffn::Glu, g(1, 8, 16, 2), None),
    v("glu d p8", Ffn::Glu, None, g(1, 8, 16, 0)),
    v("glu d p16", Ffn::Glu, None, g(1, 16, 16, 0)),
    v("glu d t1", Ffn::Glu, None, g(1, 8, 16, 1)),
    v("glu t1 p16", Ffn::Glu, g(1, 16, 16, 1), None),
    v("glu tt2", Ffn::Glu, g(1, 8, 16, 2), g(1, 8, 16, 2)),
    tile_nt(2, v("glu nt2", Ffn::Glu, None, None)),
    tile_nt(4, v("glu nt4", Ffn::Glu, None, None)),
    exact(v("glu exact", Ffn::Glu, None, None)),
    exact(tile_nt(2, v("glu nt2 exact", Ffn::Glu, None, None))),
    exact(tile_nt(4, v("glu nt4 exact", Ffn::Glu, None, None))),
];

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub k: usize,
    pub m: usize,
    pub n_expert: usize,
    pub n_used: usize,
    /// Layers a timed graph holds: the same weights, routed differently (each layer reads
    /// other experts than the last, as in the model).
    pub layers: usize,
    /// Tokens per step.
    pub toks: Vec<usize>,
    /// Each layer's tokens pick their experts from this many of them (a served batch uses
    /// ~90 of 512 per layer at 16 sequences).
    pub pools: Vec<usize>,
    /// Routings per round.
    pub sets: usize,
    pub rounds: usize,
    /// Copies of the experts' weights, layer `l` reading copy `l % copies`: 48 spreads the
    /// routed reads over a served model's expert memory (Flash-Next: 68 GB) instead of 1.4 GB.
    pub copies: usize,
    /// Also time the read probe: the same experts' bytes read as a pure stream (no math), in
    /// chunks of [`PROBE_CHUNKS`] bytes per CTA: the ceiling of the mv kernels' access.
    pub probe: bool,
    /// Only these variants (by name; the reference always runs). Empty: all.
    pub only: Vec<String>,
    /// Served routings ([`recorded_routings`]): each step size's layers take these, in order,
    /// instead of random pools.
    pub routing: Vec<Vec<i32>>,
}

/// The routings of a `legion serve --moe-route-record` file (one record per layer and step:
/// layer, experts, ids per token, tokens as u32 LE, then the ids as i32 LE) of layers below
/// `layers` (the MTP head's records are past the model's layers) and `n_used` ids per token, in
/// file order. A truncated last record is dropped.
pub fn recorded_routings(bytes: &[u8], layers: usize, n_used: usize) -> Vec<Vec<i32>> {
    let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    let mut at = 0;
    while at + 16 <= bytes.len() {
        let (il, k, n_tok) = (u(at), u(at + 8), u(at + 12));
        let len = k * n_tok * 4;
        if at + 16 + len > bytes.len() {
            break;
        }
        if il < layers && k == n_used && n_tok > 0 {
            out.push(
                bytes[at + 16..at + 16 + len]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect(),
            );
        }
        at += 16 + len;
    }
    out
}

/// Up to `sets` routing sets of `layers` routings of `n_tok` tokens each from `recorded`, in
/// order (none when there are fewer than `layers` such routings).
fn recorded_sets(
    recorded: &[Vec<i32>],
    n_used: usize,
    n_tok: usize,
    layers: usize,
    sets: usize,
) -> Vec<Vec<Vec<i32>>> {
    let of_size: Vec<&Vec<i32>> = recorded
        .iter()
        .filter(|ids| ids.len() == n_used * n_tok)
        .collect();
    of_size
        .chunks_exact(layers.max(1))
        .take(sets)
        .map(|c| c.iter().map(|&ids| ids.clone()).collect())
        .collect()
}

/// The token tiles of `bn` pairs a grouped launch over the routing `ids` has: each expert's
/// pairs in tiles of their own.
pub fn routing_tiles(ids: &[i32], bn: usize) -> usize {
    let mut count = std::collections::HashMap::new();
    for &e in ids {
        *count.entry(e).or_insert(0usize) += 1;
    }
    count.values().map(|c| c.div_ceil(bn)).sum()
}

/// The read probe's bytes per CTA, and whether it reads the routed experts or as many
/// adjacent ones (experts 0, 1, ...: long contiguous runs, the ceiling without scatter).
pub const PROBE_CHUNKS: [(&str, i32, bool); 3] = [
    ("probe 16K", 16 << 10, false),
    ("probe 64K", 64 << 10, false),
    ("probe seq", 64 << 10, true),
];

impl Default for Config {
    fn default() -> Self {
        Self {
            k: 2560,
            m: 640,
            n_expert: 512,
            n_used: 10,
            layers: 8,
            toks: vec![16],
            pools: vec![128, 512],
            sets: 4,
            rounds: 5,
            copies: 1,
            probe: false,
            only: Vec::new(),
            routing: Vec::new(),
        }
    }
}

/// One variant at one step.
#[derive(Clone, Debug)]
pub struct Row {
    pub variant: &'static str,
    pub n_tok: usize,
    pub pool: usize,
    /// Experts used per layer (mean).
    pub experts: f64,
    /// Best mean time per layer's FFN over the rounds (seconds).
    pub secs: f64,
    /// Weight bytes per layer's FFN (the experts used, gate, up and down).
    pub bytes: f64,
    /// Output values whose bits differ from the chain's; `None` where the variant has no kernel.
    pub differ: Option<usize>,
    /// Token tiles per layer (mean) at 8, 16 and 32 tokens per tile ([`routing_tiles`]).
    pub tiles: [f64; 3],
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

/// The graph of `layers` expert FFNs of `ffn` on the weights and scales `w` (gate, up, down;
/// layer `l` on copy `l % w.len()`), each layer with its own routing: the input, the routings,
/// the graph and the outputs.
#[allow(clippy::type_complexity)]
fn build<'a>(
    ctx: &'a Context,
    ffn: Ffn,
    w: &[[Tensor<'a>; 6]],
    k: usize,
    n_used: usize,
    n_tok: usize,
    layers: usize,
) -> (Tensor<'a>, Vec<Tensor<'a>>, Graph<'a>, Vec<Tensor<'a>>) {
    let x = ctx
        .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
        .as_input();
    let ids: Vec<_> = (0..layers)
        .map(|_| {
            ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                .as_input()
        })
        .collect();
    let mut graph = ctx.new_graph(32 * layers + 16);
    let outs = ids
        .iter()
        .enumerate()
        .map(|(l, &id)| {
            let [wg, wu, wd, sg, su, sd] = w[l % w.len()];
            let act = match ffn {
                Ffn::Chain => {
                    let up = mul_mat_id(ctx, wu, x, id, Some(su));
                    let gate = mul_mat_id(ctx, wg, x, id, Some(sg));
                    ctx.mul(ctx.silu(gate), up)
                }
                Ffn::Glu => swiglu_id(ctx, wg, wu, x, id, Some(sg), Some(su)),
            };
            let y = mul_mat_id(ctx, wd, act, id, Some(sd)).as_output();
            graph.expand(y);
            y
        })
        .collect();
    (x, ids, graph, outs)
}

fn set_variant(v: Option<&Variant>) {
    force_glu_mv(v.and_then(|v| v.glu));
    force_grouped_mv(v.and_then(|v| v.mv));
    set_expert_pdl(v.is_none_or(|v| v.pdl));
    set_expert_tile_prefetch(v.is_none_or(|v| v.tile));
    force_grouped_tiles(v.and_then(|v| (v.nt > 0).then_some(v.nt)), None);
}

/// [`Variant::exact`]'s grid for the routing set `set`: the most token tiles of any of its
/// layers at the variant's tile (`8 * nt` tokens).
fn exact_tiles(v: &Variant, set: &[Vec<i32>]) -> Option<i32> {
    let bn = 8 * v.nt.max(1) as usize;
    v.exact.then(|| {
        set.iter()
            .map(|ids| routing_tiles(ids, bn))
            .max()
            .unwrap_or(1) as i32
    })
}

/// The indices of the [`VARIANTS`] named in `only` (all of them when it is empty), the
/// reference (index 0, which the others are checked against) always first.
pub fn picked_variants(only: &[String]) -> Vec<usize> {
    (0..VARIANTS.len())
        .filter(|&i| i == 0 || only.is_empty() || only.iter().any(|n| n == VARIANTS[i].name))
        .collect()
}

/// Every variant at every step size and pool of `cfg` on `gpu`; the variants take turns within
/// a round, the best round counts. The kernel choice is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let (k, m, n_expert) = (cfg.k, cfg.m, cfg.n_expert);
    let blocks = |k: usize, m: usize| k / 64 * 36 * m;
    let copies = cfg.copies.max(1);
    let weights_ctx = Context::new(8 * copies, 0);
    let shape = |a: usize, b: usize| [a as i64, b as i64, n_expert as i64];
    let w: Vec<[Tensor<'_>; 6]> = (0..copies)
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
    // Every copy holds the same values (the layers' bits do not depend on the copy).
    for i in 0..3 {
        let bytes = weights(blocks(k, m) * n_expert, 3 + i as u64);
        for c in &w {
            c[i].set_bytes(&bytes);
        }
    }
    for i in 0..3 {
        let mut rng = Rng(17 + i as u64);
        let v: Vec<f32> = (0..n_expert)
            .map(|_| 0.002 + (rng.next() % 1000) as f32 * 1e-5)
            .collect();
        for c in &w {
            c[3 + i].set_f32(&v);
        }
    }
    let expert_bytes = (2 * blocks(k, m) + blocks(m, k)) as f64;
    let mut rows = Vec::new();
    // Served routings: one pass per step size (pool 0).
    let pools = if cfg.routing.is_empty() {
        cfg.pools.clone()
    } else {
        vec![0]
    };
    for &n_tok in &cfg.toks {
        for &pool in &pools {
            let sets: Vec<Vec<Vec<i32>>> = if cfg.routing.is_empty() {
                (0..cfg.sets as u64)
                    .map(|i| {
                        (0..cfg.layers as u64)
                            .map(|l| routing(cfg.n_used, n_tok, n_expert, pool, 1 + 64 * i + l))
                            .collect()
                    })
                    .collect()
            } else {
                recorded_sets(&cfg.routing, cfg.n_used, n_tok, cfg.layers, cfg.sets)
            };
            if sets.is_empty() {
                continue;
            }
            let tiles = [8, 16, 32].map(|bn| {
                sets.iter()
                    .flatten()
                    .map(|ids| routing_tiles(ids, bn) as f64)
                    .sum::<f64>()
                    / (sets.len() * cfg.layers) as f64
            });
            let used: f64 = sets
                .iter()
                .flatten()
                .map(|ids| ids.iter().collect::<std::collections::HashSet<_>>().len() as f64)
                .sum::<f64>()
                / (sets.len() * cfg.layers) as f64;
            let ctxs = [Ffn::Chain, Ffn::Glu]
                .map(|_| Context::new(8 + 24 * cfg.layers, 32 * cfg.layers + 16));
            let graphs: Vec<_> = [Ffn::Chain, Ffn::Glu]
                .into_iter()
                .zip(&ctxs)
                .map(|(ffn, ctx)| build(ctx, ffn, &w, k, cfg.n_used, n_tok, cfg.layers))
                .collect();
            for (ctx, (x, _, graph, _)) in ctxs.iter().zip(&graphs) {
                ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                let mut rng = Rng(5 + n_tok as u64);
                let xs: Vec<f32> = (0..k * n_tok)
                    .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0)
                    .collect();
                x.set_f32(&xs);
            }
            // A warm-up run of each variant (and its check), then the rounds.
            let picked = picked_variants(&cfg.only);
            let order: Vec<usize> = picked
                .iter()
                .copied()
                .chain((0..cfg.rounds).flat_map(|_| picked.iter().copied()))
                .collect();
            let warm = picked.len();
            let mut best = [f64::MAX; VARIANTS.len()];
            let mut differ: [Option<usize>; VARIANTS.len()] = [Some(0); VARIANTS.len()];
            let mut reference: Vec<f32> = Vec::new();
            for (i, &vi) in order.iter().enumerate() {
                let var = &VARIANTS[vi];
                if differ[vi].is_none() {
                    continue;
                }
                let (_, ids, graph, outs) = &graphs[(var.ffn == Ffn::Glu) as usize];
                set_variant(Some(var));
                let mut total = 0.0;
                for (si, set) in sets.iter().enumerate() {
                    for (it, ids) in ids.iter().zip(set) {
                        it.set_i32(ids);
                    }
                    if var.exact {
                        force_grouped_tiles((var.nt > 0).then_some(var.nt), exact_tiles(var, set));
                    }
                    let t = std::time::Instant::now();
                    let r = gpu.compute(graph);
                    total += t.elapsed().as_secs_f64();
                    if let Err(e) = r {
                        if vi == 0 {
                            set_variant(None);
                            return Err(format!("{}: {e}", var.name));
                        }
                        differ[vi] = None;
                        break;
                    }
                    // The check: the warm-up run's last routing against the chain's.
                    if i < warm && si == sets.len() - 1 {
                        let got: Vec<f32> = outs.iter().flat_map(|o| o.to_f32()).collect();
                        if vi == 0 {
                            reference = got;
                        } else {
                            differ[vi] = Some(
                                got.iter()
                                    .zip(&reference)
                                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                                    .count(),
                            );
                        }
                    }
                }
                if i >= warm && differ[vi].is_some() {
                    best[vi] = best[vi].min(total / (sets.len() * cfg.layers) as f64);
                }
            }
            set_variant(None);
            for vi in picked {
                let var = &VARIANTS[vi];
                rows.push(Row {
                    variant: var.name,
                    n_tok,
                    pool,
                    experts: used,
                    secs: best[vi],
                    bytes: used * expert_bytes,
                    differ: differ[vi],
                    tiles,
                });
            }
            if cfg.probe {
                for (name, chunk, seq) in PROBE_CHUNKS {
                    let secs = probe(gpu, &w, &sets, chunk, seq, cfg.rounds)?;
                    rows.push(Row {
                        variant: name,
                        n_tok,
                        pool,
                        experts: used,
                        secs,
                        bytes: used * expert_bytes,
                        differ: Some(0),
                        tiles,
                    });
                }
            }
        }
    }
    Ok(rows)
}

static READ_PROBE: ExternalOp = ExternalOp {
    name: "nvfp4_read_probe",
    cpu: cpu_read_probe,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_read_probe),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The probe has no CPU side: it only measures a GPU's reads.
unsafe fn cpu_read_probe(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn legion_nvfp4_read_probe_launch(
        w0: *const u8,
        w1: *const u8,
        w2: *const u8,
        w_e: i64,
        ranges: *const i32,
        n: i32,
        chunk: i64,
        sink: *mut u32,
        stream: *mut std::os::raw::c_void,
    ) -> i32;
}

/// The read probe node: src 0 the (weight, expert) pairs, 1-3 the gate, up and down experts.
#[cfg(feature = "cuda")]
unsafe fn cuda_read_probe(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `probe`, all tensors in device memory of the stream's device.
    let (ranges, wg, wu, wd) = unsafe { (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]) };
    let chunk = crate::ggml::external_params(dst)[0] as i64;
    // SAFETY: device pointers of the node's inputs; the node's stream.
    let code = unsafe {
        legion_nvfp4_read_probe_launch(
            wg.data as *const u8,
            wu.data as *const u8,
            wd.data as *const u8,
            wg.nb[2] as i64,
            ranges.data as *const i32,
            (ranges.ne[1]) as i32,
            chunk,
            dst.data as *mut u32,
            stream,
        )
    };
    match code {
        0 => Ok(()),
        e => Err(format!("nvfp4 read probe: error {e}")),
    }
}

/// The read probe over the experts each layer of each routing set reads (gate, up and down of
/// each expert used, whole), `chunk` bytes per CTA: the best mean time per layer over `rounds`.
/// `seq`: as many adjacent experts from 0 instead of the routed ones.
fn probe(
    gpu: &mut Backend,
    w: &[[Tensor<'_>; 6]],
    sets: &[Vec<Vec<i32>>],
    chunk: i32,
    seq: bool,
    rounds: usize,
) -> Result<f64, String> {
    let layers = sets[0].len();
    let ctxs: Vec<Context> = sets
        .iter()
        .map(|_| Context::new(2 * layers + 8, 4 * layers + 16))
        .collect();
    let graphs: Vec<_> = sets
        .iter()
        .zip(&ctxs)
        .map(|(set, ctx)| {
            let mut graph = ctx.new_graph(4 * layers + 16);
            let mut inputs = Vec::new();
            for (l, ids) in set.iter().enumerate() {
                let mut experts: Vec<i32> = ids.clone();
                experts.sort_unstable();
                experts.dedup();
                if seq {
                    experts = (0..experts.len() as i32).collect();
                }
                let pairs: Vec<i32> = (0..3)
                    .flat_map(|t| experts.iter().flat_map(move |&e| [t, e]))
                    .collect();
                let r = ctx
                    .new_tensor(types::I32, &[2, pairs.len() as i64 / 2])
                    .as_input();
                let [wg, wu, wd, ..] = w[l % w.len()];
                let y = ctx
                    .external_with(
                        &READ_PROBE,
                        types::F32,
                        [1, 1, 1, 1],
                        &[r, wg, wu, wd],
                        &[chunk],
                    )
                    .as_output();
                graph.expand(y);
                inputs.push((r, pairs));
            }
            (graph, inputs)
        })
        .collect();
    for (ctx, (graph, inputs)) in ctxs.iter().zip(&graphs) {
        ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        for (r, pairs) in inputs {
            r.set_i32(pairs);
        }
    }
    let mut best = f64::MAX;
    for round in 0..=rounds {
        let mut total = 0.0;
        for (graph, _) in &graphs {
            let t = std::time::Instant::now();
            gpu.compute(graph).map_err(|e| e.to_string())?;
            total += t.elapsed().as_secs_f64();
        }
        // Round 0 warms up.
        if round > 0 {
            best = best.min(total / (graphs.len() * layers) as f64);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests;
