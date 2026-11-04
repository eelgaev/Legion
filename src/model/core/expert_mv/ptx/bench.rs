//! The PTX kernels in isolation (#703): Qwen3.8-Flash-Next's decode expert ops on one GPU,
//! each run with the nvcc tile kernel at a tile per CTA (`--expert-mv-pipe 0`), the nvcc
//! in-place kernel (`--expert-mv-prefetch`), the PTX kernels (`--expert-mv-ptx`), those on the
//! side stream (`--expert-mv-ptx-side`) and the pipelined tile kernel (`--expert-mv-pipe`, #816).
//! Checks that all give the same bits and that they match a plain f32 reference, and times them.
//! Used by the hidden `legion expert-mv-ptx-bench` subcommand (a build tree's binary on the GPU
//! box) and the ignored GPU test `tests/expert_mv_ptx_cuda.rs`.

use crate::ggml::{Backend, ComputeAllocator, Context, dequantize, quantize, row_size, types};
use crate::model::core::expert_mv;

/// One op shape.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub name: &'static str,
    pub ty: u32,
    pub k: usize,
    pub rows: usize,
    /// Gate and up through the SwiGLU op (one activation column per token); else one matrix
    /// with a column per pair (the down projection).
    pub glu: bool,
}

/// Qwen3.8-Flash-Next's expert ops, then its MTP layer's (Q8_0, the shared MTP GGUF; the PTX
/// kernels do not take Q8_0, so their columns run the nvcc kernels).
pub const CASES: [Case; 4] = [
    Case {
        name: "gate/up Q4_K",
        ty: types::Q4_K,
        k: 2560,
        rows: 640,
        glu: true,
    },
    Case {
        name: "down Q5_1",
        ty: types::Q5_1,
        k: 640,
        rows: 2560,
        glu: false,
    },
    // Its gate and up are two ops (two Q8_0 tiles of 4 rows pass 16 KiB), here with a column per
    // pair as the down projection has.
    Case {
        name: "gate Q8_0",
        ty: types::Q8_0,
        k: 2560,
        rows: 640,
        glu: false,
    },
    Case {
        name: "down Q8_0",
        ty: types::Q8_0,
        k: 640,
        rows: 2560,
        glu: false,
    },
];

/// The kernels compared, the default first.
pub const MODES: [&str; 5] = ["tile", "prefetch", "ptx", "ptx side", "pipe"];

/// Mode `m`; "tile" is the tile kernel with a tile per CTA, "pipe" the tile kernel with `pipe`
/// tiles per CTA (`--expert-mv-pipe`, #816; `None`: the default choice).
fn set_mode(m: usize, pipe: Option<usize>) {
    expert_mv::set_prefetch(m == 1);
    super::set_enabled(m == 2 || m == 3);
    super::set_side(m == 3);
    expert_mv::set_pipe_tiles(if m == 4 { pipe } else { Some(0) });
}

/// One case at one step size.
#[derive(Clone, Debug)]
pub struct Row {
    pub case: &'static str,
    pub n_tok: usize,
    /// Experts below this id are VRAM cache slots (the others take the tile kernel).
    pub vram_below: usize,
    pub n_expert: usize,
    /// Best mean time per op over the rounds, per mode (seconds).
    pub secs: [f64; 5],
    /// Output values whose bits differ from the tile kernel's, per mode.
    pub differ: [usize; 5],
    /// The tile kernel's largest error against the f32 reference, relative to the largest
    /// reference value.
    pub ref_err: f64,
    /// FNV-1a hash of the tile kernel's output bits (the last set), to compare two builds.
    pub bits_hash: u64,
}

/// FNV-1a over the bits of `v`.
pub fn bits_hash(v: &[f32]) -> u64 {
    v.iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Distinct ids per token; later tokens share half of theirs with token 0 (MTP verify steps).
fn ids(n_used: usize, n_tok: usize, n_expert: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let mut out: Vec<i32> = Vec::new();
    for t in 0..n_tok {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < n_used {
            let e = if t > 0 && chosen.len() < n_used / 2 {
                out[chosen.len()]
            } else {
                (rng.next() % n_expert as u64) as i32
            };
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        out.extend(chosen);
    }
    out
}

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub n_expert: usize,
    pub n_used: usize,
    /// Copies of the stacks a timed graph holds (each op reads other memory than the last).
    pub layers: usize,
    pub toks: Vec<usize>,
    /// Id sets per round.
    pub sets: usize,
    pub rounds: usize,
    /// Also each step size with half the experts in cache slots (the rest the tile kernel's).
    pub mixed: bool,
    /// Tiles per CTA of the "pipe" mode (`None`: the default choice).
    pub pipe_tiles: Option<usize>,
    /// Graphs built with the Q4_K per-unit activations ([`expert_mv::set_q4k_units`]).
    pub q4k_units: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_expert: 256,
            n_used: 10,
            layers: 4,
            toks: vec![1, 2, 3, 4],
            sets: 10,
            rounds: 5,
            mixed: true,
            pipe_tiles: None,
            q4k_units: false,
        }
    }
}

/// Every case at every step size of `cfg` on `gpu`; the modes run interleaved, the best round
/// counts. The switches are back at their defaults (all off) afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    expert_mv::set_q4k_units(cfg.q4k_units);
    for c in CASES {
        let stacks: Vec<Vec<u8>> = (0..if c.glu { 2 } else { 1 })
            .map(|m| {
                let mut rng = Rng(7 + m as u64);
                let w: Vec<f32> = (0..c.k * c.rows * cfg.n_expert)
                    .map(|_| rng.f32() * 0.05)
                    .collect();
                quantize(c.ty, &w, c.k)
            })
            .collect();
        let belows = if cfg.mixed {
            vec![cfg.n_expert, cfg.n_expert / 2]
        } else {
            vec![cfg.n_expert]
        };
        for &n_tok in &cfg.toks {
            for &below in &belows {
                let r = case(gpu, cfg, c, &stacks, n_tok, below);
                set_mode(0, None);
                expert_mv::set_pipe_tiles(None);
                rows.push(r?);
            }
        }
    }
    expert_mv::set_q4k_units(false);
    Ok(rows)
}

fn case(
    gpu: &mut Backend,
    cfg: &Config,
    c: Case,
    stacks: &[Vec<u8>],
    n_tok: usize,
    below: usize,
) -> Result<Row, String> {
    let (layers, n_used, n_expert) = (cfg.layers, cfg.n_used, cfg.n_expert);
    let weights = Context::new(2 * layers, 0);
    let shape = [c.k as i64, c.rows as i64, n_expert as i64];
    let ws: Vec<Vec<_>> = (0..layers)
        .map(|_| {
            stacks
                .iter()
                .map(|_| weights.new_tensor(c.ty, &shape))
                .collect()
        })
        .collect();
    weights
        .alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for l in &ws {
        for (w, q) in l.iter().zip(stacks) {
            w.set_bytes(q);
        }
    }
    let n_x = if c.glu { 1 } else { n_used };
    let ctx = Context::new(1 + 8 * layers, 16 * layers + 8);
    let xt = ctx.new_tensor(types::F32, &[c.k as i64, n_x as i64, n_tok as i64]);
    let its: Vec<_> = (0..layers)
        .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
        .collect();
    ctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    let mut rng = Rng(11 + n_tok as u64);
    let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
    xt.set_f32(&x);
    let outs: Vec<_> = ws
        .iter()
        .zip(&its)
        .map(|(l, &it)| {
            if c.glu {
                expert_mv::glu_tiered(&ctx, l[0], l[1], xt, it, below)
            } else {
                expert_mv::mul_mat_id_tiered(&ctx, l[0], xt, it, below)
            }
        })
        .collect();
    let mut graph = ctx.new_graph(16 * layers + 8);
    for o in &outs {
        graph.expand(*o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .map_err(|e| e.to_string())?;
    let sets: Vec<Vec<Vec<i32>>> = (0..cfg.sets as u64)
        .map(|i| {
            (0..layers as u64)
                .map(|l| ids(n_used, n_tok, n_expert, 100 + 16 * i + l))
                .collect()
        })
        .collect();
    // A warm-up run of each mode (the PTX module loads on its first launch), then the rounds.
    let plan: Vec<usize> = (0..MODES.len())
        .chain((0..cfg.rounds).flat_map(|_| 0..MODES.len()))
        .collect();
    let mut best = [f64::MAX; 5];
    let mut last: [Vec<f32>; 5] = Default::default();
    for (i, &m) in plan.iter().enumerate() {
        set_mode(m, cfg.pipe_tiles);
        let mut total = 0.0;
        for set in &sets {
            for (it, ids) in its.iter().zip(set) {
                it.set_i32(ids);
            }
            let t = std::time::Instant::now();
            gpu.compute(&graph).map_err(|e| e.to_string())?;
            total += t.elapsed().as_secs_f64();
        }
        if i >= MODES.len() {
            best[m] = best[m].min(total / (sets.len() * layers) as f64);
        }
        last[m] = outs.iter().flat_map(|o| o.to_f32()).collect();
    }
    let differ = [0, 1, 2, 3, 4].map(|m| {
        last[m]
            .iter()
            .zip(&last[0])
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    });
    let want = reference(c, stacks, &x, &sets[sets.len() - 1][0], n_used, n_tok);
    let got = &last[0][..want.len()];
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-12);
    let ref_err = got
        .iter()
        .zip(&want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
        / scale;
    Ok(Row {
        case: c.name,
        n_tok,
        vram_below: below,
        n_expert,
        secs: best,
        differ,
        ref_err: ref_err as f64,
        bits_hash: bits_hash(&last[0]),
    })
}

/// Layer 0's output in f32 from the dequantized experts and the unquantized activations
/// (`[n_tok][n_used][rows]`).
fn reference(
    c: Case,
    stacks: &[Vec<u8>],
    x: &[f32],
    ids: &[i32],
    n_used: usize,
    n_tok: usize,
) -> Vec<f32> {
    let row_bytes = row_size(c.ty, c.k);
    let n_x = if c.glu { 1 } else { n_used };
    let mut out = Vec::with_capacity(n_tok * n_used * c.rows);
    for t in 0..n_tok {
        for j in 0..n_used {
            let e = ids[t * n_used + j] as usize;
            let col = &x[(t * n_x + if c.glu { 0 } else { j }) * c.k..][..c.k];
            for r in 0..c.rows {
                let dot = |s: &Vec<u8>| {
                    let at = (e * c.rows + r) * row_bytes;
                    let w = dequantize(c.ty, &s[at..at + row_bytes], c.k);
                    w.iter().zip(col).map(|(a, b)| a * b).sum::<f32>()
                };
                out.push(if c.glu {
                    let g = dot(&stacks[0]);
                    g / (1.0 + (-g).exp()) * dot(&stacks[1])
                } else {
                    dot(&stacks[0])
                });
            }
        }
    }
    out
}
