//! The op in isolation (#836, #814): Qwen3.8-Flash-Next's expert matmuls (gate/up Q4_K, down
//! Q5_1) at 9-63 token steps on one GPU, ggml's `mul_mat_id` (MMQ) against [`super::mul_mat_id`]
//! and Legion's expert matvec ([`crate::model::core::expert_mv::mul_mat_id`], what
//! `--expert-mv-max-tokens` gives those steps). Checks that the op gives MMQ's bits, counts the
//! matvec's differing values, and times all three against the bytes the step's distinct experts
//! hold. Used by the hidden `legion mmq-ids-bench` subcommand (a build tree's binary on the GPU
//! box).

use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, quantize, types};

/// One op shape.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub name: &'static str,
    pub ty: u32,
    pub k: usize,
    pub rows: usize,
    /// One activation column per token (gate, up); else one per pair (down).
    pub shared_x: bool,
}

pub const CASES: [Case; 2] = [
    Case {
        name: "gate/up Q4_K",
        ty: types::Q4_K,
        k: 2560,
        rows: 640,
        shared_x: true,
    },
    Case {
        name: "down Q5_1",
        ty: types::Q5_1,
        k: 640,
        rows: 2560,
        shared_x: false,
    },
];

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    pub n_expert: usize,
    pub n_used: usize,
    /// Distinct experts quantized (the stack repeats them; the addresses differ).
    pub distinct: usize,
    /// Copies of the stacks a timed graph holds (each op reads other memory than the last).
    pub layers: usize,
    pub toks: Vec<usize>,
    /// Id sets per round.
    pub sets: usize,
    pub rounds: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_expert: 512,
            n_used: 10,
            distinct: 32,
            layers: 4,
            toks: vec![9, 16, 32, 48, 63],
            sets: 8,
            rounds: 5,
        }
    }
}

/// One case at one step size.
#[derive(Clone, Debug)]
pub struct Row {
    pub case: &'static str,
    pub n_tok: usize,
    /// Best mean time per op over the rounds: ggml's MMQ, the op, the expert matvec (seconds).
    pub secs: [f64; 3],
    /// Output values whose bits differ from ggml's, over every set and layer: the op's ...
    pub differ: usize,
    /// ... and the expert matvec's.
    pub differ_mv: usize,
    pub values: usize,
    /// Mean bytes of the distinct experts an op reads (the floor of what it must stream).
    pub bytes: f64,
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

/// Distinct ids per token; tokens of one sequence (groups of 4: a token and its drafts) share
/// half of theirs.
fn ids(n_used: usize, n_tok: usize, n_expert: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed | 1);
    let mut out: Vec<i32> = Vec::new();
    for t in 0..n_tok {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < n_used {
            let e = if t % 4 > 0 && chosen.len() < n_used / 2 {
                out[(t - t % 4) * n_used + chosen.len()]
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

/// Every case at every step size of `cfg` on `gpu`; the two take turns within a round, the best
/// round counts.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    // The op computes the chain of the GPU it runs on (device 0 of the process).
    #[cfg(feature = "cuda")]
    super::set_gpu(crate::residency::cuda::caps(0).map(|c| (c.cc_major, c.cc_minor)));
    crate::model::core::expert_mv::set_enabled(true);
    let mut rows = Vec::new();
    for c in CASES {
        let mut rng = Rng(7);
        let w: Vec<f32> = (0..c.k * c.rows * cfg.distinct)
            .map(|_| rng.f32() * 0.05)
            .collect();
        let q = quantize(c.ty, &w, c.k);
        let stack: Vec<u8> = (0..cfg.n_expert.div_ceil(cfg.distinct))
            .flat_map(|_| q.iter().copied())
            .take(q.len() / cfg.distinct * cfg.n_expert)
            .collect();
        for &n_tok in &cfg.toks {
            rows.push(case(gpu, cfg, c, &stack, n_tok)?);
        }
    }
    Ok(rows)
}

struct Built<'a> {
    xt: Tensor<'a>,
    its: Vec<Tensor<'a>>,
    outs: Vec<Tensor<'a>>,
}

fn case(
    gpu: &mut Backend,
    cfg: &Config,
    c: Case,
    stack: &[u8],
    n_tok: usize,
) -> Result<Row, String> {
    let (layers, n_used, n_expert) = (cfg.layers, cfg.n_used, cfg.n_expert);
    let weights = Context::new(layers, 0);
    let shape = [c.k as i64, c.rows as i64, n_expert as i64];
    let ws: Vec<_> = (0..layers)
        .map(|_| weights.new_tensor(c.ty, &shape))
        .collect();
    weights
        .alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for w in &ws {
        w.set_bytes(stack);
    }
    let n_x = if c.shared_x { 1 } else { n_used };
    let mut rng = Rng(11 + n_tok as u64);
    let x: Vec<f32> = (0..c.k * n_x * n_tok).map(|_| rng.f32()).collect();
    let ctxs = [
        Context::new(1 + 8 * layers, 16 * layers + 8),
        Context::new(1 + 8 * layers, 16 * layers + 8),
        Context::new(1 + 8 * layers, 16 * layers + 8),
    ];
    let mut built = Vec::new();
    let mut graphs = Vec::new();
    for (m, ctx) in ctxs.iter().enumerate() {
        let xt = ctx.new_tensor(types::F32, &[c.k as i64, n_x as i64, n_tok as i64]);
        let its: Vec<_> = (0..layers)
            .map(|_| ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64]))
            .collect();
        ctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        xt.set_f32(&x);
        let outs: Vec<_> = ws
            .iter()
            .zip(&its)
            .map(|(&w, &it)| match m {
                0 => ctx.mul_mat_id(w, xt, it),
                1 => super::mul_mat_id(ctx, w, xt, it),
                _ => crate::model::core::expert_mv::mul_mat_id(ctx, w, xt, it),
            })
            .collect();
        let mut graph = ctx.new_graph(16 * layers + 8);
        for o in &outs {
            graph.expand(*o);
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
        built.push(Built { xt, its, outs });
        graphs.push(graph);
    }
    let sets: Vec<Vec<Vec<i32>>> = (0..cfg.sets as u64)
        .map(|i| {
            (0..layers as u64)
                .map(|l| ids(n_used, n_tok, n_expert, 100 + 16 * i + l))
                .collect()
        })
        .collect();
    let expert_bytes = stack.len() as f64 / n_expert as f64;
    let distinct: usize = sets
        .iter()
        .flatten()
        .map(|ids| {
            let mut e = ids.clone();
            e.sort_unstable();
            e.dedup();
            e.len()
        })
        .sum();
    let bytes = distinct as f64 * expert_bytes / (sets.len() * layers) as f64;
    let mut best = [f64::MAX; 3];
    let (mut differ, mut differ_mv, mut values) = (0, 0, 0);
    for round in 0..=cfg.rounds {
        let mut total = [0.0; 3];
        for set in &sets {
            let mut got: [Vec<f32>; 3] = Default::default();
            for m in 0..3 {
                let b = &built[m];
                b.xt.set_f32(&x);
                for (it, ids) in b.its.iter().zip(set) {
                    it.set_i32(ids);
                }
                let t = std::time::Instant::now();
                gpu.compute(&graphs[m]).map_err(|e| e.to_string())?;
                total[m] += t.elapsed().as_secs_f64();
                if round == 0 {
                    got[m] = b.outs.iter().flat_map(|o| o.to_f32()).collect();
                }
            }
            if round == 0 {
                values += got[0].len();
                let count = |o: &[f32]| {
                    got[0]
                        .iter()
                        .zip(o)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                };
                differ += count(&got[1]);
                differ_mv += count(&got[2]);
            }
        }
        // Round 0 is the warm-up and the check.
        if round > 0 {
            for m in 0..3 {
                best[m] = best[m].min(total[m] / (sets.len() * layers) as f64);
            }
        }
    }
    Ok(Row {
        case: c.name,
        n_tok,
        secs: best,
        differ,
        differ_mv,
        values,
        bytes,
    })
}
