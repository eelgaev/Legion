//! The routed experts of a decode or verify step in isolation (#1176): Qwen3.8-Flash-Next
//! UD-Q4_K_XL's shapes (gate and up Q4_K `[2560, 640]`, down Q5_1 `[640, 2560]`, 10 experts per
//! token) at 1-8 tokens on one GPU, the gate + up + SwiGLU part and the down part timed apart, each
//! on ggml's `mul_mat_id` (quantize_q8_1 + MMVQ, then `silu` and `mul`), on [`swiglu_id`] and
//! [`mul_mat_id`] at the default [`Tuning`] and at the bench's others, and with the SwiGLU split
//! (each matmul on [`mul_mat_id`], then ggml's `silu` and `mul`). Checks that every variant gives
//! ggml's bits and gives each part's time per layer and the bandwidth of its distinct experts'
//! weights. The tokens pick their experts from a pool: a pool of 10 is a step whose tokens all
//! pick the same experts, a big pool one whose tokens share few. Used by the hidden
//! `legion mmvq-once-bench` subcommand (a build tree's binary on the GPU box) and by the GPU test.

use super::{Tuning, default_tuning, fits_shape, force_tuning, glu_fits, mul_mat_id, swiglu_id};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};
use crate::model::core::nvfp4_once::bench::routing;

/// Shapes and runs.
#[derive(Clone, Debug)]
pub struct Config {
    /// The model's width (gate and up: `[k, m]`, down: `[m, k]`).
    pub k: usize,
    pub m: usize,
    /// The gate and up weights' type and the down weights'.
    pub glu_ty: ffi::ggml_type,
    pub down_ty: ffi::ggml_type,
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
    /// The tunings timed besides the default (the same bits each).
    pub tunings: Vec<Tuning>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k: 2560,
            m: 640,
            glu_ty: types::Q4_K,
            down_ty: types::Q5_1,
            n_expert: 128,
            n_used: 10,
            layers: 8,
            toks: vec![1, 2, 4, 8],
            pools: vec![10, 32, 128],
            sets: 4,
            rounds: 5,
            tunings: BENCH_TUNINGS.to_vec(),
        }
    }
}

/// The tunings the bench times by default besides the default ([`default_tuning`]).
pub const BENCH_TUNINGS: [Tuning; 6] = [
    Tuning {
        stage_bytes: 4 * 1024,
        warps: 4,
    },
    Tuning {
        stage_bytes: 16 * 1024,
        warps: 4,
    },
    Tuning {
        stage_bytes: 4 * 1024,
        warps: 8,
    },
    Tuning {
        stage_bytes: 8 * 1024,
        warps: 8,
    },
    Tuning {
        stage_bytes: 8 * 1024,
        warps: 4,
    },
    Tuning {
        stage_bytes: 24 * 1024,
        warps: 8,
    },
];

/// The part of the expert FFN a graph runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Gate, up and the SwiGLU.
    Glu,
    /// The down matmul (of a random activation).
    Down,
}

impl Part {
    pub fn name(self) -> &'static str {
        match self {
            Part::Glu => "glu",
            Part::Down => "down",
        }
    }
}

/// How a graph computes its part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Ggml,
    Once,
    /// The SwiGLU's matmuls on [`mul_mat_id`] apart (the down part: the same as `Once`).
    Split,
}

/// One variant at one part, step size and pool.
#[derive(Clone, Debug)]
pub struct Row {
    pub part: Part,
    /// `ggml`, `once 8K/4w` (the stage bytes and warps) or `split 8K/4w`.
    pub variant: String,
    pub n_tok: usize,
    pub pool: usize,
    /// Distinct experts per layer (mean).
    pub experts: f64,
    /// Best mean time per layer over the rounds (seconds).
    pub secs: f64,
    /// The distinct experts' weight bytes per layer (mean).
    pub bytes: f64,
    /// Output values whose bits differ from ggml's.
    pub differ: usize,
    /// Output values compared.
    pub outputs: usize,
}

impl Row {
    /// The weights' bandwidth: their bytes over the time (GB/s).
    pub fn gbs(&self) -> f64 {
        self.bytes / self.secs / 1e9
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Rows of `ty` blocks of `k` values from ggml's own quantizer (valid scales, no NaN): a pool of
/// quantized random rows, each expert's row a random one of them (quantizing every row would take
/// minutes), `n` rows in all.
pub fn quantized_rows(ty: ffi::ggml_type, k: usize, n: usize, seed: u64) -> Vec<u8> {
    const POOL: usize = 512;
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let pool_rows = POOL.min(n).max(1);
    // Rows of different scales, so the per-row sums differ in size as a model's do.
    let floats: Vec<f32> = (0..pool_rows * k)
        .map(|i| 0.02 * (1 + (i / k) % 7) as f32 * rng.unit())
        .collect();
    let pool = crate::ggml::quantize(ty, &floats, k);
    let row = crate::ggml::row_size(ty, k);
    let mut out = Vec::with_capacity(row * n);
    for _ in 0..n {
        let r = (rng.next() % pool_rows as u64) as usize;
        out.extend_from_slice(&pool[r * row..(r + 1) * row]);
    }
    out
}

/// The graph of `w.len()` layers' `part` (layer `l` on weights `w[l]`: gate, up, down), each layer
/// with its own ids: the input, the ids, the graph and the outputs.
fn build<'a>(
    ctx: &'a Context,
    (part, kind): (Part, Kind),
    w: &[[Tensor<'a>; 3]],
    (k, m): (usize, usize),
    n_used: usize,
    n_tok: usize,
) -> Result<(Tensor<'a>, Vec<Tensor<'a>>, Graph<'a>, Vec<Tensor<'a>>), String> {
    let x = match part {
        Part::Glu => ctx.new_tensor(types::F32, &[k as i64, 1, n_tok as i64]),
        Part::Down => ctx.new_tensor(types::F32, &[m as i64, n_used as i64, n_tok as i64]),
    }
    .as_input();
    let ids: Vec<_> = w
        .iter()
        .map(|_| {
            ctx.new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                .as_input()
        })
        .collect();
    let mut graph = ctx.new_graph(16 * w.len() + 16);
    let mut outs = Vec::with_capacity(w.len());
    for (&id, &[wg, wu, wd]) in ids.iter().zip(w) {
        let fits = match part {
            Part::Glu => glu_fits(wg, wu, x, id),
            Part::Down => fits_shape(wd, x, id),
        };
        if kind != Kind::Ggml && !fits {
            return Err(format!(
                "the op does not take the {} part's shapes",
                part.name()
            ));
        }
        let y = match (part, kind) {
            (Part::Glu, Kind::Ggml) => {
                let up = ctx.mul_mat_id(wu, x, id);
                let gate = ctx.mul_mat_id(wg, x, id);
                ctx.mul(ctx.silu(gate), up)
            }
            (Part::Glu, Kind::Once) => swiglu_id(ctx, wg, wu, x, id),
            (Part::Glu, Kind::Split) => {
                let up = mul_mat_id(ctx, wu, x, id);
                let gate = mul_mat_id(ctx, wg, x, id);
                ctx.mul(ctx.silu(gate), up)
            }
            (Part::Down, Kind::Ggml) => ctx.mul_mat_id(wd, x, id),
            (Part::Down, _) => mul_mat_id(ctx, wd, x, id),
        }
        .as_output();
        graph.expand(y);
        outs.push(y);
    }
    Ok((x, ids, graph, outs))
}

/// The variants of a part: ggml, then the op at the default tuning and at `tunings`, then (the
/// SwiGLU part) split at the default.
fn variants(part: Part, n_tok: usize, tunings: &[Tuning]) -> Vec<(String, Kind, Tuning)> {
    let name = |t: Tuning| format!("{}K/{}w", t.stage_bytes / 1024, t.warps);
    let d = default_tuning(n_tok);
    let mut v = vec![("ggml".to_string(), Kind::Ggml, d)];
    v.push((format!("once {}", name(d)), Kind::Once, d));
    for &t in tunings.iter().filter(|&&t| t != d) {
        v.push((format!("once {}", name(t)), Kind::Once, t));
    }
    if part == Part::Glu {
        v.push((format!("split {}", name(d)), Kind::Split, d));
    }
    v
}

/// ggml and every variant at every part, step size and pool of `cfg` on `gpu`; the variants take
/// turns within a round, the best round counts. The tuning is back at its default afterwards.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let r = run_inner(gpu, cfg);
    force_tuning(None);
    r
}

fn run_inner(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let (k, m, n_expert) = (cfg.k, cfg.m, cfg.n_expert);
    let weights_ctx = Context::new(4 * cfg.layers, 0);
    let shape = |a: usize, b: usize| [a as i64, b as i64, n_expert as i64];
    let w: Vec<[Tensor<'_>; 3]> = (0..cfg.layers)
        .map(|_| {
            [
                weights_ctx.new_tensor(cfg.glu_ty, &shape(k, m)),
                weights_ctx.new_tensor(cfg.glu_ty, &shape(k, m)),
                weights_ctx.new_tensor(cfg.down_ty, &shape(m, k)),
            ]
        })
        .collect();
    weights_ctx
        .alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    for (l, lw) in w.iter().enumerate() {
        for (i, t) in lw.iter().enumerate() {
            let (ty, kk, rows) = if i < 2 {
                (cfg.glu_ty, k, m)
            } else {
                (cfg.down_ty, m, k)
            };
            t.set_bytes(&quantized_rows(
                ty,
                kk,
                rows * n_expert,
                (3 + 8 * l + i) as u64,
            ));
        }
    }
    let glu_row = crate::ggml::row_size(cfg.glu_ty, k);
    let down_row = crate::ggml::row_size(cfg.down_ty, m);
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
            for part in [Part::Glu, Part::Down] {
                let (n_in, bytes) = match part {
                    Part::Glu => (k * n_tok, used * (2 * m * glu_row) as f64),
                    Part::Down => (m * cfg.n_used * n_tok, used * (k * down_row) as f64),
                };
                let mut rng = Rng(5 + n_tok as u64);
                let xs: Vec<f32> = (0..n_in).map(|_| 2.0 * rng.unit()).collect();
                let vars = variants(part, n_tok, &cfg.tunings);
                rows.extend(time_part(
                    gpu,
                    cfg,
                    &w,
                    (part, n_tok, pool),
                    (&sets, &xs, &vars),
                    (used, bytes),
                )?);
            }
        }
    }
    Ok(rows)
}

/// One part at one step size and pool: every variant timed and checked against ggml's bits.
#[allow(clippy::type_complexity)]
fn time_part(
    gpu: &mut Backend,
    cfg: &Config,
    w: &[[Tensor<'_>; 3]],
    (part, n_tok, pool): (Part, usize, usize),
    (sets, xs, vars): (&[Vec<Vec<i32>>], &[f32], &[(String, Kind, Tuning)]),
    (used, bytes): (f64, f64),
) -> Result<Vec<Row>, String> {
    let kinds = [Kind::Ggml, Kind::Once, Kind::Split];
    let kinds: Vec<Kind> = kinds
        .into_iter()
        .filter(|kind| vars.iter().any(|v| v.1 == *kind))
        .collect();
    let ctxs: Vec<Context> = kinds
        .iter()
        .map(|_| Context::new(8 + 16 * cfg.layers, 16 * cfg.layers + 16))
        .collect();
    let graphs = kinds
        .iter()
        .zip(&ctxs)
        .map(|(&kind, ctx)| build(ctx, (part, kind), w, (cfg.k, cfg.m), cfg.n_used, n_tok))
        .collect::<Result<Vec<_>, _>>()?;
    for (ctx, (_, _, graph, _)) in ctxs.iter().zip(&graphs) {
        ctx.alloc_graph(graph, ComputeAllocator::new(gpu.buffer_type()))
            .map_err(|e| e.to_string())?;
    }
    let n_var = vars.len();
    // A warm-up run of each variant (its outputs checked), then the timed rounds.
    let order: Vec<usize> = (0..n_var)
        .chain((0..cfg.rounds.max(1)).flat_map(|_| 0..n_var))
        .collect();
    let mut best = vec![f64::MAX; n_var];
    let mut differ = vec![0usize; n_var];
    let mut first: Vec<Option<(usize, f32, f32)>> = vec![None; n_var];
    let mut reference: Vec<Vec<f32>> = Vec::new();
    let mut outputs = 0;
    for (i, &vi) in order.iter().enumerate() {
        let (_, kind, tuning) = vars[vi];
        let g = kinds.iter().position(|&k| k == kind).unwrap_or(0);
        let (x, ids, graph, outs) = &graphs[g];
        force_tuning(Some(tuning));
        let mut total = 0.0;
        for (si, set) in sets.iter().enumerate() {
            // Every input before every run, as a served step fills its inputs: ggml's allocator
            // may place a later node over an input after its last use.
            x.set_f32(xs);
            for (it, ids) in ids.iter().zip(set) {
                it.set_i32(ids);
            }
            let t = std::time::Instant::now();
            gpu.compute(graph)
                .map_err(|e| format!("{} {}: {e}", part.name(), vars[vi].0))?;
            total += t.elapsed().as_secs_f64();
            if i < n_var {
                let got: Vec<f32> = outs.iter().flat_map(|o| o.to_f32()).collect();
                if vi == 0 {
                    outputs += got.len();
                    reference.push(got);
                } else {
                    for (o, (a, b)) in got.iter().zip(&reference[si]).enumerate() {
                        if a.to_bits() != b.to_bits() {
                            differ[vi] += 1;
                            first[vi].get_or_insert((o, *a, *b));
                        }
                    }
                }
            }
        }
        if i >= n_var {
            best[vi] = best[vi].min(total / (sets.len() * cfg.layers) as f64);
        }
    }
    let rows_per = match part {
        Part::Glu => cfg.m,
        Part::Down => cfg.k,
    };
    for (vi, f) in first.iter().enumerate() {
        if let Some((o, a, b)) = f {
            let per_layer = rows_per * cfg.n_used * n_tok;
            let (layer, rest) = (o / per_layer, o % per_layer);
            let (t, j, r) = (
                rest / (rows_per * cfg.n_used),
                rest / rows_per % cfg.n_used,
                rest % rows_per,
            );
            eprintln!(
                "{} toks {n_tok} pool {pool} {}: {} outputs differ; first: layer {layer} token \
                 {t} slot {j} row {r}: {a:e} vs {b:e} ({:#010x} vs {:#010x})",
                part.name(),
                vars[vi].0,
                differ[vi],
                a.to_bits(),
                b.to_bits()
            );
        }
    }
    Ok(vars
        .iter()
        .enumerate()
        .map(|(vi, (name, _, _))| Row {
            part,
            variant: name.clone(),
            n_tok,
            pool,
            experts: used,
            secs: best[vi],
            bytes,
            differ: differ[vi],
            outputs,
        })
        .collect())
}
