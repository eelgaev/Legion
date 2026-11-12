//! The op against ggml in isolation (#1133): `w @ x` times a one-value scale for 2D NVFP4
//! weights (by default Qwen3.8-27B's FFN shapes: gate and up `[5120, 17408]`, down
//! `[17408, 5120]`) at 1-8 columns on one GPU, once on ggml (quantize_q8_1 + MMVQ, then `mul`,
//! with the precision hints [`super::super::linear`] gives NVFP4 matmuls) and once on the op.
//! Checks that every output has ggml's bits and times both per matmul. Used by
//! the hidden `legion nvfp4-mvq-bench` subcommand (a build tree's binary on the GPU box) and by
//! `tests/nvfp4_mvq.rs`.

use super::{Tune, fits, mul_mat, set_tune};
use crate::ggml::{Backend, ComputeAllocator, Context, Tensor, ffi, types};

/// Shapes and runs.
#[derive(Clone, Debug)]
pub struct Config {
    /// Weight shapes `(k, rows)`.
    pub shapes: Vec<(usize, usize)>,
    /// Columns per step.
    pub cols: Vec<usize>,
    /// Weights per shape, each its own (more than the L2 cache holds, so the reads stream).
    pub layers: usize,
    /// Timed rounds (the best counts; ggml and the variants take turns within a round).
    pub rounds: usize,
    /// Each matmul times a one-value scale (as NVFP4 checkpoints have; the 27B's do).
    pub scale: bool,
    /// Each variant's timed rounds back to back (for a power log), with its start and end
    /// times on stderr, instead of in turns.
    pub blocks: bool,
    /// Launch shapes of the op to time besides the default (5-8 columns; each checked too).
    pub tunes: Vec<Tune>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            shapes: vec![(5120, 17408), (17408, 5120)],
            cols: (1..=8).collect(),
            layers: 4,
            rounds: 5,
            scale: true,
            blocks: false,
            tunes: Vec::new(),
        }
    }
}

/// One variant at one shape and column count.
#[derive(Clone, Debug)]
pub struct Row {
    /// `ggml` or the op (`mvq`).
    pub variant: String,
    pub k: usize,
    pub rows: usize,
    pub cols: usize,
    /// Best mean time per matmul over the rounds (seconds).
    pub secs: f64,
    /// Output values whose bits differ from ggml's.
    pub differ: usize,
    /// Output values compared.
    pub outputs: usize,
    /// The first differing output: (layer, column, row, got, ggml's).
    pub first: Option<(usize, usize, usize, f32, f32)>,
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

/// Random NVFP4 blocks: sub-block scales of moderate size (or, `any_scale`, any byte: signed,
/// tiny, huge and the NaN codes ggml reads as 0), codes uniform.
pub(crate) fn nvfp4_bytes(bytes: usize, seed: u64, any_scale: bool) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(bytes);
    while out.len() < bytes {
        let r = rng.next();
        for i in 0..4 {
            out.push(if any_scale {
                (r >> (8 * i)) as u8
            } else {
                (((3 + (r >> (8 * i)) % 6) << 3) | (r >> (32 + 3 * i)) & 7) as u8
            });
        }
        for _ in 0..4 {
            out.extend(rng.next().to_le_bytes());
        }
    }
    out.truncate(bytes);
    out
}

/// Activations for `cols` columns of `k`: uniform in [-2, 2), column 1's first block all zero
/// (ggml's amax == 0 case) and column 2's tiny.
pub(crate) fn activations(k: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut x: Vec<f32> = (0..k * cols)
        .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0)
        .collect();
    if cols > 1 {
        x[k..k + 64].fill(0.0);
    }
    if cols > 2 {
        x[2 * k..3 * k].iter_mut().for_each(|v| *v *= 1e-30);
    }
    x
}

/// ggml's `w @ x` as [`super::super::linear::Linear::mul_mat`] builds it for an NVFP4 weight
/// with A8 activations (its precision hints), times `s`.
fn ggml_mul_mat<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
    s: Option<Tensor<'a>>,
) -> Tensor<'a> {
    let res = ctx.mul_mat(w, x);
    ctx.prec_set_src(res, ffi::ggml_prec_GGML_PREC_Q8, 1);
    ctx.prec_set_acc(res, ffi::ggml_prec_GGML_PREC_BF16);
    match s {
        Some(s) => ctx.mul(res, s),
        None => res,
    }
}

/// The variants: ggml, then the op.
const VARIANTS: [&str; 2] = ["ggml", "mvq"];

/// The variants' names, ggml first.
pub fn variant_names() -> Vec<String> {
    VARIANTS.iter().map(|v| v.to_string()).collect()
}

/// A tuned launch shape's variant name.
fn tune_name(t: &Tune) -> String {
    format!("r{}c{}k{}a{}", t.region, t.ctas, t.chunks, t.ahead)
}

/// ggml and the op at every shape and column count of `cfg` on `gpu`; they take turns within
/// a round, the best round counts.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for (si, &(k, m)) in cfg.shapes.iter().enumerate() {
        let weights_ctx = Context::new(2 * cfg.layers + 2, 0);
        let w: Vec<Tensor<'_>> = (0..cfg.layers)
            .map(|_| weights_ctx.new_tensor(types::NVFP4, &[k as i64, m as i64]))
            .collect();
        let s = weights_ctx.new_tensor(types::F32, &[1]);
        weights_ctx
            .alloc_tensors(gpu.buffer_type())
            .map_err(|e| e.to_string())?;
        for (l, wl) in w.iter().enumerate() {
            // Every other weight with scale bytes of any value.
            wl.set_bytes(&nvfp4_bytes(
                k / 64 * 36 * m,
                (1 + 16 * si + l) as u64,
                l % 2 == 1,
            ));
        }
        s.set_f32(&[0.0123]);
        for &n in &cfg.cols {
            // ggml, the op, then the op at each tuned shape (5-8 columns).
            let tunes: Vec<Tune> = if n > 4 { cfg.tunes.clone() } else { Vec::new() };
            let nv = 2 + tunes.len();
            let names: Vec<String> = VARIANTS
                .iter()
                .map(|v| v.to_string())
                .chain(tunes.iter().map(tune_name))
                .collect();
            let tune_of = |vi: usize| {
                if vi < 2 {
                    Tune::default()
                } else {
                    tunes[vi - 2]
                }
            };
            let ctxs: Vec<Context> = (0..nv)
                .map(|_| Context::new(8 + 4 * cfg.layers, 4 * cfg.layers + 16))
                .collect();
            let mut graphs = Vec::new();
            for (vi, ctx) in ctxs.iter().enumerate() {
                let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                let mut graph = ctx.new_graph(4 * cfg.layers + 16);
                let scale = cfg.scale.then_some(s);
                if vi >= 1
                    && !(fits(w[0], x, scale)
                        && w[0]
                            .cuda_device()
                            .is_some_and(crate::model::nvfp4_a4::supported_on))
                {
                    return Err(format!(
                        "the op does not take {k}x{m} at {n} cols on this GPU (see `fits`: k a \
                         multiple of 256 up to 65536, rows a multiple of 4, 1-8 cols, an \
                         sm_120a/sm_121a device)"
                    ));
                }
                let outs: Vec<Tensor<'_>> = w
                    .iter()
                    .map(|&wl| {
                        let y = if vi == 0 {
                            ggml_mul_mat(ctx, wl, x, scale)
                        } else {
                            mul_mat(ctx, wl, x, scale)
                        }
                        .as_output();
                        graph.expand(y);
                        y
                    })
                    .collect();
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| e.to_string())?;
                graphs.push((x, graph, outs));
            }
            let xs = activations(k, n, (7 + 31 * n + si) as u64);
            let mut best = vec![f64::MAX; nv];
            let mut reference: Vec<f32> = Vec::new();
            let mut differ = vec![0usize; nv];
            let mut first = vec![None; nv];
            // A warm-up run of each variant (checked), then the timed rounds.
            let rounds = cfg.rounds.max(1);
            let order: Vec<(usize, usize)> = if cfg.blocks {
                (0..nv)
                    .map(|vi| (0, vi))
                    .chain((0..nv).flat_map(|vi| (1..=rounds).map(move |r| (r, vi))))
                    .collect()
            } else {
                // Round 0 runs ggml first (its outputs are the reference); odd rounds run the
                // variants in reverse, so none always follows the same one.
                (0..=rounds)
                    .flat_map(|r| {
                        (0..nv).map(move |i| (r, if r % 2 == 1 { nv - 1 - i } else { i }))
                    })
                    .collect()
            };
            for &(round, vi) in &order {
                let (x, graph, outs) = &graphs[vi];
                let mark = |what: &str| {
                    let t = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0.0, |d| d.as_secs_f64());
                    eprintln!("{t:.3} {} {k}x{m} cols {n} {what}", names[vi]);
                };
                if cfg.blocks && round == 1 {
                    // A gap in the power log between the blocks.
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    mark("start");
                }
                x.set_f32(&xs);
                set_tune(tune_of(vi));
                let t = std::time::Instant::now();
                let r = gpu.compute(graph);
                let secs = t.elapsed().as_secs_f64();
                set_tune(Tune::default());
                if let Err(e) = r {
                    return Err(format!("{}: {e}", names[vi]));
                }
                if round > 0 {
                    best[vi] = best[vi].min(secs / cfg.layers as f64);
                } else {
                    let got: Vec<f32> = outs.iter().flat_map(|o| o.to_f32()).collect();
                    if vi == 0 {
                        reference = got;
                    } else {
                        for (o, (a, b)) in got.iter().zip(&reference).enumerate() {
                            if a.to_bits() != b.to_bits() {
                                differ[vi] += 1;
                                let (l, c, r) = (o / (m * n), o / m % n, o % m);
                                first[vi].get_or_insert((l, c, r, *a, *b));
                            }
                        }
                    }
                }
                if cfg.blocks && round == rounds {
                    mark("end");
                }
            }
            for (vi, name) in names.iter().enumerate() {
                rows.push(Row {
                    variant: name.clone(),
                    k,
                    rows: m,
                    cols: n,
                    secs: best[vi],
                    differ: differ[vi],
                    outputs: reference.len(),
                    first: first[vi],
                });
            }
        }
    }
    Ok(rows)
}
