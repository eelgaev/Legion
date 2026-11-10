//! The BF16 LM head of verify steps in isolation (#1181): `w @ x`, the soft-cap and the argmax
//! for a BF16 `[k, m]` head and 49-128 columns on one GPU, as ggml's nodes (`mul_mat`, the
//! scale, tanh and scale ggml-cuda fuses into its soft-cap, `argmax`: the reference bits) and
//! as the op ([`super::head`]), timed. [`check`] compares every logit's and pick's bits in one
//! graph that holds both, on random weights and on weights whose rows repeat (ties, which
//! `argmax_f32` breaks by its threads' order). Used by the hidden `legion head-mm-bench`
//! subcommand (a build tree's binary on the GPU box).

use super::{MAX_COLS, MIN_COLS, fits, head};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, types};

/// What [`run`] and [`check`] measure.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per row.
    pub k: usize,
    /// Rows of the head.
    pub m: usize,
    /// Column counts of the input.
    pub cols: Vec<usize>,
    pub rounds: usize,
    /// The soft-cap (0: none).
    pub cap: f32,
}

impl Default for Config {
    /// Gemma 4 31B's head and soft-cap, at 7 to 16 sequences of 8 verified tokens.
    fn default() -> Self {
        Self {
            k: 5376,
            m: 262144,
            cols: (7..=16).map(|s| 8 * s).collect(),
            rounds: 5,
            cap: 30.0,
        }
    }
}

/// One column count in [`run`].
#[derive(Clone, Debug)]
pub struct Row {
    pub cols: usize,
    /// Best time of ggml's nodes and of the op (seconds).
    pub ggml_secs: f64,
    pub op_secs: f64,
    /// Weight bytes.
    pub bytes: f64,
}

/// One column count in [`check`]: logits and picks whose bits differ from ggml's, the cases
/// (weights x inputs) compared, the picks with a tie at the maximum, and an FNV-1a hash of the
/// op's picks.
#[derive(Clone, Debug)]
pub struct Check {
    pub cols: usize,
    pub cases: usize,
    pub logits_differing: usize,
    pub picks_differing: usize,
    pub ties: usize,
    pub hash: u64,
}

/// BF16 bytes of `n` deterministic values in [-amp, amp); a pattern of `period` values
/// repeated when `period` > 0.
fn bf16_noise(n: usize, seed: u64, amp: f32, period: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        if period > 0 && i >= period {
            let j = 2 * (i % period);
            let (a, b) = (out[j], out[j + 1]);
            out.push(a);
            out.push(b);
            continue;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let v = amp * ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
        out.extend_from_slice(&((super::bf16_round(v).to_bits() >> 16) as u16).to_le_bytes());
    }
    out
}

/// Deterministic values in [-amp, amp).
fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            amp * ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        })
        .collect()
}

/// ggml's head nodes as Gemma 4 builds them: the matmul, `cap * tanh(v / cap)` (which
/// ggml-cuda fuses into one soft-cap kernel), the argmax.
fn reference<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>, cap: f32) -> [Tensor<'a>; 2] {
    let l = ctx.mul_mat(w, x);
    let l = if cap > 0.0 {
        ctx.scale(ctx.tanh(ctx.scale(l, 1.0 / cap)), cap)
    } else {
        l
    };
    [l, ctx.argmax(l)]
}

fn check_shape(cfg: &Config, n: usize) -> Result<(), String> {
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::BF16, &[cfg.k as i64, cfg.m as i64]);
    let x = ctx.new_tensor(types::F32, &[cfg.k as i64, n as i64]);
    if !fits(w, x) {
        return Err(format!(
            "k {}, m {}, cols {n}: the op does not take it ({MIN_COLS}-{MAX_COLS} columns, k a \
             multiple of 32, m of 256)",
            cfg.k, cfg.m
        ));
    }
    if !super::device_supported() {
        return Err("this GPU does not run the head op (compute capability 8.0+)".into());
    }
    Ok(())
}

/// The weights `[k, m]` in `gpu`'s memory: random (`tied` false) or 1,021 random rows repeated.
fn weights<'w>(
    wctx: &'w Context,
    gpu: &Backend,
    cfg: &Config,
    tied: bool,
) -> Result<Tensor<'w>, String> {
    let w = wctx.new_tensor(types::BF16, &[cfg.k as i64, cfg.m as i64]);
    wctx.alloc_tensors(gpu.buffer_type())
        .map_err(|e| e.to_string())?;
    let period = if tied { 1021 * cfg.k } else { 0 };
    w.set_bytes(&bf16_noise(
        cfg.k * cfg.m,
        11 + u64::from(tied),
        0.05,
        period,
    ));
    Ok(w)
}

/// ggml's head and the op for every column count of `cfg` on `gpu`, taking turns within a
/// round; the best round counts.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let mut out = Vec::new();
    let wctx = Context::new(2, 0);
    let w = weights(&wctx, gpu, cfg, false)?;
    for &n in &cfg.cols {
        check_shape(cfg, n)?;
        let ctxs = [Context::new(16, 16), Context::new(16, 16)];
        let graphs: Vec<(Tensor<'_>, Graph<'_>)> = ctxs
            .iter()
            .enumerate()
            .map(|(i, ctx)| {
                let x = ctx
                    .new_tensor(types::F32, &[cfg.k as i64, n as i64])
                    .as_input();
                let [l, a] = if i == 0 {
                    reference(ctx, w, x, cfg.cap)
                } else {
                    let (l, a) = head(ctx, w, x, cfg.cap);
                    [l, a]
                };
                let mut g = ctx.new_graph(16);
                g.expand(l.as_output());
                g.expand(a.as_output());
                (x, g)
            })
            .collect();
        let xs = noise(cfg.k * n, 5, 2.0);
        for (ctx, (x, g)) in ctxs.iter().zip(&graphs) {
            ctx.alloc_graph(g, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            x.set_f32(&xs);
        }
        let mut best = [f64::MAX; 2];
        for round in 0..=cfg.rounds {
            for (i, (_, g)) in graphs.iter().enumerate() {
                let t = std::time::Instant::now();
                gpu.compute(g)
                    .map_err(|e| format!("{n} cols {}: {e}", ["ggml", "op"][i]))?;
                if round > 0 {
                    best[i] = best[i].min(t.elapsed().as_secs_f64());
                }
            }
        }
        out.push(Row {
            cols: n,
            ggml_secs: best[0],
            op_secs: best[1],
            bytes: (2 * cfg.k * cfg.m) as f64,
        });
    }
    Ok(out)
}

/// FNV-1a over `v`.
fn hash(v: &[i32]) -> u64 {
    v.iter()
        .flat_map(|f| f.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
}

/// The op's logits and picks against ggml's in one graph, for every column count of `cfg` on
/// `gpu`: random weights and weights of repeated rows, each with inputs of two sizes (the
/// larger saturating the soft-cap).
pub fn check(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Check>, String> {
    let wctx = [Context::new(2, 0), Context::new(2, 0)];
    let ws = [
        weights(&wctx[0], gpu, cfg, false)?,
        weights(&wctx[1], gpu, cfg, true)?,
    ];
    let mut out = Vec::new();
    for &n in &cfg.cols {
        check_shape(cfg, n)?;
        let mut c = Check {
            cols: n,
            cases: 0,
            logits_differing: 0,
            picks_differing: 0,
            ties: 0,
            hash: 0,
        };
        let mut picks = Vec::new();
        for &w in &ws {
            let ctx = Context::new(32, 32);
            let x = ctx
                .new_tensor(types::F32, &[cfg.k as i64, n as i64])
                .as_input();
            let [rl, ra] = reference(&ctx, w, x, cfg.cap);
            let (ol, oa) = head(&ctx, w, x, cfg.cap);
            let outs = [rl, ra, ol, oa].map(|t| t.as_output());
            let mut graph = ctx.new_graph(32);
            for t in outs {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| e.to_string())?;
            for (seed, amp) in [(21u64, 2.0f32), (22, 40.0)] {
                x.set_f32(&noise(cfg.k * n, seed + n as u64, amp));
                gpu.compute(&graph).map_err(|e| format!("{n} cols: {e}"))?;
                let (a, b) = (outs[2].to_f32(), outs[0].to_f32());
                c.logits_differing += a
                    .iter()
                    .zip(&b)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count();
                let (pa, pb) = (outs[3].to_i32(), outs[1].to_i32());
                c.picks_differing += pa.iter().zip(&pb).filter(|(p, q)| p != q).count();
                for (j, &p) in pb.iter().enumerate() {
                    if p >= 0 {
                        let col = &b[j * cfg.m..(j + 1) * cfg.m];
                        let top = col[p as usize];
                        c.ties += usize::from(col.iter().filter(|&&v| v == top).count() > 1);
                    }
                }
                picks.extend_from_slice(&pa);
                c.cases += 1;
            }
        }
        c.hash = hash(&picks);
        out.push(c);
    }
    Ok(out)
}
