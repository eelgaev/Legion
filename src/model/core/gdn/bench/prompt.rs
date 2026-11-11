//! `legion gdn-bench --prompt` (#1183): a prompt step's delta rule at one GPU's head counts of
//! Qwen3.8-Flash-Next (48 value, 16 key heads), ggml's op (its chunked kernel on the GPUs ggml-cuda
//! gives it to) against [`gdn_mma`](super::super::super::gdn_mma) on each of its kernels: the
//! bits of every kernel's rows and final states against ggml's, and the time per layer over
//! `layers` layers with their own states. q, k and v are views of one conv-like row per token, as
//! a DeltaNet layer has them.

use super::super::super::gdn_mma::{self, Schedule};
use super::{Config, noise};
use crate::ggml::{Backend, ComputeAllocator, Context, GgmlError, Graph, Tensor, types};

/// Head size.
const S: i64 = 128;

/// The kernels compared after ggml's op, by [`gdn_mma`]'s schedule they force.
const KERNELS: [Schedule; 4] = [
    Schedule::Heads,
    Schedule::Staged,
    Schedule::SlicesTwo,
    Schedule::Slices,
];

/// Their names, in [`PromptRow::secs`] order after ggml's.
pub const PROMPT_KERNELS: [&str; 4] = ["heads", "staged", "slices x2", "slices"];

/// One prompt size.
#[derive(Clone, Debug)]
pub struct PromptRow {
    pub tokens: i64,
    pub seqs: i64,
    /// Seconds per layer: ggml's op, then [`PROMPT_KERNELS`] in order (0 when not timed).
    pub secs: Vec<f64>,
    /// Row and state values whose bits differ from ggml's, per kernel.
    pub differ: Vec<usize>,
}

/// One layer's inputs: a conv-like row per token (q, k, v views), g and beta.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    conv: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
}

impl<'a> Inputs<'a> {
    fn new(ctx: &'a Context, (h, nk): (i64, i64), nt: i64, ns: i64) -> Self {
        let row = 2 * S * nk + S * h;
        let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input().as_output();
        let conv = t(&[row, nt, ns]);
        let es = 4usize;
        let nb = [S as usize * es, row as usize * es, (row * nt) as usize * es];
        Self {
            conv,
            g: t(&[1, h, nt, ns]),
            beta: t(&[1, h, nt, ns]),
            q: ctx.view_4d(conv, [S, nk, nt, ns], nb, 0),
            k: ctx.view_4d(conv, [S, nk, nt, ns], nb, (S * nk) as usize * es),
            v: ctx.view_4d(conv, [S, h, nt, ns], nb, (2 * S * nk) as usize * es),
        }
    }

    /// Unit q and k rows per head (as the l2 norms leave them), g < 0 (a decay), beta in (0, 1).
    fn fill(&self, nk: i64) {
        let row = self.conv.ne()[0] as usize;
        let mut x = noise(self.conv.elements(), 1);
        for tok in x.chunks_mut(row) {
            for head in tok[..(2 * S * nk) as usize].chunks_mut(S as usize) {
                let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
                head.iter_mut().for_each(|v| *v /= n);
            }
        }
        self.conv.set_f32(&x);
        let n = self.g.elements();
        let map = |seed, f: fn(f32) -> f32| noise(n, seed).into_iter().map(f).collect::<Vec<_>>();
        self.g.set_f32(&map(2, |v| 0.5 * (v - 1.0)));
        self.beta.set_f32(&map(3, |v| 0.5 * (v + 1.0)));
    }

    /// ggml's op (`ours` false) or [`gdn_mma`]'s on `state`.
    fn op(&self, ctx: &'a Context, ours: bool, state: Tensor<'a>) -> Tensor<'a> {
        let Self {
            q, k, v, g, beta, ..
        } = *self;
        if ours {
            gdn_mma::gdn_mma(ctx, q, k, v, g, beta, state)
        } else {
            ctx.gated_delta_net(q, k, v, g, beta, state, 1)
        }
    }
}

/// Start states: small values, the same for every layer and kernel.
fn states(n: usize) -> Vec<f32> {
    noise(n, 4).iter().map(|v| 0.1 * v).collect()
}

/// The values of ggml's op and of each kernel in [`KERNELS`] over one layer's inputs.
fn outputs(
    gpu: &mut Backend,
    heads: (i64, i64),
    nt: i64,
    ns: i64,
) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
    let err = |e: GgmlError| format!("gdn-bench: {e:?}");
    let (h, nk) = heads;
    let ctx = Context::new(64, 64);
    let x = Inputs::new(&ctx, heads, nt, ns);
    let st = ctx.new_tensor(types::F32, &[S, S, h, ns]).as_input();
    let reference = x.op(&ctx, false, st).as_output();
    let ours = x.op(&ctx, true, st).as_output();
    let (mut a, mut b) = (ctx.new_graph(16), ctx.new_graph(16));
    a.expand(reference);
    b.expand(ours);
    let mut both = ctx.new_graph(32);
    both.expand(reference);
    both.expand(ours);
    ctx.alloc_graph(&both, ComputeAllocator::new(gpu.buffer_type()))
        .map_err(err)?;
    x.fill(nk);
    st.set_f32(&states(st.elements()));
    gpu.compute(&a).map_err(err)?;
    let r = reference.to_f32();
    let mut outs = Vec::new();
    for s in KERNELS {
        gdn_mma::force_schedule(Some(s));
        let run = gpu.compute(&b);
        gdn_mma::force_schedule(None);
        run.map_err(err)?;
        outs.push(ours.to_f32());
    }
    Ok((r, outs))
}

fn differ(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Bits of every kernel against ggml's for each of `tokens` (one sequence) and for a two-sequence
/// group with a chunk tail; times per layer for `tokens[0]`.
pub fn run_prompt(
    gpu: &mut Backend,
    cfg: &Config,
    tokens: &[i64],
) -> Result<Vec<PromptRow>, String> {
    let err = |e: GgmlError| format!("gdn-bench: {e:?}");
    let heads = cfg.heads;
    let (h, nk) = heads;
    let mut rows = Vec::new();
    let cases = tokens
        .iter()
        .map(|&nt| (nt, 1))
        .chain(std::iter::once((70, 2)));
    for (i, (nt, ns)) in cases.enumerate() {
        let (r, outs) = outputs(gpu, heads, nt, ns)?;
        let differ: Vec<usize> = outs.iter().map(|o| differ(&r, o)).collect();
        let mut secs = vec![0.0; KERNELS.len() + 1];
        if i == 0 {
            // ggml's graph and one per kernel (each built and run with its kernel forced).
            let n = 4 * cfg.layers + 16;
            let ctxs: Vec<Context> = (0..=KERNELS.len()).map(|_| Context::new(n, n)).collect();
            let mut graphs: Vec<Graph<'_>> = Vec::new();
            for (g, ctx) in ctxs.iter().enumerate() {
                let x = Inputs::new(ctx, heads, nt, 1);
                let sts: Vec<Tensor<'_>> = (0..cfg.layers)
                    .map(|_| {
                        ctx.new_tensor(types::F32, &[S, S, h, 1])
                            .as_input()
                            .as_output()
                    })
                    .collect();
                let mut graph = ctx.new_graph(n);
                for &st in &sts {
                    graph.expand(x.op(ctx, g > 0, st));
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(err)?;
                x.fill(nk);
                for st in &sts {
                    st.set_f32(&states(st.elements()));
                }
                graphs.push(graph);
            }
            let mut best = vec![f64::INFINITY; graphs.len()];
            for _ in 0..cfg.rounds {
                for (g, graph) in graphs.iter().enumerate() {
                    gdn_mma::force_schedule(g.checked_sub(1).map(|k| KERNELS[k]));
                    let mut timed = || -> Result<f64, GgmlError> {
                        gpu.compute(graph)?;
                        let t = std::time::Instant::now();
                        for _ in 0..cfg.reps {
                            gpu.compute(graph)?;
                        }
                        Ok(t.elapsed().as_secs_f64() / (cfg.reps * cfg.layers) as f64)
                    };
                    let s = timed();
                    gdn_mma::force_schedule(None);
                    best[g] = best[g].min(s.map_err(err)?);
                }
            }
            secs = best;
        }
        rows.push(PromptRow {
            tokens: nt,
            seqs: ns,
            secs,
            differ,
        });
    }
    Ok(rows)
}
