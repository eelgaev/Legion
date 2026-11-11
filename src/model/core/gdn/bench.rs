//! [`decode`] (#836) against ggml's `gated_delta_net` on one GPU: Qwen3.8-Flash-Next's one-token
//! steps (48 value heads over 16 q / k heads), each layer with its own states so nothing stays in
//! L2. Checks that every kernel variant gives ggml's bits (rows and final states) and times them.
//! Used by the hidden `legion gdn-bench` subcommand (a build tree's binary on the GPU box).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

mod prompt;
mod replay;
pub use prompt::{PROMPT_KERNELS, PromptRow, run_prompt};
pub use replay::{REPLAY_VARIANTS, REPLAY_VERIFY, ReplayRow, run_replay};

/// The kernel variants compared (`src/cuda/gdn.cu`'s `GDN_DECODE_*` bits), the default (-1) first.
pub const VARIANTS: [(i32, &str); 9] = [
    (-1, "default"),
    (0, "head grid"),
    (1, "linear"),
    (2, "head w8"),
    (3, "linear w8"),
    (5, "linear async"),
    (9, "linear stream"),
    (11, "linear w8 stream"),
    (13, "linear async stream"),
];

/// What to run.
#[derive(Clone, Debug)]
pub struct Config {
    /// Value heads and q / k heads.
    pub heads: (i64, i64),
    /// Sequences (one token each) per step.
    pub seqs: Vec<i64>,
    /// Layers per graph, each with its own states.
    pub layers: usize,
    /// Timed rounds (the best counts; the kernels take turns within a round).
    pub rounds: usize,
    /// Graph runs per timing.
    pub reps: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            heads: (48, 16),
            seqs: vec![1, 4, 16],
            layers: 36,
            rounds: 5,
            reps: 20,
        }
    }
}

/// One step size: ggml's time and each variant's, per layer.
#[derive(Clone, Debug)]
pub struct Row {
    pub seqs: i64,
    /// State bytes read and written per layer.
    pub bytes: f64,
    /// Seconds per layer: ggml's op, then [`VARIANTS`] in order.
    pub secs: Vec<f64>,
    /// Output values whose bits differ from ggml's, per variant.
    pub differ: Vec<usize>,
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

/// The graph inputs of one step: q, k, v, g, beta (shared by the layers).
#[derive(Clone, Copy)]
struct Inputs<'a> {
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
}

impl<'a> Inputs<'a> {
    fn new(ctx: &'a Context, h: i64, nk: i64, ns: i64) -> Self {
        // Inputs are outputs too: the allocator would hand their memory to later nodes.
        let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input().as_output();
        Self {
            q: t(&[S, nk, 1, ns]),
            k: t(&[S, nk, 1, ns]),
            v: t(&[S, h, 1, ns]),
            g: t(&[1, h, 1, ns]),
            beta: t(&[1, h, 1, ns]),
        }
    }

    fn fill(&self) {
        let scaled = |t: Tensor<'_>, seed: u64, a: f32| {
            t.set_f32(
                &noise(t.elements(), seed)
                    .iter()
                    .map(|x| x * a)
                    .collect::<Vec<_>>(),
            )
        };
        // q, k about unit length per head (as l2_norm leaves them).
        scaled(self.q, 1, 0.15);
        scaled(self.k, 2, 0.15);
        scaled(self.v, 3, 1.0);
        let n = self.g.elements();
        // Log decays in (-1, 0], betas in (0, 1).
        self.g.set_f32(
            &noise(n, 4)
                .iter()
                .map(|x| -0.5 * (x + 1.0))
                .collect::<Vec<_>>(),
        );
        self.beta.set_f32(
            &noise(n, 5)
                .iter()
                .map(|x| 0.5 * (x + 1.0))
                .collect::<Vec<_>>(),
        );
    }

    fn op(&self, ctx: &'a Context, ours: bool, state: Tensor<'a>) -> Tensor<'a> {
        let Self { q, k, v, g, beta } = *self;
        if ours {
            decode(ctx, q, k, v, g, beta, state)
        } else {
            ctx.gated_delta_net(q, k, v, g, beta, state, 1)
        }
    }
}

/// ggml's output and the op's (variant `variant`) over the same inputs and states.
fn outputs(gpu: &mut Backend, h: i64, nk: i64, ns: i64, variant: i32) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let x = Inputs::new(&ctx, h, nk, ns);
    let st_ggml = ctx.new_tensor(types::F32, &[S, S, h, ns]).as_input();
    let st_op = ctx.new_tensor(types::F32, &[S, S, h, ns]).as_input();
    let reference = x.op(&ctx, false, st_ggml).as_output();
    let ours = x.op(&ctx, true, st_op).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(reference);
    graph.expand(ours);
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    x.fill();
    let s0: Vec<f32> = noise((S * S * h * ns) as usize, 6)
        .iter()
        .map(|v| v * 0.1)
        .collect();
    st_ggml.set_f32(&s0);
    st_op.set_f32(&s0);
    set_decode_variant(variant);
    gpu.compute(&graph).unwrap();
    set_decode_variant(-1);
    (reference.to_f32(), ours.to_f32())
}

/// Runs the comparison.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    let (h, nk) = cfg.heads;
    let mut rows = Vec::new();
    for &ns in &cfg.seqs {
        let differ = VARIANTS
            .iter()
            .map(|&(v, _)| {
                let (a, b) = outputs(gpu, h, nk, ns, v);
                assert_eq!(a.len(), b.len());
                a.iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            })
            .collect();
        // One graph per kernel (ggml's op, the op), each in its own context; every layer has
        // its own states.
        let n = 8 * cfg.layers + 16;
        let ctxs = [Context::new(n, n), Context::new(n, n)];
        let mut graphs = Vec::new();
        for (ours, ctx) in [false, true].into_iter().zip(&ctxs) {
            let x = Inputs::new(ctx, h, nk, ns);
            let states: Vec<Tensor<'_>> = (0..cfg.layers)
                .map(|_| {
                    ctx.new_tensor(types::F32, &[S, S, h, ns])
                        .as_input()
                        .as_output()
                })
                .collect();
            let mut graph = ctx.new_graph(n);
            for &st in &states {
                graph.expand(x.op(ctx, ours, st));
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(|e| format!("gdn-bench: {e:?}"))?;
            x.fill();
            for st in &states {
                st.set_f32(&vec![0.0; st.elements()]);
            }
            graphs.push(graph);
        }
        // ggml, then each variant.
        let kernels: Vec<(usize, i32)> = std::iter::once((0, -1))
            .chain(VARIANTS.iter().map(|&(v, _)| (1, v)))
            .collect();
        let mut best = vec![f64::INFINITY; kernels.len()];
        for _ in 0..cfg.rounds {
            for (i, &(g, v)) in kernels.iter().enumerate() {
                set_decode_variant(v);
                gpu.compute(&graphs[g]).map_err(|e| format!("{e:?}"))?;
                let t = std::time::Instant::now();
                for _ in 0..cfg.reps {
                    gpu.compute(&graphs[g]).map_err(|e| format!("{e:?}"))?;
                }
                let s = t.elapsed().as_secs_f64() / (cfg.reps * cfg.layers) as f64;
                best[i] = best[i].min(s);
            }
        }
        set_decode_variant(-1);
        rows.push(Row {
            seqs: ns,
            bytes: (2 * S * S * h * ns * 4) as f64,
            secs: best,
            differ,
        });
    }
    Ok(rows)
}

/// One set of keep counts of a replay commit ([`run_commit`]).
#[derive(Clone, Debug)]
pub struct CommitRow {
    /// Tokens each slot keeps (0: idle).
    pub keep: Vec<i32>,
    /// State values whose bits differ from ggml's (its op per slot over the kept tokens).
    pub differ: usize,
    /// Seconds per layer: [`commit_slots`] (one launch), then [`commit`] per slot.
    pub slots: f64,
    pub per_slot: f64,
}

/// The keep counts [`run_commit`] measures over 16 slots: C=16 verify steps of depth 3-5, a C=4
/// step among 16 slots, every slot keeping one token, and every count the kernel unrolls.
pub const COMMIT_KEEPS: [[i32; 16]; 5] = [
    [3, 1, 2, 4, 1, 3, 2, 2, 4, 1, 3, 2, 1, 2, 3, 1],
    [5, 2, 3, 1, 4, 6, 2, 3, 1, 2, 5, 3, 2, 4, 1, 3],
    [0, 3, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 0],
    [1; 16],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
];

/// Verify-pool tokens per slot in [`run_commit`] (the server's for depth 8 plus one).
const COMMIT_POOL: i64 = 9;

/// A replay commit's graph ([`commit_graph`]) and its tensors.
struct CommitGraph<'a> {
    graph: crate::ggml::Graph<'a>,
    inputs: Inputs<'a>,
    states: Vec<Tensor<'a>>,
    /// The keep input (the all-slot graph only).
    keep: Option<Tensor<'a>>,
}

/// The graph of one replay commit over `layers` layers with their own states, from pools of
/// [`COMMIT_POOL`] tokens: all slots at once (`how` "slots") or one [`commit`] per slot keeping
/// tokens ("per_slot"); "ggml": ggml's op per slot and the copy of its final state.
fn commit_graph<'a>(
    ctx: &'a Context,
    h: i64,
    nk: i64,
    keep: &[i32],
    how: &str,
    layers: usize,
) -> CommitGraph<'a> {
    let ns = keep.len() as i64;
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input().as_output();
    let nt = COMMIT_POOL;
    let inputs = Inputs {
        q: t(&[S, nk, nt, ns]),
        k: t(&[S, nk, nt, ns]),
        v: t(&[S, h, nt, ns]),
        g: t(&[1, h, nt, ns]),
        beta: t(&[1, h, nt, ns]),
    };
    let Inputs { q, k, v, g, beta } = inputs;
    let keep_t = ctx.new_tensor(types::I32, &[ns]).as_input().as_output();
    let mut graph = ctx.new_graph(ctx_nodes(keep.len(), layers));
    let mut states = Vec::new();
    for _ in 0..layers {
        let st = t(&[S, S, h, ns]);
        states.push(st);
        if how == "slots" {
            graph.expand(commit_slots(ctx, q, k, v, g, beta, st, keep_t));
            continue;
        }
        for (s, &kp) in keep.iter().enumerate() {
            if kp == 0 {
                continue;
            }
            let kp = kp as i64;
            let view = |x: Tensor<'a>| {
                let [a, b, _, _] = x.ne();
                let nb = x.nb();
                ctx.view_4d(x, [a, b, kp, 1], [nb[1], nb[2], nb[3]], s * nb[3])
            };
            let nb = st.nb();
            let sv = ctx.view_4d(st, [S, S, h, 1], [nb[1], nb[2], nb[3]], s * nb[3]);
            if how == "ggml" {
                let out =
                    ctx.gated_delta_net(view(q), view(k), view(v), view(g), view(beta), sv, 1);
                let es = out.element_size();
                let fin = ctx.view_4d(
                    out,
                    sv.ne(),
                    [
                        S as usize * es,
                        (S * S) as usize * es,
                        (S * S * h) as usize * es,
                    ],
                    (S * h * kp) as usize * es,
                );
                graph.expand(ctx.cpy(fin, sv));
            } else {
                graph.expand(commit(
                    ctx,
                    view(q),
                    view(k),
                    view(v),
                    view(g),
                    view(beta),
                    sv,
                ));
            }
        }
    }
    CommitGraph {
        graph,
        inputs,
        states,
        keep: (how == "slots").then_some(keep_t),
    }
}

fn ctx_nodes(slots: usize, layers: usize) -> usize {
    16 * slots * layers + 64
}

impl CommitGraph<'_> {
    fn fill(&self, keep: &[i32]) {
        self.inputs.fill();
        let s0: Vec<f32> = noise(self.states[0].elements(), 6)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        for st in &self.states {
            st.set_f32(&s0);
        }
        if let Some(t) = self.keep {
            t.set_i32(keep);
        }
    }
}

/// [`commit_slots`] against ggml's op per slot over [`COMMIT_KEEPS`] (bits), and its time
/// against one [`commit`] launch per slot keeping tokens (the graphs per set of runs), per layer
/// over `layers` layers.
pub fn run_commit(gpu: &mut Backend, cfg: &Config) -> Result<Vec<CommitRow>, String> {
    let (h, nk) = cfg.heads;
    let err = |e: crate::ggml::GgmlError| format!("gdn-bench: {e:?}");
    let mut rows = Vec::new();
    for keep in COMMIT_KEEPS {
        // Bits: one layer, ggml's op per slot and ours over all slots.
        let differ = {
            let n = ctx_nodes(keep.len(), 1);
            let (ca, cb) = (Context::new(n, n), Context::new(n, n));
            let a = commit_graph(&ca, h, nk, &keep, "ggml", 1);
            let b = commit_graph(&cb, h, nk, &keep, "slots", 1);
            for (c, x) in [(&ca, &a), (&cb, &b)] {
                c.alloc_graph(&x.graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(err)?;
                x.fill(&keep);
                gpu.compute(&x.graph).map_err(err)?;
            }
            let (ra, rb) = (a.states[0].to_f32(), b.states[0].to_f32());
            ra.iter()
                .zip(&rb)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        let n = ctx_nodes(keep.len(), cfg.layers);
        let ctxs = [Context::new(n, n), Context::new(n, n)];
        let mut graphs = Vec::new();
        for (how, ctx) in ["slots", "per_slot"].into_iter().zip(&ctxs) {
            let x = commit_graph(ctx, h, nk, &keep, how, cfg.layers);
            ctx.alloc_graph(&x.graph, ComputeAllocator::new(gpu.buffer_type()))
                .map_err(err)?;
            x.fill(&keep);
            graphs.push(x);
        }
        let mut best = [f64::INFINITY; 2];
        for _ in 0..cfg.rounds {
            for (i, x) in graphs.iter().enumerate() {
                gpu.compute(&x.graph).map_err(err)?;
                let t = std::time::Instant::now();
                for _ in 0..cfg.reps {
                    gpu.compute(&x.graph).map_err(err)?;
                }
                let s = t.elapsed().as_secs_f64() / (cfg.reps * cfg.layers) as f64;
                best[i] = best[i].min(s);
            }
        }
        rows.push(CommitRow {
            keep: keep.to_vec(),
            differ,
            slots: best[0],
            per_slot: best[1],
        });
    }
    Ok(rows)
}
