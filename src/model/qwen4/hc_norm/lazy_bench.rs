//! Lazy streams (#1183, src/model/core/node_fuse/fold.rs) on one GPU: qwen4exp's prompt-step
//! layers as the fold's relayout sees them (each layer's attention mixer norm, its up-mix and a
//! stand-in attention block, the FFN mixer norm, its down matmul and up-mix, and a MoE tail the
//! next layer's norm folds), allocated with the relayout as the engine allocates a CUDA graph:
//! main's graph (the FFN mixer norms write their streams) against the lazy one (their readers
//! make them), timed, and every output compared bit for bit. Used by the hidden
//! `legion hc-lazy-bench` subcommand (a build tree's binary on the GPU box); the unit tests build
//! the same graph on the CPU.

use super::super::{hc_chain, hc_up_mix};
use super::{P_NO_RES, hc_post_norm_rebuilt_xnb, set_hc_lazy_streams, set_moe_fold};
use crate::ggml::{Backend, ComputeAllocator, Context, Graph, Tensor, ffi, types};
use crate::model::nvfp4_mm::bf16_bits;

/// What [`run`] measures.
#[derive(Clone, Debug)]
pub struct Config {
    /// Values per stream (qwen4exp: 2,560 in 4 streams).
    pub e: i64,
    /// The mixers' low rank (qwen4exp: 320).
    pub rank: i64,
    /// Routed experts a token (the tail's sum; fewer than the model's, its cost is the same in
    /// both graphs).
    pub k: i64,
    /// Tokens of the step.
    pub tokens: Vec<i64>,
    /// Layers a graph chains.
    pub layers: usize,
    pub rounds: usize,
}

impl Default for Config {
    /// Qwen3.8-Flash-Next's mixers at prompt steps of 2K, 4K and 8K tokens.
    fn default() -> Self {
        Self {
            e: 2560,
            rank: 320,
            k: 4,
            tokens: vec![2048, 4096, 8192],
            layers: 4,
            rounds: 8,
        }
    }
}

/// One graph at one token count.
#[derive(Clone, Debug)]
pub struct Row {
    pub tokens: i64,
    /// The lazy graph (else main's).
    pub lazy: bool,
    /// Best time per layer over the rounds (seconds).
    pub secs: f64,
    /// FFN mixer norms whose streams their readers make.
    pub made: usize,
    /// Compute buffer bytes.
    pub buffer: usize,
    /// View nodes nothing reads ([`unread_views`]; each holds memory to the graph's end).
    pub unread: usize,
    /// Output values compared with main's graph, and those that differ (bit for bit).
    pub values: usize,
    pub differing: usize,
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

fn bf16_bytes(x: impl Iterator<Item = f32>) -> Vec<u8> {
    x.flat_map(|v| bf16_bits(v).to_ne_bytes()).collect()
}

/// One mixer's weights: gamma, inject weights, down and up.
type Mixer<'a> = [Tensor<'a>; 4];

/// Each layer's two mixers' weights (the attention mixer's, the FFN mixer's), in `ctx`, filled.
pub(super) fn weights<'a>(
    ctx: &'a Context,
    backend: &Backend,
    cfg: &Config,
) -> Result<Vec<[Mixer<'a>; 2]>, String> {
    let kk = cfg.e * 4;
    let mixer = || {
        [
            ctx.new_tensor(types::F32, &[kk]),
            ctx.new_tensor(types::F32, &[kk, 4]),
            ctx.new_tensor(types::BF16, &[kk, cfg.rank]),
            ctx.new_tensor(types::BF16, &[cfg.rank, kk]),
        ]
    };
    let sets: Vec<[Mixer<'a>; 2]> = (0..cfg.layers).map(|_| [mixer(), mixer()]).collect();
    ctx.alloc_tensors(backend.buffer_type())
        .map_err(|e| e.to_string())?;
    for (i, [g, w, down, up]) in sets.iter().flatten().enumerate() {
        let s = 10 * i as u64 + 1000;
        g.set_f32(
            &noise(kk as usize, s + 1)
                .iter()
                .map(|v| 1.0 + 0.2 * v)
                .collect::<Vec<_>>(),
        );
        w.set_f32(
            &noise((kk * 4) as usize, s + 2)
                .iter()
                .map(|v| v * 0.01)
                .collect::<Vec<_>>(),
        );
        down.set_bytes(&bf16_bytes(
            noise((kk * cfg.rank) as usize, s + 3)
                .into_iter()
                .map(|v| 0.02 * v),
        ));
        up.set_bytes(&bf16_bytes(
            noise((kk * cfg.rank) as usize, s + 4)
                .into_iter()
                .map(|v| 0.05 * v),
        ));
    }
    Ok(sets)
}

/// A step's inputs, also marked as outputs: ggml's allocator would otherwise hand their memory
/// to later nodes once read, and the graph runs again without refilling them.
pub(super) struct Inputs<'a> {
    /// The first norm's streams, block output and inject weights.
    res: Tensor<'a>,
    out: Tensor<'a>,
    inj: Tensor<'a>,
    /// Added to the attention mixer's up-mix as the attention block's output.
    attn: Tensor<'a>,
    /// The MoE tail's experts' rows, router weights, shared expert (the FFN mixer's up-mix is
    /// added) and gate logits (three a token, the gate's the second).
    experts: Tensor<'a>,
    wr: Tensor<'a>,
    sh: Tensor<'a>,
    joined: Tensor<'a>,
}

pub(super) fn inputs<'a>(ctx: &'a Context, cfg: &Config, n: i64) -> Inputs<'a> {
    let t = |ne: &[i64]| ctx.new_tensor(types::F32, ne).as_input().as_output();
    let e = cfg.e;
    Inputs {
        res: t(&[e, 4, n]),
        out: t(&[e, n]),
        inj: t(&[4, n]),
        attn: t(&[e, n]),
        experts: t(&[e, cfg.k, n]),
        wr: t(&[1, cfg.k, n]),
        sh: t(&[e, n]),
        joined: t(&[3, n]),
    }
}

pub(super) fn fill(i: &Inputs<'_>) {
    let n = |t: Tensor<'_>| t.elements();
    i.res.set_f32(
        &noise(n(i.res), 91)
            .iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>(),
    );
    i.out.set_f32(&noise(n(i.out), 92));
    i.inj.set_f32(&noise(n(i.inj), 93));
    i.attn.set_f32(&noise(n(i.attn), 94));
    i.experts.set_f32(&noise(n(i.experts), 95));
    i.wr.set_f32(
        &noise(n(i.wr), 96)
            .iter()
            .map(|v| v.abs() + 0.05)
            .collect::<Vec<_>>(),
    );
    i.sh.set_f32(&noise(n(i.sh), 97));
    i.joined.set_f32(
        &noise(n(i.joined), 98)
            .iter()
            .map(|v| v * 8.0)
            .collect::<Vec<_>>(),
    );
}

/// A built chain: the graph, its outputs (the last norm's streams, scales, inject weights and
/// BF16 xn as contiguous copies, and every layer's FFN mixer up-mix), and each layer's FFN
/// mixer streams and up-mix node (for [`made`]).
pub(super) struct Chain<'a> {
    pub graph: Graph<'a>,
    pub outs: Vec<Tensor<'a>>,
    pub ffn: Vec<(Tensor<'a>, Tensor<'a>)>,
}

/// The layers of `w` chained over `i` in `ctx` (see the module doc).
pub(super) fn build<'a>(ctx: &'a Context, w: &[[Mixer<'a>; 2]], i: &Inputs<'a>) -> Chain<'a> {
    let norm = |res, out, inj, m: &Mixer<'a>| {
        hc_post_norm_rebuilt_xnb(ctx, res, out, inj, m[0], m[1], 1e-6, m[2], false)
    };
    // The mixer's down matmul, activation and BF16 up-mix, as qwen4exp's step makes them.
    let mix = |p: &super::PostNormRebuiltXnb<'a>, m: &Mixer<'a>| {
        let lo = hc_chain::lo_act(ctx, ctx.mul_mat(m[2], p.xnb), 0.25, hc_chain::on_gpu(m[3]));
        hc_up_mix::up_mix_rebuilt(ctx, m[3], lo, p.res, p.scales, m[0], 4, 0.25)
    };
    let [e, k, n, _] = i.experts.ne();
    let mut outs = Vec::new();
    let mut ffn = Vec::new();
    let (mut res, mut out, mut inj) = (i.res, i.out, i.inj);
    for [a, f] in w {
        let pa = norm(res, out, inj, a);
        let oa = ctx.add(mix(&pa, a), i.attn);
        let pf = norm(pa.res, oa, pa.inject, f);
        let mixed = mix(&pf, f);
        // The MoE tail (#836's shape): the routed experts' weighted sum and the shared
        // expert's gate.
        let weighted = ctx.mul(i.experts, i.wr);
        let nb = weighted.nb();
        let view = |j: i64| ctx.view_2d(weighted, [e, n], nb[2], j as usize * nb[1]);
        let mut sum = ctx.add(view(0), view(1));
        for j in 2..k {
            sum = ctx.add(sum, view(j));
        }
        let sg = ctx.view_2d(i.joined, [1, n], i.joined.nb()[1], 4);
        let sh = ctx.add(i.sh, mixed);
        out = ctx.add(sum, ctx.mul(sh, ctx.sigmoid(sg)));
        outs.push(mixed.as_output());
        ffn.push((pf.res, mixed));
        (res, inj) = (pf.res, pf.inject);
    }
    let last = norm(res, out, inj, &w[0][0]);
    for t in [last.res, last.scales, last.inject, last.xnb] {
        outs.push(ctx.cont(t).as_output());
    }
    let mut graph = ctx.new_graph(64 * w.len() + 64);
    for &t in &outs {
        graph.expand(t);
    }
    Chain { graph, outs, ffn }
}

/// The FFN mixer norms of an allocated chain whose streams their readers make: the norm writes
/// none and its up-mix reads the attention block's output.
pub(super) fn made(c: &Chain<'_>) -> usize {
    c.ffn
        .iter()
        .filter(|(res, mixed)| {
            // SAFETY: nodes of the allocated graph.
            unsafe {
                let nf = &*(*res.raw()).view_src;
                crate::ggml::external_params(nf)[P_NO_RES] != 0 && !(*mixed.raw()).src[6].is_null()
            }
        })
        .count()
}

/// View nodes of an allocated `graph` that run, are no output and that no node reads: ggml's
/// allocator keeps each one's source alive to the end of the graph.
pub(super) fn unread_views(graph: &Graph<'_>) -> usize {
    // SAFETY: an allocated graph whose node array holds valid tensors.
    unsafe {
        let n = ffi::ggml_graph_n_nodes(graph.raw()) as usize;
        let nodes = std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph.raw()), n);
        nodes
            .iter()
            .filter(|&&t| {
                (*t).op != ffi::ggml_op_GGML_OP_NONE
                    && !(*t).view_src.is_null()
                    && (*t).flags & ffi::ggml_tensor_flag_GGML_TENSOR_FLAG_OUTPUT as i32 == 0
                    && !nodes.iter().any(|&u| (*u).src.contains(&t))
            })
            .count()
    }
}

/// Graph contexts big enough for a chain of `layers`.
pub(super) fn context(layers: usize) -> Context {
    Context::new(96 * layers + 96, 64 * layers + 64)
}

/// Main's graph and the lazy one at every token count of `cfg` on `gpu`: each computed once and
/// its outputs compared, then timed in turns, the best round counting. Turns the MoE fold on
/// (and lazy streams back on) for the process.
pub fn run(gpu: &mut Backend, cfg: &Config) -> Result<Vec<Row>, String> {
    if cfg.layers == 0 || cfg.rounds == 0 || cfg.k < 2 || cfg.tokens.iter().any(|&n| n < 1) {
        return Err(format!(
            "layers > 0, rounds > 0, k >= 2, tokens > 0: {cfg:?}"
        ));
    }
    set_moe_fold(true);
    crate::ggml::post_alloc::set_relayout(crate::model::core::node_fuse::fold::RELAYOUT);
    let wctx = Context::new(8 * cfg.layers + 8, 0);
    let w = weights(&wctx, gpu, cfg)?;
    let mut rows = Vec::new();
    for &n in &cfg.tokens {
        let ctxs = [context(cfg.layers), context(cfg.layers)];
        let mut chains = Vec::new();
        for (ctx, lazy) in ctxs.iter().zip([false, true]) {
            let i = inputs(ctx, cfg, n);
            let c = build(ctx, &w, &i);
            set_hc_lazy_streams(lazy);
            let r = ctx.alloc_graph(&c.graph, ComputeAllocator::new(gpu.buffer_type()));
            set_hc_lazy_streams(true);
            r.map_err(|e| format!("{n} tokens: {e}"))?;
            fill(&i);
            gpu.compute(&c.graph)
                .map_err(|e| format!("{n} tokens: {e}"))?;
            chains.push((c, ctx.allocator_size()));
        }
        let want: Vec<Vec<u8>> = chains[0].0.outs.iter().map(|t| t.bytes()).collect();
        let mut best = [f64::MAX; 2];
        for round in 0..=cfg.rounds {
            for (vi, (c, _)) in chains.iter().enumerate() {
                let t = std::time::Instant::now();
                gpu.compute(&c.graph)
                    .map_err(|e| format!("{n} tokens: {e}"))?;
                if round > 0 {
                    best[vi] = best[vi].min(t.elapsed().as_secs_f64() / cfg.layers as f64);
                }
            }
        }
        for (vi, (c, buffer)) in chains.iter().enumerate() {
            let (mut values, mut differing) = (0, 0);
            for (t, want) in c.outs.iter().zip(&want) {
                let got = t.bytes();
                let size = got.len() / t.elements();
                values += t.elements();
                differing += got
                    .chunks(size)
                    .zip(want.chunks(size))
                    .filter(|(p, q)| p != q)
                    .count();
            }
            rows.push(Row {
                tokens: n,
                lazy: vi == 1,
                secs: best[vi],
                made: made(c),
                buffer: *buffer,
                unread: unread_views(&c.graph),
                values,
                differing,
            });
        }
    }
    Ok(rows)
}

/// A chain small enough for the CPU tests.
#[cfg(test)]
pub(super) fn tiny() -> Config {
    Config {
        e: 32,
        rank: 16,
        k: 3,
        tokens: vec![5],
        layers: 3,
        rounds: 1,
    }
}
