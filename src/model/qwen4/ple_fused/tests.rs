//! The PLE ops against ggml's chain as [`Model::ple`](super::super::Model) builds it for prompt
//! chunks: on the CPU within rounding (the CPU references), on CUDA (ignored tests) bit for
//! bit, the new streams and the carried histories alike, over consecutive chunks.

use std::cell::RefCell;

use super::super::{PleParams, ple_conv_rows};
use super::*;
use crate::ggml::{Backend, ComputeAllocator, Graph};
use crate::model::core::{cols, concat_cols};

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

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// qwen4exp's conv: kernel 4 dilated by the trigram size (a history of 9).
fn ple_params() -> PleParams {
    PleParams {
        layer: 1,
        ngram: 3,
        heads_per_ngram: 1,
        conv_kernel: 4,
        eos: 0,
        head_dim: 1,
        multipliers: vec![1; 3],
        offsets: Vec::new(),
        vocab_sizes: Vec::new(),
        blocks: None,
        fp8_scale: None,
    }
}

const EPS: f32 = 1e-6;

/// The weights and inputs of one step.
struct Case {
    e: i64,
    hc: i64,
    groups: Vec<(i64, i64)>,
    key: Vec<f32>,
    res: Vec<f32>,
    value: Vec<f32>,
    gammas: [Vec<f32>; 3],
    w: Vec<f32>,
}

impl Case {
    fn new(e: i64, hc: i64, groups: &[(i64, i64)], seed: u64) -> Self {
        let n: i64 = groups.iter().map(|(nt, ns)| nt * ns).sum();
        let k = (e * hc) as usize;
        // The gammas around 1, as trained norms are; the key and the value scaled apart so the
        // gates spread over (0, 1).
        let gamma = |s| noise(k, s).iter().map(|v| 1.0 + 0.5 * v).collect();
        Self {
            e,
            hc,
            groups: groups.to_vec(),
            key: noise(k * n as usize, seed)
                .iter()
                .map(|v| 3.0 * v)
                .collect(),
            res: noise(k * n as usize, seed + 1),
            value: noise(e as usize * n as usize, seed + 2),
            gammas: [gamma(seed + 3), gamma(seed + 4), gamma(seed + 5)],
            w: noise(4 * k, seed + 6),
        }
    }

    fn n(&self) -> i64 {
        self.groups.iter().map(|(nt, ns)| nt * ns).sum()
    }
}

/// ggml's chain of [`Model::ple`] after the matmuls, for prompt groups `(nt, ns)` with their
/// histories `states`.
#[allow(clippy::too_many_arguments)]
fn chain<'a>(
    ctx: &'a Context,
    graph: &RefCell<Graph<'a>>,
    (key, res, value): (Tensor<'a>, Tensor<'a>, Tensor<'a>),
    gammas: [Tensor<'a>; 3],
    w: Tensor<'a>,
    states: &[Tensor<'a>],
    groups: &[(i64, i64)],
    ple: &PleParams,
) -> Tensor<'a> {
    let [e, hc, n, _] = res.ne();
    let hc_dim = e * hc;
    let grouped_norm = |x: Tensor<'a>, g: Tensor<'a>| {
        let x = ctx.reshape_3d(x, e, hc, n);
        ctx.mul(ctx.rms_norm(x, EPS), ctx.reshape_2d(g, e, hc))
    };
    let key = grouped_norm(key, gammas[0]);
    let query = grouped_norm(res, gammas[1]);
    let s = ctx.sum_rows(ctx.mul(key, query));
    let s = ctx.scale(s, 1.0 / (e as f32).sqrt());
    let mag = ctx.sqrt(ctx.clamp(ctx.abs(s), 1e-6, f32::INFINITY));
    let gate = ctx.sigmoid(ctx.mul(ctx.sgn(s), mag));
    let v3 = ctx.repeat_4d(ctx.reshape_3d(value, e, 1, n), [e, hc, n, 1]);
    let gated = ctx.mul(v3, gate);
    let normalized = grouped_norm(ctx.reshape_2d(gated, hc_dim, n), gammas[2]);
    let normalized = ctx.reshape_2d(normalized, hc_dim, n);
    let mut outs = Vec::new();
    let mut off = 0;
    for (&(nt, ns), &state) in groups.iter().zip(states) {
        let x = ctx.reshape_3d(cols(ctx, normalized, off, nt * ns), hc_dim, nt, ns);
        outs.push(ple_conv_rows(ctx, graph, w, state, x, ple));
        off += nt * ns;
    }
    let conv = ctx.silu(concat_cols(ctx, outs));
    let conv = ctx.reshape_3d(ctx.cont(conv), e, hc, n);
    ctx.add(res, ctx.add(gated, conv))
}

/// One step through the ops and the chain, each from its own copy of the histories `hist`:
/// (ops streams, ops histories, chain streams, chain histories, ops gates).
type Step = (Vec<f32>, Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>);
fn step(backend: &mut Backend, c: &Case, hist: &[Vec<f32>]) -> Step {
    let ple = ple_params();
    let (e, hc, n) = (c.e, c.hc, c.n());
    let k = e * hc;
    let ctx = Context::new(1024, 1024);
    let input = |shape: &[i64]| ctx.new_tensor(types::F32, shape).as_input();
    let key = input(&[k, n]);
    let res = input(&[e, hc, n]);
    let value = input(&[e, n]);
    let gammas = [input(&[k]), input(&[k]), input(&[k])];
    let w = input(&[4, k]);
    let states = |_| -> Vec<Tensor<'_>> {
        c.groups
            .iter()
            .map(|&(_, ns)| input(&[ple.history(), k, ns]).as_output())
            .collect()
    };
    let (sa, sb) = (states(0), states(1));
    let graph = RefCell::new(ctx.new_graph(1024));
    let ob = chain(
        &ctx,
        &graph,
        (key, res, value),
        gammas,
        w,
        &sb,
        &c.groups,
        &ple,
    );
    let ob = ctx.cont(ob).as_output();
    let (xn, gate) = gate_norm(
        &ctx,
        key,
        res,
        value,
        gammas,
        EPS,
        1.0 / (e as f32).sqrt(),
        1e-6,
    );
    let gate_out = ctx.cont(gate).as_output();
    let streams = ctx.view_2d(res, [k, n], res.nb()[2], 0);
    let mut outs = Vec::new();
    let mut off = 0;
    for (&(nt, ns), &state) in c.groups.iter().zip(&sa) {
        let m = nt * ns;
        outs.push(conv_out(
            &ctx,
            cols(&ctx, xn, off, m),
            state,
            w,
            cols(&ctx, value, off, m),
            cols(&ctx, gate, off, m),
            cols(&ctx, streams, off, m),
            nt,
            ple.ngram as i64,
        ));
        off += m;
    }
    let oa = ctx.cont(concat_cols(&ctx, outs)).as_output();
    let mut graph = graph.into_inner();
    for t in [ob, oa, gate_out] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    key.set_f32(&c.key);
    res.set_f32(&c.res);
    value.set_f32(&c.value);
    for (t, v) in gammas.iter().zip(&c.gammas) {
        t.set_f32(v);
    }
    w.set_f32(&c.w);
    for (i, h) in hist.iter().enumerate() {
        sa[i].set_f32(h);
        sb[i].set_f32(h);
    }
    backend.compute(&graph).unwrap();
    let read = |s: &[Tensor<'_>]| s.iter().map(|t| t.to_f32()).collect();
    (
        oa.to_f32(),
        read(&sa),
        ob.to_f32(),
        read(&sb),
        gate_out.to_f32(),
    )
}

/// Two chunks in turn (the second from the histories the first left) through both paths;
/// `check(what, ops, chain)` on every output and history.
type Check<'c> = &'c dyn Fn(&str, &[f32], &[f32]);
fn run(backend: &mut Backend, e: i64, groups: &[(i64, i64)], check: Check<'_>) {
    let ple = ple_params();
    let mut hist: Vec<Vec<f32>> = groups
        .iter()
        .enumerate()
        .map(|(i, &(_, ns))| noise((ple.history() * e * 4 * ns) as usize, 50 + i as u64))
        .collect();
    for chunk in 0..2 {
        let c = Case::new(e, 4, groups, 100 * chunk + 7);
        let (oa, ha, ob, hb, gates) = step(backend, &c, &hist);
        // The gates spread: neither all saturated nor all equal.
        assert!(
            gates.iter().any(|&g| g > 0.05 && g < 0.95),
            "gates {gates:?}"
        );
        check(&format!("streams, chunk {chunk}"), &oa, &ob);
        for (i, (a, b)) in ha.iter().zip(&hb).enumerate() {
            check(&format!("history {i}, chunk {chunk}"), a, b);
            assert!(!same_bits(a, &hist[i]), "history {i} moved, chunk {chunk}");
        }
        hist = ha;
    }
}

#[test]
fn ple_ops_match_the_chain_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let near = |what: &str, a: &[f32], b: &[f32]| {
        let r = rel(a, b);
        assert!(r < 1e-5, "{what}: rel {r:e}");
    };
    // A chunk longer than the history; one shorter; several groups and sequences.
    for groups in [&[(13, 1)][..], &[(5, 1)], &[(7, 1), (4, 2), (12, 1)]] {
        run(&mut backend, 64, groups, &near);
    }
}

/// On CUDA the ops compute what ggml-cuda's kernels for the chain compute, bit for bit, with
/// both of sum_rows' blocks (512 threads below two rows per SM, 128 from there).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_ple_ops_are_bit_identical_to_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let exact = |what: &str, a: &[f32], b: &[f32]| {
        assert!(same_bits(a, b), "{what}: differs (rel {:e})", rel(a, b));
    };
    for groups in [
        &[(4096, 1)][..],
        &[(300, 1)],
        &[(16, 1)],
        &[(5, 1)],
        &[(7, 1), (4, 2), (37, 1)],
        &[(200, 2), (90, 1)],
    ] {
        run(&mut gpu, 2560, groups, &exact);
    }
}
