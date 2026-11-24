//! The PLE ops against ggml's chain as [`Model::ple`](super::super::Model) builds it for prompt
//! chunks: on the CPU within rounding (the CPU references), on CUDA (ignored tests) bit for
//! bit, the new streams and the carried histories alike, over consecutive chunks. Built
//! directly, and as graphs take them: the chain marked, allocated as built and rewritten
//! ([`super::rewrite`]).

use std::cell::RefCell;

use super::super::{PleParams, ple_conv_rows};
use super::rewrite::{GAMMAS, KEY, OUT, VALUE, X, XN};
use super::*;
use crate::ggml::{Backend, ComputeAllocator, Graph};
use crate::model::core::node_fuse::fold::RELAYOUT;
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
    /// The conv weight in BF16 (as the NVFP4 checkpoints ship it), `w` rounded to it.
    bf16_w: bool,
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
            bf16_w: false,
        }
    }

    fn n(&self) -> i64 {
        self.groups.iter().map(|(nt, ns)| nt * ns).sum()
    }
}

/// ggml's chain of [`Model::ple`] after the matmuls, for prompt groups `(nt, ns)` with their
/// histories `states`; `mark`: named as [`Model::ple`] names it for the PLE ops.
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
    mark: bool,
) -> Tensor<'a> {
    let [e, hc, n, _] = res.ne();
    let hc_dim = e * hc;
    let named = |t: Tensor<'a>, name: &str| if mark { t.named(name) } else { t };
    let (key, value) = (named(key, KEY), named(value, VALUE));
    let grouped_norm = |x: Tensor<'a>, g: usize| {
        let x = ctx.reshape_3d(x, e, hc, n);
        ctx.mul(
            ctx.rms_norm(x, EPS),
            named(ctx.reshape_2d(gammas[g], e, hc), GAMMAS[g]),
        )
    };
    let key = grouped_norm(key, 0);
    let query = grouped_norm(res, 1);
    let s = ctx.sum_rows(ctx.mul(key, query));
    let s = ctx.scale(s, 1.0 / (e as f32).sqrt());
    let mag = ctx.sqrt(ctx.clamp(ctx.abs(s), 1e-6, f32::INFINITY));
    let gate = ctx.sigmoid(ctx.mul(ctx.sgn(s), mag));
    let v3 = ctx.repeat_4d(ctx.reshape_3d(value, e, 1, n), [e, hc, n, 1]);
    let gated = ctx.mul(v3, gate);
    let normalized = named(grouped_norm(ctx.reshape_2d(gated, hc_dim, n), 2), XN);
    let normalized = ctx.reshape_2d(normalized, hc_dim, n);
    let mut outs = Vec::new();
    let mut off = 0;
    for (&(nt, ns), &state) in groups.iter().zip(states) {
        let x = named(
            ctx.reshape_3d(cols(ctx, normalized, off, nt * ns), hc_dim, nt, ns),
            X,
        );
        outs.push(ple_conv_rows(ctx, graph, w, state, x, ple, mark));
        off += nt * ns;
    }
    let conv = ctx.silu(concat_cols(ctx, outs));
    let conv = ctx.reshape_3d(ctx.cont(conv), e, hc, n);
    named(ctx.add(res, ctx.add(gated, conv)), OUT)
}

/// The inputs of one graph, for [`fill`].
struct Inputs<'a> {
    key: Tensor<'a>,
    res: Tensor<'a>,
    value: Tensor<'a>,
    gammas: [Tensor<'a>; 3],
    w: Tensor<'a>,
}

impl<'a> Inputs<'a> {
    fn new(ctx: &'a Context, c: &Case) -> Self {
        let (e, hc, n) = (c.e, c.hc, c.n());
        let k = e * hc;
        let input = |shape: &[i64]| ctx.new_tensor(types::F32, shape).as_input();
        let wty = if c.bf16_w { types::BF16 } else { types::F32 };
        Self {
            key: input(&[k, n]),
            res: input(&[e, hc, n]),
            value: input(&[e, n]),
            gammas: [input(&[k]), input(&[k]), input(&[k])],
            w: ctx.new_tensor(wty, &[4, k]).as_input(),
        }
    }

    /// Writes `c`'s values (the graph allocated).
    fn fill(&self, c: &Case) {
        self.key.set_f32(&c.key);
        self.res.set_f32(&c.res);
        self.value.set_f32(&c.value);
        for (t, v) in self.gammas.iter().zip(&c.gammas) {
            t.set_f32(v);
        }
        if c.bf16_w {
            let bytes: Vec<u8> =
                c.w.iter()
                    .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                    .collect();
            self.w.set_bytes(&bytes);
        } else {
            self.w.set_f32(&c.w);
        }
    }
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
    let inputs = Inputs::new(&ctx, c);
    let Inputs {
        key,
        res,
        value,
        gammas,
        w,
    } = inputs;
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
        false,
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
    inputs.fill(c);
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
/// `bf16_w`: the conv weight in BF16 (the ops read it as such).
fn run(backend: &mut Backend, e: i64, groups: &[(i64, i64)], bf16_w: bool, check: Check<'_>) {
    let ple = ple_params();
    let mut hist: Vec<Vec<f32>> = groups
        .iter()
        .enumerate()
        .map(|(i, &(_, ns))| noise((ple.history() * e * 4 * ns) as usize, 50 + i as u64))
        .collect();
    for chunk in 0..2 {
        let c = Case {
            bf16_w,
            ..Case::new(e, 4, groups, 100 * chunk + 7)
        };
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
        run(&mut backend, 64, groups, false, &near);
    }
    // The conv weight in BF16, as the NVFP4 checkpoints ship it.
    for groups in [&[(13, 1)][..], &[(7, 1), (4, 2), (12, 1)]] {
        run(&mut backend, 64, groups, true, &near);
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
        run(&mut gpu, 2560, groups, false, &exact);
    }
    // The conv weight in BF16 (the NVFP4 checkpoints'), at the model's width.
    for groups in [&[(4096, 1)][..], &[(300, 1)], &[(7, 1), (4, 2), (37, 1)]] {
        run(&mut gpu, 2560, groups, true, &exact);
    }
}

/// One chunk of one group through ggml's chain in a graph of its own, allocated with the fold's
/// relayout: as built (`mark` false), or marked, so the PLE ops take the chain's nodes once the
/// graph is allocated. (Streams, history, whether a [`PLE_CONV`] node ran.)
fn chain_step(
    backend: &mut Backend,
    c: &Case,
    hist: &[f32],
    mark: bool,
) -> (Vec<f32>, Vec<f32>, bool) {
    let ple = ple_params();
    let [(_, ns)] = c.groups[..] else {
        panic!("one group")
    };
    let ctx = Context::new(1024, 1024);
    let inputs = Inputs::new(&ctx, c);
    let state = ctx
        .new_tensor(types::F32, &[ple.history(), c.e * c.hc, ns])
        .as_input()
        .as_output();
    let graph = RefCell::new(ctx.new_graph(1024));
    let out = chain(
        &ctx,
        &graph,
        (inputs.key, inputs.res, inputs.value),
        inputs.gammas,
        inputs.w,
        &[state],
        &c.groups,
        &ple,
        mark,
    );
    let out = ctx.cont(out).as_output();
    let mut graph = graph.into_inner();
    graph.expand(out);
    let allocator = ComputeAllocator::new(backend.buffer_type()).with_relayout(RELAYOUT);
    ctx.alloc_graph(&graph, allocator).unwrap();
    inputs.fill(c);
    state.set_f32(hist);
    backend.compute(&graph).unwrap();
    // SAFETY: the graph's node array holds its nodes, valid tensors.
    let ops = unsafe {
        let n = ffi::ggml_graph_n_nodes(graph.raw()) as usize;
        std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph.raw()), n)
            .iter()
            .any(|&t| is_ple_conv(t))
    };
    (out.to_f32(), state.to_f32(), ops)
}

/// Two chunks of `groups` (one group) in turn through the chain marked for the ops and the
/// chain as built: `check(what, marked, as built)` on the streams and the history; whether the
/// ops took the marked chain (checked against `ops`).
fn run_marked(
    backend: &mut Backend,
    e: i64,
    groups: &[(i64, i64)],
    bf16_w: bool,
    ops: bool,
    check: Check<'_>,
) {
    let ple = ple_params();
    let ns = groups[0].1;
    let mut hist = noise((ple.history() * e * 4 * ns) as usize, 60);
    for chunk in 0..2 {
        let c = Case {
            bf16_w,
            ..Case::new(e, 4, groups, 100 * chunk + 9)
        };
        let (oa, ha, ran) = chain_step(backend, &c, &hist, true);
        let (ob, hb, built) = chain_step(backend, &c, &hist, false);
        assert_eq!(ran, ops, "the ops ran in the marked chain, chunk {chunk}");
        assert!(!built, "the ops ran in the chain as built, chunk {chunk}");
        check(&format!("streams, chunk {chunk}"), &oa, &ob);
        check(&format!("history, chunk {chunk}"), &ha, &hb);
        assert!(!same_bits(&ha, &hist), "history moved, chunk {chunk}");
        hist = ha;
    }
}

#[test]
fn marked_chain_runs_as_the_ops_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let near = |what: &str, a: &[f32], b: &[f32]| {
        let r = rel(a, b);
        assert!(r < 1e-5, "{what}: rel {r:e}");
    };
    // The ops take 4 streams of 1,024 or more.
    for bf16_w in [false, true] {
        for groups in [&[(13, 1)][..], &[(5, 1)], &[(6, 2)]] {
            run_marked(&mut backend, 1024, groups, bf16_w, true, &near);
        }
    }
    // Narrower: the marked chain stays ggml's, bit for bit.
    let exact = |what: &str, a: &[f32], b: &[f32]| {
        assert!(same_bits(a, b), "{what}: differs (rel {:e})", rel(a, b));
    };
    run_marked(&mut backend, 64, &[(13, 1)], true, false, &exact);
}

/// On CUDA the marked chain, rewritten into the ops after allocation, gives the chain's bits,
/// the conv weight in F32 and in BF16 (the NVFP4 checkpoints'), at the model's width.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_marked_chain_is_bit_identical_to_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let exact = |what: &str, a: &[f32], b: &[f32]| {
        assert!(same_bits(a, b), "{what}: differs (rel {:e})", rel(a, b));
    };
    for bf16_w in [true, false] {
        for groups in [&[(4096, 1)][..], &[(300, 1)], &[(5, 1)], &[(90, 2)]] {
            run_marked(&mut gpu, 2560, groups, bf16_w, true, &exact);
        }
    }
}
