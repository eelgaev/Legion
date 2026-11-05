//! The fused ops against the ggml node chains they replace, on the CPU; on CUDA (ignored
//! tests) against the CPU implementations.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

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

/// `conv_silu` and ggml's `silu(ssm_conv(concat(state, transpose(x)), w))` with its history
/// view, on `backend`: (op out, op hist, chain out, chain hist). `x` is a column view of a
/// wider tensor, as the graph's are.
fn conv_case(
    backend: &mut Backend,
    c: i64,
    nt: i64,
    ns: i64,
    d: i64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let wide = ctx.new_tensor(types::F32, &[c + 8, nt * ns]).as_input();
    let x = ctx.reshape_3d(
        ctx.cont(ctx.view_2d(wide, [c, nt * ns], wide.nb()[1], 16)),
        c,
        nt,
        ns,
    );
    let state = ctx.new_tensor(types::F32, &[d - 1, c, ns]).as_input();
    let w = ctx.new_tensor(types::F32, &[d, c]).as_input();
    let (out, hist) = conv_silu(&ctx, x, state, w);
    let out = ctx.cont(out).as_output();
    let hist = ctx.cont(hist).as_output();
    let conv_in = ctx.concat(state, ctx.transpose(x), 0);
    let cnb = conv_in.nb();
    let chain_hist = ctx
        .cont(ctx.view_3d(
            conv_in,
            [d - 1, c, ns],
            [cnb[1], cnb[2]],
            nt as usize * cnb[0],
        ))
        .as_output();
    let chain_out = ctx.silu(ctx.ssm_conv(conv_in, w)).as_output();
    let mut graph = ctx.new_graph(32);
    for t in [out, hist, chain_out, chain_hist] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    wide.set_f32(&noise(((c + 8) * nt * ns) as usize, 1));
    state.set_f32(&noise(((d - 1) * c * ns) as usize, 2));
    w.set_f32(&noise((d * c) as usize, 3));
    backend.compute(&graph).unwrap();
    (
        out.to_f32(),
        hist.to_f32(),
        chain_out.to_f32(),
        chain_hist.to_f32(),
    )
}

/// Prompt-sized and decode-sized steps, one and two sequences.
/// (ggml's CUDA `ssm_conv` takes multiples of 128 channels.)
const CONV_CASES: [(i64, i64, i64, i64); 4] = [
    (128, 70, 1, 4),
    (128, 1, 3, 4),
    (256, 2, 2, 4),
    (384, 37, 2, 3),
];

#[test]
fn conv_silu_matches_ggml_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (c, nt, ns, d) in CONV_CASES {
        let (out, hist, chain_out, chain_hist) = conv_case(&mut backend, c, nt, ns, d);
        assert_eq!(hist, chain_hist, "history, {c}x{nt}x{ns}");
        let r = rel(&out, &chain_out);
        assert!(r < 1e-6, "{c}x{nt}x{ns} d {d}: {r:e}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_conv_silu_matches_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    for (c, nt, ns, d) in CONV_CASES.into_iter().chain([(10240, 300, 1, 4)]) {
        let (out, hist, _, _) = conv_case(&mut gpu, c, nt, ns, d);
        let (cpu_out, cpu_hist, _, _) = conv_case(&mut cpu, c, nt, ns, d);
        assert_eq!(hist, cpu_hist, "history, {c}x{nt}x{ns}");
        let r = rel(&out, &cpu_out);
        assert!(r < 1e-6, "{c}x{nt}x{ns} d {d}: {r:e}");
    }
}

/// [`conv_silu_window`] keeping every input against ggml's `concat(state, transpose(x))`, and
/// its conv output against [`conv_silu`]'s, on `backend`: (window, concat, out, conv_silu out).
fn window_case(
    backend: &mut Backend,
    c: i64,
    nt: i64,
    ns: i64,
    d: i64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[c, nt, ns]).as_input();
    let state = ctx.new_tensor(types::F32, &[d - 1, c, ns]).as_input();
    let w = ctx.new_tensor(types::F32, &[d, c]).as_input();
    let (out, window) = conv_silu_window(&ctx, x, state, w, d - 1 + nt);
    let (plain, _) = conv_silu(&ctx, x, state, w);
    let window = ctx.cont(window).as_output();
    let out = ctx.cont(out).as_output();
    let plain = ctx.cont(plain).as_output();
    let concat = ctx.cont(ctx.concat(state, ctx.transpose(x), 0)).as_output();
    let mut graph = ctx.new_graph(32);
    for t in [window, concat, out, plain] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((c * nt * ns) as usize, 1));
    state.set_f32(&noise(((d - 1) * c * ns) as usize, 2));
    w.set_f32(&noise((d * c) as usize, 3));
    backend.compute(&graph).unwrap();
    (
        window.to_f32(),
        concat.to_f32(),
        out.to_f32(),
        plain.to_f32(),
    )
}

#[test]
fn conv_silu_window_is_the_concat() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (c, nt, ns, d) in CONV_CASES {
        let (window, concat, out, plain) = window_case(&mut backend, c, nt, ns, d);
        assert_eq!(window, concat, "window, {c}x{nt}x{ns}");
        assert_eq!(out, plain, "output, {c}x{nt}x{ns}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_conv_silu_window_is_the_concat() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (c, nt, ns, d) in CONV_CASES.into_iter().chain([(10240, 5, 1, 4)]) {
        let (window, concat, out, plain) = window_case(&mut gpu, c, nt, ns, d);
        assert_eq!(window, concat, "window, {c}x{nt}x{ns}");
        assert_eq!(out, plain, "output, {c}x{nt}x{ns}");
    }
}

/// `router` and ggml's chain on `backend` for `n` tokens of `n_expert` logits: (op ids, op
/// weights, chain ids, chain weights).
fn router_case(
    backend: &mut Backend,
    n_expert: i64,
    k: i64,
    n: i64,
) -> (Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let logits = ctx.new_tensor(types::F32, &[n_expert, n]).as_input();
    let (ids, w) = router(&ctx, logits, k);
    let (ids, w) = (ids.as_output(), w.as_output());
    let probs = ctx.soft_max(logits);
    let sel = ctx.argsort_top_k(probs, k as i32);
    let cw = ctx.get_rows(ctx.reshape_3d(probs, 1, n_expert, n), sel);
    let cw = ctx.reshape_2d(cw, k, n);
    let cw = ctx
        .div(
            cw,
            ctx.clamp(ctx.sum_rows(cw), 6.103_515_6e-5, f32::INFINITY),
        )
        .as_output();
    let sel = ctx.cont(sel).as_output();
    let mut graph = ctx.new_graph(32);
    for t in [ids, w, sel, cw] {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Spread logits (no ties) as a router's are.
    let l: Vec<f32> = noise((n_expert * n) as usize, 7)
        .iter()
        .map(|v| v * 6.0)
        .collect();
    logits.set_f32(&l);
    backend.compute(&graph).unwrap();
    let i32s = |t: Tensor<'_>| -> Vec<i32> {
        t.bytes()
            .chunks(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    };
    (i32s(ids), w.to_f32(), i32s(sel), cw.to_f32())
}

#[test]
fn router_matches_ggml_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (n_expert, k, n) in [(512, 10, 37), (64, 8, 5), (33, 1, 3)] {
        let (ids, w, sel, cw) = router_case(&mut backend, n_expert, k, n);
        assert_eq!(ids, sel, "{n_expert} experts, top {k}");
        let r = rel(&w, &cw);
        assert!(r < 1e-6, "{n_expert} experts, top {k}: weights {r:e}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_router_matches_cpu() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let mut cpu = Backend::cpu().unwrap();
    for (n_expert, k, n) in [(512, 10, 4096), (64, 8, 5), (33, 1, 3)] {
        let (ids, w, sel, _) = router_case(&mut gpu, n_expert, k, n);
        let (cpu_ids, cpu_w, _, _) = router_case(&mut cpu, n_expert, k, n);
        assert_eq!(ids, cpu_ids, "{n_expert} experts, top {k}");
        // ggml's CUDA argsort picks the same experts (its order of them may differ)
        let sets = |v: &[i32]| -> Vec<Vec<i32>> {
            v.chunks(k as usize)
                .map(|c| {
                    let mut c = c.to_vec();
                    c.sort();
                    c
                })
                .collect()
        };
        assert_eq!(sets(&ids), sets(&sel), "{n_expert} experts, top {k}: ggml");
        let r = rel(&w, &cpu_w);
        assert!(r < 1e-6, "{n_expert} experts, top {k}: weights {r:e}");
    }
}

/// `delta_prep` and ggml's chain (l2 norms, `sigmoid`, `add`, `softplus`, `mul`) on `backend`,
/// with the replay stores: per output (q, k, g, beta), the op's and the chain's values, then
/// per store (q, k, v, g, beta) what the op wrote and what the chain's value is.
fn prep_case(
    backend: &mut Backend,
    heads: DeltaHeads,
    nt: i64,
    ns: i64,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (head_k, n_k, head_v, n_v) = heads;
    let qkv_dim = 2 * head_k * n_k + head_v * n_v;
    let n = nt * ns;
    let eps = 1e-6f32;
    let ctx = Context::new(96, 96);
    let conv = ctx.new_tensor(types::F32, &[qkv_dim, nt, ns]).as_input();
    // Projections a column view of wider tensors, as the graph's groups are.
    let alpha_all = ctx.new_tensor(types::F32, &[n_v, n + 2]).as_input();
    let beta_all = ctx.new_tensor(types::F32, &[n_v, n + 2]).as_input();
    let dt_bias = ctx.new_tensor(types::F32, &[n_v]).as_input();
    let a = ctx.new_tensor(types::F32, &[n_v]).as_input();
    let alpha = ctx.view_2d(alpha_all, [n_v, n], alpha_all.nb()[1], alpha_all.nb()[1]);
    let beta = ctx.view_2d(beta_all, [n_v, n], beta_all.nb()[1], beta_all.nb()[1]);
    // Replay pools with room for one more token and slot; the group at slot 1.
    let abs = [
        [head_k, n_k],
        [head_k, n_k],
        [head_v, n_v],
        [1, n_v],
        [1, n_v],
    ];
    let pools: Vec<_> = abs
        .iter()
        .map(|ab| {
            ctx.new_tensor(types::F32, &[ab[0] * ab[1], nt + 1, ns + 1])
                .as_input()
        })
        .collect();
    let views: Vec<_> = pools
        .iter()
        .zip(&abs)
        .map(|(p, ab)| {
            ctx.view_4d(
                *p,
                [ab[0], ab[1], nt, ns],
                [ab[0] as usize * 4, p.nb()[1], p.nb()[2]],
                p.nb()[2],
            )
        })
        .collect();
    let st = [views[0], views[1], views[3], views[4]];
    let [q, k, g, b] = delta_prep(
        &ctx,
        conv,
        alpha,
        beta,
        dt_bias,
        a,
        heads,
        eps / head_k as f32,
        Some(st),
    );
    let es = 4usize;
    let row = qkv_dim as usize * es;
    let nb = [head_k as usize * es, row, row * nt as usize];
    let qv = ctx.view_4d(conv, [head_k, n_k, nt, ns], nb, 0);
    let kv = ctx.view_4d(
        conv,
        [head_k, n_k, nt, ns],
        nb,
        (head_k * n_k) as usize * es,
    );
    let vv = ctx.view_4d(
        conv,
        [head_v, n_v, nt, ns],
        [head_v as usize * es, row, row * nt as usize],
        (2 * head_k * n_k) as usize * es,
    );
    let cq = super::super::l2_norm(&ctx, qv, eps);
    let ck = super::super::l2_norm(&ctx, kv, eps);
    let cb = ctx.sigmoid(beta);
    let cg = ctx.mul(ctx.softplus(ctx.add(alpha, dt_bias)), a);
    let outs: Vec<_> = [q, k, g, b, cq, ck, cg, cb, vv]
        .iter()
        .map(|t| ctx.cont(*t).as_output())
        .collect();
    // v is the caller's copy.
    let vcopy = ctx.cpy(vv, views[2]);
    let stored: Vec<_> = [views[0], views[1], vcopy, views[3], views[4]]
        .iter()
        .map(|t| ctx.cont(*t).as_output())
        .collect();
    let mut graph = ctx.new_graph(256);
    for t in outs.iter().chain(&stored) {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    conv.set_f32(&noise((qkv_dim * n) as usize, 11));
    // Wide alpha values, so softplus takes both of its branches.
    let wide: Vec<f32> = noise((n_v * (n + 2)) as usize, 12)
        .iter()
        .map(|v| v * 30.0)
        .collect();
    alpha_all.set_f32(&wide);
    beta_all.set_f32(&noise((n_v * (n + 2)) as usize, 13));
    dt_bias.set_f32(&noise(n_v as usize, 14));
    a.set_f32(&noise(n_v as usize, 15));
    for p in &pools {
        p.set_f32(&vec![0.0; p.elements()]);
    }
    backend.compute(&graph).unwrap();
    let v: Vec<Vec<f32>> = outs.iter().map(|t| t.to_f32()).collect();
    let s: Vec<Vec<f32>> = stored.iter().map(|t| t.to_f32()).collect();
    vec![
        (v[0].clone(), v[4].clone()),
        (v[1].clone(), v[5].clone()),
        (v[2].clone(), v[6].clone()),
        (v[3].clone(), v[7].clone()),
        (s[0].clone(), v[4].clone()),
        (s[1].clone(), v[5].clone()),
        (s[2].clone(), v[8].clone()),
        (s[3].clone(), v[6].clone()),
        (s[4].clone(), v[7].clone()),
    ]
}

const PREP_CASES: [(DeltaHeads, i64, i64); 3] = [
    ((128, 16, 128, 32), 3, 1),
    ((128, 4, 64, 6), 1, 3),
    ((64, 2, 32, 4), 5, 2),
];

#[test]
fn delta_prep_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (heads, nt, ns) in PREP_CASES {
        for (i, (op, chain)) in prep_case(&mut backend, heads, nt, ns).iter().enumerate() {
            let r = rel(op, chain);
            assert!(r < 1e-5, "{heads:?} {nt}x{ns}: output {i}: {r:e}");
        }
    }
}

/// On a GPU the op is ggml-cuda's chain bit for bit (stores included).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_delta_prep_is_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (heads, nt, ns) in PREP_CASES.into_iter().chain([((128, 16, 128, 32), 64, 1)]) {
        for (i, (op, chain)) in prep_case(&mut gpu, heads, nt, ns).iter().enumerate() {
            assert!(
                op.len() == chain.len()
                    && op
                        .iter()
                        .zip(chain)
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{heads:?} {nt}x{ns}: output {i} differs (rel {:e})",
                rel(op, chain)
            );
        }
    }
}

/// `conv_silu_norm` + `delta_gates` and `conv_silu` + `delta_prep` on `backend`: per output
/// (q, k, v, g, beta, history), the fused pair's values and the plain pair's.
fn conv_norm_case(
    backend: &mut Backend,
    heads: DeltaHeads,
    nt: i64,
    ns: i64,
    d: i64,
) -> Vec<(Vec<f32>, Vec<f32>)> {
    let (head_k, n_k, head_v, n_v) = heads;
    let qkv_dim = 2 * head_k * n_k + head_v * n_v;
    let n = nt * ns;
    let eps = 1e-6f32 / head_k as f32;
    let ctx = Context::new(96, 96);
    let x = ctx.new_tensor(types::F32, &[qkv_dim, nt, ns]).as_input();
    let state = ctx.new_tensor(types::F32, &[d - 1, qkv_dim, ns]).as_input();
    let w = ctx.new_tensor(types::F32, &[d, qkv_dim]).as_input();
    let alpha = ctx.new_tensor(types::F32, &[n_v, n]).as_input();
    let beta = ctx.new_tensor(types::F32, &[n_v, n]).as_input();
    let dt_bias = ctx.new_tensor(types::F32, &[n_v]).as_input();
    let a = ctx.new_tensor(types::F32, &[n_v]).as_input();
    let es = 4usize;
    let row = qkv_dim as usize * es;
    let views = |conv| {
        let nb = [head_k as usize * es, row, row * nt as usize];
        let at = (head_k * n_k) as usize * es;
        let nbv = [head_v as usize * es, row, row * nt as usize];
        [
            ctx.view_4d(conv, [head_k, n_k, nt, ns], nb, 0),
            ctx.view_4d(conv, [head_k, n_k, nt, ns], nb, at),
            ctx.view_4d(conv, [head_v, n_v, nt, ns], nbv, 2 * at),
        ]
    };
    let (conv_n, hist_n) = conv_silu_norm(&ctx, x, state, w, heads, eps);
    let [qn, kn, vn] = views(conv_n);
    let [gn, bn] = delta_gates(&ctx, conv_n, alpha, beta, dt_bias, a, heads, eps);
    let (conv_p, hist_p) = conv_silu(&ctx, x, state, w);
    let [_, _, vp] = views(conv_p);
    let [qp, kp, gp, bp] = delta_prep(&ctx, conv_p, alpha, beta, dt_bias, a, heads, eps, None);
    let outs: Vec<_> = [qn, kn, vn, gn, bn, hist_n, qp, kp, vp, gp, bp, hist_p]
        .iter()
        .map(|t| ctx.cont(*t).as_output())
        .collect();
    let mut graph = ctx.new_graph(128);
    for t in &outs {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((qkv_dim * n) as usize, 21));
    state.set_f32(&noise(((d - 1) * qkv_dim * ns) as usize, 22));
    w.set_f32(&noise((d * qkv_dim) as usize, 23));
    alpha.set_f32(&noise((n_v * n) as usize, 24));
    beta.set_f32(&noise((n_v * n) as usize, 25));
    dt_bias.set_f32(&noise(n_v as usize, 26));
    a.set_f32(&noise(n_v as usize, 27));
    backend.compute(&graph).unwrap();
    let v: Vec<Vec<f32>> = outs.iter().map(|t| t.to_f32()).collect();
    (0..6).map(|i| (v[i].clone(), v[i + 6].clone())).collect()
}

const CONV_NORM_CASES: [(DeltaHeads, i64, i64, i64); 3] = [
    ((128, 2, 128, 4), 5, 2, 4),
    ((128, 1, 64, 3), 70, 1, 4),
    ((128, 3, 128, 6), 33, 3, 3),
];

/// Bit for bit: the fused pair's q, k and history are the plain pair's.
fn assert_bits(cases: Vec<(Vec<f32>, Vec<f32>)>, what: &str) {
    for (i, (fused, plain)) in cases.iter().enumerate() {
        assert!(
            fused.len() == plain.len()
                && fused
                    .iter()
                    .zip(plain)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{what}: output {i} differs (rel {:e})",
            rel(fused, plain)
        );
    }
}

#[test]
fn conv_silu_norm_is_conv_silu_then_delta_prep() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (heads, nt, ns, d) in CONV_NORM_CASES {
        let what = format!("{heads:?} {nt}x{ns} d {d}");
        assert_bits(conv_norm_case(&mut backend, heads, nt, ns, d), &what);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_conv_silu_norm_is_conv_silu_then_delta_prep() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let big = ((128, 16, 128, 48), 300, 1, 4);
    for (heads, nt, ns, d) in CONV_NORM_CASES.into_iter().chain([big]) {
        let what = format!("{heads:?} {nt}x{ns} d {d}");
        assert_bits(conv_norm_case(&mut gpu, heads, nt, ns, d), &what);
    }
}

/// `gated_norm` and ggml's chain (`rms_norm`, `mul` by the weight, `mul` by the gate) on
/// `backend`: (op, chain).
fn gated_case(
    backend: &mut Backend,
    head_v: i64,
    n_v: i64,
    n: i64,
    silu: bool,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let o = ctx.new_tensor(types::F32, &[head_v * n_v, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[head_v]).as_input();
    let z = ctx.new_tensor(types::F32, &[head_v * n_v, n]).as_input();
    let op = ctx
        .cont(gated_norm(&ctx, o, w, z, head_v, eps, silu))
        .as_output();
    let z4 = ctx.reshape_4d(z, [head_v, n_v, n, 1]);
    let g = if silu { ctx.silu(z4) } else { ctx.sigmoid(z4) };
    let o4 = ctx.reshape_4d(o, [head_v, n_v, n, 1]);
    let chain = ctx.mul(ctx.mul(ctx.rms_norm(o4, eps), w), g);
    let chain = ctx.cont(ctx.reshape_2d(chain, head_v * n_v, n)).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    o.set_f32(&noise((head_v * n_v * n) as usize, 31));
    w.set_f32(&noise(head_v as usize, 32));
    let wide: Vec<f32> = noise((head_v * n_v * n) as usize, 33)
        .iter()
        .map(|v| v * 8.0)
        .collect();
    z.set_f32(&wide);
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

const GATED_CASES: [(i64, i64, i64); 3] = [(128, 32, 3), (128, 4, 1), (16, 4, 7)];

#[test]
fn gated_norm_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (head_v, n_v, n) in GATED_CASES {
        for silu in [false, true] {
            let (op, chain) = gated_case(&mut backend, head_v, n_v, n, silu);
            let r = rel(&op, &chain);
            assert!(r < 1e-5, "{head_v}x{n_v}x{n} silu {silu}: {r:e}");
        }
    }
}

/// On a GPU the op is ggml-cuda's chain bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gated_norm_is_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (head_v, n_v, n) in GATED_CASES.into_iter().chain([(128, 32, 512)]) {
        for silu in [false, true] {
            let (op, chain) = gated_case(&mut gpu, head_v, n_v, n, silu);
            assert!(
                op.iter()
                    .zip(&chain)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{head_v}x{n_v}x{n} silu {silu}: differs (rel {:e})",
                rel(&op, &chain)
            );
        }
    }
}

/// `gated_norm_bf16` and ggml's chain cast to BF16 on `backend`, both widened back to F32
/// (exactly): (op, chain).
fn gated_bf16_case(
    backend: &mut Backend,
    head_v: i64,
    n_v: i64,
    n: i64,
    silu: bool,
) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let o = ctx.new_tensor(types::F32, &[head_v * n_v, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[head_v]).as_input();
    let z = ctx.new_tensor(types::F32, &[head_v * n_v, n]).as_input();
    let op = gated_norm_bf16(&ctx, o, w, z, head_v, eps, silu);
    let op = ctx.cast(op, types::F32).as_output();
    let z4 = ctx.reshape_4d(z, [head_v, n_v, n, 1]);
    let g = if silu { ctx.silu(z4) } else { ctx.sigmoid(z4) };
    let o4 = ctx.reshape_4d(o, [head_v, n_v, n, 1]);
    let chain = ctx.mul(ctx.mul(ctx.rms_norm(o4, eps), w), g);
    let chain = ctx.reshape_2d(chain, head_v * n_v, n);
    let chain = ctx
        .cast(ctx.cast(chain, types::BF16), types::F32)
        .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    o.set_f32(&noise((head_v * n_v * n) as usize, 41));
    w.set_f32(&noise(head_v as usize, 42));
    let wide: Vec<f32> = noise((head_v * n_v * n) as usize, 43)
        .iter()
        .map(|v| v * 8.0)
        .collect();
    z.set_f32(&wide);
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

#[test]
fn gated_norm_bf16_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (head_v, n_v, n) in GATED_CASES {
        for silu in [false, true] {
            let (op, chain) = gated_bf16_case(&mut backend, head_v, n_v, n, silu);
            // The CPU reference sums in another order: at most an ulp of BF16 apart.
            let r = rel(&op, &chain);
            assert!(r < 1e-2, "{head_v}x{n_v}x{n} silu {silu}: {r:e}");
        }
    }
}

/// On a GPU the BF16 op is ggml-cuda's chain and cast bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gated_norm_bf16_is_the_chain_cast() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (head_v, n_v, n) in GATED_CASES.into_iter().chain([(128, 48, 600)]) {
        for silu in [false, true] {
            let (op, chain) = gated_bf16_case(&mut gpu, head_v, n_v, n, silu);
            assert!(
                op.iter()
                    .zip(&chain)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{head_v}x{n_v}x{n} silu {silu}: differs (rel {:e})",
                rel(&op, &chain)
            );
        }
    }
}

/// `rms_norm_bf16` and ggml's `cast(mul(rms_norm(x), w), BF16)` on `backend`, both widened
/// back to F32 (exactly): (op, chain). Rows vary in scale over six orders of magnitude.
fn norm_bf16_case(backend: &mut Backend, ncols: i64, n: i64) -> (Vec<f32>, Vec<f32>) {
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[ncols, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[ncols]).as_input();
    let op = ctx
        .cast(rms_norm_bf16(&ctx, x, w, eps), types::F32)
        .as_output();
    let chain = ctx.mul(ctx.rms_norm(x, eps), w);
    let chain = ctx
        .cast(ctx.cast(chain, types::BF16), types::F32)
        .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = noise((ncols * n) as usize, 51)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 10f32.powi((i / ncols as usize % 7) as i32 - 3))
        .collect();
    x.set_f32(&xs);
    w.set_f32(&noise(ncols as usize, 52));
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

/// Row widths: the 1024-thread order (5120, 2048, 1024) and the 256-thread one (1000, 128).
const NORM_BF16_CASES: [(i64, i64); 5] =
    [(5120, 40), (2048, 33), (1024, 35), (1000, 34), (128, 36)];

#[test]
fn rms_norm_bf16_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (ncols, n) in NORM_BF16_CASES {
        let (op, chain) = norm_bf16_case(&mut backend, ncols, n);
        let r = rel(&op, &chain);
        assert!(r < 1e-2, "{ncols}x{n}: {r:e}");
    }
}

/// On a GPU the op is ggml-cuda's `rms_norm` + `mul` + cast bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rms_norm_bf16_is_the_chain_cast() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (ncols, n) in NORM_BF16_CASES.into_iter().chain([(5120, 700)]) {
        let (op, chain) = norm_bf16_case(&mut gpu, ncols, n);
        assert!(
            op.iter()
                .zip(&chain)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{ncols}x{n}: differs (rel {:e})",
            rel(&op, &chain)
        );
    }
}

/// `gate_bf16` and ggml's ops cast to BF16 on `backend`, both widened back to F32 (exactly):
/// (op, chain). `silu`: SwiGLU of a plain `[d, n]` gate (the FFN); else the attention output
/// gate, `o * sigmoid(gate)` with the gate read from interleaved `[q | gate]` heads.
fn gate_bf16_case(
    backend: &mut Backend,
    d: i64,
    n_head: i64,
    n: i64,
    silu: bool,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let k = d * n_head;
    let (src, x) = if silu {
        let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
        (x, x)
    } else {
        let qg = ctx.new_tensor(types::F32, &[2 * k, n]).as_input();
        let nb = [(2 * d) as usize * 4, (2 * k) as usize * 4];
        (qg, ctx.view_3d(qg, [d, n_head, n], nb, d as usize * 4))
    };
    let g = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let op = ctx
        .cast(gate_bf16(&ctx, x, g, silu), types::F32)
        .as_output();
    let chain = if silu {
        ctx.swiglu_split(x, g)
    } else {
        ctx.mul(g, ctx.sigmoid(ctx.cont_2d(x, k, n)))
    };
    let chain = ctx
        .cast(ctx.cast(chain, types::BF16), types::F32)
        .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let wide: Vec<f32> = noise(src.elements(), 61).iter().map(|v| v * 12.0).collect();
    src.set_f32(&wide);
    g.set_f32(&noise((k * n) as usize, 62));
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

/// (head width, heads, tokens): the dense 27B's attention gate (256 x 24) and a SwiGLU row
/// (17408 x 1), and odd sizes.
const GATE_BF16_CASES: [(i64, i64, i64); 4] =
    [(256, 24, 40), (17408, 1, 33), (100, 3, 37), (777, 1, 34)];

#[test]
fn gate_bf16_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (d, n_head, n) in GATE_BF16_CASES {
        for silu in [false, true] {
            let (op, chain) = gate_bf16_case(&mut backend, d, n_head, n, silu);
            let r = rel(&op, &chain);
            assert!(r < 1e-2, "{d}x{n_head}x{n} silu {silu}: {r:e}");
        }
    }
}

#[test]
fn gate_bf16_fits_checks_the_layouts() {
    let ctx = Context::new(16, 0);
    let qg = ctx.new_tensor(types::F32, &[2 * 256 * 4, 64]);
    let nb = [2 * 256 * 4, 2 * 256 * 4 * 4];
    let gate = ctx.view_3d(qg, [256, 4, 64], nb, 256 * 4);
    let o = ctx.new_tensor(types::F32, &[256 * 4, 64]);
    let plain = ctx.new_tensor(types::F32, &[256 * 4, 64]);
    let few = ctx.new_tensor(types::F32, &[256 * 4, 32]);
    let half = ctx.new_tensor(types::F16, &[256 * 4, 64]);
    let other = ctx.new_tensor(types::F32, &[256 * 2, 64]);
    let gpu = cfg!(feature = "cuda") && crate::model::fp8::bf16_mma();
    assert_eq!(gate_bf16_fits(gate, o), gpu);
    assert_eq!(gate_bf16_fits(plain, o), gpu);
    assert!(!gate_bf16_fits(few, few), "32 tokens: the decode path");
    assert!(!gate_bf16_fits(half, o));
    assert!(!gate_bf16_fits(other, o));
}

/// On a GPU the op is ggml-cuda's fused unary + mul (or SwiGLU) and cast, bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_gate_bf16_is_the_chain_cast() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (d, n_head, n) in GATE_BF16_CASES.into_iter().chain([(256, 24, 700)]) {
        for silu in [false, true] {
            let (op, chain) = gate_bf16_case(&mut gpu, d, n_head, n, silu);
            assert!(
                op.iter()
                    .zip(&chain)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{d}x{n_head}x{n} silu {silu}: differs (rel {:e})",
                rel(&op, &chain)
            );
        }
    }
}

/// `moe_combine` and ggml's `mul` + adds chain on `backend`: (op, chain).
fn combine_case(backend: &mut Backend, e: i64, used: i64, n: i64) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(64, 64);
    let x = ctx.new_tensor(types::F32, &[e, used, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[1, used, n]).as_input();
    let op = ctx.cont(moe_combine(&ctx, x, w)).as_output();
    let xw = ctx.mul(x, w);
    let nb = xw.nb();
    let mut chain = ctx.view_2d(xw, [e, n], nb[2], 0);
    for k in 1..used as usize {
        chain = ctx.add(chain, ctx.view_2d(xw, [e, n], nb[2], k * nb[1]));
    }
    let chain = ctx.cont(chain).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((e * used * n) as usize, 41));
    w.set_f32(&noise((used * n) as usize, 42));
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

#[test]
fn moe_combine_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, used, n) in [(2560, 10, 3), (64, 4, 5)] {
        let (op, chain) = combine_case(&mut backend, e, used, n);
        let r = rel(&op, &chain);
        assert!(r < 1e-6, "{e}x{used}x{n}: {r:e}");
    }
}

/// On a GPU the op is ggml-cuda's fused `moe_weighted_reduction` bit for bit (the graph here
/// lets ggml fuse: its inputs do not overlap the result).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_moe_combine_is_the_fused_reduction() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, used, n) in [(2560, 10, 3), (2560, 10, 1), (64, 4, 5)] {
        let (op, chain) = combine_case(&mut gpu, e, used, n);
        assert!(
            op.iter()
                .zip(&chain)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{e}x{used}x{n}: differs (rel {:e})",
            rel(&op, &chain)
        );
    }
}

/// `shexp_gate` and ggml's `add(moe, mul(sh, sigmoid(g)))` on `backend`, the gate logits a
/// strided row of a wider `[ld, n]` result (the last, as a joined matmul's): (op, chain).
fn shexp_case(backend: &mut Backend, e: i64, n: i64, ld: i64) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(64, 64);
    let moe = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let sh = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let wide = ctx.new_tensor(types::F32, &[ld, n]).as_input();
    let nb1 = wide.nb()[1];
    let g = ctx.view_2d(wide, [1, n], nb1, (ld - 1) as usize * 4);
    let op = ctx.cont(shexp_gate(&ctx, moe, sh, g)).as_output();
    let chain = ctx.add(moe, ctx.mul(sh, ctx.sigmoid(ctx.cont(g))));
    let chain = ctx.cont(chain).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    moe.set_f32(&noise((e * n) as usize, 51));
    sh.set_f32(&noise((e * n) as usize, 52));
    // Logits over a wider range than noise's, so the sigmoid saturates too.
    let logits: Vec<f32> = noise((ld * n) as usize, 53)
        .iter()
        .map(|v| v * 12.0)
        .collect();
    wide.set_f32(&logits);
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

#[test]
fn shexp_gate_matches_the_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (e, n, ld) in [(2560, 3, 513), (2560, 1, 1), (64, 7, 9)] {
        let (op, chain) = shexp_case(&mut backend, e, n, ld);
        let r = rel(&op, &chain);
        assert!(r < 1e-6, "{e}x{n} (ld {ld}): {r:e}");
    }
}

/// On a GPU the op is ggml-cuda's sigmoid, broadcast mul and add bit for bit.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_shexp_gate_is_the_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for (e, n, ld) in [(2560, 3, 513), (2560, 1, 1), (2560, 64, 1), (64, 7, 9)] {
        let (op, chain) = shexp_case(&mut gpu, e, n, ld);
        assert!(
            op.iter()
                .zip(&chain)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{e}x{n} (ld {ld}): differs (rel {:e})",
            rel(&op, &chain)
        );
    }
}
