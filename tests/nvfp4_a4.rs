//! Legion's W4A4 NVFP4 matmuls (`legion::model::nvfp4_a4`): the CPU references against exact
//! products of the quantized activations, and (ignored, `--features cuda`, a GPU with the FP4
//! kernels: an sm_120a / sm_121a build) the activation quantizer bit for bit and every GEMM
//! variant against the CPU references, plus timings against the W4A16 kernels.
//!
//! `cargo test --release --features cuda --test nvfp4_a4 -- --include-ignored --nocapture --test-threads 1`

use legion::ggml::{Backend, ComputeAllocator, Context, types};
use legion::model::nvfp4_a4;
use legion::model::nvfp4_mm::Act;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Random NVFP4 rows as `tests/nvfp4_mm.rs` makes them.
fn weights(k: usize, m: usize, seed: u64) -> Vec<u8> {
    let mut next = rng(seed);
    let mut out = Vec::with_capacity(m * k / 64 * 36);
    for _ in 0..m * k / 64 {
        for _ in 0..4 {
            let r = next();
            out.push(match r % 64 {
                0 => 0,
                1 => (r >> 8) as u8 & 7,
                _ => ((3 + (r >> 8) % 8) << 3 | (r >> 16) & 7) as u8,
            });
        }
        for _ in 0..4 {
            out.extend(next().to_le_bytes());
        }
    }
    out
}

/// Activations with a spread of magnitudes: sub-blocks scaled by up to 2^±6, a few outliers
/// and a few all-zero sub-blocks (as relu² outputs have).
fn activations(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut next = rng(seed ^ 0x55);
    let mut out = Vec::with_capacity(k * n);
    for _ in 0..k * n / 16 {
        let r = next();
        let scale = match r % 16 {
            0 => 0.0,
            1 => 40.0,
            _ => 2f32.powi((r >> 8) as i32 % 13 - 6),
        };
        for _ in 0..16 {
            out.push((((next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale);
        }
    }
    out
}

fn rel_err(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |s, v| s.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(&x, y)| (x as f64 - y).abs())
        .fold(0f64, f64::max)
        / scale
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
        .collect()
}

/// The graph's output as F32 (BF16 widened).
fn read(out: legion::ggml::Tensor<'_>) -> Vec<f32> {
    if out.ty() == types::BF16 {
        bf16_to_f32(&out.bytes())
    } else {
        out.to_f32()
    }
}

/// `scale · (w · x)` (dense, through the epilogue `act`) on `backend`.
fn run(
    backend: &mut Backend,
    wq: &[u8],
    xs: &[f32],
    scale: Option<f32>,
    act: Act,
    (k, m, n): (usize, usize, usize),
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let w = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let s = weights.new_tensor(types::F32, &[1]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    s.set_f32(&[scale.unwrap_or(1.0)]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let out = nvfp4_a4::mul_mat_act(&ctx, w, x, scale.map(|_| s), act);
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    read(out)
}

/// The grouped matmul on `backend`: experts `[k, m, n_expert]`, x `[k, n_x, n_tok]`.
#[allow(clippy::too_many_arguments)]
fn run_id(
    backend: &mut Backend,
    wq: &[u8],
    xs: &[f32],
    ids: &[i32],
    scale: &[f32],
    act: Act,
    (k, m, n_expert): (usize, usize, usize),
    (n_x, n_used, n_tok): (usize, usize, usize),
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let w = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let s = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    s.set_f32(scale);
    let ctx = Context::new(8, 16);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let out = nvfp4_a4::mul_mat_id_act(&ctx, w, x, id, Some(s), act);
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    read(out)
}

/// The weighted expert sum on `backend`: F32 `[m, n_tok]`.
#[allow(clippy::too_many_arguments)]
fn run_weighted(
    backend: &mut Backend,
    wq: &[u8],
    xs: &[f32],
    ids: &[i32],
    scale: &[f32],
    weights: &[f32],
    (k, m, n_expert): (usize, usize, usize),
    (n_x, n_used, n_tok): (usize, usize, usize),
) -> Vec<f32> {
    let wctx = Context::new(2, 0);
    let w = wctx.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let s = wctx.new_tensor(types::F32, &[n_expert as i64]);
    wctx.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    s.set_f32(scale);
    let ctx = Context::new(8, 16);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let pw = ctx
        .new_tensor(types::F32, &[1, n_used as i64, n_tok as i64])
        .as_input();
    let out = nvfp4_a4::mul_mat_id_weighted(&ctx, w, x, id, Some(s), pw);
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    pw.set_f32(weights);
    backend.compute(&graph).unwrap();
    read(out)
}

/// The fused SwiGLU pair on `backend`.
fn run_swiglu(
    backend: &mut Backend,
    w: (&[u8], &[u8]),
    xs: &[f32],
    scales: (Option<f32>, Option<f32>),
    shape: (usize, usize, usize),
) -> Vec<f32> {
    run_glu(backend, w, xs, scales, shape, false)
}

/// The fused SwiGLU pair, or with `gelu` the GeGLU one (`nvfp4_a4::geglu`), on `backend`.
fn run_glu(
    backend: &mut Backend,
    (wg, wu): (&[u8], &[u8]),
    xs: &[f32],
    (sg, su): (Option<f32>, Option<f32>),
    (k, m, n): (usize, usize, usize),
    gelu: bool,
) -> Vec<f32> {
    let weights = Context::new(4, 0);
    let g = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let u = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let tg = weights.new_tensor(types::F32, &[1]);
    let tu = weights.new_tensor(types::F32, &[1]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    g.set_bytes(wg);
    u.set_bytes(wu);
    tg.set_f32(&[sg.unwrap_or(1.0)]);
    tu.set_f32(&[su.unwrap_or(1.0)]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let (sg, su) = (sg.map(|_| tg), su.map(|_| tu));
    let out = if gelu {
        nvfp4_a4::geglu(&ctx, g, u, x, sg, su)
    } else {
        nvfp4_a4::swiglu(&ctx, g, u, x, sg, su)
    };
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    read(out)
}

/// `scale · (w · x) + res` on `backend`: one `mul_mat_add` node (`fused`), or the matmul and
/// ggml's `add`.
fn run_add(
    backend: &mut Backend,
    wq: &[u8],
    (xs, rs): (&[f32], &[f32]),
    scale: Option<f32>,
    (k, m, n): (usize, usize, usize),
    fused: bool,
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let w = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let s = weights.new_tensor(types::F32, &[1]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    s.set_f32(&[scale.unwrap_or(1.0)]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let res = ctx.new_tensor(types::F32, &[m as i64, n as i64]).as_input();
    let out = if fused {
        nvfp4_a4::mul_mat_add(&ctx, w, x, scale.map(|_| s), res)
    } else {
        ctx.add(nvfp4_a4::mul_mat(&ctx, w, x, scale.map(|_| s)), res)
    };
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    res.set_f32(rs);
    backend.compute(&graph).unwrap();
    read(out)
}

/// The SwiGLU pair of `mul(rms_norm(x, 1e-6), nw)` on `backend`: one `swiglu_norm` node
/// (`fused`), or ggml's norm and `mul`, then `swiglu`.
fn run_swiglu_norm(
    backend: &mut Backend,
    (wg, wu): (&[u8], &[u8]),
    (xs, nws): (&[f32], &[f32]),
    (k, m, n): (usize, usize, usize),
    fused: bool,
) -> Vec<f32> {
    let weights = Context::new(4, 0);
    let g = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let u = weights.new_tensor(types::NVFP4, &[k as i64, m as i64]);
    let nw = weights.new_tensor(types::F32, &[k as i64]);
    let tg = weights.new_tensor(types::F32, &[1]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    g.set_bytes(wg);
    u.set_bytes(wu);
    nw.set_f32(nws);
    tg.set_f32(&[0.5]);
    let ctx = Context::new(8, 16);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let out = if fused {
        nvfp4_a4::swiglu_norm(&ctx, g, u, x, nw, 1e-6, Some(tg), None)
    } else {
        let h = ctx.mul(ctx.rms_norm(x, 1e-6), nw);
        nvfp4_a4::swiglu(&ctx, g, u, h, Some(tg), None)
    };
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    read(out)
}

/// Matmuls of the weights `ws` (rows `ms`, scales 0.5, none, 0.25, ..) over
/// `mul(rms_norm(x, 1e-6), nw)` (or `x` without `nws`) on `backend`, concatenated: one
/// `mul_mat_multi` node (`fused`), or (ggml's norm and `mul`, then) one `mul_mat` each.
fn run_multi(
    backend: &mut Backend,
    ws: &[Vec<u8>],
    ms: &[usize],
    (xs, nws): (&[f32], Option<&[f32]>),
    (k, n): (usize, usize),
    fused: bool,
) -> Vec<f32> {
    let weights = Context::new(2 * ws.len() + 1, 0);
    let wt: Vec<_> = ms
        .iter()
        .map(|&m| weights.new_tensor(types::NVFP4, &[k as i64, m as i64]))
        .collect();
    let st: Vec<_> = ms
        .iter()
        .map(|_| weights.new_tensor(types::F32, &[1]))
        .collect();
    let nw = weights.new_tensor(types::F32, &[k as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    let scale_of = |i: usize| [Some(0.5f32), None, Some(0.25)][i % 3];
    for (i, (w, s)) in wt.iter().zip(&st).enumerate() {
        w.set_bytes(&ws[i]);
        s.set_f32(&[scale_of(i).unwrap_or(1.0)]);
    }
    if let Some(v) = nws {
        nw.set_f32(v);
    }
    let ctx = Context::new(16, 32);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let pairs: Vec<_> = wt
        .iter()
        .zip(&st)
        .enumerate()
        .map(|(i, (&w, &s))| (w, scale_of(i).map(|_| s)))
        .collect();
    let outs = if fused {
        nvfp4_a4::mul_mat_multi(&ctx, &pairs, x, nws.map(|_| (nw, 1e-6)))
    } else {
        let h = match nws {
            Some(_) => ctx.mul(ctx.rms_norm(x, 1e-6), nw),
            None => x,
        };
        pairs
            .iter()
            .map(|&(w, s)| nvfp4_a4::mul_mat(&ctx, w, h, s))
            .collect()
    };
    let outs: Vec<_> = outs.into_iter().map(|o| ctx.cont(o).as_output()).collect();
    let mut graph = ctx.new_graph(32);
    for &o in &outs {
        graph.expand(o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    outs.into_iter().flat_map(read).collect()
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

/// Norm weights around 1 (as checkpoints' are).
fn norm_weights(k: usize, seed: u64) -> Vec<f32> {
    let mut next = rng(seed ^ 0xa4);
    (0..k)
        .map(|_| 0.5 + (next() >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

/// How [`run_swiglu_id`] computes the experts' SwiGLU.
#[derive(Clone, Copy, PartialEq)]
enum GluWay {
    /// One `nvfp4_a4::swiglu_id` node.
    Fused,
    /// The two grouped matmuls, ggml's `silu` and `mul`, as the graph ran them before it (the
    /// CUDA tests' reference).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Chain,
}

/// The routed experts' SwiGLU on `backend`: experts `[k, m, n_expert]`, x `[k, n_x, n_tok]`,
/// F32 `[m, n_used, n_tok]`.
#[allow(clippy::too_many_arguments)]
fn run_swiglu_id(
    backend: &mut Backend,
    (wg, wu): (&[u8], &[u8]),
    xs: &[f32],
    ids: &[i32],
    (sg, su): (&[f32], Option<&[f32]>),
    way: GluWay,
    (k, m, n_expert): (usize, usize, usize),
    (n_x, n_used, n_tok): (usize, usize, usize),
) -> Vec<f32> {
    let weights = Context::new(4, 0);
    let g = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let u = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let tg = weights.new_tensor(types::F32, &[n_expert as i64]);
    let tu = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    g.set_bytes(wg);
    u.set_bytes(wu);
    tg.set_f32(sg);
    tu.set_f32(su.unwrap_or(sg));
    let su = su.map(|_| tu);
    let ctx = Context::new(16, 32);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let out = match way {
        GluWay::Fused => nvfp4_a4::swiglu_id(&ctx, g, u, x, id, Some(tg), su),
        GluWay::Chain => {
            let up = nvfp4_a4::mul_mat_id(&ctx, u, x, id, su);
            let gate = nvfp4_a4::mul_mat_id(&ctx, g, x, id, Some(tg));
            ctx.mul(ctx.silu(gate), up)
        }
    };
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    read(out)
}

fn routing(n_used: usize, n_tok: usize, n_active: usize, seed: u64) -> Vec<i32> {
    let mut next = rng(seed);
    let mut ids = Vec::with_capacity(n_used * n_tok);
    for _ in 0..n_tok {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < n_used {
            let e = (next() % n_active as u64) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        ids.extend(row);
    }
    ids
}

/// Exact `scale · (w · dequantize(quantize(x)))` per token, F64 sums.
fn exact(wf: &[f32], xs: &[f32], k: usize, m: usize, scale: f64) -> Vec<f64> {
    xs.chunks_exact(k)
        .flat_map(|x| {
            let q = nvfp4_a4::dequantize(&nvfp4_a4::quantize(x));
            (0..m)
                .map(|j| {
                    scale
                        * (0..k)
                            .map(|i| wf[j * k + i] as f64 * q[i] as f64)
                            .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The fused residual add and the norm in the SwiGLU's quantization give the unfused graphs'
/// bits on the CPU (#749).
#[test]
fn cpu_fused_residual_and_norm_match_unfused() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (k, m, n, scale) in [
        (64, 3, 1, None),
        (256, 40, 9, Some(0.5)),
        (1024, 17, 70, None),
    ] {
        let wq = weights(k, m, (k + m) as u64);
        let xs = activations(k, n, n as u64);
        let rs = activations(m.div_ceil(16) * 16, n, 7)[..m * n].to_vec();
        let shape = (k, m, n);
        let plain = run_add(&mut cpu, &wq, (&xs, &rs), scale, shape, false);
        let fused = run_add(&mut cpu, &wq, (&xs, &rs), scale, shape, true);
        assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: residual add");
        let (wg, wu) = (weights(k, m, 3 * k as u64), weights(k, m, m as u64));
        let mut xs = xs;
        xs[..k].fill(0.0); // an all-zero row: rms 0
        let nws = norm_weights(k, k as u64);
        let plain = run_swiglu_norm(&mut cpu, (&wg, &wu), (&xs, &nws), shape, false);
        let fused = run_swiglu_norm(&mut cpu, (&wg, &wu), (&xs, &nws), shape, true);
        assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: SwiGLU norm");
        let ms = [m, m + 16, 2 * m];
        let ws: Vec<_> = ms.iter().map(|&mi| weights(k, mi, mi as u64)).collect();
        for norm in [None, Some(&nws[..])] {
            let plain = run_multi(&mut cpu, &ws, &ms, (&xs, norm), (k, n), false);
            let fused = run_multi(&mut cpu, &ws, &ms, (&xs, norm), (k, n), true);
            assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: multi");
        }
    }
}

#[test]
fn cpu_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n) = (256, 40, 3);
    let wq = weights(k, m, 1);
    let xs = activations(k, n, 2);
    let got = run(&mut cpu, &wq, &xs, Some(0.375), Act::None, (k, m, n));
    let wf = legion::ggml::dequantize(types::NVFP4, &wq, k * m);
    let e = rel_err(&got, &exact(&wf, &xs, k, m, 0.375));
    assert!(e < 1e-6, "relative error {e}");
}

#[test]
fn cpu_grouped_and_swiglu_match_exact_products() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n_expert, n_used, n_tok) = (128, 24, 5, 2, 3);
    let wq = weights(k, m * n_expert, 3);
    let wf = legion::ggml::dequantize(types::NVFP4, &wq, k * m * n_expert);
    let scale = [0.5, 0.25, 2.0, 1.0, 0.125];
    let ids = routing(n_used, n_tok, n_expert, 4);
    let xs = activations(k, n_tok, 5);
    let got = run_id(
        &mut cpu,
        &wq,
        &xs,
        &ids,
        &scale,
        Act::None,
        (k, m, n_expert),
        (1, n_used, n_tok),
    );
    let mut want = Vec::new();
    for t in 0..n_tok {
        for s in 0..n_used {
            let e = ids[t * n_used + s] as usize;
            let x = &xs[t * k..(t + 1) * k];
            want.extend(exact(
                &wf[e * m * k..(e + 1) * m * k],
                x,
                k,
                m,
                scale[e] as f64,
            ));
        }
    }
    let e = rel_err(&got, &want);
    assert!(e < 1e-6, "grouped: relative error {e}");
    let pairs = want;

    // Weighted and summed over the slots (BF16 expert outputs: ~2^-9 relative).
    let pw: Vec<f32> = (0..n_used * n_tok)
        .map(|i| 0.25 + i as f32 / 16.0)
        .collect();
    let got = run_weighted(
        &mut cpu,
        &wq,
        &xs,
        &ids,
        &scale,
        &pw,
        (k, m, n_expert),
        (1, n_used, n_tok),
    );
    let mut want = vec![0f64; m * n_tok];
    for t in 0..n_tok {
        for s in 0..n_used {
            for j in 0..m {
                want[t * m + j] += pairs[(t * n_used + s) * m + j] * pw[t * n_used + s] as f64;
            }
        }
    }
    let e = rel_err(&got, &want);
    assert!(e < 1e-2, "weighted: relative error {e}");

    let (wg, wu) = (weights(k, m, 7), weights(k, m, 8));
    let got = run_swiglu(&mut cpu, (&wg, &wu), &xs, (Some(0.5), None), (k, m, n_tok));
    let fg = legion::ggml::dequantize(types::NVFP4, &wg, k * m);
    let fu = legion::ggml::dequantize(types::NVFP4, &wu, k * m);
    let (g, u) = (exact(&fg, &xs, k, m, 0.5), exact(&fu, &xs, k, m, 1.0));
    let want: Vec<f64> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let e = rel_err(&got, &want);
    assert!(e < 1e-2, "swiglu (BF16 out): relative error {e}");

    let got = run_glu(
        &mut cpu,
        (&wg, &wu),
        &xs,
        (Some(0.5), None),
        (k, m, n_tok),
        true,
    );
    let want: Vec<f64> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| {
            0.5 * g * (1.0 + (0.797_884_560_802_865_4 * g * (1.0 + 0.044715 * g * g)).tanh()) * u
        })
        .collect();
    let e = rel_err(&got, &want);
    assert!(e < 1e-2, "geglu (BF16 out): relative error {e}");
}

#[test]
fn cpu_grouped_swiglu_matches_exact_products() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n_expert, n_used, n_tok) = (128, 24, 5, 2, 3);
    let (wg, wu) = (weights(k, m * n_expert, 21), weights(k, m * n_expert, 22));
    let fg = legion::ggml::dequantize(types::NVFP4, &wg, k * m * n_expert);
    let fu = legion::ggml::dequantize(types::NVFP4, &wu, k * m * n_expert);
    let sg = [0.5, 0.25, 2.0, 1.0, 0.125];
    let ids = routing(n_used, n_tok, n_expert, 23);
    for n_x in [1, n_used] {
        let xs = activations(k, n_x * n_tok, 24);
        let got = run_swiglu_id(
            &mut cpu,
            (&wg, &wu),
            &xs,
            &ids,
            (&sg, None),
            GluWay::Fused,
            (k, m, n_expert),
            (n_x, n_used, n_tok),
        );
        let mut want = Vec::new();
        for t in 0..n_tok {
            for s in 0..n_used {
                let e = ids[t * n_used + s] as usize;
                let r = t * n_x + if n_x == 1 { 0 } else { s };
                let x = &xs[r * k..(r + 1) * k];
                let ex = |f: &[f32], sc: f64| exact(&f[e * m * k..(e + 1) * m * k], x, k, m, sc);
                let (g, u) = (ex(&fg, sg[e] as f64), ex(&fu, 1.0));
                want.extend(g.iter().zip(&u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u));
            }
        }
        let e = rel_err(&got, &want);
        assert!(e < 1e-5, "n_x {n_x}: relative error {e}");
    }
}

/// The experts' SwiGLU and down matmul on the W4A4 grouped ops on `backend` (gate, up and down
/// experts `w`, `x` `[k, 1, n_tok]`), the ids marked by `nvfp4_mm::skips` when `skip`: the
/// SwiGLU's rows `[m, n_used, n_tok]` and the down rows `[k, n_used, n_tok]`.
fn run_ffn_id(
    backend: &mut Backend,
    w: (&[u8], &[u8], &[u8]),
    xs: &[f32],
    ids: &[i32],
    skip: bool,
    (k, m, n_expert): (usize, usize, usize),
    (n_used, n_tok): (usize, usize),
) -> (Vec<f32>, Vec<f32>) {
    use legion::model::nvfp4_mm::skips;
    let weights = Context::new(4, 0);
    let ne = |k: usize, m: usize| [k as i64, m as i64, n_expert as i64];
    let g = weights.new_tensor(types::NVFP4, &ne(k, m));
    let u = weights.new_tensor(types::NVFP4, &ne(k, m));
    let d = weights.new_tensor(types::NVFP4, &ne(m, k));
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    g.set_bytes(w.0);
    u.set_bytes(w.1);
    d.set_bytes(w.2);
    let ctx = Context::new(16, 32);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let mark = |t| if skip { skips(t) } else { t };
    let act = mark(nvfp4_a4::swiglu_id(&ctx, g, u, x, id, None, None));
    let down = mark(nvfp4_a4::mul_mat_id(&ctx, d, act, id, None));
    let (act, down) = (ctx.cont(act).as_output(), ctx.cont(down).as_output());
    let mut graph = ctx.new_graph(32);
    graph.expand(act);
    graph.expand(down);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    (read(act), read(down))
}

/// `--tp-moe-ep`'s skipped pairs on the W4A4 grouped ops (#1176) on `backend`: a rank holding
/// experts `0..n_expert / 2` gets ids of -1 for the others; those pairs' SwiGLU and down rows
/// are zeros, and every other pair's are the bits of the full routing.
fn check_skips(backend: &mut Backend, (k, m, n_expert): (usize, usize, usize), n_tok: usize) {
    let n_used = 4;
    let w = (
        weights(k, m * n_expert, 31),
        weights(k, m * n_expert, 32),
        weights(m, k * n_expert, 33),
    );
    let w = (&w.0[..], &w.1[..], &w.2[..]);
    let xs = activations(k, n_tok, 34);
    let ids = routing(n_used, n_tok, n_expert, n_tok as u64);
    let held = (n_expert / 2) as i32;
    let slots: Vec<i32> = ids.iter().map(|&e| if e < held { e } else { -1 }).collect();
    let shape = (k, m, n_expert);
    let (a, d) = run_ffn_id(backend, w, &xs, &ids, false, shape, (n_used, n_tok));
    let (a_s, d_s) = run_ffn_id(backend, w, &xs, &slots, true, shape, (n_used, n_tok));
    for (q, &e) in slots.iter().enumerate() {
        let (ra, rd) = (q * m..(q + 1) * m, q * k..(q + 1) * k);
        if e < 0 {
            assert!(a_s[ra].iter().all(|&v| v == 0.0), "pair {q}: SwiGLU row");
            assert!(d_s[rd].iter().all(|&v| v == 0.0), "pair {q}: down row");
        } else {
            assert!(same_bits(&a_s[ra.clone()], &a[ra]), "pair {q}: SwiGLU bits");
            assert!(same_bits(&d_s[rd.clone()], &d[rd]), "pair {q}: down bits");
        }
    }
    assert!(slots.contains(&-1) && slots.iter().any(|&e| e >= 0));
}

#[test]
fn cpu_skipped_ids_give_zero_rows_and_keep_the_others() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check_skips(&mut cpu, (128, 64, 8), 6);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use legion::model::nvfp4_a4::{Grouped, Quantized};
    use legion::model::nvfp4_mm;

    /// The decode kernel's shapes: `rt` row tiles (2 or 4 for the SwiGLU, 1 or 2 otherwise),
    /// `kw` warps over k.
    fn mv_shapes(glu: bool) -> Vec<Grouped> {
        let rts: &[i32] = if glu { &[2, 4] } else { &[1, 2] };
        rts.iter()
            .flat_map(|&rt| [2, 4, 8].map(|kw| Grouped::Mv { rt, kw }))
            .collect()
    }

    fn gpu() -> Option<Backend> {
        use legion::ggml::{DeviceKind, device_by_kind};
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU")
            .init()
            .unwrap();
        if !nvfp4_a4::device_supported() {
            println!(
                "skipped: this build has no FP4 kernels for the GPU (needs sm_120a / sm_121a)"
            );
            return None;
        }
        Some(gpu)
    }

    /// The quantized rows of `xs` (`[k, rows]`, given as F32 or rounded to BF16) on `backend`.
    fn quantize(backend: &mut Backend, xs: &[f32], k: usize, bf16: bool) -> Vec<Quantized> {
        let rows = xs.len() / k;
        let ctx = Context::new(8, 16);
        let x = ctx
            .new_tensor(types::F32, &[k as i64, rows as i64])
            .as_input();
        let xin = if bf16 { ctx.cast(x, types::BF16) } else { x };
        let out = nvfp4_a4::quantize_op(&ctx, xin).as_output();
        let mut graph = ctx.new_graph(16);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        x.set_f32(xs);
        backend.compute(&graph).unwrap();
        let bytes = out.bytes();
        let (q_ld, s_ld, [codes, scales, dec]) = nvfp4_a4::quant_layout(k as i64, rows as i64);
        (0..rows)
            .map(|r| {
                let c = codes + r * q_ld as usize;
                let s = scales + r * s_ld as usize;
                let d = dec + 4 * r;
                // Padding past k is zero.
                assert!(bytes[c + k / 2..c + q_ld as usize].iter().all(|&b| b == 0));
                assert!(bytes[s + k / 16..s + s_ld as usize].iter().all(|&b| b == 0));
                Quantized {
                    codes: bytes[c..c + k / 2].to_vec(),
                    scales: bytes[s..s + k / 16].to_vec(),
                    dec: f32::from_ne_bytes(bytes[d..d + 4].try_into().unwrap()),
                }
            })
            .collect()
    }

    /// The quantizer on the GPU gives the CPU reference's scales, decode scales and code values
    /// (a zero may differ in sign), from F32 and from BF16 activations.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_quantizer_matches_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Backend::cpu().unwrap();
        for (k, rows) in [
            (64, 1),
            (256, 7),
            (640, 41),
            (1024, 6),
            (1088, 5),
            (2688, 33),
            (5120, 4),
            (17408, 3),
        ] {
            let mut xs = activations(k, rows, (k + rows) as u64);
            // An all-zero row, and one with a single huge value.
            xs[..k].fill(0.0);
            if rows > 2 {
                xs[2 * k + 5] = 3.0e4;
            }
            for bf16 in [false, true] {
                let want = quantize(&mut cpu, &xs, k, bf16);
                let got = quantize(&mut gpu, &xs, k, bf16);
                for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g.scales, w.scales, "k {k} row {r} bf16 {bf16}: scales");
                    assert_eq!(
                        g.dec.to_bits(),
                        w.dec.to_bits(),
                        "k {k} row {r}: decode scale"
                    );
                    assert_eq!(
                        nvfp4_a4::dequantize(g),
                        nvfp4_a4::dequantize(w),
                        "k {k} row {r} bf16 {bf16}: codes"
                    );
                }
            }
            println!("quantizer k {k} rows {rows}: bit-exact");
        }
    }

    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_matches_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        for (k, m, n, scale) in [
            (64, 1, 1, None),
            (256, 130, 33, Some(0.5)),
            (1856, 70, 5, Some(0.25)),
            (2688, 300, 129, Some(3.0e-3)),
            (5120, 257, 200, None),
            (192, 48, 64, Some(1.0)),
            (320, 129, 65, Some(2.0)),
            // 16-byte aligned rows, more than 64 tokens: the 4-block, 16-byte-copy tile.
            (1280, 70, 160, Some(0.5)),
        ] {
            let wq = weights(k, m, (k + m) as u64);
            let xs = activations(k, n, n as u64);
            for act in [Act::None, Act::ReluSqr] {
                let want: Vec<f64> = run(&mut cpu, &wq, &xs, scale, act, (k, m, n))
                    .iter()
                    .map(|&v| v as f64)
                    .collect();
                let got = run(&mut gpu, &wq, &xs, scale, act, (k, m, n));
                let e = rel_err(&got, &want);
                // BF16 outputs (relu²): one rounding apart where the F32 sums differ.
                let tol = if act == Act::None { 1e-5 } else { 1e-2 };
                assert!(e < tol, "k {k} m {m} n {n} {act:?}: relative error {e}");
            }
            println!("dense k {k} m {m} n {n}: matches");
        }
    }

    /// [`check_skips`] on the GEMM (batches of more than `nvfp4_a4::MV_MAX_TOKENS` tokens, as
    /// `--tp-moe-ep` gives the W4A4 ops skips only there): the zeros stored by the CTAs past
    /// the last tile.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_skipped_ids_give_zero_rows_and_keep_the_others() {
        let Some(mut gpu) = gpu() else { return };
        for (shape, n_tok) in [
            ((256, 128, 16), 100),
            ((640, 192, 32), 300),
            ((128, 64, 8), 70),
        ] {
            check_skips(&mut gpu, shape, n_tok);
            println!("skips {shape:?} tokens {n_tok}: zeros and bit-exact");
        }
    }

    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_mul_mat_id_matches_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (128, 40, 4, 2, 1, 4),
            (256, 100, 16, 6, 16, 16),
            (1856, 48, 8, 3, 37, 5),
            (192, 130, 32, 4, 70, 32),
            (2688, 140, 8, 6, 300, 8),
            // Rows 8 bytes past a 16-byte word (the two-CTA tile's 16-byte copies) and 16-byte
            // aligned rows with more than 64 tokens per expert (nt 8's 16-byte copies).
            (640, 160, 16, 4, 200, 16),
            (1280, 96, 4, 2, 300, 4),
        ] {
            let wq = weights(k, m * n_expert, k as u64);
            let scale: Vec<f32> = (0..n_expert).map(|e| 0.5 + e as f32 / 8.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            for n_x in [1, n_used] {
                let xs = activations(k, n_x * n_tok, m as u64);
                let shape = ((k, m, n_expert), (n_x, n_used, n_tok));
                for act in [Act::None, Act::ReluSqr] {
                    let want: Vec<f64> =
                        run_id(&mut cpu, &wq, &xs, &ids, &scale, act, shape.0, shape.1)
                            .iter()
                            .map(|&v| v as f64)
                            .collect();
                    let tol = if act == Act::None { 1e-5 } else { 1e-2 };
                    // The decode kernel (the default up to `MV_MAX_TOKENS` tokens): every
                    // shape close to the CPU, the bits depending on `kw` alone.
                    let mut by_kw: Vec<(i32, Vec<f32>)> = Vec::new();
                    for g in mv_shapes(false) {
                        nvfp4_a4::force_grouped(Some(g));
                        let got = run_id(&mut gpu, &wq, &xs, &ids, &scale, act, shape.0, shape.1);
                        nvfp4_a4::force_grouped(None);
                        let e = rel_err(&got, &want);
                        assert!(e < tol, "{shape:?} {act:?} {g:?}: relative error {e}");
                        let Grouped::Mv { kw, .. } = g else {
                            unreachable!()
                        };
                        match by_kw.iter().find(|(w, _)| *w == kw) {
                            Some((_, first)) => assert!(
                                first
                                    .iter()
                                    .zip(&got)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{shape:?} {act:?} {g:?}: other bits than rt 1"
                            ),
                            None => by_kw.push((kw, got)),
                        }
                    }
                    nvfp4_a4::force_grouped(Some(Grouped::Gemm));
                    let got = run_id(&mut gpu, &wq, &xs, &ids, &scale, act, shape.0, shape.1);
                    nvfp4_a4::force_grouped(None);
                    let e = rel_err(&got, &want);
                    assert!(e < tol, "{shape:?} {act:?}: relative error {e}");
                    // Every token tile of the GEMM gives the same bits (k summed in the same
                    // order).
                    for nt in [2, 4, 8, nvfp4_a4::TWO_CTAS_4] {
                        nvfp4_a4::force_tokens_tile(Some(nt));
                        let tiled = run_id(&mut gpu, &wq, &xs, &ids, &scale, act, shape.0, shape.1);
                        nvfp4_a4::force_tokens_tile(None);
                        let same = tiled
                            .iter()
                            .zip(&got)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        assert!(same, "{shape:?} {act:?} nt {nt:#x}: other bits");
                    }
                }
                let pw: Vec<f32> = (0..n_used * n_tok)
                    .map(|i| 0.1 + (i % 7) as f32 / 9.0)
                    .collect();
                let want: Vec<f64> =
                    run_weighted(&mut cpu, &wq, &xs, &ids, &scale, &pw, shape.0, shape.1)
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                for g in [
                    None,
                    Some(Grouped::Gemm),
                    Some(Grouped::Mv { rt: 2, kw: 4 }),
                ] {
                    nvfp4_a4::force_grouped(g);
                    let got = run_weighted(&mut gpu, &wq, &xs, &ids, &scale, &pw, shape.0, shape.1);
                    nvfp4_a4::force_grouped(None);
                    let e = rel_err(&got, &want);
                    assert!(e < 1e-2, "{shape:?} weighted {g:?}: relative error {e}");
                }
            }
            println!("grouped k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok}: matches");
        }
    }

    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_swiglu_matches_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        for (k, m, n, scales) in [
            (64, 1, 1, (None, None)),
            (512, 130, 9, (Some(0.05), Some(0.5))),
            (2688, 300, 16, (Some(0.01), None)),
            (256, 200, 33, (Some(0.3), Some(0.7))),
            (1024, 130, 300, (None, None)),
        ] {
            let (wg, wu) = (weights(k, m, 3 * k as u64), weights(k, m, m as u64));
            let xs = activations(k, n, n as u64);
            let want: Vec<f64> = run_swiglu(&mut cpu, (&wg, &wu), &xs, scales, (k, m, n))
                .iter()
                .map(|&v| v as f64)
                .collect();
            let got = run_swiglu(&mut gpu, (&wg, &wu), &xs, scales, (k, m, n));
            let e = rel_err(&got, &want);
            assert!(e < 1e-2, "k {k} m {m} n {n}: relative error {e}");
            println!("swiglu k {k} m {m} n {n}: relative error {e:.2e}");
            let want: Vec<f64> = run_glu(&mut cpu, (&wg, &wu), &xs, scales, (k, m, n), true)
                .iter()
                .map(|&v| v as f64)
                .collect();
            let got = run_glu(&mut gpu, (&wg, &wu), &xs, scales, (k, m, n), true);
            let e = rel_err(&got, &want);
            assert!(e < 1e-2, "geglu k {k} m {m} n {n}: relative error {e}");
        }
    }

    /// The GEMM's fused residual add and the norm in the SwiGLU's quantization (one kernel
    /// with ggml-cuda's rms_norm bits, 256- and 1024-thread variants) give the bits of the
    /// unfused ggml-cuda graphs (#749).
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_fused_residual_and_norm_match_unfused() {
        let Some(mut gpu) = gpu() else { return };
        for (k, m, n) in [
            (64, 3, 65),
            (512, 130, 70),
            (1024, 129, 130),
            (5120, 300, 200),
            (8192, 64, 66),
        ] {
            let wq = weights(k, m, (k + m) as u64);
            let mut xs = activations(k, n, n as u64);
            xs[..k].fill(0.0);
            let rs = activations(m.div_ceil(16) * 16, n, 7)[..m * n].to_vec();
            let shape = (k, m, n);
            for scale in [None, Some(0.25)] {
                let plain = run_add(&mut gpu, &wq, (&xs, &rs), scale, shape, false);
                let fused = run_add(&mut gpu, &wq, (&xs, &rs), scale, shape, true);
                assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: residual add");
            }
            let (wg, wu) = (weights(k, m, 3 * k as u64), weights(k, m, m as u64));
            let nws = norm_weights(k, k as u64);
            let plain = run_swiglu_norm(&mut gpu, (&wg, &wu), (&xs, &nws), shape, false);
            let fused = run_swiglu_norm(&mut gpu, (&wg, &wu), (&xs, &nws), shape, true);
            assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: SwiGLU norm");
            let ms = [m, m + 16, 2 * m];
            let ws: Vec<_> = ms.iter().map(|&mi| weights(k, mi, mi as u64)).collect();
            for norm in [None, Some(&nws[..])] {
                let plain = run_multi(&mut gpu, &ws, &ms, (&xs, norm), (k, n), false);
                let fused = run_multi(&mut gpu, &ws, &ms, (&xs, norm), (k, n), true);
                assert!(same_bits(&plain, &fused), "k {k} m {m} n {n}: multi");
            }
            println!("fusions k {k} m {m} n {n}: bit-exact");
        }
    }

    /// The grouped SwiGLU: the same bits as the chain it replaces (two grouped matmuls,
    /// ggml-cuda's `silu` and `mul`) at every token tile, and the CPU reference's values.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_swiglu_id_matches_chain_and_cpu() {
        let Some(mut gpu) = gpu() else { return };
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (128, 40, 4, 2, 1, 4),
            (256, 100, 16, 6, 16, 16),
            (192, 130, 32, 4, 70, 32),
            (2560, 64, 64, 10, 200, 64),
            (2560, 640, 32, 10, 300, 32),
        ] {
            let (wg, wu) = (
                weights(k, m * n_expert, k as u64),
                weights(k, m * n_expert, 7),
            );
            let sg: Vec<f32> = (0..n_expert).map(|e| 0.5 + e as f32 / 8.0).collect();
            let su: Vec<f32> = (0..n_expert).map(|e| 0.25 + e as f32 / 16.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            for n_x in [1, n_used] {
                let xs = activations(k, n_x * n_tok, m as u64);
                let shape = ((k, m, n_expert), (n_x, n_used, n_tok));
                for su in [Some(&su[..]), None] {
                    let run = |b: &mut Backend, way| {
                        let s = (&sg[..], su);
                        run_swiglu_id(b, (&wg, &wu), &xs, &ids, s, way, shape.0, shape.1)
                    };
                    let differ = |a: &[f32], b: &[f32]| {
                        a.iter()
                            .zip(b)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count()
                    };
                    let want: Vec<f64> = run(&mut cpu, GluWay::Fused)
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                    // The default kernels: the fused SwiGLU on the chain's bits.
                    let chain = run(&mut gpu, GluWay::Chain);
                    let got = run(&mut gpu, GluWay::Fused);
                    let diff = differ(&got, &chain);
                    assert_eq!(diff, 0, "{shape:?} default: {diff} values differ");
                    // The GEMM, at every token tile.
                    nvfp4_a4::force_grouped(Some(Grouped::Gemm));
                    let chain = run(&mut gpu, GluWay::Chain);
                    nvfp4_a4::force_grouped(None);
                    for nt in [None, Some(2), Some(4), Some(8), Some(nvfp4_a4::TWO_CTAS_4)] {
                        nvfp4_a4::force_grouped(Some(Grouped::Gemm));
                        nvfp4_a4::force_tokens_tile(nt);
                        let got = run(&mut gpu, GluWay::Fused);
                        nvfp4_a4::force_tokens_tile(None);
                        nvfp4_a4::force_grouped(None);
                        let diff = differ(&got, &chain);
                        assert_eq!(diff, 0, "{shape:?} nt {nt:?}: {diff} values differ");
                    }
                    let e = rel_err(&chain, &want);
                    assert!(e < 1e-4, "{shape:?}: relative error {e} to the CPU");
                    // The decode kernel, each shape (the chain's matmuls on two row tiles and
                    // the same `kw`).
                    for g in mv_shapes(true) {
                        let Grouped::Mv { kw, .. } = g else {
                            unreachable!()
                        };
                        nvfp4_a4::force_grouped(Some(Grouped::Mv { rt: 2, kw }));
                        let chain = run(&mut gpu, GluWay::Chain);
                        nvfp4_a4::force_grouped(Some(g));
                        let got = run(&mut gpu, GluWay::Fused);
                        nvfp4_a4::force_grouped(None);
                        let diff = differ(&got, &chain);
                        assert_eq!(diff, 0, "{shape:?} {g:?}: {diff} values differ");
                        let e = rel_err(&got, &want);
                        assert!(e < 1e-4, "{shape:?} {g:?}: relative error {e} to the CPU");
                    }
                }
            }
            println!(
                "swiglu_id k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok}: bit-exact"
            );
        }
    }

    fn time(gpu: &mut Backend, graph: &legion::ggml::Graph<'_>, reps: usize) -> f64 {
        gpu.compute(graph).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            gpu.compute(graph).unwrap();
        }
        t.elapsed().as_secs_f64() / reps as f64
    }

    /// W4A4 against W4A16 (`nvfp4_mm`) on Qwen3.6-27B's MLP and Nemotron 3.5's shapes, dense
    /// and fused SwiGLU. `LEGION_NVFP4_TIMING_N` (comma list) picks the batch sizes.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_timings() {
        const L: usize = 8;
        let Some(mut gpu) = gpu() else { return };
        let ns: Vec<usize> = std::env::var("LEGION_NVFP4_TIMING_N")
            .unwrap_or_else(|_| "16,32,64,128,512,2048,4096".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        for (k, m, glu) in [
            (5120, 17408, true),
            (17408, 5120, false),
            (2688, 3712, false),
            (3712, 2688, false),
        ] {
            let wq = weights(k, m, 5);
            let weights = Context::new(2 * L + 1, 0);
            let ws: Vec<_> = (0..2 * L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64]))
                .collect();
            let st = weights.new_tensor(types::F32, &[1]);
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            st.set_f32(&[0.01]);
            let flops_per = if glu { 2.0 } else { 1.0 };
            for &n in &ns {
                let mut line = format!("k {k:5} m {m:5}{} n {n:4}:", if glu { " glu" } else { "" });
                for a4 in [false, true] {
                    let ctx = Context::new(8 * L + 4, 8 * L + 4);
                    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                    let mut graph = ctx.new_graph(8 * L + 4);
                    for l in 0..L {
                        let (w, w2) = (ws[2 * l], ws[2 * l + 1]);
                        let out = match (glu, a4) {
                            (true, true) => nvfp4_a4::swiglu(&ctx, w, w2, x, Some(st), Some(st)),
                            (true, false) => nvfp4_mm::swiglu(&ctx, w, w2, x, Some(st), Some(st)),
                            (false, true) => nvfp4_a4::mul_mat(&ctx, w, x, Some(st)),
                            (false, false) => nvfp4_mm::mul_mat(&ctx, w, x, Some(st)),
                        };
                        graph.expand(out.as_output());
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    x.set_f32(&activations(k, n, 9));
                    let s = time(&mut gpu, &graph, 5) / L as f64;
                    line += &format!(
                        "  {} {:8.1} us {:6.1} TFLOPS",
                        if a4 { "w4a4" } else { "w4a16" },
                        s * 1e6,
                        flops_per * 2.0 * (k * m * n) as f64 / s / 1e12
                    );
                }
                println!("{line}");
            }
        }
    }

    /// Grouped expert matmuls, W4A4 against W4A16, on Nemotron 3.5's expert shapes: 128
    /// experts, 6 used per token.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_mul_mat_id_timings() {
        const L: usize = 4;
        let Some(mut gpu) = gpu() else { return };
        let (n_expert, n_used) = (128usize, 6usize);
        for (k, m, n_x) in [(2688, 1856, 1), (1856, 2688, 6)] {
            let wq = weights(k, m * n_expert, 5);
            let weights = Context::new(L + 1, 0);
            let ws: Vec<_> = (0..L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]))
                .collect();
            let st = weights.new_tensor(types::F32, &[n_expert as i64]);
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            st.set_f32(&vec![0.01; n_expert]);
            let tok_list: Vec<usize> = std::env::var("LEGION_NVFP4_TIMING_TOKENS")
                .unwrap_or_else(|_| "16,64,256,512,1024,4096".into())
                .split(',')
                .map(|s| s.parse().unwrap())
                .collect();
            for n_tok in tok_list {
                let ids = routing(n_used, n_tok, n_expert, 11);
                let mut line = format!("k {k:5} m {m:5} tokens {n_tok:4}:");
                for (a4, nt) in [(false, 0), (true, 0), (true, 2), (true, 4), (true, 8)] {
                    nvfp4_a4::force_tokens_tile((nt > 0).then_some(nt));
                    let ctx = Context::new(8 * L + 8, 8 * L + 8);
                    let x = ctx
                        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
                        .as_input();
                    let id = ctx
                        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input();
                    let mut graph = ctx.new_graph(8 * L + 8);
                    for w in &ws {
                        let out = if a4 {
                            nvfp4_a4::mul_mat_id(&ctx, *w, x, id, Some(st))
                        } else {
                            nvfp4_mm::mul_mat_id(&ctx, *w, x, id, Some(st))
                        };
                        graph.expand(out.as_output());
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    x.set_f32(&activations(k, n_x * n_tok, 9));
                    id.set_i32(&ids);
                    let s = time(&mut gpu, &graph, 5) / L as f64;
                    line += &format!(
                        "  {}{} {:7.1} us",
                        if a4 { "w4a4" } else { "w4a16" },
                        if nt > 0 {
                            format!(" nt{nt}")
                        } else {
                            String::new()
                        },
                        s * 1e6,
                    );
                }
                nvfp4_a4::force_tokens_tile(None);
                println!("{line}");
            }
        }
    }

    /// Routed experts at Qwen3.8-Flash-Next's shapes (k 2560, m 640, 512 experts, top-10): the
    /// SwiGLU half (W4A16 fused GEMM, the W4A4 chain of two grouped GEMMs + silu + mul, the
    /// W4A4 fused GEMM) and the down matmul (W4A16 on BF16, W4A4 on F32), µs per layer.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_swiglu_id_timings() {
        const L: usize = 4;
        let Some(mut gpu) = gpu() else { return };
        let (k, m, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
        let wq = weights(k, m * n_expert, 5);
        let wd = weights(m, k * n_expert, 6);
        let weights = Context::new(3 * L + 1, 0);
        let new = |a: usize, b: usize| {
            weights.new_tensor(types::NVFP4, &[a as i64, b as i64, n_expert as i64])
        };
        let gs: Vec<_> = (0..L).map(|_| new(k, m)).collect();
        let us: Vec<_> = (0..L).map(|_| new(k, m)).collect();
        let ds: Vec<_> = (0..L).map(|_| new(m, k)).collect();
        let st = weights.new_tensor(types::F32, &[n_expert as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for w in gs.iter().chain(&us) {
            w.set_bytes(&wq);
        }
        for w in &ds {
            w.set_bytes(&wd);
        }
        st.set_f32(&vec![0.01; n_expert]);
        for n_tok in [512usize, 1024, 2048, 4096] {
            let ids = routing(n_used, n_tok, n_expert, 11);
            let mut line = format!("tokens {n_tok:4}:");
            // The A4 kernels also with a forced token tile of 64 and 128 (one CTA per SM).
            let mut runs = vec![("w4a16-glu", 0), ("a4-chain", 0), ("w4a16-down", 0)];
            for nt in [0, 4, 8] {
                runs.push(("a4-glu", nt));
                runs.push(("a4-down", nt));
            }
            for (what, nt) in runs {
                nvfp4_a4::force_tokens_tile((nt > 0).then_some(nt));
                let ctx = Context::new(16 * L + 8, 16 * L + 8);
                let x = ctx
                    .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
                    .as_input();
                let a = ctx
                    .new_tensor(types::F32, &[m as i64, n_used as i64, n_tok as i64])
                    .as_input();
                let ab = ctx
                    .new_tensor(types::BF16, &[m as i64, n_used as i64, n_tok as i64])
                    .as_input();
                let id = ctx
                    .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                    .as_input();
                let mut graph = ctx.new_graph(16 * L + 8);
                for i in 0..L {
                    let s = Some(st);
                    let out = match what {
                        "w4a16-glu" => nvfp4_mm::swiglu_id(&ctx, gs[i], us[i], x, id, s, s),
                        "a4-chain" => {
                            let up = nvfp4_a4::mul_mat_id(&ctx, us[i], x, id, s);
                            let gate = nvfp4_a4::mul_mat_id(&ctx, gs[i], x, id, s);
                            ctx.mul(ctx.silu(gate), up)
                        }
                        "a4-glu" => nvfp4_a4::swiglu_id(&ctx, gs[i], us[i], x, id, s, s),
                        "w4a16-down" => nvfp4_mm::mul_mat_id(&ctx, ds[i], ab, id, s),
                        _ => nvfp4_a4::mul_mat_id(&ctx, ds[i], a, id, s),
                    };
                    graph.expand(out.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                match what {
                    "w4a16-down" => ab.set_bytes(&vec![0x3cu8; m * n_used * n_tok * 2]),
                    "a4-down" => a.set_f32(&activations(m, n_used * n_tok, 10)),
                    _ => x.set_f32(&activations(k, n_tok, 9)),
                }
                id.set_i32(&ids);
                let s = time(&mut gpu, &graph, 5) / L as f64;
                nvfp4_a4::force_tokens_tile(None);
                line += &format!("  {what}/{nt} {:7.1}", s * 1e6);
            }
            println!("{line} us");
        }
    }
    /// Routed experts at Qwen3.8-Flash-Next's shapes (k 2560, m 640, 512 experts, top-10) in
    /// decode steps, µs per layer and GB/s of the experts' weights read: the gate/up half
    /// (gate, up, SwiGLU) and the down matmul on ggml's MMVQ (the default up to 8 tokens), on
    /// the W4A16 grouped matvec (the default above), on the W4A4 decode kernel (its default
    /// shape, then each shape) and on the W4A4 GEMM. `LEGION_NVFP4_TIMING_TOKENS` (comma list)
    /// picks the steps.
    #[test]
    #[ignore = "needs a GPU with the FP4 kernels"]
    fn cuda_decode_timings() {
        const L: usize = 8;
        let Some(mut gpu) = gpu() else { return };
        let (k, m, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
        let wq = weights(k, m * n_expert, 5);
        let wd = weights(m, k * n_expert, 6);
        let weights = Context::new(3 * L + 1, 0);
        let new = |a: usize, b: usize| {
            weights.new_tensor(types::NVFP4, &[a as i64, b as i64, n_expert as i64])
        };
        let gs: Vec<_> = (0..L).map(|_| new(k, m)).collect();
        let us: Vec<_> = (0..L).map(|_| new(k, m)).collect();
        let ds: Vec<_> = (0..L).map(|_| new(m, k)).collect();
        let st = weights.new_tensor(types::F32, &[n_expert as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for w in gs.iter().chain(&us) {
            w.set_bytes(&wq);
        }
        for w in &ds {
            w.set_bytes(&wd);
        }
        st.set_f32(&vec![0.01; n_expert]);
        let expert_bytes = (k * m) as f64 / 64.0 * 36.0;
        let tok_list: Vec<usize> = std::env::var("LEGION_NVFP4_TIMING_TOKENS")
            .unwrap_or_else(|_| "4,8,9,12,16,32,64".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        #[derive(Clone, Copy, PartialEq)]
        enum Way {
            Ggml,
            W4a16,
            A4(Option<Grouped>),
        }
        // (label, way, parts: the gate/up half, the down matmul)
        let mut ways = vec![
            ("ggml".to_string(), Way::Ggml, [true, true]),
            ("w4a16".into(), Way::W4a16, [true, true]),
            ("a4".into(), Way::A4(None), [true, true]),
            ("a4 gemm".into(), Way::A4(Some(Grouped::Gemm)), [true, true]),
        ];
        for glu in [true, false] {
            for g in mv_shapes(glu) {
                let Grouped::Mv { rt, kw } = g else {
                    unreachable!()
                };
                let what = if glu { "glu" } else { "down" };
                ways.push((
                    format!("a4 {what} rt{rt} kw{kw}"),
                    Way::A4(Some(g)),
                    [glu, !glu],
                ));
            }
        }
        for n_tok in tok_list {
            let ids = routing(n_used, n_tok, n_expert, 11);
            let mut hit = vec![false; n_expert];
            for &e in &ids {
                hit[e as usize] = true;
            }
            // Bytes of one weight's experts that the step reads.
            let bytes = hit.iter().filter(|&&h| h).count() as f64 * expert_bytes;
            println!("tokens {n_tok}:");
            for (what, way, parts) in &ways {
                let mut line = format!("  {what:20}");
                let mut total = 0.0;
                for glu in [true, false] {
                    if !parts[if glu { 0 } else { 1 }] {
                        continue;
                    }
                    let ctx = Context::new(16 * L + 8, 16 * L + 8);
                    let x = ctx
                        .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
                        .as_input();
                    let a = ctx
                        .new_tensor(types::F32, &[m as i64, n_used as i64, n_tok as i64])
                        .as_input();
                    let id = ctx
                        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input();
                    let mut graph = ctx.new_graph(16 * L + 8);
                    for i in 0..L {
                        let s = Some(st);
                        let out = match (way, glu) {
                            (Way::Ggml, true) => {
                                let gate = ctx.mul_mat_id(gs[i], x, id);
                                ctx.mul(ctx.silu(gate), ctx.mul_mat_id(us[i], x, id))
                            }
                            (Way::Ggml, false) => ctx.mul_mat_id(ds[i], a, id),
                            (Way::W4a16, true) => {
                                let up = nvfp4_mm::mul_mat_id(&ctx, us[i], x, id, s);
                                let gate = nvfp4_mm::mul_mat_id(&ctx, gs[i], x, id, s);
                                ctx.mul(ctx.silu(gate), up)
                            }
                            (Way::W4a16, false) => nvfp4_mm::mul_mat_id(&ctx, ds[i], a, id, s),
                            (Way::A4(_), true) => {
                                nvfp4_a4::swiglu_id(&ctx, gs[i], us[i], x, id, s, s)
                            }
                            (Way::A4(_), false) => nvfp4_a4::mul_mat_id(&ctx, ds[i], a, id, s),
                        };
                        graph.expand(out.as_output());
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    if glu {
                        x.set_f32(&activations(k, n_tok, 9));
                    } else {
                        a.set_f32(&activations(m, n_used * n_tok, 10));
                    }
                    id.set_i32(&ids);
                    if let Way::A4(g) = way {
                        nvfp4_a4::force_grouped(*g);
                    }
                    let s = time(&mut gpu, &graph, 10) / L as f64;
                    nvfp4_a4::force_grouped(None);
                    let b = bytes * if glu { 2.0 } else { 1.0 };
                    line += &format!(
                        "  {} {:6.1} us {:4.0} GB/s",
                        if glu { "glu" } else { "down" },
                        s * 1e6,
                        b / s / 1e9
                    );
                    total += s;
                }
                if parts == &[true, true] {
                    line += &format!(
                        "  ffn {:6.1} us {:4.0} GB/s",
                        total * 1e6,
                        3.0 * bytes / total / 1e9
                    );
                }
                println!("{line}");
            }
        }
    }
}
