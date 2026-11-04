//! qwen4exp's fused hyper-connection mixer (`src/cuda/qwen4_hc.cu`) on a GPU against the graph
//! it replaces, at Qwen3.8-Flash-Next's shapes (n_embd 2560, 4 streams, rank 320, Q8_0 down
//! and up, F32 inject), for 1 and 3 tokens; and the time of both.
//!
//! `cargo test --release --features cuda --test qwen4_hc_cuda -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use legion::ggml::{ComputeAllocator, Context, DeviceKind, device_by_kind, quantize, types};
use legion::model::qwen4::hc;

#[test]
#[ignore = "needs a CUDA GPU"]
fn fused_mixer_matches_the_graph() {
    let dev = device_by_kind(DeviceKind::Gpu).expect("no GPU");
    let mut gpu = dev.init().unwrap();
    let (e, nh, rank) = (2560i64, 4i64, 320i64);
    let k = e * nh;
    let mut seed = 7u64;
    let mut rnd = move |s: f32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed % 20001) as f32 / 10000.0 - 1.0) * s
    };
    let down_f: Vec<f32> = (0..k * rank).map(|_| rnd(0.02)).collect();
    let up_f: Vec<f32> = (0..rank * k).map(|_| rnd(0.05)).collect();
    let norm_f: Vec<f32> = (0..k).map(|_| 1.0 + rnd(0.2)).collect();
    let inj_f: Vec<f32> = (0..k * nh).map(|_| rnd(0.01)).collect();
    // Eight mixers, as eight layers' worth, for the timing.
    const LAYERS: usize = 8;
    let weights = Context::new(4 * LAYERS, 0);
    let ws: Vec<_> = (0..LAYERS)
        .map(|_| {
            (
                weights.new_tensor(types::F32, &[k]),
                weights.new_tensor(types::Q8_0, &[k, rank]),
                weights.new_tensor(types::Q8_0, &[rank, k]),
                weights.new_tensor(types::F32, &[k, nh]),
            )
        })
        .collect();
    weights.alloc_tensors(dev.buffer_type()).unwrap();
    let (dq, uq) = (
        quantize(types::Q8_0, &down_f, k as usize),
        quantize(types::Q8_0, &up_f, rank as usize),
    );
    for (n, d, u, i) in &ws {
        n.set_f32(&norm_f);
        d.set_bytes(&dq);
        u.set_bytes(&uq);
        i.set_f32(&inj_f);
    }
    for (n_tok, combine) in [(1i64, false), (3, false), (1, true), (3, true)] {
        let x_f: Vec<f32> = (0..k * n_tok).map(|_| rnd(3.0)).collect();
        let out_f: Vec<f32> = (0..e * n_tok).map(|_| rnd(2.0)).collect();
        let pinj_f: Vec<f32> = (0..nh * n_tok).map(|_| rnd(4.0)).collect();
        let mut results = Vec::new();
        for ours in [false, true] {
            let ctx = Context::new(64 * LAYERS, 64 * LAYERS);
            let x0 = ctx.new_tensor(types::F32, &[e, nh, n_tok]);
            let pout = ctx.new_tensor(types::F32, &[e, n_tok]);
            let pinj = ctx.new_tensor(types::F32, &[nh, n_tok]);
            ctx.alloc_tensors(dev.buffer_type()).unwrap();
            x0.set_f32(&x_f);
            pout.set_f32(&out_f);
            pinj.set_f32(&pinj_f);
            let mut outs = Vec::new();
            for &(norm, down, up, inj) in &ws {
                assert!(hc::fits(x0, norm, down, up, Some(inj)));
                let (m, i, r) = if ours {
                    let prev = combine.then_some((pout, pinj));
                    let (m, i, r) =
                        hc::combine_mix(&ctx, x0, prev, norm, down, up, Some(inj), 1e-6);
                    (m, i.unwrap(), r)
                } else {
                    // hc_combine as qwen4.rs builds it, then the mixer graph.
                    let x = if combine {
                        let w = ctx.scale(ctx.sigmoid(ctx.scale(pinj, 1.0 / nh as f32)), 2.0);
                        ctx.hc_post(pout, x0, w)
                    } else {
                        x0
                    };
                    let xn = ctx.mul(ctx.rms_norm(x, 1e-6), ctx.reshape_2d(norm, e, nh));
                    let xn = ctx.reshape_2d(xn, k, n_tok);
                    let lo = ctx.silu(ctx.scale(ctx.mul_mat(down, xn), 1.0 / nh as f32));
                    let gate = ctx.mul_mat(up, lo);
                    let m = ctx.hc_pre_gated(
                        ctx.reshape_3d(xn, e, nh, n_tok),
                        ctx.reshape_3d(gate, e, nh, n_tok),
                        1.0 / nh as f32,
                    );
                    (m, ctx.mul_mat(inj, xn), x)
                };
                outs.push((ctx.cont(m), ctx.cont(i), ctx.cont(r)));
            }
            let mut g = ctx.new_graph(64 * LAYERS);
            for (m, i, r) in &outs {
                g.expand(*m);
                g.expand(*i);
                g.expand(*r);
            }
            ctx.alloc_graph(&g, ComputeAllocator::new(dev.buffer_type()))
                .unwrap();
            gpu.compute(&g).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                gpu.compute(&g).unwrap();
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / (20 * LAYERS) as f64;
            println!(
                "{n_tok} tokens, combine {combine}, {}: {us:.1} µs per mixer",
                if ours { "fused op" } else { "graph" }
            );
            results.push((outs[0].0.to_f32(), outs[0].1.to_f32(), outs[0].2.to_f32()));
        }
        let rel = |a: &[f32], b: &[f32]| {
            let s = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
                / s
        };
        let (g, o) = (&results[0], &results[1]);
        assert!(o.0.iter().all(|v| v.is_finite()));
        let (dm, di) = (rel(&o.0, &g.0), rel(&o.1, &g.1));
        println!("{n_tok} tokens, combine {combine}: mixed rel. diff {dm:e}, inject {di:e}");
        // The combined streams as ggml computes them, bit for bit.
        assert!(o.2 == g.2, "combined streams differ");
        assert!(dm < 1e-3, "mixed: {dm}");
        assert!(di < 1e-4, "inject: {di}");
    }
}
