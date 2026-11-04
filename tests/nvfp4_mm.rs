//! Legion's W4A16 NVFP4 matmuls (`legion::model::nvfp4_mm`): the CPU references against exact
//! products, and (ignored, `--features cuda`) every CUDA kernel against the CPU references and
//! timed against ggml's NVFP4 MMVQ / MMQ with 8-bit activations (what `--nvfp4-act a8` runs).
//!
//! `cargo test --release --features cuda --test nvfp4_mm -- --include-ignored --nocapture --test-threads 1`

use legion::ggml::{Backend, ComputeAllocator, Context, types};
use legion::model::nvfp4_mm;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Random NVFP4 rows: codes uniform (both signs, every magnitude), sub-block scales with
/// exponents 3..=10 and a few subnormal and zero scales, sign bits clear (as the converter
/// writes them).
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

fn activations(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut next = rng(seed ^ 0x55);
    (0..k * n)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0)
        .collect()
}

/// `scale · (w · x)` through `nvfp4_mm::mul_mat` on `backend`.
fn run(
    backend: &mut Backend,
    wq: &[u8],
    xs: &[f32],
    scale: Option<f32>,
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
    let out = nvfp4_mm::mul_mat(&ctx, w, x, scale.map(|_| s)).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

fn bf16(v: f32) -> f64 {
    // Round to nearest even, as ggml's F32 -> BF16 cast.
    let b = v.to_bits();
    let r = (b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r) as f64
}

/// Max |a - b| over max |b|.
fn rel_err(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |s, v| s.max(v.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(&x, y)| (x as f64 - y).abs())
        .fold(0f64, f64::max)
        / scale
}

#[test]
fn cpu_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n) = (256, 40, 3);
    let wq = weights(k, m, 1);
    let xs = activations(k, n, 2);
    let got = run(&mut cpu, &wq, &xs, Some(0.375), (k, m, n));
    let wf = legion::ggml::dequantize(types::NVFP4, &wq, k * m);
    let want: Vec<f64> = (0..n)
        .flat_map(|t| {
            let (wf, xs) = (&wf, &xs);
            (0..m).map(move |j| {
                0.375
                    * (0..k)
                        .map(|i| wf[j * k + i] as f64 * bf16(xs[t * k + i]))
                        .sum::<f64>()
            })
        })
        .collect();
    let e = rel_err(&got, &want);
    assert!(e < 1e-6, "relative error {e}");
}

/// `scale[e] · (w_e · x)` per (token, slot) through `nvfp4_mm::mul_mat_id` on `backend`:
/// experts `[k, m, n_expert]`, x `[k, n_x, n_tok]`, ids `[n_used, n_tok]`.
#[allow(clippy::too_many_arguments)]
fn run_id(
    backend: &mut Backend,
    wq: &[u8],
    xs: &[f32],
    ids: &[i32],
    scale: Option<&[f32]>,
    (k, m, n_expert): (usize, usize, usize),
    (n_x, n_used, n_tok): (usize, usize, usize),
) -> Vec<f32> {
    let weights = Context::new(2, 0);
    let w = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let s = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    w.set_bytes(wq);
    s.set_f32(scale.unwrap_or(&vec![1.0; n_expert]));
    let ctx = Context::new(8, 16);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let out = nvfp4_mm::mul_mat_id(&ctx, w, x, id, scale.map(|_| s));
    let out = ctx.cont(out).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

/// Router choices: `n_used` distinct experts per token out of the first `n_active`.
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

#[test]
fn cpu_mul_mat_id_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n_expert, n_used, n_tok) = (128, 24, 5, 2, 3);
    let wq = weights(k, m * n_expert, 3);
    let wf = legion::ggml::dequantize(types::NVFP4, &wq, k * m * n_expert);
    let scale = [0.5, 0.25, 2.0, 1.0, 0.125];
    let ids = routing(n_used, n_tok, n_expert, 4);
    for n_x in [1, n_used] {
        let xs = activations(k, n_x * n_tok, 5);
        let got = run_id(
            &mut cpu,
            &wq,
            &xs,
            &ids,
            Some(&scale),
            (k, m, n_expert),
            (n_x, n_used, n_tok),
        );
        let mut want = Vec::new();
        for t in 0..n_tok {
            for s in 0..n_used {
                let e = ids[t * n_used + s] as usize;
                let xr = (t * n_x + if n_x == 1 { 0 } else { s }) * k;
                for j in 0..m {
                    let row = (e * m + j) * k;
                    let d: f64 = (0..k).map(|i| wf[row + i] as f64 * bf16(xs[xr + i])).sum();
                    want.push(d * scale[e] as f64);
                }
            }
        }
        let e = rel_err(&got, &want);
        assert!(e < 1e-6, "n_x {n_x}: relative error {e}");
    }
}

/// `silu(sg · (wg · x)) · (su · (wu · x))` through `nvfp4_mm::swiglu` on `backend`.
fn run_swiglu(
    backend: &mut Backend,
    w: (&[u8], &[u8]),
    xs: &[f32],
    scales: (Option<f32>, Option<f32>),
    shape: (usize, usize, usize),
) -> Vec<f32> {
    run_glu(backend, w, xs, scales, shape, false)
}

/// `swiglu`, or with `gelu` `gelu_tanh(sg · (wg · x)) · (su · (wu · x))` through
/// `nvfp4_mm::geglu`, on `backend`.
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
        nvfp4_mm::geglu(&ctx, g, u, x, sg, su)
    } else {
        nvfp4_mm::swiglu(&ctx, g, u, x, sg, su)
    };
    let out = ctx.cast(out, types::F32).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

#[test]
fn cpu_swiglu_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n) = (192, 24, 5);
    let (wg, wu) = (weights(k, m, 7), weights(k, m, 8));
    let xs = activations(k, n, 9);
    let got = run_swiglu(&mut cpu, (&wg, &wu), &xs, (Some(0.02), None), (k, m, n));
    let (fg, fu) = (
        legion::ggml::dequantize(types::NVFP4, &wg, k * m),
        legion::ggml::dequantize(types::NVFP4, &wu, k * m),
    );
    let mut want = Vec::new();
    for t in 0..n {
        for j in 0..m {
            let dot = |f: &[f32]| -> f64 {
                (0..k)
                    .map(|i| f[j * k + i] as f64 * bf16(xs[t * k + i]))
                    .sum()
            };
            let g = 0.02 * dot(&fg);
            want.push(bf16((g / (1.0 + (-g).exp()) * dot(&fu)) as f32));
        }
    }
    let e = rel_err(&got, &want);
    // BF16 outputs: one rounding apart where the F32 sums differ in their last bits.
    assert!(e < 1e-2, "relative error {e}");
}

/// The tanh GELU of the fused GeGLU (Gemma 4's FFN).
fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * x * (1.0 + 0.044715 * x * x)).tanh())
}

#[test]
fn cpu_geglu_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n) = (192, 24, 5);
    let (wg, wu) = (weights(k, m, 7), weights(k, m, 8));
    let xs = activations(k, n, 9);
    let got = run_glu(
        &mut cpu,
        (&wg, &wu),
        &xs,
        (Some(0.02), Some(0.5)),
        (k, m, n),
        true,
    );
    let (fg, fu) = (
        legion::ggml::dequantize(types::NVFP4, &wg, k * m),
        legion::ggml::dequantize(types::NVFP4, &wu, k * m),
    );
    let mut want = Vec::new();
    for t in 0..n {
        for j in 0..m {
            let dot = |f: &[f32]| -> f64 {
                (0..k)
                    .map(|i| f[j * k + i] as f64 * bf16(xs[t * k + i]))
                    .sum()
            };
            want.push(bf16((gelu_tanh(0.02 * dot(&fg)) * 0.5 * dot(&fu)) as f32));
        }
    }
    let e = rel_err(&got, &want);
    assert!(e < 1e-2, "relative error {e}");
}

/// The routed SwiGLU of NVFP4 experts on `backend`, as F32 values of the BF16 down-matmul
/// input `[m, n_used, n_tok]`: `fused` through `nvfp4_mm::swiglu_id`, else the graph's chain
/// (`mul_mat_id` up and gate, ggml's `silu` and `mul`, the cast to BF16). A `limit` above 0:
/// GLM-5.3-Flash's clamped SwiGLU, fused through `nvfp4_mm::swiglu_id_clamped`, else its graph
/// chain (`glm5next::ops::glu` and `bf16`).
#[allow(clippy::too_many_arguments)]
fn run_swiglu_id(
    backend: &mut Backend,
    (wg, wu): (&[u8], &[u8]),
    xs: &[f32],
    ids: &[i32],
    (sg, su): (Option<&[f32]>, Option<&[f32]>),
    (k, m, n_expert): (usize, usize, usize),
    (n_x, n_used, n_tok): (usize, usize, usize),
    fused: bool,
    limit: f32,
) -> Vec<f32> {
    let weights = Context::new(4, 0);
    let g = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let u = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
    let tg = weights.new_tensor(types::F32, &[n_expert as i64]);
    let tu = weights.new_tensor(types::F32, &[n_expert as i64]);
    weights.alloc_tensors(backend.buffer_type()).unwrap();
    g.set_bytes(wg);
    u.set_bytes(wu);
    let ones = vec![1.0; n_expert];
    tg.set_f32(sg.unwrap_or(&ones));
    tu.set_f32(su.unwrap_or(&ones));
    let ctx = Context::new(16, 32);
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
        .as_input();
    let id = ctx
        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
        .as_input();
    let (sg, su) = (sg.map(|_| tg), su.map(|_| tu));
    let act = if fused && limit > 0.0 {
        nvfp4_mm::swiglu_id_clamped(&ctx, g, u, x, id, sg, su, limit)
    } else if fused {
        nvfp4_mm::swiglu_id(&ctx, g, u, x, id, sg, su)
    } else if limit > 0.0 {
        use legion::model::glm5next::ops;
        let up = nvfp4_mm::mul_mat_id(&ctx, u, x, id, su);
        let gate = nvfp4_mm::mul_mat_id(&ctx, g, x, id, sg);
        ops::bf16(&ctx, ops::glu(&ctx, gate, up, limit))
    } else {
        let up = nvfp4_mm::mul_mat_id(&ctx, u, x, id, su);
        let gate = nvfp4_mm::mul_mat_id(&ctx, g, x, id, sg);
        ctx.cast(ctx.mul(ctx.silu(gate), up), types::BF16)
    };
    let out = ctx.cast(ctx.cont(act), types::F32).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(xs);
    id.set_i32(ids);
    backend.compute(&graph).unwrap();
    out.to_f32()
}

#[test]
fn cpu_swiglu_id_matches_exact_product() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let (k, m, n_expert, n_used, n_tok) = (128, 24, 5, 2, 3);
    let (wg, wu) = (weights(k, m * n_expert, 21), weights(k, m * n_expert, 22));
    let (fg, fu) = (
        legion::ggml::dequantize(types::NVFP4, &wg, k * m * n_expert),
        legion::ggml::dequantize(types::NVFP4, &wu, k * m * n_expert),
    );
    let sg = [0.05, 0.025, 0.2, 0.1, 0.0125];
    let ids = routing(n_used, n_tok, n_expert, 23);
    // 0: plain SwiGLU; 2: GLM-5.3-Flash's clamp, which cuts many of these sums
    for (n_x, limit) in [(1, 0.0), (n_used, 0.0), (1, 2.0), (n_used, 2.0)] {
        let xs = activations(k, n_x * n_tok, 24);
        let got = run_swiglu_id(
            &mut cpu,
            (&wg, &wu),
            &xs,
            &ids,
            (Some(&sg), None),
            (k, m, n_expert),
            (n_x, n_used, n_tok),
            true,
            limit,
        );
        let (mut want, mut clamped) = (Vec::new(), 0);
        for t in 0..n_tok {
            for s in 0..n_used {
                let e = ids[t * n_used + s] as usize;
                let xr = (t * n_x + if n_x == 1 { 0 } else { s }) * k;
                for j in 0..m {
                    let row = (e * m + j) * k;
                    let dot = |f: &[f32]| -> f64 {
                        (0..k).map(|i| f[row + i] as f64 * bf16(xs[xr + i])).sum()
                    };
                    let (mut g, mut u) = (sg[e] as f64 * dot(&fg), dot(&fu));
                    if limit > 0.0 {
                        let l = limit as f64;
                        clamped += (g > l || u.abs() > l) as usize;
                        (g, u) = (g.min(l), u.clamp(-l, l));
                    }
                    want.push(bf16((g / (1.0 + (-g).exp()) * u) as f32));
                }
            }
        }
        assert!(
            limit == 0.0 || clamped > want.len() / 4,
            "{clamped} clamped"
        );
        let e = rel_err(&got, &want);
        // BF16 outputs: one rounding apart where the F32 sums differ in their last bits.
        assert!(e < 1e-2, "n_x {n_x} limit {limit}: relative error {e}");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use legion::ggml::{DeviceKind, device_by_kind, ffi};
    use nvfp4_mm::Kernel;

    fn gpu() -> Option<Backend> {
        let gpu = device_by_kind(DeviceKind::Gpu)
            .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
            .expect("no GPU")
            .init()
            .unwrap();
        if !nvfp4_mm::device_supported() {
            println!("skipped: the NVFP4 kernels need BF16 tensor cores (sm_80+) on this GPU");
            return None;
        }
        Some(gpu)
    }

    /// The grouped SwiGLU mv and the down mv on its plan (#836), in every compiled variant of
    /// the grouped mv kernel, give the unfused chain's bits (Qwen3.8-Flash-Next's expert shapes,
    /// decode batches; the timed version is `legion nvfp4-mv-bench`).
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_grouped_mv_variants_match_the_chain() {
        use nvfp4_mm::bench;
        let Some(mut gpu) = gpu() else { return };
        let cfg = bench::Config {
            layers: 2,
            toks: vec![1, 9, 16],
            pools: vec![16, 512],
            sets: 1,
            rounds: 0,
            ..bench::Config::default()
        };
        let rows = bench::run(&mut gpu, &cfg).unwrap();
        for r in &rows {
            println!(
                "{} tokens {} pool {}: {:?}",
                r.variant, r.n_tok, r.pool, r.differ
            );
            // Flash-Next's 512 experts: one pair per expert used, so every variant compiles.
            assert_eq!(r.differ, Some(0), "{r:?}");
        }
    }

    /// A verify step's row map on the grouped mvs (#836): with the padding rows skipped
    /// (`set_pad_skip`, the default) the real rows' SwiGLU and down results are the bits of the
    /// remap (padding routed as the first row) and the padding rows' results are zeros. Token 1
    /// shares experts with token 0, so the remap puts more than 8 pairs on those experts.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_padding_rows_skipped_keep_real_bits() {
        let Some(mut gpu) = gpu() else { return };
        let (k, m, n_expert, n_used, n_tok, real) = (256usize, 128usize, 64usize, 10, 9, 3);
        let wctx = Context::new(4, 0);
        let wg = wctx.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
        let wu = wctx.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]);
        let wd = wctx.new_tensor(types::NVFP4, &[m as i64, k as i64, n_expert as i64]);
        wctx.alloc_tensors(gpu.buffer_type()).unwrap();
        wg.set_bytes(&weights(k, m * n_expert, 1));
        wu.set_bytes(&weights(k, m * n_expert, 2));
        wd.set_bytes(&weights(m, k * n_expert, 3));
        let mut ids = routing(n_used, n_tok, n_expert, 4);
        // Token 1: half of token 0's experts.
        for s in 0..n_used / 2 {
            let e = ids[s];
            if !ids[n_used..2 * n_used].contains(&e) {
                ids[n_used + s] = e;
            }
        }
        let map: Vec<i32> = (0..n_tok as i32)
            .map(|t| if t < real as i32 { t } else { 0 })
            .collect();
        let xs = activations(k, n_tok, 5);
        let ctx = Context::new(16, 32);
        let x = ctx
            .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
            .as_input();
        let id = ctx
            .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
            .as_input();
        let rows = ctx.new_tensor(types::I32, &[n_tok as i64]).as_input();
        let act = nvfp4_mm::swiglu_id_rows(&ctx, wg, wu, x, id, None, None, Some(rows));
        let down = nvfp4_mm::mul_mat_id(&ctx, wd, act, id, None).as_output();
        let act_out = ctx.cast(ctx.cont(act), types::F32).as_output();
        let mut graph = ctx.new_graph(32);
        graph.expand(down);
        graph.expand(act_out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        x.set_f32(&xs);
        id.set_i32(&ids);
        rows.set_i32(&map);
        let mut run = |skip: bool| {
            nvfp4_mm::set_pad_skip(skip);
            gpu.compute(&graph).unwrap();
            (down.to_f32(), act_out.to_f32())
        };
        let (remap_down, remap_act) = run(false);
        let (skip_down, skip_act) = run(true);
        nvfp4_mm::set_pad_skip(true);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let (d_real, a_real) = (real * n_used * k, real * n_used * m);
        assert_eq!(
            bits(&remap_down[..d_real]),
            bits(&skip_down[..d_real]),
            "down, real rows"
        );
        assert_eq!(
            bits(&remap_act[..a_real]),
            bits(&skip_act[..a_real]),
            "SwiGLU, real rows"
        );
        assert!(
            skip_down[d_real..].iter().all(|&v| v.to_bits() == 0),
            "down, padding rows"
        );
        assert!(
            skip_act[a_real..].iter().all(|&v| v.to_bits() == 0),
            "SwiGLU, padding rows"
        );
        assert!(
            remap_down[d_real..].iter().any(|&v| v != 0.0),
            "the remap computes padding"
        );
    }

    /// Every compiled variant that fits `n` tokens per tile, and the default choice (`None`).
    fn kernels(n: usize) -> Vec<Option<Kernel>> {
        let mut out = vec![None];
        for nt in [1, 2, 4] {
            for rtw in [1, 2] {
                for kw in [4, 8] {
                    if 8 * nt as usize >= n.min(32) {
                        let rpts: &[i32] = if (rtw, kw) == (1, 8) {
                            &[1, 2, 4, 8, 16]
                        } else {
                            &[1]
                        };
                        for &rpt in rpts {
                            out.push(Some(Kernel::Mv { nt, rtw, kw, rpt }));
                        }
                    }
                }
            }
        }
        for mi in [1, 2] {
            for nt in [1, 2, 4, 8, 16] {
                out.push(Some(Kernel::Gemm { mi, nt }));
            }
        }
        out
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_matches_cpu() {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let Some(mut gpu) = gpu() else { return };
        for (k, m, n, scale) in [
            (64, 1, 1, None),
            (256, 130, 33, Some(0.5)),
            (1856, 70, 5, Some(0.25)),
            (2688, 300, 129, Some(3.0e-3)),
            (5120, 257, 200, None),
            (512, 128, 16, Some(1.0)),
            (1024, 1000, 7, Some(2.0)),
            (192, 48, 32, Some(1.0)),
        ] {
            let wq = weights(k, m, (k + m) as u64);
            let xs = activations(k, n, n as u64);
            let want: Vec<f64> = run(&mut cpu, &wq, &xs, scale, (k, m, n))
                .iter()
                .map(|&v| v as f64)
                .collect();
            for kernel in kernels(n) {
                nvfp4_mm::force_kernel(kernel);
                let got = run(&mut gpu, &wq, &xs, scale, (k, m, n));
                let e = rel_err(&got, &want);
                assert!(e < 1e-5, "k {k} m {m} n {n} {kernel:?}: relative error {e}");
            }
            nvfp4_mm::force_kernel(None);
            println!("k {k} m {m} n {n}: all kernels match");
        }
    }

    /// The GEMM's bits do not depend on how full a token tile is (full tiles skip the per-n8
    /// checks) or on where a row's blocks sit in their 16-byte words (the weights are copied as
    /// the aligned words around each block): the first 200 of 256 tokens equal a 200-token call
    /// (a partial last tile), and rows 1.. equal a call on the weights without row 0 (with k =
    /// 1856 a row is 1044 bytes, so dropping a row moves every row's blocks by 4 bytes).
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_gemm_bits_ignore_tile_fill_and_row_phase() {
        let Some(mut gpu) = gpu() else { return };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (k, m) in [(1856usize, 70usize), (5120, 300)] {
            let wq = weights(k, m, 31);
            let row = k / 64 * 36;
            let xs = activations(k, 256, 32);
            for (mi, nt) in [(1, 4), (1, 16), (2, 8), (2, 16)] {
                nvfp4_mm::force_kernel(Some(Kernel::Gemm { mi, nt }));
                let full = run(&mut gpu, &wq, &xs, Some(0.5), (k, m, 256));
                let part = run(&mut gpu, &wq, &xs[..200 * k], Some(0.5), (k, m, 200));
                assert_eq!(
                    bits(&full[..200 * m]),
                    bits(&part),
                    "k {k} m {m} mi {mi} nt {nt}: tile fill changed bits"
                );
                let shifted = run(&mut gpu, &wq[row..], &xs, Some(0.5), (k, m - 1, 256));
                for t in 0..256 {
                    assert_eq!(
                        bits(&full[t * m + 1..(t + 1) * m]),
                        bits(&shifted[t * (m - 1)..(t + 1) * (m - 1)]),
                        "k {k} m {m} mi {mi} nt {nt} token {t}: row phase changed bits"
                    );
                }
            }
            nvfp4_mm::force_kernel(None);
            println!("k {k} m {m}: bitwise across tile fill and row phase");
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_mul_mat_id_matches_cpu() {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let Some(mut gpu) = gpu() else { return };
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (128, 40, 4, 2, 1, 4),
            (256, 100, 16, 6, 16, 16),
            (1856, 48, 8, 3, 37, 5),
            (192, 130, 32, 4, 70, 32),
            // Expert counts that are not a multiple of 32 (the tile lookup's runs of experts per
            // lane, some lanes without any), and 512 experts at a decode batch.
            (128, 40, 100, 6, 33, 77),
            (128, 24, 512, 10, 16, 512),
        ] {
            let wq = weights(k, m * n_expert, k as u64);
            let scale: Vec<f32> = (0..n_expert).map(|e| 0.5 + e as f32 / 8.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            for n_x in [1, n_used] {
                let xs = activations(k, n_x * n_tok, m as u64);
                let shape = ((k, m, n_expert), (n_x, n_used, n_tok));
                let want: Vec<f64> =
                    run_id(&mut cpu, &wq, &xs, &ids, Some(&scale), shape.0, shape.1)
                        .iter()
                        .map(|&v| v as f64)
                        .collect();
                for kernel in kernels(1) {
                    nvfp4_mm::force_kernel(kernel);
                    let got = run_id(&mut gpu, &wq, &xs, &ids, Some(&scale), shape.0, shape.1);
                    let e = rel_err(&got, &want);
                    assert!(e < 1e-5, "{shape:?} {kernel:?}: relative error {e}");
                }
                nvfp4_mm::force_kernel(None);
            }
            println!("k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok}: all kernels match");
        }
    }

    /// The mv kernel's row tiles per CTA (`rpt`) give the one-tile kernel's bits: Qwen3.8-Flash-
    /// Next's expert shapes at decode batches, row counts that leave the last CTA partly or
    /// wholly past the matrix, and dense matmuls.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_mv_row_tiles_bitwise() {
        let Some(mut gpu) = gpu() else { return };
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (2560, 640, 512, 10, 16, 512),
            (640, 2560, 512, 10, 16, 64),
            (640, 2560, 512, 10, 32, 512),
            (2560, 640, 512, 10, 9, 512),
            (2560, 640, 512, 10, 1024, 512),
            (640, 200, 32, 4, 16, 32),
            (1024, 48, 16, 6, 5, 16),
        ] {
            let wq = weights(k, m * n_expert, k as u64);
            let scale: Vec<f32> = (0..n_expert).map(|e| 0.5 + e as f32 / 8.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            let (pairs, ne) = ((n_used * n_tok) as i64, n_expert as i64);
            let picked = nvfp4_mm::pick_grouped(k as i64, m as i64, pairs, ne);
            for n_x in [1, n_used] {
                let xs = activations(k, n_x * n_tok, m as u64);
                let shape = ((k, m, n_expert), (n_x, n_used, n_tok));
                let mut go = |kernel| {
                    nvfp4_mm::force_kernel(kernel);
                    let r = run_id(&mut gpu, &wq, &xs, &ids, Some(&scale), shape.0, shape.1);
                    nvfp4_mm::force_kernel(None);
                    bits(r)
                };
                for nt in [1, 2, 4] {
                    let mv = |rpt| {
                        Some(Kernel::Mv {
                            nt,
                            rtw: 1,
                            kw: 8,
                            rpt,
                        })
                    };
                    let want = go(mv(1));
                    for rpt in [2, 4, 8, 16] {
                        assert!(
                            go(mv(rpt)) == want,
                            "{shape:?} nt {nt} rpt {rpt}: other bits"
                        );
                    }
                    // The default kernel (rpt > 1) against its one-tile form.
                    if let Kernel::Mv { nt: pnt, .. } = picked
                        && pnt == nt
                    {
                        assert!(go(None) == want, "{shape:?}: the default, other bits");
                    }
                }
            }
            println!("k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok}: bitwise");
        }
        for (k, m, n) in [(2560, 4096, 16), (640, 200, 3), (5120, 17408, 32)] {
            let wq = weights(k, m, (k + m) as u64);
            let xs = activations(k, n, n as u64);
            for nt in [1, 2, 4] {
                let mut go = |rpt| {
                    nvfp4_mm::force_kernel(Some(Kernel::Mv {
                        nt,
                        rtw: 1,
                        kw: 8,
                        rpt,
                    }));
                    let r = run(&mut gpu, &wq, &xs, Some(0.5), (k, m, n));
                    nvfp4_mm::force_kernel(None);
                    bits(r)
                };
                let want = go(1);
                for rpt in [2, 4, 8, 16] {
                    assert!(go(rpt) == want, "dense k {k} m {m} n {n} nt {nt} rpt {rpt}");
                }
            }
            println!("dense k {k} m {m} n {n}: bitwise");
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_swiglu_matches_cpu() {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let Some(mut gpu) = gpu() else { return };
        for (k, m, n, scales) in [
            (64, 1, 1, (None, None)),
            (512, 130, 9, (Some(0.05), Some(0.5))),
            (2688, 300, 16, (Some(0.01), None)),
            (5120, 77, 32, (None, Some(2.0))),
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
            // BF16 outputs: values may round apart where the F32 sums differ in the last bits.
            let off = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| (**g as f64 - **w).abs() > 1e-5 * w.abs().max(1e-3))
                .count();
            println!("swiglu k {k} m {m} n {n}: relative error {e:.2e}, {off} rounded apart");
            assert!(
                e < 1e-2 && off * 50 < got.len(),
                "k {k} m {m} n {n}: {e}, {off}"
            );
        }
    }

    /// The fused GeGLU (Gemma 4's FFN) on the GEMM and the dense mv.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_geglu_matches_cpu() {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let Some(mut gpu) = gpu() else { return };
        for (k, m, n, scales) in [
            (512, 130, 9, (Some(0.05), Some(0.5))),
            (5376, 77, 33, (None, None)),
            (1024, 130, 300, (Some(0.3), Some(0.7))),
        ] {
            let (wg, wu) = (weights(k, m, 3 * k as u64), weights(k, m, m as u64));
            let xs = activations(k, n, n as u64);
            let want: Vec<f64> = run_glu(&mut cpu, (&wg, &wu), &xs, scales, (k, m, n), true)
                .iter()
                .map(|&v| v as f64)
                .collect();
            let got = run_glu(&mut gpu, (&wg, &wu), &xs, scales, (k, m, n), true);
            let e = rel_err(&got, &want);
            println!("geglu k {k} m {m} n {n}: relative error {e:.2e}");
            assert!(e < 1e-2, "k {k} m {m} n {n}: {e}");
        }
    }

    /// `swiglu_id` gives the graph chain's bits (two grouped matmuls, ggml's silu and mul, the
    /// cast to BF16) at prefill batches (the GEMM) and decode ones (the grouped SwiGLU mv, #836),
    /// partial tiles included, and Qwen3.8-Flash-Next's shape; `swiglu_id_clamped` those of
    /// GLM-5.3-Flash's chain (`glm5next::ops::glu`, its shape).
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_swiglu_id_matches_unfused() {
        let Some(mut gpu) = gpu() else { return };
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (256, 200, 16, 6, 300, 16),
            // 8.75 pairs per expert: the grouped mv (#860), as the next two
            (192, 130, 32, 4, 70, 32),
            (256, 200, 16, 6, 20, 16),
            (2560, 640, 512, 10, 16, 512),
            (1024, 640, 64, 8, 1000, 40),
            (2560, 640, 512, 10, 4096, 512),
            (2560, 640, 512, 10, 1500, 512),
            (4096, 2048, 288, 8, 4096, 288),
        ] {
            let (wg, wu) = (weights(k, m * n_expert, 31), weights(k, m * n_expert, 32));
            let sg: Vec<f32> = (0..n_expert).map(|e| 0.01 + e as f32 / 4096.0).collect();
            let su: Vec<f32> = (0..n_expert).map(|e| 0.02 - e as f32 / 65536.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            let pairs = (n_used * n_tok) as i64;
            let picked = nvfp4_mm::pick_grouped(k as i64, m as i64, pairs, n_expert as i64);
            // limit 0: SwiGLU; 0.05: GLM-5.3-Flash's clamp (#575), cutting many of these sums
            for (n_x, limit) in [(1, 0.0), (n_used, 0.0), (1, 0.05)] {
                if n_x > 1 && n_tok > 1000 {
                    continue;
                }
                let xs = activations(k, n_x * n_tok, m as u64);
                let shape = ((k, m, n_expert), (n_x, n_used, n_tok));
                let scales = (Some(&sg[..]), Some(&su[..]));
                let run = |gpu: &mut Backend, fused| {
                    run_swiglu_id(
                        gpu,
                        (&wg, &wu),
                        &xs,
                        &ids,
                        scales,
                        shape.0,
                        shape.1,
                        fused,
                        limit,
                    )
                };
                let differ = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count()
                };
                // The chain on the GEMM, any tile (the same sums), sums like the fused GEMM.
                nvfp4_mm::force_kernel(Some(Kernel::Gemm { mi: 1, nt: 4 }));
                let gemm = run(&mut gpu, false);
                nvfp4_mm::force_kernel(None);
                let got = run(&mut gpu, true);
                // Where the fused op takes the grouped SwiGLU mv (decode batches, #836), it
                // keeps the sums of the chain's own kernel there, the grouped mv: the
                // reference is that chain, unforced (the same pick for the same pairs).
                let want = match picked {
                    Kernel::Gemm { .. } => gemm,
                    Kernel::Mv { .. } => {
                        let off = differ(&got, &gemm);
                        println!("  {shape:?}: {off} differ from the GEMM chain (other sums)");
                        run(&mut gpu, false)
                    }
                };
                let off = differ(&got, &want);
                assert_eq!(off, 0, "{shape:?}: {off} of {} differ", got.len());
            }
            println!(
                "swiglu_id k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok} ({picked:?}): \
                 bitwise"
            );
        }
    }

    /// The grouped GEMM's stage counts and occupancies (#824) give the default's bits, for the
    /// down `mul_mat_id` in every compiled tile and the fused `swiglu_id` in its 64- and
    /// 128-token tiles: partial tiles, row counts that leave the last tile partly empty, a k
    /// shorter than the stages and Qwen3.8-Flash-Next's shapes.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_gemm_tunes_keep_bits() {
        use nvfp4_mm::GemmTune;
        let Some(mut gpu) = gpu() else { return };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let tunes = [21, 22, 32, 41].map(GemmTune::from_code);
        // (k, m, n_expert, n_used, n_tok, experts routed to)
        for (k, m, n_expert, n_used, n_tok, n_active) in [
            (128, 200, 16, 6, 300, 16),
            (1856, 130, 32, 4, 700, 24),
            (640, 2560, 512, 10, 1500, 512),
            (2560, 640, 512, 10, 4096, 160),
        ] {
            let (wg, wu) = (weights(k, m * n_expert, 41), weights(k, m * n_expert, 42));
            let sc: Vec<f32> = (0..n_expert).map(|e| 0.01 + e as f32 / 4096.0).collect();
            let ids = routing(n_used, n_tok, n_active, n_tok as u64);
            let xs = activations(k, n_tok, m as u64);
            let shape = ((k, m, n_expert), (1, n_used, n_tok));
            let down = |gpu: &mut Backend| run_id(gpu, &wg, &xs, &ids, Some(&sc), shape.0, shape.1);
            let glu = |gpu: &mut Backend| {
                let s = Some(&sc[..]);
                run_swiglu_id(
                    gpu,
                    (&wg, &wu),
                    &xs,
                    &ids,
                    (s, s),
                    shape.0,
                    shape.1,
                    true,
                    0.0,
                )
            };
            for (mi, nt) in [(1, 8), (1, 16), (2, 8), (2, 16)] {
                nvfp4_mm::force_kernel(Some(Kernel::Gemm { mi, nt }));
                nvfp4_mm::force_gemm_tune(Some(nvfp4_mm::GEMM_DEFAULT), None);
                let want = bits(&down(&mut gpu));
                for tune in tunes
                    .into_iter()
                    .filter(|&t| nvfp4_mm::gemm_tune_fits(mi, nt, t))
                {
                    nvfp4_mm::force_gemm_tune(Some(tune), None);
                    let got = bits(&down(&mut gpu));
                    assert!(
                        got == want,
                        "{shape:?} mi {mi} nt {nt} {tune:?}: down bits differ"
                    );
                }
            }
            nvfp4_mm::force_kernel(None);
            for nt in [8, 16] {
                nvfp4_mm::force_gemm_tune(Some(nvfp4_mm::GEMM_DEFAULT), Some(nt));
                let want = bits(&glu(&mut gpu));
                for tune in tunes
                    .into_iter()
                    .filter(|&t| nvfp4_mm::gemm_tune_fits(2, nt, t))
                {
                    nvfp4_mm::force_gemm_tune(Some(tune), Some(nt));
                    let got = bits(&glu(&mut gpu));
                    assert!(
                        got == want,
                        "{shape:?} nt {nt} {tune:?}: swiglu bits differ"
                    );
                }
            }
            nvfp4_mm::force_gemm_tune(None, None);
            println!("k {k} m {m} experts {n_expert} x{n_used} tokens {n_tok}: every tune bitwise");
        }
    }

    /// Qwen3.8-Flash-Next's routed experts at prefill chunks (512 experts, 10 used): the fused
    /// gate/up/SwiGLU GEMM against the chain, and the down GEMM's tiles.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_swiglu_id_timings() {
        const L: usize = 4;
        let (k, m, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
        let Some(mut gpu) = gpu() else { return };
        let wq = weights(k, m * n_expert, 5);
        let wd = weights(m, k * n_expert, 6);
        let weights = Context::new(3 * L + 1, 0);
        let gs: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]))
            .collect();
        let us: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]))
            .collect();
        let ds: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &[m as i64, k as i64, n_expert as i64]))
            .collect();
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
            let flops = 2.0 * (k * m * n_used * n_tok) as f64;
            let mut line = format!("tokens {n_tok:4}:");
            for what in ["chain", "fused", "down", "down-mi2"] {
                let ctx = Context::new(16 * L + 8, 16 * L + 8);
                let x = ctx
                    .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
                    .as_input();
                let a = ctx
                    .new_tensor(types::BF16, &[m as i64, n_used as i64, n_tok as i64])
                    .as_input();
                let id = ctx
                    .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                    .as_input();
                let mut graph = ctx.new_graph(16 * L + 8);
                for i in 0..L {
                    let s = Some(st);
                    let out = match what {
                        "chain" => {
                            // As the graph runs it: each matmul rounds x to BF16.
                            let up = nvfp4_mm::mul_mat_id(&ctx, us[i], x, id, s);
                            let gate = nvfp4_mm::mul_mat_id(&ctx, gs[i], x, id, s);
                            ctx.cast(ctx.mul(ctx.silu(gate), up), types::BF16)
                        }
                        "fused" => nvfp4_mm::swiglu_id(&ctx, gs[i], us[i], x, id, s, s),
                        _ => nvfp4_mm::mul_mat_id(&ctx, ds[i], a, id, s),
                    };
                    graph.expand(out.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                if what == "chain" || what == "fused" {
                    x.set_f32(&activations(k, n_tok, 9));
                } else {
                    a.set_bytes(&vec![0x3cu8; m * n_used * n_tok * 2]);
                }
                id.set_i32(&ids);
                if what == "down-mi2" {
                    let nt = if n_tok >= 2048 { 16 } else { 8 };
                    nvfp4_mm::force_kernel(Some(Kernel::Gemm { mi: 2, nt }));
                }
                let s = time(&mut gpu, &graph, 5) / L as f64;
                nvfp4_mm::force_kernel(None);
                let f = if what == "chain" || what == "fused" {
                    2.0 * flops
                } else {
                    flops
                };
                line += &format!("  {what} {:7.1} us {:5.1} TFLOPS", s * 1e6, f / s / 1e12);
            }
            println!("{line}");
        }
    }

    /// Qwen3.8-Flash-Next's routed experts at decode batches (512 experts, 10 used; C=16 decode
    /// is 16 tokens, the grouped mv kernel): the gate/up/SwiGLU chain the graph runs, the fused
    /// GEMM (what `swiglu_id` would cost there: it is only used for prefill), the down matmul
    /// and its mv variants; GB/s of the used experts' weights.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_swiglu_id_decode_timings() {
        const L: usize = 4;
        let (k, m, n_expert, n_used) = (2560usize, 640usize, 512usize, 10usize);
        let Some(mut gpu) = gpu() else { return };
        let wq = weights(k, m * n_expert, 5);
        let wd = weights(m, k * n_expert, 6);
        let weights = Context::new(3 * L + 1, 0);
        let shape = |a: usize, b: usize| [a as i64, b as i64, n_expert as i64];
        let gs: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &shape(k, m)))
            .collect();
        let us: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &shape(k, m)))
            .collect();
        let ds: Vec<_> = (0..L)
            .map(|_| weights.new_tensor(types::NVFP4, &shape(m, k)))
            .collect();
        let st = weights.new_tensor(types::F32, &[n_expert as i64]);
        weights.alloc_tensors(gpu.buffer_type()).unwrap();
        for w in gs.iter().chain(&us) {
            w.set_bytes(&wq);
        }
        for w in &ds {
            w.set_bytes(&wd);
        }
        st.set_f32(&vec![0.01; n_expert]);
        // Routed to all 512 experts, and to 64 of them (a served batch's ~50-100 per layer);
        // 256 and 1024 tokens: short prompts and prefill tails (wider token tiles).
        for (n_tok, pool) in [
            (9usize, n_expert),
            (16, n_expert),
            (16, 64),
            (32, n_expert),
            (256, n_expert),
            (1024, n_expert),
        ] {
            let Kernel::Mv { nt, .. } = nvfp4_mm::pick_grouped(
                k as i64,
                m as i64,
                (n_used * n_tok) as i64,
                n_expert as i64,
            ) else {
                continue;
            };
            let mv = |rtw, kw, rpt| Some(Kernel::Mv { nt, rtw, kw, rpt });
            let ids = routing(n_used, n_tok, pool, 11);
            let used = ids.iter().collect::<std::collections::HashSet<_>>().len();
            let bytes = (wq.len() / n_expert * used) as f64;
            let mut line = format!("tokens {n_tok:4} experts {used:3} nt {nt}:");
            for (what, kernel) in [
                ("chain", None),
                ("chain-p1", mv(1, 8, 1)),
                ("chain-p2", mv(1, 8, 2)),
                ("chain-p4", mv(1, 8, 4)),
                ("chain-p8", mv(1, 8, 8)),
                ("fused", None),
                ("down", None),
                ("down-p1", mv(1, 8, 1)),
                ("down-p4", mv(1, 8, 4)),
                ("down-p8", mv(1, 8, 8)),
                ("down-p16", mv(1, 8, 16)),
            ] {
                let ctx = Context::new(16 * L + 8, 16 * L + 8);
                let x = ctx
                    .new_tensor(types::F32, &[k as i64, 1, n_tok as i64])
                    .as_input();
                let a = ctx
                    .new_tensor(types::BF16, &[m as i64, n_used as i64, n_tok as i64])
                    .as_input();
                let id = ctx
                    .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                    .as_input();
                let mut graph = ctx.new_graph(16 * L + 8);
                for i in 0..L {
                    let s = Some(st);
                    let out = match what {
                        _ if what.starts_with("chain") => {
                            let up = nvfp4_mm::mul_mat_id(&ctx, us[i], x, id, s);
                            let gate = nvfp4_mm::mul_mat_id(&ctx, gs[i], x, id, s);
                            ctx.cast(ctx.mul(ctx.silu(gate), up), types::BF16)
                        }
                        "fused" => nvfp4_mm::swiglu_id(&ctx, gs[i], us[i], x, id, s, s),
                        _ => nvfp4_mm::mul_mat_id(&ctx, ds[i], a, id, s),
                    };
                    graph.expand(out.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                let up = what.starts_with("chain") || what == "fused";
                if up {
                    x.set_f32(&activations(k, n_tok, 9));
                } else {
                    a.set_bytes(&vec![0x3cu8; m * n_used * n_tok * 2]);
                }
                id.set_i32(&ids);
                nvfp4_mm::force_kernel(kernel);
                let s = time(&mut gpu, &graph, 20) / L as f64;
                nvfp4_mm::force_kernel(None);
                let b = if up { 2.0 * bytes } else { bytes };
                line += &format!("  {what} {:6.1} us {:3.0} GB/s", s * 1e6, b / s / 1e9);
            }
            println!("{line}");
        }
    }

    /// The squared-ReLU epilogue (`Act::ReluSqr`, BF16 out) of every kernel, dense and grouped,
    /// against the CPU reference: values one BF16 rounding apart where the F32 sums differ.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_relu_sqr_matches_cpu() {
        let mut cpu = Backend::cpu().unwrap();
        cpu.set_threads(8);
        let Some(mut gpu) = gpu() else { return };
        let run_act = |backend: &mut Backend,
                       wq: &[u8],
                       xs: &[f32],
                       ids: Option<&[i32]>,
                       (k, m, ne): (usize, usize, usize),
                       (nx, nu, nt): (usize, usize, usize)| {
            let weights = Context::new(2, 0);
            let w = weights.new_tensor(types::NVFP4, &[k as i64, m as i64, ne as i64]);
            let s = weights.new_tensor(types::F32, &[ne as i64]);
            weights.alloc_tensors(backend.buffer_type()).unwrap();
            w.set_bytes(wq);
            s.set_f32(&(0..ne).map(|e| 0.5 + e as f32 / 4.0).collect::<Vec<_>>());
            let ctx = Context::new(8, 16);
            let x = ctx
                .new_tensor(types::F32, &[k as i64, nx as i64, nt as i64])
                .as_input();
            let id = ctx
                .new_tensor(types::I32, &[nu as i64, nt as i64])
                .as_input();
            let out = match ids {
                Some(_) => {
                    nvfp4_mm::mul_mat_id_act(&ctx, w, x, id, Some(s), nvfp4_mm::Act::ReluSqr)
                }
                None => {
                    let x2 = ctx.reshape_2d(x, k as i64, (nx * nt) as i64);
                    let s1 = ctx.view_1d(s, 1, 0);
                    nvfp4_mm::mul_mat_act(&ctx, w, x2, Some(s1), nvfp4_mm::Act::ReluSqr)
                }
            };
            let out = ctx.cast(ctx.cont(out), types::F32).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(out);
            ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                .unwrap();
            x.set_f32(xs);
            if let Some(ids) = ids {
                id.set_i32(ids);
            }
            backend.compute(&graph).unwrap();
            out.to_f32()
        };
        for (k, m, ne, nu, nt, grouped) in [
            (512, 130, 1, 1, 9, false),
            (2688, 200, 1, 1, 300, false),
            (256, 100, 16, 6, 16, true),
            (192, 130, 32, 4, 70, true),
        ] {
            let wq = weights(k, m * ne, k as u64);
            let nx = 1;
            let xs = activations(k, nx * nt, m as u64);
            let ids = routing(nu, nt, ne, nt as u64);
            let ids = grouped.then_some(&ids[..]);
            let shape = ((k, m, ne), (nx, nu, nt));
            let want: Vec<f64> = run_act(&mut cpu, &wq, &xs, ids, shape.0, shape.1)
                .iter()
                .map(|&v| v as f64)
                .collect();
            assert!(want.iter().all(|&v| v >= 0.0));
            for kernel in kernels(if grouped { 1 } else { nt }) {
                nvfp4_mm::force_kernel(kernel);
                let got = run_act(&mut gpu, &wq, &xs, ids, shape.0, shape.1);
                let off = got
                    .iter()
                    .zip(&want)
                    .filter(|(g, w)| (**g as f64 - **w).abs() > 1e-6 * w.abs().max(1e-3))
                    .count();
                let e = rel_err(&got, &want);
                assert!(
                    e < 1e-2 && off * 50 < got.len(),
                    "{shape:?} {kernel:?}: {e}, {off}"
                );
            }
            nvfp4_mm::force_kernel(None);
            println!("relu_sqr {shape:?} grouped {grouped}: all kernels match");
        }
    }

    /// The kernels read nothing past the weight's last row: a weight that is a view of the first
    /// rows of a larger tensor whose remaining rows are 0xFF (NaN scales and codes) gives the
    /// same, finite result as the weight alone.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_reads_nothing_past_the_last_row() {
        let Some(mut gpu) = gpu() else { return };
        for n in [3usize, 300] {
            let (k, m) = (512usize, 300usize);
            let wq = weights(k, m, 77);
            let xs = activations(k, n, 78);
            let want = run(&mut gpu, &wq, &xs, None, (k, m, n));
            let row = k / 64 * 36;
            let weights = Context::new(1, 0);
            let big = weights.new_tensor(types::NVFP4, &[k as i64, m as i64 + 64]);
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            let mut bytes = wq.clone();
            bytes.resize(wq.len() + 64 * row, 0xff);
            big.set_bytes(&bytes);
            let ctx = Context::new(8, 16);
            let w = ctx.view_2d(big, [k as i64, m as i64], row, 0);
            let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
            let out = nvfp4_mm::mul_mat(&ctx, w, x, None).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(out);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_f32(&xs);
            gpu.compute(&graph).unwrap();
            let got = out.to_f32();
            assert!(got.iter().all(|v| v.is_finite()));
            assert_eq!(got, want);
        }
    }

    /// Times `graph` (already computed once): seconds per run.
    fn time(gpu: &mut Backend, graph: &legion::ggml::Graph<'_>, reps: usize) -> f64 {
        gpu.compute(graph).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..reps {
            gpu.compute(graph).unwrap();
        }
        t.elapsed().as_secs_f64() / reps as f64
    }

    /// Legion's kernels against ggml's MMVQ / MMQ (Q8_1 activations, BF16 accumulation hint,
    /// then the scale multiply as `Linear` does it) on Qwen3.6-27B's MLP and Nemotron 3.5's
    /// shapes. `LEGION_NVFP4_TIMING_N` (comma list) picks the batch sizes.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_timings() {
        const L: usize = 8;
        let ns: Vec<usize> = std::env::var("LEGION_NVFP4_TIMING_N")
            .unwrap_or_else(|_| "1,2,4,8,16,32,64,128,256,512,2048".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let Some(mut gpu) = gpu() else { return };
        for (k, m) in [(5120, 17408), (17408, 5120), (2688, 3712), (3712, 2688)] {
            let wq = weights(k, m, 5);
            let weights = Context::new(L + 1, 0);
            let ws: Vec<_> = (0..L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64]))
                .collect();
            let st = weights.new_tensor(types::F32, &[1]);
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            st.set_f32(&[0.01]);
            let bytes = (wq.len() * L) as f64;
            for &n in &ns {
                let mut line = format!("k {k:5} m {m:5} n {n:4}:");
                for legion in [true, false] {
                    let ctx = Context::new(4 * L + 4, 4 * L + 4);
                    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                    let mut graph = ctx.new_graph(4 * L + 4);
                    for w in &ws {
                        let out = if legion {
                            nvfp4_mm::mul_mat(&ctx, *w, x, Some(st))
                        } else {
                            let r = ctx.mul_mat(*w, x);
                            ctx.prec_set_src(r, ffi::ggml_prec_GGML_PREC_Q8, 1);
                            ctx.prec_set_acc(r, ffi::ggml_prec_GGML_PREC_BF16);
                            ctx.mul(r, st)
                        };
                        graph.expand(out.as_output());
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    x.set_f32(&activations(k, n, 9));
                    let s = time(&mut gpu, &graph, 5) / L as f64;
                    line += &format!(
                        "  {} {:8.1} us {:5.0} GB/s {:5.1} TFLOPS",
                        if legion { "legion" } else { "ggml" },
                        s * 1e6,
                        bytes / L as f64 / s / 1e9,
                        2.0 * (k * m * n) as f64 / s / 1e12
                    );
                }
                println!("{line}");
            }
        }
    }

    /// The mv kernel's variants at decode batch sizes on Qwen3.6-27B's gate/up and down shapes:
    /// GB/s of the weight stream (for tuning `pick_dense`).
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_mv_variants() {
        const L: usize = 8;
        let Some(mut gpu) = gpu() else { return };
        for (k, m) in [
            (5120, 17408),
            (17408, 5120),
            (2688, 3712),
            (3712, 2688),
            (5120, 6144),
        ] {
            let wq = weights(k, m, 5);
            let weights = Context::new(L, 0);
            let ws: Vec<_> = (0..L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64]))
                .collect();
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            for n in [1usize, 8, 16, 32] {
                let ctx = Context::new(4 * L + 4, 4 * L + 4);
                let x = ctx
                    .new_tensor(types::BF16, &[k as i64, n as i64])
                    .as_input();
                let mut graph = ctx.new_graph(4 * L + 4);
                for w in &ws {
                    graph.expand(nvfp4_mm::mul_mat(&ctx, *w, x, None).as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                x.set_bytes(&vec![0u8; k * n * 2]);
                let mut line = format!("k {k:5} m {m:5} n {n:2}:");
                for kernel in kernels(n) {
                    let Some(Kernel::Mv {
                        nt,
                        rtw,
                        kw,
                        rpt: 1,
                    }) = kernel
                    else {
                        continue;
                    };
                    if 8 * nt as usize >= 2 * n.max(8) {
                        continue;
                    }
                    nvfp4_mm::force_kernel(kernel);
                    let s = time(&mut gpu, &graph, 10) / L as f64;
                    line += &format!(" r{rtw}k{kw} {:4.0}", wq.len() as f64 / s / 1e9);
                }
                nvfp4_mm::force_kernel(None);
                println!("{line}");
            }
        }
    }

    /// Grouped expert matmuls against ggml's `mul_mat_id` (Q8_1 activations, then the
    /// per-expert scale as `Linear` applies it) on Nemotron 3.5's expert shapes: 128 experts,
    /// 6 used per token.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_mul_mat_id_timings() {
        const L: usize = 4;
        let (n_expert, n_used) = (128usize, 6usize);
        let Some(mut gpu) = gpu() else { return };
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
            for n_tok in [1usize, 4, 16, 64, 512, 4096] {
                let ids = routing(n_used, n_tok, n_expert, 11);
                let used: std::collections::HashSet<_> = ids.iter().collect();
                let bytes = (wq.len() / n_expert * used.len()) as f64;
                let mut line =
                    format!("k {k:5} m {m:5} tokens {n_tok:4} experts {:3}:", used.len());
                for legion in [true, false] {
                    let ctx = Context::new(8 * L + 8, 8 * L + 8);
                    let x = ctx
                        .new_tensor(types::F32, &[k as i64, n_x as i64, n_tok as i64])
                        .as_input();
                    let id = ctx
                        .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                        .as_input();
                    let mut graph = ctx.new_graph(8 * L + 8);
                    for w in &ws {
                        let out = if legion {
                            nvfp4_mm::mul_mat_id(&ctx, *w, x, id, Some(st))
                        } else {
                            let r = ctx.mul_mat_id(*w, x, id);
                            ctx.prec_set_src(r, ffi::ggml_prec_GGML_PREC_Q8, 1);
                            ctx.prec_set_acc(r, ffi::ggml_prec_GGML_PREC_BF16);
                            let s = ctx.reshape_3d(st, 1, n_expert as i64, 1);
                            let s = ctx.repeat_4d(s, [1, n_expert as i64, n_tok as i64, 1]);
                            ctx.mul(r, ctx.get_rows(s, id))
                        };
                        graph.expand(out.as_output());
                    }
                    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                        .unwrap();
                    x.set_f32(&activations(k, n_x * n_tok, 9));
                    id.set_i32(&ids);
                    let s = time(&mut gpu, &graph, 5) / L as f64;
                    line += &format!(
                        "  {} {:8.1} us {:5.0} GB/s",
                        if legion { "legion" } else { "ggml" },
                        s * 1e6,
                        bytes / s / 1e9,
                    );
                }
                println!("{line}");
            }
        }
    }
    /// The mv kernel's variants on grouped expert matmuls at decode batch sizes (Nemotron 3.5's
    /// shapes, 128 experts, 6 used): GB/s of the experts' weights (for tuning `pick_grouped`).
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_mul_mat_id_variants() {
        const L: usize = 4;
        let (n_expert, n_used) = (128usize, 6usize);
        let Some(mut gpu) = gpu() else { return };
        for (k, m, n_x) in [(2688, 1856, 1), (1856, 2688, 6)] {
            let wq = weights(k, m * n_expert, 5);
            let weights = Context::new(L, 0);
            let ws: Vec<_> = (0..L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64, n_expert as i64]))
                .collect();
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            for n_tok in [16usize, 32] {
                let ids = routing(n_used, n_tok, n_expert, 11);
                let used: std::collections::HashSet<_> = ids.iter().collect();
                let bytes = (wq.len() / n_expert * used.len()) as f64;
                let ctx = Context::new(8 * L + 8, 8 * L + 8);
                let x = ctx
                    .new_tensor(types::BF16, &[k as i64, n_x as i64, n_tok as i64])
                    .as_input();
                let id = ctx
                    .new_tensor(types::I32, &[n_used as i64, n_tok as i64])
                    .as_input();
                let mut graph = ctx.new_graph(8 * L + 8);
                for w in &ws {
                    graph.expand(nvfp4_mm::mul_mat_id(&ctx, *w, x, id, None).as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                x.set_bytes(&vec![0u8; k * n_x * n_tok * 2]);
                id.set_i32(&ids);
                let mut line = format!("k {k:5} m {m:5} tokens {n_tok:2}:");
                let s = time(&mut gpu, &graph, 5) / L as f64;
                line += &format!(" default {:3.0}", bytes / s / 1e9);
                for (rtw, kw, rpt) in [
                    (1, 4, 1),
                    (1, 8, 1),
                    (2, 4, 1),
                    (2, 8, 1),
                    (1, 8, 2),
                    (1, 8, 4),
                    (1, 8, 8),
                ] {
                    nvfp4_mm::force_kernel(Some(Kernel::Mv {
                        nt: 1,
                        rtw,
                        kw,
                        rpt,
                    }));
                    let s = time(&mut gpu, &graph, 5) / L as f64;
                    line += &format!(" r{rtw}k{kw}p{rpt} {:3.0}", bytes / s / 1e9);
                }
                nvfp4_mm::force_kernel(None);
                println!("{line}");
            }
        }
    }

    /// `--nvfp4-wide-mv`: dense matmuls and fused SwiGLUs of 33-64 tokens on the mv kernel give
    /// every token the bits of the mv kernel at up to 32 tokens (here: the same tokens in calls
    /// of 32 and of the rest), partial row and token tiles included.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_wide_mv_keeps_per_token_bits() {
        let Some(mut gpu) = gpu() else { return };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (k, m) in [(5120, 300), (17408, 130), (256, 70)] {
            let wq = weights(k, m, (k + m) as u64);
            let wu = weights(k, m, 7 * m as u64);
            for n in [33usize, 40, 41, 57, 64] {
                let xs = activations(k, n, n as u64);
                let (head, tail) = xs.split_at(32 * k);
                nvfp4_mm::set_wide_mv(true);
                let wide = run(&mut gpu, &wq, &xs, Some(0.5), (k, m, n));
                let wide_glu = run_swiglu(&mut gpu, (&wq, &wu), &xs, (Some(0.5), None), (k, m, n));
                nvfp4_mm::set_wide_mv(false);
                let mut want = run(&mut gpu, &wq, head, Some(0.5), (k, m, 32));
                want.extend(run(&mut gpu, &wq, tail, Some(0.5), (k, m, n - 32)));
                assert!(bits(&wide) == bits(&want), "k {k} m {m} n {n}: matmul bits");
                let mut want =
                    run_swiglu(&mut gpu, (&wq, &wu), head, (Some(0.5), None), (k, m, 32));
                want.extend(run_swiglu(
                    &mut gpu,
                    (&wq, &wu),
                    tail,
                    (Some(0.5), None),
                    (k, m, n - 32),
                ));
                assert!(
                    bits(&wide_glu) == bits(&want),
                    "k {k} m {m} n {n}: swiglu bits"
                );
                // And the default (GEMM) is still what runs without the switch.
                let gemm = run(&mut gpu, &wq, &xs, Some(0.5), (k, m, n));
                nvfp4_mm::force_kernel(Some(Kernel::Gemm { mi: 1, nt: 8 }));
                let forced = run(&mut gpu, &wq, &xs, Some(0.5), (k, m, n));
                nvfp4_mm::force_kernel(None);
                assert!(
                    bits(&gemm) == bits(&forced),
                    "k {k} m {m} n {n}: default moved"
                );
            }
            println!("k {k} m {m}: bitwise");
        }
    }

    /// `--nvfp4-wide-mv` against the default GEMM at verify-sized batches on Qwen3.8-27B's MLP
    /// shapes (gate/up fused SwiGLU 5120 -> 2 x 17408, down 17408 -> 5120): us per op and GB/s
    /// of the weight stream, 8 weight copies per graph. `LEGION_NVFP4_TIMING_N` (comma list)
    /// picks the batch sizes.
    #[test]
    #[ignore = "needs a GPU"]
    fn cuda_wide_mv_timings() {
        const L: usize = 8;
        let ns: Vec<usize> = std::env::var("LEGION_NVFP4_TIMING_N")
            .unwrap_or_else(|_| "16,24,32,33,40,48,56,64".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let Some(mut gpu) = gpu() else { return };
        for (k, m, glu) in [
            (5120, 17408, true),
            (17408, 5120, false),
            (5120, 17408, false),
        ] {
            let wq = weights(k, m, 5);
            let weights = Context::new(2 * L + 1, 0);
            let ws: Vec<_> = (0..2 * L)
                .map(|_| weights.new_tensor(types::NVFP4, &[k as i64, m as i64]))
                .collect();
            weights.alloc_tensors(gpu.buffer_type()).unwrap();
            for w in &ws {
                w.set_bytes(&wq);
            }
            let bytes = wq.len() as f64 * if glu { 2.0 } else { 1.0 };
            for &n in &ns {
                let ctx = Context::new(4 * L + 4, 4 * L + 4);
                let x = ctx
                    .new_tensor(types::BF16, &[k as i64, n as i64])
                    .as_input();
                let mut graph = ctx.new_graph(4 * L + 4);
                for i in 0..L {
                    let out = if glu {
                        nvfp4_mm::swiglu(&ctx, ws[2 * i], ws[2 * i + 1], x, None, None)
                    } else {
                        nvfp4_mm::mul_mat(&ctx, ws[i], x, None)
                    };
                    graph.expand(out.as_output());
                }
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .unwrap();
                x.set_bytes(&vec![0u8; k * n * 2]);
                let mut line = format!(
                    "{} k {k:5} m {m:5} n {n:2}:",
                    if glu { "swiglu" } else { "mul_mat" }
                );
                for wide in [false, true] {
                    nvfp4_mm::set_wide_mv(wide);
                    let s = time(&mut gpu, &graph, 10) / L as f64;
                    let name = if wide { "wide" } else { "default" };
                    line += &format!("  {name} {:6.1} us {:4.0} GB/s", s * 1e6, bytes / s / 1e9);
                }
                nvfp4_mm::set_wide_mv(false);
                // The matmul's mv tiles of 64 tokens (the fused SwiGLU's are 16 rows of each).
                for rtw in [1, 2] {
                    if glu || n <= 32 {
                        break;
                    }
                    nvfp4_mm::force_kernel(Some(Kernel::Mv {
                        nt: 8,
                        rtw,
                        kw: 8,
                        rpt: 1,
                    }));
                    let s = time(&mut gpu, &graph, 10) / L as f64;
                    line += &format!("  r{rtw} {:6.1} us {:4.0} GB/s", s * 1e6, bytes / s / 1e9);
                }
                nvfp4_mm::force_kernel(None);
                println!("{line}");
            }
        }
    }
}
