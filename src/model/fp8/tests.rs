use super::*;

#[test]
fn e4m3_values() {
    assert_eq!(e4m3_to_f32(0x00), 0.0);
    assert_eq!(e4m3_to_f32(0x38), 1.0);
    assert_eq!(e4m3_to_f32(0xb8), -1.0);
    assert_eq!(e4m3_to_f32(0x7e), 448.0);
    assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    assert_eq!(e4m3_to_f32(0x08), 2f32.powi(-6));
    assert_eq!(e4m3_to_f32(0x3c), 1.5);
    assert!(e4m3_to_f32(0x7f).is_nan() && e4m3_to_f32(0xff).is_nan());
    assert_eq!(e8m0_to_f32(127), 1.0);
    assert_eq!(e8m0_to_f32(128), 2.0);
    assert_eq!(e8m0_to_f32(120), 2f32.powi(-7));
    assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
    assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
    assert!(e8m0_to_f32(255).is_nan());
    // Every finite code round-trips.
    for b in (0..=255u8).filter(|b| b & 0x7f != 0x7f) {
        let back = f32_to_e4m3(e4m3_to_f32(b));
        assert!(
            back == b || (b == 0x80 && back == 0x80),
            "{b:#04x} -> {back:#04x}"
        );
    }
    assert_eq!(f32_to_e4m3(1000.0), 0x7e);
    assert_eq!(f32_to_e4m3(-1000.0), 0xfe);
    assert_eq!(f32_to_e4m3(1.06), 0x38);
    // Ties to even: 1.0625 is between 1.0 (0x38) and 1.125 (0x39).
    assert_eq!(f32_to_e4m3(1.0625), 0x38);
    assert_eq!(f32_to_e4m3(1.1875), 0x3a);
    // Subnormals and the carry into the next binade.
    assert_eq!(f32_to_e4m3(3.0 * 2f32.powi(-10)), 0x02);
    assert_eq!(f32_to_e4m3(0.0156), 0x08);
    assert_eq!(f32_to_e4m3(15.9), 0x58);
}

/// Per-token quantization: the largest magnitude maps to 448 (0x7e / 0xfe), the scale is
/// `amax / 448`, every code is within E4M3's rounding of `x / s`, and an all-zero token gets
/// scale 0 and zero codes.
#[test]
fn quantize_token_scales_by_amax() {
    let x = [0.5f32, -3.5, 1.0, 0.0, 2.25, -0.001];
    let (q, s) = quantize_token(&x);
    assert_eq!(s, 3.5 / 448.0);
    assert_eq!(q[1], 0xfe, "-amax is -448");
    for (&v, &b) in x.iter().zip(&q) {
        let back = e4m3_to_f32(b) * s;
        assert!(
            (back - v).abs() <= v.abs() / 16.0 + s / 512.0,
            "{v} -> {back}"
        );
    }
    assert_eq!(q[3], 0);
    let (q, s) = quantize_token(&[0.0; 8]);
    assert_eq!(s, 0.0);
    assert!(q.iter().all(|&b| b == 0));
    // A token of one tiny value: still exactly +-448 at the top.
    let (q, s) = quantize_token(&[1e-30, -2e-30]);
    assert!(s > 0.0);
    assert_eq!(q[1], 0xfe);
}

/// The packed layout: codes first, then one F32 scale per token, 4-byte aligned for `k` a
/// multiple of 4.
#[test]
fn dynamic_layout_packs_codes_then_scales() {
    assert_eq!(dynamic_layout(64, 3), (192, 204));
    assert_eq!(
        dynamic_layout(5120, 4096),
        (5120 * 4096, 5120 * 4096 + 4 * 4096)
    );
    assert_eq!(dynamic_layout(4, 0), (0, 0));
}

/// The sliced-K GEMM takes 5-64 tokens when asked, leaves 5-9 to the GEMV when that is on too,
/// and needs rows a multiple of 256 long.
#[test]
fn sliced_k_split_by_width() {
    let off = SmallBatch::default();
    let sliced = SmallBatch {
        sliced_k: true,
        ..off
    };
    let both = SmallBatch {
        gemv: true,
        ..sliced
    };
    let gemv = SmallBatch { gemv: true, ..off };
    for n in 1..=80 {
        assert!(!sliced_k_takes(off, n, 5120), "n {n}, off");
        assert!(!sliced_k_takes(gemv, n, 5120), "n {n}, GEMV only");
        assert_eq!(
            sliced_k_takes(sliced, n, 5120),
            (5..=64).contains(&n),
            "n {n}, sliced-K"
        );
        assert_eq!(
            sliced_k_takes(both, n, 5120),
            (10..=64).contains(&n),
            "n {n}, both"
        );
    }
    for (k, takes) in [
        (17408, true),
        (6144, true),
        (2688, false),
        (4160, false),
        (256, true),
    ] {
        assert_eq!(sliced_k_takes(sliced, 16, k), takes, "k {k}");
    }
}

#[test]
fn mm_node_finds_the_matmul_under_a_scale() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::I8, &[64, 32]);
    let x = ctx.new_tensor(types::F32, &[64, 8]);
    let mm = mul_mat(&ctx, w, x, None);
    let found = mm_node(&ctx, mm).map(|t| t.raw());
    assert_eq!(found, Some(mm.raw()));
    let scaled = ctx.scale(mm, 2.0);
    assert_eq!(mm_node(&ctx, scaled).map(|t| t.raw()), Some(mm.raw()));
    assert!(mm_node(&ctx, x).is_none());
    // The side-stream slot is free (zero) on a new node.
    // SAFETY: a node of the live context.
    let params = crate::ggml::external_params(unsafe { &*mm.raw() });
    assert_eq!(params[P_SIDE], 0);
}

/// Random E4M3 codes: any sign and mantissa, exponents 3 to 10 (no NaN).
#[cfg(feature = "cuda")]
fn e4m3_noise(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        for b in s.to_le_bytes() {
            out.push(b & 0x87 | (3 + (b >> 3) % 8) << 3);
        }
    }
    out.truncate(len);
    out
}

/// Values in [-1, 1).
#[cfg(feature = "cuda")]
fn unit_noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

/// The GPU of the CUDA tests.
#[cfg(feature = "cuda")]
fn test_gpu() -> crate::ggml::Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// A W8A8 prompt GEMM graph (`--fp8-act a8-dynamic`, a weight scale per row) of `layers`
/// distinct `[k, m]` weights over one `[k, n]` input (one quantization), with a residual added
/// in the epilogue where `res`, allocated on `gpu` and filled; `then` gets the graph and the
/// outputs.
#[cfg(feature = "cuda")]
fn gemm8_case(
    gpu: &mut crate::ggml::Backend,
    (k, m, n, res): (i64, i64, i64, bool),
    layers: usize,
    then: &mut dyn FnMut(&mut crate::ggml::Backend, &crate::ggml::Graph<'_>, &[Tensor<'_>]),
) {
    use crate::ggml::ComputeAllocator;
    let wctx = Context::new(4 * layers, 0);
    let weights: Vec<_> = (0..layers)
        .map(|_| {
            (
                wctx.new_tensor(types::I8, &[k, m]),
                wctx.new_tensor(types::F32, &[m]),
            )
        })
        .collect();
    wctx.alloc_tensors(gpu.buffer_type()).unwrap();
    for (l, (w, sc)) in weights.iter().enumerate() {
        w.set_bytes(&e4m3_noise((k * m) as usize, 5 + l as u64 + 7 * k as u64));
        let scales: Vec<f32> = unit_noise(m as usize, 0x55 + l as u64)
            .iter()
            .map(|v| 0.002 + 0.001 * v.abs())
            .collect();
        sc.set_f32(&scales);
    }
    let ctx = Context::new(16 + 4 * layers, 16);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input().as_output();
    let r = res.then(|| ctx.new_tensor(types::F32, &[m, n]).as_input().as_output());
    let mut graph = ctx.new_graph(16 + 4 * layers);
    let outs: Vec<Tensor<'_>> = weights
        .iter()
        .map(|&(w, sc)| {
            let y = match r {
                Some(r) => mul_mat_dynamic_add(&ctx, w, x, Some(sc), r),
                None => mul_mat_dynamic(&ctx, w, x, Some(sc)),
            }
            .as_output();
            graph.expand(y);
            y
        })
        .collect();
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    x.set_f32(&unit_noise((k * n) as usize, 0x1234 + n as u64));
    if let Some(r) = r {
        r.set_f32(&unit_noise((m * n) as usize, 0x4321 + n as u64));
    }
    then(gpu, &graph, &outs);
}

/// The W8A8 prompt GEMM's TMA store epilogue (#1183, variant bit 0x800, the default) at
/// Qwen3.8-Flash-Next's dense shapes (GDN z and out, attention q+gate+k+v and out, the shared
/// expert's gate+up and down; GDN qkv's GEMM alone) at 12,288 and 8,067 tokens and ragged token
/// counts (partial token tiles, and the 128-row tile where the wide one makes too few waves),
/// against the plain epilogue's bits (variant 1): also the pick without the TMA stores (0), the
/// staged epilogue (0x200, which takes precedence), the staggered first wave (0x2000) and both
/// (0x2800), and a residual shape (no TMA store there).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU with E4M3 tensor cores"]
fn cuda_gemm8_tma_store_is_the_plain_bits() {
    let mut gpu = test_gpu();
    if !fp8_mma() {
        eprintln!("no E4M3 tensor cores: skipped");
        return;
    }
    let _lock = VARIANT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // (k, m, n, residual)
    let shapes: [(i64, i64, i64, bool); 15] = [
        (2560, 6144, 12288, false),
        (6144, 2560, 12288, false),
        (2560, 13312, 12288, false),
        (2560, 1280, 12288, false),
        (640, 2560, 12288, false),
        (2560, 10240, 12288, false),
        (2560, 6144, 8067, false),
        (6144, 2560, 8067, false),
        (2560, 13312, 8067, false),
        (640, 2560, 8067, false),
        (2560, 6144, 12289, false),
        (2560, 10240, 1153, false),
        (2560, 6144, 1153, false),
        (640, 2560, 4993, false),
        (6144, 2560, 8067, true),
    ];
    for shape in shapes {
        gemm8_case(&mut gpu, shape, 1, &mut |gpu, graph, outs| {
            let y = outs[0];
            force_gemm_variant(1);
            let ran = gpu.compute(graph);
            unforce_gemm_variant();
            ran.unwrap();
            let want = y.bytes();
            for v in [
                None,
                Some(0),
                Some(0x800),
                Some(0x200),
                Some(0x2000),
                Some(0x2800),
            ] {
                match v {
                    Some(v) => force_gemm_variant(v),
                    None => unforce_gemm_variant(),
                }
                let ran = gpu.compute(graph);
                unforce_gemm_variant();
                ran.unwrap();
                let got = y.bytes();
                let bad = got
                    .chunks_exact(4)
                    .zip(want.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(bad, 0, "{shape:?}: variant {v:x?}");
            }
        });
    }
}

/// `cargo test --release --features cuda -- --ignored --nocapture cuda_gemm8_tma_store_microbench`
/// (#1183): the W8A8 prompt GEMMs of Qwen3.8-Flash-Next at 12,288 and 8,067 tokens, two
/// distinct weights a graph (past the L2), the fragments' stores (variant 0) against the TMA
/// stores (0x800), the staged epilogue (0x200), the staggered first wave (0x2000) and both
/// (0x2800): 5 rounds taking turns, the medians per GEMM (with half of one quantization of
/// the input, the same for all).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "microbenchmark, needs a GPU with E4M3 tensor cores"]
fn cuda_gemm8_tma_store_microbench() {
    let mut gpu = test_gpu();
    if !fp8_mma() {
        eprintln!("no E4M3 tensor cores: skipped");
        return;
    }
    let _lock = VARIANT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const VARIANTS: [i32; 5] = [0, 0x800, 0x200, 0x2000, 0x2800];
    const ROUNDS: usize = 5;
    const REPS: usize = 3;
    const LAYERS: usize = 2;
    for n in [12288i64, 8067] {
        for (k, m) in [
            (2560, 6144),
            (6144, 2560),
            (2560, 13312),
            (2560, 1280),
            (640, 2560),
        ] {
            gemm8_case(&mut gpu, (k, m, n, false), LAYERS, &mut |gpu, graph, _| {
                let mut ms: Vec<Vec<f64>> = vec![Vec::new(); VARIANTS.len()];
                // A warm-up run each.
                for v in VARIANTS {
                    force_gemm_variant(v);
                    gpu.compute(graph).unwrap();
                    unforce_gemm_variant();
                }
                for _ in 0..ROUNDS {
                    for (i, &v) in VARIANTS.iter().enumerate() {
                        force_gemm_variant(v);
                        for _ in 0..REPS {
                            let t = std::time::Instant::now();
                            gpu.compute(graph).unwrap();
                            ms[i].push(t.elapsed().as_secs_f64() * 1e3 / LAYERS as f64);
                        }
                        unforce_gemm_variant();
                    }
                }
                let med: Vec<f64> = ms
                    .iter_mut()
                    .map(|xs| {
                        xs.sort_by(f64::total_cmp);
                        xs[xs.len() / 2]
                    })
                    .collect();
                let tf = |t: f64| 2.0 * (k * m * n) as f64 / (t * 1e-3) / 1e12;
                let row: Vec<String> = VARIANTS
                    .iter()
                    .zip(&med)
                    .map(|(v, t)| format!("{v:#x} {t:.3} ms {:.0} TF {:.3}x", tf(*t), med[0] / t))
                    .collect();
                eprintln!("k {k} m {m} n {n}: {}", row.join(" | "));
            });
        }
    }
}
