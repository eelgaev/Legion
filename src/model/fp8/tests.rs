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

/// The process-wide [`set_glu_wide_store`] switch: held by the tests that set it.
static WIDE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// [`mark_wide_store`] marks a dynamic W8A8 node without a residual, through a view of its rows
/// (a member of a joined run) or the node itself, and [`node_variant`] then gives
/// [`VARIANT_WIDE`] while [`set_glu_wide_store`] is on; a node with a residual, the W8A16
/// matmul and a ggml matmul stay unmarked (#1183).
#[test]
fn wide_store_marks_plain_w8a8_nodes_only() {
    let _lock = WIDE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = Context::new(32, 0);
    let (k, m, n) = (128, 256, 100);
    let w = ctx.new_tensor(types::I8, &[k, m]);
    let x = ctx.new_tensor(types::F32, &[k, n]);
    let params = |t: Tensor<'_>| {
        // SAFETY: a node of the live context.
        crate::ggml::external_params(unsafe { &*t.raw() })
    };
    // A joined gate / up: the gate's rows a view of the node.
    let mm = mul_mat_dynamic(&ctx, w, x, None);
    assert_eq!(params(mm)[P_WIDE_STORE], 0, "a new node is unmarked");
    assert_eq!(node_variant(&params(mm)), 0);
    let gate = ctx.view_2d(mm, [m / 2, n], mm.nb()[1], 0);
    assert!(mark_wide_store(gate));
    assert_eq!(params(mm)[P_WIDE_STORE], 1);
    // The other parameters stay: dynamic, no residual, run by its node.
    assert_eq!(params(mm)[..3], [1, 0, 0]);
    set_glu_wide_store(true);
    assert_eq!(node_variant(&params(mm)), VARIANT_WIDE);
    set_glu_wide_store(false);
    assert_eq!(node_variant(&params(mm)), 0);
    set_glu_wide_store(true);
    // The node itself.
    let mm2 = mul_mat_dynamic(&ctx, w, x, None);
    assert!(mark_wide_store(mm2));
    assert_eq!(node_variant(&params(mm2)), VARIANT_WIDE);
    // A residual in the epilogue: not taken.
    let res = ctx.new_tensor(types::F32, &[m, n]);
    let with_res = mul_mat_dynamic_add(&ctx, w, x, None, res);
    assert!(!mark_wide_store(with_res));
    assert_eq!(params(with_res)[P_WIDE_STORE], 0);
    assert_eq!(node_variant(&params(with_res)), 0);
    // W8A16 and a ggml matmul (BF16 weights): not W8A8 nodes.
    let a16 = mul_mat(&ctx, w, x, None);
    assert!(!mark_wide_store(a16));
    assert_eq!(params(a16)[P_WIDE_STORE], 0);
    let wf = ctx.new_tensor(types::BF16, &[k, m]);
    assert!(!mark_wide_store(ctx.mul_mat(wf, x)));
}

/// A xorshift generator for the GPU tests' operands.
#[cfg(feature = "cuda")]
struct Rng(u64);

#[cfg(feature = "cuda")]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }

    /// `len` E4M3 codes of any sign and mantissa, exponents 3 to 10 (no NaN).
    fn e4m3(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len + 8);
        while out.len() < len {
            for b in self.next().to_le_bytes() {
                out.push((b & 0x87) | ((3 + (b >> 3) % 8) << 3));
            }
        }
        out.truncate(len);
        out
    }
}

#[cfg(feature = "cuda")]
fn wide_gpu() -> crate::ggml::Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// A W8A8 shared-expert gate / up graph for the GPU tests: `layers` weights of `m` rows by `k`
/// (one F32 scale per row, or one per tensor), each a dynamic W8A8 GEMM of the same `n` tokens,
/// marked as [`Model::shexp`](crate::model::core::Model) marks its gate (through the view of its
/// first half), followed by the SwiGLU's FP8 codes ([`crate::model::fp8_act::gate`]) where
/// `codes`. The GEMM outputs are outputs, so every row is read back.
#[cfg(feature = "cuda")]
struct WideGraph<'a> {
    graph: crate::ggml::Graph<'a>,
    gemms: Vec<Tensor<'a>>,
    codes: Vec<Tensor<'a>>,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn wide_graph<'a>(
    gpu: &crate::ggml::Backend,
    weights: &'a Context,
    ctx: &'a Context,
    (k, m, n): (usize, usize, usize),
    layers: usize,
    row_scales: bool,
    codes: bool,
    seed: u64,
) -> WideGraph<'a> {
    use crate::ggml::ComputeAllocator;
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let scale_len = if row_scales { m } else { 1 };
    let ws: Vec<_> = (0..layers)
        .map(|_| {
            (
                weights.new_tensor(types::I8, &[k as i64, m as i64]),
                weights.new_tensor(types::F32, &[scale_len as i64]),
            )
        })
        .collect();
    weights.alloc_tensors(gpu.buffer_type()).unwrap();
    for (w, s) in &ws {
        w.set_bytes(&rng.e4m3(k * m));
        let v: Vec<f32> = (0..scale_len)
            .map(|_| 0.002 + 0.001 * rng.unit().abs())
            .collect();
        s.set_f32(&v);
    }
    // The input is an output too, so the allocator places no node over it once read.
    let x = ctx
        .new_tensor(types::F32, &[k as i64, n as i64])
        .as_input()
        .as_output();
    let mut graph = ctx.new_graph(16 * layers + 16);
    let mut gemms = Vec::new();
    let mut outs = Vec::new();
    for &(w, s) in &ws {
        let mm = mul_mat_dynamic(ctx, w, x, Some(s));
        let half = (m / 2) as i64;
        let nb1 = mm.nb()[1];
        let gate = ctx.view_2d(mm, [half, n as i64], nb1, 0);
        let up = ctx.view_2d(mm, [half, n as i64], nb1, half as usize * 4);
        assert!(mark_wide_store(gate));
        let mm = mm.as_output();
        graph.expand(mm);
        gemms.push(mm);
        if codes {
            assert!(crate::model::fp8_act::gate_fits(gate, up));
            let q = crate::model::fp8_act::gate(ctx, gate, up, true).as_output();
            graph.expand(q);
            outs.push(q);
        }
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = (0..k * n).map(|_| rng.unit()).collect();
    x.set_f32(&xs);
    WideGraph {
        graph,
        gemms,
        codes: outs,
    }
}

/// The shared expert gate / up GEMM's staged 16-byte stores (#1183, [`set_glu_wide_store`])
/// keep every output bit against the fragments' own stores, at Qwen3.8-Flash-Next's shapes (k
/// 2560, 640 gate + 640 up rows) at 12,288 tokens (the 256-row TMA tile on a GB10) and 8,067
/// (too few waves: the 128-row tile), and at shapes that take the other tiles and tails: 200
/// tokens (128-row tile, one partial token tile), k 2496 (the 256-row tile of four 64-byte
/// stages), 1,000 rows (a partial row tile, rows not a multiple of 64), with per-row and
/// per-tensor weight scales. The outputs are filled with a pattern before each run, so a value
/// the staged epilogue leaves out shows; the SwiGLU's codes read from them match too.
/// `cargo test --release --features cuda --lib -- --ignored
/// cuda_fp8_glu_wide_store_keeps_bits --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU with E4M3 tensor cores"]
fn cuda_fp8_glu_wide_store_keeps_bits() {
    let _lock = WIDE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = wide_gpu();
    assert!(fp8_mma(), "needs compute capability 8.9+");
    let cases: [(&str, (usize, usize, usize), bool); 7] = [
        ("flash-next shexp 12288", (2560, 1280, 12288), true),
        ("flash-next shexp 8067", (2560, 1280, 8067), true),
        ("shexp 12288, one scale", (2560, 1280, 12288), false),
        ("flash-next shexp 200", (2560, 1280, 200), true),
        ("k 2496 (64-byte stages) 12288", (2496, 1280, 12288), true),
        ("1000 rows 12288", (2560, 1000, 12288), true),
        ("1000 rows 333", (2560, 1000, 333), false),
    ];
    for (name, shape, row_scales) in cases {
        let weights = Context::new(4, 0);
        let ctx = Context::new(32, 32);
        let codes = shape.1 % 2 == 0;
        let g = wide_graph(&gpu, &weights, &ctx, shape, 1, row_scales, codes, 11);
        let mut run = |wide: bool| -> (Vec<u8>, Vec<u8>) {
            set_glu_wide_store(wide);
            for t in g.gemms.iter().chain(&g.codes) {
                t.set_bytes(&vec![0xa5; t.nbytes()]);
            }
            gpu.compute(&g.graph).unwrap();
            let bytes =
                |ts: &[Tensor<'_>]| -> Vec<u8> { ts.iter().flat_map(|t| t.bytes()).collect() };
            (bytes(&g.gemms), bytes(&g.codes))
        };
        let want = run(false);
        let got = run(true);
        let again = run(false);
        set_glu_wide_store(true);
        assert!(want == again, "{name}: the fragments' stores twice differ");
        let differ = got
            .0
            .chunks(4)
            .zip(want.0.chunks(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differ,
            0,
            "{name}: {differ} of {} outputs differ",
            want.0.len() / 4
        );
        assert!(got.1 == want.1, "{name}: the SwiGLU codes differ");
        eprintln!("{name}: the same bits ({} outputs)", want.0.len() / 4);
    }
}

/// The shared expert gate / up GEMM (Qwen3.8-Flash-Next: k 2560, 640 gate + 640 up rows, per-row
/// weight scales) at 12,288- and 8,067-token prompt steps with the fragments' own 4-byte stores
/// (old) and the staged 16-byte ones (wide), alone and followed by the SwiGLU's FP8 codes that
/// read its rows (as in a `--dense-fp8-prompt` layer): microseconds a layer (median and best
/// of 7 rounds; the variants take turns, forward in even rounds and reversed in odd ones, so
/// neither always runs first), the speedup against old, and whether the bits are old's. Each
/// graph quantizes the input once (timed alone and taken out) and runs 4 layers' weights. `*`
/// marks the default.
/// `cargo test --release --features cuda --lib -- --ignored cuda_fp8_glu_wide_store_bench
/// --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU with E4M3 tensor cores; a benchmark"]
fn cuda_fp8_glu_wide_store_bench() {
    use crate::ggml::ComputeAllocator;
    let _lock = WIDE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut gpu = wide_gpu();
    assert!(fp8_mma(), "needs compute capability 8.9+");
    const LAYERS: usize = 4;
    const ROUNDS: usize = 7;
    let (k, m) = (2560usize, 1280usize);
    let variants: [(&str, bool); 2] = [("old (fragments)", false), ("wide", true)];
    let nv = variants.len();
    for n in [12288usize, 8067] {
        // The quantization alone, as the graphs run it.
        let qctx = Context::new(4, 4);
        let qx = qctx
            .new_tensor(types::F32, &[k as i64, n as i64])
            .as_input()
            .as_output();
        let mut qgraph = qctx.new_graph(4);
        qgraph.expand(quantize_dynamic(&qctx, qx).as_output());
        qctx.alloc_graph(&qgraph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        for chain in [false, true] {
            let weights = Context::new(4 * LAYERS, 0);
            let ctx = Context::new(16 * LAYERS + 16, 16 * LAYERS + 16);
            let g = wide_graph(&gpu, &weights, &ctx, (k, m, n), LAYERS, true, chain, 3);
            let outs: Vec<Tensor<'_>> = g.gemms.iter().chain(&g.codes).copied().collect();
            // A warm-up run of each, its bits against old's.
            let mut want = Vec::new();
            let mut same = vec![true; nv];
            for (vi, &(_, wide)) in variants.iter().enumerate() {
                set_glu_wide_store(wide);
                gpu.compute(&g.graph).unwrap();
                let got: Vec<Vec<u8>> = outs.iter().map(|o| o.bytes()).collect();
                if vi == 0 {
                    want = got;
                } else {
                    same[vi] = got == want;
                }
            }
            gpu.compute(&qgraph).unwrap();
            let mut quant = Vec::with_capacity(ROUNDS);
            let mut times = vec![Vec::with_capacity(ROUNDS); nv];
            for round in 0..ROUNDS {
                let t = std::time::Instant::now();
                gpu.compute(&qgraph).unwrap();
                quant.push(t.elapsed().as_secs_f64());
                for i in 0..nv {
                    let vi = if round % 2 == 0 { i } else { nv - 1 - i };
                    set_glu_wide_store(variants[vi].1);
                    let t = std::time::Instant::now();
                    gpu.compute(&g.graph).unwrap();
                    times[vi].push(t.elapsed().as_secs_f64());
                }
            }
            set_glu_wide_store(true);
            quant.sort_by(f64::total_cmp);
            let q_med = quant[quant.len() / 2];
            let per_layer = |t: f64| ((t - q_med) / LAYERS as f64).max(1e-9);
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            let median: Vec<f64> = times.iter().map(|t| per_layer(t[t.len() / 2])).collect();
            let best: Vec<f64> = times.iter().map(|t| per_layer(t[0])).collect();
            let out_bytes = (m * n * 4) as f64;
            let flops = 2.0 * (k * m * n) as f64;
            let op = ["gate/up", "gate/up + swiglu codes"][chain as usize];
            eprintln!(
                "{op} at {n} tokens (k {k}, m {m}: {:.0} MB F32 out a layer; quantize {:.1} us \
                 a graph, taken out)",
                out_bytes / 1e6,
                q_med * 1e6
            );
            for (vi, &(name, wide)) in variants.iter().enumerate() {
                eprintln!(
                    "  {}{name:<16} median {:>8.1} us/layer ({:>5.3}x old), min {:>8.1} \
                     ({:>5.3}x)  {:>6.1} TFLOPS  {}",
                    [' ', '*'][wide as usize],
                    median[vi] * 1e6,
                    median[0] / median[vi],
                    best[vi] * 1e6,
                    best[0] / best[vi],
                    flops / median[vi] / 1e12,
                    if same[vi] { "same bits" } else { "BITS DIFFER" }
                );
            }
            assert!(same.iter().all(|&s| s), "{op} at {n}: bits differ");
        }
    }
}
