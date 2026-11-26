//! The rebuilt down matmul (#1183) against the BF16 `xn` the norm writes: with one-hot down
//! weights every output is one value of the GEMM's B operand, which must be `xnb`'s value; the
//! norm without `xnb` writes the same streams, scales and inject weights. On the CPU (the
//! reference) and on CUDA (ignored test, the kernel at qwen4exp's shape).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::nvfp4_mm::bf16_bits;
use crate::model::qwen4::hc_norm;

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

/// `x` rounded to BF16 (ties to even), as bytes.
fn bf16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|&v| bf16_bits(v).to_ne_bytes()).collect()
}

/// One-hot down weights `j` of `k` columns: row `r` is 1 at column `(j · M + r) % k`, so the
/// `ceil(k / M)` of them read every column.
fn one_hot(k: i64, j: i64) -> Vec<u8> {
    let mut w = vec![0f32; (k * M) as usize];
    for r in 0..M {
        w[(r * k + (j * M + r) % k) as usize] = 1.0;
    }
    bf16_bytes(&w)
}

/// What [`case`] returns.
struct Case {
    /// The norm writing `xnb` and the one without: streams (widened), scales and inject weights
    /// as bytes.
    with: [Vec<u8>; 3],
    without: [Vec<u8>; 3],
    /// `xnb` widened to F32 (`[k, n]`).
    xnb: Vec<f32>,
    /// [`mul_mat`] of each one-hot weight (`[M, n]` each).
    hot: Vec<Vec<f32>>,
    /// [`mul_mat`] of the random weight and ggml's `mul_mat` of it and `xnb`.
    lo: Vec<f32>,
    lo_ref: Vec<f32>,
}

/// The norm with BF16 streams on `n` tokens of 4 streams of `e` values, writing `xnb` and not,
/// [`mul_mat`] of the second's streams with one-hot and random down weights, ggml's `mul_mat` of
/// the random weight and `xnb`.
fn case(backend: &mut Backend, e: i64, n: i64) -> Case {
    let (hc, n_inj) = (4i64, 4i64);
    let k = e * hc;
    let hot_n = (k + M - 1) / M;
    let ctx = Context::new(64 + 2 * hot_n as usize, 64 + 2 * hot_n as usize);
    let res = ctx.new_tensor(types::F32, &[e, hc, n]).as_input();
    let out = ctx.new_tensor(types::F32, &[e, n]).as_input();
    let inj = ctx.new_tensor(types::F32, &[hc, n]).as_input();
    let g = ctx.new_tensor(types::F32, &[k]).as_input();
    let w = ctx.new_tensor(types::F32, &[k, n_inj]).as_input();
    let down = ctx.new_tensor(types::BF16, &[k, M]).as_input();
    let hot: Vec<Tensor<'_>> = (0..hot_n)
        .map(|_| ctx.new_tensor(types::BF16, &[k, M]).as_input())
        .collect();
    let a = hc_norm::hc_post_norm_rebuilt_xnb(&ctx, res, out, inj, g, w, 1e-6, down, true);
    let b = hc_norm::hc_post_norm_b16_streams(&ctx, res, out, inj, g, w, 1e-6, down);
    assert_eq!((b.res.ty(), b.res.ne()), (types::BF16, [e, hc, n, 1]));
    let f32_of = |t| ctx.cont(ctx.cast(t, types::F32)).as_output();
    let with = [a.res, a.scales, a.inject].map(f32_of);
    let without = [b.res, b.scales, b.inject].map(f32_of);
    let xnb = ctx.cont(ctx.cast(a.xnb, types::F32)).as_output();
    let hot_lo: Vec<Tensor<'_>> = hot
        .iter()
        .map(|&h| mul_mat(&ctx, h, b.res, b.scales, g).as_output())
        .collect();
    let lo = mul_mat(&ctx, down, b.res, b.scales, g).as_output();
    let lo_ref = ctx.mul_mat(down, a.xnb).as_output();
    let mut graph = ctx.new_graph(64 + 2 * hot_n as usize);
    for t in with
        .iter()
        .chain(&without)
        .chain(&hot_lo)
        .chain([&xnb, &lo, &lo_ref])
    {
        graph.expand(*t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    res.set_f32(
        &noise((k * n) as usize, 71)
            .iter()
            .map(|v| 3.0 * v)
            .collect::<Vec<_>>(),
    );
    out.set_f32(&noise((e * n) as usize, 72));
    inj.set_f32(&noise((hc * n) as usize, 73));
    g.set_f32(
        &noise(k as usize, 74)
            .iter()
            .map(|v| 1.0 + 0.3 * v)
            .collect::<Vec<_>>(),
    );
    w.set_f32(&noise((k * n_inj) as usize, 75));
    down.set_bytes(&bf16_bytes(
        &noise((k * M) as usize, 76)
            .iter()
            .map(|v| 0.02 * v)
            .collect::<Vec<_>>(),
    ));
    for (j, h) in hot.iter().enumerate() {
        h.set_bytes(&one_hot(k, j as i64));
    }
    backend.compute(&graph).unwrap();
    Case {
        with: with.map(|t| t.bytes()),
        without: without.map(|t| t.bytes()),
        xnb: xnb.to_f32(),
        hot: hot_lo.iter().map(|t| t.to_f32()).collect(),
        lo: lo.to_f32(),
        lo_ref: lo_ref.to_f32(),
    }
}

/// The checks of [`case`]: the norm's other outputs do not move without `xnb`; every value the
/// rebuilt GEMM multiplies is `xnb`'s (equal values; a one-hot sum turns -0 into +0); the GEMM
/// is ggml's `mul_mat` of `xnb` within F32 rounding of the sum order.
fn check(backend: &mut Backend, e: i64, ns: &[i64]) {
    for &n in ns {
        let c = case(backend, e, n);
        for (i, name) in ["streams", "scales", "inject"].iter().enumerate() {
            assert!(c.with[i] == c.without[i], "{n} tokens: {name} moved");
        }
        let k = e * 4;
        let mut seen = vec![false; k as usize];
        for (j, lo) in c.hot.iter().enumerate() {
            for t in 0..n {
                for r in 0..M {
                    let col = (j as i64 * M + r) % k;
                    let got = lo[(t * M + r) as usize];
                    let want = c.xnb[(t * k + col) as usize];
                    assert!(
                        got == want,
                        "{n} tokens: token {t} k {col}: {got} vs xnb {want}"
                    );
                    seen[col as usize] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "{n} tokens: a k value not read");
        let (num, den) =
            c.lo.iter()
                .zip(&c.lo_ref)
                .fold((0f64, 0f64), |(a, b), (&x, &y)| {
                    let d = f64::from(x) - f64::from(y);
                    (a + d * d, b + f64::from(y) * f64::from(y))
                });
        let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        assert!(rel < 1e-5, "{n} tokens: off ggml's mul_mat by {rel}");
        assert!(den > 0.0, "{n} tokens: zero output");
    }
}

/// [`check`] on the CPU (the reference).
#[test]
fn rebuilt_down_reads_xnb_on_the_cpu() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    check(&mut cpu, 256, &[1, 9]);
}

/// [`rebuild`] is the norm's arithmetic: two rounded F32 multiplies, then BF16 nearest even.
#[test]
fn rebuild_rounds_as_the_norm() {
    for (i, v) in noise(4096, 81).iter().enumerate() {
        let b = bf16_bits(4.0 * v);
        let s = 0.5 + v.abs();
        let g = 1.0 - 0.25 * v;
        let x = f32::from_bits(u32::from(b) << 16);
        assert_eq!(rebuild(b, s, g), bf16_bits((x * s) * g), "value {i}");
    }
    // A tie: 1 + 2^-8 sits halfway between two BF16 values; nearest even rounds it down.
    assert_eq!(rebuild(0x3f80, 1.0 + 1.0 / 256.0, 1.0), 0x3f80);
}

/// The token tile: fewest waves times a CTA's cost, the smaller tile on ties.
#[test]
fn token_tile_by_step_size() {
    // 4,096 tokens on 48 SMs: 64, 43 or 32 CTAs: one wave of 96-token tiles.
    assert_eq!(pick_nt(4096, 48), 3);
    assert_eq!(pick_nt(256, 48), 2);
    assert_eq!(pick_nt(6144, 48), 4);
    assert_eq!(pick_nt(8192, 48), 3);
    assert_eq!(pick_nt(1, 0), 2);
}

/// The shapes the kernel takes: qwen4exp's BF16 10240 x 320 weight and 4 BF16 streams.
#[test]
fn shapes_the_kernel_takes() {
    let nb = |k: usize| [2, 2 * k, 2 * k * 320, 2 * k * 320];
    assert!(shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(shape_fits(
        types::BF16,
        [1024, 320, 1, 1],
        nb(1024),
        [256, 4, 1, 1]
    ));
    assert!(!shape_fits(
        types::F16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 256, 1, 1],
        nb(10240),
        [2560, 4, 4096, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [2560, 4, 0, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [8000, 320, 1, 1],
        nb(8000),
        [2000, 4, 9, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        nb(10240),
        [5120, 2, 9, 1]
    ));
    assert!(!shape_fits(
        types::BF16,
        [10240, 320, 1, 1],
        [2, 20488, 0, 0],
        [2560, 4, 9, 1]
    ));
}

/// [`check`] on CUDA at qwen4exp's shape (sm_80+): the kernel's B operand is `xnb` bit for bit
/// (whole tiles, a partial last tile, every token tile).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_rebuilt_down_reads_xnb() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() || !supported_on(0) {
        println!("no sm_80+ kernel here: skipped");
        return;
    }
    for nt in [0, 2, 3, 4] {
        force_nt(nt);
        check(&mut gpu, 2560, &[1, 33, 300, 4096]);
    }
    force_nt(0);
}

/// The kernel's options: forced ones masked to [`OPT_ALL`], `None` back to the configured ones;
/// the last launch's tile and options read back as stored.
#[test]
fn kernel_options_force_and_read_back() {
    assert_eq!(DEFAULT_OPTS & !OPT_ALL, 0);
    assert_eq!(OPT_SHARED_REBUILD & OPT_LATE_COPIES, 0);
    let configured = opts();
    force_opts(Some(OPT_LATE_COPIES));
    assert_eq!(opts(), OPT_LATE_COPIES);
    force_opts(Some(0xff));
    assert_eq!(opts(), OPT_ALL);
    force_opts(Some(0));
    assert_eq!(opts(), 0);
    force_opts(None);
    assert_eq!(opts(), configured);
    LAST_LAUNCH.store((4 << 4) | OPT_SHARED_REBUILD, Ordering::Relaxed);
    assert_eq!(last_launch(), Some((4, OPT_SHARED_REBUILD)));
    clear_last_launch();
    assert_eq!(last_launch(), None);
}

/// BF16 bytes of `n` deterministic values in [-scale, scale), made without an F32 copy (the
/// 12,288-token streams are 126M values).
#[cfg(feature = "cuda")]
fn bf16_noise(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(2 * n);
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let v = ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale;
        out.extend_from_slice(&bf16_bits(v).to_ne_bytes());
    }
    out
}

/// Every option form of the kernel (#1183: RS, LI and both) against the old form (no options),
/// bit for bit, at the step sizes the one-Spark NVFP4 docs command's 32K prompt runs (12,288-token
/// steps, the 8,067-token tail; 129 and 1 tokens for partial tiles, below where the model takes
/// the kernel), with every token tile and the one chosen by step size. The output is poisoned
/// (all ones, a NaN) before every run, so a store the kernel skips fails; each run checks that
/// the kernel and options it asked for ran. All tiles also agree with each other (one chain over
/// k per output).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_kernel_option_forms_are_the_old_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() || !supported_on(0) {
        println!("no sm_80+ kernel here: skipped");
        return;
    }
    let (e, hc) = (2560i64, 4i64);
    let k = e * hc;
    let sms = crate::hw::per_device::sm_count(crate::hw::per_device::current().unwrap_or(0));
    for n in [12288i64, 8067, 129, 1] {
        let ctx = Context::new(16, 16);
        let res = ctx.new_tensor(types::BF16, &[e, hc, n]).as_input();
        let sc = ctx.new_tensor(types::F32, &[hc, n]).as_input();
        let g = ctx.new_tensor(types::F32, &[k]).as_input();
        let down = ctx.new_tensor(types::BF16, &[k, M]).as_input();
        let lo = mul_mat(&ctx, down, res, sc, g).as_output();
        let mut graph = ctx.new_graph(16);
        graph.expand(lo);
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        res.set_bytes(&bf16_noise((k * n) as usize, 101, 4.0));
        // Scales differing by token and stream, so a value scaled with another's fails.
        sc.set_f32(
            &noise((hc * n) as usize, 102)
                .iter()
                .map(|v| 0.25 + v.abs())
                .collect::<Vec<_>>(),
        );
        g.set_f32(
            &noise(k as usize, 103)
                .iter()
                .map(|v| 1.0 + 0.3 * v)
                .collect::<Vec<_>>(),
        );
        down.set_bytes(&bf16_noise((k * M) as usize, 104, 0.02));
        let poison = vec![0xffu8; (M * n * 4) as usize];
        let mut first: Option<Vec<u8>> = None;
        for nt in [0, 2, 3, 4] {
            let want_nt = if nt == 0 {
                pick_nt(n, i64::from(sms))
            } else {
                nt
            };
            let mut base: Option<Vec<u8>> = None;
            for o in [0, OPT_SHARED_REBUILD, OPT_LATE_COPIES, OPT_ALL] {
                force_nt(nt);
                force_opts(Some(o));
                clear_last_launch();
                lo.set_bytes(&poison);
                let r = gpu.compute(&graph);
                let ran = last_launch();
                force_nt(0);
                force_opts(None);
                r.unwrap();
                assert_eq!(
                    ran,
                    Some((want_nt, o)),
                    "{n} tokens, tile {nt}, options {o}: launched {ran:?}"
                );
                let got = lo.bytes();
                let skipped = got
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|w| **w == [0xff; 4])
                    .count();
                assert_eq!(
                    skipped, 0,
                    "{n} tokens, tile {want_nt}, options {o}: outputs not stored"
                );
                match &base {
                    None => base = Some(got),
                    Some(b) => {
                        let differing = got
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .zip(b.as_chunks::<4>().0)
                            .filter(|(x, y)| x != y)
                            .count();
                        assert_eq!(
                            differing, 0,
                            "{n} tokens, tile {want_nt}, options {o}: outputs differ from the old form's"
                        );
                    }
                }
            }
            let base = base.unwrap();
            match &first {
                None => first = Some(base),
                Some(f) => assert!(
                    *f == base,
                    "{n} tokens: tile {want_nt} differs from the step size's tile"
                ),
            }
            println!("{n} tokens, tile {want_nt}: every option form equals the old form");
        }
    }
}

/// The microbench of the option forms ([`bench::ab`]) at a 32K prompt's step sizes: the old form
/// against each option form, taking turns run by run (the order reversed every other round),
/// median, best and speedup per form; it fails if a run launched another tile or form than asked
/// for. Prints only (`--nocapture`); `legion hc-down-bench --ab` is the same from a binary.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_kernel_option_forms_ab() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::model::fp8::bf16_mma() || !supported_on(0) {
        println!("no sm_80+ kernel here: skipped");
        return;
    }
    let cfg = bench::Config {
        tokens: vec![12288, 8067],
        copies: 8,
        rounds: 20,
        ..bench::Config::default()
    };
    let forms = [OPT_SHARED_REBUILD, OPT_LATE_COPIES, OPT_ALL];
    let rows = bench::ab(&mut gpu, &cfg, &forms, 0).unwrap();
    assert_eq!(rows.len(), cfg.tokens.len() * (forms.len() + 1));
    println!("tokens nt form    median_us    min_us  speedup");
    for r in &rows {
        assert!(r.median > 0.0 && r.min <= r.median);
        println!(
            "{:>6} {:>2} {:<6} {:>10.1} {:>9.1} {:>8.3}",
            r.tokens,
            r.nt,
            bench::form_name(r.opts),
            r.median * 1e6,
            r.min * 1e6,
            r.speedup
        );
    }
}
