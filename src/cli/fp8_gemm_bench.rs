//! `legion fp8-gemm-bench`: the TMA-fed W8A8 prefill GEMM (`--fp8-act a8-dynamic`, #1133
//! round 15) at prefill shapes: each kernel variant's outputs checked against the default
//! kernel's bits and timed, plus probes of the kernel (no stores, no MMAs, no shared-memory
//! fragment loads, loads alone, no refills) to see what bounds it (a development tool, so a
//! build tree's release binary can check the GPU it runs on; pick it with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct Fp8GemmBenchArgs {
    /// Shapes as `k:m:n` (weight rows of `k` bytes, `m` outputs, `n` tokens) or `k:m:n:res` (a
    /// residual added in the epilogue), comma separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "5120:10240:16384,5120:6144:16384,5120:12288:16384,5120:1024:16384,6144:5120:16384:res"
    )]
    shapes: Vec<String>,
    /// Kernel variants timed against the default kernel (1): 0 the default pick, 2 the
    /// epilogue's scales loaded before its stores, 3 each stage released before its last k
    /// step's MMAs, 4 both, 5 4 on the wide 8-warp tile only (the residual tile as 1).
    #[arg(long, value_delimiter = ',', default_values_t = [2, 3, 4, 5])]
    variants: Vec<i32>,
    /// Probes of the default kernel (outputs not the GEMM's): 1 no epilogue stores, 2 no MMAs,
    /// 3 no shared-memory fragment loads, 4 the copies alone (2 and 3), 5 no refills (the first
    /// stages reread), 6 no refills and no stores.
    #[arg(long, value_delimiter = ',')]
    probes: Vec<i32>,
    /// One per-tensor weight scale instead of one per output row.
    #[arg(long)]
    tensor_scale: bool,
    /// The GPU's SMs (for which tile the GEMM takes, in the copy rate column).
    #[arg(long, default_value_t = 48)]
    sms: usize,
    /// Weights in a graph (distinct ones, so the graph streams past the L2).
    #[arg(long, default_value_t = 2)]
    layers: usize,
    /// Timed graphs per variant and round.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (the variants take turns within a round; the best counts).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

#[cfg(feature = "cuda")]
struct Shape {
    k: usize,
    m: usize,
    n: usize,
    res: bool,
}

#[cfg(feature = "cuda")]
fn parse(s: &str) -> anyhow::Result<Shape> {
    let parts: Vec<&str> = s.split(':').map(str::trim).collect();
    anyhow::ensure!(
        (3..=4).contains(&parts.len()),
        "--shapes takes k:m:n[:res], not {s}"
    );
    let num = |i: usize| -> anyhow::Result<usize> { Ok(parts[i].parse::<usize>()?) };
    let tag = parts.get(3).copied().unwrap_or("");
    anyhow::ensure!(matches!(tag, "" | "res"), "--shapes: {tag} is not res");
    let shape = Shape {
        k: num(0)?,
        m: num(1)?,
        n: num(2)?,
        res: tag == "res",
    };
    anyhow::ensure!(
        shape.k % 128 == 0 && shape.m % 128 == 0 && shape.n > 64,
        "--shapes: k and m multiples of 128, n over 64 ({s})"
    );
    Ok(shape)
}

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
}

/// Random E4M3 codes of a `[k, m]` weight: any sign and mantissa, exponents 3 to 10 (no NaN).
#[cfg(feature = "cuda")]
fn weights(k: usize, m: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(m * k);
    while out.len() < m * k {
        for b in rng.next().to_le_bytes() {
            out.push(b & 0x87 | (3 + (b >> 3) % 8) << 3);
        }
    }
    out.truncate(m * k);
    out
}

/// The probe bits of `--probes` value `p` (a8t's P_* in `src/cuda/sm90/fp8_gemm8_tma.cuh`).
#[cfg(feature = "cuda")]
fn probe_bits(p: i32) -> anyhow::Result<i32> {
    Ok(match p {
        1 => 0x100,
        2 => 0x200,
        3 => 0x400,
        4 => 0x600,
        5 => 0x800,
        6 => 0x900,
        _ => anyhow::bail!("--probes takes 1 to 6"),
    })
}

#[cfg(feature = "cuda")]
pub fn run(args: Fp8GemmBenchArgs) -> anyhow::Result<()> {
    use legion::ggml::{ComputeAllocator, Context, types};
    use legion::model::fp8;
    anyhow::ensure!(
        args.reps > 0 && args.rounds > 0,
        "--reps and --rounds take 1 or more"
    );
    anyhow::ensure!(args.layers > 0, "--layers takes 1 or more");
    let shapes = args
        .shapes
        .iter()
        .map(|s| parse(s))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let probes = args
        .probes
        .iter()
        .map(|&p| probe_bits(p))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let layers = args.layers;
    let mut differ = 0;
    println!(
        "{:>6} {:>6} {:>6} {:>4} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "k", "m", "n", "op", "variant", "ms", "TF", "x v1", "L2 GB/s", "mem GB/s", "bits"
    );
    for s in &shapes {
        let (k, m, n) = (s.k as i64, s.m as i64, s.n as i64);
        let weights_ctx = Context::new(2 * layers, 0);
        let ws: Vec<_> = (0..layers)
            .map(|_| weights_ctx.new_tensor(types::I8, &[k, m]))
            .collect();
        let scale_len = if args.tensor_scale { 1 } else { m };
        let scales: Vec<_> = (0..layers)
            .map(|_| weights_ctx.new_tensor(types::F32, &[scale_len]))
            .collect();
        weights_ctx.alloc_tensors(gpu.buffer_type())?;
        for (i, (w, sc)) in ws.iter().zip(&scales).enumerate() {
            w.set_bytes(&weights(s.k, s.m, 5 + i as u64));
            let mut rng = Rng(0x55 + i as u64);
            let v: Vec<f32> = (0..scale_len)
                .map(|_| 0.002 + 0.001 * rng.unit().abs())
                .collect();
            sc.set_f32(&v);
        }

        // The inputs are outputs too, so the allocator places no node over them once read (the
        // graph runs again without filling them again).
        let ctx = Context::new(8 * layers + 8, 8 * layers + 8);
        let x = ctx.new_tensor(types::F32, &[k, n]).as_input().as_output();
        let res = s
            .res
            .then(|| ctx.new_tensor(types::F32, &[m, n]).as_input().as_output());
        let mut graph = ctx.new_graph(8 * layers + 8);
        let outs: Vec<_> = (0..layers)
            .map(|l| {
                let y = match res {
                    Some(r) => fp8::mul_mat_dynamic_add(&ctx, ws[l], x, Some(scales[l]), r),
                    None => fp8::mul_mat_dynamic(&ctx, ws[l], x, Some(scales[l])),
                }
                .as_output();
                graph.expand(y);
                y
            })
            .collect();
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))?;
        // The activations' quantization alone (once per graph), timed to take it out of the
        // GEMMs' times.
        let qctx = Context::new(4, 4);
        let qx = qctx.new_tensor(types::F32, &[k, n]).as_input().as_output();
        let mut qgraph = qctx.new_graph(4);
        qgraph.expand(fp8::quantize_dynamic(&qctx, qx).as_output());
        qctx.alloc_graph(&qgraph, ComputeAllocator::new(gpu.buffer_type()))?;

        let mut rng = Rng(0x1234_5678 + s.n as u64);
        let xs: Vec<f32> = (0..s.k * s.n).map(|_| rng.unit()).collect();
        x.set_f32(&xs);
        qx.set_f32(&xs);
        if let Some(r) = res {
            let rs: Vec<f32> = (0..s.m * s.n).map(|_| rng.unit()).collect();
            r.set_f32(&rs);
        }

        // Variant 1 (the default kernel) is the reference, then the variants, then the probes.
        let mut runs: Vec<i32> = vec![1];
        runs.extend(args.variants.iter().copied().filter(|&v| v != 1));
        runs.extend(&probes);
        let mut reference: Vec<u8> = Vec::new();
        let mut rows: Vec<(i32, Option<f64>, Option<usize>)> = Vec::new();
        for &v in &runs {
            fp8::force_gemm_variant(v);
            // A warm-up run, then the bits.
            let ok = gpu.compute(&graph).is_ok() && gpu.compute(&graph).is_ok();
            if !ok {
                fp8::force_gemm_variant(0);
                anyhow::ensure!(v != 1, "k {k} m {m} n {n}: variant 1 does not run");
                rows.push((v, None, None));
                // A failed launch may leave the backend's error state; a plain run clears it.
                gpu.compute(&graph)?;
                continue;
            }
            let got: Vec<u8> = outs.iter().flat_map(|o| o.bytes()).collect();
            if v == 1 {
                reference = got;
                rows.push((v, Some(f64::INFINITY), Some(0)));
                continue;
            }
            let bad = got
                .chunks(4)
                .zip(reference.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            if v >> 8 == 0 {
                differ += bad;
            }
            rows.push((v, Some(f64::INFINITY), Some(bad)));
        }
        let mut quant = f64::INFINITY;
        for _ in 0..args.rounds {
            let t = std::time::Instant::now();
            for _ in 0..args.reps {
                gpu.compute(&qgraph)?;
            }
            quant = quant.min(t.elapsed().as_secs_f64() / args.reps as f64);
            for row in rows.iter_mut() {
                let Some(best) = row.1 else { continue };
                fp8::force_gemm_variant(row.0);
                let t = std::time::Instant::now();
                for _ in 0..args.reps {
                    gpu.compute(&graph)?;
                }
                // One quantization per graph, then `layers` GEMMs.
                let secs = (t.elapsed().as_secs_f64() / args.reps as f64 - quant) / layers as f64;
                row.1 = Some(best.min(secs));
            }
        }
        fp8::force_gemm_variant(0);
        let flops = 2.0 * (s.k * s.m * s.n) as f64;
        let op = if s.res { "res" } else { "mm" };
        // The tile launch_gemm8 takes: 128 x 128 (4 warps, two CTAs per SM) with the residual,
        // else 256 x 128 where it makes 8 waves, else the 128-row cp.async kernel.
        let tiles_n = s.n.div_ceil(128);
        let tm = if s.m % 256 == 0 && s.m / 256 * tiles_n >= 8 * args.sms && !s.res {
            256
        } else {
            128
        };
        let copied = (s.m / tm * tiles_n * (tm + 128) * s.k) as f64;
        let memory = (s.k * s.m + s.k * s.n + 4 * s.m * s.n * if s.res { 2 } else { 1 }) as f64;
        let base = rows[0].1;
        println!(
            "{k:>6} {m:>6} {n:>6} {op:>4} {:>8} {:>9.3}",
            "quant",
            quant * 1e3
        );
        for (v, secs, bad) in rows {
            let name = match v >> 8 {
                0 => format!("{v}"),
                _ => format!(
                    "p{}",
                    args.probes[probes.iter().position(|&p| p == v).unwrap_or(0)]
                ),
            };
            let Some(secs) = secs else {
                println!("{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9}", "none");
                continue;
            };
            let gemm = secs.max(1e-9);
            let bits = match (v >> 8, bad) {
                (0, Some(0)) => "same".to_string(),
                (0, Some(b)) => format!("{b} DIFF"),
                _ => "probe".to_string(),
            };
            println!(
                "{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9.3} {:>8.1} {:>8.3} {:>8.0} {:>8.0} {bits:>7}",
                gemm * 1e3,
                flops / gemm / 1e12,
                base.map_or(0.0, |b| b / gemm),
                copied / gemm / 1e9,
                memory / gemm / 1e9
            );
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the bits of variant 1");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: Fp8GemmBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("fp8-gemm-bench needs a CUDA build")
}
