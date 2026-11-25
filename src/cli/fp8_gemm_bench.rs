//! `legion fp8-gemm-bench`: the TMA-fed W8A8 prefill GEMM (`--fp8-act a8-dynamic`) at prefill
//! shapes, each epilogue variant's outputs checked against the plain epilogue's bits and
//! timed (#1133 round 25; a development tool, so a build tree's release binary can check the
//! GPU it runs on; pick it with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct Fp8GemmBenchArgs {
    /// Shapes as `k:m:n` (weight rows of `k` bytes, `m` outputs, `n` tokens) or `k:m:n:res` (a
    /// residual added in the epilogue), comma separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "6144:5120:8192:res,17408:5120:8192:res,5120:10240:8192"
    )]
    shapes: Vec<String>,
    /// Epilogue variants timed against the plain epilogue (1): 0 the pick without the TMA
    /// stores (the streaming route), 2 and 3 the streaming epilogue with one and two token tiles
    /// at a time, 256 the pick without the streaming route (LEGION_DEBUG_FP8_STREAM_EPI=0), 512
    /// the staged epilogue (the TP send buffer's), 2048 the wide tile's TMA stores (the
    /// default), 8192 the wide tile's staggered first wave, 10240 both; bits add up.
    #[arg(long, value_delimiter = ',', default_values_t = [0, 3, 512, 2048, 8192])]
    variants: Vec<i32>,
    /// One per-tensor weight scale instead of one per output row.
    #[arg(long)]
    tensor_scale: bool,
    /// Weights in a graph (distinct ones, so the graph streams past the L2).
    #[arg(long, default_value_t = 2)]
    layers: usize,
    /// Timed graphs per variant and round (the variants take turns graph by graph).
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (each runs `reps` turns of every variant; the median and the best of all graphs
    /// are shown).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

/// The median and the smallest of `xs` (not empty).
#[cfg(feature = "cuda")]
fn med_min(xs: &[f64]) -> (f64, f64) {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    let med = if v.len().is_multiple_of(2) {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    };
    (med, v[0])
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
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let layers = args.layers;
    let mut differ = 0;
    println!(
        "{:>6} {:>6} {:>6} {:>4} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}",
        "k", "m", "n", "op", "variant", "med ms", "min ms", "TF", "x v1", "x v0", "bits"
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

        // Variant 1 (the plain epilogue) is the reference, then the variants.
        let mut runs: Vec<i32> = vec![1];
        runs.extend(args.variants.iter().copied().filter(|&v| v != 1));
        let mut reference: Vec<u8> = Vec::new();
        let mut rows: Vec<(i32, Option<Vec<f64>>, Option<usize>)> = Vec::new();
        for &v in &runs {
            fp8::force_gemm_variant(v);
            // A warm-up run, then the bits.
            let ok = gpu.compute(&graph).is_ok() && gpu.compute(&graph).is_ok();
            if !ok {
                fp8::unforce_gemm_variant();
                anyhow::ensure!(v != 1, "k {k} m {m} n {n}: variant 1 does not run");
                rows.push((v, None, None));
                // A failed launch may leave the backend's error state; a plain run clears it.
                gpu.compute(&graph)?;
                continue;
            }
            let got: Vec<u8> = outs.iter().flat_map(|o| o.bytes()).collect();
            if v == 1 {
                reference = got;
                rows.push((v, Some(Vec::new()), Some(0)));
                continue;
            }
            let bad = got
                .chunks(4)
                .zip(reference.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            differ += bad;
            rows.push((v, Some(Vec::new()), Some(bad)));
        }
        // The variants take turns graph by graph (the quantization alone among them), so the
        // clock's drift and the box's heat spread over all of them alike.
        let mut quant: Vec<f64> = Vec::new();
        for _ in 0..args.rounds * args.reps {
            let t = std::time::Instant::now();
            gpu.compute(&qgraph)?;
            quant.push(t.elapsed().as_secs_f64() / layers as f64);
            for row in rows.iter_mut() {
                let Some(times) = row.1.as_mut() else {
                    continue;
                };
                fp8::force_gemm_variant(row.0);
                let t = std::time::Instant::now();
                gpu.compute(&graph)?;
                times.push(t.elapsed().as_secs_f64() / layers as f64);
            }
        }
        fp8::unforce_gemm_variant();
        let flops = 2.0 * (s.k * s.m * s.n) as f64;
        let op = if s.res { "res" } else { "mm" };
        let (q_med, q_min) = med_min(&quant);
        // The GEMM alone: a graph's median (best) less the quantization's median (best).
        let gemm_of = |times: &Vec<f64>| {
            let (med, min) = med_min(times);
            ((med - q_med).max(1e-9), (min - q_min).max(1e-9))
        };
        let v1 = rows[0].1.as_ref().map(gemm_of);
        let v0 = rows
            .iter()
            .find(|r| r.0 == 0)
            .and_then(|r| r.1.as_ref().map(gemm_of));
        println!(
            "{k:>6} {m:>6} {n:>6} {op:>4} {:>8} {:>9.3} {:>9.3}",
            "quant",
            q_med * 1e3,
            q_min * 1e3
        );
        for (v, times, bad) in rows {
            let Some(times) = times else {
                println!("{k:>6} {m:>6} {n:>6} {op:>4} {v:>8} {:>9}", "none");
                continue;
            };
            let (med, min) = gemm_of(&times);
            let bits = match bad {
                Some(0) => "same".to_string(),
                Some(b) => format!("{b} DIFF"),
                None => "-".to_string(),
            };
            println!(
                "{k:>6} {m:>6} {n:>6} {op:>4} {v:>8} {:>9.3} {:>9.3} {:>8.1} {:>8.3} {:>8.3} {bits:>7}",
                med * 1e3,
                min * 1e3,
                flops / med / 1e12,
                v1.map_or(0.0, |b| b.0 / med),
                v0.map_or(0.0, |b| b.0 / med)
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
