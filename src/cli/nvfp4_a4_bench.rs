//! `legion nvfp4-a4-bench`: the dense W4A4 NVFP4 GEMM's TMA-fed tiles (#1133) at prefill
//! shapes: each tile variant's outputs checked against the 4-block-stage tile's bits and timed,
//! plus probes of the mainloop (loads alone, no refills, no stores) to see what bounds it (a
//! development tool, so a build tree's release binary can check the GPU it runs on; pick it
//! with `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct Nvfp4A4BenchArgs {
    /// Shapes as `k:m:n` (weight rows of `k`, `m` outputs, `n` tokens), `k:m:n:glu` (the fused
    /// SwiGLU of two such weights) or `k:m:n:res` (a residual added in the epilogue), comma
    /// separated.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "5120:17408:8192:glu,17408:5120:8192:res,5376:21504:2048:glu,21504:5376:2048:res,5376:8192:2048"
    )]
    shapes: Vec<String>,
    /// Tile variants timed against the 4-block-stage tile (1): 0 the default pick, 2 8 token
    /// tiles x 2-block stages x 4, 3 6 x 2 x 5, 4 8 x 4 x 2, 5 6 x 4 x 3, 6 8 x 2 x 3.
    #[arg(long, value_delimiter = ',', default_values_t = [0, 2, 3, 4, 5, 6])]
    variants: Vec<i32>,
    /// Probes of tiles 2 and 4 on plain shapes (outputs not the GEMM's): 1 the loads alone, 2
    /// no refills, 3 no epilogue stores.
    #[arg(long, value_delimiter = ',')]
    probes: Vec<i32>,
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
    glu: bool,
    res: bool,
}

#[cfg(feature = "cuda")]
fn parse(s: &str) -> anyhow::Result<Shape> {
    let parts: Vec<&str> = s.split(':').map(str::trim).collect();
    anyhow::ensure!(
        (3..=4).contains(&parts.len()),
        "--shapes takes k:m:n[:glu|:res], not {s}"
    );
    let num = |i: usize| -> anyhow::Result<usize> { Ok(parts[i].parse::<usize>()?) };
    let tag = parts.get(3).copied().unwrap_or("");
    anyhow::ensure!(
        matches!(tag, "" | "glu" | "res"),
        "--shapes: {tag} is not glu or res"
    );
    let shape = Shape {
        k: num(0)?,
        m: num(1)?,
        n: num(2)?,
        glu: tag == "glu",
        res: tag == "res",
    };
    anyhow::ensure!(
        shape.k % 256 == 0 && shape.m % 128 == 0 && shape.n > 0,
        "--shapes: k a multiple of 256, m of 128 ({s})"
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

/// Random NVFP4 blocks of a `[k, m]` weight (36 bytes per 64 values: 4 sub-block scales of
/// moderate size, then 32 bytes of codes).
#[cfg(feature = "cuda")]
fn weights(k: usize, m: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut out = Vec::with_capacity(m * k / 64 * 36);
    for _ in 0..m * k / 64 {
        for _ in 0..4 {
            let r = rng.next();
            out.push(((3 + (r >> 8) % 6) << 3 | (r >> 16) & 7) as u8);
        }
        for _ in 0..4 {
            out.extend(rng.next().to_le_bytes());
        }
    }
    out
}

#[cfg(feature = "cuda")]
pub fn run(args: Nvfp4A4BenchArgs) -> anyhow::Result<()> {
    use legion::ggml::{ComputeAllocator, Context, types};
    use legion::model::nvfp4_a4 as a4;
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
        "{:>6} {:>6} {:>6} {:>4} {:>8} {:>9} {:>8} {:>8} {:>7}",
        "k", "m", "n", "op", "variant", "ms", "TF", "x tile1", "bits"
    );
    for s in &shapes {
        let (k, m, n) = (s.k as i64, s.m as i64, s.n as i64);
        let n_w = if s.glu { 2 * layers } else { layers };
        let weights_ctx = Context::new(n_w, 0);
        let ws: Vec<_> = (0..n_w)
            .map(|_| weights_ctx.new_tensor(types::NVFP4, &[k, m]))
            .collect();
        weights_ctx.alloc_tensors(gpu.buffer_type())?;
        for (i, w) in ws.iter().enumerate() {
            w.set_bytes(&weights(s.k, s.m, 5 + i as u64));
        }

        let ctx = Context::new(8 * layers + 8, 8 * layers + 8);
        let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
        let res = s
            .res
            .then(|| ctx.new_tensor(types::F32, &[m, n]).as_input());
        let mut graph = ctx.new_graph(8 * layers + 8);
        let outs: Vec<_> = (0..layers)
            .map(|l| {
                let y = if s.glu {
                    a4::swiglu(&ctx, ws[2 * l], ws[2 * l + 1], x, None, None)
                } else if let Some(r) = res {
                    a4::mul_mat_add(&ctx, ws[l], x, None, r)
                } else {
                    a4::mul_mat(&ctx, ws[l], x, None)
                }
                .as_output();
                graph.expand(y);
                y
            })
            .collect();
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))?;
        // The ops' activation quantization alone, timed to take it out of the GEMMs' times.
        let qctx = Context::new(2 * layers + 4, 2 * layers + 4);
        let qx = qctx.new_tensor(types::F32, &[k, n]).as_input();
        let mut qgraph = qctx.new_graph(2 * layers + 4);
        for _ in 0..layers {
            qgraph.expand(a4::quantize_op(&qctx, qx).as_output());
        }
        qctx.alloc_graph(&qgraph, ComputeAllocator::new(gpu.buffer_type()))?;

        let mut rng = Rng(0x1234_5678 + s.n as u64);
        let xs: Vec<f32> = (0..s.k * s.n).map(|_| rng.unit()).collect();
        x.set_f32(&xs);
        qx.set_f32(&xs);
        if let Some(r) = res {
            let rs: Vec<f32> = (0..s.m * s.n).map(|_| rng.unit()).collect();
            r.set_f32(&rs);
        }

        // Tile 1 (the 4-block stages) is the reference, then the variants, then the probes.
        let mut runs: Vec<i32> = vec![1];
        runs.extend(args.variants.iter().copied().filter(|&v| v != 1));
        if !s.glu && !s.res {
            for &p in &args.probes {
                anyhow::ensure!((1..=3).contains(&p), "--probes takes 1, 2 or 3");
                runs.push(2 | p << 8);
                runs.push(4 | p << 8);
            }
        }
        let elem = if s.glu { 2 } else { 4 };
        let mut reference: Vec<u8> = Vec::new();
        let mut rows: Vec<(i32, Option<f64>, Option<usize>)> = Vec::new();
        for &v in &runs {
            a4::force_dense_variant(v);
            // A warm-up run, then the bits.
            let ok = gpu.compute(&graph).is_ok() && gpu.compute(&graph).is_ok();
            if !ok {
                a4::force_dense_variant(0);
                anyhow::ensure!(v != 1, "k {k} m {m} n {n}: tile 1 does not run");
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
                .chunks(elem)
                .zip(reference.chunks(elem))
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
            quant = quant.min(t.elapsed().as_secs_f64() / (args.reps * layers) as f64);
            for row in rows.iter_mut() {
                let Some(best) = row.1 else { continue };
                a4::force_dense_variant(row.0);
                let t = std::time::Instant::now();
                for _ in 0..args.reps {
                    gpu.compute(&graph)?;
                }
                let secs = t.elapsed().as_secs_f64() / (args.reps * layers) as f64;
                row.1 = Some(best.min(secs));
            }
        }
        a4::force_dense_variant(0);
        let flops = 2.0 * (s.k * s.m * s.n) as f64 * if s.glu { 2.0 } else { 1.0 };
        let op = if s.glu {
            "glu"
        } else if s.res {
            "res"
        } else {
            "mm"
        };
        let base = rows[0].1.map(|t| (t - quant).max(1e-9));
        println!(
            "{k:>6} {m:>6} {n:>6} {op:>4} {:>8} {:>9.3}",
            "quant",
            quant * 1e3
        );
        for (v, secs, bad) in rows {
            let name = match v >> 8 {
                0 => format!("{v}"),
                p => format!("{}p{p}", v & 0xff),
            };
            let Some(secs) = secs else {
                println!("{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9}", "none");
                continue;
            };
            let gemm = (secs - quant).max(1e-9);
            let bits = match (v >> 8, bad) {
                (_, Some(0)) => "same".to_string(),
                (0, Some(b)) => format!("{b} DIFF"),
                _ => "probe".to_string(),
            };
            println!(
                "{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9.3} {:>8.1} {:>8.3} {bits:>7}",
                gemm * 1e3,
                flops / gemm / 1e12,
                base.map_or(0.0, |b| b / gemm)
            );
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the 4-block-stage tile's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: Nvfp4A4BenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("nvfp4-a4-bench needs a CUDA build")
}
