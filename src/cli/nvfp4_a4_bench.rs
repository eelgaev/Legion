//! `legion nvfp4-a4-bench`: the dense W4A4 NVFP4 GEMM's TMA-fed tiles (#1133) at prefill
//! shapes: each tile variant's outputs checked against the bits of the kernel before #1133
//! round 12 and timed,
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
    /// Tile variants timed against the kernel before #1133 r12 (1): 0 the default pick; rows x
    /// tokens x stages with fragments by LDS: 2 128 x 128 x 2, 3 128 x 96 x 3; by generic loads
    /// (as before): 4 128 x 128 x 2, 5 128 x 96 x 3. Plain and residual GEMMs only: 6 192 x 128
    /// x 2 (#1133 r20); the 192 x 128 tile with its epilogue through shared memory (#1133 r21):
    /// 13 persistent with 4 blocks a stage x 2 stages, 14 persistent with 2 blocks x 3 stages,
    /// 15 and 16 the same tiles one per CTA, 17 and 18 13 in groups of 16 and 4 row tiles; 10,
    /// 11 and 12 13, 18 and 15 with the residual read from global memory by every thread. Two
    /// groups of 4 warps taking turns on their own tiles (#1133 r23, persistent, groups of 8 row
    /// tiles): 30 96 x 128 x 3, 31 128 x 96 x 3, 32 128 x 128 x 2; 33 and 34 30 and 31 in
    /// groups of 4; 35 32 with streaming epilogue stores and residual loads. 36 and 37 the
    /// default 128 x 96 tile with that epilogue, plain and streaming.
    #[arg(long, value_delimiter = ',', default_values_t = [0, 2, 3, 4, 5])]
    variants: Vec<i32>,
    /// Probes of the variants 2, 3, 4, 6, 13, 14 and 30 to 32 on plain shapes (outputs not the
    /// GEMM's): 1 the loads alone, 2 no refills (not 13, 14 and 30 to 32), 3 no epilogue stores.
    #[arg(long, value_delimiter = ',')]
    probes: Vec<i32>,
    /// Instead of timing: one launch of the TMA multicast probe on the first shape (plain, at
    /// least 96 tokens) with this mode (20 plain loads, 21 CTA 0 multicasts into both CTAs of a
    /// cluster pair, 22 each CTA into itself, 23 CTA 0 into CTA 1 only), and what each CTA saw.
    /// A mode that hangs the GPU never returns: run it under a timeout.
    #[arg(long)]
    mc_probe: Option<i32>,
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

/// The bytes variant `v`'s CTAs read from the L2 for a shape (plain, residual or SwiGLU; a
/// cluster pair's shared operand counts half for each CTA), and the bytes a CTA reads per k
/// block (64 of k) and 1,000 of its row x token products.
#[cfg(feature = "cuda")]
fn tile_reads(v: i32, s: &Shape) -> (f64, f64) {
    let deep = !s.glu && s.k >= 8192 && s.n >= 2048;
    let (bm, bn, share) = match v & 0xff {
        0 | 1 if deep => (128, 96, 0),
        0 | 1 | 2 | 4 => (128, 128, 0),
        3 | 5 => (128, 96, 0),
        6 | 10..=18 => (192, 128, 0),
        30 | 33 => (96, 128, 0),
        31 | 34 | 36 | 37 => (128, 96, 0),
        _ => (128, 128, 0),
    };
    // 2 blocks a stage (14, 16): 80-byte weight boxes (72 used), 16-byte scale boxes (8 used).
    let per = if matches!(v & 0xff, 14 | 16) {
        160.0
    } else {
        144.0
    };
    let out_rows = if s.glu { bm / 2 } else { bm };
    let tiles_m = s.m.div_ceil(out_rows);
    let mut tiles_n = s.n.div_ceil(bn);
    if share == 2 {
        tiles_n = tiles_n.div_ceil(2) * 2;
    }
    let rows = (if share == 2 { bm / 2 } else { bm }) as f64;
    let toks = (if share == 1 { bn / 2 } else { bn }) as f64;
    // 256 of k: each row's 4 blocks of 36 bytes, each token's 128 code bytes and 16 scale bytes.
    let bytes = (tiles_m * tiles_n * (s.k / 256)) as f64 * per * (rows + toks);
    (bytes, per / 4.0 * (rows + toks) / (bm * bn) as f64 * 1e3)
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
        "{:>6} {:>6} {:>6} {:>4} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7}",
        "k",
        "m",
        "n",
        "op",
        "variant",
        "med ms",
        "min ms",
        "TF",
        "x v1",
        "x v0",
        "L2 MB",
        "TB/s",
        "B/kMMA",
        "bits"
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
        if let Some(mode) = args.mc_probe {
            anyhow::ensure!(
                (20..=23).contains(&mode) && !s.glu && !s.res && s.n >= 96,
                "--mc-probe takes 20 to 23 and a plain shape of 96 tokens or more"
            );
            a4::force_dense_variant(mode);
            let done = gpu.compute(&graph);
            a4::force_dense_variant(0);
            done?;
            let b = outs[0].bytes();
            let f: Vec<f32> = b[..32]
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            for (rank, v) in f.chunks(4).enumerate() {
                println!(
                    "mc-probe mode {mode} cta {rank}: {} after {} us, checksum {}, mode seen {}",
                    if v[0] > 0.0 { "done" } else { "NOT DONE" },
                    v[1],
                    v[2],
                    v[3]
                );
            }
            return Ok(());
        }

        // Variant 1 (the kernel before) is the reference, then the variants, then the probes.
        let mut runs: Vec<i32> = vec![1];
        runs.extend(args.variants.iter().copied().filter(|&v| v != 1));
        if !s.glu && !s.res {
            for &p in &args.probes {
                anyhow::ensure!((1..=3).contains(&p), "--probes takes 1, 2 or 3");
                let probed = args.variants.iter().filter(|&&v| {
                    matches!(v, 2..=4 | 6) || (matches!(v, 13 | 14 | 30..=32) && p != 2)
                });
                runs.extend(probed.map(|&v| v | p << 8));
            }
        }
        let elem = if s.glu { 2 } else { 4 };
        let mut reference: Vec<u8> = Vec::new();
        let mut rows: Vec<(i32, Option<Vec<f64>>, Option<usize>)> = Vec::new();
        for &v in &runs {
            a4::force_dense_variant(v);
            // A warm-up run, then the bits.
            let ok = gpu.compute(&graph).is_ok() && gpu.compute(&graph).is_ok();
            if !ok {
                a4::force_dense_variant(0);
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
                .chunks(elem)
                .zip(reference.chunks(elem))
                .filter(|(a, b)| a != b)
                .count();
            if v >> 8 == 0 {
                differ += bad;
            }
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
                a4::force_dense_variant(row.0);
                let t = std::time::Instant::now();
                gpu.compute(&graph)?;
                times.push(t.elapsed().as_secs_f64() / layers as f64);
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
            let name = match v >> 8 {
                0 => format!("{v}"),
                p => format!("{}p{p}", v & 0xff),
            };
            let Some(times) = times else {
                println!("{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9}", "none");
                continue;
            };
            let (med, min) = gemm_of(&times);
            let bits = match (v >> 8, bad) {
                (_, Some(0)) => "same".to_string(),
                (0, Some(b)) => format!("{b} DIFF"),
                _ => "probe".to_string(),
            };
            let (l2, per_mma) = tile_reads(v, s);
            println!(
                "{k:>6} {m:>6} {n:>6} {op:>4} {name:>8} {:>9.3} {:>9.3} {:>8.1} {:>8.3} {:>8.3} {:>9.1} {:>7.2} {:>7.1} {bits:>7}",
                med * 1e3,
                min * 1e3,
                flops / med / 1e12,
                v1.map_or(0.0, |b| b.0 / med),
                v0.map_or(0.0, |b| b.0 / med),
                l2 / 1e6,
                l2 / med / 1e12,
                per_mma
            );
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the bits of variant 1");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: Nvfp4A4BenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("nvfp4-a4-bench needs a CUDA build")
}
