//! `legion fp8-sliced-bench`: the sliced-K FP8 GEMM's tiles (`--fp8-sliced-k`, the verify steps'
//! W8A16 matmuls) at Qwen3.8-27B's FP8 shapes on one GPU, each timed and its output bits
//! checked against the tile the GEMM picks by the token count (a development tool, so a build
//! tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::ggml::{ComputeAllocator, Context, types};
use legion::model::fp8::{self, SmallBatch};

#[derive(Args)]
pub struct Fp8SlicedBenchArgs {
    /// Tokens per step, comma separated (5-64).
    #[arg(long, value_delimiter = ',', default_values_t = [8usize])]
    tokens: Vec<usize>,
    /// Weight shapes as `k x rows`, comma separated (default: the 27B's GDN qkv, z and out,
    /// attention q, MLP gate/up and down).
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<String>,
    /// Tiles to time (indices of `SLICED_KW` in src/cuda/fp8_mm.cu), comma separated; the
    /// picked tile always runs first and is the bits reference.
    #[arg(long, value_delimiter = ',', default_values_t = [0u8, 12, 13, 14, 15, 16, 17, 18, 19])]
    tiles: Vec<u8>,
    /// Timed calls per tile and round (each its own weight, more than the L2 cache holds).
    #[arg(long, default_value_t = 16)]
    layers: usize,
    /// Rounds (the best counts; the tiles take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
}

fn shape(s: &str) -> anyhow::Result<(usize, usize)> {
    let (k, rows) = s
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("shape {s:?} is not `k x rows`"))?;
    Ok((k.trim().parse()?, rows.trim().parse()?))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub fn run(args: Fp8SlicedBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let shapes: Vec<(usize, usize)> = if args.shapes.is_empty() {
        vec![
            (5120, 10240),
            (5120, 6144),
            (6144, 5120),
            (5120, 12288),
            (5120, 17408),
            (17408, 5120),
        ]
    } else {
        args.shapes
            .iter()
            .map(|s| shape(s))
            .collect::<Result<_, _>>()?
    };
    println!(
        "{:>6} {:>6} {:>3} {:>5} {:>9} {:>7} {:>7} {:>9}",
        "k", "rows", "n", "tile", "us/mm", "GB/s", "vs pick", "differ"
    );
    let mut bad = 0;
    for &(k, m) in &shapes {
        let layers = args.layers.max(1);
        let wctx = Context::new(layers + 1, 0);
        let ws: Vec<_> = (0..layers)
            .map(|_| wctx.new_tensor(types::I8, &[k as i64, m as i64]))
            .collect();
        let sc = wctx.new_tensor(types::F32, &[m as i64]);
        wctx.alloc_tensors(gpu.buffer_type())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (k * 31 + m) as u64);
        for w in &ws {
            // E4M3 codes without the NaNs (0x7f, 0xff).
            let bytes: Vec<u8> = (0..k * m)
                .map(|_| {
                    let b = (rng.next() >> 24) as u8;
                    if b & 0x7f == 0x7f { b ^ 0x01 } else { b }
                })
                .collect();
            w.set_bytes(&bytes);
        }
        let scales: Vec<f32> = (0..m)
            .map(|j| 2f32.powi(j as i32 % 7 - 3) * (1.0 + (j % 5) as f32 / 4.0))
            .collect();
        sc.set_f32(&scales);
        for &n in &args.tokens {
            let xs: Vec<f32> = (0..k * n)
                .map(|_| ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
                .collect();
            // The picked tile (None) first: the reference.
            let variants: Vec<Option<u8>> = std::iter::once(None)
                .chain(args.tiles.iter().map(|&t| Some(t)))
                .collect();
            let ctxs: Vec<Context> = variants
                .iter()
                .map(|_| Context::new(4 * layers + 8, 4 * layers + 8))
                .collect();
            let mut graphs = Vec::new();
            for (ctx, &tile) in ctxs.iter().zip(&variants) {
                let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
                let mut graph = ctx.new_graph(4 * layers + 8);
                let small = SmallBatch {
                    gemv: false,
                    sliced_k: true,
                    sliced_k_tile: tile,
                };
                let outs: Vec<_> = ws
                    .iter()
                    .map(|&w| {
                        let y = fp8::mul_mat_small(ctx, w, x, Some(sc), small).as_output();
                        graph.expand(y);
                        y
                    })
                    .collect();
                ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                graphs.push((x, graph, outs));
            }
            let mut reference: Vec<u32> = Vec::new();
            let mut differ = vec![0usize; variants.len()];
            let mut best = vec![f64::MAX; variants.len()];
            for round in 0..=args.rounds.max(1) {
                for (vi, (x, graph, outs)) in graphs.iter().enumerate() {
                    x.set_f32(&xs);
                    let t = std::time::Instant::now();
                    gpu.compute(graph).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let secs = t.elapsed().as_secs_f64() / layers as f64;
                    if round == 0 {
                        let got: Vec<u32> = outs
                            .iter()
                            .flat_map(|o| o.to_f32())
                            .map(f32::to_bits)
                            .collect();
                        if vi == 0 {
                            reference = got;
                        } else {
                            differ[vi] = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
                        }
                    } else {
                        best[vi] = best[vi].min(secs);
                    }
                }
            }
            for (vi, tile) in variants.iter().enumerate() {
                bad += usize::from(differ[vi] > 0);
                println!(
                    "{:>6} {:>6} {:>3} {:>5} {:>9.2} {:>7.1} {:>7.3} {:>9}",
                    k,
                    m,
                    n,
                    tile.map_or("pick".to_string(), |t| format!("t{t}")),
                    best[vi] * 1e6,
                    (k * m) as f64 / best[vi] / 1e9,
                    best[vi] / best[0],
                    differ[vi],
                );
            }
        }
    }
    anyhow::ensure!(bad == 0, "{bad} tiles differ from the picked tile's bits");
    println!("all bits equal to the picked tile's");
    Ok(())
}
