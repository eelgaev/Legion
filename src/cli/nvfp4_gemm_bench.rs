//! `legion nvfp4-gemm-bench`: the prefill expert GEMMs of NVFP4 checkpoints (the fused
//! gate/up/SwiGLU and the down GEMM) in the GEMM's tiles, stage counts and occupancies, checked
//! against the default's bits and timed on one GPU (#824; a development tool, so a build tree's
//! release binary can measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).
//! Also the mv kernel and the GEMM at any size (#836: short prompt steps); a variant on the
//! other kernel family than the default's gives other bits, its count in parentheses.

use clap::Args;
use legion::model::nvfp4_mm::gemm_bench::{self, Config, Op};

#[derive(Args)]
pub struct Nvfp4GemmBenchArgs {
    /// Tokens per prompt chunk, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [4096])]
    tokens: Vec<usize>,
    /// Each token picks its experts from this many, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [512, 160])]
    pools: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Layers a graph holds (the same weights, other routings per layer).
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Replay a routing recorded by `legion serve --moe-route-record` instead of random pools:
    /// every MoE layer's prompt steps, as one rank of --tp-moe-ep's contiguous split computes
    /// them (its experts only, the others' pairs skipped), cut into chunks of --tokens.
    #[arg(long, value_name = "FILE")]
    route_file: Option<std::path::PathBuf>,
    /// The rank whose experts --route-file replays.
    #[arg(long, default_value_t = 1)]
    route_rank: usize,
    /// The ranks --route-file's experts are split over.
    #[arg(long, default_value_t = 2)]
    route_ranks: usize,
    /// Chunks per layer --route-file replays, spread over the recorded prompt tokens.
    #[arg(long, default_value_t = 4)]
    route_chunks: usize,
    /// Records of at least this many tokens are --route-file's prompt steps (the rest, decode,
    /// is left out).
    #[arg(long, default_value_t = 1000)]
    route_min_tokens: usize,
    /// Only the variants whose names hold one of these, comma separated (the defaults always run).
    #[arg(long, value_delimiter = ',')]
    variants: Vec<String>,
}

pub fn run(args: Nvfp4GemmBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let route = match &args.route_file {
        Some(path) => Some(
            gemm_bench::load_route(
                &std::fs::read(path)?,
                args.route_rank,
                args.route_ranks,
                args.route_min_tokens,
                args.route_chunks,
            )
            .map_err(anyhow::Error::msg)?,
        ),
        None => None,
    };
    let defaults = Config::default();
    let cfg = Config {
        toks: args.tokens,
        pools: args.pools,
        rounds: args.rounds,
        layers: args.layers,
        n_expert: route.as_ref().map_or(defaults.n_expert, |r| r.n_expert),
        n_used: route.as_ref().map_or(defaults.n_used, |r| r.n_used),
        route,
        only: args.variants,
        ..defaults
    };
    if let Some(r) = &cfg.route {
        println!(
            "route: {} layers, {}-{} prompt tokens a layer, {} experts on this rank",
            r.layers.len(),
            r.layers.iter().map(Vec::len).min().unwrap_or(0) / r.n_used,
            r.layers.iter().map(Vec::len).max().unwrap_or(0) / r.n_used,
            r.n_expert
        );
    }
    let rows = gemm_bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<20} {:>6} {:>5} {:>9} {:>7} {:>7} {:>9} {:>16}",
        "variant", "tokens", "pool", "µs/layer", "TFLOPS", "vs def", "differ", "hash"
    );
    let mut differ = 0;
    let mut reference = [0.0; 2];
    for r in &rows {
        let op = (r.op == Op::Down) as usize;
        if r.variant.ends_with("default") {
            reference[op] = r.secs;
        }
        let Some(d) = r.differ else {
            println!(
                "{:<20} {:>6} {:>5} {:>9} {:>7} {:>7} {:>9}",
                r.variant, r.n_tok, r.pool, "-", "-", "-", "no kernel"
            );
            continue;
        };
        println!(
            "{:<20} {:>6} {:>5} {:>9.1} {:>7.1} {:>7.3} {:>9} {:>16x}",
            r.variant,
            r.n_tok,
            r.pool,
            r.secs * 1e6,
            r.flops / r.secs / 1e12,
            r.secs / reference[op],
            if r.bitwise {
                d.to_string()
            } else {
                format!("({d})")
            },
            r.hash
        );
        if r.bitwise {
            differ += d;
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} output bytes differ from the defaults' bits");
    }
    Ok(())
}
