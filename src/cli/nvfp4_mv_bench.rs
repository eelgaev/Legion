//! `legion nvfp4-mv-bench`: the decode expert FFN of NVFP4 checkpoints in the grouped mv
//! kernel's variants, checked against the unfused chain's bits and timed on one GPU (#836; a
//! development tool, so a build tree's release binary can measure the box it runs on; pick the
//! GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::nvfp4_mm::bench::{self, Config};

#[derive(Args)]
pub struct Nvfp4MvBenchArgs {
    /// Tokens per step, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [16])]
    tokens: Vec<usize>,
    /// Each layer's tokens pick their experts from this many, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [128, 512])]
    pools: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Layers a graph holds (the same weights, other experts per layer).
    #[arg(long, default_value_t = 8)]
    layers: usize,
    /// Copies of the experts' weights, layer l reading copy l % copies (48 with --layers 48:
    /// a served Flash-Next's 68 GB of experts, for effects of where the weights lie).
    #[arg(long, default_value_t = 1)]
    copies: usize,
    /// Also time a pure read of the same experts' bytes (no math): the ceiling of the stream.
    #[arg(long)]
    probe: bool,
    /// Only these variants, comma separated (the reference chain always runs).
    #[arg(long, value_delimiter = ',')]
    variants: Vec<String>,
    /// Served routings instead of random pools: a `legion serve --moe-route-record` file. Each
    /// step size of --tokens takes that file's routings of that many tokens, in order.
    #[arg(long, value_name = "FILE")]
    routing: Option<std::path::PathBuf>,
    /// With --routing: the records of layers below this (the model's layers; an MTP head's
    /// records come after them).
    #[arg(long, default_value_t = 48)]
    routing_layers: usize,
    /// Dense weights instead of the expert FFN: `K:M` shapes, comma separated (Gemma 4 31B's
    /// `5376:8192,5376:4096,8192:5376,5376:21504,21504:5376`), each times every --tokens in the
    /// mv kernel's variants, checked against the default kernel's bits.
    #[arg(long, value_delimiter = ',', value_name = "K:M")]
    dense: Vec<String>,
    /// With --dense: the SMs the default kernel is picked for.
    #[arg(long, default_value_t = 48)]
    sms: i64,
}

fn run_dense(args: &Nvfp4MvBenchArgs) -> anyhow::Result<()> {
    use legion::model::nvfp4_mm::{dense_bench, dense_mv_max_n};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let shapes = args
        .dense
        .iter()
        .map(|s| {
            let (k, m) = s
                .split_once(':')
                .ok_or_else(|| anyhow::anyhow!("--dense {s}: not K:M"))?;
            Ok((k.parse()?, m.parse()?))
        })
        .collect::<anyhow::Result<Vec<(usize, usize)>>>()?;
    let rows = dense_bench::run(
        &mut gpu,
        &shapes,
        &args.tokens,
        args.layers,
        args.rounds,
        10,
        args.sms,
        dense_mv_max_n(),
    )
    .map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:>6} {:>3}  {:<36} {:>8} {:>7} {:>7} {:>7}",
        "k", "m", "n", "kernel", "µs", "GB/s", "vs def", "differ"
    );
    let mut default = 0.0;
    for r in &rows {
        let name = format!("{}{:?}", if r.default { "default " } else { "" }, r.kernel);
        let Some(s) = r.secs else {
            println!(
                "{:>6} {:>6} {:>3}  {:<36} {:>8}",
                r.k, r.m, r.n, name, "no kernel"
            );
            continue;
        };
        if r.default {
            default = s;
        }
        println!(
            "{:>6} {:>6} {:>3}  {:<36} {:>8.1} {:>7.1} {:>7.3} {:>7}",
            r.k,
            r.m,
            r.n,
            name,
            s * 1e6,
            r.bytes / s / 1e9,
            s / default,
            r.differ
        );
    }
    Ok(())
}

pub fn run(args: Nvfp4MvBenchArgs) -> anyhow::Result<()> {
    if !args.dense.is_empty() {
        return run_dense(&args);
    }
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let defaults = Config::default();
    let routing = match &args.routing {
        Some(p) => {
            bench::recorded_routings(&std::fs::read(p)?, args.routing_layers, defaults.n_used)
        }
        None => Vec::new(),
    };
    let cfg = Config {
        toks: args.tokens,
        pools: args.pools,
        rounds: args.rounds,
        layers: args.layers,
        copies: args.copies,
        probe: args.probe,
        only: args.variants,
        routing,
        ..defaults
    };
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<13} {:>6} {:>5} {:>7} {:>9} {:>7} {:>7} {:>7}",
        "variant", "tokens", "pool", "experts", "µs/layer", "GB/s", "vs ref", "differ"
    );
    let mut differ = 0;
    let mut reference = 0.0;
    for r in &rows {
        if r.variant == bench::VARIANTS[0].name {
            reference = r.secs;
            println!(
                "# {} tokens, pool {}: {:.1} experts, token tiles of 8/16/32: {:.1}/{:.1}/{:.1}",
                r.n_tok, r.pool, r.experts, r.tiles[0], r.tiles[1], r.tiles[2]
            );
        }
        let Some(d) = r.differ else {
            println!(
                "{:<13} {:>6} {:>5} {:>7.1} {:>9} {:>7} {:>7} {:>7}",
                r.variant, r.n_tok, r.pool, r.experts, "-", "-", "-", "no kernel"
            );
            continue;
        };
        println!(
            "{:<13} {:>6} {:>5} {:>7.1} {:>9.1} {:>7.1} {:>7.3} {:>7}",
            r.variant,
            r.n_tok,
            r.pool,
            r.experts,
            r.secs * 1e6,
            r.bytes / r.secs / 1e9,
            r.secs / reference,
            d
        );
        differ += d;
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the chain's bits");
    }
    Ok(())
}
