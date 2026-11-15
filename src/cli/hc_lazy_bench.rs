//! `legion hc-lazy-bench`: qwen4exp's prompt-step layers with the FFN mixer norms' streams
//! written (main) and made by their readers (lazy streams, #1183), timed on one GPU, and every
//! output compared bit for bit (a development tool, so a build tree's release binary can measure
//! the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`). Fails on any differing value,
//! an unread view node (it holds memory to the graph's end) or a layer not made lazily.

use clap::Args;
use legion::model::qwen4::hc_lazy_bench::{self as bench, Config};

#[derive(Args)]
pub struct HcLazyBenchArgs {
    /// Token counts of the step, comma separated (default 2048, 4096 and 8192).
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<i64>,
    /// Layers a graph chains.
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Timed rounds (the best counts; the two graphs take turns within a round).
    #[arg(long, default_value_t = 8)]
    rounds: usize,
}

pub fn run(args: HcLazyBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        layers: args.layers,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.tokens.is_empty() {
        cfg.tokens = args.tokens.clone();
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:<5} {:>9} {:>8} {:>5} {:>7} {:>10} {:>6} {:>10} {:>10}",
        "tokens",
        "graph",
        "µs/layer",
        "vs main",
        "lazy",
        "compact",
        "buffer MiB",
        "unread",
        "values",
        "differing"
    );
    let mut bad = 0;
    for r in &rows {
        let main = rows
            .iter()
            .find(|m| m.tokens == r.tokens && !m.lazy)
            .map_or(f64::NAN, |m| m.secs);
        println!(
            "{:>6} {:<5} {:>9.1} {:>8.3} {:>5} {:>7} {:>10.1} {:>6} {:>10} {:>10}",
            r.tokens,
            if r.lazy { "lazy" } else { "main" },
            r.secs * 1e6,
            r.secs / main,
            r.made,
            r.compact,
            r.buffer as f64 / (1 << 20) as f64,
            r.unread,
            r.values,
            r.differing
        );
        bad += usize::from(r.differing > 0 || r.unread > 0 || (r.lazy && r.made != cfg.layers));
    }
    anyhow::ensure!(
        bad == 0,
        "{bad} graphs differ from main's, hold unread views or were not made lazily"
    );
    println!("lazy streams: every output equals main's bit for bit");
    Ok(())
}
