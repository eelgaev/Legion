//! `legion q4k-mv-bench`: the DFlash drafter's Q4_K matmuls of 1-8 columns as ggml's `mul_mat`
//! and as Legion's Q4_K matvec op in its launch shapes, timed on one GPU (#689; a development
//! tool, so a build tree's release binary can measure the GB10 it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`). `--check` compares the bits instead.

use clap::Args;
use legion::model::core::q4k_mv::bench::{self, Config};

#[derive(Args)]
pub struct Q4kMvBenchArgs {
    /// Weights as K:ROWS (K 4096 or 5120), comma separated; default: the Qwen3.8-27B DFlash 2
    /// drafter's Q4_K matmuls.
    #[arg(long, value_delimiter = ',')]
    shapes: Vec<String>,
    /// Column counts of the input (1-8), comma separated; default: 1-8.
    #[arg(long, value_delimiter = ',')]
    cols: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every launch shape instead of timing;
    /// fails on any differing output.
    #[arg(long)]
    check: bool,
}

pub fn run(args: Q4kMvBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    anyhow::ensure!(
        legion::model::core::q4k_head::device_supported(),
        "the Q4_K matvec op runs on a GB10 only (compute capability 12.1)"
    );
    let mut cfg = Config {
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.shapes.is_empty() {
        cfg.shapes = args
            .shapes
            .iter()
            .map(|s| {
                let (k, rows) = s
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("--shapes: {s} is not K:ROWS"))?;
                Ok((k.parse()?, rows.parse()?))
            })
            .collect::<anyhow::Result<_>>()?;
    }
    if !args.cols.is_empty() {
        cfg.cols = args.cols;
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>5} {:>8} {:>5} {:>9} {:>10} {:>10} {:>18}",
            "k", "rows", "cols", "variants", "outputs", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>5} {:>8} {:>5} {:>9} {:>10} {:>10} {:>18x}",
                c.k, c.rows, c.cols, c.variants, c.outputs, c.differing, c.hash
            );
        }
        let bad = checks
            .iter()
            .filter(|c| c.differing > 0 || c.variants == 0)
            .count();
        anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
        println!("all bits equal to ggml's mul_mat");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>5} {:>8} {:>5} {:<10} {:>9} {:>7} {:>7}",
        "k", "rows", "cols", "variant", "µs", "GB/s", "vs ggml"
    );
    let mut reference = 0.0;
    for r in &rows {
        if r.variant == bench::VARIANTS[0].name {
            reference = r.secs;
        }
        println!(
            "{:>5} {:>8} {:>5} {:<10} {:>9.1} {:>7.1} {:>7.3}",
            r.k,
            r.rows,
            r.cols,
            r.variant,
            r.secs * 1e6,
            r.bytes / r.secs / 1e9,
            r.secs / reference,
        );
    }
    Ok(())
}
