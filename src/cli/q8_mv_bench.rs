//! `legion q8-mv-bench`: the Q8_0 MTP draft LM head (9-16 columns) as ggml's `mul_mat` and as
//! Legion's Q8_0 matvec op in its kernel variants, timed on one GPU (#836 part 13; a development
//! tool, so a build tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`). `--check` compares the bits instead.

use clap::Args;
use legion::model::core::q8_mv::bench::{self, Config};

#[derive(Args)]
pub struct Q8MvBenchArgs {
    /// Values per row (a multiple of 512).
    #[arg(long, default_value_t = 2560)]
    k: usize,
    /// Row counts of the weights (multiples of 32), comma separated; default: the 65,536-row
    /// draft head and the whole 248,320-row head.
    #[arg(long, value_delimiter = ',')]
    rows: Vec<usize>,
    /// Columns per step (9-16), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [9, 12, 16])]
    cols: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every kernel variant instead of timing;
    /// fails on any differing output.
    #[arg(long)]
    check: bool,
}

pub fn run(args: Q8MvBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        k: args.k,
        cols: args.cols,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.rows.is_empty() {
        cfg.rows = args.rows;
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>8} {:>5} {:>9} {:>10} {:>10}",
            "rows", "cols", "variants", "outputs", "differing"
        );
        for c in &checks {
            println!(
                "{:>8} {:>5} {:>9} {:>10} {:>10}",
                c.rows, c.cols, c.variants, c.outputs, c.differing
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
        "{:>8} {:>5} {:<10} {:>9} {:>7} {:>7}",
        "rows", "cols", "variant", "µs", "GB/s", "vs ggml"
    );
    let mut reference = 0.0;
    for r in &rows {
        if r.variant == bench::VARIANTS[0].name {
            reference = r.secs;
        }
        println!(
            "{:>8} {:>5} {:<10} {:>9.1} {:>7.1} {:>7.3}",
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
