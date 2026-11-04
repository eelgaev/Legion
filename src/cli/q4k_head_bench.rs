//! `legion q4k-head-bench`: the Q4_K MTP draft LM head of 1-8 columns as ggml's `mul_mat` and as
//! Legion's Q4_K head matvec op in its kernel variants, timed on one GPU (#836; a development
//! tool, so a build tree's release binary can measure the GB10 it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`). `--check` compares the bits instead.

use clap::Args;
use legion::model::core::q4k_head::bench::{self, Config};

#[derive(Args)]
pub struct Q4kHeadBenchArgs {
    /// Values per row (2560).
    #[arg(long, default_value_t = 2560)]
    k: usize,
    /// Row counts of the weights, comma separated; default: the 65,536-row draft head and the
    /// whole 248,320-row head.
    #[arg(long, value_delimiter = ',')]
    rows: Vec<usize>,
    /// Column counts of the input (1-8), comma separated; default: 1 and 4.
    #[arg(long, value_delimiter = ',')]
    cols: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every kernel variant instead of timing;
    /// fails on any differing output.
    #[arg(long)]
    check: bool,
    /// The MTP draft pass's head with its pick instead (1-7 columns, `--cols`): the split
    /// argmax, soft_max and get_rows after the op against the picks from its partials
    /// (`q4k_head::head`); with `--check`, their logits, pick and probability bits in every
    /// kernel variant.
    #[arg(long)]
    pick: bool,
}

pub fn run(args: Q4kHeadBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    anyhow::ensure!(
        legion::model::core::q4k_head::device_supported(),
        "the Q4_K head op runs on a GB10 only (compute capability 12.1)"
    );
    let mut cfg = Config {
        k: args.k,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.rows.is_empty() {
        cfg.rows = args.rows;
    }
    if !args.cols.is_empty() {
        cfg.cols = args.cols;
    }
    if args.pick && args.check {
        let checks = bench::check_pick(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>8} {:>5} {:>7} {:>10} {:>6} {:>18}",
            "rows", "cols", "inputs", "differing", "ties", "pick hash"
        );
        for c in &checks {
            println!(
                "{:>8} {:>5} {:>7} {:>10} {:>6} {:>18x}",
                c.rows, c.cols, c.inputs, c.differing, c.ties, c.hash
            );
        }
        let bad = checks
            .iter()
            .filter(|c| c.differing > 0 || c.inputs == 0)
            .count();
        anyhow::ensure!(bad == 0, "{bad} weights give another pick or probability");
        println!("all picks and probabilities equal to the split argmax + soft_max + get_rows");
        return Ok(());
    }
    if args.pick {
        let rows = bench::run_pick(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>8} {:>5} {:<14} {:>9} {:>7}",
            "rows", "cols", "tail", "µs", "vs first"
        );
        let mut reference = 0.0;
        for r in &rows {
            if r.variant == bench::PICK_VARIANTS[0] {
                reference = r.secs;
            }
            println!(
                "{:>8} {:>5} {:<14} {:>9.1} {:>7.3}",
                r.rows,
                r.cols,
                r.variant,
                r.secs * 1e6,
                r.secs / reference,
            );
        }
        return Ok(());
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>8} {:>5} {:>9} {:>10} {:>10} {:>18}",
            "rows", "cols", "variants", "outputs", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>8} {:>5} {:>9} {:>10} {:>10} {:>18x}",
                c.rows, c.cols, c.variants, c.outputs, c.differing, c.hash
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
