//! `legion head-mm-bench`: the BF16 LM head of verify steps (49-128 columns) as ggml's
//! `mul_mat`, soft-cap and argmax and as Legion's head op, timed on one GPU (#1181; a
//! development tool, so a build tree's release binary can measure the box it runs on; pick the
//! GPU with `CUDA_VISIBLE_DEVICES`). `--check` compares every logit's and pick's bits instead.

use clap::Args;
use legion::model::core::head_mm::bench::{self, Config};

#[derive(Args)]
pub struct HeadMmBenchArgs {
    /// Values per row (Gemma 4 31B: 5,376).
    #[arg(long, default_value_t = 5376)]
    k: usize,
    /// Rows of the head (Gemma 4 31B: 262,144).
    #[arg(long, default_value_t = 262144)]
    m: usize,
    /// Column counts of the input (49-128), comma separated; default: 56, 64, ..., 128 (7-16
    /// sequences of 8 verified tokens).
    #[arg(long, value_delimiter = ',')]
    cols: Vec<usize>,
    /// Every column count from 49 to 128 (with `--check`).
    #[arg(long, conflicts_with = "cols")]
    all_cols: bool,
    /// Timed rounds (the best counts; ggml and the op take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// The final logit soft-cap (Gemma 4: 30; 0 for none).
    #[arg(long, default_value_t = 30.0)]
    cap: f32,
    /// Compare the op's logits and picks with ggml's instead of timing; fails on any differing
    /// bit.
    #[arg(long)]
    check: bool,
}

pub fn run(args: HeadMmBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        k: args.k,
        m: args.m,
        rounds: args.rounds,
        cap: args.cap,
        ..Config::default()
    };
    if args.all_cols {
        cfg.cols = (49..=128).collect();
    } else if !args.cols.is_empty() {
        cfg.cols = args.cols;
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>5} {:>6} {:>16} {:>15} {:>6} {:>18}",
            "cols", "cases", "logits differing", "picks differing", "ties", "pick hash"
        );
        let mut same = Vec::new();
        for c in &checks {
            println!(
                "{:>5} {:>6} {:>16} {:>15} {:>6} {:>18x}",
                c.cols, c.cases, c.logits_differing, c.picks_differing, c.ties, c.hash
            );
            if c.logits_differing == 0 && c.picks_differing == 0 && c.cases > 0 {
                same.push(c.cols);
            }
        }
        println!("cols with ggml's bits: {same:?}");
        let bad = checks.len() - same.len();
        anyhow::ensure!(bad == 0, "{bad} column counts differ from ggml's bits");
        println!("all logits and picks equal to ggml's mul_mat + soft-cap + argmax");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>5} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "cols", "ggml µs", "op µs", "ggml GB/s", "op GB/s", "ratio"
    );
    for r in &rows {
        println!(
            "{:>5} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.3}",
            r.cols,
            r.ggml_secs * 1e6,
            r.op_secs * 1e6,
            r.bytes / r.ggml_secs / 1e9,
            r.bytes / r.op_secs / 1e9,
            r.op_secs / r.ggml_secs,
        );
    }
    Ok(())
}
