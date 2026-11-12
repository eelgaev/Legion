//! `legion hc-down-bench`: qwen4exp's mixer combine-and-norm and down matmul of a prompt step
//! with BF16 streams (#1183): main's path (the norm writes the BF16 xn, cuBLAS reads it) against
//! the down matmul that rebuilds xn from the streams, timed on one GPU (a development tool, so a
//! build tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`). `--check` compares the rebuilt values with the norm's xn instead.

use clap::Args;
use legion::model::qwen4::hc_down::MIN_TOKENS;
use legion::model::qwen4::hc_down::bench::{self, Config};

#[derive(Args)]
pub struct HcDownBenchArgs {
    /// Token counts of the step, comma separated (default 4096 and 3971).
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<i64>,
    /// Mixers per timed graph, chained, each with a down weight of its own.
    #[arg(long, default_value_t = 8)]
    copies: usize,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Compare the rebuilt down matmul's B operand with the norm's BF16 xn (one-hot weights,
    /// every token tile) and its output with cuBLAS's instead of timing; fails on any differing
    /// value.
    #[arg(long)]
    check: bool,
}

pub fn run(args: HcDownBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        copies: args.copies,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.tokens.is_empty() {
        cfg.tokens = args.tokens.clone();
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>6} {:<10} {:>10} {:>10} {:>10}",
            "tokens", "what", "values", "differing", "rel"
        );
        for c in &checks {
            println!(
                "{:>6} {:<10} {:>10} {:>10} {:>10.2e}",
                c.tokens, c.what, c.values, c.differing, c.rel
            );
        }
        // Below MIN_TOKENS cuBLAS may split k, so only its rel counts there.
        let bad = checks
            .iter()
            .filter(|c| c.differing > 0 && (c.what != "vs_cublas" || c.tokens >= MIN_TOKENS))
            .count();
        anyhow::ensure!(bad == 0, "{bad} checks differ");
        println!("rebuilt values equal the BF16 xn; streams, scales and inject unmoved");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:<9} {:>9} {:>8}",
        "tokens", "variant", "µs", "vs main"
    );
    for n in &cfg.tokens {
        let of = |name: &str| {
            rows.iter()
                .find(|r| r.tokens == *n && r.variant == name)
                .map_or(f64::NAN, |r| r.secs)
        };
        let main = of("main");
        for r in rows.iter().filter(|r| r.tokens == *n) {
            println!(
                "{:>6} {:<9} {:>9.1} {:>8.3}",
                r.tokens,
                r.variant,
                r.secs * 1e6,
                r.secs / main
            );
        }
        // The GEMMs alone: each path less its norm.
        let cublas = main - of("norm_xnb");
        println!("{n:>6} gemm cuBLAS {:>9.1} µs", cublas * 1e6);
        for v in ["rebuilt", "nt2", "nt3", "nt4"] {
            let g = of(v) - of("norm");
            println!(
                "{n:>6} gemm {v:<6} {:>9.1} µs {:>8.3} of cuBLAS",
                g * 1e6,
                g / cublas
            );
        }
        println!(
            "{n:>6} norm xnb bytes {:>9.1} µs",
            (of("norm_xnb") - of("norm")) * 1e6
        );
    }
    Ok(())
}
