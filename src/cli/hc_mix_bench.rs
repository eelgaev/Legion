//! `legion hc-mix-bench`: qwen4exp's hyper-connection mixer op at a decode or MTP verify step's
//! shape in its V100 kernel variants, timed on one GPU (#814; a development tool, so a build
//! tree's release binary can measure the box it runs on, also under `ncu`; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`). `--check` compares the bits instead.

use clap::Args;
use legion::model::qwen4::hc::bench::{self, Config};

#[derive(Args)]
pub struct HcMixBenchArgs {
    /// Token counts of the step, comma separated (1 to 8; default 1, 2 and 3: decode and MTP
    /// verify steps of k = 1 and 2).
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<i64>,
    /// Mixers per timed graph, each with weights of its own (so none is in L2 when it starts).
    #[arg(long, default_value_t = 32)]
    copies: usize,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 20)]
    rounds: usize,
    /// Compare every variant's output bytes with main's kernels' (`ahead`) instead of
    /// timing; fails on any differing byte.
    #[arg(long)]
    check: bool,
}

pub fn run(args: HcMixBenchArgs) -> anyhow::Result<()> {
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
            "{:>6} {:<8} {:>10} {:>10} {:>18}",
            "tokens", "variant", "bytes", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>6} {:<8} {:>10} {:>10} {:>18x}",
                c.tokens, c.variant, c.bytes, c.differing, c.hash
            );
        }
        let bad = checks.iter().filter(|c| c.differing > 0).count();
        anyhow::ensure!(bad == 0, "{bad} variants differ from the reference's bits");
        println!("all variants' bits equal");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:<8} {:>9} {:>7} {:>7}",
        "tokens", "variant", "µs", "GB/s", "vs ahead"
    );
    for n in &cfg.tokens {
        let of = |name: &str| {
            rows.iter()
                .find(|r| r.tokens == *n && r.variant == name)
                .map_or(f64::NAN, |r| r.secs)
        };
        let reference = of("ahead");
        for r in rows.iter().filter(|r| r.tokens == *n) {
            println!(
                "{:>6} {:<8} {:>9.2} {:>7.1} {:>7.3}",
                r.tokens,
                r.variant,
                r.secs * 1e6,
                r.bytes / r.secs / 1e9,
                r.secs / reference,
            );
        }
    }
    Ok(())
}
