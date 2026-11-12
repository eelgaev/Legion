//! `legion hc-up-mix-bench`: qwen4exp's mixer up matmul and gated mix with BF16 up weights (the
//! NVFP4 checkpoints) at a prompt step's shape, in each of its kernels, timed on one GPU (#1183;
//! a development tool, so a build tree's release binary can measure the box it runs on; pick the
//! GPU with `CUDA_VISIBLE_DEVICES`). `--check` compares every kernel's bits with `up_mix_bf16`'s
//! instead.

use clap::Args;
use legion::model::qwen4::hc_up_mix_bench::{self as bench, Config, MODES};

#[derive(Args)]
pub struct HcUpMixBenchArgs {
    /// Token counts of the step, comma separated (default 2048, 4096, 8192 and 3971).
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<i64>,
    /// Kernels, comma separated: 1 up_mix_bf16, 0 the default, 2 up_mix_bf16_reg at any token
    /// count, 2 + 100 R its runs of R tiles a CTA (default: 1, 0, 2).
    #[arg(long, value_delimiter = ',')]
    kernels: Vec<i32>,
    /// Modes timed, comma separated: f32 (F32 streams, F32 values and BF16 copy), bf16 (the
    /// copy only), s16 (BF16 streams), xn (xn read).
    #[arg(long, value_delimiter = ',', default_value = "bf16,f32,s16")]
    modes: Vec<String>,
    /// Mixers per timed graph, each with an up weight of its own.
    #[arg(long, default_value_t = 8)]
    copies: usize,
    /// Timed rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 8)]
    rounds: usize,
    /// Compare every kernel's output bits with up_mix_bf16's in every mode instead of timing;
    /// fails on any differing value.
    #[arg(long)]
    check: bool,
}

pub fn run(args: HcUpMixBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        copies: args.copies,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.tokens.is_empty() {
        cfg.tokens = args.tokens.clone();
    }
    if !args.kernels.is_empty() {
        cfg.kernels = args.kernels.clone();
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>6} {:<5} {:>6} {:>4} {:>10} {:>10} {:>18}",
            "tokens", "mode", "kernel", "ran", "values", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>6} {:<5} {:>6} {:>4} {:>10} {:>10} {:>18x}",
                c.tokens, c.mode, c.kernel, c.ran, c.values, c.differing, c.hash
            );
        }
        // A forced kernel that fell back to another one was not checked at all.
        let skipped = checks
            .iter()
            .filter(|c| c.kernel != 0 && c.ran != c.kernel)
            .count();
        anyhow::ensure!(
            skipped == 0,
            "{skipped} checks ran another kernel than the one forced (rank must be 320 for kernel 2)"
        );
        let bad = checks.iter().filter(|c| c.differing > 0).count();
        anyhow::ensure!(bad == 0, "{bad} checks differ from up_mix_bf16's bits");
        println!("all kernels' bits equal up_mix_bf16's");
        return Ok(());
    }
    let mut modes = Vec::new();
    for m in &args.modes {
        let Some(&mode) = MODES.iter().find(|x| **x == m.as_str()) else {
            anyhow::bail!("--modes takes {MODES:?}, not {m}");
        };
        modes.push(mode);
    }
    let rows = bench::run(&mut gpu, &cfg, &modes).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:<5} {:>6} {:>4} {:>9} {:>7} {:>8}",
        "tokens", "mode", "kernel", "ran", "µs", "GB/s", "vs k1"
    );
    for n in &cfg.tokens {
        for &mode in &modes {
            let of = |k: i32| {
                rows.iter()
                    .find(|r| r.tokens == *n && r.mode == mode && r.kernel == k)
                    .map_or(f64::NAN, |r| r.secs)
            };
            let reference = of(1);
            for r in rows.iter().filter(|r| r.tokens == *n && r.mode == mode) {
                println!(
                    "{:>6} {:<5} {:>6} {:>4} {:>9.1} {:>7.1} {:>8.3}",
                    r.tokens,
                    r.mode,
                    r.kernel,
                    r.ran,
                    r.secs * 1e6,
                    r.bytes / r.secs / 1e9,
                    r.secs / reference,
                );
            }
        }
    }
    Ok(())
}
