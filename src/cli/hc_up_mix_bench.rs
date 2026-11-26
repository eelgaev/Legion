//! `legion hc-up-mix-bench`: qwen4exp's mixer up matmul and gated mix with BF16 up weights (the
//! NVFP4 checkpoints) at a prompt step's shape, in each of its kernels and `up_mix_bf16_reg`'s
//! options, timed on one GPU (#1183; a development tool, so a build tree's release binary can
//! measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`). `--check` compares
//! every variant's bits with the reference's instead; `--ab BASE` (e.g. `--ab 2:none`, the
//! kernel without options) times every variant against that one, alternating which runs first.

use clap::Args;
use legion::model::qwen4::hc_up_mix_bench::{self as bench, Config, MODES, Variant, opts_name};

#[derive(Args)]
pub struct HcUpMixBenchArgs {
    /// Token counts of the step, comma separated (default 2048, 4096, 8192 and 3971).
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<i64>,
    /// Kernels, comma separated: 1 up_mix_bf16, 0 the default, 2 up_mix_bf16_reg at any token
    /// count, 2 + 100 R its runs of R tiles a CTA (default: 1, 0, 2); each with the configured
    /// up_mix_bf16_reg options. --variants instead picks the options too.
    #[arg(long, value_delimiter = ',')]
    kernels: Vec<i32>,
    /// Variants, comma separated, instead of --kernels: KERNEL or KERNEL:OPTIONS, the options
    /// up_mix_bf16_reg's, `none` or names joined by + (pd2: the streams prefetched two tiles
    /// ahead, xs: the mix's operands staged in shared memory, ld: ldmatrix.x4, cross: the
    /// prefetch on into the next run), e.g. `2:none,2:ld,2:ld+xs`.
    #[arg(long, value_delimiter = ',', value_parser = Variant::parse)]
    variants: Vec<Variant>,
    /// What --check compares every variant with (default: 1, up_mix_bf16).
    #[arg(long, value_parser = Variant::parse)]
    reference: Option<Variant>,
    /// Time every other variant against this one instead (median and min of --rounds pairs,
    /// which of the two runs first alternating per round).
    #[arg(long, value_parser = Variant::parse)]
    ab: Option<Variant>,
    /// Modes timed, comma separated: f32 (F32 streams, F32 values and BF16 copy), bf16 (the
    /// copy only), s16 (BF16 streams), xn (xn read), pre (lazy F32 streams, F32 values and
    /// copy).
    #[arg(long, value_delimiter = ',', default_value = "bf16,f32,s16")]
    modes: Vec<String>,
    /// Mixers per timed graph, each with an up weight of its own.
    #[arg(long, default_value_t = 8)]
    copies: usize,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 8)]
    rounds: usize,
    /// Compare every variant's output bits with the reference's in every mode instead of
    /// timing; fails on any differing value.
    #[arg(long)]
    check: bool,
}

/// What ran, as `kernel` or `kernel:options` for up_mix_bf16_reg.
fn ran(kernel: i32, opts: i32) -> String {
    if kernel % 100 == 2 {
        format!("{kernel}:{}", opts_name(opts))
    } else {
        kernel.to_string()
    }
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
    if !args.variants.is_empty() {
        cfg.variants = args.variants.clone();
    } else if !args.kernels.is_empty() {
        cfg.variants = args
            .kernels
            .iter()
            .map(|&k| Variant::new(k, None))
            .collect();
    }
    if let Some(r) = args.reference {
        cfg.reference = r;
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>6} {:<5} {:>12} {:>12} {:>10} {:>10} {:>18}",
            "tokens", "mode", "variant", "ran", "values", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>6} {:<5} {:>12} {:>12} {:>10} {:>10} {:>18x}",
                c.tokens,
                c.mode,
                c.variant.to_string(),
                ran(c.ran, c.ran_opts),
                c.values,
                c.differing,
                c.hash
            );
        }
        // A forced variant that fell back to another one was not checked at all.
        let skipped = checks
            .iter()
            .filter(|c| !c.variant.ran_as(c.ran, c.ran_opts))
            .count();
        anyhow::ensure!(
            skipped == 0,
            "{skipped} checks ran another kernel or options than the ones forced (rank must be \
             320 for kernel 2)"
        );
        let bad = checks.iter().filter(|c| c.differing > 0).count();
        anyhow::ensure!(
            bad == 0,
            "{bad} checks differ from {}'s bits",
            cfg.reference
        );
        println!("all variants' bits equal {}'s", cfg.reference);
        return Ok(());
    }
    let mut modes = Vec::new();
    for m in &args.modes {
        let Some(&mode) = MODES.iter().find(|x| **x == m.as_str()) else {
            anyhow::bail!("--modes takes {MODES:?}, not {m}");
        };
        modes.push(mode);
    }
    if let Some(base) = args.ab {
        if !cfg.variants.contains(&base) {
            cfg.variants.insert(0, base);
        }
        let rows = bench::ab(&mut gpu, &cfg, &modes, base).map_err(anyhow::Error::msg)?;
        println!(
            "{:>6} {:<5} {:>12} {:>12} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7}",
            "tokens",
            "mode",
            "base",
            "variant",
            "base med",
            "var med",
            "base min",
            "var min",
            "GB/s",
            "speedup",
            "min sp"
        );
        for r in &rows {
            println!(
                "{:>6} {:<5} {:>12} {:>12} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.1} {:>7.3} {:>7.3}",
                r.tokens,
                r.mode,
                ran(r.ran[0].0, r.ran[0].1),
                ran(r.ran[1].0, r.ran[1].1),
                r.median[0] * 1e6,
                r.median[1] * 1e6,
                r.min[0] * 1e6,
                r.min[1] * 1e6,
                r.bytes / r.median[1] / 1e9,
                r.speedup(),
                r.speedup_min(),
            );
        }
        let wrong = rows
            .iter()
            .filter(|r| {
                !r.base.ran_as(r.ran[0].0, r.ran[0].1) || !r.variant.ran_as(r.ran[1].0, r.ran[1].1)
            })
            .count();
        anyhow::ensure!(
            wrong == 0,
            "{wrong} pairs ran another kernel or options than the ones forced"
        );
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg, &modes).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:<5} {:>12} {:>12} {:>9} {:>7} {:>8}",
        "tokens", "mode", "variant", "ran", "µs", "GB/s", "vs first"
    );
    for n in &cfg.tokens {
        for &mode in &modes {
            let mut here = rows.iter().filter(|r| r.tokens == *n && r.mode == mode);
            let reference = here.clone().next().map_or(f64::NAN, |r| r.secs);
            for r in &mut here {
                println!(
                    "{:>6} {:<5} {:>12} {:>12} {:>9.1} {:>7.1} {:>8.3}",
                    r.tokens,
                    r.mode,
                    r.variant.to_string(),
                    ran(r.ran, r.ran_opts),
                    r.secs * 1e6,
                    r.bytes / r.secs / 1e9,
                    r.secs / reference,
                );
            }
        }
    }
    Ok(())
}
