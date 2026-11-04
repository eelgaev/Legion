//! `legion q8-head-bench`: the one-column Q8_0 MTP draft LM head as ggml's `mul_mat` and as
//! Legion's one-column Q8_0 matvec op in its kernel variants, timed on one GPU (#814; a
//! development tool, so a build tree's release binary can measure the box it runs on; pick the
//! GPU with `CUDA_VISIBLE_DEVICES`). `--check` compares the bits instead.

use clap::Args;
use legion::model::core::q8_head::bench::{self, Config};

#[derive(Args)]
pub struct Q8HeadBenchArgs {
    /// Values per row (1024 or 2560; 6144 for the one-column projections of that width, an
    /// attention layer's output and a DeltaNet layer's ssm_out, default 2,560 rows, with or
    /// without `--group`).
    #[arg(long, default_value_t = 2560)]
    k: usize,
    /// Row counts of the weights, comma separated; default: the whole 248,320-row head, the
    /// 65,536-row draft head and a 10,240-row projection.
    #[arg(long, value_delimiter = ',')]
    rows: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every kernel variant instead of timing;
    /// fails on any differing output.
    #[arg(long)]
    check: bool,
    /// The MTP draft pass's head with its pick instead: the split argmax, soft_max and get_rows
    /// after the head against the pick ops (`q8_head::pick`); with `--check`, their pick and
    /// probability bits.
    #[arg(long)]
    pick: bool,
    /// The decode matvecs instead (#814): q8_group's one-column matvecs (`--rows`, 1-4 weights
    /// of an even row count; default a DeltaNet layer's 10,240-row qkv and 6,144-row gate) as
    /// ggml's `mul_mat`, as q8_group's copy of ggml's mmvq and on the head's kernel
    /// (`q8_head::row_mv`) in its variants; with `--check`, their bits against ggml's.
    #[arg(long)]
    group: bool,
    /// With `--group`: all weights in one q8_group node (an attention layer's Q, K and V)
    /// instead of a node each.
    #[arg(long)]
    joined: bool,
}

pub fn run(args: Q8HeadBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        k: args.k,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.rows.is_empty() {
        cfg.rows = args.rows.clone();
    }
    let out = args.k as i64 == legion::model::core::q8_head::K_OUT;
    if out && args.rows.is_empty() {
        cfg.rows = vec![2560];
    }
    // The 6,144-value projections are ggml matmuls, not q8_group nodes: `--group` times and
    // checks them against ggml's `mul_mat` as the head is.
    if args.group && !out {
        if cfg.rows == Config::default().rows {
            cfg.rows = vec![10240, 6144];
        }
        return group(&mut gpu, &cfg, &args);
    }
    if args.pick && args.check {
        let checks = bench::check_pick(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:>8} {:>7} {:>10} {:>6} {:>18}",
            "rows", "inputs", "differing", "ties", "pick hash"
        );
        for c in &checks {
            println!(
                "{:>8} {:>7} {:>10} {:>6} {:>18x}",
                c.rows, c.inputs, c.differing, c.ties, c.hash
            );
        }
        let bad = checks.iter().filter(|c| c.differing > 0).count();
        anyhow::ensure!(bad == 0, "{bad} weights give another pick or probability");
        println!("all picks and probabilities equal to the split argmax + soft_max + get_rows");
        return Ok(());
    }
    if args.pick {
        let rows = bench::run_pick(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!("{:>8} {:<14} {:>9} {:>7}", "rows", "tail", "µs", "vs first");
        let mut reference = 0.0;
        for r in &rows {
            if r.variant == bench::PICK_VARIANTS[0] {
                reference = r.secs;
            }
            println!(
                "{:>8} {:<14} {:>9.1} {:>7.3}",
                r.rows,
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
            "{:>8} {:>9} {:>10} {:>10} {:>18}",
            "rows", "variants", "outputs", "differing", "bits hash"
        );
        for c in &checks {
            println!(
                "{:>8} {:>9} {:>10} {:>10} {:>18x}",
                c.rows, c.variants, c.outputs, c.differing, c.hash
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
        "{:>8} {:<10} {:>9} {:>7} {:>7}",
        "rows", "variant", "µs", "GB/s", "vs ggml"
    );
    let mut reference = 0.0;
    for r in &rows {
        if r.variant == bench::VARIANTS[0].name {
            reference = r.secs;
        }
        println!(
            "{:>8} {:<10} {:>9.1} {:>7.1} {:>7.3}",
            r.rows,
            r.variant,
            r.secs * 1e6,
            r.bytes / r.secs / 1e9,
            r.secs / reference,
        );
    }
    Ok(())
}

/// `--group`: the decode matvecs, timed or (`--check`) their bits; with several weights and no
/// `--joined`, each weight its own run.
fn group(
    gpu: &mut legion::ggml::Backend,
    cfg: &Config,
    args: &Q8HeadBenchArgs,
) -> anyhow::Result<()> {
    let sets: Vec<Vec<usize>> = if args.joined {
        vec![cfg.rows.clone()]
    } else {
        cfg.rows.iter().map(|&r| vec![r]).collect()
    };
    if args.check {
        println!(
            "{:>8} {:>9} {:>10} {:>10} {:>18}",
            "rows", "variants", "outputs", "differing", "bits hash"
        );
        let mut bad = 0;
        for rows in sets {
            let c = Config {
                rows,
                ..cfg.clone()
            };
            for c in bench::check_group(gpu, &c, args.joined).map_err(anyhow::Error::msg)? {
                println!(
                    "{:>8} {:>9} {:>10} {:>10} {:>18x}",
                    c.rows, c.variants, c.outputs, c.differing, c.hash
                );
                bad += usize::from(c.differing > 0 || c.variants == 0);
            }
        }
        anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
        println!("all bits equal to ggml's mul_mat");
        return Ok(());
    }
    println!(
        "{:>8} {:<12} {:>9} {:>7} {:>7}",
        "rows", "variant", "µs", "GB/s", "vs group"
    );
    for rows in sets {
        let c = Config {
            rows,
            ..cfg.clone()
        };
        let out = bench::run_group(gpu, &c, args.joined).map_err(anyhow::Error::msg)?;
        let reference = out
            .iter()
            .find(|r| r.variant == bench::GROUP_VARIANTS[1].name)
            .map_or(f64::NAN, |r| r.secs);
        for r in &out {
            println!(
                "{:>8} {:<12} {:>9.2} {:>7.1} {:>7.3}",
                r.rows,
                r.variant,
                r.secs * 1e6,
                r.bytes / r.secs / 1e9,
                r.secs / reference,
            );
        }
    }
    Ok(())
}
