//! `legion dense-mv-bench`: the decode BF16 matmuls (weight groups that read one input) as
//! ggml's `mul_mat`, one op per weight and one op per group, in the kernel's variants, timed on
//! one GPU (#836; a development tool, so a build tree's release binary can measure the box it
//! runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`). The bits: `--check`, or the ignored GPU
//! test `dense_mv::tests::cuda_matches_ggml_bits`.

use clap::Args;
use legion::model::dense_mv::bench::{self, Config, Group};

#[derive(Args)]
pub struct DenseMvBenchArgs {
    /// Weight groups `k:m1+m2+...` (each `[k, m_i]`, read with one input), comma separated;
    /// `f32:k:m...` for F32 weights; default: Qwen3.8-Flash-Next's BF16 matmuls of an NVFP4
    /// checkpoint and its F32 MoE router.
    #[arg(long, value_delimiter = ',', value_parser = Group::parse)]
    groups: Vec<Group>,
    /// Tokens per step, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [4, 16])]
    tokens: Vec<usize>,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Compare the op's bits with ggml's `mul_mat` in every kernel variant instead of timing
    /// (17-48 tokens: wide chains, per weight shape); fails on any differing output.
    #[arg(long)]
    check: bool,
    /// Steps of 17-48 tokens: try wide chains in many summation orders (slices of k, several
    /// accumulators) against ggml's `mul_mat` and print the orders with its bits, per weight
    /// shape and step (how cuBLAS's kernel sums).
    #[arg(long, conflicts_with = "check")]
    wide_probe: bool,
}

pub fn run(args: DenseMvBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        toks: args.tokens,
        rounds: args.rounds,
        ..Config::default()
    };
    if !args.groups.is_empty() {
        cfg.groups = args.groups;
    }
    if args.wide_probe {
        println!(
            "{:<14} {:>6} {:>8}  orders with ggml's bits (slice/accumulators/group)",
            "wide shape", "tokens", "fewest"
        );
        for (shape, n, hits, fewest) in
            bench::probe_wide(&mut gpu, &cfg).map_err(anyhow::Error::msg)?
        {
            let hits: Vec<String> = hits
                .iter()
                .map(|o| {
                    format!(
                        "{}/{}/{}/k{}/r{}",
                        o.kslice, o.kinter, o.kgroup, o.kmma, o.kred
                    )
                })
                .collect();
            println!("{shape:<14} {n:>6} {fewest:>8}  {}", hits.join(" "));
        }
        return Ok(());
    }
    if args.check {
        let checks = bench::check(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        // Steps of 17-48 tokens: wide chains per weight shape (#836).
        let wide = bench::check_wide(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        if !wide.is_empty() {
            println!(
                "{:<14} {:>6} {:>10} {:>9} {:>9} {:>7}",
                "wide shape", "tokens", "differing", "ggml µs", "wide µs", "ratio"
            );
        }
        for c in &wide {
            println!(
                "{:<14} {:>6} {:>10} {:>9.1} {:>9.1} {:>7.3}",
                c.shape,
                c.n_tok,
                c.differing,
                c.ggml_secs * 1e6,
                c.op_secs * 1e6,
                c.op_secs / c.ggml_secs
            );
        }
        let wide_bad = wide.iter().filter(|c| c.differing > 0).count();
        println!(
            "{:<20} {:>6} {:>9} {:>10}",
            "group", "tokens", "variants", "differing"
        );
        for c in &checks {
            println!(
                "{:<20} {:>6} {:>9} {:>10}",
                c.group, c.n_tok, c.variants, c.differing
            );
        }
        let bad = checks
            .iter()
            .filter(|c| c.differing > 0 || c.variants == 0)
            .count()
            + wide_bad;
        anyhow::ensure!(bad == 0, "{bad} cases differ from ggml's bits");
        println!("all bits equal to ggml's mul_mat");
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<20} {:>6} {:<10} {:>9} {:>7} {:>7}",
        "group", "tokens", "variant", "µs", "GB/s", "vs ggml"
    );
    let mut reference = 0.0;
    for r in &rows {
        if r.variant == bench::VARIANTS[0].name {
            reference = r.secs;
        }
        println!(
            "{:<20} {:>6} {:<10} {:>9.1} {:>7.1} {:>7.3}",
            r.group,
            r.n_tok,
            r.variant,
            r.secs * 1e6,
            r.bytes / r.secs / 1e9,
            r.secs / reference,
        );
    }
    Ok(())
}
