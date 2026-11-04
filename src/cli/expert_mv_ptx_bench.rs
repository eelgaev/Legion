//! `legion expert-mv-ptx-bench`: the decode expert matvec's PTX kernels (`--expert-mv-ptx`,
//! #703) against the nvcc kernels on one GPU (a development tool, so a build tree's release
//! binary can measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::expert_mv::ptx::bench::{self, Config, MODES};

#[derive(Args)]
pub struct ExpertMvPtxBenchArgs {
    /// Tokens per step, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 3, 4])]
    tokens: Vec<usize>,
    /// Timed rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Experts per layer (all of them in cache slots, and a second run with half).
    #[arg(long, default_value_t = 256)]
    experts: usize,
    /// Copies of the expert stacks a graph holds, one op each.
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Only the runs with every expert in cache slots.
    #[arg(long)]
    no_mixed: bool,
    /// Tiles per CTA of the "pipe" column (the tile kernel with `--expert-mv-pipe`, #816);
    /// without it, the default choice per op and step size ("tile" is a tile per CTA).
    #[arg(long)]
    pipe_tiles: Option<usize>,
    /// The Q4_K gate/up kernel's per-unit activations (`--expert-mv-q4k-units`, #816) in every
    /// mode; the "pipe" column at 3-4 tokens is the kernel that reads them.
    #[arg(long)]
    q4k_units: bool,
    /// The "pipe" kernel's Q4_K gate/up without its activation hold in registers (`legion serve
    /// --no-expert-mv-q4k-hold`, #816 rounds 6 and 8; held by default).
    #[arg(long)]
    no_q4k_hold: bool,
    /// The "pipe" kernel with a CTA per group of tiles, not its tiles balanced over the resident
    /// CTAs (`legion serve --no-expert-mv-balance`, #814; balanced by default).
    #[arg(long)]
    no_balance: bool,
    /// The "pipe" kernel's Q5_1 down at 2-3 tokens on its full tiles, not tiles of half the rows
    /// (`legion serve --no-expert-mv-half-tiles`, #814; half tiles by default).
    #[arg(long)]
    no_half_tiles: bool,
}

pub fn run(args: ExpertMvPtxBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    legion::model::core::expert_mv::set_q4k_hold(!args.no_q4k_hold);
    legion::model::core::expert_mv::set_balance(!args.no_balance);
    legion::model::core::expert_mv::set_half_tiles(!args.no_half_tiles);
    let cfg = Config {
        n_expert: args.experts,
        layers: args.layers,
        toks: args.tokens,
        rounds: args.rounds,
        mixed: !args.no_mixed,
        pipe_tiles: args.pipe_tiles,
        q4k_units: args.q4k_units,
        ..Config::default()
    };
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<13} {:>6} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8} {:>16}",
        "case",
        "tokens",
        "in vram",
        "tile µs",
        "pf µs",
        "ptx µs",
        "side µs",
        "pipe µs",
        "pf/t",
        "ptx/t",
        "side/t",
        "pipe/t",
        "pf≠",
        "ptx≠",
        "side≠",
        "pipe≠",
        "ref err",
        "bits hash"
    );
    let mut differ = 0;
    for r in &rows {
        println!(
            "{:<13} {:>6} {:>8} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>6} {:>6} {:>6} {:>6} {:>8.1e} {:>16x}",
            r.case,
            r.n_tok,
            format!("{}/{}", r.vram_below, r.n_expert),
            r.secs[0] * 1e6,
            r.secs[1] * 1e6,
            r.secs[2] * 1e6,
            r.secs[3] * 1e6,
            r.secs[4] * 1e6,
            r.secs[1] / r.secs[0],
            r.secs[2] / r.secs[0],
            r.secs[3] / r.secs[0],
            r.secs[4] / r.secs[0],
            r.differ[1],
            r.differ[2],
            r.differ[3],
            r.differ[4],
            r.ref_err,
            r.bits_hash
        );
        differ += r.differ[1..].iter().sum::<usize>();
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from the {} kernel's bits", MODES[0]);
    }
    Ok(())
}
