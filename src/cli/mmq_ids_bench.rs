//! `legion mmq-ids-bench`: the Q4_K / Q5_1 expert-grid op (#836) against ggml's MMQ on one GPU (a
//! development tool, so a build tree's release binary can check and time it on the box it runs
//! on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::mmq_ids::bench::{self, Config};

#[derive(Args)]
pub struct MmqIdsBenchArgs {
    /// Tokens per step, comma separated (9-63).
    #[arg(long, value_delimiter = ',', default_values_t = [9, 16, 32, 48, 63])]
    tokens: Vec<usize>,
    /// Timed rounds (the best counts; the two take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Experts per layer.
    #[arg(long, default_value_t = 512)]
    experts: usize,
    /// Copies of the expert stacks a graph holds, one op each.
    #[arg(long, default_value_t = 4)]
    layers: usize,
}

pub fn run(args: MmqIdsBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let cfg = Config {
        n_expert: args.experts,
        layers: args.layers,
        toks: args.tokens,
        rounds: args.rounds,
        ..Config::default()
    };
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    // GB/s: the distinct experts' bytes over each kernel's time (the floor is the GPU's HBM rate).
    println!(
        "{:<13} {:>6} {:>8} {:>9} {:>9} {:>9} {:>8} {:>6} {:>6} {:>6} {:>12} {:>12}",
        "case",
        "tokens",
        "MB",
        "ggml µs",
        "op µs",
        "mv µs",
        "op/ggml",
        "ggml",
        "op",
        "mv",
        "differ",
        "mv differ"
    );
    let mut differ = 0;
    for r in &rows {
        let gbs = |s: f64| r.bytes / s / 1e9;
        println!(
            "{:<13} {:>6} {:>8.1} {:>9.1} {:>9.1} {:>9.1} {:>8.3} {:>6.0} {:>6.0} {:>6.0} {:>12} {:>12}",
            r.case,
            r.n_tok,
            r.bytes / 1e6,
            r.secs[0] * 1e6,
            r.secs[1] * 1e6,
            r.secs[2] * 1e6,
            r.secs[1] / r.secs[0],
            gbs(r.secs[0]),
            gbs(r.secs[1]),
            gbs(r.secs[2]),
            format!("{}/{}", r.differ, r.values),
            format!("{}/{}", r.differ_mv, r.values)
        );
        differ += r.differ;
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from ggml's MMQ bits");
    }
    Ok(())
}
