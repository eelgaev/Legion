//! `legion gdn-bench`: the gated DeltaNet decode kernel (#836) against ggml's `gated_delta_net`
//! on one GPU, every variant checked for ggml's bits and timed (a development tool, so a build
//! tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct GdnBenchArgs {
    /// Sequences per step (one token each), comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 4, 16])]
    seqs: Vec<i64>,
    /// Timed rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// Layers per graph, each with its own states.
    #[arg(long, default_value_t = 36)]
    layers: usize,
    /// Replay commits instead of decode steps: the all-slot commit (one launch per layer over
    /// 16 slots, each keeping its own count) against ggml's op per slot for bits, and against
    /// one commit launch per slot for time.
    #[arg(long)]
    commit: bool,
    /// Deferred commits (`--gdn-commit-in-verify`, #836 part 31): the verify step that replays
    /// the last one's kept tokens from the packed rows (and then stores its own there) against
    /// the commit and the verify step on their own, for bits (states and rows, and the fused
    /// pair run twice, and the packed rows the fused step stores, "d.store") and time per layer.
    #[arg(long, conflicts_with = "commit")]
    replay: bool,
}

#[cfg(feature = "cuda")]
pub fn run(args: GdnBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::gdn_bench::{self as bench, Config, REPLAY_VARIANTS, VARIANTS};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let cfg = Config {
        seqs: args.seqs,
        rounds: args.rounds,
        layers: args.layers,
        ..Config::default()
    };
    if args.replay {
        let rows = bench::run_replay(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:<50} {:>3} {:>10} {:>10} {:>7} {:>9} {:>7} {:>7} {:>7} {:>7}",
            "keep",
            "T",
            "sep µs/l",
            "fused µs/l",
            "ratio",
            "commit µs",
            "d.state",
            "d.rows",
            "d.rerun",
            "d.store"
        );
        let mut differ = 0;
        for r in &rows {
            differ += r.variants.iter().map(|v| v.1).sum::<usize>();
            println!(
                "{:<50} {:>3} {:>10.1} {:>10.1} {:>7.3} {:>9.1} {:>7} {:>7} {:>7} {:>7}",
                format!("{:?}", r.keep),
                r.verify,
                r.separate * 1e6,
                r.fused * 1e6,
                r.fused / r.separate,
                r.commit * 1e6,
                r.differ_state,
                r.differ_rows,
                r.differ_rerun,
                r.differ_store
            );
            differ += r.differ_state + r.differ_rows + r.differ_rerun + r.differ_store;
        }
        // The fused step's kernel variants: µs per layer, vs the per-lane kernel, differing values.
        println!();
        print!("{:<50} {:>3}", "keep (fused kernel variants)", "T");
        for (_, name) in REPLAY_VARIANTS {
            print!(" {:>16}", name);
        }
        println!();
        for r in &rows {
            print!("{:<50} {:>3}", format!("{:?}", r.keep), r.verify);
            for &(secs, d) in &r.variants {
                print!(
                    " {:>16}",
                    format!("{:.1} {:.3} {}", secs * 1e6, secs / r.variants[0].0, d)
                );
            }
            println!();
        }
        if differ > 0 {
            anyhow::bail!("{differ} values differ between the fused and the separate pair");
        }
        return Ok(());
    }
    if args.commit {
        let rows = bench::run_commit(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
        println!(
            "{:<50} {:>12} {:>14} {:>7} {:>7}",
            "keep", "slots µs/l", "per-slot µs/l", "ratio", "differ"
        );
        let mut differ = 0;
        for r in &rows {
            println!(
                "{:<50} {:>12.1} {:>14.1} {:>7.3} {:>7}",
                format!("{:?}", r.keep),
                r.slots * 1e6,
                r.per_slot * 1e6,
                r.slots / r.per_slot,
                r.differ
            );
            differ += r.differ;
        }
        if differ > 0 {
            anyhow::bail!("{differ} values differ from ggml's bits");
        }
        return Ok(());
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<20} {:>5} {:>10} {:>8} {:>8} {:>7}",
        "kernel", "seqs", "µs/layer", "GB/s", "vs ggml", "differ"
    );
    let mut differ = 0;
    for r in &rows {
        let line = |name: &str, s: f64, d: Option<usize>| {
            println!(
                "{:<20} {:>5} {:>10.1} {:>8.1} {:>8.3} {:>7}",
                name,
                r.seqs,
                s * 1e6,
                r.bytes / s / 1e9,
                s / r.secs[0],
                d.map_or("ref".to_string(), |d| d.to_string())
            )
        };
        line("ggml", r.secs[0], None);
        for (i, (_, name)) in VARIANTS.iter().enumerate() {
            line(name, r.secs[i + 1], Some(r.differ[i]));
        }
        differ += r.differ.iter().sum::<usize>();
    }
    if differ > 0 {
        anyhow::bail!("{differ} values differ from ggml's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: GdnBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("gdn-bench needs a CUDA build")
}
