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
    /// --replay with sets of this many sequences, each keeping 0, 1, 3, 6 or 8 tokens (one or a
    /// few users, #689), instead of 16 slots with mixed counts.
    #[arg(long, requires = "replay")]
    replay_seqs: Option<usize>,
    /// --replay's verify tokens per sequence, comma separated (default 4 and 1).
    #[arg(long, value_delimiter = ',', requires = "replay", value_parser = clap::value_parser!(i64).range(1..=9))]
    replay_verify: Vec<i64>,
    /// Prompt groups (#1183): ggml's op against the chunked kernels of `--gdn-mma` (whole heads,
    /// value slices) and the staged ones (round 8's, and the split layouts: two CTAs a head, 16
    /// warps a head with one or two input buffers, the default "wide x2", and 8 warps) at one
    /// GPU's head counts, for bits (rows and final states, for each of --tokens and a
    /// two-sequence group; the largest relative difference where bits differ) and time per
    /// layer (the first of --tokens, at most 8 layers).
    #[arg(long, conflicts_with_all = ["commit", "replay"])]
    prompt: bool,
    /// Tokens of the prompt groups (--prompt), each at least 64 (the chunked kernels' floor).
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [4096, 1030],
        value_parser = clap::value_parser!(i64).range(64..)
    )]
    tokens: Vec<i64>,
    /// A layer's BF16 alpha and beta gate projections of 1-8 tokens (#689): ggml's two
    /// `mul_mat_vec_f` launches against the one-launch op, for bits and time per layer (--layers
    /// layers, row length --ab-k, 48 rows each).
    #[arg(long, conflicts_with_all = ["commit", "replay", "prompt"])]
    ab: bool,
    /// Row lengths of --ab, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [5120])]
    ab_k: Vec<usize>,
}

#[cfg(feature = "cuda")]
fn run_ab(gpu: &mut legion::ggml::Backend, args: &GdnBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::mvf_bf16::bench::{self, Config};
    let cfg = Config {
        ks: args.ab_k.clone(),
        layers: args.layers,
        rounds: args.rounds,
        ..Config::default()
    };
    let rows = bench::run(gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>6} {:>4} {:<6} {:>9} {:>7} {:>7} {:>8}",
        "k", "tok", "kernel", "µs/layer", "vs ggml", "differ", "outputs"
    );
    let (mut reference, mut differ) = (0.0, 0);
    for r in &rows {
        if r.variant == "ggml" {
            reference = r.secs;
        }
        differ += r.differ;
        println!(
            "{:>6} {:>4} {:<6} {:>9.2} {:>7.3} {:>7} {:>8}",
            r.k,
            r.tokens,
            r.variant,
            r.secs * 1e6,
            r.secs / reference,
            r.differ,
            r.outputs
        );
    }
    anyhow::ensure!(differ == 0, "{differ} values differ from ggml's bits");
    println!("all bits equal to ggml's mul_mat_vec_f");
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn run(args: GdnBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::gdn_bench::{
        self as bench, Config, PROMPT_KERNELS, REPLAY_VARIANTS, VARIANTS,
    };
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    if args.ab {
        return run_ab(&mut gpu, &args);
    }
    anyhow::ensure!(args.replay_seqs != Some(0), "--replay-seqs takes 1 or more");
    let cfg = Config {
        seqs: args.seqs,
        rounds: args.rounds,
        // A prompt layer's rows are ~100 MB at 4096 tokens.
        layers: if args.prompt {
            args.layers.min(8)
        } else {
            args.layers
        },
        replay_seqs: args.replay_seqs,
        replay_verify: args.replay_verify,
        ..Config::default()
    };
    if args.prompt {
        let rows = bench::run_prompt(&mut gpu, &cfg, &args.tokens).map_err(anyhow::Error::msg)?;
        print!("{:>6} {:>4} {:>12}", "tokens", "seqs", "ggml µs/l");
        for name in PROMPT_KERNELS {
            print!(" {:>24}", format!("{name} µs/l ratio differ"));
        }
        println!();
        let mut differ = 0;
        for r in &rows {
            print!("{:>6} {:>4} {:>12.1}", r.tokens, r.seqs, r.secs[0] * 1e6);
            for (i, d) in r.differ.iter().enumerate() {
                let s = r.secs[i + 1];
                let rel = if *d > 0 {
                    format!(" ({:.1e})", r.rel[i])
                } else {
                    String::new()
                };
                print!(
                    " {:>24}",
                    format!("{:.1} {:.3} {d}{rel}", s * 1e6, s / r.secs[0].max(1e-12))
                );
            }
            println!();
            differ += r.differ.iter().sum::<usize>();
        }
        if differ > 0 {
            anyhow::bail!("{differ} values differ from ggml's bits");
        }
        return Ok(());
    }
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
