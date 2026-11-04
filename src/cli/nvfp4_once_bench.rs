//! `legion nvfp4-once-bench`: a verify step's NVFP4 expert FFN (Qwen3.8-Flash-Next's shapes) on
//! ggml's `mul_mat_id` and on Legion's read-each-expert-once op in its tiles, timed on one GPU
//! with a bits check against ggml (#836; a development tool, so a build tree's release binary can
//! measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::model::core::nvfp4_once::bench::{self, Config};

#[derive(Args)]
pub struct Nvfp4OnceBenchArgs {
    /// Tokens per step, comma separated (2-8).
    #[arg(long, value_delimiter = ',')]
    toks: Vec<usize>,
    /// Pools the tokens pick their 10 experts from, comma separated (10: every token the same
    /// experts).
    #[arg(long, value_delimiter = ',')]
    pools: Vec<usize>,
    /// Layers, each with its own expert stack.
    #[arg(long, default_value_t = 8)]
    layers: usize,
    /// Experts per layer stack.
    #[arg(long, default_value_t = 128)]
    experts: usize,
    /// Timed rounds (the best counts; the variants take turns within a round).
    #[arg(long, default_value_t = 5)]
    rounds: usize,
    /// The matmuls alone, without the per-expert scale chain after each (served graphs have it).
    #[arg(long)]
    no_scale: bool,
    /// Fill the input once after the allocation instead of before every run (the bench's old
    /// behaviour, which read a stale token 0 after the first run with the scale chain, #836).
    #[arg(long)]
    input_once: bool,
    /// Set the ids as an input instead of picking them with the model's router top-k from
    /// logits (by default both graphs route as served, and the ops take the routing plan the
    /// top-k kernel writes, #836).
    #[arg(long)]
    no_router: bool,
}

pub fn run(args: Nvfp4OnceBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        layers: args.layers,
        n_expert: args.experts,
        rounds: args.rounds,
        scale: !args.no_scale,
        input_once: args.input_once,
        router: !args.no_router,
        ..Config::default()
    };
    if !args.toks.is_empty() {
        cfg.toks = args.toks;
    }
    if !args.pools.is_empty() {
        cfg.pools = args.pools;
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:>4} {:>5} {:>7} {:<12} {:>9} {:>10} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "toks",
        "pool",
        "experts",
        "variant",
        "us/layer",
        "ms/48 lay",
        "vs ggml",
        "differ",
        "outputs",
        "ref err",
        "warm err"
    );
    let mut reference = 0.0;
    let mut bad = 0;
    for r in &rows {
        if r.variant == "ggml" {
            reference = r.secs;
        }
        bad += usize::from(r.differ > 0);
        println!(
            "{:>4} {:>5} {:>7.1} {:<12} {:>9.1} {:>10.2} {:>7.3} {:>9} {:>9} {:>9.2e} {:>9}",
            r.n_tok,
            r.pool,
            r.experts,
            r.variant,
            r.secs * 1e6,
            r.secs * 48e3,
            r.secs / reference,
            r.differ,
            r.outputs,
            r.ref_err,
            r.warm_ref_err
                .map_or("-".to_string(), |e| format!("{e:.2e}")),
        );
    }
    // The nodes ggml's allocator placed over the input (once per step size and graph).
    for r in &rows {
        if r.variant == "ggml" || r.variant == bench::variant_names()[1] {
            let graph = if r.variant == "ggml" { "ggml" } else { "once" };
            let over = if r.over_input.is_empty() {
                "none".to_string()
            } else {
                r.over_input.join("; ")
            };
            println!(
                "toks {} pool {} {graph} graph, nodes over the input: {over}",
                r.n_tok, r.pool
            );
        }
    }
    anyhow::ensure!(bad == 0, "{bad} variants differ from ggml's bits");
    println!("all bits equal to ggml's mul_mat_id");
    Ok(())
}
