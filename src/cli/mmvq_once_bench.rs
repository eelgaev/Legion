//! `legion mmvq-once-bench`: a decode or verify step's routed experts (Qwen3.8-Flash-Next
//! UD-Q4_K_XL's shapes and types by default) on ggml's `mul_mat_id` and on Legion's
//! stream-each-expert-once ops at their tunings, the gate + up + SwiGLU part and the down part
//! timed apart, with a bits check against ggml (#1176; a development tool, so a build tree's
//! release binary can measure the box it runs on; pick the GPU with `CUDA_VISIBLE_DEVICES`).

use clap::Args;
use legion::ggml::{ffi, types};
use legion::model::core::mmvq_once::Tuning;
use legion::model::core::mmvq_once::bench::{self, Config};

#[derive(Args)]
pub struct MmvqOnceBenchArgs {
    /// Tokens per step, comma separated (1-8).
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
    /// The model's width (gate and up `[k, m]`, down `[m, k]`).
    #[arg(long, default_value_t = 2560)]
    k: usize,
    /// The experts' width.
    #[arg(long, default_value_t = 640)]
    m: usize,
    /// Experts per token.
    #[arg(long, default_value_t = 10)]
    used: usize,
    /// The gate and up weights' type: q4_K, q5_K, q6_K, q5_1 or q8_0.
    #[arg(long, default_value = "q4_K")]
    glu_type: String,
    /// The down weights' type.
    #[arg(long, default_value = "q5_1")]
    down_type: String,
    /// Tunings to time besides the default, as STAGE_KB/WARPS, comma separated (`8/4,16/8`).
    #[arg(long, value_delimiter = ',')]
    tunings: Vec<String>,
}

fn weight_type(name: &str) -> anyhow::Result<ffi::ggml_type> {
    Ok(match name {
        "q4_K" => types::Q4_K,
        "q5_K" => types::Q5_K,
        "q6_K" => types::Q6_K,
        "q5_1" => types::Q5_1,
        "q8_0" => types::Q8_0,
        _ => anyhow::bail!("unknown weight type {name:?} (q4_K, q5_K, q6_K, q5_1, q8_0)"),
    })
}

fn tuning(s: &str) -> anyhow::Result<Tuning> {
    let (kb, warps) = s
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("tuning {s:?}: want STAGE_KB/WARPS"))?;
    Ok(Tuning {
        stage_bytes: kb.trim().parse::<usize>()? * 1024,
        warps: warps.trim().parse()?,
    })
}

pub fn run(args: MmvqOnceBenchArgs) -> anyhow::Result<()> {
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let mut cfg = Config {
        layers: args.layers,
        n_expert: args.experts,
        rounds: args.rounds,
        k: args.k,
        m: args.m,
        n_used: args.used,
        glu_ty: weight_type(&args.glu_type)?,
        down_ty: weight_type(&args.down_type)?,
        ..Config::default()
    };
    if !args.toks.is_empty() {
        cfg.toks = args.toks;
    }
    if !args.pools.is_empty() {
        cfg.pools = args.pools;
    }
    if !args.tunings.is_empty() {
        cfg.tunings = args
            .tunings
            .iter()
            .map(|s| tuning(s))
            .collect::<anyhow::Result<_>>()?;
    }
    let rows = bench::run(&mut gpu, &cfg).map_err(anyhow::Error::msg)?;
    println!(
        "{:<5} {:>4} {:>5} {:>7} {:<13} {:>9} {:>8} {:>7} {:>8} {:>9}",
        "part",
        "toks",
        "pool",
        "experts",
        "variant",
        "us/layer",
        "GB/s",
        "vs ggml",
        "differ",
        "outputs"
    );
    let mut reference = 0.0;
    let mut bad = 0;
    for r in &rows {
        if r.variant == "ggml" {
            reference = r.secs;
        }
        bad += usize::from(r.differ > 0);
        println!(
            "{:<5} {:>4} {:>5} {:>7.1} {:<13} {:>9.1} {:>8.1} {:>7.3} {:>8} {:>9}",
            r.part.name(),
            r.n_tok,
            r.pool,
            r.experts,
            r.variant,
            r.secs * 1e6,
            r.gbs(),
            r.secs / reference,
            r.differ,
            r.outputs,
        );
    }
    anyhow::ensure!(bad == 0, "{bad} variants differ from ggml's bits");
    println!("all bits equal to ggml's mul_mat_id");
    Ok(())
}
