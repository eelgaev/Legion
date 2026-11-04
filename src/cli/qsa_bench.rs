//! `legion qsa-bench`: qwen4exp's sparse-attention prompt kernels (#828) at the model's shape on
//! one GPU, timed against each other and checked for the same output bits (a development tool,
//! so a build tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct QsaBenchArgs {
    /// Context lengths: each step is the last --tokens tokens of one, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [32768, 65536, 123_000])]
    ctx: Vec<usize>,
    /// Tokens per step (a prompt chunk).
    #[arg(long, default_value_t = 4096)]
    tokens: usize,
    /// Timed calls per kernel and round.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

#[cfg(feature = "cuda")]
pub fn run(args: QsaBenchArgs) -> anyhow::Result<()> {
    use legion::model::core::KvTypes;
    use legion::model::qwen4::{PromptKernel, qsa_bench};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    let kernels = [PromptKernel::Mma, PromptKernel::Gb, PromptKernel::Auto];
    println!(
        "{:<8} {:>7} {:>6} {:>5} {:>9} {:>8} {:>18}",
        "pools", "ctx", "tokens", "kern", "ms/call", "vs mma", "output bits"
    );
    let mut differ = 0;
    for &n_ctx in &args.ctx {
        anyhow::ensure!(
            n_ctx >= args.tokens + 4 * 512 * 2,
            "--ctx {n_ctx} is too short for --tokens {}",
            args.tokens
        );
        for local in [true, false] {
            let t = qsa_bench::prompt_ms(
                &mut gpu,
                KvTypes::default(),
                n_ctx,
                args.tokens,
                args.reps,
                args.rounds,
                local,
                &kernels,
            );
            for r in &t {
                println!(
                    "{:<8} {:>7} {:>6} {:>5} {:>9.3} {:>8.3} {:>18}",
                    if local { "local" } else { "spread" },
                    n_ctx,
                    args.tokens,
                    format!("{:?}", r.kernel).to_lowercase(),
                    r.ms,
                    r.ms / t[0].ms,
                    format!("{:016x}", r.bits)
                );
                differ += (r.bits != t[0].bits) as usize;
            }
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} outputs differ from attn_mma_kernel's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: QsaBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("qsa-bench needs a CUDA build")
}
