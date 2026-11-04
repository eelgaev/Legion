//! `legion mla-bench`: GLM's sparse MLA prompt kernels (#862) at Maya-S's shape on one GPU,
//! timed against each other and checked for the same output bits (a development tool, so a build
//! tree's release binary can measure the box it runs on; pick the GPU with
//! `CUDA_VISIBLE_DEVICES`).

use clap::Args;

#[derive(Args)]
pub struct MlaBenchArgs {
    /// Context lengths: each step is the last --tokens tokens of one, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [2048, 8192, 32768])]
    ctx: Vec<usize>,
    /// Tokens per step (a prompt chunk).
    #[arg(long, default_value_t = 2048)]
    tokens: usize,
    /// Timed calls per kernel and round.
    #[arg(long, default_value_t = 5)]
    reps: usize,
    /// Rounds (the best counts; the kernels take turns within a round).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
}

#[cfg(feature = "cuda")]
pub fn run(args: MlaBenchArgs) -> anyhow::Result<()> {
    use legion::model::glm5next::ops::{MlaKernel, bench};
    let mut gpu = legion::ggml::backend_by_name("CUDA0")?;
    // mla_tc_kernel (`serve --glm-mla-tc`) for its time: it rounds q to F16, so its bits differ.
    let kernels = [
        MlaKernel::Tiled,
        MlaKernel::Gb,
        MlaKernel::Auto,
        MlaKernel::Tc,
    ];
    println!(
        "{:<8} {:>7} {:>6} {:>6} {:>9} {:>9} {:>18}",
        "pools", "ctx", "tokens", "kern", "ms/call", "vs tiled", "output bits"
    );
    let mut differ = 0;
    for &n_ctx in &args.ctx {
        anyhow::ensure!(
            n_ctx >= args.tokens && args.tokens >= 64,
            "--ctx {n_ctx} needs --tokens {} of 64 or more and no more than it",
            args.tokens
        );
        for local in [true, false] {
            let t = bench::prompt_ms(
                &mut gpu,
                n_ctx,
                args.tokens,
                args.reps,
                args.rounds,
                local,
                &kernels,
            );
            for r in &t {
                println!(
                    "{:<8} {:>7} {:>6} {:>6} {:>9.3} {:>9.3} {:>18}",
                    if local { "local" } else { "spread" },
                    n_ctx,
                    args.tokens,
                    format!("{:?}", r.kernel).to_lowercase(),
                    r.ms,
                    r.ms / t[0].ms,
                    format!("{:016x}", r.bits)
                );
                differ += (r.kernel != MlaKernel::Tc && r.bits != t[0].bits) as usize;
            }
        }
    }
    if differ > 0 {
        anyhow::bail!("{differ} outputs differ from mla_tiled_kernel's bits");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn run(_args: MlaBenchArgs) -> anyhow::Result<()> {
    anyhow::bail!("mla-bench needs a CUDA build")
}
