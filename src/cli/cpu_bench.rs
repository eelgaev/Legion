//! `legion cpu-bench`: the host kernels' exactness check and micro-benchmarks (a development
//! tool, #637), so a build tree's release binary can run them on the host they are for. Bind
//! the process with `numactl` to choose the socket.

use std::time::Duration;

use clap::{Args, Subcommand};
use legion::cpu::QuantFormat;
use legion::cpu::microbench::{self, ExpertShape};
use legion::residency::cpu_share::bench::{self as pool_bench, PoolBench};

#[derive(Args)]
pub struct CpuBenchArgs {
    #[command(subcommand)]
    what: What,
}

#[derive(Subcommand)]
enum What {
    /// Every quantized kernel this CPU supports against the scalar reference, bit for bit.
    Check,
    /// One matmul per format on one thread: every ISA against scalar and ggml's vec_dot.
    Kernels {
        /// Weight rows.
        #[arg(long, default_value_t = 640)]
        rows: usize,
        /// Columns (values per row).
        #[arg(long, default_value_t = 2560)]
        columns: usize,
        /// Formats, comma separated (q4_k, q5_k, q6_k, q8_0, q5_1, iq4_xs).
        #[arg(long, value_delimiter = ',', default_value = "q4_k,q5_k,q5_1,q8_0", value_parser = format)]
        formats: Vec<QuantFormat>,
        /// Tokens per matmul, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 3, 4])]
        tokens: Vec<usize>,
        /// Time per measurement, in milliseconds (best of the runs).
        #[arg(long, default_value_t = 300)]
        budget_ms: u64,
    },
    /// One Qwen3.8-Flash-Next expert at a time on one thread, cycling through experts in memory:
    /// microseconds per expert for every ISA and ggml.
    Experts {
        /// Experts in memory (more than the caches hold).
        #[arg(long, default_value_t = 64)]
        n_expert: usize,
        /// Down projection format (q5_1 in most layers of the UD-Q4_K_XL GGUF, q8_0 in some).
        #[arg(long, default_value = "q5_1", value_parser = format)]
        down: QuantFormat,
        /// Tokens per expert, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 3])]
        tokens: Vec<usize>,
        /// Time per measurement, in milliseconds (best of the runs).
        #[arg(long, default_value_t = 1000)]
        budget_ms: u64,
    },
    /// The CPU expert share's worker pool on this process's CPUs: microseconds per job and
    /// per expert by cores, SMT threads per core, tokens and experts per job.
    Pool {
        /// Experts in memory (more than the caches hold).
        #[arg(long, default_value_t = 64)]
        n_expert: usize,
        /// Down projection format.
        #[arg(long, default_value = "q5_1", value_parser = format)]
        down: QuantFormat,
        /// Pool cores, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 5, 10, 18])]
        cores: Vec<usize>,
        /// Threads per core, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 4])]
        smt: Vec<usize>,
        /// Tokens per job, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1, 3])]
        tokens: Vec<usize>,
        /// Experts per job, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [2, 8])]
        experts: Vec<usize>,
        /// Timed jobs per point (the median is shown).
        #[arg(long, default_value_t = 200)]
        jobs: usize,
    },
}

fn format(s: &str) -> Result<QuantFormat, String> {
    QuantFormat::ALL
        .into_iter()
        .find(|f| {
            format!("{f:?}")
                .replace('_', "")
                .eq_ignore_ascii_case(&s.replace('_', ""))
        })
        .ok_or_else(|| format!("unknown format {s}"))
}

pub fn run(args: CpuBenchArgs) -> anyhow::Result<()> {
    let mut line = |l: String| println!("{l}");
    let qwen = |down| ExpertShape {
        down,
        ..ExpertShape::QWEN_FLASH_NEXT
    };
    match args.what {
        What::Check => {
            let cases = microbench::check().map_err(anyhow::Error::msg)?;
            println!(
                "bit-exact: {cases} cases on {:?}",
                legion::cpu::supported_isas()
            );
        }
        What::Kernels {
            rows,
            columns,
            formats,
            tokens,
            budget_ms,
        } => microbench::kernels(
            rows,
            columns,
            &formats,
            &tokens,
            Duration::from_millis(budget_ms),
            &mut line,
        ),
        What::Experts {
            n_expert,
            down,
            tokens,
            budget_ms,
        } => microbench::experts(
            qwen(down),
            n_expert,
            &tokens,
            Duration::from_millis(budget_ms),
            &mut line,
        ),
        What::Pool {
            n_expert,
            down,
            cores,
            smt,
            tokens,
            experts,
            jobs,
        } => {
            let cfg = PoolBench {
                shape: qwen(down),
                n_expert,
                cores,
                smt,
                tokens,
                experts,
                jobs,
            };
            pool_bench::run(&cfg, &mut line)?;
        }
    }
    Ok(())
}
