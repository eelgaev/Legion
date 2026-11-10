//! The `legion` command line. One file per subcommand; flags for a server feature go in
//! `serve.rs` as their own `#[command(flatten)]` group or as fields of `ServerArgs`.

mod bench;
mod chat;
mod comm_bench;
mod completions;
mod cpu_bench;
mod dense_mv_bench;
mod ep_place;
mod expert_mv_ptx_bench;
mod gdn_bench;
mod gguf;
mod hc_mix_bench;
mod hfmodel;
mod info;
mod lin_bench;
mod logging;
mod mla_bench;
mod mmq_ids_bench;
mod model_path;
mod nvfp4_gemm_bench;
mod nvfp4_mv_bench;
mod nvfp4_once_bench;
mod paged_split_bench;
mod q4k_head_bench;
mod q8_group_check;
mod q8_head_bench;
mod q8_mmq_check;
mod q8_mv_bench;
mod q_small_m_bench;
mod qsa_bench;
mod report;
mod run;
mod serve;
mod ssd_replay;
mod st;
mod statusline;
mod tokenize;
mod top;
mod version;

use clap::{Parser, Subcommand};
use logging::LogFormat;
use std::io::IsTerminal;

#[derive(Parser)]
#[command(name = "legion", version = version::full(), about)]
struct Cli {
    /// Log format: human-readable text, or one JSON object per line for log collectors.
    /// `RUST_LOG` filters both.
    #[arg(
        long,
        global = true,
        env = "LEGION_LOG_FORMAT",
        value_enum,
        default_value_t = LogFormat::Pretty
    )]
    log_format: LogFormat,

    /// ANSI colour in log lines (every command) and in `serve`'s startup report: auto colours
    /// a terminal unless TERM=dumb; `always` also shows the report when stdout is not a
    /// terminal. NO_COLOR (any value) turns `auto` off; an explicit `always` still colours.
    #[arg(long, global = true, value_enum, default_value_t)]
    color: report::ColorChoice,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the OpenAI- and Anthropic-compatible HTTP server.
    Serve(Box<serve::ServeArgs>),
    /// Live dashboard of a running server: requests, throughput, links, memory, logs.
    Top(top::TopArgs),
    /// Generate once from a local GGUF and print the result, without a server.
    Run(Box<run::RunArgs>),
    /// Benchmark a server: TTFT, inter-token latency and throughput per concurrency level.
    Bench(bench::BenchArgs),
    /// Interactive terminal chat against an OpenAI-compatible server.
    Chat(chat::ChatArgs),
    /// Work with GGUF files.
    #[command(subcommand)]
    Gguf(gguf::GgufCommand),
    /// Work with safetensors files and HF checkpoints.
    #[command(subcommand)]
    St(st::StCommand),
    /// Show GPUs, links, NUMA topology and the unified-memory model.
    Info(info::InfoArgs),
    /// Show a model's token ids and pieces for some text (tokenizer only, no weights).
    Tokenize(tokenize::TokenizeArgs),
    /// Print a shell completion script to stdout.
    Completions(completions::CompletionsArgs),
    /// Build a --tp-moe-ep-place expert placement from router choices that `serve
    /// --moe-route-record` wrote, and report the max-over-ranks expert count per layer.
    EpPlace(ep_place::EpPlaceArgs),
    /// Replay an SSD expert tier routing trace under several slot policies (development).
    #[command(hide = true)]
    SsdReplay(ssd_replay::SsdReplayArgs),
    /// Check and benchmark the host CPU kernels and the CPU expert share's pool (development).
    #[command(hide = true)]
    CpuBench(cpu_bench::CpuBenchArgs),
    /// Measure GPU-to-GPU exchange latency and host-memory read bandwidth (development).
    #[command(hide = true)]
    CommBench(comm_bench::CommBenchArgs),
    /// Check and time the decode expert matvec's PTX kernels against nvcc's (development).
    #[command(hide = true)]
    ExpertMvPtxBench(expert_mv_ptx_bench::ExpertMvPtxBenchArgs),
    /// Check and time the Q4_K / Q5_1 expert-grid op against ggml's MMQ (development).
    #[command(hide = true)]
    MmqIdsBench(mmq_ids_bench::MmqIdsBenchArgs),
    /// Check and time the NVFP4 decode expert FFN's grouped mv variants (development).
    #[command(hide = true)]
    Nvfp4MvBench(nvfp4_mv_bench::Nvfp4MvBenchArgs),
    /// Check and time the NVFP4 prefill expert GEMM's tiles and stages (development).
    #[command(hide = true)]
    Nvfp4GemmBench(nvfp4_gemm_bench::Nvfp4GemmBenchArgs),
    /// Check and time the gated DeltaNet decode kernel against ggml's (development).
    #[command(hide = true)]
    GdnBench(gdn_bench::GdnBenchArgs),
    /// Time the decode BF16 matmul op's variants against ggml's (development).
    #[command(hide = true)]
    DenseMvBench(dense_mv_bench::DenseMvBenchArgs),
    /// Check and time the Q8_0 draft LM head op against ggml's MMQ (development).
    #[command(hide = true)]
    Q8MvBench(q8_mv_bench::Q8MvBenchArgs),
    /// Check and time the 2-8-column quantized matmul op against ggml's MMVQ (development).
    #[command(hide = true)]
    QSmallMBench(q_small_m_bench::QSmallMBenchArgs),
    /// Check and time the one-column Q8_0 draft LM head op against ggml's mmvq (development).
    #[command(hide = true)]
    Q8HeadBench(q8_head_bench::Q8HeadBenchArgs),
    /// Check and time qwen4exp's hyper-connection mixer op in its kernel variants (development).
    #[command(hide = true)]
    HcMixBench(hc_mix_bench::HcMixBenchArgs),
    /// Check and time the read-each-expert-once NVFP4 expert op against ggml's mmvq
    /// (development).
    #[command(hide = true)]
    Nvfp4OnceBench(nvfp4_once_bench::Nvfp4OnceBenchArgs),
    /// Check and time the GB10 Q4_K draft LM head op against ggml's mmvq (development).
    #[command(hide = true)]
    Q4kHeadBench(q4k_head_bench::Q4kHeadBenchArgs),
    /// Compare the grouped Q8_0 matvec op's bits with ggml's `mul_mat` (development).
    #[command(hide = true)]
    Q8GroupCheck(q8_group_check::Q8GroupCheckArgs),
    /// Compare the two-launch Q8_0 MMQ op's bits and time with ggml's `mul_mat` (development).
    #[command(hide = true)]
    Q8MmqCheck(q8_mmq_check::Q8MmqCheckArgs),
    /// Check and time GLM's F16 prompt GEMM kernels against each other (development).
    #[command(hide = true)]
    LinBench(lin_bench::LinBenchArgs),
    /// Check and time qwen4exp's sparse-attention prompt kernels against each other (development).
    #[command(hide = true)]
    QsaBench(qsa_bench::QsaBenchArgs),
    /// Check and time GLM's sparse MLA prompt kernels against each other (development).
    #[command(hide = true)]
    MlaBench(mla_bench::MlaBenchArgs),
    /// Check and time the sm_70 / sm_75 paged attention split of verify steps against the WMMA
    /// kernel (development).
    #[command(hide = true)]
    PagedSplitBench(paged_split_bench::PagedSplitBenchArgs),
}

/// The parsed `Cli` as a bare `clap::Command`, so `completions` can generate from it
/// without exposing `Cli` outside this module.
pub(super) fn command() -> clap::Command {
    use clap::CommandFactory;
    Cli::command()
}

/// Parses the command line, applies `serve --numa` while the process has one thread (#422), then
/// runs the command on a multi-threaded runtime whose threads inherit that binding.
pub fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let numa = match &cli.command {
        Command::Serve(args) => serve::bind_numa(args)?,
        Command::CommBench(_) => {
            comm_bench::prepare_env();
            None
        }
        Command::Q8MmqCheck(_) => {
            q8_mmq_check::prepare_env();
            None
        }
        _ => None,
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cli, numa))
}

async fn run(cli: Cli, numa: Option<legion::hw::numa_bind::Placement>) -> anyhow::Result<()> {
    let term = report::TermEnv::from_env();
    if matches!(cli.command, Command::Run(_) | Command::Completions(_)) {
        logging::init_stderr(
            cli.log_format,
            cli.color.use_color(std::io::stderr().is_terminal(), term),
        );
    } else {
        logging::init(
            cli.log_format,
            cli.color.use_color(std::io::stdout().is_terminal(), term),
        );
    }

    match cli.command {
        Command::Serve(args) => serve::run(*args, numa, cli.log_format, cli.color).await,
        Command::Run(args) => run::run(*args).await,
        Command::Bench(args) => bench::run(args).await,
        Command::Chat(args) => chat::run(args).await,
        Command::Top(args) => top::run(args).await,
        Command::Gguf(cmd) => gguf::run(cmd),
        Command::St(cmd) => st::run(cmd),
        Command::Info(args) => info::run(args),
        Command::Tokenize(args) => tokenize::run(args),
        Command::Completions(args) => completions::run(args),
        Command::EpPlace(args) => ep_place::run(args),
        Command::SsdReplay(args) => ssd_replay::run(args),
        Command::CpuBench(args) => cpu_bench::run(args),
        Command::CommBench(args) => comm_bench::run(args),
        Command::ExpertMvPtxBench(args) => expert_mv_ptx_bench::run(args),
        Command::MmqIdsBench(args) => mmq_ids_bench::run(args),
        Command::Nvfp4MvBench(args) => nvfp4_mv_bench::run(args),
        Command::Nvfp4GemmBench(args) => nvfp4_gemm_bench::run(args),
        Command::GdnBench(args) => gdn_bench::run(args),
        Command::DenseMvBench(args) => dense_mv_bench::run(args),
        Command::Q8MvBench(args) => q8_mv_bench::run(args),
        Command::QSmallMBench(args) => q_small_m_bench::run(args),
        Command::Q8HeadBench(args) => q8_head_bench::run(args),
        Command::HcMixBench(args) => hc_mix_bench::run(args),
        Command::Nvfp4OnceBench(args) => nvfp4_once_bench::run(args),
        Command::Q4kHeadBench(args) => q4k_head_bench::run(args),
        Command::Q8GroupCheck(args) => q8_group_check::run(args),
        Command::Q8MmqCheck(args) => q8_mmq_check::run(args),
        Command::LinBench(args) => lin_bench::run(args),
        Command::QsaBench(args) => qsa_bench::run(args),
        Command::MlaBench(args) => mla_bench::run(args),
        Command::PagedSplitBench(args) => paged_split_bench::run(args),
    }
}
