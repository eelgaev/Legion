use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::bail;
use clap::ValueEnum;
use legion::bench::{self, Api, BenchConfig};

mod bandwidth;
mod mixed;
mod save;
mod workload;

pub use workload::Workload;

#[derive(Clone, Copy, ValueEnum)]
enum ApiKind {
    Openai,
    Anthropic,
}

impl ApiKind {
    fn name(self) -> String {
        self.to_possible_value().unwrap().get_name().to_string()
    }
}

#[derive(clap::Args)]
pub struct BenchArgs {
    /// Base URL of the server under test.
    #[arg(long, default_value = "http://127.0.0.1:8090")]
    url: String,
    #[arg(long, default_value = "mock")]
    model: String,
    #[arg(long, value_enum, default_value_t = ApiKind::Openai)]
    api: ApiKind,
    /// Workload: `bench` (concurrency sweep), `chats` (3 fixed prompts), `prefill`, or
    /// `bandwidth` (one prefill of --prompt-tokens, then --concurrency decodes of
    /// --gen-tokens after the same prompt: the model bytes per phase vs the device roofline, and the GPUs' hardware
    /// counters when the server runs with --hw-counters), or `mixed` (chat streams under
    /// background long prompts: TTFT/ITL percentiles per class, the "Workload mixed" flags).
    #[arg(long, value_enum, default_value_t = Workload::Bench)]
    workload: Workload,
    /// Concurrency levels to run, in order (default 1,8,32,128; workload `bench`). Workload
    /// `bandwidth`: parallel decode requests (one value, default 1).
    #[arg(long, value_delimiter = ',')]
    concurrency: Option<Vec<usize>>,
    /// Requests per level (default: 4 x concurrency; workload `bench` only).
    #[arg(long)]
    requests: Option<usize>,
    /// Approximate prompt length (about 1.3 tokens per word). For `--workload prefill`,
    /// a comma-separated list of target lengths N[,N...]; workload `bench` takes one value.
    #[arg(long, value_delimiter = ',', default_value = "512")]
    prompt_tokens: Vec<usize>,
    /// Tokens to generate per request (default 256; workloads `bench` and `bandwidth`).
    #[arg(long)]
    gen_tokens: Option<usize>,
    /// Leading tokens shared by all prompts, to exercise prefix caching (default 0; workload
    /// `bench` only).
    #[arg(long)]
    shared_prefix_tokens: Option<usize>,
    /// Also send `min_tokens` = the generated tokens (workload `bench`, OpenAI API). Off by
    /// default: `ignore_eos` already fixes the length, and `min_tokens` makes the server mask
    /// stop tokens per row (on Legion each such row's full logits come back to the host, ~1 ms
    /// per row on a V100 box), which greedy chats don't pay.
    #[arg(long)]
    min_tokens: bool,
    /// Prompt seed (default 0; workload `bench`, and `mixed`: its texts and Poisson arrivals).
    #[arg(long)]
    seed: Option<u64>,
    /// Bearer / x-api-key token for the server.
    #[arg(long, env = "LEGION_BENCH_API_KEY")]
    api_key: Option<String>,
    /// Print the report as JSON instead of a table.
    #[arg(long)]
    json: bool,
    /// Save results, meta.json and a README.md table in this directory (created if missing).
    #[arg(long, value_name = "DIR")]
    save: Option<PathBuf>,
    /// With --save: overwrite a directory that already holds a saved run.
    #[arg(long, requires = "save")]
    force: bool,
    #[command(flatten)]
    mixed: mixed::MixedArgs,
}

pub async fn run(args: BenchArgs) -> anyhow::Result<()> {
    let api = match args.api {
        ApiKind::Openai => Api::OpenAi,
        ApiKind::Anthropic => Api::Anthropic,
    };
    if !matches!(args.workload, Workload::Bench | Workload::Bandwidth) {
        let mut ignored = bench_only_flags(&args);
        if args.workload == Workload::Mixed {
            ignored.retain(|f| *f != "--seed");
        }
        if !ignored.is_empty() {
            eprintln!(
                "warning: --workload {} ignores the bench-only flags {}",
                args.workload.to_possible_value().unwrap().get_name(),
                ignored.join(", ")
            );
        }
    }
    match args.workload {
        Workload::Bench => run_bench(args, api).await,
        Workload::Chats => {
            workload::run_chats(workload::WorkloadArgs {
                url: args.url,
                model: args.model,
                api,
                api_key: args.api_key,
                json: args.json,
                save: args.save,
                force: args.force,
            })
            .await
        }
        Workload::Bandwidth => run_bandwidth(args, api).await,
        Workload::Mixed => {
            let plan = args.mixed.plan(args.seed.unwrap_or(0))?;
            mixed::run(
                workload::WorkloadArgs {
                    url: args.url,
                    model: args.model,
                    api,
                    api_key: args.api_key,
                    json: args.json,
                    save: args.save,
                    force: args.force,
                },
                plan,
            )
            .await
        }
        Workload::Prefill => {
            workload::run_prefill(
                workload::WorkloadArgs {
                    url: args.url,
                    model: args.model,
                    api,
                    api_key: args.api_key,
                    json: args.json,
                    save: args.save,
                    force: args.force,
                },
                &args.prompt_tokens,
            )
            .await
        }
    }
}

async fn run_bandwidth(args: BenchArgs, api: Api) -> anyhow::Result<()> {
    let ignored: Vec<&str> = bench_only_flags(&args)
        .into_iter()
        .filter(|f| !["--concurrency", "--gen-tokens"].contains(f))
        .collect();
    if !ignored.is_empty() {
        eprintln!(
            "warning: --workload bandwidth ignores {}",
            ignored.join(", ")
        );
    }
    let concurrency = match args.concurrency.as_deref() {
        None => 1,
        Some([c]) => *c,
        Some(_) => bail!("--workload bandwidth takes one --concurrency value"),
    };
    let [prompt_tokens] = args.prompt_tokens[..] else {
        bail!("--workload bandwidth takes one --prompt-tokens value");
    };
    bandwidth::run(
        workload::WorkloadArgs {
            url: args.url,
            model: args.model,
            api,
            api_key: args.api_key,
            json: args.json,
            save: args.save,
            force: args.force,
        },
        prompt_tokens,
        args.gen_tokens.unwrap_or(256),
        concurrency,
    )
    .await
}

/// The bench-only flags given on the command line (the others have no effect on chats/prefill).
fn bench_only_flags(args: &BenchArgs) -> Vec<&'static str> {
    [
        ("--concurrency", args.concurrency.is_some()),
        ("--requests", args.requests.is_some()),
        ("--gen-tokens", args.gen_tokens.is_some()),
        ("--seed", args.seed.is_some()),
        (
            "--shared-prefix-tokens",
            args.shared_prefix_tokens.is_some(),
        ),
    ]
    .into_iter()
    .filter_map(|(name, given)| given.then_some(name))
    .collect()
}

async fn run_bench(args: BenchArgs, api: Api) -> anyhow::Result<()> {
    let concurrency = args.concurrency.unwrap_or_else(|| vec![1, 8, 32, 128]);
    let gen_tokens = args.gen_tokens.unwrap_or(256);
    let shared_prefix_tokens = args.shared_prefix_tokens.unwrap_or(0);
    if concurrency.is_empty() || concurrency.contains(&0) {
        bail!("--concurrency needs positive integers");
    }
    if args.prompt_tokens.is_empty() || args.prompt_tokens.contains(&0) {
        bail!("--prompt-tokens needs positive integers");
    }
    if args.prompt_tokens.len() > 1 {
        bail!(
            "--prompt-tokens takes one value for --workload bench (got {}); \
             a list N[,N...] is for --workload prefill",
            args.prompt_tokens.len()
        );
    }
    let prompt_tokens = args.prompt_tokens[0];
    if let Some(dir) = &args.save {
        save::check_free(dir, args.force)?;
    }
    let date = save::rfc3339_utc(SystemTime::now());
    let cfg = BenchConfig {
        url: args.url.clone(),
        model: args.model.clone(),
        api,
        prompt_tokens,
        gen_tokens,
        shared_prefix_tokens,
        seed: args.seed.unwrap_or(0),
        api_key: args.api_key,
        min_tokens: args.min_tokens,
        priority: None,
        declared_max_tokens: None,
    };
    let client = bench::client()?;
    let mut reports = Vec::new();
    for &c in &concurrency {
        let n = args.requests.unwrap_or(4 * c);
        eprintln!("concurrency {c}: {n} requests");
        reports.push(bench::run_level(&client, &cfg, c, n).await);
    }
    let json = serde_json::to_string_pretty(&reports)?;
    if args.json {
        println!("{json}");
    } else {
        print!("{}", bench::format_table(&reports));
    }
    if let Some(dir) = &args.save {
        let meta = save::Meta {
            date,
            argv: save::redacted_argv(std::env::args()),
            url: &args.url,
            model: &args.model,
            api: &args.api.name(),
            workload: "bench",
            prompt_tokens: prompt_tokens.into(),
            gen_tokens,
            min_tokens: Some(cfg.min_tokens),
            shared_prefix_tokens: Some(shared_prefix_tokens),
            concurrency: Some(&concurrency),
            server_health: save::server_health(&client, &args.url).await,
            legion_version: super::version::full(),
        };
        save::write(dir, &json, &meta, &reports)?;
        eprintln!("saved to {}", dir.display());
    }
    Ok(())
}
