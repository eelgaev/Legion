//! `legion bench --workload mixed` (#626, part of #51): chat streams under background long
//! prompts. `--chat-streams` clients send back-to-back streaming chats while long prompts
//! arrive every `--long-every` seconds or as a Poisson process (`--long-rate`). Requests that
//! start inside the warm-up are left out; the report gives TTFT and inter-token latency
//! percentiles per class (chat, long) and the chat class's p99 / p50 ITL ratio, which #51
//! wants at 2 or under.
//!
//! Plain OpenAI `/v1/chat/completions` streaming (temperature 0, `ignore_eos`, fixed lengths)
//! and no `/metrics`, so it measures SGLang and vLLM the same way as Legion. Each request is
//! timed by [`bench::run_request`]: one ITL gap per streamed delta, however many tokens the
//! delta carries, and the output tokens from the server's `usage`.

use std::time::{Duration, Instant, SystemTime};

use anyhow::bail;
use futures::future::join_all;
use legion::bench::{self, Api, BenchConfig, Percentiles, RequestOutcome};
use serde::Serialize;
use serde_json::json;

use super::save;
use super::workload::{SaveSpec, SplitMix64, WorkloadArgs, synthetic_prompt, write_save};

/// The `--workload mixed` flags (the seed is the shared `--seed`).
#[derive(clap::Args, Debug, Clone)]
#[command(next_help_heading = "Workload mixed")]
pub struct MixedArgs {
    /// Chat clients, each sending back-to-back streaming requests (0: long prompts only).
    #[arg(long, default_value_t = 4)]
    pub chat_streams: usize,
    /// Approximate prompt tokens of each chat request (synthetic text, different per request).
    #[arg(long, default_value_t = 512)]
    pub chat_prompt_tokens: usize,
    /// Output tokens of each chat request (`max_tokens`, with `ignore_eos`).
    #[arg(long, default_value_t = 256)]
    pub chat_max_tokens: usize,
    /// Approximate prompt tokens of each background long prompt (synthetic text like
    /// `--workload prefill`, different per request so no prefix cache hits).
    #[arg(long, default_value_t = 16384)]
    pub long_prompt_tokens: usize,
    /// Output tokens of each long prompt.
    #[arg(long, default_value_t = 64)]
    pub long_max_tokens: usize,
    /// A long prompt every SECS seconds, the first one SECS after the start (0: none, a
    /// chat-only baseline).
    #[arg(long, value_name = "SECS", default_value_t = 10.0)]
    pub long_every: f64,
    /// Long prompts as a Poisson process of RATE arrivals per second instead of
    /// --long-every (the schedule follows --seed; 0: none).
    #[arg(long, value_name = "RATE", conflicts_with = "long_every")]
    pub long_rate: Option<f64>,
    /// Measured seconds, after the warm-up.
    #[arg(long, value_name = "SECS", default_value_t = 120.0)]
    pub duration: f64,
    /// Seconds of load before the measured window; requests that start in it are left out
    /// of the stats.
    #[arg(long, value_name = "SECS", default_value_t = 10.0)]
    pub warmup: f64,
    /// The `priority` field of the chat requests (lower first in SGLang, vLLM and Legion);
    /// not sent when unset.
    #[arg(long, value_name = "INT", allow_negative_numbers = true)]
    pub priority_chat: Option<i64>,
    /// The `priority` field of the long prompts; not sent when unset.
    #[arg(long, value_name = "INT", allow_negative_numbers = true)]
    pub priority_long: Option<i64>,
    /// Send `max_tokens` N on every request and hang up once the class's output tokens
    /// (--chat-max-tokens, --long-max-tokens) have streamed: replies that end long before
    /// the client's limit, as agents' do. Shows what a server reserves for `max_tokens` at
    /// admission (#627). Unset: `max_tokens` is the output length, read to the end.
    #[arg(long, value_name = "N")]
    pub declared_max_tokens: Option<usize>,
}

/// When the long prompts arrive.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Arrivals {
    /// No long prompts.
    None,
    /// One every `secs`, the first after one period.
    Every { secs: f64 },
    /// A Poisson process of `per_sec` arrivals per second.
    Poisson { per_sec: f64 },
}

/// A validated run, recorded in the JSON report.
#[derive(Debug, Clone, Serialize)]
pub struct MixedPlan {
    pub chat_streams: usize,
    pub chat_prompt_tokens: usize,
    pub chat_max_tokens: usize,
    pub long_prompt_tokens: usize,
    pub long_max_tokens: usize,
    pub arrivals: Arrivals,
    pub warmup_s: f64,
    pub duration_s: f64,
    pub seed: u64,
    pub priority_chat: Option<i64>,
    pub priority_long: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_max_tokens: Option<usize>,
}

impl MixedArgs {
    pub fn plan(&self, seed: u64) -> anyhow::Result<MixedPlan> {
        let tokens = [
            ("--chat-prompt-tokens", self.chat_prompt_tokens),
            ("--chat-max-tokens", self.chat_max_tokens),
            ("--long-prompt-tokens", self.long_prompt_tokens),
            ("--long-max-tokens", self.long_max_tokens),
        ];
        if let Some((name, _)) = tokens.iter().find(|(_, n)| *n == 0) {
            bail!("{name} needs a positive integer");
        }
        if self.declared_max_tokens == Some(0) {
            bail!("--declared-max-tokens needs a positive integer");
        }
        if !(self.duration.is_finite() && self.duration > 0.0) {
            bail!("--duration needs a positive number of seconds");
        }
        let secs = [
            ("--warmup", Some(self.warmup)),
            ("--long-every", Some(self.long_every)),
            ("--long-rate", self.long_rate),
        ];
        for (name, v) in secs {
            if let Some(v) = v.filter(|v| !(v.is_finite() && *v >= 0.0)) {
                bail!("{name} needs a number >= 0, got {v}");
            }
        }
        let arrivals = match self.long_rate {
            Some(r) if r > 0.0 => Arrivals::Poisson { per_sec: r },
            Some(_) => Arrivals::None,
            None if self.long_every > 0.0 => Arrivals::Every {
                secs: self.long_every,
            },
            None => Arrivals::None,
        };
        if self.chat_streams == 0 && arrivals == Arrivals::None {
            bail!("--workload mixed needs chat streams or long prompts");
        }
        Ok(MixedPlan {
            chat_streams: self.chat_streams,
            chat_prompt_tokens: self.chat_prompt_tokens,
            chat_max_tokens: self.chat_max_tokens,
            long_prompt_tokens: self.long_prompt_tokens,
            long_max_tokens: self.long_max_tokens,
            arrivals,
            warmup_s: self.warmup,
            duration_s: self.duration,
            seed,
            priority_chat: self.priority_chat,
            priority_long: self.priority_long,
            declared_max_tokens: self.declared_max_tokens,
        })
    }
}

/// Arrival times of the long prompts in seconds from the start, all before `horizon`, in
/// order. The Poisson gaps are exponential draws from a SplitMix64 seeded with `seed`.
pub fn arrival_schedule(arrivals: Arrivals, horizon: f64, seed: u64) -> Vec<f64> {
    let mut times = Vec::new();
    match arrivals {
        Arrivals::None => {}
        Arrivals::Every { secs } => {
            let mut k = 1.0;
            while k * secs < horizon {
                times.push(k * secs);
                k += 1.0;
            }
        }
        Arrivals::Poisson { per_sec } => {
            let mut rng = SplitMix64(seed ^ 0x6172_7269_7661_6c73);
            let mut t = 0.0;
            loop {
                // Uniform in [0, 1) from the top 53 bits; 1 - u is in (0, 1], so ln is finite.
                let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
                t += -(1.0 - u).ln() / per_sec;
                if t >= horizon {
                    break;
                }
                times.push(t);
            }
        }
    }
    times
}

/// The two request classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Chat,
    Long,
}

/// PRNG seed of one request's text: a different text per (seed, class, index).
pub fn text_seed(seed: u64, class: Class, index: u64) -> u64 {
    let salt: u64 = match class {
        Class::Chat => 0x6368_6174_0000_0000,
        Class::Long => 0x6c6f_6e67_0000_0000,
    };
    SplitMix64(seed ^ salt ^ index.wrapping_mul(0xD1B5_4A32_D192_ED03)).next()
}

/// The text of chat request `k` of stream `stream`.
pub fn chat_prompt(plan: &MixedPlan, stream: usize, k: usize) -> String {
    let index = (k * plan.chat_streams.max(1) + stream) as u64;
    synthetic_prompt(
        plan.chat_prompt_tokens,
        text_seed(plan.seed, Class::Chat, index),
    )
}

/// The text of long prompt `i` (in arrival order).
pub fn long_prompt(plan: &MixedPlan, i: usize) -> String {
    synthetic_prompt(
        plan.long_prompt_tokens,
        text_seed(plan.seed, Class::Long, i as u64),
    )
}

/// One finished request: its class, when it started (seconds from the run's start) and
/// what [`bench::run_request`] measured.
#[derive(Debug, Clone)]
pub struct Sample {
    pub class: Class,
    pub start_s: f64,
    pub outcome: RequestOutcome,
}

/// Pause after a failed chat request, so a broken server is not hammered in a tight loop.
const FAILED_CHAT_PAUSE: Duration = Duration::from_millis(100);

/// Runs the plan against the server `base` points at and returns every request. Chat
/// streams start requests until `warmup + duration`; requests in flight then run to the end.
pub async fn drive(client: &reqwest::Client, base: &BenchConfig, plan: &MixedPlan) -> Vec<Sample> {
    let horizon = plan.warmup_s + plan.duration_s;
    let chat_cfg = BenchConfig {
        prompt_tokens: plan.chat_prompt_tokens,
        gen_tokens: plan.chat_max_tokens,
        priority: plan.priority_chat,
        declared_max_tokens: plan.declared_max_tokens,
        ..base.clone()
    };
    let long_cfg = BenchConfig {
        prompt_tokens: plan.long_prompt_tokens,
        gen_tokens: plan.long_max_tokens,
        priority: plan.priority_long,
        declared_max_tokens: plan.declared_max_tokens,
        ..base.clone()
    };
    let schedule = arrival_schedule(plan.arrivals, horizon, plan.seed);
    let n_long = schedule.len();
    let origin = Instant::now();
    let chats = (0..plan.chat_streams).map(|stream| {
        let cfg = &chat_cfg;
        async move {
            let mut samples = Vec::new();
            for k in 0.. {
                let prompt = chat_prompt(plan, stream, k);
                let start_s = origin.elapsed().as_secs_f64();
                if start_s >= horizon {
                    break;
                }
                let outcome = bench::run_request(client, cfg, &prompt).await;
                if !outcome.ok {
                    tokio::time::sleep(FAILED_CHAT_PAUSE).await;
                }
                samples.push(Sample {
                    class: Class::Chat,
                    start_s,
                    outcome,
                });
            }
            samples
        }
    });
    let longs = schedule.into_iter().enumerate().map(|(i, at)| {
        let cfg = &long_cfg;
        async move {
            let prompt = long_prompt(plan, i);
            let due = origin + Duration::from_secs_f64(at);
            tokio::time::sleep_until(tokio::time::Instant::from_std(due)).await;
            let start_s = origin.elapsed().as_secs_f64();
            let outcome = bench::run_request(client, cfg, &prompt).await;
            match (&outcome.error, outcome.ttft) {
                (Some(e), _) => eprintln!("long {}/{n_long} at {start_s:.1} s: failed: {e}", i + 1),
                (None, ttft) => eprintln!(
                    "long {}/{n_long} at {start_s:.1} s: TTFT {:.3} s, {:.3} s total",
                    i + 1,
                    ttft.unwrap_or_default().as_secs_f64(),
                    outcome.e2e.as_secs_f64()
                ),
            }
            Sample {
                class: Class::Long,
                start_s,
                outcome,
            }
        }
    });
    let (chats, longs) = futures::join!(join_all(chats), join_all(longs));
    chats.into_iter().flatten().chain(longs).collect()
}

/// One class's row. Latencies in milliseconds.
#[derive(Debug, Clone, Serialize)]
pub struct ClassReport {
    pub requests: usize,
    pub ok: usize,
    pub failed: usize,
    pub output_tokens: u64,
    /// The class's output tokens over the measured window.
    pub output_tok_s: f64,
    /// Median over requests of the per-request decode rate (tokens after the first over the
    /// time after the first): one stream's speed.
    pub per_stream_tok_s: f64,
    pub ttft_ms: Percentiles,
    pub itl_ms: Percentiles,
    pub itl_max_ms: f64,
    pub first_error: Option<String>,
}

/// `--json` output and `mixed.json`.
#[derive(Debug, Clone, Serialize)]
pub struct MixedReport {
    pub plan: MixedPlan,
    /// From the end of the warm-up to the end of the last measured request, seconds (at
    /// least `duration_s`).
    pub window_s: f64,
    /// Requests that started in the warm-up, left out of the stats.
    pub warmup_requests: usize,
    pub chat: ClassReport,
    pub long: ClassReport,
    /// Chat ITL p99 / p50, #51's metric (target: <= 2); `None` without chat ITL samples.
    pub chat_itl_p99_over_p50: Option<f64>,
}

/// One class's stats, with [`bench::summarize`]'s maths.
pub fn summarize_class(outcomes: &[RequestOutcome], window_s: f64) -> ClassReport {
    let level = bench::summarize(0, Duration::from_secs_f64(window_s), outcomes);
    let itl_max_ms = outcomes
        .iter()
        .filter(|o| o.ok)
        .flat_map(|o| o.itl.iter())
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(0.0, f64::max);
    ClassReport {
        requests: level.requests,
        ok: level.ok,
        failed: level.failed,
        output_tokens: level.output_tokens,
        output_tok_s: level.output_tok_s,
        per_stream_tok_s: level.per_user_tok_s,
        ttft_ms: level.ttft_ms,
        itl_ms: level.itl_ms,
        itl_max_ms,
        first_error: level.first_error,
    }
}

/// Splits the samples at the warm-up and summarizes each class over the measured window.
pub fn report(plan: &MixedPlan, samples: Vec<Sample>) -> MixedReport {
    let horizon = plan.warmup_s + plan.duration_s;
    let (measured, warmup): (Vec<Sample>, Vec<Sample>) = samples
        .into_iter()
        .partition(|s| s.start_s >= plan.warmup_s);
    let end = measured
        .iter()
        .map(|s| s.start_s + s.outcome.e2e.as_secs_f64())
        .fold(horizon, f64::max);
    let window_s = end - plan.warmup_s;
    let class = |c: Class| -> Vec<RequestOutcome> {
        measured
            .iter()
            .filter(|s| s.class == c)
            .map(|s| s.outcome.clone())
            .collect()
    };
    let chat = summarize_class(&class(Class::Chat), window_s);
    let long = summarize_class(&class(Class::Long), window_s);
    let chat_itl_p99_over_p50 = (chat.itl_ms.p50 > 0.0).then(|| chat.itl_ms.p99 / chat.itl_ms.p50);
    MixedReport {
        plan: plan.clone(),
        window_s,
        warmup_requests: warmup.len(),
        chat,
        long,
        chat_itl_p99_over_p50,
    }
}

/// One line saying what the run does.
pub fn describe(plan: &MixedPlan) -> String {
    let longs = match plan.arrivals {
        Arrivals::None => "no long prompts".to_string(),
        Arrivals::Every { secs } => format!(
            "long prompts of {} -> {} tokens every {secs} s",
            plan.long_prompt_tokens, plan.long_max_tokens
        ),
        Arrivals::Poisson { per_sec } => format!(
            "long prompts of {} -> {} tokens, Poisson {per_sec}/s",
            plan.long_prompt_tokens, plan.long_max_tokens
        ),
    };
    let priority = |p: Option<i64>| p.map_or("unset".to_string(), |p| p.to_string());
    let declared = plan
        .declared_max_tokens
        .map_or(String::new(), |n| format!("; max_tokens {n} sent"));
    format!(
        "{} chat streams of {} -> {} tokens; {longs}; {} s warm-up + {} s; seed {}; \
         priority chat {}, long {}{declared}",
        plan.chat_streams,
        plan.chat_prompt_tokens,
        plan.chat_max_tokens,
        plan.warmup_s,
        plan.duration_s,
        plan.seed,
        priority(plan.priority_chat),
        priority(plan.priority_long),
    )
}

fn ratio_line(r: &MixedReport) -> String {
    match r.chat_itl_p99_over_p50 {
        Some(x) => format!("chat ITL p99 / p50: {x:.2} (#51 target: <= 2)\n"),
        None => "chat ITL p99 / p50: - (no chat ITL samples)\n".to_string(),
    }
}

fn classes(r: &MixedReport) -> [(&'static str, &ClassReport); 2] {
    [("chat", &r.chat), ("long", &r.long)]
}

/// The terminal table.
pub fn format_table(r: &MixedReport) -> String {
    let mut out = format!(
        "{}\nwindow {:.1} s ({} warm-up requests left out)\n",
        describe(&r.plan),
        r.window_s,
        r.warmup_requests
    );
    out.push_str(&format!(
        "{:<5} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}\n",
        "class",
        "ok/fail",
        "out tok/s",
        "strm t/s",
        "TTFT p50",
        "TTFT p90",
        "TTFT p99",
        "ITL p50",
        "ITL p90",
        "ITL p99",
        "ITL max"
    ));
    for (name, c) in classes(r) {
        out.push_str(&format!(
            "{:<5} {:>8} {:>9.1} {:>8.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2}\n",
            name,
            format!("{}/{}", c.ok, c.failed),
            c.output_tok_s,
            c.per_stream_tok_s,
            c.ttft_ms.p50,
            c.ttft_ms.p90,
            c.ttft_ms.p99,
            c.itl_ms.p50,
            c.itl_ms.p90,
            c.itl_ms.p99,
            c.itl_max_ms,
        ));
    }
    out.push_str("(TTFT and ITL in ms; strm t/s: the median request's decode rate)\n");
    out.push_str(&ratio_line(r));
    for (name, c) in classes(r) {
        if let Some(e) = &c.first_error {
            out.push_str(&format!("{name}: {} failed, first error: {e}\n", c.failed));
        }
    }
    out
}

/// The saved README's table.
pub fn format_readme(r: &MixedReport) -> String {
    let mut out = format!(
        "{}. Window {:.1} s; {} warm-up requests left out.\n\n",
        describe(&r.plan),
        r.window_s,
        r.warmup_requests
    );
    out.push_str(
        "| class | requests | ok | failed | out tok/s | per-stream tok/s | TTFT p50 / p90 / p99 (ms) | ITL p50 / p90 / p99 / max (ms) |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for (name, c) in classes(r) {
        out.push_str(&format!(
            "| {name} | {} | {} | {} | {:.1} | {:.1} | {:.1} / {:.1} / {:.1} | {:.2} / {:.2} / {:.2} / {:.2} |\n",
            c.requests,
            c.ok,
            c.failed,
            c.output_tok_s,
            c.per_stream_tok_s,
            c.ttft_ms.p50,
            c.ttft_ms.p90,
            c.ttft_ms.p99,
            c.itl_ms.p50,
            c.itl_ms.p90,
            c.itl_ms.p99,
            c.itl_max_ms,
        ));
    }
    out.push('\n');
    out.push_str(&ratio_line(r));
    out
}

pub async fn run(args: WorkloadArgs, plan: MixedPlan) -> anyhow::Result<()> {
    if args.api != Api::OpenAi {
        bail!("--workload mixed speaks the OpenAI API only (--api openai)");
    }
    if let Some(dir) = &args.save {
        save::check_free(dir, args.force)?;
    }
    let date = save::rfc3339_utc(SystemTime::now());
    let client = bench::client()?;
    let base = BenchConfig {
        url: args.url.clone(),
        model: args.model.clone(),
        api: args.api,
        prompt_tokens: 0,
        gen_tokens: 0,
        shared_prefix_tokens: 0,
        seed: plan.seed,
        api_key: args.api_key.clone(),
        min_tokens: false,
        priority: None,
        declared_max_tokens: None,
    };
    // A short request first: fails fast on a wrong URL, model or key, and warms the server.
    let probe = BenchConfig {
        gen_tokens: 4,
        ..base.clone()
    };
    let probe = bench::run_request(&client, &probe, "Hello").await;
    if let Some(e) = probe.error {
        bail!("the first request failed: {e}");
    }
    eprintln!("mixed: {}", describe(&plan));
    let samples = drive(&client, &base, &plan).await;
    let report = report(&plan, samples);
    let body = serde_json::to_string_pretty(&report)?;
    if args.json {
        println!("{body}");
    } else {
        print!("{}", format_table(&report));
    }
    if let Some(dir) = &args.save {
        let spec = SaveSpec {
            artifact: "mixed.json",
            body: &body,
            prompt_tokens: json!({
                "chat": plan.chat_prompt_tokens,
                "long": plan.long_prompt_tokens,
            }),
            gen_tokens: plan.chat_max_tokens,
            table: &format_readme(&report),
        };
        write_save(dir, &args, date, spec, &client).await?;
        eprintln!("saved to {}", dir.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
