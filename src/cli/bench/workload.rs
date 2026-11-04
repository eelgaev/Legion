//! `--workload chats` and `--workload prefill`: the Legion-vs-Strata harness client, in-tree.
//!
//! Behaviour matches the out-of-tree Legion-vs-Strata Python client (OpenAI streaming chat),
//! except the prefill prompt PRNG is SplitMix64 seeded with N (documented below) rather than
//! Python's `random`.

use std::path::Path;
use std::time::{Instant, SystemTime};

use anyhow::{Context, bail};
use clap::ValueEnum;
use futures::StreamExt;
use legion::bench::{self, Api, SseParser};
use serde::Serialize;
use serde_json::{Value, json};

use super::save;

/// Which workload `legion bench` runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum Workload {
    /// Concurrency sweep (the original `legion bench` behaviour).
    #[default]
    Bench,
    /// Three fixed chat prompts after a warm-up (harness "decode" / chats set).
    Chats,
    /// Prefill throughput: one ~N-token prompt per N, `max_tokens` 1.
    Prefill,
    /// Memory bandwidth (#391): one prefill, then decodes, each between two `/metrics`
    /// scrapes of a Legion server; the analytic model bytes next to the hardware counters.
    Bandwidth,
    /// Chat streams under background long prompts (#626): TTFT and ITL percentiles per class
    /// and the chat p99/p50 ITL ratio (#51's metric). Any OpenAI-compatible server.
    Mixed,
}

/// Function words for the prefill synthetic prompt (same list as the Python harness).
const PREFILL_WORDS: &[&str] = &[
    "the", "of", "and", "to", "in", "is", "was", "for", "on", "that", "with", "as", "by", "at",
    "from", "his", "her", "it", "an", "are", "were", "this", "be", "which", "or", "had", "not",
    "but",
];

/// SplitMix64: tiny deterministic PRNG so the same N always yields the same prefill text.
/// Seed with N; each item draws one word index and one number in 0..=999.
pub(super) struct SplitMix64(pub(super) u64);

impl SplitMix64 {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Deterministic synthetic user message of about `n` tokens: `n / 3` `word+number` items,
/// then a summarize instruction. Seeded SplitMix64 with `n`.
pub fn prefill_prompt(n: usize) -> String {
    synthetic_prompt(n, n as u64)
}

/// [`prefill_prompt`]'s text with the PRNG seeded with `seed` instead of `n`, so several
/// prompts of one length differ (`--workload mixed` sends many and must not hit a prefix
/// cache).
pub fn synthetic_prompt(n: usize, seed: u64) -> String {
    let mut rng = SplitMix64(seed);
    let items: Vec<String> = (0..n / 3)
        .map(|_| {
            let word = PREFILL_WORDS[(rng.next() % PREFILL_WORDS.len() as u64) as usize];
            let num = rng.next() % 1000;
            format!("{word}{num}")
        })
        .collect();
    format!("{}\nSummarize in one line.", items.join(" "))
}

/// Decode speed after the first token: `(completion_tokens - 1) / (last - first)`.
/// `None` when there are fewer than 2 tokens or the span is not positive.
pub fn decode_tok_s(completion_tokens: u64, first: Option<f64>, last: Option<f64>) -> Option<f64> {
    match (first, last) {
        (Some(f), Some(l)) if completion_tokens > 1 && l > f => {
            Some((completion_tokens - 1) as f64 / (l - f))
        }
        _ => None,
    }
}

/// Prefill throughput: `prompt_tokens / ttft_s`. `None` when either side is missing or zero.
pub fn prefill_tok_s(prompt_tokens: Option<u64>, ttft_s: Option<f64>) -> Option<f64> {
    match (prompt_tokens, ttft_s) {
        (Some(p), Some(t)) if p > 0 && t > 0.0 => Some(p as f64 / t),
        _ => None,
    }
}

const CHAT_PROMPTS: &[&str] = &[
    "Write a detailed 400-word explanation of how a CPU cache hierarchy works, with L1, L2, L3 and main memory.",
    "Write a Python module implementing an LRU cache class with get, put, resize and stats methods, with docstrings and type hints, then a pytest test file for it.",
    "List twenty major rivers of the world. For each, give its length, the countries it flows through and one historical fact about it.",
];

/// Shared client settings for chats / prefill (OpenAI streaming; Anthropic if `--api` says so).
pub struct WorkloadArgs {
    pub url: String,
    pub model: String,
    pub api: Api,
    pub api_key: Option<String>,
    pub json: bool,
    pub save: Option<std::path::PathBuf>,
    pub force: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRecord {
    pub prompt: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: u64,
    pub chunks: u64,
    pub ttft_s: Option<f64>,
    pub total_s: f64,
    /// `null` when fewer than 2 completion tokens (or a non-positive time span).
    pub decode_tok_s: Option<f64>,
    pub content: String,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrefillRecord {
    pub n_arg: usize,
    pub prompt_tokens: Option<u64>,
    pub ttft_s: Option<f64>,
    pub prefill_tok_s: Option<f64>,
    pub completion_tokens: u64,
    pub chunks: u64,
    pub total_s: f64,
    pub content: String,
    pub reasoning: String,
}

#[derive(Debug, Default)]
struct StreamResult {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    chunks: u64,
    ttft_s: Option<f64>,
    total_s: f64,
    first_s: Option<f64>,
    last_s: Option<f64>,
    content: String,
    reasoning: String,
}

fn endpoint(url: &str, api: Api) -> String {
    let base = url.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    match api {
        Api::OpenAi => format!("{base}/v1/chat/completions"),
        Api::Anthropic => format!("{base}/v1/messages"),
    }
}

fn request_body(api: Api, model: &str, prompt: &str, max_tokens: u32) -> Value {
    match api {
        Api::OpenAi => json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": true,
            "stream_options": {"include_usage": true},
        }),
        Api::Anthropic => json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": true,
        }),
    }
}

/// One non-empty content / reasoning piece from an OpenAI or Anthropic SSE payload.
struct DeltaPiece {
    content: String,
    reasoning: String,
}

struct UsageDelta {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

struct ParsedEvent {
    piece: Option<DeltaPiece>,
    usage: Option<UsageDelta>,
    done: bool,
    error: Option<String>,
}

fn delta_piece(api: Api, data: &str) -> ParsedEvent {
    if data.trim() == "[DONE]" {
        return ParsedEvent {
            piece: None,
            usage: None,
            done: true,
            error: None,
        };
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return ParsedEvent {
            piece: None,
            usage: None,
            done: false,
            error: None,
        };
    };
    if let Some(err) = v.get("error") {
        return ParsedEvent {
            piece: None,
            usage: None,
            done: false,
            error: Some(err.to_string()),
        };
    }
    let mut usage = None;
    let mut done = false;
    let mut piece = None;
    match api {
        Api::OpenAi => {
            if let Some(delta) = v.pointer("/choices/0/delta") {
                let content = delta
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let reasoning = delta
                    .get("reasoning_content")
                    .or_else(|| delta.get("reasoning"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !content.is_empty() || !reasoning.is_empty() {
                    piece = Some(DeltaPiece { content, reasoning });
                }
            }
            let pt = v.pointer("/usage/prompt_tokens").and_then(Value::as_u64);
            let ct = v
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64);
            if pt.is_some() || ct.is_some() {
                usage = Some(UsageDelta {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                });
            }
        }
        Api::Anthropic => match v.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                let d = v.get("delta");
                let content = d
                    .and_then(|d| d.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let reasoning = d
                    .and_then(|d| d.get("thinking"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !content.is_empty() || !reasoning.is_empty() {
                    piece = Some(DeltaPiece { content, reasoning });
                }
            }
            Some("message_start") => {
                let pt = v
                    .pointer("/message/usage/input_tokens")
                    .and_then(Value::as_u64);
                if pt.is_some() {
                    usage = Some(UsageDelta {
                        prompt_tokens: pt,
                        completion_tokens: None,
                    });
                }
            }
            Some("message_delta") => {
                let ct = v.pointer("/usage/output_tokens").and_then(Value::as_u64);
                let pt = v.pointer("/usage/input_tokens").and_then(Value::as_u64);
                if pt.is_some() || ct.is_some() {
                    usage = Some(UsageDelta {
                        prompt_tokens: pt,
                        completion_tokens: ct,
                    });
                }
            }
            Some("message_stop") => done = true,
            _ => {}
        },
    }
    ParsedEvent {
        piece,
        usage,
        done,
        error: None,
    }
}

async fn stream_once(
    client: &reqwest::Client,
    args: &WorkloadArgs,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<StreamResult> {
    let start = Instant::now();
    let mut req = client
        .post(endpoint(&args.url, args.api))
        .header("content-type", "application/json")
        .body(request_body(args.api, &args.model, prompt, max_tokens).to_string());
    if let Some(key) = &args.api_key {
        req = req.bearer_auth(key).header("x-api-key", key);
    }
    if args.api == Api::Anthropic {
        req = req.header("anthropic-version", "2023-06-01");
    }
    let resp = req.send().await.context("send")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!(
            "HTTP {status}: {}",
            body.chars().take(200).collect::<String>()
        );
    }
    read_stream(args.api, start, resp.bytes_stream()).await
}

/// Reads one SSE response body into a [`StreamResult`]. A stream with no non-empty delta is
/// not an error here (a prefill point whose single token was a swallowed `<think>` tag or EOS
/// has no TTFT, but its usage is still worth recording); callers that need a token check
/// `chunks` themselves, as `run_chats` does.
async fn read_stream<S, B, E>(api: Api, start: Instant, stream: S) -> anyhow::Result<StreamResult>
where
    S: futures::Stream<Item = Result<B, E>>,
    B: AsRef<[u8]>,
    E: std::error::Error + Send + Sync + 'static,
{
    let mut parser = SseParser::default();
    let mut out = StreamResult::default();
    let mut done = false;
    let mut stream = std::pin::pin!(stream);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read stream")?;
        let now = start.elapsed().as_secs_f64();
        for data in parser.push(chunk.as_ref()) {
            let ev = delta_piece(api, &data);
            if let Some(e) = ev.error {
                bail!("stream error: {e}");
            }
            if let Some(p) = ev.piece {
                out.chunks += 1;
                if out.first_s.is_none() {
                    out.first_s = Some(now);
                    out.ttft_s = Some(now);
                }
                out.last_s = Some(now);
                out.content.push_str(&p.content);
                out.reasoning.push_str(&p.reasoning);
            }
            if let Some(u) = ev.usage {
                if u.prompt_tokens.is_some() {
                    out.prompt_tokens = u.prompt_tokens.or(out.prompt_tokens);
                }
                if u.completion_tokens.is_some() {
                    out.completion_tokens = u.completion_tokens.or(out.completion_tokens);
                }
            }
            done |= ev.done;
        }
    }
    out.total_s = start.elapsed().as_secs_f64();
    if !done && out.completion_tokens.is_none() {
        bail!("stream ended early");
    }
    Ok(out)
}

/// A chat answer must have at least one token (client.py fails on a `None` TTFT there too).
fn into_chat(prompt: &str, s: StreamResult) -> anyhow::Result<ChatRecord> {
    if s.chunks == 0 {
        bail!("no tokens received");
    }
    let completion_tokens = s.completion_tokens.unwrap_or(s.chunks);
    Ok(ChatRecord {
        prompt: prompt.to_string(),
        prompt_tokens: s.prompt_tokens,
        completion_tokens,
        chunks: s.chunks,
        ttft_s: s.ttft_s,
        total_s: s.total_s,
        decode_tok_s: decode_tok_s(completion_tokens, s.first_s, s.last_s),
        content: s.content,
        reasoning: s.reasoning,
    })
}

/// A point with no visible token keeps `ttft_s` / `prefill_tok_s` as `null` (as client.py
/// does) instead of aborting the sweep.
fn into_prefill(n: usize, s: StreamResult) -> PrefillRecord {
    let completion_tokens = s.completion_tokens.unwrap_or(s.chunks);
    PrefillRecord {
        n_arg: n,
        prompt_tokens: s.prompt_tokens,
        ttft_s: s.ttft_s,
        prefill_tok_s: prefill_tok_s(s.prompt_tokens, s.ttft_s),
        completion_tokens,
        chunks: s.chunks,
        total_s: s.total_s,
        content: s.content,
        reasoning: s.reasoning,
    }
}

/// Warm-up + three chat prompts; prints a table or JSON; optionally saves `chats.json`.
pub async fn run_chats(args: WorkloadArgs) -> anyhow::Result<()> {
    if let Some(dir) = &args.save {
        save::check_free(dir, args.force)?;
    }
    let date = save::rfc3339_utc(SystemTime::now());
    let client = bench::client()?;
    // Warm-up: discarded.
    let _ = stream_once(&client, &args, "Hello", 16).await?;
    let mut records = Vec::with_capacity(CHAT_PROMPTS.len());
    for (i, prompt) in CHAT_PROMPTS.iter().enumerate() {
        eprintln!(
            "chats {}/{}: {}…",
            i + 1,
            CHAT_PROMPTS.len(),
            &prompt[..prompt.len().min(48)]
        );
        let s = stream_once(&client, &args, prompt, 384).await?;
        let rec = into_chat(prompt, s)?;
        eprintln!(
            "  {} completion tokens, TTFT {:.3} s, {:.1} decode tok/s",
            rec.completion_tokens,
            rec.ttft_s.unwrap_or(0.0),
            rec.decode_tok_s.unwrap_or(0.0)
        );
        records.push(rec);
    }
    let json = serde_json::to_string_pretty(&records)?;
    if args.json {
        println!("{json}");
    } else {
        print!("{}", format_chats_table(&records));
    }
    if let Some(dir) = &args.save {
        let save = SaveSpec {
            artifact: "chats.json",
            body: &json,
            prompt_tokens: Value::Null,
            gen_tokens: 384,
            table: &format_chats_readme(&records),
        };
        write_save(dir, &args, date, save, &client).await?;
        eprintln!("saved to {}", dir.display());
    }
    Ok(())
}

/// One prefill request per N in `prompt_tokens`.
pub async fn run_prefill(args: WorkloadArgs, prompt_tokens: &[usize]) -> anyhow::Result<()> {
    if prompt_tokens.is_empty() {
        bail!("--prompt-tokens needs at least one positive integer");
    }
    if prompt_tokens.contains(&0) {
        bail!("--prompt-tokens needs positive integers");
    }
    if let Some(dir) = &args.save {
        save::check_free(dir, args.force)?;
    }
    let date = save::rfc3339_utc(SystemTime::now());
    let client = bench::client()?;
    let mut records = Vec::with_capacity(prompt_tokens.len());
    for &n in prompt_tokens {
        eprintln!("prefill N={n}");
        let prompt = prefill_prompt(n);
        let s = stream_once(&client, &args, &prompt, 1).await?;
        let rec = into_prefill(n, s);
        if rec.ttft_s.is_none() {
            eprintln!(
                "  {} prompt tokens, no visible token: ttft_s / prefill_tok_s recorded as null",
                opt_u64(rec.prompt_tokens)
            );
        } else {
            eprintln!(
                "  {} prompt tokens, TTFT {:.3} s, {:.0} prefill tok/s",
                rec.prompt_tokens.unwrap_or(0),
                rec.ttft_s.unwrap_or(0.0),
                rec.prefill_tok_s.unwrap_or(0.0)
            );
        }
        records.push(rec);
    }
    let json = serde_json::to_string_pretty(&records)?;
    if args.json {
        println!("{json}");
    } else {
        print!("{}", format_prefill_table(&records));
    }
    if let Some(dir) = &args.save {
        let save = SaveSpec {
            artifact: "prefill.json",
            body: &json,
            prompt_tokens: json!(prompt_tokens),
            gen_tokens: 1,
            table: &format_prefill_readme(&records),
        };
        write_save(dir, &args, date, save, &client).await?;
        eprintln!("saved to {}", dir.display());
    }
    Ok(())
}

fn format_chats_table(records: &[ChatRecord]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>3} {:>8} {:>8} {:>6} {:>8} {:>8} {:>10}\n",
        "#", "prompt", "compl", "chunks", "ttft_s", "total_s", "decode_t/s"
    ));
    for (i, r) in records.iter().enumerate() {
        out.push_str(&format!(
            "{:>3} {:>8} {:>8} {:>6} {:>8.3} {:>8.3} {:>10}\n",
            i + 1,
            r.prompt_tokens
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            r.completion_tokens,
            r.chunks,
            r.ttft_s.unwrap_or(0.0),
            r.total_s,
            r.decode_tok_s
                .map(|v| format!("{v:.1}"))
                .unwrap_or_else(|| "-".into()),
        ));
    }
    out
}

fn format_prefill_table(records: &[PrefillRecord]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>8} {:>8} {:>8} {:>12}\n",
        "N", "prompt", "ttft_s", "prefill_t/s"
    ));
    for r in records {
        out.push_str(&format!(
            "{:>8} {:>8} {:>8.3} {:>12}\n",
            r.n_arg,
            r.prompt_tokens
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".into()),
            r.ttft_s.unwrap_or(0.0),
            r.prefill_tok_s
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".into()),
        ));
    }
    out
}

fn format_chats_readme(records: &[ChatRecord]) -> String {
    let mut out = String::from(
        "| # | prompt_tokens | completion_tokens | chunks | ttft_s | total_s | decode_tok_s |\n|---|---|---|---|---|---|---|\n",
    );
    for (i, r) in records.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} | {:.3} | {} |\n",
            i + 1,
            opt_u64(r.prompt_tokens),
            r.completion_tokens,
            r.chunks,
            r.ttft_s.unwrap_or(0.0),
            r.total_s,
            opt_f64(r.decode_tok_s),
        ));
    }
    out
}

fn format_prefill_readme(records: &[PrefillRecord]) -> String {
    let mut out =
        String::from("| N | prompt_tokens | ttft_s | prefill_tok_s |\n|---|---|---|---|\n");
    for r in records {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {} |\n",
            r.n_arg,
            opt_u64(r.prompt_tokens),
            r.ttft_s.unwrap_or(0.0),
            opt_f64(r.prefill_tok_s),
        ));
    }
    out
}

fn opt_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "-".into())
}

fn opt_f64(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "-".into())
}

pub(super) struct SaveSpec<'a> {
    pub artifact: &'a str,
    pub body: &'a str,
    pub prompt_tokens: Value,
    pub gen_tokens: usize,
    pub table: &'a str,
}

pub(super) async fn write_save(
    dir: &Path,
    args: &WorkloadArgs,
    date: String,
    spec: SaveSpec<'_>,
    client: &reqwest::Client,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let put = |name: &str, text: String| {
        let path = dir.join(name);
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
    };
    put(spec.artifact, format!("{}\n", spec.body))?;
    let api_name = match args.api {
        Api::OpenAi => "openai",
        Api::Anthropic => "anthropic",
    };
    let health = save::server_health(client, &args.url).await;
    let meta = save::Meta {
        date: date.clone(),
        argv: save::redacted_argv(std::env::args()),
        url: &args.url,
        model: &args.model,
        api: api_name,
        workload: spec.artifact.trim_end_matches(".json"),
        prompt_tokens: spec.prompt_tokens,
        gen_tokens: spec.gen_tokens,
        min_tokens: None,
        shared_prefix_tokens: None,
        concurrency: None,
        server_health: health,
        legion_version: super::super::version::full(),
    };
    put(
        "meta.json",
        format!(
            "{}\n",
            serde_json::to_string_pretty(&save::meta_json(&meta))?
        ),
    )?;
    let readme = format!(
        "# legion bench --workload {} {} @ {}\n\n{}\n\n```sh\n{}\n```\n\n{}",
        spec.artifact.trim_end_matches(".json"),
        meta.model,
        meta.url,
        meta.date,
        meta.argv
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" "),
        spec.table,
    );
    put("README.md", readme)?;
    Ok(())
}

fn shell_quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests;
