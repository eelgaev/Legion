//! Concurrency benchmark client: drives an OpenAI- or Anthropic-compatible server (ours or
//! someone else's) with streaming chat requests and reports TTFT, inter-token latency and
//! throughput per concurrency level. The CLI wrapper lives in `src/cli/bench.rs`.

use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use futures::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

pub mod bandwidth;

/// Rough English tokens per word, used to turn `--prompt-tokens` into a word count.
pub const TOKENS_PER_WORD: f64 = 1.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    OpenAi,
    Anthropic,
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Server base URL, e.g. `http://localhost:8090`; a trailing `/v1` is accepted.
    pub url: String,
    pub model: String,
    pub api: Api,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    /// Leading tokens identical across all prompts (exercises prefix caching).
    pub shared_prefix_tokens: usize,
    pub seed: u64,
    pub api_key: Option<String>,
    /// Also send `min_tokens` = `gen_tokens` (OpenAI API). Off by default: `ignore_eos` alone
    /// fixes the length, and a server that masks stop tokens on the host for rows under
    /// `min_tokens` (Legion reads back each such row's full logits) would be measured slower
    /// than greedy chats run.
    pub min_tokens: bool,
    /// The request's `priority` field (lower first, as SGLang, vLLM and Legion read it);
    /// left out of the body when `None`.
    pub priority: Option<i64>,
    /// `max_tokens` to send instead of `gen_tokens`, the client hanging up after `gen_tokens`
    /// deltas: a reply that ends long before the client's limit, as agents' replies do
    /// (#627: what a server reserves at admission). `None`: `gen_tokens`, read to the end.
    pub declared_max_tokens: Option<usize>,
}

/// What one request measured. Times are relative to the request start.
#[derive(Debug, Clone, Default)]
pub struct RequestOutcome {
    pub ok: bool,
    pub error: Option<String>,
    pub ttft: Option<Duration>,
    /// Gaps between consecutive content/reasoning deltas.
    pub itl: Vec<Duration>,
    pub e2e: Duration,
    /// From the server's `usage` when it reports one, else the number of deltas.
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Percentiles {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

/// One row of the report. Latencies are in milliseconds.
#[derive(Debug, Clone, Serialize)]
pub struct LevelReport {
    pub concurrency: usize,
    pub requests: usize,
    pub ok: usize,
    pub failed: usize,
    pub wall_secs: f64,
    pub output_tokens: u64,
    /// Total output tokens / wall time of the level.
    pub output_tok_s: f64,
    /// Median over requests of the per-request decode rate.
    pub per_user_tok_s: f64,
    pub ttft_ms: Percentiles,
    pub itl_ms: Percentiles,
    pub e2e_ms: Percentiles,
    /// First error seen, if any request failed.
    pub first_error: Option<String>,
    /// The server's "model bytes" over the level (#391): the bytes its steps had to read
    /// (weights, experts, KV; device memory and host link) and their GB/s, by phase, from its
    /// `/metrics` before and after; `None` when the server does not export them.
    pub model_bytes: Option<ModelBytes>,
}

pub use bandwidth::{ModelBytes, PhaseBytes};

/// Nearest-rank percentile (`p` in 0..=100) of an already sorted slice; 0 if empty.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// p50 / p90 / p99 of `values` (sorted here), nearest rank; zeroes when empty.
pub fn percentiles(mut values: Vec<f64>) -> Percentiles {
    values.sort_by(|a, b| a.total_cmp(b));
    Percentiles {
        p50: percentile(&values, 50.0),
        p90: percentile(&values, 90.0),
        p99: percentile(&values, 99.0),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Decode rate of one request: tokens after the first over the time after the first token.
/// Falls back to tokens over e2e when there is nothing to divide by.
pub fn per_user_rate(o: &RequestOutcome) -> Option<f64> {
    if o.output_tokens == 0 {
        return None;
    }
    if let Some(ttft) = o.ttft {
        let decode = o.e2e.saturating_sub(ttft).as_secs_f64();
        if o.output_tokens > 1 && decode > 0.0 {
            return Some((o.output_tokens - 1) as f64 / decode);
        }
    }
    let e2e = o.e2e.as_secs_f64();
    (e2e > 0.0).then(|| o.output_tokens as f64 / e2e)
}

/// Folds the outcomes of one concurrency level into a report row.
pub fn summarize(concurrency: usize, wall: Duration, outcomes: &[RequestOutcome]) -> LevelReport {
    let ok: Vec<&RequestOutcome> = outcomes.iter().filter(|o| o.ok).collect();
    let output_tokens: u64 = ok.iter().map(|o| o.output_tokens).sum();
    let wall_secs = wall.as_secs_f64();
    let rates: Vec<f64> = ok.iter().filter_map(|o| per_user_rate(o)).collect();
    LevelReport {
        concurrency,
        requests: outcomes.len(),
        ok: ok.len(),
        failed: outcomes.len() - ok.len(),
        wall_secs,
        output_tokens,
        output_tok_s: if wall_secs > 0.0 {
            output_tokens as f64 / wall_secs
        } else {
            0.0
        },
        per_user_tok_s: percentiles(rates).p50,
        ttft_ms: percentiles(ok.iter().filter_map(|o| o.ttft).map(ms).collect()),
        itl_ms: percentiles(
            ok.iter()
                .flat_map(|o| o.itl.iter().copied())
                .map(ms)
                .collect(),
        ),
        e2e_ms: percentiles(ok.iter().map(|o| ms(o.e2e)).collect()),
        first_error: outcomes.iter().find_map(|o| o.error.clone()),
        model_bytes: None,
    }
}

/// Renders the report rows as a fixed-width table.
pub fn format_table(reports: &[LevelReport]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>5} {:>8} {:>7} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}\n",
        "conc",
        "ok/fail",
        "wall_s",
        "out tok/s",
        "user t/s",
        "TTFT p50",
        "TTFT p90",
        "TTFT p99",
        "ITL p50",
        "ITL p99"
    ));
    for r in reports {
        out.push_str(&format!(
            "{:>5} {:>8} {:>7.1} {:>10.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.2} {:>9.2}\n",
            r.concurrency,
            format!("{}/{}", r.ok, r.failed),
            r.wall_secs,
            r.output_tok_s,
            r.per_user_tok_s,
            r.ttft_ms.p50,
            r.ttft_ms.p90,
            r.ttft_ms.p99,
            r.itl_ms.p50,
            r.itl_ms.p99,
        ));
    }
    out.push_str("(TTFT and ITL in ms)\n");
    for r in reports {
        if let Some(e) = &r.first_error {
            out.push_str(&format!(
                "concurrency {}: {} failed, first error: {e}\n",
                r.concurrency, r.failed
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Prompts

const WORDS: &[&str] = &[
    "river", "stone", "lamp", "garden", "window", "paper", "engine", "forest", "copper", "violet",
    "market", "bridge", "silver", "winter", "harbor", "candle", "meadow", "thunder", "ribbon",
    "castle", "pebble", "orchard", "falcon", "lantern", "marble", "canyon", "feather", "island",
    "mirror", "saddle", "tunnel", "velvet", "anchor", "blossom", "compass", "dune", "ember",
    "fabric", "glacier", "hollow", "ivory", "jungle", "kettle", "ladder", "mosaic", "nectar",
    "oasis", "prairie", "quartz", "rocket", "summit", "timber", "umbrella", "valley", "willow",
    "yonder", "zephyr", "basket", "cinder", "drift", "echo", "fjord", "grove",
];

/// splitmix64: a tiny deterministic generator, so prompts need no rand dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn words_for(tokens: usize) -> usize {
    (tokens as f64 / TOKENS_PER_WORD).round() as usize
}

fn random_words(rng: &mut Rng, n: usize) -> Vec<&'static str> {
    (0..n)
        .map(|_| WORDS[(rng.next() % WORDS.len() as u64) as usize])
        .collect()
}

/// Deterministic prompt for request `index`. The first `shared_prefix_tokens` (approximate)
/// are identical for every index; the rest differ per request, so prefix caches hit on the
/// shared part only.
pub fn make_prompt(cfg: &BenchConfig, index: usize) -> String {
    let total = words_for(cfg.prompt_tokens).max(1);
    let shared = words_for(cfg.shared_prefix_tokens).min(total);
    let mut words = random_words(&mut Rng(cfg.seed), shared);
    let mut rng = Rng(cfg.seed ^ (index as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
    words.extend(random_words(&mut rng, total - shared));
    words.join(" ")
}

// ---------------------------------------------------------------------------------------
// Requests

fn endpoint(cfg: &BenchConfig) -> String {
    let base = cfg.url.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    match cfg.api {
        Api::OpenAi => format!("{base}/v1/chat/completions"),
        Api::Anthropic => format!("{base}/v1/messages"),
    }
}

pub fn request_body(cfg: &BenchConfig, prompt: &str) -> Value {
    let max_tokens = cfg.declared_max_tokens.unwrap_or(cfg.gen_tokens);
    match cfg.api {
        Api::OpenAi => {
            let mut b = json!({
                "model": cfg.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "ignore_eos": true,
                "temperature": 0,
                "stream": true,
                "stream_options": {"include_usage": true},
            });
            if cfg.declared_max_tokens.is_some() {
                // A declared length is not run to its end: the reply stops at its own end or
                // when the client hangs up. Past an ignored end of sequence a model can go on
                // with tokens that stream no text, and the client would never hang up.
                b.as_object_mut().unwrap().remove("ignore_eos");
            }
            if cfg.min_tokens {
                b["min_tokens"] = cfg.gen_tokens.into();
            }
            if let Some(p) = cfg.priority {
                b["priority"] = p.into();
            }
            b
        }
        Api::Anthropic => json!({
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": true,
        }),
    }
}

/// Incremental server-sent-events parser: feed bytes, get the `data:` payload of each
/// completed event.
#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some((end, sep)) = find_event_end(&self.buf) {
            let raw: Vec<u8> = self.buf.drain(..end + sep).collect();
            let text = String::from_utf8_lossy(&raw[..end]);
            let data: Vec<&str> = text
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                events.push(data.join("\n"));
            }
        }
        events
    }
}

/// Position of the first blank line (event terminator) and its length.
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    (0..buf.len()).find_map(|i| {
        if buf[i..].starts_with(b"\r\n\r\n") {
            Some((i, 4))
        } else if buf[i..].starts_with(b"\n\n") {
            Some((i, 2))
        } else {
            None
        }
    })
}

/// What one SSE payload contributed.
#[derive(Debug, Default, PartialEq)]
pub struct EventInfo {
    /// A content or reasoning delta arrived.
    pub delta: bool,
    /// Completion tokens reported by the server.
    pub usage: Option<u64>,
    pub done: bool,
    pub error: Option<String>,
}

pub fn parse_event(api: Api, data: &str) -> EventInfo {
    let mut info = EventInfo::default();
    if data.trim() == "[DONE]" {
        info.done = true;
        return info;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return info;
    };
    if let Some(err) = v.get("error") {
        info.error = Some(err.to_string());
        return info;
    }
    let non_empty = |x: Option<&Value>| x.and_then(Value::as_str).is_some_and(|s| !s.is_empty());
    match api {
        Api::OpenAi => {
            if let Some(delta) = v.pointer("/choices/0/delta") {
                info.delta = ["content", "reasoning_content", "reasoning"]
                    .iter()
                    .any(|k| non_empty(delta.get(k)));
            }
            info.usage = v
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64);
        }
        Api::Anthropic => match v.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                let d = v.get("delta");
                info.delta = non_empty(d.and_then(|d| d.get("text")))
                    || non_empty(d.and_then(|d| d.get("thinking")));
            }
            Some("message_delta") => {
                info.usage = v.pointer("/usage/output_tokens").and_then(Value::as_u64);
            }
            Some("message_stop") => info.done = true,
            _ => {}
        },
    }
    info
}

/// Sends one streaming request and measures it. Never fails: errors land in the outcome.
pub async fn run_request(
    client: &reqwest::Client,
    cfg: &BenchConfig,
    prompt: &str,
) -> RequestOutcome {
    let start = Instant::now();
    let mut out = RequestOutcome::default();
    let res = stream_request(client, cfg, prompt, start, &mut out).await;
    out.e2e = start.elapsed();
    match res {
        Ok(()) => out.ok = true,
        Err(e) => out.error = Some(format!("{e:#}")),
    }
    out
}

async fn stream_request(
    client: &reqwest::Client,
    cfg: &BenchConfig,
    prompt: &str,
    start: Instant,
    out: &mut RequestOutcome,
) -> anyhow::Result<()> {
    let mut req = client
        .post(endpoint(cfg))
        .header("content-type", "application/json")
        .body(request_body(cfg, prompt).to_string());
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key).header("x-api-key", key);
    }
    if cfg.api == Api::Anthropic {
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
    let mut parser = SseParser::default();
    let mut last: Option<Instant> = None;
    let mut deltas = 0u64;
    let mut usage = None;
    let mut done = false;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read stream")?;
        let now = Instant::now();
        for data in parser.push(&chunk) {
            let info = parse_event(cfg.api, &data);
            if let Some(e) = info.error {
                bail!("stream error: {e}");
            }
            if info.delta {
                deltas += 1;
                match last {
                    None => out.ttft = Some(now - start),
                    Some(prev) => out.itl.push(now - prev),
                }
                last = Some(now);
                if cfg.declared_max_tokens.is_some() && deltas >= cfg.gen_tokens as u64 {
                    // The reply this client wanted: hang up (the server cancels the rest).
                    out.output_tokens = deltas;
                    return Ok(());
                }
            }
            usage = info.usage.or(usage);
            done |= info.done;
        }
    }
    if deltas == 0 {
        bail!("no tokens received");
    }
    if !done && usage.is_none() {
        bail!("stream ended early");
    }
    out.output_tokens = usage.unwrap_or(deltas);
    Ok(())
}

/// Runs `requests` requests with at most `concurrency` in flight.
pub async fn run_level(
    client: &reqwest::Client,
    cfg: &BenchConfig,
    concurrency: usize,
    requests: usize,
) -> LevelReport {
    let before = bandwidth::scrape(client, &cfg.url, cfg.api_key.as_deref()).await;
    let start = Instant::now();
    let outcomes: Vec<RequestOutcome> = futures::stream::iter(0..requests)
        .map(|i| async move { run_request(client, cfg, &make_prompt(cfg, i)).await })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    let mut report = summarize(concurrency, start.elapsed(), &outcomes);
    if let (Some(a), Some(b)) = (
        before,
        bandwidth::scrape(client, &cfg.url, cfg.api_key.as_deref()).await,
    ) {
        report.model_bytes = bandwidth::model_bytes_between(&a, &b);
    }
    report
}

/// Builds the HTTP client used for all levels.
pub fn client() -> anyhow::Result<reqwest::Client> {
    Ok(reqwest::Client::builder().build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(api: Api) -> BenchConfig {
        BenchConfig {
            url: "http://x:1/v1/".into(),
            model: "m".into(),
            api,
            prompt_tokens: 130,
            gen_tokens: 8,
            shared_prefix_tokens: 52,
            seed: 7,
            api_key: None,
            min_tokens: false,
            priority: None,
            declared_max_tokens: None,
        }
    }

    fn outcome(ttft_ms: u64, e2e_ms: u64, tokens: u64) -> RequestOutcome {
        RequestOutcome {
            ok: true,
            ttft: Some(Duration::from_millis(ttft_ms)),
            e2e: Duration::from_millis(e2e_ms),
            output_tokens: tokens,
            itl: vec![Duration::from_millis(10); tokens.saturating_sub(1) as usize],
            ..Default::default()
        }
    }

    #[test]
    fn percentile_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 50.0), 50.0);
        assert_eq!(percentile(&v, 90.0), 90.0);
        assert_eq!(percentile(&v, 99.0), 99.0);
        assert_eq!(percentile(&v, 100.0), 100.0);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&[4.0], 99.0), 4.0);
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn percentiles_sort_their_input() {
        let p = percentiles(vec![30.0, 10.0, 20.0, 40.0]);
        assert_eq!(
            p,
            Percentiles {
                p50: 20.0,
                p90: 40.0,
                p99: 40.0
            }
        );
    }

    #[test]
    fn per_user_rate_is_decode_rate() {
        // 101 tokens, first at 100 ms, done at 1100 ms: 100 tokens in 1 s.
        let r = per_user_rate(&outcome(100, 1100, 101)).unwrap();
        assert!((r - 100.0).abs() < 1e-9);
        // single token: falls back to tokens / e2e
        let r = per_user_rate(&outcome(100, 200, 1)).unwrap();
        assert!((r - 5.0).abs() < 1e-9);
        assert!(per_user_rate(&outcome(0, 100, 0)).is_none());
    }

    #[test]
    fn summarize_counts_and_throughput() {
        let mut outcomes = vec![outcome(50, 1050, 101), outcome(70, 1070, 101)];
        outcomes.push(RequestOutcome {
            ok: false,
            error: Some("boom".into()),
            ..Default::default()
        });
        let r = summarize(4, Duration::from_secs(2), &outcomes);
        assert_eq!((r.requests, r.ok, r.failed), (3, 2, 1));
        assert_eq!(r.output_tokens, 202);
        assert!((r.output_tok_s - 101.0).abs() < 1e-9);
        assert!((r.per_user_tok_s - 100.0).abs() < 1e-9);
        assert_eq!(r.ttft_ms.p50, 50.0);
        assert_eq!(r.ttft_ms.p99, 70.0);
        assert_eq!(r.itl_ms.p50, 10.0);
        assert_eq!(r.e2e_ms.p99, 1070.0);
        assert_eq!(r.first_error.as_deref(), Some("boom"));
    }

    #[test]
    fn summarize_all_failed_is_zeroes() {
        let bad = RequestOutcome::default();
        let r = summarize(1, Duration::from_secs(1), &[bad]);
        assert_eq!((r.ok, r.failed), (0, 1));
        assert_eq!(r.output_tok_s, 0.0);
        assert_eq!(r.ttft_ms.p50, 0.0);
    }

    #[test]
    fn prompts_are_deterministic_with_shared_prefix() {
        let c = cfg(Api::OpenAi);
        assert_eq!(make_prompt(&c, 3), make_prompt(&c, 3));
        let (a, b) = (make_prompt(&c, 0), make_prompt(&c, 1));
        assert_ne!(a, b);
        let (wa, wb): (Vec<_>, Vec<_>) = (a.split(' ').collect(), b.split(' ').collect());
        assert_eq!(wa.len(), 100);
        assert_eq!(wa[..40], wb[..40]);
        assert_ne!(wa[40..], wb[40..]);
        let other = BenchConfig { seed: 8, ..c };
        assert_ne!(make_prompt(&other, 0), a);
    }

    #[test]
    fn endpoint_normalizes_base_url() {
        assert_eq!(
            endpoint(&cfg(Api::OpenAi)),
            "http://x:1/v1/chat/completions"
        );
        assert_eq!(endpoint(&cfg(Api::Anthropic)), "http://x:1/v1/messages");
    }

    #[test]
    fn openai_body_fixes_length() {
        let b = request_body(&cfg(Api::OpenAi), "hi");
        assert_eq!(b["ignore_eos"], true);
        assert_eq!(b["temperature"], 0);
        assert!(b.get("min_tokens").is_none(), "min_tokens only when asked");
        let asked = BenchConfig {
            min_tokens: true,
            ..cfg(Api::OpenAi)
        };
        assert_eq!(request_body(&asked, "hi")["min_tokens"], 8);
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert!(
            request_body(&cfg(Api::Anthropic), "hi")
                .get("ignore_eos")
                .is_none()
        );
    }

    #[test]
    fn sse_parser_handles_split_chunks() {
        let mut p = SseParser::default();
        assert!(p.push(b"data: {\"a\":").is_empty());
        let ev = p.push(b"1}\n\ndata: [DONE]\n\n");
        assert_eq!(ev, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
        let ev = p.push(b"event: x\r\ndata: y\r\n\r\n");
        assert_eq!(ev, vec!["y".to_string()]);
    }

    #[test]
    fn parses_openai_events() {
        let a = Api::OpenAi;
        let role = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        assert!(!parse_event(a, role).delta);
        let tok = r#"{"choices":[{"delta":{"content":"hi"}}]}"#;
        assert!(parse_event(a, tok).delta);
        let think = r#"{"choices":[{"delta":{"reasoning_content":"hm"}}]}"#;
        assert!(parse_event(a, think).delta);
        let usage = r#"{"choices":[],"usage":{"completion_tokens":42}}"#;
        assert_eq!(parse_event(a, usage).usage, Some(42));
        assert!(parse_event(a, "[DONE]").done);
        assert!(
            parse_event(a, r#"{"error":{"message":"x"}}"#)
                .error
                .is_some()
        );
    }

    #[test]
    fn parses_anthropic_events() {
        let a = Api::Anthropic;
        let tok =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
        assert!(parse_event(a, tok).delta);
        let md = r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":9}}"#;
        assert_eq!(parse_event(a, md).usage, Some(9));
        assert!(parse_event(a, r#"{"type":"message_stop"}"#).done);
        assert!(!parse_event(a, r#"{"type":"ping"}"#).delta);
    }

    #[tokio::test]
    async fn failed_requests_are_counted_not_fatal() {
        // Nothing listens on port 1: every request fails, the level still reports.
        let mut c = cfg(Api::OpenAi);
        c.url = "http://127.0.0.1:1".into();
        let r = run_level(&client().unwrap(), &c, 2, 3).await;
        assert_eq!((r.requests, r.ok, r.failed), (3, 0, 3));
    }
}
