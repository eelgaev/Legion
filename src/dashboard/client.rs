//! `legion top`'s client: two server-sent-event streams (`/v1/stats/stream`,
//! `/v1/logs/stream`) read into a [`Feed`], each reconnecting with backoff (0.5 s doubling to
//! 10 s, reset by a good connection) while the screen shows "disconnected".

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;

use super::logs::LogLine;
use super::snapshot::Snapshot;
use super::term::Feed;
use super::ui::Conn;

/// `legion serve`'s default listen address.
pub const DEFAULT_URL: &str = "http://127.0.0.1:8090";
const BACKOFF_START: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(10);
/// No snapshot for this long: the server is gone (it sends one every interval).
const STATS_SILENCE: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub struct Options {
    /// The server's base URL (`http://host:port`).
    pub url: String,
    pub api_key: Option<String>,
    /// Accept a self-signed (any) TLS certificate.
    pub insecure: bool,
    pub interval_ms: u64,
}

/// The next backoff after `d`.
pub fn next_backoff(d: Duration) -> Duration {
    (d * 2).min(BACKOFF_MAX)
}

/// `url` without a trailing slash, `http://` added when it names no scheme.
pub fn base_url(url: &str) -> String {
    let url = url.trim().trim_end_matches('/');
    if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}

/// Appends `chunk` to `buf` as UTF-8. A character split across two network chunks waits in
/// `pending` for the rest of its bytes instead of turning into two U+FFFD; bytes that can
/// never be valid are replaced as usual.
pub fn push_utf8(buf: &mut String, pending: &mut Vec<u8>, chunk: &[u8]) {
    pending.extend_from_slice(chunk);
    let complete = match std::str::from_utf8(pending) {
        Ok(_) => pending.len(),
        // Only an unfinished character at the very end waits.
        Err(e) if e.error_len().is_none() => e.valid_up_to(),
        Err(_) => pending.len(),
    };
    buf.push_str(&String::from_utf8_lossy(&pending[..complete]));
    pending.drain(..complete);
}

/// Splits complete server-sent events off the front of `buf`, returning each one's data
/// (its `data:` lines joined); comments and keep-alives give nothing.
pub fn take_events(buf: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let normalized_end = buf.find("\n\n").map(|i| (i, 2));
        let crlf_end = buf.find("\r\n\r\n").map(|i| (i, 4));
        let end = match (normalized_end, crlf_end) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        };
        let Some((at, len)) = end else { break };
        let event: String = buf.drain(..at + len).collect();
        let data: Vec<&str> = event
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(|d| d.strip_prefix(' ').unwrap_or(d))
            .collect();
        if !data.is_empty() {
            out.push(data.join("\n"));
        }
    }
    out
}

fn client(opts: &Options) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .tls_danger_accept_invalid_certs(opts.insecure)
        .build()
        .map_err(|e| e.to_string())
}

fn request(c: &reqwest::Client, opts: &Options, path: &str) -> reqwest::RequestBuilder {
    let mut r = c
        .get(format!("{}{path}", base_url(&opts.url)))
        .header("accept", "text/event-stream");
    if let Some(k) = &opts.api_key {
        r = r.bearer_auth(k);
    }
    r
}

/// Why a connection failed, short.
fn describe(e: &reqwest::Error) -> String {
    if e.is_connect() {
        "cannot connect".into()
    } else if e.is_timeout() {
        "timed out".into()
    } else {
        // reqwest's chain names the cause (TLS, DNS, ...).
        let mut s = e.to_string();
        let mut src = std::error::Error::source(e);
        while let Some(c) = src {
            s = format!("{s}: {c}");
            src = c.source();
        }
        s
    }
}

async fn open(
    c: &reqwest::Client,
    opts: &Options,
    path: &str,
) -> Result<reqwest::Response, String> {
    let resp = request(c, opts, path)
        .send()
        .await
        .map_err(|e| describe(&e))?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err("HTTP 401: the server wants an API key (--api-key or LEGION_API_KEY)".into());
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err("HTTP 404: the server has no /v1/stats (an older Legion?)".into());
    }
    if !status.is_success() {
        return Err(format!("HTTP {}", status.as_u16()));
    }
    Ok(resp)
}

/// Reads the stats stream until it fails.
async fn stats_once(c: &reqwest::Client, opts: &Options, feed: &Feed) -> Result<(), String> {
    let path = format!("/v1/stats/stream?interval_ms={}", opts.interval_ms);
    let resp = open(c, opts, &path).await?;
    let mut body = resp.bytes_stream();
    let mut buf = String::new();
    let mut pending = Vec::new();
    let mut connected = false;
    loop {
        let chunk = match tokio::time::timeout(STATS_SILENCE, body.next()).await {
            Err(_) => return Err("no data for 10 s".into()),
            Ok(None) => return Err("the server closed the stream".into()),
            Ok(Some(Err(e))) => return Err(describe(&e)),
            Ok(Some(Ok(b))) => b,
        };
        push_utf8(&mut buf, &mut pending, &chunk);
        for data in take_events(&mut buf) {
            match serde_json::from_str::<Snapshot>(&data) {
                Ok(s) => {
                    if !connected {
                        connected = true;
                        feed.conn(Conn::Connected);
                    }
                    feed.snapshot(s);
                }
                Err(e) => return Err(format!("bad snapshot: {e}")),
            }
        }
    }
}

/// Reads the log stream until it fails.
async fn logs_once(c: &reqwest::Client, opts: &Options, feed: &Feed) -> Result<(), String> {
    let resp = open(c, opts, "/v1/logs/stream").await?;
    let mut body = resp.bytes_stream();
    let mut buf = String::new();
    let mut pending = Vec::new();
    // A new connection sends the server's whole ring again.
    let mut reset = true;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| describe(&e))?;
        push_utf8(&mut buf, &mut pending, &chunk);
        let lines: Vec<LogLine> = take_events(&mut buf)
            .iter()
            .filter_map(|d| serde_json::from_str(d).ok())
            .collect();
        if !lines.is_empty() || reset {
            feed.logs(lines, reset);
            reset = false;
        }
    }
    Err("the server closed the log stream".into())
}

/// Starts both streams on the current runtime; they run until the runtime stops.
pub fn spawn(opts: Options, feed: Arc<Feed>) -> Result<(), String> {
    let c = client(&opts)?;
    {
        let (c, opts, feed) = (c.clone(), opts.clone(), feed.clone());
        tokio::spawn(async move {
            let mut backoff = BACKOFF_START;
            loop {
                let t = std::time::Instant::now();
                let err = stats_once(&c, &opts, &feed).await.err().unwrap_or_default();
                // A connection that lasted resets the backoff.
                if t.elapsed() > Duration::from_secs(5) {
                    backoff = BACKOFF_START;
                }
                feed.conn(Conn::Disconnected {
                    error: err,
                    retry_s: backoff.as_secs_f64(),
                });
                tokio::time::sleep(backoff).await;
                backoff = next_backoff(backoff);
                feed.conn(Conn::Connecting);
            }
        });
    }
    tokio::spawn(async move {
        let mut backoff = BACKOFF_START;
        loop {
            let t = std::time::Instant::now();
            // The stats stream reports the connection; the logs just come back with it.
            let _ = logs_once(&c, &opts, &feed).await;
            if t.elapsed() > Duration::from_secs(5) {
                backoff = BACKOFF_START;
            }
            tokio::time::sleep(backoff).await;
            backoff = next_backoff(backoff);
        }
    });
    Ok(())
}
