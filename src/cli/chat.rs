//! `legion chat`: an interactive terminal chat against a running `legion serve` (or any
//! OpenAI-compatible server). Replies stream in; `--once` sends a single message and exits,
//! which is what scripts and tests use.

use std::io::{self, BufRead, IsTerminal, Write};
use std::time::Instant;

use anyhow::{Context, bail};
use futures::StreamExt;
use legion::bench::SseParser;
use serde_json::{Value, json};

#[derive(clap::Args)]
pub struct ChatArgs {
    /// Base URL of the server.
    #[arg(long, default_value = "http://localhost:8090")]
    url: String,
    /// Model to chat with; default: the first id from GET /v1/models.
    #[arg(long)]
    model: Option<String>,
    /// Bearer / x-api-key token for the server.
    #[arg(long, env = "LEGION_API_KEY")]
    api_key: Option<String>,
    /// Sent as the first `system` message of every request.
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Ask the server for reasoning (`reasoning_effort: high`) and print it dim.
    #[arg(long)]
    thinking: bool,
    /// Send one message, print the reply and exit: no prompt, stats line or ANSI.
    #[arg(long)]
    once: Option<String>,
}

/// How a turn is presented: interactive adds the stats line and ANSI dimming (on a tty).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Interactive,
    Once,
}

/// The conversation: request defaults plus the message history sent on every turn.
struct Chat {
    client: reqwest::Client,
    /// Base URL without a trailing slash or `/v1`.
    base: String,
    /// Resolved lazily from `/v1/models` when `--model` was not given.
    model: Option<String>,
    api_key: Option<String>,
    system: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    thinking: bool,
    history: Vec<Value>,
}

impl Chat {
    fn new(args: &ChatArgs) -> Self {
        let base = args.url.trim_end_matches('/');
        Self {
            client: reqwest::Client::new(),
            base: base.strip_suffix("/v1").unwrap_or(base).to_string(),
            model: args.model.clone(),
            api_key: args.api_key.clone(),
            system: args.system.clone(),
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            thinking: args.thinking,
            history: Vec::new(),
        }
    }

    fn with_key(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => req.bearer_auth(key).header("x-api-key", key),
            None => req,
        }
    }

    fn request_body(&self) -> Value {
        let mut messages = Vec::with_capacity(self.history.len() + 1);
        if let Some(system) = &self.system {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.extend_from_slice(&self.history);
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "stream_options": {"include_usage": true},
            "reasoning_effort": if self.thinking { "high" } else { "none" },
        });
        if let Some(t) = self.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        body
    }

    /// The first served model id, used when `--model` was not given.
    async fn fetch_model(&self) -> anyhow::Result<String> {
        let req = self.with_key(self.client.get(format!("{}/v1/models", self.base)));
        let resp = req.send().await.context("GET /v1/models")?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{}", http_error(status, &body));
        }
        let v: Value = serde_json::from_str(&body).context("bad JSON from /v1/models")?;
        v["data"][0]["id"]
            .as_str()
            .map(str::to_string)
            .with_context(|| format!("{} serves no models", self.base))
    }

    /// One user turn: append the message, stream the reply, keep both in history. On error
    /// the user message is taken back out so the next turn starts from clean history.
    async fn turn(&mut self, text: &str, mode: Mode) -> anyhow::Result<()> {
        if self.model.is_none() {
            self.model = Some(self.fetch_model().await?);
        }
        self.history.push(json!({"role": "user", "content": text}));
        let body = self.request_body();
        match self.stream(&body, mode).await {
            Ok(answer) => {
                self.history
                    .push(json!({"role": "assistant", "content": answer}));
                Ok(())
            }
            Err(e) => {
                self.history.pop();
                Err(e)
            }
        }
    }

    /// Sends the history and prints the reply as it arrives. Returns the answer text.
    async fn stream(&self, body: &Value, mode: Mode) -> anyhow::Result<String> {
        let req = self
            .client
            .post(format!("{}/v1/chat/completions", self.base))
            .header("content-type", "application/json")
            .body(body.to_string());
        let resp = self.with_key(req).send().await.context("send")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("{}", http_error(status, &body));
        }
        let ansi = mode == Mode::Interactive && io::stdout().is_terminal();
        let start = Instant::now();
        let mut stream = resp.bytes_stream();
        let mut parser = SseParser::default();
        let mut answer = String::new();
        let mut thinking = false;
        let mut usage: Option<u64> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read stream")?;
            for data in parser.push(&chunk) {
                // `[DONE]` and anything unparsable carry nothing to print.
                let Ok(v) = serde_json::from_str::<Value>(&data) else {
                    continue;
                };
                if let Some(err) = v.get("error") {
                    end_thinking(&mut thinking, ansi);
                    bail!("{}", error_message(err));
                }
                let delta = v.pointer("/choices/0/delta");
                let mut reasoning = delta_text(delta, "reasoning_content");
                if reasoning.is_empty() {
                    reasoning = delta_text(delta, "reasoning");
                }
                if !reasoning.is_empty() {
                    if !thinking {
                        thinking = true;
                        print!(
                            "{}",
                            if ansi {
                                "\x1b[2mthinking: "
                            } else {
                                "thinking: "
                            }
                        );
                    }
                    print!("{reasoning}");
                    io::stdout().flush()?;
                }
                let content = delta_text(delta, "content");
                if !content.is_empty() {
                    end_thinking(&mut thinking, ansi);
                    answer.push_str(content);
                    print!("{content}");
                    io::stdout().flush()?;
                }
                if let Some(n) = v
                    .pointer("/usage/completion_tokens")
                    .and_then(Value::as_u64)
                {
                    usage = Some(n);
                }
            }
        }
        end_thinking(&mut thinking, ansi);
        println!();
        if let (Mode::Interactive, Some(n)) = (mode, usage) {
            println!("[{n} tokens, {:.1} tok/s]", rate(n, start.elapsed()));
        }
        Ok(answer)
    }
}

pub async fn run(args: ChatArgs) -> anyhow::Result<()> {
    let mut chat = Chat::new(&args);
    let Some(text) = args.once.as_deref() else {
        return interactive(&mut chat).await;
    };
    if let Err(e) = chat.turn(text, Mode::Once).await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
    Ok(())
}

/// The REPL: `> ` prompt on stdout, `/exit` or EOF quits, `/reset` clears the history,
/// errors print to stderr and the session continues.
async fn interactive(chat: &mut Chat) -> anyhow::Result<()> {
    let mut stdin = io::stdin().lock();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF (Ctrl-D)
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" {
            break;
        }
        if line == "/reset" {
            chat.history.clear();
            println!("(history cleared)");
            continue;
        }
        if let Err(e) = chat.turn(line, Mode::Interactive).await {
            eprintln!("error: {e:#}");
        }
    }
    Ok(())
}

/// Ends the dim `thinking: ` section with a reset and a newline, if one is open.
fn end_thinking(open: &mut bool, ansi: bool) {
    if *open {
        *open = false;
        if ansi {
            print!("\x1b[0m");
        }
        println!();
    }
}

/// The string a chunk delta carries under `key`, or "" when there is none.
fn delta_text<'a>(delta: Option<&'a Value>, key: &str) -> &'a str {
    delta
        .and_then(|d| d.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// Tokens per second over the turn.
fn rate(tokens: u64, elapsed: std::time::Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        tokens as f64 / secs
    } else {
        0.0
    }
}

/// The message for a failed HTTP response: the `error.message` from the JSON body when
/// present, else the status and a snippet of the body.
fn http_error(status: reqwest::StatusCode, body: &str) -> String {
    match serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").cloned())
    {
        Some(err) => error_message(&err),
        None => format!(
            "HTTP {status}: {}",
            body.chars().take(200).collect::<String>()
        ),
    }
}

/// The message of an `error` object: its `message` field, or the whole object.
fn error_message(err: &Value) -> String {
    err.get("message")
        .and_then(Value::as_str)
        .map_or_else(|| err.to_string(), str::to_string)
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use super::super::{Cli, Command};
    use super::*;

    /// Parses `argv` through the real `Cli`, so the registration and defaults are covered.
    fn args(argv: &[&str]) -> ChatArgs {
        let argv = ["legion", "chat"].into_iter().chain(argv.iter().copied());
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Chat(args) => args,
            _ => unreachable!("the parser returned a command other than chat"),
        }
    }

    #[test]
    fn base_url_drops_trailing_slash_and_v1() {
        assert_eq!(Chat::new(&args(&[])).base, "http://localhost:8090");
        assert_eq!(
            Chat::new(&args(&["--url", "http://x:1/"])).base,
            "http://x:1"
        );
        assert_eq!(
            Chat::new(&args(&["--url", "http://x:1/v1"])).base,
            "http://x:1"
        );
    }

    #[test]
    fn request_body_streams_with_usage_and_reasoning() {
        let mut chat = Chat::new(&args(&[
            "--thinking",
            "--temperature",
            "0.5",
            "--max-tokens",
            "32",
            "--system",
            "be nice",
        ]));
        chat.model = Some("m".into());
        chat.history.push(json!({"role": "user", "content": "hi"}));
        let b = chat.request_body();
        assert_eq!(b["model"], "m");
        assert_eq!(b["stream"], true);
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert_eq!(b["reasoning_effort"], "high");
        assert_eq!(b["temperature"], 0.5);
        assert_eq!(b["max_tokens"], 32);
        assert_eq!(b["messages"][0]["role"], "system");
        assert_eq!(b["messages"][1]["content"], "hi");
        // Without --thinking the server is told not to reason, and optional fields stay out.
        let chat = Chat::new(&args(&[]));
        let b = chat.request_body();
        assert_eq!(b["reasoning_effort"], "none");
        assert!(b.get("temperature").is_none());
        assert!(b.get("max_tokens").is_none());
    }

    #[test]
    fn http_error_prefers_the_json_message() {
        let body = r#"{"error":{"message":"invalid API key","type":"authentication_error"}}"#;
        assert_eq!(
            http_error(reqwest::StatusCode::UNAUTHORIZED, body),
            "invalid API key"
        );
        assert_eq!(
            http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "boom"),
            "HTTP 500 Internal Server Error: boom"
        );
        assert_eq!(
            error_message(&json!({"code": "overloaded"})),
            r#"{"code":"overloaded"}"#
        );
    }

    #[test]
    fn rate_guards_a_zero_duration() {
        assert_eq!(rate(10, std::time::Duration::from_secs(2)), 5.0);
        assert_eq!(rate(10, std::time::Duration::ZERO), 0.0);
    }
}
