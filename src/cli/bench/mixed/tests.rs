use super::*;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    mixed: MixedArgs,
}

fn args(extra: &[&str]) -> MixedArgs {
    let argv = std::iter::once("x").chain(extra.iter().copied());
    Cli::try_parse_from(argv).unwrap().mixed
}

fn plan(extra: &[&str]) -> MixedPlan {
    args(extra).plan(0).unwrap()
}

fn outcome(start_ttft_ms: u64, itl_ms: &[u64], tokens: u64) -> RequestOutcome {
    let itl: Vec<Duration> = itl_ms.iter().map(|&m| Duration::from_millis(m)).collect();
    let ttft = Duration::from_millis(start_ttft_ms);
    RequestOutcome {
        ok: true,
        ttft: Some(ttft),
        e2e: ttft + itl.iter().sum::<Duration>(),
        itl,
        output_tokens: tokens,
        ..Default::default()
    }
}

#[test]
fn defaults_match_the_issue() {
    let p = plan(&[]);
    assert_eq!(p.chat_streams, 4);
    assert_eq!((p.chat_prompt_tokens, p.chat_max_tokens), (512, 256));
    assert_eq!((p.long_prompt_tokens, p.long_max_tokens), (16384, 64));
    assert_eq!(p.arrivals, Arrivals::Every { secs: 10.0 });
    assert_eq!((p.warmup_s, p.duration_s), (10.0, 120.0));
    assert_eq!((p.priority_chat, p.priority_long), (None, None));
}

#[test]
fn rate_replaces_period_and_zero_turns_longs_off() {
    assert_eq!(
        plan(&["--long-rate", "0.5"]).arrivals,
        Arrivals::Poisson { per_sec: 0.5 }
    );
    assert_eq!(plan(&["--long-rate", "0"]).arrivals, Arrivals::None);
    assert_eq!(plan(&["--long-every", "0"]).arrivals, Arrivals::None);
    let both = ["x", "--long-rate", "1", "--long-every", "5"];
    assert!(Cli::try_parse_from(both).is_err());
}

#[test]
fn priorities_parse_negative() {
    let p = plan(&["--priority-chat", "-1", "--priority-long", "5"]);
    assert_eq!((p.priority_chat, p.priority_long), (Some(-1), Some(5)));
}

#[test]
fn bad_plans_are_rejected() {
    for bad in [
        &["--chat-max-tokens", "0"][..],
        &["--long-prompt-tokens", "0"],
        &["--duration", "0"],
        &["--warmup=-1"],
        &["--long-every=-2"],
        &["--long-rate", "NaN"],
        &["--chat-streams", "0", "--long-every", "0"],
        &["--declared-max-tokens", "0"],
    ] {
        assert!(args(bad).plan(0).is_err(), "{bad:?}");
    }
    // Long prompts only is fine.
    assert!(args(&["--chat-streams", "0"]).plan(0).is_ok());
}

#[test]
fn periodic_schedule_starts_after_one_period() {
    let every = Arrivals::Every { secs: 10.0 };
    assert_eq!(
        arrival_schedule(every, 40.0, 0),
        vec![10.0, 20.0, 30.0],
        "an arrival at the horizon is not sent"
    );
    assert_eq!(
        arrival_schedule(every, 40.0, 0),
        arrival_schedule(every, 40.0, 9)
    );
    assert!(arrival_schedule(Arrivals::None, 1e6, 0).is_empty());
}

#[test]
fn poisson_schedule_follows_the_seed() {
    let p = Arrivals::Poisson { per_sec: 0.5 };
    let a = arrival_schedule(p, 2000.0, 7);
    assert_eq!(a, arrival_schedule(p, 2000.0, 7));
    assert_ne!(a, arrival_schedule(p, 2000.0, 8));
    assert!(a.windows(2).all(|w| w[0] < w[1]));
    assert!(a.iter().all(|&t| t > 0.0 && t < 2000.0));
    // About rate x horizon = 1000 arrivals (sd ~32).
    assert!((850..1150).contains(&a.len()), "{}", a.len());
    // A longer horizon only appends.
    let b = arrival_schedule(p, 1000.0, 7);
    assert_eq!(a[..b.len()], b[..]);
}

#[test]
fn texts_are_deterministic_and_distinct() {
    let p = plan(&["--chat-prompt-tokens", "90", "--long-prompt-tokens", "300"]);
    assert_eq!(chat_prompt(&p, 1, 2), chat_prompt(&p, 1, 2));
    assert_eq!(long_prompt(&p, 3), long_prompt(&p, 3));
    assert_ne!(chat_prompt(&p, 0, 0), chat_prompt(&p, 1, 0));
    assert_ne!(chat_prompt(&p, 0, 0), chat_prompt(&p, 0, 1));
    assert_ne!(long_prompt(&p, 0), long_prompt(&p, 1));
    let other = MixedPlan {
        seed: 1,
        ..p.clone()
    };
    assert_ne!(chat_prompt(&other, 0, 0), chat_prompt(&p, 0, 0));
    assert_ne!(long_prompt(&other, 0), long_prompt(&p, 0));
    assert_ne!(
        text_seed(0, Class::Chat, 0),
        text_seed(0, Class::Long, 0),
        "classes draw different texts"
    );
    // The prefill workload's format: N / 3 word+number items, then the instruction.
    let long = long_prompt(&p, 0);
    let body = long.strip_suffix("\nSummarize in one line.").unwrap();
    assert_eq!(body.split(' ').count(), 100);
}

#[test]
fn class_stats_reuse_the_bench_maths() {
    // ITL 10 ms x 98 and one 50 ms stall: p50 and p90 10; p99 is the 99th of 99 gaps (nearest
    // rank), the stall, as is the max.
    let mut gaps = vec![10; 98];
    gaps.push(50);
    let o = outcome(100, &gaps, 100);
    let c = summarize_class(std::slice::from_ref(&o), 2.0);
    assert_eq!((c.requests, c.ok, c.failed), (1, 1, 0));
    assert_eq!(c.output_tokens, 100);
    assert!((c.output_tok_s - 50.0).abs() < 1e-9);
    assert_eq!(c.ttft_ms.p50, 100.0);
    assert_eq!(c.itl_ms.p50, 10.0);
    assert_eq!(c.itl_ms.p90, 10.0);
    assert_eq!(c.itl_ms.p99, 50.0);
    assert_eq!(c.itl_max_ms, 50.0);
    // per-stream: 99 tokens over the 1030 ms after the first.
    assert!((c.per_stream_tok_s - 99.0 / 1.03).abs() < 1e-9);
}

#[test]
fn report_leaves_out_the_warmup_and_takes_the_ratio() {
    let p = plan(&["--warmup", "1", "--duration", "2"]);
    let chat = |start_s: f64, itl: &[u64]| Sample {
        class: Class::Chat,
        start_s,
        outcome: outcome(20, itl, itl.len() as u64 + 1),
    };
    let samples = vec![
        // In the warm-up: a huge gap that must not count.
        chat(0.5, &[900, 900]),
        chat(1.0, &[10, 10, 10, 40]),
        chat(2.5, &[10, 10, 10, 10]),
        Sample {
            class: Class::Long,
            start_s: 2.0,
            outcome: RequestOutcome {
                error: Some("HTTP 503".into()),
                ..Default::default()
            },
        },
    ];
    let r = report(&p, samples);
    assert_eq!(r.warmup_requests, 1);
    assert_eq!((r.chat.requests, r.chat.ok), (2, 2));
    assert_eq!(r.chat.itl_max_ms, 40.0);
    assert_eq!(r.chat.itl_ms.p50, 10.0);
    assert_eq!(r.chat.itl_ms.p99, 40.0);
    assert_eq!(r.chat_itl_p99_over_p50, Some(4.0));
    assert_eq!((r.long.requests, r.long.failed), (1, 1));
    assert_eq!(r.long.first_error.as_deref(), Some("HTTP 503"));
    // The last chat ends at 2.5 + 0.06 s, before the horizon (3 s): window = duration.
    assert!((r.window_s - 2.0).abs() < 1e-9);
    let table = format_table(&r);
    assert!(table.contains("chat ITL p99 / p50: 4.00"), "{table}");
    assert!(
        table.contains("long: 1 failed, first error: HTTP 503"),
        "{table}"
    );
    let readme = format_readme(&r);
    assert!(readme.contains("| chat | 2 | 2 | 0 |"), "{readme}");
}

#[test]
fn no_chat_samples_has_no_ratio() {
    let p = plan(&["--chat-streams", "0"]);
    let r = report(&p, Vec::new());
    assert_eq!(r.chat_itl_p99_over_p50, None);
    assert!(format_table(&r).contains("no chat ITL samples"));
}

#[test]
fn priority_is_sent_only_when_set() {
    let cfg = BenchConfig {
        url: "http://x:1".into(),
        model: "m".into(),
        api: Api::OpenAi,
        prompt_tokens: 1,
        gen_tokens: 8,
        shared_prefix_tokens: 0,
        seed: 0,
        api_key: None,
        min_tokens: false,
        priority: None,
        declared_max_tokens: None,
    };
    let body = bench::request_body(&cfg, "hi");
    assert!(body.get("priority").is_none());
    assert_eq!(body["temperature"], 0);
    let with = BenchConfig {
        priority: Some(-3),
        ..cfg
    };
    assert_eq!(bench::request_body(&with, "hi")["priority"], -3);
    assert_eq!(bench::request_body(&with, "hi")["max_tokens"], 8);
    assert_eq!(bench::request_body(&with, "hi")["ignore_eos"], true);
    let declared = BenchConfig {
        declared_max_tokens: Some(32768),
        ..with
    };
    let body = bench::request_body(&declared, "hi");
    assert_eq!(body["max_tokens"], 32768);
    // A declared length runs to the reply's own end (or the hang-up), not past it.
    assert!(body.get("ignore_eos").is_none());
}

#[test]
fn declared_max_tokens_is_described_and_kept_in_the_plan() {
    let p = plan(&["--declared-max-tokens", "32768"]);
    assert_eq!(p.declared_max_tokens, Some(32768));
    assert!(
        describe(&p).ends_with("; max_tokens 32768 sent"),
        "{}",
        describe(&p)
    );
    let json = serde_json::to_value(&p).unwrap();
    assert_eq!(json["declared_max_tokens"], 32768);
    // Unset: the plan's JSON is as before.
    let json = serde_json::to_value(plan(&[])).unwrap();
    assert!(json.get("declared_max_tokens").is_none());
}

/// A client that declares a large `max_tokens` hangs up after its own length: the run takes
/// that many tokens per request, not the declared ones.
#[tokio::test]
async fn declared_requests_hang_up_after_their_length() {
    use legion::engine::mock::{MockConfig, MockEngine};
    use legion::server::{AppState, router};
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(1),
        ..Default::default()
    });
    let app = router(AppState::new(std::sync::Arc::new(engine)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let cfg = BenchConfig {
        url: format!("http://{addr}"),
        model: "mock".into(),
        api: Api::OpenAi,
        prompt_tokens: 20,
        gen_tokens: 6,
        shared_prefix_tokens: 0,
        seed: 0,
        api_key: None,
        min_tokens: false,
        priority: None,
        declared_max_tokens: Some(100_000),
    };
    let client = bench::client().unwrap();
    let t = std::time::Instant::now();
    // The mock echoes the prompt, one word per token: a reply longer than 6 tokens.
    let long = vec!["word"; 50].join(" ");
    let out = bench::run_request(&client, &cfg, &long).await;
    assert!(out.ok, "{out:?}");
    assert_eq!(out.output_tokens, 6);
    // A reply that ends before the client's length ends there (no `ignore_eos` filler).
    let out = bench::run_request(&client, &cfg, "hi").await;
    server.abort();
    assert!(out.ok, "{out:?}");
    assert!(out.output_tokens < 6, "{out:?}");
    // 100k mock steps of 1 ms would take minutes.
    assert!(t.elapsed() < Duration::from_secs(20));
}

/// The whole workload against the mock engine on loopback: both classes run, the warm-up is
/// split off, and the chat streams stop at the horizon.
#[tokio::test]
async fn drives_a_mock_server() {
    use legion::engine::mock::{MockConfig, MockEngine};
    use legion::server::{AppState, router};
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(2),
        ..Default::default()
    });
    let app = router(AppState::new(std::sync::Arc::new(engine)));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    let p = plan(&[
        "--chat-streams",
        "2",
        "--chat-prompt-tokens",
        "30",
        "--chat-max-tokens",
        "8",
        "--long-prompt-tokens",
        "300",
        "--long-max-tokens",
        "4",
        "--long-every",
        "0.2",
        "--warmup",
        "0.2",
        "--duration",
        "0.6",
        "--priority-long",
        "1",
    ]);
    let base = BenchConfig {
        url: format!("http://{addr}"),
        model: "mock".into(),
        api: Api::OpenAi,
        prompt_tokens: 0,
        gen_tokens: 0,
        shared_prefix_tokens: 0,
        seed: 0,
        api_key: None,
        min_tokens: false,
        priority: None,
        declared_max_tokens: None,
    };
    let client = bench::client().unwrap();
    let samples = drive(&client, &base, &p).await;
    server.abort();
    let longs: Vec<&Sample> = samples.iter().filter(|s| s.class == Class::Long).collect();
    assert_eq!(longs.len(), 3, "arrivals at 0.2, 0.4, 0.6 s");
    assert!(samples.iter().all(|s| s.outcome.ok), "{samples:?}");
    assert!(samples.iter().all(|s| s.start_s < 0.8));
    let r = report(&p, samples);
    assert!(r.warmup_requests > 0);
    assert_eq!(r.long.requests, 3);
    assert_eq!(r.long.output_tokens, 12);
    assert!(r.chat.requests > 0);
    assert_eq!(r.chat.output_tokens, 8 * r.chat.requests as u64);
    assert!(r.chat.itl_ms.p50 > 0.0);
    assert!(r.chat_itl_p99_over_p50.is_some());
}
