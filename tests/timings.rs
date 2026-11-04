mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::StatusCode;
use common::*;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, router};
use serde_json::{Value, json};

/// A router over a mock engine with a 10 ms step, so prefill and decode take measurable time.
fn app() -> Router {
    router(AppState::new(Arc::new(MockEngine::new(MockConfig {
        step: Duration::from_millis(10),
        ..Default::default()
    }))))
}

/// Asserts `t` is a llama.cpp-shaped `timings` object with all six keys and sane values.
/// The mock answers "hi there" with "You said: hi there" (4 tokens over 10 ms steps),
/// so decode runs at about 100 tokens/s.
fn assert_full_timings(t: &Value, prompt_tokens: u64, completion_tokens: u64) {
    let obj = t
        .as_object()
        .unwrap_or_else(|| panic!("expected a timings object, got {t}"));
    for key in [
        "prompt_n",
        "prompt_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_ms",
        "predicted_per_second",
    ] {
        assert!(obj.contains_key(key), "missing `{key}` in {t}");
    }
    assert_eq!(t["prompt_n"], prompt_tokens, "{t}");
    assert!(t["prompt_ms"].as_f64().unwrap() > 0.0, "{t}");
    assert!(t["prompt_per_second"].as_f64().unwrap() > 0.0, "{t}");
    assert_eq!(t["predicted_n"], completion_tokens, "{t}");
    assert!(t["predicted_ms"].as_f64().unwrap() > 0.0, "{t}");
    let pps = t["predicted_per_second"].as_f64().unwrap();
    assert!(
        (20.0..200.0).contains(&pps),
        "predicted_per_second {pps} in {t}"
    );
    // An idle mock admits at once: a queue wait is reported, and it is short.
    let queue = t["queue_ms"].as_f64().unwrap();
    assert!((0.0..1000.0).contains(&queue), "queue_ms {queue} in {t}");
}

/// Parses an SSE body into chunk JSON values, dropping the final `[DONE]`.
fn chunks(body: &str) -> Vec<Value> {
    let events = sse_events(body);
    assert_eq!(events.last().unwrap().1, "[DONE]", "{body}");
    events[..events.len() - 1]
        .iter()
        .map(|(_, d)| serde_json::from_str(d).unwrap())
        .collect()
}

#[tokio::test]
async fn chat_non_stream_has_timings() {
    let (status, body) = send_to(app(), post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_full_timings(&v["timings"], 2, 4);
    assert_eq!(v["timings"]["prompt_n"], v["usage"]["prompt_tokens"]);
    assert_eq!(v["timings"]["predicted_n"], v["usage"]["completion_tokens"]);
}

#[tokio::test]
async fn chat_stream_usage_chunk_has_timings() {
    let (status, body) = send_to(
        app(),
        post(
            "/v1/chat/completions",
            chat(json!({"stream": true, "stream_options": {"include_usage": true}})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = chunks(&body);
    let usage_chunk = chunks
        .iter()
        .rev()
        .find(|c| c["usage"].is_object())
        .expect("a usage chunk");
    assert_full_timings(&usage_chunk["timings"], 2, 4);
    // With `include_usage` the finish chunk does not carry timings.
    let finish_chunk = chunks
        .iter()
        .rev()
        .find(|c| c["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert!(finish_chunk.get("timings").is_none(), "{finish_chunk}");
}

#[tokio::test]
async fn chat_stream_finish_chunk_has_timings_without_usage() {
    let (status, body) = send_to(
        app(),
        post("/v1/chat/completions", chat(json!({"stream": true}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(!body.contains("\"usage\":{"), "{body}");
    let chunks = chunks(&body);
    let finish_chunk = chunks
        .iter()
        .rev()
        .find(|c| c["choices"][0]["finish_reason"].is_string())
        .expect("a finish chunk");
    assert_full_timings(&finish_chunk["timings"], 2, 4);
}

#[tokio::test]
async fn completions_non_stream_has_timings() {
    let (status, body) = send_to(
        app(),
        post(
            "/v1/completions",
            json!({"model": "mock", "prompt": "hi there", "max_tokens": 100}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_full_timings(&v["timings"], 2, 4);
    assert_eq!(v["timings"]["predicted_n"], v["usage"]["completion_tokens"]);
}

#[tokio::test]
async fn completions_stream_final_chunk_has_timings() {
    let (status, body) = send_to(
        app(),
        post(
            "/v1/completions",
            json!({
                "model": "mock",
                "prompt": "hi there",
                "max_tokens": 100,
                "stream": true,
                "stream_options": {"include_usage": true},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = chunks(&body);
    let last = chunks.last().expect("a final chunk");
    assert_eq!(last["choices"][0]["finish_reason"], "stop", "{last}");
    assert!(last["usage"].is_object(), "{last}");
    assert_full_timings(&last["timings"], 2, 4);
}

#[tokio::test]
async fn anthropic_messages_have_no_timings() {
    let (status, body) = send_to(app(), post("/v1/messages", messages(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["type"], "message");
    assert!(v.get("timings").is_none(), "{v}");
}
