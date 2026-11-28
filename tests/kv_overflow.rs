//! A request whose prompt and `max_tokens` need more KV cache than the whole cache holds is the
//! client's error: HTTP 400 `invalid_request_error` on every front end, streaming or not (#125),
//! with OpenAI's `context_length_exceeded` code, as a prompt over the context has.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::StatusCode;
use common::*;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, router};
use serde_json::{Value, json};

fn app() -> Router {
    router(AppState::new(Arc::new(MockEngine::new(MockConfig {
        step: Duration::ZERO,
        kv_tokens: Some(64),
        ..Default::default()
    }))))
}

async fn call(uri: &str, body: Value) -> (StatusCode, Value) {
    let (status, body) = send_to(app(), post(uri, body)).await;
    let v = serde_json::from_str(&body).unwrap_or_else(|e| panic!("bad JSON {body:?}: {e}"));
    (status, v)
}

fn openai_requests(max_tokens: u32, stream: bool) -> Vec<(&'static str, Value)> {
    vec![
        (
            "/v1/chat/completions",
            json!({"model": "mock", "max_tokens": max_tokens, "stream": stream,
                   "messages": [{"role": "user", "content": "hi there"}]}),
        ),
        (
            "/v1/completions",
            json!({"model": "mock", "max_tokens": max_tokens, "stream": stream,
                   "prompt": "hi there"}),
        ),
        (
            "/v1/responses",
            json!({"model": "mock", "max_output_tokens": max_tokens, "stream": stream,
                   "input": "hi there"}),
        ),
    ]
}

#[tokio::test]
async fn openai_endpoints_answer_400() {
    for stream in [false, true] {
        for (uri, body) in openai_requests(1000, stream) {
            let (status, v) = call(uri, body).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{uri} stream={stream}: {v}"
            );
            assert_eq!(v["error"]["type"], "invalid_request_error", "{uri}: {v}");
            assert_eq!(v["error"]["code"], "context_length_exceeded", "{uri}: {v}");
            let message = v["error"]["message"].as_str().unwrap();
            assert!(message.contains("the cache holds 64"), "{uri}: {v}");
        }
        // A request that fits runs.
        for (uri, body) in openai_requests(8, stream) {
            let (status, _) = send_to(app(), post(uri, body)).await;
            assert_eq!(status, StatusCode::OK, "{uri} stream={stream}");
        }
    }
}

#[tokio::test]
async fn a_prompt_over_the_context_is_context_length_exceeded() {
    let app = || {
        router(AppState::new(Arc::new(MockEngine::new(MockConfig {
            step: Duration::ZERO,
            context_length: 1,
            ..Default::default()
        }))))
    };
    for stream in [false, true] {
        for (uri, body) in openai_requests(8, stream) {
            let (status, body) = send_to(app(), post(uri, body)).await;
            let v: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{uri} stream={stream}: {v}"
            );
            assert_eq!(v["error"]["type"], "invalid_request_error", "{uri}: {v}");
            assert_eq!(v["error"]["code"], "context_length_exceeded", "{uri}: {v}");
            assert!(
                v["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("the context is 1"),
                "{uri}: {v}"
            );
        }
    }
}

#[tokio::test]
async fn anthropic_messages_answer_400() {
    for stream in [false, true] {
        let (status, v) = call(
            "/v1/messages",
            messages(json!({"max_tokens": 1000, "stream": stream})),
        )
        .await;
        assert_anthropic_error(status, &v);
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("the cache holds 64"),
            "{v}"
        );
        let (status, _) = send_to(
            app(),
            post(
                "/v1/messages",
                messages(json!({"max_tokens": 8, "stream": stream})),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "stream={stream}");
    }
}
