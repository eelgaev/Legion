//! Request body size limit: `--max-request-mb` raises axum's 2 MB default so long prompts
//! pass, and bodies over the limit are refused with 413 in the caller's error shape.
mod common;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use legion::server::{AppState, ServerConfig};
use serde_json::{Value, json};

async fn json_to(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let (status, body) = send_to(app, req).await;
    (
        status,
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("bad JSON {body:?}: {e}")),
    )
}

fn app_with_limit(mb: usize) -> Router {
    app_with(|state| {
        let config = ServerConfig {
            max_request_mb: mb,
            ..ServerConfig::default()
        };
        AppState::with_config(state.engine, config)
    })
}

/// A chat body whose user content is `n` bytes long.
fn big_chat(n: usize) -> Value {
    chat(json!({"messages": [{"role": "user", "content": "x".repeat(n)}]}))
}

/// The 413 body OpenAI clients expect: `{"error": {"type": "invalid_request_error", ...}}`.
fn assert_openai_too_large(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
}

/// The 413 body Anthropic clients expect: `{"type": "error", "error": {...}}`.
fn assert_anthropic_too_large(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{v}");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "request_too_large", "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
}

#[tokio::test]
async fn default_limit_allows_a_3_mb_prompt() {
    let (status, v) = json_of(post("/v1/chat/completions", big_chat(3 * 1024 * 1024))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        text.starts_with("You said: xxx"),
        "unexpected echo {text:.60}"
    );
}

#[tokio::test]
async fn openai_rejects_bodies_over_the_limit_with_413() {
    let app = app_with_limit(1);
    let (status, v) = json_to(app, post("/v1/chat/completions", big_chat(1_500_000))).await;
    assert_openai_too_large(status, &v);
}

#[tokio::test]
async fn anthropic_rejects_bodies_over_the_limit_with_413() {
    let app = app_with_limit(1);
    let body = messages(json!({"messages": [{"role": "user", "content": "x".repeat(1_500_000)}]}));
    let (status, v) = json_to(app, post("/v1/messages", body)).await;
    assert_anthropic_too_large(status, &v);
}

#[tokio::test]
async fn bodies_under_a_small_limit_still_succeed() {
    let app = app_with_limit(1);
    let (status, v) = json_to(app, post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"].as_str().unwrap(),
        "You said: hi there"
    );
}

/// Only the length rejection becomes 413; a malformed body stays 400.
#[tokio::test]
async fn malformed_json_under_a_small_limit_is_still_400() {
    let app = app_with_limit(1);
    let (status, v) = json_to(app, post("/v1/chat/completions", "{not json")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
}
