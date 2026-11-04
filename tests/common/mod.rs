//! Helpers shared by the integration tests. Each test file does `mod common;`.
#![allow(dead_code)]

pub mod serve;

use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

/// The router over a zero-delay mock engine.
pub fn app() -> Router {
    app_with(|state| state)
}

/// The router over a zero-delay mock engine, with `AppState` adjusted by `f`.
pub fn app_with(f: impl FnOnce(AppState) -> AppState) -> Router {
    let engine = MockEngine::new(MockConfig {
        step: Duration::ZERO,
        ..Default::default()
    });
    router(f(AppState::new(std::sync::Arc::new(engine))))
}

pub fn post(uri: &str, body: impl ToString) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

pub async fn send(req: Request<Body>) -> (StatusCode, String) {
    send_to(app(), req).await
}

pub async fn send_to(app: Router, req: Request<Body>) -> (StatusCode, String) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

pub async fn json_of(req: Request<Body>) -> (StatusCode, Value) {
    let (status, body) = send(req).await;
    let value = serde_json::from_str(&body).unwrap_or_else(|e| panic!("bad JSON {body:?}: {e}"));
    (status, value)
}

pub async fn get(uri: &str) -> (StatusCode, Value) {
    json_of(Request::get(uri).body(Body::empty()).unwrap()).await
}

/// Parses an SSE body into `(event name, data)` pairs.
pub fn sse_events(body: &str) -> Vec<(Option<String>, String)> {
    body.split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .map(|block| {
            let mut name = None;
            let mut data = Vec::new();
            for line in block.lines() {
                if let Some(v) = line.strip_prefix("event:") {
                    name = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("data:") {
                    data.push(v.trim_start().to_string());
                }
            }
            (name, data.join("\n"))
        })
        .collect()
}

pub fn chat(extra: Value) -> Value {
    let mut body = json!({"model": "mock", "messages": [{"role": "user", "content": "hi there"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

pub fn messages(extra: Value) -> Value {
    let mut body = json!({
        "model": "mock",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    });
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

pub fn assert_openai_error(status: StatusCode, v: &Value, expect: StatusCode) {
    assert_eq!(status, expect, "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
    assert!(v["error"]["type"].is_string(), "{v}");
}

pub fn assert_anthropic_error(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
}
