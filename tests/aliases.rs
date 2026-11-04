//! `--served-model-name`: aliases are listed, resolved, and echoed back as requested.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::StatusCode;
use common::*;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, ServerConfig, router};
use serde_json::{Value, json};

fn aliased(names: &[&str]) -> Router {
    let engine = MockEngine::new(MockConfig {
        step: Duration::ZERO,
        ..Default::default()
    });
    let config = ServerConfig {
        served_model_names: names.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    router(AppState::with_config(Arc::new(engine), config))
}

async fn json_to(app: Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let (status, text) = send_to(app, post(uri, body)).await;
    let v = serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON {text:?}: {e}"));
    (status, v)
}

#[tokio::test]
async fn models_lists_aliases_not_engine_id() {
    let (status, text) = send_to(
        aliased(&["gpt-4o", "claude-x"]),
        axum::http::Request::get("/v1/models")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_str(&text).unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["gpt-4o", "claude-x"]);

    let (status, text) = send_to(
        aliased(&["gpt-4o"]),
        axum::http::Request::get("/v1/models/gpt-4o")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("\"gpt-4o\""));
    let (status, _) = send_to(
        aliased(&["gpt-4o"]),
        axum::http::Request::get("/v1/models/mock")
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn chat_echoes_alias() {
    let body = chat(json!({"model": "gpt-4o"}));
    let (status, v) = json_to(aliased(&["gpt-4o"]), "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["model"], "gpt-4o");
}

#[tokio::test]
async fn chat_stream_echoes_alias() {
    let body =
        chat(json!({"model": "gpt-4o", "stream": true, "stream_options": {"include_usage": true}}));
    let (status, text) = send_to(aliased(&["gpt-4o"]), post("/v1/chat/completions", body)).await;
    assert_eq!(status, StatusCode::OK);
    let mut chunks = 0;
    for (_, data) in sse_events(&text) {
        if data == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(&data).unwrap();
        assert_eq!(v["model"], "gpt-4o");
        chunks += 1;
    }
    assert!(chunks > 2);
}

#[tokio::test]
async fn completions_echoes_alias() {
    let body = json!({"model": "gpt-4o", "prompt": "hi"});
    let (status, v) = json_to(aliased(&["gpt-4o"]), "/v1/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["model"], "gpt-4o");
}

#[tokio::test]
async fn messages_echoes_alias() {
    let body = messages(json!({"model": "claude-x"}));
    let (status, v) = json_to(aliased(&["claude-x"]), "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["model"], "claude-x");
}

#[tokio::test]
async fn responses_echoes_alias() {
    let body = json!({"model": "gpt-4o", "input": "hi there"});
    let (status, v) = json_to(aliased(&["gpt-4o"]), "/v1/responses", body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["model"], "gpt-4o");
}

#[tokio::test]
async fn second_alias_and_engine_id_both_work() {
    for name in ["b", "mock"] {
        let body = chat(json!({"model": name}));
        let (status, v) = json_to(aliased(&["a", "b"]), "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["model"], name);
    }
}

#[tokio::test]
async fn no_aliases_unchanged() {
    let (_, v) = get("/v1/models").await;
    assert_eq!(v["data"][0]["id"], "mock");

    // The single-model fallback still resolves any name, and echoes the engine id.
    let body = chat(json!({"model": "whatever"}));
    let (status, v) = json_to(app(), "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["model"], "mock");
}
