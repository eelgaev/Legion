mod common;

use axum::http::StatusCode;
use common::*;
use futures::StreamExt;
use http_body_util::BodyExt;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, router};
use serde_json::{Value, json};
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn unknown_model_resolves_to_the_only_model() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"model": "claude-sonnet-4-5"})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["model"], "mock");

    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"model": "gpt-4o"})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["model"], "mock");
}

// --- cancellation ---

#[tokio::test]
async fn dropped_stream_does_not_wedge_the_engine() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(2),
        max_batch: 1,
        ..Default::default()
    });
    let app = router(AppState::new(std::sync::Arc::new(engine)));

    let long = "word ".repeat(8000);
    let req = post(
        "/v1/chat/completions",
        chat(json!({"stream": true, "messages": [{"role": "user", "content": long}]})),
    );
    let resp = app.clone().oneshot(req).await.unwrap();
    let mut body = resp.into_body().into_data_stream();
    assert!(body.next().await.unwrap().is_ok());
    drop(body);

    // With max_batch 1 this only completes if the abandoned sequence was freed.
    let next = tokio::time::timeout(
        Duration::from_secs(10),
        app.oneshot(post("/v1/chat/completions", chat(json!({"max_tokens": 3})))),
    )
    .await
    .expect("engine hung after a client disconnect")
    .unwrap();
    assert_eq!(next.status(), StatusCode::OK);
    let bytes = next.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}
