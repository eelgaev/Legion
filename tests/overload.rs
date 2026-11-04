mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::StatusCode;
use common::*;
use futures::StreamExt;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::engine::{Engine, GenerateRequest, Prompt, TokenEvent, TokenStream};
use legion::server::{AppState, router};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn engine() -> Arc<MockEngine> {
    Arc::new(MockEngine::new(MockConfig {
        step: Duration::from_millis(30),
        max_batch: 1,
        max_waiting: 1,
        ..Default::default()
    }))
}

fn app_of(engine: Arc<MockEngine>) -> Router {
    router(AppState::new(engine))
}

fn request(cancel: CancellationToken) -> GenerateRequest {
    GenerateRequest {
        model: "mock".into(),
        // Long echo, so the first request keeps the only batch slot for the whole test.
        prompt: Prompt::Text("word ".repeat(2000)),
        sampling: Default::default(),
        max_tokens: 10_000,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel,
        scheduling: Default::default(),
    }
}

/// Starts a request and waits until the loop has admitted it (it then holds the batch slot).
async fn running(engine: &MockEngine) -> TokenStream {
    let mut s = engine.generate(request(Default::default())).unwrap();
    while !matches!(s.next().await, Some(TokenEvent::Started { .. })) {}
    s
}

#[tokio::test]
async fn third_request_is_refused_on_both_apis() {
    let engine = engine();
    let _first = running(&engine).await;
    let _second = engine.generate(request(Default::default())).unwrap();

    let app = app_of(engine.clone());
    let (status, body) = send_to(app.clone(), post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "rate_limit_error");

    let (status, body) = send_to(app, post("/v1/messages", messages(json!({})))).await;
    assert_eq!(status.as_u16(), 529, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "overloaded_error");
}

#[tokio::test]
async fn cancelled_waiting_request_frees_its_slot() {
    let engine = engine();
    let _first = running(&engine).await;
    let cancel = CancellationToken::new();
    let _second = engine.generate(request(cancel.clone())).unwrap();
    assert!(engine.generate(request(Default::default())).is_err());

    cancel.cancel();
    // The loop sweeps cancelled waiters once per step.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if engine.generate(request(Default::default())).is_ok() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "slot was never freed"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn finished_requests_free_their_slots() {
    let engine = Arc::new(MockEngine::new(MockConfig {
        step: Duration::ZERO,
        max_batch: 1,
        max_waiting: 1,
        ..Default::default()
    }));
    for _ in 0..5 {
        let mut s = engine
            .generate(GenerateRequest {
                max_tokens: 3,
                ..request(Default::default())
            })
            .unwrap();
        while s.next().await.is_some() {}
    }
}
