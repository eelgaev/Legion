//! Request priority (`--scheduling-policy`) and fair share (`--fair-share`) through the HTTP
//! routes (#625), on a mock engine with one batch slot so every later request queues.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::engine::queue::SchedulingPolicy;
use legion::server::fair_share::FairShare;
use legion::server::{AppState, ServerConfig, router};
use serde_json::{Value, json};

fn app(scheduling: SchedulingPolicy, fair_share: FairShare, api_keys: &[&str]) -> Router {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(5),
        max_batch: 1,
        scheduling,
        ..Default::default()
    });
    let config = ServerConfig {
        api_keys: api_keys.iter().map(|k| k.to_string()).collect(),
        fair_share,
        ..Default::default()
    };
    router(AppState::with_config(Arc::new(engine), config))
}

fn chat_request(body: Value, key: Option<&str>) -> Request<Body> {
    let mut req = post("/v1/chat/completions", chat(body));
    if let Some(key) = key {
        let value = format!("Bearer {key}").parse().unwrap();
        req.headers_mut().insert("authorization", value);
    }
    req
}

/// Sends the requests one after another, 2 ms apart so they reach the queue in this order,
/// and returns each one's `timings.queue_ms`.
async fn queue_waits(app: &Router, reqs: Vec<(Value, Option<&'static str>)>) -> Vec<f64> {
    let mut tasks = Vec::new();
    for (body, key) in reqs {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            let (status, body) = send_to(app, chat_request(body, key)).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            let v: Value = serde_json::from_str(&body).unwrap();
            v["timings"]["queue_ms"].as_f64().unwrap()
        }));
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let mut waits = Vec::new();
    for t in tasks {
        waits.push(t.await.unwrap());
    }
    waits
}

/// A request that holds the only slot, then three at priority 10 and one at -1.
fn mixed_priorities() -> Vec<(Value, Option<&'static str>)> {
    vec![
        (json!({}), None),
        (json!({"priority": 10}), None),
        (json!({"priority": 10}), None),
        (json!({"priority": 10}), None),
        (json!({"priority": -1}), None),
    ]
}

#[tokio::test]
async fn fcfs_ignores_the_priority_field() {
    let app = app(SchedulingPolicy::Fcfs, FairShare::Off, &[]);
    let w = queue_waits(&app, mixed_priorities()).await;
    assert!(w[4] > w[1] && w[4] > w[2] && w[4] > w[3], "{w:?}");
}

#[tokio::test]
async fn priority_admits_the_lowest_value_first() {
    let app = app(SchedulingPolicy::Priority, FairShare::Off, &[]);
    let w = queue_waits(&app, mixed_priorities()).await;
    assert!(w[4] < w[1] && w[4] < w[2] && w[4] < w[3], "{w:?}");
    // Equal priorities keep their arrival order.
    assert!(w[1] < w[2] && w[2] < w[3], "{w:?}");
}

#[tokio::test]
async fn fair_share_by_api_key_lets_the_second_key_in() {
    let app = app(SchedulingPolicy::Fcfs, FairShare::ApiKey, &["ka", "kb"]);
    let reqs = vec![
        (json!({}), Some("ka")),
        (json!({}), Some("ka")),
        (json!({}), Some("ka")),
        (json!({}), Some("ka")),
        (json!({}), Some("kb")),
    ];
    let w = queue_waits(&app, reqs).await;
    // kb goes right after the request holding the slot, ahead of ka's backlog.
    assert!(w[4] < w[2] && w[4] < w[3], "{w:?}");
}

#[tokio::test]
async fn fair_share_off_keeps_arrival_order_across_keys() {
    let app = app(SchedulingPolicy::Fcfs, FairShare::Off, &["ka", "kb"]);
    let reqs = vec![
        (json!({}), Some("ka")),
        (json!({}), Some("ka")),
        (json!({}), Some("ka")),
        (json!({}), Some("kb")),
    ];
    let w = queue_waits(&app, reqs).await;
    assert!(w[3] > w[1] && w[3] > w[2], "{w:?}");
}

#[tokio::test]
async fn fair_share_by_user_field() {
    let app = app(SchedulingPolicy::Fcfs, FairShare::User, &[]);
    let reqs = vec![
        (json!({"user": "alice"}), None),
        (json!({"user": "alice"}), None),
        (json!({"user": "alice"}), None),
        (json!({"user": "alice"}), None),
        (json!({"user": "bob"}), None),
    ];
    let w = queue_waits(&app, reqs).await;
    assert!(w[4] < w[2] && w[4] < w[3], "{w:?}");
}

#[tokio::test]
async fn every_front_end_accepts_priority_and_user() {
    let app = app(SchedulingPolicy::Priority, FairShare::User, &[]);
    let (status, body) = send_to(
        app.clone(),
        post(
            "/v1/completions",
            json!({"model": "mock", "prompt": "hi", "priority": -3, "user": "u"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send_to(
        app.clone(),
        post(
            "/v1/messages",
            messages(json!({"priority": 2, "metadata": {"user_id": "u"}})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = send_to(
        app,
        post(
            "/v1/responses",
            json!({"model": "mock", "input": "hi", "priority": 1, "user": "u"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
