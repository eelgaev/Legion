mod common;

use axum::http::StatusCode;
use common::*;

#[tokio::test]
async fn health() {
    let (status, v) = get("/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["status"], "ok");
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    assert!(!v["git"].as_str().unwrap_or("").is_empty());
    assert_eq!(v["models"], serde_json::json!(["mock"]));
}

#[tokio::test]
async fn models() {
    let (status, v) = get("/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "mock");

    let (status, v) = get("/v1/models/mock").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["id"], "mock");
    assert_eq!(v["object"], "model");

    let (status, v) = get("/v1/models/nope").await;
    assert_openai_error(status, &v, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn models_without_v1_is_the_same_list() {
    // llama.cpp's `/models`, which LLM proxies (simple-llm-proxy) discover models with.
    let (status, plain) = get("/models").await;
    assert_eq!(status, StatusCode::OK);
    let (_, v1) = get("/v1/models").await;
    assert_eq!(plain, v1);
}

// --- chat completions ---
