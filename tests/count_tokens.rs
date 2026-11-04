mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

#[tokio::test]
async fn matches_messages_usage() {
    for extra in [
        json!({}),
        json!({"system": "be brief and kind"}),
        json!({"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}),
    ] {
        let body = messages(extra);
        let (status, counted) = json_of(post("/v1/messages/count_tokens", body.clone())).await;
        assert_eq!(status, StatusCode::OK, "{counted}");
        let (_, real) = json_of(post("/v1/messages", body)).await;
        assert!(counted["input_tokens"].as_u64().unwrap() > 0);
        assert_eq!(counted["input_tokens"], real["usage"]["input_tokens"]);
    }
}

#[tokio::test]
async fn missing_max_tokens_is_fine() {
    let mut body = messages(json!({}));
    body.as_object_mut().unwrap().remove("max_tokens");
    let (status, v) = json_of(post("/v1/messages/count_tokens", body.clone())).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(v["input_tokens"].as_u64().unwrap() > 0);
    // ... while /v1/messages still requires it.
    let (status, _) = json_of(post("/v1/messages", body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

fn assert_anthropic_400(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].is_string());
}

#[tokio::test]
async fn invalid_requests_are_anthropic_errors() {
    let (status, v) = json_of(post(
        "/v1/messages/count_tokens",
        messages(json!({"messages": []})),
    ))
    .await;
    assert_anthropic_400(status, &v);
    let (status, v) = json_of(post("/v1/messages/count_tokens", json!({"model": "mock"}))).await;
    assert_anthropic_400(status, &v);
    let (status, v) = json_of(post("/v1/messages/count_tokens", json!([1]))).await;
    assert_anthropic_400(status, &v);
}
