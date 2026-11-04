mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

fn completion(extra: Value) -> Value {
    let mut body = json!({"model": "mock", "prompt": "hi"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test]
async fn out_of_range_penalties_are_rejected_with_param() {
    let cases = [
        ("presence_penalty", json!(2.5)),
        ("presence_penalty", json!(-2.5)),
        ("frequency_penalty", json!(2.5)),
        ("frequency_penalty", json!(-2.5)),
        ("repetition_penalty", json!(0.0)),
        ("repetition_penalty", json!(-1.0)),
    ];
    for (field, value) in cases {
        let extra = json!({ field: value });
        let (status, v) = json_of(post("/v1/chat/completions", chat(extra.clone()))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "chat {field}={value}: {v}");
        assert_eq!(v["error"]["param"], field, "chat {field}={value}: {v}");

        let (status, v) = json_of(post("/v1/completions", completion(extra))).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "completions {field}={value}: {v}"
        );
        assert_eq!(
            v["error"]["param"], field,
            "completions {field}={value}: {v}"
        );
    }
}

#[tokio::test]
async fn in_range_penalties_are_accepted() {
    let extra = json!({
        "presence_penalty": -2.0,
        "frequency_penalty": 2.0,
        "repetition_penalty": 1.1,
    });
    let (status, v) = json_of(post("/v1/chat/completions", chat(extra.clone()))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let (status, v) = json_of(post("/v1/completions", completion(extra))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}
