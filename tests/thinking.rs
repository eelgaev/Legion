mod common;

use common::*;
use serde_json::{Value, json};

fn chat(extra: Value) -> String {
    let mut v = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    v.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    v.to_string()
}

#[tokio::test]
async fn enable_thinking_true_gives_reasoning() {
    let (s, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"chat_template_kwargs": {"enable_thinking": true}})),
    ))
    .await;
    assert_eq!(s, 200, "{v}");
    let r = &v["choices"][0]["message"]["reasoning_content"];
    assert!(!r.as_str().unwrap_or("").is_empty(), "{v}");
}

#[tokio::test]
async fn enable_thinking_false_beats_reasoning_effort() {
    let (s, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false},
        })),
    ))
    .await;
    assert_eq!(s, 200, "{v}");
    let r = &v["choices"][0]["message"]["reasoning_content"];
    assert!(r.is_null() || r.as_str() == Some(""), "{v}");
}

#[tokio::test]
async fn non_bool_enable_thinking_is_400() {
    let (s, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"chat_template_kwargs": {"enable_thinking": 1}})),
    ))
    .await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(v["error"]["param"], "chat_template_kwargs.enable_thinking");
}
