mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{Value, json};

#[tokio::test]
async fn chat_non_stream() {
    let (status, v) = json_of(post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "chat.completion");
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    assert_eq!(choice["message"]["content"], "You said: hi there");
    assert_eq!(choice["finish_reason"], "stop");
    assert!(
        choice["message"]
            .get("reasoning_content")
            .is_none_or(Value::is_null)
    );
    let u = &v["usage"];
    assert!(u["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(u["completion_tokens"], 4);
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn chat_max_tokens_length() {
    let (status, v) = json_of(post("/v1/chat/completions", chat(json!({"max_tokens": 2})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn chat_stop_string() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"stop": ["there"]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.contains("there"), "{content:?}");
    assert!(content.starts_with("You said:"), "{content:?}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_stream() {
    let (status, body) = send(post(
        "/v1/chat/completions",
        chat(json!({"stream": true, "stream_options": {"include_usage": true}})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");

    let events = sse_events(&body);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|(_, d)| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");

    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "You said: hi there");

    let usage = chunks.last().unwrap();
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 4);

    let last_choice = &chunks[chunks.len() - 2];
    assert_eq!(last_choice["choices"][0]["finish_reason"], "stop");
    // No other chunk carries a finish reason.
    let finishes = chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_string())
        .count();
    assert_eq!(finishes, 1);
}

#[tokio::test]
async fn chat_stream_without_usage() {
    let (_, body) = send(post("/v1/chat/completions", chat(json!({"stream": true})))).await;
    assert!(!body.contains("\"usage\":{"), "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn chat_reasoning() {
    let (_, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"reasoning_effort": "high"})),
    ))
    .await;
    let msg = &v["choices"][0]["message"];
    assert!(
        !msg["reasoning_content"].as_str().unwrap().is_empty(),
        "{v}"
    );
    assert_eq!(msg["content"], "You said: hi there");

    let (_, body) = send(post(
        "/v1/chat/completions",
        chat(json!({"reasoning_effort": "high", "stream": true})),
    ))
    .await;
    let reasoning: String = sse_events(&body)
        .iter()
        .filter(|(_, d)| d != "[DONE]")
        .filter_map(|(_, d)| {
            let c: Value = serde_json::from_str(d).unwrap();
            c["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .map(String::from)
        })
        .collect();
    assert_eq!(reasoning, msg["reasoning_content"].as_str().unwrap());
}

#[tokio::test]
async fn chat_validation() {
    let cases = [
        chat(json!({"temperature": 3})),
        chat(json!({"n": 2})),
        chat(json!({"max_tokens": 0})),
        json!({"model": "mock", "messages": []}),
    ];
    for body in cases {
        let (status, v) = json_of(post("/v1/chat/completions", &body)).await;
        assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn chat_malformed_body() {
    for body in ["{not json", "{}", r#"{"model": 5}"#] {
        let (status, v) = json_of(post("/v1/chat/completions", body)).await;
        assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    }
    // Wrong content type is also rendered as an OpenAI error.
    let req = Request::post("/v1/chat/completions")
        .body(Body::from("{}"))
        .unwrap();
    let (status, v) = json_of(req).await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
}

// --- completions ---

#[tokio::test]
async fn completions() {
    let (status, v) = json_of(post(
        "/v1/completions",
        json!({"model": "mock", "prompt": "hello", "max_tokens": 5}),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "text_completion");
    let text = v["choices"][0]["text"].as_str().unwrap().to_string();
    assert!(!text.is_empty());
    assert!(v["usage"]["total_tokens"].as_u64().unwrap() > 0);

    let (_, echoed) = json_of(post(
        "/v1/completions",
        json!({"model": "mock", "prompt": "hello", "max_tokens": 5, "echo": true}),
    ))
    .await;
    assert_eq!(echoed["choices"][0]["text"], format!("hello{text}"));

    let (_, body) = send(post(
        "/v1/completions",
        json!({"model": "mock", "prompt": "hello", "max_tokens": 5, "stream": true}),
    ))
    .await;
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    let streamed: String = sse_events(&body)
        .iter()
        .filter(|(_, d)| d != "[DONE]")
        .filter_map(|(_, d)| {
            let c: Value = serde_json::from_str(d).unwrap();
            c["choices"][0]["text"].as_str().map(String::from)
        })
        .collect();
    assert_eq!(streamed, text);
}

// --- messages ---
