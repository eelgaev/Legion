mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

#[tokio::test]
async fn messages_non_stream() {
    let (status, v) = json_of(post("/v1/messages", messages(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "You said: hi");
    assert_eq!(v["stop_reason"], "end_turn");
    assert!(v["stop_sequence"].is_null());
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(v["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn messages_system_forms() {
    for system in [
        json!("be brief"),
        json!([{"type": "text", "text": "be brief"}]),
    ] {
        let (status, v) = json_of(post("/v1/messages", messages(json!({"system": system})))).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(v["content"][0]["text"], "You said: hi");
    }
}

#[tokio::test]
async fn messages_max_tokens_and_stop_sequences() {
    let (_, v) = json_of(post("/v1/messages", messages(json!({"max_tokens": 1})))).await;
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["usage"]["output_tokens"], 1);

    let (_, v) = json_of(post(
        "/v1/messages",
        messages(json!({"stop_sequences": ["said"]})),
    ))
    .await;
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "said");
    assert!(!v["content"][0]["text"].as_str().unwrap().contains("said"));
}

#[tokio::test]
async fn messages_thinking() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"thinking": {"type": "enabled", "budget_tokens": 1024}})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["type"], "thinking");
    assert!(!v["content"][0]["thinking"].as_str().unwrap().is_empty());
    assert_eq!(v["content"][0]["signature"], "");
    assert_eq!(v["content"][1]["type"], "text");
    assert_eq!(v["content"][1]["text"], "You said: hi");
}

#[tokio::test]
async fn messages_stream_event_order() {
    let (status, body) = send(post("/v1/messages", messages(json!({"stream": true})))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!body.contains("[DONE]"));
    let events = sse_events(&body);
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_deref().unwrap()).collect();
    assert_eq!(
        names,
        [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    // The `type` field matches the event name everywhere.
    for (name, data) in &events {
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["type"].as_str(), name.as_deref());
    }
    let start: Value = serde_json::from_str(&events[0].1).unwrap();
    assert_eq!(start["message"]["content"], json!([]));
    assert!(start["message"]["stop_reason"].is_null());
    assert!(start["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(start["message"]["usage"]["output_tokens"], 0);

    let text: String = events
        .iter()
        .filter_map(|(_, d)| {
            let v: Value = serde_json::from_str(d).unwrap();
            v["delta"]["text"].as_str().map(String::from)
        })
        .collect();
    assert_eq!(text, "You said: hi");

    let delta: Value = serde_json::from_str(&events[events.len() - 2].1).unwrap();
    assert_eq!(delta["delta"]["stop_reason"], "end_turn");
    assert_eq!(delta["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn messages_stream_thinking() {
    let (_, body) = send(post(
        "/v1/messages",
        messages(json!({"stream": true, "thinking": {"type": "enabled", "budget_tokens": 1024}})),
    ))
    .await;
    let events: Vec<(String, Value)> = sse_events(&body)
        .into_iter()
        .map(|(n, d)| (n.unwrap(), serde_json::from_str(&d).unwrap()))
        .collect();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    let starts: Vec<&Value> = events
        .iter()
        .filter(|(n, _)| n == "content_block_start")
        .map(|(_, v)| v)
        .collect();
    assert_eq!(starts.len(), 2, "{names:?}");
    assert_eq!(starts[0]["index"], 0);
    assert_eq!(starts[0]["content_block"]["type"], "thinking");
    assert_eq!(starts[1]["index"], 1);
    assert_eq!(starts[1]["content_block"]["type"], "text");
    assert_eq!(
        names.iter().filter(|n| **n == "content_block_stop").count(),
        2
    );

    // Each block is closed before the next one opens.
    let first_stop = names
        .iter()
        .position(|n| *n == "content_block_stop")
        .unwrap();
    let second_start = names
        .iter()
        .rposition(|n| *n == "content_block_start")
        .unwrap();
    assert!(first_stop < second_start);

    let text: String = events
        .iter()
        .filter_map(|(_, v)| v["delta"]["text"].as_str())
        .collect();
    assert_eq!(text, "You said: hi");
    assert!(
        events
            .iter()
            .any(|(_, v)| v["delta"]["type"] == "thinking_delta" && v["index"] == 0)
    );
    assert_eq!(names.last(), Some(&"message_stop"));
}

#[tokio::test]
async fn messages_errors() {
    let mut body = messages(json!({}));
    body.as_object_mut().unwrap().remove("max_tokens");
    let (status, v) = json_of(post("/v1/messages", body)).await;
    assert_anthropic_error(status, &v);

    let (status, v) = json_of(post("/v1/messages", messages(json!({"temperature": 1.5})))).await;
    assert_anthropic_error(status, &v);

    let (status, v) = json_of(post("/v1/messages", "{oops")).await;
    assert_anthropic_error(status, &v);
}

#[tokio::test]
async fn messages_tool_round_trip() {
    let body = messages(json!({
        "tools": [{
            "name": "get_weather",
            "description": "Weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }],
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                 "content": [{"type": "text", "text": "oops"}]},
                {"type": "text", "text": "thanks"},
            ]},
        ],
    }));
    let (status, v) = json_of(post("/v1/messages", body)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["type"], "text");
}
