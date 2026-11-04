mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

/// One OpenAI-format tool definition for `get_weather`.
fn openai_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }])
}

/// One Anthropic-format tool definition for `get_weather`.
fn anthropic_tools() -> Value {
    json!([{
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }])
}

const ARGS: &str = r#"{"city": "Paris"}"#;
const TRIGGER: &str = "/tool get_weather {\"city\": \"Paris\"}";

#[tokio::test]
async fn openai_non_stream_tool_call() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"tools": openai_tools(), "messages": [{"role": "user", "content": TRIGGER}]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls", "{v}");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function", "{v}");
    assert!(call["id"].as_str().unwrap().starts_with("call_"), "{v}");
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], ARGS);
    assert_eq!(choice["message"]["content"], "", "{v}");
}

#[tokio::test]
async fn openai_non_stream_tool_call_with_reasoning() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({
            "tools": openai_tools(),
            "messages": [{"role": "user", "content": TRIGGER}],
            "reasoning_effort": "high",
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let msg = &v["choices"][0]["message"];
    assert!(
        !msg["reasoning_content"].as_str().unwrap().is_empty(),
        "{v}"
    );
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn openai_stream_tool_call() {
    let (status, body) = send(post(
        "/v1/chat/completions",
        chat(json!({
            "tools": openai_tools(),
            "messages": [{"role": "user", "content": TRIGGER}],
            "stream": true,
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&body);
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|(_, d)| serde_json::from_str(d).unwrap())
        .collect();

    let tool_deltas: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["delta"]["tool_calls"].is_array())
        .collect();
    assert_eq!(tool_deltas.len(), 1, "{chunks:?}");
    let tc = &tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tc["index"], 0);
    assert_eq!(tc["type"], "function");
    assert!(tc["id"].as_str().unwrap().starts_with("call_"), "{tc}");
    assert_eq!(tc["function"]["name"], "get_weather");
    assert_eq!(tc["function"]["arguments"], ARGS);

    let finishes: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_string())
        .collect();
    assert_eq!(finishes.len(), 1, "{chunks:?}");
    assert_eq!(finishes[0]["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn anthropic_non_stream_tool_use() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({
            "tools": anthropic_tools(),
            "messages": [{"role": "user", "content": TRIGGER}],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["stop_reason"], "tool_use", "{v}");
    let block = &v["content"][0];
    assert_eq!(block["type"], "tool_use", "{v}");
    assert_eq!(v["content"].as_array().unwrap().len(), 1, "{v}");
    assert!(block["id"].as_str().unwrap().starts_with("call_"), "{v}");
    assert_eq!(block["name"], "get_weather");
    assert_eq!(block["input"], json!({"city": "Paris"}));
}

#[tokio::test]
async fn anthropic_stream_tool_use() {
    let (status, body) = send(post(
        "/v1/messages",
        messages(json!({
            "tools": anthropic_tools(),
            "messages": [{"role": "user", "content": TRIGGER}],
            "stream": true,
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK);
    let events: Vec<(String, Value)> = sse_events(&body)
        .into_iter()
        .map(|(n, d)| (n.unwrap(), serde_json::from_str(&d).unwrap()))
        .collect();
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "message_start",
            "ping",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );

    let start = &events[2].1;
    assert_eq!(start["content_block"]["type"], "tool_use");
    assert_eq!(start["content_block"]["name"], "get_weather");
    assert!(
        start["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("call_")
    );
    let delta = &events[3].1;
    assert_eq!(delta["index"], start["index"]);
    assert_eq!(delta["delta"]["type"], "input_json_delta");
    assert_eq!(delta["delta"]["partial_json"], ARGS);
    assert_eq!(events[4].1["index"], start["index"]);
    assert_eq!(events[5].1["delta"]["stop_reason"], "tool_use");
}

#[tokio::test]
async fn openai_tool_result_follow_up() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({
            "tools": openai_tools(),
            "messages": [
                {"role": "user", "content": TRIGGER},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": ARGS}},
                ]},
                {"role": "tool", "tool_call_id": "call_0", "content": "sunny, 24C"},
                {"role": "user", "content": "thanks"},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "You said: thanks");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn anthropic_tool_result_follow_up() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({
            "tools": anthropic_tools(),
            "messages": [
                {"role": "user", "content": TRIGGER},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_0", "name": "get_weather", "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_0", "content": "sunny, 24C"},
                    {"type": "text", "text": "all set, thanks"},
                ]},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "You said: all set, thanks");
    assert_eq!(v["stop_reason"], "end_turn");
}

#[tokio::test]
async fn openai_tool_result_no_loop() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({
            "tools": openai_tools(),
            "messages": [
                {"role": "user", "content": TRIGGER},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "get_weather", "arguments": ARGS}},
                ]},
                {"role": "tool", "tool_call_id": "call_0", "content": "sunny, 24C"},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        format!("You said: {TRIGGER}")
    );
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn anthropic_tool_result_no_loop() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({
            "tools": anthropic_tools(),
            "messages": [
                {"role": "user", "content": TRIGGER},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_0", "name": "get_weather", "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_0", "content": "sunny, 24C"},
                ]},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], format!("You said: {TRIGGER}"));
    assert_eq!(v["stop_reason"], "end_turn");
}

#[tokio::test]
async fn tool_trigger_needs_a_known_tool_and_json() {
    let cases = [
        // No tools in the request at all.
        (json!([]), "/tool get_weather {}"),
        // Name is not one of the request's tools.
        (openai_tools(), "/tool do_stuff {}"),
        // The name must be followed by whitespace, not just start with the prefix.
        (openai_tools(), "/toolkit get_weather {}"),
        // No arguments after the name.
        (openai_tools(), "/tool get_weather"),
        // Arguments that are not JSON.
        (openai_tools(), "/tool get_weather not-json"),
    ];
    for (tools, content) in cases {
        let (status, v) = json_of(post(
            "/v1/chat/completions",
            chat(json!({
                "tools": tools,
                "messages": [{"role": "user", "content": content}],
            })),
        ))
        .await;
        assert_eq!(status, StatusCode::OK, "{v}");
        let msg = &v["choices"][0]["message"];
        assert_eq!(msg["content"], format!("You said: {content}"), "{v}");
        assert!(
            msg.get("tool_calls")
                .is_none_or(|t| t.is_null() || t.as_array().is_some_and(|a| a.is_empty())),
            "{v}"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    }
}

#[tokio::test]
async fn blank_user_message_gets_the_greeting() {
    // An empty or whitespace-only last user message is not a tool trigger and is not
    // echoed: the mock falls back to its greeting, with or without tools in the request.
    for tools in [json!([]), openai_tools()] {
        for content in ["", "   ", "\n\t "] {
            let (status, v) = json_of(post(
                "/v1/chat/completions",
                chat(json!({
                    "tools": tools,
                    "messages": [{"role": "user", "content": content}],
                })),
            ))
            .await;
            assert_eq!(status, StatusCode::OK, "{v}");
            let msg = &v["choices"][0]["message"];
            assert_eq!(msg["content"], "Hello from the mock engine.", "{v}");
            assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
        }
    }
}
