mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

const TRIGGER: &str = "/tool get_weather {\"city\": \"Paris\"}";

fn tools() -> Value {
    json!([{
        "type": "function",
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }])
}

fn req(extra: Value) -> Value {
    let mut body = json!({"model": "mock", "input": "hi there"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

/// Streams a request and returns `(event name, data)` for every event, checking that the SSE
/// name matches `type` and that `sequence_number` counts up from 0.
async fn stream_events(body: Value) -> Vec<(String, Value)> {
    let (status, text) = send(post("/v1/responses", body)).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let events: Vec<(String, Value)> = sse_events(&text)
        .into_iter()
        .map(|(name, data)| (name.unwrap(), serde_json::from_str(&data).unwrap()))
        .collect();
    for (i, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], name.as_str(), "{data}");
        assert_eq!(data["sequence_number"], i, "{data}");
    }
    events
}

fn names(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(n, _)| n.as_str()).collect()
}

#[tokio::test]
async fn non_stream_text() {
    let (status, v) = json_of(post("/v1/responses", req(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert!(v["id"].as_str().unwrap().starts_with("resp_"));
    let output = v["output"].as_array().unwrap();
    assert_eq!(output.len(), 1, "{v}");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    let part = &output[0]["content"][0];
    assert_eq!(part["type"], "output_text");
    assert!(!part["text"].as_str().unwrap().is_empty());
    let u = &v["usage"];
    assert!(u["input_tokens"].as_u64().unwrap() > 0, "{v}");
    assert!(u["output_tokens"].as_u64().unwrap() > 0, "{v}");
    assert_eq!(
        u["total_tokens"].as_u64().unwrap(),
        u["input_tokens"].as_u64().unwrap() + u["output_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn instructions_and_item_input() {
    let (status, v) = json_of(post(
        "/v1/responses",
        req(json!({
            "instructions": "be brief",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "x"}]},
                {"role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "again"}]},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["instructions"], "be brief");
}

#[tokio::test]
async fn non_stream_reasoning_and_tool_call() {
    let (status, v) = json_of(post(
        "/v1/responses",
        req(json!({
            "tools": tools(),
            "input": TRIGGER,
            "reasoning": {"effort": "high"},
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let output = v["output"].as_array().unwrap();
    let kinds: Vec<_> = output.iter().map(|o| o["type"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["reasoning", "function_call"], "{v}");
    assert!(
        !output[0]["content"][0]["text"].as_str().unwrap().is_empty(),
        "{v}"
    );
    let call = &output[1];
    assert!(call["call_id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["arguments"], r#"{"city": "Paris"}"#);
}

#[tokio::test]
async fn function_call_round_trip_input() {
    let (status, v) = json_of(post(
        "/v1/responses",
        req(json!({
            "tools": tools(),
            "input": [
                {"role": "user", "content": "weather?"},
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "hmm"}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
            ],
        })),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["output"][0]["type"], "message");
}

#[tokio::test]
async fn stream_text_event_order() {
    let events = stream_events(req(json!({"stream": true}))).await;
    let names = names(&events);
    let n = names.len();
    assert_eq!(
        &names[..4],
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
        ]
    );
    assert!(
        names[4..n - 4]
            .iter()
            .all(|n| *n == "response.output_text.delta")
    );
    assert!(n > 8, "{names:?}");
    assert_eq!(
        &names[n - 4..],
        [
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let deltas: String = events
        .iter()
        .filter(|(n, _)| n == "response.output_text.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap())
        .collect();
    let done = &events[n - 4].1;
    assert_eq!(done["text"], deltas.as_str());
    let completed = &events[n - 1].1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(
        completed["output"][0]["content"][0]["text"],
        deltas.as_str()
    );
    assert!(completed["usage"]["total_tokens"].as_u64().unwrap() > 0);
    assert_eq!(events[0].1["response"]["status"], "in_progress");
    assert_eq!(events[0].1["response"]["output"], json!([]));
}

#[tokio::test]
async fn stream_reasoning_then_text() {
    let events = stream_events(req(
        json!({"stream": true, "reasoning": {"effort": "high"}}),
    ))
    .await;
    let names = names(&events);
    let first_reasoning = names
        .iter()
        .position(|n| *n == "response.reasoning_text.delta")
        .unwrap();
    let reasoning_done = names
        .iter()
        .position(|n| *n == "response.reasoning_text.done")
        .unwrap();
    let first_text = names
        .iter()
        .position(|n| *n == "response.output_text.delta")
        .unwrap();
    assert_eq!(names[first_reasoning - 1], "response.output_item.added");
    assert!(first_reasoning < reasoning_done && reasoning_done < first_text);
    assert_eq!(names[reasoning_done + 1], "response.output_item.done");
    assert_eq!(events[reasoning_done].1["output_index"], 0);
    let added: Vec<_> = events
        .iter()
        .filter(|(n, _)| n == "response.output_item.added")
        .map(|(_, d)| (d["output_index"].clone(), d["item"]["type"].clone()))
        .collect();
    assert_eq!(
        added,
        [(json!(0), json!("reasoning")), (json!(1), json!("message"))]
    );
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["output"][0]["type"], "reasoning");
    assert_eq!(completed["output"][1]["type"], "message");
}

#[tokio::test]
async fn stream_function_call() {
    let events = stream_events(req(json!({
        "stream": true,
        "tools": tools(),
        "input": TRIGGER,
    })))
    .await;
    assert_eq!(
        names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    let added = &events[2].1["item"];
    assert_eq!(added["type"], "function_call");
    assert_eq!(added["name"], "get_weather");
    let args = r#"{"city": "Paris"}"#;
    assert_eq!(events[3].1["delta"], args);
    assert_eq!(events[4].1["arguments"], args);
    assert_eq!(events[3].1["item_id"], added["id"]);
    assert_eq!(events[5].1["item"]["arguments"], args);
    assert_eq!(
        events[6].1["response"]["output"][0]["call_id"],
        added["call_id"]
    );
}

#[tokio::test]
async fn max_output_tokens_is_incomplete() {
    let (status, v) = json_of(post("/v1/responses", req(json!({"max_output_tokens": 2})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "incomplete");
    assert_eq!(v["incomplete_details"]["reason"], "max_output_tokens");

    let events = stream_events(req(json!({"stream": true, "max_output_tokens": 2}))).await;
    assert_eq!(events.last().unwrap().0, "response.incomplete");
}

#[tokio::test]
async fn stateful_requests_are_rejected() {
    for (extra, param) in [
        (
            json!({"previous_response_id": "resp_1"}),
            "previous_response_id",
        ),
        (json!({"store": true}), "store"),
    ] {
        let (status, v) = json_of(post("/v1/responses", req(extra))).await;
        assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["param"], param);
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("stateless")
        );
    }
    let (status, v) = json_of(post("/v1/responses", req(json!({"store": false})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn invalid_requests() {
    for extra in [
        json!({"input": []}),
        json!({"input": [{"type": "web_search_call"}]}),
        json!({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]}),
        json!({"tools": [{"type": "web_search"}]}),
        json!({"temperature": 5.0}),
        json!({"max_output_tokens": 0}),
    ] {
        let (status, v) = json_of(post("/v1/responses", req(extra.clone()))).await;
        assert!(status.is_client_error(), "{extra}: {v}");
        assert!(v["error"]["message"].is_string(), "{extra}: {v}");
    }
}
