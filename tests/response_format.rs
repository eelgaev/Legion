mod common;

use axum::http::StatusCode;
use common::{assert_openai_error, json_of, post};
use serde_json::{Value, json};

const ROUTES: [&str; 3] = ["/v1/chat/completions", "/v1/completions", "/v1/responses"];

fn request(route: &str, mut format: Value) -> Value {
    let mut body = match route {
        "/v1/chat/completions" => common::chat(json!({})),
        "/v1/completions" => json!({"model":"mock", "prompt":"hi there"}),
        _ => json!({"model":"mock", "input":"hi there"}),
    };
    if route == "/v1/responses" {
        if format["type"] == "json_schema" {
            let definition = format.as_object_mut().unwrap().remove("json_schema");
            if let Some(Value::Object(definition)) = definition {
                format.as_object_mut().unwrap().extend(definition);
            }
        }
        body["text"] = json!({"format":format});
    } else {
        body["response_format"] = format;
    }
    body
}

fn output<'a>(route: &str, body: &'a Value) -> &'a str {
    match route {
        "/v1/chat/completions" => body["choices"][0]["message"]["content"].as_str().unwrap(),
        "/v1/completions" => body["choices"][0]["text"].as_str().unwrap(),
        _ => body["output"][0]["content"][0]["text"].as_str().unwrap(),
    }
}

#[tokio::test]
async fn invalid_formats_have_openai_errors_before_streaming() {
    for route in ROUTES {
        for stream in [false, true] {
            for (format, message) in [
                (json!({"type":"bogus"}), "unsupported response format type"),
                (
                    json!({"type":"json_schema","json_schema":{"name":"missing"}}),
                    "requires a schema",
                ),
                (
                    json!({"type":"json_schema","json_schema":{"name":"bad","schema":{"type":"string","pattern":"x"}}}),
                    "unsupported schema feature: `pattern`",
                ),
                (
                    json!({"type":"json_schema","json_schema":{"schema":false}}),
                    "schema is unsatisfiable",
                ),
                (
                    json!({"type":"json_schema","json_schema":{"schema":42}}),
                    "schema must be an object or boolean",
                ),
            ] {
                let mut body = request(route, format);
                body["stream"] = json!(stream);
                let (status, result) = json_of(post(route, body)).await;
                assert_openai_error(status, &result, StatusCode::BAD_REQUEST);
                assert_eq!(result["error"]["type"], "invalid_request_error");
                assert_eq!(
                    result["error"]["param"],
                    if route == "/v1/responses" {
                        "text.format"
                    } else {
                        "response_format"
                    }
                );
                assert!(
                    result["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains(message),
                    "{result}"
                );
            }
        }
    }
}

#[tokio::test]
async fn text_is_a_noop_and_mock_json_formats_are_plumbed() {
    for route in ROUTES {
        for (format, expected) in [
            (json!({"type":"text"}), "You said: hi there"),
            (json!({"type":"json_object"}), "{}"),
            (
                json!({"type":"json_schema","json_schema":{"name":"empty","strict":false,"schema":{"type":"object","properties":{"a":{"type":"integer"}}}}}),
                "{}",
            ),
            (
                json!({"type":"json_schema","json_schema":{"name":"required","strict":true,"schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}}),
                "You said: hi there",
            ),
        ] {
            let req = request(route, format);
            let (status, body) = json_of(post(route, &req)).await;
            if route == "/v1/responses" {
                assert_eq!(body["text"]["format"], req["text"]["format"]);
            }
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(output(route, &body), expected);
        }
    }
}

#[tokio::test]
async fn anthropic_ignores_response_format() {
    let (status, body) = json_of(post(
        "/v1/messages",
        common::messages(json!({"response_format":{"type":"bogus"}})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn json_object_streams_on_every_openai_route() {
    for route in ROUTES {
        let mut req = request(route, json!({"type":"json_object"}));
        req["stream"] = json!(true);
        let (status, body) = common::send(post(route, req)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let mut text = String::new();
        for (event, data) in common::sse_events(&body) {
            if data == "[DONE]" {
                continue;
            }
            let value: Value = serde_json::from_str(&data).unwrap();
            let delta = match route {
                "/v1/chat/completions" => &value["choices"][0]["delta"]["content"],
                "/v1/completions" => &value["choices"][0]["text"],
                _ if event.as_deref() == Some("response.output_text.delta") => &value["delta"],
                _ => &Value::Null,
            };
            if let Some(delta) = delta.as_str() {
                text += delta;
            }
            if event.as_deref() == Some("response.completed") {
                assert_eq!(value["response"]["text"]["format"]["type"], "json_object");
            }
        }
        assert_eq!(text, "{}");
    }
}
