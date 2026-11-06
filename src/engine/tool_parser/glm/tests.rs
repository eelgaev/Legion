use super::*;
use crate::engine::TokenEvent;
use crate::engine::tool_parser::ToolParser;
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".into(),
            description: None,
            parameters: json!({"type": "object", "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "opts": {"type": "object"},
            }}),
            strict: false,
        },
        Tool {
            name: "now".into(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: false,
        },
    ]
}

fn parse(body: &str) -> Option<(String, Value)> {
    parse_glm(body, &tools()).map(|(n, a)| (n, serde_json::from_str(&a).unwrap()))
}

#[test]
fn parses_glm_calls() {
    // As GLM-5.3-Flash writes it (no newlines).
    assert_eq!(
        parse(
            "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>"
        ),
        Some(("get_weather".into(), json!({"city": "Paris", "days": 3})))
    );
    // GLM-4.5's newlines, a JSON object value and a string that looks like a number.
    assert_eq!(
        parse(
            "get_weather\n<arg_key>city</arg_key>\n<arg_value>42</arg_value>\n<arg_key>opts</arg_key>\n<arg_value>{\"units\": \"c\"}</arg_value>\n"
        ),
        Some((
            "get_weather".into(),
            json!({"city": "42", "opts": {"units": "c"}})
        ))
    );
    // No arguments.
    assert_eq!(parse("now"), Some(("now".into(), json!({}))));
    // An untyped or unparsable value stays a string, inner whitespace kept.
    assert_eq!(
        parse(
            "get_weather<arg_key>days</arg_key><arg_value>soon</arg_value><arg_key>note</arg_key><arg_value> a b </arg_value>"
        ),
        Some((
            "get_weather".into(),
            json!({"days": "soon", "note": " a b "})
        ))
    );
}

#[test]
fn rejects_what_is_not_a_call() {
    assert_eq!(
        parse("unknown<arg_key>a</arg_key><arg_value>1</arg_value>"),
        None
    );
    assert_eq!(parse(""), None);
    assert_eq!(parse("get_weather<arg_key>city</arg_key>Paris"), None);
    assert_eq!(
        parse("get_weather<arg_key>city</arg_key><arg_value>Paris"),
        None
    );
    assert_eq!(
        parse("get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value> trailing"),
        None
    );
    assert_eq!(parse("{\"name\": \"now\"}"), None);
}

#[test]
fn streams_calls_out_of_the_text() {
    let mut p = GlmParser::new(&tools());
    let text = "Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>\n<tool_call>now</tool_call>";
    let mut events = Vec::new();
    for c in text.chars() {
        events.extend(p.push(&c.to_string()));
    }
    events.extend(p.finish());
    let mut out = String::new();
    let mut calls = Vec::new();
    for e in events {
        match e {
            TokenEvent::Text(t) => out.push_str(&t),
            TokenEvent::ToolCall(c) => calls.push((c.name, c.arguments)),
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(out, "Checking.");
    assert_eq!(
        calls,
        vec![
            ("get_weather".to_string(), r#"{"city":"Paris"}"#.to_string()),
            ("now".to_string(), "{}".to_string()),
        ]
    );
    assert!(p.had_calls());
}
