use super::*;
use crate::engine::TokenEvent;
use crate::engine::tool_parser::ToolParser;
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: json!({"type": "object"}),
        strict: false,
    }]
}

fn args(body: &str) -> Option<Value> {
    parse_gemma4(body, &tools()).map(|(_, a)| serde_json::from_str(&a).unwrap())
}

#[test]
fn parses_gemma_notation() {
    assert_eq!(
        args("call:get_weather{location:<|\"|>Paris, \"FR\"<|\"|>,days:3}"),
        Some(json!({"location": "Paris, \"FR\"", "days": 3}))
    );
    assert_eq!(
        args("call:get_weather{a:[1, 2.5,<|\"|>x<|\"|>],b:{c:true,d:null}, e : false}"),
        Some(json!({"a": [1, 2.5, "x"], "b": {"c": true, "d": null}, "e": false}))
    );
    assert_eq!(args("call:get_weather{}"), Some(json!({})));
    assert_eq!(args("call:get_weather{a:[]}"), Some(json!({"a": []})));
}

#[test]
fn rejects_what_is_not_a_call() {
    assert_eq!(args("call:unknown_tool{}"), None);
    assert_eq!(args("call:get_weather{a:bare}"), None);
    assert_eq!(args("call:get_weather{a:1"), None);
    assert_eq!(args("call:get_weather{a:1} trailing"), None);
    assert_eq!(args("get_weather{}"), None);
}

#[test]
fn streams_calls_out_of_text() {
    let text = "Let me check.<|tool_call>call:get_weather{location:<|\"|>Oslo<|\"|>}\
                <tool_call|>";
    // Every split point of the stream gives the same events.
    for cut in 1..text.len() {
        if !text.is_char_boundary(cut) {
            continue;
        }
        let mut p = Gemma4Parser::new(&tools());
        let mut events = p.push(&text[..cut]);
        events.extend(p.push(&text[cut..]));
        events.extend(p.finish());
        let mut said = String::new();
        let mut calls = Vec::new();
        for e in events {
            match e {
                TokenEvent::Text(t) => said.push_str(&t),
                TokenEvent::ToolCall(c) => calls.push((c.name, c.arguments)),
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(said, "Let me check.", "cut {cut}");
        assert_eq!(
            calls,
            vec![(
                "get_weather".to_string(),
                r#"{"location":"Oslo"}"#.to_string()
            )],
            "cut {cut}"
        );
        assert!(p.had_calls());
    }
}
