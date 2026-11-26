use super::*;
use crate::engine::reasoning::ReasoningParser;
use crate::engine::tool_parser::ToolParser;
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        strict: false,
    }]
}

/// The reply through the reasoning parser and this one, one piece at a time (`pieces`: the
/// token texts): the thinking, the text and the calls `(name, arguments)`.
fn run(pieces: &[&str], tools: &[Tool]) -> (String, String, Vec<(String, Value)>) {
    let mut reasoning = ReasoningParser::new(THINKING, END, false);
    let mut parser = InklingParser::new(tools);
    let mut events = Vec::new();
    for p in pieces {
        for ev in reasoning.push(p) {
            match ev {
                TokenEvent::Text(t) => events.extend(parser.push(&t)),
                other => events.push(other),
            }
        }
    }
    for ev in reasoning.finish() {
        match ev {
            TokenEvent::Text(t) => events.extend(parser.push(&t)),
            other => events.push(other),
        }
    }
    events.extend(parser.finish());
    let (mut thought, mut text, mut calls) = (String::new(), String::new(), Vec::new());
    for ev in events {
        match ev {
            TokenEvent::Reasoning(t) => thought.push_str(&t),
            TokenEvent::Text(t) => text.push_str(&t),
            TokenEvent::ToolCall(c) => {
                calls.push((c.name, serde_json::from_str(&c.arguments).unwrap()))
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    (thought, text, calls)
}

#[test]
fn thinking_then_text() {
    let pieces = [
        MESSAGE, THINKING, "Two", " plus", " two.", END, MESSAGE, TEXT, "It", " is", " 4.", END,
    ];
    let (thought, text, calls) = run(&pieces, &[]);
    assert_eq!(thought, "Two plus two.");
    assert_eq!(text, "It is 4.");
    assert!(calls.is_empty());
}

#[test]
fn text_parts_join_and_errors_are_text() {
    let pieces = [
        MESSAGE, TEXT, "a", END, MESSAGE, TEXT, "b", END, MESSAGE, TOOL_ERROR, "c", END,
    ];
    assert_eq!(run(&pieces, &[]).1, "abc");
}

#[test]
fn a_json_message_is_a_tool_call() {
    let pieces = [
        MESSAGE,
        THINKING,
        "Call it.",
        END,
        MESSAGE,
        "get",
        "_weather",
        TOOL_JSON,
        "{\"name\":\"get_weather\",",
        "\"args\":{\"city\":\"Oslo\"}}",
        END,
    ];
    let (thought, text, calls) = run(&pieces, &tools());
    assert_eq!(thought, "Call it.");
    assert_eq!(text, "");
    assert_eq!(
        calls,
        [("get_weather".to_string(), json!({"city": "Oslo"}))]
    );
    let mut p = InklingParser::new(&tools());
    for piece in pieces {
        p.push(piece);
    }
    assert!(p.had_calls());
}

#[test]
fn unknown_tools_and_bad_json_stay_text() {
    let body = "{\"name\":\"nope\",\"args\":{}}";
    let (_, text, calls) = run(&[MESSAGE, "nope", TOOL_JSON, body, END], &tools());
    assert!(calls.is_empty());
    assert_eq!(text, body);
    let (_, text, _) = run(&[MESSAGE, TOOL_JSON, "{bad", END], &tools());
    assert_eq!(text, "{bad");
    // Unfinished: the body as text.
    let (_, text, calls) = run(&[MESSAGE, TOOL_JSON, "{\"name\":"], &tools());
    assert!(calls.is_empty());
    assert_eq!(text, "{\"name\":");
}

#[test]
fn unframed_text_passes_and_markup_split_across_pieces_is_held() {
    assert_eq!(run(&["plain", " text"], &[]).1, "plain text");
    // A tag that arrives in pieces (text the detokenizer splits) is still found.
    let (_, text, _) = run(
        &[
            "<|message_",
            "model|><|content_",
            "text|>hi<|end_",
            "message|>",
        ],
        &[],
    );
    assert_eq!(text, "hi");
    // A text part cut off by the length limit keeps its text.
    assert_eq!(run(&[MESSAGE, TEXT, "partial"], &[]).1, "partial");
}
