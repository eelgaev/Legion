use super::*;
use crate::engine::tool_parser::ToolParser;
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![Tool {
        name: "get_weather".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
        }),
        strict: false,
    }]
}

/// The reply through the parser, one piece at a time (`pieces`: the token texts): the
/// reasoning, the text and the calls `(name, arguments)`.
fn run(pieces: &[&str], tools: &[Tool]) -> (String, String, Vec<(String, Value)>) {
    let mut parser = MuseParser::new(tools);
    let mut events = Vec::new();
    for p in pieces {
        events.extend(parser.push(p));
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

/// A real reply (the server's, with the control tokens printed), and what it splits into.
const SAMPLE: &str = " to=self<|message|>What is the capital of France? Answer in one word.\n\n\
                      ...We need to answer in one word. Capital of France is Paris....\n\n\
                      Follow instruction.<|eom|><|start|>assistant to=user<|message|>Paris<|eot|>";
const SAMPLE_REASONING: &str = "What is the capital of France? Answer in one word.\n\n\
                                ...We need to answer in one word. Capital of France is Paris....\n\n\
                                Follow instruction.";

#[test]
fn sample_reply_splits_into_reasoning_and_answer() {
    let (thought, text, calls) = run(&[SAMPLE], &[]);
    assert_eq!(thought, SAMPLE_REASONING);
    assert_eq!(text, "Paris");
    assert!(calls.is_empty());
    // The stop token ends generation before it is printed: the same without it.
    let cut = SAMPLE.strip_suffix(EOT).unwrap();
    assert_eq!(run(&[cut], &[]).1, "Paris");
}

#[test]
fn sample_reply_streamed_in_any_pieces() {
    // One char at a time.
    let chars: Vec<String> = SAMPLE.chars().map(|c| c.to_string()).collect();
    let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
    let (thought, text, _) = run(&pieces, &[]);
    assert_eq!(
        (thought.as_str(), text.as_str()),
        (SAMPLE_REASONING, "Paris")
    );
    // Cut in two at every boundary, inside the markup too.
    for at in (0..=SAMPLE.len()).filter(|&i| SAMPLE.is_char_boundary(i)) {
        let (a, b) = SAMPLE.split_at(at);
        let (thought, text, _) = run(&[a, b], &[]);
        assert_eq!(
            (thought.as_str(), text.as_str()),
            (SAMPLE_REASONING, "Paris"),
            "cut at {at}"
        );
    }
    // Token-like pieces, the markup its own tokens.
    let pieces = [
        " to",
        "=self",
        MESSAGE,
        "Think",
        " hard",
        ".",
        EOM,
        START,
        "assistant",
        " to",
        "=user",
        MESSAGE,
        "Par",
        "is",
        EOT,
    ];
    let (thought, text, _) = run(&pieces, &[]);
    assert_eq!((thought.as_str(), text.as_str()), ("Think hard.", "Paris"));
}

#[test]
fn reply_without_reasoning() {
    let (thought, text, _) = run(&[" to=user", MESSAGE, "Paris", EOT], &[]);
    assert_eq!((thought.as_str(), text.as_str()), ("", "Paris"));
    // No recipient at all: the answer.
    let (thought, text, _) = run(&[MESSAGE, "Paris"], &[]);
    assert_eq!((thought.as_str(), text.as_str()), ("", "Paris"));
    // No framing at all: plain text.
    assert_eq!(run(&["plain", " text"], &[]).1, "plain text");
    assert_eq!(run(&["to", "ast"], &[]).1, "toast");
}

#[test]
fn several_reasoning_messages_join() {
    let reply = " to=self<|message|>First.<|eom|>\
                 <|start|>assistant to=self<|message|> Second.<|eom|>\
                 <|start|>assistant to=user<|message|>Answer.<|eot|>";
    let (thought, text, _) = run(&[reply], &[]);
    assert_eq!(
        (thought.as_str(), text.as_str()),
        ("First. Second.", "Answer.")
    );
    // Cut off in the reasoning by the length limit: what came is reasoning.
    let (thought, text, _) = run(&[" to=self", MESSAGE, "unfinished"], &[]);
    assert_eq!((thought.as_str(), text.as_str()), ("unfinished", ""));
}

#[test]
fn a_tool_message_is_a_tool_call() {
    let reply = " to=self<|message|>Look it up.<|eom|>\
                 <|start|>assistant to=get_weather<|message|><atem:function_calls>\n\
                 <atem:invoke name=\"get_weather\">\n\
                 <atem:parameter name=\"city\">New York</atem:parameter>\n\
                 <atem:parameter name=\"days\">3</atem:parameter>\n\
                 </atem:invoke>\n</atem:function_calls><|eot|>";
    for pieces in [vec![reply], reply.split_inclusive('>').collect()] {
        let (thought, text, calls) = run(&pieces, &tools());
        assert_eq!(thought, "Look it up.");
        assert_eq!(text, "");
        assert_eq!(
            calls,
            [(
                "get_weather".to_string(),
                json!({"city": "New York", "days": 3})
            )]
        );
    }
    let mut p = MuseParser::new(&tools());
    p.push(reply);
    assert!(p.had_calls());
}

#[test]
fn unknown_tools_and_bad_calls_stay_text() {
    let body = "<atem:function_calls>\n<atem:invoke name=\"nope\">\n</atem:invoke>\n\
                </atem:function_calls>";
    let reply = format!(" to=nope<|message|>{body}<|eom|>");
    let (_, text, calls) = run(&[&reply], &tools());
    assert!(calls.is_empty());
    assert_eq!(text, body);
    // Unfinished: the body as text.
    let (_, text, calls) = run(
        &[" to=get_weather", MESSAGE, "<atem:function_calls>"],
        &tools(),
    );
    assert!(calls.is_empty());
    assert_eq!(text, "<atem:function_calls>");
}

#[test]
fn a_next_turn_the_model_writes_is_dropped() {
    let reply = " to=user<|message|>Hi.<|eot|><|start|>user<|message|>more<|eot|>";
    assert_eq!(run(&[reply], &[]).1, "Hi.");
}
