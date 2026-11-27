use super::*;
use proptest::prelude::*;
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
                "any": {"type": ["string", "null"]},
            }}),
            strict: false,
        },
        Tool {
            name: "noop".into(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: false,
        },
    ]
}

#[derive(Debug, PartialEq)]
enum Ev {
    Text(String),
    Call(String, String),
}

fn run(mut p: impl ToolParser, pieces: &[&str]) -> (Vec<Ev>, bool) {
    let mut events = Vec::new();
    for piece in pieces {
        events.extend(p.push(piece));
    }
    events.extend(p.finish());
    let mut out: Vec<Ev> = Vec::new();
    for e in events {
        match e {
            TokenEvent::Text(t) => match out.last_mut() {
                Some(Ev::Text(prev)) => prev.push_str(&t),
                _ => out.push(Ev::Text(t)),
            },
            TokenEvent::ToolCall(c) => {
                assert!(c.id.starts_with("call_") && c.id.len() == 29, "{}", c.id);
                assert!(c.id[5..].chars().all(|ch| ch.is_ascii_hexdigit()));
                out.push(Ev::Call(c.name, c.arguments));
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    (out, p.had_calls())
}

fn text(s: &str) -> Ev {
    Ev::Text(s.into())
}
fn call(n: &str, a: &str) -> Ev {
    Ev::Call(n.into(), a.into())
}

fn hermes() -> HermesParser {
    HermesParser::new(&tools(), false)
}
fn coder() -> Qwen3CoderParser {
    Qwen3CoderParser::new(&tools(), false)
}

#[test]
fn coder_undeclared_name_follows_the_policy() {
    const FN: &str = "<function=";
    const PAR: &str = "<parameter=";
    const PEND: &str = "</parameter>";
    const FEND: &str = "</function>";
    let body = format!("{OPEN}{FN}unlisted>\n{PAR}cmd>\nls -la\n{PEND}\n{FEND}{CLOSE}");
    let (events, had) = run(Qwen3CoderParser::new(&tools(), false), &[&body]);
    assert!(!had, "text mode must not report a call");
    assert_eq!(events, vec![text(&body)]);

    let (events, had) = run(Qwen3CoderParser::new(&tools(), true), &[&body]);
    assert!(had, "call mode hands the call to the client");
    assert_eq!(events, vec![call("unlisted", "{\"cmd\":\"ls -la\"}")]);
}

#[test]
fn hermes_undeclared_name_follows_the_policy() {
    let json = r#"{"name": "unlisted", "arguments": {"x": 1}}"#;
    let body = format!("{OPEN}{json}{CLOSE}");
    assert_eq!(
        run(HermesParser::new(&tools(), false), &[&body]).0,
        vec![text(&body)]
    );
    assert_eq!(
        run(HermesParser::new(&tools(), true), &[&body]).0,
        vec![call("unlisted", "{\"x\":1}")]
    );
}

/// Runs `input` split at every char boundary and byte by byte; all must equal `expected`.
fn check_all_splits<P: ToolParser>(mk: impl Fn() -> P, input: &str, expected: &[Ev]) {
    for cut in 0..=input.len() {
        if !input.is_char_boundary(cut) {
            continue;
        }
        let (got, _) = run(mk(), &[&input[..cut], &input[cut..]]);
        assert_eq!(got, expected, "cut at {cut}");
    }
    let chars: Vec<String> = input.chars().map(String::from).collect();
    let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
    assert_eq!(run(mk(), &refs).0, expected, "char by char");
}

const H1: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call>";

#[test]
fn hermes_single() {
    let (ev, had) = run(hermes(), &[H1]);
    assert_eq!(
        ev,
        vec![call("get_weather", r#"{"city":"Paris","days":2}"#)]
    );
    assert!(had);
}

#[test]
fn hermes_text_before_and_multiple() {
    let input = format!(
        "Let me check.\n{H1}\n<tool_call>\n{{\"name\":\"noop\",\"arguments\":{{}}}}\n</tool_call>"
    );
    let expected = vec![
        text("Let me check.\n"),
        call("get_weather", r#"{"city":"Paris","days":2}"#),
        call("noop", "{}"),
    ];
    assert_eq!(run(hermes(), &[&input]).0, expected);
    check_all_splits(hermes, &input, &expected);
}

#[test]
fn hermes_interleaved_text() {
    let input = format!("a {H1} b {H1} c");
    let c = call("get_weather", r#"{"city":"Paris","days":2}"#);
    let expected = vec![
        text("a "),
        c,
        text("b "),
        call("get_weather", r#"{"city":"Paris","days":2}"#),
        text("c"),
    ];
    check_all_splits(hermes, &input, &expected);
}

#[test]
fn hermes_unicode_arguments() {
    let input = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"東京 😀 é\"}}\n</tool_call>";
    let expected = vec![call("get_weather", r#"{"city":"東京 😀 é"}"#)];
    check_all_splits(hermes, input, &expected);
}

#[test]
fn hermes_arguments_as_string_and_missing() {
    let (ev, _) = run(
        hermes(),
        &[
            "<tool_call>{\"name\":\"noop\",\"arguments\":\"{\\\"a\\\": 1}\"}</tool_call>",
            "<tool_call>{\"name\":\"noop\"}</tool_call>",
        ],
    );
    assert_eq!(ev, vec![call("noop", r#"{"a":1}"#), call("noop", "{}")]);
}

#[test]
fn hermes_malformed_falls_back_to_text() {
    for bad in [
        "<tool_call>\n{\"name\": \"noop\", \"arguments\": {oops}\n</tool_call>",
        "<tool_call>\n{\"name\": \"nope\", \"arguments\": {}}\n</tool_call>",
        "<tool_call>\n[1,2]\n</tool_call>",
        "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
    ] {
        let input = format!("x {bad} y");
        let (ev, had) = run(hermes(), &[&input]);
        assert_eq!(ev, vec![text(&input)], "{bad}");
        assert!(!had);
        check_all_splits(hermes, &input, &[text(&input)]);
    }
}

#[test]
fn hermes_unterminated_is_text() {
    let input = "ok <tool_call>\n{\"name\": \"noop\"";
    let (ev, had) = run(hermes(), &[input]);
    assert_eq!(ev, vec![text(input)]);
    assert!(!had);
}

#[test]
fn lookalike_is_released() {
    assert_eq!(
        run(hermes(), &["a <tool_c", "ool> not", " <tool_"]).0,
        vec![text("a <tool_cool> not <tool_")]
    );
}

#[test]
fn partial_open_tag_is_held_back() {
    let mut p = hermes();
    assert_eq!(p.push("x <tool_"), vec![TokenEvent::Text("x ".into())]);
    assert!(p.push("call>{\"name\":\"noop\"}").is_empty());
    assert!(matches!(
        p.push("</tool_call>")[..],
        [TokenEvent::ToolCall(_)]
    ));
}

const C1: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

#[test]
fn coder_single_with_typed_values() {
    let (ev, had) = run(coder(), &[C1]);
    assert_eq!(
        ev,
        vec![call("get_weather", r#"{"city":"Paris","days":3}"#)]
    );
    assert!(had);
}

#[test]
fn coder_string_param_stays_string() {
    let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\n42\n</parameter>\n<parameter=any>\ntrue\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(
        run(coder(), &[input]).0,
        vec![call("get_weather", r#"{"city":"42","any":"true"}"#)]
    );
}

#[test]
fn coder_object_and_unparseable_values() {
    let input = "<tool_call>\n<function=get_weather>\n<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n<parameter=days>\nmany\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(
        run(coder(), &[input]).0,
        vec![call("get_weather", r#"{"opts":{"a":[1,2]},"days":"many"}"#)]
    );
}

#[test]
fn arguments_keep_the_models_key_order() {
    let input = "<tool_call>{\"name\": \"noop\", \"arguments\": {\"b\":1,\"a\":2}}</tool_call>";
    assert_eq!(
        run(hermes(), &[input]).0,
        vec![call("noop", "{\"b\":1,\"a\":2}")]
    );
}

#[test]
fn coder_multiline_value_keeps_inner_newlines() {
    let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\nline1\nline2\n</parameter>\n</function>\n</tool_call>";
    assert_eq!(
        run(coder(), &[input]).0,
        vec![call("get_weather", r#"{"city":"line1\nline2"}"#)]
    );
}

#[test]
fn coder_multiple_interleaved_split_everywhere() {
    let input =
        format!("Sure.\n{C1}\n<tool_call>\n<function=noop>\n</function>\n</tool_call>\nDone é");
    let expected = vec![
        text("Sure.\n"),
        call("get_weather", r#"{"city":"Paris","days":3}"#),
        call("noop", "{}"),
        text("Done é"),
    ];
    check_all_splits(coder, &input, &expected);
}

#[test]
fn coder_unicode() {
    let input = "<tool_call>\n<function=get_weather>\n<parameter=city>\n東京 😀\n</parameter>\n</function>\n</tool_call>";
    check_all_splits(
        coder,
        input,
        &[call("get_weather", r#"{"city":"東京 😀"}"#)],
    );
}

#[test]
fn coder_malformed_falls_back_to_text() {
    for bad in [
        "<tool_call>\n<function=missing>\n</function>\n</tool_call>",
        "<tool_call>\n<function=noop>\n<parameter=a>\nunclosed\n</function>\n</tool_call>",
        "<tool_call>\nnot a function\n</tool_call>",
        "<tool_call>\n<function=noop>\ngarbage\n</function>\n</tool_call>",
    ] {
        let input = format!("pre {bad} post");
        let (ev, had) = run(coder(), &[&input]);
        assert_eq!(ev, vec![text(&input)], "{bad}");
        assert!(!had);
        check_all_splits(coder, &input, &[text(&input)]);
    }
}

#[test]
fn finish_reason_follows_calls() {
    assert_eq!(
        finish_reason(true, FinishReason::Stop),
        FinishReason::ToolCalls
    );
    assert_eq!(finish_reason(false, FinishReason::Stop), FinishReason::Stop);
    assert_eq!(
        finish_reason(true, FinishReason::Length),
        FinishReason::Length
    );
    assert_eq!(
        finish_reason(false, FinishReason::Cancelled),
        FinishReason::Cancelled
    );
}

#[test]
fn ids_are_distinct() {
    assert_ne!(call_id(), call_id());
}

proptest! {
    /// Chunking never changes the result.
    #[test]
    fn chunking_is_irrelevant(
        pre in "[abcé <>/\n]{0,10}",
        post in "[abcé <>/\n]{0,10}",
        city in "[a-zé 東]{0,8}",
        cuts in proptest::collection::vec(0usize..200, 0..8),
    ) {
        let args = json!({"city": city}).to_string();
        let input = format!(
            "{pre}<tool_call>\n{{\"name\":\"get_weather\",\"arguments\":{args}}}\n</tool_call>{post}"
        );
        let whole = run(hermes(), &[&input]).0;
        let mut cs: Vec<usize> = cuts.into_iter().map(|c| c % (input.len() + 1)).collect();
        cs.sort();
        let mut pieces = Vec::new();
        let mut last = 0;
        for mut c in cs {
            while !input.is_char_boundary(c) { c += 1; }
            if c > last { pieces.push(&input[last..c]); last = c; }
        }
        pieces.push(&input[last..]);
        prop_assert_eq!(run(hermes(), &pieces).0, whole);
    }

    /// Input without the open tag passes through unchanged.
    #[test]
    fn tagless_passes_through(input in "[abcé <>/_\n]{0,30}") {
        prop_assume!(!input.contains(OPEN));
        prop_assert_eq!(run(coder(), &[&input]).0, if input.is_empty() { vec![] } else { vec![text(&input)] });
    }
}
