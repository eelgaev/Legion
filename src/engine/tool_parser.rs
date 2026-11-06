//! Streaming tool-call parsers: turn `<tool_call>...</tool_call>` blocks in model text into
//! [`TokenEvent::ToolCall`] events, without leaking the markup into [`TokenEvent::Text`].
//!
//! Like [`super::reasoning::ReasoningParser`], text that could still be the start of the open tag
//! is held back until it either completes the tag or can no longer match. Inside a call nothing
//! is emitted until the close tag arrives, since a call is only useful whole. A block that does
//! not parse (bad JSON, unknown tool, ...) is emitted verbatim as `Text` rather than dropped.
//!
//! Formats follow vllm's `hermes_tool_parser.py` and `qwen3coder_tool_parser.py` (Apache-2.0);
//! Gemma 4's (`<|tool_call>call:NAME{...}<tool_call|>`) is in [`gemma4`], GLM's
//! (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`) in [`glm`].

mod gemma4;
mod glm;
pub use gemma4::Gemma4Parser;
pub use glm::GlmParser;

use serde_json::{Map, Value};

use super::reasoning::longest_suffix_prefix;
use super::{FinishReason, TokenEvent, Tool, ToolCall};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Incremental parser for the tool calls in a model reply.
pub trait ToolParser: Send {
    /// Feed the next piece of model text.
    fn push(&mut self, piece: &str) -> Vec<TokenEvent>;
    /// End of generation: release whatever was held back (an unterminated call is `Text`).
    fn finish(&mut self) -> Vec<TokenEvent>;
    /// Whether any call has been emitted so far.
    fn had_calls(&self) -> bool;
}

/// The finish reason of a reply: `Stop` becomes `ToolCalls` when a call was emitted. Other
/// reasons (length, stop sequence, cancellation) are kept, since the reply did not end by the
/// model handing over to the tool.
pub fn finish_reason(had_calls: bool, base: FinishReason) -> FinishReason {
    if had_calls && base == FinishReason::Stop {
        FinishReason::ToolCalls
    } else {
        base
    }
}

/// Parses the body between the tags into `(name, arguments json)`.
type BodyParser = fn(&str, &[Tool]) -> Option<(String, String)>;

struct Core {
    parse: BodyParser,
    open: &'static str,
    close: &'static str,
    tools: Vec<Tool>,
    inside: bool,
    held: String,
    had_calls: bool,
    /// A call was just emitted: drop whitespace-only text until real text follows.
    trim_leading: bool,
}

impl Core {
    fn new(parse: BodyParser, tools: &[Tool]) -> Self {
        Self::with_tags(parse, tools, OPEN, CLOSE)
    }

    /// A parser for calls between `open` and `close` instead of `<tool_call>` tags.
    fn with_tags(
        parse: BodyParser,
        tools: &[Tool],
        open: &'static str,
        close: &'static str,
    ) -> Self {
        Self {
            parse,
            open,
            close,
            tools: tools.to_vec(),
            inside: false,
            held: String::new(),
            had_calls: false,
            trim_leading: false,
        }
    }

    fn push(&mut self, piece: &str) -> Vec<TokenEvent> {
        self.held.push_str(piece);
        let mut events = Vec::new();
        loop {
            if self.inside {
                let Some(at) = self.held.find(self.close) else {
                    return events;
                };
                let body: String = self.held[..at].to_string();
                self.held.drain(..at + self.close.len());
                self.inside = false;
                match (self.parse)(&body, &self.tools) {
                    Some((name, arguments)) => {
                        self.had_calls = true;
                        self.trim_leading = true;
                        events.push(TokenEvent::ToolCall(ToolCall {
                            id: call_id(),
                            name,
                            arguments,
                        }));
                    }
                    None => events.push(TokenEvent::Text(format!(
                        "{}{body}{}",
                        self.open, self.close
                    ))),
                }
            } else if let Some(at) = self.held.find(self.open) {
                let before: String = self.held[..at].to_string();
                self.held.drain(..at + self.open.len());
                self.emit_text(&mut events, before);
                self.inside = true;
            } else {
                let cut = self.held.len() - longest_suffix_prefix(&self.held, self.open);
                let out: String = self.held.drain(..cut).collect();
                self.emit_text(&mut events, out);
                return events;
            }
        }
    }

    fn finish(&mut self) -> Vec<TokenEvent> {
        let mut events = Vec::new();
        let mut rest = std::mem::take(&mut self.held);
        if self.inside {
            rest.insert_str(0, self.open);
            self.inside = false;
            events.push(TokenEvent::Text(rest));
        } else {
            self.emit_text(&mut events, rest);
        }
        events
    }

    fn emit_text(&mut self, events: &mut Vec<TokenEvent>, text: String) {
        let text = if self.trim_leading {
            text.trim_start().to_string()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.trim_leading = false;
        events.push(TokenEvent::Text(text));
    }
}

fn call_id() -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("call_{}", &hex[..24])
}

fn known(tools: &[Tool], name: &str) -> bool {
    tools.iter().any(|t| t.name == name)
}

/// `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>` (Qwen2.5, Qwen3, Hermes).
pub struct HermesParser(Core);

impl HermesParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self(Core::new(parse_hermes, tools))
    }
}

fn parse_hermes(body: &str, tools: &[Tool]) -> Option<(String, String)> {
    let Value::Object(obj) = serde_json::from_str(body.trim()).ok()? else {
        return None;
    };
    let name = obj.get("name")?.as_str()?.to_string();
    if !known(tools, &name) {
        return None;
    }
    let arguments = match obj.get("arguments") {
        None => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(s).ok()?,
        Some(v) => v.clone(),
    };
    Some((name, serde_json::to_string(&arguments).ok()?))
}

/// `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...</function>\n</tool_call>`
/// (Qwen3-Coder). A value is parsed as JSON unless the tool's schema types the parameter as a
/// string (or does not mention it); if it does not parse, it stays a string.
pub struct Qwen3CoderParser(Core);

impl Qwen3CoderParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self(Core::new(parse_qwen3_coder, tools))
    }
}

fn is_string_param(tool: &Tool, key: &str) -> bool {
    match tool
        .parameters
        .pointer(&format!("/properties/{}/type", escape_pointer(key)))
    {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        _ => true,
    }
}

fn escape_pointer(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn parse_qwen3_coder(body: &str, tools: &[Tool]) -> Option<(String, String)> {
    let mut rest = body.trim().strip_prefix("<function=")?;
    let end = rest.find('>')?;
    let name = rest[..end].trim().to_string();
    let tool = tools.iter().find(|t| t.name == name)?;
    rest = &rest[end + 1..];
    let mut args = Map::new();
    loop {
        rest = rest.trim_start();
        let Some(after) = rest.strip_prefix("<parameter=") else {
            break;
        };
        let end = after.find('>')?;
        let key = after[..end].trim().to_string();
        let after = &after[end + 1..];
        let close = after.find("</parameter>")?;
        let raw = &after[..close];
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = if is_string_param(tool, &key) {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        args.insert(key, value);
        rest = &after[close + "</parameter>".len()..];
    }
    let tail = rest.trim();
    if !tail.is_empty() && tail != "</function>" {
        return None;
    }
    Some((name, serde_json::to_string(&Value::Object(args)).ok()?))
}

macro_rules! impl_tool_parser {
    ($t:ty) => {
        impl ToolParser for $t {
            fn push(&mut self, piece: &str) -> Vec<TokenEvent> {
                self.0.push(piece)
            }
            fn finish(&mut self) -> Vec<TokenEvent> {
                self.0.finish()
            }
            fn had_calls(&self) -> bool {
                self.0.had_calls
            }
        }
    };
}
impl_tool_parser!(HermesParser);
impl_tool_parser!(Qwen3CoderParser);
impl_tool_parser!(Gemma4Parser);
impl_tool_parser!(GlmParser);

#[cfg(test)]
mod tests {
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
        HermesParser::new(&tools())
    }
    fn coder() -> Qwen3CoderParser {
        Qwen3CoderParser::new(&tools())
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
}
