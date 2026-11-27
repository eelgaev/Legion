//! Streaming tool-call parsers: turn `<tool_call>...</tool_call>` blocks in model text into
//! [`TokenEvent::ToolCall`] events, without leaking the markup into [`TokenEvent::Text`].
//!
//! Like [`super::reasoning::ReasoningParser`], text that could still be the start of the open tag
//! is held back until it either completes the tag or can no longer match. Inside a call nothing
//! is emitted until the close tag arrives, since a call is only useful whole. A block that does
//! not parse (bad JSON, unknown tool, ...) is emitted verbatim as `Text` rather than dropped.
//!
//! One case is a policy rather than a bug: a block that is well formed but names a tool
//! the request did not declare. By default it stays text; `--undeclared-tool-calls call`
//! ([`crate::engine::lm::LmConfig::undeclared_tool_calls`]) hands it to the client as a call,
//! so a drifted name reaches the tool channel instead of the conversation.
//!
//! Formats follow vllm's `hermes_tool_parser.py` and `qwen3coder_tool_parser.py` (Apache-2.0);
//! Gemma 4's (`<|tool_call>call:NAME{...}<tool_call|>`) is in [`gemma4`], GLM's
//! (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`) in [`glm`],
//! Inkling-2's message framing and JSON calls in [`inkling`], Muse Glimmer's messages and ATEM
//! calls in [`muse`].

mod gemma4;
mod glm;
pub mod inkling;
pub mod muse;
pub use gemma4::Gemma4Parser;
pub use glm::GlmParser;
pub use inkling::InklingParser;
pub use muse::MuseParser;

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
type BodyParser = fn(&str, &[Tool], bool) -> Option<(String, String)>;

struct Core {
    parse: BodyParser,
    open: &'static str,
    close: &'static str,
    tools: Vec<Tool>,
    /// Hand a well-formed block naming an undeclared tool to the client.
    call_undeclared: bool,
    inside: bool,
    held: String,
    had_calls: bool,
    /// A call was just emitted: drop whitespace-only text until real text follows.
    trim_leading: bool,
}

impl Core {
    fn new(parse: BodyParser, tools: &[Tool], call_undeclared: bool) -> Self {
        Self::with_tags(parse, tools, call_undeclared, OPEN, CLOSE)
    }

    /// A parser for calls between `open` and `close` instead of `<tool_call>` tags.
    fn with_tags(
        parse: BodyParser,
        tools: &[Tool],
        call_undeclared: bool,
        open: &'static str,
        close: &'static str,
    ) -> Self {
        Self {
            parse,
            open,
            close,
            tools: tools.to_vec(),
            call_undeclared,
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
                match (self.parse)(&body, &self.tools, self.call_undeclared) {
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
    pub fn new(tools: &[Tool], call_undeclared: bool) -> Self {
        Self(Core::new(parse_hermes, tools, call_undeclared))
    }
}

fn parse_hermes(body: &str, tools: &[Tool], call_undeclared: bool) -> Option<(String, String)> {
    let Value::Object(obj) = serde_json::from_str(body.trim()).ok()? else {
        return None;
    };
    let name = obj.get("name")?.as_str()?.to_string();
    if name.is_empty() || (!known(tools, &name) && !call_undeclared) {
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
    pub fn new(tools: &[Tool], call_undeclared: bool) -> Self {
        Self(Core::new(parse_qwen3_coder, tools, call_undeclared))
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

fn parse_qwen3_coder(
    body: &str,
    tools: &[Tool],
    call_undeclared: bool,
) -> Option<(String, String)> {
    let mut rest = body.trim().strip_prefix("<function=")?;
    let end = rest.find('>')?;
    let name = rest[..end].trim().to_string();
    let tool = tools.iter().find(|t| t.name == name);
    if tool.is_none() && !(call_undeclared && !name.is_empty()) {
        return None;
    }
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
        let value = if tool.is_some_and(|t| is_string_param(t, &key)) {
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
mod tests;
