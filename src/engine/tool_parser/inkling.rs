//! Inkling-2 replies: every part is a message of its own,
//!
//! ```text
//! <|message_model|><|content_thinking|>REASONING<|end_message|>
//! <|message_model|><|content_text|>ANSWER<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
//! ```
//!
//! (no newlines between them), all of the markup control tokens. The reasoning parser takes
//! the thinking out ([`THINKING`] ... [`END`]); this parser drops the rest of the framing,
//! streams the text messages and turns the JSON messages into tool calls, as the
//! `response_template` of Inkling-2's `tokenizer_config.json` does (text-like parts joined
//! with nothing between them; a call's `args` become its arguments).

use serde_json::Value;

use super::{Tool, call_id, known};
use crate::engine::TokenEvent;
use crate::engine::reasoning::longest_suffix_prefix;

/// A model message's header (its role; a tool call's name follows it).
pub const MESSAGE: &str = "<|message_model|>";
pub const THINKING: &str = "<|content_thinking|>";
pub const END: &str = "<|end_message|>";
const TEXT: &str = "<|content_text|>";
const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
const TOOL_ERROR: &str = "<|content_tool_error|>";

/// The control tokens the detokenizer prints for the parsers.
pub const MARKUP: [&str; 7] = [
    MESSAGE, THINKING, END, TEXT, TOOL_JSON, TOOL_TEXT, TOOL_ERROR,
];

/// Where the parser is in the reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Between messages: stray text (a reply without framing) is text.
    Between,
    /// After `<|message_model|>`: the header up to the content marker (a call's name) is
    /// dropped.
    Header,
    /// In a text-like part: streamed as text.
    Text,
    /// In a JSON tool call: held until its end.
    Call,
}

pub struct InklingParser {
    tools: Vec<Tool>,
    state: State,
    held: String,
    had_calls: bool,
}

impl InklingParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            state: State::Between,
            held: String::new(),
            had_calls: false,
        }
    }

    /// The first of `tags` in the held text: its offset and the tag.
    fn first(&self, tags: &[&'static str]) -> Option<(usize, &'static str)> {
        tags.iter()
            .filter_map(|t| self.held.find(t).map(|at| (at, *t)))
            .min_by_key(|&(at, _)| at)
    }

    /// The held text that cannot start any of `tags`, taken out.
    fn release(&mut self, tags: &[&str]) -> String {
        let keep = tags
            .iter()
            .map(|t| longest_suffix_prefix(&self.held, t))
            .max()
            .unwrap_or(0);
        self.held.drain(..self.held.len() - keep).collect()
    }

    fn text(events: &mut Vec<TokenEvent>, text: String) {
        if !text.is_empty() {
            events.push(TokenEvent::Text(text));
        }
    }

    /// A call's body as an event: `{"name": ..., "args": {...}}` of a known tool, else the
    /// body as text.
    fn call(&mut self, body: &str) -> TokenEvent {
        let parsed = serde_json::from_str::<Value>(body.trim())
            .ok()
            .and_then(|v| {
                let name = v.get("name")?.as_str()?.to_string();
                let args = match v.get("args") {
                    None | Some(Value::Null) => Value::Object(Default::default()),
                    Some(a @ Value::Object(_)) => a.clone(),
                    Some(_) => return None,
                };
                Some((name, serde_json::to_string(&args).ok()?))
            })
            .filter(|(name, _)| known(&self.tools, name));
        match parsed {
            Some((name, arguments)) => {
                self.had_calls = true;
                TokenEvent::ToolCall(super::ToolCall {
                    id: call_id(),
                    name,
                    arguments,
                })
            }
            None => TokenEvent::Text(body.to_string()),
        }
    }
}

impl super::ToolParser for InklingParser {
    fn push(&mut self, piece: &str) -> Vec<TokenEvent> {
        self.held.push_str(piece);
        let mut events = Vec::new();
        const CONTENT: [&str; 5] = [TEXT, TOOL_JSON, TOOL_TEXT, TOOL_ERROR, THINKING];
        loop {
            match self.state {
                State::Between | State::Header => {
                    let tags = [
                        MESSAGE, TEXT, TOOL_JSON, TOOL_TEXT, TOOL_ERROR, THINKING, END,
                    ];
                    let Some((at, tag)) = self.first(&tags) else {
                        let out = self.release(&tags);
                        if self.state == State::Between {
                            Self::text(&mut events, out);
                        }
                        return events;
                    };
                    let before: String = self.held.drain(..at + tag.len()).collect();
                    if self.state == State::Between {
                        Self::text(&mut events, before[..at].to_string());
                    }
                    self.state = match tag {
                        MESSAGE => State::Header,
                        TOOL_JSON => State::Call,
                        END => State::Between,
                        t if CONTENT.contains(&t) => State::Text,
                        _ => unreachable!(),
                    };
                }
                State::Text => {
                    // A part ends at its `<|end_message|>`; a new message also ends it.
                    let Some((at, tag)) = self.first(&[END, MESSAGE]) else {
                        let out = self.release(&[END, MESSAGE]);
                        Self::text(&mut events, out);
                        return events;
                    };
                    let body: String = self.held.drain(..at).collect();
                    Self::text(&mut events, body);
                    if tag == END {
                        self.held.drain(..END.len());
                        self.state = State::Between;
                    } else {
                        self.held.drain(..MESSAGE.len());
                        self.state = State::Header;
                    }
                }
                State::Call => {
                    let Some(at) = self.held.find(END) else {
                        return events;
                    };
                    let body: String = self.held.drain(..at).collect();
                    self.held.drain(..END.len());
                    let ev = self.call(&body);
                    events.push(ev);
                    self.state = State::Between;
                }
            }
        }
    }

    fn finish(&mut self) -> Vec<TokenEvent> {
        let rest = std::mem::take(&mut self.held);
        let mut events = Vec::new();
        match self.state {
            // An unfinished call is text, as in the other formats.
            State::Call => Self::text(&mut events, rest),
            State::Header => {}
            State::Between | State::Text => Self::text(&mut events, rest),
        }
        self.state = State::Between;
        events
    }

    fn had_calls(&self) -> bool {
        self.had_calls
    }
}

#[cfg(test)]
mod tests;
