//! Muse Glimmer replies: one or more messages, each addressed to a recipient,
//!
//! ```text
//!  to=self<|message|>REASONING<|eom|>
//! <|start|>assistant to=self<|message|>MORE REASONING<|eom|>
//! <|start|>assistant to=NAME<|message|><atem:function_calls>
//! <atem:invoke name="NAME">
//! <atem:parameter name="KEY">VALUE</atem:parameter>
//! </atem:invoke>
//! </atem:function_calls><|eom|>
//! <|start|>assistant to=user<|message|>ANSWER<|eot|>
//! ```
//!
//! (no newlines between the messages). The generation prompt ends at `<|start|>assistant`, so
//! the first message's `<|start|>assistant` is not in the reply. `<|start|>`, `<|message|>`,
//! `<|eom|>` and `<|eot|>` are control tokens, which the detokenizer prints for this parser
//! ([`MARKUP`]). Messages to `self` are reasoning, messages to `user` (or with no recipient)
//! the answer, and messages to anything else tool calls in the ATEM format: a string
//! parameter is written as is, any other one as JSON. Follows llama.cpp's
//! `common/chat-parsers/muse-glimmer.cpp` (MIT).

use serde_json::{Map, Value};

use super::{Tool, ToolCall, call_id, is_string_param};
use crate::engine::TokenEvent;
use crate::engine::reasoning::longest_suffix_prefix;

pub const START: &str = "<|start|>";
pub const MESSAGE: &str = "<|message|>";
/// Ends a message that more messages of the same turn follow (it does not end the reply).
pub const EOM: &str = "<|eom|>";
/// Ends the turn.
pub const EOT: &str = "<|eot|>";

/// The control tokens the detokenizer prints for the parser.
pub const MARKUP: [&str; 4] = [START, MESSAGE, EOM, EOT];

/// The ATEM call markup, printed too where the vocabulary has it as control tokens.
pub const ATEM: [&str; 6] = [
    CALLS_OPEN,
    "<atem:invoke",
    "<atem:parameter",
    PARAM_CLOSE,
    INVOKE_CLOSE,
    CALLS_CLOSE,
];

const CALLS_OPEN: &str = "<atem:function_calls>";
const CALLS_CLOSE: &str = "</atem:function_calls>";
const INVOKE_OPEN: &str = "<atem:invoke name=\"";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAM_OPEN: &str = "<atem:parameter name=\"";
const PARAM_CLOSE: &str = "</atem:parameter>";

/// Where the parser is in the reply.
#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    /// The reply's start: ` to=...` (the first message's header) or plain text follows.
    Start,
    /// Between messages: stray text (a reply without framing) is text.
    Between,
    /// After `<|start|>`: the header (`assistant to=RECIPIENT`) up to `<|message|>`.
    Header,
    /// In a message to `self`: streamed as reasoning.
    Reasoning,
    /// In a message to `user`: streamed as text.
    Text,
    /// In a message to a tool: held until its end.
    Call,
    /// In a message of another role (the model writing the next turn): dropped.
    Skip,
}

pub struct MuseParser {
    tools: Vec<Tool>,
    state: State,
    held: String,
    had_calls: bool,
}

impl MuseParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            tools: tools.to_vec(),
            state: State::Start,
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

    /// `text` as an event of the state's kind (reasoning or text); nothing for the others.
    fn emit(&self, events: &mut Vec<TokenEvent>, text: String) {
        if text.is_empty() {
            return;
        }
        match self.state {
            State::Reasoning => events.push(TokenEvent::Reasoning(text)),
            State::Start | State::Between | State::Text => events.push(TokenEvent::Text(text)),
            State::Header | State::Call | State::Skip => {}
        }
    }

    /// The state a message with this header (`assistant to=RECIPIENT`, or ` to=RECIPIENT` for
    /// the first one) opens.
    fn body_state(header: &str) -> State {
        let header = header.trim();
        let rest = match header.strip_prefix("assistant") {
            Some(rest) => rest.trim_start(),
            None if header.starts_with("to=") => header,
            // Another role's message: the model went on to write the next turn.
            None => return State::Skip,
        };
        let recipient = rest
            .strip_prefix("to=")
            .map(|r| r.split_whitespace().next().unwrap_or(""))
            .unwrap_or("");
        match recipient {
            "self" => State::Reasoning,
            "" | "user" => State::Text,
            _ => State::Call,
        }
    }

    /// A tool message's body as events: its calls (every `<atem:invoke>` of a known tool),
    /// else the body as text.
    fn calls(&mut self, body: &str) -> Vec<TokenEvent> {
        match parse_atem(body, &self.tools) {
            Some(calls) if !calls.is_empty() => {
                self.had_calls = true;
                calls
                    .into_iter()
                    .map(|(name, arguments)| {
                        TokenEvent::ToolCall(ToolCall {
                            id: call_id(),
                            name,
                            arguments,
                        })
                    })
                    .collect()
            }
            _ if body.is_empty() => Vec::new(),
            _ => vec![TokenEvent::Text(body.to_string())],
        }
    }
}

/// `<atem:function_calls><atem:invoke name="NAME"><atem:parameter name="KEY">VALUE
/// </atem:parameter>...</atem:invoke>...</atem:function_calls>` into `(name, arguments json)`
/// per invoke; `None` when it does not parse or names an unknown tool.
fn parse_atem(body: &str, tools: &[Tool]) -> Option<Vec<(String, String)>> {
    let mut rest = body.trim_start().strip_prefix(CALLS_OPEN)?;
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start();
        let Some(after) = rest.strip_prefix(INVOKE_OPEN) else {
            break;
        };
        let end = after.find("\">")?;
        let name = &after[..end];
        let tool = tools.iter().find(|t| t.name == name)?;
        rest = &after[end + 2..];
        let mut args = Map::new();
        loop {
            rest = rest.trim_start();
            let Some(after) = rest.strip_prefix(PARAM_OPEN) else {
                break;
            };
            let end = after.find("\">")?;
            let key = after[..end].to_string();
            let after = &after[end + 2..];
            let close = after.find(PARAM_CLOSE)?;
            let raw = &after[..close];
            let value = if is_string_param(tool, &key) {
                Value::String(raw.to_string())
            } else {
                serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
            };
            args.insert(key, value);
            rest = &after[close + PARAM_CLOSE.len()..];
        }
        rest = rest.trim_start().strip_prefix(INVOKE_CLOSE)?;
        calls.push((
            name.to_string(),
            serde_json::to_string(&Value::Object(args)).ok()?,
        ));
    }
    let tail = rest.trim();
    if !tail.is_empty() && tail != CALLS_CLOSE {
        return None;
    }
    Some(calls)
}

impl super::ToolParser for MuseParser {
    fn push(&mut self, piece: &str) -> Vec<TokenEvent> {
        self.held.push_str(piece);
        let mut events = Vec::new();
        loop {
            match self.state {
                State::Start => {
                    // The first message's header has no `<|start|>assistant` (the prompt
                    // ends with it); anything else is an unframed reply.
                    let t = self.held.trim_start();
                    if t.starts_with("to=") {
                        self.state = State::Header;
                    } else if "to=".starts_with(t) {
                        return events;
                    } else {
                        self.state = State::Between;
                    }
                }
                State::Between => {
                    let tags = [START, MESSAGE, EOM, EOT];
                    let Some((at, tag)) = self.first(&tags) else {
                        let out = self.release(&tags);
                        self.emit(&mut events, out);
                        return events;
                    };
                    let before: String = self.held.drain(..at + tag.len()).collect();
                    self.emit(&mut events, before[..at].to_string());
                    self.state = match tag {
                        START => State::Header,
                        // A message without a header is the answer.
                        MESSAGE => State::Text,
                        _ => State::Between,
                    };
                }
                State::Header => {
                    let tags = [MESSAGE, START, EOM, EOT];
                    let Some((at, tag)) = self.first(&tags) else {
                        return events;
                    };
                    let header: String = self.held.drain(..at + tag.len()).collect();
                    self.state = match tag {
                        MESSAGE => Self::body_state(&header[..at]),
                        START => State::Header,
                        _ => State::Between,
                    };
                }
                State::Reasoning | State::Text | State::Call | State::Skip => {
                    // A message ends at `<|eom|>` or `<|eot|>`; a new message also ends it.
                    let tags = [EOM, EOT, START];
                    let Some((at, tag)) = self.first(&tags) else {
                        if self.state != State::Call {
                            let out = self.release(&tags);
                            self.emit(&mut events, out);
                        }
                        return events;
                    };
                    let body: String = self.held.drain(..at + tag.len()).collect();
                    let body = &body[..at];
                    if self.state == State::Call {
                        let calls = self.calls(body);
                        events.extend(calls);
                    } else {
                        self.emit(&mut events, body.to_string());
                    }
                    self.state = if tag == START {
                        State::Header
                    } else {
                        State::Between
                    };
                }
            }
        }
    }

    fn finish(&mut self) -> Vec<TokenEvent> {
        let rest = std::mem::take(&mut self.held);
        let mut events = Vec::new();
        match self.state {
            // An unfinished call is text, as in the other formats.
            State::Call => {
                if !rest.is_empty() {
                    events.push(TokenEvent::Text(rest));
                }
            }
            State::Header | State::Skip => {}
            State::Start | State::Between | State::Text | State::Reasoning => {
                self.emit(&mut events, rest)
            }
        }
        self.state = State::Start;
        events
    }

    fn had_calls(&self) -> bool {
        self.had_calls
    }
}

#[cfg(test)]
mod tests;
