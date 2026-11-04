//! Gemma 4 tool calls: `<|tool_call>call:NAME{key:value,...}<tool_call|>`. The arguments are
//! Gemma's own notation, not JSON: keys are bare, strings are wrapped in `<|"|>` (so they need
//! no escaping), and numbers, `true`, `false`, `null`, lists `[...]` and nested `{...}` are as in
//! JSON. The grammar and its translation to JSON follow llama.cpp's `common/parsers/gemma4.cpp`
//! and `gemma4_to_json` in `common/chat-peg-parser.cpp` (MIT).

use serde_json::{Map, Value};

use super::{Core, Tool, known};

const OPEN: &str = "<|tool_call>";
const CLOSE: &str = "<tool_call|>";
/// Gemma 4's string delimiter (one token).
const QUOTE: &str = "<|\"|>";

pub struct Gemma4Parser(pub(super) Core);

impl Gemma4Parser {
    pub fn new(tools: &[Tool]) -> Self {
        Self(Core::with_tags(parse_gemma4, tools, OPEN, CLOSE))
    }
}

/// `call:NAME{...}` into `(name, arguments json)`.
fn parse_gemma4(body: &str, tools: &[Tool]) -> Option<(String, String)> {
    let rest = body.trim().strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_string();
    if !known(tools, &name) {
        return None;
    }
    let mut s = &rest[brace..];
    let args = dict(&mut s)?;
    if !s.trim().is_empty() {
        return None;
    }
    Some((name, serde_json::to_string(&args).ok()?))
}

fn skip_ws(s: &mut &str) {
    *s = s.trim_start();
}

fn eat(s: &mut &str, prefix: &str) -> bool {
    match s.strip_prefix(prefix) {
        Some(rest) => {
            *s = rest;
            true
        }
        None => false,
    }
}

/// `{` (key `:` value (`,` key `:` value)*)? `}`
fn dict(s: &mut &str) -> Option<Value> {
    if !eat(s, "{") {
        return None;
    }
    let mut map = Map::new();
    skip_ws(s);
    if eat(s, "}") {
        return Some(Value::Object(map));
    }
    loop {
        skip_ws(s);
        // A key runs to its colon (llama.cpp's `[^:}]+`).
        let end = s.find([':', '}'])?;
        if !s[end..].starts_with(':') || end == 0 {
            return None;
        }
        let key = s[..end].trim().to_string();
        *s = &s[end + 1..];
        skip_ws(s);
        let v = value(s)?;
        map.insert(key, v);
        skip_ws(s);
        if eat(s, ",") {
            continue;
        }
        return eat(s, "}").then_some(Value::Object(map));
    }
}

/// `[` (value (`,` value)*)? `]`
fn list(s: &mut &str) -> Option<Value> {
    if !eat(s, "[") {
        return None;
    }
    let mut items = Vec::new();
    skip_ws(s);
    if eat(s, "]") {
        return Some(Value::Array(items));
    }
    loop {
        skip_ws(s);
        items.push(value(s)?);
        skip_ws(s);
        if eat(s, ",") {
            continue;
        }
        return eat(s, "]").then_some(Value::Array(items));
    }
}

fn value(s: &mut &str) -> Option<Value> {
    if eat(s, QUOTE) {
        let end = s.find(QUOTE)?;
        let text = s[..end].to_string();
        *s = &s[end + QUOTE.len()..];
        return Some(Value::String(text));
    }
    if s.starts_with('{') {
        return dict(s);
    }
    if s.starts_with('[') {
        return list(s);
    }
    // A JSON number, true, false or null: up to the next delimiter.
    let end = s.find([',', '}', ']']).unwrap_or(s.len());
    let lit = s[..end].trim();
    let v: Value = serde_json::from_str(lit).ok()?;
    if v.is_object() || v.is_array() || v.is_string() {
        return None;
    }
    *s = &s[end..];
    Some(v)
}

#[cfg(test)]
mod tests;
