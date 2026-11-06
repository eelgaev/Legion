//! GLM tool calls (GLM-4.5 and later, GLM-5.3-Flash):
//!
//! ```text
//! <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
//! ```
//!
//! with optional newlines between the pieces. The chat template writes a string argument as is
//! and any other one as JSON, so a value is read as JSON unless the tool's schema types the
//! parameter as a string (or does not mention it), as in the Qwen3-Coder format; a value that
//! does not parse stays a string. Follows vllm's `glm4_moe_tool_parser.py` (Apache-2.0).

use serde_json::{Map, Value};

use super::{Core, Tool, is_string_param};

const KEY_OPEN: &str = "<arg_key>";
const KEY_CLOSE: &str = "</arg_key>";
const VALUE_OPEN: &str = "<arg_value>";
const VALUE_CLOSE: &str = "</arg_value>";

pub struct GlmParser(pub(super) Core);

impl GlmParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self(Core::new(parse_glm, tools))
    }
}

/// `NAME<arg_key>...` into `(name, arguments json)`.
fn parse_glm(body: &str, tools: &[Tool]) -> Option<(String, String)> {
    let body = body.trim();
    let end = body.find(KEY_OPEN).unwrap_or(body.len());
    let name = body[..end].trim();
    if name.is_empty() || name.contains(['<', '>', '\n']) {
        return None;
    }
    let tool = tools.iter().find(|t| t.name == name)?;
    let mut rest = &body[end..];
    let mut args = Map::new();
    loop {
        rest = rest.trim_start();
        let Some(after) = rest.strip_prefix(KEY_OPEN) else {
            break;
        };
        let close = after.find(KEY_CLOSE)?;
        let key = after[..close].trim().to_string();
        let after = after[close + KEY_CLOSE.len()..].trim_start();
        let after = after.strip_prefix(VALUE_OPEN)?;
        let close = after.find(VALUE_CLOSE)?;
        let raw = &after[..close];
        let value = if is_string_param(tool, &key) {
            Value::String(raw.to_string())
        } else {
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        args.insert(key, value);
        rest = &after[close + VALUE_CLOSE.len()..];
    }
    if !rest.trim().is_empty() {
        return None;
    }
    Some((
        name.to_string(),
        serde_json::to_string(&Value::Object(args)).ok()?,
    ))
}

#[cfg(test)]
mod tests;
