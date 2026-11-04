//! Strict tool calls: the body of a call in the Qwen3-Coder format, which Qwen3.5 and later
//! templates teach and [`crate::engine::tool_parser::Qwen3CoderParser`] reads:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! The grammar covers what follows the `<tool_call>` token up to (not including) the
//! `</tool_call>` token, which the caller makes the grammar's only end token. `NAME` is one of
//! the request's tools. A strict tool's parameters follow its schema: its `properties` in
//! declaration order (the required ones always, the others optionally), nothing else. A value
//! the parser reads as a string (the property's `type` is or includes `"string"`, or it has no
//! `type`) is raw text up to `\n</parameter>` (or one of its `enum` / `const` strings); any other
//! value is JSON for its schema, as for `response_format`. A tool that is not strict takes any
//! parameters, so only its name is checked.

use serde_json::{Map, Value};

use super::schema::{self, Node};
use super::{GrammarError, Options};

/// One of a request's tools, for [`super::GrammarCompiler::compile_tool_calls`].
#[derive(Debug, Clone, Copy)]
pub struct ToolSpec<'a> {
    pub name: &'a str,
    /// The parameters' JSON schema of a strict tool; `None`: any parameters.
    pub strict: Option<&'a Value>,
}

const VALUE_END: &[u8] = b"\n</parameter>";

fn lit(s: &str) -> Node {
    Node::Lit(s.as_bytes().to_vec())
}

/// A raw value: text up to `\n</parameter>`. The parser cuts a value at its first
/// `</parameter>` and a call at its first `</tool_call>`, so neither is inside.
fn raw_value() -> Node {
    Node::RawUntil {
        end: VALUE_END.to_vec(),
        forbid: vec![b"</parameter>".to_vec(), b"</tool_call>".to_vec()],
    }
}

fn unsupported<T>(msg: impl Into<String>) -> Result<T, GrammarError> {
    Err(GrammarError::Unsupported(msg.into()))
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, GrammarError> {
    Err(GrammarError::Invalid(msg.into()))
}

/// The body grammar of a call to one of `tools`.
pub(crate) fn qwen3_coder(tools: &[ToolSpec<'_>], opts: &Options) -> Result<Node, GrammarError> {
    if tools.is_empty() {
        return invalid("no tools to call");
    }
    let mut alts = Vec::with_capacity(tools.len());
    for t in tools {
        if t.name.is_empty() || t.name.contains(['>', '\n']) {
            return unsupported(format!("tool name {:?}", t.name));
        }
        let params = match t.strict {
            Some(schema) => strict_params(schema, opts)
                .map_err(|e| prefix(e, &format!("tool {:?}: ", t.name)))?,
            None => free_params(),
        };
        if matches!(params, Node::Never) {
            continue;
        }
        alts.push(Node::Seq(vec![
            lit(&format!("{}>\n", t.name)),
            params,
            lit("</function>"),
            Node::Opt(Box::new(lit("\n"))),
        ]));
    }
    let names = schema::alt(alts);
    if matches!(names, Node::Never) {
        return Err(GrammarError::Unsatisfiable);
    }
    Ok(Node::Seq(vec![lit("\n<function="), names]))
}

fn prefix(e: GrammarError, p: &str) -> GrammarError {
    match e {
        GrammarError::Invalid(m) => GrammarError::Invalid(format!("{p}{m}")),
        GrammarError::Unsupported(m) => GrammarError::Unsupported(format!("{p}{m}")),
        other => other,
    }
}

/// Any parameters: `<parameter=KEY>\nVALUE\n</parameter>\n`, any number of them.
fn free_params() -> Node {
    Node::Star(Box::new(Node::Seq(vec![
        lit("<parameter="),
        Node::RawUntil {
            end: b">".to_vec(),
            forbid: Vec::new(),
        },
        lit("\n"),
        raw_value(),
        lit("\n"),
    ])))
}

/// A strict tool's parameters. Unsatisfiable: [`Node::Never`].
fn strict_params(root: &Value, opts: &Options) -> Result<Node, GrammarError> {
    let m = match root {
        Value::Null => return Ok(Node::Seq(Vec::new())),
        Value::Object(m) => m,
        _ => return invalid("`parameters` must be an object"),
    };
    for k in m.keys() {
        let ok = matches!(
            k.as_str(),
            "type" | "properties" | "required" | "additionalProperties"
        ) || schema::ANNOTATIONS_OK.contains(&k.as_str());
        if !ok {
            return unsupported(format!("`{k}` in `parameters`"));
        }
    }
    match m.get("type") {
        None => {}
        Some(Value::String(t)) if t == "object" => {}
        Some(_) => return unsupported("`parameters` that are not an object"),
    }
    let empty = Map::new();
    let props = match m.get("properties") {
        None => &empty,
        Some(Value::Object(p)) => p,
        Some(_) => return invalid("`properties` must be an object"),
    };
    let mut required: Vec<&str> = Vec::new();
    match m.get("required") {
        None => {}
        Some(Value::Array(a)) => {
            for r in a {
                match r.as_str() {
                    Some(s) if props.contains_key(s) => required.push(s),
                    Some(s) => {
                        return unsupported(format!(
                            "required property {s:?} is not in `properties`"
                        ));
                    }
                    None => return invalid("`required` entries must be strings"),
                }
            }
        }
        Some(_) => return invalid("`required` must be an array"),
    }
    match m.get("additionalProperties") {
        None | Some(Value::Bool(false)) => {}
        Some(_) if props.is_empty() => return Ok(free_params()),
        Some(_) => {
            return unsupported(
                "`additionalProperties` other than false together with `properties`",
            );
        }
    }
    let mut items = Vec::with_capacity(props.len());
    for (key, schema) in props {
        if key.is_empty() || key.contains(['>', '\n']) {
            return unsupported(format!("parameter name {key:?}"));
        }
        let req = required.contains(&key.as_str());
        let value = param_value(root, schema, opts)
            .map_err(|e| prefix(e, &format!("parameter {key:?}: ")))?;
        if matches!(value, Node::Never) {
            if req {
                return Ok(Node::Never);
            }
            continue;
        }
        let item = Node::Seq(vec![lit(&format!("<parameter={key}>\n")), value, lit("\n")]);
        items.push(if req { item } else { Node::Opt(Box::new(item)) });
    }
    Ok(Node::Seq(items))
}

/// One parameter's value and its `\n</parameter>` end.
fn param_value(root: &Value, schema: &Value, opts: &Options) -> Result<Node, GrammarError> {
    let end = || Node::Lit(VALUE_END.to_vec());
    let m = match schema {
        Value::Bool(true) => return Ok(raw_value()),
        Value::Bool(false) => return Ok(Node::Never),
        Value::Object(m) => m,
        _ => return invalid("a schema must be an object or a boolean"),
    };
    let raw = match m.get("type") {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        Some(_) => return invalid("`type` must be a string or an array"),
        None => {
            // The parser reads an untyped value as a string: only what is surely one.
            for k in m.keys() {
                if !matches!(k.as_str(), "enum" | "const")
                    && !schema::ANNOTATIONS_OK.contains(&k.as_str())
                {
                    return unsupported(format!("`{k}` without a `type`"));
                }
            }
            true
        }
    };
    if !raw {
        let value = schema::lower_at(root, schema, opts)?;
        if matches!(value, Node::Never) {
            return Ok(Node::Never);
        }
        return Ok(Node::Seq(vec![value, end()]));
    }
    // Raw text: the string itself, or the JSON text of a non-string `enum` entry.
    let text = |v: &Value| match v {
        Value::String(s) => s.as_bytes().to_vec(),
        other => serde_json::to_vec(other).expect("a JSON value serializes"),
    };
    if let Some(c) = m.get("const") {
        return Ok(Node::Seq(vec![Node::Lit(text(c)), end()]));
    }
    if let Some(e) = m.get("enum") {
        let Some(items) = e.as_array() else {
            return invalid("`enum` must be an array");
        };
        let alts = schema::alt(items.iter().map(|v| Node::Lit(text(v))).collect());
        if matches!(alts, Node::Never) {
            return Ok(Node::Never);
        }
        return Ok(Node::Seq(vec![alts, end()]));
    }
    for k in m.keys() {
        if matches!(k.as_str(), "type" | "format") || schema::ANNOTATIONS_OK.contains(&k.as_str()) {
            continue;
        }
        return unsupported(format!("`{k}` on a string parameter"));
    }
    Ok(raw_value())
}
