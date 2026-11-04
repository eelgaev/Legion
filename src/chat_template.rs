//! Chat template rendering: turns a conversation into the model's prompt text with the Jinja
//! template stored in the GGUF (`tokenizer.chat_template`), the way HF transformers'
//! `apply_chat_template(..., tokenize=False)` does.
//!
//! HF compiles templates in a sandboxed Jinja environment with `trim_blocks` and `lstrip_blocks`,
//! a `raise_exception` function, a `strftime_now` function and a `tojson` filter that keeps
//! non-ASCII text and uses Python's default separators. This module reproduces those with
//! minijinja plus its Python-method compatibility layer (`str.startswith`, `split`, ...).

use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde};
use minijinja::{Environment, Error, ErrorKind, State, Value};
use serde_json::Map;

use crate::engine::{Message, Role, Tool};

const TEMPLATE_NAME: &str = "chat_template";
const RESERVED: [&str; 6] = [
    "messages",
    "tools",
    "documents",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
];

#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    /// The template source does not compile.
    #[error("invalid chat template: {0}")]
    Compile(String),
    /// Rendering failed, including `raise_exception(...)` calls made by the template.
    #[error("chat template error: {0}")]
    Render(String),
    /// A message cannot be mapped to the HF message shape.
    #[error("invalid message: {0}")]
    Message(String),
}

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Compiles `source`. `bos` and `eos` are the token texts exposed to the template as
    /// `bos_token` and `eos_token` (empty if the model has none).
    pub fn new(source: &str, bos: &str, eos: &str) -> Result<Self, ChatTemplateError> {
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| ChatTemplateError::Compile(e.to_string()))?,
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        env.add_filter("int", int_filter);
        env.add_test("iterable", is_iterable);
        env.add_template_owned(TEMPLATE_NAME, source.to_owned())
            .map_err(|e| ChatTemplateError::Compile(format!("{e:#}")))?;
        Ok(Self {
            env,
            bos: bos.to_owned(),
            eos: eos.to_owned(),
        })
    }

    /// Renders the conversation. `system`, if set, becomes a leading system message. `kwargs`
    /// are extra template variables such as `enable_thinking`; the built-in variables
    /// (`messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token`) win over them.
    pub fn render(
        &self,
        system: Option<&str>,
        messages: &[Message],
        tools: &[Tool],
        add_generation_prompt: bool,
        kwargs: &Map<String, serde_json::Value>,
    ) -> Result<String, ChatTemplateError> {
        let mut hf_messages = Vec::with_capacity(messages.len() + 1);
        if let Some(system) = system {
            hf_messages.push(Value::from_pairs([("role", "system"), ("content", system)]));
        }
        for m in messages {
            hf_messages.push(hf_message(m)?);
        }
        let hf_tools = if tools.is_empty() {
            Value::from(())
        } else {
            tools.iter().map(hf_tool).collect()
        };

        let mut ctx: Vec<(String, Value)> = kwargs
            .iter()
            .map(|(k, v)| (k.clone(), Value::from(Serde(v))))
            .collect();
        ctx.retain(|(k, _)| !RESERVED.contains(&k.as_str()));
        ctx.extend([
            ("messages".to_owned(), Value::from(hf_messages)),
            ("tools".to_owned(), hf_tools),
            ("documents".to_owned(), Value::from(())),
            (
                "add_generation_prompt".to_owned(),
                Value::from(add_generation_prompt),
            ),
            ("bos_token".to_owned(), Value::from(self.bos.as_str())),
            ("eos_token".to_owned(), Value::from(self.eos.as_str())),
        ]);

        let render_err = |e: Error| ChatTemplateError::Render(format!("{e:#}"));
        let template = self.env.get_template(TEMPLATE_NAME).map_err(render_err)?;
        template.render(Value::from_pairs(ctx)).map_err(render_err)
    }
}

/// Our `Message` as the dict HF templates expect. Keys are kept in HF's insertion order (the
/// minijinja `preserve_order` feature), which matters when a template dumps a dict with `tojson`.
fn hf_message(m: &Message) -> Result<Value, ChatTemplateError> {
    let role = match m.role {
        Some(Role::System) => "system",
        Some(Role::User) => "user",
        Some(Role::Assistant) => "assistant",
        Some(Role::Tool) => "tool",
        None => return Err(ChatTemplateError::Message("message has no role".into())),
    };
    // A message with images is HF's list of content parts: the images first (in model order,
    // as [`Message::images`] says), then the text. Vision templates (Qwen3.5's) render each
    // image part as its placeholder, e.g. `<|vision_start|><|image_pad|><|vision_end|>`.
    let content: Value = if m.images.is_empty() {
        m.content.as_str().into()
    } else {
        let mut parts: Vec<Value> = m
            .images
            .iter()
            .map(|_| Value::from_pairs([("type", "image")]))
            .collect();
        if !m.content.is_empty() {
            parts.push(Value::from_pairs([
                ("type", "text"),
                ("text", m.content.as_str()),
            ]));
        }
        parts.into()
    };
    let mut d: Vec<(&str, Value)> = vec![("role", role.into()), ("content", content)];
    if let Some(reasoning) = &m.reasoning {
        d.push(("reasoning_content", reasoning.as_str().into()));
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                // Templates iterate over the arguments, so they must be an object. Arguments
                // that are not valid JSON are passed through as the raw string.
                let arguments = if c.arguments.trim().is_empty() {
                    Value::from_pairs(Vec::<(&str, Value)>::new())
                } else {
                    match serde_json::from_str::<serde_json::Value>(&c.arguments) {
                        Ok(v) => Value::from(Serde(v)),
                        Err(_) => Value::from(c.arguments.as_str()),
                    }
                };
                Value::from_pairs([
                    ("id", Value::from(c.id.as_str())),
                    ("type", Value::from("function")),
                    (
                        "function",
                        Value::from_pairs([
                            ("name", Value::from(c.name.as_str())),
                            ("arguments", arguments),
                        ]),
                    ),
                ])
            })
            .collect();
        d.push(("tool_calls", calls.into()));
    }
    if let Some(id) = &m.tool_call_id {
        d.push(("tool_call_id", id.as_str().into()));
    }
    Ok(Value::from_pairs(d))
}

/// A tool in the OpenAI shape that HF templates take.
fn hf_tool(t: &Tool) -> Value {
    let mut f: Vec<(&str, Value)> = vec![("name", t.name.as_str().into())];
    if let Some(description) = &t.description {
        f.push(("description", description.as_str().into()));
    }
    f.push(("parameters", Value::from(Serde(&t.parameters))));
    Value::from_pairs([
        ("type", Value::from("function")),
        ("function", Value::from_pairs(f)),
    ])
}

/// Jinja2's `iterable` test: `none` (what HF passes as `tools` when there are none) is not
/// iterable. minijinja's own test iterates `none` as empty, so templates that guard with
/// `tools is iterable and tools | length > 0` (Nemotron-H's) would then fail on `length`.
fn is_iterable(v: Value) -> bool {
    !v.is_none() && !v.is_undefined() && v.try_iter().is_ok()
}

/// Jinja2's `int`: a value that does not convert gives the default, 0 unless one is passed
/// (minijinja's raises without one), and strings may have spaces around the number, as for
/// Python's `int()`. DeepSeek-V4.1-Flash's template checks `reasoning_effort` with
/// `(x | int | string) == x` and raises its own message when that is false.
fn int_filter(
    state: &State,
    value: &Value,
    default: Option<Value>,
    base: Option<u32>,
    kwargs: Kwargs,
) -> Result<Value, Error> {
    let default = match default {
        Some(d) => d,
        None => kwargs
            .get::<Option<Value>>("default")?
            .unwrap_or(Value::from(0)),
    };
    let trimmed = value.as_str().map(|s| Value::from(s.trim()));
    let value = trimmed.as_ref().unwrap_or(value);
    minijinja::filters::int(state, value, Some(default), base, kwargs)
}

fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `strftime_now(format)`: the current time in UTC (HF uses local time).
fn strftime_now(state: &State, format: String) -> Result<String, Error> {
    let kwargs = Kwargs::from_iter([("format", Value::from(format))]);
    minijinja_contrib::filters::datetimeformat(state, minijinja_contrib::globals::now(), kwargs)
}

/// HF's `tojson`: `json.dumps(x, ensure_ascii=False, indent=None, sort_keys=False)`, i.e.
/// `", "` and `": "` separators, no HTML escaping, and no `safe` marking beyond what Python does.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = match kwargs.get::<Option<Value>>("indent")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(usize::try_from(v).map_err(|_| {
            Error::new(
                ErrorKind::InvalidOperation,
                "tojson: indent must be a non-negative integer",
            )
        })?),
    };
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    kwargs.assert_all_used()?;

    let ser_err = |e: serde_json::Error| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("cannot serialize to JSON: {e}"),
        )
    };
    // Serialize the minijinja value directly: going through `serde_json::Value` would sort keys.
    let json = if sort_keys {
        let mut json = serde_json::to_value(&value).map_err(ser_err)?;
        sort_json_keys(&mut json);
        Value::from(Serde(json))
    } else {
        value
    };
    let mut out = Vec::new();
    let result = match indent {
        Some(n) => {
            let pad = vec![b' '; n];
            let mut ser = serde_json::Serializer::with_formatter(
                &mut out,
                PyFormatter {
                    ensure_ascii,
                    inner: serde_json::ser::PrettyFormatter::with_indent(&pad),
                },
            );
            serde::Serialize::serialize(&json, &mut ser)
        }
        None => {
            let mut ser = serde_json::Serializer::with_formatter(
                &mut out,
                PyFormatter {
                    ensure_ascii,
                    inner: SpacedFormatter,
                },
            );
            serde::Serialize::serialize(&json, &mut ser)
        }
    };
    result.map_err(|e| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("cannot serialize to JSON: {e}"),
        )
    })?;
    let text = String::from_utf8(out).map_err(|e| {
        Error::new(
            ErrorKind::InvalidOperation,
            format!("cannot serialize to JSON: {e}"),
        )
    })?;
    // Python's tojson in HF output is plain text, not HTML-escaped.
    Ok(Value::from_safe_string(text))
}

fn sort_json_keys(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = std::mem::take(map).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, child) in &mut entries {
                sort_json_keys(child);
            }
            map.extend(entries);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(sort_json_keys),
        _ => {}
    }
}

/// Compact output with Python's default `", "` / `": "` separators.
struct SpacedFormatter;

impl serde_json::ser::Formatter for SpacedFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Wraps a layout formatter and optionally escapes non-ASCII text as Python's `ensure_ascii`.
struct PyFormatter<F> {
    ensure_ascii: bool,
    inner: F,
}

macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*);)*) => {$(
        fn $name<W: ?Sized + std::io::Write>(&mut self, w: &mut W $(, $arg: $ty)*) -> std::io::Result<()> {
            self.inner.$name(w $(, $arg)*)
        }
    )*};
}

impl<F: serde_json::ser::Formatter> serde_json::ser::Formatter for PyFormatter<F> {
    forward! {
        begin_array(); end_array(); begin_array_value(first: bool); end_array_value();
        begin_object(); end_object(); begin_object_key(first: bool); end_object_key();
        begin_object_value(); end_object_value();
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        if !self.ensure_ascii || fragment.is_ascii() {
            return w.write_all(fragment.as_bytes());
        }
        let mut buf = [0u16; 2];
        for ch in fragment.chars() {
            if ch.is_ascii() {
                w.write_all(&[ch as u8])?;
            } else {
                for unit in ch.encode_utf16(&mut buf) {
                    write!(w, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}
