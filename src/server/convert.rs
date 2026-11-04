//! Wire requests -> [`GenerateRequest`], including validation shared by all front ends.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use tokio_util::sync::CancellationToken;

use super::AppState;
use super::error::ApiError;
use crate::api::{anthropic, openai};
use crate::engine::queue::Scheduling;
use crate::engine::sampling_defaults::SamplingDefaults;
use crate::engine::{
    GenerateRequest, ImageInput, MAX_LOGPROBS, Message, Prompt, Role, SamplingParams, Tool,
    ToolCall,
};

/// The penalty parameters of a request; `None` keeps the engine default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Penalties {
    pub presence: Option<f32>,
    pub frequency: Option<f32>,
    pub repetition: Option<f32>,
}

/// Range checks for sampling parameters, OpenAI's limits.
pub fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    n: Option<u32>,
    penalties: Penalties,
) -> Result<(), ApiError> {
    if let Some(t) = temperature
        && !(0.0..=2.0).contains(&t)
    {
        return Err(ApiError::invalid_param(
            "temperature",
            format!("temperature must be between 0 and 2, got {t}"),
        ));
    }
    if let Some(p) = top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(ApiError::invalid_param(
            "top_p",
            format!("top_p must be in (0, 1], got {p}"),
        ));
    }
    if let Some(p) = min_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(ApiError::invalid_param(
            "min_p",
            format!("min_p must be between 0 and 1, got {p}"),
        ));
    }
    if let Some(p) = penalties.presence
        && !(-2.0..=2.0).contains(&p)
    {
        return Err(ApiError::invalid_param(
            "presence_penalty",
            format!("presence_penalty must be between -2 and 2, got {p}"),
        ));
    }
    if let Some(p) = penalties.frequency
        && !(-2.0..=2.0).contains(&p)
    {
        return Err(ApiError::invalid_param(
            "frequency_penalty",
            format!("frequency_penalty must be between -2 and 2, got {p}"),
        ));
    }
    if let Some(p) = penalties.repetition
        && !(p.is_finite() && p > 0.0)
    {
        return Err(ApiError::invalid_param(
            "repetition_penalty",
            format!("repetition_penalty must be greater than 0, got {p}"),
        ));
    }
    if let Some(n) = n
        && n != 1
    {
        return Err(ApiError::invalid_param("n", "only n=1 is supported"));
    }
    Ok(())
}

/// The request's sampling fields over the model's defaults `d`.
#[allow(clippy::too_many_arguments)]
fn sampling(
    d: &SamplingDefaults,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    seed: Option<u64>,
    penalties: Penalties,
    logprobs: Option<u32>,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(d.temperature.value),
        top_p: top_p.unwrap_or(d.top_p.value),
        top_k: top_k.unwrap_or(d.top_k.value),
        min_p: min_p.unwrap_or(d.min_p.value),
        seed,
        presence_penalty: penalties.presence.unwrap_or(d.presence_penalty.value),
        frequency_penalty: penalties.frequency.unwrap_or(d.frequency_penalty.value),
        repetition_penalty: penalties.repetition.unwrap_or(d.repetition_penalty.value),
        logprobs,
    }
}

fn penalties(presence: Option<f32>, frequency: Option<f32>, repetition: Option<f32>) -> Penalties {
    Penalties {
        presence,
        frequency,
        repetition,
    }
}

fn max_tokens(value: Option<u32>, default: u32, param: &str) -> Result<u32, ApiError> {
    match value {
        Some(0) => Err(ApiError::invalid_param(
            param,
            format!("{param} must be at least 1"),
        )),
        Some(v) => Ok(v),
        None => Ok(default),
    }
}

/// `min_tokens` (vLLM extension): off by default, at most the effective `max_tokens`.
fn min_tokens(value: Option<u32>, max: u32) -> Result<u32, ApiError> {
    match value {
        Some(n) if n > max => Err(ApiError::invalid_param(
            "min_tokens",
            format!("min_tokens must be at most max_tokens ({max}), got {n}"),
        )),
        Some(n) => Ok(n),
        None => Ok(0),
    }
}

/// Largest decoded image accepted, per image.
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Largest number of images in one request, across all messages.
const MAX_IMAGES: usize = 16;

/// Image mime types the server accepts.
const IMAGE_MIMES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];

/// Decodes a `data:<mime>;base64,<data>` URL. Remote URLs are rejected outright: the server
/// never fetches.
fn image_from_data_url(param: &str, url: &str) -> Result<ImageInput, ApiError> {
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(ApiError::invalid_param(
            param,
            "remote image URLs are not supported; send a data: URL",
        ));
    }
    let (mime, data) = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .ok_or_else(|| ApiError::invalid_param(param, "images must be sent as data: URLs"))?;
    decode_image(param, mime, data)
}

/// Checks the mime type and decodes the base64 payload of one image.
fn decode_image(param: &str, mime: &str, data: &str) -> Result<ImageInput, ApiError> {
    if !IMAGE_MIMES.contains(&mime) {
        return Err(ApiError::invalid_param(
            param,
            format!("unsupported image type {mime}"),
        ));
    }
    let bytes = BASE64
        .decode(data)
        .map_err(|_| ApiError::invalid_param(param, "invalid base64 image data"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ApiError::invalid_param(param, "image too large"));
    }
    Ok(ImageInput {
        bytes,
        mime: mime.to_string(),
    })
}

/// Anthropic image block source: only base64 sources are accepted; a `url` source would make
/// the server fetch, which it never does.
fn anthropic_image(param: &str, source: &serde_json::Value) -> Result<ImageInput, ApiError> {
    let field = |name: &str| source.get(name).and_then(|v| v.as_str());
    match field("type") {
        Some("base64") => {}
        Some("url") => {
            return Err(ApiError::invalid_param(
                param,
                "remote image URLs are not supported; send a data: URL",
            ));
        }
        Some(other) => {
            return Err(ApiError::invalid_param(
                param,
                format!("image source type `{other}` is not supported; expected `base64`"),
            ));
        }
        None => {
            return Err(ApiError::invalid_param(
                param,
                "image source requires a `type`",
            ));
        }
    }
    decode_image(
        param,
        field("media_type").unwrap_or(""),
        field("data").unwrap_or(""),
    )
}

/// OpenAI chat `image_url`: an object with a `url` field, or the URL as a plain string.
fn openai_image_url(param: &str, value: &serde_json::Value) -> Result<ImageInput, ApiError> {
    let url = match value {
        serde_json::Value::String(s) => s.as_str(),
        serde_json::Value::Object(map) => map
            .get("url")
            .and_then(|u| u.as_str())
            .ok_or_else(|| ApiError::invalid_param(param, "image_url requires a `url`"))?,
        _ => {
            return Err(ApiError::invalid_param(
                param,
                "image_url must be an object with `url` or a URL string",
            ));
        }
    };
    image_from_data_url(param, url)
}

/// Per-request image checks, run once the messages are assembled: the total image count and,
/// when the request carries images, that the resolved model supports vision.
fn check_images(state: &AppState, requested: &str, messages: &[Message]) -> Result<(), ApiError> {
    let total: usize = messages.iter().map(|m| m.images.len()).sum();
    if total > MAX_IMAGES {
        return Err(ApiError::invalid("too many images"));
    }
    if total == 0 {
        return Ok(());
    }
    let resolved = state.resolve_model(requested)?;
    let vision = state
        .engine
        .models()
        .iter()
        .any(|m| m.id == resolved.engine_id && m.vision);
    if !vision {
        return Err(ApiError::invalid(format!(
            "model {requested} does not accept images"
        )));
    }
    Ok(())
}

pub fn chat_request(
    state: &AppState,
    req: &openai::ChatCompletionRequest,
) -> Result<GenerateRequest, ApiError> {
    validate_sampling(
        req.temperature,
        req.top_p,
        req.min_p,
        req.n,
        penalties(
            req.presence_penalty,
            req.frequency_penalty,
            req.repetition_penalty,
        ),
    )?;
    if req.messages.is_empty() {
        return Err(ApiError::invalid_param(
            "messages",
            "messages must not be empty",
        ));
    }
    let max = max_tokens(
        req.max_completion_tokens.or(req.max_tokens),
        state.config.default_max_tokens,
        "max_tokens",
    )?;
    let mut messages = Vec::with_capacity(req.messages.len());
    for (i, m) in req.messages.iter().enumerate() {
        messages.push(openai_message(i, m)?);
    }
    check_images(state, &req.model, &messages)?;
    let tools = req
        .tools
        .iter()
        .map(|t| Tool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
            strict: t.function.strict == Some(true),
        })
        .collect();
    let template_kwargs = req.chat_template_kwargs.clone().unwrap_or_default();
    // `chat_template_kwargs.enable_thinking` wins over `reasoning_effort`.
    let thinking = match template_kwargs.get("enable_thinking") {
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(_) => {
            return Err(ApiError::invalid_param(
                "chat_template_kwargs.enable_thinking",
                "`enable_thinking` must be a boolean",
            ));
        }
        None => req
            .reasoning_effort
            .as_deref()
            .map(|e| !matches!(e, "none" | "minimal")),
    };
    Ok(GenerateRequest {
        model: state.resolve_model(&req.model)?.engine_id,
        prompt: Prompt::Chat {
            system: None,
            messages,
            tools,
        },
        sampling: sampling(
            &state.sampling_defaults(&req.model)?,
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.seed,
            penalties(
                req.presence_penalty,
                req.frequency_penalty,
                req.repetition_penalty,
            ),
            chat_logprobs(req)?,
        ),
        max_tokens: max,
        stop: req.stop.clone().map(|s| s.into_vec()).unwrap_or_default(),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        min_tokens: min_tokens(req.min_tokens, max)?,
        thinking,
        response_format: super::response_format::parse(
            state,
            req.response_format.as_ref(),
            "response_format",
        )?,
        template_kwargs,
        cancel: CancellationToken::new(),
        scheduling: scheduling(req.priority),
    })
}

/// A request's place in the engine's queue: its `priority` (lower first, default 0). The
/// fair-share key is set by the route, which sees the caller (`super::fair_share`).
fn scheduling(priority: Option<i32>) -> Scheduling {
    Scheduling {
        priority: priority.unwrap_or(0),
        share_key: None,
    }
}

/// `logprobs` + `top_logprobs` -> `SamplingParams::logprobs`.
fn chat_logprobs(req: &openai::ChatCompletionRequest) -> Result<Option<u32>, ApiError> {
    if !req.logprobs {
        return match req.top_logprobs {
            Some(_) => Err(ApiError::invalid_param(
                "top_logprobs",
                "top_logprobs requires logprobs to be true",
            )),
            None => Ok(None),
        };
    }
    check_logprobs(req.top_logprobs.unwrap_or(0), "top_logprobs").map(Some)
}

fn completion_logprobs(req: &openai::CompletionRequest) -> Result<Option<u32>, ApiError> {
    req.logprobs
        .map(|n| check_logprobs(n, "logprobs"))
        .transpose()
}

fn check_logprobs(n: u32, param: &str) -> Result<u32, ApiError> {
    if n > MAX_LOGPROBS {
        return Err(ApiError::invalid_param(
            param,
            format!("{param} must be between 0 and {MAX_LOGPROBS}, got {n}"),
        ));
    }
    Ok(n)
}

fn openai_message(i: usize, m: &openai::ChatMessage) -> Result<Message, ApiError> {
    let param = || format!("messages[{i}]");
    let role = match m.role.as_str() {
        "system" | "developer" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => {
            return Err(ApiError::invalid_param(
                &format!("messages[{i}].role"),
                format!("unknown role `{other}`"),
            ));
        }
    };
    let mut images = Vec::new();
    let content = match &m.content {
        None => String::new(),
        Some(openai::MessageContent::Text(t)) => t.clone(),
        Some(openai::MessageContent::Parts(parts)) => {
            let mut text = String::new();
            for (j, p) in parts.iter().enumerate() {
                match p {
                    openai::ContentPart::Text { text: t } => text.push_str(t),
                    openai::ContentPart::ImageUrl { image_url } => {
                        let cparam = format!("messages[{i}].content[{j}]");
                        images.push(openai_image_url(&cparam, image_url)?);
                    }
                    _ => {
                        return Err(ApiError::invalid_param(
                            &param(),
                            "only text content is supported for now",
                        ));
                    }
                }
            }
            text
        }
    };
    Ok(Message {
        role: Some(role),
        content,
        images,
        reasoning: m.reasoning_content.clone(),
        tool_calls: m
            .tool_calls
            .iter()
            .map(|c| ToolCall {
                id: c.id.clone(),
                name: c.function.name.clone(),
                arguments: c.function.arguments.clone(),
            })
            .collect(),
        tool_call_id: m.tool_call_id.clone(),
    })
}

pub fn completion_request(
    state: &AppState,
    req: &openai::CompletionRequest,
) -> Result<GenerateRequest, ApiError> {
    validate_sampling(
        req.temperature,
        req.top_p,
        req.min_p,
        req.n,
        penalties(
            req.presence_penalty,
            req.frequency_penalty,
            req.repetition_penalty,
        ),
    )?;
    let prompt = match req.prompt.clone().into_vec().as_slice() {
        [one] => one.clone(),
        _ => {
            return Err(ApiError::invalid_param(
                "prompt",
                "exactly one prompt string is supported",
            ));
        }
    };
    // OpenAI's default for legacy completions.
    let max = max_tokens(req.max_tokens, 16, "max_tokens")?;
    Ok(GenerateRequest {
        model: state.resolve_model(&req.model)?.engine_id,
        prompt: Prompt::Text(prompt),
        sampling: sampling(
            &state.sampling_defaults(&req.model)?,
            req.temperature,
            req.top_p,
            req.top_k,
            req.min_p,
            req.seed,
            penalties(
                req.presence_penalty,
                req.frequency_penalty,
                req.repetition_penalty,
            ),
            completion_logprobs(req)?,
        ),
        max_tokens: max,
        stop: req.stop.clone().map(|s| s.into_vec()).unwrap_or_default(),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        min_tokens: min_tokens(req.min_tokens, max)?,
        thinking: None,
        response_format: super::response_format::parse(
            state,
            req.response_format.as_ref(),
            "response_format",
        )?,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: scheduling(req.priority),
    })
}

pub fn messages_request(
    state: &AppState,
    req: &anthropic::MessagesRequest,
) -> Result<GenerateRequest, ApiError> {
    validate_sampling(req.temperature, req.top_p, None, None, Penalties::default())?;
    if req.temperature.is_some_and(|t| t > 1.0) {
        return Err(ApiError::invalid_param(
            "temperature",
            "temperature must be between 0 and 1",
        ));
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid_param(
            "messages",
            "messages must not be empty",
        ));
    }
    let system = match &req.system {
        None => None,
        Some(anthropic::SystemPrompt::Text(t)) => Some(t.clone()),
        Some(anthropic::SystemPrompt::Blocks(blocks)) => Some(
            blocks
                .iter()
                .filter_map(|b| match b {
                    anthropic::InputBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ),
    };
    let mut messages = Vec::new();
    for (i, m) in req.messages.iter().enumerate() {
        anthropic_message(i, m, &mut messages)?;
    }
    check_images(state, &req.model, &messages)?;
    let tools = req
        .tools
        .iter()
        .map(|t| Tool {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
            strict: false,
        })
        .collect();
    let thinking = req.thinking.as_ref().map(|t| t.kind != "disabled");
    Ok(GenerateRequest {
        model: state.resolve_model(&req.model)?.engine_id,
        prompt: Prompt::Chat {
            system,
            messages,
            tools,
        },
        sampling: sampling(
            &state.sampling_defaults(&req.model)?,
            req.temperature,
            req.top_p,
            req.top_k,
            None,
            None,
            Penalties::default(),
            None,
        ),
        max_tokens: max_tokens(Some(req.max_tokens), 0, "max_tokens")?,
        stop: req.stop_sequences.clone(),
        ignore_eos: false,
        min_tokens: 0,
        thinking,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: scheduling(req.priority),
    })
}

/// One Anthropic message can become several neutral ones: each `tool_result` block becomes a
/// `Role::Tool` message, placed before the user's remaining text. A `system` turn (sent by
/// Claude Code) stays one `Role::System` message at its position; only text blocks are allowed
/// in it.
fn anthropic_message(
    i: usize,
    m: &anthropic::InputMessage,
    out: &mut Vec<Message>,
) -> Result<(), ApiError> {
    let role = match m.role.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        other => {
            return Err(ApiError::invalid_param(
                &format!("messages[{i}].role"),
                format!("role must be `user` or `assistant`, got `{other}`"),
            ));
        }
    };
    if role == Role::System {
        // Join text blocks with "\n" exactly like the top-level `system` field.
        let content = match &m.content {
            anthropic::InputContent::Text(t) => t.clone(),
            anthropic::InputContent::Blocks(blocks) => {
                let mut parts = Vec::with_capacity(blocks.len());
                for b in blocks {
                    match b {
                        anthropic::InputBlock::Text { text } => parts.push(text.as_str()),
                        _ => {
                            return Err(ApiError::invalid_param(
                                &format!("messages[{i}].content"),
                                "system messages may only contain text",
                            ));
                        }
                    }
                }
                parts.join("\n")
            }
        };
        out.push(Message {
            role: Some(Role::System),
            content,
            ..Default::default()
        });
        return Ok(());
    }
    let mut msg = Message {
        role: Some(role),
        ..Default::default()
    };
    let blocks = match &m.content {
        anthropic::InputContent::Text(t) => {
            msg.content = t.clone();
            out.push(msg);
            return Ok(());
        }
        anthropic::InputContent::Blocks(b) => b,
    };
    for (j, b) in blocks.iter().enumerate() {
        match b {
            anthropic::InputBlock::Text { text } => msg.content.push_str(text),
            anthropic::InputBlock::Image { source } => {
                let cparam = format!("messages[{i}].content[{j}]");
                msg.images.push(anthropic_image(&cparam, source)?);
            }
            anthropic::InputBlock::Thinking { thinking, .. } => {
                msg.reasoning.get_or_insert_default().push_str(thinking)
            }
            anthropic::InputBlock::ToolUse { id, name, input } => msg.tool_calls.push(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: input.to_string(),
            }),
            anthropic::InputBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let mut text = match content {
                    None => String::new(),
                    Some(anthropic::ToolResultContent::Text(t)) => t.clone(),
                    Some(anthropic::ToolResultContent::Blocks(bs)) => bs
                        .iter()
                        .filter_map(|b| match b {
                            anthropic::InputBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                };
                if *is_error {
                    text = format!("Error: {text}");
                }
                out.push(Message {
                    role: Some(Role::Tool),
                    content: text,
                    tool_call_id: Some(tool_use_id.clone()),
                    ..Default::default()
                });
            }
            anthropic::InputBlock::Document { .. } => {
                return Err(ApiError::invalid_param(
                    &format!("messages[{i}].content"),
                    "only text content is supported for now",
                ));
            }
            anthropic::InputBlock::RedactedThinking { .. } | anthropic::InputBlock::Unknown => {}
        }
    }
    let empty = msg.content.is_empty()
        && msg.images.is_empty()
        && msg.tool_calls.is_empty()
        && msg.reasoning.is_none();
    if !empty {
        out.push(msg);
    }
    Ok(())
}

/// Stateless Responses API request -> neutral request. `instructions` become the system prompt;
/// reasoning, assistant text and function calls that follow each other merge into one assistant
/// message, as chat templates expect.
pub fn responses_request(
    state: &AppState,
    req: &openai::ResponsesRequest,
) -> Result<GenerateRequest, ApiError> {
    use openai::{InputContent, InputItem, InputPart, ResponsesInput};

    validate_sampling(req.temperature, req.top_p, None, None, Penalties::default())?;
    if req.previous_response_id.is_some() {
        return Err(ApiError::invalid_param(
            "previous_response_id",
            "previous_response_id is not supported: this server is stateless, send the full conversation in `input`",
        ));
    }
    if req.store == Some(true) {
        return Err(ApiError::invalid_param(
            "store",
            "store=true is not supported: this server is stateless, set `store` to false",
        ));
    }
    let items = match &req.input {
        ResponsesInput::Text(t) => {
            return responses_finish(state, req, vec![user_message(t.clone())]);
        }
        ResponsesInput::Items(items) if items.is_empty() => {
            return Err(ApiError::invalid_param("input", "input must not be empty"));
        }
        ResponsesInput::Items(items) => items,
    };

    let mut messages: Vec<Message> = Vec::new();
    let mut system: Vec<String> = req.instructions.iter().cloned().collect();
    // The assistant message being assembled from consecutive assistant-side items.
    let mut open: Option<Message> = None;
    let flush = |open: &mut Option<Message>, messages: &mut Vec<Message>| {
        if let Some(m) = open.take() {
            messages.push(m);
        }
    };
    let assistant = || Message {
        role: Some(Role::Assistant),
        ..Default::default()
    };
    for (i, item) in items.iter().enumerate() {
        let param = format!("input[{i}]");
        match item {
            InputItem::Message(m) => {
                let mut text = String::new();
                let mut images = Vec::new();
                match &m.content {
                    InputContent::Text(t) => text.push_str(t),
                    InputContent::Parts(parts) => {
                        for (j, p) in parts.iter().enumerate() {
                            match p {
                                InputPart::InputText { text: t }
                                | InputPart::OutputText { text: t } => text.push_str(t),
                                InputPart::InputImage { image_url } => {
                                    let cparam = format!("{param}.content[{j}]");
                                    images.push(image_from_data_url(&cparam, image_url)?);
                                }
                                InputPart::Unsupported => {
                                    return Err(ApiError::invalid_param(
                                        &param,
                                        "only text and image content are supported for now",
                                    ));
                                }
                            }
                        }
                    }
                }
                match m.role.as_str() {
                    "system" | "developer" => {
                        if !images.is_empty() {
                            return Err(ApiError::invalid_param(
                                &param,
                                "system messages may only contain text",
                            ));
                        }
                        flush(&mut open, &mut messages);
                        system.push(text);
                    }
                    "user" => {
                        flush(&mut open, &mut messages);
                        messages.push(Message {
                            images,
                            ..user_message(text)
                        });
                    }
                    "assistant" => {
                        let a = open.get_or_insert_with(assistant);
                        a.content.push_str(&text);
                        a.images.extend(images);
                    }
                    other => {
                        return Err(ApiError::invalid_param(
                            &format!("{param}.role"),
                            format!("unknown role `{other}`"),
                        ));
                    }
                }
            }
            InputItem::Reasoning { content, summary } => {
                let parts = if content.is_empty() { summary } else { content };
                let text: String = parts.iter().map(|p| p.text.as_str()).collect();
                if !text.is_empty() {
                    open.get_or_insert_with(assistant)
                        .reasoning
                        .get_or_insert_default()
                        .push_str(&text);
                }
            }
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => open
                .get_or_insert_with(assistant)
                .tool_calls
                .push(ToolCall {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
            InputItem::FunctionCallOutput { call_id, output } => {
                flush(&mut open, &mut messages);
                let content = match output {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                messages.push(Message {
                    role: Some(Role::Tool),
                    content,
                    tool_call_id: Some(call_id.clone()),
                    ..Default::default()
                });
            }
            InputItem::Unknown => {
                return Err(ApiError::invalid_param(
                    &format!("{param}.type"),
                    "unsupported input item type; supported: message, function_call, function_call_output, reasoning",
                ));
            }
        }
    }
    flush(&mut open, &mut messages);
    responses_finish_with_system(state, req, system, messages)
}

fn user_message(content: String) -> Message {
    Message {
        role: Some(Role::User),
        content,
        ..Default::default()
    }
}

fn responses_finish(
    state: &AppState,
    req: &openai::ResponsesRequest,
    messages: Vec<Message>,
) -> Result<GenerateRequest, ApiError> {
    let system = req.instructions.iter().cloned().collect();
    responses_finish_with_system(state, req, system, messages)
}

fn responses_finish_with_system(
    state: &AppState,
    req: &openai::ResponsesRequest,
    system: Vec<String>,
    messages: Vec<Message>,
) -> Result<GenerateRequest, ApiError> {
    check_images(state, &req.model, &messages)?;
    let mut tools = Vec::with_capacity(req.tools.len());
    for (i, t) in req.tools.iter().enumerate() {
        let bad = |msg: &str| ApiError::invalid_param(&format!("tools[{i}]"), msg.to_string());
        if t.get("type").and_then(|v| v.as_str()) != Some("function") {
            return Err(bad("only tools of type `function` are supported"));
        }
        let name = t
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| bad("function tool needs a `name`"))?;
        tools.push(Tool {
            name: name.to_string(),
            description: t
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from),
            parameters: t
                .get("parameters")
                .cloned()
                .filter(|v| !v.is_null())
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            // Only an explicit `strict: true` constrains the calls.
            strict: t.get("strict").and_then(|v| v.as_bool()) == Some(true),
        });
    }
    let thinking = req
        .reasoning
        .as_ref()
        .and_then(|r| r.effort.as_deref())
        .map(|e| !matches!(e, "none" | "minimal"));
    Ok(GenerateRequest {
        model: state.resolve_model(&req.model)?.engine_id,
        prompt: Prompt::Chat {
            system: (!system.is_empty()).then(|| system.join("\n\n")),
            messages,
            tools,
        },
        sampling: sampling(
            &state.sampling_defaults(&req.model)?,
            req.temperature,
            req.top_p,
            None,
            None,
            None,
            Penalties::default(),
            None,
        ),
        max_tokens: max_tokens(
            req.max_output_tokens,
            state.config.default_max_tokens,
            "max_output_tokens",
        )?,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking,
        response_format: super::response_format::responses(state, req.text.as_ref())?,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: scheduling(req.priority),
    })
}

#[cfg(test)]
mod sampling_tests;

#[cfg(test)]
mod strict_tools_tests;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::engine::mock::{MockConfig, MockEngine};

    fn state() -> AppState {
        AppState::new(Arc::new(MockEngine::new(MockConfig::default())))
    }

    fn pen(presence: Option<f32>, frequency: Option<f32>, repetition: Option<f32>) -> Penalties {
        Penalties {
            presence,
            frequency,
            repetition,
        }
    }

    /// name, expected `error.param` (`None` = accepted), [temperature, top_p, min_p], n, penalties
    type Case = (
        &'static str,
        Option<&'static str>,
        [Option<f32>; 3],
        Option<u32>,
        Penalties,
    );

    /// `None` = accepted, `Some(param)` = rejected with that `error.param`.
    #[test]
    fn validate_sampling_table() {
        let none = Penalties::default();
        let cases: Vec<Case> = vec![
            ("all unset", None, [None; 3], None, none),
            (
                "defaults",
                None,
                [Some(1.0), Some(1.0), Some(0.0)],
                Some(1),
                none,
            ),
            ("temperature 0", None, [Some(0.0), None, None], None, none),
            ("temperature 2", None, [Some(2.0), None, None], None, none),
            (
                "temperature -0.1",
                Some("temperature"),
                [Some(-0.1), None, None],
                None,
                none,
            ),
            (
                "temperature 2.1",
                Some("temperature"),
                [Some(2.1), None, None],
                None,
                none,
            ),
            (
                "temperature NaN",
                Some("temperature"),
                [Some(f32::NAN), None, None],
                None,
                none,
            ),
            ("top_p 1", None, [None, Some(1.0), None], None, none),
            (
                "top_p 0",
                Some("top_p"),
                [None, Some(0.0), None],
                None,
                none,
            ),
            (
                "top_p 1.1",
                Some("top_p"),
                [None, Some(1.1), None],
                None,
                none,
            ),
            ("min_p 0", None, [None, None, Some(0.0)], None, none),
            ("min_p 1", None, [None, None, Some(1.0)], None, none),
            (
                "min_p -0.1",
                Some("min_p"),
                [None, None, Some(-0.1)],
                None,
                none,
            ),
            (
                "min_p 1.1",
                Some("min_p"),
                [None, None, Some(1.1)],
                None,
                none,
            ),
            ("n 1", None, [None; 3], Some(1), none),
            ("n 2", Some("n"), [None; 3], Some(2), none),
            ("n 0", Some("n"), [None; 3], Some(0), none),
            (
                "presence -2",
                None,
                [None; 3],
                None,
                pen(Some(-2.0), None, None),
            ),
            (
                "presence 2",
                None,
                [None; 3],
                None,
                pen(Some(2.0), None, None),
            ),
            (
                "presence 2.1",
                Some("presence_penalty"),
                [None; 3],
                None,
                pen(Some(2.1), None, None),
            ),
            (
                "presence -2.1",
                Some("presence_penalty"),
                [None; 3],
                None,
                pen(Some(-2.1), None, None),
            ),
            (
                "presence NaN",
                Some("presence_penalty"),
                [None; 3],
                None,
                pen(Some(f32::NAN), None, None),
            ),
            (
                "frequency -2",
                None,
                [None; 3],
                None,
                pen(None, Some(-2.0), None),
            ),
            (
                "frequency 2",
                None,
                [None; 3],
                None,
                pen(None, Some(2.0), None),
            ),
            (
                "frequency 2.1",
                Some("frequency_penalty"),
                [None; 3],
                None,
                pen(None, Some(2.1), None),
            ),
            (
                "frequency -2.1",
                Some("frequency_penalty"),
                [None; 3],
                None,
                pen(None, Some(-2.1), None),
            ),
            (
                "frequency NaN",
                Some("frequency_penalty"),
                [None; 3],
                None,
                pen(None, Some(f32::NAN), None),
            ),
            (
                "repetition 1",
                None,
                [None; 3],
                None,
                pen(None, None, Some(1.0)),
            ),
            (
                "repetition 5",
                None,
                [None; 3],
                None,
                pen(None, None, Some(5.0)),
            ),
            (
                "repetition tiny",
                None,
                [None; 3],
                None,
                pen(None, None, Some(f32::MIN_POSITIVE)),
            ),
            (
                "repetition 0",
                Some("repetition_penalty"),
                [None; 3],
                None,
                pen(None, None, Some(0.0)),
            ),
            (
                "repetition -1",
                Some("repetition_penalty"),
                [None; 3],
                None,
                pen(None, None, Some(-1.0)),
            ),
            (
                "repetition inf",
                Some("repetition_penalty"),
                [None; 3],
                None,
                pen(None, None, Some(f32::INFINITY)),
            ),
            (
                "repetition NaN",
                Some("repetition_penalty"),
                [None; 3],
                None,
                pen(None, None, Some(f32::NAN)),
            ),
        ];
        for (name, want, [t, p, m], n, penalties) in cases {
            let got = validate_sampling(t, p, m, n, penalties);
            match (want, got) {
                (None, Ok(())) => {}
                (Some(param), Err(e)) => {
                    assert_eq!(e.param.as_deref(), Some(param), "{name}");
                    assert_eq!(
                        e.kind,
                        super::super::error::ErrorKind::InvalidRequest,
                        "{name}"
                    );
                }
                (want, got) => panic!("{name}: want {want:?}, got {got:?}"),
            }
        }
    }

    #[tokio::test]
    async fn chat_request_carries_penalties() {
        let req: openai::ChatCompletionRequest = serde_json::from_value(json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
            "presence_penalty": 0.5,
            "frequency_penalty": -1.5,
            "repetition_penalty": 1.25,
        }))
        .unwrap();
        let s = chat_request(&state(), &req).unwrap().sampling;
        assert_eq!(
            (
                s.presence_penalty,
                s.frequency_penalty,
                s.repetition_penalty
            ),
            (0.5, -1.5, 1.25)
        );
    }

    #[tokio::test]
    async fn completion_request_carries_penalties() {
        let req: openai::CompletionRequest = serde_json::from_value(json!({
            "model": "mock",
            "prompt": "hi",
            "presence_penalty": 2.0,
            "frequency_penalty": 0.25,
            "repetition_penalty": 0.5,
        }))
        .unwrap();
        let s = completion_request(&state(), &req).unwrap().sampling;
        assert_eq!(
            (
                s.presence_penalty,
                s.frequency_penalty,
                s.repetition_penalty
            ),
            (2.0, 0.25, 0.5)
        );
    }

    #[tokio::test]
    async fn penalties_default_when_absent() {
        let req: openai::ChatCompletionRequest = serde_json::from_value(json!({
            "model": "mock",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let s = chat_request(&state(), &req).unwrap().sampling;
        assert_eq!(s, SamplingParams::default());
        assert_eq!(
            (
                s.presence_penalty,
                s.frequency_penalty,
                s.repetition_penalty
            ),
            (0.0, 0.0, 1.0)
        );
    }

    fn chat_with(extra: serde_json::Value) -> Result<GenerateRequest, ApiError> {
        let mut v = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
        v.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let req: openai::ChatCompletionRequest = serde_json::from_value(v).unwrap();
        chat_request(&state(), &req)
    }

    #[tokio::test]
    async fn enable_thinking_beats_reasoning_effort() {
        let r = chat_with(json!({
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false, "x": 1},
        }))
        .unwrap();
        assert_eq!(r.thinking, Some(false));
        assert_eq!(r.template_kwargs["x"], 1);
        let r = chat_with(json!({
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": true},
        }))
        .unwrap();
        assert_eq!(r.thinking, Some(true));
    }

    #[tokio::test]
    async fn thinking_falls_back_to_effort_then_none() {
        let r = chat_with(json!({
            "reasoning_effort": "high",
            "chat_template_kwargs": {"other": "a"},
        }))
        .unwrap();
        assert_eq!(r.thinking, Some(true));
        assert!(chat_with(json!({})).unwrap().thinking.is_none());
        assert!(chat_with(json!({})).unwrap().template_kwargs.is_empty());
    }

    #[tokio::test]
    async fn non_bool_enable_thinking_is_rejected() {
        let e = chat_with(json!({"chat_template_kwargs": {"enable_thinking": "no"}}))
            .err()
            .unwrap();
        assert_eq!(
            e.param.as_deref(),
            Some("chat_template_kwargs.enable_thinking")
        );
    }

    fn messages_with(v: serde_json::Value) -> Result<GenerateRequest, ApiError> {
        let req: anthropic::MessagesRequest = serde_json::from_value(v).unwrap();
        messages_request(&state(), &req)
    }

    #[tokio::test]
    async fn inline_system_turn_keeps_its_position() {
        let r = messages_with(json!({
            "model": "mock",
            "max_tokens": 100,
            "system": "top level",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "system", "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ]},
                {"role": "user", "content": "b"},
            ],
        }))
        .unwrap();
        let Prompt::Chat {
            system, messages, ..
        } = &r.prompt
        else {
            panic!("expected a chat prompt");
        };
        assert_eq!(system.as_deref(), Some("top level"));
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, Some(Role::System));
        assert_eq!(messages[1].content, "one\ntwo");
    }

    #[tokio::test]
    async fn inline_system_turn_rejects_non_text_blocks() {
        let e = messages_with(json!({
            "model": "mock",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "system", "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "image", "source": {}},
                ]},
            ],
        }))
        .err()
        .unwrap();
        assert_eq!(e.param.as_deref(), Some("messages[1].content"));
        assert_eq!(e.message, "system messages may only contain text");
    }

    #[tokio::test]
    async fn unknown_anthropic_role_still_rejected() {
        let e = messages_with(json!({
            "model": "mock",
            "max_tokens": 100,
            "messages": [{"role": "tool", "content": "x"}],
        }))
        .err()
        .unwrap();
        assert_eq!(e.param.as_deref(), Some("messages[0].role"));
    }

    #[tokio::test]
    async fn vllm_extras_map_to_the_engine_request() {
        let r = chat_with(json!({"ignore_eos": true, "min_tokens": 8, "max_tokens": 40})).unwrap();
        assert!(r.ignore_eos);
        assert_eq!(r.min_tokens, 8);
        let r = chat_with(json!({})).unwrap();
        assert!(!r.ignore_eos);
        assert_eq!(r.min_tokens, 0);
    }

    #[tokio::test]
    async fn min_tokens_above_max_tokens_is_rejected() {
        let e = chat_with(json!({"min_tokens": 41, "max_tokens": 40}))
            .err()
            .unwrap();
        assert_eq!(e.param.as_deref(), Some("min_tokens"));
        // The effective `max_tokens` is `max_completion_tokens` when both are set.
        let e = chat_with(json!({
            "min_tokens": 11,
            "max_tokens": 100,
            "max_completion_tokens": 10,
        }))
        .err()
        .unwrap();
        assert_eq!(e.param.as_deref(), Some("min_tokens"));
        // Equal is fine.
        chat_with(json!({"min_tokens": 40, "max_tokens": 40})).unwrap();
    }

    #[tokio::test]
    async fn completion_vllm_extras() {
        let req: openai::CompletionRequest = serde_json::from_value(json!({
            "model": "mock", "prompt": "hi", "ignore_eos": true, "min_tokens": 16,
        }))
        .unwrap();
        let r = completion_request(&state(), &req).unwrap();
        assert!(r.ignore_eos);
        // The legacy-completions default of 16 is the effective maximum.
        assert_eq!(r.min_tokens, 16);
        let req: openai::CompletionRequest = serde_json::from_value(json!({
            "model": "mock", "prompt": "hi", "min_tokens": 17,
        }))
        .unwrap();
        let e = completion_request(&state(), &req).err().unwrap();
        assert_eq!(e.param.as_deref(), Some("min_tokens"));
    }
}
