//! Wire types for `POST /v1/responses`, the stateless subset.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub text: Option<serde_json::Value>,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    pub reasoning: Option<ReasoningConfig>,
    pub previous_response_id: Option<String>,
    pub store: Option<bool>,
    /// Queue priority, lower first (`--scheduling-policy priority`, #625).
    pub priority: Option<i32>,
    /// The end user, the fair-share key under `--fair-share user` (#625).
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningConfig {
    /// `none | minimal | low | medium | high`
    pub effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// One element of an `input` list. `message` is the default when `type` is absent.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "serde_json::Value")]
pub enum InputItem {
    Message(InputMessage),
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: serde_json::Value,
    },
    Reasoning {
        content: Vec<ReasoningPart>,
        summary: Vec<ReasoningPart>,
    },
    Unknown,
}

impl TryFrom<serde_json::Value> for InputItem {
    type Error = serde_json::Error;

    fn try_from(mut v: serde_json::Value) -> Result<Self, Self::Error> {
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("message");
        if !matches!(
            kind,
            "message" | "function_call" | "function_call_output" | "reasoning"
        ) {
            return Ok(Self::Unknown);
        }
        let kind = kind.to_string();
        if let Some(o) = v.as_object_mut() {
            o.insert("type".into(), kind.into());
        }
        Ok(match serde_json::from_value::<TypedItem>(v)? {
            TypedItem::Message(m) => Self::Message(m),
            TypedItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => Self::FunctionCall {
                call_id,
                name,
                arguments,
            },
            TypedItem::FunctionCallOutput { call_id, output } => {
                Self::FunctionCallOutput { call_id, output }
            }
            TypedItem::Reasoning { content, summary } => Self::Reasoning { content, summary },
        })
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TypedItem {
    Message(InputMessage),
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: serde_json::Value,
    },
    Reasoning {
        #[serde(default)]
        content: Vec<ReasoningPart>,
        #[serde(default)]
        summary: Vec<ReasoningPart>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputMessage {
    /// `system | developer | user | assistant`
    pub role: String,
    pub content: InputContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Parts(Vec<InputPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
    },
    /// A `data:<mime>;base64,<data>` URL (remote URLs are rejected by the server).
    InputImage {
        image_url: String,
    },
    #[serde(other)]
    Unsupported,
}

/// `reasoning_text` in `content`, `summary_text` in `summary`.
#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningPart {
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    /// `in_progress | completed | incomplete | failed`
    pub status: &'static str,
    pub error: Option<serde_json::Value>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub parallel_tool_calls: bool,
    pub previous_response_id: Option<String>,
    pub reasoning: ReasoningOut,
    pub store: bool,
    pub temperature: Option<f32>,
    pub text: serde_json::Value,
    pub tool_choice: &'static str,
    pub tools: Vec<serde_json::Value>,
    pub top_p: Option<f32>,
    pub truncation: &'static str,
    pub usage: Option<ResponsesUsage>,
    pub user: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct IncompleteDetails {
    pub reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningOut {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Reasoning {
        id: String,
        summary: Vec<serde_json::Value>,
        content: Vec<OutputText>,
        status: &'static str,
    },
    Message {
        id: String,
        role: &'static str,
        status: &'static str,
        content: Vec<OutputContent>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
}

/// A `reasoning_text` content part.
#[derive(Debug, Clone, Serialize)]
pub struct OutputText {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

/// An `output_text` content part of a message.
#[derive(Debug, Clone, Serialize)]
pub struct OutputContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

impl OutputContent {
    pub fn text(text: String) -> Self {
        Self {
            kind: "output_text",
            text,
            annotations: vec![],
        }
    }
}

impl OutputText {
    pub fn reasoning(text: String) -> Self {
        Self {
            kind: "reasoning_text",
            text,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl From<crate::engine::Usage> for ResponsesUsage {
    fn from(u: crate::engine::Usage) -> Self {
        Self {
            input_tokens: u.prompt_tokens,
            input_tokens_details: InputTokensDetails {
                cached_tokens: u.cached_tokens,
            },
            output_tokens: u.completion_tokens,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: u.prompt_tokens + u.completion_tokens,
        }
    }
}
