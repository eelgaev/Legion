use serde::{Deserialize, Serialize};

use super::{ChatLogprobs, StreamOptions, Timings, Usage};
use crate::api::OneOrMany;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub response_format: Option<serde_json::Value>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Not in the OpenAI spec, but widely sent to local servers.
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub seed: Option<u64>,
    /// -2.0..=2.0.
    pub presence_penalty: Option<f32>,
    /// -2.0..=2.0.
    pub frequency_penalty: Option<f32>,
    /// Not in the OpenAI spec (vLLM/llama.cpp extension); > 0.0, 1.0 disables.
    pub repetition_penalty: Option<f32>,
    /// Not in the OpenAI spec (vLLM extension): a stop token does not end the sequence.
    pub ignore_eos: Option<bool>,
    /// Not in the OpenAI spec (vLLM extension); must be <= the effective `max_tokens`.
    pub min_tokens: Option<u32>,
    /// Return log probabilities of the output tokens.
    #[serde(default)]
    pub logprobs: bool,
    /// 0..=20 alternatives per token; needs `logprobs: true`.
    pub top_logprobs: Option<u32>,
    pub n: Option<u32>,
    pub stop: Option<OneOrMany>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<serde_json::Value>,
    /// `"low" | "medium" | "high"`; any value turns reasoning output on.
    pub reasoning_effort: Option<String>,
    /// Extra chat-template variables (llama.cpp / vLLM convention), e.g. `enable_thinking`.
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Not in the OpenAI spec (vLLM extension): lower is served first under
    /// `--scheduling-policy priority`; default 0.
    pub priority: Option<i32>,
    /// The end user, for `--fair-share user`.
    pub user: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    /// `system | developer | user | assistant | tool`
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: serde_json::Value },
    InputAudio { input_audio: serde_json::Value },
    File { file: serde_json::Value },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Calls must follow `parameters` exactly (grammar-constrained).
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Server-measured prefill/decode speed; present when the engine reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: Option<&'static str>,
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Only on the final chunk, and only with `stream_options.include_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Server-measured prefill/decode speed; on the usage chunk, or on the finish
    /// chunk when usage is not included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub function: FunctionCallDelta,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}
