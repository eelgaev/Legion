use serde::{Deserialize, Serialize};

use super::{CompletionLogprobs, StreamOptions, Timings, Usage};
use crate::api::OneOrMany;

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub response_format: Option<serde_json::Value>,
    /// A string, or a list holding exactly one string. Token-id prompts are not supported yet.
    pub prompt: OneOrMany,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
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
    pub n: Option<u32>,
    pub stop: Option<OneOrMany>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub echo: bool,
    /// Number of alternatives per token (0..=20); the chosen token is always included.
    pub logprobs: Option<u32>,
    /// Not in the OpenAI spec (vLLM extension): lower is served first under
    /// `--scheduling-policy priority`; default 0.
    pub priority: Option<i32>,
    /// The end user, for `--fair-share user`.
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Server-measured prefill/decode speed; present when the engine reports it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
    pub logprobs: Option<CompletionLogprobs>,
}
