use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: Option<SystemPrompt>,
    pub messages: Vec<InputMessage>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<serde_json::Value>,
    pub thinking: Option<ThinkingConfig>,
    /// Not in the Anthropic API (vLLM's extension): lower is served first under
    /// `--scheduling-policy priority`; default 0.
    pub priority: Option<i32>,
}

impl MessagesRequest {
    /// `metadata.user_id`, the end user for `--fair-share user`.
    pub fn user_id(&self) -> Option<&str> {
        self.metadata.as_ref()?.get("user_id")?.as_str()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<InputBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputMessage {
    /// `user | assistant | system`
    pub role: String,
    pub content: InputContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    Text(String),
    Blocks(Vec<InputBlock>),
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputBlock {
    Text {
        text: String,
    },
    Image {
        source: serde_json::Value,
    },
    Document {
        source: serde_json::Value,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<ToolResultContent>,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<InputBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThinkingConfig {
    /// `enabled | disabled | adaptive`
    #[serde(rename = "type")]
    pub kind: String,
    pub budget_tokens: Option<u32>,
}

/// The assistant reply (non-streaming), also the `message` of `message_start`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    pub content: Vec<OutputBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}
