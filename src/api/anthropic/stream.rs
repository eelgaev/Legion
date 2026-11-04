use serde::{Deserialize, Serialize};

use super::{ErrorBody, MessageResponse, OutputBlock};

/// One server-sent event. The SSE `event:` name equals the `type` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessageResponse,
    },
    ContentBlockStart {
        index: u32,
        content_block: OutputBlock,
    },
    Ping,
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        usage: DeltaUsage,
    },
    MessageStop,
    Error {
        error: ErrorBody,
    },
}

impl StreamEvent {
    pub fn name(&self) -> &'static str {
        match self {
            StreamEvent::MessageStart { .. } => "message_start",
            StreamEvent::ContentBlockStart { .. } => "content_block_start",
            StreamEvent::Ping => "ping",
            StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            StreamEvent::ContentBlockStop { .. } => "content_block_stop",
            StreamEvent::MessageDelta { .. } => "message_delta",
            StreamEvent::MessageStop => "message_stop",
            StreamEvent::Error { .. } => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeltaUsage {
    pub output_tokens: u32,
}
