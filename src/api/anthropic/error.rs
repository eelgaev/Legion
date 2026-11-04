use serde::{Deserialize, Serialize};

/// `{"type": "error", "error": {"type": ..., "message": ...}}`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub kind: String,
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
    /// The request field at fault, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}
