use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    /// Non-standard extension: e.g. `["vision"]` for image-capable models, `[]` otherwise.
    /// OpenAI clients ignore unknown fields.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

impl From<crate::engine::ModelInfo> for Model {
    fn from(m: crate::engine::ModelInfo) -> Self {
        Self {
            id: m.id,
            object: "model".into(),
            created: m.created,
            owned_by: m.owned_by,
            capabilities: if m.vision {
                vec!["vision".into()]
            } else {
                Vec::new()
            },
        }
    }
}
