use axum::Json;
use axum::extract::{Path, State};

use crate::api::openai::{Model, ModelList};
use crate::server::AppState;
use crate::server::error::{ApiError, OpenAiError};

/// The models to advertise: the engine's, or with `--served-model-name`, one entry per alias
/// (sharing the engine model's metadata).
fn served_models(state: &AppState) -> Vec<Model> {
    let models: Vec<Model> = state.engine.models().into_iter().map(Model::from).collect();
    let names = &state.config.served_model_names;
    match models.first() {
        Some(first) if !names.is_empty() => names
            .iter()
            .map(|name| Model {
                id: name.clone(),
                ..first.clone()
            })
            .collect(),
        _ => models,
    }
}

pub async fn list(State(state): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: served_models(&state),
    })
}

pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Model>, OpenAiError> {
    served_models(&state)
        .into_iter()
        .find(|m| m.id == id)
        .map(Json)
        .ok_or_else(|| ApiError::not_found(format!("model `{id}` not found")).into())
}
