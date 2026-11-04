use super::AppState;
use axum::{Json, extract::State};
use serde_json::{Value, json};

pub async fn health(State(state): State<AppState>) -> Json<Value> {
    let models = state
        .engine
        .models()
        .into_iter()
        .map(|m| m.id)
        .collect::<Vec<_>>();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "git": env!("LEGION_GIT_HASH"),
        "models": models,
    }))
}
