//! `POST /v1/messages/count_tokens`

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde_json::{Value, json};

use crate::api::anthropic::MessagesRequest;
use crate::server::AppState;
use crate::server::convert;
use crate::server::error::{AnthropicError, ApiError};

/// The Messages body, where `max_tokens` is optional (it is irrelevant to counting).
pub async fn count_tokens(
    State(state): State<AppState>,
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, AnthropicError> {
    let Json(mut body) = body?;
    if let Some(obj) = body.as_object_mut() {
        obj.entry("max_tokens").or_insert(json!(1));
    }
    let req: MessagesRequest =
        serde_json::from_value(body).map_err(|e| ApiError::invalid(e.to_string()))?;
    let gen_req = convert::messages_request(&state, &req)?;
    let input_tokens = state.engine.count_tokens(&gen_req)?;
    Ok(Json(json!({ "input_tokens": input_tokens })))
}
