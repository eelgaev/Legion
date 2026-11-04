//! Validation shared by the OpenAI response-format entry points.

use serde_json::Value;

use super::{AppState, error::ApiError};
use crate::engine::ResponseFormat;

pub(super) fn parse(
    state: &AppState,
    value: Option<&Value>,
    param: &str,
) -> Result<Option<ResponseFormat>, ApiError> {
    let Some(value) = value else { return Ok(None) };
    let bad = |message| ApiError::invalid_param(param, message);
    let format = match value.get("type").and_then(Value::as_str) {
        Some("text") => return Ok(None),
        Some("json_object") => ResponseFormat::JsonObject,
        Some("json_schema") => {
            let definition = if param == "text.format" {
                value
            } else {
                value.get("json_schema").unwrap_or(&Value::Null)
            };
            let schema = definition
                .get("schema")
                .ok_or_else(|| bad("json_schema requires a schema".to_string()))?;
            if !schema.is_object() && !schema.is_boolean() {
                return Err(bad("schema must be an object or boolean".to_string()));
            }
            // `strict` is deliberately ignored: the grammar always enforces the schema.
            ResponseFormat::JsonSchema(schema.clone())
        }
        _ => {
            return Err(bad(
                "unsupported response format type; expected text, json_object or json_schema"
                    .to_string(),
            ));
        }
    };
    state
        .engine
        .check_response_format(&format)
        .map_err(|e| bad(e.to_string()))?;
    Ok(Some(format))
}

pub(super) fn responses(
    state: &AppState,
    text: Option<&Value>,
) -> Result<Option<ResponseFormat>, ApiError> {
    if text.is_some_and(|v| !v.is_object()) {
        return Err(ApiError::invalid_param(
            "text.format",
            "text must be an object",
        ));
    }
    parse(state, text.and_then(|v| v.get("format")), "text.format")
}
