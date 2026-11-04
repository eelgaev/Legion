//! API errors, rendered in the shape of whichever front end the request came through.

use axum::Json;
use axum::extract::rejection::{BytesRejection, FailedToBufferBody, JsonRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::api::{anthropic, openai};
use crate::engine::EngineError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidRequest,
    Authentication,
    NotFound,
    Overloaded,
    /// Request body over `--max-request-mb`.
    TooLarge,
    Internal,
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub kind: ErrorKind,
    pub message: String,
    /// The request field at fault, when known.
    pub param: Option<String>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidRequest,
            message: message.into(),
            param: None,
        }
    }

    pub fn invalid_param(param: &str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param.into()),
            ..Self::invalid(message)
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::NotFound,
            message: message.into(),
            param: None,
        }
    }

    pub fn too_large(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::TooLarge,
            message: message.into(),
            param: None,
        }
    }

    pub fn status(&self) -> StatusCode {
        match self.kind {
            ErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            ErrorKind::Authentication => StatusCode::UNAUTHORIZED,
            ErrorKind::NotFound => StatusCode::NOT_FOUND,
            ErrorKind::Overloaded => StatusCode::TOO_MANY_REQUESTS,
            ErrorKind::TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn to_openai(&self) -> openai::ErrorResponse {
        let (kind, code) = match self.kind {
            ErrorKind::InvalidRequest => ("invalid_request_error", None),
            ErrorKind::Authentication => ("authentication_error", Some("invalid_api_key")),
            ErrorKind::NotFound => ("invalid_request_error", Some("model_not_found")),
            ErrorKind::Overloaded => ("rate_limit_error", Some("rate_limit_exceeded")),
            ErrorKind::TooLarge => ("invalid_request_error", None),
            ErrorKind::Internal => ("server_error", None),
        };
        openai::ErrorResponse {
            error: openai::ErrorBody {
                message: self.message.clone(),
                kind: kind.into(),
                param: self.param.clone(),
                code: code.map(Into::into),
            },
        }
    }

    pub fn to_anthropic(&self) -> anthropic::ErrorResponse {
        let kind = match self.kind {
            ErrorKind::InvalidRequest => "invalid_request_error",
            ErrorKind::Authentication => "authentication_error",
            ErrorKind::NotFound => "not_found_error",
            ErrorKind::Overloaded => "overloaded_error",
            ErrorKind::TooLarge => "request_too_large",
            ErrorKind::Internal => "api_error",
        };
        anthropic::ErrorResponse {
            kind: "error".into(),
            error: anthropic::ErrorBody {
                kind: kind.into(),
                message: self.message.clone(),
                param: self.param.clone(),
            },
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        let kind = match &e {
            EngineError::ModelNotFound(_) => ErrorKind::NotFound,
            EngineError::InvalidRequest(_) | EngineError::ContextOverflow { .. } => {
                ErrorKind::InvalidRequest
            }
            EngineError::Overloaded => ErrorKind::Overloaded,
            EngineError::Internal(_) => ErrorKind::Internal,
        };
        Self {
            kind,
            message: e.to_string(),
            param: None,
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(e: JsonRejection) -> Self {
        // The body limit from `middleware::body_limit` surfaces as a length rejection while
        // buffering; everything else about the body is the caller's mistake (400).
        if matches!(
            e,
            JsonRejection::BytesRejection(BytesRejection::FailedToBufferBody(
                FailedToBufferBody::LengthLimitError(_)
            ))
        ) {
            return Self::too_large("request body too large");
        }
        Self::invalid(e.body_text())
    }
}

/// An [`ApiError`] rendered as `{"error": {...}}`.
#[derive(Debug)]
pub struct OpenAiError(pub ApiError);

impl<E: Into<ApiError>> From<E> for OpenAiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        (self.0.status(), Json(self.0.to_openai())).into_response()
    }
}

/// An [`ApiError`] rendered as `{"type": "error", "error": {...}}`.
#[derive(Debug)]
pub struct AnthropicError(pub ApiError);

impl<E: Into<ApiError>> From<E> for AnthropicError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let status = match self.0.kind {
            // Anthropic signals overload with 529.
            ErrorKind::Overloaded => StatusCode::from_u16(529).unwrap(),
            _ => self.0.status(),
        };
        (status, Json(self.0.to_anthropic())).into_response()
    }
}
