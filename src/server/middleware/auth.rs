//! Optional API-key authentication. With no keys in [`ServerConfig::api_keys`] every request
//! passes through (today's behaviour); otherwise each request must present one of the keys in
//! `Authorization: Bearer <key>` or `x-api-key: <key>` (Anthropic clients send the latter).
//! `GET /health` and the web UI static files (`/`, `/assets/*`) never need a key.

use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::engine::queue::ShareKey;
use crate::server::error::{AnthropicError, ApiError, ErrorKind, OpenAiError};
use crate::server::fair_share::{CallerKey, FairShare};
use crate::server::{AppState, routes};

pub async fn authenticate(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    // `--fair-share api-key`: the caller's key, hashed, for the engine's queue. Without
    // `--api-key` whatever key the client presents; with it, only an accepted one (below).
    let fair = state.config.fair_share == FairShare::ApiKey;
    if state.config.api_keys.is_empty() {
        if fair && let Some(key) = extract_key(req.headers()).map(ShareKey::of) {
            req.extensions_mut().insert(CallerKey(key));
        }
        return next.run(req).await;
    }
    if req.method() == Method::GET && req.uri().path() == "/health" {
        return next.run(req).await;
    }
    let path = req.uri().path();
    // The web UI's static files carry no secrets; the page sends the key with its API calls.
    if req.method() == Method::GET && routes::webui::is_static_path(path) {
        return next.run(req).await;
    }
    match extract_key(req.headers()) {
        Some(key) if key_matches(&state, key) => {
            if fair {
                let key = ShareKey::of(key);
                req.extensions_mut().insert(CallerKey(key));
            }
            next.run(req).await
        }
        Some(_) => reject(path, "invalid API key"),
        None => reject(path, "missing API key"),
    }
}

fn key_matches(state: &AppState, provided: &str) -> bool {
    state
        .config
        .api_keys
        .iter()
        .any(|key| constant_time_eq(key.as_bytes(), provided.as_bytes()))
}

/// The caller's key: `Authorization: Bearer <key>` first, then `x-api-key`. A malformed or
/// non-Bearer `Authorization` header falls through to `x-api-key`.
fn extract_key(headers: &HeaderMap) -> Option<&str> {
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|value| {
            // RFC 9110: the scheme is case-insensitive.
            let (scheme, token) = value.split_once(' ')?;
            scheme
                .eq_ignore_ascii_case("bearer")
                .then_some(token)
                .filter(|t| !t.is_empty())
        });
    bearer.or_else(|| {
        headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
    })
}

/// 401 in the shape the route's front end expects: Anthropic for `/v1/messages*`, OpenAI
/// for everything else.
fn reject(path: &str, message: &str) -> Response {
    let error = ApiError {
        kind: ErrorKind::Authentication,
        message: message.into(),
        param: None,
    };
    if path.starts_with("/v1/messages") {
        AnthropicError(error).into_response()
    } else {
        OpenAiError(error).into_response()
    }
}

/// Xor-fold over the bytes: every byte pair and the length difference are combined into one
/// accumulator, so the comparison time depends only on the lengths, not on where keys differ.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= (x ^ y) as usize;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderName;

    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (name, value) in pairs {
            h.insert(HeaderName::from_static(name), value.parse().unwrap());
        }
        h
    }

    #[test]
    fn constant_time_eq_matches_and_rejects() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"sk-abc123", b"sk-abc123"));
        assert!(!constant_time_eq(b"sk-abc123", b"sk-abc124"));
        assert!(!constant_time_eq(b"sk-abc", b"sk-abc123"));
        assert!(!constant_time_eq(b"sk-abc123", b"sk-abc"));
        // Same length, differing only in the last byte.
        assert!(!constant_time_eq(b"sk-abc123", b"sk-abc12x"));
    }

    #[test]
    fn bearer_header_is_read() {
        let h = headers(&[("authorization", "Bearer sk-abc")]);
        assert_eq!(extract_key(&h), Some("sk-abc"));
        // The scheme is case-insensitive.
        let h = headers(&[("authorization", "bearer sk-abc")]);
        assert_eq!(extract_key(&h), Some("sk-abc"));
    }

    #[test]
    fn x_api_key_header_is_read() {
        let h = headers(&[("x-api-key", "sk-abc")]);
        assert_eq!(extract_key(&h), Some("sk-abc"));
    }

    #[test]
    fn bearer_wins_over_x_api_key() {
        let h = headers(&[("authorization", "Bearer sk-a"), ("x-api-key", "sk-b")]);
        assert_eq!(extract_key(&h), Some("sk-a"));
    }

    #[test]
    fn non_bearer_or_malformed_headers_fall_through() {
        let h = headers(&[("authorization", "Basic dXNlcjpwdw==")]);
        assert_eq!(extract_key(&h), None);
        let h = headers(&[("authorization", "Bearer")]);
        assert_eq!(extract_key(&h), None);
        let h = headers(&[("authorization", "Bearer ")]);
        assert_eq!(extract_key(&h), None);
        // A broken `Authorization` does not hide a valid `x-api-key`.
        let h = headers(&[
            ("authorization", "Basic dXNlcjpwdw=="),
            ("x-api-key", "sk-b"),
        ]);
        assert_eq!(extract_key(&h), Some("sk-b"));
        let h = headers(&[]);
        assert_eq!(extract_key(&h), None);
    }
}
