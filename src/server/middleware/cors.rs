//! CORS for browser frontends on another origin. `--cors <ORIGIN>` (repeatable) lists the
//! allowed origins; `*` allows any. With no `--cors` no layer is added at all: the server
//! sends no CORS headers and `OPTIONS` keeps being answered by the router, as before.

use axum::Router;
use axum::http::{HeaderValue, Method, header::HeaderName};
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer, ExposeHeaders};

use crate::server::AppState;

/// The request headers browser clients (OpenAI/Anthropic SDKs, open-webui, Claude Code) send.
const ALLOWED_HEADERS: [&str; 6] = [
    "authorization",
    "content-type",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "x-request-id",
];

/// POST for the API routes, OPTIONS for preflights, GET for `/v1/models` and `/health`.
const ALLOWED_METHODS: [Method; 3] = [Method::GET, Method::POST, Method::OPTIONS];

/// Lets browser JS read the request id, so a report can be matched to the server log.
const EXPOSED_HEADERS: [HeaderName; 1] = [HeaderName::from_static("x-request-id")];

/// Wraps `router` in a [`CorsLayer`] built from [`ServerConfig::cors_origins`]: methods GET,
/// POST and OPTIONS, the headers browser clients send, and `x-request-id` exposed so callers
/// can tie a response to the server log. With no usable origin configured the router comes
/// back unchanged.
pub fn apply(router: Router, state: &AppState) -> Router {
    let Some(allow_origin) = allow_origin(&state.config.cors_origins) else {
        return router;
    };
    router.layer(
        CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods(AllowMethods::list(ALLOWED_METHODS))
            .allow_headers(AllowHeaders::list(
                ALLOWED_HEADERS.map(HeaderName::from_static),
            ))
            .expose_headers(ExposeHeaders::list(EXPOSED_HEADERS)),
    )
}

/// `*` anywhere allows any origin; otherwise exactly the listed ones. `None` means no CORS.
fn allow_origin(origins: &[String]) -> Option<AllowOrigin> {
    if origins.iter().any(|origin| origin == "*") {
        return Some(AllowOrigin::any());
    }
    let allowed: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| match origin.parse::<HeaderValue>() {
            Ok(value) => Some(value),
            Err(_) => {
                tracing::warn!("ignoring invalid --cors origin {origin:?}");
                None
            }
        })
        .collect();
    if allowed.is_empty() {
        return None;
    }
    Some(AllowOrigin::list(allowed))
}
