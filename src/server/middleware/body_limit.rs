//! Request body size limit. axum buffers request bodies with a built-in 2 MB cap, which is
//! too small for long prompts, so every body-reading extractor runs under the
//! [`ServerConfig::max_request_mb`] limit instead. Bodies over the limit fail with a length
//! rejection, which `error.rs` renders as 413 in the caller's format.

use axum::Router;
use axum::extract::DefaultBodyLimit;

use super::super::AppState;

/// Caps request bodies at `--max-request-mb` megabytes.
pub fn apply(router: Router, state: &AppState) -> Router {
    let max_bytes = state.config.max_request_mb.saturating_mul(1024 * 1024);
    router.layer(DefaultBodyLimit::max(max_bytes))
}
