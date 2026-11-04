//! The built-in web UI: the `web/` directory embedded in the binary. `GET /` is
//! `web/index.html`; `GET /assets/<path>` is `web/<path>`. Release builds embed the files;
//! debug builds read them from disk, so edits show up on reload without a rebuild.

use axum::Router;
use axum::extract::Path;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use rust_embed::RustEmbed;

use crate::server::AppState;

#[derive(RustEmbed)]
#[folder = "web/"]
struct Assets;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(index))
        .route("/assets/{*path}", get(asset))
}

/// True for the paths this module serves; the auth middleware lets them through.
pub(crate) fn is_static_path(path: &str) -> bool {
    path == "/" || path.starts_with("/assets/")
}

async fn index() -> Response {
    // no-cache: the browser revalidates, so a new binary is picked up on the next load.
    file("index.html", "no-cache")
}

async fn asset(Path(path): Path<String>) -> Response {
    file(&path, "no-cache")
}

fn file(path: &str, cache_control: &'static str) -> Response {
    match Assets::get(path) {
        Some(f) => (
            [
                (header::CONTENT_TYPE, content_type(path)),
                (header::CACHE_CONTROL, cache_control),
            ],
            f.data.into_owned(),
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "txt" | "md" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}
