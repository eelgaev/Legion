//! CORS: `--cors <ORIGIN>` (repeatable, `*` allows any) answers browser preflights and tags
//! real responses with CORS headers; without it the server sends no CORS headers at all.
mod common;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use common::*;
use legion::server::{AppState, ServerConfig};
use serde_json::json;
use tower::ServiceExt;

fn app_with_origins(origins: &[&str]) -> Router {
    app_with(|state| {
        let config = ServerConfig {
            cors_origins: origins.iter().map(|o| o.to_string()).collect(),
            ..ServerConfig::default()
        };
        AppState::with_config(state.engine, config)
    })
}

async fn call(app: Router, req: Request<Body>) -> (StatusCode, HeaderMap) {
    let resp = app.oneshot(req).await.unwrap();
    (resp.status(), resp.headers().clone())
}

fn preflight(origin: &str) -> Request<Body> {
    Request::options("/v1/chat/completions")
        .header(header::ORIGIN, origin)
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .body(Body::empty())
        .unwrap()
}

fn post_from(origin: &str) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header(header::ORIGIN, origin)
        .body(Body::from(chat(json!({})).to_string()))
        .unwrap()
}

/// A preflight the browser may accept: the origin is allowed and the methods and headers the
/// SDKs need are listed.
fn assert_preflight_allows(headers: &HeaderMap, allow_origin: &str) {
    assert_eq!(
        headers[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        allow_origin,
        "{headers:?}"
    );
    let methods = headers[header::ACCESS_CONTROL_ALLOW_METHODS]
        .to_str()
        .unwrap()
        .to_owned();
    for method in ["GET", "POST", "OPTIONS"] {
        assert!(
            methods.split(',').any(|m| m.trim() == method),
            "{method} missing from {methods:?}"
        );
    }
    let allowed = headers[header::ACCESS_CONTROL_ALLOW_HEADERS]
        .to_str()
        .unwrap()
        .to_owned();
    for name in [
        "authorization",
        "content-type",
        "x-api-key",
        "anthropic-version",
        "anthropic-beta",
        "x-request-id",
    ] {
        assert!(
            allowed.split(',').any(|h| h.trim() == name),
            "{name} missing from {allowed:?}"
        );
    }
}

#[tokio::test]
async fn preflight_from_allowed_origin_gets_cors_headers() {
    let (status, headers) = call(
        app_with_origins(&["http://localhost:3000"]),
        preflight("http://localhost:3000"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_preflight_allows(&headers, "http://localhost:3000");
    // `request_id` sits outside CORS, so even short-circuited preflights carry an id.
    assert!(headers.contains_key("x-request-id"));
}

#[tokio::test]
async fn preflight_from_other_origin_gets_no_allow_origin() {
    let (status, headers) = call(
        app_with_origins(&["http://localhost:3000"]),
        preflight("http://evil.example"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Without `access-control-allow-origin` the browser fails the preflight: the request
    // never leaves the page.
    assert!(
        !headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN),
        "{headers:?}"
    );
}

#[tokio::test]
async fn wildcard_origin_allows_any_origin() {
    for origin in ["http://anywhere.test", "https://open-webui.example"] {
        let (status, headers) = call(app_with_origins(&["*"]), preflight(origin)).await;
        assert_eq!(status, StatusCode::OK);
        assert_preflight_allows(&headers, "*");
    }
}

#[tokio::test]
async fn every_listed_origin_is_allowed_and_others_are_not() {
    let app = || app_with_origins(&["http://a.test", "http://b.test"]);
    for origin in ["http://a.test", "http://b.test"] {
        let (_, headers) = call(app(), preflight(origin)).await;
        assert_eq!(headers[header::ACCESS_CONTROL_ALLOW_ORIGIN], origin);
    }
    let (_, headers) = call(app(), preflight("http://c.test")).await;
    assert!(!headers.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
}

#[tokio::test]
async fn real_request_from_allowed_origin_gets_allow_origin_and_expose() {
    let (status, headers) = call(
        app_with_origins(&["http://localhost:3000"]),
        post_from("http://localhost:3000"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        "http://localhost:3000"
    );
    assert_eq!(
        headers[header::ACCESS_CONTROL_EXPOSE_HEADERS],
        "x-request-id"
    );
}

#[tokio::test]
async fn preflight_passes_even_with_api_keys() {
    // Browsers never send the key on a preflight, so CORS must sit outside auth.
    let app = app_with(|state| {
        let config = ServerConfig {
            cors_origins: vec!["http://localhost:3000".to_string()],
            api_keys: vec!["sk-secret".to_string()],
            ..ServerConfig::default()
        };
        AppState::with_config(state.engine, config)
    });
    let (status, headers) = call(app, preflight("http://localhost:3000")).await;
    assert_eq!(status, StatusCode::OK, "preflight must not be a 401");
    assert_preflight_allows(&headers, "http://localhost:3000");
}

#[tokio::test]
async fn without_cors_no_cors_headers_at_all() {
    // A preflight is answered by the router's method fallback: 405 + `allow`, and no
    // `access-control-*` or `vary` anywhere.
    let (status, headers) = call(app(), preflight("http://localhost:3000")).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert!(headers.contains_key(header::ALLOW), "{headers:?}");
    assert_no_cors_headers(&headers);

    let (status, headers) = call(app(), post_from("http://localhost:3000")).await;
    assert_eq!(status, StatusCode::OK);
    assert_no_cors_headers(&headers);
}

fn assert_no_cors_headers(headers: &HeaderMap) {
    for name in [
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        header::ACCESS_CONTROL_ALLOW_METHODS,
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        header::ACCESS_CONTROL_EXPOSE_HEADERS,
        header::VARY,
    ] {
        assert!(!headers.contains_key(&name), "{name:?} in {headers:?}");
    }
}
