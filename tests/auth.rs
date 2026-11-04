//! API-key authentication: `--api-key` gates every route except `GET /health`, accepting
//! `Authorization: Bearer <key>` or `x-api-key: <key>`, with 401s in the caller's shape.
mod common;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderName, Request, StatusCode};
use common::*;
use legion::server::{AppState, ServerConfig};
use serde_json::{Value, json};

fn app_with_keys(keys: &[&str]) -> Router {
    app_with(|state| {
        let config = ServerConfig {
            api_keys: keys.iter().map(|k| k.to_string()).collect(),
            ..ServerConfig::default()
        };
        AppState::with_config(state.engine, config)
    })
}

async fn json_to(app: Router, req: Request<Body>) -> (StatusCode, Value) {
    let (status, body) = send_to(app, req).await;
    (
        status,
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("bad JSON {body:?}: {e}")),
    )
}

fn bearer(req: Request<Body>, key: &str) -> Request<Body> {
    let mut req = req;
    req.headers_mut().insert(
        axum::http::header::AUTHORIZATION,
        format!("Bearer {key}").parse().unwrap(),
    );
    req
}

fn x_api_key(req: Request<Body>, key: &str) -> Request<Body> {
    let mut req = req;
    req.headers_mut()
        .insert(HeaderName::from_static("x-api-key"), key.parse().unwrap());
    req
}

/// The 401 body OpenAI clients expect: `{"error": {"type": "authentication_error", ...}}`.
fn assert_openai_unauthorized(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{v}");
    assert_eq!(v["error"]["type"], "authentication_error", "{v}");
    assert_eq!(v["error"]["code"], "invalid_api_key", "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
}

/// The 401 body Anthropic clients expect: `{"type": "error", "error": {...}}`.
fn assert_anthropic_unauthorized(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{v}");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "authentication_error", "{v}");
    assert!(v["error"]["message"].is_string(), "{v}");
}

#[tokio::test]
async fn no_keys_configured_changes_nothing() {
    // Auth off: every route answers without a key, exactly as before.
    let (status, v) = json_of(post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let (status, v) = json_of(post("/v1/messages", messages(json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let (status, v) = get("/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let (status, v) = get("/health").await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn health_never_needs_a_key() {
    let (status, v) = send_to(
        app_with_keys(&["sk-secret"]),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    // A wrong key on /health must not break the probe either.
    let req = bearer(Request::get("/health").body(Body::empty()).unwrap(), "nope");
    let (status, _) = send_to(app_with_keys(&["sk-secret"]), req).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn missing_key_is_401_in_openai_shape() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        post("/v1/chat/completions", chat(json!({}))),
    )
    .await;
    assert_openai_unauthorized(status, &v);

    // Other OpenAI-fronted routes are gated too.
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_openai_unauthorized(status, &v);
}

#[tokio::test]
async fn models_without_v1_needs_the_key_too() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        Request::get("/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_openai_unauthorized(status, &v);
    let req = bearer(
        Request::get("/models").body(Body::empty()).unwrap(),
        "sk-secret",
    );
    let (status, v) = json_to(app_with_keys(&["sk-secret"]), req).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["data"][0]["id"], "mock");
}

#[tokio::test]
async fn missing_key_is_401_in_anthropic_shape() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        post("/v1/messages", messages(json!({}))),
    )
    .await;
    assert_anthropic_unauthorized(status, &v);
}

#[tokio::test]
async fn wrong_key_is_401_on_both_front_ends() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        bearer(post("/v1/chat/completions", chat(json!({}))), "sk-wrong"),
    )
    .await;
    assert_openai_unauthorized(status, &v);

    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        x_api_key(
            post("/v1/messages", messages(json!({}))),
            "sk-wrong-but-longer",
        ),
    )
    .await;
    assert_anthropic_unauthorized(status, &v);
}

#[tokio::test]
async fn bearer_key_is_accepted() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        bearer(post("/v1/chat/completions", chat(json!({}))), "sk-secret"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], "You said: hi there");
}

#[tokio::test]
async fn x_api_key_is_accepted() {
    let (status, v) = json_to(
        app_with_keys(&["sk-secret"]),
        x_api_key(post("/v1/messages", messages(json!({}))), "sk-secret"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["type"], "message");
}

#[tokio::test]
async fn several_keys_any_matches() {
    let app = || app_with_keys(&["sk-first", "sk-second"]);
    let (status, v) = json_to(
        app(),
        bearer(post("/v1/chat/completions", chat(json!({}))), "sk-first"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");

    let (status, v) = json_to(
        app(),
        x_api_key(post("/v1/chat/completions", chat(json!({}))), "sk-second"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
}

#[tokio::test]
async fn non_bearer_authorization_needs_x_api_key() {
    // A Basic `Authorization` is not a key, but a valid `x-api-key` alongside it is.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Basic dXNlcjpwdw==")
        .header("x-api-key", "sk-secret")
        .body(Body::from(chat(json!({})).to_string()))
        .unwrap();
    let (status, v) = json_to(app_with_keys(&["sk-secret"]), req).await;
    assert_eq!(status, StatusCode::OK, "{v}");

    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Basic dXNlcjpwdw==")
        .body(Body::from(chat(json!({})).to_string()))
        .unwrap();
    let (status, v) = json_to(app_with_keys(&["sk-secret"]), req).await;
    assert_openai_unauthorized(status, &v);
}
