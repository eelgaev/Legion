//! `x-request-id`: every response carries one (generated `req_...` or the client's valid id
//! echoed back), including error and SSE responses. Invalid incoming ids (too long,
//! non-ASCII, empty) are replaced.
mod common;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderName, HeaderValue, Request, StatusCode};
use common::*;
use http_body_util::BodyExt;
use legion::server::{AppState, ServerConfig};
use serde_json::json;
use tower::ServiceExt;

const REQUEST_ID: &str = "x-request-id";

/// Sends `req` and returns status, the `x-request-id` response header, and the body.
async fn call(app: Router, req: Request<Body>) -> (StatusCode, String, String) {
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let id = resp
        .headers()
        .get(REQUEST_ID)
        .unwrap_or_else(|| panic!("no {REQUEST_ID} on {status} response"))
        .to_str()
        .expect("ids are printable ASCII")
        .to_owned();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, id, String::from_utf8(bytes.to_vec()).unwrap())
}

fn with_id(mut req: Request<Body>, id: HeaderValue) -> Request<Body> {
    req.headers_mut()
        .insert(HeaderName::from_static(REQUEST_ID), id);
    req
}

fn app_with_keys(keys: &[&str]) -> Router {
    app_with(|state| {
        let config = ServerConfig {
            api_keys: keys.iter().map(|k| k.to_string()).collect(),
            ..ServerConfig::default()
        };
        AppState::with_config(state.engine, config)
    })
}

#[tokio::test]
async fn every_response_gets_a_generated_id() {
    for (status, id, _) in [
        call(app(), Request::get("/health").body(Body::empty()).unwrap()).await,
        call(
            app(),
            Request::get("/v1/models").body(Body::empty()).unwrap(),
        )
        .await,
        call(app(), post("/v1/chat/completions", chat(json!({})))).await,
        call(app(), post("/v1/messages", messages(json!({})))).await,
    ] {
        assert_eq!(status, StatusCode::OK);
        assert!(id.starts_with("req_"), "generated id expected, got {id:?}");
    }
    // Two requests get two different ids.
    let (_, first, _) = call(app(), Request::get("/health").body(Body::empty()).unwrap()).await;
    let (_, second, _) = call(app(), Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_ne!(first, second);
}

#[tokio::test]
async fn valid_incoming_id_is_echoed_unchanged() {
    let id = HeaderValue::from_static("trace-abc-123");
    for (status, echoed, _) in [
        call(
            app(),
            with_id(
                Request::get("/health").body(Body::empty()).unwrap(),
                id.clone(),
            ),
        )
        .await,
        call(
            app(),
            with_id(post("/v1/chat/completions", chat(json!({}))), id.clone()),
        )
        .await,
        call(
            app(),
            with_id(post("/v1/messages", messages(json!({}))), id.clone()),
        )
        .await,
    ] {
        assert_eq!(status, StatusCode::OK);
        assert_eq!(echoed, "trace-abc-123");
    }
    // The length bounds: 1 and 128 chars are still valid and echoed.
    let (_, echoed, _) = call(
        app(),
        with_id(
            Request::get("/health").body(Body::empty()).unwrap(),
            HeaderValue::from_static("x"),
        ),
    )
    .await;
    assert_eq!(echoed, "x");
    let long = "y".repeat(128);
    let (_, echoed, _) = call(
        app(),
        with_id(
            Request::get("/health").body(Body::empty()).unwrap(),
            HeaderValue::from_str(&long).unwrap(),
        ),
    )
    .await;
    assert_eq!(echoed, long);
}

#[tokio::test]
async fn invalid_incoming_id_is_replaced() {
    // Too long (129 chars), non-ASCII, and empty are all replaced by a generated id.
    for bad in [
        HeaderValue::from_str(&"z".repeat(129)).unwrap(),
        HeaderValue::from_bytes(b"h\xc3\xa9llo").unwrap(), // "héllo"
        HeaderValue::from_bytes(b"caf\xe9").unwrap(),
        HeaderValue::from_static(""),
    ] {
        let (status, echoed, _) = call(
            app(),
            with_id(Request::get("/health").body(Body::empty()).unwrap(), bad),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            echoed.starts_with("req_"),
            "replaced id expected, got {echoed:?}"
        );
    }
}

#[tokio::test]
async fn error_responses_carry_the_id() {
    // 404 from the router.
    let (status, id, _) = call(
        app(),
        with_id(
            Request::get("/nope").body(Body::empty()).unwrap(),
            HeaderValue::from_static("missing-route"),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(id, "missing-route");

    // 400 from a malformed JSON body.
    let (status, id, _) = call(
        app(),
        with_id(
            post("/v1/chat/completions", "{not json"),
            HeaderValue::from_static("bad-json"),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(id, "bad-json");

    // 401 from the auth middleware, which runs inside this one.
    let (status, id, _) = call(
        app_with_keys(&["sk-secret"]),
        with_id(
            post("/v1/chat/completions", chat(json!({}))),
            HeaderValue::from_static("no-key"),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(id, "no-key");
}

#[tokio::test]
async fn sse_response_carries_the_id() {
    // OpenAI streaming.
    let (status, id, body) = call(
        app(),
        with_id(
            post("/v1/chat/completions", chat(json!({"stream": true}))),
            HeaderValue::from_static("stream-me"),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(id, "stream-me");
    assert!(sse_events(&body).len() > 1, "expected a streamed body");

    // Anthropic streaming, without an incoming id.
    let (status, id, body) = call(
        app(),
        post("/v1/messages", messages(json!({"stream": true}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(id.starts_with("req_"), "generated id expected, got {id:?}");
    assert!(sse_events(&body).len() > 1, "expected a streamed body");
}
