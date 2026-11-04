//! Image inputs on all three front ends: accepted data URLs, validation errors, and the
//! `capabilities` field of `/v1/models`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use common::*;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::server::{AppState, router};
use serde_json::{Value, json};

/// PNG magic bytes; the server carries images without decoding them, so this is a valid
/// payload for these tests.
fn png() -> Vec<u8> {
    vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
}

fn jpeg() -> Vec<u8> {
    vec![0xff, 0xd8, 0xff]
}

fn b64(bytes: &[u8]) -> String {
    BASE64.encode(bytes)
}

fn data_url(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", b64(bytes))
}

/// The mock announces each image of the last user message, in order, before echoing.
fn assert_announced(text: &str, images: &[(&str, &[u8])]) {
    let want: Vec<String> = images
        .iter()
        .map(|(mime, bytes)| format!("[image {mime} {} bytes]", bytes.len()))
        .collect();
    let prefix = want.join(" ");
    assert!(
        text.starts_with(&prefix),
        "reply {text:?} does not start with {prefix:?}"
    );
}

/// A router over a mock engine with `vision` off.
fn app_no_vision() -> Router {
    let engine = MockEngine::new(MockConfig {
        vision: false,
        step: Duration::ZERO,
        ..Default::default()
    });
    router(AppState::new(Arc::new(engine)))
}

/// [`common::json_of`] against a specific router.
async fn json_of_send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let (status, body) = send_to(app.clone(), req).await;
    (
        status,
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("bad JSON {body:?}: {e}")),
    )
}

fn chat_image_part(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

// --- accepted on every front end ---

#[tokio::test]
async fn openai_chat_accepts_png() {
    let img = png();
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"messages": [{"role": "user", "content": [
            chat_image_part(&data_url("image/png", &img)),
            {"type": "text", "text": "what is this?"},
        ]}]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert_announced(text, &[("image/png", &img)]);
    assert!(text.ends_with("You said: what is this?"), "{text:?}");
}

#[tokio::test]
async fn openai_chat_accepts_image_url_as_plain_string() {
    let img = png();
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": data_url("image/png", &img)},
            {"type": "text", "text": "hi"},
        ]}]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_announced(
        v["choices"][0]["message"]["content"].as_str().unwrap(),
        &[("image/png", &img)],
    );
}

#[tokio::test]
async fn openai_chat_keeps_image_order_and_counts_tokens() {
    let (img, img2) = (png(), jpeg());
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"messages": [{"role": "user", "content": [
            chat_image_part(&data_url("image/png", &img)),
            chat_image_part(&data_url("image/jpeg", &img2)),
            {"type": "text", "text": "two images"},
        ]}]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_announced(
        v["choices"][0]["message"]["content"].as_str().unwrap(),
        &[("image/png", &img), ("image/jpeg", &img2)],
    );
    // "hi there" is replaced by "two images" (2 words); each image counts 1 token.
    assert_eq!(v["usage"]["prompt_tokens"], 4, "{v}");
}

#[tokio::test]
async fn anthropic_accepts_png() {
    let img = png();
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64(&img),
            }},
            {"type": "text", "text": "describe it"},
        ]}]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_announced(
        v["content"][0]["text"].as_str().unwrap(),
        &[("image/png", &img)],
    );
    assert!(
        v["content"][0]["text"]
            .as_str()
            .unwrap()
            .ends_with("You said: describe it"),
        "{v}"
    );
}

#[tokio::test]
async fn responses_accepts_png() {
    let img = png();
    let (status, v) = json_of(post(
        "/v1/responses",
        json!({
            "model": "mock",
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": data_url("image/png", &img)},
                {"type": "input_text", "text": "what is this?"},
            ]}],
        }),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let text = v["output"][0]["content"][0]["text"].as_str().unwrap();
    assert_announced(text, &[("image/png", &img)]);
    assert!(text.ends_with("You said: what is this?"), "{text:?}");
}

// --- rejected inputs ---

async fn chat_rejected(content: Value, message: &str, param: Option<&str>) {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"messages": [{"role": "user", "content": [content]}]})),
    ))
    .await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], message, "{v}");
    match param {
        Some(p) => assert_eq!(v["error"]["param"], p, "{v}"),
        None => assert!(v["error"]["param"].is_null(), "{v}"),
    }
}

#[tokio::test]
async fn remote_urls_rejected_on_every_front_end() {
    let msg = "remote image URLs are not supported; send a data: URL";
    chat_rejected(
        chat_image_part("https://example.com/cat.png"),
        msg,
        Some("messages[0].content[0]"),
    )
    .await;
    chat_rejected(
        chat_image_part("http://example.com/cat.png"),
        msg,
        Some("messages[0].content[0]"),
    )
    .await;

    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
        ]}]})),
    ))
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(v["error"]["message"], msg, "{v}");
    assert_eq!(v["error"]["param"], "messages[0].content[0]", "{v}");

    let (status, v) = json_of(post(
        "/v1/responses",
        json!({"model": "mock", "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]}]}),
    ))
    .await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], msg, "{v}");
    assert_eq!(v["error"]["param"], "input[0].content[0]", "{v}");
}

#[tokio::test]
async fn unsupported_mime_rejected() {
    chat_rejected(
        chat_image_part(&data_url("image/tiff", &png())),
        "unsupported image type image/tiff",
        Some("messages[0].content[0]"),
    )
    .await;
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "application/pdf", "data": "eA==",
            }},
        ]}]})),
    ))
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(
        v["error"]["message"], "unsupported image type application/pdf",
        "{v}"
    );
}

#[tokio::test]
async fn accepted_mimes() {
    for mime in ["image/png", "image/jpeg", "image/webp", "image/gif"] {
        let img = png();
        let (status, v) = json_of(post(
            "/v1/chat/completions",
            chat(json!({"messages": [{"role": "user", "content": [
                chat_image_part(&data_url(mime, &img)),
            ]}]})),
        ))
        .await;
        assert_eq!(status, StatusCode::OK, "{mime}: {v}");
        assert_announced(
            v["choices"][0]["message"]["content"].as_str().unwrap(),
            &[(mime, &img)],
        );
    }
}

#[tokio::test]
async fn invalid_base64_rejected() {
    chat_rejected(
        chat_image_part("data:image/png;base64,@@@not base64@@@"),
        "invalid base64 image data",
        Some("messages[0].content[0]"),
    )
    .await;
}

#[tokio::test]
async fn non_data_url_rejected() {
    chat_rejected(
        chat_image_part("an image, trust me"),
        "images must be sent as data: URLs",
        Some("messages[0].content[0]"),
    )
    .await;
}

#[tokio::test]
async fn too_many_images_rejected() {
    let part = chat_image_part(&data_url("image/png", &png()));
    let parts = vec![part.clone(); 17];
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"messages": [
            {"role": "user", "content": [part, {"type": "text", "text": "x"}]},
            {"role": "user", "content": parts},
        ]})),
    ))
    .await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["message"], "too many images", "{v}");
}

#[tokio::test]
async fn images_to_a_non_vision_model_rejected() {
    let app = app_no_vision();
    let img = png();
    let (status, v) = json_of_send(
        &app,
        post(
            "/v1/chat/completions",
            chat(json!({"messages": [{"role": "user", "content": [
                chat_image_part(&data_url("image/png", &img)),
            ]}]})),
        ),
    )
    .await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(
        v["error"]["message"], "model mock does not accept images",
        "{v}"
    );

    let (status, v) = json_of_send(
        &app,
        post(
            "/v1/messages",
            messages(json!({"messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64(&img),
                }},
            ]}]})),
        ),
    )
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(
        v["error"]["message"], "model mock does not accept images",
        "{v}"
    );
}

// --- /v1/models capabilities ---

#[tokio::test]
async fn models_advertise_vision_capability() {
    let (status, v) = get("/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["data"][0]["id"], "mock");
    assert_eq!(v["data"][0]["capabilities"], json!(["vision"]));

    let (status, v) = json_of_send(
        &app_no_vision(),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["data"][0]["capabilities"], json!([]));
}
