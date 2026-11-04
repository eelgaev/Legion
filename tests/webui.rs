mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use common::{app_with, post, send_to};
use serde_json::json;
use tower::ServiceExt;

async fn get(app: axum::Router, uri: &str) -> (StatusCode, Option<String>, String) {
    let resp = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string());
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, ctype, String::from_utf8_lossy(&body).into_owned())
}

#[tokio::test]
async fn index_is_html() {
    let resp = common::app()
        .oneshot(Request::get("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()[header::CACHE_CONTROL], "no-cache");
    assert!(
        resp.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&body).contains("<title>Legion"));
}

#[tokio::test]
async fn assets_have_content_types() {
    let (s, ct, body) = get(common::app(), "/assets/js/main.js").await;
    assert_eq!(s, StatusCode::OK);
    assert!(ct.unwrap().starts_with("text/javascript"));
    assert!(body.contains("initChat"));
    let (s, ct, _) = get(common::app(), "/assets/style.css").await;
    assert_eq!(s, StatusCode::OK);
    assert!(ct.unwrap().starts_with("text/css"));
}

#[tokio::test]
async fn unknown_asset_is_404() {
    let (s, _, _) = get(common::app(), "/assets/nope.js").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn disabled_webui_is_404() {
    let app = || {
        app_with(|state| {
            let mut config = (*state.config).clone();
            config.webui = false;
            legion::server::AppState::with_config(state.engine, config)
        })
    };
    assert_eq!(get(app(), "/").await.0, StatusCode::NOT_FOUND);
    assert_eq!(
        get(app(), "/assets/js/main.js").await.0,
        StatusCode::NOT_FOUND
    );
    // The API is unaffected.
    let (s, _) = send_to(
        app(),
        post(
            "/v1/chat/completions",
            json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn ui_loads_without_key_but_api_needs_one() {
    let app = || {
        app_with(|state| {
            let mut config = (*state.config).clone();
            config.api_keys = vec!["secret".into()];
            legion::server::AppState::with_config(state.engine, config)
        })
    };
    assert_eq!(get(app(), "/").await.0, StatusCode::OK);
    assert_eq!(get(app(), "/assets/style.css").await.0, StatusCode::OK);
    assert_eq!(get(app(), "/v1/models").await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn ui_files_are_served() {
    for (path, prefix) in [
        ("/assets/js/markdown.js", "text/javascript"),
        ("/assets/js/chat.js", "text/javascript"),
    ] {
        let (s, ct, _) = get(common::app(), path).await;
        assert_eq!(s, StatusCode::OK, "{path}");
        assert!(ct.unwrap().starts_with(prefix), "{path}");
    }
    // The UI ships no third-party code: the old vendored libraries are gone.
    let (s, _, _) = get(common::app(), "/assets/vendor/marked.esm.js").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn settings_dialog_is_served() {
    let (s, ct, body) = get(common::app(), "/assets/js/settings.js").await;
    assert_eq!(s, StatusCode::OK);
    assert!(ct.unwrap().starts_with("text/javascript"));
    assert!(body.contains("legion.settings"));
    // The dialog markup and its trigger ship in the page.
    let (_, _, index) = get(common::app(), "/").await;
    assert!(index.contains("id=\"settings-dialog\""));
    assert!(index.contains("id=\"settings-button\""));
    assert!(index.contains("id=\"s-temperature\""));
}
