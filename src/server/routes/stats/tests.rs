use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use super::{interval, max_rank};
use crate::engine::mock::{MockConfig, MockEngine};
use crate::server::{AppState, ServerConfig, router};

const ROUTES: [&str; 3] = ["/v1/stats", "/v1/stats/stream", "/v1/logs/stream"];

fn app(keys: &[&str]) -> Router {
    let engine = MockEngine::new(MockConfig {
        step: Duration::ZERO,
        ..Default::default()
    });
    let config = ServerConfig {
        api_keys: keys.iter().map(|k| k.to_string()).collect(),
        ..ServerConfig::default()
    };
    router(AppState::with_config(Arc::new(engine), config))
}

/// The status only: the two streams never end, so their bodies are not read.
async fn status(app: Router, uri: &str, key: Option<&str>) -> StatusCode {
    let mut req = Request::get(uri);
    if let Some(k) = key {
        req = req.header("authorization", format!("Bearer {k}"));
    }
    app.oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap()
        .status()
}

#[tokio::test]
async fn dashboard_routes_need_the_key() {
    for uri in ROUTES {
        let app = app(&["sk-dash"]);
        assert_eq!(
            status(app.clone(), uri, None).await,
            StatusCode::UNAUTHORIZED,
            "{uri} without a key"
        );
        assert_eq!(
            status(app.clone(), uri, Some("sk-wrong")).await,
            StatusCode::UNAUTHORIZED,
            "{uri} with a wrong key"
        );
        assert_eq!(
            status(app, uri, Some("sk-dash")).await,
            StatusCode::OK,
            "{uri} with the key"
        );
    }
}

#[tokio::test]
async fn dashboard_routes_open_without_keys_configured() {
    for uri in ROUTES {
        assert_eq!(status(app(&[]), uri, None).await, StatusCode::OK, "{uri}");
    }
}

#[tokio::test]
async fn stats_is_a_version_1_snapshot() {
    let resp = app(&[])
        .oneshot(Request::get("/v1/stats").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let snap: crate::dashboard::snapshot::Snapshot = serde_json::from_slice(&body).unwrap();
    assert_eq!(snap.version, 1);
}

#[test]
fn interval_is_clamped() {
    assert_eq!(interval(None), Duration::from_millis(250));
    assert_eq!(interval(Some(10)), Duration::from_millis(100));
    assert_eq!(interval(Some(1000)), Duration::from_millis(1000));
    assert_eq!(interval(Some(60_000)), Duration::from_millis(5000));
}

#[test]
fn level_filter() {
    assert_eq!(max_rank(None), 5);
    assert_eq!(max_rank(Some("warn")), 2);
    assert_eq!(max_rank(Some("INFO")), 3);
    assert_eq!(max_rank(Some("1")), 1);
    assert_eq!(max_rank(Some("9")), 5);
}
