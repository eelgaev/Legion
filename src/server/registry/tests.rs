//! `--registry-api`: URL normalisation, the body, the advertised address, the key, and a
//! registration against a mock `/register` on loopback that fails once and then succeeds.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use serde_json::{Value, json};

use super::*;
use crate::server::ServerConfig;

#[test]
fn register_url_appends_register_once() {
    for (given, want) in [
        ("http://192.0.2.3:8080", "http://192.0.2.3:8080/register"),
        ("http://192.0.2.3:8080/", "http://192.0.2.3:8080/register"),
        (
            "http://192.0.2.3:8080/register",
            "http://192.0.2.3:8080/register",
        ),
        (
            "http://192.0.2.3:8080/register/",
            "http://192.0.2.3:8080/register",
        ),
        (
            "https://proxy.example/llm",
            "https://proxy.example/llm/register",
        ),
        (
            "https://proxy.example/llm/register",
            "https://proxy.example/llm/register",
        ),
        ("http://[::1]:9000?x=1#y", "http://[::1]:9000/register"),
    ] {
        assert_eq!(register_url(given).unwrap().as_str(), want, "{given}");
    }
}

#[test]
fn register_url_refuses_non_http() {
    assert!(register_url("192.0.2.3:8080").is_err());
    assert!(register_url("ftp://192.0.2.3/").is_err());
    assert!(register_url("not a url").is_err());
}

#[test]
fn body_shape_is_the_proxys() {
    let body = RegisterBody {
        host: "192.0.2.3".into(),
        port: 8090,
        api_key: "secret".into(),
    };
    assert_eq!(
        serde_json::to_value(&body).unwrap(),
        json!({"host": "192.0.2.3", "port": 8090, "api_key": "secret"})
    );
    // The key never reaches a log through Debug.
    assert!(!format!("{body:?}").contains("secret"));
}

#[test]
fn api_key_is_the_first_or_empty() {
    assert_eq!(api_key(&[]), "");
    assert_eq!(api_key(&["a".into(), "b".into()]), "a");
}

#[test]
fn advertised_host_choice() {
    let explicit: IpAddr = "192.0.2.7".parse().unwrap();
    let bound: IpAddr = "198.51.100.4".parse().unwrap();
    let any4 = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
    let any6: IpAddr = "::".parse().unwrap();
    // --registry-host wins, else a specific --host, else the route lookup (None).
    assert_eq!(fixed_host(Some(explicit), bound), Some(explicit));
    assert_eq!(fixed_host(Some(explicit), any4), Some(explicit));
    assert_eq!(fixed_host(None, bound), Some(bound));
    assert_eq!(fixed_host(None, any4), None);
    assert_eq!(fixed_host(None, any6), None);
}

#[tokio::test]
async fn route_to_a_loopback_proxy_is_loopback() {
    let ip = route_local_ip(&register_url("http://127.0.0.1:9").unwrap())
        .await
        .unwrap();
    assert_eq!(ip, IpAddr::V4(Ipv4Addr::LOCALHOST));
    let ip = route_local_ip(&register_url("http://localhost:9").unwrap())
        .await
        .unwrap();
    assert!(ip.is_loopback(), "{ip}");
}

/// The mock proxy: records each body, refuses the first with 503, then answers 201.
#[derive(Clone, Default)]
struct Proxy {
    bodies: Arc<Mutex<Vec<Value>>>,
    seen: Arc<tokio::sync::Notify>,
}

async fn register(State(p): State<Proxy>, Json(body): Json<Value>) -> (StatusCode, Json<Value>) {
    let n = {
        let mut bodies = p.bodies.lock().unwrap();
        bodies.push(body);
        bodies.len()
    };
    p.seen.notify_one();
    if n == 1 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "health check failed"})),
        );
    }
    (
        StatusCode::CREATED,
        Json(json!({"status": "registered", "url": "http://127.0.0.1:4321", "models": 1})),
    )
}

#[tokio::test]
async fn retries_until_the_proxy_takes_it() {
    let proxy = Proxy::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_addr = listener.local_addr().unwrap();
    let app = axum::Router::new()
        .route("/register", post(register))
        .with_state(proxy.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let config = ServerConfig {
        api_keys: vec!["first".into(), "second".into()],
        registry: RegistryConfig {
            proxies: vec![register_url(&format!("http://{proxy_addr}")).unwrap()],
            interval: Duration::from_secs(3600),
            backoff_min: Duration::from_millis(10),
            backoff_max: Duration::from_millis(20),
            ..Default::default()
        },
        ..Default::default()
    };
    // A wildcard bind: the address comes from the route to the proxy.
    let bound: SocketAddr = "0.0.0.0:4321".parse().unwrap();
    let tasks = start(&config, bound);
    assert_eq!(tasks.len(), 1);
    tokio::time::timeout(Duration::from_secs(10), async {
        while proxy.bodies.lock().unwrap().len() < 2 {
            proxy.seen.notified().await;
        }
    })
    .await
    .expect("the registration was not retried");
    // The interval is an hour: no third call.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let bodies = proxy.bodies.lock().unwrap().clone();
    let want = json!({"host": "127.0.0.1", "port": 4321, "api_key": "first"});
    assert_eq!(bodies, vec![want.clone(), want]);
    for t in tasks {
        t.abort();
    }
    server.abort();
}

#[test]
fn no_proxies_starts_nothing() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let tasks =
        rt.block_on(async { start(&ServerConfig::default(), "0.0.0.0:1".parse().unwrap()) });
    assert!(tasks.is_empty());
}
