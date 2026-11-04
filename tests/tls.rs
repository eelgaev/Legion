//! `legion serve` over HTTPS: a trusted self-signed certificate reaches health, regular chat,
//! and streaming chat; incomplete TLS settings fail before startup.

mod common;

use std::process::{Child, Command};

use futures::StreamExt as _;
use rcgen::{CertifiedKey, generate_simple_self_signed};
use serde_json::{Value, json};

struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start(cert: &std::path::Path, key: &std::path::Path) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
        cmd.args(["serve", "--host", "127.0.0.1"])
            .args(["--tls-cert"])
            .arg(cert)
            .args(["--tls-key"])
            .arg(key);
        let (child, port) = common::serve::spawn(cmd);
        Self { child, port }
    }

    fn url(&self) -> String {
        format!("https://localhost:{}", self.port)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
async fn serves_health_and_chat_over_https() {
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(["localhost".to_string()]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let cert_path = dir.path().join("cert.pem");
    let key_path = dir.path().join("key.pem");
    std::fs::write(&cert_path, cert.pem()).unwrap();
    std::fs::write(&key_path, signing_key.serialize_pem()).unwrap();

    let server = Server::start(&cert_path, &key_path);
    let certificate = reqwest::Certificate::from_pem(cert.pem().as_bytes()).unwrap();
    let client = reqwest::Client::builder()
        .add_root_certificate(certificate)
        .build()
        .unwrap();

    let health_response = client
        .get(format!("{}/health", server.url()))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let health: Value = serde_json::from_str(&health_response.text().await.unwrap()).unwrap();
    assert_eq!(health["status"], "ok");

    let webui = client
        .get(server.url())
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(webui.contains("<!doctype html>"), "{webui}");

    let body = json!({"model": "mock", "messages": [{"role": "user", "content": "secure"}]});
    let chat_response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let chat: Value = serde_json::from_str(&chat_response.text().await.unwrap()).unwrap();
    assert_eq!(chat["choices"][0]["message"]["content"], "You said: secure");

    let mut stream_body = body;
    stream_body["stream"] = json!(true);
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .header("content-type", "application/json")
        .body(stream_body.to_string())
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let chunks = response.bytes_stream().collect::<Vec<_>>().await;
    let bytes = chunks
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .concat();
    let stream = String::from_utf8(bytes).unwrap();
    let events = common::sse_events(&stream);
    assert_eq!(events.last().unwrap().1, "[DONE]", "{stream}");
    let text = events[..events.len() - 1]
        .iter()
        .filter_map(|(_, data)| {
            serde_json::from_str::<Value>(data).ok()?["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect::<String>();
    assert_eq!(text, "You said: secure");
}

#[test]
fn rejects_cert_or_key_on_its_own() {
    for (flag, other) in [("--tls-cert", "--tls-key"), ("--tls-key", "--tls-cert")] {
        let output = Command::new(env!("CARGO_BIN_EXE_legion"))
            .args(["serve", flag, "unused.pem"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("{flag} requires {other}")),
            "{stderr}"
        );
    }
}
