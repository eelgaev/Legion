//! `legion --version` and the `serve` startup log report the exact build:
//! crate version, legion git commit, ggml commit and CUDA on/off.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;

use serde_json::Value as Json;

/// The identity `build.rs` baked into this test binary — the same one the `legion`
/// binary under test was built with.
fn expected() -> String {
    format!(
        "{} ({}, ggml {}, cuda {})",
        env!("CARGO_PKG_VERSION"),
        env!("LEGION_GIT_HASH"),
        env!("LEGION_GGML_HASH"),
        env!("LEGION_CUDA"),
    )
}

#[test]
fn version_flag_prints_the_build_identity() {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        format!("legion {}", expected())
    );
}

/// Spawns `legion serve` with a JSON log on stdout and collects the lines until
/// (and including) the "listening" line, then stops it.
fn serve_log_lines() -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_legion"))
        .env("RUST_LOG", "info")
        .env("LEGION_LOG_FORMAT", "json")
        .args(["serve", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let lines = read_until(&rx, "listening");
    stop(child);
    lines
}

fn read_until(rx: &Receiver<String>, needle: &str) -> Vec<String> {
    let mut seen = Vec::new();
    while let Ok(line) = rx.recv_timeout(Duration::from_secs(15)) {
        let done = line.contains(needle);
        seen.push(line);
        if done {
            return seen;
        }
    }
    panic!("no line containing {needle:?}; saw: {seen:?}");
}

fn stop(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn serve_logs_the_build_identity_once_at_startup() {
    let lines = serve_log_lines();
    let versions: Vec<Json> = lines
        .iter()
        .map(|l| -> Json {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON {l:?}: {e}"))
        })
        .filter(|v| v["fields"]["message"] == "legion starting")
        .collect();
    assert_eq!(
        versions.len(),
        1,
        "want exactly one startup line in {lines:?}"
    );
    assert_eq!(versions[0]["fields"]["version"], expected());
}
