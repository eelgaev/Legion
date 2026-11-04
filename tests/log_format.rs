//! `legion --log-format pretty|json`: the real binary's log output, and `RUST_LOG`
//! filtering in both modes.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use serde_json::Value as Json;

/// Spawns the binary with `args` and an explicit environment, collecting stdout lines.
fn spawn_legion(args: &[&str], env: &[(&str, &str)]) -> (Child, Receiver<String>) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
    for var in [
        "LEGION_LOG_FORMAT",
        "RUST_LOG",
        "LEGION_HOST",
        "LEGION_PORT",
    ] {
        cmd.env_remove(var);
    }
    cmd.envs(env.iter().map(|(k, v)| (*k, *v)));
    let mut child = cmd
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if line.is_err() || tx.send(line.unwrap()).is_err() {
                break;
            }
        }
    });
    (child, rx)
}

/// Reads stdout lines until one contains `needle`, panicking (with what was seen) on timeout.
fn wait_for_line(rx: &Receiver<String>, needle: &str) -> Vec<String> {
    let mut seen = Vec::new();
    while let Ok(line) = rx.recv_timeout(Duration::from_secs(15)) {
        if line.contains(needle) {
            seen.push(line);
            return seen;
        }
        seen.push(line);
    }
    panic!("no line containing {needle:?}; saw: {seen:?}");
}

fn stop(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn parse_json(line: &str) -> Json {
    serde_json::from_str(line).unwrap_or_else(|e| panic!("not a JSON line {line:?}: {e}"))
}

#[test]
fn json_flag_prints_one_json_object_per_line() {
    let (child, rx) = spawn_legion(
        &["--log-format", "json", "serve", "--port", "0"],
        &[("RUST_LOG", "info")],
    );
    let lines = wait_for_line(&rx, "listening");
    for line in &lines {
        let v = parse_json(line);
        assert!(!v["timestamp"].is_null(), "no timestamp in {line:?}");
        assert_eq!(v["level"], "INFO");
    }
    let last = parse_json(lines.last().unwrap());
    assert!(
        last["fields"]["message"]
            .as_str()
            .unwrap()
            .starts_with("listening on http://"),
        "{last}"
    );
    stop(child);
}

#[test]
fn json_flag_also_works_after_the_subcommand() {
    let (child, rx) = spawn_legion(
        &["serve", "--log-format", "json", "--port", "0"],
        &[("RUST_LOG", "info")],
    );
    let lines = wait_for_line(&rx, "listening");
    for line in &lines {
        parse_json(line);
    }
    stop(child);
}

#[test]
fn env_var_selects_json() {
    let (child, rx) = spawn_legion(
        &["serve", "--port", "0"],
        &[("RUST_LOG", "info"), ("LEGION_LOG_FORMAT", "json")],
    );
    let lines = wait_for_line(&rx, "listening");
    for line in &lines {
        parse_json(line);
    }
    stop(child);
}

#[test]
fn default_output_is_unchanged_text() {
    let (child, rx) = spawn_legion(&["serve", "--port", "0"], &[("RUST_LOG", "info")]);
    let lines = wait_for_line(&rx, "listening");
    let line = lines.last().unwrap();
    assert!(
        !line.trim_start().starts_with('{'),
        "unexpected JSON: {line}"
    );
    assert!(
        line.contains("INFO") && line.contains("listening on http://"),
        "{line}"
    );
    stop(child);
}

#[test]
fn rust_log_filters_in_both_modes() {
    for args in [
        vec!["serve", "--port", "0"],
        vec!["--log-format", "json", "serve", "--port", "0"],
    ] {
        let (mut child, rx) = spawn_legion(&args, &[("RUST_LOG", "warn")]);
        // The info-level "listening" line must stay below the warn threshold; the server
        // itself must still be up, so an empty stdout is not just a crashed process.
        match rx.recv_timeout(Duration::from_secs(2)) {
            Err(RecvTimeoutError::Timeout) => {}
            other => panic!("RUST_LOG=warn must suppress the info log, got {other:?}"),
        }
        assert_eq!(
            child.try_wait().unwrap(),
            None,
            "server died with RUST_LOG=warn ({args:?})"
        );
        let extra: Vec<String> = rx.try_iter().collect();
        assert!(
            extra.is_empty(),
            "unexpected log lines with RUST_LOG=warn: {extra:?}"
        );
        stop(child);
    }
}
