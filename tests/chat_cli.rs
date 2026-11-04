//! `legion chat` against `legion serve` (mock engine): `--once` prints exactly the reply,
//! a wrong API key exits 1 with `error:` on stderr, and a piped interactive session
//! answers, resets and quits.

mod common;

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A `legion serve` on a port it bound itself; always killed when the guard drops.
struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start(api_key: Option<&str>) -> Server {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
        cmd.args(["serve", "--host", "127.0.0.1"]);
        if let Some(key) = api_key {
            cmd.args(["--api-key", key]);
        }
        let (child, port) = common::serve::spawn(cmd);
        Server { child, port }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn chat(url: &str, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["chat", "--url", url])
        .args(extra)
        .output()
        .unwrap()
}

/// Runs interactive mode with `input` piped in and collects the output once it exits.
fn interactive(url: &str, extra: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["chat", "--url", url])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("chat did not exit after /exit");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child.wait_with_output().unwrap()
}

#[test]
fn once_prints_the_reply_and_exits_zero() {
    let server = Server::start(None);
    let out = chat(&server.url(), &["--once", "hi"]);
    assert!(out.status.success());
    assert_eq!(out.stdout, b"You said: hi\n");
}

#[test]
fn wrong_api_key_exits_one_with_error_on_stderr() {
    let server = Server::start(Some("sk-secret"));
    let out = chat(&server.url(), &["--api-key", "wrong", "--once", "hi"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.starts_with("error:"), "{err}");
    assert!(err.contains("invalid API key"), "{err}");
    // The right key works.
    let out = chat(&server.url(), &["--api-key", "sk-secret", "--once", "hi"]);
    assert!(out.status.success());
    assert_eq!(out.stdout, b"You said: hi\n");
}

#[test]
fn interactive_answers_resets_and_quits() {
    let server = Server::start(None);
    let out = interactive(&server.url(), &[], b"\nhello\n/reset\nagain\n/exit\n");
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("> "), "{text}");
    assert!(text.contains("You said: hello"), "{text}");
    assert!(text.contains("(history cleared)"), "{text}");
    assert!(text.contains("You said: again"), "{text}");
    // Interactive mode adds the stats line from the usage chunk.
    assert!(text.contains("tok/s]"), "{text}");
}

#[test]
fn interactive_continues_after_an_error() {
    let server = Server::start(Some("sk-secret"));
    let out = interactive(
        &server.url(),
        &["--api-key", "wrong"],
        b"hi\nthere\n/exit\n",
    );
    assert!(out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(err.matches("error:").count(), 2, "{err}");
}
