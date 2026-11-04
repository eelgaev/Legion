//! SIGTERM on `legion serve`: in-flight streams finish (or are cut at the timeout) and the
//! process exits 0.

mod common;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;

/// A `legion serve` on a port it bound itself, and that port.
fn spawn(timeout_secs: u64) -> (Child, u16) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
    cmd.args(["serve", "--host", "127.0.0.1", "--mock-step-ms", "50"])
        .args(["--shutdown-timeout-secs", &timeout_secs.to_string()]);
    common::serve::spawn(cmd)
}

/// Sends a streaming chat request whose reply takes `words` decode steps.
fn start_stream(port: u16, words: usize) -> TcpStream {
    let content = vec!["word"; words].join(" ");
    let body = format!(
        r#"{{"model":"mock","stream":true,"max_tokens":1000,"messages":[{{"role":"user","content":"{content}"}}]}}"#
    );
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
         connection: close\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    // Wait for the first bytes so the request is certainly in flight.
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut first = [0u8; 1];
    s.read_exact(&mut first).unwrap();
    s
}

fn wait_exit(child: &mut Child, within: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + within;
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("server did not exit in time");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn sigterm(child: &Child) {
    kill(Pid::from_raw(child.id() as i32), Signal::SIGTERM).unwrap();
}

#[test]
fn sigterm_lets_a_stream_finish() {
    let (mut child, port) = spawn(30);
    let mut stream = start_stream(port, 20);
    sigterm(&child);
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
    assert!(
        String::from_utf8_lossy(&rest).contains("[DONE]"),
        "stream was cut short"
    );
    assert!(wait_exit(&mut child, Duration::from_secs(10)).success());
}

#[test]
fn sigterm_cuts_a_stream_at_the_timeout() {
    let (mut child, port) = spawn(1);
    let mut stream = start_stream(port, 400); // ~20 s of decoding
    let start = Instant::now();
    sigterm(&child);
    let status = wait_exit(&mut child, Duration::from_secs(10));
    assert!(status.success());
    assert!(start.elapsed() < Duration::from_secs(8));
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
    assert!(!String::from_utf8_lossy(&rest).contains("[DONE]"));
}
