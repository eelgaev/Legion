//! Starts a `legion serve` child on a port of its own choosing.
//!
//! The child gets `--port 0`, binds an ephemeral port itself and logs `listening on
//! <scheme>://<addr>` on stdout; [`spawn`] reads the port from that line. Picking a free
//! port in the test and passing it on raced: another test could bind it before the child
//! did, and the readiness check then reached the other test's server (#649).

use std::io::{BufRead, BufReader};
use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// Spawns `cmd` (a `legion serve` without `--port`) with `--port 0` and returns the child
/// once it listens, with the port it bound. Its stdout is drained in the background, so a
/// chatty server never blocks on a full pipe. Panics (after killing the child) if no
/// `listening on` line arrives within 20 s.
pub fn spawn(mut cmd: Command) -> (Child, u16) {
    let mut child = cmd
        .args(["--port", "0"])
        // The line is an info log: make sure an inherited RUST_LOG does not hide it.
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut tx = Some(tx);
        for line in BufReader::new(stdout).split(b'\n') {
            let Ok(line) = line else { break };
            if let Some(port) = listening_port(&String::from_utf8_lossy(&line))
                && let Some(tx) = tx.take()
            {
                let _ = tx.send(port);
            }
        }
    });
    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(port) => (child, port),
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            panic!("server did not start");
        }
    }
}

/// A loopback port that refuses connections for as long as the returned socket lives: the
/// socket is bound but never listens, so no other test can bind the port meanwhile (it does
/// not set `SO_REUSEADDR`), unlike a port picked and released.
pub fn refused_port() -> (tokio::net::TcpSocket, u16) {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let port = socket.local_addr().unwrap().port();
    (socket, port)
}

/// The port in a `listening on http://127.0.0.1:1234` log line (pretty or JSON format,
/// with or without ANSI colour), or `None` for any other line.
pub fn listening_port(line: &str) -> Option<u16> {
    let rest = &line[line.find("listening on ")?..];
    let addr = &rest[rest.find("://")? + 3..];
    let end = addr
        .find(|c: char| c.is_whitespace() || c == '"' || c == '\x1b')
        .unwrap_or(addr.len());
    addr[..end].parse::<SocketAddr>().ok().map(|a| a.port())
}
