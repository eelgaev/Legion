//! `legion serve --port 0` binds an ephemeral port and logs it, which the integration tests'
//! server helper relies on instead of a picked-then-released port (#649).

mod common;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::Command;

use common::serve::listening_port;

fn health(port: u16) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
        .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

#[test]
fn port_zero_servers_get_their_own_ports_and_answer() {
    let start = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
        cmd.args(["serve", "--host", "127.0.0.1"]);
        common::serve::spawn(cmd)
    };
    let (mut a, port_a) = start();
    let (mut b, port_b) = start();
    assert_ne!(port_a, 0);
    assert_ne!(port_a, port_b);
    for port in [port_a, port_b] {
        let reply = health(port);
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    }
    a.kill().unwrap();
    a.wait().unwrap();
    b.kill().unwrap();
    b.wait().unwrap();
}

#[test]
fn refused_port_refuses_while_held() {
    let (held, port) = common::serve::refused_port();
    assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
    assert!(std::net::TcpListener::bind(("127.0.0.1", port)).is_err());
    drop(held);
}

#[test]
fn listening_port_reads_pretty_json_and_coloured_lines() {
    assert_eq!(
        listening_port("2026-10-28T10:00:00.000000Z  INFO listening on http://127.0.0.1:41234"),
        Some(41234)
    );
    assert_eq!(
        listening_port(
            "\x1b[2m2026-10-28T10:00:00Z\x1b[0m \x1b[32m INFO\x1b[0m listening on https://127.0.0.1:443\x1b[0m"
        ),
        Some(443)
    );
    assert_eq!(
        listening_port(
            r#"{"timestamp":"t","level":"INFO","fields":{"message":"listening on http://[::1]:8090"}}"#
        ),
        Some(8090)
    );
    assert_eq!(listening_port("INFO model loaded"), None);
    assert_eq!(listening_port("listening on nothing"), None);
}
