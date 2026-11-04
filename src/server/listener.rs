//! The plain-HTTP listener: accepted connections send without Nagle's wait.
//!
//! A streamed response is many small writes (one SSE event each, then `[DONE]` and the chunked
//! terminator). With Nagle's algorithm on, the kernel holds a small write while an earlier small
//! one is unacknowledged, which against a client that delays its ACKs (up to 40 ms on Linux,
//! 200 ms on Windows) delays tokens and the response end. `TCP_NODELAY` sends each write at
//! once; the bytes on the wire are the same.

use std::net::SocketAddr;

use axum::serve::{ListenerExt, TapIo};
use tokio::net::{TcpListener, TcpStream};

/// What [`bind`] returns: a Tokio listener that sets `TCP_NODELAY` on every accepted stream.
pub(super) type NoDelayListener = TapIo<TcpListener, fn(&mut TcpStream)>;

/// Binds `addr`; every connection it accepts has `TCP_NODELAY` set.
pub(super) async fn bind(addr: SocketAddr) -> std::io::Result<NoDelayListener> {
    Ok(TcpListener::bind(addr)
        .await?
        .tap_io(nodelay as fn(&mut TcpStream)))
}

/// Sets `TCP_NODELAY` on an accepted stream; a failure only costs latency, so it is logged.
fn nodelay(stream: &mut TcpStream) {
    if let Err(e) = stream.set_nodelay(true) {
        tracing::debug!(error = %e, "cannot set TCP_NODELAY on an accepted connection");
    }
}

#[cfg(test)]
mod tests;
