use std::net::SocketAddr;

use axum::serve::Listener;
use tokio::net::TcpStream;

use super::bind;

#[tokio::test]
async fn accepted_streams_have_nodelay() {
    let mut listener = bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
    let addr = listener.local_addr().unwrap();
    for _ in 0..2 {
        let client = tokio::spawn(TcpStream::connect(addr));
        let (stream, _) = listener.accept().await;
        assert!(
            stream.nodelay().unwrap(),
            "accepted stream without TCP_NODELAY"
        );
        client.await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn plain_tokio_listener_leaves_nagle_on() {
    // The default this module exists to change: Tokio's accepted streams keep Nagle on.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = tokio::spawn(TcpStream::connect(addr));
    let (stream, _) = listener.accept().await.unwrap();
    assert!(!stream.nodelay().unwrap());
    client.await.unwrap().unwrap();
}
