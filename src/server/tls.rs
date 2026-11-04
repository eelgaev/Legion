//! Rustls listener for the existing server router.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context as _;
use axum_server::Handle;
use axum_server::accept::NoDelayAcceptor;
use axum_server::tls_rustls::RustlsConfig;

use super::{AppState, TlsConfig, router, shutdown_signal};

pub(super) async fn serve(addr: SocketAddr, state: AppState, tls: TlsConfig) -> anyhow::Result<()> {
    let config = RustlsConfig::from_pem_file(&tls.cert, &tls.key)
        .await
        .with_context(|| {
            format!(
                "load TLS certificate {} and private key {}",
                tls.cert.display(),
                tls.key.display()
            )
        })?;
    let listener = std::net::TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    tracing::info!("listening on https://{}", listener.local_addr()?);
    super::registry::start(&state.config, listener.local_addr()?);

    let timeout = Duration::from_secs(state.config.shutdown_timeout_secs);
    let in_flight = state.in_flight.clone();
    let handle = Handle::new();
    // TCP_NODELAY on every accepted connection, as on plain HTTP (`listener.rs`).
    let server = axum_server::from_tcp_rustls(listener, config)?
        .map(|acceptor| acceptor.acceptor(NoDelayAcceptor))
        .handle(handle.clone())
        .serve(router(state).into_make_service());
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        _ = shutdown_signal() => {
            tracing::info!("shutting down, {} requests in flight", in_flight.count());
            handle.graceful_shutdown(Some(timeout));
            server.await?;
        }
    }
    Ok(())
}
