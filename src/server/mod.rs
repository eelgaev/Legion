//! HTTP server: OpenAI-compatible and Anthropic-compatible routes over one [`Engine`].

mod collect;
pub mod config;
mod convert;
pub mod error;
pub mod fair_share;
mod listener;
pub mod metrics;
mod middleware;
pub mod registry;
mod response_format;
mod routes;
pub mod stats;
mod tls;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;

pub use config::{ServerConfig, TlsConfig};

use crate::engine::{Engine, EngineError};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub config: Arc<ServerConfig>,
    in_flight: middleware::InFlight,
    metrics: Arc<metrics::Metrics>,
    /// The live dashboard's data (`/v1/stats`, `serve --tui`; #1198).
    pub stats: Arc<stats::StatsHub>,
}

impl AppState {
    pub fn new(engine: Arc<dyn Engine>) -> Self {
        Self::with_config(engine, ServerConfig::default())
    }

    pub fn with_config(engine: Arc<dyn Engine>, config: ServerConfig) -> Self {
        let mut metrics = metrics::Metrics::new();
        if let Some(c) = &config.hw_counters {
            metrics.add_hw_counters(c.clone());
        }
        let metrics = Arc::new(metrics);
        let stats = Arc::new(stats::StatsHub::new(config.hw_counters.clone()));
        Self {
            // Every route sees the engine through the metering wrapper.
            engine: Arc::new(
                metrics::MeteredEngine::new(engine, metrics.clone()).with_stats(stats.clone()),
            ),
            config: Arc::new(config),
            in_flight: Default::default(),
            metrics,
            stats,
        }
    }

    /// Maps the requested model name to an engine model id, and the name responses echo. A
    /// served-model-name alias resolves to the engine's (first) model and is echoed back as
    /// requested. With a single model loaded, any name resolves to it, so clients with
    /// hard-coded model names (e.g. Claude Code) work.
    /// The sampling defaults of the model `requested` resolves to: the operator's over the
    /// model's over OpenAI's (see [`crate::engine::sampling_defaults`]).
    pub fn sampling_defaults(
        &self,
        requested: &str,
    ) -> Result<crate::engine::sampling_defaults::SamplingDefaults, EngineError> {
        let id = self.resolve_model(requested)?.engine_id;
        let model = self
            .engine
            .models()
            .into_iter()
            .find(|m| m.id == id)
            .map(|m| m.sampling)
            .unwrap_or_default();
        Ok(crate::engine::sampling_defaults::SamplingDefaults::resolve(
            self.config.sampling_mode,
            &self.config.sampling_overrides,
            &model,
        ))
    }

    pub fn resolve_model(&self, requested: &str) -> Result<ResolvedModel, EngineError> {
        let models = self.engine.models();
        let aliased = !self.config.served_model_names.is_empty();
        let resolved = |engine_id: &str| ResolvedModel {
            engine_id: engine_id.to_string(),
            served_name: if aliased { requested } else { engine_id }.to_string(),
        };
        if models.iter().any(|m| m.id == requested) {
            return Ok(resolved(requested));
        }
        if let Some(first) = models.first()
            && self
                .config
                .served_model_names
                .iter()
                .any(|n| n == requested)
        {
            return Ok(resolved(&first.id));
        }
        match models.as_slice() {
            [only] => Ok(resolved(&only.id)),
            _ => Err(EngineError::ModelNotFound(requested.to_string())),
        }
    }
}

/// The outcome of [`AppState::resolve_model`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// The engine's model id, used to run the request.
    pub engine_id: String,
    /// The name to put in the response's `model` field.
    pub served_name: String,
}

pub fn router(state: AppState) -> Router {
    middleware::apply(routes::router(&state).with_state(state.clone()), &state)
}

pub async fn serve(addr: SocketAddr, state: AppState) -> anyhow::Result<()> {
    if let Some(config) = state.config.tls.clone() {
        return tls::serve(addr, state, config).await;
    }
    let listener = listener::bind(addr).await?;
    let local = axum::serve::Listener::local_addr(&listener)?;
    tracing::info!("listening on http://{}", local);
    registry::start(&state.config, local);
    let timeout = Duration::from_secs(state.config.shutdown_timeout_secs);
    let in_flight = state.in_flight.clone();
    let (signalled_tx, signalled_rx) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, router(state)).with_graceful_shutdown(async move {
        shutdown_signal().await;
        tracing::info!("shutting down, {} requests in flight", in_flight.count());
        let _ = signalled_tx.send(());
    });
    tokio::select! {
        result = server => result?,
        // Graceful shutdown waits for every request; past the timeout, stop waiting.
        _ = async {
            if signalled_rx.await.is_ok() {
                tokio::time::sleep(timeout).await;
            } else {
                std::future::pending::<()>().await;
            }
        } => tracing::warn!("shutdown timeout reached, dropping remaining requests"),
    }
    Ok(())
}

/// Resolves on SIGINT or SIGTERM.
pub(super) async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}
