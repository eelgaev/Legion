//! `GET /metrics`: Prometheus text exposition format.

use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::response::IntoResponse;

use crate::server::AppState;

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.metrics.set_replicas(&state.engine.replica_stats());
    let body = state
        .metrics
        .render(state.in_flight.count(), state.engine.stats());
    (
        [(
            CONTENT_TYPE,
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
        )],
        body,
    )
}
