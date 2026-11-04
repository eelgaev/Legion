//! Counts every HTTP request by matched route and status class, for `/metrics`. A streaming
//! response is counted when its head is returned.

use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;

use crate::server::AppState;

pub async fn record(State(state): State<AppState>, req: Request, next: Next) -> Response {
    // Unmatched paths share one label so scanners cannot grow the series count.
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map_or("unmatched", |p| p.as_str())
        .to_string();
    let response = next.run(req).await;
    state
        .metrics
        .record_request(&route, response.status().as_u16());
    response
}
