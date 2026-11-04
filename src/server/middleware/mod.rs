//! HTTP middleware. Each middleware lives in its own file in this directory and is added to
//! [`apply`] with one line.

use axum::Router;
use axum::middleware::{from_fn, from_fn_with_state};

use super::AppState;

pub use in_flight::InFlight;

mod auth;
mod body_limit;
mod cors;
mod in_flight;
mod metrics;
mod request_id;

/// Wraps the router in all middleware layers. Layers added later run first (outermost).
pub fn apply(router: Router, state: &AppState) -> Router {
    let router = body_limit::apply(router, state);
    let router = router.layer(from_fn_with_state(state.clone(), auth::authenticate));
    // Outside auth: a browser preflight never carries an API key, so CORS answers it first.
    let router = cors::apply(router, state);
    // Outermost: every response, including auth rejections, carries an id and is logged.
    let router = router.layer(from_fn(request_id::request_id));
    // Counts requests by route and status, auth rejections included.
    let router = router.layer(from_fn_with_state(state.clone(), metrics::record));
    // Very outermost: counts every request, including auth rejections, until its body ends.
    router.layer(from_fn_with_state(state.clone(), in_flight::track))
}
