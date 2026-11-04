//! Counts requests that are still being served, so a shutdown can log how many it is
//! waiting for. A request counts from the moment it enters until its response body has been
//! fully sent or dropped, so a streaming response stays counted for the whole stream.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use futures::StreamExt;

use super::AppState;

/// Shared counter of requests in flight.
#[derive(Clone, Default)]
pub struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    pub fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    fn enter(&self) -> Guard {
        self.0.fetch_add(1, Ordering::SeqCst);
        Guard(self.0.clone())
    }
}

struct Guard(Arc<AtomicUsize>);

impl Drop for Guard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

pub async fn track(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let guard = state.in_flight.enter();
    let response = next.run(req).await;
    // The guard rides inside the body stream, so it is released when the body ends or the
    // client goes away, not when the response head is returned.
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().inspect(move |_| {
        let _ = &guard;
    });
    Response::from_parts(parts, Body::from_stream(stream))
}
