pub mod chat;
pub mod completions;
pub mod count_tokens;
pub mod health;
pub mod messages;
pub mod metrics;
pub mod models;
pub mod responses;
pub mod stats;
pub mod webui;

use axum::Router;
use axum::response::sse::Event;
use axum::routing::{get, post};
use serde::Serialize;

use super::AppState;

/// Every route, one line each. A new endpoint gets its own file in this directory.
pub fn router(state: &AppState) -> Router<AppState> {
    let router = Router::new()
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::metrics))
        .route("/v1/models", get(models::list))
        // llama.cpp's alias; LLM proxies (simple-llm-proxy) discover models here.
        .route("/models", get(models::list))
        .route("/v1/models/{id}", get(models::get))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/messages", post(messages::messages))
        .route(
            "/v1/messages/count_tokens",
            post(count_tokens::count_tokens),
        )
        .route("/v1/responses", post(responses::responses))
        // The live dashboard (`legion top`, #1198).
        .route("/v1/stats", get(stats::stats))
        .route("/v1/stats/stream", get(stats::stats_stream))
        .route("/v1/logs/stream", get(stats::logs_stream));
    // Optional route groups, mounted by config.
    if state.config.webui {
        router.merge(webui::router())
    } else {
        router
    }
}

pub(crate) fn json_event(value: &impl Serialize) -> Event {
    Event::default()
        .json_data(value)
        .expect("API types always serialize")
}
