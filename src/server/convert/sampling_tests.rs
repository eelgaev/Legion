//! The sampling defaults on every front end: request fields > operator flags > the model's
//! declared defaults > OpenAI's (see `crate::engine::sampling_defaults`).

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::engine::mock::{MockConfig, MockEngine};
use crate::engine::sampling_defaults::{DefaultsMode, ModelSampling, SamplingLayer};
use crate::server::ServerConfig;

/// A model declaring temperature 0.7, top_k 20, top_p 0.9; the operator sets top_p 0.8.
fn state(mode: DefaultsMode) -> AppState {
    let engine = MockEngine::new(MockConfig {
        sampling: ModelSampling {
            file: SamplingLayer {
                temperature: Some(0.7),
                top_k: Some(20),
                top_p: Some(0.9),
                ..Default::default()
            },
            architecture: SamplingLayer::for_architecture("gemma4"),
        },
        ..Default::default()
    });
    AppState::with_config(
        Arc::new(engine),
        ServerConfig {
            sampling_mode: mode,
            sampling_overrides: SamplingLayer {
                top_p: Some(0.8),
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

/// The sampling params each endpoint makes of a request with `extra` fields.
fn all_endpoints(state: &AppState, extra: Value) -> Vec<(&'static str, SamplingParams)> {
    let with = |mut base: Value| {
        for (k, v) in extra.as_object().unwrap() {
            base[k] = v.clone();
        }
        base
    };
    let chat: openai::ChatCompletionRequest = serde_json::from_value(with(json!({
        "model": "mock", "messages": [{"role": "user", "content": "hi"}]
    })))
    .unwrap();
    let completion: openai::CompletionRequest =
        serde_json::from_value(with(json!({"model": "mock", "prompt": "hi"}))).unwrap();
    let messages: anthropic::MessagesRequest = serde_json::from_value(with(json!({
        "model": "mock", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]
    })))
    .unwrap();
    let responses: openai::ResponsesRequest =
        serde_json::from_value(with(json!({"model": "mock", "input": "hi"}))).unwrap();
    vec![
        ("chat", chat_request(state, &chat).unwrap().sampling),
        (
            "completions",
            completion_request(state, &completion).unwrap().sampling,
        ),
        (
            "messages",
            messages_request(state, &messages).unwrap().sampling,
        ),
        (
            "responses",
            responses_request(state, &responses).unwrap().sampling,
        ),
    ]
}

#[tokio::test]
async fn omitted_fields_take_the_model_defaults_under_the_operator() {
    for (api, s) in all_endpoints(&state(DefaultsMode::Model), json!({})) {
        assert_eq!(
            (s.temperature, s.top_k, s.top_p, s.min_p),
            (0.7, 20, 0.8, 0.0),
            "{api}"
        );
        assert_eq!(s.repetition_penalty, 1.0, "{api}");
    }
}

#[tokio::test]
async fn request_fields_win_and_temperature_zero_stays_greedy() {
    let req = json!({"temperature": 0.0, "top_p": 0.5, "top_k": 3});
    for (api, s) in all_endpoints(&state(DefaultsMode::Model), req) {
        assert_eq!(s.temperature, 0.0, "{api}: greedy");
        assert_eq!(s.top_p, 0.5, "{api}");
        // the Responses API has no top_k: the model's
        let k = if api == "responses" { 20 } else { 3 };
        assert_eq!(s.top_k, k, "{api}");
    }
}

#[tokio::test]
async fn openai_mode_ignores_the_model() {
    for (api, s) in all_endpoints(&state(DefaultsMode::Openai), json!({})) {
        assert_eq!(
            (s.temperature, s.top_k, s.top_p),
            (1.0, 0, 0.8),
            "{api}: OpenAI's, top_p the operator's"
        );
    }
}
