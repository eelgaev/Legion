//! vLLM's `ignore_eos` and `min_tokens` on the OpenAI endpoints.
//!
//! The mock's scripted answer to `chat(..)` is four tokens long, so `ignore_eos` must pad it
//! to `max_tokens` and `min_tokens` must hold the sequence to at least that length.
//!
//! The real-model test is ignored: it needs the 4B GGUF.
//! `LEGION_TEST_QWEN35=<model.gguf> cargo test --release --test ignore_eos -- --ignored
//! --test-threads 1` runs it on the CPU; without the model file it skips itself.

mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

fn completion(extra: Value) -> Value {
    let mut body = json!({"model": "mock", "prompt": "hi"});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test]
async fn chat_ignore_eos_fills_up_to_max_tokens() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"ignore_eos": true, "max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 40, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
}

#[tokio::test]
async fn completions_ignore_eos_fills_up_to_max_tokens() {
    let (status, v) = json_of(post(
        "/v1/completions",
        completion(json!({"ignore_eos": true, "max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 40, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{v}");
}

#[tokio::test]
async fn chat_min_tokens_pads_the_answer() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"min_tokens": 30, "max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let n = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(n >= 30, "expected at least 30 tokens, got {n}: {v}");
}

#[tokio::test]
async fn min_tokens_above_max_tokens_is_a_400() {
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"min_tokens": 41, "max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["param"], "min_tokens", "{v}");

    let (status, v) = json_of(post(
        "/v1/completions",
        completion(json!({"min_tokens": 17, "max_tokens": 16})),
    ))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["param"], "min_tokens", "{v}");
}

#[tokio::test]
async fn without_the_fields_behaviour_is_unchanged() {
    // The scripted answer ("You said: hi there") is four tokens; the mock then stops.
    let (status, v) = json_of(post(
        "/v1/chat/completions",
        chat(json!({"max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 4, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");

    let (status, v) = json_of(post(
        "/v1/completions",
        completion(json!({"max_tokens": 40})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 3, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
}

/// Greedy generation with `ignore_eos` on the real model: exactly `max_tokens` tokens, even
/// though the model would have ended its turn much earlier.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Qwen3.5 GGUF (LEGION_TEST_QWEN35), runs on the CPU"]
async fn real_model_ignore_eos_reaches_max_tokens() {
    use futures::StreamExt;
    use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
    use legion::engine::{
        Engine, FinishReason, GenerateRequest, Message, Prompt, Role, SamplingParams, TokenEvent,
    };
    use tokio_util::sync::CancellationToken;

    let Ok(path) = std::env::var("LEGION_TEST_QWEN35") else {
        eprintln!("skipping: LEGION_TEST_QWEN35 is not set");
        return;
    };
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} does not exist");
        return;
    }
    let engine = LmEngine::load(LmConfig {
        model_name: Some("qwen".into()),
        ctx_size: 2048,
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    })
    .unwrap();
    let mut stream = engine
        .generate(GenerateRequest {
            model: "qwen".into(),
            prompt: Prompt::Chat {
                system: None,
                messages: vec![Message {
                    role: Some(Role::User),
                    content: "Say hello.".into(),
                    ..Default::default()
                }],
                tools: vec![],
            },
            sampling: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            max_tokens: 64,
            stop: vec![],
            ignore_eos: true,
            min_tokens: 0,
            thinking: Some(false),
            response_format: None,
            template_kwargs: Default::default(),
            cancel: CancellationToken::new(),
            scheduling: Default::default(),
        })
        .unwrap();
    let mut finished = None;
    while let Some(ev) = stream.next().await {
        if let TokenEvent::Finished { reason, usage, .. } = ev {
            finished = Some((reason, usage.completion_tokens));
        }
    }
    assert_eq!(
        finished,
        Some((FinishReason::Length, 64)),
        "ignore_eos must generate exactly max_tokens tokens"
    );
}
