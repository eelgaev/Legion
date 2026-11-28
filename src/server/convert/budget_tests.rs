//! A request's thinking budget on each front end: `thinking_budget_tokens` (or Strata's
//! `reasoning_budget_tokens`) on chat completions, `thinking.budget_tokens` on Anthropic
//! messages, carried to the engine under `BUDGET_KWARG` and nowhere else.

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::engine::mock::{MockConfig, MockEngine};

fn state() -> AppState {
    AppState::new(Arc::new(MockEngine::new(MockConfig::default())))
}

fn chat(extra: serde_json::Value) -> GenerateRequest {
    let mut v = json!({"model": "mock", "messages": [{"role": "user", "content": "hi"}]});
    v.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let req: openai::ChatCompletionRequest = serde_json::from_value(v).unwrap();
    chat_request(&state(), &req).unwrap()
}

fn messages(thinking: Option<serde_json::Value>) -> GenerateRequest {
    let mut v = json!({
        "model": "mock",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    });
    if let Some(t) = thinking {
        v["thinking"] = t;
    }
    let req: anthropic::MessagesRequest = serde_json::from_value(v).unwrap();
    messages_request(&state(), &req).unwrap()
}

#[tokio::test]
async fn chat_budget_fields_reach_the_engine() {
    for field in ["thinking_budget_tokens", "reasoning_budget_tokens"] {
        let mut extra = serde_json::Map::new();
        extra.insert(field.into(), 64.into());
        let r = chat(serde_json::Value::Object(extra));
        assert_eq!(r.template_kwargs[BUDGET_KWARG], 64, "{field}");
        // A budget alone does not switch thinking on or off.
        assert_eq!(r.thinking, None);
    }
    let r = chat(json!({"thinking_budget_tokens": 0, "reasoning_effort": "high"}));
    assert_eq!(r.template_kwargs[BUDGET_KWARG], 0);
    assert_eq!(r.thinking, Some(true));
}

#[tokio::test]
async fn no_budget_sends_nothing() {
    assert!(chat(json!({})).template_kwargs.is_empty());
    let r = chat(json!({"reasoning_effort": "low"}));
    assert!(!r.template_kwargs.contains_key(BUDGET_KWARG));
    assert!(messages(None).template_kwargs.is_empty());
}

#[tokio::test]
async fn anthropic_budget_tokens_count_only_when_thinking_is_enabled() {
    let r = messages(Some(json!({"type": "enabled", "budget_tokens": 2048})));
    assert_eq!(r.template_kwargs[BUDGET_KWARG], 2048);
    assert_eq!(r.thinking, Some(true));
    for t in [
        json!({"type": "disabled", "budget_tokens": 2048}),
        json!({"type": "enabled"}),
        json!({"type": "adaptive"}),
    ] {
        let r = messages(Some(t.clone()));
        assert!(r.template_kwargs.is_empty(), "{t}");
    }
}
