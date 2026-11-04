//! `strict` on function tools reaches the engine request (chat and responses), only when true.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::engine::mock::{MockConfig, MockEngine};

fn state() -> AppState {
    AppState::new(Arc::new(MockEngine::new(MockConfig::default())))
}

fn tools(req: &GenerateRequest) -> Vec<(String, bool)> {
    match &req.prompt {
        Prompt::Chat { tools, .. } => tools.iter().map(|t| (t.name.clone(), t.strict)).collect(),
        Prompt::Text(_) => panic!("not a chat"),
    }
}

fn expected() -> Vec<(String, bool)> {
    vec![("a".into(), true), ("b".into(), false), ("c".into(), false)]
}

#[tokio::test]
async fn chat_tools_keep_strict() {
    let function = |name: &str, strict: Value| {
        let mut f = json!({"name": name, "parameters": {"type": "object"}});
        if !strict.is_null() {
            f["strict"] = strict;
        }
        json!({"type": "function", "function": f})
    };
    let req: openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [function("a", json!(true)), function("b", json!(false)), function("c", Value::Null)]
    }))
    .unwrap();
    assert_eq!(tools(&chat_request(&state(), &req).unwrap()), expected());
}

#[tokio::test]
async fn responses_tools_keep_strict() {
    let req: openai::ResponsesRequest = serde_json::from_value(json!({
        "model": "mock",
        "input": "hi",
        "tools": [
            {"type": "function", "name": "a", "strict": true},
            {"type": "function", "name": "b", "strict": false},
            {"type": "function", "name": "c"}
        ]
    }))
    .unwrap();
    assert_eq!(
        tools(&responses_request(&state(), &req).unwrap()),
        expected()
    );
}
