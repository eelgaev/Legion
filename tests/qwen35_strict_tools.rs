//! Strict tool calls (`tools[].function.strict`) end to end on the tiny CPU model, whose
//! vocabulary has Qwen's `<tool_call>` / `</tool_call>` markup tokens: calls follow the tool's
//! schema and parse, text outside calls is what it would be without `strict`, and drafts never
//! run into a call.

use std::sync::Arc;

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, FinishReason, GenerateRequest, Message, Prompt, Role, SamplingParams, TokenEvent,
    TokenStream, Tool,
};
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::{DraftRequest, ModelDrafter, SpecConfig};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}\
                        <|im_end|>\n{% endfor %}<|im_start|>assistant\n";

fn model(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("tiny-tools.gguf");
    tiny::write(
        &path,
        &TinyConfig {
            user_tokens: &["<tool_call>", "</tool_call>"],
            chat_template: Some(TEMPLATE),
            ..Default::default()
        },
    )
    .unwrap();
    path
}

fn engine(path: &std::path::Path, spec: SpecConfig, grammar_threads: usize) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 4096,
        threads: 2,
        max_seqs: 3,
        prefix_cache_states: 0,
        use_ref: true,
        spec,
        grammar_threads,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    })
    .unwrap()
}

fn weather(strict: bool) -> Tool {
    Tool {
        name: "get_weather".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer"},
                "alerts": {"type": "boolean"}
            },
            "required": ["unit", "days", "alerts"],
            "additionalProperties": false
        }),
        strict,
    }
}

fn request(strict: bool, temperature: f32, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Chat {
            system: None,
            messages: vec![Message {
                role: Some(Role::User),
                content: "Weather in Oslo?".into(),
                ..Default::default()
            }],
            tools: vec![weather(strict)],
        },
        sampling: SamplingParams {
            temperature,
            seed: Some(7),
            ..Default::default()
        },
        max_tokens,
        stop: vec![],
        // Long enough for the random model to open calls.
        ignore_eos: true,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

/// (text before the first call, every call's name and arguments, all text, finish reason)
async fn collect(mut stream: TokenStream) -> (String, Vec<(String, Value)>, String, FinishReason) {
    let (mut before, mut calls, mut text) = (None, Vec::new(), String::new());
    while let Some(event) = stream.next().await {
        match event {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::ToolCall(c) => {
                before.get_or_insert_with(|| text.clone());
                let args = serde_json::from_str(&c.arguments)
                    .unwrap_or_else(|e| panic!("arguments {:?}: {e}", c.arguments));
                calls.push((c.name, args));
            }
            TokenEvent::Finished { reason, .. } => {
                return (before.unwrap_or_else(|| text.clone()), calls, text, reason);
            }
            TokenEvent::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    panic!("missing Finished")
}

#[tokio::test]
async fn strict_calls_follow_the_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path());
    for threads in [0, 2] {
        let engine = engine(&path, SpecConfig::default(), threads);
        let (_, calls, text, reason) =
            collect(engine.generate(request(true, 1.0, 1500)).unwrap()).await;
        assert!(!calls.is_empty(), "no call in {text:?}");
        for (name, args) in &calls {
            assert_eq!(name, "get_weather");
            let args = args.as_object().unwrap();
            assert_eq!(args.len(), 3, "{args:?}");
            assert!(["c", "f"].contains(&args["unit"].as_str().unwrap()));
            assert_eq!(args["days"].as_f64().unwrap().fract(), 0.0, "{args:?}");
            assert!(args["alerts"].is_boolean(), "{args:?}");
        }
        // No call markup left in the text: every call that opened was completed and parsed
        // (but one cut by max_tokens).
        assert!(
            text.matches("<tool_call>").count() <= usize::from(reason == FinishReason::Length),
            "{text:?}"
        );
    }
}

/// Up to its first call, a request with strict tools samples exactly what it would without
/// `strict`: the grammar only applies inside calls.
#[tokio::test]
async fn text_outside_calls_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path());
    let engine = engine(&path, SpecConfig::default(), 2);
    for temperature in [0.0, 1.0] {
        let (before, calls, _, _) =
            collect(engine.generate(request(true, temperature, 600)).unwrap()).await;
        let (_, _, free, _) =
            collect(engine.generate(request(false, temperature, 600)).unwrap()).await;
        if calls.is_empty() {
            assert_eq!(before, free);
        } else {
            assert!(
                free.starts_with(&format!("{before}<tool_call>")),
                "{before:?} vs {free:?}"
            );
        }
    }
}

/// Drafts the call's open token, always.
struct Opener(u32);
impl ModelDrafter for Opener {
    fn hidden_layers(&self) -> Vec<usize> {
        vec![3]
    }
    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        for req in batch {
            req.out.tokens = vec![self.0; req.max];
        }
        Ok(())
    }
}

#[tokio::test]
async fn drafts_stop_before_a_call() {
    let dir = tempfile::tempdir().unwrap();
    let path = model(dir.path());
    // The first user-defined token, after the bytes and the three specials.
    let open = 256 + tiny::SPECIAL_TOKENS.len() as u32;
    let spec = SpecConfig {
        k: 3,
        model_drafter: Some(Arc::new(move |_| Ok(Box::new(Opener(open))))),
        ..Default::default()
    };
    let engine = engine(&path, spec, 2);
    let (_, _, _, reason) = collect(engine.generate(request(true, 1.0, 200)).unwrap()).await;
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(engine.stats().spec.drafted, 0);
    // Without strict tools the same drafts are verified.
    collect(engine.generate(request(false, 1.0, 50)).unwrap()).await;
    assert!(engine.stats().spec.drafted > 0);
}
