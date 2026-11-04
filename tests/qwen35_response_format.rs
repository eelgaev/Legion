mod common;

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, FinishReason, GenerateRequest, Prompt, ResponseFormat, SamplingParams, TokenEvent,
    TokenStream,
};
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::{DraftRequest, ModelDrafter, SpecConfig};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

fn engine(path: &std::path::Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 2048,
        threads: 2,
        max_seqs: 3,
        prefix_cache_states: 0,
        use_ref: true,
        spec,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    })
    .unwrap()
}

fn request(format: ResponseFormat, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Text("JSON:".into()),
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: Some(true),
        response_format: Some(format),
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

async fn collect(mut stream: TokenStream) -> (String, FinishReason) {
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { reason, .. } => return (text, reason),
            TokenEvent::Error(e) => panic!("{e}"),
            TokenEvent::Reasoning(t) => panic!("JSON must not contain reasoning: {t}"),
            _ => {}
        }
    }
    panic!("missing Finished")
}

fn schema() -> ResponseFormat {
    ResponseFormat::JsonSchema(
        json!({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"}},"required":["a","b"]}),
    )
}

#[tokio::test]
async fn greedy_json_schema_object_length_and_shared_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let engine = engine(&path, SpecConfig::default());
    assert_eq!(engine.grammar_cache_stats(), (0, 0));
    let first = engine.generate(request(schema(), 1024)).unwrap();
    let second = engine.generate(request(schema(), 1024)).unwrap();
    assert_eq!(engine.grammar_cache_stats(), (1, 1));
    for stream in [first, second] {
        let (text, reason) = collect(stream).await;
        assert_eq!(reason, FinishReason::Stop, "{text}");
        let value: Value = serde_json::from_str(&text).unwrap();
        // JSON Schema considers -0 an integer; serde_json stores it as a float.
        assert_eq!(value["a"].as_f64().unwrap().fract(), 0.0, "{text}");
        assert!(value["b"].is_boolean(), "{text}");
    }
    let (text, reason) = collect(
        engine
            .generate(request(ResponseFormat::JsonObject, 1024))
            .unwrap(),
    )
    .await;
    assert_eq!(reason, FinishReason::Stop, "{text}");
    assert!(serde_json::from_str::<Value>(&text).unwrap().is_object());
    let (text, reason) = collect(engine.generate(request(schema(), 1)).unwrap()).await;
    assert_eq!(reason, FinishReason::Length);
    assert!(serde_json::from_str::<Value>(&text).is_err());
}

#[tokio::test]
async fn grammar_stops_at_eos_even_with_ignore_eos() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let engine = engine(&path, SpecConfig::default());
    let mut req = request(ResponseFormat::JsonSchema(json!({"const":{"ok":true}})), 64);
    req.ignore_eos = true;
    let (text, reason) = collect(engine.generate(req).unwrap()).await;
    assert_eq!(text, "{\"ok\":true}");
    assert_eq!(reason, FinishReason::Stop);
}

struct Drafter(Arc<Mutex<Vec<usize>>>);
impl ModelDrafter for Drafter {
    fn hidden_layers(&self) -> Vec<usize> {
        vec![3]
    }
    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        for req in batch {
            if req.tokens.first() == Some(&u32::from(b'J')) {
                self.0.lock().unwrap().push(req.max);
            }
            req.out.tokens = vec![u32::from(b'x'); req.max];
        }
        Ok(())
    }
}

#[tokio::test]
async fn grammar_skips_drafts_in_mixed_speculative_batches() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let caps = Arc::new(Mutex::new(Vec::new()));
    let seen = caps.clone();
    let engine = engine(
        &path,
        SpecConfig {
            k: 3,
            model_drafter: Some(Arc::new(move |_| Ok(Box::new(Drafter(seen.clone()))))),
            ..Default::default()
        },
    );
    let mut plain = request(schema(), 80);
    plain.response_format = None;
    plain.prompt = Prompt::Text("plain".into());
    plain.ignore_eos = true;
    let plain_stream = engine.generate(plain).unwrap();
    let constrained = engine
        .generate(request(
            ResponseFormat::JsonSchema(json!({"const":{"answer":42}})),
            64,
        ))
        .unwrap();
    let (text, reason) = collect(constrained).await;
    assert_eq!(text, "{\"answer\":42}");
    assert_eq!(reason, FinishReason::Stop);
    collect(plain_stream).await;
    let caps = caps.lock().unwrap();
    assert!(!caps.is_empty());
    assert!(caps.iter().all(|&cap| cap == 0), "{caps:?}");
    assert!(engine.stats().spec.drafted > 0);
}

#[test]
fn grammar_cache_is_lru_and_validation_reuses_compilation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let engine = engine(&path, SpecConfig::default());
    let format = |i| ResponseFormat::JsonSchema(json!({"const":i}));
    for i in 0..64 {
        engine.check_response_format(&format(i)).unwrap();
    }
    engine.check_response_format(&format(0)).unwrap();
    engine.check_response_format(&format(64)).unwrap();
    engine.check_response_format(&format(0)).unwrap();
    assert_eq!(engine.grammar_cache_stats(), (65, 2));
    engine.check_response_format(&format(1)).unwrap();
    assert_eq!(engine.grammar_cache_stats(), (66, 2));
}
