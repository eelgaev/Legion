//! Sequences' text on the stream thread (`LmConfig::stream_thread_seqs`, #47): with several
//! requests decoding at once, every stream must carry exactly the events it carries when the
//! engine thread handles all the text, in the same order: greedy text, log probabilities,
//! stop strings and grammars (whose sequences stay on the engine thread) and the finish
//! reasons.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, FinishReason, GenerateRequest, Prompt, ResponseFormat, SamplingParams, TokenEvent,
    TokenLogprob,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

fn engine(path: &std::path::Path, stream_thread_seqs: usize) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 6,
        stream_thread_seqs,
        ..LmConfig::new(path.to_path_buf(), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// What a stream carries, without the timings in `Finished`.
#[derive(Debug, PartialEq)]
enum Event {
    Text(String),
    Logprobs(Vec<TokenLogprob>),
    Finished(FinishReason, Option<String>, u32),
}

/// Eight requests over six slots, of different lengths, some with log probabilities, one
/// with a stop string the greedy text reaches, one with a JSON schema (a grammar).
async fn streams(e: &LmEngine, stop: &str) -> Vec<Vec<Event>> {
    let prompts = [
        ("Many users at once: ".repeat(4), 40, None, false, false),
        ("speculative".into(), 33, Some(2), false, false),
        ("routing 0123456789".repeat(3), 21, None, true, false),
        ("abc".into(), 28, Some(0), false, false),
        ("stream thread ".repeat(2), 36, None, false, false),
        ("groups in flight".into(), 17, Some(3), false, false),
        ("the last one".into(), 25, None, false, false),
        ("JSON:".into(), 64, None, false, true),
    ];
    let streams: Vec<_> = prompts
        .iter()
        .map(|(p, max_tokens, logprobs, with_stop, format)| {
            e.generate(GenerateRequest {
                model: "tiny".into(),
                prompt: Prompt::Text(p.clone()),
                sampling: SamplingParams {
                    temperature: 0.0,
                    logprobs: *logprobs,
                    ..Default::default()
                },
                max_tokens: *max_tokens,
                stop: if *with_stop {
                    vec![stop.to_string()]
                } else {
                    vec![]
                },
                ignore_eos: true,
                min_tokens: 0,
                thinking: None,
                response_format: format
                    .then(|| ResponseFormat::JsonSchema(json!({"const": {"ok": true}}))),
                template_kwargs: Default::default(),
                cancel: CancellationToken::new(),
                scheduling: Default::default(),
            })
            .unwrap()
        })
        .collect();
    let mut out = Vec::new();
    for mut s in streams {
        let mut events = Vec::new();
        loop {
            match s.next().await.expect("stream ended") {
                TokenEvent::Text(t) => events.push(Event::Text(t)),
                TokenEvent::Logprobs(l) => events.push(Event::Logprobs(l)),
                TokenEvent::Finished {
                    reason,
                    stop_sequence,
                    usage,
                } => {
                    events.push(Event::Finished(
                        reason,
                        stop_sequence,
                        usage.completion_tokens,
                    ));
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
        assert!(s.next().await.is_none(), "events after Finished");
        out.push(events);
    }
    out
}

fn text(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_stream_thread_streams_the_same_events() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    legion::model::qwen4::tiny::write(&path, None, 11).unwrap();
    // A stop string the third request's greedy text reaches: a few of its middle characters.
    let probe = streams(&engine(&path, 0), "\u{0}never").await;
    let third = text(&probe[2]);
    let chars: Vec<char> = third.chars().collect();
    assert!(chars.len() > 8, "the third answer is too short: {third:?}");
    let stop: String = chars[4..7].iter().collect();

    let here = streams(&engine(&path, 0), &stop).await;
    assert!(matches!(
        here[2].last(),
        Some(Event::Finished(FinishReason::StopSequence, Some(_), _))
    ));
    assert!(
        here.iter()
            .flatten()
            .any(|e| matches!(e, Event::Logprobs(_)))
    );
    // The grammar decided the JSON request's text.
    assert_eq!(text(&here[7]), r#"{"ok":true}"#);
    // 1: every sequence's text away but the stop string's and the grammar's; 3: from the
    // third live one on.
    for seqs in [1, 3] {
        let away = streams(&engine(&path, seqs), &stop).await;
        assert_eq!(here, away, "--stream-thread-seqs {seqs}");
    }
}
