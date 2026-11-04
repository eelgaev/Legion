//! `--tp 2 --tp-per-gpu` (#691) through the engine on the CPU: the two ranks of one process
//! (rank 1 on a thread of its own, the partial sums added through the pair all-reduce's CPU
//! twin) give the dense tiny `gemma4`'s greedy output of one rank, batched, with warm
//! sequences, on a second run, and with `--spec mtp` drafting with a split assistant.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::gemma4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

const MAX_TOKENS: u32 = 60;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4),
        "0123456789".repeat(9),
        "tensor parallel".into(),
    ]
}

fn config(path: &std::path::Path, per_gpu: bool, spec: &SpecConfig) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        ring_step: Some(16),
        use_ref: true,
        tp: if per_gpu { 2 } else { 1 },
        tp_per_gpu: per_gpu,
        spec: spec.clone(),
        ..LmConfig::new(path, DeviceChoice::Cpu)
    }
}

fn request(prompt: &str) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Text(prompt.into()),
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens: MAX_TOKENS,
        stop: vec![],
        ignore_eos: true,
        min_tokens: 0,
        thinking: None,
        template_kwargs: Default::default(),
        response_format: None,
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

/// Every prompt at once (more than `max_seqs`), then the first again (a warm sequence).
async fn run_all(engine: &LmEngine) -> Vec<(String, FinishReason, u32)> {
    let ps = prompts();
    let streams: Vec<_> = ps
        .iter()
        .map(|p| engine.generate(request(p)).unwrap())
        .collect();
    let mut replies = Vec::new();
    for mut s in streams {
        replies.push(collect(&mut s).await);
    }
    let mut s = engine.generate(request(&ps[0])).unwrap();
    replies.push(collect(&mut s).await);
    replies
}

async fn collect(
    s: &mut (impl futures::Stream<Item = TokenEvent> + Unpin),
) -> (String, FinishReason, u32) {
    let mut text = String::new();
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { reason, usage, .. } => {
                return (text, reason, usage.completion_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

async fn load(c: LmConfig) -> LmEngine {
    tokio::task::spawn_blocking(move || LmEngine::load(c))
        .await
        .unwrap()
        .unwrap()
}

/// The output of one rank, then of the pair (twice) must be the same, with `spec`.
async fn pair_matches_one(path: &std::path::Path, spec: SpecConfig) {
    let one = run_all(&load(config(path, false, &spec)).await).await;
    assert!(
        one.iter().all(|r| r.2 == MAX_TOKENS),
        "generations too short"
    );
    let pair = load(config(path, true, &spec)).await;
    for round in 0..2 {
        let two = run_all(&pair).await;
        for (i, (a, b)) in one.iter().zip(&two).enumerate() {
            assert_eq!(
                a, b,
                "round {round}, prompt {i}: the pair differs from one rank"
            );
        }
    }
    if spec.enabled() {
        assert!(pair.stats().spec.drafted > 0, "rank 0 drafted nothing");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_pair_gives_the_output_of_one_rank() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    pair_matches_one(&path, SpecConfig::default()).await;
}

/// `--spec mtp`: the assistant split like the target, each rank's head over its own share of
/// the cache.
#[tokio::test(flavor = "multi_thread")]
async fn the_pair_drafts_with_a_split_assistant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    let assistant = dir.path().join("assistant.gguf");
    tiny::write_assistant(&assistant, 5).unwrap();
    let spec = SpecConfig {
        kind: SpecKind::Mtp,
        k: 3,
        fixed: true,
        p_min: 0.0,
        draft: Some(assistant),
        ..Default::default()
    };
    pair_matches_one(&path, spec).await;
}

/// What the pair cannot do is refused before either rank loads.
#[test]
fn other_layouts_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    let spec = SpecConfig::default();
    for (c, want) in [
        (
            LmConfig {
                tp: 4,
                ..config(&path, true, &spec)
            },
            "needs --tp 2",
        ),
        (
            LmConfig {
                gpus: 2,
                ..config(&path, true, &spec)
            },
            "one GPU per rank",
        ),
    ] {
        let e = LmEngine::load(c).err().expect("refused").to_string();
        assert!(e.contains(want), "{e}");
    }
}
