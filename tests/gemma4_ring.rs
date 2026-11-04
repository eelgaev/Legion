//! Sliding-window rings (#297) through the engine on the CPU, with the tiny random `gemma4`
//! (window 8, shared-KV layers) and reference kernels: paged attention over rings gives the
//! greedy output of paged attention over every position (`swa_full`), for prompts and
//! generations many times longer than the ring, batched, with drafts rolled back (n-gram
//! drafts, mostly wrong) and with the MTP assistant reading the target's rings.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::core::Attention;
use legion::model::gemma4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

const MAX_TOKENS: u32 = 150;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(8),
        "0123456789".repeat(21),
        "sliding windows".into(),
        "Ring buffers keep the last few positions only; ".repeat(5),
    ]
}

/// The tiny model's engine with `attention`, the sliding-window layers in rings unless
/// `swa_full` (with gather attention too). Prompt chunks of 16 tokens: rings of 2 blocks (128
/// positions).
fn engine(
    path: &std::path::Path,
    spec: SpecConfig,
    attention: Attention,
    swa_full: bool,
) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        ring_step: Some(16),
        prefix_cache_states: 0,
        attention,
        swa_full,
        ring_gather: true,
        spec,
        use_ref: true,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    })
    .unwrap()
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

/// Every prompt at once (more than `max_seqs`, so sequences also wait and reuse freed rings);
/// (text, finish reason, tokens) in prompt order.
async fn run_all(engine: &LmEngine) -> Vec<(String, FinishReason, u32)> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| engine.generate(request(p)).unwrap())
        .collect();
    let mut replies = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended without Finished") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { reason, usage, .. } => {
                    replies.push((text, reason, usage.completion_tokens));
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
    }
    replies
}

#[tokio::test(flavor = "multi_thread")]
async fn rings_give_the_output_of_the_full_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let assistant = dir.path().join("assistant.gguf");
    tiny::write_assistant(&assistant, 7).unwrap();
    let specs = [
        ("plain", SpecConfig::default()),
        (
            "n-gram drafts",
            SpecConfig {
                kind: SpecKind::Ngram,
                k: 4,
                fixed: true,
                ..Default::default()
            },
        ),
        (
            "mtp",
            SpecConfig {
                kind: SpecKind::Mtp,
                k: 4,
                fixed: true,
                p_min: 0.0,
                draft: Some(assistant),
                ..Default::default()
            },
        ),
    ];
    // Paged attention reads the rings in place; gather attention (tensor parallelism's path)
    // gathers through them, the window masking the entries newer positions took over.
    for attention in [Attention::Paged, Attention::Gather] {
        for (name, spec) in &specs {
            let full = run_all(&engine(&path, spec.clone(), attention, true)).await;
            let ring = engine(&path, spec.clone(), attention, false);
            for round in 0..2 {
                let got = run_all(&ring).await;
                assert!(
                    got.iter().any(|r| r.2 == MAX_TOKENS),
                    "{name}: generations too short"
                );
                for (i, (a, b)) in full.iter().zip(&got).enumerate() {
                    assert_eq!(
                        a, b,
                        "{attention:?}, {name}, round {round}: prompt {i} differs"
                    );
                }
            }
        }
    }
}

/// A prompt pipelined over two stages in micro-chunks (`pipeline_micro`), alone in the batch
/// but in the second ring (the first sequence took ring 0 and finished): the micro-chunks
/// read and write the sequence's own ring, so its output is the full cache's. Alone, the prompt
/// runs as one long step of micro-chunks (`pipeline_prompt`, past `max_batch_tokens` and the
/// ring step) or, with that off, as steps of a chunk: the same output.
#[tokio::test(flavor = "multi_thread")]
async fn micro_chunks_keep_their_sequence_ring() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    // Per-layer embeddings run as one stage only.
    tiny::write_without_ple(&path, 3).unwrap();
    let staged = |swa_full: bool, pipeline_prompt: usize| {
        LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 40,
            prefill_chunk: 16,
            ring_step: Some(16),
            pipeline_micro: 8,
            pipeline_prompt,
            // Layers 2.. share caches: the split goes before them.
            layer_split: if swa_full { Vec::new() } else { vec![2] },
            prefix_cache_states: 0,
            attention: Attention::Paged,
            swa_full,
            use_ref: true,
            ..LmConfig::new(&path, DeviceChoice::Cpu)
        })
        .unwrap()
    };
    let long = "Ring buffers keep the last few positions only; ".repeat(6);
    let run = |engine: LmEngine| {
        let long = long.clone();
        async move {
            let mut short = request("abc");
            short.max_tokens = 2;
            let first = engine.generate(short).unwrap();
            let second = engine.generate(request(&long)).unwrap();
            let collect = |mut s: legion::engine::TokenStream| async move {
                let mut text = String::new();
                loop {
                    match s.next().await.expect("stream ended without Finished") {
                        TokenEvent::Text(t) => text += &t,
                        TokenEvent::Finished { .. } => return text,
                        TokenEvent::Error(e) => panic!("engine error: {e}"),
                        _ => {}
                    }
                }
            };
            let a = collect(first).await;
            (a, collect(second).await)
        }
    };
    let want = run(staged(true, 0)).await;
    for pipeline_prompt in [0, 4096] {
        let got = run(staged(false, pipeline_prompt)).await;
        assert_eq!(got, want, "pipeline_prompt {pipeline_prompt}");
    }
}
