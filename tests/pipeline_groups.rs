//! Decoding groups in flight over the pipeline stages (`LmConfig::pipeline_groups`, #415):
//! with the layers split over CPU stages (each its own backend, a GPU each in production),
//! greedy answers must be exactly those of whole steps, with continuous batching (groups
//! draining for prompts and regrouping as sequences finish), MTP verify steps and a swapping
//! expert cache.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::residency::policy::SwapRule;
use legion::residency::{ExpertMode, ResidencyConfig};
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

fn write_models(dir: &std::path::Path) {
    legion::model::qwen4::tiny::write(dir.join("tiny.gguf"), Some(&dir.join("mtp.gguf")), 11)
        .unwrap();
}

fn engine(
    dir: &std::path::Path,
    residency: ResidencyConfig,
    k: usize,
    layer_split: &[usize],
    pipeline_groups: usize,
    pipeline_min_group: usize,
) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 6,
        max_batch_tokens: 64,
        spec: SpecConfig {
            kind: if k > 0 { SpecKind::Mtp } else { SpecKind::None },
            k,
            draft: Some(dir.join("mtp.gguf")),
            p_min: 0.0,
            ..Default::default()
        },
        use_ref: true,
        residency,
        layer_split: layer_split.to_vec(),
        prefill_chunk: 40,
        pipeline_micro: 16,
        pipeline_groups,
        pipeline_min_group,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// Seven requests over six sequence slots, of different lengths: sequences finish one by
/// one, inside groups, and later prompts join (the groups drain for them).
async fn answers(e: &LmEngine) -> Vec<String> {
    let prompts = [
        ("Many users at once: ".repeat(4), 40),
        ("speculative".into(), 33),
        ("routing 0123456789".repeat(3), 21),
        ("abc".into(), 28),
        ("pipeline stages ".repeat(2), 36),
        ("groups in flight".into(), 17),
        ("the last one".into(), 25),
    ];
    let streams: Vec<_> = prompts
        .iter()
        .map(|(p, max_tokens)| {
            e.generate(GenerateRequest {
                model: "tiny".into(),
                prompt: Prompt::Text(p.clone()),
                sampling: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                max_tokens: *max_tokens,
                stop: vec![],
                ignore_eos: true,
                min_tokens: 0,
                thinking: None,
                response_format: None,
                template_kwargs: Default::default(),
                cancel: CancellationToken::new(),
                scheduling: Default::default(),
            })
            .unwrap()
        })
        .collect();
    let mut out = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { .. } => break,
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
        out.push(text);
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn pipeline_groups_answer_the_same() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let cached = ResidencyConfig {
        mode: ExpertMode::Cache,
        cache_bytes: Some(5 * 2 * 6528),
        swaps: SwapRule {
            budget: 4,
            margin: 0.0,
            ratio: 0.0,
        },
        ..Default::default()
    };
    for k in [0, 3] {
        for residency in [ResidencyConfig::default(), cached.clone()] {
            let whole2 = answers(&engine(dir.path(), residency.clone(), k, &[2], 0, 2)).await;
            let whole13 = answers(&engine(dir.path(), residency.clone(), k, &[1, 3], 0, 2)).await;
            // Groups of one sequence (`--pipeline-min-group 1`, #752) too.
            for (split, groups, min) in [
                (&[2][..], 2, 2),
                (&[2][..], 3, 2),
                (&[1, 3][..], 3, 2),
                (&[2][..], 2, 1),
                (&[1, 3][..], 3, 1),
            ] {
                let whole = if split.len() == 1 { &whole2 } else { &whole13 };
                let piped = answers(&engine(
                    dir.path(),
                    residency.clone(),
                    k,
                    split,
                    groups,
                    min,
                ))
                .await;
                assert_eq!(
                    whole, &piped,
                    "spec k = {k}, {:?}, split {split:?}, {groups} groups of at least {min}",
                    residency.mode
                );
                assert!(whole.iter().all(|t| !t.is_empty()));
            }
        }
    }
}
