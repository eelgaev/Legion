//! `--itl-target` / `--ttft-target` (#628): the targets move only step boundaries (how many
//! prompt tokens each step takes, which prompt goes first). On the CPU reference kernels a
//! token's bits do not depend on the step it runs in, so every request's greedy answer must be
//! the one without targets.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, KvAdmission, LmConfig, LmEngine, SloTargets};
use legion::engine::queue::{Scheduling, SchedulingPolicy};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

/// The tiny Qwen (recurrent DeltaNet layers, saved prefix states, MTP head).
fn qwen(
    dir: &std::path::Path,
    slo: SloTargets,
    k: usize,
    kv_admission: KvAdmission,
    scheduling: SchedulingPolicy,
) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 1024,
        threads: 2,
        max_seqs: 6,
        max_batch_tokens: 256,
        prefill_chunk: 256,
        spec: SpecConfig {
            kind: if k > 0 { SpecKind::Mtp } else { SpecKind::None },
            k,
            draft: Some(dir.join("mtp.gguf")),
            p_min: 0.0,
            ..Default::default()
        },
        use_ref: true,
        scheduling,
        kv_admission,
        slo,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// Chats decoding while long prompts arrive behind them, in several priority classes.
async fn answers(e: &LmEngine) -> Vec<(String, u32)> {
    let prompts = [
        ("a chat".to_string(), 120, 0),
        ("another chat, a bit longer".into(), 100, 1),
        ("long prompt ".repeat(40), 30, 0),
        ("a third chat".into(), 90, 0),
        ("an even longer prompt ".repeat(30), 20, -1),
        ("last".into(), 60, 2),
    ];
    let mut streams = Vec::new();
    for (p, max_tokens, priority) in &prompts {
        streams.push(
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
                scheduling: Scheduling {
                    priority: *priority,
                    share_key: None,
                },
            })
            .unwrap(),
        );
        // The chats start decoding before the long prompts arrive.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    let mut out = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { usage, .. } => {
                    out.push((text, usage.completion_tokens));
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn targets_answer_as_without() {
    let dir = tempfile::tempdir().unwrap();
    legion::model::qwen4::tiny::write(
        dir.path().join("tiny.gguf"),
        Some(&dir.path().join("mtp.gguf")),
        11,
    )
    .unwrap();
    // 1 ms: every step with decoding rows is over the targets, so prompts read a block a step
    // and every prompt is at risk (reordered). 400 ms: whole chunks, as without targets.
    let targets = [
        SloTargets {
            itl_ms: Some(1.0),
            ttft_ms: Some(1.0),
        },
        SloTargets {
            itl_ms: Some(1.0),
            ttft_ms: None,
        },
        SloTargets {
            itl_ms: None,
            ttft_ms: Some(1.0),
        },
        SloTargets {
            itl_ms: Some(400.0),
            ttft_ms: Some(400.0),
        },
    ];
    for (k, admission, policy) in [
        (0, KvAdmission::Reserve, SchedulingPolicy::Fcfs),
        (3, KvAdmission::OnDemand, SchedulingPolicy::Priority),
    ] {
        let off = qwen(dir.path(), SloTargets::default(), k, admission, policy);
        let want = answers(&off).await;
        assert!(want.iter().all(|(t, _)| !t.is_empty()));
        for slo in targets {
            let on = qwen(dir.path(), slo, k, admission, policy);
            let got = answers(&on).await;
            assert_eq!(want, got, "k = {k}, {admission:?}, {policy:?}, {slo:?}");
        }
    }
}

#[test]
fn groups_in_flight_refuse_the_targets() {
    let err = LmEngine::load(LmConfig {
        pipeline_groups: 2,
        slo: SloTargets {
            itl_ms: Some(100.0),
            ttft_ms: None,
        },
        ..LmConfig::new("/nonexistent/model.gguf", DeviceChoice::Cpu)
    })
    .err()
    .expect("refused")
    .to_string();
    assert!(err.contains("--pipeline-groups"), "{err}");
}
