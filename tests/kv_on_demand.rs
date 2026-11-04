//! `--kv-admission on-demand` (#627): sequences admitted on their prompt's KV blocks, growing
//! as they decode, preempted by recompute when the pool runs short. On the CPU reference
//! kernels a prompt step gives a decode step's bits, so every request's greedy answer must be
//! the one `reserve` admission gives, preempted or not, with and without MTP drafts.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, KvAdmission, LmConfig, LmEngine, STREAM_THREAD_SEQS};
use legion::engine::queue::{Scheduling, SchedulingPolicy};
use legion::engine::{Engine, GenerateRequest, Prompt, ResponseFormat, SamplingParams, TokenEvent};
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

fn write_models(dir: &std::path::Path) {
    legion::model::qwen4::tiny::write(dir.join("tiny.gguf"), Some(&dir.join("mtp.gguf")), 11)
        .unwrap();
}

/// The tiny Qwen (recurrent DeltaNet layers, saved prefix states, MTP head).
fn qwen(
    dir: &std::path::Path,
    kv_admission: KvAdmission,
    k: usize,
    scheduling: SchedulingPolicy,
) -> LmEngine {
    qwen_streaming(dir, kv_admission, k, scheduling, STREAM_THREAD_SEQS)
}

/// [`qwen`] with `--stream-thread-seqs` `stream_thread_seqs`.
fn qwen_streaming(
    dir: &std::path::Path,
    kv_admission: KvAdmission,
    k: usize,
    scheduling: SchedulingPolicy,
    stream_thread_seqs: usize,
) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 6,
        max_batch_tokens: 64,
        prefill_chunk: 40,
        // Ten blocks: `reserve` runs two of these requests at once, `on-demand` all six
        // until they outgrow the pool.
        kv_cache_tokens: Some(640),
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
        stream_thread_seqs,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// The tiny Gemma 4 (no recurrent state; sliding-window layers in rings, finished sequences
/// kept warm for prefix caching).
fn gemma(dir: &std::path::Path, kv_admission: KvAdmission) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 6,
        max_batch_tokens: 64,
        prefill_chunk: 40,
        kv_cache_tokens: Some(640),
        use_ref: true,
        kv_admission,
        ..LmConfig::new(dir.join("gemma.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

/// Eight requests of different lengths and priorities, each asking for more tokens than the
/// pool holds for all of them at once.
async fn answers(e: &LmEngine) -> Vec<(String, u32)> {
    answers_as(e, None).await
}

/// [`answers`], every request under `response_format` when given.
async fn answers_as(e: &LmEngine, format: Option<ResponseFormat>) -> Vec<(String, u32)> {
    let prompts = [
        ("Many users at once: ".repeat(3), 180, 0),
        ("speculative".into(), 230, 1),
        ("routing 0123456789".repeat(2), 150, 0),
        ("abc".into(), 260, 2),
        ("pipeline stages ".repeat(2), 200, -1),
        ("groups in flight".into(), 120, 1),
        ("the last one".into(), 210, 0),
        ("one more, queued".into(), 90, 0),
    ];
    let streams: Vec<_> = prompts
        .iter()
        .map(|(p, max_tokens, priority)| {
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
                response_format: format.clone(),
                template_kwargs: Default::default(),
                cancel: CancellationToken::new(),
                scheduling: Scheduling {
                    priority: *priority,
                    share_key: None,
                },
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
                TokenEvent::Finished { usage, .. } => {
                    out.push((text, usage.completion_tokens));
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
        assert!(s.next().await.is_none(), "events after Finished");
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn on_demand_answers_as_reserve_with_preemptions() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    for k in [0, 3] {
        for policy in [SchedulingPolicy::Fcfs, SchedulingPolicy::Priority] {
            let reserve = qwen(dir.path(), KvAdmission::Reserve, k, policy);
            let want = answers(&reserve).await;
            assert_eq!(reserve.stats().preemptions, 0);
            let on_demand = qwen(dir.path(), KvAdmission::OnDemand, k, policy);
            let got = answers(&on_demand).await;
            let stats = on_demand.stats();
            assert!(
                stats.preemptions > 0,
                "spec k = {k}, {policy:?}: nothing was preempted"
            );
            assert_eq!(want, got, "spec k = {k}, {policy:?}");
            assert!(want.iter().all(|(t, _)| !t.is_empty()));
            // A recomputed sequence's next token was sampled once, before its preemption.
            assert_eq!(
                stats.generated_tokens,
                reserve.stats().generated_tokens,
                "spec k = {k}, {policy:?}: tokens counted twice"
            );
        }
    }
}

/// A preempted grammar-constrained sequence keeps its matcher (and its pending mask) in its
/// request state; after the recompute it samples on from the same grammar state.
#[tokio::test(flavor = "multi_thread")]
async fn on_demand_keeps_the_grammar_state_of_a_preempted_sequence() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    for k in [0, 3] {
        let policy = SchedulingPolicy::Fcfs;
        let format = Some(ResponseFormat::JsonObject);
        let reserve = qwen(dir.path(), KvAdmission::Reserve, k, policy);
        let want = answers_as(&reserve, format.clone()).await;
        let on_demand = qwen(dir.path(), KvAdmission::OnDemand, k, policy);
        let got = answers_as(&on_demand, format).await;
        assert!(
            on_demand.stats().preemptions > 0,
            "spec k = {k}: nothing was preempted"
        );
        assert_eq!(want, got, "spec k = {k}");
        for (text, _) in &want {
            assert!(text.trim_start().starts_with('{'), "not JSON: {text:?}");
        }
    }
}

/// Preempted sequences whose text is on the stream thread (`--stream-thread-seqs 1`: every
/// sequence's) stream what `reserve` streams with all text on the engine thread: the text
/// queued before a preemption, then the rest after the recompute, then one `Finished`.
#[tokio::test(flavor = "multi_thread")]
async fn on_demand_streams_preempted_sequences_from_the_stream_thread() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    for k in [0, 3] {
        let policy = SchedulingPolicy::Fcfs;
        let reserve = qwen_streaming(dir.path(), KvAdmission::Reserve, k, policy, 0);
        let want = answers(&reserve).await;
        let on_demand = qwen_streaming(dir.path(), KvAdmission::OnDemand, k, policy, 1);
        let got = answers(&on_demand).await;
        assert!(
            on_demand.stats().preemptions > 0,
            "spec k = {k}: nothing was preempted"
        );
        assert_eq!(want, got, "spec k = {k}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn on_demand_answers_as_reserve_with_rings() {
    let dir = tempfile::tempdir().unwrap();
    legion::model::gemma4::tiny::write(dir.path().join("gemma.gguf"), 1).unwrap();
    let want = answers(&gemma(dir.path(), KvAdmission::Reserve)).await;
    let on_demand = gemma(dir.path(), KvAdmission::OnDemand);
    let got = answers(&on_demand).await;
    assert!(on_demand.stats().preemptions > 0, "nothing was preempted");
    assert_eq!(want, got);
}
