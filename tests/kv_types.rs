//! Quantized KV caches (`--cache-type-k` / `--cache-type-v`, #295) through the engine on the
//! CPU, with the tiny random `gemma4` (window 8, shared-KV layers) and reference kernels: a
//! quantized cache is gathered and dequantized; with its sliding-window layers in rings it gives
//! the output of the same types over every position, batched, with n-gram drafts rolled back
//! and with the MTP assistant reading the target's quantized cache, through gather attention
//! and through the paged op.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::core::{Attention, KvType, KvTypes};
use legion::model::gemma4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use tokio_util::sync::CancellationToken;

const MAX_TOKENS: u32 = 100;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(6),
        "0123456789".repeat(15),
        "Ring buffers keep the last few positions only; ".repeat(4),
    ]
}

fn config(
    path: &std::path::Path,
    spec: SpecConfig,
    types: KvTypes,
    swa_full: bool,
    attention: Attention,
) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 2,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        ring_step: Some(16),
        prefix_cache_states: 0,
        attention,
        swa_full,
        cache_types: types,
        spec,
        use_ref: true,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    }
}

async fn run_all(engine: &LmEngine) -> Vec<(String, u32)> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| {
            engine
                .generate(GenerateRequest {
                    model: "tiny".into(),
                    prompt: Prompt::Text(p.clone()),
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
                })
                .unwrap()
        })
        .collect();
    let mut out = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended without Finished") {
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
async fn quantized_caches_through_rings_match_every_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let assistant = dir.path().join("assistant.gguf");
    tiny::write_assistant(&assistant, 7).unwrap();
    let specs = [
        SpecConfig::default(),
        SpecConfig {
            kind: SpecKind::Ngram,
            k: 4,
            fixed: true,
            ..Default::default()
        },
        SpecConfig {
            kind: SpecKind::Mtp,
            k: 3,
            fixed: true,
            p_min: 0.0,
            draft: Some(assistant),
            ..Default::default()
        },
    ];
    // Gather attention dequantizes blocks; paged attention (the CPU op here, the kernels on
    // a GPU) dequantizes in place.
    for attention in [Attention::Gather, Attention::Paged] {
        for (k, v) in [(KvType::Q8_0, KvType::Q8_0), (KvType::Q8_0, KvType::Q4_0)] {
            let types = KvTypes { k, v };
            for spec in &specs {
                let cfg = |swa_full| config(&path, spec.clone(), types, swa_full, attention);
                let full = run_all(&LmEngine::load(cfg(true)).unwrap()).await;
                let ring = run_all(&LmEngine::load(cfg(false)).unwrap()).await;
                assert_eq!(
                    full, ring,
                    "{attention:?}, {types:?}, {:?}: rings differ",
                    spec.kind
                );
                assert!(ring.iter().all(|r| r.1 == MAX_TOKENS));
            }
        }
    }
}

/// Tensor parallelism's read of a quantized cache: gather by cells (one index per key, so a
/// cache split by heads gathers on each GPU), through the rings, on one device here. Rings give
/// the output of the same types over every position.
#[tokio::test(flavor = "multi_thread")]
async fn quantized_caches_through_rings_by_cells_match_every_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    for (k, v) in [(KvType::Q8_0, KvType::Q8_0), (KvType::Q8_0, KvType::Q4_0)] {
        let types = KvTypes { k, v };
        let cfg = |swa_full| LmConfig {
            cell_gather: true,
            ring_gather: true,
            ..config(
                &path,
                SpecConfig::default(),
                types,
                swa_full,
                Attention::Gather,
            )
        };
        let full = run_all(&LmEngine::load(cfg(true)).unwrap()).await;
        let ring = run_all(&LmEngine::load(cfg(false)).unwrap()).await;
        assert_eq!(full, ring, "{types:?}: rings differ");
        assert!(ring.iter().all(|r| r.1 == MAX_TOKENS));
    }
}
