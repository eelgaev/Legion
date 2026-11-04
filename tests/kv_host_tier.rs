//! A KV cache with a host tier (#352, `--kv-host-tokens`): the engine's greedy output equals
//! the output with every block in device memory, with sequences that need host blocks (the
//! device holds 4 blocks, the prompts take more), batched; for Gemma 4 and for qwen4exp, whose
//! sparse-attention layers keep their indexer keys in VRAM and their K and V tiered. Needs a
//! GPU whose driver maps host NUMA memory.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::gemma4::tiny;
use tokio_util::sync::CancellationToken;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(6),
        "0123456789".repeat(20),
        "host tiers hold the blocks that do not fit".into(),
    ]
}

async fn run_all(engine: &LmEngine) -> Vec<String> {
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
                    max_tokens: 80,
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
#[ignore = "needs a GPU whose driver maps host NUMA memory"]
async fn host_tier_gives_the_device_caches_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 5).unwrap();
    let cfg = |device_tokens: usize, host_tokens: usize| LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        max_seqs: 3,
        prefix_cache_states: 0,
        kv_cache_tokens: Some(device_tokens),
        kv_host_tokens: Some(host_tokens),
        ..LmConfig::new(&path, DeviceChoice::Cuda)
    };
    let all = run_all(&LmEngine::load(cfg(3 * 512, 0)).unwrap()).await;
    // 4 blocks in device memory; every other block in the host tier.
    let tiered = run_all(&LmEngine::load(cfg(256, 3 * 512)).unwrap()).await;
    assert_eq!(tiered, all);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a GPU whose driver maps host NUMA memory"]
async fn host_tier_gives_the_device_caches_output_with_sparse_attention() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny-qwen4exp.gguf");
    legion::model::qwen4::tiny::write_cuda_indexer(&path, None, 7).unwrap();
    let cfg = |device_tokens: usize, host_tokens: usize| LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        max_seqs: 3,
        prefix_cache_states: 0,
        kv_cache_tokens: Some(device_tokens),
        kv_host_tokens: Some(host_tokens),
        ..LmConfig::new(&path, DeviceChoice::Cuda)
    };
    let all = run_all(&LmEngine::load(cfg(3 * 512, 0)).unwrap()).await;
    let tiered = run_all(&LmEngine::load(cfg(256, 3 * 512)).unwrap()).await;
    assert_eq!(tiered, all);
}
