//! Tensor parallelism over machines (#619) through the engine on the CPU, ranks as threads over
//! loopback: the dense tiny `gemma4` over two ranks gives the greedy output of one (up to the
//! rounding of the split sums, which this model's greedy picks do not feel), batched, with the
//! sliding-window rings saved and restored for warm sequences on every rank, and the same
//! output on a second run; also with `--spec mtp` drafting with a split Gemma 4 assistant, and
//! with `--spec dflash|dspark` drafting on rank 0 alone.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::model::gemma4::tiny;
use legion::spec::{SpecConfig, SpecKind};
use legion::tp_net::TpNetConfig;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const MAX_TOKENS: u32 = 60;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4),
        "0123456789".repeat(9),
        "tensor parallel".into(),
    ]
}

fn config(path: &std::path::Path, tp_net: Option<TpNetConfig>) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 2,
        max_seqs: 2,
        max_batch_tokens: 40,
        prefill_chunk: 16,
        ring_step: Some(16),
        use_ref: true,
        tp_net,
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

fn free_ports(n: usize) -> Vec<String> {
    let ls: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    ls.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect()
}

/// The output of one rank, then of two ranks (twice) must be the same, with `spec`.
async fn two_ranks_match_one(path: &std::path::Path, spec: SpecConfig) {
    let config = |tp_net| LmConfig {
        spec: spec.clone(),
        ..config(path, tp_net)
    };
    let one = {
        let e = LmEngine::load(config(None)).unwrap();
        run_all(&e).await
    };
    assert!(
        one.iter().all(|r| r.2 == MAX_TOKENS),
        "generations too short"
    );

    let peers = free_ports(2);
    let cfg = |rank| TpNetConfig {
        rank,
        peers: peers.clone(),
        timeout: Duration::from_secs(60),
        ..Default::default()
    };
    // Rank 1 serves rank 0 until it disconnects; its load returns then.
    let follower = {
        let c = config(Some(cfg(1)));
        std::thread::spawn(move || LmEngine::load(c).err().map(|e| e.to_string()))
    };
    let lead = {
        let c = config(Some(cfg(0)));
        tokio::task::spawn_blocking(move || LmEngine::load(c))
            .await
            .unwrap()
            .unwrap()
    };
    for round in 0..2 {
        let two = run_all(&lead).await;
        for (i, (a, b)) in one.iter().zip(&two).enumerate() {
            assert_eq!(a, b, "round {round}, prompt {i}: two ranks differ from one");
        }
    }
    if spec.enabled() {
        assert!(lead.stats().spec.drafted > 0, "rank 0 drafted nothing");
    }
    drop(lead);
    let e = follower.join().unwrap().expect("rank 1 stops with rank 0");
    assert!(e.contains("rank 0 disconnected"), "{e}");
}

#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_give_the_output_of_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    two_ranks_match_one(&path, SpecConfig::default()).await;
}

/// `--spec mtp` with a Gemma 4 assistant: its queries and FFN split like the target's, over
/// each rank's share of the target's cache.
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_draft_with_a_split_assistant() {
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
    two_ranks_match_one(&path, spec).await;
}

/// `--spec dflash` with a DFlash 2 drafter (and `--spec dspark`): rank 0 alone drafts, from
/// the target's hidden states (whole on every rank after the all-reduces), and rank 1 runs its
/// verify steps. Without `--spec-k` both ranks take the DFlash 2 block (rank 1 from the
/// drafter's config alone).
#[tokio::test(flavor = "multi_thread")]
async fn two_ranks_draft_with_dflash_on_rank_0() {
    use legion::spec::dflash::tiny::{self as tiny_draft, TINY_V2, TinyDraft};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    let dflash2 = dir.path().join("dflash2.gguf");
    tiny_draft::write_gguf(
        &dflash2,
        &TinyDraft {
            v2: Some(TINY_V2),
            ..TinyDraft::default()
        },
    )
    .unwrap();
    let spec = SpecConfig {
        kind: SpecKind::Dflash,
        fixed: true,
        draft: Some(dflash2),
        ..Default::default()
    };
    two_ranks_match_one(&path, spec).await;
    let dspark = dir.path().join("dspark.gguf");
    tiny_draft::write_gguf(
        &dspark,
        &TinyDraft {
            markov_rank: Some(8),
            ..TinyDraft::default()
        },
    )
    .unwrap();
    let spec = SpecConfig {
        kind: SpecKind::Dspark,
        k: 3,
        fixed: true,
        draft: Some(dspark),
        ..Default::default()
    };
    two_ranks_match_one(&path, spec).await;
}

/// A drafter model rank 0 cannot drive the others with is refused before connecting.
#[test]
fn other_drafter_models_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    tiny::write_dense(&path, 11).unwrap();
    let c = LmConfig {
        spec: SpecConfig {
            kind: SpecKind::Eagle3,
            k: 3,
            draft: Some(dir.path().join("eagle3")),
            ..Default::default()
        },
        ..config(
            &path,
            Some(TpNetConfig {
                rank: 1,
                peers: free_ports(2),
                timeout: Duration::from_secs(5),
                ..Default::default()
            }),
        )
    };
    let e = LmEngine::load(c).err().expect("refused").to_string();
    assert!(e.contains("--spec mtp|dflash|dspark"), "{e}");
}

#[test]
fn a_moe_model_is_refused_before_connecting() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("moe.gguf");
    tiny::write(&path, 11).unwrap();
    let c = config(
        &path,
        Some(TpNetConfig {
            rank: 1,
            peers: free_ports(2),
            timeout: Duration::from_secs(5),
            ..Default::default()
        }),
    );
    let e = LmEngine::load(c).err().expect("refused").to_string();
    assert!(e.contains("experts"), "{e}");
}
