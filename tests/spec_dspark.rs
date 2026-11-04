//! `--spec dspark` through the Qwen3.5 engine on the CPU, with the tiny random target and a tiny
//! random DSpark drafter (Markov head, confidence head, attention sinks; also with NVFP4
//! matrices): greedy output is bit-identical to the same engine without speculation (the drafts
//! are mostly wrong, so every step exercises the drafter's rollback), with several sequences
//! batched and prompts that resume from the prefix cache; sampled drafts with their
//! distributions run through the exact rejection path.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::ffi;
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::dflash::tiny::{self as tiny_draft, HfDtype, Sinks, TinyDraft};
use legion::spec::{SpecConfig, SpecCounters, SpecKind};
use tokio_util::sync::CancellationToken;

const K: usize = 4;
const MAX_TOKENS: u32 = 40;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4),
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "semi-autoregressive".into(),
        "0123456789".repeat(3),
        "abc".into(),
    ]
}

fn engine(path: &std::path::Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        // Small enough that a prompt prefills in several steps next to decoding sequences.
        max_batch_tokens: 40,
        spec,
        use_ref: true,
        ..LmConfig::new(path, DeviceChoice::Cpu)
    })
    .unwrap()
}

fn request(prompt: &str, sampling: SamplingParams) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: Prompt::Text(prompt.into()),
        sampling,
        max_tokens: MAX_TOKENS,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

#[derive(Debug, PartialEq)]
struct Reply {
    text: String,
    reason: FinishReason,
    tokens: u32,
    spec: Option<SpecCounters>,
}

/// Every prompt submitted at once (so they batch); the replies in prompt order.
async fn run_all(engine: &LmEngine, sampling: SamplingParams) -> Vec<Reply> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| engine.generate(request(p, sampling.clone())).unwrap())
        .collect();
    let mut replies = Vec::new();
    for mut s in streams {
        let mut text = String::new();
        loop {
            match s.next().await.expect("stream ended without Finished") {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { reason, usage, .. } => {
                    replies.push(Reply {
                        text,
                        reason,
                        tokens: usage.completion_tokens,
                        spec: usage.spec,
                    });
                    break;
                }
                TokenEvent::Error(e) => panic!("engine error: {e}"),
                _ => {}
            }
        }
    }
    replies
}

fn totals(replies: &[Reply]) -> SpecCounters {
    let mut c = SpecCounters::default();
    for r in replies {
        c.add(&r.spec.expect("speculation counts"));
    }
    c
}

struct Files {
    _dir: tempfile::TempDir,
    target: std::path::PathBuf,
    gguf: std::path::PathBuf,
    nvfp4: std::path::PathBuf,
    hf: std::path::PathBuf,
}

fn spark() -> TinyDraft {
    TinyDraft {
        markov_rank: Some(64),
        confidence: true,
        sinks: Sinks::Weight,
        n_ff: 128,
        ..TinyDraft::default()
    }
}

fn files() -> Files {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("tiny.gguf");
    tiny::write(&target, &TinyConfig::default()).unwrap();
    let gguf = dir.path().join("drafter.gguf");
    tiny_draft::write_gguf(&gguf, &spark()).unwrap();
    let nvfp4 = dir.path().join("drafter-nvfp4.gguf");
    let q = TinyDraft {
        quant: Some(ffi::ggml_type_GGML_TYPE_NVFP4),
        // The bonus-anchor layout, drafting from the masks.
        sample_from_anchor: false,
        sinks: Sinks::Bare,
        ..spark()
    };
    tiny_draft::write_gguf(&nvfp4, &q).unwrap();
    let hf = dir.path().join("drafter-hf");
    tiny_draft::write_hf(&hf, &spark(), HfDtype::F32).unwrap();
    Files {
        _dir: dir,
        target,
        gguf,
        nvfp4,
        hf,
    }
}

fn dspark(draft: &std::path::Path) -> SpecConfig {
    SpecConfig {
        kind: SpecKind::Dspark,
        k: K,
        draft: Some(draft.to_path_buf()),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn greedy_dspark_matches_plain_decoding() {
    let f = files();
    let plain = run_all(&engine(&f.target, SpecConfig::default()), greedy()).await;
    assert!(plain.iter().any(|r| r.tokens > 10), "{plain:?}");
    for (name, draft) in [
        ("gguf", &f.gguf),
        ("nvfp4 gguf", &f.nvfp4),
        ("hugging face", &f.hf),
    ] {
        let e = engine(&f.target, dspark(draft));
        // Twice: the second round's prompts resume from the prefix cache, so the drafter gets
        // sequences that start past position 0.
        for round in 0..2 {
            let replies = run_all(&e, greedy()).await;
            for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
                assert_eq!(
                    (&a.text, a.reason, a.tokens),
                    (&b.text, b.reason, b.tokens),
                    "{name}, round {round}: prompt {i} differs"
                );
            }
            let c = totals(&replies);
            println!("{name}, round {round}: {c:?}");
            assert!(c.drafted > 0, "{name}: nothing drafted");
            assert!(
                c.accepted < c.drafted,
                "{name}: a random drafter cannot be right"
            );
        }
        assert!(e.stats().prefix_cache_hit_tokens > 0, "no prefix reuse");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sampled_dspark_drafts_use_exact_rejection() {
    let f = files();
    let plain = run_all(&engine(&f.target, SpecConfig::default()), greedy()).await;
    let spec = SpecConfig {
        draft_temperature: 1.0,
        // Blocks are cut where the confidence head doubts them.
        min_confidence: Some(0.3),
        ..dspark(&f.gguf)
    };
    let e = engine(&f.target, spec);
    // A greedy request is exact whatever the drafter draws.
    let replies = run_all(&e, greedy()).await;
    for (a, b) in plain.iter().zip(&replies) {
        assert_eq!((&a.text, a.tokens), (&b.text, b.tokens));
    }
    // A sampling request runs through rejection sampling with the drafter's distributions.
    let sampling = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        seed: Some(3),
        ..Default::default()
    };
    let replies = run_all(&e, sampling).await;
    for r in &replies {
        let c = r.spec.unwrap();
        assert!(c.accepted <= c.drafted);
        assert!(r.tokens > 0);
    }
    assert!(totals(&replies).drafted > 0);
}

#[test]
fn engine_load_reports_a_missing_or_wrong_drafter() {
    let f = files();
    let load = |spec: SpecConfig| {
        LmEngine::load(LmConfig {
            ctx_size: 512,
            threads: 2,
            max_seqs: 1,
            spec,
            use_ref: true,
            ..LmConfig::new(&f.target, DeviceChoice::Cpu)
        })
        .err()
        .map(|e| e.to_string())
    };
    let none = load(SpecConfig {
        kind: SpecKind::Dspark,
        k: 4,
        ..Default::default()
    });
    assert!(none.unwrap().contains("--spec-draft"));
    // The target model is not a drafter.
    assert!(
        load(dspark(&f.target))
            .unwrap()
            .contains("not a DFlash drafter")
    );
    // A DFlash drafter has no Markov head; a DSpark drafter is not a DFlash.
    let flash = f.target.with_file_name("flash.gguf");
    tiny_draft::write_gguf(&flash, &TinyDraft::default()).unwrap();
    assert!(load(dspark(&flash)).unwrap().contains("Markov head"));
    let e = load(SpecConfig {
        kind: SpecKind::Dflash,
        ..dspark(&f.gguf)
    });
    assert!(e.unwrap().contains("--spec dspark"));
}
