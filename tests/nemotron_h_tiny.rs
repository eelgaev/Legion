//! Nemotron-H through the real graph and engine on the CPU, with the tiny random
//! `nemotron_h_moe` (Mamba-2, NVFP4 MoE with per-tensor scales, attention) and reference
//! kernels: decode steps carry the Mamba-2 conv and SSM state exactly, and the engine's greedy
//! output is the model's own greedy continuation whatever the batching (sequences decoded
//! together, prefill chunks next to decode steps), the prefix cache (prompts resuming from
//! saved recurrent states) and speculative decoding (verify steps rolled back by replaying the
//! selective scan) do.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::sampling::Rng;
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::model::nemotron_h::Block;
use legion::model::nemotron_h::tiny;
use legion::spec::{SpecConfig, SpecCounters, SpecKind, TokenDrafter, TokenDrafterFactory};
use legion::tokenizer::{TokenKind, Tokenizer};
use tokio_util::sync::CancellationToken;

const K: usize = 3;
const MAX_TOKENS: u32 = 40;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4), // shares blocks with the next
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "state space models".into(),
        "0123456789".repeat(3),
        "abc".into(),
    ]
}

fn cpu() -> Backend {
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    b.set_use_ref(true);
    b
}

fn argmax(x: &[f32]) -> u32 {
    legion::engine::sampling::argmax(x) as u32
}

#[test]
fn tiny_model_loads_as_nemotron_h() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let nm = legion::model::nemotron_h::params(&model.hp).expect("Nemotron-H hyperparameters");
    let blocks: String = nm
        .blocks
        .iter()
        .map(|b| match b {
            Block::Mamba => 'M',
            Block::Attention => '*',
            Block::Ffn => 'E',
        })
        .collect();
    assert_eq!(blocks, tiny::PATTERN);
    assert!(model.hp.moe && model.hp.is_recurrent(0) && model.hp.is_attention(3));
    assert!(!model.hp.is_attention(0) && !model.hp.is_recurrent(1));
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let logits = model
        .prompt_logits(&mut backend, &tok.encode("hello", false))
        .unwrap();
    assert_eq!(logits.len(), model.hp.n_vocab as usize);
    assert!(logits.iter().all(|x| x.is_finite()), "{logits:?}");
}

/// Prefill, then one token per step, against one pass over the whole text: bit-identical
/// logits at every position (reference kernels are batch-invariant), so the conv history and
/// the SSM state carried between steps are exactly the ones a longer prefill computes.
#[test]
fn tiny_decode_matches_full_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 2).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let prompt: Vec<u32> = "Mamba-2 and attention".bytes().map(u32::from).collect();
    let mut state = model.new_state(backend.buffer_type(), 256).unwrap();
    let mut alloc = None;
    // The prompt in two chunks (the second resumes from the first's states), then decode.
    let split = prompt.len() / 2;
    model
        .forward(
            &mut backend,
            &mut alloc,
            &mut state,
            &prompt[..split],
            false,
        )
        .unwrap();
    let mut rows = vec![
        model
            .forward(
                &mut backend,
                &mut alloc,
                &mut state,
                &prompt[split..],
                false,
            )
            .unwrap(),
    ];
    let mut ids = prompt.clone();
    for _ in 0..12 {
        let next = argmax(rows.last().unwrap());
        ids.push(next);
        rows.push(
            model
                .forward(&mut backend, &mut alloc, &mut state, &[next], false)
                .unwrap(),
        );
    }
    let full = model.prompt_logits_all(&mut backend, &ids).unwrap();
    for (i, row) in rows.iter().enumerate() {
        let p = prompt.len() - 1 + i;
        assert_eq!(
            row.as_slice(),
            &full[p * n_vocab..(p + 1) * n_vocab],
            "position {p}"
        );
    }
}

fn engine(path: &std::path::Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        // Small enough that prompts prefill in several chunks next to decoding sequences.
        max_batch_tokens: 40,
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
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        template_kwargs: Default::default(),
        response_format: None,
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
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
async fn run_all(engine: &LmEngine) -> Vec<Reply> {
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

/// The greedy continuation of every prompt, one sequence at a time through `Model::forward`.
fn truths(path: &std::path::Path) -> Vec<(Vec<u32>, Vec<u32>)> {
    let gguf = GgufFile::open(path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    prompts()
        .iter()
        .map(|p| {
            let prompt: Vec<u32> = p.bytes().map(u32::from).collect();
            let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
            let mut alloc = None;
            let mut logits = model
                .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
                .unwrap();
            let mut out = Vec::new();
            for _ in 0..MAX_TOKENS as usize + K {
                let t = argmax(&logits);
                out.push(t);
                logits = model
                    .forward(&mut backend, &mut alloc, &mut state, &[t], false)
                    .unwrap();
            }
            (prompt, out)
        })
        .collect()
}

/// The true continuation after `tokens` (a prompt plus generated tokens), or nothing once
/// `tokens` left it.
fn truth_after<'a>(truths: &'a [(Vec<u32>, Vec<u32>)], tokens: &[u32]) -> &'a [u32] {
    truths
        .iter()
        .filter(|(p, _)| tokens.len() > p.len() && tokens.starts_with(p))
        .max_by_key(|(p, _)| p.len())
        .and_then(|(p, t)| {
            let done = &tokens[p.len()..];
            t.starts_with(done).then(|| &t[done.len()..])
        })
        .unwrap_or(&[])
}

/// Drafts the true continuation, always wrong ones, or a mix.
#[derive(Clone, Copy)]
enum Mode {
    Oracle,
    Adversary,
    Noisy(f64),
}

struct TestDrafter {
    truths: Arc<Vec<(Vec<u32>, Vec<u32>)>>,
    mode: Mode,
    rng: Rng,
}

impl TokenDrafter for TestDrafter {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        let truth = truth_after(&self.truths, tokens);
        let n = match self.mode {
            Mode::Noisy(_) => (self.rng.next_u64() % (max as u64 + 1)) as usize,
            _ => max,
        };
        for i in 0..n {
            let right = truth.get(i).copied().unwrap_or(0);
            out.push(match self.mode {
                Mode::Oracle => right,
                Mode::Adversary => (right + 1) % 259,
                Mode::Noisy(p) if self.rng.next_f64() < p => right,
                Mode::Noisy(_) => (self.rng.next_u64() % 259) as u32,
            });
        }
    }
}

fn spec(truths: &Arc<Vec<(Vec<u32>, Vec<u32>)>>, mode: Mode) -> SpecConfig {
    let truths = truths.clone();
    let seed = Mutex::new(0u64);
    let factory: TokenDrafterFactory = Arc::new(move || {
        let mut s = seed.lock().unwrap();
        *s += 1;
        Box::new(TestDrafter {
            truths: truths.clone(),
            mode,
            rng: Rng::new(*s),
        })
    });
    SpecConfig {
        k: K,
        token_drafter: Some(factory),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tiny_engine_greedy_matches_alone_batched_cached_and_speculative() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let truths = Arc::new(truths(&path));
    let tok = Tokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();

    // Batched and chunked, twice (the second round resumes from the prefix cache): each reply
    // is its prompt's greedy continuation computed alone.
    let plain = engine(&path, SpecConfig::default());
    let first = run_all(&plain).await;
    for (i, (r, (_, truth))) in first.iter().zip(truths.iter()).enumerate() {
        let n = r.tokens as usize;
        assert!(n > 0, "prompt {i}: nothing generated");
        // A stop token ends the reply without counting as a completion token.
        assert_eq!(r.text, tok.decode(&truth[..n]), "prompt {i}");
        if r.reason == FinishReason::Stop {
            assert!(
                tok.token_kind(truth[n]) == Some(TokenKind::Control),
                "prompt {i}"
            );
        }
    }
    assert!(first.iter().any(|r| r.tokens > 10), "{first:?}");
    let again = run_all(&plain).await;
    assert_eq!(first, again, "prefix-cached round differs");
    assert!(plain.stats().prefix_cache_hit_tokens > 0, "no prefix reuse");

    // The model's own MTP head drafts; whatever it proposes, the output stays.
    let e = engine(
        &path,
        SpecConfig {
            kind: SpecKind::Mtp,
            k: 2,
            p_min: 0.0,
            ..Default::default()
        },
    );
    for round in 0..2 {
        let replies = run_all(&e).await;
        for (i, (a, b)) in first.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "mtp, round {round}: prompt {i} differs"
            );
        }
        let mut c = SpecCounters::default();
        for r in &replies {
            c.add(&r.spec.expect("speculation counts"));
        }
        assert!(c.drafted > 0, "mtp: nothing drafted");
    }

    for (name, mode) in [
        ("oracle", Mode::Oracle),
        ("adversary", Mode::Adversary),
        ("partly right", Mode::Noisy(0.7)),
    ] {
        let e = engine(&path, spec(&truths, mode));
        for round in 0..2 {
            let replies = run_all(&e).await;
            for (i, (a, b)) in first.iter().zip(&replies).enumerate() {
                assert_eq!(
                    (&a.text, a.reason, a.tokens),
                    (&b.text, b.reason, b.tokens),
                    "{name}, round {round}: prompt {i} differs"
                );
            }
            let mut c = SpecCounters::default();
            for r in &replies {
                c.add(&r.spec.expect("speculation counts"));
            }
            assert!(c.drafted > 0, "{name}: nothing drafted");
            match mode {
                Mode::Oracle => assert_eq!(c.accepted, c.drafted, "{name}"),
                Mode::Adversary => assert_eq!(c.accepted, 0, "{name}"),
                Mode::Noisy(_) => assert!(c.accepted > 0 && c.accepted < c.drafted, "{name}"),
            }
        }
    }
}

/// The tiny model on the CUDA backend against the CPU (`--features cuda`; ignored: needs a GPU).
/// On the GPU the NVFP4 experts' scales fold into the routing weights, the CPU applies them
/// as llama.cpp does, so this is a rounding-level comparison.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 5).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut cpu = cpu();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU")
        .init()
        .unwrap();
    let prompt: Vec<u32> = "Mamba-2 and attention on a GPU"
        .bytes()
        .map(u32::from)
        .collect();
    let want = Model::load(&gguf, |_| cpu.buffer_type())
        .unwrap()
        .prompt_logits_all(&mut cpu, &prompt)
        .unwrap();
    let on_gpu = Model::load(&gguf, |_| gpu.buffer_type()).unwrap();
    let got = on_gpu.prompt_logits_all(&mut gpu, &prompt).unwrap();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let n_vocab = on_gpu.hp.n_vocab as usize;
    let top1 = got
        .chunks(n_vocab)
        .zip(want.chunks(n_vocab))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    let rows = want.len() / n_vocab;
    println!("max |diff| {diff} of {scale}, top-1 {top1}/{rows}");
    assert!(diff < 0.05 * scale, "max |diff| {diff} of {scale}");
}
