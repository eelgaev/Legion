//! Gemma 4 through the real graph and engine on the CPU, with the tiny random `gemma4`
//! (sliding-window and global layers of different head sizes, a keys-as-values layer,
//! shared-KV layers, MoE, per-layer embeddings, soft-cap) and reference kernels: decode steps
//! read exactly the cache a longer prefill writes (also past the sliding window and through the
//! shared-KV layers), and the engine's greedy output is the model's own greedy continuation
//! whatever the batching, the prefix cache (resuming at cached blocks, no saved state) and
//! speculative decoding do.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::sampling::Rng;
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::model::gemma4::tiny;
use legion::spec::{SpecConfig, SpecCounters, SpecKind, TokenDrafter, TokenDrafterFactory};
use legion::tokenizer::{TokenKind, Tokenizer};
use tokio_util::sync::CancellationToken;

const K: usize = 3;
const MAX_TOKENS: u32 = 40;

fn prompts() -> Vec<String> {
    vec![
        // Long enough to share KV blocks (64 positions) with the next one.
        "The quick brown fox jumps over the lazy dog. ".repeat(4),
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "sliding windows".into(),
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
fn tiny_model_loads_as_gemma4() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 1).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let hp = &model.hp;
    let g = legion::model::gemma4::params(hp).expect("Gemma 4 hyperparameters");
    let n = tiny::PATTERN.len();
    assert_eq!(hp.n_layer, n);
    assert_eq!(g.window, tiny::WINDOW as usize);
    assert_eq!(g.n_layer_kv, n - tiny::SHARED);
    assert!(hp.moe && !hp.has_recurrent());
    // Sliding-window layers: 128-wide heads, 2 kv heads; global: 256-wide, 1.
    assert_eq!(hp.kv_dims(0), (128, 2));
    assert_eq!(hp.kv_dims(1), (256, 1));
    // The shared layers read the last cached layer of their kind.
    let sources: Vec<usize> = (0..n).map(|il| hp.kv_source(il)).collect();
    assert_eq!(sources, [0, 1, 2, 3, 2, 2, 3]);
    assert!((0..n).all(|il| hp.owns_kv(il) == (il < g.n_layer_kv)));
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let logits = model
        .prompt_logits(&mut backend, &tok.encode("hello", false))
        .unwrap();
    assert_eq!(logits.len(), hp.n_vocab as usize);
    // Soft-capped.
    assert!(logits.iter().all(|x| x.abs() < 30.0), "{logits:?}");
}

/// Prefill in chunks, then one token per step, against one pass over the whole text:
/// bit-identical logits at every position (reference kernels are batch-invariant), well past
/// the sliding window.
#[test]
fn tiny_decode_matches_full_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 2).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let n_vocab = model.hp.n_vocab as usize;
    let prompt: Vec<u32> = "Sliding windows, shared caches and experts"
        .bytes()
        .map(u32::from)
        .collect();
    assert!(prompt.len() > 4 * tiny::WINDOW as usize);
    let mut state = model.new_state(backend.buffer_type(), 256).unwrap();
    let mut alloc = None;
    let split = prompt.len() / 3;
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

/// A token far past the sliding window still reaches the last position, through the global
/// layers.
#[test]
fn tiny_window_and_global_layers_both_matter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 4).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mut backend = cpu();
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let a: Vec<u32> = "abcdefghijklmnopqrstuvwxyz"
        .bytes()
        .map(u32::from)
        .collect();
    let mut b = a.clone();
    b[0] = u32::from(b'Z');
    let la = model.prompt_logits(&mut backend, &a).unwrap();
    let lb = model.prompt_logits(&mut backend, &b).unwrap();
    assert!(la.iter().all(|x| x.is_finite()));
    assert_ne!(la, lb, "the global layers must see the first token");
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
    // No recurrent state: a cached prompt resumes at its last cached block, no saved state.
    assert!(plain.stats().prefix_cache_hit_tokens > 0, "no prefix reuse");

    // The MTP assistant (a GGUF of its own, reading the target's cache) drafts; whatever it
    // proposes, the output stays.
    let assistant = dir.path().join("assistant.gguf");
    tiny::write_assistant(&assistant, 7).unwrap();
    let e = engine(
        &path,
        SpecConfig {
            kind: SpecKind::Mtp,
            k: 2,
            p_min: 0.0,
            draft: Some(assistant),
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

/// The tiny model on the CUDA backend (`--features cuda`; ignored: needs a GPU): paged
/// attention (the kernels, at this model's head sizes, sliding window, unit scale and shared
/// caches) against the gather path on the same GPU, and both against the CPU.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_tiny_matches_cpu() {
    use legion::ggml::{DeviceKind, device_by_kind};
    use legion::model::core::{Attention, Placement};
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
    let on_cpu = Model::load(&gguf, |_| cpu.buffer_type()).unwrap();
    let mut on_gpu = Model::load(&gguf, |i| {
        if Model::placement(&i.name, false) == Placement::Host {
            cpu.buffer_type()
        } else {
            gpu.buffer_type()
        }
    })
    .unwrap();
    // Past the window, in two chunks then decode steps, so prefill and decode kernels both run.
    let ids: Vec<u32> = "Sliding windows, shared caches and experts, paged in place."
        .bytes()
        .map(u32::from)
        .collect();
    let run = |model: &Model, backend: &mut Backend| -> Vec<f32> {
        let mut state = model.new_state(backend.buffer_type(), 256).unwrap();
        let mut alloc = None;
        let mut rows = Vec::new();
        let (a, b) = ids.split_at(ids.len() / 2);
        model
            .forward(backend, &mut alloc, &mut state, a, false)
            .unwrap();
        rows.extend(
            model
                .forward(backend, &mut alloc, &mut state, &b[..b.len() - 4], true)
                .unwrap(),
        );
        for &t in &b[b.len() - 4..] {
            rows.extend(
                model
                    .forward(backend, &mut alloc, &mut state, &[t], false)
                    .unwrap(),
            );
        }
        rows
    };
    let max_diff = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max)
    };
    let reference = run(&on_cpu, &mut cpu);
    on_gpu.set_attention(Attention::Paged);
    let paged = run(&on_gpu, &mut gpu);
    on_gpu.set_attention(Attention::Gather);
    let gathered = run(&on_gpu, &mut gpu);
    let (pg, pc) = (max_diff(&paged, &gathered), max_diff(&paged, &reference));
    println!("paged vs gather (CUDA): max |diff| {pg:.2e}; paged CUDA vs CPU: {pc:.2e}");
    assert!(pg < 0.05, "paged and gather attention differ by {pg}");
    assert!(pc < 0.5, "CUDA and the CPU differ by {pc}");
}
