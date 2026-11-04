//! Speculative decoding through the Qwen3.5 engine on the CPU, with the tiny random model and
//! reference kernels: greedy output is bit-identical to the same engine without speculation,
//! with oracle, adversarial and random drafters, several sequences batched (drafts of
//! different lengths padded into one verify group, prefill chunks alongside), penalties, and
//! prompts that share prefix-cache blocks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::sampling::Rng;
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::{
    Draft, DraftRequest, ModelDrafter, SpecConfig, SpecCounters, TargetInfo, TokenDrafter,
    TokenDrafterFactory,
};
use tokio_util::sync::CancellationToken;

const K: usize = 4;
const MAX_TOKENS: u32 = 48;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4), // shares blocks with the next
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "speculative decoding".into(),
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
    run_all_with(engine, sampling, |_| {}).await
}

async fn run_all_with(
    engine: &LmEngine,
    sampling: SamplingParams,
    tweak: impl Fn(&mut GenerateRequest),
) -> Vec<Reply> {
    run_prompts(engine, &prompts(), sampling, tweak).await
}

async fn run_prompts(
    engine: &LmEngine,
    prompts: &[String],
    sampling: SamplingParams,
    tweak: impl Fn(&mut GenerateRequest),
) -> Vec<Reply> {
    let streams: Vec<_> = prompts
        .iter()
        .map(|p| {
            let mut r = request(p, sampling.clone());
            tweak(&mut r);
            engine.generate(r).unwrap()
        })
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

/// The greedy continuation of every prompt, computed without the engine.
fn truths(path: &std::path::Path) -> Vec<(Vec<u32>, Vec<u32>)> {
    let gguf = GgufFile::open(path).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    backend.set_use_ref(true);
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
                let t = legion::engine::sampling::argmax(&logits) as u32;
                out.push(t);
                logits = model
                    .forward(&mut backend, &mut alloc, &mut state, &[t], false)
                    .unwrap();
            }
            (prompt, out)
        })
        .collect()
}

/// The true greedy continuation after `tokens` (a prompt plus generated tokens), or nothing
/// once `tokens` left it.
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

/// How a test drafter picks its drafts from the true continuation.
#[derive(Clone, Copy)]
enum Mode {
    Oracle,
    Adversary,
    /// Each draft right with this probability, random otherwise, and a random count.
    Noisy(f64),
}

struct TestDrafter {
    truths: Arc<Vec<(Vec<u32>, Vec<u32>)>>,
    mode: Mode,
    rng: Rng,
    /// The tokens `accepted` reported, to check against the history `propose` gets.
    seen: Vec<u32>,
    /// The `max` of every `propose` call: the depth the engine's policy allowed that step.
    log: Arc<Mutex<Vec<usize>>>,
}

impl TokenDrafter for TestDrafter {
    fn propose(&mut self, tokens: &[u32], max: usize, out: &mut Vec<u32>) {
        // The history ends with what `accepted` reported.
        assert!(
            tokens.ends_with(&self.seen),
            "history and accepted disagree"
        );
        self.log.lock().unwrap().push(max);
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
                Mode::Noisy(p) => {
                    if self.rng.next_f64() < p {
                        right
                    } else {
                        (self.rng.next_u64() % 259) as u32
                    }
                }
            });
        }
    }

    fn accepted(&mut self, appended: &[u32]) {
        assert!(!appended.is_empty());
        self.seen = appended.to_vec();
    }
}

/// Each sequence's drafter logs its `propose` calls: one entry per sequence, in creation order.
type DepthLogs = Arc<Mutex<Vec<Arc<Mutex<Vec<usize>>>>>>;

fn spec(truths: &Arc<Vec<(Vec<u32>, Vec<u32>)>>, mode: Mode) -> SpecConfig {
    spec_logged(truths, mode).0
}

fn spec_logged(truths: &Arc<Vec<(Vec<u32>, Vec<u32>)>>, mode: Mode) -> (SpecConfig, DepthLogs) {
    let truths = truths.clone();
    let seed = Mutex::new(0u64);
    let logs = DepthLogs::default();
    let shared = logs.clone();
    let factory: TokenDrafterFactory = Arc::new(move || {
        let mut s = seed.lock().unwrap();
        *s += 1;
        let log = Arc::new(Mutex::new(Vec::new()));
        shared.lock().unwrap().push(log.clone());
        Box::new(TestDrafter {
            truths: truths.clone(),
            mode,
            rng: Rng::new(*s),
            seen: Vec::new(),
            log,
        })
    });
    let cfg = SpecConfig {
        k: K,
        token_drafter: Some(factory),
        ..Default::default()
    };
    (cfg, logs)
}

fn totals(replies: &[Reply]) -> SpecCounters {
    let mut c = SpecCounters::default();
    for r in replies {
        c.add(&r.spec.expect("speculation counts"));
    }
    c
}

#[tokio::test(flavor = "multi_thread")]
async fn greedy_speculation_matches_plain_decoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));

    let plain = run_all(&engine(&path, SpecConfig::default()), greedy()).await;
    assert!(plain.iter().all(|r| r.spec.is_none()));
    assert!(plain.iter().any(|r| r.tokens > 10), "{plain:?}");

    for (name, mode) in [
        ("oracle", Mode::Oracle),
        ("adversary", Mode::Adversary),
        ("random", Mode::Noisy(0.0)),
        ("partly right", Mode::Noisy(0.7)),
    ] {
        let e = engine(&path, spec(&truths, mode));
        // Twice: the second round's prompts resume from the prefix cache, so they share
        // blocks the first round's verify steps must have left intact.
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
            match mode {
                Mode::Oracle => assert_eq!(c.accepted, c.drafted, "{name}"),
                Mode::Adversary => assert_eq!(c.accepted, 0, "{name}"),
                Mode::Noisy(p) if p > 0.0 => {
                    assert!(c.accepted > 0 && c.accepted < c.drafted, "{name}")
                }
                Mode::Noisy(_) => {}
            }
        }
        let stats = e.stats();
        assert!(stats.spec.drafted > 0);
        assert!(stats.prefix_cache_hit_tokens > 0, "no prefix reuse");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn penalties_apply_per_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    // Greedy with penalties: every row is a full logits row, penalized by the tokens accepted
    // before it in the same verify step.
    let sampling = SamplingParams {
        temperature: 0.0,
        presence_penalty: 0.4,
        frequency_penalty: 0.3,
        repetition_penalty: 1.3,
        ..Default::default()
    };
    let plain = run_all(&engine(&path, SpecConfig::default()), sampling.clone()).await;
    let replies = run_all(&engine(&path, spec(&truths, Mode::Noisy(0.8))), sampling).await;
    assert_eq!(
        plain
            .iter()
            .map(|r| (&r.text, r.tokens))
            .collect::<Vec<_>>(),
        replies
            .iter()
            .map(|r| (&r.text, r.tokens))
            .collect::<Vec<_>>()
    );
    assert!(totals(&replies).accepted > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn sampled_speculation_runs_to_completion() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let sampling = SamplingParams {
        temperature: 0.7,
        top_k: 20,
        seed: Some(3),
        logprobs: Some(2),
        ..Default::default()
    };
    let e = engine(&path, spec(&truths, Mode::Oracle));
    let a = run_all(&e, sampling.clone()).await;
    let b = run_all(&e, sampling).await;
    // Seeded: the same text again (the draws are per sequence, the scheduling irrelevant).
    assert_eq!(a, b);
    for r in &a {
        let c = r.spec.unwrap();
        assert!(c.accepted <= c.drafted);
        assert!(r.tokens as u64 >= c.accepted + c.steps || r.reason != FinishReason::Length);
    }
}

/// A model drafter that checks what the engine hands it: every position of every sequence
/// exactly once (prefill chunks and accepted verify positions; rejected ones rolled back), with
/// one hidden row per position and layer. It proposes the true continuation.
struct HiddenChecker {
    info: TargetInfo,
    truths: Arc<Vec<(Vec<u32>, Vec<u32>)>>,
    /// Per sequence: positions taken in so far.
    next: HashMap<u64, usize>,
    released: Arc<Mutex<usize>>,
}

impl ModelDrafter for HiddenChecker {
    fn hidden_layers(&self) -> Vec<usize> {
        vec![self.info.n_layer - 1, 0]
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        for r in batch.iter_mut() {
            // A sequence starts where its prompt left the prefix cache.
            let next = self.next.entry(r.seq).or_insert(r.start);
            assert_eq!(
                r.start, *next,
                "seq {}: positions skipped or repeated",
                r.seq
            );
            *next = r.start + r.ran;
            assert!(r.start + r.ran <= r.tokens.len());
            assert_eq!(r.hidden.len(), 2);
            for h in &r.hidden {
                assert_eq!(h.len(), r.ran * self.info.n_embd);
                assert!(h.iter().all(|x| x.is_finite()));
            }
            let truth = truth_after(&self.truths, r.tokens);
            *r.out = Draft {
                tokens: truth.iter().take(r.max).copied().collect(),
                ..Default::default()
            };
        }
        Ok(())
    }

    fn release(&mut self, seq: u64) {
        self.next.remove(&seq);
        *self.released.lock().unwrap() += 1;
    }
}

fn hidden_spec(
    truths: &Arc<Vec<(Vec<u32>, Vec<u32>)>>,
    released: &Arc<Mutex<usize>>,
) -> SpecConfig {
    let (t, r) = (truths.clone(), released.clone());
    SpecConfig {
        k: K,
        model_drafter: Some(Arc::new(move |info: &TargetInfo| {
            Ok(Box::new(HiddenChecker {
                info: info.clone(),
                truths: t.clone(),
                next: HashMap::new(),
                released: r.clone(),
            }) as Box<dyn ModelDrafter>)
        })),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn model_drafters_get_every_kept_position_with_hidden_states() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let released = Arc::new(Mutex::new(0));
    let cfg = hidden_spec(&truths, &released);
    let plain = run_all(&engine(&path, SpecConfig::default()), greedy()).await;
    let e = engine(&path, cfg);
    let replies = run_all(&e, greedy()).await;
    for (a, b) in plain.iter().zip(&replies) {
        assert_eq!((&a.text, a.tokens), (&b.text, b.tokens));
    }
    let c = totals(&replies);
    assert!(c.drafted > 0 && c.accepted == c.drafted, "{c:?}");
    drop(e);
    assert_eq!(*released.lock().unwrap(), prompts().len());
}

/// `--spec ngram` and `--spec suffix`: the real model-free drafters, wired by kind.
#[tokio::test(flavor = "multi_thread")]
async fn ngram_and_suffix_drafters_keep_greedy_output() {
    use legion::spec::SpecKind;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let plain = run_all(&engine(&path, SpecConfig::default()), greedy()).await;
    for kind in [SpecKind::Ngram, SpecKind::Suffix] {
        let e = engine(
            &path,
            SpecConfig {
                kind,
                k: K,
                ..Default::default()
            },
        );
        // Twice: the suffix drafter's cache then holds the first round's sequences.
        for round in 0..2 {
            let replies = run_all(&e, greedy()).await;
            for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
                assert_eq!(
                    (&a.text, a.tokens),
                    (&b.text, b.tokens),
                    "{kind:?}, round {round}: prompt {i}"
                );
            }
            println!("{kind:?}, round {round}: {:?}", totals(&replies));
        }
        assert!(e.stats().spec.drafted > 0, "{kind:?} never drafted");
    }
}

/// `min_tokens` masks end tokens per verify row, and `ignore_eos` lets drafts run through
/// them, as in plain decoding.
#[tokio::test(flavor = "multi_thread")]
async fn min_tokens_and_ignore_eos_apply_per_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let plain_engine = engine(&path, SpecConfig::default());
    let spec_engine = engine(&path, spec(&truths, Mode::Noisy(0.7)));
    type Tweak = fn(&mut GenerateRequest);
    let tweaks: [(&str, Tweak); 2] = [
        ("min_tokens", |r| r.min_tokens = 30),
        ("ignore_eos", |r| r.ignore_eos = true),
    ];
    for (name, tweak) in tweaks {
        let plain = run_all_with(&plain_engine, greedy(), tweak).await;
        let replies = run_all_with(&spec_engine, greedy(), tweak).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "{name}: prompt {i}"
            );
            assert!(a.tokens >= 30, "{name}: prompt {i} stopped at {}", a.tokens);
        }
    }
}

/// One sequence that always generates `MAX_TOKENS` tokens, so its steps are countable.
fn lone_prompt() -> Vec<String> {
    vec!["speculative decoding".into()]
}

fn ignore_eos(r: &mut GenerateRequest) {
    r.ignore_eos = true;
}

fn same_output(plain: &[Reply], replies: &[Reply], what: &str) {
    assert_eq!(plain.len(), replies.len());
    for (i, (a, b)) in plain.iter().zip(replies).enumerate() {
        assert_eq!(
            (&a.text, a.reason, a.tokens),
            (&b.text, b.reason, b.tokens),
            "{what}: prompt {i} differs"
        );
    }
}

/// An adversarial drafter is never right: after two rejected drafts the policy drafts nothing
/// but a one-token probe every 16th step, and the output is unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn adaptive_depth_backs_off_from_an_adversarial_drafter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let prompts = lone_prompt();
    let plain = run_prompts(
        &engine(&path, SpecConfig::default()),
        &prompts,
        greedy(),
        ignore_eos,
    )
    .await;
    assert_eq!(plain[0].tokens, MAX_TOKENS);

    let (cfg, logs) = spec_logged(&truths, Mode::Adversary);
    let e = engine(&path, cfg);
    let replies = run_prompts(&e, &prompts, greedy(), ignore_eos).await;
    same_output(&plain, &replies, "adversary");

    assert_eq!(logs.lock().unwrap().len(), 1);
    let depths = logs.lock().unwrap()[0].lock().unwrap().clone();
    assert!(depths.len() > 33, "{depths:?}");
    // Alpha 0.7 allows one draft, 0.56 one more, then 0.45 none: from there only probes.
    let drafting: Vec<usize> = (0..depths.len()).filter(|&i| depths[i] > 0).collect();
    assert_eq!(drafting, vec![0, 1, 17, 33], "{depths:?}");
    assert!(drafting.iter().all(|&i| depths[i] == 1), "{depths:?}");
    let spec = replies[0].spec.unwrap();
    assert_eq!((spec.drafted, spec.accepted), (4, 0));
    assert_eq!(e.stats().spec_mean_k, 0.0, "the last step drafted nothing");

    // `--spec-fixed` drafts the full depth on every step instead.
    let fixed = SpecConfig {
        fixed: true,
        ..spec_logged(&truths, Mode::Adversary).0
    };
    let replies = run_prompts(&engine(&path, fixed), &prompts, greedy(), ignore_eos).await;
    same_output(&plain, &replies, "fixed adversary");
    let spec = replies[0].spec.unwrap();
    assert!(spec.drafted > 4 * 40, "{spec:?}");
}

/// A perfect drafter ramps up (alpha starts at 0.7) and then drafts `k_max` on every step,
/// until fewer tokens than that remain to emit.
#[tokio::test(flavor = "multi_thread")]
async fn adaptive_depth_reaches_k_max_with_an_oracle_drafter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let prompts = lone_prompt();
    let plain = run_prompts(
        &engine(&path, SpecConfig::default()),
        &prompts,
        greedy(),
        ignore_eos,
    )
    .await;

    let (cfg, logs) = spec_logged(&truths, Mode::Oracle);
    let e = engine(&path, cfg);
    let replies = run_prompts(&e, &prompts, greedy(), ignore_eos).await;
    same_output(&plain, &replies, "oracle");

    let depths = logs.lock().unwrap()[0].lock().unwrap().clone();
    assert_eq!(depths[..4], [1, 2, 3, K], "{depths:?}");
    let full = depths[3..].iter().take_while(|&&d| d == K).count();
    assert!(full >= 5, "{depths:?}");
    assert!(
        depths[3 + full..].iter().all(|&d| d < K),
        "only the last steps, short of tokens, draft less: {depths:?}"
    );
    let spec = replies[0].spec.unwrap();
    assert_eq!(spec.accepted, spec.drafted);
    assert_eq!(e.stats().spec_mean_k, K as f64);
}

/// `--spec-max-batch 1`: two concurrent sequences, and even a lone one, never draft; with 2 a
/// lone sequence drafts. Output is the plain output every time, for token and model drafters.
#[tokio::test(flavor = "multi_thread")]
async fn batch_gate_stops_drafting_when_compute_bound() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, &TinyConfig::default()).unwrap();
    let truths = Arc::new(truths(&path));
    let pair = vec![prompts()[0].clone(), prompts()[2].clone()];
    let plain_engine = engine(&path, SpecConfig::default());
    let plain = run_prompts(&plain_engine, &pair, greedy(), ignore_eos).await;
    let plain_lone = run_prompts(&plain_engine, &lone_prompt(), greedy(), ignore_eos).await;

    let gated = SpecConfig {
        max_batch: 1,
        ..spec(&truths, Mode::Oracle)
    };
    let e = engine(&path, gated);
    let replies = run_prompts(&e, &pair, greedy(), ignore_eos).await;
    same_output(&plain, &replies, "gated pair");
    assert!(replies.iter().all(|r| r.tokens > 10), "{replies:?}");
    let stats = e.stats();
    assert_eq!(stats.spec.drafted, 0);
    assert_eq!(stats.spec_mean_k, 0.0);
    assert_eq!(totals(&replies).drafted, 0);
    let replies = run_prompts(&e, &lone_prompt(), greedy(), ignore_eos).await;
    same_output(&plain_lone, &replies, "gated lone");
    assert_eq!(e.stats().spec.drafted, 0);

    // The same oracle drafter under a gate of 2 drafts when alone in the batch.
    let open = SpecConfig {
        max_batch: 2,
        ..spec(&truths, Mode::Oracle)
    };
    let e = engine(&path, open);
    let replies = run_prompts(&e, &lone_prompt(), greedy(), ignore_eos).await;
    same_output(&plain_lone, &replies, "open lone");
    assert!(e.stats().spec.drafted > 0);

    // Model drafters: the policy picks the depth after each step, the gate applies again
    // when the drafts are verified.
    let released = Arc::new(Mutex::new(0));
    let model = SpecConfig {
        max_batch: 1,
        ..hidden_spec(&truths, &released)
    };
    let e = engine(&path, model);
    let replies = run_prompts(&e, &pair, greedy(), ignore_eos).await;
    same_output(&plain, &replies, "gated model drafter");
    assert_eq!(e.stats().spec.drafted, 0);
    let model = SpecConfig {
        max_batch: 2,
        ..hidden_spec(&truths, &released)
    };
    let e = engine(&path, model);
    let replies = run_prompts(&e, &lone_prompt(), greedy(), ignore_eos).await;
    same_output(&plain_lone, &replies, "open model drafter");
    let c = totals(&replies);
    assert!(c.drafted > 0 && c.accepted == c.drafted, "{c:?}");
}
