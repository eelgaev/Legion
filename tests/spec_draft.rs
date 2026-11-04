//! `--spec draft` on the CPU with tiny random models and reference kernels: a draft model of
//! the target's vocabulary (a different random model, or the target itself) never changes the
//! greedy output; a vocabulary mismatch is refused; and the drafter's state rolls back after
//! partial acceptance (its drafts equal those of a fresh run of the draft model over the
//! sequence's true history).

use std::path::{Path, PathBuf};

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine, LoadError};
use legion::engine::sampling::{Rng, argmax};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::model::qwen35::tiny::{self, TinyConfig};
use legion::spec::TargetInfo;
use legion::spec::draft::{DraftModel, check_same_vocab};
use legion::spec::{Draft, DraftRequest, ModelDrafter, SpecConfig, SpecCounters, SpecKind};
use tokio_util::sync::CancellationToken;

const K: usize = 4;
const MAX_TOKENS: u32 = 48;
const CTX: usize = 512;

fn prompts() -> Vec<String> {
    vec![
        "The quick brown fox jumps over the lazy dog. ".repeat(4), // shares blocks with the next
        "The quick brown fox jumps over the lazy dog. ".repeat(4) + "Again!",
        "speculative decoding".into(),
        "0123456789".repeat(3),
        "abc".into(),
    ]
}

struct Models {
    _dir: tempfile::TempDir,
    target: PathBuf,
    /// Another random model of the same vocabulary, narrower.
    draft: PathBuf,
    /// The target's architecture over a larger vocabulary.
    other_vocab: PathBuf,
}

fn models() -> Models {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.gguf");
    let draft = dir.path().join("draft.gguf");
    let other_vocab = dir.path().join("other-vocab.gguf");
    tiny::write(&target, &TinyConfig::default()).unwrap();
    tiny::write(
        &draft,
        &TinyConfig {
            n_embd: 32,
            n_ff: 64,
            seed: 7,
            ..Default::default()
        },
    )
    .unwrap();
    tiny::write(
        &other_vocab,
        &TinyConfig {
            extra_tokens: 5,
            ..Default::default()
        },
    )
    .unwrap();
    Models {
        _dir: dir,
        target,
        draft,
        other_vocab,
    }
}

fn config(target: &Path, spec: SpecConfig) -> LmConfig {
    LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: CTX,
        threads: 4,
        max_seqs: 3,
        // Small enough that a prompt prefills in several steps next to decoding sequences.
        max_batch_tokens: 40,
        spec,
        use_ref: true,
        ..LmConfig::new(target, DeviceChoice::Cpu)
    }
}

fn engine(target: &Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(config(target, spec)).unwrap()
}

fn draft_spec(draft: &Path) -> SpecConfig {
    SpecConfig {
        kind: SpecKind::Draft,
        k: K,
        draft: Some(draft.into()),
        ..Default::default()
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
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
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

fn totals(replies: &[Reply]) -> SpecCounters {
    let mut c = SpecCounters::default();
    for r in replies {
        c.add(&r.spec.expect("speculation counts"));
    }
    c
}

fn same_output(plain: &[Reply], other: &[Reply], what: &str) {
    for (i, (a, b)) in plain.iter().zip(other).enumerate() {
        assert_eq!(
            (&a.text, a.reason, a.tokens),
            (&b.text, b.reason, b.tokens),
            "{what}: prompt {i} differs"
        );
    }
}

/// A different small random model drafts: mostly wrong, and the output is the plain output
/// all the same, through prefix-cache hits and batches of different draft lengths.
#[tokio::test(flavor = "multi_thread")]
async fn draft_model_speculation_matches_plain_decoding() {
    let m = models();
    let plain = run_all(&engine(&m.target, SpecConfig::default())).await;
    assert!(plain.iter().any(|r| r.tokens > 10), "{plain:?}");

    let e = engine(&m.target, draft_spec(&m.draft));
    // Twice: the second round's prompts resume from the prefix cache, which the draft model
    // never saw and has to catch up on.
    for round in 0..2 {
        let replies = run_all(&e).await;
        same_output(&plain, &replies, &format!("round {round}"));
        let c = totals(&replies);
        println!("round {round}: {c:?}");
        assert!(c.drafted > 0, "nothing drafted");
    }
    let stats = e.stats();
    assert!(stats.spec.drafted > 0);
    assert!(stats.prefix_cache_hit_tokens > 0, "no prefix reuse");
}

/// The target as its own draft model drafts exactly what it will pick.
#[tokio::test(flavor = "multi_thread")]
async fn the_target_as_its_own_drafter_is_always_accepted() {
    let m = models();
    let plain = run_all(&engine(&m.target, SpecConfig::default())).await;
    let e = engine(&m.target, draft_spec(&m.target));
    for round in 0..2 {
        let replies = run_all(&e).await;
        same_output(&plain, &replies, &format!("round {round}"));
        let c = totals(&replies);
        println!("round {round}: {c:?}");
        assert!(c.drafted > 0);
        assert_eq!(c.accepted, c.drafted, "round {round}");
        assert_eq!(c.acceptance_rate(), Some(1.0));
    }
}

/// A draft model of another vocabulary is refused, naming both files.
#[test]
fn a_vocabulary_mismatch_is_refused() {
    let m = models();
    let err = check_same_vocab(&m.target, &m.other_vocab).unwrap_err();
    assert!(
        err.contains("target.gguf") && err.contains("other-vocab.gguf"),
        "{err}"
    );
    assert!(
        err.contains("259 tokens") && err.contains("264 tokens"),
        "{err}"
    );
    check_same_vocab(&m.target, &m.draft).unwrap();
    check_same_vocab(&m.target, &m.target).unwrap();

    // Through the engine: it does not load.
    let Err(e) = LmEngine::load(config(&m.target, draft_spec(&m.other_vocab))) else {
        panic!("a draft model of another vocabulary loaded");
    };
    let LoadError::Device(msg) = &e else {
        panic!("unexpected error {e:?}");
    };
    assert!(
        msg.contains("target.gguf") && msg.contains("other-vocab.gguf"),
        "{msg}"
    );
    // And without a path.
    let cfg = SpecConfig {
        draft: None,
        ..draft_spec(&m.draft)
    };
    assert!(LmEngine::load(config(&m.target, cfg)).is_err());
}

/// The draft model's greedy continuation of `history`, computed on its own (one sequence,
/// plain forward passes).
fn reference(model: &Model, backend: &mut Backend, history: &[u32], n: usize) -> Vec<u32> {
    let mut state = model.new_state(backend.buffer_type(), CTX + 16).unwrap();
    let mut alloc = None;
    let mut logits = model
        .forward(backend, &mut alloc, &mut state, history, false)
        .unwrap();
    let mut out = Vec::new();
    for _ in 0..n {
        let t = argmax(&logits) as u32;
        out.push(t);
        logits = model
            .forward(backend, &mut alloc, &mut state, &[t], false)
            .unwrap();
    }
    out
}

/// One simulated sequence of the rollback test.
struct Sim {
    id: u64,
    history: Vec<u32>,
    /// Positions the drafter has been told about.
    told: usize,
    /// Drafts of the last call.
    drafts: Vec<u32>,
}

/// Drives the drafter as the engine does, with random acceptance, and compares every draft
/// with the draft model run from scratch over the true history: whatever the previous drafts
/// left in the drafter's state must not show.
#[test]
fn drafts_follow_the_true_history_after_partial_acceptance() {
    let m = models();
    let info = TargetInfo {
        n_layer: 4,
        n_embd: 64,
        n_vocab: 259,
        model: m.target.clone(),
        draft: Some(m.draft.clone()),
        gpu: false,
        threads: 4,
        ctx_size: CTX,
        max_seqs: 4,
        use_ref: true,
    };
    let mut drafter = DraftModel::load(&info, K).unwrap();
    // Small chunks: the late start below catches up over several calls.
    drafter.set_feed_chunk(7);

    let gguf = GgufFile::open(&m.draft).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    backend.set_use_ref(true);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();

    let mut rng = Rng::new(99);
    let tok = |rng: &mut Rng| (rng.next_u64() % 259) as u32;
    // Prompt lengths differ; sequence 3 starts at position 10 (as if from the prefix cache).
    let lens = [20usize, 9, 33, 25];
    let mut sims: Vec<Sim> = lens
        .iter()
        .enumerate()
        .map(|(i, &n)| Sim {
            id: 100 + i as u64,
            history: (0..n + 1).map(|_| tok(&mut rng)).collect(),
            told: if i == 3 { 10 } else { 0 },
            drafts: Vec::new(),
        })
        .collect();
    let (mut rounds_partial, mut rounds_full, mut rounds_none) = (0, 0, 0);

    for round in 0..14 {
        // The call after a target step: `ran` positions from `told` on (the first call: the
        // prompt, at once), the history ending with the target's pick.
        let mut outs: Vec<Draft> = sims.iter().map(|_| Draft::default()).collect();
        let mut max = vec![K; sims.len()];
        // Different draft lengths wanted; sequence 1 only takes in a prefill chunk first.
        max[round % sims.len()] = round % 3;
        let mut batch: Vec<DraftRequest<'_>> = Vec::new();
        for ((s, out), &max) in sims.iter().zip(outs.iter_mut()).zip(&max) {
            let ran = s.history.len() - 1 - s.told;
            batch.push(DraftRequest {
                seq: s.id,
                tokens: &s.history,
                start: s.told,
                ran,
                hidden: Vec::new(),
                max,
                out,
            });
        }
        drafter.step(&mut batch).unwrap();
        drop(batch);

        for ((s, out), &max) in sims.iter_mut().zip(outs).zip(&max) {
            s.told = s.history.len() - 1;
            let live = drafter.live_tokens(s.id).unwrap();
            if out.tokens.is_empty() {
                // Only catching up (no drafts wanted, or still behind): the live state is
                // never ahead of the history.
                assert!(live <= s.history.len(), "round {round}");
                s.drafts.clear();
            } else {
                assert_eq!(live, s.history.len(), "round {round}");
                assert!(out.tokens.len() <= max && out.probs.is_empty());
                let want = reference(&model, &mut backend, &s.history, out.tokens.len());
                assert_eq!(out.tokens, want, "round {round}, seq {}", s.id);
                s.drafts = out.tokens.clone();
            }
        }

        // The target's verdict: accept a random prefix of the drafts and pick another token
        // (not the next draft); a sequence with no drafts just decodes one token.
        for s in sims.iter_mut() {
            let accepted = if s.drafts.is_empty() {
                0
            } else {
                (rng.next_u64() % (s.drafts.len() as u64 + 1)) as usize
            };
            match (accepted, s.drafts.len()) {
                (0, 1..) => rounds_none += 1,
                (a, n) if a == n && n > 0 => rounds_full += 1,
                (_, 1..) => rounds_partial += 1,
                _ => {}
            }
            // `history` ends with `last`, fed at `told`; the accepted drafts follow it.
            let mut pick = tok(&mut rng);
            if accepted < s.drafts.len() {
                while pick == s.drafts[accepted] {
                    pick = tok(&mut rng);
                }
            }
            s.history.extend(&s.drafts[..accepted]);
            s.history.push(pick);
        }
    }
    println!("partial {rounds_partial}, full {rounds_full}, none {rounds_none}");
    assert!(
        rounds_partial > 0 && rounds_none > 0,
        "rolled back too rarely"
    );

    // Releasing frees the slots and blocks for new sequences.
    for s in &sims {
        drafter.release(s.id);
    }
    assert_eq!(drafter.followed(), 0);
}
