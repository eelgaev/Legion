//! MTP speculative decoding on the tiny random qwen4exp (hyper-connections, per-layer
//! embedding, sparse attention) with its MTP head as a GGUF of its own, on the CPU with
//! reference kernels: greedy output with `--spec mtp` is bit-identical to plain decoding, for
//! every draft length, several sequences batched; and the head's drafts are what it computes
//! over a whole history at once (its cache takes in the kept positions correctly).

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{Engine, FinishReason, GenerateRequest, Prompt, SamplingParams, TokenEvent};
use legion::spec::{SpecConfig, SpecCounters, SpecKind};
use tokio_util::sync::CancellationToken;

fn prompts() -> Vec<String> {
    vec![
        "Multi-token prediction drafts the next tokens. ".repeat(3),
        "speculative decoding".into(),
        "0123456789".repeat(5),
        "abc".into(),
    ]
}

fn engine(dir: &std::path::Path, spec: SpecConfig) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        max_batch_tokens: 48,
        spec,
        use_ref: true,
        ..LmConfig::new(dir.join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap()
}

fn mtp(dir: &std::path::Path, k: usize, p_min: f32) -> SpecConfig {
    SpecConfig {
        kind: SpecKind::Mtp,
        k,
        draft: Some(dir.join("mtp.gguf")),
        p_min,
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

async fn run_all(engine: &LmEngine, sampling: SamplingParams) -> Vec<Reply> {
    run_n(engine, sampling, 40).await
}

async fn run_n(engine: &LmEngine, sampling: SamplingParams, max_tokens: u32) -> Vec<Reply> {
    let streams: Vec<_> = prompts()
        .iter()
        .map(|p| {
            engine
                .generate(GenerateRequest {
                    model: "tiny".into(),
                    prompt: Prompt::Text(p.clone()),
                    sampling: sampling.clone(),
                    max_tokens,
                    stop: vec![],
                    // Longer runs go on past an end-of-generation token.
                    ignore_eos: max_tokens > 40,
                    min_tokens: 0,
                    thinking: None,
                    response_format: None,
                    template_kwargs: Default::default(),
                    cancel: CancellationToken::new(),
                    scheduling: Default::default(),
                })
                .unwrap()
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

fn write_models(dir: &std::path::Path) {
    legion::model::qwen4::tiny::write(dir.join("tiny.gguf"), Some(&dir.join("mtp.gguf")), 7)
        .unwrap();
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn mtp_greedy_output_is_bit_identical() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    assert!(plain.iter().all(|r| r.tokens > 5), "{plain:?}");
    for k in 1..=4 {
        let e = engine(dir.path(), mtp(dir.path(), k, 0.0));
        let replies = run_all(&e, greedy()).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "k {k}: prompt {i} differs"
            );
        }
        let s = e.stats().spec;
        println!("k {k}: {s:?}");
        assert!(s.drafted > 0, "k {k}: nothing drafted");
    }
}

/// `--spec-mtp-ngram` (#387): drafts copied from earlier occurrences in the sequence keep the
/// greedy output bit-identical to plain decoding, alone (a head that never drafts: every draft
/// is the n-gram's) and mixed with the head's drafts under the depth policy, also with an
/// n-gram ceiling above the head's (`--spec-mtp-ngram-k`). The random model
/// repeats itself only now and then: 400 tokens past end-of-generation give some accepted
/// n-gram drafts.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_ngram_keeps_greedy_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_n(&engine(dir.path(), SpecConfig::default()), greedy(), 400).await;
    for (name, spec) in [
        (
            "n-gram only",
            SpecConfig {
                fixed: true,
                mtp_ngram: 1,
                ..mtp(dir.path(), 4, 2.0)
            },
        ),
        (
            "adaptive",
            SpecConfig {
                mtp_ngram: 1,
                ..mtp(dir.path(), 4, 0.0)
            },
        ),
        (
            "n-gram deeper than the head",
            SpecConfig {
                mtp_ngram: 1,
                mtp_ngram_k: 8,
                ..mtp(dir.path(), 2, 0.0)
            },
        ),
    ] {
        let e = engine(dir.path(), spec);
        let replies = run_n(&e, greedy(), 400).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "{name}: prompt {i} differs"
            );
        }
        let s = e.stats().spec;
        println!("{name}: {s:?}");
        assert!(s.drafted > 0 && s.accepted > 0, "{name}: {s:?}");
    }
}

/// `--spec-draft-vocab`: a head drafting over the first tokens only proposes other drafts,
/// but verification keeps the output bit-identical to plain decoding.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_draft_vocab_keeps_greedy_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    for vocab in [1, 40] {
        let e = LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 48,
            spec: mtp(dir.path(), 3, 0.0),
            draft_vocab: vocab,
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny.gguf"), DeviceChoice::Cpu)
        })
        .unwrap();
        let replies = run_all(&e, greedy()).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "draft vocab {vocab}: prompt {i} differs"
            );
        }
        let s = e.stats().spec;
        println!("draft vocab {vocab}: {s:?}");
        assert!(s.drafted > 0, "draft vocab {vocab}: nothing drafted");
        if vocab == 1 {
            // One draftable token: most drafts are wrong, so the rejection path runs.
            assert!(s.accepted < s.drafted, "draft vocab 1: {s:?}");
        }
    }
}

/// `--mtp-window`: a head attending to the last 1, 4 or 16 positions (fewer than the prompts
/// hold, so the window binds) drafts otherwise, but with paged attention the output stays
/// bit-identical to plain paged decoding, rejected drafts included.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_window_keeps_greedy_output() {
    use legion::model::core::Attention;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let paged = |spec: SpecConfig, mtp_window: usize| {
        LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 48,
            spec,
            mtp_window: Some(mtp_window),
            attention: Attention::Paged,
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny.gguf"), DeviceChoice::Cpu)
        })
        .unwrap()
    };
    let plain = run_all(&paged(SpecConfig::default(), 0), greedy()).await;
    for window in [1, 4, 16] {
        let e = paged(mtp(dir.path(), 3, 0.0), window);
        let replies = run_all(&e, greedy()).await;
        for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
            assert_eq!(
                (&a.text, a.reason, a.tokens),
                (&b.text, b.reason, b.tokens),
                "mtp window {window}: prompt {i} differs"
            );
        }
        let s = e.stats().spec;
        println!("mtp window {window}: {s:?}");
        assert!(s.drafted > 0, "mtp window {window}: nothing drafted");
        assert!(
            s.accepted < s.drafted,
            "mtp window {window}: no rejection: {s:?}"
        );
    }
}

/// `--mtp-ahead` (#827): a draft round run as a chain whose passes stage the next one (on the
/// CPU nothing queues, so it stages nothing) drafts and outputs exactly what
/// the plain loop of passes does, with drafts stopped by the confidence floor and by the cap.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_ahead_keeps_drafts_and_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let load = |ahead: bool, p_min: f32| {
        LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens: 48,
            // A fixed depth: the depth policy follows measured step times.
            spec: SpecConfig {
                fixed: true,
                ..mtp(dir.path(), 3, p_min)
            },
            switches: legion::engine::switches::Switches {
                mtp_ahead: Some(ahead),
                ..Default::default()
            },
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny.gguf"), DeviceChoice::Cpu)
        })
        .unwrap()
    };
    // Rounds that stop at the cap only; and at the floor too (the random head's picks have
    // probabilities about 0.01).
    for p_min in [0.0, 0.01] {
        let off = run_n(&load(false, p_min), greedy(), 80).await;
        let on = run_n(&load(true, p_min), greedy(), 80).await;
        assert_eq!(off, on, "p_min {p_min}");
        let s: Vec<_> = off.iter().map(|r| r.spec.unwrap_or_default()).collect();
        println!("p_min {p_min}: {s:?}");
        assert!(s.iter().all(|s| s.drafted > 0), "p_min {p_min}: {s:?}");
    }
}

/// The head's cache across passes: taking a prompt in as two chunks and then the bonus token
/// gives the same draft, residual and probability, bit for bit, as one pass over all of it.
/// The first chunk asks for no output, so its pass only writes the head's K/V.
#[test]
fn mtp_passes_chain_through_the_head_cache() {
    use legion::ggml::Backend;
    use legion::gguf::GgufFile;
    use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let new_exec = || {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(4);
        b.set_use_ref(true);
        let mut m = Model::load(
            &GgufFile::open(dir.path().join("tiny.gguf")).unwrap(),
            |_| b.buffer_type(),
        )
        .unwrap();
        m.load_mtp(
            &GgufFile::open(dir.path().join("mtp.gguf")).unwrap(),
            |_| b.buffer_type(),
        )
        .unwrap();
        let n_layer = m.hp.n_layer;
        let mut e = Executor::new(m, b, 4, 1, 0).unwrap();
        e.enable_mtp().unwrap();
        e.capture_hidden(vec![n_layer - 1]);
        e
    };
    let prompt: Vec<u32> = "multi-token prediction".bytes().map(u32::from).collect();
    let blocks = [0u32, 1, 2, 3];
    let group = |tokens, n_past, logits| Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens,
            n_past,
            blocks: &blocks,
            ring: 0,
            logits,
            media: None,
        }],
        verify: false,
    };
    let mut e = new_exec();
    let out = e.step(&[group(&prompt, 0, Logits::Last)]).unwrap();
    let bonus = out.argmax[0];
    let h = &out.hidden[0];
    let w = e.mtp_width();
    assert_eq!(h.len(), prompt.len() * w);
    // Position p pairs token p with the residual at p - 1 (zeros before the start).
    let mut fed = prompt.clone();
    fed.push(bonus);
    let mut rows = vec![0f32; w];
    rows.extend_from_slice(h);
    let whole = e.mtp_step(&[group(&fed, 0, Logits::Last)], &rows).unwrap();

    for cut in [1, 9, fed.len() - 1] {
        let mut e = new_exec();
        e.step(&[group(&prompt, 0, Logits::Last)]).unwrap();
        e.mtp_step(&[group(&fed[..cut], 0, Logits::None)], &rows[..cut * w])
            .unwrap();
        let split = e
            .mtp_step(&[group(&fed[cut..], cut, Logits::Last)], &rows[cut * w..])
            .unwrap();
        assert_eq!(whole.argmax, split.argmax, "cut {cut}");
        assert_eq!(whole.pmax, split.pmax, "cut {cut}");
        assert!(
            whole
                .hidden
                .iter()
                .zip(&split.hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "cut {cut}"
        );
        assert_eq!(whole.hidden.len(), w);
    }
}

/// #289: take-in passes whose row count follows the accepted drafts (here 3, 2, 4, then 3
/// and 2 again) build each shape's graph once, and a kept graph run again gives what a fresh
/// executor's pass over the same history gives, bit for bit.
#[test]
fn mtp_take_in_shapes_keep_their_graphs() {
    use legion::ggml::Backend;
    use legion::gguf::GgufFile;
    use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let new_exec = || {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(4);
        b.set_use_ref(true);
        let mut m = Model::load(
            &GgufFile::open(dir.path().join("tiny.gguf")).unwrap(),
            |_| b.buffer_type(),
        )
        .unwrap();
        m.load_mtp(
            &GgufFile::open(dir.path().join("mtp.gguf")).unwrap(),
            |_| b.buffer_type(),
        )
        .unwrap();
        let n_layer = m.hp.n_layer;
        let mut e = Executor::new(m, b, 4, 1, 0).unwrap();
        e.enable_mtp().unwrap();
        e.capture_hidden(vec![n_layer - 1]);
        e
    };
    let text: Vec<u32> = "kept graphs for every accepted count"
        .bytes()
        .map(u32::from)
        .collect();
    let blocks = [0u32, 1, 2, 3];
    let group = |tokens, n_past| Group {
        slot: 0,
        seqs: vec![SeqInput {
            tokens,
            n_past,
            blocks: &blocks,
            ring: 0,
            logits: Logits::Last,
            media: None,
        }],
        verify: false,
    };
    // The target's residuals over the whole text; the head's position p pairs token p with
    // the residual at p - 1.
    let mut e = new_exec();
    let h = e.step(&[group(&text, 0)]).unwrap().hidden.remove(0);
    let w = e.mtp_width();
    let mut rows = vec![0f32; w];
    rows.extend_from_slice(&h[..(text.len() - 1) * w]);
    // Take-ins of these many rows, one after another.
    let cuts = [8, 3, 2, 4, 3, 2, 3];
    let mut at = 0;
    for n in cuts {
        let kept = e
            .mtp_step(&[group(&text[at..at + n], at)], &rows[at * w..(at + n) * w])
            .unwrap();
        // A fresh executor's head over the same history: all before `at` in one pass, then
        // the same rows.
        let mut fresh = new_exec();
        fresh.step(&[group(&text, 0)]).unwrap();
        if at > 0 {
            fresh
                .mtp_step(&[group(&text[..at], 0)], &rows[..at * w])
                .unwrap();
        }
        let f = fresh
            .mtp_step(&[group(&text[at..at + n], at)], &rows[at * w..(at + n) * w])
            .unwrap();
        assert_eq!((&f.argmax, &f.pmax), (&kept.argmax, &kept.pmax), "at {at}");
        assert!(
            f.hidden
                .iter()
                .zip(&kept.hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "at {at}"
        );
        at += n;
    }
    // One build per shape: 8, 3, 2 and 4 rows.
    assert_eq!(e.mtp_graph_builds(), 4);
}

/// The step buffers the expert cache leaves room for: a full decode batch, and a full verify
/// batch of the same sequences (every row's logits), measured by building their graphs; the
/// executor then still runs steps.
#[test]
fn measures_decode_and_verify_step_buffers() {
    use legion::ggml::Backend;
    use legion::gguf::GgufFile;
    use legion::model::core::{Executor, Group, Logits, Model, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let mut b = Backend::cpu().unwrap();
    b.set_threads(4);
    let m = Model::load(
        &GgufFile::open(dir.path().join("tiny.gguf")).unwrap(),
        |_| b.buffer_type(),
    )
    .unwrap();
    let mut e = Executor::new(m, b, 32, 4, 0).unwrap();
    e.enable_spec(4).unwrap();
    let prompt = e.measure_prompt_step(8, 20).unwrap();
    let decode = e.measure_step(4, 1, 20, false).unwrap();
    let verify = e.measure_step(4, 4, 20, true).unwrap();
    assert_eq!((prompt.len(), decode.len(), verify.len()), (1, 1, 1));
    assert!(decode[0] > 0 && prompt[0] > 0, "{decode:?} {prompt:?}");
    assert!(verify[0] > decode[0], "verify {verify:?} decode {decode:?}");
    let tokens = [1u32, 2, 3];
    let blocks = [0u32, 1];
    let out = e
        .step(&[Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens: &tokens,
                n_past: 0,
                blocks: &blocks,
                ring: 0,
                logits: Logits::Last,
                media: None,
            }],
            verify: false,
        }])
        .unwrap();
    assert_eq!(out.rows.len(), 1);
}

/// `--spec-draft-head-type f16` (#264): F16 copies of the head's BF16 indexer and F32 router
/// drafts otherwise, but verification keeps the output bit-identical to plain decoding.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_f16_head_keeps_greedy_output() {
    use legion::model::core::DraftHeadType;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let plain = run_all(&engine(dir.path(), SpecConfig::default()), greedy()).await;
    let e = LmEngine::load(LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 512,
        threads: 4,
        max_seqs: 3,
        max_batch_tokens: 48,
        spec: mtp(dir.path(), 3, 0.0),
        draft_head_type: DraftHeadType::F16,
        use_ref: true,
        ..LmConfig::new(dir.path().join("tiny.gguf"), DeviceChoice::Cpu)
    })
    .unwrap();
    let replies = run_all(&e, greedy()).await;
    for (i, (a, b)) in plain.iter().zip(&replies).enumerate() {
        assert_eq!(
            (&a.text, a.reason, a.tokens),
            (&b.text, b.reason, b.tokens),
            "f16 head: prompt {i} differs"
        );
    }
    let s = e.stats().spec;
    println!("f16 head: {s:?}");
    assert!(s.drafted > 0, "f16 head: nothing drafted");
}

/// The F16 copies (#264) are the head's BF16 and F32 matrices, rounded to F16 and read in
/// their place; the hyper-connection inject weights stay F32 (the fused mixer norm reads F32
/// ones), the quantized matrices and the target's weights as they are.
#[test]
fn mtp_f16_head_copies_its_bf16_and_f32_matrices() {
    use legion::ggml::{Backend, types};
    use legion::gguf::GgufFile;
    use legion::model::core::Model;
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let b = Backend::cpu().unwrap();
    let mut m = Model::load(
        &GgufFile::open(dir.path().join("tiny.gguf")).unwrap(),
        |_| b.buffer_type(),
    )
    .unwrap();
    m.load_mtp(
        &GgufFile::open(dir.path().join("mtp.gguf")).unwrap(),
        |_| b.buffer_type(),
    )
    .unwrap();
    let il = m.hp.n_layer;
    let name = |s: &str| format!("blk.{il}.{s}");
    let copied = [
        "ffn_gate_inp.weight",
        "indexer.k_proj.weight",
        "indexer.q_proj.weight",
    ];
    let values = |m: &Model, s: &str| {
        let t = m.tensor(&name(s)).unwrap();
        if t.ty() == types::F32 {
            t.to_f32()
        } else {
            legion::ggml::dequantize(t.ty(), &t.bytes(), t.elements())
        }
    };
    let before: Vec<Vec<f32>> = copied.iter().map(|s| values(&m, s)).collect();
    let (n, mib_before, mib_after) = m.requant_mtp(types::F16).unwrap();
    assert_eq!(n, copied.len());
    assert!(mib_after < mib_before, "{mib_before} -> {mib_after}");
    for (s, was) in copied.iter().zip(&before) {
        assert_eq!(m.tensor(&name(s)).unwrap().ty(), types::F16, "{s}");
        let now = values(&m, s);
        let scale = was.iter().fold(0f32, |a, v| a.max(v.abs()));
        for (x, y) in was.iter().zip(&now) {
            assert!((x - y).abs() <= scale / 1024.0, "{s}: {x} -> {y}");
        }
    }
    for (s, ty) in [
        ("hc_attn_inject.weight", types::F32),
        ("hc_ffn_inject.weight", types::F32),
        ("ffn_gate_inp_shexp.weight", types::F32),
        ("nextn.eh_proj.weight", types::Q8_0),
        ("attn_q.weight", types::Q8_0),
    ] {
        assert_eq!(m.tensor(&name(s)).unwrap().ty(), ty, "{s}");
    }
    assert_eq!(
        m.tensor("blk.0.ffn_gate_inp.weight").unwrap().ty(),
        types::F32,
        "a target weight was copied"
    );
}

/// One request at a time: `(reply, prompt tokens)`.
async fn run_one(engine: &LmEngine, prompt: &str, max_tokens: u32) -> (Reply, u32) {
    let mut s = engine
        .generate(GenerateRequest {
            model: "tiny".into(),
            prompt: Prompt::Text(prompt.into()),
            sampling: greedy(),
            max_tokens,
            stop: vec![],
            ignore_eos: false,
            min_tokens: 0,
            thinking: None,
            response_format: None,
            template_kwargs: Default::default(),
            cancel: CancellationToken::new(),
            scheduling: Default::default(),
        })
        .unwrap();
    let (mut text, mut prompt_tokens) = (String::new(), 0);
    loop {
        match s.next().await.expect("stream ended without Finished") {
            TokenEvent::Started { prompt_tokens: n } => prompt_tokens = n,
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { reason, usage, .. } => {
                let reply = Reply {
                    text,
                    reason,
                    tokens: usage.completion_tokens,
                    spec: usage.spec,
                };
                return (reply, prompt_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            _ => {}
        }
    }
}

/// #477: a prompt that is not a whole number of KV blocks stops a step at its last full block
/// (its state saved for the prefix cache) and reads the rest in a step of its own; the head's
/// take-in of the long chunk waits for that step, after the first token. Greedy output stays
/// bit-identical to plain decoding, and the drafts (so the spec counters) are those of an
/// engine that reads each prompt in one step (no saved states, a bigger batch) and so defers
/// nothing. A one-token request (whose take-in still runs) and the same prompt again from the
/// prefix cache keep the output too.
#[tokio::test(flavor = "multi_thread")]
async fn mtp_deferred_take_in_keeps_drafts_and_output() {
    let dir = tempfile::tempdir().unwrap();
    write_models(dir.path());
    let load = |spec: SpecConfig, prefix_cache_states: usize, max_batch_tokens: usize| {
        LmEngine::load(LmConfig {
            model_name: Some("tiny".into()),
            ctx_size: 512,
            threads: 4,
            max_seqs: 3,
            max_batch_tokens,
            spec,
            prefix_cache_states,
            use_ref: true,
            ..LmConfig::new(dir.path().join("tiny.gguf"), DeviceChoice::Cpu)
        })
        .unwrap()
    };
    // Fixed depth (no timing in it), and a confidence floor that the head's probabilities on
    // this model cross both ways: the counters follow the drafts' confidences.
    let fixed = SpecConfig {
        fixed: true,
        ..mtp(dir.path(), 3, 0.015)
    };
    let base = "The head takes the prompt in after its first token. ".repeat(3);
    // No two share a KV block, so every first run reads its whole prompt.
    let prompt = |lead: &str| format!("{lead} {base}");
    let engines = [
        load(SpecConfig::default(), 8, 48),
        load(fixed.clone(), 8, 48),
        load(fixed, 0, 256),
    ];
    let same = |a: &Reply, b: &Reply, what: &str| {
        assert_eq!(
            (&a.text, a.reason, a.tokens),
            (&b.text, b.reason, b.tokens),
            "{what} differs"
        );
    };
    let mut lengths = Vec::new();
    for lead in ["a", "bb", "ccc"] {
        let p = prompt(lead);
        let (plain, n) = run_one(&engines[0], &p, 40).await;
        let (deferred, _) = run_one(&engines[1], &p, 40).await;
        let (whole, _) = run_one(&engines[2], &p, 40).await;
        same(&plain, &deferred, lead);
        same(&plain, &whole, lead);
        assert_eq!(deferred.spec, whole.spec, "{lead}: other drafts");
        lengths.push(n);
    }
    assert!(
        lengths.iter().any(|n| *n > 64 && n % 64 != 0),
        "no prompt with a partial last block: {lengths:?}"
    );
    for lead in ["dd", "eee", "f"] {
        let p = prompt(lead);
        for max_tokens in [1, 40] {
            let (plain, _) = run_one(&engines[0], &p, max_tokens).await;
            let (deferred, _) = run_one(&engines[1], &p, max_tokens).await;
            same(&plain, &deferred, &format!("{lead}, {max_tokens} tokens"));
        }
    }
    let s = engines[1].stats().spec;
    println!("{s:?}");
    assert!(s.drafted > 0, "nothing drafted");
}
