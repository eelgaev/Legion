//! The Qwen3.5 engine end to end: chat template, prefill + decode, sampling, reasoning split,
//! stop strings, usage and cancellation, on the CPU.
//!
//! Ignored: needs the real model. `LEGION_TEST_QWEN35=<model.gguf> cargo test --release --test
//! qwen35_engine -- --ignored --test-threads 1`. With `LEGION_TEST_QWEN35_CUDA=<model.gguf>` and
//! `--features cuda`, `cuda_engine_generates` runs the same on the GPU.

use futures::StreamExt;
use legion::engine::lm::{DeviceChoice, LmConfig, LmEngine};
use legion::engine::{
    Engine, FinishReason, GenerateRequest, Message, Prompt, Role, SamplingParams, TokenEvent,
};
use legion::ggml::Backend;
use legion::gguf::GgufFile;
use legion::model::core::Model;
use legion::tokenizer::Tokenizer;
use tokio_util::sync::CancellationToken;

fn engine(path: &str, device: DeviceChoice) -> LmEngine {
    LmEngine::load(LmConfig {
        model_name: Some("qwen".into()),
        ctx_size: 2048,
        ..LmConfig::new(path, device)
    })
    .unwrap()
}

fn chat(text: &str, thinking: Option<bool>, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model: "qwen".into(),
        prompt: Prompt::Chat {
            system: None,
            messages: vec![Message {
                role: Some(Role::User),
                content: text.into(),
                ..Default::default()
            }],
            tools: vec![],
        },
        sampling: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        max_tokens,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: CancellationToken::new(),
        scheduling: Default::default(),
    }
}

/// (reasoning, text, finish reason, completion tokens)
async fn run(engine: &LmEngine, req: GenerateRequest) -> (String, String, FinishReason, u32) {
    let mut stream = engine.generate(req).unwrap();
    let (mut reasoning, mut text) = (String::new(), String::new());
    let mut started = false;
    while let Some(ev) = stream.next().await {
        match ev {
            TokenEvent::Started { prompt_tokens } => {
                assert!(prompt_tokens > 0);
                started = true;
            }
            TokenEvent::Reasoning(r) => reasoning += &r,
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Finished { reason, usage, .. } => {
                assert!(started);
                // Prefill always takes some time; decode is measured from the first token.
                let prefill = usage.prefill_ms.expect("prefill_ms measured");
                let decode = usage.decode_ms.expect("decode_ms measured");
                assert!(prefill > 0.0 && decode >= 0.0, "{usage:?}");
                if usage.completion_tokens > 1 {
                    assert!(decode > 0.0, "{usage:?}");
                }
                return (reasoning, text, reason, usage.completion_tokens);
            }
            TokenEvent::Error(e) => panic!("engine error: {e}"),
            other => panic!("unexpected {other:?}"),
        }
    }
    panic!("stream ended without Finished");
}

fn model_path() -> String {
    std::env::var("LEGION_TEST_QWEN35").expect("LEGION_TEST_QWEN35 is not set")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the real model"]
async fn engine_generates() {
    let path = model_path();
    let engine = engine(&path, DeviceChoice::Cpu);

    // Thinking off: the template closes <think> in the prompt, the answer is plain text.
    let (reasoning, text, reason, n) = run(
        &engine,
        chat("What is the capital of France? One word.", Some(false), 32),
    )
    .await;
    println!("no thinking: {text:?} ({reason:?}, {n} tokens)");
    assert!(reasoning.is_empty(), "{reasoning:?}");
    assert!(text.contains("Paris"), "{text:?}");
    assert_eq!(reason, FinishReason::Stop);

    // Greedy output is the argmax chain of the model itself: same tokens as a plain loop over
    // `Model::forward` on the rendered prompt.
    let gguf = GgufFile::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(8);
    let model = Model::load(&gguf, |_| backend.buffer_type()).unwrap();
    let prompt = tok.encode_with(
        "<|im_start|>user\nWhat is the capital of France? One word.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n",
        true,
        true,
    );
    let mut state = model.new_state(backend.buffer_type(), 512).unwrap();
    let mut alloc = None;
    let mut logits = model
        .forward(&mut backend, &mut alloc, &mut state, &prompt, false)
        .unwrap();
    let mut ids = Vec::new();
    for _ in 0..n {
        let next = legion::engine::sampling::argmax(&logits) as u32;
        ids.push(next);
        logits = model
            .forward(&mut backend, &mut alloc, &mut state, &[next], false)
            .unwrap();
    }
    assert_eq!(tok.decode(&ids), text);

    // Thinking on (the default differs: on in the 3.6 templates, off in the 3.5 4B's): the
    // template opens <think>, reasoning comes first, then the answer.
    let (reasoning, text, reason, n) = run(
        &engine,
        chat("What is 2 + 3? Answer briefly.", Some(true), 400),
    )
    .await;
    println!("thinking: {reasoning:?} / {text:?} ({reason:?}, {n} tokens)");
    assert!(!reasoning.is_empty());
    assert!(!reasoning.contains("<think>") && !reasoning.contains("</think>"));

    // max_tokens and stop strings.
    let (_, _, reason, n) = run(&engine, chat("Count from 1 to 50.", Some(false), 5)).await;
    assert_eq!((reason, n), (FinishReason::Length, 5));
    let mut req = chat("Count from 1 to 50, comma separated.", Some(false), 200);
    req.stop = vec!["5".into()];
    let (_, text, reason, _) = run(&engine, req).await;
    assert_eq!(reason, FinishReason::StopSequence);
    assert!(!text.contains('5'), "{text:?}");

    // Seeded sampling is reproducible.
    let sampled = |seed| {
        let mut req = chat("Write a short poem about the sea.", Some(false), 24);
        req.sampling = SamplingParams {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            seed: Some(seed),
            ..Default::default()
        };
        req
    };
    let a = run(&engine, sampled(11)).await.1;
    let b = run(&engine, sampled(11)).await.1;
    assert_eq!(a, b);

    // Cancellation: dropping the stream stops generation and the next request still runs.
    let req = chat("Count from 1 to 1000.", Some(false), 1000);
    let cancel = req.cancel.clone();
    let mut stream = engine.generate(req).unwrap();
    let mut seen = 0;
    while let Some(ev) = stream.next().await {
        if matches!(ev, TokenEvent::Text(_)) {
            seen += 1;
            if seen == 3 {
                cancel.cancel();
            }
        }
        if let TokenEvent::Finished { reason, .. } = ev {
            assert_eq!(reason, FinishReason::Cancelled);
            break;
        }
    }
    let (_, text, _, _) = run(
        &engine,
        chat("What is the capital of France? One word.", Some(false), 32),
    )
    .await;
    assert!(text.contains("Paris"));
}

/// Several requests at once: all are admitted together and batched (prompts prefilled next to
/// each other's decode steps), and each answer is still right. Then a prompt sharing a long
/// system prompt with an earlier one resumes from the prefix cache (KV blocks plus a saved
/// recurrent state) and reports the reused tokens as `cached_tokens`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the real model"]
async fn engine_batches_and_reuses_prefixes() {
    let path = model_path();
    let engine = LmEngine::load(LmConfig {
        model_name: Some("qwen".into()),
        ctx_size: 2048,
        max_seqs: 4,
        max_batch_tokens: 256,
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    })
    .unwrap();
    let questions = [
        ("What is the capital of France? One word.", "Paris"),
        ("What is the capital of Japan? One word.", "Tokyo"),
        ("What is the capital of Italy? One word.", "Rome"),
        ("What is the capital of Egypt? One word.", "Cairo"),
        ("What is the capital of Peru? One word.", "Lima"),
    ];
    let answers = futures::future::join_all(
        questions
            .iter()
            .map(|(q, _)| run(&engine, chat(q, Some(false), 16))),
    )
    .await;
    for ((q, want), (_, text, reason, _)) in questions.iter().zip(&answers) {
        println!("{q} -> {text:?}");
        assert!(text.contains(want), "{q}: {text:?}");
        assert_eq!(*reason, FinishReason::Stop);
    }

    let system = "You are a terse assistant for a geography quiz. ".repeat(20);
    let ask = |q: &str| {
        let mut req = chat(q, Some(false), 16);
        if let Prompt::Chat { system: s, .. } = &mut req.prompt {
            *s = Some(system.clone());
        }
        req
    };
    let cached = |req| async {
        let mut stream = engine.generate(req).unwrap();
        let mut text = String::new();
        while let Some(ev) = stream.next().await {
            match ev {
                TokenEvent::Text(t) => text += &t,
                TokenEvent::Finished { usage, .. } => return (text, usage.cached_tokens),
                TokenEvent::Error(e) => panic!("{e}"),
                _ => {}
            }
        }
        panic!("no Finished");
    };
    // The first prompt saves its state at its last full KV block, inside the system prompt
    // here, so the next ones resume there.
    let (_, first) = cached(ask("Capital of France?")).await;
    let (_, second) = cached(ask("Capital of Spain?")).await;
    let (text, third) = cached(ask("Capital of Kenya?")).await;
    println!("cached tokens: {first}, {second}, {third}; {text:?}");
    assert_eq!(first, 0);
    assert!(second >= 128 && third >= second, "{second}, {third}");
    assert!(text.contains("Nairobi"), "{text:?}");
    let stats = engine.stats();
    assert_eq!(stats.prefix_cache_hit_tokens, (second + third) as u64);
    assert_eq!(stats.running, 0);
}

/// `count_tokens` is exactly what `generate` reports as `prompt_tokens`; a system turn in the
/// middle of a chat (Claude Code sends them; Qwen templates reject them) is accepted.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the real model"]
async fn engine_counts_tokens_and_accepts_late_system_turns() {
    let engine = engine(&model_path(), DeviceChoice::Cpu);
    let mut req = chat("What is the capital of France? One word.", Some(false), 8);
    if let Prompt::Chat { messages, .. } = &mut req.prompt {
        messages.push(Message {
            role: Some(Role::System),
            content: "Answer in English.".into(),
            ..Default::default()
        });
    }
    let counted = engine.count_tokens(&req).unwrap();
    let mut stream = engine.generate(req).unwrap();
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev {
            TokenEvent::Started { prompt_tokens } => assert_eq!(prompt_tokens, counted),
            TokenEvent::Text(t) => text += &t,
            TokenEvent::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    assert!(text.contains("Paris"), "{text:?}");
    let mut other = chat("hi", None, 8);
    other.model = "nope".into();
    assert!(engine.count_tokens(&other).is_err());
}

/// The engine on CUDA0. Opt-in like the other GPU test.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a GPU and the real model"]
async fn cuda_engine_generates() {
    let path = std::env::var("LEGION_TEST_QWEN35_CUDA").expect("LEGION_TEST_QWEN35_CUDA");
    let engine = engine(&path, DeviceChoice::Cuda);
    let (_, text, reason, n) = run(
        &engine,
        chat("What is the capital of France? One word.", Some(false), 32),
    )
    .await;
    println!("cuda: {text:?} ({reason:?}, {n} tokens)");
    assert!(text.contains("Paris"), "{text:?}");
    assert_eq!(reason, FinishReason::Stop);
}
