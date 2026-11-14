use super::*;
use crate::chat_template::ChatTemplate;
use crate::engine::lm::LmEngine;
use crate::engine::{GenerateRequest, ModelInfo, Prompt, SamplingParams};
use crate::gguf::GgufFile;
use std::sync::{Arc, Mutex};

#[test]
fn json_overrides_thinking_and_template_kwargs() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ggml-vocab-qwen35.gguf");
    let tokenizer = Arc::new(Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap());
    let engine = LmEngine {
        vision: None,
        info: ModelInfo::default(),
        chat_format: super::super::chat_format::ChatFormat::detect(&tokenizer),
        tokenizer,
        template: Some(
            ChatTemplate::new(
                "{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}",
                "",
                "",
            )
            .unwrap(),
        ),
        grammars: Mutex::new(GrammarCache::default()),
        ctx_size: 128,
        max_waiting: 1,
        queued: Default::default(),
        warming: Default::default(),
        stats: Default::default(),
        submit: None,
        worker: None,
        load_report: Default::default(),
        split_probe: None,
    };
    let mut req = GenerateRequest {
        model: String::new(),
        prompt: Prompt::Chat {
            system: None,
            messages: vec![],
            tools: vec![],
        },
        sampling: SamplingParams::default(),
        max_tokens: 32,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: Some(true),
        response_format: Some(ResponseFormat::JsonObject),
        template_kwargs: serde_json::from_value(serde_json::json!({"enable_thinking":true}))
            .unwrap(),
        cancel: Default::default(),
        scheduling: Default::default(),
    };
    assert_eq!(
        engine.render(&req).unwrap(),
        ("<think></think>".into(), None)
    );
    req.response_format = None;
    assert_eq!(engine.render(&req).unwrap().0, "<think>");
}

fn vocab() -> &'static Tokenizer {
    static T: std::sync::OnceLock<Tokenizer> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/ggml-vocab-qwen35.gguf");
        Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
    })
}

fn compiler() -> &'static crate::grammar::GrammarCompiler {
    static C: std::sync::OnceLock<crate::grammar::GrammarCompiler> = std::sync::OnceLock::new();
    C.get_or_init(|| crate::grammar::GrammarCompiler::new(vocab()))
}

/// The ids `mask` leaves finite in a row of `n` zero logits.
fn kept(c: &mut Constraint, n: usize) -> Vec<u32> {
    let mut logits = vec![0.0f32; n];
    c.mask(&mut logits);
    (0..n as u32)
        .filter(|&i| logits[i as usize].is_finite())
        .collect()
}

fn allowed(m: &crate::grammar::Matcher) -> Vec<u32> {
    let mut words = vec![0u32; m.mask_words()];
    m.fill_mask(&mut words);
    (0..(words.len() * 32) as u32)
        .filter(|&i| words[i as usize / 32] >> (i % 32) & 1 == 1)
        .collect()
}

#[test]
fn apply_mask_keeps_only_allowed_ids() {
    let mask = [0b101u32, 0, 1 << 31];
    let mut logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
    assert!(apply_mask(&mask, &mut logits));
    for (i, l) in logits.iter().enumerate() {
        // Ids past the mask (96..100) are not allowed either.
        let keep = matches!(i, 0 | 2 | 95);
        assert_eq!(l.is_finite(), keep, "id {i}");
        if keep {
            assert_eq!(*l, i as f32);
        }
    }
    // Allowed ids already at minus infinity (min_tokens masking): nothing left.
    let mut logits = vec![0.0f32; 64];
    logits[0] = f32::NEG_INFINITY;
    logits[2] = f32::NEG_INFINITY;
    assert!(!apply_mask(&[0b101], &mut logits));
    assert!(logits.iter().all(|l| *l == f32::NEG_INFINITY));
    assert!(!apply_mask(&[], &mut [1.0, 2.0]));
}

/// The mask threads give the masks the engine thread would find itself, token after token.
#[test]
fn mask_threads_match_inline_masks() {
    let g = compiler()
        .compile_json_schema(&serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"enum": ["x", "y"]}},
            "required": ["a", "b"]
        }))
        .unwrap();
    let n = vocab().vocab_size() + 7;
    let ids = vocab().encode(r#"{"a": 12, "b": "y"}"#, false);
    let inline = MaskPool::new(0);
    for threads in [1, 2, 4] {
        let pool = MaskPool::new(threads);
        let (mut a, mut b) = (
            Constraint::format(&g, &pool),
            Constraint::format(&g, &inline),
        );
        let mut m = g.matcher();
        for &id in &ids {
            assert!(a.active() && a.trigger().is_none());
            let want = allowed(&m);
            assert_eq!(kept(&mut a, n), want);
            assert_eq!(kept(&mut b, n), want);
            m.accept_token(id).unwrap();
            assert_eq!(a.accept(id, false), Ok(false));
            assert_eq!(b.accept(id, false), Ok(false));
        }
        let eos = vocab().eos_id().unwrap();
        assert!(kept(&mut a, n).contains(&eos));
        assert_eq!(a.accept(eos, false), Ok(true));
        // A token off the grammar is an error.
        let mut c = Constraint::format(&g, &pool);
        assert!(c.accept(vocab().encode("]", false)[0], false).is_err());
    }
}

/// Many sequences' masks at once over a few threads.
#[test]
fn mask_threads_serve_many_sequences() {
    let g = compiler()
        .compile_json_schema(&serde_json::json!({"type": "array", "items": {"type": "boolean"}}))
        .unwrap();
    let pool = MaskPool::new(2);
    let n = vocab().vocab_size();
    let ids = vocab().encode("[true, false, true]", false);
    let mut seqs: Vec<Constraint> = (0..32).map(|_| Constraint::format(&g, &pool)).collect();
    let mut m = g.matcher();
    for &id in &ids {
        let want = allowed(&m);
        for c in &mut seqs {
            assert_eq!(kept(c, n), want);
            c.accept(id, false).unwrap();
        }
        m.accept_token(id).unwrap();
    }
}

#[test]
fn strict_tools_constrain_only_inside_a_call() {
    let open = vocab().token_id("<|endoftext|>").unwrap();
    let end = vocab().token_id("<|im_start|>").unwrap();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"unit": {"enum": ["c", "f"]}},
        "required": ["unit"]
    });
    let g = compiler()
        .compile_tool_calls(
            &[crate::grammar::ToolSpec {
                name: "w",
                strict: Some(&schema),
            }],
            end,
        )
        .unwrap();
    let n = vocab().vocab_size();
    for threads in [0, 2] {
        let pool = MaskPool::new(threads);
        let mut c = Constraint::tools(&g, open, &pool);
        let free = vocab().encode("Let me check.", false);
        for _ in 0..2 {
            // Free text, and the open token inside reasoning, leave it unconstrained.
            for &id in free.iter().chain([&open]) {
                assert!(!c.active());
                assert_eq!(c.trigger(), Some(open));
                assert_eq!(c.accept(id, true), Ok(false));
            }
            for &id in &free {
                assert_eq!(c.accept(id, false), Ok(false));
            }
            assert_eq!(c.accept(open, false), Ok(false));
            assert!(c.active());
            assert_eq!(c.trigger(), None);
            let body = vocab().encode(
                "\n<function=w>\n<parameter=unit>\nf\n</parameter>\n</function>\n",
                false,
            );
            let mut m = g.matcher();
            for &id in &body {
                assert_eq!(kept(&mut c, n), allowed(&m));
                assert_eq!(c.accept(id, false), Ok(false));
                m.accept_token(id).unwrap();
            }
            assert_eq!(kept(&mut c, n), vec![end]);
            // The call's end: free text again, for the next call.
            assert_eq!(c.accept(end, false), Ok(false));
            assert!(!c.active());
        }
        // Inside a call, a token off the grammar is an error.
        c.accept(open, false).unwrap();
        assert!(c.accept(vocab().encode("x", false)[0], false).is_err());
    }
}
