use super::*;
use crate::engine::lm::LmEngine;
use crate::engine::lm::response_format::GrammarCache;
use crate::engine::{GenerateRequest, ModelInfo, Prompt, SamplingParams};
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;
use std::sync::{Arc, Mutex};

#[test]
fn blocks_cover_prompt_output_and_drafts() {
    // 100 + 28 tokens, no drafts: two blocks.
    assert_eq!(blocks(100, 28, 1 << 20, 0), 2);
    // A verify step's drafts past the last kept position take a third.
    assert_eq!(blocks(100, 28, 1 << 20, 3), 3);
    // `max_tokens` past the context counts up to the context only.
    assert_eq!(blocks(100, u32::MAX, 256, 0), 4);
    assert_eq!(blocks(100, u32::MAX, 256, 1), 5);
}

#[test]
fn the_report_case_is_refused() {
    // #125's numbers: a 262,144-token context and cache, MTP drafts on, and a request whose
    // prompt + max_tokens reach the context.
    let ctx = 262_144;
    let pool = ctx / KV_BLOCK;
    let e = check(228_413, 40_000, ctx, 3, pool).unwrap_err();
    assert!(
        matches!(
            e,
            EngineError::KvOverflow {
                need: 262_208,
                holds: 262_144
            }
        ),
        "{e:?}"
    );
    assert_eq!(
        e.to_string(),
        "the request needs 262208 KV cache tokens (prompt + max_tokens), the cache holds 262144"
    );
    // The same prompt with room for its output fits.
    check(228_413, 16_000, ctx, 3, pool).unwrap();
}

#[test]
fn the_automatic_pool_holds_a_full_context_with_drafts() {
    // The same request against the automatic pool, one sequence or a GPU sharing the host's
    // memory (whose pool is capped at a context): it counts the drafts' positions per context.
    let ctx = 262_144;
    assert_eq!(context_blocks(ctx, 0), ctx / KV_BLOCK);
    assert_eq!(context_blocks(ctx, 3), ctx / KV_BLOCK + 1);
    for (max_seqs, shared) in [(1, false), (1, true), (4, true)] {
        let pool = crate::engine::lm::auto_kv_want(max_seqs, context_blocks(ctx, 3), false, shared);
        check(228_413, 40_000, ctx, 3, pool).unwrap();
        check(ctx - 1, u32::MAX, ctx, 3, pool).unwrap();
        // Without drafts the pool is a context, as before.
        let pool = crate::engine::lm::auto_kv_want(max_seqs, context_blocks(ctx, 0), false, shared);
        let contexts = if shared { 1 } else { max_seqs };
        assert_eq!(pool, contexts * ctx / KV_BLOCK);
        check(ctx - 1, u32::MAX, ctx, 0, pool).unwrap();
    }
    // Any context size: the longest request [`blocks`] counts fits [`context_blocks`].
    for ctx in [1, 63, 64, 65, 4096, 100_000] {
        for spec_k in [0, 1, 3, 16, 64] {
            assert_eq!(
                blocks(0, u32::MAX, ctx, spec_k),
                context_blocks(ctx, spec_k)
            );
        }
    }
}

#[test]
fn fits_exactly() {
    check(64, 64, 1 << 20, 0, 2).unwrap();
    assert!(check(64, 65, 1 << 20, 0, 2).is_err());
}

fn engine(kv_blocks: usize, spec_k: usize) -> LmEngine {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ggml-vocab-qwen35.gguf");
    let tokenizer = Arc::new(Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap());
    LmEngine {
        vision: None,
        info: ModelInfo::default(),
        chat_format: super::super::chat_format::ChatFormat::detect(&tokenizer),
        undeclared_tool_calls: false,
        tokenizer,
        template: None,
        grammars: Mutex::new(GrammarCache::default()),
        ctx_size: 4096,
        kv_blocks,
        spec_k,
        max_waiting: 1,
        queued: Default::default(),
        live: Default::default(),
        warming: Default::default(),
        stats: Default::default(),
        submit: None,
        worker: None,
        load_report: Default::default(),
        split_probe: None,
        _tier_plan: None,
    }
}

fn request(max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        model: String::new(),
        prompt: Prompt::Text("hello world".into()),
        sampling: SamplingParams::default(),
        max_tokens,
        stop: vec![],
        ignore_eos: false,
        min_tokens: 0,
        thinking: None,
        response_format: None,
        template_kwargs: Default::default(),
        cancel: Default::default(),
        scheduling: Default::default(),
    }
}

/// `prepare` refuses a request the cache can never hold before it queues, as the client's
/// error (HTTP 400 `invalid_request_error`, not the 500 the worker's admission gave).
#[test]
fn prepare_refuses_before_queueing() {
    let e = engine(2, 0);
    // A few prompt tokens and 64 output tokens fit two blocks.
    assert!(e.prepare(request(64)).is_ok());
    let err = e.prepare(request(128)).err().unwrap();
    assert!(
        matches!(err, EngineError::KvOverflow { holds: 128, .. }),
        "{err:?}"
    );
    assert_eq!(
        crate::server::error::ApiError::from(err).status(),
        axum::http::StatusCode::BAD_REQUEST
    );
    // Drafts count: the same 64 output tokens with 64 drafts need a third block.
    let e = engine(2, 64);
    assert!(matches!(
        e.prepare(request(64)).err().unwrap(),
        EngineError::KvOverflow { .. }
    ));
}

#[test]
fn huge_sizes_saturate_instead_of_wrapping() {
    assert_eq!(context_blocks(usize::MAX, 3), usize::MAX.div_ceil(KV_BLOCK));
    assert_eq!(
        blocks(usize::MAX, u32::MAX, usize::MAX, 3),
        usize::MAX.div_ceil(KV_BLOCK)
    );
    assert!(context_blocks(usize::MAX - 1, 8) >= context_blocks(1 << 20, 8));
}
