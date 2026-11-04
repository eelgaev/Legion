use super::*;

#[test]
fn prefill_prompt_item_count_is_n_over_3() {
    for n in [0, 1, 2, 3, 9, 512] {
        let text = prefill_prompt(n);
        let body = text.strip_suffix("\nSummarize in one line.").unwrap();
        if n / 3 == 0 {
            assert!(body.is_empty(), "n={n}: {body:?}");
        } else {
            assert_eq!(body.split(' ').count(), n / 3, "n={n}");
        }
    }
}

#[test]
fn prefill_prompt_is_stable_for_fixed_n() {
    let a = prefill_prompt(512);
    let b = prefill_prompt(512);
    assert_eq!(a, b);
    assert_ne!(prefill_prompt(256), a);
}

#[test]
fn prefill_prompt_golden_first_items_for_512() {
    // SplitMix64 seeded with 512: first three word+number items (documented golden).
    let text = prefill_prompt(512);
    let body = text.strip_suffix("\nSummarize in one line.").unwrap();
    let items: Vec<&str> = body.split(' ').collect();
    assert_eq!(items.len(), 512 / 3);
    assert_eq!(&items[..3], &["from438", "from988", "an827"]);
}

#[test]
fn decode_tok_s_edge_cases() {
    assert_eq!(decode_tok_s(0, Some(0.1), Some(0.2)), None);
    assert_eq!(decode_tok_s(1, Some(0.1), Some(0.2)), None);
    assert_eq!(decode_tok_s(2, Some(0.1), Some(0.1)), None);
    assert_eq!(decode_tok_s(2, None, Some(0.2)), None);
    assert_eq!(decode_tok_s(2, Some(0.1), None), None);
    let r = decode_tok_s(101, Some(0.1), Some(1.1)).unwrap();
    assert!((r - 100.0).abs() < 1e-9);
}

#[test]
fn prefill_tok_s_edge_cases() {
    assert_eq!(prefill_tok_s(None, Some(0.5)), None);
    assert_eq!(prefill_tok_s(Some(100), None), None);
    assert_eq!(prefill_tok_s(Some(0), Some(0.5)), None);
    assert_eq!(prefill_tok_s(Some(100), Some(0.0)), None);
    let r = prefill_tok_s(Some(512), Some(0.5)).unwrap();
    assert!((r - 1024.0).abs() < 1e-9);
}

/// An OpenAI stream whose only token was swallowed (empty content, e.g. a `<think>` tag or EOS
/// with `max_tokens` 1): usage arrives, no visible delta does.
const SWALLOWED: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

const ONE_TOKEN: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
);

/// Feeds `body` to `read_stream` in two pieces, split mid-event.
fn read(body: &str) -> anyhow::Result<StreamResult> {
    let (a, b) = body.as_bytes().split_at(body.len() / 2);
    let chunks: Vec<Result<Vec<u8>, std::io::Error>> = vec![Ok(a.to_vec()), Ok(b.to_vec())];
    futures::executor::block_on(read_stream(
        Api::OpenAi,
        Instant::now(),
        futures::stream::iter(chunks),
    ))
}

#[test]
fn prefill_point_without_a_visible_token_records_nulls() {
    let s = read(SWALLOWED).unwrap();
    assert_eq!(s.chunks, 0);
    let rec = into_prefill(512, s);
    assert_eq!(rec.n_arg, 512);
    assert_eq!(rec.prompt_tokens, Some(42));
    assert_eq!(rec.ttft_s, None);
    assert_eq!(rec.prefill_tok_s, None);
    assert_eq!(rec.completion_tokens, 1);
    let v = serde_json::to_value(&rec).unwrap();
    assert!(v["ttft_s"].is_null() && v["prefill_tok_s"].is_null(), "{v}");
}

#[test]
fn chat_without_a_visible_token_is_an_error() {
    let err = into_chat("p", read(SWALLOWED).unwrap()).unwrap_err();
    assert!(err.to_string().contains("no tokens received"), "{err}");
}

#[test]
fn a_visible_token_sets_ttft() {
    let s = read(ONE_TOKEN).unwrap();
    assert_eq!(s.chunks, 1);
    assert_eq!(s.content, "Hi");
    let rec = into_prefill(512, s);
    assert!(rec.ttft_s.is_some());
    assert!(rec.prefill_tok_s.unwrap() > 0.0);
    into_chat("p", read(ONE_TOKEN).unwrap()).unwrap();
}

#[test]
fn a_stream_cut_before_done_or_usage_is_still_an_error() {
    let cut = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"}}]}\n\n";
    let err = read(cut).unwrap_err();
    assert!(err.to_string().contains("stream ended early"), "{err}");
}

#[test]
fn any_saved_run_blocks_save_unless_forced() {
    for name in ["meta.json", "bench.json", "chats.json", "prefill.json"] {
        let dir = tempfile::tempdir().unwrap();
        save::check_free(dir.path(), false).unwrap();
        std::fs::write(dir.path().join(name), "{}").unwrap();
        let err = save::check_free(dir.path(), false).unwrap_err();
        assert!(err.to_string().contains(name), "{name}: {err}");
        save::check_free(dir.path(), true).unwrap();
    }
    // README.md alone is not a saved run.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "notes").unwrap();
    save::check_free(dir.path(), false).unwrap();
}
