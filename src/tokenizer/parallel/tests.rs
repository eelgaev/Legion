use super::*;
use crate::gguf::{GgufFile, GgufWriter, Value};
use crate::tokenizer::byte_to_char_table;

/// Every byte as a token, plus a few merges so BPE has work to do.
fn tokenizer(pre: &str) -> Tokenizer {
    let mut tokens: Vec<String> = byte_to_char_table().iter().map(char::to_string).collect();
    let merges = [
        "Ġ t", "h e", "Ġt he", "Ġ a", "n d", "Ġa nd", "1 2", "Ġ w", "o r", "Ġw or", "l d",
    ];
    for m in merges {
        tokens.push(m.replace(' ', ""));
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let mut writer = GgufWriter::new();
    for (key, value) in [
        ("tokenizer.ggml.model", Value::String("gpt2".into())),
        ("tokenizer.ggml.pre", Value::String(pre.into())),
        (
            "tokenizer.ggml.tokens",
            Value::Array(tokens.into_iter().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.merges",
            Value::Array(merges.iter().map(|m| Value::String((*m).into())).collect()),
        ),
    ] {
        writer.set(key, &value).unwrap();
    }
    writer.write(&path).unwrap();
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

/// Mixed text: words with numbers, punctuation, contractions, runs of spaces and newlines, tabs
/// and non-ASCII letters, from a fixed LCG.
fn mixed_text(bytes: usize) -> String {
    let pieces = "the and world 12 345 don't it's . , ( café 日本語 x\u{300} !? (a) or";
    let pieces: Vec<&str> = pieces
        .split(' ')
        .chain(["  ", "\n", "\n\n", "\t", "   \n"])
        .collect();
    let mut state = 12345u64;
    let mut text = String::new();
    while text.len() < bytes {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let piece = pieces[(state >> 33) as usize % pieces.len()];
        text.push_str(piece);
        if !(state >> 20).is_multiple_of(3) {
            text.push(' ');
        }
    }
    text
}

#[test]
fn parallel_ids_match_one_thread() {
    for pre in ["qwen2", "qwen35", "llama3"] {
        let tok = tokenizer(pre);
        for (bytes, threads, chunk) in [(4000, 7, 64), (20_000, 16, 256), (999, 3, 1)] {
            let text = mixed_text(bytes);
            assert!(
                text.len() < MIN_BYTES,
                "the reference must stay on one thread"
            );
            let serial = tok.encode_with(&text, false, false);
            assert!(
                cut_points(&text, threads, chunk).len() > 3,
                "{pre}: too few cuts"
            );
            let parallel = tok.encode_text_parallel(&text, threads, chunk);
            assert_eq!(
                parallel.as_ref(),
                Some(&serial),
                "{pre} {bytes} {threads} {chunk}"
            );
        }
    }
}

#[test]
fn bench_style_prompt_cuts_and_matches() {
    // The owner's prefill prompt shape: "word+number" items, single spaces.
    let words = ["the", "of", "and", "to", "in", "is", "was", "for"];
    let text: String = (0..3000)
        .map(|i| format!("{}{} ", words[i * 7 % words.len()], i * 37 % 1000))
        .collect();
    let tok = tokenizer("qwen35");
    let cuts = cut_points(&text, 8, 512);
    assert_eq!(cuts.len(), 9);
    for &c in &cuts[1..cuts.len() - 1] {
        assert_eq!(&text[c..c + 1], " ");
    }
    assert_eq!(
        tok.encode_text_parallel(&text, 8, 512),
        Some(tok.encode_with(&text, false, false))
    );
}

#[test]
fn a_cut_inside_a_pre_token_is_refused() {
    let tok = tokenizer("qwen2");
    let text = "hello world and more";
    assert_eq!(tok.encode_span(text, 0, 3), None);
    assert!(tok.encode_span(text, 0, 5).is_some());
    assert!(tok.encode_span(text, 5, text.len()).is_some());
}

#[test]
fn text_without_cut_points_stays_on_one_thread() {
    let tok = tokenizer("qwen2");
    let text = "x".repeat(10_000);
    assert_eq!(cut_points(&text, 8, 16), vec![0, text.len()]);
    assert_eq!(tok.encode_text_parallel(&text, 8, 16), None);
    // Too short for two chunks.
    assert_eq!(cut_points("a b c d", 8, 16), vec![0, 7]);
}

#[test]
fn cuts_keep_the_chunk_size() {
    let text = mixed_text(10_000);
    let cuts = cut_points(&text, 16, 1000);
    assert!(cuts.len() <= 11);
    for w in cuts.windows(2) {
        assert!(w[1] - w[0] >= 1000, "{cuts:?}");
    }
}
