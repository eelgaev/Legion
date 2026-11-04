mod common;

use std::path::Path;
use std::time::Instant;

use legion::gguf::{GgufFile, GgufWriter, Value};
use legion::tokenizer::Tokenizer;

fn load() -> Tokenizer {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/ggml-vocab-gemma-4.gguf");
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

fn synthetic(pre: Option<&str>, add_bos: Option<bool>, special_ids: bool) -> Tokenizer {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let mut writer = GgufWriter::new();
    writer
        .set("tokenizer.ggml.model", &Value::String("gemma4".into()))
        .unwrap();
    let tokens = [
        "<pad>", "<eos>", "<bos>", "<unk>", "a", "▁", "b", "a▁b", "a▁", "\n", "\n\n", "<0xF0>",
        "<0x9F>", "<0xA6>", "<0x99>", "é",
    ];
    writer
        .set(
            "tokenizer.ggml.tokens",
            &Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
        )
        .unwrap();
    writer
        .set(
            "tokenizer.ggml.token_type",
            &Value::Array(
                (0..tokens.len())
                    .map(|i| {
                        Value::I32(match i {
                            0..=2 => 3,
                            3 => 2,
                            11..=14 => 6,
                            _ => 1,
                        })
                    })
                    .collect(),
            ),
        )
        .unwrap();
    writer
        .set(
            "tokenizer.ggml.merges",
            &Value::Array(vec![
                Value::String("a ▁".into()),
                Value::String("a▁ b".into()),
            ]),
        )
        .unwrap();
    if let Some(pre) = pre {
        writer
            .set("tokenizer.ggml.pre", &Value::String(pre.into()))
            .unwrap();
    }
    if let Some(add_bos) = add_bos {
        writer
            .set("tokenizer.ggml.add_bos_token", &Value::Bool(add_bos))
            .unwrap();
    }
    if special_ids {
        writer
            .set("tokenizer.ggml.bos_token_id", &Value::U32(2))
            .unwrap();
        writer
            .set("tokenizer.ggml.eos_token_id", &Value::U32(1))
            .unwrap();
    }
    writer.write(&path).unwrap();
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

#[test]
fn ignores_pre_and_forces_bos_like_llama_cpp() {
    for pre in [None, Some("gemma4"), Some("qwen2"), Some("unrecognized")] {
        for add_bos in [None, Some(false), Some(true)] {
            let tok = synthetic(pre, add_bos, true);
            assert!(tok.add_bos());
            assert_eq!(tok.encode("a b", true), [2, 7]);
            assert_eq!(tok.encode_with("a b", false, true), [7]);
            assert_eq!(tok.encode_with("a b", false, false), [7]);
            assert_eq!(tok.encode("", true), [2]);
            assert!(tok.encode("", false).is_empty());
        }
    }
    let tok = synthetic(None, None, false);
    assert_eq!(tok.bos_id(), None);
    assert_eq!(tok.eos_id(), None);
}

#[test]
fn raw_unicode_merges_newline_shortcut_and_byte_fallback() {
    let tok = synthetic(None, None, true);
    for (text, expected) in [
        ("a b", vec![7]),
        ("é", vec![15]),
        ("🦙", vec![11, 12, 13, 14]),
        ("\n\n", vec![10]),
        // No whole-word token and no newline merges: emit individual newlines.
        ("\n\n\n", vec![9, 9, 9]),
        ("a\nb", vec![4, 9, 6]),
        ("  a b  ", vec![5, 5, 7, 5, 5]),
    ] {
        let ids = tok.encode(text, false);
        assert_eq!(ids, expected, "{text:?}");
        assert_eq!(tok.decode(&ids), text);
    }
    let mut stream = tok.decode_stream();
    assert_eq!(stream.push(11), "");
    assert_eq!(stream.push(12), "");
    assert_eq!(stream.push(13), "");
    assert_eq!(stream.push(14), "🦙");
    assert_eq!(stream.finish(), "");
}

#[test]
fn fixture_metadata_bos_and_special_tokens() {
    let tok = load();
    assert_eq!(tok.vocab_size(), 262_144);
    assert_eq!(tok.bos_id(), Some(2));
    assert_eq!(tok.eos_id(), Some(1));
    assert_eq!(tok.pad_id(), Some(0));
    assert_eq!(tok.unk_id(), Some(3));
    assert!(tok.add_bos());
    assert!(!tok.add_eos());
    assert_eq!(tok.token_text(2), Some("<bos>"));
    assert_eq!(tok.token_text(1), Some("<eos>"));

    let text = "Hello world\n\n日本語 🦙";
    let plain = tok.encode_with(text, false, false);
    let mut expected = vec![2];
    expected.extend(&plain);
    assert_eq!(tok.encode(text, true), expected);
    assert_eq!(tok.encode_with(text, false, true), plain);
    assert_eq!(tok.decode(&expected), text);
    assert_eq!(tok.encode("", true), [2]);
    assert!(tok.encode("", false).is_empty());

    let text = "a<bos>b<eos>";
    let parsed = tok.encode_with(text, false, true);
    assert!(parsed.contains(&2) && parsed.contains(&1));
    assert_eq!(tok.decode(&parsed), "ab");
    assert_eq!(tok.decode_with(&parsed, true), text);
    let literal = tok.encode_with(text, false, false);
    assert!(!literal.contains(&2) && !literal.contains(&1));
    assert_eq!(tok.decode(&literal), text);
}

/// `cargo test --release --test tokenizer_gemma4 encode_speed -- --ignored --nocapture`
#[test]
#[ignore]
fn encode_speed() {
    let tok = load();
    let unit = "The quick brown fox jumps over the lazy dog. fn main() { println!(\"hello, world {}\", 42); }\n\
                Ünïcödé text and 日本語のテキスト, numbers 3.14159 and 2026-10-16.\n";
    let mut text = unit.repeat(100_000 / unit.len() + 1);
    text.truncate(text.floor_char_boundary(100_000));
    while text.len() < 100_000 {
        text.push(' ');
    }
    let start = Instant::now();
    let ids = tok.encode(&text, false);
    let elapsed = start.elapsed();
    println!(
        "Gemma 4 encoded {} bytes -> {} tokens in {elapsed:?}",
        text.len(),
        ids.len()
    );
    assert_eq!(tok.decode(&ids), text);
}
