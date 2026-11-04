//! Tokenizer tests against llama.cpp's own vocab test vectors (`tests/data/ggml-vocab-*.gguf` with
//! `.inp`/`.out`, MIT, copied from `models/` of llama.cpp).

use std::path::PathBuf;
use std::time::Instant;

use legion::gguf::GgufFile;
use legion::tokenizer::Tokenizer;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(vocab: &str) -> Tokenizer {
    let gguf = GgufFile::open(data(&format!("ggml-vocab-{vocab}.gguf"))).unwrap();
    Tokenizer::from_gguf(&gguf).unwrap()
}

/// Same parsing as llama.cpp's `tests/test-tokenizer-0.cpp::read_tests`.
fn vectors(vocab: &str) -> Vec<(String, Vec<u32>)> {
    let inp = std::fs::read_to_string(data(&format!("ggml-vocab-{vocab}.gguf.inp"))).unwrap();
    let out = std::fs::read_to_string(data(&format!("ggml-vocab-{vocab}.gguf.out"))).unwrap();
    let sep = "\n__ggml_vocab_test__\n";
    let mut inputs = Vec::new();
    let mut pos = 0;
    while pos < inp.len() {
        match inp[pos..].find(sep) {
            Some(i) => {
                inputs.push(inp[pos..pos + i].to_string());
                pos += i + sep.len();
            }
            None => {
                inputs.push(inp[pos..].to_string());
                break;
            }
        }
    }
    let outputs: Vec<Vec<u32>> = out
        .lines()
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect();
    assert_eq!(inputs.len(), outputs.len(), "inp/out case counts differ");
    inputs.into_iter().zip(outputs).collect()
}

/// GPT-2 byte-to-unicode mapping, written out independently of the tokenizer.
fn gpt2_char(byte: u8) -> char {
    let printable = |b: u32| matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
    let b = byte as u32;
    if printable(b) {
        return char::from_u32(b).unwrap();
    }
    let index = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + index).unwrap()
}

fn check_vectors(vocab: &str) {
    let tok = load(vocab);
    let cases = vectors(vocab);
    assert!(cases.len() > 30);
    let mut failures = Vec::new();
    for (i, (text, want)) in cases.iter().enumerate() {
        // llama.cpp's test uses add_special = false, parse_special = false.
        let got = tok.encode_with(text, false, false);
        if &got != want {
            failures.push(format!(
                "case {i} {text:?}\n  want {want:?}\n  got  {got:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn check_roundtrip(vocab: &str) {
    let tok = load(vocab);
    for (text, _) in vectors(vocab) {
        let ids = tok.encode_with(&text, false, false);
        assert_eq!(tok.decode(&ids), text, "roundtrip of {text:?}");
    }
}

#[test]
fn qwen2_matches_llama_cpp() {
    check_vectors("qwen2");
}

#[test]
fn qwen35_matches_llama_cpp() {
    check_vectors("qwen35");
}

#[test]
fn gemma4_matches_llama_cpp() {
    check_vectors("gemma-4");
}

#[test]
fn gemma4_roundtrip() {
    check_roundtrip("gemma-4");
}

#[test]
fn qwen2_roundtrip() {
    check_roundtrip("qwen2");
}

#[test]
fn qwen35_roundtrip() {
    check_roundtrip("qwen35");
}

#[test]
fn metadata_and_special_ids() {
    for vocab in ["qwen2", "qwen35"] {
        let tok = load(vocab);
        assert!(tok.vocab_size() > 100_000, "{vocab}");
        let im_start = tok.token_id("<|im_start|>").unwrap();
        let im_end = tok.token_id("<|im_end|>").unwrap();
        assert!(tok.is_special(im_start) && tok.is_special(im_end));
        assert_eq!(tok.eot_id(), Some(im_end));
        assert_eq!(tok.token_to_piece(im_start), b"<|im_start|>");
        let hello = tok.encode("Hello", false)[0];
        assert!(!tok.is_special(hello));
        assert_eq!(tok.token_to_piece(hello), b"Hello");
        assert!(!tok.is_special(tok.vocab_size() as u32));
    }
}

#[test]
fn special_tokens_are_parsed_only_when_asked() {
    for vocab in ["qwen2", "qwen35"] {
        let tok = load(vocab);
        let im_start = tok.token_id("<|im_start|>").unwrap();
        let im_end = tok.token_id("<|im_end|>").unwrap();
        let text = "<|im_start|>user\nhi<|im_end|>\n";

        let on = tok.encode(text, true);
        assert_eq!(on[0], im_start, "{vocab}: {on:?}");
        assert!(on.contains(&im_end));
        assert_eq!(on.iter().filter(|&&i| i == im_start).count(), 1);
        // The text between and around the markers is encoded like plain text.
        let mut want = vec![im_start];
        want.extend(tok.encode_with("user\nhi", false, false));
        want.push(im_end);
        want.extend(tok.encode_with("\n", false, false));
        assert_eq!(on, want);
        assert_eq!(tok.decode(&on), "user\nhi\n"); // control tokens are hidden
        assert_eq!(tok.decode_with(&on, true), text);

        let off = tok.encode_with(text, false, false);
        assert!(!off.contains(&im_start) && !off.contains(&im_end));
        assert_eq!(tok.decode(&off), text);
    }
}

#[test]
fn empty_text_gives_no_tokens() {
    let tok = load("qwen2");
    assert!(tok.encode("", false).is_empty());
    assert_eq!(tok.decode(&[]), "");
}

#[test]
fn decode_stream_holds_back_split_characters() {
    let tok = load("qwen2");
    // U+1F999 (llama emoji) is 4 UTF-8 bytes; byte-level BPE can split it across tokens.
    let text = "a \u{1F999} and \u{00E9}x";
    let ids = tok.encode(text, false);
    assert_eq!(tok.decode(&ids), text);

    // Feeding one token at a time reproduces the text, and every chunk is valid UTF-8 by type.
    let mut stream = tok.decode_stream();
    let mut out = String::new();
    for &id in &ids {
        out.push_str(&stream.push(id));
    }
    out.push_str(&stream.finish());
    assert_eq!(out, text);

    // Force the split: one token per byte of the emoji.
    let bytes: Vec<u32> = "\u{1F999}"
        .bytes()
        .map(|b| tok.token_id(&gpt2_char(b).to_string()).unwrap())
        .collect();
    assert_eq!(bytes.len(), 4);
    let mut stream = tok.decode_stream();
    let chunks: Vec<String> = bytes.iter().map(|&id| stream.push(id)).collect();
    assert_eq!(chunks, ["", "", "", "\u{1F999}"]);
    assert_eq!(stream.finish(), "");

    // A stream that ends inside a character flushes a replacement character.
    let mut stream = tok.decode_stream();
    assert_eq!(stream.push(bytes[0]), "");
    assert_eq!(stream.push(bytes[1]), "");
    assert_eq!(stream.finish(), "\u{FFFD}");
}

#[test]
fn decode_stream_skips_control_tokens() {
    let tok = load("qwen35");
    let ids = tok.encode("<|im_start|>hi<|im_end|>", true);
    let mut stream = tok.decode_stream();
    let mut out: String = ids.iter().map(|&i| stream.push(i)).collect();
    out.push_str(&stream.finish());
    assert_eq!(out, "hi");
    let mut stream = tok.decode_stream().render_special(true);
    let mut out: String = ids.iter().map(|&i| stream.push(i)).collect();
    out.push_str(&stream.finish());
    assert_eq!(out, "<|im_start|>hi<|im_end|>");
}

#[test]
fn rejects_non_bpe_models() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("llama.gguf");
    let mut w = legion::gguf::GgufWriter::new();
    w.set(
        "tokenizer.ggml.model",
        &legion::gguf::Value::String("llama".into()),
    )
    .unwrap();
    w.write(&path).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let err = Tokenizer::from_gguf(&gguf).err().unwrap();
    assert!(err.to_string().contains("llama"), "{err}");
}

/// Release-mode timing of 100 KB of English/code/CJK text. Run with
/// `cargo test --release --test tokenizer encode_speed -- --ignored --nocapture`.
#[test]
#[ignore]
fn encode_speed() {
    let tok = load("qwen2");
    let unit = "The quick brown fox jumps over the lazy dog. fn main() { println!(\"hello, world {}\", 42); }\n\
                Ünïcödé text and 日本語のテキスト, numbers 3.14159 and 2026-10-16.\n";
    let text = unit.repeat(100_000 / unit.len() + 1);
    let text = &text[..text.floor_char_boundary(100_000)];
    let start = Instant::now();
    let ids = tok.encode(text, false);
    let dt = start.elapsed();
    println!(
        "encoded {} bytes -> {} tokens in {dt:?}",
        text.len(),
        ids.len()
    );
    assert_eq!(tok.decode(&ids), text);
    assert!(dt.as_millis() < 100, "{dt:?}");
}

/// A byte-level vocabulary plus `ab` and `123` as whole tokens, but no merges that build them,
/// under the pre-tokenizer `pre`.
fn merge_free_vocab(pre: &str) -> Tokenizer {
    use legion::gguf::{GgufWriter, Value};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let mut tokens: Vec<Value> = (0..=255u8)
        .map(|b| Value::String(gpt2_char(b).to_string()))
        .collect();
    tokens.push(Value::String("ab".into())); // 256
    tokens.push(Value::String("123".into())); // 257
    let mut w = GgufWriter::new();
    w.set("tokenizer.ggml.model", &Value::String("gpt2".into()))
        .unwrap();
    w.set("tokenizer.ggml.pre", &Value::String(pre.into()))
        .unwrap();
    w.set("tokenizer.ggml.tokens", &Value::Array(tokens))
        .unwrap();
    w.set(
        "tokenizer.ggml.merges",
        &Value::Array(vec![Value::String("x y".into())]),
    )
    .unwrap();
    w.write(&path).unwrap();
    Tokenizer::from_gguf(&GgufFile::open(&path).unwrap()).unwrap()
}

/// llama.cpp's llama3-family pre-tokenizers (`pixtral` is Nemotron-H's): digits split in runs
/// of up to three, and a pre-token that is a whole vocabulary entry is that token even without
/// the merges to build it (`ignore_merges`). The qwen2 rules do neither.
#[test]
fn llama3_pre_splits_digit_runs_and_ignores_merges() {
    let pixtral = merge_free_vocab("pixtral");
    assert_eq!(pixtral.encode_with("ab", false, false), vec![256]);
    assert_eq!(
        pixtral.encode_with("12345", false, false),
        vec![257, 52, 53]
    );
    assert_eq!(pixtral.encode_with("abc", false, false), vec![97, 98, 99]);
    let qwen2 = merge_free_vocab("qwen2");
    assert_eq!(qwen2.encode_with("ab", false, false), vec![97, 98]);
    assert_eq!(qwen2.encode_with("123", false, false), vec![49, 50, 51]);
}
