//! `legion tokenize` against llama.cpp's Qwen3.5 vocab fixture (`tests/data/`).

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use legion::gguf::GgufFile;
use legion::tokenizer::Tokenizer;
use serde_json::Value as Json;

fn model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join("ggml-vocab-qwen35.gguf")
}

fn tokenizer() -> Tokenizer {
    let gguf = GgufFile::open(model()).unwrap();
    Tokenizer::from_gguf(&gguf).unwrap()
}

fn tokenize(args: &[&str], stdin: Option<&str>) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
    cmd.args(["tokenize", "--model"]).arg(model()).args(args);
    // `spawn` + `wait_with_output` only capture what we pipe explicitly.
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd.spawn().unwrap();
    if let Some(text) = stdin {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}

/// Runs the command expecting success and returns stdout as text.
fn ok(args: &[&str], stdin: Option<&str>) -> String {
    let out = tokenize(args, stdin);
    assert!(
        out.status.success(),
        "legion tokenize {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn ok_json(args: &[&str]) -> Json {
    let stdout = ok(args, None);
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("bad JSON {stdout:?}: {e}"))
}

#[test]
fn ids_only_matches_library_encode() {
    let expected = tokenizer().encode_with("Hello world", false, false);
    let stdout = ok(&["--ids-only", "Hello world"], None);
    let ids: Vec<u32> = stdout
        .trim_end()
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(ids, expected);
    // Exactly one line, newline-terminated, nothing else.
    assert!(
        stdout.ends_with('\n') && stdout.matches('\n').count() == 1,
        "{stdout:?}"
    );
}

#[test]
fn text_output_lists_pieces_and_total() {
    let expected = tokenizer().encode_with("The capital of France is", false, false);
    let stdout = ok(&["The capital of France is"], None);
    let mut lines = stdout.lines();
    let last = lines.next_back().unwrap();
    assert_eq!(
        last,
        format!("total: {} tokens", expected.len()),
        "{stdout}"
    );
    let ids: Vec<u32> = lines
        .map(|l| {
            let (id, piece) = l
                .split_once('\t')
                .unwrap_or_else(|| panic!("no tab: {l:?}"));
            // Pieces are {:?}-escaped, so the leading space of " Paris"-style pieces is visible.
            assert!(piece.starts_with('"') && piece.ends_with('"'), "{l:?}");
            id.parse().unwrap()
        })
        .collect();
    assert_eq!(ids, expected);
}

#[test]
fn json_output_has_tokens_and_count() {
    let expected = tokenizer().encode_with("Hello world", false, false);
    let v = ok_json(&["--json", "Hello world"]);
    assert_eq!(v["count"].as_u64().unwrap() as usize, expected.len());
    let tokens = v["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), expected.len());
    for (t, &id) in tokens.iter().zip(&expected) {
        assert_eq!(t["id"].as_u64().unwrap() as u32, id);
        // `piece` is the raw lossy UTF-8 text, not {:?}-escaped.
        assert_eq!(
            t["piece"].as_str().unwrap(),
            String::from_utf8_lossy(tokenizer().token_to_piece(id))
        );
    }
}

#[test]
fn stdin_is_used_when_text_is_omitted() {
    let via_stdin = ok(&["--ids-only"], Some("Hello world"));
    let via_arg = ok(&["--ids-only", "Hello world"], None);
    assert_eq!(via_stdin, via_arg);
}

#[test]
fn ids_only_conflicts_with_json() {
    let out = tokenize(&["--ids-only", "--json", "x"], None);
    assert!(!out.status.success());
}

#[test]
fn special_tokens_are_single_ids_only_with_special() {
    let tok = tokenizer();
    let im_start = tok
        .token_id("<|im_start|>")
        .expect("fixture has <|im_start|>");
    let text = "<|im_start|>";

    let plain = ok_json(&["--json", text]);
    assert!(plain["count"].as_u64().unwrap() > 1, "{plain}");

    let special = ok_json(&["--json", "--special", text]);
    assert_eq!(special["count"].as_u64().unwrap(), 1, "{special}");
    assert_eq!(
        special["tokens"][0]["id"].as_u64().unwrap(),
        im_start as u64
    );
    assert_eq!(special["tokens"][0]["piece"].as_str().unwrap(), text);
}

#[test]
fn missing_model_file_fails() {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args([
            "tokenize",
            "--model",
            "tests/data/does-not-exist.gguf",
            "hi",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("does-not-exist.gguf"), "{stderr}");
}
