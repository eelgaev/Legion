//! `legion gguf inspect` and `diff` on files written with `GgufWriter`.

use std::path::{Path, PathBuf};
use std::process::Command;

use legion::ggml::ffi;
use legion::gguf::{GgufWriter, Value};
use serde_json::Value as Json;

/// Deliberately ends without a newline and has multiple lines and quotes.
const CHAT_TEMPLATE: &str =
    "{% for m in messages %}\n<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}";

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// A small GGUF: scalars, a short string array, a 100-entry "vocab" (long enough
/// to be summarised) and two tensors.
fn write_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("tiny.gguf");
    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("qwen3next".into()))
        .unwrap()
        .set("qwen3next.block_count", &Value::U32(2))
        .unwrap()
        .set("qwen3next.rope.freq_base", &Value::F32(1e6))
        .unwrap()
        .set("tokenizer.ggml.add_bos_token", &Value::Bool(false))
        .unwrap()
        .set(
            "tokenizer.ggml.tokens",
            &Value::Array((0..100).map(|i| Value::String(format!("<t{i}>"))).collect()),
        )
        .unwrap()
        .set(
            "tokenizer.ggml.pre",
            &Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
        )
        .unwrap()
        .set("tokenizer.ggml.model", &Value::String("gpt2".into()))
        .unwrap()
        .set("tokenizer.ggml.bos_token_id", &Value::U32(1))
        .unwrap()
        .set("tokenizer.ggml.eos_token_id", &Value::U32(7))
        .unwrap()
        .set("tokenizer.ggml.add_eos_token", &Value::Bool(true))
        .unwrap()
        .set(
            "tokenizer.ggml.merges",
            &Value::Array(vec![Value::String("a b".into()); 3]),
        )
        .unwrap()
        .set(
            "tokenizer.chat_template",
            &Value::String(CHAT_TEMPLATE.into()),
        )
        .unwrap();
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    w.add_tensor(
        "blk.0.attn_q.weight",
        ffi::ggml_type_GGML_TYPE_F32,
        &[3, 2],
        &f32_bytes(&a),
    )
    .unwrap();
    w.add_tensor(
        "blk.0.ffn_up.weight",
        ffi::ggml_type_GGML_TYPE_Q8_0,
        &[64],
        &[0u8; 68],
    )
    .unwrap();
    w.write(&path).unwrap();
    path
}

fn inspect(path: &Path, flags: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .arg("gguf")
        .arg("inspect")
        .arg(path)
        .args(flags)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "legion gguf inspect {flags:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn inspect_json(path: &Path, flags: &[&str]) -> Json {
    let stdout = inspect(path, flags);
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("bad JSON {stdout:?}: {e}"))
}

#[test]
fn json_output_is_valid_and_complete() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let v = inspect_json(&path, &["--json", "--tensors"]);
    assert_eq!(v["file"], path.display().to_string());
    assert_eq!(v["version"], 3);
    assert_eq!(v["architecture"], "qwen3next");
    assert!(v["alignment"].is_number(), "{v}");
    assert!(v["data_offset"].is_number(), "{v}");
    assert_eq!(v["metadata"]["general.architecture"], "qwen3next");
    assert_eq!(v["metadata"]["qwen3next.block_count"], 2);
    assert_eq!(v["metadata"]["qwen3next.rope.freq_base"], 1e6);
    assert_eq!(v["metadata"]["tokenizer.ggml.add_bos_token"], false);

    let ts = v["tensors"].as_array().unwrap();
    assert_eq!(ts.len(), 2);
    assert_eq!(ts[0]["name"], "blk.0.attn_q.weight");
    assert_eq!(ts[0]["type"], "f32");
    assert_eq!(ts[0]["shape"], Json::Array(vec![3.into(), 2.into()]));
    assert_eq!(ts[0]["bytes"], 24);
    assert_eq!(ts[1]["name"], "blk.0.ffn_up.weight");
    assert_eq!(ts[1]["type"], "q8_0");
    assert_eq!(ts[1]["bytes"], 68);
    // File order: the second tensor sits at or after the first one's data.
    assert!(ts[0]["offset"].as_u64().unwrap() >= v["data_offset"].as_u64().unwrap());
    assert!(ts[1]["offset"].as_u64().unwrap() > ts[0]["offset"].as_u64().unwrap());
}

#[test]
fn json_omits_tensors_without_the_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let v = inspect_json(&path, &["--json"]);
    assert!(v.get("tensors").is_none(), "{v}");
    assert!(v["metadata"].is_object(), "{v}");
}

#[test]
fn long_arrays_are_summarised_unless_full() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let v = inspect_json(&path, &["--json"]);
    let vocab = &v["metadata"]["tokenizer.ggml.tokens"];
    assert_eq!(vocab["type"], "array[string]");
    assert_eq!(vocab["len"], 100);
    let head = vocab["head"].as_array().unwrap();
    assert_eq!(head.len(), 8);
    assert_eq!(head[0], "<t0>");
    assert_eq!(head[7], "<t7>");
    // Short arrays are never summarised.
    assert_eq!(
        v["metadata"]["tokenizer.ggml.pre"],
        Json::Array(vec!["a".into(), "b".into()])
    );

    let v = inspect_json(&path, &["--json", "--full"]);
    let vocab = v["metadata"]["tokenizer.ggml.tokens"].as_array().unwrap();
    assert_eq!(vocab.len(), 100);
    assert_eq!(vocab[99], "<t99>");
}

#[test]
fn text_table_lists_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let out = inspect(&path, &["--tensors"]);
    assert!(out.contains("architecture:  qwen3next"), "{out}");
    assert!(out.contains("MiB"), "{out}");
    assert!(out.contains("blk.0.attn_q.weight"), "{out}");
    assert!(out.contains("blk.0.ffn_up.weight"), "{out}");
    assert!(out.contains("[3, 2]"), "{out}");

    // Without --tensors the metadata stays, the table does not.
    let out = inspect(&path, &[]);
    assert!(out.contains("general.architecture"), "{out}");
    assert!(!out.contains("blk.0.attn_q.weight"), "{out}");
}

#[test]
fn text_output_has_tokenizer_section_and_skips_summarised_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let out = inspect(&path, &[]);
    assert!(out.contains("tokenizer:"), "{out}");
    assert!(out.contains("gpt2"), "{out}");
    assert!(out.contains("100"), "{out}");
    assert!(out.contains("bos          1 \"<t1>\""), "{out}");
    assert!(out.contains("eos          7 \"<t7>\""), "{out}");
    assert!(out.contains("pad          -"), "{out}");
    assert!(out.contains("add_bos      false"), "{out}");
    assert!(out.contains("add_eos      true"), "{out}");
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.chat_template",
    ] {
        assert!(!out.contains(key), "{key} should be summarised: {out}");
    }
    // Other tokenizer keys stay in the listing.
    assert!(out.contains("tokenizer.ggml.model"), "{out}");
}

#[test]
fn json_has_tokenizer_object() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let v = inspect_json(&path, &["--json"]);
    let t = &v["tokenizer"];
    assert_eq!(t["model"], "gpt2");
    assert_eq!(t["vocab_size"], 100);
    assert_eq!(t["merges"], 3);
    assert_eq!(t["special_tokens"]["bos"]["id"], 1);
    assert_eq!(t["special_tokens"]["bos"]["token"], "<t1>");
    assert_eq!(t["special_tokens"]["eos"]["token"], "<t7>");
    assert!(t["special_tokens"].get("pad").is_none(), "{t}");
    assert_eq!(t["add_bos_token"], false);
    assert_eq!(t["add_eos_token"], true);
    assert_eq!(t["has_chat_template"], true);
}

#[test]
fn chat_template_is_printed_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    assert_eq!(inspect(&path, &["--chat-template"]), CHAT_TEMPLATE);
}

#[test]
fn chat_template_errors_when_missing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bare.gguf");
    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("x".into()))
        .unwrap();
    w.write(&path).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["gguf", "inspect"])
        .arg(&path)
        .arg("--chat-template")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no tokenizer.chat_template"));
}

/// Runs `legion gguf diff a b`, returning the exit code and stdout.
fn diff(a: &Path, b: &Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .arg("gguf")
        .arg("diff")
        .arg(a)
        .arg(b)
        .output()
        .unwrap();
    (out.status.code(), String::from_utf8(out.stdout).unwrap())
}

/// Like [`write_fixture`] but with `qwen3next.block_count` changed to 3,
/// `general.name` added, `qwen3next.rope.freq_base` removed and
/// `blk.0.ffn_up.weight` stored as f32 instead of q8_0.
fn write_variant_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("variant.gguf");
    let mut w = GgufWriter::new();
    w.set("general.architecture", &Value::String("qwen3next".into()))
        .unwrap()
        .set("general.name", &Value::String("variant".into()))
        .unwrap()
        .set("qwen3next.block_count", &Value::U32(3))
        .unwrap()
        .set("tokenizer.ggml.add_bos_token", &Value::Bool(false))
        .unwrap()
        .set(
            "tokenizer.ggml.tokens",
            &Value::Array((0..100).map(|i| Value::String(format!("<t{i}>"))).collect()),
        )
        .unwrap()
        .set(
            "tokenizer.ggml.pre",
            &Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
        )
        .unwrap()
        .set("tokenizer.ggml.model", &Value::String("gpt2".into()))
        .unwrap()
        .set("tokenizer.ggml.bos_token_id", &Value::U32(1))
        .unwrap()
        .set("tokenizer.ggml.eos_token_id", &Value::U32(7))
        .unwrap()
        .set("tokenizer.ggml.add_eos_token", &Value::Bool(true))
        .unwrap()
        .set(
            "tokenizer.ggml.merges",
            &Value::Array(vec![Value::String("a b".into()); 3]),
        )
        .unwrap()
        .set(
            "tokenizer.chat_template",
            &Value::String(CHAT_TEMPLATE.into()),
        )
        .unwrap();
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    w.add_tensor(
        "blk.0.attn_q.weight",
        ffi::ggml_type_GGML_TYPE_F32,
        &[3, 2],
        &f32_bytes(&a),
    )
    .unwrap();
    let b: Vec<f32> = (0..64).map(|i| i as f32).collect();
    w.add_tensor(
        "blk.0.ffn_up.weight",
        ffi::ggml_type_GGML_TYPE_F32,
        &[64],
        &f32_bytes(&b),
    )
    .unwrap();
    w.write(&path).unwrap();
    path
}

/// Writes a 3-shard split model; `variant` renames the architecture and stores
/// `s1.a` as f16 instead of f32. Returns the shard paths in order.
fn write_split_fixture(dir: &Path, variant: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for i in 0..3usize {
        let path = dir.join(format!("model-{:05}-of-00003.gguf", i + 1));
        let mut g = GgufWriter::new();
        if i == 0 {
            let arch = if variant { "test2" } else { "test" };
            g.set("general.architecture", &Value::String(arch.into()))
                .unwrap();
        }
        g.set("split.no", &Value::U16(i as u16)).unwrap();
        g.set("split.count", &Value::U16(3)).unwrap();
        g.set("split.tensors.count", &Value::I32(6)).unwrap();
        for (name, n) in [
            (format!("s{i}.a"), 16usize),
            (format!("s{i}.b"), 8 * (i + 1)),
        ] {
            let (ty, data) = if variant && name == "s1.a" {
                (ffi::ggml_type_GGML_TYPE_F16, vec![0u8; n * 2])
            } else {
                (ffi::ggml_type_GGML_TYPE_F32, vec![0u8; n * 4])
            };
            g.add_tensor(&name, ty, &[n as i64], &data).unwrap();
        }
        g.write(&path).unwrap();
        paths.push(path);
    }
    paths
}

#[test]
fn diff_of_identical_files_is_clean() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let a = write_fixture(dir_a.path());
    let b = write_fixture(dir_b.path());

    let (code, out) = diff(&a, &b);
    assert_eq!(code, Some(0), "{out}");
    assert_eq!(
        out,
        "metadata:\n\ntensors:\n\n0 metadata and 0 tensor differences\n"
    );
}

#[test]
fn diff_lists_metadata_and_tensor_changes() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let a = write_fixture(dir_a.path());
    let b = write_variant_fixture(dir_b.path());

    let (code, out) = diff(&a, &b);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(
        out,
        "metadata:\n\
         + general.name = \"variant\"\n\
         ~ qwen3next.block_count: 2 -> 3\n\
         - qwen3next.rope.freq_base = 1000000\n\
         \n\
         tensors:\n\
         ~ blk.0.ffn_up.weight: q8_0 [64] -> f32 [64]\n\
         \n\
         3 metadata and 1 tensor differences\n"
    );
}

#[test]
fn diff_opens_split_models_through_any_shard() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let a = write_split_fixture(dir_a.path(), false);
    let b = write_split_fixture(dir_b.path(), true);

    // The same model seen from two different shards compares equal.
    let (code, out) = diff(&a[0], &a[2]);
    assert_eq!(code, Some(0), "{out}");

    // A shard-2 view of a against a shard-3 view of b still sees every
    // metadata key (first shard) and every tensor (all shards).
    let (code, out) = diff(&a[1], &b[2]);
    assert_eq!(code, Some(1), "{out}");
    assert_eq!(
        out,
        "metadata:\n\
         ~ general.architecture: \"test\" -> \"test2\"\n\
         \n\
         tensors:\n\
         ~ s1.a: f32 [16] -> f16 [16]\n\
         \n\
         1 metadata and 1 tensor differences\n"
    );
}
