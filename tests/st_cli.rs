//! `legion st inspect` on a hand-written checkpoint directory.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value as Json;

/// Write a minimal checkpoint: `model.safetensors` with one F32 tensor and a `config.json`
/// containing `config`.
fn write_checkpoint(dir: &Path, config: &str) -> PathBuf {
    let header = serde_json::to_vec(&serde_json::json!({
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    }))
    .unwrap();
    let mut image = Vec::new();
    image.extend_from_slice(&(header.len() as u64).to_le_bytes());
    image.extend_from_slice(&header);
    image.extend_from_slice(&[0u8; 8]);
    std::fs::write(dir.join("model.safetensors"), image).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    dir.to_path_buf()
}

fn inspect_ok(dir: &Path, flags: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .arg("st")
        .arg("inspect")
        .arg(dir)
        .args(flags)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "legion st inspect {flags:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

const BOGUS_CONFIG: &str =
    r#"{"model_type": "fake", "quantization_config": {"quant_method": "made-up"}}"#;

#[test]
fn unknown_quant_method_is_listed_not_fatal() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = write_checkpoint(tmp.path(), BOGUS_CONFIG);
    let out = inspect_ok(&dir, &[]);
    assert!(out.contains("quantization:  unknown ("), "{out}");
    assert!(out.contains("made-up"), "{out}");
    // The rest of the report is still printed.
    assert!(out.contains("model_type:    \"fake\""), "{out}");
    assert!(out.contains("tensors:       1"), "{out}");
}

#[test]
fn unknown_quant_method_in_json() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = write_checkpoint(tmp.path(), BOGUS_CONFIG);
    let out = inspect_ok(&dir, &["--json"]);
    let v: Json = serde_json::from_str(&out).unwrap_or_else(|e| panic!("bad JSON {out:?}: {e}"));
    let err = v["quantization"]["error"]
        .as_str()
        .unwrap_or_else(|| panic!("no quantization.error in {v}"));
    assert!(err.contains("made-up"), "{err}");
    // The rest of the document is still there.
    assert_eq!(v["tensor_count"], 1);
    assert_eq!(v["model_type"], "fake");
}

#[test]
fn known_quant_method_prints_as_before() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = write_checkpoint(
        tmp.path(),
        r#"{"model_type": "fake", "quantization_config": {"quant_method": "fp8"}}"#,
    );
    let out = inspect_ok(&dir, &[]);
    assert!(out.contains("quantization:  fp8\n"), "{out}");
    let out = inspect_ok(&dir, &["--json"]);
    let v: Json = serde_json::from_str(&out).unwrap_or_else(|e| panic!("bad JSON {out:?}: {e}"));
    assert_eq!(
        v["quantization"],
        serde_json::json!({"fp8": {"block": null}})
    );
}
