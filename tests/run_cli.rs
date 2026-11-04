//! `legion run`: error exit, and (with `LEGION_TEST_GGUF`) greedy generation on the CPU.

use std::process::{Command, Output};

fn legion(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_legion"))
        .arg("run")
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn missing_model_exits_1_with_message() {
    let out = legion(&["--model", "/nonexistent.gguf", "hi"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(!out.stderr.is_empty());
}

#[test]
fn model_flag_is_required() {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .env_remove("LEGION_MODEL")
        .args(["run", "hi"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--model"));
}

#[test]
#[ignore = "needs LEGION_TEST_GGUF (a Qwen3.5 GGUF); runs on the CPU"]
fn greedy_run_is_deterministic_and_prints_stats() {
    let gguf = std::env::var("LEGION_TEST_GGUF").expect("set LEGION_TEST_GGUF");
    let args = [
        "--model",
        &gguf,
        "--threads",
        "8",
        "--temperature",
        "0",
        "--max-tokens",
        "8",
        "--raw",
        "--stats",
        "The capital of France is",
    ];
    let a = legion(&args);
    let b = legion(&args);
    for o in [&a, &b] {
        assert_eq!(
            o.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    assert!(!a.stdout.is_empty());
    assert_eq!(a.stdout, b.stdout);
    let err = String::from_utf8_lossy(&a.stderr);
    assert!(
        err.lines()
            .any(|l| l.starts_with("prompt ") && l.contains(" tok/s) · output ")),
        "no stats line in: {err}"
    );
}
