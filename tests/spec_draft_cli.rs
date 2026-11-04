mod common;

use std::process::{Command, Output};

fn parse(temperature: &str, confidence: &str, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_legion"))
        .env_remove("LEGION_MODEL")
        .env("LEGION_SPEC_DRAFT_TEMPERATURE", temperature)
        .env("LEGION_SPEC_DRAFT_MIN_CONFIDENCE", confidence)
        // Stop after parsing, before loading a model or starting a server.
        .args(["serve", "--engine", "model"])
        .args(args)
        .output()
        .unwrap()
}

fn assert_parsed(out: Output) {
    assert_eq!(out.status.code(), Some(1));
    let error = String::from_utf8_lossy(&out.stderr);
    assert!(error.contains("no model to load"), "{error}");
}

#[test]
fn draft_environment_values_are_validated_by_clap() {
    assert_parsed(parse("0.75", "0.3", &[]));
    for (temperature, confidence, flag) in [
        ("-1", "0.3", "--spec-draft-temperature"),
        ("0.75", "1.1", "--spec-min-confidence"),
    ] {
        let out = parse(temperature, confidence, &[]);
        assert_eq!(out.status.code(), Some(2));
        let error = String::from_utf8_lossy(&out.stderr);
        assert!(error.contains(flag), "{error}");
    }
}

#[test]
fn explicit_draft_flags_override_environment() {
    assert_parsed(parse(
        "invalid",
        "invalid",
        &[
            "--spec-draft-temperature",
            "0.5",
            "--spec-min-confidence",
            "0.2",
        ],
    ));
}
