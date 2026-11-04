use crate::cli::{Cli, Command};
use clap::Parser;

#[test]
fn flags_reach_config() {
    for extra in [
        vec![],
        vec![
            "--spec-fixed",
            "--spec-max-batch",
            "1",
            "--spec-p-floor",
            "0.2",
            "--spec-gate-cost-model",
            "off",
        ],
    ] {
        let Command::Serve(args) = Cli::try_parse_from(
            ["legion", "serve", "--spec", "ngram"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .unwrap()
        .command
        else {
            unreachable!()
        };
        let cfg = args.engine.spec().unwrap();
        assert_eq!(cfg.fixed, !extra.is_empty());
        assert_eq!(cfg.max_batch, if extra.is_empty() { 8 } else { 1 });
        assert_eq!(cfg.p_floor, if extra.is_empty() { 0.5 } else { 0.2 });
        assert_eq!(cfg.gate_cost_model, extra.is_empty());
        // Not given: the engine may pick the layout's default (#814).
        assert_eq!(cfg.gate_cost_model_given, !extra.is_empty());
    }
}

#[test]
fn gate_cost_model_takes_on_and_off() {
    let parse = |value: &str| {
        let flag = format!("--spec-gate-cost-model={value}");
        let Command::Serve(args) = Cli::try_parse_from(["legion", "serve", "--spec", "mtp", &flag])
            .ok()?
            .command
        else {
            unreachable!()
        };
        Some(args.engine.spec().unwrap().gate_cost_model)
    };
    assert_eq!(parse("on"), Some(true));
    assert_eq!(parse("off"), Some(false));
    assert_eq!(parse("maybe"), None);
}

#[test]
fn row_cost_defaults_on_and_turns_off() {
    let parse = |extra: &[&str]| {
        let Command::Serve(args) = Cli::try_parse_from(
            ["legion", "serve", "--spec", "mtp"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .ok()?
        .command
        else {
            unreachable!()
        };
        Some(args.engine.spec().unwrap().row_cost)
    };
    assert_eq!(parse(&[]), Some(true));
    assert_eq!(parse(&["--spec-row-cost", "off"]), Some(false));
    assert_eq!(parse(&["--spec-row-cost=on"]), Some(true));
    assert_eq!(parse(&["--spec-row-cost=maybe"]), None);
}

#[test]
fn pad_longest_defaults_off_and_turns_on() {
    let parse = |extra: &[&str]| {
        let Command::Serve(args) = Cli::try_parse_from(
            ["legion", "serve", "--spec", "mtp"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .ok()?
        .command
        else {
            unreachable!()
        };
        Some(args.engine.spec().unwrap().pad_longest)
    };
    assert_eq!(parse(&[]), Some(false));
    assert_eq!(parse(&["--spec-pad-longest", "on"]), Some(true));
    assert_eq!(parse(&["--spec-pad-longest=off"]), Some(false));
    assert_eq!(parse(&["--spec-pad-longest=maybe"]), None);
}

#[test]
fn invalid_probability_is_rejected() {
    for value in ["0", "1", "-0.1", "1.1", "NaN", "inf", "abc"] {
        let flag = format!("--spec-p-floor={value}");
        assert!(Cli::try_parse_from(["legion", "serve", &flag]).is_err());
    }
}

#[test]
fn max_rows_defaults_to_no_cap() {
    let parse = |extra: &[&str]| {
        let Command::Serve(args) = Cli::try_parse_from(
            ["legion", "serve", "--spec", "mtp"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .unwrap()
        .command
        else {
            unreachable!()
        };
        args.engine.spec().unwrap().max_rows
    };
    assert_eq!(parse(&[]), 0);
    assert_eq!(parse(&["--spec-max-rows", "63"]), 63);
}
