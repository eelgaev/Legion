//! CLI settings for DFlash and DSpark draft selection.

#[derive(clap::Args)]
pub(super) struct DraftSettings {
    /// DFlash/DSpark draft temperature, at least 0 (0: greedy point-mass drafts).
    #[arg(
        long,
        env = "LEGION_SPEC_DRAFT_TEMPERATURE",
        default_value_t = 0.0,
        value_parser = temperature
    )]
    pub spec_draft_temperature: f32,
    /// DSpark confidence threshold in [0, 1]; stop before a draft below it (unset: no cut).
    #[arg(
        long,
        env = "LEGION_SPEC_DRAFT_MIN_CONFIDENCE",
        value_parser = confidence
    )]
    pub spec_min_confidence: Option<f32>,
    /// DFlash/DSpark: positions of context a sequence keeps for drafting, a ring of the newest
    /// (default 4096). Replaces LEGION_SPEC_DRAFT_WINDOW.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub spec_draft_window: Option<u32>,
    /// DFlash/DSpark: the weights the drafter drafts with. `model` (default): as stored.
    /// `q8_0`: Q8_0 copies of a Hugging Face checkpoint's 16- and 32-bit matrices, made at load
    /// (z-lab's Qwen3.8-27B-DFlash2: 3.3 -> 1.75 GB a draft pass reads). `q4`: 4-bit copies of
    /// the same matrices (Q4_K; Q4_0 for rows not a multiple of 256 wide). Drafts only: every
    /// token is verified by the target, so the output does not change; only acceptance can.
    /// A GGUF drafter is read as stored. An FP8 Hugging Face drafter's block-scaled matrices
    /// are read as Q8_0 with every value.
    #[arg(long, value_enum, default_value_t = legion::spec::DraftWeights::Model)]
    pub spec_draft_weights: legion::spec::DraftWeights,
    /// The output head rows drafts are scored with: by a DFlash drafter that borrows the
    /// target's head (`--spec dflash`), or by the MTP head of qwen4exp and qwen35 (`--spec
    /// mtp`). `model`: as stored (all of them, or the first --spec-draft-vocab).
    /// `q4`: a 4-bit copy of those rows made at load (Q4_K, the head's scales folded in), which
    /// a draft pass reads instead (Qwen3.8-27B's FP8 head, 131,072 rows: 671 -> 377 MB).
    /// `auto` (default): `q4` for the qwen4exp MTP head on an integrated GPU (GB10) when the
    /// target's head is 16- or 32-bit (Qwen3.8-Flash-Next-NVFP4's BF16 head), else `model`
    /// (always `model` for DFlash). Drafts only: every token is verified by the target, so the
    /// output does not change; only acceptance can. Ignored for a DFlash drafter with its own
    /// head.
    #[arg(long, value_enum, default_value_t = legion::spec::DraftLmHead::Auto)]
    pub spec_draft_lm_head: legion::spec::DraftLmHead,
    /// DFlash 2 (opt-in): the draft pass over several sequences attends all of them in one
    /// flash attention per drafter layer, their caches slots of one pool (a slot for each of
    /// --max-seqs, made when the first sequence drafts), instead of one small attention per
    /// sequence and layer. Drafts only: every token is verified by the target, but the drafts'
    /// numerics differ slightly, so with 2 or more sequences drafting at once acceptance (and
    /// under --deterministic the text) can move. One sequence drafts as without it.
    #[arg(long)]
    pub spec_draft_batched_attn: bool,
}

fn temperature(value: &str) -> Result<f32, String> {
    let value = value.parse::<f32>().map_err(|e| e.to_string())?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err("draft temperature must be at least 0".into())
    }
}

fn confidence(value: &str) -> Result<f32, String> {
    let value = value.parse::<f32>().map_err(|e| e.to_string())?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("minimum confidence must be in [0, 1]".into())
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, FromArgMatches};
    use legion::spec::SpecConfig;

    use crate::cli::{Cli, Command};

    fn parse(args: &[&str]) -> Result<SpecConfig, clap::Error> {
        // Isolate these tests from the caller's environment without changing process state.
        let command = Cli::command().mut_subcommand("serve", |c| {
            c.mut_arg("spec_draft_temperature", |a| a.env(None::<&str>))
                .mut_arg("spec_min_confidence", |a| a.env(None::<&str>))
        });
        let matches = command
            .try_get_matches_from(["legion", "serve"].into_iter().chain(args.iter().copied()))?;
        match Cli::from_arg_matches(&matches)?.command {
            Command::Serve(args) => Ok(args.engine.spec().unwrap()),
            _ => unreachable!(),
        }
    }

    #[test]
    fn draft_flags_reach_spec_config() {
        for kind in ["dflash", "dspark"] {
            let base = ["--spec", kind, "--spec-draft", "draft.gguf"];
            let defaults = parse(&base).unwrap();
            assert_eq!(defaults.draft_temperature, 0.0);
            assert_eq!(defaults.min_confidence, None);
            for (temperature, confidence) in [("0", "0"), ("0.75", "0.3"), ("2", "1")] {
                let args = [
                    base.as_slice(),
                    &[
                        "--spec-draft-temperature",
                        temperature,
                        "--spec-min-confidence",
                        confidence,
                    ],
                ]
                .concat();
                let config = parse(&args).unwrap();
                assert_eq!(
                    config.draft_temperature,
                    temperature.parse::<f32>().unwrap()
                );
                assert_eq!(config.min_confidence, Some(confidence.parse().unwrap()));
            }
        }
    }

    #[test]
    fn invalid_draft_flags_are_clap_errors() {
        for (flag, values) in [
            ("--spec-draft-temperature", &["-0.1", "NaN", "abc"][..]),
            (
                "--spec-min-confidence",
                &["-0.1", "1.1", "NaN", "inf", "abc"],
            ),
        ] {
            for value in values {
                let arg = format!("{flag}={value}");
                let error = parse(&[&arg]).unwrap_err();
                assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
                assert!(error.to_string().contains(flag));
            }
        }
    }

    #[test]
    fn serve_help_documents_draft_flags() {
        let help = Cli::command()
            .try_get_matches_from(["legion", "serve", "--help"])
            .unwrap_err();
        assert_eq!(help.kind(), clap::error::ErrorKind::DisplayHelp);
        let help = help.to_string();
        for text in [
            "--spec-draft-temperature",
            "--spec-min-confidence",
            "LEGION_SPEC_DRAFT_TEMPERATURE",
            "LEGION_SPEC_DRAFT_MIN_CONFIDENCE",
            "greedy point-mass drafts",
            "unset: no cut",
        ] {
            assert!(help.contains(text), "missing {text} in {help}");
        }
    }
}
