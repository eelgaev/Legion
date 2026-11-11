//! Command-line parsing of `legion serve`'s engine settings.

use clap::Parser as _;

use super::super::{Cli, Command};
use super::*;

fn spec(argv: &[&str]) -> anyhow::Result<SpecConfig> {
    let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
    match Cli::try_parse_from(argv).unwrap().command {
        Command::Serve(args) => args.engine.spec(),
        _ => unreachable!("the parser returned a command other than serve"),
    }
}

fn residency(argv: &[&str]) -> ResidencyConfig {
    let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
    match Cli::try_parse_from(argv).unwrap().command {
        Command::Serve(args) => args.engine.model.residency(),
        _ => unreachable!("the parser returned a command other than serve"),
    }
}

#[test]
fn expert_heat_pass_cap_flag() {
    assert_eq!(residency(&[]).heat_pass_cap, 0.0);
    assert_eq!(ResidencyConfig::default().heat_pass_cap, 0.0);
    let c = residency(&["--expert-heat-pass-cap", "1"]);
    assert_eq!(c.heat_pass_cap, 1.0);
    for bad in ["-1", "nan", "inf"] {
        let argv = ["legion", "serve", "--expert-heat-pass-cap", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad}");
    }
}

#[test]
fn expert_profile_split_only_flag() {
    assert!(!residency(&[]).profile_split_only);
    assert!(!ResidencyConfig::default().profile_split_only);
    let c = residency(&["--expert-profile", "p.bin", "--expert-profile-split-only"]);
    assert!(c.profile_split_only);
    assert_eq!(c.profile.as_deref(), Some(std::path::Path::new("p.bin")));
    // It needs a profile.
    let argv = ["legion", "serve", "--expert-profile-split-only"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn expert_split_misses_flag() {
    assert_eq!(residency(&[]).split_misses, None);
    assert_eq!(ResidencyConfig::default().split_misses, None);
    let c = residency(&["--expert-split-misses", "m.json"]);
    assert_eq!(
        c.split_misses.as_deref(),
        Some(std::path::Path::new("m.json"))
    );
    // A profile sizes the layers itself.
    let argv = [
        "legion",
        "serve",
        "--expert-split-misses",
        "m.json",
        "--expert-profile",
        "p.bin",
    ];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn expert_swap_rule_flags() {
    let d = residency(&[]);
    assert_eq!(d.swaps, SwapRule::default());
    assert_eq!(d.decay, ResidencyConfig::default().decay);
    // Values other than the defaults, so an unwired flag shows.
    let c = residency(&[
        "--expert-swaps",
        "5",
        "--expert-swap-margin",
        "3",
        "--expert-swap-ratio",
        "0.5",
        "--expert-heat-decay",
        "0.9",
    ]);
    assert_eq!(
        c.swaps,
        SwapRule {
            budget: 5,
            margin: 3.0,
            ratio: 0.5
        }
    );
    assert_eq!(c.decay, 0.9);
    // Out-of-range values are refused, not clamped.
    for bad in [
        &["--expert-heat-decay", "nan"][..],
        &["--expert-heat-decay", "0"],
        &["--expert-heat-decay", "1.5"],
        &["--expert-swap-margin", "-1"],
        &["--expert-swap-ratio", "inf"],
    ] {
        let argv = ["legion", "serve"].into_iter().chain(bad.iter().copied());
        assert!(Cli::try_parse_from(argv).is_err(), "{bad:?}");
    }
    // The environment sets them too (a variable no other test reads).
    // SAFETY: no other test reads or writes LEGION_EXPERT_SWAP_RATIO.
    unsafe { std::env::set_var("LEGION_EXPERT_SWAP_RATIO", "0.75") };
    let e = residency(&[]);
    unsafe { std::env::remove_var("LEGION_EXPERT_SWAP_RATIO") };
    assert_eq!(e.swaps.ratio, 0.75);
}

#[test]
fn expert_swaps_device_flag() {
    // Default: no per-device budgets, --expert-swaps everywhere.
    assert!(residency(&[]).device_swaps.is_empty());
    assert_eq!(
        residency(&[]).device_swaps,
        ResidencyConfig::default().device_swaps
    );
    let c = residency(&["--expert-swaps-device", "48,24"]);
    assert_eq!(c.device_swaps, vec![48, 24]);
    assert_eq!(c.swaps, SwapRule::default());
    for bad in ["-1", "x", "1.5"] {
        let argv = ["legion", "serve", "--expert-swaps-device", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad}");
    }
}

#[test]
fn prefill_ring_flag() {
    // Default: the two buffers staging had before the flag.
    assert_eq!(residency(&[]).ring, 2);
    assert_eq!(residency(&[]).ring, ResidencyConfig::default().ring);
    assert_eq!(residency(&["--prefill-ring", "3"]).ring, 3);
    assert_eq!(residency(&["--prefill-ring", "8"]).ring, 8);
    // Out of range is refused, not clamped.
    for bad in ["0", "1", "9", "x"] {
        let argv = ["legion", "serve", "--prefill-ring", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad}");
    }
}

#[test]
fn prefill_staging_reuse_flag() {
    // Default: kept, as the library default.
    assert!(residency(&[]).stage_reuse);
    assert!(ResidencyConfig::default().stage_reuse);
    assert!(residency(&["--prefill-staging-reuse", "true"]).stage_reuse);
    assert!(!residency(&["--prefill-staging-reuse", "false"]).stage_reuse);
}

#[test]
fn prefill_keep_repack_flag() {
    // Default: prompts of 40K tokens and more (64K and 123K, not 32K), as the library default.
    assert_eq!(residency(&[]).lend_repack, 40960);
    assert_eq!(ResidencyConfig::default().lend_repack, 40960);
    assert_eq!(
        residency(&["--prefill-keep-repack", "100000"]).lend_repack,
        100000
    );
    assert_eq!(residency(&["--prefill-keep-repack", "0"]).lend_repack, 0);
}

#[test]
fn expert_cache_fill_prompt_room_flag() {
    // Default: on for V100s (an AC922), off elsewhere.
    let auto = residency(&[]);
    assert_eq!(auto.fill_prompt_room, None);
    assert!(auto.fills_prompt_room(true));
    assert!(!auto.fills_prompt_room(false));
    assert!(!ResidencyConfig::default().fills_prompt_room(false));
    let on = residency(&["--expert-cache-fill-prompt-room"]);
    assert_eq!(on.fill_prompt_room, Some(true));
    assert!(on.fills_prompt_room(false));
    let off = residency(&["--no-expert-cache-fill-prompt-room"]);
    assert_eq!(off.fill_prompt_room, Some(false));
    assert!(!off.fills_prompt_room(true));
    assert!(
        Cli::try_parse_from([
            "legion",
            "serve",
            "--expert-cache-fill-prompt-room",
            "--no-expert-cache-fill-prompt-room",
        ])
        .is_err()
    );
}

#[test]
fn prefill_stage_min_tokens_flag() {
    assert_eq!(
        residency(&[]).stage_min_tokens,
        ResidencyConfig::default().stage_min_tokens
    );
    assert_eq!(
        residency(&["--prefill-stage-min-tokens", "16"]).stage_min_tokens,
        16
    );
}

#[test]
fn prefill_peer_stage_flag() {
    assert!(!residency(&[]).stage_peer);
    assert!(residency(&["--prefill-peer-stage"]).stage_peer);
}

#[test]
fn expert_cache_reserve_flag() {
    // Default: the library's 1 GiB.
    assert_eq!(
        residency(&[]).cache_reserve,
        ResidencyConfig::default().cache_reserve
    );
    assert_eq!(residency(&[]).cache_reserve, 1 << 30);
    assert_eq!(
        residency(&["--expert-cache-reserve-mib", "512"]).cache_reserve,
        512 << 20
    );
}

#[test]
fn spec_flags() {
    assert!(!spec(&[]).unwrap().enabled());
    let c = spec(&["--spec", "ngram", "--spec-k", "6"]).unwrap();
    assert_eq!((c.kind, c.k), (SpecKind::Ngram, 6));
    assert!(c.enabled());
    // No --spec-k: 0, the model's default (the engine picks 2 for qwen4exp, 4 otherwise).
    assert_eq!(spec(&["--spec", "suffix"]).unwrap().k, 0);
    let c = spec(&["--spec", "eagle3", "--spec-draft", "e.gguf"]).unwrap();
    assert_eq!(c.draft, Some(PathBuf::from("e.gguf")));
    // MTP: the head in the model's GGUF, or in its own.
    assert!(spec(&["--spec", "mtp"]).is_ok());
    assert!(spec(&["--spec", "mtp", "--spec-draft", "mtp.gguf"]).is_ok());
    for bad in [
        &["--spec", "eagle3"][..],
        &["--spec", "ngram", "--spec-draft", "x.gguf"],
        &["--spec-draft", "x.gguf"],
        &["--spec", "ngram", "--spec-k", "0"],
    ] {
        assert!(spec(bad).is_err(), "{bad:?}");
    }
    assert!(
        Cli::try_parse_from(["legion", "serve", "--spec", "medusa"]).is_err(),
        "unknown drafter"
    );
}

#[test]
fn spec_mtp_ngram_flag() {
    assert_eq!(spec(&["--spec", "mtp"]).unwrap().mtp_ngram, 0);
    let c = spec(&["--spec", "mtp", "--spec-mtp-ngram", "8"]).unwrap();
    assert_eq!((c.kind, c.mtp_ngram, c.mtp_ngram_k), (SpecKind::Mtp, 8, 0));
    // The n-gram's own ceiling: up to 16, only with the n-gram.
    let c = spec(&[
        "--spec",
        "mtp",
        "--spec-k",
        "8",
        "--spec-mtp-ngram",
        "4",
        "--spec-mtp-ngram-k",
        "16",
    ])
    .unwrap();
    assert_eq!((c.k, c.ngram_k(), c.verify_k()), (8, 16, 16));
    let c = spec(&[
        "--spec",
        "mtp",
        "--spec-k",
        "8",
        "--spec-mtp-ngram",
        "4",
        "--spec-mtp-ngram-k",
        "2",
    ])
    .unwrap();
    assert_eq!((c.ngram_k(), c.verify_k()), (2, 8));
    assert!(spec(&["--spec", "mtp", "--spec-mtp-ngram-k", "16"]).is_err());
    let deep = ["legion", "serve", "--spec", "mtp", "--spec-mtp-ngram", "4"]
        .into_iter()
        .chain(["--spec-mtp-ngram-k", "17"]);
    assert!(Cli::try_parse_from(deep).is_err(), "at most 16");
    // MTP only: the other drafters, and no drafter, refuse it.
    for bad in [
        &["--spec-mtp-ngram", "8"][..],
        &["--spec", "ngram", "--spec-mtp-ngram", "8"],
    ] {
        assert!(spec(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn mtp_window_flag() {
    let parse = |argv: &[&str]| {
        let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Serve(args) => args.engine.mtp_window,
            _ => unreachable!("the parser returned a command other than serve"),
        }
    };
    assert_eq!(parse(&[]), None);
    assert_eq!(parse(&["--mtp-window", "32768"]), Some(32768));
    assert_eq!(parse(&["--mtp-window", "0"]), Some(0));
    // Past i32 (the kernels' window parameter): refused, not truncated.
    for bad in ["4294967297", "3000000000", "-1"] {
        let argv = ["legion", "serve", "--mtp-window", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad}");
    }
}

#[test]
fn spec_draft_head_type_flag() {
    let parse = |argv: &[&str]| {
        let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Serve(args) => args.engine.spec_draft_head_type,
            _ => unreachable!("the parser returned a command other than serve"),
        }
    };
    assert_eq!(parse(&[]), DraftHeadType::Auto);
    assert_eq!(
        parse(&["--spec-draft-head-type", "q8_0"]),
        DraftHeadType::Q8_0
    );
    assert_eq!(
        parse(&["--spec-draft-head-type", "model"]),
        DraftHeadType::Model
    );
    assert_eq!(
        parse(&["--spec-draft-head-type", "f16"]),
        DraftHeadType::F16
    );
    assert!(Cli::try_parse_from(["legion", "serve", "--spec-draft-head-type", "q4"]).is_err());
}

#[test]
fn spec_draft_weights_flag() {
    use legion::spec::DraftWeights;
    for kind in ["dflash", "dspark"] {
        let base = ["--spec", kind, "--spec-draft", "d"];
        assert_eq!(spec(&base).unwrap().draft_weights, DraftWeights::Model);
        let q8 = [&base[..], &["--spec-draft-weights", "q8_0"]].concat();
        assert_eq!(spec(&q8).unwrap().draft_weights, DraftWeights::Q8_0);
        let q4 = [&base[..], &["--spec-draft-weights", "q4"]].concat();
        assert_eq!(spec(&q4).unwrap().draft_weights, DraftWeights::Q4);
    }
    for other in [
        &[][..],
        &["--spec", "mtp"],
        &["--spec", "draft", "--spec-draft", "d"],
    ] {
        let argv = [other, &["--spec-draft-weights", "q8_0"]].concat();
        let e = spec(&argv).unwrap_err().to_string();
        assert!(e.contains("--spec-draft-weights needs"), "{e}");
    }
    assert!(Cli::try_parse_from(["legion", "serve", "--spec-draft-weights", "q4_k"]).is_err());
}

#[test]
fn spec_draft_lm_head_flag() {
    use legion::spec::DraftLmHead;
    let base = ["--spec", "dflash", "--spec-draft", "d"];
    assert_eq!(spec(&base).unwrap().draft_lm_head, DraftLmHead::Auto);
    let model = [&base[..], &["--spec-draft-lm-head", "model"]].concat();
    assert_eq!(spec(&model).unwrap().draft_lm_head, DraftLmHead::Model);
    // `model` and the default `auto` need no drafter.
    for v in ["model", "auto"] {
        assert!(spec(&["--spec-draft-lm-head", v]).is_ok(), "{v}");
    }
    let q4 = [&base[..], &["--spec-draft-lm-head", "q4"]].concat();
    assert_eq!(spec(&q4).unwrap().draft_lm_head, DraftLmHead::Q4);
    let mtp = ["--spec", "mtp", "--spec-draft-lm-head", "q4"];
    assert_eq!(spec(&mtp).unwrap().draft_lm_head, DraftLmHead::Q4);
    for other in [&[][..], &["--spec", "dspark", "--spec-draft", "d"]] {
        let argv = [other, &["--spec-draft-lm-head", "q4"]].concat();
        let e = spec(&argv).unwrap_err().to_string();
        assert!(e.contains("--spec-draft-lm-head needs"), "{e}");
    }
    assert!(Cli::try_parse_from(["legion", "serve", "--spec-draft-lm-head", "q8_0"]).is_err());
}

#[test]
fn spec_draft_batched_attn_flag() {
    let base = ["--spec", "dflash", "--spec-draft", "d"];
    assert!(!spec(&base).unwrap().draft_batched_attn);
    let on = [&base[..], &["--spec-draft-batched-attn"]].concat();
    assert!(spec(&on).unwrap().draft_batched_attn);
    for other in [&[][..], &["--spec", "dspark", "--spec-draft", "d"]] {
        let argv = [other, &["--spec-draft-batched-attn"]].concat();
        let e = spec(&argv).unwrap_err().to_string();
        assert!(e.contains("--spec-draft-batched-attn needs"), "{e}");
    }
}

fn serve_args(argv: &[&str]) -> Box<ServeArgs> {
    let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
    match Cli::try_parse_from(argv).unwrap().command {
        Command::Serve(args) => args,
        _ => unreachable!("the parser returned a command other than serve"),
    }
}

#[test]
fn keep_load_cache_flag() {
    assert!(!serve_args(&[]).engine.model.keep_load_cache);
    assert!(
        serve_args(&["--keep-load-cache"])
            .engine
            .model
            .keep_load_cache
    );
}

#[test]
fn numa_flag() {
    assert_eq!(serve_args(&[]).engine.numa, NumaMode::Auto);
    assert_eq!(serve_args(&["--numa", "off"]).engine.numa, NumaMode::Off);
    assert_eq!(
        serve_args(&["--numa", "8,0"]).engine.numa,
        NumaMode::Nodes(vec![0, 8])
    );
    let argv = ["legion", "serve", "--numa", "eight"];
    assert!(Cli::try_parse_from(argv).is_err());
    // The mock engine and the CPU bind nothing (no GPUs to follow).
    assert!(bind_numa(&serve_args(&[])).unwrap().is_none());
    let cpu = serve_args(&["--model", "m.gguf", "--numa", "0"]);
    assert!(bind_numa(&cpu).unwrap().is_none());
}

#[test]
fn glm_parallel_decode_runs_without_cuda_graphs() {
    let args = serve_args(&[
        "--model",
        "m.gguf",
        "--device",
        "cpu",
        "--glm-parallel-decode",
    ]);
    assert!(bind_numa(&args).unwrap().is_none());
    assert!(std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some());
    assert!(std::env::var_os("CUDA_MODULE_LOADING").is_some());
}

#[test]
fn glm_mtp_overlap_flags() {
    assert_eq!(
        serve_args(&[]).engine.model.glm_mtp_overlap_seqs(),
        1,
        "on for one sequence by default (#725)"
    );
    assert_eq!(
        legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu)
            .glm_mtp_overlap_seqs,
        1,
        "the engine's default is the flag's"
    );
    let off = serve_args(&["--glm-mtp-overlap-seqs", "0"]);
    assert_eq!(off.engine.model.glm_mtp_overlap_seqs(), 0);
    let args = serve_args(&["--glm-parallel-mtp", "--glm-mtp-overlap"]);
    assert!(args.engine.model.glm_parallel_mtp);
    assert_eq!(args.engine.model.glm_mtp_overlap_seqs(), usize::MAX);
}

#[test]
fn glm_pair_balance_flag() {
    assert!(
        !serve_args(&[]).engine.model.glm_pair_balance,
        "off by default"
    );
    assert!(
        !legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu)
            .glm_pair_balance,
        "the engine's default is the flag's"
    );
    let on = serve_args(&["--glm-parallel-decode", "--glm-pair-balance"]);
    assert!(on.engine.model.glm_pair_balance);
}

#[test]
fn glm_prefetch_stats_flag() {
    assert!(
        !serve_args(&[]).engine.model.glm_prefetch_stats,
        "off by default"
    );
    assert!(
        !legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu)
            .glm_prefetch_stats,
        "the engine's default is the flag's"
    );
    let on = serve_args(&["--glm-parallel-decode", "--glm-prefetch-stats"]);
    assert!(on.engine.model.glm_prefetch_stats);
}

#[test]
fn glm_cross_balance_flag() {
    assert!(
        !serve_args(&[]).engine.model.glm_cross_balance,
        "off by default"
    );
    assert!(
        !legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu)
            .glm_cross_balance,
        "the engine's default is the flag's"
    );
    let on = serve_args(&["--glm-parallel-decode", "--glm-cross-balance"]);
    assert!(on.engine.model.glm_cross_balance);
    assert!(!on.engine.model.glm_pair_balance);
}

#[test]
fn glm_prefetch_flag() {
    assert_eq!(
        serve_args(&[]).engine.model.glm_prefetch,
        0,
        "off by default"
    );
    assert_eq!(
        legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu).glm_prefetch,
        0,
        "the engine's default is the flag's"
    );
    let on = serve_args(&["--glm-parallel-decode", "--glm-prefetch", "4"]);
    assert_eq!(on.engine.model.glm_prefetch, 4);
}

#[test]
fn glm_pair_split_flag() {
    assert!(
        !serve_args(&[]).engine.model.glm_pair_split,
        "off by default"
    );
    assert!(
        !legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu)
            .glm_pair_split,
        "the engine's default is the flag's"
    );
    let on = serve_args(&["--glm-parallel-decode", "--glm-pair-split"]);
    assert!(on.engine.model.glm_pair_split);
}

#[test]
fn tp_replicas_flag() {
    use legion::engine::replicas::ReplicaCount;
    let one = serve_args(&[]);
    assert_eq!(one.engine.model.tp_replicas, ReplicaCount::Fixed(1));
    assert_eq!(
        one.engine.model.replicas().unwrap(),
        1,
        "one copy by default"
    );
    let args = serve_args(&["--tp", "2", "--tp-replicas", "auto"]);
    assert_eq!(args.engine.model.tp_replicas, ReplicaCount::Auto);
    // Without --tp, `auto` is one copy; a count needs --tp.
    assert_eq!(
        serve_args(&["--tp-replicas", "auto"])
            .engine
            .model
            .replicas()
            .unwrap(),
        1
    );
    let err = serve_args(&["--tp-replicas", "2"]).engine.model.replicas();
    assert!(err.unwrap_err().to_string().contains("--tp 2"));
    assert!(Cli::try_parse_from(["legion", "serve", "--tp-replicas", "0"]).is_err());
    assert!(Cli::try_parse_from(["legion", "serve", "--tp-replicas", "many"]).is_err());
}

#[test]
fn replica_node_needs_one_node() {
    assert_eq!(replica_node(&[Some(8), Some(8)]), Some(8));
    assert_eq!(replica_node(&[Some(0)]), Some(0));
    assert_eq!(replica_node(&[Some(0), Some(8)]), None);
    assert_eq!(replica_node(&[Some(0), None]), None);
    assert_eq!(replica_node(&[None, None]), None);
    assert_eq!(replica_node(&[]), None);
}

#[test]
fn pair_tp_flag() {
    assert!(!serve_args(&[]).engine.model.pair_tp, "off by default");
    let args = serve_args(&["--model", "m.gguf", "--device", "cpu", "--pair-tp"]);
    assert!(args.engine.model.pair_tp);
    assert!(bind_numa(&args).unwrap().is_none());
    assert!(std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some());
}

#[test]
fn deterministic_flag() {
    // Off: prefix caching and the adaptive depth as configured.
    let d = serve_args(&["--spec", "mtp"]);
    assert_eq!(d.engine.prefix_cache_states(), 8);
    assert!(!d.engine.spec().unwrap().fixed);
    // On: no prefix caching, the depth fixed.
    let on = serve_args(&["--spec", "mtp", "--deterministic"]);
    assert_eq!(on.engine.prefix_cache_states(), 0);
    assert!(on.engine.spec().unwrap().fixed);
    // Without drafts it still turns prefix caching off.
    assert_eq!(
        serve_args(&["--deterministic"])
            .engine
            .prefix_cache_states(),
        0
    );
    let check = |argv: &[&str]| {
        let a = serve_args(argv);
        let switches = a.engine.model.switches.switches();
        a.engine.check_deterministic(&switches)
    };
    // It would override a nonzero --prefix-cache-states: refused; 0 agrees with it.
    assert!(check(&["--deterministic", "--prefix-cache-states", "8"]).is_err());
    assert!(check(&["--deterministic", "--prefix-cache-states", "0"]).is_ok());
    assert!(check(&["--prefix-cache-states", "8"]).is_ok());
    // Expert matvec modes that make the output follow the expert cache are refused.
    for mode in ["1", "2"] {
        assert!(check(&["--deterministic", "--expert-mv-vram", mode]).is_err());
        assert!(check(&["--expert-mv-vram", mode]).is_ok());
    }
    // The settings it agrees with are fine.
    assert!(check(&["--deterministic", "--expert-mv-vram", "0"]).is_ok());
    let ok = serve_args(&["--deterministic", "--spec", "mtp", "--spec-fixed"]);
    assert!(ok.engine.spec().unwrap().fixed);
}

#[test]
fn deterministic_reaches_the_expert_cache() {
    // #1165: the residency settings carry --deterministic; off without it.
    assert!(!serve_args(&[]).engine.residency().deterministic);
    let on = serve_args(&["--deterministic"]).engine.residency();
    assert!(on.deterministic);
    // The model's own residency flags are kept beside it.
    let c = serve_args(&["--deterministic", "--expert-heat-pass-cap", "1"])
        .engine
        .residency();
    assert!(c.deterministic);
    assert_eq!(c.heat_pass_cap, 1.0);
}

#[test]
fn deterministic_gather_flag() {
    // Off without --deterministic: requests are admitted as they arrive.
    assert_eq!(serve_args(&[]).engine.deterministic_gather(), None);
    // On with it: waves gathered over 200 ms by default.
    assert_eq!(
        serve_args(&["--deterministic"])
            .engine
            .deterministic_gather(),
        Some(Duration::from_millis(200))
    );
    assert_eq!(
        serve_args(&["--deterministic", "--deterministic-gather-ms", "20"])
            .engine
            .deterministic_gather(),
        Some(Duration::from_millis(20))
    );
    // 0 turns the waves off.
    assert_eq!(
        serve_args(&["--deterministic", "--deterministic-gather-ms", "0"])
            .engine
            .deterministic_gather(),
        None
    );
    // It needs --deterministic.
    assert!(Cli::try_parse_from(["legion", "serve", "--deterministic-gather-ms", "20"]).is_err());
}

#[test]
fn admit_gather_flag() {
    // Off by default: requests are admitted as they arrive.
    assert_eq!(serve_args(&[]).engine.admit_gather(), None);
    assert_eq!(
        serve_args(&["--admit-gather-ms", "5"])
            .engine
            .admit_gather(),
        Some(Duration::from_millis(5))
    );
    assert_eq!(
        serve_args(&["--admit-gather-ms", "0"])
            .engine
            .admit_gather(),
        None
    );
    // --deterministic gathers its waves itself.
    assert!(
        Cli::try_parse_from([
            "legion",
            "serve",
            "--deterministic",
            "--admit-gather-ms",
            "5"
        ])
        .is_err()
    );
}

#[test]
fn registry_flags() {
    // Off by default.
    let r = serve_args(&[]).server.config().unwrap().registry;
    assert!(r.proxies.is_empty());
    assert_eq!(r.host, None);
    assert_eq!(r.interval, Duration::from_secs(300));
    let a = serve_args(&[
        "--registry-api",
        "http://192.0.2.1:8080",
        "--registry-api",
        "https://proxy.example/register",
        "--registry-host",
        "192.0.2.9",
        "--registry-interval",
        "60",
        "--api-key",
        "k",
    ]);
    let c = a.server.config().unwrap();
    let urls: Vec<&str> = c.registry.proxies.iter().map(|u| u.as_str()).collect();
    assert_eq!(
        urls,
        [
            "http://192.0.2.1:8080/register",
            "https://proxy.example/register"
        ]
    );
    assert_eq!(c.registry.host, Some("192.0.2.9".parse().unwrap()));
    assert_eq!(c.registry.interval, Duration::from_secs(60));
    for bad in [
        &["--registry-api", "192.0.2.1:8080"][..],
        &["--registry-host", "192.0.2.9"],
        &[
            "--registry-api",
            "http://192.0.2.1",
            "--registry-host",
            "proxy.example",
        ],
        &[
            "--registry-api",
            "http://192.0.2.1",
            "--registry-interval",
            "0",
        ],
    ] {
        let argv = ["legion", "serve"].into_iter().chain(bad.iter().copied());
        assert!(Cli::try_parse_from(argv).is_err(), "{bad:?}");
    }
}

#[test]
fn stage_graphs_flag() {
    // On by default (#849), and `--stage-graphs` only states it.
    for flags in [&[][..], &["--stage-graphs"]] {
        let argv: Vec<&str> = ["--model", "m.gguf", "--device", "cpu"]
            .into_iter()
            .chain(flags.iter().copied())
            .collect();
        let args = serve_args(&argv);
        assert!(args.engine.model.stage_graphs(), "{flags:?}");
        assert_eq!(
            args.engine.model.served_switches().stage_graphs,
            Some(true),
            "{flags:?}"
        );
        // ggml's own CUDA graphs go off for them.
        assert!(bind_numa(&args).unwrap().is_none());
        assert!(std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_some());
    }
    assert!(serve_args(&[]).engine.model.stage_graphs());
    // `--no-stage-graphs` turns them off: the switch keeps the built-in default (off), and the
    // other switches are untouched.
    let off = serve_args(&["--model", "m.gguf", "--no-stage-graphs", "--hc-fuse"]);
    assert!(!off.engine.model.stage_graphs());
    let s = off.engine.model.served_switches();
    assert_eq!(s.stage_graphs, None);
    assert_eq!(s.hc_fuse, Some(true));
    let argv = ["legion", "serve", "--stage-graphs", "--no-stage-graphs"];
    assert!(Cli::try_parse_from(argv).is_err(), "the two conflict");
    // Not with tensor parallelism or the spinning exchanges, given or not.
    for extra in [
        &["--pair-tp"][..],
        &["--glm-parallel-decode"],
        &["--tp", "2"],
    ] {
        for given in [&[][..], &["--stage-graphs"]] {
            let argv: Vec<&str> = ["--model", "m.gguf"]
                .into_iter()
                .chain(extra.iter().copied())
                .chain(given.iter().copied())
                .collect();
            let a = serve_args(&argv);
            assert!(!a.engine.model.stage_graphs(), "{argv:?}");
            assert_eq!(a.engine.model.served_switches().stage_graphs, None);
        }
    }
}

#[test]
fn prefix_cache_interval_flag() {
    assert_eq!(serve_args(&[]).engine.prefix_cache_interval, 0);
    let a = serve_args(&["--prefix-cache-interval", "2048"]);
    assert_eq!(a.engine.prefix_cache_interval, 2048);
}

#[test]
fn prefix_debug_flags_default_off() {
    let a = serve_args(&[]);
    assert!(!a.engine.prefix_capture_cut && !a.engine.prefix_state_digests);
    let a = serve_args(&["--prefix-capture-cut", "--prefix-state-digests"]);
    assert!(a.engine.prefix_capture_cut && a.engine.prefix_state_digests);
}

#[test]
fn uma_free_gib_flag() {
    let parse = |argv: &[&str]| {
        let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
        Cli::try_parse_from(argv).map(|c| match c.command {
            Command::Serve(args) => args.engine.uma_free_gib,
            _ => unreachable!("the parser returned a command other than serve"),
        })
    };
    // The default is the engine's 16 GiB.
    let d = parse(&[]).unwrap();
    assert_eq!(uma_free_bytes(d), legion::engine::lm::UMA_HEADROOM);
    assert_eq!(
        uma_free_bytes(parse(&["--uma-free-gib", "8"]).unwrap()),
        8 << 30
    );
    assert_eq!(
        uma_free_bytes(parse(&["--uma-free-gib", "0.5"]).unwrap()),
        512 << 20
    );
    for bad in ["-1", "nan", "inf"] {
        assert!(parse(&["--uma-free-gib", bad]).is_err(), "{bad}");
    }
}

fn ngram(argv: &[&str]) -> anyhow::Result<lookup_table::TableOptions> {
    let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
    match Cli::try_parse_from(argv).unwrap().command {
        Command::Serve(args) => args.engine.model.ngram_options(),
        _ => unreachable!("the parser returned a command other than serve"),
    }
}

/// `--ngram-ram` (#1209): auto by default, sizes in binary units, `--ngram-ram-gib N` as
/// `--ngram-ram NG`, and the pairs with `--ngram-io` that contradict themselves refused.
#[test]
fn ngram_ram_flag() {
    let gib = 1u64 << 30;
    let o = ngram(&[]).unwrap();
    assert_eq!(
        (o.io, o.ram, o.ram_cache_bytes),
        (TableIo::Auto, RamBudget::Auto, None)
    );
    assert_eq!(
        ngram(&["--ngram-ram", "8G"]).unwrap().ram,
        RamBudget::Bytes(8 * gib)
    );
    assert_eq!(
        ngram(&["--ngram-ram", "512M"]).unwrap().ram,
        RamBudget::Bytes(512 << 20)
    );
    assert_eq!(
        ngram(&["--ngram-ram", "1.5"]).unwrap().ram,
        RamBudget::Bytes(3 * gib / 2)
    );
    assert_eq!(
        ngram(&["--ngram-ram", "30%"]).unwrap().ram,
        RamBudget::Percent(30.0)
    );
    assert_eq!(ngram(&["--ngram-ram", "all"]).unwrap().ram, RamBudget::All);
    // The old spelling, in GiB.
    let old = ngram(&["--ngram-ram-gib", "2"]).unwrap();
    assert_eq!(
        (old.ram, old.ram_cache_bytes),
        (RamBudget::Bytes(2 * gib), None)
    );
    assert_eq!(
        ngram(&["--ngram-ram-gib", "0.5"]).unwrap().ram,
        RamBudget::Bytes(gib / 2)
    );
    // A size with direct, all with ram.
    let d = ngram(&["--ngram-io", "direct", "--ngram-ram", "2G"]).unwrap();
    assert_eq!((d.io, d.ram), (TableIo::Direct, RamBudget::Bytes(2 * gib)));
    let r = ngram(&["--ngram-io", "ram", "--ngram-ram", "all"]).unwrap();
    assert_eq!((r.io, r.ram), (TableIo::Ram, RamBudget::All));
    // Contradictions.
    for bad in [
        &["--ngram-io", "mmap", "--ngram-ram", "2G"][..],
        &["--ngram-io", "mmap", "--ngram-ram", "all"],
        &["--ngram-io", "direct", "--ngram-ram", "all"],
        &["--ngram-ram", "2G", "--ngram-ram-gib", "2"],
        &["--ngram-ram-gib", "0"],
    ] {
        assert!(ngram(bad).is_err(), "{bad:?}");
    }
    // Not sizes.
    for bad in ["0", "0G", "-1G", "0%", "101%", "8X", "lots"] {
        let argv = ["legion", "serve", "--ngram-ram", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad}");
    }
}

#[test]
fn tu_attn_ffma_flag() {
    let model = |argv: &[&str]| {
        let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Serve(args) => args.engine.model.tu_attn_ffma,
            _ => unreachable!("the parser returned a command other than serve"),
        }
    };
    assert!(!model(&[]));
    assert!(model(&["--tu-attn-ffma"]));
    assert!(
        !legion::engine::lm::LmConfig::new("m", legion::engine::lm::DeviceChoice::Cpu).tu_attn_ffma
    );
}
