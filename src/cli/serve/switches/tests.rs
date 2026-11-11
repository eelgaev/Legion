//! The kernel and load switches' flags.

use clap::{CommandFactory as _, Parser as _};
use legion::engine::switches::Switches;
use legion::model::core::expert_mv::Dot;
use legion::model::core::expert_mv::peer::Share;
use legion::model::moe_ffn::PairOut;
use legion::model::qwen4::hc::L2Prefetch;

use crate::cli::{Cli, Command};

fn args(argv: &[&str]) -> super::super::ServeArgs {
    let argv = ["legion", "serve"].into_iter().chain(argv.iter().copied());
    match Cli::try_parse_from(argv).unwrap().command {
        Command::Serve(args) => *args,
        _ => unreachable!("the parser returned a command other than serve"),
    }
}

/// The switches for these flags.
fn switches(argv: &[&str]) -> Switches {
    args(argv).engine.model.switches.switches()
}

#[test]
fn nothing_given_sets_nothing() {
    // No flag: every switch keeps the built-in default.
    assert_eq!(switches(&[]), Switches::default());
}

#[test]
fn flags_set_their_switches() {
    let s = switches(&[
        "--gated-norm",
        "--moe-combine",
        "--stable-fusions",
        "--delta-prep-ggml",
        "--gdn-scan-ggml",
        "--gdn-mma",
        "--gdn-ggml",
        "--gdn-decode-ggml",
        "--gdn-replay-smem",
        "--no-gdn-pad-skip",
        "--gdn-commit-runs",
        "--gdn-commit-in-verify",
        "--gdn-state-defer",
        "--gdn-state-defer-rows",
        "16",
        "--gdn-state-defer-min-seqs",
        "8",
        "--norm-rope",
        "--kv-append",
        "--head-rope-fused",
        "--no-joined-matmuls",
        "--joined-prompt-min-tokens",
        "2048",
        "--no-joined-prompt-z",
        "--no-joined-prompt-qkv",
        "--dense-mv-ggml",
        "--draft-head-q8-mv",
        "--no-draft-head-mv",
        "--no-draft-head-pick",
        "--no-q8-group-row-mv",
        "--no-draft-head-q4k-mv",
        "--q8-out-row-mv",
        "--no-draft-head-q4k-pick",
        "--no-draft-head-q4k-pick-cols",
        "--no-head-mm",
        "--q8-group",
        "--dense-mv-small",
        "--no-cluster-select",
        "--dense-mv-f32",
        "--no-dense-mv-ab",
        "--dense-mv-split",
        "--dense-mv-wide-chain",
        "--no-router-mv",
        "--no-router-join",
        "--no-router-bf16",
        "--no-ple-early-reads",
        "--nvfp4-mmvq-glu",
        "--fp8-small-batch-gemv",
        "--fp8-sliced-k",
        "--hopper-gemm",
        "--hopper-attn",
        "--hopper-pdl",
        "--nvfp4-wide-mv",
        "--no-nvfp4-expert-pdl",
        "--no-nvfp4-expert-tile-prefetch",
        "--nvfp4-expert-mv-max",
        "8",
        "--nvfp4-expert-mv-tile",
        "2",
        "--nvfp4-expert-mv-split",
        "expert-down",
        "--nvfp4-expert-rows-bf16",
        "--no-moe-pad-remap",
        "--moe-pad-remap-tiered",
        "--moe-pad-remap-probe",
        "kernels",
        "--moe-pad-skip-tiered",
        "--moe-pad-skip-layout-only",
        "--no-moe-pad-skip",
        "--no-tp-moe-ep-skip",
        "--paged-verify-split",
        "--paged-verify-split-wmma",
        "--shexp-gate",
        "--decode-fuse",
        "--node-fuse-ggml",
        "--moe-pair-out",
        "bf16",
        "--no-moe-raw-staging",
        "--moe-down-sum",
        "--expert-mv-vram",
        "2",
        "--expert-mv-prefetch",
        "--expert-mv-link-ctas",
        "32",
        "--expert-mv-ptx",
        "--expert-mv-ptx-side",
        "--expert-mv-device",
        "--expert-mv-max-tokens",
        "64",
        "--expert-mv-split-glu",
        "--expert-mv-dot",
        "sub16",
        "--expert-mv-pipe",
        "4",
        "--expert-mv-q4k-units",
        "--no-expert-mv-q4k-hold",
        "--no-expert-mv-balance",
        "--no-expert-mv-half-tiles",
        "--expert-miss-overlap",
        "--expert-peer-fetch",
        "--expert-peer-share",
        "3/4",
        "--expert-peer-ctas",
        "48",
        "--qsa-scores-tc",
        "--qsa-always-score",
        "--qsa-attn-chain",
        "--hc-mix-ggml",
        "--hc-mix-bf16",
        "--hc-l2-prefetch",
        "both",
        "--no-hc-norm-regs",
        "--no-hc-down-split",
        "--no-hc-down-hoist",
        "--no-hc-up-stage",
        "--hc-post-norm-ggml",
        "--hc-moe-fold",
        "--hc-fuse",
        "--hc-xn-f32",
        "--hc-quant-ggml",
        "--expert-mmq",
        "--nvfp4-once",
        "--nvfp4-mvq",
        "--q8-mmq",
        "--q8-mmq-stream",
        "--q8-mmq-fixup",
        "launch",
        "--tu-mmq-ggml",
        "--hc-up-mix-ggml",
        "--no-hc-up-mix-bf16",
        "--hc-xn-written",
        "--hc-streams-bf16",
        "--hc-mix-f16",
        "--hc-mix-rebuild-bf16",
        "--hc-chain-ggml",
        "--hc-pre-ggml",
        "--gate-quant-ggml",
        "--gate-norm-ggml",
        "--shared-quant-ggml",
        "--shared-quant-small-ggml",
        "--shexp-sum-ggml",
        "--hc-norm-fma",
        "--ple-ops",
        "--no-after-overlap",
        "--no-input-pack",
        "--no-stage-launch-ahead",
        "--no-commit-beside-draft",
        "--commit-last-late",
        "--commit-capture",
        "--mtp-ahead",
        "--mtp-pipe-rows",
        "2",
        "--mtp-moe-fused-rows",
        "9",
        "--prefix-states-vram",
        "--load-threads",
        "6",
        "--load-parallel-device",
        "--no-mmap",
        "--load-page-cache",
        "keep",
        "--glm-expert-mv",
        "--glm-verify-mv-max-tokens",
        "8",
        "--glm-decode-mv",
        "--glm-linear-plain",
        "--stage-graphs",
        "--no-stage-graph-split",
        "--no-stage-graph-early-capture",
        "--no-stage-graph-capture-on-reuse",
        "--no-stage-graph-words-skip",
        "--stage-graph-hold-kept",
        "--stage-graph-stats",
        "--no-prompt-build-ahead",
        "--no-prompt-buffer-reserve",
        "--prompt-step-stats",
        "--audit-graph-inputs",
        "--stage-graph-cache",
        "0",
        "--stage-graph-spare-execs",
        "0",
        "--stage-graph-exec-steal-ms",
        "0",
        "--mtp-kept-graphs",
        "32",
        "--verify-kept-graphs",
        "12",
        "--verify-kept-mib",
        "512",
        "--no-mtp-take-in-sort",
        "--qsa-pool-pad-steps",
        "64",
        "--kv-vmm",
        "--kv-vmm-keep",
        "32768",
    ]);
    assert_eq!(
        s,
        Switches {
            gated_norm: Some(true),
            moe_combine: Some(true),
            stable_fusions: Some(true),
            delta_prep: Some(false),
            gdn_scan: Some(false),
            gdn_mma: Some(true),
            gdn: Some(false),
            gdn_decode: Some(false),
            gdn_replay_staged: Some(true),
            gdn_pad_skip: Some(false),
            gdn_commit_slots: Some(false),
            gdn_commit_in_verify: Some(true),
            gdn_state_defer: Some(true),
            gdn_state_defer_rows: Some(16),
            gdn_state_defer_min_seqs: Some(8),
            norm_rope: Some(true),
            kv_append: Some(true),
            head_rope_fused: Some(true),
            joined_matmuls: Some(false),
            joined_prompt_min_tokens: Some(2048),
            joined_prompt_z: Some(false),
            joined_prompt_qkv: Some(false),
            dense_mv: Some(false),
            q8_mv: Some(true),
            q8_head: Some(false),
            q8_head_pick: Some(false),
            q8_row_mv: Some(false),
            q4k_head: Some(false),
            q8_out_mv: Some(true),
            q4k_head_pick: Some(false),
            q4k_head_pick_cols: Some(false),
            head_mm: Some(false),
            q8_group: Some(true),
            q8_group_parts: None,
            dense_mv_small: Some(true),
            cluster_select: Some(false),
            dense_mv_f32: Some(true),
            dense_mv_ab: Some(false),
            dense_mv_split: Some(true),
            dense_mv_wide_chain: Some(true),
            router_mv: Some(false),
            router_join: Some(false),
            router_bf16: Some(false),
            ple_early_reads: Some(false),
            nvfp4_mmvq_glu: Some(true),
            fp8_small_batch_gemv: Some(true),
            fp8_sliced_k: Some(true),
            hopper_gemm: Some(true),
            hopper_attn: Some(true),
            hopper_pdl: Some(true),
            nvfp4_wide_mv: Some(true),
            nvfp4_expert_pdl: Some(false),
            nvfp4_expert_tile_prefetch: Some(false),
            nvfp4_expert_mv_max: Some(8),
            nvfp4_expert_mv_tile: Some(2),
            nvfp4_expert_mv_split: Some(legion::model::nvfp4_mm::MvSplit::ExpertDown),
            nvfp4_expert_rows_bf16: Some(true),
            moe_pad_remap: Some(false),
            moe_pad_remap_tiered: Some(true),
            moe_pad_remap_probe: Some(legion::model::core::pad_rows::Probe::Kernels),
            moe_pad_skip_tiered: Some(true),
            moe_pad_skip_layout_only: Some(true),
            moe_pad_skip: Some(false),
            tp_moe_ep_skip: Some(false),
            paged_verify_split: Some(true),
            paged_verify_split_wmma: Some(true),
            shexp_gate: Some(true),
            decode_fuse: Some(true),
            node_fuse: Some(false),
            moe_pair_out: Some(PairOut::Bf16),
            moe_raw_staging: Some(false),
            moe_down_sum: Some(true),
            expert_mv_vram: Some(2),
            expert_mv_prefetch: Some(true),
            expert_mv_link_ctas: Some(32),
            expert_mv_ptx: Some(true),
            expert_mv_ptx_side: Some(true),
            expert_mv_device: Some(true),
            expert_mv_max_tokens: Some(64),
            expert_mv_glu: Some(false),
            expert_mv_dot: Some(Dot::Sub16),
            expert_mv_pipe: Some(4),
            expert_mv_q4k_units: Some(true),
            expert_mv_q4k_hold: Some(false),
            expert_mv_balance: Some(false),
            expert_mv_half_tiles: Some(false),
            expert_miss_overlap: Some(true),
            expert_peer_fetch: Some(true),
            expert_peer_share: Some(Share { take: 3, every: 4 }),
            expert_peer_ctas: Some(48),
            qsa_scores_tc: Some(true),
            qsa_attn_chain: Some(true),
            qsa_pick_all: Some(false),
            hc_mix: Some(false),
            hc_mix_bf16: Some(true),
            hc_l2_prefetch: Some(L2Prefetch::Both),
            hc_norm_regs: Some(false),
            hc_down_split: Some(false),
            hc_down_hoist: Some(false),
            hc_up_stage: Some(false),
            hc_post_norm: Some(false),
            hc_moe_fold: Some(true),
            hc_fuse: Some(true),
            hc_xn16: Some(false),
            hc_quant: Some(false),
            mmq_ids: Some(true),
            nvfp4_once: Some(true),
            nvfp4_mvq: Some(true),
            q8_mmq: Some(true),
            q8_mmq_stream: Some(true),
            q8_mmq_fixup: Some(legion::model::q8_mmq::Fixup::Launch),
            tu_mmq: Some(false),
            hc_up_mix: Some(false),
            hc_up_mix_bf16: Some(false),
            hc_xn_rebuild: Some(false),
            hc_streams_bf16: Some(true),
            hc_mix16: Some(true),
            hc_mix_rebuild_bf16: Some(true),
            hc_chain: Some(false),
            hc_pre: Some(false),
            gate_q8: Some(false),
            gate_norm: Some(false),
            shared_quant: Some(false),
            shared_quant_small: Some(false),
            shexp_sum: Some(false),
            hc_norm_fma: Some(true),
            ple: Some(true),
            after_overlap: Some(false),
            input_pack: Some(false),
            launch_ahead: Some(false),
            commit_beside_draft: Some(false),
            commit_last_late: Some(true),
            commit_capture: Some(true),
            mtp_ahead: Some(true),
            mtp_pipe_rows: Some(2),
            mtp_moe_fused_rows: Some(9),
            saved_states_vram: Some(true),
            load_threads: Some(6),
            load_parallel_device: Some(true),
            mmap: Some(false),
            load_page_cache: Some(legion::load_io::PageCache::Keep),
            glm_expert_mv: Some(true),
            glm_verify_mv_tokens: Some(8),
            glm_decode_mv: Some(true),
            glm_gb10_linear: Some(false),
            // conflicts with --glm-linear-plain (below)
            glm_mla_tc: None,
            glm_expert_q8: None,
            stage_graphs: Some(true),
            stage_graph_split: Some(false),
            stage_graph_early: Some(false),
            stage_graph_reuse_capture: Some(false),
            stage_graph_words_skip: Some(false),
            stage_graph_hold_kept: Some(true),
            stage_graph_stats: Some(true),
            prompt_build_ahead: Some(false),
            prompt_buffer_reserve: Some(false),
            prompt_step_stats: Some(true),
            audit_graph_inputs: Some(true),
            stage_graph_cache: Some(0),
            stage_graph_spare_execs: Some(0),
            stage_graph_exec_steal_ms: Some(0),
            mtp_kept_graphs: Some(32),
            verify_kept_graphs: Some(12),
            verify_kept_mib: Some(512),
            mtp_take_in_sort: Some(false),
            qsa_pool_pad_steps: Some(64),
            kv_vmm: Some(true),
            kv_vmm_keep: Some(32768),
        }
    );
}

#[test]
fn expert_peer_fetch_is_automatic_unless_forced() {
    // No flag: the load's automatic choice (#522).
    assert_eq!(switches(&[]).expert_peer_fetch, None);
    assert_eq!(
        switches(&["--expert-peer-fetch"]).expert_peer_fetch,
        Some(true)
    );
    assert_eq!(
        switches(&["--no-expert-peer-fetch"]).expert_peer_fetch,
        Some(false)
    );
    let both = [
        "legion",
        "serve",
        "--expert-peer-fetch",
        "--no-expert-peer-fetch",
    ];
    assert!(Cli::try_parse_from(both).is_err());
}

#[test]
fn expert_peer_share_takes_take_of_every() {
    assert_eq!(switches(&[]).expert_peer_share, None);
    assert_eq!(
        switches(&["--expert-peer-share", "2/3"]).expert_peer_share,
        Some(Share { take: 2, every: 3 })
    );
    for bad in ["0/2", "3/2", "1/65", "2", "a/3", "2/"] {
        let argv = ["legion", "serve", "--expert-peer-share", bad];
        assert!(Cli::try_parse_from(argv).is_err(), "{bad} parsed");
    }
    let argv = ["legion", "serve", "--expert-peer-ctas", "0"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn dense_mv_small_follows_stage_graphs_unless_forced() {
    // No flag: on where the stage graphs are (#836).
    assert_eq!(switches(&[]).dense_mv_small, None);
    assert_eq!(switches(&["--dense-mv-small"]).dense_mv_small, Some(true));
    assert_eq!(
        switches(&["--no-dense-mv-small"]).dense_mv_small,
        Some(false)
    );
    let both = ["legion", "serve", "--dense-mv-small", "--no-dense-mv-small"];
    assert!(Cli::try_parse_from(both).is_err());
}

#[test]
fn dense_mv_wide_chain_bitwise_unless_forced() {
    // No flag: the bitwise shapes only (#836).
    assert_eq!(switches(&[]).dense_mv_wide_chain, None);
    assert_eq!(
        switches(&["--dense-mv-wide-chain"]).dense_mv_wide_chain,
        Some(true)
    );
    assert_eq!(
        switches(&["--no-dense-mv-wide-chain"]).dense_mv_wide_chain,
        Some(false)
    );
    let both = [
        "legion",
        "serve",
        "--dense-mv-wide-chain",
        "--no-dense-mv-wide-chain",
    ];
    assert!(Cli::try_parse_from(both).is_err());
}

#[test]
fn glm_gb10_kernels_forced_either_way() {
    assert_eq!(switches(&[]).glm_gb10_linear, None);
    assert_eq!(switches(&["--glm-gb10-linear"]).glm_gb10_linear, Some(true));
    assert_eq!(
        switches(&["--glm-linear-plain"]).glm_gb10_linear,
        Some(false)
    );
    let argv = ["legion", "serve", "--glm-gb10-linear", "--glm-linear-plain"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn hc_moe_fold_forced_either_way() {
    assert_eq!(switches(&[]).hc_moe_fold, None);
    assert_eq!(switches(&["--hc-moe-fold"]).hc_moe_fold, Some(true));
    assert_eq!(switches(&["--no-hc-moe-fold"]).hc_moe_fold, Some(false));
    let argv = ["legion", "serve", "--hc-moe-fold", "--no-hc-moe-fold"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn expert_mv_ptx_side_needs_the_ptx_kernels() {
    assert_eq!(switches(&[]).expert_mv_ptx_side, None);
    let s = switches(&["--expert-mv-ptx", "--expert-mv-ptx-side"]);
    assert_eq!(
        (s.expert_mv_ptx, s.expert_mv_ptx_side),
        (Some(true), Some(true))
    );
    let argv = ["legion", "serve", "--expert-mv-ptx-side"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn glm_mla_tc_is_opt_in() {
    assert_eq!(switches(&[]).glm_mla_tc, None);
    assert_eq!(switches(&["--glm-mla-tc"]).glm_mla_tc, Some(true));
    // it runs with the GB10 kernels only
    let argv = ["legion", "serve", "--glm-mla-tc", "--glm-linear-plain"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn nvfp4_mvq_either_way() {
    // No flag: the default (on, on 12.x); the old opt-in flag is still accepted.
    assert_eq!(switches(&[]).nvfp4_mvq, None);
    assert_eq!(switches(&["--nvfp4-mvq"]).nvfp4_mvq, Some(true));
    assert_eq!(switches(&["--no-nvfp4-mvq"]).nvfp4_mvq, Some(false));
    let argv = ["legion", "serve", "--nvfp4-mvq", "--no-nvfp4-mvq"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn q8_group_either_way() {
    // No flag: the op's default (q8_group::Mode::Default).
    assert_eq!(switches(&[]).q8_group, None);
    assert_eq!(switches(&["--q8-group"]).q8_group, Some(true));
    assert_eq!(switches(&["--no-q8-group"]).q8_group, Some(false));
    assert_eq!(switches(&["--q8-group-parts", "4"]).q8_group_parts, Some(4));
    let argv = ["legion", "serve", "--q8-group", "--no-q8-group"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn gdn_commit_in_verify_either_way() {
    // No flag: the default (on since #836 part 38); the old opt-in flag is still accepted.
    assert_eq!(switches(&[]).gdn_commit_in_verify, None);
    let on = switches(&["--gdn-commit-in-verify"]);
    assert_eq!(on.gdn_commit_in_verify, Some(true));
    let off = switches(&["--no-gdn-commit-in-verify"]);
    assert_eq!(off.gdn_commit_in_verify, Some(false));
    let argv = [
        "legion",
        "serve",
        "--gdn-commit-in-verify",
        "--no-gdn-commit-in-verify",
    ];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn gdn_state_defer_either_way() {
    // No flag: the default (off, #836).
    let none = switches(&[]);
    assert_eq!(none.gdn_state_defer, None);
    assert_eq!(none.gdn_state_defer_rows, None);
    assert_eq!(switches(&["--gdn-state-defer"]).gdn_state_defer, Some(true));
    assert_eq!(
        switches(&["--no-gdn-state-defer"]).gdn_state_defer,
        Some(false)
    );
    let rows = switches(&["--gdn-state-defer-rows", "0"]);
    assert_eq!(rows.gdn_state_defer_rows, Some(0));
    assert_eq!(none.gdn_state_defer_min_seqs, None);
    let wide = switches(&["--gdn-state-defer-min-seqs", "8"]);
    assert_eq!(wide.gdn_state_defer_min_seqs, Some(8));
    let argv = [
        "legion",
        "serve",
        "--gdn-state-defer",
        "--no-gdn-state-defer",
    ];
    assert!(Cli::try_parse_from(argv).is_err());
    let argv = ["legion", "serve", "--gdn-state-defer-rows", "33"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn glm_expert_q8_is_opt_in() {
    assert_eq!(switches(&[]).glm_expert_q8, None);
    assert_eq!(switches(&["--glm-expert-q8"]).glm_expert_q8, Some(true));
    // with --glm-mla-tc: the fast prefill pair
    let both = switches(&["--glm-expert-q8", "--glm-mla-tc"]);
    assert_eq!(
        (both.glm_expert_q8, both.glm_mla_tc),
        (Some(true), Some(true))
    );
    // it runs with the GB10 kernels only
    let argv = ["legion", "serve", "--glm-expert-q8", "--glm-linear-plain"];
    assert!(Cli::try_parse_from(argv).is_err());
}

#[test]
fn out_of_range_values_are_refused() {
    for bad in [
        &["--expert-mv-vram", "3"][..],
        &["--expert-mv-max-tokens", "0"],
        &["--expert-mv-max-tokens", "65"],
        &["--glm-verify-mv-max-tokens", "0"],
        &["--glm-verify-mv-max-tokens", "65"],
        &["--qsa-pool-pad-steps", "1025"],
        &["--stage-graph-cache", "4097"],
        &["--stage-graph-spare-execs", "257"],
        &["--stage-graph-exec-steal-ms", "600001"],
        &["--mtp-kept-graphs", "0"],
        &["--verify-kept-graphs", "257"],
        &["--verify-kept-mib", "1048577"],
        &["--load-threads", "0"],
        &["--load-threads", "9"],
        &["--moe-pair-out", "f8"],
        &["--expert-mv-dot", "fast"],
        &["--hc-l2-prefetch", "all"],
        &["--expert-mv-pipe", "1025"],
        &["--spec-draft-window", "0"],
        &["--expert-profile-seed", "-1"],
        &["--nvfp4-expert-mv-max", "256"],
        &["--nvfp4-expert-mv-tile", "3"],
        &["--nvfp4-expert-mv-tile", "8"],
        &["--nvfp4-expert-mv-split", "gemm"],
    ] {
        let argv = ["legion", "serve"].into_iter().chain(bad.iter().copied());
        assert!(Cli::try_parse_from(argv).is_err(), "{bad:?}");
    }
}

#[test]
fn draft_window_and_profile_seed() {
    // No flag: the defaults (0: the drafter's own window).
    let a = args(&["--spec", "dflash", "--spec-draft", "d.gguf"]);
    assert_eq!(a.engine.spec().unwrap().draft_window, 0);
    assert_eq!(
        a.engine.model.residency().profile_seed,
        legion::residency::ResidencyConfig::default().profile_seed
    );
    // The flags reach the configs.
    let a = args(&[
        "--spec",
        "dflash",
        "--spec-draft",
        "d.gguf",
        "--spec-draft-window",
        "1024",
        "--expert-profile-seed",
        "8",
    ]);
    assert_eq!(a.engine.spec().unwrap().draft_window, 1024);
    assert_eq!(a.engine.model.residency().profile_seed, 8.0);
}

#[test]
fn help_lists_the_switches_under_their_heading() {
    let help = Cli::command()
        .find_subcommand_mut("serve")
        .unwrap()
        .render_long_help()
        .to_string();
    let heading = help
        .find("Kernel and load switches:")
        .expect("no switch heading");
    for flag in [
        "--stage-graphs",
        "--no-stage-graphs",
        "--no-stage-graph-split",
        "--no-stage-graph-early-capture",
        "--no-stage-graph-capture-on-reuse",
        "--no-stage-graph-words-skip",
        "--no-prompt-build-ahead",
        "--no-prompt-buffer-reserve",
        "--prompt-step-stats",
        "--stage-graph-cache",
        "--stage-graph-spare-execs",
        "--stage-graph-exec-steal-ms",
        "--mtp-kept-graphs",
        "--verify-kept-graphs",
        "--verify-kept-mib",
        "--no-mtp-take-in-sort",
        "--gated-norm",
        "--moe-combine",
        "--delta-prep-ggml",
        "--gdn-ggml",
        "--gdn-decode-ggml",
        "--gdn-replay-smem",
        "--no-gdn-pad-skip",
        "--gdn-commit-runs",
        "--gdn-commit-in-verify",
        "--no-gdn-commit-in-verify",
        "--gdn-state-defer",
        "--no-gdn-state-defer",
        "--gdn-state-defer-rows",
        "--gdn-state-defer-min-seqs",
        "--norm-rope",
        "--kv-append",
        "--head-rope-fused",
        "--no-joined-matmuls",
        "--joined-prompt-min-tokens",
        "--no-joined-prompt-z",
        "--no-joined-prompt-qkv",
        "--dense-mv-small",
        "--no-dense-mv-small",
        "--no-dense-mv-ab",
        "--dense-mv-split",
        "--dense-mv-wide-chain",
        "--no-dense-mv-wide-chain",
        "--no-router-mv",
        "--no-router-join",
        "--no-router-bf16",
        "--no-ple-early-reads",
        "--nvfp4-mmvq-glu",
        "--q8-group",
        "--no-q8-group",
        "--shexp-gate",
        "--decode-fuse",
        "--node-fuse-ggml",
        "--moe-pair-out",
        "--no-moe-raw-staging",
        "--moe-down-sum",
        "--expert-mv-vram",
        "--expert-mv-prefetch",
        "--expert-mv-link-ctas",
        "--expert-mv-ptx",
        "--expert-mv-ptx-side",
        "--expert-mv-device",
        "--expert-mv-max-tokens",
        "--glm-verify-mv-max-tokens",
        "--qsa-pool-pad-steps",
        "--expert-mv-split-glu",
        "--expert-miss-overlap",
        "--expert-peer-fetch",
        "--no-expert-peer-fetch",
        "--qsa-scores-tc",
        "--qsa-attn-chain",
        "--qsa-always-score",
        "--hc-mix-ggml",
        "--hc-mix-bf16",
        "--hc-l2-prefetch",
        "--no-hc-norm-regs",
        "--no-hc-down-split",
        "--no-hc-down-hoist",
        "--no-hc-up-stage",
        "--hc-post-norm-ggml",
        "--hc-moe-fold",
        "--no-hc-moe-fold",
        "--hc-fuse",
        "--hc-xn-f32",
        "--hc-quant-ggml",
        "--expert-mmq",
        "--nvfp4-once",
        // --nvfp4-mvq is hidden since it became the default (#1133).
        "--no-nvfp4-mvq",
        // --q8-mmq is hidden since it became the default (#858).
        "--q8-mmq-ggml",
        "--q8-mmq-stream",
        "--q8-mmq-fixup",
        "--tu-mmq-ggml",
        "--hc-up-mix-ggml",
        "--no-hc-up-mix-bf16",
        "--hc-xn-written",
        "--hc-streams-bf16",
        "--hc-mix-f16",
        "--hc-mix-rebuild-bf16",
        "--hc-chain-ggml",
        "--hc-pre-ggml",
        "--gate-quant-ggml",
        "--gate-norm-ggml",
        "--shared-quant-ggml",
        "--shared-quant-small-ggml",
        "--shexp-sum-ggml",
        "--hc-norm-fma",
        "--ple-ops",
        "--no-after-overlap",
        "--no-input-pack",
        "--no-stage-launch-ahead",
        "--no-commit-beside-draft",
        "--commit-last-late",
        "--commit-capture",
        "--mtp-ahead",
        "--mtp-pipe-rows",
        "--mtp-moe-fused-rows",
        "--load-threads",
        "--load-parallel-device",
        "--no-mmap",
    ] {
        let at = help
            .find(&format!("{flag} "))
            .or_else(|| help.find(&format!("{flag}\n")))
            .unwrap_or_else(|| panic!("{flag} missing from the help"));
        assert!(at > heading, "{flag} is not under the switch heading");
    }
    // The flags after the switches in the code keep the general heading.
    for flag in ["--model-name", "--spec-draft-window", "--api-key"] {
        assert!(
            help.find(flag).unwrap() < heading,
            "{flag} is under the switch heading"
        );
    }
}

#[test]
fn stage_graph_cache_is_the_built_in_default_unless_given() {
    // No flag: the built-in 64 sets per device (#858).
    assert_eq!(switches(&[]).stage_graph_cache, None);
    assert_eq!(
        switches(&["--stage-graph-cache", "16"]).stage_graph_cache,
        Some(16)
    );
    assert_eq!(
        legion::ggml::stage_graph::DEFAULT_CACHE_SETS,
        64,
        "the --help text states 64"
    );
}

#[test]
fn stage_graph_spare_execs_is_the_built_in_default_unless_given() {
    // No flag: the built-in 8 spares per device (#836).
    assert_eq!(switches(&[]).stage_graph_spare_execs, None);
    assert_eq!(
        switches(&["--stage-graph-spare-execs", "0"]).stage_graph_spare_execs,
        Some(0)
    );
    assert_eq!(
        legion::ggml::stage_graph::DEFAULT_SPARE_EXECS,
        8,
        "the --help text states 8"
    );
    assert_eq!(switches(&[]).stage_graph_exec_steal_ms, None);
    assert_eq!(
        legion::ggml::stage_graph::DEFAULT_EXEC_STEAL_MS,
        1000,
        "the --help text states 1000"
    );
}

#[test]
fn stage_graphs_flag_sets_its_switch() {
    // The flags alone (`legion run` too); `legion serve` sets the default itself (#849).
    assert_eq!(switches(&[]).stage_graphs, None);
    assert_eq!(switches(&["--stage-graphs"]).stage_graphs, Some(true));
    assert_eq!(switches(&["--no-stage-graphs"]).stage_graphs, Some(false));
    let a = args(&[]);
    assert!(a.engine.model.switches.stage_graphs(), "wanted by default");
    assert!(!a.engine.model.switches.stage_graphs_given());
    let off = args(&["--no-stage-graphs"]);
    assert!(!off.engine.model.switches.stage_graphs());
    // The head split is on unless turned off (#836).
    assert_eq!(switches(&[]).stage_graph_split, None);
    assert_eq!(
        switches(&["--no-stage-graph-split"]).stage_graph_split,
        Some(false)
    );
    // Early captures are on unless turned off (#827).
    assert_eq!(switches(&[]).stage_graph_early, None);
    assert_eq!(
        switches(&["--no-stage-graph-early-capture"]).stage_graph_early,
        Some(false)
    );
    // Several sequences' take-ins wait for their reuse unless turned off (#836).
    assert_eq!(switches(&[]).stage_graph_reuse_capture, None);
    assert_eq!(
        switches(&["--no-stage-graph-capture-on-reuse"]).stage_graph_reuse_capture,
        Some(false)
    );
    // Unchanged kept graphs skip their words check unless turned off (#814).
    assert_eq!(switches(&[]).stage_graph_words_skip, None);
    assert_eq!(
        switches(&["--no-stage-graph-words-skip"]).stage_graph_words_skip,
        Some(false)
    );
    // Kept graphs' captures go through the shared store unless held (#858).
    assert_eq!(switches(&[]).stage_graph_hold_kept, None);
    assert_eq!(
        switches(&["--stage-graph-hold-kept"]).stage_graph_hold_kept,
        Some(true)
    );
}

#[test]
fn kv_vmm_keep_implies_kv_vmm_and_no_kv_vmm_turns_both_off() {
    // No flag: the engine picks (on for a V100 layer split of 2 GPUs with the cache, #814).
    let s = switches(&[]);
    assert_eq!((s.kv_vmm, s.kv_vmm_keep), (None, None));
    let s = switches(&["--kv-vmm-keep", "16384"]);
    assert_eq!((s.kv_vmm, s.kv_vmm_keep), (Some(true), Some(16384)));
    let s = switches(&["--no-kv-vmm"]);
    assert_eq!((s.kv_vmm, s.kv_vmm_keep), (Some(false), None));
    for bad in [
        &["--no-kv-vmm", "--kv-vmm"][..],
        &["--no-kv-vmm", "--kv-vmm-keep", "1"],
    ] {
        let argv = ["legion", "serve"].into_iter().chain(bad.iter().copied());
        assert!(Cli::try_parse_from(argv).is_err(), "{bad:?} parsed");
    }
}
