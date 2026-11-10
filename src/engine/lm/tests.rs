use super::*;
use crate::model::core::Hparams;
use crate::model::{gemma4, nemotron_h, qwen4, qwen35};

#[test]
fn qwen4exp_drafts_two_tokens_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    qwen4::tiny::write(&path, None, 1).unwrap();
    let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    assert_eq!(hp.arch.default_spec_k(), Some(2));
}

#[test]
fn the_load_drops_its_page_cache_on_integrated_gpus_only() {
    use crate::ggml::DeviceKind;
    assert!(release_load_cache(false, Some(DeviceKind::IntegratedGpu)));
    assert!(!release_load_cache(true, Some(DeviceKind::IntegratedGpu)));
    assert!(!release_load_cache(false, Some(DeviceKind::Gpu)));
    assert!(!release_load_cache(false, Some(DeviceKind::Cpu)));
    assert!(!release_load_cache(false, None));
}

#[test]
fn qwen4exp_drafts_up_to_eight_on_integrated_gpus() {
    assert_eq!(default_spec_k("qwen4exp", Some(2), true, true), 8);
    assert_eq!(default_spec_k("qwen4exp", Some(2), false, true), 2);
    assert_eq!(default_spec_k("gemma4", Some(8), true, true), 8);
    assert_eq!(default_spec_k("qwen35", None, true, false), 4);
    assert_eq!(default_spec_k("qwen35", None, false, true), 4);
    // qwen35's own MTP head on an integrated GPU.
    assert_eq!(default_spec_k("qwen35", None, true, true), 8);
    assert_eq!(default_spec_k("qwen35moe", None, true, true), 8);
}

#[test]
fn qwen4exp_windows_the_mtp_head_on_integrated_gpus() {
    assert_eq!(default_mtp_window("qwen4exp", true, true), 32768);
    assert_eq!(default_mtp_window("qwen4exp", true, false), 0);
    assert_eq!(default_mtp_window("qwen4exp", false, true), 0);
    assert_eq!(default_mtp_window("gemma4", true, true), 0);
}

#[test]
fn draft_head_type_flag_wins_over_the_default() {
    use crate::model::core::DraftHeadType as T;
    assert!(draft_head_q8(T::Q8_0, "qwen4exp", false, false));
    assert!(!draft_head_q8(T::Model, "qwen35", true, true));
    assert!(!draft_head_q8(T::Model, "qwen4exp", true, true));
    assert!(!draft_head_q8(T::Auto, "qwen4exp", false, true));
    assert!(draft_head_q8(T::Auto, "qwen35", true, false));
    assert!(!draft_head_q8(T::Auto, "qwen35", false, true));
}

#[test]
fn qwen35_mtp_head_runs_as_q8_on_integrated_gpus() {
    assert!(default_mtp_head_q8("qwen35", true, false));
    assert!(default_mtp_head_q8("qwen35moe", true, false));
    assert!(!default_mtp_head_q8("qwen35", false, true));
    // qwen4exp: only a head shipped 16-bit (the NVFP4 checkpoint's own, #823); the released
    // Q8_0 head GGUF stays as stored.
    assert!(!default_mtp_head_q8("qwen4exp", true, false));
    assert!(default_mtp_head_q8("qwen4exp", true, true));
    assert!(!default_mtp_head_q8("qwen4exp", false, true));
    assert!(!default_mtp_head_q8("gemma4", true, true));
}

#[test]
fn only_16_and_32_bit_floats_are_wide() {
    use crate::ggml::types;
    for ty in [types::BF16, types::F16, types::F32] {
        assert!(wide_float(ty));
    }
    for ty in [types::Q8_0, types::Q4_K, types::Q6_K, types::I8] {
        assert!(!wide_float(ty));
    }
}

#[test]
fn mtp_lm_head_auto_is_q4_for_a_wide_qwen4exp_head_on_integrated_gpus() {
    use crate::spec::DraftLmHead::{Auto, Model, Q4};
    assert_eq!(mtp_lm_head(Auto, "qwen4exp", true, true), Q4);
    // A quantized output head (the GGUF), a discrete GPU, another arch: as stored.
    assert_eq!(mtp_lm_head(Auto, "qwen4exp", true, false), Model);
    assert_eq!(mtp_lm_head(Auto, "qwen4exp", false, true), Model);
    assert_eq!(mtp_lm_head(Auto, "qwen35", true, true), Model);
    // An explicit choice wins.
    assert_eq!(mtp_lm_head(Model, "qwen4exp", true, true), Model);
    assert_eq!(mtp_lm_head(Q4, "qwen35", false, false), Q4);
}

#[test]
fn gemma4_drafts_up_to_eight_tokens_by_default() {
    // The depth policy (`crate::spec::depth`) picks fewer where verify rows cost.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    gemma4::tiny::write(&path, 1).unwrap();
    let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    assert_eq!(hp.arch.default_spec_k(), Some(8));
}

#[test]
fn other_architectures_use_the_engine_draft_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    for arch in ["qwen35", "nemotron_h_moe"] {
        match arch {
            // Dense and MoE Qwen3.5/3.6 share the same Arch implementation.
            "qwen35" => qwen35::tiny::write(&path, &Default::default()).unwrap(),
            "nemotron_h_moe" => nemotron_h::tiny::write(&path, 1).unwrap(),
            _ => unreachable!(),
        }
        let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
        assert_eq!(hp.arch.default_spec_k(), None, "{arch}");
        assert_eq!(hp.arch.default_spec_k().unwrap_or(4), 4, "{arch}");
    }
}

/// Long prompts' micro-chunks: about two per stage, 256-token steps, 2048..=4096 tokens.
#[test]
fn long_prompts_pick_their_micro_chunk() {
    let long = LongPrompts::default();
    // 10K tokens: 2 GPUs 10291 / 4 -> 2816; 4 GPUs -> the 2048 floor.
    assert_eq!(long.micro_for(10291, 2), 2816);
    assert_eq!(long.micro_for(10291, 4), 2048);
    // A 64K step keeps the full micro-chunk.
    assert_eq!(long.micro_for(65536, 2), 4096);
    assert_eq!(long.micro_for(65536, 4), 4096);
    // One stage, or adapting off: always the configured one.
    assert_eq!(long.micro_for(3000, 1), 4096);
    let fixed = LongPrompts {
        micro_per_stage: 0.0,
        ..long
    };
    assert_eq!(fixed.micro_for(10291, 2), 4096);
    // A configured micro-chunk under the floor stays the cap.
    let small = LongPrompts {
        micro: 1024,
        ..long
    };
    assert_eq!(small.micro_for(10291, 2), 1024);
}
/// `--short-prompt-tokens` (#579): short prompts borrow the cache for staging, by default only
/// for GLM-5.3-Flash over several stages, and only below `--long-prompt-tokens`.
#[test]
fn short_prompts_borrow_the_cache_for_staging() {
    // Qwen, Gemma and one-stage GLM: off.
    assert_eq!(short_prompt_default(false, 4), 0);
    assert_eq!(short_prompt_default(true, 1), 0);
    assert_eq!(short_prompt_default(true, 4), GLM_SHORT_PROMPT);
    assert_eq!(GLM_SHORT_PROMPT, crate::residency::STAGE_MIN_TOKENS);
    assert_eq!(LongPrompts::default().short_from(), None);
    let with = |short: usize, min: usize| LongPrompts {
        short_min_tokens: Some(short),
        min_tokens: min,
        ..LongPrompts::default()
    };
    assert_eq!(with(0, 2048).short_from(), None);
    assert_eq!(with(128, 2048).short_from(), Some(128));
    // At or above the long-prompt threshold, a prompt is a long one.
    assert_eq!(with(2048, 2048).short_from(), None);
    assert_eq!(with(4096, 2048).short_from(), None);
}

/// `--long-micro-even`: micro_for's count of micro-chunks, of equal size in whole KV blocks.
#[test]
fn even_micro_chunks_share_a_step_out() {
    let long = LongPrompts {
        even: true,
        ..LongPrompts::default()
    };
    // The cut the executor makes with a micro-chunk size: full ones, then the rest.
    let cut = |tokens: usize, micro: usize| -> Vec<usize> {
        (0..tokens)
            .step_by(micro)
            .map(|at| micro.min(tokens - at))
            .collect()
    };
    // 123K over 2 GPUs: 30 x 4096 and 320 by default; 31 of 4032 and the rest evenly.
    let step = 123_200;
    assert_eq!(cut(step, long.micro_for(step, 2)).len(), 31);
    assert_eq!(*cut(step, long.micro_for(step, 2)).last().unwrap(), 320);
    let even = long.even_micro(step, 2);
    assert_eq!(even, 4032);
    let parts = cut(step, even);
    assert_eq!(parts.len(), 31);
    assert_eq!(*parts.last().unwrap(), 2240);
    // 10K over 2 GPUs (10,291 tokens left, a 10,240-token step to the last whole block):
    // 4 x 2560 rather than the default's 3 x 2816 and 1792 (sized from the tokens left);
    // over 4 GPUs 5 x 2048 either way.
    assert_eq!(long.even_micro(10_240, 2), 2560);
    assert_eq!(long.even_micro(10_240, 4), 2048);
    // 18K over 2 GPUs: 5 micro-chunks either way, of 3712 (the last 3584) not 4096 and 2048.
    assert_eq!(long.micro_for(18_432, 2), 4096);
    assert_eq!(long.even_micro(18_432, 2), 3712);
    assert_eq!(cut(18_432, 3712), vec![3712, 3712, 3712, 3712, 3584]);
    // Every size: never more micro-chunks than micro_for's, never more than its size, whole
    // blocks.
    for tokens in (1..140_000).step_by(97) {
        for stages in [1, 2, 4] {
            let cap = long.micro_for(tokens, stages);
            let m = long.even_micro(tokens, stages);
            assert!(m <= cap && m > 0, "{tokens} {stages}: {m} vs {cap}");
            assert!(
                cut(tokens, m).len() <= cut(tokens, cap).len(),
                "{tokens} {stages}"
            );
            if tokens > cap {
                assert_eq!(m % crate::model::core::KV_BLOCK, 0);
            }
        }
    }
    // Pipelining off stays off.
    let off = LongPrompts { micro: 0, ..long };
    assert_eq!(off.even_micro(10_240, 2), 0);
}

#[test]
fn ring_steps_are_whole_batches_on_integrated_gpus_unless_set() {
    let mut cfg = LmConfig::new("m.gguf", DeviceChoice::Cpu);
    cfg.max_batch_tokens = 1024;
    assert_eq!(ring_step(&cfg, true), 1024);
    assert_eq!(ring_step(&cfg, false), PREFILL_CHUNK);
    cfg.prefill_chunk = 256;
    assert_eq!(ring_step(&cfg, false), 256);
    // --prefill-chunk given: that on every device.
    cfg.ring_step = Some(384);
    assert_eq!(ring_step(&cfg, true), 384);
    assert_eq!(ring_step(&cfg, false), 384);
}

/// #1132: rings too big for the context get shorter steps, halved down to the floor; a step
/// that fits stays, and none fitting keeps the step (the fit check refuses as before).
#[test]
fn ring_steps_shrink_only_until_the_context_fits() {
    assert_eq!(fitting_ring_step(4096, 512, |_| true), 4096);
    assert_eq!(fitting_ring_step(4096, 512, |s| s <= 2048), 2048);
    assert_eq!(fitting_ring_step(4096, 512, |s| s <= 600), 512);
    assert_eq!(fitting_ring_step(4096, 512, |_| false), 4096);
    // Whole KV blocks: 4160 / 2 = 2080 rounds down to 2048.
    assert_eq!(fitting_ring_step(4160, 512, |s| s < 4160), 2048);
    // Never below the floor, never longer than asked.
    assert_eq!(fitting_ring_step(512, 512, |_| false), 512);
    assert_eq!(fitting_ring_step(300, 512, |_| false), 300);
}

/// `--tp` plans dense Gemma 4 only, and sizes the cache itself: other architectures, Gemma 4
/// with experts, drafter models and `--kv-cache-tokens` are load errors (before any GPU is
/// looked for).
#[test]
fn tensor_parallelism_rejects_what_it_does_not_support() {
    let dir = tempfile::tempdir().unwrap();
    let (qwen, gemma) = (dir.path().join("qwen.gguf"), dir.path().join("gemma.gguf"));
    qwen35::tiny::write(&qwen, &Default::default()).unwrap();
    gemma4::tiny::write(&gemma, 1).unwrap();
    let plan = |path: &std::path::Path, cfg: LmConfig| {
        layout::Layout::plan(&cfg, &GgufFile::open(path).unwrap())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    };
    let mut tp = LmConfig::new("unused", DeviceChoice::Cuda);
    tp.tp = 2;
    let e = plan(&qwen, tp.clone());
    assert!(e.contains("dense Gemma 4 models only, not qwen35"), "{e}");
    // The tiny Gemma 4 has routed experts (as the 26B-A4B).
    let e = plan(&gemma, tp.clone());
    assert!(e.contains("not gemma4 with experts"), "{e}");
    let mut drafter = tp.clone();
    drafter.spec.kind = crate::spec::SpecKind::Dflash;
    let e = plan(&gemma, drafter);
    assert!(e.contains("not a drafter model"), "{e}");
    tp.kv_cache_tokens = Some(8192);
    let e = plan(&gemma, tp.clone());
    assert!(e.contains("not with --kv-cache-tokens"), "{e}");
    // Managed memory takes a given size (the plan then stops at the experts, as above).
    tp.kv_memory = KvMemory::Managed;
    let e = plan(&gemma, tp);
    assert!(
        !e.contains("--kv-cache-tokens") && e.contains("experts"),
        "{e}"
    );
}

#[test]
fn managed_kv_memory_needs_ggml_unified_memory_set() {
    assert!(check_kv_memory(KvMemory::Device, None).is_ok());
    assert!(check_kv_memory(KvMemory::Managed, Some("1")).is_ok());
    for v in [None, Some("0"), Some("")] {
        let e = check_kv_memory(KvMemory::Managed, v).unwrap_err();
        assert!(e.contains(UNIFIED_MEMORY_ENV), "{e}");
    }
}

#[test]
fn kv_pool_leaves_room_on_shared_memory() {
    use crate::ggml::DeviceKind;
    assert_eq!(
        kv_headroom(DeviceKind::Gpu, 8192, UMA_HEADROOM),
        COMPUTE_HEADROOM
    );
    let gib = 1usize << 30;
    // 16 GiB for the OS beside the compute headroom, the fixed buffers and 1.5 GiB per 1K
    // step tokens.
    assert_eq!(
        kv_headroom(DeviceKind::IntegratedGpu, 1024, UMA_HEADROOM),
        (2 + 16 + 2) * gib + gib * 3 / 2
    );
    assert_eq!(
        kv_headroom(DeviceKind::IntegratedGpu, 4096, UMA_HEADROOM),
        (2 + 16 + 2 + 6) * gib
    );
    assert_eq!(
        LmConfig::new("m.gguf", DeviceChoice::Cuda).uma_free,
        UMA_HEADROOM
    );
}

#[test]
fn uma_free_moves_only_the_shared_memory_reserve() {
    use crate::ggml::DeviceKind;
    let gib = 1usize << 30;
    // --uma-free-gib 8 (#1172): the OS's share drops by 8 GiB, the step buffers' stays.
    assert_eq!(
        kv_headroom(DeviceKind::IntegratedGpu, 4096, 8 * gib),
        (2 + 8 + 2 + 6) * gib
    );
    assert_eq!(
        kv_headroom(DeviceKind::IntegratedGpu, 4096, 0),
        (2 + 2 + 6) * gib
    );
    // Discrete GPUs keep only the compute headroom.
    assert_eq!(
        kv_headroom(DeviceKind::Gpu, 4096, 8 * gib),
        COMPUTE_HEADROOM
    );
}

#[test]
fn request_timings_split_prompt_and_decode() {
    let t0 = Instant::now();
    let first = t0 + std::time::Duration::from_millis(500);
    let end = first + std::time::Duration::from_secs(2);
    let (p, d) = done_timings(t0, Some(first), end, 21);
    assert!((p.unwrap() - 0.5).abs() < 1e-9);
    assert!((d.unwrap() - 10.0).abs() < 1e-9);
    assert_eq!(done_timings(t0, None, end, 0), (None, None));
    assert_eq!(done_timings(t0, Some(first), end, 1).1, None);
}

#[test]
fn moe_kernel_auto_by_gpu_and_residency() {
    use crate::model::core::MoeKernel;
    let r = resolve_moe_kernel;
    // A GPU sharing the host's memory, every expert in device memory: ggml's MMQ while the
    // op only has Volta's MMA, the op once it runs on m16n8k16.
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Device, true, Some(12), false),
        MoeKernel::Graph
    );
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Device, true, Some(12), true),
        MoeKernel::Auto
    );
    // Discrete sm_80+ GPUs, Volta, host experts, the CPU: as asked.
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Device, false, Some(12), false),
        MoeKernel::Auto
    );
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Device, false, Some(7), false),
        MoeKernel::Auto
    );
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Cache, true, Some(12), false),
        MoeKernel::Auto
    );
    assert_eq!(
        r(MoeKernel::Auto, ExpertMode::Device, true, None, false),
        MoeKernel::Auto
    );
    // An explicit choice stays.
    assert_eq!(
        r(MoeKernel::Fused, ExpertMode::Device, true, Some(12), false),
        MoeKernel::Fused
    );
    assert_eq!(
        r(MoeKernel::Graph, ExpertMode::Device, true, Some(12), true),
        MoeKernel::Graph
    );
}

#[test]
fn step_defaults_per_device() {
    assert_eq!(
        default_max_batch_tokens(DeviceChoice::Cuda, true),
        MAX_BATCH_TOKENS_INTEGRATED
    );
    assert_eq!(
        default_max_batch_tokens(DeviceChoice::Cuda, false),
        MAX_BATCH_TOKENS
    );
    assert_eq!(
        default_max_batch_tokens(DeviceChoice::Cpu, true),
        MAX_BATCH_TOKENS
    );
    assert_eq!(default_prefill_chunk(DeviceChoice::Cuda, true, 4096), 4096);
    assert_eq!(
        default_prefill_chunk(DeviceChoice::Cuda, true, 256),
        PREFILL_CHUNK
    );
    assert_eq!(
        default_prefill_chunk(DeviceChoice::Cuda, false, 4096),
        PREFILL_CHUNK
    );
    assert_eq!(
        default_prefill_chunk(DeviceChoice::Cpu, true, 4096),
        PREFILL_CHUNK
    );
}

#[test]
fn auto_kv_pool_is_capped_on_shared_memory() {
    let ctx = 262_144 / KV_BLOCK;
    assert_eq!(auto_kv_want(16, ctx, false, false), 16 * ctx);
    assert_eq!(auto_kv_want(16, ctx, false, true), ctx);
    // A context longer than the cap is still one context.
    assert_eq!(auto_kv_want(16, 2 * ctx, false, true), 2 * ctx);
    let small = 8192 / KV_BLOCK;
    assert_eq!(auto_kv_want(16, small, false, true), 16 * small);
    assert_eq!(
        auto_kv_want(16, small, true, false),
        CACHE_KV_TOKENS.div_ceil(KV_BLOCK)
    );
}

#[test]
fn kv_floor_never_overrides_the_reserve_on_shared_memory() {
    assert_eq!(
        kv_floor(100, 64, false, true, false, Default::default()),
        Ok(100)
    );
    // Discrete GPUs: an automatic pool is raised to one context, as before.
    assert_eq!(
        kv_floor(10, 64, false, false, false, Default::default()),
        Ok(64)
    );
    // A given size below a context, anywhere: an error.
    assert!(kv_floor(10, 64, true, false, false, Default::default()).is_err());
    // Shared memory: no silent raise past the reserve.
    let e = kv_floor(10, 64, false, true, false, Default::default()).unwrap_err();
    assert!(e.contains("--ctx-size"), "{e}");
    // The model's native context (no --ctx-size): in full or an error, never raised past
    // what fits; the error names what fits and the options.
    assert_eq!(
        kv_floor(100, 64, false, false, true, Default::default()),
        Ok(100)
    );
    let e = kv_floor(10, 64, false, false, true, Default::default()).unwrap_err();
    for part in [
        "native context of 4096",
        "room for 640",
        "--ctx-size 640",
        "q8_0",
        "--tp",
    ] {
        assert!(e.contains(part), "{part}: {e}");
    }
}

mod hc_inject_kl;
mod hc_norm_fma_kl;
mod hc_quant;
mod kv_quant_kl;
mod pair_out_kl;
mod ple_ops_kl;
mod stage_trim;

#[test]
fn explicit_kv_pool_respects_the_reserve_on_shared_memory() {
    assert_eq!(kv_explicit(100, Some(200), true), Ok(100));
    assert_eq!(kv_explicit(300, None, true), Ok(300));
    // Discrete GPUs: the user's call.
    assert_eq!(kv_explicit(300, Some(200), false), Ok(300));
    let e = kv_explicit(300, Some(200), true).unwrap_err();
    assert!(e.contains("room for"), "{e}");
    // The automatic path's hint no longer points at --kv-cache-tokens.
    assert!(
        !kv_floor(10, 64, false, true, false, Default::default())
            .unwrap_err()
            .contains("--kv-cache-tokens")
    );
}

#[test]
fn draft_vocab_applies_to_qwen4exp_and_qwen35_mtp_and_dflash_only() {
    assert_eq!(draft_vocab_unused(0, false, false, "gemma4", 1000), None);
    assert_eq!(
        draft_vocab_unused(65536, true, false, "qwen4exp", 248320),
        None
    );
    assert_eq!(
        draft_vocab_unused(65536, true, false, "qwen35", 248320),
        None
    );
    assert_eq!(
        draft_vocab_unused(65536, true, false, "qwen35moe", 248320),
        None
    );
    assert!(draft_vocab_unused(65536, true, false, "gemma4", 262144).is_some());
    assert!(draft_vocab_unused(65536, false, false, "qwen4exp", 248320).is_some());
    assert!(draft_vocab_unused(248320, true, false, "qwen4exp", 248320).is_some());
    // A DFlash drafter borrows the target's head, whatever the architecture.
    assert_eq!(
        draft_vocab_unused(131072, false, true, "qwen35", 248320),
        None
    );
    assert_eq!(
        draft_vocab_unused(131072, false, true, "gemma4", 262144),
        None
    );
    assert!(draft_vocab_unused(248320, false, true, "qwen35", 248320).is_some());
}

#[test]
fn mtp_lm_head_applies_to_qwen4exp_and_qwen35_only() {
    use crate::spec::DraftLmHead::{Auto, Model, Q4};
    for arch in ["qwen4exp", "qwen35", "qwen35moe"] {
        assert!(mtp_lm_head_unused(Q4, arch).is_none(), "{arch}");
    }
    for arch in ["glm5-next", "gemma4"] {
        assert!(mtp_lm_head_unused(Q4, arch).is_some(), "{arch}");
        assert!(mtp_lm_head_unused(Model, arch).is_none(), "{arch}");
        // The default warns about nothing.
        assert!(mtp_lm_head_unused(Auto, arch).is_none(), "{arch}");
    }
}

#[test]
fn glm_skips_its_mtp_layer_when_not_drafting() {
    let names = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.44.ffn_down_exps.weight",
        "blk.45.nextn.eh_proj.weight",
        "blk.45.ffn_down_exps.weight",
        "output.weight",
    ];
    assert_eq!(
        unused_mtp_tensors(names, "glm5-next", 45, false),
        ["blk.45.nextn.eh_proj.weight", "blk.45.ffn_down_exps.weight"]
    );
    // Drafting with it, or another architecture: nothing skipped.
    assert!(unused_mtp_tensors(names, "glm5-next", 45, true).is_empty());
    assert!(unused_mtp_tensors(names, "qwen35moe", 45, false).is_empty());
    assert!(unused_mtp_tensors(names, "qwen4exp", 45, false).is_empty());
}

/// `--spec mtp` without `--spec-k` drafts with the MTP head: `k` is still 0 when the load
/// decides which tensors to skip, before it gets its default.
#[test]
fn mtp_drafts_before_spec_k_gets_its_default() {
    let mut spec = SpecConfig::default();
    assert!(!drafts_with_mtp_head(&spec));
    spec.kind = crate::spec::SpecKind::Mtp;
    spec.k = 0;
    assert!(drafts_with_mtp_head(&spec));
    spec.k = 3;
    assert!(drafts_with_mtp_head(&spec));
}

/// `--glm-parallel-decode` drafts with the MTP head or not at all (#669).
#[test]
fn glm_par_drafts_with_the_mtp_head_only() {
    use crate::spec::SpecKind;
    let mut spec = SpecConfig::default();
    assert!(glm_par_spec_refusal(&spec).is_none());
    spec.kind = SpecKind::Mtp;
    assert!(glm_par_spec_refusal(&spec).is_none());
    for kind in [SpecKind::Ngram, SpecKind::Suffix, SpecKind::Dflash] {
        spec.kind = kind;
        assert!(glm_par_spec_refusal(&spec).is_some(), "{kind:?}");
    }
}

/// A parallel step's logits rows: a row per sequence and prompt slot, `k + 1` per sequence
/// verifying `k` drafts, at most a step's tokens (#669).
#[test]
fn glm_par_out_rows_grow_with_verify_rows() {
    let max = crate::model::glm5next::par::MAX_TOKENS;
    assert_eq!(glm_par_out_rows(4, PREFILL_SLOTS, None), 4 + PREFILL_SLOTS);
    assert_eq!(glm_par_out_rows(0, PREFILL_SLOTS, None), 1 + PREFILL_SLOTS);
    assert_eq!(
        glm_par_out_rows(4, PREFILL_SLOTS, Some(0)),
        4 + PREFILL_SLOTS
    );
    assert_eq!(
        glm_par_out_rows(4, PREFILL_SLOTS, Some(2)),
        3 * (4 + PREFILL_SLOTS)
    );
    assert_eq!(glm_par_out_rows(256, PREFILL_SLOTS, Some(4)), max);
}

/// `--glm-parallel-decode`: steps of at most 512 tokens, and prompts of 128 tokens or more in
/// steps of `--glm-prompt-step` over the lent expert cache (#686); 512 keeps the short-prompt
/// lend's chunks.
#[test]
fn glm_par_prompts_borrow_the_cache_for_long_steps() {
    use crate::model::glm5next::par::{MAX_TOKENS, PROMPT_TOKENS};
    let mut cfg = LmConfig::new("m.gguf", DeviceChoice::Cuda);
    assert_eq!(cfg.glm_prompt_step, PROMPT_TOKENS);
    glm_par_steps(&mut cfg);
    assert_eq!(cfg.max_batch_tokens, MAX_TOKENS);
    assert_eq!(cfg.prefill_chunk, MAX_TOKENS);
    assert_eq!((cfg.pipeline_micro, cfg.pipeline_prompt), (0, 0));
    assert_eq!(cfg.kv_host_tokens, Some(0));
    let long = cfg.long_prompts;
    assert_eq!(long.min_tokens, GLM_SHORT_PROMPT);
    assert_eq!(long.batch_tokens, 2048);
    assert_eq!(long.alone_tokens, 2048);
    // The startup measures a step of `micro` tokens for what a lend frees.
    assert_eq!(long.micro, 2048);
    assert_eq!(long.short_from(), None);

    let mut cfg = LmConfig::new("m.gguf", DeviceChoice::Cuda);
    cfg.glm_prompt_step = 512;
    glm_par_steps(&mut cfg);
    let long = cfg.long_prompts;
    assert_eq!(long.min_tokens, usize::MAX);
    assert_eq!(long.micro, MAX_TOKENS);
    assert_eq!(long.short_from(), Some(GLM_SHORT_PROMPT));
    assert_eq!(cfg.max_batch_tokens, MAX_TOKENS);
}

/// The tensor-parallel fit's margin grows with the prompt tokens a step can take.
#[test]
fn tp_margin_grows_with_batch_tokens() {
    const MIB: usize = 1 << 20;
    assert_eq!(tp_margin(512), 512 * MIB);
    assert_eq!(tp_margin(1024), 512 * MIB);
    assert_eq!(tp_margin(2048), 640 * MIB);
    assert_eq!(tp_margin(4096), 896 * MIB);
}

#[test]
fn mtp_window_applies_to_paged_dense_heads_only() {
    let on = |n| mtp_window_check(n, "qwen4exp", true, false, 262_144);
    assert_eq!(
        on(0),
        MtpWindow {
            window: 0,
            note: None
        }
    );
    assert_eq!(
        on(32_768),
        MtpWindow {
            window: 32_768,
            note: None
        }
    );
    // A tiny window applies, with a note.
    let tiny = on(16);
    assert_eq!(tiny.window, 16);
    assert!(tiny.note.is_some());
    // The whole context, Gemma 4, a gather path, a sparse head: off, with a reason.
    for c in [
        on(262_144),
        mtp_window_check(32_768, "gemma4", true, false, 262_144),
        mtp_window_check(32_768, "qwen4exp", false, false, 262_144),
        mtp_window_check(32_768, "qwen4exp", true, true, 262_144),
    ] {
        assert_eq!(c.window, 0, "{c:?}");
        assert!(c.note.is_some(), "{c:?}");
    }
}

#[test]
fn iq4_nl_caches_need_whole_super_blocks_per_gpu_under_tp() {
    use crate::model::core::{KvType, KvTypes};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    gemma4::tiny::write(&path, 1).unwrap();
    let hp = Hparams::from_gguf(&GgufFile::open(&path).unwrap()).unwrap();
    let iq4 = KvTypes {
        k: KvType::Q8_0,
        v: KvType::Iq4Nl,
    };
    // The tiny model's sliding-window layers: 2 KV heads of 128, one per GPU: 128 values.
    let e = tp_cache_rows_fit(&hp, iq4, 2).unwrap_err();
    assert!(e.contains("layer 0") && e.contains("128 values"), "{e}");
    // Other types need no such rows; neither does one GPU's whole row of 256 values.
    let q = KvTypes {
        k: KvType::Q8_0,
        v: KvType::Q4_0,
    };
    assert!(tp_cache_rows_fit(&hp, q, 2).is_ok());
    assert!(tp_cache_rows_fit(&hp, iq4, 1).is_ok());
}

#[test]
fn a_lone_long_prompt_steps_past_the_shared_batch() {
    // Alone, a 123K prompt is one step (one pipeline fill and drain); with sequences decoding
    // the step stays at the batch size they wait for.
    let long = LongPrompts::default();
    assert!(long.alone_tokens >= 131072);
    assert!(long.alone_tokens > long.batch_tokens);
    assert_eq!(long.batch_tokens, 65536);
}

/// The startup report's placement is the load's own: its stages' layers are `Layout::plan`'s,
/// and each stage's weight bytes are the tensors the layout puts on that stage's device.
#[test]
fn load_report_placement_matches_the_layout() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    qwen35::tiny::write(&path, &Default::default()).unwrap();
    let cfg = LmConfig {
        model_name: Some("tiny".into()),
        ctx_size: 256,
        threads: 2,
        max_seqs: 2,
        layer_split: vec![1, 3],
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    };
    let gguf = GgufFile::open(&path).unwrap();
    let layout = layout::Layout::plan(&cfg, &gguf).unwrap();
    let engine = LmEngine::load(cfg).unwrap();
    let report = engine.load_report();
    let ranges: Vec<_> = report.stages.iter().map(|s| s.layers.clone()).collect();
    let planned: Vec<_> = layout.stages.iter().map(|(_, r)| r.clone()).collect();
    assert_eq!(ranges, planned);
    assert_eq!(ranges, vec![0..1, 1..3, 3..4]);
    let mut bytes = vec![0u64; planned.len()];
    for t in gguf.tensors() {
        if let Some(s) = layout.device_stage(&t.name) {
            bytes[s] += t.size;
        }
    }
    let weights: Vec<u64> = report.stages.iter().map(|s| s.weights).collect();
    assert_eq!(weights, bytes);
    // On the CPU every tensor is some stage's: none left out.
    let total: u64 = gguf.tensors().iter().map(|t| t.size).sum();
    assert_eq!(weights.iter().sum::<u64>(), total);
    for s in &report.stages {
        assert_eq!(s.device, "CPU");
        assert_eq!(s.kv_tokens, report.kv.device_tokens);
        assert!(s.free_bytes.is_none());
    }
    // Layer 0 is a recurrent (DeltaNet) layer: no KV on its stage, states only.
    assert_eq!(report.stages[0].kv_bytes, 0);
    assert!(report.stages[0].states_bytes > 0);
    assert!(report.stages[2].kv_bytes > 0);
    assert_eq!(report.kv.ctx_size, 256);
    assert!(!report.kv.native);
    assert_eq!(report.kv.host_tokens, 0);
    assert_eq!(report.kv.types, "f16/f16");
    assert!(report.ngram.is_none());
}

#[test]
fn saved_states_go_to_host_memory_with_the_expert_cache_on_discrete_gpus() {
    // The default with the expert cache on discrete GPUs (#441).
    assert!(saved_states_host(true, true, 8, true, 1, None));
    // `--prefix-states-vram` keeps them in VRAM.
    assert!(!saved_states_host(true, true, 8, true, 1, Some(true)));
    // Without the expert cache, on integrated GPUs or the CPU, under tensor parallelism, with
    // no saved slots or a model without recurrent state: unchanged.
    assert!(!saved_states_host(false, true, 8, true, 1, None));
    assert!(!saved_states_host(true, false, 8, true, 1, None));
    assert!(!saved_states_host(true, true, 8, true, 2, None));
    assert!(!saved_states_host(true, true, 0, true, 1, None));
    assert!(!saved_states_host(true, true, 8, false, 1, None));
}

#[test]
fn draft_head_type_f16_copies_to_f16() {
    use crate::ggml::types;
    use crate::model::core::DraftHeadType as T;
    assert_eq!(
        draft_head_copy(T::F16, "qwen4exp", false, false),
        Some(types::F16)
    );
    assert_eq!(
        draft_head_copy(T::F16, "qwen35", true, true),
        Some(types::F16)
    );
    assert_eq!(
        draft_head_copy(T::Q8_0, "qwen4exp", false, false),
        Some(types::Q8_0)
    );
    assert_eq!(draft_head_copy(T::Model, "qwen35", true, true), None);
    // The default stays as it was: no copy on discrete GPUs.
    assert_eq!(draft_head_copy(T::Auto, "qwen4exp", false, true), None);
    assert_eq!(draft_head_copy(T::Auto, "qwen35", false, true), None);
    assert_eq!(
        draft_head_copy(T::Auto, "qwen35", true, false),
        Some(types::Q8_0)
    );
    // Flash-Next's Q8_0 head GGUF on a GB10: as stored, as before.
    assert_eq!(draft_head_copy(T::Auto, "qwen4exp", true, false), None);
    assert_eq!(
        draft_head_copy(T::Auto, "qwen4exp", true, true),
        Some(types::Q8_0)
    );
}

/// DeepSeek-V4.1-Flash's Engram tables load as a lookup table, not as weights (no stage holds
/// them), on the CPU as on GPUs, and the report says where they went.
#[test]
fn engram_tables_load_as_a_lookup_table() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    crate::model::deepseek41::tiny::write_with(
        &path,
        5,
        crate::model::deepseek41::tiny::Tables::Separate,
    )
    .unwrap();
    let cfg = LmConfig {
        ctx_size: 256,
        threads: 2,
        max_seqs: 2,
        ..LmConfig::new(&path, DeviceChoice::Cpu)
    };
    let gguf = GgufFile::open(&path).unwrap();
    let hp = Hparams::from_gguf(&gguf).unwrap();
    let (parts, names) = crate::model::deepseek41::engram::lookup_table(&gguf, &hp).unwrap();
    let engine = LmEngine::load(cfg).unwrap();
    let report = engine.load_report();
    let tables: u64 = names.iter().map(|n| gguf.tensor(n).unwrap().size).sum();
    assert_eq!(tables, parts.iter().map(|p| p.bytes()).sum::<u64>());
    let engram = report.engram.as_ref().expect("an Engram table");
    assert_eq!(engram.bytes, tables);
    assert_ne!(engram.io, "auto");
    assert!(report.ngram.is_none());
    let total: u64 = gguf.tensors().iter().map(|t| t.size).sum();
    let weights: u64 = report.stages.iter().map(|s| s.weights).sum();
    // Loaded as weights the tables would count in the stages' bytes.
    assert!(weights + tables <= total, "{weights} + {tables} > {total}");
}

#[test]
fn qwen4exp_keeps_more_graph_shapes_with_mtp_on_integrated_gpus() {
    assert_eq!(default_kept_graphs("qwen4exp", true, true), Some((256, 32)));
    assert_eq!(default_kept_graphs("qwen4exp", true, false), None);
    assert_eq!(default_kept_graphs("qwen4exp", false, true), None);
    assert_eq!(default_kept_graphs("qwen35moe", true, true), None);
}

#[test]
fn kept_verify_graphs_have_a_byte_budget() {
    const GIB: usize = 1 << 30;
    // Given: as given (0: no limit).
    assert_eq!(verify_budget(Some(512), 40 * GIB), 512 << 20);
    assert_eq!(verify_budget(Some(0), 40 * GIB), 0);
    // Not given: an eighth of the free memory, at most 2 GiB, never "no limit".
    assert_eq!(verify_budget(None, 40 * GIB), 2 * GIB);
    assert_eq!(verify_budget(None, 8 * GIB), GIB);
    assert_eq!(verify_budget(None, 0), 1);
}

#[test]
fn verify_rows_pad_to_the_longest_draft_only_when_asked() {
    // Default, fixed depth: 1 + spec_k whatever was drafted.
    assert_eq!(verify_rows(2, None, 8, false), 9);
    // Default, depth policy: its depth, or the longest draft when deeper (n-gram drafts).
    assert_eq!(verify_rows(2, Some(4), 8, false), 5);
    assert_eq!(verify_rows(6, Some(4), 8, false), 7);
    // `--spec-pad-longest on`: the longest draft alone.
    assert_eq!(verify_rows(2, None, 8, true), 3);
    assert_eq!(verify_rows(2, Some(4), 8, true), 3);
    assert_eq!(verify_rows(6, Some(4), 8, true), 7);
}

#[test]
fn the_decode_state_spills_only_where_it_does_not_fit() {
    let gib = 1usize << 30;
    // Gemma 4 31B over 2 GPUs with 13 GiB free between them and a 2 GiB reserve each: 16
    // sequences' rings (19.5 GiB) spill, 4 sequences' (4.9 GiB) stay in VRAM, as before.
    let free = 13 * gib;
    assert!(spill_needed(
        (19 * gib + gib / 2, 19 * gib + gib / 2),
        free,
        2 * gib,
        2,
        true
    ));
    assert!(!spill_needed((5 * gib, 5 * gib), free, 2 * gib, 2, true));
    // Under tensor parallelism the reserve counts: 8 sequences' rings (9.8 GiB) spill.
    assert!(spill_needed(
        (9 * gib + gib * 4 / 5, 9 * gib + gib * 4 / 5),
        free,
        2 * gib,
        2,
        true
    ));
    // One GPU or a layer split: only once it would not fit at all.
    assert!(!spill_needed(
        (5 * gib, 5 * gib),
        6 * gib,
        2 * gib,
        1,
        false
    ));
    assert!(spill_needed((6 * gib, 6 * gib), 6 * gib, 2 * gib, 1, false));
    // Without rings nothing spills (the other states are small and stay where they were).
    assert!(!spill_needed((20 * gib, 0), free, 2 * gib, 2, true));
}

#[test]
fn lazily_backed_slots_leave_a_chunk_each_for_rounding() {
    assert_eq!(lazy_slot_cap(10_000, 16), 10_000 - 16 * LAZY_CHUNK_BLOCKS);
    assert_eq!(lazy_slot_cap(100, 16), 0);
}

#[test]
fn slot_prompt_copies_fit_in_the_asked_contexts_buffer() {
    // A mixed steps' buffer of 396 MiB at 32768 growing 3 KiB per position, 442 MiB with the
    // copies, growing 3 + 2.9 KiB per position; sized for 262144.
    let mib = 1 << 20;
    let (c, asked) = ((16_384, 32_768), 262_144);
    let (base, with) = ((396 * mib, 3072.0), (442 * mib, 6042.0));
    let at = |(b, s): (usize, f64), ctx: usize| b as f64 + s * (ctx as f64 - 32_768.0);
    let span = slot_stage_span(base, with, c, asked);
    assert_eq!(span % KV_BLOCK, 0);
    assert!(at(with, span) <= at(base, asked));
    assert!(at(with, span + KV_BLOCK) > at(base, asked));
    assert!(span > 122_880 && span < asked);
}

#[test]
fn slot_prompt_copies_that_do_not_grow() {
    let (c, mib) = ((16_384, 32_768), 1 << 20);
    assert_eq!(
        slot_stage_span((400 * mib, 0.0), (400 * mib, 0.0), c, 262_144),
        262_144
    );
    assert_eq!(
        slot_stage_span((400 * mib, 0.0), (401 * mib, 0.0), c, 262_144),
        0
    );
}

#[test]
fn slot_prompt_copies_never_below_the_first_measurement() {
    // The copies take the room only short of c0: none.
    let (c, mib) = ((16_384, 32_768), 1 << 20);
    assert_eq!(
        slot_stage_span((400 * mib, 1.0), (500 * mib, 4096.0), c, 40_960),
        0
    );
}

#[test]
fn ring_checkpoint_slots_by_free_memory() {
    let gib = 1usize << 30;
    // 16 rings of 1 GiB: 3 per ring where a third of the free memory holds them.
    assert_eq!(ring_checkpoint_slots(None, 16, gib, Some(300 << 30)), 48);
    assert_eq!(ring_checkpoint_slots(None, 16, gib, Some(90 << 30)), 30);
    // Never fewer than one per ring, nor more without knowing the free memory.
    assert_eq!(ring_checkpoint_slots(None, 16, gib, Some(1 << 30)), 16);
    assert_eq!(ring_checkpoint_slots(None, 16, gib, None), 16);
    assert_eq!(ring_checkpoint_slots(Some(5), 16, gib, None), 5);
    // Off: one per ring, the single copy at a prompt's end, whatever the memory.
    assert_eq!(ring_checkpoint_slots(Some(0), 16, gib, Some(300 << 30)), 16);
    // Which takes nothing from a lazily backed host tier.
    assert_eq!(
        ring_checkpoints_in_budget(16, 16, gib, Some(200 * 1024), 1 << 20, 1024),
        (16, Some(200 * 1024))
    );
}

#[test]
fn ring_checkpoints_come_out_of_a_lazy_host_tier() {
    let gib = 1usize << 30;
    let mib = 1usize << 20;
    // No lazily backed tier: as sized.
    assert_eq!(
        ring_checkpoints_in_budget(48, 16, gib, None, mib, 0),
        (48, None)
    );
    // 32 checkpoints past one per ring take 32 GiB of a 200 GiB budget.
    assert_eq!(
        ring_checkpoints_in_budget(48, 16, gib, Some(200 * 1024), mib, 1024),
        (48, Some(168 * 1024))
    );
    // The budget keeps `keep` blocks: only 4 more checkpoints fit.
    assert_eq!(
        ring_checkpoints_in_budget(48, 16, gib, Some(10 * 1024), mib, 6 * 1024),
        (20, Some(6 * 1024))
    );
    // One per ring was there before: never fewer.
    assert_eq!(
        ring_checkpoints_in_budget(48, 16, gib, Some(1024), mib, 2048),
        (16, Some(1024))
    );
}

#[test]
fn long_step_buffer_grows_with_the_past_up_to_its_measurement() {
    let mib = 1usize << 20;
    let s = LongStep {
        device: None,
        first: 1000 * mib,
        last: 2000 * mib,
    };
    let past = 100_000;
    assert_eq!(s.at(0, past), 1000 * mib);
    assert_eq!(s.at(50_000, past), 1500 * mib);
    assert_eq!(s.at(past, past), 2000 * mib);
    // Past the measurement (or an unknown end): the measured longest.
    assert_eq!(s.at(usize::MAX, past), 2000 * mib);
    // A buffer measured no bigger after a long past stays flat.
    let flat = LongStep {
        device: None,
        first: 700 * mib,
        last: 700 * mib,
    };
    assert_eq!(flat.at(12_345, past), 700 * mib);
}
