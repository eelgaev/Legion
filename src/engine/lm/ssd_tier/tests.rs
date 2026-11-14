use super::*;

const IGPU: Option<DeviceKind> = Some(DeviceKind::IntegratedGpu);

#[test]
fn auto_reads_experts_from_the_ssd_only_when_they_do_not_fit_on_an_integrated_gpu() {
    assert!(wanted(ExpertSsd::Auto, "glm5-next", 1, 1, IGPU, false));
    assert!(!wanted(ExpertSsd::Auto, "glm5-next", 1, 1, IGPU, true));
    assert!(!wanted(
        ExpertSsd::Auto,
        "glm5-next",
        1,
        1,
        Some(DeviceKind::Gpu),
        false
    ));
    assert!(!wanted(ExpertSsd::Auto, "glm5-next", 1, 1, None, false));
}

#[test]
fn the_tier_is_glm_only_and_single_stage() {
    for mode in [ExpertSsd::Auto, ExpertSsd::On] {
        for arch in ["qwen35moe", "gemma4", "nemotron_h_moe", "glm4moe"] {
            assert!(!wanted(mode, arch, 1, 1, IGPU, false), "{arch}");
        }
        assert!(!wanted(mode, "glm5-next", 2, 1, IGPU, false));
        assert!(!wanted(mode, "glm5-next", 1, 2, IGPU, false));
    }
    assert!(!wanted(ExpertSsd::Off, "glm5-next", 1, 1, IGPU, false));
}

#[test]
fn on_forces_the_tier_on_the_cpu_and_integrated_gpus() {
    assert!(wanted(ExpertSsd::On, "glm5-next", 1, 1, None, true));
    assert!(wanted(ExpertSsd::On, "glm5-next", 1, 1, IGPU, true));
    assert!(!wanted(
        ExpertSsd::On,
        "glm5-next",
        1,
        1,
        Some(DeviceKind::Gpu),
        true
    ));
}

#[test]
fn the_budget_is_free_memory_less_kv_steps_and_reserve_unless_given() {
    let args = SsdArgs::default();
    assert_eq!(budget(&args, Some(100 << 30), 30 << 30).unwrap(), 68 << 30);
    assert_eq!(budget(&args, Some(20 << 30), 30 << 30).unwrap(), 0);
    assert!(budget(&args, None, 0).is_err());
    let args = SsdArgs {
        cache_bytes: Some(5 << 30),
        ..SsdArgs::default()
    };
    assert_eq!(budget(&args, None, 30 << 30).unwrap(), 5 << 30);
}

#[test]
fn the_host_tier_is_for_the_expert_cache_of_one_discrete_gpu_with_a_pinned_arena() {
    let gpu = Some(DeviceKind::Gpu);
    let cache = ExpertMode::Cache;
    let pc = "x86_64";
    assert!(host_tier_wanted(pc, cache, "qwen4exp", 1, 1, gpu, true));
    // Integrated GPUs, the CPU, several stages, tensor parallelism, other models, other
    // expert modes and a host without an arena stay as they are.
    assert!(!host_tier_wanted(pc, cache, "qwen4exp", 1, 1, IGPU, true));
    assert!(!host_tier_wanted(pc, cache, "qwen4exp", 1, 1, None, true));
    assert!(!host_tier_wanted(pc, cache, "qwen4exp", 2, 1, gpu, true));
    assert!(!host_tier_wanted(pc, cache, "qwen4exp", 1, 2, gpu, true));
    assert!(!host_tier_wanted(pc, cache, "glm5-next", 1, 1, gpu, true));
    assert!(!host_tier_wanted(pc, cache, "qwen4exp", 1, 1, gpu, false));
    assert!(!host_tier_wanted(
        pc,
        ExpertMode::Device,
        "qwen4exp",
        1,
        1,
        gpu,
        true
    ));
}

#[test]
fn a_host_tier_pass_takes_the_tokens_the_free_vram_fits() {
    let gib = 1usize << 30;
    // A step of 8704 tokens measured at 4 GiB: 2 GiB free fit half of it, in whole blocks.
    assert_eq!(host_tier_pass_cap(2 * gib, 4 * gib, 8704, 64), 4352);
    assert_eq!(host_tier_pass_cap(2 * gib + 1000, 4 * gib, 8704, 64), 4352);
    // More than the measured step when more is free.
    assert_eq!(host_tier_pass_cap(8 * gib, 4 * gib, 8704, 64), 17408);
    // Never below the smallest pass; no cap without a measurement.
    assert_eq!(host_tier_pass_cap(0, 4 * gib, 8704, 64), HOST_TIER_MIN_PASS);
    assert_eq!(host_tier_pass_cap(gib, 0, 8704, 64), 0);
}

#[test]
fn the_host_tier_is_off_on_power9_and_arm_hosts() {
    let gpu = Some(DeviceKind::Gpu);
    let cache = ExpertMode::Cache;
    // A V100 box with one stage and little free memory, and an ARM host with a discrete GPU:
    // neither changes its load.
    assert!(!host_tier_wanted(
        "powerpc64",
        cache,
        "qwen4exp",
        1,
        1,
        gpu,
        true
    ));
    assert!(!host_tier_wanted(
        "aarch64", cache, "qwen4exp", 1, 1, gpu, true
    ));
    assert!(!host_tier_wanted(
        "aarch64", cache, "qwen4exp", 1, 1, IGPU, true
    ));
}

#[test]
fn the_heat_file_is_in_the_state_directory_and_names_the_checkpoint() {
    let model = std::path::Path::new("/models/q/Model-00001-of-00004.gguf");
    let at = |state, home| host_tier_heat_file(state, home, model, 123);
    let name = "legion/expert-heat/Model-00001-of-00004.gguf.123.heat";
    assert_eq!(
        at(Some("/x/state"), Some("/home/u")),
        Some(std::path::Path::new("/x/state").join(name))
    );
    assert_eq!(
        at(None, Some("/home/u")),
        Some(std::path::Path::new("/home/u/.local/state").join(name))
    );
    // A relative XDG_STATE_HOME is not one.
    assert_eq!(
        at(Some("state"), Some("/home/u")),
        Some(std::path::Path::new("/home/u/.local/state").join(name))
    );
    assert_eq!(at(None, None), None);
    // Another checkpoint's bytes: another file.
    assert_ne!(
        host_tier_heat_file(None, Some("/home/u"), model, 124),
        at(None, Some("/home/u"))
    );
}
