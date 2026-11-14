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
    assert!(host_tier_wanted(cache, "qwen4exp", 1, 1, gpu, true));
    // Integrated GPUs, the CPU, several stages, tensor parallelism, other models, other
    // expert modes and a host without an arena stay as they are.
    assert!(!host_tier_wanted(cache, "qwen4exp", 1, 1, IGPU, true));
    assert!(!host_tier_wanted(cache, "qwen4exp", 1, 1, None, true));
    assert!(!host_tier_wanted(cache, "qwen4exp", 2, 1, gpu, true));
    assert!(!host_tier_wanted(cache, "qwen4exp", 1, 2, gpu, true));
    assert!(!host_tier_wanted(cache, "glm5-next", 1, 1, gpu, true));
    assert!(!host_tier_wanted(cache, "qwen4exp", 1, 1, gpu, false));
    assert!(!host_tier_wanted(
        ExpertMode::Device,
        "qwen4exp",
        1,
        1,
        gpu,
        true
    ));
}
