use super::*;

#[test]
fn listed_ops_are_unique() {
    let mut names = PURE.to_vec();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PURE.len());
}

/// Inkling-2's own ops are captured in small steps (#1221); the GLM and DeepSeek ops its graph
/// shares only with its switch on, so GLM-5.3-Flash's and DeepSeek-V4.1's graphs capture what
/// they did before. Prompt steps keep all of them between their captures.
#[test]
fn inkling_ops_follow_the_small_step_mark_and_their_switch() {
    let mut all: Vec<&str> = [PURE, SMALL, SMALL_MM, COMMIT, INKLING, INKLING_SHARED].concat();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "an op in two lists");
    for name in INKLING {
        assert!(name.starts_with("ink_"), "{name}");
        for inkling in [false, true] {
            assert!(listed_as(name, true, inkling), "{name}");
            assert!(!listed_as(name, false, inkling), "{name}");
        }
    }
    for name in INKLING_SHARED {
        assert!(
            name.starts_with("glm_") || name.starts_with("ds4_"),
            "{name}"
        );
        assert!(listed_as(name, true, true), "{name}");
        assert!(!listed_as(name, true, false), "{name}");
        assert!(!listed_as(name, false, true), "{name}");
    }
    // The other lists do not depend on the switch.
    for inkling in [false, true] {
        assert!(listed_as("nvfp4_mm_id", false, inkling));
        assert!(listed_as("fp8_mm", true, inkling));
        assert!(!listed_as("glm_mla_attn", true, inkling));
    }
}

/// Ops that wait on the host, use side streams or change with host state between steps stay
/// out of captures.
#[test]
fn host_bound_ops_are_not_listed() {
    for name in [
        "stage_wait",
        "stage_done",
        "expert_mv_stage",
        "ssd_expert_fetch",
        "ssd_expert_predict",
        "cpu_share_publish",
        "cpu_share_glu",
        "cpu_share_down",
        "tp_all_reduce",
        "tp_reduce_scatter",
        "pair_link",
        "pair_sum",
        "glm_exchange",
    ] {
        assert!(!PURE.contains(&name), "{name}");
    }
}

/// The decode matmuls of the NVFP4 and BF16 checkpoints are captured: each one left out cuts a
/// decode step's capture in two (#836).
#[test]
fn decode_matmuls_are_listed() {
    for name in [
        "dense_mv",
        "nvfp4_mm",
        "nvfp4_mm_id",
        "nvfp4_swiglu",
        "nvfp4_swiglu_id",
        "nvfp4_once",
        "nvfp4_once_glu",
        "mmq_ids",
        "q8_mmq",
        "shared_q8_small",
        "q8_head_out",
        "q_small_m_quantize",
        "q_small_m",
        "q4k_mv",
        "l2_prefetch",
        "l2_join",
        "side_join",
    ] {
        assert!(PURE.contains(&name), "{name}");
    }
}

unsafe fn no_cpu(_: &ffi::ggml_tensor, _: usize, _: usize) {}

/// A verify commit's window copy is captured only while the commit capture is on (#814), and
/// it is not in the unconditional list.
#[test]
fn the_commit_window_follows_its_switch() {
    assert!(COMMIT.iter().all(|name| !PURE.contains(name)));
    let op = ExternalOp {
        name: "spec_window",
        cpu: no_cpu,
        cuda: None,
    };
    // SAFETY: a plain C struct; `capture_safe` reads nothing of it for this op.
    let node: ffi::ggml_tensor = unsafe { std::mem::zeroed() };
    let all = ExternalOp {
        name: "spec_windows",
        ..op
    };
    set_commit_capture(true);
    assert!(capture_safe(&op, &node));
    assert!(capture_safe(&all, &node));
    set_commit_capture(false);
    assert!(!capture_safe(&op, &node));
    assert!(!capture_safe(&all, &node));
    // The listed ops do not depend on it.
    let gdn = ExternalOp { name: "gdn", ..op };
    assert!(capture_safe(&gdn, &node));
}

/// The W4A4 NVFP4 ops are captured in small steps (a 16-user verify step), not in prompt
/// steps (#1202), and are in neither of the other lists.
#[test]
fn w4a4_ops_follow_the_small_step_mark() {
    let names = [
        "nvfp4_a4_mm",
        "nvfp4_a4_mm_multi",
        "nvfp4_a4_mm_id",
        "nvfp4_a4_mm_id_weighted",
        "nvfp4_a4_swiglu",
        "nvfp4_a4_swiglu_id",
        "nvfp4_a4_quant",
    ];
    assert_eq!(SMALL, &names[..]);
    assert!(
        SMALL
            .iter()
            .all(|n| !PURE.contains(n) && !COMMIT.contains(n))
    );
    for name in names {
        assert!(listed(name, true), "{name}");
        assert!(!listed(name, false), "{name}");
    }
    // The other listed ops do not depend on the step.
    assert!(listed("nvfp4_mm", true) && listed("nvfp4_mm", false));
    assert!(!listed("tp_all_reduce", true) && !listed("tp_all_reduce", false));
}

/// The verify step's dense matmuls (`nvfp4_mvq`, `fp8_mm`) are captured in small steps only
/// (#689), like the W4A4 ops, and are in none of the other lists.
#[test]
fn verify_matmuls_follow_the_small_step_mark() {
    assert_eq!(SMALL_MM, &["nvfp4_mvq", "fp8_mm"][..]);
    for name in SMALL_MM {
        assert!(
            !PURE.contains(name) && !SMALL.contains(name) && !COMMIT.contains(name),
            "{name}"
        );
        assert!(listed(name, true), "{name}");
        assert!(!listed(name, false), "{name}");
    }
}
