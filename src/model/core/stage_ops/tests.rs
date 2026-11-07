use super::*;

#[test]
fn listed_ops_are_unique() {
    let mut names = PURE.to_vec();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PURE.len());
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
        "cpu_share_collect",
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
        "nvfp4_a4_mm",
        "nvfp4_a4_mm_multi",
        "nvfp4_a4_mm_id",
        "nvfp4_a4_mm_id_weighted",
        "nvfp4_a4_swiglu",
        "nvfp4_a4_swiglu_id",
        "nvfp4_a4_quant",
        "mmq_ids",
        "q8_mmq",
        "shared_q8_small",
        "q8_head_out",
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
    set_commit_capture(true);
    assert!(capture_safe(&op, &node));
    set_commit_capture(false);
    assert!(!capture_safe(&op, &node));
    // The listed ops do not depend on it.
    let gdn = ExternalOp { name: "gdn", ..op };
    assert!(capture_safe(&gdn, &node));
}
