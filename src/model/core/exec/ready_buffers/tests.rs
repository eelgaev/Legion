use super::{KEEP_FREE, PROMPT_BUFFERS, fits, prompt_buffers, shared, wanted};
use crate::ggml::DeviceKind;

#[test]
fn integrated_gpus_reserve_one_buffer_per_kept_graph_up_to_two() {
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, false, false, 3), 2);
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, false, false, 2), 2);
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, false, false, 1), 1);
}

#[test]
fn two_reserved_buffers_of_one_stage_are_shared() {
    assert!(shared(true, PROMPT_BUFFERS));
    assert!(!shared(true, 1));
    assert!(!shared(true, 0));
    assert!(!shared(false, PROMPT_BUFFERS));
}

#[test]
fn shared_buffers_charge_two_for_three_kept_graphs() {
    assert_eq!(prompt_buffers(3, true), 2);
    assert_eq!(prompt_buffers(2, true), 2);
    assert_eq!(prompt_buffers(1, true), 1);
    for kept in 1..=3 {
        assert_eq!(prompt_buffers(kept, false), kept);
    }
}

#[test]
fn off_meta_lent_and_other_devices_reserve_none() {
    assert_eq!(wanted(false, DeviceKind::IntegratedGpu, false, false, 3), 0);
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, true, false, 3), 0);
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, false, true, 3), 0);
    for kind in [
        DeviceKind::Gpu,
        DeviceKind::Cpu,
        DeviceKind::Accelerator,
        DeviceKind::Other,
    ] {
        assert_eq!(wanted(true, kind, false, false, 3), 0);
    }
}

#[test]
fn a_buffer_must_leave_the_reserve_free() {
    let gib = 1usize << 30;
    assert!(fits(KEEP_FREE + 2 * gib, 2 * gib));
    assert!(!fits(KEEP_FREE + 2 * gib - 1, 2 * gib));
    assert!(!fits(KEEP_FREE, 1));
    assert!(!fits(0, gib));
}

#[test]
fn an_empty_measurement_reserves_nothing() {
    assert!(!fits(usize::MAX, 0));
}

#[test]
fn on_by_default() {
    assert!(super::enabled());
}
