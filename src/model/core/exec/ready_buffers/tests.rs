use super::{KEEP_FREE, fits, wanted};
use crate::ggml::DeviceKind;

#[test]
fn integrated_gpus_reserve_one_buffer_per_kept_graph() {
    assert_eq!(wanted(true, DeviceKind::IntegratedGpu, false, false, 3), 3);
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
