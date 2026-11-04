use super::roomy;
use crate::ggml::DeviceKind;

#[test]
fn integrated_prompt_graphs_get_headroom() {
    assert!(roomy(DeviceKind::IntegratedGpu, false));
}

#[test]
fn decode_graphs_and_discrete_gpus_stay_exact() {
    assert!(!roomy(DeviceKind::IntegratedGpu, true));
    for kind in [DeviceKind::Gpu, DeviceKind::Cpu, DeviceKind::Accelerator] {
        assert!(!roomy(kind, false));
        assert!(!roomy(kind, true));
    }
}
