use super::{hopeless, roomy};
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

#[test]
fn hopeless_only_when_nothing_kept_could_make_it_fit() {
    const MIB: usize = 1 << 20;
    // The 4090's lent 32K-token prompt: 12,594 MiB allocated, 40 free, 36 kept: short of the
    // 13,106 with the scratch.
    assert!(hopeless(40 * MIB, 12_594 * MIB, 36 * MIB, 13_106 * MIB, 0));
    // The same buffer failing in cudaMalloc while lent: the scratch counts on top.
    assert!(hopeless(12_634 * MIB, 0, 36 * MIB, 12_594 * MIB, 512 * MIB));
    // A kept graph big enough to give the rest: the retries run.
    let (held, kept) = (12_594 * MIB, 600 * MIB);
    assert!(!hopeless(40 * MIB, held, kept, 13_106 * MIB, 0));
    // Within the slack: the retries run.
    assert!(!hopeless(12_600 * MIB, 0, 0, 12_650 * MIB, 0));
}
