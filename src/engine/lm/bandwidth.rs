//! The engine's side of #391's model bytes: each stage's roofline, found once at load, and the
//! request-done line's bandwidth fields.

use crate::ggml::model_bytes::{MAX_STAGES, Phase, Totals};
use crate::hw::roofline::{StageRoofline, share};

use super::layout::Layout;

/// Each pipeline stage's roofline (its GPU's, or its GPUs' together under tensor
/// parallelism), from the GPU names; 0 for unmeasured devices and the CPU.
pub(super) fn stage_rooflines(layout: &Layout) -> [StageRoofline; MAX_STAGES] {
    let mut out = [StageRoofline::default(); MAX_STAGES];
    for (si, (d, _)) in layout.stages.iter().enumerate().take(MAX_STAGES) {
        out[si] = match layout.tp_devices.first() {
            Some(gpu) => StageRoofline::of(&gpu.description(), layout.tp_devices.len()),
            None => d.map_or_else(StageRoofline::default, |d| {
                StageRoofline::of(&d.description(), 1)
            }),
        };
    }
    out
}

/// A phase's bandwidth figures for a log line: model GB, device-memory GB/s, its share of the
/// roofline (percent, 0 when unknown), link GB/s.
pub(super) struct PhaseFigures {
    pub gb: f64,
    pub device_gb_s: f64,
    pub roofline_pct: f64,
    pub link_gb_s: f64,
}

impl PhaseFigures {
    pub fn of(p: &Phase, roofline: StageRoofline) -> Self {
        let device_gb_s = p.device_gb_s();
        Self {
            gb: p.total() as f64 / 1e9,
            device_gb_s,
            roofline_pct: share(device_gb_s, roofline.memory_gb_s).map_or(0.0, |s| s * 100.0),
            link_gb_s: p.link_gb_s(),
        }
    }

    /// A request's prefill and decoding (decode and verify steps with the draft passes).
    pub fn of_request(t: &Totals, roofline: StageRoofline) -> (Self, Self) {
        (
            Self::of(&t.prefill, roofline),
            Self::of(&t.decoding(), roofline),
        )
    }
}
