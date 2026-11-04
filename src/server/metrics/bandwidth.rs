//! `/metrics` families of #391's model bytes beyond the per-phase totals: the bytes by kind,
//! device-memory and link GB/s and their share of the measured roofline, each pipeline
//! stage's part, and with `legion serve --hw-counters` the GPUs' hardware counters.

use std::sync::Arc;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;

use super::{F64Gauge, PhaseLabels};
use crate::engine::EngineStats;
use crate::ggml::model_bytes::Phase;
use crate::hw::counters::HwCounters;
use crate::hw::roofline::share;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct KindLabels {
    phase: &'static str,
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StageLabels {
    phase: &'static str,
    stage: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StageKindLabels {
    phase: &'static str,
    stage: String,
    kind: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RooflineLabels {
    stage: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct GpuLabels {
    gpu: String,
}

/// A phase's bytes of one kind.
type KindBytes = fn(&Phase) -> u64;

/// A phase's bytes by kind, as the `kind` label says.
pub const KINDS: [(&str, KindBytes); 7] = [
    ("dense", |p| p.dense),
    ("experts_upper", |p| p.experts_upper),
    ("experts_device", |p| p.experts_device),
    ("experts_host", |p| p.experts_host),
    ("staged", |p| p.staged),
    ("kv", |p| p.kv),
    ("kv_host", |p| p.kv_host),
];

/// The hardware counters' gauges, by GPU (NVML index).
struct HwFamilies {
    counters: Arc<HwCounters>,
    nvlink_tx: Family<GpuLabels, F64Gauge>,
    nvlink_rx: Family<GpuLabels, F64Gauge>,
    pcie_tx: Family<GpuLabels, F64Gauge>,
    pcie_rx: Family<GpuLabels, F64Gauge>,
    memory_busy: Family<GpuLabels, F64Gauge>,
    sm_busy: Family<GpuLabels, F64Gauge>,
    sampled: Family<GpuLabels, F64Gauge>,
}

pub struct BandwidthMetrics {
    kind_bytes: Family<KindLabels, F64Gauge>,
    memory_gb_s: Family<PhaseLabels, F64Gauge>,
    link_gb_s: Family<PhaseLabels, F64Gauge>,
    roofline_ratio: Family<PhaseLabels, F64Gauge>,
    roofline_memory: Family<RooflineLabels, F64Gauge>,
    roofline_link: Family<RooflineLabels, F64Gauge>,
    stage_bytes: Family<StageKindLabels, F64Gauge>,
    stage_seconds: Family<StageLabels, F64Gauge>,
    stage_memory_gb_s: Family<StageLabels, F64Gauge>,
    stage_roofline_ratio: Family<StageLabels, F64Gauge>,
    hw: Option<HwFamilies>,
}

fn family<L>(registry: &mut Registry, name: &str, help: &str) -> Family<L, F64Gauge>
where
    L: Clone + std::hash::Hash + Eq + EncodeLabelSet + std::fmt::Debug + Send + Sync + 'static,
{
    let f = Family::<L, F64Gauge>::default();
    registry.register(name, help, f.clone());
    f
}

impl BandwidthMetrics {
    pub fn new(registry: &mut Registry) -> Self {
        let r = registry;
        Self {
            kind_bytes: family(
                r,
                "legion_engine_model_kind_bytes",
                "Model bytes since start by phase and kind: dense weights, experts (upper bound \
                 for experts in the weights; an expert cache's from VRAM and from host memory), \
                 staged expert copies, KV from VRAM and from the host tier",
            ),
            memory_gb_s: family(
                r,
                "legion_engine_model_memory_gb_per_second",
                "Model bytes read from device memory over the steps' time, GB/s, by phase",
            ),
            link_gb_s: family(
                r,
                "legion_engine_model_link_gb_per_second",
                "Model bytes over the host link (in-place host reads and staging copies) over \
                 the steps' time, GB/s, by phase",
            ),
            roofline_ratio: family(
                r,
                "legion_engine_model_roofline_ratio",
                "legion_engine_model_memory_gb_per_second over the device's measured memory \
                 roofline (0 when unknown), by phase",
            ),
            roofline_memory: family(
                r,
                "legion_engine_roofline_memory_gb_per_second",
                "Measured device-memory read roofline of each pipeline stage's GPUs, GB/s",
            ),
            roofline_link: family(
                r,
                "legion_engine_roofline_link_gb_per_second",
                "Measured host-link read roofline of each pipeline stage's GPUs, GB/s",
            ),
            stage_bytes: family(
                r,
                "legion_engine_stage_model_kind_bytes",
                "legion_engine_model_kind_bytes of the target steps by pipeline stage",
            ),
            stage_seconds: family(
                r,
                "legion_engine_stage_busy_seconds",
                "Time each pipeline stage spent running the target steps (launch to done)",
            ),
            stage_memory_gb_s: family(
                r,
                "legion_engine_stage_memory_gb_per_second",
                "A stage's model bytes from device memory over its busy time, GB/s",
            ),
            stage_roofline_ratio: family(
                r,
                "legion_engine_stage_roofline_ratio",
                "legion_engine_stage_memory_gb_per_second over the stage's memory roofline",
            ),
            hw: None,
        }
    }

    /// Exports `counters` (`legion serve --hw-counters`) from now on.
    pub fn add_hw(&mut self, registry: &mut Registry, counters: Arc<HwCounters>) {
        let r = registry;
        self.hw = Some(HwFamilies {
            counters,
            nvlink_tx: family(
                r,
                "legion_gpu_nvlink_tx_bytes",
                "NVLink data bytes the GPU sent since sampling started (NVML, all links)",
            ),
            nvlink_rx: family(
                r,
                "legion_gpu_nvlink_rx_bytes",
                "NVLink data bytes the GPU received since sampling started (NVML, all links)",
            ),
            pcie_tx: family(
                r,
                "legion_gpu_pcie_tx_bytes",
                "PCIe bytes the GPU sent (NVML throughput samples integrated over time)",
            ),
            pcie_rx: family(
                r,
                "legion_gpu_pcie_rx_bytes",
                "PCIe bytes the GPU received (NVML throughput samples integrated over time)",
            ),
            memory_busy: family(
                r,
                "legion_gpu_memory_busy_seconds",
                "Seconds the GPU's memory was read or written (NVML utilization integrated)",
            ),
            sm_busy: family(
                r,
                "legion_gpu_sm_busy_seconds",
                "Seconds a kernel ran on the GPU (NVML utilization integrated)",
            ),
            sampled: family(
                r,
                "legion_gpu_sampled_seconds",
                "Seconds the hardware counters were sampled over",
            ),
        });
    }

    /// Refreshes every gauge from `stats` (and the hardware counters).
    pub fn set(&self, stats: &EngineStats) {
        let mb = &stats.model_bytes;
        let roof = stats.roofline();
        for (phase, p) in [
            ("prefill", &mb.prefill),
            ("decode", &mb.decode),
            ("draft", &mb.draft),
        ] {
            for (kind, f) in KINDS {
                self.kind_bytes
                    .get_or_create(&KindLabels { phase, kind })
                    .set(f(p) as f64);
            }
            let l = PhaseLabels { phase };
            self.memory_gb_s.get_or_create(&l).set(p.device_gb_s());
            self.link_gb_s.get_or_create(&l).set(p.link_gb_s());
            self.roofline_ratio
                .get_or_create(&l)
                .set(share(p.device_gb_s(), roof.memory_gb_s).unwrap_or(0.0));
        }
        for (s, st) in mb.stages.iter().enumerate().take(mb.n_stages.max(1)) {
            let stage = s.to_string();
            let r = stats.rooflines[s];
            let rl = RooflineLabels {
                stage: stage.clone(),
            };
            self.roofline_memory.get_or_create(&rl).set(r.memory_gb_s);
            self.roofline_link.get_or_create(&rl).set(r.link_gb_s);
            for (phase, p) in [("prefill", &st.prefill), ("decode", &st.decode)] {
                for (kind, f) in KINDS {
                    self.stage_bytes
                        .get_or_create(&StageKindLabels {
                            phase,
                            stage: stage.clone(),
                            kind,
                        })
                        .set(f(p) as f64);
                }
                let l = StageLabels {
                    phase,
                    stage: stage.clone(),
                };
                self.stage_seconds.get_or_create(&l).set(p.seconds);
                self.stage_memory_gb_s
                    .get_or_create(&l)
                    .set(p.device_gb_s());
                self.stage_roofline_ratio
                    .get_or_create(&l)
                    .set(share(p.device_gb_s(), r.memory_gb_s).unwrap_or(0.0));
            }
        }
        if let Some(hw) = &self.hw {
            for (index, g) in hw.counters.snapshot() {
                let l = GpuLabels {
                    gpu: index.to_string(),
                };
                hw.nvlink_tx.get_or_create(&l).set(g.nvlink_tx_bytes);
                hw.nvlink_rx.get_or_create(&l).set(g.nvlink_rx_bytes);
                hw.pcie_tx.get_or_create(&l).set(g.pcie_tx_bytes);
                hw.pcie_rx.get_or_create(&l).set(g.pcie_rx_bytes);
                hw.memory_busy.get_or_create(&l).set(g.memory_busy_seconds);
                hw.sm_busy.get_or_create(&l).set(g.sm_busy_seconds);
                hw.sampled.get_or_create(&l).set(g.seconds);
            }
        }
    }
}
