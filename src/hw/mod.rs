//! Host hardware topology and placement helpers.

pub mod comm_bench;
pub mod counters;
pub mod cuda_arch;
pub mod gpu;
pub mod host_core;
pub mod numa;
pub mod numa_bind;
pub mod per_device;
pub mod roofline;

use std::fmt::Write as _;
use std::fs;

use serde::Serialize;

use gpu::GpuReport;
use numa::Topology;

/// Everything `legion info` reports; later input for the planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HardwareInfo {
    pub arch: String,
    /// `None` when NVML is unavailable; `gpu_error` says why.
    pub gpu: Option<GpuReport>,
    pub gpu_error: Option<String>,
    pub numa_nodes: Vec<NumaNodeInfo>,
    pub unified_memory: UnifiedMemory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NumaNodeInfo {
    pub id: u32,
    pub cpus: usize,
    pub cpu_list: String,
    pub mem_total: u64,
    pub mem_free: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// GB10 (DGX Spark): the GPU is integrated and shares LPDDR5X with the CPU.
    IntegratedShared,
    /// POWER + NVLink to the CPU: ATS makes system memory coherent for the GPU.
    CoherentNvlink,
    /// Separate VRAM behind PCIe.
    Discrete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnifiedMemory {
    pub kind: MemoryKind,
    /// What was detected and why.
    pub reason: String,
}

impl HardwareInfo {
    pub fn detect() -> Self {
        let gpu = gpu::detect().map_err(|e| e.to_string());
        let topo = Topology::detect().ok();
        let model = fs::read_to_string("/proc/device-tree/model").ok();
        Self::assemble(std::env::consts::ARCH, model.as_deref(), gpu, topo.as_ref())
    }

    pub fn assemble(
        arch: &str,
        machine_model: Option<&str>,
        gpu: Result<GpuReport, String>,
        topo: Option<&Topology>,
    ) -> Self {
        let numa_nodes = topo.map(numa_infos).unwrap_or_default();
        let unified_memory = classify_memory(arch, machine_model, gpu.as_ref().ok());
        let (gpu, gpu_error) = match gpu {
            Ok(r) => (Some(r), None),
            Err(e) => (None, Some(e)),
        };
        Self {
            arch: arch.to_string(),
            gpu,
            gpu_error,
            numa_nodes,
            unified_memory,
        }
    }

    /// Human-readable report.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Architecture: {}", self.arch);
        match &self.gpu {
            None => {
                let _ = writeln!(
                    out,
                    "GPUs: no NVIDIA driver ({})",
                    self.gpu_error.as_deref().unwrap_or("NVML unavailable")
                );
            }
            Some(r) => {
                let _ = writeln!(
                    out,
                    "Driver: {}  (CUDA {})",
                    r.driver_version.as_deref().unwrap_or("unknown"),
                    r.cuda_driver_version.as_deref().unwrap_or("unknown")
                );
                if r.gpus.is_empty() {
                    out.push_str("GPUs: none found\n");
                }
                for g in &r.gpus {
                    let _ = writeln!(out, "GPU {}: {}", g.index, g.name);
                    let _ = writeln!(out, "  UUID:     {}", g.uuid);
                    let _ = writeln!(out, "  PCI:      {}", g.pci_bus_id);
                    if let Some((major, minor)) = g.compute_capability {
                        let _ = writeln!(out, "  Compute:  {major}.{minor}");
                    }
                    let _ = writeln!(
                        out,
                        "  Memory:   {} total, {} free",
                        gib(g.memory_total),
                        gib(g.memory_free)
                    );
                    let _ = writeln!(
                        out,
                        "  PCIe:     gen{} x{} (max gen{} x{})",
                        opt(g.pcie_gen),
                        opt(g.pcie_width),
                        opt(g.pcie_max_gen),
                        opt(g.pcie_max_width)
                    );
                    let _ = writeln!(out, "  NUMA:     {}", opt(g.numa_node));
                    if g.nvlinks.is_empty() {
                        out.push_str("  NVLink:   none\n");
                    } else {
                        for l in &g.nvlinks {
                            let _ = writeln!(
                                out,
                                "  NVLink {}: {}{}",
                                l.link,
                                if l.up { "up" } else { "down" },
                                l.remote
                                    .as_ref()
                                    .map(|r| format!(" -> {r}"))
                                    .unwrap_or_default()
                            );
                        }
                    }
                }
            }
        }
        out.push_str("NUMA nodes:\n");
        if self.numa_nodes.is_empty() {
            out.push_str("  unknown\n");
        }
        for n in &self.numa_nodes {
            let _ = writeln!(
                out,
                "  node {}: {} CPUs ({}), {} total, {} free",
                n.id,
                n.cpus,
                n.cpu_list,
                gib(n.mem_total),
                gib(n.mem_free)
            );
        }
        let _ = writeln!(
            out,
            "Memory model: {:?}: {}",
            self.unified_memory.kind, self.unified_memory.reason
        );
        out
    }
}

fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(|| "?".into(), |v| v.to_string())
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn numa_infos(topo: &Topology) -> Vec<NumaNodeInfo> {
    topo.nodes
        .iter()
        .map(|n| NumaNodeInfo {
            id: n.id,
            cpus: n.cpus.len(),
            cpu_list: compress_cpulist(&n.cpus),
            mem_total: n.mem_total,
            mem_free: n.mem_free,
        })
        .collect()
}

/// Inverse of `numa::parse_cpulist`: `[0,1,2,5]` -> `0-2,5`.
fn compress_cpulist(cpus: &[u32]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        parts.push(if i == j {
            cpus[i].to_string()
        } else {
            format!("{}-{}", cpus[i], cpus[j])
        });
        i = j + 1;
    }
    parts.join(",")
}

/// Decide the memory model from the architecture, the device-tree machine model and GPU data.
pub fn classify_memory(
    arch: &str,
    machine_model: Option<&str>,
    gpu: Option<&GpuReport>,
) -> UnifiedMemory {
    if let Some(g) = gpu.and_then(|r| r.gpus.iter().find(|g| g.name.contains("GB10"))) {
        return UnifiedMemory {
            kind: MemoryKind::IntegratedShared,
            reason: format!(
                "GPU {} is a GB10 (DGX Spark): integrated, shares LPDDR5X with the CPU",
                g.index
            ),
        };
    }
    if arch.starts_with("powerpc64") || arch == "ppc64le" {
        let cpu_link = gpu.and_then(|r| {
            r.gpus.iter().find(|g| {
                g.nvlinks
                    .iter()
                    .any(|l| l.up && l.remote.as_deref() == Some("cpu"))
            })
        });
        let ac922 = machine_model.is_some_and(|m| m.contains("8335-G") || m.contains("AC922"));
        if let Some(g) = cpu_link {
            return UnifiedMemory {
                kind: MemoryKind::CoherentNvlink,
                reason: format!(
                    "{arch} with NVLink to the CPU on GPU {}: ATS gives coherent access to system memory",
                    g.index
                ),
            };
        }
        if ac922 {
            return UnifiedMemory {
                kind: MemoryKind::CoherentNvlink,
                reason: format!(
                    "{arch} machine model {:?} is an AC922 (NVLink to CPU, ATS)",
                    machine_model.unwrap_or_default().trim_end_matches('\0')
                ),
            };
        }
    }
    UnifiedMemory {
        kind: MemoryKind::Discrete,
        reason: "no GB10 and no NVLink to the CPU: discrete VRAM behind PCIe".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::gpu::{GpuInfo, NvLinkInfo};
    use super::numa::NumaNode;
    use super::*;

    fn gpu(name: &str, links: Vec<NvLinkInfo>) -> GpuReport {
        GpuReport {
            driver_version: Some("570.1".into()),
            cuda_driver_version: Some("12.8".into()),
            gpus: vec![GpuInfo {
                index: 0,
                name: name.into(),
                uuid: "GPU-1".into(),
                pci_bus_id: "00000000:01:00.0".into(),
                compute_capability: Some((8, 9)),
                memory_total: 24 << 30,
                memory_free: 23 << 30,
                pcie_gen: Some(4),
                pcie_width: Some(16),
                pcie_max_gen: Some(4),
                pcie_max_width: Some(16),
                numa_node: Some(0),
                nvlinks: links,
            }],
        }
    }

    fn topo() -> Topology {
        Topology {
            nodes: vec![NumaNode {
                id: 0,
                cpus: vec![0, 1, 2, 3, 8],
                mem_total: 64 << 30,
                mem_free: 32 << 30,
            }],
        }
    }

    #[test]
    fn no_driver_prints_message_and_still_lists_numa() {
        let info = HardwareInfo::assemble("x86_64", None, Err("lib missing".into()), Some(&topo()));
        let text = info.render();
        assert!(text.contains("no NVIDIA driver"), "{text}");
        assert!(text.contains("node 0: 5 CPUs (0-3,8)"), "{text}");
        assert_eq!(info.unified_memory.kind, MemoryKind::Discrete);
        let json = serde_json::to_value(&info).unwrap();
        assert!(json["gpu"].is_null());
        assert_eq!(json["gpu_error"], "lib missing");
    }

    #[test]
    fn empty_gpu_list_renders() {
        let empty = GpuReport {
            driver_version: None,
            cuda_driver_version: None,
            gpus: vec![],
        };
        let info = HardwareInfo::assemble("x86_64", None, Ok(empty), None);
        let text = info.render();
        assert!(text.contains("GPUs: none found"), "{text}");
        assert!(text.contains("unknown"), "{text}");
    }

    #[test]
    fn discrete_gpu_renders_details() {
        let info = HardwareInfo::assemble(
            "x86_64",
            None,
            Ok(gpu("NVIDIA GeForce RTX 4090", vec![])),
            Some(&topo()),
        );
        let text = info.render();
        assert!(text.contains("Compute:  8.9"), "{text}");
        assert!(text.contains("24.0 GiB total"), "{text}");
        assert!(text.contains("gen4 x16"), "{text}");
        assert!(text.contains("NVLink:   none"), "{text}");
        assert!(text.contains("Discrete"), "{text}");
    }

    #[test]
    fn gb10_is_integrated() {
        let r = gpu("NVIDIA GB10", vec![]);
        let m = classify_memory("aarch64", None, Some(&r));
        assert_eq!(m.kind, MemoryKind::IntegratedShared);
        assert!(m.reason.contains("LPDDR5X"));
    }

    #[test]
    fn power9_with_cpu_nvlink_is_coherent() {
        let link = NvLinkInfo {
            link: 0,
            up: true,
            remote: Some("cpu".into()),
        };
        let r = gpu("Tesla V100-SXM2-32GB", vec![link]);
        let m = classify_memory("powerpc64", Some("PowerNV 8335-GTH"), Some(&r));
        assert_eq!(m.kind, MemoryKind::CoherentNvlink);
        // Same arch without CPU links or a known model stays discrete.
        let r = gpu("Tesla V100-SXM2-32GB", vec![]);
        let m = classify_memory("powerpc64", Some("something"), Some(&r));
        assert_eq!(m.kind, MemoryKind::Discrete);
        // The device-tree model alone identifies an AC922 when NVML is absent.
        let m = classify_memory("powerpc64", Some("PowerNV 8335-GTW\0"), None);
        assert_eq!(m.kind, MemoryKind::CoherentNvlink);
    }

    #[test]
    fn x86_cpu_link_does_not_imply_coherence() {
        let link = NvLinkInfo {
            link: 0,
            up: true,
            remote: Some("cpu".into()),
        };
        let r = gpu("Some GPU", vec![link]);
        assert_eq!(
            classify_memory("x86_64", None, Some(&r)).kind,
            MemoryKind::Discrete
        );
    }

    #[test]
    fn cpulist_compression() {
        assert_eq!(compress_cpulist(&[]), "");
        assert_eq!(compress_cpulist(&[0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
    }

    #[test]
    fn topology_from_fixture_dir() {
        let dir = tempfile::tempdir().unwrap();
        for (id, cpus, total) in [(0, "0-1", 4096), (1, "2,3", 2048)] {
            let n = dir.path().join(format!("node{id}"));
            fs::create_dir(&n).unwrap();
            fs::write(n.join("cpulist"), format!("{cpus}\n")).unwrap();
            fs::write(
                n.join("meminfo"),
                format!(
                    "Node {id} MemTotal:       {total} kB\nNode {id} MemFree:        1024 kB\n"
                ),
            )
            .unwrap();
        }
        let t = Topology::from_sysfs(dir.path()).unwrap();
        let infos = numa_infos(&t);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[1].cpu_list, "2-3");
        assert_eq!(infos[0].mem_total, 4096 * 1024);
        assert_eq!(infos[1].mem_free, 1024 * 1024);
    }

    #[test]
    fn pci_numa_node_from_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let dev = dir.path().join("0000:01:00.0");
        fs::create_dir(&dev).unwrap();
        fs::write(dev.join("numa_node"), "8\n").unwrap();
        assert_eq!(numa::pci_numa_node_in(dir.path(), "0000:01:00.0"), Some(8));
        fs::write(dev.join("numa_node"), "-1\n").unwrap();
        assert_eq!(numa::pci_numa_node_in(dir.path(), "0000:01:00.0"), None);
    }
}
