//! NVIDIA GPU discovery through NVML (read-only queries; libnvidia-ml is loaded at runtime).

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::nv_link::IntDeviceType;
use nvml_wrapper::error::NvmlError;
use serde::Serialize;

use super::numa::pci_numa_node;

/// NVML never reports more NVLink links per GPU than this.
const MAX_NVLINKS: u32 = 18;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub pci_bus_id: String,
    /// CUDA compute capability as `(major, minor)`.
    pub compute_capability: Option<(i32, i32)>,
    pub memory_total: u64,
    pub memory_free: u64,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub pcie_max_gen: Option<u32>,
    pub pcie_max_width: Option<u32>,
    /// NUMA node the GPU is attached to, when the kernel knows it.
    pub numa_node: Option<u32>,
    pub nvlinks: Vec<NvLinkInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NvLinkInfo {
    pub link: u32,
    pub up: bool,
    /// What the link goes to, when NVML says: `gpu`, `cpu`, `switch` or `unknown`.
    pub remote: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuReport {
    pub driver_version: Option<String>,
    /// CUDA version the driver supports, as `major.minor`.
    pub cuda_driver_version: Option<String>,
    pub gpus: Vec<GpuInfo>,
}

/// Query every GPU. `Err` means NVML itself is unavailable (no driver or library).
pub fn detect() -> Result<GpuReport, NvmlError> {
    let nvml = Nvml::init()?;
    let count = nvml.device_count()?;
    let mut gpus = Vec::new();
    for index in 0..count {
        let dev = nvml.device_by_index(index)?;
        let mem = dev.memory_info()?;
        let pci_bus_id = dev.pci_info().map(|p| p.bus_id).unwrap_or_default();
        let bdf = normalize_bdf(&pci_bus_id);
        let nvlinks = (0..MAX_NVLINKS)
            .map_while(|link| {
                let l = dev.link_wrapper_for(link);
                let up = l.is_active().ok()?;
                let remote = l.remote_device_type(link).ok().map(|t| {
                    match t {
                        IntDeviceType::Gpu => "gpu",
                        IntDeviceType::Ibmnpu => "cpu",
                        IntDeviceType::Switch => "switch",
                        IntDeviceType::Unknown => "unknown",
                    }
                    .to_string()
                });
                Some(NvLinkInfo { link, up, remote })
            })
            .collect();
        gpus.push(GpuInfo {
            index,
            name: dev.name()?,
            uuid: dev.uuid()?,
            compute_capability: dev
                .cuda_compute_capability()
                .ok()
                .map(|c| (c.major, c.minor)),
            memory_total: mem.total,
            memory_free: mem.free,
            pcie_gen: dev.current_pcie_link_gen().ok(),
            pcie_width: dev.current_pcie_link_width().ok(),
            pcie_max_gen: dev.max_pcie_link_gen().ok(),
            pcie_max_width: dev.max_pcie_link_width().ok(),
            numa_node: pci_numa_node(&bdf),
            pci_bus_id,
            nvlinks,
        });
    }
    let cuda_driver_version = nvml.sys_cuda_driver_version().ok().map(|v| {
        format!(
            "{}.{}",
            nvml_wrapper::cuda_driver_version_major(v),
            nvml_wrapper::cuda_driver_version_minor(v)
        )
    });
    Ok(GpuReport {
        driver_version: nvml.sys_driver_version().ok(),
        cuda_driver_version,
        gpus,
    })
}

/// NVML reports `00000000:01:00.0` (8-digit domain); sysfs names the device `0000:01:00.0`.
pub fn normalize_bdf(bus_id: &str) -> String {
    let lower = bus_id.to_ascii_lowercase();
    match lower.split_once(':') {
        Some((domain, rest)) if domain.len() > 4 => {
            format!("{}:{rest}", &domain[domain.len() - 4..])
        }
        _ => lower,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf_is_normalized_to_sysfs_spelling() {
        assert_eq!(normalize_bdf("00000000:01:00.0"), "0000:01:00.0");
        assert_eq!(normalize_bdf("0000:C1:00.0"), "0000:c1:00.0");
        assert_eq!(normalize_bdf(""), "");
    }
}
