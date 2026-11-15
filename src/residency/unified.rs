//! How a GPU reaches host memory, from the driver's device attributes rather than GPU names
//! (#1215): a Grace Hopper (GH200) is an Arm host whose discrete GPU reads pageable host memory
//! through the host's page tables (ATS over NVLink-C2C). Plain functions of the attribute values,
//! so the CPU tests check them against the values an AC922, a DGX Spark (GB10), a GH200 and a
//! PCIe card report.

use super::Platform;
use crate::hw::MemoryKind;

/// The device attributes the decisions here read (`CU_DEVICE_ATTRIBUTE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LinkAttrs {
    /// `INTEGRATED`: the GPU shares the host's memory (GB10).
    pub integrated: bool,
    /// `PAGEABLE_MEMORY_ACCESS`: kernels read pageable host memory.
    pub pageable_access: bool,
    /// `PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`: ... through the host's page tables (ATS:
    /// an AC922's NVLink 2.0, a Grace CPU's NVLink-C2C).
    pub pageable_uses_host_page_tables: bool,
    /// The compute capability (major, minor).
    pub cc: (i32, i32),
}

#[cfg(feature = "cuda")]
impl From<&super::cuda::Caps> for LinkAttrs {
    fn from(c: &super::cuda::Caps) -> Self {
        Self {
            integrated: c.integrated,
            pageable_access: c.pageable_access,
            pageable_uses_host_page_tables: c.pageable_uses_host_page_tables,
            cc: (c.cc_major, c.cc_minor),
        }
    }
}

/// Whether a GPU reaches host memory over NVLink-C2C as a Grace Hopper's does: an Arm host
/// (`arch` as `std::env::consts::ARCH` names it), a discrete GPU that reads pageable host memory
/// through the host's page tables. An AC922 (POWER) and a GB10 (integrated) are not.
pub fn c2c(arch: &str, a: LinkAttrs) -> bool {
    arch == "aarch64" && !a.integrated && a.pageable_access && a.pageable_uses_host_page_tables
}

/// The platform of a machine whose report says `kind` (GPU names, the device tree), on `arch`,
/// with every visible GPU's attributes (`None`: none could be read) and whether NVML names a
/// GH200 (`nvml_gh200`). An integrated GPU and an AC922 stay as the report says. A discrete GPU
/// is [`Platform::NvlinkC2c`] when every GPU's attributes say [`c2c`]; without attributes, the
/// NVML name decides as before.
pub fn platform(
    kind: MemoryKind,
    arch: &str,
    attrs: Option<&[LinkAttrs]>,
    nvml_gh200: bool,
) -> Platform {
    match kind {
        MemoryKind::IntegratedShared => Platform::Integrated,
        MemoryKind::CoherentNvlink => Platform::CoherentNvlink,
        MemoryKind::Discrete => match attrs.filter(|a| !a.is_empty()) {
            Some(a) if a.iter().all(|&a| c2c(arch, a)) => Platform::NvlinkC2c,
            Some(_) => Platform::DiscretePcie,
            None if arch == "aarch64" && nvml_gh200 => Platform::NvlinkC2c,
            None => Platform::DiscretePcie,
        },
    }
}

/// Whether a GPU's step buffers spilling into host memory is the default it is on a V100: the
/// expert cache takes the prompt steps' compute room (`ResidencyConfig::fills_prompt_room`) and
/// a 2-GPU layer split puts the KV pools in reserved virtual ranges (`--kv-vmm`). Compute
/// capability 7.0, as before (an AC922's V100s, measured there); and a Hopper (9.x) over
/// NVLink-C2C ([`c2c`], a GH200), whose spilled buffers are read over ATS as an AC922's are
/// (unmeasured). A GB10 (12.1, integrated) and every other GPU: no.
pub fn spill_default(arch: &str, a: LinkAttrs) -> bool {
    a.cc == (7, 0) || (a.cc.0 == 9 && c2c(arch, a))
}

/// [`spill_default`] for a device whose attributes are `attrs` (`None`: unknown, no) on the host
/// running this process.
pub fn spill_default_here(attrs: Option<LinkAttrs>) -> bool {
    attrs.is_some_and(|a| spill_default(std::env::consts::ARCH, a))
}

#[cfg(test)]
mod tests;
