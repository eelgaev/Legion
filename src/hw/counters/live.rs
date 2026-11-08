//! What the dashboard (#1198) needs from NVML beyond `/metrics`' sums: each GPU's links and
//! their full scales (read once), and per sample its power, clocks, VRAM, and GPM's DRAM
//! bandwidth and C2C throughput where the GPU has GPM (Hopper and newer). NVML queries only.

use nvml_wrapper::enum_wrappers::device::Clock;
use nvml_wrapper::enum_wrappers::nv_link::IntDeviceType;
use nvml_wrapper::{Device, Nvml};

use crate::dashboard::scale::{self, LinkClass};

/// What a GPU is and how it is connected, read when the sampler starts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    /// The link's max generation and width (the GPU's and the slot's), not the current ones:
    /// an idle link trains down to gen 1.
    pub pcie_max_gen: Option<u32>,
    pub pcie_max_width: Option<u32>,
    /// Each active NVLink link's class, by link index.
    pub nvlinks: Vec<LinkClass>,
    pub nvlink_version: Option<u32>,
    /// NVLink-C2C links and each one's max bandwidth (MB/s), when NVML reports them.
    pub c2c_links: Option<u32>,
    pub c2c_max_mb_s: Option<u32>,
    pub mem_bus_bits: Option<u32>,
    pub max_mem_mhz: Option<u32>,
    /// One memory shared with the CPU (GB10): NVML has no VRAM of its own to report.
    pub unified: bool,
    /// GPM metrics are available (Hopper and newer).
    pub gpm: bool,
}

impl GpuInfo {
    pub fn cpu_links(&self) -> u32 {
        self.nvlinks
            .iter()
            .filter(|&&c| c == LinkClass::Cpu)
            .count() as u32
    }

    pub fn peer_links(&self) -> u32 {
        self.nvlinks
            .iter()
            .filter(|&&c| c == LinkClass::Peer)
            .count() as u32
    }

    /// Whether the GPU sits on NVLink-C2C (C2C links reported, or a part known to have it).
    pub fn has_c2c(&self) -> bool {
        self.c2c_links.is_some_and(|n| n > 0) || self.unified || {
            let n = self.name.to_ascii_uppercase();
            ["GH200", "GB200", "GB300", "GB10"]
                .iter()
                .any(|p| n.contains(p))
        }
    }
}

/// One field value as an unsigned integer, `None` when NVML does not have it.
pub(super) fn field_u64(nvml: &Nvml, d: &Device<'_>, field: u32, scope: u32) -> Option<u64> {
    use nvml_wrapper_sys::bindings::{
        nvmlFieldValue_t, nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_INT,
        nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG,
        nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG_LONG,
    };
    // SAFETY: a plain C struct; all zeroes is a valid value.
    let mut v: nvmlFieldValue_t = unsafe { std::mem::zeroed() };
    v.fieldId = field;
    v.scopeId = scope;
    let f = nvml.lib().nvmlDeviceGetFieldValues.as_ref().ok()?;
    // SAFETY: one initialized entry NVML fills in place; the handle is the live device's.
    let ret = unsafe { f(d.handle(), 1, &mut v) };
    if ret != 0 || v.nvmlReturn != 0 {
        return None;
    }
    // SAFETY: the union member `valueType` names.
    unsafe {
        if v.valueType == nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_INT {
            Some(u64::from(v.value.uiVal))
        } else if v.valueType == nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG {
            Some(v.value.ulVal)
        } else if v.valueType == nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG_LONG {
            Some(v.value.ullVal)
        } else {
            None
        }
    }
}

/// Reads GPU `d`'s links and limits. `gpu_bus_ids` are every GPU's PCI bus ids (lowercase),
/// to tell a peer link from a CPU link where NVML does not name the remote device type.
pub(super) fn probe(nvml: &Nvml, d: &Device<'_>, links: u32, gpu_bus_ids: &[String]) -> GpuInfo {
    use nvml_wrapper_sys::bindings::field_id::{
        NVML_FI_DEV_C2C_LINK_COUNT, NVML_FI_DEV_C2C_LINK_GET_MAX_BW,
    };
    let name = d.name().unwrap_or_default();
    let nvlinks = (0..links)
        .map(|l| {
            let w = d.link_wrapper_for(l);
            let remote = w.remote_device_type(l).ok().map(|t| match t {
                IntDeviceType::Gpu => "gpu",
                IntDeviceType::Ibmnpu => "cpu",
                IntDeviceType::Switch => "switch",
                IntDeviceType::Unknown => "unknown",
            });
            let remote_is_gpu = w.remote_pci_info().ok().map(|p| {
                let id = p.bus_id.to_ascii_lowercase();
                gpu_bus_ids.iter().any(|g| *g == id)
            });
            scale::classify(remote, remote_is_gpu)
        })
        .collect();
    let nvlink_version = (links > 0)
        .then(|| d.link_wrapper_for(0).version().ok())
        .flatten();
    let c2c_links = field_u64(nvml, d, NVML_FI_DEV_C2C_LINK_COUNT, 0).map(|n| n as u32);
    let c2c_max_mb_s = c2c_links
        .filter(|&n| n > 0)
        .and_then(|_| field_u64(nvml, d, NVML_FI_DEV_C2C_LINK_GET_MAX_BW, 0))
        .map(|n| n as u32);
    let unified = name.to_ascii_uppercase().contains("GB10") || d.memory_info().is_err();
    GpuInfo {
        pcie_max_gen: d.max_pcie_link_gen().ok(),
        pcie_max_width: d.max_pcie_link_width().ok(),
        nvlinks,
        nvlink_version,
        c2c_links,
        c2c_max_mb_s,
        mem_bus_bits: d.memory_bus_width().ok(),
        max_mem_mhz: d.max_clock_info(Clock::Memory).ok(),
        unified,
        gpm: d.gpm_support().unwrap_or(false),
        name,
    }
}

/// GPM metric ids (nvml.h `nvmlGpmMetricId_t`).
const GPM_DRAM_BW_UTIL: u32 = 10;
const GPM_C2C_DATA_TX_PER_SEC: u32 = 102;
const GPM_C2C_DATA_RX_PER_SEC: u32 = 103;

/// GPM's figures between two samples.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct GpmFigures {
    /// DRAM bandwidth utilization, 0-100.
    pub dram_pct: Option<f64>,
    /// C2C data sent and received, bytes per second.
    pub c2c_tx: Option<f64>,
    pub c2c_rx: Option<f64>,
}

/// GPM: one sample per sampler tick, figures between consecutive ones.
pub(super) struct Gpm<'n> {
    nvml: &'n Nvml,
    last: Option<nvml_wrapper::gpm::GpmSample<'n>>,
    c2c: bool,
}

impl<'n> Gpm<'n> {
    pub(super) fn new(nvml: &'n Nvml, c2c: bool) -> Self {
        Self {
            nvml,
            last: None,
            c2c,
        }
    }

    /// Takes a sample of `d` and returns the figures since the previous one.
    pub(super) fn tick(&mut self, d: &Device<'n>) -> Option<GpmFigures> {
        use nvml_wrapper_sys::bindings::{NVML_GPM_METRICS_GET_VERSION, nvmlGpmMetricsGet_t};
        let now = d.gpm_sample().ok()?;
        let prev = self.last.replace(now)?;
        let now = self.last.as_ref()?;
        let f = self.nvml.lib().nvmlGpmMetricsGet.as_ref().ok()?;
        let ids: &[u32] = if self.c2c {
            &[
                GPM_DRAM_BW_UTIL,
                GPM_C2C_DATA_TX_PER_SEC,
                GPM_C2C_DATA_RX_PER_SEC,
            ]
        } else {
            &[GPM_DRAM_BW_UTIL]
        };
        // SAFETY: a plain C struct (about 8 KiB); all zeroes is a valid value.
        let mut req: Box<nvmlGpmMetricsGet_t> = Box::new(unsafe { std::mem::zeroed() });
        req.version = NVML_GPM_METRICS_GET_VERSION;
        req.numMetrics = ids.len() as u32;
        // SAFETY: the handles stay owned by `prev` and `now`, alive through the call.
        unsafe {
            req.sample1 = prev.handle();
            req.sample2 = now.handle();
        }
        for (m, &id) in req.metrics.iter_mut().zip(ids) {
            m.metricId = id;
        }
        // SAFETY: a request NVML reads and fills in place; the samples are this device's.
        if unsafe { f(&mut *req) } != 0 {
            return None;
        }
        let value = |i: usize| {
            let m = &req.metrics[i];
            (i < ids.len() && m.nvmlReturn == 0 && m.value.is_finite()).then_some(m.value)
        };
        // MiB/s.
        let mib = |v: Option<f64>| v.map(|v| v * 1024.0 * 1024.0);
        Some(GpmFigures {
            dram_pct: value(0),
            c2c_tx: if self.c2c { mib(value(1)) } else { None },
            c2c_rx: if self.c2c { mib(value(2)) } else { None },
        })
    }
}
