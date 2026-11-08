//! The GPUs' hardware counters while serving (#391, opt-in with `legion serve
//! --hw-counters`): a background thread samples NVML for each GPU the process sees (NVLink data
//! bytes per link, PCIe throughput, how busy the memory and the SMs were) and adds them up, so
//! `/metrics` and `legion bench --workload bandwidth` can put measured traffic next to the
//! analytic model bytes. NVML queries only: nothing runs on the GPUs, and without the flag no
//! thread starts. The live dashboard (#1198, `serve --tui` or a `/v1/stats` client) starts one
//! too while it is attached, and reads [`HwCounters::live`]: the same sums split by link class,
//! plus power, clocks, VRAM and GPM figures ([`live`]).

pub mod live;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use nvml_wrapper::enum_wrappers::device::{Clock, PcieUtilCounter};
use nvml_wrapper::{Device, Nvml};

use crate::dashboard::scale::LinkClass;
pub use live::GpuInfo;

/// NVML's NVLink data counters (KiB since the driver loaded; one per link, `scopeId`).
const NVLINK_DATA_TX: u32 = 138;
const NVLINK_DATA_RX: u32 = 139;
/// NVML never reports more NVLink links per GPU than this.
const MAX_NVLINKS: u32 = 18;

/// One GPU's counters since sampling started.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuSample {
    /// NVLink data bytes sent and received, over every link (to the host CPU and to peers).
    pub nvlink_tx_bytes: f64,
    pub nvlink_rx_bytes: f64,
    /// PCIe bytes sent and received (NVML's throughput samples integrated over time).
    pub pcie_tx_bytes: f64,
    pub pcie_rx_bytes: f64,
    /// Seconds device memory was being read or written, and seconds a kernel ran (NVML's
    /// utilization percentages integrated over time).
    pub memory_busy_seconds: f64,
    pub sm_busy_seconds: f64,
    /// Seconds sampled.
    pub seconds: f64,
}

/// One GPU's sampled state for the dashboard: [`GpuSample`] plus the NVLink bytes by link
/// class, C2C bytes and the last sample's gauges. `None`: NVML does not report it here.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuLive {
    pub index: u32,
    pub info: GpuInfo,
    pub totals: GpuSample,
    /// Whether NVML's PCIe throughput counters answer on this GPU.
    pub pcie_counted: bool,
    /// NVLink data bytes `(sent, received)` over the links to the CPU and to peers.
    pub nvlink_cpu: (f64, f64),
    pub nvlink_peer: (f64, f64),
    /// NVLink-C2C data bytes `(sent, received)`, from GPM's C2C throughput.
    pub c2c: Option<(f64, f64)>,
    pub power_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    /// The last sample's NVML utilization (percent of time busy).
    pub mem_util_pct: Option<f64>,
    pub sm_util_pct: Option<f64>,
    /// GPM's DRAM bandwidth utilization over the last sample, 0-100.
    pub gpm_dram_pct: Option<f64>,
}

/// Sums ([`GpuSample`]'s fields first) and gauges, as f64 bits.
const S_NVLINK_TX: usize = 0;
const S_NVLINK_RX: usize = 1;
const S_PCIE_TX: usize = 2;
const S_PCIE_RX: usize = 3;
const S_MEM_BUSY: usize = 4;
const S_SM_BUSY: usize = 5;
const S_SECONDS: usize = 6;
const S_CPU_TX: usize = 7;
const S_CPU_RX: usize = 8;
const S_PEER_TX: usize = 9;
const S_PEER_RX: usize = 10;
const S_C2C_TX: usize = 11;
const S_C2C_RX: usize = 12;
const N_SUMS: usize = 13;
/// Gauges: NaN until a sample has the value.
const G_POWER_W: usize = 0;
const G_SM_MHZ: usize = 1;
const G_MEM_MHZ: usize = 2;
const G_VRAM_USED: usize = 3;
const G_VRAM_TOTAL: usize = 4;
const G_MEM_UTIL: usize = 5;
const G_SM_UTIL: usize = 6;
const G_GPM_DRAM: usize = 7;
/// 1 once GPM reported C2C bytes, 1 once NVML's PCIe throughput answered.
const G_C2C_SEEN: usize = 8;
const G_PCIE_SEEN: usize = 9;
const N_GAUGES: usize = 10;

/// One watched GPU.
#[derive(Debug)]
struct Gpu {
    /// NVML index (PCI order).
    index: u32,
    /// Its NVLink links (0 without).
    links: u32,
    info: GpuInfo,
    sums: [AtomicU64; N_SUMS],
    gauges: [AtomicU64; N_GAUGES],
}

impl Gpu {
    fn add(&self, i: usize, v: f64) {
        let cur = f64::from_bits(self.sums[i].load(Ordering::Relaxed));
        self.sums[i].store((cur + v).to_bits(), Ordering::Relaxed);
    }

    fn get(&self, i: usize) -> f64 {
        f64::from_bits(self.sums[i].load(Ordering::Relaxed))
    }

    fn set_gauge(&self, i: usize, v: Option<f64>) {
        self.gauges[i].store(v.unwrap_or(f64::NAN).to_bits(), Ordering::Relaxed);
    }

    fn gauge(&self, i: usize) -> Option<f64> {
        Some(f64::from_bits(self.gauges[i].load(Ordering::Relaxed))).filter(|v| !v.is_nan())
    }

    fn sample(&self) -> GpuSample {
        GpuSample {
            nvlink_tx_bytes: self.get(S_NVLINK_TX),
            nvlink_rx_bytes: self.get(S_NVLINK_RX),
            pcie_tx_bytes: self.get(S_PCIE_TX),
            pcie_rx_bytes: self.get(S_PCIE_RX),
            memory_busy_seconds: self.get(S_MEM_BUSY),
            sm_busy_seconds: self.get(S_SM_BUSY),
            seconds: self.get(S_SECONDS),
        }
    }
}

/// The sampler ([`HwCounters::start`]); stops when dropped.
#[derive(Debug)]
pub struct HwCounters {
    gpus: Arc<Vec<Gpu>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HwCounters {
    /// Starts sampling every `interval` the GPUs `CUDA_VISIBLE_DEVICES` names (all without
    /// it). `Err` when NVML is unavailable or sees none of them.
    pub fn start(interval: Duration) -> Result<Self, String> {
        let nvml = Nvml::init().map_err(|e| format!("NVML: {e}"))?;
        let count = nvml.device_count().map_err(|e| format!("NVML: {e}"))?;
        let uuids: Vec<String> = (0..count)
            .map(|i| {
                nvml.device_by_index(i)
                    .and_then(|d| d.uuid())
                    .unwrap_or_default()
            })
            .collect();
        let spec = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let indices = visible_gpus(spec.as_deref(), &uuids);
        if indices.is_empty() {
            return Err("no GPU visible to NVML".into());
        }
        let bus_ids: Vec<String> = (0..count)
            .map(|i| {
                nvml.device_by_index(i)
                    .and_then(|d| d.pci_info())
                    .map(|p| p.bus_id.to_ascii_lowercase())
                    .unwrap_or_default()
            })
            .collect();
        let gpus: Vec<Gpu> = indices
            .iter()
            .map(|&index| {
                let device = nvml.device_by_index(index).ok();
                let links = device.as_ref().map_or(0, |d| {
                    (0..MAX_NVLINKS)
                        .take_while(|&l| d.link_wrapper_for(l).is_active().is_ok())
                        .count() as u32
                });
                let info = device
                    .as_ref()
                    .map(|d| live::probe(&nvml, d, links, &bus_ids))
                    .unwrap_or_default();
                Gpu {
                    index,
                    links,
                    info,
                    sums: Default::default(),
                    gauges: std::array::from_fn(|_| AtomicU64::new(f64::NAN.to_bits())),
                }
            })
            .collect();
        let gpus = Arc::new(gpus);
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (gpus, stop) = (gpus.clone(), stop.clone());
            std::thread::Builder::new()
                .name("hw-counters".into())
                .spawn(move || sample_loop(&nvml, &gpus, &stop, interval))
                .map_err(|e| e.to_string())?
        };
        Ok(Self {
            gpus,
            stop,
            thread: Some(thread),
        })
    }

    /// `(NVML index, NVLink links)` per watched GPU.
    pub fn gpus(&self) -> Vec<(u32, u32)> {
        self.gpus.iter().map(|g| (g.index, g.links)).collect()
    }

    /// `(NVML index, counters since start)` per watched GPU.
    pub fn snapshot(&self) -> Vec<(u32, GpuSample)> {
        self.gpus.iter().map(|g| (g.index, g.sample())).collect()
    }

    /// Every watched GPU's state for the dashboard.
    pub fn live(&self) -> Vec<GpuLive> {
        self.gpus
            .iter()
            .map(|g| {
                let seen = |i| g.gauge(i).is_some_and(|v| v > 0.0);
                GpuLive {
                    index: g.index,
                    info: g.info.clone(),
                    totals: g.sample(),
                    pcie_counted: seen(G_PCIE_SEEN),
                    nvlink_cpu: (g.get(S_CPU_TX), g.get(S_CPU_RX)),
                    nvlink_peer: (g.get(S_PEER_TX), g.get(S_PEER_RX)),
                    c2c: seen(G_C2C_SEEN).then(|| (g.get(S_C2C_TX), g.get(S_C2C_RX))),
                    power_w: g.gauge(G_POWER_W),
                    sm_clock_mhz: g.gauge(G_SM_MHZ).map(|v| v as u32),
                    mem_clock_mhz: g.gauge(G_MEM_MHZ).map(|v| v as u32),
                    vram_used: g.gauge(G_VRAM_USED).map(|v| v as u64),
                    vram_total: g.gauge(G_VRAM_TOTAL).map(|v| v as u64),
                    mem_util_pct: g.gauge(G_MEM_UTIL),
                    sm_util_pct: g.gauge(G_SM_UTIL),
                    gpm_dram_pct: g.gauge(G_GPM_DRAM),
                }
            })
            .collect()
    }
}

impl Drop for HwCounters {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The NVML indices of the GPUs `spec` (`CUDA_VISIBLE_DEVICES`: indices or UUIDs, possibly
/// prefixes) names among `uuids` (NVML order), all of them without a spec. CUDA numbers GPUs
/// in PCI order like NVML when they are all the same model (`CUDA_DEVICE_ORDER` aside).
pub fn visible_gpus(spec: Option<&str>, uuids: &[String]) -> Vec<u32> {
    let Some(spec) = spec.map(str::trim) else {
        return (0..uuids.len() as u32).collect();
    };
    let mut out = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let found = match item.parse::<u32>() {
            Ok(i) => (i < uuids.len() as u32).then_some(i),
            Err(_) => uuids
                .iter()
                .position(|u| !u.is_empty() && u.starts_with(item))
                .map(|i| i as u32),
        };
        // CUDA stops at the first invalid entry.
        let Some(i) = found else { break };
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

/// A KiB counter's increase, `None` on a reset (the driver reloaded) or before a first value.
fn increase(last: &mut Option<u64>, now: Option<u64>) -> Option<u64> {
    let now = now?;
    let prev = last.replace(now)?;
    now.checked_sub(prev)
}

fn sample_loop(nvml: &Nvml, gpus: &[Gpu], stop: &AtomicBool, interval: Duration) {
    let devices: Vec<Option<Device<'_>>> = gpus
        .iter()
        .map(|g| nvml.device_by_index(g.index).ok())
        .collect();
    // Per GPU, per link: the last (sent, received) KiB counter values.
    let mut last_link: Vec<Vec<(Option<u64>, Option<u64>)>> = gpus
        .iter()
        .map(|g| vec![(None, None); g.links as usize])
        .collect();
    let mut gpm: Vec<Option<live::Gpm<'_>>> = gpus
        .iter()
        .map(|g| g.info.gpm.then(|| live::Gpm::new(nvml, g.info.has_c2c())))
        .collect();
    let mut last = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(interval);
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        for (((g, d), prev), gpm) in gpus.iter().zip(&devices).zip(&mut last_link).zip(&mut gpm) {
            let Some(d) = d else { continue };
            if let Some(per_link) = nvlink_kib(nvml, d, g.links) {
                for (l, ((tx, rx), prev)) in per_link.into_iter().zip(prev.iter_mut()).enumerate() {
                    let peer = g.info.nvlinks.get(l) == Some(&LinkClass::Peer);
                    let (stx, srx) = if peer {
                        (S_PEER_TX, S_PEER_RX)
                    } else {
                        (S_CPU_TX, S_CPU_RX)
                    };
                    if let Some(kib) = increase(&mut prev.0, tx) {
                        g.add(S_NVLINK_TX, kib as f64 * 1024.0);
                        g.add(stx, kib as f64 * 1024.0);
                    }
                    if let Some(kib) = increase(&mut prev.1, rx) {
                        g.add(S_NVLINK_RX, kib as f64 * 1024.0);
                        g.add(srx, kib as f64 * 1024.0);
                    }
                }
            }
            // KB/s over NVML's last sample period.
            if let Ok(kb_s) = d.pcie_throughput(PcieUtilCounter::Send) {
                g.add(S_PCIE_TX, f64::from(kb_s) * 1024.0 * dt);
                g.set_gauge(G_PCIE_SEEN, Some(1.0));
            }
            if let Ok(kb_s) = d.pcie_throughput(PcieUtilCounter::Receive) {
                g.add(S_PCIE_RX, f64::from(kb_s) * 1024.0 * dt);
            }
            match d.utilization_rates() {
                Ok(u) => {
                    g.add(S_MEM_BUSY, f64::from(u.memory) / 100.0 * dt);
                    g.add(S_SM_BUSY, f64::from(u.gpu) / 100.0 * dt);
                    g.set_gauge(G_MEM_UTIL, Some(f64::from(u.memory)));
                    g.set_gauge(G_SM_UTIL, Some(f64::from(u.gpu)));
                }
                Err(_) => {
                    g.set_gauge(G_MEM_UTIL, None);
                    g.set_gauge(G_SM_UTIL, None);
                }
            }
            g.add(S_SECONDS, dt);
            g.set_gauge(
                G_POWER_W,
                d.power_usage().ok().map(|mw| f64::from(mw) / 1e3),
            );
            g.set_gauge(G_SM_MHZ, d.clock_info(Clock::SM).ok().map(f64::from));
            g.set_gauge(G_MEM_MHZ, d.clock_info(Clock::Memory).ok().map(f64::from));
            let mem = (!g.info.unified).then(|| d.memory_info().ok()).flatten();
            g.set_gauge(G_VRAM_USED, mem.as_ref().map(|m| m.used as f64));
            g.set_gauge(G_VRAM_TOTAL, mem.as_ref().map(|m| m.total as f64));
            if let Some(gpm) = gpm {
                let f = gpm.tick(d);
                g.set_gauge(G_GPM_DRAM, f.and_then(|f| f.dram_pct));
                if let Some((tx, rx)) = f.and_then(|f| f.c2c_tx.zip(f.c2c_rx)) {
                    g.add(S_C2C_TX, tx * dt);
                    g.add(S_C2C_RX, rx * dt);
                    g.set_gauge(G_C2C_SEEN, Some(1.0));
                }
            }
        }
    }
}

/// NVLink data KiB `(sent, received)` since the driver loaded, per link (`None` for a link
/// NVML did not answer for): NVML's field values per link (`nvml-wrapper`'s field query asks
/// for link 0 only).
#[allow(clippy::type_complexity)]
fn nvlink_kib(nvml: &Nvml, d: &Device<'_>, links: u32) -> Option<Vec<(Option<u64>, Option<u64>)>> {
    use nvml_wrapper_sys::bindings::{
        nvmlFieldValue_t, nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_INT,
        nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG,
    };
    if links == 0 {
        return None;
    }
    let mut values: Vec<nvmlFieldValue_t> = (0..links)
        .flat_map(|l| {
            [NVLINK_DATA_TX, NVLINK_DATA_RX].map(|f| {
                // SAFETY: a plain C struct; all zeroes is a valid value.
                let mut v: nvmlFieldValue_t = unsafe { std::mem::zeroed() };
                v.fieldId = f;
                v.scopeId = l;
                v
            })
        })
        .collect();
    // SAFETY: `values` holds `len` initialized entries NVML fills in place; the handle is the
    // live device's.
    let ret = unsafe {
        nvml.lib()
            .nvmlDeviceGetFieldValues(d.handle(), values.len() as i32, values.as_mut_ptr())
    };
    if ret != 0 {
        return None;
    }
    let mut out = vec![(None, None); links as usize];
    let mut any = false;
    for v in &values {
        if v.nvmlReturn != 0 {
            continue;
        }
        // SAFETY: the union member `valueType` names.
        let n = unsafe {
            if v.valueType == nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_INT {
                u64::from(v.value.uiVal)
            } else if v.valueType == nvmlValueType_enum_NVML_VALUE_TYPE_UNSIGNED_LONG {
                // c_ulong: 64 bits on the Linux targets.
                v.value.ulVal
            } else {
                v.value.ullVal
            }
        };
        let Some(slot) = out.get_mut(v.scopeId as usize) else {
            continue;
        };
        any = true;
        if v.fieldId == NVLINK_DATA_TX {
            slot.0 = Some(n);
        } else {
            slot.1 = Some(n);
        }
    }
    any.then_some(out)
}

#[cfg(test)]
mod tests;
