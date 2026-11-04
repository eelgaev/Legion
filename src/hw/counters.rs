//! The GPUs' hardware counters while serving (#391, opt-in with `legion serve
//! --hw-counters`): a background thread samples NVML for each GPU the process sees (NVLink data
//! bytes per link, PCIe throughput, how busy the memory and the SMs were) and adds them up, so
//! `/metrics` and `legion bench --workload bandwidth` can put measured traffic next to the
//! analytic model bytes. NVML queries only: nothing runs on the GPUs, and without the flag no
//! thread starts.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use nvml_wrapper::enum_wrappers::device::PcieUtilCounter;
use nvml_wrapper::{Device, Nvml};

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

/// One watched GPU.
#[derive(Debug)]
struct Gpu {
    /// NVML index (PCI order).
    index: u32,
    /// Its NVLink links (0 without).
    links: u32,
    /// [`GpuSample`]'s fields as f64 bits.
    sums: [AtomicU64; 7],
}

impl Gpu {
    fn add(&self, i: usize, v: f64) {
        let cur = f64::from_bits(self.sums[i].load(Ordering::Relaxed));
        self.sums[i].store((cur + v).to_bits(), Ordering::Relaxed);
    }

    fn get(&self, i: usize) -> f64 {
        f64::from_bits(self.sums[i].load(Ordering::Relaxed))
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
        let gpus: Vec<Gpu> = indices
            .iter()
            .map(|&index| {
                let links = nvml.device_by_index(index).map_or(0, |d| {
                    (0..MAX_NVLINKS)
                        .take_while(|&l| d.link_wrapper_for(l).is_active().is_ok())
                        .count() as u32
                });
                Gpu {
                    index,
                    links,
                    sums: Default::default(),
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
        self.gpus
            .iter()
            .map(|g| {
                (
                    g.index,
                    GpuSample {
                        nvlink_tx_bytes: g.get(0),
                        nvlink_rx_bytes: g.get(1),
                        pcie_tx_bytes: g.get(2),
                        pcie_rx_bytes: g.get(3),
                        memory_busy_seconds: g.get(4),
                        sm_busy_seconds: g.get(5),
                        seconds: g.get(6),
                    },
                )
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
    let mut last_link: Vec<(Option<u64>, Option<u64>)> = vec![(None, None); gpus.len()];
    let mut last = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(interval);
        let dt = last.elapsed().as_secs_f64();
        last = Instant::now();
        for ((g, d), prev) in gpus.iter().zip(&devices).zip(&mut last_link) {
            let Some(d) = d else { continue };
            let (tx, rx) = nvlink_kib(nvml, d, g.links).unzip();
            if let Some(kib) = increase(&mut prev.0, tx) {
                g.add(0, kib as f64 * 1024.0);
            }
            if let Some(kib) = increase(&mut prev.1, rx) {
                g.add(1, kib as f64 * 1024.0);
            }
            // KB/s over NVML's last sample period.
            if let Ok(kb_s) = d.pcie_throughput(PcieUtilCounter::Send) {
                g.add(2, f64::from(kb_s) * 1024.0 * dt);
            }
            if let Ok(kb_s) = d.pcie_throughput(PcieUtilCounter::Receive) {
                g.add(3, f64::from(kb_s) * 1024.0 * dt);
            }
            if let Ok(u) = d.utilization_rates() {
                g.add(4, f64::from(u.memory) / 100.0 * dt);
                g.add(5, f64::from(u.gpu) / 100.0 * dt);
            }
            g.add(6, dt);
        }
    }
}

/// NVLink data KiB sent and received since the driver loaded, summed over `links` links:
/// NVML's field values per link (`nvml-wrapper`'s field query asks for link 0 only).
fn nvlink_kib(nvml: &Nvml, d: &Device<'_>, links: u32) -> Option<(u64, u64)> {
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
    let (mut tx, mut rx, mut any) = (0u64, 0u64, false);
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
        any = true;
        if v.fieldId == NVLINK_DATA_TX {
            tx += n;
        } else {
            rx += n;
        }
    }
    any.then_some((tx, rx))
}

#[cfg(test)]
mod tests;
