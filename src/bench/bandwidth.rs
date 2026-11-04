//! Memory bandwidth from a Legion server's `/metrics` (#391): the analytic model bytes its
//! steps had to read (weights, experts from VRAM and from host memory, staged expert copies,
//! KV) and, with `legion serve --hw-counters`, what the GPUs' hardware counters saw over the
//! same window. Two scrapes, before and after, give the window's figures.

use std::collections::HashMap;

use serde::Serialize;

/// One `/metrics` body: each sample's value by name and label set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scrape {
    samples: HashMap<String, f64>,
}

/// `name{a="x",b="y"}` with the labels sorted, so lookups don't depend on their order.
fn key(name: &str, labels: &[(&str, &str)]) -> String {
    let mut l: Vec<String> = labels.iter().map(|(k, v)| format!("{k}=\"{v}\"")).collect();
    l.sort();
    format!("{name}{{{}}}", l.join(","))
}

impl Scrape {
    /// Parses the Prometheus text format; comments and unparsable lines are skipped.
    pub fn parse(body: &str) -> Self {
        let mut samples = HashMap::new();
        for line in body.lines().filter(|l| !l.starts_with('#')) {
            let Some((series, value)) = line.rsplit_once(' ') else {
                continue;
            };
            let Ok(v) = value.parse::<f64>() else {
                continue;
            };
            let (name, labels) = match series.split_once('{') {
                Some((name, rest)) => (name, rest.trim_end_matches('}')),
                None => (series, ""),
            };
            let labels: Vec<(&str, &str)> = labels
                .split(',')
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.trim(), v.trim().trim_matches('"')))
                .collect();
            samples.insert(key(name, &labels), v);
        }
        Self { samples }
    }

    /// A sample's value, `None` when the server does not export it.
    pub fn get(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        self.samples.get(&key(name, labels)).copied()
    }

    /// The label values of `label` among `name`'s samples, sorted.
    fn values_of(&self, name: &str, label: &str) -> Vec<String> {
        let prefix = format!("{name}{{");
        let tag = format!("{label}=\"");
        let mut out: Vec<String> = self
            .samples
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix))
            .filter_map(|rest| rest.split_once(&tag))
            .filter_map(|(_, v)| v.split_once('"').map(|(v, _)| v.to_string()))
            .collect();
        out.sort_by_key(|v| (v.parse::<u64>().unwrap_or(u64::MAX), v.clone()));
        out.dedup();
        out
    }
}

/// One phase's model bytes over a window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct PhaseBytes {
    /// Model bytes the steps had to read, GB (required traffic, not measured): device memory
    /// and host memory together.
    pub gb: f64,
    /// Wall time of those steps, seconds.
    pub seconds: f64,
    /// `gb / seconds`.
    pub gb_s: f64,
    /// The part from device memory (dense weights, experts in VRAM, KV in VRAM), GB.
    pub device_gb: f64,
    /// The part over the host link: experts and KV read from host memory in place, plus
    /// staged expert copies, GB.
    pub link_gb: f64,
    /// Experts (VRAM and host; staged copies are read from VRAM, so they are already in) and
    /// KV (both tiers), GB.
    pub experts_gb: f64,
    pub kv_gb: f64,
    /// `device_gb / seconds` and `link_gb / seconds`.
    pub device_gb_s: f64,
    pub link_gb_s: f64,
    /// `device_gb_s` over the server's device-memory roofline, percent; `None` when the
    /// server does not know its device's roofline.
    pub roofline_pct: Option<f64>,
}

/// The model bytes over a window, by phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ModelBytes {
    pub prefill: PhaseBytes,
    pub decode: PhaseBytes,
    pub draft: PhaseBytes,
    /// The first pipeline stage's device-memory roofline the percentages use, GB/s (0:
    /// unknown).
    pub roofline_memory_gb_s: f64,
}

/// One GPU's hardware counters over a window (`legion serve --hw-counters`).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GpuCounters {
    /// NVML index.
    pub gpu: String,
    /// Seconds sampled in the window.
    pub seconds: f64,
    /// NVLink data received and sent (host-memory reads arrive as received data), GB.
    pub nvlink_rx_gb: f64,
    pub nvlink_tx_gb: f64,
    /// PCIe received and sent, GB.
    pub pcie_rx_gb: f64,
    pub pcie_tx_gb: f64,
    /// `nvlink_rx_gb / seconds`.
    pub nvlink_rx_gb_s: f64,
    /// Share of the window the memory was busy and a kernel ran, percent (NVML utilization:
    /// time, not bytes).
    pub memory_busy_pct: f64,
    pub sm_busy_pct: f64,
}

fn rate(gb: f64, seconds: f64) -> f64 {
    if seconds > 0.0 { gb / seconds } else { 0.0 }
}

/// A phase's model bytes between two scrapes.
fn phase_between(before: &Scrape, after: &Scrape, phase: &str, roofline: f64) -> PhaseBytes {
    let d = |name: &str, extra: &[(&str, &str)]| {
        let mut labels = vec![("phase", phase)];
        labels.extend_from_slice(extra);
        after.get(name, &labels).unwrap_or(0.0) - before.get(name, &labels).unwrap_or(0.0)
    };
    let kind = |k: &str| d("legion_engine_model_kind_bytes", &[("kind", k)]) / 1e9;
    let gb = d("legion_engine_model_bytes", &[]) / 1e9;
    let seconds = d("legion_engine_model_seconds", &[]);
    let (dense, upper) = (kind("dense"), kind("experts_upper"));
    let (exp_dev, exp_host, staged) =
        (kind("experts_device"), kind("experts_host"), kind("staged"));
    let (kv, kv_host) = (kind("kv"), kind("kv_host"));
    // A server without the per-kind families counted only weights from device memory.
    let device_gb = if after
        .get(
            "legion_engine_model_kind_bytes",
            &[("phase", phase), ("kind", "dense")],
        )
        .is_some()
    {
        dense + upper + exp_dev + kv
    } else {
        gb
    };
    let link_gb = exp_host + kv_host + staged;
    let device_gb_s = rate(device_gb, seconds);
    PhaseBytes {
        gb,
        seconds,
        gb_s: rate(gb, seconds),
        device_gb,
        link_gb,
        experts_gb: upper + exp_dev + exp_host,
        kv_gb: kv + kv_host,
        device_gb_s,
        link_gb_s: rate(link_gb, seconds),
        roofline_pct: (roofline > 0.0).then(|| device_gb_s / roofline * 100.0),
    }
}

/// The model bytes between two scrapes; `None` when the server exports none.
pub fn model_bytes_between(before: &Scrape, after: &Scrape) -> Option<ModelBytes> {
    after.get("legion_engine_model_bytes", &[("phase", "decode")])?;
    let roofline = after
        .get(
            "legion_engine_roofline_memory_gb_per_second",
            &[("stage", "0")],
        )
        .unwrap_or(0.0);
    Some(ModelBytes {
        prefill: phase_between(before, after, "prefill", roofline),
        decode: phase_between(before, after, "decode", roofline),
        draft: phase_between(before, after, "draft", roofline),
        roofline_memory_gb_s: roofline,
    })
}

/// Each GPU's hardware counters between two scrapes; empty without `--hw-counters`.
pub fn gpu_counters_between(before: &Scrape, after: &Scrape) -> Vec<GpuCounters> {
    after
        .values_of("legion_gpu_sampled_seconds", "gpu")
        .into_iter()
        .map(|gpu| {
            let l = [("gpu", gpu.as_str())];
            let d = |name: &str| {
                after.get(name, &l).unwrap_or(0.0) - before.get(name, &l).unwrap_or(0.0)
            };
            let seconds = d("legion_gpu_sampled_seconds");
            let busy = |name: &str| rate(d(name), seconds) * 100.0;
            let nvlink_rx_gb = d("legion_gpu_nvlink_rx_bytes") / 1e9;
            GpuCounters {
                seconds,
                nvlink_rx_gb,
                nvlink_tx_gb: d("legion_gpu_nvlink_tx_bytes") / 1e9,
                pcie_rx_gb: d("legion_gpu_pcie_rx_bytes") / 1e9,
                pcie_tx_gb: d("legion_gpu_pcie_tx_bytes") / 1e9,
                nvlink_rx_gb_s: rate(nvlink_rx_gb, seconds),
                memory_busy_pct: busy("legion_gpu_memory_busy_seconds"),
                sm_busy_pct: busy("legion_gpu_sm_busy_seconds"),
                gpu,
            }
        })
        .collect()
}

/// Scrapes `{url}/metrics` (`url` may end in `/v1`) with the server's API key, if it has one;
/// `None` when it fails.
pub async fn scrape(client: &reqwest::Client, url: &str, api_key: Option<&str>) -> Option<Scrape> {
    let base = url.trim_end_matches('/');
    let base = base.strip_suffix("/v1").unwrap_or(base);
    let mut req = client.get(format!("{base}/metrics"));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    Some(Scrape::parse(&resp.text().await.ok()?))
}

/// Renders one phase's analytic figures next to the GPUs' counters, as table rows.
pub fn format_phase(name: &str, p: &PhaseBytes, gpus: &[GpuCounters]) -> String {
    let pct = p
        .roofline_pct
        .map_or_else(|| "-".to_string(), |v| format!("{v:.0}%"));
    let mut out = format!(
        "{name:<8} {:>8.3} {:>9.2} {:>8.2} {:>8.2} {:>7.2} {:>11.1} {:>9} {:>9.1}\n",
        p.seconds, p.gb, p.experts_gb, p.kv_gb, p.link_gb, p.device_gb_s, pct, p.link_gb_s
    );
    for g in gpus {
        out.push_str(&format!(
            "  GPU {:<3} mem busy {:>5.1}%  SM busy {:>5.1}%  NVLink rx {:>7.2} GB ({:>6.1} GB/s) \
             tx {:>7.2} GB  PCIe rx {:>6.2} GB tx {:>6.2} GB\n",
            g.gpu,
            g.memory_busy_pct,
            g.sm_busy_pct,
            g.nvlink_rx_gb,
            g.nvlink_rx_gb_s,
            g.nvlink_tx_gb,
            g.pcie_rx_gb,
            g.pcie_tx_gb,
        ));
    }
    out
}

/// The header [`format_phase`]'s rows go under.
pub fn table_header() -> String {
    format!(
        "{:<8} {:>8} {:>9} {:>8} {:>8} {:>7} {:>11} {:>9} {:>9}\n",
        "phase",
        "model_s",
        "model_GB",
        "exp_GB",
        "kv_GB",
        "link_GB",
        "device_GB/s",
        "roofline",
        "link_GB/s"
    )
}

#[cfg(test)]
mod tests;
