//! Link and memory full scales: what a gauge's 100% is, from what the hardware reports.

/// PCIe usable bandwidth per lane and direction in GB/s for generation `gen` (after the line
/// encoding: 8b/10b for gen 1-2, 128b/130b from gen 3); `None` for an unknown generation.
pub fn pcie_lane_gb_s(generation: u32) -> Option<f64> {
    Some(match generation {
        1 => 0.25,
        2 => 0.5,
        3 => 0.985,
        4 => 1.969,
        5 => 3.938,
        6 => 7.563,
        _ => return None,
    })
}

/// A PCIe link's full scale per direction: max generation x max width.
pub fn pcie_gb_s(generation: u32, width: u32) -> Option<f64> {
    pcie_lane_gb_s(generation).map(|l| l * f64::from(width))
}

/// `PCIe 4.0 x16`.
pub fn pcie_label(generation: u32, width: u32) -> String {
    format!("PCIe {generation}.0 x{width}")
}

/// One NVLink link's bandwidth per direction in GB/s by NVLink version (NVML's
/// `nvmlDeviceGetNvLinkVersion`): 1.0 20, 2.0-4.0 25, 5.0 50.
pub fn nvlink_link_gb_s(version: u32) -> Option<f64> {
    Some(match version {
        1 => 20.0,
        2..=4 => 25.0,
        5 => 50.0,
        _ => return None,
    })
}

/// What an NVLink link goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkClass {
    /// The host CPU (NVML: an IBM NPU, the POWER9's NVLink endpoint).
    Cpu,
    /// Another GPU, directly or through an NVSwitch.
    Peer,
}

/// Classifies a link from NVML's remote device type (`gpu`, `cpu`, `switch`, `unknown`) and,
/// when that is unknown, from whether the remote PCI bus id is one of the GPUs'.
pub fn classify(remote: Option<&str>, remote_is_gpu: Option<bool>) -> LinkClass {
    match remote {
        Some("gpu") | Some("switch") => LinkClass::Peer,
        Some("cpu") => LinkClass::Cpu,
        _ => match remote_is_gpu {
            Some(true) => LinkClass::Peer,
            // An AC922's links that are not to a GPU go to the POWER9.
            _ => LinkClass::Cpu,
        },
    }
}

/// The full scale of `links` links of NVLink `version`.
pub fn nvlink_gb_s(links: u32, version: u32) -> Option<f64> {
    (links > 0)
        .then(|| nvlink_link_gb_s(version).map(|l| l * f64::from(links)))
        .flatten()
}

/// NVLink-C2C per direction: NVML's link count x max bandwidth per link (MB/s), else 450 GB/s
/// for the Grace parts NVIDIA quotes it for (GH200, GB200, GB300); `None` otherwise.
pub fn c2c_gb_s(links: Option<u32>, max_mb_s: Option<u32>, name: &str) -> (Option<f64>, String) {
    if let (Some(n), Some(mb)) = (links, max_mb_s)
        && n > 0
        && mb > 0
    {
        return (
            Some(f64::from(n) * f64::from(mb) / 1e3),
            format!("NVML: {n} C2C links x {mb} MB/s"),
        );
    }
    let upper = name.to_ascii_uppercase();
    if ["GH200", "GB200", "GB300"]
        .iter()
        .any(|p| upper.contains(p))
    {
        return (Some(450.0), "NVIDIA spec (450 GB/s per direction)".into());
    }
    (None, "not reported by NVML".into())
}

/// Device-memory peak in GB/s: bus width (bits) x the max memory clock (MHz, NVML reports
/// it so that x2 gives the transfer rate), else the known figure for the part.
pub fn memory_peak_gb_s(
    bus_bits: Option<u32>,
    max_mem_mhz: Option<u32>,
    name: &str,
) -> (Option<f64>, String) {
    let upper = name.to_ascii_uppercase();
    // GB10 (DGX Spark): LPDDR5X-8533, 256 bit; NVML reports no memory clock for it.
    let known = upper.contains("GB10").then_some(273.0);
    if let (Some(bits), Some(mhz)) = (bus_bits, max_mem_mhz)
        && bits > 0
        && mhz > 0
    {
        let gb_s = f64::from(bits) / 8.0 * f64::from(mhz) * 2.0 / 1e3;
        // A plausible figure only: some integrated parts report placeholder values.
        if (20.0..20_000.0).contains(&gb_s) && known.is_none() {
            return (Some(gb_s), format!("NVML: {bits}-bit x {mhz} MHz"));
        }
    }
    match known {
        Some(k) => (Some(k), "LPDDR5X spec (273 GB/s)".into()),
        None => (None, "not reported by NVML".into()),
    }
}

/// A NIC's speed in GB/s from sysfs text: `/sys/class/infiniband/*/ports/*/rate`
/// (`200 Gb/sec (4X HDR)`) or `/sys/class/net/*/speed` (Mb/s, `-1` when down).
pub fn nic_gb_s_from_rate(rate: &str) -> Option<f64> {
    let mut it = rate.split_whitespace();
    let n: f64 = it.next()?.parse().ok()?;
    match it.next() {
        Some(u) if u.starts_with("Gb") => Some(n / 8.0),
        Some(u) if u.starts_with("Mb") => Some(n / 8.0 / 1e3),
        // `speed` files carry a bare number of Mb/s.
        None if n > 0.0 => Some(n / 8.0 / 1e3),
        _ => None,
    }
}

/// The fastest active RDMA port (or network interface) on this machine, GB/s per direction.
pub fn fastest_nic() -> Option<(f64, String)> {
    let mut best: Option<(f64, String)> = None;
    fn consider(best: &mut Option<(f64, String)>, gb: Option<f64>, what: String) {
        if let Some(gb) = gb
            && best.as_ref().is_none_or(|(b, _)| gb > *b)
        {
            *best = Some((gb, what));
        }
    }
    if let Ok(devs) = std::fs::read_dir("/sys/class/infiniband") {
        for d in devs.flatten() {
            let ports = d.path().join("ports");
            for p in std::fs::read_dir(ports).into_iter().flatten().flatten() {
                let state = std::fs::read_to_string(p.path().join("state")).unwrap_or_default();
                if !state.contains("ACTIVE") {
                    continue;
                }
                let rate = std::fs::read_to_string(p.path().join("rate")).unwrap_or_default();
                consider(
                    &mut best,
                    nic_gb_s_from_rate(rate.trim()),
                    format!("{} {}", d.file_name().to_string_lossy(), rate.trim()),
                );
            }
        }
    }
    if best.is_none()
        && let Ok(devs) = std::fs::read_dir("/sys/class/net")
    {
        for d in devs.flatten() {
            let name = d.file_name().to_string_lossy().to_string();
            if name == "lo" {
                continue;
            }
            let speed = std::fs::read_to_string(d.path().join("speed")).unwrap_or_default();
            consider(
                &mut best,
                nic_gb_s_from_rate(speed.trim()),
                format!("{name} {} Mb/s", speed.trim()),
            );
        }
    }
    best
}
