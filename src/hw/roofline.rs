//! Measured bandwidth rooflines per device (#391): what a step's model bytes
//! ([`crate::ggml::model_bytes`]) are compared against. Each figure is a measurement on the
//! project's machines (microbenchmark runs), not a datasheet number;
//! `docs/bandwidth.md` says how to measure a new device.

/// One device's measured limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    /// The device, as named in reports.
    pub device: &'static str,
    /// Substrings of the GPU's name (its CUDA description) that select this entry.
    pub matches: &'static [&'static str],
    /// Device-memory read bandwidth, GB/s (10^9 bytes per second).
    pub memory_gb_s: f64,
    /// Host memory read in place over the host link per GPU, GB/s; 0 where host memory is the
    /// device memory (integrated GPUs).
    pub link_gb_s: f64,
    /// How the figures were measured.
    pub measured: &'static str,
}

/// The measured devices.
pub const TABLE: &[Roofline] = &[
    Roofline {
        device: "Tesla V100-SXM2 (POWER9 AC922, NVLink 2.0 to the host)",
        matches: &["V100"],
        memory_gb_s: 894.0,
        link_gb_s: 68.5,
        measured: "hostbw (2026-10-19, an AC922 with 4x V100): a read-only loop \
                   over 4 GiB of device memory, best of 5 at 8-16 CTAs per SM (893-894 GB/s on \
                   GPUs 0 and 2); pinned host memory on the GPU's own NUMA node read in place, \
                   68.5 GB/s per GPU at every CTA count and block size (two GPUs of one socket \
                   reach 133-135 GB/s together; the other socket's memory 46-56 GB/s)",
    },
    Roofline {
        device: "NVIDIA GB10 (DGX Spark, LPDDR5X shared with the CPU)",
        matches: &["GB10"],
        memory_gb_s: 233.0,
        link_gb_s: 0.0,
        measured: "mvbench (2026-10-19, a DGX Spark, Qwen3.8-Flash-Next decode-verify run): \
                   a copy loop \
                   over device memory, 233-234 GB/s; host and device memory are one pool, so \
                   there is no separate link",
    },
];

/// The entry for a GPU named `description` (e.g. `Tesla V100-SXM2-32GB`), if measured.
pub fn lookup(description: &str) -> Option<&'static Roofline> {
    TABLE
        .iter()
        .find(|r| r.matches.iter().any(|m| description.contains(m)))
}

/// One pipeline stage's rooflines (its GPUs' together under tensor parallelism), GB/s; 0 when
/// the device was not measured.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageRoofline {
    pub memory_gb_s: f64,
    pub link_gb_s: f64,
}

impl StageRoofline {
    /// The roofline of `gpus` GPUs named `description` running a stage together.
    pub fn of(description: &str, gpus: usize) -> Self {
        lookup(description).map_or_else(Self::default, |r| Self {
            memory_gb_s: r.memory_gb_s * gpus.max(1) as f64,
            link_gb_s: r.link_gb_s * gpus.max(1) as f64,
        })
    }
}

/// `gb_s` as a share of `roofline` (0..1, more when over it); `None` without a roofline.
pub fn share(gb_s: f64, roofline: f64) -> Option<f64> {
    (roofline > 0.0).then(|| gb_s / roofline)
}

#[cfg(test)]
mod tests;
