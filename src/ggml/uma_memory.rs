//! Free memory of a CUDA GPU that shares the host's memory (the GB10), as ggml-cuda reports it,
//! without its `cudaGetDeviceProperties` per query (#827).
//!
//! ggml-cuda's `ggml_backend_cuda_device_get_memory` asks `cudaGetDeviceProperties` on every
//! call, only to learn whether the device is integrated, and then reads `/proc/meminfo`. On a
//! GB10 that call took 12.4 ms (nsys, 2026-10-31), and a request's first graph allocation makes
//! one, with the GPU idle. [`crate::ggml::Device::memory`] decides "integrated" once per device
//! and reads the same numbers itself: `cudaMemGetInfo`, then the free bytes from
//! `/proc/meminfo` by [`free_bytes`], which parses the text as ggml's
//! `ggml_backend_cuda_get_available_uma_memory` does (MIT, ggml `src/ggml-cuda/ggml-cuda.cu`).
// Read on CUDA only; the CPU build keeps it for the tests.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(test)]
mod tests;

/// The bytes of `/proc/meminfo` ggml reads (its 2 KiB buffer, NUL-terminated).
const READ_MAX: usize = 2047;

/// The free bytes ggml reports for a UMA device given `/proc/meminfo`'s text: the free huge
/// pages when the kernel has a huge page pool (`HugePages_Total` neither 0 nor missing), else
/// `MemAvailable`; `None` where ggml keeps `cudaMemGetInfo`'s free bytes (the text has no
/// positive value of these).
pub(crate) fn free_bytes(meminfo: &[u8]) -> Option<usize> {
    let text = &meminfo[..meminfo.len().min(READ_MAX)];
    if text.is_empty() {
        return None;
    }
    let mut available: i64 = -1;
    let (mut huge_total, mut huge_free, mut huge_size): (i64, i64, i64) = (-1, -1, -1);
    for line in text.split(|&b| b == b'\n') {
        if let Some(v) = field(line, b"MemAvailable:") {
            available = v;
        } else if field(line, b"SwapFree:").is_some() {
            // Read by ggml, unused for the free bytes.
        } else if let Some(v) = field(line, b"HugePages_Total:") {
            huge_total = v;
        } else if let Some(v) = field(line, b"HugePages_Free:") {
            huge_free = v;
        } else if let Some(v) = field(line, b"Hugepagesize:") {
            huge_size = v;
        }
    }
    if huge_total != 0 && huge_total != -1 {
        available = huge_free.wrapping_mul(huge_size);
    }
    (available > 0).then(|| (available as usize).wrapping_mul(1024))
}

/// `sscanf(line, "<name> %ld", &v) == 1`: the line starts with `name`, then (after any
/// whitespace) a decimal integer with an optional sign.
fn field(line: &[u8], name: &[u8]) -> Option<i64> {
    let rest = line.strip_prefix(name)?;
    let rest = &rest[rest.iter().take_while(|b| b.is_ascii_whitespace()).count()..];
    let (neg, digits) = match rest.first() {
        Some(b'-') => (true, &rest[1..]),
        Some(b'+') => (false, &rest[1..]),
        _ => (false, rest),
    };
    let n = digits.iter().take_while(|b| b.is_ascii_digit()).count();
    if n == 0 {
        return None;
    }
    let v = digits[..n].iter().fold(0i64, |v, &d| {
        v.saturating_mul(10).saturating_add(i64::from(d - b'0'))
    });
    Some(if neg { -v } else { v })
}
