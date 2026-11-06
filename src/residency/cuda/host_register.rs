//! Experts read in place from plain host memory on GPUs that cannot read pageable memory (no
//! ATS or HMM: an x86 box with V100s on PCIe, an older kernel or the proprietary driver): the
//! expert ranges are registered with the driver (`cuMemHostRegister`, device-mapped), so the
//! kernels read them through the same addresses, as an AC922 reads them through ATS. The
//! choice itself is [`fallback::host_expert_read`]'s.

use super::*;

/// Host ranges registered with the driver for the GPUs to read ([`register_host`]),
/// unregistered on drop: before the memory they cover goes away.
pub struct HostRegistration {
    /// Each registered span's first byte.
    spans: Vec<usize>,
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        for &p in &self.spans {
            // SAFETY: a span registered by `register_host` and not unregistered since.
            unsafe { cuMemHostUnregister(p as *mut c_void) };
        }
    }
}

/// The host page size (the unit `cuMemHostRegister` pins).
fn page_size() -> usize {
    // SAFETY: a plain query.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 { p as usize } else { 4096 }
}

/// Registers the byte ranges `(start, len)` of host memory with the driver
/// (`cuMemHostRegister`, portable and device-mapped; read-only with `read_only`, so a private
/// file mapping's pages are pinned as they are, not copied), from `device` (made current), and
/// checks that the GPUs read them at their host addresses: the kernels take the tensors' host
/// pointers. Ranges are widened to whole pages and merged ([`fallback::page_spans`]). On an
/// error, what was registered is unregistered again.
pub fn register_host(
    device: usize,
    ranges: &[(usize, usize)],
    read_only: bool,
) -> Result<HostRegistration, String> {
    use_device(device)?;
    let mut flags = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
    if read_only {
        flags |= CU_MEMHOSTREGISTER_READ_ONLY;
    }
    let mut reg = HostRegistration { spans: Vec::new() };
    for (start, len) in fallback::page_spans(ranges, page_size()) {
        let p = start as *mut c_void;
        // SAFETY: `start..start + len` covers whole pages of live host memory (the tensors'
        // own bytes, widened to their pages, which belong to the same mapping or buffer).
        let r = unsafe { cuMemHostRegister(p, len, flags) };
        cu(r, &format!("cuMemHostRegister of {} MiB", len >> 20))?;
        reg.spans.push(start);
        let mut d: CUdeviceptr = 0;
        // SAFETY: writes one device pointer for a registered range.
        cu(
            unsafe { cuMemHostGetDevicePointer(&mut d, p, 0) },
            "cuMemHostGetDevicePointer",
        )?;
        if d as usize != start {
            return Err(format!(
                "the GPU maps registered host memory at another address ({d:#x} for {start:#x}), \
                 and the kernels read experts at their host addresses"
            ));
        }
    }
    Ok(reg)
}
