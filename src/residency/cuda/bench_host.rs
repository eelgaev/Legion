//! Pinned host memory for `legion cpu-bench pool --host-mem` (#37): the share pool's experts
//! placed as the server places them (the expert arena's `cuMemCreate` host memory, which the
//! driver maps for the CPU in 4 KiB pages) or in an anonymous mapping with transparent huge
//! pages registered with the driver (`cuMemHostRegister`), so the two can be timed side by
//! side. A measurement tool only: the server never allocates through here.

use super::*;

/// One pinned host range made by [`bench_host`]; freed on drop.
pub struct BenchHost {
    ptr: *mut u8,
    len: usize,
    kind: Kind,
}

enum Kind {
    /// cuMemCreate memory mapped at `ptr` (reserved range of `len` bytes).
    Arena(CUmemGenericAllocationHandle),
    /// An anonymous mapping of `len` bytes, registered while it lives.
    Thp(Option<HostRegistration>),
}

// SAFETY: plain memory; the bench hands its address to the pool's threads.
unsafe impl Send for BenchHost {}

impl BenchHost {
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for BenchHost {
    fn drop(&mut self) {
        match &mut self.kind {
            // SAFETY: mapped and reserved by `bench_host`, unmapped once.
            Kind::Arena(h) => unsafe {
                cuMemUnmap(self.ptr as CUdeviceptr, self.len);
                cuMemRelease(*h);
                cuMemAddressFree(self.ptr as CUdeviceptr, self.len);
            },
            Kind::Thp(reg) => {
                // Unregistered before the pages go.
                drop(reg.take());
                // SAFETY: the mapping `bench_host` made.
                unsafe { libc::munmap(self.ptr as *mut c_void, self.len) };
            }
        }
    }
}

/// `bytes` of pinned host memory for `device`: with `huge`, an anonymous mapping advised for
/// transparent huge pages (`MADV_HUGEPAGE`), touched, then registered with the driver
/// (portable, device-mapped); without, `cuMemCreate` on the device's host allocation node
/// with the CPU's and the device's access, as the expert arena maps its pieces.
pub fn bench_host(device: usize, bytes: usize, huge: bool) -> Result<BenchHost, String> {
    use_device(device)?;
    if huge {
        let len = round_up(bytes.max(1), ARENA_ALIGN);
        // SAFETY: a new private anonymous mapping.
        let p = unsafe {
            libc::mmap(
                null_mut(),
                len + ARENA_ALIGN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(format!("mmap of {} MiB failed", len >> 20));
        }
        // 2 MiB aligned, so every page of it can be a huge one; the ends go back.
        let start = round_up(p as usize, ARENA_ALIGN);
        // SAFETY: the unused head and tail of the mapping just made.
        unsafe {
            if start > p as usize {
                libc::munmap(p, start - p as usize);
            }
            let tail = (p as usize + len + ARENA_ALIGN) - (start + len);
            if tail > 0 {
                libc::munmap((start + len) as *mut c_void, tail);
            }
        }
        let mut b = BenchHost {
            ptr: start as *mut u8,
            len,
            kind: Kind::Thp(None),
        };
        // SAFETY: the aligned mapping; advice, then every page written once (faulted in).
        unsafe {
            libc::madvise(start as *mut c_void, len, libc::MADV_HUGEPAGE);
            std::ptr::write_bytes(b.ptr, 0, len);
        }
        let reg = register_host(device, &[(start, len)], false)?;
        b.kind = Kind::Thp(Some(reg));
        return Ok(b);
    }
    let node = alloc_node(device).ok_or("no pinned host memory through VMM for this GPU")?;
    let len = round_up(bytes.max(1), host_granularity(node).max(ARENA_ALIGN));
    let mut va: CUdeviceptr = 0;
    // SAFETY: writes the reserved address.
    cu(
        unsafe { cuMemAddressReserve(&mut va, len, ARENA_ALIGN, 0, 0) },
        "cuMemAddressReserve",
    )?;
    let prop = host_prop(node);
    let mut h = 0;
    // SAFETY: one allocation mapped into the range just reserved; on an error what was made
    // so far is released.
    unsafe {
        if let Err(e) = cu(cuMemCreate(&mut h, len, &prop, 0), "cuMemCreate(host)") {
            cuMemAddressFree(va, len);
            return Err(e);
        }
        if let Err(e) = cu(cuMemMap(va, len, 0, h, 0), "cuMemMap") {
            cuMemRelease(h);
            cuMemAddressFree(va, len);
            return Err(e);
        }
    }
    let b = BenchHost {
        ptr: va as *mut u8,
        len,
        kind: Kind::Arena(h),
    };
    let desc = [
        host_access(node),
        access(CU_MEM_LOCATION_TYPE_DEVICE, device as i32),
    ];
    // SAFETY: the range just mapped.
    cu(
        unsafe { cuMemSetAccess(va, len, desc.as_ptr(), desc.len()) },
        "cuMemSetAccess",
    )?;
    Ok(b)
}

/// The kernel page size and the huge-page bytes of the mapping holding `ptr` (from
/// `/proc/self/smaps`): `(KernelPageSize kB, AnonHugePages kB)`.
pub fn bench_host_pages(ptr: *const u8) -> Option<(u64, u64)> {
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let at = ptr as u64;
    let mut inside = false;
    let (mut kp, mut huge) = (None, 0);
    for line in smaps.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((a, b)) = range.split_once('-')
            && let (Ok(a), Ok(b)) = (u64::from_str_radix(a, 16), u64::from_str_radix(b, 16))
        {
            if inside {
                break;
            }
            inside = a <= at && at < b;
            continue;
        }
        if !inside {
            continue;
        }
        let num = |l: &str| {
            l.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
        };
        if line.starts_with("KernelPageSize:") {
            kp = num(line);
        } else if line.starts_with("AnonHugePages:") {
            huge = num(line).unwrap_or(0);
        }
    }
    kp.map(|k| (k, huge))
}
