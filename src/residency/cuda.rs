//! The CUDA side of the residency layer, in Rust over the driver and runtime APIs: host-NUMA
//! arenas and tiered ranges as ggml buffer types, the DMA copier, and the bandwidth probe.
//!
//! Host memory comes from CUDA's virtual memory management (`cuMemCreate` with a host NUMA
//! node as the location): pinned, placed on the node asked for whatever thread allocates it,
//! and mappable at any virtual address, more than once. Measured on an AC922 (V100, NVLink 2.0
//! with ATS, GPU on node 8): kernels read it in place at 68.7 GB/s and DMA copies it at
//! 67.8 GB/s, the same as `cudaHostAlloc` memory. The node rules (memory local to the GPU's
//! socket, written by that socket's CPUs) follow Strata-AC922's measurements (MIT,
//! eelgaev/Strata-AC922, its pinned arenas).
//!
//! The arena and the tiered ranges are ggml buffer types implemented here (ggml's
//! `ggml-backend-impl.h` interfaces through bindgen), so ggml allocates tensors in them, the
//! loader fills them like any host buffer, and graphs read them like device weights.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr::{NonNull, null_mut};
use std::sync::Mutex;

use super::fallback::{self, HostLoc, NO_NODE};
use crate::ggml::{BufferType, Context, GgmlError, Tensor, ffi};

mod host_register;
pub use host_register::{HostRegistration, register_host};
mod kv_vmm;
pub use kv_vmm::{kv_vmm_host_ok, kv_vmm_remap, kv_vmm_type, kv_vmm_unmap};
mod tp_tier;
pub use tp_tier::{
    LAZY_CHUNK_ROWS, tp_tier_back, tp_tier_devices, tp_tier_granularity, tp_tier_lend, tp_tier_trim,
};

/// The driver and runtime API calls used here (`libcuda` and `libcudart` are linked with the
/// `cuda` feature), with the struct layouts of CUDA 12's `cuda.h`.
#[allow(non_camel_case_types, non_snake_case)]
mod sys {
    use std::ffi::{c_char, c_int, c_uint, c_void};

    pub type CUresult = c_int;
    pub type CUdevice = c_int;
    pub type CUdeviceptr = u64;
    pub type CUmemGenericAllocationHandle = u64;
    pub type cudaError_t = c_int;
    pub type cudaStream_t = *mut c_void;
    pub type cudaEvent_t = *mut c_void;

    pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
    pub const CU_MEM_ALLOCATION_TYPE_PINNED: c_uint = 1;
    pub const CU_MEM_LOCATION_TYPE_DEVICE: c_uint = 1;
    pub const CU_MEM_LOCATION_TYPE_HOST: c_uint = 2;
    pub const CU_MEM_LOCATION_TYPE_HOST_NUMA: c_uint = 3;
    pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: c_uint = 3;
    pub const CU_MEM_ALLOC_GRANULARITY_MINIMUM: c_uint = 0;
    pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_uint = 16;
    pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_uint = 18;
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_uint = 75;
    pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_uint = 76;
    pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: c_uint = 88;
    pub const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: c_uint = 91;
    pub const CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED: c_uint = 99;
    pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: c_uint = 100;
    pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: c_uint = 102;
    pub const CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED: c_uint = 113;
    pub const CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID: c_uint = 134;
    pub const CU_MEMHOSTREGISTER_PORTABLE: c_uint = 0x01;
    pub const CU_MEMHOSTREGISTER_DEVICEMAP: c_uint = 0x02;
    pub const CU_MEMHOSTREGISTER_READ_ONLY: c_uint = 0x08;
    pub const CUDA_MEMCPY_DEFAULT: c_int = 4;
    pub const CUDA_STREAM_NON_BLOCKING: c_uint = 1;
    pub const CUDA_EVENT_DISABLE_TIMING: c_uint = 2;
    pub const CUDA_ERROR_NOT_READY: cudaError_t = 600;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct CUmemLocation {
        pub type_: c_uint,
        pub id: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct CUmemAllocFlags {
        pub compressionType: u8,
        pub gpuDirectRDMACapable: u8,
        pub usage: u16,
        pub reserved: [u8; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CUmemAllocationProp {
        pub type_: c_uint,
        pub requestedHandleTypes: c_uint,
        pub location: CUmemLocation,
        pub win32HandleMetaData: *mut c_void,
        pub allocFlags: CUmemAllocFlags,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CUmemAccessDesc {
        pub location: CUmemLocation,
        pub flags: c_uint,
    }

    unsafe extern "C" {
        pub fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
        pub fn cuInit(flags: c_uint) -> CUresult;
        pub fn cuDeviceGetPCIBusId(id: *mut c_char, len: c_int, d: CUdevice) -> CUresult;
        pub fn cuDeviceGet(d: *mut CUdevice, ordinal: c_int) -> CUresult;
        pub fn cuDeviceGetCount(n: *mut c_int) -> CUresult;
        pub fn cuDeviceGetAttribute(v: *mut c_int, attr: c_uint, d: CUdevice) -> CUresult;
        pub fn cuMemGetAllocationGranularity(
            g: *mut usize,
            prop: *const CUmemAllocationProp,
            option: c_uint,
        ) -> CUresult;
        pub fn cuMemCreate(
            h: *mut CUmemGenericAllocationHandle,
            size: usize,
            prop: *const CUmemAllocationProp,
            flags: u64,
        ) -> CUresult;
        pub fn cuMemRelease(h: CUmemGenericAllocationHandle) -> CUresult;
        pub fn cuMemAddressReserve(
            p: *mut CUdeviceptr,
            size: usize,
            align: usize,
            addr: CUdeviceptr,
            flags: u64,
        ) -> CUresult;
        pub fn cuMemAddressFree(p: CUdeviceptr, size: usize) -> CUresult;
        pub fn cuMemMap(
            p: CUdeviceptr,
            size: usize,
            offset: usize,
            h: CUmemGenericAllocationHandle,
            flags: u64,
        ) -> CUresult;
        pub fn cuMemUnmap(p: CUdeviceptr, size: usize) -> CUresult;
        pub fn cuMemSetAccess(
            p: CUdeviceptr,
            size: usize,
            desc: *const CUmemAccessDesc,
            count: usize,
        ) -> CUresult;
        // `cuda.h` maps both names to their `_v2` symbols.
        #[link_name = "cuMemHostRegister_v2"]
        pub fn cuMemHostRegister(p: *mut c_void, size: usize, flags: c_uint) -> CUresult;
        pub fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
        #[link_name = "cuMemHostGetDevicePointer_v2"]
        pub fn cuMemHostGetDevicePointer(
            d: *mut CUdeviceptr,
            p: *mut c_void,
            flags: c_uint,
        ) -> CUresult;

        pub fn cudaGetErrorString(e: cudaError_t) -> *const c_char;
        pub fn cudaSetDevice(d: c_int) -> cudaError_t;
        pub fn cudaGetDevice(d: *mut c_int) -> cudaError_t;
        pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
        pub fn cudaFree(p: *mut c_void) -> cudaError_t;
        pub fn cudaDeviceSynchronize() -> cudaError_t;
        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
        ) -> cudaError_t;
        pub fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            n: usize,
            kind: c_int,
            s: cudaStream_t,
        ) -> cudaError_t;
        pub fn cudaMemcpy2D(
            dst: *mut c_void,
            dpitch: usize,
            src: *const c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: c_int,
        ) -> cudaError_t;
        pub fn cudaMemset(p: *mut c_void, v: c_int, n: usize) -> cudaError_t;
        pub fn cudaStreamCreateWithFlags(s: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
        pub fn cudaStreamDestroy(s: cudaStream_t) -> cudaError_t;
        pub fn cudaStreamSynchronize(s: cudaStream_t) -> cudaError_t;
        pub fn cudaEventCreateWithFlags(e: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
        pub fn cudaEventRecord(e: cudaEvent_t, s: cudaStream_t) -> cudaError_t;
        pub fn cudaEventQuery(e: cudaEvent_t) -> cudaError_t;
        pub fn cudaEventSynchronize(e: cudaEvent_t) -> cudaError_t;
        pub fn cudaEventDestroy(e: cudaEvent_t) -> cudaError_t;
        pub fn cudaStreamWaitEvent(s: cudaStream_t, e: cudaEvent_t, flags: c_uint) -> cudaError_t;
        pub fn cudaMemsetAsync(p: *mut c_void, v: c_int, n: usize, s: cudaStream_t) -> cudaError_t;
    }
}

use sys::*;

mod spill;
pub use spill::{spill_devices, spill_type, spill_type_keeping};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// Set while [`free_staged`] drops ranges it synchronized the device for.
    static SYNCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn set_error(e: String) {
    LAST_ERROR.with(|l| *l.borrow_mut() = e);
}

fn last_error() -> String {
    LAST_ERROR.with(|l| l.borrow().clone())
}

fn cu(r: CUresult, what: &str) -> Result<(), String> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    // SAFETY: writes a pointer to a static string (or leaves it null).
    unsafe { cuGetErrorString(r, &mut s) };
    let msg = if s.is_null() {
        format!("CUDA driver error {r}")
    } else {
        // SAFETY: a static NUL-terminated string.
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    };
    Err(format!("{what}: {msg}"))
}

fn rt(e: cudaError_t, what: &str) -> Result<(), String> {
    if e == 0 {
        return Ok(());
    }
    // SAFETY: a static NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(cudaGetErrorString(e)) }.to_string_lossy();
    Err(format!("{what}: {msg}"))
}

/// Makes `device` current on this thread, with its primary context (driver calls need one).
fn use_device(device: usize) -> Result<(), String> {
    // SAFETY: plain runtime calls.
    unsafe {
        rt(cudaSetDevice(device as c_int), "cudaSetDevice")?;
        rt(cudaFree(null_mut()), "context init")
    }
}

/// Memory calls at least this slow are logged (debug). #450: they take milliseconds, but each
/// waits for the driver's global lock, which another process pinning host memory (a model
/// loading) holds for seconds; every call of ours then takes tens of milliseconds.
const SLOW_CALL_MS: f64 = 50.0;

/// Runs `f`, a driver memory call `what` on `device` over `bytes`, and logs it if slow.
fn timed<T>(what: &'static str, device: usize, bytes: usize, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let r = f();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    if ms >= SLOW_CALL_MS {
        tracing::debug!(
            call = what,
            device,
            mib = bytes >> 20,
            ms,
            "slow CUDA memory call"
        );
    }
    r
}

fn round_up(x: usize, to: usize) -> usize {
    x.div_ceil(to) * to
}

fn granularity(location_type: u32, id: i32) -> usize {
    let prop = alloc_prop(location_type, id);
    let mut g = 0usize;
    // SAFETY: reads `prop`, writes `g`.
    let r =
        unsafe { cuMemGetAllocationGranularity(&mut g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) };
    if r == 0 { g } else { 0 }
}

fn alloc_prop(location_type: u32, id: i32) -> CUmemAllocationProp {
    CUmemAllocationProp {
        type_: CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: 0,
        location: CUmemLocation {
            type_: location_type,
            id,
        },
        win32HandleMetaData: null_mut(),
        allocFlags: CUmemAllocFlags::default(),
    }
}

fn access(location_type: u32, id: i32) -> CUmemAccessDesc {
    CUmemAccessDesc {
        location: CUmemLocation {
            type_: location_type,
            id,
        },
        flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    }
}

/// The driver location of host allocator node `node` ([`HostLoc::of`]): `HOST_NUMA` on a real
/// node, as always; `HOST` for [`NO_NODE`] (a GPU with no host NUMA node).
fn host_location(node: u32) -> (c_uint, c_int) {
    match HostLoc::of(node) {
        HostLoc::Numa(n) => (CU_MEM_LOCATION_TYPE_HOST_NUMA, n as c_int),
        HostLoc::Host => (CU_MEM_LOCATION_TYPE_HOST, 0),
    }
}

/// [`granularity`] of pinned host memory on allocator node `node` ([`host_location`]).
fn host_granularity(node: u32) -> usize {
    let (ty, id) = host_location(node);
    granularity(ty, id)
}

/// [`alloc_prop`] of pinned host memory on allocator node `node` ([`host_location`]).
fn host_prop(node: u32) -> CUmemAllocationProp {
    let (ty, id) = host_location(node);
    alloc_prop(ty, id)
}

/// The CPU's [`access`] to pinned host memory on allocator node `node` ([`host_location`]).
fn host_access(node: u32) -> CUmemAccessDesc {
    let (ty, id) = host_location(node);
    access(ty, id)
}

/// Whether the driver allocates, maps and lets `device` and the CPU access pinned host memory
/// of location type `HOST` (no NUMA node; CUDA 12.2+ drivers): one granule tried for real, as
/// the granularity query alone can pass where `cuMemCreate` then refuses. `device` is current.
fn host_type_works(device: usize) -> bool {
    let g = granularity(CU_MEM_LOCATION_TYPE_HOST, 0);
    if g == 0 {
        return false;
    }
    let prop = alloc_prop(CU_MEM_LOCATION_TYPE_HOST, 0);
    let (mut h, mut va) = (0, 0);
    // SAFETY: one allocation in a range of its own, all released before returning.
    unsafe {
        if cuMemCreate(&mut h, g, &prop, 0) != 0 {
            return false;
        }
        let mut ok = cuMemAddressReserve(&mut va, g, g, 0, 0) == 0;
        let mapped = ok && cuMemMap(va, g, 0, h, 0) == 0;
        if mapped {
            let desc = [
                access(CU_MEM_LOCATION_TYPE_HOST, 0),
                access(CU_MEM_LOCATION_TYPE_DEVICE, device as c_int),
            ];
            ok = cuMemSetAccess(va, g, desc.as_ptr(), desc.len()) == 0;
            cuMemUnmap(va, g);
        } else {
            ok = false;
        }
        if va != 0 {
            cuMemAddressFree(va, g);
        }
        cuMemRelease(h);
        ok
    }
}

/// Per CUDA device without a known host NUMA node, the node its pinned host memory uses
/// ([`no_node_host`]).
static NO_NODE_HOST: Mutex<Vec<(usize, Option<u32>)>> = Mutex::new(Vec::new());

/// The allocator node for pinned host memory of `device`, a GPU whose host NUMA node is unknown
/// ([`fallback::host_node`] with no node): [`NO_NODE`] where the driver takes
/// `CU_MEM_LOCATION_TYPE_HOST`, else node 0 where it takes `HOST_NUMA` 0. Probed once per
/// device and said once in the log; the current device is left as it was.
fn no_node_host(device: usize) -> Option<u32> {
    if let Some(&(_, n)) = NO_NODE_HOST
        .lock()
        .unwrap()
        .iter()
        .find(|&&(d, _)| d == device)
    {
        return n;
    }
    let mut current: c_int = -1;
    // SAFETY: writes one int.
    let had = unsafe { cudaGetDevice(&mut current) } == 0;
    let chosen = use_device(device).ok().and_then(|()| {
        fallback::host_node(
            None,
            |n| host_granularity(n) > 0,
            || host_type_works(device),
        )
    });
    if had && current >= 0 {
        // SAFETY: a plain runtime call.
        unsafe { cudaSetDevice(current) };
    }
    let mut list = NO_NODE_HOST.lock().unwrap();
    if !list.iter().any(|&(d, _)| d == device) {
        list.push((device, chosen));
        tracing::info!(
            device,
            location = match chosen.map(HostLoc::of) {
                Some(HostLoc::Host) => "CU_MEM_LOCATION_TYPE_HOST",
                Some(HostLoc::Numa(_)) => "CU_MEM_LOCATION_TYPE_HOST_NUMA 0",
                None => "none",
            },
            "{}",
            fallback::host_node_note(chosen)
        );
    }
    chosen
}

/// The allocator node for pinned host memory of CUDA device `device`: its host NUMA node
/// ([`device_numa_node`]), as always; with no node known, [`no_node_host`]'s choice. `None`:
/// no node and no pinned host memory through VMM for it.
fn alloc_node(device: usize) -> Option<u32> {
    device_numa_node(device).or_else(|| no_node_host(device))
}

/// Per CUDA device, the peer that stages a prompt's experts into it ([`set_stage_peer`]).
static STAGE_PEERS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// From now on `device`'s staged ranges ([`staged`] and the like) are writable by `peer` too,
/// which copies experts into them (`--prefill-peer-stage`).
pub fn set_stage_peer(device: usize, peer: usize) {
    let mut p = STAGE_PEERS.lock().unwrap();
    p.retain(|&(d, _)| d != device);
    p.push((device, peer));
}

/// `device`'s staging peer ([`set_stage_peer`]).
pub fn stage_peer(device: usize) -> Option<usize> {
    STAGE_PEERS
        .lock()
        .unwrap()
        .iter()
        .find(|&&(d, _)| d == device)
        .map(|&(_, p)| p)
}

/// The access a staged range of `device` grants: the device, and its staging peer if any.
fn range_access(device: usize) -> Vec<CUmemAccessDesc> {
    let mut a = vec![access(CU_MEM_LOCATION_TYPE_DEVICE, device as i32)];
    if let Some(p) = stage_peer(device) {
        a.push(access(CU_MEM_LOCATION_TYPE_DEVICE, p as i32));
    }
    a
}

/// What a CUDA device supports for the residency layer.
#[derive(Debug, Default, Clone, Copy)]
pub struct Caps {
    pub vmm: bool,
    /// The host NUMA node closest to the device (-1 unknown).
    pub host_numa_id: i32,
    pub pageable_access: bool,
    pub pageable_uses_host_page_tables: bool,
    /// Host memory can be registered with the driver (`cuMemHostRegister`).
    pub host_register: bool,
    /// ... read-only (`CU_MEMHOSTREGISTER_READ_ONLY`).
    pub read_only_register: bool,
    /// Kernels read registered host memory at its host address.
    pub host_pointer_registered: bool,
    pub integrated: bool,
    /// The compute capability's major version (7: Volta/Turing, 8: Ampere/Ada, 12: GB10).
    pub cc_major: i32,
    /// The compute capability's minor version.
    pub cc_minor: i32,
    /// Streaming multiprocessors.
    pub sms: i32,
    /// Pinned host-NUMA memory can be allocated through VMM.
    pub host_numa_vmm: bool,
    /// The node the host allocators place this device's pinned memory on: `host_numa_id` where
    /// it is known and the driver places memory there; with no node known,
    /// [`fallback::NO_NODE`] (`CU_MEM_LOCATION_TYPE_HOST`) or node 0 ([`fallback::host_node`]).
    /// `None`: no pinned host memory through VMM.
    pub host_node: Option<u32>,
    pub host_granularity: usize,
    pub device_granularity: usize,
}

/// What CUDA device `device` (ggml's index) supports, or `None` if it cannot be queried.
pub fn caps(device: usize) -> Option<Caps> {
    use_device(device).ok()?;
    let mut d: CUdevice = 0;
    // SAFETY: writes the handle.
    cu(
        unsafe { cuDeviceGet(&mut d, device as c_int) },
        "cuDeviceGet",
    )
    .ok()?;
    // `None` where the driver does not know the attribute (an older driver).
    let query = |a: u32| {
        let mut v = 0;
        // SAFETY: writes one int.
        (unsafe { cuDeviceGetAttribute(&mut v, a, d) } == 0).then_some(v)
    };
    let attr = |a: u32| query(a).unwrap_or(0);
    let vmm = attr(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) != 0;
    // The driver's node; where it has none (or no such attribute), the one sysfs gives the
    // GPU's PCI address, else -1: a failed query is "unknown", never node 0.
    let host_numa_id = match query(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID) {
        Some(id) if id >= 0 => id,
        _ => device_numa_node(device).map_or(-1, |n| n as i32),
    };
    let (host_granularity, host_node) = match (vmm, u32::try_from(host_numa_id)) {
        (false, _) => (0, None),
        (true, Ok(n)) => {
            let g = granularity(CU_MEM_LOCATION_TYPE_HOST_NUMA, n as i32);
            (g, (g > 0).then_some(n))
        }
        (true, Err(_)) => {
            let node = no_node_host(device);
            (node.map_or(0, host_granularity), node)
        }
    };
    Some(Caps {
        vmm,
        host_numa_id,
        pageable_access: attr(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) != 0,
        pageable_uses_host_page_tables: attr(
            CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
        ) != 0,
        host_register: attr(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0,
        read_only_register: attr(CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED) != 0,
        host_pointer_registered: attr(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM)
            != 0,
        integrated: attr(CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0,
        cc_major: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        cc_minor: attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        sms: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        host_numa_vmm: host_granularity > 0,
        host_node,
        host_granularity,
        device_granularity: granularity(CU_MEM_LOCATION_TYPE_DEVICE, device as i32),
    })
}

/// The CUDA devices visible to the process (after `CUDA_VISIBLE_DEVICES`), from the driver
/// alone, as [`device_numa_node`]; 0 without a driver.
pub fn device_count() -> usize {
    let mut n: c_int = 0;
    // SAFETY: cuInit is idempotent; cuDeviceGetCount writes one int.
    if unsafe { cuInit(0) } != 0 || unsafe { cuDeviceGetCount(&mut n) } != 0 {
        return 0;
    }
    n.max(0) as usize
}

/// The host NUMA node of CUDA device `device` (ggml's index), from the driver alone: no context
/// is created, so `--numa` can ask before anything else touches the GPU (#422). The driver's
/// host NUMA id, else the node sysfs gives the device's PCI address; `None` when neither knows.
pub fn device_numa_node(device: usize) -> Option<u32> {
    let mut d: CUdevice = 0;
    // SAFETY: cuInit is idempotent; cuDeviceGet writes the handle.
    if unsafe { cuInit(0) } != 0 || unsafe { cuDeviceGet(&mut d, device as c_int) } != 0 {
        return None;
    }
    let mut id: c_int = -1;
    // SAFETY: writes one int.
    let r = unsafe { cuDeviceGetAttribute(&mut id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, d) };
    if r == 0 && id >= 0 {
        return Some(id as u32);
    }
    let mut bus = [0 as c_char; 32];
    // SAFETY: writes a NUL-terminated string of at most `bus.len()` bytes.
    if unsafe { cuDeviceGetPCIBusId(bus.as_mut_ptr(), bus.len() as c_int, d) } != 0 {
        return None;
    }
    // SAFETY: NUL-terminated by the driver (the buffer is zeroed and longer than any id).
    let bus = unsafe { CStr::from_ptr(bus.as_ptr()) }.to_string_lossy();
    crate::hw::numa::pci_numa_node(&crate::hw::gpu::normalize_bdf(&bus))
}

// ---------------------------------------------------------------------------------------------
// Host arena: a ggml buffer type of host-NUMA memory mapped for the CPU and the GPUs that read
// it. Every tensor gets a physical allocation of its own (the driver maps host-NUMA
// allocations only whole and at offset 0, so a tiered range can map a tensor's bytes a second
// time only if they are a handle of their own), mapped back to back in one virtual range.

/// Tensors start on mapping granules (tiered ranges map them).
const ARENA_ALIGN: usize = 2 << 20;

struct ArenaType {
    node: u32,
    devices: Vec<usize>,
    gran: usize,
    name: CString,
    buft: ffi::ggml_backend_buffer_type,
    /// Routed experts split by NUMA node ([`host_arena_split`]); `None`: all on `node`.
    split: Option<ExpertNodes>,
}

/// Where each routed expert's host bytes go in a [`host_arena_split`] arena.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpertNodes {
    /// Per expert id: its NUMA node.
    node: Vec<u32>,
}

impl ExpertNodes {
    /// The pieces (bytes, node) of a tensor of `alloc` bytes holding experts of `per` bytes
    /// each ([`super::split::expert_pieces`]).
    fn pieces(&self, per: usize, alloc: usize, gran: usize) -> Vec<(usize, u32)> {
        super::split::expert_pieces(&self.node, per, alloc, gran)
    }
}

struct Piece {
    offset: usize,
    size: usize,
    handle: CUmemGenericAllocationHandle,
}

struct ArenaBuffer {
    pieces: Vec<Piece>,
    va: CUdeviceptr,
    size: usize,
}

/// Buffer types are static descriptors: these live for the process.
struct Registry<T: 'static>(Mutex<Vec<&'static mut T>>);

// SAFETY: the registered types are only read after registration, through ggml's pointers.
unsafe impl<T: 'static> Sync for Registry<T> {}

static ARENAS: Registry<ArenaType> = Registry(Mutex::new(Vec::new()));
static TIERED: Registry<TieredType> = Registry(Mutex::new(Vec::new()));

/// The arena type behind `buft`, if it is one.
fn arena_of(buft: *const ffi::ggml_backend_buffer_type) -> Option<&'static ArenaType> {
    let list = ARENAS.0.lock().unwrap();
    list.iter()
        .find(|t| std::ptr::eq(&t.buft, buft))
        // SAFETY: registered types are leaked boxes, never freed or moved.
        .map(|t| unsafe { &*(&**t as *const ArenaType) })
}

unsafe extern "C" fn arena_name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    // SAFETY: the context is the registered ArenaType.
    unsafe { (*((*buft).context as *const ArenaType)).name.as_ptr() }
}

#[allow(clippy::boxed_local)] // takes ownership: the buffer ends here
unsafe fn arena_release(b: Box<ArenaBuffer>) {
    for p in &b.pieces {
        // SAFETY: mapped by arena_map at these addresses.
        unsafe {
            cuMemUnmap(b.va + p.offset as u64, p.size);
            cuMemRelease(p.handle);
        }
    }
    if b.va != 0 {
        // SAFETY: reserved by arena_map.
        unsafe { cuMemAddressFree(b.va, b.size) };
    }
}

unsafe extern "C" fn arena_free(buffer: ffi::ggml_backend_buffer_t) {
    // SAFETY: the context is the boxed ArenaBuffer of this buffer.
    unsafe { arena_release(Box::from_raw((*buffer).context as *mut ArenaBuffer)) }
}

unsafe extern "C" fn arena_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    // SAFETY: as above.
    unsafe { (*((*buffer).context as *const ArenaBuffer)).va as *mut c_void }
}

unsafe extern "C" fn arena_init_tensor(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
) -> ffi::ggml_status {
    // Zero what ggml-cuda's quantized kernels may read past the last row (see
    // arena_alloc_size): never NaN, so a padded block times a zero activation stays zero.
    // SAFETY: the tensor lies in this host-mapped buffer, its allocation size included.
    unsafe {
        if (*t).view_src.is_null() {
            let n = ffi::ggml_nbytes(t);
            let alloc = ffi::ggml_backend_buffer_get_alloc_size(buffer, t);
            if alloc > n {
                std::ptr::write_bytes(((*t).data as *mut u8).add(n), 0, alloc - n);
            }
        }
    }
    ffi::ggml_status_GGML_STATUS_SUCCESS
}

unsafe extern "C" fn arena_memset(
    _: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    v: u8,
    off: usize,
    n: usize,
) {
    // SAFETY: ggml checks the range against the tensor; the memory is host-mapped.
    unsafe { std::ptr::write_bytes(((*t).data as *mut u8).add(off), v, n) }
}

unsafe extern "C" fn arena_set(
    _: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe { std::ptr::copy_nonoverlapping(data as *const u8, ((*t).data as *mut u8).add(off), n) }
}

unsafe extern "C" fn arena_get(
    _: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe { std::ptr::copy_nonoverlapping(((*t).data as *const u8).add(off), data as *mut u8, n) }
}

unsafe extern "C" fn arena_cpy(
    _: ffi::ggml_backend_buffer_t,
    src: *const ffi::ggml_tensor,
    dst: *mut ffi::ggml_tensor,
) -> bool {
    // SAFETY: both host-addressable when the source buffer is a host buffer.
    unsafe {
        if !(*src).buffer.is_null() && ffi::ggml_backend_buffer_is_host((*src).buffer) {
            let n = ffi::ggml_nbytes(src);
            std::ptr::copy_nonoverlapping((*src).data as *const u8, (*dst).data as *mut u8, n);
            return true;
        }
    }
    false
}

unsafe extern "C" fn arena_clear(buffer: ffi::ggml_backend_buffer_t, v: u8) {
    // SAFETY: the whole mapped range.
    unsafe {
        let b = &*((*buffer).context as *const ArenaBuffer);
        let mapped: usize = b.pieces.iter().map(|p| p.size).sum();
        std::ptr::write_bytes(b.va as *mut u8, v, mapped);
    }
}

/// Pieces of the given sizes (granule multiples) on the given NUMA nodes, back to back in one
/// virtual range mapped for the CPU and the type's devices.
fn arena_map(t: &ArenaType, sizes: &[(usize, u32)]) -> Result<Box<ArenaBuffer>, String> {
    let total = sizes.iter().map(|p| p.0).sum::<usize>().max(t.gran);
    if let Some(&d) = t.devices.first() {
        use_device(d)?;
    }
    let mut b = Box::new(ArenaBuffer {
        pieces: Vec::new(),
        va: 0,
        size: total,
    });
    let fail = |b: Box<ArenaBuffer>, e: String| {
        // SAFETY: releases what was mapped so far.
        unsafe { arena_release(b) };
        Err(e)
    };
    // SAFETY: writes the reserved address.
    if let Err(e) = cu(
        unsafe { cuMemAddressReserve(&mut b.va, total, ARENA_ALIGN, 0, 0) },
        "cuMemAddressReserve",
    ) {
        b.va = 0;
        return fail(b, e);
    }
    let mut off = 0usize;
    for &(size, node) in sizes.iter().filter(|z| z.0 > 0) {
        let prop = host_prop(node);
        let mut handle = 0;
        // SAFETY: creates and maps one allocation inside the reserved range.
        unsafe {
            if let Err(e) = cu(
                cuMemCreate(&mut handle, size, &prop, 0),
                "cuMemCreate(host NUMA)",
            ) {
                return fail(b, e);
            }
            if let Err(e) = cu(
                cuMemMap(b.va + off as u64, size, 0, handle, 0),
                "cuMemMap(arena)",
            ) {
                cuMemRelease(handle);
                return fail(b, e);
            }
        }
        let mut desc = vec![host_access(node)];
        desc.extend(
            t.devices
                .iter()
                .map(|&d| access(CU_MEM_LOCATION_TYPE_DEVICE, d as i32)),
        );
        // SAFETY: the piece just mapped.
        if let Err(e) = cu(
            unsafe { cuMemSetAccess(b.va + off as u64, size, desc.as_ptr(), desc.len()) },
            "cuMemSetAccess",
        ) {
            b.pieces.push(Piece {
                offset: off,
                size,
                handle,
            });
            return fail(b, e);
        }
        b.pieces.push(Piece {
            offset: off,
            size,
            handle,
        });
        off += size;
    }
    Ok(b)
}

fn arena_wrap(
    buft: ffi::ggml_backend_buffer_type_t,
    b: Box<ArenaBuffer>,
) -> ffi::ggml_backend_buffer_t {
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(arena_free),
        get_base: Some(arena_base),
        init_tensor: Some(arena_init_tensor),
        memset_tensor: Some(arena_memset),
        set_tensor: Some(arena_set),
        get_tensor: Some(arena_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: Some(arena_cpy),
        clear: Some(arena_clear),
        reset: None,
    };
    let size = b.size;
    // SAFETY: ggml takes the interface and the boxed context (freed by arena_free).
    unsafe { ffi::ggml_backend_buffer_init(buft, iface, Box::into_raw(b) as *mut c_void, size) }
}

unsafe extern "C" fn arena_alloc(
    buft: ffi::ggml_backend_buffer_type_t,
    size: usize,
) -> ffi::ggml_backend_buffer_t {
    // SAFETY: the context is the registered ArenaType.
    let t = unsafe { &*((*buft).context as *const ArenaType) };
    match arena_map(t, &[(round_up(size.max(1), t.gran), t.node)]) {
        Ok(b) => arena_wrap(buft, b),
        Err(e) => {
            set_error(e);
            null_mut()
        }
    }
}

/// Every tensor in a piece of its own, laid out as ggml's linear allocator places them: in
/// order, each at a multiple of the alignment.
unsafe extern "C" fn arena_alloc_n(
    buft: ffi::ggml_backend_buffer_type_t,
    tensors: *mut *mut ffi::ggml_tensor,
    n: c_int,
) -> ffi::ggml_backend_buffer_t {
    // SAFETY: ggml passes `n` live tensors and our registered type.
    unsafe {
        let t = &*((*buft).context as *const ArenaType);
        let tensors = std::slice::from_raw_parts(tensors, n as usize);
        let sizes: Vec<(usize, u32)> = tensors
            .iter()
            .filter(|&&x| (*x).data.is_null() && (*x).view_src.is_null())
            .flat_map(|&x| {
                let alloc = round_up(arena_alloc_size(buft, x), t.gran);
                let name = CStr::from_ptr((*x).name.as_ptr()).to_string_lossy();
                match &t.split {
                    Some(s) if name.contains("_exps.") && (*x).ne[2] == s.node.len() as i64 => {
                        s.pieces((*x).nb[2], alloc, t.gran)
                    }
                    _ => vec![(alloc, t.node)],
                }
            })
            .collect();
        let b = match arena_map(t, &sizes) {
            Ok(b) => b,
            Err(e) => {
                set_error(e);
                return null_mut();
            }
        };
        let buf = arena_wrap(buft, b);
        let mut talloc = ffi::ggml_tallocr_new(buf);
        for &x in tensors {
            let st = if (*x).data.is_null() && (*x).view_src.is_null() {
                ffi::ggml_tallocr_alloc(&mut talloc, x)
            } else if !(*x).view_src.is_null() && (*x).buffer.is_null() {
                ffi::ggml_backend_view_init(x)
            } else {
                ffi::ggml_status_GGML_STATUS_SUCCESS
            };
            if st != ffi::ggml_status_GGML_STATUS_SUCCESS {
                set_error(format!(
                    "arena: cannot place tensor {}",
                    CStr::from_ptr((*x).name.as_ptr()).to_string_lossy()
                ));
                ffi::ggml_backend_buffer_free(buf);
                return null_mut();
            }
        }
        buf
    }
}

unsafe extern "C" fn arena_alignment(buft: ffi::ggml_backend_buffer_type_t) -> usize {
    // SAFETY: the context is the registered ArenaType.
    unsafe { (*((*buft).context as *const ArenaType)).gran }
}

/// As ggml-cuda's device buffers: a quantized tensor gets one padded row more, which its
/// matmul kernels may read. Rounded to the granule, so the next tensor starts on one.
unsafe extern "C" fn arena_alloc_size(
    buft: ffi::ggml_backend_buffer_type_t,
    t: *const ffi::ggml_tensor,
) -> usize {
    const PAD: i64 = 512; // ggml-cuda's MATRIX_ROW_PADDING
    // SAFETY: a live tensor and our registered type.
    unsafe {
        let mut size = ffi::ggml_nbytes(t);
        let ne0 = (*t).ne[0];
        if ffi::ggml_is_quantized((*t).type_) && ne0 % PAD != 0 {
            size += ffi::ggml_row_size((*t).type_, PAD - ne0 % PAD);
        }
        round_up(size, (*((*buft).context as *const ArenaType)).gran)
    }
}

unsafe extern "C" fn arena_is_host(_: ffi::ggml_backend_buffer_type_t) -> bool {
    true
}

/// A buffer type of pinned host memory on NUMA node `node`, mapped for the CPU and
/// `devices`. Fails when this driver cannot allocate host-NUMA memory through VMM.
pub fn host_arena(node: u32, devices: &[usize]) -> Result<BufferType, String> {
    arena_type(node, devices, None)
}

/// [`host_arena`] with the routed experts split by NUMA node: expert `e`'s host bytes of every
/// expert tensor (`*_exps.*`, `expert_node.len()` experts) on node `expert_node[e]`, each run of
/// experts on one node a piece of its own (other tensors on the first expert's node). For
/// expert parallelism over GPUs on several sockets (GLM's `--glm-parallel-decode`): each GPU
/// reads the experts it owns from its own socket's memory.
pub fn host_arena_split(expert_node: &[u32], devices: &[usize]) -> Result<BufferType, String> {
    let node = *expert_node.first().ok_or("no experts to split")?;
    arena_type(
        node,
        devices,
        Some(ExpertNodes {
            node: expert_node.to_vec(),
        }),
    )
}

fn arena_type(
    node: u32,
    devices: &[usize],
    split: Option<ExpertNodes>,
) -> Result<BufferType, String> {
    let mut devs = devices.to_vec();
    devs.sort_unstable();
    let mut list = ARENAS.0.lock().unwrap();
    if let Some(t) = list
        .iter_mut()
        .find(|t| t.node == node && t.devices == devs && t.split == split)
    {
        return Ok(BufferType(NonNull::from(&mut t.buft)));
    }
    if let Some(&d) = devs.first() {
        use_device(d)?;
    }
    let mut nodes = vec![node];
    nodes.extend(split.iter().flat_map(|s| s.node.iter().copied()));
    nodes.sort_unstable();
    nodes.dedup();
    let mut gran = 0;
    for &n in &nodes {
        let g = host_granularity(n);
        if g == 0 {
            return Err("this driver cannot allocate host NUMA memory through cuMemCreate".into());
        }
        gran = gran.max(g);
    }
    // A GPU with no host NUMA node: "Host" in place of a node number.
    let label = |n: u32| {
        if n == NO_NODE {
            "Host".to_string()
        } else {
            n.to_string()
        }
    };
    let name = match &split {
        Some(_) => format!(
            "HostNuma{}",
            nodes
                .iter()
                .map(|&n| label(n))
                .collect::<Vec<_>>()
                .join("+")
        ),
        None if node == NO_NODE => "HostPinned".to_string(),
        None => format!("HostNuma{node}"),
    };
    let t: &'static mut ArenaType = Box::leak(Box::new(ArenaType {
        node,
        devices: devs,
        gran: gran.max(ARENA_ALIGN),
        name: CString::new(name).unwrap(),
        split,
        buft: ffi::ggml_backend_buffer_type {
            iface: ffi::ggml_backend_buffer_type_i {
                get_name: Some(arena_name),
                alloc_buffer: Some(arena_alloc),
                alloc_buffer_n: Some(arena_alloc_n),
                get_alignment: Some(arena_alignment),
                get_max_size: None,
                get_alloc_size: Some(arena_alloc_size),
                get_alloc_size_n: None,
                is_host: Some(arena_is_host),
            },
            // SAFETY: a registry query.
            device: unsafe {
                ffi::ggml_backend_dev_by_type(
                    ffi::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU,
                )
            },
            context: null_mut(),
        },
    }));
    t.buft.context = t as *mut ArenaType as *mut c_void;
    let buft = BufferType(NonNull::from(&mut t.buft));
    list.push(t);
    Ok(buft)
}

/// The NUMA node of an arena buffer type made by [`host_arena`] (of a split one, its first
/// expert's).
pub fn arena_node(buft: BufferType) -> Option<u32> {
    arena_of(buft.raw()).map(|t| t.node)
}

/// The physical pieces `(bytes, node)` of tensor `x`'s allocation in the host arena `buft`,
/// from its first byte (as [`arena_alloc_n`] maps them): one per run of experts on a node for
/// an expert tensor of a [`host_arena_split`] arena, else the whole allocation on the arena's
/// node. `None` when `buft` is no arena.
pub fn arena_pieces(buft: BufferType, x: Tensor<'_>) -> Option<Vec<(usize, u32)>> {
    let t = arena_of(buft.raw())?;
    // SAFETY: a live tensor and our registered type.
    let alloc = round_up(unsafe { arena_alloc_size(buft.raw(), x.raw()) }, t.gran);
    Some(match &t.split {
        Some(s) if x.name().contains("_exps.") && x.ne()[2] == s.node.len() as i64 => {
            s.pieces(x.nb()[2], alloc, t.gran)
        }
        _ => vec![(alloc, t.node)],
    })
}

// ---------------------------------------------------------------------------------------------
// Tiered ranges: [VRAM slots | an arena tensor's host bytes] in one virtual range of a device.

struct TieredType {
    device: usize,
    name: CString,
    buft: ffi::ggml_backend_buffer_type,
}

#[derive(Clone)]
struct TieredBuffer {
    device: usize,
    /// The slot part's VRAM up to `split` (all of it without a tail).
    vram: CUmemGenericAllocationHandle,
    /// The slot part's VRAM from `split` on, an allocation of its own that a long prompt can
    /// borrow alone ([`unback_tail`]); 0 without one.
    tail: CUmemGenericAllocationHandle,
    va: CUdeviceptr,
    vram_bytes: usize,
    /// Where `tail` starts in the slot part (`vram_bytes` without a tail).
    split: usize,
    host_bytes: usize,
    /// The first `vram_bytes` are not mapped: the slots' VRAM was lent out ([`unback`]), or a
    /// staged range over lent slots ([`staged_over_hole`], [`staged_packed`]).
    hole: bool,
    /// The slot part from `split` on is not mapped: its tail was lent out ([`unback_tail`]).
    tail_hole: bool,
    /// `vram` (head) or `tail` belong to a device's kept staging buffers
    /// ([`Slots::with_parts`]): unmapped here, never released.
    shared_head: bool,
    shared_tail: bool,
    /// A view of a range another owns ([`StagedRange::view`]): releases nothing.
    borrowed: bool,
}

impl TieredBuffer {
    /// Unmaps and frees what it holds (the host part's physical memory stays the arena's).
    fn release(&self) {
        if self.borrowed {
            return;
        }
        // SAFETY: only what this buffer mapped and reserved.
        unsafe {
            let total = self.vram_bytes + self.host_bytes;
            if self.hole || self.tail_hole {
                if !self.hole && self.split > 0 {
                    cuMemUnmap(self.va, self.split);
                }
                if self.host_bytes > 0 {
                    cuMemUnmap(self.va + self.vram_bytes as u64, self.host_bytes);
                }
            } else {
                cuMemUnmap(self.va, total);
            }
            cuMemAddressFree(self.va, total);
            if self.vram != 0 && !self.shared_head {
                cuMemRelease(self.vram);
            }
            if self.tail != 0 && !self.shared_tail {
                cuMemRelease(self.tail);
            }
        }
    }

    /// Whether `h`, mapped in the slot part, is a kept staging buffer's.
    fn is_shared(&self, h: CUmemGenericAllocationHandle) -> bool {
        (h == self.vram && self.shared_head) || (h == self.tail && self.shared_tail)
    }

    /// The handles mapped in the slot part, with their sizes.
    fn slot_parts(&self) -> Vec<(CUmemGenericAllocationHandle, usize)> {
        let mut parts = Vec::new();
        if self.hole {
            return parts;
        }
        if self.vram != 0 {
            parts.push((self.vram, self.split));
        }
        if self.tail != 0 && !self.tail_hole {
            parts.push((self.tail, self.vram_bytes - self.split));
        }
        parts
    }
}

/// The tiered buffer behind tensor `t`, if it is in one.
fn tiered_of(t: Tensor<'_>) -> Option<*mut TieredBuffer> {
    // SAFETY: reading the buffer of a live tensor; a tiered buffer's context is TieredBuffer.
    unsafe {
        let buf = (*t.raw()).buffer;
        let ok = !buf.is_null()
            && TIERED
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|x| std::ptr::eq(&x.buft, (*buf).buft));
        ok.then(|| (*buf).context as *mut TieredBuffer)
    }
}

/// Copies `bytes` from `src` to `dst` (device addresses, on one GPU or two) and waits for it.
///
/// A device-to-device `cudaMemcpy` returns before the copy is done, on the legacy stream,
/// which ggml's streams do not wait for: the copy is waited for here, before the caller
/// reuses either range.
///
/// # Safety
/// Both ranges must be live device memory that nothing else reads or writes meanwhile.
pub unsafe fn copy_now(dst: usize, src: usize, bytes: usize) -> Result<(), String> {
    // SAFETY: the caller's ranges; the legacy stream of the current device.
    unsafe {
        rt(
            cudaMemcpy(
                dst as *mut c_void,
                src as *const c_void,
                bytes,
                CUDA_MEMCPY_DEFAULT,
            ),
            "cudaMemcpy",
        )?;
        rt(cudaStreamSynchronize(null_mut()), "cudaStreamSynchronize")
    }
}

/// `(free, total)` VRAM of `device`, as the driver reports it.
pub fn vram(device: usize) -> Result<(usize, usize), String> {
    use_device(device)?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two out-parameters.
    rt(
        unsafe { cudaMemGetInfo(&mut free, &mut total) },
        "cudaMemGetInfo",
    )?;
    Ok((free, total))
}

/// Free VRAM on `device`, as the driver reports it.
pub fn free_vram(device: usize) -> Result<usize, String> {
    use_device(device)?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two out-parameters.
    let e = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    if e != 0 {
        return Err(format!("cudaMemGetInfo: error {e}"));
    }
    Ok(free)
}

/// Maps the host part of the tiered tensor `t` (from [`tiered`]) for CUDA device `peer` too,
/// which may then read the tensor's uncached experts at the same addresses (the idle NVLink peer
/// fetching them for the owning device, #252). Returns the tensor's address.
pub fn share_host_part(t: Tensor<'_>, peer: usize) -> Result<usize, String> {
    let b = tiered_of(t).ok_or("not a tiered tensor")?;
    // SAFETY: the tensor's own tiered buffer; access is granted on its mapped host part only,
    // the owning device's included again (the same flags).
    unsafe {
        let b = &*b;
        if b.host_bytes > 0 {
            use_device(b.device)?;
            let a = [
                access(CU_MEM_LOCATION_TYPE_DEVICE, b.device as i32),
                access(CU_MEM_LOCATION_TYPE_DEVICE, peer as i32),
            ];
            cu(
                cuMemSetAccess(b.va + b.vram_bytes as u64, b.host_bytes, a.as_ptr(), 2),
                "cuMemSetAccess(host part, peer)",
            )?;
        }
        Ok((*t.raw()).data as usize)
    }
}

/// Lends out the VRAM slots of the tiered tensor `t` (from [`tiered`]): unmaps and frees
/// their memory, leaving the range's slot part unmapped. Nothing may read the slots until
/// [`Slots::reback`] (the caller points the layer's table at host memory first). Returns the bytes
/// freed.
pub fn unback(t: Tensor<'_>) -> Result<usize, String> {
    let b = tiered_of(t).ok_or("not a tiered tensor")?;
    // SAFETY: the tensor's own tiered buffer; no kernel reads its slots (see above).
    unsafe {
        let b = &mut *b;
        let parts = b.slot_parts();
        if parts.is_empty() {
            return Ok(0);
        }
        use_device(b.device)?;
        let d = b.device;
        let n: usize = parts.iter().map(|p| p.1).sum();
        cu(
            timed("cuMemUnmap", d, n, || cuMemUnmap(b.va, n)),
            "cuMemUnmap(VRAM slots)",
        )?;
        let own: Vec<_> = parts.into_iter().filter(|p| !b.is_shared(p.0)).collect();
        for (h, size) in own {
            cu(
                timed("cuMemRelease", d, size, || cuMemRelease(h)),
                "cuMemRelease(VRAM slots)",
            )?;
        }
        b.vram = 0;
        b.tail = 0;
        b.hole = true;
        b.tail_hole = false;
        b.shared_head = false;
        b.shared_tail = false;
        Ok(n)
    }
}

/// The slot range of a tiered tensor, to give VRAM back from any thread ([`Slots::reback`]).
pub struct Slots(*mut TieredBuffer, Option<(u64, u64)>);

// SAFETY: the tiered buffer outlives the handle (the residency joins the thread using it before
// it drops its tensors), and nothing else touches its slot part meanwhile (see `unback`).
unsafe impl Send for Slots {}

impl Slots {
    /// The slot range of the tiered tensor `t`.
    pub fn of(t: Tensor<'_>) -> Result<Self, String> {
        tiered_of(t)
            .map(|b| Self(b, None))
            .ok_or_else(|| "not a tiered tensor".into())
    }

    /// The range's own allocations `(vram, tail)` (from [`slot_handles`], 0: none), which a
    /// device's kept staging buffers hold while its slots are lent: [`Self::reback`] maps them
    /// back instead of new memory, and they stay the staging buffers' (`shared`).
    pub fn with_parts(self, parts: Option<(u64, u64)>) -> Self {
        Self(self.0, parts)
    }

    /// Gives the range VRAM slots again after [`unback`] (their contents undefined).
    pub fn reback(&self) -> Result<(), String> {
        match self.1 {
            Some(parts) => reback_shared(self.0, parts),
            None => reback_buffer(self.0),
        }
    }
}

/// The allocations `(vram, tail)` mapped in the slot part of the tiered tensor `t` (0: none)
/// that a lend gives away, if the whole slot part is mapped and they are its own: both, or
/// with `tail_only` (a lend of the tail, [`unback_tail`]) the tail only.
pub fn slot_handles(t: Tensor<'_>, tail_only: bool) -> Option<(u64, u64)> {
    let b = tiered_of(t)?;
    // SAFETY: reading the fields of the tensor's live tiered buffer.
    let b = unsafe { &*b };
    if b.hole || b.tail_hole || b.shared_head || b.shared_tail {
        return None;
    }
    let head = if tail_only { 0 } else { b.vram };
    (head != 0 || b.tail != 0).then_some((head, b.tail))
}

/// [`reback_buffer`] mapping `(vram, tail)` (0: new memory), the range's own allocations of
/// before the lend (now a kept staging buffer's), at their places again.
fn reback_shared(b: *mut TieredBuffer, (vram, tail): (u64, u64)) -> Result<(), String> {
    // SAFETY: maps the range's former allocations or new memory into its unmapped slot part.
    unsafe {
        let b = &mut *b;
        if !b.hole && !b.tail_hole {
            return Ok(());
        }
        use_device(b.device)?;
        let d = b.device;
        let a = access(CU_MEM_LOCATION_TYPE_DEVICE, d as i32);
        // One part of the slot part: `h` mapped again, or new memory.
        let map = |at: usize, n: usize, h: u64| -> Result<(u64, bool), String> {
            let va = b.va + at as u64;
            if h == 0 {
                return map_fresh(d, va, n).map(|h| (h, false));
            }
            cu(
                timed("cuMemMap", d, n, || cuMemMap(va, n, 0, h, 0)),
                "cuMemMap(shared VRAM slots)",
            )?;
            if let Err(e) = cu(
                timed("cuMemSetAccess", d, n, || cuMemSetAccess(va, n, &a, 1)),
                "cuMemSetAccess(shared VRAM slots)",
            ) {
                cuMemUnmap(va, n);
                return Err(e);
            }
            Ok((h, true))
        };
        let (split, tail_n) = (b.split, b.vram_bytes - b.split);
        let head = if b.hole && split > 0 {
            Some(map(0, split, vram)?)
        } else {
            None
        };
        let back = if tail_n > 0 {
            match map(split, tail_n, tail) {
                Ok(t) => Some(t),
                Err(e) => {
                    if let Some((h, shared)) = head {
                        cuMemUnmap(b.va, split);
                        if !shared {
                            cuMemRelease(h);
                        }
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };
        if b.hole {
            let (h, shared) = head.unwrap_or((0, false));
            b.vram = h;
            b.shared_head = shared;
        }
        let (h, shared) = back.unwrap_or((0, false));
        b.tail = h;
        b.shared_tail = shared;
        b.hole = false;
        b.tail_hole = false;
        Ok(())
    }
}

fn reback_buffer(b: *mut TieredBuffer) -> Result<(), String> {
    // SAFETY: maps new memory into the range's unmapped slot part.
    unsafe {
        let b = &mut *b;
        if !b.hole && !b.tail_hole {
            return Ok(());
        }
        use_device(b.device)?;
        // The same allocations as before the lend: a tail can be lent alone again.
        let tail_at = b.va + b.split as u64;
        let tail_n = b.vram_bytes - b.split;
        if b.hole {
            let head = if b.split > 0 {
                map_fresh(b.device, b.va, b.split)?
            } else {
                0
            };
            let tail = if tail_n > 0 {
                match map_fresh(b.device, tail_at, tail_n) {
                    Ok(h) => h,
                    Err(e) => {
                        if head != 0 {
                            cuMemUnmap(b.va, b.split);
                            cuMemRelease(head);
                        }
                        return Err(e);
                    }
                }
            } else {
                0
            };
            b.vram = head;
            b.tail = tail;
        } else {
            b.tail = map_fresh(b.device, tail_at, tail_n)?;
        }
        b.hole = false;
        b.tail_hole = false;
        Ok(())
    }
}

/// A new allocation of `n` bytes of `device`, mapped at `at` and accessible to the device.
///
/// # Safety
/// `at..at + n` must be reserved and unmapped.
unsafe fn map_fresh(
    device: usize,
    at: CUdeviceptr,
    n: usize,
) -> Result<CUmemGenericAllocationHandle, String> {
    let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
    let mut h: CUmemGenericAllocationHandle = 0;
    // SAFETY: creates an allocation and maps it into the caller's reserved range.
    unsafe {
        cu(
            timed("cuMemCreate", device, n, || {
                cuMemCreate(&mut h, n, &prop, 0)
            }),
            "cuMemCreate(VRAM slots)",
        )?;
        if let Err(e) = cu(
            timed("cuMemMap", device, n, || cuMemMap(at, n, 0, h, 0)),
            "cuMemMap(VRAM slots)",
        ) {
            cuMemRelease(h);
            return Err(e);
        }
        let a = access(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
        if let Err(e) = cu(
            timed("cuMemSetAccess", device, n, || cuMemSetAccess(at, n, &a, 1)),
            "cuMemSetAccess(VRAM slots)",
        ) {
            cuMemUnmap(at, n);
            cuMemRelease(h);
            return Err(e);
        }
    }
    Ok(h)
}

unsafe extern "C" fn tiered_name(buft: ffi::ggml_backend_buffer_type_t) -> *const c_char {
    // SAFETY: the context is the registered TieredType.
    unsafe { (*((*buft).context as *const TieredType)).name.as_ptr() }
}

unsafe extern "C" fn tiered_alloc(
    _: ffi::ggml_backend_buffer_type_t,
    _: usize,
) -> ffi::ggml_backend_buffer_t {
    set_error("tiered buffers are made by residency::cuda::tiered".into());
    null_mut()
}

unsafe extern "C" fn tiered_alignment(_: ffi::ggml_backend_buffer_type_t) -> usize {
    256
}

fn tiered_ctx<'a>(buffer: ffi::ggml_backend_buffer_t) -> &'a TieredBuffer {
    // SAFETY: the context is the boxed TieredBuffer of this buffer.
    unsafe { &*((*buffer).context as *const TieredBuffer) }
}

unsafe extern "C" fn tiered_free(buffer: ffi::ggml_backend_buffer_t) {
    let b = tiered_ctx(buffer);
    let n = b.vram_bytes + b.host_bytes;
    // SAFETY: no kernel may still read the range.
    if !SYNCED.with(|s| s.get()) {
        timed("cudaDeviceSynchronize", b.device, n, || unsafe {
            cudaSetDevice(b.device as c_int);
            cudaDeviceSynchronize();
        });
    }
    timed("release range", b.device, n, || b.release());
    // SAFETY: frees the box handed to ggml in `tiered`.
    drop(unsafe { Box::from_raw((*buffer).context as *mut TieredBuffer) });
}

unsafe extern "C" fn tiered_base(buffer: ffi::ggml_backend_buffer_t) -> *mut c_void {
    tiered_ctx(buffer).va as *mut c_void
}

unsafe extern "C" fn tiered_memset(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    v: u8,
    off: usize,
    n: usize,
) {
    // SAFETY: a range of the tensor, addressable by the device.
    unsafe {
        cudaSetDevice(tiered_ctx(buffer).device as c_int);
        cudaMemset(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            v as c_int,
            n,
        );
    }
}

unsafe extern "C" fn tiered_set(
    buffer: ffi::ggml_backend_buffer_t,
    t: *mut ffi::ggml_tensor,
    data: *const c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(tiered_ctx(buffer).device as c_int);
        cudaMemcpy(
            ((*t).data as *mut u8).add(off) as *mut c_void,
            data,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn tiered_get(
    buffer: ffi::ggml_backend_buffer_t,
    t: *const ffi::ggml_tensor,
    data: *mut c_void,
    off: usize,
    n: usize,
) {
    // SAFETY: as above.
    unsafe {
        cudaSetDevice(tiered_ctx(buffer).device as c_int);
        cudaMemcpy(
            data,
            ((*t).data as *const u8).add(off) as *const c_void,
            n,
            CUDA_MEMCPY_DEFAULT,
        );
    }
}

unsafe extern "C" fn tiered_clear(buffer: ffi::ggml_backend_buffer_t, v: u8) {
    let b = tiered_ctx(buffer);
    // SAFETY: the VRAM part only; the host part belongs to the arena.
    unsafe {
        cudaSetDevice(b.device as c_int);
        cudaMemset(b.va as *mut c_void, v as c_int, b.vram_bytes);
        cudaDeviceSynchronize();
    }
}

fn tiered_type(device: usize) -> ffi::ggml_backend_buffer_type_t {
    let mut list = TIERED.0.lock().unwrap();
    if let Some(t) = list.iter_mut().find(|t| t.device == device) {
        return &mut t.buft;
    }
    let t: &'static mut TieredType = Box::leak(Box::new(TieredType {
        device,
        name: CString::new(format!("CUDA{device}_Tiered")).unwrap(),
        buft: ffi::ggml_backend_buffer_type {
            iface: ffi::ggml_backend_buffer_type_i {
                get_name: Some(tiered_name),
                alloc_buffer: Some(tiered_alloc),
                alloc_buffer_n: None,
                get_alignment: Some(tiered_alignment),
                get_max_size: None,
                get_alloc_size: None,
                get_alloc_size_n: None,
                is_host: None,
            },
            // SAFETY: the CUDA device of ggml's own buffer type for it.
            device: unsafe {
                ffi::ggml_backend_buft_get_device(ffi::ggml_backend_cuda_buffer_type(
                    device as c_int,
                ))
            },
            context: null_mut(),
        },
    }));
    t.buft.context = t as *mut TieredType as *mut c_void;
    let p: *mut ffi::ggml_backend_buffer_type = &mut t.buft;
    list.push(t);
    p
}

/// A tensor of `n_slots + host.ne[2]` experts on `device`: VRAM slots, then `host`'s bytes
/// (a tensor in a [`host_arena`] buffer, its arena piece mapped a second time), in one
/// virtual range. The returned context owns the range; its only tensor is the tiered one.
///
/// With `keep`, the slots' VRAM is two allocations: about the first `keep` slots' (cut at a
/// mapping granule, [`tiered_split`]), and a tail [`unback_tail`] lends alone.
pub fn tiered(
    device: usize,
    host: Tensor<'_>,
    n_slots: usize,
    keep: Option<usize>,
    name: &str,
) -> Result<Context, TieredError> {
    let err = |e: BufError| TieredError {
        error: GgmlError::Alloc {
            bytes: n_slots * host.nb()[2],
            buffer: format!("tiered range on CUDA{device}: {}", e.msg),
        },
        no_vram: e.no_vram,
    };
    let (buf, offset) =
        tiered_buffer(device, host, host.nb()[2], n_slots, keep, true).map_err(err)?;
    let ne = host.ne();
    let ctx = Context::new(1, 0);
    let t = ctx
        .new_tensor(host.ty(), &[ne[0], ne[1], n_slots as i64 + ne[2]])
        .named(name);
    // SAFETY: the buffer is new and owned by ctx from here; the tensor fits from `offset`.
    unsafe { ctx.adopt_buffer(t, buf, offset) }.map_err(|error| TieredError {
        error,
        no_vram: false,
    })?;
    Ok(ctx)
}

/// Why [`tiered`] failed.
#[derive(Debug)]
pub struct TieredError {
    pub error: GgmlError,
    /// The slots' VRAM did not fit: `cuMemCreate` ran out of device memory (#1208: the expert
    /// cache then shrinks to what fits). Any other failure is an error of the load.
    pub no_vram: bool,
}

impl std::fmt::Display for TieredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.error, f)
    }
}

/// A [`tiered_buffer`] failure: the message, and whether a slot allocation ran out of VRAM.
struct BufError {
    msg: String,
    no_vram: bool,
}

impl From<String> for BufError {
    fn from(msg: String) -> Self {
        Self {
            msg,
            no_vram: false,
        }
    }
}

impl From<&str> for BufError {
    fn from(msg: &str) -> Self {
        msg.to_string().into()
    }
}

/// `cuMemCreate` of `bytes` of a tiered range's VRAM slots, out of device memory told apart.
/// `hook`: the expert slots, which `LEGION_DEBUG_FAIL_CUMEMCREATE_AFTER` may fail.
///
/// # Safety
/// As `cuMemCreate`: `h` receives the new handle.
unsafe fn create_slots(
    h: &mut CUmemGenericAllocationHandle,
    bytes: usize,
    prop: &CUmemAllocationProp,
    what: &str,
    hook: bool,
) -> Result<(), BufError> {
    let r = if hook && injected_no_vram() {
        CUDA_ERROR_OUT_OF_MEMORY
    } else {
        // SAFETY: the caller's handle and a valid property struct.
        unsafe { cuMemCreate(h, bytes, prop, 0) }
    };
    cu(r, what).map_err(|msg| BufError {
        msg,
        no_vram: r == CUDA_ERROR_OUT_OF_MEMORY,
    })
}

/// Debug hook (docs/debug-env.md): `LEGION_DEBUG_FAIL_CUMEMCREATE_AFTER=<n>[,<count>]` makes
/// the expert slots' `cuMemCreate` calls after the first `n` fail as out of memory, `count` of
/// them in a row (1 if left out), to try the cache's shrink to fit (#1208).
fn injected_no_vram() -> bool {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static FAIL: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let fail = FAIL.get_or_init(|| {
        let v = std::env::var("LEGION_DEBUG_FAIL_CUMEMCREATE_AFTER").ok()?;
        let (n, count) = v.split_once(',').unwrap_or((v.as_str(), "1"));
        Some((n.trim().parse().ok()?, count.trim().parse().ok()?))
    });
    let Some((n, count)) = *fail else {
        return false;
    };
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
    (n..n.saturating_add(count)).contains(&call)
}

/// [`tiered`] by rows: `n_rows` rows in VRAM, then the rows of the arena tensor `host` (2D
/// `[ne0, ne1]`, a piece of its own), as one tensor `name` of `[ne0, n_rows + ne1]` in one
/// virtual range of `device`. A KV pool's host tier (#352): kernels and ggml's ops read every
/// row through the same pointer, the first `n_rows` from VRAM and the rest over the link.
pub fn tiered_rows(
    device: usize,
    host: Tensor<'_>,
    n_rows: usize,
    name: &str,
) -> Result<Context, GgmlError> {
    let err = |msg: String| GgmlError::Alloc {
        bytes: n_rows * host.nb()[1],
        buffer: format!("tiered rows on CUDA{device}: {msg}"),
    };
    let (buf, offset) =
        tiered_buffer(device, host, host.nb()[1], n_rows, None, false).map_err(|e| err(e.msg))?;
    // Written by the graphs (a cache), not weights.
    // SAFETY: a live buffer.
    unsafe {
        ffi::ggml_backend_buffer_set_usage(
            buf.as_ptr(),
            ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_ANY,
        )
    };
    let ne = host.ne();
    let ctx = Context::new(1, 0);
    let t = ctx
        .new_tensor(host.ty(), &[ne[0], n_rows as i64 + ne[1]])
        .named(name);
    // SAFETY: the buffer is new and owned by ctx from here; the tensor fits from `offset`.
    unsafe { ctx.adopt_buffer(t, buf, offset)? };
    Ok(ctx)
}

/// A tiered buffer: `n_slots` slots of `slot_bytes` in VRAM right before the arena tensor
/// `host`'s bytes; with the offset of the slots in it. `keep`: see [`tiered`]. `hook`: expert
/// slots ([`create_slots`]).
fn tiered_buffer(
    device: usize,
    host: Tensor<'_>,
    slot_bytes: usize,
    n_slots: usize,
    keep: Option<usize>,
    hook: bool,
) -> Result<(NonNull<ffi::ggml_backend_buffer>, usize), BufError> {
    // SAFETY: reading the fields of a live tensor and of its buffer.
    let (arena_buf, data, view) = unsafe {
        let t = host.raw();
        ((*t).buffer, (*t).data as u64, !(*t).view_src.is_null())
    };
    // SAFETY: as above.
    let at = (!arena_buf.is_null())
        .then(|| arena_of(unsafe { (*arena_buf).buft }))
        .flatten()
        .filter(|_| !view)
        .ok_or("the host tensor is not in a host arena buffer")?;
    // SAFETY: the buffer is an arena buffer, so its context is an ArenaBuffer.
    let ab = unsafe { &*((*arena_buf).context as *const ArenaBuffer) };
    let off = (data - ab.va) as usize;
    // The tensor's pieces: one, or one per NUMA node of a split arena (back to back).
    let first_piece = ab
        .pieces
        .iter()
        .position(|p| p.offset == off)
        .ok_or("the host tensor is not an arena piece of its own")?;
    let mut pieces: Vec<&Piece> = Vec::new();
    let mut covered = 0;
    for p in &ab.pieces[first_piece..] {
        if covered >= host.nbytes() {
            break;
        }
        pieces.push(p);
        covered += p.size;
    }
    if covered < host.nbytes() {
        return Err("the host tensor runs past its arena pieces".into());
    }
    let host_bytes = covered;
    use_device(device)?;
    let dgran = granularity(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
    if dgran == 0 || !at.gran.is_multiple_of(dgran) {
        return Err("device and host mapping granules do not line up".into());
    }
    let slots = slot_bytes * n_slots;
    let vram_bytes = round_up(slots, at.gran);
    let total = vram_bytes + host_bytes;
    // The slots end where the host part starts.
    let first = vram_bytes - slots;
    let split = match keep {
        Some(k) if k < n_slots => (first + k * slot_bytes) / at.gran * at.gran,
        _ => vram_bytes,
    };
    let mut b = Box::new(TieredBuffer {
        device,
        vram: 0,
        tail: 0,
        va: 0,
        vram_bytes,
        split,
        host_bytes,
        hole: false,
        tail_hole: false,
        shared_head: false,
        shared_tail: false,
        borrowed: false,
    });
    // SAFETY: reserves, maps and grants access within the new range; on failure releases it.
    unsafe {
        cu(
            cuMemAddressReserve(&mut b.va, total, at.gran, 0, 0),
            "cuMemAddressReserve",
        )?;
        let mut done = || -> Result<(), BufError> {
            let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
            if split > 0 {
                create_slots(&mut b.vram, split, &prop, "cuMemCreate(VRAM slots)", hook)?;
                cu(cuMemMap(b.va, split, 0, b.vram, 0), "cuMemMap(VRAM slots)")?;
            }
            if split < vram_bytes {
                let n = vram_bytes - split;
                create_slots(&mut b.tail, n, &prop, "cuMemCreate(VRAM slot tail)", hook)?;
                cu(
                    cuMemMap(b.va + split as u64, n, 0, b.tail, 0),
                    "cuMemMap(VRAM slot tail)",
                )?;
            }
            let mut at = vram_bytes as u64;
            for p in &pieces {
                cu(
                    cuMemMap(b.va + at, p.size, 0, p.handle, 0),
                    "cuMemMap(host part)",
                )?;
                at += p.size as u64;
            }
            let a = access(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
            cu(cuMemSetAccess(b.va, total, &a, 1), "cuMemSetAccess(tiered)")?;
            Ok(())
        };
        if let Err(e) = done() {
            b.release();
            return Err(e);
        }
    }
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(tiered_free),
        get_base: Some(tiered_base),
        init_tensor: None,
        memset_tensor: Some(tiered_memset),
        set_tensor: Some(tiered_set),
        get_tensor: Some(tiered_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: Some(tiered_clear),
        reset: None,
    };
    // SAFETY: ggml takes the interface and the boxed context (freed by tiered_free).
    let buf = unsafe {
        let buf = ffi::ggml_backend_buffer_init(
            tiered_type(device),
            iface,
            Box::into_raw(b) as *mut c_void,
            total,
        );
        ffi::ggml_backend_buffer_set_usage(
            buf,
            ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
        );
        buf
    };
    let buf = NonNull::new(buf).ok_or_else(last_error)?;
    Ok((buf, vram_bytes - slots))
}

// ---------------------------------------------------------------------------------------------
// The copier: asynchronous copies on a stream of their own (DMA engines: no SM time taken from
// the compute stream).

pub struct Copier {
    device: usize,
    stream: cudaStream_t,
}

// SAFETY: a stream handle; used from one thread at a time (&mut).
unsafe impl Send for Copier {}

impl Drop for Copier {
    fn drop(&mut self) {
        // SAFETY: waits for the stream, then frees it.
        unsafe {
            cudaSetDevice(self.device as c_int);
            cudaStreamSynchronize(self.stream);
            cudaStreamDestroy(self.stream);
        }
    }
}

/// Marks the copies queued before it; done once they completed.
pub struct CopyEvent(cudaEvent_t);

// SAFETY: an event handle, queried from one thread at a time.
unsafe impl Send for CopyEvent {}

impl Drop for CopyEvent {
    fn drop(&mut self) {
        // SAFETY: the event is ours.
        unsafe { cudaEventDestroy(self.0) };
    }
}

impl CopyEvent {
    /// True once the copies before it completed.
    pub fn done(&self) -> Result<bool, String> {
        // SAFETY: a live event.
        match unsafe { cudaEventQuery(self.0) } {
            CUDA_ERROR_NOT_READY => Ok(false),
            e => rt(e, "copy").map(|()| true),
        }
    }
}

impl Copier {
    pub fn new(device: usize) -> Result<Self, String> {
        use_device(device)?;
        let mut stream = null_mut();
        // SAFETY: creates a stream on the current device.
        rt(
            unsafe { cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING) },
            "cudaStreamCreate",
        )?;
        Ok(Self { device, stream })
    }

    /// Queues a copy of `bytes` from `src` to `dst`.
    ///
    /// # Safety
    /// Both ranges must be memory the device can address, valid until the copy completes, and
    /// `dst` must not be read or written by anything else meanwhile.
    pub unsafe fn copy(&mut self, dst: usize, src: usize, bytes: usize) -> Result<(), String> {
        unsafe {
            cudaSetDevice(self.device as c_int);
            rt(
                cudaMemcpyAsync(
                    dst as *mut c_void,
                    src as *const c_void,
                    bytes,
                    CUDA_MEMCPY_DEFAULT,
                    self.stream,
                ),
                "cudaMemcpyAsync",
            )
        }
    }

    pub fn mark(&mut self) -> Result<CopyEvent, String> {
        let mut e = null_mut();
        // SAFETY: creates and records an event on the stream.
        unsafe {
            cudaSetDevice(self.device as c_int);
            rt(
                cudaEventCreateWithFlags(&mut e, CUDA_EVENT_DISABLE_TIMING),
                "cudaEventCreate",
            )?;
            let ev = CopyEvent(e);
            rt(cudaEventRecord(e, self.stream), "cudaEventRecord")?;
            Ok(ev)
        }
    }

    pub fn sync(&mut self) -> Result<(), String> {
        // SAFETY: waits for the stream.
        unsafe {
            cudaSetDevice(self.device as c_int);
            rt(cudaStreamSynchronize(self.stream), "copier sync")
        }
    }

    /// Queues zeroing `bytes` at `dst`.
    ///
    /// # Safety
    /// As for [`Self::copy`].
    pub unsafe fn zero(&mut self, dst: usize, bytes: usize) -> Result<(), String> {
        unsafe {
            cudaSetDevice(self.device as c_int);
            rt(
                cudaMemsetAsync(dst as *mut c_void, 0, bytes, self.stream),
                "cudaMemsetAsync",
            )
        }
    }

    /// Later copies wait until work recorded in `e` (on any stream) is done.
    pub fn wait(&mut self, e: &Event) -> Result<(), String> {
        // SAFETY: a live stream and event.
        rt(
            unsafe { cudaStreamWaitEvent(self.stream, e.0, 0) },
            "cudaStreamWaitEvent",
        )
    }

    /// Records `e` after the copies queued so far.
    pub fn record(&mut self, e: &Event) -> Result<(), String> {
        // SAFETY: the copier's own live stream.
        unsafe { e.record(self.stream) }
    }
}

/// Thread blocks of a [`PeerPuller`]'s copy kernel: about two per V100 SM, which keeps its
/// host link busy (68 GB/s from 80 to 640 blocks of 512 threads).
const PULL_CTAS: c_int = 160;

unsafe extern "C" {
    fn legion_stage_pull(
        dst: *mut c_void,
        src: *const c_void,
        n: usize,
        ctas: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

/// A copier's helper on another GPU (`--prefill-peer-stage`): a stream on `peer` whose kernel
/// copies from host memory into `device`'s memory (peer stores over NVLink), in the order of
/// the copier's stream. `device`'s memory it writes must grant `peer` access
/// ([`set_stage_peer`]), the host memory it reads too.
pub struct PeerPuller {
    device: usize,
    peer: usize,
    stream: cudaStream_t,
    /// On `device`: recorded on the copier's stream, waited for by `stream`.
    fence: Event,
    /// On `peer`: recorded on `stream`, waited for by the copier's stream.
    done: Event,
}

// SAFETY: CUDA handles, used from one thread at a time (&mut).
unsafe impl Send for PeerPuller {}

impl Drop for PeerPuller {
    fn drop(&mut self) {
        // SAFETY: waits for the stream, then frees it; the device is put back.
        unsafe {
            cudaSetDevice(self.peer as c_int);
            cudaStreamSynchronize(self.stream);
            cudaStreamDestroy(self.stream);
            cudaSetDevice(self.device as c_int);
        }
    }
}

impl PeerPuller {
    pub fn new(device: usize, peer: usize) -> Result<Self, String> {
        let fence = Event::new(device)?;
        let done = Event::new(peer)?;
        use_device(peer)?;
        let mut stream = null_mut();
        // SAFETY: creates a stream on the current device (the peer).
        let made = rt(
            unsafe { cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING) },
            "cudaStreamCreate (peer staging)",
        );
        use_device(device)?;
        made?;
        Ok(Self {
            device,
            peer,
            stream,
            fence,
            done,
        })
    }

    /// The peer's later copies wait for what `copier` has queued so far.
    pub fn follow(&mut self, copier: &mut Copier) -> Result<(), String> {
        copier.record(&self.fence)?;
        // SAFETY: the peer's live stream may wait for another device's event.
        rt(
            unsafe { cudaStreamWaitEvent(self.stream, self.fence.0, 0) },
            "cudaStreamWaitEvent",
        )
    }

    /// Queues a copy of `bytes` from `src` to `dst` on the peer.
    ///
    /// # Safety
    /// As for [`Copier::copy`], the ranges reachable by the peer.
    pub unsafe fn pull(&mut self, dst: usize, src: usize, bytes: usize) -> Result<(), String> {
        // SAFETY: the peer's stream with the peer current; the device is put back.
        unsafe {
            rt(cudaSetDevice(self.peer as c_int), "cudaSetDevice")?;
            let e = legion_stage_pull(
                dst as *mut c_void,
                src as *const c_void,
                bytes,
                PULL_CTAS,
                self.stream,
            );
            let back = rt(cudaSetDevice(self.device as c_int), "cudaSetDevice");
            rt(e, "stage_pull").and(back)
        }
    }

    /// `copier`'s later work waits for the peer's copies queued so far.
    pub fn join(&mut self, copier: &mut Copier) -> Result<(), String> {
        // SAFETY: the peer's live stream and its own event.
        unsafe { self.done.record(self.stream)? };
        copier.wait(&self.done)
    }

    /// The peer.
    pub fn peer(&self) -> usize {
        self.peer
    }
}

/// A reusable CUDA event (no timing).
pub struct Event(cudaEvent_t);

// SAFETY: an event handle, used from one thread at a time.
unsafe impl Send for Event {}

// SAFETY: the runtime's event calls are thread-safe; `--long-dataflow-async` records an event
// on one thread and waits for it on another, and keeps a record and a wait of the same use
// apart in time (see `dataflow::hand_off`).
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: the event is ours.
        unsafe { cudaEventDestroy(self.0) };
    }
}

impl Event {
    pub fn new(device: usize) -> Result<Self, String> {
        let mut e = null_mut();
        // SAFETY: creates an event on the device.
        unsafe {
            cudaSetDevice(device as c_int);
            rt(
                cudaEventCreateWithFlags(&mut e, CUDA_EVENT_DISABLE_TIMING),
                "cudaEventCreate",
            )?;
        }
        Ok(Self(e))
    }

    /// Records the event on `stream` (a `cudaStream_t`).
    ///
    /// # Safety
    /// `stream` must be a live stream of the event's device.
    pub unsafe fn record(&self, stream: *mut c_void) -> Result<(), String> {
        rt(
            unsafe { cudaEventRecord(self.0, stream) },
            "cudaEventRecord",
        )
    }

    /// Makes `stream` wait for the work recorded in this event.
    ///
    /// # Safety
    /// As [`Self::record`].
    pub unsafe fn wait_on(&self, stream: *mut c_void) -> Result<(), String> {
        rt(
            unsafe { cudaStreamWaitEvent(stream, self.0, 0) },
            "cudaStreamWaitEvent",
        )
    }

    /// Blocks until the work last recorded in this event is done (at once if never recorded).
    pub fn synchronize(&self) -> Result<(), String> {
        // SAFETY: a live event.
        rt(
            unsafe { cudaEventSynchronize(self.0) },
            "cudaEventSynchronize",
        )
    }
}

/// Queues a copy of `bytes` from `src` to `dst` (device addresses, on one GPU or two) on
/// `stream`, a stream of `device`.
///
/// # Safety
/// Both ranges must be live device memory, `dst` untouched by anything else and `src` not
/// written until the copy is done; `stream` a live stream of `device`.
pub unsafe fn copy_async(
    device: usize,
    dst: usize,
    src: usize,
    bytes: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: the caller's ranges and stream.
    unsafe {
        rt(cudaSetDevice(device as c_int), "cudaSetDevice")?;
        rt(
            cudaMemcpyAsync(
                dst as *mut c_void,
                src as *const c_void,
                bytes,
                CUDA_MEMCPY_DEFAULT,
                stream,
            ),
            "cudaMemcpyAsync",
        )
    }
}

/// Zeroes `n` bytes at `ptr` on `stream`.
///
/// # Safety
/// `ptr..ptr + n` must be device-addressable memory that nothing else uses meanwhile.
pub unsafe fn zero_async(ptr: *mut c_void, n: usize, stream: *mut c_void) -> Result<(), String> {
    rt(
        unsafe { cudaMemsetAsync(ptr, 0, n, stream) },
        "cudaMemsetAsync",
    )
}

/// The CUDA device whose plain ggml buffer type `buft` is (`ggml_backend_cuda_buffer_type`),
/// or `None` for any other buffer type (host, pinned, split, meta, tiered, spill).
pub fn plain_cuda_device(buft: BufferType) -> Option<usize> {
    // SAFETY: ggml's static buffer type of a device index.
    (0..crate::ggml::cuda_device_count())
        .find(|&d| unsafe { ffi::ggml_backend_cuda_buffer_type(d as c_int) } == buft.raw())
}

/// Zeroes the ranges `(address, bytes)` of `device`'s memory and waits for it: what one
/// `ggml_backend_tensor_memset` per range does (a `cudaMemsetAsync` on the per-thread default
/// stream, then a synchronize of that stream), with one synchronize for them all instead of one
/// each (#836: a new sequence's state slot is ~60 ranges, ~22 us a pair).
///
/// # Safety
/// Every range must be device memory of `device` that nothing else uses meanwhile.
pub unsafe fn zero_ranges_now(device: usize, ranges: &[(usize, usize)]) -> Result<(), String> {
    // `cudaStreamPerThread` in the CUDA headers.
    let per_thread = 0x2 as cudaStream_t;
    // SAFETY: the caller's ranges; the per-thread stream of the current device.
    unsafe {
        rt(cudaSetDevice(device as c_int), "cudaSetDevice")?;
        for &(at, n) in ranges {
            rt(
                cudaMemsetAsync(at as *mut c_void, 0, n, per_thread),
                "cudaMemsetAsync",
            )?;
        }
        rt(cudaStreamSynchronize(per_thread), "cudaStreamSynchronize")
    }
}

/// VRAM allocated through VMM, to be mapped into ranges (prefill staging buffers): one or
/// more physical allocations, mapped back to back.
pub struct DeviceMem {
    parts: Vec<(CUmemGenericAllocationHandle, usize)>,
    size: usize,
    device: usize,
}

// SAFETY: allocation handles.
unsafe impl Send for DeviceMem {}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        for &(handle, size) in &self.parts {
            // SAFETY: ours; ranges that still map it keep the memory until they unmap.
            timed("cuMemRelease", self.device, size, || unsafe {
                cuMemRelease(handle)
            });
        }
    }
}

impl DeviceMem {
    /// At least `bytes` of VRAM on `device`, zeroed.
    pub fn new(device: usize, bytes: usize) -> Result<Self, String> {
        Self::from_parts(device, bytes, Vec::new())
    }

    /// At least `bytes` of VRAM on `device`, zeroed, made of `parts` (whole allocations of
    /// the same device, such as lent slots: [`unback_keep`]) and, if they fall short, one new
    /// allocation for the rest. Lent slots mapped this way cost a lend no `cuMemCreate`
    /// (#439: the driver clears new memory, ~9 ms per staging buffer on a V100).
    pub fn from_parts(device: usize, bytes: usize, parts: Vec<DeviceMem>) -> Result<Self, String> {
        use_device(device)?;
        let gran = granularity(CU_MEM_LOCATION_TYPE_DEVICE, device as i32).max(1);
        let want = round_up(bytes.max(1), gran);
        let mut me = Self {
            parts: Vec::new(),
            size: 0,
            device,
        };
        for mut p in parts {
            debug_assert_eq!(p.device, device, "parts of another device");
            me.size += p.size;
            me.parts.append(&mut p.parts);
        }
        if me.size < want {
            let size = want - me.size;
            let prop = alloc_prop(CU_MEM_LOCATION_TYPE_DEVICE, device as i32);
            let mut handle = 0;
            // SAFETY: creates an allocation.
            cu(
                timed("cuMemCreate", device, size, || unsafe {
                    cuMemCreate(&mut handle, size, &prop, 0)
                }),
                "cuMemCreate(staging)",
            )?;
            me.parts.push((handle, size));
            me.size = want;
        }
        let size = me.size;
        // Zero it once through a temporary mapping (its padding must never hold NaNs).
        let (va, _) = timed("map", device, size, || map_range(device, &me.parts))?;
        // SAFETY: the mapping just made, then torn down.
        unsafe {
            // The memset runs asynchronously: finish it before the mapping goes away.
            let r = timed("cudaMemset+sync", device, size, || {
                rt(cudaMemset(va as *mut c_void, 0, size), "cudaMemset")
                    .and_then(|()| rt(cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
            });
            timed("unmap", device, size, || {
                cuMemUnmap(va, size);
                cuMemAddressFree(va, size);
            });
            r?;
        }
        Ok(me)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Its allocations, with their sizes.
    pub fn handles(&self) -> impl Iterator<Item = (u64, usize)> + '_ {
        self.parts.iter().copied()
    }

    /// A mapping of it for `device`'s kernels and copies, until dropped.
    pub fn map(&self) -> Result<Mapping, String> {
        use_device(self.device)?;
        let (va, size) = map_range(self.device, &self.parts)?;
        Ok(Mapping {
            device: self.device,
            va,
            size,
        })
    }
}

/// A [`DeviceMem`] mapped at an address of its own ([`DeviceMem::map`]).
pub struct Mapping {
    device: usize,
    va: CUdeviceptr,
    size: usize,
}

impl Mapping {
    /// `(address, bytes)`.
    pub fn range(&self) -> (usize, usize) {
        (self.va as usize, self.size)
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the range `DeviceMem::map` reserved and mapped; the memory stays its owner's.
        unsafe {
            let _ = use_device(self.device);
            cuMemUnmap(self.va, self.size);
            cuMemAddressFree(self.va, self.size);
        }
    }
}

/// Lends out the VRAM slots of the tiered tensor `t` like [`unback`], but keeps their memory:
/// unmapped from the range and handed back (not zeroed) for [`DeviceMem::from_parts`], or
/// freed when dropped. `None` if the slots have no VRAM.
pub fn unback_keep(t: Tensor<'_>) -> Result<Option<DeviceMem>, String> {
    let b = tiered_of(t).ok_or("not a tiered tensor")?;
    // SAFETY: the tensor's own tiered buffer; no kernel reads its slots (see `unback`).
    unsafe {
        let b = &mut *b;
        let parts = b.slot_parts();
        if parts.is_empty() {
            return Ok(None);
        }
        use_device(b.device)?;
        let d = b.device;
        let n: usize = parts.iter().map(|p| p.1).sum();
        cu(
            timed("cuMemUnmap", d, n, || cuMemUnmap(b.va, n)),
            "cuMemUnmap(VRAM slots)",
        )?;
        // A kept staging buffer's allocations: that buffer still holds them.
        let parts: Vec<_> = parts.into_iter().filter(|p| !b.is_shared(p.0)).collect();
        b.vram = 0;
        b.tail = 0;
        b.hole = true;
        b.tail_hole = false;
        b.shared_head = false;
        b.shared_tail = false;
        if parts.is_empty() {
            return Ok(None);
        }
        let size = parts.iter().map(|p| p.1).sum();
        Ok(Some(DeviceMem {
            parts,
            size,
            device: d,
        }))
    }
}

/// [`unback_keep`] of the slots' tail only ([`tiered`]'s `keep`): the slots before
/// [`tiered_split`] stay. `None` if there is no tail or it is lent already.
pub fn unback_tail(t: Tensor<'_>) -> Result<Option<DeviceMem>, String> {
    let b = tiered_of(t).ok_or("not a tiered tensor")?;
    // SAFETY: the tensor's own tiered buffer; no kernel reads its tail's slots (the caller
    // points their experts' table entries elsewhere first).
    unsafe {
        let b = &mut *b;
        if b.hole || b.tail_hole || b.tail == 0 {
            return Ok(None);
        }
        use_device(b.device)?;
        let (d, n) = (b.device, b.vram_bytes - b.split);
        cu(
            timed("cuMemUnmap", d, n, || cuMemUnmap(b.va + b.split as u64, n)),
            "cuMemUnmap(VRAM slot tail)",
        )?;
        let mem = DeviceMem {
            parts: vec![(b.tail, n)],
            size: n,
            device: d,
        };
        let shared = std::mem::take(&mut b.shared_tail);
        b.tail = 0;
        b.tail_hole = true;
        // A kept staging buffer's allocation: that buffer still holds it.
        if shared {
            std::mem::forget(mem);
            return Ok(None);
        }
        Ok(Some(mem))
    }
}

/// `(first, split)` of the tiered tensor `t`: where slot 0 starts in its range and where the
/// slots' tail starts (the end of the slot part without one), in bytes from the range's start.
pub fn tiered_split(t: Tensor<'_>) -> Option<(usize, usize)> {
    let b = tiered_of(t)?;
    // SAFETY: reading the fields of the tensor's live tiered buffer.
    unsafe { Some((((*t.raw()).data as u64 - (*b).va) as usize, (*b).split)) }
}

/// Maps `parts` (whole allocations) back to back in a new range of `device`, readable and
/// writable by it. Returns the range and its size.
fn map_range(
    device: usize,
    parts: &[(CUmemGenericAllocationHandle, usize)],
) -> Result<(CUdeviceptr, usize), String> {
    let total: usize = parts.iter().map(|p| p.1).sum();
    let mut va: CUdeviceptr = 0;
    // SAFETY: reserves and maps within the new range; on failure releases it.
    unsafe {
        cu(
            cuMemAddressReserve(&mut va, total, ARENA_ALIGN, 0, 0),
            "cuMemAddressReserve",
        )?;
        let mut off = 0usize;
        let mut done = || -> Result<(), String> {
            for &(h, size) in parts {
                cu(cuMemMap(va + off as u64, size, 0, h, 0), "cuMemMap")?;
                off += size;
            }
            let a = range_access(device);
            cu(
                cuMemSetAccess(va, total, a.as_ptr(), a.len()),
                "cuMemSetAccess",
            )
        };
        if let Err(e) = done() {
            cuMemUnmap(va, total);
            cuMemAddressFree(va, total);
            return Err(e);
        }
    }
    Ok((va, total))
}

/// The tensor `tiered` (from [`tiered`]) with its host part replaced by `mem`: one range
/// mapping the same VRAM slots again, then `mem`, so expert `e` not in a slot reads
/// `mem`'s bytes at the host part's offset. Prefill copies the experts it needs into `mem`
/// and multiplies from VRAM only. The returned context owns the range (not the memory).
pub fn staged(
    device: usize,
    tiered_t: Tensor<'_>,
    mem: &DeviceMem,
    name: &str,
) -> Result<Context, GgmlError> {
    let err = |msg: String| GgmlError::Alloc {
        bytes: mem.size,
        buffer: format!("staged range on CUDA{device}: {msg}"),
    };
    // SAFETY: reading the buffer of a live tensor; a tiered buffer's context is TieredBuffer.
    let (vram, vram_bytes, base, data) = unsafe {
        let buf = (*tiered_t.raw()).buffer;
        let ok = !buf.is_null()
            && TIERED
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|t| std::ptr::eq(&t.buft, (*buf).buft));
        if !ok {
            return Err(err("not a tiered tensor".into()));
        }
        let b = &*((*buf).context as *const TieredBuffer);
        (
            b.slot_parts(),
            b.vram_bytes,
            b.va,
            (*tiered_t.raw()).data as u64,
        )
    };
    if vram.iter().map(|p| p.1).sum::<usize>() != vram_bytes {
        return Err(err("the slots are not all mapped".into()));
    }
    let mut parts = vram;
    parts.extend_from_slice(&mem.parts);
    let (va, total) = map_range(device, &parts).map_err(err)?;
    let range = Box::new(TieredBuffer {
        device,
        vram: 0,
        tail: 0,
        va,
        vram_bytes: total,
        split: total,
        host_bytes: 0,
        hole: false,
        tail_hole: false,
        shared_head: false,
        shared_tail: false,
        borrowed: false,
    });
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(tiered_free),
        get_base: Some(tiered_base),
        init_tensor: None,
        memset_tensor: Some(tiered_memset),
        set_tensor: Some(tiered_set),
        get_tensor: Some(tiered_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: None,
        reset: None,
    };
    // SAFETY: ggml takes the interface and the boxed context (freed by tiered_free, which
    // unmaps the range; it owns no allocation: vram is 0).
    let buf = unsafe {
        let buf = ffi::ggml_backend_buffer_init(
            tiered_type(device),
            iface,
            Box::into_raw(range) as *mut c_void,
            total,
        );
        ffi::ggml_backend_buffer_set_usage(
            buf,
            ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
        );
        buf
    };
    let buf = NonNull::new(buf).ok_or_else(|| err(last_error()))?;
    let ctx = Context::new(1, 0);
    let t = ctx.new_tensor(tiered_t.ty(), &tiered_t.ne()).named(name);
    // SAFETY: the same offset as in the tiered range: slots first, then the staged part.
    unsafe { ctx.adopt_buffer(t, buf, (data - base) as usize)? };
    Ok(ctx)
}

/// Drops `device`'s staged ranges ([`staged_over_hole`]) after one device synchronization, not
/// one each: off the engine thread (#450) each would wait for the step running there. Nothing
/// may read the ranges any more (the graphs that did are gone). On a failed synchronization
/// each range synchronizes as usual, and the error is returned.
pub fn free_staged(device: usize, ranges: Vec<Context>) -> Result<(), String> {
    let synced = use_device(device).and_then(|()| {
        // SAFETY: a plain runtime call.
        let e = timed("cudaDeviceSynchronize", device, 0, || unsafe {
            cudaDeviceSynchronize()
        });
        rt(e, "cudaDeviceSynchronize")
    });
    SYNCED.with(|s| s.set(synced.is_ok()));
    drop(ranges);
    SYNCED.with(|s| s.set(false));
    synced
}

/// A staged range of `n_ids` experts for packed staging, over the tiered tensor `tiered_t`
/// whose slots' tail is lent ([`unback_tail`]) or whose slots are all lent ([`unback`]): the
/// slots up to [`tiered_split`] mapped again (if they still have VRAM), then `mem` from there
/// on. Prefill copies a layer's uncached experts into `mem` back to back, at ids past the
/// kept slots (the staged table's), so `mem` holds only what the layer misses, not one entry
/// per expert. The returned context owns the range (not the memory).
pub fn staged_packed(
    device: usize,
    tiered_t: Tensor<'_>,
    mem: &DeviceMem,
    n_ids: usize,
    name: &str,
) -> Result<Context, GgmlError> {
    let err = |msg: String| GgmlError::Alloc {
        bytes: mem.size,
        buffer: format!("packed staged range on CUDA{device}: {msg}"),
    };
    let b = tiered_of(tiered_t).ok_or_else(|| err("not a tiered tensor".into()))?;
    // SAFETY: reading the fields of the tensor's live tiered buffer.
    let (head, split, first) = unsafe {
        let b = &*b;
        let head = (!b.hole && b.vram != 0).then_some((b.vram, b.split));
        (
            head,
            b.split,
            ((*tiered_t.raw()).data as u64 - b.va) as usize,
        )
    };
    let row = tiered_t.nb()[2];
    let total = split + mem.size;
    if first + n_ids * row > total {
        return Err(err(format!("{n_ids} experts do not fit")));
    }
    let mut va: CUdeviceptr = 0;
    // SAFETY: reserves a new range and maps the kept slots and `mem` into it; on failure
    // frees it.
    unsafe {
        use_device(device).map_err(err)?;
        cu(
            cuMemAddressReserve(&mut va, total, ARENA_ALIGN, 0, 0),
            "cuMemAddressReserve",
        )
        .map_err(err)?;
        let mut parts: Vec<(CUmemGenericAllocationHandle, usize)> = Vec::new();
        parts.extend(head);
        parts.extend_from_slice(&mem.parts);
        let start = if head.is_some() { 0 } else { split };
        let mut off = start;
        for &(handle, n) in &parts {
            let mapped = cu(
                timed("cuMemMap", device, n, || {
                    cuMemMap(va + off as u64, n, 0, handle, 0)
                }),
                "cuMemMap",
            );
            if let Err(e) = mapped {
                if off > start {
                    cuMemUnmap(va + start as u64, off - start);
                }
                cuMemAddressFree(va, total);
                return Err(err(e));
            }
            off += n;
        }
        let a = range_access(device);
        if let Err(e) = cu(
            cuMemSetAccess(va + start as u64, total - start, a.as_ptr(), a.len()),
            "cuMemSetAccess",
        ) {
            cuMemUnmap(va + start as u64, total - start);
            cuMemAddressFree(va, total);
            return Err(err(e));
        }
    }
    // Without the kept slots, the slot part is a hole (as in `staged_over_hole`).
    let range = Box::new(TieredBuffer {
        device,
        vram: 0,
        tail: 0,
        va,
        vram_bytes: split,
        split,
        host_bytes: mem.size,
        hole: head.is_none(),
        tail_hole: false,
        shared_head: false,
        shared_tail: false,
        borrowed: false,
    });
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(tiered_free),
        get_base: Some(tiered_base),
        init_tensor: None,
        memset_tensor: Some(tiered_memset),
        set_tensor: Some(tiered_set),
        get_tensor: Some(tiered_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: None,
        reset: None,
    };
    // SAFETY: as in `staged`.
    let buf = unsafe {
        let buf = ffi::ggml_backend_buffer_init(
            tiered_type(device),
            iface,
            Box::into_raw(range) as *mut c_void,
            total,
        );
        ffi::ggml_backend_buffer_set_usage(
            buf,
            ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
        );
        buf
    };
    let buf = NonNull::new(buf).ok_or_else(|| err(last_error()))?;
    let ne = tiered_t.ne();
    let ctx = Context::new(1, 0);
    let t = ctx
        .new_tensor(tiered_t.ty(), &[ne[0], ne[1], n_ids as i64])
        .named(name);
    // SAFETY: slot 0 where it is in the tiered range; ids past the kept slots land in `mem`.
    unsafe { ctx.adopt_buffer(t, buf, first)? };
    Ok(ctx)
}

/// A staged range kept across lends ([`super::staging::Staging::keep`]): the VA range of
/// [`staged_packed`] (`packed`) or [`staged_over_hole`] with its mappings, made once; each lend
/// only wraps it in a tensor again ([`Self::view`], no driver calls). It maps the slots' head
/// allocation (packed, while it has VRAM) and a staging buffer, which must outlive it.
pub struct StagedRange {
    range: TieredBuffer,
    /// The tiered tensor's head allocation it maps (0: none, a hole).
    head: u64,
    packed: bool,
    /// Where slot 0 starts in the range.
    first: usize,
}

// SAFETY: a VA range and allocation handles.
unsafe impl Send for StagedRange {}

/// The head allocation [`staged_packed`] maps again for the tiered buffer `b` (0: none).
fn packed_head(b: &TieredBuffer) -> u64 {
    if !b.hole && b.vram != 0 { b.vram } else { 0 }
}

/// A [`StagedRange`] over the tiered tensor `tiered_t` and `mem`, as [`staged_packed`]
/// (`packed`) or [`staged_over_hole`] map it.
pub fn staged_range(
    device: usize,
    tiered_t: Tensor<'_>,
    mem: &DeviceMem,
    packed: bool,
) -> Result<StagedRange, String> {
    let b = tiered_of(tiered_t).ok_or("not a tiered tensor")?;
    // SAFETY: reading the fields of the tensor's live tiered buffer.
    let (b, data) = unsafe { (&*b, (*tiered_t.raw()).data as u64) };
    let first = (data - b.va) as usize;
    let head = if packed { packed_head(b) } else { 0 };
    // Packed: the slots up to the split, then `mem`; else the whole slot part a hole.
    let hole = if packed { b.split } else { b.vram_bytes };
    let total = hole + mem.size;
    let mut va: CUdeviceptr = 0;
    // SAFETY: reserves a new range and maps the head and `mem` into it; on failure frees it.
    unsafe {
        use_device(device)?;
        cu(
            cuMemAddressReserve(&mut va, total, ARENA_ALIGN, 0, 0),
            "cuMemAddressReserve",
        )?;
        let mut parts: Vec<(CUmemGenericAllocationHandle, usize)> = Vec::new();
        if head != 0 {
            parts.push((head, b.split));
        }
        parts.extend_from_slice(&mem.parts);
        let start = if head != 0 { 0 } else { hole };
        let mut off = start;
        for &(handle, n) in &parts {
            let mapped = cu(
                timed("cuMemMap", device, n, || {
                    cuMemMap(va + off as u64, n, 0, handle, 0)
                }),
                "cuMemMap",
            );
            if let Err(e) = mapped {
                if off > start {
                    cuMemUnmap(va + start as u64, off - start);
                }
                cuMemAddressFree(va, total);
                return Err(e);
            }
            off += n;
        }
        let a = range_access(device);
        if let Err(e) = cu(
            cuMemSetAccess(va + start as u64, total - start, a.as_ptr(), a.len()),
            "cuMemSetAccess",
        ) {
            cuMemUnmap(va + start as u64, total - start);
            cuMemAddressFree(va, total);
            return Err(e);
        }
    }
    let range = TieredBuffer {
        device,
        vram: 0,
        tail: 0,
        va,
        vram_bytes: hole,
        split: hole,
        host_bytes: mem.size,
        hole: head == 0,
        tail_hole: false,
        shared_head: false,
        shared_tail: false,
        borrowed: false,
    };
    Ok(StagedRange {
        range,
        head,
        packed,
        first,
    })
}

impl StagedRange {
    /// Whether it still serves `tiered_t` as `packed` (the head it maps is still the slots').
    pub fn serves(&self, tiered_t: Tensor<'_>, packed: bool) -> bool {
        let Some(b) = tiered_of(tiered_t) else {
            return false;
        };
        // SAFETY: reading the fields of the tensor's live tiered buffer.
        let head = if packed {
            packed_head(unsafe { &*b })
        } else {
            0
        };
        self.packed == packed && self.head == head
    }

    /// A tensor over it as [`staged_packed`] (`n_ids` experts) or [`staged_over_hole`] (`None`)
    /// make one; the context owns a view (freeing it releases nothing).
    pub fn view(
        &self,
        tiered_t: Tensor<'_>,
        n_ids: Option<usize>,
        name: &str,
    ) -> Result<Context, GgmlError> {
        let r = &self.range;
        let total = r.vram_bytes + r.host_bytes;
        let err = |msg: String| GgmlError::Alloc {
            bytes: r.host_bytes,
            buffer: format!("kept staged range on CUDA{}: {msg}", r.device),
        };
        let mut ne = tiered_t.ne();
        if let Some(n) = n_ids {
            if self.first + n * tiered_t.nb()[2] > total {
                return Err(err(format!("{n} experts do not fit")));
            }
            ne[2] = n as i64;
        }
        let view = Box::new(TieredBuffer {
            borrowed: true,
            ..r.clone()
        });
        let iface = ffi::ggml_backend_buffer_i {
            free_buffer: Some(tiered_free),
            get_base: Some(tiered_base),
            init_tensor: None,
            memset_tensor: Some(tiered_memset),
            set_tensor: Some(tiered_set),
            get_tensor: Some(tiered_get),
            set_tensor_2d: None,
            get_tensor_2d: None,
            cpy_tensor: None,
            clear: None,
            reset: None,
        };
        // SAFETY: as in `staged`.
        let buf = unsafe {
            let buf = ffi::ggml_backend_buffer_init(
                tiered_type(r.device),
                iface,
                Box::into_raw(view) as *mut c_void,
                total,
            );
            ffi::ggml_backend_buffer_set_usage(
                buf,
                ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
            );
            buf
        };
        let buf = NonNull::new(buf).ok_or_else(|| err(last_error()))?;
        let ctx = Context::new(1, 0);
        let shape = if n_ids.is_some() { &ne[..3] } else { &ne[..] };
        let t = ctx.new_tensor(tiered_t.ty(), shape).named(name);
        // SAFETY: slot 0 where it is in the tiered range, as `staged_packed` and
        // `staged_over_hole` place it.
        unsafe { ctx.adopt_buffer(t, buf, self.first)? };
        Ok(ctx)
    }
}

impl Drop for StagedRange {
    fn drop(&mut self) {
        let r = &self.range;
        // SAFETY: no kernel may still read the range (its views are gone).
        timed("cudaDeviceSynchronize", r.device, 0, || unsafe {
            cudaSetDevice(r.device as c_int);
            cudaDeviceSynchronize();
        });
        r.release();
    }
}

/// [`staged`] over a tiered tensor whose slots are lent out ([`unback`]): the range leaves the
/// slot part unmapped and maps `mem` after it, so every expert (all read from host memory while
/// the slots are lent) reads `mem`'s bytes at the host part's offset.
pub fn staged_over_hole(
    device: usize,
    tiered_t: Tensor<'_>,
    mem: &DeviceMem,
    name: &str,
) -> Result<Context, GgmlError> {
    let err = |msg: String| GgmlError::Alloc {
        bytes: mem.size,
        buffer: format!("staged range on CUDA{device}: {msg}"),
    };
    let b = tiered_of(tiered_t).ok_or_else(|| err("not a tiered tensor".into()))?;
    // SAFETY: reading the fields of the tensor's live tiered buffer.
    let (hole, base, data) = unsafe { ((*b).vram_bytes, (*b).va, (*tiered_t.raw()).data as u64) };
    let total = hole + mem.size;
    let mut va: CUdeviceptr = 0;
    // SAFETY: reserves a new range and maps `mem` into its second part; on failure frees it.
    unsafe {
        use_device(device).map_err(err)?;
        cu(
            cuMemAddressReserve(&mut va, total, ARENA_ALIGN, 0, 0),
            "cuMemAddressReserve",
        )
        .map_err(err)?;
        // `mem`'s parts back to back after the hole.
        let mut off = hole;
        for &(handle, n) in &mem.parts {
            let mapped = cu(
                timed("cuMemMap", device, n, || {
                    cuMemMap(va + off as u64, n, 0, handle, 0)
                }),
                "cuMemMap",
            );
            if let Err(e) = mapped {
                if off > hole {
                    cuMemUnmap(va + hole as u64, off - hole);
                }
                cuMemAddressFree(va, total);
                return Err(err(e));
            }
            off += n;
        }
        let a = range_access(device);
        if let Err(e) = cu(
            cuMemSetAccess(va + hole as u64, mem.size, a.as_ptr(), a.len()),
            "cuMemSetAccess",
        ) {
            cuMemUnmap(va + hole as u64, mem.size);
            cuMemAddressFree(va, total);
            return Err(err(e));
        }
    }
    let range = Box::new(TieredBuffer {
        device,
        vram: 0,
        tail: 0,
        va,
        vram_bytes: hole,
        split: hole,
        host_bytes: mem.size,
        hole: true,
        tail_hole: false,
        shared_head: false,
        shared_tail: false,
        borrowed: false,
    });
    let iface = ffi::ggml_backend_buffer_i {
        free_buffer: Some(tiered_free),
        get_base: Some(tiered_base),
        init_tensor: None,
        memset_tensor: Some(tiered_memset),
        set_tensor: Some(tiered_set),
        get_tensor: Some(tiered_get),
        set_tensor_2d: None,
        get_tensor_2d: None,
        cpy_tensor: None,
        clear: None,
        reset: None,
    };
    // SAFETY: as in `staged`.
    let buf = unsafe {
        let buf = ffi::ggml_backend_buffer_init(
            tiered_type(device),
            iface,
            Box::into_raw(range) as *mut c_void,
            total,
        );
        ffi::ggml_backend_buffer_set_usage(
            buf,
            ffi::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
        );
        buf
    };
    let buf = NonNull::new(buf).ok_or_else(|| err(last_error()))?;
    let ctx = Context::new(1, 0);
    let t = ctx.new_tensor(tiered_t.ty(), &tiered_t.ne()).named(name);
    // SAFETY: the same offset as in the tiered range: slot part (unmapped), then `mem`.
    unsafe { ctx.adopt_buffer(t, buf, (data - base) as usize)? };
    Ok(ctx)
}

/// Measured bandwidths (GB/s) of a device reading host memory.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Probe {
    pub in_place_gbs: f64,
    pub copy_gbs: f64,
}

/// DMA copy bandwidth of `device` from `bytes` of host memory at `host` into `vram` (a VRAM
/// range of at least `chunk` bytes), in `chunk`-sized copies at pseudo-random offsets.
pub fn probe_copy(
    device: usize,
    host: usize,
    bytes: usize,
    vram: usize,
    chunk: usize,
) -> Result<f64, String> {
    let mut c = Copier::new(device)?;
    let n = 128;
    let blobs = (bytes / chunk).max(1);
    let t = std::time::Instant::now();
    for i in 0..n {
        let b = (i * 7919) % blobs;
        // SAFETY: inside the given ranges; nothing else uses `vram` meanwhile.
        unsafe { c.copy(vram, host + b * chunk, chunk)? };
    }
    c.sync()?;
    Ok((n * chunk) as f64 / t.elapsed().as_secs_f64() / 1e9)
}
