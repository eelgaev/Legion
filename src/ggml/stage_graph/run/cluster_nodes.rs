//! Where a big capture's kernels on thread-block clusters are ([`super::super::spare::cluster_key`],
//! #1132), read through the CUDA driver: each kernel node's cluster dimension attribute.
//!
//! Only devices of compute capability 9.0 or newer launch clusters; on older ones (the AC922's
//! V100s) no node is read and every capture's key is 0, as before.

use std::ffi::CStr;
use std::os::raw::{c_int, c_uint, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use super::sig;

unsafe extern "C" {
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    fn cudaGetLastError() -> c_int;
}

/// `cudaDevAttrComputeCapabilityMajor`.
const ATTR_CC_MAJOR: c_int = 75;

/// `CU_GRAPH_NODE_TYPE_KERNEL`.
const NODE_KERNEL: c_uint = 0;

/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` (`CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION`).
const ATTR_CLUSTER_DIMENSION: c_int = 4;

type Node = *mut c_void;
type GetType = unsafe extern "C" fn(node: Node, t: *mut c_uint) -> c_int;
/// `cuGraphKernelNodeGetAttribute`: the value is a `CUlaunchAttributeValue`, a 64-byte union
/// whose `clusterDim` is its first three `unsigned int`s.
type GetAttr = unsafe extern "C" fn(node: Node, attr: c_int, value: *mut [c_uint; 16]) -> c_int;

/// The driver calls, both or none.
struct Driver {
    node_type: GetType,
    attr: GetAttr,
}

fn driver() -> Option<&'static Driver> {
    static D: OnceLock<Option<Driver>> = OnceLock::new();
    D.get_or_init(|| {
        let find = |name: &CStr| {
            // SAFETY: a lookup in the symbols loaded already.
            let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
            (!p.is_null()).then_some(p)
        };
        // SAFETY: each symbol has the signature it is cast to under that name (cuda.h).
        unsafe {
            Some(Driver {
                node_type: std::mem::transmute::<*mut c_void, GetType>(find(
                    c"cuGraphNodeGetType",
                )?),
                attr: std::mem::transmute::<*mut c_void, GetAttr>(find(
                    c"cuGraphKernelNodeGetAttribute",
                )?),
            })
        }
    })
    .as_ref()
}

/// Per device: 0 not checked yet, 1 it launches clusters (compute capability 9.0+), 2 not.
static CLUSTERS: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];

/// Whether device `dev` may run kernels on clusters (compute capability 9.0 or newer).
fn has_clusters(dev: c_int) -> bool {
    let Some(slot) = usize::try_from(dev).ok().and_then(|d| CLUSTERS.get(d)) else {
        return false;
    };
    match slot.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let mut major = 0;
    // SAFETY: a plain query of device `dev`.
    let ok = unsafe { cudaDeviceGetAttribute(&mut major, ATTR_CC_MAJOR, dev) } == 0;
    if !ok {
        // SAFETY: clears the query's error.
        unsafe { cudaGetLastError() };
    }
    let on = ok && major >= 9;
    slot.store(if on { 1 } else { 2 }, Ordering::Relaxed);
    on
}

/// [`super::super::spare::cluster_key`] of `graph`: its kernel nodes' cluster dimensions by
/// node index (the capture's order, as CUDA pairs the nodes of an update). 0 on a device
/// without clusters or a driver without the calls.
///
/// # Safety
/// `graph` is a live graph of device `dev`, captured by this thread and not launched.
pub(super) unsafe fn key(graph: *mut c_void, dev: c_int) -> u64 {
    if !has_clusters(dev) {
        return 0;
    }
    let Some(d) = driver() else {
        return 0;
    };
    // SAFETY: the caller's.
    let list = unsafe { sig::nodes(graph) };
    let dims: Vec<(usize, [c_uint; 3])> = list
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| {
            let mut t = 0;
            // SAFETY: a node of the caller's graph.
            if unsafe { (d.node_type)(n, &mut t) } != 0 || t != NODE_KERNEL {
                return None;
            }
            let mut v: [c_uint; 16] = [0; 16];
            // SAFETY: a kernel node of the caller's graph; the value is written by the driver.
            let ok = unsafe { (d.attr)(n, ATTR_CLUSTER_DIMENSION, &mut v) } == 0;
            let dims = [v[0], v[1], v[2]];
            (ok && dims.iter().map(|&x| x.max(1)).product::<c_uint>() > 1).then_some((i, dims))
        })
        .collect();
    // SAFETY: clears an error a refused driver call may have left for the runtime.
    unsafe { cudaGetLastError() };
    let k = super::super::spare::cluster_key(dims.iter().copied());
    tracing::debug!(
        nodes = list.len(),
        clustered = dims.len(),
        key = k,
        "stage graph capture: kernels on clusters"
    );
    k
}
