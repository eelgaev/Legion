//! A big capture's device-to-device memcpy nodes made kernel nodes of the copy kernels
//! ([`super::super::copy_kernel`], #836), so that an update may move their pointers.
//!
//! Each such node is replaced by a kernel node with the same dependencies and dependents: added
//! after the capture's nodes, the memcpy node then destroyed. Two captures of one topology get
//! the same replacements in the same order, so CUDA pairs their nodes as before. A node is
//! left as it was when its copy is not between memory of the current device (or managed
//! memory), is not 1-D or 2-D, has edges of another kind than plain ones, or the driver lacks a
//! call this needs (the `_v2` graph calls came with CUDA 12.3).

use std::ffi::CStr;
use std::os::raw::{c_int, c_uint, c_void};
use std::sync::OnceLock;

use super::super::copy_kernel::{self, Copy2d, MODULE};
use super::sig::{self, KernelParams};

unsafe extern "C" {
    fn cudaPointerGetAttributes(attr: *mut PointerAttributes, ptr: *const c_void) -> c_int;
    fn cudaGetLastError() -> c_int;
}

/// `cudaPointerAttributes` (driver_types.h), with room to spare past its end.
#[repr(C)]
#[allow(dead_code)]
struct PointerAttributes {
    kind: c_int,
    device: c_int,
    device_pointer: *mut c_void,
    host_pointer: *mut c_void,
    spare: [u64; 8],
}

/// `cudaMemoryTypeDevice`, `cudaMemoryTypeManaged`.
const MEMORY_DEVICE: c_int = 2;
const MEMORY_MANAGED: c_int = 3;

/// `CU_MEMORYTYPE_DEVICE`, `CU_MEMORYTYPE_UNIFIED`.
const CU_DEVICE: c_uint = 2;
const CU_UNIFIED: c_uint = 4;

/// `CUDA_MEMCPY3D` (cuda.h).
#[repr(C)]
#[allow(dead_code)]
struct Memcpy3d {
    src_x: usize,
    src_y: usize,
    src_z: usize,
    src_lod: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: u64,
    src_array: *mut c_void,
    reserved0: *mut c_void,
    src_pitch: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    dst_z: usize,
    dst_lod: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: *mut c_void,
    reserved1: *mut c_void,
    dst_pitch: usize,
    dst_height: usize,
    width: usize,
    height: usize,
    depth: usize,
}

/// `CUgraphEdgeData` (cuda.h): all zero for a plain edge.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EdgeData([u8; 8]);

type Node = *mut c_void;
type GetType = unsafe extern "C" fn(node: Node, t: *mut c_uint) -> c_int;
type GetMemcpy = unsafe extern "C" fn(node: Node, p: *mut Memcpy3d) -> c_int;
type GetEdges =
    unsafe extern "C" fn(node: Node, list: *mut Node, edges: *mut EdgeData, n: *mut usize) -> c_int;
type AddKernel = unsafe extern "C" fn(
    node: *mut Node,
    graph: *mut c_void,
    deps: *const Node,
    n: usize,
    p: *const KernelParams,
) -> c_int;
type AddEdges = unsafe extern "C" fn(
    graph: *mut c_void,
    from: *const Node,
    to: *const Node,
    edges: *const EdgeData,
    n: usize,
) -> c_int;
type DestroyNode = unsafe extern "C" fn(node: Node) -> c_int;

/// The driver calls, all of them or none.
struct Driver {
    node_type: GetType,
    memcpy: GetMemcpy,
    deps: GetEdges,
    dependents: GetEdges,
    add_kernel: AddKernel,
    add_edges: AddEdges,
    destroy: DestroyNode,
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
                memcpy: std::mem::transmute::<*mut c_void, GetMemcpy>(find(
                    c"cuGraphMemcpyNodeGetParams",
                )?),
                deps: std::mem::transmute::<*mut c_void, GetEdges>(find(
                    c"cuGraphNodeGetDependencies_v2",
                )?),
                dependents: std::mem::transmute::<*mut c_void, GetEdges>(find(
                    c"cuGraphNodeGetDependentNodes_v2",
                )?),
                add_kernel: std::mem::transmute::<*mut c_void, AddKernel>(find(
                    c"cuGraphAddKernelNode_v2",
                )?),
                add_edges: std::mem::transmute::<*mut c_void, AddEdges>(find(
                    c"cuGraphAddDependencies_v2",
                )?),
                destroy: std::mem::transmute::<*mut c_void, DestroyNode>(find(
                    c"cuGraphDestroyNode",
                )?),
            })
        }
    })
    .as_ref()
}

/// Whether `ptr` is memory of device `dev` (or managed memory) by its attributes.
fn on_device(ptr: u64, dev: c_int) -> bool {
    // SAFETY: zeroed plain data, written by CUDA.
    let mut a: PointerAttributes = unsafe { std::mem::zeroed() };
    // SAFETY: an attribute query of an address; an unknown one is an error, cleared.
    if unsafe { cudaPointerGetAttributes(&mut a, ptr as *const c_void) } != 0 {
        // SAFETY: clears the query's error.
        unsafe { cudaGetLastError() };
        return false;
    }
    a.kind == MEMORY_MANAGED || (a.kind == MEMORY_DEVICE && a.device == dev)
}

/// Memcpy node parameters `p` as a copy between memory of device `dev`, if they are one.
fn as_copy(p: &Memcpy3d, dev: c_int) -> Option<Copy2d> {
    let typed = |t: c_uint| t == CU_DEVICE || t == CU_UNIFIED;
    let flat = p.src_z == 0
        && p.src_lod == 0
        && p.dst_z == 0
        && p.dst_lod == 0
        && p.depth == 1
        && p.src_array.is_null()
        && p.dst_array.is_null();
    if !typed(p.src_type) || !typed(p.dst_type) || !flat {
        return None;
    }
    let src = p.src_device + (p.src_y * p.src_pitch + p.src_x) as u64;
    let dst = p.dst_device + (p.dst_y * p.dst_pitch + p.dst_x) as u64;
    (on_device(src, dev) && on_device(dst, dev)).then_some(Copy2d {
        dst,
        dpitch: p.dst_pitch as u64,
        src,
        spitch: p.src_pitch as u64,
        width: p.width as u64,
        rows: p.height as u64,
    })
}

/// Node `n`'s edges (`deps`: its dependencies, else its dependents), if all are plain ones.
///
/// # Safety
/// `n` is a node of a live graph.
unsafe fn edges(d: &Driver, n: Node, deps: bool) -> Option<Vec<Node>> {
    let get = if deps { d.deps } else { d.dependents };
    let mut count = 0usize;
    // SAFETY: the caller's; a count query, then a list of that many with their edge data.
    unsafe {
        if get(n, std::ptr::null_mut(), std::ptr::null_mut(), &mut count) != 0 {
            return None;
        }
        let mut list = vec![std::ptr::null_mut(); count];
        let mut data = vec![EdgeData::default(); count];
        if count > 0 && get(n, list.as_mut_ptr(), data.as_mut_ptr(), &mut count) != 0 {
            return None;
        }
        list.truncate(count);
        data.iter()
            .take(count)
            .all(|e| e.0 == [0; 8])
            .then_some(list)
    }
}

/// Replaces memcpy node `n` of `graph` by a kernel node for `copy`; whether it did.
///
/// # Safety
/// `n` is a node of `graph`, a live graph of the current device, not being launched.
unsafe fn replace(d: &Driver, graph: *mut c_void, n: Node, copy: Copy2d) -> bool {
    let Some(launch) = copy_kernel::plan(copy) else {
        return false;
    };
    // SAFETY: the caller's.
    let (Some(deps), Some(dependents)) =
        (unsafe { edges(d, n, true) }, unsafe { edges(d, n, false) })
    else {
        return false;
    };
    let Ok(f) = crate::ptx::function(&MODULE, launch.entry) else {
        return false;
    };
    let mut values = [
        copy.dst,
        copy.dpitch,
        copy.src,
        copy.spitch,
        launch.n,
        copy.rows,
    ];
    let mut args: Vec<*mut c_void> = values
        .iter_mut()
        .map(|v| v as *mut u64 as *mut c_void)
        .collect();
    let p = KernelParams {
        func: f.raw(),
        grid: launch.grid,
        block: launch.block,
        smem: 0,
        params: args.as_mut_ptr(),
        extra: std::ptr::null_mut(),
        kern: std::ptr::null_mut(),
        ctx: std::ptr::null_mut(),
        spare: [0; 8],
    };
    let mut k: Node = std::ptr::null_mut();
    // SAFETY: the caller's graph; the node's own dependencies; parameters the kernel takes
    // (six u64), copied by the driver.
    unsafe {
        if (d.add_kernel)(&mut k, graph, deps.as_ptr(), deps.len(), &p) != 0 || k.is_null() {
            return false;
        }
        let from = vec![k; dependents.len()];
        if !dependents.is_empty()
            && (d.add_edges)(
                graph,
                from.as_ptr(),
                dependents.as_ptr(),
                std::ptr::null(),
                dependents.len(),
            ) != 0
        {
            (d.destroy)(k);
            return false;
        }
        if (d.destroy)(n) != 0 {
            // Both copy the same bytes; the kernel's node goes, the graph is as captured.
            (d.destroy)(k);
            return false;
        }
    }
    true
}

/// Memcpy node `n`'s parameters, briefly (for a refused update's log line).
///
/// # Safety
/// `n` is a memcpy node of a live graph.
pub(super) unsafe fn describe(n: Node) -> String {
    let Some(d) = driver() else {
        return String::new();
    };
    // SAFETY: zeroed plain data, written by the driver.
    let mut p: Memcpy3d = unsafe { std::mem::zeroed() };
    // SAFETY: the caller's.
    if unsafe { (d.memcpy)(n, &mut p) } != 0 {
        return String::new();
    }
    format!(
        "types {}->{} width {} height {} depth {} pitches {}->{} offsets {},{}->{},{}",
        p.src_type,
        p.dst_type,
        p.width,
        p.height,
        p.depth,
        p.src_pitch,
        p.dst_pitch,
        p.src_x,
        p.src_y,
        p.dst_x,
        p.dst_y
    )
}

/// Makes `graph`'s device-to-device memcpy nodes kernel nodes: how many it made, and how many
/// memcpy nodes it left.
///
/// # Safety
/// `graph` is a live graph of device `dev`, the current one, captured by this thread and not
/// instantiated yet.
pub(super) unsafe fn convert(graph: *mut c_void, dev: c_int) -> (usize, usize) {
    let Some(d) = driver() else {
        return (0, 0);
    };
    let (mut made, mut left) = (0, 0);
    // SAFETY: the caller's.
    for n in unsafe { sig::nodes(graph) } {
        let mut t = 0;
        // SAFETY: a node of the caller's graph.
        if unsafe { (d.node_type)(n, &mut t) } != 0 || t != 1 {
            continue;
        }
        // SAFETY: zeroed plain data, written by the driver.
        let mut p: Memcpy3d = unsafe { std::mem::zeroed() };
        // SAFETY: a memcpy node of the caller's graph.
        let copy = (unsafe { (d.memcpy)(n, &mut p) } == 0)
            .then(|| as_copy(&p, dev))
            .flatten();
        // SAFETY: the caller's graph and its node.
        match copy {
            Some(c) if unsafe { replace(d, graph, n, c) } => made += 1,
            _ => left += 1,
        }
    }
    // SAFETY: clears an error a refused driver call may have left for the runtime.
    unsafe { cudaGetLastError() };
    (made, left)
}
