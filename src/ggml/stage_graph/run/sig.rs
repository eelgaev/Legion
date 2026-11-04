//! A captured graph's node list as [`super::super::topo`] compares it, and a refused update's
//! error node, read through the CUDA driver (#836; only while `--stage-graph-stats` logs).
//!
//! The driver calls are looked up by name in the loaded libraries (libcuda is linked), so a
//! driver without one (`cuFuncGetName` and `cuKernelGetName` came with CUDA 12.3) only names
//! fewer kernels.

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::{LazyLock, Mutex, OnceLock};

use super::super::topo::{self, ErrorNode};

unsafe extern "C" {
    fn cudaGraphGetNodes(graph: *mut c_void, nodes: *mut *mut c_void, num: *mut usize) -> c_int;
    fn cudaGetLastError() -> c_int;
}

/// `CUDA_KERNEL_NODE_PARAMS_v2` (cuda.h, CUDA 12+), with room to spare past its end.
#[repr(C)]
pub(super) struct KernelParams {
    pub func: *mut c_void,
    pub grid: [c_uint; 3],
    pub block: [c_uint; 3],
    pub smem: c_uint,
    pub params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: *mut c_void,
    pub ctx: *mut c_void,
    pub spare: [u64; 8],
}

/// `CUDA_MEMSET_NODE_PARAMS` (cuda.h), with room to spare past its end.
#[repr(C)]
struct MemsetParams {
    dst: u64,
    pitch: usize,
    value: c_uint,
    element: c_uint,
    width: usize,
    height: usize,
    spare: [u64; 4],
}

type GetType = unsafe extern "C" fn(node: *mut c_void, t: *mut c_uint) -> c_int;
type GetKernel = unsafe extern "C" fn(node: *mut c_void, p: *mut KernelParams) -> c_int;
type GetMemset = unsafe extern "C" fn(node: *mut c_void, p: *mut MemsetParams) -> c_int;
type GetName = unsafe extern "C" fn(name: *mut *const c_char, f: *mut c_void) -> c_int;

/// The driver calls, `None` where the driver has none.
struct Driver {
    node_type: Option<GetType>,
    kernel: Option<GetKernel>,
    memset: Option<GetMemset>,
    func_name: Option<GetName>,
    kernel_name: Option<GetName>,
}

fn driver() -> &'static Driver {
    static D: OnceLock<Driver> = OnceLock::new();
    D.get_or_init(|| {
        let find = |name: &CStr| {
            // SAFETY: a lookup in the symbols loaded already.
            let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
            (!p.is_null()).then_some(p)
        };
        // SAFETY: each symbol has the signature it is cast to under that name (cuda.h).
        unsafe {
            Driver {
                node_type: find(c"cuGraphNodeGetType")
                    .map(|p| std::mem::transmute::<*mut c_void, GetType>(p)),
                kernel: find(c"cuGraphKernelNodeGetParams_v2")
                    .map(|p| std::mem::transmute::<*mut c_void, GetKernel>(p)),
                memset: find(c"cuGraphMemsetNodeGetParams")
                    .map(|p| std::mem::transmute::<*mut c_void, GetMemset>(p)),
                func_name: find(c"cuFuncGetName")
                    .map(|p| std::mem::transmute::<*mut c_void, GetName>(p)),
                kernel_name: find(c"cuKernelGetName")
                    .map(|p| std::mem::transmute::<*mut c_void, GetName>(p)),
            }
        }
    })
}

/// Kernel names by function (or kernel) handle.
static NAMES: LazyLock<Mutex<HashMap<usize, String>>> = LazyLock::new(Mutex::default);

/// `graph`'s nodes, in the order CUDA lists them (the capture's order).
///
/// # Safety
/// `graph` is a live CUDA graph.
pub(super) unsafe fn nodes(graph: *mut c_void) -> Vec<*mut c_void> {
    let mut n = 0usize;
    // SAFETY: the caller's; a count query, then a list of that many.
    unsafe {
        if cudaGraphGetNodes(graph, std::ptr::null_mut(), &mut n) != 0 {
            cudaGetLastError();
            return Vec::new();
        }
        let mut v = vec![std::ptr::null_mut(); n];
        if cudaGraphGetNodes(graph, v.as_mut_ptr(), &mut n) != 0 {
            cudaGetLastError();
            return Vec::new();
        }
        v.truncate(n);
        v
    }
}

/// The name of the kernel `p` launches (`?` when the driver cannot tell).
fn kernel_name(d: &Driver, p: &KernelParams) -> String {
    let (h, get) = if !p.func.is_null() {
        (p.func, d.func_name)
    } else {
        (p.kern, d.kernel_name)
    };
    if h.is_null() {
        return "?".into();
    }
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = names.get(&(h as usize)) {
        return s.clone();
    }
    let mut s: *const c_char = std::ptr::null();
    // SAFETY: a name query of a function or kernel handle the driver gave out; the string is
    // the driver's.
    let name = match get {
        Some(f) if unsafe { f(&mut s, h) } == 0 && !s.is_null() => {
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        }
        _ => format!("{h:p}"),
    };
    names.insert(h as usize, name.clone());
    name
}

/// Node `n`'s kind (`type` or `kernel:<name>`) and its parameters, briefly (`params`: only
/// then).
///
/// # Safety
/// `n` is a node of a live CUDA graph.
unsafe fn describe(d: &Driver, n: *mut c_void, params: bool) -> (String, String) {
    let Some(get_type) = d.node_type else {
        return ("?".into(), String::new());
    };
    let mut t = 0;
    // SAFETY: the caller's.
    if unsafe { get_type(n, &mut t) } != 0 {
        return ("?".into(), String::new());
    }
    let ty = topo::node_type_name(t);
    match (t, d.kernel, d.memset) {
        (0, Some(get), _) => {
            // SAFETY: zeroed plain data, written by the driver.
            let mut p: KernelParams = unsafe { std::mem::zeroed() };
            // SAFETY: the caller's; a kernel node.
            if unsafe { get(n, &mut p) } != 0 {
                return (ty.into(), String::new());
            }
            let kind = format!("kernel:{}", kernel_name(d, &p));
            let text = if params {
                format!("grid {:?} block {:?} smem {}", p.grid, p.block, p.smem)
            } else {
                String::new()
            };
            (kind, text)
        }
        (2, _, Some(get)) if params => {
            // SAFETY: zeroed plain data, written by the driver.
            let mut p: MemsetParams = unsafe { std::mem::zeroed() };
            // SAFETY: the caller's; a memset node.
            if unsafe { get(n, &mut p) } != 0 {
                return (ty.into(), String::new());
            }
            let text = format!(
                "width {} height {} element {} pitch {}",
                p.width, p.height, p.element, p.pitch
            );
            (ty.into(), text)
        }
        // SAFETY: the caller's; a memcpy node.
        (1, _, _) if params => (ty.into(), unsafe { super::copy_node::describe(n) }),
        _ => (ty.into(), String::new()),
    }
}

/// `graph`'s node list as kind ids ([`topo::kind_id`]).
///
/// # Safety
/// `graph` is a live CUDA graph.
pub(super) unsafe fn of(graph: *mut c_void) -> Vec<u32> {
    let d = driver();
    // SAFETY: the caller's.
    let list = unsafe { nodes(graph) };
    let mut ids: HashMap<String, u32> = HashMap::new();
    list.into_iter()
        .map(|n| {
            // SAFETY: a node of the caller's graph.
            let (kind, _) = unsafe { describe(d, n, false) };
            *ids.entry(kind).or_insert_with_key(|k| topo::kind_id(k))
        })
        .collect()
}

/// A refused update's error node (`node`, and `from`: the mismatched dependency) found in
/// `graph`, the capture the update was given.
///
/// # Safety
/// `graph` is a live CUDA graph; `node` and `from` are null or nodes the update named.
pub(super) unsafe fn error_node(
    graph: *mut c_void,
    node: *mut c_void,
    from: *mut c_void,
) -> ErrorNode {
    // SAFETY: the caller's.
    let list = unsafe { nodes(graph) };
    let index = |x: *mut c_void| {
        (!x.is_null())
            .then(|| list.iter().position(|&n| n == x))
            .flatten()
    };
    let mut out = ErrorNode {
        index: index(node),
        from: index(from),
        ..Default::default()
    };
    if !node.is_null() {
        // SAFETY: a node of the update's graph (or of the exec's, which outlives this call).
        (out.kind, out.params) = unsafe { describe(driver(), node, true) };
    }
    out
}
