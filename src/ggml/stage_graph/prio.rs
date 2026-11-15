//! Kernel priorities inside a stage graph's capture (#689 round 13, an A/B probe,
//! `LEGION_DEBUG_SIDE_PRIO=1`, off by default).
//!
//! A kernel forked onto a side stream (the DeltaNet z projection beside the layer's small
//! kernels, `l2_prefetch::on_side`) queues all its blocks at once; at equal priority the GPU
//! hands out a later kernel's blocks only once the earlier one has none left to place, so the
//! small kernels on the compute stream waited for z's last wave (measured on a GB10: the conv
//! started 117 us into z's 134). With the switch on, each kernel node captured on the side
//! stream is noted ([`mark_side`]), and once the capture ends every other kernel node of that
//! graph gets the device's greatest priority, the graph instantiated with
//! `cudaGraphInstantiateFlagUseNodePriority`: a freed slot goes to the compute stream's blocks
//! first. Priorities order the blocks only, so no output changes. Captures with no side
//! kernel are left as they are.

use std::sync::OnceLock;

#[cfg(test)]
mod tests;

/// Whether the switch is on (read once).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| switch_from(std::env::var("LEGION_DEBUG_SIDE_PRIO").ok().as_deref()))
}

/// [`enabled`] for the switch's value `v`.
pub fn switch_from(v: Option<&str>) -> bool {
    v.is_some_and(|v| v.trim() == "1")
}

#[cfg(feature = "cuda")]
pub(crate) use cuda::mark_side;
#[cfg(feature = "cuda")]
pub(super) use cuda::{INSTANTIATE_FLAG, apply, begin};

#[cfg(feature = "cuda")]
mod cuda {
    use std::cell::RefCell;
    use std::os::raw::{c_int, c_void};
    use std::sync::OnceLock;

    unsafe extern "C" {
        fn cudaGraphGetNodes(graph: *mut c_void, nodes: *mut *mut c_void, num: *mut usize)
        -> c_int;
        fn cudaGraphNodeGetType(node: *mut c_void, t: *mut c_int) -> c_int;
        fn cudaGraphKernelNodeSetAttribute(
            node: *mut c_void,
            attr: c_int,
            value: *const u32,
        ) -> c_int;
        fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    /// `cudaGraphInstantiateFlagUseNodePriority`.
    pub(in super::super) const INSTANTIATE_FLAG: u64 = 8;

    /// `cudaGraphNodeTypeKernel`.
    const NODE_KERNEL: c_int = 0;

    /// `cudaLaunchAttributePriority` (`cudaKernelNodeAttributePriority`).
    const ATTR_PRIORITY: c_int = 8;

    /// `CU_STREAM_CAPTURE_STATUS_ACTIVE`.
    const CAPTURE_ACTIVE: c_int = 1;

    /// `cuStreamGetCaptureInfo_v2` (driver API, cuda.h): stream, status, id, graph,
    /// dependencies, their count.
    type CaptureInfo = unsafe extern "C" fn(
        *mut c_void,
        *mut c_int,
        *mut u64,
        *mut *mut c_void,
        *mut *const *mut c_void,
        *mut usize,
    ) -> c_int;

    fn capture_info() -> Option<CaptureInfo> {
        static F: OnceLock<Option<CaptureInfo>> = OnceLock::new();
        *F.get_or_init(|| {
            // SAFETY: a lookup in the symbols loaded already (the driver is, once a stream is).
            let p =
                unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"cuStreamGetCaptureInfo_v2".as_ptr()) };
            // SAFETY: the symbol has this signature under that name (cuda.h).
            (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, CaptureInfo>(p) })
        })
    }

    thread_local! {
        /// The side-stream kernel nodes of this thread's capture under way.
        static LOW: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    }

    /// Forgets the nodes of an earlier capture (one CUDA refused, say): called as a capture
    /// begins.
    pub(in super::super) fn begin() {
        if super::enabled() {
            LOW.with(|l| l.borrow_mut().clear());
        }
    }

    /// Notes the node(s) side stream `side` just captured, the ones its next op would depend
    /// on; nothing outside a capture or with the switch off.
    pub(crate) fn mark_side(side: *mut c_void) {
        if !super::enabled() {
            return;
        }
        let Some(info) = capture_info() else { return };
        let (mut status, mut id, mut graph) = (0, 0u64, std::ptr::null_mut());
        let (mut deps, mut n) = (std::ptr::null(), 0usize);
        // SAFETY: a live stream; out pointers; the dependencies stay valid until the stream's
        // next capture call, and are copied right away.
        unsafe {
            if info(side, &mut status, &mut id, &mut graph, &mut deps, &mut n) != 0
                || status != CAPTURE_ACTIVE
                || deps.is_null()
            {
                return;
            }
            let nodes = std::slice::from_raw_parts(deps, n);
            LOW.with(|l| l.borrow_mut().extend(nodes.iter().map(|&p| p as usize)));
        }
    }

    /// Gives every kernel node of captured `graph` but the side stream's ones ([`mark_side`])
    /// the greatest priority; whether it did (the graph is then to be instantiated with
    /// [`INSTANTIATE_FLAG`]). Nothing without side nodes.
    ///
    /// # Safety
    /// `graph` is a live CUDA graph of the current device, this thread's last capture.
    pub(in super::super) unsafe fn apply(graph: *mut c_void) -> bool {
        let low = LOW.with(|l| std::mem::take(&mut *l.borrow_mut()));
        if low.is_empty() {
            return false;
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        // SAFETY: the caller's; out pointers.
        unsafe {
            let (mut least, mut greatest) = (0, 0);
            if cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) != 0 || greatest == least
            {
                cudaGetLastError();
                return false;
            }
            let mut n = 0usize;
            if cudaGraphGetNodes(graph, std::ptr::null_mut(), &mut n) != 0 {
                cudaGetLastError();
                return false;
            }
            let mut nodes = vec![std::ptr::null_mut(); n];
            if cudaGraphGetNodes(graph, nodes.as_mut_ptr(), &mut n) != 0 {
                cudaGetLastError();
                return false;
            }
            nodes.truncate(n);
            // cudaLaunchAttributeValue: a 64-byte union, the priority an int at its start.
            let mut val = [0u32; 16];
            val[0] = greatest as u32;
            let (mut high, mut kept) = (0usize, 0usize);
            for &node in &nodes {
                let mut t = -1;
                if cudaGraphNodeGetType(node, &mut t) != 0 || t != NODE_KERNEL {
                    continue;
                }
                if low.contains(&(node as usize)) {
                    kept += 1;
                    continue;
                }
                if cudaGraphKernelNodeSetAttribute(node, ATTR_PRIORITY, val.as_ptr()) != 0 {
                    cudaGetLastError();
                    return false;
                }
                high += 1;
            }
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    "stage graph priorities (LEGION_DEBUG_SIDE_PRIO): {high} kernel nodes at {greatest}, \
                     {kept} side-stream nodes left at {least}, of {} marked",
                    low.len()
                );
            }
            kept > 0
        }
    }
}
