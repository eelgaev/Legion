//! A stage graph launched behind a condition the GPU decides (#827, `--mtp-pipe-rows`).
//!
//! An MTP draft pass queued before the host has read the pass before it (`mtp_pipe` in the
//! executor) should only run if every row of that pass went on: each row's pick reached the
//! confidence floor. Launched plainly, a pass the host then drops still costs its whole GPU
//! time (~1-1.5 ms on a GB10), and most rounds end on the floor. Here:
//!
//! - after a pass that stages the next one, [`decide`] queues a one-thread kernel
//!   (`src/cuda/mtp_cond.cu`) that reads the pass's pick probabilities and keeps the device's
//!   round flag up only if every row reached the floor (and the round went on up to that pass);
//!   it runs before the next pass's inputs are written, which may reuse the outputs' memory;
//! - the queued pass, launched within a [`CondScope`], runs as a [`CondExec`]: the body of a
//!   CUDA conditional IF node (CUDA 12.3+) behind a one-thread kernel that sets the node's
//!   condition from the flag, so a pass the host would drop is skipped on the GPU (a few µs).
//!
//! The body is the graph's units recorded on the stream (`cudaStreamBeginCaptureToGraph`), the
//! same units the stage graph's own captures record, so a pass that runs computes the same
//! bits. A skipped pass leaves the flag down, so the passes queued after it are skipped too,
//! whatever its stale outputs hold. Where CUDA refuses the graph the pass runs as before,
//! unconditionally (the host drops it if it should not have run).

use std::cell::Cell;
use std::os::raw::{c_int, c_void};

#[cfg(test)]
mod tests;

thread_local! {
    static COND: Cell<bool> = const { Cell::new(false) };
}

/// Makes the graphs this thread computes while it lives run behind the device's round flag
/// (where a stage graph can: a graph whose units are all captured, see `run.rs`), set by the
/// last [`decide`] on their stream; the mark before it comes back when it is dropped.
pub struct CondScope(bool);

impl CondScope {
    pub fn enter(on: bool) -> Self {
        Self(COND.with(|c| c.replace(on)))
    }
}

impl Drop for CondScope {
    fn drop(&mut self) {
        COND.with(|c| c.set(self.0));
    }
}

/// Whether the graphs computed now run behind the device's round flag ([`CondScope`]).
pub(super) fn current() -> bool {
    COND.with(Cell::get)
}

/// `cudaGraphNodeTypeConditional`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const NODE_CONDITIONAL: c_int = 0x0d;
/// `cudaGraphCondTypeIf`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const COND_IF: c_int = 0;
/// `cudaGraphCondAssignDefault`: the handle takes its default (0, skip) at every launch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const COND_ASSIGN_DEFAULT: u32 = 1;

/// `dim3`.
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

/// `cudaKernelNodeParams`.
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
struct KernelNodeParams {
    func: *const c_void,
    grid: Dim3,
    block: Dim3,
    shared: u32,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
}

/// `cudaConditionalNodeParams` (the same on CUDA 12.3-13).
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
struct ConditionalParams {
    handle: u64,
    ty: c_int,
    size: u32,
    graphs_out: *mut *mut c_void,
}

/// `cudaGraphNodeParams` with its union as the conditional member: 256 bytes.
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
struct GraphNodeParams {
    ty: c_int,
    reserved0: [c_int; 3],
    cond: ConditionalParams,
    /// The rest of the union's 29 words.
    pad: [i64; 26],
    reserved2: i64,
}

/// The decide kernel's parameter block (`legion_mtp_decide`, src/cuda/mtp_cond.cu).
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DecideArgs {
    pmax: *const f32,
    alive: *mut u32,
    p_min: f32,
    rows: u32,
    reset: u32,
    pad: u32,
}

/// The condition kernel's parameter block (`legion_mtp_cond`, src/cuda/mtp_cond.cu).
// Written for CUDA, which reads them.
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CondArgs {
    handle: u64,
    alive: *const u32,
}

#[cfg(feature = "cuda")]
pub(super) use cuda::CondExec;
#[cfg(feature = "cuda")]
pub use cuda::decide;

#[cfg(feature = "cuda")]
mod cuda {
    use std::os::raw::{c_int, c_void};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        COND_ASSIGN_DEFAULT, COND_IF, CondArgs, ConditionalParams, DecideArgs, Dim3,
        GraphNodeParams, KernelNodeParams, NODE_CONDITIONAL,
    };
    use crate::ggml::GgmlError;

    unsafe extern "C" {
        static legion_mtp_decide_kernel: *const c_void;
        static legion_mtp_cond_kernel: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGraphCreate(graph: *mut *mut c_void, flags: u32) -> c_int;
        fn cudaGraphDestroy(graph: *mut c_void) -> c_int;
        fn cudaGraphConditionalHandleCreate(
            handle: *mut u64,
            graph: *mut c_void,
            default: u32,
            flags: u32,
        ) -> c_int;
        fn cudaGraphAddKernelNode(
            node: *mut *mut c_void,
            graph: *mut c_void,
            deps: *const *mut c_void,
            n_deps: usize,
            params: *const KernelNodeParams,
        ) -> c_int;
        fn cudaStreamBeginCaptureToGraph(
            stream: *mut c_void,
            graph: *mut c_void,
            deps: *const *mut c_void,
            edges: *const c_void,
            n_deps: usize,
            mode: c_int,
        ) -> c_int;
        fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> c_int;
        fn cudaGraphInstantiateWithFlags(
            exec: *mut *mut c_void,
            graph: *mut c_void,
            flags: u64,
        ) -> c_int;
        fn cudaGraphExecDestroy(exec: *mut c_void) -> c_int;
        fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
        fn cudaRuntimeGetVersion(version: *mut c_int) -> c_int;
    }

    /// `cudaGraphAddNode` with edge data (CUDA 13's `cudaGraphAddNode`, CUDA 12.3-12.9's
    /// `cudaGraphAddNode_v2`).
    type AddNode = unsafe extern "C" fn(
        node: *mut *mut c_void,
        graph: *mut c_void,
        deps: *const *mut c_void,
        edges: *const c_void,
        n_deps: usize,
        params: *mut GraphNodeParams,
    ) -> c_int;

    /// The runtime's [`AddNode`], looked up once: `None` before CUDA 12.3.
    fn add_node() -> Option<AddNode> {
        static F: OnceLock<Option<AddNode>> = OnceLock::new();
        *F.get_or_init(|| {
            let find = |name: &std::ffi::CStr| {
                // SAFETY: a lookup in the symbols loaded already (libcudart is linked).
                let p = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
                // SAFETY: the symbol has AddNode's signature under that name (cuda_runtime_api.h).
                (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, AddNode>(p) })
            };
            if let Some(f) = find(c"cudaGraphAddNode_v2") {
                return Some(f);
            }
            let mut v = 0;
            // SAFETY: a plain query.
            let ok = unsafe { cudaRuntimeGetVersion(&mut v) } == 0;
            if ok && v >= 13000 {
                find(c"cudaGraphAddNode")
            } else {
                None
            }
        })
    }

    /// `cudaStreamCaptureModeRelaxed`, as the stage graphs capture.
    const CAPTURE_RELAXED: c_int = 2;

    const ONE: Dim3 = Dim3 { x: 1, y: 1, z: 1 };

    /// Per device: its round flag (a `u32` in its memory), 0 until made.
    static ALIVE: [AtomicUsize; 64] = [const { AtomicUsize::new(0) }; 64];

    /// The current device's round flag, made the first time.
    fn alive() -> Option<*mut u32> {
        let mut dev = 0;
        // SAFETY: plain queries and a 4-byte allocation kept for the process.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..64).contains(&dev) {
                cudaGetLastError();
                return None;
            }
            let slot = &ALIVE[dev as usize];
            let p = slot.load(Ordering::Relaxed);
            if p != 0 {
                return Some(p as *mut u32);
            }
            let mut p = std::ptr::null_mut();
            if cudaMalloc(&mut p, 4) != 0 || p.is_null() {
                cudaGetLastError();
                return None;
            }
            slot.store(p as usize, Ordering::Relaxed);
            Some(p.cast())
        }
    }

    /// Queues the round's stop check on `stream` (on the current device): the flag stays up
    /// only if it was (or `reset`: the pass ran unconditionally) and none of the `rows` F32
    /// probabilities at `pmax` is below `p_min`. False if it could not be queued.
    ///
    /// # Safety
    /// `stream` is a stream of the current device; `pmax` holds `rows` F32 values there,
    /// written by work queued on it before.
    pub unsafe fn decide(
        stream: *mut c_void,
        pmax: usize,
        rows: u32,
        p_min: f32,
        reset: bool,
    ) -> bool {
        let Some(alive) = alive() else {
            return false;
        };
        let mut args = DecideArgs {
            pmax: pmax as *const f32,
            alive,
            p_min,
            rows,
            reset: reset.into(),
            pad: 0,
        };
        let mut argv = [(&raw mut args).cast::<c_void>()];
        // SAFETY: the caller's; the runtime copies the parameter block at launch.
        let e = unsafe {
            cudaLaunchKernel(
                legion_mtp_decide_kernel,
                ONE,
                ONE,
                argv.as_mut_ptr(),
                0,
                stream,
            )
        };
        if e != 0 {
            // SAFETY: a plain call.
            unsafe { cudaGetLastError() };
        }
        e == 0
    }

    /// A graph's units as the body of a conditional node, behind the kernel that sets its
    /// condition from the round flag: one executable graph.
    pub struct CondExec {
        graph: *mut c_void,
        exec: *mut c_void,
    }

    // SAFETY: graphs and their executables are handles any thread may launch or destroy; a
    // CondExec is used by one thread at a time (its stage graph entry's).
    unsafe impl Send for CondExec {}

    impl Drop for CondExec {
        fn drop(&mut self) {
            // SAFETY: made by `build` (`exec` null where it failed first), destroyed once; a
            // launch still running is freed by CUDA when it completes.
            unsafe {
                if !self.exec.is_null() {
                    cudaGraphExecDestroy(self.exec);
                }
                cudaGraphDestroy(self.graph);
            }
        }
    }

    impl CondExec {
        /// Builds it, `record` recording the body's units on `stream` (captured into the
        /// body). `None` when CUDA refused any step (nothing ran; the caller runs the units
        /// as before).
        ///
        /// # Safety
        /// `stream` is the graph's backend's stream on the current device, capturing nothing
        /// now; `record` only queues work on it.
        pub unsafe fn build(
            stream: *mut c_void,
            record: impl FnOnce() -> Result<(), GgmlError>,
        ) -> Option<Self> {
            let add = add_node()?;
            let alive = alive()?;
            // SAFETY: the caller's; the parent graph owns its nodes, handle and body, and is
            // destroyed with `me` on failure.
            unsafe {
                let mut graph = std::ptr::null_mut();
                if cudaGraphCreate(&mut graph, 0) != 0 || graph.is_null() {
                    cudaGetLastError();
                    return None;
                }
                let mut me = Self {
                    graph,
                    exec: std::ptr::null_mut(),
                };
                if !me.fill(add, alive, stream, record) {
                    cudaGetLastError();
                    return None;
                }
                Some(me)
            }
        }

        /// The condition kernel's node, the IF node with the units as its body, and the
        /// executable.
        ///
        /// # Safety
        /// As for [`Self::build`].
        unsafe fn fill(
            &mut self,
            add: AddNode,
            alive: *mut u32,
            stream: *mut c_void,
            record: impl FnOnce() -> Result<(), GgmlError>,
        ) -> bool {
            // SAFETY: the caller's.
            unsafe {
                let mut handle = 0;
                if cudaGraphConditionalHandleCreate(&mut handle, self.graph, 0, COND_ASSIGN_DEFAULT)
                    != 0
                {
                    return false;
                }
                let mut args = CondArgs {
                    handle,
                    alive: alive.cast_const(),
                };
                let mut argv = [(&raw mut args).cast::<c_void>()];
                let kp = KernelNodeParams {
                    func: legion_mtp_cond_kernel,
                    grid: ONE,
                    block: ONE,
                    shared: 0,
                    params: argv.as_mut_ptr(),
                    extra: std::ptr::null_mut(),
                };
                let mut check = std::ptr::null_mut();
                if cudaGraphAddKernelNode(&mut check, self.graph, std::ptr::null(), 0, &kp) != 0 {
                    return false;
                }
                let mut np = GraphNodeParams {
                    ty: NODE_CONDITIONAL,
                    reserved0: [0; 3],
                    cond: ConditionalParams {
                        handle,
                        ty: COND_IF,
                        size: 1,
                        graphs_out: std::ptr::null_mut(),
                    },
                    pad: [0; 26],
                    reserved2: 0,
                };
                let mut node = std::ptr::null_mut();
                if add(&mut node, self.graph, &check, std::ptr::null(), 1, &mut np) != 0
                    || np.cond.graphs_out.is_null()
                {
                    return false;
                }
                let body = *np.cond.graphs_out;
                if cudaStreamBeginCaptureToGraph(
                    stream,
                    body,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    CAPTURE_RELAXED,
                ) != 0
                {
                    return false;
                }
                let recorded = record();
                let mut out = std::ptr::null_mut();
                let end = cudaStreamEndCapture(stream, &mut out);
                if recorded.is_err() || end != 0 || out != body {
                    return false;
                }
                let mut exec = std::ptr::null_mut();
                if cudaGraphInstantiateWithFlags(&mut exec, self.graph, 0) != 0 || exec.is_null() {
                    return false;
                }
                self.exec = exec;
                true
            }
        }

        /// Launches it on `stream` (the one it was built on): its body runs if the round flag
        /// is up. An error is CUDA's.
        ///
        /// # Safety
        /// `stream` is the stream it was built on; the [`decide`] it should follow is queued
        /// on it.
        pub unsafe fn launch(&mut self, stream: *mut c_void) -> Result<(), c_int> {
            // SAFETY: the caller's.
            match unsafe { cudaGraphLaunch(self.exec, stream) } {
                0 => Ok(()),
                e => {
                    // SAFETY: a plain call.
                    unsafe { cudaGetLastError() };
                    Err(e)
                }
            }
        }
    }
}
