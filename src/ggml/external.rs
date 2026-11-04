//! External ops: graph nodes with an implementation of our own for each backend.
//!
//! On the CPU an external op is an ordinary `GGML_OP_CUSTOM` node that ggml runs on its thread
//! pool. ggml's CUDA backend cannot run custom ops, so on CUDA [`super::Backend::compute`]
//! runs the graph in segments: the nodes up to an external op as a view of the graph (ggml's
//! own CUDA graph capture keys on the segment's first node, so a reused graph replays every
//! segment as a CUDA graph), then the op's kernel, launched on the backend's stream so it is
//! ordered after the segment without a host sync, and so on. The compute allocator plans
//! memory over the whole graph, so tensors stay valid across segments.

use std::os::raw::{c_int, c_void};

use super::context::{Context, Tensor};
use super::ffi;

/// A CUDA implementation: enqueues the op's work on `stream` (a `cudaStream_t`).
pub type CudaLaunch = unsafe fn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String>;

/// One external op: what it computes per backend. Instances are `static`s; graph nodes point
/// at them.
pub struct ExternalOp {
    pub name: &'static str,
    /// The CPU implementation, called on `nth` threads at once with `ith` in `0..nth`. Reads
    /// `dst.src[..]` and writes `dst.data` (host memory).
    pub cpu: unsafe fn(dst: &ffi::ggml_tensor, ith: usize, nth: usize),
    /// The CUDA implementation: enqueues the work on `stream` (a `cudaStream_t`); tensor data
    /// pointers are device memory. `None`: the op has no CUDA kernel.
    pub cuda: Option<CudaLaunch>,
}

unsafe extern "C" fn trampoline(
    dst: *mut ffi::ggml_tensor,
    ith: c_int,
    nth: c_int,
    op: *mut c_void,
) {
    // SAFETY: `op` is the `&'static ExternalOp` given to ggml_custom_4d in `Context::external`;
    // ggml passes the node being computed.
    unsafe {
        let op = &*(op as *const ExternalOp);
        (op.cpu)(&*dst, ith as usize, nth as usize)
    }
}

/// `ggml_custom_op_params` (ggml-impl.h): what a `GGML_OP_CUSTOM` node keeps in `op_params`.
#[repr(C)]
struct CustomParams {
    fun: ffi::ggml_custom_op_t,
    n_tasks: c_int,
    userdata: *mut c_void,
}

/// The external op computed by `node`, if it is one.
///
/// # Safety
/// `node` must point to a valid tensor.
pub(crate) unsafe fn op_of(node: *const ffi::ggml_tensor) -> Option<&'static ExternalOp> {
    unsafe {
        if (*node).op != ffi::ggml_op_GGML_OP_CUSTOM {
            return None;
        }
        // op_params is an int32 array (4-byte aligned) and CustomParams holds pointers (8-byte
        // aligned), so the header is read unaligned.
        let p = std::ptr::read_unaligned((*node).op_params.as_ptr() as *const CustomParams);
        let ours = p.fun.map(|f| f as *const () as usize) == Some(trampoline as *const () as usize);
        ours.then(|| &*(p.userdata as *const ExternalOp))
    }
}

impl Context {
    /// A node computing `op` from `args` (its `src[0..]`, at most 9: ggml_custom_4d asserts
    /// fewer than `GGML_MAX_SRC`) into a new tensor of type `ty` and shape `ne`.
    pub fn external<'a>(
        &'a self,
        op: &'static ExternalOp,
        ty: ffi::ggml_type,
        ne: [i64; 4],
        args: &[Tensor<'a>],
    ) -> Tensor<'a> {
        assert!(
            args.len() < ffi::GGML_MAX_SRC as usize,
            "{}: too many args",
            op.name
        );
        let mut raw: Vec<*mut ffi::ggml_tensor> = args.iter().map(|t| t.raw()).collect();
        // SAFETY: valid context and tensors; `op` is 'static, so the pointer ggml keeps in the
        // node stays valid.
        let t = unsafe {
            ffi::ggml_custom_4d(
                self.raw(),
                ty,
                ne[0],
                ne[1],
                ne[2],
                ne[3],
                raw.as_mut_ptr(),
                raw.len() as c_int,
                Some(trampoline),
                ffi::GGML_N_TASKS_MAX,
                op as *const ExternalOp as *mut c_void,
            )
        };
        self.wrap(t)
    }

    /// [`Self::external`] with `params` (at most [`EXTERNAL_PARAMS`]) stored in the node, for
    /// the op to read with [`external_params`]. ggml compares a node's `op_params` when it
    /// decides whether a captured CUDA graph still matches, so they may change between steps.
    pub fn external_with<'a>(
        &'a self,
        op: &'static ExternalOp,
        ty: ffi::ggml_type,
        ne: [i64; 4],
        args: &[Tensor<'a>],
        params: &[i32],
    ) -> Tensor<'a> {
        assert!(
            params.len() <= EXTERNAL_PARAMS,
            "{}: too many params",
            op.name
        );
        let t = self.external(op, ty, ne, args);
        // SAFETY: a fresh node; the params go into op_params after ggml's custom-op header,
        // within its bounds.
        unsafe {
            let p = (*t.raw()).op_params.as_mut_ptr() as *mut u8;
            let p = p.add(std::mem::size_of::<CustomParams>()) as *mut i32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), p, params.len());
        }
        t
    }
}

impl Context {
    /// [`Self::external_with`] in place: the node is a view of `a` (its `src[0]`; `args` follow
    /// it), and the op writes its result over `a`'s data, as ggml's in-place ops do. ggml's
    /// allocator gives the node no memory of its own and keeps `a`'s until the node's readers
    /// ran, which is the plan it makes for an in-place-capable ggml op (`ggml_scale`, a unary
    /// op) that takes over its only reader's parent. Only for an `a` nothing else reads.
    pub fn external_inplace<'a>(
        &'a self,
        op: &'static ExternalOp,
        a: Tensor<'a>,
        args: &[Tensor<'a>],
        params: &[i32],
    ) -> Tensor<'a> {
        assert!(
            args.len() < ffi::GGML_MAX_SRC as usize - 1,
            "{}: too many args",
            op.name
        );
        assert!(
            params.len() <= EXTERNAL_PARAMS,
            "{}: too many params",
            op.name
        );
        let mut raw: Vec<*mut ffi::ggml_tensor> = args.iter().map(|t| t.raw()).collect();
        // SAFETY: valid context and tensors; `op` is 'static, so the pointer ggml keeps in the
        // node stays valid. The params go into op_params after ggml's custom-op header, within
        // its bounds.
        let t = unsafe {
            let t = ffi::ggml_custom_inplace(
                self.raw(),
                a.raw(),
                raw.as_mut_ptr(),
                raw.len() as c_int,
                Some(trampoline),
                ffi::GGML_N_TASKS_MAX,
                op as *const ExternalOp as *mut c_void,
            );
            let p = (*t).op_params.as_mut_ptr() as *mut u8;
            let p = p.add(std::mem::size_of::<CustomParams>()) as *mut i32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), p, params.len());
            t
        };
        self.wrap(t)
    }
}

/// Turns the allocated node `node` into one of external op `op` reading `srcs` (at most 10),
/// with `params` as [`Context::external_with`] stores them: the same tensor (shape, type and
/// memory), computed by `op` instead (a post-allocation rewrite, [`super::post_alloc`]).
///
/// # Safety
/// `node` is a valid node of an allocated graph that has not run, not a view; `srcs` are valid
/// tensors of that graph.
pub(crate) unsafe fn make_external(
    node: *mut ffi::ggml_tensor,
    op: &'static ExternalOp,
    srcs: &[*mut ffi::ggml_tensor],
    params: &[i32],
) {
    assert!(
        srcs.len() <= ffi::GGML_MAX_SRC as usize && params.len() <= EXTERNAL_PARAMS,
        "{}: too many args",
        op.name
    );
    // SAFETY: the caller's; the header and params stay within op_params.
    unsafe {
        let n = &mut *node;
        n.op = ffi::ggml_op_GGML_OP_CUSTOM;
        n.op_params = [0; ffi::GGML_MAX_OP_PARAMS as usize / 4];
        let header = CustomParams {
            fun: Some(trampoline),
            n_tasks: ffi::GGML_N_TASKS_MAX,
            userdata: op as *const ExternalOp as *mut c_void,
        };
        std::ptr::write_unaligned(n.op_params.as_mut_ptr() as *mut CustomParams, header);
        let p = (n.op_params.as_mut_ptr() as *mut u8).add(std::mem::size_of::<CustomParams>());
        std::ptr::copy_nonoverlapping(params.as_ptr(), p as *mut i32, params.len());
        n.src = [std::ptr::null_mut(); ffi::GGML_MAX_SRC as usize];
        n.src[..srcs.len()].copy_from_slice(srcs);
    }
}

/// How many i32 parameters [`Context::external_with`] can store in a node.
pub const EXTERNAL_PARAMS: usize =
    (ffi::GGML_MAX_OP_PARAMS as usize - std::mem::size_of::<CustomParams>()) / 4;

/// The parameters an external op node was made with by [`Context::external_with`] (zeros past
/// those given; all zeros for [`Context::external`]).
pub fn external_params(dst: &ffi::ggml_tensor) -> [i32; EXTERNAL_PARAMS] {
    let mut out = [0; EXTERNAL_PARAMS];
    // SAFETY: reading op_params after the custom-op header, within its bounds.
    unsafe {
        let p = (dst.op_params.as_ptr() as *const u8).add(std::mem::size_of::<CustomParams>());
        std::ptr::copy_nonoverlapping(p as *const i32, out.as_mut_ptr(), EXTERNAL_PARAMS);
    }
    out
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use std::cell::Cell;
    use std::os::raw::{c_int, c_void};

    use super::super::ffi;

    thread_local! {
        static BACKEND: Cell<ffi::ggml_backend_t> = const { Cell::new(std::ptr::null_mut()) };
        static GRAPH: Cell<*mut ffi::ggml_cgraph> = const { Cell::new(std::ptr::null_mut()) };
    }

    /// The CUDA backend whose graph this thread is computing, for an op that calls into ggml-cuda
    /// itself (src/model/core/mmq_preq.rs); null outside a compute.
    pub(crate) fn current_backend() -> ffi::ggml_backend_t {
        BACKEND.with(Cell::get)
    }

    /// The graph this thread is computing (its nodes run in order on the backend's stream), for
    /// an op whose launch depends on the nodes before it (src/model/core/shared_q8/small.rs);
    /// null outside a compute.
    pub(crate) fn current_graph() -> *mut ffi::ggml_cgraph {
        GRAPH.with(Cell::get)
    }

    /// Makes `raw` this thread's [`current_backend`] and `g` its [`current_graph`] until the
    /// guard drops.
    pub(crate) struct BackendScope(ffi::ggml_backend_t, *mut ffi::ggml_cgraph);

    impl BackendScope {
        pub(crate) fn enter(raw: ffi::ggml_backend_t, g: *mut ffi::ggml_cgraph) -> Self {
            Self(
                BACKEND.with(|b| b.replace(raw)),
                GRAPH.with(|c| c.replace(g)),
            )
        }
    }

    impl Drop for BackendScope {
        fn drop(&mut self) {
            BACKEND.with(|b| b.set(self.0));
            GRAPH.with(|c| c.set(self.1));
        }
    }

    unsafe extern "C" {
        /// The stream ggml's CUDA backend computes on (src/cuda/shim.cu).
        pub fn legion_cuda_stream(backend: ffi::ggml_backend_t) -> *mut c_void;
        /// Makes the backend's device current on this thread; a CUDA error code.
        pub fn legion_cuda_set_device(backend: ffi::ggml_backend_t) -> c_int;
        /// `ggml_backend_graph_compute_async` on nodes `i0..i1` of `graph` (a ggml graph view).
        pub fn legion_cuda_compute_range(
            backend: ffi::ggml_backend_t,
            graph: *mut ffi::ggml_cgraph,
            i0: c_int,
            i1: c_int,
        ) -> c_int;
    }
}
