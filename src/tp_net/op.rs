//! The all-reduce graph op: the sum over the ranks of a partial output, every rank's added in
//! rank order (the same bits on every rank). On the CPU the op's first thread runs the exchange
//! ([`Comm::all_reduce_host`]); on CUDA [`cuda::launch`] enqueues it on the stream without a
//! host sync (see the module docs of [`super`]).

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

static ALL_REDUCE: ExternalOp = ExternalOp {
    name: "tp_all_reduce",
    cpu: cpu_all_reduce,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::launch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The sum of `x` (F32) over the ranks of communicator `comm` ([`super::register`]): a new
/// tensor of `x`'s shape.
pub fn all_reduce<'a>(ctx: &'a Context, comm: i32, x: Tensor<'a>) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "tp all-reduce: F32 only");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    ctx.external_with(&ALL_REDUCE, types::F32, x.ne(), &[x], &[comm])
}

unsafe fn cpu_all_reduce(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // One thread runs it; ggml's barrier after the node holds the others.
    if ith != 0 {
        return;
    }
    // SAFETY: a node made by `all_reduce` on the CPU: a contiguous F32 source of its own
    // shape, both with host data, which ggml gives the node for its compute.
    let (src, out) = unsafe {
        let n = ffi::ggml_nelements(dst) as usize;
        (
            std::slice::from_raw_parts((*dst.src[0]).data as *const f32, n),
            std::slice::from_raw_parts_mut(dst.data as *mut f32, n),
        )
    };
    match super::comm(external_params(dst)[0]) {
        // A failure stays in the communicator, for the step to report.
        Some(c) => {
            let _ = c.all_reduce_host(src, out);
        }
        None => {
            tracing::error!("tp all-reduce: the communicator is gone");
            out.copy_from_slice(src);
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) mod cuda {
    use std::os::raw::{c_int, c_uint, c_void};

    use crate::ggml::{external_params, ffi};
    use crate::tp_net::Wire;

    /// `CU_STREAM_WAIT_VALUE_GEQ`: `(int32_t)(*addr - value) >= 0`, so wrapping is fine.
    const WAIT_GEQ: u32 = 0;
    /// `cudaHostAllocPortable | cudaHostAllocMapped`.
    const HOST_FLAGS: c_uint = 1 | 2;
    /// `cudaMemcpyDefault` (unified addressing tells the directions apart).
    const COPY_DEFAULT: c_int = 4;
    /// The flags: one cache line each.
    const READY: usize = 0;
    const CONSUMED: usize = 64;
    const fn recv_flag(peer: usize) -> usize {
        128 + 64 * peer
    }

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `legion_tp_sum_args` of `src/cuda/tp_net.cu`.
    #[repr(C)]
    struct SumArgs {
        src: [*const f32; super::super::MAX_RANKS],
        dst: *mut f32,
        n: i64,
        n_src: i32,
    }

    /// `tp_pack_args` of `src/cuda/tp_net.cu`.
    #[repr(C)]
    struct PackArgs {
        src: *const f32,
        dst: *mut u16,
        n: i64,
    }

    /// `tp_copy_args` of `src/cuda/tp_net.cu`.
    #[repr(C)]
    struct CopyArgs {
        src: *const u16,
        dst: *mut u16,
        n: i64,
    }

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs): the F32 sum, the sum
        /// of BF16 partials (also the BF16 to F32 copy, with one source) and the BF16 rounding.
        static legion_tp_net_sum_kernel: *const c_void;
        static legion_tp_net_sum_bf16_kernel: *const c_void;
        static legion_tp_net_pack_bf16_kernel: *const c_void;
        /// The BF16 copy of an all-gather with a BF16 output (#1176).
        static legion_tp_net_copy_bf16_kernel: *const c_void;
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaHostAlloc(p: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
        fn cudaHostGetDevicePointer(d: *mut *mut c_void, h: *mut c_void, flags: c_uint) -> c_int;
        fn cudaFreeHost(p: *mut c_void) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaFree(p: *mut c_void) -> c_int;
        fn cudaMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: c_int,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
        fn cuStreamWriteValue32_v2(stream: *mut c_void, addr: u64, value: u32, flags: u32)
        -> c_int;
        fn cuStreamWaitValue32_v2(stream: *mut c_void, addr: u64, value: u32, flags: u32) -> c_int;
    }

    fn check(what: &str, e: c_int) -> Result<(), String> {
        if e == 0 {
            Ok(())
        } else {
            Err(format!("tp all-reduce: {what}: CUDA error {e}"))
        }
    }

    /// Page-locked host memory mapped for the GPU, zeroed.
    struct HostMem {
        host: usize,
        dev: u64,
    }

    impl HostMem {
        fn new(bytes: usize) -> Result<Self, String> {
            let mut h = std::ptr::null_mut();
            let mut d = std::ptr::null_mut();
            // SAFETY: plain allocation calls; zeroed before anything reads it.
            unsafe {
                check("cudaHostAlloc", cudaHostAlloc(&mut h, bytes, HOST_FLAGS))?;
                if let Err(e) = check(
                    "cudaHostGetDevicePointer",
                    cudaHostGetDevicePointer(&mut d, h, 0),
                ) {
                    cudaFreeHost(h);
                    return Err(e);
                }
                std::ptr::write_bytes(h as *mut u8, 0, bytes);
            }
            Ok(Self {
                host: h as usize,
                dev: d as u64,
            })
        }
    }

    impl Drop for HostMem {
        fn drop(&mut self) {
            // SAFETY: memory from `cudaHostAlloc`, freed once, after its users (the
            // communicator's threads are joined first; the stream was synchronized on growth).
            unsafe {
                cudaFreeHost(self.host as *mut c_void);
            }
        }
    }

    /// The sum slot of a communicator ([`crate::tp_net::seq_par::sum_slot`], #1176): device
    /// memory for two `u64`s, the addresses of the two ranks' BF16 partials of the latest
    /// deferred reduce-scatter, written on the stream by [`hand_over`] and read by its reader's
    /// kernel. Made once, at the first deferred sum, and kept at the same address while the
    /// communicator lives, so a captured reader's launch stays valid.
    pub(crate) struct SumSlot {
        dev: u64,
    }

    impl SumSlot {
        fn new() -> Result<Self, String> {
            let mut d = std::ptr::null_mut();
            // SAFETY: a plain allocation; written on the stream before anything reads it.
            check("cudaMalloc", unsafe { cudaMalloc(&mut d, 64) })?;
            Ok(Self { dev: d as u64 })
        }
    }

    impl Drop for SumSlot {
        fn drop(&mut self) {
            // SAFETY: memory from `cudaMalloc`, freed once, when the communicator goes.
            unsafe {
                cudaFree(self.dev as *mut c_void);
            }
        }
    }

    /// The device address of `comm`'s sum slot ([`SumSlot`]), once a deferred sum made it.
    pub(crate) fn sum_slot(comm: &super::super::Comm) -> Option<u64> {
        comm.sum_slot
            .lock()
            .expect("sum slot lock")
            .as_ref()
            .map(|s| s.dev)
    }

    /// The CUDA path's buffers: the flags, and per buffer set (`nbuf` of them: an exchange
    /// uses set `seq % nbuf`) one send buffer and one receive buffer per rank, `cap` bytes each.
    pub(crate) struct Pinned {
        flags: HostMem,
        /// `data` registered with the RDMA peers (dropped before `data`, declared first).
        _mrs: Vec<crate::tp_net::rdma::Mr>,
        data: HostMem,
        cap: usize,
        /// 2, or 4 with `--tp-overlap` (see [`launch_micro_start`]).
        nbuf: usize,
    }

    impl Pinned {
        fn send(&self, k: usize) -> usize {
            k * self.cap
        }
        fn recv(&self, peer: usize, k: usize) -> usize {
            (self.nbuf + self.nbuf * peer + k) * self.cap
        }
    }

    /// Enqueues the all-reduce of `dst.src[0]` into `dst` on `stream`.
    ///
    /// Sequence `s` uses buffers of parity `s % 2`. The GPU copies its partial into the send
    /// buffer and raises "ready" to `s`, which the writers wait for; it waits for every peer's
    /// flag to reach `s` (its reader filled the receive buffer), adds the partials and raises
    /// "consumed". No buffer is overwritten while in use, without more waits: the GPU reaches
    /// copy `s + 2` only after the sum `s + 1`, which needs every peer's partial `s + 1`, which
    /// a peer sends only after its own sum `s`, so after it received this rank's partial `s`;
    /// and a peer's partial `s + 2` arrives only after this rank sent partial `s + 1`, which
    /// follows its sum `s` on the stream (the readers check "consumed" anyway).
    ///
    /// The reduce-scatter and all-gather of [`crate::tp_net::seq_par`] are exchanges of the same
    /// kind, on the same buffers and sequence numbers: a rank sends each peer a share of its
    /// columns (reduce-scatter: the peer's columns; all-gather: its own) and receives one from
    /// each.
    ///
    /// # Safety
    /// `dst` is an all-reduce node with device data on `stream`'s device.
    pub(crate) unsafe fn launch(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { run(Kind::AllReduce, dst, stream) }
    }

    /// [`launch`] for a node of [`crate::tp_net::seq_par::reduce_scatter`].
    ///
    /// # Safety
    /// As [`launch`].
    pub(crate) unsafe fn launch_reduce_scatter(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { run(Kind::ReduceScatter, dst, stream) }
    }

    /// [`launch`] for a node of [`crate::tp_net::seq_par::all_gather`].
    ///
    /// # Safety
    /// As [`launch`].
    pub(crate) unsafe fn launch_all_gather(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: the caller's.
        unsafe { run(Kind::AllGather, dst, stream) }
    }

    /// The first half of one micro-batch's reduce-scatter or all-gather (a node of
    /// [`crate::tp_net::micro`]'s `*_start`): this rank's values into the send buffer, "ready"
    /// raised, the exchange queued with the writers and readers. The GPU goes on with the
    /// graph while the shares travel; [`launch_micro_finish`] waits for them.
    ///
    /// With `--tp-overlap` there are 4 buffer sets, and an exchange starts only while at most
    /// one other is in flight (started, not finished), so `s - 2` was finished before `s`
    /// started. Then nothing is overwritten in use: set `s % 4` was last used by `s - 4`,
    /// which this rank finished before it started `s - 2`; a peer sends partial `s` only after
    /// finishing `s - 2`, which needs this rank's partial `s - 2`, raised on the stream after
    /// this rank's sum `s - 4`; and this rank's finish of `s - 2` (before its copy `s`) needs
    /// the peers' partial `s - 2`, sent after they finished `s - 4`, so after they received
    /// this rank's partial `s - 4`.
    ///
    /// # Safety
    /// `dst` is a micro start node with device data on `stream`'s device.
    pub(crate) unsafe fn launch_micro_start(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let m = crate::tp_net::micro::Params::of(dst);
        let comm = super::super::comm(m.comm).ok_or("tp exchange: the communicator is gone")?;
        if let Some(e) = comm.error() {
            return Err(e);
        }
        // SAFETY: the caller's.
        let src = unsafe { &*dst.src[0] };
        let plan = micro_plan(&comm, &m, src);
        // SAFETY: the caller's.
        let r = unsafe { start(&comm, &plan, src.data as usize, stream, true, None, None) };
        if let Err(e) = &r {
            comm.fail(e.clone());
        }
        r.map(|_| ())
    }

    /// The second half of [`launch_micro_start`]'s exchange (a node of
    /// [`crate::tp_net::micro`]'s `*_finish`, the oldest one in flight): waits for the peers'
    /// shares and adds or copies them into `dst`.
    ///
    /// # Safety
    /// `dst` is a micro finish node with device data on `stream`'s device.
    pub(crate) unsafe fn launch_micro_finish(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let m = crate::tp_net::micro::Params::of(dst);
        let comm = super::super::comm(m.comm).ok_or("tp exchange: the communicator is gone")?;
        if let Some(e) = comm.error() {
            return Err(e);
        }
        // SAFETY: the caller's.
        let src = unsafe { &*dst.src[0] };
        let plan = micro_plan(&comm, &m, src);
        let seq = comm.in_flight.lock().expect("in-flight lock").pop_front();
        let r = match seq {
            // SAFETY: the caller's.
            Some(seq) => unsafe {
                finish(
                    &comm,
                    &plan,
                    seq,
                    src.data as usize,
                    dst.data as usize,
                    stream,
                )
            },
            None => Err("tp exchange: a finish without its start".into()),
        };
        if let Err(e) = &r {
            comm.fail(e.clone());
        }
        r
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        AllReduce,
        ReduceScatter,
        AllGather,
    }

    /// One exchange as this rank sees it.
    struct Plan {
        kind: Kind,
        /// Values of the tensor the buffers are sized for: an all-reduce's or reduce-scatter's
        /// source (the send buffer keeps each value at its offset there), an all-gather's output.
        span: usize,
        wire: Wire,
        /// Per rank: its piece `(first value, values)` of the source (all-reduce,
        /// reduce-scatter) or of the output (all-gather).
        pieces: Vec<(usize, usize)>,
        /// BF16: the whole source rounded into the send buffer (the plain all-reduce and
        /// reduce-scatter), not only the pieces (a micro-batch's).
        pack_whole: bool,
        /// An all-gather with a BF16 output (`seq_par::all_gather_bf16`): the BF16 values
        /// copied as they travelled, not widened to F32.
        out_bf16: bool,
        /// With `out_bf16`: the device address of the output's F32 copy that the exchange
        /// writes too ([`crate::tp_net::seq_par::take_cast`]), this rank's share before the
        /// wait for the peers'.
        widen: Option<usize>,
    }

    /// The plan of a plain exchange node.
    ///
    /// # Safety
    /// `dst` is a node of [`super::all_reduce`] or of `seq_par`'s reduce-scatter / all-gather.
    unsafe fn plain_plan(comm: &super::super::Comm, kind: Kind, dst: &ffi::ggml_tensor) -> Plan {
        // SAFETY: the caller's.
        let src = unsafe { &*dst.src[0] };
        let n_ranks = comm.n_ranks();
        // The full tensor: the same on every rank, so the type on the wire is, and the buffers
        // grow at the same exchange.
        let (row, cols) = match kind {
            // SAFETY: the caller's.
            Kind::AllReduce => unsafe { (ffi::ggml_nelements(dst) as usize, 1) },
            Kind::ReduceScatter => (src.ne[0] as usize, src.ne[1] as usize),
            Kind::AllGather => (dst.ne[0] as usize, dst.ne[1] as usize),
        };
        let full = row * cols;
        // Rank `r`'s values: (first, count); an all-reduce's are all of them.
        let pieces = (0..n_ranks)
            .map(|r| match kind {
                Kind::AllReduce => (0, full),
                _ => {
                    let (a, c) = crate::tp_net::seq_par::share(cols, r, n_ranks);
                    (a * row, c * row)
                }
            })
            .collect();
        Plan {
            kind,
            span: full,
            wire: comm.wire(full),
            pieces,
            pack_whole: true,
            out_bf16: kind == Kind::AllGather && dst.type_ == crate::ggml::types::BF16,
            widen: match kind {
                Kind::AllGather => crate::tp_net::seq_par::cast_of(dst),
                _ => None,
            },
        }
    }

    /// The plan of a micro-batch exchange whose node has the source `src`.
    fn micro_plan(
        comm: &super::super::Comm,
        m: &crate::tp_net::micro::Params,
        src: &ffi::ggml_tensor,
    ) -> Plan {
        let n_ranks = comm.n_ranks();
        let row = src.ne[0] as usize;
        let cols = crate::tp_net::micro::span_cols(m.n, n_ranks, m.j, m.compact);
        let pieces = (0..n_ranks)
            .map(|r| {
                let (a, c) = crate::tp_net::micro::piece(m.n, r, n_ranks, m.j, m.compact);
                (a * row, c * row)
            })
            .collect();
        Plan {
            kind: if m.gather {
                Kind::AllGather
            } else {
                Kind::ReduceScatter
            },
            span: row * cols,
            // As the plain exchange of the step's whole tensor.
            wire: comm.wire(row * m.n),
            pieces,
            pack_whole: false,
            out_bf16: false,
            widen: None,
        }
    }

    /// # Safety
    /// As [`launch`].
    unsafe fn run(kind: Kind, dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        let comm = super::super::comm(external_params(dst)[0])
            .ok_or("tp all-reduce: the communicator is gone")?;
        if let Some(e) = comm.error() {
            return Err(e);
        }
        // SAFETY: the caller's.
        let r = unsafe {
            let src = (*dst.src[0]).data as usize;
            let plan = plain_plan(&comm, kind, dst);
            // A reduce-scatter whose reader adds the partials itself (#1176). Its reader reads
            // them whatever this node does, so it must hand them over.
            let defer = kind == Kind::ReduceScatter && crate::tp_net::seq_par::deferred(dst);
            if defer && (plan.wire != Wire::Bf16 || comm.n_ranks() != 2) {
                let e = "tp reduce-scatter: a deferred sum needs two ranks and BF16 partials";
                comm.fail(e.to_string());
                return Err(e.to_string());
            }
            if plan.out_bf16 && plan.wire != Wire::Bf16 {
                let e = "tp all-gather: a BF16 output needs BF16 on the wire";
                comm.fail(e.to_string());
                return Err(e.to_string());
            }
            // A MoE tail this reduce-scatter runs itself (#1176): its BF16 values straight
            // into the send buffer, or with F32 on the wire into its own output first.
            let tail = match kind {
                Kind::ReduceScatter => crate::tp_net::seq_par::tail_of(dst),
                _ => None,
            };
            let tail = match tail {
                Some(t) if plan.wire != Wire::Bf16 => {
                    crate::model::core::node_fuse::launch_tail(t, None, None, stream).map(|_| None)
                }
                t => Ok(t),
            };
            // A hyper-connection mix this all-gather runs itself (#1176): its BF16 values
            // straight into the send buffer, or with F32 on the wire into its own output first.
            let mix = match kind {
                Kind::AllGather => crate::tp_net::seq_par::mix_of(dst),
                _ => None,
            };
            let mix = match mix {
                Some(m) if plan.wire != Wire::Bf16 => {
                    crate::model::qwen4::launch_mix(m, None, stream).map(|_| None)
                }
                m => Ok(m),
            };
            let started = tail.and_then(|tail| {
                mix.and_then(|mix| start(&comm, &plan, src, stream, false, tail, mix))
            });
            started.and_then(|seq| {
                if defer {
                    hand_over(&comm, &plan, seq, stream)
                } else {
                    finish(&comm, &plan, seq, src, dst.data as usize, stream)
                }
            })
        };
        if let Err(e) = &r {
            // Final: the peers' and this stream's waits are released, later all-reduces refused.
            comm.fail(e.clone());
        }
        r
    }

    /// Launches `kernel` (one of `src/cuda/tp_net.cu`'s sums) over `n` values.
    ///
    /// # Safety
    /// Device-readable sources and a device destination of `n` values each, on `stream`.
    unsafe fn sum(
        kernel: *const c_void,
        srcs: &[*const f32],
        dst: *mut f32,
        n: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut args = SumArgs {
            src: [std::ptr::null(); super::super::MAX_RANKS],
            dst,
            n: n as i64,
            n_src: srcs.len() as i32,
        };
        args.src[..srcs.len()].copy_from_slice(srcs);
        let mut argv = [(&raw mut args).cast::<c_void>()];
        // SAFETY: the caller's; the kernel reads `args` from the launch's copy.
        unsafe { launch_1d(kernel, &mut argv, n, stream) }
    }

    /// # Safety
    /// `kernel` takes `argv` and covers `n` values with a grid-stride loop.
    unsafe fn launch_1d(
        kernel: *const c_void,
        argv: &mut [*mut c_void],
        n: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        const THREADS: u32 = 256;
        let blocks = n.div_ceil(THREADS as usize).clamp(1, 1024) as u32;
        let dim = |x| Dim3 { x, y: 1, z: 1 };
        // SAFETY: the caller's.
        check("tp kernel", unsafe {
            cudaLaunchKernel(
                kernel,
                dim(blocks),
                dim(THREADS),
                argv.as_mut_ptr(),
                0,
                stream,
            )
        })
    }

    /// Copies `n` BF16 values from `src` to `dst` on `stream`.
    ///
    /// # Safety
    /// Device-readable `src`, device-writable `dst`, `n` values each.
    unsafe fn copy_bf16(src: u64, dst: usize, n: usize, stream: *mut c_void) -> Result<(), String> {
        let mut args = CopyArgs {
            src: src as *const u16,
            dst: dst as *mut u16,
            n: n as i64,
        };
        let mut argv = [(&raw mut args).cast::<c_void>()];
        // SAFETY: the caller's.
        unsafe { launch_1d(legion_tp_net_copy_bf16_kernel, &mut argv, n, stream) }
    }

    /// Rounds `n` F32 values at `src` to BF16 at `dst` on `stream`.
    ///
    /// # Safety
    /// Device-readable `src`, device-writable `dst`, `n` values each.
    unsafe fn pack(src: usize, dst: u64, n: usize, stream: *mut c_void) -> Result<(), String> {
        let mut args = PackArgs {
            src: src as *const f32,
            dst: dst as *mut u16,
            n: n as i64,
        };
        let mut argv = [(&raw mut args).cast::<c_void>()];
        // SAFETY: the caller's.
        unsafe { launch_1d(legion_tp_net_pack_bf16_kernel, &mut argv, n, stream) }
    }

    /// Whether a whole-source exchange's send buffer can be filled a rank's piece at a time:
    /// the pieces are the source's values, each once, in rank order.
    fn own_last(plan: &Plan) -> bool {
        let mut at = 0;
        for &(i, c) in &plan.pieces {
            if i != at {
                return false;
            }
            at += c;
        }
        at == plan.span
    }

    /// The first half of an exchange on a working communicator: the buffers (grown if needed),
    /// this rank's values into the send buffer, "ready" raised, the writers' and readers' jobs
    /// queued. Returns the exchange's sequence number; `micro`: it stays in flight (for
    /// [`launch_micro_finish`]). `tail` (a plain BF16 reduce-scatter's, see
    /// [`crate::tp_net::seq_par::take_tail`]): the MoE tail node whose values `src` holds,
    /// not run yet; it runs here with its values rounded straight into the send buffer. `mix`
    /// (a plain BF16 all-gather's, see [`crate::tp_net::seq_par::take_mix`]): the same for the
    /// hyper-connection mix node whose values `src` holds.
    ///
    /// # Safety
    /// `src` is the device address of the plan's source on `stream`'s device; `tail` as
    /// [`crate::model::core::node_fuse::launch_tail`]'s node, `mix` as
    /// [`crate::model::qwen4::launch_mix`]'s.
    unsafe fn start(
        comm: &super::super::Comm,
        plan: &Plan,
        src: usize,
        stream: *mut c_void,
        micro: bool,
        tail: Option<&ffi::ggml_tensor>,
        mix: Option<&ffi::ggml_tensor>,
    ) -> Result<u32, String> {
        // Only the BF16 whole-source pack below runs a tail; anywhere else its node (a no-op
        // now) would leave stale values to send. Checked before the sequence number moves.
        if tail.is_some()
            && !(plan.wire == Wire::Bf16 && plan.kind == Kind::ReduceScatter && plan.pack_whole)
        {
            return Err("tp exchange: a tail needs a BF16 whole-source reduce-scatter".into());
        }
        if mix.is_some()
            && !(plan.wire == Wire::Bf16 && plan.kind == Kind::AllGather && plan.pack_whole)
        {
            return Err("tp exchange: a mix needs a BF16 whole-source all-gather".into());
        }
        let (rank, n_ranks) = (comm.rank(), comm.n_ranks());
        let kind = plan.kind;
        let eb = plan.wire.bytes();
        let mine = plan.pieces[rank];
        let bytes = plan.span * eb;
        let seq = comm.next_seq();
        // The deferred sum before this exchange (#1176): its reader ran before it on the stream
        // (`seq_par::check_order`), so its buffers free now, ahead of this exchange's.
        let left = comm.deferred.lock().expect("deferred lock").take();
        if let Some(d) = left {
            // SAFETY: the stream the exchanges run on.
            unsafe { release_deferred(comm, d.seq, stream)? };
        }
        let mut in_flight = comm.in_flight.lock().expect("in-flight lock");
        if in_flight.len() > 1 {
            return Err(format!(
                "tp exchange {seq}: {} exchanges in flight already (at most 2)",
                in_flight.len()
            ));
        }
        let mut guard = comm.pinned.lock().expect("pinned lock");
        if guard.as_ref().is_none_or(|p| p.cap < bytes) {
            if let Some(&f) = in_flight.front() {
                return Err(format!(
                    "tp exchange {seq} needs larger buffers while exchange {f} is in flight"
                ));
            }
            let old = guard.take();
            let flags = match old {
                Some(p) => {
                    // Everything before this all-reduce is done with the old buffers.
                    // SAFETY: the backend's stream.
                    check("cudaStreamSynchronize", unsafe {
                        cudaStreamSynchronize(stream)
                    })?;
                    comm.wait_sent(seq.wrapping_sub(1))?;
                    p.flags
                }
                None => {
                    let f = HostMem::new(recv_flag(n_ranks))?;
                    let peers: Vec<usize> = (0..n_ranks)
                        .filter(|&p| p != rank)
                        .map(|p| f.host + recv_flag(p))
                        .collect();
                    comm.add_gpu_flags(&peers);
                    f
                }
            };
            let nbuf = if comm.overlap() == crate::tp_net::Overlap::Off {
                2
            } else {
                4
            };
            let cap = bytes.next_multiple_of(1 << 20);
            let size = (nbuf + nbuf * n_ranks) * cap;
            let data = HostMem::new(size)?;
            // The RDMA peers write into it directly (nothing without them).
            let mrs = comm.register_region(data.host, size, seq)?;
            *guard = Some(Pinned {
                flags,
                _mrs: mrs,
                data,
                cap,
                nbuf,
            });
        }
        let p = guard.as_ref().expect("pinned buffers");
        let k = seq as usize % p.nbuf;
        let (fh, fd, dh, dd) = (p.flags.host, p.flags.dev, p.data.host, p.data.dev);
        let send = p.send(k);
        let bf16 = plan.wire == Wire::Bf16;
        let at = |base: usize, i: usize| base + i * eb;
        // The send buffer: an all-reduce's and a reduce-scatter's values where they are in
        // `src` (a reduce-scatter's own share only when BF16, which its sum reads rounded), an
        // all-gather's own share at its start.
        // SAFETY: device tensors of the stream's device; the pinned buffers are mapped for it
        // and live as long as the communicator.
        unsafe {
            // A BF16 whole-source reduce-scatter (#1176): the peers' pieces first, "ready"
            // raised, then this rank's own piece (only its sum reads it, after the wait for the
            // peers' values), so its kernel runs while the pieces travel. The kernels work a
            // value (the tail a token) at a time: every value as the whole launch makes it.
            let split = bf16 && kind == Kind::ReduceScatter && plan.pack_whole && own_last(plan);
            let fill = |own: bool| -> Result<(), String> {
                for (r, &(i, c)) in plan.pieces.iter().enumerate() {
                    if (r == rank) != own || c == 0 {
                        continue;
                    }
                    match tail {
                        Some(t) => {
                            let e = t.ne[0] as usize;
                            crate::model::core::node_fuse::launch_tail(
                                t,
                                Some(dd + send as u64),
                                Some((i / e, c / e)),
                                stream,
                            )?;
                        }
                        None => pack(src + i * 4, dd + (send + i * eb) as u64, c, stream)?,
                    }
                }
                Ok(())
            };
            if split {
                fill(false)?;
            } else if bf16 && let Some(t) = tail {
                // The whole source as the pack below rounds it (#1176).
                crate::model::core::node_fuse::launch_tail(
                    t,
                    Some(dd + send as u64),
                    None,
                    stream,
                )?;
            } else if bf16 && let Some(m) = mix {
                // This rank's share as the pack below rounds it (#1176), the mix's F32 values
                // never written.
                crate::model::qwen4::launch_mix(m, Some(dd + send as u64), stream)?;
            } else if bf16 {
                let packs: Vec<(usize, usize)> = match kind {
                    Kind::AllGather => vec![(0, mine.1)],
                    _ if plan.pack_whole => vec![(0, plan.span)],
                    _ => plan.pieces.clone(),
                };
                for (i, c) in packs {
                    pack(src + i * 4, dd + (send + i * eb) as u64, c, stream)?;
                }
            } else {
                let copies: Vec<(usize, usize)> = match kind {
                    Kind::AllReduce => vec![(0, plan.span)],
                    Kind::AllGather => vec![(0, mine.1)],
                    Kind::ReduceScatter => (0..n_ranks)
                        .filter(|&r| r != rank)
                        .map(|r| plan.pieces[r])
                        .collect(),
                };
                for (i, c) in copies {
                    check(
                        "cudaMemcpyAsync",
                        cudaMemcpyAsync(
                            at(dh + send, i) as *mut c_void,
                            (src + i * 4) as *const c_void,
                            c * 4,
                            COPY_DEFAULT,
                            stream,
                        ),
                    )?;
                }
            }
            // Ordered after the copy (no `CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`).
            check(
                "cuStreamWriteValue32",
                cuStreamWriteValue32_v2(stream, fd + READY as u64, seq, 0),
            )?;
            if split {
                fill(true)?;
            }
        }
        comm.queue_pinned(
            (seq, seq.wrapping_sub(p.nbuf as u32)),
            |peer| match kind {
                Kind::AllReduce => (dh + send, bytes),
                Kind::ReduceScatter => {
                    let (i, c) = plan.pieces[peer];
                    (at(dh + send, i), c * eb)
                }
                Kind::AllGather => (dh + send, mine.1 * eb),
            },
            fh + READY,
            |peer| {
                let len = match kind {
                    Kind::AllReduce | Kind::ReduceScatter => mine.1 * eb,
                    Kind::AllGather => plan.pieces[peer].1 * eb,
                };
                (dh + p.recv(peer, k), fh + recv_flag(peer), len)
            },
            fh + CONSUMED,
            p.recv(rank, k),
        )?;
        if micro {
            in_flight.push_back(seq);
        }
        Ok(seq)
    }

    /// A reduce-scatter's sum handed to its reader ([`crate::tp_net::seq_par::defer_sum`]):
    /// the exchange whose buffers stay held until the next exchange starts.
    pub(crate) struct Deferred {
        seq: u32,
    }

    /// The second half of a deferred reduce-scatter `seq` ([`start`]): waits for the peer's
    /// share, then leaves the sum to the node's reader, the addresses of the two ranks' BF16
    /// partials (rank order) written to the sum slot on the stream ([`SumSlot`]); "consumed"
    /// is raised when the next exchange starts, after that reader ([`release_deferred`]). Two
    /// ranks, BF16 partials.
    ///
    /// # Safety
    /// As [`finish`].
    unsafe fn hand_over(
        comm: &super::super::Comm,
        plan: &Plan,
        seq: u32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let rank = comm.rank();
        let eb = plan.wire.bytes();
        let mine = plan.pieces[rank];
        let guard = comm.pinned.lock().expect("pinned lock");
        let p = guard.as_ref().ok_or("tp exchange: no buffers")?;
        let k = seq as usize % p.nbuf;
        let (fd, dd) = (p.flags.dev, p.data.dev);
        let own = dd + (p.send(k) + mine.0 * eb) as u64;
        let peer = 1 - rank;
        // SAFETY: as in `finish`.
        check("cuStreamWaitValue32", unsafe {
            cuStreamWaitValue32_v2(stream, fd + recv_flag(peer) as u64, seq, WAIT_GEQ)
        })?;
        let theirs = dd + p.recv(peer, k) as u64;
        let srcs = if rank == 0 {
            [own, theirs]
        } else {
            [theirs, own]
        };
        let mut slot = comm.sum_slot.lock().expect("sum slot lock");
        if slot.is_none() {
            *slot = Some(SumSlot::new()?);
        }
        let at = slot.as_ref().map_or(0, |s| s.dev);
        // Each address as two 32-bit words, low first (little-endian), ordered after the wait
        // and before the reader's kernel on the stream.
        for (i, a) in srcs.into_iter().enumerate() {
            for (j, w) in [a as u32, (a >> 32) as u32].into_iter().enumerate() {
                let addr = at + (8 * i + 4 * j) as u64;
                // SAFETY: the slot is device memory of the stream's device, live as long as the
                // communicator.
                check("cuStreamWriteValue32", unsafe {
                    cuStreamWriteValue32_v2(stream, addr, w, 0)
                })?;
            }
        }
        *comm.deferred.lock().expect("deferred lock") = Some(Deferred { seq });
        Ok(())
    }

    /// Raises "consumed" for the deferred sum `seq` on `stream`, after its reader's kernel
    /// (the next exchange's [`start`]).
    ///
    /// # Safety
    /// `stream` is the stream the exchange ran on.
    unsafe fn release_deferred(
        comm: &super::super::Comm,
        seq: u32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let guard = comm.pinned.lock().expect("pinned lock");
        let p = guard.as_ref().ok_or("tp exchange: no buffers")?;
        // SAFETY: the flags are mapped for the stream's device and live with the communicator.
        check("cuStreamWriteValue32", unsafe {
            cuStreamWriteValue32_v2(stream, p.flags.dev + CONSUMED as u64, seq, 0)
        })
    }

    /// The second half of exchange `seq` ([`start`]): waits for every peer's share, adds them
    /// in rank order (or copies them, an all-gather) into `dst`, raises "consumed".
    ///
    /// # Safety
    /// `src` and `dst` are the device addresses of the plan's source and output on `stream`'s
    /// device.
    unsafe fn finish(
        comm: &super::super::Comm,
        plan: &Plan,
        seq: u32,
        src: usize,
        dst: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let (rank, n_ranks) = (comm.rank(), comm.n_ranks());
        let kind = plan.kind;
        let eb = plan.wire.bytes();
        let mine = plan.pieces[rank];
        let guard = comm.pinned.lock().expect("pinned lock");
        let p = guard.as_ref().ok_or("tp exchange: no buffers")?;
        let k = seq as usize % p.nbuf;
        let (fd, dd) = (p.flags.dev, p.data.dev);
        let send = p.send(k);
        let bf16 = plan.wire == Wire::Bf16;
        let src_at = |i: usize| (src + i * 4) as *const f32;
        let recv_dev = |r: usize| (dd + p.recv(r, k) as u64) as *const f32;
        let send_dev = |i: usize| (dd + (send + i * eb) as u64) as *const f32;
        let dst_at = |i: usize| (dst + i * 4) as *mut f32;
        // A BF16 output: the shares as they travelled, at 2 bytes a value.
        let dst16 = |i: usize| dst + i * 2;
        // SAFETY: as in `start`.
        unsafe {
            // An all-gather's own share needs nothing from the peers.
            if kind == Kind::AllGather {
                if plan.out_bf16 {
                    copy_bf16(dd + send as u64, dst16(mine.0), mine.1, stream)?;
                    // The F32 copy of this rank's share (#1176), while the peers' travel.
                    if let Some(w) = plan.widen {
                        sum(
                            legion_tp_net_sum_bf16_kernel,
                            &[send_dev(0)],
                            (w + mine.0 * 4) as *mut f32,
                            mine.1,
                            stream,
                        )?;
                    }
                } else if bf16 {
                    let one = [send_dev(0)];
                    sum(
                        legion_tp_net_sum_bf16_kernel,
                        &one,
                        dst_at(mine.0),
                        mine.1,
                        stream,
                    )?;
                } else {
                    check(
                        "cudaMemcpyAsync",
                        cudaMemcpyAsync(
                            dst_at(mine.0).cast(),
                            src as *const c_void,
                            mine.1 * 4,
                            COPY_DEFAULT,
                            stream,
                        ),
                    )?;
                }
            }
            for peer in (0..n_ranks).filter(|&q| q != rank) {
                check(
                    "cuStreamWaitValue32",
                    cuStreamWaitValue32_v2(stream, fd + recv_flag(peer) as u64, seq, WAIT_GEQ),
                )?;
            }
            if kind == Kind::AllGather {
                for r in (0..n_ranks).filter(|&q| q != rank) {
                    let (i, c) = plan.pieces[r];
                    if plan.out_bf16 {
                        copy_bf16(recv_dev(r) as u64, dst16(i), c, stream)?;
                        if let Some(w) = plan.widen {
                            sum(
                                legion_tp_net_sum_bf16_kernel,
                                &[recv_dev(r)],
                                (w + i * 4) as *mut f32,
                                c,
                                stream,
                            )?;
                        }
                    } else if bf16 {
                        sum(
                            legion_tp_net_sum_bf16_kernel,
                            &[recv_dev(r)],
                            dst_at(i),
                            c,
                            stream,
                        )?;
                    } else {
                        check(
                            "cudaMemcpyAsync",
                            cudaMemcpyAsync(
                                dst_at(i).cast(),
                                recv_dev(r).cast(),
                                c * 4,
                                COPY_DEFAULT,
                                stream,
                            ),
                        )?;
                    }
                }
            } else {
                // Every rank's share of this rank's values, in rank order.
                let own = if bf16 {
                    send_dev(mine.0)
                } else {
                    src_at(mine.0)
                };
                let srcs: Vec<*const f32> = (0..n_ranks)
                    .map(|r| if r == rank { own } else { recv_dev(r) })
                    .collect();
                let kernel = if bf16 {
                    legion_tp_net_sum_bf16_kernel
                } else {
                    legion_tp_net_sum_kernel
                };
                sum(kernel, &srcs, dst as *mut f32, mine.1, stream)?;
            }
            check(
                "cuStreamWriteValue32",
                cuStreamWriteValue32_v2(stream, fd + CONSUMED as u64, seq, 0),
            )?;
        }
        Ok(())
    }
}
