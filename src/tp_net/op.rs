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
    ctx.external_with(&ALL_REDUCE, types::F32, x.ne(), &[x], &[comm, 0])
}

/// `add(all_reduce(x), res)` in one op (#1133 r2): the residual (F32, `x`'s shape) added to
/// every value's sum in the sum's kernel, after the partials. The same bits as the two ops (the
/// sum in rank order, then one addition), without the add's pass over the sum.
pub fn all_reduce_add<'a>(
    ctx: &'a Context,
    comm: i32,
    x: Tensor<'a>,
    res: Tensor<'a>,
) -> Tensor<'a> {
    assert_eq!(x.ty(), types::F32, "tp all-reduce: F32 only");
    assert_eq!(res.ty(), types::F32, "tp all-reduce: an F32 residual");
    assert_eq!(res.ne(), x.ne(), "tp all-reduce: the residual's shape");
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let res = if res.is_contiguous() {
        res
    } else {
        ctx.cont(res)
    };
    ctx.external_with(&ALL_REDUCE, types::F32, x.ne(), &[x, res], &[comm, 1])
}

/// The next layer's input norm, `mul(rms_norm(y, eps), w)`, that a finish of an all-reduce
/// with a residual computes from its sums `y` in the same pass (#1133): the packed per-token FP8
/// activations of [`crate::model::fp8_act::rms_norm`] (and with `bf16` the BF16 rows of
/// [`crate::model::fp8_act::rms_norm_bf16`]), the bits of those ops on `y`, without reading `y`
/// back. GPUs only ([`crate::model::fp8_act::rms_norm_fits`]); the CPU's are references.
#[derive(Clone, Copy)]
pub struct Norm<'a> {
    pub w: Tensor<'a>,
    pub eps: f32,
    pub bf16: bool,
}

impl Norm<'_> {
    /// Its params: the eps bits and the flags ([`NORM_ON`], [`NORM_BF16`]).
    pub(super) fn params(&self) -> [i32; 2] {
        let flags = NORM_ON | if self.bf16 { NORM_BF16 } else { 0 };
        [self.eps.to_bits() as i32, flags]
    }
}

/// A finish's [`Norm`] flags: on, and the BF16 rows too.
pub(super) const NORM_ON: i32 = 1;
pub(super) const NORM_BF16: i32 = 2;

/// A node's [`Norm`] as the kernels read it: the weight's address, eps, BF16 rows.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy)]
pub(crate) struct NormParams {
    pub w: usize,
    pub eps: f32,
    pub bf16: bool,
}

impl NormParams {
    /// The norm of params `[eps bits, flags]` (`None` without [`NORM_ON`]), its weight `w`.
    pub(super) fn of(p: [i32; 2], w: impl FnOnce() -> usize) -> Option<Self> {
        (p[1] & NORM_ON != 0).then(|| Self {
            w: w(),
            eps: f32::from_bits(p[0] as u32),
            bf16: p[1] & NORM_BF16 != 0,
        })
    }
}

/// The byte offsets in a normed finish's node of `row` values x `width` tokens: the sums
/// (F32) first, then the BF16 rows (with `bf16`), then the packed FP8 activations
/// ([`crate::model::fp8::dynamic_layout`]): `(BF16 rows, packed activations, total)`.
pub(crate) fn norm_layout(row: usize, width: usize, bf16: bool) -> (usize, usize, usize) {
    let y = 4 * row * width;
    let b = if bf16 { 2 * row * width } else { 0 };
    let (_, q) = crate::model::fp8::dynamic_layout(row as i64, width as i64);
    (y, y + b, y + b + q)
}

/// The tokens of a normed finish's node of `bytes` bytes and rows of `row` values
/// ([`norm_layout`]).
pub(crate) fn norm_width(bytes: usize, row: usize, bf16: bool) -> usize {
    bytes / (row * (5 + if bf16 { 2 } else { 0 }) + 4)
}

/// What a normed finish's node holds, as views: the sums `y` (F32 `[row, width]`), the packed
/// activations `q` and the BF16 rows `b` (with [`Norm::bf16`]).
#[derive(Clone, Copy)]
pub struct Normed<'a> {
    pub y: Tensor<'a>,
    pub q: Tensor<'a>,
    pub b: Option<Tensor<'a>>,
}

/// The views of `node` (a normed finish of `row` x `width`, the last one to write it).
pub fn normed<'a>(
    ctx: &'a Context,
    node: Tensor<'a>,
    row: i64,
    width: i64,
    bf16: bool,
) -> Normed<'a> {
    let (b_at, q_at, total) = norm_layout(row as usize, width as usize, bf16);
    let vals = row * width;
    let y = ctx.reinterpret(
        ctx.view_1d(node, 4 * vals, 0),
        types::F32,
        [row, width, 1, 1],
    );
    let q = ctx.view_1d(node, (total - q_at) as i64, q_at);
    let b = bf16.then(|| {
        ctx.reinterpret(
            ctx.view_1d(node, 2 * vals, b_at),
            types::BF16,
            [row, width, 1, 1],
        )
    });
    Normed { y, q, b }
}

/// [`all_reduce_add`] with `norm` computed from the sums in the same kernel ([`Norm`]): `x`
/// and `res` F32 `[row, n]`.
pub fn all_reduce_add_norm<'a>(
    ctx: &'a Context,
    comm: i32,
    x: Tensor<'a>,
    res: Tensor<'a>,
    norm: Norm<'a>,
) -> Normed<'a> {
    let [row, n, a, b] = x.ne();
    assert_eq!((a, b), (1, 1), "tp all-reduce: a 2-D tensor to norm");
    assert_eq!(x.ty(), types::F32, "tp all-reduce: F32 only");
    assert_eq!(res.ty(), types::F32, "tp all-reduce: an F32 residual");
    assert_eq!(res.ne(), x.ne(), "tp all-reduce: the residual's shape");
    assert_eq!(
        norm.w.elements(),
        row as usize,
        "tp all-reduce: the norm's weight"
    );
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let res = if res.is_contiguous() {
        res
    } else {
        ctx.cont(res)
    };
    let (_, _, total) = norm_layout(row as usize, n as usize, norm.bf16);
    let [eps, flags] = norm.params();
    let node = ctx.external_with(
        &ALL_REDUCE,
        types::I8,
        [total as i64, 1, 1, 1],
        &[x, res, norm.w],
        &[comm, 1, eps, flags],
    );
    normed(ctx, node, row, n, norm.bf16)
}

/// `out[i] += res[i]`: the residual of [`all_reduce_add`] and `micro::finish_add` on the CPU,
/// ggml's add of the sum and the residual.
pub(super) fn add_host(out: &mut [f32], res: &[f32]) {
    for (o, r) in out.iter_mut().zip(res) {
        *o += *r;
    }
}

/// [`all_reduce`] (with `res`: [`all_reduce_add`]) that also reads the first bytes of the
/// weights `ws` (each with its bytes, whole [`crate::model::core::l2_prefetch::CHUNK`]s, from
/// [`crate::model::core::l2_prefetch::tp::plan`]) into L2 on the side stream while it waits
/// for the peers: on CUDA, launched after this rank's partial is on its way and before the
/// wait. The sum is [`all_reduce`]'s, bit for bit; the CPU ignores `ws`.
pub fn all_reduce_ahead<'a>(
    ctx: &'a Context,
    comm: i32,
    x: Tensor<'a>,
    res: Option<Tensor<'a>>,
    ws: &[(Tensor<'a>, usize)],
) -> Tensor<'a> {
    use crate::model::core::l2_prefetch::{CHUNK, tp::MAX_WEIGHTS};
    assert_eq!(x.ty(), types::F32, "tp all-reduce: F32 only");
    assert!(
        ws.len() <= MAX_WEIGHTS,
        "tp all-reduce: too many weights ahead"
    );
    let x = if x.is_contiguous() { x } else { ctx.cont(x) };
    let mut srcs = vec![x];
    let mut params = vec![comm, i32::from(res.is_some()), ws.len() as i32];
    if let Some(res) = res {
        assert_eq!(res.ty(), types::F32, "tp all-reduce: an F32 residual");
        assert_eq!(res.ne(), x.ne(), "tp all-reduce: the residual's shape");
        srcs.push(if res.is_contiguous() {
            res
        } else {
            ctx.cont(res)
        });
    }
    for &(w, bytes) in ws {
        srcs.push(w);
        params.push((bytes / CHUNK) as i32);
    }
    ctx.external_with(&ALL_REDUCE, types::F32, x.ne(), &srcs, &params)
}

unsafe fn cpu_all_reduce(dst: &ffi::ggml_tensor, ith: usize, _nth: usize) {
    // One thread runs it; ggml's barrier after the node holds the others.
    if ith != 0 {
        return;
    }
    let p = external_params(dst);
    // SAFETY: a node made by `all_reduce`, `all_reduce_add` or `all_reduce_add_norm` on the
    // CPU: a contiguous F32 source (and residual) of the sums' shape, all with host data, which
    // ggml gives the node for its compute.
    let (src, out, res) = unsafe {
        let n = ffi::ggml_nelements(dst.src[0]) as usize;
        (
            std::slice::from_raw_parts((*dst.src[0]).data as *const f32, n),
            std::slice::from_raw_parts_mut(dst.data as *mut f32, n),
            (p[1] != 0).then(|| std::slice::from_raw_parts((*dst.src[1]).data as *const f32, n)),
        )
    };
    // SAFETY: `all_reduce_add_norm`'s weight is its source 2.
    let norm = NormParams::of([p[2], p[3]], || unsafe { (*dst.src[2]).data as usize });
    match super::comm(p[0]) {
        // A failure stays in the communicator, for the step to report.
        Some(c) => {
            let _ = c.all_reduce_host(src, out);
        }
        None => {
            tracing::error!("tp all-reduce: the communicator is gone");
            out.copy_from_slice(src);
        }
    }
    if let Some(res) = res {
        add_host(out, res);
    }
    if let Some(nm) = norm {
        // SAFETY: as above; the weight has a row of values and the node holds `norm_layout`'s
        // bytes.
        unsafe {
            let src = &*dst.src[0];
            let (row, cols) = (src.ne[0] as usize, src.ne[1] as usize);
            norm_host(out, &nm, row, (0, cols), dst.data as *mut u8);
        }
    }
}

/// A normed finish's norm on the CPU: of the sums `ys` (tokens `t0..`) of a node of `width`
/// tokens of `row` values at `base` ([`norm_layout`]).
///
/// # Safety
/// `nm.w` holds `row` host F32 values; `base` the node's bytes.
pub(super) unsafe fn norm_host(
    ys: &[f32],
    nm: &NormParams,
    row: usize,
    (t0, width): (usize, usize),
    base: *mut u8,
) {
    // SAFETY: the caller's.
    unsafe {
        let w = std::slice::from_raw_parts(nm.w as *const f32, row);
        let aux = base.add(4 * row * width);
        crate::model::fp8_act::rms_norm_host(ys, w, nm.eps, row, (t0, width), aux, nm.bf16);
    }
}

#[cfg(test)]
mod tests;

/// Whether the CUDA exchanges copy an F32 partial into the pinned send buffer with the copy
/// kernel (the values' bits as 16-bit words, the BF16 all-gather's kernel) rather than
/// `cudaMemcpyAsync`, for a `LEGION_DEBUG_TP_SEND_KERNEL` value (#1176): "0" the copy (as
/// before), anything else (or unset) the kernel. The same bytes either way. On a GB10 a 16-user
/// decode sum's ~480 KiB took ~12 us on the copy engine and the sum after it waited ~22 us at
/// the least; by the kernel ~4 us and ~6 us.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn send_kernel_of(v: Option<&str>) -> bool {
    v.is_none_or(|s| s.trim() != "0")
}

/// Values the buffers of a micro all-reduce of `cols` of a step's `n` columns (`row` values
/// each) are reserved for (#619): the larger of its own and the rest of the step's. A later part
/// of the step (an odd step's second half has one more column than the first) then never needs
/// larger buffers while this one is in flight.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn reduce_reserve(row: usize, cols: usize, n: usize) -> usize {
    row * cols.max(n.saturating_sub(cols))
}

/// The pinned buffers an exchange of `bytes` starts on, with `want` bytes reserved
/// ([`reduce_reserve`], else `bytes`) and `sets` buffer sets: `have` (`(cap, sets)`, none yet:
/// `None`) as it is (`Ok(None)`), or grown to `Ok(Some((cap, sets)))` (`cap` in whole MiB, never
/// smaller than before). `busy`: another exchange is in flight, so the buffers grow only if this
/// one cannot start without (then `Err`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn buffers_for(
    have: Option<(usize, usize)>,
    bytes: usize,
    want: usize,
    sets: usize,
    busy: bool,
) -> Result<Option<(usize, usize)>, ()> {
    let short = have.is_none_or(|(cap, n)| cap < bytes || n < sets);
    if short && busy {
        return Err(());
    }
    if !short && (busy || have.is_some_and(|(cap, _)| cap >= want)) {
        return Ok(None);
    }
    let (old_cap, old_sets) = have.unwrap_or((0, 2));
    let cap = bytes.max(want).next_multiple_of(1 << 20).max(old_cap);
    Ok(Some((cap, sets.max(old_sets))))
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
        res: *const f32,
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
    /// With micro-batch exchanges there are 4 buffer sets, and an exchange starts only while at most
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
        // SAFETY: the caller's; a start the send-buffer rewrite marked runs its matmul's
        // columns, whose inputs the graph's order keeps (`micro::send_gemm::take`).
        let r = unsafe {
            let gemm = crate::tp_net::micro::send_gemm::taken(dst);
            let src = src.data as usize;
            start(&comm, &plan, src, stream, true, None, None, gemm.as_ref())
        };
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
        // An all-reduce's residual and the column it writes from (`micro::finish_add`).
        let (res, at) = crate::tp_net::micro::add_of(dst);
        // SAFETY: the caller's; `finish_add` sizes the output for the columns from `at`.
        let row = src.ne[0] as usize;
        let out = dst.data as usize + at * row * 4;
        // And its norm (`micro::finish_add_norm`), for its tokens from `at`.
        let norm = crate::tp_net::micro::norm_of(dst)
            .map(|(nm, width)| norm_at(&nm, dst.data as usize, row, width, at));
        // This rank's values in the send buffer only: its start ran their matmul there (#1133).
        // SAFETY: the caller's: a finish's source 1 is its start node.
        let sent = plan.wire == Wire::F32
            && plan.kind == Kind::AllReduce
            && unsafe { crate::tp_net::micro::send_gemm::taken(&*dst.src[1]) }.is_some();
        let seq = comm.in_flight.lock().expect("in-flight lock").pop_front();
        let r = match seq {
            // SAFETY: the caller's.
            Some(seq) => unsafe {
                let s = src.data as usize;
                finish(&comm, &plan, seq, s, out, res, sent, norm.as_ref(), stream)
            },
            None => Err("tp exchange: a finish without its start".into()),
        };
        if let Err(e) = &r {
            comm.fail(e.clone());
        }
        r
    }

    /// Where a normed finish's kernel writes ([`super::Norm`]): the norm's weight and eps, the
    /// row size, and the piece's sums, packed activations and BF16 rows.
    struct NormAt {
        w: *const f32,
        eps: f32,
        row: usize,
        out: crate::model::fp8_act::SumNormOut,
    }

    /// [`NormAt`] of the tokens from `at` in a normed node at `base` of `width` tokens of `row`
    /// values ([`super::norm_layout`]).
    fn norm_at(nm: &super::NormParams, base: usize, row: usize, width: usize, at: usize) -> NormAt {
        let (b_at, q_at, _) = super::norm_layout(row, width, nm.bf16);
        NormAt {
            w: nm.w as *const f32,
            eps: nm.eps,
            row,
            out: crate::model::fp8_act::SumNormOut {
                xs: (base + 4 * row * at) as *mut f32,
                out: (base + q_at + row * at) as *mut u8,
                scales: (base + q_at + row * width + 4 * at) as *mut f32,
                bf: if nm.bf16 {
                    (base + b_at + 2 * row * at) as *mut u16
                } else {
                    std::ptr::null_mut()
                },
            },
        }
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
        /// Values the buffers are reserved for when they grow, at least `span`
        /// ([`super::reduce_reserve`] for a micro all-reduce).
        reserve: usize,
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
            // The source's values (a normed node's output holds more, `all_reduce_add_norm`).
            // SAFETY: the caller's.
            Kind::AllReduce => unsafe { (ffi::ggml_nelements(src) as usize, 1) },
            Kind::ReduceScatter => (src.ne[0] as usize, src.ne[1] as usize),
            Kind::AllGather => (dst.ne[0] as usize, dst.ne[1] as usize),
        };
        let full = row * cols;
        // An I32 all-gather's shares may be cut elsewhere (`seq_par::share_cut`, #560 r10).
        let cut = match kind {
            Kind::AllGather => crate::tp_net::seq_par::cut_of(dst),
            _ => 0,
        };
        // Rank `r`'s values: (first, count); an all-reduce's are all of them.
        let pieces = (0..n_ranks)
            .map(|r| match kind {
                Kind::AllReduce => (0, full),
                _ => {
                    let (a, c) = crate::tp_net::seq_par::share_cut(cols, r, n_ranks, cut);
                    (a * row, c * row)
                }
            })
            .collect();
        // An I32 all-gather (`seq_par::all_gather_i32`) copies its 4-byte values as they are:
        // the F32 wire's plain copies, whatever `--tp-wire` says.
        let exact = kind == Kind::AllGather && dst.type_ == crate::ggml::types::I32;
        Plan {
            kind,
            span: full,
            wire: if exact { Wire::F32 } else { comm.wire(full) },
            pieces,
            pack_whole: true,
            reserve: full,
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
        if m.reduce {
            // All of the source, on every rank, as the plain all-reduce's (whole-source pack).
            let span = row * src.ne[1] as usize;
            return Plan {
                kind: Kind::AllReduce,
                span,
                wire: comm.wire(row * m.n),
                pieces: vec![(0, span); n_ranks],
                pack_whole: true,
                // An odd step's second half is one column larger: buffers for it from the
                // first half on, as they cannot grow while the first half is in flight.
                reserve: super::reduce_reserve(row, src.ne[1] as usize, m.n),
                out_bf16: false,
                widen: None,
            };
        }
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
            reserve: row * cols,
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
                mix.and_then(|mix| start(&comm, &plan, src, stream, false, tail, mix, None))
            });
            // An all-reduce's residual ([`super::all_reduce_add`]).
            let res = (kind == Kind::AllReduce && external_params(dst)[1] != 0)
                .then(|| (*dst.src[1]).data as usize);
            // And its norm ([`super::all_reduce_add_norm`]).
            let norm = match kind {
                Kind::AllReduce if res.is_some() => {
                    let p = external_params(dst);
                    super::NormParams::of([p[2], p[3]], || (*dst.src[2]).data as usize)
                }
                _ => None,
            }
            .map(|nm| {
                let s = &*dst.src[0];
                let (row, width) = (s.ne[0] as usize, s.ne[1] as usize);
                norm_at(&nm, dst.data as usize, row, width, 0)
            });
            started.and_then(|seq| {
                // Weights an `all_reduce_ahead` reads into L2 while it waits: forked now, with
                // this rank's partial already queued to go.
                if kind == Kind::AllReduce && external_params(dst)[2] > 0 {
                    let first = 1 + usize::from(res.is_some());
                    crate::model::core::l2_prefetch::tp::launch_beside(dst, first, stream)?;
                }
                if defer {
                    hand_over(&comm, &plan, seq, stream)
                } else {
                    let d = dst.data as usize;
                    finish(&comm, &plan, seq, src, d, res, false, norm.as_ref(), stream)
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
        // SAFETY: the caller's.
        unsafe { sum_add(kernel, srcs, None, dst, n, stream) }
    }

    /// [`sum`] with `res` (`n` device-readable values) added to every sum, after the partials.
    ///
    /// # Safety
    /// As [`sum`].
    unsafe fn sum_add(
        kernel: *const c_void,
        srcs: &[*const f32],
        res: Option<*const f32>,
        dst: *mut f32,
        n: usize,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let mut args = SumArgs {
            src: [std::ptr::null(); super::super::MAX_RANKS],
            dst,
            res: res.unwrap_or(std::ptr::null()),
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

    /// Whether an F32 exchange's partial goes into the send buffer by the copy kernel instead
    /// of `cudaMemcpyAsync` (`LEGION_DEBUG_TP_SEND_KERNEL`, see [`super::send_kernel_of`]).
    fn send_kernel() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            super::send_kernel_of(std::env::var("LEGION_DEBUG_TP_SEND_KERNEL").ok().as_deref())
        })
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
    /// [`crate::model::qwen4::launch_mix`]'s; `gemm` as [`crate::tp_net::micro::send_gemm`]'s
    /// `Taken::launch`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn start(
        comm: &super::super::Comm,
        plan: &Plan,
        src: usize,
        stream: *mut c_void,
        micro: bool,
        tail: Option<&ffi::ggml_tensor>,
        mix: Option<&ffi::ggml_tensor>,
        gemm: Option<&crate::tp_net::micro::send_gemm::Taken<'_>>,
    ) -> Result<u32, String> {
        // The matmul columns this exchange runs (#1133): with F32 on the wire straight into the
        // send buffer, else (the rewrite saw F32) into their own output first, sent from there.
        let gemm = match gemm {
            Some(g) if plan.wire == Wire::F32 && plan.kind == Kind::AllReduce => {
                if g.values != plan.span {
                    return Err(format!(
                        "tp exchange: a matmul of {} values for a sum of {}",
                        g.values, plan.span
                    ));
                }
                Some(g)
            }
            Some(g) => {
                // SAFETY: the caller's (`launch_micro_start`).
                unsafe { g.launch(None, stream)? };
                None
            }
            None => None,
        };
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
        // A micro-batch exchange needs 4 buffer sets (see `launch_micro_start`): with
        // `--tp-overlap` from the first exchange, else (a model's own micro-batches, #619) from
        // the first micro-batch exchange, which then grows the buffers as a larger one would.
        let sets = if micro || comm.overlap() != crate::tp_net::Overlap::Off {
            4
        } else {
            2
        };
        let have = guard.as_ref().map(|p| (p.cap, p.nbuf));
        let want = plan.reserve.max(plan.span) * eb;
        let grow =
            super::buffers_for(have, bytes, want, sets, !in_flight.is_empty()).map_err(|()| {
                let f = in_flight.front().copied().unwrap_or_default();
                format!("tp exchange {seq} needs larger buffers while exchange {f} is in flight")
            })?;
        if let Some((cap, nbuf)) = grow {
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
            } else if let Some(g) = gemm {
                // This rank's partial made in the send buffer, never in `src` (#1133).
                g.launch(Some(dd + send as u64), stream)?;
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
                    if send_kernel() {
                        // The F32 values' bits as twice as many 16-bit words.
                        copy_bf16(
                            (src + i * 4) as u64,
                            (dd + (send + i * eb) as u64) as usize,
                            2 * c,
                            stream,
                        )?;
                        continue;
                    }
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
    /// in rank order (or copies them, an all-gather) into `dst`, raises "consumed". `res`: an
    /// all-reduce's residual (#1133 r2), added to every sum after the partials.
    ///
    /// # Safety
    /// `src` and `dst` are the device addresses of the plan's source and output on `stream`'s
    /// device, `res` of the source's values to add (an all-reduce's only). `sent`: the start
    /// made this rank's F32 values in the send buffer only (#1133), where the sum reads them.
    /// `norm`: the sums' norm too, in the sum's kernel (an all-reduce with a residual; `dst` is
    /// its `xs`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn finish(
        comm: &super::super::Comm,
        plan: &Plan,
        seq: u32,
        src: usize,
        dst: usize,
        res: Option<usize>,
        sent: bool,
        norm: Option<&NormAt>,
        stream: *mut c_void,
    ) -> Result<(), String> {
        if res.is_some() && plan.kind != Kind::AllReduce {
            return Err("tp exchange: only an all-reduce adds a residual".into());
        }
        if norm.is_some_and(|nm| res.is_none() || nm.out.xs as usize != dst) {
            return Err("tp exchange: a norm needs a residual and its sums' address".into());
        }
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
                let own = if bf16 || sent {
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
                // An all-reduce's piece is all of its values.
                let res = res.map(|r| r as *const f32);
                match norm {
                    Some(nm) => {
                        let srcs: Vec<*const c_void> = srcs.iter().map(|&p| p.cast()).collect();
                        let code = crate::model::fp8_act::sum_rms_norm(
                            &srcs,
                            bf16,
                            res,
                            nm.w,
                            nm.eps,
                            &nm.out,
                            (nm.row, mine.1 / nm.row),
                            stream,
                        );
                        if code != 0 {
                            return Err(format!("tp all-reduce: the sum's norm: error {code}"));
                        }
                    }
                    None => sum_add(kernel, &srcs, res, dst as *mut f32, mine.1, stream)?,
                }
            }
            check(
                "cuStreamWriteValue32",
                cuStreamWriteValue32_v2(stream, fd + CONSUMED as u64, seq, 0),
            )?;
        }
        Ok(())
    }
}
