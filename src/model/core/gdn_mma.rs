//! `legion serve --gdn-mma` (#781, #790): a gated DeltaNet prompt group that ggml-cuda would run
//! on its per-token kernel only because of its head counts goes to a chunked tensor-core kernel
//! instead, as an external op ([`crate::ggml::ExternalOp`]).
//!
//! ggml-cuda picks its chunked kernel (`gated-delta-net-mma.cu`) on sm_86, sm_89, sm_120 and the
//! GB10 for 16 key heads and 16, 32, 48 or 64 value heads only; the kernel reads key head
//! `h % n_k` for any count. Qwen3.8-Flash-Next has 16 key and 48 value heads, so one GPU runs the
//! chunked kernel; a tensor parallel rank holds 8 key and 24 value heads (#733) and fell back to
//! the per-token kernel: on two DGX Sparks a rank's recurrence took 954 ms per 8 prompt graphs,
//! one Spark's whole layer 767 ms (#758's op profile). Its rounding is the chunked kernel's, not
//! the per-token kernel's: a tensor parallel run's bits move, so the flag is opt-in. Groups
//! ggml-cuda already gives its chunked kernel keep ggml's op.
//!
//! The kernel is Legion's own port of ggml's (`src/cuda/gdn_mma.cu`, MIT; with Legion's flags its
//! SASS is that of ggml's source, checked on sm_86 and sm_121a), kernels only. This module holds the rest: the
//! shape gate ([`shapes_fit`]), ggml's schedule rule ([`Schedule`]: whole heads, or value slices
//! when whole heads leave half the SMs idle), the device facts it needs (compute capability and
//! SM count, read once per device) and the launch (`cudaLaunchKernel` with the grid, block and
//! shared memory chosen here).

#[cfg(feature = "cuda")]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use super::gdn_scan;
#[cfg(feature = "cuda")]
use crate::ggml::ffi;
use crate::ggml::{Context, ExternalOp, Tensor, types};

static GDN_MMA: ExternalOp = ExternalOp {
    name: "gdn_mma",
    // The CPU graphs keep ggml's op; this is the per-token reference (tests).
    cpu: gdn_scan::cpu_gdn_scan,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_gdn_mma),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// The head size the kernel takes (keys and values).
const HEAD: i64 = 128;

/// The fewest tokens of a group it takes (ggml's chunked kernel's floor): decode and MTP verify
/// groups keep ggml's op.
pub(super) const MIN_TOKENS: i64 = 64;

/// Tokens a CTA advances by (the kernel's `C`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CHUNK: i64 = 16;

/// Threads of a CTA: 8 warps (`dim3(32, 8)`).
#[cfg(feature = "cuda")]
const BLOCK_THREADS: [u32; 3] = [32, 8, 1];

/// How [`gdn_mma`] spreads a group over the GPU: ggml's `select_gdn_mma_path` for NVIDIA GPUs.
/// The discriminants index `legion_gdn_mma_kernels` (`src/cuda/gdn_mma.cu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) enum Schedule {
    /// Four CTAs a head, 32 value columns each, at one CTA an SM (`gdn_single<16, 32, 1>`).
    Slices = 0,
    /// As [`Schedule::Slices`], with launch bounds for two CTAs an SM (`gdn_single<16, 32, 2>`).
    SlicesTwo = 1,
    /// One CTA a head (`gdn_single<16, 128>`).
    Heads = 2,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl Schedule {
    /// ggml's rule for `heads` value heads of `seqs` sequences on `sms` SMs: value slices when
    /// whole heads leave at least half the SMs idle, at one CTA an SM (a bigger register budget)
    /// when even the slices fit one wave.
    pub(super) fn pick(heads: i64, seqs: i64, sms: i64) -> Self {
        let blocks = heads * seqs;
        if blocks * 2 <= sms {
            if blocks * 4 <= sms {
                Schedule::Slices
            } else {
                Schedule::SlicesTwo
            }
        } else {
            Schedule::Heads
        }
    }

    /// The value columns a CTA owns (the kernel's `V`).
    pub(super) fn cols(self) -> i64 {
        match self {
            Schedule::Slices | Schedule::SlicesTwo => 32,
            Schedule::Heads => HEAD,
        }
    }

    /// CTAs for `heads` value heads of `seqs` sequences.
    pub(super) fn grid(self, heads: i64, seqs: i64) -> i64 {
        heads * seqs * (HEAD / self.cols())
    }

    /// Dynamic shared memory of a CTA: `sizeof(shared<C, V>)` in `gdn_mma.cu` (its
    /// static_asserts check these numbers), BF16 high and low halves of q and k (C x 128), p and
    /// the inverse (C x C), delta (C x V) and the state (128 x V); C x C + 2 C floats; rounded up
    /// to 128 bytes.
    pub(super) fn smem(self) -> usize {
        let (c, v) = (CHUNK as usize, self.cols() as usize);
        let bf16s = 2 * c * HEAD as usize + 2 * c * c + c * v + HEAD as usize * v;
        let bytes = 2 * 2 * bf16s + 4 * (c * c + 2 * c);
        bytes.div_ceil(128) * 128
    }
}

/// Whether graphs built from now on use [`gdn_mma`] ([`set_gdn_mma`]).
static ON: AtomicBool = AtomicBool::new(false);

/// Whether the engine's GPU has the kernel's tensor cores (compute capability 8.0 or later),
/// set when the engine loads ([`set_gpu`]): the answer for a group whose state names no device
/// ([`gpu_for`]).
static GPU: AtomicBool = AtomicBool::new(false);

/// Sets whether graphs built from now on use [`gdn_mma`] where it fits (off by default; `legion
/// serve --gdn-mma`).
pub fn set_gdn_mma(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// The engine's GPU: the kernel needs compute capability 8.0 or later (`cc_major`).
pub fn set_gpu(cc_major: Option<i32>) {
    GPU.store(gpu_takes(cc_major), Ordering::Relaxed);
}

/// Whether a GPU of compute capability major `cc_major` has the kernel's tensor cores.
pub(super) fn gpu_takes(cc_major: Option<i32>) -> bool {
    cc_major.is_some_and(|m| m >= 8)
}

/// Whether the GPU of device `dev` (the state's) runs the kernel, asked per device (#1201: a box
/// may hold GPUs of different architectures); the engine's answer where `dev` is unknown.
fn gpu_for(dev: Option<usize>) -> bool {
    gpu_for_by(dev, GPU.load(Ordering::Relaxed), crate::hw::per_device::cc)
}

/// [`gpu_for`] with the engine's answer `engine` and the compute capabilities `cc_of` gives.
pub(super) fn gpu_for_by(
    dev: Option<usize>,
    engine: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, |cc| gpu_takes(cc.map(|c| c.0)), cc_of)
}

/// Whether ggml-cuda gives a group of `n_k` key and `n_v` value heads its chunked kernel itself.
fn ggml_takes(n_k: i64, n_v: i64) -> bool {
    n_k == 16 && [16, 32, 48, 64].contains(&n_v)
}

/// Whether a prompt group of these shapes fits the op: heads of [`HEAD`], value heads a multiple
/// of the key heads, at least [`MIN_TOKENS`] tokens, a head count ggml-cuda keeps on its
/// per-token kernel, a grid of at most `i32::MAX` CTAs in any schedule (ggml's bound), and the
/// layouts the kernel reads (q and k alike with contiguous rows, v rows contiguous, g, beta and
/// the state contiguous).
pub(super) fn shapes_fit(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    let ([head_k, n_k, nt, nsq], [head_v, n_v, ntv, ns]) = (q.ne(), v.ne());
    [q, k, v, g, beta, state]
        .iter()
        .all(|t| t.ty() == types::F32)
        && head_k == HEAD
        && head_v == HEAD
        && nt == ntv
        && nt >= MIN_TOKENS
        && n_k > 0
        && n_v > 0
        && n_v % n_k == 0
        && !ggml_takes(n_k, n_v)
        && nsq > 0
        && ns > 0
        && ns % nsq == 0
        && ns <= i32::MAX as i64 / (n_v * 4)
        && k.ne() == q.ne()
        && k.nb() == q.nb()
        && [q, v].iter().all(|t| t.nb()[0] == 4)
        && g.ne() == [1, n_v, nt, ns]
        && beta.ne() == g.ne()
        && [g, beta, state].iter().all(|t| t.is_contiguous())
        && state.ne() == [HEAD, HEAD, n_v, ns]
}

/// Whether [`gdn_mma`] takes this prompt group: switched on, a CUDA build, the state in the
/// memory of one GPU with the kernel's tensor cores ([`gpu_for`]; the CPU graph is the reference
/// and keeps ggml's op), and [`shapes_fit`].
pub(super) fn fits(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && !state.is_host()
        && !state.is_meta()
        && shapes_fit(q, k, v, g, beta, state)
        && gpu_for(state.cuda_device())
}

/// ggml's `gated_delta_net(q, k, v, g, beta, state, 1)` on its chunked kernel: the same shape
/// out (`[128 * H, n_tokens * n_seqs + 128 * n_seqs]`: the tokens' outputs, then the final
/// states), so the caller's views of the result do not change.
pub(super) fn gdn_mma<'a>(
    ctx: &'a Context,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    g: Tensor<'a>,
    beta: Tensor<'a>,
    state: Tensor<'a>,
) -> Tensor<'a> {
    assert!(shapes_fit(q, k, v, g, beta, state), "gdn_mma: shapes");
    let [s_v, h, nt, ns] = v.ne();
    ctx.external(
        &GDN_MMA,
        types::F32,
        [s_v * h, nt * ns + s_v * ns, 1, 1],
        &[q, k, v, g, beta, state],
    )
}

/// The kernel's argument, `legion_gdn_mma_args` in `src/cuda/gdn_mma.cu` (ggml's
/// `ggml_cuda_gdn_mma_args` without `eligible`; 184 bytes, checked on both sides): q/k element
/// (i, head, t, seq) at `i + head * sq[0] + t * sq[1] + (seq / rq3) * sq[2]` floats, key head
/// `h % h_k` for value head `h`; v at `i + h * sv[0] + t * sv[1] + seq * sv[2]`; g and beta
/// (h, t, seq) at `h * sb[0] + t * sb[1] + seq * sb[2]`; `state` and `state_out` `[128, 128, h,
/// n_seqs]` contiguous; `dst` the tokens' outputs, `[128 * h, n_tokens * n_seqs]`.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[repr(C)]
pub(super) struct Args {
    q: *const f32,
    k: *const f32,
    v: *const f32,
    g: *const f32,
    beta: *const f32,
    state: *const f32,
    dst: *mut f32,
    state_out: *mut f32,
    h: i64,
    h_k: i64,
    n_tokens: i64,
    n_seqs: i64,
    rq3: i64,
    sq: [i64; 3],
    sv: [i64; 3],
    sb: [i64; 3],
    scale: f32,
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI8, AtomicI32, Ordering};

    use super::{Args, BLOCK_THREADS, Schedule};

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    unsafe extern "C" {
        /// The kernels as the CUDA runtime knows them (their host stubs), indexed by
        /// [`Schedule`].
        static legion_gdn_mma_kernels: [*const c_void; 3];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
    }

    const DEV_ATTR_SMS: c_int = 16;
    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const FUNC_ATTR_MAX_DYNAMIC_SHARED: c_int = 8;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// SMs of each device; 0 not yet read, -1 a device the kernels cannot run on (compute
    /// capability under 8.0, or the queries failed).
    static SMS: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// Each device's shared-memory opt-in of each kernel: 0 not yet, 1 done, -1 refused (more
    /// than the device's opt-in limit).
    static SMEM: [[AtomicI8; 3]; DEVICES] = [const { [const { AtomicI8::new(0) }; 3] }; DEVICES];

    /// The current device and its SMs, read once per device (ggml's `ggml_cuda_info().devices[
    /// device].nsm` and `ampere_mma_available(cc)`).
    fn device() -> Result<(usize, i64), String> {
        let mut dev: c_int = 0;
        // SAFETY: plain queries of the current device.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
                return Err("gdn_mma: no current device".into());
            }
            let slot = &SMS[dev as usize];
            let mut sms = slot.load(Ordering::Relaxed);
            if sms == 0 {
                let (mut n, mut major) = (0, 0);
                sms = if cudaDeviceGetAttribute(&mut n, DEV_ATTR_SMS, dev) == 0
                    && cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0
                    && major >= 8
                    && n > 0
                {
                    n
                } else {
                    cudaGetLastError();
                    -1
                };
                slot.store(sms, Ordering::Relaxed);
            }
            if sms < 0 {
                return Err("gdn_mma: the GPU cannot run the kernel".into());
            }
            Ok((dev as usize, sms as i64))
        }
    }

    /// Gives kernel `s` its dynamic shared memory on device `dev`, once (ggml's `init`).
    fn opt_in(dev: usize, s: Schedule) -> bool {
        let slot = &SMEM[dev][s as usize];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            state = unsafe {
                if cudaFuncSetAttribute(
                    legion_gdn_mma_kernels[s as usize],
                    FUNC_ATTR_MAX_DYNAMIC_SHARED,
                    s.smem() as c_int,
                ) == 0
                {
                    1
                } else {
                    cudaGetLastError();
                    -1
                }
            };
            slot.store(state, Ordering::Relaxed);
        }
        state > 0
    }

    /// Launches `a` on `stream` with ggml's schedule for the current device.
    pub(super) fn run(a: &mut Args, stream: *mut c_void) -> Result<(), String> {
        let (dev, sms) = device()?;
        let s = Schedule::pick(a.h, a.n_seqs, sms);
        if !opt_in(dev, s) {
            return Err(format!(
                "gdn_mma: the GPU cannot give {} bytes of shared memory",
                s.smem()
            ));
        }
        let grid = Dim3 {
            x: s.grid(a.h, a.n_seqs) as u32,
            y: 1,
            z: 1,
        };
        let [x, y, z] = BLOCK_THREADS;
        let block = Dim3 { x, y, z };
        let mut argv = [(a as *mut Args).cast::<c_void>()];
        // SAFETY: the kernel reads `a` from the launch's copy of it; the pointers in it are
        // device memory of the stream's device (the caller's node).
        match unsafe {
            cudaLaunchKernel(
                legion_gdn_mma_kernels[s as usize],
                grid,
                block,
                argv.as_mut_ptr(),
                s.smem(),
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("gdn_mma: CUDA error {e}")),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gdn_mma(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gdn_mma` (its shapes passed `shapes_fit`), device tensors of the
    // stream's device.
    unsafe {
        let [q, k, v, g, beta, s0] = [0, 1, 2, 3, 4, 5].map(|i| &*dst.src[i]);
        let (s_v, h, nt, ns) = (v.ne[0], v.ne[1], v.ne[2], v.ne[3]);
        let out = dst.data as *mut f32;
        let mut a = Args {
            q: q.data as *const f32,
            k: k.data as *const f32,
            v: v.data as *const f32,
            g: g.data as *const f32,
            beta: beta.data as *const f32,
            state: s0.data as *const f32,
            dst: out,
            state_out: out.add((s_v * h * nt * ns) as usize),
            h,
            h_k: q.ne[1],
            n_tokens: nt,
            n_seqs: ns,
            rq3: ns / q.ne[3],
            sq: gdn_scan::strides(q),
            sv: gdn_scan::strides(v),
            sb: gdn_scan::strides(beta),
            // ggml: 1.0f / sqrtf((float) S_v)
            scale: 1.0 / (s_v as f32).sqrt(),
        };
        launch::run(&mut a, stream)
    }
}

#[cfg(test)]
mod tests;
