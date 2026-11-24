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
//!
//! #1183: whole heads run `gdn_split<128, 16, 1, 1, 2>` ([`Schedule::WideTwo`]), ggml's
//! whole-head kernel with the same arithmetic in the same order (its bits) on 16 warps, each
//! owning one 8-column block of the state in registers (the MMA operands come from there), with
//! the next chunk's inputs copied into shared memory while the current one computes and warp
//! barriers only for the per-column steps. On compute capability 12.x
//! (the GB10, RTX 50) the groups ggml-cuda gives its own chunked kernel (one GPU's 16 key and 48
//! value heads of Qwen3.8-Flash-Next) come here by default for it ([`set_gdn_staged`]; `legion
//! serve --gdn-prompt-ggml` keeps ggml's op).

#[cfg(any(feature = "cuda", test))]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicI8, Ordering};

use super::gdn_scan;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

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
    /// As [`Schedule::Heads`], the inputs staged through shared memory (`gdn_staged<16>`, the
    /// same bits; #1183 round 8).
    Staged = 3,
    /// Two CTAs a head, 64 value columns each, two CTAs an SM (`gdn_split<64, 8, 1, 2>`): the
    /// staged kernel's bits, each CTA recomputing the chunk's Gram, P and inverse.
    Pair = 4,
    /// One CTA of 16 warps a head (`gdn_split<128, 16, 1, 1>`), the same bits.
    Wide = 5,
    /// One CTA of 8 warps a head (`gdn_split<128, 8, 2, 1>`, the staged kernel's layout with
    /// warp barriers for the per-column steps), the same bits.
    Lean = 6,
    /// [`Schedule::Wide`] with two input buffers, the next chunk's whole inputs copied while
    /// the current one computes (`gdn_split<128, 16, 1, 1, 2>`), the same bits: what
    /// [`Schedule::Heads`] runs when [`set_gdn_staged`] is on, on a [`STAGED_GPU`], and the
    /// inputs are 16-byte aligned (a GB10 layer of 4K prompt tokens: 1.45 ms staged, 1.33 ms).
    WideTwo = 7,
}

/// Kernels in `legion_gdn_mma_kernels`.
#[cfg(feature = "cuda")]
const KERNELS: usize = 10;

/// The table's [`Schedule::WideTwo`] with the gated output norm (`gdn_split<128, 16, 1, 1, 2,
/// true>`, the same shared memory; [`gdn_mma_gn`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const GN_KERNEL: usize = 8;

/// [`GN_KERNEL`] writing the norm's rows as FP8 codes (`gdn_split<128, 16, 1, 1, 2, true,
/// true>`; [`codes_on`]).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CODES_KERNEL: usize = 9;

/// [`CODES_KERNEL`]'s dynamic shared memory: [`Schedule::WideTwo`]'s, then each warp's BF16 row
/// of three chunks (16 x 128 x 3) and a chunk's amax slots (16 x [`CODES_HEADS`] words), as
/// `gdn_mma.cu`'s `codes_shared` (static_asserted there).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CODES_SMEM: usize = 85376 + 2 * 3 * 16 * 128 + 4 * 16 * CODES_HEADS as usize;

/// Most value heads [`CODES_KERNEL`] takes (`CODES_HEADS` in `gdn_mma.cu`): one GPU's heads of
/// Qwen3.8-Flash-Next.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CODES_HEADS: i64 = 48;

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
            Schedule::Pair => 64,
            Schedule::Heads
            | Schedule::Staged
            | Schedule::Wide
            | Schedule::Lean
            | Schedule::WideTwo => HEAD,
        }
    }

    /// Warps of a CTA (`dim3(32, warps)`).
    pub(super) fn warps(self) -> u32 {
        if matches!(self, Schedule::Wide | Schedule::WideTwo) {
            16
        } else {
            8
        }
    }

    /// Whether the kernel copies its inputs with cp.async (16-byte aligned q, k and v rows,
    /// [`Args::staged_aligned`]).
    pub(super) fn staged(self) -> bool {
        matches!(
            self,
            Schedule::Staged | Schedule::Pair | Schedule::Wide | Schedule::Lean | Schedule::WideTwo
        )
    }

    /// CTAs for `heads` value heads of `seqs` sequences.
    pub(super) fn grid(self, heads: i64, seqs: i64) -> i64 {
        heads * seqs * (HEAD / self.cols())
    }

    /// Dynamic shared memory of a CTA: `sizeof(shared<C, V>)` in `gdn_mma.cu` (its
    /// static_asserts check these numbers), BF16 high and low halves of q and k (C x 128), p and
    /// the inverse (C x C), delta (C x V) and the state (128 x V); C x C + 2 C floats; rounded up
    /// to 128 bytes. [`Schedule::Staged`] (`staged_shared<16>`): q, k, delta and the solved
    /// rows (C x 128), p and the inverse, the same floats, and two buffers of a chunk's F32 q, k
    /// and v rows, g and beta. The split kernels (`split_shared<V>`): q and k (C x 128), p and
    /// the inverse, delta and the solved rows (C x V), the same floats, and one buffer of a
    /// chunk's F32 q and k rows, g, beta and v rows (C x V).
    pub(super) fn smem(self) -> usize {
        let (c, v, d) = (CHUNK as usize, self.cols() as usize, HEAD as usize);
        let bytes = match self {
            Schedule::Staged => {
                2 * 2 * (4 * c * d + 2 * c * c) + 4 * (c * c + 2 * c) + 2 * 4 * (3 * c * d + 2 * c)
            }
            Schedule::Pair | Schedule::Wide | Schedule::Lean | Schedule::WideTwo => {
                let stages = if self == Schedule::WideTwo { 2 } else { 1 };
                2 * 2 * (2 * c * d + 2 * c * c + 2 * c * v)
                    + 4 * (c * c + 2 * c)
                    + stages * 4 * (2 * c * d + 2 * c + c * v)
            }
            Schedule::Slices | Schedule::SlicesTwo | Schedule::Heads => {
                let bf16s = 2 * c * d + 2 * c * c + c * v + d * v;
                2 * 2 * bf16s + 4 * (c * c + 2 * c)
            }
        };
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

/// The engine's GPU's compute capability (`cc`, major and minor): the kernel needs 8.0 or later,
/// and the groups ggml-cuda keeps come here on 12.0 and 12.1 only ([`STAGED_GPU`]). These are the
/// answers for a group whose state names no device ([`gpu_for`], [`staged_for`]).
pub fn set_gpu(cc: Option<(i32, i32)>) {
    GPU.store(gpu_takes(cc.map(|c| c.0)), Ordering::Relaxed);
    STAGED_GPU.store(staged_gpu(cc), Ordering::Relaxed);
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

/// Compute capability 12.0 (RTX 50) and 12.1 (the GB10): the consumer Blackwell GPUs ggml-cuda
/// gives its chunked kernel to (`GGML_CUDA_CC_BLACKWELL`, `GGML_CUDA_CC_DGX_SPARK`).
fn staged_gpu(cc: Option<(i32, i32)>) -> bool {
    matches!(cc, Some((12, 0 | 1)))
}

/// Whether device `dev` (the state's, or the launch's) is one [`STAGED`] takes ggml-cuda's
/// groups on ([`staged_gpu`]), asked per device like [`gpu_for`]; the engine's answer
/// ([`STAGED_GPU`]) where `dev` is unknown.
fn staged_for(dev: Option<usize>) -> bool {
    staged_for_by(
        dev,
        STAGED_GPU.load(Ordering::Relaxed),
        crate::hw::per_device::cc,
    )
}

/// [`staged_for`] with the engine's answer `engine` and the compute capabilities `cc_of` gives.
pub(super) fn staged_for_by(
    dev: Option<usize>,
    engine: bool,
    cc_of: impl FnOnce(usize) -> Option<(i32, i32)>,
) -> bool {
    crate::hw::per_device::pick_by(dev, engine, staged_gpu, cc_of)
}

/// Whether whole heads run [`Schedule::WideTwo`] and the groups ggml-cuda runs on its own chunked kernel
/// come to [`gdn_mma`] for it (on by default; `legion serve --gdn-prompt-ggml` turns it off).
static STAGED: AtomicBool = AtomicBool::new(true);

/// Whether the engine's GPU is one [`STAGED`] takes ggml-cuda's groups on ([`staged_gpu`]):
/// ggml-cuda runs its chunked kernel for them there, and the staged kernels were measured on the GB10
/// (#1183).
static STAGED_GPU: AtomicBool = AtomicBool::new(false);

/// Sets whether whole heads run [`Schedule::WideTwo`] (the same bits as ggml's whole-head
/// kernel) and,
/// on compute capability 12.x, whether the prompt groups ggml-cuda gives its own chunked kernel
/// run it here instead (on by default; `legion serve --gdn-prompt-ggml` turns it off).
pub fn set_gdn_staged(on: bool) {
    STAGED.store(on, Ordering::Relaxed);
}

/// A kernel forced for every launch (`legion gdn-bench --prompt`): a [`Schedule`] discriminant,
/// or -1 for the schedule rule.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
static FORCE: AtomicI8 = AtomicI8::new(-1);

/// Forces kernel `s` for every launch from now on, or the schedule rule again with `None`
/// (`legion gdn-bench --prompt`).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn force_schedule(s: Option<Schedule>) {
    FORCE.store(s.map_or(-1, |s| s as i8), Ordering::Relaxed);
}

/// Whether ggml-cuda gives a group of `n_k` key and `n_v` value heads its chunked kernel itself.
fn ggml_takes(n_k: i64, n_v: i64) -> bool {
    n_k == 16 && [16, 32, 48, 64].contains(&n_v)
}

/// Whether ggml-cuda runs a group of `n_k` key and `n_v` value heads token after token and
/// [`gdn_mma`] leaves it there (`--gdn-mma` off): the groups [`gdn_scan`] takes on compute
/// capability 12.x (#1133: a tensor parallel rank's share).
pub(super) fn ggml_per_token(n_k: i64, n_v: i64) -> bool {
    !ggml_takes(n_k, n_v) && !ON.load(Ordering::Relaxed)
}

/// Whether [`gdn_mma`] takes a group of `n_k` key and `n_v` value heads: the ones ggml-cuda keeps
/// on its per-token kernel with `--gdn-mma`; ggml-cuda's own with [`set_gdn_staged`] on a
/// compute capability 12.x GPU (device `dev`'s, [`staged_for`]).
fn takes(n_k: i64, n_v: i64, dev: Option<usize>) -> bool {
    if ggml_takes(n_k, n_v) {
        STAGED.load(Ordering::Relaxed) && staged_for(dev)
    } else {
        ON.load(Ordering::Relaxed)
    }
}

/// Whether a prompt group of these shapes fits the op: heads of [`HEAD`], value heads a multiple
/// of the key heads, at least [`MIN_TOKENS`] tokens (ggml-cuda's floor for its chunked kernel,
/// so a group under it keeps ggml's per-token op either way), a grid of at most `i32::MAX` CTAs
/// in any schedule (ggml's bound), and the layouts the kernel reads (q and k alike with
/// contiguous rows, v rows contiguous, g, beta and the state contiguous). Which head counts the
/// op takes is [`fits`]'s choice.
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

/// Whether [`gdn_mma`] takes this prompt group: switched on for its head counts ([`takes`]), a
/// CUDA build, the state in the memory of one GPU with the kernel's tensor cores ([`gpu_for`];
/// the CPU graph is the reference and keeps ggml's op), and [`shapes_fit`].
pub(super) fn fits(
    q: Tensor<'_>,
    k: Tensor<'_>,
    v: Tensor<'_>,
    g: Tensor<'_>,
    beta: Tensor<'_>,
    state: Tensor<'_>,
) -> bool {
    takes(q.ne()[1], v.ne()[1], state.cuda_device())
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
/// `ggml_cuda_gdn_mma_args` without `eligible`, then the gated output norm's; 232 bytes,
/// checked on both sides): q/k element (i, head, t, seq) at `i + head * sq[0] + t * sq[1] +
/// (seq / rq3) * sq[2]` floats, key head `h % h_k` for value head `h`; v at `i + h * sv[0] + t *
/// sv[1] + seq * sv[2]`; g and beta (h, t, seq) at `h * sb[0] + t * sb[1] + seq * sb[2]`;
/// `state` and `state_out` `[128, 128, h, n_seqs]` contiguous; `dst` the tokens' outputs, `[128
/// * h, n_tokens * n_seqs]`. The norm (one sequence, [`GN_KERNEL`] only): token t's z row at `z
/// + t * z_ld`, `norm_w` `head_v` floats, BF16 rows `[128 * h, n_tokens]` to `gout`; with
/// [`CODES_KERNEL`] instead the rows' FP8 codes and scales to `codes` and `scales`, `slots`
/// (`[h, n_tokens]` words, zeroed) the CTAs' amaxes ([`codes_layout`]).
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
    z: *const f32,
    z_ld: i64,
    norm_w: *const f32,
    gout: *mut c_void,
    eps: f32,
    silu: i32,
    head_v: i32,
    codes: *mut u8,
    scales: *mut f32,
    slots: *mut u32,
}

#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl Args {
    /// Whether the staged kernels ([`Schedule::staged`]) can copy the q, k and v rows in 16-byte pieces: the pointers 16-byte
    /// aligned and the strides whole multiples of 4 floats.
    fn staged_aligned(&self) -> bool {
        [self.q, self.k, self.v]
            .iter()
            .all(|p| (*p as usize).is_multiple_of(16))
            && self.sq.iter().chain(&self.sv).all(|s| s % 4 == 0)
    }

    /// Whether [`GN_KERNEL`] can copy the z rows in 16-byte pieces (one sequence).
    fn gn_aligned(&self) -> bool {
        self.n_seqs == 1 && (self.z as usize).is_multiple_of(16) && self.z_ld % 4 == 0
    }
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicI8, AtomicI32, Ordering};

    use super::{
        Args, CODES_HEADS, CODES_KERNEL, CODES_SMEM, FORCE, GN_KERNEL, KERNELS, Ran, STAGED,
        Schedule, staged_for,
    };

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
        static legion_gdn_mma_kernels: [*const c_void; KERNELS];
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
    /// `cudaFuncAttributePreferredSharedMemoryCarveout` and `cudaSharedmemCarveoutMaxShared`.
    const FUNC_ATTR_CARVEOUT: c_int = 9;
    const CARVEOUT_MAX_SHARED: c_int = 100;

    /// Devices the caches below cover.
    const DEVICES: usize = 64;

    /// SMs of each device; 0 not yet read, -1 a device the kernels cannot run on (compute
    /// capability under 8.0, or the queries failed).
    static SMS: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

    /// Each device's shared-memory opt-in of each kernel: 0 not yet, 1 done, -1 refused (more
    /// than the device's opt-in limit).
    static SMEM: [[AtomicI8; KERNELS]; DEVICES] =
        [const { [const { AtomicI8::new(0) }; KERNELS] }; DEVICES];

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

    /// Kernel `k`'s dynamic shared memory: schedule `s`'s, or [`CODES_SMEM`].
    fn smem(s: Schedule, k: usize) -> usize {
        if k == CODES_KERNEL {
            CODES_SMEM
        } else {
            s.smem()
        }
    }

    /// Gives kernel `k` of the table (schedule `s`'s, [`GN_KERNEL`] or [`CODES_KERNEL`]) its
    /// dynamic shared memory on device `dev`, once (ggml's `init`); the two-CTAs-an-SM kernel
    /// also asks for the whole shared-memory carveout.
    fn opt_in(dev: usize, s: Schedule, k: usize) -> bool {
        let slot = &SMEM[dev][k];
        let mut state = slot.load(Ordering::Relaxed);
        if state == 0 {
            // SAFETY: a kernel of the table and its shared memory size.
            state = unsafe {
                let func = legion_gdn_mma_kernels[k];
                if cudaFuncSetAttribute(func, FUNC_ATTR_MAX_DYNAMIC_SHARED, smem(s, k) as c_int)
                    == 0
                    && (s != Schedule::Pair
                        || cudaFuncSetAttribute(func, FUNC_ATTR_CARVEOUT, CARVEOUT_MAX_SHARED) == 0)
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

    /// The schedule forced by [`super::force_schedule`], if any.
    fn forced() -> Option<Schedule> {
        match FORCE.load(Ordering::Relaxed) {
            0 => Some(Schedule::Slices),
            1 => Some(Schedule::SlicesTwo),
            2 => Some(Schedule::Heads),
            3 => Some(Schedule::Staged),
            4 => Some(Schedule::Pair),
            5 => Some(Schedule::Wide),
            6 => Some(Schedule::Lean),
            7 => Some(Schedule::WideTwo),
            _ => None,
        }
    }

    /// Launches `a` on `stream` with ggml's schedule for the current device; whole heads on
    /// [`Schedule::WideTwo`] where [`STAGED`] is on, on a device [`staged_for`] picks, and
    /// [`Args::staged_aligned`]. With `gn` (a gated output norm in `a`), [`GN_KERNEL`] where
    /// that schedule is picked and [`Args::gn_aligned`], and with `codes` too (`a`'s codes,
    /// scales and zeroed slots) [`CODES_KERNEL`] where its grid is co-resident: at most
    /// [`CODES_HEADS`] heads, one CTA an SM. Returns which ran.
    pub(super) fn run(
        a: &mut Args,
        stream: *mut c_void,
        gn: bool,
        codes: bool,
    ) -> Result<Ran, String> {
        let (dev, sms) = device()?;
        let s = match forced() {
            Some(s) => s,
            None => match Schedule::pick(a.h, a.n_seqs, sms) {
                Schedule::Heads
                    if STAGED.load(Ordering::Relaxed)
                        && staged_for(Some(dev))
                        && a.staged_aligned() =>
                {
                    Schedule::WideTwo
                }
                s => s,
            },
        };
        if s.staged() && !a.staged_aligned() {
            return Err(format!(
                "gdn_mma: {s:?} needs 16-byte aligned q, k and v rows"
            ));
        }
        let gn = gn && s == Schedule::WideTwo && a.gn_aligned();
        let codes = gn
            && codes
            && a.h <= CODES_HEADS.min(sms)
            && !a.codes.is_null()
            && opt_in(dev, s, CODES_KERNEL);
        let k = match (gn, codes) {
            (_, true) => CODES_KERNEL,
            (true, false) => GN_KERNEL,
            (false, false) => s as usize,
        };
        if !opt_in(dev, s, k) {
            return Err(format!(
                "gdn_mma: the GPU cannot give {} bytes of shared memory",
                smem(s, k)
            ));
        }
        let grid = Dim3 {
            x: s.grid(a.h, a.n_seqs) as u32,
            y: 1,
            z: 1,
        };
        let block = Dim3 {
            x: 32,
            y: s.warps(),
            z: 1,
        };
        let mut argv = [(a as *mut Args).cast::<c_void>()];
        // SAFETY: the kernel reads `a` from the launch's copy of it; the pointers in it are
        // device memory of the stream's device (the caller's node).
        match unsafe {
            cudaLaunchKernel(
                legion_gdn_mma_kernels[k],
                grid,
                block,
                argv.as_mut_ptr(),
                smem(s, k),
                stream,
            )
        } {
            0 => Ok(match (gn, codes) {
                (_, true) => Ran::Codes,
                (true, false) => Ran::Norm,
                (false, false) => Ran::Plain,
            }),
            e => Err(format!("gdn_mma: CUDA error {e}")),
        }
    }
}

/// What [`launch::run`] ran: the delta rule alone (the tokens' F32 outputs), with the gated
/// output norm (its BF16 rows), or with the norm's FP8 codes.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ran {
    Plain,
    Norm,
    Codes,
}

/// What the last [`codes_on`] node's launch ran, a [`Ran`] discriminant (-1: none yet): the
/// tests' check of which path a case took.
#[cfg(all(test, feature = "cuda"))]
static CODES_RAN: AtomicI8 = AtomicI8::new(-1);

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, stream: *mut c_void) -> i32;
}

#[cfg(feature = "cuda")]
unsafe fn cuda_gdn_mma(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node made by `gdn_mma` (its shapes passed `shapes_fit`), device tensors of the
    // stream's device.
    unsafe {
        let [q, k, v, g, beta, s0] = [0, 1, 2, 3, 4, 5].map(|i| &*dst.src[i]);
        let (s_v, h, nt, ns) = (v.ne[0], v.ne[1], v.ne[2], v.ne[3]);
        let out = dst.data as *mut f32;
        // A node of `gdn_mma_gn` whose norm `norm_on` (or `codes_on`) switched on: z and the
        // norm's weight.
        let params = crate::ggml::external_params(dst);
        let mode = if dst.src[6].is_null() { 0 } else { params[0] };
        let gn = (mode == NORM_ROWS || mode == NORM_CODES).then(|| (&*dst.src[6], &*dst.src[7]));
        let (silu, eps) = (params[1] != 0, f32::from_bits(params[2] as u32));
        let gout = out.add((s_v * h * nt * ns + s_v * s_v * h * ns) as usize);
        // `codes_on`: the packed activations and the slots in the tokens' F32 rows, which the
        // norm leaves unwritten (`codes_layout`).
        let (total, slots_at, slots_len) = codes_layout(h, nt);
        let codes = (mode == NORM_CODES).then(|| {
            let base = out as *mut u8;
            (
                base,
                base.add((HEAD * h * nt) as usize) as *mut f32,
                base.add(slots_at) as *mut u32,
            )
        });
        debug_assert!(total <= slots_at && slots_at + slots_len <= (4 * HEAD * h * nt) as usize);
        if let Some((_, _, slots)) = codes {
            let e = cudaMemsetAsync(slots.cast(), 0, slots_len, stream);
            if e != 0 {
                return Err(format!("gdn_mma: cudaMemsetAsync failed ({e})"));
            }
        }
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
            z: gn.map_or(std::ptr::null(), |(z, _)| z.data as *const f32),
            z_ld: gn.map_or(0, |(z, _)| (z.nb[1] / 4) as i64),
            norm_w: gn.map_or(std::ptr::null(), |(_, w)| w.data as *const f32),
            gout: if gn.is_some() {
                gout.cast()
            } else {
                std::ptr::null_mut()
            },
            eps,
            silu: i32::from(silu),
            head_v: s_v as i32,
            codes: codes.map_or(std::ptr::null_mut(), |c| c.0),
            scales: codes.map_or(std::ptr::null_mut(), |c| c.1),
            slots: codes.map_or(std::ptr::null_mut(), |c| c.2),
        };
        let ran = launch::run(&mut a, stream, gn.is_some(), codes.is_some())?;
        #[cfg(test)]
        if codes.is_some() {
            CODES_RAN.store(ran as i8, Ordering::Relaxed);
        }
        if let Some((z, w)) = gn
            && ran == Ran::Plain
        {
            // The norm from the tokens' F32 outputs, as `fused::gated_norm_bf16`.
            super::fused::gated_norm_bf16_rows(
                (out, s_v * h),
                (z.data as *const f32, (z.nb[1] / 4) as i64),
                w.data as *const f32,
                gout.cast(),
                (s_v as i32, h as i32, nt as i32),
                silu,
                eps,
                stream,
            )?;
        }
        match codes {
            // The codes from the BF16 rows, as `fp8_act::plain_bf16` (over the F32 rows, read
            // by then).
            Some((c, s, _)) if ran != Ran::Codes => {
                let k = s_v * h;
                match crate::model::fp8_act::plain_bf16_raw(gout as *const u16, k, c, s, nt, stream)
                {
                    0 => Ok(()),
                    e => Err(format!("gdn_mma: fp8_act plain_bf16 failed ({e})")),
                }
            }
            _ => Ok(()),
        }
    }
}

/// Whether prompt steps fuse the gated output norm into [`gdn_mma`] ([`gdn_mma_gn`]):
/// on by default (#1176 round 44); `LEGION_DEBUG_GDN_GATED_NORM=0` runs the norm's own kernel
/// after [`gdn_mma`] (A/B runs).
pub(super) fn gn_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("LEGION_DEBUG_GDN_GATED_NORM").is_none_or(|v| v != "0"))
}

/// Whether [`gdn_mma_gn`] takes this prompt group: [`fits`], one sequence, and z (the gate,
/// F32 `[128 * h, n_tokens]`) with rows of whole 16-byte pieces.
pub(super) fn gn_fits(
    [q, k, v, g, beta]: [Tensor<'_>; 5],
    state: Tensor<'_>,
    z: Tensor<'_>,
    norm_w: Tensor<'_>,
) -> bool {
    let [s_v, h, nt, ns] = v.ne();
    fits(q, k, v, g, beta, state)
        && ns == 1
        && z.ty() == types::F32
        && z.ne()[..2] == [s_v * h, nt]
        && z.nb()[0] == 4
        && z.nb()[1] % 16 == 0
        && norm_w.ty() == types::F32
        && norm_w.is_contiguous()
        && norm_w.elements() == s_v as usize
        && !norm_w.is_host()
        && !norm_w.is_meta()
}

/// [`gdn_mma`] (the same outputs and states at the same offsets) that can also write the
/// layer's gated output norm, `fused::gated_norm_bf16(tokens, norm_w, z, 128, eps, silu)`'s
/// BF16 rows: in the same kernel ([`GN_KERNEL`], the norm's bits from the outputs in shared
/// memory) where it runs, else the norm's own kernel after it. The norm is off until
/// [`norm_on`] (the reader that takes the rows decides); the second tensor is its rows,
/// `[128 * h, n_tokens]` BF16. With the norm on the fused kernel does not write the tokens'
/// F32 outputs: nothing may read them.
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_mma_gn<'a>(
    ctx: &'a Context,
    [q, k, v, g, beta]: [Tensor<'a>; 5],
    state: Tensor<'a>,
    z: Tensor<'a>,
    norm_w: Tensor<'a>,
    eps: f32,
    silu: bool,
) -> (Tensor<'a>, Tensor<'a>) {
    assert!(
        gn_fits([q, k, v, g, beta], state, z, norm_w),
        "gdn_mma_gn: shapes"
    );
    let [s_v, h, nt, _] = v.ne();
    // F32 rows: the tokens', the states', then the BF16 rows two to a row.
    let (rows, half) = (nt + s_v, (nt + 1) / 2);
    let out = ctx.external_with(
        &GDN_MMA,
        types::F32,
        [s_v * h, rows + half, 1, 1],
        &[q, k, v, g, beta, state, z, norm_w],
        &[0, i32::from(silu), eps.to_bits() as i32],
    );
    let es = out.element_size();
    let tail = ctx.view_2d(
        out,
        [s_v * h, half],
        out.nb()[1],
        rows as usize * (s_v * h) as usize * es,
    );
    let bf = ctx.reinterpret(tail, types::BF16, [s_v * h, 2 * half, 1, 1]);
    (out, ctx.view_2d(bf, [s_v * h, nt], bf.nb()[1], 0))
}

/// Parameter 0 of a [`gdn_mma_gn`] node: the norm off (0), its BF16 rows ([`norm_on`]), or its
/// FP8 codes ([`codes_on`]).
const NORM_ROWS: i32 = 1;
const NORM_CODES: i32 = 2;

/// Switches the norm of a [`gdn_mma_gn`] node on: a reader takes its BF16 rows.
pub(super) fn norm_on(node: Tensor<'_>) {
    // SAFETY: an external op node of `gdn_mma_gn`; parameter 0 is its norm switch.
    unsafe { crate::ggml::set_external_param(node.raw(), 0, NORM_ROWS) };
}

/// `--dense-fp8-prompt` (#1183): switches the norm of a [`gdn_mma_gn`] node on as the packed
/// FP8 activations of the projection that reads it (`fp8_act::plain_bf16` of its BF16 rows,
/// bit for bit) and returns them, I8 `[fp8::dynamic_layout(128 * h, n_tokens).1]`: the codes
/// and scales from [`CODES_KERNEL`] where it runs, else from `fp8_act`'s kernel after the
/// norm. They sit in the tokens' F32 rows ([`codes_layout`]); the BF16 rows are then not
/// written (nothing may read them, nor the F32 rows).
pub(super) fn codes_on<'a>(ctx: &'a Context, node: Tensor<'a>) -> Tensor<'a> {
    // SAFETY: an external op node of `gdn_mma_gn`; parameter 0 is its norm switch.
    unsafe { crate::ggml::set_external_param(node.raw(), 0, NORM_CODES) };
    // SAFETY: the node's v (source 2), `[128, h, n_tokens, 1]`.
    let [_, h, nt, _] = unsafe { (*(*node.raw()).src[2]).ne };
    let (total, _, _) = codes_layout(h, nt);
    let head = ctx.view_1d(node, (total / 4) as i64, 0);
    ctx.reinterpret(head, types::I8, [total as i64, 1, 1, 1])
}

/// Where a [`codes_on`] node of `h` heads and `nt` tokens keeps its outputs in the tokens' F32
/// rows (`4 * 128 * h * nt` bytes): the packed activations (`total` bytes: the codes, then the
/// scales) at 0, then the kernel's amax slots (`[h, nt]` words) at the next 256-byte boundary.
/// Returns (total, the slots' offset, the slots' bytes).
pub(super) fn codes_layout(h: i64, nt: i64) -> (usize, usize, usize) {
    let (_, total) = crate::model::fp8::dynamic_layout(HEAD * h, nt);
    (total, total.next_multiple_of(256), 4 * (h * nt) as usize)
}

/// Whether `node` is a [`gdn_mma_gn`] node switched to codes ([`codes_on`]): `node_fuse`'s
/// count of producers that write an FP8 GEMM's codes themselves.
///
/// # Safety
/// `node` must point to a valid tensor.
pub(crate) unsafe fn is_codes(node: *const ffi::ggml_tensor) -> bool {
    // SAFETY: the caller's.
    unsafe {
        crate::ggml::op_of(node).is_some_and(|op| std::ptr::eq(op, &GDN_MMA))
            && crate::ggml::external_params(&*node)[0] == NORM_CODES
    }
}

#[cfg(test)]
mod tests;
