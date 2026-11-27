//! The CUDA side of the ops: the kernels of `src/cuda/inkling2.cu`, launched from here with
//! `cudaLaunchKernelEx`, each taking its node described generically (data, shape, byte strides,
//! type) and its external params. On compute capability 12.x (the GB10) every launch is a
//! programmatic dependent launch (#1221 r5), as `glm5next.cu`'s kernels are: the GPU sets a
//! kernel up while the one ahead is still running, and the kernel's first act is to wait for
//! it (`pdl_wait`), so nothing runs in another order.

use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicI32, Ordering};

use crate::ggml::{external_params, ffi, types};

const MAX_SRC: usize = 10;
const _: () = assert!(ffi::GGML_MAX_SRC as usize == MAX_SRC);

/// `legion_ink_args` of `src/cuda/inkling2.cu`.
#[repr(C)]
struct Args {
    dst: *mut c_void,
    ne: [i64; 4],
    nb: [i64; 4],
    src: [*const c_void; MAX_SRC],
    src_ne: [[i64; 4]; MAX_SRC],
    src_nb: [[i64; 4]; MAX_SRC],
    src_type: [i32; MAX_SRC],
    n_src: i32,
    params: [i32; 8],
}

/// `dim3`.
#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

/// `a / b` rounded up (`a >= 0`, `b > 0`).
fn cdiv(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

fn dim(x: i64, y: i64) -> Dim3 {
    Dim3 {
        x: x as u32,
        y: y as u32,
        z: 1,
    }
}

/// `cudaLaunchAttribute`: the id, padding to 8 bytes, a 64-byte value union.
#[repr(C, align(8))]
struct LaunchAttribute {
    id: c_int,
    pad: [u8; 4],
    val: [u32; 16],
}

/// `cudaLaunchConfig_t`.
#[repr(C)]
struct LaunchConfig {
    grid: Dim3,
    block: Dim3,
    dynamic_smem: usize,
    stream: *mut c_void,
    attrs: *mut LaunchAttribute,
    num_attrs: u32,
}

/// `cudaLaunchAttributeProgrammaticStreamSerialization`.
const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

/// The kernels of `legion_ink_kernels`.
const KERNELS: usize = 24;

unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs' addresses), in
    /// [`Kernel`] order.
    static legion_ink_kernels: [*const c_void; KERNELS];
    fn cudaLaunchKernelExC(
        config: *const LaunchConfig,
        func: *const c_void,
        args: *mut *mut c_void,
    ) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
    fn cudaGetLastError() -> c_int;
}

const DEV_ATTR_CC_MAJOR: c_int = 75;

/// Devices [`MAJOR`] covers.
const DEVICES: usize = 64;

/// Each device's compute capability major; 0 not yet read, -1 unknown.
static MAJOR: [AtomicI32; DEVICES] = [const { AtomicI32::new(0) }; DEVICES];

/// Whether the current device takes programmatic dependent launches of these kernels: compute
/// capability 12.x (GB10, RTX 50), read once per device. Their wait is compiled in there, since
/// the load refuses a 9.0+ GPU whose code is older than sm_90 (`crate::hw::cuda_arch`), as for
/// `nvfp4_once`'s kernels.
fn pdl() -> bool {
    let mut dev: c_int = 0;
    // SAFETY: plain queries of the current device; an error is cleared.
    unsafe {
        if cudaGetDevice(&mut dev) != 0 || !(0..DEVICES as c_int).contains(&dev) {
            cudaGetLastError();
            return false;
        }
        let slot = &MAJOR[dev as usize];
        let mut major = slot.load(Ordering::Relaxed);
        if major == 0 {
            let mut v: c_int = 0;
            major = if cudaDeviceGetAttribute(&mut v, DEV_ATTR_CC_MAJOR, dev) == 0 && v > 0 {
                v
            } else {
                cudaGetLastError();
                -1
            };
            slot.store(major, Ordering::Relaxed);
        }
        major == 12
    }
}

/// The kernels' order in `legion_ink_kernels`.
#[derive(Clone, Copy)]
enum Kernel {
    Mix,
    CtrlIn,
    CtrlOut,
    Conv,
    Clip,
    /// `attn_kernel<1>`, `<2>` and `<4>`: heads of 32, 64 and 128.
    Attn32,
    Attn64,
    Attn128,
    Sigmul,
    Situ,
    MmId,
    Softcap,
    /// A prompt step's conv: its outputs a thread per token, then its history (#1221).
    ConvPrompt,
    ConvHist,
    /// The attention kernels for blocks of one token (`ATTN_DECODE_ROWS` rows, #1221).
    Attn32Dec,
    Attn64Dec,
    Attn128Dec,
    /// The expert matmul of steps with no more pairs than experts, a warp per few rows of a
    /// pair (#1221); `MmId` runs the rest in tiles, both with the same sums.
    MmIdVec,
    /// A controller's gain and forget products and output at small steps (#1221 r5).
    CtrlGate,
    /// `rms_kernel<1>`, `<3>`, `<9>`, `<16>`: `glm_rms_norm`'s sums, a thread's elements in
    /// registers (#1221 r6).
    Rms1,
    Rms3,
    Rms9,
    Rms16,
    /// A decode step's K and V: convs, clip, rms and cache writes (#1221 r6).
    Kv,
}

/// `RMS_THREADS` of the kernels.
const RMS_THREADS: i64 = 256;

/// `MIX_STATES`, `MIX_QUERIES`, `CONV_TAPS` of the kernels.
const MIX_STATES: i32 = 8;
const MIX_QUERIES: i64 = 4;
const CONV_TAPS: i64 = 8;
/// `ATTN_TQ`, `ATTN_ROWS`, `ATTN_NR` and the threads of the attention kernel.
const ATTN_TQ: i64 = 4;
const ATTN_ROWS: i64 = 16;
const ATTN_DECODE_ROWS: i64 = 4;
const ATTN_NR: i64 = 16;
const ATTN_THREADS: u32 = 128;
/// `MM_BM` and `MM_THREADS` of the expert matmul's tiles, and the rows of a block of its
/// decode kernel (`MM_VEC_ROWS` a warp).
const MM_BM: i64 = 32;
const MM_THREADS: u32 = 256;
const MM_VEC_BLOCK_ROWS: i64 = 8 * 4;

/// The node `dst` as the kernels read it.
unsafe fn args(dst: &ffi::ggml_tensor) -> Args {
    let mut a = Args {
        dst: dst.data,
        ne: dst.ne,
        nb: dst.nb.map(|b| b as i64),
        src: [std::ptr::null(); MAX_SRC],
        src_ne: [[0; 4]; MAX_SRC],
        src_nb: [[0; 4]; MAX_SRC],
        src_type: [0; MAX_SRC],
        n_src: 0,
        params: [0; 8],
    };
    for (i, &s) in dst.src.iter().enumerate() {
        if s.is_null() {
            break;
        }
        // SAFETY: the node's sources are valid tensors.
        let s = unsafe { &*s };
        a.src[i] = s.data;
        a.src_ne[i] = s.ne;
        a.src_nb[i] = s.nb.map(|b| b as i64);
        a.src_type[i] = s.type_ as i32;
        a.n_src = i as i32 + 1;
    }
    let p = external_params(dst);
    let n = p.len().min(8);
    a.params[..n].copy_from_slice(&p[..n]);
    a
}

/// Enqueues kernel `k` for the node `dst` on `stream`; nothing for an empty grid.
unsafe fn launch(
    k: Kernel,
    name: &str,
    dst: &ffi::ggml_tensor,
    grid: Dim3,
    block: u32,
    shared: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    if grid.x == 0 || grid.y == 0 {
        return Ok(());
    }
    // SAFETY: device tensors of the stream's device; the kernel reads `a` from the launch's
    // copy of its argument.
    unsafe {
        let mut a = args(dst);
        let mut argv = [(&raw mut a).cast::<c_void>()];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid,
            block: dim(block as i64, 1),
            dynamic_smem: shared,
            stream,
            attrs: &mut attr,
            num_attrs: u32::from(pdl()),
        };
        match cudaLaunchKernelExC(&config, legion_ink_kernels[k as usize], argv.as_mut_ptr()) {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }
}

/// Source `i` of `dst`.
unsafe fn src(dst: &ffi::ggml_tensor, i: usize) -> &ffi::ggml_tensor {
    // SAFETY: the ops' builders give these sources.
    unsafe { &*dst.src[i] }
}

/// The node's element count.
fn elements(dst: &ffi::ggml_tensor) -> i64 {
    dst.ne.iter().product()
}

/// One thread per element, 256 per block.
unsafe fn elementwise(
    k: Kernel,
    name: &str,
    dst: &ffi::ggml_tensor,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            k,
            name,
            dst,
            dim(cdiv(elements(dst), 256), 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn mix(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    if external_params(dst)[0] > MIX_STATES || dst.ne[2] > MIX_QUERIES {
        return Err("ink_mix: more states or queries than the kernel holds".into());
    }
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Mix,
            "ink_mix",
            dst,
            dim(dst.ne[1], 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn rms(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    let k = match cdiv(dst.ne[0], RMS_THREADS) {
        1 => Kernel::Rms1,
        2..=3 => Kernel::Rms3,
        4..=9 => Kernel::Rms9,
        10..=16 => Kernel::Rms16,
        _ => return Err("ink_rms: rows wider than the kernel holds".into()),
    };
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            k,
            "ink_rms",
            dst,
            dim(elements(dst) / dst.ne[0], 1),
            RMS_THREADS as u32,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn ctrl_in(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::CtrlIn,
            "ink_ctrl_in",
            dst,
            dim(dst.ne[1], 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn ctrl_out(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe { elementwise(Kernel::CtrlOut, "ink_ctrl_out", dst, stream) }
}

/// `CTRL_TOK`, `CTRL_RANK` and `CTRL_ROWS` of `ctrl_gate_kernel` (its threads twice the rows).
const CTRL_TOK: i64 = 8;
const CTRL_RANK: i64 = 512;
const CTRL_ROWS: i64 = 32;

/// Whether `t`'s data and row stride suit 16-byte loads of its rows.
fn rows16(t: &ffi::ggml_tensor) -> bool {
    (t.data as usize | t.nb[1]) % 16 == 0
}

pub(super) unsafe fn ctrl_gate(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        let k = src(dst, 2).ne[0];
        if dst.ne[1] > CTRL_TOK
            || k % 32 != 0
            || k > CTRL_RANK
            || !rows16(src(dst, 3))
            || !rows16(src(dst, 4))
        {
            return Err("ink_ctrl_gate: a shape or rows the kernel does not take".into());
        }
        launch(
            Kernel::CtrlGate,
            "ink_ctrl_gate",
            dst,
            dim(cdiv(dst.ne[0], CTRL_ROWS), 1),
            2 * CTRL_ROWS as u32,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn conv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        if src(dst, 1).ne[0] > CONV_TAPS {
            return Err("ink_conv: more taps than the kernel holds".into());
        }
        let n_tok = external_params(dst)[0] as i64;
        let n_seqs = dst.ne[1] / n_tok;
        let blocks = cdiv(dst.ne[0], 128);
        if n_tok < CONV_PROMPT_MIN || n_tok > GRID_YZ || n_seqs > GRID_YZ {
            return launch(
                Kernel::Conv,
                "ink_conv",
                dst,
                dim(blocks, n_seqs),
                128,
                0,
                stream,
            );
        }
        // A prompt step: the outputs over (channels, tokens, sequences), then the history
        // from the step's last inputs (the same values as the token walk's, #1221).
        let grid = Dim3 {
            x: blocks as u32,
            y: n_tok as u32,
            z: n_seqs as u32,
        };
        launch(Kernel::ConvPrompt, "ink_conv", dst, grid, 128, 0, stream)?;
        launch(
            Kernel::ConvHist,
            "ink_conv",
            dst,
            dim(blocks, n_seqs),
            128,
            0,
            stream,
        )
    }
}

/// The fewest tokens a sequence of a step has for [`conv`] to run them a thread each
/// (`conv_prompt_kernel`) rather than walk them (`conv_kernel`); the same outputs either way.
const CONV_PROMPT_MIN: i64 = 2;

/// The most blocks along a grid's y or z.
const GRID_YZ: i64 = 65_535;

pub(super) unsafe fn clip(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    if dst.ne[0] > 128 {
        return Err("ink_clip: rows wider than 128".into());
    }
    let rows = dst.ne[1] * dst.ne[2];
    // SAFETY: a node of this op, in device memory.
    unsafe {
        launch(
            Kernel::Clip,
            "ink_clip",
            dst,
            dim(cdiv(rows, 8), 1),
            256,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn kv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        if src(dst, 2).ne[0] > CONV_TAPS {
            return Err("ink_kv: more taps than the kernel holds".into());
        }
        let d = external_params(dst)[5] as i64;
        if d > 128 {
            return Err("ink_kv: heads wider than 128".into());
        }
        let rows = 2 * (dst.ne[0] / d) * dst.ne[1];
        launch(
            Kernel::Kv,
            "ink_kv",
            dst,
            dim(cdiv(rows, 4), 1),
            128,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn attn(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        let d = dst.ne[0];
        let k = match d {
            32 => Kernel::Attn32,
            64 => Kernel::Attn64,
            128 => Kernel::Attn128,
            _ => {
                return Err(format!(
                    "ink_attn: heads of {d} (the kernels take 32, 64, 128)"
                ));
            }
        };
        let (kc, vc) = (src(dst, 1), src(dst, 2));
        if kc.type_ != types::F16 || vc.type_ != types::F16 {
            return Err("ink_attn: F16 caches only".into());
        }
        let nkv = kc.ne[0] / d;
        let pk = dst.ne[1] / nkv;
        if pk * ATTN_TQ > ATTN_ROWS || src(dst, 6).ne[0] > ATTN_NR {
            return Err(format!(
                "ink_attn: {pk} heads per kv head, {} bias ranks (the kernel takes 4 and 16)",
                src(dst, 6).ne[0]
            ));
        }
        let n_tok = external_params(dst)[0] as i64;
        let n_seqs = dst.ne[2] / n_tok;
        // Blocks of one token hold pk rows: the kernel with that few in registers (#1221).
        let k = match k {
            Kernel::Attn32 if n_tok == 1 && pk <= ATTN_DECODE_ROWS => Kernel::Attn32Dec,
            Kernel::Attn64 if n_tok == 1 && pk <= ATTN_DECODE_ROWS => Kernel::Attn64Dec,
            Kernel::Attn128 if n_tok == 1 && pk <= ATTN_DECODE_ROWS => Kernel::Attn128Dec,
            k => k,
        };
        // CUDA's grid y limit: a group this large is refused, not launched wrong.
        let grid_y = n_seqs * cdiv(n_tok, ATTN_TQ);
        if grid_y > GRID_YZ {
            return Err(format!(
                "ink_attn: {n_seqs} sequences of {n_tok} tokens need a grid y of {grid_y} \
                 (CUDA takes {GRID_YZ})"
            ));
        }
        launch(
            k,
            "ink_attn",
            dst,
            dim(nkv, grid_y),
            ATTN_THREADS,
            0,
            stream,
        )
    }
}

pub(super) unsafe fn sigmul(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe { elementwise(Kernel::Sigmul, "ink_sigmul", dst, stream) }
}

pub(super) unsafe fn situ(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe { elementwise(Kernel::Situ, "ink_situ", dst, stream) }
}

pub(super) unsafe fn mm_id(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        let experts = src(dst, 0).ne[2];
        let pairs = dst.ne[1] * dst.ne[2];
        // Few pairs (decode): a block per pair reads only its expert's weights. More: a block
        // per expert reuses each weight for up to `MM_BN` pairs. Same sums either way.
        let (k, grid) = if pairs <= experts.min(GRID_YZ) {
            (
                Kernel::MmIdVec,
                dim(cdiv(dst.ne[0], MM_VEC_BLOCK_ROWS), pairs),
            )
        } else {
            (Kernel::MmId, dim(cdiv(dst.ne[0], MM_BM), experts))
        };
        launch(k, "ink_mm_id", dst, grid, MM_THREADS, 0, stream)
    }
}

pub(super) unsafe fn softcap(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe { elementwise(Kernel::Softcap, "ink_softcap", dst, stream) }
}
