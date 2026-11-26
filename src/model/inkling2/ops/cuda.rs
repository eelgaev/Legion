//! The CUDA side of the ops: the kernels of `src/cuda/inkling2.cu`, launched from here with
//! `cudaLaunchKernel`, each taking its node described generically (data, shape, byte strides,
//! type) and its external params.

use std::os::raw::c_void;

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

fn dim(x: i64, y: i64) -> Dim3 {
    Dim3 {
        x: x as u32,
        y: y as u32,
        z: 1,
    }
}

unsafe extern "C" {
    /// The kernels as the CUDA runtime knows them (their host stubs' addresses), in
    /// [`Kernel`] order.
    static legion_ink_kernels: [*const c_void; 12];
    fn cudaLaunchKernel(
        func: *const c_void,
        grid: Dim3,
        block: Dim3,
        args: *mut *mut c_void,
        shared: usize,
        stream: *mut c_void,
    ) -> i32;
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
}

/// `MIX_STATES`, `MIX_QUERIES`, `CONV_TAPS` of the kernels.
const MIX_STATES: i32 = 9;
const MIX_QUERIES: i64 = 4;
const CONV_TAPS: i64 = 8;
/// `ATTN_TQ`, `ATTN_ROWS`, `ATTN_NR` and the threads of the attention kernel.
const ATTN_TQ: i64 = 4;
const ATTN_ROWS: i64 = 16;
const ATTN_NR: i64 = 16;
const ATTN_THREADS: u32 = 128;
/// `MM_ROWS`, `MM_P` of the expert matmul, and the most shared memory its inputs take.
const MM_ROWS: i64 = 64;
const MM_P: i64 = 4;
const MM_SHARED: i64 = 40 * 1024;

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
        match cudaLaunchKernel(
            legion_ink_kernels[k as usize],
            grid,
            dim(block as i64, 1),
            argv.as_mut_ptr(),
            shared,
            stream,
        ) {
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
            dim(elements(dst).div_ceil(256), 1),
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

pub(super) unsafe fn conv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op; its sources were checked by the builder.
    unsafe {
        if src(dst, 1).ne[0] > CONV_TAPS {
            return Err("ink_conv: more taps than the kernel holds".into());
        }
        let n_tok = external_params(dst)[0] as i64;
        launch(
            Kernel::Conv,
            "ink_conv",
            dst,
            dim(dst.ne[0].div_ceil(128), dst.ne[1] / n_tok),
            128,
            0,
            stream,
        )
    }
}

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
            dim(rows.div_ceil(8), 1),
            256,
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
        if pk * ATTN_TQ > ATTN_ROWS || src(dst, 7).ne[0] > ATTN_NR {
            return Err(format!(
                "ink_attn: {pk} heads per kv head, {} bias ranks (the kernel takes 4 and 16)",
                src(dst, 7).ne[0]
            ));
        }
        let n_tok = external_params(dst)[0] as i64;
        let n_seqs = dst.ne[2] / n_tok;
        launch(
            k,
            "ink_attn",
            dst,
            dim(nkv, n_seqs * n_tok.div_ceil(ATTN_TQ)),
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
        let w = src(dst, 0);
        let k = w.ne[0];
        let shared = MM_P * k * 4;
        if shared > MM_SHARED {
            return Err(format!(
                "ink_mm_id: rows of {k} (the kernel takes up to 2,560)"
            ));
        }
        launch(
            Kernel::MmId,
            "ink_mm_id",
            dst,
            dim(dst.ne[0].div_ceil(MM_ROWS), w.ne[2]),
            256,
            shared as usize,
            stream,
        )
    }
}

pub(super) unsafe fn softcap(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
    // SAFETY: a node of this op, in device memory.
    unsafe { elementwise(Kernel::Softcap, "ink_softcap", dst, stream) }
}
