//! The CUDA side of the ops: one launcher (`src/cuda/glm5next.cu`) takes any of them by id,
//! with the node's tensors described generically (data, shape, byte strides, type) and its
//! external params.

use std::os::raw::c_void;

use crate::ggml::{external_params, ffi};

const MAX_SRC: usize = 10;
const _: () = assert!(ffi::GGML_MAX_SRC as usize == MAX_SRC);

/// `legion_glm_args` of `src/cuda/glm5next.cu`.
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

unsafe extern "C" {
    fn legion_glm_launch(op: i32, a: *const Args, stream: *mut c_void) -> i32;
}

/// Enqueues op `op` (the kernel file's `Op` order) for the node `dst` on `stream`.
unsafe fn launch(
    op: i32,
    name: &str,
    dst: &ffi::ggml_tensor,
    stream: *mut c_void,
) -> Result<(), String> {
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
    a.params.copy_from_slice(&external_params(dst)[..8]);
    // SAFETY: device tensors of the stream's device, described by `a`.
    match unsafe { legion_glm_launch(op, &a, stream) } {
        0 => Ok(()),
        1 => Err(format!("{name}: no CUDA kernel for this shape or type")),
        e => Err(format!("{name}: CUDA error {e}")),
    }
}

macro_rules! ops {
    ($($name:ident = $id:literal,)*) => {
        $(
            pub(super) unsafe fn $name(
                dst: &ffi::ggml_tensor,
                stream: *mut c_void,
            ) -> Result<(), String> {
                // SAFETY: a node of this op, in device memory.
                unsafe { launch($id, concat!("glm_", stringify!($name)), dst, stream) }
            }
        )*
    };
}

ops! {
    embed = 0,
    linear = 1,
    bf16 = 2,
    rms_norm = 3,
    layer_norm = 4,
    hc_pre = 5,
    hc_post = 6,
    hc_head = 7,
    kda_conv = 8,
    kda_gate = 9,
    sigmoid = 10,
    kda_scan = 11,
    gated_rms = 12,
    glu = 13,
    router_ids = 14,
    router_w = 15,
    weighted_sum = 16,
    cache_rows = 17,
    pool_keys = 18,
    mla_attn = 19,
    join_cols = 20,
    argmax = 21,
    pmax = 22,
    nextn_in = 23,
    get_cols = 24,
    add = 25,
    f16 = 26,
    wf16 = 27,
    linear_set = 28,
    glu_mv = 29,
}
