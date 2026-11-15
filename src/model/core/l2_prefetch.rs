//! A weight's first bytes read into L2 ahead of the matmul that reads it, as an external op that
//! writes nothing (#689 round 11; kernel in `src/cuda/l2_prefetch.cu`).
//!
//! In a dense gated DeltaNet layer's one-user verify step on a GB10 (Qwen3.8-27B NVFP4, DFlash 2)
//! the qkv projection is followed by ~60-70 us of small kernels (conv, alpha/beta, delta prep,
//! the delta rule, the gated norm's norm) with DRAM mostly idle, and the z projection after them
//! ran ~20 us over the time its bytes take; out_proj takes that cost instead when z goes first.
//! [`ahead`] after the qkv projection reads z's first rows into L2 meanwhile. Nothing reads the
//! node's result and no tensor changes, so every output keeps its bits.
//!
//! Off by default ([`GDN_Z_MIB`] is 0): measured on a GB10 it saves z what it costs elsewhere.
//! The kernel holds the stream until its prefetches are served (4 MiB ~10 us, 8 MiB ~24 us,
//! z -8 / -18 us); issued before its wait it takes the qkv projection's tail bandwidth (qkv
//! +9 us, z -7 us); issued from the conv kernel it lengthens the conv by as much. Where it
//! applies ([`takes`]: CUDA, compute capability 12.x, stage graphs on so the node costs no
//! capture split, dense models, steps of up to [`MAX_TOKENS`] tokens),
//! `LEGION_DEBUG_GDN_PF=<MiB>` turns it on (read once); `LEGION_DEBUG_GDN_PF_FAR=1` prefetches
//! as many bytes of the LM head instead, rows nobody reads before they leave L2 (a probe: DRAM
//! traffic without the data; z does not move, so the cost is z's data, not a DRAM ramp).

use std::sync::OnceLock;

#[cfg(any(test, feature = "cuda"))]
use crate::ggml::external_params;
use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

#[cfg(test)]
mod tests;

static L2_PREFETCH: ExternalOp = ExternalOp {
    name: "l2_prefetch",
    cpu: cpu_l2_prefetch,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda_l2_prefetch),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Bytes of one thread's bulk prefetch; offsets and sizes are whole chunks.
pub const CHUNK: usize = 4096;

/// Most tokens a step may have for [`takes`]: the verify steps of 1-8 users (the sliced-K FP8
/// GEMM's range).
pub const MAX_TOKENS: i64 = 64;

/// The default size ahead of a DeltaNet layer's z projection, in MiB (0: off; 4 is the
/// measured one).
pub const GDN_Z_MIB: usize = 0;

/// Bytes ahead of a DeltaNet layer's z projection: `LEGION_DEBUG_GDN_PF` in MiB (0: off), else
/// [`GDN_Z_MIB`]; read once.
pub fn gdn_z_bytes() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| gdn_z_bytes_from(std::env::var("LEGION_DEBUG_GDN_PF").ok().as_deref()))
}

/// [`gdn_z_bytes`] for the switch's value `v`: whole MiB up to 64, else the default.
pub fn gdn_z_bytes_from(v: Option<&str>) -> usize {
    let mib = v
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&m| m <= 64)
        .unwrap_or(GDN_Z_MIB);
    mib << 20
}

/// `LEGION_DEBUG_GDN_PF_FAR=1` (a probe, read once): the LM head's rows instead of z's.
pub fn far() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_FAR").is_ok_and(|v| v.trim() == "1"))
}

/// `LEGION_DEBUG_GDN_PF_LATE=1` (an A/B probe, read once): the prefetch kernel issues its
/// prefetches before it waits for the qkv projection (during its last CTAs) instead of after.
pub fn late() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_LATE").is_ok_and(|v| v.trim() == "1"))
}

/// `LEGION_DEBUG_GDN_PF_CONV=1` (an A/B probe, read once): no prefetch node; the layer's
/// `conv_silu` kernel issues the prefetch at its start (see [`fused::conv_silu_window_pf`]).
///
/// [`fused::conv_silu_window_pf`]: super::fused::conv_silu_window_pf
pub fn in_conv() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LEGION_DEBUG_GDN_PF_CONV").is_ok_and(|v| v.trim() == "1"))
}

/// Whether a step of `n` tokens may prefetch `w`: a CUDA build with stage graphs on, a GPU of
/// compute capability 12.x, `w` in that GPU's memory and 16-byte aligned, `n` up to
/// [`MAX_TOKENS`].
pub fn takes(w: Tensor<'_>, n: i64) -> bool {
    cfg!(feature = "cuda")
        && (1..=MAX_TOKENS).contains(&n)
        && crate::ggml::stage_graph::enabled()
        && !w.is_host()
        && !w.is_meta()
        && w.nbytes() >= CHUNK
        && super::q8_group::sm12x()
}

/// Whether node `t` reads `w` itself or through a source of its own (a matmul followed by its
/// scale's `mul`, say).
pub fn reads(t: Tensor<'_>, w: Tensor<'_>) -> bool {
    /// # Safety
    /// `t` is a valid tensor whose sources are null or valid tensors.
    unsafe fn within(t: *const ffi::ggml_tensor, w: *const ffi::ggml_tensor, depth: u32) -> bool {
        // SAFETY: the caller's; each source is null or a valid tensor.
        unsafe {
            (*t).src.iter().any(|&s| {
                !s.is_null() && (std::ptr::eq(s, w) || (depth > 0 && within(s, w, depth - 1)))
            })
        }
    }
    // SAFETY: a valid node.
    unsafe { within(t.raw(), w.raw(), 1) }
}

/// Chunks of a prefetch of `bytes` from byte `offset` of a `total`-byte tensor: (first chunk,
/// chunks), within the tensor, at least one; whole chunks only (a partial one is dropped), so a
/// tensor under one chunk gets none.
pub fn span(offset: usize, bytes: usize, total: usize) -> (usize, usize) {
    let all = total / CHUNK;
    if all == 0 {
        return (0, 0);
    }
    let first = (offset / CHUNK).min(all.saturating_sub(1));
    let n = (bytes / CHUNK).clamp(1, all - first);
    (first, n)
}

/// A node that, once `after` has run, reads `bytes` of `w` from byte `offset` into L2 (whole
/// [`CHUNK`]s, see [`span`]). Its F32 `[1]` result holds nothing: expand it into the graph right
/// after `after` is built so it runs next to it.
pub fn ahead<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    offset: usize,
    bytes: usize,
    after: Tensor<'a>,
) -> Tensor<'a> {
    let (first, n) = span(offset, bytes, w.nbytes());
    ctx.external_with(
        &L2_PREFETCH,
        types::F32,
        [1, 1, 1, 1],
        &[w, after],
        &[first as i32, n as i32],
    )
}

unsafe fn cpu_l2_prefetch(_dst: &ffi::ggml_tensor, _ith: usize, _nth: usize) {
    // Nothing to compute: the result is never read.
}

#[cfg(feature = "cuda")]
unsafe fn cuda_l2_prefetch(
    dst: &ffi::ggml_tensor,
    stream: *mut std::os::raw::c_void,
) -> Result<(), String> {
    // SAFETY: a node made by `ahead`: src[0] the weight, in the stream's device's memory.
    let w = unsafe { &*dst.src[0] };
    let [first, n, ..] = external_params(dst);
    let base = w.data as usize + first as usize * CHUNK;
    if base % 16 != 0 || n <= 0 {
        // The bulk prefetch takes 16-byte aligned addresses; nothing to do otherwise.
        return Ok(());
    }
    launch::launch(base, n as usize * CHUNK, stream)
}

#[cfg(feature = "cuda")]
mod launch {
    use std::os::raw::{c_int, c_void};

    /// `dim3`.
    #[repr(C)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
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

    /// Threads a CTA, one chunk each.
    const THREADS: u32 = 32;

    /// `LEGION_DEBUG_GDN_PF_SHAPE=<KiB>x<threads>` (an A/B probe, read once): bytes per bulk
    /// prefetch (a multiple of 4 KiB, up to 1 MiB) and threads per CTA (1-32); else 4 KiB x 32.
    fn shape() -> (usize, u32) {
        static V: std::sync::OnceLock<(usize, u32)> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::env::var("LEGION_DEBUG_GDN_PF_SHAPE")
                .ok()
                .and_then(|s| {
                    let (k, t) = s.trim().split_once('x')?;
                    let k = k.parse::<usize>().ok()?;
                    let t = t.parse::<u32>().ok()?;
                    ((4..=1024).contains(&k) && k % 4 == 0 && (1..=32).contains(&t))
                        .then_some((k << 10, t))
                })
                .unwrap_or((super::CHUNK, THREADS))
        })
    }

    unsafe extern "C" {
        static legion_l2_prefetch_kernels: [*const c_void; 1];
        fn cudaLaunchKernelExC(
            config: *const LaunchConfig,
            func: *const c_void,
            args: *mut *mut c_void,
        ) -> c_int;
    }

    /// Launches the prefetch of `bytes` (whole chunks) from device address `base` on `stream`,
    /// as a programmatic dependent launch (the GPUs [`super::takes`] gives it are 12.x).
    pub(super) fn launch(base: usize, bytes: usize, stream: *mut c_void) -> Result<(), String> {
        let (size, threads) = shape();
        let chunks = bytes.div_ceil(size);
        let mut p = base as *const u8;
        let mut total = bytes as i64;
        let mut chunk = size as i32;
        let mut late = i32::from(super::late());
        let mut argv = [
            (&mut p as *mut *const u8).cast::<c_void>(),
            (&mut total as *mut i64).cast::<c_void>(),
            (&mut chunk as *mut i32).cast::<c_void>(),
            (&mut late as *mut i32).cast::<c_void>(),
        ];
        let mut attr = LaunchAttribute {
            id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            val: [0; 16],
        };
        attr.val[0] = 1;
        let config = LaunchConfig {
            grid: Dim3 {
                x: chunks.div_ceil(threads as usize) as u32,
                y: 1,
                z: 1,
            },
            block: Dim3 {
                x: threads,
                y: 1,
                z: 1,
            },
            dynamic_smem: 0,
            stream,
            attrs: &mut attr,
            num_attrs: 1,
        };
        // SAFETY: the table's kernel takes (const char *, int64_t, int32_t, int32_t), given here by
        // pointers to the launch's copies; `base` is device memory of the stream's device.
        match unsafe {
            cudaLaunchKernelExC(&config, legion_l2_prefetch_kernels[0], argv.as_mut_ptr())
        } {
            0 => Ok(()),
            e => Err(format!("l2_prefetch: CUDA error {e}")),
        }
    }
}
