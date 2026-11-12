//! The hyper-connection mixer's down matmul in prompt steps with BF16 streams (#1183, `legion
//! serve --hc-bf16`): `lo = down @ xn` for the BF16 down weight (`[e · hc, rank]`, Qwen3.8-Flash-
//! Next's NVFP4 checkpoints: 10240 x 320) with `xn` rebuilt from the streams, so the
//! combine-and-norm before it does not write `xn` as BF16.
//!
//! Without it, [`super::hc_norm::hc_post_norm_rebuilt_xnb`] writes `xnb = bf16((v · scale) ·
//! g)` per value (20 KB a token; the norm kernels are memory bound there) and ggml-cuda's
//! cuBLAS BF16 GEMM reads it back. Here one external op ([`mul_mat`], kernel in
//! `src/cuda/hc_down.cu`) computes the GEMM on Legion's own `mma.sync` code with the B operand
//! made in registers from the BF16 streams, each stream's norm scale and gamma, with the norm's
//! own two rounded multiplies and its BF16 rounding: `xnb`'s bits (checked bit for bit against
//! it, `legion hc-down-bench --check` on a GPU and this module's tests on the CPU). Each output
//! is one chain over `k` from 0. On a GB10 cuBLAS sums the same way at 2,048, 3,971 and 4,096
//! tokens (the outputs were equal bit for bit) and splits `k` on small steps (300 tokens: within
//! F32 rounding, 9e-6), so the [`MIN_TOKENS`] floor also keeps `--hc-bf16`'s output where it was.
//!
//! Taken where [`takes`]: a CUDA build, a compute capability 12.x GPU (the GB10, as
//! `--hc-bf16`'s BF16 up-mix), qwen4exp's shape (rank [`M`], 4 streams of `e` values, `e` a
//! multiple of [`BK`]), steps of [`MIN_TOKENS`] or more and the kernels in the binary. The CPU
//! implementation is a reference (the same rebuilt values, products summed in F32 in `k` order).

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::ggml::{Context, ExternalOp, Tensor, ffi, types};

static HC_DOWN: ExternalOp = ExternalOp {
    name: "hc_down_bf16",
    cpu: cpu_hc_down,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::hc_down),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Rows of the down weight the kernel takes (`HD_M`): the whole weight in one CTA.
pub const M: i64 = 320;

/// Fewest tokens of a step [`takes`] takes. On a GB10 the norm and down matmul together take
/// 0.89x main's time at 1,536 tokens, 0.83x at 2,048 and 0.76x at 4,096, but 1.04x at 1,024 (the
/// CTAs fill too few of the 48 SMs). From 2,048 tokens up the output was also cuBLAS's bit for
/// bit, which it is not on split-`k` steps.
pub const MIN_TOKENS: i64 = 2048;

/// `k` values a shared-memory stage holds (`HD_BK`); each stream's length is a multiple.
pub const BK: i64 = 32;

/// The streams the kernel takes (the scales' row).
const HC: i64 = 4;

/// Whether graphs built from now on take [`mul_mat`] where [`takes`] (on by default).
static ON: AtomicBool = AtomicBool::new(true);

/// The kernel's token tile forced (2, 3 or 4 n8 tiles a warp; 0: chosen by the step's size).
static FORCE_NT: AtomicI32 = AtomicI32::new(0);

/// Sets whether graphs built from now on take [`mul_mat`] (for `legion hc-down-bench` and the
/// tests; `--hc-bf16` turns the path on).
#[doc(hidden)]
pub fn set_on(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// Forces the kernel's token tile (`nt` n8 tiles a warp, 32 · `nt` tokens a CTA: 2, 3 or 4);
/// 0 goes back to the choice by step size (for `legion hc-down-bench`).
#[doc(hidden)]
pub fn force_nt(nt: i32) {
    FORCE_NT.store(nt, Ordering::Relaxed);
}

/// The token tile for `n` tokens on `sms` SMs: the fewest waves of CTAs times a CTA's cost,
/// taken as `nt + 2` (the products grow with the tile, the weight's loads and fragments do not),
/// the smaller tile on ties. One CTA per SM (its shared memory).
pub fn pick_nt(n: i64, sms: i64) -> i32 {
    let forced = FORCE_NT.load(Ordering::Relaxed);
    if (2..=4).contains(&forced) {
        return forced;
    }
    let sms = sms.max(1);
    (2..=4)
        .min_by_key(|&nt| {
            let tiles = (n + 32 * nt - 1) / (32 * nt);
            (tiles + sms - 1) / sms * (nt + 2)
        })
        .unwrap_or(3) as i32
}

/// Whether [`mul_mat`] takes the down weight `down`, the gamma `gamma` and the BF16 streams of
/// shape `streams` (`[e, hc, n]`): its shapes ([`shape_fits`]), the weights in one GPU's memory
/// (16-byte aligned), [`MIN_TOKENS`] or more tokens, a compute capability 12.x GPU with the
/// kernels in this binary, and switched on.
pub(super) fn takes(down: Tensor<'_>, gamma: Tensor<'_>, streams: [i64; 4]) -> bool {
    // SAFETY: live tensors; reads a field.
    let aligned = |t: Tensor<'_>| unsafe { (*t.raw()).data as usize }.is_multiple_of(16);
    ON.load(Ordering::Relaxed)
        && cfg!(feature = "cuda")
        && streams[2] >= MIN_TOKENS
        && shape_fits(down.ty(), down.ne(), down.nb(), streams)
        && down.is_contiguous()
        && gamma.ty() == types::F32
        && gamma.is_contiguous()
        && gamma.elements() == down.ne()[0] as usize
        && !down.is_host()
        && !down.is_meta()
        && crate::model::cluster_select::on_cuda()
        && aligned(down)
        && aligned(gamma)
        && down
            .cuda_device()
            .is_some_and(|dev| crate::hw::per_device::sms_if_cc12(dev) > 0 && supported_on(dev))
}

/// Whether the kernel takes a down weight of type `ty`, shape `ne` and strides `nb` times BF16
/// streams of shape `streams` (`[e, hc, n]`): BF16 `[e · hc, M]` with 16-byte rows, 4 streams of
/// a multiple of [`BK`] values, at least one token and below 2^31.
pub fn shape_fits(ty: ffi::ggml_type, ne: [i64; 4], nb: [usize; 4], streams: [i64; 4]) -> bool {
    let [k, m, w2, w3] = ne;
    let [e, hc, n, s3] = streams;
    ty == types::BF16
        && [w2, w3, s3] == [1, 1, 1]
        && m == M
        && hc == HC
        && e > 0
        && e % BK == 0
        && k == e * hc
        && (1..=i64::from(i32::MAX)).contains(&n)
        && nb[0] == 2
        && nb[1].is_multiple_of(16)
        && nb[1] >= 2 * k as usize
}

/// Whether device `dev` runs the kernels (sm_80+, the code in this binary, its shared memory
/// granted).
pub fn supported_on(dev: usize) -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::supported_on(dev)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = dev;
        false
    }
}

/// `lo = down @ xn` (F32 `[M, n]`, as `mul_mat(down, xnb)` shapes it) with `xn` rebuilt from the
/// BF16 streams `res` (`[e, hc, n]`, contiguous; a view of the norm's node), their norm scales
/// `scales` (F32 `[hc, n]`) and `gamma` (F32, `e · hc` values) as
/// [`super::hc_norm::hc_post_norm_b16_streams`]'s norm computes and rounds it. Needs
/// [`shape_fits`].
pub(super) fn mul_mat<'a>(
    ctx: &'a Context,
    down: Tensor<'a>,
    res: Tensor<'a>,
    scales: Tensor<'a>,
    gamma: Tensor<'a>,
) -> Tensor<'a> {
    let [e, hc, n, _] = res.ne();
    assert!(
        shape_fits(down.ty(), down.ne(), down.nb(), res.ne()),
        "hc_down: down {:?}, streams {:?}",
        down.ne(),
        res.ne()
    );
    assert_eq!(res.ty(), types::BF16, "hc_down: BF16 streams");
    assert!(res.is_contiguous(), "hc_down: contiguous streams");
    assert_eq!(scales.ty(), types::F32, "hc_down: F32 scales");
    assert_eq!(scales.ne(), [hc, n, 1, 1], "hc_down: scales");
    assert_eq!(scales.nb()[0], 4, "hc_down: scales' rows contiguous");
    assert_eq!(gamma.ty(), types::F32, "hc_down: F32 gamma");
    assert_eq!(gamma.elements(), (e * hc) as usize, "hc_down: gamma");
    assert!(gamma.is_contiguous(), "hc_down: contiguous gamma");
    ctx.external(
        &HC_DOWN,
        types::F32,
        [M, n, 1, 1],
        &[down, res, scales, gamma],
    )
}

/// One value of `xn` as the BF16 streams' norm rounds it into `xnb` (`hc_post_norm_b16_kernel`)
/// and the kernel rebuilds it: `bf16((v · scale) · g)`, each multiply rounded, then rounded to
/// the nearest BF16 (even on ties). `v` holds BF16 bits.
pub fn rebuild(v: u16, scale: f32, g: f32) -> u16 {
    let x = f32::from_bits(u32::from(v) << 16);
    crate::model::nvfp4_mm::bf16_bits((x * scale) * g)
}

unsafe fn cpu_hc_down(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `mul_mat`, host tensors.
    unsafe {
        let (w, x, sc, g) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
        let (k, m) = (w.ne[0] as usize, w.ne[1] as usize);
        let (e, hc, n) = (x.ne[0] as usize, x.ne[1] as usize, x.ne[2] as usize);
        let gp = std::slice::from_raw_parts(g.data as *const f32, k);
        let mut b = vec![0f32; k];
        for t in (ith..n).step_by(nth) {
            let xr =
                std::slice::from_raw_parts((x.data as *const u8).add(t * x.nb[2]) as *const u16, k);
            let sr = (sc.data as *const u8).add(t * sc.nb[1]) as *const f32;
            for h in 0..hc {
                let s = *sr.add(h);
                for i in h * e..(h + 1) * e {
                    b[i] = f32::from_bits(u32::from(rebuild(xr[i], s, gp[i])) << 16);
                }
            }
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(t * dst.nb[1]) as *mut f32,
                m,
            );
            for (r, o) in out.iter_mut().enumerate() {
                let wr = std::slice::from_raw_parts(
                    (w.data as *const u8).add(r * w.nb[1]) as *const u16,
                    k,
                );
                *o = wr.iter().zip(&b).fold(0.0f32, |s, (&wv, &bv)| {
                    f32::from_bits(u32::from(wv) << 16).mul_add(bv, s)
                });
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The kernel's launch: `cudaLaunchKernel` from Rust (no host code in the `.cu`).

    use std::os::raw::{c_int, c_void};
    use std::sync::atomic::{AtomicU8, Ordering};

    use super::{BK, HC, M, ffi};

    /// `legion_hc_down` (src/cuda/hc_down.cu).
    #[repr(C)]
    struct Params {
        w: *const c_void,
        x: *const c_void,
        sc: *const f32,
        g: *const f32,
        dst: *mut f32,
        w_ld: i64,
        x_ld: i64,
        sc_ld: i64,
        dst_ld: i64,
        m: i32,
        k: i32,
        n: i32,
        e: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    /// `cudaFuncAttributes`' leading fields (the rest padding).
    #[repr(C)]
    struct FuncAttributes {
        shared: usize,
        constant: usize,
        local: usize,
        max_threads: c_int,
        regs: c_int,
        ptx_version: c_int,
        binary_version: c_int,
        rest: [u8; 256],
    }

    const DEV_ATTR_CC_MAJOR: c_int = 75;
    const FUNC_ATTR_MAX_DYNAMIC_SMEM: c_int = 8;
    /// `cudaFuncAttributePreferredSharedMemoryCarveout` and `cudaSharedmemCarveoutMaxShared`.
    const FUNC_ATTR_CARVEOUT: c_int = 9;
    const CARVEOUT_MAX_SHARED: c_int = 100;
    /// Threads of a CTA (`HD_THREADS`).
    const THREADS: u32 = 512;

    unsafe extern "C" {
        /// The GEMM for NT = 2, 3 and 4.
        static legion_hc_down_kernels: [*const c_void; 3];
        static legion_hc_down_smem: [c_int; 3];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaSetDevice(device: c_int) -> c_int;
        fn cudaGetLastError() -> c_int;
        fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> c_int;
        fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> c_int;
        fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> c_int;
    }

    /// Per device: 0 not asked yet, 1 runs the kernels, 2 does not.
    static CHECKED: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];

    /// Whether the current device runs the kernels; grants their shared memory on it the first
    /// time.
    pub(super) fn supported() -> bool {
        let mut dev = 0;
        // SAFETY: plain queries of the current device; the attributes set are the kernels' own.
        unsafe {
            if cudaGetDevice(&mut dev) != 0 || !(0..64).contains(&dev) {
                cudaGetLastError();
                return false;
            }
            let slot = &CHECKED[dev as usize];
            match slot.load(Ordering::Relaxed) {
                1 => return true,
                2 => return false,
                _ => {}
            }
            let mut major = 0;
            let mut ok =
                cudaDeviceGetAttribute(&mut major, DEV_ATTR_CC_MAJOR, dev) == 0 && major >= 8;
            for (i, &f) in legion_hc_down_kernels.iter().enumerate() {
                if !ok {
                    break;
                }
                let mut a: FuncAttributes = std::mem::zeroed();
                ok = cudaFuncGetAttributes(&mut a, f) == 0
                    && a.binary_version >= 80
                    && cudaFuncSetAttribute(f, FUNC_ATTR_MAX_DYNAMIC_SMEM, legion_hc_down_smem[i])
                        == 0;
                if ok {
                    // A hint; the kernels run without it.
                    cudaFuncSetAttribute(f, FUNC_ATTR_CARVEOUT, CARVEOUT_MAX_SHARED);
                }
            }
            cudaGetLastError();
            slot.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }

    /// [`supported`] for device `dev`, the current device put back.
    pub(super) fn supported_on(dev: usize) -> bool {
        let Ok(dev) = c_int::try_from(dev) else {
            return false;
        };
        let mut was = 0;
        // SAFETY: device queries; the current device is put back before returning.
        unsafe {
            if cudaGetDevice(&mut was) != 0 {
                cudaGetLastError();
                return false;
            }
            if was == dev {
                return supported();
            }
            if cudaSetDevice(dev) != 0 {
                cudaGetLastError();
                return false;
            }
            let ok = supported();
            cudaSetDevice(was);
            ok
        }
    }

    pub(super) unsafe fn hc_down(
        dst: &ffi::ggml_tensor,
        stream: *mut c_void,
    ) -> Result<(), String> {
        // SAFETY: a node made by `mul_mat`, device tensors of the stream's device; the kernel
        // reads and writes within the shapes checked here.
        unsafe {
            let (w, x, sc, g) = (&*dst.src[0], &*dst.src[1], &*dst.src[2], &*dst.src[3]);
            let (k, m) = (w.ne[0], w.ne[1]);
            let (e, hc, n) = (x.ne[0], x.ne[1], x.ne[2]);
            let ok = m == M
                && hc == HC
                && e % BK == 0
                && k == e * hc
                && (1..=i64::from(i32::MAX)).contains(&n)
                && w.nb[0] == 2
                && w.nb[1].is_multiple_of(16)
                && x.nb[0] == 2
                && x.nb[1] as i64 == 2 * e
                && x.nb[2].is_multiple_of(16)
                && sc.nb[0] == 4
                && sc.nb[1].is_multiple_of(4)
                && dst.nb[1].is_multiple_of(4)
                && (w.data as usize).is_multiple_of(16)
                && (x.data as usize).is_multiple_of(16)
                && (g.data as usize).is_multiple_of(16);
            if !ok {
                return Err("hc_down: shape or alignment the kernel does not take".into());
            }
            // Also grants the kernels' shared memory on this device.
            if !supported() {
                return Err("hc_down: this GPU does not run the kernel".into());
            }
            let sms = crate::hw::per_device::current().map_or(0, crate::hw::per_device::sm_count);
            let nt = super::pick_nt(n, i64::from(sms));
            let i = (nt - 2) as usize;
            let mut p = Params {
                w: w.data as *const c_void,
                x: x.data as *const c_void,
                sc: sc.data as *const f32,
                g: g.data as *const f32,
                dst: dst.data as *mut f32,
                w_ld: (w.nb[1] / 2) as i64,
                x_ld: (x.nb[2] / 2) as i64,
                sc_ld: (sc.nb[1] / 4) as i64,
                dst_ld: (dst.nb[1] / 4) as i64,
                m: m as i32,
                k: k as i32,
                n: n as i32,
                e: e as i32,
            };
            let tile = 32 * i64::from(nt);
            let grid = Dim3 {
                x: ((n + tile - 1) / tile) as u32,
                y: 1,
                z: 1,
            };
            let block = Dim3 {
                x: THREADS,
                y: 1,
                z: 1,
            };
            let mut args = [(&mut p as *mut Params).cast::<c_void>()];
            // SAFETY: the kernel takes one argument of type `Params`; the launch copies it.
            match cudaLaunchKernel(
                legion_hc_down_kernels[i],
                grid,
                block,
                args.as_mut_ptr(),
                legion_hc_down_smem[i] as usize,
                stream,
            ) {
                0 => Ok(()),
                e => Err(format!("hc_down: CUDA error {e}")),
            }
        }
    }
}

pub mod bench;

#[cfg(test)]
mod tests;
