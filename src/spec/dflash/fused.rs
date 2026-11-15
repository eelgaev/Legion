//! Two ops of the DFlash 2 draft pass on CUDA, each in place of a chain of ggml-cuda kernels
//! (#1181; kernels in `src/cuda/dflash_draft.cu`):
//!
//! - [`top_k`]: the head's top-k candidates of every draft row. ggml-cuda's radix top-k reads
//!   each row five times (four histogram passes and a gather); this op reads it once, each CTA
//!   keeping the best `k` of its part of the row and one CTA a row joining the parts. At 16
//!   sequences (112 rows of Gemma 4's 262,144 logits) that is ~0.5 ms for ~2.5 ms on a DGX Spark
//!   (GB10). The keys are ggml's, so the set is ggml's; at a tie for the k-th place, which
//!   ggml's gather settles by the order its atomics land in, the lower token wins. The indexes
//!   come best first (ggml's in no order; [`super::v2::Selector::walk`] sorts them anyway).
//! - [`conv`]: one side of the grouped dynamic causal convolution ([`super::v2::conv`]) in one
//!   kernel, for ggml's pad, three broadcast products and sums a tap and the sum over the taps
//!   (~10 kernels, 4 convolutions a layer). Every product and sum is rounded on its own, in
//!   ggml's order, so the bits are ggml's.
//!
//! Both run only where the drafter's weights sit on a GPU ([`on_gpu`]); the CPU keeps ggml's
//! ops. Their CPU implementations exist for the tests.

use std::os::raw::c_void;

use crate::ggml::{Context, ExternalOp, Tensor, external_params, ffi, types};

use super::v2::Dflash2;

static TOPK: ExternalOp = ExternalOp {
    name: "dflash_topk",
    cpu: cpu_topk,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::topk),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

static CONV: ExternalOp = ExternalOp {
    name: "dflash_conv",
    cpu: cpu_conv,
    #[cfg(feature = "cuda")]
    cuda: Some(cuda::conv),
    #[cfg(not(feature = "cuda"))]
    cuda: None,
};

/// Most candidates a row [`top_k`] takes (a warp's list).
pub const MAX_TOPK: i64 = 32;

/// Fewest columns a row [`top_k`] takes: vocabularies (shorter rows stay on ggml's top-k).
pub const MIN_TOPK_COLS: i64 = 16384;

/// Columns a CTA of [`top_k`] reads, unless a row needs more parts than the join takes.
const TOPK_SPAN: i64 = 16384;

/// Candidates the join of one row takes at most (its CTA's threads).
const TOPK_JOIN: i64 = 1024;

/// Bytes of the parts' lists a device and stream keep for [`top_k`] (8 bytes a candidate).
pub const TOPK_PART_BYTES: usize = 4 << 20;

/// Whether `w` (a weight of the drafter) is a CUDA device's (so the drafter computes there),
/// not split over GPUs; and this build has CUDA.
pub fn on_gpu(w: Tensor<'_>) -> bool {
    cfg!(feature = "cuda") && w.cuda_device().is_some() && !w.is_meta()
}

/// `(parts, span)` of [`top_k`] for rows of `n` columns and `k` candidates: parts of `span`
/// columns (a multiple of 4) each, at most [`TOPK_JOIN`] / `k` of them.
pub fn topk_split(n: i64, k: i64) -> (i64, i64) {
    let most = (TOPK_JOIN / k.max(1)).max(1);
    let parts = ((n + TOPK_SPAN - 1) / TOPK_SPAN).clamp(1, most);
    let span = ((n + parts - 1) / parts + 3) / 4 * 4;
    ((n + span - 1) / span, span)
}

/// The shape test of [`top_k`]: F32 `[n, rows]` with rows 16-byte aligned, `n` a multiple of 4
/// and at least [`MIN_TOPK_COLS`], `1 <= k <= `[`MAX_TOPK`], the parts' lists within
/// [`TOPK_PART_BYTES`].
pub fn topk_fits(ty: ffi::ggml_type, ne: [i64; 4], nb: [usize; 4], k: i64) -> bool {
    let (n, rows) = (ne[0], ne[1]);
    let (parts, _) = topk_split(n, k);
    ty == types::F32
        && nb[0] == 4
        && nb[1] % 16 == 0
        && ne[2] == 1
        && ne[3] == 1
        && n % 4 == 0
        && (MIN_TOPK_COLS..=i32::MAX as i64).contains(&n)
        && (1..=MAX_TOPK).contains(&k)
        && rows >= 1
        && (rows * parts * k * 8) as u64 <= TOPK_PART_BYTES as u64
}

/// Each row's `k` largest values of `logits` (F32 `[n, rows]`) as I32 `[k, rows]`: this op
/// where it fits and the head `w` is on a GPU, else ggml's top-k. The same set as ggml's but at
/// a tie for the k-th place; best first.
pub fn top_k<'a>(ctx: &'a Context, logits: Tensor<'a>, k: i64, w: Tensor<'_>) -> Tensor<'a> {
    if on_gpu(w) && topk_fits(logits.ty(), logits.ne(), logits.nb(), k) {
        top_k_op(ctx, logits, k)
    } else {
        ctx.top_k(logits, k as i32)
    }
}

/// The op of [`top_k`] whatever the backend.
pub(crate) fn top_k_op<'a>(ctx: &'a Context, logits: Tensor<'a>, k: i64) -> Tensor<'a> {
    ctx.external(&TOPK, types::I32, [k, logits.ne()[1], 1, 1], &[logits])
}

/// The shape test of [`conv`]: F32 `x` `[e, n]` (rows of positions, any stride a multiple of
/// 4 bytes), contiguous F32 `dynamic` `[2 · kernel · groups, n]` and `base` `[e, kernel, 2]`,
/// `e` a multiple of the group, `n` of the block.
pub fn conv_fits(
    x: Tensor<'_>,
    dynamic: Tensor<'_>,
    base: Tensor<'_>,
    v2: &Dflash2,
    block: i64,
) -> bool {
    let [e, n, x2, x3] = x.ne();
    let (k, group) = (v2.conv_kernel as i64, v2.conv_group as i64);
    group > 0
        && k > 0
        && e % group == 0
        && block > 0
        && n % block == 0
        && x.ty() == types::F32
        && x.nb()[0] == 4
        && x.nb()[1] % 4 == 0
        && x2 == 1
        && x3 == 1
        && dynamic.ty() == types::F32
        && dynamic.is_contiguous()
        && dynamic.ne() == [2 * k * (e / group), n, 1, 1]
        && base.ty() == types::F32
        && base.is_contiguous()
        && base.ne() == [e, k, 2, 1]
        && e <= i32::MAX as i64
        && n <= u16::MAX as i64
}

/// Whether [`conv`] takes one side of the convolution ([`super::v2::conv`]'s arguments, `base`
/// in F32): [`conv_fits`], and `w` (the layer's kernel projection) on a GPU.
pub fn takes_conv(
    w: Tensor<'_>,
    x: Tensor<'_>,
    dynamic: Tensor<'_>,
    base: Tensor<'_>,
    v2: &Dflash2,
    block: i64,
) -> bool {
    on_gpu(w) && conv_fits(x, dynamic, base, v2, block)
}

/// One side of the convolution as one op ([`super::v2::conv`]'s arguments and bits).
pub fn conv<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    dynamic: Tensor<'a>,
    base: Tensor<'a>,
    side: usize,
    v2: &Dflash2,
    block: i64,
) -> Tensor<'a> {
    let [e, n, _, _] = x.ne();
    let k = v2.conv_kernel as i32;
    let params = [
        side as i32,
        k,
        v2.conv_group as i32,
        block as i32,
        k.min(block as i32),
    ];
    ctx.external_with(
        &CONV,
        types::F32,
        [e, n, 1, 1],
        &[x, dynamic, base],
        &params,
    )
}

/// The arguments of a [`conv`] node: `(side, kernel, group, block, taps)`.
fn conv_params(dst: &ffi::ggml_tensor) -> (usize, usize, usize, usize, usize) {
    let p = external_params(dst);
    (
        p[0] as usize,
        p[1] as usize,
        p[2] as usize,
        p[3] as usize,
        p[4] as usize,
    )
}

/// A candidate as one integer, as the kernel packs it: ggml's ordered key of the value
/// (`top_k_float_to_ordered`), then the inverted column (lower columns rank higher).
pub(crate) fn topk_key(v: f32, col: usize) -> u64 {
    let bits = v.to_bits();
    let mask = (((bits >> 31) as i32).wrapping_neg() as u32) | 0x8000_0000;
    (((bits ^ mask) as u64) << 32) | (!(col as u32)) as u64
}

/// One row of [`top_k`]: the `out.len()` best columns of `row`, best first.
pub(crate) fn top_k_row(row: &[f32], out: &mut [i32]) {
    let mut keys: Vec<u64> = row
        .iter()
        .enumerate()
        .map(|(c, &v)| topk_key(v, c))
        .collect();
    let k = out.len();
    keys.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
    keys.truncate(k);
    keys.sort_unstable_by(|a, b| b.cmp(a));
    for (o, key) in out.iter_mut().zip(keys) {
        *o = !(key as u32) as i32;
    }
}

/// One value of [`conv`] as the kernel computes it: per tap `x · d + x · b`, the taps summed in
/// order, each product and sum rounded on its own.
pub(crate) fn conv_value(x: &[f32], d: &[f32], b: &[f32]) -> f32 {
    let mut out = 0.0f32;
    for tap in 0..x.len() {
        let term = x[tap] * d[tap] + x[tap] * b[tap];
        out = if tap == 0 { term } else { out + term };
    }
    out
}

unsafe fn cpu_topk(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `top_k_op`, host tensors.
    unsafe {
        let x = &*dst.src[0];
        let (n, k, rows) = (x.ne[0] as usize, dst.ne[0] as usize, dst.ne[1] as usize);
        for r in (ith..rows).step_by(nth) {
            let row =
                std::slice::from_raw_parts((x.data as *const u8).add(r * x.nb[1]) as *const f32, n);
            let out = std::slice::from_raw_parts_mut(
                (dst.data as *mut u8).add(r * dst.nb[1]) as *mut i32,
                k,
            );
            top_k_row(row, out);
        }
    }
}

unsafe fn cpu_conv(dst: &ffi::ggml_tensor, ith: usize, nth: usize) {
    // SAFETY: a node made by `conv`, host tensors (shapes as `conv_fits` checks).
    unsafe {
        let (x, dynamic, base) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
        let (side, k, group, block, taps) = conv_params(dst);
        let (e, n) = (x.ne[0] as usize, x.ne[1] as usize);
        let groups = e / group;
        let at = |p: *mut c_void, off: usize| *((p as *const u8).add(off) as *const f32);
        let (mut xv, mut dv, mut bv) = (vec![0.0; taps], vec![0.0; taps], vec![0.0; taps]);
        for t in (ith..n).step_by(nth) {
            let j = t % block;
            for i in 0..e {
                for tap in 0..taps {
                    let s = side * k + tap;
                    xv[tap] = if j >= tap {
                        at(x.data, (t - tap) * x.nb[1] + i * 4)
                    } else {
                        0.0
                    };
                    dv[tap] = at(
                        dynamic.data,
                        t * dynamic.nb[1] + (s * groups + i / group) * 4,
                    );
                    bv[tap] = at(base.data, (s * e + i) * 4);
                }
                *((dst.data as *mut u8).add(t * dst.nb[1] + i * 4) as *mut f32) =
                    conv_value(&xv, &dv, &bv);
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The kernels' launches (`cudaLaunchKernel` from Rust; no host code in the `.cu`).

    use std::os::raw::{c_int, c_void};

    use super::ffi;

    /// `legion_draft_topk` (`src/cuda/dflash_draft.cu`).
    #[repr(C)]
    struct TopK {
        x: *const f32,
        x_nb1: i64,
        dst: *mut i32,
        dst_nb1: i64,
        part: *mut u64,
        n: i32,
        k: i32,
        rows: i32,
        parts: i32,
        span: i32,
    }

    /// `legion_draft_conv`.
    #[repr(C)]
    struct Conv {
        x: *const f32,
        x_nb1: i64,
        dynamic: *const f32,
        dynamic_nb1: i64,
        base: *const f32,
        dst: *mut f32,
        dst_nb1: i64,
        e: i32,
        n: i32,
        group: i32,
        groups: i32,
        k: i32,
        taps: i32,
        side: i32,
        block: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Dim3 {
        x: u32,
        y: u32,
        z: u32,
    }

    unsafe extern "C" {
        /// Top-k part, top-k join, convolution.
        static legion_dflash_draft_kernels: [*const c_void; 3];
        fn cudaLaunchKernel(
            func: *const c_void,
            grid: Dim3,
            block: Dim3,
            args: *mut *mut c_void,
            shared: usize,
            stream: *mut c_void,
        ) -> c_int;
        fn cudaGetDevice(device: *mut c_int) -> c_int;
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
    }

    /// Threads of a top-k part CTA.
    const PART_THREADS: u32 = 256;

    /// Threads of a convolution CTA.
    const CONV_THREADS: u32 = 256;

    /// The parts' buffers: (device, stream, buffer of [`super::TOPK_PART_BYTES`]).
    static PARTS: std::sync::Mutex<Vec<(c_int, usize, usize)>> = std::sync::Mutex::new(Vec::new());

    /// The parts' buffer for `stream` on the current device, made at its first use (a stage
    /// graph's capture runs in relaxed mode, where `cudaMalloc` is allowed) and kept: launches
    /// on one stream run in order, and a captured graph replays on its stream.
    fn part_buffer(stream: *mut c_void) -> Result<*mut c_void, String> {
        let mut dev = 0;
        // SAFETY: a plain query of the current device.
        if unsafe { cudaGetDevice(&mut dev) } != 0 {
            return Err("dflash_topk: cudaGetDevice failed".into());
        }
        let mut parts = PARTS.lock().unwrap();
        if let Some(&(_, _, p)) = parts
            .iter()
            .find(|&&(d, s, _)| d == dev && s == stream as usize)
        {
            return Ok(p as *mut c_void);
        }
        let mut p = std::ptr::null_mut();
        // SAFETY: a plain allocation on the current device, kept for the process.
        match unsafe { cudaMalloc(&mut p, super::TOPK_PART_BYTES) } {
            0 => {
                parts.push((dev, stream as usize, p as usize));
                Ok(p)
            }
            e => Err(format!("dflash_topk: cudaMalloc: CUDA error {e}")),
        }
    }

    /// Launches kernel `which` on `grid` CTAs of `threads` threads with `p`.
    unsafe fn launch<P>(
        name: &str,
        which: usize,
        grid: Dim3,
        threads: u32,
        p: &mut P,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let block = Dim3 {
            x: threads,
            y: 1,
            z: 1,
        };
        let mut args = [(p as *mut P).cast::<c_void>()];
        // SAFETY: the kernel takes one argument of type `P`; the launch copies it.
        match unsafe {
            cudaLaunchKernel(
                legion_dflash_draft_kernels[which],
                grid,
                block,
                args.as_mut_ptr(),
                0,
                stream,
            )
        } {
            0 => Ok(()),
            e => Err(format!("{name}: CUDA error {e}")),
        }
    }

    pub(super) unsafe fn topk(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `top_k_op`, device tensors of the stream's device.
        unsafe {
            let x = &*dst.src[0];
            let (n, k, rows) = (x.ne[0], dst.ne[0], dst.ne[1]);
            let ne = [x.ne[0], x.ne[1], x.ne[2], x.ne[3]];
            let nb = [x.nb[0], x.nb[1], x.nb[2], x.nb[3]];
            if !super::topk_fits(x.type_, ne, nb, k) || rows != x.ne[1] || x.data as usize % 16 != 0
            {
                return Err("dflash_topk: unsupported shape".into());
            }
            let (parts, span) = super::topk_split(n, k);
            let mut p = TopK {
                x: x.data as *const f32,
                x_nb1: x.nb[1] as i64,
                dst: dst.data as *mut i32,
                dst_nb1: dst.nb[1] as i64,
                part: part_buffer(stream)?.cast(),
                n: n as i32,
                k: k as i32,
                rows: rows as i32,
                parts: parts as i32,
                span: span as i32,
            };
            let grid = Dim3 {
                x: parts as u32,
                y: rows as u32,
                z: 1,
            };
            launch("dflash_topk", 0, grid, PART_THREADS, &mut p, stream)?;
            let threads = ((parts * k) as u32).div_ceil(32) * 32;
            let grid = Dim3 {
                x: rows as u32,
                y: 1,
                z: 1,
            };
            launch("dflash_topk", 1, grid, threads, &mut p, stream)
        }
    }

    pub(super) unsafe fn conv(dst: &ffi::ggml_tensor, stream: *mut c_void) -> Result<(), String> {
        // SAFETY: a node made by `conv`, device tensors of the stream's device.
        unsafe {
            let (x, dynamic, base) = (&*dst.src[0], &*dst.src[1], &*dst.src[2]);
            let (side, k, group, block, taps) = super::conv_params(dst);
            let (e, n) = (x.ne[0], x.ne[1]);
            if group == 0 || block == 0 || e % group as i64 != 0 || x.nb[0] != 4 {
                return Err("dflash_conv: unsupported shape".into());
            }
            let mut p = Conv {
                x: x.data as *const f32,
                x_nb1: x.nb[1] as i64,
                dynamic: dynamic.data as *const f32,
                dynamic_nb1: dynamic.nb[1] as i64,
                base: base.data as *const f32,
                dst: dst.data as *mut f32,
                dst_nb1: dst.nb[1] as i64,
                e: e as i32,
                n: n as i32,
                group: group as i32,
                groups: (e / group as i64) as i32,
                k: k as i32,
                taps: taps as i32,
                side: side as i32,
                block: block as i32,
            };
            let grid = Dim3 {
                x: (e as u32).div_ceil(CONV_THREADS),
                y: n as u32,
                z: 1,
            };
            launch("dflash_conv", 2, grid, CONV_THREADS, &mut p, stream)
        }
    }
}

#[cfg(test)]
mod tests;
