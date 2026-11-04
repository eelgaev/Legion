//! Legion's coalesced/vectorized CUDA dequantizers (`src/cuda/dequant.cu`) against ggml's
//! original ones, byte for byte.
//!
//! The build links with `--wrap` on ggml's `ggml_get_to_{fp16,bf16,fp32}_cuda` dispatchers
//! (build/cuda_wrap.rs), so `__wrap_<symbol>` is Legion's dispatcher and `__real_<symbol>` is
//! ggml's. These dispatchers serve cuBLAS mul_mat, not graph casts (and ggml's CPU backend only
//! dequantizes to F32), so the test calls both directly on the same device buffer.
//!
//! `cargo test --release --features cuda --test dequant_cuda -- --ignored --test-threads 1`

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr::null_mut;

use legion::ggml::{quantize, types};

/// `void (*)(const void * x, T * y, int64_t k, cudaStream_t stream)` from ggml-cuda/convert.cuh.
type ToT = unsafe extern "C" fn(*const c_void, *mut c_void, i64, *mut c_void);
/// `to_t_cuda_t<T> ggml_get_to_*_cuda(ggml_type)`: C++ linkage, C calling convention.
type Getter = unsafe extern "C" fn(u32) -> Option<ToT>;

unsafe extern "C" {
    #[link_name = "__wrap__Z21ggml_get_to_fp16_cuda9ggml_type"]
    fn wrap_fp16(ty: u32) -> Option<ToT>;
    #[link_name = "__real__Z21ggml_get_to_fp16_cuda9ggml_type"]
    fn real_fp16(ty: u32) -> Option<ToT>;
    #[link_name = "__wrap__Z21ggml_get_to_bf16_cuda9ggml_type"]
    fn wrap_bf16(ty: u32) -> Option<ToT>;
    #[link_name = "__real__Z21ggml_get_to_bf16_cuda9ggml_type"]
    fn real_bf16(ty: u32) -> Option<ToT>;
    #[link_name = "__wrap__Z21ggml_get_to_fp32_cuda9ggml_type"]
    fn wrap_fp32(ty: u32) -> Option<ToT>;
    #[link_name = "__real__Z21ggml_get_to_fp32_cuda9ggml_type"]
    fn real_fp32(ty: u32) -> Option<ToT>;

    fn cudaMalloc(p: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMemset(p: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

const HOST_TO_DEVICE: i32 = 1;
const DEVICE_TO_HOST: i32 = 2;

fn ck(err: i32, what: &str) {
    assert_eq!(err, 0, "{what} failed with CUDA error {err}");
}

/// A device allocation, freed on drop.
struct DevBuf(*mut c_void, usize);

impl DevBuf {
    fn new(len: usize, fill: u8) -> Self {
        let mut p = null_mut();
        // SAFETY: plain cudart calls on a fresh allocation of `len` bytes.
        unsafe {
            ck(cudaMalloc(&mut p, len), "cudaMalloc");
            ck(cudaMemset(p, i32::from(fill), len), "cudaMemset");
        }
        Self(p, len)
    }

    /// `bytes` followed by `pad` zero bytes (the device size is `bytes.len() + pad`).
    fn from_host(bytes: &[u8], pad: usize) -> Self {
        let buf = Self::new(bytes.len() + pad, 0);
        // SAFETY: `buf` holds at least `bytes.len()` bytes.
        unsafe {
            ck(
                cudaMemcpy(buf.0, bytes.as_ptr().cast(), bytes.len(), HOST_TO_DEVICE),
                "cudaMemcpy",
            );
        }
        buf
    }

    fn to_host(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.1];
        // SAFETY: `out` holds exactly the allocation's size.
        unsafe {
            ck(
                cudaMemcpy(out.as_mut_ptr().cast(), self.0, self.1, DEVICE_TO_HOST),
                "cudaMemcpy",
            );
        }
        out
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by cudaMalloc, freed once.
        unsafe { cudaFree(self.0) };
    }
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed | 1;
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * (1.0 + (i % 31) as f32)
        })
        .collect()
}

/// Runs `f` over `n` values of `src` into a buffer prefilled with `fill`, returning its bytes.
fn run(f: ToT, src: &DevBuf, n: usize, elem: usize, fill: u8) -> Vec<u8> {
    let dst = DevBuf::new(n * elem, fill);
    // SAFETY: `src` holds `n` quantized values and `dst` `n` outputs; default stream.
    unsafe {
        f(src.0, dst.0, n as i64, null_mut());
        ck(cudaDeviceSynchronize(), "dequant kernel");
    }
    dst.to_host()
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn optimized_conversions_are_bit_identical() {
    let dsts: [(&str, usize, Getter, Getter); 3] = [
        ("F16", 2, wrap_fp16, real_fp16),
        ("BF16", 2, wrap_bf16, real_bf16),
        ("F32", 4, wrap_fp32, real_fp32),
    ];
    let srcs = [
        ("Q4_K", types::Q4_K),
        ("Q5_K", types::Q5_K),
        ("Q5_1", types::Q5_1),
        ("Q8_0", types::Q8_0),
    ];
    // Many CTAs, a partial last CTA of eight superblocks, and for the 32-value formats a
    // partial final 256-value group.
    let sizes: [usize; 4] = [17 * 256, 17 * 256 + 32, 4099 * 256, 4099 * 256 + 96];
    let mut checked = 0;
    for (si, &(src_name, src_type)) in srcs.iter().enumerate() {
        for (ni, &n) in sizes.iter().enumerate() {
            if (src_type == types::Q4_K || src_type == types::Q5_K) && !n.is_multiple_of(256) {
                continue;
            }
            let raw = values(n, 0x261 + (si * sizes.len() + ni) as u64);
            let q = quantize(src_type, &raw, n);
            // Legion's kernels get an exact-size source, so compute-sanitizer flags any read
            // past the last block. ggml's dequantize_block_q8_0_f16 (the reference here, and the
            // kept Q8_0 -> F16 kernel) loads the blocks as 4-byte ints and reads 2 bytes past
            // the end when the block count is odd. ggml's own buffers never end there (rows of
            // 512 values or row padding), so its runs get a padded copy.
            let exact = DevBuf::from_host(&q, 0);
            let padded = DevBuf::from_host(&q, 4);
            for &(dst_name, elem, wrap, real) in &dsts {
                let case = format!("{src_name} -> {dst_name}, n = {n}");
                // SAFETY: the dispatchers only look the type up.
                let (new, old) = unsafe { (wrap(src_type), real(src_type)) };
                let (new, old) = (new.expect(&case), old.expect(&case));
                // ggml's own Q8_0 -> F16 kernel is kept (already at bandwidth).
                let kept = src_type == types::Q8_0 && dst_name == "F16";
                assert_eq!(
                    new as usize == old as usize,
                    kept,
                    "{case}: wrong dispatcher"
                );
                // Different fills, so a value either kernel leaves unwritten shows up.
                let want = run(old, &padded, n, elem, 0xab);
                let got = run(new, if kept { &padded } else { &exact }, n, elem, 0xcd);
                assert!(got == want, "{case}: outputs differ");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 36);
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn other_types_keep_ggml_dispatchers() {
    for ty in [types::IQ4_NL, types::Q6_K, types::F16] {
        for (wrap, real) in [
            (wrap_fp16 as Getter, real_fp16 as Getter),
            (wrap_bf16, real_bf16),
            (wrap_fp32, real_fp32),
        ] {
            // SAFETY: the dispatchers only look the type up.
            let (new, old) = unsafe { (wrap(ty), real(ty)) };
            assert_eq!(
                new.map(|f| f as usize),
                old.map(|f| f as usize),
                "type {ty}"
            );
        }
    }
}
