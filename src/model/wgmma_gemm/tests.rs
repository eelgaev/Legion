//! The Hopper GEMMs' index math on the CPU (the descriptor bits, the swizzle, the tile order, the
//! shared-memory plan, the shape checks), and on CUDA (ignored test; it skips without a device of
//! compute capability 9.0 and an sm_90a build) the kernels against F64 sums.

use super::*;

/// The descriptor's fields (PTX ISA, "Matrix Descriptor Format"): start address, leading and
/// stride byte offsets in 16-byte units, base offset, swizzle mode.
fn fields(d: u64) -> (u64, u64, u64, u64, u64) {
    (
        d & 0x3FFF,
        (d >> 16) & 0x3FFF,
        (d >> 32) & 0x3FFF,
        (d >> 49) & 0x7,
        d >> 62,
    )
}

#[test]
fn descriptor_bits() {
    let (start, lbo, sbo, base, swizzle) = fields(desc_hi());
    assert_eq!(start, 0, "the kernel ORs in the address");
    assert_eq!(lbo, 1, "16 bytes");
    assert_eq!(sbo, 64, "1,024 bytes: 8 rows of 128 bytes");
    assert_eq!(base, 0, "stages start on 1,024-byte atoms");
    assert_eq!(swizzle, 1, "128-byte swizzle");
    // Bits 14-15, 30-31, 46-48 and 52-61 are reserved.
    let used = 0x3FFFu64 | (0x3FFF << 16) | (0x3FFF << 32) | (0x7 << 49) | (0x3 << 62);
    assert_eq!(desc_hi() & !used, 0);
    // A shared address in 16-byte units; the K slices within a stage step 32 bytes (2 units).
    let a = 0x1_2400;
    assert_eq!(fields(desc(a)).0, (a as u64) >> 4);
    assert_eq!(desc(a + 32) - desc(a), 2);
    assert_eq!(desc(a) & !0x3FFF, desc_hi());
}

#[test]
fn swizzle_permutes_each_row_in_16_byte_chunks() {
    for row in 0..16 {
        let mut seen = [false; 128];
        for byte in 0..128 {
            let at = swizzle_128b(row, byte);
            assert_eq!(at / 128, row, "row {row} stays in its 128 bytes");
            assert!(!seen[at % 128], "row {row} byte {byte}");
            seen[at % 128] = true;
            assert_eq!(at % 16, byte % 16, "chunks keep their bytes in order");
        }
        assert_eq!(swizzle_128b(row, 0) % 128, (row % 8) * 16, "row {row}");
    }
    // The pattern repeats every 8 rows (one 1,024-byte atom, the descriptor's stride).
    for byte in (0..128).step_by(16) {
        assert_eq!(swizzle_128b(9, byte) - swizzle_128b(1, byte), 1024);
    }
}

#[test]
fn tile_order_covers_each_tile_once() {
    for tiles_m in [1, 2, 7, 8, 9, 16, 17, 33, 56] {
        for tiles_n in [1, 2, 3, 5, 32] {
            let mut seen = vec![false; (tiles_m * tiles_n) as usize];
            for pid in 0..tiles_m * tiles_n {
                let (tm, tn) = tile_of(pid, tiles_m, tiles_n);
                assert!(
                    (0..tiles_m).contains(&tm) && (0..tiles_n).contains(&tn),
                    "{tiles_m} x {tiles_n}: pid {pid} -> ({tm}, {tn})"
                );
                let i = (tm * tiles_n + tn) as usize;
                assert!(!seen[i], "{tiles_m} x {tiles_n}: ({tm}, {tn}) twice");
                seen[i] = true;
            }
        }
    }
    // Within a group, consecutive CTAs walk GROUP_M row tiles at one token tile.
    let order: Vec<_> = (0..10).map(|pid| tile_of(pid, 16, 4)).collect();
    assert_eq!(&order[..3], &[(0, 0), (1, 0), (2, 0)]);
    assert_eq!(order[8], (0, 1));
}

#[test]
fn shared_memory_plan() {
    // As the kernel file's static_asserts.
    assert_eq!(Kind::Bf16.smem_bytes(), 197_696);
    assert_eq!(Kind::E4m3.smem_bytes(), 197_728);
    for kind in [Kind::Bf16, Kind::E4m3] {
        assert!(kind.smem_bytes() <= SMEM_LIMIT, "{kind:?}");
        assert_eq!(kind.stage_bytes() % 1024, 0, "{kind:?}: swizzle atoms");
        assert_eq!(kind.bk() * kind.elem_bytes(), ROW_BYTES, "{kind:?}");
        // wgmma n = 128 per instruction, two consumer warpgroups of 64 rows.
        assert_eq!(kind.bn() % 128, 0, "{kind:?}");
    }
    assert_eq!(
        [
            Kind::Bf16.kernel_index(false),
            Kind::Bf16.kernel_index(true),
            Kind::E4m3.kernel_index(false),
            Kind::E4m3.kernel_index(true),
        ],
        [0, 1, 2, 3]
    );
}

#[test]
fn shapes_that_fit() {
    assert!(fits(Kind::E4m3, 7168, 2048, 4096));
    assert!(fits(Kind::E4m3, 128, 1, 65));
    assert!(!fits(Kind::E4m3, 64, 256, 256), "half a K step");
    assert!(!fits(Kind::E4m3, 7168 + 16, 256, 256));
    assert!(fits(Kind::Bf16, 64, 256, 256));
    assert!(!fits(Kind::Bf16, 96, 256, 256));
    assert!(!fits(Kind::Bf16, 0, 256, 256));
    assert!(!fits(Kind::Bf16, 64, 0, 256));
    assert!(!fits(Kind::Bf16, 64, 256, 0));
    assert!(!fits(Kind::E4m3, 1 << 31, 256, 256), "i32 coordinates");
    assert_eq!(tiles(Kind::E4m3, 200, 300), (2, 3));
    assert_eq!(tiles(Kind::Bf16, 200, 300), (2, 2));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn unsupported_without_cuda() {
    assert!(!supported());
}

/// Deterministic values in [-1, 1).
#[cfg(feature = "cuda")]
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

#[cfg(feature = "cuda")]
mod device {
    use std::os::raw::{c_int, c_void};

    unsafe extern "C" {
        fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> c_int;
        fn cudaFree(p: *mut c_void) -> c_int;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> c_int;
        fn cudaDeviceSynchronize() -> c_int;
    }

    /// A device copy of `v`.
    pub fn upload<T: Copy>(v: &[T]) -> *mut c_void {
        let bytes = std::mem::size_of_val(v);
        let mut p = std::ptr::null_mut();
        // SAFETY: a fresh allocation of `bytes`, filled from `v`.
        unsafe {
            assert_eq!(cudaMalloc(&mut p, bytes.max(1)), 0);
            assert_eq!(cudaMemcpy(p, v.as_ptr().cast(), bytes, 1), 0);
        }
        p
    }

    /// `n` F32s from device memory `p`, after the device is idle.
    pub fn download(p: *mut c_void, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        // SAFETY: `p` holds at least `n` F32s.
        unsafe {
            assert_eq!(cudaDeviceSynchronize(), 0, "the kernel failed");
            assert_eq!(cudaMemcpy(v.as_mut_ptr().cast(), p, 4 * n, 2), 0);
        }
        v
    }

    pub fn free(p: *mut c_void) {
        // SAFETY: from `upload`.
        unsafe { cudaFree(p) };
    }
}

/// One GEMM on the device against F64 sums: `w` and `x` the operands' bytes, `val` their values.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn check(
    kind: Kind,
    k: usize,
    m: usize,
    n: usize,
    w: &[u8],
    x: &[u8],
    val: impl Fn(&[u8], usize) -> f64,
    rel: f64,
    seed: u64,
) {
    let scale: Vec<f32> = noise(n, seed).iter().map(|v| 0.5 + v.abs()).collect();
    let wscale: Vec<f32> = noise(m, seed + 1).iter().map(|v| 0.5 + v.abs()).collect();
    let res = noise(m * n, seed + 2);
    for with_res in [false, true] {
        let (dw, dx) = (device::upload(w), device::upload(x));
        let (ds, dws, dr) = (
            device::upload(&scale),
            device::upload(&wscale),
            device::upload(&res),
        );
        let dd = device::upload(&vec![f32::NAN; m * n]);
        let mut a = Args {
            dst: dd.cast(),
            res: if with_res {
                dr.cast_const().cast()
            } else {
                std::ptr::null()
            },
            scale: ds.cast_const().cast(),
            wscale: dws.cast_const().cast(),
            k: k as i64,
            m: m as i64,
            n: n as i64,
            scale_stride: 1,
            wscale_stride: 1,
            desc_hi: 0,
            tiles_m: 0,
            tiles_n: 0,
        };
        // SAFETY: device buffers of the shapes above.
        let ran = unsafe {
            run(
                kind,
                dw.cast_const().cast(),
                dx.cast_const().cast(),
                &mut a,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(ran, Ok(true), "{kind:?} {k} x {m} x {n}");
        let got = device::download(dd, m * n);
        for p in [dw, dx, ds, dws, dr, dd] {
            device::free(p);
        }
        for t in 0..n {
            for r in 0..m {
                let (mut sum, mut abs) = (0f64, 0f64);
                for i in 0..k {
                    let p = val(w, r * k + i) * val(x, t * k + i);
                    sum += p;
                    abs += p.abs();
                }
                let s = scale[t] as f64 * wscale[r] as f64;
                let mut want = sum * s;
                if with_res {
                    want += res[t * m + r] as f64;
                }
                let g = got[t * m + r] as f64;
                assert!(
                    (g - want).abs() <= rel * abs * s + 1e-6,
                    "{kind:?} {k} x {m} x {n} res {with_res}: row {r} token {t}: {g} vs {want}"
                );
            }
        }
    }
}

/// Whether this device and binary run the kernels; why not on stderr.
#[cfg(feature = "cuda")]
fn hopper() -> bool {
    if !supported() {
        eprintln!(
            "wgmma_gemm: needs a GPU of compute capability 9.0 (H100, H200, GH200) and a build \
             with CMAKE_CUDA_ARCHITECTURES=90a; skipped"
        );
        return false;
    }
    true
}

/// The E4M3 kernel against F64 sums of the same codes: edge tiles (rows and tokens past the
/// last full tile), several K blocks, per-token and per-row scales, the residual. The FP8
/// wgmma's narrower accumulator within each 128-deep block bounds the error.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a Hopper GPU"]
fn cuda_e4m3_matches_f64() {
    if !hopper() {
        return;
    }
    let code = |v: f32| crate::model::fp8::f32_to_e4m3(v * 4.0);
    for (k, m, n) in [(128, 128, 128), (512, 200, 300), (1024, 384, 65)] {
        let w: Vec<u8> = noise(m * k, 1).into_iter().map(code).collect();
        let x: Vec<u8> = noise(n * k, 2).into_iter().map(code).collect();
        let val = |b: &[u8], i: usize| crate::model::fp8::e4m3_to_f32(b[i]) as f64;
        check(Kind::E4m3, k, m, n, &w, &x, val, 1e-3, 3);
    }
}

/// The BF16 kernel against F64 sums: edge tiles, several stages around the ring, the residual.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a Hopper GPU"]
fn cuda_bf16_matches_f64() {
    if !hopper() {
        return;
    }
    let bytes = |v: Vec<f32>| -> Vec<u8> {
        v.iter()
            .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    for (k, m, n) in [(64, 128, 256), (640, 200, 300), (2048, 384, 513)] {
        let (w, x) = (bytes(noise(m * k, 4)), bytes(noise(n * k, 5)));
        let val = |b: &[u8], i: usize| {
            f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16) as f64
        };
        check(Kind::Bf16, k, m, n, &w, &x, val, 1e-5, 6);
    }
}
