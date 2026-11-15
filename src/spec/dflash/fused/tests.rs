//! The draft pass's fused ops: the top-k's parts, its CPU op against a sort and ggml's top-k,
//! the convolution's CPU op against ggml's ops bit for bit, and on CUDA (ignored tests) the
//! kernels against ggml-cuda's ops.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Deterministic values in [-1, 1).
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

/// Rows of `n` distinct values (a shuffle of a ramp, with negatives and a signed zero).
fn distinct_rows(n: usize, rows: usize, seed: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|r| {
            (0..n).map(move |i| {
                let j = (i * 7919 + r * 104_729 + seed) % n;
                (j as f32 - (n / 2) as f32) * 0.25
            })
        })
        .collect()
}

/// The best `k` of a row by value, then lower index: a full sort.
fn sorted_top(row: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| topk_key(row[b], b).cmp(&topk_key(row[a], a)));
    idx[..k].iter().map(|&i| i as i32).collect()
}

#[test]
fn topk_parts_cover_rows() {
    for n in [16384i64, 16388, 65536, 262_144, 262_148, 1 << 20, 3 << 20] {
        for k in [1i64, 7, 16, 32] {
            let (parts, span) = topk_split(n, k);
            assert!(span % 4 == 0, "{n} {k}");
            assert!(parts * span >= n && (parts - 1) * span < n, "{n} {k}");
            assert!(parts * k <= TOPK_JOIN, "{n} {k}");
        }
    }
    assert_eq!(topk_split(262_144, 16), (16, 16384));
}

#[test]
fn topk_fits_shapes() {
    let nb = |n: i64| [4usize, 4 * n as usize, 0, 0];
    assert!(topk_fits(types::F32, [262_144, 112, 1, 1], nb(262_144), 16));
    assert!(!topk_fits(
        types::F32,
        [262_144, 112, 1, 1],
        nb(262_144),
        33
    ));
    assert!(!topk_fits(types::F32, [8192, 4, 1, 1], nb(8192), 16));
    assert!(!topk_fits(types::F32, [262_146, 4, 1, 1], nb(262_146), 16));
    assert!(!topk_fits(
        types::F16,
        [262_144, 4, 1, 1],
        [2, 524_288, 0, 0],
        16
    ));
    assert!(!topk_fits(
        types::F32,
        [262_144, 4096, 1, 1],
        nb(262_144),
        16
    ));
}

#[test]
fn topk_key_orders_as_floats() {
    let v = [
        f32::NEG_INFINITY,
        -f32::MAX,
        -1.5,
        -0.0,
        0.0,
        1e-30,
        2.0,
        f32::MAX,
        f32::INFINITY,
    ];
    for w in v.windows(2) {
        assert!(topk_key(w[0], 0) < topk_key(w[1], 0), "{w:?}");
    }
    assert!(topk_key(1.0, 3) > topk_key(1.0, 4));
    assert!(topk_key(f32::NEG_INFINITY, i32::MAX as usize) > 0);
}

fn topk_case(backend: &mut Backend, values: &[f32], n: usize, k: usize) -> (Vec<i32>, Vec<i32>) {
    let rows = values.len() / n;
    let ctx = Context::new(8, 8);
    let x = ctx
        .new_tensor(types::F32, &[n as i64, rows as i64])
        .as_input();
    let own = top_k_op(&ctx, x, k as i64).as_output();
    let ggml = ctx.top_k(x, k as i32).as_output();
    let mut g = ctx.new_graph(8);
    g.expand(own);
    g.expand(ggml);
    ctx.alloc_graph(&g, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(values);
    backend.compute(&g).unwrap();
    (own.to_i32(), ggml.to_i32())
}

/// The CPU op is the sort's top-k, best first, ties to the lower index; on distinct values its
/// set is ggml's.
#[test]
fn cpu_topk_matches_sort_and_ggml() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let (n, rows) = (20_000usize, 5usize);
    for k in [1usize, 16, 32] {
        let values = distinct_rows(n, rows, k);
        let (own, ggml) = topk_case(&mut backend, &values, n, k);
        for r in 0..rows {
            let row = &values[r * n..(r + 1) * n];
            assert_eq!(
                own[r * k..(r + 1) * k],
                sorted_top(row, k),
                "row {r}, k {k}"
            );
            let mut a = own[r * k..(r + 1) * k].to_vec();
            let mut b = ggml[r * k..(r + 1) * k].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {r}, k {k}");
        }
    }
    // Ties: values on a few levels, so the k-th place is shared; the lower indexes win.
    let values: Vec<f32> = noise(n * rows, 7)
        .iter()
        .map(|v| (v * 3.0).floor())
        .collect();
    let (own, _) = topk_case(&mut backend, &values, n, 16);
    for r in 0..rows {
        assert_eq!(
            own[r * 16..(r + 1) * 16],
            sorted_top(&values[r * n..(r + 1) * n], 16)
        );
    }
}

fn v2(kernel: usize, group: usize) -> Dflash2 {
    Dflash2 {
        conv_kernel: kernel,
        conv_group: group,
        selector_rank: 4,
        selector_top_k: 4,
    }
}

/// `(fused, ggml)` of one side of the convolution of `x` (`[e, n]`) on `backend`.
fn conv_case(
    backend: &mut Backend,
    v2: &Dflash2,
    e: usize,
    n: usize,
    block: i64,
    side: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let k = v2.conv_kernel;
    let groups = e / v2.conv_group;
    let ctx = Context::new(256, 256);
    let x = ctx.new_tensor(types::F32, &[e as i64, n as i64]).as_input();
    let dynamic = ctx
        .new_tensor(types::F32, &[(2 * k * groups) as i64, n as i64])
        .as_input();
    let base = ctx
        .new_tensor(types::F32, &[e as i64, k as i64, 2])
        .as_input();
    let own = conv(&ctx, x, dynamic, base, side, v2, block).as_output();
    let ggml = super::super::v2::conv(&ctx, x, dynamic, base, side, v2, block).as_output();
    let mut g = ctx.new_graph(256);
    g.expand(own);
    g.expand(ggml);
    ctx.alloc_graph(&g, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise(e * n, seed));
    dynamic.set_f32(&noise(2 * k * groups * n, seed + 1));
    base.set_f32(&noise(e * k * 2, seed + 2));
    backend.compute(&g).unwrap();
    (own.to_f32(), ggml.to_f32())
}

/// The CPU op gives ggml's ops' bits: one block and several, both sides, more taps than the
/// block has positions.
#[test]
fn cpu_conv_matches_ggml_bits() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    // (kernel, group, e, n, block)
    let cases = [
        (2usize, 4usize, 32usize, 8usize, 8i64),
        (2, 4, 32, 24, 8),
        (2, 16, 64, 16, 4),
        (3, 8, 32, 12, 2),
        (2, 8, 32, 5, 1),
    ];
    for (i, &(k, group, e, n, block)) in cases.iter().enumerate() {
        let v2 = v2(k, group);
        for side in 0..2 {
            let (own, ggml) = conv_case(&mut backend, &v2, e, n, block, side, 10 * i as u64);
            let a: Vec<u32> = own.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = ggml.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "case {i}, side {side}");
        }
    }
}

#[cfg(feature = "cuda")]
fn gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap()
}

/// On a GPU, the kernels: the top-k is the sort's (Gemma 4's vocabulary, 1 to 112 rows) and on
/// distinct values ggml-cuda's set; the convolution ggml-cuda's bits (Gemma 4's drafter shape,
/// one and 16 sequences).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_ggml() {
    let mut gpu = gpu();
    let n = 262_144usize;
    for (rows, k) in [(1usize, 16usize), (7, 16), (112, 16), (9, 32), (3, 1)] {
        let values = distinct_rows(n, rows, rows + k);
        let (own, ggml) = topk_case(&mut gpu, &values, n, k);
        for r in 0..rows {
            let row = &values[r * n..(r + 1) * n];
            assert_eq!(
                own[r * k..(r + 1) * k],
                sorted_top(row, k),
                "row {r} of {rows}"
            );
            let mut a = own[r * k..(r + 1) * k].to_vec();
            let mut b = ggml[r * k..(r + 1) * k].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {r} of {rows}");
        }
        let values: Vec<f32> = noise(n * rows, rows as u64)
            .iter()
            .map(|v| (v * 1000.0).floor())
            .collect();
        let (own, _) = topk_case(&mut gpu, &values, n, k);
        for r in 0..rows {
            let row = &values[r * n..(r + 1) * n];
            assert_eq!(own[r * k..(r + 1) * k], sorted_top(row, k), "ties, row {r}");
        }
    }
    let v2 = v2(2, 256);
    for (n, block) in [(8usize, 8i64), (128, 8), (7, 7)] {
        for side in 0..2 {
            let (own, ggml) = conv_case(&mut gpu, &v2, 5376, n, block, side, n as u64);
            let a: Vec<u32> = own.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = ggml.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "{n} positions, side {side}");
        }
    }
}
