//! `tu_mmq`'s plans and shapes on the CPU; on a Turing GPU without tensor cores (ignored tests)
//! the op against ggml's matmul bit for bit.

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

#[test]
fn ggml_tile_widths() {
    assert_eq!(ggml_tile_width(128), 128);
    assert_eq!(ggml_tile_width(512), 128);
    assert_eq!(ggml_tile_width(1024), 128);
    // 200 columns: two tiles of 112 (the first width with two tiles).
    assert_eq!(ggml_tile_width(200), 112);
    // 129 columns: two tiles of 80 (the first width with two).
    assert_eq!(ggml_tile_width(129), 80);
}

#[test]
fn ggml_unsplit_plans_on_24_sms() {
    // 9216 rows: 72 row tiles, 4 column tiles at 512 columns, 288 tiles in 12 full waves.
    assert!(ggml_unsplit(9216, 512, 24));
    assert!(ggml_unsplit(9216, 1024, 24));
    // 2560 rows at 512 columns: 80 tiles in 4 waves of 24 (83%), stream-k splits them.
    assert!(!ggml_unsplit(2560, 512, 24));
    // At 1024: 160 tiles in 7 waves (95%).
    assert!(ggml_unsplit(2560, 1024, 24));
    // 2048 rows at 512 columns: 64 tiles in 3 waves (88%).
    assert!(!ggml_unsplit(2048, 512, 24));
    assert!(ggml_unsplit(6144, 512, 24));
    assert!(!ggml_unsplit(6144, 512, 0));
}

#[test]
fn tile_width_fills_the_waves() {
    // 288 CTAs of 128 columns in 12 waves; 576 of 64 in 24: the same columns, 128 wins.
    assert_eq!(tile_width(9216, 512, 24), 128);
    // 200 columns: two tiles of 128 (256 columns) against four of 64 (256): a tie.
    assert_eq!(tile_width(9216, 200, 24), 128);
    // 192 columns on 2560 rows: 40 CTAs of 128 in 2 waves (256 columns), 60 of 64 in 3 (192).
    assert_eq!(tile_width(2560, 192, 24), 64);
}

#[test]
fn plans_mirror_ggml_grids() {
    // Unsplit: a CTA per tile of the op's width.
    assert_eq!(plan(9216, 512, 24), Some((128, false)));
    assert_eq!(plan(9216, 128, 24), Some((128, false)));
    // ggml splits these over its 24 SMs on 128-column tiles: ggml's stream-k grid.
    assert_eq!(plan(2560, 512, 24), Some((128, true)));
    assert_eq!(plan(1024, 1024, 24), Some((128, true)));
    assert_eq!(plan(2560, 256, 24), Some((128, true)));
    assert_eq!(plan(1024, 128, 24), Some((128, true)));
    // ggml's tiles 112 wide (200 columns), split: not mirrored.
    assert!(!ggml_unsplit(2560, 200, 24));
    assert_eq!(plan(2560, 200, 24), None);
    // More SMs than the fixup scratch holds; no GPU.
    assert_eq!(plan(2560, 512, MAX_STREAM_K_SMS + 1), None);
    assert_eq!(plan(2560, 512, 0), None);
}

#[test]
fn quant_bytes_pad_to_tiles() {
    assert_eq!(quant_bytes(2560, 512, 128), 20 * 512 * 144);
    assert_eq!(quant_bytes(2560, 500, 128), 20 * 512 * 144);
    assert_eq!(quant_bytes(256, 65, 64), 2 * 128 * 144);
}

fn shape(ty: ffi::ggml_type, k: i64, rows: i64) -> (ffi::ggml_type, [i64; 4], [usize; 4]) {
    let (bytes, qk) = block(ty).unwrap_or((4, 1));
    let row = (k / qk) as usize * bytes;
    (
        ty,
        [k, rows, 1, 1],
        [bytes, row, row * rows as usize, row * rows as usize],
    )
}

fn input(k: i64, n: i64) -> (ffi::ggml_type, [i64; 4], [usize; 4]) {
    let col = k as usize * 4;
    (
        types::F32,
        [k, n, 1, 1],
        [4, col, col * n as usize, col * n as usize],
    )
}

#[test]
fn fits_takes_the_mirrored_quant_shapes_only() {
    for ty in [types::Q4_K, types::Q5_K, types::Q6_K, types::Q8_0] {
        assert!(fits(shape(ty, 2560, 9216), input(2560, 512), 24));
        assert!(fits(shape(ty, 256, 9216), input(256, MAX_TOKENS), 24));
        // a decode step, a short chunk, a long one
        assert!(!fits(shape(ty, 2560, 9216), input(2560, 1), 24));
        assert!(!fits(
            shape(ty, 2560, 9216),
            input(2560, MIN_TOKENS - 1),
            24
        ));
        assert!(!fits(
            shape(ty, 2560, 9216),
            input(2560, MAX_TOKENS + 1),
            24
        ));
        // plans ggml splits: on 128-column tiles (stream-k), on 112-column ones
        assert!(fits(shape(ty, 9216, 2560), input(9216, 512), 24));
        assert!(fits(shape(ty, 2560, 1024), input(2560, 1024), 24));
        assert!(!fits(shape(ty, 9216, 2560), input(9216, 200), 24));
        // rows not whole tiles, K not whole iterations
        assert!(!fits(shape(ty, 2560, 9200), input(2560, 512), 24));
        assert!(!fits(shape(ty, 2336, 9216), input(2336, 512), 24));
        // a mismatched input, an F16 input
        assert!(!fits(shape(ty, 2560, 9216), input(2048, 512), 24));
        let (_, ne, nb) = input(2560, 512);
        assert!(!fits(shape(ty, 2560, 9216), (types::F16, ne, nb), 24));
        // no GPU
        assert!(!fits(shape(ty, 2560, 9216), input(2560, 512), 0));
    }
    // other weight types
    for ty in [types::F16, types::F32, types::Q4_0, types::IQ4_XS] {
        assert!(!fits(shape(ty, 2560, 9216), input(2560, 512), 24));
    }
    // More quantized input than the scratch holds: 2560 values of 8192 columns (22.5 MiB).
    assert!(!fits(
        shape(types::Q4_K, 2560, 9216),
        input(2560, MAX_TOKENS),
        24
    ));
}

#[test]
fn off_by_switch() {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::Q4_K, &[2560, 9216]);
    let x = ctx.new_tensor(types::F32, &[2560, 512]);
    set_tu_mmq(false);
    assert!(!takes(w, x));
    set_tu_mmq(true);
    // Without a TU11x GPU set the op is never taken.
    assert!(!takes(w, x));
}

/// The op (planned for `sms` SMs) and ggml's matmul of a `ty` weight `[k, rows]` and `n`
/// columns on `backend`.
fn case(
    backend: &mut Backend,
    ty: ffi::ggml_type,
    k: i64,
    rows: i64,
    n: i64,
    sms: i64,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(16, 64);
    let w = ctx.new_tensor(ty, &[k, rows]).as_input();
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ours = mul_mat_sms(&ctx, w, x, sms).as_output();
    let theirs = ctx.mul_mat(w, x).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(ours);
    graph.expand(theirs);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // Rows of different magnitudes, an input with outliers (as activations have).
    let wv: Vec<f32> = noise((k * rows) as usize, 21 + ty as u64)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.02 * (1.0 + (i / k as usize % 7) as f32))
        .collect();
    let mut xv = noise((k * n) as usize, 22);
    for i in (0..xv.len()).step_by(97) {
        xv[i] *= 30.0;
    }
    w.set_bytes(&crate::ggml::quantize(ty, &wv, k as usize));
    x.set_f32(&xv);
    backend.compute(&graph).unwrap();
    (ours.to_f32(), theirs.to_f32())
}

#[test]
fn cpu_reference_is_the_dequantized_product() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for ty in [types::Q4_K, types::Q6_K, types::Q8_0] {
        let (ours, theirs) = case(&mut backend, ty, 256, 128, 128, 1);
        // ggml's CPU matmul quantizes the input: close, not equal.
        let scale = theirs.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, b) in ours.iter().zip(&theirs) {
            assert!((a - b).abs() <= 0.05 * scale, "{ty}: {a} vs {b}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a Turing GPU without tensor cores"]
fn cuda_tu_mmq_is_ggml_bit_for_bit() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .expect("no GPU device")
        .init()
        .unwrap();
    if !crate::hw::per_device::turing_without_tensor_cores(0) {
        eprintln!("not a Turing GPU without tensor cores: skipped");
        return;
    }
    let sms = crate::hw::per_device::sm_count(0) as i64;
    // Qwen3.5-4B and Gemma-4-E2B prompt shapes, then odd column counts; the rest ggml splits
    // over its SMs (stream-k): chunk tails, the 1024-row K/V projections, a long K.
    let shapes = [
        (2560, 9216, 512),
        (2560, 9216, 1024),
        (2560, 2560, 1024),
        (1536, 6144, 512),
        (1536, 6144, 300),
        (2560, 9216, 1000),
        (2560, 9216, 4096),
        (2560, 1024, 1024),
        (2560, 1024, 128),
        (2560, 2560, 256),
        (2560, 9216, 128),
        (9216, 2560, 256),
        (9216, 2560, 1024),
        (2560, 4096, 256),
        (1536, 512, 384),
        // A partial column tile; also the fourth stream-k shape on a 14-SM GTX 1650.
        (2560, 1024, 255),
    ];
    let mut checked = 0;
    let mut split = 0;
    for ty in [types::Q4_K, types::Q5_K, types::Q6_K, types::Q8_0] {
        for (k, rows, n) in shapes {
            if !fits(shape(ty, k, rows), input(k, n), sms) {
                eprintln!("{ty} {k}x{rows} n={n}: ggml splits it on {sms} SMs, skipped");
                continue;
            }
            split += usize::from(plan(rows, n, sms).is_some_and(|p| p.1));
            let (ours, theirs) = case(&mut gpu, ty, k, rows, n, sms);
            let diff = ours
                .iter()
                .zip(&theirs)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(diff, 0, "{ty} {k}x{rows} n={n}: {diff} outputs differ");
            checked += 1;
        }
    }
    assert!(checked >= 32, "only {checked} cases ran");
    assert!(split >= 16, "only {split} stream-k cases ran");
}
