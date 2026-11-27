//! Tests of the Q4_0 prompt matmul op: its picks and tiling on the CPU, the reference product,
//! and (ignored, on a GPU of compute capability 12.x) its bits against ggml's `mul_mat` and its
//! time.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::noise;

/// Gemma 4 31B's matmul weights, `[k, rows]`: FFN gate / up, FFN down, the local attention's
/// q and out, k / v, the global attention's q and out, its k.
const GEMMA: [(i64, i64); 8] = [
    (5376, 21504),
    (21504, 5376),
    (5376, 8192),
    (8192, 5376),
    (5376, 4096),
    (5376, 16384),
    (16384, 5376),
    (5376, 2048),
];

/// The shapes and step sizes of the GPU check.
const CHECKED: [(i64, i64); 5] = [
    (5376, 21504),
    (21504, 5376),
    (5376, 8192),
    (8192, 5376),
    (5376, 4096),
];
const CHECKED_N: [i64; 4] = [512, 1024, 4095, 4096];

fn shape(ty: ffi::ggml_type, ne: [i64; 4]) -> (ffi::ggml_type, [i64; 4], [usize; 4]) {
    let (ts, row) = if ty == types::Q4_0 {
        (BLOCK_BYTES, (ne[0] / QK) as usize * BLOCK_BYTES)
    } else {
        (4, ne[0] as usize * 4)
    };
    let nb2 = row * ne[1] as usize;
    (ty, ne, [ts, row, nb2, nb2 * ne[2] as usize])
}

/// A Q4_0 matrix of `rows` rows of `k` values: random nibbles, scales in [0.002, 0.012).
fn q4_0_matrix(k: usize, rows: usize, seed: u64) -> Vec<u8> {
    let blocks = k * rows / QK as usize;
    let qs = noise(blocks * 16, seed);
    let ds = noise(blocks, seed + 1);
    let mut out = Vec::with_capacity(blocks * BLOCK_BYTES);
    for (b, d) in ds.iter().enumerate() {
        // SAFETY: a plain conversion.
        let h = unsafe { ffi::ggml_fp32_to_fp16(0.007 + 0.005 * d) };
        out.extend_from_slice(&h.to_le_bytes());
        for q in &qs[b * 16..(b + 1) * 16] {
            out.push(((q + 1.0) * 128.0).min(255.0) as u8);
        }
    }
    out
}

/// Inputs over several magnitudes, an all-zero column and one of tiny values.
fn input(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut x: Vec<f32> = noise(k * n, seed)
        .iter()
        .enumerate()
        .map(|(i, v)| v * 4f32.powi((i / k % 5) as i32 - 2))
        .collect();
    if n > 3 {
        x[k..2 * k].fill(0.0);
        x[3 * k..4 * k].iter_mut().for_each(|v| *v *= 1e-20);
    }
    x
}

#[test]
fn fits_q4_0_weights_from_512_columns() {
    for (k, rows) in GEMMA {
        let w = shape(types::Q4_0, [k, rows, 1, 1]);
        for n in [512, 1024, 4095, 4096, 8192] {
            assert!(
                fits(w, shape(types::F32, [k, n, 1, 1])),
                "{k}x{rows}, {n} columns"
            );
        }
        assert!(!fits(w, shape(types::F32, [k, 511, 1, 1])));
    }
    let w = shape(types::Q4_0, [5376, 21504, 1, 1]);
    let x = shape(types::F32, [5376, 4096, 1, 1]);
    // Other weight types, k not a multiple of 256, rows not a multiple of 128, a mismatched k,
    // 3D operands, too much quantized input.
    assert!(!fits(shape(types::Q8_0, [5376, 21504, 1, 1]), x));
    assert!(!fits(shape(types::Q4_K, [5376, 21504, 1, 1]), x));
    assert!(!fits(
        shape(types::Q4_0, [5248, 21504, 1, 1]),
        shape(types::F32, [5248, 4096, 1, 1])
    ));
    assert!(!fits(shape(types::Q4_0, [5376, 21500, 1, 1]), x));
    assert!(!fits(w, shape(types::F32, [8192, 4096, 1, 1])));
    assert!(!fits(w, shape(types::F32, [5376, 4096, 2, 1])));
    assert!(!fits(shape(types::Q4_0, [5376, 21504, 2, 1]), x));
    assert!(!fits(
        shape(types::Q4_0, [21504, 5376, 1, 1]),
        shape(types::F32, [21504, 16384, 1, 1])
    ));
    // Unaligned weight rows: 256 values are 144 bytes, so only padded strides break it.
    let (ty, ne, mut nb) = w;
    nb[1] += 4;
    assert!(!fits((ty, ne, nb), x));
}

/// Where ggml-cuda runs each tile on one CTA on a GB10 (48 SMs) at Gemma's shapes: every
/// shape from 4095 columns, the FFN gate / up and the global q at every size from 512, the
/// narrower weights only once their tiles fill 90% of the waves.
#[test]
fn ggml_tiles_on_a_gb10_at_gemma_shapes() {
    let tiles = |k, rows, n| ggml_tiles(rows, k, n, 48);
    for (k, rows) in GEMMA {
        for n in [4095, 4096, 8192] {
            assert!(tiles(k, rows, n), "{k}x{rows}, {n} columns");
        }
    }
    let want = [
        // [k, rows]: 512, 1024, 2048 columns.
        ((5376, 21504), [true, true, true]),
        ((21504, 5376), [false, true, true]),
        ((5376, 8192), [false, true, true]),
        ((8192, 5376), [false, true, true]),
        ((5376, 4096), [false, false, true]),
        ((5376, 16384), [true, true, true]),
        ((16384, 5376), [false, true, true]),
        ((5376, 2048), [false, false, false]),
    ];
    for ((k, rows), w) in want {
        for (n, w) in [512, 1024, 2048].into_iter().zip(w) {
            assert_eq!(tiles(k, rows, n), w, "{k}x{rows}, {n} columns");
        }
    }
    // 512 columns of a 5376-row weight: 168 tiles on 4 waves of 48 (87%), split.
    assert!(crate::model::q8_mmq::plan(5376, 21504, 512, 48).split());
    assert!(!ggml_tiles(21504, 5376, 4096, 0));
}

#[test]
fn off_without_a_gpu() {
    // The engine sets the GPU on load; the tests never do (and run without CUDA here).
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q4_0, &[5376, 21504]);
    let x = ctx.new_tensor(types::F32, &[5376, 4096]);
    assert!(!takes(w, x));
}

#[test]
fn grid_and_quant_bytes_pad_columns_to_tiles() {
    assert_eq!(grid(21504, 4096), (32, 168));
    assert_eq!(grid(5376, 4095), (32, 42));
    assert_eq!(grid(5376, 4097), (33, 42));
    assert_eq!(n_padded(4095), 4096);
    assert_eq!(n_padded(513), 640);
    // 672 blocks x 4096 columns x (32 quants + an F32 scale).
    assert_eq!(quant_bytes(21504, 4096), 672 * 4096 * 36);
    assert!(quant_bytes(21504, 4096) <= MAX_QUANT_BYTES);
    assert_eq!(SMEM_BYTES, 86016);
}

/// The raster hands every tile to exactly one CTA, for grids whose row tiles are and are not a
/// multiple of the group.
#[test]
fn tile_of_covers_every_tile_once() {
    for (ntx, nty) in [
        (32, 168),
        (32, 42),
        (4, 64),
        (33, 42),
        (1, 1),
        (5, 3),
        (7, 17),
    ] {
        let mut seen = vec![false; (ntx * nty) as usize];
        for b in 0..ntx * nty {
            let (it, jt) = tile_of(b, ntx, nty);
            assert!(
                (0..nty).contains(&it) && (0..ntx).contains(&jt),
                "{ntx}x{nty}: {b}"
            );
            let at = (it * ntx + jt) as usize;
            assert!(!seen[at], "{ntx}x{nty}: tile ({it}, {jt}) twice");
            seen[at] = true;
        }
        assert!(seen.iter().all(|&s| s), "{ntx}x{nty}");
    }
}

/// A wave of 48 CTAs on the FFN down's grid (32 column tiles x 42 row tiles) reads 8 row tiles
/// of the weight and 6 column tiles of the input, not 48 column tiles of one or two row tiles.
#[test]
fn tile_of_groups_row_tiles() {
    let wave = |w: i64| {
        let mut rows = std::collections::BTreeSet::new();
        let mut cols = std::collections::BTreeSet::new();
        for b in w * 48..(w + 1) * 48 {
            let (it, jt) = tile_of(b, 32, 42);
            rows.insert(it);
            cols.insert(jt);
        }
        (rows.len(), cols.len())
    };
    assert_eq!(wave(0), (8, 6));
    assert_eq!(tile_of(0, 32, 42), (0, 0));
    assert_eq!(tile_of(1, 32, 42), (1, 0));
    assert_eq!(tile_of(8, 32, 42), (0, 1));
    // The last group holds 42 - 40 = 2 row tiles.
    assert_eq!(tile_of(5 * 8 * 32, 32, 42), (40, 0));
    assert_eq!(tile_of(5 * 8 * 32 + 1, 32, 42), (41, 0));
    assert_eq!(tile_of(5 * 8 * 32 + 2, 32, 42), (40, 1));
}

#[test]
fn quantize_col_is_ggml_mmq_q8_1_ds4() {
    let x = noise(64, 3);
    let (q, d) = quantize_col(&x);
    for b in 0..2 {
        let v = &x[b * 32..b * 32 + 32];
        let amax = v.iter().fold(0f32, |m, v| m.max(v.abs()));
        let i = v.iter().position(|v| v.abs() == amax).unwrap();
        assert_eq!(q[b * 32 + i].unsigned_abs(), 127);
        // The scale as the F16 ggml stores it.
        // SAFETY: plain conversions.
        let h = unsafe { ffi::ggml_fp16_to_fp32(ffi::ggml_fp32_to_fp16(1.0 / (127.0 / amax))) };
        assert_eq!(d[b].to_bits(), h.to_bits());
    }
    let (q, d) = quantize_col(&[0.0; 32]);
    assert_eq!((q, d), (vec![0; 32], vec![0.0]));
}

/// `row_dot` sums ggml's way: the blocks in K order, `fma(fl(C * dA), dB, sum)`.
#[test]
fn row_dot_is_ggml_chain() {
    let row = q4_0_matrix(64, 1, 5);
    let (q, d) = quantize_col(&noise(64, 6));
    let mut want = 0f32;
    for b in 0..2 {
        let blk = &row[b * BLOCK_BYTES..];
        // SAFETY: a plain conversion.
        let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
        let mut c = 0i32;
        for v in 0..32 {
            let byte = blk[2 + v % 16];
            let nib = if v < 16 { byte & 0xF } else { byte >> 4 };
            c += (nib as i32 - 8) * q[b * 32 + v] as i32;
        }
        want = (c as f32 * dw).mul_add(d[b], want);
    }
    assert_eq!(row_dot(&row, 2, &q, &d).to_bits(), want.to_bits());
}

/// `mul_mat` on the CPU against the product of the dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    let (k, rows, n) = (512usize, 128usize, 520usize);
    let wb = q4_0_matrix(k, rows, 7);
    let x = input(k, n, 8);
    let ctx = Context::new(16, 16);
    let w = ctx
        .new_tensor(types::Q4_0, &[k as i64, rows as i64])
        .as_input();
    let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let y = mul_mat(&ctx, w, xt).as_output();
    assert_eq!(y.ne(), [rows as i64, n as i64, 1, 1]);
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&wb);
    xt.set_f32(&x);
    backend.compute(&graph).unwrap();
    let got = y.to_f32();
    let mut worst = 0f64;
    for j in 0..n {
        let (qx, dx) = quantize_col(&x[j * k..(j + 1) * k]);
        for r in 0..rows {
            let (mut want, mut mag) = (0f64, 0f64);
            for b in 0..k / 32 {
                let blk = &wb[(r * k / 32 + b) * BLOCK_BYTES..];
                // SAFETY: a plain conversion.
                let dw = unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                for v in 0..32 {
                    let byte = blk[2 + v % 16];
                    let nib = if v < 16 { byte & 0xF } else { byte >> 4 };
                    let t = dw as f64 * (nib as f64 - 8.0) * dx[b] as f64 * qx[b * 32 + v] as f64;
                    want += t;
                    mag += t.abs();
                }
            }
            let g = got[j * rows + r] as f64;
            worst = worst.max((g - want).abs() / mag.max(1e-30));
        }
    }
    assert!(worst < 1e-5, "worst relative error {worst}");
    // The all-zero column gives zeros.
    assert!(got[rows..2 * rows].iter().all(|&v| v == 0.0));
}

/// The quantize launches of a graph's nodes: a node whose last op node before it read the same
/// input reuses its blocks; another input, or another of the op's nodes between, quantizes.
#[test]
fn quant_run_reuses_the_blocks_of_the_same_input() {
    let ctx = Context::new(32, 32);
    let wa = ctx.new_tensor(types::Q4_0, &[512, 256]).as_input();
    let wb = ctx.new_tensor(types::Q4_0, &[512, 128]).as_input();
    let x1 = ctx.new_tensor(types::F32, &[512, 512]).as_input();
    let x2 = ctx.new_tensor(types::F32, &[512, 512]).as_input();
    let a = mul_mat(&ctx, wa, x1);
    let b = mul_mat(&ctx, wb, x1);
    let other = ctx.add(x1, x1);
    let c = mul_mat(&ctx, wa, x1);
    let d = mul_mat(&ctx, wa, x2);
    let e = mul_mat(&ctx, wb, x1);
    let mut graph = ctx.new_graph(32);
    for t in [a, b, other, c, d, e] {
        graph.expand(t);
    }
    // SAFETY: the graph's node array holds its nodes, valid tensors.
    let nodes =
        unsafe { std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph.raw()), graph.nodes()) };
    let at = |t: Tensor<'_>| nodes.iter().position(|&n| n == t.raw()).unwrap();
    // SAFETY: nodes of the graph; `at` picks the op's nodes.
    let run = |t: Tensor<'_>| unsafe { quant_run(nodes, at(t)) };
    assert!(run(a));
    assert!(!run(b));
    // Other ops between leave the blocks.
    assert!(!run(c));
    assert!(run(d));
    assert!(run(e));
}

/// A GPU of compute capability 12.x (the op's), with its SM count, else `None` (the check is
/// skipped: ggml's plans and the op's gate are the GB10's).
#[cfg(feature = "cuda")]
fn gb10() -> Option<(crate::ggml::Backend, i64)> {
    use crate::ggml::{DeviceKind, device_by_kind};
    let sms = crate::hw::per_device::sms_if_cc12(0) as i64;
    if sms == 0 {
        println!("no compute capability 12.x GPU: skipped");
        return None;
    }
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    Some((gpu, sms))
}

/// `w @ x` of a `[k, rows]` Q4_0 weight and `n` columns on the op (`ours`) or ggml's
/// `mul_mat`: the output's bytes.
#[cfg(feature = "cuda")]
fn case(backend: &mut Backend, ours: bool, k: i64, rows: i64, n: i64) -> Vec<u8> {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let y = if ours {
        mul_mat(&ctx, w, x)
    } else {
        ctx.mul_mat(w, x)
    }
    .as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(y);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&q4_0_matrix(k as usize, rows as usize, (k + rows) as u64));
    x.set_f32(&input(k as usize, n as usize, n as u64));
    backend.compute(&graph).unwrap();
    y.bytes()
}

/// On a GB10 the op gives ggml-cuda's MMQ product bit for bit at Gemma 4 31B's shapes and 512
/// to 4096 columns, wherever ggml runs a CTA per tile (the others are ggml's anyway).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_is_ggml_bit_for_bit() {
    let Some((mut gpu, sms)) = gb10() else { return };
    let mut checked = 0;
    for (k, rows) in CHECKED {
        for n in CHECKED_N {
            if !ggml_tiles(rows, k, n, sms) {
                println!("{k}x{rows}, {n} columns: ggml splits the tiles, not taken");
                continue;
            }
            let ours = case(&mut gpu, true, k, rows, n);
            let ggml = case(&mut gpu, false, k, rows, n);
            let diff = ours
                .chunks(4)
                .zip(ggml.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            println!("{k}x{rows}, {n} columns: {diff} values differ");
            assert_eq!(diff, 0, "{k}x{rows}, {n} columns: {diff} values moved");
            checked += 1;
        }
    }
    // 15 on a GB10's 48 SMs.
    assert!(checked > 0, "no case taken");
}

/// The time per call of ggml's MMQ and the op at Gemma 4 31B's shapes and 4096 columns: 8
/// calls a graph on two inputs in turn (so every call quantizes), the second of two runs. The
/// weight is the same for all 8 calls, so it may stay in L2 a little more than in a model.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_time_at_4096_rows() {
    const CALLS: usize = 8;
    let Some((mut gpu, _)) = gb10() else { return };
    let n = 4096i64;
    for (k, rows) in CHECKED {
        let mut ms = [0f64; 2];
        for (i, ours) in [false, true].into_iter().enumerate() {
            let ctx = Context::new(64, 64);
            let w = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
            // Outputs too, so the second run reads them as set.
            let xs = [0, 1].map(|_| ctx.new_tensor(types::F32, &[k, n]).as_input().as_output());
            let mut graph = ctx.new_graph(64);
            let mut last = None;
            for c in 0..CALLS {
                let x = xs[c % 2];
                let y = if ours {
                    mul_mat(&ctx, w, x)
                } else {
                    ctx.mul_mat(w, x)
                };
                graph.expand(y);
                last = Some(y);
            }
            last.unwrap().as_output();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            w.set_bytes(&q4_0_matrix(k as usize, rows as usize, 11));
            for (s, x) in xs.iter().enumerate() {
                x.set_f32(&input(k as usize, n as usize, 12 + s as u64));
            }
            gpu.compute(&graph).unwrap();
            let t0 = std::time::Instant::now();
            gpu.compute(&graph).unwrap();
            ms[i] = t0.elapsed().as_secs_f64() * 1e3 / CALLS as f64;
        }
        let tflops = |ms: f64| 2.0 * (k * rows * n) as f64 / ms / 1e9;
        println!(
            "{k}x{rows}, {n} columns: ggml {:.2} ms ({:.0} TFLOPS), q40_mm {:.2} ms ({:.0} TFLOPS), {:.2}x",
            ms[0],
            tflops(ms[0]),
            ms[1],
            tflops(ms[1]),
            ms[0] / ms[1]
        );
    }
}
