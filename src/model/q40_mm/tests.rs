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

/// The shapes of the timing test: the FFN's and the local attention's.
const CHECKED: [(i64, i64); 5] = [
    (5376, 21504),
    (21504, 5376),
    (5376, 8192),
    (8192, 5376),
    (5376, 4096),
];

/// The step sizes of the GPU check (at [`GEMMA`]'s shapes): whole and partial last column tiles.
const CHECKED_N: [i64; 5] = [512, 1024, 4095, 4096, 4097];

/// The GPU tests switch [`VARIANT`] for the whole process: one at a time.
#[cfg(feature = "cuda")]
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The table's matmul `i` (0 the op's) from here on.
#[cfg(feature = "cuda")]
fn use_variant(i: usize) {
    VARIANT.store(i, Ordering::Relaxed);
}

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
    // Padded strides up to MAX_ROW_BYTES, not past it (the kernel's 32-bit in-tile offsets).
    nb[1] = MAX_ROW_BYTES;
    assert!(fits((ty, ne, nb), x));
    nb[1] = MAX_ROW_BYTES + 16;
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
    assert_eq!(grid(21504, 4096), (32, 336));
    assert_eq!(grid(5376, 4095), (32, 84));
    assert_eq!(grid(5376, 4097), (33, 84));
    assert_eq!(n_padded(4095), 4096);
    assert_eq!(n_padded(513), 640);
    // 672 blocks x 4096 columns x (32 quants + an F32 scale).
    assert_eq!(quant_bytes(21504, 4096), 672 * 4096 * 36);
    assert!(quant_bytes(21504, 4096) <= MAX_QUANT_BYTES);
    // src/cuda/q40_mm.cu's static asserts; two of the op's CTAs (and their 1 KiB each the GPU
    // reserves) fit a GB10 SM's 100 KiB.
    assert_eq!(SMEM_BYTES, 47616);
    assert_eq!(SMEM_BYTES_V1, 86016);
    assert!(2 * (SMEM_BYTES + 1024) <= 100 << 10);
}

/// The raster hands every tile to exactly one CTA, for grids whose tiles are and are not a
/// multiple of the group, in both versions' orders.
#[test]
fn tile_of_covers_every_tile_once() {
    for (group, cols) in [(GROUP, true), (GROUP_V1, false)] {
        for (ntx, nty) in [
            (32, 336),
            (32, 84),
            (4, 64),
            (33, 42),
            (1, 1),
            (5, 3),
            (7, 17),
        ] {
            let mut seen = vec![false; (ntx * nty) as usize];
            for b in 0..ntx * nty {
                let (it, jt) = tile_of(b, ntx, nty, group, cols);
                assert!(
                    (0..nty).contains(&it) && (0..ntx).contains(&jt),
                    "{ntx}x{nty}: {b}"
                );
                let at = (it * ntx + jt) as usize;
                assert!(!seen[at], "{ntx}x{nty}: tile ({it}, {jt}) twice");
                seen[at] = true;
            }
            assert!(seen.iter().all(|&s| s), "{ntx}x{nty}, groups of {group}");
        }
    }
}

/// On the FFN down's 4096-column grid a wave of the op's 96 CTAs (two an SM of a GB10; 32 column
/// tiles x 84 row tiles of 64) reads 24 row tiles of the weight and 4 column tiles of the input,
/// the first version's 48 CTAs (32 x 42 tiles of 128) 8 and 6, not 48 column tiles of one or
/// two row tiles.
#[test]
fn tile_of_groups_tiles() {
    let wave = |w: i64, ctas: i64, (ntx, nty): (i64, i64), group: i64, cols: bool| {
        let mut rows = std::collections::BTreeSet::new();
        let mut cs = std::collections::BTreeSet::new();
        for b in w * ctas..(w + 1) * ctas {
            let (it, jt) = tile_of(b, ntx, nty, group, cols);
            rows.insert(it);
            cs.insert(jt);
        }
        (rows.len(), cs.len())
    };
    assert_eq!(grid(5376, 4096), (32, 84));
    assert_eq!(wave(0, 96, (32, 84), GROUP, true), (24, 4));
    assert_eq!(wave(0, 48, (32, 42), GROUP_V1, false), (8, 6));
    assert_eq!(tile_of(0, 32, 84, GROUP, true), (0, 0));
    assert_eq!(tile_of(1, 32, 84, GROUP, true), (0, 1));
    assert_eq!(tile_of(4, 32, 84, GROUP, true), (1, 0));
    assert_eq!(tile_of(4 * 84, 32, 84, GROUP, true), (0, 4));
    // At 4097 columns the last group holds 33 - 32 = 1 column tile.
    assert_eq!(tile_of(8 * 4 * 84, 33, 84, GROUP, true), (0, 32));
    assert_eq!(tile_of(8 * 4 * 84 + 1, 33, 84, GROUP, true), (1, 32));
    // The first version's: its last group holds 42 - 40 = 2 row tiles.
    assert_eq!(tile_of(1, 32, 42, GROUP_V1, false), (1, 0));
    assert_eq!(tile_of(8, 32, 42, GROUP_V1, false), (0, 1));
    assert_eq!(tile_of(5 * 8 * 32 + 1, 32, 42, GROUP_V1, false), (41, 0));
    assert_eq!(tile_of(5 * 8 * 32 + 2, 32, 42, GROUP_V1, false), (40, 1));
}

/// Where src/cuda/q40_mm.cu's `q40_quant<1>` puts word `w` of tile column `c` among a block's
/// 1024 words of the tile (as the matmul reads them from shared memory too).
fn quant_word(c: usize, w: usize) -> usize {
    let (nt, g) = (c / 8, c % 8);
    ((nt / 2 * 8 + g) * 4 + w / 2) * 4 + nt % 2 * 2 + w % 2
}

/// Where the matmul's copy puts the scale of tile column `c` in a block's scales in shared
/// memory: `q40_quant<1>`'s rows of 32 (row `c % 8 / 2`), 36 words apart.
fn quant_scale(c: usize) -> usize {
    let (nt, g) = (c / 8, c % 8);
    g / 2 * 36 + 2 * nt + g % 2
}

/// The op's quantized-input layout (src/cuda/q40_mm.cu's layout 1, mirrored by
/// [`quant_word`] and [`quant_scale`]): every word and scale of a 128-column tile's block has
/// one place, and lane (g, q)'s 16-byte reads of n8 tile pair `pr` hold, in order, its B
/// fragments (slots q and q + 4, words 2q and 2q + 1) of columns 16 pr + g and 16 pr + 8 + g,
/// and the scales of columns 16 pr + 2q, + 1, 16 pr + 8 + 2q, + 1 (the accumulators' columns).
#[test]
fn quant_layout_is_one_read_per_tile_pair() {
    let mut words = vec![false; 1024];
    let mut scales = vec![false; 128];
    for c in 0..128 {
        for w in 0..8 {
            let at = quant_word(c, w);
            assert!(!words[at], "column {c} word {w}");
            words[at] = true;
        }
        let at = quant_scale(c);
        let row = at / 36 * 32 + at % 36;
        assert!(at % 36 < 32 && !scales[row], "column {c}");
        scales[row] = true;
    }
    assert!(words.iter().chain(&scales).all(|&s| s));
    for pr in 0..8 {
        for (g, q) in (0..8).flat_map(|g| (0..4).map(move |q| (g, q))) {
            let bq = ((pr * 8 + g) * 4 + q) * 4;
            let (c0, c1) = (16 * pr + g, 16 * pr + 8 + g);
            let want = [(c0, 2 * q), (c0, 2 * q + 1), (c1, 2 * q), (c1, 2 * q + 1)];
            for (i, (c, w)) in want.into_iter().enumerate() {
                assert_eq!(quant_word(c, w), bq + i, "pair {pr} ({g}, {q}) word {i}");
            }
            let (bd, c0, c1) = (q * 36 + pr * 4, 16 * pr + 2 * q, 16 * pr + 8 + 2 * q);
            for (i, c) in [c0, c0 + 1, c1, c1 + 1].into_iter().enumerate() {
                assert_eq!(quant_scale(c), bd + i, "pair {pr} ({g}, {q}) scale {i}");
            }
        }
    }
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

/// On a GB10 every matmul of the table (the op's, the first version's, ...) gives ggml-cuda's MMQ product bit for bit
/// at Gemma 4 31B's shapes and 512 to 4097 columns, wherever ggml runs a CTA per tile (the
/// others are ggml's anyway).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_is_ggml_bit_for_bit() {
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, sms)) = gb10() else { return };
    let mut checked = 0;
    let mut skipped = Vec::new();
    // Per matmul of the table: the cases with values that differ, and how many values.
    let mut moved = vec![(0, 0); cuda::kernels()];
    for (k, rows) in GEMMA {
        for n in CHECKED_N {
            if !ggml_tiles(rows, k, n, sms) {
                skipped.push(format!("{k}x{rows} at {n}"));
                continue;
            }
            let ggml = case(&mut gpu, false, k, rows, n);
            for (v, moved) in moved.iter_mut().enumerate() {
                use_variant(v);
                let ours = case(&mut gpu, true, k, rows, n);
                use_variant(0);
                let diff = ours
                    .chunks(4)
                    .zip(ggml.chunks(4))
                    .filter(|(a, b)| a != b)
                    .count();
                if diff > 0 {
                    let name = cuda::kernel_name(v);
                    println!("{k}x{rows}, {n} columns, {name}: {diff} values differ");
                    *moved = (moved.0 + 1, moved.1 + diff);
                }
            }
            checked += 1;
        }
    }
    println!("ggml splits the tiles, not taken: {}", skipped.join(", "));
    for (v, (cases, values)) in moved.iter().enumerate() {
        let name = cuda::kernel_name(v);
        println!("{name}: {checked} cases, {cases} with values that differ ({values} values)");
    }
    // 32 on a GB10's 48 SMs.
    assert!(checked > 0, "no case taken");
    assert!(moved.iter().all(|m| m.0 == 0), "values moved");
}

/// The time per call of ggml's MMQ and of every matmul of the table at Gemma 4 31B's
/// shapes and 4096 columns: 8 calls a graph on two inputs in turn (so every call quantizes),
/// the second of two runs. The weight is the same for all 8 calls, so it may stay in L2 a
/// little more than in a model. Each as a share of the GPU's S8 MMA peak ([`mma_peak`]'s).
///
/// [`mma_peak`]: crate::model::core::mma_peak
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_time_at_4096_rows() {
    use crate::model::core::mma_peak;
    const CALLS: usize = 8;
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, _)) = gb10() else { return };
    let s8 = mma_peak::FORMS
        .iter()
        .position(|f| f.name.contains("S8"))
        .unwrap();
    let peak = match mma_peak::measure(0, 20.0) {
        Ok((_, runs, _)) => mma_peak::best(&runs)
            .iter()
            .find(|r| r.form == s8)
            .map_or(0.0, |r| r.tflops),
        Err(e) => {
            println!("mma_peak: {e}");
            0.0
        }
    };
    println!("S8 MMA peak {peak:.0} TOPS");
    let n = 4096i64;
    // ggml, then the table's matmuls.
    let runs = 1 + cuda::kernels();
    let v1 = (0..cuda::kernels())
        .find(|&v| cuda::kernel_name(v) == "first version")
        .map_or(1, |v| v + 1);
    for (k, rows) in CHECKED {
        let mut ms = vec![0f64; runs];
        for (i, ms) in ms.iter_mut().enumerate() {
            let ours = i > 0;
            use_variant(i.saturating_sub(1));
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
            *ms = t0.elapsed().as_secs_f64() * 1e3 / CALLS as f64;
            use_variant(0);
        }
        let tops = |ms: f64| 2.0 * (k * rows * n) as f64 / ms / 1e9;
        for (i, &t) in ms.iter().enumerate() {
            let name = if i == 0 {
                "ggml".to_string()
            } else {
                cuda::kernel_name(i - 1)
            };
            let share = if peak > 0.0 {
                format!(", {:.0}% of peak", 100.0 * tops(t) / peak)
            } else {
                String::new()
            };
            println!(
                "{k}x{rows}, {n} columns, {name}: {t:.2} ms ({:.0} TOPS{share}), {:.2}x ggml, {:.2}x the first version",
                tops(t),
                ms[0] / t,
                ms[v1] / t
            );
        }
    }
}

/// `x` (`k` x `n` F32 columns) quantized on the host in the op's layout ([`quant_bytes`]).
fn quant_buf(x: &[f32], k: usize, n: usize) -> Vec<u8> {
    let n_pad = n_padded(n as i64) as usize;
    let mut buf = vec![0u8; quant_bytes(k as i64, n as i64)];
    for c in 0..n {
        let (q, d) = quantize_col(&x[c * k..(c + 1) * k]);
        // SAFETY: `buf` holds the blocks and scales of `k` x `n_pad`.
        unsafe { put_col(buf.as_mut_ptr(), k, n_pad, c, &q, &d) };
    }
    buf
}

/// The layout's places: every column's blocks and scales come back as written, and no two
/// columns' bytes meet.
#[test]
fn quant_columns_round_trip() {
    let (k, n) = (512usize, 130usize);
    let x = input(k, n, 21);
    let buf = quant_buf(&x, k, n);
    let n_pad = n_padded(n as i64) as usize;
    for c in 0..n {
        let (q, d) = quantize_col(&x[c * k..(c + 1) * k]);
        let (gq, gd) = get_col(&buf, k, n_pad, c);
        assert_eq!(q, gq, "column {c}");
        assert_eq!(
            d.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            gd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "column {c}"
        );
    }
    let mut seen = vec![false; buf.len()];
    for c in 0..n_pad {
        for kb in 0..k / 32 {
            for w in 0..8 {
                let (at, sd) = quant_at(c, kb, w, n_pad);
                for b in at..at + 4 {
                    assert!(!seen[b], "byte {b}");
                    seen[b] = true;
                }
                if w == 0 {
                    let s = k / 32 * n_pad * 32 + sd * 4;
                    assert!(!seen[s], "scale {s}");
                    seen[s..s + 4].fill(true);
                }
            }
        }
    }
    assert!(seen.iter().all(|&s| s));
}

/// On the CPU, the op over its input quantized by its own node ([`quantize`], [`mul_mat_q`])
/// is the op's product, bit for bit.
#[test]
fn cpu_q40_mm_reads_its_quantized_input() {
    let (k, rows, n) = (512i64, 128i64, 520i64);
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let a = mul_mat(&ctx, w, x).as_output();
    let b = mul_mat_q(&ctx, w, quantize(&ctx, x), n).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(a);
    graph.expand(b);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_bytes(&q4_0_matrix(k as usize, rows as usize, 5));
    x.set_f32(&input(k as usize, n as usize, 6));
    backend.compute(&graph).unwrap();
    assert_eq!(b.ne(), [rows, n, 1, 1]);
    assert!(a.bytes() == b.bytes(), "the products differ");
}

/// On the CPU, [`quantize_norm`] and [`quantize_geglu`] keep (with `y`) values close to ggml's
/// norm and GeGLU, and their blocks are those values' quantization.
#[test]
fn cpu_fused_quantizations_quantize_their_values() {
    let (k, n) = (512usize, 130usize);
    let eps = 1e-6f32;
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let nw = ctx.new_tensor(types::F32, &[k as i64]).as_input();
    let up = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let normed = ctx.mul(ctx.rms_norm(x, eps), nw).as_output();
    let nq = quantize_norm(&ctx, x, nw, eps, true).as_output();
    let act = ctx.geglu_split(x, up).as_output();
    let gq = quantize_geglu(&ctx, x, up, true).as_output();
    let mut graph = ctx.new_graph(32);
    for t in [normed, nq, act, gq] {
        graph.expand(t);
    }
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&input(k, n, 31));
    let wv: Vec<f32> = noise(k, 32).iter().map(|v| 1.0 + 0.5 * v).collect();
    nw.set_f32(&wv);
    up.set_f32(&noise(k * n, 33));
    backend.compute(&graph).unwrap();
    let n_pad = n_padded(n as i64) as usize;
    let qb = quant_bytes(k as i64, n as i64);
    for (want, node) in [(normed, nq), (act, gq)] {
        let want = want.to_f32();
        let bytes = node.bytes();
        let got: Vec<f32> = bytes[qb..]
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(got.len(), k * n);
        for c in 0..n {
            let col = &got[c * k..(c + 1) * k];
            let wc = &want[c * k..(c + 1) * k];
            let mag = wc.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (a, b) in col.iter().zip(wc) {
                assert!(
                    (a - b).abs() <= 2e-3 * mag + 1e-12,
                    "column {c}: {a} vs {b}"
                );
            }
            let (q, d) = quantize_col(col);
            let (gq, gd) = get_col(&bytes, k, n_pad, c);
            assert_eq!(q, gq, "column {c}");
            assert!(d.iter().zip(&gd).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

/// Whether columns `0..n` of two quantizations of `k` values agree, bit for bit: how many
/// columns differ.
fn quant_cols_differ(a: &[u8], b: &[u8], k: usize, n: usize) -> usize {
    let n_pad = n_padded(n as i64) as usize;
    (0..n)
        .filter(|&c| {
            let (qa, da) = get_col(a, k, n_pad, c);
            let (qb, db) = get_col(b, k, n_pad, c);
            qa != qb || da.iter().zip(&db).any(|(x, y)| x.to_bits() != y.to_bits())
        })
        .count()
}

/// On a GB10 the fused quantizations write the bytes of the op's quantization of ggml-cuda's
/// F32 norm (`mul(rms_norm(x), w)`) and GeGLU (`geglu_split`), and the values they quantize
/// (with `y`) are ggml's bit for bit; the op over a quantized input ([`mul_mat_q`]) is
/// ggml's product bit for bit (FFN shapes), all at 512 to 4097 columns.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_fused_quant_is_byte_exact() {
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, sms)) = gb10() else { return };
    let eps = 1e-6f32;
    let mut cases = 0;
    let mut bad = Vec::new();
    for n in CHECKED_N {
        // The norm over Gemma 4 31B's residual, the GeGLU over its FFN's hidden values.
        for (kind, k) in [("norm", 5376i64), ("geglu", 21504i64)] {
            let ctx = Context::new(32, 32);
            let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
            let x2 = if kind == "norm" {
                ctx.new_tensor(types::F32, &[k]).as_input()
            } else {
                ctx.new_tensor(types::F32, &[k, n]).as_input()
            };
            let (want, fused) = if kind == "norm" {
                (
                    ctx.mul(ctx.rms_norm(x, eps), x2),
                    quantize_norm(&ctx, x, x2, eps, true),
                )
            } else {
                (ctx.geglu_split(x, x2), quantize_geglu(&ctx, x, x2, true))
            };
            let want = want.as_output();
            let plain = quantize(&ctx, want).as_output();
            let fused = fused.as_output();
            let mut graph = ctx.new_graph(32);
            for t in [want, plain, fused] {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_f32(&input(k as usize, n as usize, 40 + n as u64));
            if kind == "norm" {
                let wv: Vec<f32> = noise(k as usize, 41)
                    .iter()
                    .map(|v| 1.0 + 0.5 * v)
                    .collect();
                x2.set_f32(&wv);
            } else {
                x2.set_f32(&input(k as usize, n as usize, 42 + n as u64));
            }
            gpu.compute(&graph).unwrap();
            let qb = quant_bytes(k, n);
            let fb = fused.bytes();
            let values = want
                .bytes()
                .chunks(4)
                .zip(fb[qb..].chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            let cols = quant_cols_differ(&plain.bytes(), &fb, k as usize, n as usize);
            if values > 0 || cols > 0 {
                println!(
                    "{kind} {k} x {n}: {values} F32 values differ, {cols} quantized columns differ"
                );
                bad.push(format!("{kind} {k} x {n}"));
            }
            cases += 1;
        }
        // The op on a quantized input against ggml's mul_mat.
        for (k, rows) in [(5376i64, 21504i64), (21504, 5376), (5376, 8192)] {
            if !ggml_tiles(rows, k, n, sms) {
                continue;
            }
            let ggml = case(&mut gpu, false, k, rows, n);
            let ctx = Context::new(16, 16);
            let w = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
            let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
            let y = mul_mat_q(&ctx, w, quantize(&ctx, x), n).as_output();
            let mut graph = ctx.new_graph(16);
            graph.expand(y);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            w.set_bytes(&q4_0_matrix(k as usize, rows as usize, (k + rows) as u64));
            x.set_f32(&input(k as usize, n as usize, n as u64));
            gpu.compute(&graph).unwrap();
            let diff = y
                .bytes()
                .chunks(4)
                .zip(ggml.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            if diff > 0 {
                println!(
                    "{k}x{rows}, {n} columns, quantized input: {diff} values differ from ggml"
                );
                bad.push(format!("{k}x{rows} at {n}"));
            }
            cases += 1;
        }
    }
    println!(
        "fused quantizations and quantized inputs: {cases} cases, {} differ",
        bad.len()
    );
    assert!(bad.is_empty(), "differ: {}", bad.join(", "));
}

/// The time per call at 4096 columns of the op over an input quantized in advance (no
/// quantization in the timed graph) at the timing test's shapes, and of the producers it
/// replaces: ggml's norm and GeGLU (F32 out) with the op's quantization of their output, and
/// the fused quantizations. 8 calls a graph on two inputs in turn, the second of two runs.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_fused_time_at_4096_rows() {
    const CALLS: usize = 8;
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, _)) = gb10() else { return };
    let n = 4096i64;
    // The second run's time per call.
    fn time(gpu: &mut Backend, graph: &crate::ggml::Graph<'_>) -> f64 {
        gpu.compute(graph).unwrap();
        let t0 = std::time::Instant::now();
        gpu.compute(graph).unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / CALLS as f64
    }
    for (k, rows) in CHECKED {
        let ctx = Context::new(64, 64);
        let w = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
        let bytes = quant_bytes(k, n) as i64;
        let xs = [0, 1].map(|_| ctx.new_tensor(types::I8, &[bytes]).as_input().as_output());
        let mut graph = ctx.new_graph(64);
        let mut last = None;
        for c in 0..CALLS {
            let y = mul_mat_q(&ctx, w, xs[c % 2], n);
            graph.expand(y);
            last = Some(y);
        }
        last.unwrap().as_output();
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        w.set_bytes(&q4_0_matrix(k as usize, rows as usize, 11));
        for (s, x) in xs.iter().enumerate() {
            let v = input(k as usize, n as usize, 12 + s as u64);
            x.set_bytes(&quant_buf(&v, k as usize, n as usize));
        }
        let ms = time(&mut gpu, &graph);
        let tops = 2.0 * (k * rows * n) as f64 / ms / 1e9;
        println!("{k}x{rows}, {n} columns, quantized input: {ms:.2} ms ({tops:.0} TOPS)");
    }
    // The producers: 0 ggml's F32 op, 1 it plus the op's quantization, 2 the fused one.
    let eps = 1e-6f32;
    for (kind, k) in [("norm", 5376i64), ("geglu", 21504i64)] {
        let mut ms = [0f64; 3];
        for (i, ms) in ms.iter_mut().enumerate() {
            let ctx = Context::new(64, 128);
            let xs = [0, 1].map(|_| ctx.new_tensor(types::F32, &[k, n]).as_input().as_output());
            let x2s = [0, 1].map(|_| {
                if kind == "norm" {
                    ctx.new_tensor(types::F32, &[k])
                } else {
                    ctx.new_tensor(types::F32, &[k, n])
                }
                .as_input()
                .as_output()
            });
            let mut graph = ctx.new_graph(128);
            let mut last = None;
            for c in 0..CALLS {
                let (x, x2) = (xs[c % 2], x2s[c % 2]);
                let y = match (i, kind) {
                    (2, "norm") => quantize_norm(&ctx, x, x2, eps, false),
                    (2, _) => quantize_geglu(&ctx, x, x2, false),
                    (_, "norm") => ctx.mul(ctx.rms_norm(x, eps), x2),
                    _ => ctx.geglu_split(x, x2),
                };
                let y = if i == 1 { quantize(&ctx, y) } else { y };
                graph.expand(y);
                last = Some(y);
            }
            last.unwrap().as_output();
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            for (s, (x, x2)) in xs.iter().zip(&x2s).enumerate() {
                x.set_f32(&input(k as usize, n as usize, 50 + s as u64));
                if kind == "norm" {
                    x2.set_f32(&vec![1.0; k as usize]);
                } else {
                    x2.set_f32(&input(k as usize, n as usize, 52 + s as u64));
                }
            }
            *ms = time(&mut gpu, &graph);
        }
        println!(
            "{kind} {k} x {n}: ggml {:.2} ms, + quantization {:.2} ms, fused {:.2} ms (saves {:.2} ms)",
            ms[0],
            ms[1],
            ms[2],
            ms[1] - ms[2]
        );
    }
}

/// On the CPU the GeGLU pair ([`mul_mat_geglu_q`]) writes the bytes of the gate and up matmuls
/// over the quantized input then [`quantize_geglu`] of their outputs.
#[test]
fn cpu_q40_geglu_pair_is_the_matmuls_and_quantize_geglu() {
    let (k, rows, n) = (512i64, 256i64, 520i64);
    let ctx = Context::new(32, 32);
    let gate = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
    let up = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let xq = quantize(&ctx, x);
    let g = mul_mat_q(&ctx, gate, xq, n);
    let u = mul_mat_q(&ctx, up, xq, n);
    let want = quantize_geglu(&ctx, g, u, false).as_output();
    let got = mul_mat_geglu_q(&ctx, gate, up, xq, n).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(want);
    graph.expand(got);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    gate.set_bytes(&q4_0_matrix(k as usize, rows as usize, 61));
    up.set_bytes(&q4_0_matrix(k as usize, rows as usize, 62));
    x.set_f32(&input(k as usize, n as usize, 63));
    backend.compute(&graph).unwrap();
    assert_eq!(got.ne()[0] as usize, quant_bytes(rows, n));
    let (a, b) = (want.bytes(), got.bytes());
    assert_eq!(quant_cols_differ(&a, &b, rows as usize, n as usize), 0);
    // The quantized GeGLU is not trivial: the all-zero input column quantizes to zeros, the
    // others not.
    let (q0, _) = get_col(&b, rows as usize, n_padded(n) as usize, 0);
    let (q1, _) = get_col(&b, rows as usize, n_padded(n) as usize, 1);
    assert!(q0.iter().any(|&v| v != 0) && q1.iter().all(|&v| v == 0));
}

/// The pair accepts matching gate and up weights only.
#[test]
fn geglu_pair_fits_matching_weights() {
    let ctx = Context::new(8, 0);
    let gate = ctx.new_tensor(types::Q4_0, &[5376, 21504]);
    let up = ctx.new_tensor(types::Q4_0, &[5376, 21504]);
    let other = ctx.new_tensor(types::Q4_0, &[5376, 8192]);
    let q8 = ctx.new_tensor(types::Q8_0, &[5376, 21504]);
    assert!(geglu_pair_fits(gate, up));
    assert!(!geglu_pair_fits(gate, other));
    assert!(!geglu_pair_fits(gate, q8));
}

/// On a GB10 the GeGLU pair ([`mul_mat_geglu_q`]) writes round 4's bytes, the op's gate and up
/// matmuls over the quantized input and then [`quantize_geglu`] of their F32 outputs, at the
/// timing test's shapes as gate / up pairs and 512 to 4097 columns.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_geglu_pair_is_byte_exact() {
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, _)) = gb10() else { return };
    let mut cases = 0;
    let mut bad = Vec::new();
    for (k, rows) in CHECKED {
        let wg = q4_0_matrix(k as usize, rows as usize, (k + rows) as u64);
        let wu = q4_0_matrix(k as usize, rows as usize, (k + rows + 1) as u64);
        for n in CHECKED_N {
            let ctx = Context::new(32, 32);
            let gate = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
            let up = ctx.new_tensor(types::Q4_0, &[k, rows]).as_input();
            let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
            let xq = quantize(&ctx, x);
            let g = mul_mat_q(&ctx, gate, xq, n);
            let u = mul_mat_q(&ctx, up, xq, n);
            let want = quantize_geglu(&ctx, g, u, false).as_output();
            let got = mul_mat_geglu_q(&ctx, gate, up, xq, n).as_output();
            let mut graph = ctx.new_graph(32);
            graph.expand(want);
            graph.expand(got);
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            gate.set_bytes(&wg);
            up.set_bytes(&wu);
            x.set_f32(&input(k as usize, n as usize, 70 + n as u64));
            gpu.compute(&graph).unwrap();
            let cols = quant_cols_differ(&want.bytes(), &got.bytes(), rows as usize, n as usize);
            if cols > 0 {
                println!("GeGLU pair {k}x{rows}, {n} columns: {cols} quantized columns differ");
                bad.push(format!("{k}x{rows} at {n}"));
            }
            cases += 1;
        }
    }
    println!(
        "GeGLU pair vs matmuls + quantize_geglu: {cases} cases, {} differ",
        bad.len()
    );
    assert!(bad.is_empty(), "differ: {}", bad.join(", "));
}

/// The time at 4096 columns of Gemma 4 31B's FFN over an input quantized in advance: round 4's
/// gate and up matmuls and fused GeGLU quantization against the GeGLU pair, then each with the
/// down matmul (the whole FFN). 4 FFNs a graph on two inputs in turn, the second of two runs.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_q40_mm_ffn_time_at_4096_rows() {
    const CALLS: usize = 4;
    let _one = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some((mut gpu, _)) = gb10() else { return };
    let (n, k, hid) = (4096i64, 5376i64, 21504i64);
    let wg = q4_0_matrix(k as usize, hid as usize, 81);
    let wu = q4_0_matrix(k as usize, hid as usize, 82);
    let wd = q4_0_matrix(hid as usize, k as usize, 83);
    let bufs: Vec<Vec<u8>> = (0..2)
        .map(|s| {
            quant_buf(
                &input(k as usize, n as usize, 84 + s),
                k as usize,
                n as usize,
            )
        })
        .collect();
    // 0 round 4 gate / up + GeGLU, 1 the pair, 2 and 3 the same with down.
    let mut ms = [0f64; 4];
    for (i, ms) in ms.iter_mut().enumerate() {
        let ctx = Context::new(64, 64);
        let gate = ctx.new_tensor(types::Q4_0, &[k, hid]).as_input();
        let up = ctx.new_tensor(types::Q4_0, &[k, hid]).as_input();
        let down = ctx.new_tensor(types::Q4_0, &[hid, k]).as_input();
        let bytes = quant_bytes(k, n) as i64;
        let xs = [0, 1].map(|_| ctx.new_tensor(types::I8, &[bytes]).as_input().as_output());
        let mut graph = ctx.new_graph(64);
        let mut last = None;
        for c in 0..CALLS {
            let xq = xs[c % 2];
            let act = if i % 2 == 0 {
                let g = mul_mat_q(&ctx, gate, xq, n);
                let u = mul_mat_q(&ctx, up, xq, n);
                quantize_geglu(&ctx, g, u, false)
            } else {
                mul_mat_geglu_q(&ctx, gate, up, xq, n)
            };
            let y = if i >= 2 {
                mul_mat_q(&ctx, down, act, n)
            } else {
                act
            };
            graph.expand(y);
            last = Some(y);
        }
        last.unwrap().as_output();
        ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        gate.set_bytes(&wg);
        up.set_bytes(&wu);
        down.set_bytes(&wd);
        for (x, b) in xs.iter().zip(&bufs) {
            x.set_bytes(b);
        }
        gpu.compute(&graph).unwrap();
        let t0 = std::time::Instant::now();
        gpu.compute(&graph).unwrap();
        *ms = t0.elapsed().as_secs_f64() * 1e3 / CALLS as f64;
    }
    let tops = |mm: i64, t: f64| 2.0 * (mm * k * hid * n) as f64 / t / 1e9;
    println!(
        "FFN gate/up + GeGLU at {n}: round 4 {:.2} ms ({:.0} TOPS), pair {:.2} ms ({:.0} TOPS), {:.3}x",
        ms[0],
        tops(2, ms[0]),
        ms[1],
        tops(2, ms[1]),
        ms[0] / ms[1]
    );
    println!(
        "FFN gate/up + GeGLU + down at {n}: round 4 {:.2} ms ({:.0} TOPS), pair {:.2} ms ({:.0} TOPS), {:.3}x",
        ms[2],
        tops(3, ms[2]),
        ms[3],
        tops(3, ms[3]),
        ms[2] / ms[3]
    );
}
