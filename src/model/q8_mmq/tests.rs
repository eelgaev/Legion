//! The op's plan (ggml's MMQ tile widths, CTA counts and stream-k split), its shape rules and its
//! CPU implementation. The GPU bits against ggml's `mul_mat`: `legion q8-mmq-check` and the
//! server's `--deterministic` chat hashes on a GPU machine.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::mmq_preq::tests::{noise, q8_0_matrix};

#[test]
fn tile_width_is_ggml_j_best() {
    // Rows a multiple of 128: ggml's Q8_0 widths 8..=128 (no 56, 72, 88, 104, 120), the first
    // with the fewest column tiles.
    for (n, j) in [
        (9, 16),
        (16, 16),
        (17, 24),
        (48, 48),
        (49, 64),
        (65, 80),
        (128, 128),
        (129, 80),
        (255, 128),
    ] {
        assert_eq!(tile_width(2560, n), j, "{n} columns");
    }
    // Other row counts: the fallback widths 8, 16, 32, 64, 128.
    for (n, j) in [(9, 16), (17, 32), (33, 64), (48, 64), (65, 128)] {
        assert_eq!(tile_width(320, n), j, "{n} columns, 320 rows");
    }
}

#[test]
fn plan_is_ggml_stream_k_grid() {
    // 20 tiles on 48 SMs: under 90% of a wave, so the SMs' count of CTAs and a fixup.
    let p = plan(2560, 2560, 48, 48);
    assert_eq!((p.j, p.ntx, p.nty, p.bpt, p.ctas), (48, 1, 20, 80, 48));
    assert!(p.split() && !p.fallback);
    // 48 tiles fill a wave: a CTA per tile, nothing split.
    let p = plan(6144, 2560, 16, 48);
    assert_eq!((p.tiles(), p.ctas), (48, 48));
    assert!(!p.split());
    // 80 tiles are 83% of two waves.
    let p = plan(10240, 2560, 48, 48);
    assert_eq!((p.tiles(), p.ctas), (80, 48));
    assert!(p.split());
    // Rows not a multiple of 128.
    let p = plan(320, 10240, 48, 48);
    assert!(p.fallback);
    assert_eq!((p.j, p.nty, p.kernel()), (64, 3, 20));
    assert_eq!(plan(2560, 2560, 48, 48).kernel(), 6);
    // A CTA per tile: the kernels without the split-tile code.
    assert_eq!(plan(6144, 2560, 16, 48).kernel(), 2 + 21);
}

#[test]
fn pieces_cover_each_tile_in_k_order() {
    for &(rows, k, n) in &[
        (2560i64, 2560i64, 48i64),
        (10240, 2560, 32),
        (320, 10240, 48),
        (2560, 320, 9),
        (2560, 6144, 255),
    ] {
        let p = plan(rows, k, n, 48);
        for t in 0..p.tiles() {
            let pieces = p.pieces(t);
            assert!(!pieces.is_empty());
            assert_eq!(pieces[0].0, 0);
            assert_eq!(pieces.last().unwrap().1, p.bpt);
            for w in pieces.windows(2) {
                assert_eq!(w[0].1, w[1].0);
                // Pieces start on ggml's K iterations.
                assert_eq!(w[1].0 % BLOCKS_PER_ITER, 0);
            }
            if !p.split() {
                assert_eq!(pieces.len(), 1);
            }
        }
    }
}

#[test]
fn smem_is_ggml_nbytes_shared() {
    // J ids + 128 rows x 76 ints of weight tile + J blocks of 144 bytes padded to 1024.
    assert_eq!(plan(2560, 2560, 48, 48).smem(), 48 * 4 + 38912 + 7168);
    assert_eq!(plan(2560, 2560, 128, 48).smem(), 128 * 4 + 38912 + 18432);
    assert_eq!(plan(2560, 2560, 9, 48).smem(), 16 * 4 + 38912 + 3072);
}

fn shape(ty: ffi::ggml_type, ne: [i64; 4]) -> (ffi::ggml_type, [i64; 4], [usize; 4]) {
    let ts = if ty == types::Q8_0 { BLOCK_BYTES } else { 4 };
    let row = if ty == types::Q8_0 {
        (ne[0] / QK) as usize * BLOCK_BYTES
    } else {
        ne[0] as usize * 4
    };
    let nb1 = row;
    let nb2 = nb1 * ne[1] as usize;
    (ty, ne, [ts, nb1, nb2, nb2 * ne[2] as usize])
}

#[test]
fn fits_q8_0_weights_of_9_to_255_columns() {
    let w = shape(types::Q8_0, [2560, 10240, 1, 1]);
    let x = |n| shape(types::F32, [2560, n, 1, 1]);
    for n in [9, 16, 48, 64, 128, 255] {
        assert!(fits(w, x(n), 48), "{n} columns");
    }
    assert!(!fits(w, x(8), 48));
    assert!(!fits(w, x(256), 48));
    assert!(!fits(w, x(48), 0));
    assert!(!fits(w, x(48), MAX_CTAS + 1));
    // Another type, a mismatched k, a 3D input.
    assert!(!fits(shape(types::Q4_0, [2560, 10240, 1, 1]), x(48), 48));
    assert!(!fits(w, shape(types::F32, [5120, 48, 1, 1]), 48));
    assert!(!fits(w, shape(types::F32, [2560, 16, 3, 1]), 48));
    // Short rows (k under 512) and odd row counts fit.
    assert!(fits(
        shape(types::Q8_0, [320, 10240, 1, 1]),
        shape(types::F32, [320, 48, 1, 1]),
        48
    ));
    assert!(fits(shape(types::Q8_0, [2560, 330, 1, 1]), x(48), 48));
}

#[test]
fn off_without_a_gpu() {
    // The engine sets the GPU on load; the tests never do (and run without CUDA here).
    let ctx = Context::new(4, 0);
    let w = ctx.new_tensor(types::Q8_0, &[2560, 256]);
    let x = ctx.new_tensor(types::F32, &[2560, 48]);
    assert!(!takes(w, x));
}

#[test]
fn quantize_col_is_ggml_mmq_q8_1() {
    let x = noise(64, 3);
    let (q, d) = quantize_col(&x);
    for b in 0..2 {
        let v = &x[b * 32..b * 32 + 32];
        let amax = v.iter().fold(0f32, |m, v| m.max(v.abs()));
        let i = v.iter().position(|v| v.abs() == amax).unwrap();
        assert_eq!(q[b * 32 + i].unsigned_abs(), 127);
        assert_eq!(d[b].to_bits(), (1.0 / (127.0 / amax)).to_bits());
    }
    let (q, d) = quantize_col(&[0.0; 32]);
    assert_eq!((q, d), (vec![0; 32], vec![0.0]));
}

/// `mul_mat` on the CPU (split tiles: 20 tiles on 48 "SMs") against the product of the
/// dequantized operands in f64.
#[test]
fn cpu_mul_mat_is_the_dequantized_product() {
    for &(k, rows, n) in &[(1024usize, 2560usize, 48usize), (320, 330, 9)] {
        let wb = q8_0_matrix(k, rows, 7);
        let x = noise(k * n, 8);
        let ctx = Context::new(16, 16);
        let w = ctx
            .new_tensor(types::Q8_0, &[k as i64, rows as i64])
            .as_input();
        let xt = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
        let y = mul_mat_sms(&ctx, w, xt, 48).as_output();
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
                    let dw =
                        unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]])) };
                    for v in 0..32 {
                        let t = dw as f64
                            * blk[2 + v] as i8 as f64
                            * dx[b] as f64
                            * qx[b * 32 + v] as f64;
                        want += t;
                        mag += t.abs();
                    }
                }
                let g = got[j * rows + r] as f64;
                worst = worst.max((g - want).abs() / mag.max(1e-30));
            }
        }
        assert!(
            worst < 1e-5,
            "k {k}, rows {rows}: worst relative error {worst}"
        );
    }
}

#[test]
fn fast_kernel_takes_k_multiple_of_256_and_aligned_rows() {
    // Flash-Next's verify shapes at 16 and 144 columns on 48 SMs.
    for &(k, rows) in &[
        (2560i64, 10240i64),
        (6144, 2560),
        (10240, 2560),
        (10240, 320),
    ] {
        for n in [16, 36, 144, 255] {
            let p = plan(rows, k, n, 48);
            for warps in [8, 4] {
                assert!(fast_fits(&p, k, k / QK, warps), "k {k} rows {rows} n {n}");
            }
            assert!(!fast_fits(&p, k, k / QK, 0));
            assert!(!fast_fits(&p, k, k / QK, 2));
            // A row stride not a multiple of 8 blocks puts a row's stages off 16-byte boundaries.
            assert!(!fast_fits(&p, k, k / QK + 1, 8));
        }
    }
    // k = 320 is not whole 256-value stages: ggml's tile code runs it.
    let p = plan(10240, 320, 16, 48);
    assert!(!fast_fits(&p, 320, 10, 8));
    assert_eq!((fast_slices(8), fast_slices(4)), (1, 2));
}

#[test]
fn fast_quant_bytes_pads_columns_to_tiles() {
    let p = plan(2560, 2560, 144, 48);
    assert_eq!((p.j, p.ntx), (80, 2));
    // 80 blocks of K, 160 padded columns: 32 bytes of quants and 4 of scale each.
    assert_eq!(fast_quant_bytes(2560, &p), 80 * 160 * 32 + 80 * 160 * 4);
    assert!(fast_quant_bytes(10240, &plan(2560, 10240, 255, 48)) <= MAX_QUANT_BYTES);
}

#[test]
fn set_q8_mmq_fast_keeps_known_kernels() {
    let old = q8_mmq_fast();
    for (w, want) in [(8, 8), (4, 4), (0, 0), (3, 0), (16, 0)] {
        set_q8_mmq_fast(w);
        assert_eq!(q8_mmq_fast(), want);
    }
    set_q8_mmq_fast(old);
}

#[test]
fn fixup_launch_kernels_index_the_table() {
    // Split, ggml's widths: the count-free matmuls 42 after the counting ones (43-63), the
    // fixups at 64 + J / 8 - 1 (64-79).
    let p = plan(2560, 2560, 48, 48);
    assert!(p.split());
    assert_eq!(
        (p.kernel(), p.kernel_no_count(), p.fixup_kernel()),
        (6, 48, 69)
    );
    let p = plan(320, 10240, 48, 48);
    assert!(p.split() && p.fallback);
    assert_eq!(
        (p.kernel(), p.kernel_no_count(), p.fixup_kernel()),
        (20, 62, 71)
    );
    // The widest tiles: the table's last fixup.
    assert_eq!(plan(2560, 2560, 128, 48).fixup_kernel(), 79);
    // Every split plan's indices stay inside the 80-entry table.
    for &(rows, k) in &[(10240i64, 2560i64), (512, 2560), (2560, 640), (320, 10240)] {
        for n in [9i64, 16, 36, 100, 144, 255] {
            let p = plan(rows, k, n, 48);
            if p.split() {
                assert!((43..64).contains(&p.kernel_no_count()), "{rows} {k} {n}");
                assert!((64..80).contains(&p.fixup_kernel()), "{rows} {k} {n}");
            }
        }
    }
}

#[test]
fn fixup_launch_only_for_split_plans() {
    let unsplit = plan(6144, 2560, 16, 48);
    assert!(!unsplit.split());
    let split = plan(512, 2560, 36, 48);
    assert!(split.split());
    for f in [Fixup::Auto, Fixup::Last, Fixup::Launch] {
        assert!(!f.launch(&unsplit), "{f:?}");
        assert_eq!(Fixup::from_code(f.code()), f);
    }
    assert!(Fixup::Launch.launch(&split));
    assert!(!Fixup::Last.launch(&split));
    assert_eq!(Fixup::from_code(7), Fixup::Auto);
    assert_eq!(Fixup::default(), Fixup::Auto);
}

#[test]
fn fixup_auto_follows_the_gb10_measurements() {
    // (rows, k, columns, launch): the faster add-up per Flash-Next case on 48 SMs (#858).
    let cases = [
        // Tiles over 4+ CTAs: the in-kernel add-up for 16-wide tiles, the launch from 32-wide.
        (512, 2560, 16, false),
        (512, 2560, 36, true),
        (512, 2560, 144, true),
        (640, 2560, 16, false),
        (640, 2560, 36, true),
        (320, 10240, 16, false),
        (320, 10240, 36, true),
        (320, 10240, 144, true),
        // Tiles over 1-4 CTAs: the launch.
        (2560, 640, 16, true),
        (2560, 640, 144, true),
        (2560, 6144, 36, true),
        (2560, 10240, 144, true),
        // Fewer CTAs than tiles: the launch for long K only.
        (10240, 2560, 16, true),
        (10240, 2560, 144, true),
        (10240, 320, 16, false),
        (10240, 320, 144, false),
    ];
    for (rows, k, n, launch) in cases {
        let p = plan(rows, k, n, 48);
        assert!(p.split(), "{rows} {k} {n}");
        assert_eq!(Fixup::Auto.launch(&p), launch, "{rows} {k} {n}: {p:?}");
    }
    // Unsplit plans never launch a fixup.
    assert!(!Fixup::Auto.launch(&plan(12288, 2560, 144, 48)));
}

/// The quantize launches of a graph's nodes (#858): a node whose last op node before it read the
/// same input on ggml's tile code reuses its blocks; the node that quantizes zeroes the counters
/// of its run's most tiles; the streaming kernel (another layout) and other inputs break a run.
#[test]
fn quant_run_reuses_the_blocks_of_the_same_input() {
    let ctx = Context::new(32, 32);
    let wa = ctx.new_tensor(types::Q8_0, &[2560, 512]).as_input();
    let wb = ctx.new_tensor(types::Q8_0, &[2560, 2560]).as_input();
    let x1 = ctx.new_tensor(types::F32, &[2560, 36]).as_input();
    let x2 = ctx.new_tensor(types::F32, &[2560, 36]).as_input();
    let a = mul_mat_sms(&ctx, wa, x1, 48);
    let b = mul_mat_sms(&ctx, wb, x1, 48);
    let other = ctx.add(x1, x1);
    let d = mul_mat_sms(&ctx, wa, x2, 48);
    let e = mul_mat_sms(&ctx, wb, x1, 48);
    let f = mul_mat_fast(&ctx, wb, x1, 48, 8, Fixup::Auto);
    let g = mul_mat_sms(&ctx, wa, x1, 48);
    let mut graph = ctx.new_graph(32);
    for t in [a, b, other, d, e, f, g] {
        graph.expand(t);
    }
    // SAFETY: the graph's node array holds its nodes, valid tensors.
    let nodes =
        unsafe { std::slice::from_raw_parts(ffi::ggml_graph_nodes(graph.raw()), graph.nodes()) };
    let at = |t: Tensor<'_>| nodes.iter().position(|&n| n == t.raw()).unwrap();
    let tiles = |w: i64| plan(w, 2560, 36, 48).tiles();
    // SAFETY: nodes of the graph; `at` picks the op's nodes.
    let run = |t: Tensor<'_>| unsafe { quant_run(nodes, at(t)) };
    assert_eq!(run(a), Some(tiles(512).max(tiles(2560))));
    assert_eq!(run(b), None);
    assert_eq!(run(d), Some(tiles(512)));
    assert_eq!(run(e), Some(tiles(2560)));
    // SAFETY: as above.
    assert!(!unsafe { runs_tile(&*f.raw()) });
    assert_eq!(run(f), Some(tiles(2560)));
    assert_eq!(run(g), Some(tiles(512)));
}

#[test]
fn fastdiv_divides_as_integer_division() {
    // The divisors of the op's plans (K blocks per tile, column tiles) and powers of two; the
    // numerators up to the op's largest K block index (tiles * bpt < 2^30) and its edges.
    let big = (1u32 << 30) + 3;
    for d in [
        1u32, 2, 3, 5, 7, 10, 20, 31, 32, 64, 80, 192, 320, 1000, 4096, big,
    ] {
        let (mp, l) = fastdiv(d);
        let div = |n: u32| ((((n as u64 * mp as u64) >> 32) as u32 + n) >> l, n / d);
        for n in (0..100_000u32).chain([(1 << 30) - 1, (1 << 31) - 1, d - 1, d, d + 1]) {
            let (got, want) = div(n);
            assert_eq!(got, want, "{n} / {d}");
        }
    }
}

#[test]
fn beats_ggml_is_the_narrow_deep_split_plans() {
    // Qwen3.8-Flash-Next's Q8_0 weights (`k x rows`) on a GB10's 48 SMs: only the 10240x320
    // weight at 9-16 columns (J 16 on ggml's fallback widths, 3 tiles over 48 CTAs).
    let shapes = [
        (2560, 10240),
        (2560, 6144),
        (2560, 12288),
        (2560, 512),
        (6144, 2560),
        (10240, 2560),
        (2560, 640),
        (640, 2560),
        (10240, 320),
        (320, 10240),
    ];
    for (k, rows) in shapes {
        for n in MIN_TOKENS..=MAX_TOKENS {
            let want = (k, rows) == (10240, 320) && n <= 16;
            assert_eq!(beats_ggml(&plan(rows, k, n, 48)), want, "{k}x{rows} at {n}");
        }
    }
}
