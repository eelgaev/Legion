//! The cluster selections: ggml-cuda's argmax tie rule against a thread-by-thread model of
//! `argmax_f32`, the shape checks, the CPU ops against their references, and on CUDA (ignored
//! tests) the kernels' bits against `qsa_topk` and ggml's argmax, with their times.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::qwen4::qsa_attn::top_k_row;

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

/// ggml-cuda's `argmax_f32` (`third_party/ggml/src/ggml-cuda/argmax.cu`) step by step for a row
/// of 1,024 columns or more: 1,024 threads scan, then a xor-butterfly in each warp and one over
/// the warps' results, each keeping the strictly larger value.
fn ggml_cuda_argmax(row: &[f32]) -> i32 {
    const T: usize = 1024;
    assert!(row.len() >= T);
    let mut v = [-f32::MAX; T];
    let mut a = [-1i32; T];
    for t in 0..T {
        for c in (t..row.len()).step_by(T) {
            if row[c] > v[t] {
                v[t] = row[c];
                a[t] = c as i32;
            }
        }
    }
    let butterfly = |v: &mut [f32], a: &mut [i32]| {
        let mut off = 16;
        while off > 0 {
            let (pv, pa) = (v.to_vec(), a.to_vec());
            for l in 0..32 {
                if pv[l ^ off] > v[l] {
                    v[l] = pv[l ^ off];
                    a[l] = pa[l ^ off];
                }
            }
            off >>= 1;
        }
    };
    for w in 0..T / 32 {
        butterfly(&mut v[w * 32..(w + 1) * 32], &mut a[w * 32..(w + 1) * 32]);
    }
    let mut wv: Vec<f32> = (0..32).map(|w| v[w * 32]).collect();
    let mut wa: Vec<i32> = (0..32).map(|w| a[w * 32]).collect();
    butterfly(&mut wv, &mut wa);
    wa[0]
}

/// Rows with many equal maxima: values on a few levels, some rows with -inf, NaN, -FLT_MAX and
/// signed zeros.
fn tie_rows(n: usize, rows: usize, seed: u64) -> Vec<f32> {
    let mut v = noise(n * rows, seed);
    for (r, row) in v.chunks_mut(n).enumerate() {
        let levels = [2.0f32, 3.0, 7.0, 1000.0][r % 4];
        for (j, x) in row.iter_mut().enumerate() {
            *x = (x.abs() * levels).floor();
            match (r % 5, j % 97) {
                (1, 0) => *x = f32::NEG_INFINITY,
                (2, 3) => *x = f32::NAN,
                (3, 5) => *x = -f32::MAX,
                (4, _) => *x = if j % 2 == 0 { 0.0 } else { -0.0 },
                _ => {}
            }
        }
    }
    v
}

#[test]
fn argmax_row_is_ggml_cuda_pick() {
    // Equal maxima at columns 1 and 2: lane 1 (bitrev 16) loses to lane 2 (bitrev 8).
    let mut row = vec![0f32; 4096];
    row[1] = 1.0;
    row[2] = 1.0;
    assert_eq!(argmax_row(&row), 2);
    assert_eq!(ggml_cuda_argmax(&row), 2);
    // Lane 1 of warp 0 beats lane 0 of warp 1 (the warps' butterfly comes last).
    row.fill(0.0);
    row[1] = 1.0;
    row[32] = 1.0;
    assert_eq!(argmax_row(&row), 1);
    assert_eq!(ggml_cuda_argmax(&row), 1);
    // One thread's columns: the first.
    row.fill(0.0);
    row[1027] = 1.0;
    row[3] = 1.0;
    assert_eq!(argmax_row(&row), 3);
    // All equal: column 0; nothing above -FLT_MAX: -1.
    assert_eq!(argmax_row(&vec![5.0; 2000]), 0);
    assert_eq!(argmax_row(&vec![-f32::MAX; 2000]), -1);
    assert_eq!(ggml_cuda_argmax(&vec![-f32::MAX; 2000]), -1);
    assert_eq!(argmax_row(&vec![f32::NAN; 2000]), -1);
    assert_eq!(argmax_row(&vec![f32::NEG_INFINITY; 2000]), -1);
    for (i, &n) in [1024usize, 1025, 2047, 4096, 5000, 33333]
        .iter()
        .enumerate()
    {
        let rows = 20;
        let v = tie_rows(n, rows, 3 + i as u64);
        for (r, row) in v.chunks(n).enumerate() {
            assert_eq!(
                argmax_row(row),
                ggml_cuda_argmax(row),
                "n {n} row {r}: tie rule"
            );
        }
    }
}

#[test]
fn argmax_rank_round_trips_columns() {
    // The kernel's rank holds the column: (warp, lane) bit-reversed, then column / 1024.
    let mut seen = std::collections::HashSet::new();
    for col in (0..1 << 20).step_by(7) {
        assert!(seen.insert(argmax_rank(col)), "{col}");
        let r = argmax_rank(col);
        let rev5 = |v: u64| ((v as u32).reverse_bits() >> 27) as usize;
        let back =
            (((r & 0xffff_ffff) as usize) << 10) | (rev5(r >> 37) << 5) | rev5((r >> 32) & 31);
        assert_eq!(back, col);
    }
}

#[test]
fn fits_take_decode_shapes() {
    let nb = |n: i64| [4usize, 4 * n as usize, 4 * n as usize, 4 * n as usize];
    let f32t = types::F32;
    // Top-k: k up to n, MIN_TOPK_COLS or more scores, 1 to 4 rows (MAX_TOPK_ROWS from
    // MIN_TOPK_COLS_8).
    assert!(topk_fits(f32t, [8192, 1, 1, 1], nb(8192), 512));
    assert!(topk_fits(f32t, [8192, 4, 1, 1], nb(8192), 8192), "k == n");
    assert!(!topk_fits(f32t, [8192, 4, 1, 1], nb(8192), 8193), "k > n");
    assert!(
        !topk_fits(f32t, [8191, 1, 1, 1], nb(8191), 512),
        "short rows"
    );
    assert!(!topk_fits(f32t, [8192, 4, 1, 1], nb(8192), 0));
    assert!(!topk_fits(f32t, [16383, 5, 1, 1], nb(16383), 512));
    assert!(topk_fits(
        f32t,
        [16384, MAX_TOPK_ROWS, 1, 1],
        nb(16384),
        512
    ));
    assert!(
        !topk_fits(f32t, [25000, MAX_TOPK_ROWS + 1, 1, 1], nb(25000), 512),
        "prefill"
    );
    assert!(!topk_fits(types::F16, [8192, 1, 1, 1], nb(8192), 512));
    assert!(!topk_fits(f32t, [8192, 1, 2, 1], nb(8192), 512));
    assert!(!topk_fits(
        f32t,
        [8192, 1, 1, 1],
        [8, 65536, 65536, 65536],
        512
    ));
    // Argmax: rows of 1,024 columns or more, 1 to MAX_ARGMAX_ROWS rows, contiguous.
    assert!(argmax_fits(f32t, [248320, 1, 1, 1], nb(248320)));
    assert!(argmax_fits(
        f32t,
        [262144, MAX_ARGMAX_ROWS, 1, 1],
        nb(262144)
    ));
    assert!(!argmax_fits(
        f32t,
        [262144, MAX_ARGMAX_ROWS + 1, 1, 1],
        nb(262144)
    ));
    assert!(argmax_fits(f32t, [1024, 1, 1, 1], nb(1024)));
    assert!(
        !argmax_fits(f32t, [1023, 1, 1, 1], nb(1023)),
        "fewer threads in ggml"
    );
    assert!(
        !argmax_fits(f32t, [4096, 2, 1, 1], nb(8192)),
        "not contiguous"
    );
    assert!(!argmax_fits(types::F16, [4096, 2, 1, 1], nb(4096)));
}

#[test]
fn ops_off_without_cuda() {
    // The CPU device (and builds without CUDA) keep ggml's argmax and Legion's qsa_topk.
    set_cuda(false);
    let ctx = Context::new(8, 0);
    let x = ctx.new_tensor(types::F32, &[4096, 2]);
    assert!(!active());
    assert!(!takes_top_k(x, 512));
    assert_eq!(argmax(&ctx, x).ty(), types::I32);
}

/// The two ops on `backend` (built directly, whatever [`active`] says): the top-k of `scores`
/// and the argmax of `logits`.
fn op_case(
    backend: &mut Backend,
    scores: &[f32],
    n: i64,
    k: i64,
    logits: &[f32],
    vocab: i64,
) -> (Vec<i32>, Vec<i32>) {
    let ctx = Context::new(8, 8);
    let s = ctx
        .new_tensor(types::F32, &[n, scores.len() as i64 / n])
        .as_input();
    let l = ctx
        .new_tensor(types::F32, &[vocab, logits.len() as i64 / vocab])
        .as_input();
    let top = top_k(&ctx, s, k).as_output();
    let am = ctx
        .external(&ARGMAX, types::I32, [l.ne()[1], 1, 1, 1], &[l])
        .as_output();
    let mut g = ctx.new_graph(8);
    g.expand(top);
    g.expand(am);
    ctx.alloc_graph(&g, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    s.set_f32(scores);
    l.set_f32(logits);
    backend.compute(&g).unwrap();
    (top.to_i32(), am.to_i32())
}

#[test]
fn cpu_ops_match_references() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let (n, k, rows, vocab) = (3000usize, 512usize, 3usize, 5000usize);
    let scores = tie_rows(n, rows, 21);
    let logits = tie_rows(vocab, 6, 22);
    let (top, am) = op_case(
        &mut backend,
        &scores,
        n as i64,
        k as i64,
        &logits,
        vocab as i64,
    );
    let mut want = vec![0i32; k * rows];
    for (r, out) in want.chunks_mut(k).enumerate() {
        top_k_row(&scores[r * n..(r + 1) * n], out);
    }
    assert_eq!(top, want);
    let want: Vec<i32> = logits.chunks(vocab).map(ggml_cuda_argmax).collect();
    assert_eq!(am, want);
}

#[test]
fn argmax_parts_cover_rows() {
    assert_eq!(argmax_parts(1024), 1);
    assert_eq!(argmax_parts(4096), 1);
    assert_eq!(argmax_parts(4097), 2);
    assert_eq!(argmax_parts(248_320), 61);
    assert_eq!(argmax_parts(262_144), MAX_ARGMAX_PARTS);
    assert_eq!(argmax_parts(i32::MAX as i64), MAX_ARGMAX_PARTS);
    // Every part's slice (a multiple of 4, as the kernel takes it) covers the row.
    for n in [1024i64, 1025, 5003, 40_000, 248_320, 262_144, 1 << 24] {
        let parts = argmax_parts(n);
        let slice = ((n + parts - 1) / parts + 3) & !3;
        assert!(slice * parts >= n, "{n}");
    }
}

/// Top-k scores of the QSA indexer's kind: finite values with runs of -inf (masked pools), on a
/// few levels when `levels` is set (ties at the threshold).
#[cfg(feature = "cuda")]
fn topk_scores(n: usize, rows: usize, levels: u32, seed: u64) -> Vec<f32> {
    let mut v = noise(n * rows, seed);
    for (r, row) in v.chunks_mut(n).enumerate() {
        let (first, end) = (r % 13, n - (3 * r) % (n / 2));
        for (j, x) in row.iter_mut().enumerate() {
            *x = if j < first || j >= end {
                f32::NEG_INFINITY
            } else if levels > 0 {
                (x.abs() * levels as f32).floor()
            } else if j % 17 == 0 {
                0.0
            } else {
                x.abs()
            };
        }
    }
    v
}

#[cfg(feature = "cuda")]
fn gpu() -> Option<Backend> {
    use crate::ggml::{DeviceKind, device_by_kind};
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        eprintln!("cluster_select: not supported on this GPU, skipped");
        return None;
    }
    Some(gpu)
}

#[test]
fn best_rank_orders_as_argmax_rank() {
    let cols: Vec<usize> = (0..300_000).step_by(13).collect();
    for w in cols.windows(2) {
        let (a, b) = (w[0], w[1]);
        assert_eq!(best_col(best_rank(a)), a);
        assert_eq!(
            best_rank(a) < best_rank(b),
            argmax_rank(a) < argmax_rank(b),
            "{a} {b}"
        );
    }
    // A column offset of whole 1,024s moves the rank by that many, nothing else.
    assert_eq!(best_rank(5000 + 122 * 1024), best_rank(5000) + 122);
}

#[test]
fn joined_parts_are_the_argmax() {
    for (n, rows, seed) in [(2000usize, 3usize, 1u64), (124_928, 2, 2), (123_392, 5, 3)] {
        let x = tie_rows(n, rows, seed);
        let parts = argmax_parts(n as i64) as usize;
        let best: Vec<u32> = x
            .chunks(n)
            .flat_map(|row| best_row(row, parts))
            .flat_map(|(v, r)| [v, r])
            .collect();
        let (cols, vals) = join_best(&best, rows);
        for (r, row) in x.chunks(n).enumerate() {
            assert_eq!(cols[r] as i32, argmax_row(row), "n {n} row {r}");
            assert_eq!(vals[r], row[cols[r] as usize]);
        }
    }
    // No candidate: argmax's -1.
    let none = best_row(&vec![f32::NEG_INFINITY; 5000], 2);
    let flat: Vec<u32> = none.into_iter().flat_map(|(v, r)| [v, r]).collect();
    assert_eq!(join_best(&flat, 1), (vec![u32::MAX], vec![-f32::MAX]));
}

/// On a GPU where the kernels run, the cluster top-k gives `qsa_topk`'s output index for index
/// and the cluster argmax ggml-cuda's argmax, ties included: forced ties, k == n, -inf runs,
/// NaN, -FLT_MAX and signed zeros, rows from 1 to the limits.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_one_cta_kernels() {
    let Some(mut gpu) = gpu() else {
        return;
    };
    if let Some((size, active)) = cluster_limits() {
        eprintln!(
            "cluster limits: max cluster size {size}, active clusters of {CLUSTER}: {active}"
        );
    }
    // (n, rows, k, levels): 4K, 32K and 100K tokens of context at 4 tokens a pool, k == n,
    // one entry a thread and the re-read path.
    let topk_cases = [
        (1024usize, 1usize, 512usize, 0u32),
        (1024, 8, 512, 3),
        (8192, 1, 512, 0),
        (8192, 4, 512, 50),
        (25000, 2, 512, 0),
        (25000, 8, 512, 1000),
        (512, 3, 512, 0),
        (600, 5, 600, 2),
        (3000, 7, 64, 1),
        (300_000, 2, 512, 0),
        (300_000, 3, 2048, 20),
    ];
    for (i, &(n, rows, k, levels)) in topk_cases.iter().enumerate() {
        let scores = topk_scores(n, rows, levels, 50 + i as u64);
        let ctx = Context::new(8, 8);
        let s = ctx
            .new_tensor(types::F32, &[n as i64, rows as i64])
            .as_input();
        let own = crate::model::qwen4::qsa_attn::top_k(&ctx, s, k as i64).as_output();
        let cl = top_k(&ctx, s, k as i64).as_output();
        let mut g = ctx.new_graph(8);
        g.expand(own);
        g.expand(cl);
        ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        s.set_f32(&scores);
        gpu.compute(&g).unwrap();
        assert_eq!(
            cl.to_i32(),
            own.to_i32(),
            "top-k {n} x {rows}, k {k}, levels {levels}"
        );
    }
    // (vocab, rows): Qwen3.8-Flash-Next's and Gemma 4's vocabularies, a draft vocabulary,
    // short and odd rows.
    let argmax_cases = [
        (248_320usize, 1usize),
        (248_320, 4),
        (248_320, 16),
        (262_144, 1),
        (262_144, 16),
        (32_768, 3),
        (1024, 2),
        (1025, 5),
        (5003, 11),
    ];
    for (i, &(vocab, rows)) in argmax_cases.iter().enumerate() {
        for ties in [false, true] {
            let logits = if ties {
                tie_rows(vocab, rows, 70 + i as u64)
            } else {
                noise(vocab * rows, 90 + i as u64)
            };
            let ctx = Context::new(8, 8);
            let l = ctx
                .new_tensor(types::F32, &[vocab as i64, rows as i64])
                .as_input();
            let ggml = ctx.argmax(l).as_output();
            let cl = ctx
                .external(&ARGMAX, types::I32, [rows as i64, 1, 1, 1], &[l])
                .as_output();
            let mut g = ctx.new_graph(8);
            g.expand(ggml);
            g.expand(cl);
            ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            l.set_f32(&logits);
            gpu.compute(&g).unwrap();
            let want: Vec<i32> = logits.chunks(vocab).map(ggml_cuda_argmax).collect();
            assert_eq!(
                ggml.to_i32(),
                want,
                "argmax {vocab} x {rows}: model of ggml"
            );
            assert_eq!(cl.to_i32(), want, "argmax {vocab} x {rows}, ties {ties}");
        }
    }
    // An unaligned row start (scalar loads): a view one float in.
    let (vocab, rows) = (40_000usize, 3usize);
    let logits = tie_rows(vocab * rows + 1, 1, 99);
    let ctx = Context::new(8, 8);
    let l = ctx
        .new_tensor(types::F32, &[(vocab * rows + 1) as i64])
        .as_input();
    let v = ctx.view_2d(l, [vocab as i64, rows as i64], vocab * 4, 4);
    let cl = ctx
        .external(&ARGMAX, types::I32, [rows as i64, 1, 1, 1], &[v])
        .as_output();
    let mut g = ctx.new_graph(8);
    g.expand(cl);
    ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    l.set_f32(&logits);
    gpu.compute(&g).unwrap();
    let want: Vec<i32> = logits[1..].chunks(vocab).map(ggml_cuda_argmax).collect();
    assert_eq!(cl.to_i32(), want, "unaligned rows");
}

/// The argmax op on any GPU (split over CTAs and joined without clusters, the cluster kernel
/// on compute capability 9.0+) against ggml-cuda's argmax, ties included, aligned and unaligned
/// rows.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_split_argmax_matches_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let cases = [
        (248_320usize, 1usize, 0usize),
        (248_320, 3, 0),
        (248_320, 16, 0),
        (262_144, 2, 0),
        (32_768, 3, 0),
        (1024, 2, 0),
        (1025, 5, 0),
        (5003, 11, 0),
        (40_000, 3, 1),
        (248_320, 3, 1),
    ];
    for (i, &(vocab, rows, off)) in cases.iter().enumerate() {
        for ties in [false, true] {
            let total = vocab * rows + off;
            let logits = if ties {
                tie_rows(total, 1, 170 + i as u64)
            } else {
                noise(total, 190 + i as u64)
            };
            let ctx = Context::new(8, 8);
            let l = ctx.new_tensor(types::F32, &[total as i64]).as_input();
            let v = ctx.view_2d(l, [vocab as i64, rows as i64], vocab * 4, off * 4);
            let am = ctx
                .external(&ARGMAX, types::I32, [rows as i64, 1, 1, 1], &[v])
                .as_output();
            let mut g = ctx.new_graph(8);
            g.expand(am);
            ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            l.set_f32(&logits);
            gpu.compute(&g).unwrap();
            let want: Vec<i32> = logits[off..].chunks(vocab).map(ggml_cuda_argmax).collect();
            assert_eq!(
                am.to_i32(),
                want,
                "split argmax {vocab} x {rows}, offset {off}, ties {ties}"
            );
        }
    }
}

/// Times per op, old kernel and cluster kernel, `REPS` copies of the op in one graph (kernel
/// times: run under `nsys profile --stats=true`). Top-k at 4K, 32K and 100K tokens of context
/// (pools of 4) for 1-16 rows; argmax over Qwen3.8-Flash-Next's vocabulary for 1-32 rows.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; timing"]
fn cuda_times() {
    const REPS: usize = 64;
    let Some(mut gpu) = gpu() else {
        return;
    };
    if let Some((size, active)) = cluster_limits() {
        eprintln!(
            "cluster limits: max cluster size {size}, active clusters of {CLUSTER}: {active}"
        );
    }
    let mut run = |n: usize, rows: usize, op: &str| -> f64 {
        let ctx = Context::new(4 * REPS, 4 * REPS);
        let x = ctx
            .new_tensor(types::F32, &[n as i64, rows as i64])
            .as_input();
        let mut g = ctx.new_graph(4 * REPS);
        for _ in 0..REPS {
            let o = match op {
                "topk" => crate::model::qwen4::qsa_attn::top_k(&ctx, x, 512),
                "topk_cluster" => top_k(&ctx, x, 512),
                "argmax" => ctx.argmax(x),
                _ => ctx.external(&ARGMAX, types::I32, [rows as i64, 1, 1, 1], &[x]),
            };
            g.expand(o.as_output());
        }
        ctx.alloc_graph(&g, ComputeAllocator::new(gpu.buffer_type()))
            .unwrap();
        x.set_f32(&topk_scores(n, rows, 0, 7));
        gpu.compute(&g).unwrap();
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gpu.compute(&g).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e6 / (iters * REPS) as f64
    };
    for n in [1024usize, 2048, 4096, 8192, 25000, 100000] {
        for rows in [1usize, 2, 4, 8, 12, 16] {
            let (a, b) = (run(n, rows, "topk"), run(n, rows, "topk_cluster"));
            eprintln!(
                "topk n {n} rows {rows}: qsa_topk {a:.2} us, cluster {b:.2} us (wall per op)"
            );
        }
    }
    for rows in [1usize, 4, 8, 16, 32] {
        let (a, b) = (
            run(248_320, rows, "argmax"),
            run(248_320, rows, "argmax_cluster"),
        );
        eprintln!("argmax 248320 rows {rows}: ggml {a:.2} us, cluster {b:.2} us (wall per op)");
    }
}
