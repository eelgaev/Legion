//! [`norm_rope`] and [`pool_key`] against the ggml chains they replace: on the CPU (the
//! reference implementations, within rounding), on CUDA (ignored tests) bit for bit, and a
//! microbenchmark of both against the chains.

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

/// Max |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale
}

const EPS: f32 = 1e-6;

/// qwen3.5 / qwen4exp's RoPE (64 of 256 dims, sections 11, 11, 10), or plain MROPE.
fn rope(imrope: bool, n_dims: i32) -> Rope {
    Rope {
        n_dims,
        sections: [11, 11, 10, 0],
        mode: if imrope { ROPE_IMROPE } else { ROPE_MROPE },
        freq_base: 10_000_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
    }
}

fn chain_rope<'a>(ctx: &'a Context, t: Tensor<'a>, pos: Tensor<'a>, r: &Rope) -> Tensor<'a> {
    ctx.rope_multi(
        t,
        pos,
        r.n_dims,
        r.sections,
        r.mode,
        262_144,
        r.freq_base,
        r.freq_scale,
        r.ext_factor,
        r.attn_factor,
        32.0,
        1.0,
    )
}

/// Positions: per section row, token t at `base + t` (rows 1-3 offset a little, so each
/// section's row is seen).
fn positions(n: usize, base: i32) -> Vec<i32> {
    (0..4)
        .flat_map(|s| (0..n as i32).map(move |t| base + t + 3 * s))
        .collect()
}

/// (op, chain) of `x` `[d, h, n]` read as a view of a row with gates between the heads (as
/// the attention's q is), on `backend`.
fn norm_rope_case(
    backend: &mut Backend,
    d: i64,
    h: i64,
    n: i64,
    r: &Rope,
    base: i32,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = Context::new(32, 32);
    let qg = ctx.new_tensor(types::F32, &[2 * d * h, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[d]).as_input();
    let pos = ctx.new_tensor(types::I32, &[4 * n]).as_input();
    let x = ctx.view_3d(
        qg,
        [d, h, n],
        [2 * d as usize * 4, (2 * d * h) as usize * 4],
        0,
    );
    let op = norm_rope(&ctx, x, w, pos, EPS, r).as_output();
    let chain = chain_rope(&ctx, ctx.mul(ctx.rms_norm(x, EPS), w), pos, r).as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(chain);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    qg.set_f32(&noise((2 * d * h * n) as usize, 7));
    w.set_f32(
        &noise(d as usize, 8)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect::<Vec<_>>(),
    );
    pos.set_bytes(as_bytes(&positions(n as usize, base)));
    backend.compute(&graph).unwrap();
    (op.to_f32(), chain.to_f32())
}

fn as_bytes(v: &[i32]) -> &[u8] {
    // SAFETY: plain integers.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 4) }
}

fn as_bytes64(v: &[i64]) -> &[u8] {
    // SAFETY: plain integers.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), v.len() * 8) }
}

/// (op cache, chain cache) after [`pool_key`] and qsa_top's chain (`get_rows`, `cont`, `add`s,
/// `scale`, `rms_norm`, `mul`, `rope_multi`, `set_rows`) wrote `n_new` pool keys of `kpool`
/// members each into two copies of one F16 indexer cache `[2 d, rows]`.
fn pool_key_case(
    backend: &mut Backend,
    d: i64,
    kpool: i64,
    n_new: i64,
    r: &Rope,
) -> (Vec<f32>, Vec<f32>) {
    // Room for every pool's own key row (distinct `rep`s, as qsa_top's are).
    let rows = 256i64;
    let ctx = Context::new(64, 64);
    // Outputs too: the allocator must not hand their memory to the chain's intermediates.
    let caches = [0, 1].map(|_| {
        ctx.new_tensor(types::F16, &[2 * d, rows])
            .as_input()
            .as_output()
    });
    let members = ctx.new_tensor(types::I32, &[kpool * n_new]).as_input();
    let pos = ctx.new_tensor(types::I32, &[4 * n_new]).as_input();
    let rep = ctx.new_tensor(types::I64, &[n_new]).as_input();
    let w = ctx.new_tensor(types::F32, &[d]).as_input();
    fn halves<'a>(ctx: &'a Context, t: Tensor<'a>, d: i64, rows: i64) -> (Tensor<'a>, Tensor<'a>) {
        let nb1 = t.nb()[1];
        (
            ctx.view_2d(t, [d, rows], nb1, 0),
            ctx.view_2d(t, [d, rows], nb1, d as usize * 2),
        )
    }
    let mut graph = ctx.new_graph(64);
    let (raw, pooled) = halves(&ctx, caches[0], d, rows);
    graph.expand(pool_key(&ctx, raw, pooled, members, pos, rep, w, EPS, r));
    let (raw, pooled) = halves(&ctx, caches[1], d, rows);
    let m = ctx.reshape_3d(ctx.get_rows(raw, members), d, kpool, n_new);
    let mnb = m.nb();
    let slice = |i: i64| ctx.view_2d(m, [d, n_new], mnb[2], i as usize * mnb[1]);
    let mut sum = ctx.cont(slice(0));
    for i in 1..kpool {
        sum = ctx.add(sum, slice(i));
    }
    let pk = ctx.scale(sum, 1.0 / kpool as f32);
    let pk = ctx.mul(ctx.rms_norm(pk, EPS), w);
    let pk = ctx.reshape_3d(pk, d, 1, n_new);
    let pk = ctx.reshape_2d(chain_rope(&ctx, pk, pos, r), d, n_new);
    graph.expand(ctx.set_rows(pooled, pk, rep));
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let init: Vec<u16> = noise((2 * d * rows) as usize, 9)
        .iter()
        .map(|&v| unsafe { ffi::ggml_fp32_to_fp16(v) })
        .collect();
    for c in caches {
        // SAFETY: plain integers.
        c.set_bytes(unsafe { std::slice::from_raw_parts(init.as_ptr().cast(), init.len() * 2) });
    }
    // Pool j: members scattered over the rows (any may repeat), its key in row
    // j * kpool + kpool - 1.
    let mem: Vec<i32> = (0..(kpool * n_new) as i32)
        .map(|i| (i * 5) % rows as i32)
        .collect();
    members.set_bytes(as_bytes(&mem));
    let reps: Vec<i64> = (0..n_new).map(|j| (j * kpool + kpool - 1) % rows).collect();
    rep.set_bytes(as_bytes64(&reps));
    pos.set_bytes(as_bytes(&positions(n_new as usize, 4093)));
    w.set_f32(
        &noise(d as usize, 10)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect::<Vec<_>>(),
    );
    backend.compute(&graph).unwrap();
    let f = |t: Tensor<'_>| -> Vec<f32> {
        t.bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes(b)) })
            .collect()
    };
    (f(caches[0]), f(caches[1]))
}

#[test]
fn cpu_norm_rope_matches_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (d, h, n, imrope, n_dims) in [
        (256, 4, 3, true, 64),
        (128, 2, 5, true, 128),
        (64, 3, 2, false, 32),
    ] {
        let (op, chain) = norm_rope_case(&mut cpu, d, h, n, &rope(imrope, n_dims), 17);
        let e = rel(&op, &chain);
        assert!(e < 1e-4, "{d}x{h}x{n}: {e:e}");
    }
}

#[test]
fn cpu_pool_key_matches_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    for (d, kpool, n_new) in [(128, 4, 3), (64, 2, 1)] {
        let (op, chain) = pool_key_case(&mut cpu, d, kpool, n_new, &rope(true, d as i32));
        let e = rel(&op, &chain);
        assert!(e < 2e-3, "{d} kpool {kpool}: {e:e}");
    }
}

#[test]
fn ropes_the_kernels_do_not_take_do_not_fit() {
    let r = rope(true, 64);
    assert!(r.fits(256));
    assert!(!r.fits(32));
    for bad in [
        Rope { mode: 2, ..r },
        Rope {
            freq_scale: 0.5,
            ..r
        },
        Rope {
            ext_factor: 1.0,
            ..r
        },
        Rope {
            attn_factor: 2.0,
            ..r
        },
        Rope { n_dims: 63, ..r },
        Rope {
            sections: [0; 4],
            ..r
        },
    ] {
        assert!(!bad.fits(256), "{bad:?}");
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

#[cfg(feature = "cuda")]
fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// qwen4exp's shapes (q 16 x 256, k 2 x 256, indexer q 128-wide) and others, positions near
/// 0 and 200K: bit for bit ggml-cuda's chain.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_norm_rope_is_bit_identical_to_ggml() {
    let mut gpu = gpu();
    for (d, h, n, imrope, n_dims) in [
        (256, 16, 3, true, 64),
        (256, 2, 1, true, 64),
        (128, 32, 3, true, 128),
        (128, 8, 48, true, 64),
        (64, 3, 2, false, 32),
        (512, 4, 2, true, 64),
    ] {
        for base in [0, 199_999] {
            let (op, chain) = norm_rope_case(&mut gpu, d, h, n, &rope(imrope, n_dims), base);
            assert!(
                same_bits(&op, &chain),
                "{d}x{h}x{n} imrope {imrope} pos {base}: rel {:e}",
                rel(&op, &chain)
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_pool_key_is_bit_identical_to_ggml() {
    let mut gpu = gpu();
    for (d, kpool, n_new) in [(128, 4, 1), (128, 4, 3), (128, 8, 16), (64, 2, 5)] {
        let (op, chain) = pool_key_case(&mut gpu, d, kpool, n_new, &rope(true, d as i32));
        assert!(
            same_bits(&op, &chain),
            "{d} kpool {kpool} x {n_new}: rel {:e}",
            rel(&op, &chain)
        );
    }
}

/// Copies of an op per benchmark graph.
#[cfg(feature = "cuda")]
const LAYERS: usize = 12;

/// Time per copy of `graph`'s `LAYERS` copies of an op or chain.
#[cfg(feature = "cuda")]
/// The benchmarks' inputs are outputs too: the allocator frees an input after its last reader
/// and hands its memory to later nodes, so a rerun would read their leftovers (as indexes).
fn time(gpu: &mut Backend, graph: &crate::ggml::Graph<'_>) -> f64 {
    const REPS: usize = 200;
    gpu.compute(graph).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        gpu.compute(graph).unwrap();
    }
    t0.elapsed().as_secs_f64() * 1e6 / (REPS * LAYERS) as f64
}

/// Time per call of [`norm_rope`] and the chain, `LAYERS` copies in one graph:
/// `cargo test --release --features cuda norm_rope::tests::cuda_bench -- --ignored --nocapture`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_bench_norm_rope() {
    let mut gpu = gpu();
    let r = rope(true, 64);
    for (what, d, h, n) in [
        ("q", 256, 16, 3),
        ("k", 256, 2, 3),
        ("iq", 128, 32, 3),
        ("q", 256, 16, 48),
    ] {
        let mut us = [0.0; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let ctx = Context::new(16 * LAYERS + 8, 16 * LAYERS + 8);
            let x = ctx
                .new_tensor(types::F32, &[d, h, n])
                .as_input()
                .as_output();
            let w = ctx.new_tensor(types::F32, &[d]).as_input().as_output();
            let pos = ctx.new_tensor(types::I32, &[4 * n]).as_input().as_output();
            let mut graph = ctx.new_graph(16 * LAYERS + 8);
            for _ in 0..LAYERS {
                let out = if fused {
                    norm_rope(&ctx, x, w, pos, EPS, &r)
                } else {
                    chain_rope(&ctx, ctx.mul(ctx.rms_norm(x, EPS), w), pos, &r)
                };
                graph.expand(out.as_output());
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_f32(&noise((d * h * n) as usize, 1));
            w.set_f32(&vec![1.0; d as usize]);
            pos.set_bytes(as_bytes(&positions(n as usize, 1000)));
            us[i] = time(&mut gpu, &graph);
        }
        println!(
            "norm_rope {what} {d}x{h}x{n}: chain {:.2} us, op {:.2} us",
            us[0], us[1]
        );
    }
}

/// [`pool_key`] against qsa_top's chain, as [`cuda_bench_norm_rope`].
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_bench_pool_key() {
    let mut gpu = gpu();
    let r = rope(true, 64);
    for (kpool, n_new) in [(4i64, 1i64), (4, 3), (4, 16)] {
        let d = 128i64;
        let rows = 1024i64;
        let mut us = [0.0; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let ctx = Context::new(32 * LAYERS + 8, 32 * LAYERS + 8);
            let cache = ctx
                .new_tensor(types::F16, &[2 * d, rows])
                .as_input()
                .as_output();
            let members = ctx
                .new_tensor(types::I32, &[kpool * n_new])
                .as_input()
                .as_output();
            let pos = ctx
                .new_tensor(types::I32, &[4 * n_new])
                .as_input()
                .as_output();
            let rep = ctx.new_tensor(types::I64, &[n_new]).as_input().as_output();
            let w = ctx.new_tensor(types::F32, &[d]).as_input().as_output();
            let nb1 = cache.nb()[1];
            let raw = ctx.view_2d(cache, [d, rows], nb1, 0);
            let pooled = ctx.view_2d(cache, [d, rows], nb1, d as usize * 2);
            let mut graph = ctx.new_graph(32 * LAYERS + 8);
            for _ in 0..LAYERS {
                if fused {
                    graph.expand(pool_key(&ctx, raw, pooled, members, pos, rep, w, EPS, &r));
                    continue;
                }
                let m = ctx.reshape_3d(ctx.get_rows(raw, members), d, kpool, n_new);
                let mnb = m.nb();
                let slice = |i: i64| ctx.view_2d(m, [d, n_new], mnb[2], i as usize * mnb[1]);
                let mut sum = ctx.cont(slice(0));
                for i in 1..kpool {
                    sum = ctx.add(sum, slice(i));
                }
                let pk = ctx.mul(ctx.rms_norm(ctx.scale(sum, 1.0 / kpool as f32), EPS), w);
                let pk = ctx.reshape_2d(
                    chain_rope(&ctx, ctx.reshape_3d(pk, d, 1, n_new), pos, &r),
                    d,
                    n_new,
                );
                graph.expand(ctx.set_rows(pooled, pk, rep));
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            let mem: Vec<i32> = (0..(kpool * n_new) as i32).collect();
            members.set_bytes(as_bytes(&mem));
            let reps: Vec<i64> = (0..n_new).map(|j| j * kpool + kpool - 1).collect();
            rep.set_bytes(as_bytes64(&reps));
            pos.set_bytes(as_bytes(&positions(n_new as usize, 1000)));
            w.set_f32(&vec![1.0; d as usize]);
            us[i] = time(&mut gpu, &graph);
        }
        println!(
            "pool_key kpool {kpool} x {n_new}: chain {:.2} us, op {:.2} us",
            us[0], us[1]
        );
    }
}

/// Reads an F16 tensor as F32.
fn f16s(t: Tensor<'_>) -> Vec<f32> {
    t.bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| unsafe { ffi::ggml_fp16_to_fp32(u16::from_le_bytes(b)) })
        .collect()
}

/// Two F16 caches `[row, rows]` (copies of the same noise) that the graph writes in place.
fn f16_caches<'a>(ctx: &'a Context, row: i64, rows: i64) -> [Tensor<'a>; 2] {
    // Outputs too: the allocator must not hand their memory to the chains' intermediates.
    [0, 1].map(|_| {
        ctx.new_tensor(types::F16, &[row, rows])
            .as_input()
            .as_output()
    })
}

fn fill_f16(t: Tensor<'_>, seed: u64) {
    let h: Vec<u8> = noise(t.elements(), seed)
        .iter()
        .flat_map(|&v| unsafe { ffi::ggml_fp32_to_fp16(v) }.to_le_bytes())
        .collect();
    t.set_bytes(&h);
}

/// Distinct cells of `n` tokens among `rows`.
fn cells(n: i64, rows: i64) -> Vec<i32> {
    (0..n).map(|t| ((t * 7 + 3) % rows) as i32).collect()
}

/// ([op K cache, op V cache], [chain K cache, chain V cache]) after [`kv_append`] and the chain
/// (`rms_norm`, `mul`, `rope_multi`, two `set_rows`) wrote `n` tokens' K and V, `k` read as a
/// view with gaps between its heads.
fn kv_append_case(backend: &mut Backend, d: i64, h: i64, n: i64, r: &Rope) -> [Vec<f32>; 4] {
    let rows = 64i64;
    let ctx = Context::new(64, 64);
    let kk = ctx.new_tensor(types::F32, &[2 * d * h, n]).as_input();
    let k = ctx.view_3d(
        kk,
        [d, h, n],
        [2 * d as usize * 4, (2 * d * h) as usize * 4],
        0,
    );
    let v = ctx.new_tensor(types::F32, &[d, h, n]).as_input();
    let w = ctx.new_tensor(types::F32, &[d]).as_input();
    let pos = ctx.new_tensor(types::I32, &[4 * n]).as_input();
    let cl = ctx.new_tensor(types::I32, &[n]).as_input();
    let [k0, k1] = f16_caches(&ctx, d * h, rows);
    let [v0, v1] = f16_caches(&ctx, d * h, rows);
    let mut graph = ctx.new_graph(64);
    graph.expand(kv_append(&ctx, k, w, pos, v, k0, v0, cl, EPS, r));
    let kr = chain_rope(&ctx, ctx.mul(ctx.rms_norm(k, EPS), w), pos, r);
    graph.expand(ctx.set_rows(k1, ctx.reshape_2d(kr, d * h, n), cl));
    graph.expand(ctx.set_rows(v1, ctx.reshape_2d(v, d * h, n), cl));
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    kk.set_f32(&noise((2 * d * h * n) as usize, 11));
    v.set_f32(&noise((d * h * n) as usize, 12));
    w.set_f32(
        &noise(d as usize, 13)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect::<Vec<_>>(),
    );
    pos.set_bytes(as_bytes(&positions(n as usize, 77_777)));
    cl.set_bytes(as_bytes(&cells(n, rows)));
    for (i, c) in [k0, k1, v0, v1].into_iter().enumerate() {
        fill_f16(c, 20 + (i / 2) as u64);
    }
    backend.compute(&graph).unwrap();
    [f16s(k0), f16s(v0), f16s(k1), f16s(v1)]
}

/// (op cache, chain cache) after [`idx_append`] and qsa_prepare's chain (`fill`, `concat`,
/// `set_rows`) wrote `n` tokens' indexer rows.
fn idx_append_case(backend: &mut Backend, d: i64, n: i64) -> (Vec<f32>, Vec<f32>) {
    let rows = 64i64;
    let ctx = Context::new(32, 32);
    let raw = ctx.new_tensor(types::F32, &[d, n]).as_input();
    let cl = ctx.new_tensor(types::I32, &[n]).as_input();
    let [c0, c1] = f16_caches(&ctx, 2 * d, rows);
    let mut graph = ctx.new_graph(32);
    graph.expand(idx_append(&ctx, raw, c0, cl));
    let zeros = ctx.fill(ctx.new_tensor(types::F32, &[d, n]), 0.0);
    graph.expand(ctx.set_rows(c1, ctx.concat(raw, zeros, 0), cl));
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    raw.set_f32(&noise((d * n) as usize, 14));
    cl.set_bytes(as_bytes(&cells(n, rows)));
    fill_f16(c0, 30);
    fill_f16(c1, 30);
    backend.compute(&graph).unwrap();
    (f16s(c0), f16s(c1))
}

#[test]
fn cpu_appends_match_ggml() {
    let mut cpu = Backend::cpu().unwrap();
    cpu.set_threads(4);
    let [k0, v0, k1, v1] = kv_append_case(&mut cpu, 256, 2, 3, &rope(true, 64));
    assert!(rel(&k0, &k1) < 2e-3, "K {:e}", rel(&k0, &k1));
    assert_eq!(v0, v1);
    let (a, b) = idx_append_case(&mut cpu, 128, 5);
    assert_eq!(a, b);
}

/// Bit for bit ggml-cuda's chains, qwen4exp's shapes (K and V 2 x 256, indexer 128) and
/// others.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_appends_are_bit_identical_to_ggml() {
    let mut gpu = gpu();
    for (d, h, n) in [(256, 2, 1), (256, 2, 3), (128, 8, 3), (256, 4, 48)] {
        let [k0, v0, k1, v1] = kv_append_case(&mut gpu, d, h, n, &rope(true, 64));
        assert!(same_bits(&k0, &k1), "K {d}x{h}x{n}: {:e}", rel(&k0, &k1));
        assert!(same_bits(&v0, &v1), "V {d}x{h}x{n}");
    }
    for (d, n) in [(128, 1), (128, 3), (128, 48), (64, 5)] {
        let (a, b) = idx_append_case(&mut gpu, d, n);
        assert!(same_bits(&a, &b), "idx {d}x{n}");
    }
}

/// [`kv_append`] and [`idx_append`] against their chains, as [`cuda_bench_norm_rope`].
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a benchmark"]
fn cuda_bench_appends() {
    let mut gpu = gpu();
    let r = rope(true, 64);
    let rows = 4096i64;
    for (d, h, n) in [(256i64, 2i64, 3i64), (256, 2, 1), (256, 2, 48)] {
        let mut us = [0.0; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let ctx = Context::new(16 * LAYERS + 16, 16 * LAYERS + 16);
            let k = ctx
                .new_tensor(types::F32, &[d, h, n])
                .as_input()
                .as_output();
            let v = ctx
                .new_tensor(types::F32, &[d, h, n])
                .as_input()
                .as_output();
            let w = ctx.new_tensor(types::F32, &[d]).as_input().as_output();
            let pos = ctx.new_tensor(types::I32, &[4 * n]).as_input().as_output();
            let cl = ctx.new_tensor(types::I32, &[n]).as_input().as_output();
            let [kc, vc] = f16_caches(&ctx, d * h, rows);
            let mut graph = ctx.new_graph(16 * LAYERS + 16);
            for _ in 0..LAYERS {
                if fused {
                    graph.expand(kv_append(&ctx, k, w, pos, v, kc, vc, cl, EPS, &r));
                    continue;
                }
                let kr = chain_rope(&ctx, ctx.mul(ctx.rms_norm(k, EPS), w), pos, &r);
                graph.expand(ctx.set_rows(kc, ctx.reshape_2d(kr, d * h, n), cl));
                graph.expand(ctx.set_rows(vc, ctx.reshape_2d(v, d * h, n), cl));
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            k.set_f32(&noise((d * h * n) as usize, 1));
            v.set_f32(&noise((d * h * n) as usize, 2));
            w.set_f32(&vec![1.0; d as usize]);
            pos.set_bytes(as_bytes(&positions(n as usize, 1000)));
            cl.set_bytes(as_bytes(&cells(n, rows)));
            us[i] = time(&mut gpu, &graph);
        }
        println!(
            "kv_append {d}x{h}x{n}: chain (norm, rope, 2 set_rows) {:.2} us, op {:.2} us",
            us[0], us[1]
        );
    }
    for n in [3i64, 1, 48] {
        let d = 128i64;
        let mut us = [0.0; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let ctx = Context::new(8 * LAYERS + 16, 8 * LAYERS + 16);
            let raw = ctx.new_tensor(types::F32, &[d, n]).as_input().as_output();
            let cl = ctx.new_tensor(types::I32, &[n]).as_input().as_output();
            let [c, _] = f16_caches(&ctx, 2 * d, rows);
            let mut graph = ctx.new_graph(8 * LAYERS + 16);
            for _ in 0..LAYERS {
                if fused {
                    graph.expand(idx_append(&ctx, raw, c, cl));
                    continue;
                }
                let zeros = ctx.fill(ctx.new_tensor(types::F32, &[d, n]), 0.0);
                graph.expand(ctx.set_rows(c, ctx.concat(raw, zeros, 0), cl));
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            raw.set_f32(&noise((d * n) as usize, 3));
            cl.set_bytes(as_bytes(&cells(n, rows)));
            us[i] = time(&mut gpu, &graph);
        }
        println!(
            "idx_append {d}x{n}: chain (fill, concat, set_rows) {:.2} us, op {:.2} us",
            us[0], us[1]
        );
    }
}
