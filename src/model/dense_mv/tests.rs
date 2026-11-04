//! The dense matmul op: its shape checks, the CPU reference against F64 sums, and on CUDA
//! (ignored test) the same bits as ggml-cuda's own `mul_mat` for 2-16 tokens.

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

/// The bytes of `n` noise values in `ty` (BF16: truncated F32s, or F32), and the values they hold.
fn weight_noise(ty: ffi::ggml_type, n: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    if ty == types::F32 {
        let vals = noise(n, seed);
        return (vals.iter().flat_map(|v| v.to_ne_bytes()).collect(), vals);
    }
    bf16_noise(n, seed)
}

/// BF16 bytes of `n` noise values (truncated F32s), and the values they hold.
fn bf16_noise(n: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let bits: Vec<u16> = noise(n, seed)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let vals = bits
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    (bits.iter().flat_map(|b| b.to_ne_bytes()).collect(), vals)
}

#[test]
fn fits_takes_ggml_kernel_shapes() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::BF16, &[2560, 6144]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in MIN_TOKENS..=MAX_TOKENS {
        assert!(fits(w, x(n)), "{n} tokens");
    }
    assert!(!fits(w, x(1)), "one token stays on ggml");
    assert!(!fits(w, x(17)), "past mul_mat_f");
    // mul_mat_f's tiles: rows a multiple of 32, k of 64.
    assert!(!fits(ctx.new_tensor(types::BF16, &[2560, 6100]), x(2)));
    let short_k = ctx.new_tensor(types::BF16, &[2528, 64]);
    assert!(!fits(short_k, ctx.new_tensor(types::F32, &[2528, 4])));
    // Not BF16 weights, F32 inputs, matrices or 2-D inputs.
    assert!(!fits(ctx.new_tensor(types::F16, &[2560, 64]), x(2)));
    assert!(!fits(w, ctx.new_tensor(types::BF16, &[2560, 2])));
    assert!(!fits(ctx.new_tensor(types::BF16, &[2560, 64, 2]), x(2)));
    assert!(!fits(w, ctx.new_tensor(types::F32, &[2560, 2, 2])));
}

/// F32 weights (the MoE router) fit from 4 tokens: ggml-cuda keeps 1-3 on `mul_mat_vec_f`.
#[test]
fn fits_f32_weights_from_four_tokens() {
    let ctx = Context::new(64, 0);
    let router = ctx.new_tensor(types::F32, &[2560, 512]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in MIN_TOKENS_F32..=MAX_TOKENS {
        assert!(fits(router, x(n)), "{n} tokens");
    }
    for n in [1, 2, 3, 17] {
        assert!(!fits(router, x(n)), "{n} tokens");
    }
    assert!(!fits(ctx.new_tensor(types::F32, &[2560, 500]), x(16)));
    assert!(!fits(ctx.new_tensor(types::F32, &[2528, 512]), x(16)));
    // One op takes one weight type.
    let bf16 = ctx.new_tensor(types::BF16, &[2560, 6144]);
    assert!(!takes_all(&[bf16, router], x(16)));
    assert!(!takes_all(&[router], x(16)), "off by default");
    if !cfg!(feature = "cuda") {
        set_f32(true);
        assert!(!takes_all(&[router], x(16)), "the op needs CUDA");
        set_f32(false);
    }
}

/// A GDN layer's alpha and beta (48 BF16 rows each, which ggml runs on cuBLAS from 9 tokens) fit
/// as chain weights at 9-16 tokens, only after weights that fit the op themselves.
#[test]
fn chain_weights_ride_along_from_nine_tokens() {
    let ctx = Context::new(64, 0);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    let qkv = ctx.new_tensor(types::BF16, &[2560, 10240]);
    let z = ctx.new_tensor(types::BF16, &[2560, 6144]);
    let ab = ctx.new_tensor(types::BF16, &[2560, 48]);
    for n in CHAIN_MIN_TOKENS..=MAX_TOKENS {
        assert!(fits_chain(ab, x(n)), "{n} tokens");
        assert_eq!(chain_count(&[qkv, z, ab, ab], x(n)), Some(2));
    }
    for n in [2, 8, 17] {
        assert!(
            !fits_chain(ab, x(n)),
            "{n} tokens: not on cuBLAS in ggml, or past the op"
        );
    }
    assert!(!fits(ab, x(16)), "not a full 32-row tile");
    assert_eq!(chain_count(&[qkv, z], x(16)), Some(0));
    assert_eq!(chain_count(&[ab, qkv], x(16)), None, "chain weights last");
    assert_eq!(chain_count(&[ab], x(16)), None, "not alone");
    assert_eq!(chain_count(&[qkv, ab], x(8)), None);
    assert!(fits_chain(ctx.new_tensor(types::BF16, &[2560, 16]), x(9)));
    // Rows a multiple of 32 are mul_mat_f's; past 48 the cuBLAS kernel was not checked.
    for m in [32, 64, 80, 40] {
        assert!(
            !fits_chain(ctx.new_tensor(types::BF16, &[2560, m]), x(16)),
            "{m}"
        );
    }
    assert!(!fits_chain(ctx.new_tensor(types::F32, &[2560, 48]), x(16)));
    assert!(!fits_chain(ctx.new_tensor(types::BF16, &[2528, 48]), x(16)));
    if !cfg!(feature = "cuda") {
        assert!(!takes_all(&[qkv, z, ab, ab], x(16)), "the op needs CUDA");
    }
}

/// A shared expert's gate and up (2560 x 640 each, ~6.6 MB together) stay under `takes_all`'s
/// size floor; only `takes_small` takes them, and only where `small` says so.
#[test]
fn small_weights_only_through_takes_small() {
    let ctx = Context::new(64, 0);
    let x = ctx.new_tensor(types::F32, &[2560, 16]);
    let gate = ctx.new_tensor(types::BF16, &[2560, 640]);
    let up = ctx.new_tensor(types::BF16, &[2560, 640]);
    assert!(fits(gate, x) && fits(up, x));
    assert!(gate.nbytes() + up.nbytes() < MIN_BYTES);
    assert!(!takes_all(&[gate, up], x));
    assert!(!takes(gate, x));
    // `small` follows the stage graphs unless `--dense-mv-small` or `--no-dense-mv-small` set
    // it (#836): the op costs no graph split inside a stage graph. They stay off in the unit
    // tests (only `legion serve` turns them on).
    assert!(!crate::ggml::stage_graph::enabled());
    assert_eq!(SMALL.load(Ordering::Relaxed), SMALL_AUTO, "the default");
    assert!(!small(), "no stage graphs");
    assert!(!takes_small(&[gate, up], x));
    set_small(true);
    assert!(small());
    if !cfg!(feature = "cuda") {
        assert!(!takes_small(&[gate, up], x), "the op needs CUDA");
    }
    set_small(false);
    assert!(!small());
    SMALL.store(SMALL_AUTO, Ordering::Relaxed);
}

/// The op's results (one per weight of shape `[k, m_i]`) and the F64 sums, for `n` tokens.
fn op_case(
    backend: &mut Backend,
    ty: ffi::ggml_type,
    k: i64,
    ms: &[i64],
    n: i64,
) -> Vec<(Vec<f32>, Vec<f64>)> {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = ms
        .iter()
        .map(|&m| ctx.new_tensor(ty, &[k, m]).as_input())
        .collect();
    let outs: Vec<_> = mul_mat(&ctx, &ws, x)
        .into_iter()
        .map(|o| ctx.cont(o).as_output())
        .collect();
    let mut graph = ctx.new_graph(32);
    for &o in &outs {
        graph.expand(o);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let xv = noise((k * n) as usize, 7);
    x.set_f32(&xv);
    let mut res = Vec::new();
    let mut wvs = Vec::new();
    for (i, (&w, &m)) in ws.iter().zip(ms).enumerate() {
        let (bytes, vals) = weight_noise(ty, (k * m) as usize, 100 + i as u64);
        w.set_bytes(&bytes);
        wvs.push(vals);
    }
    backend.compute(&graph).unwrap();
    for ((o, &m), wv) in outs.iter().zip(ms).zip(&wvs) {
        let mut want = vec![0f64; (m * n) as usize];
        for t in 0..n as usize {
            for r in 0..m as usize {
                want[t * m as usize + r] = (0..k as usize)
                    .map(|c| f64::from(wv[r * k as usize + c]) * f64::from(xv[t * k as usize + c]))
                    .sum();
            }
        }
        res.push((o.to_f32(), want));
    }
    res
}

#[test]
fn cpu_matches_f64_sums() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (ty, k, ms, n) in [
        (types::BF16, 256, vec![64], 2),
        (types::BF16, 128, vec![32, 96], 3),
        (types::BF16, 64, vec![32, 64, 32, 64], 16),
        (types::F32, 256, vec![64], 4),
        (types::F32, 128, vec![32, 96], 16),
    ] {
        for (i, (got, want)) in op_case(&mut backend, ty, k, &ms, n).iter().enumerate() {
            for (a, b) in got.iter().zip(want) {
                assert!(
                    (f64::from(*a) - b).abs() <= 1e-4 * (1.0 + b.abs()),
                    "k {k} weight {i}: {a} vs {b}"
                );
            }
        }
    }
}

/// The kernel variants [`force_variant`] takes; `None` is the default choice.
#[cfg(feature = "cuda")]
const VARIANTS: [Option<(i32, i32)>; 6] = [
    None,
    Some((1, 2)),
    Some((1, 4)),
    Some((1, 8)),
    Some((2, 2)),
    Some((2, 4)),
];

/// On a GPU where the op runs, `mul_mat` gives the bits of ggml-cuda's `mul_mat` (its
/// `mul_mat_f` kernel) for 2-16 tokens (F32 weights: 4-16) and the shapes of a decode step, one weight per op and
/// several, in every kernel variant.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_matches_ggml_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        eprintln!("dense_mv: not supported on this GPU, skipped");
        return;
    }
    // (k, rows of each weight): Qwen3.8-Flash-Next's decode matmuls, a GLU-sized one and short
    // rows that change ggml's warp count.
    // F32 weights: the MoE router and short rows.
    // Chain weights (48 or 16 rows, last): cuBLAS's bits, from 9 tokens.
    let shapes: [(ffi::ggml_type, i64, &[i64]); 16] = [
        (types::BF16, 2560, &[10240, 6144, 48, 48]),
        (types::BF16, 2560, &[512, 16]),
        (types::F32, 2560, &[512]),
        (types::F32, 2560, &[512, 64]),
        (types::F32, 192, &[96]),
        (types::BF16, 2560, &[10240]),
        (types::BF16, 2560, &[6144]),
        (types::BF16, 6144, &[2560]),
        (types::BF16, 2560, &[12288]),
        (types::BF16, 2560, &[512, 512]),
        (types::BF16, 2560, &[640, 640]),
        (types::BF16, 2560, &[12288, 512, 512]),
        (types::BF16, 10240, &[320]),
        (types::BF16, 320, &[10240]),
        (types::BF16, 2560, &[10240, 6144]),
        (types::BF16, 192, &[64, 32, 96]),
    ];
    for (ty, k, ms) in shapes {
        let min_n = if ty == types::F32 {
            MIN_TOKENS_F32
        } else if ms.iter().any(|m| m % 32 != 0) {
            CHAIN_MIN_TOKENS
        } else {
            MIN_TOKENS
        };
        for n in min_n..=MAX_TOKENS {
            let ctx = Context::new(32, 32);
            let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
            let ws: Vec<_> = ms
                .iter()
                .map(|&m| ctx.new_tensor(ty, &[k, m]).as_input())
                .collect();
            let outs: Vec<_> = mul_mat(&ctx, &ws, x)
                .into_iter()
                .map(|o| ctx.cont(o).as_output())
                .collect();
            let refs: Vec<_> = ws.iter().map(|&w| ctx.mul_mat(w, x).as_output()).collect();
            let mut graph = ctx.new_graph(32);
            for &t in outs.iter().chain(&refs) {
                graph.expand(t);
            }
            ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
                .unwrap();
            x.set_f32(&noise((k * n) as usize, 1000 + n as u64));
            for (i, (&w, &m)) in ws.iter().zip(ms).enumerate() {
                w.set_bytes(&weight_noise(ty, (k * m) as usize, 2000 + i as u64).0);
            }
            for var in VARIANTS {
                assert!(force_variant(var), "variant {var:?}");
                let r = gpu.compute(&graph);
                force_variant(None);
                r.unwrap();
                for (i, (o, r)) in outs.iter().zip(&refs).enumerate() {
                    let (a, b) = (o.to_f32(), r.to_f32());
                    let same = a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits());
                    assert!(
                        same,
                        "{ty:?} k {k} weight {i} ({} rows), {n} tokens, variant {var:?}",
                        ms[i]
                    );
                }
            }
        }
    }
}

/// A router's BF16 copy (#836) fits `mul_mat_wide` where its F32 matrix fits the F32 op: 4-16
/// tokens; `fits_wide` takes BF16 weights only.
#[test]
fn fits_wide_takes_router_copies_from_four_tokens() {
    let ctx = Context::new(64, 0);
    let copy = ctx.new_tensor(types::BF16, &[2560, 512]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in MIN_TOKENS_F32..=MAX_TOKENS {
        assert!(fits_wide(copy, x(n)), "{n} tokens");
    }
    for n in [1, 2, 3, 17] {
        assert!(!fits_wide(copy, x(n)), "{n} tokens");
    }
    assert!(!fits_wide(ctx.new_tensor(types::F32, &[2560, 512]), x(8)));
    assert!(!fits_wide(ctx.new_tensor(types::BF16, &[2560, 500]), x(8)));
    if !cfg!(feature = "cuda") {
        assert!(!takes_wide(copy, x(8)), "the op needs CUDA");
    }
}

/// `--dense-mv-wide-chain` (#836) covers 17-48 tokens of BF16 weights whose rows are a multiple
/// of 16 and k of 64; it stays off unless switched on.
#[test]
fn fits_wide_chain_from_seventeen_tokens() {
    let ctx = Context::new(64, 0);
    let w = ctx.new_tensor(types::BF16, &[2560, 6160]);
    let x = |n: i64| ctx.new_tensor(types::F32, &[2560, n]);
    for n in MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS {
        assert!(fits_wide_chain(w, x(n)), "{n} tokens");
    }
    for n in [2, MAX_TOKENS, WIDE_CHAIN_MAX_TOKENS + 1] {
        assert!(!fits_wide_chain(w, x(n)), "{n} tokens");
    }
    // Rows a multiple of 16, k of 64; BF16 weights, F32 inputs.
    let odd_rows = ctx.new_tensor(types::BF16, &[2560, 6100]);
    assert!(!fits_wide_chain(odd_rows, x(36)));
    let short_k = ctx.new_tensor(types::BF16, &[2500, 6160]);
    let short_x = ctx.new_tensor(types::F32, &[2500, 36]);
    assert!(!fits_wide_chain(short_k, short_x));
    let f32_w = ctx.new_tensor(types::F32, &[2560, 6160]);
    assert!(!fits_wide_chain(f32_w, x(36)));
    let bf16_x = ctx.new_tensor(types::BF16, &[2560, 36]);
    assert!(!fits_wide_chain(w, bf16_x));
    assert!(!takes_wide_chain(w, x(36)), "no GPU tensor");
}

/// The default wide chains (#836): shapes the op takes (16 MiB and up, `k` a multiple of 64, `m`
/// of 16), no step size of a shape listed twice, slices of whole k steps that split `k`, and
/// nothing outside 17-48 tokens.
#[test]
fn wide_chain_bitwise_table_fits() {
    for (i, &(k, m, steps, kslice)) in WIDE_CHAIN_BITWISE.iter().enumerate() {
        assert!(k % 64 == 0 && m % 16 == 0, "{k}:{m}");
        assert!((k * m * 2) as usize >= MIN_BYTES, "{k}:{m} under MIN_BYTES");
        assert_ne!(steps, 0, "{k}:{m} without steps");
        assert!(
            kslice == 0 || (kslice % 16 == 0 && (kslice as i64) < k),
            "{k}:{m} slices of {kslice}"
        );
        assert!(
            WIDE_CHAIN_BITWISE[..i]
                .iter()
                .all(|&(pk, pm, ps, _)| (pk, pm) != (k, m) || ps & steps == 0),
            "{k}:{m} steps twice"
        );
        for n in MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS {
            if steps >> (n - MAX_TOKENS - 1) & 1 != 0 {
                assert_eq!(
                    wide_chain_order(k, m, n).map(|o| o.kslice),
                    Some(kslice),
                    "{k}:{m} at {n}"
                );
            }
        }
        for n in [1, 2, MAX_TOKENS, WIDE_CHAIN_MAX_TOKENS + 1, 64] {
            assert!(!wide_chain_bitwise(k, m, n), "{k}:{m} at {n}");
        }
    }
    assert!(!wide_chain_bitwise(2560, 1000, 36), "unlisted shape");
    // The GB10 check: DeltaNet qkv at every step in one chain, the output projections at 26-32
    // tokens in one chain and at 33-48 in cuBLAS's 8 slices of 768, not at 17-25.
    assert!(wide_chain_bitwise(2560, 10240, 17) && wide_chain_bitwise(2560, 10240, 48));
    assert_eq!(
        wide_chain_order(2560, 10240, 36),
        Some(WideOrder::default())
    );
    for n in MAX_TOKENS + 1..=WIDE_CHAIN_MAX_TOKENS {
        let kslice = match n {
            26..=32 => Some(0),
            33..=48 => Some(768),
            _ => None,
        };
        assert_eq!(
            wide_chain_order(6144, 2560, n).map(|o| o.kslice),
            kslice,
            "{n}"
        );
        assert_eq!(wide_chain_bitwise(6144, 2560, n), kslice.is_some(), "{n}");
    }
}

/// `mul_mat_wide(copy, x)` and the F32 matmul of the values the copy holds, on `backend`
/// (`f32_ref`: the F32 op, else ggml's `mul_mat`): the results' bits.
fn wide_case(backend: &mut Backend, k: i64, m: i64, n: i64, f32_ref: bool) -> (Vec<u32>, Vec<u32>) {
    let ctx = Context::new(32, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let wf = ctx.new_tensor(types::F32, &[k, m]).as_input();
    let wb = ctx.new_tensor(types::BF16, &[k, m]).as_input();
    let out = mul_mat_wide(&ctx, wb, x).as_output();
    let r = if f32_ref {
        mul_mat(&ctx, &[wf], x)[0]
    } else {
        ctx.mul_mat(wf, x)
    };
    let r = r.as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    graph.expand(r);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((k * n) as usize, 3000 + n as u64));
    let (bytes, vals) = bf16_noise((k * m) as usize, 4000);
    wb.set_bytes(&bytes);
    wf.set_f32(&vals);
    backend.compute(&graph).unwrap();
    let bits = |t: Tensor<'_>| -> Vec<u32> { t.to_f32().iter().map(|v| v.to_bits()).collect() };
    (bits(out), bits(r))
}

#[test]
fn cpu_wide_is_the_f32_op() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    for (k, m, n) in [(256, 64, 4), (128, 96, 16)] {
        let (a, b) = wide_case(&mut backend, k, m, n, true);
        assert_eq!(a, b, "k {k}, {m} rows, {n} tokens");
    }
}

/// On a GPU where the op runs, `mul_mat_wide` of a router's BF16 copy gives the bits of
/// ggml-cuda's F32 `mul_mat` (`mul_mat_f<float>`) of the F32 matrix, in every kernel variant.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_wide_matches_ggml_f32_bits() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        eprintln!("dense_mv: not supported on this GPU, skipped");
        return;
    }
    for (k, m) in [(2560, 512), (2048, 256), (192, 96)] {
        for n in MIN_TOKENS_F32..=MAX_TOKENS {
            for var in VARIANTS {
                assert!(force_variant(var), "variant {var:?}");
                let (a, b) = wide_case(&mut gpu, k, m, n, false);
                force_variant(None);
                assert_eq!(a, b, "k {k}, {m} rows, {n} tokens, variant {var:?}");
            }
        }
    }
}

/// Wide chains of several weights in one node (#836, a DeltaNet layer's qkv and z): each result
/// has the bits of that weight's own node, on `backend`.
fn wide_chains_case(backend: &mut Backend, k: i64, ms: &[i64], n: i64) {
    let ctx = Context::new(64, 32);
    let x = ctx.new_tensor(types::F32, &[k, n]).as_input();
    let ws: Vec<_> = ms
        .iter()
        .map(|&m| ctx.new_tensor(types::BF16, &[k, m]).as_input())
        .collect();
    let joint: Vec<_> = mul_mat_wide_chains(&ctx, &ws, x)
        .into_iter()
        .map(|t| ctx.cont(t).as_output())
        .collect();
    let each: Vec<_> = ws
        .iter()
        .map(|&w| mul_mat_wide_chain(&ctx, w, x).as_output())
        .collect();
    let mut graph = ctx.new_graph(64);
    for &t in joint.iter().chain(&each) {
        graph.expand(t);
    }
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    x.set_f32(&noise((k * n) as usize, 3100 + n as u64));
    for (i, (w, &m)) in ws.iter().zip(ms).enumerate() {
        w.set_bytes(&bf16_noise((k * m) as usize, 4100 + i as u64).0);
    }
    backend.compute(&graph).unwrap();
    let bits = |t: Tensor<'_>| -> Vec<u32> { t.to_f32().iter().map(|v| v.to_bits()).collect() };
    for (i, (&a, &b)) in joint.iter().zip(&each).enumerate() {
        assert_eq!(a.ne(), [ms[i], n, 1, 1], "weight {i} shape");
        assert_eq!(
            bits(a),
            bits(b),
            "k {k}, rows {ms:?}, {n} tokens, weight {i}"
        );
    }
}

#[test]
fn cpu_wide_chains_are_each_weights_chain() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    wide_chains_case(&mut backend, 128, &[96, 64], 20);
    wide_chains_case(&mut backend, 192, &[32, 48, 16], 36);
}

/// The joined wide chains take what each weight takes, in one order: none without a GPU tensor,
/// at most [`MAX_WEIGHTS`] weights.
#[test]
fn takes_wide_chains_needs_each_weight() {
    let ctx = Context::new(64, 0);
    let qkv = ctx.new_tensor(types::BF16, &[2560, 10240]);
    let z = ctx.new_tensor(types::BF16, &[2560, 6144]);
    let x = ctx.new_tensor(types::F32, &[2560, 36]);
    assert!(!takes_wide_chains(&[qkv, z], x), "no GPU tensor");
    assert!(!takes_wide_chains(&[], x));
    assert!(!takes_wide_chains(&[qkv; MAX_WEIGHTS + 1], x));
}

/// On a GPU where the op runs, joined wide chains give each weight's own launch's bits, also
/// for weights of other orders forced together and for row tiles that straddle two weights.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_wide_chains_are_each_weights_chain() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    if !device_supported() {
        eprintln!("dense_mv: not supported on this GPU, skipped");
        return;
    }
    for n in [17, 24, 36, 48] {
        wide_chains_case(&mut gpu, 2560, &[10240, 6144], n);
        wide_chains_case(&mut gpu, 256, &[80, 48, 16], n);
    }
}
