//! The paged attention op (its CPU implementation) against the gather path it replaces: the
//! same blocks gathered with `KvPool::read`, then mul_mat / soft_max_ext / mul_mat.

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{Hparams, KvPool};

pub(super) fn hparams(head_dim: i64, n_head: i64, n_head_kv: i64) -> Hparams {
    Hparams {
        n_layer: 1,
        n_embd: head_dim * n_head,
        n_head,
        n_head_kv,
        head_dim,
        n_rot: head_dim as i32,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        // Layer 0 is an attention layer.
        full_attention_interval: 1,
        ssm_d_conv: 4,
        ssm_d_state: 16,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

/// Deterministic values in [-1, 1).
pub(super) fn noise(n: usize, seed: u64) -> Vec<f32> {
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

pub(super) fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let mut h = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::ggml_fp32_to_fp16_row(v.as_ptr(), h.as_mut_ptr(), v.len() as i64) };
    h.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

pub(super) struct Shape {
    pub(super) d: usize,
    pub(super) n_head: usize,
    pub(super) n_head_kv: usize,
    pub(super) n_tok: usize,
    /// Softmax scale.
    pub(super) scale: f64,
    /// Sliding window, 0 for none.
    pub(super) window: usize,
}

thread_local! {
    /// Whether [`oracle`] on this thread leaves the query and the softmax weights in full
    /// precision: for kernels that round neither (the FP32 FMA prefill, `ffma_tests.rs`).
    pub(super) static EXACT_ORACLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The op's result in f64, from the F16-rounded cache and query: scores and the weighted sum
/// accumulate exactly (as far as f32 output shows), so only the two F16 roundings the gather
/// path also makes (the query and the softmax weights) remain; none with [`EXACT_ORACLE`].
pub(super) fn oracle(
    sh: &Shape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_past: &[usize],
    tables: &[Vec<u32>],
) -> Vec<f32> {
    let Shape {
        d,
        n_head,
        n_head_kv,
        n_tok,
        scale,
        window,
    } = *sh;
    let g = n_head / n_head_kv;
    let kv_row = d * n_head_kv;
    let mut out = Vec::new();
    for (s, tab) in tables.iter().enumerate() {
        for t in 0..n_tok {
            for h in 0..n_head {
                let at = ((s * n_tok + t) * n_head + h) * d;
                let exact = EXACT_ORACLE.get();
                let mut qr = q[at..at + d].to_vec();
                if !exact {
                    round_f16(&mut qr);
                }
                let cell = |p: usize| {
                    (tab[p / KV_BLOCK] as usize * KV_BLOCK + p % KV_BLOCK) * kv_row + h / g * d
                };
                let visible = n_past[s] + t + 1;
                let first = if window > 0 {
                    visible.saturating_sub(window)
                } else {
                    0
                };
                let scores: Vec<f64> = (first..visible)
                    .map(|p| {
                        let dot: f64 = (0..d).map(|i| qr[i] as f64 * k[cell(p) + i] as f64).sum();
                        dot * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                let mut w: Vec<f32> = exp.iter().map(|x| (x / sum) as f32).collect();
                if !exact {
                    round_f16(&mut w);
                }
                let w: Vec<f64> = if exact {
                    exp.iter().map(|x| x / sum).collect()
                } else {
                    w.iter().map(|&x| x as f64).collect()
                };
                for i in 0..d {
                    let acc: f64 = (first..visible)
                        .zip(&w)
                        .map(|(p, &w)| w * v[cell(p) + i] as f64)
                        .sum();
                    out.push(acc as f32);
                }
            }
        }
    }
    out
}

/// `n_past` per sequence, `tables` per sequence (blocks in order), `n_tok` tokens each.
fn check(
    head_dim: i64,
    n_head: i64,
    n_head_kv: i64,
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    check_with(head_dim, n_head, n_head_kv, n_tok, n_past, tables, scale, 0);
}

/// [`check`] with softmax scale `scale` and sliding window `window` (0: none).
#[allow(clippy::too_many_arguments)]
fn check_with(
    head_dim: i64,
    n_head: i64,
    n_head_kv: i64,
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
    scale: f32,
    window: usize,
) {
    let hp = hparams(head_dim, n_head, n_head_kv);
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    let n_blocks = tables.iter().flatten().max().unwrap() + 1;
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks as usize).unwrap();
    let (kp, vp) = kv.pools(0);
    let (mut kdat, mut vdat) = (noise(kp.elements(), 1), noise(vp.elements(), 2));
    kp.set_bytes(&f16_bytes(&kdat));
    vp.set_bytes(&f16_bytes(&vdat));
    round_f16(&mut kdat);
    round_f16(&mut vdat);

    let ns = n_past.len();
    let (d, nt) = (head_dim, n_tok as i64);
    let end = n_past.iter().max().unwrap() + n_tok;
    let n_kv = end.div_ceil(KV_BLOCK) * KV_BLOCK;
    let width = tables.iter().map(Vec::len).max().unwrap();

    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(types::F32, &[d, n_head, nt, ns as i64])
        .as_input();
    let table = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch_ne = scratch_shape(d, n_head, n_head_kv, n_tok, ns);
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let paged = attention_with(&ctx, q, kp, vp, table, past, scratch, scale, window, 0).as_output();

    let blocks = ctx
        .new_tensor(types::I32, &[(n_kv / KV_BLOCK * ns) as i64])
        .as_input();
    let mask = ctx
        .new_tensor(types::F32, &[n_kv as i64, nt, 1, ns as i64])
        .as_input();
    let (k, vt) = kv.read(&ctx, 0, blocks, n_kv as i64, ns as i64);
    let kq = ctx.mul_mat(ctx.permute(k, [0, 2, 1, 3]), ctx.permute(q, [0, 2, 1, 3]));
    let kq = ctx.soft_max_ext(kq, mask, scale, 0.0);
    let gathered = ctx
        .cont(ctx.permute(ctx.mul_mat(vt, kq), [0, 2, 1, 3]))
        .as_output();

    let mut graph = ctx.new_graph(64);
    graph.expand(paged);
    graph.expand(gathered);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();

    let qdat = noise(q.elements(), 3);
    q.set_f32(&qdat);
    let mut t = Vec::new();
    let mut b = Vec::new();
    let mut m = vec![0f32; n_kv * n_tok * ns];
    for (s, tab) in tables.iter().enumerate() {
        t.extend(tab.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
        let used = (n_past[s] + n_tok).div_ceil(KV_BLOCK);
        b.extend((0..n_kv / KV_BLOCK).map(|i| tab[i.min(used - 1)] as i32));
        for tok in 0..n_tok {
            let p = n_past[s] + tok;
            let row = &mut m[(s * n_tok + tok) * n_kv..][..n_kv];
            row[p + 1..].fill(f32::NEG_INFINITY);
            if window > 0 {
                row[..(p + 1).saturating_sub(window)].fill(f32::NEG_INFINITY);
            }
        }
    }
    table.set_i32(&t);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    blocks.set_i32(&b);
    mask.set_f32(&m);
    backend.compute(&graph).unwrap();

    let (a, b) = (paged.to_f32(), gathered.to_f32());
    assert_eq!(a.len(), b.len());
    let max_diff = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };

    let shape = Shape {
        d: d as usize,
        n_head: n_head as usize,
        n_head_kv: n_head_kv as usize,
        n_tok,
        scale: scale as f64,
        window,
    };
    let want = oracle(&shape, &qdat, &kdat, &vdat, n_past, tables);
    let label = format!("d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, window {window}");
    // The op is exact up to f32 summation order, on every architecture.
    let vs_oracle = max_diff(&a, &want);
    assert!(
        vs_oracle < 1e-5,
        "{label}: max |paged - f64 oracle| {vs_oracle}"
    );
    // The gather path is only as exact as ggml's CPU matmuls. On aarch64 with FP16 vector
    // arithmetic, ggml's llamafile sgemm (used for several tokens per sequence) accumulates F16
    // products in F16 registers, which is off by ~2e-4 here (x86: ~6e-8). That is
    // ggml's precision, not the op's, so the comparison is a wiring check: a wrong block, mask
    // or head mapping is off by orders of magnitude more.
    let vs_gather = max_diff(&a, &b);
    assert!(
        vs_gather < GATHER_TOLERANCE,
        "{label}: max |paged - gather| {vs_gather} (|gather - oracle| {})",
        max_diff(&b, &want)
    );
}

#[test]
fn sliding_window_and_scale_match_gather() {
    // Sliding-window layers with a unit scale: keys of the last `window` positions only.
    // Decode with the window starting mid-block, and a chunk whose rows start their windows
    // in different blocks.
    check_with(
        32,
        4,
        2,
        1,
        &[70, 130],
        &[vec![4, 1], vec![0, 5, 2]],
        0.3,
        40,
    );
    check_with(32, 4, 1, 9, &[120], &[vec![2, 0, 1]], 0.25, 70);
    // A window longer than the context changes nothing.
    check_with(32, 4, 2, 3, &[20], &[vec![1]], 0.2, 1024);
}

/// See [`check`]: covers ggml's F16 accumulation on aarch64 with ~10x margin.
const GATHER_TOLERANCE: f32 = 2e-3;

#[test]
fn decode_matches_gather() {
    // Two sequences over interleaved, out-of-order blocks; GQA 4.
    check(64, 8, 2, 1, &[70, 130], &[vec![4, 1], vec![0, 5, 2]]);
}

#[test]
fn prefill_chunks_match_gather() {
    // Several tokens per sequence, crossing a block boundary; causal within the chunk.
    check(64, 8, 2, 5, &[60, 0], &[vec![3, 1], vec![2]]);
    check(32, 4, 4, 3, &[127], &[vec![1, 0, 2]]);
    check(32, 8, 1, 2, &[5], &[vec![0]]);
    // GQA 6 (Qwen3.5 / 3.6 / 3.8-27B: 24 query heads over 4 kv heads), scaled down.
    check(64, 12, 2, 5, &[60, 0], &[vec![3, 1], vec![2]]);
    check(32, 12, 2, 1, &[70, 130], &[vec![4, 1], vec![0, 5, 2]]);
}

#[test]
fn supported_shapes() {
    // Qwen3.5 / 3.6 / 3.8-27B (GQA 6), Qwen3.5-4B (GQA 4), qwen4exp's MTP head (GQA 12).
    assert!(supported(256, 24, 4));
    assert!(supported(256, 16, 4));
    assert!(supported(256, 24, 2));
    for g in [1, 2, 4, 6, 8, 12, 16] {
        assert!(supported(128, 4 * g, 4), "GQA {g}");
    }
    // Ratios without a kernel instance, uneven ratios, other head sizes.
    for g in [3, 5, 7, 10, 32] {
        assert!(!supported(128, 4 * g, 4), "GQA {g}");
    }
    assert!(!supported(256, 24, 5));
    assert!(!supported(64, 24, 4));
    assert!(!supported(256, 24, 0));
}

/// The CUDA kernels over a long context (8K keys, Qwen3.5-4B's head shape) against the f64
/// oracle. V is positive, so every P V product has the same sign and an accumulator that rounds
/// toward zero shows up as a bias of the output (Volta's and Turing's tensor cores do, see
/// paged_attn.cu), next to the unbiased rounding noise of the F16 operands. Prefill (a 64-token
/// chunk at the end of the context, tensor cores) and decode (flash-decoding, plain F32) are
/// both checked. Measured on a V100: prefill max |diff| 2.2e-6, relative bias -9e-8; with one
/// accumulator chain per output tile instead (the kernel before per-step accumulation), 9e-5
/// and -1.6e-4.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_long_context_matches_oracle() {
    // Qwen3.5-4B's heads (GQA 4), qwen4exp's MTP head's (24 over 2: GQA 12, which does not
    // divide a 64-row prefill tile) and Qwen3.5 / 3.6 / 3.8-27B's (24 over 4: GQA 6, neither).
    for (n_head, n_head_kv) in [(16, 4), (24, 2), (24, 4)] {
        cuda_long_context(256, n_head, n_head_kv, 1.0 / 16.0, 0);
    }
}

/// The same at a unit scale (smaller queries, so the softmax does not collapse to one key):
/// 256-wide heads, GQA 2, a 1024-position sliding window; and 512-wide heads, GQA 8.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_wide_heads_and_window_match_oracle() {
    cuda_long_context(256, 16, 8, 1.0 / 16.0, 1024);
    cuda_long_context(512, 32, 4, 1.0 / 22.6, 0);
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_long_context(
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    scale: f32,
    window: usize,
) {
    use crate::ggml::{DeviceKind, device_by_kind};
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let ctx_len = 8192;
    let n_blocks = ctx_len / KV_BLOCK;
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks).unwrap();
    let (kp, vp) = kv.pools(0);
    let mut kdat = noise(kp.elements(), 11);
    let mut vdat: Vec<f32> = noise(vp.elements(), 12).iter().map(|x| x.abs()).collect();
    kp.set_bytes(&f16_bytes(&kdat));
    vp.set_bytes(&f16_bytes(&vdat));
    round_f16(&mut kdat);
    round_f16(&mut vdat);
    // Blocks out of order: sequence block i lives in pool block i * 37 mod n_blocks.
    let table: Vec<u32> = (0..n_blocks).map(|i| (i * 37 % n_blocks) as u32).collect();

    for n_tok in [64, 7, 1] {
        let n_past = ctx_len - n_tok;
        let ctx = Context::new(64, 64);
        let q = ctx
            .new_tensor(types::F32, &[d as i64, n_head as i64, n_tok as i64, 1])
            .as_input();
        let tab = ctx.new_tensor(types::I32, &[n_blocks as i64, 1]).as_input();
        let past = ctx.new_tensor(types::I32, &[1]).as_input();
        let scratch_ne = scratch_shape(d as i64, n_head as i64, n_head_kv as i64, n_tok, 1);
        let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
        let out = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0).as_output();
        let mut graph = ctx.new_graph(64);
        graph.expand(out);
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
            .unwrap();
        // Larger than unit queries, so the softmax is far from uniform.
        let qv: Vec<f32> = noise(q.elements(), 13).iter().map(|x| 3.0 * x).collect();
        q.set_f32(&qv);
        tab.set_i32(&table.iter().map(|&b| b as i32).collect::<Vec<_>>());
        past.set_i32(&[n_past as i32]);
        backend.compute(&graph).unwrap();

        let sh = Shape {
            d,
            n_head,
            n_head_kv,
            n_tok,
            scale: scale as f64,
            window,
        };
        let want = oracle(
            &sh,
            &qv,
            &kdat,
            &vdat,
            &[n_past],
            std::slice::from_ref(&table),
        );
        let got = out.to_f32();
        assert_eq!(got.len(), want.len());
        let (mut max_diff, mut bias, mut mag) = (0f64, 0f64, 0f64);
        for (&g, &w) in got.iter().zip(&want) {
            let e = g as f64 - w as f64;
            max_diff = max_diff.max(e.abs());
            bias += e;
            mag += (w as f64).abs();
        }
        let rel_bias = bias / mag;
        println!(
            "d {d}, GQA {}, window {window}: {n_tok} tokens over {ctx_len} keys: max |diff| \
             {max_diff:.2e}, relative bias {rel_bias:.2e}",
            n_head / n_head_kv
        );
        // The sm_80+ prefill kernel rounds each tile's weights to F16 against the running row
        // max rather than the final one, as flash attention does: a little more rounding noise
        // than the oracle's (GB10: up to 3.6e-5 at d 256 with a window), no bias.
        assert!(max_diff < 6e-5, "{n_tok} tokens: max |diff| {max_diff}");
        assert!(
            rel_bias.abs() < 1e-5,
            "{n_tok} tokens: relative bias {rel_bias}"
        );
    }
}

/// The prefill kernel's edges against the f64 oracle, per head size and GQA ratio: several
/// sequences at different positions over shuffled blocks, chunks that are no multiple of a
/// CTA's tokens (the last tile part padding), causal rows that start mid-block, and sliding
/// windows that start and end inside blocks, with a unit scale (Gemma 4) and 1 / sqrt(d).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_prefill_edges_match_oracle() {
    cuda_prefill_edges();
}

#[cfg(feature = "cuda")]
pub(super) fn cuda_prefill_edges() {
    let tables = [vec![5, 0, 9, 3, 7], vec![2, 10], vec![8, 1, 6, 4]];
    for (d, n_head, n_head_kv) in [
        (128, 32, 2),
        (128, 8, 8),
        (256, 16, 8),
        (256, 16, 4),
        (256, 24, 4),
        (256, 24, 2),
        (512, 16, 2),
        (512, 32, 4),
    ] {
        for (n_tok, window, scale) in [
            (37, 0, 1.0 / (d as f32).sqrt()),
            (70, 100, 1.0 / (d as f32).sqrt()),
            (5, 64, 0.08),
        ] {
            cuda_check(
                d,
                n_head,
                n_head_kv,
                n_tok,
                &[190, 3, 120],
                &tables,
                scale,
                window,
            );
        }
    }
}

/// The CUDA op on `n_tok` tokens per sequence (`n_past`, `tables` per sequence; V in [-1, 1))
/// against the f64 oracle.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_check(
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
    scale: f32,
    window: usize,
) {
    let scratch_ne = scratch_shape(
        d as i64,
        n_head as i64,
        n_head_kv as i64,
        n_tok,
        n_past.len(),
    );
    cuda_check_with(
        d, n_head, n_head_kv, n_tok, n_past, tables, scale, window, scratch_ne,
    );
}

/// [`cuda_check`] with the given scratch (`split_scratch_shape`: the group split over the key
/// blocks).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn cuda_check_with(
    d: usize,
    n_head: usize,
    n_head_kv: usize,
    n_tok: usize,
    n_past: &[usize],
    tables: &[Vec<u32>],
    scale: f32,
    window: usize,
    scratch_ne: [i64; 4],
) {
    use crate::ggml::{DeviceKind, device_by_kind};
    let hp = hparams(d as i64, n_head as i64, n_head_kv as i64);
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let n_blocks = tables.iter().flatten().max().unwrap() + 1;
    let kv = KvPool::new(&hp, backend.buffer_type(), n_blocks as usize).unwrap();
    let (kp, vp) = kv.pools(0);
    let mut kdat = noise(kp.elements(), 21);
    let mut vdat = noise(vp.elements(), 22);
    kp.set_bytes(&f16_bytes(&kdat));
    vp.set_bytes(&f16_bytes(&vdat));
    round_f16(&mut kdat);
    round_f16(&mut vdat);

    let ns = n_past.len();
    let width = tables.iter().map(Vec::len).max().unwrap();
    let ctx = Context::new(64, 64);
    let q = ctx
        .new_tensor(
            types::F32,
            &[d as i64, n_head as i64, n_tok as i64, ns as i64],
        )
        .as_input();
    let tab = ctx
        .new_tensor(types::I32, &[width as i64, ns as i64])
        .as_input();
    let past = ctx.new_tensor(types::I32, &[ns as i64]).as_input();
    let scratch = ctx.new_tensor(types::F32, &scratch_ne).as_input();
    let out = attention_with(&ctx, q, kp, vp, tab, past, scratch, scale, window, 0).as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // With a unit scale, smaller queries keep the softmax from collapsing to one key.
    let qmul = 2.0 * scale * (d as f32).sqrt().min(16.0);
    let qv: Vec<f32> = noise(q.elements(), 23).iter().map(|x| qmul * x).collect();
    q.set_f32(&qv);
    let mut t = Vec::new();
    for (s, tb) in tables.iter().enumerate() {
        assert!(tb.len() * KV_BLOCK >= n_past[s] + n_tok);
        t.extend(tb.iter().map(|&x| x as i32));
        t.resize((s + 1) * width, 0);
    }
    tab.set_i32(&t);
    past.set_i32(&n_past.iter().map(|&p| p as i32).collect::<Vec<_>>());
    backend.compute(&graph).unwrap();

    let sh = Shape {
        d,
        n_head,
        n_head_kv,
        n_tok,
        scale: scale as f64,
        window,
    };
    let want = oracle(&sh, &qv, &kdat, &vdat, n_past, tables);
    let got = out.to_f32();
    assert_eq!(got.len(), want.len());
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, window {window}: max |diff| \
         {max_diff:.2e}"
    );
    // Over a few hundred keys the weights are large, and F16 rounds each to within a relative
    // 2^-12: the oracle its normalized weights, the sm_80+ kernel its weights against the
    // running max. With |V| < 1 either is within 2^-12 of the exact sum, so within 2^-11 =
    // 4.9e-4 of the other (GB10: 2.6e-4 to 3.8e-4 here); a wrong key, mask or head is off by
    // orders of magnitude more.
    assert!(
        max_diff < 5e-4,
        "d {d}, heads {n_head}/{n_head_kv}, {n_tok} tokens, window {window}: max |diff| {max_diff}"
    );
}
