//! The sparse attention op over quantized caches (#263: q8_0 / q4_0 K and V): the CPU op and,
//! on CUDA (ignored tests), the decode and prompt kernels against the f64 oracle over the
//! cache's values as F16 (dequantized, then rounded, as the gather path reads them) and against
//! the gather path over the same selection.

use super::tests::{Case, GATHER_TOLERANCE, cpu, run_typed};
use super::*;
use crate::model::core::KvType::{self, F16, Q4_0, Q5_1, Q8_0};
use crate::model::core::KvTypes;

/// The quantized pairs the CUDA kernels read.
const PAIRS: [(KvType, KvType); 3] = [(Q8_0, Q8_0), (Q8_0, Q4_0), (Q4_0, Q4_0)];

#[test]
fn reads_types_takes_f16_or_both_quantized() {
    let t = |k, v| reads_types(KvTypes { k, v });
    assert!(t(F16, F16));
    for (k, v) in PAIRS {
        assert!(t(k, v) && t(v, k), "{k:?} / {v:?}");
    }
    assert!(
        !t(F16, Q8_0) && !t(Q4_0, F16),
        "F16 beside a quantized cache"
    );
    assert!(
        !t(Q5_1, Q5_1) && !t(Q8_0, Q5_1),
        "types the kernels do not read"
    );
}

#[test]
fn cpu_op_reads_quantized_caches() {
    let cases = [
        Case {
            d: 32,
            n_head: 4,
            n_head_kv: 2,
            kpool: 4,
            n_top: 64,
            n_tok: 9,
            n_past: vec![40],
            tables: vec![vec![1, 0]],
        },
        Case {
            d: 64,
            n_head: 8,
            n_head_kv: 1,
            kpool: 4,
            n_top: 3,
            n_tok: 6,
            n_past: vec![61, 100],
            tables: vec![vec![3, 1], vec![0, 4]],
        },
    ];
    let mut backend = cpu();
    for c in &cases {
        // Every type the gather path reads, not only the kernels' pairs.
        for (k, v) in PAIRS.into_iter().chain([(F16, Q8_0), (Q5_1, Q5_1)]) {
            let r = run_typed(c, &mut backend, 1.0, KvTypes { k, v });
            assert!(
                r.op_rounded < 1e-5,
                "{k:?} / {v:?}: |op - oracle| {} (unrounded oracle: op {})",
                r.op_rounded,
                r.op_exact
            );
            assert!(
                r.op_gather < GATHER_TOLERANCE,
                "{k:?} / {v:?}: |op - gather| {} (|gather - oracle| {})",
                r.op_gather,
                r.gather_rounded
            );
        }
    }
}

/// The CUDA kernels over q8_0 / q4_0 caches with the tolerances of the F16 test
/// (`cuda_matches_oracle_and_dense_path`): qwen4exp's shape (head size 256, 24 heads over 2 kv
/// heads) in decode, verify and prompt steps, and head size 128; the prompt kernel with its
/// chain off and on (on every device: a quantized cache takes the m8n8k4 kernel).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_reads_quantized_caches() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use std::sync::atomic::Ordering;
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    // (head size, heads, kv heads, context, tokens, sequences)
    let cases = [
        (256, 24, 2, 8192, 1, 1),
        (256, 24, 2, 5000, 3, 2),
        (256, 24, 2, 300, 300, 1),
        (256, 24, 2, 8192, 128, 1),
        (128, 16, 2, 4096, 1, 1),
        (128, 16, 2, 4096, 2048, 1),
        (128, 8, 8, 4096, 128, 1),
    ];
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            CHAIN_OVERRIDE.store(0, Ordering::Relaxed);
        }
    }
    let _reset = Reset;
    for chain in [false, true] {
        CHAIN_OVERRIDE.store(if chain { 2 } else { 1 }, Ordering::Relaxed);
        for &(d, n_head, n_head_kv, n_ctx, n_tok, n_seqs) in &cases {
            if chain && n_tok * n_seqs <= DECODE_TOKENS as usize {
                continue; // the split kernel has no chain
            }
            let n_blocks = n_ctx / KV_BLOCK + 1;
            let tables: Vec<Vec<u32>> = (0..n_seqs)
                .map(|s| {
                    (0..n_blocks)
                        .map(|i| (s * n_blocks + i * 37 % n_blocks) as u32)
                        .collect()
                })
                .collect();
            let c = Case {
                d,
                n_head,
                n_head_kv,
                kpool: 4,
                n_top: 512,
                n_tok,
                n_past: (0..n_seqs).map(|s| n_ctx - n_tok - 7 * s).collect(),
                tables,
            };
            for (k, v) in PAIRS {
                let r = run_typed(&c, &mut backend, 3.0, KvTypes { k, v });
                println!(
                    "{k:?} / {v:?}, chain {chain}, d {d}, {n_head}/{n_head_kv} heads, {n_seqs} x \
                     {n_tok} tokens at {n_ctx}: op vs exact {:.2e} (bias {:.1e}), vs dense {:.2e}; \
                     op bits {:016x}",
                    r.op_exact, r.op_bias, r.op_gather, r.op_hash
                );
                assert!(r.op_exact < 3e-5, "|op - exact| {}", r.op_exact);
                assert!(r.op_bias.abs() < 1e-6, "relative bias {}", r.op_bias);
                assert!(r.op_gather < 2e-3, "|op - dense| {}", r.op_gather);
            }
        }
    }
}

/// Times the CUDA kernels at qwen4exp's shape per cache type, twice over (the types in turn): a
/// 4,096-token prompt chunk at the end of a 32K and of a 123K context, and decode steps of 1 and
/// 4 tokens at 123K.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU; a timing, not a check"]
fn cuda_quantized_attention_speed() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    let shapes = [
        (32768, 4096, 10),
        (123_000, 4096, 10),
        (123_000, 1, 400),
        (123_000, 4, 400),
    ];
    for (n_ctx, n_tok, reps) in shapes {
        for round in 0..2 {
            for (k, v) in [(F16, F16)].into_iter().chain(PAIRS) {
                let (ms, bits) =
                    super::tests::attention_ms(&mut backend, KvTypes { k, v }, n_ctx, n_tok, reps);
                println!(
                    "qsa_attn {k:?} / {v:?}: {n_tok} tokens at {n_ctx} (round {round}): {ms:.3} ms \
                     per call (output bits {bits:016x})"
                );
            }
        }
    }
}
