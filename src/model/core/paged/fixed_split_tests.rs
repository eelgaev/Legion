//! Verify steps split over the keys at fixed places (#1133, `--paged-verify-split-keys`): the
//! scratch shapes, then on a GPU the op against the f64 oracle and, bit for bit, the quad
//! kernel's runs against `paged_prefill_mma`'s, rows within one run against the default and a
//! sequence in a batch against the same sequence alone.

use super::*;

/// A slot per run of the table; one run is no split.
#[test]
fn fixed_runs_shapes() {
    // Qwen3.8-27B (24 heads of 256), runs of 256 blocks (16384 keys).
    assert_eq!(fixed_runs_shape(256, 256, 24, 4, 1, 1024), [258, 4, 24, 4]);
    assert_eq!(fixed_runs_shape(256, 256, 24, 4, 1, 1025), [258, 5, 24, 4]);
    assert_eq!(fixed_runs_shape(256, 256, 24, 3, 2, 257), [258, 2, 24, 6]);
    assert_eq!(fixed_runs_shape(256, 256, 24, 4, 1, 256), [1, 1, 1, 1]);
    assert_eq!(fixed_runs_shape(256, 256, 24, 4, 1, 1), [1, 1, 1, 1]);
    // Runs of one block.
    assert_eq!(fixed_runs_shape(1, 256, 24, 2, 1, 3), [258, 3, 24, 2]);
}

/// Off unless set; on, only for groups of 2 to 16 tokens a sequence (and on sm_80+).
#[test]
fn fixed_split_is_off_by_default_and_skips_other_groups() {
    assert_eq!(fixed_split_shape(256, 24, 4, 1, 1024), None);
    set_verify_split_keys_here(Some(16384));
    for n_tok in [1, 17, 2048] {
        assert_eq!(fixed_split_shape(256, 24, n_tok, 1, 1024), None, "{n_tok}");
    }
    set_verify_split_keys_here(None);
    assert_eq!(fixed_split_shape(256, 24, 4, 1, 1024), None);
}

/// Block tables for sequences of `len` positions each, block i of sequence s at pool block
/// (i × n + s) × 7 mod total.
#[cfg(feature = "cuda")]
fn tables(len: &[usize]) -> Vec<Vec<u32>> {
    let width = len.iter().max().unwrap().div_ceil(KV_BLOCK);
    let total = width * len.len();
    assert_ne!(total % 7, 0);
    (0..len.len())
        .map(|s| {
            (0..len[s].div_ceil(KV_BLOCK))
                .map(|i| ((i * len.len() + s) * 7 % total) as u32)
                .collect()
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_fixed_split_matches_oracle() {
    // Runs of 2 blocks (128 keys), so short contexts span several. Qwen3.8-27B's heads take
    // the quad kernel at one sequence and paged_prefill_mma at more; GQA 8 and heads of 128
    // paged_prefill_mma only.
    set_verify_split_keys_here(Some(128));
    for (d, n_head, n_head_kv) in [(256, 24, 4), (256, 32, 4), (128, 16, 2)] {
        for n_tok in [2, 3, 4, 8, 11, 16] {
            let ctx: Vec<usize> = [63, 127, 128, 129, 300, 5000]
                .into_iter()
                .filter(|&c| c >= n_tok)
                .collect();
            let mut runs: Vec<Vec<usize>> = ctx.iter().map(|&c| vec![c]).collect();
            runs.push(ctx.clone());
            for len in &runs {
                let n_past: Vec<usize> = len.iter().map(|&c| c - n_tok).collect();
                let tb = tables(len);
                let width = tb.iter().map(Vec::len).max().unwrap();
                for (window, scale) in [(0, 1.0 / (d as f32).sqrt()), (100, 0.08)] {
                    let ne = fixed_split_shape(d as i64, n_head as i64, n_tok, len.len(), width)
                        .expect("sm_80+");
                    super::tests::cuda_check_with(
                        d, n_head, n_head_kv, n_tok, &n_past, &tb, scale, window, ne,
                    );
                }
            }
        }
    }
    set_verify_split_keys_here(None);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_fixed_split_bits() {
    use super::grid_bench::{fixed_check, run, set_split_keys};
    use super::wmma_bench::{Group, pool};
    let mut gpu = crate::ggml::backend_by_name("CUDA0").expect("CUDA0");
    let heads = (256, 24, 4);
    let kv = pool(&gpu, heads, 400);
    for keys in [64, 128, 1024] {
        set_split_keys(keys);
        // Contexts ending before, on and after run boundaries, 1 to 4 sequences (the quad
        // kernel's grid and paged_prefill_mma's), with and without a sliding window.
        for n_tok in [2, 4, 8, 16] {
            for past in [[60usize, 1000, 3000, 64], [126, 1022, 2047, 300]] {
                for seqs in [1, 2, 4] {
                    let n_past = &past[..seqs];
                    let width = (n_past.iter().max().unwrap() + n_tok).div_ceil(KV_BLOCK);
                    let tables: Vec<Vec<u32>> = (0..seqs)
                        .map(|s| {
                            (0..width)
                                .map(|i| ((i * 37 + s * 11) % 400) as u32)
                                .collect()
                        })
                        .collect();
                    for window in [0, 700] {
                        let g = Group {
                            heads,
                            n_tok,
                            n_past,
                            tables: &tables,
                            width,
                            window,
                            ring: 0,
                            scale: 1.0 / 16.0,
                        };
                        let (a, _) = run(&mut gpu, &kv, &g, "main", 0, 0);
                        let (x, _) = run(&mut gpu, &kv, &g, "fixed", 0, 0);
                        let bad = fixed_check(&mut gpu, &kv, &g, "fixed", &a, &x);
                        assert_eq!(bad, 0, "keys {keys} n_tok {n_tok} {n_past:?} w {window}");
                    }
                }
            }
        }
    }
}
