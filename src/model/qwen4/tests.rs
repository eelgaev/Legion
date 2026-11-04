//! The per-layer embedding's conv: the token-major path (verify groups, one-token steps) and the
//! row path prompt chunks take give the same bits, output and carried history alike (#462).

use std::cell::RefCell;

use super::*;
use crate::ggml::{Backend, ComputeAllocator};
use crate::model::core::{Logits, SeqInput};

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

fn ple_params(conv_kernel: i64, ngram: usize) -> PleParams {
    PleParams {
        layer: 1,
        ngram,
        heads_per_ngram: 1,
        conv_kernel,
        eos: 0,
        head_dim: 1,
        multipliers: vec![1; ngram],
        offsets: Vec::new(),
        vocab_sizes: Vec::new(),
        blocks: None,
        fp8_scale: None,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// One chunk of `nt` tokens through both paths, each from its own history (`ha`, `hb`):
/// (token-major output, its new history, row output, its new history).
fn conv_step(
    backend: &mut Backend,
    width: i64,
    ns: i64,
    nt: i64,
    ple: &PleParams,
    (ha, hb, wv, xv): (&[f32], &[f32], &[f32], &[f32]),
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hist = ple.history();
    let ctx = Context::new(256, 512);
    let state = |h: &[f32]| {
        let t = ctx.new_tensor(types::F32, &[hist, width, ns]);
        (t.as_input().as_output(), h.to_vec())
    };
    let ((sa, ha), (sb, hb)) = (state(ha), state(hb));
    let w = ctx
        .new_tensor(types::F32, &[ple.conv_kernel, width])
        .as_input();
    let x = ctx.new_tensor(types::F32, &[width, nt, ns]).as_input();
    let graph = RefCell::new(ctx.new_graph(256));
    // The token-major path with the history update one-token steps take (and prompt chunks
    // took before #462).
    let padded = ctx.concat(sa, ctx.transpose(x), 0);
    ple_keep_history(&ctx, &graph, padded, sa, nt, ple);
    let oa = ctx
        .cont(ple_conv_time_major(&ctx, w, padded, nt, ple))
        .as_output();
    let ob = ctx
        .cont(ple_conv_rows(&ctx, &graph, w, sb, x, ple))
        .as_output();
    let mut graph = graph.into_inner();
    graph.expand(oa);
    graph.expand(ob);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    sa.set_f32(&ha);
    sb.set_f32(&hb);
    w.set_f32(wv);
    x.set_f32(xv);
    backend.compute(&graph).unwrap();
    (oa.to_f32(), sa.to_f32(), ob.to_f32(), sb.to_f32())
}

/// Runs chunks of `chunks` tokens in turn, each path carrying its own history, and asserts
/// every output and history matches bit for bit.
fn conv_paths_match(width: i64, ns: i64, chunks: &[i64], ple: &PleParams) {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    let hist = ple.history();
    let wv = noise((ple.conv_kernel * width) as usize, 4);
    let mut ha = noise((hist * width * ns) as usize, 3);
    let mut hb = ha.clone();
    for (i, &nt) in chunks.iter().enumerate() {
        let xv = noise((width * nt * ns) as usize, 10 + i as u64);
        let (oa, na, ob, nb) = conv_step(&mut backend, width, ns, nt, ple, (&ha, &hb, &wv, &xv));
        assert_eq!(oa.len(), (width * nt * ns) as usize);
        assert_eq!(bits(&oa), bits(&ob), "output, chunk {i} of {nt}");
        assert_eq!(bits(&na), bits(&nb), "history, chunk {i} of {nt}");
        assert_ne!(bits(&na), bits(&ha), "history moved, chunk {i}");
        (ha, hb) = (na, nb);
    }
}

#[test]
fn ple_conv_row_path_matches_token_major_bitwise() {
    // qwen4exp's kernel and n-gram size: a history of 9.
    let ple = ple_params(4, 3);
    // A long chunk, one shorter than the history, then single-token steps.
    conv_paths_match(40, 1, &[37, 5, 1, 1], &ple);
    // Several sequences side by side.
    conv_paths_match(24, 3, &[12, 1, 9], &ple);
}

#[test]
fn ple_conv_row_path_matches_other_shapes() {
    for (kern, ngram) in [(2, 2), (3, 4), (2, 5)] {
        let ple = ple_params(kern, ngram);
        conv_paths_match(16, 2, &[11, 2, 1], &ple);
    }
}

/// Rows the n-gram store read into RAM stay on the calling thread up to a big prompt step's
/// count; rows read from the weights' memory (page faults) take a thread per 64 as before.
#[test]
fn ple_gather_threads_keep_decode_steps_on_the_engine_thread() {
    // Decode steps of 1 to 255 sequences, 16 rows a token.
    for n_seqs in [1, 8, 16, 64, 255] {
        assert_eq!(
            ple_gather_threads(16 * n_seqs, true),
            1,
            "{n_seqs} sequences"
        );
    }
    // A 2,048-token prompt chunk.
    assert_eq!(ple_gather_threads(16 * 2048, true), PLE_GATHER_THREADS);
    assert_eq!(ple_gather_threads(2 * 4096, true), 2);
    // The weights' own memory: unchanged.
    assert_eq!(ple_gather_threads(16, false), 1);
    assert_eq!(ple_gather_threads(16 * 16, false), 4);
    assert_eq!(ple_gather_threads(16 * 2048, false), PLE_GATHER_THREADS);
    assert_eq!(ple_gather_threads(0, true), 1);
}

/// A group of `n_seqs` sequences of `n_tok` tokens each, `n_past` already cached.
fn pool_group(tokens: &[u32], n_seqs: usize, n_past: usize) -> Group<'_> {
    Group {
        slot: 0,
        seqs: (0..n_seqs)
            .map(|_| SeqInput {
                tokens,
                n_past,
                blocks: &[],
                ring: 0,
                logits: Logits::All,
                media: None,
            })
            .collect(),
        verify: true,
    }
}

/// #858: at 64 steps one sequence lists the pools it did before (64-padded); more sequences'
/// decode and verify steps pad coarser, so the shape lasts at least 64 steps. 0 steps (the
/// default) lists what [`pools_listed`] does.
#[test]
fn step_pools_listed_pads_many_sequences_coarser() {
    let kpool = 4;
    let verify = [0u32; 3];
    let (one, four) = ([0u32; 1], [0u32; 4]);
    for n_past in [0, 37, 250, 1000, 5000] {
        for tokens in [&one[..], &verify, &four] {
            let g = pool_group(tokens, 1, n_past);
            assert_eq!(step_pools_listed(&g, kpool, 64), pools_listed(&g, kpool));
        }
        // Two sequences decoding without drafts: 64 x 2 / 4 = 32, still 64.
        let g = pool_group(&one, 2, n_past);
        assert_eq!(step_pools_listed(&g, kpool, 64), pools_listed(&g, kpool));
    }
    // 16 sequences of 3 tokens: 64 steps x 48 / 4 = 768 pools, padded to 1,024.
    for n_past in [0, 37, 250, 1000, 5000] {
        let g = pool_group(&verify, 16, n_past);
        let real = 16 * (n_past + 3) / kpool;
        let n = step_pools_listed(&g, kpool, 64);
        assert!(n > real && n >= pools_listed(&g, kpool), "{n_past}: {n}");
        assert_eq!(n % 1024, 0, "{n_past}: {n}");
    }
    // 4 sequences: 64 x 12 / 4 = 192, padded to 256.
    let g = pool_group(&verify, 4, 300);
    assert_eq!(step_pools_listed(&g, kpool, 64), 512);
    // Prompt chunks (more than kpool tokens a sequence) as before, and no pools without them.
    let chunk = [0u32; 600];
    // A prompt chunk is no verify group (those pad since #836, below).
    let mut g = pool_group(&chunk, 2, 4096);
    g.verify = false;
    assert_eq!(step_pools_listed(&g, kpool, 64), pools_listed(&g, kpool));
    let small = [0u32; 7];
    let mut g = pool_group(&small, 16, 100);
    g.verify = false;
    assert_eq!(step_pools_listed(&g, kpool, 64), pools_listed(&g, kpool));
    // A verify group of more than kpool tokens a sequence (MTP's 9-row verify, #836) pads too.
    let g = pool_group(&small, 16, 100);
    let n = step_pools_listed(&g, kpool, 64);
    assert!(
        n >= pools_listed(&g, kpool) && n.is_multiple_of(1024),
        "{n}"
    );
    assert_eq!(step_pools_listed(&pool_group(&verify, 16, 100), 0, 64), 0);
    for n_past in [0, 37, 250, 1000, 5000] {
        for n_seqs in [1, 4, 16] {
            let g = pool_group(&verify, n_seqs, n_past);
            assert_eq!(step_pools_listed(&g, kpool, 0), pools_listed(&g, kpool));
        }
    }
}

/// #836: a verify group of more than kpool tokens a sequence computes a fixed number of pool
/// keys (its shape-only bound, at least what it completes), whatever its positions; other
/// groups as [`pools_new`].
#[test]
fn step_pools_new_pads_long_verify_groups() {
    let kpool = 4;
    let nine = [0u32; 9];
    for n_past in 0..64 {
        for n_seqs in [1, 4, 16] {
            let g = pool_group(&nine, n_seqs, n_past);
            let real = n_seqs * ((n_past + 9) / kpool - n_past / kpool);
            assert_eq!(step_pools_new(&g, kpool), n_seqs * 3);
            assert!(step_pools_new(&g, kpool) >= real);
        }
    }
    // Plain decode and short verify groups: as before (already the bound).
    let (one, three) = ([0u32; 1], [0u32; 3]);
    for n_past in [0, 3, 37, 250] {
        for tokens in [&one[..], &three] {
            let g = pool_group(tokens, 16, n_past);
            assert_eq!(step_pools_new(&g, kpool), pools_new(&g, kpool));
        }
    }
    // Prompt chunks (not verify groups) and bounds past the scratch rows: the real count.
    let mut g = pool_group(&nine, 16, 37);
    g.verify = false;
    assert_eq!(step_pools_new(&g, kpool), pools_new(&g, kpool));
    let long = [0u32; 64];
    let g = pool_group(&long, 16, 37);
    const { assert!(16 * 16 >= QSA_SCRATCH) };
    assert_eq!(step_pools_new(&g, kpool), pools_new(&g, kpool));
    assert_eq!(step_pools_new(&pool_group(&nine, 16, 37), 0), 0);
}

/// #858: a 16-sequence verify batch keeps its listed pools (its graph shape) for at least
/// 64 steps of 3 tokens each, where 64-padding changed them every ~5 steps.
#[test]
fn step_pools_listed_keeps_the_shape_for_many_steps() {
    let kpool = 4;
    let verify = [0u32; 3];
    let listed = |n_past| step_pools_listed(&pool_group(&verify, 16, n_past), kpool, 64);
    let plain = |n_past| pools_listed(&pool_group(&verify, 16, n_past), kpool);
    let changes = |f: &dyn Fn(usize) -> usize| {
        (1..=256 / 3)
            .filter(|&s| f(100 + 3 * s) != f(100 + 3 * (s - 1)))
            .count()
    };
    let ours = changes(&listed);
    assert!(ours <= 2, "{ours} changes in 85 steps");
    assert!(changes(&plain) >= 12);
}
