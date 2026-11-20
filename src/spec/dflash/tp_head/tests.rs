use super::*;
use crate::spec::dflash::fused::top_k_row;
use crate::spec::dflash::{RoundShape, round_sums};

/// A row of `n` logits from a few values only (many ties, both zeros), from seed `seed`.
fn row(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let values = [-3.5, -0.0, 0.0, 1.25, 2.0, 7.75, f32::NEG_INFINITY, 7.5];
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Mostly low values, a few high ones: ties at the k-th place.
            let i = (s % 64) as usize;
            if i < values.len() {
                values[i]
            } else {
                -1.0 - (s % 5) as f32
            }
        })
        .collect()
}

/// The top-k op's candidates of `row` (columns from 0) and their logits.
fn top(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut out = vec![0i32; k];
    top_k_row(row, &mut out);
    let cols: Vec<u32> = out.iter().map(|&c| c as u32).collect();
    let logits = cols.iter().map(|&c| row[c as usize]).collect();
    (cols, logits)
}

/// The exchange of a pass whose output rows are `rows` (each the whole head's logits): each
/// rank's top-k of its share, packed in its half, `-0.0` added to the other half.
fn exchange(rows: &[Vec<f32>], sp: Split) -> Vec<f32> {
    let block = |rank: usize| -> Vec<f32> {
        let (first, len) = sp.share(rank);
        rows.iter()
            .flat_map(|r| {
                let (c, l) = top(&r[first..first + len], sp.k);
                pack(&c, &l, sp.k)
            })
            .collect()
    };
    let zeros = vec![-0.0f32; 2 * sp.k * rows.len()];
    let half0: Vec<f32> = block(0).iter().zip(&zeros).map(|(a, b)| a + b).collect();
    let half1: Vec<f32> = zeros.iter().zip(block(1)).map(|(a, b)| a + b).collect();
    half0.into_iter().chain(half1).collect()
}

/// The merge of the two shares' top-k is the whole head's top-k: the same candidates in the
/// same order with the same logits, ties at the k-th place and across the shares included.
#[test]
fn the_merged_halves_are_the_whole_heads_top_k() {
    let n_vocab = 4096 + 128;
    for k in [1, 2, 7, 16, 32] {
        let sp = Split::new(k, n_vocab).expect("fits");
        let rows: Vec<Vec<f32>> = (0..6).map(|s| row(n_vocab, s + 11 * k as u64)).collect();
        let (cols, logits) = unpack(
            &exchange(&rows, sp),
            k,
            rows.len(),
            [sp.share(0), sp.share(1)],
        );
        for (i, r) in rows.iter().enumerate() {
            let (want_c, want_l) = top(r, k);
            assert_eq!(&cols[i * k..(i + 1) * k], &want_c[..], "k {k} row {i}");
            let bits = |l: &[f32]| l.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&logits[i * k..(i + 1) * k]),
                bits(&want_l),
                "k {k} row {i}"
            );
        }
        let whole: (Vec<u32>, Vec<f32>) = rows.iter().fold((vec![], vec![]), |mut a, r| {
            let (c, l) = top(r, k);
            a.0.extend(c);
            a.1.extend(l);
            a
        });
        assert_eq!(rows_differ((&cols, &logits), (&whole.0, &whole.1), k), 0);
    }
}

/// A tie across the shares goes to the lower column (rank 0's), as in the whole head.
#[test]
fn a_tie_across_the_shares_goes_to_the_lower_column() {
    let sp = Split::new(2, 8192).expect("fits");
    let (first, _) = sp.share(1);
    let mut r = vec![-1.0f32; 8192];
    r[first + 5] = 3.0;
    r[17] = 3.0;
    r[first] = 0.0;
    r[9] = -0.0;
    let (cols, logits) = unpack(&exchange(&[r], sp), 2, 1, [sp.share(0), sp.share(1)]);
    assert_eq!(cols, vec![17, first as u32 + 5]);
    assert_eq!(logits, vec![3.0, 3.0]);
    // +0.0 ranks above -0.0 (ggml's keys), whatever the column.
    assert_eq!(
        merge(&[(9, -0.0)], &[(first as u32, 0.0)], 1),
        vec![(first as u32, 0.0)]
    );
}

/// A rank whose round failed sent zeros: rank 0's candidates stay and nothing breaks; columns
/// outside a share are dropped and a short row is filled at minus infinity.
#[test]
fn a_failed_ranks_zeros_and_bad_columns_do_not_break_the_merge() {
    let sp = Split::new(2, 8192).expect("fits");
    let shares = [sp.share(0), sp.share(1)];
    let mut all = vec![0.0f32; 2 * 2 * 2];
    // Rank 0: columns 3 and 1 at 5.0 and 4.0. Rank 1: NaN and out of its share.
    all[..4].copy_from_slice(&[3.0, 1.0, 5.0, 4.0]);
    all[4..].copy_from_slice(&[f32::NAN, 1e9, 9.0, 9.0]);
    let (cols, logits) = unpack(&all, 2, 1, shares);
    assert_eq!(cols, vec![3, 1]);
    assert_eq!(logits, vec![5.0, 4.0]);
    let all = [0.5f32, -1.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0];
    let (cols, logits) = unpack(&all, 2, 1, shares);
    assert_eq!(cols, vec![shares[1].0 as u32, shares[1].0 as u32]);
    assert_eq!(logits, vec![0.0, 0.0]);
    let all = [f32::NAN; 8];
    let (cols, logits) = unpack(&all, 2, 1, shares);
    assert_eq!(cols, vec![0, 0]);
    assert_eq!(logits, vec![f32::NEG_INFINITY; 2]);
}

/// Gemma 4's head: two shares of 131,072 rows; a pass splits where the top-k op takes every
/// share and the exchange stays in F32.
#[test]
fn the_split_takes_the_passes_the_top_k_op_takes() {
    let sp = Split::new(32, 262_144).expect("fits");
    assert_eq!(sp.share(0), (0, 131_072));
    assert_eq!(sp.share(1), (131_072, 131_072));
    assert_eq!(sp.values(15), 2 * 2 * 32 * 15);
    // 1, 4 and 16 sequences of 15 output rows.
    for rows in [15, 60, 240] {
        assert!(sp.takes(rows), "{rows} rows");
    }
    assert!(!sp.takes(0));
    // The exchange would go in BF16 under `--tp-wire bf16`.
    assert!(!sp.takes(crate::tp_net::BF16_MIN_VALUES / (4 * 32)));
    // Shares shorter than the op's rows, a vocabulary of no whole tiles, no candidates.
    assert!(!Split::new(32, 8192).expect("fits").takes(15));
    assert!(Split::new(32, 259).is_none());
    assert!(Split::new(0, 262_144).is_none());
    assert!(!Split::new(33, 262_144).expect("fits").takes(15));
}

/// `--tp-draft-split`: a DFlash 2 pass whose head splits sums its layers and then exchanges
/// the head's candidates; DFlash and passes the split does not take do not.
#[test]
fn round_sums_count_the_head_exchange() {
    let head = Split::new(4, 32_768);
    let s = RoundShape {
        v2: true,
        n_layer: 2,
        n_embd: 4,
        block: 8,
        first: 1,
        head,
    };
    // Two sequences: 16 rows a layer, 14 output rows.
    assert_eq!(
        round_sums(s, [(3, true), (5, true), (0, true)].into_iter()),
        vec![64, 64, 2 * 2 * 4 * 14]
    );
    assert_eq!(
        round_sums(RoundShape { head: None, ..s }, [(3, true)].into_iter()),
        vec![32, 32]
    );
    assert_eq!(
        round_sums(RoundShape { v2: false, ..s }, [(3, true)].into_iter()),
        vec![16, 16]
    );
    // A vocabulary whose shares the top-k op does not take.
    let small = RoundShape {
        head: Split::new(4, 8192),
        ..s
    };
    assert_eq!(round_sums(small, [(3, true)].into_iter()), vec![32, 32]);
}
