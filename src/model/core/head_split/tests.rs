use super::*;
use crate::model::cluster_select::{argmax_parts, argmax_row, best_row, join_best};
use crate::model::core::SeqInput;

fn seq(tokens: &[u32], logits: Logits) -> SeqInput<'_> {
    SeqInput {
        tokens,
        n_past: 0,
        blocks: &[],
        ring: 0,
        logits,
        media: None,
    }
}

fn group(seqs: Vec<SeqInput<'_>>) -> Group<'_> {
    Group {
        slot: 0,
        seqs,
        verify: false,
    }
}

#[test]
fn shares_cover_the_vocabulary_at_a_tie_boundary() {
    // Qwen3.8-Flash-Next's vocabulary.
    let n = 248_320;
    assert!(vocab_fits(n));
    let (a, b) = (range(n, 1), range(n, 2));
    assert_eq!(a, (0, 124_928));
    assert_eq!(b, (124_928, 123_392));
    assert_eq!(a.1 + b.1, n);
    assert_eq!(b.0 % ALIGN, 0);
    assert_eq!((a.1 % 128, b.1 % 128), (0, 0));
    assert!(!vocab_fits(248_321));
    assert!(!vocab_fits(2048));
}

#[test]
fn only_greedy_decode_and_verify_steps_split() {
    let t = [1u32, 2, 3];
    let greedy = group(vec![
        seq(&t[..1], Logits::Greedy),
        seq(&t[..1], Logits::None),
    ]);
    assert!(fits(std::slice::from_ref(&greedy), 16));
    let verify = group(vec![seq(&t, Logits::FirstGreedy(3))]);
    assert!(fits(&[greedy.clone(), verify.clone()], 16));
    // A prompt chunk, sampled rows, picks below min_tokens, or no rows at all.
    assert!(!fits(std::slice::from_ref(&verify), 2));
    for other in [
        Logits::Last,
        Logits::All,
        Logits::First(2),
        Logits::Pick,
        Logits::FirstPick(2, 1),
    ] {
        assert!(!fits(&[greedy.clone(), group(vec![seq(&t, other)])], 16));
    }
    assert!(!fits(&[group(vec![seq(&t, Logits::None)])], 16));
}

#[test]
fn the_better_share_wins_by_value_then_tie_order() {
    assert_eq!(pick((5, 1.0), (2000, 2.0)), 2000);
    assert_eq!(pick((5, 2.0), (2000, 1.0)), 5);
    // Equal values: the device argmax's order, not the lower column. Thread 0's columns
    // (1024, 3072) come before lane 16's (2064) and lane 1's (1).
    assert_eq!(pick((1024, 1.0), (2064, 1.0)), 1024);
    assert_eq!(pick((1, 1.0), (3072, 1.0)), 3072);
    assert_eq!(pick((u32::MAX, -f32::MAX), (7, -3.0)), 7);
    assert_eq!(pick((7, -3.0), (u32::MAX, -f32::MAX)), 7);
    assert_eq!(pick((u32::MAX, 0.0), (u32::MAX, 0.0)), u32::MAX);
    assert_eq!(
        join((&[1u32][..], &[1.0f32][..]), (&[2u32][..], &[][..])),
        None
    );
}

/// Rows with ties across the split and rows without a candidate: the two shares' picks join
/// to the whole row's device argmax.
#[test]
fn joined_shares_are_the_whole_rows_argmax() {
    let n = 16 * 1024 + 512;
    assert!(vocab_fits(n));
    let rows = 6;
    let mut x = vec![0f32; n * rows];
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for (r, row) in x.chunks_mut(n).enumerate() {
        for v in row.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // A few levels: many equal maxima on both sides of the split.
            *v = (s % [3, 50, 1000][r % 3]) as f32;
        }
    }
    x[4 * n..5 * n].fill(f32::NEG_INFINITY);
    let (first, _) = range(n, 2);
    x[5 * n..5 * n + first].fill(-f32::MAX);
    let share = |half: u8| {
        let (off, len) = range(n, half);
        let parts = argmax_parts(len as i64) as usize;
        let best: Vec<u32> = x
            .chunks(n)
            .flat_map(|row| best_row(&row[off..off + len], parts))
            .flat_map(|(v, r)| [v, r])
            .collect();
        let (cols, vals) = join_best(&best, rows);
        let cols: Vec<u32> = cols
            .into_iter()
            .map(|c| if c == u32::MAX { c } else { c + off as u32 })
            .collect();
        (cols, vals)
    };
    let (a, b) = (share(1), share(2));
    let got = join((&a.0[..], &a.1[..]), (&b.0[..], &b.1[..])).unwrap();
    for (r, row) in x.chunks(n).enumerate() {
        assert_eq!(got[r] as i32, argmax_row(row), "row {r}");
    }
    assert_eq!(got[4], u32::MAX);
    assert!(got[5] as usize >= first);
}
