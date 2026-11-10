use super::*;
use crate::engine::sampling;
use crate::model::cluster_select::argmax_row;

/// The host's greedy pick of a whole row: the stop tokens set to -inf when `masked`, then
/// [`sampling::argmax`] (what the engine did with a row read in full below `min_tokens`).
fn host(row: &[f32], masked: bool, stop: &[u32]) -> u32 {
    let mut row = row.to_vec();
    if masked {
        for &s in stop {
            row[s as usize] = f32::NEG_INFINITY;
        }
    }
    sampling::argmax(&row) as u32
}

/// What a step hands the engine for row `i` of kind `logits` with device argmax `t`: the
/// values [`Need::of`] reads, [`took`], and the whole row when that is not enough (as
/// `Inputs::output_with` does). Returns the argmax slot, the row slot and the values read.
fn step(logits: Logits, i: usize, row: &[f32], t: u32, picks: Picks<'_>) -> (u32, Vec<f32>, usize) {
    let n = row.len();
    let need = Need::of(logits, i, t, n, picks);
    let (mut t, mut out) = (t, Vec::new());
    let mut read = need.len(n);
    if !took(need, &row[..read], &mut t, &mut out, n, picks) {
        let Need::Pick { masked, .. } = need else {
            panic!("only picks fall back");
        };
        let whole = Need::Pick { len: n, masked };
        assert!(took(whole, row, &mut t, &mut out, n, picks));
        read += n;
    }
    (t, out, read)
}

/// The device argmax of `row`: ggml-cuda's `argmax_f32` order among equal maxima.
fn device(row: &[f32]) -> u32 {
    argmax_row(row) as u32
}

const N: usize = 2048;
const STOP: [u32; 2] = [7, 300];

fn picks(exact: bool) -> Picks<'static> {
    Picks { stop: &STOP, exact }
}

#[test]
fn min_tokens_with_a_stop_token_argmax_reads_the_whole_row() {
    // The row's largest value is a stop token: below `min_tokens` the next best is picked.
    let mut row = vec![0.0f32; N];
    row[300] = 9.0;
    row[1500] = 5.0;
    row[40] = 4.0;
    let t = device(&row);
    assert_eq!(t, 300);
    for exact in [false, true] {
        let p = picks(exact);
        assert_eq!(
            Need::of(Logits::Pick, 0, t, N, p),
            Need::Pick {
                len: N,
                masked: true
            }
        );
        let (pick, out, read) = step(Logits::Pick, 0, &row, t, p);
        assert_eq!(pick, 1500);
        assert_eq!(pick, host(&row, true, &STOP));
        assert!(out.is_empty(), "a pick hands back no row");
        assert_eq!(read, N);
    }
    // Above `min_tokens` (a verify row past the masked ones) the stop token is the pick.
    let (pick, _, read) = step(Logits::FirstPick(3, 1), 1, &row, t, picks(false));
    assert_eq!(pick, 300);
    assert_eq!(read, 301);
    assert_eq!(
        step(Logits::FirstPick(3, 1), 0, &row, t, picks(false)).0,
        1500
    );
}

#[test]
fn a_tie_before_the_device_argmax_is_picked_first() {
    // ggml-cuda's argmax takes column 2 over column 1 on a tie; the host takes column 1.
    let mut row = vec![-1.0f32; N];
    row[1] = 3.0;
    row[2] = 3.0;
    let t = device(&row);
    assert_eq!(t, 2);
    let (pick, _, read) = step(Logits::Pick, 0, &row, t, picks(false));
    assert_eq!(pick, 1);
    assert_eq!(pick, host(&row, true, &STOP));
    assert_eq!(read, 3, "only up to the device argmax");
}

#[test]
fn a_tie_after_the_device_argmax_is_not_read() {
    let mut row = vec![-1.0f32; N];
    row[2] = 3.0;
    row[3] = 3.0;
    row[1999] = 3.0;
    let t = device(&row);
    assert_eq!(t, 2);
    let (pick, _, read) = step(Logits::Pick, 0, &row, t, picks(false));
    assert_eq!((pick, read), (2, 3));
    assert_eq!(pick, host(&row, true, &STOP));
}

#[test]
fn a_masked_stop_token_tied_before_the_argmax_is_skipped() {
    let mut row = vec![-1.0f32; N];
    row[7] = 3.0;
    row[8] = 3.0;
    let t = device(&row);
    assert_eq!(t, 8, "rev5 order: column 8 before column 7");
    let (pick, _, read) = step(Logits::Pick, 0, &row, t, picks(false));
    assert_eq!((pick, read), (8, 9));
    assert_eq!(pick, host(&row, true, &STOP));
    // Unmasked, the stop token is the first of the tie.
    let (pick, _, _) = step(Logits::FirstPick(2, 0), 1, &row, t, picks(false));
    assert_eq!(pick, 7);
    assert_eq!(pick, host(&row, false, &STOP));
}

#[test]
fn an_exact_device_argmax_reads_nothing() {
    let mut row = vec![0.0f32; N];
    row[900] = 2.0;
    let t = device(&row);
    assert_eq!(Need::of(Logits::Pick, 0, t, N, picks(true)), Need::Nothing);
    assert_eq!(
        Need::of(Logits::FirstPick(2, 2), 1, t, N, picks(true)),
        Need::Nothing
    );
    let (pick, out, read) = step(Logits::Pick, 0, &row, t, picks(true));
    assert_eq!((pick, read), (900, 0));
    assert!(out.is_empty());
}

#[test]
fn other_kinds_read_what_they_did() {
    let row: Vec<f32> = (0..N).map(|c| c as f32).collect();
    let t = device(&row);
    for kind in [Logits::Greedy, Logits::FirstGreedy(2)] {
        assert_eq!(Need::of(kind, 0, t, N, picks(false)), Need::Nothing);
    }
    for kind in [Logits::Last, Logits::First(2), Logits::All] {
        assert_eq!(Need::of(kind, 0, t, N, picks(false)), Need::Row);
        let (argmax, out, read) = step(kind, 0, &row, t, picks(false));
        assert_eq!((argmax, read), (t, N));
        assert_eq!(out, row, "a sampled row comes back whole");
    }
}

#[test]
fn rows_the_prefix_cannot_decide_are_read_whole() {
    let p = picks(false);
    let mut row = vec![0.0f32; N];
    row[10] = 1.0;
    // No device argmax (every value -inf, NaN or -f32::MAX): -1.
    assert_eq!(
        Need::of(Logits::Pick, 0, u32::MAX, N, p),
        Need::Pick {
            len: N,
            masked: true
        }
    );
    // A NaN first value: the host keeps it.
    let mut nan0 = row.clone();
    nan0[0] = f32::NAN;
    assert_eq!(pick(&nan0[..11], 10, true, N, &STOP), None);
    assert_eq!(
        step(Logits::Pick, 0, &nan0, 10, p).0,
        host(&nan0, true, &STOP)
    );
    // A NaN or -inf at the argmax.
    let mut bad = row.clone();
    bad[10] = f32::NAN;
    assert_eq!(pick(&bad[..11], 10, true, N, &STOP), None);
    bad[10] = f32::NEG_INFINITY;
    assert_eq!(pick(&bad[..11], 10, true, N, &STOP), None);
    // A larger value before the given argmax.
    let mut big = row.clone();
    big[3] = 2.0;
    assert_eq!(pick(&big[..11], 10, true, N, &STOP), None);
    assert_eq!(step(Logits::Pick, 0, &big, 10, p).0, 3);
    // A prefix of the wrong length.
    assert_eq!(pick(&row[..12], 10, true, N, &STOP), None);
    assert_eq!(pick(&row[..11], 10, true, N, &STOP), Some(10));
}

#[test]
fn picks_match_the_host_on_rows_full_of_ties() {
    // Few distinct values, so most rows tie at their maximum, some on a stop token.
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) as u32
    };
    let stop = [7u32, 300, 1025, 2047];
    let mut prefix = 0;
    for case in 0..400 {
        let levels = 2 + case % 6;
        let mut row: Vec<f32> = (0..N).map(|_| (next() as usize % levels) as f32).collect();
        if case % 5 == 0 {
            row[stop[case % stop.len()] as usize] = levels as f32;
        }
        if case % 7 == 0 {
            row[next() as usize % N] = f32::NAN;
        }
        let t = device(&row);
        for exact in [false, true] {
            let p = Picks { stop: &stop, exact };
            for (kind, i, masked) in [
                (Logits::Pick, 0, true),
                (Logits::FirstPick(4, 2), 1, true),
                (Logits::FirstPick(4, 2), 2, false),
            ] {
                if exact && host(&row, masked, &stop) != t && !(masked && stop.contains(&t)) {
                    // An exact device argmax is the host's by definition; ggml-cuda's is not.
                    continue;
                }
                let (pick, out, read) = step(kind, i, &row, t, p);
                assert_eq!(
                    pick,
                    host(&row, masked, &stop),
                    "case {case} {kind:?} row {i}"
                );
                assert!(out.is_empty());
                if !exact {
                    prefix += read;
                }
            }
        }
    }
    assert!(prefix < 400 * 3 * N, "prefixes read less than whole rows");
}

#[test]
fn whole_rows_side_by_side_are_one_read_and_picks_are_not() {
    const V: usize = 248_320;
    // Three whole rows, then two picks' prefixes of rows 3 and 5.
    let parts = [(0, V), (V, V), (2 * V, V), (3 * V, 900), (5 * V, 40_000)];
    let cap = 48 * V;
    assert_eq!(spans(&parts, SPAN_GAP, cap), vec![0..3, 3..4, 4..5]);
    // A buffer of one row: every row is its own read.
    assert_eq!(
        spans(&parts, SPAN_GAP, V),
        vec![0..1, 1..2, 2..3, 3..4, 4..5]
    );
    // A pick's prefix right before the next row's start joins it.
    let near = [(0, V - 1000), (V, V)];
    assert_eq!(spans(&near, SPAN_GAP, cap), vec![0..2]);
    // A part longer than the buffer is read alone.
    assert_eq!(spans(&[(0, 10), (10, 100)], SPAN_GAP, 50), vec![0..1, 1..2]);
    assert!(spans(&[], SPAN_GAP, cap).is_empty());
}
