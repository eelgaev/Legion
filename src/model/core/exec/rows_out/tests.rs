use super::{SPARE_KEPT, SPARE_MAX_BYTES, Spare, copy_out, copy_out_tail, take_layer};

fn floats(seed: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| (seed * 1_000_000 + i) as f32).collect()
}

#[test]
fn copy_out_copies_every_float_in_order() {
    for n in [0, 1, 3, 4, 5, 1023, 1 << 20] {
        let src = floats(7, n);
        let got = copy_out(&mut Spare::default(), src.as_ptr() as usize, n);
        assert_eq!(got, src, "n = {n}");
    }
}

/// The next chunk's rows go into the vector handed back, all of them overwritten.
#[test]
fn copy_out_reuses_a_spare_with_room() {
    let mut spare = Spare::default();
    let old = floats(1, 1000);
    let at = old.as_ptr();
    spare.keep([old], 0);
    let src = floats(2, 900);
    let got = copy_out(&mut spare, src.as_ptr() as usize, src.len());
    assert_eq!(got.as_ptr(), at, "the spare's memory");
    assert_eq!(got, src);
    assert_eq!(spare.capacity(), 0, "taken");
}

#[test]
fn copy_out_leaves_a_spare_too_small() {
    let mut spare = Spare::default();
    spare.keep([floats(1, 100)], 0);
    let src = floats(2, 101);
    assert_eq!(copy_out(&mut spare, src.as_ptr() as usize, 101), src);
    assert_eq!(spare.capacity(), 100, "kept for later");
}

#[test]
fn keep_takes_the_vectors_worth_keeping() {
    let mut spare = Spare::default();
    // Under the minimum: dropped.
    spare.keep([floats(1, 10)], 400);
    assert_eq!(spare.capacity(), 0);
    assert_eq!(spare.kept(), 0);
    spare.keep([floats(1, 200), floats(2, 300), floats(3, 250)], 400);
    assert_eq!(spare.capacity(), 300);
    assert_eq!(spare.kept(), 3);
    spare.keep([Vec::with_capacity(SPARE_MAX_BYTES / 4 + 1)], 400);
    assert_eq!(spare.capacity(), 300, "a whole prompt's output is not kept");
    assert_eq!(spare.kept(), 3);
}

/// A drafter reading several layers: every layer's copy of the next chunk goes into a vector
/// handed back, the smallest one with room each time.
#[test]
fn several_copies_reuse_several_spares() {
    let mut spare = Spare::default();
    let old: Vec<Vec<f32>> = (0..5).map(|k| floats(k, 1000 + k)).collect();
    let at: Vec<*const f32> = old.iter().map(|v| v.as_ptr()).collect();
    spare.keep(old, 0);
    let mut got = Vec::new();
    for k in 0..5 {
        let src = floats(10 + k, 1000);
        let v = copy_out(&mut spare, src.as_ptr() as usize, src.len());
        assert_eq!(v, src);
        got.push(v.as_ptr());
    }
    assert_eq!(spare.kept(), 0, "all taken");
    // Smallest first: the vectors in the order they were made.
    assert_eq!(got, at);
    // A copy bigger than all of them makes a new vector and leaves them.
    spare.keep([floats(1, 100), floats(2, 200)], 0);
    let src = floats(3, 201);
    assert_eq!(copy_out(&mut spare, src.as_ptr() as usize, src.len()), src);
    assert_eq!(spare.kept(), 2);
    assert_eq!(spare.capacity(), 200);
}

#[test]
fn keep_holds_at_most_spare_kept_vectors_biggest_first() {
    let mut spare = Spare::default();
    spare.keep((0..SPARE_KEPT + 3).map(|k| floats(k, 100 + k)), 0);
    assert_eq!(spare.kept(), SPARE_KEPT);
    assert_eq!(spare.capacity(), 100 + SPARE_KEPT + 2);
    // The smallest kept is the one the smallest copy takes.
    let src = floats(0, 1);
    let v = copy_out(&mut spare, src.as_ptr() as usize, 1);
    assert_eq!(v.capacity(), 103);
}

#[test]
fn keep_holds_at_most_spare_max_bytes_together() {
    let mut spare = Spare::default();
    let half = SPARE_MAX_BYTES / 4 / 2;
    spare.keep(
        [
            Vec::<f32>::with_capacity(half),
            Vec::with_capacity(half),
            Vec::with_capacity(half),
        ],
        0,
    );
    assert_eq!(spare.kept(), 2);
}

#[test]
fn one_micro_chunk_moves_its_vector() {
    let rows = floats(1, 4096);
    let at = rows.as_ptr();
    let mut hidden = vec![vec![(5, Vec::new()), (47, rows.clone())]];
    hidden[0][1].1 = rows;
    let got = take_layer(&mut hidden, 47);
    assert_eq!(got, floats(1, 4096));
    assert_eq!(got.as_ptr(), at, "moved, not copied");
    assert!(hidden[0][1].1.is_empty());
}

#[test]
fn only_one_micro_chunk_with_rows_moves_it() {
    let rows = floats(2, 300);
    let at = rows.as_ptr();
    let mut hidden = vec![
        vec![(47, Vec::new())],
        vec![(47, rows)],
        vec![(47, Vec::new())],
    ];
    let got = take_layer(&mut hidden, 47);
    assert_eq!(got.as_ptr(), at);
    assert_eq!(got, floats(2, 300));
}

#[test]
fn several_micro_chunks_are_concatenated_in_order() {
    let mut hidden: Vec<Vec<(usize, Vec<f32>)>> = (0..3)
        .map(|k| vec![(3, floats(9, 1)), (47, floats(k, 10 + k))])
        .collect();
    let want: Vec<f32> = (0..3).flat_map(|k| floats(k, 10 + k)).collect();
    assert_eq!(take_layer(&mut hidden, 47), want);
    assert!(hidden.iter().all(|h| h[1].1.is_empty()));
    assert!(
        hidden.iter().all(|h| h[0].1 == floats(9, 1)),
        "other layers untouched"
    );
}

/// Only the floats from `from` on are copied; the vector keeps its full length (#1133).
#[test]
fn copy_out_tail_copies_the_last_floats() {
    for (n, from) in [(0, 0), (5, 0), (5, 5), (1023, 1000), (1 << 20, 3 << 18)] {
        let src = floats(3, n - from);
        let got = copy_out_tail(&mut Spare::default(), src.as_ptr() as usize, n, from);
        assert_eq!(got.len(), n, "n = {n}");
        assert!(
            got[..from].iter().all(|&x| x == 0.0),
            "a new vector's head is zeros"
        );
        assert_eq!(got[from..], src[..], "n = {n}, from = {from}");
    }
}

/// A handed-back vector keeps its old head; longer or shorter ones are cut or padded to `n`.
#[test]
fn copy_out_tail_reuses_a_spare() {
    for old_len in [1000, 800] {
        let mut spare = Spare::default();
        let old = floats(1, old_len);
        let mut big = Vec::with_capacity(1000);
        big.extend_from_slice(&old);
        let at = big.as_ptr();
        spare.keep([big], 0);
        let src = floats(2, 100);
        let got = copy_out_tail(&mut spare, src.as_ptr() as usize, 900, 800);
        assert_eq!(got.as_ptr(), at, "the spare's memory");
        assert_eq!(got.len(), 900);
        assert_eq!(got[..800], old[..800]);
        assert_eq!(got[800..], src[..]);
        assert_eq!(spare.kept(), 0, "taken");
    }
}

#[test]
fn no_rows_give_an_empty_vector() {
    let mut hidden = vec![vec![(47, Vec::new())], vec![(47, Vec::new())]];
    assert!(take_layer(&mut hidden, 47).is_empty());
}
