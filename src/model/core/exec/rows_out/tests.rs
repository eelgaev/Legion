use super::{SPARE_MAX_BYTES, Spare, copy_out, take_layer};

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
fn keep_takes_the_biggest_worth_keeping() {
    let mut spare = Spare::default();
    // Under the minimum: dropped.
    spare.keep([floats(1, 10)], 400);
    assert_eq!(spare.capacity(), 0);
    spare.keep([floats(1, 200), floats(2, 300), floats(3, 250)], 400);
    assert_eq!(spare.capacity(), 300);
    // Not bigger than the one kept: dropped.
    spare.keep([floats(4, 300)], 400);
    assert_eq!(spare.capacity(), 300);
    spare.keep([Vec::with_capacity(SPARE_MAX_BYTES / 4 + 1)], 400);
    assert_eq!(spare.capacity(), 300, "a whole prompt's output is not kept");
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

#[test]
fn no_rows_give_an_empty_vector() {
    let mut hidden = vec![vec![(47, Vec::new())], vec![(47, Vec::new())]];
    assert!(take_layer(&mut hidden, 47).is_empty());
}
