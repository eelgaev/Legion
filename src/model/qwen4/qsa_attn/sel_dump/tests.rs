use super::*;

/// Selection rows of `row` I32s: each list's count, its positions, -1 after.
fn rows_of(lists: &[Vec<i32>], row: usize) -> Vec<i32> {
    let mut v = Vec::new();
    for l in lists {
        v.push(l.len() as i32);
        v.extend(l);
        v.resize(v.len() + row - 1 - l.len(), -1);
    }
    v
}

#[test]
fn same_lists_share_every_row() {
    let l: Vec<i32> = (0..40).collect();
    let rows = rows_of(&[l.clone(), l], 50);
    let o = overlap(&rows, 50, 2, 2);
    assert_eq!((o.groups, o.sum, o.union, o.all), (1, 80, 40, 40));
    assert_eq!(o.lock, 40);
    assert_eq!(o.drift, [40, 0, 0]);
    // Both chunks end together: two steps, each row loaded once.
    assert_eq!(pair_window(&rows, 50, 2), (40, 2));
}

#[test]
fn window_follows_positions() {
    // The second token's first chunk is the first token's third: lock step loads it twice (it
    // is gone by then), the window once.
    let a: Vec<i32> = (0..96).collect();
    let b: Vec<i32> = (64..160).collect();
    let rows = rows_of(&[a, b], 100);
    let o = overlap(&rows, 100, 2, 2);
    assert_eq!((o.sum, o.union, o.lock), (192, 160, 192));
    assert_eq!(o.drift, [0, 0, 32]);
    assert_eq!(pair_window(&rows, 100, 2), (160, 5));
}

#[test]
fn groups_stay_in_a_sequence() {
    // Three tokens a sequence: the third has no neighbour (its rows on their own).
    let l: Vec<i32> = (0..8).collect();
    let rows = rows_of(&vec![l; 6], 10);
    let o = overlap(&rows, 10, 3, 2);
    assert_eq!((o.groups, o.sum, o.union), (2, 32, 16));
    assert_eq!(pair_window(&rows, 10, 3), (32, 4));
}
