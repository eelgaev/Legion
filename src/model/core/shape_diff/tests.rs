use super::*;

fn group(n_tok: usize, n_pool: usize, n_new: usize) -> GroupShape {
    GroupShape {
        slot: 0,
        n_seqs: 16,
        n_tok,
        n_kv: 0,
        table: 1280,
        ring: 0,
        logits: vec![n_tok; 16],
        last_only: false,
        verify: n_tok > 1,
        n_pool,
        n_new,
        pick_all: true,
        kv_slots: Vec::new(),
        par: false,
        pair: false,
        head_half: 0,
        img: 0,
        captures: Vec::new(),
    }
}

#[test]
fn diff_names_the_fields_that_differ() {
    let a = [group(9, 384, 36)];
    assert!(diff(&a, &a).is_empty());
    let b = [group(9, 448, 36)];
    assert_eq!(diff(&a, &b), ["g0.n_pool 384->448"]);
    let c = [group(9, 1024, 48)];
    assert_eq!(diff(&a, &c), ["g0.n_pool 384->1024", "g0.n_new 36->48"]);
    // Another group count: said first, the groups both have compared.
    let d = [group(9, 384, 36), group(1, 384, 16)];
    assert_eq!(diff(&a, &d), ["groups 1->2"]);
    // The logits rows: their totals, or each sequence's when only their split moved.
    let mut e = group(2, 384, 36);
    e.logits = vec![2, 2];
    let mut f = e.clone();
    assert_eq!(diff(&[e.clone()], &[f.clone()]), Vec::<String>::new());
    f.logits = vec![1, 3];
    assert_eq!(diff(&[e.clone()], &[f]), ["g0.logits [2, 2]->[1, 3]"]);
    let mut g = e.clone();
    g.logits = vec![2, 3];
    assert_eq!(diff(&[e], &[g]), ["g0.logits 4->5"]);
}

#[test]
fn nearest_is_the_kept_shape_with_the_fewest_differences() {
    let shape = [group(9, 448, 36)];
    let far = vec![group(9, 1024, 48)];
    let near = vec![group(9, 384, 36)];
    let kept = [far.as_slice(), near.as_slice()];
    assert_eq!(
        nearest(kept, &shape),
        Some(vec!["g0.n_pool 384->448".to_string()])
    );
    assert_eq!(nearest(std::iter::empty(), &shape), None);
    assert_eq!(describe(None), "none kept");
}

#[test]
fn describe_shows_the_first_differences() {
    let d: Vec<String> = (0..9).map(|i| format!("f{i}")).collect();
    assert_eq!(describe(Some(d)), "f0, f1, f2, f3, f4, f5, +3 more");
    assert_eq!(describe(Some(vec!["a".into()])), "a");
}

#[test]
fn why_says_new_evicted_or_dropped() {
    assert_eq!(why(true, 256, 256), "new shape");
    assert_eq!(why(false, 256, 256), "evicted at the cap");
    assert_eq!(why(false, 11, 32), "dropped below the cap");
    assert_eq!(why(false, 3, 0), "dropped below the cap");
}

#[test]
fn step_kinds() {
    assert_eq!(step_kind(&[group(1, 64, 16)], false), "decode");
    assert_eq!(step_kind(&[group(9, 64, 16)], true), "verify");
    assert_eq!(step_kind(&[group(512, 64, 16)], false), "prompt");
}
