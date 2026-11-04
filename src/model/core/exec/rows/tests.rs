//! Deferred row copies into the step's output vector (#435).

use super::*;

/// Micro-chunk `k`'s rows of `layer`: `tokens * width` distinct floats.
fn rows(layer: usize, k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (layer * 1_000_000 + k * 10_000 + i) as f32)
        .collect()
}

/// Runs a step's copies the way `Executor::step` does: the rows of the micro-chunks in `deferred`
/// go through `start` (from a buffer kept alive until joined), the others straight to `hidden`.
fn run(tokens: &[usize], width: &[usize], deferred: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let layer = 47;
    let mut d = DeferredRows::new(tokens.to_vec());
    let mut hidden: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); tokens.len()];
    let mut sources = Vec::new();
    let mut want = Vec::new();
    for k in 0..tokens.len() {
        let r = rows(layer, k, tokens[k] * width[k]);
        want.extend_from_slice(&r);
        if deferred[k] {
            d.start(k, layer, k % 2, r.as_ptr() as usize, r.len() * 4);
            sources.push(r);
            hidden[k].push((layer, Vec::new()));
        } else {
            hidden[k].push((layer, r));
        }
        if k % 2 == 1 {
            d.join(Some(0), &mut hidden);
        }
    }
    let got = d
        .assemble(layer, &mut hidden)
        .unwrap_or_else(|| hidden.iter().flat_map(|h| h[0].1.clone()).collect());
    drop(sources);
    (got, want)
}

#[test]
fn every_micro_chunk_in_place() {
    let (got, want) = run(&[4, 4, 4, 2], &[3, 3, 3, 3], &[true; 4]);
    assert_eq!(got, want);
    assert_eq!(got.capacity(), got.len());
}

#[test]
fn a_rows_list_that_was_not_deferred_is_copied_to_its_place() {
    let (got, want) = run(&[4, 4, 1], &[3, 3, 3], &[true, false, true]);
    assert_eq!(got, want);
    let (got, want) = run(&[4, 4, 1], &[3, 3, 3], &[false, true, true]);
    assert_eq!(got, want);
}

#[test]
fn rows_of_another_width_fall_back() {
    // The second micro-chunk has rows of another width: it gets its own vector, and the output
    // is still every micro-chunk's rows one after another.
    let (got, want) = run(&[4, 4, 2], &[3, 5, 3], &[true, true, true]);
    assert_eq!(got, want);
    // Not deferred and the wrong size.
    let (got, want) = run(&[4, 4, 2], &[3, 3, 2], &[true, true, false]);
    assert_eq!(got, want);
}

#[test]
fn nothing_deferred_or_another_layer() {
    let (got, want) = run(&[4, 4], &[3, 3], &[false, false]);
    assert_eq!(got, want);
    let mut d = DeferredRows::new(vec![2]);
    let src = rows(1, 0, 6);
    d.start(0, 1, 0, src.as_ptr() as usize, src.len() * 4);
    let mut hidden = vec![vec![(1, Vec::new()), (2, rows(2, 0, 6))]];
    assert!(d.assemble(2, &mut hidden).is_none());
    assert_eq!(d.assemble(1, &mut hidden), Some(src));
}

#[test]
fn gather_writes_the_parts_in_order() {
    // Small (one thread) and big (four threads, chunk edges inside parts), empty parts too.
    for scale in [1usize, 400_000] {
        let a: Vec<f32> = (0..3 * scale).map(|i| i as f32).collect();
        let b: Vec<f32> = Vec::new();
        let c: Vec<f32> = (0..5 * scale + 1).map(|i| -(i as f32)).collect();
        let parts = [&a[..], &b[..], &c[..]];
        let want: Vec<f32> = parts.concat();
        let mut dst = vec![0f32; want.len()];
        gather(&parts, &mut dst);
        assert_eq!(dst, want);
    }
    let mut empty: Vec<f32> = Vec::new();
    gather(&[], &mut empty);
}
