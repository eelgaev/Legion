use super::*;

fn at(v: &[f32]) -> (usize, usize) {
    (v.as_ptr() as usize, v.len())
}

fn check(v: &[f32]) -> u64 {
    if cfg!(debug_assertions) {
        crate::node::rows_check(v)
    } else {
        0
    }
}

#[test]
fn rows_of_the_last_steps_are_named_others_sent() {
    let layer0: Vec<f32> = (0..40).map(|i| i as f32).collect();
    let layer1: Vec<f32> = (0..12).map(|i| -(i as f32)).collect();
    let before: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let pending = vec![7.5f32; 4];
    let last = [at(&layer0), at(&layer1)];
    let prev = [at(&before)];
    let mut h = Vec::new();
    let parts = rows_parts(
        &[&last, &prev],
        &[
            &pending,
            &layer0[8..36],
            &[],
            &before[4..12],
            &layer1[4..],
            &layer0[..],
        ],
        &mut h,
    );
    let step = |back, layer, start, v: &[f32]| RowsPart::Step {
        back,
        layer,
        start,
        len: v.len() as u64,
        check: check(v),
    };
    assert_eq!(
        parts,
        vec![
            RowsPart::Sent { len: 4 },
            step(0, 0, 8, &layer0[8..36]),
            RowsPart::Sent { len: 0 },
            step(1, 0, 4, &before[4..12]),
            step(0, 1, 4, &layer1[4..]),
            step(0, 0, 0, &layer0),
        ]
    );
    assert_eq!(h, pending);
}

#[test]
fn without_a_step_every_part_is_sent() {
    let rows = [1.0f32, 2.0, 3.0];
    let mut h = vec![9.0];
    let parts = rows_parts(&[&[], &[]], &[&rows[1..], &rows[..1]], &mut h);
    assert_eq!(
        parts,
        vec![RowsPart::Sent { len: 2 }, RowsPart::Sent { len: 1 }]
    );
    assert_eq!(h, vec![9.0, 2.0, 3.0, 1.0]);
}
