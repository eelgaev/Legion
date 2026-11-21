use super::*;

#[test]
fn only_take_in_rounds_that_read_rows_wait() {
    // A prompt chunk that takes in the window's rows from its 2,032nd on.
    assert!(defers(true, false, [0], [(2032, 4096)]));
    // Switched off, or rows left on the device: at once.
    assert!(!defers(false, false, [0], [(2032, 4096)]));
    assert!(!defers(true, true, [0], [(2032, 4096)]));
    // A sequence drafts in the round: at once (its drafts need the round's sums).
    assert!(!defers(true, false, [0, 3], [(2032, 4096), (0, 1)]));
    // No request takes in a row (the ring keeps none of them): at once, it costs nothing.
    assert!(!defers(true, false, [0, 0], [(4096, 4096), (5000, 2048)]));
    // One of several does.
    assert!(defers(true, false, [0, 0], [(4096, 4096), (0, 9)]));
    // An empty round: nothing to wait with.
    assert!(!defers(true, false, [], []));
}

#[test]
fn rows_of_slices_each_layer() {
    // Two layers of a 3-position step, 2 floats a row.
    let hidden = vec![
        vec![0., 1., 2., 3., 4., 5.],
        vec![10., 11., 12., 13., 14., 15.],
    ];
    assert_eq!(
        rows_of(&hidden, 3, 1, 2),
        vec![&[2., 3., 4., 5.][..], &[12., 13., 14., 15.]]
    );
    assert_eq!(rows_of(&hidden, 3, 0, 1), vec![&[0., 1.][..], &[10., 11.]]);
}

/// Records each round's requests.
#[derive(Default)]
struct Recorder {
    rounds: Vec<Vec<(SeqId, Vec<u32>, usize, usize, Vec<Vec<f32>>, usize)>>,
}

impl ModelDrafter for Recorder {
    fn hidden_layers(&self) -> Vec<usize> {
        vec![0, 1]
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        self.rounds.push(
            batch
                .iter()
                .map(|r| {
                    let rows = r.hidden.iter().map(|h| h.to_vec()).collect();
                    (r.seq, r.tokens.to_vec(), r.start, r.ran, rows, r.max)
                })
                .collect(),
        );
        Ok(())
    }
}

fn req(seq: SeqId, tokens: &[u32], start: usize, ran: usize, off: usize) -> Req {
    Req {
        seq,
        tokens: tokens.to_vec(),
        start,
        ran,
        off,
    }
}

#[test]
fn a_round_runs_its_requests_as_left() {
    // Two sequences' chunks in one 3-position step, 1 float a row.
    let hidden = Arc::new(vec![vec![0., 1., 2.], vec![10., 11., 12.]]);
    let round = Round::new(
        hidden,
        3,
        vec![req(7, &[1, 2, 3], 1, 2, 0), req(9, &[4], 0, 1, 2)],
    );
    let mut d = Recorder::default();
    let (result, rows) = round.run(&mut d);
    assert_eq!(result, Ok(()));
    assert_eq!(rows, Some(vec![vec![0., 1., 2.], vec![10., 11., 12.]]));
    assert_eq!(
        d.rounds,
        vec![vec![
            (
                7,
                vec![1, 2, 3],
                1,
                2,
                vec![vec![0., 1.], vec![10., 11.]],
                0
            ),
            (9, vec![4], 0, 1, vec![vec![2.], vec![12.]], 0),
        ]]
    );
}

#[test]
fn rows_still_held_elsewhere_stay() {
    let hidden = Arc::new(vec![vec![0., 1.]]);
    let kept = hidden.clone();
    let round = Round::new(hidden, 2, vec![req(1, &[5, 6], 0, 2, 0)]);
    let (result, rows) = round.run(&mut Recorder::default());
    assert_eq!(result, Ok(()));
    assert_eq!(rows, None);
    assert_eq!(Arc::strong_count(&kept), 1);
}

#[test]
fn a_released_sequence_leaves_the_round() {
    let hidden = Arc::new(vec![vec![0., 1.]]);
    let mut round = Round::new(
        hidden,
        2,
        vec![req(1, &[5], 0, 1, 0), req(2, &[6], 0, 1, 1)],
    );
    // Another sequence's release leaves it as it was.
    assert!(round.release(3));
    assert!(round.release(1));
    let mut d = Recorder::default();
    let _ = round.run(&mut d);
    assert_eq!(d.rounds, vec![vec![(2, vec![6], 0, 1, vec![vec![1.]], 0)]]);
    // Its last sequence gone, nothing is left.
    let mut round = Round::new(Arc::new(vec![vec![0.]]), 1, vec![req(4, &[7], 0, 1, 0)]);
    assert!(!round.release(4));
}
