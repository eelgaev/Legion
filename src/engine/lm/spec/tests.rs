use super::*;

#[test]
fn caps() {
    assert_eq!(draft_cap(4, 100, 10, 8192, 50), 4);
    // Two tokens left: one draft plus the bonus.
    assert_eq!(draft_cap(4, 12, 10, 8192, 50), 1);
    assert_eq!(draft_cap(4, 11, 10, 8192, 50), 0);
    assert_eq!(draft_cap(4, 10, 10, 8192, 50), 0);
    // Context: positions n_past..=n_past + k must stay below max_ctx.
    assert_eq!(draft_cap(4, 100, 10, 53, 50), 2);
    assert_eq!(draft_cap(4, 100, 10, 51, 50), 0);
}

#[test]
fn kinds_pick_their_drafters() {
    let info = TargetInfo {
        n_layer: 4,
        n_embd: 64,
        n_vocab: 259,
        model: "tiny.gguf".into(),
        draft: None,
        gpu: false,
        threads: 1,
        ctx_size: 512,
        max_seqs: 4,
        use_ref: true,
    };
    let cfg = |kind| SpecConfig {
        kind,
        k: 4,
        ..Default::default()
    };
    assert!(matches!(
        Drafting::new(&cfg(SpecKind::None), &info, 0),
        Ok(Drafting::Off)
    ));
    assert!(matches!(
        Drafting::new(&cfg(SpecKind::Ngram), &info, 0),
        Ok(Drafting::Tokens(_))
    ));
    assert!(matches!(
        Drafting::new(&cfg(SpecKind::Mtp), &info, 0),
        Ok(Drafting::Mtp(_))
    ));
}

/// A greedy target over `rows` (device argmax, logits), stop token 9 masked on the first row.
fn greedy_target<'s>(
    sampler: &'s mut Sampler,
    rows: Vec<(u32, Vec<f32>)>,
    device_picks: bool,
) -> SeqTarget<'s> {
    SeqTarget {
        sampler,
        rows,
        keep_raw: false,
        device_picks,
        raw: Vec::new(),
        mask_rows: 1,
        stop_tokens: &[9],
    }
}

fn greedy() -> Sampler {
    Sampler::new(&crate::engine::SamplingParams {
        temperature: 0.0,
        ..Default::default()
    })
}

#[test]
fn device_picks_skip_the_scan_unless_a_masked_stop_token_won() {
    // The rows' own argmax is token 3; the device argmax given differs, so the result shows
    // which of the two was used.
    let row = |hot: usize| {
        let mut r = vec![0.0f32; 16];
        r[hot] = 1.0;
        r[9] = 0.5;
        r
    };
    let mut s = greedy();
    let mut t = greedy_target(&mut s, vec![(5, row(3)), (7, row(3))], true);
    assert_eq!(t.dist(0), Dist::Greedy(5));
    assert_eq!(t.dist(1), Dist::Greedy(7));
    // Device argmax 9 is a stop token below min_tokens (row 0): the row is masked and scanned.
    let mut s = greedy();
    let mut t = greedy_target(&mut s, vec![(9, row(3)), (9, row(3))], true);
    assert_eq!(t.dist(0), Dist::Greedy(3));
    // Past min_tokens (row 1) a stop token is the pick.
    assert_eq!(t.dist(1), Dist::Greedy(9));
    // Without device picks the rows are scanned as before.
    let mut s = greedy();
    let mut t = greedy_target(&mut s, vec![(5, row(3)), (7, row(3))], false);
    assert_eq!(t.dist(0), Dist::Greedy(3));
    assert_eq!(t.dist(1), Dist::Greedy(3));
}

#[test]
fn device_picks_equal_the_masked_host_argmax() {
    // Rows with ties at the top: the first of the largest (the device argmax of a model with
    // `argmax_first_of_ties`) is the masked row's greedy pick whenever it is not masked.
    let mut rng = crate::engine::sampling::Rng::new(11);
    for _ in 0..2000 {
        let row: Vec<f32> = (0..64).map(|_| (rng.next_u64() % 6) as f32).collect();
        let device = crate::engine::sampling::argmax(&row) as u32;
        let stop = (rng.next_u64() % 64) as u32;
        let stops = [stop];
        let mut s = greedy();
        let mut host = SeqTarget {
            stop_tokens: &stops,
            ..greedy_target(&mut s, vec![(device, row.clone())], false)
        };
        let want = host.dist(0);
        let mut s = greedy();
        let mut fast = SeqTarget {
            stop_tokens: &stops,
            ..greedy_target(&mut s, vec![(device, row)], true)
        };
        assert_eq!(fast.dist(0), want);
    }
}

#[test]
fn device_take_in_reads_the_tail_and_its_pending_row() {
    // The pieces end TAKE_IN_TAIL rows before the end; the last piece's last row is pending.
    let ran = 4096;
    let mut skip = 0;
    while ran - skip > TAKE_IN_TAIL {
        skip += TAKE_IN_PIECE.min(ran - skip - TAKE_IN_TAIL);
    }
    assert_eq!(take_in_host_from(ran), skip - 1);
    assert_eq!(take_in_host_from(TAKE_IN_TAIL + 2), 1);
    assert_eq!(take_in_host_from(3), 0);
}
